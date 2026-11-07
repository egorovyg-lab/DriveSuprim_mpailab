"""Scan ALL numerical deviations; writes audit JSON only, never accepted features."""
import json
import os
from pathlib import Path
import numpy as np
from navsim.agents.drivemem_determinism import PROFILE, prepare_environment, configure, runtime, repeat_parity
from navsim.agents.drivemem_numeric_contract import CONTRACT, audit, parity, aggregate
from navsim.agents.drivemem_frozen_pool import apply_pool, describe_pool
from navsim.planning.script.run_drivemem_decoder_export import parity as legacy_parity
from navsim.planning.script.run_drivemem_decoder_numeric_export import wanted_records, key
from navsim.planning.script.run_drivemem_expanded_utility import load_inputs, read, write, sha, load_sample, REPO


def assess(old, unhooked, fresh, vocab, active_vocab, alignment):
    errors = {}
    for name, check in (("repeat", lambda: repeat_parity(unhooked, fresh)),
                        ("contract", lambda: parity(old, fresh, vocab)),
                        ("legacy", lambda: legacy_parity(old, fresh, vocab))):
        try:
            check()
        except (RuntimeError, ValueError, KeyError) as exc:
            errors[name] = str(exc)
    try:
        measured = audit(old, fresh)
    except (RuntimeError, ValueError, KeyError) as exc:
        measured = None
        errors["arrays"] = str(exc)
    return dict(errors=errors, repeat_bitwise_equal="repeat" not in errors,
        contract_passed="contract" not in errors,
        legacy_passed="legacy" not in errors,
        baseline_equal=bool(np.array_equal(old["selected_local"], fresh["selected_local"])),
        vocabulary_equal=bool(np.array_equal(vocab, active_vocab)),
        head_alignment=alignment, numeric_audit=measured)


def summarize(rows, total):
    valid = {str(i): dict(numeric_audit=r["numeric_audit"]) for i, r in enumerate(rows) if r["numeric_audit"] is not None}
    def accepted(r):
        return r["contract_passed"] and r["repeat_bitwise_equal"] and r["vocabulary_equal"] and all(r["head_alignment"].values())
    return dict(status="complete" if len(rows) == total else "partial", required_scenes=total,
        audited_scenes=len(rows), contract_and_invariants_passed=sum(accepted(r) for r in rows),
        repeat_failures=sum(not r["repeat_bitwise_equal"] for r in rows),
        baseline_changes=sum(not r["baseline_equal"] for r in rows),
        top16_changes=sum(r["numeric_audit"] is not None and not r["numeric_audit"]["top16_set_equal"] for r in rows),
        vocabulary_changes=sum(not r["vocabulary_equal"] for r in rows),
        alignment_failures=sum(not all(r["head_alignment"].values()) for r in rows),
        numeric_audit=aggregate(valid),
        violations=[dict(token=r["token"], role=r["role"], errors=r["errors"]) for r in rows if not accepted(r)],
        note="Completion means all scenes audited, not feature acceptance. Failures kept, no retries, no exported features, fitting or scorer calls.")


def main(cfg):
    prepare_environment()
    from navsim.planning.script.run_drivemem_memory import verify_identity
    source = Path(cfg.drivemem.root).resolve()
    verify_identity(cfg, source)
    base = Path(os.environ["NAVSIM_EXP_ROOT"]) / "drivemem_memory"
    prior = Path(os.environ.get("DRIVEMEM_PRIOR_UTILITY_ROOT", str(base / "experiments/pair_utility_v1"))).resolve()
    addition = Path(os.environ.get("DRIVEMEM_NEW_MF_ROOT", str(base / "mf_new4_shard174_v1"))).resolve()
    out = Path(os.environ.get("DRIVEMEM_DECODER_AUDIT_ROOT", str(base / "decoder_numeric_audit_v1"))).resolve()
    _, _, _, vocab, actual_source, validation, _, bindings, _, _ = load_inputs(prior, addition)
    if source != actual_source:
        raise ValueError("Original budget source required")
    for protected in (source, addition, validation, prior) + tuple(base / n for n in (
            "decoder_features_v1", "decoder_features_replay_v1", "decoder_features_deterministic_v1", "decoder_features_numeric_v1")):
        if out == protected or out in protected.parents or protected in out.parents:
            raise ValueError("Separate audit-only directory required")
    original_identity = read(source / "identity.json")
    for root in (addition, validation):
        other = read(root / "identity.json")
        if {k: v for k, v in original_identity.items() if k != "settings"} != {k: v for k, v in other.items() if k != "settings"}:
            raise ValueError("Teacher/scorer identities differ")
    records = wanted_records(source, addition, validation)
    if len(records) != 152:
        raise ValueError("Exactly 152 permitted M/F/development records required")
    names = ("navsim/agents/drivemem_determinism.py", "navsim/agents/drivemem_numeric_contract.py",
             "navsim/agents/drivemem_frozen_pool.py", "navsim/planning/script/run_drivemem_decoder_export.py",
             "navsim/planning/script/run_drivemem_decoder_numeric_export.py")
    identity = dict(inputs=bindings, runner=sha(Path(__file__)), profile=PROFILE, contract=CONTRACT,
        helpers={name: sha(REPO / name) for name in names},
        records=[dict(source=str(root), **r) for root, r in records])
    marker = out / "identity.json"
    if marker.exists() and read(marker) != identity:
        raise RuntimeError("Audit protocol changed; use a separate audit root")
    if out.exists() and not marker.exists() and any(out.iterdir()):
        raise RuntimeError("Nonempty directory without audit identity")
    write(marker, identity)
    index_path = out / "audit_index.json"
    index = read(index_path) if index_path.exists() else {}
    expected = {key(root, r["token"]) for root, r in records}
    if set(index) - expected:
        raise RuntimeError("Unexpected audit records")
    def row_path(k):
        return out / "scenes" / (k.replace(":", "_") + ".json")
    rows = []
    for k, digest in index.items():
        if sha(row_path(k)) != digest:
            raise RuntimeError("Audit record changed")
        row = read(row_path(k))
        if not (out / "runtime.json").exists() or row["runtime_sha256"] != sha(out / "runtime.json"):
            raise RuntimeError("Audit runtime binding changed")
        rows.append(row)
    todo = [(root, r) for root, r in records if key(root, r["token"]) not in index]
    if not todo:
        report = summarize(rows, len(records))
        write(out / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return
    import torch
    configure(torch)
    from hydra.utils import instantiate
    from torch.utils.data import DataLoader
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.training.dataset_ssl import DatasetSSL
    from navsim.planning.script.run_drivemem_oracle_pilot import on_device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; existing audit rows preserved")
    actual_runtime = runtime(torch)
    runtime_path = out / "runtime.json"
    if runtime_path.exists() and read(runtime_path) != actual_runtime:
        raise RuntimeError("Audit runtime changed")
    write(runtime_path, actual_runtime)
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()
    agent.to("cuda")
    if cfg.agent.config.inference.model != "teacher" or cfg.agent.config.refinement.num_refinement_stage != 1:
        raise ValueError("Teacher/one refinement stage required")
    teacher = agent.model.teacher["model"]
    decoder = teacher._trajectory_offset_head.transformer_blocks[-1]
    capture = {}
    replay = {}
    def coarse_hook(module, inputs):
        replay["coarse"] = inputs[0].detach()
    def pool_hook(module, inputs, output):
        natural = output["refinement"][0]["indices_absolute"].detach().cpu().numpy()[0].copy()
        replay["natural"].append(describe_pool(replay["ids"], natural))
        return apply_pool(output, replay.pop("coarse"), replay["ids"])
    coarse_handle = teacher._trajectory_head.heads["no_at_fault_collisions"].register_forward_pre_hook(coarse_hook)
    pool_handle = teacher._trajectory_head.register_forward_hook(pool_hook)
    def hook(module, inputs, output):
        if not isinstance(output, list) or not output:
            raise RuntimeError("Unexpected decoder output")
        capture["tensor"] = output[-1].detach().clone()
        capture["calls"] = capture.get("calls", 0) + 1
    def snapshot(pred):
        stage = pred["refinement"][-1]
        arrays = dict(candidate_ids=stage["indices_absolute"].cpu().numpy()[0].copy(),
                      selected_local=np.array(int(stage["final_selected_local_indices"].cpu().numpy()[0])),
                      selection_scores=stage["final_selection_scores"].cpu().numpy()[0].copy(),
                      final_traj=pred["final_traj"].cpu().numpy()[0].copy())
        arrays.update({"head_" + k: v.cpu().numpy()[0].copy() for k, v in stage["layer_results"][-1].items()})
        return arrays
    try:
        for root in (source, addition, validation):
            for log in sorted({r["log_name"] for rt, r in todo if rt == root}):
                if sha(Path(cfg.navsim_log_path) / (log + ".pkl")) != read(root / "splits.json")["log_sha256"][log]:
                    raise RuntimeError("Source log changed")
                selected = [r for rt, r in todo if rt == root and r["log_name"] == log]
                scene_filter = instantiate(cfg.train_test_split.scene_filter)
                scene_filter.log_names, scene_filter.tokens = [log], [r["token"] for r in selected]
                loader = SceneLoader(Path(cfg.navsim_log_path), Path(cfg.original_sensor_path), scene_filter, sensor_config=agent.get_sensor_config())
                if set(loader.tokens) != {r["token"] for r in selected}:
                    raise RuntimeError("Decoder loader tokens differ")
                dataset = DatasetSSL(loader, agent.get_feature_builders(), agent.get_target_builders(), cfg=cfg.agent.config, cache_path=None, append_token_to_batch=True)
                source_index = read(root / "export_index.json")
                by_token = {r["token"]: r for r in selected}
                for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
                    batch = on_device(batch, "cuda")
                    token = str(batch[2][0])
                    old = load_sample(root, by_token[token], source_index)
                    replay["ids"], replay["natural"] = old["candidate_ids"], []
                    with torch.no_grad():
                        agent._config.inference.export_candidates = True
                        first, _ = agent.forward(batch)
                        unhooked = snapshot(first)
                        del first
                        capture.clear()
                        handle = decoder.register_forward_hook(hook)
                        try:
                            pred, _ = agent.forward(batch)
                        finally:
                            handle.remove()
                        fresh = snapshot(pred)
                        if len(replay["natural"]) != 2:
                            raise RuntimeError("Expected one frozen-pool replay per forward")
                        if capture.get("calls") != 1:
                            raise RuntimeError("Unexpected decoder hook call count")
                        active_vocab = pred["trajectory_vocab"].detach().cpu().numpy()
                        hidden = capture["tensor"]
                        alignment = {}
                        alignment["latent_finite"] = bool(torch.isfinite(hidden).all().item())
                        for name, head in teacher._trajectory_offset_head.multi_stage_heads[-1].items():
                            check = head(hidden).squeeze(-1).cpu().numpy()[0]
                            alignment[name] = bool(np.allclose(check, fresh["head_" + name], atol=1e-5, rtol=1e-6))
                        result = assess(old, unhooked, fresh, vocab, active_vocab, alignment)
                    result.update(token=token, role=by_token[token]["role"], source=str(root),
                                  runtime_sha256=sha(runtime_path), natural_pool_passes=replay["natural"])
                    k = key(root, token)
                    write(row_path(k), result)
                    index[k] = sha(row_path(k))
                    write(index_path, index)
                    rows.append(result)
                    print(f"NUMERIC AUDIT {len(rows)}/152 {token} errors={result['errors']}", flush=True)
                    write(out / "report.json", summarize(rows, len(records)))
                    del pred, fresh, unhooked, hidden, batch
                    capture.clear()
        print(json.dumps(summarize(rows, len(records)), indent=2), flush=True)
    finally:
        coarse_handle.remove()
        pool_handle.remove()


if __name__ == "__main__":
    import hydra
    hydra.main(config_path="config/pdm_scoring", config_name="default_drivemem_memory", version_base=None)(main)()
