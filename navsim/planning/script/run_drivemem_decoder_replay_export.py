"""Conditional decoder export on the frozen candidate pool, with final parity."""
import json
import os
import time
from pathlib import Path

import numpy as np

from navsim.agents.drivemem_frozen_pool import apply_pool, describe_pool
from navsim.planning.script.run_drivemem_expanded_utility import load_inputs, read, write, save, sha, load_sample, REPO

ATOL, RTOL = 1e-5, 1e-6


def key(root, token):
    return root.name + ":" + token


def feature_path(out, root, token):
    return out / "features" / root.name / (token + ".npz")


def wanted_records(source, addition, validation):
    result = []
    for root in (source, addition, validation):
        selected = {r["token"] for r in read(addition / "label_selection.json")["records"]} if root == addition else None
        for r in read(root / "splits.json")["records"]:
            if root == source:
                allowed = r["role"] in ("memory", "fit", "development")
            elif root == validation:
                allowed = r["role"] == "development"
            else:
                allowed = r["role"] in ("memory", "fit") and r["token"] in selected
            if allowed:
                result.append((root, r))
    keys = [key(root, r["token"]) for root, r in result]
    if len(set(keys)) != len(keys) or len({r["token"] for _, r in result}) != len(result):
        raise ValueError("Duplicate decoder records")
    return result


def parity(old, fresh, vocab):
    for field in ("candidate_ids", "selected_local"):
        if not np.array_equal(old[field], fresh[field]):
            raise RuntimeError("Decoder export changed " + field)
    errors = {}
    for field in ("selection_scores",) + tuple(k for k in old if k.startswith("head_")):
        if not np.allclose(old[field], fresh[field], atol=ATOL, rtol=RTOL):
            raise RuntimeError("Decoder export numeric parity failed: " + field)
        errors[field] = float(np.max(np.abs(old[field] - fresh[field])))
    ids, b = fresh["candidate_ids"], int(fresh["selected_local"])
    if b != int(np.argmax(fresh["selection_scores"])) or not np.array_equal(fresh["final_traj"], vocab[ids[b]]):
        raise RuntimeError("Decoder export changed selected trajectory")
    return max(errors.values())


def status(out, records, identity):
    index = read(out / "export_index.json") if (out / "export_index.json").exists() else {}
    expected = {key(root, r["token"]) for root, r in records}
    if set(index) - expected:
        raise RuntimeError("Unexpected decoder tokens")
    for root, r in records:
        k = key(root, r["token"])
        if k in index:
            entry = index[k]
            if sha(feature_path(out, root, r["token"])) != entry["sha256"] or sha(root / "features" / (r["token"] + ".npz")) != entry["source_sha256"]:
                raise RuntimeError("Decoder/source feature changed")
    return index, dict(status="complete" if len(index) == len(records) else "partial", required_scenes=len(records),
                       exported_scenes=len(index), missing_scenes=len(records) - len(index), identity_sha256=sha(out / "identity.json"))


def replay_report(index, report):
    passes = [entry["natural_pool_passes"] for entry in index.values()]
    return dict(report, protocol="conditional_frozen_pool_replay",
                scenes_with_natural_order_drift=sum(any(not p["same_order"] for p in pair) for pair in passes),
                scenes_with_natural_membership_drift=sum(any(not p["same_set"] for p in pair) for pair in passes))


def main(cfg):
    from navsim.planning.script.run_drivemem_memory import verify_identity
    source = Path(cfg.drivemem.root).resolve()
    verify_identity(cfg, source)
    base = Path(os.environ["NAVSIM_EXP_ROOT"]) / "drivemem_memory"
    prior = Path(os.environ.get("DRIVEMEM_PRIOR_UTILITY_ROOT", str(base / "experiments/pair_utility_v1"))).resolve()
    addition = Path(os.environ.get("DRIVEMEM_NEW_MF_ROOT", str(base / "mf_new4_shard174_v1"))).resolve()
    out = Path(os.environ.get("DRIVEMEM_DECODER_FEATURE_ROOT", str(base / "decoder_features_replay_v1"))).resolve()
    _, _, _, vocab, actual_source, validation, _, bindings, _, _ = load_inputs(prior, addition)
    if source != actual_source:
        raise ValueError("Export config must use original budget source")
    for root in (source, addition, validation, prior):
        if root == out or root in out.parents or out in root.parents:
            raise ValueError("Separate decoder sidecar required")
    original = read(source / "identity.json")
    for root in (addition, validation):
        other = read(root / "identity.json")
        if {k: v for k, v in original.items() if k != "settings"} != {k: v for k, v in other.items() if k != "settings"}:
            raise ValueError("Teacher/scorer identities differ")
    records = wanted_records(source, addition, validation)
    if len(records) != 152:
        raise ValueError(f"Expected 152 M/F/development records, got {len(records)}")
    identity = dict(inputs=bindings, exporter_sha256=sha(Path(__file__)), atol=ATOL, rtol=RTOL,
        exporter_path="navsim/planning/script/run_drivemem_decoder_replay_export.py",
        pool_helper_sha256=sha(REPO / "navsim/agents/drivemem_frozen_pool.py"),
        protocol="Conditional replay of saved 256 IDs before refinement; decoder hook/unhooked comparisons both use frozen pool. Natural coarse drift is recorded. Original final scores/heads/selection must still match.",
        hook="teacher.model._trajectory_offset_head.transformer_blocks[-1]: final layer output",
        records=[dict(key=key(root, r["token"]), role=r["role"], source=str(root), token=r["token"]) for root, r in records])
    stage = str(cfg.drivemem.stage)
    marker = out / "identity.json"
    if marker.exists() and read(marker) != identity:
        raise RuntimeError("Decoder export protocol changed; use a new DRIVEMEM_DECODER_FEATURE_ROOT")
    if out.exists() and not marker.exists() and any(out.iterdir()):
        raise ValueError("Nonempty sidecar without identity")
    if stage == "decoder-replay-plan":
        print(json.dumps(dict(output=str(out), scenes=152, protocol=identity["protocol"],
                              gpu_required_for_export=True, new_scorer_calls=0, download_GB=0), indent=2))
        return
    if stage not in ("decoder-replay-export", "decoder-replay-status"):
        raise ValueError("Unknown decoder stage")
    if stage == "decoder-replay-status" and not marker.exists():
        raise FileNotFoundError("Decoder export has not started")
    write(marker, identity)
    index, report = status(out, records, identity)
    report = replay_report(index, report)
    todo = [(root, r) for root, r in records if key(root, r["token"]) not in index]
    if stage == "decoder-replay-status" or not todo:
        write(out / "report.json", report)
        print(json.dumps(report, indent=2))
        return
    started = time.monotonic()
    import torch
    from hydra.utils import instantiate
    from torch.utils.data import DataLoader
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.training.dataset_ssl import DatasetSSL
    from navsim.planning.script.run_drivemem_oracle_pilot import on_device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; completed sidecars are preserved")
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
        limit = int(os.environ.get("DRIVEMEM_DECODER_EXPORT_LIMIT", "152"))
        if not 1 <= limit <= 152:
            raise ValueError("Export limit must be 1..152")
        done = 0
        for root in (source, addition, validation):
            for log in sorted({r["log_name"] for rt, r in todo if rt == root}):
                if done >= limit:
                    break
                if sha(Path(cfg.navsim_log_path) / (log + ".pkl")) != read(root / "splits.json")["log_sha256"][log]:
                    raise RuntimeError("Source log changed")
                selected = [r for rt, r in todo if rt == root and r["log_name"] == log][:limit - done]
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
                        if not np.array_equal(vocab, active_vocab):
                            raise RuntimeError("Vocabulary changed")
                        try:
                            error = max(parity(old, fresh, vocab), parity(unhooked, fresh, vocab))
                        except RuntimeError as exc:
                            raise RuntimeError(f"Frozen-pool replay parity failed token={token}: {exc}") from exc
                        hidden = capture["tensor"]
                        for name, head in teacher._trajectory_offset_head.multi_stage_heads[-1].items():
                            check = head(hidden).squeeze(-1).cpu().numpy()[0]
                            if not np.allclose(check, fresh["head_" + name], atol=ATOL, rtol=RTOL):
                                raise RuntimeError("Decoder/candidate alignment failed at head " + name)
                        latent = hidden.cpu().numpy()[0].copy()
                    if latent.ndim != 2 or latent.shape[0] != len(fresh["candidate_ids"]) or not np.isfinite(latent).all():
                        raise RuntimeError("Invalid decoder feature shape/values")
                    path = feature_path(out, root, token)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save(path, candidate_ids=fresh["candidate_ids"], decoder_states=latent)
                    index[key(root, token)] = dict(sha256=sha(path), source_sha256=source_index[token], max_parity_error=error, natural_pool_passes=replay["natural"])
                    write(out / "export_index.json", index)
                    done += 1
                    print(f"DECODER EXPORT {len(index)}/152 {root.name} {token} shape={latent.shape}", flush=True)
                    del pred, fresh, hidden, latent, batch
                    capture.clear()
        _, report = status(out, records, identity)
        report = replay_report(index, report)
        report.update(newly_exported_scenes=done, seconds=time.monotonic() - started)
        write(out / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)

    finally:
        coarse_handle.remove()
        pool_handle.remove()


if __name__ == "__main__":
    import hydra
    hydra.main(config_path="config/pdm_scoring", config_name="default_drivemem_memory", version_base=None)(main)()
