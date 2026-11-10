"""Frozen CPU component-critic study, matched to the previous training sessions."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
import torch

from navsim.agents import selector_component_critic as model
from navsim.agents import selector_training_bundle as bundles
from navsim.agents.drivemem import METRICS
from navsim.planning.script import run_selector_residual_night as previous
from navsim.planning.script.run_drivemem_action_transfer import record, summary
from navsim.planning.script.run_drivemem_expanded_utility import REPO, read, write, sha

sys.path.insert(0, str(REPO / "scripts/drivesuprim/evaluation"))
import selector_teacher_veto as veto

POLICIES = ("raw", "gain_m020")
FIXED_CANDIDATE = "mlp/ensemble/gain_m020"
NOTE = ("Training-session OOF tuning study prompted by observed errors, not independent validation. "
    "Same frozen inputs, whole-session folds, seeds and epoch budget as the residual-critic study. "
    "The new readout and component supervision change the algorithm, not just its parameter count. "
    "Predicted components and their product are uncalibrated; product of means is not the expected joint score. "
    "Mean-minus-seed-SD is a heuristic, not a confidence bound. Candidate rows and CV fits are correlated. "
    "Only the fixed MLP/gain_m020 candidate can become primary. Local PDM, not EPDMS. Fresh holdout unopened.")


def sources():
    result = previous.source_snapshot()
    names = ("navsim/agents/selector_component_critic.py", "navsim/planning/script/run_selector_component_critic.py",
        "scripts/drivesuprim/evaluation/run_selector_component_critic.sh", "scripts/drivesuprim/evaluation/selector_teacher_veto.py",
        "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
        "navsim/planning/script/config/pdm_scoring/scorer/pdm_scorer.yaml")
    result.update({name: sha(REPO / name) for name in names})
    return result


def load_data(bundle, night, config, seeds):
    identity, old_report = veto.verify_night(night, bundle)
    runtime = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                   platform=platform.system(), threads=torch.get_num_threads())
    if (any(identity[k] != v for k, v in runtime.items()) or config.epochs != identity["config"]["epochs"]
            or list(seeds) != identity["seeds"]):
        raise ValueError("Use the frozen night study's runtime, CPU threads, seeds and epoch budget")
    examples, vocab = bundles.load_bundle(bundle)
    tags = [Path(p) for p in identity["inputs"] if p.endswith("label_selection.json")]
    if len(tags) != 1 or sha(tags[0]) != identity["inputs"][str(tags[0])]:
        raise ValueError("Frozen challenge tags changed or ambiguous")
    challenge = {r["token"] for r in read(tags[0])["records"] if r["sampling"].startswith("challenge")}
    entries = model.build_entries(examples, vocab, challenge, config)
    tokens = {e["sample"]["token"] for e in entries}
    with np.load(night / "mlp_oof_ensemble.npz", allow_pickle=False) as f:
        if len(f["tokens"]) != len(tokens) or set(f["tokens"].tolist()) != tokens:
            raise ValueError("Reference OOF coverage differs from the training bundle")
    frame = pd.read_csv(night / "oof_per_scene.csv", dtype={"token": str}, low_memory=False, float_precision="round_trip")
    controls = ("linear/ensemble/gain_m020", "mlp/ensemble/gain_m020", "mlp/ensemble/risk_t05_m020")
    frame = frame[frame.method.isin(controls)].copy()
    if frame.duplicated(["method", "token"]).any() or set(frame.method) != set(controls) or any(set(g.token) != tokens for _, g in frame.groupby("method")):
        raise ValueError("Frozen control decisions incomplete or duplicated")
    lookup = {e["sample"]["token"]: e for e in entries}
    for row in frame.itertuples():
        e = lookup[row.token]
        if row.candidate_local not in e["indices"]:
            raise ValueError("Reference selection outside top16")
        actual = record(e["sample"], e["outcomes"], int(row.candidate_local))
        numeric = ("raw_score", "raw_gain_pp", "filtered_score", "filtered_gain_pp") + tuple("raw_delta_" + m for m in METRICS)
        if (row.candidate_id != actual["candidate_id"] or row.session_id != actual["session_id"]
                or row.raw_harm != actual["raw_harm"]
                or any(not np.isfinite(getattr(row, k)) or abs(getattr(row, k) - actual[k]) > 1e-8 for k in numeric)):
            raise ValueError("Frozen reference outcome replay failed")
    inputs = {str(p): sha(p) for p in (bundle / "identity.json", bundle / "bundle_manifest.json", tags[0],
        night / "identity.json", night / "report.json", night / "oof_per_scene.csv", night / "mlp_oof_ensemble.npz")}
    frame["method"] = "reference/" + frame.method
    return entries, inputs, frame.to_dict("records"), old_report["identity_sha256"]


def freeze(out, entries, config, seeds, variants, inputs, synthetic):
    identity = dict(schema=1, config=asdict(config), seeds=list(seeds), variants=list(variants), inputs=inputs,
        sources=sources(), formula=model.FORMULA, policies=list(POLICIES), fixed_candidate=FIXED_CANDIDATE,
        counts=dict(scenes=len(entries), sessions=len({e["sample"]["session_id"] for e in entries})),
        python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, platform=platform.system(),
        threads=torch.get_num_threads(), synthetic=synthetic, note=NOTE)
    marker = out / "identity.json"
    if marker.exists():
        if read(marker) != identity:
            raise RuntimeError("Component code/inputs/runtime/config changed; use a new --out root")
    elif any(p.name != ".run.lock" for p in out.iterdir()):
        raise ValueError("Nonempty component output without identity")
    write(marker, identity)
    return identity


def formula_stage(entries, out):
    report, mismatches = model.formula_audit(entries)
    write(out / "formula_audit.json", report)
    if mismatches:
        previous.csv(out / "formula_mismatches.csv", mismatches)
        raise RuntimeError("Stored RAW PDM does not match the fixed formula; inspect formula_audit.json/formula_mismatches.csv. No fitting performed.")
    return report


def fold(entries, variant, seed, sid, config, out):
    train = [e for e in entries if e["sample"]["session_id"] != sid]
    valid = [e for e in entries if e["sample"]["session_id"] == sid]
    if not train or not valid:
        raise ValueError("Nonempty whole-session training/validation required")
    partition = dict(train_sessions=sorted({e["sample"]["session_id"] for e in train}), valid_session=sid,
        train_tokens=[e["sample"]["token"] for e in train], valid_tokens=[e["sample"]["token"] for e in valid])
    path = out / "cv" / f"{variant}_s{seed}_{hashlib.sha256(sid.encode()).hexdigest()[:16]}"
    binding = dict(identity_sha256=sha(out / "identity.json"), variant=variant, seed=seed, partition=partition)
    stats = model.fit_statistics(train)
    if path.with_suffix(".json").exists():
        meta = read(path.with_suffix(".json"))
        if meta["binding"] != binding or meta["sha256"] != sha(path.with_suffix(".npz")):
            raise RuntimeError("Frozen component fold changed")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as f:
            prediction = {k: f[k].copy() for k in ("components", "score", "gain")}
            if any(not np.array_equal(stats[k], f["stats_" + k]) for k in stats):
                raise RuntimeError("TRAIN-only fold statistics changed")
    else:
        print(f"FIT {variant} seed={seed} held-out={sid} train={len(train)} valid={len(valid)}", flush=True)
        started = time.monotonic()
        net, stats, history = model.train(train, variant, seed, config,
            lambda item: print(f"  epoch={item['epoch']}/{config.epochs} loss={item['loss']:.6f}", flush=True))
        prediction = model.predict(net, valid, stats)
        bundles.save(path.with_suffix(".npz"), **prediction, **{"stats_" + k: v for k, v in stats.items()})
        write(path.with_suffix(".json"), dict(binding=binding, sha256=sha(path.with_suffix(".npz")), history=history,
            training_seconds=time.monotonic() - started))
    model.validate(prediction, valid)
    return valid, prediction


def final_model(entries, variant, seed, config, out):
    path = out / "models" / f"{variant}_s{seed}"
    binding = dict(identity_sha256=sha(out / "identity.json"), variant=variant, seed=seed)
    if path.with_suffix(".json").exists():
        meta = read(path.with_suffix(".json"))
        if meta["binding"] != binding or meta["sha256"] != sha(path.with_suffix(".npz")):
            raise RuntimeError("Frozen component checkpoint changed")
        return meta
    print(f"FINAL {variant} seed={seed} train={len(entries)}", flush=True)
    net, stats, history = model.train(entries, variant, seed, config)
    bundles.save(path.with_suffix(".npz"), **model.checkpoint(net, stats))
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as f:
        restored, other = model.restore({k: f[k].copy() for k in f.files}, variant, config)
    probe = entries[:min(4, len(entries))]
    a, b = model.predict(net, probe, stats), model.predict(restored, probe, other)
    if any(not np.array_equal(a[k], b[k]) for k in a):
        raise RuntimeError("Component checkpoint replay mismatch")
    meta = dict(binding=binding, sha256=sha(path.with_suffix(".npz")), history=history,
        parameters=sum(p.numel() for p in net.parameters()), checkpoint_replay_exact=True)
    write(path.with_suffix(".json"), meta)
    return meta


def teacher_prediction(entries):
    with torch.no_grad():
        c = torch.tensor(np.stack([e["teacher_logits"] for e in entries]), dtype=torch.float32).sigmoid().numpy()
    score = model.pdm(c)
    b = [e["baseline"] for e in entries]
    result = dict(components=c, score=score, gain=score - score[np.arange(len(entries)), b, None])
    model.validate(result, entries)
    return result


def policy_rows(entries, prediction, variant, ensemble):
    rows = []
    for i, e in enumerate(entries):
        b = e["baseline"]
        for policy in POLICIES:
            chosen = model.choose(e, prediction, i, policy)
            row = dict(method=f"{variant}/{ensemble}/{policy}", ensemble=ensemble, slice=e["sampling_slice"],
                origin=e["metadata"]["origin"], predicted_gain_pp=100 * float(prediction["gain"][i, chosen]),
                gain_lower_pp=100 * float(prediction["gain_lower"][i, chosen]),
                oracle_score=max(v["raw"]["pdm_score"] for v in e["outcomes"].values()),
                **record(e["sample"], e["outcomes"], int(e["indices"][chosen])))
            for j, m in enumerate(METRICS):
                row["predicted_delta_" + m] = float(prediction["components"][i, chosen, j] - prediction["components"][i, b, j])
            rows.append(row)
    return rows


def component_diagnostics(entries, prediction, teacher):
    truth = np.stack([e["targets"]["components"] for e in entries]).astype(float)
    b = [e["baseline"] for e in entries]
    mask = np.ones(truth.shape[:2], bool)
    mask[np.arange(len(entries)), b] = False
    w = model.original.session_weights([e["sample"] for e in entries])
    target_delta = truth - truth[np.arange(len(entries)), b, None, :]
    result = {}
    for name, pred in (("teacher", teacher), ("component_model", prediction)):
        c = pred["components"]
        delta = c - c[np.arange(len(entries)), b, None, :]
        rows = {}
        for j, metric in enumerate(METRICS):
            absolute = np.abs(c[..., j] - truth[..., j]).mean(1)
            error = (np.abs(delta[..., j] - target_delta[..., j]) * mask).sum(1) / mask.sum(1)
            down = (target_delta[..., j] < -1e-3) & mask
            rows[metric] = dict(absolute_mae_macro=float(absolute @ w), delta_mae_macro=float(error @ w),
                actual_down_actions=int(down.sum()), predicted_down_on_actual_down_fraction=float((delta[..., j][down] < 0).mean()) if down.any() else None)
        result[name] = rows
    return result


def paired_comparisons(rows, controls, seed):
    if not controls:
        return {}
    current, old = pd.DataFrame(rows), pd.DataFrame(controls)
    result = {}
    for method, group in current.groupby("method"):
        if "/ensemble/" not in method:
            continue
        for control, ref in old.groupby("method"):
            paired = group[["token", "session_id", "raw_gain_pp"]].merge(ref[["token", "session_id", "raw_gain_pp"]],
                on=["token", "session_id"], validate="one_to_one", suffixes=("_new", "_old"))
            if len(paired) != len(group) or len(paired) != len(ref):
                raise RuntimeError("Matched comparison coverage mismatch")
            paired["difference"] = paired.raw_gain_pp_new - paired.raw_gain_pp_old
            sessions = paired.groupby("session_id").difference.mean().to_numpy()
            draws = np.random.default_rng(seed).integers(0, len(sessions), size=(2000, len(sessions)))
            result[method + " vs " + control] = dict(delta_macro_pp=float(sessions.mean()),
                delta_micro_pp=float(paired.difference.mean()), delta_ci95_sessions=np.quantile(sessions[draws].mean(1), [.025, .975]).tolist())
    return result


def concise(result):
    keys = ("status", "primary", "fixed_candidate", "counts", "formula_audit", "trials", "rows", "component_diagnostics",
            "paired_comparisons", "elapsed_seconds", "fresh_holdout_processed", "new_scorer_calls", "additional_download_GB", "note")
    return {k: result[k] for k in keys}


def run(entries, config, seeds, variants, out, inputs, controls=(), synthetic=False, stage="all"):
    if (not entries or len(set(seeds)) != len(seeds) or len(seeds) < 2 or not variants
            or len(set(variants)) != len(variants) or any(v not in model.VARIANTS for v in variants)):
        raise ValueError("Nonempty entries, unique variants and at least two unique seeds required")
    sessions = sorted({e["sample"]["session_id"] for e in entries})
    if len(sessions) < 3:
        raise ValueError("At least three training sessions required")
    with previous.locked(out):
        identity = freeze(out, entries, config, seeds, variants, inputs, synthetic)
        if (out / "report.json").exists():
            return previous.verify_report(out)
        started = time.monotonic()
        try:
            audit = formula_stage(entries, out)
            print(json.dumps(audit, indent=2), flush=True)
            if stage == "formula":
                return audit
            total = len(variants) * len(seeds) * (len(sessions) + 1)
            plan = dict(status="planned", **identity["counts"], variants=list(variants), seeds=list(seeds),
                fold_fits=len(variants) * len(seeds) * len(sessions), final_fits=len(variants) * len(seeds),
                total_fits=total, epochs=config.epochs, fixed_candidate=FIXED_CANDIDATE, output=str(out), note=NOTE)
            write(out / "plan.json", plan)
            print(json.dumps(plan, indent=2), flush=True)
            if stage == "plan":
                return plan
            progress = dict(status="running", completed_fits=0, total_fits=total, pid=os.getpid(), started_utc=previous.now())
            write(out / "progress.json", progress)
            teacher = teacher_prediction(entries)
            teacher_rows = policy_rows(entries, model.aggregate([teacher], entries), "teacher_formula", "ensemble")
            all_rows, ensemble_rows, diagnostics, final = list(teacher_rows), list(teacher_rows), {}, {}
            positions = {e["sample"]["token"]: i for i, e in enumerate(entries)}
            for variant in variants:
                replicas = []
                for seed in seeds:
                    arrays = dict(components=np.empty((len(entries), 16, 8), dtype=np.float32),
                                  score=np.empty((len(entries), 16), dtype=np.float32), gain=np.empty((len(entries), 16), dtype=np.float32))
                    covered = set()
                    for sid in sessions:
                        valid, prediction = fold(entries, variant, seed, sid, config, out)
                        for j, e in enumerate(valid):
                            i = positions[e["sample"]["token"]]
                            if i in covered:
                                raise RuntimeError("Duplicate component OOF coverage")
                            covered.add(i)
                            for key in arrays:
                                arrays[key][i] = prediction[key][j]
                        progress.update(completed_fits=progress["completed_fits"] + 1, variant=variant, seed=seed,
                            held_out=sid, elapsed_seconds=time.monotonic() - started, updated_utc=previous.now())
                        write(out / "progress.json", progress)
                        print(f"PROGRESS {progress['completed_fits']}/{total}", flush=True)
                    if covered != set(range(len(entries))):
                        raise RuntimeError("Incomplete component OOF coverage")
                    model.validate(arrays, entries)
                    replicas.append(arrays)
                    all_rows.extend(policy_rows(entries, model.aggregate([arrays], entries), variant, "seed" + str(seed)))
                    final[f"{variant}_s{seed}"] = final_model(entries, variant, seed, config, out)
                    progress.update(completed_fits=progress["completed_fits"] + 1, updated_utc=previous.now())
                    write(out / "progress.json", progress)
                if variant == "linear" and any(not np.array_equal(replicas[0][k], p[k]) for p in replicas[1:] for k in p):
                    raise RuntimeError("Zero-initialized full-batch linear replicas must match")
                combined = model.aggregate(replicas, entries)
                current = policy_rows(entries, combined, variant, "ensemble")
                all_rows.extend(current)
                ensemble_rows.extend(current)
                diagnostics[variant] = component_diagnostics(entries, combined, teacher)
                bundles.save(out / (variant + "_oof_ensemble.npz"), **combined,
                    tokens=np.array([e["sample"]["token"] for e in entries]))
            baseline = [dict(method="baseline", slice=e["sampling_slice"], ensemble="baseline",
                oracle_score=max(v["raw"]["pdm_score"] for v in e["outcomes"].values()),
                **record(e["sample"], e["outcomes"], int(e["sample"]["selected_local"]))) for e in entries]
            previous.csv(out / "oof_per_scene.csv", all_rows + baseline)
            previous.csv(out / "reference_per_scene.csv", controls)
            df = pd.DataFrame(ensemble_rows)
            previous.csv(out / "oof_by_session.csv", df.groupby(["method", "session_id"]).agg(
                gain_pp=("raw_gain_pp", "mean"), harm=("raw_harm", "sum"), interventions=("intervention", "sum")).reset_index())
            previous.csv(out / "worst_decisions.csv", df[df.raw_gain_pp < -1e-8].sort_values("raw_gain_pp").groupby("method").head(5))
            trials = {name: previous.eligibility(group.to_dict("records")) for name, group in df.groupby("method")}
            primary = FIXED_CANDIDATE if trials.get(FIXED_CANDIDATE, {}).get("eligible", False) else "baseline"
            methods = summary(ensemble_rows + baseline + list(controls), seeds[0])
            rows = [dict(method=name, gain_macro_pp=v["gain_macro_pp"], gain_micro_pp=v["gain_micro_pp"], harm=v["harm"],
                quality_degraded=v["quality_degraded"], interventions=v["interventions"], gain_ci95_sessions=v["gain_ci95_sessions"])
                for name, v in methods.items()]
            artifact_paths = list((out / "cv").glob("*")) + list((out / "models").glob("*"))
            artifact_paths += [p for p in out.iterdir() if p.suffix in (".csv", ".npz") or p.name in ("formula_audit.json", "plan.json")]
            if sources() != identity["sources"] or any(sha(Path(path)) != digest for path, digest in inputs.items()):
                raise RuntimeError("Frozen code/input manifests changed during component study")
            result = dict(status="complete", identity_sha256=sha(out / "identity.json"), primary=primary,
                fixed_candidate=FIXED_CANDIDATE, counts=identity["counts"], formula_audit=audit, trials=trials, rows=rows,
                ensemble_methods=methods, component_diagnostics=diagnostics, models=final,
                paired_comparisons=paired_comparisons(ensemble_rows, controls, seeds[0]),
                artifacts={str(p.relative_to(out)): sha(p) for p in artifact_paths}, elapsed_seconds=time.monotonic() - started,
                fresh_holdout_processed=0, new_scorer_calls=0, additional_download_GB=0, synthetic=synthetic, note=NOTE)
            write(out / "summary.json", concise(result))
            result["artifacts"]["summary.json"] = sha(out / "summary.json")
            write(out / "report.json", result)
            progress.update(status="complete", completed_fits=total, updated_utc=previous.now())
            write(out / "progress.json", progress)
            return result
        except Exception as exc:
            failure = dict(status="failed", error=repr(exc), traceback=traceback.format_exc(), time_utc=previous.now())
            write(out / "failures" / (str(time.time_ns()) + ".json"), failure)
            write(out / "failure.json", failure)
            state = read(out / "progress.json") if (out / "progress.json").exists() else {}
            state.update(status="failed", error=repr(exc), updated_utc=previous.now())
            write(out / "progress.json", state)
            raise


def smoke():
    rng = np.random.default_rng(5)
    vocab = rng.normal(0, .01, (256, 40, 3)).astype(np.float32)
    examples = []
    for i in range(8):
        sample = dict(candidate_ids=np.arange(256), selected_local=np.array(0), selection_scores=np.linspace(1, .9, 256),
            decoder_states=rng.normal(0, .01, (256, 256)).astype(np.float32), scene_embedding=np.ones(8), ego_status=np.zeros(8))
        sample.update({"head_" + k: np.zeros(256) for k in (*METRICS, "imi")})
        sample["decoder_states"][3, 0], sample["decoder_states"][5, 0] = 1., -1.
        outcomes = {}
        for j in range(16):
            c = np.ones(8)
            c[4] = .9 if j == 3 else .5
            if j == 5:
                c[1] = 0
            raw = dict(pdm_score=float(model.pdm(c)), **dict(zip(METRICS, c.tolist())))
            outcomes[str(j)] = dict(raw=raw, filtered=raw.copy())
        examples.append(bundles.example(sample, outcomes, vocab, dict(token="smoke" + str(i), session_id="s" + str(i // 2),
            role="train", origin="synthetic", map_name="synthetic")))
    config = model.ComponentConfig(width=8, epochs=3)
    entries = model.build_entries(examples, vocab, set(), config)
    with tempfile.TemporaryDirectory(dir=str(Path.cwd()), prefix="component_smoke_") as tmp:
        out = Path(tmp).resolve() / "study"
        result = run(entries, config, (11, 12), model.VARIANTS, out, {}, synthetic=True)
        repeated = run(entries, config, (11, 12), model.VARIANTS, out, {}, synthetic=True)
        if result != repeated:
            raise RuntimeError("Component smoke resume mismatch")
    return dict(status="complete", synthetic=True, smoke_fits=20, scenes=8, sessions=4,
                formula_checked=True, checkpoint_replay=True, completed_run_resume=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory"
    p.add_argument("stage", choices=("formula", "plan", "all", "report", "status", "smoke"))
    p.add_argument("--bundle", type=Path, default=base / "selector_training_bundle_v3_30gb")
    p.add_argument("--night", type=Path, default=base / "experiments/residual_critic_night_v1")
    p.add_argument("--out", type=Path, default=base / "experiments/component_critic_v1")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=list(model.SEEDS))
    p.add_argument("--variants", choices=model.VARIANTS, nargs="+", default=list(model.VARIANTS))
    p.add_argument("--full-report", action="store_true")
    a = p.parse_args()
    if a.threads < 1:
        raise ValueError("Positive CPU thread count required")
    torch.set_num_threads(a.threads)
    if a.stage == "smoke":
        print(json.dumps(smoke(), indent=2), flush=True)
        return
    bundle, night, out = (x.resolve() for x in (a.bundle, a.night, a.out))
    for root in (bundle, night):
        if out == root or out in root.parents or root in out.parents:
            raise ValueError("Separate component-study output required")
    if a.stage == "status":
        state = read(out / "progress.json") if (out / "progress.json").exists() else dict(status="not_started")
        active = False
        if out.exists():
            try:
                with previous.locked(out):
                    pass
            except RuntimeError:
                active = True
        state["process_holds_run_lock"] = active
        print(json.dumps(state, indent=2), flush=True)
        return
    config = model.ComponentConfig(epochs=a.epochs)
    if a.stage == "report" and not (out / "report.json").exists():
        raise SystemExit("Component study is not complete; use status and inspect the log/failure.json")
    entries, inputs, controls, _ = load_data(bundle, night, config, a.seeds)
    if a.stage == "report":
        with previous.locked(out):
            freeze(out, entries, config, a.seeds, a.variants, inputs, False)
            result = previous.verify_report(out)
    else:
        result = run(entries, config, a.seeds, a.variants, out, inputs, controls, stage=a.stage)
    print(json.dumps(concise(result) if result.get("status") == "complete" and not a.full_report else result, indent=2), flush=True)


if __name__ == "__main__":
    main()
