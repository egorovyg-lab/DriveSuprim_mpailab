"""Resumable CPU residual-critic study. No new data or holdout evaluation."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
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

from navsim.agents import selector_residual_critic as model
from navsim.agents import selector_training_bundle as bundles
from navsim.agents.drivemem import METRICS, SAFETY
from navsim.planning.script.run_drivemem_action_transfer import record, summary
from navsim.planning.script.run_drivemem_expanded_utility import REPO, read, write, sha

sys.path.insert(0, str(REPO / "scripts/drivesuprim/evaluation"))
from selector_matched import load_metadata, load_run

NOTE = ("Training-session OOF tuning study; no independent performance or safety claim. "
        "Class-weighted risk estimates and mean-minus-seed-SD gain are uncalibrated. "
        "Candidate rows and overlapping CV training sets are correlated. Local PDM, not EPDMS. "
        "Existing historical development and four fresh holdout sessions are not evaluated.")
SELECTION = dict(minimum_interventions=10, minimum_intervention_sessions=3, minimum_improved_sessions=3,
                 require_zero_observed_harm=True, maximum_single_quality_loss_pp=5.,
                 require_positive_uniform_macro_gain=True)


def csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def locked(root):
    """OS releases the advisory lock even after a crash; no stale-lock deletion."""
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / ".run.lock").open("a+b")
    acquired = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0, 2)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise RuntimeError("Another critic process is using " + str(root)) from exc
        yield
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def source_snapshot():
    names = ("navsim/agents/selector_residual_critic.py", "navsim/planning/script/run_selector_residual_night.py",
             "navsim/agents/trajectory_selector.py", "navsim/agents/trajectory_selector_v2.py",
             "navsim/agents/selector_training_bundle.py", "navsim/agents/drivemem.py",
             "navsim/planning/script/run_drivemem_action_transfer.py", "navsim/planning/script/run_drivemem_expanded_utility.py",
             "scripts/drivesuprim/evaluation/selector_matched.py", "scripts/drivesuprim/evaluation/run_selector_residual_night.sh")
    return {name: sha(REPO / name) for name in names}


def freeze_identity(out, entries, config, seeds, variants, inputs, synthetic=False):
    identity = dict(schema=1, config=asdict(config), seeds=list(seeds), variants=list(variants),
        inputs=inputs, sources=source_snapshot(), policies=model.policies(), selection=SELECTION,
        counts=dict(scenes=len(entries), sessions=len({e["sample"]["session_id"] for e in entries})),
        threads=torch.get_num_threads(), python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        platform=platform.system(), device="cpu", synthetic=synthetic, note=NOTE)
    path = out / "identity.json"
    if path.exists():
        if read(path) != identity:
            raise RuntimeError("Critic code, inputs, runtime or configuration changed; use a new --out root")
    else:
        if any(p.name != ".run.lock" for p in out.iterdir()):
            raise ValueError("Nonempty output without critic identity")
        write(path, identity)
    return identity


def load_data(bundle, reference, addition, config):
    examples, vocab = bundles.load_bundle(bundle)
    bundle_id, index = load_metadata(bundle)
    tag_path = addition / "label_selection.json"
    if bundle_id["inputs"].get(str(tag_path)) != sha(tag_path):
        raise ValueError("Challenge tags are not the bundled snapshot")
    challenge = {r["token"] for r in read(tag_path)["records"] if r["sampling"].startswith("challenge")}
    entries = model.build_entries(examples, vocab, challenge, config)
    reference_id, fit, frame = load_run(reference, bundle, index)
    if fit["config"]["epochs"] != config.epochs:
        raise ValueError("Use the same epoch budget as the frozen control")
    runtime = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                   platform=platform.system(), threads=torch.get_num_threads())
    if any(reference_id[k] != value for k, value in runtime.items()):
        raise ValueError("Frozen control/runtime mismatch, including CPU threads")
    lookup = {e["sample"]["token"]: e for e in entries}
    # Reference decisions must still describe these exact frozen top16 outcomes.
    for row in frame.itertuples():
        e = lookup[row.token]
        local = int(row.candidate_local)
        if local not in e["indices"] or int(e["sample"]["candidate_ids"][local]) != row.candidate_id:
            raise ValueError("Reference candidate alignment mismatch")
        actual = record(e["sample"], e["outcomes"], local)
        if (abs(actual["raw_score"] - row.raw_score) > 1e-10 or abs(actual["raw_gain_pp"] - row.raw_gain_pp) > 1e-8
                or actual["raw_harm"] != row.raw_harm
                or any(abs(actual["raw_delta_" + m] - getattr(row, "raw_delta_" + m)) > 1e-10 for m in METRICS)):
            raise ValueError("Reference outcome mismatch")
    inputs = {str(bundle / n): sha(bundle / n) for n in ("bundle_manifest.json", "identity.json", "examples_index.json", "vocab.npz")}
    for path in (tag_path, reference / "identity.json", reference / "fit.json", reference / "fit_cv_per_scene.csv"):
        inputs[str(path)] = sha(path)
    return entries, inputs, fit["cv_methods"]


def audit(entries, config):
    if len({e["sample"]["session_id"] for e in entries}) < 3:
        raise ValueError("At least three independent training sessions required")
    stats = model.fit_statistics(entries, config)
    support = {}
    for j, key in enumerate(METRICS[i] for i in SAFETY):
        support[key] = dict(degradation_actions=int(sum(e["targets"]["risk"][:, j].sum() for e in entries)),
            degradation_scenes=sum(bool(np.any(e["targets"]["risk"][:, j])) for e in entries),
            degradation_sessions=int(stats["risk_support_sessions"][j]))
    return dict(status="complete", scenes=len(entries), sessions=len({e["sample"]["session_id"] for e in entries}),
        feature_dim=int(entries[0]["x"].shape[1]), topk=16, risk_support=support,
        origin_scenes=pd.Series([e["metadata"]["origin"] for e in entries]).value_counts().to_dict(),
        fresh_holdout_processed=0, new_scorer_calls=0, additional_download_GB=0, note=NOTE)


def validate_prediction(prediction, entries, config):
    expected = dict(gain=(len(entries), 16), correction=(len(entries), 16), risk_logits=(len(entries), 16, 5))
    if any(prediction[k].shape != shape or not np.isfinite(prediction[k]).all() for k, shape in expected.items()):
        raise RuntimeError("Invalid cached prediction shape/value")
    b = [e["baseline"] for e in entries]
    if np.any(prediction["gain"][np.arange(len(entries)), b] != 0):
        raise RuntimeError("Cached baseline gain is not zero")
    if np.max(np.abs(prediction["correction"])) > config.residual_cap + 1e-6:
        raise RuntimeError("Cached residual bound violated")


def fold(entries, variant, seed, sid, config, out):
    training = [e for e in entries if e["sample"]["session_id"] != sid]
    valid = [e for e in entries if e["sample"]["session_id"] == sid]
    partition = dict(train_sessions=sorted({e["sample"]["session_id"] for e in training}), valid_session=sid,
                    train_tokens=[e["sample"]["token"] for e in training], valid_tokens=[e["sample"]["token"] for e in valid])
    if not training or not valid or sid in partition["train_sessions"]:
        raise ValueError("Invalid whole-session partition")
    path = out / "cv" / f"{variant}_s{seed}_{hashlib.sha256(sid.encode()).hexdigest()[:16]}"
    binding = dict(identity_sha256=sha(out / "identity.json"), variant=variant, seed=seed, partition=partition)
    if path.with_suffix(".json").exists():
        meta = read(path.with_suffix(".json"))
        if meta["binding"] != binding or sha(path.with_suffix(".npz")) != meta["sha256"]:
            raise RuntimeError("Frozen critic fold changed")
        fresh = model.fit_statistics(training, config)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            prediction = {k: data[k].copy() for k in ("gain", "correction", "risk_logits")}
            stats = {k: data["stats_" + k].copy() for k in fresh}
        # Recompute TRAIN-only statistics to check cached normalization/prior/support.
        if any(not np.array_equal(stats[k], fresh[k]) for k in fresh):
            raise RuntimeError("Cached train-only statistics changed")
    else:
        print(f"FIT {variant} seed={seed} held-out={sid} train={len(training)} valid={len(valid)}", flush=True)
        started = time.monotonic()
        net, stats, history = model.train(training, variant, seed, config,
            lambda item: print(f"  epoch={item['epoch']}/{config.epochs} loss={item['loss']:.6f}", flush=True))
        prediction = model.predict(net, valid, stats)
        bundles.save(path.with_suffix(".npz"), **prediction, **{"stats_" + k: v for k, v in stats.items()})
        write(path.with_suffix(".json"), dict(binding=binding, sha256=sha(path.with_suffix(".npz")), history=history,
              training_seconds=time.monotonic() - started))
    validate_prediction(prediction, valid, config)
    return valid, prediction, stats


def final_model(entries, variant, seed, config, out):
    path = out / "models" / f"{variant}_s{seed}"
    binding = dict(identity_sha256=sha(out / "identity.json"), variant=variant, seed=seed)
    if path.with_suffix(".json").exists():
        meta = read(path.with_suffix(".json"))
        if meta["binding"] != binding or sha(path.with_suffix(".npz")) != meta["sha256"]:
            raise RuntimeError("Frozen final critic model changed")
        return meta
    print(f"FINAL {variant} seed={seed} train={len(entries)}", flush=True)
    net, stats, history = model.train(entries, variant, seed, config,
        lambda item: print(f"  epoch={item['epoch']}/{config.epochs} loss={item['loss']:.6f}", flush=True))
    arrays = model.checkpoint(net, stats)
    bundles.save(path.with_suffix(".npz"), **arrays)
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
        restored, restored_stats = model.restore({k: saved[k].copy() for k in saved.files}, variant, config)
    probe = entries[:min(4, len(entries))]
    before, after = model.predict(net, probe, stats), model.predict(restored, probe, restored_stats)
    if any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("Final checkpoint replay mismatch")
    meta = dict(binding=binding, sha256=sha(path.with_suffix(".npz")), history=history,
                parameters=sum(p.numel() for p in net.parameters()), risk_support_sessions=stats["risk_support_sessions"].tolist(),
                checkpoint_replay_exact=True)
    write(path.with_suffix(".json"), meta)
    return meta


def policy_rows(entries, predictions, support, config, variant, ensemble_name):
    combined = model.aggregate(predictions)
    rows = []
    for i, e in enumerate(entries):
        for name, policy in model.policies().items():
            p, diagnostics = model.choose(e, combined, i, policy, support[i], config)
            rows.append(dict(method=variant + "/" + ensemble_name + "/" + name, variant=variant, ensemble=ensemble_name,
                policy=name, slice=e["sampling_slice"], origin=e["metadata"]["origin"], **diagnostics,
                oracle_score=max(v["raw"]["pdm_score"] for v in e["outcomes"].values()),
                **record(e["sample"], e["outcomes"], int(e["indices"][p]))))
    return rows, combined


def eligibility(rows):
    frame = pd.DataFrame(rows)
    uniform = frame[frame["slice"] == "uniform"]
    gain = float(uniform.groupby("session_id").raw_gain_pp.mean().mean()) if not uniform.empty else 0.
    active = frame[frame.intervention]
    improved_sessions = int(frame.loc[frame.raw_gain_pp >= .5, "session_id"].nunique())
    worst = float(min(0, frame.raw_gain_pp.min()))
    return dict(uniform_gain_macro_pp=gain, harm=int(frame.raw_harm.sum()), quality_degraded=int(frame.raw_quality_degraded.sum()),
                interventions=len(active), intervention_sessions=int(active.session_id.nunique()),
                improved_sessions=improved_sessions, worst_gain_pp=worst,
                eligible=bool(gain > 1e-8 and not frame.raw_harm.any() and worst >= -SELECTION["maximum_single_quality_loss_pp"]
                    and len(active) >= SELECTION["minimum_interventions"]
                    and active.session_id.nunique() >= SELECTION["minimum_intervention_sessions"]
                    and improved_sessions >= SELECTION["minimum_improved_sessions"]))


def risk_diagnostics(entries, prediction):
    labels = np.stack([e["targets"]["risk"] for e in entries])
    mask = np.ones(labels.shape[:2], dtype=bool)
    mask[np.arange(len(entries)), [e["baseline"] for e in entries]] = False
    output = {}
    for j, metric in enumerate(METRICS[i] for i in SAFETY):
        y, p = labels[..., j][mask], prediction["risk"][..., j][mask]
        positives, negatives = int(y.sum()), int((1 - y).sum())
        rank = pd.Series(p).rank(method="average").to_numpy()
        auc = float((rank[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)) if positives and negatives else None
        output[metric] = dict(positive_candidate_rows=positives, negative_candidate_rows=negatives, auc=auc,
            mean_estimate_positive=float(p[y == 1].mean()) if positives else None,
            mean_estimate_negative=float(p[y == 0].mean()) if negatives else None,
            note="Correlated candidate rows; weighted training logits are not calibrated probabilities")
    return output


def verify_report(out):
    result = read(out / "report.json")
    if result["identity_sha256"] != sha(out / "identity.json"):
        raise RuntimeError("Report belongs to another identity")
    for name, digest in result["artifacts"].items():
        if sha(out / name) != digest:
            raise RuntimeError("Frozen night-study artifact changed: " + name)
    return result


def concise(result):
    rows = []
    for name, value in result["ensemble_methods"].items():
        row = dict(method=name, gain_macro_pp=value["gain_macro_pp"], gain_micro_pp=value["gain_micro_pp"],
                   harm=value["harm"], quality_degraded=value["quality_degraded"],
                   interventions=value["interventions"], coverage=value["coverage"],
                   gain_ci95_sessions=value["gain_ci95_sessions"])
        if name != "baseline":
            variant, _, policy = name.split("/")
            row["eligible_for_primary"] = result["trials"][name]["eligible"]
            row["intervention_sessions"] = result["trials"][name]["intervention_sessions"]
            row["improved_sessions"] = result["trials"][name]["improved_sessions"]
            row["seed_gains_macro_pp"] = {seed_name: stat["gain_macro_pp"] for seed_name, stat in result["per_seed_methods"].items()
                                        if seed_name.startswith(variant + "/") and seed_name.endswith("/" + policy)}
            row.update(result["support_and_coverage"][name])
        rows.append(row)
    return dict(status=result["status"], primary=result["primary"], counts=result["counts"], rows=rows,
                reference_methods=result["reference_methods"], risk_diagnostics=result["risk_diagnostics"],
                reproducibility=result["reproducibility"],
                elapsed_seconds=result["elapsed_seconds"], fresh_holdout_processed=0, new_scorer_calls=0,
                additional_download_GB=0, note=NOTE)


def run(entries, config, seeds, variants, out, inputs, reference_methods=None, synthetic=False, plan_only=False):
    if (len(set(seeds)) != len(seeds) or len(seeds) < 2 or not variants or len(set(variants)) != len(variants)
            or any(v not in model.VARIANTS for v in variants)):
        raise ValueError("Unique variants and at least two unique seeds required")
    sessions = sorted({e["sample"]["session_id"] for e in entries})
    with locked(out):
        identity = freeze_identity(out, entries, config, seeds, variants, inputs, synthetic)
        if (out / "report.json").exists():
            return verify_report(out)
        started = time.monotonic()
        try:
            report_audit = audit(entries, config)
            write(out / "audit.json", report_audit)
            total = len(variants) * len(seeds) * (len(sessions) + 1)
            plan = dict(status="planned", **identity["counts"], variants=list(variants), seeds=list(seeds),
                        fold_fits=len(variants) * len(seeds) * len(sessions), final_fits=len(variants) * len(seeds),
                        epochs=config.epochs, total_fits=total, policies=list(model.policies()), cpu_threads=torch.get_num_threads(),
                        output=str(out), note=NOTE)
            write(out / "plan.json", plan)
            print(json.dumps(plan, indent=2), flush=True)
            if plan_only:
                return plan
            progress = dict(status="running", started_utc=now(), completed_fits=0, total_fits=total, pid=os.getpid(), host=platform.node())
            write(out / "progress.json", progress)
            rows, ensemble_rows, diagnostics, final = [], [], {}, {}
            artifacts = {}
            for variant in variants:
                predictions, support = [], None
                for seed in seeds:
                    arrays = dict(gain=np.empty((len(entries), 16), dtype=np.float32), correction=np.empty((len(entries), 16), dtype=np.float32),
                                  risk_logits=np.empty((len(entries), 16, 5), dtype=np.float32))
                    support_for_seed = np.empty((len(entries), 5), dtype=np.int64)
                    tokens = {e["sample"]["token"]: i for i, e in enumerate(entries)}
                    covered = set()
                    for sid in sessions:
                        valid, prediction, stats = fold(entries, variant, seed, sid, config, out)
                        for j, e in enumerate(valid):
                            i = tokens[e["sample"]["token"]]
                            if i in covered:
                                raise RuntimeError("Duplicate OOF prediction")
                            covered.add(i)
                            for key in arrays:
                                arrays[key][i] = prediction[key][j]
                            support_for_seed[i] = stats["risk_support_sessions"]
                        progress.update(completed_fits=progress["completed_fits"] + 1, variant=variant, seed=seed, held_out=sid,
                                        elapsed_seconds=time.monotonic() - started, updated_utc=now())
                        write(out / "progress.json", progress)
                        print(f"PROGRESS {progress['completed_fits']}/{total}", flush=True)
                    if covered != set(range(len(entries))):
                        raise RuntimeError("Incomplete OOF coverage")
                    if support is not None and not np.array_equal(support, support_for_seed):
                        raise RuntimeError("Seed support statistics differ")
                    support = support_for_seed
                    predictions.append(arrays)
                    seed_rows, _ = policy_rows(entries, [arrays], support, config, variant, "seed" + str(seed))
                    rows.extend(seed_rows)
                    final[f"{variant}_s{seed}"] = final_model(entries, variant, seed, config, out)
                    progress.update(completed_fits=progress["completed_fits"] + 1, updated_utc=now())
                    write(out / "progress.json", progress)
                current_rows, combined = policy_rows(entries, predictions, support, config, variant, "ensemble")
                if variant == "linear" and any(not np.array_equal(predictions[0][k], p[k])
                         for p in predictions[1:] for k in predictions[0]):
                    raise RuntimeError("Zero-initialized full-batch linear replicas must be deterministic")
                rows.extend(current_rows)
                ensemble_rows.extend(current_rows)
                diagnostics[variant] = risk_diagnostics(entries, combined)
                bundles.save(out / (variant + "_oof_ensemble.npz"), **combined, risk_support_sessions=support,
                             tokens=np.array([e["sample"]["token"] for e in entries]))
            baseline_rows = [dict(method="baseline", variant="baseline", ensemble="baseline", policy="baseline", slice=e["sampling_slice"],
                                 oracle_score=max(v["raw"]["pdm_score"] for v in e["outcomes"].values()),
                                 **record(e["sample"], e["outcomes"], int(e["sample"]["selected_local"]))) for e in entries]
            csv(out / "oof_per_scene.csv", rows + baseline_rows)
            frame = pd.DataFrame(rows)
            by_session = frame.groupby(["method", "session_id"]).agg(gain_pp=("raw_gain_pp", "mean"), harm=("raw_harm", "sum"),
                                      quality_drops=("raw_quality_degraded", "sum"), interventions=("intervention", "sum")).reset_index()
            csv(out / "oof_by_session.csv", by_session.to_dict("records"))
            trials = {name: eligibility(group.to_dict("records")) for name, group in pd.DataFrame(ensemble_rows).groupby("method")}
            candidates = [name for name, value in trials.items() if value["eligible"] and "/risk_" in name]
            primary = max(candidates, key=lambda name: (trials[name]["uniform_gain_macro_pp"], -trials[name]["interventions"], name)) if candidates else "baseline"
            worst = frame[(frame.ensemble == "ensemble") & (frame.raw_gain_pp < -1e-8)].sort_values("raw_gain_pp").groupby("method").head(5)
            csv(out / "worst_decisions.csv", worst)
            selected_frame = pd.DataFrame(ensemble_rows)
            block_counts = {name: dict(scenes_blocked_by_support=int(g.support_blocked.sum()),
                              mean_eligible_alternatives=float(g.eligible_alternatives.mean()),
                              worst_gain_pp=float(min(0, g.raw_gain_pp.min())),
                              losses_ge10pp=int((g.raw_gain_pp <= -10).sum()),
                              component_degradations={m: int((g["raw_delta_" + m] < -1e-8).sum()) for m in (METRICS[j] for j in SAFETY)})
                              for name, g in selected_frame.groupby("method")}
            for path in list((out / "cv").glob("*")) + list((out / "models").glob("*")) + [
                    out / n for n in ("oof_per_scene.csv", "oof_by_session.csv", "worst_decisions.csv", "audit.json", "plan.json")] + [
                    out / (v + "_oof_ensemble.npz") for v in variants]:
                if path.suffix in (".json", ".npz", ".csv"):
                    artifacts[str(path.relative_to(out))] = sha(path)
            if source_snapshot() != identity["sources"] or any(sha(Path(path)) != digest for path, digest in inputs.items()):
                raise RuntimeError("Frozen code or input manifests changed during the night study")
            result = dict(status="complete", identity_sha256=sha(out / "identity.json"), primary=primary, counts=identity["counts"],
                selection=SELECTION, trials=trials, ensemble_methods=summary(ensemble_rows + baseline_rows, seeds[0]),
                per_seed_methods=summary([r for r in rows if r["ensemble"] != "ensemble"], seeds[0]),
                reference_methods=reference_methods or {}, risk_diagnostics=diagnostics, support_and_coverage=block_counts,
                models=final, artifacts=artifacts, elapsed_seconds=time.monotonic() - started,
                reproducibility=dict(linear_seed_replicates_exact="linear" in variants,
                    note="Linear zero initialization/full-batch optimization is deterministic; its seed SD is not uncertainty. "
                         "MLP seeds vary initialization. Neither seed SD nor risk estimates are calibrated bounds."),
                fresh_holdout_processed=0, new_scorer_calls=0, additional_download_GB=0, synthetic=synthetic, note=NOTE)
            write(out / "summary.json", concise(result))
            result["artifacts"]["summary.json"] = sha(out / "summary.json")
            write(out / "report.json", result)
            progress.update(status="complete", completed_fits=total, updated_utc=now())
            write(out / "progress.json", progress)
            return result
        except Exception as exc:
            failure = dict(status="failed", time_utc=now(), error=repr(exc), traceback=traceback.format_exc())
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            write(out / "failures" / (stamp + ".json"), failure)
            write(out / "failure.json", failure)
            progress = read(out / "progress.json") if (out / "progress.json").exists() else {}
            progress.update(status="failed", error=repr(exc), updated_utc=now())
            write(out / "progress.json", progress)
            raise


def smoke():
    """Small full study with synthetic inputs/outcomes; does not read real data."""
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
            raw = dict(pdm_score=.8 if j == 3 else (.1 if j == 5 else .6), **{k: 1. for k in METRICS})
            if j == 5:
                raw["drivable_area_compliance"] = 0.
            outcomes[str(j)] = dict(raw=raw, filtered=raw.copy())
        examples.append(bundles.example(sample, outcomes, vocab, dict(token="smoke" + str(i), session_id="s" + str(i // 2),
                            role="train", origin="synthetic", map_name="synthetic")))
    config = model.CriticConfig(width=8, epochs=3)
    entries = model.build_entries(examples, vocab, set(), config)
    with tempfile.TemporaryDirectory(dir=str(Path.cwd()), prefix="critic_smoke_") as tmp:
        out = Path(tmp).resolve() / "study"
        result = run(entries, config, (11, 12), model.VARIANTS, out, {}, synthetic=True)
        again = run(entries, config, (11, 12), model.VARIANTS, out, {}, synthetic=True)
        if result != again or result["status"] != "complete":
            raise RuntimeError("Synthetic full-study resume failed")
    return dict(status="complete", synthetic=True, scenes=len(entries), sessions=4, smoke_fits=20,
                checkpoint_and_completed_run_resume=True, fresh_holdout_processed=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory"
    p.add_argument("stage", choices=("plan", "all", "report", "status", "smoke"))
    p.add_argument("--bundle", type=Path, default=base / "selector_training_bundle_v3_30gb")
    p.add_argument("--reference", type=Path, default=base / "experiments/trajectory_selector_v3_30gb_e120")
    p.add_argument("--addition", type=Path, default=base / "mf_new4_shard174_v1")
    p.add_argument("--out", type=Path, default=base / "experiments/residual_critic_night_v1")
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
    bundle, reference, addition, out = [x.resolve() for x in (a.bundle, a.reference, a.addition, a.out)]
    for root in (bundle, reference, addition):
        if out == root or out in root.parents or root in out.parents:
            raise ValueError("Separate night-study root required")
    if a.stage == "status":
        state = read(out / "progress.json") if (out / "progress.json").exists() else dict(status="not_started")
        active = False
        if out.exists():
            try:
                with locked(out):
                    pass
            except RuntimeError:
                active = True
        state["process_holds_run_lock"] = active
        if state["status"] == "running" and not active:
            state["hint"] = "Previous process stopped; rerun all with unchanged code/configuration to resume"
        print(json.dumps(state, indent=2))
        return
    config = model.CriticConfig(epochs=a.epochs)
    if a.stage == "report" and not (out / "report.json").exists():
        raise SystemExit("Night study is not complete; use status and inspect the log/failure.json")
    entries, inputs, reference_methods = load_data(bundle, reference, addition, config)
    if a.stage == "report":
        with locked(out):
            freeze_identity(out, entries, config, a.seeds, a.variants, inputs)
            result = verify_report(out)
    else:
        result = run(entries, config, a.seeds, a.variants, out, inputs, reference_methods, plan_only=a.stage == "plan")
    print(json.dumps(result if a.full_report or result.get("status") != "complete" else concise(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
