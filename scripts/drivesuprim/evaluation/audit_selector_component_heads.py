"""Frozen OOF component-head substitutions; no fitting or new scorer calls.

Both baseline and all alternatives receive each substitution, in every seed.
Choices are frozen before measured outcomes are used to evaluate them.
"""
import argparse
import json
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd

import selector_teacher_veto as common

METRICS, SAFETY = common.METRICS, common.SAFETY
POLICIES = ("raw", "gain_m020")
SUBSTITUTIONS = {"student": (), "teacher": tuple(range(8))}
SUBSTITUTIONS.update({"restore_" + m: (j,) for j, m in enumerate(METRICS)})
SUBSTITUTIONS["restore_DAC_LK"] = (1, 6)
NOTE = ("Exploratory training OOF ablation prompted by observed failures, not independent validation. "
    "Eight single-head substitutions and the DAC+LK interaction are diagnostics, not a search for a new primary. "
    "Each head is replaced for baseline and all top16 actions in each frozen seed; ensemble scores and seed SD "
    "are recomputed. Labels are used only after all choices are frozen. No fitting, labels or downloads added. "
    "Seed SD is uncalibrated; component products are surrogate scores. Local PDM, not EPDMS. Fresh holdout unopened.")


def pdm(c):
    if c.shape[-1] != 8:
        raise ValueError("Eight components required")
    return (c[..., 0] * c[..., 1] * c[..., 2] * c[..., 3]) * (
        5 * c[..., 4] + 5 * c[..., 5] + 2 * c[..., 6] + 2 * c[..., 7]) / 14


def verify_source(root, bundle):
    identity, report = common.read(root / "identity.json"), common.read(root / "report.json")
    if (report["status"] != "complete" or report["fresh_holdout_processed"] != 0
            or report["identity_sha256"] != common.sha(root / "identity.json")
            or report["formula_audit"]["status"] != "passed" or report["formula_audit"]["mismatches"] != 0):
        raise ValueError("Completed component study with accepted raw formula required")
    for name, digest in report["artifacts"].items():
        if common.sha(root / name) != digest:
            raise RuntimeError("Frozen component artifact changed: " + name)
    for name, digest in identity["sources"].items():
        if common.sha(common.REPO / name) != digest:
            raise RuntimeError("Frozen component source changed: " + name)
    for name, digest in identity["inputs"].items():
        if common.sha(Path(name)) != digest:
            raise RuntimeError("Frozen component input changed: " + name)
    for name in ("identity.json", "bundle_manifest.json"):
        if identity["inputs"].get(str(bundle / name)) != common.sha(bundle / name):
            raise ValueError("Component study belongs to another bundle")
    if "mlp" not in identity["variants"] or len(set(identity["seeds"])) != len(identity["seeds"]) or len(identity["seeds"]) < 2:
        raise ValueError("Frozen MLP with at least two unique seeds required")
    return identity, report


def teacher_probabilities(examples, source_identity):
    # Only this existing-library sigmoid needs Torch. No networks are loaded.
    import torch
    if (source_identity["torch"] != torch.__version__ or source_identity["numpy"] != np.__version__
            or source_identity["python"] != platform.python_version()):
        raise ValueError("Use the component study's Python/NumPy/Torch runtime for exact replay")
    torch.set_num_threads(source_identity["threads"])
    logits = np.stack([np.column_stack([e["sample"]["head_" + m][e["label_indices"]] for m in METRICS]) for e in examples])
    with torch.no_grad():
        return torch.tensor(logits, dtype=torch.float32).sigmoid().numpy()


def combined(seed_components, baselines, cached_gains=None):
    if seed_components.ndim != 4 or seed_components.shape[2:] != (16, 8):
        raise ValueError("Expected seeds x scenes x top16 x metrics")
    if not np.isfinite(seed_components).all() or np.any((seed_components < 0) | (seed_components > 1)):
        raise ValueError("Invalid cached components")
    components = seed_components.astype(float).mean(0)
    score = pdm(components)
    gain = score - score[np.arange(len(baselines)), baselines, None]
    if cached_gains is None:
        # Same float32 formula arithmetic as the cached per-seed predictions.
        per_seed_score = pdm(seed_components)
        cached_gains = per_seed_score - per_seed_score[:, np.arange(len(baselines)), baselines, None]
    if cached_gains.shape != seed_components.shape[:3] or not np.isfinite(cached_gains).all():
        raise ValueError("Invalid per-seed gains")
    lower = gain - cached_gains.astype(float).std(0)
    if np.any(gain[np.arange(len(baselines)), baselines] != 0) or np.any(lower[np.arange(len(baselines)), baselines] != 0):
        raise ValueError("Nonzero baseline gain")
    return dict(components=components, score=score, gain=gain, gain_lower=lower)


def substitute(seed_components, teacher, heads):
    if teacher.shape != seed_components.shape[1:]:
        raise ValueError("Teacher/student component alignment mismatch")
    if not np.isfinite(teacher).all() or np.any((teacher < 0) | (teacher > 1)):
        raise ValueError("Invalid teacher components")
    result = seed_components.copy()
    for j in heads:
        result[..., j] = teacher[None, ..., j]
    return result


def choose(gain, lower, ids, baseline, policy):
    if policy not in POLICIES or not np.isfinite(gain).all() or not np.isfinite(lower).all():
        raise ValueError("Invalid head-ablation policy/prediction")
    eligible = (lower if policy == "gain_m020" else gain) >= (.02 if policy == "gain_m020" else 0.)
    eligible[baseline] = True
    scores = np.where(eligible, gain, -np.inf)
    tied = np.flatnonzero(scores >= scores.max() - 1e-7)
    return baseline if baseline in tied else int(tied[np.argmin(ids[tied])])


def load_replicas(root, identity, report, examples):
    tokens = [e["metadata"]["token"] for e in examples]
    positions = {t: i for i, t in enumerate(tokens)}
    sessions = {e["metadata"]["session_id"] for e in examples}
    token_session = {e["metadata"]["token"]: e["metadata"]["session_id"] for e in examples}
    baseline = np.array([int(np.flatnonzero(e["label_indices"] == int(e["sample"]["selected_local"]))[0]) for e in examples])
    artifact_names = {n.replace("\\", "/") for n in report["artifacts"]}
    seeds, gains = [], []
    for seed in identity["seeds"]:
        arrays = np.empty((len(examples), 16, 8), dtype=np.float32)
        gain = np.empty((len(examples), 16), dtype=np.float32)
        covered, seen_sessions = set(), set()
        names = sorted(n for n in artifact_names if n.startswith(f"cv/mlp_s{seed}_") and n.endswith(".json"))
        for name in names:
            meta = common.read(root / name)
            binding, partition = meta["binding"], meta["binding"]["partition"]
            sid = partition["valid_session"]
            valid_tokens = partition["valid_tokens"]
            expected_valid = {t for t in tokens if token_session[t] == sid}
            npz = (root / name).with_suffix(".npz")
            if (binding["identity_sha256"] != common.sha(root / "identity.json") or binding["variant"] != "mlp"
                    or binding["seed"] != seed or sid in seen_sessions or sid not in sessions
                    or not expected_valid or len(valid_tokens) != len(expected_valid) or set(valid_tokens) != expected_valid
                    or set(partition["train_sessions"]) != sessions - {sid}
                    or len(partition["train_tokens"]) != len(tokens) - len(valid_tokens)
                    or set(partition["train_tokens"]) != set(tokens) - expected_valid
                    or str(npz.relative_to(root)).replace("\\", "/") not in artifact_names
                    or meta["sha256"] != common.sha(npz)):
                raise ValueError("Frozen whole-session fold binding mismatch")
            with np.load(npz, allow_pickle=False) as data:
                c, score, g = [data[k].copy() for k in ("components", "score", "gain")]
            if (c.shape != (len(valid_tokens), 16, 8) or score.shape != (len(valid_tokens), 16)
                    or g.shape != score.shape or not np.isfinite(c).all() or not np.isfinite(score).all()
                    or not np.isfinite(g).all() or np.any((c < 0) | (c > 1))
                    or not np.allclose(pdm(c), score, atol=1e-6, rtol=0)):
                raise ValueError("Invalid fold component predictions")
            for j, token in enumerate(valid_tokens):
                i = positions[token]
                if token in covered or g[j, baseline[i]] != 0 or not np.allclose(score[j] - score[j, baseline[i]], g[j], atol=1e-6, rtol=0):
                    raise ValueError("OOF baseline/coverage mismatch")
                arrays[i], gain[i] = c[j], g[j]
                covered.add(token)
            seen_sessions.add(sid)
        if covered != set(tokens) or seen_sessions != sessions:
            raise ValueError("Missing whole-session OOF predictions")
        seeds.append(arrays)
        gains.append(gain)
    original = combined(np.stack(seeds), baseline, np.stack(gains))
    with np.load(root / "mlp_oof_ensemble.npz", allow_pickle=False) as data:
        order = {str(t): i for i, t in enumerate(data["tokens"])}
        if len(data["tokens"]) != len(tokens) or set(order) != set(tokens):
            raise ValueError("Frozen ensemble token coverage mismatch")
        ix = [order[t] for t in tokens]
        for key in original:
            if not np.array_equal(original[key], data[key][ix]):
                raise ValueError("Exact frozen ensemble reconstruction failed: " + key)
    return np.stack(seeds), np.stack(gains), baseline


def freeze_decisions(examples, seeds, cached_gains, teacher, baselines, source_frame, out):
    frame = source_frame.set_index(["method", "token"])
    if frame.index.has_duplicates:
        raise ValueError("Duplicate frozen source decisions")
    decisions = []
    for name, heads in SUBSTITUTIONS.items():
        arrays = substitute(seeds, teacher, heads)
        pred = combined(arrays, baselines, cached_gains if name == "student" else None)
        for i, e in enumerate(examples):
            token, ix = e["metadata"]["token"], e["label_indices"]
            ids = e["sample"]["candidate_ids"][ix]
            for policy in POLICIES:
                chosen = choose(pred["gain"][i], pred["gain_lower"][i], ids, baselines[i], policy)
                row = dict(method=name + "/" + policy, token=token, candidate_local=int(ix[chosen]),
                    candidate_id=int(ids[chosen]), predicted_gain_pp=100 * float(pred["gain"][i, chosen]),
                    gain_lower_pp=100 * float(pred["gain_lower"][i, chosen]))
                row.update({"predicted_delta_" + m: float(pred["components"][i, chosen, j] - pred["components"][i, baselines[i], j]) for j, m in enumerate(METRICS)})
                if name in ("student", "teacher"):
                    previous_method = ("mlp" if name == "student" else "teacher_formula") + "/ensemble/" + policy
                    previous = frame.loc[(previous_method, token)]
                    if (int(previous.candidate_local) != row["candidate_local"] or int(previous.candidate_id) != row["candidate_id"]
                            or abs(previous.predicted_gain_pp - row["predicted_gain_pp"]) > 1e-8
                            or abs(previous.gain_lower_pp - row["gain_lower_pp"]) > 1e-8):
                        raise ValueError("Frozen student/teacher decision replay failed")
                decisions.append(row)
    marker = out / "decisions.json"
    if marker.exists() and common.read(marker) != decisions:
        raise RuntimeError("Frozen head-ablation decisions changed")
    common.write(marker, decisions)
    return decisions


def evaluate(examples, decisions, source_frame):
    lookup = {e["metadata"]["token"]: e for e in examples}
    original = source_frame.set_index(["method", "token"])
    rows = []
    for decision in decisions:
        e = lookup[decision["token"]]
        sample = dict(e["sample"], token=e["metadata"]["token"], session_id=e["metadata"]["session_id"])
        outcomes = {str(int(local)): {v: dict(zip(common.bundles.LABEL_KEYS, e["label_" + v][j].tolist()))
                    for v in ("raw", "filtered")} for j, local in enumerate(e["label_indices"])}
        row = {**decision, "oracle_score": float(e["label_raw"][:, 0].max()),
               **common.record(sample, outcomes, decision["candidate_local"])}
        head, policy = decision["method"].split("/")
        if head in ("student", "teacher"):
            method = ("mlp" if head == "student" else "teacher_formula") + "/ensemble/" + policy
            previous = original.loc[(method, decision["token"])]
            keys = ("raw_score", "raw_gain_pp", "filtered_score", "filtered_gain_pp") + tuple(v + "_delta_" + m for v in ("raw", "filtered") for m in METRICS)
            if (any(not np.isfinite(previous[k]) or abs(previous[k] - row[k]) > 1e-8 for k in keys)
                    or any(previous[v + "_harm"] not in (True, False) or bool(previous[v + "_harm"]) != row[v + "_harm"] for v in ("raw", "filtered"))):
                raise ValueError("Frozen raw/filtered outcome replay failed")
        rows.append(row)
    return rows


def summarize(rows, seed):
    stats, result = common.summary(rows, seed), []
    frame = pd.DataFrame(rows)
    for method, group in frame.groupby("method", sort=True):
        policy = method.split("/")[1]
        source = frame[frame.method == "student/" + policy]
        paired = group.merge(source[["token", "session_id", "raw_gain_pp", "raw_harm"]], on=["token", "session_id"],
            suffixes=("", "_source"), validate="one_to_one")
        if len(paired) != len(group) or len(paired) != len(source):
            raise ValueError("Paired head-ablation coverage mismatch")
        difference = paired.assign(delta=paired.raw_gain_pp - paired.raw_gain_pp_source).groupby("session_id").delta.mean().to_numpy()
        draws = np.random.default_rng(seed).integers(0, len(difference), size=(2000, len(difference)))
        v = stats[method]
        result.append(dict(method=method, gain_macro_pp=v["gain_macro_pp"], gain_micro_pp=v["gain_micro_pp"],
            gain_ci95_sessions=v["gain_ci95_sessions"], interventions=v["interventions"], harm=v["harm"],
            quality_degraded=v["quality_degraded"], worst_gain_pp=float(min(0, group.raw_gain_pp.min())),
            source_harm_scenes_still_harm=int((paired.raw_harm & paired.raw_harm_source).sum()),
            source_harm_scenes_no_longer_harm=int((~paired.raw_harm & paired.raw_harm_source).sum()),
            new_harm_scenes=int((paired.raw_harm & ~paired.raw_harm_source).sum()),
            paired_delta_macro_pp=float(difference.mean()),
            paired_delta_ci95_sessions=np.quantile(difference[draws].mean(1), [.025, .975]).tolist()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory"
    p.add_argument("--source", type=Path, default=base / "experiments/component_critic_v1")
    p.add_argument("--bundle", type=Path, default=base / "selector_training_bundle_v3_30gb")
    p.add_argument("--out", type=Path, default=base / "experiments/component_heads_audit_v1")
    a = p.parse_args()
    source, bundle, out = (x.resolve() for x in (a.source, a.bundle, a.out))
    for root in (source, bundle):
        if out == root or root in out.parents or out in root.parents:
            raise ValueError("Separate head-audit output required")
    source_id, source_report = verify_source(source, bundle)
    examples, _ = common.bundles.load_bundle(bundle)
    tokens = {e["metadata"]["token"] for e in examples}
    if (len(examples) != source_report["counts"]["scenes"] or len(tokens) != len(examples)
            or len({e["metadata"]["session_id"] for e in examples}) != source_report["counts"]["sessions"]):
        raise ValueError("Source/bundle scene count mismatch")
    for e in examples:
        if e["metadata"]["role"] != "train" or not np.array_equal(e["label_indices"], np.argsort(-e["sample"]["selection_scores"], kind="stable")[:16]):
            raise ValueError("Train-only frozen top16 required")
    identity = dict(inputs={str(path): common.sha(path) for path in (source / "report.json", bundle / "bundle_manifest.json")},
        source_sha256=common.sha(Path(__file__)), helper_sha256=common.sha(Path(common.__file__)),
        substitutions={k: list(v) for k, v in SUBSTITUTIONS.items()}, policies=list(POLICIES), note=NOTE)
    marker = out / "identity.json"
    if marker.exists() and common.read(marker) != identity:
        raise RuntimeError("Head-audit inputs/code changed; use a new --out root")
    if out.exists() and not marker.exists() and any(out.iterdir()):
        raise ValueError("Nonempty head-audit output without identity")
    if (out / "report.json").exists():
        report = common.read(out / "report.json")
        if report["identity_sha256"] != common.sha(marker):
            raise RuntimeError("Head-audit identity changed")
        for name, digest in report["artifacts"].items():
            if common.sha(out / name) != digest:
                raise RuntimeError("Head-audit artifact changed: " + name)
        print(json.dumps(report, indent=2), flush=True)
        return
    replicas, cached_gains, baselines = load_replicas(source, source_id, source_report, examples)
    teacher = teacher_probabilities(examples, source_id)
    source_frame = pd.read_csv(source / "oof_per_scene.csv", dtype={"token": str}, low_memory=False, float_precision="round_trip")
    common.write(marker, identity)
    decisions = freeze_decisions(examples, replicas, cached_gains, teacher, baselines, source_frame, out)
    rows = evaluate(examples, decisions, source_frame)
    frame = pd.DataFrame(rows)
    source_harm = frame[(frame.method == "student/gain_m020") & frame.raw_harm]
    bad_tokens = set(source_harm.token)
    artifacts = {"per_scene.csv": frame, "source_harm_cases.csv": frame[frame.token.isin(bad_tokens)]}
    for name, content in artifacts.items():
        temp = out / (name + ".tmp")
        content.to_csv(temp, index=False)
        temp.replace(out / name)
    report = dict(status="complete", scenes=len(examples), sessions=source_report["counts"]["sessions"],
        frozen_primary=source_report["primary"], selection_performed=False,
        source_harm_tokens=sorted(bad_tokens), rows=summarize(rows, source_id["seeds"][0]),
        checks=dict(source_hashes_verified=True, whole_session_seed_predictions_verified=True,
                    exact_student_ensemble_reconstructed=True, student_and_teacher_choices_replayed=True,
                    raw_filtered_outcomes_replayed=True, decisions_frozen_before_evaluation=True),
        identity_sha256=common.sha(marker),
        artifacts={name: common.sha(out / name) for name in (*artifacts, "decisions.json")},
        fresh_holdout_processed=0, new_training_runs=0, new_scorer_calls=0, additional_download_GB=0, note=NOTE)
    common.write(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
