"""Read-only comparison of training sizes on identical session-held-out scenes.

No fitting, scorer calls, image transfers or historical/fresh holdout lookup.
Raw policy isolates the trained scorer at fixed settings. Selected gated policy
also changes the fitted threshold; it is reported separately as a tuning result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = ("control_v1", "consistent_v2")
METHODS = tuple(f + "_" + suffix for f in FAMILIES for suffix in ("raw", "gated"))
SAFETY = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
          "traffic_light_compliance", "time_to_collision_within_bound")
METRICS = SAFETY + ("ego_progress", "lane_keeping", "history_comfort")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def load_metadata(root):
    identity, manifest = read(root / "identity.json"), read(root / "bundle_manifest.json")
    for name, key in (("identity.json", "identity_sha256"), ("examples_index.json", "index_sha256"), ("vocab.npz", "vocab_sha256")):
        if sha(root / name) != manifest[key]:
            raise RuntimeError("Bundle metadata changed: " + str(root / name))
    index = read(root / "examples_index.json")
    if (len(index) != identity["counts"]["scenes"] or manifest["counts"] != identity["counts"]
            or len({e["metadata"]["session_id"] for e in index.values()}) != identity["counts"]["sessions"]):
        raise ValueError("Bundle counts mismatch")
    for token, e in index.items():
        m = e["metadata"]
        if m["token"] != token or m["role"] != "train" or m["session_id"] in identity["protected_sessions"]:
            raise ValueError("Only unprotected training scenes may be compared")
    return identity, index


def verify_shared_examples(old_root, new_root, old_index, new_index):
    if not set(old_index) < set(new_index):
        raise ValueError("Expanded bundle must strictly contain the entire reference bundle")
    for token, old in old_index.items():
        if old["metadata"] != new_index[token]["metadata"]:
            raise ValueError("Shared scene metadata changed: " + token)
        paths = [root / "examples" / (token + ".npz") for root in (old_root, new_root)]
        for path, entry in zip(paths, (old, new_index[token])):
            if sha(path) != entry["sha256"]:
                raise RuntimeError("Bundled example changed: " + str(path))
        with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError("Shared input/label array keys changed: " + token)
            for key in a.files:
                if a[key].dtype != b[key].dtype or not np.array_equal(a[key], b[key]):
                    raise ValueError("Shared input/label values changed: " + token + "/" + key)
    with np.load(old_root / "vocab.npz", allow_pickle=False) as a, np.load(new_root / "vocab.npz", allow_pickle=False) as b:
        if not np.array_equal(a["vocab"], b["vocab"]):
            raise ValueError("Vocabulary changed")


def load_run(root, bundle, index):
    identity, fit = read(root / "identity.json"), read(root / "fit.json")
    if fit["status"] != "complete" or fit["fresh_holdout_processed"] != 0 or fit["config"] != identity["config"]:
        raise ValueError("Complete train-only fit required")
    for name, digest in fit["artifacts"].items():
        if sha(root / name) != digest:
            raise RuntimeError("Frozen fitted artifact changed: " + str(root / name))
    for name in ("identity.json", "bundle_manifest.json"):
        if identity["inputs"].get(str(bundle / name)) != sha(bundle / name):
            raise ValueError("Fit belongs to another bundle")
    frame = pd.read_csv(root / "fit_cv_per_scene.csv", dtype={"token": str, "session_id": str}, float_precision="round_trip")
    if (frame.duplicated(["method", "token"]).any() or set(frame.method) != set(METHODS)
            or any(set(frame.loc[frame.method == m, "token"]) != set(index) for m in METHODS)):
        raise ValueError("Complete unique scene coverage required for every method")
    for key in ("intervention", "raw_harm", "raw_quality_degraded"):
        if not frame[key].isin([True, False]).all():
            raise ValueError("Invalid boolean policy field")
        frame[key] = frame[key].astype(bool)
    numeric = ["raw_score", "raw_gain_pp", "oracle_score"] + ["raw_delta_" + m for m in SAFETY]
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("Nonfinite policy outcomes")
    for row in frame.itertuples():
        m = index[row.token]["metadata"]
        if row.session_id != m["session_id"] or row.origin != m["origin"]:
            raise ValueError("CSV/bundle scene membership mismatch")
    # Check the actual persisted partitions, including new scenes in old sessions.
    ordered = sorted(index, key=lambda t: (index[t]["metadata"]["session_id"], t))
    sessions = sorted({index[t]["metadata"]["session_id"] for t in ordered})
    for family in FAMILIES:
        for sid in sessions:
            fold = root / "cv" / (family + "_" + hashlib.sha256(sid.encode()).hexdigest()[:16])
            frozen = read(fold.with_suffix(".json"))
            expected = dict(train_sessions=[s for s in sessions if s != sid], valid_session=sid,
                train_tokens=[t for t in ordered if index[t]["metadata"]["session_id"] != sid],
                valid_tokens=[t for t in ordered if index[t]["metadata"]["session_id"] == sid])
            if frozen["partition"] != expected or sha(fold.with_suffix(".npz")) != frozen["sha256"]:
                raise ValueError("Whole-session CV partition/artifact mismatch")
    return identity, fit, frame


def stats(frame):
    if frame.empty:
        return dict(scenes=0, sessions=0, methods={})
    methods = {}
    for method, group in frame.groupby("method"):
        methods[method] = dict(gain_macro_pp=float(group.groupby("session_id").raw_gain_pp.mean().mean()),
            gain_micro_pp=float(group.raw_gain_pp.mean()), interventions=int(group.intervention.sum()),
            harm=int(group.raw_harm.sum()), quality_degraded=int(group.raw_quality_degraded.sum()),
            improved=int((group.raw_gain_pp > 1e-6).sum()), worst_loss_pp=float(min(0, group.raw_gain_pp.min())),
            positive_gain_sum_pp=float(group.loc[group.raw_gain_pp > 0, "raw_gain_pp"].sum()),
            negative_gain_sum_pp=float(group.loc[group.raw_gain_pp < 0, "raw_gain_pp"].sum()),
            component_degradation_scenes={m: int((group["raw_delta_" + m] < -1e-8).sum()) for m in SAFETY})
    raw = frame[frame.method == "control_v1_raw"]
    baseline = raw.raw_score - raw.raw_gain_pp / 100
    return dict(scenes=int(raw.token.nunique()), sessions=int(raw.session_id.nunique()),
                baseline_mean=float(baseline.mean()), oracle_reserve_micro_pp=float((100 * (raw.oracle_score - baseline)).mean()),
                methods=methods)


def paired(old, new, seed):
    result = {}
    for method in METHODS:
        a = old[old.method == method].set_index("token").sort_index()
        b = new[new.method == method].set_index("token").sort_index()
        if not a.index.equals(b.index) or not a.session_id.equals(b.session_id) or not a["slice"].equals(b["slice"]):
            raise ValueError("Paired scenes/sessions/sampling slices differ")
        if (not np.allclose(a.raw_score - a.raw_gain_pp / 100, b.raw_score - b.raw_gain_pp / 100, atol=1e-10, rtol=0)
                or not np.allclose(a.oracle_score, b.oracle_score, atol=1e-10, rtol=0)):
            raise ValueError("Shared baseline/oracle labels differ")
        delta = (b.raw_gain_pp - a.raw_gain_pp).groupby(a.session_id).mean()
        picks = np.random.default_rng(seed).integers(0, len(delta), size=(2000, len(delta)))
        result[method] = dict(expanded_minus_reference_macro_pp=float(delta.mean()),
            ci95_sessions=np.quantile(delta.to_numpy()[picks].mean(1), [.025, .975]).tolist(),
            sessions_better=int((delta > 1e-8).sum()), sessions_worse=int((delta < -1e-8).sum()),
            sessions_tied=int((delta.abs() <= 1e-8).sum()), by_session_pp={sid: float(v) for sid, v in delta.items()})
    return result


def compare(reference, expanded, old_bundle, new_bundle):
    old_id, old_index = load_metadata(old_bundle)
    new_id, new_index = load_metadata(new_bundle)
    if new_id["inputs"].get(str(old_bundle / "bundle_manifest.json")) != sha(old_bundle / "bundle_manifest.json"):
        raise ValueError("Expanded bundle must inherit this exact parent")
    verify_shared_examples(old_bundle, new_bundle, old_index, new_index)
    old_run, old_fit, old = load_run(reference, old_bundle, old_index)
    new_run, new_fit, new = load_run(expanded, new_bundle, new_index)
    for key in ("config", "sources", "threads", "python", "numpy", "torch", "platform", "protocol", "thresholds"):
        if old_run[key] != new_run[key]:
            raise ValueError("Matched experiment condition changed: " + key)
    shared = new[new.token.isin(old_index)].copy()
    added = new[~new.token.isin(old_index)].copy()
    old_sessions = {e["metadata"]["session_id"] for e in old_index.values()}
    old_section, new_section = stats(old), stats(shared)
    worst = []
    for method in (f + "_raw" for f in FAMILIES):
        for cohort, group in (("shared", shared), ("added", added)):
            for row in group[(group.method == method) & (group.raw_gain_pp < -1e-8)].nsmallest(3, "raw_gain_pp").to_dict("records"):
                worst.append(dict(cohort=cohort, method=method, token=row["token"], session_id=row["session_id"],
                    candidate_id=int(row["candidate_id"]), gain_pp=float(row["raw_gain_pp"]),
                    predicted_gain_pp=float(row["predicted_gain_pp"]), benefit_estimate=float(row["benefit_estimate"]),
                    component_changes={m: float(row["raw_delta_" + m]) for m in METRICS if abs(row["raw_delta_" + m]) > 1e-8}))
    return dict(status="complete", config=old_fit["config"], reference_counts=old_id["counts"], expanded_counts=new_id["counts"],
        shared_examples_verified=len(old_index), shared=dict(reference=old_section, expanded=new_section,
            paired=paired(old, shared, old_fit["config"]["seed"])),
        selected_gates=dict(reference={f: old_fit["selected"][f]["threshold"] for f in FAMILIES},
                            expanded={f: new_fit["selected"][f]["threshold"] for f in FAMILIES}),
        added_scenes=stats(added), added_new_session_scenes=stats(added[~added.session_id.isin(old_sessions)]),
        added_old_session_scenes=stats(added[added.session_id.isin(old_sessions)]),
        worst_expanded_raw_decisions=worst, fresh_holdout_processed=0, historical_outcomes_read=0,
        note="Read-only training-session OOF comparison. Shared inputs/labels are array-equal and entire-session folds verified. "
             "Raw policies compare training sizes at fixed settings on identical scenes. Selected gated policies also "
             "change fitted thresholds; that comparison includes gate retuning. These are tuning diagnostics, not "
             "independent evidence or calibrated safety. Session bootstrap is descriptive; folds share training data. "
             "Local PDM, not EPDMS. Gain sums describe scenes, not independent samples.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory"
    p.add_argument("--reference", type=Path, default=base / "experiments/trajectory_selector_v2")
    p.add_argument("--expanded", type=Path, default=base / "experiments/trajectory_selector_v3_30gb_e120")
    p.add_argument("--old-bundle", type=Path, default=base / "selector_training_bundle_v2")
    p.add_argument("--new-bundle", type=Path, default=base / "selector_training_bundle_v3_30gb")
    a = p.parse_args()
    result = compare(*(x.resolve() for x in (a.reference, a.expanded, a.old_bundle, a.new_bundle)))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
