"""Read-only metadata capacity and duration-comparison reports for selector v2.

Never opens feature/label arrays, trains, downloads or extracts archives. The
capacity report is conditional on cached archive size; extraction must verify SHA.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "download"))
import prepare_selector_dataset as acquisition

DURATIONS = (120, 360, 1000)
FAMILIES = ("control_v1", "consistent_v2")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def count_capacity(rows, active, existing, cap, gap_s):
    """Rows already exclude protected sessions and times near ALL train anchors."""
    counts, last = {}, {}
    for r in sorted(rows, key=lambda r: (r["session_id"], int(r["timestamp_us"]), r["token"])):
        sid, t = r["session_id"], int(r["timestamp_us"])
        if r["shard"] not in active or counts.get(sid, 0) >= cap - existing.get(sid, 0):
            continue
        if sid not in last or t - last[sid] >= round(gap_s * 1e6):
            counts[sid], last[sid] = counts.get(sid, 0) + 1, t
    return counts


def capacity_section(counts, existing, minimum):
    new = {sid: n for sid, n in counts.items() if sid not in existing and n >= minimum}
    old = {sid: n for sid, n in counts.items() if sid in existing and n > 0}
    extra = sum(old.values()) + sum(new.values())
    return dict(additional_scenes=extra, existing_sessions_extra_scenes=sum(old.values()),
                new_eligible_sessions=len(new), new_session_scenes=sum(new.values()),
                total_train_scenes_if_all_eligible_sessions_used=sum(existing.values()) + extra,
                total_train_sessions_if_all_eligible_sessions_used=len(existing) + len(new),
                extra_scenes_by_existing_session=old, new_eligible_session_scenes=new)


def capacity(data, bundle, archive_dir, cap=128, minimum=16, gap_s=10.):
    if cap < minimum or minimum < 1 or gap_s < 10:
        raise ValueError("Require cap >= minimum >= 1 and gap >= 10 seconds")
    plan, _ = acquisition.verify(data)
    selection = read(data / "selection.json")
    manifest, identity = read(bundle / "bundle_manifest.json"), read(bundle / "identity.json")
    for name, key in (("identity.json", "identity_sha256"), ("examples_index.json", "index_sha256")):
        if sha(bundle / name) != manifest[key]:
            raise RuntimeError("Frozen bundle metadata changed")
    if identity["inputs"].get(str(data / "selection.json")) != sha(data / "selection.json"):
        raise RuntimeError("Acquisition selection does not match training bundle")
    index = read(bundle / "examples_index.json")
    anchors = selection["legacy_training_records"] + [r for r in selection["records"] if r["role"] == "train"]
    if len(anchors) != len(index) or {r["token"] for r in anchors} != set(index):
        raise ValueError("All bundled training scenes must be spacing anchors")
    for r in anchors:
        m = index[r["token"]]["metadata"]
        if m["role"] != "train" or m["session_id"] != r["session_id"]:
            raise ValueError("Bundle/selection membership mismatch")
    existing = Counter(r["session_id"] for r in anchors)
    protected = set(identity["protected_sessions"]) | set(selection["excluded_sessions"]) | set(selection["fresh_holdout_sessions"])
    if set(existing) & protected:
        raise ValueError("Training/protected session overlap")
    # metadata() can fetch missing JSONs. Demand the existing cache first so this
    # audit cannot make any network requests, including manifests.
    if not all((data / "metadata" / name).is_file() for name in ("mapping.json", "archives.json")):
        raise FileNotFoundError("Cached official metadata required; this audit never downloads")
    reverse, entries = acquisition.metadata(data)
    source = read(Path(plan["options"]["source"]) / "identity.json")
    logs = {p.stem for p in Path(source["navsim_log_path"]).glob("*.pkl")}
    inventory = Path(plan["options"]["inventory"]) / "inventory.csv"
    with inventory.open(encoding="utf-8", newline="") as stream:
        rows = acquisition.candidate_rows(list(csv.DictReader(stream)), anchors, protected, logs, reverse, gap_s)
    cached = {i for i, e in entries.items() if (archive_dir / Path(e["path"]).name).is_file()
              and (archive_dir / Path(e["path"]).name).stat().st_size == e["size"]}
    base_counts = count_capacity(rows, cached, existing, cap, gap_s)
    cached_section = capacity_section(base_counts, existing, minimum)
    choices = []
    relevant = {r["shard"] for r in rows}
    for i in sorted(relevant - cached):
        section = capacity_section(count_capacity(rows, cached | {i}, existing, cap, gap_s), existing, minimum)
        extra_scenes = section["additional_scenes"] - cached_section["additional_scenes"]
        new_sessions = section["new_eligible_sessions"] - cached_section["new_eligible_sessions"]
        if extra_scenes > 0:
            choices.append(dict(shard=i, additional_GB=entries[i]["size"] / 1e9,
                                additional_eligible_scenes=extra_scenes, additional_eligible_sessions=new_sessions))
    choices.sort(key=lambda c: (-c["additional_eligible_sessions"] / c["additional_GB"],
                               -c["additional_eligible_scenes"] / c["additional_GB"], c["shard"]))
    return dict(status="metadata_only", current_scenes=len(anchors), current_sessions=len(existing),
        max_scenes_per_session=cap, min_new_session_scenes=minimum, min_gap_s=gap_s,
        protected_sessions=len(protected), fresh_holdout_sessions_preserved=len(selection["fresh_holdout_sessions"]),
        cached_archives=sorted(cached), cached_capacity=cached_section,
        metadata_capacity_all_archives=capacity_section(count_capacity(rows, set(entries), existing, cap, gap_s), existing, minimum),
        additional_archive_options=choices[:15], additional_download_GB=0, features_or_labels_processed=0,
        note="Capacity estimate, not a frozen selection or a representative benchmark. No download/extraction, "
             "features, outcomes or fitting. Cached capacity requires archives still present; size checked, SHA "
             "must be verified before extraction. Already extracted images outside retained archives are not "
             "credited. Additional options overlap: do not add session/scene counts. New eligible sessions are "
             "candidate training sessions; any additional validation reservation reduces training totals.")


def verify_run(root):
    identity, fitted = read(root / "identity.json"), read(root / "fit.json")
    if fitted["status"] != "complete" or fitted["config"] != identity["config"] or fitted["fresh_holdout_processed"] != 0:
        raise RuntimeError("Complete train-only fit required: " + str(root))
    for name, digest in fitted["artifacts"].items():
        if sha(root / name) != digest:
            raise RuntimeError("Fitted artifact changed: " + str(root / name))
    return identity, fitted


def compare(reference, sweep_root):
    rows, canonical = [], None
    for epoch, root in zip(DURATIONS, (reference, sweep_root / "epochs360", sweep_root / "epochs1000")):
        identity, fitted = verify_run(root)
        if fitted["config"]["epochs"] != epoch:
            raise ValueError("Wrong duration at " + str(root))
        comparable = dict(identity)
        comparable["config"] = {k: v for k, v in identity["config"].items() if k != "epochs"}
        if canonical is not None and comparable != canonical:
            raise ValueError("Duration runs must differ only in epoch count")
        canonical = comparable
        for family in FAMILIES:
            raw, gated = (fitted["cv_methods"][family + "_" + suffix] for suffix in ("raw", "gated"))
            rows.append(dict(epochs=epoch, family=family,
                train_loss_final=fitted["models"][family]["history"][-1]["loss"],
                oof_raw_gain_macro_pp=raw["gain_macro_pp"], oof_raw_harm=raw["harm"],
                oof_raw_quality_degraded=raw["quality_degraded"],
                gate=fitted["selected"][family]["threshold"],
                oof_gated_uniform_gain_macro_pp=fitted["selected"][family]["uniform_gain_macro_pp"],
                oof_gated_gain_macro_pp=gated["gain_macro_pp"], oof_gated_harm=gated["harm"],
                oof_gated_quality_degraded=gated["quality_degraded"], interventions=gated["interventions"]))
    return dict(status="complete", durations=list(DURATIONS), rows=rows,
                historical_outcomes_read=0, fresh_holdout_processed=0,
                note="Controlled duration study; optimizer, seed, folds, inputs and model code fixed. "
                     "Gate and any duration selection on OOF are tuning, not independent evaluation. "
                     "Loss magnitudes are comparable within a family only. More epochs can overfit.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "capacity", "compare"))
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory"
    p.add_argument("--data", type=Path, default=base / "selector_data_expansion_v2")
    p.add_argument("--bundle", type=Path, default=base / "selector_training_bundle_v2")
    p.add_argument("--archive-dir", type=Path, default=Path("/workspace/exp_v2/drivemem_data/.downloads/drivemem_camera"))
    p.add_argument("--reference", type=Path, default=base / "experiments/trajectory_selector_v2")
    p.add_argument("--sweep-root", type=Path, default=base / "experiments/selector_duration_v1")
    a = p.parse_args()
    if a.stage == "plan":
        result = dict(durations=list(DURATIONS), reuse_reference=str(a.reference),
            new_training_runs=[dict(epochs=e, root=str(a.sweep_root / ("epochs" + str(e)))) for e in DURATIONS[1:]],
            execution="Sequential CPU-only fit stages; both families; same whole-session folds. "
                      "360/1000 epochs use 3x/8.33x optimizer steps per fold versus 120. "
                      "Historical diagnosis and fresh holdout are not run.", additional_download_GB=0)
    elif a.stage == "capacity":
        result = capacity(a.data.resolve(), a.bundle.resolve(), a.archive_dir.resolve())
    else:
        result = compare(a.reference.resolve(), a.sweep_root.resolve())
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
