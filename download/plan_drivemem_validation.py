"""Inventory NEW validation sessions in already planned/cached camera archives.

Prints metadata-only options. No scenes/images are selected or downloaded, no
outcomes are read and no frozen experiment files are changed. Public JSON
manifests are small; image archives are never fetched by this command.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim.agents.drivemem import session_id, spaced_records, token_seed
from plan_drivemem_images import BASE, MAPPING, MAPPING_SHA256, fetch_bytes, archive_entry, sha


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inventory_candidates(rows, mapping, excluded, count, gap, seed):
    reverse = {}
    for key, logs in mapping.items():
        match = re.fullmatch(r"openscene_sensor_trainval_(\d+)", key)
        if not match or not isinstance(logs, list):
            raise ValueError("Unexpected mapping schema")
        for log in logs:
            if log in reverse:
                raise ValueError("Ambiguous mapping")
            reverse[log] = int(match[1])
    groups = {}
    for row in rows:
        if session_id(row["log_name"]) != row["session_id"]:
            raise ValueError("Session identity mismatch")
        if row["session_id"] in excluded or row["log_name"] not in reverse:
            continue
        groups.setdefault(row["log_name"], []).append(row)
    result = {}
    for log, records in sorted(groups.items()):
        try:
            spaced_records(records, count, gap, token_seed(log, seed))
        except ValueError:
            continue
        result.setdefault(reverse[log], {}).setdefault(records[0]["session_id"], []).append(log)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--inventory-run", type=Path)
    parser.add_argument("--exclude-run", type=Path, action="append", default=[])
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--scenes-per-session", type=int, default=4)
    parser.add_argument("--target-sessions", type=int, default=12)
    args = parser.parse_args()
    if args.scenes_per_session < 1 or args.target_sessions < 1:
        raise ValueError("Counts must be positive")
    source = args.source
    original = args.inventory_run or source.parent / "pilot_v1"
    identity = read(source / "identity.json")
    split = read(source / "splits.json")
    for root in (source, original):
        frozen = read(root / "split_identity.json")
        if frozen != {"splits": sha(root / "splits.json"), "inventory": sha(root / "inventory.csv")}:
            raise RuntimeError(f"Frozen inventory/splits changed: {root}")
    excluded = {r["session_id"] for r in split["records"]}
    original_split = read(original / "splits.json")
    reserved = {r["session_id"] for r in original_split["records"] if r["role"] in ("calibration", "test")}
    excluded.update(reserved)
    excluded.update(original_split["navtest_sessions_excluded"])
    for root in args.exclude_run:
        excluded.update(r["session_id"] for r in read(root / "splits.json")["records"])
    with (original / "inventory.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["timestamp_us"] = int(row["timestamp_us"])
    installed_logs = {p.stem for p in Path(identity["navsim_log_path"]).glob("*.pkl")}
    absent_logs = sorted({r["log_name"] for r in rows} - installed_logs)
    rows = [r for r in rows if r["log_name"] in installed_logs]
    raw = fetch_bytes(BASE + "/resolve/main/" + MAPPING + "?download=1&cachebust=" + str(int(time.time())))
    if hashlib.sha256(raw).hexdigest() != MAPPING_SHA256:
        raise RuntimeError("Official mapping changed")
    available = inventory_candidates(rows, json.loads(raw), excluded, args.scenes_per_session, 10., 20261027)
    plan_path = source / "image_preparation/archive_plan.json"
    budget = read(source / "budget_plan.json")
    if sha(plan_path) != budget["archive_plan_sha256"]:
        raise RuntimeError("Frozen camera archive plan changed")
    archives = read(plan_path)["routes"]["camera_log_shards"]["archives"]
    data_root = Path(identity["original_sensor_path"]).parents[1]
    cache = args.archive_dir or Path(os.environ.get("DRIVEMEM_IMAGE_ARCHIVE_DIR", str(data_root / ".downloads/drivemem_camera")))
    cached_ids, planned_ids, archive_reports = set(), set(), []
    for entry in archives:
        index = int(re.search(r"_(\d+)\.tgz$", entry["path"])[1])
        planned_ids.add(index)
        path = cache / Path(entry["path"]).name
        present = path.is_file() and path.stat().st_size == entry["size"]
        if present:
            cached_ids.add(index)
        archive_reports.append(dict(shard=index, archive=str(path), size_GB=entry["size"] / 1e9,
            cached_size_matches=present, new_sessions=len(available.get(index, {}))))
    cached_sessions = sorted({sid for i in cached_ids for sid in available.get(i, {})})
    covered_sessions = sorted({sid for i in planned_ids for sid in available.get(i, {})})
    report = dict(status="metadata_only", source=str(source), inventory_run=str(original),
        missing_metadata_log_files=len(absent_logs),
        excluded_current_sessions=len({r["session_id"] for r in split["records"]}),
        also_preserving_original_calibration_test_sessions=len(reserved),
        target_sessions=args.target_sessions, scenes_per_session=args.scenes_per_session, min_gap_s=10.,
        archives=archive_reports, available_new_sessions_without_image_transfer=len(cached_sessions),
        new_session_ids=cached_sessions, new_sessions_in_original_four_shards=len(covered_sessions),
        target_shortfall=max(0, args.target_sessions - len(cached_sessions)),
        eligible_logs_in_cache={sid: sorted({log for i in cached_ids for log in available.get(i, {}).get(sid, [])}) for sid in cached_sessions},
        note="No image archive downloaded; no split created; B3 remains frozen. Archive presence checked by size only; "
             "SHA256 verification is required before extraction. Existing C/T remain reserved. "
             "Pass --exclude-run for any other session set whose outcomes were already used. "
             "New sessions within the same archives do not remove archive-selection bias.")
    if len(cached_sessions) < args.target_sessions:
        api = "https://huggingface.co/api/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_trainval_camera?limit=1000"
        options = []
        for e in json.loads(fetch_bytes(api)):
            match = re.fullmatch(r"openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_(\d+)\.tgz", e.get("path", ""))
            if not match:
                continue
            i = int(match[1])
            new_sessions = sorted(set(available.get(i, {})) - set(cached_sessions))
            if not new_sessions:
                continue
            entry = archive_entry(e)
            local = cache / Path(entry["path"]).name
            present = local.is_file() and local.stat().st_size == entry["size"]
            cost = 0 if present else entry["size"]
            options.append(dict(shard=i, archive_path=entry["path"], additional_GB_upper_bound=cost / 1e9,
                new_sessions=len(new_sessions), session_ids=new_sessions, cached_size_matches=present))
        options.sort(key=lambda x: (x["additional_GB_upper_bound"] / x["new_sessions"], x["shard"]))
        report["additional_archive_options_not_downloaded"] = options[:5]
        report["options_note"] = "Single-archive options can overlap in sessions; do not add counts blindly. No download is started."
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
