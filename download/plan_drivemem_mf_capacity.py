"""Read-only capacity/acquisition diagnosis. Fetches JSON metadata, never images."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from prepare_drivemem_mf_expansion import read, spaced_capacity
from plan_drivemem_images import BASE, MAPPING, MAPPING_SHA256, archive_entry, fetch_bytes, sha


def eligible_rows(rows, old, roles, gap=10.):
    past = {sid: [int(r["timestamp_us"]) for r in old if r["session_id"] == sid] for sid in roles}
    used = {r["token"] for r in old}
    return [dict(r, timestamp_us=int(r["timestamp_us"])) for r in rows
            if r["session_id"] in roles and r["token"] not in used
            and all(abs(int(r["timestamp_us"]) - t) >= round(gap * 1e6) for t in past[r["session_id"]])]


def capacity(rows, roles, logs):
    return {sid: spaced_capacity([r["timestamp_us"] for r in rows if r["session_id"] == sid and r["log_name"] in logs])
            for sid in roles}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--archive-dir", type=Path, required=True)
    p.add_argument("--max-extra-gb", type=float, default=5.)
    a = p.parse_args()
    if a.max_extra_gb <= 0:
        raise ValueError("Positive planning budget required")
    for root in (a.source, a.inventory, a.validation):
        if read(root / "split_identity.json") != dict(splits=sha(root / "splits.json"), inventory=sha(root / "inventory.csv")):
            raise RuntimeError("Frozen inventory/splits changed")
    identity = read(a.source / "identity.json")
    source = read(a.source / "splits.json")
    roles = {r["session_id"]: r["role"] for r in source["records"] if r["role"] in ("memory", "fit")}
    old = source["records"] + read(a.inventory / "splits.json")["records"] + read(a.validation / "splits.json")["records"]
    with (a.inventory / "inventory.csv").open(encoding="utf-8", newline="") as f:
        rows = eligible_rows(list(csv.DictReader(f)), old, roles)
    installed = {p.stem for p in Path(identity["navsim_log_path"]).glob("*.pkl")}
    rows = [r for r in rows if r["log_name"] in installed]
    raw = fetch_bytes(BASE + "/resolve/main/" + MAPPING + "?download=1")
    if hashlib.sha256(raw).hexdigest() != MAPPING_SHA256:
        raise RuntimeError("Official mapping changed")
    mapping = {int(re.fullmatch(r"openscene_sensor_trainval_(\d+)", k)[1]): set(v) for k, v in json.loads(raw).items()}
    entries = {}
    url = "https://huggingface.co/api/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_trainval_camera?limit=1000"
    for e in json.loads(fetch_bytes(url)):
        match = re.fullmatch(r"openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_(\d+)\.tgz", e.get("path", ""))
        if match:
            entries[int(match[1])] = archive_entry(e)
    cached = {i for i, e in entries.items() if (a.archive_dir / Path(e["path"]).name).is_file()
              and (a.archive_dir / Path(e["path"]).name).stat().st_size == e["size"]}
    logs = set().union(*(mapping[i] for i in cached)) if cached else set()
    current = capacity(rows, roles, logs)
    global_caps = capacity(rows, roles, installed)
    def gain(caps, previous):
        return sum(min(16, caps[s]) - min(16, previous[s]) for s in roles)
    options = []
    for i, e in entries.items():
        if i in cached or i not in mapping or not any(r["log_name"] in mapping[i] for r in rows):
            continue
        caps = capacity(rows, roles, logs | mapping[i])
        extra = gain(caps, current)
        if extra > 0:
            options.append(dict(shard=i, additional_GB=e["size"] / 1e9, added_screening_slots=extra,
                newly_complete_sessions=[s for s in roles if current[s] < 16 <= caps[s]], capacity=caps))
    options.sort(key=lambda o: (o["additional_GB"] / o["added_screening_slots"], o["shard"]))
    selected, spent, coverage, caps = [], 0, set(logs), current
    # A bounded heuristic, not a proof of cheapest set cover. Recompute gains
    # after each archive so overlaps never count twice.
    while any(v < 16 for v in caps.values()):
        trials = []
        for o in options:
            i = o["shard"]
            if i in selected or spent + o["additional_GB"] > a.max_extra_gb:
                continue
            after = capacity(rows, roles, coverage | mapping[i])
            extra = gain(after, caps)
            if extra:
                trials.append((o["additional_GB"] / extra, i, after))
        if not trials:
            break
        _, i, caps = min(trials, key=lambda t: (t[0], t[1]))
        selected.append(i)
        spent += entries[i]["size"] / 1e9
        coverage.update(mapping[i])
    report = dict(status="capacity_only", no_outputs_created=True, image_download_GB=0,
        gap_s=10, requested_per_session=16, cached_shards=sorted(cached),
        cache_validation="Size only for planning; require SHA256 verification before any extraction",
        sessions=[dict(session_id=s, role=roles[s], cached_capacity=current[s],
                       all_installed_logs_capacity=global_caps[s], capped_shortfall=max(0, 16 - current[s])) for s in sorted(roles)],
        total_cached_screening_slots=sum(min(16, v) for v in current.values()),
        impossible_even_with_more_archives=[s for s in roles if global_caps[s] < 16],
        single_archive_options=options[:8],
        bounded_greedy_plan=dict(planning_budget_GB=a.max_extra_gb, shards=selected, compressed_GB=spent,
            capacity=caps, fulfills_original_quota=all(v >= 16 for v in caps.values())),
        note="No images fetched and no protocol changed. Options concern existing M/F sessions, not fresh validation. "
             "Greedy plan is not globally optimal. Do not sum single-option gains: sessions/logs overlap.")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
