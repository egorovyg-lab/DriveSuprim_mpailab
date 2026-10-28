"""Prepare an archive-budgeted pilot from metadata, never from quality labels.

Creates a NEW compatible frozen run. Existing pilot splits/caches are untouched.
Only small public manifests are fetched; this command downloads no image archive.
"""
import argparse
import copy
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim.agents.drivemem import ROLES, session_id, spaced_records, split_sessions, token_seed
from plan_drivemem_images import BASE, MAPPING, MAPPING_SHA256, archive_entry, collect_requests, fetch_bytes, save_json, sha


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def select_archives(coverage, entries, required, budget):
    """Deterministic multistart greedy set cover; cost is full compressed bytes.

    This is a heuristic, not a proof of the globally cheapest feasible subset.
    Failure never silently increases the cap or reduces the session requirement.
    """
    candidates = sorted(k for k in coverage if coverage[k] and entries[k]["size"] <= budget)
    successes = []
    for first in [None] + candidates:
        chosen = [] if first is None else [first]
        covered = set() if first is None else set(coverage[first])
        cost = 0 if first is None else entries[first]["size"]
        while len(covered) < required:
            options = [k for k in candidates if k not in chosen and
                       cost + entries[k]["size"] <= budget and coverage[k] - covered]
            if not options:
                break
            key = min(options, key=lambda k: (entries[k]["size"] / len(coverage[k] - covered),
                                               entries[k]["size"], k))
            chosen.append(key)
            covered.update(coverage[key])
            cost += entries[key]["size"]
        if len(covered) >= required:
            successes.append((cost, len(chosen), sorted(chosen)))
    if not successes:
        raise ValueError(f"No subset found for {required} sessions within {budget / 1e9:g} GB. "
                         "Use a new root with fewer sessions or a different budget; no image archives downloaded.")
    return min(successes)[2]


def make_selection(rows, mapping, entries, counts, count, gap, seed, budget):
    reverse = {}
    for name, logs in mapping.items():
        match = re.fullmatch(r"openscene_sensor_trainval_(\d+)", name)
        if match is None or not isinstance(logs, list):
            raise ValueError("Unexpected shard mapping schema")
        for log in logs:
            if log in reverse:
                raise ValueError("Ambiguous shard mapping")
            reverse[log] = int(match[1])
    by_log = {}
    for row in rows:
        if session_id(row["log_name"]) != row["session_id"]:
            raise ValueError("Inventory session identity does not match log name")
        if row["log_name"] in reverse:
            by_log.setdefault(row["log_name"], []).append(row)
    viable, coverage = {}, {}
    for log, group in sorted(by_log.items()):
        shard = reverse[log]
        if shard not in entries:
            continue
        try:
            spaced_records(group, count, gap, token_seed(log, seed))
        except ValueError:
            continue
        sid = group[0]["session_id"]
        viable[log] = group
        coverage.setdefault(shard, set()).add(sid)
    chosen = select_archives(coverage, entries, sum(counts.values()), budget)
    pool = [row for log, group in sorted(viable.items()) if reverse[log] in chosen for row in group]
    allocation = split_sessions(pool, counts, seed)
    selected = []
    for role in ROLES:
        for sid in allocation[role]:
            # One segment per selected session keeps every chosen scene in one shard.
            logs = [log for log in viable if reverse[log] in chosen and viable[log][0]["session_id"] == sid]
            log = min(logs, key=lambda x: (token_seed(x, seed), x))
            selected.extend(dict(row, role=role) for row in
                            spaced_records(viable[log], count, gap, token_seed(sid, seed)))
    used = sorted({reverse[r["log_name"]] for r in selected})
    return allocation, selected, pool, [dict(entries[k], logs=sorted({r["log_name"] for r in selected
                                                                    if reverse[r["log_name"]] == k})) for k in used]


def prepare(source, destination, mapping, entries, counts, count, budget):
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("New run must be separate from the source run")
    frozen = read_json(source / "split_identity.json")
    for key, filename in (("splits", "splits.json"), ("inventory", "inventory.csv")):
        if frozen[key] != sha(source / filename):
            raise RuntimeError("Source frozen inventory/splits changed")
    template = read_json(source / "identity.json")
    repo = Path(__file__).resolve().parents[1]
    for name, digest in template["sources"].items():
        if sha(repo / name) != digest:
            raise RuntimeError(f"Source pilot code identity changed: {name}")
    old_split = read_json(source / "splits.json")
    excluded = set(old_split["navtest_sessions_excluded"])
    with (source / "inventory.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["timestamp_us"] = int(row["timestamp_us"])
        if row["session_id"] in excluded:
            raise RuntimeError("Source inventory contains excluded navtest session")
    if count < 1 or budget <= 0 or any(counts[r] < 1 for r in ROLES):
        raise ValueError("Budget and all role/scene counts must be positive")
    settings = template["settings"]
    spec = {"source_identity_sha256": sha(source / "identity.json"), "source_split_identity": frozen,
            "budget_bytes": budget, "sessions": counts, "scenes_per_session": count,
            "seed": settings["seed"], "mapping_sha256": MAPPING_SHA256}
    if destination.exists():
        marker = destination / "budget_plan.json"
        if marker.exists() and read_json(marker).get("spec") == spec:
            frozen_new = read_json(destination / "split_identity.json")
            if frozen_new != {"splits": sha(destination / "splits.json"), "inventory": sha(destination / "inventory.csv")}:
                raise RuntimeError("Budget run split identity changed")
            print(json.dumps(read_json(marker), indent=2), flush=True)
            return
        raise FileExistsError(f"Destination exists; use a new root: {destination}")
    allocation, selected, pool, archives = make_selection(
        rows, mapping, entries, counts, count, float(settings["gap_s"]), int(settings["seed"]), budget)
    # Check source metadata and collect exact camera requests before publishing files.
    log_hashes = {log: sha(Path(template["navsim_log_path"]) / (log + ".pkl"))
                  for log in sorted({r["log_name"] for r in selected})}
    new_identity = copy.deepcopy(template)
    new_identity["settings"].update(root=str(destination), sessions=counts, scenes={r: count for r in ROLES})
    destination.mkdir(parents=True)
    save_json(destination / "identity.json", new_identity)
    with (destination / "inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(pool)
    save_json(destination / "splits.json", {"allocation": allocation, "records": selected,
        "log_sha256": log_hashes, "navtest_sessions_excluded": sorted(excluded),
        "purpose": "archive-budgeted technical pilot; selected by metadata/cost, no quality outcomes used"})
    save_json(destination / "split_identity.json", {"splits": sha(destination / "splits.json"),
                                                    "inventory": sha(destination / "inventory.csv")})
    requests = collect_requests(destination)
    out = destination / "image_preparation"
    out.mkdir()
    save_json(out / "requests.json", requests)
    total = sum(a["size"] for a in archives)
    save_json(out / "archive_plan.json", {"requests_sha256": sha(out / "requests.json"),
        "routes": {"camera_log_shards": {"archives": archives, "size_upper_bound": total,
                    "description": "Archive-budgeted pilot; only exact requested camera files are extracted"}},
        "suggested_route": "camera_log_shards", "mapping_sha256": MAPPING_SHA256})
    report = {"status": "planned", "spec": spec, "root": str(destination),
        "archive_count": len(archives), "compressed_GB": total / 1e9, "limit_GB": budget / 1e9,
        "largest_archive_GB": max(a["size"] for a in archives) / 1e9,
        "selected_scenes": len(selected), "selected_sessions": sum(counts.values()),
        "required_images": len(requests["images"]),
        "roles": {r: {"sessions": counts[r], "scenes": counts[r] * count} for r in ROLES},
        "archive_plan_sha256": sha(out / "archive_plan.json"),
        "preparer_sha256": sha(Path(__file__)),
        "note": "No image archive downloaded. Budget covers unique compressed archives, not protocol overhead/retries. "
                "Selection favors archive availability/cost; this is a technical pilot, not representative benchmark evidence."}
    save_json(destination / "inventory_report.json", report)
    save_json(destination / "budget_plan.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-gb", type=float, default=20)
    parser.add_argument("--sessions", type=int, nargs=5, default=[6, 4, 3, 2, 2], metavar="N")
    parser.add_argument("--scenes-per-session", type=int, default=8)
    args = parser.parse_args()
    raw = fetch_bytes(BASE + "/resolve/main/" + MAPPING + "?download=1&cachebust=" + str(int(time.time())))
    if hashlib.sha256(raw).hexdigest() != MAPPING_SHA256:
        raise RuntimeError("Official mapping changed")
    api = "https://huggingface.co/api/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_trainval_camera?limit=1000"
    entries = {}
    for entry in json.loads(fetch_bytes(api)):
        match = re.fullmatch(r"openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_(\d+)\.tgz", entry.get("path", ""))
        if match:
            entries[int(match[1])] = archive_entry(entry)
    if set(entries) != set(range(200)):
        raise RuntimeError("Incomplete official camera archive listing")
    prepare(args.source, args.destination, json.loads(raw), entries, dict(zip(ROLES, args.sessions)),
            args.scenes_per_session, int(args.max_gb * 1e9))


if __name__ == "__main__":
    main()
