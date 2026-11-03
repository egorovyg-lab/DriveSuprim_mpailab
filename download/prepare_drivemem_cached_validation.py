"""Freeze four new sessions using cached archives, no query outcomes or image transfer."""
import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim.agents.drivemem import ROLES, spaced_records, token_seed
from plan_drivemem_validation import inventory_candidates
from plan_drivemem_images import BASE, MAPPING, MAPPING_SHA256, collect_requests, fetch_bytes, save_json, sha


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(source, experiment, inventory, destination, cache):
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Validation root must be separate from source")
    template = read(source / "identity.json")
    for run in (source, inventory):
        if read(run / "split_identity.json") != {"splits": sha(run / "splits.json"), "inventory": sha(run / "inventory.csv")}:
            raise RuntimeError("Frozen source/inventory changed")
    model_identity = read(experiment / "identity.json")
    fit = read(experiment / "fit.json")
    if model_identity["source"] != str(source) or sha(experiment / "ridge.npz") != fit["ridge_sha256"]:
        raise RuntimeError("Wrong frozen model source or ridge changed")
    repo = Path(__file__).resolve().parents[1]
    for mapping in (template["sources"], model_identity["sources"]):
        for name, digest in mapping.items():
            if sha(repo / name) != digest:
                raise RuntimeError(f"Frozen source code changed: {name}")
    protocol = dict(schema=1, source=str(source), experiment=str(experiment.resolve()),
        source_identity=sha(source / "identity.json"), model_identity=sha(experiment / "identity.json"),
        fit_sha256=sha(experiment / "fit.json"), preparer_sha256=sha(Path(__file__)),
        inventory_identity=sha(inventory / "split_identity.json"), mapping_sha256=MAPPING_SHA256,
        scenes_per_session=4, sessions=4, gap_s=10., seed=20261027,
        primary=fit["primary"], archive_cache=str(cache.resolve()), network_image_bytes=0,
        purpose="new-session validation of frozen proposer; no refit or calibration; archive-selection bias remains")
    if destination.exists():
        if read(destination / "validation_protocol.json") != protocol:
            raise RuntimeError("Validation protocol changed; choose a new root")
        frozen = read(destination / "split_identity.json")
        if frozen != {"splits": sha(destination / "splits.json"), "inventory": sha(destination / "inventory.csv")}:
            raise RuntimeError("Validation selection changed")
        print(json.dumps(read(destination / "inventory_report.json"), indent=2))
        return
    original_split = read(inventory / "splits.json")
    excluded = {r["session_id"] for r in read(source / "splits.json")["records"]}
    excluded.update(r["session_id"] for r in original_split["records"] if r["role"] in ("calibration", "test"))
    excluded.update(original_split["navtest_sessions_excluded"])
    with (inventory / "inventory.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    installed = {p.stem for p in Path(template["navsim_log_path"]).glob("*.pkl")}
    rows = [dict(r, timestamp_us=int(r["timestamp_us"])) for r in rows if r["log_name"] in installed]
    raw = fetch_bytes(BASE + "/resolve/main/" + MAPPING + "?download=1&cachebust=" + str(int(time.time())))
    if hashlib.sha256(raw).hexdigest() != MAPPING_SHA256:
        raise RuntimeError("Official mapping changed")
    candidates = inventory_candidates(rows, json.loads(raw), excluded, 4, 10., 20261027)
    old_plan = source / "image_preparation/archive_plan.json"
    if sha(old_plan) != read(source / "budget_plan.json")["archive_plan_sha256"]:
        raise RuntimeError("Source archive plan changed")
    import re
    archives, logs_by_session, shard_of = [], {}, {}
    for entry in read(old_plan)["routes"]["camera_log_shards"]["archives"]:
        i = int(re.search(r"_(\d+)\.tgz$", entry["path"])[1])
        path = cache / Path(entry["path"]).name
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise RuntimeError(f"Complete cached source archive required, no download: {path}")
        archives.append(entry)
        for sid, logs in candidates.get(i, {}).items():
            logs_by_session.setdefault(sid, []).extend(logs)
            shard_of.update({log: i for log in logs})
    if len(logs_by_session) != 4:
        raise RuntimeError(f"Expected exactly four available new sessions; found {len(logs_by_session)}. Review metadata before creating cohort.")
    selected, chosen_logs = [], set()
    for sid in sorted(logs_by_session):
        log = min(logs_by_session[sid], key=lambda name: (token_seed(name, 20261027), name))
        chosen_logs.add(log)
        group = [r for r in rows if r["log_name"] == log]
        selected.extend(dict(r, role="development") for r in spaced_records(group, 4, 10., token_seed(sid, 20261027)))
    picked = {shard_of[log] for log in chosen_logs}
    archives = [dict(e, logs=sorted(log for log in chosen_logs if shard_of[log] == int(re.search(r"_(\d+)\.tgz$", e["path"])[1])))
                for e in archives if int(re.search(r"_(\d+)\.tgz$", e["path"])[1]) in picked]
    new_identity = copy.deepcopy(template)
    new_identity["settings"].update(root=str(destination), sessions={r: 4 if r == "development" else 0 for r in ROLES},
                                    scenes={r: 4 for r in ROLES})
    destination.mkdir(parents=True)
    save_json(destination / "identity.json", new_identity)
    with (destination / "inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(r for r in rows if r["log_name"] in chosen_logs)
    save_json(destination / "splits.json", dict(records=selected,
        allocation={r: sorted(logs_by_session) if r == "development" else [] for r in ROLES},
        log_sha256={log: sha(Path(template["navsim_log_path"]) / (log + ".pkl")) for log in sorted(chosen_logs)},
        navtest_sessions_excluded=original_split["navtest_sessions_excluded"],
        all_excluded_sessions=sorted(excluded), purpose=protocol["purpose"]))
    save_json(destination / "split_identity.json", dict(splits=sha(destination / "splits.json"), inventory=sha(destination / "inventory.csv")))
    requests = collect_requests(destination)
    prep = destination / "image_preparation"
    prep.mkdir()
    save_json(prep / "requests.json", requests)
    total = sum(e["size"] for e in archives)
    save_json(prep / "archive_plan.json", dict(requests_sha256=sha(prep / "requests.json"),
        routes={"camera_log_shards": dict(archives=archives, size_upper_bound=total)},
        suggested_route="camera_log_shards", mapping_sha256=MAPPING_SHA256))
    save_json(destination / "budget_plan.json", dict(spec=dict(budget_bytes=total), archive_plan_sha256=sha(prep / "archive_plan.json")))
    save_json(destination / "validation_protocol.json", protocol)
    report = dict(status="prepared", selected_scenes=16, selected_sessions=4, sessions=sorted(logs_by_session),
        required_images=len(requests["images"]), image_download_GB=0, frozen_primary=fit["primary"],
        source=str(source), validation_root=str(destination), model_root=str(experiment),
        note="Metadata-selected new sessions; no model fitting, evaluation or image transfer yet")
    save_json(destination / "inventory_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--inventory-run", type=Path)
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.experiment, args.inventory_run or args.source.parent / "pilot_v1", args.destination, args.archive_dir)
