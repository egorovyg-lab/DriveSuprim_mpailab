"""Metadata-only plan for four new M/F sessions in the approved shard 174."""
import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim.agents.drivemem import ROLES, spaced_records, token_seed
from plan_drivemem_validation import inventory_candidates
from plan_drivemem_images import BASE, MAPPING, MAPPING_SHA256, archive_entry, collect_requests, fetch_bytes, save_json, sha

EXPECTED = {
    "2021.06.09.19.40.26_veh-12", "2021.08.17.16.48.45_veh-43",
    "2021.08.31.13.27.52_veh-40", "2021.10.11.07.12.18_veh-50",
}


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def allocation(sessions, seed):
    if set(sessions) != EXPECTED:
        raise ValueError("Shard 174 session set changed; review before collection")
    order = np.random.default_rng(seed).permutation(sorted(sessions)).tolist()
    return {r: sorted(order[:2] if r == "memory" else order[2:] if r == "fit" else []) for r in ROLES}


def prepare(source, inventory, validation, destination):
    source, inventory, validation, destination = [p.resolve() for p in (source, inventory, validation, destination)]
    for root in (source, inventory, validation):
        if destination == root or root in destination.parents or destination in root.parents:
            raise ValueError("New dataset root must be separate")
        if read(root / "split_identity.json") != dict(splits=sha(root / "splits.json"), inventory=sha(root / "inventory.csv")):
            raise RuntimeError("Frozen inputs changed")
    template = read(source / "identity.json")
    repo = Path(__file__).resolve().parents[1]
    for name, digest in template["sources"].items():
        if sha(repo / name) != digest:
            raise RuntimeError("Frozen teacher/scorer code changed")
    protocol = dict(source=str(source), inventory=str(inventory), validation=str(validation),
        inputs={str(r): {n: sha(r / n) for n in ("identity.json", "split_identity.json")} for r in (source, inventory, validation)},
        preparer_path="download/prepare_drivemem_new_mf.py", preparer=sha(Path(__file__)),
        seed=int(template["settings"]["seed"]), screening_per_session=16, uniform_per_session=4,
        challenge_per_session=4, session_count=4, shard=174, max_archive_bytes=8500000000,
        purpose="Four new TRAINING/TUNING sessions: M2/F2. Not independent final validation.")
    if destination.exists():
        if read(destination / "expansion_protocol.json") != protocol:
            raise RuntimeError("New-MF protocol changed; choose a new root")
        if read(destination / "split_identity.json") != dict(splits=sha(destination / "splits.json"), inventory=sha(destination / "inventory.csv")):
            raise RuntimeError("New-MF selection changed")
        print(json.dumps(read(destination / "expansion_plan.json"), indent=2))
        return
    original, historical, seen = [read(r / "splits.json") for r in (source, inventory, validation)]
    excluded = {r["session_id"] for r in original["records"] + seen["records"]}
    excluded.update(r["session_id"] for r in historical["records"] if r["role"] in ("calibration", "test"))
    excluded.update(historical["navtest_sessions_excluded"])
    with (inventory / "inventory.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    installed = {p.stem for p in Path(template["navsim_log_path"]).glob("*.pkl")}
    rows = [dict(r, timestamp_us=int(r["timestamp_us"])) for r in rows if r["log_name"] in installed]
    raw = fetch_bytes(BASE + "/resolve/main/" + MAPPING + "?download=1")
    if hashlib.sha256(raw).hexdigest() != MAPPING_SHA256:
        raise RuntimeError("Official mapping changed")
    available = inventory_candidates(rows, json.loads(raw), excluded, 16, 10., protocol["seed"]).get(174, {})
    groups = allocation(available, protocol["seed"])
    api = "https://huggingface.co/api/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_trainval_camera?limit=1000"
    path = "openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_174.tgz"
    matching = [e for e in json.loads(fetch_bytes(api)) if e.get("path") == path]
    if len(matching) != 1:
        raise RuntimeError("Approved archive not found")
    entry = archive_entry(matching[0])
    if entry["size"] > protocol["max_archive_bytes"]:
        raise RuntimeError("Approved 8.5 GB budget exceeded; no archive downloaded")
    selected = []
    for role in ("memory", "fit"):
        for sid in groups[role]:
            log = min(available[sid], key=lambda log: (token_seed(log, protocol["seed"]), log))
            selected.extend(dict(r, role=role) for r in spaced_records(
                [r for r in rows if r["log_name"] == log], 16, 10., token_seed(sid, protocol["seed"])))
    logs = sorted({r["log_name"] for r in selected})
    new_identity = copy.deepcopy(template)
    new_identity["settings"].update(root=str(destination), sessions={r: len(groups[r]) for r in ROLES}, scenes={r: 16 for r in ROLES})
    destination.mkdir(parents=True)
    save_json(destination / "identity.json", new_identity)
    with (destination / "inventory.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(selected[0]))
        w.writeheader()
        w.writerows(selected)
    save_json(destination / "splits.json", dict(records=selected, allocation=groups,
        log_sha256={log: sha(Path(template["navsim_log_path"]) / (log + ".pkl")) for log in logs},
        navtest_sessions_excluded=historical["navtest_sessions_excluded"], all_excluded_sessions=sorted(excluded), purpose=protocol["purpose"]))
    save_json(destination / "split_identity.json", dict(splits=sha(destination / "splits.json"), inventory=sha(destination / "inventory.csv")))
    prep = destination / "image_preparation"
    prep.mkdir()
    requests = collect_requests(destination)
    save_json(prep / "requests.json", requests)
    entry["logs"] = logs
    save_json(prep / "archive_plan.json", dict(requests_sha256=sha(prep / "requests.json"),
        routes={"camera_log_shards": dict(archives=[entry], size_upper_bound=entry["size"])},
        suggested_route="camera_log_shards", mapping_sha256=MAPPING_SHA256))
    save_json(destination / "budget_plan.json", dict(spec=dict(budget_bytes=protocol["max_archive_bytes"]), archive_plan_sha256=sha(prep / "archive_plan.json")))
    report = dict(status="planned", allocation=groups, screening_scenes=64, label_scenes=32,
        uniform_scenes=16, challenge_scenes=16, candidate_evaluations=512, raw_filtered_calls=1024,
        archive=entry["path"], archive_sha256=entry["sha256"], compressed_GB=entry["size"] / 1e9,
        max_GB=8.5, required_images=len(requests["images"]), image_archive_downloaded=False,
        note=protocol["purpose"])
    save_json(destination / "expansion_plan.json", report)
    save_json(destination / "expansion_protocol.json", protocol)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    a = p.parse_args()
    prepare(a.source, a.inventory, a.validation, a.destination)
