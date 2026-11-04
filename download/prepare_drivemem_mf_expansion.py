"""Freeze a bounded M/F screening pool using cached camera archives only."""
import argparse
import copy
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim.agents.drivemem import ROLES, spaced_records, token_seed
from plan_drivemem_images import BASE, MAPPING, MAPPING_SHA256, collect_requests, fetch_bytes, save_json, sha


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def select_pool(rows, old, roles, logs, count=16, gap=10., seed=20261027):
    chosen, report = [], []
    for sid, role in sorted(roles.items()):
        past = [int(r["timestamp_us"]) for r in old if r["session_id"] == sid]
        tokens = {r["token"] for r in old}
        unique = {}
        for r in sorted(rows, key=lambda x: (x["token"], x["log_name"])):
            if r["session_id"] != sid or r["log_name"] not in logs or r["token"] in tokens:
                continue
            t = int(r["timestamp_us"])
            if all(abs(t - p) >= int(gap * 1e6) for p in past):
                unique.setdefault(t, dict(r, timestamp_us=t, role=role))
        group = list(unique.values())
        try:
            pick = spaced_records(group, count, gap, token_seed(sid, seed))
        except ValueError:
            pick = []
        report.append(dict(session_id=sid, role=role, eligible_scenes=len(group),
                           requested=count, selected=len(pick)))
        chosen.extend(pick)
    return chosen, report


def prepare(source, inventory, validation, destination, cache):
    source, inventory, validation, destination, cache = [p.resolve() for p in (source, inventory, validation, destination, cache)]
    for root in (source, inventory, validation):
        if root == destination or root in destination.parents or destination in root.parents:
            raise ValueError("Expansion must be a separate root")
        if read(root / "split_identity.json") != dict(splits=sha(root / "splits.json"), inventory=sha(root / "inventory.csv")):
            raise RuntimeError("Frozen input split changed")
    template = read(source / "identity.json")
    repo = Path(__file__).resolve().parents[1]
    for name, digest in template["sources"].items():
        if sha(repo / name) != digest:
            raise RuntimeError(f"Frozen code changed: {name}")
    protocol = dict(source=str(source), inventory=str(inventory), validation=str(validation),
        inputs={str(r): {n: sha(r / n) for n in ("identity.json", "split_identity.json")} for r in (source, inventory, validation)},
        preparer=sha(Path(__file__)), archive_dir=str(cache), seed=int(template["settings"]["seed"]),
        screening_per_session=16, uniform_per_session=4, challenge_per_session=4, gap_s=10.,
        purpose="M/F expansion only; no new independent sessions; no query labels used")
    if destination.exists():
        if read(destination / "expansion_protocol.json") != protocol:
            raise RuntimeError("Expansion protocol changed; use a new root")
        if read(destination / "split_identity.json") != dict(splits=sha(destination / "splits.json"), inventory=sha(destination / "inventory.csv")):
            raise RuntimeError("Expansion split changed")
        print(json.dumps(read(destination / "expansion_plan.json"), indent=2))
        return
    original = read(source / "splits.json")
    roles = {}
    for r in original["records"]:
        if r["role"] in ("memory", "fit"):
            if roles.setdefault(r["session_id"], r["role"]) != r["role"]:
                raise ValueError("Session has multiple roles")
    if list(roles.values()).count("memory") != 6 or list(roles.values()).count("fit") != 4:
        raise ValueError("Expected original 6 M / 4 F sessions")
    query_sessions = {r["session_id"] for r in original["records"] if r["role"] not in ("memory", "fit")}
    query_sessions.update(r["session_id"] for r in read(validation / "splits.json")["records"])
    query_sessions.update(original["navtest_sessions_excluded"])
    if set(roles) & query_sessions:
        raise RuntimeError("M/F session overlaps inspected query or reserved role")
    # Original inventory allocations may precede the budget split. Disclose this
    # history; already-used training sessions can never become fresh C/T again.
    historical = read(inventory / "splits.json")
    historical_conflicts = sorted(set(roles) & {r["session_id"] for r in historical["records"] if r["role"] in ("calibration", "test")})
    old = original["records"] + historical["records"] + read(validation / "splits.json")["records"]
    plan_file = source / "image_preparation/archive_plan.json"
    if sha(plan_file) != read(source / "budget_plan.json")["archive_plan_sha256"]:
        raise RuntimeError("Source archive plan changed")
    archives = read(plan_file)["routes"]["camera_log_shards"]["archives"]
    raw = fetch_bytes(BASE + "/resolve/main/" + MAPPING + "?download=1")
    if hashlib.sha256(raw).hexdigest() != MAPPING_SHA256:
        raise RuntimeError("Official mapping changed")
    mapping, by_log = json.loads(raw), {}
    for entry in archives:
        number = int(re.search(r"_(\d+)\.tgz$", entry["path"])[1])
        local = cache / Path(entry["path"]).name
        print(f"VERIFY LOCAL ARCHIVE {local.name} ({entry['size'] / 1e9:.3f} GB)", flush=True)
        if not local.is_file() or local.stat().st_size != entry["size"] or sha(local) != entry["sha256"]:
            raise RuntimeError(f"Complete SHA256-valid local archive required (no download): {local}")
        for log in mapping[f"openscene_sensor_trainval_{number}"]:
            if log in by_log:
                raise RuntimeError("Ambiguous archive mapping")
            by_log[log] = entry
    with (inventory / "inventory.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    installed = {p.stem for p in Path(template["navsim_log_path"]).glob("*.pkl")}
    selected, sessions = select_pool(rows, old, roles, set(by_log) & installed, seed=protocol["seed"])
    report = dict(status="planned" if len(selected) == 160 else "shortfall", sessions=sessions,
        screening_scenes=len(selected), label_scenes=80, candidate_evaluations=1280, raw_filtered_calls=2560,
        network_image_GB=0, metadata_download_only=True, historical_role_conflicts=historical_conflicts,
        note="Original budget M/F roles are authoritative; history disclosed. No fresh holdout or labels created.")
    if len(selected) != 160:
        print(json.dumps(report, indent=2), flush=True)
        raise RuntimeError("Cannot provide 16 new spaced scenes per session from cached archives; no output published")
    counts = {r: sum(v == r for v in roles.values()) for r in ROLES}
    new_identity = copy.deepcopy(template)
    new_identity["settings"].update(root=str(destination), sessions=counts, scenes={r: 16 for r in ROLES})
    logs = sorted({r["log_name"] for r in selected})
    destination.mkdir(parents=True)
    save_json(destination / "identity.json", new_identity)
    with (destination / "inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    save_json(destination / "splits.json", dict(records=selected, allocation={r: sorted(s for s, v in roles.items() if v == r) for r in ROLES},
        log_sha256={log: sha(Path(template["navsim_log_path"]) / (log + ".pkl")) for log in logs},
        navtest_sessions_excluded=original["navtest_sessions_excluded"], purpose=protocol["purpose"]))
    save_json(destination / "split_identity.json", dict(splits=sha(destination / "splits.json"), inventory=sha(destination / "inventory.csv")))
    requests = collect_requests(destination)
    prep = destination / "image_preparation"
    prep.mkdir()
    save_json(prep / "requests.json", requests)
    used = [dict(e, logs=[log for log in logs if by_log[log]["path"] == e["path"]]) for e in archives]
    used = [e for e in used if e["logs"]]
    total = sum(e["size"] for e in used)
    save_json(prep / "archive_plan.json", dict(requests_sha256=sha(prep / "requests.json"),
        routes={"camera_log_shards": dict(archives=used, size_upper_bound=total)}, suggested_route="camera_log_shards", mapping_sha256=MAPPING_SHA256))
    save_json(destination / "budget_plan.json", dict(spec=dict(budget_bytes=total), archive_plan_sha256=sha(prep / "archive_plan.json")))
    report.update(required_images=len(requests["images"]), requested_archive_scan_GB=total / 1e9)
    save_json(destination / "expansion_plan.json", report)
    save_json(destination / "expansion_protocol.json", protocol)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--archive-dir", type=Path, required=True)
    a = p.parse_args()
    prepare(a.source, a.inventory, a.validation, a.destination, a.archive_dir)
