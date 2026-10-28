"""Plan camera acquisition for frozen memory splits, without large downloads.

Reads only past/current camera paths from the selected, checksummed source logs.
Downloads small official archive manifests to estimate compressed transfer size.
"""
import argparse
import hashlib
import json
import pickle
import re
import time
import urllib.request
from pathlib import Path, PurePosixPath

BASE = "https://huggingface.co/datasets/OpenDriveLab/OpenScene"
MAPPING = "openscene-v1.1/openscene_sensor_trainval_0-199.json"
MAPPING_SHA256 = "52ee3d5eb7c4b7e3681bea6fe410458d3b30a2c8d480ac9882d6c6f5a6039bf2"
CAMERAS = {"cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"}


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def safe_image_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Empty camera path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise ValueError(f"Unexpected camera path: {value}")
    return str(path)


def collect_requests(run):
    identity = json.loads((run / "identity.json").read_text(encoding="utf-8"))
    split = json.loads((run / "splits.json").read_text(encoding="utf-8"))
    frozen = json.loads((run / "split_identity.json").read_text(encoding="utf-8"))
    if frozen["splits"] != sha(run / "splits.json") or frozen["inventory"] != sha(run / "inventory.csv"):
        raise RuntimeError("Frozen split/inventory changed")
    history = int(identity["train_test_split"]["scene_filter"]["num_history_frames"])
    if history != 4:
        raise RuntimeError("Expected original DriveSuprim SensorConfig: eight cameras on four history frames")
    records = split["records"]
    if not records:
        raise RuntimeError("Empty selection")
    requests = set()
    images_by_log = {}
    for log in sorted({r["log_name"] for r in records}):
        path = Path(identity["navsim_log_path"]) / (log + ".pkl")
        if sha(path) != split["log_sha256"][log]:
            raise RuntimeError(f"Source log changed: {log}")
        with path.open("rb") as stream:
            frames = pickle.load(stream)
        positions = {}
        for i, frame in enumerate(frames):
            if frame["token"] in positions:
                raise RuntimeError(f"Duplicate frame token in source log: {log}")
            positions[frame["token"]] = i
        log_requests = set()
        for record in (r for r in records if r["log_name"] == log):
            current = positions[record["token"]]
            if current < history - 1:
                raise RuntimeError("Insufficient camera history")
            for frame in frames[current - history + 1:current + 1]:
                present = set()
                for name, camera in frame["cams"].items():
                    if name.lower() in CAMERAS:
                        present.add(name.lower())
                        log_requests.add(safe_image_path(camera["data_path"]))
                if present != CAMERAS:
                    raise RuntimeError(f"Missing camera metadata: {log}/{frame['token']}")
        images_by_log[log] = sorted(log_requests)
        requests.update(log_requests)
    return {"schema_version": 1, "split_sha256": sha(run / "splits.json"),
            "sensor_root": identity["original_sensor_path"], "images": sorted(requests),
            "scenes": len(records), "sessions": len({r["session_id"] for r in records}),
            "logs": sorted({r["log_name"] for r in records}), "images_by_log": images_by_log,
            "sensor_policy": "original agent SensorConfig: eight cameras at history indices 0,1,2,3; no LiDAR"}


def fetch_bytes(url):
    request = urllib.request.Request(url, headers={"User-Agent": "DriveMem-image-plan/1", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=90) as response:
        content = response.read(8 * 1024 ** 2 + 1)
        if len(content) > 8 * 1024 ** 2:
            raise RuntimeError("Unexpectedly large manifest response")
        return content


def archive_entry(entry):
    digest = entry.get("lfs", {}).get("oid", "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or int(entry.get("size", 0)) <= 0:
        raise RuntimeError(f"Missing archive SHA256/size: {entry.get('path')}")
    return {"path": entry["path"], "size": int(entry["size"]), "sha256": digest,
            "url": BASE + "/resolve/main/" + entry["path"]}


def route_plans(requests, mapping, compact_entries, full_entries):
    compact = [archive_entry(e) for e in compact_entries
               if re.fullmatch(r"navsim/navtrain_(current|history)_\d+\.tgz", e.get("path", ""))]
    expected = {f"navsim/navtrain_{kind}_{i}.tgz" for kind in ("current", "history") for i in range(1, 33)}
    if {e["path"] for e in compact} != expected:
        raise RuntimeError("Official compact archive listing is incomplete; refusing inaccurate plan")
    compact.sort(key=lambda e: ("history" in e["path"], int(re.search(r"_(\d+)\.tgz$", e["path"])[1])))
    reverse = {}
    for key, logs in mapping.items():
        match = re.fullmatch(r"openscene_sensor_trainval_(\d+)", key)
        if match is None or not isinstance(logs, list):
            raise RuntimeError("Unexpected official log/shard mapping schema")
        for log in logs:
            if log in reverse:
                raise RuntimeError(f"Ambiguous shard mapping: {log}")
            reverse[log] = int(match[1])
    routes = {"navtrain_compact": {"archives": compact, "size_upper_bound": sum(e["size"] for e in compact),
                                   "description": "64 compact archives; no public per-log index for these archives; can stop when all requested files are found"}}
    unknown = sorted(set(requests["logs"]) - set(reverse))
    if not unknown:
        ids = {reverse[log] for log in requests["logs"]}
        wanted = {f"openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_{i}.tgz" for i in ids}
        full = [archive_entry(e) for e in full_entries if e.get("path") in wanted]
        if {e["path"] for e in full} != wanted:
            raise RuntimeError("Full camera archive listing is incomplete")
        routes["camera_log_shards"] = {"archives": sorted(full, key=lambda e: e["path"]),
                                        "size_upper_bound": sum(e["size"] for e in full),
                                        "description": "Only full-camera shards containing selected log names; extract only requested images"}
    return routes, unknown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    requests = collect_requests(run)
    out = run / "image_preparation"
    out.mkdir(exist_ok=True)
    save_json(out / "requests.json", requests)
    sensor_root = Path(requests["sensor_root"])
    existing = sum((sensor_root / image).is_file() for image in requests["images"])
    print(f"Selected scenes={requests['scenes']}; required images={len(requests['images'])}; existing={existing}", flush=True)
    if existing == len(requests["images"]):
        print("All required files already exist; no archive planning is needed.")
        return
    stamp = str(int(time.time()))
    mapping_bytes = fetch_bytes(BASE + "/resolve/main/" + MAPPING + "?download=1&cachebust=" + stamp)
    if hashlib.sha256(mapping_bytes).hexdigest() != MAPPING_SHA256:
        raise RuntimeError("Official log/shard mapping changed; inspect before planning")
    api = "https://huggingface.co/api/datasets/OpenDriveLab/OpenScene/tree/main/"
    compact = json.loads(fetch_bytes(api + "navsim?limit=1000"))
    full = json.loads(fetch_bytes(api + "openscene-v1.1/openscene_sensor_trainval_camera?limit=1000"))
    routes, unknown = route_plans(requests, json.loads(mapping_bytes), compact, full)
    preferred = min(routes, key=lambda name: routes[name]["size_upper_bound"])
    report = {"status": "planned", "scenes": requests["scenes"], "sessions": requests["sessions"],
              "selected_log_segments": len(requests["logs"]), "required_images": len(requests["images"]),
              "existing_images": existing, "missing_images": len(requests["images"]) - existing,
              "sensor_root": str(sensor_root), "suggested_route": preferred,
              "routes": {name: {"archive_count": len(route["archives"]),
                                "compressed_GB_upper_bound": route["size_upper_bound"] / 1e9,
                                "largest_archive_GB": max(e["size"] for e in route["archives"]) / 1e9}
                         for name, route in routes.items()}, "logs_without_full_shard_mapping": unknown,
              "note": "Only small JSON manifests were downloaded. Archive transfer can greatly exceed extracted pilot image size."}
    save_json(out / "archive_plan.json", {"requests_sha256": sha(out / "requests.json"), "routes": routes,
                                          "suggested_route": preferred, "mapping_sha256": MAPPING_SHA256})
    save_json(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
