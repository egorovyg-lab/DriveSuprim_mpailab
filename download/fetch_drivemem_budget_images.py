"""Fetch only a frozen budget pilot's archives and extract exact requested images.

Processes archives sequentially, verifies SHA256 before extraction, supports
resumable curl downloads and checksummed per-archive installation receipts.
The cap is unique compressed archive size; retries can transfer extra bytes.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath

from plan_drivemem_images import BASE, collect_requests, safe_image_path, save_json, sha
from prepare_drivemem_navtrain import ensure_writable_directory


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_target(root, relative):
    relative = safe_image_path(relative)
    root = root.resolve()
    target = root / relative
    if target.is_symlink() or root not in target.resolve().parents:
        raise RuntimeError(f"Unsafe image destination: {relative}")
    return target


def extract_images(archive, sensor_root, wanted):
    sensor_root.mkdir(parents=True, exist_ok=True)
    completed = {}
    last_print = time.monotonic()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            # Official archives add openscene-v1.1/sensor_blobs/trainval prefixes.
            matches = [str(PurePosixPath(*path.parts[i:])) for i in range(len(path.parts))]
            matches = [name for name in matches if name in wanted]
            if len(matches) > 1:
                raise RuntimeError("Ambiguous requested image path")
            if time.monotonic() - last_print > 15:
                print(f"SCAN {archive.name}: extracted {len(completed)} requested images", flush=True)
                last_print = time.monotonic()
            if not matches:
                continue
            relative = matches[0]
            if not member.isfile() or relative in completed or member.size <= 0:
                raise RuntimeError(f"Invalid/duplicate requested image: {relative}")
            target = checked_target(sensor_root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(target.parent).free < member.size + 100 * 1024 ** 2:
                raise RuntimeError("Insufficient disk space for requested image")
            temp = target.with_name(target.name + ".drivemem.tmp")
            if temp.is_symlink():
                raise RuntimeError(f"Unsafe temporary image path: {temp}")
            stream = tar.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Cannot read requested image: {relative}")
            try:
                with stream, temp.open("wb") as sink:
                    shutil.copyfileobj(stream, sink, 1024 ** 2)
                if temp.stat().st_size != member.size:
                    raise RuntimeError(f"Truncated image: {relative}")
                checksum = sha(temp)
                if target.exists():
                    if target.stat().st_size != member.size or sha(target) != checksum:
                        raise RuntimeError(f"Existing image differs: {target}; preserved")
                else:
                    os.replace(temp, target)
                completed[relative] = {"size": member.size, "sha256": checksum}
            finally:
                temp.unlink(missing_ok=True)
    return completed


def validate_plan(run, max_bytes):
    prep = run / "image_preparation"
    budget = read_json(run / "budget_plan.json")
    plan = read_json(prep / "archive_plan.json")
    if sha(prep / "archive_plan.json") != budget["archive_plan_sha256"]:
        raise RuntimeError("Budget archive plan changed")
    if sha(prep / "requests.json") != plan["requests_sha256"]:
        raise RuntimeError("Image request manifest changed")
    requests = read_json(prep / "requests.json")
    if requests != collect_requests(run):
        raise RuntimeError("Image requests do not match frozen source metadata")
    archives = plan["routes"]["camera_log_shards"]["archives"]
    if not archives or len({a["path"] for a in archives}) != len(archives):
        raise RuntimeError("Empty or duplicate archive plan")
    for entry in archives:
        if (not re.fullmatch(r"openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_\d+\.tgz", entry["path"])
                or entry["url"] != BASE + "/resolve/main/" + entry["path"]
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) or entry["size"] <= 0):
            raise RuntimeError("Unexpected archive metadata")
    logs = [log for entry in archives for log in entry.get("logs", [])]
    if len(set(logs)) != len(logs) or set(logs) != set(requests["logs"]):
        raise RuntimeError("Archive plan must cover each selected log exactly once")
    total = sum(a["size"] for a in archives)
    if total > min(max_bytes, budget["spec"]["budget_bytes"]):
        raise RuntimeError(f"Archive plan requires {total / 1e9:.3f} GB and exceeds the configured cap")
    return requests, archives


def installed_images(prep, sensor_root, archives):
    installed = {}
    for archive in archives:
        receipt = prep / "receipts" / (Path(archive["path"]).name + ".json")
        if not receipt.exists():
            continue
        saved = read_json(receipt)
        if saved["archive_sha256"] != archive["sha256"]:
            raise RuntimeError("Installation receipt refers to a different archive")
        for relative, info in saved["images"].items():
            target = checked_target(sensor_root, relative)
            if not target.is_file() or target.stat().st_size != info["size"] or sha(target) != info["sha256"]:
                raise RuntimeError(f"Installed image missing/changed: {target}")
            installed[relative] = info
    return installed


def download(entry, directory):
    ensure_writable_directory(directory)
    archive = directory / Path(entry["path"]).name
    part = archive.with_name(archive.name + ".part")
    if archive.exists():
        if archive.stat().st_size != entry["size"] or sha(archive) != entry["sha256"]:
            raise RuntimeError(f"Existing archive has incorrect size/SHA256: {archive}")
        return archive
    offset = part.stat().st_size if part.exists() else 0
    if offset > entry["size"]:
        raise RuntimeError(f"Partial archive exceeds expected size: {part}")
    if shutil.disk_usage(directory).free < entry["size"] - offset + 1024 ** 3:
        raise RuntimeError("Insufficient free space for the next archive plus 1 GiB reserve")
    if offset < entry["size"]:
        if not shutil.which("curl"):
            raise RuntimeError("curl is required")
        print(f"DOWNLOAD {archive.name}: {entry['size'] / 1e9:.3f} GB; resume at {offset} bytes", flush=True)
        subprocess.run(["curl", "--fail", "--location", "--connect-timeout", "30",
                        "--max-filesize", str(entry["size"]),
                        "--continue-at", "-", "--output", str(part),
                        entry["url"] + "?download=1&cachebust=" + str(int(time.time()))], check=True)
    if part.stat().st_size != entry["size"] or sha(part) != entry["sha256"]:
        raise RuntimeError(f"Download size/SHA256 mismatch: {part}; no extraction performed")
    os.replace(part, archive)
    return archive


def run_stage(run, stage, max_bytes, archive_dir=None, keep_archives=False):
    requests, archives = validate_plan(run, max_bytes)
    sensor_root = Path(requests["sensor_root"])
    prep = run / "image_preparation"
    wanted = set(requests["images"])
    installed = installed_images(prep, sensor_root, archives)
    print(f"Verified images: {len(wanted & installed.keys())}/{len(wanted)}; "
          f"archive size: {sum(a['size'] for a in archives) / 1e9:.3f} GB", flush=True)
    if stage == "fetch":
        ensure_writable_directory(sensor_root)
        (prep / "receipts").mkdir(exist_ok=True)
        for entry in archives:
            receipt = prep / "receipts" / (Path(entry["path"]).name + ".json")
            if receipt.exists():
                continue  # All its files have just been checksummed above.
            archive = download(entry, archive_dir or prep / "downloads")
            expected = {image for log in entry["logs"] for image in requests["images_by_log"][log]}
            completed = extract_images(archive, sensor_root, expected)
            if set(completed) != expected:
                raise RuntimeError(f"Archive lacks {len(expected - completed.keys())} requested images: "
                                   f"{archive}; archive preserved, no receipt written")
            save_json(receipt, {"archive_sha256": entry["sha256"], "images": completed})
            installed.update(completed)
            if not keep_archives:
                # Only this verified local archive is removed, after writing its receipt.
                archive.unlink()
            print(f"INSTALLED {archive.name}: {len(completed)} images; "
                  f"archive {'retained for reuse' if keep_archives else 'removed'}", flush=True)
    missing = sorted(wanted - installed.keys())
    report = {"status": "complete" if not missing else "incomplete", "required_images": len(wanted),
              "verified_images": len(wanted & installed.keys()), "missing_images": len(missing),
              "missing_examples": missing[:20], "sensor_root": str(sensor_root)}
    if stage == "fetch":
        save_json(prep / "install_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if missing:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fetch", "status"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-gb", type=float, required=True)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--discard-archives", action="store_true", help="Remove verified archives after extraction")
    args = parser.parse_args()
    run_stage(args.run_dir, args.stage, int(args.max_gb * 1e9), args.archive_dir,
              keep_archives=not args.discard_archives)


if __name__ == "__main__":
    main()
