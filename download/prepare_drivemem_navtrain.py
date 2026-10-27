"""Download and install only official navtrain metadata; no sensor downloads.

The official metadata archive is ~7.05 GB. Its expected digest/size were read
from the dataset's public Git-LFS pointer on 2026-10-27.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath

URL = "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz"
EXPECTED_SIZE = 7052210642
EXPECTED_SHA256 = "193138ac68b29d977d4438a95dfda35a07ff4ad3efb1800b3ee47d6e1dcb3764"


def digest(path, progress=False):
    value = hashlib.sha256()
    total, next_print = 0, 1024 ** 3
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            value.update(block)
            total += len(block)
            if progress and total >= next_print:
                print(f"VERIFY {total / 1e9:.2f} GB", flush=True)
                next_print += 1024 ** 3
    return value.hexdigest()


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def download_archive(directory):
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "openscene_metadata_trainval.tgz"
    part = archive.with_name(archive.name + ".part")
    if archive.exists():
        if archive.stat().st_size != EXPECTED_SIZE or digest(archive, True) != EXPECTED_SHA256:
            raise RuntimeError(f"Existing archive has unexpected size/hash: {archive}; preserved for inspection")
        return archive
    if not shutil.which("curl"):
        raise RuntimeError("curl is required for resumable download; not found in PATH")
    already = part.stat().st_size if part.exists() else 0
    if already > EXPECTED_SIZE:
        raise RuntimeError(f"Partial file is larger than expected: {part}")
    if shutil.disk_usage(directory).free < EXPECTED_SIZE - already + 1024 ** 3:
        raise RuntimeError("Insufficient free disk space for the remaining 7.05 GB archive plus 1 GiB headroom")
    if already < EXPECTED_SIZE:
        print(f"DOWNLOAD {URL}\nDestination: {part}\nResume offset: {already} bytes", flush=True)
        subprocess.run(["curl", "--fail", "--location", "--retry", "4", "--retry-delay", "5",
                        "--connect-timeout", "30", "--continue-at", "-", "--output", str(part), URL], check=True)
    if part.stat().st_size != EXPECTED_SIZE or digest(part, True) != EXPECTED_SHA256:
        raise RuntimeError(f"Download size/SHA256 mismatch: {part}; no extraction performed")
    os.replace(part, archive)
    return archive


def extract_logs(archive, destination, allowed):
    """Stream archive; flatten only exact allowlisted *.pkl filenames.

    Never call extractall, write symlinks, or replace differing existing files.
    """
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    completed, seen = {}, set()
    last_print = time.monotonic()
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if not member.isfile() or "meta_datas" not in path.parts or path.suffix != ".pkl" or path.stem not in allowed:
                continue
            if path.stem in seen:
                raise RuntimeError(f"Duplicate requested log in archive: {path.stem}")
            seen.add(path.stem)
            target = destination / path.name
            if target.is_symlink() or target.resolve().parent != destination:
                raise RuntimeError(f"Unsafe output target: {target}")
            stream = tar.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Unable to read: {member.name}")
            exists = target.exists()
            if not exists and shutil.disk_usage(destination).free < member.size + 100 * 1024 ** 2:
                raise RuntimeError(f"Not enough disk space to extract {path.name}; previous files preserved")
            temp = target.with_name(target.name + ".tmp")
            hasher, written = hashlib.sha256(), 0
            sink = None if exists else temp.open("wb")
            try:
                for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
                    hasher.update(block)
                    written += len(block)
                    if sink is not None:
                        sink.write(block)
            finally:
                stream.close()
                if sink is not None:
                    sink.close()
            if written != member.size:
                raise RuntimeError(f"Truncated member: {path.name}")
            checksum = hasher.hexdigest()
            if exists:
                if target.stat().st_size != member.size or digest(target) != checksum:
                    raise RuntimeError(f"Existing log differs from verified archive: {target}; not overwritten")
            else:
                os.replace(temp, target)
            completed[path.stem] = {"size": written, "sha256": checksum}
            if len(completed) % 25 == 0 or time.monotonic() - last_print > 10:
                print(f"EXTRACT {len(completed)}/{len(allowed)} navtrain logs", flush=True)
                last_print = time.monotonic()
    missing = sorted(set(allowed) - set(completed))
    if missing:
        raise RuntimeError(f"Archive lacks {len(missing)} configured logs: {missing[:10]}")
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("metadata", "status"))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("OPENSCENE_DATA_ROOT", "/workspace/dataset")))
    parser.add_argument("--download-dir", type=Path)
    args = parser.parse_args()
    root = args.data_root.resolve()
    destination = root / "navsim_logs/trainval"
    sensor_path = root / "sensor_blobs/trainval"
    if args.stage == "status":
        print("Trainval logs:", destination, "exists:", destination.is_dir())
        print("Log files:", sum(1 for _ in destination.glob("*.pkl")))
        print("Trainval images:", sensor_path, "exists:", sensor_path.is_dir())
        return
    from omegaconf import OmegaConf
    cfg_path = Path(__file__).resolve().parents[1] / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
    cfg = OmegaConf.load(cfg_path)
    allowed = {str(name) for name in cfg.log_names}
    if not allowed or any(Path(name).name != name for name in allowed):
        raise RuntimeError("Invalid configured log allowlist")
    marker = root / "navsim_logs/navtrain_metadata_install.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("config_sha256") == digest(cfg_path) and saved.get("archive_sha256") == EXPECTED_SHA256:
            for name, info in saved["logs"].items():
                path = destination / (name + ".pkl")
                if not path.is_file() or digest(path) != info["sha256"]:
                    raise RuntimeError(f"Installed metadata missing/changed: {path}")
            if set(saved["logs"]) == allowed:
                print(f"Metadata already installed and verified: {len(allowed)} logs at {destination}")
                return
    archive = download_archive(args.download_dir or root / ".downloads/drivemem")
    completed = extract_logs(archive, destination, allowed)
    write_json(marker, {"status": "complete", "url": URL, "archive_sha256": EXPECTED_SHA256,
                        "archive_size": EXPECTED_SIZE, "config_sha256": digest(cfg_path), "logs": completed})
    print(f"METADATA COMPLETE: {len(completed)} logs at {destination}", flush=True)
    print("Images have not been downloaded. Next run the memory inventory stage.", flush=True)


if __name__ == "__main__":
    main()
