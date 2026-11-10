"""Isolated Drex runtime checks. No NAVSIM data and no full-model inference."""
import argparse
import fnmatch
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

MODEL = "nace-ai/drex-max-v1"
PATTERNS = ("*.json", "*.jinja", "model-*.safetensors", "readout.safetensors", "LICENSE")
REQUIRED = {"config.json", "decision_config.json", "model.safetensors.index.json",
            "readout.safetensors", "processor_config.json", "tokenizer.json"}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def manifest(info):
    rows = []
    for item in info.siblings:
        if not any(fnmatch.fnmatch(item.rfilename, p) for p in PATTERNS):
            continue
        if Path(item.rfilename).name != item.rfilename or not item.size or item.size < 0:
            raise ValueError("Unexpected path or missing size: " + item.rfilename)
        rows.append(dict(name=item.rfilename, size=item.size))
    if not REQUIRED <= {r["name"] for r in rows} or not any(r["name"].startswith("model-") for r in rows):
        raise ValueError("Incomplete model file listing")
    if len(info.sha) != 40 or any(c not in "0123456789abcdef" for c in info.sha):
        raise ValueError("Immutable revision required")
    return dict(model=MODEL, revision=info.sha, files=sorted(rows, key=lambda x: x["name"]),
                total_bytes=sum(r["size"] for r in rows))


def check_lock(path, current):
    if not path.is_file() or path.read_text(encoding="utf-8").splitlines() != current.splitlines():
        raise RuntimeError("Environment differs from setup lock")


def environment(root, gpu):
    if sys.version_info < (3, 12):
        raise RuntimeError("Python >=3.12 required")
    current = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    check_lock(root / "requirements.lock.txt", current)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    import torch
    from transformers import AutoProcessor, Qwen3_5Model
    from drex_max.model import DecisionModel
    result = dict(python=sys.version, torch=torch.__version__, cuda_build=torch.version.cuda,
                  transformers=importlib.metadata.version("transformers"), model_imports="ok",
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    if gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use gpu-check through Slurm")
        props = torch.cuda.get_device_properties(0)
        if props.total_memory < 70 * 1024**3 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 protocol requires >=70 GiB VRAM and BF16 support")
        a = torch.ones((32, 32), device="cuda:0", dtype=torch.bfloat16)
        if (a @ a).float().mean().item() != 32:
            raise RuntimeError("BF16 calculation failed")
        result.update(gpu=props.name, vram_bytes=props.total_memory, bf16="ok")
    write(root / ("gpu_check.json" if gpu else "environment_check.json"), result)
    return result


def plan(root):
    path = root / "model_plan.json"
    if path.exists():
        return read(path)
    from huggingface_hub import HfApi
    value = manifest(HfApi().model_info(MODEL, files_metadata=True))
    write(path, value)
    return value


def remaining_bytes(target, files):
    return sum(r["size"] for r in files if not (target / r["name"]).is_file()
               or (target / r["name"]).stat().st_size != r["size"])


def download(root):
    path = root / "model_plan.json"
    if not path.exists():
        raise RuntimeError("Run model-plan first")
    value = read(path)
    target = root / "models/drex-max-v1"
    target.mkdir(parents=True, exist_ok=True)
    needed = remaining_bytes(target, value["files"])
    if shutil.disk_usage(target).free < needed + 4 * 1024**3:
        raise RuntimeError("Insufficient disk space for remaining files plus 4 GiB reserve")
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, revision=value["revision"], local_dir=str(target),
                      allow_patterns=[r["name"] for r in value["files"]])
    if remaining_bytes(target, value["files"]):
        raise RuntimeError("Missing files or size mismatch after download")
    return dict(status="downloaded_sizes_checked", revision=value["revision"], model_path=str(target),
                note="Size check only; no full-file checksum verification or model acceptance yet.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "gpu-check", "model-plan", "download"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    result = environment(root, a.stage == "gpu-check") if a.stage in ("check", "gpu-check") else {
        "model-plan": plan, "download": download}[a.stage](root)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
