"""Explicit numerical profile for a NEW decoder sidecar; no tolerance changes."""
import os
import random

import numpy as np

SEED = 20261027
PROFILE = dict(seed=SEED, cublas_workspace_config=":4096:8", deterministic_algorithms=True,
               warn_only=False, cudnn_benchmark=False, cudnn_deterministic=True,
               cudnn_allow_tf32=False, matmul_allow_tf32=False, matmul_precision="highest",
               flash_sdp=False, mem_efficient_sdp=False, math_sdp=True)
PILOT_TOKENS = ("64f4a68010155184", "c5c16e45232453e3")


def pilot_order(records):
    if not set(PILOT_TOKENS).issubset({r["token"] for _, r in records}):
        raise ValueError("Required numerical pilot tokens are absent")
    priority = {token: i for i, token in enumerate(PILOT_TOKENS)}
    return sorted(records, key=lambda item: priority.get(item[1]["token"], len(priority)))


def prepare_environment():
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise RuntimeError("Start through run_drivemem_decoder_deterministic.sh (PYTHONHASHSEED must be set before Python)")
    for name, expected in (("CUBLAS_WORKSPACE_CONFIG", PROFILE["cublas_workspace_config"]),):
        if os.environ.get(name, expected) != expected:
            raise RuntimeError("Conflicting numerical setting: " + name)
        os.environ[name] = expected
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or visible == "-1":
        raise RuntimeError("Exactly one CUDA device must be selected before Python starts")


def configure(torch):
    prepare_environment()
    if torch.cuda.is_initialized():
        raise RuntimeError("Numerical profile must be configured before CUDA initialization")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def runtime(torch):
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Deterministic export requires exactly one visible CUDA device")
    observed = dict(seed=SEED, cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision(),
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled())
    if observed != PROFILE:
        raise RuntimeError("Observed Torch settings differ from the frozen numerical profile")
    props = torch.cuda.get_device_properties(0)
    return dict(settings=observed, torch=str(torch.__version__), torch_cuda=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(),
                cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                gpu_name=props.name, capability=[props.major, props.minor],
                total_memory=props.total_memory)


def validate_runtime(out, index, read, sha):
    path = out / "runtime.json"
    if not path.exists():
        if index:
            raise RuntimeError("Completed deterministic scenes have no runtime record")
        return
    if read(path)["settings"] != PROFILE:
        raise RuntimeError("Stored numerical profile changed")
    digest = sha(path)
    if any(entry.get("runtime_sha256") != digest or entry.get("repeat_bitwise_equal") is not True
           for entry in index.values()):
        raise RuntimeError("Deterministic scene/runtime binding changed")


def repeat_parity(reference, current):
    fields = ("candidate_ids", "selected_local", "selection_scores", "final_traj") + tuple(
        k for k in reference if k.startswith("head_"))
    for field in fields:
        if not np.array_equal(reference[field], current[field]):
            raise RuntimeError("Current passes are not bitwise identical: " + field)


def failure_details(reference, current):
    fields = {}
    for name in ("selection_scores",) + tuple(k for k in reference if k.startswith("head_")):
        a, b = np.asarray(reference[name], dtype=float), np.asarray(current[name], dtype=float)
        if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
            fields[name] = dict(comparable=False)
            continue
        delta, allowed = np.abs(a - b), 1e-5 + 1e-6 * np.abs(b)
        fields[name] = dict(max_abs_diff=float(delta.max()), failed_elements=int((delta > allowed).sum()),
                            max_tolerance_ratio=float((delta / allowed).max()))
    return dict(ordered_ids_equal=bool(np.array_equal(reference["candidate_ids"], current["candidate_ids"])),
                selected_local_equal=bool(np.array_equal(reference["selected_local"], current["selected_local"])),
                fields=fields)
