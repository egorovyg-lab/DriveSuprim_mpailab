"""Resumable synthetic diagram benchmark, never a NAVSIM evaluation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

from runtime import environment, read, remaining_bytes, write
import synthetic as task

CONFIG = dict(protocol="synthetic_diagrams_v1", scenes=384, seed=task.SEED,
              tasks=task.TASKS, max_image_pixels=2048*1024, max_length=8192,
              precision="bf16", batch_size=1)
NOTE = ("Synthetic axis-aligned vehicle footprints on 2D diagrams; no kinematics, "
        "traffic prediction, NAVSIM, training or driving safety claim. Blank images "
        "are an information-ablation control. Model weights checked by size only.")


def digest(value):
    return hashlib.sha256(task.canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path, value):
    value = json.loads(task.canonical(value))
    if path.exists():
        if read(path) != value:
            raise RuntimeError("Frozen inputs changed: " + str(path))
    else:
        write(path, value)


def prepare(root, out):
    identity = dict(config=CONFIG, model_plan=read(root / "model_plan.json"),
        environment_lock_sha256=sha(root / "requirements.lock.txt"),
        sources={n: sha(Path(__file__).with_name(n)) for n in ("night.py", "synthetic.py", "runtime.py")})
    freeze(out / "identity.json", identity)
    worlds = [task.scene(i) for i in range(CONFIG["scenes"])]
    freeze(out / "scenes.json", worlds)
    return worlds


def payload(world, mode, ordering):
    ids = task.order(world, ordering)
    row = task.request(world, mode, ids)
    image = None
    image_sha = None
    if mode != "text":
        from PIL import Image
        image = task.board(world, ids)
        if mode == "blank":
            image = Image.new("RGB", image.size, "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_sha = hashlib.sha256(buffer.getvalue()).hexdigest()
    request_sha = digest(dict(row=row, image_sha256=image_sha, candidate_ids=ids))
    if image is not None:
        row["images"] = [image]
    return row, ids, request_sha


def validate_probabilities(p, n):
    if (len(p) != n or any(not math.isfinite(x) or not 0 <= x <= 1 for x in p)
            or abs(sum(p) - 1) > 1e-5):
        raise ValueError("Invalid model probability distribution")


def load_result(path, request_sha=None):
    value = read(path)
    checksum = value.pop("checksum")
    if digest(value) != checksum:
        raise RuntimeError("Changed result: " + str(path))
    if request_sha is not None and value["request_sha256"] != request_sha:
        raise RuntimeError("Changed request: " + str(path))
    if value["status"] != "ok":
        raise RuntimeError("Recorded technical failure; inspect before resuming: " + str(path))
    validate_probabilities(value["probabilities"], value["k"])
    return value


def save_result(path, value):
    write(path, dict(value, checksum=digest(value)))


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered)-1)*q)]


def section(rows):
    n = len(rows)
    moving = [r for r in rows if r["grade"]["best_progress"] > 0]
    warm = [r["seconds"] for r in rows if not r.get("cold_call", False)]
    return dict(requests=n, unsafe=sum(not r["grade"]["safe"] for r in rows),
        optimal=sum(r["grade"]["optimal"] for r in rows),
        stopped=sum(r["grade"]["baseline_selected"] for r in rows),
        optimal_fraction=sum(r["grade"]["optimal"] for r in rows)/n if n else None,
        forward_feasible_requests=len(moving),
        forward_progress_capture=sum(r["grade"]["safe_progress"]/r["grade"]["best_progress"] for r in moving)/len(moving) if moving else None,
        warm_latency_p50_seconds=quantile(warm, .5), warm_latency_p95_seconds=quantile(warm, .95),
        max_input_tokens=max((r.get("input_tokens", 0) for r in rows), default=0))


def summarize(rows, expected):
    groups = defaultdict(list)
    for r in rows:
        groups[r["mode"] + "/" + r["ordering"]].append(r)
    by_scene = defaultdict(dict)
    for r in rows:
        by_scene[r["scene"]][(r["mode"], r["ordering"])] = r
    pairs = {}
    for ordering in ("reversed", "permuted", "repeat"):
        found = [(v[("image", "canonical")], v[("image", ordering)]) for v in by_scene.values()
                 if ("image", "canonical") in v and ("image", ordering) in v]
        pairs[ordering] = dict(pairs=len(found), choice_flips=sum(a["chosen_id"] != b["chosen_id"] for a, b in found),
            safety_or_progress_flips=sum((a["grade"]["safe"], a["grade"]["safe_progress"]) !=
                                        (b["grade"]["safe"], b["grade"]["safe_progress"]) for a, b in found))
        if ordering == "repeat":
            pairs[ordering]["max_probability_abs_diff"] = max((abs(x-y) for a, b in found
                for x, y in zip(a["probabilities"], b["probabilities"])), default=None)
    return dict(status="complete" if len(rows) == expected else "partial", completed_requests=len(rows),
        expected_requests=expected, completed_scenes=len({r["scene"] for r in rows}),
        by_condition={k: section(v) for k, v in sorted(groups.items())},
        by_k_and_condition={str(k): {key: section([r for r in v if r["k"] == k]) for key, v in sorted(groups.items())}
                            for k in task.COUNTS},
        by_family_and_condition={family: {key: section([r for r in v if r["family"] == family])
            for key, v in sorted(groups.items())} for family in task.FAMILIES},
        order_and_repeat=pairs, note=NOTE)


def exercise(worlds, out, infer, max_new=None, budget_seconds=6*3600):
    started, new = time.monotonic(), 0
    rows = []
    stop = False
    expected_paths = {f"{w['index']:04d}-{mode}-{ordering}.json" for w in worlds for mode, ordering in task.TASKS}
    if {p.name for p in (out / "results").glob("*.json")} - expected_paths:
        raise RuntimeError("Unexpected cached request files")
    for world in worlds:
        for mode, ordering in task.TASKS:
            path = out / "results" / f"{world['index']:04d}-{mode}-{ordering}.json"
            if path.exists():
                # Recreate and hash the exact row and pixels, including repeat inputs.
                _, _, request_sha = payload(world, mode, ordering)
                rows.append(load_result(path, request_sha))
                continue
            if stop or (max_new is not None and new >= max_new) or time.monotonic()-started >= budget_seconds:
                stop = True
                continue
            row, ids, request_sha = payload(world, mode, ordering)
            result = dict(status="ok", scene=world["index"], family=world["family"], k=len(ids),
                mode=mode, ordering=ordering, candidate_ids=ids, request_sha256=request_sha)
            try:
                measured = infer(row)
                probabilities = [float(p) for p in measured["probabilities"]]
                validate_probabilities(probabilities, len(ids))
                chosen = ids[max(range(len(ids)), key=probabilities.__getitem__)]
                result.update(measured, probabilities=probabilities, chosen_id=chosen, grade=task.grade(world, chosen))
            except Exception as exc:
                result.update(status="technical_failure", error=repr(exc), traceback=traceback.format_exc())
                save_result(path, result)
                report = summarize(rows, len(worlds)*len(task.TASKS))
                report.update(status="failed", failed_request=str(path), error=repr(exc))
                write(out / "report.json", report)
                raise
            save_result(path, result)
            rows.append(result)
            new += 1
            if new % 14 == 0:
                print(f"NIGHT completed={len(rows)} new={new} scene={world['index']} k={len(ids)}", flush=True)
                write(out / "report.json", summarize(rows, len(worlds)*len(task.TASKS)))
    report = summarize(rows, len(worlds)*len(task.TASKS))
    report.update(new_requests=new, invocation_seconds=time.monotonic()-started,
                  stop_reason="budget_or_pilot_limit" if stop else "all_requests_complete")
    write(out / "report.json", report)
    return report


def run(root, out, worlds, max_new):
    import torch
    from drex_max.model import DecisionModel
    env = environment(root, True)
    freeze(out / "inference_environment.json", {k: env[k] for k in
        ("python", "torch", "cuda_build", "transformers", "gpu", "vram_bytes")})
    target = root / "models/drex-max-v1"
    if remaining_bytes(target, read(root / "model_plan.json")["files"]):
        raise RuntimeError("Missing/changed model files; this benchmark never downloads weights")
    cfg = read(target / "decision_config.json")
    if cfg.get("format_version") != 1 or cfg.get("attention_mode", "noncausal_full_attention") != "noncausal_full_attention":
        raise RuntimeError("Unexpected model attention protocol")
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    write(out / "attempts" / (attempt + ".json"), env)
    torch.cuda.reset_peak_memory_stats()
    print("Loading Drex BF16, batch=1, image limit=2 Mpixels, context limit=8192", flush=True)
    tick = time.monotonic()
    model = DecisionModel(target, device="cuda:0", max_image_pixels=CONFIG["max_image_pixels"])
    torch.cuda.synchronize()
    load_seconds = time.monotonic()-tick
    if model.readout.weight.dtype != torch.bfloat16:
        raise RuntimeError("Unexpected readout precision")
    calls = 0
    def infer(row):
        nonlocal calls
        start = time.monotonic()
        batch = model.prepare([row], max_length=CONFIG["max_length"])
        tokens = int(batch.inputs["input_ids"].shape[1])
        image_present = "pixel_values" in batch.inputs
        del batch
        if bool(row.get("images")) != image_present:
            raise RuntimeError("Image inputs dropped/unexpected")
        if tokens >= CONFIG["max_length"]:
            raise RuntimeError("Context reached limit; possible truncation")
        torch.cuda.synchronize()
        before = time.monotonic()
        p = model.predict([row], batch_size=1, max_length=CONFIG["max_length"])[0]
        torch.cuda.synchronize()
        calls += 1
        return dict(probabilities=p, input_tokens=tokens, has_image_pixels=image_present,
                    seconds=time.monotonic()-before, prepare_and_predict_seconds=time.monotonic()-start,
                    cold_call=calls == 1)
    report = exercise(worlds, out, infer, max_new=max_new)
    report.update(load_seconds=load_seconds, peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    write(out / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "run", "report"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--max-new", type=int)
    a = p.parse_args()
    name = os.environ.get("DREX_NIGHT_NAME", "synthetic_night_v1")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
        raise ValueError("Invalid DREX_NIGHT_NAME")
    root = a.root.expanduser().resolve()
    out = root / "runs" / name
    worlds = prepare(root, out)
    if a.stage == "report":
        result = exercise(worlds, out, lambda _: None, max_new=0)
    elif a.stage == "plan":
        for world in worlds[:4]:
            task.board(world, task.order(world, "canonical")).save(out / f"preview-k{len(world['candidates'])}.png")
        result = dict(status="planned", scenes=len(worlds), requests=len(worlds)*len(task.TASKS),
            pilot_new_requests=28, counts=task.COUNTS, families=task.FAMILIES, config=CONFIG, output=str(out), note=NOTE)
        write(out / "plan.json", result)
    else:
        try:
            result = run(root, out, worlds, a.max_new)
        except Exception as exc:
            write(out / "failure.json", dict(error=repr(exc), traceback=traceback.format_exc()))
            raise
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
