"""Synthetic text/image checks of the downloaded Drex; no driving data."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

from runtime import environment, read, remaining_bytes, write


def cases(image_factory):
    text = dict(state="The identifier is BLUE.", question=dict(type="choice",
        instructions="Which identifier is explicitly stated?", criteria={"RED": None, "BLUE": None}))
    question = dict(type="choice", instructions="What is the uniform background colour in the image?",
                    criteria={"red": "Red", "blue": "Blue", "green": "Green"})
    rows = [("text", text, "BLUE")]
    for colour in ("red", "blue"):
        rows.append(("image_" + colour, dict(state="Identify the background colour.", question=question,
                    images=[image_factory(colour)]), colour))
    return rows + [("text_repeat", text, "BLUE")]


def exercise(infer, image_factory, on_result=lambda _: None):
    results = []
    for name, row, expected in cases(image_factory):
        measured = infer(row)  # Only inference fields, never the expected answer.
        p = measured["probabilities"]
        keys = list(row["question"]["criteria"])
        if (len(p) != len(keys) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p)
                or abs(sum(p) - 1) > 1e-5):
            raise ValueError("Invalid probability distribution for " + name)
        choice = keys[max(range(len(keys)), key=p.__getitem__)]
        results.append(dict(case=name, expected=expected, choice=choice, passed=choice == expected, **measured))
        on_result(results[-1])
    delta = max(abs(a-b) for a, b in zip(results[0]["probabilities"], results[-1]["probabilities"]))
    return dict(passed=all(r["passed"] for r in results), cases=results, repeat_max_abs_diff=delta)


def run(root):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = root / "runs" / ("smoke-" + stamp)
    report = dict(status="running", report_path=str(out / "report.json"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note="Synthetic runtime test only. Size checks, not full weight checksums. No NAVSIM, training or safety claim.")
    write(out / "report.json", report)
    started = time.monotonic()
    phase = "environment"
    try:
        import torch
        from PIL import Image
        from drex_max.model import DecisionModel
        report["environment"] = environment(root, True)
        phase = "model_files"
        plan = read(root / "model_plan.json")
        target = root / "models/drex-max-v1"
        if remaining_bytes(target, plan["files"]):
            raise RuntimeError("Model files missing or sizes changed; smoke does not download weights")
        decision = read(target / "decision_config.json")
        if decision.get("format_version") != 1 or decision.get("attention_mode", "noncausal_full_attention") != "noncausal_full_attention":
            raise RuntimeError("Expected official Drex non-causal attention protocol")
        report["revision"] = plan["revision"]
        phase = "load_bf16_model"
        print("Loading complete Drex Max in BF16, batch=1", flush=True)
        torch.cuda.reset_peak_memory_stats()
        before = time.monotonic()
        model = DecisionModel(target, device="cuda:0", max_image_pixels=256 * 256)
        torch.cuda.synchronize()
        if model.readout.weight.dtype != torch.bfloat16:
            raise RuntimeError("Unexpected readout precision")
        report["load_seconds"] = time.monotonic() - before
        print("Model loaded; running synthetic requests", flush=True)
        phase = "text_and_image_inference"

        def infer(row):
            # prepare for token accounting; timed predict includes its own preprocessing.
            batch = model.prepare([row], max_length=2048)
            tokens = int(batch.inputs["input_ids"].shape[1])
            image_present = "pixel_values" in batch.inputs
            del batch
            if bool(row.get("images")) != image_present:
                raise RuntimeError("Image tokens are missing/unexpected")
            torch.cuda.synchronize()
            tick = time.monotonic()
            p = model.predict([row], batch_size=1, max_length=2048)[0]
            torch.cuda.synchronize()
            return dict(probabilities=p, input_tokens=tokens, has_image_pixels=image_present,
                        seconds=time.monotonic() - tick)

        report["partial_cases"] = []
        def completed(row):
            report["partial_cases"].append(row)
            write(out / "report.json", report)
            print(json.dumps(row), flush=True)
        result = exercise(infer, lambda colour: Image.new("RGB", (256, 256), colour), completed)
        report.update(result, status="complete" if result["passed"] else "semantic_smoke_failed",
                      peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                      peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    except Exception as exc:
        report.update(status="failed", phase=phase, error=repr(exc), traceback=traceback.format_exc())
    report["seconds"] = time.monotonic() - started
    write(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report["status"] == "complete"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    raise SystemExit(0 if run(args.root.expanduser().resolve()) else 1)
