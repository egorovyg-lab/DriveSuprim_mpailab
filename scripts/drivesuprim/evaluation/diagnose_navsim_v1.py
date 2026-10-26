"""Diagnose a completed NAVSIM v1 run without changing model/scorer weights.

Default: CSV/manifest audit using only Python's standard library.
Optional: compare saved predictions; --rescore-comparison evaluates alternative
trajectories with the same v1 cache and scorer, without GPU model inference.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

METRICS = [
    "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
    "time_to_collision_within_bound", "comfort", "pdm_score",
]
PAPER_R34 = dict(zip(METRICS, [0.978, 0.973, 0.867, 0.936, 1.0, 0.899]))
PAPER_HUMAN = dict(zip(METRICS, [1.0, 1.0, 0.875, 1.0, 0.999, 0.948]))
LABELS = dict(zip(METRICS, ["NC", "DAC", "EP", "TTC", "C", "PDMS"]))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_csv(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"token", "valid", *METRICS}.issubset(reader.fieldnames or []):
            raise ValueError("CSV is missing required NAVSIM v1 columns.")
        all_rows = list(reader)
    rows = [row for row in all_rows if not row["token"].startswith(("average_", "extended_pdm_score"))]
    if not rows:
        raise ValueError("CSV contains no scenario rows.")
    tokens = [row["token"] for row in rows]
    if len(tokens) != len(set(tokens)):
        raise ValueError("CSV contains duplicate scenario tokens.")
    valid = [row for row in rows if row["valid"].lower() in ("true", "1")]
    if not valid:
        raise ValueError("All scenario rows are invalid.")
    values = [{key: float(row[key]) for key in METRICS} for row in valid]
    if any(not math.isfinite(value) for row in values for value in row.values()):
        raise ValueError("CSV contains non-finite metrics.")
    if any(not -1e-8 <= value <= 1 + 1e-8 for row in values for value in row.values()):
        raise ValueError("Expected normalized metrics in [0, 1].")
    averages = {key: mean(row[key] for row in values) for key in METRICS}
    formula_error = max(abs(
        row["pdm_score"] - row["no_at_fault_collisions"] * row["drivable_area_compliance"] * (
            5 * row["ego_progress"] + 5 * row["time_to_collision_within_bound"] + 2 * row["comfort"]
        ) / 12
    ) for row in values)
    weighted_losses = {
        "NC_DAC_penalties": mean(1 - row["no_at_fault_collisions"] * row["drivable_area_compliance"] for row in values),
    }
    for name, metric, weight in [("EP", "ego_progress", 5), ("TTC", "time_to_collision_within_bound", 5), ("C", "comfort", 2)]:
        weighted_losses[name] = mean(
            row["no_at_fault_collisions"] * row["drivable_area_compliance"] * weight * (1 - row[metric]) / 12
            for row in values
        )
    return {
        "csv": str(path), "scene_count": len(rows), "valid_count": len(valid),
        "invalid_count": len(rows) - len(valid), "averages": averages,
        "max_formula_error": formula_error, "loss_from_perfect_score": weighted_losses,
        "tokens": tokens,
    }


def print_summary(report, paper_reference=PAPER_R34, reference_name="DriveSuprim ResNet34, paper Table 2"):
    print("Reference:", reference_name)
    print("Scenes: {scene_count}; valid: {valid_count}; invalid: {invalid_count}".format(**report))
    print("Metric      ours %     paper %    delta pp")
    for key in METRICS:
        actual = report["averages"][key]
        print(f"{LABELS[key]:<8} {100 * actual:9.4f} {100 * paper_reference[key]:11.4f} {100 * (actual - paper_reference[key]):11.4f}")
    print("Maximum per-scene PDMS formula error:", report["max_formula_error"])
    print("Loss from a perfect score (pp; these are NOT contributions to the paper gap):")
    for key, value in report["loss_from_perfect_score"].items():
        print(f"  {key}: {100 * value:.4f}")
    if report["max_formula_error"] > 1e-9:
        print("CHECK FAILED: saved scores do not match the stated v1 formula.")


def audit_manifest(run_dir, report):
    path = run_dir / "reproduction_manifest.json"
    if not path.is_file():
        print("No reproduction_manifest.json; checkpoint/source identity cannot be verified.")
        return
    manifest = json.loads(path.read_text())
    print("Recorded protocol:", manifest.get("protocol"))
    print("Recorded selection coefficients:", manifest.get("selection_coefficients"))
    print("Recorded comfort head:", manifest.get("comfort_prediction_head"))
    print("Recorded checkpoint SHA256:", manifest.get("checkpoint_sha256"))
    print("CSV checksum matches manifest:", sha256(report["csv"]) == manifest.get("csv_sha256"))
    print("Scene count matches manifest:", report["scene_count"] == manifest.get("expected_scene_count"))
    project_root = Path(__file__).resolve().parents[3]
    mismatches = [
        relative for relative, expected in manifest.get("source_sha256", {}).items()
        if not (project_root / relative).is_file() or sha256(project_root / relative) != expected
    ]
    print("Source files differing from recorded run:", mismatches)
    print("Note: equality of published rounded score alone does not establish protocol identity.")


def load_predictions(path):
    # User-owned saved NAVSIM predictions; importing their dataclasses requires the project environment.
    import pickle
    with Path(path).open("rb") as stream:
        raw = pickle.load(stream)
    return {token: {"trajectory": value["trajectory"]} for token, value in raw.items()}


def compare_predictions(reference, alternative, tokens):
    import numpy as np
    missing_reference = set(tokens) - set(reference)
    missing_alternative = set(tokens) - set(alternative)
    print("Missing saved v1 predictions:", len(missing_reference))
    print("Missing comparison predictions:", len(missing_alternative))
    if missing_reference or missing_alternative:
        raise ValueError("Prediction coverage differs; refusing to compare a smaller subset.")
    changed = [
        token for token in tokens
        if not np.array_equal(reference[token]["trajectory"].poses, alternative[token]["trajectory"].poses)
    ]
    print(f"Different trajectories: {len(changed)} / {len(tokens)} ({100 * len(changed) / len(tokens):.2f}%).")
    return changed


def rescore_comparison(run_dir, baseline_report, alternative, changed):
    import sys
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    import pandas as pd
    from omegaconf import OmegaConf
    from nuplan.planning.utils.multithreading.worker_utils import worker_map
    from navsim.planning.script.builders.worker_pool_builder import build_worker
    from navsim.planning.script.run_drivesuprim_navsim_v1 import read_cache_index, score_worker

    config_path = run_dir / "code" / "hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError("Saved Hydra configuration not found: " + str(config_path))
    cfg = OmegaConf.load(config_path)
    diagnostic_dir = run_dir / "diagnostics_selection_comparison"
    if diagnostic_dir.exists():
        raise FileExistsError("Comparison output already exists; move it aside before repeating.")
    diagnostic_dir.mkdir()
    cfg.output_dir = str(diagnostic_dir)
    cache_paths = read_cache_index(cfg, baseline_report["tokens"])
    by_log = {}
    for token in baseline_report["tokens"]:
        log_name = Path(cache_paths[token]).parents[2].name
        by_log.setdefault(log_name, []).append(token)
    jobs = [
        {"cfg": cfg, "tokens": sorted(tokens), "predictions": alternative, "cache_paths": cache_paths}
        for _, tokens in sorted(by_log.items())
    ]
    print("Rescoring alternative predictions with the SAME v1 cache/scorer; no GPU model inference.")
    records = worker_map(build_worker(cfg), score_worker, jobs)
    frame = pd.DataFrame(records).sort_values("token")
    out_path = diagnostic_dir / "comparison_predictions_v1_scores.csv"
    frame.to_csv(out_path, index=False)
    alternative_report = audit_csv(out_path)
    if alternative_report["scene_count"] != baseline_report["scene_count"] or alternative_report["invalid_count"]:
        raise RuntimeError("Incomplete comparison; inspect " + str(out_path))
    print_summary(alternative_report)
    before = baseline_report["averages"]["pdm_score"]
    after = alternative_report["averages"]["pdm_score"]
    print(f"Trajectory-set difference on the same scorer: {100 * (after - before):+.4f} pp.")
    serializable = {
        "baseline": {k: v for k, v in baseline_report.items() if k != "tokens"},
        "alternative": {k: v for k, v in alternative_report.items() if k != "tokens"},
        "different_trajectories": len(changed),
        "trajectory_set_difference_pp": 100 * (after - before),
        "interpretation": "Attribution to selection requires identical checkpoints/data; not proof of the exact paper protocol.",
    }
    (diagnostic_dir / "comparison_summary.json").write_text(json.dumps(serializable, indent=2))
    print("Saved diagnostic comparison to", diagnostic_dir)



def human_score_worker(args):
    from hydra.utils import instantiate
    from navsim.agents.human_agent import HumanAgent
    from navsim.planning.script.run_drivesuprim_navsim_v1 import make_loader, score_worker

    cfg = args[0]["cfg"]
    requested = sorted(token for item in args for token in item["tokens"])
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = [item["log_name"] for item in args]
    scene_filter.tokens = requested
    loader = make_loader(cfg, scene_filter)
    if set(loader.tokens) != set(requested):
        raise RuntimeError("Human baseline scene coverage differs from the baseline run.")
    agent = HumanAgent()  # Official human agent: 8 future poses, 0.5 s interval.
    predictions = {}
    for index, token in enumerate(requested):
        scene = loader.get_scene_from_token(token)
        predictions[token] = {"trajectory": agent.compute_trajectory(None, scene)}
        if (index + 1) % 500 == 0:
            print(f"Human trajectories prepared: {index + 1}/{len(requested)}", flush=True)
    return score_worker([{
        "cfg": cfg, "tokens": requested, "predictions": predictions,
        "cache_paths": args[0]["cache_paths"],
    }])


def run_human_baseline(run_dir, baseline_report):
    import sys
    from datetime import datetime
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    import pandas as pd
    from omegaconf import OmegaConf
    from nuplan.planning.utils.multithreading.worker_utils import worker_map
    from navsim.planning.script.builders.worker_pool_builder import build_worker
    from navsim.planning.script.run_drivesuprim_navsim_v1 import read_cache_index

    config_path = run_dir / "code" / "hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError("Saved Hydra configuration not found: " + str(config_path))
    cfg = OmegaConf.load(config_path)
    if cfg.scorer._target_ != "navsim.evaluate.navsim_v1.pdm_scorer.PDMScorer":
        raise ValueError("Expected the original run's NAVSIM v1 scorer.")
    cache_paths = read_cache_index(cfg, baseline_report["tokens"])
    out_dir = run_dir / "diagnostics_human_baseline" / datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    cfg.output_dir = str(out_dir)
    by_log = {}
    for token in baseline_report["tokens"]:
        by_log.setdefault(Path(cache_paths[token]).parents[2].name, []).append(token)
    jobs = [
        {"cfg": cfg, "log_name": log_name, "tokens": sorted(tokens), "cache_paths": cache_paths}
        for log_name, tokens in sorted(by_log.items())
    ]
    print("Evaluating HumanAgent on the SAME v1 cache/scorer; no DriveSuprim checkpoint or GPU inference.", flush=True)
    records = worker_map(build_worker(cfg), human_score_worker, jobs)
    frame = pd.DataFrame(records).sort_values("token")
    csv_path = out_dir / "human_navsim_v1_scores.csv"
    frame.to_csv(csv_path, index=False)
    report = audit_csv(csv_path)
    if set(report["tokens"]) != set(baseline_report["tokens"]) or report["invalid_count"]:
        raise RuntimeError("Human baseline incomplete; inspect " + str(csv_path))
    print_summary(report, paper_reference=PAPER_HUMAN, reference_name="Human Agent, paper Table 2")
    summary = {
        "reference": "DriveSuprim paper v3, Table 2, Human Agent",
        "paper_metrics": PAPER_HUMAN,
        "observed": {key: value for key, value in report.items() if key != "tokens"},
        "cache_manifest_sha256": sha256(Path(cfg.metric_cache_path) / "navsim_v1_manifest.json"),
        "human_agent_sha256": sha256(project_root / "navsim/agents/human_agent.py"),
        "scope": "Checks the scoring/data pipeline independently of DriveSuprim predictions. Agreement alone does not prove full protocol identity.",
    }
    (out_dir / "human_baseline_summary.json").write_text(json.dumps(summary, indent=2))
    print("Human baseline complete. Saved results:", out_dir, flush=True)




def vocab_difference_stats(candidate, reference):
    """Measure index-wise geometry differences; no nearest-neighbor assumptions."""
    import numpy as np
    a = np.asarray(candidate)
    b = np.asarray(reference)
    report = {"candidate_dtype": str(a.dtype), "reference_dtype": str(b.dtype), "same_shape": a.shape == b.shape}
    if a.shape != b.shape:
        report.update(candidate_shape=list(a.shape), reference_shape=list(b.shape))
        return report
    if a.ndim != 3 or a.shape[-1] != 3 or not a.size:
        raise ValueError("Expected a nonempty [trajectory, pose, 3] vocabulary.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Vocabulary contains non-finite coordinates.")
    delta = a.astype(np.float64) - b.astype(np.float64)
    position_error = np.linalg.norm(delta[..., :2], axis=-1)
    heading_error = np.abs(np.arctan2(np.sin(delta[..., 2]), np.cos(delta[..., 2])))
    max_per_trajectory = position_error.max(axis=1)
    report.update(
        exact_equal=bool(np.array_equal(a, b)),
        equal_after_reference_dtype_cast=bool(np.array_equal(a, b.astype(a.dtype))),
        max_position_error_m=float(position_error.max()),
        mean_position_error_m=float(position_error.mean()),
        rms_position_error_m=float(np.sqrt(np.mean(position_error ** 2))),
        max_endpoint_error_m=float(position_error[:, -1].max()),
        max_heading_error_rad=float(heading_error.max()),
        max_heading_error_deg=float(np.degrees(heading_error.max())),
        fraction_trajectories_within_1um=float(np.mean(max_per_trajectory <= 1e-6)),
        fraction_trajectories_within_0_1mm=float(np.mean(max_per_trajectory <= 1e-4)),
        fraction_trajectories_within_1mm=float(np.mean(max_per_trajectory <= 1e-3)),
        fraction_trajectories_within_1cm=float(np.mean(max_per_trajectory <= 1e-2)),
    )
    return report


def sampled_vocab_index_match(candidate, reference, sample_count=64):
    """Position-only nearest-neighbor diagnostic on a deterministic sample."""
    import numpy as np
    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError("Vocabularies must have matching [N, H, 3] shapes.")
    indices = np.linspace(0, len(a) - 1, min(sample_count, len(a)), dtype=int)
    matches = 0
    nearest_distances = []
    for index in indices:
        squared_distance = ((b[..., :2] - a[index, None, :, :2]) ** 2).sum(axis=(1, 2))
        nearest = int(np.argmin(squared_distance))
        matches += int(nearest == int(index))
        nearest_distances.append(float(np.sqrt(squared_distance[nearest] / a.shape[1])))
    return {
        "sample_count": len(indices), "same_index_nearest_count": int(matches),
        "same_index_nearest_fraction": float(matches / len(indices)),
        "largest_nearest_rms_position_error_m": max(nearest_distances),
        "scope": "Position-only nearest matching on a sample, not a proof of all-index permutation identity.",
    }


def run_vocab_audit(run_dir):
    import sys
    from datetime import datetime
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(run_dir / "code" / "hydra" / "config.yaml")
    checkpoint = torch.load(cfg.agent.checkpoint_path, map_location=torch.device("cpu"))
    state = {key.replace("agent.", ""): value for key, value in checkpoint["state_dict"].items()}
    teacher = state["model.teacher.model._trajectory_head.vocab"].detach().cpu().numpy()
    student = state["model.student.model._trajectory_head.vocab"].detach().cpu().numpy()
    configured = np.load(cfg.agent.config.vocab_path)
    report = {
        "teacher_vs_configured": vocab_difference_stats(teacher, configured),
        "student_vs_configured": vocab_difference_stats(student, configured),
        "teacher_vs_student": vocab_difference_stats(teacher, student),
        "teacher_vs_student_index_sample": sampled_vocab_index_match(teacher, student),
        "checkpoint_sha256": sha256(cfg.agent.checkpoint_path),
        "scope": "Read-only numeric audit; checkpoint and model vocabulary are not modified.",
        "ema_note": "The released update_teacher() updates all parameters, including the non-trainable vocabulary. A difference may be floating-point EMA drift; its cause and score effect are not proven by this audit.",
    }
    for name in ["teacher_vs_configured", "student_vs_configured", "teacher_vs_student", "teacher_vs_student_index_sample"]:
        print(name + ":", json.dumps(report[name], sort_keys=True), flush=True)
    out = run_dir / "diagnostics_vocab" / datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    out.mkdir(parents=True, exist_ok=False)
    (out / "vocab_audit.json").write_text(json.dumps(report, indent=2))
    print("Vocabulary audit complete. Saved report:", out, flush=True)


def checkpoint_key_report(expected_keys, missing_keys, unexpected_keys):
    missing = sorted(missing_keys)
    unexpected = sorted(unexpected_keys)
    return {
        "expected_key_count": len(expected_keys),
        "missing_keys": missing,
        "missing_teacher_keys": [key for key in missing if key.startswith("model.teacher.")],
        "missing_student_keys": [key for key in missing if key.startswith("model.student.")],
        "unexpected_keys": unexpected,
    }


def run_checkpoint_audit(run_dir, baseline_report):
    import sys
    from datetime import datetime
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    import torch
    import numpy as np
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    expected_tokens_sha256 = "8d42fef68542f095d0e03894fdaf38babfdc89f3fd7cdc26080c3a6009cb4108"
    expected_checkpoint_sha256 = "73db58e4ed4353f2238623f11c9ab409cf02470929a585a1b425266077d52a21"
    token_digest = hashlib.sha256("\n".join(sorted(baseline_report["tokens"])).encode()).hexdigest()
    print("Run token set matches official NAVSIM v1.1 navtest:", token_digest == expected_tokens_sha256, flush=True)
    print("Run tokens SHA256:", token_digest, flush=True)
    cfg = OmegaConf.load(run_dir / "code" / "hydra" / "config.yaml")
    if cfg.agent.config.training or not cfg.agent.config.only_ori_input:
        raise ValueError("Checkpoint audit requires the saved evaluation config.")
    if cfg.agent.config.backbone_type != "resnet34":
        raise ValueError("This official checkpoint fingerprint is for ResNet34 only.")
    path = Path(cfg.agent.checkpoint_path)
    current_digest = sha256(path)
    print("Checkpoint matches public R34 file:", current_digest == expected_checkpoint_sha256, flush=True)
    print("Loading checkpoint and constructing agent on CPU; no inference.", flush=True)
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    if "state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no Lightning state_dict.")
    # Match DriveSuprimAgent.initialize exactly, but retain its compatibility report.
    state = {key.replace("agent.", ""): value for key, value in checkpoint["state_dict"].items()}
    agent = instantiate(cfg.agent)
    expected = agent.state_dict()
    try:
        incompatible = agent.load_state_dict(state, strict=False)
    except RuntimeError as error:
        print("CHECKPOINT SHAPE/TYPE FAILURE:", error, flush=True)
        raise
    report = checkpoint_key_report(expected, incompatible.missing_keys, incompatible.unexpected_keys)
    loaded = agent.state_dict()
    unequal = []
    for key in expected.keys() & state.keys():
        value = state[key]
        if torch.is_tensor(value):
            value = value.to(device="cpu", dtype=loaded[key].dtype)
            if not torch.equal(loaded[key].cpu(), value):
                unequal.append(key)
    report["loaded_values_differ_from_checkpoint"] = sorted(unequal)
    vocab = torch.from_numpy(np.load(cfg.agent.config.vocab_path))
    vocab_reports = []
    for key in sorted(state):
        if key.endswith("_trajectory_head.vocab"):
            value = state[key].cpu()
            vocab_reports.append({
                "key": key, "checkpoint_shape": list(value.shape),
                "configured_vocab_shape": list(vocab.shape),
                "equals_configured_vocab": bool(value.shape == vocab.shape and torch.equal(value, vocab.to(dtype=value.dtype))),
            })
    report["vocabularies"] = vocab_reports
    prefix = "model.teacher.model._trajectory_head.heads."
    report["teacher_metric_heads"] = sorted({key[len(prefix):].split(".")[0] for key in state if key.startswith(prefix)})
    manifest_path = run_dir / "reproduction_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    report.update({
        "tokens_sha256": token_digest,
        "tokens_match_official_navtest": token_digest == expected_tokens_sha256,
        "checkpoint_sha256": current_digest,
        "checkpoint_matches_public_r34": current_digest == expected_checkpoint_sha256,
        "checkpoint_matches_recorded_run": current_digest == previous.get("checkpoint_sha256"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "checkpoint_lightning_version": checkpoint.get("pytorch-lightning_version"),
        "has_hyper_parameters": "hyper_parameters" in checkpoint,
        "teacher_weights_complete": not report["missing_teacher_keys"],
        "comparison_scope": "Checks loading now using the saved config; it does not retrospectively prove every byte loaded in an earlier process.",
    })
    out = run_dir / "diagnostics_checkpoint" / datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoint_audit.json").write_text(json.dumps(report, indent=2, default=str))
    print("Expected state keys:", report["expected_key_count"], flush=True)
    print("Missing keys:", len(report["missing_keys"]), report["missing_keys"][:30], flush=True)
    print("Missing teacher keys:", len(report["missing_teacher_keys"]), flush=True)
    print("Unexpected keys:", len(report["unexpected_keys"]), report["unexpected_keys"][:30], flush=True)
    print("Loaded values differing from checkpoint:", len(unequal), flush=True)
    print("Teacher metric heads:", report["teacher_metric_heads"], flush=True)
    print("Checkpoint epoch/global_step:", report["checkpoint_epoch"], report["checkpoint_global_step"], flush=True)
    print("Vocab comparison:", vocab_reports, flush=True)
    print("Checkpoint audit complete. Saved report:", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--csv", type=Path, help="CSV inside the run; defaults to the sole CSV in run_dir.")
    parser.add_argument("--compare-predictions", type=Path, help="Saved predictions from another run, e.g. v2.")
    parser.add_argument("--rescore-comparison", action="store_true", help="CPU scoring of alternative predictions; requires --compare-predictions.")
    parser.add_argument("--human-baseline", action="store_true", help="Evaluate the official HumanAgent on the same v1 cache, without DriveSuprim inference.")
    parser.add_argument("--checkpoint-audit", action="store_true", help="Audit state_dict loading and official R34/split fingerprints on CPU; no inference.")
    parser.add_argument("--vocab-audit", action="store_true", help="Read-only numeric teacher/student vocabulary audit on CPU; no model construction or inference.")
    args = parser.parse_args()
    if args.vocab_audit and (args.checkpoint_audit or args.human_baseline or args.compare_predictions or args.rescore_comparison):
        parser.error("--vocab-audit is a separate diagnostic; omit other mode flags.")
    if args.checkpoint_audit and (args.human_baseline or args.compare_predictions or args.rescore_comparison):
        parser.error("--checkpoint-audit is a separate diagnostic; omit other mode flags.")
    if args.human_baseline and (args.compare_predictions or args.rescore_comparison):
        parser.error("--human-baseline is a separate diagnostic; omit comparison flags.")
    if args.rescore_comparison and not args.compare_predictions:
        parser.error("--rescore-comparison requires --compare-predictions")
    run_dir = args.run_dir.resolve()
    csv_paths = sorted(run_dir.glob("*.csv"))
    if args.csv is not None:
        csv_path = args.csv
    elif len(csv_paths) == 1:
        csv_path = csv_paths[0]
    else:
        parser.error("Expected exactly one CSV in run_dir; specify --csv.")
    report = audit_csv(csv_path)
    print_summary(report)
    audit_manifest(run_dir, report)
    if args.vocab_audit:
        run_vocab_audit(run_dir)
        return
    if args.checkpoint_audit:
        run_checkpoint_audit(run_dir, report)
        return
    if args.human_baseline:
        run_human_baseline(run_dir, report)
        return
    if args.compare_predictions is not None:
        reference_path = run_dir / "predictions_navsim_v1.pkl"
        if not reference_path.is_file() or not args.compare_predictions.is_file():
            parser.error("Prediction file missing. Both the v1 predictions and comparison file must exist.")
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        reference = load_predictions(reference_path)
        alternative = load_predictions(args.compare_predictions)
        print("Comparing supplied trajectory files. Verify they came from the same checkpoint before attributing changes to selection.")
        changed = compare_predictions(reference, alternative, report["tokens"])
        if args.rescore_comparison:
            rescore_comparison(run_dir, report, alternative, changed)


if __name__ == "__main__":
    main()
