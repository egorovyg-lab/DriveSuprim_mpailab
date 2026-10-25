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


def print_summary(report):
    print("Scenes: {scene_count}; valid: {valid_count}; invalid: {invalid_count}".format(**report))
    print("Metric      ours %     paper %    delta pp")
    for key in METRICS:
        actual = report["averages"][key]
        print(f"{LABELS[key]:<8} {100 * actual:9.4f} {100 * PAPER_R34[key]:11.4f} {100 * (actual - PAPER_R34[key]):11.4f}")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--csv", type=Path, help="CSV inside the run; defaults to the sole CSV in run_dir.")
    parser.add_argument("--compare-predictions", type=Path, help="Saved predictions from another run, e.g. v2.")
    parser.add_argument("--rescore-comparison", action="store_true", help="CPU scoring of alternative predictions; requires --compare-predictions.")
    args = parser.parse_args()
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
