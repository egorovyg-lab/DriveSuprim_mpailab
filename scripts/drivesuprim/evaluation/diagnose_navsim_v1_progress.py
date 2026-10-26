"""Compare cached PDM reference trajectories without model inference or cache writes.

This is a controlled diagnostic, not a new benchmark configuration. For each
selected scene it first replays model/human baseline metrics, then changes only
the cached reference trajectory, and finally its centerline as well.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
METRICS = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
           "time_to_collision_within_bound", "comfort", "pdm_score"]


def paired_metrics(raw_reference, raw_prediction, reference_multiplier, prediction_multiplier, ttc, comfort, threshold):
    """NAVSIM v1.1 normalization for a reference/prediction pair only."""
    if not all(math.isfinite(value) for value in (raw_reference, raw_prediction, reference_multiplier, prediction_multiplier, ttc, comfort, threshold)):
        raise ValueError("Non-finite input to pair normalization.")
    denominator = max(raw_reference * reference_multiplier, raw_prediction * prediction_multiplier)
    if denominator > threshold:
        progress = raw_prediction * prediction_multiplier / denominator
    else:
        progress = 1.0 if prediction_multiplier != 0.0 else 0.0
    score = prediction_multiplier * (5.0 * progress + 5.0 * ttc + 2.0 * comfort) / 12.0
    return progress, score, denominator


def read_rows(path):
    with Path(path).open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if not row["token"].startswith("average_")]
    result = {}
    for row in rows:
        token = row["token"]
        if token in result:
            raise ValueError("Duplicate CSV token: " + token)
        if row["valid"].lower() not in ("true", "1"):
            raise ValueError("Invalid CSV row: " + token)
        metrics = {key: float(row[key]) for key in METRICS}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("Non-finite CSV row: " + token)
        result[token] = metrics
    return result


def discover_cache(root):
    paths = {}
    for path in root.glob("*/*/*/metric_cache.pkl"):
        token = path.parent.name
        if token in paths:
            raise ValueError("Duplicate cache token: " + token)
        paths[token] = str(path)
    return paths


def describe_cache(path):
    import lzma
    import pickle
    with lzma.open(path, "rb") as stream:
        cache = pickle.load(stream)
    return {
        "example_file": str(path), "class": type(cache).__module__ + "." + type(cache).__name__,
        "fields": sorted(vars(cache)),
        "observation_samples": getattr(cache.observation, "_observation_samples", None),
        "reference_horizon_s": cache.trajectory.end_time.time_s - cache.trajectory.start_time.time_s,
    }


def comparison_worker(args):
    import lzma
    import pickle
    import traceback
    import numpy as np
    from hydra.utils import instantiate
    from navsim.agents.human_agent import HumanAgent
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.evaluate.navsim_v1.pdm_enums import MultiMetricIndex, WeightedMetricIndex
    from navsim.planning.script.run_drivesuprim_navsim_v1 import make_loader
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array

    cfg = args[0]["cfg"]
    tokens = sorted(token for item in args for token in item["tokens"])
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = [item["log_name"] for item in args]
    scene_filter.tokens = tokens
    loader = make_loader(cfg, scene_filter)
    if set(loader.tokens) != set(tokens):
        raise RuntimeError("Scene loader coverage differs from diagnostic sample.")
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    human = HumanAgent()
    data = args[0]
    rows = []
    for token in tokens:
        try:
            with lzma.open(data["base_paths"][token], "rb") as stream:
                base = pickle.load(stream)
            with lzma.open(data["other_paths"][token], "rb") as stream:
                other = pickle.load(stream)
            initial_delta = float(np.max(np.abs(
                ego_state_to_state_array(base.ego_state) - ego_state_to_state_array(other.ego_state)
            )))
            if not math.isfinite(initial_delta) or initial_delta > 1e-4 or base.ego_state.time_point.time_us != other.ego_state.time_point.time_us:
                raise ValueError("Caches disagree about initial ego state/time: " + str(initial_delta))
            start = base.ego_state.time_point
            end_us = start.time_us + int(simulator.proposal_sampling.time_horizon * 1e6)
            for reference in [base.trajectory, other.trajectory]:
                if reference.start_time.time_us > start.time_us or reference.end_time.time_us < end_us:
                    raise ValueError("Reference trajectory does not cover the full scoring horizon.")
            base.observation[49]
            model = data["predictions"][token]["trajectory"]
            human_trajectory = human.compute_trajectory(None, loader.get_scene_from_token(token))
            trajectories = [
                base.trajectory, other.trajectory,
                transform_trajectory(model, base.ego_state),
                transform_trajectory(human_trajectory, base.ego_state),
            ]
            states = np.stack([
                get_trajectory_as_array(trajectory, simulator.proposal_sampling, start)
                for trajectory in trajectories
            ])
            simulated = simulator.simulate_proposals(states, base.ego_state)
            # Metrics apart from progress are proposal-local. Do NOT use the
            # 4-proposal aggregate score: every reported EP/PDMS below is rebuilt
            # with a 2-proposal denominator, as in the original evaluator.
            scorer.score_proposals(
                simulated, base.observation, base.centerline, base.route_lane_ids, base.drivable_area_map,
            )
            nc = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION].copy()
            dac = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA].copy()
            ttc = scorer._weighted_metrics[WeightedMetricIndex.TTC].copy()
            comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE].copy()
            raw_current = scorer._progress_raw.copy()
            scorer._centerline = other.centerline
            scorer._calculate_progress()
            raw_other_centerline = scorer._progress_raw.copy()
            threshold = scorer._config.progress_distance_threshold
            same_line = base.centerline.linestring.wkb == other.centerline.linestring.wkb
            for actor, pred_idx in [("model", 2), ("human", 3)]:
                expected = data["expected_" + actor][token]
                actor_rows = []
                for variant, ref_idx, raw in [
                    ("current_reference", 0, raw_current),
                    ("other_reference_only", 1, raw_current),
                    ("other_reference_and_centerline", 1, raw_other_centerline),
                ]:
                    progress, pdms, denominator = paired_metrics(
                        float(raw[ref_idx]), float(raw[pred_idx]),
                        float(nc[ref_idx] * dac[ref_idx]), float(nc[pred_idx] * dac[pred_idx]),
                        float(ttc[pred_idx]), float(comfort[pred_idx]), threshold,
                    )
                    row = {
                        "token": token, "actor": actor, "variant": variant, "valid": True,
                        "no_at_fault_collisions": float(nc[pred_idx]),
                        "drivable_area_compliance": float(dac[pred_idx]),
                        "ego_progress": progress, "time_to_collision_within_bound": float(ttc[pred_idx]),
                        "comfort": float(comfort[pred_idx]), "pdm_score": pdms,
                        "reference_raw_progress_m": float(raw[ref_idx]),
                        "prediction_raw_progress_m": float(raw[pred_idx]),
                        "reference_multiplier": float(nc[ref_idx] * dac[ref_idx]),
                        "progress_denominator_m": denominator,
                        "initial_state_max_abs_delta": initial_delta,
                        "same_centerline_bytes": same_line,
                    }
                    actor_rows.append(row)
                error = max(abs(actor_rows[0][key] - expected[key]) for key in METRICS)
                if error > 1e-8:
                    raise ValueError(f"Baseline replay mismatch for {actor}: {error}")
                for row in actor_rows:
                    row["baseline_replay_max_abs_error"] = error
                rows.extend(actor_rows)
        except Exception:
            rows.append({"token": token, "valid": False, "error": traceback.format_exc()})
    return rows


def main():
    from datetime import datetime
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference-cache", required=True, type=Path, help="Existing independent cache root with log/type/token/metric_cache.pkl.")
    parser.add_argument("--human-csv", type=Path, help="Completed human baseline CSV; defaults to the latest recorded human run.")
    parser.add_argument("--limit", type=int, default=256, help="Diagnostic sample size; 0 means all scenes.")
    parser.add_argument("--probe-only", action="store_true", help="Inspect the independent cache and coverage without rescoring.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    run_dir = args.run_dir.resolve()
    csv_files = list(run_dir.glob("*.csv"))
    if len(csv_files) != 1:
        parser.error("Expected exactly one baseline CSV in run_dir.")
    expected_model = read_rows(csv_files[0])
    other_paths = discover_cache(args.reference_cache)
    if not other_paths:
        parser.error("No cache files found at log/type/token/metric_cache.pkl under --reference-cache.")
    common = sorted(set(expected_model) & set(other_paths))
    print(f"Independent cache: {args.reference_cache}; files: {len(other_paths)}; matching baseline tokens: {len(common)}/{len(expected_model)}", flush=True)
    if not common:
        parser.error("The independent cache has no matching scene tokens.")
    description = describe_cache(other_paths[common[0]])
    print(json.dumps(description, indent=2), flush=True)
    print("Cache provenance is not inferred from its directory name.", flush=True)
    if args.probe_only:
        return
    human_files = sorted((run_dir / "diagnostics_human_baseline").glob("*/human_navsim_v1_scores.csv"))
    human_csv = args.human_csv if args.human_csv else (human_files[-1] if human_files else None)
    if human_csv is None:
        parser.error("Human baseline CSV missing; provide --human-csv.")
    expected_human = read_rows(human_csv)
    if set(expected_model) != set(expected_human):
        parser.error("Human and model baseline token sets differ.")
    if args.limit == 0 and len(common) != len(expected_model):
        parser.error("A full comparison requires full independent cache coverage.")
    ranked = sorted(common, key=lambda token: hashlib.sha256(token.encode()).digest())
    selected = sorted(ranked[:args.limit] if args.limit else ranked)
    print(f"Diagnostic sample: {len(selected)} scenes; full baseline: {len(expected_model)}. Subset means are NOT paper benchmark scores.", flush=True)
    cfg = OmegaConf.load(run_dir / "code" / "hydra" / "config.yaml")
    if cfg.scorer._target_ != "navsim.evaluate.navsim_v1.pdm_scorer.PDMScorer":
        raise ValueError("Expected NAVSIM v1 scorer.")
    from navsim.planning.script.run_drivesuprim_navsim_v1 import read_cache_index
    base_paths = read_cache_index(cfg, expected_model)
    import pickle
    with (run_dir / "predictions_navsim_v1.pkl").open("rb") as stream:
        raw_predictions = pickle.load(stream)
    if not set(selected).issubset(raw_predictions):
        raise ValueError("Saved model predictions missing diagnostic tokens.")
    predictions = {token: {"trajectory": raw_predictions[token]["trajectory"]} for token in selected}
    del raw_predictions
    output = run_dir / "diagnostics_progress_references" / datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    output.mkdir(parents=True, exist_ok=False)
    cfg.output_dir = str(output)
    by_log = {}
    for token in selected:
        by_log.setdefault(Path(base_paths[token]).parents[2].name, []).append(token)
    shared = {
        "cfg": cfg, "base_paths": {t: base_paths[t] for t in selected},
        "other_paths": {t: other_paths[t] for t in selected}, "predictions": predictions,
        "expected_model": {t: expected_model[t] for t in selected},
        "expected_human": {t: expected_human[t] for t in selected},
    }
    jobs = [{"log_name": log, "tokens": tokens, **shared} for log, tokens in sorted(by_log.items())]
    from nuplan.planning.utils.multithreading.worker_utils import worker_map
    from navsim.planning.script.builders.worker_pool_builder import build_worker
    import pandas as pd
    print("Comparing reference trajectory and centerline; keeping predictions, observation, ego state, map and scorer fixed.", flush=True)
    records = worker_map(build_worker(cfg), comparison_worker, jobs)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "per_scene_progress_comparison.csv", index=False)
    if not frame["valid"].all():
        raise RuntimeError("Diagnostic failed; inspect per_scene_progress_comparison.csv. No incomplete means reported.")
    if len(frame) != len(selected) * 6:
        raise RuntimeError("Unexpected diagnostic result count.")
    print("Baseline replay matched previous model/human CSVs.", flush=True)
    print("actor   variant                              EP %       PDMS %   change pp", flush=True)
    summaries = []
    for actor in ["human", "model"]:
        baseline = frame[(frame["actor"] == actor) & (frame["variant"] == "current_reference")]["pdm_score"].mean()
        for variant in ["current_reference", "other_reference_only", "other_reference_and_centerline"]:
            subset = frame[(frame["actor"] == actor) & (frame["variant"] == variant)]
            row = {
                "actor": actor, "variant": variant, "scene_count": len(subset),
                "mean_ep": float(subset["ego_progress"].mean()),
                "mean_pdms": float(subset["pdm_score"].mean()),
                "change_pdms_pp": float(100 * (subset["pdm_score"].mean() - baseline)),
            }
            summaries.append(row)
            print(f"{actor:<7} {variant:<36} {100 * row['mean_ep']:9.4f} {100 * row['mean_pdms']:12.4f} {row['change_pdms_pp']:+11.4f}", flush=True)
    summary = {
        "diagnostic_only": True, "sample_size": len(selected), "full_baseline_size": len(expected_model),
        "other_cache": str(args.reference_cache), "other_cache_description": description,
        "selected_tokens_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "baseline_replay_max_abs_error": float(frame["baseline_replay_max_abs_error"].max()),
        "exact_centerline_match_fraction": float(frame["same_centerline_bytes"].mean()),
        "results": summaries,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Progress/reference diagnostic complete. Results:", output, flush=True)


if __name__ == "__main__":
    main()
