"""Compare DriveSuprim scoring definitions using existing CSVs and predictions.

CSV mode uses only the standard library. --ttc-window additionally replays the
saved v1 predictions on the same v1 cache and changes ONLY the TTC loop endpoint.
These are diagnostic counterfactuals, not a declaration of paper reproduction.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from diagnose_navsim_v1 import METRICS, PAPER_R34, audit_csv, sha256

COMMON = METRICS[:4]
V2_METRICS = COMMON + ["history_comfort", "score"]
AUTHOR_COMMIT = "80fe792d7654a596d92e20d030d1650f6f605c02"
AUTHOR_AGGREGATION = (
    "https://github.com/William-Yao-2000/DriveSuprim/blob/" + AUTHOR_COMMIT
    + "/navsim/planning/script/grid_search_unlog_expand.py"
)


def read_snapshot(path, metrics):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"token", "valid", *metrics}.issubset(reader.fieldnames or []):
            raise ValueError("Required metric columns missing: " + str(path))
        rows, summaries = {}, {}
        for raw in reader:
            token = raw["token"]
            if token.startswith(("average_", "extended_pdm_score")):
                summaries[token] = raw
                continue
            if not token or token in rows:
                raise ValueError("Empty or duplicate scenario token: " + repr(token))
            if raw["valid"].strip().lower() not in ("true", "1"):
                raise ValueError("Invalid scenario; refusing partial means: " + token)
            values = {key: float(raw[key]) for key in metrics}
            if any(not math.isfinite(x) or not -1e-8 <= x <= 1 + 1e-8 for x in values.values()):
                raise ValueError("Non-finite or unnormalized metrics: " + token)
            rows[token] = values
    if not rows:
        raise ValueError("No scenarios: " + str(path))
    if "average_all_frames" in summaries:
        for key in metrics:
            recorded = float(summaries["average_all_frames"][key])
            if not math.isfinite(recorded) or abs(recorded - mean(r[key] for r in rows.values())) > 1e-8:
                raise ValueError("Stored average differs from scenario mean: " + key)
    return rows


def old_pdms(row):
    # Same scalar formula as aggregate_old_pdms; C is explicitly history_comfort.
    return row["no_at_fault_collisions"] * row["drivable_area_compliance"] * (
        5 * row["ego_progress"] + 5 * row["time_to_collision_within_bound"]
        + 2 * row["history_comfort"]
    ) / 12


def write_rows(path, records):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def csv_comparison(v1_path, v2_path, output):
    v1 = read_snapshot(v1_path, METRICS)
    v2 = read_snapshot(v2_path, V2_METRICS)
    if set(v1) != set(v2):
        raise ValueError("v1/v2 CSV token sets differ; refusing a silent intersection.")
    records = []
    for token in sorted(v1):
        records.append({
            "token": token, "v2_old_formula_hc": old_pdms(v2[token]),
            **{"v1_" + key: value for key, value in v1[token].items()},
            **{"v2_" + key: value for key, value in v2[token].items()},
        })
    result = {
        "scene_count": len(records),
        "v1_csv": str(v1_path), "v1_csv_sha256": sha256(v1_path),
        "v2_csv": str(v2_path), "v2_csv_sha256": sha256(v2_path),
        "tokens_sha256": hashlib.sha256("\n".join(sorted(v1)).encode()).hexdigest(),
        "v1_averages": {k: mean(r[k] for r in v1.values()) for k in METRICS},
        "v2_averages": {k: mean(r[k] for r in v2.values()) for k in V2_METRICS},
        "v2_old_formula_hc": mean(r["v2_old_formula_hc"] for r in records),
        "author_aggregation_source": AUTHOR_AGGREGATION,
        "scope": (
            "Row-wise NC*DAC*(5*EP+5*TTC+2*HC)/12 on saved v2 submetrics. "
            "No v1 comfort recomputation; no change to EP/TTC definitions. "
            "The author's grid-search script does not prove the Table 2 protocol. "
            "Same scene tokens do not establish identical selected trajectories."
        ),
    }
    result["change_vs_saved_v1_pp"] = 100 * (
        result["v2_old_formula_hc"] - result["v1_averages"]["pdm_score"]
    )
    result["gap_to_paper_v1_pp"] = 100 * (result["v2_old_formula_hc"] - PAPER_R34["pdm_score"])
    write_rows(output / "per_scene_csv_comparison.csv", records)
    (output / "csv_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Matched CSV scenes: {len(records)}", flush=True)
    print("Metric             saved v1 %    saved v2 %", flush=True)
    for label, k1, k2 in [
        ("NC", COMMON[0], COMMON[0]), ("DAC", COMMON[1], COMMON[1]),
        ("EP", COMMON[2], COMMON[2]), ("TTC", COMMON[3], COMMON[3]),
        ("C(v1) / HC(v2)", "comfort", "history_comfort"),
    ]:
        print(f"{label:<18} {100 * result['v1_averages'][k1]:11.4f} {100 * result['v2_averages'][k2]:13.4f}", flush=True)
    print(f"Saved v1 PDMS: {100 * result['v1_averages']['pdm_score']:.4f}%", flush=True)
    print(f"Saved v2 EPDMS: {100 * result['v2_averages']['score']:.4f}%", flush=True)
    print(f"Old formula on v2 EP/TTC/HC: {100 * result['v2_old_formula_hc']:.4f}%", flush=True)
    print(f"Difference from saved v1: {result['change_vs_saved_v1_pp']:+.4f} pp", flush=True)
    print("This changes several definitions AND possibly predictions; it is not a single-factor test.", flush=True)
    return result


def recalculate_ttc_prefix(scorer, last_index):
    """Only rerun v1 _calculate_ttc with a shorter loop, on the SAME state arrays.

    The v1 method reads proposal_sampling only for num_poses (its outer-loop
    bound) and interval_length (unchanged). No simulator/sampling/reset is called.
    Full-horizon TTC and its earliest-failure index also give an independent
    expected prefix result, checked by the caller for BOTH proposals.
    """
    import numpy as np
    sampling = scorer.proposal_sampling
    old_ttc = scorer._weighted_metrics.copy()
    old_times = scorer._ttc_time_idcs.copy()
    if not 0 <= last_index <= sampling.num_poses:
        raise ValueError("Invalid TTC loop endpoint.")
    try:
        scorer.proposal_sampling = SimpleNamespace(
            num_poses=last_index, interval_length=sampling.interval_length,
        )
        scorer._ttc_time_idcs = np.full_like(old_times, np.inf)
        scorer._calculate_ttc()
        from navsim.evaluate.navsim_v1.pdm_enums import WeightedMetricIndex
        return scorer._weighted_metrics[WeightedMetricIndex.TTC].copy()
    finally:
        scorer.proposal_sampling = sampling
        scorer._weighted_metrics[:] = old_ttc
        scorer._ttc_time_idcs = old_times


def ttc_worker(args):
    import lzma
    import pickle
    import traceback
    import numpy as np
    from hydra.utils import instantiate
    from navsim.evaluate.navsim_v1.pdm_enums import MultiMetricIndex, WeightedMetricIndex
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory

    shared = args[0]
    cfg = shared["cfg"]
    simulator, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise ValueError("Simulator/scorer sampling mismatch.")
    sampling = scorer.proposal_sampling
    if sampling.num_poses != 40 or not math.isclose(sampling.interval_length, 0.1):
        raise ValueError("Diagnostic is pinned to 40 poses at 0.1 s.")
    if [scorer._config.progress_weight, scorer._config.ttc_weight,
        scorer._config.comfortable_weight, scorer._config.driving_direction_weight] != [5, 5, 2, 0]:
        raise ValueError("Unexpected scorer aggregation weights.")
    last_index = 31  # 40 - max([0, 3, 6, 9]), as in public DriveSuprim.
    records = []
    for token in sorted(t for item in args for t in item["tokens"]):
        row = {"token": token, "valid": False}
        try:
            with lzma.open(shared["cache_paths"][token], "rb") as stream:
                cache = pickle.load(stream)
            cache.observation[49]  # Full v1 horizon must exist; never substitute the v2 cache.
            trajectory = shared["predictions"][token]["trajectory"]
            pred = transform_trajectory(trajectory, cache.ego_state)
            states = np.stack([
                get_trajectory_as_array(cache.trajectory, sampling, cache.ego_state.time_point),
                get_trajectory_as_array(pred, sampling, cache.ego_state.time_point),
            ])
            simulated = simulator.simulate_proposals(states, cache.ego_state)
            scores = scorer.score_proposals(
                simulated, cache.observation, cache.centerline, cache.route_lane_ids, cache.drivable_area_map,
            )
            metrics = {
                "no_at_fault_collisions": float(scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, 1]),
                "drivable_area_compliance": float(scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 1]),
                "ego_progress": float(scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, 1]),
                "time_to_collision_within_bound": float(scorer._weighted_metrics[WeightedMetricIndex.TTC, 1]),
                "comfort": float(scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, 1]),
                "pdm_score": float(scores[1]),
            }
            if any(not math.isfinite(value) for value in metrics.values()):
                raise ValueError("Non-finite baseline metrics.")
            error = max(abs(metrics[k] - shared["expected"][token][k]) for k in METRICS)
            if error > 1e-8:
                raise ValueError(f"Baseline replay differs from saved CSV: {error}")
            earliest = scorer._ttc_time_idcs.copy()
            full_ttc = scorer._weighted_metrics[WeightedMetricIndex.TTC].copy()
            np.testing.assert_array_equal(full_ttc, np.where(np.isfinite(earliest), 0.0, 1.0))
            short_ttc = recalculate_ttc_prefix(scorer, last_index)
            np.testing.assert_array_equal(short_ttc, np.where(earliest <= last_index, 0.0, 1.0))
            if np.any(short_ttc < full_ttc):
                raise ValueError("Shorter TTC window unexpectedly penalizes a previously valid proposal.")
            gain = metrics["no_at_fault_collisions"] * metrics["drivable_area_compliance"] * (
                5 / 12 * (float(short_ttc[1]) - metrics["time_to_collision_within_bound"])
            )
            row.update(
                valid=True, **metrics, ttc_short=float(short_ttc[1]),
                pdms_short=metrics["pdm_score"] + gain, pdms_gain=gain,
                first_ttc_violation_s=float(earliest[1] * sampling.interval_length)
                    if math.isfinite(float(earliest[1])) else None,
                baseline_replay_max_abs_error=error,
            )
        except Exception:
            row["error"] = traceback.format_exc()
        records.append(row)
        if len(records) % 250 == 0:
            print(f"TTC window scenes: {len(records)}", flush=True)
    return records


def run_ttc(run_dir, baseline_path, output, limit, threads):
    from omegaconf import OmegaConf
    from navsim.planning.script.run_drivesuprim_navsim_v1 import read_cache_index
    from diagnose_navsim_v1 import load_predictions

    expected = read_snapshot(baseline_path, METRICS)
    cfg = OmegaConf.load(run_dir / "code" / "hydra" / "config.yaml")
    if cfg.scorer._target_ != "navsim.evaluate.navsim_v1.pdm_scorer.PDMScorer":
        raise ValueError("Expected the frozen v1 scorer.")
    all_paths = read_cache_index(cfg, expected)  # Also verifies frozen v1 source hashes.
    manifest_path = run_dir / "reproduction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(baseline_path) != manifest.get("csv_sha256") or not manifest.get("complete"):
        raise ValueError("Baseline CSV does not match a completed run manifest.")
    cache_manifest = Path(cfg.metric_cache_path) / "navsim_v1_manifest.json"
    if sha256(cache_manifest) != manifest.get("cache_manifest_sha256"):
        raise ValueError("Cache manifest differs from the original run.")
    source_mismatches = [
        relative for relative, digest in manifest.get("source_sha256", {}).items()
        if not (ROOT / relative).is_file() or sha256(ROOT / relative) != digest
    ]
    if source_mismatches:
        raise ValueError("Recorded evaluation sources changed: " + repr(source_mismatches))
    selected = sorted(expected, key=lambda t: hashlib.sha256(t.encode()).digest())
    selected = sorted(selected[:limit] if limit else selected)
    prediction_path = run_dir / "predictions_navsim_v1.pkl"
    predictions = load_predictions(prediction_path)
    if not set(expected).issubset(predictions):
        raise ValueError("Saved v1 predictions have incomplete coverage.")
    predictions = {token: predictions[token] for token in selected}
    cfg.output_dir = str(output)
    cfg.worker.threads_per_node = threads
    by_log = {}
    for token in selected:
        by_log.setdefault(Path(all_paths[token]).parents[2].name, []).append(token)
    shared = {
        "cfg": cfg, "predictions": predictions,
        "cache_paths": {t: all_paths[t] for t in selected},
        "expected": {t: expected[t] for t in selected},
    }
    jobs = [{"tokens": tokens, **shared} for _, tokens in sorted(by_log.items())]
    print(f"TTC diagnostic: {len(selected)}/{len(expected)} scenes; CPU workers: {threads}; no model inference.", flush=True)
    print("Only TTC ego loop changes: 0..4.0 s -> 0..3.1 s. Projection offsets remain 0/0.3/0.6/0.9 s.", flush=True)
    from nuplan.planning.utils.multithreading.worker_utils import worker_map
    from navsim.planning.script.builders.worker_pool_builder import build_worker
    import pandas as pd
    records = worker_map(build_worker(cfg), ttc_worker, jobs)
    frame = pd.DataFrame(records).sort_values("token")
    frame.to_csv(output / "per_scene_ttc_windows.csv", index=False)
    if len(frame) != len(selected) or frame["token"].duplicated().any() or set(frame["token"]) != set(selected):
        raise RuntimeError("TTC diagnostic coverage/uniqueness mismatch.")
    if not frame["valid"].all():
        raise RuntimeError("Some replay checks failed. Inspect per_scene_ttc_windows.csv; no partial means reported.")
    result = {
        "diagnostic_only": True, "scene_count": len(selected), "full_baseline_size": len(expected),
        "full_split": len(selected) == len(expected), "cpu_workers": threads,
        "tokens_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "baseline_csv_sha256": sha256(baseline_path), "predictions_sha256": sha256(prediction_path),
        "cache_manifest_sha256": sha256(cache_manifest), "script_sha256": sha256(__file__),
        "ego_last_time_s": {"baseline": 4.0, "counterfactual": 3.1},
        "observation_last_time_s": {"baseline": 4.9, "counterfactual": 4.0},
        "projection_offsets_s": [0.0, 0.3, 0.6, 0.9],
        "baseline_replay_max_abs_error": float(frame["baseline_replay_max_abs_error"].max()),
        "prefix_crosscheck": "Recomputed TTC agrees with earliest full-window violation for both proposals.",
        "ttc_full": float(frame["time_to_collision_within_bound"].mean()),
        "ttc_short": float(frame["ttc_short"].mean()),
        "pdms_full": float(frame["pdm_score"].mean()),
        "pdms_short": float(frame["pdms_short"].mean()),
        "pdms_change_pp": float(100 * frame["pdms_gain"].mean()),
        "ttc_changed_scenes": int((frame["ttc_short"] != frame["time_to_collision_within_bound"]).sum()),
        "positive_pdms_gain_scenes": int((frame["pdms_gain"] > 0).sum()),
        "scope": (
            "Same predictions, simulator states, 5s observation cache, reference, map, EP and comfort. "
            "Only the TTC outer-loop endpoint is shortened; this is not a full v2 scorer. "
            "Sample means must not be reported as full-split reproduction."
        ),
    }
    (output / "ttc_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Baseline replay matched saved v1 CSV; independent TTC prefix crosscheck passed.", flush=True)
    print("variant        TTC %      PDMS %", flush=True)
    print(f"v1 full       {100 * result['ttc_full']:8.4f} {100 * result['pdms_full']:11.4f}", flush=True)
    print(f"short TTC     {100 * result['ttc_short']:8.4f} {100 * result['pdms_short']:11.4f}", flush=True)
    print(f"Isolated TTC-window PDMS change: {result['pdms_change_pp']:+.4f} pp", flush=True)
    print(f"Changed TTC scenes: {result['ttc_changed_scenes']}; positive PDMS gain: {result['positive_pdms_gain_scenes']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed strict-v1 run with saved predictions/config/manifest.")
    parser.add_argument("--v2-csv", required=True, type=Path)
    parser.add_argument("--v1-csv", type=Path, help="Defaults to the sole CSV in run_dir.")
    parser.add_argument("--ttc-window", action="store_true", help="Also replay the controlled TTC window test on CPU.")
    parser.add_argument("--limit", type=int, default=0, help="TTC sample size; 0 means all scenes (default).")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 0 or args.threads < 1:
        parser.error("--limit must be >=0 and --threads >=1.")
    if args.limit and not args.ttc_window:
        parser.error("--limit applies only with --ttc-window.")
    run_dir = args.run_dir.resolve()
    candidates = sorted(run_dir.glob("*.csv"))
    baseline_path = args.v1_csv
    if baseline_path is None:
        if len(candidates) != 1:
            parser.error("Expected one baseline CSV in run_dir; provide --v1-csv.")
        baseline_path = candidates[0]
    baseline = audit_csv(baseline_path)
    if baseline["invalid_count"] or baseline["max_formula_error"] > 1e-8:
        raise ValueError("Baseline CSV has invalid scenes or fails the v1 formula check.")
    output = run_dir / "diagnostics_protocols" / datetime.now(timezone.utc).strftime("%Y.%m.%d.%H.%M.%S.%f")
    output.mkdir(parents=True, exist_ok=False)
    print("Diagnostic output:", output, flush=True)
    csv_comparison(baseline_path, args.v2_csv.resolve(), output)
    if args.ttc_window:
        run_ttc(run_dir, baseline_path, output, args.limit, args.threads)
    print("Protocol diagnostic complete. Results:", output, flush=True)


if __name__ == "__main__":
    main()
