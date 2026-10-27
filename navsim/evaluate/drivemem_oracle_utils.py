"""Deterministic sampling and descriptive statistics for the local oracle.

No model, simulation, or future outcome is consulted when selecting scenes.
"""
import numpy as np
import pandas as pd


def sample_logs(records, n_logs, per_log, min_gap_s, seed, excluded_logs=()):
    if n_logs < 1 or per_log < 1 or min_gap_s < 0:
        raise ValueError("n_logs/per_log must be positive; min_gap_s must be nonnegative")
    frame = pd.DataFrame(records)
    if frame.empty or frame.token.duplicated().any():
        raise ValueError("Empty scene pool or duplicate tokens")
    frame = frame[~frame.log_name.isin(excluded_logs)]
    rng = np.random.default_rng(seed)
    eligible = {}
    for log, group in frame.groupby("log_name", sort=True):
        group = group.sort_values(["timestamp_us", "token"]).reset_index(drop=True)
        times = group.timestamp_us.to_numpy(dtype=np.int64)
        next_index = np.maximum(np.arange(len(times)) + 1,
                                np.searchsorted(times, times + int(round(min_gap_s * 1e6)), side="left"))
        capacity = np.zeros(len(times) + 1, dtype=np.int64)
        for i in range(len(times) - 1, -1, -1):
            capacity[i] = 1 + capacity[next_index[i]]
        if capacity[0] >= per_log:
            eligible[log] = (group, next_index, capacity)
    if len(eligible) < n_logs:
        raise ValueError(f"Requested {n_logs} logs but only {len(eligible)} support "
                         f"{per_log} scenes separated by {min_gap_s} seconds")
    chosen_logs = rng.choice(sorted(eligible), n_logs, replace=False).tolist()
    selected = []
    for log in sorted(chosen_logs):
        group, next_index, capacity = eligible[log]
        start = 0
        for remaining in range(per_log, 0, -1):
            choices = np.flatnonzero(capacity[:-1] >= remaining)
            choices = choices[choices >= start]
            index = int(rng.choice(choices))
            row = group.iloc[index]
            selected.append({"token": str(row.token), "log_name": str(log),
                             "timestamp_us": int(row.timestamp_us)})
            start = int(next_index[index])
    return selected, {"method": "uniform_logs_then_random_feasible_times", "seed": int(seed),
                      "n_logs": n_logs, "scenes_per_log": per_log, "min_gap_s": min_gap_s,
                      "eligible_logs": len(eligible), "excluded_logs": list(excluded_logs)}


def summarize_candidates(rows):
    frame = pd.DataFrame(rows).sort_values(["model_rank", "candidate_id"], kind="stable")
    base = frame[frame.is_baseline]
    if len(base) != 1 or frame.candidate_id.duplicated().any():
        raise ValueError("Expected one baseline and unique candidates")
    base = base.iloc[0]
    best = frame.loc[frame.pdm_score.idxmax()]
    safe = frame[frame.safe_vs_baseline]
    if safe.empty:
        raise ValueError("Baseline must remain safety-feasible")
    safe_best = safe.loc[safe.pdm_score.idxmax()]
    result = {"token": str(base.token), "log_name": str(base.log_name), "n_candidates": len(frame),
              "baseline_local_pdm": float(base.pdm_score), "oracle_local_pdm": float(best.pdm_score),
              "safe_oracle_local_pdm": float(safe_best.pdm_score),
              "oracle_gain_pp": 100 * float(best.pdm_score - base.pdm_score),
              "safe_oracle_gain_pp": 100 * float(safe_best.pdm_score - base.pdm_score),
              "oracle_candidate_id": int(best.candidate_id), "oracle_model_rank": int(best.model_rank),
              "safe_oracle_model_rank": int(safe_best.model_rank),
              "better_candidates": int((frame.pdm_score > base.pdm_score + 1e-8).sum()),
              "baseline_nc": float(base.no_at_fault_collisions),
              "baseline_ttc": float(base.time_to_collision_within_bound),
              "oracle_nc": float(best.no_at_fault_collisions),
              "oracle_ttc": float(best.time_to_collision_within_bound),
              "predicted_nc_baseline": float(base.predicted_nc),
              "predicted_nc_oracle": float(best.predicted_nc),
              "predicted_ttc_baseline": float(base.predicted_ttc),
              "predicted_ttc_oracle": float(best.predicted_ttc)}
    for k in (8, 16, 32, 64, 256):
        subset = frame[(frame.model_rank <= k) | frame.is_baseline]
        if len(frame) >= k:
            result[f"oracle_top{k}_gain_pp"] = 100 * float(subset.pdm_score.max() - base.pdm_score)
    return result


def aggregate_report(summary, seed=20261027, bootstrap_reps=2000):
    grouped = summary.groupby("log_name", sort=True)
    log_summary = grouped.agg(scenes=("token", "size"), baseline_local_pdm=("baseline_local_pdm", "mean"),
                              oracle_gain_pp=("oracle_gain_pp", "mean"),
                              safe_oracle_gain_pp=("safe_oracle_gain_pp", "mean"))
    rng = np.random.default_rng(seed)
    report = {"logs": len(log_summary), "scenes": len(summary),
              "logs_with_improvement": int((log_summary.oracle_gain_pp > 1e-6).sum()),
              "scenes_with_improvement": int((summary.oracle_gain_pp > 1e-6).sum()),
              "scene_gain_median_pp": float(summary.oracle_gain_pp.median()),
              "scene_gain_max_pp": float(summary.oracle_gain_pp.max()),
              "bootstrap_reps": bootstrap_reps, "bootstrap_seed": seed}
    counts = log_summary.scenes.to_numpy()
    for key in ("oracle_gain_pp", "safe_oracle_gain_pp"):
        means = log_summary[key].to_numpy()
        report[key + "_log_mean"] = float(means.mean())
        if len(means) >= 2:
            samples = rng.integers(0, len(means), size=(bootstrap_reps, len(means)))
            sampled_counts = counts[samples]
            boot = (means[samples] * sampled_counts).sum(axis=1) / sampled_counts.sum(axis=1)
            report[key + "_cluster_ci95"] = [float(v) for v in np.quantile(boot, [.025, .975])]
        else:
            report[key + "_cluster_ci95"] = None
    report["topk_mean_gain_pp"] = {key: float(summary[key].mean()) for key in summary.columns
                                   if key.startswith("oracle_top")}
    return report, log_summary.reset_index()
