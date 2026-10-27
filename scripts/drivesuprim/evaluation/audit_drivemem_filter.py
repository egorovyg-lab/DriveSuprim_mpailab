"""Read-only audit of human penalty filtering on saved local-oracle candidates.

Requires an existing complete multilog export and candidate_metrics.csv. Only a
few candidates per affected scene are rescored; old artifacts are not changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score

METRICS = (
    "no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
    "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound",
    "lane_keeping", "history_comfort", "pdm_score",
)
SAFETY = METRICS[:4] + ("time_to_collision_within_bound",)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_formula(frame):
    gate = (frame["no_at_fault_collisions"] * frame["drivable_area_compliance"]
            * frame["driving_direction_compliance"] * frame["traffic_light_compliance"])
    return gate * (5 * frame["ego_progress"] + 5 * frame["time_to_collision_within_bound"]
                   + 2 * frame["lane_keeping"] + 2 * frame["history_comfort"]) / 14


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed export directory")
    parser.add_argument("--candidate-limit", type=int, default=256)
    parser.add_argument("--samples-per-scene", type=int, default=3,
                        help="In addition to baseline and oracle: largest mismatches")
    parser.add_argument("--all-scenes", action="store_true",
                        help="Audit baseline and saved oracle for every scene")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print scene details only if filter changes a selected metric or safety comparison")
    args = parser.parse_args()
    if args.samples_per_scene < 0:
        parser.error("--samples-per-scene must be nonnegative")

    root = args.run_dir
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "export_complete":
        raise RuntimeError("Export is incomplete")
    scoring = root / f"scoring_k{args.candidate_limit}"
    report = json.loads((scoring / "report.json").read_text(encoding="utf-8"))
    identity = json.loads((scoring / "scoring_identity.json").read_text(encoding="utf-8"))
    if report.get("status") != "complete" or identity["export_manifest_sha256"] != sha256(root / "manifest.json"):
        raise RuntimeError("Report/export identity mismatch or incomplete report")
    if sha256(root / "vocab.npz") != manifest["vocab_sha256"]:
        raise RuntimeError("Vocabulary hash mismatch")

    df = pd.read_csv(scoring / "candidate_metrics.csv")
    if len(df) != report["scenes"] * args.candidate_limit:
        raise RuntimeError("Unexpected number of candidate rows")
    df["reconstructed"] = local_formula(df)
    df["difference"] = df["pdm_score"] - df["reconstructed"]
    mismatch = df[~np.isclose(df["difference"], 0, atol=1e-6, rtol=1e-6)]
    mismatch_tokens = sorted(mismatch["token"].unique())
    tokens = sorted(df["token"].unique()) if args.all_scenes else mismatch_tokens
    print(f"Candidates={len(df)}; mismatches={len(mismatch)}; affected scenes={len(mismatch_tokens)}")
    print("This audit rescores selected candidates, not the complete 256-candidate oracle.")
    if not tokens:
        return

    cfg = OmegaConf.create(manifest["config"])
    if cfg.traffic_agents != "non_reactive" or not cfg.scorer.config.human_penalty_filter:
        raise RuntimeError("This audit expects the original non-reactive, filter-enabled protocol")
    cache = MetricCacheLoader(Path(cfg.metric_cache_path))
    simulator = instantiate(cfg.simulator)
    scorer_raw = instantiate(cfg.scorer)
    scorer_filtered = instantiate(cfg.scorer)
    scorer_raw._config.human_penalty_filter = False
    scorer_filtered._config.human_penalty_filter = True
    policy = instantiate(cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling)
    sampling = instantiate(cfg.agent.config.trajectory_sampling)
    with np.load(root / "vocab.npz", allow_pickle=False) as data:
        vocab = data["vocab"]

    overview = {"scenes": 0, "candidate_evaluations": 0, "raw_safe": 0, "filtered_safe": 0,
                "safety_disagreement": 0, "baseline_score_changed": 0, "oracle_score_changed": 0,
                "baseline_filter_changed": 0, "oracle_filter_changed": 0,
                "raw_gain_sum_pp": 0.0, "filtered_gain_sum_pp": 0.0}
    for token in tokens:
        group = df[df.token == token]
        if len(group) != args.candidate_limit or int(group.is_baseline.sum()) != 1:
            raise RuntimeError(f"Invalid candidate rows: {token}")
        saved_done = json.loads((scoring / f"{token}_done.json").read_text(encoding="utf-8"))
        candidate_path = root / "candidates" / f"{token}.npz"
        if sha256(candidate_path) != saved_done["candidate_sha256"]:
            raise RuntimeError(f"Candidate file hash mismatch: {token}")
        with np.load(candidate_path, allow_pickle=False) as data:
            ids = data["candidate_ids"]
        baseline = group[group.is_baseline].iloc[0]
        oracle = group.sort_values(["pdm_score", "model_rank"], ascending=[False, True]).iloc[0]
        worst = (group.assign(abs_diff=group.difference.abs())
                 .sort_values(["abs_diff", "model_rank"], ascending=[False, True])
                 .head(args.samples_per_scene))
        if args.all_scenes:
            worst = worst.iloc[:0]
        selected = list(dict.fromkeys([int(baseline.candidate_local), int(oracle.candidate_local)]
                                      + [int(x) for x in worst.candidate_local]))
        details = [f"\nSCENE {token} log={baseline.log_name} candidates_to_rescore={selected}",
                   f"Saved baseline score={baseline.pdm_score:.6f}; saved oracle score={oracle.pdm_score:.6f}"]
        results = {}
        for local in selected:
            saved = group[group.candidate_local == local]
            if len(saved) != 1:
                raise RuntimeError(f"Duplicate/missing candidate {token}/{local}")
            saved = saved.iloc[0]
            if int(saved.candidate_id) != int(ids[local]):
                raise RuntimeError(f"Candidate ID mismatch: {token}/{local}")

            def evaluate(scorer):
                scene_cache = cache.get_from_token(token)  # pdm_score mutates observation
                result, _ = pdm_score(scene_cache, Trajectory(vocab[ids[local]], sampling),
                                      simulator.proposal_sampling, simulator, scorer, policy)
                values = {key: float(result.iloc[0][key]) for key in METRICS}
                if not np.isfinite(list(values.values())).all():
                    raise RuntimeError(f"Non-finite values: {token}/{local}")
                return values

            raw, filtered = evaluate(scorer_raw), evaluate(scorer_filtered)
            for key in METRICS:
                if not np.isclose(filtered[key], float(saved[key]), atol=1e-8, rtol=1e-7):
                    raise RuntimeError(f"Stored/scorer mismatch {token}/{local} {key}: "
                                       f"{saved[key]} != {filtered[key]}")
            reconstructed_raw = float(local_formula(raw))
            if not np.isclose(raw["pdm_score"], reconstructed_raw, atol=1e-8, rtol=1e-7):
                raise RuntimeError(f"Raw local score/formula mismatch: {token}/{local}")
            changed = {key: [raw[key], filtered[key]] for key in METRICS
                       if not np.isclose(raw[key], filtered[key], atol=1e-8, rtol=1e-7)}
            results[local] = (raw, filtered)
            kind = "baseline" if local == int(baseline.candidate_local) else (
                "saved_oracle" if local == int(oracle.candidate_local) else "mismatch_sample")
            details.append(f"  {kind} local={local} vocab={ids[local]} model_rank={int(saved.model_rank)} "
                           f"raw_score={raw['pdm_score']:.6f} filtered_score={filtered['pdm_score']:.6f} "
                           f"filtered_formula={float(local_formula(filtered)):.6f}")
            details.append("    filtered_changes=" + json.dumps(changed, sort_keys=True))
        base_raw, base_filtered = results[int(baseline.candidate_local)]
        best_raw, best_filtered = results[int(oracle.candidate_local)]
        safe_results = {}
        for name, base, best in (("raw", base_raw, best_raw), ("filtered", base_filtered, best_filtered)):
            safe = all(best[key] + 1e-8 >= base[key] for key in SAFETY)
            safe_results[name] = safe
            details.append(f"  saved_oracle_vs_baseline {name}: gain_pp="
                           f"{100 * (best['pdm_score'] - base['pdm_score']):.6f}; "
                           f"safety_components_not_worse={safe}")
        changed_base = any(not np.isclose(base_raw[k], base_filtered[k], atol=1e-8, rtol=1e-7) for k in METRICS)
        changed_oracle = any(not np.isclose(best_raw[k], best_filtered[k], atol=1e-8, rtol=1e-7) for k in METRICS)
        if not args.summary_only or changed_base or changed_oracle or safe_results["raw"] != safe_results["filtered"]:
            print("\n".join(details))
        overview["scenes"] += 1
        overview["candidate_evaluations"] += len(selected)
        overview["raw_safe"] += int(safe_results["raw"])
        overview["filtered_safe"] += int(safe_results["filtered"])
        overview["safety_disagreement"] += int(safe_results["raw"] != safe_results["filtered"])
        overview["baseline_filter_changed"] += int(changed_base)
        overview["oracle_filter_changed"] += int(changed_oracle)
        overview["baseline_score_changed"] += int(not np.isclose(base_raw["pdm_score"], base_filtered["pdm_score"], atol=1e-8, rtol=1e-7))
        overview["oracle_score_changed"] += int(not np.isclose(best_raw["pdm_score"], best_filtered["pdm_score"], atol=1e-8, rtol=1e-7))
        overview["raw_gain_sum_pp"] += 100 * (best_raw["pdm_score"] - base_raw["pdm_score"])
        overview["filtered_gain_sum_pp"] += 100 * (best_filtered["pdm_score"] - base_filtered["pdm_score"])
    if overview["scenes"]:
        overview["raw_gain_mean_pp"] = overview["raw_gain_sum_pp"] / overview["scenes"]
        overview["filtered_gain_mean_pp"] = overview["filtered_gain_sum_pp"] / overview["scenes"]
    print("\nAUDIT SUMMARY: " + json.dumps(overview, sort_keys=True))
    print("\nAudit complete. Raw/filtered comparisons cover only the selected candidates above.")


if __name__ == "__main__":
    main()
