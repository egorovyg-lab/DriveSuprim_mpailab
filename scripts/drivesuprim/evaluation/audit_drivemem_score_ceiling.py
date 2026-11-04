"""Analytic ceiling of baseline-anchored head corrections, plus oracle-head probes.

Uses only stored predictions/outcomes. Oracle heads are diagnostic future labels,
not deployable inputs. No fitting, simulator calls, downloads or artifact writes.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_drivemem_action_support import load_frozen_bank, load_sample, read_json, sha
from navsim.agents.drivemem import METRICS, probabilities, metric_selection_score
from navsim.agents.drivemem_action_transfer import pool, choose


def ceiling_case(sample, outcomes):
    ix = pool(sample)
    base = int(sample["selected_local"])
    if base not in ix or any(str(i) not in outcomes for i in ix):
        raise RuntimeError("Incomplete pool or invalid baseline")
    p = probabilities(sample)
    scores = sample["selection_scores"].astype(float)
    ceiling_g = metric_selection_score(np.ones((1, len(METRICS))))[0]
    # G is coordinatewise increasing on [0,1]. Baseline is fixed by centering.
    upper = scores + ceiling_g - metric_selection_score(p)
    upper[base] = scores[base]
    reachable = [i for i in ix if i == base or upper[i] > scores[base]]
    quality = {int(i): float(outcomes[str(i)]["raw"]["pdm_score"]) for i in ix}
    baseline = quality[base]
    best = max(ix, key=lambda i: quality[i])
    best_reachable = max(reachable, key=lambda i: quality[i])
    optimal = [i for i in ix if np.isclose(quality[i], quality[best], atol=1e-10, rtol=0)]
    result = dict(token=sample["token"], session_id=sample["session_id"],
        action_oracle_gain_pp=100 * (quality[best] - baseline), oracle_rank=ix.tolist().index(best) + 1,
        anchored_ceiling_gain_pp=100 * (quality[best_reachable] - baseline),
        reachable_candidates=len(reachable), any_optimal_reachable=any(i in reachable for i in optimal),
        optimal_ceiling_margin=max(float(upper[i] - scores[base]) for i in optimal),
        baseline_metric_slack=float(ceiling_g - metric_selection_score(p[[base]])[0]))
    absolute = np.zeros_like(p)
    truth = np.array([[outcomes[str(i)]["raw"][key] for key in METRICS] for i in ix])
    if not np.isfinite(truth).all() or np.any((truth < 0) | (truth > 1)):
        raise RuntimeError("Invalid raw metric targets")
    absolute[ix] = truth - p[ix]
    contrast = absolute - absolute[base].copy()
    for name, correction in (("absolute", absolute), ("contrast", contrast)):
        for beta, suffix in ((.5, "05"), (1., "10")):
            index = choose(sample, correction, beta)
            result[f"oracle_heads_{name}_beta{suffix}_gain_pp"] = 100 * (quality[index] - baseline)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("validation", type=Path)
    args = parser.parse_args()
    source, _, vocab, fit, _ = load_frozen_bank(args.experiment)
    repo = Path(__file__).resolve().parents[3]
    original = read_json(source / "identity.json")
    if sha(repo / "navsim/agents/drivemem.py") != original["sources"]["navsim/agents/drivemem.py"]:
        raise RuntimeError("Original scoring formula changed")
    print("DIAGNOSTIC ONLY; no training or new labels. Frozen primary:", fit["primary"])
    for root in (source, args.validation):
        if read_json(root / "split_identity.json") != dict(splits=sha(root / "splits.json"), inventory=sha(root / "inventory.csv")):
            raise RuntimeError("Frozen cohort selection changed")
        with np.load(root / "vocab.npz", allow_pickle=False) as data:
            if not np.array_equal(data["vocab"], vocab):
                raise RuntimeError("Vocabulary mismatch")
        index = read_json(root / "export_index.json")
        rows = []
        for record in read_json(root / "splits.json")["records"]:
            if record["role"] != "development":
                continue
            sample = load_sample(root, record, index)
            outcomes = read_json(root / "development_oracle/scenes" / (sample["token"] + ".json"))["outcomes"]
            rows.append(ceiling_case(sample, outcomes))
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError("Empty inspected cohort")
        gains = [c for c in frame if c.endswith("gain_pp")]
        improved = frame[frame.action_oracle_gain_pp > 1e-6]
        print("\nCOHORT:", root, "SCENES:", len(frame), "SESSIONS:", frame.session_id.nunique())
        print("MEAN DIAGNOSTIC GAINS:")
        print(frame[gains].mean().to_string())
        print("BY SESSION:")
        print(frame.groupby("session_id")[gains].mean().to_string())
        print("ORACLE-POSITIVE CASES:")
        print(improved.to_string(index=False))
        print("Oracle-positive cases with NO optimal action reachable under anchoring:",
              int((~improved.any_optimal_reachable).sum()), "/", len(improved))
    print("\nAnchored ceiling is an optimistic necessary-condition bound, not an achievable method. "
          "It applies when baseline head correction is zero, head values are clipped to [0,1], "
          "and the original non-metric score term is preserved. Oracle-head probes expose "
          "score/aggregation limitations but use current query outcomes and cannot be deployed.")


if __name__ == "__main__":
    main()
