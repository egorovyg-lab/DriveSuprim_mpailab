"""Read-only component audit of frozen validation interventions; no simulator calls."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    output = args.root / "frozen_evaluation"
    report = json.loads((output / "report.json").read_text())
    frame = pd.read_csv(output / "per_scene.csv", dtype={"token": str})
    q = frame[frame.method == report["primary"]]
    print("PRIMARY:", report["primary"], "SCENES:", len(q))
    print("\nGAIN BY SESSION:")
    print(frame.pivot_table(index="session_id", columns="method", values="raw_gain_pp").to_string())
    cols = ["token", "session_id", "candidate_id", "raw_gain_pp", "raw_harm", "raw_quality_degraded"]
    cols += [c for c in q if c.startswith("raw_delta_")]
    print("\nPRIMARY INTERVENTIONS (all measured component deltas):")
    print(q.loc[q.intervention, cols].to_string(index=False))
    if not q.empty:
        worst = q.loc[q.raw_gain_pp.idxmin()]
        state = json.loads((output / "scenes" / (str(worst.token) + ".json")).read_text())
        base = state["outcomes"][str(state["choices"]["baseline"])]["raw"]
        selected = state["outcomes"][str(state["choices"][report["primary"]])]["raw"]
        keys = [c.removeprefix("raw_delta_") for c in q if c.startswith("raw_delta_")]
        keys += ["multiplicative_metrics_prod", "pdm_score", "aggregate_without_two_frame", "raw_progress_m"]
        print("\nWORST PRIMARY CASE:", worst.token)
        print(pd.DataFrame([dict(metric=k, baseline=base[k], selected=selected[k], delta=selected[k] - base[k])
                            for k in keys]).to_string(index=False))
    print("\nObserved validation outcomes only; no retraining or parameter selection.")


if __name__ == "__main__":
    main()
