"""Read-only inspection of frozen fit and already evaluated development CSVs."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    fitted = json.loads((root / "fit.json").read_text(encoding="utf-8"))
    print("FROZEN F-CV SELECTION:")
    print(json.dumps({key: fitted[key] for key in ("primary", "selected")}, indent=2))
    methods = ["memory_guarded", "no_memory_guarded", "memory_component", "no_memory_component", "random_component", "shuffled_component"]
    for cohort in ("budget20gb_v1", "validation_cached4_v1"):
        frame = pd.read_csv(root / (cohort + "_per_scene.csv"), dtype={"token": str, "session_id": str})
        chosen = frame[frame.method.isin(methods)]
        print("\nCOHORT:", cohort)
        print("GAIN PP BY SESSION:")
        print(chosen.pivot_table(index="session_id", columns="method", values="raw_gain_pp", aggfunc="mean").to_string())
        for method in ("memory_guarded", "memory_component"):
            rows = frame[frame.method == method].copy()
            if rows.token.duplicated().any():
                raise ValueError("Duplicate scene for " + method)
            active = rows[rows.intervention].copy()
            active["baseline_score"] = active.raw_score - active.raw_gain_pp / 100
            columns = ["token", "session_id", "candidate_id", "baseline_score", "raw_score", "raw_gain_pp", "raw_harm", "raw_quality_degraded"]
            columns += [c for c in frame if c.startswith("raw_delta_")]
            print("\nINTERVENTIONS:", method)
            print(active[columns].to_string(index=False))
            session_gain = rows.groupby("session_id").raw_gain_pp.mean()
            print("LEAVE-ONE-SESSION-OUT (diagnostic; no refitting):")
            for sid in session_gain.index:
                remaining = session_gain.drop(sid)
                print(sid, "excluded ->", f"{remaining.mean():.6f} pp" if len(remaining) else "undefined")


if __name__ == "__main__":
    main()
