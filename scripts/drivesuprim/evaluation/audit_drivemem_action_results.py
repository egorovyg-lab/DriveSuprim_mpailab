"""Read-only concentration and largest-intervention component audit. No simulator."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    run = args.experiment
    identity, fit, report = (read(run / name) for name in ("identity.json", "fit.json", "report.json"))
    if sha(run / "ridge.npz") != fit["ridge_sha256"]:
        raise RuntimeError("Frozen ridge changed")
    decisions = read(run / "development_decisions.json")
    if decisions["fit_sha256"] != sha(run / "fit.json"):
        raise RuntimeError("Decision/fit identity mismatch")
    frame = pd.read_csv(run / "development_per_scene.csv", dtype={"token": str})
    primary = report["primary"]
    rows = frame[frame.method == primary]
    if rows.empty:
        raise RuntimeError("No primary results")
    largest = rows.loc[rows.raw_gain_pp.idxmax()]
    token = str(largest.token)
    state = next(item for item in decisions["decisions"] if item["token"] == token)
    base_index, new_index = state["choices"]["baseline"], state["choices"][primary]
    source = Path(identity["source"])
    path = source / "development_oracle/scenes" / (token + ".json")
    if sha(path) != report["oracle_inputs"][token]:
        raise RuntimeError("Oracle changed since evaluation")
    outcomes = read(path)["outcomes"]
    baseline, selected = outcomes[str(base_index)], outcomes[str(new_index)]
    if not np.isclose(100 * (selected["raw"]["pdm_score"] - baseline["raw"]["pdm_score"]), largest.raw_gain_pp):
        raise RuntimeError("CSV and frozen outcomes disagree")
    total = float(rows.raw_gain_pp.sum())
    other = rows.drop(largest.name)
    print(json.dumps(dict(primary=primary, token=token, session_id=largest.session_id,
        baseline_local=base_index, selected_local=new_index, selected_vocab=int(largest.candidate_id),
        total_gain_sum_pp=total, largest_gain_pp=float(largest.raw_gain_pp),
        largest_share_of_net_gain=float(largest.raw_gain_pp / total) if total > 0 else None,
        gain_without_largest_micro_pp=float(other.raw_gain_pp.mean()),
        gain_without_largest_macro_pp=float(other.groupby("session_id").raw_gain_pp.mean().mean())), indent=2))
    keys = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
            "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
            "history_comfort", "multiplicative_metrics_prod", "pdm_score", "aggregate_without_two_frame", "raw_progress_m"]
    for version in ("raw", "filtered"):
        a, b = baseline[version], selected[version]
        print("\n" + version.upper() + " COMPONENTS:")
        table = pd.DataFrame([dict(metric=k, baseline=a[k], selected=b[k], delta=b[k] - a[k]) for k in keys])
        print(table.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
        for value in (a, b):
            if not np.isclose(value["pdm_score"], value["aggregate_without_two_frame"], atol=1e-8, rtol=1e-7):
                if version == "raw":
                    raise RuntimeError("Stored raw scalar/aggregate mismatch")
                print("FILTER AUDIT: stored filtered aggregate differs from scalar score; inspect snapshots.")
        print("Weighted metric snapshots:", json.dumps({name: {k: value[k] for k in ("weighted_metrics", "weighted_metrics_array")}
            for name, value in (("baseline", a), ("selected", b))}))
    print("\nAll data were read from existing outcomes; no reevaluation or mutation.")


if __name__ == "__main__":
    main()
