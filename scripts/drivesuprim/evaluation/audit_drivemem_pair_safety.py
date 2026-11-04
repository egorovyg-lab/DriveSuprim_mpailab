"""Read-only utility optimism / safety-label audit. No fitting or threshold search."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_drivemem_action_support import load_frozen_bank, load_sample, read_json, sha
from navsim.agents.drivemem import METRICS, SAFETY, probabilities
from navsim.agents.drivemem_action_transfer import pool
from navsim.agents.drivemem_pair_utility import attach_utility, features, predict, choose


def safety_rows(sample, outcomes):
    base = int(sample["selected_local"])
    rows = []
    for i in pool(sample):
        for j in SAFETY:
            key = METRICS[j]
            before, after = outcomes[str(base)]["raw"][key], outcomes[str(i)]["raw"][key]
            rows.append(dict(role=sample["role"], session_id=sample["session_id"], token=sample["token"],
                metric=key, nonbaseline=int(i) != base, below_one=after < 1 - 1e-8,
                degraded=after + 1e-8 < before, improved=after > before + 1e-8))
    return rows


def case_rows(s, outcomes, method, predicted):
    ix = pool(s).tolist()
    base = int(s["selected_local"])
    chosen = choose(s, predicted)
    oracle = max(ix, key=lambda i: outcomes[str(i)]["raw"]["pdm_score"])
    p = probabilities(s)
    rows = []
    for kind, i in (("selected", chosen), ("oracle", oracle)):
        a, b = outcomes[str(base)]["raw"], outcomes[str(i)]["raw"]
        row = dict(token=s["token"], session_id=s["session_id"], method=method, kind=kind,
            candidate_id=int(s["candidate_ids"][i]), rank=ix.index(i) + 1,
            predicted_gain_pp=100 * float(predicted[ix.index(i)]),
            measured_gain_pp=100 * (b["pdm_score"] - a["pdm_score"]))
        row["optimism_pp"] = row["predicted_gain_pp"] - row["measured_gain_pp"]
        for j in SAFETY:
            key = METRICS[j]
            row["measured_delta_" + key] = b[key] - a[key]
            row["head_baseline_" + key] = float(p[base, j])
            row["head_action_" + key] = float(p[i, j])
            row["head_delta_" + key] = float(p[i, j] - p[base, j])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    root = args.experiment
    identity, fitted = read_json(root / "identity.json"), read_json(root / "fit.json")
    repo = Path(__file__).resolve().parents[3]
    for name, digest in identity["sources"].items():
        if sha(repo / name) != digest:
            raise RuntimeError("Utility code changed")
    completion = Path(identity["completion"])
    for name, key in (("identity.json", "completion_identity"), ("fit.json", "completion_fit"), ("report.json", "completed_report")):
        if sha(completion / name) != identity[key]:
            raise RuntimeError("Completed-M input changed")
    completed, completed_fit = read_json(completion / "identity.json"), read_json(completion / "fit.json")
    previous = Path(completed["previous"])
    for name, key in (("identity.json", "previous_identity"), ("fit.json", "previous_fit")):
        if sha(previous / name) != completed[key]:
            raise RuntimeError("Previous model identity changed")
    source, old_bank, vocab, old_fit, _ = load_frozen_bank(previous)
    validation = Path(identity["validation"])
    if str(source) != identity["source"] or str(source) != completed["source"] or str(validation) != completed["validation"]:
        raise RuntimeError("Source roots changed")
    for cohort in (source, validation):
        for name, digest in identity["query_inputs"][cohort.name].items():
            if sha(cohort / name) != digest:
                raise RuntimeError("Frozen cohort input changed")
        with np.load(cohort / "vocab.npz", allow_pickle=False) as data:
            if not np.array_equal(vocab, data["vocab"]):
                raise RuntimeError("Vocabulary mismatch")
    training, bank = [], []
    index = read_json(source / "export_index.json")
    if set(completed_fit["memory_labels"]) != {s["token"] + ".json" for s in old_bank}:
        raise RuntimeError("M membership changed")
    for r in read_json(source / "splits.json")["records"]:
        if r["role"] not in ("memory", "fit"):
            continue
        s = load_sample(source, r, index)
        folder, expected = (completion / "memory_labels", completed_fit["memory_labels"]) if r["role"] == "memory" else (previous / "fit_labels", old_fit["sidecar_labels"])
        path = folder / (s["token"] + ".json")
        if sha(path) != expected[path.name]:
            raise RuntimeError("M/F labels changed")
        state = read_json(path)
        if state["token"] != s["token"]:
            raise RuntimeError("Training token mismatch")
        training.extend(safety_rows(s, state["outcomes"]))
        if s["role"] == "memory":
            bank.append(attach_utility(s, state["outcomes"]))
    frame = pd.DataFrame(training)
    frame = frame[frame.nonbaseline]
    print("M/F TOP16 SAFETY SUPPORT (baseline excluded; candidate rows are correlated):")
    print(frame.groupby(["role", "metric"]).agg(actions=("token", "size"),
        below_one=("below_one", "sum"), degraded=("degraded", "sum"), improved=("improved", "sum")).to_string())
    bad = frame[frame.degraded]
    print("\nDEGRADATION SUPPORT BY SESSION:")
    print(bad.groupby(["role", "metric", "session_id"]).size().to_string() if len(bad) else "None")
    frozen = read_json(root / "query_decisions.json")
    if frozen["fit_sha256"] != sha(root / "fit.json"):
        raise RuntimeError("Decision/fit mismatch")
    models = {}
    for method, digest in fitted["models"].items():
        path = root / (method + ".npz")
        if sha(path) != digest:
            raise RuntimeError("Utility model changed")
        with np.load(path, allow_pickle=False) as data:
            models[method] = {k: data[k].copy() for k in data.files}
    prepared = []
    for cohort in (source, validation):
        index = read_json(cohort / "export_index.json")
        for r in read_json(cohort / "splits.json")["records"]:
            if r["role"] != "development":
                continue
            s = load_sample(cohort, r, index)
            for method, model in models.items():
                x = features(s, vocab, bank if method == "direct_memory" else None, fitted["h"], fitted["seed"])
                prediction = predict(x, model)
                if choose(s, prediction) != frozen["cohorts"][cohort.name][s["token"]][method]:
                    raise RuntimeError("Frozen choice not reproduced")
                prepared.append((cohort, s, method, prediction))
    # All inference and parity checks precede query outcome lookup.
    expected = read_json(completion / "report.json")["oracle_inputs"]
    rows = []
    for cohort, s, method, prediction in prepared:
        path = cohort / "development_oracle/scenes" / (s["token"] + ".json")
        if sha(path) != expected[cohort.name][s["token"]]:
            raise RuntimeError("Query oracle changed")
        state = read_json(path)
        if state["token"] != s["token"]:
            raise RuntimeError("Query token mismatch")
        rows.extend(dict(cohort=cohort.name, **r) for r in case_rows(s, state["outcomes"], method, prediction))
    frame = pd.DataFrame(rows)
    print("\nSELECTED-ACTION PREDICTION ERROR (descriptive, not calibration):")
    selected = frame[frame.kind == "selected"]
    print(selected.groupby(["cohort", "method"])[["predicted_gain_pp", "measured_gain_pp", "optimism_pp"]].mean().to_string())
    print("\nSAFETY FAILURES AND TWO LARGE ORACLE CASES:")
    keys = ["measured_delta_" + METRICS[j] for j in SAFETY]
    bad_tokens = set(selected.loc[(selected[keys] < -1e-8).any(axis=1), "token"])
    bad_tokens.update(("c77e7e3d0ff458e4", "a5e429799b6753f5"))
    print(frame[frame.token.isin(bad_tokens)].to_string(index=False))
    print("\nRead-only: no retraining, threshold selection, scorer calls or artifact writes. "
          "Heads are uncalibrated predictions; zero observed harm is not a guarantee. "
          "Both query cohorts remain inspected development data.")


if __name__ == "__main__":
    main()
