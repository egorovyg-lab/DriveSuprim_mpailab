"""Read-only diagnostics of the frozen development run. No GPU or new labels.

Does not open calibration/test outcomes, fit models, or change experiment files.
The best already evaluated action is a lower bound on a candidate oracle.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from navsim.agents.drivemem import (
    METRICS, probabilities, nearest_donors, memory_correction, corrected_choice,
    metric_selection_score,
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_sample(root, record, index):
    path = root / "features" / (record["token"] + ".npz")
    if sha(path) != index[record["token"]]:
        raise RuntimeError(f"Feature checksum mismatch: {record['token']}")
    with np.load(path, allow_pickle=False) as data:
        sample = {k: data[k].copy() for k in data.files}
    sample.update(token=record["token"], session_id=record["session_id"], role=record["role"])
    return sample


def diagnose_query(query, bank, vocab, settings, threshold, saved):
    ids, distance = nearest_donors(query, bank, int(settings["neighbors"]), int(settings["seed"]))
    correction = memory_correction(query, bank, ids, vocab)
    strength = float(settings["strength"])
    chosen = corrected_choice(query, correction, strength)
    base = int(query["selected_local"])
    if chosen != saved["decisions"]["similarity_memory"] or base != saved["decisions"]["baseline"]:
        raise RuntimeError(f"Recomputed decisions differ from frozen evaluation: {query['token']}")
    donor_tokens = [str(bank[i]["token"]) for i in ids]
    if donor_tokens != saved["retrieval"]["donor_tokens"]:
        raise RuntimeError("Recomputed retrieval differs from frozen evaluation")
    p = probabilities(query)
    change = metric_selection_score(np.clip(p + strength * correction, 0, 1)) - metric_selection_score(p)
    scores = query["selection_scores"].astype(float)
    order = np.argsort(-scores, kind="stable")
    baseline_row = next(r for r in saved["rows"] if r["method"] == "baseline")
    sim_row = next(r for r in saved["rows"] if r["method"] == "similarity_memory")
    result = dict(token=query["token"], session_id=query["session_id"],
        baseline_score=baseline_row["raw_score"], donor_count=len(ids),
        donor_sessions=len({bank[i]["session_id"] for i in ids}),
        nearest_distance=distance if np.isfinite(distance) else None,
        threshold_pass=bool(distance <= threshold),
        baseline_top2_margin=float(scores[order[0]] - scores[order[1]]),
        correction_abs_mean=float(np.abs(correction).mean()),
        correction_candidate_std_mean=float(correction.std(axis=0).mean()),
        selection_delta_span=float(np.ptp(change)),
        similarity_intervention=chosen != base, similarity_gain_pp=sim_row["raw_gain_pp"],
        evaluated_union_actions=len(saved["outcomes"]),
        evaluated_union_best_gain_pp=100 * (max(v["raw"]["pdm_score"] for v in saved["outcomes"].values())
                                          - baseline_row["raw_score"]))
    for key in METRICS:
        result["delta_" + key] = sim_row["raw_" + key] - baseline_row["raw_" + key]
    return result, donor_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    identity = read_json(root / "identity.json")
    frozen = read_json(root / "split_identity.json")
    if frozen != {"splits": sha(root / "splits.json"), "inventory": sha(root / "inventory.csv")}:
        raise RuntimeError("Frozen splits/inventory changed")
    repo = Path(__file__).resolve().parents[3]
    module = "navsim/agents/drivemem.py"
    if sha(repo / module) != identity["sources"][module]:
        raise RuntimeError("Frozen predictor code changed")
    fit = read_json(root / "fit.json")
    index = read_json(root / "export_index.json")
    records = read_json(root / "splits.json")["records"]
    settings = identity["settings"]
    bank = []
    for record in records:
        if record["role"] != "memory":
            continue
        sample = load_sample(root, record, index)
        filename = record["token"] + ".json"
        path = root / "labels" / filename
        if sha(path) != fit["labels"][filename]:
            raise RuntimeError("Frozen bank labels changed")
        labels = read_json(path)
        if labels["feature_sha256"] != index[record["token"]] or labels["role"] != "memory":
            raise RuntimeError("Bank label provenance mismatch")
        sample["label_local_indices"] = np.asarray(labels["local_indices"], dtype=int)
        sample["label_values"] = np.asarray([[v["raw"][k] for k in METRICS] for v in labels["values"]])
        bank.append(sample)
    with np.load(root / "vocab.npz", allow_pickle=False) as data:
        vocab = data["vocab"].copy()
    rows, methods, used = [], [], Counter()
    for record in records:
        if record["role"] != "development":
            continue
        sample = load_sample(root, record, index)
        saved = read_json(root / "evaluation_development" / (record["token"] + ".json"))
        row, donors = diagnose_query(sample, bank, vocab, settings, float(fit["similarity_threshold"]), saved)
        rows.append(row)
        methods.extend(saved["rows"])
        used.update(donors)
    if not rows:
        raise RuntimeError("No development results")
    frame = pd.DataFrame(rows)
    method_frame = pd.DataFrame(methods)
    print("DEVELOPMENT ONLY; READ-ONLY; no GPU or new simulator calls")
    print("FIT:", json.dumps(fit | {"labels": f"{len(fit['labels'])} frozen label files"}, indent=2))
    print("\nRETRIEVAL / RANKING (selection margins use model-score units, not PDM points):")
    columns = ["baseline_score", "donor_count", "donor_sessions", "nearest_distance", "baseline_top2_margin",
               "correction_abs_mean", "correction_candidate_std_mean", "selection_delta_span",
               "evaluated_union_best_gain_pp"]
    print(frame[columns].describe().to_string())
    print(f"No compatible donors: {int((frame.donor_count == 0).sum())}/{len(frame)}")
    print(f"Pass distance threshold: {int(frame.threshold_pass.sum())}/{len(frame)}")
    print(f"Distinct retrieved episodes: {len(used)}/{len(bank)}")
    session_of = {s["token"]: s["session_id"] for s in bank}
    print("Retrieved donor sessions (episode occurrences):", dict(Counter(
        {sid: sum(n for token, n in used.items() if session_of[token] == sid) for sid in set(session_of.values())})))
    print("\nMETHODS BY SESSION:")
    print(method_frame.groupby(["session_id", "method"]).agg(
        scenes=("token", "size"), interventions=("intervention", "sum"),
        gain_pp=("raw_gain_pp", "mean"), harm=("raw_harm", "sum"),
        quality_degraded=("raw_quality_degraded", "sum")).to_string())
    print("\nALL INTERVENTIONS:")
    keys = ["token", "method", "candidate_id", "raw_gain_pp", "raw_harm", "raw_quality_degraded"]
    print(method_frame.loc[method_frame.intervention, keys].to_string(index=False))
    print("\nSIMILARITY SAFETY/QUALITY COMPONENT DIFFERENCES:")
    keys = ["token", "similarity_gain_pp"] + ["delta_" + k for k in METRICS]
    print(frame.loc[frame.similarity_intervention, keys].to_string(index=False))
    print("\nBest evaluated-union action is NOT a full/top-k oracle; unevaluated alternatives remain unknown.")


if __name__ == "__main__":
    main()
