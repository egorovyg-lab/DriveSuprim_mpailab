"""Read-only bank label/rank/geometry support audit on already evaluated cohorts.

Oracle outcomes are used only to diagnose action support, never to fit a model.
No GPU, simulator calls, downloads, parameter changes or artifact writes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from navsim.agents.drivemem import METRICS, geometry, nearest_donors
from diagnose_drivemem_memory import sha, read_json, load_sample


def ranks(sample):
    result = np.empty(len(sample["candidate_ids"]), dtype=int)
    result[np.argsort(-sample["selection_scores"], kind="stable")] = np.arange(1, len(result) + 1)
    return result


def label_bands(bank):
    rows = []
    for name, lo, hi in (("1-8", 1, 8), ("9-16", 9, 16), ("17-256", 17, 256)):
        count, scenes, slots = 0, 0, 0
        failures = {k: 0 for k in METRICS}
        for sample in bank:
            order = ranks(sample)
            mask = (order[sample["label_local_indices"]] >= lo) & (order[sample["label_local_indices"]] <= hi)
            count += int(mask.sum())
            scenes += int(mask.any())
            slots += int(((order >= lo) & (order <= hi)).sum())
            for j, metric in enumerate(METRICS):
                failures[metric] += int((sample["label_values"][mask, j] < 1 - 1e-8).sum())
        rows.append(dict(rank_band=name, labels=count, scenes_with_labels=scenes, candidate_slots=slots,
                         labelled_fraction=count / slots if slots else None, **{"below_one_" + k: v for k, v in failures.items()}))
    return rows


def support(query, local, bank, vocab, h, seed):
    if h <= 0:
        raise ValueError("Bandwidth must be positive")
    ids, scene_distance = nearest_donors(query, bank, 5, seed)
    minima, ess, exact, global_minima, global_exact = [], [], 0, [], 0
    all_ids, _ = nearest_donors(query, bank, len(bank), seed)
    qg = geometry(vocab[query["candidate_ids"][[local]]])[0]
    for i in all_ids:
        s = bank[i]
        dg = geometry(vocab[s["candidate_ids"][s["label_local_indices"]]])
        d = np.square(qg - dg).mean(axis=1)
        global_minima.append(float(d.min()))
        global_exact += int(np.any(d <= 1e-12))
        if i not in ids:
            continue
        w = np.exp(-(d - d.min()) / h)
        w /= w.sum()
        minima.append(float(d.min()))
        ess.append(float(1 / np.square(w).sum()))
        exact += int(np.any(d <= 1e-12))
    result = dict(donors=len(ids), donor_sessions=len({bank[i]["session_id"] for i in ids}),
        scene_distance=float(scene_distance) if np.isfinite(scene_distance) else None,
        min_distance_over_h=min(minima) / h if minima else None,
        mean_nearest_distance_over_h=float(np.mean(minima)) / h if minima else None,
        exact_supporting_donors=exact, kernel_ess=float(np.mean(ess)) if ess else None)
    result.update(compatible_episodes=len(all_ids), global_exact_supporting_episodes=global_exact,
                  global_min_distance_over_h=min(global_minima) / h if global_minima else None)
    return result


def audit_cohort(root, bank, vocab, h, seed):
    frozen = read_json(root / "split_identity.json")
    if frozen != dict(splits=sha(root / "splits.json"), inventory=sha(root / "inventory.csv")):
        raise RuntimeError("Cohort selection changed")
    with np.load(root / "vocab.npz", allow_pickle=False) as data:
        if not np.array_equal(data["vocab"], vocab):
            raise RuntimeError("Vocabulary differs across cohorts")
    index = read_json(root / "export_index.json")
    rows = []
    for record in read_json(root / "splits.json")["records"]:
        if record["role"] != "development":
            continue
        s = load_sample(root, record, index)
        state = read_json(root / "development_oracle/scenes" / (s["token"] + ".json"))
        ix = np.argsort(-s["selection_scores"], kind="stable")[:16].tolist()
        base = int(s["selected_local"])
        if base != ix[0] or any(str(i) not in state["outcomes"] for i in ix):
            raise RuntimeError("Incomplete top-16 or wrong baseline")
        scores = {i: state["outcomes"][str(i)]["raw"]["pdm_score"] for i in ix}
        best8 = max(ix[:8], key=lambda i: scores[i])
        best16 = max(ix, key=lambda i: scores[i])
        row = dict(token=s["token"], session_id=s["session_id"], baseline=scores[base],
                   gain8_pp=100 * (scores[best8] - scores[base]), gain16_pp=100 * (scores[best16] - scores[base]),
                   increment9_16_pp=100 * (scores[best16] - scores[best8]), oracle_rank=ix.index(best16) + 1,
                   oracle_vocab=int(s["candidate_ids"][best16]))
        for name, local in (("baseline", base), ("oracle", best16)):
            row.update({name + "_" + k: v for k, v in support(s, local, bank, vocab, h, seed).items()})
        rows.append(row)
    if not rows:
        raise RuntimeError("No development-role cohort records")
    frame = pd.DataFrame(rows)
    print("\nCOHORT:", root)
    print("MEAN RESERVE:", frame[["gain8_pp", "gain16_pp", "increment9_16_pp"]].mean().to_dict())
    print("BY SESSION:")
    print(frame.groupby("session_id")[["gain8_pp", "gain16_pp", "increment9_16_pp"]].mean().to_string())
    improved = frame[frame.gain16_pp > 1e-6]
    print("ORACLE-POSITIVE CASES, SUPPORT IN THE RETRIEVED FIVE DONORS:")
    columns = ["token", "session_id", "gain8_pp", "gain16_pp", "increment9_16_pp", "oracle_rank", "oracle_vocab",
               "oracle_min_distance_over_h", "oracle_mean_nearest_distance_over_h", "oracle_exact_supporting_donors",
               "oracle_global_min_distance_over_h", "oracle_global_exact_supporting_episodes"]
    print(improved[columns].to_string(index=False))
    print("GEOMETRY SUPPORT MEANS (all / oracle-positive):")
    keys = [c for c in frame if c.endswith(("distance_over_h", "supporting_donors", "kernel_ess", "scene_distance"))]
    print(pd.DataFrame(dict(all_queries=frame[keys].mean(), oracle_positive=improved[keys].mean())).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("validation", type=Path)
    args = parser.parse_args()
    experiment = args.experiment
    identity, fit = read_json(experiment / "identity.json"), read_json(experiment / "fit.json")
    source = Path(identity["source"])
    repo = Path(__file__).resolve().parents[3]
    for name, digest in identity["sources"].items():
        if sha(repo / name) != digest:
            raise RuntimeError("Frozen model code changed")
    for name, digest in identity["inputs"].items():
        if sha(source / name) != digest:
            raise RuntimeError("Frozen source input changed")
    if sha(experiment / "ridge.npz") != fit["ridge_sha256"]:
        raise RuntimeError("Frozen fitted model changed")
    index, bank = read_json(source / "export_index.json"), []
    for record in read_json(source / "splits.json")["records"]:
        if record["role"] != "memory":
            continue
        sample = load_sample(source, record, index)
        path = source / "labels" / (record["token"] + ".json")
        if sha(path) != identity["labels"][path.name]:
            raise RuntimeError("Memory labels changed")
        labels = read_json(path)
        if labels["role"] != "memory" or labels["feature_sha256"] != index[record["token"]]:
            raise RuntimeError("Memory provenance mismatch")
        sample["label_local_indices"] = np.asarray(labels["local_indices"], dtype=int)
        sample["label_values"] = np.asarray([[v["raw"][k] for k in METRICS] for v in labels["values"]])
        bank.append(sample)
    if not bank:
        raise RuntimeError("Empty memory bank")
    print("MEMORY LABEL COVERAGE BY DONOR MODEL RANK:")
    print(pd.DataFrame(label_bands(bank)).to_string(index=False))
    missing = sum(len(set(np.argsort(-s["selection_scores"], kind="stable")[:16]) - set(s["label_local_indices"])) for s in bank)
    print(json.dumps(dict(memory_scenes=len(bank), missing_M_top16_actions=missing,
        scorer_calls_if_M_top16_completed=2 * (missing + len(bank)),
        note="Cost estimate only. No scoring or label changes performed."), indent=2))
    with np.load(source / "vocab.npz", allow_pickle=False) as data:
        vocab = data["vocab"].copy()
    h = fit["configs"][fit["primary"]]["h"]
    for cohort in (source, args.validation):
        audit_cohort(cohort, bank, vocab, h, int(identity["seed"]))
    print("\nRanks are scene-relative. Geometric support is not evidence of outcome transferability. "
          "Oracle data used for diagnosis only; all inspected cohorts remain development data.")


if __name__ == "__main__":
    main()
