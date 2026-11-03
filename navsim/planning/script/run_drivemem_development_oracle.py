"""Top-k oracle on the frozen DEVELOPMENT split; no training or GPU inference.

Reuses evaluated actions, verifies baseline parity, scores reference/action pairs
individually via the original evaluator, and resumes per candidate. These are
development diagnostic labels, never memory/fit supervision or held-out results.
"""
import json
import os
from pathlib import Path

import numpy as np

from navsim.agents.drivemem import METRICS, SAFETY


def ranked_indices(sample, k):
    scores = np.asarray(sample["selection_scores"], dtype=float)
    if not 1 <= k <= len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid top-k or selection scores")
    base = int(sample["selected_local"])
    if base != int(np.argmax(scores)):
        raise RuntimeError("Saved baseline is not the original model argmax")
    return np.argsort(-scores, kind="stable")[:k].tolist()


def best_action(outcomes, indices, base, version="raw", safe=False):
    baseline = outcomes[str(base)][version]
    possible = [base]
    for index in indices:
        values = outcomes[str(index)][version]
        if not safe or all(values[METRICS[j]] + 1e-8 >= baseline[METRICS[j]] for j in SAFETY):
            possible.append(index)
    # Baseline first preserves the original choice in exact score ties.
    return max(possible, key=lambda index: outcomes[str(index)][version]["pdm_score"])


def verify_baseline(original, repeated):
    for version in ("raw", "filtered"):
        for key in (*METRICS, "pdm_score", "aggregate_without_two_frame"):
            if not np.isclose(original[version][key], repeated[version][key], atol=1e-8, rtol=1e-7):
                raise RuntimeError(f"Baseline parity failure: {version}/{key}")


def audit(root, samples, saved, evaluate, k, provenance, read_json, write_json, dry_run=False):
    if not samples or any(s["role"] != "development" for s in samples):
        raise ValueError("A nonempty DEVELOPMENT-only sample list is required")
    out = root / "development_oracle"
    marker = out / "identity.json"
    if marker.exists() and read_json(marker) != provenance:
        raise RuntimeError("Oracle inputs/code changed; preserve existing audit and investigate")
    prepared, new_actions = [], 0
    for sample in samples:
        token = sample["token"]
        base = int(sample["selected_local"])
        indices = ranked_indices(sample, k)
        if saved[token]["decisions"]["baseline"] != base:
            raise RuntimeError("Evaluation baseline differs from frozen features")
        path = out / "scenes" / (token + ".json")
        if path.exists() and not marker.exists():
            raise RuntimeError("Oracle scene exists without its provenance manifest")
        state = read_json(path) if path.exists() else {"token": token, "outcomes": dict(saved[token]["outcomes"])}
        if state["token"] != token:
            raise RuntimeError("Oracle token mismatch")
        new_actions += sum(str(i) not in state["outcomes"] for i in indices)
        prepared.append((sample, path, state, indices))
    plan = {"status": "planned", "role": "development", "scenes": len(samples), "topk": k,
            "new_candidate_evaluations": new_actions, "baseline_repeat_evaluations": len(samples),
            "raw_filtered_scorer_calls": 2 * (new_actions + len(samples)),
            "note": "No GPU/new images. Individual reference/candidate pairs; not batched progress normalization."}
    print(json.dumps(plan, indent=2), flush=True)
    if dry_run:
        return plan
    write_json(marker, provenance)
    rows, topk_gains = [], {n: [] for n in sorted({1, k} | {n for n in (8, 16, 32, 64) if n <= k})}
    for number, (sample, path, state, indices) in enumerate(prepared):
        token, base = sample["token"], int(sample["selected_local"])
        original = saved[token]["outcomes"][str(base)]
        repeated, _ = evaluate(token, int(sample["candidate_ids"][base]))
        verify_baseline(original, repeated)
        verify_baseline(original, state["outcomes"][str(base)])
        for index in indices:
            if str(index) in state["outcomes"]:
                continue
            values, _ = evaluate(token, int(sample["candidate_ids"][index]))
            state["outcomes"][str(index)] = values
            write_json(path, state)  # Every completed candidate can resume.
            print(f"ORACLE scene={number + 1}/{len(samples)} token={token} model_rank={indices.index(index) + 1}/{k}", flush=True)
        write_json(path, state)
        row = {"token": token, "session_id": sample["session_id"], "n_candidates": k}
        for version in ("raw", "filtered"):
            baseline = original[version]["pdm_score"]
            row[version + "_baseline"] = baseline
            for safe in (False, True):
                label = version + ("_safe_oracle" if safe else "_oracle")
                index = best_action(state["outcomes"], indices, base, version, safe)
                score = state["outcomes"][str(index)][version]["pdm_score"]
                row[label] = score
                row[label + "_gain_pp"] = 100 * (score - baseline)
                row[label + "_model_rank"] = indices.index(index) + 1
        for n in topk_gains:
            best = best_action(state["outcomes"], indices[:n], base)
            topk_gains[n].append(100 * (state["outcomes"][str(best)]["raw"]["pdm_score"] - original["raw"]["pdm_score"]))
        rows.append(row)
    report = {"status": "complete", "role": "development", "topk": k, "scenes": len(rows),
              "sessions": len({r["session_id"] for r in rows}),
              "raw_baseline_mean": float(np.mean([r["raw_baseline"] for r in rows])),
              "topk_mean_gain_pp": {str(n): float(np.mean(gains)) for n, gains in topk_gains.items()},
              "note": "Local PDM without two-frame extended comfort, not full EPDMS. "
                      "Oracle uses development future outcomes; cannot be deployed or used as donor labels. "
                      "Small archive-budgeted sample; not a representative benchmark."}
    sessions = sorted({r["session_id"] for r in rows})
    for key in ("raw_oracle", "raw_safe_oracle", "filtered_oracle", "filtered_safe_oracle"):
        gains = np.array([r[key + "_gain_pp"] for r in rows])
        by_session = np.array([np.mean([r[key + "_gain_pp"] for r in rows if r["session_id"] == s]) for s in sessions])
        counts = np.array([sum(r["session_id"] == s for r in rows) for s in sessions])
        pick = np.random.default_rng(20261027).integers(0, len(sessions), size=(2000, len(sessions)))
        boot = (by_session[pick] * counts[pick]).sum(axis=1) / counts[pick].sum(axis=1)
        report[key] = {"score_mean": float(np.mean([r[key] for r in rows])),
                       "gain_mean_pp": float(gains.mean()), "improved_scenes": int((gains > 1e-6).sum()),
                       "gain_ci95_sessions": np.quantile(boot, [.025, .975]).tolist()}
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / f"scene_summary_top{k}.csv", index=False)
    write_json(out / f"report_top{k}.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(cfg):
    from navsim.planning.script.run_drivemem_memory import (
        verify_identity, read_json, write_json, sha, sample, evaluators,
    )
    root = Path(cfg.drivemem.root)
    verify_identity(cfg, root)
    records = [r for r in read_json(root / "splits.json")["records"] if r["role"] == "development"]
    samples = [sample(root, r) for r in records]
    paths = {r["token"]: root / "evaluation_development" / (r["token"] + ".json") for r in records}
    saved = {token: read_json(path) for token, path in paths.items()}
    provenance = {"schema_version": 1, "role": "development", "runner_sha256": sha(Path(__file__)),
                  "inputs": {name: sha(root / name) for name in (
                      "identity.json", "split_identity.json", "vocab.npz", "export_index.json", "cache_index.json")},
                  "development_evaluations": {t: sha(p) for t, p in paths.items()}}
    # Construct the CPU simulator lazily: planning needs no simulation at all.
    engine = []

    def evaluate(token, candidate):
        if not engine:
            engine.append(evaluators(cfg, root)[0])
        return engine[0](token, candidate)

    audit(root, samples, saved, evaluate, int(os.environ.get("DRIVEMEM_ORACLE_TOPK", "16")),
          provenance, read_json, write_json, dry_run=str(cfg.drivemem.stage).endswith("-plan"))


if __name__ == "__main__":
    import hydra
    hydra.main(config_path="config/pdm_scoring", config_name="default_drivemem_memory", version_base=None)(main)()
