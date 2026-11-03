"""New-session evaluation of frozen action-transfer models. No fitting or oracle."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.agents.drivemem_action_transfer import transfer, choose, ridge_predict
from navsim.planning.script.run_drivemem_action_transfer import record, summary


def decide(sample, bank, vocab, model, fit, seed):
    if fit["topk"] != 16 or fit["neighbors"] != 5 or fit["beta"] != .5:
        raise ValueError("Unsupported frozen protocol")
    result = dict(baseline=int(sample["selected_local"]), no_memory_ridge_full=choose(sample, ridge_predict(sample, vocab, model)))
    for family, spec in fit["configs"].items():
        c, _ = transfer(sample, bank, vocab, spec["h"], spec["contrast"], seed)
        result[family] = choose(sample, c)
    spec = fit["configs"][fit["primary"]]
    for i in range(3):
        c, _ = transfer(sample, bank, vocab, spec["h"], spec["contrast"], seed + i, random=True)
        result[f"random_{i}"] = choose(sample, c)
    c, _ = transfer(sample, bank, vocab, .25, seed=seed, scene_only=True)
    result["scene_only"] = choose(sample, c)
    return result


def evaluate_union(samples, choices, evaluate, output, read, write, verify_baseline):
    rows = []
    for number, sample in enumerate(samples):
        token = sample["token"]
        selected = choices[token]
        path = output / "scenes" / (token + ".json")
        state = read(path) if path.exists() else dict(token=token, choices=selected, outcomes={})
        if state["token"] != token or state["choices"] != selected:
            raise RuntimeError("Frozen validation decisions changed")
        baseline = selected["baseline"]
        repeated, _ = evaluate(token, int(sample["candidate_ids"][baseline]))
        if str(baseline) in state["outcomes"]:
            verify_baseline(state["outcomes"][str(baseline)], repeated)
        else:
            state["outcomes"][str(baseline)] = repeated
            write(path, state)
        for i in sorted(set(selected.values())):
            if str(i) not in state["outcomes"]:
                state["outcomes"][str(i)], _ = evaluate(token, int(sample["candidate_ids"][i]))
                write(path, state)
        # Verify a fresh baseline again before committing per-query results.
        repeated, _ = evaluate(token, int(sample["candidate_ids"][baseline]))
        verify_baseline(state["outcomes"][str(baseline)], repeated)
        rows.extend(dict(method=method, **record(sample, state["outcomes"], i)) for method, i in selected.items())
        print(f"VALIDATION {number + 1}/{len(samples)} token={token} actions={len(set(selected.values()))}", flush=True)
    return rows


def main(cfg):
    from navsim.planning.script.run_drivemem_memory import (
        verify_identity, read_json as read, write_json as write, sha, sample as load_sample,
        training_samples, evaluators,
    )
    from navsim.planning.script.run_drivemem_development_oracle import verify_baseline
    root = Path(cfg.drivemem.root)
    verify_identity(cfg, root)
    protocol = read(root / "validation_protocol.json")
    source, experiment = Path(protocol["source"]), Path(protocol["experiment"])
    if sha(source / "identity.json") != protocol["source_identity"] or sha(experiment / "identity.json") != protocol["model_identity"]:
        raise RuntimeError("Frozen source/model identity changed")
    if sha(experiment / "fit.json") != protocol["fit_sha256"]:
        raise RuntimeError("Frozen fit changed")
    fit, identity = read(experiment / "fit.json"), read(experiment / "identity.json")
    for name, digest in identity["inputs"].items():
        if sha(source / name) != digest:
            raise RuntimeError(f"Frozen source input changed: {name}")
    for name, digest in identity["labels"].items():
        if sha(source / "labels" / name) != digest:
            raise RuntimeError("Frozen donor/fit source labels changed")
    repo = Path(__file__).resolve().parents[3]
    for name, digest in identity["sources"].items():
        if sha(repo / name) != digest:
            raise RuntimeError(f"Frozen model code changed: {name}")
    for name, digest in fit["sidecar_labels"].items():
        if sha(experiment / "fit_labels" / name) != digest:
            raise RuntimeError("Frozen F labels changed")
    if sha(experiment / "ridge.npz") != fit["ridge_sha256"]:
        raise RuntimeError("Frozen ridge changed")
    if sha(root / "vocab.npz") != read(root / "vocab_identity.json")["sha256"]:
        raise RuntimeError("Validation vocabulary file changed")
    with np.load(root / "vocab.npz", allow_pickle=False) as new, np.load(source / "vocab.npz", allow_pickle=False) as old:
        if not np.array_equal(new["vocab"], old["vocab"]):
            raise RuntimeError("Validation uses a different vocabulary")
    bank = [s for s in training_samples(cfg, source) if s["role"] == "memory"]
    records = read(root / "splits.json")["records"]
    forbidden = set(read(root / "splits.json")["all_excluded_sessions"])
    if not records or any(r["role"] != "development" or r["session_id"] in forbidden for r in records):
        raise RuntimeError("Invalid or overlapping validation sessions")
    samples = [load_sample(root, r) for r in records]
    if set(read(root / "export_index.json")) != {r["token"] for r in records}:
        raise RuntimeError("Validation export incomplete or contains extra tokens")
    with np.load(source / "vocab.npz", allow_pickle=False) as data:
        vocab = data["vocab"].copy()
    with np.load(experiment / "ridge.npz", allow_pickle=False) as data:
        model = {k: data[k].copy() for k in data.files}
    output = root / "frozen_evaluation"
    provenance = dict(protocol_sha256=sha(root / "validation_protocol.json"),
        runner_sha256=sha(Path(__file__)), features_sha256=sha(root / "export_index.json"),
        cache_index_sha256=sha(root / "cache_index.json"), vocab_sha256=sha(root / "vocab.npz"),
        splits_sha256=sha(root / "splits.json"), fit_sha256=sha(experiment / "fit.json"))
    marker = output / "identity.json"
    if marker.exists() and read(marker) != provenance:
        raise RuntimeError("Validation runner or inputs changed; preserve old run and investigate")
    write(marker, provenance)
    choices = {s["token"]: decide(s, bank, vocab, model, fit, int(identity["seed"])) for s in samples}
    decision_path = output / "decisions.json"
    if decision_path.exists() and read(decision_path) != choices:
        raise RuntimeError("Frozen decisions changed")
    write(decision_path, choices)  # All proposals fixed before ANY query outcome is requested.
    evaluate, _ = evaluators(cfg, root)
    rows = evaluate_union(samples, choices, evaluate, output, read, write, verify_baseline)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_scene.csv", index=False)
    frame.groupby(["session_id", "method"]).agg(gain_pp=("raw_gain_pp", "mean"),
        interventions=("intervention", "sum"), harm=("raw_harm", "sum")).to_csv(output / "by_session.csv")
    pivot = frame.pivot(index=["session_id", "token"], columns="method", values="raw_gain_pp")
    difference = (pivot[fit["primary"]] - pivot.no_memory_ridge_full).groupby("session_id").mean().to_numpy()
    pick = np.random.default_rng(int(identity["seed"])).integers(0, len(difference), size=(2000, len(difference)))
    report = dict(status="complete", primary=fit["primary"], scenes=len(samples),
        sessions=len({s["session_id"] for s in samples}), methods=summary(rows, int(identity["seed"])),
        paired_primary_minus_ridge=dict(gain_macro_pp=float(difference.mean()),
            ci95_sessions=np.quantile(difference[pick].mean(axis=1), [.025, .975]).tolist()),
        note="Frozen-model new-session check in cached archives. No refit, no oracle, local PDM not full EPDMS. "
             "Four sessions are preliminary and archive selection bias remains; no calibrated safety guarantee.")
    write(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    import hydra
    hydra.main(config_path="config/pdm_scoring", config_name="default_drivemem_memory", version_base=None)(main)()
