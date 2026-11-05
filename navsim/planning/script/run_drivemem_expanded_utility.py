"""Data-only refit of frozen pair-utility algorithm; CPU, no new labels."""
import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/drivesuprim/evaluation"))
from audit_drivemem_action_support import load_frozen_bank, load_sample, read_json as read, sha
from navsim.agents.drivemem_pair_utility import attach_utility, features, predict, choose
from navsim.planning.script.run_drivemem_pair_utility import fit_models, evaluate, METHODS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def save(path, **arrays):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def partition(samples):
    if len({s["token"] for s in samples}) != len(samples):
        raise ValueError("Duplicate training tokens")
    bank = [s for s in samples if s["role"] == "memory"]
    fitting = [s for s in samples if s["role"] == "fit"]
    if len(bank) + len(fitting) != len(samples) or not bank or not fitting:
        raise ValueError("Only nonempty M/F may train")
    if {s["session_id"] for s in bank} & {s["session_id"] for s in fitting}:
        raise ValueError("M/F session overlap")
    return bank, fitting


def load_inputs(prior, addition):
    bindings = {}
    def bind(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise RuntimeError(f"Frozen input changed: {path}")
        bindings[str(path)] = digest
    old = read(prior / "identity.json")
    for name, digest in old["sources"].items():
        bind(REPO / name, digest)
    completion = Path(old["completion"])
    for name, key in (("identity.json", "completion_identity"), ("fit.json", "completion_fit"), ("report.json", "completed_report")):
        bind(completion / name, old[key])
    completed, completed_fit = read(completion / "identity.json"), read(completion / "fit.json")
    bind(REPO / "navsim/planning/script/run_drivemem_memory_completion.py", completed["runner_sha256"])
    previous = Path(completed["previous"])
    bind(previous / "identity.json", completed["previous_identity"])
    bind(previous / "fit.json", completed["previous_fit"])
    source, _, vocab, original_fit, original_identity = load_frozen_bank(previous)
    validation = Path(old["validation"])
    if str(source) != old["source"] or str(source) != completed["source"] or str(validation) != completed["validation"]:
        raise RuntimeError("Old input roots changed")
    for cohort in (source, validation):
        for name, digest in old["query_inputs"][cohort.name].items():
            bind(cohort / name, digest)
    for name, digest in original_identity["labels"].items():
        bind(source / "labels" / name, digest)
    protocol = read(addition / "expansion_protocol.json")
    selection = read(addition / "label_selection.json")
    report = read(addition / "support_report.json")
    if report["status"] != "complete" or report["completed_scenes"] != 32 or len(selection["records"]) != 32:
        raise ValueError("Complete shard-174 M/F labels required")
    if protocol.get("shard") != 174 or protocol["session_count"] != 4 or protocol["source"] != str(source):
        raise ValueError("Unexpected additional dataset")
    if protocol.get("preparer_path") != "download/prepare_drivemem_new_mf.py":
        raise ValueError("Unexpected collection preparer")
    for name, key in (("expansion_protocol.json", "protocol_sha256"), ("export_index.json", "export_index"),
                      ("vocab.npz", "vocab"), ("split_identity.json", "splits"), ("cache_index.json", "cache_index")):
        bind(addition / name, selection["identity"][key])
    bind(REPO / "navsim/planning/script/run_drivemem_safety_data.py", selection["identity"]["code"])
    bind(REPO / protocol["preparer_path"], protocol["preparer"])
    for location, files in protocol["inputs"].items():
        for name, digest in files.items():
            bind(Path(location) / name, digest)
    frozen = read(addition / "split_identity.json")
    bind(addition / "splits.json", frozen["splits"])
    bind(addition / "inventory.csv", frozen["inventory"])
    with np.load(addition / "vocab.npz", allow_pickle=False) as f:
        if not np.array_equal(vocab, f["vocab"]):
            raise ValueError("Vocabulary mismatch")
    # Agent/scorer/runtime must be identical; only root/count settings differ.
    a, b = read(source / "identity.json"), read(addition / "identity.json")
    if {k: v for k, v in a.items() if k != "settings"} != {k: v for k, v in b.items() if k != "settings"}:
        raise ValueError("Teacher/scorer identity mismatch")
    index = read(source / "export_index.json")
    samples, targets = [], {}
    for r in read(source / "splits.json")["records"]:
        if r["role"] not in ("memory", "fit"):
            continue
        s = load_sample(source, r, index)
        folder, expected = (completion / "memory_labels", completed_fit["memory_labels"]) if r["role"] == "memory" else (previous / "fit_labels", original_fit["sidecar_labels"])
        path = folder / (r["token"] + ".json")
        bind(path, expected[path.name])
        state = read(path)
        if state["token"] != s["token"]:
            raise ValueError("Training token mismatch")
        targets[s["token"]] = state["outcomes"]
        samples.append(attach_utility(s, state["outcomes"]))
    seen_sessions = {r["session_id"] for r in read(source / "splits.json")["records"] + read(validation / "splits.json")["records"]}
    index = read(addition / "export_index.json")
    records = {r["token"]: r for r in read(addition / "splits.json")["records"]}
    if set(report["labels"]) != {r["token"] + ".json" for r in selection["records"]}:
        raise ValueError("Additional label membership mismatch")
    for r in selection["records"]:
        actual = records[r["token"]]
        if any(actual[k] != r[k] for k in ("role", "session_id")) or r["session_id"] in seen_sessions:
            raise ValueError("Additional session overlaps old roles or selection mismatch")
        s = load_sample(addition, actual, index)
        path = addition / "top16_labels" / (r["token"] + ".json")
        bind(path, report["labels"][path.name])
        state = read(path)
        if any(state[k] != r[k] for k in ("token", "role", "session_id", "sampling")) or state["feature_sha256"] != index[r["token"]]:
            raise ValueError("New label provenance mismatch")
        targets[s["token"]] = state["outcomes"]
        samples.append(attach_utility(s, state["outcomes"]))
    bank, fitting = partition(samples)
    counts = dict(memory_scenes=len(bank), fit_scenes=len(fitting), memory_sessions=len({s["session_id"] for s in bank}), fit_sessions=len({s["session_id"] for s in fitting}))
    if counts != dict(memory_scenes=64, fit_scenes=48, memory_sessions=8, fit_sessions=6):
        raise ValueError(f"Unexpected merged training counts: {counts}")
    for root, names in ((prior, ("identity.json", "report.json")), (addition, ("identity.json", "label_selection.json", "support_report.json"))):
        for name in names:
            bind(root / name)
    return bank, fitting, targets, vocab, source, validation, completion, bindings, counts, int(original_identity["seed"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "fit", "evaluate", "report", "all"))
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--addition", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    prior, addition, out = args.prior.resolve(), args.addition.resolve(), args.out.resolve()
    data = load_inputs(prior, addition)
    bank, fitting, targets, vocab, source, validation, completion, bindings, counts, seed = data
    for root in (prior, addition, source, validation, completion):
        if out == root or out in root.parents or root in out.parents:
            raise ValueError("Separate output required")
    identity = dict(schema=1, inputs=bindings, code=sha(Path(__file__)), seed=seed,
        helper_sources={name: sha(REPO / name) for name in (
            "scripts/drivesuprim/evaluation/audit_drivemem_action_support.py",
            "scripts/drivesuprim/evaluation/diagnose_drivemem_memory.py")},
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, counts=counts,
        protocol="Same frozen pair-utility code, lambda grid, features and F-CV objective. Added M/F supervision only. No risk gate.")
    marker = out / "identity.json"
    if marker.exists() and read(marker) != identity:
        raise RuntimeError("Refit inputs/code changed; use a new output root")
    if out.exists() and not marker.exists() and any(out.iterdir()):
        raise ValueError("Nonempty output without identity")
    if args.stage == "plan":
        print(json.dumps(dict(output=str(out), **counts, new_scorer_calls=0, gpu_required=False, download_GB=0), indent=2))
        return
    if args.stage == "report":
        print(json.dumps(read(out / "report.json"), indent=2))
        return
    write(marker, identity)
    if args.stage == "evaluate" and not (out / "fit.json").exists():
        raise ValueError("Run fit first")
    fitted = fit_models(bank, fitting, targets, vocab, seed, out, read, write, save, sha)
    if args.stage == "fit":
        print(json.dumps(fitted, indent=2))
        return
    models = {}
    for method in METHODS:
        with np.load(out / (method + ".npz"), allow_pickle=False) as f:
            models[method] = {k: f[k].copy() for k in f.files}
    prepared = []
    used = {s["session_id"] for s in bank + fitting}
    for root in (source, validation):
        cohort, sessions = [], set()
        index = read(root / "export_index.json")
        for r in read(root / "splits.json")["records"]:
            if r["role"] != "development":
                continue
            if r["session_id"] in used:
                raise ValueError("Query session overlap")
            s = load_sample(root, r, index)
            choices = dict(baseline=int(s["selected_local"]))
            for method in METHODS:
                x = features(s, vocab, bank if method == "direct_memory" else None, fitted["h"], seed)
                choices[method] = choose(s, predict(x, models[method]))
            cohort.append((s, choices))
            sessions.add(s["session_id"])
        prepared.append((root, cohort))
        used.update(sessions)
    result = evaluate(prepared, fitted, out, read, write, sha, read(completion / "report.json")["oracle_inputs"])
    old_report = read(prior / "report.json")
    result["training_counts"] = counts
    result["comparison_to_original"] = {name: {method: dict(
        gain_change_pp=result["cohorts"][name]["methods"][method]["gain_macro_pp"] - old_report["cohorts"][name]["methods"][method]["gain_macro_pp"],
        harm_change=result["cohorts"][name]["methods"][method]["harm"] - old_report["cohorts"][name]["methods"][method]["harm"])
        for method in METHODS} for name in result["cohorts"]}
    result["data_control_note"] = "Same algorithm refit on more M/F; h/lambda refitted by original procedure. New F includes uniform and challenge; CV keeps original pooled macro objective for this control. Not risk calibration or fresh validation."
    write(out / "report.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
