"""Matched direct-utility models. Read-only inputs, no new simulator calls."""
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.agents.drivemem_action_transfer import pool
from navsim.agents.drivemem_pair_utility import attach_utility, bandwidth, features, ridge_path, predict, choose
from navsim.planning.script.run_drivemem_action_transfer import record, objective, summary


METHODS = ("direct_no_memory", "direct_memory")
LAMBDAS = (1., 10., 100., 1000.)


def fit_models(bank, fit, targets, vocab, seed, out, read, write, save, sha):
    if (not bank or not fit or any(s["role"] != "memory" for s in bank)
            or any(s["role"] != "fit" for s in fit)
            or {s["session_id"] for s in bank} & {s["session_id"] for s in fit}):
        raise ValueError("Disjoint nonempty M/F sessions required")
    if (out / "fit.json").exists():
        frozen = read(out / "fit.json")
        for name, digest in frozen["models"].items():
            if sha(out / (name + ".npz")) != digest:
                raise RuntimeError("Frozen utility model changed")
        return frozen
    h = bandwidth(bank, vocab, seed)
    samples = bank + fit
    sessions = sorted({s["session_id"] for s in fit})
    if len(sessions) < 2:
        raise ValueError("Grouped CV needs at least two F sessions")
    trials, model_hashes, selected = {}, {}, {}
    for method in METHODS:
        print(f"FEATURES {method}", flush=True)
        xs = [features(s, vocab, bank if method == "direct_memory" else None, h, seed) for s in samples]
        ys = [s["utility_values"] for s in samples]
        cv = {lam: [] for lam in LAMBDAS}
        for sid in sessions:
            train = [i for i, s in enumerate(samples) if s["session_id"] != sid]
            valid = [i for i, s in enumerate(samples) if s["role"] == "fit" and s["session_id"] == sid]
            print(f"UTILITY CV {method} held-out={sid}", flush=True)
            models = ridge_path([xs[i] for i in train], [ys[i] for i in train], LAMBDAS)
            for lam, model in models.items():
                for i in valid:
                    s = samples[i]
                    cv[lam].append(record(s, targets[s["token"]], choose(s, predict(xs[i], model))))
        best = min(LAMBDAS, key=lambda lam: (objective(cv[lam]), -lam))
        selected[method] = dict(regularization=best, gain_macro_pp=-objective(cv[best])[0], harm=objective(cv[best])[1])
        trials[method] = [dict(regularization=lam, gain_macro_pp=-objective(cv[lam])[0], harm=objective(cv[lam])[1]) for lam in LAMBDAS]
        save(out / (method + ".npz"), **ridge_path(xs, ys, [best])[best])
        model_hashes[method] = sha(out / (method + ".npz"))
    # Baseline wins exact ties. A negative F-CV result cannot become primary.
    candidates = dict(baseline=dict(gain_macro_pp=0., harm=0), **selected)
    primary = min(candidates, key=lambda m: (-candidates[m]["gain_macro_pp"], candidates[m]["harm"], list(candidates).index(m)))
    fitted = dict(status="complete", primary=primary, h=h, topk=16, neighbors=5, seed=seed,
        target="raw PDM(candidate) minus raw PDM(baseline)", selected=selected, cv=trials, models=model_hashes,
        selection="Leave-one-F-session-out macro gain; ties by harm then stronger ridge. Primary from pooled F-CV; baseline wins ties.",
        protocol="Fixed M bank; M query excludes own entire session; no query labels in features; RMS scale fitted within fold; zero intercept.")
    write(out / "fit.json", fitted)
    return fitted


def evaluate(prepared, fitted, out, read, write, sha, expected_oracles):
    decisions = dict(fit_sha256=sha(out / "fit.json"),
        cohorts={root.name: {s["token"]: choices for s, choices in cohort} for root, cohort in prepared})
    marker = out / "query_decisions.json"
    if marker.exists() and read(marker) != decisions:
        raise RuntimeError("Frozen query decisions changed")
    write(marker, decisions)
    reports = {}
    # Both cohorts' proposals have been frozen before opening query outcomes.
    for root, cohort in prepared:
        rows = []
        for s, choices in cohort:
            path = root / "development_oracle/scenes" / (s["token"] + ".json")
            if sha(path) != expected_oracles[root.name][s["token"]]:
                raise RuntimeError("Query oracle changed since completed-M experiment")
            state = read(path)
            if state["token"] != s["token"]:
                raise RuntimeError("Oracle token mismatch")
            outcomes = state["outcomes"]
            oracle = max(outcomes[str(i)]["raw"]["pdm_score"] for i in pool(s))
            for method, index in choices.items():
                rows.append(dict(method=method, oracle_score=oracle, **record(s, outcomes, index)))
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError("Empty query cohort")
        frame.to_csv(out / (root.name + "_per_scene.csv"), index=False)
        frame.groupby(["session_id", "method"]).agg(gain_pp=("raw_gain_pp", "mean"), harm=("raw_harm", "sum")).to_csv(out / (root.name + "_by_session.csv"))
        pivot = frame.pivot(index=["session_id", "token"], columns="method", values="raw_gain_pp")
        delta = (pivot.direct_memory - pivot.direct_no_memory).groupby("session_id").mean().to_numpy()
        draws = np.random.default_rng(fitted["seed"]).integers(0, len(delta), size=(2000, len(delta)))
        reports[root.name] = dict(scenes=len(cohort), sessions=len(delta), methods=summary(rows, fitted["seed"]),
            paired_memory_minus_no_memory=dict(gain_macro_pp=float(delta.mean()),
                ci95_sessions=np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()))
    report = dict(status="complete", primary=fitted["primary"], cohorts=reports,
        note="Both query cohorts already inspected: exploratory development only. Matched M/F raw-utility supervision. "
             "Fixed simulator-enriched M, not continual evaluation. Local PDM, not full EPDMS. Empirical component harm, no risk guarantee.")
    write(out / "report.json", report)
    return report


def main(cfg):
    from navsim.planning.script.run_drivemem_memory import (
        verify_identity, read_json as read, write_json as write, sha, sample as load_sample,
        training_samples, save_npz,
    )
    source = Path(cfg.drivemem.root).resolve()
    verify_identity(cfg, source)
    if str(cfg.drivemem.memory_mode) != "enriched":
        raise ValueError("Utility experiment requires the completed enriched M bank")
    stage = str(cfg.drivemem.stage).removeprefix("utility-")
    if stage not in ("plan", "fit", "evaluate", "report", "all"):
        raise ValueError("Unknown utility stage")
    base = Path(os.environ["NAVSIM_EXP_ROOT"]) / "drivemem_memory"
    completion = Path(os.environ.get("DRIVEMEM_COMPLETION_ROOT", str(base / "experiments/memory_top16_v1"))).resolve()
    old = read(completion / "identity.json")
    completed_fit = read(completion / "fit.json")
    previous, validation = Path(old["previous"]), Path(old["validation"])
    out = Path(os.environ.get("DRIVEMEM_UTILITY_ROOT", str(base / "experiments/pair_utility_v1"))).resolve()
    for root in (source, completion, previous, validation):
        if out == root or root in out.parents or out in root.parents:
            raise ValueError("Utility output must be a separate root")
    if str(source) != old["source"] or validation == source or validation.name == source.name:
        raise ValueError("Unexpected source/cohort membership")
    repo = Path(__file__).resolve().parents[3]
    if sha(repo / "navsim/planning/script/run_drivemem_memory_completion.py") != old["runner_sha256"]:
        raise RuntimeError("Frozen completion runner changed")
    for name, expected in (("identity.json", old["previous_identity"]), ("fit.json", old["previous_fit"])):
        if sha(previous / name) != expected:
            raise RuntimeError("Previous experiment changed")
    original, fit = read(previous / "identity.json"), read(previous / "fit.json")
    for table, root in ((original["sources"], repo), (original["inputs"], source),
                        (original["labels"], source / "labels"), (fit["sidecar_labels"], previous / "fit_labels"),
                        (completed_fit["memory_labels"], completion / "memory_labels"),
                        (old["validation_inputs"], validation)):
        for name, expected in table.items():
            if sha(root / name) != expected:
                raise RuntimeError(f"Frozen input changed: {root / name}")
    samples = training_samples(cfg, source)
    bank = [s for s in samples if s["role"] == "memory"]
    fitting = [s for s in samples if s["role"] == "fit"]
    if set(completed_fit["memory_labels"]) != {s["token"] + ".json" for s in bank}:
        raise RuntimeError("Completed memory membership mismatch")
    targets = {}
    for s in bank + fitting:
        root = completion / "memory_labels" if s["role"] == "memory" else previous / "fit_labels"
        state = read(root / (s["token"] + ".json"))
        if state["token"] != s["token"]:
            raise RuntimeError("Training label token mismatch")
        targets[s["token"]] = state["outcomes"]
    bank = [attach_utility(s, targets[s["token"]]) for s in bank]
    fitting = [attach_utility(s, targets[s["token"]]) for s in fitting]
    manifest = dict(schema=1, source=str(source), completion=str(completion), validation=str(validation),
        completion_identity=sha(completion / "identity.json"), completion_fit=sha(completion / "fit.json"),
        completed_report=sha(completion / "report.json"),
        sources={name: sha(repo / name) for name in (
            "navsim/agents/drivemem_pair_utility.py", "navsim/planning/script/run_drivemem_pair_utility.py")},
        query_inputs={root.name: {name: sha(root / name) for name in (
            "identity.json", "split_identity.json", "splits.json", "inventory.csv", "export_index.json", "vocab.npz")}
            for root in (source, validation)},
        numpy=np.__version__, pandas=pd.__version__, python=platform.python_version(), seed=int(original["seed"]))
    marker = out / "identity.json"
    if marker.exists() and read(marker) != manifest:
        raise RuntimeError("Utility protocol/inputs changed; select a new DRIVEMEM_UTILITY_ROOT")
    if out.exists() and not marker.exists() and any(out.iterdir()):
        raise RuntimeError("Nonempty output without identity")
    if stage == "plan":
        print(json.dumps(dict(output=str(out), memory_scenes=len(bank), fit_scenes=len(fitting),
            new_scorer_calls=0, gpu_required=False, new_download_GB=0, query_cohorts="Both already inspected"), indent=2))
        return
    write(marker, manifest)
    if stage == "report":
        print(json.dumps(read(out / "report.json"), indent=2))
        return
    with np.load(source / "vocab.npz", allow_pickle=False) as data:
        vocab = data["vocab"].copy()
    if stage == "evaluate" and not (out / "fit.json").exists():
        raise RuntimeError("Run utility-fit first")
    fitted = fit_models(bank, fitting, targets, vocab, int(original["seed"]), out, read, write, save_npz, sha)
    print(json.dumps(fitted, indent=2), flush=True)
    if stage == "fit":
        return
    models = {}
    for method in METHODS:
        with np.load(out / (method + ".npz"), allow_pickle=False) as data:
            models[method] = {k: data[k].copy() for k in data.files}
    prepared, used_sessions = [], {s["session_id"] for s in bank + fitting}
    for root in (source, validation):
        if read(root / "split_identity.json") != dict(splits=sha(root / "splits.json"), inventory=sha(root / "inventory.csv")):
            raise RuntimeError("Query split changed")
        with np.load(root / "vocab.npz", allow_pickle=False) as data:
            if not np.array_equal(vocab, data["vocab"]):
                raise RuntimeError("Query vocabulary mismatch")
        cohort = []
        query_sessions = set()
        for r in read(root / "splits.json")["records"]:
            if r["role"] != "development":
                continue
            s = load_sample(root, r)
            if s["session_id"] in used_sessions:
                raise RuntimeError("Query session overlaps M/F or another cohort")
            query_sessions.add(s["session_id"])
            choices = dict(baseline=int(s["selected_local"]))
            for method in METHODS:
                x = features(s, vocab, bank if method == "direct_memory" else None, fitted["h"], fitted["seed"])
                choices[method] = choose(s, predict(x, models[method]))
            cohort.append((s, choices))
        used_sessions.update(query_sessions)
        prepared.append((root, cohort))
    report = evaluate(prepared, fitted, out, read, write, sha, read(completion / "report.json")["oracle_inputs"])
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    import hydra
    hydra.main(config_path="config/pdm_scoring", config_name="default_drivemem_memory", version_base=None)(main)()
