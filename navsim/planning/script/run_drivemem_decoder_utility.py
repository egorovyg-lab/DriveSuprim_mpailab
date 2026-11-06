"""One bounded component family and matched controls, reusing immutable M/F."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.agents import drivemem_decoder_utility as model
from navsim.agents.drivemem_decoder_utility import load_decoder_inputs, load_decoder_sample
from navsim.agents.drivemem_action_transfer import pool
from navsim.agents.drivemem_pair_utility import bandwidth
from navsim.planning.script.run_drivemem_action_transfer import record, summary
from navsim.planning.script.run_drivemem_expanded_utility import load_inputs, write, save, read, sha, load_sample, REPO

FAMILIES = ("no_memory", "memory", "random", "shuffled")
POLICIES = ("raw", "component", "guarded")
LAMBDAS = (1., 10., 100., 1000.)


def build_features(s, family, bank, vocab, h, seed):
    return model.features(s, vocab, None if family == "no_memory" else bank, h, seed,
                          random=family == "random", shuffle=family == "shuffled")


def cv_score(rows):
    frame = pd.DataFrame(rows)
    uniform = frame[frame["slice"] == "uniform"]
    if uniform.empty:
        raise ValueError("Uniform F required")
    return dict(uniform_gain_macro_pp=float(uniform.groupby("session_id").raw_gain_pp.mean().mean()),
                harm=int(frame.raw_harm.sum()), interventions=int(frame.intervention.sum()))


def fit_models(bank, fitting, targets, vocab, h, seed, out):
    if (out / "fit.json").exists():
        frozen = read(out / "fit.json")
        for family, digest in frozen["models"].items():
            if sha(out / (family + ".npz")) != digest:
                raise RuntimeError("Frozen component model changed")
        return frozen
    samples = bank + fitting
    if any(s["role"] != "memory" for s in bank) or any(s["role"] != "fit" for s in fitting):
        raise ValueError("M/F only")
    if {s["session_id"] for s in bank} & {s["session_id"] for s in fitting}:
        raise ValueError("M/F overlap")
    selected, trials, hashes, selected_rows = {}, {}, {}, []
    for family in FAMILIES:
        print(f"COMPONENT FEATURES {family}", flush=True)
        feature_data = [build_features(s, family, bank, vocab, h, seed) for s in samples]
        xs = [x for x, _ in feature_data]
        ys = [s["outcome_values"] for s in samples]
        cv = {lam: {policy: [] for policy in POLICIES} for lam in LAMBDAS}
        for sid in sorted({s["session_id"] for s in fitting}):
            train = [i for i, s in enumerate(samples) if s["session_id"] != sid]
            valid = [i for i, s in enumerate(samples) if s["role"] == "fit" and s["session_id"] == sid]
            print(f"COMPONENT CV {family} held-out={sid}", flush=True)
            path = model.ridge_path([xs[i] for i in train], [ys[i] for i in train], LAMBDAS)
            for lam, fitted in path.items():
                for i in valid:
                    s = samples[i]
                    prediction = model.predict(xs[i], fitted)
                    for policy in POLICIES:
                        chosen = model.choose(s, prediction, policy, feature_data[i][1])
                        cv[lam][policy].append(dict(slice=s["sampling_slice"], **record(s, targets[s["token"]], chosen)))
        scores = {lam: cv_score(cv[lam]["guarded"]) for lam in LAMBDAS}
        def key(lam):
            r = scores[lam]
            return (r["harm"] > 0, -r["uniform_gain_macro_pp"], r["harm"], -lam)
        best = min(LAMBDAS, key=key)
        selected[family] = dict(regularization=best, **scores[best])
        trials[family] = {str(lam): {p: cv_score(cv[lam][p]) for p in POLICIES} for lam in LAMBDAS}
        for policy in POLICIES:
            selected_rows.extend(dict(method=family + "_" + policy, **r) for r in cv[best][policy])
        save(out / (family + ".npz"), **model.ridge_path(xs, ys, [best])[best])
        hashes[family] = sha(out / (family + ".npz"))
    eligible = [f for f in ("no_memory", "memory") if selected[f]["harm"] == 0 and selected[f]["uniform_gain_macro_pp"] > 0]
    primary = max(eligible, key=lambda f: (selected[f]["uniform_gain_macro_pp"], f == "no_memory")) + "_guarded" if eligible else "baseline"
    fitted = dict(primary=primary, selected=selected, cv=trials, models=hashes, h=h, seed=seed,
        epsilon=model.EPS, numeric_tolerance=model.TOL, outcome_keys=list(model.KEYS),
        selection="Same lambda grid per family. Guarded F-CV: prefer zero harm, then uniform macro gain, harm, stronger ridge. Primary only true-memory/no-memory guarded with zero observed F harm and positive uniform gain, otherwise baseline.",
        note="CV also selects lambda: not an unbiased performance estimate. Fixed component/teacher checks are engineering constraints, not calibrated risk bounds.")
    pd.DataFrame(selected_rows).to_csv(out / "fit_cv_per_scene.csv", index=False)
    write(out / "fit.json", fitted)
    return fitted


def evaluate(prepared, fitted, out, expected):
    decisions = dict(fit_sha256=sha(out / "fit.json"), cohorts={root.name: {s["token"]: d for s, d, _ in cohort} for root, cohort in prepared})
    marker = out / "query_decisions.json"
    if marker.exists() and read(marker) != decisions:
        raise RuntimeError("Frozen decisions changed")
    write(marker, decisions)
    cohorts = {}
    # Both cohorts fixed before any oracle outcome lookup.
    for root, cohort in prepared:
        rows = []
        for s, choices, diagnostics in cohort:
            path = root / "development_oracle/scenes" / (s["token"] + ".json")
            if sha(path) != expected[root.name][s["token"]]:
                raise RuntimeError("Query oracle changed")
            state = read(path)
            if state["token"] != s["token"]:
                raise ValueError("Query token mismatch")
            outcomes = state["outcomes"]
            oracle = max(outcomes[str(i)]["raw"]["pdm_score"] for i in pool(s))
            for name, chosen in choices.items():
                rows.append(dict(method=name, oracle_score=oracle, **diagnostics.get(name, {}), **record(s, outcomes, chosen)))
        frame = pd.DataFrame(rows)
        frame.to_csv(out / (root.name + "_per_scene.csv"), index=False)
        frame.groupby(["session_id", "method"]).agg(gain_pp=("raw_gain_pp", "mean"), harm=("raw_harm", "sum")).to_csv(out / (root.name + "_by_session.csv"))
        pivot = frame.pivot(index=["session_id", "token"], columns="method", values="raw_gain_pp")
        delta = (pivot.memory_guarded - pivot.no_memory_guarded).groupby("session_id").mean().to_numpy()
        draws = np.random.default_rng(fitted["seed"]).integers(0, len(delta), size=(2000, len(delta)))
        cohorts[root.name] = dict(methods=summary(rows, fitted["seed"]),
            paired_memory_minus_no_memory_guarded=dict(gain_macro_pp=float(delta.mean()), ci95_sessions=np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()))
    result = dict(status="complete", primary=fitted["primary"], cohorts=cohorts,
        note="Already inspected development cohorts, not fresh validation. Local PDM, not EPDMS. Bounded scores and pointwise component checks are not probabilities/confidence bounds or risk guarantees. Unsupported DDC/TLC remain limitations.")
    write(out / "report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "fit", "evaluate", "report", "all"))
    for name in ("prior", "addition", "out", "decoder"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    prior, addition, out = args.prior.resolve(), args.addition.resolve(), args.out.resolve()
    bank, fitting, targets, vocab, source, validation, completion, bindings, counts, seed = load_decoder_inputs(prior, addition, args.decoder.resolve())
    for root in (prior, addition, source, validation, completion, args.decoder.resolve()):
        if root == out or root in out.parents or out in root.parents:
            raise ValueError("Separate output required")
    tags = {r["token"]: r["sampling"] for r in read(addition / "label_selection.json")["records"]}
    bank = [model.attach(s, targets[s["token"]]) for s in bank]
    fitting = [dict(model.attach(s, targets[s["token"]]), sampling_slice="challenge" if tags.get(s["token"], "uniform").startswith("challenge") else "uniform") for s in fitting]
    identity = dict(inputs=bindings, counts=counts, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        sources={name: sha(REPO / name) for name in (
            "navsim/agents/drivemem_component_utility.py", "navsim/planning/script/run_drivemem_component_utility.py",
            "navsim/agents/drivemem_decoder_utility.py", "navsim/planning/script/run_drivemem_decoder_utility.py",
            "navsim/planning/script/run_drivemem_decoder_export.py",
            "navsim/planning/script/run_drivemem_decoder_replay_export.py", "navsim/agents/drivemem_frozen_pool.py",
            "navsim/planning/script/run_drivemem_decoder_deterministic_export.py", "navsim/agents/drivemem_determinism.py",
            "navsim/planning/script/run_drivemem_expanded_utility.py", "scripts/drivesuprim/evaluation/audit_drivemem_action_support.py",
            "scripts/drivesuprim/evaluation/diagnose_drivemem_memory.py")},
        protocol="Additive decoder features; frozen bounded absolute outcome learner and policies; fixed policy checks; uniform F selection, challenge F harm check; M-only recall; no query labels for fitting.")
    marker = out / "identity.json"
    if marker.exists() and read(marker) != identity:
        raise RuntimeError("Component protocol changed; use a new output root")
    if out.exists() and not marker.exists() and any(out.iterdir()):
        raise ValueError("Nonempty output without identity")
    if args.stage == "plan":
        print(json.dumps(dict(output=str(out), **counts, models=list(FAMILIES), policies=list(POLICIES),
                              gpu_required=False, new_scorer_calls=0, download_GB=0), indent=2))
        return
    if args.stage == "report":
        print(json.dumps(read(out / "report.json"), indent=2))
        return
    write(marker, identity)
    if args.stage == "evaluate" and not (out / "fit.json").exists():
        raise ValueError("Run fit first")
    h = bandwidth(bank, vocab, seed)
    fitted = fit_models(bank, fitting, targets, vocab, h, seed, out)
    if args.stage == "fit":
        print(json.dumps(fitted, indent=2))
        return
    models = {}
    for family in FAMILIES:
        with np.load(out / (family + ".npz"), allow_pickle=False) as f:
            models[family] = {k: f[k].copy() for k in f.files}
    prepared, used = [], {s["session_id"] for s in bank + fitting}
    for root in (source, validation):
        cohort, sessions = [], set()
        index = read(root / "export_index.json")
        for r in read(root / "splits.json")["records"]:
            if r["role"] != "development":
                continue
            if r["session_id"] in used:
                raise ValueError("Query overlaps training or another cohort")
            s = load_decoder_sample(root, r, index, args.decoder.resolve())
            choices, details = dict(baseline=int(s["selected_local"])), {}
            for family in FAMILIES:
                x, supported = build_features(s, family, bank, vocab, h, seed)
                prediction = model.predict(x, models[family])
                ix = pool(s).tolist()
                b = ix.index(int(s["selected_local"]))
                for policy in POLICIES:
                    name = family + "_" + policy
                    chosen = model.choose(s, prediction, policy, supported)
                    choices[name] = chosen
                    details[name] = dict(predicted_gain_pp=100 * float(prediction[ix.index(chosen), 0] - prediction[b, 0]), supported=supported)
            cohort.append((s, choices, details))
            sessions.add(s["session_id"])
        prepared.append((root, cohort))
        used.update(sessions)
    result = evaluate(prepared, fitted, out, read(completion / "report.json")["oracle_inputs"])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
