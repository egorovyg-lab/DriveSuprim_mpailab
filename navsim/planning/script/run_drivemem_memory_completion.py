"""Controlled M top-16 completion x absolute/contrast experiment. No query training."""
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.agents.drivemem import METRICS
from navsim.agents.drivemem_action_transfer import pool, transfer, choose, ridge_path, ridge_predict
from navsim.planning.script.run_drivemem_action_transfer import record, objective, summary


VARIANTS = {
    "original_M_contrast": (False, True), "original_M_absolute": (False, False),
    "completed_M_contrast": (True, True), "completed_M_absolute": (True, False),
}


def fill_memory(bank, targets, evaluate, out, read, write, verify):
    if any(s["role"] != "memory" for s in bank):
        raise ValueError("Only M actions may be newly labelled")
    for number, s in enumerate(bank):
        token = s["token"]
        values = targets[token]
        baseline = int(s["selected_local"])
        repeated, _ = evaluate(token, int(s["candidate_ids"][baseline]))
        verify(values[str(baseline)], repeated)
        path = out / "memory_labels" / (token + ".json")
        for i in pool(s):
            if str(i) not in values:
                values[str(i)], _ = evaluate(token, int(s["candidate_ids"][i]))
                write(path, dict(token=token, outcomes=values))
        write(path, dict(token=token, outcomes=values))
        print(f"MEMORY TOP16 {number + 1}/{len(bank)} {token}", flush=True)


def labelled(sample, outcomes):
    indices = np.array(sorted(map(int, outcomes)), dtype=int)
    if any(str(i) not in outcomes for i in pool(sample)):
        raise RuntimeError("Top-16 labels incomplete")
    values = np.array([[outcomes[str(i)]["raw"][k] for k in METRICS] for i in indices])
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise RuntimeError("Invalid component labels")
    return dict(sample, label_local_indices=indices, label_values=values)


def fit_models(old_bank, new_bank, fit_samples, fit_targets, vocab, h, seed, out, read, write, save, sha):
    if (any(s["role"] != "memory" for s in old_bank + new_bank) or any(s["role"] != "fit" for s in fit_samples)
            or {s["token"] for s in old_bank} != {s["token"] for s in new_bank}
            or {s["session_id"] for s in new_bank} & {s["session_id"] for s in fit_samples}):
        raise ValueError("Invalid M/F roles, session overlap or changed bank membership")
    label_hashes = {p.name: sha(p) for p in sorted((out / "memory_labels").glob("*.json"))}
    if (out / "fit.json").exists():
        fitted = read(out / "fit.json")
        if fitted["memory_labels"] != label_hashes or fitted["ridge_sha256"] != sha(out / "ridge.npz"):
            raise RuntimeError("Frozen completed M/model changed")
        return fitted
    trials = {}
    for name, (completed, contrast) in VARIANTS.items():
        bank = new_bank if completed else old_bank
        scores = []
        for s in fit_samples:
            correction, _ = transfer(s, bank, vocab, h, contrast, seed)
            scores.append(record(s, fit_targets[s["token"]], choose(s, correction)))
        key = objective(scores)
        trials[name] = dict(gain_macro_pp=-key[0], harm=key[1])
    primary = min(VARIANTS, key=lambda n: (-trials[n]["gain_macro_pp"], trials[n]["harm"], n))
    lambdas = [1., 10., 100., 1000.]
    cv = {lam: [] for lam in lambdas}
    sessions = sorted({s["session_id"] for s in fit_samples})
    if len(sessions) < 2:
        raise ValueError("Grouped ridge CV requires at least two F sessions")
    for sid in sessions:
        train = new_bank + [s for s in fit_samples if s["session_id"] != sid]
        print(f"COMPLETED-M RIDGE CV held-out={sid}", flush=True)
        models = ridge_path(train, vocab, lambdas)
        for lam, model in models.items():
            for s in fit_samples:
                if s["session_id"] == sid:
                    cv[lam].append(record(s, fit_targets[s["token"]], choose(s, ridge_predict(s, vocab, model))))
    best = min(lambdas, key=lambda lam: (objective(cv[lam]), -lam))
    save(out / "ridge.npz", **ridge_path(new_bank + fit_samples, vocab, [best])[best])
    fitted = dict(status="complete", primary=primary, h=h, beta=.5, neighbors=5, topk=16,
        selection="Four matched-h variants selected by F macro gain, ties by harm/name. No query outcomes used.",
        fit_trials=trials, ridge_lambda=best,
        ridge_cv=[dict(regularization=lam, gain_macro_pp=-objective(cv[lam])[0], harm=objective(cv[lam])[1]) for lam in lambdas],
        old_memory_labels=sum(len(s["label_local_indices"]) for s in old_bank),
        new_memory_labels=sum(len(s["label_local_indices"]) for s in new_bank),
        memory_labels=label_hashes, ridge_sha256=sha(out / "ridge.npz"))
    write(out / "fit.json", fitted)
    return fitted


def main(cfg):
    from navsim.planning.script.run_drivemem_memory import (
        verify_identity, read_json as read, write_json as write, sha, sample as load_sample,
        training_samples, evaluators, save_npz,
    )
    from navsim.planning.script.run_drivemem_development_oracle import verify_baseline
    source = Path(cfg.drivemem.root).resolve()
    verify_identity(cfg, source)
    stage = str(cfg.drivemem.stage).removeprefix("completion-")
    if stage not in ("plan", "label", "fit", "evaluate", "report", "all"):
        raise ValueError("Unknown completion stage")
    base = Path(os.environ["NAVSIM_EXP_ROOT"]) / "drivemem_memory"
    previous = Path(os.environ.get("DRIVEMEM_FROZEN_MODEL_ROOT", str(base / "experiments/action_transfer_v2"))).resolve()
    validation = Path(os.environ.get("DRIVEMEM_VALIDATION_ROOT", str(base / "validation_cached4_v1"))).resolve()
    if validation == source:
        raise ValueError("Expected two distinct inspected query cohorts")
    out = Path(os.environ.get("DRIVEMEM_COMPLETION_ROOT", str(base / "experiments/memory_top16_v1"))).resolve()
    for original in (source, previous, validation):
        if out == original or original in out.parents or out in original.parents:
            raise ValueError("Completion experiment must have a separate root")
    old_identity, old_fit = read(previous / "identity.json"), read(previous / "fit.json")
    if old_identity["source"] != str(source) or old_fit["primary"] != "B3" or old_fit["beta"] != .5 or old_fit["neighbors"] != 5:
        raise RuntimeError("Unexpected frozen source protocol")
    repo = Path(__file__).resolve().parents[3]
    for name, digest in old_identity["sources"].items():
        if sha(repo / name) != digest:
            raise RuntimeError("Frozen v2 code changed")
    for name, digest in old_identity["inputs"].items():
        if sha(source / name) != digest:
            raise RuntimeError("Frozen source input changed")
    for name, digest in old_identity["labels"].items():
        if sha(source / "labels" / name) != digest:
            raise RuntimeError("Frozen source labels changed")
    for name, digest in old_fit["sidecar_labels"].items():
        if sha(previous / "fit_labels" / name) != digest:
            raise RuntimeError("Frozen F sidecars changed")
    if sha(previous / "ridge.npz") != old_fit["ridge_sha256"]:
        raise RuntimeError("Frozen previous ridge changed")
    h, seed = old_fit["configs"]["B3"]["h"], int(old_identity["seed"])
    manifest = dict(schema=1, source=str(source), previous=str(previous), validation=str(validation),
        runner_sha256=sha(Path(__file__)), previous_identity=sha(previous / "identity.json"),
        previous_fit=sha(previous / "fit.json"), h=h, beta=.5, neighbors=5, topk=16,
        validation_inputs={name: sha(validation / name) for name in ("identity.json", "split_identity.json")},
        numpy=np.__version__, pandas=pd.__version__, python=platform.python_version(),
        protocol="M top16+old random labels; matched-h absolute/contrast; F-only selection; both query sets already inspected")
    marker = out / "identity.json"
    if marker.exists() and read(marker) != manifest:
        raise RuntimeError("Completion inputs/code changed; choose new DRIVEMEM_COMPLETION_ROOT")
    if out.exists() and not marker.exists() and any(out.iterdir()):
        raise RuntimeError("Nonempty output without identity")
    if stage == "report":
        print(json.dumps(read(out / "report.json"), indent=2), flush=True)
        return
    samples = training_samples(cfg, source)
    old_bank = [s for s in samples if s["role"] == "memory"]
    fit_samples = [s for s in samples if s["role"] == "fit"]
    targets, missing = {}, 0
    for s in old_bank:
        labels = read(source / "labels" / (s["token"] + ".json"))
        original = {str(i): v for i, v in zip(labels["local_indices"], labels["values"])}
        path = out / "memory_labels" / (s["token"] + ".json")
        state = read(path) if path.exists() else dict(token=s["token"], outcomes=original)
        if state["token"] != s["token"] or any(state["outcomes"].get(k) != v for k, v in original.items()):
            raise RuntimeError("M sidecar disagrees with original labels")
        targets[s["token"]] = state["outcomes"]
        missing += sum(str(i) not in state["outcomes"] for i in pool(s))
    print(json.dumps(dict(stage=stage, output=str(out), new_memory_actions=missing,
        M_baseline_repeats_if_label_stage_run=len(old_bank),
        label_stage_scorer_calls_if_run=2 * (missing + len(old_bank)), gpu_required=False, new_image_GB=0), indent=2), flush=True)
    if stage == "plan":
        return
    write(marker, manifest)
    with np.load(source / "vocab.npz", allow_pickle=False) as data:
        vocab = data["vocab"].copy()
    if stage in ("all", "label"):
        if (out / "fit.json").exists() and missing:
            raise RuntimeError("Cannot append M labels after frozen fit")
        evaluate, _ = evaluators(cfg, source)
        fill_memory(old_bank, targets, evaluate, out, read, write, verify_baseline)
        missing = 0
    if stage in ("all", "fit", "evaluate"):
        if missing or any(not (out / "memory_labels" / (s["token"] + ".json")).exists() for s in old_bank):
            raise RuntimeError("Run completion-label first")
        new_bank = [labelled(s, targets[s["token"]]) for s in old_bank]
        fit_targets = {s["token"]: read(previous / "fit_labels" / (s["token"] + ".json"))["outcomes"] for s in fit_samples}
        fit_samples = [labelled(s, fit_targets[s["token"]]) for s in fit_samples]
        fitted = fit_models(old_bank, new_bank, fit_samples, fit_targets, vocab, h, seed, out, read, write, save_npz, sha)
        print(json.dumps(fitted, indent=2), flush=True)
    if stage in ("all", "evaluate"):
        with np.load(out / "ridge.npz", allow_pickle=False) as data:
            ridge_new = {k: data[k].copy() for k in data.files}
        with np.load(previous / "ridge.npz", allow_pickle=False) as data:
            ridge_old = {k: data[k].copy() for k in data.files}
        prepared = []
        original_choices = {item["token"]: item["choices"] for item in read(previous / "development_decisions.json")["decisions"]}
        validation_choices = read(validation / "frozen_evaluation/decisions.json")
        for root in (source, validation):
            frozen = read(root / "split_identity.json")
            if frozen != dict(splits=sha(root / "splits.json"), inventory=sha(root / "inventory.csv")):
                raise RuntimeError("Query selection changed")
            with np.load(root / "vocab.npz", allow_pickle=False) as data:
                if not np.array_equal(vocab, data["vocab"]):
                    raise RuntimeError("Query vocabulary mismatch")
            cohort = []
            for r in read(root / "splits.json")["records"]:
                if r["role"] != "development":
                    continue
                s = load_sample(root, r)
                decisions = dict(baseline=int(s["selected_local"]), ridge_original=choose(s, ridge_predict(s, vocab, ridge_old)),
                                 ridge_completed_M=choose(s, ridge_predict(s, vocab, ridge_new)))
                for name, (completed, contrast) in VARIANTS.items():
                    c, _ = transfer(s, new_bank if completed else old_bank, vocab, h, contrast, seed)
                    decisions[name] = choose(s, c)
                expected = original_choices if root == source else validation_choices
                if decisions["original_M_contrast"] != expected[s["token"]]["B3"]:
                    raise RuntimeError("Original B3 not reproduced")
                completed, contrast = VARIANTS[fitted["primary"]]
                for i in range(3):
                    c, _ = transfer(s, new_bank if completed else old_bank, vocab, h, contrast, seed + i, random=True)
                    decisions[f"random_{i}"] = choose(s, c)
                cohort.append((s, decisions))
            prepared.append((root, cohort))
        frozen_decisions = dict(fit_sha256=sha(out / "fit.json"),
            cohorts={root.name: {s["token"]: d for s, d in cohort} for root, cohort in prepared})
        if (out / "query_decisions.json").exists() and read(out / "query_decisions.json") != frozen_decisions:
            raise RuntimeError("Previously frozen query decisions changed")
        write(out / "query_decisions.json", frozen_decisions)
        # No query outcomes are opened until both cohorts' decisions and F-fit are fixed.
        reports, oracle_hashes = {}, {}
        previous_report = read(out / "report.json") if (out / "report.json").exists() else None
        for root, cohort in prepared:
            results = []
            oracle_hashes[root.name] = {}
            for s, decisions in cohort:
                path = root / "development_oracle/scenes" / (s["token"] + ".json")
                digest = sha(path)
                if previous_report and previous_report["oracle_inputs"][root.name][s["token"]] != digest:
                    raise RuntimeError("Previously used query oracle changed")
                oracle_hashes[root.name][s["token"]] = digest
                outcomes = read(path)["outcomes"]
                if any(str(i) not in outcomes for i in pool(s)):
                    raise RuntimeError("Query oracle top16 incomplete")
                oracle = max(outcomes[str(i)]["raw"]["pdm_score"] for i in pool(s))
                for method, index in decisions.items():
                    results.append(dict(method=method, oracle_score=oracle, **record(s, outcomes, index)))
            frame = pd.DataFrame(results)
            frame.to_csv(out / (root.name + "_per_scene.csv"), index=False)
            frame.groupby(["session_id", "method"]).agg(gain_pp=("raw_gain_pp", "mean"), harm=("raw_harm", "sum")).to_csv(out / (root.name + "_by_session.csv"))
            reports[root.name] = dict(scenes=len(cohort), sessions=frame.session_id.nunique(), methods=summary(results, seed))
            pivot = frame.pivot(index=["session_id", "token"], columns="method", values="raw_gain_pp")
            delta = (pivot[fitted["primary"]] - pivot.ridge_completed_M).groupby("session_id").mean().to_numpy()
            pick = np.random.default_rng(seed).integers(0, len(delta), size=(2000, len(delta)))
            reports[root.name]["paired_primary_minus_completed_ridge"] = dict(gain_macro_pp=float(delta.mean()),
                ci95_sessions=np.quantile(delta[pick].mean(axis=1), [.025, .975]).tolist())
        report = dict(status="complete", primary=fitted["primary"], h=h, beta=.5, ridge_lambda=fitted["ridge_lambda"],
            cohorts=reports, oracle_inputs=oracle_hashes,
            note="Exploratory analysis of BOTH already inspected query cohorts. No fresh holdout. "
                 "No query outcomes used for fitting/parameter selection. Local PDM, not EPDMS; empirical harm only.")
        write(out / "report.json", report)
    if stage in ("all", "report"):
        print(json.dumps(read(out / "report.json"), indent=2), flush=True)


if __name__ == "__main__":
    import hydra
    hydra.main(config_path="config/pdm_scoring", config_name="default_drivemem_memory", version_base=None)(main)()
