"""CPU-only selector experiment on immutable functional decoder sidecars.

The two already inspected development cohorts are diagnostic, not fresh tests.
Calibration/test roles are never opened. Both cohorts' decisions are persisted
before opening either cohort's outcome files. Original experiments are read-only.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch

from navsim.agents import trajectory_selector as model
from navsim.agents.drivemem_decoder_utility import load_decoder_inputs, load_decoder_sample
from navsim.planning.script.run_drivemem_action_transfer import record, summary
from navsim.planning.script.run_drivemem_expanded_utility import REPO, read, write, save, sha

FAMILIES = ("mlp", "attention")
THRESHOLDS = (.5, .75, .9, 1.01)  # Last entry explicitly disables interventions.
NOTE = ("Local PDM, not full EPDMS. Both development cohorts were previously inspected. "
        "Archive-budgeted scenes are not a representative benchmark. "
        "Thresholds and family selected on M/F out-of-fold results; this is tuning, not "
        "an unbiased performance estimate or calibrated safety. No donor retrieval.")


def csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(path)


def build_training(bank, fitting, outcomes, vocab, tags, config):
    samples = sorted(bank + fitting, key=lambda s: (s["session_id"], s["token"]))
    if len({s["token"] for s in samples}) != len(samples):
        raise ValueError("Duplicate training tokens")
    if len({s["session_id"] for s in samples}) < 3:
        raise ValueError("At least three independent training sessions required")
    entries = []
    for sample in samples:
        if sample["role"] not in ("memory", "fit"):
            raise ValueError("Only M/F may train")
        x, ix, baseline = model.inputs(sample, vocab)
        sample = dict(sample, baseline_position=baseline,
                      sampling_slice="challenge" if tags.get(sample["token"], "uniform").startswith("challenge") else "uniform")
        entries.append(dict(sample=sample, x=x, indices=ix, baseline=baseline,
                            targets=model.targets(sample, outcomes[sample["token"]], config)))
    if len({e["x"].shape for e in entries}) != 1:
        raise ValueError("Inconsistent feature shapes")
    return entries


def audit(entries, out, config):
    rows = []
    for e in entries:
        s, y, b = e["sample"], e["targets"], e["baseline"]
        rows.append(dict(token=s["token"], session_id=s["session_id"], role=s["role"], slice=s["sampling_slice"],
                         meaningful_ranking=bool(np.ptp(y["quality"]) > config.tie_epsilon),
                         improving_actions=int(y["good"].sum()), degrading_actions=int(y["harm"].sum()),
                         baseline_score=float(y["quality"][b]), oracle_gain_pp=100 * float(y["gain"].max())))
    report = dict(status="complete", scenes=len(rows), sessions=len({r["session_id"] for r in rows}),
                  feature_dim=entries[0]["x"].shape[1], topk=16,
                  meaningful_ranking_scenes=sum(r["meaningful_ranking"] for r in rows),
                  scenes_with_useful_alternative=sum(r["improving_actions"] > 0 for r in rows),
                  sessions_with_useful_alternative=len({r["session_id"] for r in rows if r["improving_actions"]}),
                  scenes_with_degrading_alternative=sum(r["degrading_actions"] > 0 for r in rows),
                  sessions_with_degrading_alternative=len({r["session_id"] for r in rows if r["degrading_actions"]}),
                  note="Candidate rows are correlated. Benefit logits are not calibrated collision probabilities.",
                  new_scorer_calls=0, download_GB=0, gpu_required=False)
    csv(out / "training_support.csv", rows)
    write(out / "audit.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def fit_one(entries, family, config, label):
    def progress(epoch, loss):
        print(f"TRAIN {label} epoch={epoch}/{config.epochs} loss={loss:.6f}", flush=True)
    return model.train([e["x"] for e in entries], [e["targets"] for e in entries],
                       [e["sample"] for e in entries], family, config, progress)


def predict_entries(net, entries, mean, scale):
    return model.predict(net, [e["x"] for e in entries], [e["baseline"] for e in entries], mean, scale)


def fold_predictions(entries, family, heldout, config, out):
    train = [e for e in entries if e["sample"]["session_id"] != heldout]
    valid = [e for e in entries if e["sample"]["session_id"] == heldout]
    fold = out / "cv" / (family + "_" + hashlib.sha256(heldout.encode()).hexdigest()[:16])
    expected = dict(train_sessions=sorted({e["sample"]["session_id"] for e in train}),
                    valid_session=heldout, valid_tokens=[e["sample"]["token"] for e in valid])
    if fold.with_suffix(".json").exists():
        frozen = read(fold.with_suffix(".json"))
        if frozen["partition"] != expected or sha(fold.with_suffix(".npz")) != frozen["sha256"]:
            raise RuntimeError("Cached CV fold changed")
        with np.load(fold.with_suffix(".npz"), allow_pickle=False) as arrays:
            prediction = {k: arrays[k].copy() for k in ("rank", "gain", "good_logit")}
        return valid, prediction
    print(f"CV family={family} held-out={heldout} train={len(train)} valid={len(valid)}", flush=True)
    net, mean, scale, history = fit_one(train, family, config, family + "/" + heldout)
    prediction = predict_entries(net, valid, mean, scale)
    fold.parent.mkdir(parents=True, exist_ok=True)
    save(fold.with_suffix(".npz"), **prediction, mean=mean, scale=scale)
    write(fold.with_suffix(".json"), dict(partition=expected, sha256=sha(fold.with_suffix(".npz")), history=history))
    return valid, prediction


def chosen_row(e, prediction, j, threshold, outcomes, config, method):
    sample, ix, b = e["sample"], e["indices"], e["baseline"]
    position = model.choose(*(prediction[k][j] for k in ("rank", "gain", "good_logit")), b,
                            sample["candidate_ids"][ix], threshold, config.improvement_min)
    return dict(method=method, slice=sample["sampling_slice"],
                **record(sample, outcomes[sample["token"]], int(ix[position])))


def cv_objective(rows):
    frame = pd.DataFrame(rows)
    uniform = frame[frame["slice"] == "uniform"]
    if uniform.empty:
        raise ValueError("Uniform training support required")
    return dict(uniform_gain_macro_pp=float(uniform.groupby("session_id").raw_gain_pp.mean().mean()),
                harm=int(frame.raw_harm.sum()), interventions=int(frame.intervention.sum()))


def verify_fit(out):
    fit = read(out / "fit.json")
    for name, digest in fit["artifacts"].items():
        if sha(out / name) != digest:
            raise RuntimeError("Frozen fitted artifact changed: " + name)
    return fit


def fit(entries, outcomes, config, out):
    if (out / "fit.json").exists():
        return verify_fit(out)
    selected, trials, cv_rows, final_models = {}, {}, [], {}
    for family in FAMILIES:
        rows = {threshold: [] for threshold in THRESHOLDS}
        raw = []
        for sid in sorted({e["sample"]["session_id"] for e in entries}):
            valid, prediction = fold_predictions(entries, family, sid, config, out)
            for j, e in enumerate(valid):
                raw.append(chosen_row(e, prediction, j, None, outcomes, config, family + "_raw"))
                for threshold in THRESHOLDS:
                    rows[threshold].append(chosen_row(e, prediction, j, threshold, outcomes, config, family + "_gated"))
        scores = {t: cv_objective(r) for t, r in rows.items()}
        best = min(THRESHOLDS, key=lambda t: (scores[t]["harm"] > 0, -scores[t]["uniform_gain_macro_pp"], scores[t]["harm"], -t))
        selected[family] = dict(threshold=best, **scores[best])
        trials[family] = {str(t): v for t, v in scores.items()}
        cv_rows.extend(raw + rows[best])
        final = out / (family + ".npz")
        marker = out / (family + "_model.json")
        if marker.exists():
            meta = read(marker)
            if sha(final) != meta["sha256"]:
                raise RuntimeError("Final model checkpoint changed")
        else:
            net, mean, scale, history = fit_one(entries, family, config, family + "/all-MF")
            save(final, **model.checkpoint(net, mean, scale))
            meta = dict(sha256=sha(final), parameters=sum(p.numel() for p in net.parameters()), history=history)
            write(marker, meta)
        final_models[family] = meta
    eligible = [f for f in FAMILIES if selected[f]["harm"] == 0 and selected[f]["uniform_gain_macro_pp"] > 1e-8]
    primary = max(eligible, key=lambda f: (selected[f]["uniform_gain_macro_pp"], f == "mlp")) + "_gated" if eligible else "baseline"
    csv(out / "fit_cv_per_scene.csv", cv_rows)
    artifacts = {p.name: sha(p) for p in [out / "fit_cv_per_scene.csv"] +
                 [out / (f + suffix) for f in FAMILIES for suffix in (".npz", "_model.json")]}
    result = dict(status="complete", primary=primary, config=asdict(config), selected=selected, trials=trials,
                  models=final_models, artifacts=artifacts,
                  cv_methods=summary(cv_rows, config.seed), note=NOTE,
                  selection="Leave-one-entire-M/F-session-out; train-only scaling; fixed epochs/seed. "
                            "Gate tuned by uniform macro gain, zero observed harm on uniform+challenge. "
                            "Tie favors higher threshold; primary tie favors MLP; no positive eligible gain => baseline.")
    write(out / "fit.json", result)
    return result


def prepare_decisions(source, validation, decoder, vocab, entries, fitted, config, out):
    """No development outcome lookup in this function, including either cohort."""
    nets = {}
    for family in FAMILIES:
        with np.load(out / (family + ".npz"), allow_pickle=False) as arrays:
            nets[family] = model.restore({k: arrays[k].copy() for k in arrays.files}, family, config)
    used = {e["sample"]["session_id"] for e in entries}
    cohorts = {}
    for root in (source, validation):
        cohort, sessions = [], set()
        index = read(root / "export_index.json")
        for r in read(root / "splits.json")["records"]:
            if r["role"] != "development":
                continue
            if r["session_id"] in used:
                raise ValueError("Development/training or cross-cohort session overlap")
            s = load_decoder_sample(root, r, index, decoder)
            x, ix, b = model.inputs(s, vocab)
            choices = dict(baseline=int(s["selected_local"]))
            diagnostics = {}
            for family in FAMILIES:
                net, mean, scale = nets[family]
                pred = model.predict(net, [x], [b], mean, scale)
                for suffix, threshold in (("raw", None), ("gated", fitted["selected"][family]["threshold"])):
                    method = family + "_" + suffix
                    position = model.choose(*(pred[k][0] for k in ("rank", "gain", "good_logit")),
                                            b, s["candidate_ids"][ix], threshold, config.improvement_min)
                    choices[method] = int(ix[position])
                    diagnostics[method] = dict(predicted_gain_pp=100 * float(pred["gain"][0, position]),
                        benefit_estimate=float(1 / (1 + np.exp(-np.clip(pred["good_logit"][0, position], -700, 700)))))
            cohort.append(dict(token=s["token"], session_id=s["session_id"],
                               candidate_ids=s["candidate_ids"].tolist(), selected_local=int(s["selected_local"]),
                               top16=ix.tolist(), choices=choices, diagnostics=diagnostics))
            sessions.add(s["session_id"])
        if not cohort:
            raise ValueError("Empty development cohort")
        cohorts[root.name] = cohort
        used.update(sessions)
    frozen = dict(fit_sha256=sha(out / "fit.json"), primary=fitted["primary"], cohorts=cohorts)
    marker = out / "query_decisions.json"
    if marker.exists() and read(marker) != frozen:
        raise RuntimeError("Frozen development decisions changed")
    write(marker, frozen)  # BOTH cohorts fixed before evaluate() can read outcomes.
    return frozen


def evaluate(decisions, source, validation, expected, out, config):
    if read(out / "query_decisions.json") != decisions:
        raise RuntimeError("Persist both cohorts' decisions before outcome lookup")
    result = dict(status="complete", primary=decisions["primary"], cohorts={}, note=NOTE)
    for root in (source, validation):
        rows = []
        for s in decisions["cohorts"][root.name]:
            path = root / "development_oracle/scenes" / (s["token"] + ".json")
            if sha(path) != expected[root.name][s["token"]]:
                raise RuntimeError("Frozen development oracle changed")
            state = read(path)
            if state["token"] != s["token"]:
                raise ValueError("Development oracle token mismatch")
            outcomes = state["outcomes"]
            for local in s["top16"]:
                for version in ("raw", "filtered"):
                    values = [outcomes[str(local)][version][k] for k in ("pdm_score",) + model.METRICS]
                    if not np.isfinite(values).all() or min(values) < 0 or max(values) > 1:
                        raise ValueError("Invalid development outcomes")
            oracle = max(outcomes[str(i)]["raw"]["pdm_score"] for i in s["top16"])
            for method, chosen in s["choices"].items():
                if chosen not in s["top16"]:
                    raise ValueError("Decision outside frozen top16")
                rows.append(dict(method=method, oracle_score=oracle, **s["diagnostics"].get(method, {}),
                                 **record(s, outcomes, chosen)))
        csv(out / (root.name + "_per_scene.csv"), rows)
        frame = pd.DataFrame(rows)
        by_session = frame.groupby(["session_id", "method"]).agg(gain_pp=("raw_gain_pp", "mean"),
                              harm=("raw_harm", "sum"), interventions=("intervention", "sum")).reset_index()
        csv(out / (root.name + "_by_session.csv"), by_session.to_dict("records"))
        leave_one_out = {}
        for method, group in by_session.groupby("method"):
            leave_one_out[method] = {sid: float(group.loc[group.session_id != sid, "gain_pp"].mean())
                                     for sid in group.session_id} if len(group) > 1 else {}
        result["cohorts"][root.name] = dict(scenes=len(decisions["cohorts"][root.name]),
                    sessions=int(frame.session_id.nunique()), methods=summary(rows, config.seed),
                    leave_one_session_out_gain_pp=leave_one_out)
    result["decisions_sha256"] = sha(out / "query_decisions.json")
    write(out / "report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("audit", "fit", "predict", "evaluate", "report", "all"))
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory"
    p.add_argument("--prior", type=Path, default=base / "experiments/pair_utility_v1")
    p.add_argument("--addition", type=Path, default=base / "mf_new4_shard174_v1")
    p.add_argument("--decoder", type=Path, default=base / "decoder_features_functional_v1")
    p.add_argument("--out", type=Path, default=base / "experiments/trajectory_selector_v1")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    config = model.SelectorConfig(epochs=args.epochs)
    if args.threads < 1:
        raise ValueError("Positive thread count required")
    torch.set_num_threads(args.threads)
    prior, addition, decoder, out = [x.resolve() for x in (args.prior, args.addition, args.decoder, args.out)]
    # Loader validates the unchanged old manifests and functional export; no RGB/GPU work.
    bank, fitting, outcomes, vocab, source, validation, completion, bindings, counts, _ = load_decoder_inputs(prior, addition, decoder)
    for root in (prior, addition, decoder, source, validation, completion):
        if out == root or out in root.parents or root in out.parents:
            raise ValueError("Separate experiment output required")
    if read(decoder / "identity.json").get("exporter_path") != "navsim/planning/script/run_drivemem_decoder_functional_export.py":
        raise ValueError("Use the accepted functional decoder sidecar")
    identity = dict(schema=1, config=asdict(config), inputs=bindings, counts=counts, threads=args.threads,
                    python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, platform=platform.system(),
                    sources={name: sha(REPO / name) for name in (
                        "navsim/agents/trajectory_selector.py", "navsim/planning/script/run_trajectory_selector.py",
                        "navsim/agents/drivemem.py", "navsim/agents/drivemem_action_transfer.py",
                        "navsim/agents/drivemem_decoder_utility.py", "navsim/planning/script/run_drivemem_expanded_utility.py",
                        "navsim/planning/script/run_drivemem_action_transfer.py")},
                    thresholds=list(THRESHOLDS), protocol="supervised_top16_no_memory", note=NOTE)
    marker = out / "identity.json"
    if marker.exists() and read(marker) != identity:
        raise RuntimeError("Selector inputs/code/runtime changed; use a new --out directory")
    if out.exists() and not marker.exists() and any(out.iterdir()):
        raise ValueError("Nonempty output without identity")
    if args.stage == "report":
        print(json.dumps(read(out / "report.json"), indent=2))
        return
    write(marker, identity)
    tags = {r["token"]: r["sampling"] for r in read(addition / "label_selection.json")["records"]}
    entries = build_training(bank, fitting, outcomes, vocab, tags, config)
    if args.stage in ("audit", "all"):
        report = audit(entries, out, config)
        if not report["meaningful_ranking_scenes"]:
            raise ValueError("No meaningful ranking supervision; inspect audit.json")
        if args.stage == "audit":
            return
    if args.stage in ("fit", "all"):
        fitted = fit(entries, outcomes, config, out)
        print(json.dumps(dict(primary=fitted["primary"], selected=fitted["selected"], models=fitted["models"]), indent=2), flush=True)
        if args.stage == "fit":
            return
    else:
        fitted = verify_fit(out)
    decisions = prepare_decisions(source, validation, decoder, vocab, entries, fitted, config, out)
    if args.stage == "predict":
        print("Both development cohorts frozen in " + str(out / "query_decisions.json"), flush=True)
        return
    expected = read(completion / "report.json")["oracle_inputs"]
    result = evaluate(decisions, source, validation, expected, out, config)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
