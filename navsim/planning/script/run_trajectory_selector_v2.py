"""CPU experiment on the frozen 399-scene bundle; fresh holdout stays unopened.

Compare unchanged v1 MLP training with consistent gain ranking, on identical
session folds, inputs and labels. Old development cohorts are historical only.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch

from navsim.agents import trajectory_selector as original
from navsim.agents import trajectory_selector_v2 as model
from navsim.agents.selector_training_bundle import load_bundle
from navsim.agents.drivemem_decoder_utility import load_decoder_sample, verify_decoder_protocol
from navsim.planning.script.run_drivemem_action_transfer import record, summary
from navsim.planning.script.run_drivemem_expanded_utility import REPO, read, write, save, sha

THRESHOLDS = (.5, .75, .9, 1.01)
NOTE = ("Local PDM, not EPDMS. Equal session weighting. Gate/family selection uses training-session OOF "
        "predictions and is tuning, not unbiased evaluation. Benefit estimates and zero observed component "
        "degradation are not calibrated safety. Old development cohorts are historical diagnostics. "
        "Four fresh holdout sessions remain unopened. No additional images, scorer calls or GPU work.")


def csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(path)


def audit(entries, out, config):
    if len({e["sample"]["session_id"] for e in entries}) < 3:
        raise ValueError("At least three training sessions required")
    rows = [dict(token=e["sample"]["token"], session_id=e["sample"]["session_id"],
                 origin=e["metadata"]["origin"], slice=e["sampling_slice"],
                 baseline=float(e["targets"]["quality"][e["baseline"]]),
                 oracle_gain_pp=100 * float(e["targets"]["gain"].max()),
                 useful_alternatives=int(e["targets"]["good"].sum())) for e in entries]
    frame = pd.DataFrame(rows)
    result = dict(status="complete", scenes=len(rows), sessions=int(frame.session_id.nunique()),
                  feature_dim=entries[0]["x"].shape[1], slices=dict(Counter(frame["slice"])),
                  origins=dict(Counter(frame.origin)),
                  oracle_gain_macro_pp=float(frame.groupby("session_id").oracle_gain_pp.mean().mean()),
                  scenes_with_useful_alternative=int((frame.useful_alternatives > 0).sum()),
                  fresh_holdout_processed=0, families=list(model.FAMILIES), note=NOTE)
    csv(out / "training_support.csv", rows)
    write(out / "audit.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def fit_one(entries, family, config, label):
    def progress(epoch, loss):
        print(f"TRAIN {label} epoch={epoch}/{config.epochs} loss={loss:.6f}", flush=True)
    return model.train(entries, family, config, progress)


def fold_predictions(entries, family, heldout, config, out):
    training = [e for e in entries if e["sample"]["session_id"] != heldout]
    valid = [e for e in entries if e["sample"]["session_id"] == heldout]
    if not training or not valid:
        raise ValueError("Empty training/validation fold")
    fold = out / "cv" / (family + "_" + hashlib.sha256(heldout.encode()).hexdigest()[:16])
    partition = dict(train_sessions=sorted({e["sample"]["session_id"] for e in training}),
                     train_tokens=[e["sample"]["token"] for e in training],
                     valid_session=heldout, valid_tokens=[e["sample"]["token"] for e in valid])
    if fold.with_suffix(".json").exists():
        frozen = read(fold.with_suffix(".json"))
        if frozen["partition"] != partition or sha(fold.with_suffix(".npz")) != frozen["sha256"]:
            raise RuntimeError("Cached CV fold changed")
        with np.load(fold.with_suffix(".npz"), allow_pickle=False) as data:
            prediction = {k: data[k].copy() for k in ("rank", "gain", "good_logit")}
        return valid, prediction
    print(f"CV family={family} held-out={heldout} train={len(training)} valid={len(valid)}", flush=True)
    net, mean, scale, history = fit_one(training, family, config, family + "/" + heldout)
    prediction = model.predict(net, valid, mean, scale)
    fold.parent.mkdir(parents=True, exist_ok=True)
    save(fold.with_suffix(".npz"), **prediction, mean=mean, scale=scale)
    write(fold.with_suffix(".json"), dict(partition=partition, sha256=sha(fold.with_suffix(".npz")), history=history))
    return valid, prediction


def chosen_row(e, prediction, j, family, threshold, config, method):
    p = model.chosen_position(e, prediction, j, family, threshold, config)
    gain = float(prediction["gain"][j, p])
    return dict(method=method, slice=e["sampling_slice"], origin=e["metadata"]["origin"],
                predicted_gain_pp=100 * gain,
                benefit_estimate=float(1 / (1 + np.exp(-np.clip(prediction["good_logit"][j, p], -700, 700)))),
                negative_predicted_gain_selected=bool(p != e["baseline"] and gain < -1e-7),
                oracle_score=max(v["raw"]["pdm_score"] for v in e["outcomes"].values()),
                **record(e["sample"], e["outcomes"], int(e["indices"][p])))


def objective(rows):
    frame = pd.DataFrame(rows)
    uniform = frame[frame["slice"] == "uniform"]
    if uniform.empty:
        raise ValueError("Uniform training scenes required for gate selection")
    return dict(uniform_gain_macro_pp=float(uniform.groupby("session_id").raw_gain_pp.mean().mean()),
                harm=int(frame.raw_harm.sum()), quality_degraded=int(frame.raw_quality_degraded.sum()),
                interventions=int(frame.intervention.sum()),
                negative_predicted_gain_selected=int(frame.negative_predicted_gain_selected.sum()))


def paired_comparison(rows, seed):
    """Pair methods on identical scenes, then resample whole sessions."""
    frame = pd.DataFrame(rows)
    result = {}
    for suffix in ("raw", "gated"):
        control, consistent = (family + "_" + suffix for family in model.FAMILIES)
        table = frame[frame.method.isin((control, consistent))].pivot(
            index=["session_id", "token"], columns="method", values="raw_gain_pp")
        if table.empty or table.isna().any().any() or set(table.columns) != {control, consistent}:
            raise ValueError("Paired comparison requires identical scene coverage")
        delta = table[consistent] - table[control]
        sessions = delta.groupby("session_id").mean()
        picks = np.random.default_rng(seed).integers(0, len(sessions), size=(2000, len(sessions)))
        result[suffix] = dict(consistent_minus_control_macro_pp=float(sessions.mean()),
            consistent_minus_control_micro_pp=float(delta.mean()),
            ci95_sessions=np.quantile(sessions.to_numpy()[picks].mean(1), [.025, .975]).tolist(),
            by_session_pp={sid: float(value) for sid, value in sessions.items()})
    return result


def verify_fit(out):
    fitted = read(out / "fit.json")
    for name, digest in fitted["artifacts"].items():
        if sha(out / name) != digest:
            raise RuntimeError("Frozen fitted artifact changed: " + name)
    return fitted


def fit(entries, config, out):
    if (out / "fit.json").exists():
        return verify_fit(out)
    selected, trials, cv_rows, models = {}, {}, [], {}
    for family in model.FAMILIES:
        gated, raw = {t: [] for t in THRESHOLDS}, []
        for sid in sorted({e["sample"]["session_id"] for e in entries}):
            valid, prediction = fold_predictions(entries, family, sid, config, out)
            for j, e in enumerate(valid):
                raw.append(chosen_row(e, prediction, j, family, None, config, family + "_raw"))
                for t in THRESHOLDS:
                    gated[t].append(chosen_row(e, prediction, j, family, t, config, family + "_gated"))
        scores = {t: objective(rows) for t, rows in gated.items()}
        best = min(THRESHOLDS, key=lambda t: (scores[t]["harm"] > 0, -scores[t]["uniform_gain_macro_pp"], scores[t]["harm"], -t))
        selected[family] = dict(threshold=best, **scores[best])
        trials[family] = {str(t): score for t, score in scores.items()}
        cv_rows.extend(raw + gated[best])
        final, marker = out / (family + ".npz"), out / (family + "_model.json")
        if marker.exists():
            meta = read(marker)
            if sha(final) != meta["sha256"]:
                raise RuntimeError("Final model checkpoint changed")
        else:
            net, mean, scale, history = fit_one(entries, family, config, family + "/all-train")
            save(final, **original.checkpoint(net, mean, scale))
            meta = dict(sha256=sha(final), parameters=sum(p.numel() for p in net.parameters()), history=history)
            write(marker, meta)
        models[family] = meta
    eligible = [f for f in model.FAMILIES if selected[f]["harm"] == 0 and selected[f]["uniform_gain_macro_pp"] > 1e-8]
    primary = (max(eligible, key=lambda f: (selected[f]["uniform_gain_macro_pp"], f == "control_v1")) + "_gated"
               if eligible else "baseline")
    csv(out / "fit_cv_per_scene.csv", cv_rows)
    artifacts = {p.name: sha(p) for p in [out / "fit_cv_per_scene.csv"] +
                 [out / (f + suffix) for f in model.FAMILIES for suffix in (".npz", "_model.json")]}
    result = dict(status="complete", primary=primary, config=asdict(config), selected=selected, trials=trials,
                  models=models, artifacts=artifacts, cv_methods=summary(cv_rows, config.seed),
                  paired_comparison=paired_comparison(cv_rows, config.seed),
                  fresh_holdout_processed=0, note=NOTE,
                  selection="Leave-one-entire-session-out, train-only scaler, fixed epochs/seed. Gate tuned on "
                            "uniform macro gain with zero observed harm across uniform+challenge. Ties favor "
                            "higher threshold and control_v1. No positive eligible result => baseline.")
    write(out / "fit.json", result)
    return result


def prepare_decisions(roots, decoder, vocab, entries, fitted, config, out):
    """Only historical development features; no outcome lookup in either cohort."""
    nets = {}
    for family in model.FAMILIES:
        with np.load(out / (family + ".npz"), allow_pickle=False) as data:
            nets[family] = model.restore({k: data[k].copy() for k in data.files}, family, config)
    used = {e["sample"]["session_id"] for e in entries}
    cohorts = {}
    for root in roots:
        cohort, sessions = [], set()
        index = read(root / "export_index.json")
        for r in read(root / "splits.json")["records"]:
            if r["role"] != "development":
                continue
            if r["session_id"] in used:
                raise ValueError("Historical/training or cross-cohort session overlap")
            s = load_decoder_sample(root, r, index, decoder)
            x, ix, b = original.inputs(s, vocab)
            e = dict(sample=s, x=x, indices=ix, baseline=b)
            choices, diagnostics = dict(baseline=int(s["selected_local"])), {}
            for family in model.FAMILIES:
                net, mean, scale = nets[family]
                prediction = model.predict(net, [e], mean, scale)
                for suffix, threshold in (("raw", None), ("gated", fitted["selected"][family]["threshold"])):
                    method = family + "_" + suffix
                    p = model.chosen_position(e, prediction, 0, family, threshold, config)
                    choices[method] = int(ix[p])
                    gain = float(prediction["gain"][0, p])
                    diagnostics[method] = dict(predicted_gain_pp=100 * gain,
                        benefit_estimate=float(1 / (1 + np.exp(-np.clip(prediction["good_logit"][0, p], -700, 700)))),
                        negative_predicted_gain_selected=bool(p != b and gain < -1e-7))
            cohort.append(dict(token=s["token"], session_id=s["session_id"], selected_local=int(s["selected_local"]),
                               candidate_ids=s["candidate_ids"].tolist(), top16=ix.tolist(), choices=choices, diagnostics=diagnostics))
            sessions.add(s["session_id"])
        if not cohort:
            raise ValueError("Empty historical development cohort")
        cohorts[root.name] = cohort
        used.update(sessions)
    frozen = dict(fit_sha256=sha(out / "fit.json"), primary=fitted["primary"], cohorts=cohorts, fresh_holdout_processed=0)
    marker = out / "query_decisions.json"
    if marker.exists() and read(marker) != frozen:
        raise RuntimeError("Frozen historical decisions changed")
    write(marker, frozen)
    return frozen


def evaluate(decisions, roots, expected, out, config):
    if read(out / "query_decisions.json") != decisions:
        raise RuntimeError("Freeze BOTH historical cohorts before outcome lookup")
    result = dict(status="complete", primary=decisions["primary"], cohorts={}, fresh_holdout_processed=0, note=NOTE)
    for root in roots:
        rows = []
        for s in decisions["cohorts"][root.name]:
            path = root / "development_oracle/scenes" / (s["token"] + ".json")
            if sha(path) != expected[root.name][s["token"]]:
                raise RuntimeError("Frozen historical oracle changed")
            state = read(path)
            if state["token"] != s["token"]:
                raise ValueError("Historical oracle token mismatch")
            outcomes = state["outcomes"]
            for local in s["top16"]:
                for version in ("raw", "filtered"):
                    values = [outcomes[str(local)][version][k] for k in ("pdm_score",) + original.METRICS]
                    if not np.isfinite(values).all() or min(values) < 0 or max(values) > 1:
                        raise ValueError("Invalid historical outcomes")
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
        leave_one_out = {method: {sid: float(g.loc[g.session_id != sid, "gain_pp"].mean()) for sid in g.session_id}
                         if len(g) > 1 else {} for method, g in by_session.groupby("method")}
        result["cohorts"][root.name] = dict(scenes=len(decisions["cohorts"][root.name]), sessions=int(frame.session_id.nunique()),
                    methods=summary(rows, config.seed), paired_comparison=paired_comparison(rows, config.seed),
                    leave_one_session_out_gain_pp=leave_one_out)
    result["decisions_sha256"] = sha(out / "query_decisions.json")
    write(out / "report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("audit", "fit", "historical", "report", "all"))
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory"
    p.add_argument("--bundle", type=Path, default=base / "selector_training_bundle_v2")
    p.add_argument("--addition", type=Path, default=base / "mf_new4_shard174_v1")
    p.add_argument("--prior", type=Path, default=base / "experiments/pair_utility_v1")
    p.add_argument("--decoder", type=Path, default=base / "decoder_features_functional_v1")
    p.add_argument("--out", type=Path, default=base / "experiments/trajectory_selector_v2")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    if min(a.threads, a.epochs) < 1:
        raise ValueError("Positive epochs and thread count required")
    torch.set_num_threads(a.threads)
    config = original.SelectorConfig(epochs=a.epochs)
    bundle, addition, prior, decoder, out = [x.resolve() for x in (a.bundle, a.addition, a.prior, a.decoder, a.out)]
    examples, vocab = load_bundle(bundle)
    bundle_identity = read(bundle / "identity.json")
    tags_path = addition / "label_selection.json"
    if bundle_identity["inputs"].get(str(tags_path)) != sha(tags_path):
        raise RuntimeError("Challenge selection does not match training bundle")
    challenge = {r["token"] for r in read(tags_path)["records"] if r["sampling"].startswith("challenge")}
    entries = model.entries_from_bundle(examples, vocab, challenge, config)
    state = read(prior / "identity.json")
    roots = [Path(state[k]).resolve() for k in ("source", "validation")]
    completion = Path(state["completion"]).resolve()
    for root in (bundle, addition, prior, decoder, completion, *roots):
        if out == root or out in root.parents or root in out.parents:
            raise ValueError("Separate experiment output required")
    decoder_identity = read(decoder / "identity.json")
    if verify_decoder_protocol(decoder_identity) != "navsim/planning/script/run_drivemem_decoder_functional_export.py":
        raise ValueError("Accepted functional sidecar required for historical diagnostics")
    paths = [bundle / name for name in ("bundle_manifest.json", "identity.json")] + [tags_path,
             prior / "identity.json", completion / "report.json", decoder / "identity.json", decoder / "export_index.json"] + [
             root / name for root in roots for name in ("splits.json", "export_index.json")]
    # Training bundle already pins the old diagnostic manifests. Reject unrelated roots.
    for path in paths[2:]:
        if bundle_identity["inputs"].get(str(path)) != sha(path):
            raise RuntimeError("Historical input is not the bundled snapshot: " + str(path))
    sources = ("navsim/agents/trajectory_selector_v2.py", "navsim/planning/script/run_trajectory_selector_v2.py",
               "navsim/agents/trajectory_selector.py", "navsim/agents/selector_training_bundle.py",
               "navsim/agents/drivemem.py", "navsim/agents/drivemem_action_transfer.py",
               "navsim/agents/drivemem_decoder_utility.py", "navsim/planning/script/run_drivemem_expanded_utility.py",
               "navsim/planning/script/run_drivemem_action_transfer.py")
    identity = dict(schema=1, inputs={str(path): sha(path) for path in paths}, sources={name: sha(REPO / name) for name in sources},
                    config=asdict(config), protocol=model.PROTOCOL, thresholds=list(THRESHOLDS), threads=a.threads,
                    python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, platform=platform.system(),
                    counts=bundle_identity["counts"], fresh_holdout_processed=0, note=NOTE)
    marker = out / "identity.json"
    if marker.exists() and read(marker) != identity:
        raise RuntimeError("Selector v2 inputs/code/runtime changed; use a new --out directory")
    if out.exists() and not marker.exists() and any(out.iterdir()):
        raise ValueError("Nonempty output without identity")
    if a.stage == "report":
        verify_fit(out)
        result = read(out / "report.json")
        if result["decisions_sha256"] != sha(out / "query_decisions.json"):
            raise RuntimeError("Frozen historical decisions changed")
        print(json.dumps(result, indent=2), flush=True)
        return
    write(marker, identity)
    if a.stage in ("audit", "all"):
        audit(entries, out, config)
        if a.stage == "audit":
            return
    fitted = fit(entries, config, out) if a.stage in ("fit", "all") else verify_fit(out)
    print(json.dumps(dict(primary=fitted["primary"], selected=fitted["selected"], models=fitted["models"]), indent=2), flush=True)
    if a.stage == "fit":
        return
    decisions = prepare_decisions(roots, decoder, vocab, entries, fitted, config, out)
    result = evaluate(decisions, roots, read(completion / "report.json")["oracle_inputs"], out, config)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
