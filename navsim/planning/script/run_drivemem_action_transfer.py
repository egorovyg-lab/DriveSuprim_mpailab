"""Sidecar experiment: F top-16 labels, M/F-only selection, frozen V lookup.

Never opens calibration/test outcomes. Original pilot artifacts are read-only.
"""
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.agents.drivemem import METRICS, SAFETY
from navsim.agents.drivemem_action_transfer import (
    pool, transfer, choose, bandwidth_scale, ridge_path, ridge_predict,
)


def record(sample, outcomes, index):
    base = outcomes[str(int(sample["selected_local"]))]
    value = outcomes[str(index)]
    row = dict(token=sample["token"], session_id=sample["session_id"], candidate_local=index,
               candidate_id=int(sample["candidate_ids"][index]), intervention=index != int(sample["selected_local"]))
    for version in ("raw", "filtered"):
        a, b = base[version], value[version]
        row[version + "_score"] = b["pdm_score"]
        row[version + "_gain_pp"] = 100 * (b["pdm_score"] - a["pdm_score"])
        row[version + "_harm"] = any(b[METRICS[j]] + 1e-8 < a[METRICS[j]] for j in SAFETY)
        row[version + "_absolute_violation"] = any(b[METRICS[j]] < 1 - 1e-8 for j in SAFETY)
        row[version + "_quality_degraded"] = b["pdm_score"] + 1e-8 < a["pdm_score"]
        row.update({version + "_delta_" + key: b[key] - a[key] for key in METRICS})
    return row


def objective(rows):
    frame = pd.DataFrame(rows)
    return (-float(frame.groupby("session_id").raw_gain_pp.mean().mean()), int(frame.raw_harm.sum()))


def summary(rows, seed):
    frame = pd.DataFrame(rows)
    methods = {}
    for name, group in frame.groupby("method", sort=True):
        sessions = group.groupby("session_id").raw_gain_pp.mean().to_numpy()
        pick = np.random.default_rng(seed).integers(0, len(sessions), size=(2000, len(sessions)))
        active = group[group.intervention]
        methods[name] = dict(score_mean=float(group.raw_score.mean()), gain_macro_pp=float(sessions.mean()),
            gain_micro_pp=float(group.raw_gain_pp.mean()), sessions=len(sessions),
            gain_ci95_sessions=np.quantile(sessions[pick].mean(axis=1), [.025, .975]).tolist(),
            interventions=len(active), coverage=len(active) / len(group), harm=int(group.raw_harm.sum()),
            harm_among_interventions=float(active.raw_harm.mean()) if len(active) else None,
            quality_degraded=int(group.raw_quality_degraded.sum()), improved=int((group.raw_gain_pp > 1e-6).sum()),
            absolute_violation=float(group.raw_absolute_violation.mean()),
            filtered_gain_pp=float(group.filtered_gain_pp.mean()))
        if "oracle_score" in group:
            baseline = group.raw_score - group.raw_gain_pp / 100
            reserve = float((100 * (group.oracle_score - baseline)).sum())
            methods[name]["oracle_reserve_fraction"] = float(group.raw_gain_pp.sum() / reserve) if reserve > 1e-8 else None
            methods[name]["oracle_regret_mean_pp"] = float((100 * (group.oracle_score - group.raw_score)).mean())
    return methods


def select_models(bank, fit, targets, vocab, seed, out, read, write, save_npz, sha):
    labels_hashes = {p.name: sha(p) for p in sorted((out / "fit_labels").glob("*.json"))}
    if (out / "fit.json").exists():
        frozen = read(out / "fit.json")
        if frozen["sidecar_labels"] != labels_hashes or frozen["ridge_sha256"] != sha(out / "ridge.npz"):
            raise RuntimeError("Frozen fit/model changed")
        return frozen
    h0 = bandwidth_scale(bank, vocab, seed)
    configs, trials = {}, []
    for family in ("B0", "B1", "B2", "B3"):
        widths = [.25] if family in ("B0", "B2") else [h0 * x for x in (.5, 1., 2.)]
        candidates = []
        for h in widths:
            rows = []
            for sample in fit:
                c, _ = transfer(sample, bank, vocab, h, family in ("B2", "B3"), seed)
                rows.append(record(sample, targets[sample["token"]], choose(sample, c)))
            key = objective(rows)
            trials.append(dict(family=family, h=h, gain_macro_pp=-key[0], harm=key[1]))
            candidates.append((key, h))
        key, h = min(candidates)
        configs[family] = dict(h=h, contrast=family in ("B2", "B3"), fit_gain_macro_pp=-key[0], fit_harm=key[1])
    primary = min(("B1", "B2", "B3"), key=lambda f: (-configs[f]["fit_gain_macro_pp"], configs[f]["fit_harm"], f))
    lambdas = [1., 10., 100., 1000.]
    cv = {lam: [] for lam in lambdas}
    for sid in sorted({s["session_id"] for s in fit}):
        train = bank + [s for s in fit if s["session_id"] != sid]
        valid = [s for s in fit if s["session_id"] == sid]
        print(f"RIDGE CV held-out session={sid}; training scenes={len(train)}", flush=True)
        models = ridge_path(train, vocab, lambdas)
        for lam, model in models.items():
            for sample in valid:
                cv[lam].append(record(sample, targets[sample["token"]], choose(sample, ridge_predict(sample, vocab, model))))
    best_lambda = min(lambdas, key=lambda lam: (objective(cv[lam]), -lam))
    model = ridge_path(bank + fit, vocab, [best_lambda])[best_lambda]
    save_npz(out / "ridge.npz", **model)
    result = dict(status="complete", primary=primary, configs=configs, h0=h0, beta=.5, neighbors=5, topk=16,
        ridge_lambda=best_lambda, ridge_representation="full normalized visual embedding + ego + action + heads + interactions",
        ridge_cv=[dict(regularization=lam, gain_macro_pp=-objective(cv[lam])[0], harm=objective(cv[lam])[1]) for lam in lambdas],
        trials=trials, sidecar_labels=labels_hashes, ridge_sha256=sha(out / "ridge.npz"),
        selection="M bandwidth scale; F macro gain, ties by harm; ridge leave-one-F-session-out; V outcomes unopened")
    write(out / "fit.json", result)
    return result


def main(cfg):
    from navsim.planning.script.run_drivemem_memory import (
        verify_identity, read_json as read, write_json as write, sha, sample as load_sample,
        training_samples, evaluators, save_npz,
    )
    from navsim.planning.script.run_drivemem_development_oracle import verify_baseline
    source = Path(cfg.drivemem.root)
    verify_identity(cfg, source)
    if str(cfg.drivemem.memory_mode) != "enriched":
        raise ValueError("B0-B3 experiment requires enriched memory; executed-only is a separate mechanism")
    stage = str(cfg.drivemem.stage).removeprefix("action-")
    if stage not in ("all", "plan", "diagnose", "label", "fit", "evaluate", "report"):
        raise ValueError("Unknown action-transfer stage")
    out = Path(os.environ.get("DRIVEMEM_ACTION_ROOT", str(Path(os.environ["NAVSIM_EXP_ROOT"]) / "drivemem_memory/experiments/action_transfer_v2"))).resolve()
    if out == source.resolve() or out in source.resolve().parents or source.resolve() in out.parents:
        raise ValueError("Experiment root must be separate from source root")
    repo = Path(__file__).resolve().parents[3]
    provenance = dict(schema=1, source=str(source.resolve()), seed=int(cfg.drivemem.seed), python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__,
        sources={name: sha(repo / name) for name in (
            "navsim/agents/drivemem_action_transfer.py", "navsim/planning/script/run_drivemem_action_transfer.py",
            "navsim/planning/script/run_drivemem_development_oracle.py")},
        inputs={name: sha(source / name) for name in (
            "identity.json", "split_identity.json", "vocab.npz", "export_index.json", "cache_index.json", "fit.json")},
        labels={p.name: sha(p) for p in sorted((source / "labels").glob("*.json"))},
        protocol="top16; beta=.5; k=5; B0-B3; full-embedding ridge lambda=1,10,100,1000; random seeds seed+0,1,2")
    marker = out / "identity.json"
    if marker.exists() and read(marker) != provenance:
        raise RuntimeError("Experiment inputs/code/runtime changed; choose a new DRIVEMEM_ACTION_ROOT")
    if out.exists() and not marker.exists() and any(out.iterdir()):
        raise RuntimeError("Nonempty experiment directory without identity")
    rows = read(source / "splits.json")["records"]
    samples = training_samples(cfg, source)  # Only M/F labels; original hashes verified.
    bank = [s for s in samples if s["role"] == "memory"]
    fit = [s for s in samples if s["role"] == "fit"]
    if len({s["session_id"] for s in fit}) < 2:
        raise ValueError("At least two F sessions required for grouped CV")
    with np.load(source / "vocab.npz", allow_pickle=False) as data:
        vocab = data["vocab"].copy()
    targets, missing = {}, 0
    for s in fit:
        labels = read(source / "labels" / (s["token"] + ".json"))
        original = {str(i): v for i, v in zip(labels["local_indices"], labels["values"])}
        path = out / "fit_labels" / (s["token"] + ".json")
        state = read(path) if path.exists() else dict(token=s["token"], outcomes=original)
        if state["token"] != s["token"]:
            raise RuntimeError("Sidecar token mismatch")
        for key, value in original.items():
            if state["outcomes"].get(key) != value:
                raise RuntimeError("Sidecar differs from original labels")
        targets[s["token"]] = state["outcomes"]
        missing += sum(str(i) not in state["outcomes"] for i in pool(s))
    print(json.dumps(dict(stage=stage, source=str(source), output=str(out), fit_scenes=len(fit),
        new_actions=missing, baseline_repeats=len(fit), scorer_calls=2 * (missing + len(fit)),
        new_image_download_GB=0, gpu_required=False), indent=2), flush=True)
    if stage == "plan":
        return
    write(marker, provenance)
    if stage in ("all", "diagnose"):
        diagnostics = []
        for r in rows:
            if r["role"] not in ("fit", "development"):
                continue
            s = load_sample(source, r)
            _, stats = transfer(s, bank, vocab, .25, seed=int(cfg.drivemem.seed))
            diagnostics.append(dict(token=s["token"], role=s["role"], **stats))
        pd.DataFrame(diagnostics).to_csv(out / "diagnostics.csv", index=False)
        print(pd.DataFrame(diagnostics).groupby("role").mean(numeric_only=True).to_string(), flush=True)
    if stage in ("all", "label"):
        if (out / "fit.json").exists() and missing:
            raise RuntimeError("Cannot append labels after fit is frozen")
        evaluate, _ = evaluators(cfg, source)
        for number, s in enumerate(fit):
            original = targets[s["token"]][str(int(s["selected_local"]))]
            repeated, _ = evaluate(s["token"], int(s["candidate_ids"][int(s["selected_local"])]))
            verify_baseline(original, repeated)
            path = out / "fit_labels" / (s["token"] + ".json")
            for i in pool(s):
                if str(i) not in targets[s["token"]]:
                    targets[s["token"]][str(i)], _ = evaluate(s["token"], int(s["candidate_ids"][i]))
                    write(path, dict(token=s["token"], outcomes=targets[s["token"]]))
            write(path, dict(token=s["token"], outcomes=targets[s["token"]]))
            print(f"FIT LABELS {number + 1}/{len(fit)}", flush=True)
        missing = 0
    if stage in ("all", "fit", "evaluate"):
        if missing or any(not (out / "fit_labels" / (s["token"] + ".json")).exists() for s in fit):
            raise RuntimeError("Run action-label first to complete F top-16")
        for s in fit:
            indices = np.array(sorted(map(int, targets[s["token"]])), dtype=int)
            s.update(label_local_indices=indices, label_values=np.array([[targets[s["token"]][str(i)]["raw"][k] for k in METRICS] for i in indices]))
        frozen = select_models(bank, fit, targets, vocab, int(cfg.drivemem.seed), out, read, write, save_npz, sha)
        print(json.dumps(frozen, indent=2), flush=True)
    if stage in ("all", "evaluate"):
        # Only AFTER fit is frozen may development outcomes be opened.
        expected_oracles = {r["token"]: sha(source / "development_oracle/scenes" / (r["token"] + ".json"))
                            for r in rows if r["role"] == "development"}
        if (out / "report.json").exists() and read(out / "report.json")["oracle_inputs"] != expected_oracles:
            raise RuntimeError("Previously evaluated development oracle inputs changed")
        with np.load(out / "ridge.npz", allow_pickle=False) as data:
            model = {k: data[k].copy() for k in data.files}
        decisions, all_rows, hashes = [], [], {}
        for r in rows:
            if r["role"] != "development":
                continue
            s = load_sample(source, r)
            chosen = dict(baseline=int(s["selected_local"]), no_memory_ridge_full=choose(s, ridge_predict(s, vocab, model)))
            for family, spec in frozen["configs"].items():
                c, _ = transfer(s, bank, vocab, spec["h"], spec["contrast"], int(cfg.drivemem.seed))
                chosen[family] = choose(s, c)
            spec = frozen["configs"][frozen["primary"]]
            for offset in range(3):
                c, _ = transfer(s, bank, vocab, spec["h"], spec["contrast"], int(cfg.drivemem.seed) + offset, random=True)
                chosen[f"random_{offset}"] = choose(s, c)
            c, _ = transfer(s, bank, vocab, .25, seed=int(cfg.drivemem.seed), scene_only=True)
            chosen["scene_only"] = choose(s, c)
            decisions.append(dict(token=s["token"], choices=chosen))
        write(out / "development_decisions.json", dict(fit_sha256=sha(out / "fit.json"), decisions=decisions))
        by_token = {r["token"]: r for r in rows}
        for decision in decisions:
            token = decision["token"]
            s = load_sample(source, by_token[token])
            path = source / "development_oracle/scenes" / (token + ".json")
            hashes[token] = sha(path)
            outcomes = read(path)["outcomes"]
            if any(str(i) not in outcomes for i in pool(s)):
                raise RuntimeError("Development oracle top-16 incomplete")
            oracle_score = max(outcomes[str(i)]["raw"]["pdm_score"] for i in pool(s))
            for method, index in decision["choices"].items():
                all_rows.append(dict(method=method, oracle_score=oracle_score, **record(s, outcomes, index)))
        pd.DataFrame(all_rows).to_csv(out / "development_per_scene.csv", index=False)
        report = dict(status="complete", primary=frozen["primary"], scenes=len(decisions),
            sessions=len({row["session_id"] for row in all_rows}), topk=16,
            methods=summary(all_rows, int(cfg.drivemem.seed)), oracle_inputs=hashes,
            note="Exploratory development; local PDM, not full EPDMS. Enriched bank; no calibrated guarantee. "
                 "Random seeds reported individually; no selection by development outcomes.")
        frame = pd.DataFrame(all_rows)
        pivot = frame.pivot(index=["session_id", "token"], columns="method", values="raw_gain_pp")
        difference = (pivot[frozen["primary"]] - pivot.no_memory_ridge_full).groupby("session_id").mean().to_numpy()
        pick = np.random.default_rng(int(cfg.drivemem.seed)).integers(0, len(difference), size=(2000, len(difference)))
        report["paired_primary_minus_ridge"] = dict(gain_macro_pp=float(difference.mean()),
            ci95_sessions=np.quantile(difference[pick].mean(axis=1), [.025, .975]).tolist())
        random_gains = [report["methods"][f"random_{i}"]["gain_macro_pp"] for i in range(3)]
        report["random_seed_summary"] = dict(mean_gain_pp=float(np.mean(random_gains)), std_gain_pp=float(np.std(random_gains)))
        frame.groupby(["session_id", "method"]).agg(gain_pp=("raw_gain_pp", "mean"),
            interventions=("intervention", "sum"), harm=("raw_harm", "sum")).to_csv(out / "development_by_session.csv")
        write(out / "report.json", report)
    if stage in ("all", "report"):
        print(json.dumps(read(out / "report.json"), indent=2), flush=True)
    if stage not in ("all", "plan", "diagnose", "label", "fit", "evaluate", "report"):
        raise ValueError("Unknown stage")


if __name__ == "__main__":
    import hydra
    hydra.main(config_path="config/pdm_scoring", config_name="default_drivemem_memory", version_base=None)(main)()
