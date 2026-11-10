"""Read-only diagnosis of frozen teacher-veto decisions and training top16 labels.

No policy search, fitting, scorer calls or holdout access. Stored PDM scores are
authoritative: component changes describe co-occurrence, not additive attribution.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import selector_teacher_veto as veto

METRICS, SAFETY = veto.METRICS, veto.SAFETY
SAFETY_NAMES = tuple(METRICS[j] for j in SAFETY)
HOST = "gain_m020"
PREFIX = "mlp/" + HOST + "/"
MEASURED_EPS = 1e-8
NOTE = ("Descriptive training OOF error audit, not independent validation. Candidate rows and CV fits are correlated. "
        "Per-component counts overlap; exact veto patterns are disjoint. Removing one constraint is an oracle "
        "availability diagnostic, not a proposed policy. No thresholds tuned or models selected. Stored raw PDM "
        "is not reconstructed; metric drops are not an additive attribution. Teacher/risk estimates are uncalibrated. "
        "Local PDM, not EPDMS. Fresh holdout remains unopened.")


def read_csv(path):
    return pd.read_csv(path, dtype={"token": str}, low_memory=False, float_precision="round_trip")


def verify_veto(root, night, bundle):
    identity, report = veto.read(root / "identity.json"), veto.read(root / "report.json")
    if report["status"] != "complete" or report["fresh_holdout_processed"] != 0:
        raise ValueError("Completed training-only teacher-veto report required")
    for name, digest in report["artifacts"].items():
        if veto.sha(root / name) != digest:
            raise RuntimeError("Frozen veto artifact changed: " + name)
    if identity["source_sha256"] != veto.sha(Path(veto.__file__)):
        raise RuntimeError("Frozen teacher-veto source changed")
    for path in (night / "report.json", night / "identity.json", bundle / "bundle_manifest.json"):
        if identity["inputs"].get(str(path)) != veto.sha(path):
            raise ValueError("Veto report belongs to other night/bundle inputs")
    if (identity["epsilon"] != veto.EPSILON or identity["candidate"] != veto.CANDIDATE
            or report["fixed_candidate"] != veto.CANDIDATE):
        raise ValueError("Unexpected frozen veto protocol")
    return identity, report


def scene_arrays(example):
    sample, meta, ix = example["sample"], example["metadata"], example["label_indices"]
    if meta["role"] != "train":
        raise ValueError("Training scenes only")
    if not np.array_equal(ix, np.argsort(-sample["selection_scores"], kind="stable")[:16]):
        raise ValueError("Top16 alignment mismatch")
    positions = np.flatnonzero(ix == int(sample["selected_local"]))
    if len(positions) != 1:
        raise ValueError("Unique baseline required")
    b = int(positions[0])
    raw, filtered = example["label_raw"], example["label_filtered"]
    if any(a.shape != (16, 9) or not np.isfinite(a).all() for a in (raw, filtered)):
        raise ValueError("Invalid top16 outcomes")
    logits = np.column_stack([sample["head_" + m][ix] for m in METRICS]).astype(float)
    if not np.isfinite(logits).all():
        raise ValueError("Invalid teacher heads")
    teacher = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
    delta = raw[:, 1:] - raw[b, 1:]
    blocked = teacher[:, SAFETY] + veto.EPSILON < teacher[b, SAFETY]
    blocked[b] = False
    gain = 100 * (raw[:, 0] - raw[b, 0])
    harm = (delta[:, SAFETY] < -MEASURED_EPS).any(axis=1)
    useful = (gain >= 100 * veto.bundles.IMPROVEMENT_MIN) & ~harm
    return dict(sample=sample, meta=meta, ix=ix, baseline=b, raw=raw, filtered=filtered,
                teacher=teacher, blocked=blocked, delta=delta, gain=gain, harm=harm, useful=useful)


def constraint_rows(a):
    """All 15 alternatives once; every exact intersection gets its own pattern."""
    rows = []
    for j, local in enumerate(a["ix"]):
        if j == a["baseline"]:
            continue
        flags = a["blocked"][j]
        pattern = "+".join(m for m, flag in zip(SAFETY_NAMES, flags) if flag) or "none"
        row = dict(token=a["meta"]["token"], session_id=a["meta"]["session_id"],
            candidate_local=int(local), candidate_id=int(a["sample"]["candidate_ids"][local]), teacher_rank=j + 1,
            raw_gain_pp=float(a["gain"][j]), raw_harm=bool(a["harm"][j]), useful=bool(a["useful"][j]),
            veto_pattern=pattern, veto_count=int(flags.sum()))
        for k, m in enumerate(METRICS):
            row["raw_delta_" + m] = float(a["delta"][j, k])
            row["teacher_delta_" + m] = float(a["teacher"][j, k] - a["teacher"][a["baseline"], k])
        row.update({"blocked_" + m: bool(flag) for m, flag in zip(SAFETY_NAMES, flags)})
        rows.append(row)
    return rows


def opportunities(a):
    blocked, b = a["blocked"], a["baseline"]
    masks = dict(unrestricted=np.ones(16, bool), all5=~blocked.any(axis=1))
    for j, m in enumerate(SAFETY_NAMES):
        masks[m] = ~blocked[:, j]
        masks["without_" + m] = ~np.delete(blocked, j, axis=1).any(axis=1)
    rows = []
    for scope, mask in masks.items():
        if not mask[b]:
            raise ValueError("A diagnostic mask excluded the baseline")
        rows.append(dict(token=a["meta"]["token"], session_id=a["meta"]["session_id"], scope=scope,
            oracle_gain_pp=float(a["gain"][mask].max()),
            relative_safe_oracle_gain_pp=float(a["gain"][mask & ~a["harm"]].max()),
            useful_alternatives=int((a["useful"] & mask).sum()), allowed_alternatives=int(mask.sum()) - 1))
    return rows


def check_record(row, a, chosen):
    b, ix, raw, filtered = a["baseline"], a["ix"], a["raw"], a["filtered"]
    if (int(row.candidate_local) != int(ix[chosen])
            or int(row.candidate_id) != int(a["sample"]["candidate_ids"][ix[chosen]])
            or row.session_id != a["meta"]["session_id"]):
        raise ValueError("Saved candidate/session alignment mismatch")
    for version, labels in (("raw", raw), ("filtered", filtered)):
        delta = labels[chosen, 1:] - labels[b, 1:]
        expected = {version + "_score": labels[chosen, 0],
                    version + "_gain_pp": 100 * (labels[chosen, 0] - labels[b, 0])}
        expected.update({version + "_delta_" + m: delta[k] for k, m in enumerate(METRICS)})
        if any(not np.isfinite(float(row[k])) or abs(float(row[k]) - float(value)) > 1e-8
               for k, value in expected.items()):
            raise ValueError("Stored raw/filtered outcome replay failed")
        flags = {version + "_harm": bool((delta[list(SAFETY)] < -MEASURED_EPS).any()),
                 version + "_quality_degraded": bool(labels[chosen, 0] + MEASURED_EPS < labels[b, 0])}
        for k, value in flags.items():
            if row[k] not in (True, False) or bool(row[k]) != value:
                raise ValueError("Saved measured flag mismatch")
    if row.intervention not in (True, False) or bool(row.intervention) != (chosen != b):
        raise ValueError("Saved intervention mismatch")


def intervention(a, chosen, original, post, pred, position):
    b, ix = a["baseline"], a["ix"]
    row = dict(token=a["meta"]["token"], session_id=a["meta"]["session_id"],
        candidate_local=int(ix[chosen]), candidate_id=int(a["sample"]["candidate_ids"][ix[chosen]]),
        teacher_rank=chosen + 1, original_local=int(ix[original]), post_local=int(ix[post]),
        replacement_after_veto=chosen != original, raw_baseline_score=float(a["raw"][b, 0]),
        raw_score=float(a["raw"][chosen, 0]), raw_gain_pp=float(a["gain"][chosen]),
        raw_harm=bool(a["harm"][chosen]), pre_vs_post_gain_pp=float(a["gain"][chosen] - a["gain"][post]),
        predicted_gain_pp=100 * float(pred["gain"][position, chosen]),
        gain_lower_pp=100 * float(pred["gain_lower"][position, chosen]),
        predicted_risk_max=float(pred["risk"][position, chosen].max()),
        allowed_oracle_gain_pp=float(a["gain"][~a["blocked"].any(axis=1)].max()))
    row["gain_overestimate_pp"] = row["predicted_gain_pp"] - row["raw_gain_pp"]
    row["allowed_oracle_regret_pp"] = row["allowed_oracle_gain_pp"] - row["raw_gain_pp"]
    for k, m in enumerate(METRICS):
        row["teacher_baseline_" + m] = float(a["teacher"][b, k])
        row["teacher_candidate_" + m] = float(a["teacher"][chosen, k])
        row["teacher_delta_" + m] = float(a["teacher"][chosen, k] - a["teacher"][b, k])
        for version in ("raw", "filtered"):
            labels = a[version]
            row[version + "_baseline_" + m] = float(labels[b, k + 1])
            row[version + "_candidate_" + m] = float(labels[chosen, k + 1])
            row[version + "_delta_" + m] = float(labels[chosen, k + 1] - labels[b, k + 1])
    drops = [m for k, m in enumerate(METRICS) if a["delta"][chosen, k] < -MEASURED_EPS]
    row["raw_dropped_components"] = "+".join(drops) or "none"
    row["loss_without_component_drop"] = bool(a["gain"][chosen] < -1e-6 and not drops)
    row["raw_filtered_components_differ"] = bool(np.any(np.abs(a["raw"][[b, chosen], 1:] - a["filtered"][[b, chosen], 1:]) > MEASURED_EPS))
    return row


def analyze(examples, prediction, positions, night_identity, saved, frozen, saved_opportunity):
    tokens = {e["metadata"]["token"] for e in examples}
    if len(tokens) != len(examples) or set(positions) != tokens:
        raise ValueError("Duplicate/missing prediction or bundle tokens")
    decisions = {(r["method"], r["token"]): r for r in frozen}
    if len(decisions) != len(frozen):
        raise ValueError("Duplicate frozen decisions")
    frame = saved.set_index(["method", "token"])
    if frame.index.has_duplicates:
        raise ValueError("Duplicate saved rows")
    candidates, opportunity, interventions, transitions = [], [], [], []
    for e in examples:
        a = scene_arrays(e)
        token, b, i = a["meta"]["token"], a["baseline"], positions[a["meta"]["token"]]
        choices, _, _ = veto.decisions(a["sample"], a["ix"], b, prediction, i,
            night_identity["policies"], night_identity["config"]["minimum_support_sessions"])
        choices = choices[HOST]
        for mode, chosen in choices.items():
            key = (PREFIX + mode, token)
            decision = decisions[key]
            if (decision["chosen_local"] != int(a["ix"][chosen])
                    or decision["original_local"] != int(a["ix"][choices["original"]])):
                raise ValueError("Frozen decision replay failed")
            delta = a["teacher"][chosen, SAFETY] - a["teacher"][b, SAFETY]
            if not np.allclose(delta, decision["teacher_compliance_deltas"], atol=1e-12, rtol=0):
                raise ValueError("Frozen teacher delta replay failed")
            check_record(frame.loc[key], a, chosen)
        pre, post = choices["all5_pre"], choices["all5_post"]
        transitions.append(dict(token=token, session_id=a["meta"]["session_id"], changed=pre != post,
            pre_intervention=pre != b, post_intervention=post != b,
            pre_minus_post_gain_pp=float(a["gain"][pre] - a["gain"][post])))
        if pre != b:
            interventions.append(intervention(a, pre, choices["original"], post, prediction, i))
        candidates.extend(constraint_rows(a))
        opportunity.extend(opportunities(a))
    opportunities_frame = pd.DataFrame(opportunity)
    previous = saved_opportunity.set_index(["scope", "token"])
    if previous.index.has_duplicates:
        raise ValueError("Duplicate saved opportunity rows")
    for scope, old_scope in (("unrestricted", "unrestricted"), ("all5", "all5"), (SAFETY_NAMES[1], "dac")):
        for row in opportunities_frame[opportunities_frame.scope == scope].itertuples():
            old = previous.loc[(old_scope, row.token)]
            if (abs(old.oracle_gain_pp - row.oracle_gain_pp) > 1e-8
                    or int(old.useful_alternatives) != row.useful_alternatives
                    or int(old.allowed_alternatives) != row.allowed_alternatives):
                raise ValueError("Frozen opportunity replay failed")
    intervention_frame = pd.DataFrame(interventions) if interventions else pd.DataFrame(columns=[
        "token", "session_id", "candidate_local", "candidate_id", "raw_gain_pp", "predicted_gain_pp",
        "gain_lower_pp", "raw_harm", "replacement_after_veto", "raw_dropped_components"])
    return pd.DataFrame(candidates), opportunities_frame, intervention_frame, pd.DataFrame(transitions)


def aggregate(candidates, opportunity, interventions, transitions):
    reserves = {}
    for scope, g in opportunity.groupby("scope", sort=True):
        reserves[scope] = dict(oracle_gain_micro_pp=float(g.oracle_gain_pp.mean()),
            oracle_gain_macro_pp=float(g.groupby("session_id").oracle_gain_pp.mean().mean()),
            relative_safe_oracle_gain_micro_pp=float(g.relative_safe_oracle_gain_pp.mean()),
            scenes_with_useful_alternative=int((g.useful_alternatives > 0).sum()))
    full = opportunity[opportunity.scope == "all5"].set_index("token")
    constraints = {}
    for m in SAFETY_NAMES:
        blocked = candidates["blocked_" + m]
        degraded = candidates["raw_delta_" + m] < -MEASURED_EPS
        exclusive = blocked & (candidates.veto_count == 1)
        without = opportunity[opportunity.scope == "without_" + m].set_index("token").loc[full.index]
        recovered = (full.useful_alternatives == 0) & (without.useful_alternatives > 0)
        constraints[m] = dict(blocked_actions=int(blocked.sum()),
            component_degradation_actions=int(degraded.sum()),
            detected_component_degradations=int((blocked & degraded).sum()),
            missed_component_degradations=int((~blocked & degraded).sum()),
            blocked_any_safety_degraded_actions=int((blocked & candidates.raw_harm).sum()),
            blocked_useful_actions=int((blocked & candidates.useful).sum()),
            blocked_useful_scenes=int(candidates.loc[blocked & candidates.useful, "token"].nunique()),
            exclusively_blocked_useful_actions=int((exclusive & candidates.useful).sum()),
            recovered_useful_scenes_if_only_this_constraint_removed=int(recovered.sum()),
            recovered_sessions=int(without.loc[recovered, "session_id"].nunique()))
    patterns = [dict(pattern=pattern, actions=len(g), useful_actions=int(g.useful.sum()),
        safety_degraded_actions=int(g.raw_harm.sum()), scenes=int(g.token.nunique()), sessions=int(g.session_id.nunique()))
        for pattern, g in candidates.groupby("veto_pattern", sort=True)]
    if sum(r["actions"] for r in patterns) != len(candidates):
        raise RuntimeError("Exact intersection partition failed")
    losses = interventions[interventions.raw_gain_pp < -1e-6] if not interventions.empty else interventions
    compact_losses = []
    for row in losses.sort_values("raw_gain_pp").to_dict("records") if not losses.empty else []:
        detail = {k: row[k] for k in ("token", "session_id", "candidate_id", "raw_gain_pp",
            "predicted_gain_pp", "gain_lower_pp", "raw_harm", "replacement_after_veto", "raw_dropped_components",
            "loss_without_component_drop", "raw_filtered_components_differ")}
        detail["raw_component_deltas"] = {m: row["raw_delta_" + m] for m in METRICS
                                           if abs(row["raw_delta_" + m]) > MEASURED_EPS}
        detail["teacher_safety_deltas"] = {m: row["teacher_delta_" + m] for m in SAFETY_NAMES}
        compact_losses.append(detail)
    component_drops = {m: int((losses["raw_delta_" + m] < -MEASURED_EPS).sum()) if not losses.empty else 0 for m in METRICS}
    changed = transitions[transitions.changed]
    primary = dict(interventions=len(interventions), losses=len(losses),
        replacement_after_veto=int(interventions.replacement_after_veto.sum()) if not interventions.empty else 0,
        mean_gain_overestimate_pp=float(interventions.gain_overestimate_pp.mean()) if not interventions.empty else None,
        losses_without_component_drop=int(losses.loss_without_component_drop.sum()) if not losses.empty else 0,
        raw_filtered_component_disagreements=int(interventions.raw_filtered_components_differ.sum()) if not interventions.empty else 0,
        loss_scenes_with_component_drop=component_drops, losses_detail=compact_losses)
    comparison = dict(changed_scenes=len(changed),
        worsened_scenes=int((changed.pre_minus_post_gain_pp < -1e-6).sum()),
        improved_scenes=int((changed.pre_minus_post_gain_pp > 1e-6).sum()),
        gain_micro_pp=float(transitions.pre_minus_post_gain_pp.mean()),
        gain_macro_pp=float(transitions.groupby("session_id").pre_minus_post_gain_pp.mean().mean()))
    return dict(primary_interventions=primary, pre_vs_post=comparison, constraints=constraints,
                exact_veto_patterns=patterns, oracle_opportunity=reserves)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory"
    p.add_argument("--veto", type=Path, default=base / "experiments/teacher_veto_v1")
    p.add_argument("--night", type=Path, default=base / "experiments/residual_critic_night_v1")
    p.add_argument("--bundle", type=Path, default=base / "selector_training_bundle_v3_30gb")
    p.add_argument("--out", type=Path, default=base / "experiments/teacher_veto_audit_v1")
    args = p.parse_args()
    root, night, bundle, out = (x.resolve() for x in (args.veto, args.night, args.bundle, args.out))
    for source in (root, night, bundle):
        if out == source or source in out.parents or out in source.parents:
            raise ValueError("Separate audit output required")
    night_identity, _ = veto.verify_night(night, bundle)
    _, previous = verify_veto(root, night, bundle)
    inputs = {str(path): veto.sha(path) for path in (root / "report.json", night / "report.json", bundle / "bundle_manifest.json")}
    identity = dict(inputs=inputs, source_sha256=veto.sha(Path(__file__)), veto_source_sha256=veto.sha(Path(veto.__file__)),
        analyzed_method=veto.CANDIDATE, epsilon=veto.EPSILON, measured_epsilon=MEASURED_EPS,
        improvement_min=veto.bundles.IMPROVEMENT_MIN, numpy=np.__version__, pandas=pd.__version__, note=NOTE)
    marker = out / "identity.json"
    if marker.exists() and veto.read(marker) != identity:
        raise RuntimeError("Audit code/inputs changed; use a new --out root")
    if out.exists() and not marker.exists() and any(out.iterdir()):
        raise ValueError("Nonempty audit output without identity")
    examples, _ = veto.bundles.load_bundle(bundle)
    tokens = {e["metadata"]["token"] for e in examples}
    if previous["scenes"] != len(examples) or previous["sessions"] != len({e["metadata"]["session_id"] for e in examples}):
        raise ValueError("Veto/bundle scene count mismatch")
    if (out / "report.json").exists():
        report = veto.read(out / "report.json")
        if report["status"] != "complete" or report["identity_sha256"] != veto.sha(marker):
            raise RuntimeError("Audit identity changed")
        for name, digest in report["artifacts"].items():
            if veto.sha(out / name) != digest:
                raise RuntimeError("Audit artifact changed: " + name)
        print(json.dumps(report, indent=2), flush=True)
        return
    with np.load(night / "mlp_oof_ensemble.npz", allow_pickle=False) as data:
        pred = {k: data[k].copy() for k in ("gain", "gain_lower", "risk", "risk_support_sessions")}
        positions = {str(t): i for i, t in enumerate(data["tokens"])}
        if len(data["tokens"]) != len(tokens) or set(positions) != tokens:
            raise ValueError("OOF token coverage mismatch")
    saved, opportunity = read_csv(root / "per_scene.csv"), read_csv(root / "opportunity.csv")
    frozen = veto.read(root / "decisions.json")
    methods = {f"{v}/{h}/{m}" for v in night_identity["variants"] for h in veto.HOSTS for m in veto.MODES}
    expected = {(m, t) for m in methods for t in tokens}
    if (len(frozen) != len(expected) or {(r["method"], r["token"]) for r in frozen} != expected
            or len(saved) != len(expected) + len(tokens)
            or set(zip(saved.method, saved.token)) != expected | {("baseline", t) for t in tokens}):
        raise ValueError("Frozen veto decision coverage mismatch")
    candidates, opportunities_frame, interventions, transitions = analyze(
        examples, pred, positions, night_identity, saved, frozen, opportunity)
    result = aggregate(candidates, opportunities_frame, interventions, transitions)
    veto.write(marker, identity)
    outputs = {"candidate_diagnostics.csv": candidates, "constraint_opportunity_per_scene.csv": opportunities_frame,
               "primary_interventions.csv": interventions, "pre_vs_post_per_scene.csv": transitions}
    for name, frame in outputs.items():
        temp = out / (name + ".tmp")
        frame.to_csv(temp, index=False)
        temp.replace(out / name)
    if any(veto.sha(Path(path)) != digest for path, digest in inputs.items()):
        raise RuntimeError("Frozen input manifests changed during audit")
    report = dict(status="complete", analyzed_method=veto.CANDIDATE, frozen_primary=previous["primary"],
        scenes=len(examples), sessions=previous["sessions"], candidate_actions=len(candidates), **result,
        checks=dict(frozen_artifacts_verified=True, primary_host_decisions_replayed=True,
                    raw_filtered_outcomes_replayed=True, previous_opportunity_replayed=True,
                    exact_intersections_partition_all_actions=True),
        fresh_holdout_processed=0, new_training_runs=0, new_scorer_calls=0, additional_download_GB=0,
        identity_sha256=veto.sha(marker), artifacts={name: veto.sha(out / name) for name in outputs}, note=NOTE)
    veto.write(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
