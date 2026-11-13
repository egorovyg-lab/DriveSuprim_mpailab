"""Post-hoc attribution of frozen C2-vs-C1 decisions. NumPy/Pandas CPU only.

No fitting, new predictions, policy tuning, scorer calls or holdout access.
Query outcomes are diagnostic evidence only, never a replacement policy.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from navsim.agents import drivemem_residual_memory as model
from navsim.planning.script import run_drivemem_residual_memory as source

NOTE = ("Post-hoc diagnostic of inspected train OOF decisions, not new validation or a new policy. "
        "True query outcomes used only for attribution. Feature distance cannot establish equivalence "
        "of road layout or actor interaction. Local PDM, not EPDMS. No fitting, scorer or Drex calls.")


def relation(required, correction):
    if abs(required) <= 1e-8:
        return "no_residual_gap_needed"
    if abs(correction) <= 1e-8:
        return "no_memory_gap"
    if required * correction < 0:
        return "wrong_direction"
    if abs(required - correction) < abs(required) - 1e-8:
        return "gap_error_reduced"
    return "same_direction_overshoot"


def key_vector(entry, p, mean, scale):
    parts = []
    for name in model.BLOCKS:
        a, b = entry["blocks"][name]
        parts.append(np.clip((entry["x"][p, a:b].astype(float) - mean[a:b]) / scale[a:b], -8, 8)
                     / np.sqrt(4 * (b-a)))
    return parts


def donor_action(query, p, qi, fold, entries, raws):
    """Replay the stored weighted correction, retain every donor contribution."""
    b = query["baseline"]
    idx = fold["traces"]["nearest_index"][qi, p]
    dist = fold["traces"]["nearest_distance"][qi, p]
    weight = fold["traces"]["nearest_weight"][qi, p]
    if (idx.dtype.kind not in "iu" or len(idx) != len(set(idx.tolist())) or
            np.any(idx < 0) or np.any(idx >= len(fold["records"])) or
            not np.isfinite(dist).all() or np.any(dist < 0) or not np.isfinite(weight).all()):
        raise ValueError("Invalid stored donor trace")
    if p == b:
        if np.any(weight != 0):
            raise ValueError("Baseline has nonzero donor weights")
        return dict(correction=0., unshrunk=0., alpha=0., weight_sum=0., effective_donors=0., donors=[])
    expected = np.exp(-dist / fold["h"])
    if not np.allclose(weight, expected, atol=1e-12, rtol=1e-10):
        raise ValueError("Stored donor weights do not replay")
    mass = float(weight.sum())
    values = fold["residuals"][idx]
    correction = float(weight @ values / (1 + mass))
    qz = key_vector(query, p, fold["mean"], fold["scale"])
    used_sessions, rows = set(), []
    for index, distance, w, residual in zip(idx, dist, weight, values):
        record = fold["records"][int(index)]
        donor = entries[record["token"]]
        dp = int(record["position"])
        sid = record["session_id"]
        if (sid == query["session_id"] or sid in used_sessions or sid not in fold["training_sessions"]
                or donor["session_id"] != sid or dp == donor["baseline"] or not 0 <= dp < 64
                or int(donor["ids"][dp]) != record["candidate_id"]):
            raise ValueError("Donor session/physical action alignment failed")
        used_sessions.add(sid)
        provenance = fold["sources"][donor["token"]]
        if sid in provenance["training_sessions"] or set(provenance["training_sessions"]) - set(fold["training_sessions"]):
            raise ValueError("Residual cross-fit session leakage")
        dz = key_vector(donor, dp, fold["mean"], fold["scale"])
        blocks = {name: float(np.square(a-c).sum()) for name, a, c in zip(model.BLOCKS, qz, dz)}
        if not np.isclose(sum(blocks.values()), distance, atol=1e-7, rtol=1e-6):
            raise ValueError("Stored donor distance does not replay from input features")
        raw = raws[donor["token"]]
        delta = raw[dp] - raw[donor["baseline"]]
        row = dict(token=donor["token"], session_id=sid, candidate_id=int(donor["ids"][dp]), ds_rank=dp+1,
                   distance=float(distance), distance_blocks=blocks, weight=float(w), normalized_weight=float(w/(1+mass)),
                   residual_pp=float(100*residual), contribution_pp=float(100*w*residual/(1+mass)),
                   actual_gain_pp=float(100*delta[0]),
                   donor_oof_predicted_gain_pp=float(100*(float(np.float32(delta[0]))-residual)),
                   donor_harm=bool(np.any(delta[1:][list(model.SAFETY)] < -1e-8)),
                   donor_baseline_pdm=float(raw[donor["baseline"], 0]), donor_action_pdm=float(raw[dp, 0]),
                   residual_source=provenance["fit"],
                   component_deltas={m: float(delta[j+1]) for j, m in enumerate(model.METRICS)})
        rows.append(row)
    return dict(correction=correction, unshrunk=float(weight @ values/mass) if mass else 0.,
                alpha=mass/(1+mass), weight_sum=mass,
                effective_donors=float(mass**2 / (float(weight @ weight)+1e-30)), donors=rows)


def audit(root, out):
    root, out = root.resolve(), out.resolve()
    if root == out or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("Audit output must be separate from the frozen source tree")
    identity = source.read(root/"identity.json")
    if identity["protocol"] != source.PROTOCOL or identity["sources"] != source.source_snapshot():
        raise ValueError("Original experiment source snapshot differs; preserve the original code")
    config = model.Config(**identity["config"])
    if source.read(root/"split.json")["groups"] != identity["groups"]:
        raise ValueError("Session split changed")
    entries_list = source.prepared(root)
    decisions = source.decision_rows(root, entries_list, identity["groups"], config)
    report_commit = source.read(root/"report_commit.json")
    spec = dict(identity_sha256=source.sha(root/"identity.json"), data_sha256=source.sha(root/"data/manifest.json"),
                folds={str(i): source.sha(root/"folds"/str(i)/"commit.json") for i in range(4)})
    if report_commit["spec"] != spec:
        raise ValueError("Completed source report binding mismatch")
    source.files_checked(root, report_commit)
    original = source.read(root/"summary.json")
    if original["status"] != "complete":
        raise ValueError("Completed experiment required")
    audit_id = dict(protocol="residual_memory_attribution_v1", source=str(root),
                    report_commit_sha256=source.sha(root/"report_commit.json"),
                    source_identity_sha256=source.sha(root/"identity.json"),
                    audit_code_sha256=source.sha(Path(__file__)), selection=dict(top_losses=10, top_benefits=5), note=NOTE)
    with source.locked(out):
        source.freeze(out/"identity.json", audit_id)
        if (out/"commit.json").exists():
            commit = source.read(out/"commit.json")
            if commit["identity_sha256"] != source.sha(out/"identity.json"):
                raise ValueError("Audit identity changed")
            source.files_checked(out, commit)
            return source.read(out/"summary.json")
        # Entire source decision/report commitment verified before outcome diagnosis.
        entries = {e["token"]: e for e in entries_list}
        raws = source.labels_for(root, entries_list)
        folds = {}
        for i in range(4):
            folder = root/"folds"/str(i)
            donors = source.read(folder/"donors.json")
            r = source.load_npz(folder/"residuals.npz")
            commit = source.read(folder/"commit.json")
            if set(donors["training_tokens"]) != {t for t, e in entries.items() if e["session_id"] in commit["training_sessions"]}:
                raise ValueError("Bank is not the complete permitted training cohort")
            folds[i] = dict(records=donors["records"], sources=donors["residual_sources"], h=donors["h"],
                            residuals=r["values"], mean=r["mean"], scale=r["scale"],
                            traces=source.load_npz(folder/"donor_traces.npz"),
                            training_sessions=commit["training_sessions"],
                            token_index={t: j for j, t in enumerate(commit["tokens"])})
        rows, actions = [], {}
        for e in entries_list:
            token, b = e["token"], e["baseline"]
            outer, positions, gains = decisions[token]
            a, c = int(positions[1]), int(positions[2])
            raw = raws[token]
            qa, qc, qb = raw[a], raw[c], raw[b]
            ca, cc = qa[1:]-qb[1:], qc[1:]-qb[1:]
            harm_a = bool(np.any(ca[list(model.SAFETY)] < -1e-8))
            harm_c = bool(np.any(cc[list(model.SAFETY)] < -1e-8))
            actual_gap = float(qc[0]-qa[0])
            base_gap = float(gains[1, c]-gains[1, a])
            required = actual_gap-base_gap
            result = dict(token=token, session_id=e["session_id"], outer=outer, changed=a != c,
                          c1_rank=a+1, c2_rank=c+1, c1_id=int(e["ids"][a]), c2_id=int(e["ids"][c]),
                          c1_gain_pp=float(100*(qa[0]-qb[0])), c2_gain_pp=float(100*(qc[0]-qb[0])),
                          delta_vs_C1_pp=100*actual_gap, c1_harm=harm_a, c2_harm=harm_c,
                          new_safety_harm=harm_c and not harm_a, removed_safety_harm=harm_a and not harm_c,
                          safety_worse_than_C1=bool(np.any((qc[1:]-qa[1:])[list(model.SAFETY)] < -1e-8)),
                          c1_model_gap_pp=100*base_gap, required_residual_gap_pp=100*required)
            if a != c:
                detail = {}
                for name, p in (("C1", a), ("C2", c)):
                    replay = donor_action(e, p, folds[outer]["token_index"][token], folds[outer], entries, raws)
                    m = float(gains[2, p]-gains[1, p])
                    if not np.isclose(m, replay["correction"], atol=1e-10, rtol=1e-9):
                        raise ValueError("Stored C2 correction does not replay")
                    detail[name] = dict(position=p, candidate_id=int(e["ids"][p]), ds_rank=p+1,
                                        raw_metrics=dict(zip(model.LABEL_KEYS, raw[p].tolist())),
                                        c1_predicted_gain_pp=float(100*gains[1, p]),
                                        c2_predicted_gain_pp=float(100*gains[2, p]),
                                        required_action_residual_pp=float(100*(raw[p, 0]-qb[0]-gains[1, p])),
                                        **replay)
                gap = detail["C2"]["correction"]-detail["C1"]["correction"]
                unshrunk_gap = detail["C2"]["unshrunk"]-detail["C1"]["unshrunk"]
                result.update(memory_gap_pp=100*gap, unshrunk_gap_diagnostic_pp=100*unshrunk_gap,
                              gap_error_before_pp=100*abs(required), gap_error_after_pp=100*abs(required-gap),
                              residual_relation=relation(required, gap),
                              boosts_C2=detail["C2"]["correction"] > 1e-8,
                              suppresses_C1=detail["C1"]["correction"] < -1e-8)
                actions[token] = detail
            for j, metric in enumerate(model.METRICS):
                result["delta_vs_C1_"+metric] = float(qc[j+1]-qa[j+1])
            rows.append(result)
        frame = pd.DataFrame(rows)
        switched = frame[frame.changed]
        selected = set(frame.loc[frame.new_safety_harm | frame.safety_worse_than_C1, "token"])
        selected.update(switched[switched.delta_vs_C1_pp < -1e-6].nsmallest(10, "delta_vs_C1_pp").token)
        selected.update(switched[switched.delta_vs_C1_pp > 1e-6].nlargest(5, "delta_vs_C1_pp").token)
        cases = []
        for token in sorted(selected):
            e = entries[token]
            row = next(r for r in rows if r["token"] == token)
            reason = [k for k in ("new_safety_harm", "safety_worse_than_C1") if row[k]]
            reason.append("loss" if row["delta_vs_C1_pp"] < 0 else "benefit")
            case = dict(**row, reasons=reason, baseline_raw=dict(zip(model.LABEL_KEYS, raws[token][e["baseline"]].tolist())),
                        actions=actions[token], note=NOTE)
            source.write(out/"cases"/(token+".json"), case)
            cases.append(dict(token=token, reasons=reason, delta_vs_C1_pp=row["delta_vs_C1_pp"],
                              c1_rank=row["c1_rank"], c2_rank=row["c2_rank"],
                              residual_relation=row["residual_relation"],
                              required_residual_gap_pp=row["required_residual_gap_pp"],
                              memory_gap_pp=row["memory_gap_pp"],
                              c1_correction_pp=100*actions[token]["C1"]["correction"],
                              c2_correction_pp=100*actions[token]["C2"]["correction"],
                              component_changes={m: row["delta_vs_C1_"+m] for m in model.METRICS
                                                 if abs(row["delta_vs_C1_"+m]) > 1e-8}))
        sessions = frame.groupby("session_id").agg(scenes=("token", "size"), changed=("changed", "sum"),
                  delta_vs_C1_pp=("delta_vs_C1_pp", "mean"), new_safety_harm=("new_safety_harm", "sum"),
                  removed_safety_harm=("removed_safety_harm", "sum"))
        delta = float(sessions.delta_vs_C1_pp.mean())
        if (not np.isclose(delta, original["paired"]["C2_minus_C1_selector"]["mean_pp"], atol=1e-8)
                or int(frame.new_safety_harm.sum()) != original["methods"]["C2_memory"]["new_safety_harm_vs_C1"]):
            raise ValueError("Audit does not reproduce the completed report")
        summary = dict(status="complete", scenes=len(frame), changed_scenes=len(switched),
                       delta_vs_C1_macro_pp=delta, new_safety_harm=int(frame.new_safety_harm.sum()),
                       removed_safety_harm=int(frame.removed_safety_harm.sum()),
                       safety_worse_than_C1=int(frame.safety_worse_than_C1.sum()),
                       changed_improvements=int((switched.delta_vs_C1_pp > 1e-6).sum()),
                       changed_losses=int((switched.delta_vs_C1_pp < -1e-6).sum()),
                       changed_equal=int((switched.delta_vs_C1_pp.abs() <= 1e-6).sum()),
                       residual_relations=switched.residual_relation.value_counts().to_dict() if len(switched) else {},
                       losses_residual_relations=switched.loc[switched.delta_vs_C1_pp < -1e-6, "residual_relation"].value_counts().to_dict() if len(switched) else {},
                       cases=cases, new_models=0, new_scorer_calls=0, source_writes=0, note=NOTE)
        frame.to_csv(out/"all_scenes.csv", index=False)
        sessions.to_csv(out/"per_session.csv")
        source.write(out/"summary.json", summary)
        names = ["all_scenes.csv", "per_session.csv", "summary.json"] + ["cases/"+c["token"]+".json" for c in cases]
        source.write(out/"commit.json", dict(identity_sha256=source.sha(out/"identity.json"),
                     files={n: source.sha(out/n) for n in names}))
        return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    result = audit(a.root, a.out or a.root.with_name(a.root.name+"_diagnostic_v1"))
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    print("CASES (one line per diagnostic example):")
    for case in result["cases"]:
        print(json.dumps(case))
