"""Compare rejection rules at equal coverage on frozen memory proposals."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from navsim.agents import drivemem_residual_memory as model
from navsim.planning.script import run_drivemem_residual_memory as source

LEVELS = (.10, .25, .50, .75, 1.)
NOTE = ("Retrospective matched intervention counts on fixed proposals from inspected train sessions. "
        "Cutoffs describe this cohort. Oracle sees outcomes. Intervals resample sessions and hold decisions fixed.")


def proposals(root, entries, decisions, raws):
    rows = []
    for outer in range(4):
        folder = root/"folds"/str(outer)
        tokens = source.read(folder/"commit.json")["tokens"]
        bandwidth = source.read(folder/"donors.json")["h"]
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("Invalid frozen retrieval bandwidth")
        with np.load(folder/"donor_traces.npz", allow_pickle=False) as traces:
            nearest = traces["nearest_distance"].min(axis=2) / bandwidth
        if nearest.shape != (len(tokens), 64) or not np.isfinite(nearest).all() or np.any(nearest < 0):
            raise ValueError("Invalid frozen distances")
        for i, token in enumerate(tokens):
            e = entries[token]
            fold, chosen, gains = decisions[token]
            if fold != outer:
                raise ValueError("Decision fold mismatch")
            b, candidate = e["baseline"], int(chosen[2])
            for reference, control in (("DS", b), ("C1", int(chosen[1]))):
                if candidate == control:
                    continue
                # A return to DS has no donor of its own. In that case use
                # the support for the alternative that memory suppresses.
                positions = [p for p in (candidate, control) if p != b]
                raw = raws[token]
                delta = raw[candidate] - raw[control]
                row = dict(reference=reference, token=token, session_id=e["session_id"], outer=outer,
                           reference_rank=control+1, proposal_rank=candidate+1,
                           reference_id=int(e["ids"][control]), proposal_id=int(e["ids"][candidate]),
                           sampling_slice=e["sampling_slice"], distance=float(nearest[i, positions].max()),
                           expected_gain_pp=float(100 * (gains[2, candidate] - gains[2, control])),
                           gain_pp=float(100 * delta[0]), harm=bool(np.any(delta[1:][list(model.SAFETY)] < -1e-8)),
                           pdm_loss=bool(delta[0] < -1e-8))
                row.update({"delta_" + name: float(delta[j+1]) for j, name in enumerate(model.METRICS)})
                rows.append(row)
    columns = ("reference", "token", "session_id", "outer", "reference_rank", "proposal_rank",
               "reference_id", "proposal_id", "sampling_slice", "distance", "expected_gain_pp",
               "gain_pp", "harm", "pdm_loss", *("delta_" + n for n in model.METRICS))
    return pd.DataFrame(rows, columns=columns)


def orderings(proposed):
    return {
        "distance": proposed.sort_values(["distance", "token"]),
        "predicted_utility": proposed.sort_values(["expected_gain_pp", "token"], ascending=[False, True]),
        "oracle_safety_first": proposed.sort_values(["harm", "gain_pp", "token"], ascending=[True, False, True]),
    }


def compare(frame, cohort, bootstrap_reps=10000):
    sessions = cohort.groupby("session_id").size().sort_index()
    rng = np.random.default_rng(20261113)
    draws = rng.integers(0, len(sessions), (bootstrap_reps, len(sessions)))
    curves, comparisons, bins = [], [], []
    for reference in ("DS", "C1"):
        proposed = frame[frame.reference == reference]
        counts = sorted({0, *(int(np.ceil(x*len(proposed))) for x in LEVELS)})
        ordered = orderings(proposed)
        for n in counts:
            gains, harms = {}, {}
            for rule, table in ordered.items():
                q = table.iloc[:n]
                gain = q.groupby("session_id").gain_pp.sum().reindex(sessions.index, fill_value=0) / sessions
                harm = q.groupby("session_id").harm.sum().reindex(sessions.index, fill_value=0) / sessions
                gains[rule], harms[rule] = gain.to_numpy(dtype=float), harm.to_numpy(dtype=float)
                curves.append(dict(reference=reference, rule=rule, proposals=len(proposed), interventions=n,
                                   coverage=n/len(cohort), intervening_sessions=int(q.session_id.nunique()),
                                   gain_macro_pp=float(gain.mean()), gain_micro_pp=float(q.gain_pp.sum()/len(cohort)),
                                   safety_harms=int(q.harm.sum()), pdm_losses=int(q.pdm_loss.sum()),
                                   harmful_per_all=float(q.harm.sum()/len(cohort)),
                                   harmful_per_intervention=float(q.harm.mean()) if n else None))
            contrast = gains["predicted_utility"] - gains["distance"]
            harm_contrast = harms["predicted_utility"] - harms["distance"]
            comparisons.append(dict(reference=reference, interventions=n, coverage=n/len(cohort),
                utility_minus_distance_macro_pp=float(contrast.mean()),
                gain_ci95_sessions=np.quantile(contrast[draws].mean(axis=1), [.025, .975]).tolist(),
                utility_minus_distance_harm_rate_macro=float(harm_contrast.mean()),
                harm_rate_ci95_sessions=np.quantile(harm_contrast[draws].mean(axis=1), [.025, .975]).tolist()))
        bands = pd.cut(proposed.distance, [-np.inf, .5, 1., 2., np.inf],
                       labels=["<=0.5", "0.5–1", "1–2", ">2"])
        # Split descriptively by rank, data provenance and donor distance.
        slices = {"all": proposed, "DS_top16": proposed[proposed.proposal_rank <= 16],
                  "DS_17_64": proposed[proposed.proposal_rank > 16]}
        slices.update({str(k): sub for k, sub in proposed.groupby("sampling_slice")})
        for name, sub in slices.items():
            for band, rows in sub.groupby(bands, observed=True):
                bins.append(dict(reference=reference, slice=name, distance_over_h=str(band),
                                 proposals=len(rows), sessions=int(rows.session_id.nunique()),
                                 gain_macro_pp=float(rows.groupby("session_id").gain_pp.mean().mean()),
                                 gain_mean_pp=float(rows.gain_pp.mean()), safety_harm_rate=float(rows.harm.mean()),
                                 pdm_loss_rate=float(rows.pdm_loss.mean())))
    return curves, comparisons, bins


def run(root, out):
    root, out = root.resolve(), out.resolve()
    if out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("Use an output separate from the frozen experiment")
    identity = source.read(root/"identity.json")
    if identity["protocol"] != source.PROTOCOL or identity["sources"] != source.source_snapshot():
        raise ValueError("Frozen source snapshot differs")
    if source.read(root/"split.json")["groups"] != identity["groups"]:
        raise ValueError("Frozen session split differs")
    entries = source.prepared(root)
    decisions = source.decision_rows(root, entries, identity["groups"], model.Config(**identity["config"]))
    report = source.read(root/"report_commit.json")
    spec = dict(identity_sha256=source.sha(root/"identity.json"), data_sha256=source.sha(root/"data/manifest.json"),
                folds={str(i): source.sha(root/"folds"/str(i)/"commit.json") for i in range(4)})
    if report["spec"] != spec:
        raise ValueError("Frozen report does not match decisions")
    source.files_checked(root, report)
    with source.locked(out):
        if not (out/"identity.json").exists() and any(p.name != ".run.lock" for p in out.iterdir()):
            raise ValueError("Nonempty output without audit identity")
        source.freeze(out/"identity.json", dict(protocol="matched_risk_coverage_v1", parent=str(root),
                      report_commit_sha256=source.sha(root/"report_commit.json"), code_sha256=source.sha(__file__)))
        if (out/"commit.json").exists():
            source.files_checked(out, source.read(out/"commit.json"))
            return source.read(out/"summary.json")
        frame = proposals(root, {e["token"]: e for e in entries}, decisions, source.labels_for(root, entries))
        cohort = pd.DataFrame([{k: e[k] for k in ("token", "session_id")} for e in entries])
        curves, comparisons, bins = compare(frame, cohort)
        original = source.read(root/"summary.json")
        memory = original["methods"]["C2_memory"]
        for reference, expected_pp, expected_harm in (
            ("DS", memory["mean_pp"], memory["safety_harm_scenes"]),
            ("C1", original["paired"]["C2_minus_C1_selector"]["mean_pp"], memory["safety_worse_than_C1_choice"]),
        ):
            final = next(r for r in curves if r["reference"] == reference and r["rule"] == "distance"
                         and r["interventions"] == r["proposals"])
            if not np.isclose(final["gain_macro_pp"], expected_pp, atol=1e-8) or final["safety_harms"] != expected_harm:
                raise ValueError("Full coverage does not reproduce frozen result")
        if int((frame.reference == "DS").sum()) != memory["interventions"]:
            raise ValueError("Proposal count against DS differs from frozen report")
        frame.to_csv(out/"proposals.csv", index=False)
        pd.DataFrame(curves).to_csv(out/"risk_coverage.csv", index=False)
        pd.DataFrame(bins).to_csv(out/"distance_bins.csv", index=False)
        summary = dict(status="complete", scenes=len(entries), sessions=int(cohort.session_id.nunique()),
                       proposed={ref: int((frame.reference == ref).sum()) for ref in ("DS", "C1")},
                       paired_utility_vs_distance=comparisons, curves=curves, distance_bins=bins, note=NOTE)
        source.write(out/"summary.json", summary)
        names = ("identity.json", "proposals.csv", "risk_coverage.csv", "distance_bins.csv", "summary.json")
        source.write(out/"commit.json", dict(files={n: source.sha(out/n) for n in names}))
        return summary


if __name__ == "__main__":
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory/experiments"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=base/"residual_memory_top64_v1")
    parser.add_argument("--out", type=Path, default=base/"drivemem_matched_coverage_v1")
    args = parser.parse_args()
    summary = run(args.root, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("curves", "distance_bins")}, indent=2, allow_nan=False))
