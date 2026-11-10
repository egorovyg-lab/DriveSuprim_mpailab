"""Teacher-relative veto ablation on frozen training OOF predictions.

Selection functions consume teacher inputs and cached predictions only. Outcomes
are used after decisions are frozen. Existing experiments are read-only.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.agents import selector_training_bundle as bundles
from navsim.agents.drivemem import METRICS, SAFETY
from navsim.planning.script.run_drivemem_action_transfer import record, summary
from navsim.planning.script.run_drivemem_expanded_utility import REPO, read, write, sha

EPSILON = 1e-6  # Numerical tolerance, not tuned on the two failures.
HOSTS = ("gain_m005", "gain_m020", "risk_t05_m005", "risk_t05_m020")
MODES = ("original", "dac_post", "all5_post", "all5_pre")
CANDIDATE = "mlp/gain_m020/all5_pre"  # Fixed before running this ablation.
NOTE = ("Training OOF diagnostic motivated by observed failures; not independent confirmation. "
        "Teacher-relative compliance estimates are not guarantees of actual safety. DAC-only is a "
        "diagnostic; the fixed candidate is all-five prefilter with MLP gain_m020. No threshold grid "
        "or best-method selection. Local PDM, not EPDMS. Fresh holdout remains unopened.")


def teacher_inputs(sample, indices):
    logits = np.column_stack([sample["head_" + METRICS[j]][indices] for j in SAFETY]).astype(float)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite teacher compliance logits")
    return 1 / (1 + np.exp(-np.clip(logits, -50, 50)))


def allowed(probabilities, baseline, scope):
    axes = [1] if scope == "dac" else list(range(5))
    mask = np.all(probabilities[:, axes] + EPSILON >= probabilities[baseline, axes], axis=1)
    mask[baseline] = True
    return mask


def select(baseline, ids, gain, lower, risk, support, policy, minimum_support, veto=None):
    """Exact NumPy replay of the frozen critic's inference policy."""
    if not all(np.isfinite(a).all() for a in (gain, lower, risk)) or risk.shape != (len(gain), 5):
        raise ValueError("Invalid cached policy prediction")
    if gain[baseline] != 0 or lower[baseline] != 0:
        raise ValueError("Nonzero baseline score")
    eligible = (lower if policy["uncertainty"] else gain) >= policy["margin"]
    if policy["risk_limit"] is not None:
        eligible &= (risk <= policy["risk_limit"]).all(-1)
        if np.any(support < minimum_support):
            eligible[:] = False
    if veto is not None:
        eligible &= veto
    eligible[baseline] = True
    score = np.where(eligible, gain, -np.inf)
    tied = np.flatnonzero(score >= score.max() - 1e-7)
    return baseline if baseline in tied else int(tied[np.argmin(ids[tied])])


def decisions(sample, indices, baseline, prediction, row, policies, minimum_support):
    """No measured labels or outcome dictionaries are accepted here."""
    ids = sample["candidate_ids"][indices]
    teacher = teacher_inputs(sample, indices)
    dac, all5 = allowed(teacher, baseline, "dac"), allowed(teacher, baseline, "all5")
    args = (baseline, ids, prediction["gain"][row], prediction["gain_lower"][row],
            prediction["risk"][row], prediction["risk_support_sessions"][row])
    choices = {}
    for host in HOSTS:
        policy = policies[host]
        original = select(*args, policy, minimum_support)
        choices[host] = dict(original=original,
            dac_post=original if dac[original] else baseline,
            all5_post=original if all5[original] else baseline,
            all5_pre=select(*args, policy, minimum_support, veto=all5))
    return choices, teacher, dict(dac=dac, all5=all5)


def verify_night(root, bundle):
    identity, report = read(root / "identity.json"), read(root / "report.json")
    if (report["status"] != "complete" or report["fresh_holdout_processed"] != 0
            or report["identity_sha256"] != sha(root / "identity.json")):
        raise ValueError("Completed frozen training-only night study required")
    for name, digest in report["artifacts"].items():
        if sha(root / name) != digest:
            raise RuntimeError("Frozen night artifact changed: " + name)
    for name, digest in identity["sources"].items():
        if sha(REPO / name) != digest:
            raise RuntimeError("Frozen night source changed: " + name)
    for name in ("identity.json", "bundle_manifest.json"):
        if identity["inputs"].get(str(bundle / name)) != sha(bundle / name):
            raise ValueError("Night study belongs to another bundle")
    return identity, report


def eligible(rows, criteria):
    frame = pd.DataFrame(rows)
    uniform = frame[frame["slice"] == "uniform"]
    gain = float(uniform.groupby("session_id").raw_gain_pp.mean().mean()) if not uniform.empty else 0.
    active = frame[frame.intervention]
    improved = int(frame.loc[frame.raw_gain_pp >= .5, "session_id"].nunique())
    return bool(gain > 1e-8 and not frame.raw_harm.any()
        and frame.raw_gain_pp.min() >= -criteria["maximum_single_quality_loss_pp"]
        and len(active) >= criteria["minimum_interventions"]
        and active.session_id.nunique() >= criteria["minimum_intervention_sessions"]
        and improved >= criteria["minimum_improved_sessions"])


def evaluate(examples, snapshots, night_identity, night_frame, out):
    # Freeze inference decisions for ALL scenes before looking at measured outcomes.
    frozen, contexts = [], {}
    replay = night_frame.set_index(["method", "token"])
    if replay.index.has_duplicates:
        raise ValueError("Duplicate cached OOF decisions")
    for example in examples:
        sample, meta, ix = example["sample"], example["metadata"], example["label_indices"]
        token = meta["token"]
        if meta["role"] != "train":
            raise ValueError("Training scenes only")
        if not np.array_equal(ix, np.argsort(-sample["selection_scores"], kind="stable")[:16]):
            raise ValueError("Frozen top16 alignment mismatch")
        baseline = int(np.flatnonzero(ix == int(sample["selected_local"]))[0])
        for variant, (prediction, positions) in snapshots.items():
            i = positions[token]
            choices, teacher, masks = decisions(sample, ix, baseline, prediction, i,
                night_identity["policies"], night_identity["config"]["minimum_support_sessions"])
            contexts[token] = (teacher, masks, baseline)
            for host, modes in choices.items():
                original = modes["original"]
                previous = replay.loc[(variant + "/ensemble/" + host, token)]
                if int(previous.candidate_local) != int(ix[original]) or int(previous.candidate_id) != int(sample["candidate_ids"][ix[original]]):
                    raise ValueError("Exact replay of frozen critic failed")
                for mode, chosen in modes.items():
                    frozen.append(dict(method=f"{variant}/{host}/{mode}", token=token, chosen_local=int(ix[chosen]),
                        original_local=int(ix[original]), vetoed_original=bool(mode != "original" and chosen != original),
                        teacher_compliance_deltas=(teacher[chosen] - teacher[baseline]).tolist()))
    marker = out / "decisions.json"
    if marker.exists() and read(marker) != frozen:
        raise RuntimeError("Frozen teacher-veto decisions changed")
    write(marker, frozen)
    lookup = {e["metadata"]["token"]: e for e in examples}
    rows, opportunity, measured = [], [], {}
    def outcomes_for(token):
        if token not in measured:
            e = lookup[token]
            measured[token] = {str(int(local)): {v: dict(zip(bundles.LABEL_KEYS, e["label_" + v][j].astype(float).tolist()))
                              for v in ("raw", "filtered")} for j, local in enumerate(e["label_indices"])}
        return measured[token]
    for d in frozen:
        e = lookup[d["token"]]
        sample = dict(e["sample"], token=d["token"], session_id=e["metadata"]["session_id"])
        outcomes = outcomes_for(d["token"])
        row = {**d, "slice": e["sampling_slice"], "origin": e["metadata"]["origin"],
               "oracle_score": float(e["label_raw"][:, 0].max()), **record(sample, outcomes, d["chosen_local"])}
        if d["method"].endswith("/original"):
            variant, host, _ = d["method"].split("/")
            previous = replay.loc[(variant + "/ensemble/" + host, d["token"])]
            if abs(previous.raw_gain_pp - row["raw_gain_pp"]) > 1e-8 or bool(previous.raw_harm) != row["raw_harm"]:
                raise ValueError("Frozen measured outcome replay failed")
        rows.append(row)
    for token, e in lookup.items():
        teacher, masks, b = contexts[token]
        quality, components = e["label_raw"][:, 0], e["label_raw"][:, 1:][:, list(SAFETY)]
        useful = ((quality - quality[b]) >= .005) & np.all(components + 1e-8 >= components[b], axis=1)
        for scope, mask in dict(unrestricted=np.ones(16, dtype=bool), **masks).items():
            opportunity.append(dict(token=token, session_id=e["metadata"]["session_id"], scope=scope,
                oracle_gain_pp=100 * float(quality[mask].max() - quality[b]),
                useful_alternatives=int((useful & mask).sum()), allowed_alternatives=int(mask.sum()) - 1))
        sample = dict(e["sample"], token=token, session_id=e["metadata"]["session_id"])
        outcomes = outcomes_for(token)
        rows.append(dict(method="baseline", slice=e["sampling_slice"], oracle_score=float(quality.max()),
                         **record(sample, outcomes, int(sample["selected_local"]))))
    return rows, opportunity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory"
    p.add_argument("--night", type=Path, default=base / "experiments/residual_critic_night_v1")
    p.add_argument("--bundle", type=Path, default=base / "selector_training_bundle_v3_30gb")
    p.add_argument("--out", type=Path, default=base / "experiments/teacher_veto_v1")
    a = p.parse_args()
    night, bundle, out = (x.resolve() for x in (a.night, a.bundle, a.out))
    for root in (night, bundle):
        if out == root or root in out.parents or out in root.parents:
            raise ValueError("Separate teacher-veto output required")
    night_id, night_report = verify_night(night, bundle)
    examples, _ = bundles.load_bundle(bundle)
    tags_path = next(Path(path) for path in night_id["inputs"] if path.endswith("label_selection.json"))
    if sha(tags_path) != night_id["inputs"][str(tags_path)]:
        raise RuntimeError("Challenge tags changed")
    challenge = {r["token"] for r in read(tags_path)["records"] if r["sampling"].startswith("challenge")}
    for e in examples:
        e["sampling_slice"] = "challenge" if e["metadata"]["token"] in challenge else "uniform"
    snapshots = {}
    tokens = {e["metadata"]["token"] for e in examples}
    for variant in night_id["variants"]:
        with np.load(night / (variant + "_oof_ensemble.npz"), allow_pickle=False) as data:
            prediction = {k: data[k].copy() for k in ("gain", "gain_lower", "risk", "risk_support_sessions")}
            positions = {str(t): i for i, t in enumerate(data["tokens"])}
            count = len(data["tokens"])
        if count != len(tokens) or len(positions) != len(tokens) or set(positions) != tokens:
            raise ValueError("OOF token coverage mismatch")
        snapshots[variant] = prediction, positions
    identity = dict(inputs={str(path): sha(path) for path in (night / "report.json", night / "identity.json", bundle / "bundle_manifest.json")},
        source_sha256=sha(Path(__file__)), epsilon=EPSILON, hosts=list(HOSTS), modes=list(MODES), candidate=CANDIDATE, note=NOTE)
    marker = out / "identity.json"
    if marker.exists() and read(marker) != identity:
        raise RuntimeError("Veto protocol/inputs changed; use a new --out root")
    if out.exists() and not marker.exists() and any(out.iterdir()):
        raise ValueError("Nonempty veto output without identity")
    write(marker, identity)
    frame = pd.read_csv(night / "oof_per_scene.csv", dtype={"token": str}, low_memory=False, float_precision="round_trip")
    if not frame["raw_harm"].isin([True, False]).all():
        raise ValueError("Invalid saved harm flags")
    frame = frame[frame["ensemble"] == "ensemble"]
    rows, opportunity = evaluate(examples, snapshots, night_id, frame, out)
    for name, content in (("per_scene.csv", rows), ("opportunity.csv", opportunity)):
        temp = out / (name + ".tmp")
        pd.DataFrame(content).to_csv(temp, index=False)
        temp.replace(out / name)
    methods = summary(rows, night_id["seeds"][0])
    result_rows = []
    data = pd.DataFrame(rows)
    for name, values in methods.items():
        group = data[data.method == name]
        result_rows.append(dict(method=name, gain_macro_pp=values["gain_macro_pp"], gain_micro_pp=values["gain_micro_pp"],
            harm=values["harm"], quality_degraded=values["quality_degraded"], interventions=values["interventions"],
            coverage=values["coverage"], worst_gain_pp=float(min(0, group.raw_gain_pp.min())), gain_ci95_sessions=values["gain_ci95_sessions"]))
    candidate_rows = [r for r in rows if r["method"] == CANDIDATE]
    qualifies = bool(candidate_rows and eligible(candidate_rows, night_report["selection"]))
    opportunity_frame = pd.DataFrame(opportunity)
    reserve = {scope: dict(oracle_gain_micro_pp=float(g.oracle_gain_pp.mean()),
        oracle_gain_macro_pp=float(g.groupby("session_id").oracle_gain_pp.mean().mean()),
        scenes_with_useful_alternative=int((g.useful_alternatives > 0).sum()),
        mean_allowed_alternatives=float(g.allowed_alternatives.mean())) for scope, g in opportunity_frame.groupby("scope")}
    result = dict(status="complete", scenes=len(examples), sessions=len({e["metadata"]["session_id"] for e in examples}),
        fixed_candidate=CANDIDATE, candidate_passes_previous_criteria=qualifies, primary=CANDIDATE if qualifies else "baseline",
        rows=result_rows, oracle_opportunity=reserve, fresh_holdout_processed=0, new_scorer_calls=0, additional_download_GB=0,
        artifacts={name: sha(out / name) for name in ("identity.json", "decisions.json", "per_scene.csv", "opportunity.csv")}, note=NOTE)
    write(out / "report.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
