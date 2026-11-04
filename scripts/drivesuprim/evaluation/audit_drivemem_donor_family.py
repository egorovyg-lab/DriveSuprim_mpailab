"""Retrospective proposal-family oracle; not a deployable selector or training run.

All proposals use the existing frozen M bank, h and beta. Only after all proposals
are fixed do we look up known query outcomes. No new labels or simulator calls.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_drivemem_action_support import load_frozen_bank, load_sample, read_json, sha
from navsim.agents.drivemem import nearest_donors, METRICS, SAFETY
from navsim.agents.drivemem_action_transfer import transfer, choose, pool


def proposals(query, bank, vocab, fit, seed):
    spec = fit["configs"][fit["primary"]]
    if fit["beta"] != .5 or fit["neighbors"] != 5:
        raise ValueError("Unexpected frozen protocol")
    compatible, _ = nearest_donors(query, bank, len(bank), seed)
    near = compatible[:5]

    def action(donors, random=False, random_seed=seed):
        correction, _ = transfer(query, donors, vocab, spec["h"], spec["contrast"], random_seed, random=random)
        return choose(query, correction, fit["beta"])

    result = dict(baseline=int(query["selected_local"]), frozen_primary=action(bank))
    near_names = {"baseline", "frozen_primary"}
    for i in compatible:
        name = "single_" + str(bank[i]["token"])
        result[name] = action([bank[i]])
        if i in near:
            near_names.add(name)
    for k in range(1, len(near) + 1):
        name = f"nearest_prefix_{k}"
        result[name] = action([bank[i] for i in near[:k]])
        near_names.add(name)
    if len(near) > 1:
        for removed in near:
            name = "leave_one_out_" + str(bank[removed]["token"])
            result[name] = action([bank[i] for i in near if i != removed])
            near_names.add(name)
    for offset in range(3):
        result[f"random5_{offset}"] = action(bank, random=True, random_seed=seed + offset)
    return result, near_names


def lookup(query, proposed, near_names, outcomes):
    baseline = int(query["selected_local"])
    indices = pool(query).tolist()
    if any(i not in indices for i in proposed.values()) or any(str(i) not in outcomes for i in indices):
        raise RuntimeError("Proposal outside fixed pool or incomplete oracle")
    base = outcomes[str(baseline)]["raw"]

    def gain(i):
        return 100 * (outcomes[str(i)]["raw"]["pdm_score"] - base["pdm_score"])

    def safe(i):
        return all(outcomes[str(i)]["raw"][METRICS[j]] + 1e-8 >= base[METRICS[j]] for j in SAFETY)

    near = {proposed[name] for name in near_names}
    all_actions = set(proposed.values())
    best = max(all_actions, key=gain)
    return dict(token=query["token"], session_id=query["session_id"],
        frozen_gain_pp=gain(proposed["frozen_primary"]),
        gate_oracle_gain_pp=max(0., gain(proposed["frozen_primary"])),
        retrieved_family_oracle_gain_pp=max(map(gain, near)),
        all_family_oracle_gain_pp=gain(best),
        safe_all_family_oracle_gain_pp=max(gain(i) for i in all_actions if safe(i)),
        action_oracle_gain_pp=max(map(gain, indices)),
        policy_count=len(proposed), distinct_proposed_actions=len(all_actions),
        degraded_proposed_actions=sum(gain(i) < -1e-6 for i in all_actions),
        unsafe_proposed_actions=sum(not safe(i) for i in all_actions),
        best_proposed_rank=indices.index(best) + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("validation", type=Path)
    args = parser.parse_args()
    source, bank, vocab, fit, identity = load_frozen_bank(args.experiment)
    original_choices = {r["token"]: r["choices"] for r in read_json(args.experiment / "development_decisions.json")["decisions"]}
    new_choices = read_json(args.validation / "frozen_evaluation/decisions.json")
    prepared = []
    for root, expected in ((source, original_choices), (args.validation, new_choices)):
        if read_json(root / "split_identity.json") != dict(splits=sha(root / "splits.json"), inventory=sha(root / "inventory.csv")):
            raise RuntimeError("Cohort selection changed")
        with np.load(root / "vocab.npz", allow_pickle=False) as data:
            if not np.array_equal(data["vocab"], vocab):
                raise RuntimeError("Cohort vocabulary mismatch")
        index = read_json(root / "export_index.json")
        cohort = []
        for record in read_json(root / "splits.json")["records"]:
            if record["role"] != "development":
                continue
            sample = load_sample(root, record, index)
            policy, near = proposals(sample, bank, vocab, fit, int(identity["seed"]))
            if policy["frozen_primary"] != expected[sample["token"]][fit["primary"]]:
                raise RuntimeError("Recomputed primary differs from original frozen decision")
            cohort.append((sample, policy, near))
            if len(cohort) % 4 == 0:
                print(f"PROPOSALS {root.name}: {len(cohort)} queries; no outcome lookup yet", flush=True)
        prepared.append((root, cohort))
    # Outcomes are opened only after proposals for BOTH cohorts have been fixed.
    for root, cohort in prepared:
        rows = []
        for sample, policy, near in cohort:
            state = read_json(root / "development_oracle/scenes" / (sample["token"] + ".json"))
            rows.append(lookup(sample, policy, near, state["outcomes"]))
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError("No cohort records")
        gains = [c for c in frame if c.endswith("gain_pp")]
        print("\nCOHORT:", root, "SCENES:", len(frame), "SESSIONS:", frame.session_id.nunique())
        print("MEAN DIAGNOSTIC GAINS (oracle values use true query outcomes):")
        print(frame[gains].mean().to_string())
        print("BY SESSION:")
        print(frame.groupby("session_id")[gains].mean().to_string())
        print("ALL CASES:")
        print(frame.to_string(index=False))
    print("\nThese are retrospective capacities of the specified proposal families, NOT learned method results. "
          "They do not exhaust all donor combinations. Donor-policy pairs are not independent observations. "
          "No fitting, simulator calls, downloads or artifact changes were performed.")


if __name__ == "__main__":
    main()
