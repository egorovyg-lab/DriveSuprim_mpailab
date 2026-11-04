"""Final proposal-family capacity check across the four completed-M experiment cells.

Read-only; no fitting, simulator calls or new data. These are retrospective oracle
diagnostics on inspected development cohorts, not learned selector performance.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_drivemem_donor_family import proposals, lookup
from audit_drivemem_action_support import load_frozen_bank, load_sample, read_json, sha
from navsim.planning.script.run_drivemem_memory_completion import labelled, VARIANTS


def family_fit(fitted, variant):
    if fitted["beta"] != .5 or fitted["neighbors"] != 5 or variant not in VARIANTS:
        raise ValueError("Unsupported factorial protocol")
    return dict(primary=variant, beta=.5, neighbors=5,
                configs={variant: dict(h=fitted["h"], contrast=VARIANTS[variant][1])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("completion", type=Path)
    parser.add_argument("--all-cases", action="store_true")
    args = parser.parse_args()
    root = args.completion
    manifest, fitted = read_json(root / "identity.json"), read_json(root / "fit.json")
    previous, validation = Path(manifest["previous"]), Path(manifest["validation"])
    source, old_bank, vocab, _, old_identity = load_frozen_bank(previous)
    if str(source) != manifest["source"] or sha(previous / "identity.json") != manifest["previous_identity"] or sha(previous / "fit.json") != manifest["previous_fit"]:
        raise RuntimeError("Completion source/frozen model identity mismatch")
    repo = Path(__file__).resolve().parents[3]
    if sha(repo / "navsim/planning/script/run_drivemem_memory_completion.py") != manifest["runner_sha256"]:
        raise RuntimeError("Completion runner changed")
    if sha(root / "ridge.npz") != fitted["ridge_sha256"]:
        raise RuntimeError("Completion fitted model changed")
    new_bank = []
    if set(fitted["memory_labels"]) != {s["token"] + ".json" for s in old_bank}:
        raise RuntimeError("Completion bank membership mismatch")
    for s in old_bank:
        path = root / "memory_labels" / (s["token"] + ".json")
        if sha(path) != fitted["memory_labels"][path.name]:
            raise RuntimeError("Completed M labels changed")
        state = read_json(path)
        if state["token"] != s["token"]:
            raise RuntimeError("Completed M token mismatch")
        new_bank.append(labelled(s, state["outcomes"]))
    decisions = read_json(root / "query_decisions.json")
    if decisions["fit_sha256"] != sha(root / "fit.json"):
        raise RuntimeError("Completion decisions/fit mismatch")
    prepared = []
    for cohort_root in (source, validation):
        if read_json(cohort_root / "split_identity.json") != dict(splits=sha(cohort_root / "splits.json"), inventory=sha(cohort_root / "inventory.csv")):
            raise RuntimeError("Query selection changed")
        with np.load(cohort_root / "vocab.npz", allow_pickle=False) as data:
            if not np.array_equal(data["vocab"], vocab):
                raise RuntimeError("Vocabulary mismatch")
        index = read_json(cohort_root / "export_index.json")
        samples = [load_sample(cohort_root, r, index) for r in read_json(cohort_root / "splits.json")["records"] if r["role"] == "development"]
        for variant, (completed, _) in VARIANTS.items():
            virtual_fit = family_fit(fitted, variant)
            bank = new_bank if completed else old_bank
            queries = []
            for s in samples:
                policy, near = proposals(s, bank, vocab, virtual_fit, int(old_identity["seed"]))
                if policy["frozen_primary"] != decisions["cohorts"][cohort_root.name][s["token"]][variant]:
                    raise RuntimeError("Completed-experiment action not reproduced")
                queries.append((s, policy, near))
                if len(queries) % 4 == 0:
                    print(f"PROPOSALS {cohort_root.name} {variant}: {len(queries)}", flush=True)
            prepared.append((cohort_root, variant, queries))
    # All four cells and both cohorts are fixed before opening any query outcomes.
    reports, case_tables = [], []
    expected = read_json(root / "report.json")["oracle_inputs"]
    for cohort_root, variant, queries in prepared:
        rows = []
        for s, policy, near in queries:
            path = cohort_root / "development_oracle/scenes" / (s["token"] + ".json")
            if sha(path) != expected[cohort_root.name][s["token"]]:
                raise RuntimeError("Query oracle changed since completion report")
            state = read_json(path)
            rows.append(lookup(s, policy, near, state["outcomes"]))
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError("Empty cohort")
        gains = [c for c in frame if c.endswith("gain_pp")]
        row = dict(cohort=cohort_root.name, variant=variant, chosen_on_F=variant == fitted["primary"],
                   scenes=len(frame), sessions=frame.session_id.nunique())
        row.update(frame[gains].mean().to_dict())
        row["family_improved_scenes"] = int((frame.all_family_oracle_gain_pp > 1e-6).sum())
        reports.append(row)
        case_tables.append((cohort_root.name, variant, frame, gains))
    print("\nCAPACITY SUMMARY — ORACLES, NOT TRAINED SELECTORS:")
    print(pd.DataFrame(reports).to_string(index=False))
    for cohort, variant, frame, gains in case_tables:
        print("\nBY SESSION:", cohort, variant)
        print(frame.groupby("session_id")[gains].mean().to_string())
        print("LARGEST ACTION-ORACLE GAPS:" if not args.all_cases else "ALL CASES:")
        gap = frame.action_oracle_gain_pp - frame.all_family_oracle_gain_pp
        chosen = frame if args.all_cases else frame.loc[gap.nlargest(3).index]
        print(chosen.to_string(index=False))
    print("\nRead-only diagnostic on inspected data. No parameter selection/training. "
          "Families do not exhaust all donor combinations; oracle improvements are not deployable results. "
          "If enriched absolute families still offer little useful action diversity, stop tuning this residual-transfer family.")


if __name__ == "__main__":
    main()
