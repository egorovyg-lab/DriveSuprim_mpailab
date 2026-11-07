import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from navsim.agents import drivemem_functional_contract as contract
from navsim.agents.drivemem import METRICS
from navsim.agents.drivemem_determinism import PROFILE, repeat_parity
from navsim.agents.drivemem_numeric_contract import parity as numeric_parity
from navsim.agents import drivemem_decoder_utility as loader
from navsim.planning.script import run_drivemem_decoder_functional_export as exporter


def fixture():
    vocab = np.arange(256 * 40 * 3, dtype=float).reshape(256, 40, 3)
    old = dict(candidate_ids=np.arange(256), selected_local=np.array(0), selection_scores=np.linspace(2., 0., 256))
    old.update({"head_" + name: np.full(256, 2.) for name in METRICS})
    old["head_imi"] = np.linspace(-1., 1., 256)
    return old, copy.deepcopy(dict(old, final_traj=vocab[0].copy())), vocab


class FunctionalContractTests(unittest.TestCase):
    def test_pilot_covers_both_audit_violations_before_remaining_cohort(self):
        rows = [(Path("source"), dict(token=t)) for t in ("other",) + contract.PILOT_TOKENS[::-1]]
        ordered = contract.pilot_order(rows)
        self.assertEqual(tuple(r["token"] for _, r in ordered[:4]), contract.PILOT_TOKENS)
        self.assertEqual(len(ordered), len(rows))

    def test_imi_offset_is_diagnostic_while_current_repeat_must_remain_exact(self):
        old, fresh, vocab = fixture()
        fresh["head_imi"] += .1  # Constant offset leaves softmax unchanged.
        with self.assertRaises(RuntimeError):
            numeric_parity(old, fresh, vocab)
        contract.parity(old, fresh, vocab)
        report = contract.aggregate({"q": dict(numeric_audit=contract.audit(old, fresh))})
        self.assertTrue(report["imi_drift_is_diagnostic"])
        self.assertGreater(report["field_maxima"]["head_imi"]["max_abs_diff"], .09)
        with self.assertRaises(RuntimeError):
            repeat_parity(dict(old, final_traj=vocab[0]), fresh)

    def test_metric_logit_sigmoid_and_score_constraints_unchanged(self):
        for change, message in (("logit", "Metric logit"), ("sigmoid", "Metric sigmoid"), ("score", "score drift")):
            old, fresh, vocab = fixture()
            if change == "logit":
                fresh["head_no_at_fault_collisions"][0] += 6e-5
            elif change == "sigmoid":
                old["head_no_at_fault_collisions"][:] = 0.
                fresh["head_no_at_fault_collisions"][:] = 4.5e-5
            else:
                fresh["selection_scores"][0] += 1e-3
            with self.assertRaisesRegex(RuntimeError, message):
                contract.parity(old, fresh, vocab)

    def test_action_pool_and_trajectory_changes_are_rejected(self):
        for name in ("candidate_ids", "selected_local", "final_traj"):
            old, fresh, vocab = fixture()
            fresh[name] = fresh[name] + 1
            with self.assertRaises(RuntimeError):
                contract.parity(old, fresh, vocab)
        old, fresh, vocab = fixture()
        cut = float(np.mean(old["selection_scores"][15:17]))
        old["selection_scores"][15:17] = [cut + 1e-7, cut - 1e-7]
        fresh["selection_scores"] = old["selection_scores"].copy()
        fresh["selection_scores"][15] -= 3e-7
        with self.assertRaisesRegex(RuntimeError, "top16"):
            contract.parity(old, fresh, vocab)

    def test_missing_heads_bad_shapes_and_nonfinite_imi_are_rejected(self):
        for change in ("nan", "shape", "missing"):
            old, fresh, vocab = fixture()
            if change == "nan":
                fresh["head_imi"][0] = np.nan
            elif change == "shape":
                fresh["head_imi"] = np.zeros(3)
            else:
                del fresh["head_imi"]
            with self.assertRaises(RuntimeError):
                contract.parity(old, fresh, vocab)

    def test_complete_prior_audit_is_bound_including_its_failures(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp)
            source = root / "source"
            bindings = {"frozen-input": "digest"}
            records = [(source, dict(token=str(i), role="memory" if i < 100 else "fit")) for i in range(152)]
            exporter.write(root / "identity.json", dict(inputs=bindings, profile=PROFILE,
                runner=exporter.sha(loader.REPO / "navsim/planning/script/run_drivemem_decoder_numeric_audit.py"), helpers={},
                records=[dict(source=str(rt), **r) for rt, r in records]))
            exporter.write(root / "report.json", dict(status="complete", audited_scenes=152, required_scenes=152, contract_and_invariants_passed=150))
            exporter.write(root / "runtime.json", dict(settings=PROFILE))
            runtime_digest = exporter.sha(root / "runtime.json")
            index = {}
            for rt, r in records:
                k = rt.name + ":" + r["token"]
                path = root / "scenes" / (k.replace(":", "_") + ".json")
                exporter.write(path, dict(**r, source=str(rt), repeat_bitwise_equal=True, baseline_equal=True,
                    vocabulary_equal=True, head_alignment={"imi": True}, numeric_audit=dict(top16_set_equal=True),
                    contract_passed=int(r["token"]) >= 2, runtime_sha256=runtime_digest))
                index[k] = exporter.sha(path)
            exporter.write(root / "audit_index.json", index)
            snapshot = contract.prior_audit_snapshot(root, bindings, records, loader.REPO, exporter.read, exporter.sha)
            self.assertEqual(len(snapshot["files"]), 156)
            self.assertTrue(snapshot["criteria_revised_after_audit"])
            path = root / "scenes/source_0.json"
            row = exporter.read(path)
            row["baseline_equal"] = False
            exporter.write(path, row)
            with self.assertRaisesRegex(RuntimeError, "scene/runtime changed"):
                contract.prior_audit_snapshot(root, bindings, records, loader.REPO, exporter.read, exporter.sha)
            index["source:0"] = exporter.sha(path)
            exporter.write(root / "audit_index.json", index)
            with self.assertRaisesRegex(RuntimeError, "invariant failure"):
                contract.prior_audit_snapshot(root, bindings, records, loader.REPO, exporter.read, exporter.sha)

    def test_loader_binds_functional_criteria_and_helper_hashes(self):
        name = "navsim/planning/script/run_drivemem_decoder_functional_export.py"
        identity = dict(exporter_path=name, exporter_sha256=exporter.sha(loader.REPO / name),
            pool_helper_sha256=exporter.sha(loader.REPO / "navsim/agents/drivemem_frozen_pool.py"), numerical_profile=PROFILE,
            numerical_helper_sha256=exporter.sha(loader.REPO / "navsim/agents/drivemem_determinism.py"),
            functional_contract=copy.deepcopy(contract.CONTRACT),
            functional_helper_sha256=exporter.sha(loader.REPO / "navsim/agents/drivemem_functional_contract.py"),
            numeric_audit_helper_sha256=exporter.sha(loader.REPO / "navsim/agents/drivemem_numeric_contract.py"))
        self.assertEqual(loader.verify_decoder_protocol(identity), name)
        identity["functional_contract"]["score_atol"] *= 10
        with self.assertRaisesRegex(RuntimeError, "Functional acceptance protocol"):
            loader.verify_decoder_protocol(identity)


if __name__ == "__main__":
    unittest.main()
