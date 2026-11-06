import copy
import unittest

import numpy as np

from navsim.agents import drivemem_numeric_contract as contract
from navsim.agents import drivemem_decoder_utility as model
from navsim.agents.drivemem_determinism import PROFILE, repeat_parity
from navsim.planning.script import run_drivemem_decoder_deterministic_export as strict
from navsim.planning.script import run_drivemem_decoder_numeric_export as exporter


class NumericContractTests(unittest.TestCase):
    def fixture(self):
        vocab = np.arange(32 * 40 * 3, dtype=float).reshape(32, 40, 3)
        old = dict(candidate_ids=np.arange(32), selected_local=np.array(0),
                   selection_scores=np.linspace(2., 0., 32), head_drivable_area_compliance=np.full(32, 4.),
                   head_imi=np.linspace(-2., 2., 32))
        fresh = copy.deepcopy(dict(old, final_traj=vocab[0].copy()))
        return old, fresh, vocab

    def test_small_old_new_drift_accepted_and_recorded_but_current_repeat_remains_exact(self):
        old, fresh, vocab = self.fixture()
        fresh["head_drivable_area_compliance"][3] += 1.6e-5
        fresh["head_imi"][5] += 2.6e-5
        with self.assertRaises(RuntimeError):
            strict.parity(old, fresh, vocab)
        contract.parity(old, fresh, vocab)
        report = contract.audit(old, fresh)
        self.assertEqual(report["fields"]["head_drivable_area_compliance"]["legacy_failed_elements"], 1)
        self.assertTrue(report["top16_set_equal"])
        with self.assertRaises(RuntimeError):
            repeat_parity(dict(old, final_traj=vocab[0]), fresh)
        aggregate = contract.aggregate({"q": dict(numeric_audit=report)})
        self.assertEqual(aggregate["scenes_outside_legacy_tolerance"], 1)

    def test_absolute_logit_and_probability_budgets_are_independent(self):
        old, fresh, vocab = self.fixture()
        fresh["head_drivable_area_compliance"][2] += 6e-5
        with self.assertRaisesRegex(RuntimeError, "Logit budget"):
            contract.parity(old, fresh, vocab)
        old["head_drivable_area_compliance"][:] = 0.
        fresh["head_drivable_area_compliance"][:] = 0.
        fresh["head_drivable_area_compliance"][2] = 4.5e-5
        with self.assertRaisesRegex(RuntimeError, "Sigmoid budget"):
            contract.parity(old, fresh, vocab)

    def test_scores_baseline_trajectory_and_pool_membership_stay_guarded(self):
        old, fresh, vocab = self.fixture()
        fresh["selection_scores"][5] += 1e-4
        with self.assertRaisesRegex(RuntimeError, "score drift"):
            contract.parity(old, fresh, vocab)
        old["selection_scores"][15:17] = [1. + 1e-7, 1. - 1e-7]
        fresh = copy.deepcopy(dict(old, final_traj=vocab[0]))
        fresh["selection_scores"][15] -= 3e-7
        with self.assertRaisesRegex(RuntimeError, "top16"):
            contract.parity(old, fresh, vocab)
        for field in ("candidate_ids", "selected_local", "final_traj"):
            old, fresh, vocab = self.fixture()
            fresh[field] = fresh[field] + 1
            with self.assertRaises(RuntimeError):
                contract.parity(old, fresh, vocab)

    def test_nonfinite_and_wrong_shapes_are_rejected(self):
        old, fresh, vocab = self.fixture()
        fresh["head_imi"][0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Invalid numerical"):
            contract.parity(old, fresh, vocab)
        fresh["head_imi"] = np.zeros(3)
        with self.assertRaisesRegex(RuntimeError, "Invalid numerical"):
            contract.parity(old, fresh, vocab)

    def test_numeric_protocol_is_hash_bound_in_loader(self):
        name = "navsim/planning/script/run_drivemem_decoder_numeric_export.py"
        identity = dict(exporter_path=name, exporter_sha256=exporter.sha(model.REPO / name),
            pool_helper_sha256=exporter.sha(model.REPO / "navsim/agents/drivemem_frozen_pool.py"),
            numerical_profile=PROFILE, numerical_helper_sha256=exporter.sha(model.REPO / "navsim/agents/drivemem_determinism.py"),
            numeric_contract=copy.deepcopy(contract.CONTRACT), numeric_helper_sha256=exporter.sha(model.REPO / "navsim/agents/drivemem_numeric_contract.py"))
        self.assertEqual(model.verify_decoder_protocol(identity), name)
        identity["numeric_contract"]["logit_max_abs"] *= 10
        with self.assertRaisesRegex(RuntimeError, "Numeric budget protocol"):
            model.verify_decoder_protocol(identity)


if __name__ == "__main__":
    unittest.main()
