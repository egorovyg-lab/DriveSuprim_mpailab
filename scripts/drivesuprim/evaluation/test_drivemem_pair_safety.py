import unittest

import numpy as np

from audit_drivemem_pair_safety import safety_rows, case_rows
from test_drivemem_action_transfer import make_sample
from navsim.agents.drivemem import METRICS


class PairSafetyAuditTests(unittest.TestCase):
    def setUp(self):
        self.sample = make_sample("q", "sid", "memory", np.random.default_rng(4))
        self.values = {}
        for i in range(16):
            value = dict.fromkeys(METRICS, 1.)
            value["pdm_score"] = .5
            self.values[str(i)] = dict(raw=value, filtered=value.copy())
        self.values["1"]["raw"].update(driving_direction_compliance=.5, pdm_score=.2)
        self.values["2"]["raw"]["pdm_score"] = .9

    def test_safety_counts_distinguish_absolute_failure_from_degradation(self):
        self.values["0"]["raw"]["no_at_fault_collisions"] = .5
        self.values["3"]["raw"]["no_at_fault_collisions"] = .5
        rows = [r for r in safety_rows(self.sample, self.values) if r["nonbaseline"]]
        self.assertEqual(len(rows), 75)
        self.assertEqual(sum(r["degraded"] for r in rows), 1)
        self.assertEqual(sum(r["below_one"] for r in rows), 2)
        self.assertEqual(sum(r["improved"] for r in rows), 14)

    def test_selected_and_oracle_prediction_error_and_head_deltas(self):
        prediction = np.full(16, -.1)
        prediction[0], prediction[1] = 0, .1
        selected, oracle = case_rows(self.sample, self.values, "direct_memory", prediction)
        self.assertEqual(selected["candidate_id"], 1)
        self.assertAlmostEqual(selected["predicted_gain_pp"], 10)
        self.assertAlmostEqual(selected["measured_gain_pp"], -30)
        self.assertAlmostEqual(selected["optimism_pp"], 40)
        self.assertEqual(selected["measured_delta_driving_direction_compliance"], -.5)
        self.assertEqual(oracle["candidate_id"], 2)
        self.assertAlmostEqual(oracle["measured_gain_pp"], 40)
        self.assertAlmostEqual(oracle["predicted_gain_pp"], -10)

    def test_optional_tail_is_separated_from_top16(self):
        sample = dict(self.sample, candidate_ids=np.arange(18), selection_scores=np.arange(18., 0., -1))
        values = dict(self.values)
        extra = dict.fromkeys(METRICS, 1.)
        extra.update(pdm_score=.1, driving_direction_compliance=.5)
        values["17"] = dict(raw=extra)
        top = safety_rows(sample, values)
        all_rows = safety_rows(sample, values, include_tail=True)
        self.assertEqual(len(top), 80)
        tail = [r for r in all_rows if r["rank"] > 16]
        self.assertEqual(len(tail), 5)
        self.assertEqual(sum(r["degraded"] for r in tail), 1)
        self.assertTrue(all(r["rank"] == 18 for r in tail))


if __name__ == "__main__":
    unittest.main()
