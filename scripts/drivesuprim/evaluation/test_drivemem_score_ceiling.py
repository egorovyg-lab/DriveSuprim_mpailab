import unittest

import numpy as np

from audit_drivemem_score_ceiling import ceiling_case, METRICS
from navsim.agents.drivemem import probabilities, metric_selection_score


class ScoreCeilingTests(unittest.TestCase):
    def fixture(self):
        sample = dict(token="q", session_id="s", selected_local=np.array(0), candidate_ids=np.arange(16))
        sample.update({"head_" + k: np.full(16, 10.) for k in METRICS})
        sample["selection_scores"] = metric_selection_score(probabilities(sample)) - .1 * np.arange(16)
        outcomes = {}
        for i in range(16):
            raw = {k: 1. for k in METRICS}
            if i != 15:
                raw["drivable_area_compliance"] = 0.
            raw["pdm_score"] = 1. if i == 15 else 0.
            outcomes[str(i)] = dict(raw=raw)
        return sample, outcomes

    def test_optimistic_baseline_can_make_good_action_structurally_unreachable(self):
        sample, outcomes = self.fixture()
        row = ceiling_case(sample, outcomes)
        self.assertEqual(row["action_oracle_gain_pp"], 100.)
        self.assertEqual(row["anchored_ceiling_gain_pp"], 0.)
        self.assertFalse(row["any_optimal_reachable"])
        self.assertLess(row["optimal_ceiling_margin"], 0.)
        self.assertEqual(row["oracle_heads_contrast_beta10_gain_pp"], 0.)
        self.assertEqual(row["oracle_heads_absolute_beta10_gain_pp"], 100.)

    def test_ceiling_tracks_reachability_and_never_exceeds_action_oracle(self):
        sample, outcomes = self.fixture()
        sample["selection_scores"][15] = sample["selection_scores"][0] - 1e-6
        row = ceiling_case(sample, outcomes)
        self.assertTrue(row["any_optimal_reachable"])
        self.assertEqual(row["anchored_ceiling_gain_pp"], row["action_oracle_gain_pp"])
        for key, value in row.items():
            if key.startswith("oracle_heads_contrast_"):
                self.assertLessEqual(value, row["anchored_ceiling_gain_pp"] + 1e-8)


if __name__ == "__main__":
    unittest.main()
