import unittest

import numpy as np

from audit_drivemem_donor_family import lookup, proposals, METRICS
from test_drivemem_action_transfer import make_sample


class DonorFamilyTests(unittest.TestCase):
    def test_proposals_use_no_query_labels_and_keep_fixed_pool(self):
        rng = np.random.default_rng(4)
        vocab = rng.normal(size=(16, 40, 3))
        query = make_sample("q", "q", "development", rng)
        del query["label_values"], query["label_local_indices"]
        bank = [make_sample(f"m{i}", f"s{i}", "memory", rng) for i in range(6)]
        fit = dict(primary="B3", beta=.5, neighbors=5, configs={"B3": dict(h=.01, contrast=True)})
        p, near = proposals(query, bank, vocab, fit, 1)
        self.assertEqual(p, proposals(query, bank, vocab, fit, 1)[0])
        self.assertTrue(near <= set(p))
        self.assertEqual(sum(name.startswith("single_") for name in p), 6)
        self.assertEqual(sum(name.startswith("single_") for name in near), 5)
        self.assertTrue(all(i in range(16) for i in p.values()))
        self.assertEqual(p["nearest_prefix_5"], p["frozen_primary"])

    def test_family_ceiling_is_distinct_from_gate_and_full_action_oracle(self):
        query = dict(token="q", session_id="s", selected_local=np.array(0),
                     selection_scores=np.arange(16., 0., -1), candidate_ids=np.arange(16))
        outcomes = {}
        for i in range(16):
            values = {k: 1. for k in METRICS}
            values["pdm_score"] = {0: .5, 1: .4, 2: .6, 3: .8, 15: .9}.get(i, .5)
            if i == 3:
                values["no_at_fault_collisions"] = .5
            outcomes[str(i)] = dict(raw=values)
        policy = dict(baseline=0, frozen_primary=1, near=2, far=3)
        result = lookup(query, policy, {"baseline", "frozen_primary", "near"}, outcomes)
        self.assertAlmostEqual(result["frozen_gain_pp"], -10.)
        self.assertEqual(result["gate_oracle_gain_pp"], 0.)
        self.assertAlmostEqual(result["retrieved_family_oracle_gain_pp"], 10.)
        self.assertAlmostEqual(result["all_family_oracle_gain_pp"], 30.)
        self.assertAlmostEqual(result["safe_all_family_oracle_gain_pp"], 10.)
        self.assertAlmostEqual(result["action_oracle_gain_pp"], 40.)
        policy["outside"] = 99
        with self.assertRaisesRegex(RuntimeError, "outside fixed pool"):
            lookup(query, policy, {"baseline"}, outcomes)


if __name__ == "__main__":
    unittest.main()
