import unittest

import numpy as np

from diagnose_drivemem_memory import diagnose_query, METRICS


class DiagnosticsTests(unittest.TestCase):
    def fixture(self):
        query = dict(token="query", session_id="query_session", selected_local=np.array(0),
                     candidate_ids=np.arange(2), selection_scores=np.array([1., 0.]),
                     ego_status=np.zeros(8), scene_embedding=np.ones(3))
        query.update({"head_" + k: np.zeros(2) for k in METRICS})
        base = dict(method="baseline", raw_score=.5, raw_gain_pp=0.)
        base.update({"raw_" + k: 1. for k in METRICS})
        sim = dict(base, method="similarity_memory")
        saved = dict(decisions={"baseline": 0, "similarity_memory": 0},
                     retrieval={"donor_tokens": []}, rows=[base, sim],
                     outcomes={"0": {"raw": {"pdm_score": .5}}, "1": {"raw": {"pdm_score": .9}}})
        return query, np.zeros((2, 4, 3)), {"neighbors": 5, "seed": 1, "strength": .5}, saved

    def test_no_donors_and_evaluated_union_do_not_imply_memory_improvement(self):
        query, vocab, settings, saved = self.fixture()
        row, donors = diagnose_query(query, [], vocab, settings, 1., saved)
        self.assertEqual(donors, [])
        self.assertIsNone(row["nearest_distance"])
        self.assertFalse(row["threshold_pass"])
        self.assertEqual(row["selection_delta_span"], 0.)
        self.assertEqual(row["similarity_gain_pp"], 0.)
        self.assertAlmostEqual(row["evaluated_union_best_gain_pp"], 40.)

    def test_retrieval_counts_sessions_and_checks_saved_provenance(self):
        query, vocab, settings, saved = self.fixture()
        bank = []
        for i in range(2):
            donor = dict(query, token=f"donor{i}", session_id="one_donor_session",
                         label_local_indices=np.array([0]), label_values=np.full((1, len(METRICS)), .5))
            bank.append(donor)
        saved["retrieval"]["donor_tokens"] = ["donor0", "donor1"]
        row, _ = diagnose_query(query, bank, vocab, settings, 1., saved)
        self.assertEqual(row["donor_count"], 2)
        self.assertEqual(row["donor_sessions"], 1)
        self.assertTrue(row["threshold_pass"])
        saved["retrieval"]["donor_tokens"] = ["wrong"]
        with self.assertRaisesRegex(RuntimeError, "retrieval differs"):
            diagnose_query(query, bank, vocab, settings, 1., saved)

    def test_changed_policy_is_rejected(self):
        query, vocab, settings, saved = self.fixture()
        saved["decisions"]["similarity_memory"] = 1
        with self.assertRaisesRegex(RuntimeError, "decisions differ"):
            diagnose_query(query, [], vocab, settings, 1., saved)


if __name__ == "__main__":
    unittest.main()
