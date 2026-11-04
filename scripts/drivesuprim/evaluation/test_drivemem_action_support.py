import unittest

import numpy as np

from audit_drivemem_action_support import label_bands, support, METRICS


class ActionSupportTests(unittest.TestCase):
    def sample(self, token, embedding, indices):
        return dict(token=token, session_id=token, candidate_ids=np.arange(16),
                    selection_scores=np.arange(16., 0., -1), ego_status=np.zeros(8),
                    scene_embedding=np.array(embedding, dtype=float), label_local_indices=np.array(indices),
                    label_values=np.ones((len(indices), len(METRICS))))

    def test_rank_coverage_does_not_assume_random_labels_cover_top16(self):
        bank = [self.sample("m", [1, 0], list(range(8)) + [10])]
        bands = {r["rank_band"]: r for r in label_bands(bank)}
        self.assertEqual(bands["1-8"]["labels"], 8)
        self.assertEqual(bands["9-16"]["labels"], 1)
        self.assertEqual(bands["9-16"]["candidate_slots"], 8)
        self.assertEqual(bands["9-16"]["labelled_fraction"], .125)

    def test_global_support_separates_bank_coverage_from_retrieval(self):
        query = self.sample("q", [1, 0], [])
        bank = [self.sample(f"m{i}", [1, 0], [0]) for i in range(5)]
        bank.append(self.sample("far", [0, 1], [8]))
        vocab = np.zeros((16, 40, 3))
        vocab[8, :, 0] = 20
        stats = support(query, 8, bank, vocab, .01, 1)
        self.assertEqual(stats["donors"], 5)
        self.assertEqual(stats["exact_supporting_donors"], 0)
        self.assertEqual(stats["global_exact_supporting_episodes"], 1)
        self.assertEqual(stats["global_min_distance_over_h"], 0.)
        self.assertGreater(stats["min_distance_over_h"], 0.)


if __name__ == "__main__":
    unittest.main()
