import copy
import unittest

import numpy as np

from audit_drivemem_completed_families import family_fit, VARIANTS
from audit_drivemem_donor_family import proposals
from navsim.agents.drivemem_action_transfer import transfer, choose
from test_drivemem_action_transfer import make_sample


class CompletedFamilyTests(unittest.TestCase):
    def test_all_four_cells_reproduce_full_bank_action_with_matching_injection(self):
        rng = np.random.default_rng(3)
        vocab = rng.normal(size=(16, 40, 3))
        query = make_sample("q", "q", "development", rng)
        del query["label_values"], query["label_local_indices"]
        old = [make_sample(f"m{i}", f"m{i}", "memory", rng) for i in range(3)]
        new = copy.deepcopy(old)
        for s in new:
            s["label_local_indices"] = np.arange(16)
            s["label_values"] = rng.uniform(0, 1, (16, 8))
        fitted = dict(h=.001, beta=.5, neighbors=5)
        for variant, (completed, contrast) in VARIANTS.items():
            bank = new if completed else old
            spec = family_fit(fitted, variant)
            generated, _ = proposals(query, bank, vocab, spec, 1)
            correction, _ = transfer(query, bank, vocab, .001, contrast, 1)
            self.assertEqual(generated["frozen_primary"], choose(query, correction))
            self.assertEqual(spec["configs"][variant]["contrast"], contrast)
        self.assertEqual(len(old[0]["label_local_indices"]), 8)
        self.assertEqual(len(new[0]["label_local_indices"]), 16)

    def test_changed_strength_is_not_silently_used(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            family_fit(dict(h=.001, beta=1., neighbors=5), "completed_M_absolute")


if __name__ == "__main__":
    unittest.main()
