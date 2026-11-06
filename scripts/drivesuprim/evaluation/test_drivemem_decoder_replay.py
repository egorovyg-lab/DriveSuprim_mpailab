import ast
import copy
import unittest
from pathlib import Path

import numpy as np

from navsim.agents.drivemem_frozen_pool import apply_pool, describe_pool
from navsim.planning.script import run_drivemem_decoder_export as original
from navsim.planning.script import run_drivemem_decoder_replay_export as replay


class FrozenPoolTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(41)
        vocab = rng.normal(size=(8192, 40, 3))
        states = rng.normal(size=(1, 8192, 8))
        heads = {k: rng.normal(size=(1, 8192)) for k in ("imi", "no_at_fault_collisions", "ego_progress")}
        natural = np.arange(256)[None]
        stage = dict(indices_absolute=natural, trajs=vocab[natural],
                     trajs_status=states[:, natural[0]],
                     coarse_score={k: v[:, natural[0]] for k, v in heads.items()})
        return dict(heads, trajectory_vocab=vocab, refinement=[stage]), states

    def test_replay_repairs_order_and_membership_with_all_candidate_inputs_aligned(self):
        output, states = self.fixture()
        frozen = np.arange(512, 256, -1)
        natural = output["refinement"][0]["indices_absolute"][0].copy()
        heads = {k: output[k].copy() for k in output["refinement"][0]["coarse_score"]}
        self.assertIs(apply_pool(output, states, frozen), output)
        stage = output["refinement"][0]
        np.testing.assert_array_equal(stage["indices_absolute"][0], frozen)
        np.testing.assert_array_equal(stage["trajs"][0], output["trajectory_vocab"][frozen])
        np.testing.assert_array_equal(stage["trajs_status"][0], states[0, frozen])
        for k, value in heads.items():
            np.testing.assert_array_equal(stage["coarse_score"][k][0], value[0, frozen])
            np.testing.assert_array_equal(output[k], value)
        self.assertFalse(describe_pool(frozen, natural)["same_set"])
        order_only = describe_pool(frozen, frozen[::-1])
        self.assertTrue(order_only["same_set"])
        self.assertFalse(order_only["same_order"])
        report = replay.replay_report({
            "order": {"natural_pool_passes": [order_only, describe_pool(frozen, frozen)]},
            "membership": {"natural_pool_passes": [describe_pool(frozen, natural)] * 2},
        }, {"exported_scenes": 2})
        self.assertEqual(report["scenes_with_natural_order_drift"], 2)
        self.assertEqual(report["scenes_with_natural_membership_drift"], 1)

    def test_invalid_pool_or_full_vocab_states_rejected_before_mutation(self):
        output, states = self.fixture()
        for invalid in (np.zeros(256, dtype=int), np.arange(256, dtype=float),
                        np.arange(-1, 255), np.arange(8000, 8256), np.arange(16)):
            before = copy.deepcopy(output)
            with self.assertRaises(ValueError):
                apply_pool(output, states, invalid)
            np.testing.assert_array_equal(output["refinement"][0]["indices_absolute"], before["refinement"][0]["indices_absolute"])
        with self.assertRaises(ValueError):
            apply_pool(output, states[:, :256], np.arange(256))

    def test_final_parity_is_unchanged_and_rejects_changed_outcomes_of_refinement(self):
        def functions(module):
            return {n.name: ast.dump(n) for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
        for name in ("parity", "wanted_records", "status"):
            self.assertEqual(functions(original)[name], functions(replay)[name])
        vocab = np.zeros((256, 40, 3))
        old = dict(candidate_ids=np.arange(256), selected_local=np.array(0),
                   selection_scores=np.arange(256, 0, -1, dtype=float), head_imi=np.zeros(256))
        fresh = dict(old, final_traj=vocab[0].copy())
        self.assertEqual(replay.parity(old, fresh, vocab), 0)
        for field in ("selection_scores", "head_imi", "selected_local", "final_traj"):
            bad = copy.deepcopy(fresh)
            bad[field] = bad[field] + 1
            with self.assertRaises(RuntimeError):
                replay.parity(old, bad, vocab)


if __name__ == "__main__":
    unittest.main()
