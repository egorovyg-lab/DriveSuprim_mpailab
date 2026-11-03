import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from navsim.planning.script.run_drivemem_development_oracle import audit, best_action, ranked_indices, METRICS


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def outcome(index):
    values = {k: 1. for k in METRICS}
    values.update(pdm_score=[.5, .7, .9, .95][index], aggregate_without_two_frame=[.5, .7, .9, .95][index])
    if index == 1:
        values[METRICS[0]] = .5
    return {"raw": dict(values), "filtered": dict(values)}


class DevelopmentOracleTests(unittest.TestCase):
    def fixture(self):
        sample = dict(token="q", session_id="s", role="development", selected_local=np.array(0),
                      selection_scores=np.array([3., 2., 1., 0.]), candidate_ids=np.arange(4))
        # A good action outside top-k is already evaluated: it must not enter the top-k oracle.
        saved = {"q": {"decisions": {"baseline": 0}, "outcomes": {"0": outcome(0), "3": outcome(3)}}}
        return [sample], saved

    def run_audit(self, root, samples, saved, evaluate, k, dry=False, provenance=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return audit(root, samples, saved, evaluate, k, provenance or {"source": "frozen"},
                         read_json, write_json, dry_run=dry)

    def test_plan_has_no_simulator_calls_or_file_writes(self):
        samples, saved = self.fixture()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            plan = self.run_audit(root, samples, saved, lambda *a: self.fail("Simulation during plan"), 2, True)
            self.assertEqual(plan["new_candidate_evaluations"], 1)
            self.assertEqual(plan["raw_filtered_scorer_calls"], 4)
            self.assertEqual(list(root.iterdir()), [])

    def test_resume_expand_safety_and_outside_topk_exclusion(self):
        samples, saved = self.fixture()
        calls = []

        def evaluate(token, index):
            calls.append(index)
            return outcome(index), None

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            first = self.run_audit(root, samples, saved, evaluate, 2)
            self.assertEqual(calls, [0, 1])
            self.assertAlmostEqual(first["raw_oracle"]["gain_mean_pp"], 20.)
            self.assertEqual(first["raw_safe_oracle"]["gain_mean_pp"], 0.)
            calls.clear()
            repeated = self.run_audit(root, samples, saved, evaluate, 2)
            self.assertEqual(calls, [0])
            self.assertEqual(repeated, first)
            calls.clear()
            expanded = self.run_audit(root, samples, saved, evaluate, 3)
            self.assertEqual(calls, [0, 2])
            self.assertAlmostEqual(expanded["raw_safe_oracle"]["gain_mean_pp"], 40.)
            self.assertAlmostEqual(expanded["raw_oracle"]["gain_mean_pp"], 40.)
            self.assertAlmostEqual(expanded["topk_mean_gain_pp"]["1"], 0.)
            with self.assertRaisesRegex(RuntimeError, "inputs/code changed"):
                self.run_audit(root, samples, saved, evaluate, 3, provenance={"source": "changed"})

    def test_interruption_resumes_completed_candidates(self):
        samples, saved = self.fixture()

        def interrupted(token, index):
            if index == 2:
                raise RuntimeError("Interrupted")
            return outcome(index), None

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "Interrupted"):
                self.run_audit(root, samples, saved, interrupted, 3)
            state = read_json(root / "development_oracle/scenes/q.json")
            self.assertIn("1", state["outcomes"])
            calls = []

            def resume(token, index):
                calls.append(index)
                return outcome(index), None

            self.run_audit(root, samples, saved, resume, 3)
            self.assertEqual(calls, [0, 2])

    def test_parity_failure_and_heldout_roles_are_rejected(self):
        samples, saved = self.fixture()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "Baseline parity failure"):
                self.run_audit(root, samples, saved, lambda *a: (outcome(1), None), 2)
            for role in ("test", "calibration", "memory", "fit"):
                samples[0]["role"] = role
                with self.assertRaisesRegex(ValueError, "DEVELOPMENT-only"):
                    self.run_audit(root, samples, saved, lambda *a: self.fail("Held-out call"), 2)

    def test_k_validation_and_baseline_tie(self):
        samples, _ = self.fixture()
        with self.assertRaises(ValueError):
            ranked_indices(samples[0], 5)
        outcomes = {"0": outcome(0), "1": outcome(0)}
        self.assertEqual(best_action(outcomes, [1, 0], 0), 0)


if __name__ == "__main__":
    unittest.main()
