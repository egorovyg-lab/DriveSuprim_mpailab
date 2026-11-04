import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from navsim.agents.drivemem import METRICS
from navsim.planning.script.run_drivemem_memory_completion import fill_memory, labelled, fit_models, VARIANTS
from navsim.planning.script.run_drivemem_development_oracle import verify_baseline
from test_drivemem_action_transfer import make_sample


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome(i):
    values = {k: 1. for k in METRICS}
    values["ego_progress"] = .2 + .05 * i
    score = (5 * values["ego_progress"] + 9) / 14
    values.update(pdm_score=score, aggregate_without_two_frame=score)
    return dict(raw=dict(values), filtered=dict(values))


class MemoryCompletionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(8)
        self.vocab = self.rng.normal(size=(16, 40, 3))
        self.old = [make_sample("m1", "m1", "memory", self.rng), make_sample("m2", "m2", "memory", self.rng)]
        for s in self.old:
            s["label_values"] = np.array([[outcome(i)["raw"][k] for k in METRICS] for i in range(8)])

    def test_only_memory_is_scored_and_completed_candidates_resume(self):
        original = {s["token"]: {str(i): outcome(i) for i in range(8)} for s in self.old}
        targets = copy.deepcopy(original)
        calls, failed = [], [False]

        def evaluate(token, i):
            self.assertTrue(token.startswith("m"))
            calls.append((token, i))
            if i == 10 and not failed[0]:
                failed[0] = True
                raise RuntimeError("interrupted")
            return outcome(i), None

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                fill_memory(self.old, targets, evaluate, root, read, write, verify_baseline)
            self.assertIn("9", read(root / "memory_labels/m1.json")["outcomes"])
            fill_memory(self.old, targets, evaluate, root, read, write, verify_baseline)
            self.assertEqual(calls.count(("m1", 8)), 1)
            new = [labelled(s, targets[s["token"]]) for s in self.old]
            self.assertEqual(len(new[0]["label_local_indices"]), 16)
            self.assertEqual(len(self.old[0]["label_local_indices"]), 8)
            for token, values in original.items():
                for key, value in values.items():
                    self.assertEqual(targets[token][key], value)
            bad = [dict(self.old[0], role="development")]
            with self.assertRaisesRegex(ValueError, "Only M"):
                fill_memory(bad, targets, evaluate, root, read, write, verify_baseline)

    def test_factorial_fit_freezes_h_and_cannot_read_query_outcomes(self):
        outcomes = {s["token"]: {str(i): outcome(i) for i in range(16)} for s in self.old}
        new = [labelled(s, outcomes[s["token"]]) for s in self.old]
        f = [make_sample("f1", "f1", "fit", self.rng), make_sample("f2", "f2", "fit", self.rng)]
        targets = {s["token"]: {str(i): outcome(i) for i in range(16)} for s in f}
        f = [labelled(s, targets[s["token"]]) for s in f]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            for token, values in outcomes.items():
                write(root / "memory_labels" / (token + ".json"), dict(token=token, outcomes=values))
            args = (self.old, new, f, targets, self.vocab, .001, 1, root, read, write,
                    lambda path, **data: np.savez(path, **data), sha)
            result = fit_models(*args)
            self.assertEqual(result["h"], .001)
            self.assertEqual(result["beta"], .5)
            self.assertEqual(set(result["fit_trials"]), set(VARIANTS))
            self.assertEqual(result["old_memory_labels"], 16)
            self.assertEqual(result["new_memory_labels"], 32)
            self.assertEqual(fit_models(*args), result)
            bad_f = [dict(f[0], role="development")]
            with self.assertRaisesRegex(ValueError, "Invalid M/F"):
                fit_models(self.old, new, bad_f, targets, self.vocab, .001, 1, root, read, write, None, sha)
            path = root / "memory_labels/m1.json"
            state = read(path)
            state["outcomes"]["15"]["raw"]["ego_progress"] = .1
            write(path, state)
            with self.assertRaisesRegex(RuntimeError, "completed M/model changed"):
                fit_models(*args)


if __name__ == "__main__":
    unittest.main()
