import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from navsim.planning.script.run_drivemem_validation_oracle import adapt
from navsim.planning.script.run_drivemem_development_oracle import audit, METRICS


class ValidationOracleTests(unittest.TestCase):
    def test_adapter_rejects_changed_decisions_and_incomplete_scoring(self):
        sample = dict(token="q", selected_local=np.array(0))
        choices = dict(baseline=0, B3=3)
        state = dict(token="q", choices=choices, outcomes={"0": {}, "3": {}})
        self.assertEqual(adapt(sample, state, choices)["decisions"], choices)
        with self.assertRaisesRegex(RuntimeError, "decisions/token"):
            adapt(sample, state, dict(baseline=0, B3=2))
        del state["outcomes"]["3"]
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            adapt(sample, state, choices)

    def test_frozen_actions_reused_without_mutation_or_outside_topk_selection(self):
        sample = dict(token="q", selected_local=np.array(0), role="development", session_id="new",
                      candidate_ids=np.arange(16), selection_scores=np.arange(16., 0., -1))

        def outcome(i):
            value = {k: 1. for k in METRICS}
            value.update(pdm_score=.5 + .01 * i, aggregate_without_two_frame=.5 + .01 * i)
            return dict(raw=dict(value), filtered=dict(value))

        state = dict(token="q", choices=dict(baseline=0, B3=3), outcomes={"0": outcome(0), "3": outcome(3)})
        original = copy.deepcopy(state)
        saved = {"q": adapt(sample, state, state["choices"])}
        calls = []

        def evaluate(token, index):
            calls.append(index)
            return outcome(index), None

        def read(path):
            return json.loads(path.read_text())

        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            plan = audit(root, [sample], saved, evaluate, 2, {"source": "frozen"}, read, write, dry_run=True)
            self.assertEqual(plan["new_candidate_evaluations"], 1)
            self.assertEqual(calls, [])
            self.assertEqual(list(root.iterdir()), [])
            report = audit(root, [sample], saved, evaluate, 2, {"source": "frozen"}, read, write)
            self.assertAlmostEqual(report["raw_oracle"]["gain_mean_pp"], 1.)
            calls.clear()
            report = audit(root, [sample], saved, evaluate, 8, {"source": "frozen"}, read, write)
            self.assertNotIn(3, calls)
            self.assertAlmostEqual(report["raw_oracle"]["gain_mean_pp"], 7.)
            self.assertEqual(state, original)


if __name__ == "__main__":
    unittest.main()
