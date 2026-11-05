import contextlib
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from navsim.agents import drivemem_component_utility as model
from navsim.planning.script import run_drivemem_component_utility as runner
from test_drivemem_action_transfer import make_sample
from test_drivemem_pair_utility import outcomes


class ComponentTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(98)
        self.vocab = self.rng.normal(size=(16, 40, 3))
        self.q = make_sample("q", "q", "development", self.rng)
        self.bank = []
        for i in range(3):
            s = make_sample("m" + str(i), "m" + str(i), "memory", self.rng)
            self.bank.append(model.attach(s, outcomes(s, self.rng)))

    def test_absolute_features_no_query_labels_and_no_self_leak_even_shuffled(self):
        x, supported = model.features(self.q, self.vocab, self.bank, .1)
        self.assertTrue(supported)
        self.assertGreater(np.linalg.norm(x[0]), 0)
        poisoned = dict(self.q, label_values=np.full((8, 8), np.nan), outcome_values=np.full((16, 9), np.nan))
        np.testing.assert_array_equal(x, model.features(poisoned, self.vocab, self.bank, .1)[0])
        for shuffle in (False, True):
            a = model.recall(self.bank[0], self.bank, self.vocab, .1, 1, shuffle=shuffle)[0]
            bad = copy.deepcopy(self.bank)
            bad[0]["memory_values"][:] = np.nan
            np.testing.assert_array_equal(a, model.recall(self.bank[0], bad, self.vocab, .1, 1, shuffle=shuffle)[0])
        recalled, ok = model.recall(self.bank[0], self.bank[:1], self.vocab, .1, 1)
        self.assertFalse(ok)
        np.testing.assert_array_equal(recalled, 0)
        shuffled = model.shuffled_bank(self.bank, 1)
        for a, b in zip(self.bank, shuffled):
            np.testing.assert_array_equal(a["outcome_values"], b["outcome_values"])

    def test_multioutput_dual_equals_primal_and_predictions_bounded(self):
        xs = [self.rng.normal(size=(7, 4)), self.rng.normal(size=(3, 4))]
        ys = [self.rng.uniform(0, 1, (7, 9)), self.rng.uniform(0, 1, (3, 9))]
        fitted = model.ridge_path(xs, ys, [3.])[3.]
        x, y = np.concatenate(xs), np.clip(np.concatenate(ys), model.EPS, 1 - model.EPS)
        y = np.log(y / (1 - y))
        w = np.r_[np.full(7, 1 / 7), np.full(3, 1 / 3)]
        w /= w.mean()
        z = (x - fitted["mean"]) / fitted["scale"]
        expected = np.linalg.solve(z.T @ (w[:, None] * z) + 3 * np.eye(4), z.T @ (w[:, None] * (y - fitted["ym"])))
        np.testing.assert_allclose(fitted["coef"], expected, atol=1e-11)
        prediction = model.predict(x, fitted)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())

    def test_no_safety_tradeoff_teacher_guard_ties_and_support(self):
        q = copy.deepcopy(self.q)
        for k in model.METRICS:
            q["head_" + k][:] = 0
        p = np.full((16, 9), .9)
        p[:, 0] = .3
        p[0, 0], p[1, 0] = .4, .8
        p[1, 3] = .2  # DDC regression cannot be paid for by Q improvement.
        self.assertEqual(model.choose(q, p, "raw"), 1)
        self.assertEqual(model.choose(q, p, "component"), 0)
        p[1, 3] = .9
        q["head_driving_direction_compliance"][1] = -2
        self.assertEqual(model.choose(q, p, "component"), 1)
        self.assertEqual(model.choose(q, p, "guarded"), 0)
        self.assertEqual(model.choose(q, p, "raw", supported=False), 0)
        p[:, 0] = .4
        self.assertEqual(model.choose(q, p, "raw"), 0)

    def test_f_cv_separates_uniform_quality_from_challenge_harm(self):
        rows = [dict(slice="uniform", session_id="a", raw_gain_pp=1., raw_harm=False, intervention=True),
                dict(slice="challenge", session_id="a", raw_gain_pp=-100., raw_harm=True, intervention=True)]
        score = runner.cv_score(rows)
        self.assertEqual(score["uniform_gain_macro_pp"], 1.)
        self.assertEqual(score["harm"], 1)

    def test_end_to_end_freeze_before_oracles_and_model_tamper(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            source, val, comp, addition, out = [root / n for n in ("source", "val", "comp", "addition", "out")]
            samples, targets = [], {}
            for i in range(4):
                s = make_sample(str(i), str(i), "memory" if i < 2 else "fit", self.rng)
                targets[s["token"]] = outcomes(s, self.rng)
                samples.append(s)
            runner.write(addition / "label_selection.json", dict(records=[]))
            expected, queries = {}, {}
            for n, cohort in enumerate((source, val)):
                q = make_sample("q" + str(n), "q" + str(n), "development", self.rng)
                queries[q["token"]] = q
                runner.write(cohort / "splits.json", dict(records=[dict(token=q["token"], session_id=q["session_id"], role="development"), dict(token="closed", role="test")]))
                runner.write(cohort / "export_index.json", {})
                path = cohort / "development_oracle/scenes" / (q["token"] + ".json")
                runner.write(path, dict(token=q["token"], outcomes=outcomes(q, self.rng)))
                expected[cohort.name] = {q["token"]: runner.sha(path)}
            runner.write(comp / "report.json", dict(oracle_inputs=expected))
            data = (samples[:2], samples[2:], targets, self.vocab, source, val, comp, {}, {}, 1)
            original = runner.read
            def read(p):
                if "development_oracle" in str(p):
                    self.assertTrue((out / "fit.json").exists())
                    self.assertEqual(set(original(out / "query_decisions.json")["cohorts"]), {"source", "val"})
                return original(p)
            argv = ["runner", "plan", "--prior", str(root / "prior"), "--addition", str(addition), "--out", str(out)]
            with patch.object(runner, "load_inputs", return_value=data), patch.object(runner, "read", side_effect=read), patch.object(runner, "load_sample", side_effect=lambda r, row, ix: copy.deepcopy(queries[row["token"]])), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
                self.assertFalse(out.exists())
                argv[1] = "all"
                runner.main()
                digest = runner.sha(out / "fit.json")
                runner.main()
                self.assertEqual(runner.sha(out / "fit.json"), digest)
                self.assertEqual(len(original(out / "report.json")["cohorts"]["source"]["methods"]), 13)
                (out / "memory.npz").write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "model changed"):
                    runner.main()


if __name__ == "__main__":
    unittest.main()
