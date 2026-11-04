import copy
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch

import numpy as np

from test_drivemem_action_transfer import make_sample
from navsim.agents import drivemem_pair_utility as utility
from navsim.planning.script import run_drivemem_pair_utility as runner
from navsim.agents.drivemem import METRICS


def outcomes(sample, rng):
    result = {}
    for i in range(16):
        value = dict(zip(METRICS, rng.uniform(.5, 1, len(METRICS))))
        value["pdm_score"] = float(rng.uniform(.2, 1))
        result[str(i)] = dict(raw=value, filtered=value.copy())
    return result


class PairUtilityTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(94)
        self.vocab = self.rng.normal(size=(16, 40, 3))
        self.query = make_sample("q", "q", "development", self.rng)
        self.bank = []
        for i in range(3):
            s = make_sample("m" + str(i), "m" + str(i), "memory", self.rng)
            self.bank.append(utility.attach_utility(s, outcomes(s, self.rng)))

    def test_zero_baseline_and_no_query_label_consumption(self):
        a = utility.features(self.query, self.vocab, self.bank, .1)
        np.testing.assert_array_equal(a[0], 0)
        changed = dict(self.query, label_values=np.full((8, 8), np.nan),
                       utility_values=np.full(16, np.nan))
        np.testing.assert_array_equal(a, utility.features(changed, self.vocab, self.bank, .1))
        model = utility.ridge_path([a], [np.arange(16.) / 16], [10.])[10.]
        self.assertEqual(utility.predict(a, model)[0], 0)
        self.assertEqual(utility.choose(self.query, np.zeros(16)), 0)
        self.assertEqual(utility.choose(self.query, -np.ones(16)), 0)
        self.assertEqual(utility.choose(self.query, np.arange(16.)), 15)

    def test_own_entire_session_excluded_and_no_donor_fallback(self):
        a = utility.recall_features(self.bank[0], self.bank, self.vocab, .1, 1)
        poisoned = copy.deepcopy(self.bank)
        poisoned[0]["utility_values"][:] = np.nan
        # A second scene from the same session must also be excluded.
        poisoned.append(dict(poisoned[0], token="m0-another"))
        np.testing.assert_allclose(a, utility.recall_features(self.bank[0], poisoned, self.vocab, .1, 1))
        np.testing.assert_array_equal(utility.recall_features(self.bank[0], poisoned[:1], self.vocab, .1, 1), 0)
        with self.assertRaises(ValueError):
            utility.recall_features(self.query, [self.query], self.vocab, .1, 1)

    def test_dual_ridge_equals_weighted_primal_and_train_only_scale(self):
        xs = [self.rng.normal(size=(7, 4)), self.rng.normal(size=(3, 4))]
        ys = [self.rng.normal(size=7), self.rng.normal(size=3)]
        model = utility.ridge_path(xs, ys, [3.])[3.]
        x, y = np.concatenate(xs), np.concatenate(ys)
        w = np.r_[np.full(7, 1 / 7), np.full(3, 1 / 3)]
        w /= w.mean()
        expected_scale = np.sqrt(np.average(x * x, axis=0, weights=w))
        np.testing.assert_allclose(model["scale"], expected_scale)
        z = x / expected_scale
        coef = np.linalg.solve(z.T @ (w[:, None] * z) + 3 * np.eye(4), z.T @ (w * y))
        np.testing.assert_allclose(model["coef"], coef, atol=1e-11)
        before = model["scale"].copy()
        utility.predict(np.full((2, 4), 1e6), model)
        np.testing.assert_array_equal(model["scale"], before)

    def test_fit_freeze_and_matched_supervision(self):
        fitting, targets = [], {}
        for i in range(2):
            s = make_sample("f" + str(i), "f" + str(i), "fit", self.rng)
            targets[s["token"]] = outcomes(s, self.rng)
            fitting.append(utility.attach_utility(s, targets[s["token"]]))
        def read(p):
            return json.loads(p.read_text())
        def write(p, v):
            p.write_text(json.dumps(v))
        def sha(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp)
            args = (self.bank, fitting, targets, self.vocab, 1, root, read, write, np.savez, sha)
            fitted = runner.fit_models(*args)
            self.assertEqual(set(fitted["models"]), set(runner.METHODS))
            self.assertEqual(fitted, runner.fit_models(*args))
            (root / "direct_memory.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "model changed"):
                runner.fit_models(*args)

    def test_labels_require_top16_and_valid_scores(self):
        values = outcomes(self.query, self.rng)
        bad = copy.deepcopy(values)
        del bad["15"]
        with self.assertRaises(ValueError):
            utility.attach_utility(self.query, bad)
        bad = copy.deepcopy(values)
        bad["1"]["raw"]["pdm_score"] = float("nan")
        with self.assertRaises(ValueError):
            utility.attach_utility(self.query, bad)
        s = utility.attach_utility(self.query, values)
        self.assertEqual(s["utility_values"][0], 0)
        self.assertAlmostEqual(s["utility_values"][1], values["1"]["raw"]["pdm_score"] - values["0"]["raw"]["pdm_score"])

    def test_pipeline_freezes_both_cohorts_before_lookup_and_resumes(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            base = Path(tmp).resolve()
            source, validation = base / "source", base / "validation"
            previous, completion, out = base / "previous", base / "completion", base / "utility"
            def write(p, v):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(v))
            def sha(p):
                return hashlib.sha256(p.read_bytes()).hexdigest()
            def read(p):
                if "development_oracle" in str(p):
                    self.assertTrue((out / "fit.json").is_file())
                    frozen = json.loads((out / "query_decisions.json").read_text())
                    self.assertEqual(set(frozen["cohorts"]), {"source", "validation"})
                return json.loads(p.read_text())
            fitting = [make_sample("f" + str(i), "f" + str(i), "fit", self.rng) for i in range(2)]
            samples = self.bank + fitting
            for s in samples:
                root = completion / "memory_labels" if s["role"] == "memory" else previous / "fit_labels"
                write(root / (s["token"] + ".json"), dict(token=s["token"], outcomes=outcomes(s, self.rng)))
            hashes = lambda root: {p.name: sha(p) for p in root.glob("*.json")}
            write(previous / "identity.json", dict(seed=1, sources={}, inputs={}, labels={}))
            write(previous / "fit.json", dict(sidecar_labels=hashes(previous / "fit_labels")))
            query2 = make_sample("q2", "q2", "development", self.rng)
            by_token = {s["token"]: s for s in samples + [self.query, query2]}
            oracle_inputs = {}
            for root, query in ((source, self.query), (validation, query2)):
                rows = [dict(token=query["token"], session_id=query["session_id"], role="development")]
                rows += [dict(token="unread_" + r, session_id=r, role=r) for r in ("calibration", "test")]
                write(root / "splits.json", dict(records=rows))
                (root / "inventory.csv").write_text("inventory")
                write(root / "split_identity.json", dict(splits=sha(root / "splits.json"), inventory=sha(root / "inventory.csv")))
                write(root / "identity.json", {})
                write(root / "export_index.json", {})
                np.savez(root / "vocab.npz", vocab=self.vocab)
                path = root / "development_oracle/scenes" / (query["token"] + ".json")
                write(path, dict(token=query["token"], outcomes=outcomes(query, self.rng)))
                oracle_inputs[root.name] = {query["token"]: sha(path)}
            write(completion / "fit.json", dict(memory_labels=hashes(completion / "memory_labels")))
            write(completion / "report.json", dict(oracle_inputs=oracle_inputs))
            repo = Path(runner.__file__).resolve().parents[3]
            write(completion / "identity.json", dict(source=str(source), previous=str(previous), validation=str(validation),
                previous_identity=sha(previous / "identity.json"), previous_fit=sha(previous / "fit.json"),
                validation_inputs={}, runner_sha256=sha(repo / "navsim/planning/script/run_drivemem_memory_completion.py")))
            fake = N(verify_identity=lambda *a: None, read_json=read, write_json=write, sha=sha,
                training_samples=lambda *a: copy.deepcopy(samples), save_npz=np.savez,
                sample=lambda root, r: copy.deepcopy(by_token[r["token"]]))
            cfg = N(drivemem=N(root=str(source), stage="utility-plan", memory_mode="enriched"))
            env = dict(NAVSIM_EXP_ROOT=str(base), DRIVEMEM_COMPLETION_ROOT=str(completion), DRIVEMEM_UTILITY_ROOT=str(out))
            with patch.dict(sys.modules, {"navsim.planning.script.run_drivemem_memory": fake}), patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
                runner.main(cfg)
                self.assertFalse(out.exists())
                cfg.drivemem.stage = "utility-all"
                runner.main(cfg)
                original_fit = sha(out / "fit.json")
                runner.main(cfg)
                self.assertEqual(sha(out / "fit.json"), original_fit)
                self.assertEqual(set(read(out / "report.json")["cohorts"]), {"source", "validation"})
                write(validation / "export_index.json", {"changed": True})
                with self.assertRaisesRegex(RuntimeError, "protocol/inputs changed"):
                    runner.main(cfg)


if __name__ == "__main__":
    unittest.main()
