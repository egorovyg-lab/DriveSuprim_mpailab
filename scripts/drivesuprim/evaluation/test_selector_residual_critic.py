import copy
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from navsim.agents import selector_residual_critic as model
from navsim.planning.script import run_selector_residual_night as pipeline
from test_selector_bundle import fixture as bundle_fixture


def tiny_entries(sessions=3):
    entries = []
    rng = np.random.default_rng(7)
    for i in range(sessions * 2):
        x = rng.normal(0, .002, (16, 5)).astype(np.float32)
        x[:, 0], x[3, 0], x[5, 0] = 0., 1., -1.
        gain = np.zeros(16, dtype=np.float32)
        gain[3], gain[5] = .1, -.5
        risk = np.zeros((16, 5), dtype=np.float32)
        risk[5, 1] = 1
        sample = dict(token="t" + str(i), session_id="s" + str(i // 2), role="fit",
                      candidate_ids=np.arange(16), selected_local=np.array(0))
        outcomes = {}
        for j in range(16):
            raw = dict(pdm_score=.6 + float(gain[j]), **{k: 1. for k in pipeline.METRICS})
            if j == 5:
                raw["drivable_area_compliance"] = 0.
            outcomes[str(j)] = dict(raw=raw, filtered=raw.copy())
        entries.append(dict(sample=sample, metadata=dict(token=sample["token"], role="train", origin="synthetic"),
            baseline=0, indices=np.arange(16), x=x, teacher_gap=-np.arange(16, dtype=np.float32) * .001,
            targets=dict(gain=gain, quality=gain + .6, risk=risk), outcomes=outcomes, sampling_slice="uniform"))
    return entries


class CriticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_residual_preserves_teacher_baseline_and_bounds(self):
        config = model.CriticConfig(width=8, epochs=1)
        data = tiny_entries()
        stats = model.fit_statistics(data, config)
        for variant in model.VARIANTS:
            net = model.ResidualCritic(5, variant, config)
            pred = model.predict(net, data, stats)
            combined = model.aggregate([pred, pred])
            for i, e in enumerate(data):
                p, _ = model.choose(e, combined, i, model.policies()["raw"], np.ones(5) * 3, config)
                self.assertEqual(p, e["baseline"])
            with torch.no_grad():
                net.readout.weight[0].fill_(1e3)
            changed = model.predict(net, data, stats)
            self.assertLessEqual(float(np.max(np.abs(changed["correction"]))), config.residual_cap + 1e-6)
            self.assertTrue(np.all(changed["gain"][:, 0] == 0))

    def test_permutation_equivariance_padding_and_baseline_validation(self):
        config = model.CriticConfig(width=8, epochs=1)
        torch.manual_seed(4)
        net = model.ResidualCritic(5, "mlp", config)
        with torch.no_grad():
            net.readout.weight.normal_(0, .2)
        x, base = torch.randn(2, 16, 5), torch.tensor([1, 7])
        gap = -torch.rand(2, 16)
        gap.scatter_(1, base[:, None], 0.)
        order = torch.randperm(16)
        pb = torch.stack([(order == b).nonzero()[0, 0] for b in base])
        before, after = net(x, base, gap, .1), net(x[:, order], pb, gap[:, order], .1)
        padded = net(torch.cat([x, torch.ones((2, 2, 5)) * 999], 1), base,
                     torch.cat([gap, torch.zeros((2, 2))], 1), .1, torch.tensor([[True] * 16 + [False] * 2] * 2))
        for key in before:
            torch.testing.assert_close(before[key][:, order], after[key], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(before[key], padded[key][:, :16], atol=1e-6, rtol=1e-5)
        with self.assertRaisesRegex(ValueError, "baseline"):
            net(x, base, gap, .1, torch.zeros((2, 16), dtype=torch.bool))

    def test_filtering_happens_before_argmax_and_support_is_train_only(self):
        e, config = tiny_entries()[0], model.CriticConfig()
        gain = np.full((1, 16), -.1)
        gain[0, :3] = [0, .08, .05]
        risk = np.zeros((1, 16, 5)) + .001
        risk[0, 1, 1] = .99
        pred = dict(gain=gain, gain_lower=gain.copy(), risk=risk)
        policy = dict(margin=.005, risk_limit=.05, uncertainty=True)
        p, info = model.choose(e, pred, 0, policy, np.ones(5) * 3, config)
        self.assertEqual(p, 2)
        self.assertEqual(info["eligible_alternatives"], 1)
        p, info = model.choose(e, pred, 0, policy, [3, 3, 3, 1, 3], config)
        self.assertEqual(p, 0)
        self.assertTrue(info["support_blocked"])
        p, _ = model.choose(e, pred, 0, model.policies()["raw"], [0] * 5, config)
        self.assertEqual(p, 1)

    def test_ensemble_uses_all_seeds_and_disagreement_is_not_hidden(self):
        a = dict(gain=np.array([[0., .1, .03]]), risk_logits=np.zeros((1, 3, 5)))
        b = dict(gain=np.array([[0., -.1, .03]]), risk_logits=np.ones((1, 3, 5)) * 2)
        result = model.aggregate([a, b])
        self.assertAlmostEqual(result["gain"][0, 1], 0.)
        self.assertAlmostEqual(result["gain_lower"][0, 1], -.1)
        self.assertAlmostEqual(result["gain_lower"][0, 2], .03)
        np.testing.assert_allclose(result["risk"], 1 / (1 + np.exp(-2)))

    def test_equal_session_scaling_class_weights_and_roles(self):
        config = model.CriticConfig()
        data = tiny_entries()
        for e in data:
            e["x"].fill(0 if e["sample"]["session_id"] == "s0" else 10)
        uneven = [data[0], data[1], data[2]]
        stats = model.fit_statistics(uneven, config)
        np.testing.assert_allclose(stats["mean"], 5.)
        self.assertEqual(stats["risk_support_sessions"].tolist(), [0, 2, 0, 0, 0])
        self.assertTrue(np.all(stats["positive_weight"] <= config.positive_weight_cap))
        self.assertGreaterEqual(float(stats["prior_slope"]), 0.)
        invalid = copy.deepcopy(uneven)
        invalid[0]["metadata"]["role"] = "holdout"
        with self.assertRaisesRegex(ValueError, "training"):
            model.fit_statistics(invalid, config)

    def test_tiny_training_learns_gain_and_risk_and_roundtrips(self):
        config = model.CriticConfig(width=8, epochs=60, learning_rate=.02)
        data = tiny_entries()
        for variant in model.VARIANTS:
            net, stats, history = model.train(data, variant, 11, config)
            pred = model.predict(net, data, stats)
            self.assertLess(history[-1]["loss"], history[0]["loss"])
            self.assertTrue(np.all(pred["gain"][:, 3] > 0))
            self.assertTrue(np.all(pred["gain"][:, 5] < 0))
            self.assertTrue(np.all(pred["risk_logits"][:, 5, 1] > pred["risk_logits"][:, 3, 1]))
            restored, other = model.restore(model.checkpoint(net, stats), variant, config)
            repeated = model.predict(restored, data, other)
            for key in pred:
                np.testing.assert_array_equal(pred[key], repeated[key])

    def test_fold_excludes_whole_session_never_reads_heldout_targets_and_resumes(self):
        data, config = tiny_entries(), model.CriticConfig(width=8, epochs=1)
        class NoTargets:
            def __getitem__(self, key):
                raise AssertionError("Validation outcomes consumed by training/prediction")
        for e in data:
            if e["sample"]["session_id"] == "s0":
                e["targets"] = NoTargets()
                e["x"].fill(1e6)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp).resolve()
            pipeline.write(out / "identity.json", dict(frozen=True))
            with redirect_stdout(io.StringIO()):
                valid, first, stats = pipeline.fold(data, "linear", 11, "s0", config, out)
            self.assertEqual(len(valid), 2)
            self.assertLess(float(stats["mean"].max()), 100)
            self.assertEqual(stats["risk_support_sessions"].tolist(), [0, 2, 0, 0, 0])
            with mock.patch.object(model, "train", side_effect=AssertionError("must resume")):
                _, repeated, _ = pipeline.fold(data, "linear", 11, "s0", config, out)
            np.testing.assert_array_equal(first["gain"], repeated["gain"])
            path = next((out / "cv").glob("*.npz"))
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(RuntimeError, "fold changed"):
                pipeline.fold(data, "linear", 11, "s0", config, out)

    def test_run_lock_prevents_concurrent_writers_and_releases(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp)
            with pipeline.locked(out):
                with self.assertRaisesRegex(RuntimeError, "Another critic process"):
                    with pipeline.locked(out):
                        pass
            with pipeline.locked(out):
                pass

    def test_real_bundle_loader_checks_reference_outcomes_before_any_fit(self):
        config = model.CriticConfig(width=8, epochs=1)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            base = Path(tmp).resolve()
            root, addition = base / "bundle", base / "addition"
            pipeline.write(addition / "label_selection.json", dict(records=[]))
            examples, rows = [], []
            for i in range(3):
                sample, outcomes, vocab, metadata = bundle_fixture("t" + str(i), "s" + str(i))
                examples.append(pipeline.bundles.example(sample, outcomes, vocab, metadata))
                query = dict(sample, token=metadata["token"], session_id=metadata["session_id"])
                rows.append(dict(method="control_v1_raw", **pipeline.record(query, outcomes, 0)))
            with redirect_stdout(io.StringIO()):
                pipeline.bundles.build(root, examples, vocab,
                    {str(addition / "label_selection.json"): pipeline.sha(addition / "label_selection.json")}, {"closed"})
            frame = pipeline.pd.DataFrame(rows)
            fitted = dict(config=dict(epochs=1), cv_methods={})
            runtime = dict(python=pipeline.platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                           platform=pipeline.platform.system(), threads=torch.get_num_threads())
            with mock.patch.object(pipeline, "load_run", return_value=(runtime, fitted, frame)):
                # The loader pins the completed reference manifests after checking
                # outcomes. Keep the synthetic reference deliberately minimal.
                for name in ("identity.json", "fit.json", "fit_cv_per_scene.csv"):
                    path = base / "reference" / name
                    path.parent.mkdir(exist_ok=True)
                    path.write_text("frozen", encoding="utf-8")
                loaded, inputs, _ = pipeline.load_data(root, base / "reference", addition, config)
                self.assertEqual(len(loaded), 3)
                self.assertIn(str(root / "bundle_manifest.json"), inputs)
                frame.loc[0, "raw_score"] += .1
                with self.assertRaisesRegex(ValueError, "Reference outcome mismatch"):
                    pipeline.load_data(root, base / "reference", addition, config)

    def test_interrupted_study_resumes_and_frozen_config_and_models_are_checked(self):
        data, config = tiny_entries(), model.CriticConfig(width=8, epochs=1)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp).resolve() / "study"
            actual_fold, calls = pipeline.fold, []
            def interrupted(*args):
                calls.append(args[3])
                if len(calls) == 3:
                    raise RuntimeError("synthetic interruption")
                return actual_fold(*args)
            with mock.patch.object(pipeline, "fold", side_effect=interrupted), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    pipeline.run(data, config, (11, 12), ("linear",), out, {}, synthetic=True)
            self.assertEqual(pipeline.read(out / "progress.json")["status"], "failed")
            self.assertEqual(len(list((out / "cv").glob("*.json"))), 2)
            with redirect_stdout(io.StringIO()):
                result = pipeline.run(data, config, (11, 12), ("linear",), out, {}, synthetic=True)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["primary"], "baseline")  # No fake success from zero coverage.
            self.assertEqual(pipeline.read(out / "progress.json")["completed_fits"], 8)
            with mock.patch.object(model, "train", side_effect=AssertionError("completed run must resume")):
                repeated = pipeline.run(data, config, (11, 12), ("linear",), out, {}, synthetic=True)
            self.assertEqual(result, repeated)
            with self.assertRaisesRegex(RuntimeError, "configuration changed"):
                pipeline.run(data, replace(config, epochs=2), (11, 12), ("linear",), out, {}, synthetic=True)
            model_path = next((out / "models").glob("*.npz"))
            model_path.write_bytes(model_path.read_bytes() + b"changed")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                pipeline.verify_report(out)


if __name__ == "__main__":
    unittest.main()
