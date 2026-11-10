import copy
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from navsim.agents import selector_component_critic as model
from navsim.planning.script import run_selector_component_critic as pipeline
from test_selector_bundle import fixture


def tiny_entries(sessions=3):
    rng = np.random.default_rng(7)
    entries = []
    for i in range(2 * sessions):
        x = rng.normal(0, .001, (16, 5)).astype(np.float32)
        x[:, 0], x[3, 0], x[5, 0] = 0., 1., -1.
        c = np.ones((16, 8), dtype=np.float64)
        c[:, 4], c[3, 4], c[5, 1] = .5, .95, 0.
        quality = model.pdm(c)
        teacher = np.ones((16, 8), dtype=np.float32) * .8
        teacher[:, 4] = .6
        logits = np.log(teacher / (1 - teacher))
        sample = dict(token="t" + str(i), session_id="s" + str(i // 2), role="fit",
                      candidate_ids=np.arange(16), selected_local=np.array(0))
        outcomes = {}
        for j in range(16):
            raw = dict(pdm_score=float(quality[j]), **dict(zip(pipeline.METRICS, c[j].tolist())))
            outcomes[str(j)] = dict(raw=raw, filtered=raw.copy())
        entries.append(dict(sample=sample, metadata=dict(token=sample["token"], role="train", origin="synthetic"),
            baseline=0, indices=np.arange(16), x=x, teacher_logits=logits,
            targets=dict(components=c.astype(np.float32), quality=quality.astype(np.float32),
                         gain=(quality - quality[0]).astype(np.float32)), outcomes=outcomes, sampling_slice="uniform"))
    return entries


class ComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_formula_known_values_torch_numpy_and_monotonicity(self):
        c = np.ones((4, 8))
        c[1, 0], c[2, 6], c[3, 4] = .5, 0., .7
        np.testing.assert_allclose(model.pdm(c), [1., .5, 12 / 14, 12.5 / 14])
        np.testing.assert_allclose(model.pdm(torch.tensor(c)).numpy(), model.pdm(c))
        rng = np.random.default_rng(2)
        baseline = rng.uniform(0, 1, (100, 8))
        candidate = baseline * rng.uniform(0, 1, (100, 8))
        self.assertTrue(np.all(model.pdm(candidate) <= model.pdm(baseline) + 1e-12))
        with self.assertRaisesRegex(ValueError, "Eight"):
            model.pdm(np.ones((2, 9)))

    def test_formula_raw_only_and_mismatch_stops_before_any_fitting(self):
        entries = tiny_entries()
        entries[0]["outcomes"]["5"]["filtered"]["drivable_area_compliance"] = 1.
        audit, mismatches = model.formula_audit(entries)
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["raw_filtered_component_disagreements"], 1)
        self.assertFalse(mismatches)
        entries[0]["outcomes"]["5"]["raw"]["pdm_score"] = .3
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp).resolve() / "study"
            with mock.patch.object(model, "train", side_effect=AssertionError("must not fit")), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Stored RAW PDM"):
                    pipeline.run(entries, model.ComponentConfig(epochs=1), (11, 12), ("mlp",), out, {}, synthetic=True)
            self.assertEqual(pipeline.read(out / "formula_audit.json")["mismatches"], 1)
            self.assertFalse((out / "cv").exists())
            self.assertEqual(entries[0]["outcomes"]["5"]["raw"]["pdm_score"], .3)

    def test_zero_correction_matches_teacher_formula_not_a_free_gain(self):
        entries = tiny_entries()
        stats = model.fit_statistics(entries)
        teacher = pipeline.teacher_prediction(entries)
        for variant in model.VARIANTS:
            net = model.ComponentCritic(5, variant, model.ComponentConfig(width=8, epochs=1))
            actual = model.predict(net, entries, stats)
            for key in actual:
                np.testing.assert_array_equal(actual[key], teacher[key])
            self.assertTrue(np.all(actual["gain"][:, 0] == 0))

    def test_permutation_equivariance_padding_and_baseline(self):
        torch.manual_seed(7)
        net = model.ComponentCritic(5, "mlp", model.ComponentConfig(width=8, epochs=1))
        with torch.no_grad():
            net.readout.weight.normal_(0, .2)
        x, b, logits = torch.randn(2, 16, 5), torch.tensor([1, 7]), torch.randn(2, 16, 8)
        order = torch.randperm(16)
        pb = torch.stack([(order == index).nonzero()[0, 0] for index in b])
        original, permuted = net(x, b, logits), net(x[:, order], pb, logits[:, order])
        padded = net(torch.cat([x, torch.ones((2, 2, 5)) * 999], 1), b,
            torch.cat([logits, torch.ones((2, 2, 8)) * 999], 1), torch.tensor([[True] * 16 + [False] * 2] * 2))
        for key in original:
            torch.testing.assert_close(original[key][:, order], permuted[key], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(original[key], padded[key][:, :16], atol=1e-6, rtol=1e-5)
        with self.assertRaisesRegex(ValueError, "baseline"):
            net(x, b, logits, torch.zeros((2, 16), dtype=torch.bool))

    def test_training_learns_components_gains_and_checkpoint_roundtrips(self):
        entries = tiny_entries()
        config = model.ComponentConfig(width=8, epochs=80, learning_rate=.02)
        for variant in model.VARIANTS:
            net, stats, history = model.train(entries, variant, 11, config)
            pred = model.predict(net, entries, stats)
            self.assertLess(history[-1]["loss"], history[0]["loss"])
            self.assertTrue(np.all(pred["gain"][:, 3] > 0))
            self.assertTrue(np.all(pred["gain"][:, 5] < 0))
            self.assertTrue(np.all(pred["components"][:, 3, 4] > pred["components"][:, 0, 4]))
            self.assertTrue(np.all(pred["components"][:, 5, 1] < pred["components"][:, 0, 1]))
            restored, other = model.restore(model.checkpoint(net, stats), variant, config)
            repeated = model.predict(restored, entries, other)
            for key in pred:
                np.testing.assert_array_equal(pred[key], repeated[key])

    def test_ensemble_uses_mean_components_and_policy_keeps_baseline_ties(self):
        entries = tiny_entries()[:1]
        a = pipeline.teacher_prediction(entries)
        b = copy.deepcopy(a)
        b["components"][0, 1, 0] = .1
        b["score"] = model.pdm(b["components"])
        b["gain"] = b["score"] - b["score"][:, :1]
        combined = model.aggregate([a, b], entries)
        np.testing.assert_allclose(combined["score"], model.pdm((a["components"].astype(float) + b["components"]) / 2))
        self.assertTrue(np.all(combined["gain_lower"] <= combined["gain"]))
        tied = model.aggregate([a], entries)
        for policy in pipeline.POLICIES:
            self.assertEqual(model.choose(entries[0], tied, 0, policy), 0)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            model.choose(entries[0], tied, 0, "unknown")

    def test_matched_comparison_uses_same_tokens_and_equal_session_weight(self):
        rows = [dict(method=pipeline.FIXED_CANDIDATE, token=t, session_id=s, raw_gain_pp=g)
                for t, s, g in (("a0", "a", 1.), ("a1", "a", 1.), ("b0", "b", -2.))]
        controls = [dict(r, method="reference/mlp/ensemble/gain_m020", raw_gain_pp=0.) for r in rows]
        comparison = next(iter(pipeline.paired_comparisons(rows, controls, 11).values()))
        self.assertEqual(comparison["delta_micro_pp"], 0.)
        self.assertEqual(comparison["delta_macro_pp"], -.5)
        self.assertEqual(len(comparison["delta_ci95_sessions"]), 2)
        with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
            pipeline.paired_comparisons(rows, controls[:-1], 11)

    def test_train_only_scaling_and_whole_session_fold_do_not_read_validation_targets(self):
        data = tiny_entries()
        for e in data:
            e["x"].fill(0 if e["sample"]["session_id"] == "s0" else 10)
        stats = model.fit_statistics([data[0], data[1], data[2]])
        np.testing.assert_allclose(stats["mean"], 5.)
        class NoLabels:
            def __getitem__(self, key):
                raise AssertionError("Held-out labels consumed by training/inference")
        for e in data:
            if e["sample"]["session_id"] == "s0":
                e["targets"], e["outcomes"] = NoLabels(), NoLabels()
                e["x"].fill(1e6)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp)
            pipeline.write(out / "identity.json", dict(frozen=True))
            with redirect_stdout(io.StringIO()):
                _, first = pipeline.fold(data, "linear", 11, "s0", model.ComponentConfig(width=8, epochs=1), out)
            with mock.patch.object(model, "train", side_effect=AssertionError("must resume")):
                _, second = pipeline.fold(data, "linear", 11, "s0", model.ComponentConfig(width=8, epochs=1), out)
            np.testing.assert_array_equal(first["components"], second["components"])
            with np.load(next((out / "cv").glob("*.npz")), allow_pickle=False) as f:
                self.assertLess(float(f["stats_mean"].max()), 100.)
        data[0]["metadata"]["role"] = "test"
        with self.assertRaisesRegex(ValueError, "training"):
            model.fit_statistics(data)

    def test_interruption_resume_and_tampered_outputs(self):
        data, config = tiny_entries(), model.ComponentConfig(width=8, epochs=1)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp).resolve() / "study"
            actual, calls = pipeline.fold, []
            def interrupted(*args):
                calls.append(args[3])
                if len(calls) == 3:
                    raise RuntimeError("synthetic interruption")
                return actual(*args)
            with mock.patch.object(pipeline, "fold", side_effect=interrupted), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    pipeline.run(data, config, (11, 12), ("linear",), out, {}, synthetic=True)
            self.assertEqual(len(list((out / "cv").glob("*.json"))), 2)
            with redirect_stdout(io.StringIO()):
                result = pipeline.run(data, config, (11, 12), ("linear",), out, {}, synthetic=True)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["primary"], "baseline")
            self.assertEqual(pipeline.read(out / "progress.json")["completed_fits"], 8)
            with mock.patch.object(model, "train", side_effect=AssertionError("completed study must resume")):
                repeated = pipeline.run(data, config, (11, 12), ("linear",), out, {}, synthetic=True)
            self.assertEqual(result, repeated)
            with self.assertRaisesRegex(RuntimeError, "config changed"):
                pipeline.run(data, replace(config, epochs=2), (11, 12), ("linear",), out, {}, synthetic=True)
            path = next((out / "models").glob("*.npz"))
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                pipeline.previous.verify_report(out)

    def test_real_loader_pins_frozen_control_and_rejects_wrong_outcomes(self):
        config = model.ComponentConfig(width=8, epochs=1)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp, redirect_stdout(io.StringIO()):
            base = Path(tmp).resolve()
            bundle, night, tags = [base / name for name in ("bundle", "night", "tags")]
            pipeline.write(tags / "label_selection.json", dict(records=[]))
            examples, controls = [], []
            for i in range(3):
                sample, outcomes, vocab, meta = fixture("t" + str(i), "s" + str(i))
                for value in outcomes.values():
                    value["raw"]["pdm_score"] = float(model.pdm(np.array([value["raw"][m] for m in pipeline.METRICS])))
                    value["filtered"] = value["raw"].copy()
                examples.append(pipeline.bundles.example(sample, outcomes, vocab, meta))
                query = dict(sample, token=meta["token"], session_id=meta["session_id"])
                for method in ("linear/ensemble/gain_m020", "mlp/ensemble/gain_m020", "mlp/ensemble/risk_t05_m020"):
                    controls.append(dict(method=method, **pipeline.record(query, outcomes, 0)))
            pipeline.bundles.build(bundle, examples, vocab, {}, {"closed"})
            night.mkdir()
            np.savez_compressed(night / "mlp_oof_ensemble.npz", tokens=np.array([e["metadata"]["token"] for e in examples]))
            pd.DataFrame(controls).to_csv(night / "oof_per_scene.csv", index=False)
            identity = dict(config=dict(epochs=1), seeds=[11, 12], sources={},
                inputs={str(bundle / name): pipeline.sha(bundle / name) for name in ("identity.json", "bundle_manifest.json")},
                python=platform_version(), numpy=np.__version__, torch=torch.__version__, platform=pipeline.platform.system(), threads=1)
            identity["inputs"][str(tags / "label_selection.json")] = pipeline.sha(tags / "label_selection.json")
            pipeline.write(night / "identity.json", identity)
            report = dict(status="complete", fresh_holdout_processed=0, identity_sha256=pipeline.sha(night / "identity.json"),
                artifacts={name: pipeline.sha(night / name) for name in ("oof_per_scene.csv", "mlp_oof_ensemble.npz")})
            pipeline.write(night / "report.json", report)
            entries, inputs, rows, _ = pipeline.load_data(bundle, night, config, (11, 12))
            self.assertEqual(len(entries), 3)
            self.assertEqual(len(rows), 9)
            self.assertEqual(model.formula_audit(entries)[0]["status"], "passed")
            controls[0]["raw_score"] += .1
            pd.DataFrame(controls).to_csv(night / "oof_per_scene.csv", index=False)
            report["artifacts"]["oof_per_scene.csv"] = pipeline.sha(night / "oof_per_scene.csv")
            pipeline.write(night / "report.json", report)
            with self.assertRaisesRegex(ValueError, "outcome replay"):
                pipeline.load_data(bundle, night, config, (11, 12))


def platform_version():
    return pipeline.platform.python_version()


if __name__ == "__main__":
    unittest.main()
