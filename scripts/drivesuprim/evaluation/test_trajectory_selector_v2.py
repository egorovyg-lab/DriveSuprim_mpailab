from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from navsim.agents import trajectory_selector as original
from navsim.agents import trajectory_selector_v2 as model
from navsim.agents.selector_training_bundle import LABEL_KEYS
from navsim.planning.script import run_trajectory_selector_v2 as pipeline
from test_trajectory_selector import fixture


def bundled(token="t", session="s0"):
    sample, vocab, outcomes = fixture(token, session)
    ix = original.inputs(sample, vocab)[1]
    raw = np.array([[outcomes[str(i)]["raw"][k] for k in LABEL_KEYS] for i in ix])
    return dict(sample=sample, label_indices=ix, label_raw=raw, label_filtered=raw.copy(),
                metadata=dict(token=token, session_id=session, role="train", origin="legacy")), vocab


def entries(config):
    examples = [bundled(sid + str(j), sid)[0] for sid in ("s0", "s1", "s2") for j in range(2)]
    return model.entries_from_bundle(examples, bundled()[1], {"s00"}, config)


class SelectorV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matched_initialization_rank_gain_identity_and_unused_rank_readout(self):
        torch.manual_seed(17)
        control = original.Selector(12, "mlp", width=16)
        torch.manual_seed(17)
        coherent = model.ConsistentSelector(12, width=16)
        for a, b in zip(control.parameters(), coherent.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        x, baseline = torch.randn(2, 16, 12), torch.tensor([0, 7])
        before = coherent(x, baseline)
        torch.testing.assert_close(before["rank"], before["gain"] / model.TEMPERATURE)
        torch.testing.assert_close(before["gain"].gather(1, baseline[:, None]), torch.zeros(2, 1))
        with torch.no_grad():
            coherent.readout.weight[0].fill_(1000)
            coherent.readout.bias[0].fill_(-900)
        after = coherent(x, baseline)
        for k in before:
            torch.testing.assert_close(before[k], after[k], rtol=0, atol=0)

    def test_permutation_padding_and_no_negative_gain_selection(self):
        torch.manual_seed(19)
        net = model.ConsistentSelector(10, width=16).eval()
        x, base = torch.randn(2, 16, 10), torch.tensor([4, 7])
        order = torch.randperm(16)
        perm_base = torch.stack([(order == b).nonzero()[0, 0] for b in base])
        prediction, permuted = net(x, base), net(x[:, order], perm_base)
        padded = net(torch.cat([x, torch.full((2, 2, 10), 999.)], 1), base,
                     torch.tensor([[True] * 16 + [False] * 2] * 2))
        for k in prediction:
            torch.testing.assert_close(permuted[k], prediction[k][:, order], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(padded[k][:, :16], prediction[k], atol=1e-6, rtol=1e-5)
        self.assertTrue(bool(torch.isneginf(padded["rank"][:, 16:]).all()))
        arrays = {k: v.detach().numpy() for k, v in prediction.items()}
        for j in range(2):
            e = dict(baseline=int(base[j]), indices=np.arange(16), sample=dict(candidate_ids=np.arange(16)))
            p = model.chosen_position(e, arrays, j, "consistent_v2", None, original.SelectorConfig())
            self.assertGreaterEqual(arrays["gain"][j, p], 0)

    def test_expected_regret_respects_magnitude_ties_and_mask(self):
        logits = torch.zeros((1, 3), requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        small = model.regret_loss(logits, torch.tensor([[1., .99, -999.]]), mask)
        large = model.regret_loss(logits, torch.tensor([[1., .1, 999.]]), mask)
        self.assertGreater(float(large.detach()), 80 * float(small.detach()))
        equal = model.regret_loss(logits, torch.ones((1, 3)), mask)
        self.assertEqual(float(equal.detach()), 0)
        large.sum().backward()
        self.assertLess(float(logits.grad[0, 0]), 0)
        self.assertGreater(float(logits.grad[0, 1]), 0)
        self.assertEqual(float(logits.grad[0, 2]), 0)

    def test_bundle_targets_match_v1_and_labels_never_become_features(self):
        config = original.SelectorConfig()
        e, vocab = bundled()
        first = model.entries_from_bundle([e], vocab, set(), config)[0]
        expected = original.targets(first["sample"], first["outcomes"], config)
        for k in ("quality", "gain", "good"):
            np.testing.assert_array_equal(first["targets"][k], expected[k])
        e["label_raw"][3, 0] = .1
        e["sample"]["outcome_values"] = np.full((16, 9), 99)
        second = model.entries_from_bundle([e], vocab, {"t"}, config)[0]
        np.testing.assert_array_equal(first["x"], second["x"])
        self.assertEqual(second["targets"]["gain"][3], np.float32(-.4))
        self.assertEqual(second["sampling_slice"], "challenge")
        e["metadata"]["role"] = "holdout"
        with self.assertRaisesRegex(ValueError, "training"):
            model.entries_from_bundle([e], vocab, set(), config)
        e["metadata"]["role"] = "train"
        e["label_indices"] = e["label_indices"][::-1]
        with self.assertRaisesRegex(ValueError, "labels"):
            model.entries_from_bundle([e], vocab, set(), config)

    def test_both_networks_learn_and_checkpoint_replays(self):
        config = original.SelectorConfig(width=16, epochs=40, learning_rate=.01)
        data = entries(config)
        rng = np.random.default_rng(6)
        for e in data:
            e["x"] = rng.normal(0, .03, (16, 5)).astype(np.float32)
            e["x"][:, 0], e["x"][3, 0] = 0, 1
        for family in model.FAMILIES:
            net, mean, scale, history = model.train(data, family, config)
            pred = model.predict(net, data, mean, scale)
            self.assertTrue(np.all(pred["rank"].argmax(1) == 3))
            self.assertLess(history[-1]["loss"], history[0]["loss"])
            restored, m, s = model.restore(original.checkpoint(net, mean, scale), family, config)
            repeated = model.predict(restored, data, m, s)
            for key in pred:
                np.testing.assert_array_equal(pred[key], repeated[key])

    def test_control_delegates_to_unchanged_v1(self):
        config = original.SelectorConfig(width=16, epochs=1)
        data = entries(config)
        with mock.patch.object(original, "train", return_value="control-result") as train:
            self.assertEqual(model.train(data, "control_v1", config), "control-result")
            self.assertEqual(train.call_args.args[3], "mlp")
            self.assertIs(train.call_args.args[4], config)

    def test_paired_comparison_equal_session_weight_and_identical_coverage(self):
        rows = []
        for sid, count, difference in (("short", 1, 4.), ("long", 3, 0.)):
            for i in range(count):
                for suffix in ("raw", "gated"):
                    for family, gain in (("control_v1", 1.), ("consistent_v2", 1. + difference)):
                        rows.append(dict(session_id=sid, token=sid + str(i), method=family + "_" + suffix, raw_gain_pp=gain))
        result = pipeline.paired_comparison(rows, 12)
        self.assertEqual(result["raw"]["consistent_minus_control_macro_pp"], 2.)
        self.assertEqual(result["gated"]["consistent_minus_control_micro_pp"], 1.)
        with self.assertRaisesRegex(ValueError, "identical scene"):
            pipeline.paired_comparison(rows[:-1], 12)

    def test_cv_excludes_entire_session_train_only_scaling_and_resumes(self):
        config = original.SelectorConfig(width=16, epochs=1)
        data = entries(config)
        for e in data:
            if e["sample"]["session_id"] == "s0":
                e["x"].fill(1e6)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp)
            seen = []
            actual = model.train
            def tracked(training, *args):
                seen.extend(training)
                return actual(training, *args)
            with mock.patch.object(model, "train", side_effect=tracked):
                valid, first = pipeline.fold_predictions(data, "consistent_v2", "s0", config, out)
            self.assertEqual({e["sample"]["session_id"] for e in seen}, {"s1", "s2"})
            self.assertEqual(len(valid), 2)
            path = next((out / "cv").glob("*.npz"))
            with np.load(path, allow_pickle=False) as f:
                expected, _ = original.fit_scaler([e["x"] for e in seen], original.session_weights([e["sample"] for e in seen]))
                np.testing.assert_array_equal(expected, f["mean"])
            with mock.patch.object(model, "train", side_effect=AssertionError("must resume")):
                _, repeated = pipeline.fold_predictions(data, "consistent_v2", "s0", config, out)
            np.testing.assert_array_equal(first["gain"], repeated["gain"])
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(RuntimeError, "CV fold changed"):
                pipeline.fold_predictions(data, "consistent_v2", "s0", config, out)

    def test_fit_and_historical_freeze_both_cohorts_before_lookup_no_closed_roles(self):
        config = original.SelectorConfig(width=16, epochs=1)
        data, vocab = entries(config), bundled()[1]
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            base = Path(tmp)
            out, roots = base / "out", [base / "old", base / "cached"]
            out.mkdir()
            queries, expected = {}, {}
            for root in roots:
                s, _, y = fixture(root.name, root.name + "-session", "development")
                queries[s["token"]] = s
                pipeline.write(root / "splits.json", dict(records=[{k: s[k] for k in ("token", "session_id", "role")},
                    dict(role="test", session_id="closed-test", token="test"),
                    dict(role="calibration", session_id="closed-cal", token="cal")]))
                pipeline.write(root / "export_index.json", {})
                path = root / "development_oracle/scenes" / (s["token"] + ".json")
                pipeline.write(path, dict(token=s["token"], outcomes=y))
                expected[root.name] = {s["token"]: pipeline.sha(path)}
            actual_read, opened = pipeline.read, []
            def guarded_read(path):
                if "development_oracle" in Path(path).parts:
                    saved = actual_read(out / "query_decisions.json")
                    self.assertEqual(set(saved["cohorts"]), {r.name for r in roots})
                    opened.append(path)
                return actual_read(path)
            def query_sample(root, r, index, decoder):
                self.assertEqual(r["role"], "development")
                return queries[r["token"]]
            with mock.patch.object(pipeline, "read", side_effect=guarded_read), \
                    mock.patch.object(pipeline, "load_decoder_sample", side_effect=query_sample):
                fitted = pipeline.fit(data, config, out)
                self.assertEqual(opened, [])
                decisions = pipeline.prepare_decisions(roots, base / "decoder", vocab, data, fitted, config, out)
                self.assertEqual(opened, [])
                result = pipeline.evaluate(decisions, roots, expected, out, config)
                self.assertEqual(len(opened), 2)
                self.assertEqual(result["fresh_holdout_processed"], 0)
                with mock.patch.object(model, "train", side_effect=AssertionError("frozen")):
                    self.assertEqual(pipeline.fit(data, config, out), fitted)
                self.assertEqual(pipeline.prepare_decisions(roots, base / "decoder", vocab, data, fitted, config, out), decisions)
                pipeline.write(out / "query_decisions.json", {})
                with self.assertRaisesRegex(RuntimeError, "BOTH"):
                    pipeline.evaluate(decisions, roots, expected, out, config)
                path = out / "consistent_v2.npz"
                path.write_bytes(path.read_bytes() + b"changed")
                with self.assertRaisesRegex(RuntimeError, "Frozen fitted artifact"):
                    pipeline.verify_fit(out)

    def test_cli_all_resumes_and_rejects_changed_protocol(self):
        examples = [bundled("train" + str(i), "train-session" + str(i))[0] for i in range(3)]
        vocab = bundled()[1]
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            base = Path(tmp)
            bundle, addition, prior, decoder, completion, out = [base / n for n in ("bundle", "addition", "prior", "decoder", "completion", "out")]
            roots = [base / "old", base / "cached"]
            pipeline.write(prior / "identity.json", dict(source=str(roots[0]), validation=str(roots[1]), completion=str(completion)))
            pipeline.write(addition / "label_selection.json", dict(records=[]))
            exporter = "navsim/planning/script/run_drivemem_decoder_functional_export.py"
            pipeline.write(decoder / "identity.json", dict(exporter_path=exporter))
            pipeline.write(decoder / "export_index.json", {})
            queries, expected = {}, {}
            for root in roots:
                s, _, y = fixture(root.name, root.name + "-session", "development")
                queries[s["token"]] = s
                pipeline.write(root / "splits.json", dict(records=[{k: s[k] for k in ("token", "session_id", "role")}]))
                pipeline.write(root / "export_index.json", {})
                path = root / "development_oracle/scenes" / (s["token"] + ".json")
                pipeline.write(path, dict(token=s["token"], outcomes=y))
                expected[root.name] = {s["token"]: pipeline.sha(path)}
            pipeline.write(completion / "report.json", dict(oracle_inputs=expected))
            paths = [addition / "label_selection.json", prior / "identity.json", completion / "report.json",
                     decoder / "identity.json", decoder / "export_index.json"] + [
                     r / n for r in roots for n in ("splits.json", "export_index.json")]
            pipeline.write(bundle / "identity.json", dict(inputs={str(p.resolve()): pipeline.sha(p) for p in paths},
                                                         counts=dict(scenes=3, sessions=3)))
            pipeline.write(bundle / "bundle_manifest.json", {})
            argv = ["selector", "all", "--bundle", str(bundle), "--addition", str(addition), "--prior", str(prior),
                    "--decoder", str(decoder), "--out", str(out), "--epochs", "1", "--threads", "1"]
            with mock.patch("sys.argv", argv), mock.patch.object(pipeline, "load_bundle", return_value=(examples, vocab)), \
                    mock.patch.object(pipeline, "verify_decoder_protocol", return_value=exporter), \
                    mock.patch.object(pipeline, "load_decoder_sample", side_effect=lambda root, r, index, dec: queries[r["token"]]), \
                    redirect_stdout(io.StringIO()):
                pipeline.main()
                first = pipeline.sha(out / "query_decisions.json")
                with mock.patch.object(model, "train", side_effect=AssertionError("must resume")):
                    pipeline.main()
                self.assertEqual(first, pipeline.sha(out / "query_decisions.json"))
                self.assertEqual(pipeline.read(out / "report.json")["status"], "complete")
                argv[argv.index("--epochs") + 1] = "2"
                with self.assertRaisesRegex(RuntimeError, "inputs/code/runtime changed"):
                    pipeline.main()


if __name__ == "__main__":
    unittest.main()
