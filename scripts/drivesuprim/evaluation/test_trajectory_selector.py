from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from navsim.agents import trajectory_selector as selector
from navsim.agents.drivemem import METRICS
from navsim.planning.script import run_trajectory_selector as pipeline


def fixture(token="t", session="s0", role="fit"):
    rng = np.random.default_rng(7)
    vocab = rng.normal(size=(256, 40, 3)).astype(np.float32)
    s = dict(token=token, session_id=session, role=role, candidate_ids=np.arange(256),
             selected_local=np.array(0), selection_scores=np.linspace(2, 0, 256),
             decoder_states=rng.normal(size=(256, 8)).astype(np.float32),
             ego_status=np.zeros(8), scene_embedding=np.array([1., 2., 3.]), head_imi=np.zeros(256))
    s.update({"head_" + name: np.zeros(256) for name in METRICS})
    outcomes = {}
    for i in range(16):
        value = dict(pdm_score=1. if i == 3 else .5, **{k: 1. for k in METRICS})
        outcomes[str(i)] = dict(raw=value, filtered=value.copy())
    return s, vocab, outcomes


class SelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_inputs_use_whitelist_and_alignment(self):
        s, vocab, _ = fixture()
        x, ix, b = selector.inputs(s, vocab)
        poisoned = dict(s, utility_values=np.full(16, 999), outcome_values=np.full((16, 9), 999),
                        role="development", token="future-token", session_id="future-session")
        np.testing.assert_array_equal(x, selector.inputs(poisoned, vocab)[0])
        self.assertEqual(b, 0)
        self.assertEqual(ix.tolist(), list(range(16)))
        s["decoder_states"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            selector.inputs(s, vocab)

    def test_training_requires_complete_MF_and_stored_pdm(self):
        s, _, outcomes = fixture()
        config = selector.SelectorConfig()
        y = selector.targets(s, outcomes, config)
        self.assertEqual(y["gain"][3], .5)
        self.assertEqual(y["good"][3], 1)
        # Component vector is all ones; stored score still must be used as-is.
        self.assertEqual(y["quality"][0], .5)
        outcomes["3"]["raw"]["drivable_area_compliance"] = 0
        self.assertEqual(selector.targets(s, outcomes, config)["good"][3], 0)
        with self.assertRaisesRegex(ValueError, "M/F"):
            selector.targets(dict(s, role="development"), outcomes, config)
        del outcomes["15"]
        with self.assertRaisesRegex(ValueError, "Complete top16"):
            selector.targets(s, outcomes, config)

    def test_tie_aware_rank_loss_and_padding(self):
        logits = torch.tensor([[.1, .2, -.1, .3]], requires_grad=True)
        mask = torch.tensor([[True, True, True, False]])
        equal = torch.tensor([[1., 1., 1., 100.]])
        loss = selector.ranking_loss(logits, equal, mask, .001)
        self.assertAlmostEqual(float(loss.detach()), 0., places=6)
        loss.sum().backward()
        self.assertLess(float(logits.grad.abs().max()), 1e-6)
        quality = torch.tensor([[1., 1., 0., 100.]])
        a = selector.ranking_loss(logits, quality, mask, .001)
        b = selector.ranking_loss(logits[:, [1, 0, 2, 3]], quality, mask, .001)
        torch.testing.assert_close(a, b)

    def test_permutation_equivariance_padding_and_parameter_control(self):
        torch.manual_seed(5)
        x = torch.randn(2, 16, 30)
        baseline = torch.tensor([2, 5])
        permutation = torch.randperm(16)
        permuted_baseline = torch.stack([(permutation == b).nonzero()[0, 0] for b in baseline])
        counts = []
        for family in ("mlp", "attention"):
            net = selector.Selector(30, family, width=16, heads=4).eval()
            counts.append(sum(p.numel() for p in net.parameters()))
            original = net(x, baseline)
            changed = net(x[:, permutation], permuted_baseline)
            padded = net(torch.cat([x, torch.full((2, 3, 30), 999.)], dim=1), baseline,
                         torch.tensor([[True] * 16 + [False] * 3] * 2))
            for key in original:
                torch.testing.assert_close(changed[key], original[key][:, permutation], atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(padded[key][:, :16], original[key], atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(original["gain"].gather(1, baseline[:, None]), torch.zeros(2, 1))
        self.assertLess(abs(counts[0] - counts[1]) / max(counts), .05)

    def test_gate_is_candidate_specific_and_ties_preserve_baseline(self):
        rank = np.array([0., 2., 1.])
        gain = np.array([0., .1, .2])
        good = np.array([-3., -3., 5.])  # Good third action cannot license the second.
        self.assertEqual(selector.choose(rank, gain, good, 0, [10, 11, 12]), 1)
        self.assertEqual(selector.choose(rank, gain, good, 0, [10, 11, 12], .5), 0)
        self.assertEqual(selector.choose(np.zeros(3), gain, good, 2, [10, 11, 12]), 2)
        ids = np.array([7, 4, 9])
        rank = np.array([2., 2., 0.])
        order = np.array([2, 0, 1])
        before = selector.choose(rank, gain, good, 2, ids)
        after = selector.choose(rank[order], gain[order], good[order], 0, ids[order])
        self.assertEqual(ids[before], ids[order][after])

    def test_tiny_network_learns_and_checkpoint_replays(self):
        config = selector.SelectorConfig(width=16, heads=4, epochs=40, learning_rate=.01)
        samples = [dict(role="fit", session_id="s" + str(i // 2), baseline_position=0) for i in range(6)]
        rng = np.random.default_rng(4)
        xs = rng.normal(0, .05, (6, 16, 5)).astype(np.float32)
        xs[:, :, 0] = 0
        xs[:, 3, 0] = 1
        q = np.zeros(16, dtype=np.float32)
        q[3] = 1
        labels = [dict(quality=q, gain=q, good=q) for _ in samples]
        for family in ("mlp", "attention"):
            net, mean, scale, history = selector.train(xs, labels, samples, family, config)
            prediction = selector.predict(net, xs, [0] * 6, mean, scale)
            self.assertTrue(np.all(prediction["rank"].argmax(1) == 3))
            self.assertLess(history[-1]["loss"], history[0]["loss"])
            restored, m, sd = selector.restore(selector.checkpoint(net, mean, scale), family, config)
            repeated = selector.predict(restored, xs, [0] * 6, m, sd)
            for key in prediction:
                np.testing.assert_array_equal(prediction[key], repeated[key])

    def test_scaling_has_equal_session_mass(self):
        samples = [dict(session_id="a"), dict(session_id="a"), dict(session_id="b")]
        xs = [np.zeros((16, 2)), np.zeros((16, 2)), np.ones((16, 2)) * 10]
        mean, _ = selector.fit_scaler(xs, selector.session_weights(samples))
        np.testing.assert_allclose(mean, [5., 5.])

    def test_cv_excludes_entire_session_and_resumes(self):
        config = selector.SelectorConfig(width=16, heads=4, epochs=1)
        samples, outcomes = [], {}
        for sid in ("s0", "s1", "s2"):
            for j in range(2):
                s, vocab, y = fixture(sid + str(j), sid)
                samples.append(s)
                outcomes[s["token"]] = y
        entries = pipeline.build_training([], samples, outcomes, vocab, {}, config)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp)
            seen = []
            original = selector.train
            def tracked(xs, ys, training_samples, *args, **kwargs):
                seen.extend(training_samples)
                return original(xs, ys, training_samples, *args, **kwargs)
            with mock.patch.object(selector, "train", side_effect=tracked):
                valid, first = pipeline.fold_predictions(entries, "mlp", "s0", config, out)
            self.assertEqual({s["session_id"] for s in seen}, {"s1", "s2"})
            self.assertEqual(len(valid), 2)
            with mock.patch.object(selector, "train", side_effect=AssertionError("must resume")):
                _, repeated = pipeline.fold_predictions(entries, "mlp", "s0", config, out)
            np.testing.assert_array_equal(first["rank"], repeated["rank"])
            metadata = next((out / "cv").glob("*.json"))
            state = pipeline.read(metadata)
            state["partition"]["valid_session"] = "wrong"
            pipeline.write(metadata, state)
            with self.assertRaisesRegex(RuntimeError, "CV fold changed"):
                pipeline.fold_predictions(entries, "mlp", "s0", config, out)

    def test_full_fit_freezes_both_cohorts_before_outcomes_and_never_opens_CT(self):
        config = selector.SelectorConfig(width=16, heads=4, epochs=2)
        samples, targets = [], {}
        for i in range(3):
            s, vocab, y = fixture("train" + str(i), "train-session" + str(i))
            samples.append(s)
            targets[s["token"]] = y
        entries = pipeline.build_training([], samples, targets, vocab, {}, config)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            base = Path(tmp)
            out, source, validation = [base / name for name in ("out", "source", "validation")]
            out.mkdir()
            queries, expected = {}, {}
            for root in (source, validation):
                s, _, y = fixture(root.name, root.name + "-session", "development")
                queries[s["token"]] = s
                r = {k: s[k] for k in ("token", "session_id", "role")}
                pipeline.write(root / "splits.json", dict(records=[r,
                    dict(token="closed-test", session_id="ct", role="test"),
                    dict(token="closed-cal", session_id="cc", role="calibration")]))
                pipeline.write(root / "export_index.json", {})
                path = root / "development_oracle/scenes" / (s["token"] + ".json")
                pipeline.write(path, dict(token=s["token"], outcomes=y))
                expected[root.name] = {s["token"]: pipeline.sha(path)}
            original_read = pipeline.read
            outcome_reads = []
            def guarded_read(path):
                path = Path(path)
                if "development_oracle" in path.parts:
                    saved = original_read(out / "query_decisions.json")
                    self.assertEqual(set(saved["cohorts"]), {source.name, validation.name})
                    outcome_reads.append(path)
                return original_read(path)
            def query_sample(root, r, index, decoder):
                self.assertEqual(r["role"], "development")
                return queries[r["token"]]
            with mock.patch.object(pipeline, "read", side_effect=guarded_read), \
                    mock.patch.object(pipeline, "load_decoder_sample", side_effect=query_sample):
                fitted = pipeline.fit(entries, targets, config, out)
                self.assertEqual(outcome_reads, [])
                decisions = pipeline.prepare_decisions(source, validation, base / "decoder", vocab, entries, fitted, config, out)
                self.assertEqual(outcome_reads, [])
                result = pipeline.evaluate(decisions, source, validation, expected, out, config)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(len(outcome_reads), 2)
                with mock.patch.object(selector, "train", side_effect=AssertionError("frozen fit")):
                    self.assertEqual(pipeline.fit(entries, targets, config, out), fitted)
                again = pipeline.prepare_decisions(source, validation, base / "decoder", vocab, entries, fitted, config, out)
                self.assertEqual(again, decisions)
                model_path = out / "mlp.npz"
                model_path.write_bytes(model_path.read_bytes() + b"tampered")
                with self.assertRaisesRegex(RuntimeError, "Frozen fitted artifact"):
                    pipeline.verify_fit(out)

    def test_cli_all_then_resume_with_frozen_identity(self):
        samples, outcomes = [], {}
        for i in range(3):
            s, vocab, y = fixture("train" + str(i), "train-session" + str(i))
            samples.append(s)
            outcomes[s["token"]] = y
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            base = Path(tmp)
            prior, addition, decoder, out, source, validation, completion = [
                base / name for name in ("prior", "addition", "decoder", "out", "source", "validation", "completion")]
            pipeline.write(decoder / "identity.json", dict(
                exporter_path="navsim/planning/script/run_drivemem_decoder_functional_export.py"))
            pipeline.write(addition / "label_selection.json", dict(records=[]))
            queries, expected = {}, {}
            for root in (source, validation):
                s, _, y = fixture(root.name, root.name + "-session", "development")
                queries[s["token"]] = s
                pipeline.write(root / "splits.json", dict(records=[{k: s[k] for k in ("token", "session_id", "role")}]))
                pipeline.write(root / "export_index.json", {})
                path = root / "development_oracle/scenes" / (s["token"] + ".json")
                pipeline.write(path, dict(token=s["token"], outcomes=y))
                expected[root.name] = {s["token"]: pipeline.sha(path)}
            pipeline.write(completion / "report.json", dict(oracle_inputs=expected))
            data = ([], samples, outcomes, vocab, source, validation, completion, {}, {}, 20261108)
            argv = ["selector", "all", "--prior", str(prior), "--addition", str(addition),
                    "--decoder", str(decoder), "--out", str(out), "--epochs", "1", "--threads", "1"]
            with mock.patch("sys.argv", argv), mock.patch.object(pipeline, "load_decoder_inputs", return_value=data), \
                    mock.patch.object(pipeline, "load_decoder_sample", side_effect=lambda root, r, index, decoder: queries[r["token"]]), \
                    redirect_stdout(io.StringIO()):
                pipeline.main()
                first = pipeline.sha(out / "query_decisions.json")
                with mock.patch.object(selector, "train", side_effect=AssertionError("must reuse fitted model")):
                    pipeline.main()
                self.assertEqual(first, pipeline.sha(out / "query_decisions.json"))
                self.assertEqual(pipeline.read(out / "report.json")["status"], "complete")
                argv[argv.index("--epochs") + 1] = "2"
                with self.assertRaisesRegex(RuntimeError, "inputs/code/runtime changed"):
                    pipeline.main()


if __name__ == "__main__":
    unittest.main()
