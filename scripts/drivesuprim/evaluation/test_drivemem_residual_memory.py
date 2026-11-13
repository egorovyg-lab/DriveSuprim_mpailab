import copy
from contextlib import redirect_stdout
from dataclasses import replace
import importlib.util
import io
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from navsim.agents import drivemem_residual_memory as model
from navsim.planning.script import run_drivemem_residual_memory as pipeline

HAS_TORCH = importlib.util.find_spec("torch") is not None


def fixture(root):
    """Real bundle/table-label schemas, tiny synthetic cohort; no NAVSIM imports."""
    bundle = root / "drivemem_memory/bundle"
    artifact = root / "drex_rollout_audit/rollout_audit_table_fixture/artifacts"
    package = root / "drex_transfer/navsim_train_v1/package_manifest.json"
    rng = np.random.default_rng(82)
    vocab = rng.normal(size=(256, 40, 3))
    pipeline.save(bundle / "vocab.npz", vocab=vocab)
    index, rows, requests = {}, [], []
    for i in range(8):
        token = f"t{i}"
        s = dict(candidate_ids=np.arange(256), selected_local=np.array(0), selection_scores=-np.arange(256)*.01,
                 decoder_states=rng.normal(size=(256, 256)), scene_embedding=rng.normal(size=4), ego_status=np.zeros(8))
        s.update({"head_"+m: rng.normal(size=256) for m in (*model.METRICS, "imi")})
        raw = np.ones((64, 9)); raw[:, 0] = .6; raw[3, 0] = .8
        path = bundle / "examples" / (token + ".npz")
        pipeline.save(path, **s, label_indices=np.arange(16), label_raw=raw[:16], label_filtered=np.ones((16, 9)))
        index[token] = dict(sha256=pipeline.sha(path), metadata=dict(token=token, session_id=f"s{i}", role="train", source_role="train", origin="new"))
        rows.extend(dict(token=token, candidate_local=j, candidate_vocab=j, raw=raw[j].tolist()) for j in range(64))
        requests.append(dict(token=token, candidates=[[j, j] for j in range(64)]))
    identity = dict(counts=dict(scenes=8, sessions=8), protected_sessions=["closed"], sources={}, inputs={})
    pipeline.write(bundle / "identity.json", identity)
    pipeline.write(bundle / "examples_index.json", index)
    manifest = dict(counts=identity["counts"], identity_sha256=pipeline.sha(bundle/"identity.json"),
                    index_sha256=pipeline.sha(bundle/"examples_index.json"), vocab_sha256=pipeline.sha(bundle/"vocab.npz"),
                    input_keys=list(model.INPUT_KEYS), label_keys=list(model.LABEL_KEYS))
    pipeline.write(bundle / "bundle_manifest.json", manifest)
    files = {"bundle/"+n: dict(sha256=pipeline.sha(bundle/n)) for n in ("identity.json", "examples_index.json", "vocab.npz", "bundle_manifest.json")}
    files.update({"bundle/examples/"+t+".npz": dict(sha256=e["sha256"]) for t, e in index.items()})
    pipeline.write(package, dict(protocol="portable_train_v1", holdout_scenes=0, files=files))
    request = dict(protocol="table_top64_labels_v1", metrics=list(model.LABEL_KEYS), scenes=requests,
                   package_manifest_sha256=pipeline.sha(package), plan_sha256="fixture")
    pipeline.write(artifact.parent / "audit_request.json", request)
    labels = dict(status="complete", metrics=list(model.LABEL_KEYS), rows=rows, request_digest=model.digest(request))
    pipeline.write(artifact / "labels.json", labels)
    pipeline.write(artifact / "identity.json", dict(request_sha256=pipeline.sha(artifact.parent / "audit_request.json"),
                   score_protocol="single_candidate_raw_v1", human_penalty_filter=False))
    pipeline.write(artifact / "report.json", dict(status="complete", scenes=8, candidates=512, labels_sha256=pipeline.sha(artifact/"labels.json")))
    return type("Args", (), dict(bundle=bundle, labels=None, package_manifest=None, label_request=None))()


class DataAndMemoryTests(unittest.TestCase):
    def test_cohort_alignment_raw_scalar_and_readonly_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            entries, raw, provenance = pipeline.collect(args, (8, 8))
            self.assertEqual((len(entries), len(raw), len(provenance["protected_sessions"])), (8, 8, 1))
            self.assertEqual(entries[0]["sampling_slice"], "uniform")
            self.assertEqual(raw["t0"][0, 0], .6)  # Deliberately inconsistent with all-one components.
            self.assertNotIn("targets", entries[0])
            self.assertEqual(pipeline.read(args.bundle / "identity.json")["counts"]["scenes"], 8)
            with self.assertRaisesRegex(ValueError, "Expected 651"):
                pipeline.collect(args)

    def test_missing_duplicate_misaligned_and_changed_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp)); paths, packages = pipeline.label_paths(args)
            labels = pipeline.read(paths[0]); labels["rows"][0]["candidate_vocab"] = 99
            pipeline.write(paths[0], labels)
            report = pipeline.read(paths[0].parent/"report.json"); report["labels_sha256"] = pipeline.sha(paths[0])
            pipeline.write(paths[0].parent/"report.json", report)
            with self.assertRaisesRegex(ValueError, "Vocabulary/local"):
                pipeline.collect(args, (8, 8))
            labels["rows"][0]["candidate_vocab"] = 0
            labels["rows"][-1] = labels["rows"][0]
            pipeline.write(paths[0], labels); report["labels_sha256"] = pipeline.sha(paths[0])
            pipeline.write(paths[0].parent/"report.json", report)
            with self.assertRaisesRegex(FileNotFoundError, "Duplicate"):
                pipeline.collect(args, (8, 8))

    def test_protected_or_nontraining_sessions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            index = pipeline.read(args.bundle/"examples_index.json")
            index["t0"]["metadata"]["session_id"] = "closed"
            pipeline.write(args.bundle/"examples_index.json", index)
            manifest = pipeline.read(args.bundle/"bundle_manifest.json")
            manifest["index_sha256"] = pipeline.sha(args.bundle/"examples_index.json")
            pipeline.write(args.bundle/"bundle_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "training-only"):
                pipeline.bundle_metadata(args.bundle)

    def test_top16_scalar_parity_is_required_and_filtered_values_are_not_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            path = pipeline.label_paths(args)[0][0]
            labels = pipeline.read(path)
            labels["rows"][0]["raw"][0] = .1
            pipeline.write(path, labels)
            report = pipeline.read(path.parent/"report.json")
            report["labels_sha256"] = pipeline.sha(path)
            pipeline.write(path.parent/"report.json", report)
            with self.assertRaisesRegex(ValueError, "top16 raw label parity"):
                pipeline.collect(args, (8, 8))
        entries, raw = pipeline.synthetic()
        bad = raw["t0"].copy(); bad[63, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite top64"):
            model.supervised(entries[0], bad)

    def test_partition_is_outcome_blind_group_exclusion_and_fourteen_fits(self):
        metadata = [dict(token=f"t{i}", session_id=f"s{i//2}") for i in range(64)]
        groups = model.partition(metadata)
        self.assertEqual(list(map(len, groups)), [8]*4)
        self.assertEqual(model.partition(list(reversed(metadata))), groups)
        specs = model.fit_specs(groups)
        self.assertEqual(len(specs), 14)
        for outer in range(4):
            which = [i for i in range(4) if i != outer]
            for held in which:
                fitting = set(s for i in which if i != held for s in groups[i])
                self.assertFalse(fitting & set(groups[outer]))
                self.assertFalse(fitting & set(groups[held]))

    def test_session_equal_scaling_and_identical_action_features(self):
        entries, raw = pipeline.synthetic()
        uneven = [copy.deepcopy(entries[0]), copy.deepcopy(entries[0]), copy.deepcopy(entries[1])]
        for e in uneven:
            e["x"].fill(0 if e["session_id"] == entries[0]["session_id"] else 10)
        mean, scale = model.scaler(uneven)
        np.testing.assert_allclose(mean, 5)
        np.testing.assert_allclose(scale, 5)
        attached = model.supervised(entries[0], raw["t0"])
        np.testing.assert_array_equal(attached["x"], entries[0]["x"])
        self.assertEqual(attached["targets"]["gain"][0], 0)
        self.assertEqual(attached["targets"]["good"][5], 0)

    def test_memory_excludes_entire_own_session_and_is_order_invariant(self):
        entries, raw = pipeline.synthetic()
        entries.append(dict(entries[0], token="same_session_second_scene"))
        residuals = {e["token"]: np.r_[0., np.ones(63)*(.1 if e["session_id"] == "s0" else -.1)] for e in entries}
        mean, scale = model.scaler(entries)
        bank = model.Memory(entries, residuals, mean, scale)
        config = replace(model.Config(), probes=16)
        h = bank.bandwidth(entries, config)
        query = entries[0]
        recalled, trace = bank.recall(query, h, config)
        self.assertTrue(np.all(recalled[1:] < 0))
        for ix in trace["index"].reshape(-1):
            self.assertNotEqual(bank.records[ix]["session_id"], "s0")
        self.assertEqual(recalled[query["baseline"]], 0)
        permutation = np.random.default_rng(11).permutation(64)
        changed = dict(query, x=query["x"][permutation], ids=query["ids"][permutation],
                       indices=query["indices"][permutation], baseline=int(np.flatnonzero(permutation == 0)[0]))
        again, _ = bank.recall(changed, h, config)
        np.testing.assert_allclose(again, recalled[permutation], rtol=1e-10, atol=1e-10)
        lone = model.Memory([query], {query["token"]: residuals[query["token"]]}, mean, scale)
        np.testing.assert_array_equal(lone.recall(query, h, config)[0], 0)

    def test_shuffled_vector_keys_support_and_random_determinism(self):
        entries, _ = pipeline.synthetic()
        residuals = {e["token"]: np.r_[0., np.arange(1, 64)+i*100.] for i, e in enumerate(entries)}
        mean, scale = model.scaler(entries)
        bank = model.Memory(entries, residuals, mean, scale)
        values, provenance = bank.shuffled(55)
        for target, source in provenance.items():
            self.assertNotEqual(bank.scene_entries[target]["session_id"], bank.scene_entries[source]["session_id"])
        query = dict(entries[0], session_id="unseen")
        config = model.Config()
        _, real = bank.recall(query, 1., config)
        _, fake = bank.recall(query, 1., config, values=values)
        np.testing.assert_array_equal(real["index"], fake["index"])
        np.testing.assert_array_equal(real["weight"], fake["weight"])
        a, _ = bank.recall(query, 1., config, "random", 77)
        b, _ = bank.recall(query, 1., config, "random", 77)
        np.testing.assert_array_equal(a, b)

    def test_baseline_margin_ties_and_true_labels_cannot_be_selection_inputs(self):
        entries, _ = pipeline.synthetic(); e = entries[0]
        gain = np.zeros(64); gain[3] = .0049
        self.assertEqual(model.choose(e, gain), 0)
        gain[3] = gain[5] = .1
        self.assertEqual(model.choose(e, gain), 3)
        e["targets"] = object()
        self.assertEqual(model.choose(e, gain), 3)
        gain[0] = .01
        with self.assertRaisesRegex(ValueError, "baseline"):
            model.choose(e, gain)

    def test_session_bootstrap_and_concentration(self):
        vector = pipeline.aggregate(np.array([100., 100., 0.]), np.array(["a", "a", "b"]), ["a", "b"])
        np.testing.assert_array_equal(vector, [100, 0])
        stats = pipeline.effect_stats(np.array([0., 0., 10., 0.]), np.random.default_rng(4).integers(0, 4, (100, 4)))
        self.assertEqual(stats["leave_best_one_pp"], 0)
        self.assertEqual(stats["mean_pp"], 2.5)

    def test_os_lock_prevents_concurrent_writers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with pipeline.locked(root):
                with self.assertRaisesRegex(RuntimeError, "Another DriveMem"):
                    with pipeline.locked(root):
                        self.fail("Second writer entered")
            with pipeline.locked(root):
                self.assertTrue((root/".run.lock").exists())


@unittest.skipUnless(HAS_TORCH, "CPU torch required for integration fits")
class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_top64_features_match_existing_selector_on_shared_top16(self):
        from navsim.agents.trajectory_selector import inputs
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            entries, _, _ = pipeline.collect(args, (8, 8))
            sample = pipeline.load_npz(args.bundle/"examples/t0.npz")
            imi = sample["head_imi"].copy()
            vocab = pipeline.load_npz(args.bundle/"vocab.npz")["vocab"]
            x, ix, base = inputs(sample, vocab)
            np.testing.assert_array_equal(entries[0]["x"][:16], x)
            np.testing.assert_array_equal(entries[0]["indices"][:16], ix)
            self.assertEqual(entries[0]["baseline"], base)
            model.input_entry(sample, vocab, dict(token="t0", session_id="s0", role="train"))
            np.testing.assert_array_equal(sample["head_imi"], imi)

    def test_training_checkpoint_replay_and_query_targets_not_read(self):
        config = replace(model.Config(), width=8, epochs=12, learning_rate=.01)
        entries, raws = pipeline.synthetic()
        training = [model.supervised(e, raws[e["token"]]) for e in entries[:4]]
        checkpoint, history = model.train(training, config)
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        mean, _ = model.scaler(entries[:4])
        np.testing.assert_array_equal(checkpoint["mean"], mean)
        class NoTargets:
            def __getitem__(self, key):
                raise AssertionError("Query target was consumed")
        queries = [dict(e, targets=NoTargets()) for e in entries[4:]]
        prediction = model.predict(checkpoint, queries, config)
        np.testing.assert_array_equal(prediction[:, 0], 0)
        p = np.random.default_rng(4).permutation(64)
        shuffled = [dict(e, x=e["x"][p], baseline=int(np.flatnonzero(p == 0)[0])) for e in queries]
        np.testing.assert_allclose(model.predict(checkpoint, shuffled, config), prediction[:, p], rtol=1e-4, atol=1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.npz"; pipeline.save(path, **checkpoint)
            np.testing.assert_array_equal(prediction, model.predict(pipeline.load_npz(path), queries, config))

    def test_fold_training_donor_provenance_no_query_labels_and_replay(self):
        config = replace(model.Config(), width=8, epochs=1, probes=8, bootstrap_reps=100)
        entries, raws = pipeline.synthetic()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            out = Path(tmp)
            identity = pipeline.initialize(out, entries, dict(bindings={}), config, 1, True)
            pipeline.prepare(out, entries, raws)
            entries = pipeline.prepared(out); groups = identity["groups"]
            outer = 0; forbidden = set(groups[outer])
            original = pipeline.labels_for
            def guarded(root, selected):
                self.assertFalse({e["session_id"] for e in selected} & forbidden)
                return original(root, selected)
            with mock.patch.object(pipeline, "labels_for", side_effect=guarded):
                pipeline.run_fold(out, entries, groups, outer, config)
            donors = pipeline.read(out/"folds/0/donors.json")
            for record in donors["records"]:
                self.assertNotIn(record["session_id"], forbidden)
                self.assertNotIn(record["session_id"], donors["residual_sources"][record["token"]]["training_sessions"])
            before = pipeline.load_npz(out/"folds/0/decisions.npz")
            with mock.patch.object(model, "train", side_effect=AssertionError("Committed fold must resume")):
                pipeline.run_fold(out, entries, groups, outer, config)
            np.testing.assert_array_equal(before["positions"], pipeline.load_npz(out/"folds/0/decisions.npz")["positions"])
            with mock.patch.object(pipeline, "labels_for", side_effect=AssertionError("Incomplete report read labels")):
                with self.assertRaisesRegex(RuntimeError, "All four"):
                    pipeline.evaluate(out, entries, groups, config)
            # Tampered decisions are rejected before any outcome lookup.
            path = out/"folds/0/decisions.npz"
            before["positions"][0, 0] = 3; pipeline.save(path, **before)
            with self.assertRaisesRegex(RuntimeError, "Committed artifact changed"):
                pipeline.fold_commit(out, 0, entries, groups)

    def test_complete_pipeline_fourteen_fits_and_readonly_resume(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            result = pipeline.smoke(Path(tmp))
            self.assertEqual(result["fits"], 14)
            self.assertTrue(result["resumed"])

    def test_cli_smoke_in_separate_process_and_unknown_stage_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = str(Path(pipeline.__file__))
            args = [sys.executable, "-u", script, "smoke", "--out", str(Path(tmp)/"experiment")]
            done = subprocess.run(args, capture_output=True, text=True, timeout=90)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertIn('"smoke_passed"', done.stdout)
            rejected = subprocess.run([sys.executable, script, "unknown"], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
    unittest.main()
