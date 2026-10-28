import ast
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch
import sys
from pathlib import Path
from types import SimpleNamespace as N

import numpy as np
import pandas as pd

from navsim.agents.drivemem import (
    METRICS, ROLES, session_id, split_sessions, spaced_records, probabilities,
    training_candidate_indices, nearest_donors, memory_correction, corrected_choice,
    fit_ridge, ridge_correction, SAFETY,
)
from check_drivemem_cache import inspect as inspect_cache


def sample(token="q", session="query", n=4):
    data = {"token": token, "session_id": session, "candidate_ids": np.arange(n),
            "selection_scores": np.linspace(1., 0., n), "selected_local": np.array(0),
            "scene_embedding": np.array([1., 0., 0., 0.]),
            "ego_status": np.array([1., 0., 0., 0., 5., 0., 0., 0.])}
    data.update({"head_" + k: np.zeros(n) for k in METRICS})
    return data


def vocab(n=4):
    value = np.zeros((n, 40, 3))
    for i in range(n):
        value[i, :, 0] = np.linspace(0, 10 + i, 40)
        value[i, :, 1] = np.linspace(0, i, 40)
    return value


class SplitTests(unittest.TestCase):
    def test_same_session_across_log_segments(self):
        self.assertEqual(session_id("2021.09.29.19.02.14_veh-28_00540_00917"),
                         session_id("2021.09.29.19.02.14_veh-28_02451_02708"))
        with self.assertRaises(ValueError):
            session_id("unknown_format")

    def test_roles_disjoint_and_order_independent(self):
        rows = [dict(session_id=str(i)) for i in range(20)] * 2
        counts = {r: 3 for r in ROLES}
        first = split_sessions(rows, counts, 1)
        self.assertEqual(first, split_sessions(list(reversed(rows)), counts, 1))
        self.assertEqual(len({s for group in first.values() for s in group}), 15)
        with self.assertRaises(ValueError):
            split_sessions(rows, {r: 5 for r in ROLES}, 1)

    def test_temporal_spacing(self):
        rows = [dict(token=str(t), timestamp_us=t * 1000000) for t in range(80)]
        result = spaced_records(rows, 8, 10, 3)
        self.assertTrue(np.all(np.diff([r["timestamp_us"] for r in result]) >= 10000000))
        self.assertEqual(result, spaced_records(list(reversed(rows)), 8, 10, 3))


class CacheStatusTests(unittest.TestCase):
    def test_complete_cache_then_missing_file(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent, Path.cwd().resolve())
            (root / "metric_cache").mkdir()
            file = root / "metric_cache/a.pkl"
            file.write_bytes(b"cache fixture")
            (root / "splits.json").write_text(json.dumps({"records": [{"token": "a"}]}))
            (root / "cache_index.json").write_text(json.dumps({"a": {"path": str(file),
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}}))
            self.assertEqual(inspect_cache(root)["status"], "complete")
            file.unlink()
            report = inspect_cache(root)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["invalid_files"][0]["reason"], "missing file")

    def test_corrupt_cache_hash_is_detected(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent, Path.cwd().resolve())
            (root / "metric_cache").mkdir()
            file = root / "metric_cache/a.pkl"
            file.write_bytes(b"changed")
            (root / "splits.json").write_text(json.dumps({"records": [{"token": "a"}]}))
            (root / "cache_index.json").write_text(json.dumps({"a": {"path": str(file), "sha256": "wrong"}}))
            self.assertEqual(inspect_cache(root)["invalid_files"][0]["reason"], "SHA256 mismatch")

    def test_empty_scene_selection_is_not_success(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent, Path.cwd().resolve())
            (root / "splits.json").write_text(json.dumps({"records": []}))
            self.assertEqual(inspect_cache(root)["status"], "incomplete")


class PredictorTests(unittest.TestCase):
    def test_noop_preserves_baseline_and_full_pool_is_available(self):
        s = sample(n=256)
        self.assertEqual(corrected_choice(s, np.zeros((256, 8)), .5), 0)
        correction = np.zeros((256, 8))
        correction[-1, [0, 4, 5]] = 1.
        self.assertEqual(corrected_choice(s, correction, 1.), 255)

    def test_candidate_sampling_is_reproducible_without_labels(self):
        s = sample(n=256)
        a = training_candidate_indices(s, 8, 8, 1)
        s["query_outcome"] = np.ones((256, 8)) * 999
        b = training_candidate_indices(s, 8, 8, 1)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(len(set(a.tolist())), 16)
        self.assertIn(0, a)

    def test_memory_cannot_access_query_labels(self):
        q, donor = sample(), sample("d", "donor")
        donor.update(label_local_indices=np.array([0]), label_values=np.ones((1, 8)) * .8)
        ids, _ = nearest_donors(q, [donor], 5, 1)
        before = memory_correction(q, [donor], ids, vocab())
        q["label_values"] = np.zeros((4, 8))
        after = memory_correction(q, [donor], ids, vocab())
        self.assertTrue(np.array_equal(before, after))
        self.assertTrue(np.allclose(before, .3))
        self.assertTrue(np.allclose(memory_correction(q, [donor], [], vocab()), 0))

    def test_same_session_memory_is_rejected(self):
        with self.assertRaises(ValueError):
            nearest_donors(sample(), [sample("donor")], 1, 1)

    def test_incompatible_command_falls_back(self):
        donor = sample("d", "donor")
        donor["ego_status"][:4] = [0, 1, 0, 0]
        ids, distance = nearest_donors(sample(), [donor], 1, 1)
        self.assertEqual(ids, [])
        self.assertTrue(np.isinf(distance))

    def test_parametric_control_uses_provided_training_labels(self):
        s = sample("train", "train")
        s.update(label_local_indices=np.arange(4), label_values=np.ones((4, 8)) * .75)
        model = fit_ridge([s], vocab(), regularization=10.)
        q = sample()
        prediction = ridge_correction(q, vocab(), model)
        self.assertTrue(np.allclose(prediction, .25, atol=1e-8))
        q["label_values"] = np.zeros((4, 8))
        self.assertTrue(np.array_equal(prediction, ridge_correction(q, vocab(), model)))

    def test_extreme_head_logits_remain_finite(self):
        s = sample()
        s["head_" + METRICS[0]][:] = [10000, -10000, 0, 1]
        self.assertTrue(np.isfinite(probabilities(s)).all())
        self.assertIn(corrected_choice(s, np.zeros((4, 8)), .5), range(4))


class ConfigurationTests(unittest.TestCase):
    def test_python39_syntax(self):
        for name in ("navsim/agents/drivemem.py", "navsim/planning/script/run_drivemem_memory.py"):
            ast.parse(Path(name).read_text(encoding="utf-8"), feature_version=(3, 9))

    def test_hydra_composition_when_available(self):
        try:
            from hydra import compose, initialize_config_dir
        except ImportError:
            self.skipTest("Hydra unavailable in local lightweight runtime; run check in server container")
        path = str(Path("navsim/planning/script/config/pdm_scoring").resolve())
        with initialize_config_dir(version_base=None, config_dir=path):
            cfg = compose(config_name="default_drivemem_memory")
        self.assertEqual(cfg.train_test_split.data_split, "trainval")
        self.assertFalse(cfg.agent.config.training)
        self.assertTrue(cfg.agent.config.inference.export_candidates)
        self.assertEqual(cfg.drivemem.stage, "inventory")

    def test_paths_diagnostic_distinguishes_logs_and_missing_images(self):
        text = Path("scripts/drivesuprim/evaluation/run_drivemem_memory.sh").read_text(encoding="utf-8")
        code = text.split('python - "$root" <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent, Path.cwd().resolve())
            logs = root / "navsim_logs/trainval"
            logs.mkdir(parents=True)
            (logs / "example.pkl").write_text("test")
            with patch.dict(os.environ, {"OPENSCENE_DATA_ROOT": str(root), "NAVSIM_DEVKIT_ROOT": str(root)}), \
                    patch.object(sys, "argv", ["paths", str(root / "run")]), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                exec(compile(code, "<paths diagnostic>", "exec"), {"__name__": "__main__"})
            self.assertIn("FOUND " + str(logs), output.getvalue())
            self.assertIn("MISSING " + str(root / "sensor_blobs/trainval"), output.getvalue())


class PipelineTests(unittest.TestCase):
    def test_frozen_training_evaluation_and_resume_without_query_labels(self):
        path = Path("navsim/planning/script/run_drivemem_memory.py")
        names = {"sha", "write_json", "read_json", "save_npz", "records", "sample", "label_indices",
                 "training_samples", "fit", "choices", "run_evaluation"}
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        ns = dict(Path=Path, hashlib=hashlib, json=json, os=os, np=np, pd=pd,
                  METRICS=METRICS, SAFETY=SAFETY, training_candidate_indices=training_candidate_indices,
                  probabilities=probabilities, nearest_donors=nearest_donors, memory_correction=memory_correction,
                  corrected_choice=corrected_choice, fit_ridge=fit_ridge, ridge_correction=ridge_correction)
        exec(compile(module, str(path), "exec"), ns)
        cfg = N(drivemem=N(memory_mode="enriched", label_top=1, label_random=1, seed=1,
                           neighbors=1, ridge_lambda=10., threshold_quantile=.75, strength=.5))
        calls = []

        def fake_evaluators(cfg, root):
            def evaluate(token, candidate):
                calls.append((token, candidate))
                metrics = {key: 1. for key in METRICS}
                metrics.update(pdm_score=.5 + .1 * candidate, aggregate_without_two_frame=.5 + .1 * candidate)
                return {"raw": dict(metrics), "filtered": dict(metrics)}, np.zeros((41, 11))
            return evaluate, vocab()

        ns["evaluators"] = fake_evaluators
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent, Path.cwd().resolve())
            (root / "features").mkdir()
            (root / "labels").mkdir()
            rows = [dict(token="m", session_id="m", role="memory"),
                    dict(token="f", session_id="f", role="fit"),
                    dict(token="v1", session_id="v1", role="development"),
                    dict(token="v2", session_id="v2", role="development"),
                    dict(token="c", session_id="c", role="calibration"),
                    dict(token="t", session_id="t", role="test")]
            ns["write_json"](root / "splits.json", {"records": rows})
            index = {}
            for row in rows:
                s = sample(row["token"], row["session_id"])
                arrays = {k: v for k, v in s.items() if k not in ("token", "session_id")}
                fp = root / "features" / (row["token"] + ".npz")
                ns["save_npz"](fp, **arrays)
                index[row["token"]] = ns["sha"](fp)
                if row["role"] in ("memory", "fit"):
                    s["role"] = row["role"]
                    indices = ns["label_indices"](cfg, s)
                    values = [{"raw": {key: .75 for key in METRICS}} for _ in indices]
                    ns["write_json"](root / "labels" / (row["token"] + ".json"),
                                     dict(token=row["token"], role=row["role"], local_indices=indices.tolist(),
                                          values=values, feature_sha256=index[row["token"]]))
            ns["write_json"](root / "export_index.json", index)
            ns["save_npz"](root / "vocab.npz", vocab=vocab())
            with contextlib.redirect_stdout(io.StringIO()):
                ns["fit"](cfg, root)
                first_hash = ns["sha"](root / "ridge.npz")
                ns["fit"](cfg, root)
                self.assertEqual(first_hash, ns["sha"](root / "ridge.npz"))
                ns["run_evaluation"](cfg, root, "development")
                report = ns["read_json"](root / "evaluation_development/report.json")
                self.assertEqual(report["scenes"], 2)
                self.assertEqual(len(report["methods"]), 5)
                self.assertEqual(report["methods"]["baseline"]["coverage"], 0)
                self.assertTrue(all(t in ("v1", "v2") for t, _ in calls))
                calls.clear()
                ns["run_evaluation"](cfg, root, "development")
                self.assertEqual(calls, [])
                self.assertFalse((root / "labels/c.json").exists())
                self.assertFalse((root / "labels/t.json").exists())


if __name__ == "__main__":
    unittest.main()
