"""CPU tests: python -m unittest discover -s scripts/drivesuprim/evaluation -p test_drivemem_oracle_utils.py"""
import unittest
import ast
import contextlib
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from navsim.evaluate.drivemem_oracle_utils import sample_logs, summarize_candidates, aggregate_report
import navsim.evaluate.drivemem_oracle_utils as oracle_utils


class SamplingTests(unittest.TestCase):
    def test_reproducibility_exclusion_and_spacing(self):
        rows = [{"token": f"{log}_{t}", "log_name": log, "timestamp_us": t * 1000000}
                for log in ("a", "b", "c", "d") for t in range(40)]
        first, info = sample_logs(rows, 3, 3, 10, 123, ["d"])
        reverse, _ = sample_logs(list(reversed(rows)), 3, 3, 10, 123, ["d"])
        self.assertEqual(first, reverse)
        self.assertEqual(info["eligible_logs"], 3)
        frame = pd.DataFrame(first)
        self.assertEqual(set(frame.log_name), {"a", "b", "c"})
        for _, group in frame.groupby("log_name"):
            self.assertEqual(len(group), 3)
            self.assertTrue((group.timestamp_us.diff().dropna() >= 10000000).all())

    def test_infeasible_selection_fails(self):
        rows = [{"token": "x", "log_name": "a", "timestamp_us": 0},
                {"token": "y", "log_name": "a", "timestamp_us": 1000000}]
        with self.assertRaises(ValueError):
            sample_logs(rows, 1, 2, 10, 1)

    def test_duplicate_times_never_repeat_a_scene(self):
        rows = [{"token": str(i), "log_name": "a", "timestamp_us": 0} for i in range(4)]
        result, _ = sample_logs(rows, 1, 4, 0, 1)
        self.assertEqual(len({r["token"] for r in result}), 4)


class SummaryTests(unittest.TestCase):
    def test_safety_constraint_and_best_ranked_tie(self):
        rows = []
        for rank, score, safe in ((1, .5, True), (2, .9, False), (3, .8, True), (4, .9, False)):
            rows.append(dict(token="t", log_name="log", candidate_id=rank, model_rank=rank,
                             is_baseline=rank == 1, pdm_score=score, safe_vs_baseline=safe,
                             no_at_fault_collisions=1., time_to_collision_within_bound=1.,
                             predicted_nc=.9, predicted_ttc=.8))
        result = summarize_candidates(list(reversed(rows)))
        self.assertEqual(result["oracle_model_rank"], 2)
        self.assertAlmostEqual(result["oracle_gain_pp"], 40.)
        self.assertAlmostEqual(result["safe_oracle_gain_pp"], 30.)
        self.assertEqual(result["better_candidates"], 3)

    def test_cluster_interval_resamples_whole_logs(self):
        frame = pd.DataFrame([dict(token=str(i), log_name=log, baseline_local_pdm=.5,
                                   oracle_gain_pp=10., safe_oracle_gain_pp=5.)
                              for i, log in enumerate(["a", "a", "b", "c"])])
        report, logs = aggregate_report(frame, seed=1, bootstrap_reps=100)
        self.assertEqual(report["logs"], 3)
        self.assertEqual(report["oracle_gain_pp_cluster_ci95"], [10., 10.])
        self.assertEqual(report["safe_oracle_gain_pp_cluster_ci95"], [5., 5.])
        self.assertEqual(logs.scenes.sum(), 4)


class ResumeTests(unittest.TestCase):
    def test_interruption_resume_and_corruption_detection(self):
        # Execute the real persistence/scoring orchestration with a deterministic
        # fake simulator; NAVSIM/torch are not required for this failure-path test.
        script = Path("navsim/planning/script/run_drivemem_oracle_pilot.py").resolve()
        tree = ast.parse(script.read_text(encoding="utf-8"))
        names = {"sha256", "write_json", "write_npz", "compare_rows", "score"}
        module = ast.Module(body=[node for node in tree.body
                                  if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
        metrics = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
                   "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound",
                   "lane_keeping", "history_comfort", "pdm_score")
        interrupted = [False]
        calls = []

        class Cache:
            def __init__(self, path):
                pass

            def get_from_token(self, token):
                return SimpleNamespace(token=token, log_name="log_" + token,
                                       timepoint=SimpleNamespace(time_s=1.))

        def fake_score(meta, trajectory, *args):
            calls.append(meta.token)
            if meta.token == "t2" and not interrupted[0]:
                interrupted[0] = True
                raise RuntimeError("synthetic interruption")
            idx = int(trajectory[0, 0])
            values = {key: 1. for key in metrics}
            values["pdm_score"] = [.5, .8, .7][idx]
            values["no_at_fault_collisions"] = [1., 0., 1.][idx]
            return pd.DataFrame([values]), np.full((41, 11), idx, dtype=float)

        ns = dict(__file__=str(script), hashlib=hashlib, json=json, os=os, time=time,
                  Path=Path, np=np, pd=pd, oracle_utils=oracle_utils,
                  OmegaConf=SimpleNamespace(to_container=lambda cfg, resolve: cfg.raw),
                  MetricCacheLoader=Cache, Trajectory=lambda poses, sampling: poses,
                  instantiate=lambda value, *args: value, pdm_score=fake_score,
                  METRICS=metrics, SAFETY=metrics[:4] + (metrics[5],),
                  summarize_candidates=summarize_candidates, aggregate_report=aggregate_report,
                  git_info=lambda: {"commit": "test", "diff_sha256": "test"})
        exec(compile(module, str(script), "exec"), ns)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as temporary:
            out = Path(temporary).resolve()
            self.assertEqual(out.parent, Path.cwd().resolve())
            (out / "candidates").mkdir()
            checkpoint = out / "checkpoint"
            checkpoint.write_text("synthetic")
            vocab = np.zeros((3, 40, 3), dtype=np.float32)
            vocab[:, 0, 0] = np.arange(3)
            ns["write_npz"](out / "vocab.npz", vocab=vocab)
            for token in ("t1", "t2"):
                ns["write_npz"](out / "candidates" / f"{token}.npz", candidate_ids=np.arange(3),
                                selection_scores=np.array([3., 2., 1.]), selected_local=np.array(0),
                                selected_trajectory=vocab[0], head_no_at_fault_collisions=np.zeros(3),
                                head_time_to_collision_within_bound=np.zeros(3))
            keys = ("agent", "simulator", "scorer", "traffic_agents", "traffic_agents_policy",
                    "metric_cache_path", "proposal_sampling")
            raw = {key: key for key in keys}
            manifest = dict(status="export_complete", config=raw, git=ns["git_info"](), tokens=["t1", "t2"],
                            checkpoint_sha256=ns["sha256"](checkpoint),
                            vocab_sha256=ns["sha256"](out / "vocab.npz"),
                            pilot_script_sha256=ns["sha256"](script),
                            utils_sha256=ns["sha256"](Path(oracle_utils.__file__)))
            ns["write_json"](out / "manifest.json", manifest)
            cfg = SimpleNamespace(raw=raw, agent=SimpleNamespace(checkpoint_path=str(checkpoint),
                                  config=SimpleNamespace(trajectory_sampling="sampling")),
                                  simulator=SimpleNamespace(proposal_sampling="sampling"),
                                  scorer=SimpleNamespace(proposal_sampling="sampling"),
                                  traffic_agents_policy=SimpleNamespace(non_reactive="policy"),
                                  metric_cache_path="cache",
                                  drivemem=SimpleNamespace(candidate_limit=3, reference_csv=""))
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    ns["score"](cfg, out)
                self.assertTrue((out / "scoring_k3/t1_done.json").exists())
                self.assertFalse((out / "scoring_k3/report.json").exists())
                calls.clear()
                ns["score"](cfg, out)
                self.assertNotIn("t1", calls)
                report = json.loads((out / "scoring_k3/report.json").read_text())
                self.assertEqual(report["resumed_scenes"], 1)
                self.assertAlmostEqual(report["oracle_gain_mean_pp"], 30.)
                self.assertAlmostEqual(report["safe_oracle_gain_mean_pp"], 20.)
                (out / "scoring_k3/t1_states.npz").write_bytes(b"corrupted")
                with self.assertRaisesRegex(RuntimeError, "Corrupted completed scene"):
                    ns["score"](cfg, out)


if __name__ == "__main__":
    unittest.main()
