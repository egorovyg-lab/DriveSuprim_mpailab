import copy
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "download"))
import prepare_drivemem_mf_expansion as prep
import plan_drivemem_mf_capacity as capacity_plan
from navsim.planning.script import run_drivemem_safety_data as runner
from test_drivemem_action_transfer import make_sample
from test_drivemem_pair_utility import outcomes


def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v), encoding="utf-8")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class SafetyDataTests(unittest.TestCase):
    def test_dense_frames_are_not_independent_slots(self):
        self.assertEqual(prep.spaced_capacity([i * 500000 for i in range(111)]), 6)
        self.assertEqual(prep.spaced_capacity([0, 0, 9999999, 10000000, 20000000]), 3)
        rows = [dict(token=str(i), timestamp_us=i * 500000, log_name="a", session_id="s") for i in range(111)]
        _, report = prep.select_pool(rows, [], {"s": "fit"}, {"a"})
        self.assertEqual(report[0]["max_spaced_scenes"], 6)
        self.assertEqual(report[0]["selected"], 0)
        eligible = capacity_plan.eligible_rows(rows, [rows[0]], {"s": "fit"})
        self.assertTrue(all(r["timestamp_us"] >= 10000000 for r in eligible))
        _, report = prep.select_pool(rows, [rows[0]], {"s": "fit"}, {"a"})
        self.assertEqual(capacity_plan.capacity(eligible, {"s": "fit"}, {"a"})["s"], report[0]["max_spaced_scenes"])

    def test_pool_excludes_old_times_and_roles_and_is_deterministic(self):
        rows = [dict(token=str(i), timestamp_us=i * 1000000, log_name="log", session_id="s") for i in range(500)]
        old = [dict(rows[100])]
        a, report = prep.select_pool(rows, old, {"s": "memory"}, {"log"})
        self.assertEqual(len(a), 16)
        self.assertTrue(all(abs(r["timestamp_us"] - 100000000) >= 10000000 for r in a))
        ts = sorted(r["timestamp_us"] for r in a)
        self.assertTrue(all(b - a >= 10000000 for a, b in zip(ts, ts[1:])))
        self.assertEqual((a, report), prep.select_pool(rows[::-1], old, {"s": "memory"}, {"log"}))
        a, report = prep.select_pool(rows[:5], old, {"s": "fit"}, {"log"})
        self.assertEqual(a, [])
        self.assertEqual(report[0]["selected"], 0)

    def test_screening_is_balanced_and_ignores_labels(self):
        rng = np.random.default_rng(4)
        samples = [make_sample(f"q{i}", "s", "fit", rng) for i in range(16)]
        a = runner.select_labels(samples, 1)
        self.assertEqual(len(a), 8)
        self.assertEqual(sum(r["sampling"] == "uniform" for r in a), 4)
        bad = copy.deepcopy(samples)
        for s in bad:
            s["label_values"][:] = np.nan
        self.assertEqual(a, runner.select_labels(bad[::-1], 1))
        with self.assertRaises(ValueError):
            runner.select_labels(samples[:-1], 1)

    def test_label_resume_commits_per_action_and_caps_new_scenes(self):
        rng = np.random.default_rng(3)
        samples = {str(i): make_sample(str(i), "s", "fit", rng) for i in range(2)}
        selection = [dict(token=k, session_id="s", role="fit", sampling="uniform") for k in samples]
        values = {k: outcomes(s, rng) for k, s in samples.items()}
        calls = []
        def evaluate(token, candidate):
            calls.append((token, candidate))
            if len(calls) == 6:
                raise RuntimeError("interrupted")
            return values[token][str(candidate)], None
        read = lambda p: json.loads(p.read_text())
        verify = lambda a, b: self.assertEqual(a, b)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp)
            (root / "features").mkdir()
            for k in samples:
                (root / "features" / (k + ".npz")).write_bytes(b"feature")
            args = (samples, selection, root, evaluate, read, write, sha, verify, 1)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                runner.collect(*args)
            report = runner.collect(*args)
            self.assertEqual(report["completed_scenes"], 1)
            self.assertFalse((root / "top16_labels/1.json").exists())
            self.assertEqual(calls.count(("0", 1)), 1)
            runner.collect(*args)
            self.assertEqual(runner.support(samples, selection, root, read, sha)["status"], "complete")
            before = len(calls)
            runner.collect(*args)
            self.assertEqual(len(calls), before)

    def test_preparer_publishes_compatible_layout_and_resumes_without_network(self):
        import csv
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            source, inv, val, out, cache = [root / n for n in ("source", "inv", "val", "out", "cache")]
            cache.mkdir()
            logs = root / "logs"
            logs.mkdir()
            archive = cache / "openscene_sensor_trainval_camera_1.tgz"
            archive.write_bytes(b"cached")
            records, rows = [], []
            for i in range(10):
                log, sid = "log" + str(i), "s" + str(i)
                (logs / (log + ".pkl")).write_bytes(b"log")
                records.append(dict(token="old" + str(i), session_id=sid, log_name=log, timestamp_us=0, role="memory" if i < 6 else "fit"))
                rows.extend(dict(token=f"{i}-{j}", session_id=sid, log_name=log, timestamp_us=j * 20000000) for j in range(1, 41))
            template = dict(settings=dict(seed=1), sources={}, navsim_log_path=str(logs))
            for p, recs in ((source, records), (inv, records), (val, [])):
                write(p / "identity.json", template)
                write(p / "splits.json", dict(records=recs, navtest_sessions_excluded=[]))
                with (p / "inventory.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                write(p / "split_identity.json", dict(splits=sha(p / "splits.json"), inventory=sha(p / "inventory.csv")))
            entry = dict(path=archive.name, size=6, sha256=sha(archive))
            write(source / "image_preparation/archive_plan.json", dict(routes=dict(camera_log_shards=dict(archives=[entry]))))
            write(source / "budget_plan.json", dict(archive_plan_sha256=sha(source / "image_preparation/archive_plan.json")))
            raw = json.dumps({"openscene_sensor_trainval_1": ["log" + str(i) for i in range(10)]}).encode()
            with patch.object(prep, "fetch_bytes", return_value=raw) as fetch, patch.object(prep, "MAPPING_SHA256", hashlib.sha256(raw).hexdigest()), patch.object(prep, "collect_requests", return_value=dict(images=[], logs=[])), contextlib.redirect_stdout(io.StringIO()):
                prep.prepare(source, inv, val, out, cache)
                self.assertEqual(len(prep.read(out / "splits.json")["records"]), 160)
                self.assertEqual(prep.read(out / "identity.json")["settings"]["sessions"]["test"], 0)
                prep.prepare(source, inv, val, out, cache)
                self.assertEqual(fetch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
