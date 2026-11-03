import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path("download").resolve()))
from fetch_drivemem_budget_images import download
from navsim.planning.script.run_drivemem_frozen_validation import decide, evaluate_union
from navsim.planning.script.run_drivemem_development_oracle import verify_baseline
from navsim.agents.drivemem_action_transfer import ridge_path
from navsim.agents.drivemem import METRICS
from test_drivemem_action_transfer import make_sample
import test_drivemem_budget as budget_tests
import prepare_drivemem_cached_validation as preparation


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FrozenValidationTests(unittest.TestCase):
    def test_preparer_freezes_four_new_sessions_without_changing_source(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp).resolve()
            original, mapping, entries = budget_tests.BudgetPilotTests().source(root)
            original_split = read(original / "splits.json")
            original_split["records"] = []
            write(original / "splits.json", original_split)

            def checksum(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            write(original / "split_identity.json", dict(splits=checksum(original / "splits.json"), inventory=checksum(original / "inventory.csv")))
            source, model, destination, cache = root / "source", root / "model", root / "validation", root / "archives"
            source.mkdir()
            cache.mkdir()
            write(source / "identity.json", read(original / "identity.json"))
            (source / "inventory.csv").write_bytes((original / "inventory.csv").read_bytes())
            write(source / "splits.json", dict(records=[dict(session_id="2021.01.01.12.00.00_veh-1")]))
            write(source / "split_identity.json", dict(splits=checksum(source / "splits.json"), inventory=checksum(source / "inventory.csv")))
            for entry in entries.values():
                entry.update(size=4, sha256=hashlib.sha256(b"data").hexdigest())
                (cache / Path(entry["path"]).name).write_bytes(b"data")
            write(source / "image_preparation/archive_plan.json", dict(routes={"camera_log_shards": dict(archives=list(entries.values()))}))
            write(source / "budget_plan.json", dict(archive_plan_sha256=checksum(source / "image_preparation/archive_plan.json")))
            write(model / "identity.json", dict(source=str(source), sources={}))
            (model / "ridge.npz").write_bytes(b"frozen")
            write(model / "fit.json", dict(primary="B3", ridge_sha256=checksum(model / "ridge.npz")))
            before = {str(p.relative_to(source)): checksum(p) for p in source.rglob("*") if p.is_file()}
            raw = json.dumps(mapping).encode()
            with patch.object(preparation, "fetch_bytes", return_value=raw), patch.object(preparation, "MAPPING_SHA256", hashlib.sha256(raw).hexdigest()):
                preparation.prepare(source, model, original, destination, cache)
                frozen = read(destination / "splits.json")
                self.assertEqual(len(frozen["records"]), 16)
                self.assertEqual(len(frozen["allocation"]["development"]), 4)
                self.assertTrue(all(r["role"] == "development" for r in frozen["records"]))
                self.assertNotIn("2021.01.01.12.00.00_veh-1", frozen["allocation"]["development"])
                settings = read(destination / "identity.json")["settings"]
                self.assertEqual(settings["sessions"]["memory"], 0)
                self.assertEqual(settings["sessions"]["development"], 4)
                self.assertEqual(settings["scenes"]["development"], 4)
                digest = checksum(destination / "splits.json")
                preparation.prepare(source, model, original, destination, cache)
                self.assertEqual(checksum(destination / "splits.json"), digest)
                self.assertEqual(before, {str(p.relative_to(source)): checksum(p) for p in source.rglob("*") if p.is_file()})

    def test_offline_never_downloads_missing_or_partial_archive(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            entry = dict(path="x.tgz", size=4, sha256=hashlib.sha256(b"data").hexdigest())
            with patch("fetch_drivemem_budget_images.subprocess.run", side_effect=AssertionError("Network forbidden")):
                with self.assertRaisesRegex(RuntimeError, "Offline"):
                    download(entry, root, offline=True)
                (root / "x.tgz.part").write_bytes(b"da")
                with self.assertRaisesRegex(RuntimeError, "Offline"):
                    download(entry, root, offline=True)
                (root / "x.tgz").write_bytes(b"data")
                self.assertEqual(download(entry, root, offline=True), root / "x.tgz")

    def test_frozen_proposals_do_not_need_query_labels_or_modify_models(self):
        rng = np.random.default_rng(1)
        vocab = rng.normal(size=(16, 40, 3))
        bank = [make_sample("m1", "m1", "memory", rng), make_sample("m2", "m2", "memory", rng)]
        query = make_sample("q", "q", "development", rng)
        del query["label_values"], query["label_local_indices"]
        model = ridge_path(bank, vocab, [10.])[10.]
        before = copy.deepcopy(model)
        fit = dict(topk=16, neighbors=5, beta=.5, primary="B3", configs={
            "B1": dict(h=.02, contrast=False), "B3": dict(h=.01, contrast=True)})
        a = decide(query, bank, vocab, model, fit, 1)
        self.assertEqual(a, decide(query, bank, vocab, model, fit, 1))
        self.assertEqual(set(a), {"baseline", "no_memory_ridge_full", "B1", "B3", "random_0", "random_1", "random_2", "scene_only"})
        for key in model:
            np.testing.assert_array_equal(model[key], before[key])

    def test_union_scoring_resumes_per_action_and_rechecks_baseline(self):
        sample = dict(token="q", session_id="new_session", selected_local=np.array(0), candidate_ids=np.arange(16))
        choices = {"q": dict(baseline=0, B3=1, no_memory_ridge_full=2)}
        calls, failed = [], [False]

        def evaluate(token, candidate):
            calls.append(candidate)
            if candidate == 2 and not failed[0]:
                failed[0] = True
                raise RuntimeError("interrupted")
            values = {k: 1. for k in METRICS}
            values.update(pdm_score=.5 + .1 * candidate, aggregate_without_two_frame=.5 + .1 * candidate)
            return dict(raw=dict(values), filtered=dict(values)), None

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                evaluate_union([sample], choices, evaluate, root, read, write, verify_baseline)
            self.assertEqual(set(read(root / "scenes/q.json")["outcomes"]), {"0", "1"})
            rows = evaluate_union([sample], choices, evaluate, root, read, write, verify_baseline)
            self.assertEqual(calls.count(1), 1)
            self.assertEqual(len(rows), 3)
            self.assertAlmostEqual(next(r for r in rows if r["method"] == "B3")["raw_gain_pp"], 10.)
            before = list(calls)
            evaluate_union([sample], choices, evaluate, root, read, write, verify_baseline)
            self.assertEqual(calls[len(before):], [0, 0])
            choices["q"]["B3"] = 3
            with self.assertRaisesRegex(RuntimeError, "decisions changed"):
                evaluate_union([sample], choices, evaluate, root, read, write, verify_baseline)


if __name__ == "__main__":
    unittest.main()
