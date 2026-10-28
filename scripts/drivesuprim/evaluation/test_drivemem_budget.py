import contextlib
import csv
import io
import json
import pickle
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path("download").resolve()))
import prepare_drivemem_budget as budget
import fetch_drivemem_budget_images as fetch
from plan_drivemem_images import CAMERAS, BASE, save_json, sha


class BudgetPilotTests(unittest.TestCase):
    def entry(self, i, size):
        path = f"openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_{i}.tgz"
        return {"path": path, "size": size, "sha256": "0" * 64, "url": BASE + "/resolve/main/" + path}

    def source(self, root):
        source, logs = root / "original", root / "logs"
        source.mkdir()
        logs.mkdir()
        rows, mapping = [], {"openscene_sensor_trainval_0": [], "openscene_sensor_trainval_1": []}
        for i in range(5):
            sid = f"2021.01.01.12.00.0{i}_veh-1"
            log = sid + "_00000_00100"
            mapping[f"openscene_sensor_trainval_{i % 2}"].append(log)
            frames = [{"token": f"{i}-{j}", "cams": {cam.upper(): {"data_path": f"{log}/{cam}/{j}.jpg"}
                        for cam in CAMERAS}} for j in range(12)]
            (logs / (log + ".pkl")).write_bytes(pickle.dumps(frames))
            rows.extend(dict(token=f"{i}-{j}", session_id=sid, log_name=log,
                             timestamp_us=j * 10_000_000, map_name="map") for j in range(3, 12))
        with (source / "inventory.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_json(source / "identity.json", {"settings": {"root": str(source), "seed": 1, "gap_s": 10,
            "sessions": {r: 1 for r in budget.ROLES}, "scenes": {r: 2 for r in budget.ROLES}},
            "sources": {"navsim/agents/drivemem.py": sha(Path("navsim/agents/drivemem.py"))},
            "navsim_log_path": str(logs), "original_sensor_path": str(root / "sensors"),
            "train_test_split": {"scene_filter": {"num_history_frames": 4}}})
        save_json(source / "splits.json", {"navtest_sessions_excluded": []})
        save_json(source / "split_identity.json", {"splits": sha(source / "splits.json"),
                                                    "inventory": sha(source / "inventory.csv")})
        return source, mapping, {0: self.entry(0, 100), 1: self.entry(1, 120)}

    def test_selection_respects_cost_and_rejects_impossible_budget(self):
        entries = {i: self.entry(i, cost) for i, cost in enumerate([10, 12, 100])}
        coverage = {0: {"a", "b", "c"}, 1: {"c", "d", "e"}, 2: {"a", "b", "c", "d", "e"}}
        selected = budget.select_archives(coverage, entries, 5, 22)
        self.assertEqual(selected, [0, 1])
        self.assertLessEqual(sum(entries[i]["size"] for i in selected), 22)
        with self.assertRaisesRegex(ValueError, "No subset found"):
            budget.select_archives(coverage, entries, 5, 21)

    def test_new_split_is_deterministic_disjoint_and_does_not_mutate_old_run(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp).resolve()
            source, mapping, entries = self.source(root)
            before = {p.name: sha(p) for p in source.iterdir()}
            target = root / "budget"
            with contextlib.redirect_stdout(io.StringIO()):
                budget.prepare(source, target, mapping, entries, {r: 1 for r in budget.ROLES}, 8, 220)
            split = json.loads((target / "splits.json").read_text())
            self.assertEqual(len(split["records"]), 40)
            sessions = [s for values in split["allocation"].values() for s in values]
            self.assertEqual(len(set(sessions)), 5)
            self.assertEqual({p.name: sha(p) for p in source.iterdir()}, before)
            frozen = sha(target / "splits.json")
            with contextlib.redirect_stdout(io.StringIO()):
                budget.prepare(source, target, mapping, entries, {r: 1 for r in budget.ROLES}, 8, 220)
            self.assertEqual(sha(target / "splits.json"), frozen)
            with self.assertRaises(FileExistsError):
                budget.prepare(source, target, mapping, entries, {r: 1 for r in budget.ROLES}, 8, 219)
            fetch.validate_plan(target, 220)
            with self.assertRaisesRegex(RuntimeError, "exceeds"):
                fetch.validate_plan(target, 219)

    def test_failure_leaves_destination_uncreated_and_reads_no_outcome_files(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp).resolve()
            source, mapping, entries = self.source(root)
            with self.assertRaisesRegex(ValueError, "No subset found"):
                budget.prepare(source, root / "budget", mapping, entries, {r: 1 for r in budget.ROLES}, 8, 219)
            self.assertFalse((root / "budget").exists())
            save_json(source / "splits.json", {"navtest_sessions_excluded": ["2021.01.01.12.00.00_veh-1"]})
            save_json(source / "split_identity.json", {"splits": sha(source / "splits.json"),
                                                        "inventory": sha(source / "inventory.csv")})
            with self.assertRaisesRegex(RuntimeError, "excluded navtest"):
                budget.prepare(source, root / "budget", mapping, entries, {r: 1 for r in budget.ROLES}, 8, 220)

    def tar(self, archive, files):
        with tarfile.open(archive, "w:gz") as tar:
            for name, content in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(content)
                tar.addfile(member, io.BytesIO(content))

    def test_extractor_only_requested_images_rejects_traversal_and_preserves_different_existing(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp).resolve()
            archive, sensors = root / "a.tgz", root / "sensors"
            self.tar(archive, {"openscene-v1.1/sensor_blobs/trainval/log/cam_f0/a.jpg": b"jpeg",
                               "openscene-v1.1/sensor_blobs/trainval/log/cam_f0/future.jpg": b"future"})
            result = fetch.extract_images(archive, sensors, {"log/cam_f0/a.jpg"})
            self.assertEqual(set(result), {"log/cam_f0/a.jpg"})
            self.assertFalse((sensors / "log/cam_f0/future.jpg").exists())
            (sensors / "log/cam_f0/a.jpg").write_bytes(b"original")
            with self.assertRaisesRegex(RuntimeError, "Existing image differs"):
                fetch.extract_images(archive, sensors, {"log/cam_f0/a.jpg"})
            self.assertEqual((sensors / "log/cam_f0/a.jpg").read_bytes(), b"original")
            self.tar(archive, {"../outside.jpg": b"bad"})
            with self.assertRaisesRegex(RuntimeError, "Unsafe archive member"):
                fetch.extract_images(archive, sensors, set())

    def test_full_local_install_resume_and_missing_image_detection(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp).resolve()
            source, mapping, entries = self.source(root)
            target = root / "budget"
            with contextlib.redirect_stdout(io.StringIO()):
                budget.prepare(source, target, mapping, entries, {r: 1 for r in budget.ROLES}, 1, 220)
            prep = target / "image_preparation"
            requests = json.loads((prep / "requests.json").read_text())
            plan = json.loads((prep / "archive_plan.json").read_text())
            local = {}
            for i, entry in enumerate(plan["routes"]["camera_log_shards"]["archives"]):
                archive = root / f"real_{i}.tgz"
                images = {image for log in entry["logs"] for image in requests["images_by_log"][log]}
                self.tar(archive, {"openscene-v1.1/sensor_blobs/trainval/" + p: p.encode() for p in images})
                entry.update(sha256=sha(archive), size=archive.stat().st_size)
                local[entry["path"]] = archive
            save_json(prep / "archive_plan.json", plan)
            report = json.loads((target / "budget_plan.json").read_text())
            report["archive_plan_sha256"] = sha(prep / "archive_plan.json")
            report["spec"]["budget_bytes"] = 100_000
            save_json(target / "budget_plan.json", report)
            with patch.object(fetch, "download", side_effect=lambda e, d: local[e["path"]]) as mocked:
                with contextlib.redirect_stdout(io.StringIO()):
                    fetch.run_stage(target, "fetch", 100_000, keep_archives=True)
                    fetch.run_stage(target, "fetch", 100_000)
                self.assertEqual(mocked.call_count, 2)
            self.assertTrue(all(p.exists() for p in local.values()))
            (root / "sensors" / requests["images"][0]).write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "missing/changed"):
                fetch.run_stage(target, "status", 100_000)

    def test_shared_verified_archive_is_reused_without_network(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp).resolve()
            entry = self.entry(0, 4)
            archive = root / Path(entry["path"]).name
            archive.write_bytes(b"data")
            entry["sha256"] = sha(archive)
            with patch.object(fetch.subprocess, "run", side_effect=AssertionError("Unexpected download")):
                self.assertEqual(fetch.download(entry, root), archive)
            archive.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "incorrect size/SHA256"):
                fetch.download(entry, root)
            self.assertEqual(archive.read_bytes(), b"changed")


if __name__ == "__main__":
    unittest.main()
