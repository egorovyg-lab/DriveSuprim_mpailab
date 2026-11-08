import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import shutil
import tarfile
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("selector_dataset", Path("download/prepare_selector_dataset.py"))
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)


def row(sid, token, seconds, shard=0):
    return dict(session_id=sid, token=token, timestamp_us=int(seconds * 1e6),
                log_name=sid + "_00000_00100", map_name="city", shard=shard)


class DatasetTests(unittest.TestCase):
    def test_candidates_exclude_whole_protected_sessions_duplicates_and_nearby_anchors(self):
        sid, blocked = "train_veh-1", "closed_veh-2"
        old = [row(sid, "old", 0)]
        rows = [row(sid, "old", 0), row(sid, "near", 5), row(sid, "far", 10),
                row(sid, "duplicate-time", 10), row(blocked, "closed", 20)]
        logs = {r["log_name"] for r in rows}
        result = ds.candidate_rows(rows, old, {blocked}, logs, {log: 0 for log in logs}, 10.)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["timestamp_us"], 10_000_000)

    def test_acquisition_respects_additional_cap_and_counts_overlapping_sessions_once(self):
        rows = [row("old_veh-0", "o0", 0, 0), row("old_veh-0", "o1", 10, 0)]
        for shard, sid in ((1, "a_veh-1"), (2, "a_veh-1"), (3, "b_veh-2")):
            rows.extend([row(sid, f"{shard}a", 0, shard), row(sid, f"{shard}b", 10, shard)])
        rows.sort(key=lambda r: (r["session_id"], r["timestamp_us"]))
        entries = {i: dict(size=size) for i, size in enumerate((100, 6, 7, 6))}
        with redirect_stdout(io.StringIO()):
            chosen, caps, initial = ds.acquire(rows, entries, {0}, set(), {"old_veh-0": 1}, 12, 2, 4, 2, 10.)
        self.assertEqual(chosen, {1, 3})
        self.assertLessEqual(sum(entries[i]["size"] for i in chosen), 12)
        self.assertEqual(sum(n >= 2 for sid, n in caps.items() if sid != "old_veh-0"), 2)
        self.assertEqual(initial["old_veh-0"], 2)

    def test_camera_requests_only_use_four_past_current_frames(self):
        frames = [dict(token=str(i), cams={cam: dict(data_path=f"log/{cam}/{i}.jpg") for cam in ds.CAMERAS}) for i in range(9)]
        paths = ds.camera_paths(frames, 4)
        self.assertEqual(len(paths), 32)
        self.assertEqual({int(Path(p).stem) for p in paths}, {1, 2, 3, 4})
        with self.assertRaisesRegex(ValueError, "history"):
            ds.camera_paths(frames, 2)
        frames[4]["cams"]["cam_f0"]["data_path"] = "../../escape.jpg"
        with self.assertRaises(ValueError):
            ds.camera_paths(frames, 4)

    def test_existing_log_directory_is_not_treated_as_complete_sensor_data(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            logs, sensors = root / "logs", root / "sensors"
            logs.mkdir()
            sid = "rgb_veh-1"
            log = sid + "_00000_00100"
            frames = [dict(token=f"f{i}", cams={cam: dict(data_path=f"{log}/{cam}/{i}.jpg") for cam in ds.CAMERAS}) for i in range(45)]
            (logs / (log + ".pkl")).write_bytes(pickle.dumps(frames))
            for relative in ds.camera_paths(frames, 23):
                path = sensors / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")
            rows = [row(sid, "f23", 10), row(sid, "f43", 20)]
            with redirect_stdout(io.StringIO()):
                found = ds.complete_rgb_tokens(rows, logs, sensors, {log})
            self.assertEqual(found, {"f23"})
            (sensors / next(iter(ds.camera_paths(frames, 23)))).unlink()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(ds.complete_rgb_tokens(rows, logs, sensors, {log}), set())

    def setup_dataset(self, root, holdout=1):
        logs, sensors, cache, pool = [root / name for name in ("logs", "sensors", "archives", "fixture_pool")]
        for path in (logs, sensors, cache, pool):
            path.mkdir()
        names = ["oldm_veh-0", "oldf_veh-1", "newa_veh-2", "newb_veh-3", "newc_veh-4", "seen_veh-5", "closed_veh-6"]
        mapping, listing, records, archive_sources = {}, [], [], {}
        for shard, sid in enumerate(names):
            log = sid + "_00000_00100"
            frames = [dict(token=f"{sid}-f{i}", timestamp=i * 500_000,
                           cams={cam: dict(data_path=f"{log}/{cam}/{i}.jpg") for cam in ds.CAMERAS}) for i in range(64)]
            (logs / (log + ".pkl")).write_bytes(pickle.dumps(frames))
            for i in (3, 23, 43, 63):
                records.append(dict(token=frames[i]["token"], log_name=log, session_id=sid,
                                    timestamp_us=frames[i]["timestamp"], map_name="city"))
            archive = pool / f"openscene_sensor_trainval_camera_{shard}.tgz"
            with tarfile.open(archive, "w:gz") as tar:
                for frame in frames:
                    for camera in frame["cams"].values():
                        payload = b"test-image-content"
                        member = tarfile.TarInfo("openscene-v1.1/sensor_blobs/trainval/" + camera["data_path"])
                        member.size = len(payload)
                        tar.addfile(member, io.BytesIO(payload))
            archive_sources[shard] = archive
            if shard < 2:
                shutil.copyfile(archive, cache / archive.name)
            mapping[f"openscene_sensor_trainval_{shard}"] = [log]
            listing.append(dict(path="openscene-v1.1/openscene_sensor_trainval_camera/" + archive.name,
                                size=archive.stat().st_size, lfs=dict(oid=ds.sha(archive))))
        source, addition, inventory, validation = [root / name for name in ("source", "addition", "inventory", "validation")]
        legacy_m = dict(records[0], role="memory")
        legacy_f = dict(records[4], role="fit")
        seen = dict(records[20], role="development")
        closed = dict(records[24], role="calibration")
        for path, selected in ((source, [legacy_m, seen]), (addition, [legacy_f]),
                               (inventory, [closed]), (validation, [seen])):
            path.mkdir()
            ds.write(path / "identity.json", dict(navsim_log_path=str(logs), original_sensor_path=str(sensors)))
            ds.write(path / "splits.json", dict(records=selected, navtest_sessions_excluded=[]))
            with (path / "inventory.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            ds.write(path / "split_identity.json", dict(splits=ds.sha(path / "splits.json"), inventory=ds.sha(path / "inventory.csv")))
        ds.write(addition / "label_selection.json", dict(records=[legacy_f]))
        args = argparse.Namespace(stage="plan", source=source, addition=addition, inventory=inventory,
            validation=validation, exclude_run=[], archive_dir=cache, out=root / "expanded", max_extra_gb=.001,
            target_new_sessions=3, holdout_sessions=holdout, min_scenes=2, max_scenes=4, gap_s=10., seed=20261109)
        raw_mapping = json.dumps(mapping).encode()
        raw_listing = json.dumps(listing).encode()
        def metadata_fetch(url):
            if ds.MAPPING in url:
                return raw_mapping
            self.assertIn("/api/datasets/", url)
            self.assertNotIn(".tgz", url)
            return raw_listing
        return args, archive_sources, raw_mapping, metadata_fetch

    def planned(self, root, holdout=1):
        args, archives, raw, fetch = self.setup_dataset(root, holdout)
        with mock.patch.object(ds, "MAPPING_SHA256", hashlib.sha256(raw).hexdigest()), \
                mock.patch.object(ds, "fetch_bytes", side_effect=fetch), redirect_stdout(io.StringIO()):
            result = ds.plan(args)
        return args, archives, result

    def test_plan_freezes_protected_holdout_and_budget_without_downloads(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            args, archives, result = self.planned(Path(tmp).resolve())
            self.assertEqual(result["summary"]["legacy_train_scenes"], 2)
            self.assertEqual(result["summary"]["new_training_sessions"], 2)
            self.assertEqual(result["summary"]["fresh_holdout_sessions"], 1)
            self.assertLessEqual(result["summary"]["additional_compressed_GB"], args.max_extra_gb)
            selected = ds.read(args.out / "selection.json")
            train = {r["session_id"] for r in selected["records"] if r["role"] == "train"}
            holdout = {r["session_id"] for r in selected["records"] if r["role"] == "holdout"}
            self.assertFalse(train & holdout)
            self.assertFalse((train | holdout) & {"seen_veh-5", "closed_veh-6"})
            self.assertEqual(len(list(args.archive_dir.glob("*.tgz"))), 2)
            self.assertFalse(list((args.source).glob("**/top16_labels/*")))
            with mock.patch.object(ds, "fetch_bytes", side_effect=AssertionError("frozen metadata")), redirect_stdout(io.StringIO()):
                self.assertEqual(ds.plan(args), result)
            args.max_scenes += 1
            with self.assertRaisesRegex(RuntimeError, "options changed"):
                ds.verify(args.out, args)

    def test_fetch_extracts_verified_subset_resumes_and_detects_image_changes(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            args, archives, _ = self.planned(Path(tmp).resolve())
            _, requests = ds.verify(args.out, args)
            calls = []
            def download(entry, directory):
                self.assertFalse(entry["cached"])
                calls.append(entry["shard"])
                destination = directory / Path(entry["path"]).name
                shutil.copyfile(archives[entry["shard"]], destination)
                return destination
            with mock.patch.object(ds, "download", side_effect=download), redirect_stdout(io.StringIO()):
                report = ds.fetch(args.out, requests, args.archive_dir)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(set(calls), {2, 3, 4})
            committed, _ = ds.image_status(args.out, requests)
            self.assertEqual(set(committed), set(requests["images"]))
            # Only selected frames, not every file in the full image archives.
            self.assertLess(len(committed), 5 * 64 * 8)
            with mock.patch.object(ds, "download", side_effect=AssertionError("already complete")), redirect_stdout(io.StringIO()):
                self.assertEqual(ds.fetch(args.out, requests, args.archive_dir)["status"], "complete")
            target = ds.checked_target(Path(requests["sensor_root"]), requests["images"][0])
            target.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "image missing/changed"):
                ds.image_status(args.out, requests)

    def test_vanished_cached_archive_stops_before_network(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            args, _, _ = self.planned(Path(tmp).resolve())
            _, requests = ds.verify(args.out, args)
            cached = next(e for e in requests["archives"] if e["cached"])
            (args.archive_dir / Path(cached["path"]).name).unlink()
            with mock.patch.object(ds, "download", side_effect=AssertionError("no network allowed")), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "no additional downloads started"):
                    ds.fetch(args.out, requests, args.archive_dir)

    def test_shortfall_does_not_publish_dataset_or_download_archives(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            args, _, raw, fetch = self.setup_dataset(root, holdout=3)
            with mock.patch.object(ds, "MAPPING_SHA256", hashlib.sha256(raw).hexdigest()), \
                    mock.patch.object(ds, "fetch_bytes", side_effect=fetch), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "new eligible sessions"):
                    ds.plan(args)
            self.assertFalse((args.out / "acquisition_plan.json").exists())
            self.assertEqual(len(list(args.archive_dir.glob("*.tgz"))), 2)


if __name__ == "__main__":
    unittest.main()
