import hashlib
import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("image_plan", Path("download/plan_drivemem_images.py"))
images = importlib.util.module_from_spec(spec)
spec.loader.exec_module(images)


class ImagePlanTests(unittest.TestCase):
    def setup_run(self, root):
        logs = root / "logs"
        logs.mkdir()
        frames = [{"token": f"f{i}", "cams": {cam.upper(): {"data_path": f"log/{cam}/{i}.jpg"}
                                                for cam in images.CAMERAS}} for i in range(12)]
        path = logs / "log.pkl"
        path.write_bytes(pickle.dumps(frames))
        split = {"records": [dict(token="f3", log_name="log", session_id="s1"),
                             dict(token="f7", log_name="log", session_id="s1")],
                 "log_sha256": {"log": images.sha(path)}}
        (root / "identity.json").write_text(json.dumps({"navsim_log_path": str(logs),
            "original_sensor_path": str(root / "images"),
            "train_test_split": {"scene_filter": {"num_history_frames": 4}}}))
        (root / "inventory.csv").write_text("frozen inventory")
        self.save_split(root, split)
        return frames, split

    def save_split(self, root, split):
        (root / "splits.json").write_text(json.dumps(split))
        (root / "split_identity.json").write_text(json.dumps({"splits": images.sha(root / "splits.json"),
                                                              "inventory": images.sha(root / "inventory.csv")}))

    def test_requests_include_history_not_future_and_no_lidar(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent, Path.cwd().resolve())
            self.setup_run(root)
            result = images.collect_requests(root)
            self.assertEqual(result["scenes"], 2)
            self.assertEqual(len(result["images"]), 64)
            self.assertTrue(all(int(Path(p).stem) < 8 for p in result["images"]))
            self.assertFalse((root / "images").exists())

    def test_changed_log_hash_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent, Path.cwd().resolve())
            self.setup_run(root)
            (root / "logs/log.pkl").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "Source log changed"):
                images.collect_requests(root)

    def test_unsafe_image_paths_are_rejected(self):
        for value in ("../../bad.jpg", "/absolute.jpg", "x\\bad.jpg", "data.pcd", ""):
            with self.assertRaises(ValueError):
                images.safe_image_path(value)

    def test_archive_routes_are_based_on_log_mapping_and_verified_file_metadata(self):
        def entry(path, size):
            return {"path": path, "size": size, "lfs": {"oid": "0" * 64}}
        compact = [entry(f"navsim/navtrain_{kind}_{i}.tgz", 100) for kind in ("current", "history") for i in range(1, 33)]
        full = [entry("openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_3.tgz", 200)]
        routes, unknown = images.route_plans({"logs": ["log"]}, {"openscene_sensor_trainval_3": ["log"]}, compact, full)
        self.assertEqual(unknown, [])
        self.assertEqual(routes["camera_log_shards"]["size_upper_bound"], 200)
        self.assertEqual(routes["navtrain_compact"]["size_upper_bound"], 6400)
        with self.assertRaisesRegex(RuntimeError, "listing is incomplete"):
            images.route_plans({"logs": ["log"]}, {}, compact[:1], [])


if __name__ == "__main__":
    unittest.main()
