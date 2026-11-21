import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from shapely.geometry import box

from navsim.planning.script import prepare_drivemem_spatial_dac as prepare
from navsim.planning.script import run_drivemem_spatial_dac as pilot
from test_corridor_geometry import prepared, packet
from test_corridor_projection import frame


def synthetic():
    shape = (2, 3, 4)
    view = np.zeros(shape, dtype=np.int8)
    view[0, 0, 0] = -1
    entry = dict(view=view, baseline=np.array(0), order=np.arange(2), candidate_ids=np.array([41, 97]),
                 decoder=np.zeros((2, 2)), teacher_dac=np.array([-5., 5.]),
                 ego_xy=np.zeros((*shape, 2)), uv=np.zeros((*shape, 2)), times=np.array([1., 2., 3.]))
    outside = np.zeros(shape, dtype=bool)
    outside[0, 1] = True
    raw = np.ones((2, 1+len(pilot.model.METRICS)))
    raw[0, [0, pilot.DAC]] = 0
    edge = np.zeros(shape, dtype=bool)
    edge[1, 1] = True
    visual = np.zeros((int((view >= 0).sum()), 2), dtype=np.float16)
    visual[:, 0] = 4*outside[view >= 0]
    visual[:, 1] = np.arange(len(visual))
    return entry, dict(raw=raw, outside=outside, near_edge=edge), dict(aligned=visual, global_pool=visual.mean(0))


class SpatialDACTests(unittest.TestCase):
    def test_packet_projection_and_map_free_inputs(self):
        p = packet()
        examples, _ = prepared(1)
        cameras = dict(observation=prepare.proj.current_observation(frame()),
                       images={k: {"size": [1920, 1080]} for k in prepare.proj.CAMERAS})
        inp, labels = prepare.packet_arrays(p, examples[0], cameras, (2048, 512))
        self.assertEqual(inp["view"].shape, (16, 36, 4))
        self.assertTrue((inp["view"] >= 0).any())
        self.assertEqual(inp["times"][[0, -1]].tolist(), [.5, 4.])
        self.assertTrue(labels["outside"][0].any())
        self.assertFalse(labels["outside"][2].any())
        edge = prepare.near_road_edge(np.array([[.2, 1.], [1., 1.]]), [box(0., 0., 2., 2.)],
                                      np.array([True, True]))
        self.assertEqual(edge.tolist(), [True, False])
        altered = dict(p)
        altered["polygon_bytes"], altered["polygon_offsets"] = prepare.geom.encode_polygons([box(-100, -100, 100, 100)])
        example = dict(examples[0], label_raw=labels["raw"].copy())
        example["label_raw"][:, pilot.DAC] = 1.
        alt, changed = prepare.packet_arrays(altered, example, cameras, (2048, 512))
        for name in inp:
            np.testing.assert_array_equal(inp[name], alt[name])
        self.assertFalse(changed["outside"].any())

    def test_visual_controls_and_invisible_risky_corner(self):
        entry, labels, image = synthetic()
        aligned, full, shuffled = (pilot.features(entry, image, "token", method) for method in pilot.METHODS)
        np.testing.assert_array_equal(aligned[:, 2:], full[:, 2:])
        np.testing.assert_array_equal(aligned[:, 2:], shuffled[:, 2:])
        np.testing.assert_array_equal(sorted(aligned[:, 1]), sorted(shuffled[:, 1]))
        self.assertFalse(np.array_equal(aligned[:, 1], shuffled[:, 1]))
        score = pilot.dense_score(entry, labels["outside"][entry["view"] >= 0].astype(float))
        self.assertEqual(pilot.select(entry, score), 1)
        entry["view"][1, 1, 0] = -1
        self.assertEqual(pilot.select(entry, score), 0)

    def test_image_digest_and_camera_crop_order(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            records = {}
            for i, name in enumerate(prepare.proj.CAMERAS):
                path = root/(name+".png")
                Image.new("RGB", (900, 64), (60*i, 40, 60)).save(path)
                records[name] = dict(relative=path.name, size=[900, 64], sha256=pilot.io.sha(path))
            pixels = pilot.camera_image(root, {"images": records}, prepare.proj.CAMERAS, (1036, 8))
            self.assertEqual(pixels.shape, (8, 1036, 3))
            self.assertEqual(pixels[4, [0, 68, 968], 0].tolist(), [0, 60, 120])
            records["cam_f0"]["sha256"] = ""
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                pilot.camera_image(root, {"images": records}, prepare.proj.CAMERAS, (1036, 8))

    @unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("torch", "torchvision")),
                         "Requires PyTorch and torchvision")
    def test_encoder_weights_and_actual_cpu_feature_export(self):
        import torch
        from torchvision.models import resnet34

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            data, images, out = (root/n for n in ("data", "images", "out"))
            data.mkdir()
            images.mkdir()
            source = resnet34(weights=None)
            checkpoint = root/"ds.ckpt"
            prefix = "agent."+prepare.PREFIX
            torch.save({"state_dict": {prefix+n: w for n, w in source.state_dict().items()}}, checkpoint)
            weights = prepare.r34_weights(checkpoint)
            pilot.io.save(data/"r34.npz", **weights)
            encoder = pilot.frozen_encoder(weights, "cpu")
            self.assertFalse(encoder.training)
            self.assertTrue(torch.equal(encoder.layer3[-1].conv2.weight, source.layer3[-1].conv2.weight))
            records = {}
            for name in prepare.proj.CAMERAS:
                path = images/(name+".png")
                Image.new("RGB", (900, 80), (60, 70, 100)).save(path)
                records[name] = dict(relative=path.name, size=[900, 80], sha256=pilot.io.sha(path))
            token = f"{1:016x}"
            entry, _, _ = synthetic()
            entry["uv"][:] = [20, 10]
            name = f"inputs/{token}.npz"
            pilot.io.save(data/name, **entry)
            manifest = dict(cohort=[dict(token=token, images=records)], images=prepare.proj.CAMERAS,
                            image_size=[96, 32], files={p: pilot.io.sha(data/p) for p in ("r34.npz", name)})
            pilot.io.write(data/"manifest.json", manifest)
            self.assertEqual(pilot.export(data, images, out, manifest, range(1), "cpu")["scenes"], 1)
            features = pilot.io.load_npz(out/f"features/{token}.npz")
            self.assertEqual(features["aligned"].shape, (23, 256))
            self.assertEqual(features["global_pool"].shape, (256,))
            self.assertTrue(np.isfinite(features["aligned"]).all())
            self.assertTrue((features["global_pool"] > 0).any())

    def test_oof_reads_training_labels_then_commits_before_evaluation_and_resumes(self):
        entry, labels, image = synthetic()
        with tempfile.TemporaryDirectory() as scratch:
            io = pilot.io
            data, out = Path(scratch)/"data", Path(scratch)/"out"
            rows = [dict(token=f"{i:016x}", session_id=f"session{i}") for i in range(8)]
            manifest = dict(protocol="drivemem_spatial_dac_r34_v1", images=list(prepare.proj.CAMERAS),
                            cohort=rows, groups=pilot.model.partition(rows), files={})
            hashes = {}
            io.save(data/"r34.npz", fake=np.zeros(1))
            manifest["files"]["r34.npz"] = io.sha(data/"r34.npz")
            for row in rows:
                token = row["token"]
                for section, arrays in (("inputs", entry), ("labels", labels)):
                    name = f"{section}/{token}.npz"
                    io.save(data/name, **arrays)
                    manifest["files"][name] = io.sha(data/name)
                name = f"features/{token}.npz"
                io.save(out/name, **image)
                hashes[name] = io.sha(out/name)
            io.write(data/"manifest.json", manifest)
            io.write(out/"features.json", hashes)
            io.write(out/"feature_identity.json", dict(manifest=io.sha(data/"manifest.json"), code=io.sha(pilot.__file__)))
            expected_reads = [row["token"] for held in manifest["groups"] for row in rows if row["session_id"] not in held]
            first_reads = []
            read = pilot.read_arrays

            def record(root, m, name):
                if name.startswith("labels/") and not (out/"decision_commit.json").is_file():
                    first_reads.append(Path(name).stem)
                return read(root, m, name)

            with mock.patch.object(pilot, "read_arrays", side_effect=record):
                result = pilot.study(data, out, pilot.read_manifest(data))
            self.assertEqual(first_reads, expected_reads)
            self.assertGreater(result["methods"]["aligned"]["point_detection"]["ap"],
                               result["methods"]["shuffled"]["point_detection"]["ap"])
            self.assertEqual(result["methods"]["aligned"]["edge_negative_points"], 32)
            self.assertEqual(result["methods"]["teacher"]["repairs"], 8)
            self.assertTrue((out/"per_session.csv").is_file())
            with mock.patch.object(pilot, "fit", side_effect=AssertionError("Must resume without refit")):
                self.assertEqual(pilot.study(data, out, manifest), result)


if __name__ == "__main__":
    unittest.main()
