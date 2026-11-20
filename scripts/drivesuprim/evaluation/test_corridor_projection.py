from contextlib import redirect_stdout
from collections import Counter
import importlib.util
import io
import lzma
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from navsim.agents import corridor_projection as projection
from navsim.planning.script import run_corridor_projection as audit
from test_corridor_geometry import backend, prepared


def camera(distortion=None):
    # Optical camera axes: right=-ego Y, down=-ego Z, forward=ego X.
    return dict(sensor2lidar_rotation=[[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
        sensor2lidar_translation=[0, 0, 1.5],
        cam_intrinsic=[[500, 0, 960], [0, 500, 540], [0, 0, 1]],
        distortion=[0, 0, 0, 0, 0] if distortion is None else distortion,
        data_path="images/current.png")


def frame(token="t0", timestamp=1000):
    return dict(token=token, timestamp=timestamp, ego2global_translation=[0, 0, 0],
        ego2global_rotation=[1, 0, 0, 0],
        cams={name.upper(): camera() for name in projection.CAMERAS})


def fixture(base, unsupported=False):
    parent = audit.parent
    source, bundle, root = base / "data", base / "bundle", base / "geometry"
    logs, sensors = source / "logs", source / "sensors"
    examples, vocab = prepared()
    frames, records, cache_index = [], [], {}
    for i, e in enumerate(examples):
        token = e["metadata"]["token"]
        f = frame(token, 1000+i)
        for name in projection.CAMERAS:
            relative = "images/"+token+"_"+name+".png"
            f["cams"][name.upper()]["data_path"] = relative
            path = sensors / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1920, 1080), (100, 120, 140)).save(path)
        if unsupported and i == 0:
            f["cams"]["CAM_F0"]["distortion"] = [0]*12
        frames.append(f)
        records.append(dict(token=token, role="train", session_id=e["metadata"]["session_id"],
            log_name="selected_log", timestamp_us=1000+i))
        path = source / (token + ".xz")
        with lzma.open(path, "wb") as stream:
            pickle.dump(SimpleNamespace(ego_state="current-only", drivable_area_map="static-only"), stream)
        cache_index[token] = dict(path=str(path), sha256=audit.sha(path))
    # Future/holdout images must not be looked up; these frames lack all other fields.
    frames.append(dict(token="reserved_future_frame"))
    log = logs / "selected_log.pkl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        pickle.dump(frames, stream)
    audit.write(source / "cache_index.json", cache_index)
    audit.write(source / "source_config.json", dict(proposal_sampling=dict(num_poses=40, interval_length=.1),
        navsim_log_path=str(logs), original_sensor_path=str(sensors),
        agent=dict(config=dict(n_camera=3, camera_width=2048, camera_height=512))))
    audit.write(source / "splits.json", dict(records=records, log_sha256=dict(selected_log=audit.sha(log))))
    inputs = {str(source / n): audit.sha(source / n) for n in ("source_config.json", "cache_index.json", "splits.json")}
    # Historical decoder identities have abbreviated records: they are not
    # dataset frame definitions and must not be parsed as split records.
    decoder = source / "decoder/identity.json"
    audit.write(decoder, dict(records=[dict(token="t0", role="memory", source=str(source))]))
    inputs[str(decoder)] = audit.sha(decoder)
    historical = source / "old_inventory/splits.json"
    audit.write(historical, dict(records=[dict(records[0], role="development", log_name="never-open-old-role")]))
    inputs[str(historical)] = audit.sha(historical)
    parent.bundles.build(bundle, examples, vocab, inputs, {"reserved_session"})
    original = parent.make_packet
    def converted(cache, example, vocabulary, sampling):
        return original(cache, example, vocabulary, sampling, backend)
    with mock.patch("sys.argv", ["geometry", "all", "--bundle", str(bundle), "--out", str(root)]), mock.patch.object(parent, "make_packet", side_effect=converted):
        parent.main()
    return root, source


class ProjectionTests(unittest.TestCase):
    def test_camera_to_ego_projection_and_ground_plane_preserve_world_xy(self):
        c = camera()
        uv, front, depth = projection.project(np.array([[10, 0, 0], [-10, 0, 0]]), c)
        np.testing.assert_allclose(uv[0], [960, 615])
        self.assertTrue(front[0])
        self.assertFalse(front[1])
        self.assertTrue(np.isnan(uv[1]).all())
        self.assertEqual(depth[0], 10)
        pitch = .13
        r = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        observation = dict(ego_rotation=r, ego_translation=[100, 30, 5])
        world = np.array([[110, 32], [95, 31]])
        ego = projection.ground_ego_points(world, observation)
        np.testing.assert_allclose(ego[:, 2], 0, atol=1e-12)
        reconstructed = ego @ r.T + observation["ego_translation"]
        np.testing.assert_allclose(reconstructed[:, :2], world, atol=1e-12)

    def test_radial_tangential_rational_distortion_and_near_plane(self):
        c = camera([.1, 0, 0, 0, 0, .2, 0, 0])
        points = np.array([[10, -2, .5], [0, 0, 0], [.05, 0, 0]])
        uv, valid, _ = projection.project(points, c)
        radius = .2**2 + .1**2
        factor = (1+.1*radius) / (1+.2*radius)
        np.testing.assert_allclose(uv[0], [960 + 500*.2*factor, 540 + 500*.1*factor])
        self.assertEqual(valid.tolist(), [True, False, False])
        bad = camera([0]*12)
        with self.assertRaisesRegex(ValueError, "Unsupported Brown-Conrady"):
            projection.project(points, bad)
        bad["distortion"] = None
        with self.assertRaisesRegex(ValueError, "Missing distortion"):
            projection.calibration(bad)

    @unittest.skipUnless(importlib.util.find_spec("cv2") is not None, "OpenCV cross-check runs in server container")
    def test_distortion_matches_opencv(self):
        import cv2
        points = np.array([[12, -2, .3], [8, 1, -.2], [24, 4, 0]], dtype=float)
        for count in (4, 5, 8):
            c = camera([.01, -.001, .0002, -.0003, .0001, .001, -.0001, .00001][:count])
            r, t, k, d = projection.calibration(c)
            camera_points = (points-t) @ r
            expected, _ = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), k, d)
            actual, _, _ = projection.project(points, c)
            np.testing.assert_allclose(actual, expected.reshape(-1, 2), atol=1e-8, rtol=0)

    def test_crop_stitch_pixel_coordinates_and_current_body_unknown(self):
        cameras = {n: camera() for n in projection.CAMERAS}
        sizes = {n: (1920, 1080) for n in projection.CAMERAS}
        points = np.array([[10, 0, 0], [1, 0, 0]])
        result = projection.panorama_projection(points, cameras, sizes, (2048, 512))
        # First camera center: (960-416+.5)*.5-.5; vertical crop similarly.
        np.testing.assert_allclose(result["cam_l0"]["panorama"][0], [271.75, 293.25])
        self.assertTrue(result["cam_l0"]["in_crop"][0])
        self.assertFalse(result["cam_f0"]["in_crop"][1])
        mask = projection.current_body_mask(np.array([[0, 0], [4, 0]]), [0, 0, 0], [4, 2, 1])
        self.assertEqual(mask.tolist(), [True, False])
        with self.assertRaisesRegex(ValueError, "too small"):
            projection.crop_boxes({n: (800, 500) for n in projection.CAMERAS})

    def test_current_observation_allowlist_ignores_future_fields_and_misleading_lidar_keys(self):
        class CurrentFrame(dict):
            def __getitem__(self, key):
                if key not in ("token", "timestamp", "cams", "ego2global_rotation", "ego2global_translation"):
                    raise AssertionError("Future/extra field read: " + key)
                return super().__getitem__(key)
        observation = projection.current_observation(CurrentFrame(frame()))
        self.assertEqual(set(observation), {"token", "timestamp_us", "ego_rotation", "ego_translation", "cameras"})
        # No lidar2ego transform is silently composed into source camera-to-ego values.
        uv, _, _ = projection.project(np.array([[10, 0, 0]]), observation["cameras"]["cam_f0"])
        np.testing.assert_allclose(uv, [[960, 615]])

    def test_case_selection_matches_controls_by_session_and_never_uses_outside_membership(self):
        metadata = {"a": dict(session_id="s0"), "b": dict(session_id="s0"), "c": dict(session_id="s1")}
        decisions = [dict(token="a", baseline_passed=False, reason="baseline_rejected_choose_first_valid"),
                     dict(token="b", baseline_passed=True, reason="baseline_passes"),
                     dict(token="c", baseline_passed=False, reason="no_candidate_passes_keep_baseline")]
        cases = audit.select_cases(metadata, decisions)
        self.assertEqual(Counter(c["kind"] for c in cases), dict(repair=1, matched_control=1, unrepairable=1))
        self.assertEqual(next(c for c in cases if c["kind"] == "matched_control")["matched_to"], "a")

    def test_safe_image_paths_reject_escape(self):
        root = Path.cwd() / "sensors"
        for value in ("../elsewhere.png", "/root/a.jpg", "C:/a.png", "x\\a.jpg", "file.pkl"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.safe_image(root, value)

    def test_footprint_render_edges_are_perimeter_not_diagonals(self):
        corners = audit.geometry.corners(np.array([0, 0, 0]), np.array([4, 2, 1]))
        lengths = [np.linalg.norm(corners[a]-corners[b]) for a, b in projection.FOOTPRINT_EDGES]
        self.assertEqual(sorted(lengths), [2, 2, 4, 4])

    def test_metadata_graph_follows_grouped_relative_legacy_inputs_and_checks_hashes(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            base = Path(tmp).resolve()
            data, legacy, bundle = base / "data", base / "legacy", base / "bundle"
            audit.write(data / "splits.json", dict(records=[]))
            audit.write(data / "split_identity.json", dict(splits=audit.sha(data / "splits.json")))
            audit.write(legacy / "identity.json", dict(source="../data", inputs={
                "split_identity.json": audit.sha(data / "split_identity.json")}))
            audit.write(bundle / "identity.json", dict(inputs={"../legacy": {
                "identity.json": audit.sha(legacy / "identity.json")}}))
            digest = audit.sha(bundle / "identity.json")
            seen, pinned = audit.metadata_graph(bundle, digest)
            self.assertIn(data / "splits.json", seen)
            self.assertEqual(pinned[data / "splits.json"], audit.sha(data / "splits.json"))
            (data / "splits.json").write_bytes(b'{"changed": true}')
            with self.assertRaisesRegex(RuntimeError, "Pinned projection metadata changed/missing"):
                audit.metadata_graph(bundle, digest)

    def test_camera_geometry_origin_mismatch_is_rejected(self):
        examples, vocab = prepared(1)
        packet = audit.parent.make_packet(SimpleNamespace(ego_state="current-only", drivable_area_map="static-only"),
            examples[0], vocab, (40, .1), backend)
        observation = projection.current_observation(frame())
        observation["ego_translation"][0] += 1
        with self.assertRaisesRegex(ValueError, "camera ego and geometry packet origin disagree"):
            audit.audit_case(dict(observation=observation), packet, (2048, 512))

    def test_separate_cli_stages_resume_preserve_parent_and_image_tampering_fails(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp, redirect_stdout(io.StringIO()):
            base = Path(tmp).resolve()
            root, source = fixture(base)
            out = base / "projection"
            original = {str(p): audit.sha(p) for r in (root, source) for p in r.rglob("*") if p.is_file()}
            frozen = None
            for seed, stage in enumerate(("plan", "pilot", "all", "report"), 1):
                env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONDONTWRITEBYTECODE="1")
                env["PYTHONPATH"] = str(audit.REPO) + os.pathsep + env.get("PYTHONPATH", "")
                result = subprocess.run([sys.executable, str(Path(audit.__file__)), stage,
                    "--geometry", str(root), "--out", str(out)], cwd=str(audit.REPO), env=env,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                digest = audit.sha(out / "identity.json")
                if frozen is None:
                    frozen = digest
                self.assertEqual(digest, frozen)
            self.assertEqual(original, {str(p): audit.sha(p) for r in (root, source) for p in r.rglob("*") if p.is_file()})
            report = audit.read(out / "report.json")
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["rendered_cases"], 3)
            self.assertEqual(report["fresh_holdout_processed"], 0)
            self.assertTrue(report["human_projection_review_required"])
            with Image.open(next((out / "preview").glob("*.png"))) as image:
                self.assertEqual(image.size, (2048, 560))
            image = next((source / "sensors/images").glob("*.png"))
            image.write_bytes(image.read_bytes()+b"changed")
            with audit.parent.locked(out), self.assertRaisesRegex(RuntimeError, "Projection inputs/code/runtime changed"):
                audit.execute("report", root, out)

    def test_unsupported_calibration_is_preserved_as_blocked_not_pinhole_fallback(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp, redirect_stdout(io.StringIO()):
            base = Path(tmp).resolve()
            root, _ = fixture(base, unsupported=True)
            out = base / "projection"
            with audit.parent.locked(out):
                plan = audit.execute("plan", root, out)
                report = audit.execute("all", root, out)
            self.assertIn("t0", plan["blocked_cases"])
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["rendered_cases"], 2)
            self.assertFalse((out / "preview/t0.png").exists())


if __name__ == "__main__":
    unittest.main()
