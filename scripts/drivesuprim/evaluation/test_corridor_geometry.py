from contextlib import redirect_stdout
import copy
import io
import json
import lzma
from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np
from shapely.geometry import Polygon, box

from navsim.agents import corridor_geometry as geometry
from navsim.planning.script import run_corridor_geometry as pipeline
from test_selector_bundle import fixture


def prepared(n=3):
    examples = []
    vocab = np.zeros((256, 40, 3), dtype=np.float64)
    vocab[:, :, 0] = np.linspace(.1, 4., 40)
    vocab[:, :, 1] = np.linspace(.15, 6., 40)
    vocab[0, :, 1] = np.linspace(.125, 5., 40)
    vocab[1:3, :, 1] = 0.
    for i in range(n):
        sample, outcomes, _, meta = fixture("t" + str(i), "s" + str(i))
        for key, value in outcomes.items():
            c = {m: 1. for m in pipeline.METRICS}
            c["drivable_area_compliance"] = 1. if int(key) == 2 else 0.
            if int(key) == 2:
                c["no_at_fault_collisions"] = .5
            raw = dict(pdm_score=c["drivable_area_compliance"] * c["no_at_fault_collisions"], **c)
            value["raw"], value["filtered"] = raw, raw.copy()
        examples.append(pipeline.bundles.example(sample, outcomes, vocab, meta))
    return examples, vocab


def backend(ego, road, trajectories, sampling):
    assert ego == "current-only"
    assert road == "static-only"
    assert sampling == (40, .1)
    raw = np.concatenate([np.zeros((16, 1, 3)), trajectories], axis=1)
    tracked = raw.copy()
    tracked[1, :, 1] = np.linspace(0, 5., 41)
    return dict(raw_poses=raw, rollout_poses=tracked,
                vehicle_dimensions=np.array([4., 2., 1.]), polygons=[box(-10, -2, 20, 2)])


def packet():
    examples, vocab = prepared(1)
    cache = SimpleNamespace(ego_state="current-only", drivable_area_map="static-only")
    return pipeline.make_packet(cache, examples[0], vocab, (40, .1), backend)


def source_bundle(base, mixed=False):
    examples, vocab = prepared()
    source, bundle = base / "source", base / "bundle"
    cache_index = {}
    for e in examples:
        token = e["metadata"]["token"]
        path = source / (token + ".xz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with lzma.open(path, "wb") as stream:
            pickle.dump(SimpleNamespace(ego_state="current-only", drivable_area_map="static-only"), stream)
        cache_index[token] = dict(path=str(path), sha256=pipeline.sha(path))
    # Metadata for a reserved token may be present, but its file must not be opened.
    cache_index["heldout"] = dict(path=str(base / "never-open-this"), sha256="not-a-real-hash")
    pipeline.write(source / "cache_index.json", cache_index)
    pipeline.write(source / "source_config.json", dict(proposal_sampling=dict(num_poses=40, interval_length=.1)))
    if mixed:
        # Real legacy provenance: only the experiment identity is bound in the
        # bundle; its relative inputs bind the old source's cache index/config.
        legacy = base / "legacy_experiment"
        new = base / "new_collection"
        pipeline.write(new / "cache_index.json", {"t2": cache_index.pop("t2")})
        cache_index["t0"]["path"] = "t0.xz"
        pipeline.write(source / "cache_index.json", cache_index)
        pipeline.write(source / "identity.json", pipeline.read(source / "source_config.json"))
        pipeline.write(legacy / "identity.json", dict(source=str(source), inputs={
            n: pipeline.sha(source / n) for n in ("identity.json", "cache_index.json")}))
        paths = (legacy / "identity.json", new / "cache_index.json", source / "source_config.json")
        bindings = {str(p): pipeline.sha(p) for p in paths}
    else:
        bindings = {str(source / n): pipeline.sha(source / n) for n in ("cache_index.json", "source_config.json")}
    pipeline.bundles.build(bundle, examples, vocab, bindings, {"reserved_session"})
    return examples, vocab, bundle, source


class CorridorTests(unittest.TestCase):
    def test_vehicle_corners_include_heading_and_rear_axle_offset(self):
        value = geometry.corners(np.array([0., 0., np.pi / 2]), [4., 2., 1.])
        np.testing.assert_allclose(value.mean(0), [0., 1.], atol=1e-12)
        np.testing.assert_allclose(np.sort(value[:, 0]), [-1., -1., 1., 1.], atol=1e-12)
        np.testing.assert_allclose(np.sort(value[:, 1]), [-1., -1., 3., 3.], atol=1e-12)

    def test_polygon_holes_boundaries_and_empty_map(self):
        poses = np.array([[[0., 0., 0.], [4., 0., 0.]]])
        road = Polygon([(-10,-10),(10,-10),(10,10),(-10,10)], holes=[[(3,-1),(5,-1),(5,1),(3,1)]])
        data, offsets = geometry.encode_polygons([road])
        result = geometry.check_poses(poses, [1., 1., 0.], geometry.decode_polygons(data, offsets))
        self.assertFalse(result["passed"][0])
        self.assertEqual(result["first_outside"][0], 1)
        boundary = geometry.check_poses(np.zeros((1,1,3)), [4., 2., 0.], [box(-2,-2,2,2)])
        self.assertFalse(boundary["passed"][0])
        with self.assertRaisesRegex(ValueError, "empty map"):
            geometry.check_poses(poses, [1., 1., 0.], [])

    def test_policy_preserves_baseline_and_distinguishes_tracking(self):
        p = packet()
        self.assertEqual(geometry.decide(p, "raw")["chosen"], 1)
        self.assertEqual(geometry.decide(p, "rollout")["chosen"], 2)
        p["raw_poses"][0] = p["raw_poses"][2]
        self.assertEqual(geometry.decide(p, "raw")["chosen"], 0)
        self.assertEqual(geometry.decide(p, "raw")["reason"], "baseline_passes")
        p["raw_poses"][:, 1:, 1] = 100.
        result = geometry.decide(p, "raw")
        self.assertEqual(result["chosen"], 0)
        self.assertEqual(result["reason"], "no_candidate_passes_keep_baseline")

    def test_converter_and_inference_reject_future_fields_and_ignore_labels(self):
        class CurrentOnlyCache:
            @property
            def ego_state(self):
                return "current-only"
            @property
            def drivable_area_map(self):
                return "static-only"
            def __getattr__(self, name):
                raise AssertionError("Forbidden cache field: " + name)
        class NoLabels:
            def __getitem__(self, key):
                raise AssertionError("Labels read by geometry converter")
        e, vocab = prepared(1)
        e[0]["label_raw"], e[0]["label_filtered"] = NoLabels(), NoLabels()
        p = pipeline.make_packet(CurrentOnlyCache(), e[0], vocab, (40, .1), backend)
        self.assertEqual(set(p), geometry.PACKET_KEYS)
        self.assertEqual(geometry.decide(p, "rollout")["chosen"], 2)
        p["future_tracked_objects"] = np.zeros(1)
        with self.assertRaisesRegex(ValueError, "allowlist"):
            geometry.decide(p, "raw")
        e[0]["metadata"]["role"] = "holdout"
        with self.assertRaisesRegex(ValueError, "training scenes"):
            pipeline.make_packet(CurrentOnlyCache(), e[0], vocab, (40, .1), backend)

    def test_cache_resolution_uses_only_pinned_train_tokens(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp, redirect_stdout(io.StringIO()):
            examples, _, bundle, source = source_bundle(Path(tmp).resolve())
            selected, sampling, _ = pipeline.resolve_inputs(bundle, examples)
            self.assertEqual(set(selected), {"t0", "t1", "t2"})
            self.assertEqual(sampling, (40, .1))
            path = source / "t0.xz"
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(RuntimeError, "selected metric cache"):
                pipeline.resolve_inputs(bundle, examples)

    def test_cache_resolution_combines_direct_and_transitive_legacy_inputs(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp, redirect_stdout(io.StringIO()):
            examples, _, bundle, source = source_bundle(Path(tmp).resolve(), mixed=True)
            self.assertNotIn(str(source / "cache_index.json"), pipeline.read(bundle / "identity.json")["inputs"])
            selected, sampling, bindings = pipeline.resolve_inputs(bundle, examples)
            self.assertEqual(set(selected), {"t0", "t1", "t2"})
            self.assertEqual(sampling, (40, .1))
            self.assertEqual(selected["t0"]["path"], str(source / "t0.xz"))
            self.assertEqual(selected["t0"]["index"], str(source / "cache_index.json"))
            self.assertIn(str(source / "identity.json"), bindings)
            self.assertIn(str(source / "cache_index.json"), bindings)

    def test_transitive_geometry_manifests_require_unchanged_hashes(self):
        for filename in ("cache_index.json", "identity.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp, redirect_stdout(io.StringIO()):
                examples, _, bundle, source = source_bundle(Path(tmp).resolve(), mixed=True)
                path = source / filename
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaisesRegex(RuntimeError, "Pinned geometry source changed"):
                    pipeline.resolve_inputs(bundle, examples)

    def test_grouped_protocol_inputs_and_relative_source_are_hash_bound(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp, redirect_stdout(io.StringIO()):
            base = Path(tmp).resolve()
            examples, _, bundle, source = source_bundle(base, mixed=True)
            legacy = base / "legacy_experiment"
            state = pipeline.read(legacy / "identity.json")
            state["source"] = "../source"
            pipeline.write(legacy / "identity.json", state)
            protocol = base / "addition/expansion_protocol.json"
            pipeline.write(protocol, dict(inputs={"../legacy_experiment": {
                "identity.json": pipeline.sha(legacy / "identity.json")}}))
            marker = bundle / "identity.json"
            state = pipeline.read(marker)
            state["inputs"].pop(str(legacy / "identity.json"))
            state["inputs"][str(protocol)] = pipeline.sha(protocol)
            # Only the resolver is tested here; normal CLI uses load_bundle
            # before resolving and separately verifies the bundle manifest.
            pipeline.write(marker, state)
            selected, sampling, bindings = pipeline.resolve_inputs(bundle, examples)
            self.assertEqual(set(selected), {"t0", "t1", "t2"})
            self.assertEqual(sampling, (40, .1))
            self.assertIn(str(protocol), bindings)
            self.assertIn(str(legacy / "identity.json"), bindings)
            self.assertIn(str(source / "cache_index.json"), bindings)

    def test_unbound_sibling_cache_index_is_not_a_fallback(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp, redirect_stdout(io.StringIO()):
            base = Path(tmp).resolve()
            examples, _, bundle, source = source_bundle(base, mixed=True)
            legacy = base / "legacy_experiment/identity.json"
            state = pipeline.read(legacy)
            del state["inputs"]["cache_index.json"]
            pipeline.write(legacy, state)
            marker = bundle / "identity.json"
            state = pipeline.read(marker)
            state["inputs"][str(legacy)] = pipeline.sha(legacy)
            pipeline.write(marker, state)
            self.assertTrue((source / "cache_index.json").is_file())
            with self.assertRaisesRegex(ValueError, "Missing pinned metric caches for 2/3 training tokens"):
                pipeline.resolve_inputs(bundle, examples)

    def test_full_pipeline_resume_labels_after_freeze_and_non_DAC_harm_is_reported(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp, redirect_stdout(io.StringIO()):
            base = Path(tmp).resolve()
            _, _, bundle, source = source_bundle(base, mixed=True)
            out = base / "experiment"
            before = {str(p): pipeline.sha(p) for root in (bundle, source) for p in root.rglob("*") if p.is_file()}
            actual_packet, actual_record = pipeline.make_packet, pipeline.record
            def converted(cache, example, vocab, sampling):
                return actual_packet(cache, example, vocab, sampling, backend)
            def guarded(*args):
                state = pipeline.read(out / "decisions.json")
                self.assertEqual(len(state["decisions"]), 6)
                return actual_record(*args)
            argv = ["corridor", "all", "--bundle", str(bundle), "--out", str(out)]
            with mock.patch("sys.argv", argv), mock.patch.object(pipeline, "make_packet", side_effect=converted), mock.patch.object(pipeline, "record", side_effect=guarded):
                pipeline.main()
            report = pipeline.read(out / "report.json")
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["privileged"])
            self.assertFalse(report["camera_only_comparable"])
            self.assertEqual(report["primary"], "baseline")
            rows = {r["method"]: r for r in report["rows"]}
            self.assertEqual(rows["rollout"]["candidate_DAC_mismatches"], 0)
            self.assertEqual(rows["rollout"]["gain_macro_pp"], 50.)
            self.assertEqual(rows["rollout"]["component_degradations"]["no_at_fault_collisions"], 3)
            self.assertEqual(rows["rollout"]["harm"], 3)
            self.assertEqual(report["fresh_holdout_processed"], 0)
            self.assertEqual(report["new_full_PDM_scorer_calls"], 0)
            first = pipeline.sha(out / "report.json")
            with mock.patch("sys.argv", argv), mock.patch.object(pipeline, "make_packet", side_effect=AssertionError("must reuse packets")):
                pipeline.main()
            self.assertEqual(first, pipeline.sha(out / "report.json"))
            self.assertEqual(before, {str(p): pipeline.sha(p) for root in (bundle, source) for p in root.rglob("*") if p.is_file()})
            for path in (out / "preview").glob("*.svg"):
                ET.parse(path)
            path = out / "packets/t0.npz"
            path.write_bytes(path.read_bytes() + b"changed")
            with mock.patch("sys.argv", argv), self.assertRaisesRegex(RuntimeError, "packet changed"):
                pipeline.main()

    def test_rollout_mismatch_is_preserved_and_stops_the_cli(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp, redirect_stdout(io.StringIO()):
            base = Path(tmp).resolve()
            _, _, bundle, _ = source_bundle(base)
            out = base / "wrong_geometry"
            original = pipeline.make_packet
            def incorrect_backend(*args):
                values = backend(*args)
                values["rollout_poses"][1] = values["raw_poses"][1]
                return values
            def converted(cache, example, vocab, sampling):
                return original(cache, example, vocab, sampling, incorrect_backend)
            argv = ["corridor", "all", "--bundle", str(bundle), "--out", str(out)]
            with mock.patch("sys.argv", argv), mock.patch.object(pipeline, "make_packet", side_effect=converted):
                with self.assertRaisesRegex(SystemExit, "Rollout DAC mismatch"):
                    pipeline.main()
            report = pipeline.read(out / "report.json")
            self.assertFalse(report["rollout_DAC_consistency_passed"])
            self.assertEqual(report["interpretation"], "stop_and_inspect_rollout_DAC_mismatch")
            self.assertTrue((out / "packets/t0.npz").exists())
            self.assertTrue((out / "preview/t0.svg").exists())

    def test_writer_lock_and_polygon_packet_validation(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp)
            with pipeline.locked(out):
                with self.assertRaisesRegex(RuntimeError, "Another corridor writer"):
                    with pipeline.locked(out):
                        pass
            with pipeline.locked(out):
                pass
            with self.assertRaisesRegex(ValueError, "Invalid static polygon encoding"):
                geometry.decode_polygons(np.zeros(4, dtype=np.uint8), np.array([0, 9]))

    def test_frozen_pilot_decisions_cannot_be_silently_replaced_by_full_membership(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp)
            identity = dict(tokens=["t0", "t1"])
            pipeline.write(out / "identity.json", identity)
            path = out / "packets/t0.npz"
            pipeline.save(path, **packet())
            index = dict(t0=dict(sha256=pipeline.sha(path), identity_sha256=pipeline.sha(out / "identity.json")))
            pipeline.write(out / "packet_index.json", index)
            pipeline.infer(out, identity)
            frozen = pipeline.sha(out / "decisions.json")
            path = out / "packets/t1.npz"
            pipeline.save(path, **packet())
            index["t1"] = dict(sha256=pipeline.sha(path), identity_sha256=pipeline.sha(out / "identity.json"))
            pipeline.write(out / "packet_index.json", index)
            with self.assertRaisesRegex(RuntimeError, "membership/decisions changed"):
                pipeline.infer(out, identity)
            self.assertEqual(frozen, pipeline.sha(out / "decisions.json"))


if __name__ == "__main__":
    unittest.main()
