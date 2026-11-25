from pathlib import Path
from math import pi
import json
import shlex
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import navsim_night as night
from . import navsim_straight as straight


def example_plan():
    return {"protocol": night.PROTOCOL, "screened_scenes": 2, "rejected": {},
            "variants": {name: list(values) for name, values in night.VARIANTS.items()},
            "scenes": [
                {"token": "a", "session_id": "train", "pilot_session": False,
                 "leader_category": "free-road", "speed_mps": 4.0, "leaders": []},
                {"token": "b", "session_id": "pilot", "pilot_session": True,
                 "leader_category": "stopped-leader", "speed_mps": 2.0,
                 "leaders": [{"gap_m": 8.0, "speed_mps": 0.0}]},
            ]}


class NightTests(unittest.TestCase):
    def test_brake_gap_replays_historical_requests_and_poses_without_numpy(self):
        scene = {**example_plan()["scenes"][0], "corridor_objects": [], "ego_bounds": {"front_m": 3, "rear_m": -1}}
        archived = straight.build_plan(scene, lambda _: ("maintain", None), observation_variant="corridor-preview")
        with patch.dict(sys.modules, {"numpy": None}):
            a, b = night.check_brake_gap_trace(scene, archived)
            self.assertEqual(a["poses"], b["poses"])
            archived["decisions"][0]["request"]["question"]["instructions"] += " changed"
            with self.assertRaisesRegex(RuntimeError, "Archive request parity failed"):
                night.check_brake_gap_trace(scene, archived)

    def test_brake_gap_pose_validation_preserves_geometry_gates(self):
        valid = [[i * .1, 0, 0] for i in range(1, straight.POSES + 1)]
        night.check_brake_gap_pose_rows({"token": "a", "poses": valid})
        invalid_rows = ([4, 0], [float("nan"), 0, 0], [float("inf"), 0, 0],
                        [4, float("nan"), 0], [4, 0.1, 0], [4, 0, .1],
                        [3.8, 0, 0], [41, 0, 0], ["4.0", 0, 0])
        for poses in (valid[:-1], [[-.1, 0, 0], *valid[1:]],
                      *([*valid[:-1], row] for row in invalid_rows)):
            with self.subTest(last=poses[-1], count=len(poses)), \
                    self.assertRaisesRegex(ValueError, "Invalid straight trajectory"):
                night.check_brake_gap_pose_rows({"token": "a", "poses": poses})

    def test_brake_gap_tokenizer_gate_rejects_truncation(self):
        def prep(requests, max_length):
            n = min(len(requests[0]), max_length)
            return SimpleNamespace(inputs={
                "input_ids": SimpleNamespace(tolist=lambda: [list(range(n))]),
                "attention_mask": SimpleNamespace(tolist=lambda: [[1] * n])})

        model = SimpleNamespace(prepare=prep)
        night.no_brake_gap_truncation(model, range(80))
        with self.assertRaisesRegex(RuntimeError, "truncation"):
            night.no_brake_gap_truncation(model, range(2100))

    def test_brake_gap_rejects_an_invisible_field(self):
        scene = {**example_plan()["scenes"][1], "corridor_objects": [], "ego_bounds": {"front_m": 3, "rear_m": -1}}
        trial = straight.build_plan(scene, lambda _: ("maintain", None),
                                    observation_variant="corridor-preview", lead_brake_gap=True)
        request = trial["decisions"][0]["request"]
        with patch.object(night, "no_brake_gap_truncation", return_value=[42]):
            with self.assertRaisesRegex(RuntimeError, "erased"):
                night.check_brake_gap_token_pair(None, request)
        with patch.object(night, "no_brake_gap_truncation", side_effect=([42], [42, 17])) as tokens:
            night.check_brake_gap_token_pair(None, request)
        self.assertEqual(tokens.call_args_list[0].args[1], night.without_brake_gap(request))
        self.assertEqual(tokens.call_args_list[1].args[1], request)

    def test_brake_gap_receipt_rejects_predictions_and_smoke_transplanted_between_attempts(self):
        chain = {"attempt": "/runs/attempt-001", "manifest_sha256": "m1", "baseline_sha256": "cpu1"}
        smoke = {"status": "passed", **chain, "runtime": {"weights_sha256": "w"}, "provenance_sha256": "p"}
        cases = {"a": {"token": "a", "poses": []}}
        receipt = night.brake_gap_receipt(chain, smoke, cases)
        night.check_brake_gap_receipt(receipt, chain, smoke, cases, {"a"})
        other = {**chain, "attempt": "/runs/attempt-002", "manifest_sha256": "m2", "baseline_sha256": "cpu2"}
        other_smoke = {**smoke, **other}
        for smoke_for_score in (smoke, other_smoke):
            with self.assertRaisesRegex(RuntimeError, "does not belong"):
                night.check_brake_gap_receipt(receipt, other, smoke_for_score, cases, {"a"})
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            night.check_brake_gap_receipt(receipt, chain, smoke, {}, {"a"})

    def test_brake_gap_artifact_does_not_overwrite_existing_destination(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "receipt.json"
            night.write_brake_gap_artifact(target, {"original": True})
            with self.assertRaises(FileExistsError):
                night.write_brake_gap_artifact(target, {"original": False})
            self.assertEqual(straight.read(target), {"original": True})

    def test_brake_gap_score_stops_at_first_protected_regression(self):
        plan = example_plan()
        report_hash = straight.digest(plan)
        export = {"observation_sha256": report_hash, "objects": {"a": [], "b": []}}
        good = {key: 1.0 for key in (*night.BRAKE_GAP_PROTECTED, "ego_progress", "pdm_score")}
        regressed = {**good, "lane_keeping": 0.0}
        archived = {t: {"token": t, "poses": [[1.0, 0, 0]]} for t in ("a", "b")}
        predictions = {t: {"token": t, "poses": [[2.0, 0, 0]]} for t in ("a", "b")}
        reference = dict.fromkeys(archived, {"ds": good, "scores": {night.BRAKE_GAP_CONTROL: good}})
        payload = {"plan": plan, "export": export, "reference": reference, "archived": archived}
        context = Mock(return_value=({}, {}, None, {}, {}))
        evaluate = Mock(side_effect=lambda case: good if case["poses"][0][0] == 1.0 else regressed)
        for_scene = Mock(return_value=(good, evaluate))
        args = SimpleNamespace(repo=Path.cwd(), source=Path.cwd())
        config = SimpleNamespace(config_core=lambda _: {})
        with patch.dict(sys.modules, rollout_replay=config), patch.object(night, "archive_context", context), \
                patch.object(night, "archive_evaluator", return_value=(for_scene, ())), \
                patch.object(night, "tree_sha256", return_value="frozen"), patch.object(night, "version", return_value="mock"):
            baseline = night.score_brake_gap(args, payload)
            self.assertEqual(baseline["status"], "baseline_verified")
            self.assertEqual(len(baseline["cases"]), 2)
            for_scene.reset_mock()
            outcome = night.score_brake_gap(args, {**payload, "predictions": predictions,
                                                   "expected_scorer": baseline["scorer"]})
            for_scene.assert_called_once()
            self.assertEqual(outcome["status"], "rejected")
            self.assertEqual(outcome["cases"][0]["regressions"], ["lane_keeping"])
            self.assertEqual(outcome["unscored_tokens"], ["b"])
            self.assertNotIn("summary", outcome)

    def test_regression_selection_catches_ttc_only_and_keeps_the_rescued_scene(self):
        plan = example_plan()
        plan["scenes"][0]["token"] = night.COLLISION_TOKEN
        binding = straight.digest(plan)
        good = dict(pdm_score=1.0, no_at_fault_collisions=1.0, time_to_collision_within_bound=1.0)
        contact = dict(pdm_score=0.0, no_at_fault_collisions=0.0, time_to_collision_within_bound=0.0)
        ttc_only = dict(pdm_score=.64, no_at_fault_collisions=1.0, time_to_collision_within_bound=0.0)
        def report(name, first, second):
            return {"observation_sha256": binding, "cases": [
                {"token": token, "ds": good, "scores": {name: metrics}}
                for token, metrics in zip((night.COLLISION_TOKEN, "b"), (first, second))]}

        a, b = night.PREVIEW_SETUPS
        suite = report("control", contact, good)
        corridor = report(night.CORRIDOR_VARIANT, contact, good)
        preview = {"variants": {a: report(a, good, good), b: report(b, good, ttc_only)}}
        cases = night.regression_cases(plan, suite, corridor, preview)
        self.assertEqual(set(cases), {night.COLLISION_TOKEN, "b"})
        self.assertTrue(cases[night.COLLISION_TOKEN]["positive_control"])
        self.assertEqual(cases["b"]["regressions"], {a: [], b: ["time_to_collision_within_bound"]})

    def test_archived_regression_replay_checks_metrics_and_labels_object_inputs(self):
        plan = example_plan()
        binding = straight.digest(plan)
        good = {"pdm_score": 1.0}
        diagnostic = {"metrics": good, "initial_leader_track_tokens": ["vehicle"],
                      "contacts_through_first_at_fault": [], "ttc_geometric_candidates": [{"track_token": "vehicle"}]}
        evaluate = Mock(return_value=diagnostic)
        reference = {"regressions": {}, "positive_control": False, "ds": good,
                     "scores": dict.fromkeys(night.REGRESSION_SETUPS, good)}
        trajectory = {"token": "a", "poses": [[1.0, 0.0, 0.0]], "decisions": []}
        payload = {"plan": plan, "observation_sha256": binding,
                   "export": {"observation_sha256": binding, "objects": {
                       "a": [{"track_token": "vehicle"}], "b": []}},
                   "selected": {"a": reference}, "predictions": {"a": dict.fromkeys(night.REGRESSION_SETUPS, trajectory)}}
        with patch.object(night, "archive_evaluator", return_value=(lambda _: ({**diagnostic}, evaluate), ())):
            report = night.score_regressions(None, payload)
            self.assertEqual(evaluate.call_count, 1)
            paths = report["cases"][0]["profiles"]
            self.assertTrue(paths["corridor-preview"]["ttc_geometric_candidates"][0]["in_initial_leader_radar"])
            self.assertTrue(paths["corridor-preview"]["ttc_geometric_candidates"][0]["corridor_object_field_used"])
            self.assertFalse(paths["control"]["ttc_geometric_candidates"][0]["corridor_object_field_used"])
            reference["scores"]["control"] = {"pdm_score": .9}
            with self.assertRaisesRegex(RuntimeError, "Archived raw PDM did not reproduce"):
                night.score_regressions(None, payload)

    def test_preview_distinguishes_acceleration_from_holding_at_a_narrow_gap(self):
        parked = {"type": "VEHICLE", "x_m": 10.3315, "y_m": -3.359, "heading_rad": .5965,
                  "length_m": 4.8091, "width_m": 1.8969, "vx_mps": 0.0, "vy_mps": 0.0,
                  "lateral_reach_m": 3.284, "half_length_along_path_m": 2.5221}
        scene = {"token": "preview", "speed_mps": 1.117, "leaders": [], "corridor_objects": [parked],
                 "ego_bounds": {"front_m": 3.8, "rear_m": -1.2}}
        choose = lambda _: ("maintain", None)
        old = straight.build_plan(scene, choose, observation_variant=night.CORRIDOR_VARIANT)
        preview = straight.build_plan(scene, choose, observation_variant="corridor-preview")
        instructed = straight.build_plan(scene, choose, "corridor-risk", "corridor-preview")
        self.assertEqual(old["poses"], preview["poses"])
        self.assertEqual(preview["poses"], instructed["poses"])
        states = [run["decisions"][0]["request"]["state"] for run in (old, preview, instructed)]
        self.assertEqual(states[1], states[2])
        old_forecast = states[0]["action_effects"]["forecast"]
        forecast = states[1]["action_effects"]["forecast"]
        self.assertEqual(forecast["proceed"]["near_path_overlap_first_s"], 1.9)
        self.assertAlmostEqual(forecast["proceed"]["near_path_clearance_with_margin_m"], -.175)
        self.assertGreater(forecast["maintain"]["near_path_clearance_with_margin_m"], 0)
        self.assertIsNone(forecast["maintain"]["near_path_overlap_first_s"])
        self.assertEqual({key: forecast["proceed"][key] for key in old_forecast["proceed"]}, old_forecast["proceed"])
        clear = straight.build_plan({**scene, "corridor_objects": [{**parked, "y_m": -3.7}]}, choose,
                                    observation_variant="corridor-preview")
        clear_forecast = clear["decisions"][0]["request"]["state"]["action_effects"]["forecast"]
        self.assertGreater(clear_forecast["proceed"]["near_path_clearance_with_margin_m"], 0)
        self.assertIsNone(clear_forecast["proceed"]["near_path_overlap_first_s"])
        empty = straight.build_plan({**scene, "corridor_objects": []}, choose, observation_variant="corridor-preview")
        empty_forecast = empty["decisions"][0]["request"]["state"]["action_effects"]["forecast"]
        self.assertIsNone(empty_forecast["proceed"]["near_path_clearance_with_margin_m"])

    def test_corridor_includes_a_rotated_vehicle_outside_the_leader_strip(self):
        import numpy as np

        def actor(token, y, heading):
            center = SimpleNamespace(x=np.float64(10.33), y=np.float64(y), heading=np.float64(heading))
            return SimpleNamespace(track_token=token, center=center, tracked_object_type=SimpleNamespace(name="VEHICLE"),
                                   box=SimpleNamespace(center=center, length=np.float64(4.8), width=np.float32(2.0)))

        ego = SimpleNamespace(rear_axle=SimpleNamespace(x=0.0, y=0.0, heading=0.0),
                              dynamic_car_state=SimpleNamespace(speed=1.117),
                              car_footprint=SimpleNamespace(vehicle_parameters=SimpleNamespace(
                                  rear_axle_to_center=1.3, length=4.8, width=np.float64(2.0))))
        cache = SimpleNamespace(ego_state=ego, current_tracked_objects=[SimpleNamespace(tracked_objects=[
            actor("leader", 0.0, 0.0), actor("parallel-side", -3.36, 0.0),
            actor("rotated-side", -3.36, pi / 2), actor("near-clear", 2.4, 0.0),
        ])])
        objects = night.corridor_objects(cache)
        self.assertEqual(json.loads(json.dumps(objects, allow_nan=False)), objects)
        self.assertEqual([obj["track_token"] for obj in objects], ["near-clear", "rotated-side"])
        self.assertEqual([obj["projected_lateral_overlap"] for obj in objects], [False, True])
        record = {"token": "example", "speed_mps": 1.117, "leaders": [], "corridor_objects": objects}
        choose = lambda _: ("maintain", None)
        base = straight.build_plan(record, choose, observation_variant="action-effects")
        trial = straight.build_plan(record, choose, observation_variant=night.CORRIDOR_VARIANT)
        self.assertEqual(base["poses"], trial["poses"])
        old = base["decisions"][0]["request"]
        new = trial["decisions"][0]["request"]
        self.assertEqual(new["question"], old["question"])
        exported = new["state"].pop("near_path_objects")
        self.assertEqual(new, old)
        self.assertNotIn("track_token", exported[1])
        self.assertEqual(exported[1]["lateral_clearance_m"], -.04)
        self.assertEqual(exported[0]["lateral_clearance_m"], .4)

    def test_corridor_report_tracks_coverage_and_a_matched_collision_case(self):
        plan = example_plan()
        plan["scenes"][0]["token"] = night.COLLISION_TOKEN
        tokens = night.check_plan(plan)
        binding = straight.digest(plan)
        exported = {"observation_sha256": binding, "objects": {
            tokens[0]: [{"track_token": "obstacle", "projected_lateral_overlap": True}], tokens[1]: []}}
        ds = {"pdm_score": .8, "history_comfort": 1.0}
        reference = {"observation_sha256": binding, "cases": [
            {"token": token, "ds": ds, "scores": {"control": ds}} for token in tokens]}
        predictions = {token: {"token": token, "poses": [[4.0, 0, 0]], "actions": ["maintain"]} for token in tokens}
        payload = {"plan": plan, "export": exported, "reference": reference, "predictions": predictions,
                   "prediction_plan_sha256": straight.digest(night.with_corridor(plan, exported)),
                   "collision_track_token": "obstacle"}
        with patch.object(night, "archive_evaluator", return_value=(lambda _: (ds, lambda _: ds), ("history_comfort",))):
            result = night.score_corridor(None, payload)
        self.assertTrue(result["collision_track_in_input"])
        self.assertEqual(result["collision_case"]["object_category"], "projected_overlap")
        self.assertEqual(result["by_object_category"]["no_extra_objects"]["vs_control"]["ties"], 1)

    def test_preview_batch_reuses_only_identical_poses_on_the_same_scene(self):
        plan = example_plan()
        tokens = night.check_plan(plan)
        binding = straight.digest(plan)
        exported = {"observation_sha256": binding, "objects": {token: [] for token in tokens}}
        ds = {"pdm_score": .8, "history_comfort": 1.0}
        reference = {"observation_sha256": binding, "cases": [
            {"token": token, "ds": ds, "scores": {"control": ds}} for token in tokens]}
        predictions = {name: {token: {"token": token, "poses": [[4.0, 0, 0]], "actions": ["maintain"]}
                              for token in tokens} for name in night.PREVIEW_SETUPS}
        predictions["corridor-preview-instruction"]["b"]["poses"] = [[5.0, 0, 0]]
        payload = {"plan": plan, "export": exported, "reference": reference, "predictions_by_variant": predictions,
                   "prediction_plan_sha256": straight.digest(night.with_corridor(plan, exported))}
        replay = Mock(side_effect=lambda x: {**ds, "pdm_score": x["poses"][-1][0] / 10})
        evaluate_scene = Mock(return_value=(ds, replay))
        with patch.object(night, "archive_evaluator", return_value=(evaluate_scene, ("history_comfort",))) as init:
            result = night.score_previews(None, payload)
        self.assertEqual(init.call_count, 1)
        self.assertEqual(evaluate_scene.call_count, 2)
        self.assertEqual(replay.call_count, 3)
        self.assertEqual(result["scoring"]["ds_replays"], 2)
        self.assertEqual(result["scoring"]["unique_drex_replays"], 3)
        self.assertEqual(result["scoring"]["reused_drex_scores"], 1)
        self.assertAlmostEqual(result["variants"]["corridor-preview-instruction"]["all"]["mean_pdm"], .45)

    def test_one_model_and_per_scene_resume(self):
        plan = example_plan()
        choose = Mock(return_value=("maintain", None))
        load = Mock(return_value=(choose, 1.0))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            straight.write(root / "observations.json", plan)
            night.predict_night(root, root, load)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(choose.call_count, len(night.VARIANTS) * 2 * 8)
            night.predict_night(root, root, load)
            self.assertEqual(load.call_count, 1)
            (root / "dual-preview" / "b.json").unlink()
            night.predict_night(root, root, load)
            self.assertEqual(load.call_count, 2)
            self.assertEqual(choose.call_count, (len(night.VARIANTS) * 2 + 1) * 8)
            compact = night.compact_predictions(root, plan)
            self.assertEqual(set(compact), {"a", "b"})
            self.assertEqual(set(compact["a"]), set(night.VARIANTS))

    def test_scoring_reports_missing_predictions_and_the_common_cohort(self):
        plan = example_plan()
        predictions = {row["token"]: {name: {"token": row["token"], "name": name} for name in night.VARIANTS}
                       for row in plan["scenes"]}
        predictions["b"]["leader-context"]["error"] = "test failure"
        ds = {"pdm_score": .8, "history_comfort": 1.0}
        def evaluator(_):
            return ds, lambda row: {**ds, "pdm_score": .7 if row["name"] == "control" else .75}
        with patch.object(night, "archive_evaluator", return_value=(evaluator, ("history_comfort",))):
            report = night.score(None, {"plan": plan, "predictions": predictions})
        self.assertEqual(report["common_scenes"], 1)
        self.assertEqual(report["predicted_trajectories"], len(night.VARIANTS) * 2 - 1)
        self.assertEqual(report["variants"]["leader-context"]["prediction_failures"], 1)
        result = report["variants"]["leader-context"]["outside_pilot_sessions"]
        self.assertAlmostEqual(result["vs_control"]["mean_delta"]["pdm_score"], .05)
        self.assertAlmostEqual(result["vs_ds"]["mean_delta"]["pdm_score"], -.05)

    def test_probes_use_the_original_trace_and_deduplicate_trajectories(self):
        plan = example_plan()
        cases = [{"token": row["token"], "scores": {"control": {"pdm_score": 1}}} for row in plan["scenes"]]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "control").mkdir()
            for row in plan["scenes"]:
                result = straight.build_plan(row, lambda _: ("maintain", None), observation_variant="action-effects")
                straight.write(root / "control" / f'{row["token"]}.json', result)
            payload = night.make_probes(root, plan, {"cases": cases})
        probes = payload["a"]["probes"]
        self.assertEqual(len(probes), 24)
        self.assertEqual({row["after"] for row in probes}, {"proceed", "slow", "stop"})
        self.assertEqual({row["decision_index"] for row in probes}, set(range(8)))

    def test_astra_transport_compiles_both_sources_with_argument_headroom(self):
        with patch.object(straight, "on_astra") as invoke:
            night.on_astra("astra-night-score", SimpleNamespace(limit=8), {})
        code = invoke.call_args.kwargs["source_code"]
        compile(code, "<night-transport>", "exec")
        remote = shlex.join(["docker", "exec", "container", "python", "-c", code])
        self.assertLess(len(remote.encode("utf-8")), 120_000)

    def test_collision_profiles_reproduce_control_and_hold_braking(self):
        scene = {"token": night.COLLISION_TOKEN, "speed_mps": 4.0, "leaders": []}
        original = straight.build_plan(scene, lambda _: ("proceed", None), observation_variant="action-effects")
        profiles = night.collision_profiles(scene, original)
        self.assertEqual(len(profiles), 6)
        self.assertEqual(profiles["control"]["poses"], original["poses"])
        braking = profiles["stop_from_start"]["poses"]
        self.assertAlmostEqual(braking[-1][0], 4.0**2 / (2 * 3.0))
        self.assertEqual(braking[-1], braking[-2])
        self.assertEqual(profiles["stop_after_0_5s"]["poses"][:5], original["poses"][:5])
        self.assertEqual(profiles["stop_after_1s"]["poses"][:10], original["poses"][:10])

    def test_collision_review_explains_initial_leader_filter(self):
        self.assertEqual(night.initial_leader_filter(None), "absent_from_initial_tracks")
        actor = {"type": "PEDESTRIAN", "longitudinal_m": 10.0, "lateral_m": 0.0}
        self.assertEqual(night.initial_leader_filter(actor), "non_vehicle")
        self.assertEqual(night.initial_leader_filter({**actor, "type": "VEHICLE", "lateral_m": 4.0}),
                         "outside_lateral_band")
        self.assertEqual(night.initial_leader_filter({**actor, "type": "VEHICLE", "longitudinal_m": 80.0}),
                         "outside_forward_range")

    def test_contact_review_uses_scorer_time_and_the_tracks_at_each_frame(self):
        from enum import Enum
        import numpy as np

        kind = Enum("CollisionType", "ACTIVE_FRONT_COLLISION STOPPED_TRACK_COLLISION ACTIVE_LATERAL_COLLISION REAR")
        def actor(token, y):
            return SimpleNamespace(track_token=token, center=SimpleNamespace(x=10.0, y=y),
                                   tracked_object_type=SimpleNamespace(name="VEHICLE"))

        side = actor("side", 3.1)
        entries = [actor("rear", 0.0), side]

        class Frame(dict):
            @property
            def token_to_idx(self):
                return self

            def intersects(self, ego):
                return ego

        class Replay(list):
            red_light_token = "red_light"
            collided_track_ids = []
            unique_objects = {track.track_token: track for track in entries}
            detections_tracks = [SimpleNamespace(tracked_objects=entries),
                                 SimpleNamespace(tracked_objects=[entries[0], actor("side", .4)])]

        state_index = SimpleNamespace(X=0, Y=1, VELOCITY_X=2, VELOCITY_Y=3)
        modules = {
            "nuplan.planning.metrics.utils.collision_utils": SimpleNamespace(CollisionType=kind),
            "navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils": SimpleNamespace(
                get_collision_type=lambda _, __, track, ___: (
                    kind.ACTIVE_FRONT_COLLISION if track.track_token == "side" else kind.REAR)),
            "navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums": SimpleNamespace(
                EgoAreaIndex=SimpleNamespace(MULTIPLE_LANES=0, NON_DRIVABLE_AREA=1), StateIndex=state_index),
        }
        scene = SimpleNamespace(ego_state=SimpleNamespace(rear_axle=SimpleNamespace(x=0, y=0, heading=0)),
                                current_tracked_objects=[SimpleNamespace(tracked_objects=entries)])
        polygons = np.empty((2, 2), dtype=object)
        polygons[1, 0], polygons[1, 1] = ["rear"], ["rear", "side"]
        scorer = SimpleNamespace(_observation=Replay([Frame(rear=None, side=None)] * 2), _ego_polygons=polygons,
                                 _ego_areas=np.zeros((2, 2, 2), dtype=bool),
                                 proposal_sampling=SimpleNamespace(interval_length=.1),
                                 time_to_at_fault_collision=lambda _: .1, time_to_ttc_infraction=lambda _: 0.)
        with patch.dict(sys.modules, modules):
            result = night.inspect_contacts(scene, scorer, np.zeros((2, 4)))
        before, fault = result["contacts_through_first_at_fault"]
        self.assertFalse(before["at_fault"])
        self.assertTrue(fault["at_fault"])
        self.assertEqual(fault["initial_leader_filter"], "outside_lateral_band")
        self.assertEqual(fault["track_at_contact"]["lateral_m"], .4)
        self.assertEqual(result["at_fault_collision_time_s"], .1)
        polygons[1, 0], polygons[1, 1] = [], []
        scorer.time_to_at_fault_collision = lambda _: np.float64(np.inf)
        with patch.dict(sys.modules, modules), np.errstate(invalid="raise"):
            result = night.inspect_contacts(scene, scorer, np.zeros((2, 4)))
        self.assertIsNone(result["at_fault_collision_time_s"])
        self.assertEqual(result["contacts_through_first_at_fault"], [])


if __name__ == "__main__":
    unittest.main()
