import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .driver import Action, DrexMax
from .navsim_straight import (ACTION_PROBES, EXTRA_INPUT_VARIANTS, FOLLOWING_VARIANTS, HOLDOUT_PROTOCOL, PROTOCOL,
                             SPEED_CAP, build_plan, digest,
                             fresh_report, holdout_records, main, on_astra, past_leader_speeds, predict, predict_suite,
                             prepare_action_probes, read, score, straight_observation, summarize_actions, summarize_inputs,
                             write)


class LeadBrakeGapTests(unittest.TestCase):
    field = "minimum_lead_gap_if_braking_m"

    def setUp(self):
        self.scene = dict(token="synthetic", speed_mps=4.0, leaders=[{"gap_m": 12.0, "speed_mps": 2.0}],
                          corridor_objects=[], ego_bounds={"front_m": 3.0, "rear_m": -1.0})

    def plan(self, *, enabled=True, actions=("maintain",) * 8):
        commands = iter(actions)
        return build_plan(self.scene, lambda _: (next(commands), None), observation_variant="corridor-preview",
                          **({"lead_brake_gap": True} if enabled else {}))

    def preview_at(self, plan, index):
        return plan["decisions"][index]["request"]["state"]["action_effects"]["forecast"]

    def test_lead_brake_gap_adds_only_one_field_to_each_action_at_all_decisions(self):
        self.scene["corridor_objects"] = [
            dict(type="vehicle", x_m=17.0, y_m=0.2, vx_mps=2.0, vy_mps=0.0, heading_rad=0.0,
                 length_m=4.0, width_m=2.0, half_length_along_path_m=2.0, lateral_reach_m=2.0),
        ]
        actions = ("proceed", "maintain", "slow", "stop", "wait", "proceed", "slow", "maintain")
        before = self.plan(enabled=False, actions=actions)
        after = self.plan(actions=actions)
        self.assertEqual(before["poses"], after["poses"])
        self.assertEqual(len(before["decisions"]), 8)
        self.assertEqual(len(after["decisions"]), 8)
        for index, (original, augmented) in enumerate(zip(before["decisions"], after["decisions"])):
            with self.subTest(step=index):
                self.assertEqual(original["action"], actions[index])
                self.assertEqual(augmented["action"], actions[index])
                for effect in self.preview_at(after, index).values():
                    effect.pop(self.field)
                self.assertEqual(json.dumps(original["request"], allow_nan=False),
                                 json.dumps(augmented["request"], allow_nan=False))

    def test_late_preview_uses_scene_time_after_the_leader_has_stopped(self):
        plan = self.plan()
        self.assertEqual(plan["decisions"][7]["time_s"], 3.5)
        effect = self.preview_at(plan, 7)["maintain"]
        self.assertEqual(effect["minimum_gap_m"], 1.0)
        # The lead rear stopped at x=13, and ego reaches absolute x=22 at 5.5 s.
        self.assertEqual(effect[self.field], -9.0)

    def test_preview_uses_the_current_radar_leader_and_includes_tau_zero(self):
        self.scene["leaders"] = [{"gap_m": 8.0, "speed_mps": 5.0}, {"gap_m": 12.0, "speed_mps": 1.0}]
        plan = self.plan()
        first = self.preview_at(plan, 0)
        self.assertEqual(first["maintain"][self.field], 6.0)
        self.assertEqual(first["stop"][self.field], 8.0)  # The minimum is at tau=0.
        state = plan["decisions"][3]["request"]["state"]
        self.assertEqual(state["lead_radar"]["range_m"], 7.5)  # Radar switches to the second lead.
        self.assertEqual(state["lead_radar"]["closing_speed_mps"], 3.0)
        # The second lead stopped at x=12.25, and ego reaches x=14 at 3.5 s.
        self.assertEqual(self.preview_at(plan, 3)["maintain"][self.field], -1.75)

    def test_stationary_leader_has_signed_gaps_and_missing_leader_has_nulls(self):
        self.scene["leaders"] = [{"gap_m": 1.234, "speed_mps": 0.0}]
        stationary = self.preview_at(self.plan(), 0)
        self.assertEqual({action: effect[self.field] for action, effect in stationary.items()},
                         {"proceed": -8.52, "maintain": -6.77, "slow": -5.45, "stop": -4.14, "wait": -4.14})
        self.scene["leaders"] = []
        empty = self.plan()
        for index in range(8):
            self.assertIsNone(empty["decisions"][index]["request"]["state"]["lead_radar"])
            for effect in self.preview_at(empty, index).values():
                self.assertIsNone(effect[self.field])

    def test_lead_brake_gap_rejects_each_nonstandard_variant_before_choosing(self):
        choose = Mock()
        for prompt, observation, action in (("progress", "corridor-preview", "standard"),
                                            ("baseline", "radar", "standard"),
                                            ("baseline", "corridor-preview", "gentle")):
            with self.subTest(prompt=prompt, observation=observation, action=action):
                with self.assertRaisesRegex(ValueError, "lead_brake_gap requires baseline/corridor-preview/standard"):
                    build_plan(self.scene, choose, prompt, observation, action, lead_brake_gap=True)
        choose.assert_not_called()


class StraightPlanTests(unittest.TestCase):
    def test_following_inputs_add_only_their_fields_and_handle_stops(self):
        scene = dict(token="example", speed_mps=4.0, leaders=[{"gap_m": 12.0, "speed_mps": 2.0}])
        choose = lambda _: ("maintain", None)
        standard = build_plan(scene, choose, observation_variant="action-effects")
        current = build_plan(scene, choose, observation_variant="leader-context")
        preview = build_plan(scene, choose, observation_variant="action-gap-end")
        for prediction in (current, preview):
            self.assertEqual(prediction["poses"], standard["poses"])
            self.assertEqual(prediction["decisions"][0]["request"]["question"],
                             standard["decisions"][0]["request"]["question"])
        base = standard["decisions"][0]["request"]["state"]
        current_state = current["decisions"][0]["request"]["state"]
        self.assertEqual(current_state, {**base, "lead_radar": {
            **base["lead_radar"], "leader_speed_mps": 2.0, "headway_s": 3.0}})
        end_state = preview["decisions"][0]["request"]["state"]
        forecast = end_state["action_effects"]["forecast"]
        for action, existing in base["action_effects"]["forecast"].items():
            self.assertEqual({key: forecast[action][key] for key in existing}, existing)
        self.assertEqual(forecast["proceed"]["gap_at_horizon_m"], 6.25)
        self.assertEqual(forecast["proceed"]["closing_speed_at_horizon_mps"], 3.0)
        stopped = build_plan({**scene, "speed_mps": 0.0}, choose, observation_variant="leader-context")
        self.assertIsNone(stopped["decisions"][0]["request"]["state"]["lead_radar"]["headway_s"])
        empty = build_plan({**scene, "leaders": []}, choose, observation_variant="action-gap-end")
        no_lead = empty["decisions"][0]["request"]["state"]["action_effects"]["forecast"]["proceed"]
        self.assertIsNone(no_lead["closing_speed_at_horizon_mps"])
        self.assertIsNone(no_lead["gap_at_horizon_m"])

    def test_following_suite_uses_one_model_and_reuses_completed_predictions(self):
        scene = dict(token="example", speed_mps=4.0, leaders=[])
        plan = dict(protocol=HOLDOUT_PROTOCOL, speed_cap_mps=SPEED_CAP, scenes=[scene])
        choose = Mock(return_value=("maintain", None))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for variant in FOLLOWING_VARIANTS:
                (root / variant).mkdir()
                write(root / variant / "observations.json", plan)
            with patch("scripts.drex_drive.navsim_straight.load_drex", return_value=(choose, 0.0)) as load:
                self.assertFalse(predict_suite(root, root, FOLLOWING_VARIANTS))
                self.assertFalse(predict_suite(root, root, FOLLOWING_VARIANTS))
                load.assert_called_once_with(root)
            self.assertEqual(choose.call_count, 16)
            for variant in FOLLOWING_VARIANTS:
                result = read(root / variant / "predictions.json")
                self.assertEqual((result["observation_variant"], result["action_variant"]), (variant, "standard"))

    def test_action_probes_replace_only_the_specified_command(self):
        scenes = [dict(token=token, speed_mps=2.0, leaders=[]) for token in ("control", *ACTION_PROBES)]
        plan = dict(protocol=HOLDOUT_PROTOCOL, speed_cap_mps=SPEED_CAP, scenes=scenes)
        binding = dict(observation_sha256=digest(plan), prompt_variant="baseline", observation_variant="action-effects")
        originals = []
        for scene in scenes:
            actions = ["maintain"] * 8
            if scene["token"] in ACTION_PROBES:
                index, before, _ = ACTION_PROBES[scene["token"]]
                actions[index] = before
            commands = iter(actions)
            originals.append(build_plan(scene, lambda _: (next(commands), None), observation_variant="action-effects"))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report.json"
            write(root / "observations.json", plan)
            write(report, dict(**binding, protocol=HOLDOUT_PROTOCOL, compared_scenes=4, prediction_failures=[]))
            write(root / "predictions.json", dict(**binding, cases=originals))
            payload = prepare_action_probes(report)
            self.assertIsNone(payload["cases"][0]["probe"])
            for case in payload["cases"][1:]:
                index, before, after = ACTION_PROBES[case["token"]]
                probe = case["probe"]
                self.assertEqual((probe["decision_index"], probe["time_s"], probe["before"], probe["after"]),
                                 (index, index * 0.5, before, after))
                self.assertEqual(probe["poses"][:index * 5], case["poses"][:index * 5])
                self.assertGreater(probe["poses"][-1][0], case["poses"][-1][0])
            originals[1]["poses"][-1][0] += 1.0
            write(root / "predictions.json", dict(**binding, cases=originals))
            with self.assertRaisesRegex(RuntimeError, "Archived command replay differs"):
                prepare_action_probes(report)

    def test_fresh_score_exits_successfully_after_writing_a_complete_report(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {"anchor_identity_sha256": "anchor", "scenes": [{"token": "fresh"}]}
            pilot = {"protocol": PROTOCOL, "anchor_identity_sha256": "anchor", "scenes": [{"token": "train"}]}
            write(root / "observations.json", plan)
            write(root / "pilot.json", pilot)
            write(root / "predictions.json", {"cases": [{"token": "fresh"}], "action_variant": "gentle"})
            command = ["navsim_straight", "bridge-fresh-score", "--out", str(root),
                       "--pilot-observations", str(root / "pilot.json"), "--action-variant", "gentle"]
            target = "scripts.drex_drive.navsim_straight.on_astra"
            with patch.object(sys, "argv", command), patch(target) as score_stage:
                score_stage.return_value = {"compared_scenes": 1, "prediction_failures": []}
                self.assertEqual(main(), 0)
                self.assertEqual(read(root / "report.json"), score_stage.return_value)
                score_stage.return_value = {"compared_scenes": 1, "prediction_failures": [{"token": "other"}]}
                self.assertEqual(main(), 1)

    def test_fresh_score_passes_an_explicit_cpu_device_to_astra(self):
        path = "scripts.drex_drive.navsim_straight.subprocess.run"
        with patch.dict("os.environ", {"DREX_ASTRA_DS_DEVICE": "cpu", "DREX_ASTRA_DS_GPU": "2"}):
            with patch(path, return_value=SimpleNamespace(stdout="{}", returncode=0)) as run:
                self.assertEqual(on_astra("fresh-score", SimpleNamespace(limit=8)), {})
        remote = run.call_args.args[0][-1]
        self.assertIn("--ds-device cpu", remote)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=2", remote)

    def test_holdout_rejects_a_session_seen_in_the_original_audit(self):
        held = dict(token="a", session_id="hold", timestamp_us=10, role="holdout")
        train = dict(token="b", session_id="train", timestamp_us=20, role="train")
        selection = dict(records=[train, held], legacy_training_records=[], fresh_holdout_sessions=["hold"])
        self.assertEqual(holdout_records(selection, [train]), [held])
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            holdout_records(selection, [dict(token="different-frame", session_id="hold")])

    def test_holdout_prediction_accepts_only_the_frozen_input(self):
        scene = dict(token="example", speed_mps=4.0, leaders=[])
        plan = dict(protocol=HOLDOUT_PROTOCOL, speed_cap_mps=SPEED_CAP, scenes=[scene])
        with TemporaryDirectory() as root:
            output = Path(root) / "predictions.json"
            with self.assertRaisesRegex(ValueError, "fixed"):
                predict(plan, Path(root), output, "progress", "action-effects")
            with self.assertRaisesRegex(ValueError, "fixed"):
                predict(plan, Path(root), output, "baseline", "gap-forecast")
            choose = Mock(return_value=(Action.MAINTAIN, None))
            self.assertFalse(predict(plan, Path(root), output, "baseline", "action-effects", model=(choose, 0.0)))
            self.assertEqual(read(output)["observation_sha256"], digest(plan))
            self.assertEqual(choose.call_count, 8)

    def test_fresh_report_weights_eligible_sessions_equally_and_counts_safety_losses(self):
        scenes = [dict(token=token, session_id=session) for token, session in zip("abc", ("x", "x", "y"))]
        plan = dict(reserved_sessions=["x", "y", "z"], reserved_scenes=10, scenes=scenes,
                    rejected={"intersection": 7}, teacher_checkpoint_sha256="frozen")
        rows = []
        for scene, pdm, collision in zip(scenes, (.9, .9, .4), (1.0, 1.0, 0.0)):
            rows.append(dict(**scene, ds=dict(pdm_score=.8), drex=dict(pdm_score=pdm),
                             delta=dict(pdm_score=pdm - .8, no_at_fault_collisions=collision - 1.0)))
        report = fresh_report(plan, ("pdm_score", "no_at_fault_collisions"), ("no_at_fault_collisions",), rows, [], "pilot")
        self.assertAlmostEqual(report["mean_delta"]["pdm_score"], -1 / 15)
        self.assertAlmostEqual(report["mean_delta_equal_session"]["pdm_score"], -.15)
        self.assertEqual((report["pdm_wins"], report["pdm_losses"]), (2, 1))
        self.assertEqual(report["safety_regressions"], {"no_at_fault_collisions": 1})
        self.assertIsNone(report["sessions"][2]["mean_delta"])

    def test_straight_filter_uses_current_tracks_and_static_road(self):
        layers = SimpleNamespace(INTERSECTION="intersection", ROADBLOCK="roadblock", DRIVABLE_AREA="drivable",
                                 CARPARK_AREA="carpark")
        origin = SimpleNamespace(x=0, y=0, heading=0)
        leader = SimpleNamespace(center=SimpleNamespace(x=10.0, y=0, heading=0),
                                 velocity=SimpleNamespace(x=2.0, y=0), box=SimpleNamespace(length=4, width=2))
        route = Mock(length=60.0)
        route.project.return_value = 0.0
        route.interpolate.side_effect = lambda ds, as_array: np.column_stack((ds, np.zeros((len(ds), 2))))
        road = Mock()
        road.get_indices_of_map_type.side_effect = lambda kinds: [0] if kinds == [layers.INTERSECTION] else [1]
        road.points_in_polygons.return_value = np.stack((np.zeros((41, 2, 2)), np.ones((41, 2, 2)))).astype(bool)
        ego = SimpleNamespace(rear_axle=origin, dynamic_car_state=SimpleNamespace(speed=4.0), car_footprint=
                              SimpleNamespace(vehicle_parameters=SimpleNamespace(rear_axle_to_center=1, length=4, width=2)))
        tracks = Mock()
        tracks.get_tracked_objects_of_type.return_value = [leader]
        cache = SimpleNamespace(ego_state=ego, centerline=route, drivable_area_map=road,
                                current_tracked_objects=[SimpleNamespace(tracked_objects=tracks)])
        imports = {
            "nuplan.common.actor_state.tracked_objects_types": SimpleNamespace(TrackedObjectType=SimpleNamespace(VEHICLE=1)),
            "nuplan.common.maps.maps_datatypes": SimpleNamespace(SemanticMapLayer=layers),
            "shapely.geometry": SimpleNamespace(Point=lambda x, y: (x, y)),
        }
        with patch.dict(sys.modules, imports):
            info, error = straight_observation(cache)
            self.assertIsNone(error)
            self.assertEqual(info, {"speed_mps": 4.0, "leaders": [{"gap_m": 5.0, "speed_mps": 2.0}]})
            self.assertEqual(road.points_in_polygons.call_args.args[0].shape, (41, 2, 2, 2))
            road.points_in_polygons.return_value[0, 10, 0, 0] = True
            self.assertEqual(straight_observation(cache), (None, "intersection"))

    def test_rollout_builds_40_local_poses_using_only_current_and_forecast_observations(self):
        record = {"token": "example", "speed_mps": 4.0, "leaders": [{"gap_m": 12.0, "speed_mps": 2.0}]}
        choose = Mock(return_value=(Action.MAINTAIN, None))
        prediction = build_plan(record, choose)
        self.assertEqual(len(prediction["poses"]), 40)
        self.assertEqual(len(prediction["decisions"]), 8)
        self.assertAlmostEqual(prediction["poses"][0][0], 0.4)
        self.assertAlmostEqual(prediction["poses"][-1][0], 16.0)
        self.assertTrue(all(pose[1:] == [0.0, 0.0] for pose in prediction["poses"]))
        first, second = [entry.args[0] for entry in choose.call_args_list[:2]]
        self.assertEqual(first["state"]["lead_radar"], {"range_m": 12.0, "closing_speed_mps": 2.0})
        self.assertEqual(second["state"]["lead_radar"], {"range_m": 11.0, "closing_speed_mps": 2.0})
        self.assertNotIn("token", first["state"])

    def test_braking_changes_the_continuous_path(self):
        record = {"token": "example", "speed_mps": 4.0, "leaders": []}
        stop = build_plan(record, lambda _: (Action.STOP, None))
        self.assertAlmostEqual(stop["poses"][-1][0], 4.0**2 / (2 * 3.0))
        self.assertEqual(stop["poses"][-1], stop["poses"][-2])
        self.assertIsNone(stop["decisions"][0]["request"]["state"]["lead_radar"])

    def test_progress_variant_changes_only_the_instructions_for_equal_actions(self):
        record = {"token": "example", "speed_mps": 4.0, "leaders": [{"gap_m": 12.0, "speed_mps": 2.0}]}
        choose = Mock(return_value=(Action.MAINTAIN, None))
        baseline = build_plan(record, choose)
        progress = build_plan(record, choose, "progress")
        self.assertEqual(baseline["poses"], progress["poses"])
        for before, after in zip(baseline["decisions"], progress["decisions"]):
            self.assertEqual(before["request"]["state"], after["request"]["state"])
            old_question = before["request"]["question"]
            new_question = after["request"]["question"]
            amended = new_question["instructions"].removeprefix(old_question["instructions"])
            self.assertNotEqual(new_question["instructions"], amended)
            self.assertIn("Prefer maintain", amended)
            self.assertEqual({**new_question, "instructions": old_question["instructions"]}, old_question)

    def test_gap_forecast_adds_only_two_fields_without_changing_the_prompt(self):
        record = {"token": "example", "speed_mps": 4.0, "leaders": [{"gap_m": 3.0, "speed_mps": 2.0}]}
        choose = Mock(return_value=(Action.MAINTAIN, None))
        before = build_plan(record, choose)
        after = build_plan(record, choose, observation_variant="gap-forecast")
        self.assertEqual(before["poses"], after["poses"])
        for index, (original, augmented) in enumerate(zip(before["decisions"], after["decisions"])):
            old_row, new_row = original["request"], augmented["request"]
            self.assertEqual(new_row["question"], old_row["question"])
            self.assertEqual(new_row["state"], {**old_row["state"], "lead_radar": {
                **old_row["state"]["lead_radar"],
                "gap_if_maintain_1s_m": 1.0 - index,
                "gap_if_maintain_2s_m": -1.0 - index,
            }})
        empty_road = build_plan({**record, "leaders": []}, choose, observation_variant="gap-forecast")
        self.assertIsNone(empty_road["decisions"][0]["request"]["state"]["lead_radar"])

    def test_history_matches_the_leader_and_preserves_measurement_age(self):
        def track(token, time_us, speed):
            return SimpleNamespace(track_token=token, metadata=SimpleNamespace(timestamp_us=time_us),
                                   velocity=SimpleNamespace(x=speed, y=0))

        past = [SimpleNamespace(tracked_objects=Mock()) for _ in range(2)]
        past[0].tracked_objects.get_tracked_objects_of_type.return_value = [track("lead", 9000000, 3)]
        past[1].tracked_objects.get_tracked_objects_of_type.return_value = [
            track("other", 9500000, 9), track("lead", 9500000, 2),
        ]
        history = past_leader_speeds(SimpleNamespace(past_detections_tracks=past), track("lead", 10000000, 1),
                                     SimpleNamespace(heading=0), "vehicle")
        self.assertEqual(history, [{"age_at_start_s": 1.0, "speed_mps": 3.0},
                                  {"age_at_start_s": 0.5, "speed_mps": 2.0}])
        record = {"token": "example", "speed_mps": 4.0, "leaders": [{"gap_m": 12.0, "speed_mps": 1.0}]}
        choose = lambda _: (Action.MAINTAIN, None)
        original = build_plan(record, choose, observation_variant="gap-forecast")
        variant = build_plan({**record, "leader_history": [history]}, choose, observation_variant="leader-history")
        self.assertEqual(original["poses"], variant["poses"])
        for index, (before, after) in enumerate(zip(original["decisions"], variant["decisions"])):
            first, second = before["request"], after["request"]
            self.assertEqual(first["question"], second["question"])
            samples = second["state"]["lead_radar"].pop("measured_lead_speed_history")
            self.assertEqual(samples[1], {"age_s": 0.5 + index * 0.5, "speed_mps": 2.0})
            self.assertEqual(first["state"], second["state"])

    def test_action_effects_are_extra_predictions_with_the_same_dynamics(self):
        record = {"token": "example", "speed_mps": 4.0, "leaders": [{"gap_m": 3.0, "speed_mps": 2.0}]}
        choose = lambda _: (Action.MAINTAIN, None)
        baseline = build_plan(record, choose, observation_variant="gap-forecast")
        variant = build_plan(record, choose, observation_variant="action-effects")
        self.assertEqual(variant["poses"], baseline["poses"])
        for first, second in zip(baseline["decisions"], variant["decisions"]):
            self.assertEqual(first["request"]["question"], second["request"]["question"])
            effects = second["request"]["state"].pop("action_effects")
            self.assertEqual(first["request"]["state"], second["request"]["state"])
            self.assertEqual(effects["horizon_s"], 2.0)
            self.assertEqual(effects["apply_action_s"], 0.5)
            self.assertEqual(effects["forecast"]["maintain"]["progress_m"], 8.0)
            self.assertGreater(effects["forecast"]["slow"]["minimum_gap_m"], effects["forecast"]["maintain"]["minimum_gap_m"])

    def test_gentle_commands_are_scored_by_drex_and_use_the_preview_acceleration(self):
        driver = DrexMax.__new__(DrexMax)
        driver.model = Mock()
        driver.model.predict.side_effect = (
            [[[0, 0, 0, 0, 0, 1, 0]], [[0, 0, 0, 0, 0, 0, 1]]] + [[[0, 1, 0, 0, 0, 0, 0]]] * 6
        )
        scene = {"token": "example", "speed_mps": 4.0, "leaders": [{"gap_m": 12.0, "speed_mps": 2.0}]}
        standard = build_plan(scene, lambda _: (Action.MAINTAIN, None), observation_variant="action-effects")
        gentle = build_plan(scene, driver.choose, observation_variant="action-effects", action_variant="gentle")
        first, second, third = gentle["decisions"][:3]
        self.assertEqual([first["action"], second["action"]], ["gentle_proceed", "gentle_slow"])
        self.assertEqual(second["request"]["state"]["ego_speed_mps"], 4.5)
        self.assertEqual(second["request"]["state"]["previous_action"], "gentle_proceed")
        self.assertAlmostEqual(third["request"]["state"]["ego_speed_mps"], 4.125, delta=0.0051)
        self.assertAlmostEqual(gentle["poses"][4][0], 2.125)
        self.assertAlmostEqual(gentle["poses"][9][0], 4.28125)
        before = standard["decisions"][0]["request"]
        after = first["request"]
        self.assertEqual(before["question"]["instructions"], after["question"]["instructions"])
        self.assertEqual(list(before["question"]["criteria"]), list(after["question"]["criteria"])[:5])
        forecast = after["state"]["action_effects"]["forecast"]
        for name, effect in before["state"]["action_effects"]["forecast"].items():
            self.assertEqual(forecast[name], effect)
        self.assertEqual(forecast["gentle_proceed"]["progress_m"], 8.88)
        self.assertEqual(forecast["gentle_slow"]["progress_m"], 7.34)

    def test_action_summary_compares_to_archived_standard_and_rejects_changed_ds(self):
        ds = {"pdm_score": .9, "no_at_fault_collisions": 1.0}
        standard = {"pdm_score": .8, "no_at_fault_collisions": 1.0}
        gentle = {"pdm_score": .7, "no_at_fault_collisions": 0.0}
        case = dict(token="example", session_id="trip", ds=ds)
        report = dict(protocol=HOLDOUT_PROTOCOL, observation_variant="action-effects", prompt_variant="baseline",
                      observation_sha256="same", planned_scenes=1, compared_scenes=1, prediction_failures=[],
                      teacher_checkpoint_sha256="checkpoint", baseline_parity_token="pilot", ds_inference_device="cpu")
        predictions = dict(observation_sha256="same", model_path="drex", cases=[
            dict(token="example", decisions=[dict(action="gentle_proceed")])])
        gentle_report = dict(**report, action_variant="gentle",
                             mean_delta={"pdm_score": -.2, "no_at_fault_collisions": -1.0},
                             safety_regressions={"no_at_fault_collisions": 1})
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "standard").mkdir()
            reference = root / "standard/report.json"
            write(reference, {**report, "cases": [{**case, "drex": standard}]})
            write(reference.with_name("predictions.json"), predictions)
            write(root / "predictions.json", {**predictions, "action_variant": "gentle"})
            write(root / "report.json", {**gentle_report, "cases": [
                {**case, "drex": gentle, "delta": gentle_report["mean_delta"]}]})
            result = summarize_actions(root, reference)
            self.assertAlmostEqual(result["mean_delta_vs_standard"]["pdm_score"], -.1)
            self.assertEqual(result["gentle_action_counts"], {"gentle_proceed": 1, "gentle_slow": 0})
            self.assertEqual(result["pdm_losses_vs_standard"], 1)
            self.assertEqual(result["safety_regressions_vs_standard"], {"no_at_fault_collisions": 1})
            changed = {**case, "ds": {**ds, "pdm_score": .93}}
            write(root / "report.json", {**gentle_report, "cases": [
                {**changed, "drex": gentle, "delta": gentle_report["mean_delta"]}]})
            with self.assertRaisesRegex(RuntimeError, "baseline or session changed"):
                summarize_actions(root, reference)

    def test_input_suite_loads_model_once_for_both_variants(self):
        record = {"token": "example", "speed_mps": 4.0, "leaders": [{"gap_m": 12.0, "speed_mps": 2.0}]}
        plan = {"protocol": PROTOCOL, "speed_cap_mps": SPEED_CAP, "scenes": [record]}
        history = {"observation_sha256": digest(plan), "scenes": [{"token": "example", "leader_history": [[]]}]}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for variant in EXTRA_INPUT_VARIANTS:
                (root / variant).mkdir()
                write(root / variant / "observations.json", plan)
            write(root / "leader-history" / "history.json", history)
            with patch("scripts.drex_drive.navsim_straight.load_drex", return_value=(lambda _: (Action.MAINTAIN, None), 0.0)) as load:
                self.assertFalse(predict_suite(root, root))
                self.assertFalse(predict_suite(root, root))
                load.assert_called_once_with(root)
            for variant in EXTRA_INPUT_VARIANTS:
                result = read(root / variant / "predictions.json")
                self.assertEqual(result["observation_variant"], variant)
                self.assertEqual(result["prompt_variant"], "baseline")
                self.assertEqual(len(result["cases"][0]["poses"]), 40)

    def test_input_suite_summary_uses_matched_gap_forecast_and_ds_scores(self):
        base = {"protocol": PROTOCOL, "prompt_variant": "baseline", "observation_sha256": "frozen",
                "planned_scenes": 1, "compared_scenes": 1, "prediction_failures": [],
                "mean_delta": {"pdm_score": -0.05, "no_at_fault_collisions": 0},
                "safety_regressions": {"no_at_fault_collisions": 0}}
        ds = {"pdm_score": 0.9, "no_at_fault_collisions": 1.0}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.json"
            write(reference, {**base, "observation_variant": "gap-forecast", "cases": [
                {"token": "example", "ds": ds, "drex": {**ds, "pdm_score": 0.85}},
            ]})
            for variant in EXTRA_INPUT_VARIANTS:
                (root / variant).mkdir()
                write(root / variant / "report.json", {**base, "observation_variant": variant, "cases": [
                    {"token": "example", "ds": ds, "drex": {**ds, "pdm_score": 0.88}},
                ]})
            write(root / "leader-history" / "history.json", {"scenes": [{"leader_history": [[{}]]}]})
            result = summarize_inputs(root, reference)
            self.assertEqual(result["history_available_scenes"], 1)
            for variant in result["variants"]:
                self.assertAlmostEqual(variant["mean_delta_vs_gap_forecast"]["pdm_score"], 0.03)
                self.assertEqual(variant["pdm_better_vs_gap_forecast"], 1)

    def test_paired_scoring_verifies_original_ds_anchor(self):
        record = {"token": "example", "session_id": "session", "cache_sha256": "cache", "speed_mps": 4.0, "leaders": []}
        plan = {"protocol": PROTOCOL, "anchor_identity_sha256": "original", "scenes": [record]}
        prediction = build_plan(record, lambda _: (Action.MAINTAIN, None))
        payload = {"plan": plan, "predictions": {"observation_sha256": digest(plan), "cases": [prediction]}}
        candidates = {"example": [{"candidate_local": 3, "candidate_vocab": 0, "expected_raw": [0.7, 1.0],
                                    "metric_keys": ["pdm_score", "no_at_fault_collisions"]}]}
        anchor_poses = [[0.0, 0.0, 0.0]] * 40
        sources = {
            "real_data": SimpleNamespace(METRICS=("no_at_fault_collisions",), SAFETY=("no_at_fault_collisions",),
                                         load_inputs=Mock(return_value={"baseline_local": 3})),
            "rollout_replay": SimpleNamespace(bound_context=Mock(return_value=(
                {}, {}, [anchor_poses], {}, {"example": {"sha256": "cache"}},
            ))),
        }
        baseline = {"pdm_score": 0.7, "no_at_fault_collisions": 1.0}
        drex = {"pdm_score": 0.5, "no_at_fault_collisions": 0.0}
        with patch.dict(sys.modules, sources), patch(
            "scripts.drex_drive.navsim_straight.source_context", return_value=([], candidates, "original"),
        ), patch("scripts.drex_drive.navsim_straight.load_cache"), patch(
            "scripts.drex_drive.navsim_straight.straight_observation", return_value=({"speed_mps": 4.0, "leaders": []}, None),
        ), patch("scripts.drex_drive.navsim_straight.raw_score", side_effect=[baseline, drex, drex]) as scorer:
            report = score(None, None, None, payload)
            self.assertEqual(report["compared_scenes"], 1)
            self.assertAlmostEqual(report["mean_delta"]["pdm_score"], -0.2)
            self.assertEqual(report["safety_regressions"]["no_at_fault_collisions"], 1)
            self.assertIs(scorer.call_args_list[0].args[1], anchor_poses)
            self.assertEqual(scorer.call_args_list[1].args[1].tolist(), prediction["poses"])
            with self.assertRaisesRegex(RuntimeError, "anchor parity failed"):
                score(None, None, None, payload)


if __name__ == "__main__":
    unittest.main()
