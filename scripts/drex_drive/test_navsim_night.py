from pathlib import Path
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
