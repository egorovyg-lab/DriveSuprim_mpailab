import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from .driver import Action
from .navsim_straight import PROTOCOL, build_plan, digest, score, straight_observation


class StraightPlanTests(unittest.TestCase):
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
