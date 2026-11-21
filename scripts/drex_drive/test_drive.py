import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from .__main__ import main, run_episode
from .driver import DrexMax, reference
from .simulation import Action, SCENARIOS, Scenario, Vehicle, World


class DriveTests(unittest.TestCase):
    def test_reference_completes_all_scenarios(self):
        for name, scenario in SCENARIOS.items():
            with self.subTest(name=name):
                trace = []
                report = run_episode(World(scenario), reference, trace.append, 600)
                self.assertTrue(report["clean_finish"], report)
                self.assertEqual(report["driver_control_fraction"], 1.0)
                self.assertEqual(trace[-1]["ego_speed_mps"], 0.0)
                if scenario.stop_line_m is not None:
                    states = [event["request"]["state"] for event in trace if event["event"] == "decision"]
                    approaching = [state for state in states if state["traffic_light"] is not None]
                    self.assertTrue(any(state["traffic_light"]["state"] == "green" for state in approaching))
                    self.assertTrue(any(
                        state["traffic_light"]["state"] == "red"
                        and state["ego_speed_mps"] == 0
                        and state["traffic_light"]["stop_line_distance_m"] < 10
                        for state in approaching
                    ))

    def test_drex_actions_change_next_observation(self):
        driver = DrexMax.__new__(DrexMax)
        driver.model = Mock()
        driver.model.predict.side_effect = [[[0.9, 0.0, 0.0, 0.1, 0.0]], [[0.0, 0.0, 0.0, 1.0, 0.0]]]
        trace = []
        report = run_episode(World(Scenario(60.0)), driver, trace.append, 10)
        requests = [call.args[0][0] for call in driver.model.predict.call_args_list]
        self.assertEqual(requests[0]["state"]["ego_speed_mps"], 0.0)
        self.assertEqual(requests[1]["state"]["ego_speed_mps"], 1.0)
        self.assertEqual(requests[1]["state"]["previous_action"], "proceed")
        self.assertEqual(trace[0]["probabilities"]["proceed"], 0.9)
        self.assertEqual(trace[-1]["ego_speed_mps"], 0.0)
        self.assertEqual(report["fallback_ticks"], 0)

    def test_invalid_drex_output_brakes_and_is_counted(self):
        driver = DrexMax.__new__(DrexMax)
        driver.model = Mock()
        driver.model.predict.side_effect = [[[1.0, 0.0, 0.0, 0.0, 0.0]], [[0.9, 0.0, 0.0, 0.0, 0.0]]]
        trace = []
        report = run_episode(World(Scenario(60.0)), driver, trace.append, 10)
        self.assertEqual(report["decision_errors"], 1)
        self.assertEqual(report["fallback_ticks"], 5)
        self.assertEqual(trace[-1]["source"], "fallback")
        self.assertEqual(trace[-1]["ego_speed_mps"], 0.0)

    def test_bad_driver_crosses_red_and_misses_destination(self):
        report = run_episode(World(SCENARIOS["red_light"]), lambda _: (Action.PROCEED, None), lambda _: None, 600)
        self.assertEqual(report["red_crossings"], 1)
        self.assertEqual(report["outcome"], "missed_destination")
        self.assertFalse(report["clean_finish"])

    def test_emergency_brake_is_counted_as_an_override(self):
        world = World(Scenario(100.0, lead_start_m=7.5))
        world.ego = Vehicle(0.0, 6.0)
        world.lead = Vehicle(7.5, 0.0)
        trace = []
        report = run_episode(world, lambda _: (Action.PROCEED, None), trace.append, 1)
        self.assertEqual(trace[-1]["source"], "emergency_brake")
        self.assertEqual(trace[-1]["ego_speed_mps"], 5.3)
        self.assertEqual(report["emergency_brake_ticks"], 1)
        self.assertEqual(report["driver_control_fraction"], 0.0)

    def test_cli_uses_drex_by_default_and_refuses_existing_output(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            args = ["drex_drive", "--model", tmp, "--out", str(output)]
            with patch("sys.argv", args), patch("builtins.print"), patch("sys.stderr"):
                with patch("scripts.drex_drive.__main__.DrexMax", return_value=reference) as load_drex:
                    self.assertEqual(main(), 0)
                    load_drex.assert_called_once_with(Path(tmp).resolve())
                    original = (output / "report.json").read_bytes()
                    self.assertIn(b'"driver": "drex"', original)
                    with self.assertRaises(SystemExit):
                        main()
                    self.assertEqual((output / "report.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
