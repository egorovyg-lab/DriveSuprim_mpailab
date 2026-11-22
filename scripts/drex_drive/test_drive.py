import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, get_ident
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

    def test_realtime_keeps_moving_and_discards_expired_reply(self):
        release_slow_reply = Event()
        trace = []
        requests = []

        def choose(row):
            requests.append(row)
            if len(requests) == 2:
                if not release_slow_reply.wait(timeout=3):
                    raise RuntimeError("test did not release the delayed reply")
                return Action.PROCEED, None
            return (Action.PROCEED if len(requests) == 1 else Action.STOP), None

        def record(event):
            trace.append(event)
            if event["event"] == "step" and event["tick"] == 13:
                release_slow_reply.set()

        report = run_episode(World(Scenario(100.0)), choose, record, 19, realtime=True)
        slow_reply = next(event for event in trace if event["event"] == "decision" and event["request_tick"] == 5)
        self.assertEqual(slow_reply["status"], "late")
        self.assertEqual(report["decision_deadline_misses"], 1)
        self.assertEqual(report["decision_late_replies"], 1)
        self.assertGreaterEqual(report["decision_slots_busy"], 1)
        self.assertGreater(report["fallback_ticks"], 0)
        moving_steps = [event for event in trace if event["event"] == "step" and event["tick"] in (6, 7)]
        self.assertGreater(moving_steps[1]["ego_front_m"], moving_steps[0]["ego_front_m"])
        after_late = next(event for event in trace if event["event"] == "step"
                          and event["tick"] == slow_reply["tick"] + 1)
        self.assertEqual(after_late["action"], "stop")
        self.assertEqual(after_late["source"], "fallback")

    def test_realtime_warmup_uses_the_inference_thread_and_discards_action(self):
        calls = []
        trace = []

        def choose(_):
            calls.append(get_ident())
            return (Action.PROCEED if len(calls) == 1 else Action.WAIT), None

        report = run_episode(World(Scenario(100.0)), choose, trace.append, 4, realtime=True, warmup=True)
        self.assertEqual(calls, [calls[0], calls[0]])
        self.assertIsNotNone(report["warmup_wall_s"])
        self.assertEqual(report["decision_count"], 1)
        self.assertEqual(report["decision_applied"], 1)
        self.assertEqual(report["decision_deadline_misses"], 0)
        self.assertEqual(trace[-1]["ego_front_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
