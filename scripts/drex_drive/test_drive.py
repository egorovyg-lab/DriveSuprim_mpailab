import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, get_ident
from unittest.mock import Mock, patch

from .__main__ import SUITES, main, run_episode
from .camera import FRAME_SIZE
from .driver import DrexMax, driving_request, reference
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

        report = run_episode(World(Scenario(100.0)), choose, record, 19, realtime=True, observation="camera-pair")
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
        self.assertEqual([len(row["images"]) for row in requests], [1, 2, 2])
        self.assertIs(requests[0]["images"][0], requests[1]["images"][0])
        self.assertIsNot(requests[1]["images"][1], requests[2]["images"][0])
        self.assertEqual(report["paired_requests"], 2)
        self.assertEqual(report["paired_decisions_applied"], 1)

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

    def test_injected_timeout_discards_reply_and_records_recovery(self):
        trace = []
        report = run_episode(World(Scenario(100.0)), lambda _: (Action.PROCEED, None), trace.append, 20,
                             realtime=True, injected_timeouts={5: 650})
        self.assertEqual(report["injected_timeouts"], [
            {"request_tick": 5, "sim_time_s": 0.5, "extra_delay_ms": 650, "status": "discarded"},
        ])
        self.assertEqual(report["decision_deadline_misses"], 1)
        self.assertEqual(report["fallback_activations"], 1)
        self.assertEqual(len(report["fallback_recoveries_wall_s"]), 1)
        self.assertGreater(report["fallback_recoveries_wall_s"][0], 0.5)
        self.assertFalse(report["latency_stress_pass"])
        recovered = next(event for event in trace if event["event"] == "fallback_recovered")
        step = next(event for event in trace if event["event"] == "step" and event["tick"] == recovered["tick"] + 1)
        self.assertEqual(step["source"], "driver")

    def test_camera_request_hides_structured_traffic_and_shows_signal_changes(self):
        world = World(SCENARIOS["route"])
        world.ego = Vehicle(37.0, 0.0)
        state = driving_request(world, Action.WAIT, None, "state")
        mixed = driving_request(world, Action.WAIT, None, "state-camera")
        vision = driving_request(world, Action.WAIT, None, "camera")
        self.assertNotIn("images", state)
        self.assertEqual(state["state"], mixed["state"])
        self.assertEqual(set(state["state"]) - set(vision["state"]), {"traffic_light", "lead_vehicle"})
        red = vision["images"][0]
        self.assertEqual(red.size, FRAME_SIZE)
        world.tick = 170
        green = driving_request(world, Action.WAIT, None, "camera")["images"][0]
        self.assertNotEqual(red.tobytes(), green.tobytes())
        self.assertIn((255, 37, 33), [color for _, color in red.getcolors()])
        self.assertIn((35, 246, 92), [color for _, color in green.getcolors()])

    def test_suite_loads_model_once_and_records_all_episodes_and_frames(self):
        for suite_name, (scenarios, conditions) in SUITES.items():
            with self.subTest(suite=suite_name), TemporaryDirectory() as tmp:
                output = Path(tmp) / "suite"
                args = ["drex_drive", "--suite", suite_name, "--max-seconds", "1", "--model", tmp, "--out", str(output)]
                with patch("sys.argv", args), patch("builtins.print"):
                    with patch("scripts.drex_drive.__main__.DrexMax", return_value=lambda _: (Action.WAIT, None)) as model:
                        self.assertEqual(main(), 1)
                        model.assert_called_once()
                suite = json.loads((output / "suite.json").read_text(encoding="utf-8"))
                self.assertEqual([(case["scenario"], case["observation"]) for case in suite["cases"]],
                                 [(scenario, condition) for scenario in scenarios for condition in conditions])
                for case in suite["cases"]:
                    case_path = (output / case["report"]).parent
                    trace = (case_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
                    requests = [event for event in map(json.loads, trace) if "request" in event]
                    first_frame = requests[0]["request"].get("images", [])
                    self.assertEqual(len(first_frame), int(case["observation"] != "state"))
                    for image_path in first_frame:
                        self.assertTrue((case_path / image_path).is_file())
                    if case["observation"] == "camera-pair":
                        older_frame, current_frame = requests[1]["request"]["images"]
                        self.assertEqual(requests[1]["frame_ticks"], [0, 5])
                        self.assertEqual(older_frame, first_frame[0])
                        self.assertTrue((case_path / current_frame).is_file())

    def test_drex_camera_request_checks_native_pixel_inputs(self):
        driver = DrexMax.__new__(DrexMax)
        driver.verified_frame_counts = set()
        driver.model = Mock()
        prepared = driver.model.prepare.return_value
        prepared.inputs = {}
        row = driving_request(World(SCENARIOS["route"]), Action.WAIT, None, "camera")
        with self.assertRaisesRegex(RuntimeError, "camera pixel inputs"):
            driver(row)
        driver.model.predict.assert_not_called()
        prepared.inputs = {"pixel_values": object(), "image_grid_thw": [[1, 32, 20]]}
        driver.model.predict.return_value = [[1.0, 0.0, 0.0, 0.0, 0.0]]
        self.assertEqual(driver(row)[0], Action.PROCEED)
        self.assertEqual(driver(row)[0], Action.PROCEED)
        self.assertEqual(driver.model.prepare.call_count, 2)
        pair = {**row, "images": row["images"] * 2}
        with self.assertRaisesRegex(RuntimeError, "2 frames"):
            driver(pair)
        prepared.inputs["image_grid_thw"] *= 2
        self.assertEqual(driver(pair)[0], Action.PROCEED)


if __name__ == "__main__":
    unittest.main()
