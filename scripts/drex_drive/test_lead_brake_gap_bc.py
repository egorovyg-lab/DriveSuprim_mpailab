from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import driver, lead_brake_gap_bc as bc


def scene(token=bc.TARGET, leader_speed=5.0, drive="synthetic"):
    return {"token": token, "session_id": drive, "speed_mps": 5.0,
            "leaders": [] if leader_speed is None else [{"speed_mps": leader_speed, "gap_m": 20.0}],
            "corridor_objects": [], "ego_bounds": {"front_m": 3, "rear_m": -1}}


class Tokens:
    def prepare(self, requests, max_length):
        ids = list(bc.night.request_json(requests[0]).encode())[:max_length]
        return SimpleNamespace(inputs={"input_ids": Mock(tolist=lambda: [ids]),
                                       "attention_mask": Mock(tolist=lambda: [[1] * len(ids)])})


class Model:
    model = Tokens()

    def choose(self, request):
        forecast = request["state"]["action_effects"]["forecast"]
        return ("maintain" if bc.FIELD in forecast["maintain"] else "proceed"), None


def raw(ep=0.5, pdm=0.8, nc=1.0):
    return {**dict.fromkeys(bc.night.BRAKE_GAP_PROTECTED, 1.0), "no_at_fault_collisions": nc,
            "ego_progress": ep, "pdm_score": pdm}


class BrakeGapBCTests(unittest.TestCase):
    def test_gate_and_unchanged_always_on_control_at_every_step(self):
        for lead, expected in ((4, "closing"), (5, "zero"), (6, "opening"), (None, "no_leader"),
                               (4.999, "closing")):
            with self.subTest(reason=expected):
                rows = []

                def audit(b, reason):
                    c, info = bc.counterfactual(b, reason, Tokens())
                    rows.append((b, c, info))
                    return c

                kwargs = {"observation_variant": "corridor-preview", "lead_brake_gap": True}
                maintain = lambda request: ("maintain", None)
                old = bc.straight.build_plan(scene(leader_speed=lead), maintain, **kwargs)
                new = bc.straight.build_plan(scene(leader_speed=lead), maintain, brake_gap_request=audit, **kwargs)
                self.assertEqual(len(new["poses"]), 40)
                self.assertEqual(len(rows), 8)
                self.assertEqual(old["poses"], new["poses"])
                for i, (b, c, info) in enumerate(rows):
                    self.assertEqual(info["reason"], expected)
                    self.assertEqual(bc.night.request_json(b), bc.night.request_json(old["decisions"][i]["request"]))
                    self.assertEqual(b, c) if expected == "closing" else self.assertEqual(bc.night.without_brake_gap(b), c)
                    fields = b["state"]["action_effects"]["forecast"]
                    self.assertEqual(all(row[bc.FIELD] is None for row in fields.values()), lead is None)
                    self.assertEqual(info["b_tokens"] == info["c_tokens"], expected == "closing")

    def test_invalid_leader_is_not_a_valid_abstention(self):
        for present, speed in ((True, None), (False, 0), (True, float("nan")), (True, float("inf")), (True, True)):
            with self.assertRaisesRegex(ValueError, "Invalid"):
                bc.straight.lead_brake_gap_gate(present, speed)
        for corrupt in (None, [{}], [{"gap_m": 10, "speed_mps": -1}], [{"gap_m": float("inf"), "speed_mps": 0}]):
            choose = Mock()
            with self.assertRaisesRegex(ValueError, "Invalid initial radar tracks"):
                bc.straight.build_plan({**scene(), "leaders": corrupt}, choose, observation_variant="corridor-preview",
                                       lead_brake_gap=True, brake_gap_request=Mock())
            choose.assert_not_called()

    def test_truncation_erased_field_and_order_all_fail(self):
        case = bc.straight.build_plan(scene(), lambda _: ("maintain", None),
                                      observation_variant="corridor-preview", lead_brake_gap=True)
        b = case["decisions"][0]["request"]
        erased = Tokens()
        erased.prepare = lambda requests, max_length: Tokens().prepare([bc.night.without_brake_gap(requests[0])
            if bc.FIELD in requests[0]["state"]["action_effects"]["forecast"]["maintain"] else requests[0]], max_length)
        with self.assertRaisesRegex(RuntimeError, "erased"):
            bc.counterfactual(b, "zero", erased)
        with self.assertRaisesRegex(RuntimeError, "truncation"):
            bc.counterfactual({**b, "oversize": "z" * 5000}, "zero", Tokens())
        forecast = b["state"]["action_effects"]["forecast"]
        row = forecast["maintain"]
        forecast["maintain"] = {bc.FIELD: row[bc.FIELD], **row}
        with self.assertRaisesRegex(RuntimeError, "ordering"):
            bc.counterfactual(b, "zero", Tokens())

    def test_bundled_astra_entrypoint_uses_stdin_and_json_stdout(self):
        with patch.object(bc.straight, "on_astra") as transport:
            bc.night.on_astra("astra-score", SimpleNamespace(), {}, extra_source=Path(bc.__file__))
        code = transport.call_args.kwargs["source_code"]
        result = StringIO()
        with patch.object(sys, "argv", ["-c", "astra-score"]), patch.object(sys, "stdin", StringIO('{"stage":"score"}')), \
                patch.dict(sys.modules), redirect_stdout(result), self.assertRaises(SystemExit) as exit_info:
            exec(code, {"__name__": "__main__", "__package__": None})
        self.assertEqual(exit_info.exception.code, 0)
        self.assertEqual(json.loads(result.getvalue())["status"], "incomplete")

    def test_full_transaction_and_failed_gates(self):
        tokens = [bc.TARGET, bc.OTHER, *(f"{i:016x}" for i in range(21))]
        plan = {"scenes": [scene(t, drive=f"fake-{i % 14}") for i, t in enumerate(tokens)]}
        frozen = {"reference": {t: {"ds": raw()} for t in tokens}, "plan": plan, "export": {}}
        binding, identity, scorer = {"observation": "frozen"}, {"weights": "pinned"}, {"scorer": "pinned"}
        runtime = {"identity": identity, "slurm_job_id": bc.PRIOR_JOB, "seed_per_arm": bc.SEED}
        lane = {"mode": "pass"}

        def for_scene(record):
            def evaluate(pred):
                is_c = pred["poses"][-1][0] > 20.05
                if lane["mode"] == "baseline" and not is_c:
                    return raw(pdm=0.79)
                if is_c and lane["mode"] == "reject":
                    return raw(ep=0.50, pdm=0.77, nc=0.0 if record["token"] == tokens[1] else 1.0)
                return raw(ep=0.60 if is_c else 0.50)
            return raw(), evaluate

        def fake_bridge(stage, args, payload, **kwargs):
            self.assertEqual(stage, "astra-score")
            self.assertEqual(set(payload["pairs"]["b"][bc.TARGET]), {"token", "poses", "gate"})
            return bc.scoring_report(args, payload)

        with TemporaryDirectory() as tmp, ExitStack() as stack, redirect_stdout(StringIO()):
            runs = Path(tmp).resolve() / "runs"
            archive = runs / bc.night.PROTOCOL
            archive.mkdir(parents=True)
            prior = runs / bc.night.BRAKE_GAP_PAIRED_EXPERIMENT / "attempt-002"
            (prior / "paired-pilot").mkdir(parents=True)
            manifest = {"attempt": str(prior), "experiment_id": bc.night.BRAKE_GAP_PAIRED_EXPERIMENT, "binding": binding}
            old_predictions = dict.fromkeys(tokens, "frozen")
            marker = {"attempt": str(prior), "manifest_sha256": bc.digest(manifest), "status": "passed",
                      "runtime_sha256": bc.digest(runtime), "first_pair_sha256": {"b": old_predictions[bc.TARGET]}}
            receipt = {"pilot_smoke_sha256": bc.digest(marker), "prediction_sha256": {"b": old_predictions}}
            report = {"status": "pass", "scorer": scorer, "unscored_tokens": [],
                      "prediction_receipt_sha256": bc.digest(receipt),
                      "cases": [{"token": r["token"], "session_id": r["session_id"], "b": raw()}
                                for r in plan["scenes"]]}
            for name, value in (("manifest.json", manifest), ("comparison.json", report),
                                ("paired-pilot/smoke.json", marker), ("paired-pilot/runtime.json", runtime),
                                ("paired-pilot/prediction-receipt.json", receipt)):
                bc.write(prior / name, value)
            stack.enter_context(patch.multiple(bc, PRIOR_COMPARISON=bc.digest(report), PRIOR_RUNTIME=bc.digest(runtime),
                                               PRIOR_RECEIPT=bc.digest(receipt)))
            stack.enter_context(patch.multiple(bc.night, brake_gap_archive=Mock(return_value=(binding, plan, frozen)),
                on_astra=Mock(side_effect=fake_bridge), with_corridor=Mock(return_value=plan),
                archive_evaluator=Mock(return_value=(for_scene, None))))
            identity_mock = stack.enter_context(patch.object(bc.night, "brake_gap_scorer_identity", return_value=scorer))
            seed = Mock()
            device = stack.enter_context(patch.object(bc.night, "current_brake_gap_runtime", return_value=(
                {**runtime, "slurm_job_id": "900"}, seed)))
            load = stack.enter_context(patch.object(driver, "DrexMax", side_effect=lambda _: Model()))

            results = {"pass": 0, "reject": 3, "missing": 2, "scorer": 2, "baseline": 2, "replay": 2}
            for i, (mode, expected) in enumerate(results.items()):
                with self.subTest(mode=mode):
                    root = runs / bc.EXPERIMENT / f"attempt-{i+1:03}"
                    args = SimpleNamespace(out=root, archive=archive, git_head="code", git_diff_sha256="clean", weights=None)

                    def run(stage):
                        args.stage = stage
                        return bc.run(args)

                    lane["mode"] = "pass"
                    device.return_value = ({**runtime, "slurm_job_id": "900"}, seed)
                    self.assertEqual(run("verify"), 0)
                    self.assertEqual(run("ready-smoke"), 0)
                    with self.assertRaises(FileNotFoundError):
                        run("ready-predict")
                    if mode == "baseline":
                        device.return_value = ({**runtime, "identity": {"weights": "wrong"}}, seed)
                        with self.assertRaisesRegex(RuntimeError, "checkpoint"):
                            run("smoke")
                        device.return_value = ({**runtime, "slurm_job_id": "900"}, seed)
                    self.assertEqual(run("smoke"), 0)
                    lane["mode"] = "baseline" if mode == "baseline" else "pass"
                    self.assertEqual(run("score-smoke"), 2 if mode == "baseline" else 0)
                    if mode == "baseline":
                        with self.assertRaisesRegex(RuntimeError, "has not scored"):
                            run("ready-predict")
                        self.assertFalse((root / "bc-pilot").exists())
                        continue
                    smoke_report = bc.read(root / "smoke-score.json")
                    self.assertEqual(smoke_report["status"], "smoke_verified")
                    self.assertEqual(run("ready-predict"), 0)
                    device.return_value = ({**runtime, "slurm_job_id": "901"}, seed)
                    if mode == "replay":
                        with patch.object(Model, "choose", return_value=("stop", None)), self.assertRaisesRegex(
                                RuntimeError, "does not replay"):
                            run("predict")
                        self.assertFalse((root / "bc-pilot" / "prediction-receipt.json").exists())
                        self.assertTrue((root / "bc-pilot" / "b" / f"{bc.TARGET}.json").exists())
                        continue
                    self.assertEqual(run("predict"), 0)
                    if mode == "missing":
                        (root / "bc-pilot" / "c" / f"{tokens[-1]}.json").unlink()
                    if mode == "scorer":
                        identity_mock.return_value = {"scorer": "mismatch"}
                    # Keep the replayed target raw unchanged while simulating a complete reject elsewhere.
                    real_evaluator = bc.night.archive_evaluator.return_value
                    if mode == "reject":
                        def reject_except_target(record):
                            lane["mode"] = "pass" if record["token"] == bc.TARGET else "reject"
                            return for_scene(record)
                        bc.night.archive_evaluator.return_value = (reject_except_target, None)
                    self.assertEqual(run("score"), expected)
                    identity_mock.return_value = scorer
                    bc.night.archive_evaluator.return_value = real_evaluator
                    result = bc.read(root / "comparison.json")
                    if expected in (0, 3):
                        self.assertEqual(len(result["cases"]), 23)
                        self.assertEqual(len(result["summary"]["per_drive"]), 14)
                        self.assertEqual(sum(result["summary"]["gate_coverage"]["c"].values()), 184)
                        self.assertEqual(result["summary"]["bootstrap_replicates"], 20_000)
                        self.assertEqual(result["status"], "pass" if expected == 0 else "rejected")
                        self.assertEqual(result["scored_smoke_sha256"], bc.digest(smoke_report))
                        if expected == 3:
                            self.assertEqual(len(result["scientific_gate_failures"]), 4)
                    else:
                        self.assertEqual(result["status"], "incomplete")
                        self.assertEqual(len(result["unscored_tokens"]), 23)
                        self.assertIn("first_failed_condition", result)
                    with self.assertRaises(FileExistsError):
                        run("score")
                    self.assertEqual(bc.read(root / "comparison.json"), result)
            self.assertEqual(load.call_count, 11)
            self.assertEqual(seed.call_count, 6 * 2 + 4 * 46 + 2)
            self.assertTrue(all(c.args == (bc.SEED,) for c in seed.call_args_list))


if __name__ == "__main__":
    unittest.main()
