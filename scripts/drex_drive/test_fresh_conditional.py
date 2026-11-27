"""Synthetic transaction tests only; no Drex runtime, raw caches, torch or scorer imports."""

from copy import deepcopy
from hashlib import sha256
import io
import os
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from . import fresh_conditional as fc

H = "ab" * 32
REV = "12" * 20


def pack():
    return {
        "experiment_id": fc.EXPERIMENT, "dataset": {"source_id": "fictional", "version": "v1", "sha256": H},
        "frame": {"frame_id": "fictional-frame", "universe_sha256": H, "adapter_sha256": H,
                  "filter_sha256": fc.night.file_sha256(Path(fc.straight.__file__))},
        "exclusions": {name: {"evidence_sha256": H, "tokens": [f"{1000 + j * 100 + i:016x}" for i in range(n)],
                              "session_ids": [f"{name}-{i}" for i in range(n)]}
                       for j, (name, n) in enumerate(zip(fc.EXCLUSIONS, (32, 4, 2)))},
        "eligible_scenes": [
            {"token": f"{i:016x}", "session_id": f"drive-{i:03d}", "cache_sha256": H, "speed_mps": 5.0,
             "leaders": [] if i % 7 == 0 else [{"gap_m": 16.0, "speed_mps": 4.9999}],
             "corridor_objects": [], "ego_bounds": {"front_m": 3.8, "rear_m": -1.0}}
            for i in range(125)
        ],
    }


class Array:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class FakeModel:
    def __init__(self):
        self.model = self

    def prepare(self, requests, max_length):
        has_field = fc.night.BRAKE_GAP_FIELD in fc.night.request_json(requests[0])
        # Same token length but different token IDs. Truncation check must compare sequences.
        return SimpleNamespace(inputs={"input_ids": Array([[101, int(has_field)]]),
                                       "attention_mask": Array([[1, 1]])})

    def choose(self, request):
        has_field = fc.night.BRAKE_GAP_FIELD in fc.night.request_json(request)
        action = "proceed" if has_field else "maintain"
        return action, None


def fake_evaluator(scene):
    def evaluate(case):
        ep = 0.6 if case["decisions"][0]["action"] == "maintain" else 0.7 + int(scene["token"], 16) % 5 / 100
        return {**dict.fromkeys(fc.PROTECTED, 1.0), fc.EP: ep, fc.PDM: (5*ep + 9) / 14}
    return evaluate


def write(path, obj):
    fc.night.write_brake_gap_artifact(path, obj)
    return fc.night.file_sha256(path)


class FreshConditional(TestCase):
    def test_fixed_draw_and_full_drive_retention(self):
        frame = pack()
        extra = deepcopy(frame["eligible_scenes"][0])
        extra.update(token=f"{140:016x}", session_id="drive-001")
        frame["eligible_scenes"].append(extra)
        order = fc.select(frame, {})
        signature = sha256("\n".join(order["session_ids"]).encode()).hexdigest()
        # Independent comparison against explicit fixed ranking, without relying on input order.
        expected = sorted(sorted((f"drive-{i:03d}" for i in range(125)),
                                 key=lambda s: sha256(("20261125:" + s).encode()).hexdigest())[:120])
        self.assertEqual(signature, sha256("\n".join(expected).encode()).hexdigest())
        self.assertEqual(sum(r["session_id"] == "drive-001" for r in order["scenes"]), 2)
        frame["eligible_scenes"].reverse()
        self.assertEqual(order, fc.select(frame, {}))
        self.assertTrue(any(not r["leaders"] for r in order["scenes"]))

    def test_missing_frame_exclusions_and_leakage(self):
        frame = pack()
        scene = frame["eligible_scenes"][0]
        for label, edit, message in (
            ("future", {"future_tracks": []}, "unexpected/missing fields"),
            ("nested", {"leaders": [{"speed_mps": 1, "gap_m": 3, "poses": []}]}, "unexpected/missing fields"),
            ("score", {"pdm_score": 1}, "unexpected/missing fields"),
            ("session", {"session_id": "reserve4-1"}, "conflicts"),
            ("token", {"token": frame["exclusions"]["other_prior"]["tokens"][0]}, "conflicts"),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, message):
                fc.select({**frame, "eligible_scenes": [{**scene, **edit}]}, {})
        with self.assertRaisesRegex(ValueError, "Need 120"):
            fc.select({**frame, "eligible_scenes": frame["eligible_scenes"][:119]}, {})
        with self.assertRaisesRegex(ValueError, "verified archived subset"):
            fc.select(frame, {"f43160170a665c31": "other-drive"})

    def test_own_simulated_gate_and_token_budget(self):
        row = pack()["eligible_scenes"][1]
        model = FakeModel()
        pair = fc.make_pair(row, model, lambda _: None)
        fc.check_pair(pair, row)
        self.assertEqual(pair["a"]["decisions"][0]["reason"], "closing")  # printed radar rounds to 0
        self.assertEqual(pair["c"]["decisions"][0]["reason"], "closing")
        self.assertNotEqual(pair["a"]["pose_sha256"], pair["c"]["pose_sha256"])
        for speed, reason in ((5.0, "zero"), (5.0001, "opening"), (None, "no_leader")):
            scene = {**row, "leaders": [] if speed is None else [{"gap_m": 16.0, "speed_mps": speed}]}
            trace = fc.make_pair(scene, model, lambda _: None)
            self.assertTrue(all(trace[a]["decisions"][0]["reason"] == reason for a in fc.ARMS))
            self.assertEqual(trace["a"]["poses"], trace["c"]["poses"])
        model.prepare = lambda *_args, **_kwargs: SimpleNamespace(
            inputs={"input_ids": Array([[1]]), "attention_mask": Array([[1]])})
        with self.assertRaisesRegex(RuntimeError, "erased"):
            fc.make_pair(row, model, lambda _: None)

    def test_pair_replay_works_with_python39_zip(self):
        row = pack()["eligible_scenes"][1]
        # Astra's pinned Python 3.9 has no zip(strict=...).
        with patch.object(fc, "zip", lambda *items: zip(*items), create=True):
            pair = fc.make_pair(row, FakeModel(), lambda _: None)
            fc.check_pair(pair, row)
            with self.assertRaisesRegex(ValueError, "eight matched"):
                fc.compact({"decisions": [{}] * 8}, [{}] * 7)

    def test_astra_replay_imports_do_not_evaluate_union_annotations(self):
        # This checks the import-time fix on any test host. It is not a substitute
        # for running the replay in Astra's pinned interpreter.
        from . import driver, simulation

        self.assertIsInstance(simulation.Scenario.__annotations__["stop_line_m"], str)
        self.assertIsInstance(simulation.World.__annotations__["lead"], str)
        self.assertIsInstance(driver.driving_request.__annotations__["previous_speed"], str)
        scene = pack()["eligible_scenes"][1]
        fc.check_pair(fc.make_pair(scene, FakeModel(), lambda _: None), scene)

    def test_complete_transaction_and_first_incomplete(self):
        with TemporaryDirectory(prefix="drex-ac-synthetic-", dir=os.environ.get("DREX_SYNTHETIC_TMP")) as temp:
            root = Path(temp).resolve()
            folders = {name: root / name for name in ("steward", "smoke", "freeze", "predict", "score", "validate",
                                                       "bad", "bad-score", "dummy")}
            for p in folders.values():
                p.mkdir()
            sources = fc.code(REV, H)
            old_runtime = {"identity": {"fixture": H}, "seed_per_arm": fc.SEED, "slurm_job_id": "fictional-smoke",
                           "gpu": "fictional", "gpu_capability": [9, 9]}
            runtime = {**old_runtime, "slurm_job_id": "fictional-full"}
            scorer = {"synthetic": H}
            archive_path = folders["smoke"] / "score.json"
            data = pack()
            archived_tokens = data["exclusions"]["archive32"]["tokens"]
            archived_tokens[:2] = [fc.bc.TARGET, fc.bc.OTHER]
            known = {t: f"archive32-{i % 14}" for i, t in enumerate(archived_tokens[:23])}
            archive_sha = write(archive_path, {"experiment_id": fc.EXPERIMENT, "stage": "archive-score",
                "status": "smoke_verified", "code": sources, "prior_comparison_sha256": fc.bc.PRIOR_COMPARISON,
                "token": fc.bc.TARGET, "known_archive": known, "runtime": old_runtime, "scorer": scorer})
            pack_path = folders["steward"] / "export.json"
            approval = {**dict.fromkeys(("steward", "independence_owner", "reviewed_by", "source_approval_ref",
                "independence_approval_ref", "exclusion_review_ref", "preoutcome_review_ref", "precision_decision_ref",
                "capacity_decision_ref"), "synthetic-only"), "experiment_id": fc.EXPERIMENT, "code": sources,
                "max_scenes": 130, "msu_wall_seconds": 120, "astra_wall_seconds": 120,
                "archive_score_sha256": archive_sha, "export_sha256": write(pack_path, data)}
            approval_path = folders["steward"] / "approval.json"
            approval_sha = write(approval_path, approval)
            name = "attempt-001"
            frozen = folders["freeze"] / name
            with redirect_stdout(io.StringIO()):
                self.assertEqual(fc.main(["freeze", "--out", str(frozen), "--export", str(pack_path),
                     "--attestation", str(approval_path), "--attestation-sha256", approval_sha,
                     "--archive-score", str(archive_path), "--operator", "synthetic-only", "--revision", REV,
                     "--diff-sha256", H]), 0)
            manifest = frozen / "manifest.json"
            plan_sha = fc.night.file_sha256(manifest)
            plan = fc.load_plan(manifest, plan_sha)[0]
            self.assertEqual(len(plan["scenes"]), 120)
            prediction, scores = folders["predict"] / name, folders["score"] / name
            seeds, loads = [], []
            plan_args = ["--manifest", str(manifest), "--manifest-sha256", plan_sha]
            dummy_path = folders["dummy"] / "dummy.json"
            dummy_sha = write(dummy_path, {})
            with redirect_stdout(io.StringIO()), patch.object(fc.night, "current_brake_gap_runtime",
                    return_value=(runtime, seeds.append)), patch("scripts.drex_drive.driver.DrexMax",
                    side_effect=lambda path: (loads.append(path) or FakeModel())):
                self.assertEqual(fc.main(["predict", *plan_args, "--out", str(prediction),
                    "--weights", str(folders["dummy"]), "--upstream", str(folders["dummy"])]), 0)
            prediction_path = prediction / "prediction.json"
            pred_sha = fc.night.file_sha256(prediction_path)
            prediction_args = ["--prediction", str(prediction_path), "--prediction-sha256", pred_sha]
            astra_args = ["score", *plan_args, *prediction_args, "--navsim-repo", str(folders["dummy"]),
                          "--tools", str(folders["dummy"]), "--template", str(dummy_path),
                          "--template-sha256", dummy_sha, "--cache-root", str(folders["dummy"]),
                          "--cache-index", str(dummy_path), "--cache-index-sha256", dummy_sha]
            with redirect_stdout(io.StringIO()), patch.object(fc, "astra_evaluator",
                    return_value=(scorer, fake_evaluator, {"cache_index_sha256": H, "template_sha256": H})):
                self.assertEqual(fc.main([*astra_args, "--out", str(scores)]), 0)
            self.assertEqual(seeds, [fc.SEED] * 240)
            self.assertEqual(loads, [folders["dummy"]])
            score_path = scores / "score.json"
            args = ["validate", *plan_args, *prediction_args, "--scores", str(score_path),
                    "--scores-sha256", fc.night.file_sha256(score_path)]
            valid = folders["validate"] / name
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(fc.main([*args, "--out", str(valid)]), 0)
                self.assertEqual(fc.main([*args, "--out", str(valid)]), 2)  # exclusive retry
            report = fc.read(valid / "report.json")[0]
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["bootstrap"]["nearest_rank_zero_based"], 999)
            self.assertAlmostEqual(report["zero_events"]["upper_95"], 0.024655, places=5)
            self.assertEqual(len(report["cases"]), 120)
            self.assertGreater(report["gate_counts"]["c"]["no_leader"], 0)
            broken = folders["bad"] / name
            second = plan["scenes"][1]["token"]
            good = prediction / "pairs" / f"{second}.json"
            # In the fixture, mutate only the synthetic receipt to verify byte pin and first stop.
            good.write_bytes(good.read_bytes() + b" ")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(fc.main([*args, "--out", str(broken)]), 2)
            self.assertIn("Byte SHA-256 mismatch", fc.read(broken / "failure.json")[0]["first_failed_condition"])
            self.assertFalse((broken / "report.json").exists())
            failed_score = folders["bad-score"] / name
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), patch.object(fc, "astra_evaluator",
                    return_value=(scorer, fake_evaluator, {})):
                self.assertEqual(fc.main([*astra_args, "--out", str(failed_score)]), 2)
            self.assertEqual([p.stem for p in (failed_score / "pairs").iterdir()], [plan["scenes"][0]["token"]])
            self.assertFalse((failed_score / "score.json").exists())
            good.unlink()
            with self.assertRaisesRegex(ValueError, "membership"):
                fc.predicted(prediction_path, pred_sha, plan["scenes"])
            with self.assertRaisesRegex(ValueError, "Byte SHA-256"):
                fc.load_plan(manifest, H)

    def test_degenerate_and_complete_rejection(self):
        rows = [{"token": f"{i:016x}", "session_id": str(i), "delta": dict.fromkeys(fc.METRICS, 0.0),
                 "raw_a": dict.fromkeys(fc.PROTECTED, 1.0), "raw_c": dict.fromkeys(fc.PROTECTED, 1.0),
                 "gates": {a: {"no_leader": 8} for a in fc.ARMS}, "distance_delta_m": 0.0} for i in range(120)]
        result = fc.analyse(rows)
        self.assertEqual(result["status"], "inconclusive")
        self.assertIsNone(result["bootstrap"])
        rows[0]["delta"][fc.NC] = -fc.TOL
        self.assertEqual(fc.analyse(rows)["status"], "inconclusive")
        rows[0]["delta"][fc.NC] *= 2
        rows[0]["raw_c"][fc.NC] += rows[0]["delta"][fc.NC]
        result = fc.analyse(rows)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["protected_losses"][0]["metric"], fc.NC)
        with self.assertRaisesRegex(ValueError, "formula"):
            fc.raw({**dict.fromkeys(fc.PROTECTED, 1.0), fc.EP: 1, fc.PDM: 0}, "fixture")
