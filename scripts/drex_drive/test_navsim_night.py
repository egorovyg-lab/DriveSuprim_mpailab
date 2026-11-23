from pathlib import Path
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
        self.assertLess(len(code.encode("utf-8")), 95_000)


if __name__ == "__main__":
    unittest.main()
