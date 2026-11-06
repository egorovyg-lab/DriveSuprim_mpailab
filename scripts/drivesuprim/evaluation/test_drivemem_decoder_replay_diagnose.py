import contextlib
import copy
import io
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch

import numpy as np

from navsim.planning.script import run_drivemem_decoder_replay_diagnose as diagnostic
from test_drivemem_action_transfer import make_sample


class ReplayDiagnosticTests(unittest.TestCase):
    def test_reports_actual_tolerance_and_failed_candidates(self):
        reference = np.array([1.00002, 0., -0.50002])
        current = np.array([1., 0., -.5])
        result = diagnostic.numeric_detail(reference, current, [8, 9, 10], 1e-5, 1e-6)
        self.assertFalse(result["allclose"])
        self.assertEqual(result["failed_elements"], 2)
        self.assertEqual({r["vocab_id"] for r in result["worst_by_tolerance"]}, {8, 10})
        for row in result["worst_by_tolerance"]:
            self.assertAlmostEqual(row["allowed_abs_diff"], 1e-5 + 1e-6 * abs(row["current"]))
        invalid = diagnostic.numeric_detail(np.array([np.nan]), np.zeros(1), [5], 1e-5, 1e-6)
        self.assertEqual(invalid["nonfinite_elements"], 1)
        json.dumps(invalid, allow_nan=False)

    def test_constant_imi_shift_is_visible_without_relaxing_parity(self):
        rng = np.random.default_rng(104)
        old = make_sample("q", "q", "development", rng)
        old["head_imi"] = rng.normal(size=16)
        vocab = rng.normal(size=(16, 40, 3))
        fresh = copy.deepcopy(dict(old, final_traj=vocab[0]))
        fresh["head_imi"] += .001
        result = diagnostic.detailed_compare(old, fresh, vocab, "frozen_vs_hooked_replay")
        self.assertFalse(result["numeric_details"]["head_imi"]["allclose"])
        self.assertTrue(result["selected_vocab_equal"])
        self.assertTrue(result["top16_order_equal"])
        self.assertLess(result["imi_diagnostics"]["log_softmax_max_abs_diff_on_common_pool"], 1e-12)
        with self.assertRaisesRegex(RuntimeError, "head_imi"):
            diagnostic.exporter.parity(old, fresh, vocab)

    def test_targeted_probe_verifies_full_index_writes_nothing_and_restores_state(self):
        rng = np.random.default_rng(105)
        old = make_sample("q", "q", "development", rng)
        old["head_imi"] = rng.normal(size=16)
        vocab = rng.normal(size=(16, 40, 3))
        fresh = copy.deepcopy(dict(old, final_traj=vocab[0]))
        fresh["head_imi"] += .001
        unhooked = {k: v for k, v in fresh.items() if k != "token"}
        exporter = diagnostic.exporter
        originals = {name: getattr(exporter, name) for name in ("parity", "status", "write", "save")}
        cfg = N(drivemem=N(stage="decoder-replay-diagnose"))
        root = Path("source")
        persisted_index = {"source:q": {"checked": True}, "source:other": {"checked": True}}

        def verify_full_index(out, records, identity):
            self.assertEqual({r["token"] for _, r in records}, {"q", "other"})
            return persisted_index, dict(status="partial")

        def fake_main(cfg):
            exporter.write(None, {})
            records = [(root, dict(token=t)) for t in ("other", "q")]
            index, _ = exporter.status(None, records, {})
            self.assertEqual([r["token"] for _, r in records], ["q"])
            self.assertNotIn("source:q", index)
            self.assertIn("source:q", persisted_index)
            exporter.parity(old, fresh, vocab)
            exporter.parity(unhooked, fresh, vocab)
            exporter.save(None)
            self.fail("Diagnostic must stop before artifact saving")

        with patch.dict(os.environ, {"DRIVEMEM_DECODER_DIAG_TOKEN": "q"}), \
                patch.object(exporter, "main", side_effect=fake_main), \
                patch.object(exporter, "status", side_effect=verify_full_index), \
                patch.object(exporter, "write") as writer, patch.object(exporter, "save") as saver, \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            diagnostic.main(cfg)
            writer.assert_not_called()
            saver.assert_not_called()
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["token"], "q")
        self.assertEqual([r["comparison"] for r in result["comparisons"]],
                         ["frozen_vs_hooked_replay", "unhooked_vs_hooked_replay", "frozen_vs_unhooked_replay"])
        self.assertIn("head_imi", result["comparisons"][0]["strict_parity"])
        self.assertEqual(result["comparisons"][1]["strict_parity"], "passed")
        self.assertIn("head_imi", result["comparisons"][2]["strict_parity"])
        self.assertEqual(cfg.drivemem.stage, "decoder-replay-diagnose")
        for name, value in originals.items():
            self.assertIs(getattr(exporter, name), value)

    def test_failed_gpu_initialization_restores_exporter_and_stage(self):
        exporter = diagnostic.exporter
        cfg = N(drivemem=N(stage="decoder-replay-diagnose"))
        parity, writer = exporter.parity, exporter.write
        with patch.object(exporter, "main", side_effect=RuntimeError("CUDA unavailable")):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                diagnostic.main(cfg)
        self.assertIs(exporter.parity, parity)
        self.assertIs(exporter.write, writer)
        self.assertEqual(cfg.drivemem.stage, "decoder-replay-diagnose")


if __name__ == "__main__":
    unittest.main()
