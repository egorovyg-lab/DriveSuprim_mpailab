import copy
import unittest
from pathlib import Path
import ast

import numpy as np

from navsim.planning.script import run_drivemem_decoder_numeric_audit as scan
import test_drivemem_decoder_numeric as fixtures


class FullAuditTests(unittest.TestCase):
    def test_failed_contract_is_retained_and_does_not_stop_next_scene(self):
        old, fresh, vocab = fixtures.NumericContractTests().fixture()
        fresh["head_imi"][3] += 5.245208740234375e-5
        failed = scan.assess(old, copy.deepcopy(fresh), fresh, vocab, vocab, {"imi": True})
        self.assertTrue(failed["repeat_bitwise_equal"])
        self.assertFalse(failed["contract_passed"])
        self.assertIn("Logit budget", failed["errors"]["contract"])
        failed.update(token="failed", role="memory")
        old, fresh, vocab = fixtures.NumericContractTests().fixture()
        passed = scan.assess(old, copy.deepcopy(fresh), fresh, vocab, vocab, {"imi": True})
        passed.update(token="passed", role="fit")
        summary = scan.summarize([failed, passed], 2)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["audited_scenes"], 2)
        self.assertEqual(summary["contract_and_invariants_passed"], 1)
        self.assertEqual(summary["repeat_failures"], 0)
        self.assertEqual([r["token"] for r in summary["violations"]], ["failed"])

    def test_current_repeat_vocabulary_and_head_alignment_failures_are_separate(self):
        old, fresh, vocab = fixtures.NumericContractTests().fixture()
        changed = copy.deepcopy(fresh)
        changed["head_imi"][0] += 1e-6
        row = scan.assess(old, changed, fresh, vocab, vocab + 1, {"imi": False})
        row.update(token="q", role="development")
        report = scan.summarize([row], 152)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["repeat_failures"], 1)
        self.assertEqual(report["alignment_failures"], 1)
        self.assertEqual(report["vocabulary_changes"], 1)
        self.assertEqual(report["contract_and_invariants_passed"], 0)

    def test_audit_module_has_no_feature_save_or_learner_calls(self):
        tree = ast.parse(Path(scan.__file__).read_text(encoding="utf-8"))
        functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertNotIn("fit", functions)
        calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertTrue({"save", "save_npz", "fit_models", "evaluate", "evaluators"}.isdisjoint(calls))


if __name__ == "__main__":
    unittest.main()
