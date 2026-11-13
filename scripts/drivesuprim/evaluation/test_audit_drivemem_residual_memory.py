from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import audit_drivemem_residual_memory as audit
from navsim.agents import drivemem_residual_memory as model
from navsim.planning.script import run_drivemem_residual_memory as pipeline


class AuditTests(unittest.TestCase):
    def test_relations_distinguish_wrong_sign_overshoot_and_help(self):
        self.assertEqual(audit.relation(-.5, .1), "wrong_direction")
        self.assertEqual(audit.relation(.1, .3), "same_direction_overshoot")
        self.assertEqual(audit.relation(.1, .15), "gap_error_reduced")
        self.assertEqual(audit.relation(.1, 0), "no_memory_gap")

    def test_donor_weights_geometry_and_query_session_are_checked(self):
        entries, raws = pipeline.synthetic()
        mean, scale = model.scaler(entries)
        residuals = {e["token"]: np.r_[0., np.ones(63)*.2] for e in entries}
        bank = model.Memory(entries, residuals, mean, scale)
        query = entries[0]
        correction, traces = bank.recall(query, 1., model.Config())
        fold = dict(records=bank.records, mean=mean, scale=scale, residuals=bank.residuals, h=1.,
                    training_sessions=bank.sessions,
                    sources={e["token"]: dict(fit="test", training_sessions=["other"]) for e in entries},
                    traces={"nearest_"+k: v[None] for k, v in traces.items()})
        fold["training_sessions"].append("other")
        result = audit.donor_action(query, 3, 0, fold, {e["token"]: e for e in entries}, raws)
        self.assertAlmostEqual(result["correction"], correction[3])
        self.assertAlmostEqual(sum(r["contribution_pp"] for r in result["donors"]), 100*correction[3])
        self.assertEqual(len({r["session_id"] for r in result["donors"]}), 5)
        fold["traces"]["nearest_weight"][0, 3, 0] *= .5
        with self.assertRaisesRegex(ValueError, "weights do not replay"):
            audit.donor_action(query, 3, 0, fold, {e["token"]: e for e in entries}, raws)

    def test_complete_audit_is_readonly_reproducible_and_never_fits(self):
        import torch
        torch.set_num_threads(1)
        config = replace(model.Config(), width=8, epochs=1, probes=8, bootstrap_reps=100)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)/"source"; root.mkdir()
            out = Path(tmp)/"audit"
            entries, raws = pipeline.synthetic()
            identity = pipeline.initialize(root, entries, dict(bindings={}), config, 1, True)
            pipeline.prepare(root, entries, raws)
            entries = pipeline.prepared(root)
            for i in range(4):
                pipeline.run_fold(root, entries, identity["groups"], i, config)
            pipeline.evaluate(root, entries, identity["groups"], config)
            before = {str(p.relative_to(root)): pipeline.sha(p) for p in root.rglob("*") if p.is_file()}
            with mock.patch.object(model, "train", side_effect=AssertionError("Audit trained")), \
                 mock.patch.object(model, "predict", side_effect=AssertionError("Audit predicted")):
                result = audit.audit(root, out)
                self.assertEqual(result, audit.audit(root, out))
            self.assertEqual(result["new_models"], 0)
            after = {str(p.relative_to(root)): pipeline.sha(p) for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            with self.assertRaisesRegex(ValueError, "separate"):
                audit.audit(root, root/"unsafe")
            with (root/"folds/0/decisions.npz").open("ab") as f:
                f.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                audit.audit(root, Path(tmp)/"another")


if __name__ == "__main__":
    unittest.main()
