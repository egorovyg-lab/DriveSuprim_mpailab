import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from navsim.planning.script import run_drivemem_expanded_utility as runner
from navsim.agents.drivemem_pair_utility import attach_utility
from test_drivemem_action_transfer import make_sample
from test_drivemem_pair_utility import outcomes


class ExpandedUtilityTests(unittest.TestCase):
    def test_partition_rejects_queries_duplicates_and_session_overlap(self):
        samples = [dict(token="m", role="memory", session_id="m"), dict(token="f", role="fit", session_id="f")]
        self.assertEqual(tuple(map(len, runner.partition(samples))), (1, 1))
        for bad in (samples + [samples[0]], samples + [dict(token="t", role="test", session_id="t")],
                    [samples[0], dict(samples[1], session_id="m")]):
            with self.assertRaises(ValueError):
                runner.partition(bad)

    def test_full_refit_freezes_decisions_before_queries_and_resumes(self):
        rng = np.random.default_rng(62)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            prior, source, validation, completion, out = [root / n for n in ("prior", "source", "validation", "completion", "out")]
            train, targets = [], {}
            for i in range(4):
                s = make_sample(str(i), str(i), "memory" if i < 2 else "fit", rng)
                targets[s["token"]] = outcomes(s, rng)
                train.append(attach_utility(s, targets[s["token"]]))
            bank, fitting = runner.partition(train)
            vocab = rng.normal(size=(16, 40, 3))
            oracles, by_token = {}, {}
            for n, cohort in enumerate((source, validation)):
                q = make_sample("q" + str(n), "q" + str(n), "development", rng)
                by_token[q["token"]] = q
                runner.write(cohort / "splits.json", dict(records=[dict(token=q["token"], session_id=q["session_id"], role="development"),
                    dict(token="never_read", session_id="ct", role="calibration")]))
                runner.write(cohort / "export_index.json", {})
                path = cohort / "development_oracle/scenes" / (q["token"] + ".json")
                runner.write(path, dict(token=q["token"], outcomes=outcomes(q, rng)))
                oracles[cohort.name] = {q["token"]: runner.sha(path)}
            runner.write(completion / "report.json", dict(oracle_inputs=oracles))
            runner.write(prior / "report.json", dict(cohorts={r.name: dict(methods={m: dict(gain_macro_pp=0., harm=0) for m in runner.METHODS}) for r in (source, validation)}))
            data = (bank, fitting, targets, vocab, source, validation, completion, {}, dict(memory_scenes=2, fit_scenes=2), 1)
            original_read = runner.read
            def read(p):
                if "development_oracle" in str(p):
                    self.assertTrue((out / "fit.json").exists())
                    saved = original_read(out / "query_decisions.json")
                    self.assertEqual(set(saved["cohorts"]), {"source", "validation"})
                return original_read(p)
            # evaluate() is reused unchanged; inject the read function explicitly
            # through main so the same ordering assertion covers both cohorts.
            argv = ["runner", "plan", "--prior", str(prior), "--addition", str(root / "addition"), "--out", str(out)]
            with patch.object(runner, "load_inputs", return_value=data), patch.object(runner, "read", side_effect=read), patch.object(runner, "load_sample", side_effect=lambda r, row, ix: copy.deepcopy(by_token[row["token"]])), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
                self.assertFalse(out.exists())
                argv[1] = "all"
                runner.main()
                fitted = runner.sha(out / "fit.json")
                self.assertEqual(original_read(out / "report.json")["status"], "complete")
                runner.main()
                self.assertEqual(runner.sha(out / "fit.json"), fitted)
                (out / "direct_memory.npz").write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "model changed"):
                    runner.main()


if __name__ == "__main__":
    unittest.main()
