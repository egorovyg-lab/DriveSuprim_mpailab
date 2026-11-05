import ast
import contextlib
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from navsim.agents import drivemem_component_utility as old_model
from navsim.agents import drivemem_decoder_utility as model
from navsim.planning.script import run_drivemem_decoder_export as exporter
from navsim.planning.script import run_drivemem_decoder_utility as runner
from navsim.planning.script import run_drivemem_component_utility as old_runner
from test_drivemem_action_transfer import make_sample
from test_drivemem_pair_utility import outcomes


class DecoderTests(unittest.TestCase):
    def test_feature_only_ablation_and_shared_algorithms(self):
        rng = np.random.default_rng(9)
        s = make_sample("q", "q", "development", rng)
        s["decoder_states"] = rng.normal(size=(16, 7))
        vocab = rng.normal(size=(16, 40, 3))
        old, _ = old_model.features(s, vocab)
        new, _ = model.features(s, vocab)
        np.testing.assert_array_equal(new[:, :old.shape[1]], old)
        np.testing.assert_array_equal(new[:, -14:-7], s["decoder_states"])
        np.testing.assert_array_equal(new[:, -7:], s["decoder_states"] - s["decoder_states"][0])
        self.assertIs(model.ridge_path, old_model.ridge_path)
        self.assertIs(model.choose, old_model.choose)
        def functions(module):
            return {n.name: ast.dump(n) for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
        a, b = functions(old_runner), functions(runner)
        for name in ("build_features", "cv_score", "fit_models", "evaluate"):
            self.assertEqual(a[name], b[name])

    def test_export_parity_rejects_alignment_and_score_changes(self):
        rng = np.random.default_rng(6)
        s = make_sample("q", "q", "development", rng)
        vocab = rng.normal(size=(16, 40, 3))
        fresh = dict(s, final_traj=vocab[0].copy())
        self.assertEqual(exporter.parity(s, fresh, vocab), 0)
        bad = copy.deepcopy(fresh)
        bad["candidate_ids"] = bad["candidate_ids"][::-1]
        with self.assertRaises(RuntimeError):
            exporter.parity(s, bad, vocab)
        bad = copy.deepcopy(fresh)
        bad["selection_scores"][2] += .1
        with self.assertRaises(RuntimeError):
            exporter.parity(s, bad, vocab)

    def test_export_selection_excludes_closed_and_unselected_scenes(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            source, addition, val = [Path(tmp) / n for n in ("source", "addition", "val")]
            for root, rows in ((source, [dict(token="m", role="memory"), dict(token="f", role="fit"), dict(token="v", role="development"), dict(token="c", role="calibration"), dict(token="t", role="test")]),
                               (addition, [dict(token="new", role="memory"), dict(token="skip", role="fit")]),
                               (val, [dict(token="v2", role="development"), dict(token="closed", role="test")])):
                exporter.write(root / "splits.json", dict(records=rows))
            exporter.write(addition / "label_selection.json", dict(records=[dict(token="new")]))
            rows = exporter.wanted_records(source, addition, val)
            self.assertEqual({r["token"] for _, r in rows}, {"m", "f", "v", "new", "v2"})

    def test_sidecar_candidate_mismatch_is_rejected(self):
        rng = np.random.default_rng(7)
        s = make_sample("q", "q", "fit", rng)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root, decoder = Path(tmp) / "source", Path(tmp) / "decoder"
            path = exporter.feature_path(decoder, root, "q")
            path.parent.mkdir(parents=True)
            exporter.save(path, candidate_ids=s["candidate_ids"][::-1], decoder_states=np.zeros((16, 3)))
            exporter.write(decoder / "export_index.json", {exporter.key(root, "q"): dict(sha256=exporter.sha(path), source_sha256="source")})
            with patch.object(model, "load_sample", return_value=s):
                with self.assertRaisesRegex(ValueError, "alignment"):
                    model.load_decoder_sample(root, dict(token="q"), {"q": "source"}, decoder)

    def test_decoder_pipeline_freezes_queries_and_resumes(self):
        rng = np.random.default_rng(32)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            source, val, comp, addition, out, decoder = [root / n for n in ("source", "val", "comp", "addition", "out", "decoder")]
            samples, targets = [], {}
            for i in range(4):
                s = make_sample(str(i), str(i), "memory" if i < 2 else "fit", rng)
                s["decoder_states"] = rng.normal(size=(16, 7))
                targets[s["token"]] = outcomes(s, rng)
                samples.append(s)
            vocab = rng.normal(size=(16, 40, 3))
            runner.write(addition / "label_selection.json", dict(records=[]))
            expected, queries = {}, {}
            for n, cohort in enumerate((source, val)):
                q = make_sample("q" + str(n), "q" + str(n), "development", rng)
                q["decoder_states"] = rng.normal(size=(16, 7))
                queries[q["token"]] = q
                runner.write(cohort / "splits.json", dict(records=[dict(token=q["token"], session_id=q["session_id"], role="development")]))
                runner.write(cohort / "export_index.json", {})
                path = cohort / "development_oracle/scenes" / (q["token"] + ".json")
                runner.write(path, dict(token=q["token"], outcomes=outcomes(q, rng)))
                expected[cohort.name] = {q["token"]: runner.sha(path)}
            runner.write(comp / "report.json", dict(oracle_inputs=expected))
            data = (samples[:2], samples[2:], targets, vocab, source, val, comp, {}, {}, 1)
            original = runner.read
            def read(p):
                if "development_oracle" in str(p):
                    self.assertEqual(set(original(out / "query_decisions.json")["cohorts"]), {"source", "val"})
                return original(p)
            argv = ["runner", "all", "--prior", str(root / "prior"), "--addition", str(addition), "--out", str(out), "--decoder", str(decoder)]
            with patch.object(runner, "load_decoder_inputs", return_value=data), patch.object(runner, "read", side_effect=read), patch.object(runner, "load_decoder_sample", side_effect=lambda rt, r, ix, d: copy.deepcopy(queries[r["token"]])), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
                digest = runner.sha(out / "fit.json")
                runner.main()
                self.assertEqual(runner.sha(out / "fit.json"), digest)
                self.assertEqual(original(out / "report.json")["status"], "complete")


if __name__ == "__main__":
    unittest.main()
