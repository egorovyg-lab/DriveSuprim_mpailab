import copy
import contextlib
import io
import json
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch

import numpy as np

from navsim.agents.drivemem import METRICS, memory_correction, probabilities
from navsim.agents.drivemem_action_transfer import transfer, choose, ridge_path, features, ridge_predict
from navsim.planning.script import run_drivemem_action_transfer as runner


def make_sample(token, sid, role, rng):
    s = dict(token=token, session_id=sid, role=role, candidate_ids=np.arange(16),
             selected_local=np.array(0), selection_scores=np.arange(16., 0., -1),
             scene_embedding=rng.normal(size=3), ego_status=np.zeros(16))
    s.update({"head_" + k: rng.normal(size=16) for k in METRICS})
    s.update(label_local_indices=np.arange(8), label_values=rng.uniform(.2, 1, size=(8, len(METRICS))))
    return s


class ActionTransferTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4)
        self.vocab = self.rng.normal(size=(16, 40, 3))
        self.q = make_sample("q", "q", "development", self.rng)
        self.bank = [make_sample("m1", "m1", "memory", self.rng), make_sample("m2", "m2", "memory", self.rng)]

    def test_B0_parity_contrast_and_noop(self):
        c, _ = transfer(self.q, self.bank, self.vocab, .25)
        np.testing.assert_allclose(c, memory_correction(self.q, self.bank, [0, 1], self.vocab), atol=1e-12)
        contrast, _ = transfer(self.q, self.bank, self.vocab, .25, contrast=True)
        np.testing.assert_allclose(contrast, c - c[0], atol=1e-12)
        self.assertEqual(choose(self.q, contrast, beta=0), 0)
        empty, stats = transfer(self.q, [], self.vocab, .25)
        self.assertEqual(choose(self.q, empty), 0)
        self.assertEqual(stats["donors"], 0)

    def test_candidate_permutation_and_single_action_degeneracy(self):
        c, _ = transfer(self.q, self.bank, self.vocab, .02)
        order = self.rng.permutation(16)
        permuted = dict(self.q, selected_local=np.array(int(np.flatnonzero(order == 0)[0])))
        for key in ["candidate_ids", "selection_scores"] + ["head_" + k for k in METRICS]:
            permuted[key] = self.q[key][order]
        other, _ = transfer(permuted, self.bank, self.vocab, .02)
        np.testing.assert_allclose(other, c[order], atol=1e-12)
        self.assertEqual(int(permuted["candidate_ids"][choose(permuted, other)]), int(self.q["candidate_ids"][choose(self.q, c)]))
        one = copy.deepcopy(self.bank)
        for donor in one:
            donor["label_local_indices"] = donor["label_local_indices"][:1]
            donor["label_values"] = donor["label_values"][:1]
        c, _ = transfer(self.q, one, self.vocab, .001, contrast=True)
        np.testing.assert_allclose(c, 0)

    def test_full_embedding_dual_ridge_matches_primal_solution(self):
        self.bank[1]["label_local_indices"] = self.bank[1]["label_local_indices"][:4]
        self.bank[1]["label_values"] = self.bank[1]["label_values"][:4]
        models = ridge_path(self.bank, self.vocab, [10.])
        model = models[10.]
        x = np.concatenate([features(s, self.vocab, s["label_local_indices"]) for s in self.bank])
        y = np.concatenate([s["label_values"] - probabilities(s)[s["label_local_indices"]] for s in self.bank])
        z = (x - model["mean"]) / model["scale"]
        w = np.concatenate([np.full(len(s["label_local_indices"]), 1 / len(s["label_local_indices"])) for s in self.bank])
        w /= w.mean()
        coef = np.linalg.solve(z.T @ (z * w[:, None]) + 10 * np.eye(z.shape[1]), z.T @ ((y - model["ym"]) * w[:, None]))
        np.testing.assert_allclose(model["coef"], coef, atol=1e-10)
        self.assertTrue(np.isfinite(ridge_predict(self.q, self.vocab, model)).all())

    def test_pipeline_resumes_and_freezes_fit_before_development_lookup(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            source, out = root / "source", root / "experiment"
            source.mkdir()
            samples = copy.deepcopy(self.bank) + [make_sample("f1", "f1", "fit", self.rng), make_sample("f2", "f2", "fit", self.rng)]
            rows = [dict(token=s["token"], session_id=s["session_id"], role=s["role"]) for s in samples + [self.q]]
            rows += [dict(token="unread_" + role, session_id=role, role=role) for role in ("calibration", "test")]

            def read(path):
                if "development_oracle" in str(path):
                    self.assertTrue((out / "fit.json").is_file())
                    self.assertTrue((out / "development_decisions.json").is_file())
                return json.loads(path.read_text())

            def write(path, data):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))

            def sha(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            def outcome(index):
                values = {k: 1. for k in METRICS}
                values.update(pdm_score=.5 + .02 * index, aggregate_without_two_frame=.5 + .02 * index)
                return dict(raw=dict(values), filtered=dict(values))

            for name in ("identity.json", "split_identity.json", "export_index.json", "cache_index.json", "fit.json"):
                write(source / name, {})
            write(source / "splits.json", dict(records=rows))
            np.savez(source / "vocab.npz", vocab=self.vocab)
            for s in samples:
                write(source / "labels" / (s["token"] + ".json"), dict(local_indices=list(range(8)), values=[outcome(i) for i in range(8)]))
            write(source / "development_oracle/scenes/q.json", dict(outcomes={str(i): outcome(i) for i in range(16)}))
            by_token = {s["token"]: s for s in samples + [self.q]}
            calls = []
            failed = [False]

            def evaluate(token, index):
                self.assertTrue(token.startswith("f"), "Simulator must only label F")
                calls.append((token, index))
                if token == "f1" and index == 10 and not failed[0]:
                    failed[0] = True
                    raise RuntimeError("simulated interruption")
                return outcome(index), None

            fake = N(verify_identity=lambda *a: None, read_json=read, write_json=write, sha=sha,
                     sample=lambda r, row: copy.deepcopy(by_token[row["token"]]),
                     training_samples=lambda *a: copy.deepcopy(samples), evaluators=lambda *a: (evaluate, self.vocab),
                     save_npz=lambda path, **data: np.savez(path, **data))
            cfg = N(drivemem=N(root=str(source), memory_mode="enriched", stage="action-plan", seed=1))
            env = {"NAVSIM_EXP_ROOT": str(root), "DRIVEMEM_ACTION_ROOT": str(out)}
            with patch.dict(sys.modules, {"navsim.planning.script.run_drivemem_memory": fake}), patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
                runner.main(cfg)
                self.assertFalse(out.exists())
                cfg.drivemem.stage = "action-label"
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    runner.main(cfg)
                partial = read(out / "fit_labels/f1.json")["outcomes"]
                self.assertIn("9", partial)
                self.assertNotIn("10", partial)
                cfg.drivemem.stage = "action-all"
                runner.main(cfg)
                self.assertEqual(len(calls), 20)
                self.assertEqual(calls.count(("f1", 8)), 1)
                old_fit = sha(out / "fit.json")
                runner.main(cfg)
                self.assertEqual(len(calls), 22)  # Baseline parity only on resume.
                self.assertEqual(old_fit, sha(out / "fit.json"))
                self.assertEqual(read(out / "report.json")["scenes"], 1)


if __name__ == "__main__":
    unittest.main()
