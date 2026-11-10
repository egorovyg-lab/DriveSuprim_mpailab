import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import audit_selector_component_heads as audit
from test_selector_bundle import fixture

common = audit.common


def prepared(n=3):
    examples, teachers = [], []
    for i in range(n):
        sample, outcomes, vocab, meta = fixture("t" + str(i), "s" + str(i))
        probabilities = np.ones((256, 8), dtype=np.float32) * .95
        probabilities[:, 4] = .7
        probabilities[3, 1], probabilities[3, 4] = .8, .65
        for j, m in enumerate(audit.METRICS):
            sample["head_" + m] = np.log(probabilities[:, j] / (1 - probabilities[:, j]))
        for j, outcome in outcomes.items():
            c = np.ones(8)
            c[4] = .7
            if int(j) == 3:
                c[1], c[4] = 0., .65
            raw = dict(pdm_score=float(audit.pdm(c)), **dict(zip(audit.METRICS, c.tolist())))
            outcome["raw"], outcome["filtered"] = raw, raw.copy()
        examples.append(common.bundles.example(sample, outcomes, vocab, meta))
        teachers.append(probabilities[:16].copy())
    teacher = np.stack(teachers)
    student = teacher.copy()
    student[:, :, 1], student[:, :, 6] = .7, .2
    student[:, 3, 1], student[:, 3, 6] = .99, .99
    replicas = np.stack([student.copy(), student.copy()])
    replicas[0, :, 3, 4] -= .002
    replicas[1, :, 3, 4] += .002
    baseline = np.zeros(n, dtype=np.int64)
    scores = audit.pdm(replicas)
    cached_gain = scores - scores[:, :, :1]
    rows = []
    for name, values in (("student", replicas), ("teacher", audit.substitute(replicas, teacher, tuple(range(8))))):
        combined = audit.combined(values, baseline, cached_gain if name == "student" else None)
        for i, e in enumerate(examples):
            sample = dict(e["sample"], token=e["metadata"]["token"], session_id=e["metadata"]["session_id"])
            outcomes = {str(int(local)): {v: dict(zip(common.bundles.LABEL_KEYS, e["label_" + v][j].tolist()))
                        for v in ("raw", "filtered")} for j, local in enumerate(e["label_indices"])}
            for policy in audit.POLICIES:
                j = audit.choose(combined["gain"][i], combined["gain_lower"][i], e["sample"]["candidate_ids"][:16], 0, policy)
                rows.append(dict(method=("mlp" if name == "student" else "teacher_formula") + "/ensemble/" + policy,
                    predicted_gain_pp=100 * float(combined["gain"][i, j]), gain_lower_pp=100 * float(combined["gain_lower"][i, j]),
                    **common.record(sample, outcomes, int(e["label_indices"][j]))))
    return examples, vocab, replicas, cached_gain, teacher, baseline, pd.DataFrame(rows)


def make_source(base):
    examples, vocab, replicas, gains, teacher, baseline, frame = prepared()
    bundle, source = base / "bundle", base / "source"
    common.bundles.build(bundle, examples, vocab, {}, {"closed"})
    source.mkdir()
    identity = dict(variants=["mlp"], seeds=[11, 12], sources={},
        inputs={str(bundle / name): common.sha(bundle / name) for name in ("identity.json", "bundle_manifest.json")})
    common.write(source / "identity.json", identity)
    for s, seed in enumerate(identity["seeds"]):
        for i, e in enumerate(examples):
            sid, token = e["metadata"]["session_id"], e["metadata"]["token"]
            path = source / "cv" / f"mlp_s{seed}_{sid}"
            path.parent.mkdir(exist_ok=True)
            partition = dict(train_sessions=[x["metadata"]["session_id"] for j, x in enumerate(examples) if j != i],
                train_tokens=[x["metadata"]["token"] for j, x in enumerate(examples) if j != i], valid_session=sid, valid_tokens=[token])
            binding = dict(identity_sha256=common.sha(source / "identity.json"), variant="mlp", seed=seed, partition=partition)
            np.savez_compressed(path.with_suffix(".npz"), components=replicas[s, i:i+1],
                score=audit.pdm(replicas[s, i:i+1]), gain=gains[s, i:i+1])
            common.write(path.with_suffix(".json"), dict(binding=binding, sha256=common.sha(path.with_suffix(".npz"))))
    np.savez_compressed(source / "mlp_oof_ensemble.npz", **audit.combined(replicas, baseline, gains),
                        tokens=np.array([e["metadata"]["token"] for e in examples]))
    frame.to_csv(source / "oof_per_scene.csv", index=False)
    artifacts = {str(p.relative_to(source)): common.sha(p) for p in source.rglob("*") if p.is_file() and p.name != "identity.json"}
    report = dict(status="complete", fresh_holdout_processed=0, identity_sha256=common.sha(source / "identity.json"),
        formula_audit=dict(status="passed", mismatches=0), counts=dict(scenes=len(examples), sessions=len(examples)),
        primary="baseline", artifacts=artifacts)
    common.write(source / "report.json", report)
    return bundle, source, examples, teacher


class HeadAuditTests(unittest.TestCase):
    def test_substitution_updates_baseline_all_actions_and_all_seeds(self):
        _, _, replicas, _, teacher, _, _ = prepared(1)
        changed = audit.substitute(replicas, teacher, (1, 6))
        for seed in range(2):
            np.testing.assert_array_equal(changed[seed, ..., 1], teacher[..., 1])
            np.testing.assert_array_equal(changed[seed, ..., 6], teacher[..., 6])
        np.testing.assert_array_equal(changed[..., 4], replicas[..., 4])
        self.assertNotEqual(float(changed[0, 0, 0, 6]), float(replicas[0, 0, 0, 6]))
        self.assertNotEqual(float(changed[0, 0, 3, 6]), float(replicas[0, 0, 3, 6]))

    def test_seed_disagreement_is_recomputed_after_head_replacement(self):
        _, _, replicas, _, teacher, baseline, _ = prepared(1)
        replicas[0, 0, 3, 4], replicas[1, 0, 3, 4] = .4, .9
        original = audit.combined(replicas, baseline)
        restored = audit.combined(audit.substitute(replicas, teacher, (4,)), baseline)
        self.assertGreater(original["gain"][0, 3] - original["gain_lower"][0, 3], .01)
        np.testing.assert_array_equal(restored["gain"], restored["gain_lower"])

    def test_joint_restore_removes_synthetic_invented_baseline_deficit(self):
        _, _, replicas, gains, teacher, baseline, _ = prepared(1)
        student = audit.combined(replicas, baseline, gains)
        restored = audit.combined(audit.substitute(replicas, teacher, (1, 6)), baseline)
        self.assertEqual(audit.choose(student["gain"][0], student["gain_lower"][0], np.arange(16), 0, "gain_m020"), 3)
        self.assertEqual(audit.choose(restored["gain"][0], restored["gain_lower"][0], np.arange(16), 0, "gain_m020"), 0)

    def test_choices_do_not_consume_outcomes_and_are_frozen_before_evaluation(self):
        examples, _, replicas, gains, teacher, baseline, frame = prepared(1)
        actual = copy.deepcopy(examples)
        class NoLabels:
            def __getitem__(self, key):
                raise AssertionError("Measured labels consumed while choosing")
        examples[0]["label_raw"], examples[0]["label_filtered"] = NoLabels(), NoLabels()
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp)
            decisions = audit.freeze_decisions(examples, replicas, gains, teacher, baseline, frame, out)
            original = common.record
            def guarded(*args):
                self.assertEqual(len(common.read(out / "decisions.json")), len(audit.SUBSTITUTIONS) * len(audit.POLICIES))
                return original(*args)
            with mock.patch.object(common, "record", side_effect=guarded):
                rows = audit.evaluate(actual, decisions, frame)
            self.assertEqual(len(rows), len(audit.SUBSTITUTIONS) * len(audit.POLICIES))
            bad = frame.copy()
            bad.loc[0, "candidate_id"] = 255
            with self.assertRaisesRegex(ValueError, "decision replay"):
                audit.freeze_decisions(actual, replicas, gains, teacher, baseline, bad, out)

    def test_replicas_require_whole_session_partitions_and_exact_ensemble(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp, redirect_stdout(io.StringIO()):
            bundle, root, examples, _ = make_source(Path(tmp).resolve())
            identity, report = audit.verify_source(root, bundle)
            replicas, gains, baseline = audit.load_replicas(root, identity, report, examples)
            self.assertEqual(replicas.shape, (2, 3, 16, 8))
            path = next((root / "cv").glob("*.json"))
            meta = common.read(path)
            meta["binding"]["partition"]["train_tokens"].append(meta["binding"]["partition"]["valid_tokens"][0])
            common.write(path, meta)
            with self.assertRaisesRegex(ValueError, "fold binding"):
                audit.load_replicas(root, identity, report, examples)

    def test_full_audit_resumes_preserves_sources_and_never_selects_primary(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp, redirect_stdout(io.StringIO()):
            base = Path(tmp).resolve()
            bundle, source, _, teacher = make_source(base)
            before = {str(p): common.sha(p) for root in (bundle, source) for p in root.rglob("*") if p.is_file()}
            out = base / "audit"
            argv = ["audit", "--source", str(source), "--bundle", str(bundle), "--out", str(out)]
            with mock.patch("sys.argv", argv), mock.patch.object(audit, "teacher_probabilities", return_value=teacher):
                audit.main()
            report = common.read(out / "report.json")
            self.assertEqual(report["frozen_primary"], "baseline")
            self.assertFalse(report["selection_performed"])
            self.assertEqual(report["new_training_runs"], 0)
            self.assertEqual(report["fresh_holdout_processed"], 0)
            rows = {r["method"]: r for r in report["rows"]}
            self.assertEqual(rows["student/gain_m020"]["harm"], 3)
            self.assertEqual(rows["restore_DAC_LK/gain_m020"]["harm"], 0)
            self.assertEqual(rows["restore_DAC_LK/gain_m020"]["source_harm_scenes_no_longer_harm"], 3)
            self.assertEqual(rows["restore_DAC_LK/gain_m020"]["new_harm_scenes"], 0)
            digest = common.sha(out / "report.json")
            with mock.patch("sys.argv", argv), mock.patch.object(audit, "teacher_probabilities", side_effect=AssertionError("completed audit must resume")):
                audit.main()
            self.assertEqual(digest, common.sha(out / "report.json"))
            self.assertEqual(before, {str(p): common.sha(p) for root in (bundle, source) for p in root.rglob("*") if p.is_file()})
            path = out / "per_scene.csv"
            path.write_bytes(path.read_bytes() + b"changed")
            with mock.patch("sys.argv", argv), self.assertRaisesRegex(RuntimeError, "Head-audit artifact changed"):
                audit.main()
            path = next((source / "cv").glob("*.npz"))
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(RuntimeError, "Frozen component artifact changed"):
                audit.verify_source(source, bundle)


if __name__ == "__main__":
    unittest.main()
