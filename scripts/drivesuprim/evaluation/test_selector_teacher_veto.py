from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import selector_teacher_veto as veto
from test_selector_bundle import fixture


def prepared(n=2):
    examples = []
    for i in range(n):
        sample, outcomes, vocab, meta = fixture("t" + str(i), "s" + str(i))
        sample["head_drivable_area_compliance"][0] = np.log(.9 / .1)
        sample["head_drivable_area_compliance"][3] = np.log(.76 / .24)
        sample["head_drivable_area_compliance"][4] = np.log(.95 / .05)
        outcomes["3"]["raw"].update(pdm_score=0., drivable_area_compliance=0.)
        outcomes["4"]["raw"]["pdm_score"] = .9
        for value in outcomes.values():
            value["filtered"] = value["raw"].copy()
        e = veto.bundles.example(sample, outcomes, vocab, meta)
        e["sampling_slice"] = "uniform"
        examples.append(e)
    gain = np.ones((n, 16)) * -.01
    gain[:, 0], gain[:, 3], gain[:, 4] = 0., .08, .06
    prediction = dict(gain=gain, gain_lower=gain.copy(), risk=np.ones((n, 16, 5)) * .001,
                      risk_support_sessions=np.ones((n, 5), dtype=np.int64) * 3)
    positions = {e["metadata"]["token"]: i for i, e in enumerate(examples)}
    policies = {name: dict(margin=.02 if name.endswith("020") else .005,
                 risk_limit=.05 if name.startswith("risk") else None, uncertainty=True) for name in veto.HOSTS}
    identity = dict(policies=policies, config=dict(minimum_support_sessions=2), variants=["mlp"], seeds=[1, 2, 3])
    rows = []
    for e in examples:
        sample = dict(e["sample"], token=e["metadata"]["token"], session_id=e["metadata"]["session_id"])
        outcomes = {str(int(local)): {v: dict(zip(veto.bundles.LABEL_KEYS, e["label_" + v][j].tolist()))
                    for v in ("raw", "filtered")} for j, local in enumerate(e["label_indices"])}
        for host in veto.HOSTS:
            rows.append(dict(method="mlp/ensemble/" + host, ensemble="ensemble", **veto.record(sample, outcomes, 3)))
    return examples, vocab, prediction, positions, identity, pd.DataFrame(rows)


class TeacherVetoTests(unittest.TestCase):
    def test_post_veto_preserves_baseline_and_prefilter_finds_allowed_runner_up(self):
        examples, _, pred, _, identity, _ = prepared(1)
        e = examples[0]
        choices, _, masks = veto.decisions(e["sample"], e["label_indices"], 0, pred, 0, identity["policies"], 2)
        for decision in choices.values():
            self.assertEqual(decision, dict(original=3, dac_post=0, all5_post=0, all5_pre=4))
        self.assertFalse(masks["all5"][3])
        self.assertTrue(masks["all5"][0])

    def test_all_five_rule_checks_other_components_and_not_only_the_two_failure_ids(self):
        examples, _, pred, _, identity, _ = prepared(1)
        e = examples[0]
        e["sample"]["head_traffic_light_compliance"][4] = -1.
        choices, _, _ = veto.decisions(e["sample"], e["label_indices"], 0, pred, 0, identity["policies"], 2)
        self.assertEqual(choices["gain_m020"]["all5_pre"], 0)

    def test_tolerance_is_numerical_and_baseline_always_allowed(self):
        p = np.ones((3, 5)) * .8
        p[1, 1] -= .5e-6
        p[2, 1] -= 2e-6
        self.assertEqual(veto.allowed(p, 0, "all5").tolist(), [True, True, False])

    def test_choice_does_not_consume_outcomes_and_ties_preserve_baseline(self):
        examples, _, pred, _, identity, _ = prepared(1)
        e = examples[0]
        first = veto.decisions(e["sample"], e["label_indices"], 0, pred, 0, identity["policies"], 2)[0]
        e["sample"]["measured_outcomes"] = object()
        self.assertEqual(first, veto.decisions(e["sample"], e["label_indices"], 0, pred, 0, identity["policies"], 2)[0])
        policy = dict(margin=0., risk_limit=None, uncertainty=False)
        self.assertEqual(veto.select(0, np.arange(16), np.zeros(16), np.zeros(16), pred["risk"][0], np.ones(5), policy, 2), 0)

    def test_all_decisions_frozen_before_measured_outcomes_and_replay_is_verified(self):
        examples, _, pred, pos, identity, frame = prepared()
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp)
            original = veto.record
            def guarded(*args):
                frozen = veto.read(out / "decisions.json")
                self.assertEqual(len(frozen), len(examples) * len(veto.HOSTS) * len(veto.MODES))
                return original(*args)
            with mock.patch.object(veto, "record", side_effect=guarded):
                rows, opportunity = veto.evaluate(examples, dict(mlp=(pred, pos)), identity, frame, out)
            improved = [r for r in rows if r["method"] == veto.CANDIDATE]
            self.assertTrue(all(r["raw_gain_pp"] > 0 and not r["raw_harm"] for r in improved))
            self.assertEqual(len(opportunity), len(examples) * 3)
            frame.loc[0, "candidate_id"] = 255
            with self.assertRaisesRegex(ValueError, "Exact replay"):
                veto.evaluate(examples, dict(mlp=(pred, pos)), identity, frame, out)

    def test_complete_report_resumes_without_modifying_existing_inputs(self):
        examples, vocab, pred, _, identity, frame = prepared()
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            base = Path(tmp).resolve()
            bundle, night, addition, out = [base / n for n in ("bundle", "night", "addition", "out")]
            veto.write(addition / "label_selection.json", dict(records=[]))
            with redirect_stdout(io.StringIO()):
                veto.bundles.build(bundle, examples, vocab, {}, {"closed"})
            night.mkdir()
            pred["tokens"] = np.array([e["metadata"]["token"] for e in examples])
            np.savez_compressed(night / "mlp_oof_ensemble.npz", **pred)
            frame.to_csv(night / "oof_per_scene.csv", index=False)
            identity.update(sources={}, inputs={str(bundle / name): veto.sha(bundle / name) for name in ("identity.json", "bundle_manifest.json")})
            identity["inputs"][str(addition / "label_selection.json")] = veto.sha(addition / "label_selection.json")
            veto.write(night / "identity.json", identity)
            veto.write(night / "report.json", dict(status="complete", fresh_holdout_processed=0,
                identity_sha256=veto.sha(night / "identity.json"),
                artifacts={name: veto.sha(night / name) for name in ("oof_per_scene.csv", "mlp_oof_ensemble.npz")},
                selection=dict(minimum_interventions=1, minimum_intervention_sessions=1, minimum_improved_sessions=1,
                               maximum_single_quality_loss_pp=5.)))
            before = {str(p): veto.sha(p) for parent in (bundle, night) for p in parent.rglob("*") if p.is_file()}
            argv = ["veto", "--night", str(night), "--bundle", str(bundle), "--out", str(out)]
            with mock.patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                veto.main()
                first = veto.sha(out / "decisions.json")
                veto.main()
            self.assertEqual(first, veto.sha(out / "decisions.json"))
            self.assertEqual(veto.read(out / "report.json")["primary"], veto.CANDIDATE)
            self.assertEqual(before, {str(p): veto.sha(p) for parent in (bundle, night) for p in parent.rglob("*") if p.is_file()})
            path = night / "mlp_oof_ensemble.npz"
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                veto.verify_night(night, bundle)


if __name__ == "__main__":
    unittest.main()
