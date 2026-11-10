from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import audit_selector_teacher_veto as audit
from test_selector_bundle import fixture

veto = audit.veto


def prepared(n=1):
    examples = []
    for i in range(n):
        sample, outcomes, vocab, meta = fixture("t" + str(i), "s" + str(i // 2))
        sample["head_drivable_area_compliance"][:] = np.log(.9 / .1)
        sample["head_drivable_area_compliance"][3] = np.log(.76 / .24)
        sample["head_drivable_area_compliance"][4] = np.log(.95 / .05)
        sample["head_drivable_area_compliance"][5] = np.log(.85 / .15)
        sample["head_time_to_collision_within_bound"][[5, 6]] = -1.
        for outcome in outcomes.values():
            outcome["raw"].update(pdm_score=.8, **{m: 1. for m in audit.METRICS})
        outcomes["3"]["raw"].update(pdm_score=0., drivable_area_compliance=0.)
        outcomes["4"]["raw"].update(pdm_score=.65, lane_keeping=.5)
        outcomes["5"]["raw"]["pdm_score"] = .9
        outcomes["6"]["raw"]["pdm_score"] = .91
        for outcome in outcomes.values():
            outcome["filtered"] = outcome["raw"].copy()
        # A stored filtered component changed without reconstructing stored PDM.
        outcomes["4"]["filtered"]["lane_keeping"] = 1.
        e = veto.bundles.example(sample, outcomes, vocab, meta)
        e["sampling_slice"] = "uniform"
        examples.append(e)
    gain = np.ones((n, 16)) * -.01
    gain[:, 0], gain[:, 3], gain[:, 4] = 0., .08, .06
    pred = dict(gain=gain, gain_lower=gain.copy(), risk=np.ones((n, 16, 5)) * .001,
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
    return examples, vocab, pred, positions, identity, pd.DataFrame(rows)


def saved_run(base, n=3):
    examples, vocab, pred, pos, identity, frame = prepared(n)
    bundle, night, tags, root = [base / name for name in ("bundle", "night", "tags", "veto")]
    veto.write(tags / "label_selection.json", dict(records=[]))
    veto.bundles.build(bundle, examples, vocab, {}, {"closed"})
    night.mkdir()
    np.savez_compressed(night / "mlp_oof_ensemble.npz", **pred, tokens=np.array(list(pos)))
    frame.to_csv(night / "oof_per_scene.csv", index=False)
    identity.update(sources={}, inputs={str(bundle / name): veto.sha(bundle / name) for name in ("identity.json", "bundle_manifest.json")})
    identity["inputs"][str(tags / "label_selection.json")] = veto.sha(tags / "label_selection.json")
    veto.write(night / "identity.json", identity)
    veto.write(night / "report.json", dict(status="complete", fresh_holdout_processed=0,
        identity_sha256=veto.sha(night / "identity.json"),
        artifacts={name: veto.sha(night / name) for name in ("oof_per_scene.csv", "mlp_oof_ensemble.npz")},
        selection=dict(minimum_interventions=10, minimum_intervention_sessions=3, minimum_improved_sessions=3,
                       maximum_single_quality_loss_pp=5.)))
    argv = ["veto", "--night", str(night), "--bundle", str(bundle), "--out", str(root)]
    with mock.patch("sys.argv", argv):
        veto.main()
    return bundle, night, root


class VetoAuditTests(unittest.TestCase):
    def test_overlap_is_disjoint_and_removing_one_constraint_requires_all_others(self):
        e = prepared()[0][0]
        a = audit.scene_arrays(e)
        candidates = pd.DataFrame(audit.constraint_rows(a))
        opportunity = pd.DataFrame(audit.opportunities(a))
        transitions = pd.DataFrame([dict(session_id="s0", changed=False, pre_minus_post_gain_pp=0.)])
        report = audit.aggregate(candidates, opportunity, pd.DataFrame(), transitions)
        dac, ttc = [report["constraints"][m] for m in ("drivable_area_compliance", "time_to_collision_within_bound")]
        self.assertEqual(dac["blocked_actions"], 2)
        self.assertEqual(dac["detected_component_degradations"], 1)
        self.assertEqual(dac["blocked_useful_actions"], 1)
        self.assertEqual(ttc["blocked_useful_actions"], 2)
        self.assertEqual(ttc["component_degradation_actions"], 0)
        self.assertEqual(dac["recovered_useful_scenes_if_only_this_constraint_removed"], 0)
        self.assertEqual(ttc["recovered_useful_scenes_if_only_this_constraint_removed"], 1)
        self.assertEqual(sum(r["actions"] for r in report["exact_veto_patterns"]), 15)
        self.assertEqual(report["oracle_opportunity"]["all5"]["scenes_with_useful_alternative"], 0)

    def test_harm_zero_loss_and_filter_difference_do_not_trigger_score_reconstruction(self):
        examples, _, pred, _, _, _ = prepared()
        a = audit.scene_arrays(examples[0])
        row = audit.intervention(a, chosen=4, original=3, post=0, pred=pred, position=0)
        self.assertFalse(row["raw_harm"])
        self.assertAlmostEqual(row["raw_gain_pp"], -15.)
        self.assertEqual(row["raw_dropped_components"], "lane_keeping")
        self.assertTrue(row["raw_filtered_components_differ"])
        self.assertAlmostEqual(row["gain_overestimate_pp"], 21.)
        self.assertFalse(row["loss_without_component_drop"])
        examples[0]["label_raw"][4, 1 + audit.METRICS.index("lane_keeping")] = 1.
        row = audit.intervention(audit.scene_arrays(examples[0]), 4, 3, 0, pred, 0)
        self.assertTrue(row["loss_without_component_drop"])
        self.assertAlmostEqual(row["raw_score"], .65)

    def test_frozen_choices_and_outcomes_replayed_before_diagnosis(self):
        examples, _, pred, pos, identity, frame = prepared()
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp)
            rows, opportunity = veto.evaluate(examples, dict(mlp=(pred, pos)), identity, frame, out)
            frozen = veto.read(out / "decisions.json")
            args = (examples, pred, pos, identity, pd.DataFrame(rows), frozen, pd.DataFrame(opportunity))
            candidates, _, primary, transitions = audit.analyze(*args)
            self.assertEqual(len(candidates), 15)
            self.assertEqual(len(primary), 1)
            self.assertTrue(primary.iloc[0].replacement_after_veto)
            self.assertAlmostEqual(transitions.iloc[0].pre_minus_post_gain_pp, -15.)
            changed = pd.DataFrame(rows)
            changed.loc[changed.method == veto.CANDIDATE, "raw_gain_pp"] = 999.
            with self.assertRaisesRegex(ValueError, "outcome replay"):
                audit.analyze(examples, pred, pos, identity, changed, frozen, pd.DataFrame(opportunity))
            corrupt = copy.deepcopy(frozen)
            next(r for r in corrupt if r["method"] == veto.CANDIDATE)["chosen_local"] = 6
            with self.assertRaisesRegex(ValueError, "decision replay"):
                audit.analyze(examples, pred, pos, identity, pd.DataFrame(rows), corrupt, pd.DataFrame(opportunity))

    def test_macro_comparison_gives_sessions_equal_weight(self):
        examples, _, _, _, _, _ = prepared()
        candidates = pd.DataFrame(audit.constraint_rows(audit.scene_arrays(examples[0])))
        opportunity = pd.DataFrame(audit.opportunities(audit.scene_arrays(examples[0])))
        transitions = pd.DataFrame([dict(session_id=s, changed=True, pre_minus_post_gain_pp=g)
                                    for s, g in (("s0", -10.), ("s0", -10.), ("s1", 20.))])
        report = audit.aggregate(candidates, opportunity, pd.DataFrame(), transitions)
        self.assertEqual(report["pre_vs_post"]["gain_micro_pp"], 0.)
        self.assertEqual(report["pre_vs_post"]["gain_macro_pp"], 5.)

    def test_holdout_and_invalid_top16_are_rejected(self):
        e = prepared()[0][0]
        e["metadata"]["role"] = "test"
        with self.assertRaisesRegex(ValueError, "Training scenes"):
            audit.scene_arrays(e)
        e["metadata"]["role"] = "train"
        e["label_indices"][[1, 2]] = e["label_indices"][[2, 1]]
        with self.assertRaisesRegex(ValueError, "alignment"):
            audit.scene_arrays(e)

    def test_full_cli_preserves_inputs_resumes_and_detects_tampering(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp, redirect_stdout(io.StringIO()):
            base = Path(tmp).resolve()
            bundle, night, root = saved_run(base)
            parents = (bundle, night, root)
            before = {str(p): veto.sha(p) for parent in parents for p in parent.rglob("*") if p.is_file()}
            out = base / "audit"
            argv = ["audit", "--veto", str(root), "--night", str(night), "--bundle", str(bundle), "--out", str(out)]
            with mock.patch("sys.argv", argv):
                audit.main()
                digest = veto.sha(out / "report.json")
                audit.main()
            self.assertEqual(digest, veto.sha(out / "report.json"))
            report = veto.read(out / "report.json")
            self.assertEqual((report["scenes"], report["sessions"], report["candidate_actions"]), (3, 2, 45))
            self.assertEqual(report["primary_interventions"]["losses"], 3)
            self.assertEqual(report["fresh_holdout_processed"], 0)
            self.assertEqual(report["new_training_runs"], 0)
            self.assertEqual(before, {str(p): veto.sha(p) for parent in parents for p in parent.rglob("*") if p.is_file()})
            path = out / "primary_interventions.csv"
            path.write_bytes(path.read_bytes() + b"tampered")
            with mock.patch("sys.argv", argv), self.assertRaisesRegex(RuntimeError, "Audit artifact changed"):
                audit.main()
            path = root / "decisions.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(RuntimeError, "Frozen veto artifact changed"):
                audit.verify_veto(root, night, bundle)


if __name__ == "__main__":
    unittest.main()
