from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import audit_drivemem_risk_coverage as risk
from navsim.planning.script import run_drivemem_dac_window as dac
from test_corridor_projection import fixture


class DacTests(unittest.TestCase):
    def test_unknown_replacement_corner_rejects_candidate(self):
        packet = dict(baseline_position=0, selection_scores=np.array([3., 2., 1.]))
        inside = np.ones((3, 11, 4), dtype=bool)
        inside[0, 2, 1] = False
        covered = np.ones_like(inside)
        covered[1, 3, 3] = False
        self.assertEqual(dac.choose(packet, inside, covered)[:2], (1, 2))
        covered[0, 2, 1] = False
        self.assertEqual(dac.choose(packet, inside, covered)[:2], (1, 0))

    def test_complete_audit_reads_frozen_geometry_and_keeps_review_separate(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            base = Path(folder)
            root, _ = fixture(base)
            out = base/"audit"
            original = {p: dac.io.sha(p) for p in root.rglob("*") if p.is_file()}
            report = dac.run(root, out)
            self.assertEqual(report["scenes"], 3)
            self.assertEqual(report["failure_times"]["in_window"], 3)
            self.assertEqual(report["map_reference"]["safety_harm"], 3)
            points = dac.io.load_npz(out/"window_points.npz")
            np.testing.assert_allclose(points["times_s"], np.arange(10, 21)*.1)
            self.assertEqual(points["inside_road"].shape, (3, 16, 11, 4))
            self.assertEqual(report, dac.run(root, out))
            template = out/"review_template.csv"
            self.assertFalse(dac.review(out, template)["proceed_to_verifier"])
            self.assertEqual(original, {p: dac.io.sha(p) for p in original})
            (out/"per_scene.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Committed artifact changed"):
                dac.review(out, template)


class CoverageTests(unittest.TestCase):
    def test_distances_for_two_actions_and_c2_reversion_to_ds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entries, decisions, raws = {}, {}, {}
            for outer in range(4):
                token = "t" + str(outer)
                fold = root/"folds"/str(outer)
                risk.source.write(fold/"commit.json", dict(tokens=[token]))
                risk.source.write(fold/"donors.json", dict(h=2.))
                traces = np.ones((1, 64, 1))
                traces[0, 1, 0] = 4.
                risk.source.save(fold/"donor_traces.npz", nearest_distance=traces)
                entries[token] = dict(baseline=0, session_id=token, ids=np.arange(64), sampling_slice="uniform")
                candidate = 0 if outer % 2 else 2
                gains = np.zeros((3, 64))
                gains[2, 1], gains[2, 2] = -.1, .1
                decisions[token] = (outer, [0, 1, candidate], gains)
                raws[token] = np.ones((64, 9))
                raws[token][1, 0] = .5
            computed = risk.proposals(root, entries, decisions, raws)
            self.assertEqual(computed[computed.reference == "C1"].distance.tolist(), [2.]*4)
            self.assertEqual(computed[computed.reference == "DS"].distance.tolist(), [.5]*2)
            self.assertEqual(computed[computed.reference == "C1"].gain_pp.tolist(), [50.]*4)

    def test_matched_counts_and_full_cohort_denominator(self):
        cohort = pd.DataFrame(dict(token=list("abcdefgh"), session_id=["a"]*6+["b"]*2))
        proposed = pd.DataFrame([
            dict(reference=ref, token=token, session_id=session, distance=distance, expected_gain_pp=expected,
                 gain_pp=gain, harm=gain < 0, pdm_loss=gain < 0, proposal_rank=rank, sampling_slice="uniform")
            for ref in ("C1", "DS")
            for token, session, distance, expected, gain, rank in
            (("a", "a", .1, .1, -100., 30), ("b", "a", .6, 1., 5., 4),
             ("g", "b", .7, 2., 40., 5), ("h", "b", .9, .2, 15., 40))
        ])
        rows, contrasts, bins = risk.compare(proposed, cohort, bootstrap_reps=80)
        first = {r["rule"]: r for r in rows if r["reference"] == "DS" and r["interventions"] == 1}
        self.assertEqual(first["distance"]["harmful_per_intervention"], 1)
        self.assertEqual(first["distance"]["harmful_per_all"], 1/8)
        self.assertEqual(first["predicted_utility"]["gain_macro_pp"], 10)
        full = [r for r in rows if r["reference"] == "DS" and r["interventions"] == 4]
        self.assertEqual(len({r["gain_macro_pp"] for r in full}), 1)
        pair = next(c for c in contrasts if c["reference"] == "DS" and c["interventions"] == 4)
        self.assertEqual(pair["gain_ci95_sessions"], [0, 0])
        self.assertTrue(any(b["distance_over_h"] == "<=0.5" and b["safety_harm_rate"] == 1 for b in bins))


if __name__ == "__main__":
    unittest.main()
