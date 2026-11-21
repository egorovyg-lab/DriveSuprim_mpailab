import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import pointwise_value as value
from consequence_model import folds
from transfer import read


class PointwiseValueTests(unittest.TestCase):
    def test_numeric_and_binary_models_fit_a_fold(self):
        rng = np.random.default_rng(43)
        n, k = 48, 3
        sessions = np.asarray([f'session_{i//4}' for i in range(n)])
        x = rng.normal(size=(n, k, 5))
        gains = rng.normal(size=(n, k, 5))
        events = (rng.random((n, k, 3)) < .15).astype(float)
        test = folds(sessions) == 0
        best, confidence, threshold = value.fit_fold(x, gains, events, sessions, ~test, test)
        self.assertEqual(best.shape, (int(test.sum()),))
        self.assertTrue(np.all((0 <= best) & (best < k)))
        self.assertFalse(np.isnan(confidence).any())
        self.assertGreaterEqual(threshold, 0.)

    def test_eligibility_and_exact_matched_coverage_ignore_ineligible_rows(self):
        prediction = np.ones((20, 2, 5))
        probabilities = np.full((20, 2, 3), .05)
        probabilities[0, 1] = (.01, .01, .9)
        probabilities[0, 0] = (.2, .01, .99)
        prediction[1, :, 3] = -1.
        best, score = value.eligible_choices(prediction, probabilities)
        self.assertEqual(best[0], 1)
        self.assertEqual(score[1], -np.inf)
        sparse = np.full(20, -np.inf)
        sparse[12] = .2
        dense_mask, sparse_mask = value.matched([score, sparse], .1)
        self.assertEqual(np.flatnonzero(dense_mask).tolist(), [0])
        self.assertEqual(np.flatnonzero(sparse_mask).tolist(), [12])
        threshold = value.train_threshold(np.arange(20) / 20.)
        self.assertEqual(int((np.arange(20) / 20. > threshold).sum()), 2)

    def test_pilot_then_full_reuses_fold_and_compares_at_same_budget(self):
        n, k = 48, 15
        sessions = [f'session_{i//4}' for i in range(n)]
        records = [dict(token=str(i), session_id=s) for i, s in enumerate(sessions)]
        features = dict(fold=folds(sessions))
        arms = {a: np.full((n, k, 1), i) for i, a in enumerate(value.ARMS)}
        gains = np.zeros((n, k, 5))
        gains[:, :, 3] = np.arange(k)
        events = np.zeros((n, k, 3))

        def fake_fit(x, g, e, s, train, test):
            self.assertFalse(set(s[train]) & set(s[test]))
            count = int(test.sum())
            return np.full(count, x[0, 0, 0], dtype=int), np.linspace(0.5, 0.9, count), .8

        with tempfile.TemporaryDirectory() as d, patch.object(value, 'inputs',
                return_value=(records, features, arms, gains, events, {'test': True})), \
                patch.object(value, 'fit_fold', side_effect=fake_fit) as fit:
            out = Path(d)
            with self.assertRaisesRegex(ValueError, 'pilot'):
                value.run(None, None, out, 'full')
            value.run(None, None, out, 'pilot')
            self.assertEqual(fit.call_count, 3)
            value.run(None, None, out, 'full')
            self.assertEqual(fit.call_count, 12)
            report = read(out / 'report.json')
            self.assertEqual(report['scenes'], n)
            m = report['metrics']['matched_10pct']
            self.assertEqual([m[a]['interventions'] for a in value.ARMS], [4, 4, 4])
            self.assertGreater(report['pdm_gain_pp_vs_numeric']['matched_10pct']['ds_drex_road']['macro_mean'], 0)
            with (out / 'fold_0.npz').open('ab') as f:
                f.write(b'damage')
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                value.run(None, None, out, 'full')


if __name__ == '__main__':
    unittest.main()
