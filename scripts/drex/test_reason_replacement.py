import csv
import gzip
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import reason_replacement as reason
from test_direct255 import fixture
from transfer import read, sha, write


def votes(reason_choice='TTC', binary_choice='REPLACE', reversed_reason=None):
    return {'reason': {'primary': reason_choice, 'reversed': reversed_reason or reason_choice},
            'binary': {'primary': binary_choice, 'reversed': binary_choice},
            'margin': {kind: dict.fromkeys(reason.ORDERS, 1.) for kind in reason.KINDS}}


class PairTests(unittest.TestCase):
    def test_pair_contains_rank_one_and_one_top16_alt_and_prefers_guard(self):
        arrays, vocab = fixture()
        arrays['head_time_to_collision_within_bound'][8] = 1.
        arrays['head_ego_progress'][8] = -.02  # Tolerated progress difference.
        arrays['head_time_to_collision_within_bound'][9] = 10.
        arrays['head_no_at_fault_collisions'][9] = -.01  # Does not pass guard.
        arrays['head_time_to_collision_within_bound'][20] = 10.  # Outside top16.
        pair = reason.make_pair(arrays, vocab)
        self.assertEqual(pair['reference']['local'], 0)
        self.assertEqual(pair['challenger']['local'], 8)
        self.assertTrue(pair['numeric_guard'])
        self.assertTrue(pair['numeric_replace'])
        self.assertTrue(reason.policies(pair, votes())['drex_reason'])
        self.assertFalse(reason.policies(pair, votes(reversed_reason='PROGRESS'))['drex_reason'])
        self.assertFalse(reason.policies(pair, votes(reason_choice='KEEP'))['drex_reason'])
        pair['numeric_guard'] = False
        self.assertFalse(reason.policies(pair, votes())['drex_reason'])
        self.assertFalse(reason.policies(pair, votes())['drex_binary'])

    def test_controls_share_state_and_reversal_changes_only_option_order(self):
        pair = reason.make_pair(*fixture())
        primary, keys, h1 = reason.payload(pair, 'reason', 'primary')
        reversed_, reversed_keys, h2 = reason.payload(pair, 'reason', 'reversed')
        binary, _, _ = reason.payload(pair, 'binary', 'primary')
        self.assertEqual(primary['state'], reversed_['state'])
        self.assertEqual(primary['state'], binary['state'])
        self.assertEqual(keys, reversed_keys[::-1])
        self.assertNotEqual(h1, h2)
        self.assertNotIn('numeric_guard', primary['state'])
        self.assertNotIn('images', primary)
        self.assertEqual(reason.top_option([0.5, 0.5], ['REPLACE', 'KEEP']), 'KEEP')
        self.assertEqual(reason.top_option([1., 0.], ['REPLACE', 'KEEP']), 'REPLACE')

    def test_outcome_checks_named_reason_and_progress_separately(self):
        reference = dict.fromkeys(reason.METRICS, .5)
        reference.update(pdm_score=.8, route_progress_m=10.)
        challenger = dict(reference, pdm_score=.81, ego_progress=.6, route_progress_m=9.4,
                          time_to_collision_within_bound=.7)
        _, harm, violation, supported, helpful = reason.outcome(reference, challenger)
        self.assertEqual(harm, [])
        self.assertTrue(violation)
        self.assertFalse(helpful)
        self.assertEqual(supported, {'TTC'})
        challenger['route_progress_m'] = 10.5
        _, _, violation, supported, helpful = reason.outcome(reference, challenger)
        self.assertFalse(violation)
        self.assertTrue(helpful)
        self.assertEqual(supported, {'TTC', 'PROGRESS'})


class PipelineTests(unittest.TestCase):
    def test_pilot_is_reused_by_full_and_committed_votes_are_reassembled(self):
        pair = reason.make_pair(*fixture())
        tokens = ['pilot', 'second']
        plan = dict(records=[dict(token=t) for t in tokens], pairs=dict.fromkeys(tokens, pair),
                    pilot=tokens[:1], model_plan=dict(files=[]))
        torch = ModuleType('torch')
        torch.bfloat16 = 'bf16'
        sdk = ModuleType('drex_max.model')
        calls = []

        class Model:
            readout = SimpleNamespace(weight=SimpleNamespace(dtype='bf16'))

            def __init__(self, *args, **kwargs):
                pass

            def prepare(self, rows, max_length):
                return SimpleNamespace(inputs={'input_ids': np.empty((1, 25))})

            def predict(self, rows, **kwargs):
                keys = list(rows[0]['question']['criteria'])
                calls.append(keys)
                return [[float(key in ('TTC', 'REPLACE')) for key in keys]]

        sdk.DecisionModel = Model
        env = dict.fromkeys(('python', 'torch', 'cuda_build', 'transformers', 'gpu', 'vram_bytes'), 'fixture')
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {'torch': torch, 'drex_max.model': sdk}), \
                patch.object(reason, 'load_plan', return_value=plan), patch.object(reason, 'gpu_binding', return_value=0), \
                patch.object(reason, 'environment', return_value=env), patch('builtins.print'):
            root = Path(folder)
            out = root / 'run'
            write(out / 'plan.json', plan)
            write(root / 'models/drex-max-v1/decision_config.json', {'format_version': 1})
            write(root / 'requirements.lock.txt', {})
            with self.assertRaises(FileNotFoundError):
                reason.worker(root, out, 1, 'full')
            reason.worker(root, out, 1, 'pilot')
            reason.collect(root, out, 1, 'pilot')
            reason.worker(root, out, 1, 'full')
            reason.collect(root, out, 1, 'full')
            self.assertEqual(len(calls), 8)
            self.assertEqual(read(out / 'decisions.json'), {t: votes() for t in tokens})
            self.assertEqual(read(out / 'report.json')['requests'], 8)

    def test_evaluation_checks_commits_and_joins_physical_identity(self):
        pair = reason.make_pair(*fixture())
        metrics = ('pdm_score', *reason.METRICS, *reason.CONTINUOUS)
        reference = dict.fromkeys(metrics, .5)
        reference['route_progress_m'] = 10.
        challenger = dict(reference, pdm_score=.51, time_to_collision_within_bound=.75)
        tokens = ['a', 'b']
        records = [dict(token=t, session_id=t, role='train') for t in tokens]
        with tempfile.TemporaryDirectory() as folder, patch('builtins.print'):
            astra = Path(folder) / 'astra'
            out = Path(folder) / 'run'
            write(out / 'plan.json', dict(config=reason.CONFIG, records=records, pairs=dict.fromkeys(tokens, pair), manifest_sha256='fixture'))
            write(out / 'decisions.json', {t: votes() for t in tokens})
            report = dict(status='choices_complete', scenes=2, requests=8, plan_sha256=sha(out / 'plan.json'),
                          decisions_sha256=sha(out / 'decisions.json'))
            write(out / 'report.json', report)
            write(astra / 'request.json', dict(manifest_sha256='fixture',
                  candidates={t: dict(ranked=list(range(256)), vocab=list(range(256))) for t in tokens}))
            write(astra / 'report.json', dict(status='complete', holdout_processed=0, scenes=2,
                  completed_candidates=512, anchors_parity_passed=2))
            with gzip.open(astra / 'per_candidate.csv.gz', 'wt', encoding='utf-8', newline='') as stream:
                columns = ['token', 'session_id', 'candidate_local', 'candidate_vocab', 'teacher_rank', *metrics]
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for t in tokens:
                    for i in range(256):
                        writer.writerow(dict(token=t, session_id=t, candidate_local=i, candidate_vocab=i, teacher_rank=i+1,
                                             **(challenger if i == 1 else reference)))
            summary = reason.evaluate(out, astra)
            self.assertEqual(summary['helpful_pairs'], 2)
            self.assertEqual(summary['methods']['numeric']['interventions'], 0)
            self.assertEqual(summary['matched_coverage']['interventions'], 0)
            self.assertEqual(summary['measured_reasons']['TTC']['supported'], 2)
            self.assertAlmostEqual(summary['paired_pdm_differences_pp']['reason_minus_numeric']['macro_mean'], 1.)
            write(out / 'decisions.json', {})
            with patch.object(reason, 'measured_pairs') as measured, self.assertRaisesRegex(ValueError, 'committed'):
                reason.evaluate(out, astra)
            measured.assert_not_called()


class SlurmTests(unittest.TestCase):
    def test_night_submits_one_unpinned_allocation_and_pilot_precedes_full(self):
        import test_runtime

        shell = test_runtime.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        shell.setUp()
        self.addCleanup(shell.doCleanups)
        launcher = Path(reason.__file__).with_name('run_reason_replacement.sh').resolve()
        shell.script = launcher
        shell.stub('sbatch', 'printf "%s\\n" "$@" > "$DREX_TEST_LOG"\n')
        result = shell.run_stage('submit-night', '4')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shell.log.read_text().splitlines()
        self.assertIn('--nodes=1', args)
        self.assertIn('--gres=gpu:4', args)
        self.assertNotIn('--nodelist=cn29', args)
        self.assertIn('coordinator-night', args)

        shell.fake_setup()
        shell.stub('srun', 'printf "%s\\n" "srun $*" >> "$DREX_TEST_CALLS"\n')
        result = shell.run_stage('--inside', 'coordinator-night', shell.posix(launcher.parents[2]),
                                 shell.posix(shell.envroot), '4', 'reason_fixture')
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = shell.calls.read_text()
        self.assertIn('--gpus-per-task=1', calls)
        self.assertLess(calls.index('worker-pilot'), calls.index('--phase pilot'))
        self.assertLess(calls.index('--phase pilot'), calls.index('worker-full'))


if __name__ == '__main__':
    unittest.main()
