"""Checks for a matched pointwise image experiment and resumable scores."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import pointwise as pw
from test_direct255 import fixture
from transfer import read, sha, write


class PointwiseTests(unittest.TestCase):
    def test_inputs_only_use_geometry_ids_and_ego_motion(self):
        arrays, vocab = fixture()
        inputs = pw.scene_inputs(arrays, len(vocab))
        self.assertEqual(inputs['local'], list(range(64)))
        self.assertEqual(inputs['vocab'], arrays['candidate_ids'][:64].tolist())
        self.assertEqual(inputs['velocity'], arrays['ego_status'][4:6].tolist())
        self.assertEqual(set(inputs), {'local', 'vocab', 'velocity', 'acceleration'})

    def test_only_photographic_background_changes(self):
        path = np.zeros((40, 3))
        path[:, 0] = np.arange(1, 41) / 2
        inputs = dict(velocity=[5., 0.], acceleration=[0., 0.])
        photo = Image.new('RGB', (768, 352), (20, 30, 40))
        with patch.object(pw.projection, 'path_segments', return_value=[((200, 100), (200, 220))]):
            images, visible = pw.candidate_images(photo, None, path, pw.scale_for(path[None]))
        self.assertTrue(visible)
        blank, road = images['blank'], images['road']
        self.assertEqual(blank.crop((0, 352, 768, 640)).tobytes(), road.crop((0, 352, 768, 640)).tobytes())
        self.assertEqual(blank.getpixel((200, 150)), road.getpixel((200, 150)))
        self.assertEqual(blank.getpixel((250, 150)), (255, 255, 255))
        self.assertEqual(road.getpixel((250, 150)), (20, 30, 40))
        prompts = [pw.payload(inputs, image) for image in (blank, road)]
        self.assertEqual(prompts[0]['state'], prompts[1]['state'])
        self.assertEqual(prompts[0]['question'], prompts[1]['question'])
        self.assertEqual(list(prompts[0]['question']['criteria']), list(pw.OPTIONS))

    def test_resume_reuses_pilot_scores_and_selection_uses_the_same_top64(self):
        arrays, vocab = fixture()
        entry = pw.scene_inputs(arrays, len(vocab))
        photo = Image.new('RGB', (200, 100), 'white')
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder, patch.object(pw.projection, 'path_segments', return_value=[]):
            out = Path(folder)
            calls = []

            def infer(row):
                calls.append(row['state'])
                return dict(score=.5, tokens=40, seconds=1.)

            save = lambda rows: pw.checkpoint(out, 'a', 'plan', entry, rows)
            pw.infer_scene(entry, photo, None, vocab, pw.PILOT_RANKS, {}, infer, save)
            self.assertEqual(len(calls), 16)
            values = pw.checkpoint(out, 'a', 'plan', entry)
            self.assertFalse(values['0']['visible'])
            pw.infer_scene(entry, photo, None, vocab, range(64), values, infer, save)
            self.assertEqual(len(calls), 128)
            values = pw.checkpoint(out, 'a', 'plan', entry)
            values['63']['blank']['score'] = .8
            values['15']['blank']['score'] = .7
            self.assertEqual(pw.select(values, 'blank', 16)['position'], 15)
            self.assertEqual(pw.select(values, 'blank', 64)['position'], 63)
            self.assertEqual(pw.select(values, 'road', 64), dict(position=0, tie=True, score=.5))
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                pw.checkpoint(out, 'a', 'another plan', entry)

    def test_evaluation_uses_committed_choices_and_physical_ids(self):
        arrays, vocab = fixture()
        entry = pw.scene_inputs(arrays, len(vocab))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder, patch('builtins.print'):
            out, astra = Path(folder) / 'run', Path(folder) / 'astra'
            write(out / 'plan.json', dict(config=pw.CONFIG, manifest_sha256='fixture',
                records=[dict(token='a', session_id='s', role='train')], entries={'a': entry}))
            choices = {'a': {f'{m}/top{k}': dict(position=k-1 if m == 'road' else 0, score=.9, tie=False)
                             for m in pw.MODES for k in pw.POOLS}}
            write(out / 'decisions.json', choices)
            write(out / 'report.json', dict(status='choices_complete', outcomes_read=False, scenes=1, requests=128,
                plan_sha256=sha(out / 'plan.json'), decisions_sha256=sha(out / 'decisions.json')))
            write(astra / 'request.json', dict(manifest_sha256='fixture', candidates={'a': dict(
                ranked=list(range(256)), vocab=arrays['candidate_ids'].tolist())}))
            write(astra / 'report.json', dict(status='complete', oracle_scenes=1, completed_candidates=256,
                                             anchors_parity_passed=1, holdout_processed=0))
            metrics = dict.fromkeys(pw.METRICS, .5)
            labels = [dict(metrics, token='a', session_id='s', candidate_local=i,
                           candidate_vocab=int(arrays['candidate_ids'][i]), teacher_rank=i+1,
                           pdm_score=.5 + i / 1000) for i in (0, 15, 63)]
            pw.csv_gz(astra / 'per_candidate.csv.gz', labels)
            result = pw.evaluate(out, astra)
            self.assertEqual(result['methods']['road/top64']['helpful'], 1)
            self.assertAlmostEqual(result['contrasts']['road/top64 minus road/top16']['pdm_gain_pp']['macro_mean'], 4.8)
            self.assertEqual(result['contrasts']['blank/top64 minus blank/top16']['choices_changed'], 0)
            labels[2]['teacher_rank'] = 16
            pw.csv_gz(astra / 'per_candidate.csv.gz', labels)
            with self.assertRaisesRegex(ValueError, 'Astra row'):
                pw.evaluate(out, astra)
            write(out / 'decisions.json', {})
            with self.assertRaisesRegex(ValueError, 'committed'):
                pw.evaluate(out, astra)
            self.assertEqual(read(out / 'evaluation.json'), result)


class SlurmTests(unittest.TestCase):
    def test_night_has_one_auto_node_allocation_and_runs_pilot_before_full(self):
        from test_runtime import SlurmRoutingTests

        shell = SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        shell.setUp()
        self.addCleanup(shell.doCleanups)
        shell.script = Path(pw.__file__).with_name('run_pointwise.sh').resolve()
        shell.env['DREX_POINTWISE_NAME'] = 'pointwise_fixture'
        shell.stub('sbatch', 'printf "%s\\n" "$@" > "$DREX_TEST_LOG"\n')
        for replicas in ('1', '2', '4', '8'):
            with self.subTest(replicas=replicas):
                result = shell.run_stage('submit-night', replicas)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = shell.log.read_text().splitlines()
                self.assertIn('--nodes=1', args)
                self.assertIn(f'--gres=gpu:{replicas}', args)
                self.assertFalse(any(arg.startswith('--nodelist=') for arg in args))

        shell.fake_setup()
        shell.stub('srun', 'printf "%s\\n" "srun $*" >> "$DREX_TEST_CALLS"\n')
        result = shell.run_stage('--inside', 'coordinator-night', shell.posix(shell.script.parents[2]),
                                 shell.posix(shell.envroot), '4', 'pointwise_fixture')
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = shell.calls.read_text()
        self.assertIn('--gpus-per-task=1', calls)
        self.assertLess(calls.index('worker-pilot'), calls.index('--phase pilot'))
        self.assertLess(calls.index('--phase pilot'), calls.index('worker-full'))


if __name__ == '__main__':
    unittest.main()
