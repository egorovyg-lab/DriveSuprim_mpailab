import ast
import copy
import csv
import gzip
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import consequence_cli as cli
import consequence_data as data
import consequence_model as model
import consequence_run as run
from transfer import sha, write


def sample():
    table = np.arange(64*len(data.TABLE), dtype=float).reshape(64, -1) / 100
    return dict(baseline_local=0, ego_velocity=[5., 0.], ego_acceleration=[0., .1],
                candidates=[dict(local=i, vocab=i+100, values=table[i].tolist()) for i in range(64)])


def fixture(root):
    rng = np.random.default_rng(55)
    records = [dict(token=f'{i:016x}', session_id=f's{i//2}', role='train') for i in range(16)]
    x = rng.normal(size=(16, 63, len(data.FEATURES)))
    prefs = rng.random((16, 63, 3, 4)); prefs[:, :, :, 2] = prefs[:, :, :, :2].mean(axis=3)
    prefs[:, :, :, 3] = abs(prefs[:, :, :, 0]-prefs[:, :, :, 1])
    outcomes = np.zeros((16, 64, len(data.OUTCOMES)))
    outcomes[:, :, :9] = 1.; ix = {k: i for i, k in enumerate(data.OUTCOMES)}
    outcomes[:, :, ix['route_progress_m']] = 10
    outcomes[:, :, ix['all_jerk_norm_mps3_rms']] = 1 + .1*rng.normal(size=(16, 64))
    outcomes[:, :, ix['all_jerk_norm_mps3_abs_max']] = 3
    outcomes[:, :, ix['all_accel_lat_mps2_rms']] = 1
    outcomes[:, :, ix['all_actor_observed_interval_fraction']] = 1
    outcomes[:, :, -2:] = 2.
    outcomes[:, 1, ix['all_jerk_norm_mps3_rms']] = .3
    run.save_arrays(root/'features.npz', x=x, preferences=prefs,
                    local_ids=np.tile(np.arange(64), (16, 1)), vocab_ids=np.tile(np.arange(100, 164), (16, 1)),
                    fold=model.folds([r['session_id'] for r in records]))
    run.save_arrays(root/'labels.npz', outcomes=outcomes)
    request = dict(protocol='drex_consequence_probe_v1', config=copy.deepcopy(model.CONFIG), records=records,
                   features=list(data.FEATURES), outcomes=list(data.OUTCOMES),
                   tools={n: sha(Path(run.__file__).with_name(n)) for n in run.TOOLS},
                   files={n: sha(root/n) for n in ('features.npz', 'labels.npz')})
    write(root/'audit_request.json', request)
    write(root/'plan.json', dict(request_sha256=sha(root/'audit_request.json')))
    return request


class ProbeTests(unittest.TestCase):
    def test_feature_whitelist_and_heading_wrap(self):
        s = sample(); heading = data.TABLE.index('heading_4s_deg')
        s['candidates'][0]['values'][heading] = 179
        s['candidates'][1]['values'][heading] = -179
        x = data.feature_block(s)
        self.assertEqual(x[0, heading], 2)
        # Privileged data and IDs have no numerical route into the predictor.
        s['measured_outcomes'] = {'pdm_score': -1e9}
        s['candidates'][1]['measured_clearance'] = -1e9
        np.testing.assert_array_equal(x, data.feature_block(s))
        self.assertFalse(any(k in data.FEATURES for k in data.OUTCOMES))

    def test_pair_physical_alignment_and_duplicates(self):
        s = sample(); pairs = []
        for rank, c in enumerate(s['candidates'][1:], 2):
            for mode in ('none', 'blank', 'road'):
                pairs.append(dict(candidate_local=c['local'], candidate_vocab=c['vocab'], teacher_rank=rank,
                                  mode=mode, reference_first_preference=.3, challenger_first_preference=.7, mean_preference=.5))
        self.assertEqual(data.preference_block(s, list(reversed(pairs))).shape, (63, 3, 4))
        pairs[0]['candidate_vocab'] = -1
        with self.assertRaises(ValueError): data.preference_block(s, pairs)
        pairs[0]['candidate_vocab'] = 101
        with self.assertRaises(ValueError): data.preference_block(s, pairs + pairs[:1])

    def test_truth_budgets_missing_margins_and_physical_harm(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture(root)
            o = run.load_arrays(root/'labels.npz')['outcomes']; ix = {k:i for i,k in enumerate(data.OUTCOMES)}
            o[:, 1, ix['route_progress_m']] = 9.6  # Within both 1m and 5%.
            o[:, 2, ix['route_progress_m']] = 9.4  # Fails 5% even though less than 1m.
            o[:, 1, -1] = np.nan
            o[:, 3, ix['all_jerk_norm_mps3_rms']] = o[:, 0, ix['all_jerk_norm_mps3_rms']]
            o[:, 3, ix['all_accel_lat_mps2_rms']] = .5
            y, f = data.truth(o)
            self.assertTrue(np.isfinite(y).all()); self.assertTrue(np.isnan(o[:, 1, -1]).all())
            self.assertTrue(y[:, 0, 6].all()); self.assertTrue(f['progress_budget_exceeded'][:, 1].all())
            self.assertTrue(y[:, 2, 6].all())  # Lateral-only benefit is a valid comfort improvement.
            o[0, 1, ix['drivable_area_compliance']] = 0
            self.assertEqual(data.truth(o)[0][0, 0, 6], 0)
            o[1, 1, ix['all_actor_contact_time_s']] = .1
            self.assertTrue(data.truth(o)[1]['exposure_worse'][1, 0])

    def test_session_folds_and_balanced_weights(self):
        sessions = ['a', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        f = model.folds(sessions); self.assertEqual(f[0], f[1])
        w = model.weights(sessions, 63).reshape(-1, 63).sum(axis=1)
        for s in set(sessions): self.assertAlmostEqual(w[np.array(sessions) == s].sum(), 1/8)
        for k in range(4): self.assertFalse(set(np.array(sessions)[f == k]) & set(np.array(sessions)[f != k]))

    def test_train_only_scaling_and_weighted_ridge_solution(self):
        rng = np.random.default_rng(3); x = rng.normal(size=(40, 4)); y = rng.normal(size=(40, 7))
        w = np.arange(1, 41, dtype=float); w /= w.sum()
        m = model.fit(x, y, w, 'linear')
        np.testing.assert_allclose(m['x_mean'], w @ x)
        a = model.design(x, m); regularizer = np.eye(a.shape[1])*.1; regularizer[0, 0] = 0
        expected = np.linalg.solve(a.T @ (w[:, None]*a) + regularizer,
                                   a.T @ (w[:, None]*((y-m['y_mean'])/m['y_scale'])))
        np.testing.assert_allclose(expected, m['coef'])
        old = m['x_mean'].copy(); model.predict(m, x*1e6)
        np.testing.assert_array_equal(old, m['x_mean'])

    def test_shuffled_control_preserves_within_scene_scores(self):
        rng = np.random.default_rng(6); x = rng.normal(size=(2, 63, 4)); p = rng.random((2, 63, 3, 4))
        road = model.inputs(x, p, ['a', 'b'], 'ds_drex_road')
        shuffled = model.inputs(x, p, ['a', 'b'], 'ds_drex_shuffled')
        np.testing.assert_array_equal(road[:, :, :-4], shuffled[:, :, :-4])
        for i in range(2):
            self.assertEqual(sorted(map(tuple, road[i, :, -4:])), sorted(map(tuple, shuffled[i, :, -4:])))
        self.assertFalse(np.array_equal(road, shuffled))
        np.testing.assert_array_equal(shuffled, model.inputs(x, p, ['a', 'b'], 'ds_drex_shuffled'))

    def test_policy_fallback_and_no_target_consumption(self):
        p = np.zeros((2, 63, 7)); p[:, :, 0] = .3; p[:, :, 6] = .1
        np.testing.assert_array_equal(model.choose(p, .1), [-1, -1])
        p[0, 5, 6] = .8; p[1, 5, 6] = .8; p[1, 5, 5] = .2
        np.testing.assert_array_equal(model.choose(p, .1), [5, -1])
        self.assertEqual(list(inspect.signature(model.choose).parameters), ['prediction', 'threshold'])
        self.assertIsNone(model.average_precision([0, 0], [.9, .2]))
        self.assertAlmostEqual(model.average_precision([0, 1], [.5, .5]), .5)

    def test_fold_predictions_do_not_consume_held_out_targets(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(model.CONFIG, random_features=4):
            root=Path(d); r=fixture(root); _, f, _, y, _=run.context(root/'audit_request.json')
            with patch('builtins.print'):
                first=run.fold_fit(f, y, r['records'], 0)
                changed=y.copy(); changed[f['fold'] == 0] = 1e6
                second=run.fold_fit(f, changed, r['records'], 0)
            np.testing.assert_array_equal(first['predictions'], second['predictions'])

    def test_pilot_resume_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(model.CONFIG, random_features=4, bootstrap_reps=20):
            root=Path(d); fixture(root); out=root/'artifacts'
            with patch('builtins.print'):
                pilot=run.run_locked(root/'audit_request.json', out, 'pilot')
                before=sha(out/'folds/0/model_predictions.npz')
                full=run.run_locked(root/'audit_request.json', out, 'full')
            self.assertEqual(pilot['status'], 'pilot_complete'); self.assertEqual(full['status'], 'complete')
            self.assertEqual(full['sessions'], 8); self.assertEqual(full['new_drex_calls'], 0)
            self.assertEqual(before, sha(out/'folds/0/model_predictions.npz'))
            self.assertTrue((out/'per_scene.csv.gz').is_file())
            with (out/'folds/0/model_predictions.npz').open('ab') as f: f.write(b'tamper')
            with self.assertRaisesRegex(RuntimeError, 'checkpoint binding'):
                run.run_locked(root/'audit_request.json', out, 'full')

    def test_frozen_upload_and_python39_compatibility(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); fixture(root)
            self.assertLess(len(cli.upload(root)), 32*1024**2)
            with (root/'features.npz').open('ab') as f: f.write(b'tamper')
            with self.assertRaisesRegex(RuntimeError, 'dataset changed'): cli.upload(root)
        for module in (cli, data, model, run):
            ast.parse(Path(module.__file__).read_text(encoding='utf-8'), feature_version=(3, 9))

    def test_full_cohort_preparation_and_bounded_upload(self):
        import replacement
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'runs/replacement_test'; source.mkdir(parents=True)
            table=root/'runs/table_test'; table.mkdir()
            pool=root/'runs/rollout_audit_pool_test'; pool.mkdir()
            archive=pool/'placeholder.tar'; archive.write_bytes(b'archive fixture')
            digest=sha(archive); archive.rename(pool/('audit-'+digest+'.tar'))
            folder=pool/('result_'+digest[:12]); folder.mkdir()
            records=[dict(token=f'{i:016x}', session_id=f's{i%32}', role='train') for i in range(651)]
            s=sample(); samples={r['token']: s for r in records}
            candidates=[dict(token=r['token'], candidate_local=c['local'], candidate_vocab=c['vocab'], expected_raw=[1.]*9)
                        for r in records for c in s['candidates']]
            write(source/'report.json', dict(status='complete')); write(source/'decision_commit.json', {})
            write(pool/'audit_request.json', dict(source=str(table), candidates=candidates))
            write(folder/'identity.json', dict(request_sha256=sha(pool/'audit_request.json')))
            write(pool/'latest_result.json', dict(archive_sha256=digest, remote_exit_code=0, result=str(folder)))
            values={k: 0. for k in data.OUTCOMES}; values.update({k: 1. for k in data.OUTCOMES[:9]})
            values.update(route_progress_m=10., all_jerk_norm_mps3_rms=1., all_accel_lat_mps2_rms=1.,
                          all_jerk_norm_mps3_abs_max=3., all_actor_observed_interval_fraction=1.)
            with gzip.open(folder/'per_candidate.csv.gz', 'wt', newline='') as f:
                writer=csv.DictWriter(f, fieldnames=['token','session_id','candidate_local','candidate_vocab','teacher_rank',*data.OUTCOMES])
                writer.writeheader()
                for r in records:
                    for rank, c in enumerate(s['candidates'], 1):
                        writer.writerow(dict(token=r['token'], session_id=r['session_id'], candidate_local=c['local'],
                                             candidate_vocab=c['vocab'], teacher_rank=rank, **values))
            def pairs():
                for r in records:
                    for rank, c in enumerate(s['candidates'][1:], 2):
                        for mode in ('none','blank','road'):
                            yield dict(token=r['token'], candidate_local=c['local'], candidate_vocab=c['vocab'],
                                       teacher_rank=rank, mode=mode, reference_first_preference=.3,
                                       challenger_first_preference=.7, mean_preference=.5)
            args=SimpleNamespace(root=root, source=source.name, pool=pool.name, name='rollout_audit_probe_test')
            with patch.object(replacement, 'collect', return_value=(dict(records=records), pairs(), None, None)), \
                    patch.object(replacement, 'context', return_value=(None, None, table, None)), \
                    patch.object(replacement, 'inputs', return_value=samples), \
                    patch.object(cli.transport.old, 'extract_result', return_value=dict(status='complete', required_candidates=41664, metric_parity_passed=41664)), \
                    patch('builtins.print'):
                cli.prepare(args)
            output=root/'runs'/args.name
            request, f, _, y, _=run.context(output/'audit_request.json')
            self.assertEqual(f['x'].shape[:2], (651, 63)); self.assertEqual(y.shape, (651, 63, 7))
            self.assertEqual(len(request['records']), 651); self.assertLess(len(cli.upload(output)), 32*1024**2)

    def test_detached_controller_pilot_failure_stops_full(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as d:
                root=Path(d); job=root/'rollout_audit_test'; (job/'tools').mkdir(parents=True)
                (job/'audit_request.json').write_text('{}')
                (job/'tools/consequence_run.py').write_text('''import sys,json,hashlib
from pathlib import Path
r=Path(sys.argv[sys.argv.index('--out')+1]);r.mkdir(exist_ok=True)
p=sys.argv[sys.argv.index('--stage')+1]
with (r/'phases.txt').open('a') as f:f.write(p+'\\n')
q=Path(sys.argv[sys.argv.index('--input')+1])
(r/'identity.json').write_text(json.dumps({'request_sha256':hashlib.sha256(q.read_bytes()).hexdigest()}))
(r/'report.json').write_text(json.dumps({'status':'pilot_complete' if p=='pilot' else 'complete'}))
''' + ('raise SystemExit(1)\n' if failure else ''))
                preface = ''
                if sys.platform == 'win32':
                    preface='import sys,types\nsys.modules["fcntl"]=types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1,LOCK_NB=2)\n'
                source=inspect.getsource(cli.supervise).replace("Path('/workspace/exp_v2/drex_rollout_audit').resolve()", f'Path({str(root)!r}).resolve()')
                result=subprocess.run([sys.executable, '-B', '-c', preface+source+f'\nraise SystemExit(supervise({str(job)!r},{str(root)!r}))'],
                                      capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, int(failure), result.stderr)
                self.assertEqual((job/'artifacts/phases.txt').read_text().splitlines(), ['pilot'] if failure else ['pilot', 'full'])

    def test_slurm_cpu_routing(self):
        import test_runtime as fixtures
        f=fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        f.setUp(); self.addCleanup(f.doCleanups)
        for key in ('SLURM_JOB_ID', 'SLURM_STEP_ID'): f.env.pop(key, None)
        f.script=Path(cli.__file__).with_name('run_consequence_probe.sh').resolve()
        (f.envroot/'setup_node.txt').write_text('cn29\n')
        f.stub('srun', 'printf "%s\\n" "$*" >> "$DREX_TEST_LOG"\n')
        result=f.run_stage('prepare')
        self.assertEqual(result.returncode, 0, result.stderr)
        text=f.log.read_text(); self.assertIn('--gres=none', text); self.assertIn('--inside prepare', text)
        self.assertNotIn('--gres=gpu', text)
        result=f.run_stage('--inside', 'prepare')
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__': unittest.main()
