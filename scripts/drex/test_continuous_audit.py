import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import continuous_audit as audit
import continuous_metrics as metrics
import continuous_remote as remote
import continuous_report as report
import real_data as base
import test_runtime as fixtures
from transfer import write, read, sha


def states():
    s = np.zeros((41, 11)); s[:, 0] = np.arange(41)*.5; s[:, 3] = 5.
    return s


def frames(margin=1., actor=5.):
    return [dict(full_footprint_covered=True, road_clearance_m=margin, outside_area_m2=0.,
        min_corner_signed_margin_m=margin, actor_clearance_m=actor, nearest_actor_token='other',
        intersecting_actor_tokens=[]) for _ in range(41)]


def fixture(count=5):
    decisions, old, labels, samples = [], [], {}, {}; manifest = dict(files={})
    for i in range(count):
        token = f'{i+1:016x}'; metadata = dict(token=token, session_id='session'+str(i))
        methods = {m: (1 if m == 'none/both050' else 2 if m == 'road/both050' else 0)
                   for m in metrics.METHODS if m not in ('baseline', 'old_table', 'old_camera')}
        decisions.append(dict(metadata=metadata, methods=methods))
        old.append(dict(metadata=metadata, k=64, methods={'table/mean3': 1, 'table_camera/mean3': 0}))
        samples[token] = dict(baseline_local=0, candidates=[dict(local=j, vocab=j+10) for j in range(3)])
        labels[token] = {j: np.ones(9) for j in range(3)}
        manifest['files']['bundle/examples/'+token+'.npz'] = dict(sha256='0'*64)
    request = audit.assemble(decisions, old, labels, samples, manifest)
    return request


class SignalTests(unittest.TestCase):
    def test_constant_speed_no_fake_comfort_and_four_second_duration(self):
        r = metrics.measure(states(), frames(), .1, 20.)
        self.assertEqual(r['all_mean_speed_mps'], 5.)
        self.assertEqual(r['all_jerk_norm_mps3_rms'], 0.)
        self.assertAlmostEqual(r['all_moving_time_s'], 4.)
        self.assertAlmostEqual(r['late_moving_time_s'], 3.)
        self.assertEqual(r['all_stationary_fraction'], 0.)
        f = frames(); f = [dict(x, full_footprint_covered=False, road_clearance_m=None, outside_area_m2=2.) for x in f]
        r = metrics.spatial(f, .1)
        self.assertAlmostEqual(r['outside_footprint_time_s'], 4.)
        self.assertAlmostEqual(r['outside_area_integral_m2s'], 8.)
        self.assertIsNone(r['road_clearance_min_m'])

    def test_linear_acceleration_has_known_jerk_without_boundary_padding(self):
        s = states(); t = np.arange(41)*.1; s[:, 3] = 5.+t**2
        r = metrics.kinematics(s, .1)
        self.assertAlmostEqual(r['jerk_norm_mps3_rms'], 2.)
        self.assertAlmostEqual(r['jerk_lon_mps3_rms'], 2.)
        self.assertEqual(r['jerk_lat_mps3_abs_max'], 0.)

    def test_turning_has_centripetal_acceleration_even_with_zero_stored_lateral(self):
        s = states(); t = np.arange(41)*.1; s[:, 2] = t*.25; s[:, 9] = .25
        r = metrics.kinematics(s, .1)
        self.assertAlmostEqual(r['accel_lat_mps2_rms'], 10*np.sin(.25*.1/2)/.1)
        self.assertAlmostEqual(r['jerk_norm_mps3_rms'], 20*np.sin(.25*.1/2)**2/.1**2)
        # Constant world acceleration represented in a rotating body has zero world jerk.
        s[:, 3] = (5.+2*t)*np.cos(s[:, 2]); s[:, 4] = -(5.+2*t)*np.sin(s[:, 2])
        self.assertLess(metrics.kinematics(s, .1)['jerk_norm_mps3_abs_max'], 1e-12)

    def test_global_heading_rotation_does_not_change_body_metrics(self):
        a = states(); a[:, 3] += (np.arange(41)*.1)**2; a[:, 4] = .3
        b = a.copy(); b[:, 2] += 2.2
        for k in ('accel_lon_mps2_rms', 'jerk_norm_mps3_rms', 'jerk_lon_mps3_abs_max'):
            self.assertAlmostEqual(metrics.kinematics(a, .1)[k], metrics.kinematics(b, .1)[k])

    def test_stopped_motion_has_null_moving_comfort_not_zero(self):
        s = states(); s[:, 3] = 0
        r = metrics.kinematics(s, .1)
        self.assertEqual(r['stationary_fraction'], 1.)
        self.assertIsNone(r['jerk_norm_mps3_rms_moving'])
        self.assertIsNone(r['accel_lat_mps2_rms_moving'])
        self.assertEqual(r['moving_jerk_interval_time_s'], 0.)

    def test_initial_clearance_is_preserved_separately_from_late_window(self):
        f = frames(margin=1.); f[0]['road_clearance_m'] = .01
        self.assertEqual(metrics.spatial(f, .1)['road_clearance_min_m'], .01)
        self.assertEqual(metrics.spatial(f, .1, 1.)['road_clearance_min_m'], 1.)
        self.assertAlmostEqual(metrics.spatial(f, .1)['road_below_0.1m_time_s'], .05)

    def test_absent_and_partial_actor_observations_report_coverage(self):
        f = frames(actor=None); r = metrics.spatial(f, .1)
        self.assertIsNone(r['actor_clearance_min_m'])
        self.assertIsNone(r['actor_below_1.0m_observed_time_s'])
        self.assertEqual(r['actor_observed_interval_fraction'], 0.)
        f[0]['actor_clearance_m'] = .2; f[1]['actor_clearance_m'] = .2
        r = metrics.spatial(f, .1)
        self.assertAlmostEqual(r['actor_below_1.0m_observed_time_s'], .1)
        self.assertAlmostEqual(r['actor_observed_interval_fraction'], 1/40)

    def test_bad_time_grid_nonfinite_and_inconsistent_coverage_rejected(self):
        with self.assertRaises(ValueError): metrics.kinematics(states(), .2)
        s = states(); s[3, 5] = np.nan
        with self.assertRaises(ValueError): metrics.kinematics(s, .1)
        f = frames(); f[0]['road_clearance_m'] = None
        with self.assertRaises(ValueError): metrics.spatial(f, .1)


class ContractTests(unittest.TestCase):
    def test_full_cohort_dedup_pilot_outcome_independence_and_schema(self):
        request = fixture(); metrics.validate_request(request)
        self.assertEqual(len(request['records']), 5)
        self.assertEqual(len(request['candidates']), 15)
        self.assertEqual(len(request['pilot_tokens']), 4)
        altered = copy.deepcopy(request)
        for c in altered['candidates']: c['expected_raw'] = [0.]*9
        self.assertEqual(request['pilot_tokens'], altered['pilot_tokens'])
        for mutation in ('duplicate', 'nonfinite', 'method', 'identity'):
            r = copy.deepcopy(request)
            if mutation == 'duplicate': r['candidates'].append(r['candidates'][0])
            if mutation == 'nonfinite': r['candidates'][0]['expected_raw'][0] = float('nan')
            if mutation == 'method': del r['records'][0]['methods']['old_camera']
            if mutation == 'identity': r['candidates'][0]['candidate_vocab'] = -1
            with self.assertRaises(ValueError): metrics.validate_request(r)

    def test_parity_failures_block_metrics(self):
        remote.parity(np.ones(9), np.ones(9))
        for value in (np.zeros(9), np.full(9, np.nan), np.ones(8)):
            with self.assertRaises(RuntimeError): remote.parity(value, np.ones(9))

    def test_progress_budget_requires_measured_benefit_and_guards(self):
        b = metrics.measure(states(), frames(margin=.2), .1, 20.)
        c = metrics.measure(states(), frames(margin=.4), .1, 19.2)
        r = metrics.tradeoff(c, b, True, True)
        self.assertFalse(r['0.5']['eligible']); self.assertTrue(r['1.0']['eligible'])
        self.assertTrue(r['1.0']['road_clearance_min_m'])
        self.assertFalse(r['1.0']['jerk_norm_mps3_rms'])
        self.assertFalse(metrics.tradeoff(c, b, False, True)['1.0']['eligible'])
        self.assertFalse(metrics.tradeoff(c, b, True, False)['1.0']['eligible'])
        c['all_outside_area_max_m2'] = 1.
        self.assertFalse(metrics.tradeoff(c, b, True, True)['2.0']['eligible'])

    def test_session_macro_and_missing_pairs_are_not_imputed(self):
        r = metrics.paired_stats([1., 1., 1., 5., None], ['a', 'a', 'a', 'b', 'c'])
        self.assertEqual(r['micro_mean'], 2.)
        self.assertEqual(r['macro_mean'], 3.)
        self.assertEqual(r['paired_scenes'], 4)
        self.assertEqual(r['paired_sessions'], 2)
        self.assertIsNone(metrics.paired_stats([None], ['a'])['macro_mean'])

    def test_transport_size_and_python39_syntax(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); write(out/'audit_request.json', fixture())
            payload = audit.transport.input_archive(out, audit.TOOLS)
            # Request growth to all 651 scenes leaves ample space within the existing transport cap.
            self.assertLess(len(payload)+651*3*1300, 4*1024**2)
        for name in (*audit.TOOLS, 'continuous_audit.py', 'test_continuous_audit.py'):
            if name.endswith('.py'):
                ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'), feature_version=(3, 9))


class PipelineTests(unittest.TestCase):
    def setup_fake(self, directory):
        directory = Path(directory); package = directory/'package'; repo = directory/'repo'; out = directory/'out'
        request = fixture(); manifest = dict(files={})
        for r in request['records']:
            path = package/'bundle/examples'/(r['token']+'.npz'); path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, candidate_ids=np.array([10, 11, 12]), label_indices=np.array([0]), label_raw=np.ones((1, 9)))
            rel = path.relative_to(package).as_posix(); manifest['files'][rel] = dict(sha256=sha(path))
            for c in request['candidates']:
                if c['token'] == r['token']: c['example_sha256'] = sha(path)
        write(package/'package_manifest.json', manifest)
        for name in (*remote.replay.REPLAY_SOURCES, *remote.EXTRA_SOURCES):
            target = repo/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_text('fixture')
        request.update(remote_package=str(package), package_manifest_sha256=sha(package/'package_manifest.json'),
            vehicle_profile=read(Path(__file__).with_name('vehicle_profile_astra.json')), tools={})
        path = directory/'request.json'; write(path, request)
        context = (manifest, {}, None, {}, {})
        context = (*context[:-1], {r['token']: {} for r in request['records']})
        def evaluate(c, *args):
            fs = frames(margin=1.+c['candidate_local']*.1)
            result = dict(token=c['token'], candidate_local=c['candidate_local'], candidate_vocab=c['candidate_vocab'],
                teacher_rank=c['teacher_rank'], measured_raw=c['expected_raw'],
                metrics=metrics.measure(states(), fs, .1, 20.-.25*c['candidate_local']))
            return result, dict(simulated_states=states()), fs
        return request, path, out, repo, context, evaluate

    def test_pilot_full_resume_commit_tamper_and_no_queries(self):
        with tempfile.TemporaryDirectory() as d:
            request, path, out, repo, context, evaluate = self.setup_fake(d)
            with patch('builtins.print'), patch.object(remote.replay, 'bound_context', return_value=context), \
                    patch.object(remote.replay, 'config_core', return_value={}), \
                    patch.object(remote, 'evaluate', side_effect=evaluate) as scoring:
                self.assertEqual(remote.run(path, out, repo, 'pilot'), 0)
                self.assertEqual(scoring.call_count, 12)
                self.assertEqual(read(out/'report.json')['status'], 'pilot_complete')
                self.assertFalse((out/'per_scene.csv').exists())
                self.assertEqual(remote.run(path, out, repo, 'full'), 0)
                self.assertEqual(scoring.call_count, 15)
                result = read(out/'report.json')
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['model_inference_calls'], 0)
                self.assertEqual(result['metric_parity_passed'], 15)
                self.assertEqual(result['comparisons_vs_DS']['road/both050']['interventions'], 5)
                self.assertAlmostEqual(result['comparisons_vs_DS']['road/both050']['deltas_all_scenes']['route_progress_m']['macro_mean'], -.5)
                self.assertEqual(remote.run(path, out, repo, 'full'), 0)
                self.assertEqual(scoring.call_count, 15)
                c = request['candidates'][0]; target = out/'candidates'/remote.candidate_key(c)/'summary.json'
                target.write_text('{}')
                with self.assertRaises(RuntimeError): remote.run(path, out, repo, 'full')
                self.assertEqual(scoring.call_count, 15)

    def test_failed_candidate_preserved_and_cannot_silently_retry(self):
        with tempfile.TemporaryDirectory() as d:
            request, path, out, repo, context, _ = self.setup_fake(d)
            with patch.object(remote.replay, 'bound_context', return_value=context), \
                    patch.object(remote.replay, 'config_core', return_value={}), \
                    patch.object(remote, 'evaluate', side_effect=RuntimeError('raw parity')) as scoring:
                with self.assertRaises(RuntimeError): remote.run(path, out, repo, 'pilot')
                with self.assertRaisesRegex(RuntimeError, 'Prior candidate failure'): remote.run(path, out, repo, 'pilot')
                self.assertEqual(scoring.call_count, 1)
                self.assertFalse((out/'per_scene.csv').exists())

    def test_full_replay_requires_pilot(self):
        with tempfile.TemporaryDirectory() as d:
            _, path, out, repo, context, _ = self.setup_fake(d)
            with patch.object(remote.replay, 'bound_context', return_value=context), \
                    patch.object(remote.replay, 'config_core', return_value={}), patch.object(remote, 'evaluate') as scoring:
                with self.assertRaises(FileNotFoundError): remote.run(path, out, repo, 'full')
                scoring.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('shapely'), 'Optional Shapely polygon tests')
class PolygonTests(unittest.TestCase):
    def test_union_seam_and_body_hole_and_contact(self):
        from shapely.geometry import box, Polygon
        import clearance_math as geometry
        c = np.array([[11, 6], [9, 6], [9, 4], [11, 4], [10, 5.]])
        r = geometry.frame_measurement(c, geometry.road_union([box(0, 0, 10, 10), box(10, 0, 20, 10)]), [])
        self.assertEqual(r['road_clearance_m'], 4.)
        road = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)], [[(9.5, 4.5), (10.5, 4.5), (10.5, 5.5), (9.5, 5.5)]])
        r = geometry.frame_measurement(c, road, [('car', box(10, 4, 12, 6))])
        self.assertFalse(r['full_footprint_covered']); self.assertIsNone(r['road_clearance_m'])
        self.assertGreater(r['min_corner_signed_margin_m'], 0.)
        self.assertGreater(r['outside_area_m2'], 0.)
        self.assertEqual(r['actor_clearance_m'], 0.)
        self.assertEqual(r['intersecting_actor_tokens'], ['car'])


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.script = Path(__file__).with_name('run_continuous_audit.sh').resolve()
        (self.f.envroot/'setup_node.txt').write_text('cn29\n')
        for k in ('DREX_CONTINUOUS_NAME', 'DREX_CONTINUOUS_SOURCE'): self.f.env.pop(k, None)

    def test_checks_and_preparation_request_cpu_only(self):
        for stage in ('check', 'prepare'):
            r = self.f.run_stage(stage)
            self.assertEqual(r.returncode, 0, r.stderr)
            args = self.f.log.read_text().splitlines()
            self.assertIn('--gres=none', args)
            self.assertIn('--nodelist=cn29', args)
            self.assertEqual(args[-2:], ['--inside', stage])

    def test_successful_mini_tests_return_zero_and_prepare_runs_after_them(self):
        self.f.fake_setup()
        r = self.f.run_stage('--inside', 'check')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('test_continuous_audit.py', self.f.calls.read_text())
        self.assertNotIn('continuous_audit.py prepare', self.f.calls.read_text())
        r = self.f.run_stage('--inside', 'prepare')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('continuous_audit.py prepare', self.f.calls.read_text())

    def test_failed_mini_tests_prevent_prepare(self):
        self.f.fake_setup()
        path = self.f.envroot/'venv/bin/python'
        path.write_text('#!/usr/bin/env bash\nexit 9\n', newline='\n'); path.chmod(0o755)
        r = self.f.run_stage('--inside', 'prepare')
        self.assertEqual(r.returncode, 9, r.stderr)
        self.assertFalse((self.f.envroot/'runs/rollout_audit_continuous_v1/audit_request.json').exists())

    def test_full_worker_runs_only_after_successful_pilot(self):
        self.f.stub('flock', 'exit 0\n')
        path = self.f.stub('continuous-python', 'printf "%s\\n" "$*" >> "$DREX_TEST_CALLS"\n'
            'if [[ ${DREX_TEST_FAIL_PILOT:-} == 1 && "$*" == *"continuous_audit.py pilot "* ]]; then exit 7; fi\n')
        self.f.env['PYTHON'] = self.f.posix(path)
        r = self.f.run_stage('--worker', 'start')
        self.assertEqual(r.returncode, 0, r.stderr)
        calls = self.f.calls.read_text().splitlines()
        self.assertEqual(len(calls), 2)
        self.assertIn('continuous_audit.py pilot ', calls[0]); self.assertIn('continuous_audit.py run ', calls[1])
        self.f.calls.unlink(); self.f.env['DREX_TEST_FAIL_PILOT'] = '1'
        r = self.f.run_stage('--worker', 'start')
        self.assertEqual(r.returncode, 7, r.stderr)
        self.assertEqual(len(self.f.calls.read_text().splitlines()), 1)


if __name__ == '__main__': unittest.main()
