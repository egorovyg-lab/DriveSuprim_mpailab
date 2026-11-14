import ast
import copy
import gzip
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import pool_audit as audit
import pool_remote as remote
import pool_reserve as spec
import pool_transport as transport
import continuous_metrics as cm
import real_data as base
import test_continuous_audit as old_tests
import test_runtime as fixtures
from transfer import write, read, sha


def candidate(token, j):
    return dict(token=token, candidate_local=j, candidate_vocab=100+j, teacher_rank=j+1,
        metric_keys=['pdm_score', *base.METRICS], expected_raw=[1.]*9, example_sha256='a'*64)


def summary(c):
    values = cm.measure(old_tests.states(), old_tests.frames(), .1, 20. if c['candidate_local'] == 0 else 19.5)
    values.update(all_jerk_norm_mps3_rms=1. if c['candidate_local'] == 0 else .6,
        all_accel_lat_mps2_rms=.4, all_accel_lon_mps2_rms=.1,
        all_jerk_norm_mps3_abs_max=4., all_accel_lon_mps2_abs_max=2., all_accel_lat_mps2_abs_max=2.,
        all_yaw_rate_radps_abs_max=.5, all_yaw_accel_radps2_abs_max=.5)
    return dict(token=c['token'], candidate_local=c['candidate_local'], candidate_vocab=c['candidate_vocab'],
        teacher_rank=c['teacher_rank'], measured_raw=c['expected_raw'], metrics=values)


def request(count=5):
    records = [dict(token=f'{i+1:016x}', session_id='session'+str(i), baseline_local=0) for i in range(count)]
    return dict(protocol='continuous_pool_v1', config=spec.CONFIG, metric_config=cm.CONFIG,
        records=records, candidates=[candidate(r['token'], j) for r in records for j in range(64)],
        pilot_tokens=[r['token'] for r in records[:4]], reuse=dict(name='rollout_audit_continuous_v1', candidate_commits={}))


class OracleTests(unittest.TestCase):
    def setUp(self): self.group = [summary(candidate('0000000000000001', j)) for j in range(64)]

    def test_complete_unique_pool_schema_and_pilot(self):
        r = request(); spec.validate(r)
        for kind in ('missing', 'duplicate_rank', 'duplicate_vocab', 'baseline', 'label', 'reuse'):
            a = copy.deepcopy(r)
            if kind == 'missing': a['candidates'].pop()
            if kind == 'duplicate_rank': a['candidates'][1]['teacher_rank'] = 1
            if kind == 'duplicate_vocab': a['candidates'][1]['candidate_vocab'] = 100
            if kind == 'baseline': a['records'][0]['baseline_local'] = 1
            if kind == 'label': a['candidates'][0]['expected_raw'][0] = float('nan')
            if kind == 'reuse': a['reuse']['name'] = '../escape'
            with self.assertRaises(ValueError): spec.validate(a)

    def test_assembly_keeps_all_tail_labels_and_requires_exact_reuse_ids(self):
        r = request(2); samples = {}; labels = {}; manifest = dict(files={})
        for record in r['records']:
            t = record['token']
            samples[t] = dict(baseline_local=0, candidates=[dict(local=j,vocab=100+j) for j in range(64)])
            labels[t] = {j:np.ones(9) for j in range(64)}
            manifest['files']['bundle/examples/'+t+'.npz'] = dict(sha256='a'*64)
        identity = dict(records=r['records'], package_manifest_sha256='c'*64)
        old = dict(records=[dict(token=x['token'],session_id=x['session_id'],methods={'baseline':0}) for x in r['records']],
            candidates=[c for c in r['candidates'] if c['candidate_local']==0], pilot_tokens=r['pilot_tokens'],
            remote_package='/fixture', vehicle_profile={})
        result = audit.assemble(identity,samples,labels,manifest,old,r['reuse'])
        self.assertEqual(len(result['candidates']),128)
        self.assertEqual(result['candidates'][-1]['teacher_rank'],64)
        labels[r['records'][0]['token']][0][0] = .8
        with self.assertRaisesRegex(RuntimeError,'labels/candidate IDs changed'):
            audit.assemble(identity,samples,labels,manifest,old,r['reuse'])

    def test_absolute_and_relative_budgets_both_apply(self):
        c, b = self.group[1], self.group[0]
        self.assertTrue(spec.eligible(c, b, .5, .05, 'geometry_peaks', 'jerk'))
        c['metrics']['route_progress_m'] = 19.49
        self.assertFalse(spec.eligible(c, b, .5, .05, 'geometry_peaks', 'jerk'))
        b['metrics']['route_progress_m'] = 5.; c['metrics']['route_progress_m'] = 4.7
        self.assertFalse(spec.eligible(c, b, 1., .05, 'geometry_peaks', 'jerk'))

    def test_contact_margin_peak_and_other_comfort_regressions_rejected(self):
        for key, value in [('all_actor_contact_time_s', .1), ('all_outside_area_integral_m2s', .1),
            ('all_accel_lat_mps2_abs_max', 3.), ('all_accel_lat_mps2_rms', .5),
            ('all_jerk_norm_mps3_abs_max', 5.), ('all_yaw_rate_radps_abs_max', .6),
            ('all_actor_clearance_min_m', .1), ('late_road_clearance_min_m', .1)]:
            c = copy.deepcopy(self.group[1]); c['metrics'][key] = value
            self.assertFalse(spec.eligible(c, self.group[0], 1., .05, 'geometry_peaks', 'jerk'), key)
            self.assertTrue(spec.eligible(c, self.group[0], 1., .05, 'raw_only', 'jerk'))

    def test_missing_or_partial_actor_data_does_not_become_safety(self):
        for key, value in [('all_actor_clearance_min_m', None), ('all_actor_observed_interval_fraction', .5),
                           ('late_road_clearance_min_m', None)]:
            c = copy.deepcopy(self.group[1]); c['metrics'][key] = value
            self.assertFalse(spec.eligible(c, self.group[0], 1., .05, 'geometry_peaks', 'jerk'))

    def test_raw_safety_history_and_lane_guards(self):
        for name in (*base.SAFETY, 'lane_keeping', 'history_comfort'):
            c = copy.deepcopy(self.group[1]); c['measured_raw'][1+base.METRICS.index(name)] = 0.
            self.assertFalse(spec.eligible(c, self.group[0], 1., .05, 'raw_only', 'jerk'))

    def test_nested_pool_reserve_and_baseline_ties(self):
        self.group[32]['metrics']['all_jerk_norm_mps3_rms'] = .2
        for k in (16, 32): self.assertEqual(spec.pick(self.group, 0, k, 1., .05, 'geometry_peaks', 'jerk')['candidate_local'], 1)
        self.assertEqual(spec.pick(self.group, 0, 64, 1., .05, 'geometry_peaks', 'jerk')['candidate_local'], 32)
        for c in self.group: c['metrics']['all_jerk_norm_mps3_rms'] = 1.
        self.assertEqual(spec.pick(self.group, 0, 64, 1., .05, 'geometry_peaks', 'jerk')['candidate_local'], 0)
        self.assertEqual(len(list(spec.specs())), 48)
        self.assertIn(spec.CONFIG['primary'], dict(spec.specs()))

    def test_offroad_baseline_can_only_switch_to_full_coverage(self):
        b = copy.deepcopy(self.group[0]); c = copy.deepcopy(self.group[1])
        b['metrics']['all_road_clearance_min_m'] = None
        self.assertTrue(spec.eligible(c, b, 1., .05, 'geometry_peaks', 'jerk'))
        c['metrics']['all_road_clearance_min_m'] = None
        self.assertFalse(spec.eligible(c, b, 1., .05, 'geometry_peaks', 'jerk'))


class PipelineTests(unittest.TestCase):
    def setup_fixture(self, directory):
        root = Path(directory); package = root/'package'; repo = root/'repo'; out = root/'out'; source = root/'reuse'
        r = request(); manifest = dict(files={})
        for record in r['records']:
            token = record['token']; path = package/'bundle/examples'/(token+'.npz'); path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, candidate_ids=np.arange(100, 164), label_indices=np.array([0]), label_raw=np.ones((1, 9)))
            manifest['files'][path.relative_to(package).as_posix()] = dict(sha256=sha(path))
            for c in r['candidates']:
                if c['token'] == token: c['example_sha256'] = sha(path)
        write(package/'package_manifest.json', manifest)
        for name in (*remote.previous.replay.REPLAY_SOURCES, *remote.previous.EXTRA_SOURCES):
            p = repo/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('fixture')
        r.update(remote_package=str(package), package_manifest_sha256=sha(package/'package_manifest.json'),
            vehicle_profile=read(Path(__file__).with_name('vehicle_profile_astra.json')), tools={})
        r['reuse']['identity_sha256'] = 'b'*64
        def arrays(): return dict(simulated_states=old_tests.states(), center_states=old_tests.states(), scorer_coords=np.zeros((41, 5, 2)))
        for c in r['candidates']:
            if c['candidate_local'] != 0: continue
            folder = source/'candidates'/remote.previous.candidate_key(c)
            write(folder/'summary.json', summary(c)); write(folder/'frames.json', old_tests.frames())
            np.savez_compressed(folder/'traces.npz', **arrays())
            write(folder/'commit.json', dict(identity_sha256='b'*64, candidate=c,
                files={n:sha(folder/n) for n in ('summary.json', 'frames.json', 'traces.npz')}))
            r['reuse']['candidate_commits'][remote.previous.candidate_key(c)] = sha(folder/'commit.json')
        path = root/'request.json'; write(path, r)
        ctx = (manifest, {}, None, {}, {x['token']:{} for x in r['records']})
        return r, path, out, repo, source, ctx, lambda c,*args: (summary(c), arrays(), old_tests.frames())

    def test_pilot_reuse_full_resume_export_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            r, path, out, repo, source, ctx, evaluate = self.setup_fixture(d)
            with patch('builtins.print'), patch.object(remote.previous.replay, 'bound_context', return_value=ctx), \
                    patch.object(remote.previous.replay, 'config_core', return_value={}), \
                    patch.object(remote, 'reuse_context', return_value=source), \
                    patch.object(remote.previous, 'evaluate', side_effect=evaluate) as scorer:
                self.assertEqual(remote.run(path, out, repo, 'pilot'), 0)
                self.assertEqual(scorer.call_count, 4*63)
                self.assertEqual(read(out/'report.json')['status'], 'pilot_complete')
                self.assertEqual(remote.run(path, out, repo, 'full'), 0)
                report = read(out/'report.json')
                self.assertEqual(report['status'], 'complete'); self.assertEqual(report['metric_parity_passed'], 320)
                self.assertEqual(report['reused_candidates_committed'],5); self.assertEqual(report['new_candidates_committed'],315)
                self.assertEqual(scorer.call_count, 5*63)
                self.assertEqual(report['oracles'][spec.CONFIG['primary']]['meaningful_improvements'], 5)
                with gzip.open(out/'per_candidate.csv.gz', 'rt') as f: self.assertEqual(sum(1 for _ in f), 321)
                self.assertEqual(len(read(out/'oracle_choices.json')), 48*5)
                self.assertEqual(remote.run(path, out, repo, 'full'), 0); self.assertEqual(scorer.call_count, 5*63)
                (out/'scenes'/r['records'][0]['token']/'states.npz').write_bytes(b'changed')
                with self.assertRaises(RuntimeError): remote.run(path, out, repo, 'full')
                self.assertEqual(scorer.call_count, 5*63)

    def test_failure_blocks_retry_and_full_requires_pilot(self):
        with tempfile.TemporaryDirectory() as d:
            _, path, out, repo, source, ctx, _ = self.setup_fixture(d)
            with patch('builtins.print'), patch.object(remote.previous.replay, 'bound_context', return_value=ctx), \
                    patch.object(remote.previous.replay, 'config_core', return_value={}), \
                    patch.object(remote, 'reuse_context', return_value=source), \
                    patch.object(remote.previous, 'evaluate', side_effect=RuntimeError('parity failed')) as scorer:
                with self.assertRaisesRegex(RuntimeError, 'committed pilot'): remote.run(path, out, repo, 'full')
                scorer.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, 'parity failed'): remote.run(path, out, repo, 'pilot')
                with self.assertRaisesRegex(RuntimeError, 'Previous candidate failure'): remote.run(path, out, repo, 'pilot')
                self.assertEqual(scorer.call_count, 1)

    def test_reuse_context_rejects_environment_or_original_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); r = request(2); source = root/r['reuse']['name']; folder = source/'artifacts'
            names = ('continuous_metrics.py','continuous_remote.py','clearance_math.py','rollout_replay.py','real_data.py')
            identity = dict(python='fixture', packages={}, config={}, caches={}, sources={}, tools={n:'a'*64 for n in names})
            write(folder/'identity.json', identity); write(source/'audit_request.json', {'fixture':True})
            write(folder/'report.json', dict(status='complete', metric_parity_passed=1))
            key = remote.previous.candidate_key(r['candidates'][0]); write(folder/'candidates'/key/'commit.json', {'fixture':True})
            r['reuse'].update(identity_sha256=sha(folder/'identity.json'), request_sha256=sha(source/'audit_request.json'),
                report_sha256=sha(folder/'report.json'), candidate_commits={key:sha(folder/'candidates'/key/'commit.json')})
            r['tools'] = identity['tools']
            self.assertEqual(remote.reuse_context(r, identity, root), folder)
            current = copy.deepcopy(identity); current['packages']['numpy'] = 'different'
            with self.assertRaisesRegex(RuntimeError, 'environment/cache/config'): remote.reuse_context(r, current, root)
            (folder/'candidates'/key/'commit.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'commitment differs'): remote.reuse_context(r, identity, root)

    def test_reused_trace_tamper_blocks_new_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            r, path, out, repo, source, ctx, _ = self.setup_fixture(d)
            key = remote.previous.candidate_key(r['candidates'][0]); (source/'candidates'/key/'traces.npz').write_bytes(b'changed')
            with patch('builtins.print'), patch.object(remote.previous.replay, 'bound_context', return_value=ctx), \
                    patch.object(remote.previous.replay, 'config_core', return_value={}), \
                    patch.object(remote, 'reuse_context', return_value=source), patch.object(remote.previous, 'evaluate') as scorer:
                with self.assertRaisesRegex(RuntimeError, 'Completed candidate file changed'): remote.run(path, out, repo, 'pilot')
                scorer.assert_not_called()


class PackagingTests(unittest.TestCase):
    def test_python39_and_staged_imports(self):
        for n in (*audit.TOOLS, 'pool_audit.py', 'pool_transport.py'):
            if n.endswith('.py'): ast.parse(Path(__file__).with_name(n).read_text(encoding='utf-8'), feature_version=(3, 9))
        with tempfile.TemporaryDirectory() as d:
            for n in audit.TOOLS: shutil.copyfile(Path(__file__).with_name(n), Path(d)/n)
            p = subprocess.run([sys.executable, '-B', '-c', 'import pool_remote'], cwd=d, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)

    def test_compressed_request_contains_all_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); full = request(651)
            rng = np.random.default_rng(1)
            for c in full['candidates']:
                c['expected_raw'][0] = float(rng.random()); c['expected_raw'][5] = float(rng.random())
            write(p/'audit_request.json', full)
            value = transport.upload(p, audit.TOOLS)
            with tarfile.open(fileobj=io.BytesIO(value), mode='r:gz') as a:
                r = json.load(a.extractfile('audit_request.json'))
                self.assertEqual(len(r['candidates']), 41664)
                self.assertEqual(set(a.getnames()), {'audit_request.json', 'upload_manifest.json', *('tools/'+n for n in audit.TOOLS)})
            self.assertLess(len(value), 32*1024**2)

    def test_remote_archive_rejects_traversal_before_write(self):
        import hashlib
        buf = io.BytesIO()
        manifest = json.dumps({'../escape':dict(size=1, sha256=hashlib.sha256(b'x').hexdigest())}).encode()
        with tarfile.open(fileobj=buf, mode='w:gz') as a:
            for name, value in [('upload_manifest.json', manifest), ('../escape', b'x')]:
                t = tarfile.TarInfo(name); t.size = len(value); a.addfile(t, io.BytesIO(value))
        payload = buf.getvalue()
        code = transport.old.remote_python(transport.receive_archive)
        code += f'\nreceive_archive("/workspace/exp_v2/drex_rollout_audit/test_pool_escape",{hashlib.sha256(payload).hexdigest()!r})'
        p = subprocess.run([sys.executable, '-B', '-c', code], input=payload, capture_output=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn(b'Unsafe upload path', p.stderr)

    def test_compact_export_excludes_states_and_verifies_checksums(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); destination = root/'rollout_audit_test'; out = destination/'artifacts'
            write(out/'report.json', dict(status='pilot_complete')); write(out/'identity.json', dict(fixture=True))
            (out/'scenes').mkdir(); (out/'scenes/states.npz').write_bytes(b'large private state block')
            real_path = Path
            def mapped(p): return root if p == '/workspace/exp_v2/drex_rollout_audit' else real_path(p)
            with patch.object(transport, 'Path', side_effect=mapped), patch('builtins.print'):
                transport.result_archive(str(destination))
            with tarfile.open(destination/'artifacts.tar') as a:
                self.assertEqual(set(a.getnames()), {'artifact_manifest.json','report.json','identity.json'})
            result = root/'download'; value = transport.old.extract_result(destination/'artifacts.tar',result)
            self.assertEqual(value['status'],'pilot_complete')
            self.assertEqual(sha(result/'identity.json'),sha(out/'identity.json'))


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.script = Path(__file__).with_name('run_pool_audit.sh').resolve()
        (self.f.envroot/'setup_node.txt').write_text('cn29\n')
        for n in ('DREX_POOL_NAME','DREX_POOL_SOURCE','DREX_POOL_REUSE'): self.f.env.pop(n, None)

    def test_prepare_cpu_and_checks_exit_zero(self):
        p = self.f.run_stage('prepare'); self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('--gres=none', self.f.log.read_text().splitlines())
        self.f.fake_setup(); p = self.f.run_stage('--inside','check'); self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn('pool_audit.py prepare', self.f.calls.read_text())

    def test_pilot_precedes_full_and_failure_blocks_full(self):
        self.f.stub('flock', 'exit 0\n')
        p = self.f.stub('pool-python', 'printf "%s\\n" "$*" >> "$DREX_TEST_CALLS"\n'
            'if [[ ${DREX_TEST_FAIL_PILOT:-} == 1 && "$*" == *"pool_audit.py pilot "* ]]; then exit 7; fi\n')
        self.f.env['PYTHON'] = self.f.posix(p)
        value = self.f.run_stage('--worker','start'); self.assertEqual(value.returncode,0,value.stderr)
        calls = self.f.calls.read_text().splitlines(); self.assertEqual(len(calls),2)
        self.assertIn('pool_audit.py pilot ',calls[0]); self.assertIn('pool_audit.py run ',calls[1])
        self.f.calls.unlink(); self.f.env['DREX_TEST_FAIL_PILOT']='1'
        value = self.f.run_stage('--worker','start'); self.assertEqual(value.returncode,7,value.stderr)
        self.assertEqual(len(self.f.calls.read_text().splitlines()),1)


if __name__ == '__main__': unittest.main()
