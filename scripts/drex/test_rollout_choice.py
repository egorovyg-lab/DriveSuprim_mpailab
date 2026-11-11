from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import unittest
from unittest.mock import patch

import test_runtime as slurm_fixtures

AVAILABLE = bool(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'))
if AVAILABLE:
    import numpy as np
    import rollout_choice as choice
    import rollout_choice_data as view
    import test_pair_diagnostic as fixtures
    from runtime import read, write
    from transfer import sha


@unittest.skipUnless(AVAILABLE, 'NumPy/Pillow needed')
class GeometryTests(unittest.TestCase):
    def test_full_path_inset_keeps_all_four_time_markers_inside_the_image(self):
        poses = np.zeros((2, 40, 3)); poses[:, :, 0] = np.linspace(2.5, 100, 40)
        paths = {k: poses.copy() for k in view.GEOMETRIES}
        image = np.asarray(view.shared_inset(paths, 'nominal', 0, 'centerline'))
        orange = (image == [204, 85, 0]).all(axis=2)
        rows = np.flatnonzero(orange.any(axis=1))
        groups = 0 if not len(rows) else 1+int((np.diff(rows) > 1).sum())
        self.assertEqual(groups, 4, 'The full-path inset must show all 1/2/3/4-second markers')

    def test_tracking_error_separates_time_lag_from_lateral_error_and_wraps_heading(self):
        reference = np.zeros((41, 3)); reference[:, 0] = np.arange(41)*.5
        executed = reference.copy(); executed[:, 0] -= 2
        result = view.tracking_error(reference, executed)
        self.assertEqual(result['max_abs_longitudinal_error_m'], 2)
        self.assertEqual(result['max_abs_lateral_error_m'], 0)
        self.assertEqual(result['endpoint_longitudinal_error_m'], -2)
        reference[:, 2] = np.pi/2; executed = reference.copy(); executed[:, 0] += 1
        executed[:, 2] += 2*np.pi
        result = view.tracking_error(reference, executed)
        self.assertAlmostEqual(result['max_abs_longitudinal_error_m'], 0)
        self.assertAlmostEqual(result['endpoint_lateral_error_m'], -1)
        self.assertAlmostEqual(result['max_abs_heading_error_deg'], 0)

    def test_96_requests_partition_without_duplicates_for_all_supported_gpu_counts(self):
        pairs = {s['pair_id']: s for s in choice.prior.SPECS}
        tasks = choice.tasks_for(pairs)
        self.assertEqual(len(tasks), 96)
        for count in (1, 2, 4, 8):
            groups = [tasks[i::count] for i in range(count)]
            self.assertEqual(len({t['request_id'] for group in groups for t in group}), 96)
            self.assertEqual(sum(map(len, groups)), 96)
            self.assertEqual(len({len(g) for g in groups}), 1)


@unittest.skipUnless(AVAILABLE, 'NumPy/Pillow needed')
class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.PairProtocolTests('test_blind_inputs_controls_order_and_source_preservation')
        with redirect_stdout(io.StringIO()): self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.root, self.package, self.pair = self.f.root, self.f.package, self.f.out
        with redirect_stdout(io.StringIO()):
            for i, assignments in enumerate(self.f.plan['assignments']):
                worker = self.pair/'shards'/f'{i:02d}'
                tasks = [t for t in self.f.tasks if t['request_id'] in assignments]
                choice.prior.infer_tasks(self.package, worker, tasks, *self.f.inputs,
                                        lambda _: dict(probabilities=[.7, .3], seconds=.1))
                write(worker/'environment.json', dict(fixture='same'))
            choice.prior.merge(self.root, self.package, self.pair, self.f.plan, self.f.tasks, self.f.pairs, self.f.inputs)
        self.audit = self.root/'runs/rollout_audit_fixture'
        self.fake_audit()
        self.old = {p: p.read_bytes() for root in (self.pair, self.audit) for p in root.rglob('*') if p.is_file()}
        self.out = self.root/'runs/rollout_choice_fixture'

    def fake_audit(self, parity=True):
        candidates = []; files = {}
        truth = {t['pair_id']: t for t in read(self.pair/'expected.json')}
        dump = lambda v: json.dumps(v, indent=2, allow_nan=False).encode()
        for pair_id, pair in self.f.pairs.items():
            for member in (0, 1):
                sample = self.f.inputs[2][pair['token']]
                i = sample['pool_local'].index(pair['local_ids'][member])
                reference = np.vstack([np.zeros((1, 3)), sample['trajectories'][i]])
                executed = reference.copy(); executed[:, 1] += np.linspace(0, .4, 41)
                candidate = dict(pair_id=pair_id, member=member, token=pair['token'],
                    candidate_local=pair['local_ids'][member], candidate_vocab=pair['vocab_ids'][member],
                    metric_keys=truth[pair_id]['metric_keys'], expected_raw=truth[pair_id]['raw_scores'][member])
                candidates.append(candidate)
                summary = dict(pair_id=pair_id, member=member, token=pair['token'], candidate_vocab=candidate['candidate_vocab'],
                    parity_passed=True, samples=41, dt_s=.1,
                    expected_raw=dict(zip(candidate['metric_keys'], candidate['expected_raw'])))
                folder = pair_id+'-member'+str(member)
                files[folder+'/summary.json'] = dump(summary)
                buffer = io.BytesIO()
                np.savez_compressed(buffer, nominal_local=reference, simulated_local=executed,
                                    forbidden_outcomes=np.array([dict(secret='must not deserialize')], dtype=object))
                files[folder+'/traces.npz'] = buffer.getvalue()
                files[folder+'/commit.json'] = dump(dict(files={name: hashlib.sha256(files[folder+'/'+name]).hexdigest()
                    for name in ('summary.json', 'traces.npz')}))
        request = dict(protocol='nominal_rollout_audit_v1', candidates=candidates,
            package_manifest_sha256=sha(self.package/'package_manifest.json'), pair_plan_sha256=sha(self.pair/'plan.json'),
            pair_decision_commit_sha256=sha(self.pair/'decision_commit.json'), vehicle_profile=read(Path(view.__file__).with_name('vehicle_profile_astra.json')))
        write(self.audit/'audit_request.json', request)
        files['identity.json'] = dump(dict(request_sha256=sha(self.audit/'audit_request.json')))
        files['report.json'] = dump(dict(status='complete' if parity else 'parity_failed', metric_parity_passed=len(candidates),
                                        candidates=len(candidates), geometry_interpretation_ready=parity))
        manifest = {n: dict(size=len(v), sha256=hashlib.sha256(v).hexdigest()) for n, v in files.items()}
        files['artifact_manifest.json'] = dump(manifest)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as archive:
            for name, value in files.items():
                info = tarfile.TarInfo(name); info.size = len(value); archive.addfile(info, io.BytesIO(value))
        content = buffer.getvalue(); digest = hashlib.sha256(content).hexdigest()
        archive = self.audit/('audit-'+digest+'.tar'); archive.write_bytes(content)
        self.result = self.audit/('result_'+digest[:12])
        for name, value in files.items():
            path = self.result/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
        write(self.audit/'latest_result.json', dict(result=str(self.result), archive_sha256=digest, remote_exit_code=0))

    def prepare(self):
        with redirect_stdout(io.StringIO()):
            self.summary = choice.freeze_plan(self.root, self.package, self.pair, self.audit, self.out, 2)
        self.plan = choice.checked_plan(self.root, self.package, self.out)
        self.tasks = read(self.out/'requests.json'); self.inputs = choice.inputs_for(self.package, self.out)

    def test_preparation_uses_only_pose_arrays_and_preserves_sources(self):
        self.prepare()
        self.assertEqual(self.summary['requests'], 32)
        for pair_id in self.f.pairs:
            with np.load(self.out/'paths'/(pair_id+'.npz'), allow_pickle=False) as arrays:
                self.assertEqual(set(arrays.files), {'nominal', 'simulated'})
        for path, content in self.old.items(): self.assertEqual(path.read_bytes(), content)
        self.assertTrue((self.out/'tracking_error_report.json').exists())
        pair_id = next(iter(self.f.pairs)); pair = self.f.pairs[pair_id]; token = pair['token']
        manifest, scenes, samples, _, paths = self.inputs
        rows = []
        for geometry in view.GEOMETRIES:
            rows.append(view.payload(self.package, scenes[token], manifest, samples[token], pair, paths[pair_id],
                                     geometry, 'footprint', 'road', 'canonical'))
        self.assertEqual(rows[0][0]['state'], rows[1][0]['state'])
        self.assertEqual(rows[0][0]['question'], rows[1][0]['question'])
        self.assertEqual(rows[0][1], rows[1][1]); self.assertNotEqual(rows[0][2], rows[1][2])
        self.assertEqual(rows[0][0]['images'][0].tobytes(), rows[1][0]['images'][0].tobytes())
        text = str({k: v for k, v in rows[0][0].items() if k != 'images'}).lower()
        for word in ('oracle', 'baseline', 'pdm', 'expected', 'nominal', 'simulated', 'focus_metric', token):
            self.assertNotIn(word, text)

    def test_identical_paths_produce_identical_pixels_and_hash_for_both_conditions(self):
        self.prepare()
        manifest, scenes, samples, pairs, paths = self.inputs
        pair_id = next(iter(pairs)); pair = pairs[pair_id]; token = pair['token']
        identical = {k: paths[pair_id]['nominal'].copy() for k in view.GEOMETRIES}
        outputs = [view.payload(self.package, scenes[token], manifest, samples[token], pair, identical,
                   geometry, 'centerline', 'blank', 'reversed') for geometry in view.GEOMETRIES]
        self.assertEqual(outputs[0][2], outputs[1][2])

    def test_worker_is_blind_and_resumes_then_merge_requires_every_shard(self):
        self.prepare(); original_read = choice.read; original_load = np.load
        def guarded_read(path):
            if Path(path).name in ('expected.json', 'summary.json', 'audit_request.json'):
                raise AssertionError('Privileged metadata deserialization on worker')
            return original_read(path)
        def guarded_load(path, *args, **kwargs):
            if Path(path).name == 'traces.npz': raise AssertionError('Audit traces on worker')
            return original_load(path, *args, **kwargs)
        calls = []
        def fill(index):
            worker = self.out/'shards'/f'{index:02d}'
            subset = [t for t in self.tasks if t['request_id'] in self.plan['assignments'][index]]
            with patch.object(choice, 'read', side_effect=guarded_read), patch.object(np, 'load', side_effect=guarded_load):
                inputs = choice.inputs_for(self.package, self.out)
                choice.infer_tasks(self.package, worker, subset, inputs,
                                   lambda _: (calls.append(1) or dict(probabilities=[.7, .3], seconds=.1)))
                choice.infer_tasks(self.package, worker, subset, inputs, lambda _: self.fail('Repeated inference'))
            write(worker/'environment.json', dict(fixture='same'))
        with redirect_stdout(io.StringIO()):
            fill(0)
            with patch.object(choice, 'score_committed', side_effect=AssertionError('Early scoring')):
                with self.assertRaises(FileNotFoundError): choice.merge(self.package, self.out, self.plan, self.tasks, self.inputs)
            fill(1)
            report = choice.merge(self.package, self.out, self.plan, self.tasks, self.inputs)
        self.assertEqual(len(calls), len(self.tasks))
        self.assertEqual(report['status'], 'complete'); self.assertEqual(len(report['conditions']), 8)
        self.assertEqual(report['requests'], 32); self.assertEqual(report['new_scorer_calls'], 0)
        self.assertEqual(report['reference_policies']['always_keep_DS_correct'], 1)
        self.assertEqual(report['reference_policies']['always_choose_challenger_correct'], 1)
        self.assertEqual(len(report['paired_changes']), 8)

    def test_tampered_trace_and_failed_audit_are_rejected(self):
        path = self.result/'case01-member0/traces.npz'; path.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Audit artifact changed'):
            choice.freeze_plan(self.root, self.package, self.pair, self.audit, self.out, 2)
        self.fake_audit(parity=False)
        with self.assertRaisesRegex(RuntimeError, 'failed metric parity'):
            choice.freeze_plan(self.root, self.package, self.pair, self.audit, self.out, 2)

    def test_changed_sanitized_paths_are_rejected_before_inference(self):
        self.prepare()
        path = self.out/'paths/case01.npz'; path.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'frozen input changed'):
            choice.checked_plan(self.root, self.package, self.out)


class SlurmTests(unittest.TestCase):
    def setUp(self):
        self.f = slurm_fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.script = Path(__file__).with_name('run_rollout_choice.sh').resolve()
        (self.f.envroot/'setup_node.txt').write_text('cn29\n')
        for key in ('DREX_ROLLOUT_CHOICE_NAME', 'DREX_ROLLOUT_PAIR_SOURCE', 'DREX_ROLLOUT_AUDIT_NAME'):
            self.f.env.pop(key, None)
        self.f.env['DREX_TEST_PREFIX'] = self.f.posix(self.f.root/'batch')
        self.f.stub('sbatch', 'n=0; [[ ! -f "$DREX_TEST_PREFIX.count" ]] || n=$(cat "$DREX_TEST_PREFIX.count")\n'
                    'n=$((n+1)); echo "$n" > "$DREX_TEST_PREFIX.count"\n'
                    'printf "%s\\n" "$@" > "$DREX_TEST_PREFIX-$n.args"\n'
                    'cat > "$DREX_TEST_PREFIX-$n.sh"\necho "$((800+n))"\n')

    def test_pipeline_supports_1_2_4_8_gpus_and_afterok_dependencies(self):
        for count in ('1', '2', '4', '8'):
            result = self.f.run_stage('submit', count)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('rollout_choice_gpu'+count+'_v1', result.stdout)
        worker = (self.f.root/'batch-11.args').read_text().splitlines()
        self.assertIn('--array=0-7%8', worker); self.assertIn('--gres=gpu:1', worker)
        merge = (self.f.root/'batch-12.args').read_text().splitlines()
        self.assertIn('--dependency=afterok:811', merge); self.assertIn('--gres=none', merge)
        self.assertNotEqual(self.f.run_stage('submit', '3').returncode, 0)

    def test_status_needs_no_compute_and_old_output_name_is_rejected(self):
        result = self.f.run_stage('status', '4'); self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.f.log.exists())
        self.f.env['DREX_ROLLOUT_CHOICE_NAME'] = 'pair_diag_gpu4_v1'
        self.assertNotEqual(self.f.run_stage('submit', '4').returncode, 0)


if __name__ == '__main__': unittest.main()
