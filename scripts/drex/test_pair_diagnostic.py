from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import test_runtime as slurm_fixtures

AVAILABLE = bool(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'))
if AVAILABLE:
    import numpy as np
    import pair_diagnostic as diag
    import pair_diagnostic_data as view
    import test_overlay_v2 as fixtures
    from night import save_result
    from runtime import read, write
    from transfer import sha


@unittest.skipUnless(AVAILABLE, 'NumPy/Pillow needed')
class FootprintTests(unittest.TestCase):
    def test_real_dimensions_and_rear_axle_rotation(self):
        d = view.vehicle()
        self.assertEqual(d['width'], 2.297)
        self.assertEqual(d['front_length'], 4.049)
        self.assertEqual(d['rear_length'], 1.127)
        zero = view.corners([[0, 0, 0]], d)[0]
        np.testing.assert_allclose(zero, [[4.049, 1.1485], [4.049, -1.1485],
                                         [-1.127, -1.1485], [-1.127, 1.1485]])
        rotated = view.corners([[10, -2, np.pi/2]], d)[0]
        np.testing.assert_allclose(rotated, zero@np.array([[0, 1], [-1, 0]])+[10, -2])

    def test_vehicle_edge_does_not_bridge_rejected_projection_points(self):
        def reject(points, _):
            uv = np.zeros((len(points), 2)); uv[0] = [1, 1]; uv[2] = [50, 50]
            valid = np.zeros(len(points), dtype=bool); valid[[0, 2]] = True
            return uv, valid
        with patch.object(view.projection, 'project', side_effect=reject):
            self.assertEqual(view.projected_lines(np.array([[0, 0], [.2, 0], [.4, 0]]), None, (100, 100)), [])

    def test_footprint_edges_do_not_restore_the_old_sky_foldback(self):
        r = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        calib = (r, np.array([1.5, 0, 1.5]),
                 np.array([[100, 0, 320], [0, 100, 240], [0, 0, 1]], float),
                 np.array([-.3, 0, 0, 0, 0]))
        poses = np.zeros((40, 3)); poses[:, 0] = np.linspace(.1, 20., 40)
        lines = view.footprint_lines(poses, view.vehicle(), calib, (640, 480))
        self.assertTrue(lines)
        self.assertTrue(all(min(p[1] for p in line) > 240 for line in lines))

    def test_eight_workers_can_partition_the_fixed_48_tasks(self):
        pairs = [dict(pair_id=s['pair_id']) for s in diag.SPECS]
        tasks = diag.tasks_for(pairs)
        self.assertEqual(len(tasks), 48)
        for n in (1, 2, 4, 8):
            groups = [[t['request_id'] for t in tasks[i::n]] for i in range(n)]
            self.assertEqual({t for group in groups for t in group}, {t['request_id'] for t in tasks})
            self.assertEqual(sum(map(len, groups)), 48)
            self.assertEqual(len({len(g) for g in groups}), 1)


@unittest.skipUnless(AVAILABLE, 'NumPy/Pillow needed')
class PairProtocolTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.OverlayV2ProtocolTests('test_matching_blank_control_and_candidate_ids_are_preserved')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.preflight()
        source = diag.source
        results = {}
        for record in self.f.records:
            token = record['token']
            for mode, kind in source.view.TASKS:
                positions = source.view.order(token, kind); p = [0.]*4; p[positions.index(3)] = 1.
                _, _, digest = source.view.payload(self.f.f.f.package, self.f.scenes[token], self.f.manifest,
                                                   self.f.samples[token], mode, kind)
                name = source.name(token, mode, kind)
                item = dict(status='ok', k=4, token=token, mode=mode, ordering=kind,
                            pool_positions=positions, probabilities=p, request_sha256=digest)
                save_result(self.f.out/'results'/name, item); results[name] = item
        decisions = source.build_decisions(self.f.records, self.f.samples, results)
        source.base.freeze(self.f.out/'decisions.json', decisions)
        source.base.freeze(self.f.out/'decision_commit.json', dict(plan_sha256=sha(self.f.out/'plan.json'),
            decisions_sha256=sha(self.f.out/'decisions.json'),
            results={name: sha(self.f.out/'results'/name) for name in results}))
        source.score(self.f.f.f.package, self.f.out, self.f.manifest, self.f.records, self.f.samples)
        self.root = self.f.f.f.root
        self.package = self.root/'data/navsim_train_v1/package'
        self.package.parent.mkdir(parents=True)
        self.f.f.f.package.rename(self.package)
        self.out = self.root/'diagnostic'
        self.specs = [dict(pair_id='case01', token='token0', kind='repair', metric='pdm_score'),
                      dict(pair_id='case02', token='token1', kind='regression', metric='drivable_area_compliance')]
        self.old = {p.relative_to(self.f.out): p.read_bytes() for p in self.f.out.rglob('*') if p.is_file()}
        with redirect_stdout(io.StringIO()):
            self.summary = diag.freeze_plan(self.root, self.package, self.f.out, self.out, 2, self.specs)
        self.plan = diag.checked_plan(self.root, self.package, self.out)
        self.tasks = read(self.out/'requests.json')
        self.inputs = diag.inputs_for(self.package, self.out)
        self.pairs = self.inputs[-1]

    def test_blind_inputs_controls_order_and_source_preservation(self):
        manifest, scenes, samples, pairs = self.inputs
        pair = pairs['case01']; token = pair['token']
        road, positions, hr = view.payload(self.package, scenes[token], manifest, samples[token], pair,
                                           'footprint', 'road', 'canonical')
        blank, other, hb = view.payload(self.package, scenes[token], manifest, samples[token], pair,
                                        'footprint', 'blank', 'canonical')
        self.assertEqual(positions, other)
        self.assertEqual(road['state'], blank['state']); self.assertEqual(road['question'], blank['question'])
        self.assertNotEqual(hr, hb)
        self.assertEqual(len(road['images']), 3)
        self.assertTrue(all(image.size == (768, 640) for image in road['images']))
        names, forward = view.names_and_order(pair, 'canonical')
        other_names, reverse = view.names_and_order(pair, 'reversed')
        self.assertEqual(names, other_names); self.assertEqual(forward[::-1], reverse)
        text = str({k: v for k, v in road.items() if k != 'images'}).lower()
        for forbidden in ('oracle', 'baseline', 'pdm', 'expected', 'teacher_rank', 'case_kind', 'focus_metric', token):
            self.assertNotIn(forbidden, text)
        for filename, content in self.old.items():
            self.assertEqual((self.f.out/filename).read_bytes(), content)
        self.assertTrue(self.summary['selection_uses_known_outcomes'])
        self.assertFalse(self.summary['expected_labels_in_model_inputs'])

    def test_worker_never_deserializes_expected_labels_and_resumes_without_inference(self):
        original = diag.read
        def guarded(path):
            if Path(path).name == 'expected.json': raise AssertionError('Expected labels on worker')
            return original(path)
        worker = self.out/'shards/00'; calls = []
        def infer(row): calls.append(1); return dict(probabilities=[.7, .3], seconds=.1)
        subset = [t for t in self.tasks if t['request_id'] in self.plan['assignments'][0]]
        with patch.object(diag, 'read', side_effect=guarded), \
             patch.object(diag.scoring, 'load_labels', side_effect=AssertionError('Outcomes on worker')), \
             redirect_stdout(io.StringIO()):
            inputs = diag.inputs_for(self.package, self.out)
            diag.infer_tasks(self.package, worker, subset, *inputs, infer)
            diag.infer_tasks(self.package, worker, subset, *inputs,
                             lambda _: self.fail('Repeated finished inference'))
        self.assertEqual(len(calls), len(subset))
        self.assertEqual(read(worker/'inference_status.json')['status'], 'complete')

    def test_incomplete_source_is_rejected_before_new_label_selection(self):
        path = self.f.out/'decision_commit.json'; commit = read(path)
        del commit['results'][next(iter(commit['results']))]; write(path, commit)
        with patch.object(diag.scoring, 'load_labels', side_effect=AssertionError('Premature labels')) as labels:
            with self.assertRaisesRegex(RuntimeError, 'Complete source cohort'):
                diag.verify_source(self.package, self.f.out)
            labels.assert_not_called()

    def test_merge_requires_all_workers_then_scores_complete_blind_decisions(self):
        def fill(index):
            worker = self.out/'shards'/f'{index:02d}'
            subset = [t for t in self.tasks if t['request_id'] in self.plan['assignments'][index]]
            diag.infer_tasks(self.package, worker, subset, *self.inputs,
                             lambda _: dict(probabilities=[.7, .3], seconds=.1))
            write(worker/'environment.json', dict(fixture='same profile'))
        with redirect_stdout(io.StringIO()):
            fill(0)
            with patch.object(diag, 'score_committed', side_effect=AssertionError('Early scoring')) as score:
                with self.assertRaises(FileNotFoundError):
                    diag.merge(self.root, self.package, self.out, self.plan, self.tasks, self.pairs, self.inputs)
                score.assert_not_called()
            fill(1)
            report = diag.merge(self.root, self.package, self.out, self.plan, self.tasks, self.pairs, self.inputs)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['requests'], 16)
        self.assertEqual(len(report['conditions']), 4)
        self.assertEqual(report['reference_policies']['always_keep_DS_correct'], 1)
        self.assertEqual(report['reference_policies']['always_choose_challenger_correct'], 1)
        for value in report['conditions'].values():
            self.assertEqual(value['cases'], 2)
            self.assertEqual(value['order_choice_flips'], 2)
        self.assertTrue((self.out/'decision_commit.json').exists())

    def test_changed_expected_file_and_changed_request_are_rejected(self):
        task = dict(self.tasks[0], request_sha256='0'*64)
        with self.assertRaisesRegex(RuntimeError, 'request/pixels changed'):
            diag.infer_tasks(self.package, self.out/'worker', [task], *self.inputs,
                             lambda _: self.fail('Inference on changed request'))
        path = self.out/'expected.json'; expected = read(path); expected[0]['better_member'] = 0; write(path, expected)
        with self.assertRaisesRegex(RuntimeError, 'frozen input changed'):
            diag.checked_plan(self.root, self.package, self.out)


class PairSlurmTests(unittest.TestCase):
    def setUp(self):
        self.f = slurm_fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.script = Path(__file__).with_name('run_pair_diagnostic.sh').resolve()
        (self.f.envroot/'setup_node.txt').write_text('cn29\n')
        for key in ('DREX_PAIR_DIAG_NAME', 'DREX_PAIR_SOURCE'):
            self.f.env.pop(key, None)
        self.f.env['DREX_TEST_PREFIX'] = self.f.posix(self.f.root/'batch')
        self.f.stub('sbatch', 'n=0; [[ ! -f "$DREX_TEST_PREFIX.count" ]] || n=$(cat "$DREX_TEST_PREFIX.count")\n'
                    'n=$((n+1)); echo "$n" > "$DREX_TEST_PREFIX.count"\n'
                    'printf "%s\\n" "$@" > "$DREX_TEST_PREFIX-$n.args"\n'
                    'cat > "$DREX_TEST_PREFIX-$n.sh"\necho "$((800+n))"\n')

    def test_cpu_plan_and_full_pipeline_for_1_2_4_8_gpus(self):
        result = self.f.run_stage('submit-plan', '4')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--gres=none', (self.f.root/'batch-1.args').read_text())
        self.assertFalse((self.f.root/'batch-2.args').exists())
        self.assertFalse((self.f.envroot/'venv').exists())
        for count in ('1', '2', '4', '8'):
            result = self.f.run_stage('submit', count)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('pair_diag_gpu'+count+'_v1', result.stdout)
        worker = (self.f.root/'batch-12.args').read_text().splitlines()
        self.assertIn('--array=0-7%8', worker); self.assertIn('--gres=gpu:1', worker)
        merge = (self.f.root/'batch-13.args').read_text().splitlines()
        self.assertIn('--dependency=afterok:812', merge); self.assertIn('--gres=none', merge)
        self.assertNotEqual(self.f.run_stage('submit', '3').returncode, 0)

    def test_old_output_names_and_missing_slurm_step_are_rejected(self):
        self.f.env['DREX_PAIR_DIAG_NAME'] = 'overlay_v2_top4_gpu4_v1'
        self.assertNotEqual(self.f.run_stage('submit', '4').returncode, 0)
        self.f.env.pop('DREX_PAIR_DIAG_NAME')
        result = self.f.run_stage('--inside', 'worker', '4')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('actual Slurm job step', result.stderr)


if __name__ == '__main__':
    unittest.main()
