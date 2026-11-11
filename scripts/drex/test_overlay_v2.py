import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import test_runtime as slurm_fixtures

AVAILABLE = bool(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'))
if AVAILABLE:
    import numpy as np
    import overlay_data as v1
    import overlay_v2_data as view
    import overlay_v2_pilot as pilot
    import test_overlay as fixtures
    from night import load_result, save_result
    from runtime import read, write
    from transfer import sha


@unittest.skipUnless(AVAILABLE, 'NumPy/Pillow needed')
class CentralLensDomainTests(unittest.TestCase):
    def setUp(self):
        self.r = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        self.t = np.array([1.5, 0, 1.5])
        self.k = np.array([[100, 0, 320], [0, 100, 240], [0, 0, 1]], float)

    def test_ground_foldback_is_rejected_and_visible_road_path_is_retained(self):
        calib = self.r, self.t, self.k, np.array([-.3, 0, 0, 0, 0])
        point = [[2.3, 0, 0]]
        old_uv, old_valid = v1.project(point, calib)
        self.assertTrue(old_valid[0])
        self.assertLess(old_uv[0, 1], 240)
        uv, valid = view.project(point, calib)
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(uv).all())
        poses = np.zeros((40, 3))
        poses[:, 0] = np.linspace(.1, 20., 40)
        old = v1.path_segments(poses, calib, (640, 480))
        corrected = view.path_segments(poses, calib, (640, 480))
        self.assertTrue(any(min(p[1] for p in s) < 240 for s in old))
        self.assertTrue(corrected)
        self.assertTrue(all(min(p[1] for p in s) > 240 for s in corrected))
        self.assertAlmostEqual(view.lens_domain(tuple(calib[3])), 1/np.sqrt(.9))

    def test_outer_positive_branch_cannot_be_reentered(self):
        distortion = np.array([-1., .1, 0, 0, 0])
        calib = self.r, self.t, self.k, distortion
        point = [[1.5+1.5/3.1, 0, 0]]
        self.assertTrue(v1.project(point, calib)[1][0])
        self.assertFalse(view.project(point, calib)[1][0])
        self.assertAlmostEqual(view.lens_domain(tuple(distortion)), np.sqrt(3-np.sqrt(7)))

    def test_tangential_bound_covers_the_entire_disk_not_just_this_ray(self):
        distortion = (0., 0., 0., .1)
        calib = np.eye(3), np.zeros(3), np.eye(3), np.array(distortion)
        self.assertAlmostEqual(view.lens_domain(distortion), 1/.6)
        uv, valid = view.project([[1., 0, 1.], [2., 0, 1.]], calib)
        self.assertEqual(valid.tolist(), [True, False])
        np.testing.assert_allclose(uv[0], [1.3, 0])

    def test_rational_pole_and_nonfinite_inputs_fail_closed(self):
        distortion = (0., 0., 0., 0., 0., -2., 0., 0.)
        self.assertAlmostEqual(view.lens_domain(distortion), 1/np.sqrt(2))
        calib = np.eye(3), np.zeros(3), np.eye(3), np.array(distortion)
        self.assertEqual(view.project([[.6, 0, 1], [.8, 0, 1]], calib)[1].tolist(), [True, False])
        with self.assertRaises(ValueError):
            view.project([[np.nan, 0, 1]], calib)
        with self.assertRaises(ValueError):
            view.lens_domain((0, 0, 0))

    def test_no_distortion_preserves_projection_and_stationary_visibility(self):
        calib = self.r, self.t, self.k, np.zeros(5)
        points = [[11.5, 0, 0], [11.5, 1, 0], [1, 0, 0]]
        old_uv, old_valid = v1.project(points, calib)
        uv, valid = view.project(points, calib)
        np.testing.assert_array_equal(old_valid, valid)
        np.testing.assert_allclose(old_uv[valid], uv[valid])
        self.assertTrue(np.isinf(view.lens_domain(tuple(calib[3]))))
        poses = np.zeros((40, 3))
        self.assertEqual(view.path_segments(poses, calib, (640, 480)), [])

    def test_segments_never_join_across_rejected_points(self):
        points = np.zeros((40, 3))
        uv = np.zeros((360, 2))
        uv[0], uv[2] = [10, 10], [500, 400]
        valid = np.zeros(360, dtype=bool)
        valid[[0, 2]] = True
        with patch.object(view, 'project', return_value=(uv, valid)):
            self.assertEqual(view.path_segments(points, None, (640, 480)), [])

    def test_independent_finite_difference_jacobian_is_positive_inside_domain(self):
        distortion = np.array([-.25, .04, .005, -.003, -.001])
        radius = min(view.lens_domain(tuple(distortion))*.95, 3.)
        calib = np.eye(3), np.zeros(3), np.eye(3), distortion
        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            for fraction in (.1, .5, .95):
                xy = radius*fraction*np.array([np.cos(angle), np.sin(angle)])
                columns = []
                for axis in range(2):
                    step = np.zeros(2); step[axis] = 1e-6
                    # Independent unconstrained v1 formula supplies the finite differences.
                    plus = v1.project([[*(xy+step), 1]], calib)[0][0]
                    minus = v1.project([[*(xy-step), 1]], calib)[0][0]
                    columns.append((plus-minus)/(2e-6))
                jacobian = np.column_stack(columns)
                self.assertGreater(np.linalg.eigvalsh(jacobian).min(), 0)
                self.assertTrue(view.project([[*xy, 1]], calib)[1][0])


@unittest.skipUnless(AVAILABLE, 'NumPy/Pillow needed')
class OverlayV2ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.OverlayProtocolTests('test_five_images_and_matched_white_photo_control')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.old_bytes = {p.relative_to(self.f.out): p.read_bytes()
                          for p in self.f.out.rglob('*') if p.is_file()}
        self.out = self.f.f.root/'v2'
        self.plan, self.manifest, self.records, self.scenes, self.vocab = pilot.prepare(
            self.f.f.root, self.f.f.package, self.out, 2)
        self.samples = pilot.samples_for(self.f.f.package, self.manifest, self.records, self.vocab)

    def preflight(self):
        return pilot.projection_preflight(self.f.f.package, self.out, self.manifest,
                                         self.records, self.scenes, self.vocab)

    def test_preflight_is_input_only_preserves_v1_and_commits_diagnostics(self):
        with patch.object(pilot.old, 'load_labels', side_effect=AssertionError('Early outcomes')) as labels:
            report = self.preflight()
            pilot.check_plan(self.out)
            labels.assert_not_called()
        self.assertEqual(report['retained_legacy_suspect_segments'], 0)
        self.assertFalse(report['visual_alignment_verified'])
        self.assertEqual(report['scenes'], 4)
        self.assertEqual(len(list(self.out.glob('preview-*.png'))), 20)
        for filename, value in self.old_bytes.items():
            self.assertEqual((self.f.out/filename).read_bytes(), value)
        report['retained_legacy_suspect_segments'] = 1
        write(self.out/'projection_report.json', report)
        with self.assertRaisesRegex(RuntimeError, 'diagnostics changed'):
            pilot.check_plan(self.out)

    def test_matching_blank_control_and_candidate_ids_are_preserved(self):
        token = self.records[0]['token']; sample = self.samples[token]
        road, order, h = view.payload(self.f.f.package, self.scenes[token], self.manifest,
                                      sample, 'overlay', 'canonical')
        blank, other, hb = view.payload(self.f.f.package, self.scenes[token], self.manifest,
                                        sample, 'blank', 'canonical')
        self.assertEqual(order, other)
        self.assertEqual(road['state'], blank['state'])
        self.assertEqual(road['question'], blank['question'])
        self.assertEqual(view.ids(token), v1.ids(token))
        self.assertNotEqual(h, hb)
        self.assertEqual(len(road['images']), 5)
        self.assertEqual(len(road['question']['criteria']), 4)

    def test_preflight_rejects_changed_preview_instead_of_repairing_it(self):
        self.preflight()
        path = next(self.out.glob('preview-*.png'))
        path.write_bytes(b'changed preview')
        with self.assertRaisesRegex(RuntimeError, 'frozen preview changed'):
            self.preflight()
        self.assertEqual(path.read_bytes(), b'changed preview')

    def test_v1_cached_answers_are_rejected_for_changed_v2_pixels_and_text(self):
        token = self.records[0]['token']; sample = self.samples[token]
        _, order, old_hash = v1.payload(self.f.f.package, self.scenes[token], self.manifest,
                                       sample, 'overlay', 'canonical')
        _, _, new_hash = view.payload(self.f.f.package, self.scenes[token], self.manifest,
                                      sample, 'overlay', 'canonical')
        self.assertNotEqual(old_hash, new_hash)
        save_result(self.out/'results'/pilot.name(token, 'overlay', 'canonical'), dict(
            status='ok', k=4, token=token, mode='overlay', ordering='canonical',
            pool_positions=order, request_sha256=old_hash, probabilities=[.1, .2, .3, .4]))
        with self.assertRaises(RuntimeError):
            pilot.infer_subset(self.f.f.package, self.out, self.manifest, self.records,
                               self.scenes, self.samples, lambda _: self.fail('Inference on changed cache'))

    def test_partial_resume_and_full_commit_precede_all_outcome_reads(self):
        self.preflight(); calls = []
        def infer(row):
            calls.append(1)
            return dict(probabilities=[.1, .2, .3, .4], seconds=.1)
        status = pilot.infer_subset(self.f.f.package, self.out, self.manifest, self.records,
                                    self.scenes, self.samples, infer, max_new=5)
        self.assertEqual(status['status'], 'partial')
        with patch.object(pilot.old, 'load_labels', side_effect=AssertionError('Early labels')) as labels:
            with self.assertRaises(FileNotFoundError):
                pilot.score(self.f.f.package, self.out, self.manifest, self.records, self.samples)
            labels.assert_not_called()
        status = pilot.infer_subset(self.f.f.package, self.out, self.manifest, self.records,
                                    self.scenes, self.samples, infer)
        self.assertEqual(status['status'], 'complete')
        self.assertEqual(len(calls), 36)
        results = {p.name: load_result(p) for p in (self.out/'results').glob('*.json')}
        decisions = pilot.build_decisions(self.records, self.samples, results)
        pilot.base.freeze(self.out/'decisions.json', decisions)
        pilot.base.freeze(self.out/'decision_commit.json', dict(
            plan_sha256=sha(self.out/'plan.json'), decisions_sha256=sha(self.out/'decisions.json'),
            results={n: sha(self.out/'results'/n) for n in results}))
        report = pilot.score(self.f.f.package, self.out, self.manifest, self.records, self.samples)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['projection']['retained_legacy_suspect_segments'], 0)
        self.assertIn('oracle_top16_upper_bound', report['methods'])
        self.assertIn('overlay_mean3_minus_blank_mean3', report['paired_contrasts'])


class OverlayV2SlurmTests(unittest.TestCase):
    def setUp(self):
        self.f = slurm_fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.script = Path(__file__).with_name('run_overlay_v2.sh').resolve()
        (self.f.envroot/'setup_node.txt').write_text('cn29\n')
        self.f.env.pop('DREX_OVERLAY_V2_NAME', None)
        self.f.env['DREX_TEST_PREFIX'] = self.f.posix(self.f.root/'batch')
        self.f.stub('sbatch', 'n=0; [[ ! -f "$DREX_TEST_PREFIX.count" ]] || n=$(cat "$DREX_TEST_PREFIX.count")\n'
                    'n=$((n+1)); echo "$n" > "$DREX_TEST_PREFIX.count"\n'
                    'printf "%s\\n" "$@" > "$DREX_TEST_PREFIX-$n.args"\n'
                    'cat > "$DREX_TEST_PREFIX-$n.sh"\necho "$((800+n))"\n')

    def test_submit_plan_is_one_cpu_job_without_workers_or_login_venv(self):
        result = self.f.run_stage('submit-plan', '4')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = (self.f.root/'batch-1.args').read_text().splitlines()
        self.assertIn('--gres=none', args)
        self.assertFalse((self.f.root/'batch-2.args').exists())
        self.assertIn('overlay_v2_top4_gpu4_v1', result.stdout)
        self.assertFalse((self.f.envroot/'venv').exists())

    def test_full_pipeline_and_all_supported_replica_counts(self):
        for count in ('1', '2', '4', '8'):
            result = self.f.run_stage('submit', count)
            self.assertEqual(result.returncode, 0, result.stderr)
        args = (self.f.root/'batch-11.args').read_text().splitlines()
        self.assertIn('--array=0-7%8', args)
        self.assertIn('--gres=gpu:1', args)
        merge = (self.f.root/'batch-12.args').read_text().splitlines()
        self.assertIn('--dependency=afterok:811', merge)
        self.assertIn('--gres=none', merge)
        self.assertNotEqual(self.f.run_stage('submit', '3').returncode, 0)

    def test_pilot_has_no_merge_and_v1_output_name_is_rejected(self):
        result = self.f.run_stage('submit-pilot', '4')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = (self.f.root/'batch-2.args').read_text().splitlines()
        self.assertIn('--time=00:30:00', args)
        self.assertIn('--dependency=afterok:801', args)
        self.assertFalse((self.f.root/'batch-3.args').exists())
        self.f.env['DREX_OVERLAY_V2_NAME'] = 'overlay_top4_gpu4_v1'
        self.assertNotEqual(self.f.run_stage('submit-plan', '4').returncode, 0)


if __name__ == '__main__':
    unittest.main()
