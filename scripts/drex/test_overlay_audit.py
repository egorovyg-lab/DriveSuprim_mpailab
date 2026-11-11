import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import test_runtime as slurm_fixtures

AVAILABLE = bool(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'))
if AVAILABLE:
    import numpy as np
    import audit_overlay_projection as audit
    import overlay_data as view
    import test_overlay as fixtures
    from runtime import read


@unittest.skipUnless(AVAILABLE, 'NumPy/Pillow needed')
class ProjectionAuditTests(unittest.TestCase):
    def setUp(self):
        self.r = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        self.t = np.array([1.5, 0, 1.5])
        self.k = np.array([[100, 0, 320], [0, 100, 240], [0, 0, 1]], float)

    def test_ground_point_can_fold_into_sky_in_original_renderer(self):
        calib = self.r, self.t, self.k, np.array([-.3, 0, 0, 0, 0])
        # This ground point is below the optical axis before distortion.
        point = np.array([[2.3, 0, 0]])
        uv, valid = view.project(point, calib)
        self.assertTrue(valid[0])
        self.assertTrue(0 <= uv[0, 1] < 240)  # Wrongly appears above the horizon.
        poses = np.zeros((40, 3))
        poses[:, 0] = np.linspace(2., 20., 40)
        stats, segments, _ = audit.inspect_path(poses, calib, (640, 480))
        self.assertGreater(stats['suspect_visible_segments'], 0)
        self.assertTrue(any(s['suspect'] and min(p[1] for p in s['line']) < 240
                            for s in segments))
        self.assertAlmostEqual(audit.radial_branch_limit(calib[3]), 1/.9)

    def test_first_fold_is_kept_even_when_later_branch_has_positive_jacobian(self):
        distortion = np.array([-1., .1, 0, 0, 0])
        self.assertAlmostEqual(audit.radial_branch_limit(distortion), 3-np.sqrt(7))
        poses = np.zeros((40, 3))
        poses[:, 0] = 1.5 + 1.5/3.1
        _, _, trace = audit.inspect_path(poses, (self.r, self.t, self.k, distortion), (640, 480))
        j = -1
        self.assertGreater(trace['radial'][j], 0)
        self.assertGreater(trace['radial_slope'][j], 0)
        self.assertGreater(trace['jacobian_det'][j], 0)
        self.assertTrue(trace['original_in_frame'][j])
        self.assertTrue(trace['suspect'][j])

    def test_zero_distortion_behind_and_near_camera_paths_are_not_invented(self):
        calib = self.r, self.t, self.k, np.zeros(5)
        self.assertTrue(np.isinf(audit.radial_branch_limit(calib[3])))
        poses = np.zeros((40, 3))
        poses[:, 0] = np.linspace(0, 20, 40)
        stats, _, _ = audit.inspect_path(poses, calib, (640, 480))
        self.assertEqual(stats['suspect_visible_segments'], 0)
        self.assertGreater(stats['original_visible_segments'], 0)
        poses[:, 0] = np.linspace(0, 1, 40)
        stats, _, _ = audit.inspect_path(poses, calib, (640, 480))
        self.assertEqual(stats['original_visible_segments'], 0)

    def test_audit_preserves_frozen_run_and_does_not_read_labels(self):
        fixture = fixtures.OverlayProtocolTests('test_five_images_and_matched_white_photo_control')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        name = 'overlay_audit_test'
        destination = fixture.f.root/'runs'/name
        destination.parent.mkdir()
        fixture.out.rename(destination)
        package = fixture.f.root/'data/navsim_train_v1/package'
        package.parent.mkdir(parents=True)
        fixture.f.package.rename(package)
        before = {p.relative_to(destination): p.read_bytes()
                  for p in destination.rglob('*') if p.is_file()}
        with patch.object(audit.pilot.old, 'load_labels', side_effect=AssertionError('Labels')) as labels:
            report = audit.run(fixture.f.root, name, preview_scenes=1)
            labels.assert_not_called()
        self.assertEqual(report['status'], 'diagnostic_only')
        self.assertEqual(report['scenes'], 4)
        self.assertFalse(report['outcomes_read'])
        for relative, value in before.items():
            self.assertEqual((destination/relative).read_bytes(), value)
        self.assertFalse((destination/'results').exists())
        self.assertFalse((destination/'decisions.json').exists())
        rows = read(destination/'projection_audit_v1/per_scene.json')
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(list((destination/'projection_audit_v1').glob('*.png'))), 4)


class ProjectionAuditSlurmTests(unittest.TestCase):
    def test_audit_is_queued_cpu_only_without_login_venv(self):
        fixture = slurm_fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.script = Path(__file__).with_name('run_overlay_audit.sh').resolve()
        (fixture.envroot/'setup_node.txt').write_text('cn29\n')
        fixture.stub('sbatch', 'printf "%s\\n" "$@" > "$DREX_TEST_LOG"\ncat >/dev/null\necho 900\n')
        result = fixture.run_stage('submit')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = fixture.log.read_text().splitlines()
        self.assertIn('--gres=none', args)
        self.assertIn('--nodelist=cn29', args)
        self.assertNotIn('--gres=gpu:1', args)
        self.assertFalse((fixture.envroot/'venv').exists())
        result = fixture.run_stage('--inside')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('actual Slurm job step', result.stderr)


if __name__ == '__main__':
    unittest.main()
