import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import rollout_bridge as bridge
from transfer import sha, write, read
import test_runtime as slurm_fixtures

AVAILABLE = bool(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'))
if AVAILABLE:
    import numpy as np
    from PIL import Image
    import rollout_audit_math as geometry
    import rollout_replay as replay


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=str(Path.cwd()))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fake_pair_run(self):
        pair = self.root/'runs/pair_diag_fake'
        pair.mkdir(parents=True)
        package = self.root/'data/navsim_train_v1/package'; package.mkdir(parents=True)
        files = {}; pairs = []; truth = []; tasks = []
        for i in range(6):
            token = 'token'+str(i); pair_id = 'case'+str(i)
            files['bundle/examples/'+token+'.npz'] = dict(sha256='a'*64)
            pairs.append(dict(pair_id=pair_id, token=token, local_ids=[0, 1], vocab_ids=[100, 101]))
            truth.append(dict(pair_id=pair_id, token=token, focus_metric='lane_keeping', better_member=0,
                              raw_scores=[[1.]*9, [1.]*9], metric_keys=['pdm_score']+['m'+str(j) for j in range(8)]))
            for j in range(8):
                name = pair_id+'-'+str(j)
                tasks.append(dict(request_id=name))
                write(pair/'results'/(name+'.json'), dict(request_id=name))
        write(package/'package_manifest.json', dict(files=files))
        write(pair/'pair_inputs.json', pairs); write(pair/'expected.json', truth); write(pair/'requests.json', tasks)
        write(pair/'decisions.json', [])
        write(pair/'plan.json', dict(sources={}, previews={},
            files={name: sha(pair/name) for name in ('pair_inputs.json', 'expected.json', 'requests.json')},
            package_manifest_sha256=sha(package/'package_manifest.json')))
        write(pair/'decision_commit.json', dict(plan_sha256=sha(pair/'plan.json'),
            decisions_sha256=sha(pair/'decisions.json'), expected_sha256=sha(pair/'expected.json'),
            results={p.name: sha(p) for p in (pair/'results').glob('*.json')}))
        write(pair/'report.json', dict(status='complete', requests=48, cases=6))
        return pair

    def test_request_preserves_all_12_candidates_and_rejects_partial_or_changed_source(self):
        pair = self.fake_pair_run()
        before = {p.relative_to(pair): p.read_bytes() for p in pair.rglob('*') if p.is_file()}
        request = bridge.prepare_request(self.root, pair, self.root/'audit')
        self.assertEqual(len(request['candidates']), 12)
        self.assertEqual({r['member'] for r in request['candidates']}, {0, 1})
        for name, content in before.items():
            self.assertEqual((pair/name).read_bytes(), content)
        write(pair/'report.json', dict(status='partial', requests=48, cases=6))
        with self.assertRaisesRegex(RuntimeError, 'Completed six-case'):
            bridge.prepare_request(self.root, pair, self.root/'other')

    def test_output_archive_members_and_checksums(self):
        archive = self.root/'out.tar'
        content = json.dumps(dict(status='complete')).encode()
        manifest = {'report.json': dict(size=len(content), sha256=hashlib.sha256(content).hexdigest())}
        with tarfile.open(archive, mode='w') as target:
            for name, value in [('artifact_manifest.json', json.dumps(manifest).encode()), ('report.json', content)]:
                info = tarfile.TarInfo(name); info.size = len(value); target.addfile(info, io.BytesIO(value))
        self.assertEqual(bridge.extract_result(archive, self.root/'result')['status'], 'complete')
        (self.root/'result/report.json').write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'artifact changed'):
            bridge.extract_result(archive, self.root/'result')

    def test_result_archive_rejects_path_escape_and_symlinks(self):
        for name, symbolic in [('../outside.json', False), ('link', True)]:
            archive = self.root/'bad.tar'
            manifest = {name: dict(size=1, sha256=hashlib.sha256(b'x').hexdigest())}
            with tarfile.open(archive, mode='w') as target:
                data = json.dumps(manifest).encode(); info = tarfile.TarInfo('artifact_manifest.json'); info.size = len(data)
                target.addfile(info, io.BytesIO(data))
                info = tarfile.TarInfo(name)
                if symbolic:
                    info.type = tarfile.SYMTYPE; info.linkname = '/tmp/outside'
                    target.addfile(info)
                else:
                    info.size = 1; target.addfile(info, io.BytesIO(b'x'))
            with self.assertRaises(ValueError):
                bridge.extract_result(archive, self.root/'bad_result')
        self.assertFalse((self.root/'outside.json').exists())

    def test_staged_tools_parse_on_astra_python39_and_remote_helpers_are_self_contained(self):
        for name in bridge.TOOLS:
            ast.parse(Path(bridge.__file__).with_name(name).read_text(encoding='utf-8'), feature_version=(3, 9))
        for function in (bridge.receive_archive, bridge.result_archive):
            compile(bridge.remote_python(function), '<remote>', 'exec')
        pair = self.fake_pair_run(); out = self.root/'audit'
        bridge.prepare_request(self.root, pair, out)
        data = bridge.input_archive(out)
        self.assertLess(len(data), 4*1024**2)
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            self.assertEqual(len(archive.getmembers()), len(bridge.TOOLS)+2)


@unittest.skipUnless(AVAILABLE, 'NumPy/Pillow needed')
class GeometryTests(unittest.TestCase):
    def bound_fixture(self):
        import test_real as fixtures
        f = fixtures.RealPilotTests('test_input_allowlist_never_deserializes_labels_or_teacher_heads')
        f.setUp(); self.addCleanup(f.doCleanups)
        repo = f.root/'repo'
        source_hashes = {}
        for name in replay.REPLAY_SOURCES:
            path = repo/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture source')
            source_hashes[name] = sha(path)
        provider = f.root/'provider'; provider.mkdir()
        cache = provider/'cache.xz'; cache.write_bytes(b'committed cache bytes')
        write(provider/'cache_index.json', {'token0': dict(path=str(cache), sha256=sha(cache))})
        feature = provider/'features/token0.npz'; feature.parent.mkdir()
        feature.write_bytes((f.package/'bundle/examples/token0.npz').read_bytes())
        template = dict(scorer={}, simulator={}, proposal_sampling={}, traffic_agents='non_reactive',
                        traffic_agents_policy={}, agent=dict(config=dict(trajectory_sampling={})), sources=source_hashes)
        write(provider/'source_config.json', template)
        return f, repo, provider, cache, feature

    def bind_package(self, f, bindings):
        identity = read(f.package/'bundle/identity.json')
        identity['inputs'] = bindings
        write(f.package/'bundle/identity.json', identity)
        manifest = read(f.package/'bundle/bundle_manifest.json')
        manifest['identity_sha256'] = sha(f.package/'bundle/identity.json'); write(f.package/'bundle/bundle_manifest.json', manifest)
        portable = read(f.package/'package_manifest.json')
        for name in ('bundle/identity.json', 'bundle/bundle_manifest.json'):
            path = f.package/name; portable['files'][name] = dict(size=path.stat().st_size, sha256=sha(path))
        write(f.package/'package_manifest.json', portable)

    def test_bound_cache_and_configuration_resolution_rejects_changed_cache(self):
        f, repo, provider, cache, feature = self.bound_fixture()
        self.bind_package(f, {str(p): sha(p) for p in (provider/'cache_index.json', provider/'source_config.json', feature)})
        *_, context = replay.bound_context(f.package, repo, [dict(token='token0')])
        self.assertEqual(context['token0']['path'], str(cache))
        self.assertTrue(context['token0']['matching_source_feature'])
        cache.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Metric cache changed'):
            replay.bound_context(f.package, repo, [dict(token='token0')])

    def test_legacy_cache_is_resolved_through_hashed_identity_inputs_without_changing_sources(self):
        f, repo, provider, cache, feature = self.bound_fixture()
        write(provider/'identity.json', read(provider/'source_config.json'))
        previous = f.root/'action_transfer/identity.json'
        write(previous, dict(source=str(provider), inputs={name: sha(provider/name)
              for name in ('identity.json', 'cache_index.json')}))
        intermediate = f.root/'decoder/identity.json'
        write(intermediate, dict(inputs={str(previous): sha(previous)}))
        # A directly bound newer index does not contain this legacy scene.
        direct = f.root/'new/cache_index.json'; write(direct, {})
        self.bind_package(f, {str(p): sha(p) for p in (intermediate, direct, feature)})
        watched = (previous, intermediate, provider/'identity.json', provider/'cache_index.json', cache, feature)
        before = {p: p.read_bytes() for p in watched}
        *_, context = replay.bound_context(f.package, repo, [dict(token='token0')])
        self.assertEqual(context['token0']['path'], str(cache))
        self.assertTrue(context['token0']['matching_source_feature'])
        chain = context['token0']['cache_index_binding_chain']
        self.assertEqual([r['path'] for r in chain], [str(p.resolve()) for p in
                         (intermediate, previous, provider/'cache_index.json')])
        self.assertEqual(chain[-1]['sha256'], sha(provider/'cache_index.json'))
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        previous.write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'Frozen upstream metadata changed'):
            replay.bound_context(f.package, repo, [dict(token='token0')])

    def test_cache_index_is_not_trusted_just_because_it_exists_beside_a_config(self):
        f, repo, provider, _, feature = self.bound_fixture()
        self.bind_package(f, {str(p): sha(p) for p in (provider/'source_config.json', feature)})
        with self.assertRaisesRegex(FileNotFoundError, 'No bound cache index contains token0'):
            replay.bound_context(f.package, repo, [dict(token='token0')])

    def test_nested_cache_index_hash_and_conflicting_bindings_are_checked(self):
        f, _, provider, _, _ = self.bound_fixture()
        first = f.root/'a/identity.json'; second = f.root/'b/identity.json'
        index = provider/'cache_index.json'
        write(first, dict(source=str(provider), inputs={'cache_index.json': sha(index)}))
        write(second, dict(inputs={str(index): '0'*64}))
        with self.assertRaisesRegex(RuntimeError, 'Conflicting frozen input hashes'):
            replay.bound_metadata({str(p): sha(p) for p in (first, second)})
        index.write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'Frozen upstream metadata changed'):
            replay.bound_metadata({str(first): sha(first)})
        write(first, dict(inputs={'cache_index.json': sha(index)}))
        with self.assertRaisesRegex(ValueError, 'no absolute source root'):
            replay.bound_metadata({str(first): sha(first)})

    def test_local_coordinates_and_heading_wrap(self):
        origin = np.array([10., 20., np.pi/2])
        result = geometry.local_poses(np.array([[10., 21., -3*np.pi/2]]), origin)
        np.testing.assert_allclose(result, [[1., 0., 0.]], atol=1e-12)
        value = geometry.trajectory_difference(result, result+np.array([0, 0, 2*np.pi]))
        self.assertAlmostEqual(value['max_heading_error_rad'], 0.)

    def test_dac_reports_actual_pdm_corner_order_and_includes_initial_sample(self):
        inside = np.ones((4, 4), dtype=bool); inside[0, 1] = False
        trace = geometry.dac_trace(inside, .1)
        self.assertEqual(trace['first_index'], 0)
        self.assertEqual(trace['first_outside_corners'], ['REAR_LEFT'])
        self.assertEqual(trace['score'], 0.)

    def test_lane_reproduces_sample_count_skip_without_reset_and_strict_threshold(self):
        trace = geometry.lane_trace([.6, .6, .9, .6], [False, False, True, False], .1, .5, .3)
        self.assertEqual(trace['required_samples'], 3)
        self.assertEqual(trace['first_index'], 3)
        self.assertEqual(trace['counters'], [1, 2, 2, 3])
        self.assertEqual(trace['first_run_start_time_s'], 0)
        self.assertAlmostEqual(trace['first_time_s'], .3)
        reset = geometry.lane_trace([.6, .5, .6], [False]*3, .1, .5, .2)
        self.assertEqual(reset['score'], 1.)
        with self.assertRaises(ValueError):
            geometry.lane_trace([np.nan], [False], .1, .5, 2.)

    def test_positive_depth_is_not_confused_with_visible_pixel_or_occlusion(self):
        r = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        calib = r, np.array([1.5, 0, 1.5]), np.array([[100, 0, 320], [0, 100, 240], [0, 0, 1]], float), np.zeros(5)
        near = geometry.point_projection([1.8, 0], calib, (640, 480))
        self.assertTrue(near['admissible_ray']); self.assertFalse(near['projects_inside_current_photo'])
        far = geometry.point_projection([11.5, 0], calib, (640, 480))
        self.assertTrue(far['projects_inside_current_photo']); self.assertFalse(far['occlusion_tested'])

    def test_pair_summary_compares_the_same_corner_at_the_recorded_event_time(self):
        temp = tempfile.TemporaryDirectory(dir=str(Path.cwd())); self.addCleanup(temp.cleanup)
        root = Path(temp.name); photo = root/'photo.png'; Image.new('RGB', (640, 480), 'gray').save(photo)
        request = dict(candidates=[dict(pair_id='case01', member=i, token='token0', better_member=1,
                                        focus_metric='drivable_area_compliance') for i in (0, 1)])
        rows = [dict(pair_id='case01', member=i, parity_passed=True,
                     focus_event=dict(index=1, time_s=.1, point_kind='FRONT_RIGHT')) for i in (0, 1)]
        for i in (0, 1):
            poses = np.zeros((3, 3)); poses[:, 0] = [0, 11.5, 20]; poses[:, 1] = i
            boxes = np.repeat(poses[:, None, :2], 5, axis=1)
            folder = root/('case01-member'+str(i)); folder.mkdir()
            np.savez(folder/'traces.npz', nominal_local=poses, simulated_local=poses,
                     nominal_boxes_local=boxes, simulated_boxes_local=boxes)
        calibration = dict(sensor2lidar_rotation=[[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                           sensor2lidar_translation=[1.5, 0, 1.5],
                           cam_intrinsic=[[100, 0, 320], [0, 100, 240], [0, 0, 1]], distortion=[0]*5)
        scenes = {'token0': dict(current_camera_calibrations=dict(cam_f0=calibration))}
        with patch.object(replay.base, 'camera_paths', return_value=[photo]*6):
            rows = replay.pair_summaries(root, root, request, rows, scenes, {})
        self.assertAlmostEqual(rows[0]['max_nominal_path_difference_m'], 1.)
        self.assertAlmostEqual(rows[0]['nominal_between_candidates_projection']['separation_pixels'], 10.)
        self.assertFalse(rows[0]['nominal_between_candidates_projection']['occlusion_tested'])

    def test_audit_figures_render_without_sdk_and_config_core_rejects_unrelated_metadata(self):
        self.assertIsNone(replay.config_core(dict(agent='not a config')))
        temp = tempfile.TemporaryDirectory(dir=str(Path.cwd())); self.addCleanup(temp.cleanup)
        out = Path(temp.name)
        poses = np.zeros((41, 3)); poses[:, 0] = np.linspace(0, 20, 41)
        boxes = np.zeros((41, 5, 2)); boxes[:, :, 0] = poses[:, 0, None]
        boxes[:, :4, 1] = [1., 1., -1., -1.]
        r = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        calib = r, np.array([1.5, 0, 1.5]), np.array([[100, 0, 320], [0, 100, 240], [0, 0, 1]], float), np.zeros(5)
        event = dict(index=20, points_local=boxes[20, :1].tolist())
        geometry.camera_figure(Image.new('RGB', (640, 480), 'gray'), calib, poses, poses, boxes, boxes,
                               event, 'fixture', out/'camera.png')
        geometry.map_figure(poses, poses, boxes, [], poses[:, :2], event, [], 'fixture', out/'map.png')
        with Image.open(out/'camera.png') as image: self.assertEqual(image.size, (960, 640))
        with Image.open(out/'map.png') as image: self.assertEqual(image.size, (960, 960))


class RoutingTests(unittest.TestCase):
    def test_status_uses_login_python_without_gpu_or_slurm_and_rejects_source_run_output(self):
        fixture = slurm_fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.script = Path(__file__).with_name('run_rollout_audit.sh').resolve()
        fixture.env.pop('DREX_ROLLOUT_AUDIT_NAME', None)
        fixture.env.pop('DREX_ROLLOUT_PAIR_SOURCE', None)
        result = fixture.run_stage('status')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(fixture.log.exists())
        fixture.env['DREX_ROLLOUT_AUDIT_NAME'] = 'pair_diag_gpu4_v1'
        self.assertNotEqual(fixture.run_stage('start').returncode, 0)


if __name__ == '__main__':
    unittest.main()
