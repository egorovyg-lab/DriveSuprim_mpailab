import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import clearance_bridge as bridge
import clearance_remote as remote
import clearance_summary as display
from transfer import read, write, sha
import test_runtime as shell_fixtures

AVAILABLE = bool(importlib.util.find_spec('numpy') and importlib.util.find_spec('shapely'))
if AVAILABLE:
    import numpy as np
    from shapely.geometry import box, Polygon
    import clearance_math as math


@unittest.skipUnless(AVAILABLE, 'NumPy/Shapely required for actual polygon tests')
class GeometryTests(unittest.TestCase):
    def coords(self, x0, y0, x1, y1):
        return np.array([[x1,y1], [x0,y1], [x0,y0], [x1,y0], [(x0+x1)/2,(y0+y1)/2]])

    def lane(self):
        return dict(score=1., required_samples=20, counters=[0,0], first_time_s=None)

    def test_union_removes_internal_seam_from_the_clearance(self):
        road = math.road_union([box(0,0,10,10), box(10,0,20,10)])
        value = math.frame_measurement(self.coords(9,4,11,6), road, [])
        self.assertTrue(value['full_footprint_covered'])
        self.assertAlmostEqual(value['road_clearance_m'], 4)
        self.assertIsNone(value['actor_clearance_m'])

    def test_whole_body_catches_a_hole_that_all_four_corners_miss(self):
        road = math.road_union([Polygon([(0,0),(20,0),(20,20),(0,20)],
                                       [[(9,9),(11,9),(11,11),(9,11)]])])
        value = math.frame_measurement(self.coords(8,8,12,12), road, [])
        self.assertGreater(value['min_corner_signed_margin_m'], 0)
        self.assertFalse(value['full_footprint_covered'])
        self.assertIsNone(value['road_clearance_m'])
        self.assertAlmostEqual(value['outside_area_m2'], 4)
        summary = math.summarize([value,value], .1, True, self.lane())
        self.assertFalse(any(summary['sampled_geometry_eligible_at_buffer_m'].values()))

    def test_boundary_contact_does_not_override_the_original_corner_failure(self):
        value = math.frame_measurement(self.coords(0,4,2,6), math.road_union([box(0,0,20,20)]), [])
        self.assertTrue(value['full_footprint_covered']); self.assertEqual(value['road_clearance_m'], 0)
        summary = math.summarize([value,value], .1, False, self.lane())
        self.assertFalse(summary['sampled_geometry_eligible_at_buffer_m']['0.0'])

    def test_outside_area_and_negative_corner_distance_have_distinct_units(self):
        value = math.frame_measurement(self.coords(-.2,4,2,6), math.road_union([box(0,0,20,20)]), [])
        self.assertIsNone(value['road_clearance_m'])
        self.assertAlmostEqual(value['min_corner_signed_margin_m'], -.2)
        self.assertAlmostEqual(value['outside_area_m2'], .4)
        summary = math.summarize([value,value], .1, False, self.lane())
        self.assertEqual(summary['first_full_footprint_outside_time_s'], 0)
        self.assertFalse(summary['initial_full_footprint_covered'])

    def test_traffic_distance_and_touch_are_reported_without_fault_classification(self):
        road = math.road_union([box(-20,-20,20,20)])
        coords = self.coords(0,0,2,2)
        separated = math.frame_measurement(coords, road, [('object',box(4,0,6,2))])
        touching = math.frame_measurement(coords, road, [('object',box(2,0,4,2))])
        self.assertEqual(separated['actor_clearance_m'], 2)
        self.assertEqual(touching['actor_clearance_m'], 0)
        self.assertEqual(touching['intersecting_actor_tokens'], ['object'])
        summary = math.summarize([separated,touching], .1, True, self.lane())
        self.assertEqual(summary['first_actor_contact_time_s'], .1)
        self.assertNotIn('no_at_fault_collisions', summary)
        self.assertFalse(any(summary['sampled_geometry_eligible_at_buffer_m'].values()))

    def test_fixed_buffers_and_lane_rule_require_all_constraints(self):
        frame = math.frame_measurement(self.coords(.2,4,2,6), math.road_union([box(0,0,20,20)]), [])
        result = math.summarize([frame,frame], .1, True, self.lane())['sampled_geometry_eligible_at_buffer_m']
        self.assertEqual(result, {'0.0':True, '0.1':True, '0.25':False, '0.5':False})
        lane = dict(self.lane(), score=0., counters=[19,20], first_time_s=.1)
        result = math.summarize([frame,frame], .1, True, lane)
        self.assertFalse(any(result['sampled_geometry_eligible_at_buffer_m'].values()))
        self.assertEqual(result['max_lane_counter_samples'], 20)

    def test_invalid_polygons_and_nonfinite_coordinates_are_not_silently_repaired(self):
        with self.assertRaises(ValueError): math.road_union([])
        with self.assertRaises(ValueError): math.road_union([Polygon([(0,0),(1,1),(0,1),(1,0)])])
        with self.assertRaises(ValueError): math.frame_measurement(np.full((5,2), np.nan), box(0,0,1,1), [])


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=str(Path.cwd())); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.source = self.root/'runs/rollout_audit_fixture'
        self.build_source()

    def build_source(self, success=True):
        dump = lambda value: json.dumps(value, indent=2).encode()
        request = self.source/'audit_request.json'; write(request, dict(protocol='fixture'))
        files = {'identity.json':dump(dict(request_sha256=sha(request))),
                 'report.json':dump(dict(status='complete' if success else 'parity_failed', candidates=12,
                                         metric_parity_passed=12, geometry_interpretation_ready=success))}
        for pair in range(1,7):
            for member in (0,1):
                folder = f'case{pair:02d}-member{member}'
                files[folder+'/summary.json'] = dump(dict(pair_id=f'case{pair:02d}', member=member))
                files[folder+'/traces.npz'] = b'committed replay states'
                files[folder+'/commit.json'] = dump(dict(files={name:hashlib.sha256(files[folder+'/'+name]).hexdigest()
                    for name in ('summary.json','traces.npz')}))
        self.manifest = {n:dict(size=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in files.items()}
        files['artifact_manifest.json'] = dump(self.manifest)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as archive:
            for name,value in files.items():
                info = tarfile.TarInfo(name); info.size=len(value); archive.addfile(info,io.BytesIO(value))
        data=buffer.getvalue(); digest=hashlib.sha256(data).hexdigest()
        (self.source/('audit-'+digest+'.tar')).write_bytes(data)
        self.result=self.source/('result_'+digest[:12])
        for name,value in files.items():
            path=self.result/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(value)
        write(self.source/'latest_result.json',dict(result=str(self.result),archive_sha256=digest,remote_exit_code=0))

    def test_all_12_committed_candidates_are_bound_without_writes(self):
        before={p:p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        request=bridge.source_snapshot(self.root,'rollout_audit_fixture')
        self.assertEqual(len(request['source_files']),38)
        remote.verify_source(self.result,request['source_files'])
        for path,value in before.items(): self.assertEqual(path.read_bytes(),value)
        with self.assertRaisesRegex(ValueError,'Incomplete source audit bindings'):
            remote.verify_source(self.result,{})

    def test_changed_trace_and_failed_parity_stop_before_cache_loading(self):
        (self.result/'case02-member0/traces.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'Source audit artifact changed'):
            bridge.source_snapshot(self.root,'rollout_audit_fixture')
        self.build_source(success=False)
        with self.assertRaisesRegex(RuntimeError,'Original metric parity'):
            bridge.source_snapshot(self.root,'rollout_audit_fixture')

    def test_isolated_upload_is_small_and_compatible_with_astra_python39(self):
        request=bridge.source_snapshot(self.root,'rollout_audit_fixture')
        out=self.root/'new'; write(out/'audit_request.json',request)
        data=bridge.input_archive(out); self.assertLess(len(data),4*1024**2)
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            self.assertEqual(len(archive.getmembers()),len(bridge.TOOLS)+2)
        for name in bridge.TOOLS:
            ast.parse(Path(bridge.__file__).with_name(name).read_text(encoding='utf-8'), feature_version=(3,9))
        # The bridge imports only stdlib modules; SDKs and Shapely are remote-only.
        imports={node.names[0].name.split('.')[0] for node in ast.walk(ast.parse(Path(bridge.__file__).read_text()))
                 if isinstance(node,ast.Import)}
        self.assertNotIn('numpy',imports); self.assertNotIn('shapely',imports)

    def test_failure_report_is_visible_in_compact_status(self):
        value=display.show(dict(status='technical_failure',error='fixture'))
        self.assertIn('technical_failure',value); self.assertIn('fixture',value)


class ShellTests(unittest.TestCase):
    def test_status_requires_neither_gpu_nor_slurm_and_output_cannot_replace_source(self):
        f=shell_fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        f.setUp(); self.addCleanup(f.doCleanups)
        f.script=Path(__file__).with_name('run_clearance_audit.sh').resolve()
        f.env.pop('DREX_CLEARANCE_NAME',None); f.env.pop('DREX_CLEARANCE_SOURCE',None)
        result=f.run_stage('status'); self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(f.log.exists())
        f.env['DREX_CLEARANCE_NAME']='rollout_audit_pair6_v2'
        self.assertNotEqual(f.run_stage('start').returncode,0)


if __name__=='__main__': unittest.main()
