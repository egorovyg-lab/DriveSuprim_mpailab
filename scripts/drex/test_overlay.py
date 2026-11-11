import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import test_runtime as slurm_fixtures

AVAILABLE=bool(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'))
if AVAILABLE:
    import numpy as np
    from PIL import Image
    import overlay_data as view
    import overlay_pilot as pilot
    import real_data as base
    import test_real as fixtures
    from runtime import read,write
    from transfer import sha


@unittest.skipUnless(AVAILABLE,'NumPy/Pillow needed')
class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.r=np.array([[0,0,1],[-1,0,0],[0,-1,0]],float)
        self.t=np.array([1.5,0,1.5]); self.k=np.array([[100,0,320],[0,100,240],[0,0,1]],float)
        self.c=(self.r,self.t,self.k,np.zeros(5))

    def test_forward_left_height_and_camera_origin(self):
        uv,valid=view.project([[11.5,0,0],[11.5,1,0],[1,0,0]],self.c)
        np.testing.assert_allclose(uv[:2],[[320,255],[310,255]])
        self.assertEqual(valid.tolist(),[True,True,False])

    def test_distortion_matches_analytic_radial_and_tangential_case(self):
        c=(np.eye(3),np.zeros(3),np.eye(3),np.array([.1,0,.02,.03]))
        uv,ok=view.project([[.2,.1,1]],c)
        rr=.05; x=.2*1.005+2*.02*.2*.1+.03*(rr+2*.2**2)
        y=.1*1.005+.02*(rr+2*.1**2)+2*.03*.2*.1
        np.testing.assert_allclose(uv,[[x,y]]); self.assertTrue(ok[0])

    def test_calibration_refuses_wrong_axes_nonrotations_and_unsupported_distortion(self):
        c=dict(sensor2lidar_rotation=self.r.tolist(),sensor2lidar_translation=self.t.tolist(),
               cam_intrinsic=self.k.tolist(),distortion=[0]*5)
        scene=dict(current_camera_calibrations=dict(cam_f0=c))
        view.calibration(scene,(640,480))
        for field,value in [('sensor2lidar_rotation',np.eye(3).tolist()),('distortion',[0]*14),
                            ('sensor2lidar_translation',[1.5,0,-1.5])]:
            changed=dict(c,**{field:value})
            with self.assertRaises(ValueError): view.calibration(dict(current_camera_calibrations=dict(cam_f0=changed)),(640,480))

    def test_segment_clipping_and_no_path_behind_camera(self):
        line=view.clipped_segment(np.array([-10.,5]),np.array([20.,5]),10,10)
        np.testing.assert_allclose(line,[[0,5],[9,5]])
        self.assertIsNone(view.clipped_segment(np.array([-10.,-5]),np.array([20.,-5]),10,10))
        poses=np.zeros((40,3)); poses[:,0]=-np.arange(40)
        self.assertEqual(view.path_segments(poses,self.c,(640,480)),[])

    def test_stable_names_and_disjoint_orders(self):
        names=view.ids('token')
        self.assertEqual(set(names.values()),{'TA','TB','TC','TD'})
        a=view.order('token','canonical')
        self.assertEqual(view.order('token','reversed'),a[::-1])
        self.assertNotEqual(view.order('token','permuted'),a)
        self.assertEqual(view.ids('token'),names)


@unittest.skipUnless(AVAILABLE,'NumPy/Pillow needed')
class OverlayProtocolTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.RealPilotTests('test_input_allowlist_never_deserializes_labels_or_teacher_heads')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        r=[[0,0,1],[-1,0,0],[0,-1,0]]
        for scene in self.f.scenes:
            scene['current_camera_calibrations']=dict(cam_f0=dict(sensor2lidar_rotation=r,
                sensor2lidar_translation=[1.5,0,1.5],cam_intrinsic=[[50,0,32],[0,50,16],[0,0,1]],distortion=[0]*5))
        write(self.f.package/'scene_inputs.json',self.f.scenes)
        manifest=read(self.f.package/'package_manifest.json'); path=self.f.package/'scene_inputs.json'
        manifest['files']['scene_inputs.json']=dict(size=path.stat().st_size,sha256=sha(path))
        manifest['total_bytes']=sum(e['size'] for e in manifest['files'].values()); write(self.f.package/'package_manifest.json',manifest)
        original=base.prepare
        self.patch=patch.object(base,'prepare',side_effect=lambda p,r,o:original(p,r,o,sessions=2,per_session=2))
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.out=self.f.root/'overlay'
        self.plan,self.manifest,self.records,self.scenes,self.vocab=pilot.prepare(self.f.root,self.f.package,self.out,2)
        self.samples=pilot.samples_for(self.f.package,self.manifest,self.records,self.vocab)

    def test_five_images_and_matched_white_photo_control(self):
        token=self.records[0]['token']; s=self.samples[token]
        a,order,ha=view.payload(self.f.package,self.scenes[token],self.manifest,s,'overlay','canonical')
        b,other,hb=view.payload(self.f.package,self.scenes[token],self.manifest,s,'blank','canonical')
        self.assertEqual(len(a['images']),5); self.assertEqual(len(a['question']['criteria']),4)
        self.assertEqual(order,other); self.assertEqual(a['state'],b['state']); self.assertEqual(a['question'],b['question'])
        self.assertNotEqual(ha,hb)
        self.assertTrue(all(im.size==(768,432) for im in a['images']))
        self.assertNotIn('label_raw',str(a['state'])); self.assertNotIn('selection_scores',str(a['state']))

    def test_probability_alignment_keeps_top4_and_baseline(self):
        results={}
        for r in self.records:
            for mode,kind in view.TASKS:
                order=view.order(r['token'],kind); p=[0.]*4; p[order.index(2)]=1.
                results[pilot.name(r['token'],mode,kind)]=dict(pool_positions=order,probabilities=p)
        decisions=pilot.build_decisions(self.records,self.samples,results)
        for d in decisions:
            self.assertEqual(d['methods']['overlay_mean3']['candidate_local'],d['pool_local'][2])
            self.assertEqual(d['methods']['baseline']['candidate_local'],d['baseline_local'])

    def test_partial_cohort_cannot_read_labels_and_resumes_without_repeating(self):
        calls=[]
        def infer(row): calls.append(1); return dict(probabilities=[.1,.2,.3,.4],seconds=.1)
        status=pilot.infer_subset(self.f.package,self.out,self.manifest,self.records,self.scenes,self.samples,infer,max_new=5)
        self.assertEqual(status['status'],'partial')
        with patch.object(pilot.old,'load_labels',side_effect=AssertionError('Premature labels')) as labels:
            with self.assertRaises(FileNotFoundError): pilot.score(self.f.package,self.out,self.manifest,self.records,self.samples)
            labels.assert_not_called()
        status=pilot.infer_subset(self.f.package,self.out,self.manifest,self.records,self.scenes,self.samples,infer)
        self.assertEqual(status['status'],'complete'); self.assertEqual(len(calls),36)
        with self.assertRaises(ValueError): pilot.build_decisions(self.records,self.samples,{})
        results={p.name:pilot.load_result(p) for p in (self.out/'results').glob('*.json')}
        decisions=pilot.build_decisions(self.records,self.samples,results)
        base.freeze(self.out/'decisions.json',decisions)
        base.freeze(self.out/'decision_commit.json',dict(plan_sha256=sha(self.out/'plan.json'),
            decisions_sha256=sha(self.out/'decisions.json'),results={n:sha(self.out/'results'/n) for n in results}))
        report=pilot.score(self.f.package,self.out,self.manifest,self.records,self.samples)
        self.assertEqual(report['status'],'complete')
        self.assertEqual(report['methods']['oracle_top4_upper_bound']['gain_macro_pp'],0)
        self.assertGreater(report['methods']['oracle_top16_upper_bound']['gain_macro_pp'],0)
        self.assertIn('overlay_mean3_minus_blank_mean3',report['paired_contrasts'])


class OverlaySlurmTests(unittest.TestCase):
    def setUp(self):
        self.f=slurm_fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.script=Path(__file__).with_name('run_overlay.sh').resolve()
        (self.f.envroot/'setup_node.txt').write_text('cn29\n')
        self.f.env.pop('DREX_OVERLAY_NAME',None)
        self.f.env['DREX_TEST_PREFIX']=self.f.posix(self.f.root/'batch')
        self.f.stub('sbatch','n=0; [[ ! -f "$DREX_TEST_PREFIX.count" ]] || n=$(cat "$DREX_TEST_PREFIX.count")\n'
            'n=$((n+1)); echo "$n" > "$DREX_TEST_PREFIX.count"\nprintf "%s\\n" "$@" > "$DREX_TEST_PREFIX-$n.args"\n'
            'cat > "$DREX_TEST_PREFIX-$n.sh"\necho "$((800+n))"\n')

    def test_two_gpu_pilot_has_two_jobs_and_no_label_merge(self):
        result=self.f.run_stage('submit-pilot','2'); self.assertEqual(result.returncode,0,result.stderr)
        args=(self.f.root/'batch-2.args').read_text().splitlines()
        self.assertIn('--array=0-1%2',args); self.assertIn('--gres=gpu:1',args)
        self.assertIn('--dependency=afterok:801',args); self.assertIn('--time=00:30:00',args)
        self.assertFalse((self.f.root/'batch-3.args').exists())

    def test_full_series_waits_for_all_gpu_workers_then_cpu_merge(self):
        result=self.f.run_stage('submit','2'); self.assertEqual(result.returncode,0,result.stderr)
        args=(self.f.root/'batch-3.args').read_text().splitlines()
        self.assertIn('--dependency=afterok:802',args); self.assertIn('--gres=none',args)
        self.assertFalse((self.f.envroot/'venv').exists())

    def test_supported_gpu_counts_and_invalid_count(self):
        for count in ('1','2','4','8'):
            result=self.f.run_stage('submit-pilot',count); self.assertEqual(result.returncode,0,result.stderr)
        result=self.f.run_stage('submit','3'); self.assertNotEqual(result.returncode,0)


if __name__=='__main__': unittest.main()
