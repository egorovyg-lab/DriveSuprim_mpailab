"""CPU contract tests; no Drex weights, network, CUDA or real data required."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import test_runtime as slurm_fixtures

AVAILABLE = bool(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'))
if AVAILABLE:
    import numpy as np
    from PIL import Image
    import table_data as data
    import table_pilot as pilot
    import table_score as scorer
    import table_labels as label_bridge
    import inventory_trainval as inventory
    import real_data as base
    import test_real as fixtures
    from runtime import write,read
    from transfer import sha


@unittest.skipUnless(AVAILABLE,'NumPy and Pillow required; run run_table.sh check on the compute node')
class TableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.out = self.root/'run'; self.package = self.root/'package'
        self.capture = redirect_stdout(io.StringIO()); self.capture.__enter__()
        self.addCleanup(self.capture.__exit__,None,None,None)
        self.vocab = np.zeros((300,40,3),dtype=np.float32)
        self.vocab[:,:,0] = np.arange(1,41)*.2
        # Vocabulary IDs differ from candidate-local IDs.
        self.arrays = dict(candidate_ids=np.roll(np.arange(256),37),selection_scores=np.arange(256,dtype=float),
            selected_local=np.asarray(17),ego_status=np.array([1,0,0,0,2,0,0,0],dtype=float))
        self.arrays['selection_scores'][17] = 1000.
        for k in (*base.METRICS,'imi'): self.arrays['head_'+k] = np.linspace(-2,2,256)
        self.sample = data.inputs(self.arrays,self.vocab)
        self.photo = Image.new('RGB',(64,32),'red')
        self.records = [dict(token='token0',session_id='session0',role='train'),
                        dict(token='token1',session_id='session1',role='train')]

    def fixture(self):
        entries = {}; index = {}; files = {}
        for i in range(2): write(self.out/'shards'/f'{i:02d}'/'environment.json',dict(gpu='fixture',python='fixture'))
        for r in self.records:
            token = r['token']
            write(self.out/'inputs'/(token+'.json'),self.sample)
            photo = self.out/'photos'/(token+'.png'); photo.parent.mkdir(parents=True,exist_ok=True); self.photo.save(photo)
            entries[token] = dict(inputs_sha256=sha(self.out/'inputs'/(token+'.json')),
                photo_sha256=sha(photo),baseline_local=17,candidates=self.sample['candidates'])
            relative = 'bundle/examples/'+token+'.npz'; p = self.package/relative; p.parent.mkdir(parents=True,exist_ok=True)
            indices = [c['local'] for c in self.sample['candidates'][:16]]
            raw = np.ones((16,9)); raw[:,0]=.8
            np.savez_compressed(p,label_indices=indices,label_raw=raw)
            index[token] = dict(metadata=r); files[relative] = dict(sha256=sha(p),size=p.stat().st_size)
        write(self.package/'package_manifest.json',dict(files=files))
        identity = dict(config=data.CONFIG,cohort='all',records=self.records,replicas=2,
            package=str(self.package),package_manifest_sha256=sha(self.package/'package_manifest.json'),
            sources={n:sha(Path(pilot.__file__).with_name(n)) for n in pilot.SOURCES})
        write(self.out/'identity.json',identity)
        plan = dict(identity_sha256=sha(self.out/'identity.json'),entries=entries,assignments=[['token0'],['token1']])
        write(self.out/'plan.json',plan)
        write(self.out/'label_request.json',dict(protocol='table_top64_labels_v1',plan_sha256=sha(self.out/'plan.json'),
            metrics=['pdm_score',*base.METRICS],scenes=[dict(token=r['token'],candidates=[[c['local'],c['vocab']]
                for c in self.sample['candidates']]) for r in self.records]))
        return plan

    def infer(self,row):
        n = len(row['question']['criteria'])
        return dict(probabilities=[1/n]*n,seconds=.1)

    def workers(self,plan):
        for i,r in enumerate(self.records):
            w = self.out/'shards'/f'{i:02d}'
            write(w/'environment.json',dict(gpu='fixture',python='fixture'))
            pilot.infer_tasks(self.out,plan,[r],w,self.infer)

    def test_input_allowlist_and_nonfinite_head_rejection(self):
        allowed = set(base.INPUT_KEYS)|set(data.HEAD_KEYS)
        class Guard:
            def __getitem__(_,key):
                if key not in allowed: raise AssertionError('Outcome or decoder state read: '+key)
                return self.arrays[key]
        sample = data.inputs(Guard(),self.vocab)
        self.assertEqual(len(sample['candidates']),64)
        self.assertNotEqual(sample['candidates'][0]['local'],sample['candidates'][0]['vocab'])
        self.arrays['head_imi'][0] = np.nan
        with self.assertRaisesRegex(ValueError,'finite'): data.inputs(self.arrays,self.vocab)

    def test_plan_all_scenes_and_frozen_cohort(self):
        fixture = fixtures.RealPilotTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        package = fixture.package; index = read(package/'bundle/examples_index.json')
        for token in index:
            p = package/'bundle/examples'/(token+'.npz')
            with np.load(p,allow_pickle=False) as f: arrays = {k:f[k].copy() for k in f.files}
            arrays.update({k:self.arrays[k] for k in data.HEAD_KEYS})
            np.savez_compressed(p,**arrays); index[token]['sha256'] = sha(p)
        write(package/'bundle/examples_index.json',index)
        bm = read(package/'bundle/bundle_manifest.json'); bm['index_sha256'] = sha(package/'bundle/examples_index.json')
        write(package/'bundle/bundle_manifest.json',bm)
        m = read(package/'package_manifest.json')
        for name in m['files']:
            p = package/name; m['files'][name] = dict(size=p.stat().st_size,sha256=sha(p))
        m['total_bytes'] = sum(e['size'] for e in m['files'].values()); write(package/'package_manifest.json',m)
        out = fixture.root/'table'
        first = pilot.plan(fixture.root,package,out,2,'all')
        self.assertEqual(first['requests'],72); self.assertEqual(first['scenes'],4)
        self.assertEqual(pilot.plan(fixture.root,package,out,2,'all'),first)
        with self.assertRaisesRegex(RuntimeError,'Frozen'):
            pilot.plan(fixture.root,package,out,1,'all')

    def test_pool_nesting_alias_alignment_and_image_ablation(self):
        aliases = data.aliases('t',self.sample)
        for k in data.KS:
            a,order,ha = data.payload('t',self.sample,k,'table','canonical')
            b,order2,hb = data.payload('t',self.sample,k,'table_camera','canonical',self.photo)
            self.assertEqual(a['state'],b['state']); self.assertEqual(a['question'],b['question'])
            self.assertNotIn('images',a); self.assertEqual(len(b['images']),1)
            self.assertEqual(order,order2); self.assertNotEqual(ha,hb)
            self.assertEqual([r[0] for r in a['state']['candidates']['rows']],
                [aliases[self.sample['candidates'][i]['local']] for i in order])
            for row in a['state']['candidates']['rows']: self.assertEqual(len(row),len(data.COLUMNS))
            for kind in data.ORDERS:
                self.assertEqual(sorted(data.order('t',k,kind)),list(range(k)))
            self.assertNotEqual(order,data.order('t',k,'reversed'))
            self.assertNotEqual(order,data.order('t',k,'permuted'))

    def test_nominal_geometry_units_and_no_camera_access_in_table_payload(self):
        c = self.sample['candidates'][0]
        values = dict(zip(data.COLUMNS[1:],c['values']))
        self.assertAlmostEqual(values['length_m'],8,places=3)
        self.assertAlmostEqual(values['max_speed_mps'],2,places=3)
        self.assertAlmostEqual(values['max_abs_acceleration_mps2'],0,places=3)
        self.assertEqual(values['max_abs_curvature_per_m'],0)
        data.payload('t',self.sample,64,'table','canonical',photo=object())
        with self.assertRaisesRegex(ValueError,'photograph'):
            data.payload('t',self.sample,64,'table_camera','canonical')

    def test_partial_resume_and_full_cohort_gate_before_labels(self):
        plan = self.fixture(); w = self.out/'shards/00'
        status = pilot.infer_tasks(self.out,plan,self.records[:1],w,self.infer,max_new=2)
        self.assertEqual(status['status'],'partial'); self.assertFalse((w/'commit.json').exists())
        with patch.object(pilot,'label_lookup',side_effect=AssertionError('Premature label lookup')) as lookup:
            with self.assertRaises(FileNotFoundError): pilot.report(self.out)
            lookup.assert_not_called()
        status = pilot.infer_tasks(self.out,plan,self.records[:1],w,self.infer)
        self.assertEqual(status['new_requests'],16)
        self.workers(plan)
        result = pilot.report(self.out)
        self.assertEqual(result['status'],'awaiting_top64_labels')
        self.assertEqual(result['requests'],36)
        self.assertIn('table/mean3/k16',result['methods'])
        self.assertNotIn('table/mean3/k64',result['methods'])
        self.assertTrue((self.out/'decision_commit.json').is_file())

    def test_probability_alignment_and_full_external_raw_metrics(self):
        plan = self.fixture()
        desired = self.sample['candidates'][7]['local']
        for i,r in enumerate(self.records):
            w = self.out/'shards'/f'{i:02d}'; write(w/'environment.json',dict(gpu='fixture'))
            def choose(row,token=r['token']):
                alias = data.aliases(token,self.sample)[desired]
                names = list(row['question']['criteria']); p = [0.]*len(names); p[names.index(alias)] = 1.
                return dict(probabilities=p)
            pilot.infer_tasks(self.out,plan,[r],w,choose)
        _,decisions,_,_ = pilot.freeze_decisions(self.out)
        self.assertTrue(all(d['methods']['table/mean3']==desired for d in decisions))
        rows = []
        for r in self.records:
            for i,c in enumerate(self.sample['candidates']):
                raw = [1.]*9; raw[0]=.8 if i<16 else .123
                rows.append(dict(token=r['token'],candidate_local=c['local'],candidate_vocab=c['vocab'],raw=raw))
        path = self.root/'labels.json'
        write(path,dict(status='complete',request_digest=base.digest(read(self.out/'label_request.json')),metrics=['pdm_score',*base.METRICS],rows=rows))
        result = pilot.report(self.out,path)
        self.assertEqual(result['status'],'complete'); self.assertIn('table/mean3/k64',result['methods'])
        # Measured score deliberately differs from any formula reconstructed from all-one components.
        labels = pilot.label_lookup(self.package,read(self.package/'package_manifest.json'),decisions,rows)
        self.assertEqual(labels['token0'][self.sample['candidates'][20]['local']][0],.123)
        rows[0]['raw'][0]=.7
        with self.assertRaisesRegex(RuntimeError,'parity'):
            pilot.label_lookup(self.package,read(self.package/'package_manifest.json'),decisions,rows)

    def test_technical_failure_is_not_silently_retried(self):
        plan = self.fixture(); calls = []
        def fail(row): calls.append(row); raise RuntimeError('fixture failure')
        with self.assertRaisesRegex(RuntimeError,'fixture failure'):
            pilot.infer_tasks(self.out,plan,self.records[:1],self.out/'shards/00',fail)
        with self.assertRaisesRegex(RuntimeError,'Recorded technical failure'):
            pilot.infer_tasks(self.out,plan,self.records[:1],self.out/'shards/00',fail)
        self.assertEqual(len(calls),1)

    def test_changed_and_extra_results_block_evaluation(self):
        plan = self.fixture(); self.workers(plan)
        path = next((self.out/'shards/00/results').glob('*.json'))
        path.write_text(path.read_text()+' ')
        with patch.object(pilot,'label_lookup',side_effect=AssertionError('Premature')) as lookup:
            with self.assertRaisesRegex(RuntimeError,'changed'): pilot.report(self.out)
            lookup.assert_not_called()

    def test_inventory_checks_camera_windows_and_path_safety(self):
        cams = {k:dict(data_path=k+'.jpg') for k in ('cam_l0','cam_f0','cam_r0','cam_l1','cam_l2','cam_r1','cam_r2','cam_b0')}
        frames = [dict(cams=cams) for _ in range(4)]
        existing = {v['data_path'] for v in cams.values()}
        self.assertEqual(inventory.coverage(frames,3,existing)[2:],(True,True))
        existing.remove('cam_b0.jpg')
        self.assertEqual(inventory.coverage(frames,3,existing)[2:],(True,False))
        cams['cam_f0']['data_path']='../escape.jpg'
        with self.assertRaisesRegex(ValueError,'Unsafe'): inventory.coverage(frames,3,existing)

    def test_remote_scoring_rejects_reordered_incomplete_and_misaligned_pools(self):
        request = dict(protocol='table_top64_labels_v1',metrics=['pdm_score',*base.METRICS],
            scenes=[dict(token='t',candidates=[[c['local'],c['vocab']] for c in self.sample['candidates']])])
        with patch.object(scorer.tables,'load_inputs',return_value=self.sample):
            self.assertEqual(len(scorer.frozen_groups(request,None,None,None)['t']),64)
            rows = request['scenes'][0]['candidates']
            for subset in (rows[:16],list(reversed(rows))):
                with self.assertRaisesRegex(RuntimeError,'frozen'):
                    scorer.frozen_groups(dict(request,scenes=[dict(token='t',candidates=subset)]),None,None,None)
            with self.assertRaisesRegex(ValueError,'duplicate'):
                scorer.frozen_groups(dict(request,scenes=request['scenes']*2),None,None,None)

    def test_651_scene_label_upload_fits_the_remote_bound(self):
        request = dict(protocol='table_top64_labels_v1',metrics=['pdm_score',*base.METRICS],
            scenes=[dict(token=f'{i:016x}',candidates=[[j,8191-j] for j in range(64)]) for i in range(651)])
        upload = self.root/'upload'; write(upload/'audit_request.json',request)
        tools = tuple(dict.fromkeys(('table_score.py','table_data.py',*label_bridge.bridge.TOOLS)))
        packet = label_bridge.bridge.input_archive(upload,tools)
        self.assertLess(len(packet),4*1024**2)

    def test_plan_wait_is_bounded_and_does_not_infer_or_read_outcomes(self):
        path = self.root/'not_ready.json'
        with self.assertRaisesRegex(RuntimeError,'timed out'): label_bridge.wait_for(path,0)
        path.write_text('{}'); label_bridge.wait_for(path,0)


class TableSlurmTests(unittest.TestCase):
    def setUp(self):
        self.f = slurm_fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.script = Path(__file__).with_name('run_table.sh').resolve()
        (self.f.envroot/'setup_node.txt').write_text('cn29\n')
        for k in ('DREX_TABLE_NAME','DREX_TABLE_COHORT','DREX_TABLE_TIME'):
            self.f.env.pop(k,None)
        self.f.env['DREX_TEST_PREFIX'] = self.f.posix(self.f.root/'batch')
        self.f.stub('sbatch','n=0; [[ ! -f "$DREX_TEST_PREFIX.count" ]] || n=$(cat "$DREX_TEST_PREFIX.count")\n'
            'n=$((n+1)); echo "$n" > "$DREX_TEST_PREFIX.count"\n'
            'printf "%s\\n" "$@" > "$DREX_TEST_PREFIX-$n.args"\n'
            'cat > "$DREX_TEST_PREFIX-$n.sh"\necho "$((800+n))"\n')

    def test_all_replica_counts_and_dependency_chain(self):
        for count in ('1','2','4','8'):
            result = self.f.run_stage('submit',count)
            self.assertEqual(result.returncode,0,result.stderr)
        args = (self.f.root/'batch-11.args').read_text().splitlines()
        self.assertIn('--array=0-7%8',args); self.assertIn('--gres=gpu:1',args)
        self.assertIn('--mem=96G',args)
        merge = (self.f.root/'batch-12.args').read_text().splitlines()
        self.assertIn('--dependency=afterok:811',merge)
        self.assertIn('--kill-on-invalid-dep=yes',merge); self.assertIn('--gres=none',merge)
        self.assertNotEqual(self.f.run_stage('submit','3').returncode,0)

    def test_pilot_does_not_schedule_premature_label_evaluation(self):
        result = self.f.run_stage('submit-pilot','4')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('table_topk_all_gpu4_v1',result.stdout)
        args = (self.f.root/'batch-2.args').read_text().splitlines()
        self.assertIn('--time=00:45:00',args)
        self.assertIn('--dependency=afterok:801',args)
        self.assertFalse((self.f.root/'batch-3.args').exists())

    def test_inventory_is_cpu_only_and_does_not_need_login_venv(self):
        result = self.f.run_stage('submit-inventory','8')
        self.assertEqual(result.returncode,0,result.stderr)
        args = (self.f.root/'batch-1.args').read_text().splitlines()
        self.assertIn('--gres=none',args); self.assertIn('--time=02:00:00',args)
        self.assertFalse((self.f.envroot/'venv').exists())
        self.assertFalse((self.f.root/'batch-2.args').exists())

    def test_run_names_cannot_replace_old_experiments(self):
        self.f.env['DREX_TABLE_NAME']='overlay_v2_top4_gpu4_v1'
        self.assertNotEqual(self.f.run_stage('submit','4').returncode,0)

    def test_night_requires_checks_and_complete_pilot_before_full_array(self):
        result = self.f.run_stage('submit-night','8')
        self.assertEqual(result.returncode,0,result.stderr)
        stages = [(self.f.root/f'batch-{i}.args').read_text().splitlines() for i in range(1,6)]
        self.assertIn('--gres=none',stages[0])
        self.assertIn('--dependency=afterok:801',stages[1])
        self.assertIn('--dependency=afterok:802',stages[2])
        self.assertIn('--time=00:45:00',stages[2])
        self.assertIn('--dependency=afterok:803',stages[3])
        self.assertIn('--time=08:00:00',stages[3]); self.assertIn('--array=0-7%8',stages[3])
        self.assertIn('--dependency=afterok:804',stages[4])
        self.assertIn('--gres=none',stages[4])


if __name__=='__main__': unittest.main()
