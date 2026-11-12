"""CPU mini integration tests: no weights, network, real dataset or CUDA."""
import ast
from contextlib import redirect_stdout
import copy
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import real_data as base
import table_data as original
import table_pilot as previous
import table_factorial_data as data
import table_factorial as pilot
import table_pair_bridge as bridge
import table_pair_remote as remote
import test_table as fixtures
import test_table_group as group_tests
from runtime import read, write
from transfer import sha


class FactorialTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.TableTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.photo = Image.new('RGB', (1536,640), 'red')
        self.f.photo = self.photo
        self.sample = self.f.sample
        self.root = self.f.root; self.source = self.f.out; self.out = self.root/'factorial'
        self.quiet = redirect_stdout(io.StringIO()); self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def source_fixture(self):
        plan = self.f.fixture()
        write(self.root/'model_plan.json', dict(files=[],revision='fixture'))
        (self.root/'requirements.lock.txt').write_text('fixture-lock')
        identity = read(self.source/'identity.json')
        identity.update(model_plan=read(self.root/'model_plan.json'),
            environment_lock_sha256=sha(self.root/'requirements.lock.txt'))
        write(self.source/'identity.json', identity)
        plan['identity_sha256'] = sha(self.source/'identity.json'); write(self.source/'plan.json', plan)
        request = read(self.source/'label_request.json'); request['plan_sha256'] = sha(self.source/'plan.json')
        write(self.source/'label_request.json', request)
        self.f.workers(plan)
        rows = []
        for r in original.requested_candidates(request):
            rank = next(i for i,c in enumerate(self.sample['candidates']) if c['local'] == r['candidate_local'])
            rows.append(dict(r,raw=[.8 if rank < 16 else .9]+[1.]*8))
        path = self.root/'labels.json'
        write(path, dict(status='complete', metrics=['pdm_score',*base.METRICS],
            request_digest=base.digest(request), rows=rows))
        previous.report(self.source, path)
        write(self.source/'label_source.json', dict(labels=str(path), sha256=sha(path),
            request_sha256=sha(self.source/'label_request.json')))
        metadata = patch.object(base, 'package_metadata', return_value=(
            read(self.f.package/'package_manifest.json'), None, None, self.f.vocab))
        metadata.start(); self.addCleanup(metadata.stop)
        return plan

    def new_plan(self, count=2):
        self.source_fixture()
        pilot.prepare(self.root,self.source,self.out,count)
        return pilot.context(self.out)

    @staticmethod
    def infer(row):
        n = len(row['question']['criteria'])
        return dict(probabilities=[1/n]*n, seconds=.1, input_tokens=100,
            has_image_pixels='images' in row, cold_call=False)

    def workers(self, ctx, max_new=None):
        identity, plan, source, source_plan = ctx
        for i,tokens in enumerate(plan['assignments']):
            w = self.out/'shards'/f'{i:02d}'
            write(w/'environment.json', dict(gpu='fixture'))
            records = [r for r in identity['records'] if r['token'] in tokens]
            pilot.infer_tasks(self.out,plan,source,source_plan,records,w,self.infer,max_new=max_new)

    def test_full_payload_exactly_reproduces_old_input_and_photo(self):
        for order in data.ORDERS:
            a,p,ha = original.payload('token',self.sample,64,'table',order)
            b,q,hb = data.payload('token',self.sample,'none_full',order,self.photo)
            self.assertEqual((a,p,ha),(b,q,hb))
            a,p,ha = original.payload('token',self.sample,64,'table_camera',order,self.photo)
            b,q,hb = data.payload('token',self.sample,'road_full',order,self.photo)
            self.assertEqual((p,ha),(q,hb)); self.assertEqual(a['state'],b['state'])

    def test_hidden_removes_only_final_score_not_heads_imitation_or_aliases(self):
        before = copy.deepcopy(self.sample)
        full,p,_ = data.payload('token',self.sample,'none_full','canonical',self.photo)
        hidden,q,_ = data.payload('token',self.sample,'none_hidden','canonical',self.photo)
        a = full['state']['candidates']; b = hidden['state']['candidates']
        self.assertEqual(p,q); self.assertEqual(full['question'],hidden['question'])
        self.assertEqual(b['columns'],[c for c in a['columns'] if c != 'ds_score_gap'])
        for x,y in zip(a['rows'],b['rows']): self.assertEqual(y,[x[0],*x[2:]])
        self.assertNotIn('ds_score_gap',str(hidden)); self.assertEqual(before,self.sample)

    def test_blank_masks_only_photos_preserving_exact_headers_and_shape(self):
        blank = data.blank_board(self.photo)
        a,b = np.array(self.photo),np.array(blank)
        self.assertEqual(blank.size,self.photo.size)
        for y in (0,320):
            np.testing.assert_array_equal(a[y:y+28],b[y:y+28])
            self.assertTrue(np.all(b[y+28:y+320] == 255))
        rows = [data.payload('t',self.sample,mode,'canonical',self.photo) for mode in ('none_full','blank_full','road_full')]
        self.assertEqual(len({r[2] for r in rows}),3)
        self.assertEqual(rows[0][0]['state'],rows[1][0]['state'])
        with self.assertRaises(ValueError): data.blank_board(Image.new('RGB',(16,16)))

    def test_all_conditions_and_orders_cover_exactly_same_candidates(self):
        self.assertEqual(len(data.tasks(self.f.records)),36)
        names = []
        for token,mode,order in data.tasks(self.f.records):
            row,p,_ = data.payload(token,self.sample,mode,order,self.photo)
            self.assertEqual(sorted(p),list(range(64)))
            self.assertEqual(len(row['question']['criteria']),64)
            names.append(data.task_name(token,mode,order))
        self.assertEqual(len(names),len(set(names)))

    def test_plan_preserves_original_and_locks_source_and_replica_count(self):
        self.source_fixture()
        before = {p:p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        value = pilot.prepare(self.root,self.source,self.out,2)
        self.assertEqual(value['requests'],36)
        self.assertEqual(value,pilot.prepare(self.root,self.source,self.out,2))
        for path,content in before.items(): self.assertEqual(path.read_bytes(),content)
        with self.assertRaises(RuntimeError): pilot.prepare(self.root,self.source,self.out,1)
        (self.source/'report.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'Original experiment changed'): pilot.context(self.out)

    def test_partial_results_block_all_outcome_lookup_and_resume(self):
        ctx = self.new_plan(); self.workers(ctx,max_new=1)
        with patch('audit_table_cases.validate_source',side_effect=AssertionError('Early labels')) as lookup:
            with self.assertRaises(FileNotFoundError): pilot.report(self.out)
            lookup.assert_not_called()
        self.workers(ctx)
        result = pilot.report(self.out)
        self.assertEqual(result['status'],'complete'); self.assertEqual(result['requests'],36)
        self.assertEqual(len(result['paired_contrasts']),9)
        self.assertEqual(result['image_score_interaction']['gain_micro_pp'],0.)
        self.assertEqual(pilot.report(self.out),result)
        with patch.object(self,'infer',side_effect=AssertionError('Repeated inference')):
            self.workers(ctx,max_new=0)

    def test_tampered_committed_result_blocks_labels(self):
        ctx = self.new_plan(); self.workers(ctx)
        path = next((self.out/'shards/00/results').glob('*.json')); path.write_text(path.read_text()+' ')
        with patch('audit_table_cases.validate_source',side_effect=AssertionError('Early labels')) as lookup:
            with self.assertRaisesRegex(RuntimeError,'Changed worker result'): pilot.report(self.out)
            lookup.assert_not_called()

    def test_saved_technical_failure_is_not_silently_retried(self):
        identity,plan,source,source_plan = self.new_plan()
        w = self.out/'shards/00'; write(w/'environment.json',{})
        records = identity['records'][:1]
        def fail(_): raise RuntimeError('fixture OOM')
        with self.assertRaisesRegex(RuntimeError,'fixture OOM'):
            pilot.infer_tasks(self.out,plan,source,source_plan,records,w,fail)
        with self.assertRaisesRegex(RuntimeError,'Recorded technical failure'):
            pilot.infer_tasks(self.out,plan,source,source_plan,records,w,self.infer)

    def test_probabilities_align_to_physical_candidates_before_averaging(self):
        identity,plan,source,source_plan = self.new_plan()
        for i,tokens in enumerate(plan['assignments']):
            w = self.out/'shards'/f'{i:02d}'; write(w/'environment.json',{})
            token = tokens[0]; chosen = self.sample['candidates'][12]['local']
            alias = original.aliases(token,self.sample)[chosen]
            def infer(row):
                ids = list(row['question']['criteria']); p = [0.]*64; p[ids.index(alias)] = 1.
                return dict(probabilities=p,seconds=.1,input_tokens=100,cold_call=False)
            pilot.infer_tasks(self.out,plan,source,source_plan,[identity['records'][i]],w,infer)
        _,decisions,_ = pilot.collect(self.out)
        for d in decisions:
            self.assertEqual(set(d['methods'].values()),{chosen})

    def test_session_macro_and_paired_contrast_use_session_clusters(self):
        rows = [dict(session_id='a',gain_pp=10.)]*3+[dict(session_id='b',gain_pp=-2.)]
        self.assertEqual(pilot.summarize(rows)['gain_macro_pp'],4.)
        self.assertEqual(pilot.summarize(rows)['gain_micro_pp'],7.)
        self.assertEqual(pilot.summarize(rows),pilot.summarize(rows))


class PairRequestTests(unittest.TestCase):
    def test_four_candidates_bound_to_actual_completed_choices(self):
        cases = bridge.CASES; decisions = []; labels = {}; samples = {}; files = {}
        for _,token,_ in cases:
            decisions.append(dict(metadata=dict(token=token),k=64,baseline_local=7,
                methods={'table/mean3':3,'table_camera/mean3':7}))
            labels[token] = {7:np.array([0.]+[1.]*8),3:np.ones(9)}
            samples[token] = dict(candidates=[dict(local=7,vocab=100),dict(local=3,vocab=400)])
            files['bundle/examples/'+token+'.npz'] = dict(sha256='a'*64)
        rows = bridge.candidates_from_decisions(decisions,labels,samples,dict(files=files))
        self.assertEqual(len(rows),4)
        self.assertEqual([c['candidate_vocab'] for c in rows],[100,400,100,400])
        request = dict(protocol='table_pair_rollout_v1',candidates=rows)
        remote.validate_request(request)
        bad = copy.deepcopy(request); bad['candidates'][1]['member'] = 0
        with self.assertRaises(ValueError): remote.validate_request(bad)
        decisions[0]['methods']['table_camera/mean3'] = 3
        with self.assertRaisesRegex(RuntimeError,'DS-retention'): bridge.candidates_from_decisions(decisions,labels,samples,dict(files=files))

    def test_staged_remote_tools_parse_under_astra_python39(self):
        for name in bridge.TOOLS:
            ast.parse(Path(bridge.__file__).with_name(name).read_text(encoding='utf-8'),feature_version=(3,9))

    def test_remote_adapter_requires_metric_parity_and_resumes_committed_artifacts(self):
        # Exercise orchestration without installing nuPlan or substituting model outcomes.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); package = root/'package'; repo = root/'repo'; out = root/'audit'
            files = {}; candidates = []; scenes = {}; caches = {}
            for pair,token,focus in bridge.CASES:
                rel = 'bundle/examples/'+token+'.npz'; path = package/rel; path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,candidate_ids=np.array([100,200]),label_indices=np.array([0,1]),label_raw=np.ones((2,9)))
                files[rel] = dict(sha256=sha(path)); scenes[token] = {}; caches[token] = {}
                for member in (0,1):
                    candidates.append(dict(pair_id=pair,token=token,member=member,candidate_local=member,
                        candidate_vocab=(100,200)[member],focus_metric=focus,better_member=1,
                        metric_keys=['pdm_score',*base.METRICS],expected_raw=[1.]*9,example_sha256=sha(path)))
            photo = package/'photo.png'; Image.new('RGB',(8,8)).save(photo); files['photo.png'] = dict(sha256=sha(photo))
            manifest = dict(files=files); write(package/'package_manifest.json',manifest)
            profile = read(Path(bridge.__file__).with_name('vehicle_profile_astra.json'))
            rp = root/'request.json'
            write(rp,dict(protocol='table_pair_rollout_v1',candidates=candidates,remote_package=str(package),
                package_manifest_sha256=sha(package/'package_manifest.json'),vehicle_profile=profile))
            for name in remote.replay.REPLAY_SOURCES:
                p = repo/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('fixture')
            def fake_replay(package,directory,c,*_):
                # One mismatching result must prevent geometry acceptance even if the loop completes.
                summary = dict(pair_id=c['pair_id'],member=c['member'],parity_passed=c['member'] == 1,
                    measured_raw={},nominal_vs_simulated={},simulated_dac={},simulated_lane_keeping={},
                    first_at_fault_collision_time_s=None,overlaps_at_recorded_at_fault_time=[])
                write(directory/'summary.json',summary)
                write(directory/'commit.json',dict(files={'summary.json':sha(directory/'summary.json')}))
                return summary
            with patch.object(remote.replay,'bound_context',return_value=(manifest,scenes,None,{},caches)), \
                 patch.object(base,'camera_paths',return_value=[photo]*6), \
                 patch.object(remote.replay,'pair_summaries',return_value=[]), \
                 patch.object(remote.replay,'replay_candidate',side_effect=fake_replay) as scorer, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(remote.run(rp,out,repo),2)
                self.assertFalse(read(out/'report.json')['geometry_interpretation_ready'])
                self.assertEqual(scorer.call_count,4)
                self.assertEqual(remote.run(rp,out,repo),2)
                self.assertEqual(scorer.call_count,4)
                path = out/'case01-member0/summary.json'; path.write_text('{}')
                with self.assertRaisesRegex(RuntimeError,'artifact changed'): remote.run(rp,out,repo)


class FactorialSlurmTests(group_tests.GroupSlurmTests):
    # Reuse the exercised shell stubs, but test the new launcher independently.
    def setUp(self):
        super().setUp()
        self.f.script = Path(__file__).with_name('run_table_factorial.sh').resolve()
        for key in ('DREX_FACTORIAL_NAME','DREX_FACTORIAL_SOURCE','DREX_FACTORIAL_TIME'):
            self.f.env.pop(key,None)

    def test_night_has_one_gpu_job_for_both_pilot_and_full(self):
        r = self.f.run_stage('submit-night','8')
        self.assertEqual(r.returncode,0,r.stderr)
        for i in (1,2,4): self.assertIn('--gres=none',(self.f.root/f'batch-{i}.args').read_text().splitlines())
        args = (self.f.root/'batch-3.args').read_text().splitlines()
        self.assertIn('--gres=gpu:8',args); self.assertIn('--ntasks=8',args)
        self.assertIn('--dependency=afterok:902',args)
        self.assertIn('--inside gpu-night 8',(self.f.root/'batch-3.sh').read_text())
        self.assertIn('--dependency=afterok:903',(self.f.root/'batch-4.args').read_text())

    def test_old_scientific_outputs_cannot_be_used_as_group_outputs(self):
        self.f.env['DREX_FACTORIAL_NAME'] = 'table_topk_night_gpu8_v1'
        self.assertNotEqual(self.f.run_stage('submit-night','8').returncode,0)
        self.assertFalse((self.f.root/'batch-1.args').exists())

    def test_failed_pilot_step_prevents_full_gpu_step(self):
        self.f.fake_setup(); self.f.env['SLURM_STEP_ID'] = 'batch'
        self.f.stub('srun','printf "%s\\n" "$@" >> "$DREX_TEST_LOG"\nexit 1\n')
        r = self.f.run_stage('--inside','gpu-night','2')
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(self.f.log.read_text().count('--gpus-per-task=1'),1)


class PairSlurmTests(unittest.TestCase):
    def setUp(self):
        import test_runtime as shell_fixtures
        self.f = shell_fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.script = Path(__file__).with_name('run_table_pair_audit.sh').resolve()
        (self.f.envroot/'setup_node.txt').write_text('cn29\n')

    def test_prepare_is_cpu_only_and_start_requires_a_prepared_request(self):
        result = self.f.run_stage('prepare')
        self.assertEqual(result.returncode,0,result.stderr)
        log = self.f.log.read_text()
        self.assertIn('--gres=none',log); self.assertIn('--cpus-per-task=2',log)
        self.assertIn('prepare',log); self.assertNotIn('--gres=gpu',log)
        result = self.f.run_stage('start')
        self.assertNotEqual(result.returncode,0); self.assertIn('Run prepare first',result.stderr)


if __name__ == '__main__': unittest.main()
