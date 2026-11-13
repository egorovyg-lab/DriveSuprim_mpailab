"""CPU contract/integration tests, with fake inference and Slurm commands."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import replacement as pilot
import replacement_data as data
import table_data as original
import test_table_factorial as fixtures
import test_table_group as shell_tests
from runtime import read, write
from transfer import sha


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.FactorialTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.root = self.f.root; self.source = self.f.source; self.out = self.root/'replacement'
        self.sample = self.f.sample; self.prepared = data.views(self.f.photo)

    def source_fixture(self):
        self.f.source_fixture()
        pilot.prepare(self.root,self.source,self.out,2)
        return pilot.context(self.out)

    @staticmethod
    def infer(row):
        chosen = row['state']['challenger_id']; ids = list(row['question']['criteria'])
        p = [.2,.2]; p[ids.index(chosen)] = .8
        return dict(probabilities=p,seconds=.1,input_tokens=300,has_image_pixels='images' in row,cold_call=False)

    def workers(self,ctx,max_new=None,infer=None):
        identity,plan,source,source_plan = ctx
        for i,tokens in enumerate(plan['assignments']):
            w = self.out/'shards'/f'{i:02d}'; write(w/'environment.json',dict(gpu='fake'))
            records = [r for r in identity['records'] if r['token'] in tokens]
            pilot.infer_tasks(self.out,source,source_plan,records,w,infer or self.infer,max_new=max_new)

    def test_all_63_challengers_include_tail_without_any_outcome_access(self):
        before = copy.deepcopy(self.sample)
        cs = data.challengers(self.sample)
        self.assertEqual(len(cs),63); self.assertNotIn(self.sample['baseline_local'],[c['local'] for c in cs])
        self.assertEqual([self.sample['candidates'].index(c)+1 for c in cs[:3]],[2,64,32])
        self.assertEqual(self.sample,before)
        tasks = data.tasks(self.f.f.records,{r['token']:self.sample for r in self.f.f.records})
        self.assertEqual(len(tasks),756); self.assertEqual(len({data.name(*t) for t in tasks}),756)
        self.assertEqual([t[1] for t in tasks[:18:6]],[c['local'] for c in cs[:3]])

    def test_swap_keeps_delta_and_physical_aliases_fixed(self):
        local = data.challengers(self.sample)[0]['local']
        a,ids,ha = data.payload('t',self.sample,local,'none',data.ORDERS[0],self.prepared)
        b,rev,hb = data.payload('t',self.sample,local,'none',data.ORDERS[1],self.prepared)
        self.assertEqual(ids,rev[::-1]); self.assertNotEqual(ha,hb)
        self.assertEqual(a['state']['change_if_replaced'],b['state']['change_if_replaced'])
        self.assertEqual(a['state']['reference_id'],b['state']['reference_id'])
        self.assertEqual(list(a['question']['criteria']),list(b['question']['criteria'])[::-1])
        self.assertEqual(a['state']['candidates']['rows'],b['state']['candidates']['rows'][::-1])
        self.assertNotIn('measured',a['state']['change_if_replaced']['columns'])
        self.assertEqual(data.preference([.2,.8],ids,local),data.preference([.8,.2],rev,local))

    def test_delta_sign_rounding_and_wrapped_heading(self):
        sample = copy.deepcopy(self.sample); baseline,alt = sample['candidates'][:2]
        heading = original.COLUMNS[1:].index('heading_4s_deg')
        baseline['values'][heading] = 179.; alt['values'][heading] = -179.
        baseline['values'][0] = 0.; alt['values'][0] = -.1
        row,_,_ = data.payload('t',sample,alt['local'],'none',data.ORDERS[0],self.prepared)
        delta = row['state']['change_if_replaced']['values']
        self.assertEqual(delta[heading],2.); self.assertEqual(delta[0],-.1)
        np.testing.assert_allclose(delta[1:heading],np.array(alt['values'][1:heading])-baseline['values'][1:heading],atol=1e-4)

    def test_image_controls_have_identical_text_and_distinct_digests(self):
        j = data.challengers(self.sample)[0]['local']
        values = [data.payload('t',self.sample,j,m,data.ORDERS[0],self.prepared) for m in data.MODES]
        self.assertEqual(len({v[2] for v in values}),3)
        self.assertNotIn('images',values[0][0]); self.assertIn('images',values[1][0])
        for row,_,_ in values[1:]: self.assertEqual(row['state'],values[0][0]['state'])
        columns = values[0][0]['state']['candidates']['columns']
        self.assertIn('ds_score_gap',columns); self.assertIn('imitation_log_probability_256',columns)

    def test_policy_fallback_ties_and_order_bias_are_input_only(self):
        a = dict(candidate_local=7,teacher_rank=2,mean_preference=.6,minimum_preference=.3)
        b = dict(candidate_local=8,teacher_rank=20,mean_preference=.7,minimum_preference=.65)
        self.assertEqual(data.select([a],1,'both050'),1)
        self.assertEqual(data.select([a,b],1,'both050'),8)
        self.assertEqual(data.select([a,b],1,'both075'),1)
        a.update(mean_preference=.7,minimum_preference=.65)
        self.assertEqual(data.select([b,a],1,'both050'),7)
        self.assertEqual(data.select([],1,'mean050'),1)
        self.assertEqual(data.select([dict(a,minimum_preference=.5)],1,'both050'),1)

    def test_bad_pair_alignment_and_probability_values_rejected(self):
        for p,ids in (([float('nan'),.2],[1,2]),([.2,.2],[1,2]),([.5,.5],[1,1]),([.5,.5],[3,4])):
            with self.assertRaises(ValueError): data.preference(p,ids,2)

    def test_plan_is_reproducible_read_only_and_replica_count_is_frozen(self):
        self.f.source_fixture()
        before = {p:p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        first = pilot.prepare(self.root,self.source,self.out,2)
        self.assertEqual(first['requests'],756); self.assertEqual(first['pairs'],126)
        self.assertEqual(pilot.prepare(self.root,self.source,self.out,2),first)
        for path,value in before.items(): self.assertEqual(path.read_bytes(),value)
        with self.assertRaises(RuntimeError): pilot.prepare(self.root,self.source,self.out,1)
        (self.source/'report.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'Original source changed'): pilot.context(self.out)

    def test_partial_resume_full_pair_commit_then_report(self):
        ctx = self.source_fixture(); self.workers(ctx,max_new=18)
        with patch('audit_table_cases.validate_source',side_effect=AssertionError('Early outcomes')) as lookup:
            with self.assertRaises(FileNotFoundError): pilot.report(self.out)
            lookup.assert_not_called()
        self.workers(ctx)
        result = pilot.report(self.out)
        self.assertEqual(result['status'],'complete'); self.assertEqual(result['requests'],756)
        self.assertEqual(result['pairs'],126); self.assertEqual(result['pair_quality']['none']['pairs'],126)
        self.assertEqual(result['methods']['none/both050']['intervention'],2)
        for r in read(self.out/'decisions.json'):
            self.assertEqual(set(r['methods'].values()),{self.sample['candidates'][1]['local']})
        self.assertEqual(pilot.report(self.out),result)
        def fail(_): raise AssertionError('Unexpected repeated model inference')
        self.workers(ctx,max_new=0,infer=fail)

    def test_tampering_blocks_labels_and_corrupt_failures_are_not_retried(self):
        ctx = self.source_fixture(); self.workers(ctx)
        path = next((self.out/'shards/00/results').glob('*.json')); path.write_text(path.read_text()+' ')
        with patch('audit_table_cases.validate_source',side_effect=AssertionError('Early outcomes')) as lookup:
            with self.assertRaisesRegex(RuntimeError,'Committed result changed'): pilot.report(self.out)
            lookup.assert_not_called()

    def test_live_progress_written_before_final_commit(self):
        identity,plan,source,source_plan = self.source_fixture()
        w = self.out/'shards/00'; write(w/'environment.json',{})
        records = [r for r in identity['records'] if r['token'] in plan['assignments'][0]]
        seen = []
        def infer(row):
            current = read(w/'inference_status.json'); seen.append(current['completed_requests'])
            return self.infer(row)
        pilot.infer_tasks(self.out,source,source_plan,records,w,infer,max_new=26)
        self.assertEqual(seen[-1],25)
        self.assertFalse((w/'commit.json').exists())
        self.assertEqual(read(w/'inference_status.json')['completed_requests'],26)

    def test_failure_is_saved_and_never_repeated_until_success(self):
        identity,plan,source,source_plan = self.source_fixture()
        w = self.out/'shards/00'; write(w/'environment.json',{})
        records = [r for r in identity['records'] if r['token'] in plan['assignments'][0]]
        def fail(_): raise RuntimeError('fake GPU failure')
        with self.assertRaisesRegex(RuntimeError,'fake GPU failure'): pilot.infer_tasks(self.out,source,source_plan,records,w,fail)
        self.assertEqual(read(w/'inference_status.json')['status'],'failed')
        with self.assertRaisesRegex(RuntimeError,'Recorded technical failure'): pilot.infer_tasks(self.out,source,source_plan,records,w,self.infer)


class ReplacementSlurmTests(shell_tests.GroupSlurmTests):
    def setUp(self):
        super().setUp()
        self.f.script = Path(__file__).with_name('run_replacement.sh').resolve()
        for key in ('DREX_REPLACEMENT_NAME','DREX_REPLACEMENT_SOURCE','DREX_REPLACEMENT_TIME'):
            self.f.env.pop(key,None)

    def test_night_has_one_gpu_job_for_both_pilot_and_full(self):
        r = self.f.run_stage('submit-night','8'); self.assertEqual(r.returncode,0,r.stderr)
        for i in (1,2,4): self.assertIn('--gres=none',(self.f.root/f'batch-{i}.args').read_text())
        args = (self.f.root/'batch-3.args').read_text()
        self.assertIn('--gres=gpu:8',args); self.assertIn('--ntasks=8',args)
        self.assertIn('--mem=768G',args); self.assertNotIn('--array',args)
        self.assertIn('--dependency=afterok:902',args)
        self.assertIn('--inside gpu-night 8',(self.f.root/'batch-3.sh').read_text())
        self.assertIn('--dependency=afterok:903',(self.f.root/'batch-4.args').read_text())

    def test_old_scientific_outputs_cannot_be_used_as_group_outputs(self):
        self.f.env['DREX_REPLACEMENT_NAME'] = 'table_factorial_gpu4_v1'
        self.assertNotEqual(self.f.run_stage('submit-night','8').returncode,0)

    def test_failed_pilot_blocks_full_step(self):
        self.f.fake_setup(); self.f.env['SLURM_STEP_ID'] = 'batch'
        self.f.stub('srun','printf "%s\\n" "$@" >> "$DREX_TEST_LOG"\nexit 1\n')
        self.assertNotEqual(self.f.run_stage('--inside','gpu-night','4').returncode,0)
        self.assertEqual(self.f.log.read_text().count('--gpus-per-task=1'),1)


if __name__=='__main__': unittest.main()
