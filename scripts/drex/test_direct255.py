import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS,ModuleType
import unittest
from unittest.mock import patch,Mock

import numpy as np
import direct255 as d
from test_direct256 import fixture as original_fixture
from transfer import write,read,sha


def fixture():
    arrays,vocab=original_fixture()
    arrays['selection_scores']=np.linspace(1.,0.,256);arrays['selected_local']=np.array(0)
    return arrays,vocab


class PoolTests(unittest.TestCase):
    def test_excludes_actual_last_rank_not_last_local_or_best_ds(self):
        a,v=fixture();a['selection_scores'][17]=-1
        s=d.inputs(a,v);ids={c['local'] for c in s['candidates']}
        self.assertEqual(len(ids),255);self.assertNotIn(17,ids);self.assertIn(255,ids);self.assertIn(0,ids)
        a['selection_scores'][:]=1.;self.assertEqual(d.membership(a)[1],255)
        a['selected_local']=np.array(12)
        with self.assertRaisesRegex(ValueError,'baseline'):d.membership(a)
        a,v=fixture();a['selection_scores'][4]=np.nan
        with self.assertRaisesRegex(ValueError,'scores'):d.membership(a)

    def test_absolute_values_unchanged_and_input_only(self):
        a,v=fixture()
        allowed=set(a)
        class Inputs(dict):
            def __getitem__(self,key):
                if key not in allowed:raise AssertionError('Forbidden read: '+key)
                return super().__getitem__(key)
        import direct256
        self.assertEqual(d.inputs(Inputs(a),v)['candidates'],direct256.inputs(a,v)['candidates'][:255])

    def test_payload_has_255_options_and_no_ranking_score_or_baseline_leakage(self):
        a,v=fixture();sample=d.inputs(a,v)
        p,ids,h=d.payload('token',sample,'primary');q,reverse,_=d.payload('token',sample,'reversed')
        self.assertEqual(len(p['question']['criteria']),255);self.assertEqual(len(p['state']['candidate_table'].splitlines()),256)
        self.assertEqual(ids,reverse[::-1]);self.assertEqual(set(ids),set(range(255)))
        for forbidden in ('teacher_rank','ds_score','selection_scores','baseline','imitation','images','excluded'):
            self.assertNotIn(forbidden,json.dumps(p))
        b=copy.deepcopy(a);b['selection_scores'][1:255]+=0.001
        self.assertEqual(d.payload('token',d.inputs(b,v),'primary')[2],h)
        ast.parse(Path(d.__file__).read_text(),feature_version=(3,9))

    def test_result_receipt_and_request_binding_prevent_tampering(self):
        a,v=fixture();_,ids,digest=d.payload('token',d.inputs(a,v),'primary')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);write(out/'identity.json',dict(protocol='test'))
            file=out/'results/token-primary.json'
            r=dict(token='token',order='primary',request_sha256=digest,local_ids=ids,probabilities=[1/255]*255)
            write(file,r);write(file.with_suffix('.commit.json'),dict(identity_sha256=sha(out/'identity.json'),result_sha256=sha(file)))
            d.checked_result(out,file,digest,ids)
            r['probabilities']=[1.]+[0.]*254;write(file,r)
            with self.assertRaisesRegex(RuntimeError,'receipt'):d.checked_result(out,file,digest,ids)


class WorkerTests(unittest.TestCase):
    def test_pilot_full_and_resume_reuse_results_across_rank_assignments(self):
        import runtime
        import table_group
        a,v=fixture();sample=d.inputs(a,v)
        identity=dict(records=[dict(token='t0'),dict(token='t1')],pilot_tokens=['t1'],model_plan=dict(files=[]))
        cuda=NS(synchronize=lambda:None,reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:42)
        torch=ModuleType('torch');torch.cuda=cuda;torch.bfloat16='bf16'
        mode={'rank':0,'tokens':100,'fail':False};calls=[]
        class Model:
            def __init__(self,*args,**kwargs):self.readout=NS(weight=NS(dtype='bf16'),out_features=255)
            def prepare(self,rows,max_length):return NS(inputs={'input_ids':np.empty((1,mode['tokens']))})
            def predict(self,rows,**kwargs):
                calls.append(rows)
                if mode['fail']:raise RuntimeError('CUDA out of memory')
                return [[1.]+[0.]*254]
        sdk=ModuleType('drex_max.model');sdk.MAX_OPTIONS=255;sdk.DecisionModel=Model
        env=dict(python='3.12',torch='test',cuda_build='test',transformers='test',gpu='A100',vram_bytes=80*1024**3)
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'torch':torch,'drex_max.model':sdk}), \
                patch.object(d,'context',return_value=(identity,{})),patch.object(d,'checked_sample',return_value=sample), \
                patch.object(runtime,'environment',return_value=env),patch.object(runtime,'remaining_bytes',return_value=0), \
                patch.object(table_group,'gpu_binding',side_effect=lambda *args:mode['rank']),patch('builtins.print'):
            root=Path(tmp);out=root/'run';write(out/'identity.json',dict(test=True))
            (root/'requirements.lock.txt').write_text('fixture')
            write(root/'models/drex-max-v1/decision_config.json',dict(format_version=1,attention_mode='noncausal_full_attention'))
            write(root/'models/drex-max-v1/config.json',dict(max_position_embeddings=1024))
            for rank in range(4):mode['rank']=rank;d.worker(root,out,4,'pilot')
            self.assertEqual(len(calls),2)
            d.report(root,out,4,'pilot')
            for rank in range(4):mode['rank']=rank;d.worker(root,out,4,'full')
            self.assertEqual(len(calls),4)  # No repeat of two warmup requests after their rank changes.
            d.report(root,out,4,'full')
            r=read(out/'report.json');self.assertEqual(r['status'],'choices_complete');self.assertFalse(r['performance_evaluated'])
            self.assertEqual(len(read(out/'decisions.json')),4)
            for rank in range(4):mode['rank']=rank;d.worker(root,out,4,'full')
            self.assertEqual(len(calls),4)
            mode['rank']=0;mode['tokens']=1024
            path=out/'results/t0-primary.json';path.with_suffix('.commit.json').unlink()
            with self.assertRaisesRegex(RuntimeError,'Context at limit'):d.worker(root,out,4,'full')
            self.assertEqual(len(calls),4)
            mode['tokens']=100;mode['fail']=True
            with self.assertRaisesRegex(RuntimeError,'out of memory'):d.worker(root,out,4,'full')
            self.assertFalse(path.with_suffix('.commit.json').exists())


class LaunchTests(unittest.TestCase):
    def test_supported_gpu_counts_auto_pin_and_one_allocation(self):
        import test_runtime as fixtures
        f=fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step');f.setUp();self.addCleanup(f.doCleanups)
        f.script=Path(d.__file__).with_name('run_direct255.sh')
        f.stub('sbatch','printf "%s\\n" "$@" > "$DREX_TEST_LOG"\n')
        for count in (1,2,4,8):
            r=f.run_stage('submit-pilot',str(count));self.assertEqual(r.returncode,0,r.stderr)
            args=f.log.read_text().splitlines();self.assertIn('--gres=gpu:'+str(count),args)
            self.assertIn('--ntasks='+str(count),args);self.assertIn('--mem='+str(96*count)+'G',args)
            self.assertFalse(any(a.startswith('--nodelist') for a in args));self.assertIn('--time=01:00:00',args)
        self.assertNotEqual(f.run_stage('submit-pilot','3').returncode,0)
        f.env['DREX_NODE']='cn03';r=f.run_stage('submit-pilot','4');self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('--nodelist=cn03',f.log.read_text())
        r=f.run_stage('submit-night','4');self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('coordinator-night',f.log.read_text());self.assertIn('--time=08:00:00',f.log.read_text())

    def test_night_uses_one_allocation_and_full_waits_for_pilot_report(self):
        import test_runtime as fixtures
        f=fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step');f.setUp();self.addCleanup(f.doCleanups)
        f.script=Path(d.__file__).with_name('run_direct255.sh').resolve();f.fake_setup()
        f.stub('srun','printf "%s\\n" "$*" >> "$DREX_TEST_CALLS"\n')
        python=f.envroot/'venv/bin/python'
        python.write_bytes(b'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$DREX_TEST_CALLS"\n'
                           b'if [[ ${DREX_TEST_FAIL_PILOT_REPORT:-0} == 1 && "$*" == *" report "* && "$*" == *"--phase pilot"* ]]; then exit 7; fi\n')
        for fail in (False,True):
            f.env['DREX_TEST_FAIL_PILOT_REPORT']='1' if fail else '0';f.calls.write_text('')
            r=f.run_stage('--inside','coordinator-night',f.posix(f.script.parents[2]),f.posix(f.envroot),'4','direct255_night_fixture')
            self.assertEqual(r.returncode,7 if fail else 0,r.stderr)
            calls=f.calls.read_text();self.assertIn('worker-pilot',calls)
            if fail:self.assertNotIn('worker-full',calls)
            else:self.assertLess(calls.index('--phase pilot'),calls.index('worker-full'))


if __name__=='__main__':unittest.main()
