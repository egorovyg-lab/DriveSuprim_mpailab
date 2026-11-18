import ast
import copy
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import astra_cpu256 as a
import astra_drex_probe as probe
from transfer import read,write,sha


SOURCE=b'MAX_OPTIONS = 255\nclass Model:\n def __init__(self):\n  self.readout = torch.nn.Linear(hidden, MAX_OPTIONS, bias=False)\n'
BLOB=hashlib.sha1(b'blob '+str(len(SOURCE)).encode()+b'\0'+SOURCE).hexdigest()


def decision():
    return dict(format_version=1,codes=['C'+str(i) for i in range(255)],token_ids=list(range(255)))


def legacy(root):
    records=[dict(token=str(i),session_id='s'+str(i%32),role='train') for i in range(651)]
    request=dict(config=a.CONFIG,records=records,oracle_tokens=[str(i) for i in range(64)],
        methods={},candidates={},formulas={},manifest_sha256='package',source='source',
        source_identity_sha256='identity',source_request_sha256='request',source_report_sha256='report',drex_sha256=None)
    write(root/'request.json',request)
    write(root/'identity.json',dict(request_sha256=sha(root/'request.json'),code_sha256=a.LEGACY_CODE_SHA256))
    write(root/'report.json',dict(status='complete',scenes=651,oracle_scenes=64,anchors_parity_passed=651))
    current=copy.deepcopy(request);current['oracle_tokens']=[str(i) for i in range(651)]
    current['reuse']=dict(path=str(root),files={n:sha(root/n) for n in ('request.json','identity.json','report.json')})
    return current


class ProbeTests(unittest.TestCase):
    def test_fixed_readout_rejects_256_independent_of_gpu_memory(self):
        with patch.object(probe,'MODEL_BLOB',BLOB):
            r=probe.contract(SOURCE,decision())
            self.assertEqual(r['status'],'blocked_choice_capacity')
            self.assertEqual(r['official_readout_outputs'],255)
            self.assertFalse(r['candidate_dropped']);self.assertFalse(r['architecture_modified'])
            self.assertEqual(r['weight_download_bytes'],0)
            self.assertEqual(probe.contract(SOURCE,decision(),255)['status'],'contract_passed_not_inference_tested')

    def test_source_and_checkpoint_mismatch_are_errors_not_capacity_success(self):
        with self.assertRaisesRegex(RuntimeError,'blob changed'):probe.source_capacity(SOURCE)
        with patch.object(probe,'MODEL_BLOB',BLOB):
            c=decision();c['token_ids'][1]=0
            with self.assertRaisesRegex(RuntimeError,'disagrees'):probe.contract(SOURCE,c)
            c=decision();c['codes'].append('new')
            with self.assertRaises(RuntimeError):probe.contract(SOURCE,c)

    def test_probe_only_fetches_two_small_metadata_files(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(probe,'MODEL_BLOB',BLOB), \
                patch.object(probe,'DECISION_SHA256',hashlib.sha256(json.dumps(decision()).encode()).hexdigest()), \
                patch.object(probe,'hardware',return_value={'gpu':'fixture'}), \
                patch.object(probe,'fetch',side_effect=[SOURCE,json.dumps(decision()).encode()]) as fetch,patch('builtins.print'):
            root=Path(tmp);self.assertEqual(probe.run(root),2)
            report=read(root/'report.json')
            self.assertEqual(report['status'],'blocked_choice_capacity');self.assertEqual(report['inference_calls'],0)
            urls=[c.args[0] for c in fetch.call_args_list]
            self.assertEqual(len(urls),2);self.assertTrue(all('.safetensors' not in u for u in urls))
            self.assertIn(probe.UPSTREAM,urls[0]);self.assertIn(probe.REVISION,urls[1])

    def test_network_failure_does_not_claim_capacity_was_verified(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(probe,'hardware',return_value={}), \
                patch.object(probe,'fetch',side_effect=TimeoutError('offline')),patch('builtins.print'):
            root=Path(tmp);self.assertEqual(probe.run(root),1)
            r=read(root/'report.json');self.assertEqual(r['status'],'technical_failure')
            self.assertNotIn('official_readout_outputs',r)


class ReuseTests(unittest.TestCase):
    def test_completed_source_must_bind_same_cohort_package_scorer_and_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);request=legacy(root)
            self.assertEqual(a.reuse_context(request),(root,sha(root/'identity.json')))
            for key in ('manifest_sha256','source_request_sha256','methods'):
                changed=copy.deepcopy(request);changed[key]='different'
                with self.assertRaisesRegex(RuntimeError,'binding differs'):a.reuse_context(changed)
            write(root/'report.json',dict(status='pilot_complete'))
            with self.assertRaisesRegex(RuntimeError,'metadata changed'):a.reuse_context(request)

    def test_unfinished_or_unrecognized_code_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r=legacy(root)
            value=read(root/'identity.json');value['code_sha256']='unknown';write(root/'identity.json',value)
            r['reuse']['files']['identity.json']=sha(root/'identity.json')
            with self.assertRaisesRegex(RuntimeError,'original cd0041e'):a.reuse_context(r)
            r=legacy(root);value=read(root/'report.json');value['status']='pilot_complete';write(root/'report.json',value)
            r['reuse']['files']['report.json']=sha(root/'report.json')
            with self.assertRaisesRegex(RuntimeError,'incomplete'):a.reuse_context(r)

    def test_receipts_physical_ids_rank_and_finite_metrics_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);info=dict(vocab=list(range(256)),ranked=list(range(256)))
            path=root/'scenes/t/100.json';commit=root/'scenes/t/100.commit.json'
            value=dict(token='t',candidate_local=100,candidate_vocab=100,teacher_rank=101,measured_raw=[1.]*9,metrics={'m':.1})
            write(path,value)
            self.assertIsNone(a.reused_candidate((root,'binding'),'t',100,info,{'m':0}))
            write(commit,dict(identity_sha256='binding',sha256=sha(path)))
            self.assertEqual(a.reused_candidate((root,'binding'),'t',100,info,{'m':0}),value)
            value['candidate_vocab']=101;write(path,value)
            with self.assertRaisesRegex(RuntimeError,'receipt changed'):a.reused_candidate((root,'binding'),'t',100,info,{'m':0})
            write(commit,dict(identity_sha256='binding',sha256=sha(path)))
            with self.assertRaisesRegex(RuntimeError,'physical'):a.reused_candidate((root,'binding'),'t',100,info,{'m':0})
            value['candidate_vocab']=100;value['measured_raw'][0]=float('nan')
            with self.assertRaisesRegex(RuntimeError,'raw metrics'):a.check_candidate(value,'t',100,info,{'m':0})
            value['measured_raw'][0]=1.;value['metrics']['m']=float('inf')
            with self.assertRaisesRegex(RuntimeError,'continuous'):a.check_candidate(value,'t',100,info,{'m':0})

    def test_pilot_all_scene_plan_is_input_only_and_keeps_full_256(self):
        from test_astra_cpu256 import fixture
        records=[dict(token=str(i),session_id='s'+str(i%32),role='train') for i in range(651)]
        index={r['token']:dict(metadata=r) for r in records};arrays=fixture()
        class Arrays(dict):
            def __enter__(self):return self
            def __exit__(self,*args):pass
        manifest=dict(files={'bundle/examples/'+r['token']+'.npz':dict(sha256='fixture') for r in records})
        with tempfile.TemporaryDirectory() as tmp,patch('real_data.package_metadata',return_value=(manifest,index,None,None)), \
                patch('transfer.sha',return_value='fixture'),patch.object(a.np,'load',return_value=Arrays(arrays)):
            root=Path(tmp);r=a.plan(root,root,root/'out',oracle_all=True)
            self.assertEqual(len(r['oracle_tokens']),651);self.assertEqual(r['config']['oracle_scenes'],651)
            self.assertEqual(len(r['pilot_tokens']),4)
            sessions={x['session_id'] for x in records if x['token'] in r['pilot_tokens']}
            self.assertEqual(len(sessions),4)
            self.assertTrue(all(len(c['vocab'])==256 for c in r['candidates'].values()))
            self.assertEqual(a.CONFIG['oracle_scenes'],64)  # Old protocol is unchanged.


class PipelineTests(unittest.TestCase):
    def test_shell_passes_full_pool_and_reuse_flags_and_stops_on_failed_pilot(self):
        bash=Path(r'C:\Program Files\Git\bin\bash.exe')
        bash=str(bash) if bash.exists() else shutil.which('bash')
        if not bash:self.skipTest('Bash unavailable')
        def posix(path):
            return subprocess.check_output([bash,'-c','cygpath -u "$1"','path',str(path)],text=True).strip() if os.name=='nt' else str(path)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root=Path(tmp);binpath=root/'bin';binpath.mkdir();log=root/'calls'
            def stub(name,text):
                file=binpath/name
                # Path.write_text(newline=...) requires Python 3.10; Astra is 3.9.
                with file.open('w',encoding='utf-8',newline='\n') as stream:
                    stream.write('#!/usr/bin/env bash\n'+text)
                file.chmod(0o755)
            stub('flock','exit 0\n')
            stub('python','printf "%s\\n" "$*" >> "$ASTRA_TEST_CALLS"\n'
                 '[[ ${CUDA_VISIBLE_DEVICES-unset} == "" && $OMP_NUM_THREADS == 2 ]] || exit 90\n'
                 'if [[ ${ASTRA_TEST_FAIL_PILOT:-0} == 1 && "$*" == *" pilot "* ]]; then exit 7; fi\n')
            script=Path(a.__file__).with_name('run_astra_cpu256.sh').resolve()
            env=dict(os.environ,ASTRA_CPU256_ROOT=posix(root/'out'),ASTRA_CPU256_SOURCE=posix(root/'source'),
                     ASTRA_CPU256_REUSE=posix(root/'reuse'),ASTRA_CPU256_ORACLE_ALL='1',ASTRA_TEST_CALLS=posix(log))
            env.pop('ASTRA_CPU256_DECISIONS',None)
            for fail in (False,True):
                (root/'out').mkdir(exist_ok=True);env['ASTRA_TEST_FAIL_PILOT']='1' if fail else '0'
                log.write_text('')
                r=subprocess.run([bash,'-c','export PATH="$1:$PATH"; exec bash "$2" --worker','runner',posix(binpath),posix(script)],
                                 env=env,capture_output=True,text=True,timeout=15)
                calls=log.read_text().splitlines();pilot=[s for s in calls if ' pilot ' in s];full=[s for s in calls if ' full ' in s]
                self.assertEqual(r.returncode,7 if fail else 0,r.stderr)
                self.assertEqual(len(pilot),1);self.assertEqual(len(full),0 if fail else 1)
                self.assertIn('--oracle-all',pilot[0]);self.assertIn('--reuse',pilot[0])

    def test_expansion_compares_matched_scenes_and_reports_session_macro(self):
        rows=[]
        for i,gain in enumerate((.01,.03,.07)):
            for pool,score in ((16,.5),(64,.5),(256,.5+gain)):
                rows.append(dict(token=str(i),session_id='s0' if i<2 else 's1',method='oracle_pdm_top'+str(pool),raw_pdm=score))
        r=a.pool_expansion(rows)['256_vs_64']
        self.assertAlmostEqual(r['gain_pp']['macro_mean'],4.5)
        self.assertAlmostEqual(r['gain_pp']['micro_mean'],11/3)
        self.assertAlmostEqual(r['top2_fraction_of_positive_micro_gain'],10/11)
        self.assertEqual(r['improved_sessions'],2)
        with self.assertRaisesRegex(RuntimeError,'cohorts differ'):a.pool_expansion(rows[:-1])

    def test_full_pool_reuses_tail_then_resumes_without_rescoring_or_csv_duplicates(self):
        import continuous_remote as old
        import continuous_metrics as cm
        import pool_remote as pool
        import pool_reserve as reserve
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';out=root/'out';out.mkdir()
            records=[dict(token=str(i),session_id='s'+str(i),baseline_local=0) for i in range(4)]
            info={r['token']:dict(vocab=list(range(256)),ranked=list(range(256))) for r in records}
            methods={r['token']:dict(baseline=0,no_final_imitation=180,absolute_head_proxy=0) for r in records}
            raw=[.9]+[1.]*8
            metric=dict(route_progress_m=10.,all_jerk_norm_mps3_rms=.2,all_accel_lat_mps2_rms=.1,
                        all_actor_contact_time_s=0.,all_outside_area_integral_m2s=0.)
            candidates=[dict(token=r['token'],candidate_local=j,candidate_vocab=j,teacher_rank=j+1,measured_raw=raw,metrics=metric)
                        for r in records for j in range(64)]
            caches={r['token']:dict(path='fixture',sha256='fixture') for r in records}
            write(source/'audit_request.json',dict(records=records,candidates=candidates,tools={},package_manifest_sha256='fixture'))
            write(source/'artifacts/identity.json',dict(request_sha256=sha(source/'audit_request.json'),caches=caches,config={},sources={}))
            write(source/'artifacts/report.json',dict(status='complete',metric_parity_passed=41664))
            write(source/'tools/vehicle_profile_astra.json',dict(dimensions_m={}))
            request=dict(config=a.CONFIG,records=records,methods=methods,candidates=info,
                         oracle_tokens=[r['token'] for r in records],pilot_tokens=['0','1'],manifest_sha256='fixture')
            write(out/'request.json',request)
            measured=[];interrupted=[False]
            def evaluate(c,*args):
                key=(c['token'],c['candidate_local'])
                if key==('1',70) and not interrupted[0]:
                    interrupted[0]=True;raise RuntimeError('simulated process interruption')
                measured.append(key)
                return dict(**c,measured_raw=raw,metrics=metric)
            def reuse(ctx,token,j,*args):
                return dict(token=token,candidate_local=j,candidate_vocab=j,teacher_rank=j+1,measured_raw=raw,metrics=metric) if token=='0' else None
            with patch.object(a,'plan',return_value=request),patch.object(a,'reuse_context',return_value=None), \
                    patch.object(old.replay,'bound_context',return_value=({},None,None,{},caches)), \
                    patch.object(old.replay,'config_core',return_value={}), \
                    patch.object(pool,'load_block',side_effect=lambda folder,*args:[c for c in candidates if c['token']==folder.name]), \
                    patch.object(a,'evaluate',side_effect=evaluate) as scorer,patch.object(a,'reused_candidate',side_effect=reuse), \
                    patch.object(reserve,'pick',side_effect=lambda g,*args:g[0]), \
                    patch.object(cm,'paired_stats',side_effect=lambda v,s:dict(macro_mean=float(np.mean(v)))),patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError,'process interruption'):a.execute(root,source,out,root,'pilot')
                self.assertFalse((out/'pilot_complete.json').exists())
                a.execute(root,source,out,root,'pilot')
                full=a.execute(root,source,out,root,'full')
                self.assertEqual(full['completed_candidates'],1024)
                self.assertEqual(full['oracle_scenes'],4)
                self.assertNotIn('control64/baseline',full['methods'])
                scorer.reset_mock();again=a.execute(root,source,out,root,'full')
                scorer.assert_not_called();self.assertEqual(again['new_candidates_this_invocation'],0)
            self.assertEqual(len(measured),len(set(measured)))
            self.assertEqual(len(measured),4+3*192)  # Four anchors, only three unknown tails.
            with gzip.open(out/'per_candidate.csv.gz','rt',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),1024)
            self.assertEqual(len({(r['token'],r['candidate_local']) for r in rows}),1024)

    def test_new_scripts_are_python39_and_launcher_is_cpu_only_detached(self):
        for module in (a,probe):ast.parse(Path(module.__file__).read_text(encoding='utf-8'),feature_version=(3,9))
        shell=Path(a.__file__).with_name('run_astra_cpu256.sh').read_text(encoding='utf-8')
        self.assertIn('start_new_session=True',shell);self.assertIn("CUDA_VISIBLE_DEVICES=''",shell)
        self.assertIn('--oracle-all',shell);self.assertIn('--reuse',shell)
        self.assertNotIn('pip install',Path(probe.__file__).read_text())


if __name__=='__main__':unittest.main()
