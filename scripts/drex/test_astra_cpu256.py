import ast
import copy
import inspect
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import astra_cpu256 as a
import real_data as base


def fixture(protocol='navsim_v2',imitation=True):
    rng=np.random.default_rng(17)
    arrays={'head_'+m:rng.normal(size=256) for m in base.METRICS}
    arrays.update(candidate_ids=np.arange(256),head_imi=rng.normal(size=256))
    p={m:1/(1+np.exp(-arrays['head_'+m])) for m in base.METRICS}
    score=.5*np.log(p['no_at_fault_collisions'])+.5*np.log(p['drivable_area_compliance'])
    if protocol=='navsim_v1':score+=8*np.log(5*p['ego_progress']+5*p['time_to_collision_within_bound']+2*p['history_comfort'])
    else:score+=.1*np.log(p['traffic_light_compliance'])+.3*np.log(p['driving_direction_compliance'])+6*np.log(5*p['ego_progress']+5*p['time_to_collision_within_bound']+2*p['lane_keeping']+p['history_comfort'])
    if imitation:
        z=arrays['head_imi'];logp=z-np.log(np.exp(z).sum())
        score=score+(.05 if protocol=='navsim_v1' else .02)*logp
    arrays['selection_scores']=score;arrays['selected_local']=int(np.argmax(score))
    return arrays


class AstraTests(unittest.TestCase):
    def test_original_protocol_and_imi_reconstruction(self):
        for protocol in ('navsim_v1','navsim_v2'):
            for imitation in (True,False):
                s=fixture(protocol,imitation);methods,actual=a.selection(s)
                self.assertEqual(actual,dict(protocol=protocol,imitation_included=imitation))
                self.assertEqual(methods['baseline'],s['selected_local'])
                if not imitation:self.assertEqual(methods['no_final_imitation'],methods['baseline'])

    def test_proxy_does_not_consume_final_score_or_imitation(self):
        x=fixture();y=copy.deepcopy(x)
        y['head_imi']*=100
        p={m:1/(1+np.exp(-y['head_'+m])) for m in base.METRICS}
        z=y['head_imi']; logp=z-z.max()-np.log(np.exp(z-z.max()).sum())
        y['selection_scores']=.1*np.log(p['traffic_light_compliance'])+.5*np.log(p['no_at_fault_collisions'])+.5*np.log(p['drivable_area_compliance'])+.3*np.log(p['driving_direction_compliance'])+6*np.log(5*p['time_to_collision_within_bound']+5*p['ego_progress']+2*p['lane_keeping']+p['history_comfort'])+.02*logp
        y['selected_local']=int(np.argmax(y['selection_scores']))
        self.assertEqual(a.selection(x)[0]['absolute_head_proxy'],a.selection(y)[0]['absolute_head_proxy'])

    def test_mismatched_original_scores_and_invalid_heads_stop(self):
        x=fixture();x['selection_scores'][5]+=1
        with self.assertRaisesRegex(RuntimeError,'reproduce'):a.selection(x)
        x=fixture();x['head_ego_progress'][1]=np.nan
        with self.assertRaises(ValueError):a.selection(x)

    def test_unknown_labels_are_not_fabricated_or_old_parity_disabled(self):
        source=inspect.getsource(a.evaluate)
        self.assertIn('if expected_raw is not None:old.parity(raw,expected_raw)',source)
        self.assertNotIn('expected_raw=np.zeros',source)
        self.assertNotIn('old.parity =',source)
        self.assertLess(inspect.getsource(a.execute).index('request=plan('),inspect.getsource(a.execute).index("prior=read("))

    def test_python39_cpu_only_and_detached_launch(self):
        ast.parse(Path(a.__file__).read_text(),feature_version=(3,9))
        shell=Path(a.__file__).with_name('run_astra_cpu256.sh').read_text()
        self.assertIn('start_new_session=True',shell);self.assertIn("CUDA_VISIBLE_DEVICES=''",shell)
        self.assertIn(' pilot ',shell);self.assertIn(' full ',shell)
        self.assertNotIn('7*3600',Path(a.__file__).read_text())

    def test_pilot_full_reuse_and_resume_with_unknown_tail_labels(self):
        import continuous_remote as old
        import continuous_metrics as cm
        import pool_remote as pool
        import pool_reserve as reserve
        from transfer import write,sha
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';out=root/'output';out.mkdir()
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
                         oracle_tokens=['0'],pilot_tokens=['0','1'],manifest_sha256='fixture')
            write(out/'request.json',request)
            def known(folder,*args):
                return [c for c in candidates if c['token']==folder.name]
            def evaluate(c,*args):
                return dict(**c,measured_raw=raw,metrics=metric)
            with patch.object(a,'plan',return_value=request),patch.object(old.replay,'bound_context',return_value=({},None,None,{},caches)), \
                    patch.object(old.replay,'config_core',return_value={}),patch.object(pool,'load_block',side_effect=known), \
                    patch.object(a,'evaluate',side_effect=evaluate) as scorer,patch.object(reserve,'pick',side_effect=lambda g,*args:g[0]), \
                    patch.object(cm,'paired_stats',side_effect=lambda v,s:dict(macro_mean=float(np.mean(v)))),patch('builtins.print'):
                pilot=a.execute(root,source,out,root,'pilot')
                self.assertEqual(pilot['new_candidates_this_invocation'],193)
                self.assertEqual(pilot['status'],'pilot_complete')
                full=a.execute(root,source,out,root,'full')
                self.assertEqual(full['new_candidates_this_invocation'],2)
                scorer.reset_mock();resumed=a.execute(root,source,out,root,'full')
                self.assertEqual(resumed['new_candidates_this_invocation'],0);scorer.assert_not_called()
                self.assertEqual(resumed['methods']['no_final_imitation']['scenes'],4)
                self.assertEqual(resumed['methods']['oracle_pdm_top256']['scenes'],1)


if __name__=='__main__':unittest.main()
