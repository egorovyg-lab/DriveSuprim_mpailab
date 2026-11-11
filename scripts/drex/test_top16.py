import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
import lzma
from pathlib import Path
import pickle
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import top16_remote as remote
import top16_bridge as bridge
import top16_summary as summary
import test_runtime as shell_fixtures
from transfer import read, write, sha

AVAILABLE = bool(importlib.util.find_spec('numpy') and importlib.util.find_spec('shapely'))
if AVAILABLE:
    import numpy as np
    from shapely.geometry import box, LineString


class SelectionTests(unittest.TestCase):
    def test_all_96_inputs_are_selected_in_fixed_teacher_order(self):
        cases={f'case{i:02d}':f'token{i}' for i in range(1,7)}
        samples={t:dict(pool_local=list(range(100,116)),pool_vocab=list(range(500,516)),baseline_local=100)
                 for t in cases.values()}
        rows=remote.selection_for(cases,samples,{t:'a'*64 for t in samples})
        self.assertEqual(len(rows),96)
        self.assertEqual(sum(r['is_baseline'] for r in rows),6)
        self.assertEqual(len({(r['token'],r['candidate_local']) for r in rows}),96)
        self.assertEqual([r['teacher_rank'] for r in rows[:16]],list(range(1,17)))
        samples['token1']['pool_local'][0]=101
        with self.assertRaises(ValueError):remote.selection_for(cases,samples,{t:'a'*64 for t in samples})

    def test_missing_ranks_are_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError,'All 16 ranks'):
            remote.compare_scenes([dict(case_id='case01',teacher_rank=1)],(0.,))

    def test_progress_summary_exposes_stopping_and_empty_feasible_sets(self):
        rows=[]
        for rank in range(1,17):
            rows.append(dict(case_id='case01',token='token',teacher_rank=rank,progress_m=0. if rank==2 else 10.,
                raw_pdm=.5 if rank==1 else .9,full_footprint_covered_all_samples=True,lane_keeping=1,
                first_actor_contact_time_s=None,sampled_geometry_eligible_at_buffer_m={
                    '0.0':rank in (2,3),'0.5':False}))
        scene=remote.compare_scenes(rows,(0.,.5))[0]
        self.assertEqual(scene['buffers']['0.0']['eligible_ranks'],[2,3])
        self.assertEqual(scene['buffers']['0.0']['progress_not_less_than_ds_ranks'],[3])
        self.assertEqual(scene['buffers']['0.0']['best_pdm_rank'],2)
        self.assertEqual(scene['buffers']['0.0']['best_pdm_progress_m'],0)
        self.assertEqual(scene['buffers']['0.0']['maximum_progress_m'],10)
        self.assertIsNone(scene['buffers']['0.5']['best_pdm_rank'])
        self.assertEqual(scene['buffers']['0.5']['eligible_ranks'],[])

    def test_bridge_stages_the_correct_python39_entrypoint(self):
        for name in bridge.TOOLS:
            ast.parse(Path(bridge.__file__).with_name(name).read_text(encoding='utf-8'),feature_version=(3,9))
        self.assertIn('top16_remote.py',bridge.TOOLS)
        with self.assertRaisesRegex(ValueError,'Remote entrypoint'):
            bridge.bridge.run(NS(),request={},tools=bridge.TOOLS,remote_tool='../other.py')

    def test_failure_is_not_displayed_as_success(self):
        self.assertIn('parity_failed',summary.show(dict(status='parity_failed')))


if AVAILABLE:
    class FixtureMap:
        def __init__(self):self._geometries=[box(-100,-100,100,100)]
        def get_indices_of_map_type(self,_):return [0]
        def is_in_layer(self,*_):return False


@unittest.skipUnless(AVAILABLE,'NumPy/Shapely needed for end-to-end fake SDK test')
class ReplayTests(unittest.TestCase):
    def test_full_replay_freezes_before_labels_scores_singly_and_resumes_96_candidates(self):
        import real_data as base
        import rollout_replay as old
        temp=tempfile.TemporaryDirectory(dir=str(Path.cwd()));self.addCleanup(temp.cleanup)
        root=Path(temp.name); source=root/'drex_rollout_audit/source/artifacts';package=root/'package';out=root/'out';repo=root/'repo'
        source.mkdir(parents=True);package.mkdir();repo.mkdir()
        vocab=np.zeros((16,40,3));vocab[:,:,0]=np.arange(1,41)[None,:]*np.arange(1,17)[:,None]/40
        cases={f'case{i:02d}':f'token{i}' for i in range(1,7)}
        samples={t:dict(pool_local=list(range(16)),pool_vocab=list(range(16)),baseline_local=0) for t in cases.values()}
        manifest=dict(files={});index={};scenes={};known=[];caches={};cache_objects=[]
        keys=['pdm_score',*base.METRICS]
        def states_for(poses):
            states=np.zeros((41,11));states[1:,:3]=poses;return states
        def coords_for(states,vehicle):
            coords=np.zeros((*states.shape[:2],5,2));coords[:]=states[:,:,:2,None].transpose(0,1,3,2)
            coords+=np.array([[3,1],[-1,1],[-1,-1],[3,-1],[1,0]])
            return coords
        def label(rank):
            values=np.ones(9);values[0]=(rank+1)/16;values[keys.index('ego_progress')]=(rank+1)/16;return values
        for case,token in cases.items():
            path=package/'bundle/examples'/(token+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,label_indices=np.arange(16),label_raw=np.asarray([label(i) for i in range(16)]))
            manifest['files'][f'bundle/examples/{token}.npz']=dict(sha256=sha(path))
            index[token]=dict(metadata=dict(role='train'));scenes[token]=dict(log_name=token)
            cache=NS(log_name=token,ego_state=NS(time_point=None,rear_axle=NS(serialize=np.zeros(3).tolist)),
                drivable_area_map=FixtureMap(),centerline=NS(linestring=LineString([(-100,0),(100,0)])))
            cache_path=root/(token+'.xz');cache_path.write_bytes(lzma.compress(pickle.dumps(cache)))
            caches[token]=dict(path=str(cache_path),sha256=sha(cache_path))
            for member,local in ((0,0),(1,15)):
                folder=source/(case+'-member'+str(member));folder.mkdir()
                write(folder/'summary.json',dict(scorer_vehicle_dimensions_m={'width':2.}))
                np.savez_compressed(folder/'traces.npz',simulated_states=states_for(vocab[local]))
                write(folder/'commit.json',dict(files={name:sha(folder/name) for name in ('summary.json','traces.npz')}))
                known.append(dict(pair_id=case,token=token,member=member,candidate_local=local))
        write(package/'package_manifest.json',manifest)
        original=dict(candidates=known,package_manifest_sha256=sha(package/'package_manifest.json'))
        write(source.parent/'audit_request.json',original)
        write(source/'identity.json',dict(config={},scorer_sources={},caches=caches,request_sha256=sha(source.parent/'audit_request.json')))
        write(source/'report.json',dict(status='complete',candidates=12,metric_parity_passed=12,geometry_interpretation_ready=True))
        bindings={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
        request=root/'request.json';write(request,dict(protocol='clearance_top16_v1',source_artifacts=str(source),
            source_files=bindings,source_request_sha256=sha(source.parent/'audit_request.json'),package=str(package),
            package_manifest_sha256=sha(package/'package_manifest.json')))
        cfg=NS(traffic_agents='non_reactive',simulator='sim',scorer='scorer',agent_sampling='sampling',traffic_agents_policy=NS(non_reactive='policy'))
        sampling=NS(interval_length=.1)
        scorer=NS(_config=NS(human_penalty_filter=True,lane_keeping_deviation_limit=.5,lane_keeping_horizon_window=2.),
                  _vehicle_parameters=NS(width=2.))
        modules={}
        def module(name,**attributes):
            for i in range(1,len(name.split('.'))+1):
                prefix='.'.join(name.split('.')[:i]);modules.setdefault(prefix,ModuleType(prefix))
                modules[prefix].__path__=[]
            for k,v in attributes.items():setattr(modules[name],k,v)
        module('hydra.utils',instantiate=lambda obj,*args:{'sim':NS(proposal_sampling=sampling),'scorer':scorer,'sampling':sampling,'policy':NS()}[obj])
        module('omegaconf',OmegaConf=NS(create=lambda _:cfg))
        module('nuplan.common.maps.maps_datatypes',SemanticMapLayer=NS(ROADBLOCK=1,INTERSECTION=2,DRIVABLE_AREA=3,CARPARK_AREA=4))
        module('navsim.common.dataclasses',Trajectory=lambda poses,sampling:NS(poses=poses))
        calls=[]
        def score(cache,trajectory,*_):
            self.assertTrue((out/'selection.json').exists())
            self.assertFalse(hasattr(cache,'used'));cache.used=True;cache_objects.append(cache)
            i=int(round(trajectory.poses[-1,0]))-1;calls.append(i)
            states=states_for(trajectory.poses);scorer._ego_coords=coords_for(np.stack([states,states]),None)
            scorer._observation=NS(detections_tracks=[NS(tracked_objects=[]) for _ in range(41)])
            scorer._progress_raw=np.array([0.,float(i+1)]);scorer._collision_time_idcs=np.array([np.inf,np.inf])
            return NS(iloc=[dict(zip(keys,label(i)))]),states
        module('navsim.evaluate.pdm_score',pdm_score=score,transform_trajectory=lambda trajectory,_:trajectory,
               get_trajectory_as_array=lambda trajectory,*_:states_for(trajectory.poses))
        module('navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation',state_array_to_coords_array=coords_for)
        path_type=Path
        def mapped_path(value):
            return root/'drex_rollout_audit' if str(value)=='/workspace/exp_v2/drex_rollout_audit' else path_type(value)
        original_labels=remote.load_labels
        def guarded_labels(*args):
            self.assertEqual(len(read(out/'selection.json')['candidates']),96)
            return original_labels(*args)
        with patch.dict(sys.modules,modules),patch.object(remote,'Path',side_effect=mapped_path), \
             patch.object(base,'package_metadata',return_value=(manifest,index,scenes,vocab)), \
             patch.object(base,'load_inputs',side_effect=lambda package,token,*_:samples[token]), \
             patch.object(old,'map_membership',return_value=(np.ones((41,4),dtype=bool),[0])), \
             patch.object(remote.importlib.metadata,'version',return_value='fixture'), \
             patch.object(remote,'load_labels',side_effect=guarded_labels),redirect_stdout(io.StringIO()):
            self.assertEqual(remote.run(request,out,repo),0)
            report=read(out/'report.json')
            self.assertEqual(len(calls),96);self.assertEqual(report['metric_parity_passed'],96)
            self.assertEqual(report['known_pair_state_parity_passed'],12)
            self.assertEqual(remote.run(request,out,repo),0)
            self.assertEqual(read(out/'report.json')['resumed_candidates'],96)
            self.assertEqual(read(out/'report.json')['scorer_calls_this_invocation'],0)
            (out/'candidates/case01-rank01/summary.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Completed candidate artifact changed'):
                remote.run(request,out,repo)
        self.assertEqual(len(calls),96)
        self.assertTrue((out/'candidates.csv').exists())

    def test_label_alignment_and_nonfinite_values_are_rejected(self):
        temp=tempfile.TemporaryDirectory(dir=str(Path.cwd()));self.addCleanup(temp.cleanup)
        path=Path(temp.name)/'labels.npz';sample=dict(pool_local=list(range(16)))
        np.savez(path,label_indices=np.arange(16)[::-1],label_raw=np.ones((16,9)))
        with self.assertRaisesRegex(ValueError,'alignment/values'):remote.load_labels(path,sample)
        values=np.ones((16,9));values[4,3]=np.nan
        np.savez(path,label_indices=np.arange(16),label_raw=values)
        with self.assertRaisesRegex(ValueError,'alignment/values'):remote.load_labels(path,sample)


class ShellTests(unittest.TestCase):
    def test_status_does_not_allocate_resources_and_old_output_is_rejected(self):
        f=shell_fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step');f.setUp();self.addCleanup(f.doCleanups)
        f.script=Path(__file__).with_name('run_top16_audit.sh').resolve()
        f.env.pop('DREX_TOP16_NAME',None);f.env.pop('DREX_TOP16_SOURCE',None)
        result=f.run_stage('status');self.assertEqual(result.returncode,0,result.stderr);self.assertFalse(f.log.exists())
        f.env['DREX_TOP16_NAME']='rollout_audit_pair6_v2'
        self.assertNotEqual(f.run_stage('start').returncode,0)


if __name__=='__main__':unittest.main()
