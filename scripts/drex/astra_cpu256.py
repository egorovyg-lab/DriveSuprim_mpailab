"""Astra-only: fixed 256 controls, balanced-64 oracle, and committed Drex choices."""
import argparse
import gzip
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

CONFIG=dict(version=1,seed=20261117,oracle_scenes=64,oracle_sessions=32,oracle_per_session=2,
            progress_budget_m=1.,progress_budget_fraction=.05,score_match_atol=1e-4)
NOTE=('Exploratory previously inspected TRAIN replay, not holdout or EPDMS. '
      'Controls freeze before outcomes; oracle uses privileged outcomes and is not a policy. '
      'Raw PDM comes from the original scorer, not reconstructed filtered components. '
      'Head proxy is an uncalibrated deterministic control, not predicted safety. '
      'Same fixed 256 vocabulary candidates; removing final imitation does not undo earlier candidate filtering.')


def selection(arrays):
    import real_data as base
    logits={m:np.asarray(arrays['head_'+m],float) for m in base.METRICS}
    ids=np.asarray(arrays['candidate_ids']); saved=np.asarray(arrays['selection_scores'],float)
    if ids.shape!=(256,) or len(set(ids.tolist()))!=256 or saved.shape!=(256,):raise ValueError('Invalid 256 pool')
    if any(v.shape!=(256,) or not np.isfinite(v).all() for v in logits.values()) or not np.isfinite(saved).all():raise ValueError('Invalid heads/scores')
    logp={k:-np.logaddexp(0.,-v) for k,v in logits.items()}; p={k:np.exp(v) for k,v in logp.items()}
    common=.5*logp['no_at_fault_collisions']+.5*logp['drivable_area_compliance']
    versions={
        'navsim_v1':common+8*np.log(5*p['ego_progress']+5*p['time_to_collision_within_bound']+2*p['history_comfort']),
        'navsim_v2':common+.1*logp['traffic_light_compliance']+.3*logp['driving_direction_compliance']
            +6*np.log(5*p['ego_progress']+5*p['time_to_collision_within_bound']+2*p['lane_keeping']+p['history_comfort'])}
    imi=np.asarray(arrays['head_imi'],float)
    if imi.shape!=(256,) or not np.isfinite(imi).all():raise ValueError('Invalid imitation logits')
    imi=imi-np.max(imi); imi-=np.log(np.exp(imi).sum())
    matches=[]
    for name,score in versions.items():
        for included in (True,False):
            full=score+({'navsim_v1':.05,'navsim_v2':.02}[name]*imi if included else 0.)
            if np.allclose(full,saved,atol=CONFIG['score_match_atol'],rtol=1e-5):matches.append((name,included,score))
    if len(matches)!=1:raise RuntimeError('Cannot uniquely reproduce original DS selection formula: '+str([(n,i) for n,i,_ in matches]))
    baseline=int(arrays['selected_local'])
    if baseline!=int(np.argmax(saved)):raise ValueError('DS baseline mismatch')
    rounded={k:np.round(v,4) for k,v in p.items()}
    proxy=np.prod([rounded[k] for k in base.SAFETY[:4]],axis=0)*(
        5*rounded['ego_progress']+5*rounded['time_to_collision_within_bound']+
        2*rounded['lane_keeping']+2*rounded['history_comfort'])/14.
    # Proxy ties use vocabulary ID, never DS score/rank or measured outcomes.
    best=min(range(256),key=lambda j:(-proxy[j],int(ids[j])))
    name,included,noimi=matches[0]
    return dict(baseline=baseline,no_final_imitation=int(np.argmax(noimi)),absolute_head_proxy=best),dict(protocol=name,imitation_included=included)


def plan(package,source,out,decisions=None):
    import real_data as base
    from transfer import read,freeze,sha
    manifest,index,_,_=base.package_metadata(package)
    records=[dict(v['metadata']) for _,v in sorted(index.items())]
    if len(records)!=651 or len({r['session_id'] for r in records})!=32 or any(r['role']!='train' for r in records):raise ValueError('Expected original TRAIN cohort')
    groups={s:[r for r in records if r['session_id']==s] for s in {r['session_id'] for r in records}}
    key=lambda v:base.digest([CONFIG['seed'],v])
    sids=sorted(groups,key=key)
    ranked={s:sorted(groups[s],key=lambda r:key(r['token'])) for s in sids}
    if any(len(v)<2 for v in ranked.values()):raise ValueError('Two scenes per session required')
    oracle=[ranked[s][i]['token'] for i in range(2) for s in sids]
    pilot=oracle[:4]; methods={}; candidates={}; formulas={}
    for r in records:
        token=r['token']; path=package/'bundle/examples'/(token+'.npz')
        if sha(path)!=manifest['files']['bundle/examples/'+token+'.npz']['sha256']:raise RuntimeError('Example changed')
        with np.load(path,allow_pickle=False) as a:
            methods[token],formulas[token]=selection(a)
            ids=a['candidate_ids'].tolist(); ranking=np.argsort(-a['selection_scores'],kind='stable').tolist()
        candidates[token]=dict(vocab=ids,ranked=ranking)
    if decisions:
        rows=read(decisions); seen=set(); lookup={r['token'] for r in records}
        for r in rows:
            key=(r['token'],r['order']);token,order=key
            if token not in lookup or order not in ('primary','reversed') or key in seen:raise ValueError('Invalid/duplicate Drex choice')
            j=r['candidate_local']
            if not isinstance(j,int) or not 0<=j<256 or candidates[token]['vocab'][j]!=r['candidate_vocab']:raise ValueError('Drex vocabulary mismatch')
            seen.add(key);methods[token]['drex_'+order]=j
        if seen!={(t,o) for t in lookup for o in ('primary','reversed')}:raise ValueError('Complete 651-scene committed Drex decisions required')
    request=dict(config=CONFIG,records=records,oracle_tokens=[] if decisions else oracle,pilot_tokens=pilot,methods=methods,candidates=candidates,
        formulas=formulas,package=str(package),manifest_sha256=sha(package/'package_manifest.json'),source=str(source),
        source_identity_sha256=sha(source/'artifacts/identity.json'),source_request_sha256=sha(source/'audit_request.json'),
        source_report_sha256=sha(source/'artifacts/report.json'),drex_sha256=sha(decisions) if decisions else None,note=NOTE)
    freeze(out/'request.json',request);return request


def evaluate(candidate, cache_info, template, scenes, vocab, expected_vehicle, expected_raw=None):
    """Known anchors require parity. New candidates have genuinely unknown labels."""
    import lzma,pickle
    import clearance_math
    import continuous_metrics as cm
    import continuous_remote as old
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_center_state_array,state_array_to_coords_array
    from transfer import sha
    cfg=OmegaConf.create(old.replay.config_core(template))
    if cfg.traffic_agents!='non_reactive':raise RuntimeError('Unexpected traffic policy')
    sim,scorer=instantiate(cfg.simulator),instantiate(cfg.scorer);scorer._config.human_penalty_filter=False
    if abs(sim.proposal_sampling.interval_length-.1)>1e-12:raise RuntimeError('Unexpected timestep')
    for k,v in expected_vehicle.items():
        if abs(float(getattr(scorer._vehicle_parameters,k))-v)>1e-9:raise RuntimeError('Vehicle profile changed')
    if sha(Path(cache_info['path']))!=cache_info['sha256']:raise RuntimeError('Cache changed')
    with lzma.open(cache_info['path'],'rb') as f:cache=pickle.load(f)
    if cache.log_name!=scenes[candidate['token']]['log_name']:raise RuntimeError('Cache scene mismatch')
    policy=instantiate(cfg.traffic_agents_policy.non_reactive,sim.proposal_sampling)
    frame,states=pdm_score(cache,Trajectory(vocab[candidate['candidate_vocab']].astype(float),instantiate(cfg.agent_sampling)),
        sim.proposal_sampling,sim,scorer,policy)
    raw=[float(frame.iloc[0][k]) for k in ['pdm_score',*old.base.METRICS]]
    if not np.isfinite(raw).all() or np.any(np.asarray(raw)<0) or np.any(np.asarray(raw)>1):raise RuntimeError('Invalid raw score')
    if expected_raw is not None:old.parity(raw,expected_raw)
    states=np.asarray(states,float)
    if states.shape!=(41,11) or not np.isfinite(states).all():raise RuntimeError('Invalid states')
    if not np.allclose(states[0,:3],cache.ego_state.rear_axle.serialize(),atol=1e-9,rtol=0):raise RuntimeError('Origin mismatch')
    vehicle=cache.ego_state.car_footprint.vehicle_parameters
    center=state_array_to_center_state_array(states[None],vehicle)[0]
    expected_velocity=states[:,3:5].copy();expected_velocity[:,1]+=float(vehicle.rear_axle_to_center)*states[:,9]
    if not np.allclose(center[:,3:5],expected_velocity,atol=1e-10,rtol=0):raise RuntimeError('Center velocity convention changed')
    coords=state_array_to_coords_array(states[None],scorer._vehicle_parameters)[0]
    if not np.allclose(coords,scorer._ego_coords[1],atol=1e-9,rtol=0):raise RuntimeError('Scorer footprint mismatch')
    indices=cache.drivable_area_map.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK,SemanticMapLayer.INTERSECTION,SemanticMapLayer.DRIVABLE_AREA,SemanticMapLayer.CARPARK_AREA])
    road=clearance_math.road_union([cache.drivable_area_map._geometries[i] for i in indices])
    traffic=policy.simulate_environment(states,cache)
    if len(traffic)!=41:raise RuntimeError('Actor grid mismatch')
    frames=[clearance_math.frame_measurement(c,road,[(obj.track_token,obj.box.geometry) for obj in f.tracked_objects]) for c,f in zip(coords,traffic)]
    metrics=cm.measure(center,frames,.1,float(scorer._progress_raw[1]))
    return dict(**candidate,measured_raw=raw,metrics=metrics,metric_parity_passed=expected_raw is not None)


def execute(package,source,out,repo,stage,decisions=None):
    import continuous_remote as old
    import continuous_metrics as cm
    import pool_remote as pool
    import pool_reserve as reserve
    from transfer import read,write,freeze,sha
    import sys
    sys.path.insert(1,str(repo))
    request=plan(package,source,out,decisions)  # No outcomes loaded until the choices commit exists.
    prior=read(source/'audit_request.json'); sid=read(source/'artifacts/identity.json'); sr=read(source/'artifacts/report.json')
    if sr['status']!='complete' or sr['metric_parity_passed']!=41664 or sid['request_sha256']!=sha(source/'audit_request.json'):
        raise RuntimeError('Complete SHA-bound top64 source required')
    if prior['package_manifest_sha256']!=request['manifest_sha256']:raise RuntimeError('Source package differs')
    if {r['token'] for r in prior['records']}!={r['token'] for r in request['records']}:raise RuntimeError('Source cohort differs')
    for name,h in prior['tools'].items():
        if sha(source/'tools'/name)!=h:raise RuntimeError('Original helper changed')
    manifest,scenes,vocab,template,caches=old.replay.bound_context(package,repo,prior['candidates'])
    if template.get('map_root'):os.environ['NUPLAN_MAPS_ROOT']=template['map_root']
    if sid['caches']!=caches or sid['config']!=old.replay.config_core(template):raise RuntimeError('Original scorer/cache binding changed')
    for name,h in sid['sources'].items():
        if sha(repo/name)!=h:raise RuntimeError('Original scorer implementation changed: '+name)
    profile=read(source/'tools/vehicle_profile_astra.json')['dimensions_m']
    identity=dict(request_sha256=sha(out/'request.json'),code_sha256=sha(Path(__file__)),python=platform.python_version(),numpy=np.__version__)
    freeze(out/'identity.json',identity); binding=sha(out/'identity.json')
    prior_records={r['token']:r for r in prior['records']}; prior_candidates={t:[] for t in prior_records}
    for c in prior['candidates']:prior_candidates[c['token']].append(c)
    rows=[]; candidate_rows=[]; new=0; baseline_replays=0; start=time.monotonic()
    records=[r for r in request['records'] if stage=='full' or r['token'] in request['pilot_tokens']]
    if stage=='full' and (not (out/'pilot_complete.json').exists() or read(out/'pilot_complete.json')!=dict(identity_sha256=binding)):
        raise RuntimeError('Successful matching pilot required')
    for number,r in enumerate(records,1):
        token=r['token']; cached=pool.load_block(source/'artifacts/scenes'/token,prior_records[token],
            sorted(prior_candidates[token],key=lambda c:c['teacher_rank']),sha(source/'artifacts/identity.json'))
        known={c['candidate_local']:c for c in cached}; wanted=set(request['methods'][token].values())
        info=request['candidates'][token]; baseline=request['methods'][token]['baseline']
        if baseline not in known:raise RuntimeError('Original baseline absent')
        for j,c in known.items():
            if info['vocab'][j]!=c['candidate_vocab'] or info['ranked'][c['teacher_rank']-1]!=j:raise RuntimeError('Cached physical ID/rank changed')
        folder=out/'scenes'/token; anchor=folder/'anchor.json'
        anchor_expected=dict(identity_sha256=binding,cache=caches[token],baseline_raw=known[baseline]['measured_raw'])
        if anchor.exists():
            if read(anchor)!=anchor_expected:raise RuntimeError('Anchor binding changed')
        else:
            check=evaluate({k:known[baseline][k] for k in ('token','candidate_local','candidate_vocab','teacher_rank')},caches[token],template,scenes,vocab,profile,known[baseline]['measured_raw'])
            baseline_replays+=1
            for k,v in known[baseline]['metrics'].items():
                a=check['metrics'][k]
                if (a is None)!=(v is None) or (v is not None and not np.isclose(a,v,atol=1e-8,rtol=1e-7)):raise RuntimeError('Anchor continuous metric differs: '+k)
            freeze(anchor,anchor_expected)
        if token in request['oracle_tokens']:wanted=set(range(256))
        group={}
        for j in sorted(wanted):
            if j in known:group[j]=known[j];continue
            path=folder/(str(j)+'.json'); commit=folder/(str(j)+'.commit.json')
            if commit.exists():
                cp=read(commit)
                if cp!=dict(identity_sha256=binding,sha256=sha(path)):raise RuntimeError('Candidate checkpoint changed')
                value=read(path)
            else:
                candidate=dict(token=token,candidate_local=j,candidate_vocab=info['vocab'][j],teacher_rank=info['ranked'].index(j)+1)
                value=evaluate(candidate,caches[token],template,scenes,vocab,profile)
                write(path,value);write(commit,dict(identity_sha256=binding,sha256=sha(path)));new+=1
            if (value['token'],value['candidate_local'],value['candidate_vocab'])!=(token,j,info['vocab'][j]):raise RuntimeError('Checkpoint ID mismatch')
            group[j]=value
        methods=dict(request['methods'][token])
        if token in request['oracle_tokens']:
            for k in (16,64,256):
                eligible=[c for c in group.values() if c['teacher_rank']<=k]
                methods['oracle_pdm_top'+str(k)]=max(eligible,key=lambda c:(c['measured_raw'][0],-c['teacher_rank']))['candidate_local']
                safe=[c for c in eligible if np.all(np.asarray(c['measured_raw'])[1:][[0,1,2,3,5,6,7]]>=np.asarray(known[baseline]['measured_raw'])[1:][[0,1,2,3,5,6,7]]-1e-8)]
                methods['relative_safe_oracle_top'+str(k)]=max(safe,key=lambda c:(c['measured_raw'][0],-c['teacher_rank']))['candidate_local']
                for objective in ('jerk','lateral'):
                    methods['oracle_'+objective+'_top'+str(k)]=reserve.pick(list(group.values()),baseline,k,1.,.05,'geometry_peaks',objective)['candidate_local']
        b=known[baseline]
        for method,j in methods.items():
            c=group[j];delta=np.asarray(c['measured_raw'])-np.asarray(b['measured_raw'])
            row=dict(token=token,session_id=r['session_id'],method=method,candidate_local=j,candidate_vocab=c['candidate_vocab'],
                intervention=j!=baseline,pdm_gain_pp=float(100*delta[0]),raw_pdm=c['measured_raw'][0],
                raw_component_harm=bool(np.any(delta[[1,2,3,4,6,7,8]] < -1e-8)))
            for key in ('route_progress_m','all_jerk_norm_mps3_rms','all_accel_lat_mps2_rms','all_actor_contact_time_s','all_outside_area_integral_m2s'):
                row['delta_'+key]=c['metrics'][key]-b['metrics'][key]
            for objective,key,absolute in [('jerk','all_jerk_norm_mps3_rms',.2),('lateral','all_accel_lat_mps2_rms',.05)]:
                gain=b['metrics'][key]-c['metrics'][key]
                row['meaningful_'+objective]=bool(gain>=absolute and gain>=.2*b['metrics'][key])
            rows.append(row)
            if token in request['oracle_tokens'] and method in request['methods'][token]:
                rows.append(dict(row,method='control64/'+method))
        for c in group.values():candidate_rows.append(dict(token=token,session_id=r['session_id'],candidate_local=c['candidate_local'],candidate_vocab=c['candidate_vocab'],teacher_rank=c['teacher_rank'],**dict(zip(['pdm_score',*old.base.METRICS],c['measured_raw'])),**c['metrics']))
        write(out/'progress.json',dict(scenes_completed=number,scenes_required=len(records),new_candidates=new,elapsed_s=time.monotonic()-start))
        print('ASTRA256 '+str(number)+'/'+str(len(records))+' new='+str(new),flush=True)
    reserve.write_gzip_csv(out/'per_scene.csv.gz',rows);reserve.write_gzip_csv(out/'per_candidate.csv.gz',candidate_rows)
    methods={}
    for method in sorted({r['method'] for r in rows}):
        g=[r for r in rows if r['method']==method];sessions=[r['session_id'] for r in g]
        methods[method]=dict(scenes=len(g),interventions=sum(r['intervention'] for r in g),raw_component_harms=sum(r['raw_component_harm'] for r in g),
            **{'meaningful_'+o:sum(r['meaningful_'+o] for r in g) for o in ('jerk','lateral')},
            **{k:cm.paired_stats([r[k] for r in g],sessions) for k in ('raw_pdm','pdm_gain_pp','delta_route_progress_m','delta_all_jerk_norm_mps3_rms','delta_all_accel_lat_mps2_rms')})
    report=dict(status='complete' if stage=='full' else 'pilot_complete',scenes=len(records),oracle_scenes=sum(r['token'] in request['oracle_tokens'] for r in records),
        new_candidates_this_invocation=new,baseline_replays_this_invocation=baseline_replays,anchors_parity_passed=len(records),
        methods=methods,holdout_processed=0,model_calls=0,note=NOTE)
    write(out/'report.json',report)
    if stage=='pilot':freeze(out/'pilot_complete.json',dict(identity_sha256=binding))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('pilot','full'))
    p.add_argument('--package',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--decisions',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    import fcntl,traceback
    from transfer import write
    with (a.out/'.execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:execute(a.package.resolve(),a.source.resolve(),a.out.resolve(),a.repo.resolve(),a.stage,a.decisions)
        except Exception as exc:
            write(a.out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
