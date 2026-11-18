"""Astra-only: fixed 256 controls, balanced-64 oracle, and committed Drex choices."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

CONFIG=dict(version=1,seed=20261117,oracle_scenes=64,oracle_sessions=32,oracle_per_session=2,
            progress_budget_m=1.,progress_budget_fraction=.05,score_match_atol=1e-4)
# Exact LF source exported at cd0041e. Reuse is deliberately limited to that
# completed experiment, whose scorer, package and per-candidate receipts bind it.
LEGACY_CODE_SHA256='6baeee9c5b8ed570dafa9aa2a9a5722b585a9449a1cd021cbcebe67858403aeb'
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


def plan(package,source,out,decisions=None,oracle_all=False,reuse=None):
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
    pilot=oracle[:4]
    if oracle_all:
        # Exercise both the already measured tail and genuinely new tail labels.
        pilot=oracle[:2]
        used={r['session_id'] for r in records if r['token'] in pilot}
        for r in sorted(records,key=lambda r:key(r['token'])):
            if r['token'] not in oracle and r['session_id'] not in used and len(pilot)<4:
                pilot.append(r['token']);used.add(r['session_id'])
        if len(pilot)!=4:raise ValueError('Four distinct pilot sessions with two new oracle scenes required')
        oracle=[r['token'] for r in records]
    if oracle_all and decisions:raise ValueError('Choose oracle-all OR committed model decisions')
    methods={}; candidates={}; formulas={}
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
    config=dict(CONFIG,version=2,oracle_scenes=651,oracle_per_session=None) if oracle_all else CONFIG
    request=dict(config=config,records=records,oracle_tokens=[] if decisions else oracle,pilot_tokens=pilot,methods=methods,candidates=candidates,
        formulas=formulas,package=str(package),manifest_sha256=sha(package/'package_manifest.json'),source=str(source),
        source_identity_sha256=sha(source/'artifacts/identity.json'),source_request_sha256=sha(source/'audit_request.json'),
        source_report_sha256=sha(source/'artifacts/report.json'),drex_sha256=sha(decisions) if decisions else None,note=NOTE)
    if reuse:
        if reuse.resolve()==out.resolve():raise ValueError('Reuse source must differ from output')
        request['reuse']=dict(path=str(reuse.resolve()),files={n:sha(reuse/n) for n in ('request.json','identity.json','report.json')})
        reuse_context(request)  # Reject incomplete/unrelated runs before freezing anything.
    freeze(out/'request.json',request);return request


def reuse_context(request):
    from transfer import read,sha
    spec=request.get('reuse')
    if not spec:return None
    root=Path(spec['path'])
    for name,digest in spec['files'].items():
        if sha(root/name)!=digest:raise RuntimeError('Reuse metadata changed: '+name)
    previous=read(root/'request.json'); identity=read(root/'identity.json'); report=read(root/'report.json')
    if identity['request_sha256']!=sha(root/'request.json') or identity['code_sha256']!=LEGACY_CODE_SHA256:
        raise RuntimeError('Reuse requires the original cd0041e implementation and request')
    if previous['config']!=CONFIG or previous['drex_sha256'] is not None:
        raise RuntimeError('Reuse requires the original balanced-64 controls experiment')
    if (report['status'],report['scenes'],report['oracle_scenes'],report['anchors_parity_passed'])!=('complete',651,64,651):
        raise RuntimeError('Reuse experiment is incomplete')
    records=previous['records']
    if len(records)!=651 or len({r['session_id'] for r in records})!=32 or any(r['role']!='train' for r in records):
        raise RuntimeError('Reuse cohort must be the original 651 TRAIN scenes / 32 sessions')
    for key in ('records','methods','candidates','formulas','manifest_sha256','source',
                'source_identity_sha256','source_request_sha256','source_report_sha256'):
        if previous[key]!=request[key]:raise RuntimeError('Reuse binding differs: '+key)
    if len(set(previous['oracle_tokens']))!=64 or not set(previous['oracle_tokens'])<=set(request['oracle_tokens']):
        raise RuntimeError('Reuse oracle membership differs')
    return root,sha(root/'identity.json')


def check_candidate(value,token,j,info,metric_keys):
    if (value['token'],value['candidate_local'],value['candidate_vocab'],value['teacher_rank'])!=(token,j,info['vocab'][j],info['ranked'].index(j)+1):
        raise RuntimeError('Checkpoint physical candidate/rank mismatch')
    raw=np.asarray(value['measured_raw'],float)
    if raw.shape!=(9,) or not np.isfinite(raw).all() or np.any(raw<0) or np.any(raw>1):
        raise RuntimeError('Invalid checkpoint raw metrics')
    if set(value['metrics'])!=set(metric_keys) or any(v is not None and not np.isfinite(v) for v in value['metrics'].values()):
        raise RuntimeError('Invalid checkpoint continuous metrics')


def reused_candidate(context,token,j,info,metric_keys):
    from transfer import read,sha
    if not context:return None
    root,binding=context; path=root/'scenes'/token/(str(j)+'.json'); commit=path.with_name(str(j)+'.commit.json')
    if not commit.exists():return None  # An uncommitted partial JSON is never reused.
    if read(commit)!=dict(identity_sha256=binding,sha256=sha(path)):
        raise RuntimeError('Reuse candidate receipt changed: '+str(path))
    value=read(path);check_candidate(value,token,j,info,metric_keys)
    return value


def pool_expansion(rows):
    import continuous_metrics as cm
    by_method={}
    for row in rows:by_method.setdefault(row['method'],{})[row['token']]=row
    result={}
    for small,large in ((16,64),(64,256),(16,256)):
        lo=by_method.get('oracle_pdm_top'+str(small),{});hi=by_method.get('oracle_pdm_top'+str(large),{})
        if not lo and not hi:continue
        if set(lo)!=set(hi):raise RuntimeError('Oracle comparison cohorts differ')
        tokens=sorted(lo);delta=[100*(hi[t]['raw_pdm']-lo[t]['raw_pdm']) for t in tokens]
        if any(v < -1e-8 for v in delta):raise RuntimeError('Nested PDM oracle reserve decreased')
        sessions=[lo[t]['session_id'] for t in tokens]
        positive=sorted((v for v in delta if v>1e-8),reverse=True)
        result[str(large)+'_vs_'+str(small)]=dict(scenes=len(tokens),gain_pp=cm.paired_stats(delta,sessions),
            improved_scenes=sum(v>1e-8 for v in delta),improved_sessions=len({s for s,v in zip(sessions,delta) if v>1e-8}),
            top2_fraction_of_positive_micro_gain=sum(positive[:2])/sum(positive) if positive else None)
    return result


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


def execute(package,source,out,repo,stage,decisions=None,oracle_all=False,reuse=None):
    import continuous_remote as old
    import continuous_metrics as cm
    import pool_remote as pool
    import pool_reserve as reserve
    from transfer import read,write,freeze,sha
    import sys
    sys.path.insert(1,str(repo))
    request=plan(package,source,out,decisions,oracle_all,reuse)  # No outcomes loaded until the choices commit exists.
    reuse_bound=reuse_context(request)
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
    rows=[]; new=0; reused=0; baseline_replays=0; start=time.monotonic(); measured_seconds=0.; completed_candidates=0
    candidate_csv=out/'per_candidate.csv.gz.tmp'
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
            generated=False
            path=folder/(str(j)+'.json'); commit=folder/(str(j)+'.commit.json')
            if commit.exists():
                cp=read(commit)
                if cp!=dict(identity_sha256=binding,sha256=sha(path)):raise RuntimeError('Candidate checkpoint changed')
                value=read(path)
            else:
                generated=True
                value=reused_candidate(reuse_bound,token,j,info,known[baseline]['metrics'])
                if value is None:
                    candidate=dict(token=token,candidate_local=j,candidate_vocab=info['vocab'][j],teacher_rank=info['ranked'].index(j)+1)
                    tick=time.monotonic();value=evaluate(candidate,caches[token],template,scenes,vocab,profile)
                    measured_seconds+=time.monotonic()-tick;new+=1
                else:reused+=1
                check_candidate(value,token,j,info,known[baseline]['metrics'])
                write(path,value);write(commit,dict(identity_sha256=binding,sha256=sha(path)))
            check_candidate(value,token,j,info,known[baseline]['metrics'])
            group[j]=value
            if generated and (new+reused)%25==0:
                write(out/'progress.json',dict(scenes_completed=number-1,scenes_required=len(records),current_token=token,
                    current_scene_candidates_completed=len(group),current_scene_candidates_required=len(wanted),new_candidates=new,
                    reused_candidates=reused,elapsed_s=time.monotonic()-start,seconds_per_new_candidate=measured_seconds/new if new else None))
                print('ASTRA256 candidate '+token+' '+str(len(group))+'/'+str(len(wanted))+' new='+str(new)+' reused='+str(reused),flush=True)
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
            if len(request['oracle_tokens'])==64 and token in request['oracle_tokens'] and method in request['methods'][token]:
                rows.append(dict(row,method='control64/'+method))
        # Bound memory to one scene, rather than retaining 166,656 metric dicts.
        candidate_rows=[dict(token=token,session_id=r['session_id'],candidate_local=c['candidate_local'],candidate_vocab=c['candidate_vocab'],teacher_rank=c['teacher_rank'],**dict(zip(['pdm_score',*old.base.METRICS],c['measured_raw'])),**c['metrics']) for c in group.values()]
        with gzip.open(candidate_csv,'wt' if number==1 else 'at',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(candidate_rows[0]))
            if number==1:writer.writeheader()
            writer.writerows(candidate_rows)
        completed_candidates+=len(group)
        write(out/'progress.json',dict(scenes_completed=number,scenes_required=len(records),new_candidates=new,reused_candidates=reused,
            completed_candidates=completed_candidates,elapsed_s=time.monotonic()-start,seconds_per_new_candidate=measured_seconds/new if new else None))
        print('ASTRA256 '+str(number)+'/'+str(len(records))+' new='+str(new)+' reused='+str(reused),flush=True)
    reserve.write_gzip_csv(out/'per_scene.csv.gz',rows);candidate_csv.replace(out/'per_candidate.csv.gz')
    methods={}
    for method in sorted({r['method'] for r in rows}):
        g=[r for r in rows if r['method']==method];sessions=[r['session_id'] for r in g]
        methods[method]=dict(scenes=len(g),interventions=sum(r['intervention'] for r in g),raw_component_harms=sum(r['raw_component_harm'] for r in g),
            **{'meaningful_'+o:sum(r['meaningful_'+o] for r in g) for o in ('jerk','lateral')},
            **{k:cm.paired_stats([r[k] for r in g],sessions) for k in ('raw_pdm','pdm_gain_pp','delta_route_progress_m','delta_all_jerk_norm_mps3_rms','delta_all_accel_lat_mps2_rms')})
    report=dict(status='complete' if stage=='full' else 'pilot_complete',scenes=len(records),oracle_scenes=sum(r['token'] in request['oracle_tokens'] for r in records),
        sessions=len({r['session_id'] for r in records}),completed_candidates=completed_candidates,
        reused_candidates_this_invocation=reused,new_candidates_this_invocation=new,baseline_replays_this_invocation=baseline_replays,anchors_parity_passed=len(records),
        methods=methods,oracle_pool_expansion=pool_expansion(rows),holdout_processed=0,model_calls=0,note=NOTE)
    write(out/'report.json',report)
    if stage=='pilot':freeze(out/'pilot_complete.json',dict(identity_sha256=binding))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('pilot','full'))
    p.add_argument('--package',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--decisions',type=Path)
    p.add_argument('--oracle-all',action='store_true');p.add_argument('--reuse',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    import fcntl,traceback
    from transfer import write
    with (a.out/'.execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:execute(a.package.resolve(),a.source.resolve(),a.out.resolve(),a.repo.resolve(),a.stage,a.decisions,a.oracle_all,a.reuse)
        except Exception as exc:
            write(a.out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
