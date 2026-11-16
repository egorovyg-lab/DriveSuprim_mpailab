"""Direct, absolute-feature-only choice from all 256 DS candidates. No images."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

import real_data as base
from transfer import read, write, sha, freeze
from night import validate_probabilities

CONFIG = dict(protocol='direct256_absolute_v1', seed=20261117, precision_decimals=4,
              max_length=65536, orders=['primary','reversed'], primary='primary', horizon_s=4.)
COLUMNS = ('id', *('pred_'+m for m in base.METRICS), 'x_1s_m','y_1s_m','x_2s_m','y_2s_m',
           'x_4s_m','y_4s_m','heading_4s_deg','length_m','max_speed_mps',
           'max_abs_acceleration_mps2','max_abs_curvature_per_m')
SOURCES = ('direct256.py','run_direct256.sh','real_data.py','runtime.py','transfer.py','night.py','synthetic.py','table_data.py','table_group.py')
NOTE = ('Direct 256-way text-only choice. All candidate descriptions are absolute DS predictions '
        'and nominal geometry, not measured outcomes or controller rollouts. No DS final score, '
        'imitation prior, DS rank, distinguished baseline, pairwise deltas, images, map or future actors. '
        'Reversed order is a separate sensitivity diagnostic, never an ensemble vote. '
        'Previously inspected TRAIN cohort; no training or fresh holdout. Preferences are not calibrated safety. '
        'Choices outside the old top64 need additional measured labels before performance evaluation.')


def inputs(arrays, vocab):
    ids=np.asarray(arrays['candidate_ids']); ego=np.asarray(arrays['ego_status'],float)
    if ids.shape!=(256,) or ids.dtype.kind not in 'iu' or len(set(ids.tolist()))!=256 or ids.min()<0 or ids.max()>=len(vocab):
        raise ValueError('Expected 256 unique physical candidate IDs')
    if ego.ndim!=1 or len(ego)<8 or not np.isfinite(ego[:8]).all(): raise ValueError('Invalid current ego status')
    heads=np.array([np.asarray(arrays['head_'+m],float) for m in base.METRICS]).T
    if heads.shape!=(256,8) or not np.isfinite(heads).all(): raise ValueError('Invalid aligned DS heads')
    probabilities=np.exp(-np.logaddexp(0.,-heads)); speed0=float(np.linalg.norm(ego[4:6]))
    candidates=[]
    for j,v in enumerate(ids):
        poses=np.asarray(vocab[v],float)
        if poses.shape!=(40,3) or not np.isfinite(poses).all(): raise ValueError('Expected finite 40x3 nominal path')
        distance=np.linalg.norm(np.diff(np.vstack((np.zeros((1,2)),poses[:,:2])),axis=0),axis=1)
        speeds=distance/.1; acceleration=np.diff(np.r_[speed0,speeds])/.1
        dh=np.diff(np.unwrap(np.r_[0.,poses[:,2]])); moving=distance>=.05
        curvature=float(np.abs(dh[moving]/distance[moving]).max()) if moving.any() else 0.
        values=[*probabilities[j],*poses[9,:2],*poses[19,:2],*poses[39,:2],
                np.degrees(np.arctan2(np.sin(poses[-1,2]),np.cos(poses[-1,2]))),
                distance.sum(),speeds.max(),np.abs(acceleration).max(),curvature]
        candidates.append(dict(local=j,vocab=int(v),values=[round(float(x),4) for x in values]))
    return dict(ego_velocity=ego[4:6].tolist(),ego_acceleration=ego[6:8].tolist(),candidates=candidates)


def payload(token, sample, ordering):
    if ordering not in CONFIG['orders'] or len(sample['candidates'])!=256: raise ValueError('Unknown request')
    candidates=sample['candidates']
    aliases={c['local']:'C'+str(i).zfill(3) for i,c in enumerate(sorted(candidates,key=lambda c:base.digest([token,c['vocab'],'alias256'])))}
    ordered=sorted(candidates,key=lambda c:base.digest([CONFIG['seed'],token,c['vocab'],'presentation']))
    if ordering=='reversed': ordered.reverse()
    rows=[[aliases[c['local']],*c['values']] for c in ordered]
    # A compact table avoids repeated JSON keys; values/precision are unchanged.
    table=' '.join(COLUMNS)+'\n'+'\n'.join(' '.join(str(v) for v in row) for row in rows)
    state=dict(task='Select one trajectory from all 256 available four-second candidates.',
        coordinates='Current rear axle origin. x forward, y left, heading counterclockwise; SI units.',
        current_velocity_xy_mps=sample['ego_velocity'],current_acceleration_xy_mps2=sample['ego_acceleration'],
        predictions='pred_* are absolute sigmoid DS head estimates, not measured outcomes or calibrated risk.',
        geometry='NOMINAL poses at 1,2,4 seconds. Acceleration includes transition from current speed. '
                 'Curvature uses steps >=0.05 m. No controller simulation, route command or actor prediction.',
        candidate_table=table)
    row=dict(state=state,question=dict(type='choice',criteria={r[0]:None for r in rows},
        instructions='Choose the best available trajectory. Prioritize avoiding collisions, road departure, '
            'wrong-way driving and traffic-light violations, then useful progress, lane keeping and comfort. '
            'All estimates are uncertain; unknown surroundings do not establish safety. Return one supplied ID.'))
    local=[c['local'] for c in ordered]
    return row,local,base.digest(row)


def plan(root, out, count):
    import table_data
    package=root/'data/navsim_train_v1/package'
    manifest,index,_,vocab=base.package_metadata(package)
    records=table_data.cohort(index,'all')
    pilot=[]; sessions=set()
    for r in records:
        if r['session_id'] not in sessions and len(pilot)<4:
            pilot.append(r['token']); sessions.add(r['session_id'])
    identity=dict(config=CONFIG,records=records,replicas=count,package=str(package),pilot_tokens=pilot,
        manifest_sha256=sha(package/'package_manifest.json'),model_plan=read(root/'model_plan.json'),
        requirements_sha256=sha(root/'requirements.lock.txt'),sources={n:sha(Path(__file__).with_name(n)) for n in SOURCES},note=NOTE)
    freeze(out/'identity.json',identity); files={}
    for r in records:
        token=r['token']; relative='bundle/examples/'+token+'.npz'; file=package/relative
        if sha(file)!=manifest['files'][relative]['sha256']: raise RuntimeError('Package features changed')
        with np.load(file,allow_pickle=False) as arrays: sample=inputs(arrays,vocab)
        freeze(out/'inputs'/(token+'.json'),sample); files[token]=sha(out/'inputs'/(token+'.json'))
    freeze(out/'plan.json',dict(identity_sha256=sha(out/'identity.json'),inputs=files))
    if pilot:
        preview,_,_=payload(pilot[0],read(out/'inputs'/(pilot[0]+'.json')),'primary')
        freeze(out/'preview_request.json',preview)
    print(json.dumps(dict(status='planned',scenes=len(records),pilot_scenes=len(pilot),
                          candidates_per_request=256,requests_full=2*len(records),outcomes_read=False)),flush=True)
    return identity


def context(root,out,count):
    identity=read(out/'identity.json'); p=read(out/'plan.json')
    if identity['config']!=CONFIG or identity['replicas']!=count or p['identity_sha256']!=sha(out/'identity.json'):
        raise RuntimeError('Frozen input/config/count changed; use original count/name')
    for n,h in identity['sources'].items():
        if sha(Path(__file__).with_name(n))!=h: raise RuntimeError('Frozen code changed: '+n)
    if identity['model_plan']!=read(root/'model_plan.json') or identity['requirements_sha256']!=sha(root/'requirements.lock.txt'):
        raise RuntimeError('Model/runtime changed')
    return identity,p


def tasks(identity,phase):
    return [(r['token'],order) for r in identity['records']
            if phase!='pilot' or r['token'] in identity['pilot_tokens'] for order in CONFIG['orders']]


def checked_sample(out,plan,token):
    file=out/'inputs'/(token+'.json')
    if sha(file)!=plan['inputs'][token]: raise RuntimeError('Input changed')
    return read(file)


def worker(root,out,count,phase):
    import torch
    import table_group
    from drex_max.model import DecisionModel
    identity,p=context(root,out,count); rank=table_group.gpu_binding(out,count,'direct256-'+phase)
    assigned=tasks(identity,phase)[rank::count]
    folder=out/'workers'/str(rank); folder.mkdir(parents=True,exist_ok=True)
    from runtime import environment,remaining_bytes
    import shutil
    envroot=folder/'runtime'; envroot.mkdir(exist_ok=True)
    shutil.copyfile(root/'requirements.lock.txt',envroot/'requirements.lock.txt')
    env=environment(envroot,True)
    freeze(folder/'environment.json',{k:env[k] for k in ('python','torch','cuda_build','transformers','gpu','vram_bytes')})
    if not assigned:return
    target=root/'models/drex-max-v1'
    if remaining_bytes(target,identity['model_plan']['files']): raise RuntimeError('Model missing/size changed')
    decision=read(target/'decision_config.json')
    if decision.get('format_version')!=1 or decision.get('attention_mode','noncausal_full_attention')!='noncausal_full_attention':
        raise RuntimeError('Unexpected Drex attention/format protocol')
    if phase=='full' and read(out/'pilot_report.json')['status']!='pilot_complete': raise RuntimeError('Successful pilot required')
    model=DecisionModel(target,device='cuda:0',max_image_pixels=256*256)
    if model.readout.weight.dtype!=torch.bfloat16: raise RuntimeError('Expected BF16')
    config=read(target/'config.json'); limit=config.get('text_config',config).get('max_position_embeddings')
    maximum=min(CONFIG['max_length'],int(limit)) if limit else CONFIG['max_length']
    torch.cuda.reset_peak_memory_stats()
    for token,order in assigned:
        sample=checked_sample(out,p,token); row,ids,digest=payload(token,sample,order)
        path=folder/'results'/(token+'-'+order+'.json')
        if path.exists():
            result=read(path)
            if result['request_sha256']!=digest or result['local_ids']!=ids: raise RuntimeError('Saved request changed')
            validate_probabilities(result['probabilities'],256); continue
        try:
            batch=model.prepare([row],max_length=maximum)
            tokens=int(batch.inputs['input_ids'].shape[1])
            if tokens>=maximum: raise RuntimeError('Context at limit; refusing any truncation: '+str(tokens))
            if 'pixel_values' in batch.inputs: raise RuntimeError('Unexpected image input')
            del batch
            torch.cuda.synchronize(); start=time.monotonic()
            probability=model.predict([row],batch_size=1,max_length=maximum)[0]
            torch.cuda.synchronize(); probability=[float(v) for v in probability]
            validate_probabilities(probability,256)
            write(path,dict(token=token,order=order,local_ids=ids,request_sha256=digest,
                probabilities=probability,input_tokens=tokens,seconds=time.monotonic()-start,
                model_context_limit=limit,request_context_limit=maximum,
                peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            print('DIRECT256 '+token+' '+order+' tokens='+str(tokens),flush=True)
        except Exception as exc:
            write(folder/'failure.json',dict(token=token,order=order,error=repr(exc),silent_fallback=False))
            raise


def report(root,out,count,phase):
    identity,p=context(root,out,count); decisions=[]
    for i,(token,order) in enumerate(tasks(identity,phase)):
        sample=checked_sample(out,p,token); _,ids,digest=payload(token,sample,order)
        path=out/'workers'/str(i%count)/'results'/(token+'-'+order+'.json'); r=read(path)
        if r['request_sha256']!=digest or r['local_ids']!=ids: raise RuntimeError('Committed choice mismatch')
        validate_probabilities(r['probabilities'],256)
        j=ids[int(np.argmax(r['probabilities']))]; candidate=next(c for c in sample['candidates'] if c['local']==j)
        decisions.append(dict(token=token,order=order,candidate_local=j,candidate_vocab=candidate['vocab'],result_sha256=sha(path)))
    freeze(out/('pilot_decisions.json' if phase=='pilot' else 'decisions.json'),decisions)
    pairs={token:{r['order']:r['candidate_local'] for r in decisions if r['token']==token} for token in {r['token'] for r in decisions}}
    value=dict(status='pilot_complete' if phase=='pilot' else 'choices_complete',scenes=len(pairs),requests=len(decisions),
               ordering_flips=sum(v['primary']!=v['reversed'] for v in pairs.values()),outcomes_read=False,
               performance_evaluated=False,note=NOTE)
    write(out/('pilot_report.json' if phase=='pilot' else 'report.json'),value); print(json.dumps(value,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('plan','worker','report'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--count',type=int,required=True)
    p.add_argument('--phase',choices=('pilot','full'),default='pilot');a=p.parse_args()
    import re
    if a.count not in (1,2,4,8) or not re.fullmatch('direct256_[A-Za-z0-9_-]{1,64}',a.name):raise ValueError('Invalid name/count')
    root=a.root.resolve();out=root/'runs'/a.name
    if a.stage=='plan':plan(root,out,a.count)
    elif a.stage=='worker':worker(root,out,a.count,a.phase)
    else:report(root,out,a.count,a.phase)


if __name__=='__main__':main()
