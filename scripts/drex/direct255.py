"""Direct absolute-feature choice from top255 of the original 256 DS pool. No images."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

import real_data as base
from transfer import read, write, sha, freeze
from night import validate_probabilities

CONFIG = dict(protocol='direct255_absolute_v1', seed=20261117, precision_decimals=4,
              max_length=65536, orders=['primary','reversed'], primary='primary', horizon_s=4.)
COLUMNS = ('id', *('pred_'+m for m in base.METRICS), 'x_1s_m','y_1s_m','x_2s_m','y_2s_m',
           'x_4s_m','y_4s_m','heading_4s_deg','length_m','max_speed_mps',
           'max_abs_acceleration_mps2','max_abs_curvature_per_m')
SOURCES = ('direct255.py','direct256.py','run_direct255.sh','slurm_options.sh','real_data.py','runtime.py','transfer.py','night.py','synthetic.py','table_data.py','table_group.py')
NOTE = ('Direct top255-of-256 text-only choice. Last candidate in stable descending original DS ranking is excluded; All candidate descriptions are absolute DS predictions '
        'and nominal geometry, not measured outcomes or controller rollouts. No DS final score, '
        'imitation prior, DS rank, distinguished baseline, pairwise deltas, images, map or future actors. '
        'Reversed order is a separate sensitivity diagnostic, never an ensemble vote. '
        'Previously inspected TRAIN cohort; no training or fresh holdout. Preferences are not calibrated safety. '
        'Choices outside the old top64 need additional measured labels before performance evaluation.')


def membership(arrays):
    scores=np.asarray(arrays['selection_scores'])
    if scores.shape!=(256,) or not np.isfinite(scores).all():raise ValueError('Invalid original 256 DS scores')
    selected=np.asarray(arrays['selected_local'])
    if selected.shape!=() or selected.dtype.kind not in 'iu':raise ValueError('Invalid DS baseline index')
    baseline=int(selected)
    if baseline!=int(np.argmax(scores)):raise ValueError('Original DS baseline/score mismatch')
    ranked=np.argsort(-scores,kind='stable').tolist()
    if baseline not in ranked[:255]:raise RuntimeError('DS baseline excluded')
    return ranked[:255],ranked[-1],baseline


def inputs(arrays,vocab):
    # Geometry math is identical to direct256; scores are used ONLY to define the pool.
    from direct256 import inputs as all_inputs
    included,_,_=membership(arrays)
    sample=all_inputs(arrays,vocab)
    sample['candidates']=[c for c in sample['candidates'] if c['local'] in included]
    if len(sample['candidates'])!=255:raise ValueError('Expected 255 physical candidates')
    return sample


def payload(token, sample, ordering):
    if ordering not in CONFIG['orders'] or len(sample['candidates'])!=255: raise ValueError('Unknown request')
    candidates=sample['candidates']
    aliases={c['local']:'C'+str(i).zfill(3) for i,c in enumerate(sorted(candidates,key=lambda c:base.digest([token,c['vocab'],'alias255'])))}
    ordered=sorted(candidates,key=lambda c:base.digest([CONFIG['seed'],token,c['vocab'],'presentation']))
    if ordering=='reversed': ordered.reverse()
    rows=[[aliases[c['local']],*c['values']] for c in ordered]
    # A compact table avoids repeated JSON keys; values/precision are unchanged.
    table=' '.join(COLUMNS)+'\n'+'\n'.join(' '.join(str(v) for v in row) for row in rows)
    state=dict(task='Select one trajectory from all 255 available four-second candidates.',
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
    if len(records)!=651 or len({r['session_id'] for r in records})!=32:raise ValueError('Expected 651 TRAIN scenes / 32 sessions')
    pilot=[]; sessions=set()
    for r in records:
        if r['session_id'] not in sessions and len(pilot)<4:
            pilot.append(r['token']); sessions.add(r['session_id'])
    identity=dict(config=CONFIG,records=records,replicas=count,package=str(package),pilot_tokens=pilot,
        manifest_sha256=sha(package/'package_manifest.json'),model_plan=read(root/'model_plan.json'),
        requirements_sha256=sha(root/'requirements.lock.txt'),sources={n:sha(Path(__file__).with_name(n)) for n in SOURCES},note=NOTE)
    freeze(out/'identity.json',identity); files={}; audit={}
    for r in records:
        token=r['token']; relative='bundle/examples/'+token+'.npz'; file=package/relative
        if sha(file)!=manifest['files'][relative]['sha256']: raise RuntimeError('Package features changed')
        with np.load(file,allow_pickle=False) as arrays:
            sample=inputs(arrays,vocab); _,excluded,baseline=membership(arrays)
            audit[token]=dict(excluded_local=excluded,excluded_vocab=int(arrays['candidate_ids'][excluded]),excluded_teacher_rank=256,baseline_local=baseline)
        freeze(out/'inputs'/(token+'.json'),sample); files[token]=sha(out/'inputs'/(token+'.json'))
    freeze(out/'pool_selection.json',audit)
    freeze(out/'plan.json',dict(identity_sha256=sha(out/'identity.json'),inputs=files,pool_selection_sha256=sha(out/'pool_selection.json')))
    if pilot:
        preview,_,_=payload(pilot[0],read(out/'inputs'/(pilot[0]+'.json')),'primary')
        freeze(out/'preview_request.json',preview)
    print(json.dumps(dict(status='planned',scenes=len(records),pilot_scenes=len(pilot),
                          candidates_per_request=255,requests_full=2*len(records),outcomes_read=False)),flush=True)
    return identity


def context(root,out,count):
    identity=read(out/'identity.json'); p=read(out/'plan.json')
    if identity['config']!=CONFIG or identity['replicas']!=count or p['identity_sha256']!=sha(out/'identity.json'):
        raise RuntimeError('Frozen input/config/count changed; use original count/name')
    for n,h in identity['sources'].items():
        if sha(Path(__file__).with_name(n))!=h: raise RuntimeError('Frozen code changed: '+n)
    if identity['model_plan']!=read(root/'model_plan.json') or identity['requirements_sha256']!=sha(root/'requirements.lock.txt'):
        raise RuntimeError('Model/runtime changed')
    if sha(out/'pool_selection.json')!=p['pool_selection_sha256']:raise RuntimeError('Pool selection audit changed')
    return identity,p


def tasks(identity,phase):
    return [(r['token'],order) for r in identity['records']
            if phase!='pilot' or r['token'] in identity['pilot_tokens'] for order in CONFIG['orders']]


def checked_sample(out,plan,token):
    file=out/'inputs'/(token+'.json')
    if sha(file)!=plan['inputs'][token]: raise RuntimeError('Input changed')
    return read(file)


def checked_result(out,path,digest,ids):
    commit=read(path.with_suffix('.commit.json'))
    if commit!=dict(identity_sha256=sha(out/'identity.json'),result_sha256=sha(path)):
        raise RuntimeError('Result receipt changed')
    result=read(path)
    if (result['request_sha256']!=digest or result['local_ids']!=ids or
        path.stem!=result['token']+'-'+result['order']):raise RuntimeError('Saved request changed')
    validate_probabilities(result['probabilities'],255)
    return result


def successful_pilot(out):
    pilot=read(out/'pilot_report.json')
    if (pilot['status']!='pilot_complete' or pilot['identity_sha256']!=sha(out/'identity.json') or
        pilot['decisions_sha256']!=sha(out/'pilot_decisions.json')):
        raise RuntimeError('Successful matching pilot required')


def worker(root,out,count,phase):
    import torch
    import table_group
    from drex_max.model import DecisionModel,MAX_OPTIONS
    if MAX_OPTIONS<255:
        raise RuntimeError('Official Drex readout supports only '+str(MAX_OPTIONS)+' choices; exact255 is unsupported. No candidate was dropped or weights loaded.')
    identity,p=context(root,out,count); rank=table_group.gpu_binding(out,count,'direct255-'+phase)
    assigned=tasks(identity,phase)[rank::count]
    folder=out/'workers'/str(rank); folder.mkdir(parents=True,exist_ok=True)
    progress=folder/'status.json'
    write(progress,dict(state='checking',phase=phase,total_requests=len(assigned),new_requests=0))
    from runtime import environment,remaining_bytes
    import shutil
    envroot=folder/'runtime'; envroot.mkdir(exist_ok=True)
    shutil.copyfile(root/'requirements.lock.txt',envroot/'requirements.lock.txt')
    env=environment(envroot,True)
    freeze(folder/'environment.json',{k:env[k] for k in ('python','torch','cuda_build','transformers','gpu','vram_bytes')})
    if not assigned:
        write(progress,dict(state='complete',phase=phase,total_requests=0,new_requests=0));return
    target=root/'models/drex-max-v1'
    if remaining_bytes(target,identity['model_plan']['files']): raise RuntimeError('Model missing/size changed')
    decision=read(target/'decision_config.json')
    if decision.get('format_version')!=1 or decision.get('attention_mode','noncausal_full_attention')!='noncausal_full_attention':
        raise RuntimeError('Unexpected Drex attention/format protocol')
    if phase=='full':successful_pilot(out)
    pending=[]
    for token,order in assigned:
        sample=checked_sample(out,p,token); row,ids,digest=payload(token,sample,order)
        path=out/'results'/(token+'-'+order+'.json')
        if path.with_suffix('.commit.json').exists():checked_result(out,path,digest,ids)
        else:pending.append((token,order,row,ids,digest,path))
    write(progress,dict(state='loading_model' if pending else 'complete',phase=phase,total_requests=len(assigned),cached_requests=len(assigned)-len(pending),new_requests=0))
    if not pending:return
    model=DecisionModel(target,device='cuda:0',max_image_pixels=256*256)
    if model.readout.weight.dtype!=torch.bfloat16: raise RuntimeError('Expected BF16')
    if model.readout.out_features!=255:raise RuntimeError('Expected official 255-way readout')
    config=read(target/'config.json'); limit=config.get('text_config',config).get('max_position_embeddings')
    maximum=min(CONFIG['max_length'],int(limit)) if limit else CONFIG['max_length']
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    for number,(token,order,row,ids,digest,path) in enumerate(pending,1):
        try:
            write(progress,dict(state='running',phase=phase,total_requests=len(assigned),cached_requests=len(assigned)-len(pending),new_requests=number-1,current_token=token,order=order,elapsed_s=time.monotonic()-started))
            if time.monotonic()-started>7*3600:raise RuntimeError('Worker budget reached; submit-full again with SAME name/count')
            batch=model.prepare([row],max_length=maximum)
            tokens=int(batch.inputs['input_ids'].shape[1])
            if tokens>=maximum: raise RuntimeError('Context at limit; refusing any truncation: '+str(tokens))
            if 'pixel_values' in batch.inputs: raise RuntimeError('Unexpected image input')
            del batch
            torch.cuda.synchronize(); start=time.monotonic()
            probability=model.predict([row],batch_size=1,max_length=maximum)[0]
            torch.cuda.synchronize(); probability=[float(v) for v in probability]
            validate_probabilities(probability,255)
            write(path,dict(token=token,order=order,local_ids=ids,request_sha256=digest,
                probabilities=probability,input_tokens=tokens,seconds=time.monotonic()-start,
                model_context_limit=limit,request_context_limit=maximum,
                peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            write(path.with_suffix('.commit.json'),dict(identity_sha256=sha(out/'identity.json'),result_sha256=sha(path)))
            print('DIRECT255 '+token+' '+order+' tokens='+str(tokens),flush=True)
        except Exception as exc:
            write(folder/'failure.json',dict(token=token,order=order,error=repr(exc),silent_fallback=False))
            raise
    write(progress,dict(state='complete',phase=phase,total_requests=len(assigned),cached_requests=len(assigned)-len(pending),new_requests=len(pending),elapsed_s=time.monotonic()-started))


def report(root,out,count,phase):
    identity,p=context(root,out,count); decisions=[]
    for i,(token,order) in enumerate(tasks(identity,phase)):
        sample=checked_sample(out,p,token); _,ids,digest=payload(token,sample,order)
        path=out/'results'/(token+'-'+order+'.json'); r=checked_result(out,path,digest,ids)
        if r['request_sha256']!=digest or r['local_ids']!=ids: raise RuntimeError('Committed choice mismatch')
        validate_probabilities(r['probabilities'],255)
        j=ids[int(np.argmax(r['probabilities']))]; candidate=next(c for c in sample['candidates'] if c['local']==j)
        decisions.append(dict(token=token,order=order,candidate_local=j,candidate_vocab=candidate['vocab'],result_sha256=sha(path)))
    freeze(out/('pilot_decisions.json' if phase=='pilot' else 'decisions.json'),decisions)
    pairs={token:{r['order']:r['candidate_local'] for r in decisions if r['token']==token} for token in {r['token'] for r in decisions}}
    value=dict(status='pilot_complete' if phase=='pilot' else 'choices_complete',scenes=len(pairs),requests=len(decisions),
               identity_sha256=sha(out/'identity.json'),decisions_sha256=sha(out/('pilot_decisions.json' if phase=='pilot' else 'decisions.json')),
               ordering_flips=sum(v['primary']!=v['reversed'] for v in pairs.values()),outcomes_read=False,
               candidates_per_request=255,original_pool_size=256,performance_evaluated=False,note=NOTE)
    write(out/('pilot_report.json' if phase=='pilot' else 'report.json'),value); print(json.dumps(value,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('plan','worker','report'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--count',type=int,required=True)
    p.add_argument('--phase',choices=('pilot','full'),default='pilot');a=p.parse_args()
    import re
    if a.count not in (1,2,4,8) or not re.fullmatch('direct255_[A-Za-z0-9_-]{1,64}',a.name):raise ValueError('Invalid name/count')
    root=a.root.resolve();out=root/'runs'/a.name
    if a.stage=='plan':plan(root,out,a.count)
    elif a.stage=='worker':
        import traceback
        try:worker(root,out,a.count,a.phase)
        except Exception as exc:
            rank=os.environ.get('SLURM_PROCID','0')
            if not rank.isdigit() or int(rank)>=a.count:rank='0'
            write(out/'workers'/rank/'status.json',dict(state='failed',phase=a.phase,error=repr(exc),traceback=traceback.format_exc(),silent_fallback=False))
            raise
    else:report(root,out,a.count,a.phase)


if __name__=='__main__':main()
