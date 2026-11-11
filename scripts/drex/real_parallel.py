"""Scene-parallel Drex replicas; no outcome lookup until all shards commit."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback

import real_data as data
import real_pilot as pilot
from runtime import environment, read, remaining_bytes, write
from transfer import sha


def assignment(records,count):
    if not 1<=count<=8 or count>len(records):
        raise ValueError('Shard count must be 1..8 and not exceed scene count')
    return [[r['token'] for r in records[i::count]] for i in range(count)]


def freeze_plan(root,package,out,count):
    manifest,records,scenes,vocab=data.prepare(package,root,out)
    plan=dict(protocol='real_scene_parallel_v1',shards=count,assignments=assignment(records,count),
        identity_sha256=sha(out/'identity.json'),selection_sha256=sha(out/'selection.json'),
        sources={name:sha(Path(__file__).with_name(name)) for name in ('real_parallel.py','run_real_parallel.sh')},
        note='Independent full BF16 replica per GPU. Same cohort, candidates, prompts and decision rule; not model parallelism.')
    data.freeze(out/'parallel_plan.json',plan)
    return plan,manifest,records,scenes,vocab


def checked_plan(out):
    plan=read(out/'parallel_plan.json')
    if (plan['protocol']!='real_scene_parallel_v1' or sha(out/'identity.json')!=plan['identity_sha256']
            or sha(out/'selection.json')!=plan['selection_sha256']):
        raise RuntimeError('Frozen parallel plan changed')
    for name,digest in plan['sources'].items():
        if sha(Path(__file__).with_name(name))!=digest:
            raise RuntimeError('Parallel runner code changed; choose a new output name')
    return plan


def records_for_worker(plan,records,index):
    if not 0<=index<plan['shards']:
        raise ValueError('Invalid worker index')
    if assignment(records,plan['shards'])!=plan['assignments']:
        raise RuntimeError('Worker membership differs from frozen cohort')
    wanted=set(plan['assignments'][index])
    return [r for r in records if r['token'] in wanted]


def run_replica(root,package,out,manifest,records,scenes,vocab):
    import torch
    from drex_max.model import DecisionModel
    # environment() writes a diagnostic JSON. Isolate that write for each replica,
    # while checking the exact same declared requirements lock.
    envroot=out/'runtime_check'
    envroot.mkdir(parents=True,exist_ok=True)
    lock=root/'requirements.lock.txt'
    target_lock=envroot/'requirements.lock.txt'
    if target_lock.exists() and sha(target_lock)!=sha(lock):
        raise RuntimeError('Worker environment lock changed')
    if not target_lock.exists():
        shutil.copyfile(lock,target_lock)
    env=environment(envroot,True)
    data.freeze(out/'inference_environment.json',{k:env[k] for k in
        ('python','torch','cuda_build','transformers','gpu','vram_bytes')})
    write(out/'execution_environment.json',env)
    target=root/'models/drex-max-v1'
    if remaining_bytes(target,read(root/'model_plan.json')['files']):
        raise RuntimeError('Model files changed; no downloads in parallel inference')
    cfg=read(target/'decision_config.json')
    if cfg.get('format_version')!=1 or cfg.get('attention_mode','noncausal_full_attention')!='noncausal_full_attention':
        raise ValueError('Unexpected model attention protocol')
    # Slurm array task receives one allocated physical GPU, renumbered locally.
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Each replica must see exactly one Slurm-allocated GPU')
    print(f'Loading Drex replica: scenes={len(records)} local_device=cuda:0',flush=True)
    torch.cuda.reset_peak_memory_stats(); tick=time.monotonic()
    model=DecisionModel(target,device='cuda:0',max_image_pixels=data.CONFIG['max_image_pixels'])
    if model.readout.weight.dtype!=torch.bfloat16:
        raise RuntimeError('Unexpected readout precision')
    torch.cuda.synchronize(); load_seconds=time.monotonic()-tick; calls=0
    def infer(row):
        nonlocal calls
        batch=model.prepare([row],max_length=data.CONFIG['max_length'])
        tokens=int(batch.inputs['input_ids'].shape[1]); grid=batch.inputs.get('image_grid_thw')
        if 'pixel_values' not in batch.inputs or grid is None or grid.shape[0]!=2 or tokens>=data.CONFIG['max_length']:
            raise RuntimeError('Missing image inputs or context limit reached')
        del batch
        torch.cuda.synchronize(); before=time.monotonic()
        p=model.predict([row],batch_size=1,max_length=data.CONFIG['max_length'])[0]
        torch.cuda.synchronize(); calls+=1
        return dict(probabilities=p,input_tokens=tokens,seconds=time.monotonic()-before,cold_call=calls==1)
    status=pilot.infer_cohort(package,out,manifest,records,scenes,vocab,infer)
    status.update(load_seconds=load_seconds,peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    write(out/'inference_status.json',status)
    return status


def worker(root,package,out,index,replica=run_replica):
    plan=checked_plan(out)
    worker_out=out/'shards'/f'{index:02d}'
    manifest,records,scenes,vocab=data.prepare(package,root,worker_out)
    if sha(worker_out/'identity.json')!=plan['identity_sha256'] or sha(worker_out/'selection.json')!=plan['selection_sha256']:
        raise RuntimeError('Worker inputs differ from coordinator')
    subset=records_for_worker(plan,records,index)
    data.freeze(worker_out/'assignment.json',dict(parallel_plan_sha256=sha(out/'parallel_plan.json'),
                                                index=index,tokens=[r['token'] for r in subset]))
    if (worker_out/'decision_commit.json').exists():
        samples={r['token']:data.load_inputs(package,r['token'],manifest,vocab) for r in subset}
        pilot.committed_decisions(worker_out,subset,samples)
        return dict(status='choices_frozen',worker=index,scenes=len(subset),new_requests=0,outcomes_read=False)
    status=replica(root,package,worker_out,manifest,subset,scenes,vocab)
    if status['status']!='choices_frozen':
        # afterok must not allow the merge following a time-budget partial result.
        raise RuntimeError('Worker is partial; outputs preserved. Resume workers before merge.')
    return dict(status,worker=index,scenes=len(subset),outcomes_read=False)


def collect_frozen_workers(package,out,plan,manifest,records,vocab):
    sources={}; profiles=[]
    for i in range(plan['shards']):
        shard=out/'shards'/f'{i:02d}'
        subset=records_for_worker(plan,records,i)
        if sha(shard/'identity.json')!=plan['identity_sha256'] or sha(shard/'selection.json')!=plan['selection_sha256']:
            raise RuntimeError('Worker identity differs from coordinator')
        expected=dict(parallel_plan_sha256=sha(out/'parallel_plan.json'),index=i,tokens=[r['token'] for r in subset])
        if read(shard/'assignment.json')!=expected:
            raise RuntimeError('Worker assignment changed')
        samples={r['token']:data.load_inputs(package,r['token'],manifest,vocab) for r in subset}
        pilot.committed_decisions(shard,subset,samples)  # No labels; reject missing/changed/extra files.
        profiles.append(read(shard/'inference_environment.json'))
        for p in sorted((shard/'results').glob('*.json')):
            if p.name in sources:
                raise RuntimeError('Duplicate scene/request across workers')
            sources[p.name]=p
    if any(p!=profiles[0] for p in profiles[1:]):
        raise RuntimeError('Replicas used different runtime/GPU profiles')
    expected={pilot.result_name(r['token'],*t) for r in records for t in data.TASKS}
    if set(sources)!=expected:
        raise RuntimeError('Workers do not exactly cover the full fixed cohort')
    return sources,profiles[0]


def merge(root,package,out):
    plan=checked_plan(out)
    manifest,records,scenes,vocab=data.prepare(package,root,out)
    sources,profile=collect_frozen_workers(package,out,plan,manifest,records,vocab)
    # All worker commitments validated before copying, central decision freeze or labels.
    data.freeze(out/'inference_environment.json',profile)
    for name,source in sources.items():
        target=out/'results'/name
        if target.exists():
            if sha(target)!=sha(source): raise RuntimeError('Central result differs from frozen worker')
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix('.json.tmp'); shutil.copyfile(source,tmp)
        if sha(tmp)!=sha(source): raise RuntimeError('Worker result changed while copying')
        tmp.replace(target)
    def forbidden_infer(_):
        raise AssertionError('Merge must never run model inference')
    # Revalidate exact pixels/rows and create one full-cohort commit using the original protocol.
    status=pilot.infer_cohort(package,out,manifest,records,scenes,vocab,forbidden_infer,max_new=0)
    if status['status']!='choices_frozen':
        raise RuntimeError('Central cohort incomplete; no outcome lookup allowed')
    result=pilot.evaluate(package,out,manifest,records,vocab)
    result['execution']=dict(protocol=plan['protocol'],replicas=plan['shards'],
                             parallel_plan_sha256=sha(out/'parallel_plan.json'))
    write(out/'report.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('plan','worker','merge'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--shards',type=int,default=2)
    p.add_argument('--index',type=int)
    a=p.parse_args(); root=a.root.expanduser().resolve()
    name=os.environ.get('DREX_REAL_NAME',f'real_train64_gpu{a.shards}_v1')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',name): raise ValueError('Invalid output name')
    if name=='real_train64_v1': raise ValueError('Parallel protocol requires a separate output from the current single-GPU pilot')
    out=root/'runs'/name
    package=Path(os.environ.get('DREX_REAL_DATA',str(root/'data/navsim_train_v1/package'))).expanduser().resolve()
    try:
        if a.stage=='plan':
            plan,*_=freeze_plan(root,package,out,a.shards)
            result=dict(status='planned',shards=plan['shards'],scenes_per_shard=[len(v) for v in plan['assignments']],outcomes_read=False)
        elif a.stage=='worker':
            if a.index is None: raise ValueError('Worker index required')
            result=worker(root,package,out,a.index)
        else:
            result=merge(root,package,out)
        print(json.dumps(result,indent=2),flush=True)
    except Exception as exc:
        target=out/'shards'/f'{a.index:02d}' if a.stage=='worker' and a.index is not None else out
        write(target/'parallel_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__': main()
