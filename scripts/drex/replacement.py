"""Single-allocation pairwise replacement study on the entire frozen train cohort."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback

import numpy as np

import real_data as base
import table_data as original
import table_pilot as previous
import table_factorial as factorial
import replacement_data as data
from night import load_result, save_result, validate_probabilities
from runtime import read, write, environment, remaining_bytes
from transfer import freeze, sha

SOURCES = tuple(dict.fromkeys(('replacement.py','replacement_data.py','run_replacement.sh',
    'table_group.py','table_factorial.py','table_factorial_data.py','audit_table_cases.py',*previous.SOURCES)))


def prepare(root, source, out, count):
    identity, plan = previous.context(source); report = read(source/'report.json')
    if report['status'] != 'complete' or report['missing_pool_labels'] or identity['cohort'] != 'all':
        raise RuntimeError('Completed full-cohort original table run required')
    if count not in (1,2,4,8) or count > len(identity['records']): raise ValueError('Invalid GPU count')
    if read(root/'model_plan.json') != identity['model_plan'] or sha(root/'requirements.lock.txt') != identity['environment_lock_sha256']:
        raise RuntimeError('Original model/runtime required')
    freeze(out/'identity.json', dict(config=data.CONFIG, replicas=count, records=identity['records'],
        source=str(source), source_files={n:sha(source/n) for n in factorial.SOURCE_FILES},
        sources={n:sha(Path(__file__).with_name(n)) for n in SOURCES},
        model_plan=identity['model_plan'], environment_lock_sha256=identity['environment_lock_sha256'], note=data.NOTE))
    for r in identity['records']:
        sample, photo = previous.scene(source, plan, r['token']); data.challengers(sample); data.views(photo)
    freeze(out/'plan.json', dict(identity_sha256=sha(out/'identity.json'),
        assignments=[[r['token'] for r in identity['records'][i::count]] for i in range(count)]))
    return dict(status='planned', scenes=len(identity['records']), sessions=len({r['session_id'] for r in identity['records']}),
        pairs=len(identity['records'])*63, requests=len(identity['records'])*378, replicas=count,
        primary=data.CONFIG['primary'], labels_loaded_for_planning=False, note=data.NOTE)


def context(out):
    identity, plan = read(out/'identity.json'), read(out/'plan.json')
    if identity['config'] != data.CONFIG or sha(out/'identity.json') != plan['identity_sha256']:
        raise RuntimeError('Frozen replacement identity changed')
    for name,digest in identity['sources'].items():
        if sha(Path(__file__).with_name(name)) != digest: raise RuntimeError('Frozen code changed: '+name)
    source = Path(identity['source'])
    for name,digest in identity['source_files'].items():
        if sha(source/name) != digest: raise RuntimeError('Original source changed: '+name)
    old, source_plan = previous.context(source)
    expected = [[r['token'] for r in identity['records'][i::identity['replicas']]] for i in range(identity['replicas'])]
    if old['records'] != identity['records'] or expected != plan['assignments']:
        raise RuntimeError('Frozen cohort/assignment changed')
    return identity, plan, source, source_plan


def inputs(source, source_plan, records):
    samples = {}
    for r in records:
        token = r['token']; path = source/'inputs'/(token+'.json'); entry = source_plan['entries'][token]
        if sha(path) != entry['inputs_sha256']: raise RuntimeError('Frozen numeric input changed')
        sample = read(path)
        if sample['candidates'] != entry['candidates'] or sample['baseline_local'] != entry['baseline_local']:
            raise RuntimeError('Original candidate alignment changed')
        data.challengers(sample); samples[token] = sample
    return samples


def validate(value, task, ids):
    token, local, mode, order = task
    if (value['token'],value['candidate_local'],value['mode'],value['ordering'],value['pair_local_ids'],value['k']) != (token,local,mode,order,ids,2):
        raise RuntimeError('Physical pair/task alignment changed')
    validate_probabilities(value['probabilities'],2)


def infer_tasks(out, source, source_plan, records, worker, infer, max_new=None, budget_seconds=7*3600):
    samples = inputs(source,source_plan,records); tasks = data.tasks(records,samples)
    expected = {data.name(*t) for t in tasks}
    if {p.name for p in (worker/'results').glob('*.json')} - expected: raise RuntimeError('Unexpected results')
    start = time.monotonic(); completed = new = 0; last = None
    def progress(status):
        value = dict(status=status, completed_requests=completed, expected_requests=len(tasks),
            new_requests=new, seconds=time.monotonic()-start, outcomes_read=False,
            last_progress_unix_seconds=time.time())
        write(worker/'inference_status.json',value); return value
    progress('running')
    for task in tasks:
        token,local,mode,order = task
        if token != last:
            sample,photo = previous.scene(source,source_plan,token)
            prepared = data.views(photo); last = token
        path = worker/'results'/data.name(*task)
        if not path.exists() and ((max_new is not None and new >= max_new) or time.monotonic()-start >= budget_seconds): break
        row,ids,digest = data.payload(token,sample,local,mode,order,prepared)
        if path.exists():
            value = load_result(path,digest); validate(value,task,ids)
        else:
            value = dict(status='ok',token=token,candidate_local=local,mode=mode,ordering=order,
                pair_local_ids=ids,k=2,request_sha256=digest)
            try:
                value.update(infer(row)); validate(value,task,ids)
            except Exception as exc:
                value.update(status='technical_failure',error=repr(exc),traceback=traceback.format_exc())
                save_result(path,value); progress('failed'); raise
            save_result(path,value); new += 1
        completed += 1
        if completed % 25 == 0:
            progress('running'); print(f'REPLACEMENT {completed}/{len(tasks)} {token}',flush=True)
    if completed == len(tasks):
        freeze(worker/'commit.json',dict(plan_sha256=sha(out/'plan.json'),
            environment_sha256=sha(worker/'environment.json'),
            results={name:sha(worker/'results'/name) for name in sorted(expected)}))
    return progress('choices_frozen' if completed == len(tasks) else 'partial')


def run_gpu(root,out,source,source_plan,records,worker,pilot):
    import torch
    from drex_max.model import DecisionModel
    envroot = worker/'runtime'; envroot.mkdir(parents=True,exist_ok=True)
    lock = envroot/'requirements.lock.txt'
    if lock.exists() and sha(lock) != sha(root/'requirements.lock.txt'): raise RuntimeError('Worker lock changed')
    shutil.copyfile(root/'requirements.lock.txt',lock)
    env = environment(envroot,True)
    freeze(worker/'environment.json',{k:env[k] for k in ('python','torch','cuda_build','transformers','gpu','vram_bytes')})
    if torch.cuda.device_count() != 1: raise RuntimeError('One GPU per rank required')
    target = root/'models/drex-max-v1'
    if remaining_bytes(target,read(root/'model_plan.json')['files']): raise RuntimeError('Model files missing/changed sizes')
    config = read(target/'decision_config.json')
    if config.get('format_version') != 1 or config.get('attention_mode','noncausal_full_attention') != 'noncausal_full_attention':
        raise RuntimeError('Wrong model attention format')
    torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
    model = DecisionModel(target,device='cuda:0',max_image_pixels=data.CONFIG['max_image_pixels'])
    if model.readout.weight.dtype != torch.bfloat16: raise RuntimeError('Expected BF16')
    torch.cuda.synchronize(); load_seconds = time.monotonic()-start; calls = 0
    def infer(row):
        nonlocal calls
        batch = model.prepare([row],max_length=data.CONFIG['max_length']); n = len(row.get('images',[]))
        tokens = int(batch.inputs['input_ids'].shape[1]); grid = batch.inputs.get('image_grid_thw')
        if tokens >= data.CONFIG['max_length'] or (0 if grid is None else int(grid.shape[0])) != n or ('pixel_values' in batch.inputs) != (n>0):
            raise RuntimeError('Truncation or image condition mismatch')
        del batch
        torch.cuda.synchronize(); start = time.monotonic()
        p = model.predict([row],batch_size=1,max_length=data.CONFIG['max_length'])[0]
        torch.cuda.synchronize(); calls += 1
        return dict(probabilities=[float(v) for v in p],seconds=time.monotonic()-start,input_tokens=tokens,
            has_image_pixels=n>0,cold_call=calls==1)
    result = infer_tasks(out,source,source_plan,records,worker,infer,max_new=18 if pilot else None)
    result.update(load_seconds=load_seconds,peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    write(worker/'inference_status.json',result); return result


def collect(out):
    identity,plan,source,source_plan = context(out)
    pairs = []; decisions = []; profiles = []; workers = {}; performance = defaultdict(list)
    for i,tokens in enumerate(plan['assignments']):
        w = out/'shards'/f'{i:02d}'; commit = read(w/'commit.json')
        records = [r for r in identity['records'] if r['token'] in set(tokens)]
        samples = inputs(source,source_plan,records); expected = {data.name(*t) for t in data.tasks(records,samples)}
        if (commit['plan_sha256'] != sha(out/'plan.json') or set(commit['results']) != expected
                or {p.name for p in (w/'results').glob('*.json')} != expected
                or sha(w/'environment.json') != commit['environment_sha256']):
            raise RuntimeError('Incomplete/changed worker; labels remain unread')
        profiles.append(read(w/'environment.json')); workers[str(i)] = sha(w/'commit.json')
        for r in records:
            token = r['token']; sample,photo = previous.scene(source,source_plan,token); prepared = data.views(photo)
            methods = {}
            for mode in data.MODES:
                group = []
                for c in data.challengers(sample):
                    preferences = []
                    for order in data.ORDERS:
                        task = (token,c['local'],mode,order); path = w/'results'/data.name(*task)
                        if sha(path) != commit['results'][path.name]: raise RuntimeError('Committed result changed')
                        _,ids,digest = data.payload(token,sample,c['local'],mode,order,prepared)
                        value = load_result(path,digest); validate(value,task,ids)
                        preferences.append(data.preference(value['probabilities'],ids,c['local']))
                        performance[mode].append({k:value.get(k) for k in ('seconds','input_tokens','cold_call')})
                    pair = dict(token=token,session_id=r['session_id'],mode=mode,candidate_local=c['local'],candidate_vocab=c['vocab'],
                        teacher_rank=sample['candidates'].index(c)+1,
                        reference_first_preference=preferences[0],challenger_first_preference=preferences[1],
                        mean_preference=float(np.mean(preferences)),minimum_preference=min(preferences),
                        order_choice_flip=(preferences[0]>.5)!=(preferences[1]>.5),
                        exact_tie=any(p==.5 for p in preferences))
                    pairs.append(pair); group.append(pair)
                for policy in data.POLICIES: methods[mode+'/'+policy] = data.select(group,sample['baseline_local'],policy)
            decisions.append(dict(metadata=r,methods=methods))
    if any(p!=profiles[0] for p in profiles[1:]): raise RuntimeError('Different rank environments')
    pairs.sort(key=lambda r:(r['token'],r['mode'],r['teacher_rank']))
    decisions.sort(key=lambda d:d['metadata']['token'])
    freeze(out/'pair_decisions.json',pairs); freeze(out/'decisions.json',decisions)
    freeze(out/'decision_commit.json',dict(plan_sha256=sha(out/'plan.json'),decisions_sha256=sha(out/'decisions.json'),
        pair_decisions_sha256=sha(out/'pair_decisions.json'),workers=workers))
    return identity,pairs,decisions,performance


def write_csv(path,rows):
    with path.with_suffix('.csv.tmp').open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    path.with_suffix('.csv.tmp').replace(path)


def report(out):
    identity,pairs,decisions,performance = collect(out)  # Full cohort commits BEFORE measured label lookup.
    import audit_table_cases as audit
    _,_,_,old_decisions,_,labels,samples,_ = audit.validate_source(Path(identity['source']))
    old = {d['metadata']['token']:d for d in old_decisions if d['k']==64}
    safety = [1+base.METRICS.index(m) for m in base.SAFETY]; lane = 1+base.METRICS.index('lane_keeping')
    rows = []
    for d in decisions:
        token = d['metadata']['token']; sample = samples[token]; baseline = sample['baseline_local']; group = labels[token]
        methods = dict(d['methods'],baseline=baseline,old_table=old[token]['methods']['table/mean3'],old_camera=old[token]['methods']['table_camera/mean3'])
        maximum = max(group[c['local']][0] for c in sample['candidates'])
        methods['oracle_top64'] = baseline if group[baseline][0]==maximum else next(c['local'] for c in sample['candidates'] if group[c['local']][0]==maximum)
        for method,j in methods.items():
            delta = group[j]-group[baseline]; c = next(c for c in sample['candidates'] if c['local']==j)
            rows.append(dict(token=token,session_id=d['metadata']['session_id'],method=method,candidate_local=j,candidate_vocab=c['vocab'],
                teacher_rank=sample['candidates'].index(c)+1,raw_pdm=float(group[j][0]),gain_pp=float(100*delta[0]),
                intervention=j!=baseline,pdm_decreased=bool(delta[0]<-1e-8),relative_component_harm=bool(np.any(delta[safety]<-1e-8)),
                **{'delta_'+m:float(delta[i+1]) for i,m in enumerate(base.METRICS)}))
    metrics = {}
    for method in sorted({r['method'] for r in rows}):
        group = [r for r in rows if r['method']==method]
        metrics[method] = dict(factorial.summarize(group),
            **{key:sum(r[key] for r in group) for key in ('intervention','pdm_decreased','relative_component_harm')},
            improved_scenes=sum(r['gain_pp']>1e-8 for r in group),
            component_degradations={m:sum(r['delta_'+m]<-1e-8 for r in group) for m in base.METRICS},
            tail_selections=sum(r['teacher_rank']>16 for r in group))
    pair_quality = {}; measured_pairs = []
    for pair in pairs:
        token = pair['token']; baseline = samples[token]['baseline_local']; delta = labels[token][pair['candidate_local']]-labels[token][baseline]
        helpful = bool(delta[0]>=data.CONFIG['improvement_min'] and np.all(delta[safety]>=-1e-8) and delta[lane]>=-1e-8)
        harmful = bool(delta[0]<=-data.CONFIG['improvement_min'] or np.any(delta[safety]<-1e-8) or delta[lane]<-1e-8)
        measured_pairs.append(dict(pair,raw_gain_pp=float(delta[0]*100),helpful=helpful,harmful=harmful,
            **{'delta_'+m:float(delta[i+1]) for i,m in enumerate(base.METRICS)}))
    for mode in data.MODES:
        group = [r for r in measured_pairs if r['mode']==mode]
        pair_quality[mode] = dict(pairs=len(group),order_choice_flips=sum(r['order_choice_flip'] for r in group),
            exact_ties=sum(r['exact_tie'] for r in group),helpful_pairs=sum(r['helpful'] for r in group),
            harmful_pairs=sum(r['harmful'] for r in group),
            helpful_accepted_both050=sum(r['helpful'] and r['minimum_preference']>.5 for r in group),
            harmful_accepted_both050=sum(r['harmful'] and r['minimum_preference']>.5 for r in group))
    contrasts = {}
    for policy in data.POLICIES:
        for a,b in (('road','none'),('road','blank'),('blank','none')):
            contrasts[a+'_minus_'+b+'/'+policy] = factorial.contrast(rows,a+'/'+policy,b+'/'+policy)
    timing = {}
    for mode,group in performance.items():
        times = [r['seconds'] for r in group if not r['cold_call']]
        timing[mode] = dict(requests=len(group),max_input_tokens=max(r['input_tokens'] for r in group),
            warm_p50_seconds=float(np.quantile(times,.5)) if times else None,
            warm_p95_seconds=float(np.quantile(times,.95)) if times else None)
    write_csv(out/'per_scene.csv',rows); write_csv(out/'per_pair.csv',measured_pairs)
    value = dict(status='complete',scenes=len(decisions),sessions=len({d['metadata']['session_id'] for d in decisions}),
        pairs=len(pairs)//3,requests=sum(len(g) for g in performance.values()),primary=data.CONFIG['primary'],
        methods=metrics,paired_image_contrasts=contrasts,pair_quality=pair_quality,performance=timing,
        pair_quality_note='Helpful: PDM gain >=0.005 and no degradation of five safety components or lane keeping. '
            'Harmful: PDM loss >=0.005 or any such degradation. Counts are correlated within scenes/sessions; not calibrated risk.',
        decision_commit_sha256=sha(out/'decision_commit.json'),fresh_holdout_processed=0,note=data.NOTE)
    write(out/'report.json',value); return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('plan','bind','worker-pilot','worker','report'))
    p.add_argument('--root',type=Path,required=True); p.add_argument('--replicas',type=int,default=8)
    a = p.parse_args(); root = a.root.expanduser().resolve()
    if a.replicas not in (1,2,4,8): raise ValueError('Use 1/2/4/8 GPUs')
    name = os.environ.get('DREX_REPLACEMENT_NAME','replacement_gpu8_v1')
    source_name = os.environ.get('DREX_REPLACEMENT_SOURCE','table_topk_night_gpu8_v1')
    if not re.fullmatch(r'replacement_[A-Za-z0-9_-]{1,64}',name) or not re.fullmatch(r'table_[A-Za-z0-9_-]{1,80}',source_name):
        raise ValueError('Invalid dedicated run/source name')
    out = root/'runs'/name; out.mkdir(parents=True,exist_ok=True)
    import fcntl
    from table_group import gpu_binding
    phase = a.stage; worker = None
    if phase in ('worker','worker-pilot','bind'):
        rank = gpu_binding(out,a.replicas,phase)
        if phase=='bind': return
        worker = out/'shards'/f'{rank:02d}'; worker.mkdir(parents=True,exist_ok=True)
    target = worker or out
    with (target/'.execution.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another process owns this stage/shard')
        try:
            if phase=='plan': result = prepare(root,root/'runs'/source_name,out,a.replicas)
            else:
                identity,plan,source,source_plan = context(out)
                if identity['replicas']!=a.replicas or source.name!=source_name: raise RuntimeError('Frozen execution changed')
                if sha(root/'requirements.lock.txt')!=identity['environment_lock_sha256'] or read(root/'model_plan.json')!=identity['model_plan']:
                    raise RuntimeError('Model/runtime changed')
                if phase=='report': result = report(out)
                else:
                    records = [r for r in identity['records'] if r['token'] in set(plan['assignments'][rank])]
                    if (worker/'commit.json').exists():
                        result = infer_tasks(out,source,source_plan,records,worker,
                            lambda _: (_ for _ in ()).throw(AssertionError('Unexpected inference')),max_new=0)
                    else: result = run_gpu(root,out,source,source_plan,records,worker,phase=='worker-pilot')
                    if result['status']=='partial' and phase!='worker-pilot':
                        raise RuntimeError('Worker time budget reached; resume SAME name/count before report')
            print(json.dumps(result,indent=2),flush=True)
        except Exception as exc:
            write(target/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc())); raise


if __name__=='__main__': main()
