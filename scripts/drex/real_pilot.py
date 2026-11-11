"""Freeze real-scene choices first, then evaluate stored raw PDM offline."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import traceback

import numpy as np

import real_data as data
from runtime import environment, read, remaining_bytes, write
from transfer import checked, sha
from night import load_result, save_result, validate_probabilities


def result_name(token, mode, ordering):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', token):
        raise ValueError('Unsafe token')
    return f'{token}-{mode}-{ordering}.json'


def choice(probabilities, inputs):
    p = np.asarray(probabilities)
    maxima = np.flatnonzero(p == p.max()).tolist()
    b = inputs['pool_local'].index(inputs['baseline_local'])
    # A fixed exact-tie fallback, never an outcome-dependent threshold.
    return inputs['pool_local'][b if b in maxima else maxima[0]]


def decisions_from_results(records, samples, results):
    by_token = defaultdict(dict)
    for r in results:
        key = (r['mode'],r['ordering'])
        if key in by_token[r['token']]:
            raise ValueError('Duplicate inference result')
        by_token[r['token']][key] = r
    decisions=[]
    if set(by_token) != {r['token'] for r in records}:
        raise ValueError('Complete cohort required before freezing decisions')
    for r in records:
        token=r['token']; s=samples[token]; group=by_token[token]
        if set(group)!=set(data.TASKS):
            raise ValueError('All camera permutations and blank control required')
        aligned=[]
        for mode,kind in data.TASKS:
            item=group[(mode,kind)]
            order=data.permutations(token,kind)
            if item['pool_positions'] != order:
                raise ValueError('Candidate permutation alignment changed')
            validate_probabilities(item['probabilities'],16)
            p=np.empty(16,dtype=np.float64); p[order]=item['probabilities']
            if mode=='camera': aligned.append(p)
        mean=np.mean(aligned,axis=0)
        blank=np.empty(16); b=group[('blank','canonical')]
        blank[b['pool_positions']]=b['probabilities']
        random_index=int(data.digest([data.SEED,token,'random_top16'])[:16],16)%16
        methods=dict(baseline=s['baseline_local'], drex_single=choice(aligned[0],s),
                     drex_mean3=choice(mean,s), blank_single=choice(blank,s),
                     random_top16=s['pool_local'][random_index])
        decisions.append(dict(metadata=r, pool_local=s['pool_local'],pool_vocab=s['pool_vocab'],
            baseline_local=s['baseline_local'], mean3_probabilities=mean.tolist(),
            methods={name:dict(candidate_local=local,candidate_vocab=s['pool_vocab'][s['pool_local'].index(local)])
                     for name,local in methods.items()}))
    return decisions


def infer_cohort(package,out,manifest,records,scenes,vocab,infer,max_new=None,budget_seconds=3*3600):
    started=time.monotonic(); new=0; results=[]; samples={}; expected=set(); stopped=False
    for r in records:
        expected.update(result_name(r['token'],*condition) for condition in data.TASKS)
    if {p.name for p in (out/'results').glob('*.json')}-expected:
        raise RuntimeError('Unexpected real-pilot result files')
    for r in records:
        token=r['token']; s=data.load_inputs(package,token,manifest,vocab); samples[token]=s
        for mode,kind in data.TASKS:
            path=out/'results'/result_name(token,mode,kind)
            if stopped and not path.exists(): continue
            if not path.exists() and ((max_new is not None and new>=max_new) or time.monotonic()-started>=budget_seconds):
                stopped=True; continue
            row,order,query_hash=data.payload(s,scenes[token],package,manifest,mode,kind)
            if path.exists():
                result=load_result(path,query_hash)
                if (result['token'],result['mode'],result['ordering'],result['pool_positions'])!=(token,mode,kind,order):
                    raise RuntimeError('Cached scene/candidate alignment changed')
            else:
                result=dict(status='ok',token=token,mode=mode,ordering=kind,k=16,
                            request_sha256=query_hash,pool_positions=order)
                try:
                    measured=infer(row)
                    probabilities=[float(x) for x in measured['probabilities']]
                    validate_probabilities(probabilities,16)
                    result.update(measured,probabilities=probabilities)
                except Exception as exc:
                    result.update(status='technical_failure',error=repr(exc),traceback=traceback.format_exc())
                    save_result(path,result)
                    write(out/'inference_status.json',dict(status='failed',failed_request=str(path),
                        completed_requests=len(results),expected_requests=len(expected),outcomes_read=False))
                    raise
                save_result(path,result); new+=1
                print(f"REAL {len(results)+1}/{len(expected)} token={token} {mode}/{kind}",flush=True)
            results.append(result)
        write(out/'inference_status.json',dict(status='running',completed_requests=len(results),
            expected_requests=len(expected),outcomes_read=False))
    complete=len(results)==len(expected)
    status=dict(status='choices_frozen' if complete else 'partial',completed_requests=len(results),
        expected_requests=len(expected),new_requests=new,outcomes_read=False,seconds=time.monotonic()-started)
    if complete:
        decisions=decisions_from_results(records,samples,results)
        data.freeze(out/'decisions.json',decisions)
        commit=dict(identity_sha256=sha(out/'identity.json'),selection_sha256=sha(out/'selection.json'),
            decisions_sha256=sha(out/'decisions.json'),
            results={p.name:sha(p) for p in sorted((out/'results').glob('*.json'))},
            note='Complete cohort choices committed before deserializing any outcome arrays.')
        data.freeze(out/'decision_commit.json',commit)
    write(out/'inference_status.json',status)
    return status


def committed_decisions(out,records,samples):
    commit=read(out/'decision_commit.json')
    for name,key in (('identity.json','identity_sha256'),('selection.json','selection_sha256'),('decisions.json','decisions_sha256')):
        if sha(out/name)!=commit[key]:
            raise RuntimeError('Decision freeze changed: '+name)
    expected={result_name(r['token'],*condition) for r in records for condition in data.TASKS}
    actual={p.name for p in (out/'results').glob('*.json')}
    if set(commit['results'])!=expected or actual!=expected:
        raise ValueError('Full cohort decisions must be frozen before any outcome lookup')
    results=[]
    for name,digest in commit['results'].items():
        path=out/'results'/name
        if sha(path)!=digest: raise RuntimeError('Committed result changed')
        results.append(load_result(path))
    decisions=decisions_from_results(records,samples,results)
    if decisions!=read(out/'decisions.json'):
        raise RuntimeError('Decisions disagree with frozen inference probabilities')
    return decisions,results


def load_labels(package,token,manifest,inputs):
    # Called only after committed_decisions() verifies the ENTIRE fixed cohort.
    name='bundle/examples/'+token+'.npz'; path=checked(package,name)
    if sha(path)!=manifest['files'][name]['sha256']: raise RuntimeError('Label source changed')
    with np.load(path,allow_pickle=False) as f:
        indices=f['label_indices'].copy(); raw=f['label_raw'].astype(np.float64)
    if (indices.dtype.kind not in 'iu' or indices.shape!=(16,) or indices.tolist()!=inputs['pool_local']
            or raw.shape!=(16,9) or not np.isfinite(raw).all() or np.any((raw<0)|(raw>1))):
        raise ValueError('Stored raw label alignment/values invalid; no reconstruction permitted')
    return raw


def score_decisions(decisions,labels):
    rows=[]
    safety_columns=[1+data.METRICS.index(k) for k in data.SAFETY]
    for d in decisions:
        token=d['metadata']['token']; raw=labels[token]; pool=d['pool_local']
        b=pool.index(d['baseline_local']); base=raw[b]
        methods=dict(d['methods'])
        maximum=raw[:,0].max()
        oracle=b if raw[b,0]==maximum else int(np.argmax(raw[:,0]))
        methods['oracle_top16_upper_bound']=dict(candidate_local=pool[oracle],candidate_vocab=d['pool_vocab'][oracle])
        for method,selected in methods.items():
            j=pool.index(selected['candidate_local'])
            if selected['candidate_vocab']!=d['pool_vocab'][j]: raise ValueError('Choice vocabulary alignment changed')
            delta=raw[j]-base
            rows.append(dict(token=token,session_id=d['metadata']['session_id'],method=method,**selected,
                raw_pdm=float(raw[j,0]),baseline_pdm=float(base[0]),raw_gain_pp=float(100*delta[0]),
                intervention=selected['candidate_local']!=d['baseline_local'],
                raw_harm=bool(np.any(delta[safety_columns]<-1e-8)),pdm_decreased=bool(delta[0]<-1e-8),
                useful_improvement=bool(delta[0]>=data.CONFIG['useful_gain_min']),
                **{'raw_delta_'+k:float(delta[1+i]) for i,k in enumerate(data.METRICS)}))
    return rows


def report_rows(rows,results):
    methods={}
    for method in sorted({r['method'] for r in rows}):
        group=[r for r in rows if r['method']==method]
        sessions=defaultdict(list)
        for r in group: sessions[r['session_id']].append(r['raw_gain_pp'])
        gains=np.asarray([np.mean(v) for _,v in sorted(sessions.items())])
        rng=np.random.default_rng(data.SEED)
        means=gains[rng.integers(0,len(gains),(data.CONFIG['bootstrap_reps'],len(gains)))].mean(axis=1)
        methods[method]=dict(scenes=len(group),sessions=len(sessions),raw_pdm_mean=float(np.mean([r['raw_pdm'] for r in group])),
            gain_micro_pp=float(np.mean([r['raw_gain_pp'] for r in group])),gain_macro_pp=float(gains.mean()),
            gain_cluster_ci95_pp=np.quantile(means,[.025,.975]).tolist(),
            interventions=sum(r['intervention'] for r in group),pdm_decreased=sum(r['pdm_decreased'] for r in group),
            useful_improvements=sum(r['useful_improvement'] for r in group),relative_component_harm=sum(r['raw_harm'] for r in group),
            component_degradations={k:sum(r['raw_delta_'+k]<-1e-8 for r in group) for k in data.SAFETY},
            uses_outcomes_for_choice=method=='oracle_top16_upper_bound',
            gain_pp_by_session={s:float(np.mean(v)) for s,v in sorted(sessions.items())})
    groups=defaultdict(dict)
    for r in results: groups[r['token']][(r['mode'],r['ordering'])]=r
    changes={}
    for kind in ('reversed','permuted'):
        changed=0
        for g in groups.values():
            a,b=g[('camera','canonical')],g[('camera',kind)]
            changed+=a['pool_positions'][int(np.argmax(a['probabilities']))]!=b['pool_positions'][int(np.argmax(b['probabilities']))]
        changes[kind]=dict(scenes=len(groups),argmax_choice_flips=changed)
    return dict(status='complete',primary=data.CONFIG['primary'],methods=methods,order_sensitivity=changes,
                requests=len(results),scenes=len(groups),note=data.NOTE)


def evaluate(package,out,manifest,records,vocab):
    samples={r['token']:data.load_inputs(package,r['token'],manifest,vocab) for r in records}
    decisions,results=committed_decisions(out,records,samples)
    labels={r['token']:load_labels(package,r['token'],manifest,samples[r['token']]) for r in records}
    rows=score_decisions(decisions,labels)
    report=report_rows(rows,results)
    report['decision_commit_sha256']=sha(out/'decision_commit.json')
    write(out/'per_scene.json',rows)
    path=out/'per_scene.csv'; tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)
    write(out/'report.json',report)
    return report


def run_gpu(root,package,out,manifest,records,scenes,vocab,max_new):
    # Import heavyweight runtime only after CPU preflight and immutable input freeze.
    import torch
    from drex_max.model import DecisionModel
    env=environment(root,True)
    data.freeze(out/'inference_environment.json',{k:env[k] for k in
        ('python','torch','cuda_build','transformers','gpu','vram_bytes')})
    target=root/'models/drex-max-v1'
    if remaining_bytes(target,read(root/'model_plan.json')['files']): raise RuntimeError('Model files changed; no downloads in real pilot')
    config=read(target/'decision_config.json')
    if config.get('format_version')!=1 or config.get('attention_mode','noncausal_full_attention')!='noncausal_full_attention':
        raise ValueError('Unexpected model attention protocol')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    write(out/'attempts'/(stamp+'.json'),env)
    print('Loading Drex BF16 for real train-only pilot; 2 images/request, batch=1',flush=True)
    torch.cuda.reset_peak_memory_stats(); tick=time.monotonic()
    model=DecisionModel(target,device='cuda:0',max_image_pixels=data.CONFIG['max_image_pixels'])
    if model.readout.weight.dtype!=torch.bfloat16: raise RuntimeError('Unexpected readout precision')
    torch.cuda.synchronize(); load_seconds=time.monotonic()-tick
    calls=0
    def infer(row):
        nonlocal calls
        batch=model.prepare([row],max_length=data.CONFIG['max_length'])
        tokens=int(batch.inputs['input_ids'].shape[1])
        grid=batch.inputs.get('image_grid_thw')
        if ('pixel_values' not in batch.inputs or grid is None or grid.shape[0]!=2
                or tokens>=data.CONFIG['max_length']):
            raise RuntimeError('Missing image inputs or context limit reached')
        del batch
        torch.cuda.synchronize(); started=time.monotonic()
        p=model.predict([row],batch_size=1,max_length=data.CONFIG['max_length'])[0]
        torch.cuda.synchronize(); calls+=1
        return dict(probabilities=p,input_tokens=tokens,seconds=time.monotonic()-started,cold_call=calls==1)
    status=infer_cohort(package,out,manifest,records,scenes,vocab,infer,max_new=max_new)
    status.update(load_seconds=load_seconds,peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    write(out/'inference_status.json',status)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('plan','infer','all','report'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--max-new',type=int)
    a=p.parse_args(); root=a.root.expanduser().resolve()
    name=os.environ.get('DREX_REAL_NAME','real_train64_v1')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',name): raise ValueError('Invalid DREX_REAL_NAME')
    out=root/'runs'/name
    package=Path(os.environ.get('DREX_REAL_DATA',str(root/'data/navsim_train_v1/package'))).expanduser().resolve()
    try:
        manifest,records,scenes,vocab=data.prepare(package,root,out)
        if a.stage=='plan':
            for r in records[:4]:
                s=data.load_inputs(package,r['token'],manifest,vocab)
                row,order,_=data.payload(s,scenes[r['token']],package,manifest,'camera','canonical')
                for i,im in enumerate(row['images']): im.save(out/f"preview-{r['token']}-{i+1}.png")
            result=dict(status='planned',scenes=len(records),sessions=len({r['session_id'] for r in records}),
                requests=len(records)*len(data.TASKS),primary=data.CONFIG['primary'],outcomes_read=False,
                vocab_shape=list(vocab.shape),output=str(out),note=data.NOTE)
            write(out/'plan.json',result)
        elif a.stage=='report':
            result=evaluate(package,out,manifest,records,vocab)
        else:
            if (out/'decision_commit.json').exists():
                samples={r['token']:data.load_inputs(package,r['token'],manifest,vocab) for r in records}
                committed_decisions(out,records,samples)
                result=dict(status='choices_frozen',completed_requests=len(records)*len(data.TASKS),
                            expected_requests=len(records)*len(data.TASKS),new_requests=0,outcomes_read=False)
                write(out/'inference_status.json',result)
            else:
                result=run_gpu(root,package,out,manifest,records,scenes,vocab,a.max_new)
            if a.stage=='all' and result['status']=='choices_frozen':
                result=evaluate(package,out,manifest,records,vocab)
        print(json.dumps(result,indent=2),flush=True)
    except Exception as exc:
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__': main()
