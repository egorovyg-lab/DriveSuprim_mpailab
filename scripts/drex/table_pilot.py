"""Scene-parallel, frozen top-K table/image ablation on portable training data."""
import argparse
from collections import defaultdict
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback

import numpy as np
from PIL import Image

import real_data as base
import table_data as data
from night import load_result, save_result, validate_probabilities
from runtime import read, write, environment, remaining_bytes
from transfer import checked, sha, verify

SOURCES = ('table_pilot.py', 'table_data.py', 'run_table.sh', 'real_data.py', 'runtime.py',
           'transfer.py', 'night.py', 'synthetic.py')


def plan(root, package, out, replicas, cohort):
    if replicas not in (1, 2, 4, 8):
        raise ValueError('Use 1, 2, 4 or 8 replicas')
    verify(package)  # Byte hashing only: no outcome arrays are deserialized.
    manifest, index, scenes, vocab = base.package_metadata(package)
    records = data.cohort(index, cohort)
    if len(records) < replicas:
        raise ValueError('More replicas than selected scenes')
    identity = dict(config=data.CONFIG, cohort=cohort, records=records, replicas=replicas,
        package=str(package), package_manifest_sha256=sha(package/'package_manifest.json'),
        model_plan=read(root/'model_plan.json'), environment_lock_sha256=sha(root/'requirements.lock.txt'),
        sources={n:sha(Path(__file__).with_name(n)) for n in SOURCES}, note=data.NOTE)
    base.freeze(out/'identity.json', identity)
    entries = {}
    for i,r in enumerate(records):
        token = r['token']
        sample = data.load_inputs(package, token, manifest, vocab)
        path = out/'inputs'/(token+'.json')
        base.freeze(path, sample)
        paths = base.camera_paths(package, scenes[token], manifest)
        for p in paths:
            with Image.open(p) as im:
                im.verify()
        photo = base.camera_board(paths)
        buffer = io.BytesIO(); photo.save(buffer, format='PNG')
        p = out/'photos'/(token+'.png'); p.parent.mkdir(parents=True, exist_ok=True)
        pixels = buffer.getvalue()
        if p.exists() and p.read_bytes() != pixels:
            raise RuntimeError('Frozen camera board changed; use a new output name')
        if not p.exists():
            tmp = p.with_suffix('.tmp'); tmp.write_bytes(pixels); tmp.replace(p)
        entries[token] = dict(inputs_sha256=sha(path), photo_sha256=sha(p),
            baseline_local=sample['baseline_local'], candidates=sample['candidates'])
        if (i+1)%100==0 or i+1==len(records):
            print(f'TABLE PLAN {i+1}/{len(records)}', flush=True)
    base.freeze(out/'plan.json', dict(identity_sha256=sha(out/'identity.json'), entries=entries,
        assignments=[[r['token'] for r in records[i::replicas]] for i in range(replicas)]))
    # Request is outcome-independent and always asks for the whole largest pool.
    request = dict(protocol='table_top64_labels_v1',
        package_manifest_sha256=identity['package_manifest_sha256'],
        plan_sha256=sha(out/'plan.json'), metrics=['pdm_score', *base.METRICS],
        scenes=[dict(token=r['token'],candidates=[[c['local'],c['vocab']] for c in entries[r['token']]['candidates']])
                for r in records])
    # Compact grouped encoding fits large cohorts in the bounded SSH upload.
    rp = out/'label_request.json'
    if rp.exists():
        if read(rp)!=request: raise RuntimeError('Frozen label request changed')
    else:
        tmp = rp.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(request,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8'); tmp.replace(rp)
    return dict(status='planned', scenes=len(records), sessions=len({r['session_id'] for r in records}),
        requests=len(records)*18, replicas=replicas, candidates_per_scene=list(data.KS),
        outcomes_read=False, label_request=str(out/'label_request.json'), note=data.NOTE)


def context(out):
    identity, plan = read(out/'identity.json'), read(out/'plan.json')
    if identity['config'] != data.CONFIG or sha(out/'identity.json') != plan['identity_sha256']:
        raise RuntimeError('Frozen table identity/config changed')
    for n,digest in identity['sources'].items():
        if sha(Path(__file__).with_name(n)) != digest:
            raise RuntimeError('Table code changed; choose a new DREX_TABLE_NAME: '+n)
    records = identity['records']
    expected = [[r['token'] for r in records[i::identity['replicas']]] for i in range(identity['replicas'])]
    if plan['assignments'] != expected or set(plan['entries']) != {r['token'] for r in records}:
        raise RuntimeError('Frozen scene assignment changed')
    return identity, plan


def scene(out, plan, token):
    entry = plan['entries'][token]
    ip, pp = out/'inputs'/(token+'.json'), out/'photos'/(token+'.png')
    if sha(ip) != entry['inputs_sha256'] or sha(pp) != entry['photo_sha256']:
        raise RuntimeError('Frozen feature/photo changed: '+token)
    sample = read(ip)
    if sample['candidates'] != entry['candidates'] or sample['baseline_local'] != entry['baseline_local']:
        raise RuntimeError('Frozen candidate alignment changed')
    with Image.open(pp) as image:
        photo = image.convert('RGB')
    return sample, photo


def infer_tasks(out, plan, records, worker_out, infer, max_new=None, budget_seconds=7*3600):
    started = time.monotonic(); new = 0; complete = 0
    tasks = data.tasks(records)
    expected = {data.task_name(*t) for t in tasks}
    if {p.name for p in (worker_out/'results').glob('*.json')} - expected:
        raise RuntimeError('Unexpected worker result files')
    last = None; sample = photo = None
    for token,k,mode,kind in tasks:
        if token != last:
            sample,photo = scene(out, plan, token); last = token
        path = worker_out/'results'/data.task_name(token,k,mode,kind)
        if not path.exists() and ((max_new is not None and new>=max_new)
                                  or time.monotonic()-started>=budget_seconds):
            break
        row,order,digest = data.payload(token,sample,k,mode,kind,photo)
        if path.exists():
            result = load_result(path,digest)
            validate_result(result,token,k,mode,kind,order)
        else:
            result = dict(status='ok',token=token,k=k,mode=mode,ordering=kind,
                          pool_positions=order,request_sha256=digest)
            try:
                measured = infer(row)
                validate_probabilities(measured['probabilities'],k)
                result.update(measured)
                validate_result(result,token,k,mode,kind,order)
            except Exception as exc:
                result.update(status='technical_failure',error=repr(exc),traceback=traceback.format_exc())
                save_result(path,result)
                write(worker_out/'inference_status.json',dict(status='failed',failed_request=str(path),
                      completed_requests=complete,expected_requests=len(tasks),outcomes_read=False))
                raise
            save_result(path,result); new += 1
            print(f'TABLE {complete+1}/{len(tasks)} {token} k={k} {mode}/{kind}',flush=True)
        complete += 1
        write(worker_out/'inference_status.json',dict(status='running',completed_requests=complete,
            expected_requests=len(tasks),new_requests=new,outcomes_read=False))
    status = dict(status='choices_frozen' if complete==len(tasks) else 'partial',
        completed_requests=complete,expected_requests=len(tasks),new_requests=new,
        seconds=time.monotonic()-started,outcomes_read=False)
    if complete==len(tasks):
        base.freeze(worker_out/'commit.json',dict(plan_sha256=sha(out/'plan.json'),
            environment_sha256=sha(worker_out/'environment.json'),
            results={n:sha(worker_out/'results'/n) for n in sorted(expected)}))
    write(worker_out/'inference_status.json',status)
    return status


def validate_result(r,token,k,mode,kind,order):
    if (r['token'],r['k'],r['mode'],r['ordering'],r['pool_positions']) != (token,k,mode,kind,order):
        raise RuntimeError('Result candidate/task alignment changed')
    validate_probabilities(r['probabilities'],k)


def run_gpu(root,out,plan,records,worker_out,pilot):
    import torch
    from drex_max.model import DecisionModel
    envroot = worker_out/'runtime'; envroot.mkdir(parents=True,exist_ok=True)
    lock = envroot/'requirements.lock.txt'
    if lock.exists() and sha(lock) != sha(root/'requirements.lock.txt'):
        raise RuntimeError('Worker environment lock changed')
    shutil.copyfile(root/'requirements.lock.txt',lock)
    env = environment(envroot,True)
    base.freeze(worker_out/'environment.json',{k:env[k] for k in
        ('python','torch','cuda_build','transformers','gpu','vram_bytes')})
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Each array worker must see exactly one allocated GPU')
    target = root/'models/drex-max-v1'
    if remaining_bytes(target,read(root/'model_plan.json')['files']):
        raise RuntimeError('Model files changed/missing; no download in inference')
    config = read(target/'decision_config.json')
    if config.get('format_version')!=1 or config.get('attention_mode','noncausal_full_attention')!='noncausal_full_attention':
        raise RuntimeError('Unexpected Drex attention protocol')
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    model = DecisionModel(target,device='cuda:0',max_image_pixels=data.CONFIG['max_image_pixels'])
    if model.readout.weight.dtype != torch.bfloat16:
        raise RuntimeError('Expected BF16 model')
    torch.cuda.synchronize(); load_seconds = time.monotonic()-started; calls = 0

    def infer(row):
        nonlocal calls
        batch = model.prepare([row],max_length=data.CONFIG['max_length'])
        tokens = int(batch.inputs['input_ids'].shape[1]); expected = len(row.get('images',[]))
        grid = batch.inputs.get('image_grid_thw')
        actual = 0 if grid is None else int(grid.shape[0])
        if tokens>=data.CONFIG['max_length'] or actual!=expected or ('pixel_values' in batch.inputs)!=(expected>0):
            raise RuntimeError('Context truncated/full or wrong image condition; no silent truncation allowed')
        del batch
        torch.cuda.synchronize(); before = time.monotonic()
        p = model.predict([row],batch_size=1,max_length=data.CONFIG['max_length'])[0]
        torch.cuda.synchronize(); calls += 1
        return dict(probabilities=[float(v) for v in p],input_tokens=tokens,has_image_pixels=expected>0,
                    seconds=time.monotonic()-before,cold_call=calls==1)

    result = infer_tasks(out,plan,records,worker_out,infer,max_new=18 if pilot else None)
    result.update(load_seconds=load_seconds,peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    write(worker_out/'inference_status.json',result)
    return result


def freeze_decisions(out):
    identity,plan = context(out); records = identity['records']; results = {}; profiles = []
    for i,tokens in enumerate(plan['assignments']):
        w = out/'shards'/f'{i:02d}'; commit = read(w/'commit.json')
        subset = [r for r in records if r['token'] in set(tokens)]
        expected = {data.task_name(*t) for t in data.tasks(subset)}
        if (commit['plan_sha256']!=sha(out/'plan.json') or set(commit['results'])!=expected
                or sha(w/'environment.json')!=commit['environment_sha256']
                or {p.name for p in (w/'results').glob('*.json')}!=expected):
            raise RuntimeError('Incomplete or changed worker commitment; labels remain unread')
        profiles.append(read(w/'environment.json'))
        for name,digest in commit['results'].items():
            p = w/'results'/name
            if sha(p)!=digest or name in results:
                raise RuntimeError('Worker result changed or duplicated')
            results[name] = load_result(p)
    if any(p!=profiles[0] for p in profiles[1:]):
        raise RuntimeError('GPU replicas used different environment profiles')
    decisions = []; sensitivity = defaultdict(lambda: dict(comparisons=0,choice_flips=0))
    for r in records:
        token = r['token']; sample,photo = scene(out,plan,token)
        for k in data.KS:
            methods = data.rules(sample,k)
            for mode in data.MODES:
                aligned = []
                for kind in data.ORDERS:
                    row,order,digest = data.payload(token,sample,k,mode,kind,photo)
                    item = results[data.task_name(token,k,mode,kind)]
                    if item['request_sha256'] != digest:
                        raise RuntimeError('Request changed before merge')
                    validate_result(item,token,k,mode,kind,order)
                    p = np.empty(k); p[order] = item['probabilities']; aligned.append(p)
                methods[mode+'/single'] = data.pick(aligned[0],sample,k)
                methods[mode+'/mean3'] = data.pick(np.mean(aligned,axis=0),sample,k)
                for j,kind in enumerate(data.ORDERS[1:],1):
                    s = sensitivity[f'{mode}/k{k}/{kind}']; s['comparisons'] += 1
                    s['choice_flips'] += data.pick(aligned[0],sample,k)!=data.pick(aligned[j],sample,k)
            pool = sample['candidates'][:k]
            decisions.append(dict(metadata=r,k=k,baseline_local=sample['baseline_local'],
                pool=[dict(local=c['local'],vocab=c['vocab']) for c in pool],methods=methods))
    base.freeze(out/'decisions.json',decisions)
    base.freeze(out/'decision_commit.json',dict(plan_sha256=sha(out/'plan.json'),
        decisions_sha256=sha(out/'decisions.json'),
        workers={str(i):sha(out/'shards'/f'{i:02d}'/'commit.json') for i in range(identity['replicas'])}))
    return identity,decisions,results,dict(sensitivity)


def label_lookup(package, manifest, decisions, external=None):
    labels = {}
    for token in sorted({d['metadata']['token'] for d in decisions}):
        rel = 'bundle/examples/'+token+'.npz'; path = checked(package,rel)
        if sha(path)!=manifest['files'][rel]['sha256']:
            raise RuntimeError('Changed label source')
        with np.load(path,allow_pickle=False) as f:
            ids,raw = f['label_indices'].copy(),f['label_raw'].astype(float)
        expected = next([c['local'] for c in d['pool']] for d in decisions
                        if d['metadata']['token']==token and d['k']==16)
        if ids.shape!=(16,) or raw.shape!=(16,9) or ids.tolist()!=expected:
            raise ValueError('Invalid original top16 label alignment')
        labels[token] = {int(j):v for j,v in zip(ids,raw)}
    if external is not None:
        for row in external:
            token,j,value = row['token'],row['candidate_local'],np.asarray(row['raw'],dtype=float)
            if token not in labels:
                raise ValueError('Unexpected external label token')
            expected = {c['local']:c['vocab'] for d in decisions if d['metadata']['token']==token for c in d['pool']}
            if j not in expected or row['candidate_vocab']!=expected[j] or value.shape!=(9,):
                raise ValueError('External candidate label alignment changed')
            if j in labels[token] and not np.allclose(labels[token][j],value,atol=1e-8,rtol=1e-7):
                raise RuntimeError('Original top16 metric parity failed')
            labels[token][j] = value
    if any(not np.isfinite(v).all() or np.any((v<0)|(v>1)) for group in labels.values() for v in group.values()):
        raise ValueError('Invalid measured label values')
    return labels


def report(out, external_path=None):
    identity,decisions,results,sensitivity = freeze_decisions(out)  # Entire cohort first; no labels above.
    package = Path(identity['package']); manifest = read(package/'package_manifest.json')
    if sha(package/'package_manifest.json')!=identity['package_manifest_sha256']:
        raise RuntimeError('Package identity changed')
    external = None
    if external_path is not None:
        external = read(external_path)
        if (external['request_digest']!=base.digest(read(out/'label_request.json')) or external['status']!='complete'
                or external['metrics']!=['pdm_score',*base.METRICS]):
            raise RuntimeError('Wrong/incomplete external label set')
        keys = [(r['token'],r['candidate_local'],r['candidate_vocab']) for r in external['rows']]
        wanted = [(r['token'],r['candidate_local'],r['candidate_vocab']) for r in data.requested_candidates(read(out/'label_request.json'))]
        if len(keys)!=len(set(keys)) or set(keys)!=set(wanted):
            raise RuntimeError('External labels must cover exactly the requested top64 pool')
        external = external['rows']
    labels = label_lookup(package,manifest,decisions,external)
    rows = []; missing = []; safety = [1+base.METRICS.index(m) for m in base.SAFETY]
    for d in decisions:
        token = d['metadata']['token']; pool = d['pool']; group = labels[token]
        absent = [c['local'] for c in pool if c['local'] not in group]
        if absent:
            missing.append(dict(token=token,k=d['k'],missing=len(absent))); continue
        baseline = group[d['baseline_local']]
        methods = dict(d['methods'])
        maximum = max(group[c['local']][0] for c in pool)
        methods['oracle_upper_bound'] = (d['baseline_local'] if baseline[0]==maximum else
            next(c['local'] for c in pool if group[c['local']][0]==maximum))
        for name,j in methods.items():
            value = group[j]; delta = value-baseline
            rows.append(dict(token=token,session_id=d['metadata']['session_id'],k=d['k'],method=name,
                candidate_local=j,candidate_vocab=next(c['vocab'] for c in pool if c['local']==j),
                raw_pdm=float(value[0]),gain_pp=float(100*delta[0]),intervention=j!=d['baseline_local'],
                relative_component_harm=bool(np.any(delta[safety]<-1e-8)),
                pdm_decreased=bool(delta[0]<-1e-8),
                **{'delta_'+m:float(delta[1+i]) for i,m in enumerate(base.METRICS)}))
    metrics = {}
    for k,name in sorted({(r['k'],r['method']) for r in rows}):
        group = [r for r in rows if (r['k'],r['method'])==(k,name)]
        by_session = defaultdict(list)
        for r in group: by_session[r['session_id']].append(r['gain_pp'])
        means = np.asarray([np.mean(v) for _,v in sorted(by_session.items())])
        rng = np.random.default_rng(data.CONFIG['seed'])
        boot = means[rng.integers(0,len(means),(data.CONFIG['bootstrap_reps'],len(means)))].mean(axis=1)
        metrics[f'{name}/k{k}'] = dict(scenes=len(group),sessions=len(means),gain_micro_pp=float(np.mean([r['gain_pp'] for r in group])),
            gain_macro_pp=float(means.mean()),gain_cluster_ci95_pp=np.quantile(boot,[.025,.975]).tolist(),
            interventions=sum(r['intervention'] for r in group),pdm_decreased=sum(r['pdm_decreased'] for r in group),
            relative_component_harm=sum(r['relative_component_harm'] for r in group),
            gain_by_session={s:float(np.mean(v)) for s,v in by_session.items()},uses_outcomes_for_choice=name=='oracle_upper_bound')
    comparisons = {}
    for k in data.KS:
        a = {r['token']:r for r in rows if r['k']==k and r['method']=='table/mean3'}
        b = {r['token']:r for r in rows if r['k']==k and r['method']=='table_camera/mean3'}
        if a:
            session_deltas = defaultdict(list)
            for t in a:
                session_deltas[a[t]['session_id']].append(b[t]['gain_pp']-a[t]['gain_pp'])
            delta_means = np.asarray([np.mean(v) for _,v in sorted(session_deltas.items())])
            rng = np.random.default_rng(data.CONFIG['seed'])
            boot = delta_means[rng.integers(0,len(delta_means),(data.CONFIG['bootstrap_reps'],len(delta_means)))].mean(axis=1)
            comparisons[str(k)] = dict(scenes=len(a),
                camera_minus_table_gain_micro_pp=float(np.mean([b[t]['gain_pp']-a[t]['gain_pp'] for t in a])),
                camera_minus_table_gain_macro_pp=float(delta_means.mean()),
                camera_minus_table_cluster_ci95_pp=np.quantile(boot,[.025,.975]).tolist(),
                camera_better_scenes=sum(b[t]['gain_pp']>a[t]['gain_pp']+1e-8 for t in a),
                camera_worse_scenes=sum(b[t]['gain_pp']<a[t]['gain_pp']-1e-8 for t in a))
    performance = {}
    for k in data.KS:
        for mode in data.MODES:
            group = [r for r in results.values() if r['k']==k and r['mode']==mode]
            times = [r['seconds'] for r in group if 'seconds' in r and not r.get('cold_call',False)]
            tokens = [r['input_tokens'] for r in group if 'input_tokens' in r]
            performance[f'{mode}/k{k}'] = dict(requests=len(group),
                max_input_tokens=max(tokens) if tokens else None,
                warm_latency_p50_seconds=float(np.quantile(times,.5)) if times else None,
                warm_latency_p95_seconds=float(np.quantile(times,.95)) if times else None)
    summary = dict(status='awaiting_top64_labels' if missing else 'complete',
        scenes=len(identity['records']),requests=len(results),primary=data.CONFIG['primary'],methods=metrics,
        paired_image_ablation=comparisons,order_sensitivity=sensitivity,performance=performance,missing_pool_labels=missing,
        decision_commit_sha256=sha(out/'decision_commit.json'),external_labels_sha256=None if external_path is None else sha(external_path),
        note=data.NOTE)
    if rows:
        with (out/'per_scene.csv.tmp').open('w',encoding='utf-8',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (out/'per_scene.csv.tmp').replace(out/'per_scene.csv')
    write(out/'report.json',summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('plan','worker','merge'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--replicas',type=int,default=2)
    p.add_argument('--index',type=int)
    p.add_argument('--cohort',choices=('all','balanced64'),default='all')
    p.add_argument('--pilot',action='store_true')
    a = p.parse_args(); root = a.root.expanduser().resolve()
    name = os.environ['DREX_TABLE_NAME']
    if not re.fullmatch(r'table_[A-Za-z0-9_-]{1,80}',name): raise ValueError('Use a separate table_* run name')
    out = root/'runs'/name
    package = Path(os.environ.get('DREX_TABLE_DATA',str(root/'data/navsim_train_v1/package'))).expanduser().resolve()
    try:
        if a.stage=='plan': result = plan(root,package,out,a.replicas,a.cohort)
        else:
            identity,frozen = context(out)
            if identity['replicas']!=a.replicas or identity['cohort']!=a.cohort:
                raise RuntimeError('Frozen cohort/replica count changed; choose a new run name')
            if sha(root/'requirements.lock.txt')!=identity['environment_lock_sha256'] or read(root/'model_plan.json')!=identity['model_plan']:
                raise RuntimeError('Frozen runtime/model changed')
            if a.stage=='merge':
                external = os.environ.get('DREX_TABLE_LABELS')
                if not external and (out/'label_source.json').exists():
                    source = read(out/'label_source.json')
                    if (source['request_sha256']!=sha(out/'label_request.json')
                            or sha(Path(source['labels']))!=source['sha256']):
                        raise RuntimeError('Retrieved label source changed')
                    external = source['labels']
                result = report(out,Path(external) if external else None)
            else:
                if a.index is None or not 0<=a.index<a.replicas: raise ValueError('Invalid worker index')
                tokens = set(frozen['assignments'][a.index])
                records = [r for r in identity['records'] if r['token'] in tokens]
                w = out/'shards'/f'{a.index:02d}'
                if (w/'commit.json').exists():
                    commit = read(w/'commit.json')
                    if commit['plan_sha256']!=sha(out/'plan.json'): raise RuntimeError('Worker plan changed')
                    # Revalidate inputs and results without loading model or any outcomes.
                    result = infer_tasks(out,frozen,records,w,lambda _: (_ for _ in ()).throw(AssertionError('Unexpected inference')),max_new=0)
                else: result = run_gpu(root,out,frozen,records,w,a.pilot)
                if result['status']=='partial' and not a.pilot:
                    raise RuntimeError('Worker time budget reached; resume submit before merge')
        print(json.dumps(result,indent=2),flush=True)
    except Exception as exc:
        target = out/'shards'/f'{a.index:02d}' if a.index is not None else out
        write(target/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__': main()
