"""Frozen 3x2 table experiment, with one GPU-bound process per Slurm rank."""
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
import table_factorial_data as data
from night import load_result, save_result, validate_probabilities
from runtime import read, write, environment, remaining_bytes
from transfer import freeze, sha

SOURCES = tuple(dict.fromkeys(('table_factorial.py', 'table_factorial_data.py', 'run_table_factorial.sh',
    'table_group.py', 'audit_table_cases.py', *previous.SOURCES)))
SOURCE_FILES = ('identity.json', 'plan.json', 'decision_commit.json', 'decisions.json',
                'report.json', 'label_source.json', 'label_request.json')


def prepare(root, source, out, count):
    identity, plan = previous.context(source)
    report = read(source/'report.json')
    if report['status'] != 'complete' or report['missing_pool_labels']:
        raise RuntimeError('Completed original table experiment required')
    if identity['cohort'] != 'all' or not 1 <= count <= len(identity['records']):
        raise ValueError('Use the full original training cohort and valid replica count')
    if read(root/'model_plan.json') != identity['model_plan'] or sha(root/'requirements.lock.txt') != identity['environment_lock_sha256']:
        raise RuntimeError('Original weights/runtime lock required')
    pointer = read(source/'label_source.json')
    if sha(Path(pointer['labels'])) != pointer['sha256'] or pointer['sha256'] != report['external_labels_sha256']:
        raise RuntimeError('Original label file changed')
    value = dict(config=data.CONFIG, replicas=count, source=str(source),
        records=identity['records'], original_files={n:sha(source/n) for n in SOURCE_FILES},
        original_labels_sha256=pointer['sha256'], model_plan=identity['model_plan'],
        environment_lock_sha256=identity['environment_lock_sha256'],
        sources={n:sha(Path(__file__).with_name(n)) for n in SOURCES}, note=data.NOTE)
    freeze(out/'identity.json', value)
    # Validate every unchanged input and image, without loading outcome arrays.
    for r in identity['records']:
        sample, photo = previous.scene(source, plan, r['token'])
        data.blank_board(photo)
        for mode in data.MODES:
            data.payload(r['token'], sample, mode, data.ORDERS[0], photo)
    freeze(out/'plan.json', dict(identity_sha256=sha(out/'identity.json'),
        assignments=[[r['token'] for r in identity['records'][i::count]] for i in range(count)]))
    return dict(status='planned', scenes=len(identity['records']),
        sessions=len({r['session_id'] for r in identity['records']}),
        requests=len(identity['records'])*18, conditions=list(data.MODES), replicas=count,
        labels_loaded_for_planning=False, fresh_holdout_processed=0, note=data.NOTE)


def context(out):
    identity, plan = read(out/'identity.json'), read(out/'plan.json')
    if identity['config'] != data.CONFIG or plan['identity_sha256'] != sha(out/'identity.json'):
        raise RuntimeError('Frozen factorial identity changed')
    for n, digest in identity['sources'].items():
        if sha(Path(__file__).with_name(n)) != digest:
            raise RuntimeError('Frozen factorial code changed: '+n+'; use a new run name')
    source = Path(identity['source'])
    for n, digest in identity['original_files'].items():
        if sha(source/n) != digest:
            raise RuntimeError('Original experiment changed: '+n)
    original_identity, original_plan = previous.context(source)
    if identity['records'] != original_identity['records']:
        raise RuntimeError('Original cohort changed')
    expected = [[r['token'] for r in identity['records'][i::identity['replicas']]] for i in range(identity['replicas'])]
    if plan['assignments'] != expected:
        raise RuntimeError('Frozen worker assignment changed')
    return identity, plan, source, original_plan


def validate_result(r, token, mode, order, permutation):
    if (r['token'], r['mode'], r['ordering'], r['pool_positions'], r['k']) != (token, mode, order, permutation, 64):
        raise RuntimeError('Factorial result alignment changed')
    validate_probabilities(r['probabilities'], 64)


def infer_tasks(out, plan, source, source_plan, records, worker, infer, max_new=None, budget_seconds=7*3600):
    tasks = data.tasks(records); expected = {data.task_name(*t) for t in tasks}
    if {p.name for p in (worker/'results').glob('*.json')} - expected:
        raise RuntimeError('Unexpected factorial results')
    started = time.monotonic(); new = completed = 0; last = None
    for token, mode, order in tasks:
        if token != last:
            sample, photo = previous.scene(source, source_plan, token); last = token
        path = worker/'results'/data.task_name(token, mode, order)
        if not path.exists() and ((max_new is not None and new >= max_new) or time.monotonic()-started >= budget_seconds):
            break
        row, permutation, digest = data.payload(token, sample, mode, order, photo)
        if path.exists():
            result = load_result(path, digest)
            validate_result(result, token, mode, order, permutation)
        else:
            result = dict(status='ok', token=token, mode=mode, ordering=order, k=64,
                pool_positions=permutation, request_sha256=digest)
            try:
                result.update(infer(row))
                validate_result(result, token, mode, order, permutation)
            except Exception as exc:
                result.update(status='technical_failure', error=repr(exc), traceback=traceback.format_exc())
                save_result(path, result)
                write(worker/'inference_status.json', dict(status='failed', request=str(path), outcomes_read=False))
                raise
            save_result(path, result); new += 1
            print(f'FACTORIAL {completed+1}/{len(tasks)} {token} {mode}/{order}', flush=True)
        completed += 1
    result = dict(status='choices_frozen' if completed == len(tasks) else 'partial',
        completed_requests=completed, expected_requests=len(tasks), new_requests=new,
        seconds=time.monotonic()-started, outcomes_read=False)
    if completed == len(tasks):
        freeze(worker/'commit.json', dict(plan_sha256=sha(out/'plan.json'),
            environment_sha256=sha(worker/'environment.json'),
            results={name:sha(worker/'results'/name) for name in sorted(expected)}))
    write(worker/'inference_status.json', result)
    return result


def run_gpu(root, out, plan, source, source_plan, records, worker, pilot):
    import torch
    from drex_max.model import DecisionModel
    envroot = worker/'runtime'; envroot.mkdir(parents=True, exist_ok=True)
    lock = envroot/'requirements.lock.txt'
    if lock.exists() and sha(lock) != sha(root/'requirements.lock.txt'):
        raise RuntimeError('Worker runtime lock changed')
    shutil.copyfile(root/'requirements.lock.txt', lock)
    env = environment(envroot, True)
    freeze(worker/'environment.json', {k:env[k] for k in
        ('python', 'torch', 'cuda_build', 'transformers', 'gpu', 'vram_bytes')})
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Each rank must see exactly one GPU')
    target = root/'models/drex-max-v1'
    if remaining_bytes(target, read(root/'model_plan.json')['files']):
        raise RuntimeError('Missing/changed model sizes; no implicit download')
    config = read(target/'decision_config.json')
    if config.get('format_version') != 1 or config.get('attention_mode', 'noncausal_full_attention') != 'noncausal_full_attention':
        raise RuntimeError('Unexpected Drex attention format')
    torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
    model = DecisionModel(target, device='cuda:0', max_image_pixels=data.CONFIG['max_image_pixels'])
    if model.readout.weight.dtype != torch.bfloat16:
        raise RuntimeError('Expected BF16')
    torch.cuda.synchronize(); load_seconds = time.monotonic()-start; calls = 0

    def infer(row):
        nonlocal calls
        batch = model.prepare([row], max_length=data.CONFIG['max_length'])
        tokens = int(batch.inputs['input_ids'].shape[1]); n = len(row.get('images', []))
        grid = batch.inputs.get('image_grid_thw')
        if (tokens >= data.CONFIG['max_length'] or (0 if grid is None else int(grid.shape[0])) != n
                or ('pixel_values' in batch.inputs) != (n > 0)):
            raise RuntimeError('Truncated context or wrong image condition')
        del batch
        torch.cuda.synchronize(); start = time.monotonic()
        p = model.predict([row], batch_size=1, max_length=data.CONFIG['max_length'])[0]
        torch.cuda.synchronize(); calls += 1
        return dict(probabilities=[float(v) for v in p], seconds=time.monotonic()-start,
            input_tokens=tokens, has_image_pixels=n > 0, cold_call=calls == 1)

    result = infer_tasks(out, plan, source, source_plan, records, worker, infer, max_new=18 if pilot else None)
    result.update(load_seconds=load_seconds, peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    write(worker/'inference_status.json', result)
    return result


def collect(out):
    identity, plan, source, source_plan = context(out)
    decisions = []; profiles = []; all_results = []
    workers = {}
    for i, tokens in enumerate(plan['assignments']):
        w = out/'shards'/f'{i:02d}'; commit = read(w/'commit.json')
        records = [r for r in identity['records'] if r['token'] in set(tokens)]
        expected = {data.task_name(*t) for t in data.tasks(records)}
        if (commit['plan_sha256'] != sha(out/'plan.json') or set(commit['results']) != expected
                or {p.name for p in (w/'results').glob('*.json')} != expected
                or sha(w/'environment.json') != commit['environment_sha256']):
            raise RuntimeError('Incomplete/changed worker; no outcome lookup allowed')
        profiles.append(read(w/'environment.json')); workers[str(i)] = sha(w/'commit.json')
        for r in records:
            token = r['token']; sample, photo = previous.scene(source, source_plan, token)
            methods = {}
            for mode in data.MODES:
                aligned = []
                for order in data.ORDERS:
                    _, permutation, digest = data.payload(token, sample, mode, order, photo)
                    name = data.task_name(token, mode, order); path = w/'results'/name
                    if sha(path) != commit['results'][name]:
                        raise RuntimeError('Changed worker result')
                    value = load_result(path, digest)
                    validate_result(value, token, mode, order, permutation)
                    p = np.empty(64); p[permutation] = value['probabilities']; aligned.append(p)
                    methods[mode+'/'+order] = original.pick(p, sample, 64)
                    all_results.append(value)
                methods[mode+'/mean3'] = original.pick(np.mean(aligned, axis=0), sample, 64)
            decisions.append(dict(metadata=r, methods=methods))
    if any(p != profiles[0] for p in profiles[1:]):
        raise RuntimeError('Ranks used different environments')
    decisions.sort(key=lambda d:d['metadata']['token'])
    freeze(out/'decisions.json', decisions)
    freeze(out/'decision_commit.json', dict(plan_sha256=sha(out/'plan.json'),
        decisions_sha256=sha(out/'decisions.json'), workers=workers))
    return identity, decisions, all_results


def summarize(rows, value='gain_pp'):
    by_session = defaultdict(list)
    for r in rows:
        by_session[r['session_id']].append(r[value])
    means = np.array([np.mean(v) for _, v in sorted(by_session.items())])
    rng = np.random.default_rng(data.CONFIG['seed'])
    bootstrap = means[rng.integers(len(means), size=(data.CONFIG['bootstrap_reps'], len(means)))].mean(axis=1)
    return dict(scenes=len(rows), sessions=len(means), gain_micro_pp=float(np.mean([r[value] for r in rows])),
        gain_macro_pp=float(means.mean()), cluster_ci95_pp=np.quantile(bootstrap, [.025, .975]).tolist(),
        gain_by_session={s:float(np.mean(v)) for s,v in by_session.items()})


def contrast(rows, a, b):
    aa = {r['token']:r for r in rows if r['method'] == a}
    bb = {r['token']:r for r in rows if r['method'] == b}
    if not aa or aa.keys() != bb.keys():
        raise RuntimeError('Paired contrast has missing scenes')
    values = [dict(token=t, session_id=aa[t]['session_id'], gain_pp=aa[t]['gain_pp']-bb[t]['gain_pp']) for t in aa]
    return dict(summarize(values), a=a, b=b,
        choice_changes=sum(aa[t]['candidate_local'] != bb[t]['candidate_local'] for t in aa),
        better_scenes=sum(r['gain_pp'] > 1e-8 for r in values), worse_scenes=sum(r['gain_pp'] < -1e-8 for r in values))


def report(out):
    identity, decisions, results = collect(out)  # Entire 3x2 cohort commits BEFORE labels.
    import audit_table_cases as audit
    _, _, _, _, _, labels, samples, _ = audit.validate_source(Path(identity['source']))
    rows = []; safety = [1+base.METRICS.index(m) for m in base.SAFETY]
    for d in decisions:
        token = d['metadata']['token']; sample = samples[token]; group = labels[token]
        baseline = group[sample['baseline_local']]; methods = dict(d['methods'])
        methods['baseline'] = sample['baseline_local']
        maximum = max(group[c['local']][0] for c in sample['candidates'])
        methods['oracle_top64'] = (sample['baseline_local'] if baseline[0] == maximum else
            next(c['local'] for c in sample['candidates'] if group[c['local']][0] == maximum))
        for method,j in methods.items():
            delta = group[j]-baseline
            c = next(c for c in sample['candidates'] if c['local'] == j)
            rows.append(dict(token=token, session_id=d['metadata']['session_id'], method=method,
                candidate_local=j, candidate_vocab=c['vocab'], teacher_rank=1+sample['candidates'].index(c),
                raw_pdm=float(group[j][0]), gain_pp=float(100*delta[0]),
                intervention=j != sample['baseline_local'], pdm_decreased=bool(delta[0] < -1e-8),
                relative_component_harm=bool(np.any(delta[safety] < -1e-8)),
                **{'delta_'+m:float(delta[i+1]) for i,m in enumerate(base.METRICS)}))
    metrics = {}
    for method in sorted({r['method'] for r in rows}):
        group = [r for r in rows if r['method'] == method]
        metrics[method] = dict(summarize(group),
            **{key:sum(r[key] for r in group) for key in ('intervention', 'pdm_decreased', 'relative_component_harm')},
            component_degradations={m:sum(r['delta_'+m] < -1e-8 for r in group) for m in base.METRICS})
    contrasts = {}
    for score in data.SCORES:
        for a,b in (('road','blank'), ('blank','none'), ('road','none')):
            contrasts[a+'_minus_'+b+'/'+score+'/mean3'] = contrast(rows, a+'_'+score+'/mean3', b+'_'+score+'/mean3')
    for image in data.IMAGES:
        contrasts['hidden_minus_full/'+image+'/mean3'] = contrast(rows, image+'_hidden/mean3', image+'_full/mean3')
    interactions = []
    lookup = {(r['token'],r['method']):r for r in rows}
    for d in decisions:
        t = d['metadata']['token']
        effect = lambda score: lookup[t,'road_'+score+'/mean3']['gain_pp']-lookup[t,'blank_'+score+'/mean3']['gain_pp']
        interactions.append(dict(session_id=d['metadata']['session_id'], gain_pp=effect('hidden')-effect('full')))
    performance = {}
    order_sensitivity = {}
    for mode in data.MODES:
        group = [r for r in results if r['mode'] == mode]
        times = [r['seconds'] for r in group if not r.get('cold_call', False)]
        performance[mode] = dict(requests=len(group), max_input_tokens=max(r['input_tokens'] for r in group),
            warm_p50_seconds=float(np.quantile(times,.5)) if times else None,
            warm_p95_seconds=float(np.quantile(times,.95)) if times else None)
        for order in data.ORDERS[1:]:
            comparisons = [(lookup[d['metadata']['token'],mode+'/canonical'],
                            lookup[d['metadata']['token'],mode+'/'+order]) for d in decisions]
            order_sensitivity[mode+'/'+order] = dict(
                comparisons=len(comparisons), choice_flips=sum(a['candidate_local'] != b['candidate_local'] for a,b in comparisons),
                pdm_worse_than_canonical=sum(b['raw_pdm'] < a['raw_pdm']-1e-8 for a,b in comparisons),
                component_degradations_vs_canonical={m:sum(b['delta_'+m] < a['delta_'+m]-1e-8 for a,b in comparisons)
                    for m in base.METRICS})
    with (out/'per_scene.csv.tmp').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'per_scene.csv.tmp').replace(out/'per_scene.csv')
    value = dict(status='complete', scenes=len(decisions), requests=len(results), primary=data.CONFIG['primary'],
        methods=metrics, paired_contrasts=contrasts, image_score_interaction=summarize(interactions),
        interaction_definition='(road - blank) hidden - (road - blank) full, in PDM percentage points',
        relative_component_harm_metrics=list(base.SAFETY), order_sensitivity=order_sensitivity,
        performance=performance, decision_commit_sha256=sha(out/'decision_commit.json'),
        fresh_holdout_processed=0, note=data.NOTE)
    write(out/'report.json', value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('plan', 'bind', 'worker-pilot', 'worker', 'report'))
    p.add_argument('--root', type=Path, required=True); p.add_argument('--replicas', type=int, default=4)
    a = p.parse_args(); root = a.root.expanduser().resolve()
    if a.replicas not in (1,2,4,8):
        raise ValueError('Use 1/2/4/8 GPUs')
    name = os.environ.get('DREX_FACTORIAL_NAME', 'table_factorial_gpu4_v1')
    source_name = os.environ.get('DREX_FACTORIAL_SOURCE', 'table_topk_night_gpu8_v1')
    if not re.fullmatch(r'table_factorial_[A-Za-z0-9_-]{1,64}', name) or not re.fullmatch(r'table_[A-Za-z0-9_-]{1,80}', source_name):
        raise ValueError('Invalid run name')
    out = root/'runs'/name; out.mkdir(parents=True, exist_ok=True)
    import fcntl
    from table_group import gpu_binding
    phase = a.stage; worker = None
    if phase in ('worker', 'worker-pilot', 'bind'):
        rank = gpu_binding(out, a.replicas, phase)
        if phase == 'bind': return
        worker = out/'shards'/f'{rank:02d}'; worker.mkdir(parents=True, exist_ok=True)
    target = worker or out
    with (target/'.execution.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another process owns this stage/shard')
        try:
            if phase == 'plan':
                result = prepare(root, root/'runs'/source_name, out, a.replicas)
            else:
                identity, plan, source, source_plan = context(out)
                if identity['replicas'] != a.replicas or source.name != source_name:
                    raise RuntimeError('Frozen source/replica count changed')
                if sha(root/'requirements.lock.txt') != identity['environment_lock_sha256'] or read(root/'model_plan.json') != identity['model_plan']:
                    raise RuntimeError('Frozen runtime changed')
                if phase == 'report': result = report(out)
                else:
                    tokens = set(plan['assignments'][rank])
                    records = [r for r in identity['records'] if r['token'] in tokens]
                    if (worker/'commit.json').exists():
                        result = infer_tasks(out, plan, source, source_plan, records, worker,
                            lambda _: (_ for _ in ()).throw(AssertionError('Unexpected inference')), max_new=0)
                    else: result = run_gpu(root, out, plan, source, source_plan, records, worker, phase == 'worker-pilot')
                    if result['status'] == 'partial' and phase != 'worker-pilot':
                        raise RuntimeError('Worker incomplete: resume same GPU count/name before reporting')
            print(json.dumps(result, indent=2), flush=True)
        except Exception as exc:
            write(target/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__': main()
