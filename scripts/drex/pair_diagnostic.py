"""48 outcome-selected pair probes. Plan is outcome-informed, inference is blind."""
import argparse
import io
import json
import os
from pathlib import Path
import re
import shutil
import traceback

import numpy as np

import pair_diagnostic_data as view
import overlay_v2_pilot as source
import real_data as base
import real_pilot as scoring
from night import load_result, save_result, validate_probabilities
from runtime import read, write, environment, remaining_bytes
from transfer import sha

SPECS = [
    dict(pair_id='case01', token='0a3804bea43d5ea8', kind='repair', metric='drivable_area_compliance'),
    dict(pair_id='case02', token='a75d642049ef528d', kind='repair', metric='drivable_area_compliance'),
    dict(pair_id='case03', token='368c895ca59e5537', kind='repair', metric='drivable_area_compliance'),
    dict(pair_id='case04', token='85e8c8e8f6c85157', kind='repair', metric='drivable_area_compliance'),
    dict(pair_id='case05', token='6394ed413b685026', kind='repair', metric='no_at_fault_collisions'),
    dict(pair_id='case06', token='5e0b5d47dcb5593c', kind='regression', metric='lane_keeping'),
]
SOURCES = ('pair_diagnostic.py', 'pair_diagnostic_data.py', 'vehicle_profile_astra.json',
           'run_pair_diagnostic.sh', 'overlay_v2_data.py', 'overlay_data.py', 'overlay_v2_pilot.py',
           'overlay_pilot.py', 'audit_overlay_projection.py', 'audit_real.py', 'real_parallel.py',
           'real_data.py', 'real_pilot.py', 'night.py', 'runtime.py', 'transfer.py')


def file_name(task):
    return task['request_id']+'.json'


def verify_source(package, source_out):
    """Verify the complete old cohort before outcome-informed case preparation."""
    plan = source.check_plan(source_out)
    commit = read(source_out/'decision_commit.json')
    if (commit['plan_sha256'] != sha(source_out/'plan.json') or
            commit['decisions_sha256'] != sha(source_out/'decisions.json')):
        raise RuntimeError('Source decisions changed')
    identity = read(source_out/'base/identity.json')
    if sha(package/'package_manifest.json') != identity['package_manifest_sha256']:
        raise RuntimeError('Source package changed')
    manifest, _, scenes, vocab = base.package_metadata(package)
    records = plan['records']
    expected = {source.name(r['token'], mode, order)
                for r in records for mode, order in source.view.TASKS}
    if (set(commit['results']) != expected or
            {p.name for p in (source_out/'results').glob('*.json')} != expected):
        raise RuntimeError('Complete source cohort required, not partial source results')
    results = {}
    for name, digest in commit['results'].items():
        path = source_out/'results'/name
        if sha(path) != digest:
            raise RuntimeError('Source result changed: '+name)
        item = load_result(path)
        if source.name(item['token'], item['mode'], item['ordering']) != name:
            raise RuntimeError('Source result scene mismatch')
        results[name] = item
    samples = source.samples_for(package, manifest, records, vocab)
    decisions = source.build_decisions(records, samples, results)
    if decisions != read(source_out/'decisions.json'):
        raise RuntimeError('Source decisions disagree with frozen probabilities')
    if read(source_out/'report.json')['status'] != 'complete':
        raise RuntimeError('Source evaluation has not completed')
    return manifest, scenes, vocab, samples, {d['metadata']['token']: d for d in decisions}


def select_pairs(package, manifest, samples, decisions, specs):
    pairs, expected = [], []
    metric_keys = ['pdm_score', *base.METRICS]
    for spec in specs:
        token = spec['token']
        if token not in decisions:
            raise ValueError('Known diagnostic scene absent from source: '+token)
        sample, decision = samples[token], decisions[token]
        # Intentional outcome-informed diagnostic selection; never inference inputs.
        raw = scoring.load_labels(package, token, manifest, sample)
        baseline = sample['pool_local'].index(sample['baseline_local'])
        focus = metric_keys.index(spec['metric'])
        if spec['kind'] == 'repair':
            other = int(np.argmax(raw[:, 0]))
            if not (raw[other, 0]-raw[baseline, 0] >= .005 and raw[other, focus] > raw[baseline, focus]):
                raise RuntimeError('Expected known repair is absent: '+token)
            better = 1
        elif spec['kind'] == 'regression':
            other = sample['pool_local'].index(decision['methods']['overlay_mean3']['candidate_local'])
            if not (raw[baseline, 0]-raw[other, 0] >= .005 and raw[other, focus] < raw[baseline, focus]):
                raise RuntimeError('Expected known regression is absent: '+token)
            better = 0
        else:
            raise ValueError('Unknown case type')
        if other == baseline:
            raise RuntimeError('Diagnostic needs two different physical candidates')
        pair = dict(pair_id=spec['pair_id'], token=token,
                    local_ids=[sample['pool_local'][i] for i in (baseline, other)],
                    vocab_ids=[sample['pool_vocab'][i] for i in (baseline, other)])
        pairs.append(pair)
        expected.append(dict(pair_id=spec['pair_id'], token=token, kind=spec['kind'],
            focus_metric=spec['metric'], better_member=better, baseline_member=0,
            raw_scores=raw[[baseline, other]].tolist(),
            teacher_ranks=[baseline+1, other+1], metric_keys=metric_keys))
    if len({p['pair_id'] for p in pairs}) != len(pairs) or len({p['token'] for p in pairs}) != len(pairs):
        raise ValueError('Duplicate diagnostic cases')
    return pairs, expected


def tasks_for(pairs):
    tasks = []
    for pair in pairs:
        for representation in view.REPRESENTATIONS:
            for mode in view.MODES:
                for ordering in view.ORDERS:
                    task = dict(pair_id=pair['pair_id'], representation=representation, mode=mode, ordering=ordering)
                    task['request_id'] = '-'.join(task.values())
                    tasks.append(task)
    return sorted(tasks, key=lambda t: base.digest([base.SEED, t['request_id'], 'assignment']))


def freeze_image(path, image):
    buffer = io.BytesIO(); image.save(buffer, format='PNG'); content = buffer.getvalue()
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError('Changed diagnostic preview: '+path.name)
    if not path.exists():
        tmp = path.with_suffix('.png.tmp'); tmp.write_bytes(content); tmp.replace(path)
    return sha(path)


def freeze_plan(root, package, source_out, out, replicas, specs=None):
    if replicas not in (1, 2, 4, 8):
        raise ValueError('Use 1,2,4 or 8 replicas')
    manifest, scenes, vocab, samples, decisions = verify_source(package, source_out)
    pairs, expected = select_pairs(package, manifest, samples, decisions, SPECS if specs is None else specs)
    tasks = tasks_for(pairs)
    if replicas > len(tasks):
        raise ValueError('More workers than requests')
    pair_map = {p['pair_id']: p for p in pairs}
    out.mkdir(parents=True, exist_ok=True)
    base.freeze(out/'pair_inputs.json', pairs)
    base.freeze(out/'expected.json', expected)
    previews = {}
    for task in tasks:
        pair = pair_map[task['pair_id']]; token = pair['token']
        photo = base.camera_paths(package, scenes[token], manifest)[4]
        if sha(photo) != manifest['files'][photo.relative_to(package).as_posix()]['sha256']:
            raise RuntimeError('Current photo changed')
        row, positions, digest = view.payload(package, scenes[token], manifest, samples[token], pair,
                                             task['representation'], task['mode'], task['ordering'])
        task.update(request_sha256=digest, pair_positions=positions)
        if task['mode'] == 'road' and task['ordering'] == 'canonical':
            names, _ = view.names_and_order(pair, 'canonical')
            for j, member in enumerate(positions):
                name = f'preview-{pair["pair_id"]}-{task["representation"]}-{names[member]}.png'
                previews[name] = freeze_image(out/name, row['images'][j+1])
            name = f'preview-{pair["pair_id"]}-reference.png'
            previews[name] = freeze_image(out/name, row['images'][0])
    base.freeze(out/'requests.json', tasks)
    plan = dict(config=view.CONFIG, replicas=replicas,
        assignments=[[t['request_id'] for t in tasks[i::replicas]] for i in range(replicas)],
        files={name: sha(out/name) for name in ('pair_inputs.json', 'expected.json', 'requests.json')},
        previews=previews, package_manifest_sha256=sha(package/'package_manifest.json'),
        model_plan_sha256=sha(root/'model_plan.json'), environment_lock_sha256=sha(root/'requirements.lock.txt'),
        source_run=str(source_out.resolve()), source_plan_sha256=sha(source_out/'plan.json'),
        source_decision_commit_sha256=sha(source_out/'decision_commit.json'),
        sources={n: sha(Path(__file__).with_name(n)) for n in SOURCES},
        selection_uses_known_outcomes=True, expected_labels_in_model_inputs=False, note=view.NOTE)
    base.freeze(out/'plan.json', plan)
    summary = dict(status='planned', cases=len(pairs), requests=len(tasks), replicas=replicas,
                   selection_uses_known_outcomes=True, expected_labels_in_model_inputs=False,
                   vehicle_dimensions_m=view.vehicle(), output=str(out), note=view.NOTE)
    base.freeze(out/'plan_summary.json', summary)
    return summary


def checked_plan(root, package, out):
    plan = read(out/'plan.json')
    for n, digest in plan['sources'].items():
        if sha(Path(__file__).with_name(n)) != digest:
            raise RuntimeError('Diagnostic code changed; use a new pair_diag_* name')
    for n, digest in {**plan['files'], **plan['previews']}.items():
        if sha(out/n) != digest:
            raise RuntimeError('Diagnostic frozen input changed: '+n)
    for path, digest in ((package/'package_manifest.json', plan['package_manifest_sha256']),
                         (root/'model_plan.json', plan['model_plan_sha256']),
                         (root/'requirements.lock.txt', plan['environment_lock_sha256'])):
        if sha(path) != digest:
            raise RuntimeError('Diagnostic package/model/environment identity changed')
    source_out = Path(plan['source_run'])
    if (sha(source_out/'plan.json') != plan['source_plan_sha256'] or
            sha(source_out/'decision_commit.json') != plan['source_decision_commit_sha256']):
        raise RuntimeError('Diagnostic source commitment changed')
    return plan


def inputs_for(package, out):
    # Do not deserialize expected.json on inference workers.
    manifest, _, scenes, vocab = base.package_metadata(package)
    pairs = {p['pair_id']: p for p in read(out/'pair_inputs.json')}
    samples = {p['token']: base.load_inputs(package, p['token'], manifest, vocab) for p in pairs.values()}
    return manifest, scenes, samples, pairs


def infer_tasks(package, worker_out, tasks, manifest, scenes, samples, pairs, infer):
    expected = {file_name(t) for t in tasks}
    if {p.name for p in (worker_out/'results').glob('*.json')}-expected:
        raise RuntimeError('Unexpected worker result files')
    committed = {}
    for task in tasks:
        pair = pairs[task['pair_id']]; token = pair['token']
        row, positions, digest = view.payload(package, scenes[token], manifest, samples[token], pair,
                                             task['representation'], task['mode'], task['ordering'])
        if digest != task['request_sha256'] or positions != task['pair_positions']:
            raise RuntimeError('Diagnostic request/pixels changed')
        path = worker_out/'results'/file_name(task)
        if path.exists():
            item = load_result(path, digest)
            if item['request_id'] != task['request_id'] or item['pair_positions'] != positions:
                raise RuntimeError('Cached diagnostic identity changed')
        else:
            item = dict(task, k=2, status='ok')
            try:
                measured = infer(row)
                validate_probabilities(measured['probabilities'], 2)
                item.update(measured)
            except Exception as exc:
                item.update(status='technical_failure', error=repr(exc))
                save_result(path, item); raise
            save_result(path, item)
            print('PAIR '+task['request_id'], flush=True)
        committed[path.name] = sha(path)
    status = dict(status='complete', completed_requests=len(committed), expected_requests=len(expected),
                  expected_labels_deserialized=False)
    base.freeze(worker_out/'worker_commit.json', dict(results=committed))
    write(worker_out/'inference_status.json', status)
    return status


def gpu_worker(root, package, worker_out, tasks, inputs):
    import time
    import torch
    from drex_max.model import DecisionModel
    envroot = worker_out/'runtime_check'; envroot.mkdir(parents=True, exist_ok=True)
    lock = envroot/'requirements.lock.txt'
    if lock.exists() and sha(lock) != sha(root/'requirements.lock.txt'):
        raise RuntimeError('Environment lock changed')
    if not lock.exists():
        shutil.copyfile(root/'requirements.lock.txt', lock)
    env = environment(envroot, True)
    base.freeze(worker_out/'environment.json', {k: env[k] for k in ('python', 'torch', 'cuda_build', 'transformers', 'gpu', 'vram_bytes')})
    target = root/'models/drex-max-v1'
    if remaining_bytes(target, read(root/'model_plan.json')['files']):
        raise RuntimeError('Missing model files')
    config = read(target/'decision_config.json')
    if config.get('format_version') != 1 or config.get('attention_mode', 'noncausal_full_attention') != 'noncausal_full_attention':
        raise RuntimeError('Unexpected model protocol')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Each worker requires one GPU')
    model = DecisionModel(target, device='cuda:0', max_image_pixels=view.CONFIG['max_image_pixels'])
    if model.readout.weight.dtype != torch.bfloat16:
        raise RuntimeError('Expected BF16')
    def infer(row):
        batch = model.prepare([row], max_length=view.CONFIG['max_length'])
        grid = batch.inputs.get('image_grid_thw'); tokens = int(batch.inputs['input_ids'].shape[1])
        if grid is None or grid.shape[0] != 3 or tokens >= view.CONFIG['max_length']:
            raise RuntimeError('Unexpected image count/context truncation')
        del batch
        torch.cuda.synchronize(); started = time.monotonic()
        p = model.predict([row], batch_size=1, max_length=view.CONFIG['max_length'])[0]
        torch.cuda.synchronize()
        return dict(probabilities=[float(x) for x in p], input_tokens=tokens, seconds=time.monotonic()-started)
    return infer_tasks(package, worker_out, tasks, *inputs, infer)


def decisions_from_results(tasks, pairs, results):
    if set(results) != {file_name(t) for t in tasks}:
        raise RuntimeError('All diagnostic conditions must complete before scoring')
    decisions = []
    for pair in pairs.values():
        for representation in view.REPRESENTATIONS:
            for mode in view.MODES:
                vectors = []
                for ordering in view.ORDERS:
                    task = next(t for t in tasks if (t['pair_id'], t['representation'], t['mode'], t['ordering']) ==
                                (pair['pair_id'], representation, mode, ordering))
                    item = results[file_name(task)]
                    validate_probabilities(item['probabilities'], 2)
                    if item['pair_positions'] != task['pair_positions']:
                        raise RuntimeError('Diagnostic option alignment changed')
                    aligned = np.empty(2); aligned[item['pair_positions']] = item['probabilities']; vectors.append(aligned)
                choose = lambda p: 0 if p[0] >= p[1] else 1
                selected = [choose(v) for v in vectors]
                mean = np.mean(vectors, axis=0)
                decisions.append(dict(pair_id=pair['pair_id'], representation=representation, mode=mode,
                    order_members=selected, order_ties=[bool(p[0] == p[1]) for p in vectors],
                    mean_probabilities=mean.tolist(), mean_member=choose(mean),
                    consistent_switch_member=1 if selected == [1, 1] else 0))
    return decisions


def score_committed(out, tasks, pairs):
    commit = read(out/'decision_commit.json')
    if (sha(out/'plan.json') != commit['plan_sha256'] or
            sha(out/'decisions.json') != commit['decisions_sha256'] or
            sha(out/'expected.json') != commit['expected_sha256']):
        raise RuntimeError('Diagnostic decision commitment changed')
    if set(commit['results']) != {file_name(t) for t in tasks}:
        raise RuntimeError('Incomplete diagnostic result commitment')
    results = {}
    for name, digest in commit['results'].items():
        path = out/'results'/name
        if sha(path) != digest:
            raise RuntimeError('Diagnostic result changed')
        task = next(t for t in tasks if file_name(t) == name)
        results[name] = load_result(path, task['request_sha256'])
    decisions = decisions_from_results(tasks, pairs, results)
    if decisions != read(out/'decisions.json'):
        raise RuntimeError('Diagnostic decisions disagree with model outputs')
    expected = {r['pair_id']: r for r in read(out/'expected.json')}
    rows = []
    for d in decisions:
        truth = expected[d['pair_id']]; raw = np.array(truth['raw_scores']); selected = d['mean_member']
        rows.append(dict(d, token=truth['token'], case_kind=truth['kind'], focus_metric=truth['focus_metric'],
            expected_member=truth['better_member'], mean_correct=selected == truth['better_member'],
            consistent_switch_correct=d['consistent_switch_member'] == truth['better_member'],
            both_orders_correct=all(m == truth['better_member'] and not tie
                                    for m, tie in zip(d['order_members'], d['order_ties'])),
            order_choice_flip=d['order_members'][0] != d['order_members'][1],
            expected_preference=float(d['mean_probabilities'][truth['better_member']]),
            selected_raw_pdm=float(raw[selected, 0]), baseline_raw_pdm=float(raw[0, 0]),
            raw_gain_pp=float(100*(raw[selected, 0]-raw[0, 0])),
            component_deltas={k: float(raw[selected, i]-raw[0, i]) for i, k in enumerate(truth['metric_keys']) if i}))
    conditions = {}
    for representation in view.REPRESENTATIONS:
        for mode in view.MODES:
            group = [r for r in rows if r['representation'] == representation and r['mode'] == mode]
            conditions[representation+'/'+mode] = dict(cases=len(group),
                expected_chosen_mean2=sum(r['mean_correct'] for r in group),
                expected_chosen_both_orders=sum(r['both_orders_correct'] for r in group),
                expected_chosen_consistent_switch=sum(r['consistent_switch_correct'] for r in group),
                order_choice_flips=sum(r['order_choice_flip'] for r in group),
                exact_ties=sum(sum(r['order_ties']) for r in group),
                repairs_chosen=sum(r['mean_correct'] for r in group if r['case_kind'] == 'repair'),
                baseline_retained_on_regression=sum(r['mean_correct'] for r in group if r['case_kind'] == 'regression'))
    report = dict(status='complete', type='outcome_selected_diagnostic', cases=len(pairs), requests=len(tasks),
                  conditions=conditions, selection_uses_known_outcomes=True,
                  reference_policies=dict(
                      always_keep_DS_correct=sum(r['better_member'] == 0 for r in expected.values()),
                      always_choose_challenger_correct=sum(r['better_member'] == 1 for r in expected.values()),
                      always_TA_correct=sum(view.names_and_order(pairs[k], 'canonical')[0][r['better_member']] == 'TA'
                                            for k, r in expected.items())),
                  expected_labels_in_model_inputs=False, note=view.NOTE)
    contrasts = {}
    def contrast(label, representation_a, mode_a, representation_b, mode_b):
        a = {r['pair_id']: r for r in rows if r['representation'] == representation_a and r['mode'] == mode_a}
        b = {r['pair_id']: r for r in rows if r['representation'] == representation_b and r['mode'] == mode_b}
        contrasts[label] = dict(
            became_correct=sum(a[k]['mean_correct'] and not b[k]['mean_correct'] for k in a),
            became_wrong=sum(b[k]['mean_correct'] and not a[k]['mean_correct'] for k in a),
            expected_preference_delta_by_case={k: a[k]['expected_preference']-b[k]['expected_preference'] for k in a})
    for representation in view.REPRESENTATIONS:
        contrast(representation+'_road_minus_blank', representation, 'road', representation, 'blank')
    for mode in view.MODES:
        contrast('footprint_minus_centerline_'+mode, 'footprint', mode, 'centerline', mode)
    report['paired_diagnostic_changes'] = contrasts
    write(out/'per_case.json', rows); write(out/'report.json', report)
    return report


def merge(root, package, out, plan, tasks, pairs, inputs):
    results = {}; profiles = []
    for i, assignments in enumerate(plan['assignments']):
        worker = out/'shards'/f'{i:02d}'; proof = read(worker/'worker_commit.json')
        expected = {name+'.json' for name in assignments}
        if set(proof['results']) != expected or {p.name for p in (worker/'results').glob('*.json')} != expected:
            raise RuntimeError('Incomplete/extra worker results')
        profiles.append(read(worker/'environment.json'))
        subset = [t for t in tasks if t['request_id'] in assignments]
        def forbidden(_):
            raise AssertionError('Merge must never run inference')
        infer_tasks(package, worker, subset, *inputs, forbidden)
        for name, digest in proof['results'].items():
            path = worker/'results'/name
            if name in results or sha(path) != digest:
                raise RuntimeError('Duplicate/changed worker result')
            results[name] = load_result(path)
    if any(p != profiles[0] for p in profiles):
        raise RuntimeError('Replica profiles differ')
    decisions = decisions_from_results(tasks, pairs, results)
    for i in range(plan['replicas']):
        for path in (out/'shards'/f'{i:02d}'/'results').glob('*.json'):
            target = out/'results'/path.name; target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha(target) != sha(path):
                raise RuntimeError('Central diagnostic result changed')
            if not target.exists():
                tmp = target.with_suffix('.json.tmp'); shutil.copyfile(path, tmp); tmp.replace(target)
    base.freeze(out/'decisions.json', decisions)
    base.freeze(out/'decision_commit.json', dict(plan_sha256=sha(out/'plan.json'),
        expected_sha256=sha(out/'expected.json'), decisions_sha256=sha(out/'decisions.json'),
        results={name: sha(out/'results'/name) for name in sorted(results)}))
    return score_committed(out, tasks, pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'worker', 'merge'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--replicas', type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument('--index', type=int)
    args = parser.parse_args(); root = args.root.expanduser().resolve()
    tag = os.environ.get('DREX_PAIR_DIAG_NAME', f'pair_diag_gpu{args.replicas}_v1')
    source_tag = os.environ.get('DREX_PAIR_SOURCE', 'overlay_v2_top4_gpu4_v1')
    if not re.fullmatch(r'pair_diag_[A-Za-z0-9_-]{1,64}', tag) or not re.fullmatch(r'overlay_v2_[A-Za-z0-9_-]{1,64}', source_tag):
        raise ValueError('Use a separate pair_diag_* output and the completed overlay_v2_* source')
    out = root/'runs'/tag; package = root/'data/navsim_train_v1/package'
    try:
        if args.stage == 'plan':
            result = freeze_plan(root, package, root/'runs'/source_tag, out, args.replicas)
        else:
            plan = checked_plan(root, package, out); inputs = inputs_for(package, out)
            tasks = read(out/'requests.json'); pairs = inputs[-1]
            if args.stage == 'worker':
                if args.index is None or not 0 <= args.index < plan['replicas']:
                    raise ValueError('Invalid worker index')
                worker = out/'shards'/f'{args.index:02d}'
                assigned = plan['assignments'][args.index]
                subset = [t for t in tasks if t['request_id'] in assigned]
                if (worker/'worker_commit.json').exists():
                    def forbidden(_): raise AssertionError('Finished worker must not repeat inference')
                    result = infer_tasks(package, worker, subset, *inputs, forbidden)
                else:
                    result = gpu_worker(root, package, worker, subset, inputs)
            else:
                result = merge(root, package, out, plan, tasks, pairs, inputs)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        target = out/'shards'/f'{args.index:02d}' if args.stage == 'worker' and args.index is not None else out
        write(target/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc())); raise


if __name__ == '__main__':
    main()
