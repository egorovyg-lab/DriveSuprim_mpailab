"""Compare nominal/executed path inputs on the completed six-pair diagnostic."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import traceback

import numpy as np

import pair_diagnostic as prior
import rollout_choice_data as view
from night import load_result, save_result, validate_probabilities
from runtime import read, write, environment, remaining_bytes
from transfer import sha, checked

base = prior.base
SOURCES = ('rollout_choice.py', 'rollout_choice_data.py', 'run_rollout_choice.sh', *prior.SOURCES)


def save_paths(path, arrays):
    view.validate_paths(arrays)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise RuntimeError('Sanitized paths changed; use a new rollout_choice_* name')
    else:
        path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix('.npz.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temp.replace(path)


def audit_paths(audit_out, pair_out, package, pairs, samples):
    """Planner only: authenticate the returned archive, then allowlist two pose arrays."""
    pointer = read(audit_out/'latest_result.json'); digest = pointer['archive_sha256']
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Invalid audit archive identity')
    archive_path = audit_out/('audit-'+digest+'.tar')
    result = (audit_out/('result_'+digest[:12])).resolve()
    if Path(pointer['result']).resolve() != result or pointer['remote_exit_code'] != 0 or sha(archive_path) != digest:
        raise RuntimeError('Successful, checksum-verified audit required')
    bindings = {str(archive_path): digest, str(audit_out/'latest_result.json'): sha(audit_out/'latest_result.json')}
    with tarfile.open(archive_path, 'r:') as archive:
        members = archive.getmembers()
        if any(not m.isfile() for m in members) or len({m.name for m in members}) != len(members):
            raise ValueError('Invalid audit archive membership')
        manifest = json.loads(archive.extractfile('artifact_manifest.json').read())
        if {m.name for m in members} != {'artifact_manifest.json', *manifest}:
            raise RuntimeError('Unexpected audit archive membership')
        for name in manifest:
            checked(result, name)
    def artifact(name):
        path = checked(result, name); entry = manifest[name]
        if path.stat().st_size != entry['size'] or sha(path) != entry['sha256']:
            raise RuntimeError('Audit artifact changed: '+name)
        bindings[str(path)] = entry['sha256']
        return path
    report, identity = read(artifact('report.json')), read(artifact('identity.json'))
    request_path = audit_out/'audit_request.json'
    if sha(request_path) != identity['request_sha256']:
        raise RuntimeError('Audit request identity changed')
    bindings[str(request_path)] = sha(request_path)
    request = read(request_path)
    if (report['status'] != 'complete' or report['metric_parity_passed'] != 2*len(pairs) or
            report['candidates'] != 2*len(pairs) or not report['geometry_interpretation_ready'] or
            request.get('protocol') != 'nominal_rollout_audit_v1' or
            request['package_manifest_sha256'] != sha(package/'package_manifest.json') or
            request['pair_plan_sha256'] != sha(pair_out/'plan.json') or
            request['pair_decision_commit_sha256'] != sha(pair_out/'decision_commit.json') or
            request['vehicle_profile'] != read(Path(__file__).with_name('vehicle_profile_astra.json'))):
        raise RuntimeError('Audit belongs to different pairs/package or failed metric parity')
    candidates = {(r['pair_id'], r['member']): r for r in request['candidates']}
    expected = {(p['pair_id'], member) for p in pairs.values() for member in (0, 1)}
    if set(candidates) != expected or len(request['candidates']) != len(expected):
        raise RuntimeError('Audit candidate membership changed')
    paths, errors = {}, []
    for pair_id, pair in pairs.items():
        paths[pair_id] = {k: [] for k in view.GEOMETRIES}
        for member in (0, 1):
            candidate = candidates[pair_id, member]
            if (candidate['token'] != pair['token'] or candidate['candidate_local'] != pair['local_ids'][member] or
                    candidate['candidate_vocab'] != pair['vocab_ids'][member]):
                raise RuntimeError('Audit candidate identity changed')
            folder = pair_id+'-member'+str(member)
            commit = read(artifact(folder+'/commit.json'))
            summary_path, traces_path = artifact(folder+'/summary.json'), artifact(folder+'/traces.npz')
            if any(sha(path) != commit['files'][path.name] for path in (summary_path, traces_path)):
                raise RuntimeError('Audit candidate commitment changed')
            summary = read(summary_path)
            if (summary['pair_id'] != pair_id or summary['member'] != member or summary['token'] != pair['token'] or
                    summary['candidate_vocab'] != pair['vocab_ids'][member] or not summary['parity_passed'] or
                    summary['samples'] != 41 or abs(summary['dt_s']-.1) > 1e-12 or
                    summary['expected_raw'] != dict(zip(candidate['metric_keys'], candidate['expected_raw']))):
                raise RuntimeError('Audit trace metadata/parity changed')
            with np.load(traces_path, allow_pickle=False) as arrays:
                # Never load observation arrays, map polygons, events, labels or audit PNGs as model inputs.
                reference = arrays['nominal_local'].copy(); executed = arrays['simulated_local'].copy()
            if any(p.shape != (41, 3) or not np.isfinite(p).all() or
                   not np.allclose(p[0], 0, atol=1e-7, rtol=0) for p in (reference, executed)):
                raise ValueError('Invalid audit pose sampling/origin')
            sample = samples[pair['token']]; index = sample['pool_local'].index(pair['local_ids'][member])
            nominal = np.asarray(sample['trajectories'][index], dtype=float)
            if not np.allclose(reference[1:, :2], nominal[:, :2], atol=1e-5, rtol=1e-7) or not np.allclose(
                    np.arctan2(np.sin(reference[1:, 2]-nominal[:, 2]), np.cos(reference[1:, 2]-nominal[:, 2])),
                    0, atol=1e-6, rtol=0):
                raise RuntimeError('Audit nominal path differs from the frozen vocabulary candidate')
            paths[pair_id]['nominal'].append(nominal)
            paths[pair_id]['simulated'].append(executed[1:])
            errors.append(dict(pair_id=pair_id, member=member, token=pair['token'],
                               **view.tracking_error(reference, executed)))
        paths[pair_id] = {k: np.asarray(v) for k, v in paths[pair_id].items()}
        view.validate_paths(paths[pair_id])
    return paths, errors, bindings


def tasks_for(pairs):
    tasks = []
    for geometry in view.GEOMETRIES:
        for old in prior.tasks_for(list(pairs.values())):
            task = dict(old, geometry=geometry, request_id=geometry+'-'+old['request_id'])
            tasks.append(task)
    return sorted(tasks, key=lambda t: base.digest([base.SEED, t['request_id'], 'assignment']))


def verified_pair(root, package, pair_out):
    prior.checked_plan(root, package, pair_out)
    inputs = prior.inputs_for(package, pair_out); pairs = inputs[3]
    tasks = read(pair_out/'requests.json'); commit = read(pair_out/'decision_commit.json')
    for name, key in (('plan.json', 'plan_sha256'), ('expected.json', 'expected_sha256'), ('decisions.json', 'decisions_sha256')):
        if sha(pair_out/name) != commit[key]: raise RuntimeError('Source pair commitment changed')
    if (set(commit['results']) != {prior.file_name(t) for t in tasks} or
            {p.name for p in (pair_out/'results').glob('*.json')} != set(commit['results'])):
        raise RuntimeError('Incomplete/extra source pair results')
    results = {}; bindings = {}
    for task in tasks:
        name = prior.file_name(task); path = pair_out/'results'/name
        if sha(path) != commit['results'][name]: raise RuntimeError('Source pair result changed')
        results[name] = load_result(path, task['request_sha256']); bindings[str(path)] = sha(path)
    if prior.decisions_from_results(tasks, pairs, results) != read(pair_out/'decisions.json'):
        raise RuntimeError('Source pair decisions changed')
    report = read(pair_out/'report.json')
    if report['status'] != 'complete' or report['cases'] != len(pairs) or report['requests'] != 8*len(pairs):
        raise RuntimeError('Completed pair diagnostic required')
    for name in ('plan.json', 'decision_commit.json', 'expected.json', 'decisions.json', 'report.json'):
        bindings[str(pair_out/name)] = sha(pair_out/name)
    return inputs, bindings


def freeze_plan(root, package, pair_out, audit_out, out, replicas):
    if replicas not in (1, 2, 4, 8):
        raise ValueError('Use 1,2,4 or 8 replicas')
    inputs, source_bindings = verified_pair(root, package, pair_out)
    manifest, scenes, samples, pairs = inputs
    paths, errors, audit_bindings = audit_paths(audit_out, pair_out, package, pairs, samples)
    out.mkdir(parents=True, exist_ok=True)
    base.freeze(out/'pair_inputs.json', list(pairs.values()))
    base.freeze(out/'expected.json', read(pair_out/'expected.json'))
    base.freeze(out/'tracking_error_report.json', dict(candidates=errors,
        note='Descriptive tracking error on outcome-selected pairs; not model performance.'))
    for pair_id, arrays in paths.items():
        save_paths(out/'paths'/(pair_id+'.npz'), arrays)
    tasks = tasks_for(pairs); previews = {}
    for task in tasks:
        pair = pairs[task['pair_id']]; token = pair['token']
        photo = base.camera_paths(package, scenes[token], manifest)[4]
        if sha(photo) != manifest['files'][photo.relative_to(package).as_posix()]['sha256']:
            raise RuntimeError('Current photo changed')
        row, positions, digest = view.payload(package, scenes[token], manifest, samples[token], pair,
            paths[pair['pair_id']], task['geometry'], task['representation'], task['mode'], task['ordering'])
        task.update(request_sha256=digest, pair_positions=positions)
        if task['mode'] == 'road' and task['ordering'] == 'canonical':
            for j, member in enumerate(positions):
                name = f'preview-{pair["pair_id"]}-{task["geometry"]}-{task["representation"]}-member{member}.png'
                previews[name] = prior.freeze_image(out/name, row['images'][j+1])
    base.freeze(out/'requests.json', tasks)
    files = ['pair_inputs.json', 'expected.json', 'requests.json', 'tracking_error_report.json']
    files += ['paths/'+pair_id+'.npz' for pair_id in pairs]
    plan = dict(config=view.CONFIG, replicas=replicas,
        assignments=[[t['request_id'] for t in tasks[i::replicas]] for i in range(replicas)],
        files={n: sha(out/n) for n in files}, previews=previews, source_bindings=source_bindings,
        audit_bindings=audit_bindings, package_manifest_sha256=sha(package/'package_manifest.json'),
        model_plan_sha256=sha(root/'model_plan.json'), environment_lock_sha256=sha(root/'requirements.lock.txt'),
        sources={n: sha(Path(__file__).with_name(n)) for n in SOURCES},
        selection_uses_known_outcomes=True, expected_labels_in_model_inputs=False, note=view.NOTE)
    base.freeze(out/'plan.json', plan)
    summary = dict(status='planned', cases=len(pairs), requests=len(tasks), replicas=replicas,
                   primary=view.CONFIG['primary'], output=str(out), note=view.NOTE)
    base.freeze(out/'plan_summary.json', summary)
    return summary


def checked_plan(root, package, out):
    plan = read(out/'plan.json')
    for n, digest in plan['sources'].items():
        if sha(Path(__file__).with_name(n)) != digest:
            raise RuntimeError('Rollout-choice code changed; use a new output name')
    for n, digest in {**plan['files'], **plan['previews']}.items():
        if sha(checked(out, n)) != digest:
            raise RuntimeError('Rollout-choice frozen input changed: '+n)
    bindings = {**plan['source_bindings'], **plan['audit_bindings'],
        str(package/'package_manifest.json'): plan['package_manifest_sha256'],
        str(root/'model_plan.json'): plan['model_plan_sha256'],
        str(root/'requirements.lock.txt'): plan['environment_lock_sha256']}
    for name, digest in bindings.items():
        if sha(Path(name)) != digest:
            raise RuntimeError('Rollout-choice source identity changed: '+name)
    return plan


def inputs_for(package, out):
    manifest, scenes, samples, pairs = prior.inputs_for(package, out)
    paths = {}
    for pair_id in pairs:
        with np.load(out/'paths'/(pair_id+'.npz'), allow_pickle=False) as arrays:
            paths[pair_id] = {k: arrays[k].copy() for k in arrays.files}
        view.validate_paths(paths[pair_id])
    return manifest, scenes, samples, pairs, paths


def infer_tasks(package, worker, tasks, inputs, infer):
    manifest, scenes, samples, pairs, paths = inputs
    expected = {prior.file_name(t) for t in tasks}
    if {p.name for p in (worker/'results').glob('*.json')}-expected:
        raise RuntimeError('Unexpected worker result files')
    committed = {}
    for task in tasks:
        pair = pairs[task['pair_id']]; token = pair['token']
        photo = base.camera_paths(package, scenes[token], manifest)[4]
        if sha(photo) != manifest['files'][photo.relative_to(package).as_posix()]['sha256']:
            raise RuntimeError('Current photo changed')
        row, positions, digest = view.payload(package, scenes[token], manifest, samples[token], pair,
            paths[pair['pair_id']], task['geometry'], task['representation'], task['mode'], task['ordering'])
        if digest != task['request_sha256'] or positions != task['pair_positions']:
            raise RuntimeError('Rollout-choice request/pixels changed')
        path = worker/'results'/prior.file_name(task)
        if path.exists():
            item = load_result(path, digest)
            if item['request_id'] != task['request_id'] or item['pair_positions'] != positions:
                raise RuntimeError('Cached worker result identity changed')
        else:
            item = dict(task, k=2, status='ok')
            try:
                item.update(infer(row)); validate_probabilities(item['probabilities'], 2)
            except Exception as exc:
                item.update(status='technical_failure', error=repr(exc)); save_result(path, item); raise
            save_result(path, item); print('ROLLOUT CHOICE '+task['request_id'], flush=True)
        committed[path.name] = sha(path)
    base.freeze(worker/'worker_commit.json', dict(results=committed))
    result = dict(status='complete', completed_requests=len(committed), expected_requests=len(expected),
                  expected_labels_deserialized=False, audit_traces_deserialized=False)
    write(worker/'inference_status.json', result)
    return result


def gpu_worker(root, package, worker, tasks, inputs):
    import time
    import torch
    from drex_max.model import DecisionModel
    check = worker/'runtime_check'; check.mkdir(parents=True, exist_ok=True)
    lock = check/'requirements.lock.txt'
    if lock.exists() and sha(lock) != sha(root/'requirements.lock.txt'):
        raise RuntimeError('Environment lock changed')
    if not lock.exists(): shutil.copyfile(root/'requirements.lock.txt', lock)
    env = environment(check, True)
    base.freeze(worker/'environment.json', {k: env[k] for k in ('python', 'torch', 'cuda_build', 'transformers', 'gpu', 'vram_bytes')})
    target = root/'models/drex-max-v1'
    if remaining_bytes(target, read(root/'model_plan.json')['files']): raise RuntimeError('Missing model files')
    config = read(target/'decision_config.json')
    if (config.get('format_version') != 1 or config.get('attention_mode', 'noncausal_full_attention') != 'noncausal_full_attention'
            or torch.cuda.device_count() != 1):
        raise RuntimeError('Expected original model protocol and one GPU per worker')
    model = DecisionModel(target, device='cuda:0', max_image_pixels=view.CONFIG['max_image_pixels'])
    if model.readout.weight.dtype != torch.bfloat16: raise RuntimeError('Expected BF16')
    def infer(row):
        batch = model.prepare([row], max_length=view.CONFIG['max_length'])
        grid = batch.inputs.get('image_grid_thw'); tokens = int(batch.inputs['input_ids'].shape[1])
        if grid is None or grid.shape[0] != 3 or tokens >= view.CONFIG['max_length']:
            raise RuntimeError('Unexpected image count/context truncation')
        del batch; torch.cuda.synchronize(); started = time.monotonic()
        probabilities = model.predict([row], batch_size=1, max_length=view.CONFIG['max_length'])[0]
        torch.cuda.synchronize()
        return dict(probabilities=[float(x) for x in probabilities], input_tokens=tokens, seconds=time.monotonic()-started)
    return infer_tasks(package, worker, tasks, inputs, infer)


def decisions_from_results(tasks, pairs, results):
    if set(results) != {prior.file_name(t) for t in tasks}: raise RuntimeError('Incomplete rollout-choice results')
    decisions = []
    for geometry in view.GEOMETRIES:
        subset = [t for t in tasks if t['geometry'] == geometry]
        selected = {prior.file_name(t): results[prior.file_name(t)] for t in subset}
        decisions.extend(dict(d, geometry=geometry) for d in prior.decisions_from_results(subset, pairs, selected))
    return decisions


def score_committed(out, tasks, pairs):
    commit = read(out/'decision_commit.json')
    for name, key in (('plan.json', 'plan_sha256'), ('expected.json', 'expected_sha256'), ('decisions.json', 'decisions_sha256')):
        if sha(out/name) != commit[key]: raise RuntimeError('Rollout-choice commitment changed')
    if set(commit['results']) != {prior.file_name(t) for t in tasks}: raise RuntimeError('Incomplete decision commitment')
    results = {}
    for task in tasks:
        name = prior.file_name(task); path = out/'results'/name
        if sha(path) != commit['results'][name]: raise RuntimeError('Committed result changed')
        results[name] = load_result(path, task['request_sha256'])
    decisions = decisions_from_results(tasks, pairs, results)
    if decisions != read(out/'decisions.json'): raise RuntimeError('Frozen decisions disagree with probabilities')
    expected = {r['pair_id']: r for r in read(out/'expected.json')}; rows = []
    for d in decisions:
        truth = expected[d['pair_id']]; selected = d['mean_member']; raw = np.asarray(truth['raw_scores'])
        rows.append(dict(d, token=truth['token'], expected_member=truth['better_member'],
            mean_correct=selected == truth['better_member'],
            consistent_switch_correct=d['consistent_switch_member'] == truth['better_member'],
            both_orders_correct=all(m == truth['better_member'] and not tie for m, tie in zip(d['order_members'], d['order_ties'])),
            order_choice_flip=d['order_members'][0] != d['order_members'][1],
            expected_preference=d['mean_probabilities'][truth['better_member']],
            raw_gain_pp=float(100*(raw[selected, 0]-raw[0, 0])),
            component_deltas={k: float(raw[selected, i]-raw[0, i]) for i, k in enumerate(truth['metric_keys']) if i}))
    conditions = {}; contrasts = {}
    for geometry in view.GEOMETRIES:
        for representation in view.original.REPRESENTATIONS:
            for mode in view.original.MODES:
                group = [r for r in rows if (r['geometry'], r['representation'], r['mode']) == (geometry, representation, mode)]
                conditions['/'.join((geometry, representation, mode))] = dict(cases=len(group),
                    expected_chosen_mean2=sum(r['mean_correct'] for r in group),
                    expected_chosen_both_orders=sum(r['both_orders_correct'] for r in group),
                    expected_chosen_consistent_switch=sum(r['consistent_switch_correct'] for r in group),
                    order_choice_flips=sum(r['order_choice_flip'] for r in group),
                    exact_ties=sum(sum(r['order_ties']) for r in group),
                    repairs_chosen=sum(r['mean_correct'] for r in group if expected[r['pair_id']]['kind'] == 'repair'),
                    baseline_retained_on_regression=sum(r['mean_correct'] for r in group if expected[r['pair_id']]['kind'] == 'regression'),
                    gain_mean_pp=float(np.mean([r['raw_gain_pp'] for r in group])))
    def contrast(name, a_key, b_key):
        a = {r['pair_id']: r for r in rows if (r['geometry'], r['representation'], r['mode']) == a_key}
        b = {r['pair_id']: r for r in rows if (r['geometry'], r['representation'], r['mode']) == b_key}
        contrasts[name] = dict(became_correct=sum(a[k]['mean_correct'] and not b[k]['mean_correct'] for k in a),
            became_wrong=sum(b[k]['mean_correct'] and not a[k]['mean_correct'] for k in a),
            expected_preference_delta_by_case={k: a[k]['expected_preference']-b[k]['expected_preference'] for k in a})
    for representation in view.original.REPRESENTATIONS:
        for mode in view.original.MODES:
            contrast('simulated_minus_nominal/'+representation+'/'+mode,
                     ('simulated', representation, mode), ('nominal', representation, mode))
        for geometry in view.GEOMETRIES:
            contrast('road_minus_blank/'+geometry+'/'+representation,
                     (geometry, representation, 'road'), (geometry, representation, 'blank'))
    report = dict(status='complete', type='outcome_selected_representation_diagnostic',
        cases=len(pairs), requests=len(tasks), primary=view.CONFIG['primary'], conditions=conditions,
        paired_changes=contrasts, reference_policies=dict(always_keep_DS_correct=sum(r['better_member'] == 0 for r in expected.values()),
            always_choose_challenger_correct=sum(r['better_member'] == 1 for r in expected.values())),
        selection_uses_known_outcomes=True, expected_labels_in_model_inputs=False,
        new_scorer_calls=0, holdout_processed=0, note=view.NOTE)
    write(out/'per_case.json', rows); write(out/'report.json', report)
    return report


def merge(package, out, plan, tasks, inputs):
    results = {}; profiles = []
    for i, assignments in enumerate(plan['assignments']):
        worker = out/'shards'/f'{i:02d}'; proof = read(worker/'worker_commit.json')
        expected = {n+'.json' for n in assignments}
        if set(proof['results']) != expected or {p.name for p in (worker/'results').glob('*.json')} != expected:
            raise RuntimeError('Incomplete/extra worker results')
        subset = [t for t in tasks if t['request_id'] in assignments]
        def forbidden(_): raise AssertionError('Merge must never run inference')
        infer_tasks(package, worker, subset, inputs, forbidden)
        profiles.append(read(worker/'environment.json'))
        for name, digest in proof['results'].items():
            path = worker/'results'/name
            if name in results or sha(path) != digest: raise RuntimeError('Changed/duplicate worker result')
            results[name] = load_result(path)
    if any(p != profiles[0] for p in profiles): raise RuntimeError('Replica profiles differ')
    decisions = decisions_from_results(tasks, inputs[3], results)
    for i in range(plan['replicas']):
        for path in (out/'shards'/f'{i:02d}'/'results').glob('*.json'):
            target = out/'results'/path.name; target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha(target) != sha(path): raise RuntimeError('Central result changed')
            if not target.exists():
                temp = target.with_suffix('.json.tmp'); shutil.copyfile(path, temp); temp.replace(target)
    base.freeze(out/'decisions.json', decisions)
    base.freeze(out/'decision_commit.json', dict(plan_sha256=sha(out/'plan.json'), expected_sha256=sha(out/'expected.json'),
        decisions_sha256=sha(out/'decisions.json'), results={n: sha(out/'results'/n) for n in sorted(results)}))
    return score_committed(out, tasks, inputs[3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'worker', 'merge'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--replicas', type=int, choices=(1, 2, 4, 8), default=2); parser.add_argument('--index', type=int)
    args = parser.parse_args(); root = args.root.expanduser().resolve()
    name = os.environ.get('DREX_ROLLOUT_CHOICE_NAME', f'rollout_choice_gpu{args.replicas}_v1')
    pair = os.environ.get('DREX_ROLLOUT_PAIR_SOURCE', 'pair_diag_gpu4_v1')
    audit = os.environ.get('DREX_ROLLOUT_AUDIT_NAME', 'rollout_audit_pair6_v2')
    if not all(re.fullmatch(prefix+'[A-Za-z0-9_-]{1,64}', value) for prefix, value in
               (('rollout_choice_', name), ('pair_diag_', pair), ('rollout_audit_', audit))):
        raise ValueError('Expected separate rollout_choice_* output and completed pair/audit sources')
    out = root/'runs'/name; package = root/'data/navsim_train_v1/package'
    try:
        if args.stage == 'plan':
            result = freeze_plan(root, package, root/'runs'/pair, root/'runs'/audit, out, args.replicas)
        else:
            plan = checked_plan(root, package, out)
            if plan['replicas'] != args.replicas: raise ValueError('Worker count differs from frozen plan')
            inputs = inputs_for(package, out); tasks = read(out/'requests.json')
            if args.stage == 'worker':
                if args.index is None or not 0 <= args.index < plan['replicas']: raise ValueError('Invalid worker index')
                worker = out/'shards'/f'{args.index:02d}'; assigned = plan['assignments'][args.index]
                subset = [t for t in tasks if t['request_id'] in assigned]
                if (worker/'worker_commit.json').exists():
                    def forbidden(_): raise AssertionError('Finished worker must not repeat inference')
                    result = infer_tasks(package, worker, subset, inputs, forbidden)
                else: result = gpu_worker(root, package, worker, subset, inputs)
            else: result = merge(package, out, plan, tasks, inputs)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        target = out/'shards'/f'{args.index:02d}' if args.stage == 'worker' and args.index is not None else out
        write(target/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc())); raise


if __name__ == '__main__': main()
