"""Resumable CPU four-fold probe, independent of NAVSIM and Torch."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

import consequence_data as data
import consequence_model as model
from transfer import freeze, read, sha, write

TOOLS = ('consequence_model.py', 'consequence_data.py', 'consequence_run.py', 'transfer.py')


def save_arrays(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.npz.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def context(path):
    request = read(path)
    if request['protocol'] != 'drex_consequence_probe_v1' or request['config'] != model.CONFIG:
        raise RuntimeError('Probe protocol/config changed; use a new run name')
    if request['features'] != list(data.FEATURES) or request['outcomes'] != list(data.OUTCOMES):
        raise RuntimeError('Probe schema changed')
    if set(request['tools']) != set(TOOLS):
        raise RuntimeError('Probe tool membership changed')
    for name, digest in request['tools'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise RuntimeError('Probe code changed: ' + name)
    for name, digest in request['files'].items():
        if name not in ('features.npz', 'labels.npz') or sha(path.parent/name) != digest:
            raise RuntimeError('Bound probe dataset changed')
    if set(request['files']) != {'features.npz', 'labels.npz'}:
        raise RuntimeError('Missing probe inputs')
    f, label = load_arrays(path.parent/'features.npz'), load_arrays(path.parent/'labels.npz')
    records = request['records']; n = len(records)
    tokens = [r['token'] for r in records]; sessions = [r['session_id'] for r in records]
    if (len(set(tokens)) != n or any(r['role'] != 'train' for r in records)
            or f['x'].shape != (n, 63, len(data.FEATURES)) or f['preferences'].shape != (n, 63, 3, 4)
            or f['local_ids'].shape != (n, 64) or f['vocab_ids'].shape != (n, 64)
            or not np.isfinite(f['x']).all() or not np.isfinite(f['preferences']).all()
            or np.any((f['preferences'] < 0) | (f['preferences'] > 1))):
        raise RuntimeError('Invalid cohort/features')
    for key in ('local_ids', 'vocab_ids'):
        if any(len(set(row)) != 64 for row in f[key]):
            raise RuntimeError('Duplicate candidate IDs')
    if not np.array_equal(f['fold'], model.folds(sessions)):
        raise RuntimeError('Session fold assignment changed')
    y, flags = data.truth(label['outcomes'])
    if y.shape != (n, 63, len(model.TARGETS)):
        raise RuntimeError('Label/cohort count changed')
    return request, f, label['outcomes'], y, flags


def fold_fit(features, y, records, fold):
    assignment = features['fold']; train, test = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
    sessions = np.asarray([r['session_id'] for r in records]); tokens = [r['token'] for r in records]
    if set(sessions[train]) & set(sessions[test]) or not len(test) or not len(train):
        raise RuntimeError('Whole-session separation failed')
    w = model.weights(sessions[train], 63)
    predictions = np.empty((len(test), len(model.FAMILIES), len(model.ARMS), 63, len(model.TARGETS)))
    stored = dict(test_indices=test, train_mean=w @ y[train].reshape(-1, y.shape[-1]))
    for ai, arm in enumerate(model.ARMS):
        x = model.inputs(features['x'], features['preferences'], tokens, arm)
        for fi, family in enumerate(model.FAMILIES):
            fitted = model.fit(x[train].reshape(-1, x.shape[-1]), y[train].reshape(-1, y.shape[-1]), w, family)
            predictions[:, fi, ai] = model.predict(fitted, x[test].reshape(-1, x.shape[-1])).reshape(len(test), 63, -1)
            stored.update({f'{fi}_{ai}_{k}': v for k, v in fitted.items()})
            print(f'PROBE fold={fold+1}/4 family={family} arm={arm}', flush=True)
    stored['predictions'] = predictions
    if not np.isfinite(predictions).all():
        raise RuntimeError('Pilot/fold prediction finite check failed')
    return stored


def csv_gz(path, rows):
    rows = iter(rows); first = next(rows)
    temp = path.with_suffix('.gz.tmp')
    with gzip.open(temp, 'wt', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(first)); writer.writeheader(); writer.writerow(first); writer.writerows(rows)
    temp.replace(path)


def diagnostic_summary(prediction, y, sessions):
    sessions = np.asarray(sessions)
    result = {}
    for j, name in enumerate(model.TARGETS):
        p = np.clip(prediction[:, :, j], 0, 1) if j >= 5 else prediction[:, :, j]
        target = y[:, :, j]; error = p - target
        per_session = [error[sessions == s].ravel() for s in sorted(set(sessions))]
        value = dict(mae_session_macro=float(np.mean([np.abs(e).mean() for e in per_session])),
                     rmse_session_macro=float(np.mean([np.sqrt((e*e).mean()) for e in per_session])))
        if j >= 5:
            aps = [model.average_precision(target[sessions == s].ravel(), p[sessions == s].ravel()) for s in sorted(set(sessions))]
            known = [v for v in aps if v is not None]
            value.update(average_precision_session_macro=float(np.mean(known)) if known else None,
                         sessions_with_positives=len(known), positive_actions=int(target.sum()),
                         positive_fraction=float(target.mean()))
        result[name] = value
    return result


def decisions(prediction, y, flags, records, ids, threshold, method):
    chosen = model.choose(prediction, threshold)
    for i, (j, r) in enumerate(zip(chosen, records)):
        row = dict(token=r['token'], session_id=r['session_id'], method=method,
                   candidate_local=int(ids[i, j+1]), teacher_rank=int(j+2) if j >= 0 else 1,
                   intervention=bool(j >= 0))
        for k, name in enumerate(model.TARGETS):
            row['actual_' + name] = float(y[i, j, k]) if j >= 0 else 0.
            row['predicted_' + name] = float(prediction[i, j, k]) if j >= 0 else 0.
        for key, v in flags.items():
            row[key] = bool(v[i, j]) if j >= 0 else False
        yield row


def summarize_rows(rows):
    sessions = [r['session_id'] for r in rows]
    keys = ['actual_jerk_gain_mps3', 'actual_lateral_gain_mps2', 'actual_progress_delta_m', 'actual_pdm_gain_pp']
    result = {key: model.paired_interval([r[key] for r in rows], sessions) for key in keys}
    for key in ('intervention', 'actual_harm', 'actual_benefit', 'raw_component_harm',
                'progress_budget_exceeded', 'exposure_worse', 'margin_loss', 'incomplete_actor_coverage',
                'road_margin_unavailable', 'actor_margin_unavailable'):
        result[key] = int(sum(r[key] for r in rows))
    result['jerk_peak_increased'] = sum(r['actual_jerk_peak_delta_mps3'] > 1e-8 for r in rows)
    return result


def report(out, request, f, y, flags, predictions, mask, train_means):
    records = [r for i, r in enumerate(request['records']) if mask[i]]
    sessions = [r['session_id'] for r in records]; y = y[mask]
    flags = {k: v[mask] for k, v in flags.items()}
    p = predictions[mask]; ids = f['local_ids'][mask]
    accuracy = {'train_mean': diagnostic_summary(train_means[mask], y, sessions)}
    policies, rows, primary = {}, [], {}
    baseline = list(decisions(np.zeros((len(records), 63, len(model.TARGETS))), y, flags, records, ids, 1.01, 'baseline_DS'))
    rows.extend(baseline); policies['baseline_DS'] = summarize_rows(baseline)
    for fi, family in enumerate(model.FAMILIES):
        for ai, arm in enumerate(model.ARMS):
            name = family + '/' + arm
            accuracy[name] = diagnostic_summary(p[:, fi, ai], y, sessions)
            for threshold in model.CONFIG['benefit_thresholds']:
                method = name + '/t' + str(threshold)
                group = list(decisions(p[:, fi, ai], y, flags, records, ids, threshold, method))
                rows.extend(group); policies[method] = summarize_rows(group)
                if family == model.CONFIG['primary_family'] and threshold == model.CONFIG['primary_threshold']:
                    primary[arm] = group
    contrasts = {}
    for arm in ('ds', 'ds_drex_text', 'ds_drex_blank', 'ds_drex_shuffled'):
        a, b = primary['ds_drex_road'], primary[arm]
        contrasts['road_minus_' + arm] = {key: model.paired_interval(
            [x[key]-z[key] for x, z in zip(a, b)], sessions) for key in (
                'actual_benefit', 'actual_harm', 'actual_jerk_gain_mps3', 'actual_lateral_gain_mps2',
                'actual_progress_delta_m', 'actual_pdm_gain_pp')}
    # Auxiliary information endpoint: positive values mean road adds held-out MAE over control (worse).
    information_contrasts = {}
    fi = model.FAMILIES.index(model.CONFIG['primary_family'])
    road = p[:, fi, model.ARMS.index('ds_drex_road')]
    for arm in ('ds', 'ds_drex_blank', 'ds_drex_shuffled'):
        control = p[:, fi, model.ARMS.index(arm)]
        information_contrasts[arm] = {name: model.paired_interval(
            (np.abs(road[:, :, j]-y[:, :, j])-np.abs(control[:, :, j]-y[:, :, j])).mean(axis=1), sessions)
            for j, name in enumerate(model.TARGETS[:5])}
    def candidate_rows():
        for fi, family in enumerate(model.FAMILIES):
            for ai, arm in enumerate(model.ARMS):
                for i, r in enumerate(records):
                    for j in range(63):
                        yield dict(token=r['token'], session_id=r['session_id'], method=family+'/'+arm,
                                   candidate_local=int(ids[i, j+1]), teacher_rank=j+2,
                                   **{'actual_'+k: float(y[i, j, z]) for z, k in enumerate(model.TARGETS)},
                                   **{'predicted_'+k: float(p[i, fi, ai, j, z]) for z, k in enumerate(model.TARGETS)})
    csv_gz(out/'per_candidate.csv.gz', candidate_rows()); csv_gz(out/'per_scene.csv.gz', rows)
    overview = {}
    for arm, group in primary.items():
        summary = summarize_rows(group)
        overview[arm] = dict(interventions=summary['intervention'], meaningful_benefits=summary['actual_benefit'],
                             observed_harms=summary['actual_harm'], raw_component_harms=summary['raw_component_harm'],
                             **{key+'_session_macro': summary[key]['mean'] for key in (
                                 'actual_jerk_gain_mps3', 'actual_lateral_gain_mps2',
                                 'actual_progress_delta_m', 'actual_pdm_gain_pp')})
    result = dict(status='complete' if mask.all() else 'pilot_complete', scenes=len(records),
                  sessions=len(set(sessions)), completed_folds=sorted(set(f['fold'][mask].tolist())),
                  held_out_pairs=int(len(records)*63), config=model.CONFIG, accuracy=accuracy, policies=policies,
                  primary_contrasts=contrasts, continuous_mae_road_minus_control=information_contrasts,
                  holdout_processed=0, new_drex_calls=0, new_scorer_calls=0,
                  note=model.NOTE, targets=list(model.TARGETS),
                  overview=overview)
    write(out/'report.json', result)
    return result


def run(path, out, stage):
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(path, out, stage)


def run_locked(path, out, stage):
    if stage not in ('pilot', 'full'):
        raise ValueError('Pilot/full required')
    request, f, outcomes, y, flags = context(path)
    identity = dict(request_sha256=sha(path), python=platform.python_version(), numpy=np.__version__,
                    thread_env={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    freeze(out/'identity.json', identity); identity_sha = sha(out/'identity.json')
    count = len(request['records']); predictions = np.full((count, 2, 5, 63, len(model.TARGETS)), np.nan)
    mask = np.zeros(count, bool); train_means = np.full_like(y, np.nan); start = time.monotonic()
    for fold in range(model.CONFIG['folds'] if stage == 'full' else 1):
        directory = out/'folds'/str(fold); checkpoint = directory/'model_predictions.npz'
        commit_path = directory/'commit.json'
        if stage == 'full' and fold == 0 and not commit_path.exists():
            raise RuntimeError('Run the complete first outer fold as pilot before full')
        if commit_path.exists():
            commit = read(commit_path)
            if commit != dict(identity_sha256=identity_sha, fold=fold, file_sha256=sha(checkpoint)):
                raise RuntimeError('Fold checkpoint binding changed')
            block = load_arrays(checkpoint)
        else:
            block = fold_fit(f, y, request['records'], fold)
            save_arrays(checkpoint, **block)
            write(commit_path, dict(identity_sha256=identity_sha, fold=fold, file_sha256=sha(checkpoint)))
        index = block['test_indices']
        if (not np.array_equal(index, np.flatnonzero(f['fold'] == fold))
                or block['predictions'].shape != predictions[index].shape or not np.isfinite(block['predictions']).all()):
            raise RuntimeError('Invalid held-out checkpoint')
        predictions[index] = block['predictions']; train_means[index] = block['train_mean']; mask[index] = True
        write(out/'progress.json', dict(stage=stage, completed_folds=fold+1, required_folds=4,
                                       elapsed_seconds=time.monotonic()-start))
    result = report(out, request, f, y, flags, predictions, mask, train_means)
    print(json.dumps({k: result[k] for k in ('status', 'scenes', 'sessions', 'completed_folds')}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('pilot', 'full'), required=True)
    args = parser.parse_args(); run(args.input.resolve(), args.out.resolve(), args.stage)
