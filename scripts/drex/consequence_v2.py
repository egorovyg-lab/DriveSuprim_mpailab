"""Nested session validation of separately calibrated consequence classifiers."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import platform

import numpy as np

import consequence_data as data
import consequence_model as old
import consequence_run as storage
from transfer import freeze, read, sha, write

CLASSES = ('raw_harm', 'geometry_harm', 'progress_violation', 'comfort_regression', 'benefit')
CONFIG = dict(version=2, inner_folds=3, logistic_lambda=.01, logistic_steps=30,
              risk_grid=[.025, .05, .1, .2, .4, 1.01], benefit_grid=[0., .005, .01, .025, .05, .1, 1.01],
              progress_grid=[.1, .25, .5, 1.01], minimum_support_sessions=3,
              primary='ds_drex_road', seed=20261116)
NOTE = ('Development-only nested session OOF, on previously inspected TRAIN scenes. '
        'No new Drex/scorer calls or holdout. Raw, geometry, progress and comfort events are separate. '
        'Inner OOF logistic calibration does not certify safety or a 5% risk bound. '
        'Absent classes use an explicit constant fallback. Gates selected only on inner TRAIN OOF; '
        'zero observed inner harm and three supporting sessions do not guarantee generalization. '
        'Frozen cached Drex preferences only, not Drex fine-tuning. Local PDM, not full EPDMS.')


def targets(outcomes):
    continuous, flags = data.truth(outcomes)
    geometry = flags['exposure_worse'] | flags['margin_loss'] | flags['incomplete_actor_coverage']
    comfort = ((continuous[:, :, 0] < -1e-8) | (continuous[:, :, 1] < -1e-8)
               | (continuous[:, :, 4] > 1e-8))
    binary = np.stack([flags['raw_component_harm'], geometry, flags['progress_budget_exceeded'],
                       comfort, continuous[:, :, 6].astype(bool)], axis=2).astype(float)
    return continuous, binary, flags


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))


def logistic(a, y, w, regularization):
    """Natural-prevalence, session-balanced log loss; no class oversampling."""
    w = w/w.sum(); coef = np.zeros((a.shape[1], y.shape[1]))
    constants = np.array([np.unique(y[:, j]).size == 1 for j in range(y.shape[1])])
    prior = np.clip(w @ y, 1e-6, 1-1e-6)
    coef[0] = np.log(prior/(1-prior))
    penalty = np.eye(a.shape[1])*regularization; penalty[0, 0] = 0.
    for j in range(y.shape[1]):
        if constants[j]: continue
        c = coef[:, j]
        for _ in range(CONFIG['logistic_steps']):
            p = sigmoid(a@c)
            h = a.T @ ((w*p*(1-p))[:, None]*a) + penalty + np.eye(a.shape[1])*1e-9
            delta = np.linalg.solve(h, a.T@(w*(p-y[:, j])) + penalty@c)
            # Backtracking avoids unstable Newton jumps for rare events.
            loss = lambda v: float(w @ (np.logaddexp(0., a@v)-y[:, j]*(a@v)) + .5*v@penalty@v)
            before = loss(c); step = 1.
            while step > 1e-6 and loss(c-step*delta) > before: step *= .5
            c = c-step*delta
            if np.max(np.abs(step*delta)) < 1e-6: break
        coef[:, j] = c
    return coef, constants


def fit_heads(x, continuous, binary, sessions):
    w = old.weights(sessions, x.shape[1]); a = x.reshape(-1, x.shape[-1])
    regression = old.fit(a, continuous.reshape(-1, continuous.shape[-1]), w, 'linear')
    design = old.design(a, regression)
    coef, constants = logistic(design, binary.reshape(-1, binary.shape[-1]), w, CONFIG['logistic_lambda'])
    return regression, coef, constants


def predict_heads(heads, x):
    regression, coef, _ = heads; a = x.reshape(-1, x.shape[-1])
    return (old.predict(regression, a).reshape(*x.shape[:2], -1),
            (old.design(a, regression) @ coef).reshape(*x.shape[:2], -1))


def calibration_design(logits, j):
    return np.column_stack((np.ones(logits.shape[0]*logits.shape[1]), logits[:, :, j].ravel()))


def calibrate(logits, binary, sessions):
    w = old.weights(sessions, logits.shape[1]); coeff = []; constants = []
    for j in range(binary.shape[-1]):
        c, absent = logistic(calibration_design(logits, j), binary[:, :, j].reshape(-1, 1), w, 1e-4)
        coeff.append(c[:, 0]); constants.append(bool(absent[0]))
    return np.asarray(coeff), constants


def probabilities(logits, calibration):
    return sigmoid(logits*calibration[:, 1] + calibration[:, 0])


def inner_predictions(x, continuous, binary, sessions):
    names = sorted(set(sessions), key=lambda s: old.digest((CONFIG['seed'], s, 'inner')))
    assignment = {s: i % CONFIG['inner_folds'] for i, s in enumerate(names)}
    fold = np.array([assignment[s] for s in sessions])
    c = np.full_like(continuous, np.nan); logits = np.full_like(binary, np.nan)
    fallbacks = []
    for k in range(CONFIG['inner_folds']):
        train, test = fold != k, fold == k
        if not train.any() or not test.any(): raise ValueError('Insufficient inner sessions')
        print('INNER fold='+str(k+1)+'/3 train_sessions='+str(len(set(sessions[train]))), flush=True)
        heads = fit_heads(x[train], continuous[train], binary[train], sessions[train])
        c[test], logits[test] = predict_heads(heads, x[test])
        fallbacks.append(heads[2].tolist())
    if not np.isfinite(c).all() or not np.isfinite(logits).all(): raise RuntimeError('Invalid inner OOF')
    return c, logits, fallbacks


def select(scores, prediction, gate):
    risk, benefit, progress = gate
    allowed = ((scores[:, :, :2].max(axis=2) <= risk) & (scores[:, :, 3] <= risk)
               & (scores[:, :, 2] <= progress) & (scores[:, :, 4] > benefit)
               & (prediction[:, :, 2] >= -1.))
    # Prefer the predicted probability of the predefined composite comfort benefit.
    rank = np.where(allowed, scores[:, :, 4], -np.inf)
    return np.where(allowed.any(axis=1), rank.argmax(axis=1), -1)


def chosen_values(values, choice):
    output = np.zeros((len(choice), values.shape[-1]))
    active = choice >= 0
    output[active] = values[np.flatnonzero(active), choice[active]]
    return output


def tune_gate(scores, prediction, continuous, binary, sessions):
    """Only inner outcomes allowed. No guarantee a nontrivial gate is supportable."""
    best = None; best_key = None
    for risk in CONFIG['risk_grid']:
        for benefit in CONFIG['benefit_grid']:
            for progress in CONFIG['progress_grid']:
                gate = (risk, benefit, progress); choice = select(scores, prediction, gate)
                observed = chosen_values(binary, choice)
                if observed[:, :4].any(): continue
                supported = len(set(sessions[observed[:, 4] > 0]))
                if supported < CONFIG['minimum_support_sessions']: continue
                macro = np.mean([observed[sessions == s, 4].mean() for s in sorted(set(sessions))])
                key = (float(macro), -int((choice >= 0).sum()), -risk, benefit, -progress)
                if best_key is None or key > best_key:
                    best_key, best = key, dict(gate=list(gate), support_sessions=supported,
                                               inner_benefits=int(observed[:, 4].sum()), inner_harms=0)
    return best or dict(gate=[0., 1.01, 0.], support_sessions=0, inner_benefits=0, inner_harms=0,
                        fallback='No nontrivial inner gate met the fixed support constraints')


def fit_outer(x, continuous, binary, sessions):
    inner_c, inner_l, absent = inner_predictions(x, continuous, binary, sessions)
    calibration, constant = calibrate(inner_l, binary, sessions)
    # Threshold selection on calibrated inner OOF is tuning, never the outer evaluation.
    gate = tune_gate(probabilities(inner_l, calibration), inner_c, continuous, binary, sessions)
    heads = fit_heads(x, continuous, binary, sessions)
    return heads, calibration, dict(**gate, calibration_constant_classes=constant,
                                    inner_constant_classes=absent, outer_train_constant_classes=heads[2].tolist())


def run(source, out, stage):
    request, features, outcomes, _, _ = storage.context(source/'audit_request.json')
    c, b, flags = targets(outcomes)
    records = request['records']; sessions = np.array([r['session_id'] for r in records]); tokens = [r['token'] for r in records]
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol='consequence_nested_v2', config=CONFIG, source_request_sha256=sha(source/'audit_request.json'),
                    code_sha256=sha(Path(__file__)), python=platform.python_version(), numpy=np.__version__,
                    threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}, note=NOTE)
    freeze(out/'identity.json', identity); binding = sha(out/'identity.json')
    choices = []; predictions = []; selections = {}; completed = []
    for fold in range(1 if stage == 'pilot' else 4):
        train, test = features['fold'] != fold, features['fold'] == fold
        if set(sessions[train]) & set(sessions[test]): raise RuntimeError('Outer session overlap')
        index = np.flatnonzero(test)
        for arm in old.ARMS:
            write(out/'progress.json', dict(state='fitting_or_loading', fold=fold+1, arm=arm, total_folds=4))
            folder = out/'folds'/str(fold)/arm; path = folder/'predictions.npz'; commit_path = folder/'commit.json'
            if stage == 'full' and fold == 0 and not commit_path.exists(): raise RuntimeError('Pilot first')
            if commit_path.exists():
                commit = read(commit_path)
                if commit['identity_sha256'] != binding or commit['sha256'] != sha(path): raise RuntimeError('Checkpoint changed')
                block = storage.load_arrays(path); selected = commit['selection']
            else:
                x = old.inputs(features['x'], features['preferences'], tokens, arm)
                heads, calibration, selected = fit_outer(x[train], c[train], b[train], sessions[train])
                pred, logits = predict_heads(heads, x[test]); prob = probabilities(logits, calibration)
                block = dict(index=index, continuous=pred, scores=prob, uncalibrated_scores=sigmoid(logits),
                             calibration=calibration, classifier=heads[1],
                             **{'regression_'+k:v for k,v in heads[0].items()})
                if not np.isfinite(pred).all() or not np.isfinite(prob).all(): raise RuntimeError('Nonfinite outer predictions')
                storage.save_arrays(path, **block)
                write(commit_path, dict(identity_sha256=binding, sha256=sha(path), selection=selected))
            if (not np.array_equal(block['index'], index) or block['continuous'].shape != c[test].shape
                    or block['scores'].shape != b[test].shape or not np.isfinite(block['continuous']).all()
                    or not np.isfinite(block['scores']).all()): raise RuntimeError('Invalid prediction checkpoint')
            pred, scores = block['continuous'], block['scores']; selections[str(fold)+'/'+arm] = selected
            print('NESTED fold='+str(fold+1)+'/4 arm='+arm+' gate='+str(selected['gate']), flush=True)
            for policy, choice in [('gated', select(scores, pred, selected['gate'])),
                                    ('ranking_only', scores[:, :, 4].argmax(axis=1))]:
                actual = chosen_values(c[test], choice); events = chosen_values(b[test], choice)
                for i, j in enumerate(choice):
                    choices.append(dict(token=tokens[index[i]], session_id=sessions[index[i]], method=arm+'/'+policy,
                        candidate_local=int(features['local_ids'][index[i], j+1]), intervention=bool(j>=0),
                        **{k:float(actual[i,z]) for z,k in enumerate(old.TARGETS[:5])},
                        **{k:bool(v[index[i],j]) if j>=0 else False for k,v in flags.items()},
                        **{k:float(events[i,z]) for z,k in enumerate(CLASSES)}))
            for i, row in enumerate(index):
                for j in range(63):
                    predictions.append(dict(token=tokens[row], session_id=sessions[row], arm=arm,
                        candidate_local=int(features['local_ids'][row,j+1]),
                        **{'prob_'+k:float(scores[i,j,z]) for z,k in enumerate(CLASSES)},
                        **{'uncalibrated_'+k:float(block['uncalibrated_scores'][i,j,z]) for z,k in enumerate(CLASSES)},
                        **{'actual_'+k:float(b[row,j,z]) for z,k in enumerate(CLASSES)}))
            write(out/'progress.json', dict(state='checkpointed', fold=fold+1, arm=arm, total_folds=4))
        completed.extend(index.tolist())
    metrics = {}
    for method in sorted({r['method'] for r in choices}):
        rows = [r for r in choices if r['method'] == method]; sid = [r['session_id'] for r in rows]
        metrics[method] = dict(interventions=sum(r['intervention'] for r in rows),
            **{k:int(sum(r[k] for r in rows)) for k in CLASSES},
            **{k:old.paired_interval([r[k] for r in rows], sid) for k in old.TARGETS[:5]})
    quality = {}
    for arm in old.ARMS:
        rows = [r for r in predictions if r['arm'] == arm]; sid = np.array([r['session_id'] for r in rows])
        quality[arm] = {}
        for k in CLASSES:
            p = np.array([r['prob_'+k] for r in rows]); y = np.array([r['actual_'+k] for r in rows])
            raw = np.array([r['uncalibrated_'+k] for r in rows])
            ap = [old.average_precision(y[sid==s],p[sid==s]) for s in sorted(set(sid))]
            bins = []
            for lo, hi in zip([0.,.01,.05,.1,.25,.5], [.01,.05,.1,.25,.5,1.000001]):
                mask = (p >= lo) & (p < hi)
                bins.append(dict(lower=lo, upper=min(hi,1.), actions=int(mask.sum()),
                    predicted_mean=float(p[mask].mean()) if mask.any() else None,
                    observed_fraction=float(y[mask].mean()) if mask.any() else None))
            quality[arm][k] = dict(brier_session_macro=float(np.mean([((p[sid==s]-y[sid==s])**2).mean() for s in sorted(set(sid))])),
                brier_uncalibrated_session_macro=float(np.mean([((raw[sid==s]-y[sid==s])**2).mean() for s in sorted(set(sid))])),
                ap_session_macro=float(np.mean([v for v in ap if v is not None])) if any(v is not None for v in ap) else None,
                sessions_with_positives=sum(v is not None for v in ap), reliability_bins_descriptive=bins)
    storage.csv_gz(out/'per_scene.csv.gz', choices); storage.csv_gz(out/'per_candidate.csv.gz', predictions)
    contrasts = {}
    for control in ('ds','ds_drex_text','ds_drex_blank','ds_drex_shuffled'):
        a={r['token']:r for r in choices if r['method']=='ds_drex_road/gated'}
        b_rows={r['token']:r for r in choices if r['method']==control+'/gated'}
        ordered=sorted(a); sid=[a[t]['session_id'] for t in ordered]
        contrasts['road_minus_'+control]={k:old.paired_interval([a[t][k]-b_rows[t][k] for t in ordered],sid)
            for k in ('benefit','raw_harm','geometry_harm','progress_violation','jerk_gain_mps3','lateral_gain_mps2','progress_delta_m','pdm_gain_pp')}
    report = dict(status='pilot_complete' if stage=='pilot' else 'complete', scenes=len(completed),
        sessions=len(set(sessions[completed])), config=CONFIG, selections=selections, metrics=metrics, quality=quality,
        baseline_DS=dict(interventions=0), primary_contrasts=contrasts,
        holdout_processed=0, new_drex_calls=0, note=NOTE)
    write(out/'report.json', report)
    write(out/'progress.json', dict(state=report['status'], completed_folds=1 if stage=='pilot' else 4))
    print(json.dumps(dict(status=report['status'], scenes=report['scenes'])), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--stage', choices=('pilot','full'),required=True)
    a=p.parse_args(); run(a.source.resolve(), a.out.resolve(), a.stage)
