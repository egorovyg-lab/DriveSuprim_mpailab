"""Compare numeric replacement with and without cached pointwise Drex scores."""
import argparse
import json
from pathlib import Path

import numpy as np

import consequence_run as storage
import consequence_v2 as consequence
import pointwise
from continuous_metrics import paired_stats
from reason_replacement import CONFIG as OUTCOME_RULE
from transfer import freeze, read, sha, write


ARMS = ('ds', 'ds_drex_blank', 'ds_drex_road')
CONFIG = dict(protocol='pointwise_value_v1', pool=16, risk_ceiling=.10,
              inner_coverage=.10, coverage_caps=[.05, .10, .20])
NOTE = ('Session out-of-fold replay on previously inspected TRAIN. Matched coverage ranks '
        'unlabelled test scenes in each fold and is a diagnostic. The train-gated policy '
        'uses only thresholds fitted on inner held-out TRAIN sessions. No new Drex or scorer calls.')
FIELDS = ('jerk_gain_mps3', 'lateral_gain_mps2', 'progress_delta_m', 'pdm_gain_pp',
          'jerk_peak_delta_mps3', 'raw_harm', 'progress_violation', 'helpful')


def inputs(source, scores):
    request, features, _, y, flags = storage.context(source / 'audit_request.json')
    plan, report = read(scores / 'plan.json'), read(scores / 'report.json')
    records = request['records']
    expected = {r['token']: r['session_id'] for r in records}
    observed = {r['token']: r['session_id'] for r in plan['records'] if r['role'] == 'train'}
    binding = sha(scores / 'plan.json')
    if (plan['config'] != pointwise.CONFIG or report['status'] != 'choices_complete'
            or report['plan_sha256'] != binding or report['decisions_sha256'] != sha(scores / 'decisions.json')
            or report['outcomes_read'] is not False or report['scenes'] != len(records)
            or report['requests'] != len(records) * 128
            or len(records) != 651 or len(set(expected.values())) != 32
            or len(plan['records']) != len(records) or observed != expected or set(plan['entries']) != set(expected)):
        raise ValueError('Expected the same completed 651-scene pointwise TRAIN cohort')

    k = CONFIG['pool']
    margin = {m: np.empty((len(records), k - 1, 1)) for m in pointwise.MODES}
    provenance = {}
    for i, r in enumerate(records):
        token = r['token']
        entry = plan['entries'][token]
        if (entry['local'] != features['local_ids'][i].tolist()
                or entry['vocab'] != features['vocab_ids'][i].tolist()):
            raise ValueError('Candidate rank or physical ID mismatch: ' + token)
        rows = pointwise.checkpoint(scores, token, binding, entry)
        if set(rows) != {str(j) for j in range(64)}:
            raise ValueError('Incomplete pointwise scores: ' + token)
        for mode in margin:
            baseline = rows['0'][mode]['score']
            margin[mode][i, :, 0] = [rows[str(j)][mode]['score'] - baseline for j in range(1, k)]
        provenance[token] = sha(scores / 'scores' / f'{token}.json')
    x = features['x'][:, :k - 1]
    arms = {'ds': np.concatenate((x, np.zeros_like(margin['blank'])), axis=-1),
            **{'ds_drex_' + m: np.concatenate((x, v), axis=-1) for m, v in margin.items()}}
    gains = y[:, :k - 1, :5]
    harm = flags['raw_component_harm'][:, :k - 1]
    progress = flags['progress_budget_exceeded'][:, :k - 1]
    helpful = (gains[:, :, 3] >= 100 * (OUTCOME_RULE['minimum_pdm_gain'] - 1e-8)) & ~harm & ~progress
    events = np.stack((harm, progress, helpful), axis=-1).astype(float)
    return records, features, arms, gains, events, dict(pointwise_plan_sha256=binding,
        checkpoint_sha256=provenance, probe_request_sha256=sha(source / 'audit_request.json'))


def eligible_choices(prediction, probabilities):
    eligible = ((probabilities[:, :, :2].max(axis=-1) <= CONFIG['risk_ceiling'])
                & (prediction[:, :, 3] > 0))
    score = np.where(eligible, probabilities[:, :, 2], -np.inf)
    best = score.argmax(axis=1)
    return best, score[np.arange(len(best)), best]


def train_threshold(scores):
    """Choose a cutoff from inner predictions under a fixed intervention budget."""
    budget = int(CONFIG['inner_coverage'] * len(scores))
    eligible = np.sort(scores[np.isfinite(scores)])[::-1]
    return float(eligible[budget]) if len(eligible) > budget else 0.


def matched(scores, rate):
    budget = min(int(rate * len(scores[0])), *(int(np.isfinite(s).sum()) for s in scores))
    masks = []
    for strength in scores:
        mask = np.zeros(len(strength), bool)
        mask[np.argsort(-strength, kind='stable')[:budget]] = True
        masks.append(mask)
    return masks


def fit_fold(x, gains, events, sessions, train, test):
    inner_gain, inner_logits, _ = consequence.inner_predictions(x[train], gains[train], events[train], sessions[train])
    calibration, _ = consequence.calibrate(inner_logits, events[train], sessions[train])
    _, inner_score = eligible_choices(inner_gain, consequence.probabilities(inner_logits, calibration))
    heads = consequence.fit_heads(x[train], gains[train], events[train], sessions[train])
    pred, logits = consequence.predict_heads(heads, x[test])
    candidates, confidence = eligible_choices(pred, consequence.probabilities(logits, calibration))
    return candidates, confidence, train_threshold(inner_score)


def summarize(rows, sessions):
    return dict(interventions=int(rows[:, 0].sum()),
                **{name: int(rows[:, 1 + i].sum()) if i >= 5 else paired_stats(rows[:, 1 + i], sessions)
                   for i, name in enumerate(FIELDS)})


def run(source, scores, out, stage):
    records, features, arms, gains, events, inputs_id = inputs(source, scores)
    if stage == 'full' and not (out / 'fold_0.json').exists():
        raise ValueError('Run the pilot fold first')
    observed = np.concatenate((gains, events), axis=-1)
    sessions = np.array([r['session_id'] for r in records])
    sources = ('pointwise_value.py', 'pointwise.py', 'consequence_v2.py', 'consequence_run.py',
               'consequence_model.py', 'consequence_data.py', 'reason_replacement.py',
               'continuous_metrics.py', 'transfer.py')
    identity = dict(config=CONFIG, inputs=inputs_id, sources={
        n: sha(Path(__file__).with_name(n)) for n in sources}, numpy=np.__version__)
    freeze(out / 'identity.json', identity)
    binding = sha(out / 'identity.json')
    policies = ('train_gate', *(f'matched_{int(100 * c)}pct' for c in CONFIG['coverage_caps']))
    all_rows = {(p, a): [] for p in policies for a in ARMS}
    idx = []
    for fold in range(1 if stage == 'pilot' else 4):
        train, test = features['fold'] != fold, features['fold'] == fold
        test_idx = np.flatnonzero(test)
        if not train.any() or not test.any() or set(sessions[train]) & set(sessions[test]):
            raise ValueError('Invalid session split')
        path = out / f'fold_{fold}.npz'
        commit_path = path.with_suffix('.json')
        if commit_path.exists():
            if read(commit_path) != dict(identity_sha256=binding, predictions_sha256=sha(path)):
                raise ValueError('Changed fold checkpoint: ' + str(path))
            cached = storage.load_arrays(path)
        else:
            fit = [fit_fold(arms[a], gains, events, sessions, train, test) for a in ARMS]
            cached = dict(index=test_idx, choices=np.array([v[0] for v in fit]),
                          scores=np.array([v[1] for v in fit]), thresholds=np.array([v[2] for v in fit]))
            storage.save_arrays(path, **cached)
            freeze(commit_path, dict(identity_sha256=binding, predictions_sha256=sha(path)))
        choices, confidence, thresholds = cached['choices'], cached['scores'], cached['thresholds']
        if (not np.array_equal(cached['index'], test_idx)
                or choices.shape != confidence.shape or choices.shape != (len(ARMS), len(test_idx))
                or thresholds.shape != (len(ARMS),) or not np.isfinite(thresholds).all()
                or np.isnan(confidence).any() or (choices < 0).any() or (choices >= CONFIG['pool'] - 1).any()):
            raise ValueError('Invalid out-of-fold predictions')
        masks = {'train_gate': confidence > thresholds[:, None],
                 **{p: matched(confidence, rate) for p, rate in zip(policies[1:], CONFIG['coverage_caps'])}}
        idx.extend(test_idx)
        for p in policies:
            for i, a in enumerate(ARMS):
                selected = np.where(masks[p][i], choices[i], -1)
                actual = consequence.chosen_values(observed[test], selected)
                all_rows[p, a].extend(np.column_stack((selected >= 0, actual)))
        print(f'POINTWISE VALUE fold={fold + 1}/4 scenes={len(test_idx)}', flush=True)
    sid = sessions[idx]
    summary = {}
    contrasts = {}
    pdm_col = 1 + FIELDS.index('pdm_gain_pp')
    for p in policies:
        summary[p] = {a: summarize(np.asarray(all_rows[p, a]), sid) for a in ARMS}
        ds = np.asarray(all_rows[p, 'ds'])
        contrasts[p] = {a: paired_stats(np.asarray(all_rows[p, a])[:, pdm_col] - ds[:, pdm_col], sid)
                        for a in ARMS[1:]}
    result = dict(status='pilot_complete' if stage == 'pilot' else 'complete', scenes=len(idx),
                  sessions=len(set(sid)), config=CONFIG, metrics=summary, pdm_gain_pp_vs_numeric=contrasts,
                  new_drex_calls=0, holdout_processed=0, note=NOTE)
    write(out / ('pilot_report.json' if stage == 'pilot' else 'report.json'), result)
    print(json.dumps({k: result[k] for k in ('status', 'scenes', 'sessions')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'scores', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage', choices=('pilot', 'full'), required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.scores.resolve(), args.out.resolve(), args.stage)
