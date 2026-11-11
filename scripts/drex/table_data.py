"""Input-only top16/32/64 tables. No map, measured outcomes or future actors."""
import io
import hashlib
import re

import numpy as np

import real_data as base
from transfer import checked, sha

KS = (16, 32, 64)
MODES = ('table', 'table_camera')
ORDERS = ('canonical', 'reversed', 'permuted')
CONFIG = dict(protocol='table_topk_v1', ks=list(KS), modes=list(MODES), orders=list(ORDERS),
              seed=20261112, max_length=16384, max_image_pixels=1536*640,
              precision='bf16', bootstrap_reps=2000, primary='table_camera/k64/mean3')
HEAD_KEYS = tuple('head_'+k for k in (*base.METRICS, 'imi'))
COLUMNS = ['id', 'ds_score_gap', *('pred_'+k for k in base.METRICS), 'imitation_log_probability_256',
           'x_1s_m', 'y_1s_m', 'x_2s_m', 'y_2s_m', 'x_4s_m', 'y_4s_m',
           'heading_4s_deg', 'length_m', 'max_speed_mps', 'max_abs_acceleration_mps2',
           'max_abs_curvature_per_m']
NOTE = ('Exploratory replay of previously labelled train scenes, not independent validation. '
        'Tables contain DS predictions and finite-difference NOMINAL vocabulary geometry, not '
        'controller rollouts, map clearance, measured PDM or predicted actor trajectories. '
        'No HD map, radar, future observations, fitting or fresh holdout. Navigation command '
        'semantics remain undecoded and are not supplied. Model probabilities are not calibrated risk. '
        'Raw local PDM is not full EPDMS. All 16/32/64 decisions must commit before outcome lookup.')


def inputs(arrays, vocab):
    # base's reader accesses only candidate IDs, selection scores and ego status.
    s = base.inputs_from_arrays(arrays, vocab)
    ids = np.asarray(arrays['candidate_ids'])
    scores = np.asarray(arrays['selection_scores'], dtype=float)
    logits = {k: np.asarray(arrays['head_'+k], dtype=float) for k in (*base.METRICS, 'imi')}
    if any(a.shape != (256,) or not np.isfinite(a).all() for a in logits.values()):
        raise ValueError('Expected finite, aligned 256-candidate DS logits')
    local = np.argsort(-scores, kind='stable')[:64]
    probability = {k: np.exp(-np.logaddexp(0., -logits[k])) for k in base.METRICS}
    imi = logits['imi'] - np.max(logits['imi'])
    imi -= np.log(np.exp(imi).sum())
    speed0 = float(np.linalg.norm(s['ego_velocity']))
    rows = []
    for j in local:
        poses = np.asarray(vocab[ids[j]], dtype=float)
        xy = np.vstack((np.zeros((1, 2)), poses[:, :2]))
        ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        speeds = ds/.1
        acceleration = np.diff(np.r_[speed0, speeds])/.1
        dh = np.diff(np.unwrap(np.r_[0., poses[:, 2]]))
        # Heading/distance near a stationary point is not a reliable curvature estimate.
        moving = ds >= .05
        curvature = float(np.max(np.abs(dh[moving]/ds[moving]))) if moving.any() else 0.
        geometry = [*poses[9, :2], *poses[19, :2], *poses[39, :2],
                    np.degrees(np.arctan2(np.sin(poses[-1, 2]), np.cos(poses[-1, 2]))),
                    ds.sum(), speeds.max(), np.abs(acceleration).max(), curvature]
        values = [scores[j]-scores[s['baseline_local']],
                  *(probability[k][j] for k in base.METRICS), imi[j], *geometry]
        # Rules and Drex receive exactly the same rounded numeric features.
        values = [round(float(v), 4) for v in values]
        rows.append(dict(local=int(j), vocab=int(ids[j]), values=values))
    return dict(baseline_local=s['baseline_local'], candidates=rows,
                ego_velocity=s['ego_velocity'], ego_acceleration=s['ego_acceleration'])


def load_inputs(package, token, manifest, vocab):
    relative = 'bundle/examples/'+token+'.npz'
    path = checked(package, relative)
    if sha(path) != manifest['files'][relative]['sha256']:
        raise RuntimeError('Changed frozen example: '+token)
    with np.load(path, allow_pickle=False) as arrays:
        return inputs(arrays, vocab)


def cohort(index, name):
    if name == 'balanced64':
        return base.choose_records(index)
    if name != 'all':
        raise ValueError('Cohort must be all or balanced64')
    groups = {}
    for token, entry in index.items():
        m = entry['metadata']
        if m['role'] != 'train' or m['token'] != token or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',token):
            raise ValueError('Only matching train metadata allowed')
        groups.setdefault(m['session_id'], []).append(dict(m))
    # Interleave sessions so a short pilot does not visit only one session.
    rank = lambda value: base.digest([CONFIG['seed'], value])
    groups = [sorted(groups[s], key=lambda r: rank(r['token'])) for s in sorted(groups, key=rank)]
    return [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]


def order(token, k, kind):
    if k not in KS or kind not in ORDERS:
        raise ValueError('Unknown pool or order')
    ids = sorted(range(k), key=lambda j: base.digest([CONFIG['seed'], token, j, 'order']))
    if kind == 'reversed':
        ids.reverse()
    elif kind == 'permuted':
        ids = ids[5:]+ids[:5]
    return ids


def aliases(token, sample):
    # Anonymous identifiers remain stable across pool sizes and all presentation orders.
    ranked = sorted(sample['candidates'], key=lambda c: base.digest([token, c['vocab'], 'alias']))
    return {c['local']: f'C{i:02d}' for i, c in enumerate(ranked)}


def payload(token, sample, k, mode, kind, photo=None):
    if mode not in MODES:
        raise ValueError('Unknown input mode')
    permutation = order(token, k, kind)
    names = aliases(token, sample)
    candidates = sample['candidates'][:k]
    rows = [[names[candidates[i]['local']], *candidates[i]['values']] for i in permutation]
    row = dict(state=dict(
        task='Choose one four-second ego trajectory from the supplied candidates.',
        coordinates='Current rear axle origin; x forward, y left, heading counterclockwise. SI units.',
        predictions='pred_* are sigmoid DS head estimates, not measured outcomes or calibrated safety. '
                    'ds_score_gap is relative to the maximum DS score. Larger is preferred by DS.',
        geometry='NOMINAL poses at 1,2,4 seconds. Speed/acceleration use 0.1 s position differences; '
                 'acceleration includes transition from measured current speed. Curvature uses heading '
                 'change/distance only for steps >=0.05 m. No map clearance, actors or controller simulation.',
        optional_photograph='If attached: previous frame above current frame, columns front-left/front/front-right. '
                            'Views are not geometrically stitched. Otherwise no road photograph is supplied.',
        current_velocity_xy_mps=sample['ego_velocity'], current_acceleration_xy_mps2=sample['ego_acceleration'],
        candidates=dict(columns=COLUMNS, rows=rows)),
        question=dict(type='choice', criteria={r[0]: 'Candidate '+r[0] for r in rows},
            instructions='Choose the best available candidate. Prioritize avoiding collisions, staying on '
                         'the road, obeying traffic lights and driving direction, then useful progress, '
                         'lane keeping and comfort. Use the estimates as uncertain evidence. Unknown '
                         'road or actor information is not evidence of a clear path. Some or all candidates '
                         'can be poor. Return one of the supplied candidate IDs.'))
    pixels = []
    if mode == 'table_camera':
        if photo is None:
            raise ValueError('Camera condition requires a photograph')
        row['images'] = [photo]
        buffer = io.BytesIO(); photo.save(buffer, format='PNG')
        pixels.append(hashlib.sha256(buffer.getvalue()).hexdigest())
    digest = base.digest(dict(row={k:v for k,v in row.items() if k != 'images'},
                              pixels=pixels, order=permutation))
    return row, permutation, digest


def pick(p, sample, k):
    p = np.asarray(p, dtype=float)
    pool = sample['candidates'][:k]
    tied = np.flatnonzero(p == p.max()).tolist()
    return next((c['local'] for i,c in enumerate(pool) if i in tied and c['local']==sample['baseline_local']),
                pool[tied[0]]['local'])


def rules(sample, k):
    pool = sample['candidates'][:k]
    preds = np.asarray([c['values'][1:9] for c in pool])
    # A declared, uncalibrated head-only heuristic, deliberately not named PDM.
    utility = np.prod(preds[:, :4], axis=1)*preds[:, 4:].mean(axis=1)
    b = next(i for i,c in enumerate(pool) if c['local']==sample['baseline_local'])
    safety = [base.METRICS.index(m) for m in base.SAFETY]
    allowed = np.all(preds[:, safety] >= preds[b, safety], axis=1)
    veto = np.where(allowed, utility, -np.inf)
    return dict(baseline=sample['baseline_local'], head_rule=pick(utility, sample, k),
                head_veto_rule=pick(veto, sample, k))


def task_name(token, k, mode, kind):
    # Token has already passed the portable package's path validation.
    return f'{token}-k{k}-{mode}-{kind}.json'


def tasks(records):
    return [(r['token'], k, mode, kind) for r in records for k in KS for mode in MODES for kind in ORDERS]


def requested_candidates(request):
    return [dict(token=g['token'],candidate_local=j,candidate_vocab=v)
            for g in request['scenes'] for j,v in g['candidates']]
