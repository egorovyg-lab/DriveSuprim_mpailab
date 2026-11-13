"""Input-only DS-vs-challenger comparisons; all 63 alternatives, no labels."""
import hashlib
import io

import numpy as np

import real_data as base
import table_data as original
from table_factorial_data import blank_board

MODES = ('none', 'blank', 'road')
ORDERS = ('reference_first', 'challenger_first')
POLICIES = ('mean050', 'both050', 'both075')
CONFIG = dict(protocol='replacement_star_v1', pool=64, challengers=63, modes=list(MODES),
    orders=list(ORDERS), primary='none/both050', policies=list(POLICIES), seed=original.CONFIG['seed'],
    max_length=8192, max_image_pixels=1536*640, bootstrap_reps=2000, improvement_min=.005)
NOTE = ('Exploratory previously inspected TRAIN cohort, not independent validation. '
    'All 63 top64 alternatives compared against the fixed DS choice; no adaptive tournament. '
    'Full DS scores, component heads and imitation prior retained. Deltas are rounded nominal '
    'geometry/prediction differences, not measured outcomes or rollouts. No map, future actors, '
    'new labels, training or holdout. Both-order agreement and preference thresholds are uncalibrated '
    'heuristics, not a safety guarantee. Strongest preference against DS does not establish a global '
    'ordering of challengers. Raw local PDM, not full EPDMS.')


def challengers(sample):
    pool = sample['candidates']
    if len(pool) != 64 or pool[0]['local'] != sample['baseline_local']:
        raise ValueError('Expected fixed DS-first top64')
    if len({c['local'] for c in pool}) != 64 or len({c['vocab'] for c in pool}) != 64:
        raise ValueError('Duplicate physical candidates')
    # Pilot covers the nearest rank, far tail and middle before any full pass.
    return [pool[i] for i in (1, 63, 31, *(i for i in range(1,64) if i not in (1,63,31)))]


def views(photo):
    images = dict(road=photo, blank=blank_board(photo)); hashes = {}
    for mode, image in images.items():
        buffer = io.BytesIO(); image.save(buffer, format='PNG')
        hashes[mode] = hashlib.sha256(buffer.getvalue()).hexdigest()
    return images, hashes


def payload(token, sample, local, mode, order, prepared_views):
    if mode not in MODES or order not in ORDERS:
        raise ValueError('Unknown pair condition/order')
    pool = sample['candidates']; challengers(sample)
    reference = pool[0]; challenger = next(c for c in pool if c['local'] == local)
    if local == reference['local']: raise ValueError('Self-comparison is not a challenge')
    names = original.aliases(token, sample)
    columns = original.COLUMNS[1:]
    delta = [round(float(b-a),4) for a,b in zip(reference['values'], challenger['values'])]
    heading = columns.index('heading_4s_deg')
    delta[heading] = round((delta[heading]+180.)%360.-180.,4)
    selected = [reference, challenger] if order == ORDERS[0] else [challenger, reference]
    row = dict(state=dict(
        task='Assess one concrete replacement of the current DriveSuprim (DS) choice.',
        coordinates='Current rear axle origin; x forward, y left, heading counterclockwise. SI units.',
        reference_id=names[reference['local']], challenger_id=names[local],
        current_velocity_xy_mps=sample['ego_velocity'], current_acceleration_xy_mps2=sample['ego_acceleration'],
        predictions='pred_* are sigmoid DS head estimates, not calibrated risk or measured outcomes. '
            'ds_score_gap is relative to the largest DS score; larger is preferred by DS. '
            'Imitation log probability is normalized over the original 256 candidates.',
        geometry='All paths and derived maxima are NOMINAL. x/y at 1,2,4 s; acceleration includes '
            'the transition from current speed; curvature excludes position steps below 0.05 m. '
            'No actor predictions, route command, map clearance or controller simulation.',
        candidates=dict(columns=original.COLUMNS, rows=[[names[c['local']],*c['values']] for c in selected]),
        change_if_replaced=dict(reference_id=names[reference['local']], challenger_id=names[local],
            definition='challenger minus reference; signed heading difference wrapped to [-180,180). '
                'A positive head delta only means DS predicts a larger value. Smaller absolute '
                'acceleration/curvature maxima are not measured comfort.',
            columns=columns, values=delta),
        optional_photograph='If attached: previous frame above current frame, columns front-left/front/front-right. '
            'Views are not stitched. A blank board supplies no road evidence.'),
        question=dict(type='choice', criteria={names[c['local']]:'Candidate '+names[c['local']] for c in selected},
            instructions='Choose the better of these two available trajectories. Prioritize avoiding collisions, '
                'staying on the road, obeying lights and direction, then useful progress, lane keeping and comfort. '
                'Compare both benefits and disadvantages of replacement. DS estimates are uncertain evidence. '
                'Unknown road/actor information is not evidence of a clear path. Both candidates can be poor. '
                'Return one supplied candidate ID.'))
    pixels = []
    if mode != 'none':
        images, hashes = prepared_views; row['images'] = [images[mode]]; pixels = [hashes[mode]]
    order_ids = [c['local'] for c in selected]
    digest = base.digest(dict(row={k:v for k,v in row.items() if k != 'images'}, pixels=pixels, order=order_ids))
    return row, order_ids, digest


def tasks(records, samples):
    return [(r['token'], c['local'], mode, order) for r in records
        for c in challengers(samples[r['token']]) for mode in MODES for order in ORDERS]


def name(token, local, mode, order):
    return f'{token}-c{local}-{mode}-{order}.json'


def select(pairs, baseline, policy):
    if policy not in POLICIES: raise ValueError('Unknown policy')
    threshold = .75 if policy == 'both075' else .5
    support = lambda p: p['mean_preference'] if policy == 'mean050' else p['minimum_preference']
    allowed = [p for p in pairs if support(p) > threshold]
    # Tie on support: use original DS rank, never a measured outcome.
    if not allowed: return baseline
    return min(allowed, key=lambda p:(-support(p), p['teacher_rank']))['candidate_local']


def preference(probabilities, order_ids, challenger):
    p = np.asarray(probabilities, dtype=float)
    if (p.shape != (2,) or len(order_ids) != 2 or len(set(order_ids)) != 2 or challenger not in order_ids
            or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or abs(float(p.sum())-1.)>1e-5):
        raise ValueError('Invalid physical pair alignment')
    return float(p[order_ids.index(challenger)])
