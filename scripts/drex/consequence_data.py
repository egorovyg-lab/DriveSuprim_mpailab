"""Explicit separation between deployable table features and privileged labels."""
import numpy as np

import consequence_model as model

METRICS = ('no_at_fault_collisions', 'drivable_area_compliance', 'driving_direction_compliance',
           'traffic_light_compliance', 'ego_progress', 'time_to_collision_within_bound',
           'lane_keeping', 'history_comfort')
TABLE = ('ds_score_gap', *('pred_' + k for k in METRICS), 'imitation_log_probability_256',
         'x_1s_m', 'y_1s_m', 'x_2s_m', 'y_2s_m', 'x_4s_m', 'y_4s_m', 'heading_4s_deg',
         'length_m', 'max_speed_mps', 'max_abs_acceleration_mps2', 'max_abs_curvature_per_m')
FEATURES = tuple('delta_' + k for k in TABLE) + tuple('baseline_' + k for k in TABLE) + (
    'ego_velocity_x', 'ego_velocity_y', 'ego_acceleration_x', 'ego_acceleration_y')
PHYSICAL = ('route_progress_m', 'all_jerk_norm_mps3_rms', 'all_accel_lat_mps2_rms',
            'all_jerk_norm_mps3_abs_max', 'all_outside_area_max_m2', 'all_outside_area_integral_m2s',
            'all_outside_footprint_time_s', 'all_actor_contact_time_s',
            'all_actor_observed_interval_fraction', 'all_road_clearance_min_m',
            'all_actor_clearance_min_m')
OUTCOMES = ('pdm_score', *METRICS, *PHYSICAL)


def feature_block(sample):
    """Only the exact table and current ego motion seen by the frozen Drex run."""
    candidates = sample['candidates']
    if len(candidates) != 64 or sample['baseline_local'] != candidates[0]['local']:
        raise ValueError('Expected DS-first original top64')
    if len({c['local'] for c in candidates}) != 64 or len({c['vocab'] for c in candidates}) != 64:
        raise ValueError('Duplicate physical candidate')
    table = np.asarray([c['values'] for c in candidates], float)
    ego = np.asarray([*sample['ego_velocity'], *sample['ego_acceleration']], float)
    if table.shape != (64, len(TABLE)) or ego.shape != (4,) or not np.isfinite(table).all():
        raise ValueError('Unexpected numeric table schema')
    delta = table[1:] - table[0]
    j = TABLE.index('heading_4s_deg'); delta[:, j] = (delta[:, j] + 180.) % 360. - 180.
    x = np.concatenate((delta, np.broadcast_to(table[0], delta.shape), np.tile(ego, (63, 1))), axis=1)
    if not np.isfinite(x).all():
        raise ValueError('Invalid ego motion')
    return x


def preference_block(sample, pairs):
    lookup = {}
    for pair in pairs:
        key = (pair['candidate_local'], pair['mode'])
        if key in lookup:
            raise ValueError('Duplicate pair preference')
        lookup[key] = pair
    wanted = {(c['local'], mode) for c in sample['candidates'][1:] for mode in ('none', 'blank', 'road')}
    if set(lookup) != wanted:
        raise ValueError('Pair preference pool/mode mismatch')
    result = []
    for rank, c in enumerate(sample['candidates'][1:], 2):
        group = []
        for mode in ('none', 'blank', 'road'):
            row = lookup[(c['local'], mode)]
            if row['candidate_vocab'] != c['vocab'] or row['teacher_rank'] != rank:
                raise ValueError('Pair vocabulary/rank alignment changed')
            a, b = row['reference_first_preference'], row['challenger_first_preference']
            if not (np.isfinite([a, b]).all() and 0 <= a <= 1 and 0 <= b <= 1):
                raise ValueError('Invalid cached preference')
            if not np.isclose(row['mean_preference'], (a+b)/2, atol=1e-12, rtol=0):
                raise ValueError('Preference aggregation changed')
            group.append([a, b, (a+b)/2, abs(a-b)])
        result.append(group)
    return np.asarray(result)


def outcome_block(sample, session, rows):
    lookup = {int(r['candidate_local']): r for r in rows}
    if len(rows) != 64 or len(lookup) != 64 or set(lookup) != {c['local'] for c in sample['candidates']}:
        raise ValueError('Outcome top64 pool mismatch')
    output = []
    for rank, c in enumerate(sample['candidates'], 1):
        r = lookup[c['local']]
        if (int(r['candidate_vocab']), int(r['teacher_rank']), r['session_id']) != (c['vocab'], rank, session):
            raise ValueError('Outcome physical ID/session alignment changed')
        values = [float(r[k]) if r[k] not in ('', None) else np.nan for k in OUTCOMES]
        # Nullable clearances are evaluated only when both exist. Never fill them with zero.
        if not np.isfinite(values[:-2]).all() or any(not 0 <= v <= 1 for v in values[:9]):
            raise ValueError('Invalid finite outcome/raw metric')
        if any(np.isinf(values)):
            raise ValueError('Infinite clearance')
        output.append(values)
    return np.asarray(output)


def truth(outcomes):
    """Build training/evaluation targets. This function must not be called by a policy."""
    o = np.asarray(outcomes, float)
    if o.ndim != 3 or o.shape[1:] != (64, len(OUTCOMES)):
        raise ValueError('Expected full aligned outcome block')
    ix = {k: i for i, k in enumerate(OUTCOMES)}
    d = o[:, 1:] - o[:, :1]
    baseline = lambda k: np.broadcast_to(o[:, :1, ix[k]], d.shape[:2])
    change = lambda k: d[:, :, ix[k]]
    raw_harm = np.any(d[:, :, [ix[m] for m in METRICS if m != 'ego_progress']] < -1e-8, axis=2)
    progress_loss = change('route_progress_m') < -np.minimum(
        model.CONFIG['progress_loss_m'], model.CONFIG['progress_loss_fraction'] * baseline('route_progress_m')) - 1e-8
    exposure_harm = np.any(np.stack([change(k) > 1e-8 for k in (
        'all_outside_area_max_m2', 'all_outside_area_integral_m2s', 'all_outside_footprint_time_s',
        'all_actor_contact_time_s')]), axis=0)
    observed = (o[:, 1:, ix['all_actor_observed_interval_fraction']] >= 1-1e-8) & (
        baseline('all_actor_observed_interval_fraction') >= 1-1e-8)
    margin_loss = np.zeros(d.shape[:2], bool)
    for key, tolerance in [('all_road_clearance_min_m', model.CONFIG['road_margin_loss_m']),
                           ('all_actor_clearance_min_m', model.CONFIG['actor_margin_loss_m'])]:
        # Missing clearances remain missing; no "safe margin" inferred from absence.
        margin_loss |= np.isfinite(change(key)) & (change(key) < -tolerance - 1e-8)
    harm = raw_harm | progress_loss | exposure_harm | margin_loss | ~observed
    jerk = -change('all_jerk_norm_mps3_rms')
    lateral = -change('all_accel_lat_mps2_rms')
    peak = change('all_jerk_norm_mps3_abs_max')
    meaningful_jerk = ((jerk >= model.CONFIG['jerk_gain_min'])
                       & (jerk >= model.CONFIG['jerk_relative_min'] * baseline('all_jerk_norm_mps3_rms')))
    meaningful_lateral = ((lateral >= model.CONFIG['lateral_gain_min'])
                          & (lateral >= model.CONFIG['lateral_relative_min'] * baseline('all_accel_lat_mps2_rms')))
    benefit = ((meaningful_jerk | meaningful_lateral) & (jerk >= -1e-8) & (lateral >= -1e-8)
               & (peak <= 1e-8) & ~harm)
    y = np.stack([jerk, lateral, change('route_progress_m'),
                  100 * change('pdm_score'), peak, harm.astype(float), benefit.astype(float)], axis=2)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite target')
    flags = dict(raw_component_harm=raw_harm, progress_budget_exceeded=progress_loss,
                 exposure_worse=exposure_harm, margin_loss=margin_loss, incomplete_actor_coverage=~observed,
                 road_margin_unavailable=~np.isfinite(change('all_road_clearance_min_m')),
                 actor_margin_unavailable=~np.isfinite(change('all_actor_clearance_min_m')))
    return y, flags
