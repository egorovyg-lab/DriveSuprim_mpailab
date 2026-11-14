"""Predeclared TRAIN comfort oracles; measured futures are evaluation-only."""
import csv
import gzip
import json
import re
import numpy as np

import continuous_metrics as cm
import real_data as base
from transfer import write

CONFIG = dict(version=1, pools=[16, 32, 64], progress_budgets=[[0., 0.], [.5, .05], [1., .05], [2., .10]],
    guards=['raw_only', 'geometry_peaks'], objectives=['jerk', 'lateral'], primary='top64/1m_5pct/geometry_peaks/jerk',
    epsilon=1e-8, jerk_improvement_absolute=.2, jerk_improvement_relative=.2,
    lateral_improvement_absolute=.05, lateral_improvement_relative=.2)
NOTE = ('Outcome-based upper bounds on the already inspected TRAIN cohort. Not deployable decisions, '
    'independent validation, calibrated safety or full EPDMS. Fixed original DS top16/32/64 pools, '
    'same simulator/continuous metric definitions. Future actors, map geometry and measured comfort '
    'never enter Drex. raw_only is a permissive reference, NOT a safety claim. geometry_peaks adds '
    'sampled geometric and comfort guards. All horizons are four seconds; no reactive traffic or '
    'continuous swept-volume guarantee. Missing comparisons fall back to DS. No model calls or holdout.')


def validate(request):
    if request['protocol'] != 'continuous_pool_v1' or request['config'] != CONFIG or request['metric_config'] != cm.CONFIG:
        raise ValueError('Pool protocol changed')
    records, candidates = request['records'], request['candidates']
    if not re.fullmatch(r'rollout_audit_[A-Za-z0-9_-]{1,64}', request['reuse']['name']):
        raise ValueError('Invalid reuse name')
    tokens = [r['token'] for r in records]
    if not tokens or len(set(tokens)) != len(tokens) or any(not re.fullmatch('[0-9a-f]{16}', t) for t in tokens):
        raise ValueError('Duplicate/invalid/empty cohort')
    grouped = {t: [] for t in tokens}
    for c in candidates:
        if c['token'] not in grouped: raise ValueError('Unexpected candidate token')
        grouped[c['token']].append(c)
        raw = np.asarray(c['expected_raw'], float)
        if (c['metric_keys'] != ['pdm_score', *base.METRICS] or raw.shape != (9,)
                or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1))
                or not isinstance(c['candidate_local'], int) or not 0 <= c['candidate_local'] < 256
                or not isinstance(c['candidate_vocab'], int) or not 0 <= c['candidate_vocab'] < 8192):
            raise ValueError('Invalid candidate/label')
    for r in records:
        group = grouped[r['token']]
        if (len(group) != 64 or {c['teacher_rank'] for c in group} != set(range(1, 65))
                or len({c['candidate_local'] for c in group}) != 64
                or len({c['candidate_vocab'] for c in group}) != 64
                or next(c for c in group if c['teacher_rank'] == 1)['candidate_local'] != r['baseline_local']):
            raise ValueError('Complete, unique ranked top64 including DS required')
    pilot = request['pilot_tokens']
    if not set(pilot) <= set(tokens) or len(pilot) != min(4, len({r['session_id'] for r in records})):
        raise ValueError('Invalid pilot')
    if len({r['session_id'] for r in records if r['token'] in pilot}) != len(pilot):
        raise ValueError('Pilot sessions must be distinct')


def available_not_lower(a, b):
    return a is not None and b is not None and a >= b-CONFIG['epsilon']


def eligible(c, b, metres, fraction, guard, objective):
    if c['candidate_local'] == b['candidate_local']: return True
    eps = CONFIG['epsilon']; a, ref = c['metrics'], b['metrics']
    loss = max(0., ref['route_progress_m']-a['route_progress_m'])
    if loss > min(metres, fraction*ref['route_progress_m'])+eps: return False
    indices = [1+base.METRICS.index(m) for m in (*base.SAFETY, 'lane_keeping', 'history_comfort')]
    if np.any((np.asarray(c['measured_raw'])-np.asarray(b['measured_raw']))[indices] < -eps): return False
    if guard == 'raw_only': return True
    if guard != 'geometry_peaks': raise ValueError('Unknown guard')
    smaller = ['all_outside_area_max_m2', 'all_outside_area_integral_m2s', 'all_outside_footprint_time_s',
        'all_actor_contact_time_s', 'all_accel_lon_mps2_abs_max', 'all_accel_lat_mps2_abs_max',
        'all_yaw_rate_radps_abs_max', 'all_yaw_accel_radps2_abs_max', 'all_jerk_norm_mps3_abs_max',
        'all_accel_lon_mps2_rms', 'all_jerk_norm_mps3_rms' if objective == 'lateral' else 'all_accel_lat_mps2_rms']
    if any(a[k] is None or ref[k] is None or a[k] > ref[k]+eps for k in smaller): return False
    # A positive clearance cannot be exchanged for an offroad footprint. Repairing
    # an offroad baseline is allowed only by reaching full footprint coverage.
    for prefix in ('all_', 'late_'):
        x, y = a[prefix+'road_clearance_min_m'], ref[prefix+'road_clearance_min_m']
        if x is None or (y is not None and x < y-eps): return False
        if not (a[prefix+'actor_observed_interval_fraction'] == ref[prefix+'actor_observed_interval_fraction'] == 1.):
            return False
        if not available_not_lower(a[prefix+'actor_clearance_min_m'], ref[prefix+'actor_clearance_min_m']): return False
    return True


def pick(group, baseline, k, metres, fraction, guard, objective):
    b = next(c for c in group if c['candidate_local'] == baseline)
    key = 'all_jerk_norm_mps3_rms' if objective == 'jerk' else 'all_accel_lat_mps2_rms'
    choices = [c for c in group if c['teacher_rank'] <= k and eligible(c, b, metres, fraction, guard, objective)]
    best = min(choices, key=lambda c: (c['metrics'][key], c['teacher_rank']))
    return b if best['metrics'][key] >= b['metrics'][key]-CONFIG['epsilon'] else best


def specs():
    for k in CONFIG['pools']:
        for metres, frac in CONFIG['progress_budgets']:
            for guard in CONFIG['guards']:
                for objective in CONFIG['objectives']:
                    name = f'top{k}/{metres:g}m_{frac*100:g}pct/{guard}/{objective}'
                    yield name, (k, metres, frac, guard, objective)


def write_gzip_csv(path, rows):
    temp = path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temp, 'wt', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    temp.replace(path)


def report(request, groups, out, complete):
    required = len(request['candidates']); done = sum(map(len, groups.values()))
    value = dict(status='complete' if complete else 'pilot_complete', scenes=len(request['records']),
        sessions=len({r['session_id'] for r in request['records']}), required_candidates=required,
        completed_candidates=done, metric_parity_passed=done, primary=CONFIG['primary'],
        model_inference_calls=0, holdout_processed=0, config=CONFIG, note=NOTE)
    if not complete: write(out/'report.json', value); return value
    if done != required: raise RuntimeError('Incomplete pool cannot produce reserve estimates')
    rows = []; candidates = []; baselines = {}; choices = []
    keys = ['route_progress_m', 'all_jerk_norm_mps3_rms', 'all_jerk_norm_mps3_rms_moving',
        'all_accel_lat_mps2_rms', 'all_jerk_norm_mps3_abs_max', 'all_accel_lat_mps2_abs_max',
        'late_road_clearance_min_m', 'late_actor_clearance_min_m', 'all_actor_contact_time_s',
        'all_outside_area_integral_m2s']
    safety = [1+base.METRICS.index(m) for m in (*base.SAFETY, 'lane_keeping', 'history_comfort')]
    for r in request['records']:
        group = groups[r['token']]; b = next(c for c in group if c['candidate_local'] == r['baseline_local'])
        baselines[r['token']] = b
        for c in group:
            candidates.append(dict(token=r['token'], session_id=r['session_id'], candidate_local=c['candidate_local'],
                candidate_vocab=c['candidate_vocab'], teacher_rank=c['teacher_rank'],
                **dict(zip(['pdm_score', *base.METRICS], c['measured_raw'])), **c['metrics']))
        for name, args in specs():
            c = pick(group, r['baseline_local'], *args); a, ref = c['metrics'], b['metrics']
            raw_delta = np.asarray(c['measured_raw'])-np.asarray(b['measured_raw'])
            objective = args[-1]; key = 'all_jerk_norm_mps3_rms' if objective == 'jerk' else 'all_accel_lat_mps2_rms'
            absolute = CONFIG[objective+'_improvement_absolute']; relative = CONFIG[objective+'_improvement_relative']
            row = dict(token=r['token'], session_id=r['session_id'], method=name,
                candidate_local=c['candidate_local'], candidate_vocab=c['candidate_vocab'], teacher_rank=c['teacher_rank'],
                intervention=c['candidate_local'] != r['baseline_local'], raw_pdm=c['measured_raw'][0],
                gain_pp=float(100*raw_delta[0]), meaningful_improvement=bool(ref[key]-a[key] >= absolute and a[key] <= (1-relative)*ref[key]),
                raw_guard_harm=bool(np.any(raw_delta[safety] < -CONFIG['epsilon'])))
            for key in keys:
                row[key] = a[key]
                row['delta_'+key] = None if a[key] is None or ref[key] is None else a[key]-ref[key]
            rows.append(row)
            choices.append(dict(token=r['token'], method=name, candidate_local=c['candidate_local'], candidate_vocab=c['candidate_vocab']))
    write_gzip_csv(out/'per_candidate.csv.gz', candidates)
    write_gzip_csv(out/'per_scene.csv.gz', rows)
    write(out/'oracle_choices.json', choices)
    methods = {}
    for name, args in specs():
        group = [r for r in rows if r['method'] == name]
        stats = lambda k: cm.paired_stats([r[k] for r in group], [r['session_id'] for r in group])
        methods[name] = dict(interventions=sum(r['intervention'] for r in group),
            meaningful_improvements=sum(r['meaningful_improvement'] for r in group),
            meaningful_sessions=len({r['session_id'] for r in group if r['meaningful_improvement']}),
            tail_selections=sum(r['teacher_rank'] > 16 for r in group),
            raw_guard_harm=sum(r['raw_guard_harm'] for r in group), gain_pp=stats('gain_pp'),
            deltas={k: stats('delta_'+k) for k in keys})
    value['oracles'] = methods
    value['overview'] = [dict(method=n, interventions=v['interventions'], meaningful=v['meaningful_improvements'],
        meaningful_sessions=v['meaningful_sessions'], gain_macro_pp=v['gain_pp']['macro_mean'],
        progress_delta_macro_m=v['deltas']['route_progress_m']['macro_mean'],
        jerk_delta_macro_mps3=v['deltas']['all_jerk_norm_mps3_rms']['macro_mean'],
        lateral_delta_macro_mps2=v['deltas']['all_accel_lat_mps2_rms']['macro_mean'])
        for n, v in methods.items() if '/1m_5pct/geometry_peaks/' in n]
    value['statistics_note'] = ('Candidate-minus-DS deltas; session-macro bootstrap intervals with 2000 draws. '
        'Missing pairs excluded with coverage counts. Oracles use measured outcomes and are upper bounds, '
        'not predicted performance. Multiple objectives/budgets are predeclared, not independent tests. '
        'Absolute AND relative progress limits apply simultaneously. DS always remains a fallback. '
        'Primary: top64, <=1 m AND <=5% lost progress, geometry/peak guards, minimum jerk RMS.')
    write(out/'report.json', value); return value
