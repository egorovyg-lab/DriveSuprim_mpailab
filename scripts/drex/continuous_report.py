"""Paired continuous diagnostics; never rerank candidates using measured outcomes."""
import csv
import numpy as np

import continuous_metrics as metrics
import real_data as base
from transfer import write

SUMMARY_METRICS = ('route_progress_m', 'all_mean_speed_mps', 'all_stationary_fraction',
    'all_road_clearance_min_m', 'late_road_clearance_min_m',
    'all_outside_area_max_m2', 'all_outside_area_integral_m2s', 'all_outside_footprint_time_s',
    'all_actor_clearance_min_m', 'late_actor_clearance_min_m', 'all_actor_contact_time_s',
    'all_actor_observed_interval_fraction', 'late_actor_observed_interval_fraction',
    'all_accel_lon_mps2_rms', 'all_accel_lon_mps2_abs_max', 'all_accel_lat_mps2_rms',
    'all_accel_lat_mps2_abs_max', 'all_jerk_norm_mps3_rms', 'all_jerk_norm_mps3_abs_max',
    'late_jerk_norm_mps3_rms', 'all_jerk_norm_mps3_rms_moving',
    'all_yaw_rate_radps_abs_max', 'all_yaw_accel_radps2_abs_max',
    'all_moving_time_s', 'all_accel_interval_time_s', 'all_jerk_interval_time_s',
    'all_moving_jerk_interval_time_s')


def csv_file(path, rows):
    with path.with_suffix('.csv.tmp').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    path.with_suffix('.csv.tmp').replace(path)


def paired_rows(request, summaries):
    rows = []; safety = [base.METRICS.index(m)+1 for m in (*base.SAFETY, 'lane_keeping')]
    for r in request['records']:
        token = r['token']; b = summaries[token, r['methods']['baseline']]
        for method in metrics.METHODS:
            j = r['methods'][method]; c = summaries[token, j]
            delta = np.asarray(c['measured_raw'])-np.asarray(b['measured_raw'])
            row = dict(token=token, session_id=r['session_id'], method=method, candidate_local=j,
                candidate_vocab=c['candidate_vocab'], teacher_rank=c['teacher_rank'],
                intervention=j != r['methods']['baseline'], raw_pdm=c['measured_raw'][0],
                gain_pp=float(delta[0]*100), raw_safety_and_lane_not_worse=bool(np.all(delta[safety] >= -1e-8)),
                **{'raw_delta_'+m: float(delta[i+1]) for i, m in enumerate(base.METRICS)})
            lost = max(0., b['metrics']['route_progress_m']-c['metrics']['route_progress_m'])
            row['route_progress_loss_m'] = lost
            row['route_progress_loss_fraction'] = (lost/b['metrics']['route_progress_m']
                if b['metrics']['route_progress_m'] > 1e-8 else None)
            for k, v in c['metrics'].items():
                row[k] = v
                # Availability must match for a paired delta; no zero imputation.
                other = b['metrics'][k]
                row['delta_'+k] = None if v is None or other is None else float(v)-float(other)
            row['_tradeoff'] = metrics.tradeoff(c['metrics'], b['metrics'], row['raw_safety_and_lane_not_worse'], row['intervention'])
            rows.append(row)
    return rows


def report(request, summaries, out, stage):
    total = len(request['candidates']); expected = {(c['token'], c['candidate_local']) for c in request['candidates']}
    if not set(summaries) <= expected: raise ValueError('Unexpected completed candidates')
    complete = set(summaries) == expected
    pilot_expected = {(c['token'], c['candidate_local']) for c in request['candidates'] if c['token'] in request['pilot_tokens']}
    pilot_ready = pilot_expected <= set(summaries)
    result = dict(status='complete' if complete else 'pilot_complete' if stage == 'pilot' and pilot_ready else 'partial',
        scenes=len(request['records']), sessions=len({r['session_id'] for r in request['records']}),
        required_candidates=total, completed_candidates=len(summaries), metric_parity_passed=len(summaries),
        pilot_candidates=len(pilot_expected), pilot_ready=pilot_ready, methods=list(metrics.METHODS),
        geometry_interpretation_ready=complete, model_inference_calls=0, holdout_processed=0,
        config=metrics.CONFIG, note=metrics.NOTE)
    if not complete:
        write(out/'report.json', result); return result
    rows = paired_rows(request, summaries)
    candidate_rows = [dict(token=t, candidate_local=j, candidate_vocab=c['candidate_vocab'],
        raw_pdm=c['measured_raw'][0], **c['metrics']) for (t, j), c in sorted(summaries.items())]
    csv_file(out/'per_candidate.csv', candidate_rows)
    csv_file(out/'per_scene.csv', [{k: v for k, v in row.items() if k != '_tradeoff'} for row in rows])
    methods = {}
    for method in metrics.METHODS[1:]:
        group = [r for r in rows if r['method'] == method]; changed = [r for r in group if r['intervention']]
        def stats(g, key): return metrics.paired_stats([r[key] for r in g], [r['session_id'] for r in g])
        tradeoffs = {}
        for budget in metrics.CONFIG['progress_loss_budgets_m']:
            eligible = [r['_tradeoff'][str(budget)] for r in group if r['_tradeoff'][str(budget)]['eligible']]
            tradeoffs[str(budget)] = dict(eligible_interventions=len(eligible),
                **{name: dict(measurable=sum(x[name] is not None for x in eligible),
                              improved=sum(x[name] is True for x in eligible))
                   for name in ('road_clearance_min_m', 'actor_clearance_min_m', 'jerk_norm_mps3_rms')})
        methods[method] = dict(interventions=len(changed), gain_pp=stats(group, 'gain_pp'),
            component_degradations={m: sum(r['raw_delta_'+m] < -1e-8 for r in group) for m in base.METRICS},
            deltas_all_scenes={k: stats(group, 'delta_'+k) for k in SUMMARY_METRICS},
            deltas_interventions={k: stats(changed, 'delta_'+k) for k in SUMMARY_METRICS},
            progress_budget_tradeoffs=tradeoffs)
    result['comparisons_vs_DS'] = methods
    result['overview'] = [dict(method=m, interventions=v['interventions'],
        pdm_gain_macro_pp=v['gain_pp']['macro_mean'],
        progress_delta_macro_m=v['deltas_all_scenes']['route_progress_m']['macro_mean'],
        jerk_rms_delta_macro_mps3=v['deltas_all_scenes']['all_jerk_norm_mps3_rms']['macro_mean'],
        lateral_accel_rms_delta_macro_mps2=v['deltas_all_scenes']['all_accel_lat_mps2_rms']['macro_mean'],
        late_road_clearance_delta_macro_m=v['deltas_all_scenes']['late_road_clearance_min_m']['macro_mean'],
        late_road_clearance_paired_scenes=v['deltas_all_scenes']['late_road_clearance_min_m']['paired_scenes'])
        for m, v in methods.items()]
    result['statistics_note'] = ('All deltas are candidate minus DS in the named units; positive distance/progress '
        'and negative acceleration/jerk/exposure are favorable on that axis only. Null minima are excluded '
        'pairwise and paired coverage is reported, not imputed. Actor minima cover observed frames; inspect '
        'coverage fractions. CI resamples session means (2000 draws); intervention-only and missing-value '
        'summaries use their own available sessions. These are descriptive, unadjusted multiple comparisons. '
        'Progress tables require an actual intervention, no drop in five raw safety scores or lane keeping, '
        'no increase in offroad area maximum/integral, offroad/contact duration, and the stated route-progress metre budget. '
        'Benefits on different axes are separate counts, not a combined safety or comfort win.')
    write(out/'report.json', result); return result
