"""Fixed, unit-bearing diagnostics of sampled simulated motion (not a policy)."""
import numpy as np
import re

METHODS = ('baseline', 'none/both050', 'blank/both050', 'road/both050',
           'blank/both075', 'road/both075', 'old_table', 'old_camera')
CONFIG = dict(version=1, dt_s=.1, samples=41, late_start_s=1., moving_speed_mps=.5,
              road_buffers_m=[.1, .25, .5], actor_buffers_m=[.5, 1., 2.],
              progress_loss_budgets_m=[0., .5, 1., 2.],
              descriptive_road_improvement_m=.1, descriptive_actor_improvement_m=.25,
              descriptive_jerk_relative_reduction=.2, descriptive_jerk_absolute_reduction_mps3=.2,
              bootstrap_reps=2000, bootstrap_seed=20261114)
NOTE = ('Exploratory audit of previously evaluated TRAIN decisions, not independent validation or a new selector. '
        'Raw local PDM remains unchanged; not full EPDMS. Maps, reference centerline and future replayed actors '
        'are privileged evaluation data, never model inputs. Distances use sampled full scorer footprints, '
        'not exact swept volumes, collision probability or fault assignment. Traffic follows the original '
        'non-reactive policy, including its exclusion of initially overlapping objects. Comfort uses the '
        'simulated vehicle center velocity and unfiltered finite differences; it is distinct from thresholded history '
        'comfort. All-time and >=1 s summaries are both retained. No Drex calls, fitting or holdout. '
        'Progress-budget tables describe observed tradeoffs; they cannot be used as a deployable outcome-based gate.')


def validate_request(request):
    import real_data as base
    if (request['protocol'] != 'continuous_decisions_v1' or request['config'] != CONFIG
            or request['methods'] != list(METHODS)):
        raise ValueError('Unknown continuous-metric contract')
    records = request['records']; candidates = request['candidates']
    tokens = [r['token'] for r in records]
    if not tokens or len(set(tokens)) != len(tokens) or any(not re.fullmatch('[0-9a-f]{16}', t) for t in tokens):
        raise ValueError('Invalid/duplicate cohort tokens')
    if len({r['session_id'] for r in records}) < 2: raise ValueError('At least two sessions required')
    keys = [(c['token'], c['candidate_local']) for c in candidates]
    if len(keys) != len(set(keys)): raise ValueError('Duplicate candidate')
    wanted = set()
    for r in records:
        if set(r['methods']) != set(METHODS): raise ValueError('Incomplete decision methods')
        wanted.update((r['token'], j) for j in r['methods'].values())
    if set(keys) != wanted: raise ValueError('Candidate union differs from frozen decisions')
    for c in candidates:
        raw = np.asarray(c['expected_raw'], float)
        if (not isinstance(c['candidate_local'], int) or not 0 <= c['candidate_local'] < 256
                or not isinstance(c['candidate_vocab'], int) or not 0 <= c['candidate_vocab'] < 8192
                or not 1 <= c['teacher_rank'] <= 64
                or c['metric_keys'] != ['pdm_score', *base.METRICS] or raw.shape != (9,)
                or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1))):
            raise ValueError('Invalid candidate/metric alignment')
    pilot = request['pilot_tokens']
    if len(pilot) != min(4, len({r['session_id'] for r in records})) or not set(pilot) <= set(tokens):
        raise ValueError('Invalid pilot cohort')
    if len({r['session_id'] for r in records if r['token'] in pilot}) != len(pilot):
        raise ValueError('Pilot must span distinct sessions')


def integral(values, times):
    values, times = np.asarray(values, float), np.asarray(times, float)
    if len(times) < 2 or values.shape != times.shape or not np.isfinite(values).all():
        raise ValueError('Finite sampled signal on at least two times required')
    return float(np.sum((values[:-1]+values[1:])*.5*np.diff(times)))


def weighted_summary(values, weights):
    """Time-weighted RMS, sample absolute p95 and peak; null for no coverage."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if values.shape != weights.shape or not np.isfinite(values).all() or np.any(weights < 0):
        raise ValueError('Invalid signal/weights')
    keep = weights > 0
    if not keep.any(): return dict(rms=None, abs_p95=None, abs_max=None)
    values, weights = np.abs(values[keep]), weights[keep]
    return dict(rms=float(np.sqrt(np.average(values**2, weights=weights))),
                abs_p95=float(np.quantile(values, .95)), abs_max=float(values.max()))


def kinematics(center_states, dt, start_s=0.):
    """PDM dynamic components are BODY-frame vectors; positions/headings are global.

    Differentiate WORLD-frame center velocity over intervals, then differentiate
    acceleration over adjacent intervals. The simulator's stored body acceleration
    omits the v*omega centripetal term; it is NOT physical lateral acceleration.
    Acceleration/jerk use interval/midpoint weights without fabricated endpoints.
    """
    s = np.asarray(center_states, float)
    if s.shape != (CONFIG['samples'], 11) or not np.isfinite(s).all() or abs(dt-CONFIG['dt_s']) > 1e-12:
        raise ValueError('Expected finite PDM (41,11) states at 0.1 s')
    t = np.arange(len(s))*dt
    take = t >= start_s-1e-10
    if take.sum() < 2: raise ValueError('Empty/short time window')
    weights = np.full(take.sum(), dt); weights[[0, -1]] *= .5
    speed = np.hypot(s[:, 3], s[:, 4]); moving = speed >= CONFIG['moving_speed_mps']
    theta = np.unwrap(s[:, 2]); cosine, sine = np.cos(theta), np.sin(theta)
    velocity_world = np.column_stack((cosine*s[:, 3]-sine*s[:, 4], sine*s[:, 3]+cosine*s[:, 4]))
    acceleration_world = np.diff(velocity_world, axis=0)/dt
    mid = (theta[:-1]+theta[1:])*.5
    accel_lon = np.cos(mid)*acceleration_world[:, 0]+np.sin(mid)*acceleration_world[:, 1]
    accel_lat = -np.sin(mid)*acceleration_world[:, 0]+np.cos(mid)*acceleration_world[:, 1]
    jerk_world = np.diff(acceleration_world, axis=0)/dt
    jerk_heading = theta[1:-1]
    jerk_lon = np.cos(jerk_heading)*jerk_world[:, 0]+np.sin(jerk_heading)*jerk_world[:, 1]
    jerk_lat = -np.sin(jerk_heading)*jerk_world[:, 0]+np.cos(jerk_heading)*jerk_world[:, 1]
    interval_take = take[:-1] & take[1:]
    interval_moving = moving[:-1] & moving[1:]
    jerk_take = interval_take[:-1] & interval_take[1:]
    jerk_moving = interval_moving[:-1] & interval_moving[1:]
    signals = {'yaw_rate_radps': s[:, 9], 'yaw_accel_radps2': s[:, 10]}
    result = dict(mean_speed_mps=integral(speed[take], t[take])/(t[take][-1]-t[take][0]),
                  stationary_fraction=integral((~moving[take]).astype(float), t[take])/(t[take][-1]-t[take][0]),
                  moving_time_s=float(np.sum(weights*moving[take])),
                  accel_interval_time_s=float(interval_take.sum()*dt),
                  jerk_interval_time_s=float(jerk_take.sum()*dt),
                  moving_jerk_interval_time_s=float(np.sum(jerk_take & jerk_moving)*dt))
    for name, signal in signals.items():
        for suffix, w in (('', weights), ('_moving', weights*moving[take])):
            for stat, value in weighted_summary(signal[take], w).items(): result[name+'_'+stat+suffix] = value
    derived = [('accel_lon_mps2', accel_lon, interval_take, interval_moving),
               ('accel_lat_mps2', accel_lat, interval_take, interval_moving),
               ('jerk_lon_mps3', jerk_lon, jerk_take, jerk_moving),
               ('jerk_lat_mps3', jerk_lat, jerk_take, jerk_moving),
               ('jerk_norm_mps3', np.linalg.norm(jerk_world, axis=1), jerk_take, jerk_moving)]
    for name, signal, selected, moving_selected in derived:
        for suffix, mask in (('', selected), ('_moving', selected & moving_selected)):
            for stat, value in weighted_summary(signal, mask.astype(float)*dt).items():
                result[name+'_'+stat+suffix] = value
    return result


def spatial(frames, dt, start_s=0.):
    """Minima have explicit coverage; offroad and absent actors are never safe zeros."""
    if len(frames) != CONFIG['samples'] or abs(dt-CONFIG['dt_s']) > 1e-12:
        raise ValueError('Expected 41 geometry frames at 0.1 s')
    first = int(round(start_s/dt)); group = frames[first:]; t = np.arange(len(group))*dt
    if len(group) < 2: raise ValueError('Short geometry window')
    duration = float(t[-1]); covered = np.array([f['full_footprint_covered'] for f in group], bool)
    outside = np.array([f['outside_area_m2'] for f in group], float)
    corner = np.array([f['min_corner_signed_margin_m'] for f in group], float)
    road = [f['road_clearance_m'] for f in group]; actor = [f['actor_clearance_m'] for f in group]
    if (not np.isfinite(outside).all() or (outside < 0).any() or not np.isfinite(corner).all()
            or any((r is None) == bool(c) for r, c in zip(road, covered))
            or any(v is not None and (not np.isfinite(v) or v < 0) for v in road+actor)):
        raise ValueError('Invalid coverage/distance trace')
    observed = np.array([a is not None for a in actor], bool)
    interval_observed = observed[:-1] & observed[1:]
    distances = np.array([0. if a is None else a for a in actor])
    result = dict(window_s=duration, road_covered_all_samples=bool(covered.all()),
        road_clearance_min_m=min(road) if covered.all() else None,
        corner_signed_margin_min_m=float(corner.min()), outside_area_max_m2=float(outside.max()),
        outside_area_integral_m2s=integral(outside, t),
        outside_footprint_time_s=integral((~covered).astype(float), t),
        actor_observed_interval_fraction=float(interval_observed.mean()),
        actor_clearance_min_m=min(a for a in actor if a is not None) if observed.any() else None,
        actor_contact_time_s=integral(np.array([bool(f['intersecting_actor_tokens']) for f in group], float), t))
    for b in CONFIG['road_buffers_m']:
        near = [(r is None or r < b) for r in road]
        result['road_below_'+str(b)+'m_time_s'] = integral(np.array(near, float), t)
    for b in CONFIG['actor_buffers_m']:
        v = (distances < b).astype(float)
        result['actor_below_'+str(b)+'m_observed_time_s'] = (
            float(np.sum((v[:-1]+v[1:])*.5*dt*interval_observed)) if interval_observed.any() else None)
    return result


def measure(center_states, frames, dt, route_progress_m):
    if not np.isfinite(route_progress_m) or route_progress_m < 0: raise ValueError('Invalid raw route progress')
    result = dict(route_progress_m=float(route_progress_m))
    for prefix, start in (('all_', 0.), ('late_', CONFIG['late_start_s'])):
        result.update({prefix+k: v for k, v in spatial(frames, dt, start).items()})
        result.update({prefix+k: v for k, v in kinematics(center_states, dt, start).items()})
    return result


def paired_stats(values, sessions):
    groups = {}
    for value, session in zip(values, sessions):
        if value is not None:
            if not np.isfinite(value): raise ValueError('Nonfinite statistic')
            groups.setdefault(session, []).append(value)
    if not groups: return dict(paired_scenes=0, paired_sessions=0, micro_mean=None, macro_mean=None, ci95_sessions=None)
    means = np.array([np.mean(groups[s]) for s in sorted(groups)])
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    boot = means[rng.integers(len(means), size=(CONFIG['bootstrap_reps'], len(means)))].mean(axis=1)
    return dict(paired_scenes=sum(map(len, groups.values())), paired_sessions=len(means),
        micro_mean=float(np.mean([x for g in groups.values() for x in g])), macro_mean=float(means.mean()),
        ci95_sessions=[float(x) for x in np.quantile(boot, [.025, .975])])


def tradeoff(candidate, baseline, raw_not_worse, intervention):
    """Separate descriptive benefits within fixed metre budgets; no aggregate utility."""
    loss = max(0., baseline['route_progress_m']-candidate['route_progress_m'])
    result = {}
    for budget in CONFIG['progress_loss_budgets_m']:
        eligible = bool(intervention and raw_not_worse and loss <= budget+1e-8
            and candidate['all_outside_area_max_m2'] <= baseline['all_outside_area_max_m2']+1e-8
            and candidate['all_outside_area_integral_m2s'] <= baseline['all_outside_area_integral_m2s']+1e-8
            and candidate['all_outside_footprint_time_s'] <= baseline['all_outside_footprint_time_s']+1e-8
            and candidate['all_actor_contact_time_s'] <= baseline['all_actor_contact_time_s']+1e-8)
        benefits = {}
        for name, threshold in (('road_clearance_min_m', CONFIG['descriptive_road_improvement_m']),
                                ('actor_clearance_min_m', CONFIG['descriptive_actor_improvement_m'])):
            a, b = candidate['late_'+name], baseline['late_'+name]
            observed = (name != 'actor_clearance_min_m' or
                        candidate['late_actor_observed_interval_fraction'] == baseline['late_actor_observed_interval_fraction'] == 1.)
            benefits[name] = None if a is None or b is None or not observed else bool(a-b >= threshold)
        a, b = candidate['all_jerk_norm_mps3_rms'], baseline['all_jerk_norm_mps3_rms']
        benefits['jerk_norm_mps3_rms'] = bool(b-a >= CONFIG['descriptive_jerk_absolute_reduction_mps3']
            and a <= b*(1-CONFIG['descriptive_jerk_relative_reduction']))
        result[str(budget)] = dict(eligible=eligible, **benefits)
    return result
