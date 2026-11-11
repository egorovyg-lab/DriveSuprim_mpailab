"""CPU-only measurements from existing replay states and SHA-bound Astra caches."""
import argparse
import csv
import importlib.metadata
import json
import lzma
from pathlib import Path
import pickle
import sys
import traceback

from transfer import read, write, sha, checked, freeze

NOTE = ('Privileged post-hoc audit of six outcome-selected pairs. No model inputs or deployment policy. '
        'Map and recorded future traffic are diagnostic ground truth. Whole-footprint road coverage '
        'is distinct from the scorer corner-only DAC. Road union removes internal polygon seams. '
        'Outside footprints have null road clearance; outside area and signed corner distances are reported. '
        'Traffic intersections do not classify collision fault or reproduce TTC. Samples every 0.1 s, '
        'including t=0; no claim of a continuous swept-volume guarantee. Lane keeping preserves the '
        'original sample counter and intersection-skip behavior. No scoring, simulation, Drex, fitting or holdout.')


def verify_source(source, expected):
    if not source.is_dir(): raise FileNotFoundError('Completed source audit missing: '+str(source))
    required = {'identity.json', 'report.json'} | {f'case{pair:02d}-member{member}/{name}'
        for pair in range(1, 7) for member in (0, 1) for name in ('commit.json', 'summary.json', 'traces.npz')}
    if set(expected) != required: raise ValueError('Incomplete source audit bindings')
    for name, digest in expected.items():
        if sha(checked(source, name)) != digest: raise RuntimeError('Source audit artifact changed: '+name)
    report = read(source/'report.json')
    if (report['status'] != 'complete' or report['candidates'] != 12 or report['metric_parity_passed'] != 12 or
            not report['geometry_interpretation_ready']):
        raise RuntimeError('Require successful original 12-candidate metric parity')


def run(request_path, out, repo):
    import numpy as np
    from shapely.geometry import Point
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import clearance_math as math
    import rollout_replay as old
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    request = read(request_path)
    if request['protocol'] != 'clearance_pair6_v1': raise ValueError('Unexpected clearance protocol')
    source = Path(request['source_artifacts']).resolve()
    if source.parent.parent != Path('/workspace/exp_v2/drex_rollout_audit') or source.name != 'artifacts':
        raise ValueError('Source must stay in the dedicated audit directory')
    verify_source(source, request['source_files'])
    identity = read(source/'identity.json')
    if sha(source.parent/'audit_request.json') != request['source_request_sha256'] or identity['request_sha256'] != request['source_request_sha256']:
        raise RuntimeError('Original audit request changed')
    for name, digest in identity['scorer_sources'].items():
        if sha(repo/name) != digest: raise RuntimeError('Original scorer source changed: '+name)
    cfg = OmegaConf.create(identity['config'])
    if cfg.traffic_agents != 'non_reactive': raise ValueError('Requires original non-reactive traffic')
    scorer = instantiate(cfg.scorer)  # Dimensions/configuration only. score_proposals is never called.
    sampling = instantiate(cfg.proposal_sampling)
    policy = instantiate(cfg.traffic_agents_policy.non_reactive, sampling)
    out.mkdir(parents=True, exist_ok=True)
    freeze(out/'identity.json', dict(request_sha256=sha(request_path), source_identity_sha256=sha(source/'identity.json'),
        tools={p.name: sha(p) for p in Path(__file__).parent.iterdir() if p.is_file() and p.suffix in ('.py', '.json')},
        versions={k: importlib.metadata.version(k) for k in ('numpy', 'shapely')},
        buffers_m=math.BUFFERS_M, scorer_sources=identity['scorer_sources']))
    rows = []; corner_checks = 0; lane_checks = 0
    for pair_id in ('case01', 'case02', 'case03', 'case04', 'case05', 'case06'):
        for member in (0, 1):
            folder = pair_id+'-member'+str(member); summary = read(source/folder/'summary.json')
            if summary['pair_id'] != pair_id or summary['member'] != member or not summary['parity_passed']:
                raise RuntimeError('Candidate identity/parity mismatch')
            cache_info = identity['caches'][summary['token']]
            if sha(Path(cache_info['path'])) != cache_info['sha256']: raise RuntimeError('Metric cache changed')
            with lzma.open(cache_info['path'], 'rb') as stream: cache = pickle.load(stream)
            dimensions = {k: float(getattr(scorer._vehicle_parameters, k)) for k in summary['scorer_vehicle_dimensions_m']}
            if dimensions != summary['scorer_vehicle_dimensions_m']: raise RuntimeError('Scorer vehicle dimensions changed')
            with np.load(source/folder/'traces.npz', allow_pickle=False) as stored:
                arrays = {k: stored[k].copy() for k in stored.files}
            origin = np.asarray(cache.ego_state.rear_axle.serialize(), dtype=float)
            if not np.allclose(arrays['simulated_states'][0, :3], origin, atol=1e-9, rtol=0):
                raise RuntimeError('Cache initial state differs from the original replay')
            indices = cache.drivable_area_map.get_indices_of_map_type([
                SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
                SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
            road = math.road_union([cache.drivable_area_map._geometries[i] for i in indices])
            dt = float(summary['dt_s'])
            if summary['samples'] != 41 or abs(dt-.1) > 1e-12: raise ValueError('Expected 41 samples at 0.1 s')
            # Same filtering of initially overlapping objects as the frozen log-replay policy.
            traffic = policy.simulate_environment(arrays['simulated_states'], cache)
            if len(traffic) != 41: raise RuntimeError('Traffic time grid mismatch')
            actors = [[(obj.track_token, obj.box.geometry) for obj in frame.tracked_objects]
                      for frame in traffic]
            for geometry in ('nominal', 'simulated'):
                states = arrays[geometry+'_states']
                coords = state_array_to_coords_array(states[None], scorer._vehicle_parameters)[0]
                if coords.shape != (41, 5, 2) or not np.isfinite(coords).all(): raise ValueError('Invalid replay coordinates')
                if not np.allclose(old.geometry.local_xy(coords, origin), arrays[geometry+'_boxes_local'], atol=1e-9, rtol=0):
                    raise RuntimeError('Reconstructed corners differ from the original replay')
                inside, _ = old.map_membership(cache, coords, SemanticMapLayer)
                if not np.array_equal(inside, arrays[geometry+'_inside']): raise RuntimeError('Original DAC corner mask changed')
                corner_checks += 1
                distances = np.array([Point(p).distance(cache.centerline.linestring) for p in coords[:, 4]])
                intersections = np.array([cache.drivable_area_map.is_in_layer(Point(p), SemanticMapLayer.INTERSECTION)
                                          for p in coords[:, 4]])
                if (not np.allclose(distances, arrays[geometry+'_lane_distance_m'], atol=1e-9, rtol=0) or
                        not np.array_equal(intersections, arrays[geometry+'_in_intersection'])):
                    raise RuntimeError('Original lane geometry changed')
                lane = old.geometry.lane_trace(distances, intersections, dt,
                    scorer._config.lane_keeping_deviation_limit, scorer._config.lane_keeping_horizon_window)
                if lane != summary[geometry+'_lane_keeping']: raise RuntimeError('Original lane counter changed')
                lane_checks += 1
                frames = [math.frame_measurement(c, road, a) for c, a in zip(coords, actors)]
                measured = math.summarize(frames, dt, bool(inside.all()), lane)
                value = dict(pair_id=pair_id, member=member, token=summary['token'], geometry=geometry,
                    **measured, max_lane_distance_m=float(distances.max()),
                    lane_distance_limit_m=float(scorer._config.lane_keeping_deviation_limit),
                    recorded_simulated_raw=summary['measured_raw'])
                directory = out/folder; directory.mkdir(parents=True, exist_ok=True)
                write(directory/(geometry+'.json'), dict(summary=value, frames=frames, lane=lane))
                with (directory/(geometry+'.csv')).open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['time_s', 'full_footprint_covered', 'road_clearance_m', 'outside_area_m2',
                        'min_corner_signed_margin_m', 'actor_clearance_m', 'actor_contacts',
                        'lane_distance_m', 'in_intersection', 'lane_counter_samples'])
                    for i, frame in enumerate(frames):
                        writer.writerow([i*dt, frame['full_footprint_covered'], frame['road_clearance_m'],
                            frame['outside_area_m2'], frame['min_corner_signed_margin_m'], frame['actor_clearance_m'],
                            '|'.join(frame['intersecting_actor_tokens']), distances[i], intersections[i], lane['counters'][i]])
                rows.append(value)
            print('CLEARANCE '+folder, flush=True)
    pairs = []
    for pair_id in ('case01', 'case02', 'case03', 'case04', 'case05', 'case06'):
        members = [next(r for r in rows if r['pair_id'] == pair_id and r['member'] == i and r['geometry'] == 'simulated') for i in (0, 1)]
        pairs.append(dict(pair_id=pair_id, eligible_members_by_buffer_m={str(buffer): [i for i in (0, 1)
            if members[i]['sampled_geometry_eligible_at_buffer_m'][str(buffer)]] for buffer in math.BUFFERS_M}))
    report = dict(status='complete', candidates=12, geometry_traces=24, original_metric_parity_passed=12,
        corner_mask_checks_passed=corner_checks, lane_counter_checks_passed=lane_checks,
        new_scorer_calls=0, simulation_calls=0, model_inference_calls=0, holdout_processed=0,
        per_candidate=rows, pairs=pairs, note=NOTE)
    write(out/'report.json', report); print(json.dumps(report, indent=2), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args(); sys.path.insert(1, str(args.repo.resolve()))
    import fcntl
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out.parent/'.clearance.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: return run(args.input.resolve(), args.out.resolve(), args.repo.resolve())
        except Exception as exc:
            value = dict(status='technical_failure', error=repr(exc), traceback=traceback.format_exc(), note=NOTE)
            write(args.out/'report.json', value); print(json.dumps(value, indent=2), flush=True); return 1


if __name__ == '__main__': raise SystemExit(main())
