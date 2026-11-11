"""Astra-only CPU audit of 12 already evaluated proposals. No model inference."""
import argparse
import csv
import importlib.metadata
import json
import lzma
import os
from pathlib import Path
import pickle
import platform
import sys
import traceback

import numpy as np
from PIL import Image

import real_data as base
import pair_diagnostic_data as pair_view
import overlay_v2_data as projection
import rollout_audit_math as geometry
from transfer import read, write, sha, freeze

REPLAY_SOURCES = (
    'navsim/evaluate/pdm_score.py',
    'navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py',
    'navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py',
    'navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py',
    'navsim/traffic_agents_policies/log_replay_traffic_agents.py',
)
NOTE = ('Privileged post-hoc geometry audit on six outcome-selected cases, not a model evaluation. '
        'Map polygons, scorer reference centerline and event-time traffic are audit-only. '
        'No Drex calls, fitting or holdout. Raw metric parity must pass before interpreting geometry. '
        'Camera projections assume flat ground and test pixel inclusion, not visibility/occlusion. '
        'Lane-keeping reproduces the scorer sample-count and intersection-skip behavior exactly.')


def config_core(value):
    keys = ('scorer', 'simulator', 'proposal_sampling', 'traffic_agents', 'traffic_agents_policy')
    agent = value.get('agent')
    if (not all(k in value for k in keys) or not isinstance(agent, dict) or
            not isinstance(agent.get('config'), dict) or 'trajectory_sampling' not in agent['config']):
        return None
    return dict({k: value[k] for k in keys}, agent_sampling=value['agent']['config']['trajectory_sampling'])


def bound_metadata(inputs):
    """Follow recorded metadata hashes, including legacy source-relative inputs.

    The original action-transfer identity binds source/cache_index.json in its
    own inputs table. Later bundles bind that identity, but do not flatten all
    of its dependencies. Never discover a cache by globbing or an unbound path.
    """
    names = {'identity.json', 'source_config.json', 'cache_index.json'}
    bindings = {}; chains = {}; pending = []; metadata = []

    def bind(path, digest, chain):
        if (not path.is_absolute() or not isinstance(digest, str) or len(digest) != 64 or
                any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Unsupported frozen input binding: '+str(path))
        path = path.resolve(); key = str(path)
        if key in bindings:
            if bindings[key] != digest:
                raise RuntimeError('Conflicting frozen input hashes: '+key)
            return
        bindings[key] = digest
        if path.name in names:
            chains[key] = chain+[dict(path=key, sha256=digest)]
            pending.append(path)

    for name, digest in sorted(inputs.items()):
        bind(Path(name), digest, [])
    cursor = 0
    while cursor < len(pending):
        if len(pending) > 1024:
            raise ValueError('Frozen metadata dependency graph is too large')
        path = pending[cursor]; cursor += 1; key = str(path)
        if sha(path) != bindings[key]:
            raise RuntimeError('Frozen upstream metadata changed: '+key)
        value = read(path)
        if not isinstance(value, dict):
            raise ValueError('Unsupported frozen metadata: '+key)
        metadata.append((path, value))
        if path.name == 'cache_index.json':
            continue
        children = value.get('inputs', {})
        if not isinstance(children, dict):
            raise ValueError('Unsupported frozen inputs table: '+key)
        for name, digest in sorted(children.items()):
            child = Path(name)
            if child.name not in names:
                continue
            if not child.is_absolute():
                source = value.get('source')
                if len(child.parts) != 1 or not isinstance(source, str) or not Path(source).is_absolute():
                    raise ValueError('Relative metadata input has no absolute source root: '+key)
                child = Path(source)/child
            bind(child, digest, chains[key])
    return sorted(metadata, key=lambda item: str(item[0])), bindings, chains


def bound_context(package, repo, candidates):
    manifest, index, scenes, vocab = base.package_metadata(package)
    identity = read(package/'bundle/identity.json')
    metadata, bindings, chains = bound_metadata(identity['inputs'])
    configurations = []; cache_indices = []
    for path, value in metadata:
        if path.name == 'cache_index.json':
            cache_indices.append((path, value))
        elif config_core(value) is not None:
            configurations.append((path, value))
    if not configurations:
        raise RuntimeError('No bound original scorer/simulator configuration found')
    template = configurations[0][1]; core = config_core(template)
    if any(config_core(v) != core for _, v in configurations):
        raise RuntimeError('Bound scorer/simulator configurations disagree')
    for name in REPLAY_SOURCES:
        expected = template.get('sources', {}).get(name)
        if expected is None or sha(repo/name) != expected:
            raise RuntimeError('Scorer/simulator source differs from label collection: '+name)
    chosen = {}
    for candidate in candidates:
        token = candidate['token']
        if token in chosen:
            continue
        if token not in index or index[token]['metadata']['role'] != 'train':
            raise ValueError('Only committed training tokens may be audited')
        sample = base.load_inputs(package, token, manifest, vocab)
        options = []
        for path, values in cache_indices:
            if token not in values:
                continue
            entry = values[token]
            if not isinstance(entry, dict) or not {'path', 'sha256'} <= set(entry):
                raise ValueError('Unsupported bound cache index format')
            feature = (path.parent/'features'/(token+'.npz')).resolve()
            matching_feature = False
            if str(feature) in bindings:
                if sha(feature) != bindings[str(feature)]:
                    raise RuntimeError('Frozen source feature changed')
                with np.load(feature, allow_pickle=False) as arrays, np.load(
                        package/('bundle/examples/'+token+'.npz'), allow_pickle=False) as original:
                    matching_feature = (np.array_equal(arrays['candidate_ids'],
                        original['candidate_ids'])
                        and int(arrays['selected_local']) == sample['baseline_local'])
            options.append((not matching_feature, str(path), entry))
        if not options:
            raise FileNotFoundError('No bound cache index contains '+token+'; checked: '+
                                    ', '.join(str(path) for path, _ in cache_indices))
        options.sort(key=lambda item: (item[0], item[1])); _, cache_index, entry = options[0]
        if sha(Path(entry['path'])) != entry['sha256']:
            raise RuntimeError('Metric cache changed for '+token)
        chosen[token] = dict(entry, cache_index=cache_index,
                             cache_index_binding_chain=chains[cache_index],
                             matching_source_feature=not options[0][0], alternatives=len(options))
    return manifest, scenes, vocab, template, chosen


def map_membership(cache, coords, SemanticMapLayer):
    indices = cache.drivable_area_map.get_indices_of_map_type([
        SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
    inside = cache.drivable_area_map.points_in_polygons(coords)
    return inside[indices, :, :4].any(axis=0), indices


def polygons_local(geometries, origin):
    result = []
    for value in geometries:
        if value.is_empty:
            continue
        if value.geom_type == 'Polygon':
            result.append(dict(exterior=geometry.local_xy(np.asarray(value.exterior.coords), origin).tolist(),
                holes=[geometry.local_xy(np.asarray(ring.coords), origin).tolist() for ring in value.interiors]))
        elif hasattr(value, 'geoms'):
            result.extend(polygons_local(value.geoms, origin))
        else:
            raise ValueError('Unexpected drivable map geometry')
    return result


def replay_candidate(package, directory, candidate, cache_info, template, scenes, vocab):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from shapely.geometry import Point
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score, transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, EgoAreaIndex
    if [int(getattr(BBCoordsIndex, name)) for name in geometry.CORNER_NAMES] != list(range(4)) or int(BBCoordsIndex.CENTER) != 4:
        raise RuntimeError('PDM corner ordering changed')
    core = config_core(template); cfg = OmegaConf.create(core)
    simulator, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    scorer._config.human_penalty_filter = False
    if cfg.traffic_agents != 'non_reactive':
        raise ValueError('Audit requires the original non-reactive replay')
    policy = instantiate(cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling)
    sampling = instantiate(cfg.agent_sampling)
    with lzma.open(cache_info['path'], 'rb') as stream:
        cache = pickle.load(stream)  # User-created cache, SHA checked against frozen index first.
    token = candidate['token']; scene = scenes[token]
    if cache.log_name != scene['log_name']:
        raise RuntimeError('Metric cache log differs from camera scene')
    dimensions = {k: float(getattr(scorer._vehicle_parameters, k)) for k in pair_view.vehicle()}
    if not all(np.isclose(dimensions[k], pair_view.vehicle()[k], atol=1e-9, rtol=0) for k in dimensions):
        raise RuntimeError('Scorer vehicle dimensions differ from diagnostic footprints')
    keys = ['pdm_score', *base.METRICS]
    if candidate['metric_keys'] != keys:
        raise ValueError('Unknown label column order')
    poses = vocab[candidate['candidate_vocab']].astype(float)
    trajectory = Trajectory(poses, sampling)
    frame, states = pdm_score(cache, trajectory, simulator.proposal_sampling, simulator, scorer, policy)
    measured = np.array([float(frame.iloc[0][k]) for k in keys])
    target = np.asarray(candidate['expected_raw'], dtype=float)
    parity = bool(np.allclose(measured, target, atol=1e-8, rtol=1e-7))
    nominal = get_trajectory_as_array(transform_trajectory(trajectory, cache.ego_state),
                                      simulator.proposal_sampling, cache.ego_state.time_point)
    if nominal.shape != states.shape or not np.isfinite(states).all():
        raise RuntimeError('Nonfinite or mismatched replay state arrays')
    origin = np.array(cache.ego_state.rear_axle.serialize(), dtype=float)
    reference, simulated = geometry.local_poses(nominal[:, :3], origin), geometry.local_poses(states[:, :3], origin)
    dt = float(simulator.proposal_sampling.interval_length)
    nominal_coords = state_array_to_coords_array(nominal[None], scorer._vehicle_parameters)[0]
    scored_coords = scorer._ego_coords[1].copy()
    nominal_inside, drivable_indices = map_membership(cache, nominal_coords, SemanticMapLayer)
    scored_inside, _ = map_membership(cache, scored_coords, SemanticMapLayer)
    if not np.array_equal(~scored_inside.all(axis=1), scorer._ego_areas[1, :, EgoAreaIndex.NON_DRIVABLE_AREA]):
        raise RuntimeError('Re-derived DAC corners disagree with scorer')
    nominal_dac, simulated_dac = geometry.dac_trace(nominal_inside, dt), geometry.dac_trace(scored_inside, dt)
    def lane(coords):
        center = coords[:, BBCoordsIndex.CENTER]
        distance = np.array([Point(p).distance(cache.centerline.linestring) for p in center])
        intersections = np.array([cache.drivable_area_map.is_in_layer(Point(p), SemanticMapLayer.INTERSECTION) for p in center])
        trace = geometry.lane_trace(distance, intersections, dt,
                                   scorer._config.lane_keeping_deviation_limit, scorer._config.lane_keeping_horizon_window)
        return trace, distance, intersections
    nominal_lane, nominal_distance, intersections_nominal = lane(nominal_coords)
    simulated_lane, simulated_distance, intersections_simulated = lane(scored_coords)
    if simulated_dac['score'] != measured[keys.index('drivable_area_compliance')] or simulated_lane['score'] != measured[keys.index('lane_keeping')]:
        raise RuntimeError('Re-derived DAC/LK scores disagree with scorer')
    nominal_boxes = geometry.local_xy(nominal_coords, origin)
    simulated_boxes = geometry.local_xy(scored_coords, origin)
    photo_path = base.camera_paths(package, scene, read(package/'package_manifest.json'))[4]
    with Image.open(photo_path) as image:
        photo = image.convert('RGB')
    calib = projection.calibration(scene, photo.size)
    outside_details = []
    first = simulated_dac['first_index']
    if first is not None:
        for corner in np.flatnonzero(~scored_inside[first]):
            point = Point(scored_coords[first, corner])
            distance = min((point.distance(cache.drivable_area_map._geometries[i]) for i in drivable_indices), default=None)
            outside_details.append(dict(corner=geometry.CORNER_NAMES[corner],
                distance_to_drivable_polygon_m=None if distance is None else float(distance),
                local_xy=simulated_boxes[first, corner].tolist(),
                projection=geometry.pixel_separation(nominal_boxes[first, corner], simulated_boxes[first, corner], calib, photo.size)))
    collision_index = scorer._collision_time_idcs[1]
    collision_index = int(collision_index) if np.isfinite(collision_index) else None
    overlaps, collision_polygons = [], []
    if collision_index is not None:
        query = scorer._observation[collision_index].query(scorer._ego_polygons[1:2, collision_index], predicate='intersects')
        for geometry_index in query[1]:
            track = scorer._observation[collision_index].tokens[geometry_index]
            if scorer._observation.red_light_token in track or track in scorer._observation.collided_track_ids:
                continue
            overlaps.append(track)
            shape = scorer._observation[collision_index][track]
            for polygon in polygons_local([shape], origin):
                collision_polygons.append(polygon['exterior'])
    focus = candidate['focus_metric']
    event_index = {'drivable_area_compliance': simulated_dac['first_index'],
                   'lane_keeping': simulated_lane['first_index'],
                   'no_at_fault_collisions': collision_index}.get(focus)
    points = []
    if event_index is not None:
        points = (simulated_boxes[event_index, np.flatnonzero(~scored_inside[event_index])].tolist()
                  if focus == 'drivable_area_compliance' else [simulated_boxes[event_index, BBCoordsIndex.CENTER].tolist()])
    event = dict(metric=focus, index=event_index,
                 time_s=None if event_index is None else float(event_index*dt), points_local=points)
    if event_index is not None:
        corner = int(np.flatnonzero(~scored_inside[event_index])[0]) if focus == 'drivable_area_compliance' else int(BBCoordsIndex.CENTER)
        event['point_kind'] = geometry.CORNER_NAMES[corner] if corner < 4 else 'VEHICLE_CENTER'
        event['nominal_vs_simulated_projection'] = geometry.pixel_separation(
            nominal_boxes[event_index, corner], simulated_boxes[event_index, corner], calib, photo.size)
    map_polygons = polygons_local([cache.drivable_area_map._geometries[i] for i in drivable_indices], origin)
    centerline = geometry.local_xy(np.asarray(cache.centerline.linestring.coords), origin)
    summary = dict(pair_id=candidate['pair_id'], member=candidate['member'], token=token,
        candidate_vocab=candidate['candidate_vocab'], parity_passed=parity,
        measured_raw=dict(zip(keys, measured.tolist())), expected_raw=dict(zip(keys, target.tolist())),
        max_metric_abs_difference=float(np.abs(measured-target).max()),
        nominal_vs_simulated=geometry.trajectory_difference(reference, simulated),
        dt_s=dt, samples=len(states), nominal_dac=nominal_dac, simulated_dac=simulated_dac,
        nominal_lane_keeping=nominal_lane, simulated_lane_keeping=simulated_lane,
        first_outside_corner_details=outside_details, focus_event=event,
        focus_points_project_inside_photo=[geometry.point_projection(p, calib, photo.size) for p in points],
        first_at_fault_collision_time_s=None if collision_index is None else float(collision_index*dt),
        overlaps_at_recorded_at_fault_time=overlaps,
        overlap_tokens_are_not_a_new_fault_classification=True, scorer_vehicle_dimensions_m=dimensions,
        simulator_vehicle_dimensions_m={k: float(getattr(cache.ego_state.car_footprint.vehicle_parameters, k)) for k in dimensions},
        note=NOTE)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/'traces.npz', nominal_states=nominal, simulated_states=states,
        nominal_local=reference, simulated_local=simulated, nominal_boxes_local=nominal_boxes,
        simulated_boxes_local=simulated_boxes, nominal_inside=nominal_inside, simulated_inside=scored_inside,
        nominal_lane_distance_m=nominal_distance, simulated_lane_distance_m=simulated_distance,
        nominal_in_intersection=intersections_nominal, simulated_in_intersection=intersections_simulated)
    with (directory/'trace.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s', 'nominal_x_m', 'nominal_y_m', 'simulated_x_m', 'simulated_y_m',
                         'nominal_lane_distance_m', 'simulated_lane_distance_m', 'nominal_offroad', 'simulated_offroad'])
        for i in range(len(states)):
            writer.writerow([i*dt, *reference[i, :2], *simulated[i, :2], nominal_distance[i], simulated_distance[i],
                             not nominal_inside[i].all(), not scored_inside[i].all()])
    caption = f'{candidate["pair_id"]} member{candidate["member"]} | parity={parity} | focus={focus}'
    geometry.camera_figure(photo, calib, reference, simulated, nominal_boxes, simulated_boxes, event, caption, directory/'camera.png')
    geometry.map_figure(reference, simulated, simulated_boxes, map_polygons, centerline, event, collision_polygons, caption, directory/'map.png')
    write(directory/'summary.json', summary)
    freeze(directory/'commit.json', dict(files={p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.name != 'commit.json'}))
    return summary


def pair_summaries(package, out, request, rows, scenes, manifest):
    result = []
    for pair_id in sorted({c['pair_id'] for c in request['candidates']}):
        specs = sorted([c for c in request['candidates'] if c['pair_id'] == pair_id], key=lambda c: c['member'])
        summaries = sorted([r for r in rows if r['pair_id'] == pair_id], key=lambda r: r['member'])
        traces = []
        for spec in specs:
            with np.load(out/(pair_id+'-member'+str(spec['member']))/'traces.npz', allow_pickle=False) as value:
                traces.append({k: value[k].copy() for k in ('nominal_local', 'simulated_local', 'nominal_boxes_local', 'simulated_boxes_local')})
        if traces[0]['nominal_local'].shape != traces[1]['nominal_local'].shape:
            raise RuntimeError('Paired replay sampling differs')
        worse = 1-specs[0]['better_member']; event = summaries[worse]['focus_event']
        entry = dict(pair_id=pair_id, focus_metric=specs[0]['focus_metric'], recorded_worse_member=worse,
            metric_parity_passed=all(r['parity_passed'] for r in summaries),
            max_nominal_path_difference_m=geometry.trajectory_difference(traces[0]['nominal_local'], traces[1]['nominal_local'])['max_position_error_m'],
            max_simulated_path_difference_m=geometry.trajectory_difference(traces[0]['simulated_local'], traces[1]['simulated_local'])['max_position_error_m'],
            first_focus_event_time_s=event['time_s'])
        if event['index'] is not None:
            index = event['index']
            corner = geometry.CORNER_NAMES.index(event['point_kind']) if event['point_kind'] in geometry.CORNER_NAMES else 4
            path = base.camera_paths(package, scenes[specs[0]['token']], manifest)[4]
            with Image.open(path) as image:
                calib = projection.calibration(scenes[specs[0]['token']], image.size)
                for key in ('nominal_boxes_local', 'simulated_boxes_local'):
                    projection_value = geometry.pixel_separation(traces[0][key][index, corner], traces[1][key][index, corner], calib, image.size)
                    entry[key.replace('_boxes_local', '_between_candidates_projection')] = dict(
                        both_project_inside_photo=projection_value['reference']['projects_inside_current_photo'] and projection_value['simulated']['projects_inside_current_photo'],
                        separation_pixels=projection_value['separation_pixels'], occlusion_tested=False)
        result.append(entry)
    return result


def run(request_path, out, repo, package):
    sys.path.insert(1, str(repo))
    out.mkdir(parents=True, exist_ok=True)
    request = read(request_path)
    if request['protocol'] != 'nominal_rollout_audit_v1' or len(request['candidates']) != 12:
        raise ValueError('Expected committed 12-candidate request')
    if len({(c['pair_id'], c['member']) for c in request['candidates']}) != 12:
        raise ValueError('Duplicate replay candidates')
    if sha(package/'package_manifest.json') != request['package_manifest_sha256']:
        raise RuntimeError('Astra portable package differs from msu270 snapshot')
    if request['vehicle_profile'] != read(Path(__file__).with_name('vehicle_profile_astra.json')):
        raise RuntimeError('Vehicle profile changed')
    manifest, scenes, vocab, template, caches = bound_context(package, repo, request['candidates'])
    if template.get('map_root'):
        os.environ['NUPLAN_MAPS_ROOT'] = template['map_root']
    for candidate in request['candidates']:
        token = candidate['token']
        relative = 'bundle/examples/'+token+'.npz'
        if candidate['example_sha256'] != manifest['files'][relative]['sha256']:
            raise RuntimeError('Candidate example identity changed')
        with np.load(package/relative, allow_pickle=False) as arrays:
            if int(arrays['candidate_ids'][candidate['candidate_local']]) != candidate['candidate_vocab']:
                raise RuntimeError('Candidate vocabulary/local index mismatch')
            row = arrays['label_indices'].tolist().index(candidate['candidate_local'])
            if not np.array_equal(arrays['label_raw'][row], candidate['expected_raw']):
                raise RuntimeError('Requested metric labels differ from original bundle')
        photo = base.camera_paths(package, scenes[token], manifest)[4]
        if sha(photo) != manifest['files'][photo.relative_to(package).as_posix()]['sha256']:
            raise RuntimeError('Audit photograph changed')
    versions = {}
    for package_name in ('numpy', 'pandas', 'scipy', 'shapely', 'nuplan-devkit'):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = 'unknown'
    identity = dict(request_sha256=sha(request_path), python=platform.python_version(), versions=versions,
        config=config_core(template), caches=caches,
        tools={p.name: sha(p) for p in Path(__file__).parent.iterdir() if p.is_file() and p.suffix in ('.py', '.json')},
        scorer_sources={name: sha(repo/name) for name in REPLAY_SOURCES})
    freeze(out/'identity.json', identity)
    rows = []; replayed = 0
    for candidate in request['candidates']:
        directory = out/(candidate['pair_id']+'-member'+str(candidate['member']))
        if (directory/'commit.json').exists():
            for name, digest in read(directory/'commit.json')['files'].items():
                if sha(directory/name) != digest:
                    raise RuntimeError('Completed replay artifact changed')
            value = read(directory/'summary.json')
        else:
            if (directory/'failure.json').exists():
                raise RuntimeError('Recorded replay failure; inspect it and choose a new audit name')
            try:
                value = replay_candidate(package, directory, candidate, caches[candidate['token']], template, scenes, vocab)
                replayed += 1
            except Exception as exc:
                write(directory/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc())); raise
        rows.append(value)
        print(f'REPLAY {len(rows)}/12 {candidate["pair_id"]} member{candidate["member"]} parity={value["parity_passed"]}', flush=True)
    passed = sum(r['parity_passed'] for r in rows)
    pairs = pair_summaries(package, out, request, rows, scenes, manifest)
    compact = [dict(pair_id=r['pair_id'], member=r['member'], parity_passed=r['parity_passed'],
        nominal_vs_simulated=r['nominal_vs_simulated'],
        nominal_dac=r['nominal_dac']['score'], simulated_dac=r['simulated_dac']['score'],
        nominal_lane_keeping=r['nominal_lane_keeping']['score'], simulated_lane_keeping=r['simulated_lane_keeping']['score'],
        first_dac_time_s=r['simulated_dac']['first_time_s'],
        first_lane_qualification_time_s=r['simulated_lane_keeping']['first_time_s'],
        first_at_fault_collision_time_s=r['first_at_fault_collision_time_s']) for r in rows]
    report = dict(status='complete' if passed == 12 else 'parity_failed', candidates=12,
        metric_parity_passed=passed, geometry_interpretation_ready=passed == 12,
        replayed_this_invocation=replayed, resumed_candidates=12-replayed, model_inference_calls=0,
        nominal_vs_simulated_dac_disagreements=sum(r['nominal_dac']['score'] != r['simulated_dac']['score'] for r in rows),
        nominal_vs_simulated_lane_disagreements=sum(r['nominal_lane_keeping']['score'] != r['simulated_lane_keeping']['score'] for r in rows),
        max_position_error_m=max(r['nominal_vs_simulated']['max_position_error_m'] for r in rows),
        candidates_with_focus_event=sum(r['focus_event']['index'] is not None for r in rows),
        candidates_with_focus_point_projected_inside_photo=sum(any(p['projects_inside_current_photo'] for p in r['focus_points_project_inside_photo']) for r in rows),
        per_candidate=compact, pairs=pairs,
        note=NOTE)
    write(out/'report.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if passed == 12 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True); parser.add_argument('--package', type=Path, required=True)
    args = parser.parse_args()
    import fcntl
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out.parent/'.replay.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another CPU audit is running in this directory')
        try:
            return run(args.input.resolve(), args.out.resolve(), args.repo.resolve(), args.package.resolve())
        except Exception as exc:
            value = dict(status='technical_failure', error=repr(exc), traceback=traceback.format_exc(),
                         geometry_interpretation_ready=False, note=NOTE)
            write(args.out/'report.json', value); print(json.dumps(value, indent=2), flush=True)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
