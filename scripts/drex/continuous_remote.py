"""Astra-only CPU replay, raw parity gate and continuous trajectory diagnostics."""
import argparse
import importlib.metadata
import json
import lzma
import os
from pathlib import Path
import pickle
import platform
import sys
import time
import traceback

import numpy as np

import continuous_metrics as metrics
import continuous_report
import real_data as base
import rollout_replay as replay
from transfer import read, write, sha, freeze

EXTRA_SOURCES = ('navsim/planning/simulation/planner/pdm_planner/utils/pdm_array_representation.py',
    'navsim/planning/simulation/planner/pdm_planner/utils/pdm_geometry_utils.py',
    'navsim/planning/simulation/planner/pdm_planner/utils/pdm_enums.py',
    'navsim/planning/simulation/planner/pdm_planner/simulation/batch_kinematic_bicycle.py')


def candidate_key(c): return c['token']+'-'+str(c['candidate_local'])


def parity(measured, expected):
    a, b = np.asarray(measured, float), np.asarray(expected, float)
    if (a.shape != (9,) or b.shape != (9,) or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.allclose(a, b, atol=1e-8, rtol=1e-7)):
        raise RuntimeError('Raw PDM/component parity failed; continuous metrics not accepted')


def verify_completed(directory, candidate, identity_sha):
    commit = read(directory/'commit.json')
    if (commit['identity_sha256'] != identity_sha or commit['candidate'] != candidate
            or set(commit['files']) != {'summary.json', 'traces.npz', 'frames.json'}):
        raise RuntimeError('Completed candidate binding changed')
    for name, digest in commit['files'].items():
        if sha(directory/name) != digest: raise RuntimeError('Completed candidate file changed: '+name)
    summary = read(directory/'summary.json')
    if (summary['token'], summary['candidate_local'], summary['candidate_vocab']) != (
            candidate['token'], candidate['candidate_local'], candidate['candidate_vocab']):
        raise RuntimeError('Completed candidate identity differs')
    parity(summary['measured_raw'], candidate['expected_raw'])
    return summary


def evaluate(candidate, cache_info, template, scenes, vocab, expected_vehicle):
    import clearance_math
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        state_array_to_center_state_array, state_array_to_coords_array)
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex, BBCoordsIndex
    expected = {'X': 0, 'Y': 1, 'HEADING': 2, 'VELOCITY_X': 3, 'VELOCITY_Y': 4,
        'ACCELERATION_X': 5, 'ACCELERATION_Y': 6, 'ANGULAR_VELOCITY': 9, 'ANGULAR_ACCELERATION': 10}
    if StateIndex.size() != 11 or any(getattr(StateIndex, k) != v for k, v in expected.items()) or int(BBCoordsIndex.CENTER) != 4:
        raise RuntimeError('Unexpected PDM state/coordinate schema')
    cfg = OmegaConf.create(replay.config_core(template))
    if cfg.traffic_agents != 'non_reactive': raise RuntimeError('Original non-reactive traffic required')
    simulator, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    scorer._config.human_penalty_filter = False
    dt = float(simulator.proposal_sampling.interval_length)
    if abs(dt-metrics.CONFIG['dt_s']) > 1e-12: raise ValueError('Unexpected sampling interval')
    dimensions = {k: float(getattr(scorer._vehicle_parameters, k)) for k in expected_vehicle}
    if any(abs(dimensions[k]-expected_vehicle[k]) > 1e-9 for k in dimensions):
        raise RuntimeError('Scorer footprint dimensions differ from confirmed Astra profile')
    if sha(Path(cache_info['path'])) != cache_info['sha256']: raise RuntimeError('Cache changed')
    # pdm_score mutates observation data: use a fresh original cache for EVERY candidate.
    with lzma.open(cache_info['path'], 'rb') as stream: cache = pickle.load(stream)
    if cache.log_name != scenes[candidate['token']]['log_name']: raise RuntimeError('Cache log differs from scene')
    policy = instantiate(cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling)
    sampling = instantiate(cfg.agent_sampling)
    frame, states = pdm_score(cache, Trajectory(vocab[candidate['candidate_vocab']].astype(float), sampling),
        simulator.proposal_sampling, simulator, scorer, policy)
    measured = [float(frame.iloc[0][k]) for k in candidate['metric_keys']]
    parity(measured, candidate['expected_raw'])  # Do not interpret new geometry on a failed replay.
    states = np.asarray(states, float)
    if states.shape != (41, 11) or not np.isfinite(states).all(): raise ValueError('Invalid simulated states')
    origin = np.asarray(cache.ego_state.rear_axle.serialize(), float)
    if not np.allclose(states[0, :3], origin, atol=1e-9, rtol=0): raise RuntimeError('Wrong replay origin')
    # Scorer dimensions define evaluated road footprint; actual cached vehicle defines comfort point.
    vehicle = cache.ego_state.car_footprint.vehicle_parameters
    center = state_array_to_center_state_array(states[None], vehicle)[0]
    expected_velocity = states[:, 3:5].copy()
    expected_velocity[:, 1] += float(vehicle.rear_axle_to_center)*states[:, 9]
    if not np.allclose(center[:, 3:5], expected_velocity, atol=1e-10, rtol=0):
        raise RuntimeError('Unexpected body-frame center velocity convention')
    coords = state_array_to_coords_array(states[None], scorer._vehicle_parameters)[0]
    if not np.allclose(coords, scorer._ego_coords[1], atol=1e-9, rtol=0): raise RuntimeError('Footprint differs from scorer')
    indices = cache.drivable_area_map.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.INTERSECTION, SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
    road = clearance_math.road_union([cache.drivable_area_map._geometries[i] for i in indices])
    traffic = policy.simulate_environment(states, cache)
    if len(traffic) != len(states): raise RuntimeError('Actor grid differs from trajectory')
    frames = [clearance_math.frame_measurement(c,
        road, [(obj.track_token, obj.box.geometry) for obj in f.tracked_objects]) for c, f in zip(coords, traffic)]
    progress = float(scorer._progress_raw[1])
    values = metrics.measure(center, frames, dt, progress)
    summary = dict(token=candidate['token'], candidate_local=candidate['candidate_local'],
        candidate_vocab=candidate['candidate_vocab'], teacher_rank=candidate['teacher_rank'], measured_raw=measured,
        metric_parity_passed=True, metrics=values, scorer_vehicle_dimensions=dimensions,
        comfort_vehicle_dimensions={k: float(getattr(vehicle, k)) for k in expected_vehicle},
        dt_s=dt, samples=len(states))
    return summary, dict(simulated_states=states, center_states=center, scorer_coords=coords), frames


def run(request_path, out, repo, stage):
    request = read(request_path); metrics.validate_request(request)
    package = Path(request['remote_package']).resolve()
    if sha(package/'package_manifest.json') != request['package_manifest_sha256']:
        raise RuntimeError('Portable package differs from frozen msu270 package')
    if request['vehicle_profile'] != read(Path(__file__).with_name('vehicle_profile_astra.json')):
        raise RuntimeError('Confirmed vehicle profile changed')
    for name, digest in request['tools'].items():
        if sha(Path(__file__).with_name(name)) != digest: raise RuntimeError('Staged tool differs from protocol: '+name)
    sys.path.insert(1, str(repo))
    manifest, scenes, vocab, template, caches = replay.bound_context(package, repo, request['candidates'])
    if template.get('map_root'): os.environ['NUPLAN_MAPS_ROOT'] = template['map_root']
    for c in request['candidates']:
        path = package/('bundle/examples/'+c['token']+'.npz')
        if sha(path) != c['example_sha256'] or manifest['files'][path.relative_to(package).as_posix()]['sha256'] != c['example_sha256']:
            raise RuntimeError('Original example changed')
        with np.load(path, allow_pickle=False) as arrays:
            if int(arrays['candidate_ids'][c['candidate_local']]) != c['candidate_vocab']:
                raise RuntimeError('Local/vocabulary alignment changed')
            # Tail labels are bound by the verified msu270 top64 label source, not in the top16 NPZ.
            if c['candidate_local'] in arrays['label_indices']:
                row = arrays['label_indices'].tolist().index(c['candidate_local'])
                parity(arrays['label_raw'][row], c['expected_raw'])
    packages = {}
    for name in ('numpy', 'scipy', 'shapely', 'nuplan-devkit', 'hydra-core'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    freeze(out/'identity.json', dict(request_sha256=sha(request_path), python=platform.python_version(),
        packages=packages, config=replay.config_core(template), caches=caches, tools=request['tools'],
        sources={name: sha(repo/name) for name in (*replay.REPLAY_SOURCES, *EXTRA_SOURCES)}))
    identity_sha = sha(out/'identity.json'); candidates = request['candidates']
    pilot = [c for c in candidates if c['token'] in request['pilot_tokens']]
    if stage == 'full':
        for c in pilot: verify_completed(out/'candidates'/candidate_key(c), c, identity_sha)
    target = pilot if stage == 'pilot' else candidates
    completed = {}; start = time.monotonic(); new_calls = 0
    # Read and verify every existing commitment even during a repeated pilot.
    for c in candidates:
        directory = out/'candidates'/candidate_key(c)
        if (directory/'commit.json').exists():
            completed[c['token'], c['candidate_local']] = verify_completed(directory, c, identity_sha)
    for c in target:
        key = c['token'], c['candidate_local']; directory = out/'candidates'/candidate_key(c)
        if key in completed: continue
        if (directory/'failure.json').exists(): raise RuntimeError('Prior candidate failure preserved; inspect before changing protocol')
        try:
            summary, arrays, frames = evaluate(c, caches[c['token']], template, scenes, vocab,
                request['vehicle_profile']['dimensions_m'])
            write(directory/'summary.json', summary); write(directory/'frames.json', frames)
            temp = directory/'traces.tmp.npz'; np.savez_compressed(temp, **arrays); temp.replace(directory/'traces.npz')
            freeze(directory/'commit.json', dict(identity_sha256=identity_sha, candidate=c,
                files={name: sha(directory/name) for name in ('summary.json', 'traces.npz', 'frames.json')}))
        except Exception as exc:
            write(directory/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc())); raise
        completed[key] = summary; new_calls += 1
        elapsed = time.monotonic()-start
        remaining = len(target)-sum((x['token'], x['candidate_local']) in completed for x in target)
        print(f'CONTINUOUS {len(completed)}/{len(candidates)} token={c["token"]} '
            f'parity=True new_calls={new_calls} elapsed_s={elapsed:.1f} eta_s={remaining*elapsed/new_calls:.1f}', flush=True)
        if new_calls % 25 == 0:
            write(out/'progress.json', dict(completed=len(completed), required=len(candidates), new_calls=new_calls,
                seconds=elapsed, estimated_remaining_seconds=remaining*elapsed/new_calls))
    result = continuous_report.report(request, completed, out, stage)
    result.update(new_scorer_calls_this_invocation=new_calls, seconds_this_invocation=time.monotonic()-start)
    write(out/'report.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'comparisons_vs_DS'}, indent=2), flush=True)
    return 0 if result['status'] in ('pilot_complete', 'complete') else 2


def main(stage='full'):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True); a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (a.out/'.execution.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another continuous audit is running')
        try: return run(a.input.resolve(), a.out.resolve(), a.repo.resolve(), stage)
        except Exception as exc:
            write(a.out/'report.json', dict(status='technical_failure', error=repr(exc),
                traceback=traceback.format_exc(), geometry_interpretation_ready=False, note=metrics.NOTE))
            traceback.print_exc(); return 1


if __name__ == '__main__': raise SystemExit(main())
