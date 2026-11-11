"""Frozen top16 CPU replay and clearance audit; no model inference or fitting."""
import argparse
import csv
import importlib.metadata
import json
import lzma
from pathlib import Path
import pickle
import sys
import traceback

from transfer import read, write, sha, freeze, checked

NOTE = ('Privileged outcome-informed six-scene diagnostic, all 16 DS-ranked candidates retained. '
        'Input selection frozen before reading stored top16 labels. Each candidate uses the original '
        'single-candidate PDM evaluation with its reference proposal; no batch progress renormalization. '
        'Raw local PDM without two-frame extended comfort, not full EPDMS. Map/future traffic are audit-only. '
        'Contacts are geometric intersections, not fault/TTC. Buffers 0/0.1/0.25/0.5 m are descriptive, '
        'not a calibrated safety policy. Includes t=0; sampled every 0.1 s, no continuous-volume guarantee. '
        'No Drex, fitting, new dataset transfer or holdout. Best-PDM eligible candidates are oracle diagnostics.')


def selection_for(cases, samples, example_hashes):
    rows = []
    if len(cases) != 6 or len(set(cases.values())) != 6: raise ValueError('Exactly six distinct original scenes required')
    for case, token in sorted(cases.items()):
        sample = samples[token]
        if (len(sample['pool_local']) != 16 or len(set(sample['pool_local'])) != 16 or
                len(sample['pool_vocab']) != 16 or len(set(sample['pool_vocab'])) != 16 or
                sample['pool_local'][0] != sample['baseline_local']):
            raise ValueError('Expected original ordered top16 with DS at rank 1')
        for rank, (local, vocab) in enumerate(zip(sample['pool_local'], sample['pool_vocab']), 1):
            rows.append(dict(case_id=case, token=token, teacher_rank=rank, candidate_local=int(local),
                candidate_vocab=int(vocab), is_baseline=rank == 1, example_sha256=example_hashes[token]))
    return rows


def load_labels(path, sample):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        indices, raw = data['label_indices'].copy(), data['label_raw'].astype(float)
    if (indices.shape != (16,) or indices.dtype.kind not in 'iu' or indices.tolist() != sample['pool_local'] or
            raw.shape != (16, 9) or not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any()):
        raise ValueError('Frozen top16 label alignment/values invalid')
    return raw


def compare_scenes(rows, buffers):
    scenes = []
    cases = sorted({r['case_id'] for r in rows})
    for case in cases:
        group = sorted([r for r in rows if r['case_id'] == case], key=lambda r: r['teacher_rank'])
        if [r['teacher_rank'] for r in group] != list(range(1, 17)): raise ValueError('All 16 ranks must finish')
        baseline = group[0]; conditions = {}
        for buffer in buffers:
            eligible = [r for r in group if r['sampled_geometry_eligible_at_buffer_m'][str(buffer)]]
            not_slower = [r for r in eligible if r['progress_m'] >= baseline['progress_m']-1e-6]
            best = max(eligible, key=lambda r: (r['raw_pdm'], -r['teacher_rank'])) if eligible else None
            conditions[str(buffer)] = dict(eligible_ranks=[r['teacher_rank'] for r in eligible],
                progress_not_less_than_ds_ranks=[r['teacher_rank'] for r in not_slower],
                best_pdm_rank=None if best is None else best['teacher_rank'],
                best_pdm=None if best is None else best['raw_pdm'],
                best_gain_pp=None if best is None else 100*(best['raw_pdm']-baseline['raw_pdm']),
                best_pdm_progress_m=None if best is None else best['progress_m'],
                maximum_progress_m=max((r['progress_m'] for r in eligible), default=None))
        scenes.append(dict(case_id=case, token=baseline['token'], baseline_pdm=baseline['raw_pdm'],
            baseline_progress_m=baseline['progress_m'],
            road_covered_candidates=sum(r['full_footprint_covered_all_samples'] for r in group),
            lane_passing_candidates=sum(r['lane_keeping'] == 1 for r in group),
            contact_free_candidates=sum(r['first_actor_contact_time_s'] is None for r in group),
            buffers=conditions))
    return scenes


def run(request_path, out, repo):
    import numpy as np
    import clearance_remote as source_audit
    import clearance_math as math
    import rollout_replay as old
    import real_data as base
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from shapely.geometry import Point
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score, transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    request = read(request_path)
    if request['protocol'] != 'clearance_top16_v1': raise ValueError('Unexpected top16 protocol')
    source = Path(request['source_artifacts']).resolve()
    if source.parent.parent != Path('/workspace/exp_v2/drex_rollout_audit') or source.name != 'artifacts':
        raise ValueError('Invalid source audit directory')
    source_audit.verify_source(source, request['source_files'])
    identity = read(source/'identity.json')
    original_request_path = source.parent/'audit_request.json'
    if sha(original_request_path) != request['source_request_sha256'] or identity['request_sha256'] != request['source_request_sha256']:
        raise RuntimeError('Original audit request changed')
    original = read(original_request_path)
    package = Path(request['package']).resolve()
    if (sha(package/'package_manifest.json') != request['package_manifest_sha256'] or
            original['package_manifest_sha256'] != request['package_manifest_sha256']):
        raise RuntimeError('Portable package differs from the original audit')
    manifest, index, scenes, vocab = base.package_metadata(package)
    cases = {}; known = {}
    for candidate in original['candidates']:
        case, token = candidate['pair_id'], candidate['token']
        if case in cases and cases[case] != token: raise RuntimeError('Original case/token mismatch')
        cases[case] = token; known[case, candidate['candidate_local']] = candidate
    if len(known) != 12: raise ValueError('Expected the original 12 reference candidates')
    samples = {token: base.load_inputs(package, token, manifest, vocab) for token in cases.values()}
    hashes = {t: manifest['files']['bundle/examples/'+t+'.npz']['sha256'] for t in samples}
    selection = selection_for(cases, samples, hashes)
    out.mkdir(parents=True, exist_ok=True)
    freeze(out/'selection.json', dict(protocol='all_top16_six_scenes_v1', candidates=selection,
        package_manifest_sha256=request['package_manifest_sha256'], labels_read_for_selection=False))
    print('TOP16 SELECTION FROZEN: 6 scenes, 96 candidates; no label arrays read', flush=True)
    for name, digest in identity['scorer_sources'].items():
        if sha(repo/name) != digest: raise RuntimeError('Original scorer source changed: '+name)
    cfg = OmegaConf.create(identity['config'])
    if cfg.traffic_agents != 'non_reactive': raise ValueError('Non-reactive replay required')
    simulator, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    scorer._config.human_penalty_filter = False
    policy = instantiate(cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling)
    sampling = instantiate(cfg.agent_sampling)
    freeze(out/'identity.json', dict(request_sha256=sha(request_path), selection_sha256=sha(out/'selection.json'),
        source_identity_sha256=sha(source/'identity.json'), config=identity['config'],
        tools={p.name: sha(p) for p in Path(__file__).parent.iterdir() if p.is_file() and p.suffix in ('.py','.json')},
        versions={k:importlib.metadata.version(k) for k in ('numpy','shapely')}, buffers_m=list(math.BUFFERS_M)))
    keys = ['pdm_score', *base.METRICS]; rows = []; newly_scored = 0
    for case, token in sorted(cases.items()):
        if index[token]['metadata']['role'] != 'train': raise ValueError('Only training scenes allowed')
        example = checked(package, 'bundle/examples/'+token+'.npz')
        if sha(example) != hashes[token]: raise RuntimeError('Frozen example changed')
        # This is the first outcome-array lookup, after the entire 96-candidate selection is committed.
        labels = load_labels(example, samples[token])
        cache_info = identity['caches'][token]; cache_path = Path(cache_info['path'])
        if sha(cache_path) != cache_info['sha256']: raise RuntimeError('Metric cache changed')
        with lzma.open(cache_path, 'rb') as stream: cache_bytes = stream.read()
        reference_cache = pickle.loads(cache_bytes)
        if reference_cache.log_name != scenes[token]['log_name']: raise RuntimeError('Cache/camera log mismatch')
        map_indices = reference_cache.drivable_area_map.get_indices_of_map_type([
            SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
        road = math.road_union([reference_cache.drivable_area_map._geometries[i] for i in map_indices])
        reference_summary = read(source/(case+'-member0')/'summary.json')
        dimensions = {k:float(getattr(scorer._vehicle_parameters,k)) for k in reference_summary['scorer_vehicle_dimensions_m']}
        if dimensions != reference_summary['scorer_vehicle_dimensions_m']: raise RuntimeError('Vehicle dimensions changed')
        for candidate in [c for c in selection if c['case_id'] == case]:
            rank = candidate['teacher_rank']; directory = out/'candidates'/f'{case}-rank{rank:02d}'
            if (directory/'commit.json').exists():
                commit = read(directory/'commit.json')
                for name,digest in commit['files'].items():
                    if sha(checked(directory,name)) != digest: raise RuntimeError('Completed candidate artifact changed')
                row = read(directory/'summary.json')
                if any(row[k] != v for k,v in candidate.items()): raise RuntimeError('Completed candidate identity changed')
                if not np.array_equal(np.asarray(row['expected_raw']),labels[rank-1]): raise RuntimeError('Completed label identity changed')
            else:
                cache = pickle.loads(cache_bytes)  # Fresh observation for every single-candidate scoring call.
                trajectory = Trajectory(vocab[candidate['candidate_vocab']].astype(float), sampling)
                frame, states = pdm_score(cache, trajectory, simulator.proposal_sampling, simulator, scorer, policy)
                newly_scored += 1
                measured = np.asarray([float(frame.iloc[0][k]) for k in keys]); target = labels[rank-1]
                parity = bool(np.allclose(measured,target,atol=1e-8,rtol=1e-7))
                nominal = get_trajectory_as_array(transform_trajectory(trajectory,cache.ego_state),
                    simulator.proposal_sampling,cache.ego_state.time_point)
                coords = state_array_to_coords_array(states[None],scorer._vehicle_parameters)[0]
                if (states.shape[0] != 41 or nominal.shape != states.shape or coords.shape != (41,5,2) or
                        not np.isfinite(states).all() or not np.isfinite(nominal).all() or not np.isfinite(coords).all()):
                    raise RuntimeError('Invalid replay sampling/states')
                if not np.array_equal(coords,scorer._ego_coords[1]): raise RuntimeError('Scorer corner alignment changed')
                inside,_ = old.map_membership(cache,coords,SemanticMapLayer)
                if float(inside.all()) != measured[keys.index('drivable_area_compliance')]: raise RuntimeError('DAC reconstruction changed')
                distances = np.asarray([Point(p).distance(cache.centerline.linestring) for p in coords[:,4]])
                intersections = np.asarray([cache.drivable_area_map.is_in_layer(Point(p),SemanticMapLayer.INTERSECTION) for p in coords[:,4]])
                dt = float(simulator.proposal_sampling.interval_length)
                if abs(dt-.1)>1e-12: raise RuntimeError('Expected 0.1 s sampling')
                lane = old.geometry.lane_trace(distances,intersections,dt,
                    scorer._config.lane_keeping_deviation_limit,scorer._config.lane_keeping_horizon_window)
                if lane['score'] != measured[keys.index('lane_keeping')]: raise RuntimeError('Lane reconstruction changed')
                actors = [[(obj.track_token,obj.box.geometry) for obj in f.tracked_objects]
                          for f in scorer._observation.detections_tracks[:41]]
                if len(actors)!=41: raise RuntimeError('Traffic sampling changed')
                frames = [math.frame_measurement(c,road,a) for c,a in zip(coords,actors)]
                geometry = math.summarize(frames,dt,bool(inside.all()),lane)
                pair_parity = None
                if (case,candidate['candidate_local']) in known:
                    previous = known[case,candidate['candidate_local']]
                    with np.load(source/(case+'-member'+str(previous['member']))/'traces.npz',allow_pickle=False) as data:
                        pair_parity = bool(np.allclose(states,data['simulated_states'],atol=1e-8,rtol=1e-7))
                collision = scorer._collision_time_idcs[1]
                row = dict(candidate, **geometry, raw_pdm=float(measured[0]), progress_m=float(scorer._progress_raw[1]),
                    ego_progress=float(measured[keys.index('ego_progress')]), raw_metrics=dict(zip(keys,measured.tolist())),
                    measured_raw=measured.tolist(),expected_raw=target.tolist(),metric_parity_passed=parity,
                    max_metric_abs_difference=float(np.abs(measured-target).max()),known_pair_state_parity=pair_parity,
                    first_at_fault_time_s=None if not np.isfinite(collision) else float(collision*dt),
                    path_length_m=float(np.linalg.norm(np.diff(states[:,:2],axis=0),axis=1).sum()))
                directory.mkdir(parents=True,exist_ok=True)
                origin = np.asarray(cache.ego_state.rear_axle.serialize(),dtype=float)
                np.savez_compressed(directory/'rollout.npz', nominal_local=old.geometry.local_poses(nominal[:,:3],origin),
                    simulated_local=old.geometry.local_poses(states[:,:3],origin), simulated_states=states,
                    simulated_boxes_local=old.geometry.local_xy(coords,origin))
                write(directory/'summary.json',row)
                write(directory/'geometry.json',dict(frames=frames,lane=lane))
                freeze(directory/'commit.json',dict(files={n:sha(directory/n) for n in ('rollout.npz','summary.json','geometry.json')}))
            rows.append(row)
            print(f'TOP16 {len(rows)}/96 {case} rank={rank:02d} parity={row["metric_parity_passed"]}',flush=True)
            write(out/'progress.json',dict(status='running',completed_candidates=len(rows),total_candidates=96,
                scorer_calls_this_invocation=newly_scored))
    passed = sum(r['metric_parity_passed'] for r in rows)
    reference_passed = sum(r['known_pair_state_parity'] is True for r in rows)
    reference_checked = sum(r['known_pair_state_parity'] is not None for r in rows)
    ready = passed==96 and reference_passed==reference_checked==12
    comparisons = compare_scenes(rows,math.BUFFERS_M)
    report = dict(status='complete' if ready else 'parity_failed',scenes=6,candidates=96,metric_parity_passed=passed,
        known_pair_state_parity_passed=reference_passed,known_pair_candidates_checked=reference_checked,
        geometry_interpretation_ready=ready,scorer_calls_this_invocation=newly_scored,resumed_candidates=96-newly_scored,
        model_inference_calls=0,holdout_processed=0,scenes_summary=comparisons,per_candidate=rows,note=NOTE)
    write(out/'report.json',report); write(out/'progress.json',dict(status=report['status'],completed_candidates=96,total_candidates=96))
    columns=['case_id','teacher_rank','candidate_local','candidate_vocab','is_baseline','metric_parity_passed',
        'raw_pdm','progress_m','ego_progress','full_footprint_covered_all_samples','min_road_clearance_m',
        'max_outside_area_m2','min_corner_signed_margin_m','lane_keeping','max_lane_counter_samples',
        'min_actor_clearance_m','first_actor_contact_time_s','first_at_fault_time_s']
    with (out/'candidates.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps(report,indent=2),flush=True)
    return 0 if ready else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--repo',type=Path,required=True)
    args=p.parse_args();sys.path.insert(1,str(args.repo.resolve()))
    import fcntl
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.top16.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:return run(args.input.resolve(),args.out.resolve(),args.repo.resolve())
        except Exception as exc:
            value=dict(status='technical_failure',error=repr(exc),traceback=traceback.format_exc(),geometry_interpretation_ready=False,note=NOTE)
            write(args.out/'report.json',value);print(json.dumps(value,indent=2),flush=True);return 1


if __name__=='__main__':raise SystemExit(main())
