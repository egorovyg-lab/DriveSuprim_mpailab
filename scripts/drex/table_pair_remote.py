"""Astra CPU replay adapter for four committed top16 candidates; no GPU."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

import real_data as base
import rollout_replay as replay
from transfer import read, write, sha, freeze

NOTE = ('Post-hoc audit of two outcome-selected TRAIN pairs, four candidates. '
        'Member0=DS/camera, member1=table. No Drex calls, fitting or holdout. '
        'Map, scorer centerline and event-time actors are privileged audit-only data. '
        'Interpret geometry only after raw metric parity 4/4. Nominal vs simulated '
        'paths share the rear-axle frame; flat-ground camera projection is not an occlusion test.')


def validate_request(request):
    candidates = request['candidates']
    if request['protocol'] != 'table_pair_rollout_v1' or len(candidates) != 4:
        raise ValueError('Expected exactly four committed candidates')
    if {(c['pair_id'],c['member']) for c in candidates} != {(p,m) for p in ('case01','case02') for m in (0,1)}:
        raise ValueError('Unexpected/duplicate case members')
    for pair in ('case01', 'case02'):
        group = [c for c in candidates if c['pair_id'] == pair]
        if len({c['token'] for c in group}) != 1 or len({c['candidate_local'] for c in group}) != 2:
            raise ValueError('Invalid paired candidate identity')
        for c in group:
            values = np.asarray(c['expected_raw'], dtype=float)
            if (c['metric_keys'] != ['pdm_score', *base.METRICS] or c['better_member'] != 1
                    or values.shape != (9,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
                raise ValueError('Invalid metric schema/outcome values')


def run(request_path, out, repo):
    request = read(request_path); validate_request(request)
    package = Path(request['remote_package']).resolve()
    if sha(package/'package_manifest.json') != request['package_manifest_sha256']:
        raise RuntimeError('Portable package differs from msu270 snapshot')
    if request['vehicle_profile'] != read(Path(__file__).with_name('vehicle_profile_astra.json')):
        raise RuntimeError('Vehicle dimensions changed')
    sys.path.insert(1, str(repo)); replay.NOTE = NOTE
    manifest, scenes, vocab, template, caches = replay.bound_context(package, repo, request['candidates'])
    if template.get('map_root'): os.environ['NUPLAN_MAPS_ROOT'] = template['map_root']
    # Validate all frozen label rows BEFORE simulation or geometry interpretation.
    for c in request['candidates']:
        rel = 'bundle/examples/'+c['token']+'.npz'; path = package/rel
        if sha(path) != c['example_sha256'] or manifest['files'][rel]['sha256'] != c['example_sha256']:
            raise RuntimeError('Example bytes changed')
        with np.load(path, allow_pickle=False) as arrays:
            if int(arrays['candidate_ids'][c['candidate_local']]) != c['candidate_vocab']:
                raise RuntimeError('Local/vocabulary alignment changed')
            row = arrays['label_indices'].tolist().index(c['candidate_local'])
            if not np.array_equal(arrays['label_raw'][row], c['expected_raw']):
                raise RuntimeError('Expected raw labels differ from original top16')
        photo = base.camera_paths(package, scenes[c['token']], manifest)[4]
        if sha(photo) != manifest['files'][photo.relative_to(package).as_posix()]['sha256']:
            raise RuntimeError('Photograph changed')
    freeze(out/'identity.json', dict(request_sha256=sha(request_path), python=platform.python_version(),
        config=replay.config_core(template), caches=caches,
        tools={p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.json')},
        scorer_sources={n:sha(repo/n) for n in replay.REPLAY_SOURCES}))
    rows = []
    for c in request['candidates']:
        directory = out/(c['pair_id']+'-member'+str(c['member']))
        if (directory/'commit.json').exists():
            for name,digest in read(directory/'commit.json')['files'].items():
                if sha(directory/name) != digest: raise RuntimeError('Completed audit artifact changed')
            summary = read(directory/'summary.json')
        else:
            if (directory/'failure.json').exists():
                raise RuntimeError('Previous failure recorded; inspect and use a new audit name')
            try:
                summary = replay.replay_candidate(package, directory, c, caches[c['token']], template, scenes, vocab)
            except Exception as exc:
                write(directory/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc())); raise
        rows.append(summary)
        print(f'TABLE PAIR REPLAY {len(rows)}/4 parity={summary["parity_passed"]}', flush=True)
    passed = sum(r['parity_passed'] for r in rows)
    result = dict(status='complete' if passed == 4 else 'parity_failed', candidates=4,
        metric_parity_passed=passed, geometry_interpretation_ready=passed == 4,
        model_inference_calls=0, pairs=replay.pair_summaries(package,out,request,rows,scenes,manifest),
        per_candidate=[dict(pair_id=r['pair_id'], member=r['member'], parity_passed=r['parity_passed'],
            measured_raw=r['measured_raw'], nominal_vs_simulated=r['nominal_vs_simulated'],
            simulated_dac=r['simulated_dac'], simulated_lane_keeping=r['simulated_lane_keeping'],
            first_at_fault_collision_time_s=r['first_at_fault_collision_time_s'],
            overlaps_at_recorded_at_fault_time=r['overlaps_at_recorded_at_fault_time']) for r in rows], note=NOTE)
    write(out/'report.json', result); print(json.dumps(result, indent=2), flush=True)
    return 0 if passed == 4 else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True); a = p.parse_args()
    import fcntl
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out.parent/'.replay.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another pair replay is running')
        try: return run(a.input.resolve(), a.out.resolve(), a.repo.resolve())
        except Exception as exc:
            value = dict(status='technical_failure', error=repr(exc), traceback=traceback.format_exc(),
                geometry_interpretation_ready=False, note=NOTE)
            write(a.out/'report.json', value); print(json.dumps(value, indent=2)); return 1


if __name__ == '__main__': raise SystemExit(main())
