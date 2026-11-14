"""Astra CPU-only full top64 replay with scene checkpoints and verified reuse."""
import argparse
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

import continuous_remote as previous
import pool_reserve as spec
from transfer import read, write, sha, freeze


def load_block(folder, record, candidates, identity_sha):
    commit = read(folder/'commit.json')
    if (commit['identity_sha256'] != identity_sha or commit['record'] != record or commit['candidates'] != candidates
            or set(commit['files']) != {'summaries.json.gz', 'states.npz'}):
        raise RuntimeError('Scene checkpoint binding changed')
    for name, digest in commit['files'].items():
        if sha(folder/name) != digest: raise RuntimeError('Scene checkpoint bytes changed')
    with gzip.open(folder/'summaries.json.gz', 'rt', encoding='utf-8') as f: group = json.load(f)
    if len(group) != 64: raise RuntimeError('Partial scene checkpoint')
    for c, s in zip(candidates, group):
        if (s['token'], s['candidate_local'], s['candidate_vocab'], s['teacher_rank']) != (
                c['token'], c['candidate_local'], c['candidate_vocab'], c['teacher_rank']):
            raise RuntimeError('Scene checkpoint candidate order changed')
        previous.parity(s['measured_raw'], c['expected_raw'])
    return group


def reuse_context(request, current_identity, root=None):
    root = Path('/workspace/exp_v2/drex_rollout_audit') if root is None else root
    binding = request['reuse']; source = root/binding['name']
    if source.parent != root: raise ValueError('Invalid reuse root')
    out = source/'artifacts'
    if (sha(out/'identity.json') != binding['identity_sha256']
            or sha(out/'report.json') != binding['report_sha256'] or sha(source/'audit_request.json') != binding['request_sha256']):
        raise RuntimeError('Original continuous audit changed')
    identity = read(out/'identity.json'); report = read(out/'report.json')
    if report['status'] != 'complete' or report['metric_parity_passed'] != len(binding['candidate_commits']):
        raise RuntimeError('Complete reusable source required')
    for key in ('python', 'packages', 'config', 'caches', 'sources'):
        if identity[key] != current_identity[key]: raise RuntimeError('Reuse environment/cache/config differs: '+key)
    for name in ('continuous_metrics.py', 'continuous_remote.py', 'clearance_math.py', 'rollout_replay.py', 'real_data.py'):
        if identity['tools'][name] != request['tools'][name]: raise RuntimeError('Reuse metric implementation changed')
    expected = {previous.candidate_key(c): c for c in request['candidates']}
    for key, digest in binding['candidate_commits'].items():
        if key not in expected or sha(out/'candidates'/key/'commit.json') != digest:
            raise RuntimeError('Reuse candidate commitment differs')
    return out


def run(path, out, repo, stage):
    request = read(path); spec.validate(request); package = Path(request['remote_package']).resolve()
    if previous.sha(package/'package_manifest.json') != request['package_manifest_sha256']: raise RuntimeError('Package changed')
    for name, digest in request['tools'].items():
        if sha(Path(__file__).with_name(name)) != digest: raise RuntimeError('Pool tool changed: '+name)
    if request['vehicle_profile'] != read(Path(__file__).with_name('vehicle_profile_astra.json')): raise RuntimeError('Vehicle changed')
    sys.path.insert(1, str(repo))
    manifest, scenes, vocab, template, caches = previous.replay.bound_context(package, repo, request['candidates'])
    if template.get('map_root'): os.environ['NUPLAN_MAPS_ROOT'] = template['map_root']
    by_token = {r['token']: [] for r in request['records']}
    for c in request['candidates']: by_token[c['token']].append(c)
    for token, group in by_token.items():
        group.sort(key=lambda c: c['teacher_rank']); example = package/'bundle/examples'/(token+'.npz')
        if sha(example) != group[0]['example_sha256'] or any(c['example_sha256'] != group[0]['example_sha256'] for c in group):
            raise RuntimeError('Example changed')
        with np.load(example, allow_pickle=False) as a:
            for c in group:
                if int(a['candidate_ids'][c['candidate_local']]) != c['candidate_vocab']: raise RuntimeError('Vocabulary alignment changed')
                if c['candidate_local'] in a['label_indices']:
                    row = a['label_indices'].tolist().index(c['candidate_local'])
                    previous.parity(a['label_raw'][row], c['expected_raw'])
    packages = {}
    for name in ('numpy', 'scipy', 'shapely', 'nuplan-devkit', 'hydra-core'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    identity = dict(request_sha256=sha(path), python=platform.python_version(), packages=packages,
        config=previous.replay.config_core(template), caches=caches, tools=request['tools'],
        sources={n:sha(repo/n) for n in (*previous.replay.REPLAY_SOURCES, *previous.EXTRA_SOURCES)})
    freeze(out/'identity.json', identity); identity_sha = sha(out/'identity.json')
    source = reuse_context(request, identity); reuse = request['reuse']
    records = request['records']; lookup = {r['token']: r for r in records}
    groups = {}; resumed = 0; new_calls = 0; reused = 0; start = time.monotonic()
    for r in records:
        folder = out/'scenes'/r['token']
        if (folder/'commit.json').exists():
            groups[r['token']] = load_block(folder, r, by_token[r['token']], identity_sha); resumed += 64
    if stage == 'full' and not set(request['pilot_tokens']) <= set(groups): raise RuntimeError('Full pool replay requires committed pilot')
    target = [lookup[t] for t in request['pilot_tokens']] if stage == 'pilot' else records
    for r in target:
        token = r['token']; folder = out/'scenes'/token
        if token in groups: continue
        if (folder/'failure.json').exists(): raise RuntimeError('Previous candidate failure retained; inspect before retry')
        summaries = []; arrays = []
        for c in by_token[token]:
            key = previous.candidate_key(c)
            try:
                if key in reuse['candidate_commits']:
                    original = source/'candidates'/key
                    summary = previous.verify_completed(original, c, reuse['identity_sha256'])
                    with np.load(original/'traces.npz', allow_pickle=False) as a: states = {n:a[n].copy() for n in a.files}
                    reused += 1
                else:
                    summary, states, _ = previous.evaluate(c, caches[token], template, scenes, vocab, request['vehicle_profile']['dimensions_m'])
                    new_calls += 1
                previous.parity(summary['measured_raw'], c['expected_raw'])
                if (summary['token'], summary['candidate_local'], summary['candidate_vocab'], summary['teacher_rank']) != (
                        c['token'], c['candidate_local'], c['candidate_vocab'], c['teacher_rank']):
                    raise RuntimeError('Measured candidate alignment differs')
                summaries.append(summary); arrays.append(states)
            except Exception as exc:
                write(folder/'failure.json', dict(candidate=c, error=repr(exc), traceback=traceback.format_exc()))
                raise RuntimeError(f'Pool failure token={token} local={c["candidate_local"]}: {exc}') from exc
        folder.mkdir(parents=True, exist_ok=True)
        temp = folder/'summaries.json.gz.tmp'
        with gzip.open(temp, 'wt', encoding='utf-8') as f: json.dump(summaries, f, allow_nan=False)
        temp.replace(folder/'summaries.json.gz')
        temp = folder/'states.tmp.npz'
        np.savez_compressed(temp, candidate_local=np.array([c['candidate_local'] for c in by_token[token]]),
            **{name:np.stack([a[name] for a in arrays]) for name in ('simulated_states', 'center_states', 'scorer_coords')})
        temp.replace(folder/'states.npz')
        freeze(folder/'commit.json', dict(identity_sha256=identity_sha, record=r, candidates=by_token[token],
            reused_candidates=[c['candidate_local'] for c in by_token[token] if previous.candidate_key(c) in reuse['candidate_commits']],
            files={n:sha(folder/n) for n in ('summaries.json.gz', 'states.npz')}))
        groups[token] = summaries
        elapsed = time.monotonic()-start; done = len(groups)*64
        remaining = sum(64 for x in target if x['token'] not in groups)
        eta = remaining*elapsed/(new_calls+reused) if new_calls+reused else None
        progress = dict(completed_candidates=done, required_candidates=len(request['candidates']),
            new_scorer_calls=new_calls, reused_this_invocation=reused, resumed_candidates=resumed,
            seconds=elapsed, estimated_remaining_seconds=eta)
        write(out/'progress.json', progress)
        print(f'POOL {done}/{len(request["candidates"])} new={new_calls} reused={reused} resumed={resumed} '
            f'elapsed_s={elapsed:.1f} eta_s={eta:.1f}', flush=True)
    result = spec.report(request, groups, out, len(groups) == len(records))
    reused_committed = sum(previous.candidate_key(c) in reuse['candidate_commits']
        for c in request['candidates'] if c['token'] in groups)
    result.update(reused_candidates_committed=reused_committed,
        new_candidates_committed=len(groups)*64-reused_committed)
    result.update(new_scorer_calls_this_invocation=new_calls, reused_this_invocation=reused,
        resumed_candidates=resumed, seconds_this_invocation=time.monotonic()-start)
    write(out/'report.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('oracles','config')}, indent=2), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True); p.add_argument('--stage', choices=('pilot','full'), required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (a.out/'.execution.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another pool replay is running')
        try: return run(a.input.resolve(), a.out.resolve(), a.repo.resolve(), a.stage)
        except Exception as exc:
            write(a.out/'report.json', dict(status='technical_failure', error=repr(exc),
                traceback=traceback.format_exc(), model_inference_calls=0, holdout_processed=0, note=spec.NOTE))
            traceback.print_exc(); return 1


if __name__ == '__main__': raise SystemExit(main())
