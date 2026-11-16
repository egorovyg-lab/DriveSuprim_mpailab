"""Prepare cached DS+Drex inputs on msu270, then run a detached CPU probe on Astra."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import inspect
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

import pool_direct as direct
import pool_transport as transport
from transfer import read, freeze, sha, remote_command


def prepare(args):
    import numpy as np
    import consequence_data as data
    import consequence_model as model
    import consequence_run as runner
    import replacement

    out = args.root/'runs'/args.name; source = args.root/'runs'/args.source
    # Verify every cached query/physical ID before looking up outcome labels.
    if read(source/'report.json')['status'] != 'complete':
        raise RuntimeError('Completed cached replacement study required')
    identity, pairs, _, _ = replacement.collect(source)
    _, _, table_source, table_plan = replacement.context(source)
    samples = replacement.inputs(table_source, table_plan, identity['records'])
    records = [dict(token=r['token'], session_id=r['session_id'], role=r['role']) for r in identity['records']]
    if len(records) != 651 or len({r['session_id'] for r in records}) != 32 or any(r['role'] != 'train' for r in records):
        raise RuntimeError('This fixed probe requires the existing 651-scene, 32-session TRAIN cohort')
    grouped_pairs = defaultdict(list)
    for row in pairs:
        grouped_pairs[row['token']].append(row)
    x = np.stack([data.feature_block(samples[r['token']]) for r in records])
    preferences = np.stack([data.preference_block(samples[r['token']], grouped_pairs[r['token']]) for r in records])
    # Now open the previously SHA-verified continuous target archive.
    pool = args.root/'runs'/args.pool; pointer = read(pool/'latest_result.json')
    digest = pointer['archive_sha256']; archive = pool/('audit-'+digest+'.tar')
    folder = pool/('result_'+digest[:12])
    if (not re.fullmatch('[0-9a-f]{64}', digest) or pointer['remote_exit_code'] != 0
            or Path(pointer['result']).resolve() != folder.resolve() or sha(archive) != digest):
        raise RuntimeError('Verified completed pool archive required')
    report = transport.old.extract_result(archive, folder)
    pool_identity, pool_request = read(folder/'identity.json'), read(pool/'audit_request.json')
    if (report['status'] != 'complete' or report['required_candidates'] != 651*64
            or report['metric_parity_passed'] != 651*64
            or pool_identity['request_sha256'] != sha(pool/'audit_request.json')
            or Path(pool_request['source']).resolve() != table_source.resolve()):
        raise RuntimeError('Complete matching continuous pool required')
    labelled = defaultdict(list)
    with gzip.open(folder/'per_candidate.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            labelled[row['token']].append(row)
    if set(labelled) != {r['token'] for r in records}:
        raise RuntimeError('Pool/feature cohort mismatch')
    outcomes = np.stack([data.outcome_block(samples[r['token']], r['session_id'], labelled[r['token']]) for r in records])
    expected = {(c['token'], c['candidate_local']): c for c in pool_request['candidates']}
    for i, r in enumerate(records):
        for j, c in enumerate(samples[r['token']]['candidates']):
            entry = expected[(r['token'], c['local'])]
            if entry['candidate_vocab'] != c['vocab'] or not np.allclose(outcomes[i, j, :9], entry['expected_raw'], atol=1e-8, rtol=1e-8):
                raise RuntimeError('Raw score/physical candidate parity changed')
    features = dict(x=x, preferences=preferences, fold=model.folds([r['session_id'] for r in records]),
                    local_ids=np.array([[c['local'] for c in samples[r['token']]['candidates']] for r in records]),
                    vocab_ids=np.array([[c['vocab'] for c in samples[r['token']]['candidates']] for r in records]))
    for name, arrays in [('features.npz', features), ('labels.npz', dict(outcomes=outcomes))]:
        path = out/name
        if path.exists():
            old = runner.load_arrays(path)
            if set(old) != set(arrays) or any(not np.array_equal(old[k], v, equal_nan=True) for k, v in arrays.items()):
                raise RuntimeError('Prepared probe arrays changed; use a new name')
        else:
            runner.save_arrays(path, **arrays)
    request = dict(protocol='drex_consequence_probe_v1', config=model.CONFIG, records=records,
                   features=list(data.FEATURES), outcomes=list(data.OUTCOMES),
                   files={n: sha(out/n) for n in ('features.npz', 'labels.npz')},
                   tools={n: sha(Path(__file__).with_name(n)) for n in runner.TOOLS},
                   source=dict(replacement_commit_sha256=sha(source/'decision_commit.json'),
                               pool_archive_sha256=digest, pool_request_sha256=sha(pool/'audit_request.json')),
                   note=model.NOTE)
    freeze(out/'audit_request.json', request)
    y, _ = data.truth(outcomes)
    plan = dict(status='planned', scenes=len(records), sessions=32, candidate_pairs=int(y.shape[0]*y.shape[1]),
                positive_benefit_actions=int(y[:, :, 6].sum()), request_sha256=sha(out/'audit_request.json'),
                model_fits=4*2*5, new_drex_calls=0, new_scorer_calls=0, holdout_processed=0)
    freeze(out/'plan.json', plan)
    print(json.dumps(plan, indent=2))


def upload(out):
    request = read(out/'audit_request.json')
    if read(out/'plan.json')['request_sha256'] != sha(out/'audit_request.json'):
        raise RuntimeError('Frozen probe request changed')
    files = {'audit_request.json': (out/'audit_request.json').read_bytes()}
    for name, digest in request['files'].items():
        if name not in ('features.npz', 'labels.npz') or sha(out/name) != digest:
            raise RuntimeError('Prepared dataset changed')
        files[name] = (out/name).read_bytes()
    allowed = {'consequence_model.py', 'consequence_data.py', 'consequence_run.py', 'transfer.py'}
    if set(request['tools']) != allowed or set(request['files']) != {'features.npz', 'labels.npz'}:
        raise ValueError('Unexpected probe upload files')
    for name, digest in request['tools'].items():
        path = Path(__file__).with_name(name)
        if sha(path) != digest:
            raise RuntimeError('Probe code changed; keep this run frozen: '+name)
        files['tools/'+name] = path.read_bytes()
    files['upload_manifest.json'] = json.dumps({n: dict(size=len(v), sha256=hashlib.sha256(v).hexdigest())
                                               for n, v in files.items()}, sort_keys=True).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, value in sorted(files.items()):
            info = tarfile.TarInfo(name); info.size = len(value); archive.addfile(info, io.BytesIO(value))
    result = buffer.getvalue()
    if len(result) > 32*1024**2:
        raise RuntimeError('Probe upload exceeds 32 MiB; inspect before changing the transport limit')
    return result


def supervise(destination, repo):
    import fcntl, hashlib, json, os, subprocess, sys, time, traceback
    from pathlib import Path
    root = Path(destination).resolve()
    if root.parent != Path('/workspace/exp_v2/drex_rollout_audit').resolve():
        raise ValueError('Invalid probe root')
    def save(value):
        p = root/'direct_job.json'; t = p.with_suffix('.json.tmp')
        t.write_text(json.dumps(value, indent=2)); t.replace(p)
    with (root/'direct.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return 0
        with (root/'direct.log').open('a', buffering=1) as log:
            os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
            request_sha = hashlib.sha256((root/'audit_request.json').read_bytes()).hexdigest()
            state = dict(state='running', pid=os.getpid(), request_sha256=request_sha, started=time.time())
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
            try:
                for phase in ('pilot', 'full'):
                    state.update(phase=phase); save(state)
                    print('CONSEQUENCE PROBE '+phase, flush=True)
                    child = subprocess.run([sys.executable, '-u', str(root/'tools/consequence_run.py'),
                        '--input', str(root/'audit_request.json'), '--out', str(root/'artifacts'), '--stage', phase],
                        cwd=repo, env=env)
                    if child.returncode: raise RuntimeError('Probe '+phase+' exit='+str(child.returncode))
                    report = json.loads((root/'artifacts/report.json').read_text())
                    identity = json.loads((root/'artifacts/identity.json').read_text())
                    if identity['request_sha256'] != request_sha or report['status'] not in ('pilot_complete', 'complete'):
                        raise RuntimeError('Probe pilot/full acceptance failed')
                if report['status'] != 'complete': raise RuntimeError('Probe incomplete')
                state.update(state='complete', finished=time.time(), exit_code=0); save(state); return 0
            except Exception as exc:
                traceback.print_exc()
                state.update(state='failed', error=repr(exc), finished=time.time(), exit_code=1); save(state); return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'stage', 'start', 'status', 'fetch'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', default='rollout_audit_drex_probe_v1')
    p.add_argument('--source', default='replacement_gpu2_v1'); p.add_argument('--pool', default='rollout_audit_pool64_v1')
    p.add_argument('--host', default='egorov@93.180.62.99'); p.add_argument('--port', type=int, default=27434)
    p.add_argument('--container', default='egorov_drivesuprim'); p.add_argument('--repo', default='/workspace/DriveSuprim')
    p.add_argument('--control', type=Path, default=Path.home()/'.ssh/drex-astra-transfer.sock')
    a = p.parse_args(); a.root = a.root.expanduser().resolve()
    if (not re.fullmatch(r'rollout_audit_[A-Za-z0-9_-]{1,64}', a.name) or a.name == a.pool
            or not re.fullmatch('[A-Za-z0-9_-]+', a.source) or not re.fullmatch('rollout_audit_[A-Za-z0-9_-]+', a.pool)):
        raise ValueError('Use separate, safe frozen run names')
    if a.stage == 'prepare':
        prepare(a); return
    out = a.root/'runs'/a.name; destination = transport.old.REMOTE_BASE+'/'+a.name
    command = lambda s: remote_command(a.host, a.port, a.control.expanduser(), a.container, s)
    if a.stage in ('stage', 'start'):
        payload = upload(out)
        subprocess.run(command(transport.old.remote_python(transport.receive_archive)+
            f'receive_archive({destination!r},{hashlib.sha256(payload).hexdigest()!r})'), input=payload, check=True)
        worker = inspect.getsource(supervise)+'\nif __name__ == "__main__":\n import sys\n raise SystemExit(supervise(sys.argv[1],sys.argv[2]))\n'
        subprocess.run(command(direct.code(direct.install_worker, destination, worker)), check=True)
    if a.stage == 'start':
        subprocess.run(command(direct.code(direct.start_job, destination, a.repo)), check=True)
    if a.stage == 'status':
        value = json.loads(subprocess.check_output(command(direct.code(direct.remote_status, destination)), text=True))
        for line in value.pop('log_tail', []): print(line, end='')
        if 'report' in value:
            value['report'] = {k: value['report'][k] for k in (
                'status', 'scenes', 'sessions', 'completed_folds', 'held_out_pairs') if k in value['report']}
        print(json.dumps(value, indent=2))
    if a.stage == 'fetch':
        direct.fetch(a, command, destination, out)


if __name__ == '__main__':
    main()
