"""Detach pool supervision inside Astra; msu270 only stages and fetches files."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess

import pool_audit
import pool_transport as transport
from transfer import read, write, sha, remote_command


def supervise(destination, repo):
    import fcntl, hashlib, json, os, subprocess, sys, time, traceback
    from pathlib import Path
    root = Path(destination).resolve()
    if root.parent != Path('/workspace/exp_v2/drex_rollout_audit').resolve(): raise ValueError('Invalid direct root')
    def save(value):
        p = root/'direct_job.json'; t = p.with_suffix('.json.tmp')
        t.write_text(json.dumps(value, indent=2)); t.replace(p)
    with (root/'direct.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return 0
        execution = root/'artifacts/.execution.lock'
        if execution.exists():
            with execution.open('r+') as other:
                try: fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    save(dict(state='external_worker_running', pid=os.getpid())); return 0
        with (root/'direct.log').open('a', buffering=1) as log:
            os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
            request_sha = hashlib.sha256((root/'audit_request.json').read_bytes()).hexdigest()
            state = dict(state='running', pid=os.getpid(), request_sha256=request_sha, started=time.time())
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
            try:
                for phase in ('pilot', 'full'):
                    state.update(phase=phase, state='running'); save(state)
                    print('DIRECT POOL '+phase, flush=True)
                    argv = [sys.executable, '-u', str(root/'tools/pool_remote.py'), '--input', str(root/'audit_request.json'),
                        '--out', str(root/'artifacts'), '--repo', repo, '--stage', phase]
                    result = subprocess.run(argv, cwd=repo, env=env)
                    if result.returncode: raise RuntimeError('Pool '+phase+' exit='+str(result.returncode))
                    report = json.loads((root/'artifacts/report.json').read_text())
                    identity = json.loads((root/'artifacts/identity.json').read_text())
                    if (identity['request_sha256'] != request_sha
                            or report['status'] not in ('pilot_complete', 'complete')):
                        raise RuntimeError('Pilot/full acceptance failed')
                    if report['status'] == 'complete': break
                if report['status'] != 'complete': raise RuntimeError('Full pool incomplete')
                state.update(state='complete', finished=time.time(), exit_code=0); save(state); return 0
            except Exception as exc:
                traceback.print_exc()
                state.update(state='failed', error=repr(exc), finished=time.time(), exit_code=1); save(state); return 1


def install_worker(destination, source):
    import hashlib
    from pathlib import Path
    root = Path(destination).resolve()
    if root.parent != Path('/workspace/exp_v2/drex_rollout_audit').resolve() or not (root/'input_manifest.json').exists():
        raise ValueError('Stage scientific inputs first')
    path = root/'direct_worker.py'; value = source.encode()
    if path.exists() and path.read_bytes() != value: raise RuntimeError('Direct controller changed; inspect before replacement')
    if not path.exists():
        p = path.with_suffix('.py.tmp'); p.write_bytes(value); p.replace(path)
    print('Direct worker installed: '+str(path)+' SHA256='+hashlib.sha256(value).hexdigest())


def remote_status(destination):
    import fcntl, json
    from collections import deque
    from pathlib import Path
    root = Path(destination).resolve()
    if root.parent != Path('/workspace/exp_v2/drex_rollout_audit').resolve(): raise ValueError('Invalid direct root')
    value = dict(output=str(root), running=False)
    for name in ('direct.lock', 'artifacts/.execution.lock'):
        p = root/name
        if p.exists():
            with p.open('r+') as f:
                try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: value['running'] = True
    for key, name in [('job', 'direct_job.json'), ('progress', 'artifacts/progress.json'), ('report', 'artifacts/report.json')]:
        p = root/name
        if p.exists():
            v = json.loads(p.read_text()); value[key] = {k:x for k,x in v.items() if k not in ('oracles', 'config')}
    if (root/'direct.log').exists():
        with (root/'direct.log').open(errors='replace') as f: value['log_tail'] = list(deque(f, maxlen=15))
    print(json.dumps(value))


def start_job(destination, repo):
    import fcntl, json, os, subprocess, sys, time
    from pathlib import Path
    root = Path(destination).resolve()
    if root.parent != Path('/workspace/exp_v2/drex_rollout_audit').resolve() or not (root/'direct_worker.py').exists():
        raise ValueError('Direct worker not staged')
    with (root/'direct_launch.lock').open('a') as launch:
        fcntl.flock(launch, fcntl.LOCK_EX)
        for name in ('direct.lock', 'artifacts/.execution.lock'):
            p = root/name
            if p.exists():
                with p.open('r+') as f:
                    try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        print(json.dumps(dict(state='already_running', output=str(root)))); return
        process = subprocess.Popen([sys.executable, '-u', str(root/'direct_worker.py'), str(root), repo],
            cwd=repo, start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, close_fds=True)
        for _ in range(30):
            p = root/'direct_job.json'
            if p.exists() and json.loads(p.read_text()).get('pid') == process.pid: break
            if process.poll() is not None: break
            time.sleep(.1)
        print(json.dumps(dict(state='launched', pid=process.pid, output=str(root), log=str(root/'direct.log'))))


def code(function, *args):
    return inspect.getsource(function)+'\n'+function.__name__+'('+','.join(repr(x) for x in args)+')'


def fetch(args, command, destination, out):
    status = json.loads(subprocess.check_output(command(code(remote_status, destination)), text=True))
    if status['running'] or status.get('report', {}).get('status') != 'complete':
        raise RuntimeError('Fetch requires a stopped, completed audit; use status first')
    info = json.loads(subprocess.check_output(command(transport.old.remote_python(transport.result_archive)+
        f'result_archive({destination!r})'), text=True))
    if info['archive'] != destination+'/artifacts.tar' or not 0 < info['size'] <= transport.old.MAX_RESULT_BYTES:
        raise RuntimeError('Unexpected export size/path')
    archive = out/('audit-'+info['sha256']+'.tar')
    if not archive.exists():
        part = archive.with_suffix('.tar.part')
        with part.open('wb') as stream:
            subprocess.run(command('import sys,shutil; '+f'f=open({info["archive"]!r},"rb"); shutil.copyfileobj(f,sys.stdout.buffer,1048576)'), stdout=stream, check=True)
        if sha(part) != info['sha256'] or part.stat().st_size != info['size']: raise RuntimeError('Export checksum failed')
        part.replace(archive)
    if sha(archive) != info['sha256']: raise RuntimeError('Existing archive changed')
    folder = out/('result_'+info['sha256'][:12]); report = transport.old.extract_result(archive, folder)
    if read(folder/'identity.json')['request_sha256'] != sha(out/'audit_request.json'): raise RuntimeError('Result request differs')
    write(out/'latest_result.json', dict(result=str(folder), archive_sha256=info['sha256'], remote_exit_code=0))
    print('Fetched completed result: '+str(folder)); print(json.dumps(report.get('overview'), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('stage','start','status','fetch'))
    p.add_argument('--root', type=Path, required=True); p.add_argument('--name', default='rollout_audit_pool64_v1')
    p.add_argument('--host', default='egorov@93.180.62.99'); p.add_argument('--port', type=int, default=27434)
    p.add_argument('--container', default='egorov_drivesuprim'); p.add_argument('--repo', default='/workspace/DriveSuprim')
    p.add_argument('--control', type=Path, default=Path.home()/'.ssh/drex-astra-transfer.sock')
    a = p.parse_args(); a.root = a.root.expanduser().resolve()
    if not re.fullmatch(r'rollout_audit_[A-Za-z0-9_-]{1,64}', a.name): raise ValueError('Invalid pool name')
    out = a.root/'runs'/a.name; destination = transport.old.REMOTE_BASE+'/'+a.name
    command = lambda s: remote_command(a.host, a.port, a.control.expanduser(), a.container, s)
    if a.stage in ('stage', 'start'):
        request = read(out/'audit_request.json')
        if read(out/'plan.json')['request_sha256'] != sha(out/'audit_request.json'): raise RuntimeError('Prepared request changed')
        for n, h in request['tools'].items():
            if sha(Path(__file__).with_name(n)) != h: raise RuntimeError('Scientific tool changed: '+n)
        data = transport.upload(out, pool_audit.TOOLS)
        subprocess.run(command(transport.old.remote_python(transport.receive_archive)+
            f'receive_archive({destination!r},{hashlib.sha256(data).hexdigest()!r})'), input=data, check=True)
        worker = inspect.getsource(supervise)+'\nif __name__ == "__main__":\n import sys\n raise SystemExit(supervise(sys.argv[1],sys.argv[2]))\n'
        subprocess.run(command(code(install_worker, destination, worker)), check=True)
    if a.stage == 'start': subprocess.run(command(code(start_job, destination, a.repo)), check=True)
    if a.stage == 'status':
        value = json.loads(subprocess.check_output(command(code(remote_status, destination)), text=True))
        for line in value.pop('log_tail', []): print(line, end='')
        print(json.dumps(value, indent=2))
    if a.stage == 'fetch': fetch(a, command, destination, out)


if __name__ == '__main__': main()
