"""Standard-library bridge: committed msu270 pairs -> CPU replay on Astra."""
import argparse
import hashlib
import inspect
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile

from transfer import sha, checked, freeze, read, write, remote_command

TOOLS = ('rollout_replay.py', 'rollout_audit_math.py', 'pair_diagnostic_data.py',
         'overlay_v2_data.py', 'overlay_data.py', 'real_data.py', 'runtime.py',
         'transfer.py', 'vehicle_profile_astra.json')
REMOTE_BASE = '/workspace/exp_v2/drex_rollout_audit'
MAX_RESULT_BYTES = 100*1024**2


def prepare_request(root, pair_out, out):
    plan = read(pair_out/'plan.json'); commit = read(pair_out/'decision_commit.json')
    report = read(pair_out/'report.json')
    if report['status'] != 'complete' or report['requests'] != 48 or report['cases'] != 6:
        raise RuntimeError('Completed six-case/48-request diagnostic required')
    for name, digest in plan['sources'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise RuntimeError('Pair diagnostic code differs from its frozen plan')
    for name, digest in {**plan['files'], **plan['previews']}.items():
        if sha(pair_out/name) != digest:
            raise RuntimeError('Pair input changed: '+name)
    if (commit['plan_sha256'] != sha(pair_out/'plan.json') or
            commit['decisions_sha256'] != sha(pair_out/'decisions.json') or
            commit['expected_sha256'] != sha(pair_out/'expected.json')):
        raise RuntimeError('Pair decision commitment changed')
    requests = read(pair_out/'requests.json')
    names = {r['request_id']+'.json' for r in requests}
    if set(commit['results']) != names or {p.name for p in (pair_out/'results').glob('*.json')} != names:
        raise RuntimeError('Incomplete/extra pair results')
    for name, digest in commit['results'].items():
        if sha(pair_out/'results'/name) != digest:
            raise RuntimeError('Pair result changed')
    package = root/'data/navsim_train_v1/package'
    if sha(package/'package_manifest.json') != plan['package_manifest_sha256']:
        raise RuntimeError('Portable package manifest changed')
    manifest = read(package/'package_manifest.json')
    pairs = read(pair_out/'pair_inputs.json')
    truth = {r['pair_id']: r for r in read(pair_out/'expected.json')}
    if len(pairs) != 6 or len(truth) != 6:
        raise RuntimeError('Unexpected pair membership')
    candidates = []
    for pair in pairs:
        value = truth[pair['pair_id']]
        if pair['token'] != value['token'] or len(set(pair['local_ids'])) != 2 or len(set(pair['vocab_ids'])) != 2:
            raise RuntimeError('Invalid diagnostic pair identity')
        for member in (0, 1):
            candidates.append(dict(pair_id=pair['pair_id'], token=pair['token'], member=member,
                candidate_local=pair['local_ids'][member], candidate_vocab=pair['vocab_ids'][member],
                focus_metric=value['focus_metric'], expected_raw=value['raw_scores'][member],
                metric_keys=value['metric_keys'], better_member=value['better_member'],
                example_sha256=manifest['files']['bundle/examples/'+pair['token']+'.npz']['sha256']))
    request = dict(protocol='nominal_rollout_audit_v1', candidates=candidates,
        package_manifest_sha256=plan['package_manifest_sha256'],
        pair_plan_sha256=sha(pair_out/'plan.json'), pair_decision_commit_sha256=sha(pair_out/'decision_commit.json'),
        vehicle_profile=read(Path(__file__).with_name('vehicle_profile_astra.json')),
        note='Privileged post-hoc CPU replay only. Map, future traffic, outcomes and simulated traces are not Drex inputs.')
    out.mkdir(parents=True, exist_ok=True)
    freeze(out/'audit_request.json', request)
    return request


def input_archive(out):
    content = {'audit_request.json': (out/'audit_request.json').read_bytes()}
    content.update({'tools/'+name: Path(__file__).with_name(name).read_bytes() for name in TOOLS})
    manifest = {name: dict(size=len(value), sha256=hashlib.sha256(value).hexdigest()) for name, value in content.items()}
    content['upload_manifest.json'] = json.dumps(manifest, sort_keys=True).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name, value in sorted(content.items()):
            info = tarfile.TarInfo(name); info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    return buffer.getvalue()


def receive_archive(destination, expected_sha):
    """Runs inside the user's existing container. Only writes a new audit root."""
    data = sys.stdin.buffer.read(4*1024**2+1)
    if len(data) > 4*1024**2 or hashlib.sha256(data).hexdigest() != expected_sha:
        raise RuntimeError('Audit upload size/checksum mismatch')
    root = Path(destination).resolve()
    if root.parent != Path('/workspace/exp_v2/drex_rollout_audit'):
        raise ValueError('Upload must stay in dedicated audit directory')
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
        members = archive.getmembers()
        if any(not m.isfile() for m in members) or len({m.name for m in members}) != len(members):
            raise ValueError('Nonregular/duplicate audit upload')
        manifest = json.loads(archive.extractfile('upload_manifest.json').read())
        if {m.name for m in members} != {'upload_manifest.json', *manifest}:
            raise ValueError('Unexpected upload membership')
        marker = root/'input_manifest.json'
        if marker.exists() and json.loads(marker.read_text()) != manifest:
            raise RuntimeError('Audit code/request changed; use a new audit name')
        if root.exists() and any(root.iterdir()) and not marker.exists():
            raise RuntimeError('Nonempty uncommitted audit directory')
        payload = []
        for name, entry in manifest.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name:
                raise ValueError('Unsafe audit upload path')
            target = root.joinpath(*relative.parts)
            if root not in target.resolve().parents:
                raise ValueError('Audit upload path escaped root')
            value = archive.extractfile(name).read()
            if len(value) != entry['size'] or hashlib.sha256(value).hexdigest() != entry['sha256']:
                raise RuntimeError('Audit member changed')
            if target.exists() and target.read_bytes() != value:
                raise RuntimeError('Existing audit input changed')
            payload.append((target, value))
        root.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text(json.dumps(manifest, sort_keys=True))
        for target, value in payload:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_suffix(target.suffix+'.tmp'); temp.write_bytes(value); temp.replace(target)
    print('Audit inputs staged in '+str(root), flush=True)


def result_archive(destination):
    """Runs remotely after replay; manifest all regular audit artifacts."""
    root = Path(destination).resolve(); artifacts = root/'artifacts'
    if root.parent != Path('/workspace/exp_v2/drex_rollout_audit') or not (artifacts/'report.json').is_file():
        raise RuntimeError('Replay produced no report; inspect the bridge log')
    files = {}
    for path in artifacts.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlink in audit artifacts')
        if path.is_file() and path.name != 'artifact_manifest.json' and not path.name.endswith('.tmp'):
            value = path.read_bytes()
            files[path.relative_to(artifacts).as_posix()] = dict(size=len(value), sha256=hashlib.sha256(value).hexdigest())
    if sum(e['size'] for e in files.values()) > 90*1024**2:
        raise RuntimeError('Audit artifacts exceed the 90 MiB output budget')
    (artifacts/'artifact_manifest.json').write_text(json.dumps(files, sort_keys=True))
    archive_path = root/'artifacts.tar'
    with tarfile.open(archive_path, mode='w') as archive:
        for name in ['artifact_manifest.json', *sorted(files)]:
            archive.add(artifacts/name, arcname=name, recursive=False)
    value = archive_path.read_bytes()
    print(json.dumps(dict(archive=str(archive_path), size=len(value), sha256=hashlib.sha256(value).hexdigest())))


def extract_result(archive_path, out):
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode='r:') as archive:
        members = archive.getmembers()
        if any(not m.isfile() for m in members) or len({m.name for m in members}) != len(members):
            raise ValueError('Invalid result archive members')
        manifest = json.loads(archive.extractfile('artifact_manifest.json').read())
        if {m.name for m in members} != {'artifact_manifest.json', *manifest}:
            raise ValueError('Unexpected result archive files')
        # Validate every path and size before writing any artifact.
        for member in members:
            checked(out, member.name)
            if member.name != 'artifact_manifest.json' and member.size != manifest[member.name]['size']:
                raise RuntimeError('Result member size mismatch')
        for member in members:
            target = checked(out, member.name)
            if target.exists():
                if member.name != 'artifact_manifest.json' and sha(target) != manifest[member.name]['sha256']:
                    raise RuntimeError('Existing result artifact changed')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix+'.tmp')
            with archive.extractfile(member) as src, tmp.open('wb') as sink:
                shutil.copyfileobj(src, sink)
            if member.name != 'artifact_manifest.json' and sha(tmp) != manifest[member.name]['sha256']:
                raise RuntimeError('Result artifact checksum mismatch')
            tmp.replace(target)
    return read(out/'report.json')


def remote_python(function):
    return ('import sys,io,json,hashlib,tarfile\nfrom pathlib import Path,PurePosixPath\n'+
            inspect.getsource(function)+'\n')


def run(args):
    root = args.root.expanduser().resolve()
    if not re.fullmatch(r'rollout_audit_[A-Za-z0-9_-]{1,64}', args.name):
        raise ValueError('Use a separate rollout_audit_* name')
    if not re.fullmatch(r'pair_diag_[A-Za-z0-9_-]{1,64}', args.pair):
        raise ValueError('Expected a completed pair_diag_* run')
    out = root/'runs'/args.name
    prepare_request(root, root/'runs'/args.pair, out)
    command = lambda code: remote_command(args.host, args.port, args.control.expanduser(), args.container, code)
    destination = REMOTE_BASE+'/'+args.name
    data = input_archive(out)
    code = remote_python(receive_archive)+f'receive_archive({destination!r}, {hashlib.sha256(data).hexdigest()!r})'
    subprocess.run(command(code), input=data, check=True)
    tool = destination+'/tools/rollout_replay.py'
    argv = [tool, '--input', destination+'/audit_request.json', '--out', destination+'/artifacts',
            '--repo', args.remote_repo, '--package', args.remote_package]
    code = ('import os,sys,runpy; os.environ["CUDA_VISIBLE_DEVICES"]=""; '
            'os.environ["OMP_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"; '
            'os.environ["MKL_NUM_THREADS"]="2"; os.environ["PYTHONDONTWRITEBYTECODE"]="1"; sys.dont_write_bytecode=True; '
            f'os.chdir({args.remote_repo!r}); sys.path.insert(0,{destination+"/tools"!r}); '
            f'sys.argv={argv!r}; runpy.run_path({tool!r},run_name="__main__")')
    process = subprocess.run(command(code), check=False)
    code = remote_python(result_archive)+f'result_archive({destination!r})'
    info = json.loads(subprocess.check_output(command(code), text=True))
    if info['archive'] != destination+'/artifacts.tar' or not 0 < info['size'] <= MAX_RESULT_BYTES:
        raise RuntimeError('Unexpected result transfer size/path')
    archive_path = out/('audit-'+info['sha256']+'.tar')
    if not archive_path.exists():
        part = archive_path.with_suffix('.tar.part')
        code = ('import sys,shutil; '
                f'f=open({info["archive"]!r},"rb"); shutil.copyfileobj(f,sys.stdout.buffer,1048576)')
        with part.open('wb') as sink:
            subprocess.run(command(code), stdout=sink, check=True)
        if part.stat().st_size != info['size'] or sha(part) != info['sha256']:
            raise RuntimeError('Result transfer checksum/size mismatch')
        part.replace(archive_path)
    elif sha(archive_path) != info['sha256']:
        raise RuntimeError('Local result archive changed')
    result_out = out/('result_'+info['sha256'][:12])
    report = extract_result(archive_path, result_out)
    write(out/'latest_result.json', dict(result=str(result_out), archive_sha256=info['sha256'], remote_exit_code=process.returncode))
    print(json.dumps(report, indent=2), flush=True)
    print('Local audit artifacts: '+str(result_out), flush=True)
    return process.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', default='rollout_audit_pair6_v1')
    parser.add_argument('--pair', default='pair_diag_gpu4_v1')
    parser.add_argument('--host', default='egorov@93.180.62.99')
    parser.add_argument('--port', type=int, default=27434)
    parser.add_argument('--container', default='egorov_drivesuprim')
    parser.add_argument('--control', type=Path, default=Path.home()/'.ssh/drex-astra-transfer.sock')
    parser.add_argument('--remote-repo', default='/workspace/DriveSuprim')
    parser.add_argument('--remote-package', default='/workspace/exp_v2/drex_transfer/navsim_train_v1')
    raise SystemExit(run(parser.parse_args()))
