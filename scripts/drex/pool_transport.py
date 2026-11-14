"""Bounded compressed uploads and compact exports for the full CPU pool audit."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import rollout_bridge as old
from transfer import read, write, sha, remote_command

EXPORTS = ('report.json', 'identity.json', 'progress.json', 'oracle_choices.json', 'per_scene.csv.gz', 'per_candidate.csv.gz')


def receive_archive(destination, expected_sha):
    data = sys.stdin.buffer.read(32*1024**2+1)
    if len(data) > 32*1024**2 or hashlib.sha256(data).hexdigest() != expected_sha:
        raise RuntimeError('Pool upload checksum/32 MiB limit failed')
    root = Path(destination).resolve()
    if root.parent != Path('/workspace/exp_v2/drex_rollout_audit').resolve(): raise ValueError('Invalid upload root')
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        members = archive.getmembers()
        if (any(not m.isfile() for m in members) or len(members) > 64
                or len({m.name for m in members}) != len(members) or sum(m.size for m in members) > 128*1024**2):
            raise ValueError('Invalid/oversized expanded pool upload')
        manifest = json.loads(archive.extractfile('upload_manifest.json').read())
        if {m.name for m in members} != {'upload_manifest.json', *manifest}: raise ValueError('Upload membership changed')
        marker = root/'input_manifest.json'
        if marker.exists() and json.loads(marker.read_text()) != manifest: raise RuntimeError('Protocol changed; use a new name')
        if root.exists() and any(root.iterdir()) and not marker.exists(): raise RuntimeError('Uncommitted output root')
        payload = []
        for m in members:
            p = PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts or '\\' in m.name or ':' in m.name: raise ValueError('Unsafe upload path')
            target = root.joinpath(*p.parts)
            if root not in target.resolve().parents: raise ValueError('Upload escaped root')
            value = archive.extractfile(m).read()
            if m.name != 'upload_manifest.json':
                entry = manifest[m.name]
                if len(value) != entry['size'] or hashlib.sha256(value).hexdigest() != entry['sha256']:
                    raise RuntimeError('Upload member checksum failed')
            if target.exists() and target.read_bytes() != value: raise RuntimeError('Existing pool tool/input changed')
            payload.append((target, value))
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(manifest, sort_keys=True))
        for path, value in payload:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(path.suffix+'.tmp'); temp.write_bytes(value); temp.replace(path)
    print('Pool audit inputs staged: '+str(root), flush=True)


def result_archive(destination):
    root = Path(destination).resolve(); artifacts = root/'artifacts'
    if root.parent != Path('/workspace/exp_v2/drex_rollout_audit').resolve() or not (artifacts/'report.json').is_file():
        raise RuntimeError('No pool audit report')
    allowed = ('report.json', 'identity.json', 'progress.json', 'oracle_choices.json', 'per_scene.csv.gz', 'per_candidate.csv.gz')
    files = {}
    for name in allowed:
        p = artifacts/name
        if p.is_symlink(): raise ValueError('Symlink in pool exports')
        if p.exists():
            value = p.read_bytes(); files[name] = dict(size=len(value), sha256=hashlib.sha256(value).hexdigest())
    if sum(v['size'] for v in files.values()) > 90*1024**2: raise RuntimeError('Pool export exceeds 90 MiB')
    (artifacts/'artifact_manifest.json').write_text(json.dumps(files, sort_keys=True))
    path = root/'artifacts.tar'
    with tarfile.open(path, 'w') as archive:
        for name in ('artifact_manifest.json', *sorted(files)): archive.add(artifacts/name, arcname=name, recursive=False)
    value = path.read_bytes()
    print(json.dumps(dict(archive=str(path), size=len(value), sha256=hashlib.sha256(value).hexdigest())))


def upload(out, tools):
    files = {'audit_request.json': (out/'audit_request.json').read_bytes()}
    files.update({'tools/'+n: Path(__file__).with_name(n).read_bytes() for n in tools})
    files['upload_manifest.json'] = json.dumps({n: dict(size=len(v), sha256=hashlib.sha256(v).hexdigest()) for n, v in files.items()}, sort_keys=True).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, value in sorted(files.items()):
            info = tarfile.TarInfo(name); info.size = len(value); archive.addfile(info, io.BytesIO(value))
    data = buffer.getvalue()
    if len(data) > 32*1024**2: raise RuntimeError('Compressed request too large')
    return data


def execute(args, tools, stage):
    out = args.root/'runs'/args.name; destination = old.REMOTE_BASE+'/'+args.name
    command = lambda code: remote_command(args.host, args.port, args.control.expanduser(), args.container, code)
    data = upload(out, tools)
    subprocess.run(command(old.remote_python(receive_archive)+f'receive_archive({destination!r},{hashlib.sha256(data).hexdigest()!r})'), input=data, check=True)
    tool = destination+'/tools/pool_remote.py'
    argv = [tool, '--input', destination+'/audit_request.json', '--out', destination+'/artifacts', '--repo', args.remote_repo, '--stage', stage]
    code = ('import os,sys,runpy; os.environ["CUDA_VISIBLE_DEVICES"]=""; '
        'os.environ["OMP_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; '
        f'sys.dont_write_bytecode=True; os.chdir({args.remote_repo!r}); sys.path.insert(0,{destination+"/tools"!r}); '
        f'sys.argv={argv!r}; runpy.run_path({tool!r},run_name="__main__")')
    result = subprocess.run(command(code), check=False)
    info = json.loads(subprocess.check_output(command(old.remote_python(result_archive)+f'result_archive({destination!r})'), text=True))
    if info['archive'] != destination+'/artifacts.tar' or not 0 < info['size'] <= old.MAX_RESULT_BYTES:
        raise RuntimeError('Unexpected pool export size/path')
    target = out/('audit-'+info['sha256']+'.tar')
    if not target.exists():
        part = target.with_suffix('.tar.part')
        with part.open('wb') as f:
            subprocess.run(command('import sys,shutil; '+f'f=open({info["archive"]!r},"rb"); shutil.copyfileobj(f,sys.stdout.buffer,1048576)'), stdout=f, check=True)
        if part.stat().st_size != info['size'] or sha(part) != info['sha256']: raise RuntimeError('Pool export checksum failed')
        part.replace(target)
    if sha(target) != info['sha256']: raise RuntimeError('Local pool archive changed')
    folder = out/('result_'+info['sha256'][:12]); report = old.extract_result(target, folder)
    write(out/'latest_result.json', dict(result=str(folder), archive_sha256=info['sha256'], remote_exit_code=result.returncode))
    print(json.dumps({k:v for k,v in report.items() if k not in ('oracles', 'config')}, indent=2), flush=True)
    print('Pool result: '+str(folder), flush=True); return result.returncode
