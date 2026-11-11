"""Standard-library upload/download for the read-only Astra clearance audit."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

import rollout_bridge as bridge
from transfer import read, write, sha, checked, freeze, remote_command

TOOLS = tuple(dict.fromkeys(('clearance_remote.py', 'clearance_math.py', *bridge.TOOLS)))


def source_snapshot(root, source_name):
    if not re.fullmatch(r'rollout_audit_[A-Za-z0-9_-]{1,64}', source_name): raise ValueError('Invalid source audit name')
    source = root/'runs'/source_name; pointer = read(source/'latest_result.json')
    digest = pointer['archive_sha256']
    if not re.fullmatch('[0-9a-f]{64}', digest): raise ValueError('Invalid source archive SHA')
    archive_path = source/('audit-'+digest+'.tar'); result = (source/('result_'+digest[:12])).resolve()
    if pointer['remote_exit_code'] != 0 or Path(pointer['result']).resolve() != result or sha(archive_path) != digest:
        raise RuntimeError('Completed checksum-verified source audit required')
    with tarfile.open(archive_path, 'r:') as archive:
        members = archive.getmembers()
        if any(not m.isfile() for m in members) or len({m.name for m in members}) != len(members):
            raise ValueError('Invalid source archive members')
        manifest = json.loads(archive.extractfile('artifact_manifest.json').read())
        if {m.name for m in members} != {'artifact_manifest.json', *manifest}: raise ValueError('Unexpected source archive files')
        for name in manifest: checked(result, name)
    expected = {}
    def committed(name):
        path = checked(result, name); entry = manifest[name]
        if path.stat().st_size != entry['size'] or sha(path) != entry['sha256']:
            raise RuntimeError('Source audit artifact changed: '+name)
        expected[name] = entry['sha256']; return path
    report = read(committed('report.json')); identity = read(committed('identity.json'))
    if (report['status'] != 'complete' or report['candidates'] != 12 or report['metric_parity_passed'] != 12 or
            not report['geometry_interpretation_ready']): raise RuntimeError('Original metric parity 12/12 required')
    request = source/'audit_request.json'
    if sha(request) != identity['request_sha256']: raise RuntimeError('Source request changed')
    for pair in range(1, 7):
        for member in (0, 1):
            folder = f'case{pair:02d}-member{member}'; commit = read(committed(folder+'/commit.json'))
            for name in ('summary.json', 'traces.npz'):
                path = committed(folder+'/'+name)
                if sha(path) != commit['files'][name]: raise RuntimeError('Source candidate commitment changed')
    return dict(protocol='clearance_pair6_v1', source_name=source_name,
        source_artifacts=bridge.REMOTE_BASE+'/'+source_name+'/artifacts', source_files=expected,
        source_request_sha256=identity['request_sha256'], source_archive_sha256=digest,
        note='Privileged geometry audit only. No Drex input, scoring, simulation, training or holdout.')


def input_archive(out, tools=TOOLS):
    files = {'audit_request.json': (out/'audit_request.json').read_bytes()}
    files.update({'tools/'+name: Path(__file__).with_name(name).read_bytes() for name in tools})
    files['upload_manifest.json'] = json.dumps({n: dict(size=len(v), sha256=hashlib.sha256(v).hexdigest())
        for n, v in files.items()}, sort_keys=True).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name, value in sorted(files.items()):
            info = tarfile.TarInfo(name); info.size = len(value); archive.addfile(info, io.BytesIO(value))
    return buffer.getvalue()


def run(args, request=None, tools=TOOLS, remote_tool='clearance_remote.py'):
    if remote_tool not in tools or Path(remote_tool).name != remote_tool:
        raise ValueError('Remote entrypoint must be a staged tool')
    if not re.fullmatch(r'rollout_audit_[A-Za-z0-9_-]{1,64}', args.name) or args.name == args.source:
        raise ValueError('Use a separate rollout_audit_* output name')
    root = args.root.expanduser().resolve(); out = root/'runs'/args.name
    request = source_snapshot(root, args.source) if request is None else request
    out.mkdir(parents=True, exist_ok=True)
    freeze(out/'audit_request.json', request)
    destination = bridge.REMOTE_BASE+'/'+args.name
    command = lambda code: remote_command(args.host, args.port, args.control.expanduser(), args.container, code)
    data = input_archive(out, tools)
    code = bridge.remote_python(bridge.receive_archive)+f'receive_archive({destination!r}, {hashlib.sha256(data).hexdigest()!r})'
    subprocess.run(command(code), input=data, check=True)
    tool = destination+'/tools/'+remote_tool
    argv = [tool, '--input', destination+'/audit_request.json', '--out', destination+'/artifacts', '--repo', args.remote_repo]
    code = ('import os,sys,runpy; os.environ["CUDA_VISIBLE_DEVICES"]=""; '
        'os.environ["OMP_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"; '
        'os.environ["MKL_NUM_THREADS"]="2"; sys.dont_write_bytecode=True; '
        f'os.chdir({args.remote_repo!r}); sys.path.insert(0,{destination+"/tools"!r}); '
        f'sys.argv={argv!r}; runpy.run_path({tool!r},run_name="__main__")')
    process = subprocess.run(command(code), check=False)
    info = json.loads(subprocess.check_output(command(bridge.remote_python(bridge.result_archive)+
        f'result_archive({destination!r})'), text=True))
    if info['archive'] != destination+'/artifacts.tar' or not 0 < info['size'] <= bridge.MAX_RESULT_BYTES:
        raise RuntimeError('Unexpected output archive path/size')
    archive = out/('audit-'+info['sha256']+'.tar')
    if not archive.exists():
        part = archive.with_suffix('.tar.part')
        code = 'import sys,shutil; '+f'f=open({info["archive"]!r},"rb"); shutil.copyfileobj(f,sys.stdout.buffer,1048576)'
        with part.open('wb') as stream: subprocess.run(command(code), stdout=stream, check=True)
        if part.stat().st_size != info['size'] or sha(part) != info['sha256']: raise RuntimeError('Output checksum/size mismatch')
        part.replace(archive)
    elif sha(archive) != info['sha256']: raise RuntimeError('Local output archive changed')
    result = out/('result_'+info['sha256'][:12]); report = bridge.extract_result(archive, result)
    write(out/'latest_result.json', dict(result=str(result), archive_sha256=info['sha256'], remote_exit_code=process.returncode))
    print(json.dumps(report, indent=2), flush=True); print('Clearance artifacts: '+str(result), flush=True)
    return process.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--name', required=True)
    parser.add_argument('--source', default='rollout_audit_pair6_v2')
    parser.add_argument('--host', default='egorov@93.180.62.99'); parser.add_argument('--port', type=int, default=27434)
    parser.add_argument('--container', default='egorov_drivesuprim')
    parser.add_argument('--control', type=Path, default=Path.home()/'.ssh/drex-astra-transfer.sock')
    parser.add_argument('--remote-repo', default='/workspace/DriveSuprim')
    raise SystemExit(run(parser.parse_args()))
