"""Portable train-only camera/bundle snapshot and resumable SSH transfer.

Export runs in the existing Astra container. Pull needs only Python's standard
library and an already authenticated SSH ControlMaster on the destination.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import shlex
import shutil
import subprocess
import tarfile
import time

CAMERAS = {"cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"}
BLOCK = 4*1024**2


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path, limit=None):
    h = hashlib.sha256()
    with path.open("rb") as f:
        left = path.stat().st_size if limit is None else limit
        while left:
            b = f.read(min(BLOCK, left))
            if not b:
                raise RuntimeError("Unexpected EOF: " + str(path))
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def checked(root, relative):
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or '..' in p.parts or '\\' in relative or ':' in relative:
        raise ValueError("Unsafe relative path: " + relative)
    root = root.resolve()
    target = root.joinpath(*p.parts)
    if target.resolve() == root or root not in target.resolve().parents:
        raise ValueError("Path escaped package: " + relative)
    return target


def freeze(path, value):
    if path.exists() and read(path) != value:
        raise RuntimeError("Snapshot inputs changed; choose a new --out: " + str(path))
    if not path.exists():
        write(path, value)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return jsonable(value.tolist())
    return value


def select_records(identity, index):
    found = {}
    # Use existing frozen metadata; no new selection or query outcomes.
    for name, expected in sorted(identity['inputs'].items()):
        path = Path(name)
        if path.name not in ('selection.json', 'splits.json'):
            continue
        if sha(path) != expected:
            raise RuntimeError("Frozen selection changed: " + name)
        value = read(path)
        for r in value.get('records', []) + value.get('legacy_training_records', []):
            token = r['token']
            if token not in index:
                continue
            meta = index[token]['metadata']
            if r['session_id'] != meta['session_id'] or meta['role'] != 'train':
                raise ValueError("Train scene/session mismatch")
            if token not in found or r['log_name'] < found[token]['log_name']:
                found[token] = r
    if set(found) != set(index):
        raise ValueError("Cannot locate frozen log records for all bundled scenes")
    return found


def export(bundle, logs, sensors, out):
    manifest, identity, index = [read(bundle/n) for n in ('bundle_manifest.json', 'identity.json', 'examples_index.json')]
    for n, k in (('identity.json','identity_sha256'), ('examples_index.json','index_sha256'), ('vocab.npz','vocab_sha256')):
        if sha(bundle/n) != manifest[k]:
            raise RuntimeError("Frozen bundle checksum mismatch: " + n)
    if len(index) != identity['counts']['scenes']:
        raise ValueError("Bundle counts mismatch")
    sessions = {e['metadata']['session_id'] for e in index.values()}
    if sessions & set(identity['protected_sessions']) or any(e['metadata']['role'] != 'train' for e in index.values()):
        raise ValueError("Only unprotected training scenes may be exported")
    if len(sessions) != identity['counts']['sessions']:
        raise ValueError("Bundle session count mismatch")
    freeze(out/'source_identity.json', dict(protocol='portable_train_v1', bundle=str(bundle),
        bundle_manifest_sha256=sha(bundle/'bundle_manifest.json'), logs=str(logs), sensors=str(sensors)))
    records = select_records(identity, index)
    sources = {}
    def add(relative, source, expected=None):
        source = source.resolve()
        measured = sha(source)
        if expected and measured != expected:
            raise RuntimeError("Source checksum mismatch: " + str(source))
        entry = dict(size=source.stat().st_size, sha256=measured)
        if relative in sources and sources[relative][1] != entry:
            raise RuntimeError("Conflicting source path")
        sources[relative] = (source, entry)
    for n in ('bundle_manifest.json','identity.json','examples_index.json','vocab.npz'):
        add('bundle/'+n, bundle/n)
    for token, entry in index.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', token) or entry['metadata']['token'] != token:
            raise ValueError("Unsafe/mismatched token")
        add('bundle/examples/'+token+'.npz', bundle/'examples'/(token+'.npz'), entry['sha256'])
    scene_inputs, log_hashes = [], {}
    for log in sorted({r['log_name'] for r in records.values()}):
        path = checked(logs, log+'.pkl')
        log_hashes[log] = sha(path)
        # These are the user's already installed, trusted OpenScene pickles.
        with path.open('rb') as f:
            frames = pickle.load(f)
        positions = {f['token']: i for i, f in enumerate(frames)}
        if len(positions) != len(frames):
            raise ValueError("Duplicate tokens within log")
        for token, r in sorted(records.items()):
            if r['log_name'] != log:
                continue
            current = positions[token]
            if current < 3:
                raise ValueError("Insufficient sensor history")
            history = []
            for frame in frames[current-3:current+1]:
                cams = {k.lower(): v for k, v in frame['cams'].items() if k.lower() in CAMERAS}
                if set(cams) != CAMERAS:
                    raise ValueError("Expected eight cameras")
                paths = {}
                for k, camera in sorted(cams.items()):
                    relative = str(PurePosixPath(camera['data_path']))
                    if PurePosixPath(relative).suffix.lower() not in ('.jpg','.jpeg','.png'):
                        raise ValueError("Expected camera image")
                    add('images/'+relative, checked(sensors, relative))
                    paths[k] = 'images/'+relative
                history.append(dict(frame_token=frame['token'], cameras=paths))
            calibrations = {k.lower(): jsonable({key: v for key,v in camera.items() if key != 'data_path'})
                            for k,camera in frames[current]['cams'].items() if k.lower() in CAMERAS}
            scene_inputs.append(dict(metadata=index[token]['metadata'], log_name=log,
                                     history=history, current_camera_calibrations=calibrations))
        print('EXPORT METADATA', log, flush=True)
    scene_inputs.sort(key=lambda r:r['metadata']['token'])
    freeze(out/'scene_inputs.json', scene_inputs)
    freeze(out/'source_logs.json', log_hashes)
    for n in ('source_identity.json','source_logs.json','scene_inputs.json'):
        add(n, out/n)
    files = {name: entry for name, (_,entry) in sorted(sources.items())}
    package = dict(protocol='portable_train_v1', counts=identity['counts'], files=files,
        images=sum(n.startswith('images/') for n in files), total_bytes=sum(e['size'] for e in files.values()),
        holdout_scenes=0, note='Train-only portable snapshot. Original bundle provenance retained verbatim; '
        'its absolute upstream paths are not relocated. Use package checksums on destination, not the old load_bundle(). '
        'Labels remain separate NPZ fields and must never be forwarded as Drex input.')
    freeze(out/'package_manifest.json', package)
    if shutil.disk_usage(out).free < sum(entry['size'] for name,(_,entry) in sources.items()
        if not checked(out,name).exists()) + package['total_bytes'] + 1024**3:
        raise RuntimeError('Not enough space for snapshot + uncompressed archive + 1 GiB reserve')
    for i,(name,(source,entry)) in enumerate(sorted(sources.items()),1):
        target = checked(out,name)
        if target.exists() and sha(target) == entry['sha256']:
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_name(target.name+'.tmp')
        shutil.copyfile(source,tmp)
        if sha(tmp) != entry['sha256']:
            raise RuntimeError('Source changed while copying: '+str(source))
        tmp.replace(target)
        if i%1000==0: print(f'PACKAGE {i}/{len(sources)}',flush=True)
    verify(out)
    archive = out.with_suffix('.tar')
    info_path = Path(str(archive)+'.json')
    if info_path.exists():
        info = read(info_path)
        if info['manifest_sha256'] != sha(out/'package_manifest.json') or sha(archive) != info['sha256']:
            raise RuntimeError('Committed archive changed')
    else:
        tmp = Path(str(archive)+'.tmp')
        with tarfile.open(tmp,'w') as tar:
            for name in ['package_manifest.json']+list(files):
                tar.add(checked(out,name),arcname=name,recursive=False)
        tmp.replace(archive)
        info = dict(protocol='portable_train_v1',archive=str(archive),size=archive.stat().st_size,
                    sha256=sha(archive),manifest_sha256=sha(out/'package_manifest.json'),counts=identity['counts'])
        write(info_path,info)
    print(json.dumps(info,indent=2),flush=True)
    return info


def verify(out):
    package = read(out/'package_manifest.json')
    if package['protocol'] != 'portable_train_v1' or package['holdout_scenes'] != 0:
        raise ValueError('Wrong package protocol')
    for i,(name,entry) in enumerate(package['files'].items(),1):
        path=checked(out,name)
        if path.stat().st_size != entry['size'] or sha(path) != entry['sha256']:
            raise RuntimeError('Package verification failed: '+name)
        if i%2000==0: print(f'VERIFY {i}/{len(package["files"])}',flush=True)
    result=dict(status='verified',**package['counts'],images=package['images'],bytes=package['total_bytes'],holdout_scenes=0)
    print(json.dumps(result),flush=True)
    return result


def extract(archive,out,expected_manifest):
    with tarfile.open(archive,'r:') as tar:
        members=tar.getmembers()
        if not members or members[0].name != 'package_manifest.json' or not members[0].isfile():
            raise ValueError('Manifest must be first regular archive member')
        raw=tar.extractfile(members[0]).read()
        if hashlib.sha256(raw).hexdigest()!=expected_manifest:
            raise RuntimeError('Archive manifest checksum mismatch')
        manifest=json.loads(raw)
        expected={'package_manifest.json':dict(size=len(raw),sha256=expected_manifest),**manifest['files']}
        if len(members)!=len(expected) or {m.name for m in members}!=set(expected):
            raise ValueError('Unexpected or duplicate archive members')
        # Validate every path/type before writing even the manifest.
        for member in members:
            checked(out,member.name)
            entry=expected[member.name]
            if not member.isfile() or member.size!=entry['size']:
                raise ValueError('Nonregular/mismatched archive member')
        for member in members:
            target=checked(out,member.name)
            entry=expected[member.name]
            if target.exists():
                if sha(target)!=entry['sha256']:
                    raise RuntimeError('Destination file changed: '+str(target))
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_name(target.name+'.tmp')
            with tar.extractfile(member) as source,tmp.open('wb') as sink:
                shutil.copyfileobj(source,sink,BLOCK)
            if sha(tmp)!=entry['sha256']:
                raise RuntimeError('Extracted file checksum mismatch')
            tmp.replace(target)
    return verify(out)


def remote_command(host,port,control,container,code):
    # BatchMode forbids password prompts in background; credentials stay in ssh.
    remote=shlex.join(['docker','exec','-i',container,'python','-c',code])
    return ['ssh','-p',str(port),'-S',str(control),'-o','BatchMode=yes',
            '-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6',host,remote]


def prepare_remote(host,port,control,container):
    source=Path(__file__).read_bytes()
    tool='/workspace/exp_v2/drex_transfer_tool.py'
    code=('import pathlib,hashlib,sys; b=sys.stdin.buffer.read(); '
          'assert hashlib.sha256(b).hexdigest()=='+repr(hashlib.sha256(source).hexdigest())+'; '
          'p=pathlib.Path('+repr(tool)+'); p.parent.mkdir(parents=True,exist_ok=True); '
          't=p.with_suffix(".py.tmp"); t.write_bytes(b); t.replace(p)')
    subprocess.run(remote_command(host,port,control,container,code),input=source,check=True)
    code='import runpy,sys; sys.argv=['+repr(tool)+',"export"]; runpy.run_path('+repr(tool)+',run_name="__main__")'
    subprocess.run(remote_command(host,port,control,container,code),check=True)


def pull(host,port,control,container,source,dest,max_gb):
    dest.mkdir(parents=True,exist_ok=True)
    cmd=lambda code:remote_command(host,port,control,container,code)
    code='import pathlib; print(pathlib.Path('+repr(str(source)+'.json')+').read_text())'
    info=json.loads(subprocess.check_output(cmd(code),text=True))
    if info['protocol']!='portable_train_v1' or info['archive']!=str(source) or not 0<info['size']<=max_gb*1e9:
        raise ValueError('Unexpected archive identity or transfer budget exceeded')
    freeze(dest/'transfer_identity.json',info)
    target=dest/'training.tar'
    part=dest/'training.tar.part'
    if not target.exists():
        offset=part.stat().st_size if part.exists() else 0
        if offset>info['size']:
            raise RuntimeError('Partial file exceeds source size')
        if shutil.disk_usage(dest).free < (info['size']-offset)+info['size']+1024**3:
            raise RuntimeError('Not enough space for remaining archive and extracted package')
        if offset:
            code=('import hashlib; f=open('+repr(str(source))+',"rb"); h=hashlib.sha256(); left='+str(offset)+'\n'
                  'while left:\n b=f.read(min(4194304,left)); assert b; h.update(b); left-=len(b)\nprint(h.hexdigest())')
            expected=subprocess.check_output(cmd(code),text=True).strip()
            if sha(part)!=expected:
                raise RuntimeError('Partial archive prefix checksum mismatch; preserved for inspection')
        code=('import shutil,sys; f=open('+repr(str(source))+',"rb"); f.seek('+str(offset)+'); '
              'shutil.copyfileobj(f,sys.stdout.buffer,4194304)')
        print(f'TRANSFER offset={offset} total={info["size"]}',flush=True)
        process=subprocess.Popen(cmd(code),stdout=subprocess.PIPE)
        tick=time.monotonic()
        try:
            with part.open('ab') as f:
                while True:
                    block=process.stdout.read(BLOCK)
                    if not block: break
                    if f.tell()+len(block)>info['size']:
                        raise RuntimeError('Received more bytes than declared')
                    f.write(block)
                    if time.monotonic()-tick>=30:
                        f.flush()
                        print(f'TRANSFER {f.tell()}/{info["size"]}',flush=True)
                        tick=time.monotonic()
            if process.wait()!=0:
                raise RuntimeError('SSH transfer interrupted; partial archive preserved')
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
        if part.stat().st_size!=info['size'] or sha(part)!=info['sha256']:
            raise RuntimeError('Downloaded archive checksum/size mismatch')
        part.replace(target)
    elif target.stat().st_size!=info['size'] or sha(target)!=info['sha256']:
        raise RuntimeError('Completed local archive changed')
    result=extract(target,dest/'package',info['manifest_sha256'])
    write(dest/'transfer_report.json',dict(result,archive_sha256=info['sha256']))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='stage',required=True)
    base=Path('/workspace/exp_v2/drivemem_memory')
    e=sub.add_parser('export')
    e.add_argument('--bundle',type=Path,default=base/'selector_training_bundle_v3_30gb')
    e.add_argument('--logs',type=Path,default=Path('/workspace/exp_v2/drivemem_data/navsim_logs/trainval'))
    e.add_argument('--sensors',type=Path,default=Path('/workspace/exp_v2/drivemem_data/sensor_blobs/trainval'))
    e.add_argument('--out',type=Path,default=Path('/workspace/exp_v2/drex_transfer/navsim_train_v1'))
    v=sub.add_parser('verify'); v.add_argument('out',type=Path)
    t=sub.add_parser('pull')
    r=sub.add_parser('prepare-remote')
    for parser in (t,r):
        parser.add_argument('--host',default='egorov@93.180.62.99')
        parser.add_argument('--port',type=int,default=27434)
        parser.add_argument('--container',default='egorov_drivesuprim')
        parser.add_argument('--control',type=Path,required=True)
    t.add_argument('--source',type=Path,default=Path('/workspace/exp_v2/drex_transfer/navsim_train_v1.tar'))
    t.add_argument('--dest',type=Path,required=True)
    t.add_argument('--max-gb',type=float,default=30)
    a=p.parse_args()
    if a.stage=='export':
        export(*[x.expanduser().resolve() for x in (a.bundle,a.logs,a.sensors,a.out)])
    elif a.stage=='verify': verify(a.out.expanduser().resolve())
    elif a.stage=='prepare-remote': prepare_remote(a.host,a.port,a.control.expanduser(),a.container)
    else: pull(a.host,a.port,a.control.expanduser(),a.container,a.source,a.dest.expanduser().resolve(),a.max_gb)


if __name__=='__main__': main()
