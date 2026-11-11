"""Camera availability audit of the user's trusted, copied OpenScene log pickles."""
import argparse
import bisect
from collections import Counter
import csv
import json
from pathlib import Path
import pickle
import re

from PIL import Image

import real_data as base
from runtime import write
from transfer import checked, sha


def sid(log):
    match = re.fullmatch(r'(.+_veh-[^_]+)_\d+_\d+',log)
    if not match: raise ValueError('Unrecognized session/log name: '+log)
    return match.group(1)


def coverage(frames, i, existing):
    def names(history, cameras):
        result = []
        for frame in history:
            cams = {k.lower():v for k,v in frame['cams'].items()}
            if not set(cameras)<=set(cams): return None
            for c in cameras:
                name = cams[c]['data_path']
                # Reject malformed metadata rather than escaping the copied root.
                if not isinstance(name,str) or name.startswith('/') or '..' in name.split('/') or '\\' in name:
                    raise ValueError('Unsafe sensor path')
                result.append(name)
        return result
    six = names(frames[i-1:i+1],base.CONFIG['cameras'])
    full = names(frames[i-3:i+1],('cam_l0','cam_f0','cam_r0','cam_l1','cam_l2','cam_r1','cam_r2','cam_b0'))
    return six,full,bool(six and all(n in existing for n in six)),bool(full and all(n in existing for n in full))


def inventory(raw,package,out):
    manifest,index,_,_ = base.package_metadata(package)
    identity = json.loads((package/'bundle/identity.json').read_text())
    train = {e['metadata']['session_id'] for e in index.values()}
    protected = set(identity['protected_sessions'])
    sensors = raw/'sensor_blobs/trainval'; logs = raw/'navsim_logs/trainval'
    files = sorted(logs.glob('*.pkl'))
    if not files or not sensors.is_dir(): raise FileNotFoundError('Copied trainval logs/cameras missing')
    existing = {p.relative_to(sensors).as_posix() for p in sensors.rglob('*')
                if p.is_file() and '.rsync-partial' not in p.parts and p.suffix.lower() in ('.jpg','.jpeg','.png')}
    rows = {}; seen_images = {}; log_hashes = {}; duplicates = 0
    for n,path in enumerate(files,1):
        session = sid(path.stem); log_hashes[path.name] = sha(path)
        # Only use the user's own trusted OpenScene metadata, never arbitrary downloaded pickles.
        with path.open('rb') as f: frames = pickle.load(f)
        if not isinstance(frames,list): raise ValueError('Expected a list of log frames')
        for i in range(3,len(frames)-10):  # Four history frames and ten future frames, like the scene filter.
            frame = frames[i]
            if not frame.get('roadblock_ids',[]): continue
            six,full,has_six,has_full = coverage(frames,i,existing)
            usable = has_six or has_full
            if usable:
                for name in set((six if has_six else [])+(full if has_full else [])):
                    if name not in seen_images:
                        try:
                            with Image.open(checked(sensors,name)) as image: image.verify()
                            seen_images[name] = True
                        except Exception:
                            seen_images[name] = False
            valid_six = has_six and all(seen_images[n] for n in six)
            valid_full = has_full and all(seen_images[n] for n in full)
            t = int(frame['timestamp']); token = str(frame['token'])
            row = dict(token=token,log_name=path.stem,session_id=session,timestamp_us=t,
                split_status='existing_training' if session in train else 'protected' if session in protected else 'unassigned',
                already_bundled=token in index,camera_3x2_valid=valid_six,camera_8x4_valid=valid_full)
            key = (session,t)
            if key in rows:
                duplicates += 1
                # Union coverage when overlapping segments describe the same scene.
                old = rows[key]
                row['camera_3x2_valid'] |= old['camera_3x2_valid']; row['camera_8x4_valid'] |= old['camera_8x4_valid']
                row['already_bundled'] |= old['already_bundled']
            rows[key] = row
        if n%50==0 or n==len(files): print(f'RAW INVENTORY {n}/{len(files)}',flush=True)
    values = sorted(rows.values(),key=lambda r:(r['session_id'],r['timestamp_us']))
    out.mkdir(parents=True,exist_ok=True)
    if values:
        with (out/'camera_windows.csv.tmp').open('w',encoding='utf-8',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
        (out/'camera_windows.csv.tmp').replace(out/'camera_windows.csv')
    groups = {}
    for status in ('existing_training','protected','unassigned'):
        group = [r for r in values if r['split_status']==status]
        spaced = 0
        for session in sorted({r['session_id'] for r in group}):
            local = [r for r in group if r['session_id']==session]
            anchors = sorted({r['timestamp_us'] for r in local if r['already_bundled']})
            for r in local:
                if r['already_bundled'] or not r['camera_8x4_valid']: continue
                t = r['timestamp_us']; j = bisect.bisect_left(anchors,t)
                if (j and t-anchors[j-1]<10_000_000) or (j<len(anchors) and anchors[j]-t<10_000_000): continue
                bisect.insort(anchors,t); spaced += 1
        groups[status] = dict(metadata_windows=len(group),
            valid_3x2_windows=sum(r['camera_3x2_valid'] for r in group),
            valid_8x4_windows=sum(r['camera_8x4_valid'] for r in group),
            additional_8x4_windows=sum(r['camera_8x4_valid'] and not r['already_bundled'] for r in group),
            additional_8x4_spaced_10s=spaced,
            full_camera_sessions=len({r['session_id'] for r in group if r['camera_8x4_valid']}))
    result = dict(status='availability_audited',log_files=len(files),image_files=len(existing),
        metadata_windows=len(values),duplicate_session_times=duplicates,groups=groups,
        invalid_images=[n for n,ok in seen_images.items() if not ok],
        full_camera_windows_by_session=dict(Counter(r['session_id'] for r in values if r['camera_8x4_valid'])),
        package_manifest_sha256=sha(package/'package_manifest.json'),log_sha256=log_hashes,
        note='Availability inventory, not a new training split or official NAVSIM eligibility check. '
             'Unassigned sessions may include historically inspected development; do not auto-train on them. '
             'Overlapping camera windows are correlated. New DS features, metric caches and labels are not created. '
             'No GPU, inference, scored outcomes or image downloads. Existing protected sessions remain protected.')
    write(out/'report.json',result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True); p.add_argument('--package',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--transfer-log',type=Path,required=True)
    a = p.parse_args()
    lines = a.transfer_log.read_text().strip().splitlines()
    if not lines or not lines[-1].startswith('FINISHED mode=run destination='):
        raise RuntimeError('Raw transfer has not reported successful completion; inventory not started')
    print(json.dumps(inventory(a.raw,a.package,a.out),indent=2),flush=True)


if __name__=='__main__': main()
