"""Read-only, input-only adapter for the portable NAVSIM train snapshot."""
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

from runtime import read, write
from transfer import checked, sha, verify

SEED = 20261111
METRICS = ('no_at_fault_collisions', 'drivable_area_compliance', 'driving_direction_compliance',
           'traffic_light_compliance', 'ego_progress', 'time_to_collision_within_bound',
           'lane_keeping', 'history_comfort')
SAFETY = ('no_at_fault_collisions', 'drivable_area_compliance', 'driving_direction_compliance',
          'traffic_light_compliance', 'time_to_collision_within_bound')
INPUT_KEYS = ('candidate_ids', 'selected_local', 'selection_scores', 'ego_status')
TASKS = (('camera', 'canonical'), ('camera', 'reversed'), ('camera', 'permuted'), ('blank', 'canonical'))
CONFIG = dict(protocol='real_train64_v1', seed=SEED, scenes_per_session=2, sessions=32,
              tasks=TASKS, primary='drex_mean3', horizon_seconds=4., poses=40,
              max_image_pixels=2048*1024, max_length=8192, batch_size=1, precision='bf16',
              cameras=['cam_l0', 'cam_f0', 'cam_r0'], history_frames_used=2,
              bootstrap_reps=2000, useful_gain_min=.005)
NOTE = ('Exploratory replay on archive-selected, previously labelled TRAIN scenes, not independent '
        'validation. No fitting, new scoring, future images or holdout. Stored raw local PDM, not '
        'full EPDMS. Relative component degradation is not absolute safety. Route command semantics '
        'are not decoded or supplied; images and ego motion are a restricted input interface. '
        'Trajectory plots are not maps or projections onto cameras. Probabilities are not calibrated risk.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def freeze(path, value):
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists() and read(path) != value:
        raise RuntimeError('Frozen real-pilot input changed; choose a new DREX_REAL_NAME: ' + str(path))
    if not path.exists():
        write(path, value)


def choose_records(index, sessions=32, per_session=2):
    groups = defaultdict(list)
    for token, entry in index.items():
        m = entry['metadata']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', token) or m['token'] != token or m['role'] != 'train':
            raise ValueError('Only train records with matching tokens allowed')
        groups[m['session_id']].append(dict(m))
    if len(groups) != sessions or any(len(g) < per_session for g in groups.values()):
        raise ValueError('Need the declared sessions and two scenes in each; no silent exclusions')
    key = lambda value: hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()
    sids = sorted(groups, key=key)
    ranked = {s: sorted(groups[s], key=lambda r: key(r['token'])) for s in sids}
    # Pilot begins with different sessions; full cohort is balanced by session.
    return [ranked[s][i] for i in range(per_session) for s in sids]


def package_metadata(package):
    manifest = read(package/'package_manifest.json')
    if manifest['protocol'] != 'portable_train_v1' or manifest['holdout_scenes'] != 0:
        raise ValueError('Expected verified portable train-only snapshot')
    for name, entry in manifest['files'].items():
        checked(package, name)
        if entry['size'] < 0 or len(entry['sha256']) != 64:
            raise ValueError('Invalid file commitment')
    def committed(name):
        path = checked(package, name)
        if sha(path) != manifest['files'][name]['sha256']:
            raise RuntimeError('Portable package file changed: ' + name)
        return path
    index = read(committed('bundle/examples_index.json'))
    identity = read(committed('bundle/identity.json'))
    bundle_manifest = read(committed('bundle/bundle_manifest.json'))
    if any(sha(committed('bundle/'+name)) != bundle_manifest[key] for name, key in (
        ('identity.json', 'identity_sha256'), ('examples_index.json', 'index_sha256'), ('vocab.npz', 'vocab_sha256'))):
        raise RuntimeError('Original bundle manifest mismatch')
    if (len(index) != identity['counts']['scenes'] or identity['counts'] != manifest['counts']
            or bundle_manifest['counts'] != identity['counts']):
        raise ValueError('Bundle scene counts mismatch')
    sids = {e['metadata']['session_id'] for e in index.values()}
    if len(sids) != identity['counts']['sessions'] or sids & set(identity['protected_sessions']):
        raise ValueError('Bundle session/protected split mismatch')
    if bundle_manifest['label_keys'] != ['pdm_score', *METRICS]:
        raise ValueError('Unrecognized label column order')
    for token, entry in index.items():
        path = 'bundle/examples/'+token+'.npz'
        checked(package,path)
        if manifest['files'][path]['sha256'] != entry['sha256']:
            raise RuntimeError('Portable example differs from original frozen bundle')
    scene_rows = read(committed('scene_inputs.json'))
    scenes = {r['metadata']['token']: r for r in scene_rows}
    if len(scenes) != len(scene_rows) or set(scenes) != set(index):
        raise ValueError('Camera scene membership mismatch')
    for token in scenes:
        if scenes[token]['metadata'] != index[token]['metadata']:
            raise ValueError('Camera metadata differs from bundle')
    with np.load(committed('bundle/vocab.npz'), allow_pickle=False) as f:
        vocab = f['vocab'].copy()
    if vocab.ndim != 3 or vocab.shape[1:] != (40, 3) or not np.isfinite(vocab).all():
        raise ValueError('Expected absolute local [x-forward, y-left, heading] poses: [N,40,3], 4s/0.1s')
    return manifest, index, scenes, vocab


def inputs_from_arrays(arrays, vocab):
    # Only this allowlist is deserialized before the decision freeze.
    sample = {k: np.asarray(arrays[k]).copy() for k in INPUT_KEYS}
    ids, scores, ego = sample['candidate_ids'], sample['selection_scores'], sample['ego_status']
    b = int(sample['selected_local'])
    if (ids.shape != (256,) or ids.dtype.kind not in 'iu' or len(set(ids.tolist())) != 256
            or ids.min() < 0 or ids.max() >= len(vocab) or scores.shape != (256,)
            or not np.isfinite(scores).all() or b != int(np.argmax(scores))):
        raise ValueError('Invalid frozen teacher candidate alignment')
    if ego.ndim != 1 or len(ego) < 8 or len(ego) % 8 or not np.isfinite(ego).all():
        raise ValueError('Expected current-first status blocks [command(4), velocity(2), acceleration(2)]')
    local = np.argsort(-scores, kind='stable')[:16]
    if b not in local:
        raise ValueError('Baseline absent from top16')
    return dict(pool_local=local.tolist(), pool_vocab=ids[local].tolist(), baseline_local=b,
                trajectories=vocab[ids[local]].astype(np.float64),
                ego_velocity=ego[4:6].tolist(), ego_acceleration=ego[6:8].tolist(),
                status_blocks=len(ego)//8)


def load_inputs(package, token, manifest, vocab):
    relative = 'bundle/examples/'+token+'.npz'
    path = checked(package, relative)
    if sha(path) != manifest['files'][relative]['sha256']:
        raise RuntimeError('Frozen example changed: '+token)
    with np.load(path, allow_pickle=False) as arrays:
        return inputs_from_arrays(arrays, vocab)


def camera_paths(package, scene, manifest):
    history = scene['history']
    if (len(history) != 4 or len({f['frame_token'] for f in history}) != 4
            or history[-1]['frame_token'] != scene['metadata']['token']):
        raise ValueError('Camera chronology/current token mismatch')
    paths = []
    for frame in history[-2:]:
        for camera in CONFIG['cameras']:
            name = frame['cameras'][camera]
            path = checked(package, name)
            if not name.startswith('images/') or name not in manifest['files']:
                raise ValueError('Camera image outside committed snapshot')
            paths.append(path)
    return paths


def permutations(token, kind):
    import random
    ids = list(range(16))
    rng = random.Random(int(digest([SEED, token, 'canonical'])[:16], 16))
    rng.shuffle(ids)
    if kind == 'reversed':
        ids.reverse()
    elif kind == 'permuted':
        # Guaranteed different from canonical, without outcome-dependent retries.
        ids = ids[5:]+ids[:5]
    elif kind != 'canonical':
        raise ValueError('Unknown option order')
    return ids


def font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def camera_board(paths, blank=False):
    from PIL import Image, ImageDraw, ImageOps
    board = Image.new('RGB', (1536, 640), 'white')
    draw = ImageDraw.Draw(board)
    for i, path in enumerate(paths):
        x, y = (i % 3)*512, (i//3)*320
        if not blank:
            with Image.open(path) as source:
                image = ImageOps.contain(source.convert('RGB'), (512, 288))
            board.paste(image, (x+(512-image.width)//2, y+28+(288-image.height)//2))
        title = ('Previous frame' if i < 3 else 'Current frame') + ': ' + ('front-left', 'front', 'front-right')[i%3]
        draw.text((x+8, y+4), title, fill='black', font=font(17))
    return board


def trajectory_board(trajectories, order):
    from PIL import Image, ImageDraw
    board = Image.new('RGB', (1024, 1024), 'white')
    draw = ImageDraw.Draw(board)
    xy_all = np.concatenate([np.zeros((1, 2)), trajectories[:, :, :2].reshape(-1, 2)])
    # Same metric scale in both axes and in every candidate panel.
    xmin, ymin = xy_all.min(axis=0)-2
    xmax, ymax = xy_all.max(axis=0)+2
    scale = min(190/max(xmax-xmin, 1), 210/max(ymax-ymin, 1))
    forward_mid, lateral_mid = (xmin+xmax)/2, (ymin+ymax)/2
    for j, pid in enumerate(order):
        ox, oy = (j%4)*256, (j//4)*256
        def pixel(x, y):
            return ox+128-(y-lateral_mid)*scale, oy+119-(x-forward_mid)*scale
        draw.rectangle((ox, oy, ox+255, oy+255), outline='#999999')
        draw.text((ox+8, oy+4), f'C{j:02d}  Up=forward, left=left', fill='black', font=font(13))
        poses = trajectories[pid]
        points = [pixel(0, 0)] + [pixel(x,y) for x,y,_ in poses]
        draw.line(points, fill='#0057a8', width=3)
        x0,y0 = pixel(0,0)
        draw.ellipse((x0-3,y0-3,x0+3,y0+3), fill='black')
        for t in range(1,5):
            px,py = pixel(*poses[t*10-1,:2])
            draw.ellipse((px-3,py-3,px+3,py+3), fill='#cf5a00')
            draw.text((px+4,py-5), str(t), fill='#8b3900', font=font(11))
        x,y,_ = poses[-1]
        draw.text((ox+8,oy+224), f'4s: forward {x:.1f} m, left {y:.1f} m', fill='black', font=font(12))
        draw.text((ox+8,oy+240), f'Common scale: {scale:.2f} pixels/metre', fill='#555555', font=font(11))
    return board


def request(inputs, order, photos, paths):
    return dict(state=dict(
        task='Select one proposed four-second ego trajectory using observed road photographs and ego motion.',
        image_1='Three camera views, previous frame above current frame. Camera images are not geometrically stitched.',
        image_2='Sixteen ego-local candidate path panels. This is NOT a road map or a camera projection. '
                'Origin is the current rear-axle pose; x is forward, y is left, metres. Blue paths have dots at 1,2,3,4 seconds. '
                'Identical metric scale across panels. Grey panel borders do not represent road boundaries.',
        current_velocity_xy_mps=inputs['ego_velocity'], current_acceleration_xy_mps2=inputs['ego_acceleration']),
        question=dict(type='choice', criteria={f'C{j:02d}':f'Trajectory in panel C{j:02d} of image 2.' for j in range(16)},
            instructions='Choose the best candidate: prioritize avoiding collisions, remaining on the drivable road, '
                         'obeying traffic lights and driving direction; then favour useful forward progress and comfort. '
                         'Infer only from the supplied observations. Some candidates may all be poor; choose the best available. '
                         'Do not assume that a plot panel is a road. No candidate is identified as the baseline.'),
        images=[photos, paths])


def payload(inputs, scene, package, manifest, mode, kind):
    order = permutations(scene['metadata']['token'], kind)
    photos = camera_board(camera_paths(package, scene, manifest), blank=mode=='blank')
    if mode not in ('camera', 'blank'):
        raise ValueError('Unknown observation mode')
    paths = trajectory_board(inputs['trajectories'], order)
    row = request(inputs, order, photos, paths)
    pixel_hashes = []
    for image in row['images']:
        buffer = io.BytesIO(); image.save(buffer, format='PNG')
        pixel_hashes.append(hashlib.sha256(buffer.getvalue()).hexdigest())
    query_hash = digest(dict(row={k:v for k,v in row.items() if k!='images'}, pixels=pixel_hashes, pool_positions=order))
    return row, order, query_hash


def prepare(package, root, out, sessions=32, per_session=2):
    verify(package)  # Hashes bytes, does not deserialize NPZ outcome arrays.
    manifest, index, scenes, vocab = package_metadata(package)
    records = choose_records(index, sessions, per_session)
    sources = {name:sha(Path(__file__).with_name(name)) for name in
               ('real_data.py','real_pilot.py','runtime.py','transfer.py','night.py')}
    identity = dict(config=dict(CONFIG,sessions=sessions,scenes_per_session=per_session), model_plan=read(root/'model_plan.json'),
        environment_lock_sha256=sha(root/'requirements.lock.txt'), sources=sources,
        package_manifest_sha256=sha(package/'package_manifest.json'), records=records)
    freeze(out/'identity.json', identity)
    inventory = []
    for r in records:
        s = load_inputs(package,r['token'],manifest,vocab)
        paths = camera_paths(package,scenes[r['token']],manifest)
        from PIL import Image
        sizes=[]
        for path in paths:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                sizes.append(list(im.size))
        inventory.append(dict(metadata=r, pool_local=s['pool_local'], pool_vocab=s['pool_vocab'],
                              baseline_local=s['baseline_local'], image_sizes=sizes, status_blocks=s['status_blocks']))
    freeze(out/'selection.json',inventory)
    return manifest, records, scenes, vocab
