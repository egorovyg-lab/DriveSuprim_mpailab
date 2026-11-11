"""Anonymous pair inputs for outcome-selected diagnostic cases, never a benchmark."""
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import overlay_data as legacy
import overlay_v2_data as projection
import real_data as base
from runtime import read

REPRESENTATIONS = ('centerline', 'footprint')
MODES = ('road', 'blank')
ORDERS = ('canonical', 'reversed')
CONFIG = dict(protocol='pair_diagnostic_v1', representations=REPRESENTATIONS,
              modes=MODES, orders=ORDERS, image_size=[768, 640], photo_size=[768, 432],
              images_per_request=3, max_image_pixels=2048*1024, max_length=8192,
              horizon_s=4, sampled_footprint_times_s=[.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.],
              footprint_edge_spacing_m=.2, max_subdivisions_per_edge=256,
              projection=projection.CONFIG)
NOTE = ('Outcome-selected diagnostic: five known repairs and one known lane-keeping '
        'regression, not representative validation. Expected outcomes are not model inputs. '
        'Ground footprints follow nominal vocabulary poses, not scorer-simulated rollouts; '
        'sampled outlines/corner traces are not an exact continuous swept volume. '
        'Flat z=0, conservative central lens domain; no terrain or occlusion reconstruction. '
        'Only current front photograph and ego motion, no future actors or route command. '
        'Preference probabilities are not calibrated risk. No training, new scoring or holdout.')


def vehicle():
    value = read(Path(__file__).with_name('vehicle_profile_astra.json'))
    d = value['dimensions_m']
    if (set(d) != {'width', 'front_length', 'rear_length', 'rear_axle_to_center'} or
            not np.isfinite(list(d.values())).all() or not .5 < d['width'] < 5 or
            not 0 < d['front_length'] < 10 or not 0 < d['rear_length'] < 5 or
            not np.isclose(d['rear_axle_to_center'], (d['front_length']-d['rear_length'])/2)):
        raise ValueError('Inconsistent scorer vehicle profile')
    return d


def corners(poses, dimensions):
    """FL, FR, RR, RL ground corners from a rear-axle SE2 pose."""
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
        raise ValueError('Expected finite rear-axle poses')
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    forward, left = np.column_stack([c, s]), np.column_stack([-s, c])
    f, r, w = dimensions['front_length'], dimensions['rear_length'], dimensions['width']/2
    return np.stack([poses[:, :2]+f*forward+w*left, poses[:, :2]+f*forward-w*left,
                     poses[:, :2]-r*forward-w*left, poses[:, :2]-r*forward+w*left], axis=1)


def projected_lines(xy, calib, size):
    dense = []
    for a, b in zip(xy, xy[1:]):
        count = min(CONFIG['max_subdivisions_per_edge'],
                    max(2, int(np.ceil(np.linalg.norm(b-a)/CONFIG['footprint_edge_spacing_m']))))
        dense.append(a+np.linspace(0, 1, count+1)[:, None]*(b-a))
    if not dense:
        return []
    points = np.concatenate(dense)
    uv, valid = projection.project(np.column_stack([points, np.zeros(len(points))]), calib)
    lines = []
    for i in range(len(points)-1):
        if valid[i] and valid[i+1]:
            segment = legacy.clipped_segment(uv[i], uv[i+1], *size)
            if segment is not None:
                lines.append(segment)
    return lines


def footprint_lines(poses, dimensions, calib, size):
    boxes = corners(np.vstack([np.zeros((1, 3)), poses]), dimensions)
    lines = []
    for j in range(4):
        lines.extend(projected_lines(boxes[:, j], calib, size))
    for index in range(5, 41, 5):
        lines.extend(projected_lines(np.vstack([boxes[index], boxes[index, :1]]), calib, size))
    return lines


def names_and_order(pair, kind):
    ids = sorted(range(2), key=lambda i: base.digest([base.SEED, pair['pair_id'], 'name', pair['vocab_ids'][i]]))
    names = {i: 'T'+chr(65+j) for j, i in enumerate(ids)}
    ordering = sorted(range(2), key=lambda i: base.digest([base.SEED, pair['pair_id'], 'order', pair['vocab_ids'][i]]))
    if kind == 'reversed':
        ordering = ordering[::-1]
    elif kind != 'canonical':
        raise ValueError('Unknown pair ordering')
    return names, ordering


def inset(trajectories, pid, dimensions, representation):
    boxes = [corners(np.vstack([np.zeros((1, 3)), p]), dimensions) for p in trajectories]
    # Identical scale in both representations, including vehicle extents.
    xy = np.concatenate([b.reshape(-1, 2) for b in boxes])
    lo, hi = xy.min(axis=0)-1, xy.max(axis=0)+1
    scale = min(135/(hi[0]-lo[0]), 205/(hi[1]-lo[1]))
    middle = (hi+lo)/2
    pixel = lambda p: (112-(p[1]-middle[1])*scale, 76-(p[0]-middle[0])*scale)
    im = Image.new('RGB', (224, 160), 'white'); draw = ImageDraw.Draw(im)
    poses = np.vstack([np.zeros((1, 3)), trajectories[pid]])
    draw.line([pixel(p) for p in poses], fill='#0066dd', width=2)
    if representation == 'footprint':
        for j in range(4):
            draw.line([pixel(p) for p in boxes[pid][:, j]], fill='#0066dd', width=1)
        for index in range(5, 41, 5):
            box = [pixel(p) for p in boxes[pid][index]]
            draw.line(box+[box[0]], fill='#0066dd', width=1)
    for index in (10, 20, 30, 40):
        x, y = pixel(poses[index]); draw.ellipse((x-2, y-2, x+2, y+2), fill='#cc5500')
    x, y = pixel(poses[0]); draw.ellipse((x-2, y-2, x+2, y+2), fill='black')
    draw.rectangle((0, 0, 223, 159), outline='#888888')
    return im


def payload(package, scene, manifest, sample, pair, representation, mode, ordering_kind):
    if representation not in REPRESENTATIONS or mode not in MODES:
        raise ValueError('Unknown diagnostic condition')
    indices = [sample['pool_local'].index(i) for i in pair['local_ids']]
    if ([sample['pool_vocab'][i] for i in indices] != pair['vocab_ids'] or len(set(indices)) != 2):
        raise ValueError('Diagnostic pair candidate alignment changed')
    trajectories = sample['trajectories'][indices]
    dimensions = vehicle()
    names, ordering = names_and_order(pair, ordering_kind)
    path = base.camera_paths(package, scene, manifest)[4]
    with Image.open(path) as f:
        photo = f.convert('RGB')
    calib = projection.calibration(scene, photo.size)
    images = []
    for pid in [None, *ordering]:
        original = photo.copy() if mode == 'road' else Image.new('RGB', photo.size, 'white')
        draw = ImageDraw.Draw(original)
        if pid is not None:
            lines = projection.path_segments(trajectories[pid], calib, photo.size)
            if representation == 'footprint':
                lines += footprint_lines(trajectories[pid], dimensions, calib, photo.size)
            for line in lines:
                draw.line(line, fill='#00bfff', width=max(2, round(photo.width/450)))
        canvas = Image.new('RGB', tuple(CONFIG['image_size']), 'white')
        canvas.paste(ImageOps.pad(original, tuple(CONFIG['photo_size']), color='white'), (0, 32))
        draw = ImageDraw.Draw(canvas)
        title = 'Current front view' if pid is None else f'{names[pid]} | four-second candidate | {representation}'
        draw.text((8, 6), title, fill='black', font=base.font(17))
        if pid is not None:
            canvas.paste(inset(trajectories, pid, dimensions, representation), (8, 472))
            endpoint = trajectories[pid, -1]
            length = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), trajectories[pid, :, :2]]), axis=0), axis=1).sum()
            for j, text in enumerate([
                'Full path: up=forward, left=left; black=start',
                'Orange dots: 1, 2, 3, 4 seconds; common metric scale',
                f'At 4s: forward {endpoint[0]:.2f} m, left {endpoint[1]:.2f} m',
                f'Path length {length:.2f} m; vehicle width {dimensions["width"]:.3f} m',
                'Flat ground; hidden or inadmissible projected parts omitted',
            ]):
                draw.text((240, 482+27*j), text, fill='black', font=base.font(13))
        images.append(canvas)
    row = dict(state=dict(task='Choose the better four-second ego trajectory from two candidates.',
        representation=representation, vehicle_ground_dimensions_m=dimensions,
        path_coordinates='Poses are relative to the current rear axle: x=forward, y=left, metres.',
        image_1='Current front reference. Road photographs may be blank in the matched control.',
        candidate_images='Images 2 and 3 each show one anonymous candidate. Blue lines are flat-ground '
            'projections. The footprint representation adds sampled ground rectangles and corner traces '
            'of the vehicle body along the proposed path. These are nominal proposals, not predicted '
            'traffic or exact swept volumes. Lens-domain rejection, viewpoint and occlusion may hide '
            'relevant parts. The full-path inset below the photograph uses the same metric scale '
            'for both options and shows dots at 1,2,3,4 seconds. Missing pixels do not imply stopping.',
        velocity_xy_mps=sample['ego_velocity'], acceleration_xy_mps2=sample['ego_acceleration']),
        question=dict(type='choice', criteria={names[pid]: f'Candidate {names[pid]} in image {j+2}.'
                                              for j, pid in enumerate(ordering)},
            instructions='Choose the better available trajectory. Prioritize remaining in the drivable '
            'road area, avoiding collisions, obeying traffic rules and keeping the lane; then useful '
            'progress and comfort. Consider the full vehicle extents, not just its reference point.'), images=images)
    hashes = []
    for image in images:
        b = io.BytesIO(); image.save(b, format='PNG'); hashes.append(hashlib.sha256(b.getvalue()).hexdigest())
    query_hash = base.digest(dict(row={k: v for k, v in row.items() if k != 'images'},
                                 image_sha256=hashes, pair_positions=ordering))
    return row, ordering, query_hash
