"""Matched nominal/executed ego paths; no maps, events or outcomes in payloads."""
import hashlib
import io

import numpy as np
from PIL import Image, ImageDraw

import pair_diagnostic_data as original
import real_data as base

GEOMETRIES = ('nominal', 'simulated')
CONFIG = dict(original.CONFIG, protocol='rollout_choice_pair6_v1', geometries=GEOMETRIES,
              common_inset_scale='all four paths, including vehicle extents', requests_per_pair=16,
              primary='simulated/centerline/road')
NOTE = ('Matched representation diagnostic on six outcome-selected pairs, not validation. '
        'Nominal and simulator-executed ego paths use the same candidates, current photos, '
        'prompts, anonymous IDs, option orders and common inset scale. Simulator paths come '
        'from the 12-candidate metric-parity audit. No audit map, centerline, event markers, '
        'future traffic, expected outcomes or variant name is supplied to the model. '
        'Flat-ground camera projection does not reconstruct terrain or occlusion. '
        'No scorer change, fitting, new labels or holdout. Preferences are not calibrated risk.')


def validate_paths(paths):
    if set(paths) != set(GEOMETRIES):
        raise ValueError('Exactly nominal/simulated paths required')
    for values in paths.values():
        if values.shape != (2, 40, 3) or not np.isfinite(values).all():
            raise ValueError('Expected two finite 40-pose paths per geometry')


def tracking_error(reference, executed, dt=.1):
    """Time-aligned error resolved along/across the nominal rear-axle heading."""
    if reference.shape != (41, 3) or executed.shape != reference.shape or dt <= 0:
        raise ValueError('Expected time-aligned 41-pose paths')
    if not np.isfinite(reference).all() or not np.isfinite(executed).all():
        raise ValueError('Nonfinite tracking poses')
    delta = executed[:, :2]-reference[:, :2]
    c, s = np.cos(reference[:, 2]), np.sin(reference[:, 2])
    longitudinal = c*delta[:, 0]+s*delta[:, 1]
    lateral = -s*delta[:, 0]+c*delta[:, 1]
    heading = executed[:, 2]-reference[:, 2]
    heading = np.arctan2(np.sin(heading), np.cos(heading))
    speeds = [np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)/dt for p in (reference, executed)]
    return dict(max_position_error_m=float(np.linalg.norm(delta, axis=1).max()),
        max_abs_longitudinal_error_m=float(np.abs(longitudinal).max()),
        max_abs_lateral_error_m=float(np.abs(lateral).max()),
        endpoint_longitudinal_error_m=float(longitudinal[-1]),
        endpoint_lateral_error_m=float(lateral[-1]),
        max_abs_heading_error_deg=float(np.rad2deg(np.abs(heading)).max()),
        max_abs_segment_speed_difference_mps=float(np.abs(speeds[1]-speeds[0]).max()),
        note='Positive longitudinal means ahead; positive lateral means left. '
             'Segment speeds are finite differences, not the tracker fitted speed profile.')


def shared_inset(paths, geometry, member, representation):
    dimensions = original.vehicle()
    all_boxes = [original.corners(np.vstack([np.zeros((1, 3)), p]), dimensions)
                 for kind in GEOMETRIES for p in paths[kind]]
    xy = np.concatenate([b.reshape(-1, 2) for b in all_boxes])
    lo, hi = xy.min(axis=0)-1, xy.max(axis=0)+1
    scale = min(135/(hi[0]-lo[0]), 205/(hi[1]-lo[1])); middle = (hi+lo)/2
    pixel = lambda p: (112-(p[1]-middle[1])*scale, 76-(p[0]-middle[0])*scale)
    poses = np.vstack([np.zeros((1, 3)), paths[geometry][member]])
    boxes = original.corners(poses, dimensions)
    image = Image.new('RGB', (224, 160), 'white'); draw = ImageDraw.Draw(image)
    draw.line([pixel(p) for p in poses], fill='#0066dd', width=2)
    if representation == 'footprint':
        for j in range(4):
            draw.line([pixel(p) for p in boxes[:, j]], fill='#0066dd', width=1)
        for index in range(5, 41, 5):
            box = [pixel(p) for p in boxes[index]]; draw.line(box+[box[0]], fill='#0066dd', width=1)
    for index in (10, 20, 30, 40):
        x, y = pixel(poses[index]); draw.ellipse((x-2, y-2, x+2, y+2), fill='#cc5500')
    x, y = pixel(poses[0]); draw.ellipse((x-2, y-2, x+2, y+2), fill='black')
    draw.rectangle((0, 0, 223, 159), outline='#888888')
    return image


def payload(package, scene, manifest, sample, pair, paths, geometry, representation, mode, ordering):
    validate_paths(paths)
    if geometry not in GEOMETRIES:
        raise ValueError('Unknown path geometry')
    indices = [sample['pool_local'].index(i) for i in pair['local_ids']]
    trajectories = np.array(sample['trajectories'], dtype=float, copy=True)
    trajectories[indices] = paths[geometry]
    # Reuse the frozen road projection, dimensions, option IDs and prompt layout.
    row, positions, _ = original.payload(package, scene, manifest, dict(sample, trajectories=trajectories),
                                         pair, representation, mode, ordering)
    row['state']['candidate_images'] = row['state']['candidate_images'].replace(
        'These are nominal proposals, not predicted traffic or exact swept volumes.',
        'These paths describe candidate ego motion, not future traffic or exact swept volumes.')
    for j, member in enumerate(positions):
        row['images'][j+1].paste(shared_inset(paths, geometry, member, representation), (8, 472))
    hashes = []
    for image in row['images']:
        buffer = io.BytesIO(); image.save(buffer, format='PNG')
        hashes.append(hashlib.sha256(buffer.getvalue()).hexdigest())
    digest = base.digest(dict(row={k: v for k, v in row.items() if k != 'images'},
                              image_sha256=hashes, pair_positions=positions))
    return row, positions, digest
