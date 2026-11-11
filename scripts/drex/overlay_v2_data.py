"""V2 overlays restricted to a conservative central invertible lens domain.

V1 is immutable. Extrinsics, cameras, candidates, names and orders are reused;
only projection admissibility and its input description change.
"""
from functools import lru_cache
import hashlib
import io

import numpy as np
from numpy.polynomial import polynomial as poly

import overlay_data as v1
import real_data as base

MODES, ORDERS, TASKS = v1.MODES, v1.ORDERS, v1.TASKS
calibration, ids, order, inset = v1.calibration, v1.ids, v1.order, v1.inset
CONFIG = dict(v1.CONFIG, protocol='overlay_top4_v2', near_depth_m=.2,
              domain_relative_margin=1e-7, domain_numeric_margin=1e-8,
              projection_domain='central convex disk with positive Brown-Conrady Jacobian bound',
              segment_rule='adjacent admissible dense points only; never bridge rejected points')
NOTE = (v1.NOTE + ' V2 excludes rays outside a conservative central lens domain '
        'computed from calibration coefficients. It may omit valid peripheral rays; '
        'this fixes detected polynomial foldbacks, not calibration/terrain/occlusion. '
        'All scenes remain included, including those with no visible path. '
        'V1 inference results are incompatible with the changed pixels.')


def _positive_root(coefficients):
    coefficients = poly.polytrim(coefficients)
    if not np.isfinite(coefficients).all():
        raise ValueError('Nonfinite lens-domain polynomial')
    roots = poly.polyroots(coefficients)
    if not np.isfinite(roots).all():
        raise ValueError('Nonfinite lens-domain roots')
    real = [float(z.real) for z in roots
            if z.real > 0 and abs(z.imag) <= 1e-8*max(1., abs(z.real))]
    return min(real, default=float('inf'))


def _even(coefficients):
    result = np.zeros(2*len(coefficients)-1)
    result[::2] = coefficients
    return result


@lru_cache(maxsize=256)
def lens_domain(distortion):
    """Return a conservative radius about the optical axis in normalized rays.

    Radial Jacobian eigenvalues are s=N/D and t=s+2*q*ds/dq, q=r^2.
    The tangential Jacobian's smallest eigenvalue is >= -6*r*hypot(p1,p2).
    Require s and t to exceed this bound on the ENTIRE disk, before the first
    root/pole. Positive-definite Jacobian on this convex disk makes the mapping
    injective there. Checking only the Jacobian at one point is insufficient:
    an outer folded branch can have a positive Jacobian again.
    """
    if len(distortion) not in (0, 4, 5, 8) or not np.isfinite(distortion).all():
        raise ValueError('Unsupported distortion')
    d = np.zeros(8)
    d[:len(distortion)] = distortion
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    n, den = np.array([1., k1, k2, k3]), np.array([1., k4, k5, k6])
    tangent = 6*np.hypot(p1, p2)
    slope = poly.polyadd(poly.polymul(n, den), 2*poly.polymul(
        [0., 1.], poly.polysub(poly.polymul(poly.polyder(n), den),
                              poly.polymul(n, poly.polyder(den)))))
    # Express the inequalities in r (not q); denominators remain positive
    # before their first root. The constant terms are all one at the axis.
    nr, dr, sr = _even(n), _even(den), _even(slope)
    radial_bound = poly.polysub(nr, tangent*poly.polymul([0., 1.], dr))
    slope_bound = poly.polysub(sr, tangent*poly.polymul([0., 1.], poly.polymul(dr, dr)))
    radius = min(_positive_root(c) for c in (dr, radial_bound, slope_bound))
    if radius <= 0:
        raise ValueError('Empty lens domain')
    return radius


def project(points, calib):
    r, t, k, distortion = calib
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite 3D points')
    camera = (points-t) @ r
    valid = camera[:, 2] > CONFIG['near_depth_m']
    normalized = np.zeros((len(points), 2))
    normalized[valid] = camera[valid, :2]/camera[valid, 2, None]
    radius = lens_domain(tuple(np.asarray(distortion, dtype=float)))
    with np.errstate(over='ignore', invalid='ignore'):
        radius_at_point = np.hypot(normalized[:, 0], normalized[:, 1])
    valid &= np.isfinite(radius_at_point)
    if np.isfinite(radius):
        valid &= radius_at_point < radius*(1-CONFIG['domain_relative_margin'])
    # Do not even evaluate the lens polynomial on discarded near-camera rays.
    x, y = normalized[valid].T
    q = x*x+y*y
    d = np.zeros(8)
    d[:len(distortion)] = distortion
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    n, den = np.array([1., k1, k2, k3]), np.array([1., k4, k5, k6])
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        numerator, denominator = poly.polyval(q, n), poly.polyval(q, den)
        s = numerator/denominator
        gradient = (poly.polyval(q, poly.polyder(n))*denominator -
                    numerator*poly.polyval(q, poly.polyder(den)))/denominator**2
        radial_slope = s+2*q*gradient
        bound = 6*np.hypot(p1, p2)*np.hypot(x, y)
        eps = CONFIG['domain_numeric_margin']
        admissible = ((denominator > eps) & (s-bound > eps) &
                      (radial_slope-bound > eps))
        xd = x*s+2*p1*x*y+p2*(q+2*x*x)
        yd = y*s+p1*(q+2*y*y)+2*p2*x*y
        pixels = np.column_stack([k[0, 0]*xd+k[0, 2], k[1, 1]*yd+k[1, 2]])
    admissible &= np.isfinite(pixels).all(axis=1)
    uv = np.full((len(points), 2), np.nan)
    indices = np.flatnonzero(valid)
    uv[indices[admissible]] = pixels[admissible]
    valid[indices[~admissible]] = False
    return uv, valid


def dense_path(poses):
    poses = np.asarray(poses, dtype=float)
    if poses.shape != (40, 3) or not np.isfinite(poses).all():
        raise ValueError('Expected 40 finite local poses')
    xy = np.vstack([np.zeros((1, 2)), poses[:, :2]])
    dense = np.concatenate([a+np.linspace(0, 1, 9)[:, None]*(b-a)
                            for a, b in zip(xy, xy[1:])])
    return np.column_stack([dense, np.zeros(len(dense))])


def segment_details(poses, calib, size):
    points = dense_path(poses)
    uv, valid = project(points, calib)
    segments = []
    for i in range(len(points)-1):
        if valid[i] and valid[i+1]:
            line = v1.clipped_segment(uv[i], uv[i+1], *size)
            if line is not None:
                segments.append(dict(line=line, point_indices=[i, i+1]))
    return segments, valid


def path_segments(poses, calib, size):
    return [s['line'] for s in segment_details(poses, calib, size)[0]]


def pictures(photo, trajectories, calib, names, mode, ordering):
    from PIL import Image, ImageDraw, ImageOps
    size = tuple(CONFIG['image_size'])
    reference = ImageOps.pad(photo, size, color='white') if mode != 'blank' else Image.new('RGB', size, 'white')
    output = [reference]
    for pid in ordering:
        original = photo.copy() if mode == 'overlay' else Image.new('RGB', photo.size, 'white')
        draw = ImageDraw.Draw(original)
        if mode in ('overlay', 'blank'):
            for line in path_segments(trajectories[pid], calib, photo.size):
                draw.line(line, fill='#00bfff', width=max(2, round(photo.width/450)))
        elif mode != 'separate':
            raise ValueError('Unknown representation mode')
        rendered = ImageOps.pad(original, size, color='white')
        draw = ImageDraw.Draw(rendered)
        draw.rectangle((0, 0, size[0], 28), fill='white')
        draw.text((8, 4), f'{names[pid]} | 4 second candidate | ground projection v2',
                  fill='black', font=base.font(16))
        rendered.paste(inset(trajectories, pid), (size[0]-168, size[1]-128))
        if mode == 'separate':
            rendered.paste(inset(trajectories, pid).resize((480, 360)), (120, 40))
        output.append(rendered)
    return output


def payload(package, scene, manifest, sample, mode, kind):
    from PIL import Image
    token = scene['metadata']['token']
    path = base.camera_paths(package, scene, manifest)[4]
    with Image.open(path) as f:
        photo = f.convert('RGB')
    calib = calibration(scene, photo.size)
    names, ordering = ids(token), order(token, kind)
    images = pictures(photo, sample['trajectories'], calib, names, mode, ordering)
    row = dict(state=dict(task='Select the best four-second ego trajectory from four candidates.',
        image_1='Unannotated current front view; may be blank in the road-image control.',
        candidate_images='Images 2..5 are the four candidates, each labelled with its stable ID. '
            'Blue projected centrelines assume flat ground and do not model occlusion. '
            'Small insets show the entire path, up=forward and left=left, same metric scale. '
            'Parts behind the camera, outside its view or outside the central lens domain are not drawn. '
            'Separate-schematic condition shows the path without a road background. '
            'Missing projected pixels do not mean the vehicle stops.',
        velocity_xy_mps=sample['ego_velocity'], acceleration_xy_mps2=sample['ego_acceleration']),
        question=dict(type='choice', criteria={names[pid]: f'Candidate {names[pid]} in image {j+2}.'
                                              for j, pid in enumerate(ordering)},
            instructions='Choose the best available candidate. Prioritize staying on the drivable road, '
            'avoiding collisions and obeying traffic rules, then useful progress and comfort. '
            'Read the full path inset as well as visible projected parts. No candidate is labelled as baseline.'),
        images=images)
    hashes = []
    for image in images:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        hashes.append(hashlib.sha256(buffer.getvalue()).hexdigest())
    return row, ordering, base.digest(dict(row={k: v for k, v in row.items() if k != 'images'},
                                          images=hashes, pool_positions=ordering))
