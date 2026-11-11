"""Input-only diagnostics of the frozen v1 overlay; never accepts/replaces features.

Brown-Conrady polynomials may fold outside the physical lens domain. A point
with positive camera depth can then re-enter the image on the wrong branch.
The radial branch and local Jacobian checks below detect such cases; passing
them does not establish calibration accuracy or a globally bijective lens.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re

import numpy as np
from numpy.polynomial import polynomial as poly

import overlay_data as view
import overlay_pilot as pilot
import real_data as base
from runtime import read, write
from transfer import sha


NOTE = (
    'Diagnostic only, not corrected/accepted model inputs. Existing plan, pixels, '
    'results and decisions are unchanged. No GPU or inference; no outcome '
    'arrays or Drex results are read. Radial branch/Jacobian violations flag extrapolation; '
    'their absence does not establish correct calibration, image rectification '
    'or road height. Magenta marks suspect segments from the original renderer; '
    'cyan marks other original segments, not verified road geometry.'
)


def lens_terms(distortion):
    d = np.zeros(8)
    d[:len(distortion)] = distortion
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    return np.array([1., k1, k2, k3]), np.array([1., k4, k5, k6]), p1, p2


def first_positive_root(coefficients):
    roots = poly.polyroots(poly.polytrim(coefficients))
    real = [float(z.real) for z in roots
            if abs(z.imag) <= 1e-8 * max(1., abs(z.real)) and z.real > 0]
    return min(real, default=float('inf'))


def radial_branch_limit(distortion):
    """First radial zero, pole or turning point, expressed as squared radius.

    For r_d=r*N(r^2)/D(r^2), the derivative numerator is
    N*D + 2*q*(N'*D - N*D'). Select the FIRST positive root, so a later
    apparently positive branch cannot be mistaken for the branch at the axis.
    Tangential distortion is checked locally below, not certified by this bound.
    """
    n, d, _, _ = lens_terms(distortion)
    slope = poly.polyadd(poly.polymul(n, d), 2 * poly.polymul(
        [0., 1.], poly.polysub(poly.polymul(poly.polyder(n), d),
                              poly.polymul(n, poly.polyder(d)))))
    return min(first_positive_root(p) for p in (n, d, slope))


def dense_path(poses):
    xy = np.vstack([np.zeros((1, 2)), poses[:, :2]])
    fraction = np.linspace(0, 1, 9)
    dense = np.concatenate([a + fraction[:, None] * (b-a)
                            for a, b in zip(xy, xy[1:])])
    times = np.concatenate([(i+fraction)*.1 for i in range(len(poses))])
    return np.column_stack([dense, np.zeros(len(dense))]), times


def inspect_path(poses, calib, size):
    points, times = dense_path(poses)
    r, t, _, distortion = calib
    camera = (points-t) @ r
    depth = camera[:, 2]
    uv, valid = view.project(points, calib)
    normalized = np.zeros((len(points), 2))
    normalized[valid] = camera[valid, :2] / depth[valid, None]
    x, y = normalized.T
    q = x*x + y*y
    n, d, p1, p2 = lens_terms(distortion)
    numerator, denominator = poly.polyval(q, n), poly.polyval(q, d)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        radial = numerator / denominator
        gradient = (poly.polyval(q, poly.polyder(n))*denominator -
                    numerator*poly.polyval(q, poly.polyder(d))) / denominator**2
        slope = radial + 2*q*gradient
        jxx = radial + 2*x*x*gradient + 2*p1*y + 6*p2*x
        jxy = 2*x*y*gradient + 2*p1*x + 2*p2*y
        jyy = radial + 2*y*y*gradient + 6*p1*y + 2*p2*x
        determinant = jxx*jyy - jxy*jxy
    limit = radial_branch_limit(distortion)
    outside = q >= limit
    suspect = valid & (outside | (radial <= 0) | (slope <= 0) |
                       (denominator <= 0) | (determinant <= 0) |
                       ~np.isfinite(determinant))
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < size[0]) & \
        (uv[:, 1] >= 0) & (uv[:, 1] < size[1])
    segments = []
    for i in range(len(points)-1):
        if valid[i] and valid[i+1]:
            line = view.clipped_segment(uv[i], uv[i+1], *size)
            if line is not None:
                segments.append(dict(line=line, suspect=bool(suspect[i] or suspect[i+1]),
                                     point_indices=[i, i+1]))
    suspicious_indices = sorted({j for s in segments if s['suspect']
                                 for j in s['point_indices']})
    length = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), poses[:, :2]]), axis=0), axis=1).sum()
    stats = dict(path_length_m=float(length), endpoint_xy_m=poses[-1, :2].tolist(),
                 max_forward_m=float(max(0., poses[:, 0].max())),
                 radial_branch_radius=None if np.isinf(limit) else float(np.sqrt(limit)),
                 original_visible_segments=len(segments),
                 suspect_visible_segments=sum(s['suspect'] for s in segments),
                 original_in_frame_points=int(inside.sum()),
                 suspect_in_frame_points=int((inside & suspect).sum()))
    if suspicious_indices:
        stats['suspect_segment_endpoint_ranges'] = dict(
            time_s=[float(times[suspicious_indices].min()), float(times[suspicious_indices].max())],
            forward_m=[float(points[suspicious_indices, 0].min()), float(points[suspicious_indices, 0].max())],
            camera_depth_m=[float(depth[suspicious_indices].min()), float(depth[suspicious_indices].max())])
    trace = dict(time_s=times, forward_m=points[:, 0], left_m=points[:, 1],
                 camera_depth_m=depth, normalized_x=x, normalized_y=y,
                 pixel_x=uv[:, 0], pixel_y=uv[:, 1], radial=radial,
                 radial_slope=slope, jacobian_det=determinant,
                 original_valid=valid, original_in_frame=inside, suspect=suspect)
    return stats, segments, trace


def save_preview(photo, segments, path, caption):
    from PIL import ImageDraw, ImageOps
    rendered = photo.copy()
    draw = ImageDraw.Draw(rendered)
    for s in segments:
        draw.line(s['line'], fill='#ff00ff' if s['suspect'] else '#00bfff',
                  width=max(2, round(photo.width/450)))
    rendered = ImageOps.pad(rendered, (768, 432), color='white')
    draw = ImageDraw.Draw(rendered)
    draw.rectangle((0, 0, 768, 46), fill='white')
    draw.text((5, 2), caption, fill='black', font=base.font(14))
    draw.text((5, 23), 'DIAGNOSTIC: magenta=suspect; cyan=other original segments',
              fill='black', font=base.font(13))
    rendered.save(path)


def run(root, tag, preview_scenes=4):
    from PIL import Image
    if not re.fullmatch(r'overlay_[A-Za-z0-9_-]{1,64}', tag):
        raise ValueError('Use the existing overlay_* run name')
    out = root/'runs'/tag
    plan = pilot.check_plan(out)
    package = root/'data/navsim_train_v1/package'
    identity = read(out/'base/identity.json')
    if sha(package/'package_manifest.json') != identity['package_manifest_sha256']:
        raise RuntimeError('Package manifest differs from frozen overlay run')
    manifest, _, scenes, vocab = base.package_metadata(package)
    target = out/'projection_audit_v1'
    base.freeze(target/'identity.json', dict(
        plan_sha256=sha(out/'plan.json'), audit_sha256=sha(Path(__file__)),
        preview_scenes=preview_scenes, note=NOTE))
    rows = []
    for i, record in enumerate(plan['records']):
        token = record['token']
        sample = base.load_inputs(package, token, manifest, vocab)
        photo_path = base.camera_paths(package, scenes[token], manifest)[4]
        relative = photo_path.relative_to(package).as_posix()
        if sha(photo_path) != manifest['files'][relative]['sha256']:
            raise RuntimeError('Current photograph changed: '+token)
        with Image.open(photo_path) as image:
            photo = image.convert('RGB')
        calib = view.calibration(scenes[token], photo.size)
        names = view.ids(token)
        candidates = []
        trace_path = target/(token+'-trace.csv')
        with trace_path.open('w', encoding='utf-8', newline='') as stream:
            writer = None
            for pid, poses in enumerate(sample['trajectories'][:4]):
                stats, segments, trace = inspect_path(poses, calib, photo.size)
                stats.update(candidate_name=names[pid], teacher_rank=pid+1,
                             candidate_vocab=sample['pool_vocab'][pid])
                candidates.append(stats)
                if i < preview_scenes:
                    save_preview(photo, segments, target/f'{token}-{names[pid]}-diagnostic.png',
                                 f'{names[pid]} | path {stats["path_length_m"]:.2f} m | '
                                 f'suspect segments {stats["suspect_visible_segments"]}/{len(segments)}')
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=['candidate_name', *trace])
                    writer.writeheader()
                for j in range(len(trace['time_s'])):
                    row = {key: value[j].item() for key, value in trace.items()}
                    row = {k: v if not isinstance(v, float) or np.isfinite(v) else None
                           for k, v in row.items()}
                    writer.writerow(dict(candidate_name=names[pid], **row))
        rows.append(dict(token=token, image_size=list(photo.size),
                         camera=scenes[token]['current_camera_calibrations']['cam_f0'],
                         ego_velocity_xy_mps=sample['ego_velocity'], candidates=candidates))
    summary = dict(status='diagnostic_only', scenes=len(rows),
                   scenes_with_suspect_segments=sum(any(c['suspect_visible_segments'] for c in r['candidates']) for r in rows),
                   scenes_without_original_projection=sum(all(c['original_visible_segments'] == 0 for c in r['candidates']) for r in rows),
                   suspect_visible_segments=sum(c['suspect_visible_segments'] for r in rows for c in r['candidates']),
                   output=str(target), outcomes_read=False, note=NOTE)
    write(target/'per_scene.json', rows)
    write(target/'report.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    for row in rows[:preview_scenes]:
        print(json.dumps(row, indent=2), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', default=os.environ.get('DREX_OVERLAY_NAME', 'overlay_top4_v1'))
    p.add_argument('--preview-scenes', type=int, default=4)
    a = p.parse_args()
    if not 0 <= a.preview_scenes <= 64:
        p.error('--preview-scenes must be between 0 and 64')
    run(a.root.expanduser().resolve(), a.name, a.preview_scenes)


if __name__ == '__main__':
    main()
