"""CPU geometry/events/figures for post-hoc replay, never Drex input generation."""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import pair_diagnostic_data as pair_view
import overlay_v2_data as projection
import real_data as base

CORNER_NAMES = ('FRONT_LEFT', 'REAR_LEFT', 'REAR_RIGHT', 'FRONT_RIGHT')
POLYGON_ORDER = [0, 3, 2, 1, 0]  # PDM BBCoordsIndex order, not pair_view.corners order.


def local_xy(points, origin):
    points = np.asarray(points, dtype=float)
    c, s = np.cos(origin[2]), np.sin(origin[2])
    return (points-np.asarray(origin[:2]))@np.array([[c, -s], [s, c]])


def local_poses(poses, origin):
    result = np.asarray(poses, dtype=float).copy()
    result[..., :2] = local_xy(result[..., :2], origin)
    angle = result[..., 2]-origin[2]
    result[..., 2] = np.arctan2(np.sin(angle), np.cos(angle))
    return result


def trajectory_difference(reference, simulated):
    if reference.shape != simulated.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError('Time-aligned SE2 arrays required')
    if not np.isfinite(reference).all() or not np.isfinite(simulated).all():
        raise ValueError('Nonfinite replay poses')
    distance = np.linalg.norm(reference[:, :2]-simulated[:, :2], axis=1)
    angle = simulated[:, 2]-reference[:, 2]
    heading = np.abs(np.arctan2(np.sin(angle), np.cos(angle)))
    return dict(mean_position_error_m=float(distance.mean()), max_position_error_m=float(distance.max()),
                endpoint_position_error_m=float(distance[-1]),
                max_heading_error_rad=float(heading.max()))


def dac_trace(inside_corners, dt):
    inside_corners = np.asarray(inside_corners, dtype=bool)
    if inside_corners.ndim != 2 or inside_corners.shape[1] != 4:
        raise ValueError('Expected time x four PDM-ordered corners')
    bad = np.flatnonzero(~inside_corners.all(axis=1))
    first = int(bad[0]) if len(bad) else None
    return dict(score=0. if first is not None else 1., first_index=first,
                first_time_s=None if first is None else float(first*dt),
                first_outside_corners=[] if first is None else [CORNER_NAMES[j] for j in np.flatnonzero(~inside_corners[first])])


def lane_trace(distance, in_intersection, dt, limit, window):
    """Match PDM's sample-count rule and skip-without-reset intersection logic."""
    distance = np.asarray(distance, dtype=float)
    intersections = np.asarray(in_intersection, dtype=bool)
    if (distance.ndim != 1 or distance.shape != intersections.shape or not np.isfinite(distance).all()
            or not np.isfinite([dt, limit, window]).all() or dt <= 0 or window <= 0):
        raise ValueError('Invalid lane trace')
    required = int(math.ceil(window/dt)); count = 0; start = None; first = None
    counts = []
    for i, (value, intersection) in enumerate(zip(distance, intersections)):
        if not intersection:
            if value > limit:
                if count == 0:
                    start = i
                count += 1
            else:
                count = 0; start = None
        counts.append(count)
        if first is None and count >= required:
            first = i
            qualified_start = start
    return dict(score=0. if first is not None else 1., first_index=first,
                first_time_s=None if first is None else float(first*dt),
                first_run_start_time_s=None if first is None else float(qualified_start*dt),
                required_samples=required, threshold_m=float(limit), window_s=float(window),
                intersection_rule='skip_without_reset', counters=counts)


def point_projection(point, calib, size):
    uv, valid = projection.project(np.array([[point[0], point[1], 0.]]), calib)
    inside = bool(valid[0] and 0 <= uv[0, 0] < size[0] and 0 <= uv[0, 1] < size[1])
    return dict(admissible_ray=bool(valid[0]), projects_inside_current_photo=inside,
                pixel_xy=uv[0].tolist() if valid[0] else None, occlusion_tested=False)


def pixel_separation(a, b, calib, size):
    x, y = point_projection(a, calib, size), point_projection(b, calib, size)
    return dict(reference=x, simulated=y,
                separation_pixels=None if not (x['projects_inside_current_photo'] and y['projects_inside_current_photo'])
                else float(np.linalg.norm(np.array(x['pixel_xy'])-y['pixel_xy'])))


def camera_figure(photo, calib, reference, simulated, reference_boxes, simulated_boxes,
                  event, caption, target):
    original = photo.copy(); draw = ImageDraw.Draw(original)
    width = max(3, round(photo.width/400))
    for poses, color in ((reference, '#0066ee'), (simulated, '#ef7900')):
        for line in pair_view.projected_lines(poses[:, :2], calib, photo.size):
            draw.line(line, fill=color, width=width)
    index = event.get('index')
    if index is not None:
        for boxes, color in ((reference_boxes, '#0066ee'), (simulated_boxes, '#ef7900')):
            for line in pair_view.projected_lines(boxes[index, POLYGON_ORDER], calib, photo.size):
                draw.line(line, fill=color, width=width)
        for point in event.get('points_local', []):
            value = point_projection(point, calib, photo.size)
            if value['projects_inside_current_photo']:
                x, y = value['pixel_xy']; radius = max(6, round(photo.width/160))
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline='red', width=width)
    image = Image.new('RGB', (960, 640), 'white')
    image.paste(ImageOps.pad(original, (960, 540), color='white'), (0, 50))
    draw = ImageDraw.Draw(image)
    draw.text((8, 4), caption, fill='black', font=base.font(16))
    draw.text((8, 27), 'AUDIT ONLY: blue=nominal; orange=simulated; red=recorded metric event', fill='black', font=base.font(14))
    draw.text((8, 603), 'Projection into current photo is not an occlusion/visibility test; flat-ground approximation.', fill='black', font=base.font(13))
    image.save(target)


def map_figure(reference, simulated, boxes, polygons, centerline, event, collision_polygons, caption, target):
    image = Image.new('RGB', (960, 960), 'white')
    xy = np.concatenate([reference[:, :2], simulated[:, :2], boxes[..., :2].reshape(-1, 2)])
    lo, hi = xy.min(axis=0)-7, xy.max(axis=0)+7
    scale = min(820/(hi[0]-lo[0]), 820/(hi[1]-lo[1])); middle = (hi+lo)/2
    pixel = lambda p: (480-(p[1]-middle[1])*scale, 490-(p[0]-middle[0])*scale)
    for polygon in polygons:
        mask = Image.new('1', image.size, 0); painter = ImageDraw.Draw(mask)
        painter.polygon([pixel(p) for p in polygon['exterior']], fill=1)
        for hole in polygon['holes']:
            painter.polygon([pixel(p) for p in hole], fill=0)
        image.paste('#e6f1e6', mask=mask)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        for ring in [polygon['exterior'], *polygon['holes']]:
            draw.line([pixel(p) for p in ring], fill='#90b090', width=1)
    if len(centerline) >= 2:
        draw.line([pixel(p) for p in centerline], fill='#996699', width=2)
    for poses, color in ((reference, '#0066ee'), (simulated, '#ef7900')):
        draw.line([pixel(p) for p in poses], fill=color, width=3)
    for i in range(0, len(boxes), max(1, len(boxes)//5)):
        draw.line([pixel(p) for p in boxes[i, POLYGON_ORDER]], fill='#ef7900', width=2)
    if event.get('index') is not None:
        draw.line([pixel(p) for p in boxes[event['index'], POLYGON_ORDER]], fill='red', width=3)
    for point in event.get('points_local', []):
        x, y = pixel(point); draw.ellipse((x-5, y-5, x+5, y+5), fill='red')
    for polygon in collision_polygons:
        draw.line([pixel(p) for p in polygon], fill='#cc00aa', width=3)
    draw.rectangle((0, 0, 960, 64), fill='white')
    draw.text((8, 5), caption, fill='black', font=base.font(16))
    draw.text((8, 29), 'AUDIT ONLY: green=drivable map; purple=scorer centerline; magenta=event-time overlaps', fill='black', font=base.font(13))
    draw.rectangle((0, 905, 960, 960), fill='white')
    draw.text((8, 915), f'Up=forward, left=left; scale {scale:.2f} pixels/metre; blue=nominal, orange=simulated', fill='black', font=base.font(13))
    draw.text((8, 936), 'Map and event-time traffic are privileged audit context, never inputs to Drex.', fill='black', font=base.font(13))
    image.save(target)
