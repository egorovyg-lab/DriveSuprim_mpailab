"""Current-image projection diagnostics, not a perception model or safety gate.

OpenScene's original converter names camera-to-ego calibration sensor2lidar.
This module uses that documented source contract explicitly. Points lie on the
current ego ground plane; projected/FOV does not mean unoccluded or observed.
"""
import numpy as np

CAMERAS = ("cam_l0", "cam_f0", "cam_r0")
CALIBRATION_CONTRACT = "openscene_original_nuplan_camera_to_ego"
# corridor_geometry.corners orders front-left, front-right, rear-left, rear-right.
FOOTPRINT_EDGES = ((0, 1), (1, 3), (3, 2), (2, 0))


def rotation(value):
    q = np.asarray(value, dtype=float)
    if q.shape == (4,):
        if not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
            raise ValueError("Expected unit quaternion in w,x,y,z order")
        w, x, y, z = q / np.linalg.norm(q)
        q = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    if (q.shape != (3, 3) or not np.isfinite(q).all()
            or not np.allclose(q.T @ q, np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(q), 1, atol=1e-5, rtol=0)):
        raise ValueError("Expected a proper 3D rotation")
    return q


def calibration(camera):
    r = rotation(camera["sensor2lidar_rotation"])
    t = np.asarray(camera["sensor2lidar_translation"], dtype=float)
    k = np.asarray(camera["cam_intrinsic"], dtype=float)
    if camera["distortion"] is None:
        raise ValueError("Missing distortion; no silent pinhole fallback")
    d = np.asarray(camera["distortion"], dtype=float).reshape(-1)
    if (t.shape != (3,) or k.shape != (3, 3) or not np.isfinite(t).all()
            or not np.isfinite(k).all() or not np.isfinite(d).all()
            or not np.allclose(k[2], [0, 0, 1], atol=1e-8, rtol=0)
            or k[0, 0] <= 0 or k[1, 1] <= 0):
        raise ValueError("Invalid camera translation/intrinsics/distortion")
    if len(d) not in (0, 4, 5, 8):
        raise ValueError("Unsupported Brown-Conrady distortion length: " + str(len(d)))
    return r, t, k, d


def current_observation(frame):
    """Only this current-frame allowlist is copied; no annotations/actors."""
    cameras = {key.lower(): value for key, value in frame["cams"].items() if key.lower() in CAMERAS}
    if set(cameras) != set(CAMERAS):
        raise ValueError("Three current camera calibrations required")
    r = rotation(frame["ego2global_rotation"])
    t = np.asarray(frame["ego2global_translation"], dtype=float)
    if t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError("Invalid current ego translation")
    result = dict(token=str(frame["token"]), timestamp_us=int(frame["timestamp"]),
        ego_rotation=r.tolist(), ego_translation=t.tolist(), cameras={})
    for name in CAMERAS:
        camera = cameras[name]
        values = {key: camera[key] for key in (
            "data_path", "sensor2lidar_rotation", "sensor2lidar_translation", "cam_intrinsic", "distortion")}
        cr, ct, k, d = calibration(values)
        result["cameras"][name] = dict(data_path=str(values["data_path"]),
            sensor2lidar_rotation=cr.tolist(), sensor2lidar_translation=ct.tolist(),
            cam_intrinsic=k.tolist(), distortion=d.tolist())
    return result


def ground_ego_points(world_xy, observation):
    """Lift XY onto the current ego z=0 plane, preserving world X/Y.

    This is a declared flat-plane assumption, not future road elevation.
    Full current roll/pitch is retained; pose origin is checked by the runner.
    """
    r = rotation(observation["ego_rotation"])
    t = np.asarray(observation["ego_translation"], dtype=float)
    xy = np.asarray(world_xy, dtype=float)
    if xy.shape[-1] != 2 or not np.isfinite(xy).all() or abs(r[2, 2]) < .5:
        raise ValueError("Invalid points/current ground-plane orientation")
    delta = xy - t[:2]
    z = -(delta @ r[:2, 2]) / r[2, 2]
    return np.concatenate([delta, z[..., None]], axis=-1) @ r


def project(points_ego, camera, near=.1):
    """Camera-to-ego extrinsics, OpenCV Brown-Conrady distortion; no clipping."""
    r, t, k, d = calibration(camera)
    points = (np.asarray(points_ego, dtype=float) - t) @ r
    depth = points[..., 2]
    front = depth > near
    safe_depth = np.where(front, depth, 1.)
    x, y = points[..., 0] / safe_depth, points[..., 1] / safe_depth
    coefficients = np.zeros(8)
    coefficients[:len(d)] = d
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
    r2 = x*x + y*y
    denominator = 1 + k4*r2 + k5*r2*r2 + k6*r2*r2*r2
    valid = front & np.isfinite(r2) & (np.abs(denominator) > 1e-8)
    denominator = np.where(valid, denominator, 1.)
    radial = (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) / denominator
    xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
    normalized = np.stack([xd, yd, np.ones_like(xd)], axis=-1) @ k.T
    uv = normalized[..., :2]
    valid &= np.isfinite(uv).all(axis=-1)
    uv = np.where(valid[..., None], uv, np.nan)
    return uv, valid, depth


def crop_boxes(sizes):
    """Exact three-camera crops in the frozen DriveSuprim feature builder."""
    boxes = {}
    for name in CAMERAS:
        width, height = sizes[name]
        side = 416 if name != "cam_f0" else 0
        if width <= 2*side or height <= 56:
            raise ValueError("Images too small for frozen DriveSuprim crops")
        boxes[name] = (side, 28, width-side, height-28)
    if len({b[3]-b[1] for b in boxes.values()}) != 1:
        raise ValueError("Camera crop heights disagree")
    return boxes


def panorama_projection(points_ego, cameras, sizes, output_size):
    boxes = crop_boxes(sizes)
    width = sum(b[2]-b[0] for b in boxes.values())
    height = boxes[CAMERAS[0]][3] - boxes[CAMERAS[0]][1]
    scale = np.array(output_size, dtype=float) / [width, height]
    if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid panorama size")
    result, offset = {}, 0
    for name in CAMERAS:
        uv, front, depth = project(points_ego, cameras[name])
        left, top, right, bottom = boxes[name]
        in_crop = front & (uv[..., 0] >= left) & (uv[..., 0] < right)
        in_crop &= (uv[..., 1] >= top) & (uv[..., 1] < bottom)
        panorama = (uv - [left, top] + [offset, 0] + .5) * scale - .5
        result[name] = dict(uv=uv, panorama=panorama, in_crop=in_crop, depth=depth)
        offset += right-left
    return result


def current_body_mask(world_xy, origin, dimensions):
    delta = np.asarray(world_xy) - np.asarray(origin)[:2]
    yaw = float(origin[2])
    r = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy = delta @ r
    length, width, offset = dimensions
    return (np.abs(xy[..., 0]-offset) <= length/2 + 1e-8) & (np.abs(xy[..., 1]) <= width/2 + 1e-8)


def road_membership(points, polygons):
    import shapely
    if hasattr(shapely, "contains_xy"):
        contains = shapely.contains_xy
    else:
        from shapely.vectorized import contains
    inside = np.zeros(np.asarray(points).shape[:-1], dtype=bool)
    for polygon in polygons:
        inside |= contains(polygon, points[..., 0], points[..., 1])
    return inside
