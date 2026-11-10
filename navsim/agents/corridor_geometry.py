"""Static-map, sampled-footprint diagnostic. No learned scores or future actors.

This is a privileged geometric positive control, not a camera-only policy and
not a proof that the continuous swept vehicle body is collision-free.
"""
import numpy as np

PACKET_KEYS = frozenset(("candidate_ids", "local_indices", "baseline_position", "selection_scores",
    "raw_poses", "rollout_poses", "vehicle_dimensions", "polygon_bytes", "polygon_offsets", "sampling"))


def corners(poses, dimensions):
    """Rear-axle SE2 -> four vehicle corners; dimensions=length,width,axle_to_center."""
    poses, dimensions = np.asarray(poses, float), np.asarray(dimensions, float)
    if (poses.shape[-1] != 3 or dimensions.shape != (3,) or not np.isfinite(poses).all()
            or not np.isfinite(dimensions).all() or np.any(dimensions[:2] <= 0)):
        raise ValueError("Invalid vehicle poses/dimensions")
    length, width, offset = dimensions
    heading = poses[..., 2]
    longitudinal = np.stack([np.cos(heading), np.sin(heading)], axis=-1)
    lateral = np.stack([-np.sin(heading), np.cos(heading)], axis=-1)
    center = poses[..., :2] + offset * longitudinal
    return np.stack([center + sx * length / 2 * longitudinal + sy * width / 2 * lateral
                     for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1))], axis=-2)


def encode_polygons(polygons):
    if not polygons:
        raise ValueError("Static drivable polygons required")
    chunks = [np.frombuffer(p.wkb, dtype=np.uint8) for p in polygons]
    return np.concatenate(chunks), np.r_[0, np.cumsum([len(c) for c in chunks])].astype(np.int64)


def decode_polygons(data, offsets):
    from shapely import wkb
    if (data.dtype != np.uint8 or data.ndim != 1 or offsets.ndim != 1 or offsets.dtype.kind not in "iu"
            or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(data) or np.any(np.diff(offsets) <= 0)):
        raise ValueError("Invalid static polygon encoding")
    polygons = [wkb.loads(data[int(a):int(b)].tobytes()) for a, b in zip(offsets[:-1], offsets[1:])]
    if any(p.is_empty or not p.is_valid or p.geom_type not in ("Polygon", "MultiPolygon") for p in polygons):
        raise ValueError("Invalid static map polygon; no automatic geometry repair")
    return polygons


def check_poses(poses, dimensions, polygons):
    import shapely
    points = corners(poses, dimensions)
    inside = np.zeros(points.shape[:-1], dtype=bool)
    if not polygons:
        raise ValueError("An empty map cannot certify a candidate")
    if hasattr(shapely, "contains_xy"):
        contains = shapely.contains_xy
    else:
        from shapely.vectorized import contains
    # Per-polygon strict containment matches the original DAC corner test.
    # Unioning polygons would change the treatment of shared boundaries.
    for polygon in polygons:
        inside |= contains(polygon, points[..., 0], points[..., 1])
    all_corners = inside.all(axis=-1)
    passed = all_corners.all(axis=-1)
    outside_samples = (~all_corners).sum(axis=-1)
    first_outside = np.where(passed, -1, np.argmax(~all_corners, axis=-1))
    return dict(passed=passed, outside_samples=outside_samples, first_outside=first_outside)


def validate_packet(packet):
    if set(packet) != PACKET_KEYS:
        raise ValueError("Geometry packet must contain only the inference allowlist")
    ids, ix, scores = [packet[k] for k in ("candidate_ids", "local_indices", "selection_scores")]
    b = int(packet["baseline_position"])
    n = len(ids)
    if (n != 16 or ids.shape != (n,) or ix.shape != (n,) or ids.dtype.kind not in "iu" or ix.dtype.kind not in "iu"
            or len(set(ids.tolist())) != n or len(set(ix.tolist())) != n or not 0 <= b < n
            or scores.shape != (n,) or b != int(np.argmax(scores))):
        raise ValueError("Invalid frozen top16 alignment")
    shape = packet["raw_poses"].shape
    if (len(shape) != 3 or shape[0] != n or shape[2] != 3 or shape[1] < 2
            or packet["rollout_poses"].shape != shape or packet["vehicle_dimensions"].shape != (3,)
            or packet["sampling"].shape != (2,)
            or packet["sampling"][0] != shape[1] - 1 or packet["sampling"][1] <= 0
            or np.any(packet["vehicle_dimensions"][:2] <= 0)):
        raise ValueError("Invalid geometry packet sampling/shape")
    if any(not np.isfinite(v).all() for v in packet.values()):
        raise ValueError("Nonfinite geometry packet")
    if not np.allclose(packet["raw_poses"][:, 0], packet["rollout_poses"][:, 0], atol=1e-8, rtol=0):
        raise ValueError("Reference/rollout initial ego mismatch")
    decode_polygons(packet["polygon_bytes"], packet["polygon_offsets"])


def decide(packet, mode):
    validate_packet(packet)
    if mode not in ("raw", "rollout"):
        raise ValueError("Unknown corridor mode")
    polygons = decode_polygons(packet["polygon_bytes"], packet["polygon_offsets"])
    check = check_poses(packet[mode + "_poses"], packet["vehicle_dimensions"], polygons)
    b = int(packet["baseline_position"])
    if check["passed"][b]:
        chosen, reason = b, "baseline_passes"
    elif check["passed"].any():
        # Preserve the teacher ordering; geometry never invents a PDM gain.
        best = np.where(check["passed"], packet["selection_scores"], -np.inf).max()
        tied = np.flatnonzero(check["passed"] & (packet["selection_scores"] == best))
        chosen, reason = int(tied[0]), "baseline_rejected_choose_first_valid"
    else:
        chosen, reason = b, "no_candidate_passes_keep_baseline"
    return dict(chosen=int(chosen), reason=reason, **check)
