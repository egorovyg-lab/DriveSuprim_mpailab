"""Privileged sampled-footprint geometry, distinct from PDM corner-only DAC."""
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

BUFFERS_M = (0., .1, .25, .5)


def road_union(polygons):
    if not polygons or any(p.is_empty or not p.is_valid or p.geom_type not in ('Polygon', 'MultiPolygon') for p in polygons):
        raise ValueError('Valid nonempty drivable polygons required; no automatic geometry repair')
    result = unary_union(polygons)
    if result.is_empty or not result.is_valid:
        raise ValueError('Invalid union of drivable polygons')
    return result


def frame_measurement(coords, road, actors):
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (5, 2) or not np.isfinite(coords).all():
        raise ValueError('Expected PDM corners FL/RL/RR/FR and center')
    footprint = Polygon(coords[:4])
    if not footprint.is_valid or footprint.area <= 0:
        raise ValueError('Invalid vehicle footprint')
    covered = bool(road.covers(footprint))
    signed = []
    for xy in coords[:4]:
        point = Point(xy)
        signed.append(float(point.distance(road.boundary) if road.contains(point) else -point.distance(road)))
    nearest, nearest_token, contacts = None, None, []
    for token, shape in actors:
        if shape.is_empty or not shape.is_valid:
            raise ValueError('Invalid traffic object geometry')
        distance = float(footprint.distance(shape))
        if nearest is None or distance < nearest:
            nearest, nearest_token = distance, token
        if footprint.intersects(shape): contacts.append(token)
    return dict(full_footprint_covered=covered,
        # Outside footprints have no positive clearance. Penetration is reported as area/corner distances instead.
        road_clearance_m=float(footprint.distance(road.boundary)) if covered else None,
        outside_area_m2=float(footprint.difference(road).area),
        corner_signed_margins_m=signed, min_corner_signed_margin_m=min(signed),
        actor_clearance_m=nearest, nearest_actor_token=nearest_token, intersecting_actor_tokens=sorted(contacts))


def summarize(frames, dt, pdm_corner_pass, lane):
    if not frames or not np.isfinite(dt) or dt <= 0: raise ValueError('Nonempty sampled trace required')
    covered = all(r['full_footprint_covered'] for r in frames)
    minimum = min(r['road_clearance_m'] for r in frames) if covered else None
    first_out = next((i for i, r in enumerate(frames) if not r['full_footprint_covered']), None)
    first_contact = next((i for i, r in enumerate(frames) if r['intersecting_actor_tokens']), None)
    actor_distances = [r['actor_clearance_m'] for r in frames if r['actor_clearance_m'] is not None]
    actor_min = min(actor_distances) if actor_distances else None
    eligible = {}
    for buffer in BUFFERS_M:
        eligible[str(buffer)] = bool(pdm_corner_pass and covered and minimum >= buffer and lane['score'] == 1 and
            first_contact is None and (actor_min is None or actor_min >= buffer))
    return dict(full_footprint_covered_all_samples=covered, min_road_clearance_m=minimum,
        initial_full_footprint_covered=frames[0]['full_footprint_covered'],
        first_full_footprint_outside_time_s=None if first_out is None else first_out*dt,
        max_outside_area_m2=max(r['outside_area_m2'] for r in frames),
        min_corner_signed_margin_m=min(r['min_corner_signed_margin_m'] for r in frames),
        pdm_corner_pass=bool(pdm_corner_pass), lane_keeping=lane['score'],
        lane_required_samples=lane['required_samples'], max_lane_counter_samples=max(lane['counters']),
        first_lane_qualification_time_s=lane['first_time_s'],
        min_actor_clearance_m=actor_min,
        first_actor_contact_time_s=None if first_contact is None else first_contact*dt,
        sampled_geometry_eligible_at_buffer_m=eligible)
