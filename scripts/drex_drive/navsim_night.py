"""Run a matched NAVSIM straight-road experiment over the full archived TRAIN inventory."""

import argparse
from base64 import b64encode
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
from random import Random
from statistics import mean
from math import cos, hypot, isclose, isfinite, pi, sin
import json
import sys
from time import perf_counter
from zlib import compress

if __package__:
    from . import navsim_straight as straight
else:
    import _drex_night_straight as straight


PROTOCOL = "navsim-straight-night-v1"
# Each variant changes one part of the action-effects control.
VARIANTS = {
    "control": ("baseline", "action-effects", "standard"),
    "radar": ("baseline", "radar", "standard"),
    "gap-forecast": ("baseline", "gap-forecast", "standard"),
    "leader-context": ("baseline", "leader-context", "standard"),
    "action-gap-end": ("baseline", "action-gap-end", "standard"),
    "gentle": ("baseline", "action-effects", "gentle"),
    "sufficient-gap": ("sufficient-gap", "action-effects", "standard"),
    "assess-hold": ("assess-hold", "action-effects", "standard"),
    "action-deltas": ("baseline", "action-deltas", "standard"),
    "short-preview": ("baseline", "short-preview", "standard"),
    "dual-preview": ("baseline", "dual-preview", "standard"),
}
EPSILON = 1e-8
COLLISION_TOKEN = "cf6c2163667d51d0"
BRAKE_GAP_EXPERIMENT = "corridor-preview-lead-brake-gap-v1"
BRAKE_GAP_CONTROL = "corridor-preview"
BRAKE_GAP_TARGET = "f43160170a665c31"
BRAKE_GAP_FIELD = "minimum_lead_gap_if_braking_m"
BRAKE_GAP_PLAN_SHA256 = "d73d68de6d9414b0ed7cd3348d10c96f084309a772261262471fca023b0588b3"
BRAKE_GAP_REPORT_SHA256 = "49fb38b5c1ccf822449b6d475b77d67a34f00c7cf3a1b81f2838361758356116"
BRAKE_GAP_PROTECTED = ("no_at_fault_collisions", "time_to_collision_within_bound", "drivable_area_compliance",
                       "driving_direction_compliance", "traffic_light_compliance", "lane_keeping", "history_comfort")
CORRIDOR_VARIANT = "corridor-objects"
CORRIDOR_MARGIN_M = 0.75
PREVIEW_SETUPS = {
    "corridor-preview": ("baseline", "corridor-preview", "standard"),
    "corridor-preview-instruction": ("corridor-risk", "corridor-preview", "standard"),
}


def leader_category(observation):
    if not observation["leaders"]:
        return "free-road"
    speed = observation["leaders"][0]["speed_mps"]
    relative = speed - observation["speed_mps"]
    if speed <= 0.05:
        return "stopped-leader"
    if relative > 0.25:
        return "opening-gap"
    if relative < -0.25:
        return "closing-gap"
    return "similar-speed"


def prepare(args, pilot):
    from rollout_replay import bound_context

    archive, candidates, source_id = straight.source_context(args.package, args.source)
    if pilot["protocol"] != straight.PROTOCOL or pilot["anchor_identity_sha256"] != source_id:
        raise RuntimeError("The original TRAIN pilot differs from the frozen archive")
    sessions = {row["session_id"] for row in pilot["scenes"]}
    rows = sorted(archive, key=lambda row: (row["session_id"], row["token"]))
    selected, rejected = [], Counter()
    for start in range(0, len(rows), 48):
        block = rows[start:start + 48]
        anchors = [candidates[record["token"]][0] for record in block]
        _, scenes, _, _, caches = bound_context(args.package, args.repo, anchors)
        for row in block:
            token = row["token"]
            cache = straight.load_cache(caches[token])
            if cache.log_name != scenes[token]["log_name"]:
                raise RuntimeError(f"Cache scene mismatch for {token}")
            observation, reason = straight.straight_observation(cache)
            if reason is not None:
                rejected[reason] += 1
                continue
            selected.append({"token": token, "session_id": row["session_id"], "cache_sha256": caches[token]["sha256"],
                             "pilot_session": row["session_id"] in sessions,
                             "leader_category": leader_category(observation), **observation})
        print(f"Screened {min(start + 48, len(rows))}/{len(rows)}. Eligible: {len(selected)}", file=sys.stderr, flush=True)
    if not selected:
        raise RuntimeError("No eligible straight scenes in the archived TRAIN inventory")
    return {"protocol": PROTOCOL, "anchor_identity_sha256": source_id, "screened_scenes": len(rows),
            "pilot_observations_sha256": straight.digest(pilot), "rejected": dict(rejected),
            "variants": {name: list(values) for name, values in VARIANTS.items()}, "scenes": selected}


def check_plan(plan):
    tokens = [row["token"] for row in plan["scenes"]]
    if (plan["protocol"] != PROTOCOL or plan["variants"] != {name: list(values) for name, values in VARIANTS.items()}
            or not tokens or len(tokens) != len(set(tokens))):
        raise RuntimeError("Invalid nightly observations or variant settings")
    return tokens


def predict_night(root, weights, load=straight.load_drex, plan=None, setups=VARIANTS):
    if plan is None:
        plan = straight.read(root / "observations.json")
    check_plan(plan)
    model = None
    for name, (prompt, observation, actions) in setups.items():
        folder = root / name
        folder.mkdir(exist_ok=True)
        binding = {"observation_sha256": straight.digest(plan), "variant": name, "setup": [prompt, observation, actions],
                   "model_path": str(weights)}
        identity = folder / "identity.json"
        if not identity.exists():
            straight.write(identity, binding)
        elif straight.read(identity) != binding:
            raise RuntimeError(f"Previous predictions have a different identity for {name}")
        for index, scene in enumerate(plan["scenes"], 1):
            token = scene["token"]
            destination = folder / f"{token}.json"
            if destination.exists():
                continue
            if model is None:
                model, load_wall_s = load(weights)
                print(f"Loaded Drex Max in {load_wall_s:.1f} s", flush=True)
            try:
                case = straight.build_plan(scene, model, prompt, observation, actions)
            except (ValueError, RuntimeError) as exc:
                case = {"token": token, "error": f"{type(exc).__name__}: {exc}"}
            straight.write(destination, case)
            if index % 10 == 0 or index == len(plan["scenes"]):
                print(f"{name} {index}/{len(plan['scenes'])}", flush=True)


def compact_predictions(root, plan):
    tokens = check_plan(plan)
    plan_hash = straight.digest(plan)
    result, model_paths = {}, set()
    for name, setup in VARIANTS.items():
        folder = root / name
        identity = straight.read(folder / "identity.json")
        if (identity["observation_sha256"] != plan_hash or identity["setup"] != list(setup)
                or identity["variant"] != name):
            raise RuntimeError(f"Prediction identity changed for {name}")
        model_paths.add(identity["model_path"])
        for token in tokens:
            case = straight.read(folder / f"{token}.json")
            if token != case["token"]:
                raise RuntimeError(f"Unexpected token in {name}/{token}")
            output = {"error": case["error"]} if "error" in case else {"poses": case["poses"]}
            result.setdefault(token, {})[name] = {"token": token, **output}
    if len(model_paths) != 1:
        raise RuntimeError("Variants were generated with different model paths")
    return result


def archive_context(args, plan):
    from rollout_replay import bound_context

    tokens = check_plan(plan)
    archive, candidates, source_id = straight.source_context(args.package, args.source)
    sessions = {row["token"]: row["session_id"] for row in archive}
    if plan["anchor_identity_sha256"] != source_id or any(
            sessions.get(row["token"]) != row["session_id"] for row in plan["scenes"]):
        raise RuntimeError("The selected scenes differ from the archived TRAIN inventory")
    manifest, _, vocab, template, caches = bound_context(args.package, args.repo, [candidates[t][0] for t in tokens])
    return candidates, manifest, vocab, template, caches


def archive_evaluator(args, plan, inspect=None):
    import numpy as np
    import real_data

    candidates, manifest, vocab, template, caches = archive_context(args, plan)
    keys = ("pdm_score", *real_data.METRICS)

    def for_scene(record):
        token = record["token"]
        cache = caches[token]
        if cache["sha256"] != record["cache_sha256"]:
            raise RuntimeError(f"Metric cache changed for {token}")
        metric_cache = straight.load_cache(cache)
        observation, reason = straight.straight_observation(metric_cache)
        if reason is not None or observation != {k: record[k] for k in ("speed_mps", "leaders")}:
            raise RuntimeError(f"Current observation changed for {token}")
        if "corridor_objects" in record and corridor_objects(metric_cache) != record["corridor_objects"]:
            raise RuntimeError(f"Current corridor objects changed for {token}")
        if "ego_bounds" in record and ego_bounds(metric_cache) != record["ego_bounds"]:
            raise RuntimeError(f"Ego footprint changed for {token}")
        sample = real_data.load_inputs(args.package, token, manifest, vocab)
        anchor = next(row for row in candidates[token] if row["candidate_local"] == sample["baseline_local"])
        if list(keys) != anchor["metric_keys"]:
            raise RuntimeError(f"DriveSuprim metric columns changed for {token}")
        def replay(poses):
            if inspect is None:
                return straight.raw_score(cache, poses, template, keys)
            metrics, scene, scorer, states = straight.raw_score_details(cache, poses, template, keys)
            return {"metrics": metrics, **inspect(scene, scorer, states)}

        baseline = replay(vocab[anchor["candidate_vocab"]])
        baseline_metrics = baseline if inspect is None else baseline["metrics"]
        if not np.allclose(list(baseline_metrics.values()), anchor["expected_raw"], atol=1e-8, rtol=1e-7):
            raise RuntimeError(f"DriveSuprim score parity failed for {token}")

        def evaluate(prediction):
            result = replay(straight.checked_straight_poses(prediction))
            metrics = result if inspect is None else result["metrics"]
            if not all(np.isfinite(list(metrics.values()))):
                raise RuntimeError(f"Nonfinite scorer output for {token}")
            return result

        return baseline, evaluate

    return for_scene, (*real_data.SAFETY, "history_comfort")


def paired(rows, name, protected):
    if not rows:
        return {"scenes": 0, "sessions": 0}
    new = [row["scores"][name] for row in rows]
    result = {"scenes": len(rows), "sessions": len({row["session_id"] for row in rows}),
              "mean_pdm": mean(score["pdm_score"] for score in new)}
    for label in ("ds", "control"):
        old = [row["ds"] if label == "ds" else row["scores"]["control"] for row in rows]
        deltas = [{key: current[key] - previous[key] for key in current} for current, previous in zip(new, old)]
        by_session = defaultdict(list)
        for row, delta in zip(rows, deltas):
            by_session[row["session_id"]].append(delta["pdm_score"])
        pdm = [delta["pdm_score"] for delta in deltas]
        result[f"vs_{label}"] = {
            "mean_delta": {key: mean(delta[key] for delta in deltas) for key in deltas[0]},
            "mean_pdm_delta_equal_session": mean(mean(values) for values in by_session.values()),
            "wins": sum(value > EPSILON for value in pdm), "losses": sum(value < -EPSILON for value in pdm),
            "ties": sum(abs(value) <= EPSILON for value in pdm),
            "regressions": {key: sum(delta[key] < -EPSILON for delta in deltas) for key in protected},
        }
    return result


def score(args, payload):
    plan, predictions = payload["plan"], payload["predictions"]
    tokens = check_plan(plan)
    if set(tokens) != set(predictions) or any(set(cases) != set(VARIANTS) for cases in predictions.values()):
        raise RuntimeError("Prediction membership differs from the nightly plan")
    for_scene, protected = archive_evaluator(args, plan)
    rows = []
    for index, record in enumerate(plan["scenes"], 1):
        token = record["token"]
        ds, evaluate = for_scene(record)
        scores, errors = {}, {}
        for name, prediction in predictions[token].items():
            if "error" in prediction:
                errors[name] = prediction["error"]
            else:
                scores[name] = evaluate(prediction)
        rows.append({"token": token, "session_id": record["session_id"], "leader_category": record["leader_category"],
                     "pilot_session": record["pilot_session"], "ds": ds, "scores": scores, "errors": errors})
        if index % 10 == 0 or index == len(tokens):
            print(f"Scored {index}/{len(tokens)} scenes", file=sys.stderr, flush=True)
    common = [row for row in rows if set(row["scores"]) == set(VARIANTS)]
    other_sessions = [row for row in common if not row["pilot_session"]]
    categories = sorted({row["leader_category"] for row in common})
    variants = {
        name: {"all_common": paired(common, name, protected),
               "outside_pilot_sessions": paired(other_sessions, name, protected),
               "by_leader_category": {category: paired([row for row in common if row["leader_category"] == category],
                                                       name, protected) for category in categories},
               "prediction_failures": sum(name in row["errors"] for row in rows)}
        for name in VARIANTS
    }
    return {"protocol": PROTOCOL, "observation_sha256": straight.digest(plan), "screened_scenes": plan["screened_scenes"],
            "eligible_scenes": len(tokens), "predicted_trajectories": sum(len(row["scores"]) for row in rows),
            "common_scenes": len(common), "rejected": plan["rejected"], "variants": variants, "cases": rows}


def make_probes(root, plan, report):
    from .simulation import ACCELERATION, Action, Vehicle

    def replay(scene, actions):
        vehicle = Vehicle(0.0, scene["speed_mps"])
        poses = []
        for action in actions:
            for _ in range(straight.ACTION_STEPS):
                vehicle = vehicle.advance(ACCELERATION[Action(action)], straight.SPEED_CAP)
                poses.append([vehicle.front_m, 0.0, 0.0])
        return poses

    payload = {}
    for scene, scores in zip(plan["scenes"], report["cases"]):
        token = scene["token"]
        if token != scores["token"]:
            raise RuntimeError("Nightly score order changed")
        if "control" not in scores["scores"]:
            continue
        control = straight.read(root / "control" / f"{token}.json")
        commands = [row["action"] for row in control["decisions"]]
        poses = replay(scene, commands)
        if control["token"] != token or poses != control["poses"]:
            raise RuntimeError(f"Archived control commands did not reproduce for {token}")
        seen = {tuple(pose[0] for pose in poses)}
        probes = []
        for index, action in enumerate(commands):
            for alternative in ("proceed", "maintain", "slow", "stop"):
                if alternative == action:
                    continue
                edited = replay(scene, [*commands[:index], alternative, *commands[index + 1:]])
                fingerprint = tuple(pose[0] for pose in edited)
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    probes.append({"token": token, "decision_index": index, "before": action, "after": alternative,
                                   "poses": edited})
        payload[token] = {"control": scores["scores"]["control"], "probes": probes}
    return payload


def score_probes(args, payload):
    plan, targets = payload["plan"], payload["targets"]
    for_scene, protected = archive_evaluator(args, plan)
    rows, total = [], 0
    for record in plan["scenes"]:
        token = record["token"]
        if token not in targets:
            continue
        _, evaluate = for_scene(record)
        control = targets[token]["control"]
        probes = []
        for candidate in targets[token]["probes"]:
            metrics = evaluate(candidate)
            delta = {key: metrics[key] - control[key] for key in control}
            probes.append({"decision_index": candidate["decision_index"], "before": candidate["before"],
                           "after": candidate["after"], "pdm_score": metrics["pdm_score"], "delta": delta})
        beneficial = [probe for probe in probes if probe["delta"]["pdm_score"] > EPSILON
                      and all(probe["delta"][key] >= -EPSILON for key in protected)]
        best = max(beneficial, key=lambda probe: probe["pdm_score"], default=None)
        rows.append({"token": token, "session_id": record["session_id"], "leader_category": record["leader_category"],
                     "best_single_edit": best, "tested_edits": len(probes), "positive_edits": len(beneficial),
                     "edits_that_lower_protected_metrics": sum(
                         any(probe["delta"][key] < -EPSILON for key in protected) for probe in probes),
                     "probes": probes})
        total += len(probes)
        if len(rows) % 10 == 0 or len(rows) == len(targets):
            print(f"Scored {total} single-command edits in {len(rows)}/{len(targets)} scenes",
                  file=sys.stderr, flush=True)
    return {"protocol": PROTOCOL, "observation_sha256": straight.digest(plan), "scenes": len(rows),
            "tested_edits": total, "scenes_with_improvement": sum(row["best_single_edit"] is not None for row in rows),
            "mean_best_pdm_delta_vs_control": mean(row["best_single_edit"]["delta"]["pdm_score"]
                                                  if row["best_single_edit"] else 0.0 for row in rows) if rows else None,
            "cases": rows}


def collision_profiles(scene, original):
    from .simulation import ACCELERATION, Action, Vehicle

    commands = [row["action"] for row in original["decisions"]]
    if len(commands) != straight.POSES // straight.ACTION_STEPS:
        raise RuntimeError("Incomplete archived trajectory in the collision scene")
    archived = [ACCELERATION[Action(action)] for action in commands]

    def replay(accelerations):
        ego = Vehicle(0.0, scene["speed_mps"])
        poses = []
        for acceleration in accelerations:
            for _ in range(straight.ACTION_STEPS):
                ego = ego.advance(acceleration, straight.SPEED_CAP)
                poses.append([ego.front_m, 0.0, 0.0])
        return {"token": scene["token"], "poses": poses}

    baseline = replay(archived)
    if original["poses"] != baseline["poses"]:
        raise RuntimeError("Control commands no longer reproduce the archived poses")
    profiles = {"control": baseline, "maintain": replay([0.0] * len(archived))}
    for name, delay, action in (("slow_from_start", 0, Action.SLOW), ("stop_from_start", 0, Action.STOP),
                                ("stop_after_0_5s", 1, Action.STOP), ("stop_after_1s", 2, Action.STOP)):
        profiles[name] = replay(archived[:delay] + [ACCELERATION[action]] * (len(archived) - delay))
    return profiles


def actor_snapshot(actor, origin):
    x, y = straight.local_xy(actor.center.x, actor.center.y, origin)
    velocity = getattr(actor, "velocity", None)
    speed = hypot(velocity.x, velocity.y) if velocity is not None else None
    return {"type": actor.tracked_object_type.name, "longitudinal_m": float(x), "lateral_m": float(y),
            "speed_mps": speed if speed is not None and isfinite(speed) else None}


def initial_leader_filter(snapshot):
    if snapshot is None:
        return "absent_from_initial_tracks"
    if snapshot["type"] != "VEHICLE":
        return "non_vehicle"
    if not 0 < snapshot["longitudinal_m"] <= 65:
        return "outside_forward_range"
    if abs(snapshot["lateral_m"]) > 1.8:
        return "outside_lateral_band"
    return "inside_vehicle_position_filter"


def inspect_contacts(scene, scorer, states):
    from nuplan.planning.metrics.utils.collision_utils import CollisionType
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex, StateIndex

    index = 1  # Model trajectory, as in pdm_score_from_interpolated_trajectory.
    observation = scorer._observation
    origin = scene.ego_state.rear_axle
    current = {actor.track_token: actor for actor in scene.current_tracked_objects[0].tracked_objects}
    ignored = set(observation.collided_track_ids)
    step = scorer.proposal_sampling.interval_length
    fault_time = scorer.time_to_at_fault_collision(index)
    ttc_time = scorer.time_to_ttc_infraction(index)
    contacts = []
    for tick, state in enumerate(states):
        time = tick * step
        if time > fault_time + EPSILON:
            break
        ego_polygon = scorer._ego_polygons[index, tick]
        occupancy = observation[tick]
        area = scorer._ego_areas[index, tick]
        actors = {actor.track_token: actor for actor in observation.detections_tracks[tick].tracked_objects}
        for token in occupancy.intersects(ego_polygon):
            if observation.red_light_token in token or token in ignored:
                continue
            actor = observation.unique_objects[token]
            collision_type = get_collision_type(state, ego_polygon, actor, occupancy[token])
            at_fault = (collision_type in (CollisionType.ACTIVE_FRONT_COLLISION, CollisionType.STOPPED_TRACK_COLLISION)
                        or (collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
                            and (area[EgoAreaIndex.MULTIPLE_LANES] or area[EgoAreaIndex.NON_DRIVABLE_AREA])))
            ignored.add(token)
            initial = actor_snapshot(current[token], origin) if token in current else None
            at_contact = actor_snapshot(actors[token], origin)
            first_replay_time = next(i * step for i in range(tick + 1) if token in observation[i].token_to_idx)
            ego_x, ego_y = straight.local_xy(state[StateIndex.X], state[StateIndex.Y], origin)
            contacts.append({"track_token": token, "time_s": round(time, 3), "type": collision_type.name,
                             "at_fault": bool(at_fault), "initial_track": initial,
                             "first_in_replay_s": round(first_replay_time, 3),
                             "initial_leader_filter": initial_leader_filter(initial), "track_at_contact": at_contact,
                             "ego_at_contact": {"longitudinal_m": float(ego_x), "lateral_m": float(ego_y),
                                                "speed_mps": hypot(state[StateIndex.VELOCITY_X],
                                                                   state[StateIndex.VELOCITY_Y])}})
    findings = [contact["time_s"] for contact in contacts if contact["at_fault"]]
    confirmed = min(findings, default=float("inf"))
    if not isclose(confirmed, fault_time, rel_tol=0.0, abs_tol=EPSILON):
        raise RuntimeError("Contact trace does not reproduce the PDM scorer's first at-fault collision time")
    return {"at_fault_collision_time_s": float(fault_time) if isfinite(fault_time) else None,
            "ttc_infraction_time_s": float(ttc_time) if isfinite(ttc_time) else None,
            "contacts_through_first_at_fault": contacts}


def score_collision(args, payload):
    import numpy as np

    plan, reference = payload["plan"], payload["reference"]
    if straight.digest(plan) != payload["observation_sha256"] or reference["token"] != COLLISION_TOKEN:
        raise RuntimeError("Collision review does not match the archived nightly scene")
    record = next(row for row in plan["scenes"] if row["token"] == COLLISION_TOKEN)
    for_scene, _ = archive_evaluator(args, {**plan, "scenes": [record]}, inspect=inspect_contacts)
    ds, evaluate = for_scene(record)
    profiles = {name: {"planned_distance_m": item["poses"][-1][0], **evaluate(item)}
                for name, item in payload["profiles"].items()}
    for result, scores in ((ds, reference["ds"]), (profiles["control"], reference["scores"]["control"])):
        if any(not np.isclose(value, scores[key], atol=1e-8, rtol=1e-7) for key, value in result["metrics"].items()):
            raise RuntimeError("Collision review did not reproduce the archived PDM metrics")
    return {"protocol": "navsim-straight-collision-v1", "observation_sha256": payload["observation_sha256"],
            "token": COLLISION_TOKEN, "initial_drex_observation": {
                "speed_mps": record["speed_mps"], "leaders": record["leaders"]},
            "ds": ds, "profiles": profiles}


def corridor_objects(cache):
    origin = cache.ego_state.rear_axle
    vehicle = cache.ego_state.car_footprint.vehicle_parameters
    reach = straight.straight_reach(float(cache.ego_state.dynamic_car_state.speed))
    front = float(vehicle.rear_axle_to_center) + float(vehicle.length) / 2
    c, s = cos(origin.heading), sin(origin.heading)
    objects = []
    for actor in cache.current_tracked_objects[0].tracked_objects:
        x, y = map(float, straight.local_xy(actor.center.x, actor.center.y, origin))
        kind = actor.tracked_object_type.name
        if kind == "VEHICLE" and 0 < x <= 65 and abs(y) <= 1.8:
            continue
        heading = float((actor.box.center.heading - origin.heading + pi) % (2 * pi) - pi)
        length, width = float(actor.box.length), float(actor.box.width)
        velocity = getattr(actor, "velocity", None)
        vx, vy = (float(velocity.x), float(velocity.y)) if velocity is not None else (0.0, 0.0)
        vx, vy = c * vx + s * vy, -s * vx + c * vy
        if not all(isfinite(value) for value in (x, y, heading, length, width, vx, vy)) or min(length, width) <= 0:
            continue
        half_along = (abs(cos(heading)) * length + abs(sin(heading)) * width) / 2
        lateral_reach = (abs(sin(heading)) * length + abs(cos(heading)) * width + float(vehicle.width)) / 2
        x_end, y_end = x + straight.HORIZON * vx, y + straight.HORIZON * vy
        if max(x, x_end) + half_along < front or min(x, x_end) - half_along > front + reach:
            continue
        clearance = (0.0 if y * y_end <= 0 else min(abs(y), abs(y_end))) - lateral_reach
        if clearance > CORRIDOR_MARGIN_M:
            continue
        objects.append({"track_token": actor.track_token, "type": kind, "x_m": round(x, 4), "y_m": round(y, 4),
                        "heading_rad": round(heading, 4), "length_m": round(length, 4), "width_m": round(width, 4),
                        "vx_mps": round(vx, 4), "vy_mps": round(vy, 4), "lateral_reach_m": round(lateral_reach, 4),
                        "half_length_along_path_m": round(half_along, 4), "projected_lateral_overlap": bool(clearance <= 0)})
    return sorted(objects, key=lambda obj: (obj["x_m"], obj["track_token"]))


def ego_bounds(cache):
    vehicle = cache.ego_state.car_footprint.vehicle_parameters
    center, half_length = float(vehicle.rear_axle_to_center), float(vehicle.length) / 2
    return {"front_m": center + half_length, "rear_m": center - half_length}


def prepare_corridor(args, plan, include_bounds=False):
    _, _, _, _, caches = archive_context(args, plan)
    result, bounds = {}, {}
    for scene in plan["scenes"]:
        token = scene["token"]
        cache = caches[token]
        if cache["sha256"] != scene["cache_sha256"]:
            raise RuntimeError(f"Metric cache changed for {token}")
        metric_cache = straight.load_cache(cache)
        result[token] = corridor_objects(metric_cache)
        if include_bounds:
            bounds[token] = ego_bounds(metric_cache)
    export = {"observation_sha256": straight.digest(plan), "objects": result}
    return {**export, "ego_bounds": bounds} if include_bounds else export


def with_corridor(plan, exported):
    tokens = check_plan(plan)
    if exported["observation_sha256"] != straight.digest(plan) or set(exported["objects"]) != set(tokens):
        raise RuntimeError("Corridor input does not match the nightly observations")
    bounds = exported.get("ego_bounds")
    if bounds is not None and set(bounds) != set(tokens):
        raise RuntimeError("Ego footprints do not match the nightly observations")
    rows = []
    for record in plan["scenes"]:
        token = record["token"]
        row = {**record, "corridor_objects": exported["objects"][token]}
        if bounds is not None:
            row["ego_bounds"] = bounds[token]
        rows.append(row)
    return {**plan, "scenes": rows}


def score_corridor(args, payload, evaluator=None):
    import numpy as np

    original, reference = payload["plan"], payload["reference"]
    variant = payload.get("variant", CORRIDOR_VARIANT)
    if variant != CORRIDOR_VARIANT and variant not in PREVIEW_SETUPS:
        raise RuntimeError("Unknown corridor experiment variant")
    plan = with_corridor(original, payload["export"])
    tokens = check_plan(plan)
    cases = {row["token"]: row for row in reference["cases"]}
    predictions = payload["predictions"]
    if (reference["observation_sha256"] != straight.digest(original) or set(cases) != set(tokens)
            or set(predictions) != set(tokens) or payload["prediction_plan_sha256"] != straight.digest(plan)):
        raise RuntimeError("Corridor predictions, observations and references differ")
    for_scene, protected = archive_evaluator(args, plan) if evaluator is None else evaluator
    rows, failures = [], []
    for record in plan["scenes"]:
        token = record["token"]
        previous, prediction = cases[token], predictions[token]
        if prediction["token"] != token:
            raise RuntimeError(f"Unexpected prediction token for {token}")
        if "error" in prediction:
            failures.append({"token": token, "error": prediction["error"]})
            continue
        ds, evaluate = for_scene(record)
        if any(not np.isclose(val, previous["ds"][key], atol=1e-8, rtol=1e-7) for key, val in ds.items()):
            raise RuntimeError(f"Archived DriveSuprim metrics did not reproduce for {token}")
        objects = record["corridor_objects"]
        category = ("projected_overlap" if any(obj["projected_lateral_overlap"] for obj in objects)
                    else "near_clear" if objects else "no_extra_objects")
        rows.append({"token": token, "session_id": record["session_id"], "leader_category": record["leader_category"],
                     "object_category": category, "objects": objects, "pilot_session": record["pilot_session"],
                     "ds": ds, "scores": {"control": previous["scores"]["control"], variant: evaluate(prediction)},
                     "actions": prediction["actions"], "planned_distance_m": prediction["poses"][-1][0],
                     **({"action_margins": prediction["action_margins"]} if "action_margins" in prediction else {})})
    collision_case = next((row for row in rows if row["token"] == COLLISION_TOKEN), None)
    target = payload.get("collision_track_token")
    target_covered = (any(obj["track_token"] == target for scene in plan["scenes"] if scene["token"] == COLLISION_TOKEN
                          for obj in scene["corridor_objects"])
                      if target is not None else None)
    return {"protocol": "navsim-straight-corridor-v1", "observation_sha256": straight.digest(original),
            "predicted_scenes": len(rows), "prediction_failures": failures,
            "all": paired(rows, variant, protected),
            "by_object_category": {
                category: paired([row for row in rows if row["object_category"] == category], variant, protected)
                for category in ("projected_overlap", "near_clear", "no_extra_objects")},
            "collision_track_token": target, "collision_track_in_input": target_covered,
            "collision_case": collision_case, "cases": rows}


def score_previews(args, payload):
    started = perf_counter()
    predictions = payload["predictions_by_variant"]
    if set(predictions) != set(PREVIEW_SETUPS):
        raise RuntimeError("Expected predictions for both corridor preview variants")
    plan = with_corridor(payload["plan"], payload["export"])
    for_scene, protected = archive_evaluator(args, plan)
    records = {record["token"]: record for record in plan["scenes"]}

    @lru_cache(maxsize=None)
    def scene_context(token):
        return for_scene(records[token])

    @lru_cache(maxsize=None)
    def score_poses(token, poses):
        return scene_context(token)[1]({"token": token, "poses": poses})

    def paired_evaluator(record):
        token = record["token"]
        ds, _ = scene_context(token)

        def evaluate(prediction):
            poses = tuple(tuple(pose) for pose in prediction["poses"])
            return score_poses(token, poses)

        return ds, evaluate

    variants = {name: score_corridor(args, {**payload, "variant": name, "predictions": predictions[name]},
                                     evaluator=(paired_evaluator, protected)) for name in PREVIEW_SETUPS}
    counted = score_poses.cache_info()
    return {"variants": variants, "scoring": {"ds_replays": scene_context.cache_info().misses,
                                               "unique_drex_replays": counted.misses, "reused_drex_scores": counted.hits,
                                               "wall_seconds": perf_counter() - started}}


def read_corridor_predictions(root, plan, variant, setup):
    folder = root / variant
    identity = straight.read(folder / "identity.json")
    control = straight.read(root / "control" / "identity.json")
    if (identity["observation_sha256"] != straight.digest(plan) or identity["variant"] != variant
            or identity["setup"] != list(setup) or identity["model_path"] != control["model_path"]):
        raise RuntimeError("Corridor predictions do not match the frozen experiment")
    predictions = {}
    for token in check_plan(plan):
        case = straight.read(folder / f"{token}.json")
        if "error" in case:
            predictions[token] = case
            continue
        decisions = case["decisions"]
        data = {"token": case["token"], "poses": case["poses"], "actions": [row["action"] for row in decisions]}
        if variant in PREVIEW_SETUPS:
            data["action_margins"] = [
                {"time_s": row["time_s"], "chosen_action": row["action"], "forecast": {
                    action: {key: preview[key] for key in ("near_path_clearance_with_margin_m", "near_path_overlap_first_s")}
                    for action, preview in row["request"]["state"]["action_effects"]["forecast"].items()}}
                for row in decisions
            ]
        predictions[token] = data
    return identity["observation_sha256"], predictions


def compare_preview_with_corridor(result, previous, variant):
    baseline = {row["token"]: row for row in previous["cases"]}
    matched = []
    for row in result["cases"]:
        old = baseline.get(row["token"])
        if old is None:
            continue
        if row["objects"] != old["objects"] or row["session_id"] != old["session_id"]:
            raise RuntimeError(f"Archived corridor objects changed for {row['token']}")
        matched.append({**row, "scores": {**row["scores"], "control": old["scores"][CORRIDOR_VARIANT]}})
    protected = result["all"].get("vs_control", {}).get("regressions", ())
    combined = paired(matched, variant, protected)
    return {"scenes": len(matched), **combined.get("vs_control", {})}


RISK_METRICS = ("no_at_fault_collisions", "time_to_collision_within_bound")
REGRESSION_SETUPS = {"control": ("baseline", "action-effects", "standard"),
                     CORRIDOR_VARIANT: ("baseline", CORRIDOR_VARIANT, "standard"), **PREVIEW_SETUPS}


def regression_cases(plan, suite, corridor, preview):
    tokens = set(check_plan(plan))
    variants = preview["variants"]
    if set(variants) != set(PREVIEW_SETUPS):
        raise RuntimeError("The preview suite is missing a variant")
    index = {}
    for name, report in (("control", suite), (CORRIDOR_VARIANT, corridor), *variants.items()):
        rows = {row["token"]: row for row in report["cases"]}
        if (report["observation_sha256"] != straight.digest(plan)
                or set(rows) != tokens or len(report["cases"]) != len(tokens)):
            raise RuntimeError(f"Regression review requires the same completed cohort for {name}")
        index[name] = rows
    selected = {}
    for scene in plan["scenes"]:
        token = scene["token"]
        control = index["control"][token]["scores"]["control"]
        corridor_scores = index[CORRIDOR_VARIANT][token]["scores"][CORRIDOR_VARIANT]
        regressions = {name: [key for key in RISK_METRICS
                              if any(index[name][token]["scores"][name][key] < base[key] - EPSILON
                                     for base in (control, corridor_scores))]
                       for name in PREVIEW_SETUPS}
        if token == COLLISION_TOKEN or any(regressions.values()):
            selected[token] = {"regressions": regressions, "positive_control": token == COLLISION_TOKEN,
                               "ds": index["control"][token]["ds"],
                               "scores": {name: index[name][token]["scores"][name] for name in REGRESSION_SETUPS}}
    return selected


def regression_trajectories(root, plan, corridor_plan, preview_plan, selected):
    plans = {"control": plan, CORRIDOR_VARIANT: corridor_plan, **dict.fromkeys(PREVIEW_SETUPS, preview_plan)}
    model_path = straight.read(root / "control" / "identity.json")["model_path"]
    result = {token: {} for token in selected}
    for name, setup in REGRESSION_SETUPS.items():
        folder = root / name
        identity = straight.read(folder / "identity.json")
        if identity != {"observation_sha256": straight.digest(plans[name]), "variant": name,
                        "setup": list(setup), "model_path": model_path}:
            raise RuntimeError(f"Prediction identity changed for {name}")
        for token in selected:
            case = straight.read(folder / f"{token}.json")
            decisions = case["decisions"]
            if (case["token"] != token or [row["time_s"] for row in decisions]
                    != [i * straight.STEP for i in range(0, straight.POSES, straight.ACTION_STEPS)]):
                raise RuntimeError(f"Invalid archived decision trace for {name}/{token}")
            result[token][name] = {"token": token, "poses": case["poses"], "decisions": [
                {"time_s": row["time_s"], "action": row["action"], "lead_radar": row["request"]["state"]["lead_radar"],
                 "chosen_forecast": row["request"]["state"]["action_effects"]["forecast"][row["action"]]}
                for row in decisions]}
    return result


def inspect_regression(scene, scorer, states):
    from shapely.affinity import translate
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

    result = inspect_contacts(scene, scorer, states)
    initial, reason = straight.straight_observation(scene, include_leader_tokens=True)
    if reason is not None:
        raise RuntimeError(f"Regression scene failed its original observation filter: {reason}")
    result["initial_leader_track_tokens"] = initial["leader_track_tokens"]
    candidates = []
    time_s = result["ttc_infraction_time_s"]
    if time_s is not None:
        step = scorer.proposal_sampling.interval_length
        tick = round(time_s / step)
        state = states[tick]
        speed = hypot(state[StateIndex.VELOCITY_X], state[StateIndex.VELOCITY_Y])
        heading = state[StateIndex.HEADING]
        obs = scorer._observation
        for offset in range(0, int(scorer._config.future_collision_horizon_window * 10), 3):
            dt = offset * step
            polygon = translate(scorer._ego_polygons[1, tick], xoff=cos(heading) * speed * dt,
                                yoff=sin(heading) * speed * dt)
            frame = obs[tick + offset]
            actors = {actor.track_token: actor for actor in obs.detections_tracks[tick + offset].tracked_objects}
            for token in frame.intersects(polygon):
                if obs.red_light_token not in token and token not in obs.collided_track_ids:
                    candidates.append({"track_token": token, "projection_s": round(dt, 3),
                                       "track_at_projection": actor_snapshot(actors[token], scene.ego_state.rear_axle)})
    result["ttc_geometric_candidates"] = candidates
    return result


def score_regressions(args, payload):
    original = payload["plan"]
    plan = with_corridor(original, payload["export"])
    selected, predictions = payload["selected"], payload["predictions"]
    if straight.digest(original) != payload["observation_sha256"] or set(selected) != set(predictions):
        raise RuntimeError("Regression replay does not match the frozen observations")
    records = [record for record in plan["scenes"] if record["token"] in selected]
    if {record["token"] for record in records} != set(selected):
        raise RuntimeError("A regression case is missing from the frozen cohort")
    for_scene, _ = archive_evaluator(args, {**plan, "scenes": records}, inspect=inspect_regression)
    rows = []
    for record in records:
        token = record["token"]
        reference, trajectories = selected[token], predictions[token]
        if set(trajectories) != set(REGRESSION_SETUPS):
            raise RuntimeError(f"Missing archived trajectory for {token}")
        ds, evaluate = for_scene(record)
        results, pose_cache = {"ds": ds}, {}
        lateral = {actor["track_token"] for actor in record["corridor_objects"]}
        for name, trajectory in trajectories.items():
            if trajectory["token"] != token:
                raise RuntimeError(f"Trajectory token differs from its regression case: {token}/{name}")
            key = tuple(tuple(pose) for pose in trajectory["poses"])
            if key not in pose_cache:
                pose_cache[key] = evaluate(trajectory)
            results[name] = {**pose_cache[key], "decisions": trajectory["decisions"]}
        for name, result in results.items():
            expected = reference["ds"] if name == "ds" else reference["scores"][name]
            if set(expected) != set(result["metrics"]) or any(
                    not isclose(val, expected[key], rel_tol=1e-7, abs_tol=1e-8)
                    for key, val in result["metrics"].items()):
                raise RuntimeError(f"Archived raw PDM did not reproduce for {token}/{name}")
            leaders = result["initial_leader_track_tokens"]
            for field in ("contacts_through_first_at_fault", "ttc_geometric_candidates"):
                result[field] = [{**event, "in_initial_leader_radar": event["track_token"] in leaders,
                                  "in_corridor_object_export": event["track_token"] in lateral,
                                  "corridor_object_field_used": name in (*PREVIEW_SETUPS, CORRIDOR_VARIANT)}
                                 for event in result[field]]
        rows.append({"token": token, "session_id": record["session_id"], "regressions": reference["regressions"],
                     "positive_control": reference["positive_control"], "profiles": results})
    return {"protocol": "navsim-straight-regression-replay-v1", "observation_sha256": straight.digest(original),
            "scenes": len(rows), "cases": rows}


def on_astra(stage, args, payload):
    straight_code = Path(straight.__file__).read_text(encoding="utf-8")
    bundled_straight = b64encode(compress(straight_code.encode())).decode()
    bundled_night = b64encode(compress(Path(__file__).read_bytes())).decode()
    launcher = ("import sys, types\nfrom base64 import b64decode\nfrom zlib import decompress\n"
                "module = types.ModuleType('_drex_night_straight')\n"
                "sys.modules[module.__name__] = module\n"
                f"exec(compile(decompress(b64decode({bundled_straight!r})), 'navsim_straight.py', 'exec'), module.__dict__)\n"
                f"exec(compile(decompress(b64decode({bundled_night!r})), 'navsim_night.py', 'exec'))\n")
    return straight.on_astra(stage, args, payload, source_code=launcher)


def file_sha256(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_brake_gap_artifact(path, data):
    # Exclusive creation does not open an existing file, hard link or symlink for writing.
    encoded = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def tree_sha256(root):
    """Hash content and relative paths, not timestamps or absolute installation paths."""
    files = {path.relative_to(root).as_posix(): file_sha256(path) for path in sorted(root.rglob("*"))
             if path.is_file() and not {".git", "__pycache__"}.intersection(path.relative_to(root).parts)}
    if not files:
        raise RuntimeError(f"Empty identity tree: {root}")
    return straight.digest(files)


def request_json(request):
    return json.dumps(request, ensure_ascii=False, allow_nan=False)


def without_brake_gap(request):
    request = json.loads(request_json(request))
    for effect in request["state"]["action_effects"]["forecast"].values():
        del effect[BRAKE_GAP_FIELD]
    return request


def check_brake_gap_pose_rows(prediction):
    """Validate the JSON rollout on MSU270 without the scorer's array dependency."""
    poses = prediction.get("poses")
    error = f"Invalid straight trajectory for {prediction['token']}"
    if not isinstance(poses, list) or len(poses) != straight.POSES:
        raise ValueError(error)
    limit = straight.SPEED_CAP * straight.HORIZON + EPSILON
    previous_x = 0.0
    for pose in poses:
        if not isinstance(pose, list) or len(pose) != 3 or any(type(value) not in (int, float) for value in pose):
            raise ValueError(error)
        x, y, heading = pose
        # These bounds also reject NaN and infinities in every coordinate.
        if not previous_x <= x <= limit or y != 0 or heading != 0:
            raise ValueError(error)
        previous_x = x


def check_brake_gap_trace(scene, archived):
    token = scene["token"]
    check_brake_gap_pose_rows(archived)
    if (archived["token"] != token or [row["time_s"] for row in archived["decisions"]]
            != [step / 10 for step in range(0, straight.POSES, straight.ACTION_STEPS)]):
        raise RuntimeError(f"Invalid archived trace: {token}")
    results = []
    for enabled in (False, True):
        actions = iter(row["action"] for row in archived["decisions"])
        result = straight.build_plan(scene, lambda _: (next(actions), None), *PREVIEW_SETUPS[BRAKE_GAP_CONTROL],
                                     lead_brake_gap=enabled)
        if result["poses"] != archived["poses"]:
            raise RuntimeError(f"Archive pose parity failed: {token}")
        for old, new in zip(archived["decisions"], result["decisions"]):
            request = without_brake_gap(new["request"]) if enabled else new["request"]
            if request_json(old["request"]) != request_json(request):
                raise RuntimeError(f"Archive request parity failed: {token} t={old['time_s']}")
        results.append(result)
    return results


def brake_gap_archive(root, *, verify_traces=False):
    plan_path, export_path, report_path = (root / name for name in (
        "observations.json", "corridor-preview-input.json", "corridor-preview.json"))
    report_hash = file_sha256(report_path)
    original, export, report = map(straight.read, (plan_path, export_path, report_path))
    with_objects = with_corridor(original, export)
    tokens = check_plan(with_objects)
    sessions = {row["session_id"] for row in with_objects["scenes"]}
    reference = report["variants"][BRAKE_GAP_CONTROL]
    reference_rows = reference["cases"]
    if (report_hash != BRAKE_GAP_REPORT_SHA256 or straight.digest(original) != BRAKE_GAP_PLAN_SHA256
            or report["protocol"] != "navsim-straight-corridor-preview-v1"
            or report["observation_sha256"] != BRAKE_GAP_PLAN_SHA256
            or reference["observation_sha256"] != BRAKE_GAP_PLAN_SHA256 or reference["prediction_failures"]
            or len(tokens) != 23 or len(sessions) != 14 or not {BRAKE_GAP_TARGET, COLLISION_TOKEN} <= set(tokens)
            or len(reference_rows) != len(tokens) or {row["token"] for row in reference_rows} != set(tokens)
            or "ego_bounds" not in export):
        raise RuntimeError("Expected the complete frozen corridor-preview A from Slurm 298185")
    folder = root / BRAKE_GAP_CONTROL
    identity = straight.read(folder / "identity.json")
    if (identity["observation_sha256"] != straight.digest(with_objects) or identity["variant"] != BRAKE_GAP_CONTROL
            or identity["setup"] != list(PREVIEW_SETUPS[BRAKE_GAP_CONTROL])
            or identity["model_path"] != straight.read(root / "control" / "identity.json")["model_path"]):
        raise RuntimeError("Historical preview prediction binding changed")
    reference_by_token = {row["token"]: row for row in reference_rows}
    archived = {}
    for scene in with_objects["scenes"]:
        token = scene["token"]
        case = straight.read(folder / f"{token}.json")
        row = reference_by_token[token]
        if (case["token"] != token or row["session_id"] != scene["session_id"]
                or row["objects"] != scene["corridor_objects"]
                or row["actions"] != [decision["action"] for decision in case["decisions"]]
                or row["planned_distance_m"] != case["poses"][-1][0]):
            raise RuntimeError(f"Historical preview row binding changed: {token}")
        if verify_traces:
            check_brake_gap_trace(scene, case)
        archived[token] = case
    binding = {"observation_sha256": straight.digest(original), "prediction_plan_sha256": straight.digest(with_objects),
               "report_sha256": report_hash, "prediction_sha256": {t: straight.digest(archived[t]) for t in tokens},
               "model_path": identity["model_path"]}
    return binding, with_objects, {"plan": original, "export": export, "reference": reference_by_token,
                                   "archived": archived}


def no_brake_gap_truncation(model, request):
    def tokens(limit):
        inputs = model.prepare([request], max_length=limit).inputs
        ids = inputs["input_ids"].tolist()
        mask = inputs["attention_mask"].tolist()
        if len(ids) != 1 or len(mask) != 1 or len(ids[0]) != len(mask[0]):
            raise RuntimeError("Unexpected Drex tokenizer batch shape")
        return [item for item, keep in zip(ids[0], mask[0]) if keep]

    normal = tokens(2048)
    if not normal or len(normal) >= 2048 or normal != tokens(4096):
        raise RuntimeError("Drex request truncation or tokenizer parity failure")
    return normal


def check_brake_gap_token_pair(model, augmented):
    normal = no_brake_gap_truncation(model, without_brake_gap(augmented))
    with_gap = no_brake_gap_truncation(model, augmented)
    if normal == with_gap:
        raise RuntimeError("Drex preprocessor erased the lead-brake gap field")


def check_brake_gap_metrics(actual, expected, token, arm):
    needed = {"pdm_score", "ego_progress", *BRAKE_GAP_PROTECTED}
    if (set(actual) != needed or set(expected) != needed
            or any(not isfinite(actual[k]) or not isfinite(expected[k])
                   or not isclose(actual[k], expected[k], rel_tol=1e-7, abs_tol=EPSILON) for k in needed)):
        raise RuntimeError(f"Historical {arm} raw-score parity failed: {token}")


def brake_gap_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["session_id"]].append(row["delta"]["pdm_score"])
    sessions = [groups[k] for k in sorted(groups)]
    totals = [(sum(values), len(values)) for values in sessions]
    random = Random(20261124)
    bootstrap = []
    for _ in range(20000):
        selected = random.choices(totals, k=len(totals))
        bootstrap.append(100 * sum(total for total, _ in selected) / sum(n for _, n in selected))
    bootstrap.sort()
    deltas = [row["delta"]["pdm_score"] for row in rows]
    return {"mean_delta_pp": {k: 100 * mean(row["delta"][k] for row in rows) for k in rows[0]["delta"]},
            "mean_distance_delta_m": mean(row["distance_delta_m"] for row in rows),
            "pdm_by_session_pp": {k: 100 * mean(v) for k, v in sorted(groups.items())},
            "pdm_equal_session_pp": 100 * mean(mean(values) for values in sessions),
            "pdm_leave_one_session_out_pp": {k: 100 * mean(d for other, ds in groups.items() if k != other for d in ds)
                                             for k in sorted(groups)} if len(groups) > 1 else {},
            "pdm_bootstrap_95_pp": [bootstrap[500], bootstrap[19499]],
            "bootstrap_seed": 20261124, "bootstrap_replicates": 20000,
            "wins": sum(d > EPSILON for d in deltas), "losses": sum(d < -EPSILON for d in deltas),
            "ties": sum(abs(d) <= EPSILON for d in deltas)}


def score_brake_gap(args, payload):
    """Run baseline or paired scoring in the existing Astra container; return partials on first failed gate."""
    from rollout_replay import config_core

    plan = with_corridor(payload["plan"], payload["export"])
    staged = payload.get("predictions")
    report = {"experiment_id": BRAKE_GAP_EXPERIMENT, "split": "archived TRAIN; diagnostic",
              "planned_scenes": len(plan["scenes"]), "cases": [], "status": "incomplete",
              "stage": "baseline" if staged is None else "comparison"}
    rows = report["cases"]
    pending = [record["token"] for record in plan["scenes"]]
    token = None
    try:
        _, _, _, template, _ = archive_context(args, plan)
        scorer = {"config_sha256": straight.digest(config_core(template)),
                  "navsim_sha256": tree_sha256(args.repo / "navsim"),
                  "tools_sha256": tree_sha256(args.source / "tools"),
                  "python": platform.python_version(),
                  "versions": {name: version(name) for name in ("numpy", "hydra-core", "shapely", "nuplan-devkit")}}
        report["scorer"] = scorer
        if staged is not None and (payload["expected_scorer"] != scorer or set(staged) != set(pending)):
            raise RuntimeError("Scorer identity or full prediction membership changed after preflight")
        for_scene, _ = archive_evaluator(args, plan)
        records = sorted(plan["scenes"], key=lambda r: (r["token"] != BRAKE_GAP_TARGET, r["token"] != COLLISION_TOKEN))
        for scene in records:
            token = scene["token"]
            archived, reference = payload["archived"][token], payload["reference"][token]
            ds, evaluate = for_scene(scene)
            actual_a = evaluate(archived)
            check_brake_gap_metrics(ds, reference["ds"], token, "DrSu")
            check_brake_gap_metrics(actual_a, reference["scores"][BRAKE_GAP_CONTROL], token, "Drex A")
            row = {"token": token, "session_id": scene["session_id"], "ds": ds, "a": actual_a}
            if staged is not None:
                candidate = staged[token]
                if candidate["token"] != token:
                    raise RuntimeError(f"Incorrect prediction token: {token}")
                b = evaluate(candidate) if candidate["poses"] != archived["poses"] else actual_a
                check_brake_gap_metrics(b, b, token, "Drex B schema")
                delta = {key: b[key] - actual_a[key] for key in actual_a}
                row.update(b=b, delta=delta, distance_delta_m=candidate["poses"][-1][0] - archived["poses"][-1][0],
                           regressions=[key for key in BRAKE_GAP_PROTECTED if delta[key] < -EPSILON])
            rows.append(row)
            pending.remove(token)
            if staged is not None and (row["regressions"] or (
                    token in (BRAKE_GAP_TARGET, COLLISION_TOKEN) and any(b[k] != 1.0 for k in RISK_METRICS))):
                report.update(status="rejected", first_failed_gate={"token": token, "reason": "protected/target gate"})
                break
        else:
            if staged is None:
                report["status"] = "baseline_verified"
            else:
                report["summary"] = brake_gap_summary(rows)
                report["status"] = "pass" if report["summary"]["mean_delta_pp"]["pdm_score"] > 0 else "rejected"
                if report["status"] == "rejected":
                    report["first_failed_gate"] = {"reason": "mean paired PDM delta not positive"}
    except (KeyError, ValueError, RuntimeError, OSError) as exc:
        report["first_failed_gate"] = {"token": token, "reason": f"{type(exc).__name__}: {exc}"}
    report["unscored_tokens"] = pending
    return report


def brake_gap_gpu_binding(args, historical_path):
    if args.weights is None or args.upstream is None or args.provenance is None:
        raise RuntimeError("GPU stages require weights, upstream and independently archived runtime provenance")
    expected = straight.read(args.provenance)
    if (expected["slurm_job_id"] != 298185 or not expected["evidence"]
            or Path(historical_path).resolve() != args.weights.resolve()):
        raise RuntimeError("Provenance must bind the original job and weight path")
    runtime = {"weights_sha256": tree_sha256(args.weights), "upstream_sha256": tree_sha256(args.upstream),
               "python": platform.python_version(),
               "versions": {name: version(name) for name in ("torch", "transformers", "tokenizers")}}
    if expected["runtime"] != runtime:
        raise RuntimeError("Current Drex weights/runtime do not match verified archive provenance")
    return {"runtime": runtime, "provenance_sha256": file_sha256(args.provenance)}


def brake_gap_receipt(chain, smoke, cases):
    return {"status": "predicted", **chain, "smoke_sha256": straight.digest(smoke),
            "runtime": smoke["runtime"], "provenance_sha256": smoke["provenance_sha256"],
            "prediction_sha256": {t: straight.digest(case) for t, case in cases.items()}}


def check_brake_gap_receipt(receipt, chain, smoke, cases, tokens):
    if (smoke.get("status") != "passed" or any(smoke.get(k) != v for k, v in chain.items())
            or set(cases) != tokens or receipt != brake_gap_receipt(chain, smoke, cases)):
        raise RuntimeError("Incomplete predictions or prediction receipt does not belong to this verified attempt")


def run_brake_gap(args):
    if args.archive is None:
        raise RuntimeError("--archive is required")
    root, archive = args.out.resolve(), args.archive.resolve(strict=True)
    if root != args.out.absolute() or root == archive or archive in root.parents or root in archive.parents:
        raise RuntimeError("Use a canonical, non-symlink output directory separate from the read-only archive")
    stage = args.stage.removeprefix("brake-gap-")
    preflight = stage.startswith("preflight-")
    if preflight:
        stage = stage.removeprefix("preflight-")
    binding, plan, payload = brake_gap_archive(archive, verify_traces=stage == "verify")
    revision = {"git_head": args.git_head, "git_diff_sha256": args.git_diff_sha256,
                "source_sha256": {p.name: file_sha256(p) for p in (Path(__file__), Path(straight.__file__),
                                  Path(straight.__file__).with_name("driver.py"),
                                  Path(straight.__file__).with_name("simulation.py"))}}
    manifest = {"experiment_id": BRAKE_GAP_EXPERIMENT, "attempt": str(root), "baseline_job": 298185,
                "archive": str(archive),
                "binding": binding, "revision": revision, "lead_brake_mps2": straight.PREVIEW_LEAD_BRAKE_MPS2,
                "inference": "argmax, no sampling; max_length=2048", "bootstrap_seed": 20261124}
    manifest_path = root / "manifest.json"
    if stage == "verify":
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        write_brake_gap_artifact(manifest_path, manifest)
        print(f"Verified 23 archived decision traces: {manifest_path}", flush=True)
        return 0
    if straight.read(manifest_path) != manifest:
        raise RuntimeError("Current code or archived inputs differ from the verified manifest")
    manifest_sha = straight.digest(manifest)
    baseline = root / "baseline-score.json"
    if stage == "baseline-score":
        if baseline.exists():
            raise RuntimeError("Baseline score already attempted; do not overwrite")
        result = on_astra("astra-night-brake-gap-score", args, {
            **payload, "archived": {t: {"token": case["token"], "poses": case["poses"]}
                                   for t, case in payload["archived"].items()}})
        write_brake_gap_artifact(baseline, {**result, "manifest_sha256": manifest_sha})
        print(f"{result['status']}: {baseline}", flush=True)
        return 0 if result["status"] == "baseline_verified" else 2
    baseline_result = straight.read(baseline)
    if baseline_result["status"] != "baseline_verified" or baseline_result.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("Astra baseline parity has not passed for this manifest")
    chain = {"attempt": str(root), "manifest_sha256": manifest_sha, "baseline_sha256": straight.digest(baseline_result)}
    smoke = root / "gpu-smoke.json"
    predictions = root / "predictions"
    receipt = root / "prediction-receipt.json"
    if stage in ("gpu-smoke", "predict"):
        binding_now = {"status": "passed", **chain, **brake_gap_gpu_binding(args, binding["model_path"])}
        if stage == "gpu-smoke":
            if smoke.exists() or (root / "gpu-smoke-a.json").exists():
                raise RuntimeError("GPU smoke already attempted; do not overwrite")
        elif straight.read(smoke) != binding_now or predictions.exists() or receipt.exists():
            raise RuntimeError("Same-runtime GPU smoke required; predictions must not exist")
        if preflight:
            print(f"GPU prerequisites verified without model load: {stage}", flush=True)
            return 0

        from .driver import DrexMax

        model = DrexMax(args.weights)

        def choose_a(request):
            no_brake_gap_truncation(model.model, request)
            return model.choose(request)

        def choose_b(request):
            check_brake_gap_token_pair(model.model, request)
            return model.choose(request)

        setup = PREVIEW_SETUPS[BRAKE_GAP_CONTROL]
        if stage == "gpu-smoke":
            scene = next(row for row in plan["scenes"] if row["token"] == BRAKE_GAP_TARGET)
            archived = payload["archived"][BRAKE_GAP_TARGET]
            _, augmented = check_brake_gap_trace(scene, archived)
            for step in augmented["decisions"]:
                check_brake_gap_token_pair(model.model, step["request"])
            actual = straight.build_plan(scene, choose_a, *setup)
            write_brake_gap_artifact(root / "gpu-smoke-a.json", actual)
            if actual["poses"] != archived["poses"] or any(
                    a["action"] != b["action"] or request_json(a["request"]) != request_json(b["request"])
                    for a, b in zip(actual["decisions"], archived["decisions"])):
                raise RuntimeError("Live Drex A did not reproduce its archived actions, poses and requests")
            write_brake_gap_artifact(smoke, binding_now)
        else:
            predictions.mkdir()
            cases = {}
            for scene in plan["scenes"]:
                token = scene["token"]
                _, augmented = check_brake_gap_trace(scene, payload["archived"][token])
                for step in augmented["decisions"]:
                    check_brake_gap_token_pair(model.model, step["request"])
                case = straight.build_plan(scene, choose_b, *setup, lead_brake_gap=True)
                check_brake_gap_pose_rows(case)
                write_brake_gap_artifact(predictions / f"{token}.json", case)
                cases[token] = case
                print(f"Predicted {token}", flush=True)
            write_brake_gap_artifact(receipt, brake_gap_receipt(chain, binding_now, cases))
        return 0
    output = root / "comparison.json"
    if output.exists():
        raise RuntimeError("Comparison already attempted; do not overwrite")
    cases = {path.stem: straight.read(path) for path in predictions.glob("*.json")}
    prediction_receipt = straight.read(receipt)
    check_brake_gap_receipt(prediction_receipt, chain, straight.read(smoke), cases, set(binding["prediction_sha256"]))
    payload["predictions"] = {t: {"token": case["token"], "poses": case["poses"]} for t, case in cases.items()}
    payload["archived"] = {t: {"token": case["token"], "poses": case["poses"]}
                           for t, case in payload["archived"].items()}
    payload["expected_scorer"] = baseline_result["scorer"]
    result = on_astra("astra-night-brake-gap-score", args, payload)
    write_brake_gap_artifact(output, {**result, "prediction_receipt_sha256": straight.digest(prediction_receipt)})
    print(f"{result['status']}: {output}", flush=True)
    return {"pass": 0, "rejected": 3}.get(result["status"], 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict", "score", "probes", "collision", "regressions",
                                          "corridor-prepare", "corridor-predict", "corridor-score",
                                          "preview-prepare", "preview-predict", "preview-score",
                                          "astra-night-prepare", "astra-night-score", "astra-night-probes",
                                          "astra-night-collision", "astra-night-corridor-prepare",
                                          "astra-night-corridor-score", "astra-night-preview-prepare",
                                          "astra-night-preview-score", "astra-night-regressions",
                                          "astra-night-brake-gap-score", "brake-gap-verify", "brake-gap-baseline-score",
                                          "brake-gap-preflight-gpu-smoke", "brake-gap-preflight-predict",
                                          "brake-gap-gpu-smoke", "brake-gap-predict", "brake-gap-score"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--git-head")
    parser.add_argument("--git-diff-sha256")
    parser.add_argument("--limit", type=int, default=8)  # Common Astra transport argument.
    args = parser.parse_args()
    if args.stage.startswith("astra-night-"):
        payload = json.load(sys.stdin)
        with redirect_stdout(sys.stderr):
            remote = {"astra-night-prepare": prepare, "astra-night-score": score,
                      "astra-night-probes": score_probes, "astra-night-collision": score_collision,
                      "astra-night-corridor-prepare": prepare_corridor, "astra-night-corridor-score": score_corridor,
                      "astra-night-preview-score": score_previews, "astra-night-regressions": score_regressions,
                      "astra-night-brake-gap-score": score_brake_gap}
            if args.stage == "astra-night-preview-prepare":
                result = prepare_corridor(args, payload, include_bounds=True)
            else:
                result = remote[args.stage](args, payload)
        print(json.dumps(result, allow_nan=False))
        return 0

    if args.out is None:
        parser.error("--out is required")
    if args.stage.startswith("brake-gap-"):
        return run_brake_gap(args)
    root = args.out
    plan_path = root / "observations.json"
    is_preview = args.stage.startswith("preview-")
    corridor_input = root / ("corridor-preview-input.json" if is_preview else "corridor-input.json")
    if args.stage == "prepare":
        if args.pilot is None:
            parser.error("--pilot is required")
        pilot = straight.read(args.pilot)
        if plan_path.exists():
            plan = straight.read(plan_path)
            if plan["pilot_observations_sha256"] != straight.digest(pilot):
                raise RuntimeError("Saved nightly selection uses a different pilot")
        else:
            plan = on_astra("astra-night-prepare", args, pilot)
            straight.write(plan_path, plan)
        tokens = check_plan(plan)
        print(f"Eligible {len(tokens)}/{plan['screened_scenes']}; {len(VARIANTS)} variants; "
              f"{len(tokens) * len(VARIANTS)} planned trajectories", flush=True)
        return 0
    if args.stage in ("predict", "corridor-predict", "preview-predict"):
        if args.weights is None or not args.weights.is_dir():
            parser.error("--weights must point to the installed Drex Max model")
        if args.stage != "predict":
            plan = with_corridor(straight.read(plan_path), straight.read(corridor_input))
            setups = PREVIEW_SETUPS if is_preview else {CORRIDOR_VARIANT: ("baseline", CORRIDOR_VARIANT, "standard")}
            predict_night(root, args.weights, plan=plan, setups=setups)
        else:
            predict_night(root, args.weights)
        return 0
    plan = straight.read(plan_path)
    suite = root / "suite.json"
    if args.stage == "score":
        result = on_astra("astra-night-score", args, {"plan": plan, "predictions": compact_predictions(root, plan)})
        straight.write(suite, result)
        print(f"Scored {result['predicted_trajectories']} trajectories; {result['common_scenes']} matched scenes: {suite}")
        return 0
    report = straight.read(suite)
    if report["observation_sha256"] != straight.digest(plan):
        raise RuntimeError("The control scores belong to a different nightly cohort")
    if args.stage == "regressions":
        corridor = straight.read(root / "corridor.json")
        preview = straight.read(root / "corridor-preview.json")
        exported = straight.read(root / "corridor-preview-input.json")
        selected = regression_cases(plan, report, corridor, preview)
        trajectories = regression_trajectories(root, plan, with_corridor(plan, straight.read(root / "corridor-input.json")),
                                               with_corridor(plan, exported), selected)
        payload = {"plan": plan, "export": exported, "observation_sha256": report["observation_sha256"],
                   "selected": selected, "predictions": trajectories}
        result = on_astra("astra-night-regressions", args, payload)
        output = root / "corridor-regressions.json"
        straight.write(output, result)
        print(f"Archived regression replay on {result['scenes']} scenes: {output}")
        return 0
    if args.stage in ("corridor-prepare", "preview-prepare"):
        if not corridor_input.exists():
            stage = "astra-night-preview-prepare" if is_preview else "astra-night-corridor-prepare"
            straight.write(corridor_input, on_astra(stage, args, plan))
        exported = straight.read(corridor_input)
        with_corridor(plan, exported)
        print(f"Prepared corridor objects on {sum(bool(rows) for rows in exported['objects'].values())} scenes.")
        return 0
    if args.stage in ("corridor-score", "preview-score"):
        exported = straight.read(corridor_input)
        prediction_plan = with_corridor(plan, exported)
        collision = root / "collision.json"
        target = None
        if collision.exists():
            review = straight.read(collision)
            if review["observation_sha256"] != straight.digest(plan):
                raise RuntimeError("Collision review belongs to a different nightly cohort")
            target = next((event["track_token"] for event in review["profiles"]["control"]["contacts_through_first_at_fault"]
                           if event["at_fault"]), None)
        details = {"plan": plan, "reference": report, "export": exported, "collision_track_token": target}
        if is_preview:
            previous = straight.read(root / "corridor.json")
            if previous["observation_sha256"] != straight.digest(plan):
                raise RuntimeError("The preceding corridor test belongs to a different nightly cohort")
            predictions_by_variant = {}
            for name, setup in PREVIEW_SETUPS.items():
                _, predictions_by_variant[name] = read_corridor_predictions(root, prediction_plan, name, setup)
            payload = {**details, "prediction_plan_sha256": straight.digest(prediction_plan),
                       "predictions_by_variant": predictions_by_variant}
            batch = on_astra("astra-night-preview-score", args, payload)
            for name, result in batch["variants"].items():
                result["vs_corridor_objects"] = compare_preview_with_corridor(result, previous, name)
            output = root / "corridor-preview.json"
            straight.write(output, {"protocol": "navsim-straight-corridor-preview-v1",
                                    "position_margin_m": straight.CORRIDOR_POSITION_MARGIN_M,
                                    "observation_sha256": straight.digest(plan), **batch})
            print(f"Astra scoring: {batch['scoring']}")
            print(f"Corridor preview comparison: {output}")
            return 0
        plan_hash, predictions = read_corridor_predictions(
            root, prediction_plan, CORRIDOR_VARIANT, ("baseline", CORRIDOR_VARIANT, "standard"))
        result = on_astra("astra-night-corridor-score", args, {**details, "prediction_plan_sha256": plan_hash,
                                                            "predictions": predictions})
        output = root / "corridor.json"
        straight.write(output, result)
        print(f"Corridor comparison on {result['predicted_scenes']} scenes: {output}")
        return 0
    if args.stage == "collision":
        scene = next(row for row in plan["scenes"] if row["token"] == COLLISION_TOKEN)
        original = straight.read(root / "control" / f"{COLLISION_TOKEN}.json")
        reference = next(row for row in report["cases"] if row["token"] == COLLISION_TOKEN)
        payload = {"plan": plan, "observation_sha256": report["observation_sha256"],
                   "reference": reference, "profiles": collision_profiles(scene, original)}
        result = on_astra("astra-night-collision", args, payload)
        output = root / "collision.json"
        straight.write(output, result)
        print(f"Collision review scored on Astra: {output}")
        return 0
    targets = make_probes(root, plan, report)
    result = on_astra("astra-night-probes", args, {"plan": plan, "targets": targets})
    output = root / "single-edits.json"
    straight.write(output, result)
    print(f"Scored {result['tested_edits']} single-command edits: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
