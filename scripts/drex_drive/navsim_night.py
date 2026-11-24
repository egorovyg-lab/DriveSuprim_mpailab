"""Run a matched NAVSIM straight-road experiment over the full archived TRAIN inventory."""

import argparse
from base64 import b64encode
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from statistics import mean
from math import cos, hypot, isclose, isfinite, pi, sin
import json
import sys
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


def score_corridor(args, payload):
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
    for_scene, protected = archive_evaluator(args, plan)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict", "score", "probes", "collision",
                                          "corridor-prepare", "corridor-predict", "corridor-score",
                                          "preview-prepare", "preview-predict", "preview-score",
                                          "astra-night-prepare", "astra-night-score", "astra-night-probes",
                                          "astra-night-collision", "astra-night-corridor-prepare",
                                          "astra-night-corridor-score", "astra-night-preview-prepare"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--limit", type=int, default=8)  # Common Astra transport argument.
    args = parser.parse_args()
    if args.stage.startswith("astra-night-"):
        payload = json.load(sys.stdin)
        with redirect_stdout(sys.stderr):
            remote = {"astra-night-prepare": prepare, "astra-night-score": score,
                      "astra-night-probes": score_probes, "astra-night-collision": score_collision,
                      "astra-night-corridor-prepare": prepare_corridor, "astra-night-corridor-score": score_corridor}
            if args.stage == "astra-night-preview-prepare":
                result = prepare_corridor(args, payload, include_bounds=True)
            else:
                result = remote[args.stage](args, payload)
        print(json.dumps(result, allow_nan=False))
        return 0

    if args.out is None:
        parser.error("--out is required")
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
            variants = {}
            for name, setup in PREVIEW_SETUPS.items():
                plan_hash, predictions = read_corridor_predictions(root, prediction_plan, name, setup)
                payload = {**details, "variant": name, "prediction_plan_sha256": plan_hash, "predictions": predictions}
                result = on_astra("astra-night-corridor-score", args, payload)
                result["vs_corridor_objects"] = compare_preview_with_corridor(result, previous, name)
                variants[name] = result
            output = root / "corridor-preview.json"
            straight.write(output, {"protocol": "navsim-straight-corridor-preview-v1",
                                    "position_margin_m": straight.CORRIDOR_POSITION_MARGIN_M,
                                    "observation_sha256": straight.digest(plan), "variants": variants})
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
