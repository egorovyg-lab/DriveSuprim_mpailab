"""Matched straight-road pilot with portable observations and Astra PDM scoring."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
import json
from math import atan2, cos, isfinite, pi, sin
import os
from pathlib import Path
import shlex
from statistics import mean
import subprocess
import sys
from time import perf_counter


POSES = 40
STEP = 0.1
ACTION_STEPS = 5
SPEED_CAP = 10.0
ACCEL_CAP = 2.0
HORIZON = POSES * STEP
PROTOCOL = "navsim-straight-train-v1"
HOLDOUT_PROTOCOL = "navsim-straight-fresh-v1"
BASE_INSTRUCTIONS = (
    "Choose the longitudinal action for the next 0.5 seconds on a straight road. Maintain a clear gap "
    "while making forward progress within the speed limit. There is no destination stop in this horizon. "
    "Radar range is bumper-to-bumper. Positive closing speed means the gap is shrinking."
)
PROMPT_VARIANTS = {
    "baseline": "",
    "progress": (
        " For maintain versus slow, consider range and closing speed together. Prefer maintain if continuing at the "
        "current speed preserves a comfortable gap, including after an earlier slow action. Choose slow when reducing "
        "speed in this interval is needed to preserve that gap. Reassess from the current range and closing speed "
        "at each decision."
    ),
}
EXTRA_INPUT_VARIANTS = ("leader-history", "action-effects")
OBSERVATION_VARIANTS = ("radar", "gap-forecast", *EXTRA_INPUT_VARIANTS)
PREVIEW_SECONDS = 2.0


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(path)


def local_xy(x, y, origin):
    dx, dy = x - origin.x, y - origin.y
    c, s = cos(origin.heading), sin(origin.heading)
    return c * dx + s * dy, -s * dx + c * dy


def straight_reach(speed):
    ramp = min(HORIZON, (SPEED_CAP - speed) / ACCEL_CAP)
    return speed * ramp + 0.5 * ACCEL_CAP * ramp**2 + SPEED_CAP * (HORIZON - ramp)


def past_leader_speeds(cache, leader, origin, vehicle_type):
    samples = []
    if not leader.track_token:
        return samples
    cosine, sine = cos(origin.heading), sin(origin.heading)
    for frame in cache.past_detections_tracks[-2:]:
        observed = frame.tracked_objects.get_tracked_objects_of_type(vehicle_type)
        track = next((obj for obj in observed if obj.track_token == leader.track_token), None)
        if track is None:
            continue
        age = (leader.metadata.timestamp_us - track.metadata.timestamp_us) / 1e6
        velocity = cosine * track.velocity.x + sine * track.velocity.y
        if age > 0 and isfinite(velocity):
            samples.append({"age_at_start_s": round(age, 4), "speed_mps": round(velocity, 4)})
    return sorted(samples, key=lambda sample: -sample["age_at_start_s"])


def straight_observation(cache, include_history=False):
    """Filter by static road geometry and extract traffic from the current frame only."""
    import numpy as np
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from shapely.geometry import Point

    ego = cache.ego_state
    speed = float(ego.dynamic_car_state.speed)
    if not isfinite(speed) or not 0.5 <= speed <= SPEED_CAP:
        return None, "ego_speed"
    if len(cache.current_tracked_objects) != 1:
        return None, "current_tracks"

    origin = ego.rear_axle
    route = cache.centerline
    start = float(route.project(Point(origin.x, origin.y)))
    span = straight_reach(speed)
    if start + span > route.length:
        return None, "short_centerline"
    distances = np.linspace(0, span, 41)
    center = route.interpolate(start + distances, as_array=True)
    _, lateral = local_xy(center[:, 0], center[:, 1], origin)
    yaw = (center[:, 2] - origin.heading + pi) % (2 * pi) - pi
    if not np.isfinite(center).all() or np.max(np.abs(lateral)) > 0.8 or np.max(np.abs(yaw)) > pi / 36:
        return None, "curved_or_misaligned"

    # Screen a static straight envelope over the full 4 s reach, independent of DS outcomes.
    vehicle = ego.car_footprint.vehicle_parameters
    front = vehicle.rear_axle_to_center + vehicle.length / 2
    rear = vehicle.rear_axle_to_center - vehicle.length / 2
    offsets, wings = np.meshgrid(distances, [-vehicle.width / 2, vehicle.width / 2], indexing="ij")
    longitudinal = np.stack((offsets + rear, offsets + front), axis=-1)
    across = wings[..., None]
    c, s = cos(origin.heading), sin(origin.heading)
    corners = np.stack((origin.x + c * longitudinal - s * across, origin.y + s * longitudinal + c * across), axis=-1)
    road = cache.drivable_area_map
    membership = road.points_in_polygons(corners)
    intersections = road.get_indices_of_map_type([SemanticMapLayer.INTERSECTION])
    if membership[intersections].any():
        return None, "intersection"
    drivable = road.get_indices_of_map_type([
        SemanticMapLayer.ROADBLOCK, SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA,
    ])
    if not membership[drivable].any(axis=0).all():
        return None, "outside_straight_envelope"

    leaders = []
    tracks = cache.current_tracked_objects[0].tracked_objects
    for actor in tracks.get_tracked_objects_of_type(TrackedObjectType.VEHICLE):
        x, y = local_xy(actor.center.x, actor.center.y, origin)
        if not 0 < x <= 65 or abs(y) > 1.8:
            continue
        vx, vy = c * actor.velocity.x + s * actor.velocity.y, -s * actor.velocity.x + c * actor.velocity.y
        yaw = atan2(sin(actor.center.heading - origin.heading), cos(actor.center.heading - origin.heading))
        if not all(isfinite(z) for z in (vx, vy, yaw, actor.box.length, actor.box.width)):
            return None, "invalid_lead"
        if vx < -0.25 or abs(vy) > 1.0 or abs(yaw) > pi / 6:
            return None, "non_longitudinal_lead"
        rear_x = x - abs(cos(yaw)) * actor.box.length / 2 - abs(sin(yaw)) * actor.box.width / 2
        if rear_x <= front:
            return None, "initial_overlap"
        radar_track = {"gap_m": round(rear_x - front, 4), "speed_mps": round(max(0.0, vx), 4)}
        leaders.append((radar_track, actor))
    leaders.sort(key=lambda entry: entry[0]["gap_m"])
    observation = {"speed_mps": round(speed, 4), "leaders": [track for track, _ in leaders]}
    if include_history:
        observation["leader_history"] = [past_leader_speeds(cache, actor, origin, TrackedObjectType.VEHICLE)
                                         for _, actor in leaders]
    return observation, None


def load_cache(info):
    import lzma
    import pickle

    # Only paths already verified against the original audit's frozen cache index reach this function.
    with lzma.open(info["path"], "rb") as stream:
        return pickle.load(stream)


def source_context(package, source):
    from transfer import sha

    prior_path = source / "audit_request.json"
    identity_path = source / "artifacts/identity.json"
    identity = read(identity_path)
    report = read(source / "artifacts/report.json")
    if (report["status"] != "complete" or report["metric_parity_passed"] != 41664
            or identity["request_sha256"] != sha(prior_path)):
        raise RuntimeError("Expected the completed frozen pool64 audit")
    prior = read(prior_path)
    if prior["package_manifest_sha256"] != sha(package / "package_manifest.json"):
        raise RuntimeError("Portable package differs from the original audit")
    by_token = {}
    for candidate in prior["candidates"]:
        by_token.setdefault(candidate["token"], []).append(candidate)
    return prior["records"], by_token, sha(identity_path)


def prepare(package, source, repo, limit):
    from rollout_replay import bound_context

    records, candidates, source_id = source_context(package, source)
    ordered = sorted(records, key=lambda row: digest([PROTOCOL, row["token"]]))
    rows, sessions, rejected = [], set(), Counter()
    for start in range(0, len(ordered), 24):
        block = [record for record in ordered[start:start + 24] if record["session_id"] not in sessions]
        if not block:
            continue
        _, scenes, _, _, caches = bound_context(package, repo, [candidates[r["token"]][0] for r in block])
        for record in block:
            token, session = record["token"], record["session_id"]
            if session in sessions:
                continue
            cache = load_cache(caches[token])
            if cache.log_name != scenes[token]["log_name"]:
                raise RuntimeError(f"Cache scene mismatch for {token}")
            observation, reason = straight_observation(cache)
            if reason is not None:
                rejected[reason] += 1
                continue
            rows.append({"token": token, "session_id": session, "cache_sha256": caches[token]["sha256"], **observation})
            sessions.add(session)
            if len(rows) == limit:
                break
        if len(rows) == limit:
            break
        print(f"Checked up to {min(start + 24, len(ordered))}/{len(ordered)} records. Selected: {len(rows)}/{limit}",
              file=sys.stderr, flush=True)
    if not rows:
        raise RuntimeError(f"No eligible straight scenes. Rejections: {dict(rejected)}")
    return {"protocol": PROTOCOL, "anchor_identity_sha256": source_id, "requested_scenes": limit,
            "rejected": dict(rejected), "scenes": rows, "speed_cap_mps": SPEED_CAP}


def export_history(package, source, repo, plan):
    from rollout_replay import bound_context

    _, candidates, source_id = source_context(package, source)
    if plan["protocol"] != PROTOCOL or plan["anchor_identity_sha256"] != source_id:
        raise RuntimeError("The archived straight-scene selection differs")
    selected = [candidates[record["token"]][0] for record in plan["scenes"]]
    _, _, _, _, caches = bound_context(package, repo, selected)
    records = []
    for record in plan["scenes"]:
        token = record["token"]
        if caches[token]["sha256"] != record["cache_sha256"]:
            raise RuntimeError(f"Metric cache changed for {token}")
        observation, reason = straight_observation(load_cache(caches[token]), include_history=True)
        if reason is not None or any(observation[k] != record[k] for k in ("speed_mps", "leaders")):
            raise RuntimeError(f"Current observation changed for {token}")
        records.append({"token": token, "leader_history": observation["leader_history"]})
    return {"observation_sha256": digest(plan), "scenes": records}


def histories_by_scene(plan, history):
    if (history["observation_sha256"] != digest(plan)
            or [r["token"] for r in history["scenes"]] != [r["token"] for r in plan["scenes"]]):
        raise RuntimeError("Leader history refers to a different scene selection")
    return {r["token"]: r["leader_history"] for r in history["scenes"]}


def action_effects(ego, gap, lead_speed):
    from .simulation import ACCELERATION, Action

    previews = {}
    for action, acceleration in ACCELERATION.items():
        simulated = ego
        min_gap = gap
        for step in range(round(PREVIEW_SECONDS / STEP)):
            requested = acceleration if step < ACTION_STEPS else ACCELERATION[Action.MAINTAIN]
            simulated = simulated.advance(requested, SPEED_CAP)
            if gap is not None:
                next_gap = gap + (step + 1) * STEP * lead_speed - (simulated.front_m - ego.front_m)
                min_gap = min(min_gap, next_gap)
        previews[action.value] = {"progress_m": round(simulated.front_m - ego.front_m, 2),
                                 "minimum_gap_m": None if min_gap is None else round(min_gap, 2)}
    return previews


def build_plan(record, choose, prompt_variant="baseline", observation_variant="radar"):
    from .driver import request_for
    from .simulation import ACCELERATION, Action, Vehicle

    ego = Vehicle(0.0, record["speed_mps"])
    poses, decisions = [], []
    previous_action = Action.MAINTAIN
    previous_speed = None
    leaders = record["leaders"]
    histories = record.get("leader_history")
    if observation_variant == "leader-history" and (histories is None or len(histories) != len(leaders)):
        raise ValueError("Leader history must match the current radar tracks")
    for step in range(0, POSES, ACTION_STEPS):
        elapsed = round(step * STEP, 9)
        lead_index = min(range(len(leaders)), key=lambda i: leaders[i]["gap_m"] + elapsed * leaders[i]["speed_mps"],
                         default=None)
        gap, leader_speed = None, None
        radar = None
        if lead_index is not None:
            lead = leaders[lead_index]
            gap = lead["gap_m"] + elapsed * lead["speed_mps"] - ego.front_m
            leader_speed = lead["speed_mps"]
            closing_speed = ego.speed_mps - leader_speed
            radar = {"range_m": round(gap, 2), "closing_speed_mps": round(closing_speed, 2)}
            if observation_variant != "radar":
                radar["gap_if_maintain_1s_m"] = round(gap - closing_speed, 2)
                radar["gap_if_maintain_2s_m"] = round(gap - 2 * closing_speed, 2)
            if observation_variant == "leader-history":
                radar["measured_lead_speed_history"] = [
                    {"age_s": round(sample["age_at_start_s"] + elapsed, 3), "speed_mps": sample["speed_mps"]}
                    for sample in histories[lead_index]
                ]
        state = {"ego_speed_mps": round(ego.speed_mps, 2), "previous_ego_speed_mps": previous_speed,
                 "speed_limit_mps": SPEED_CAP, "lead_radar": radar, "previous_action": previous_action.value}
        if observation_variant == "action-effects":
            state["action_effects"] = {"horizon_s": PREVIEW_SECONDS, "apply_action_s": ACTION_STEPS * STEP,
                                       "after_action": "maintain", "forecast": action_effects(ego, gap, leader_speed)}
        row = request_for(state)
        row["question"]["instructions"] = BASE_INSTRUCTIONS + PROMPT_VARIANTS[prompt_variant]
        started = perf_counter()
        action, probabilities = choose(row)
        decisions.append({"time_s": elapsed, "request": row, "action": action.value,
                          "probabilities": probabilities, "wall_seconds": perf_counter() - started})
        previous_action, previous_speed = action, round(ego.speed_mps, 2)
        for _ in range(ACTION_STEPS):
            ego = ego.advance(ACCELERATION[action], SPEED_CAP)
            poses.append([ego.front_m, 0.0, 0.0])
    return {"token": record["token"], "poses": poses, "decisions": decisions}


def load_drex(weights):
    from .driver import DrexMax

    started = perf_counter()
    return DrexMax(weights), perf_counter() - started


def predict(plan, weights, out, prompt_variant="baseline", observation_variant="radar", history=None, model=None):
    if plan["protocol"] not in (PROTOCOL, HOLDOUT_PROTOCOL) or plan["speed_cap_mps"] != SPEED_CAP or not plan["scenes"]:
        raise ValueError("Unsupported or empty straight-scene package")
    if plan["protocol"] == HOLDOUT_PROTOCOL and (prompt_variant, observation_variant) != ("baseline", "action-effects"):
        raise ValueError("Fresh holdout is fixed to baseline prompt and action-effects")
    histories = histories_by_scene(plan, history) if observation_variant == "leader-history" else None
    choose, load_seconds = model if model is not None else load_drex(weights)
    report = {"observation_sha256": digest(plan), "prompt_variant": prompt_variant,
              "observation_variant": observation_variant, "model_path": str(weights),
              "model_load_wall_s": load_seconds, "cases": []}
    if histories is not None:
        report["history_sha256"] = digest(history)
    for record in plan["scenes"]:
        try:
            if histories is not None:
                record = {**record, "leader_history": histories[record["token"]]}
            outcome = build_plan(record, choose, prompt_variant, observation_variant)
        except (ValueError, RuntimeError) as exc:
            outcome = {"token": record["token"], "error": f"{type(exc).__name__}: {exc}"}
        report["cases"].append(outcome)
        write(out, report)
        print(f"{len(report['cases'])}/{len(plan['scenes'])} {record['token']}", flush=True)
    return any("error" in case for case in report["cases"])


def predict_suite(root, weights):
    pending = []
    for variant in EXTRA_INPUT_VARIANTS:
        out = root / variant
        plan = read(out / "observations.json")
        history = read(out / "history.json") if variant == "leader-history" else None
        output = out / "predictions.json"
        if output.exists():
            prior = read(output)
            if (prior["observation_sha256"] != digest(plan) or prior["observation_variant"] != variant
                    or prior["prompt_variant"] != "baseline" or len(prior["cases"]) != len(plan["scenes"])
                    or (history is not None and prior.get("history_sha256") != digest(history))):
                raise RuntimeError(f"Incomplete or incompatible prior predictions in {out}. Use a new run name")
        else:
            pending.append((variant, plan, history, output))
    if not pending:
        return False
    model = load_drex(weights)
    failures = [predict(plan, weights, output, observation_variant=variant, history=history, model=model)
                for variant, plan, history, output in pending]
    return any(failures)


def raw_score(path, poses, template, keys):
    import numpy as np
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from rollout_replay import config_core

    cfg = OmegaConf.create(config_core(template))
    simulator, scorer, sampling = [instantiate(cfg[key]) for key in ("simulator", "scorer", "agent_sampling")]
    if (cfg.traffic_agents != "non_reactive" or sampling.num_poses != POSES
            or abs(sampling.interval_length - STEP) > 1e-9):
        raise RuntimeError("Original non-reactive 40-pose protocol required")
    scorer._config.human_penalty_filter = False
    policy = instantiate(cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling)
    frame, _ = pdm_score(load_cache(path), Trajectory(np.asarray(poses, dtype=float), sampling),
                          simulator.proposal_sampling, simulator, scorer, policy)
    return {key: float(frame.iloc[0][key]) for key in keys}


def checked_straight_poses(prediction):
    import numpy as np

    poses = np.asarray(prediction["poses"], dtype=float)
    if (poses.shape != (POSES, 3) or not np.isfinite(poses).all() or np.any(poses[:, 1:] != 0)
            or np.any(np.diff(poses[:, 0], prepend=0) < 0) or poses[-1, 0] > SPEED_CAP * HORIZON + 1e-8):
        raise ValueError(f"Invalid straight trajectory for {prediction['token']}")
    return poses


def score(package, source, repo, payload):
    import numpy as np
    import real_data
    from rollout_replay import bound_context

    plan, predictions = payload["plan"], payload["predictions"]
    prompt_variant = predictions.get("prompt_variant", "baseline")
    observation_variant = predictions.get("observation_variant", "radar")
    _, candidates, source_id = source_context(package, source)
    if (plan["protocol"] != PROTOCOL or prompt_variant not in PROMPT_VARIANTS
            or observation_variant not in OBSERVATION_VARIANTS or plan["anchor_identity_sha256"] != source_id
            or predictions["observation_sha256"] != digest(plan)
            or [s["token"] for s in plan["scenes"]] != [c["token"] for c in predictions["cases"]]):
        raise RuntimeError("Observation, prediction or original audit identity differs")
    histories = None
    if observation_variant == "leader-history":
        history = payload["history"]
        if predictions.get("history_sha256") != digest(history):
            raise RuntimeError("Leader history differs from the GPU input")
        histories = histories_by_scene(plan, history)
    selected = [candidates[record["token"]][0] for record in plan["scenes"]]
    manifest, _, vocab, template, caches = bound_context(package, repo, selected)
    keys = ("pdm_score", *real_data.METRICS)
    compared, failed = [], []
    for record, prediction in zip(plan["scenes"], predictions["cases"]):
        token = record["token"]
        cache = caches[token]
        if cache["sha256"] != record["cache_sha256"]:
            raise RuntimeError(f"Metric cache changed for {token}")
        observation, reason = straight_observation(load_cache(cache), include_history=histories is not None)
        expected = {k: record[k] for k in ("speed_mps", "leaders")}
        if histories is not None:
            expected["leader_history"] = histories[token]
        if reason is not None or observation != expected:
            raise RuntimeError(f"Current observation changed for {token}")
        if "error" in prediction:
            failed.append({"token": token, "error": prediction["error"]})
            continue
        sample = real_data.load_inputs(package, token, manifest, vocab)
        anchor = next(c for c in candidates[token] if c["candidate_local"] == sample["baseline_local"])
        if list(keys) != anchor["metric_keys"]:
            raise RuntimeError(f"DriveSuprim metric columns changed for {token}")
        poses = checked_straight_poses(prediction)
        baseline = raw_score(cache, vocab[anchor["candidate_vocab"]], template, keys)
        if not np.allclose(list(baseline.values()), anchor["expected_raw"], atol=1e-8, rtol=1e-7):
            raise RuntimeError(f"DriveSuprim anchor parity failed for {token}")
        drex = raw_score(cache, poses, template, keys)
        compared.append({"token": token, "session_id": record["session_id"], "ds": baseline, "drex": drex,
                         "delta": {key: drex[key] - baseline[key] for key in keys}})
    return {"protocol": PROTOCOL, "prompt_variant": prompt_variant, "observation_variant": observation_variant,
            "observation_sha256": digest(plan),
            "planned_scenes": len(plan["scenes"]), "compared_scenes": len(compared),
            "prediction_failures": failed,
            "mean_delta": {key: mean(row["delta"][key] for row in compared) for key in keys} if compared else None,
            "safety_regressions": {key: sum(row["delta"][key] < -1e-8 for row in compared) for key in real_data.SAFETY},
            "cases": compared}


def holdout_records(selection, audit_records):
    holdout = [row for row in selection["records"] if row["role"] == "holdout"]
    prior = ([row for row in selection["records"] if row["role"] == "train"]
             + selection["legacy_training_records"] + audit_records)
    sessions = set(selection["fresh_holdout_sessions"])
    tokens = {row["token"] for row in holdout}
    if (not holdout or any(row["role"] not in ("train", "holdout") for row in selection["records"])
            or {row["session_id"] for row in holdout} != sessions or len(tokens) != len(holdout)
            or any(row["session_id"] in sessions or row["token"] in tokens for row in prior)):
        raise RuntimeError("Invalid holdout membership or overlap with prior training scenes")
    return sorted(holdout, key=lambda row: (row["session_id"], row["timestamp_us"], row["token"]))


def fresh_context(args):
    from download import prepare_selector_dataset as acquisition
    from navsim.planning.script.run_selector_collection import cfg_from_source
    from transfer import sha

    audit_records, candidates, anchor_id = source_context(args.package, args.source)
    acquisition_plan, requests = acquisition.verify(args.acquisition)
    selection = read(args.acquisition / "selection.json")
    selected = holdout_records(selection, audit_records)
    config_path = Path(acquisition_plan["options"]["source"]) / "identity.json"
    template = read(config_path)
    cfg = cfg_from_source(template)
    current_maps = os.environ.get("NUPLAN_MAPS_ROOT")
    if current_maps and Path(current_maps).resolve() != Path(template["map_root"]).resolve():
        raise RuntimeError("Map root differs from original DriveSuprim configuration")
    os.environ["NUPLAN_MAPS_ROOT"] = template["map_root"]
    metadata = {
        "protocol": HOLDOUT_PROTOCOL, "anchor_identity_sha256": anchor_id,
        "acquisition_sha256": sha(args.acquisition / "acquisition_plan.json"),
        "teacher_config_sha256": sha(config_path), "teacher_checkpoint_sha256": template["checkpoint_sha256"],
        "speed_cap_mps": SPEED_CAP, "reserved_scenes": len(selected),
        "reserved_sessions": sorted(selection["fresh_holdout_sessions"]),
    }
    logs = {row["log_name"] for row in selected}
    split = {"records": selected, "log_sha256": {
        log: requests["log_hashes"][str(Path(cfg.navsim_log_path) / (log + ".pkl"))] for log in sorted(logs)}}
    return (cfg, template, metadata, split, requests, candidates)


def fresh_prepare(args):
    import fcntl
    from navsim.planning.script.run_drivemem_memory import build_cache

    cfg, _, metadata, split, requests, _ = fresh_context(args)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    with (args.cache_root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        frozen_split = args.cache_root / "splits.json"
        frozen_plan = args.cache_root / "observations.json"
        binding = {"metadata": metadata, **split}
        if not frozen_split.exists():
            write(frozen_split, binding)
        elif read(frozen_split) != binding:
            raise RuntimeError("Astra holdout cache root belongs to another acquisition or protocol")
        if frozen_plan.exists():
            return read(frozen_plan)
        build_cache(cfg, args.cache_root)
        caches = read(args.cache_root / "cache_index.json")
        selected, selected_rows, rejected = [], [], Counter()
        for row in split["records"]:
            token = row["token"]
            cache = caches[token]
            observation, reason = straight_observation(load_cache(cache))
            if reason is not None:
                rejected[reason] += 1
                continue
            selected.append({"token": token, "session_id": row["session_id"], "cache_sha256": cache["sha256"],
                             **observation})
            selected_rows.append(row)
        plan = {**metadata, "scenes": selected, "rejected": dict(rejected)}
        if selected:
            verify_fresh_images(args, requests, cfg, selected_rows)
        write(frozen_plan, plan)
        return plan


def native_ds_predictions(cfg, rows, device):
    import torch
    from hydra.utils import instantiate
    from torch.utils.data import DataLoader
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.script.run_drivemem_oracle_pilot import on_device
    from navsim.planning.training.dataset_ssl import DatasetSSL

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Astra container. Retry with DREX_ASTRA_DS_DEVICE=cpu "
                           "or restore GPU access")
    if cfg.agent.config.training or cfg.agent.config.inference.model != "teacher":
        raise RuntimeError("Original DriveSuprim evaluation teacher required")
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()
    agent.to(device)
    agent._config.inference.export_candidates = False
    predictions = {}
    for log in sorted({row["log_name"] for row in rows}):
        scene_filter = instantiate(cfg.train_test_split.scene_filter)
        scene_filter.log_names = [log]
        scene_filter.tokens = [row["token"] for row in rows if row["log_name"] == log]
        loader = SceneLoader(Path(cfg.navsim_log_path), Path(cfg.original_sensor_path), scene_filter,
                             sensor_config=agent.get_sensor_config())
        if set(loader.tokens) != set(scene_filter.tokens):
            raise RuntimeError(f"DriveSuprim input membership changed in {log}")
        dataset = DatasetSSL(loader, agent.get_feature_builders(), agent.get_target_builders(),
                             cfg=cfg.agent.config, cache_path=None, append_token_to_batch=True)
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
            token = str(batch[2][0])
            with torch.no_grad():
                prediction, _ = agent.forward(on_device(batch, device))
            predictions[token] = prediction["final_traj"].cpu().numpy()[0].copy()
    return predictions


def verify_fresh_images(args, requests, cfg, rows):
    from download import prepare_selector_dataset as acquisition
    from transfer import sha

    root = Path(cfg.original_sensor_path)
    if root.resolve() != Path(requests["sensor_root"]).resolve():
        raise RuntimeError("Original DriveSuprim camera root differs from the frozen acquisition")
    receipts = read(args.acquisition / "installed_images.json")
    relative_paths = set()
    for log in sorted({row["log_name"] for row in rows}):
        path = Path(cfg.navsim_log_path) / (log + ".pkl")
        if sha(path) != requests["log_hashes"][str(path)]:
            raise RuntimeError(f"Source log changed for {log}")
        frames, positions = acquisition.load_frames(path)
        for row in rows:
            if row["log_name"] == log:
                relative_paths.update(acquisition.camera_paths(frames, positions[row["token"]]))
    for relative in sorted(relative_paths):
        path = acquisition.checked_target(root, relative)
        info = receipts.get(relative)
        if not info or not path.is_file() or path.stat().st_size != info["size"] or sha(path) != info["sha256"]:
            raise RuntimeError(f"Frozen camera data missing or changed: {relative}")


def fresh_score(args, payload):
    import numpy as np
    import real_data
    from rollout_replay import bound_context
    from transfer import sha

    cfg, teacher, metadata, split, requests, candidates = fresh_context(args)
    plan, predictions = payload["plan"], payload["predictions"]
    pilot_token = payload["pilot_token"]
    tokens = [row["token"] for row in plan["scenes"]]
    if (plan != read(args.cache_root / "observations.json") or any(plan[k] != v for k, v in metadata.items())
            or not tokens or tokens != [row["token"] for row in predictions["cases"]]
            or predictions["observation_sha256"] != digest(plan)
            or (predictions["prompt_variant"], predictions["observation_variant"]) != ("baseline", "action-effects")):
        raise RuntimeError("Fresh holdout observations or predictions do not match the frozen protocol")
    if sha(Path(cfg.agent.checkpoint_path)) != teacher["checkpoint_sha256"]:
        raise RuntimeError("Original DriveSuprim checkpoint changed")
    for relative, expected in teacher["sources"].items():
        if sha(args.repo / relative) != expected:
            raise RuntimeError(f"Original DriveSuprim source changed: {relative}")

    manifest, pilot_scenes, vocab, template, old_caches = bound_context(args.package, args.repo, [candidates[pilot_token][0]])
    keys = ("pdm_score", *real_data.METRICS)
    pilot_sample = real_data.load_inputs(args.package, pilot_token, manifest, vocab)
    pilot = next(row for row in candidates[pilot_token] if row["candidate_local"] == pilot_sample["baseline_local"])
    if list(keys) != pilot["metric_keys"]:
        raise RuntimeError("Original DriveSuprim metric columns changed")
    pilot_row = {"token": pilot_token, "log_name": pilot_scenes[pilot_token]["log_name"]}
    token_set = set(tokens)
    selected = [row for row in split["records"] if row["token"] in token_set]
    verify_fresh_images(args, requests, cfg, selected)
    native = native_ds_predictions(cfg, [pilot_row, *selected], args.ds_device)
    if not np.allclose(native[pilot_token], vocab[pilot["candidate_vocab"]], rtol=0, atol=1e-6):
        raise RuntimeError("Recomputed DriveSuprim trajectory differs from the archived pilot baseline")
    reproduced = raw_score(old_caches[pilot_token], native[pilot_token], template, keys)
    if not np.allclose(list(reproduced.values()), pilot["expected_raw"], atol=1e-8, rtol=1e-7):
        raise RuntimeError("Recomputed DriveSuprim scores differ from the archived pilot baseline")

    caches = read(args.cache_root / "cache_index.json")
    compared, failed = [], []
    for record, prediction in zip(plan["scenes"], predictions["cases"]):
        token = record["token"]
        cache = caches[token]
        if sha(Path(cache["path"])) != record["cache_sha256"]:
            raise RuntimeError(f"Fresh cache changed for {token}")
        if "error" in prediction:
            failed.append({"token": token, "error": prediction["error"]})
            continue
        baseline = raw_score(cache, native[token], template, keys)
        drex = raw_score(cache, checked_straight_poses(prediction), template, keys)
        compared.append({"token": token, "session_id": record["session_id"], "ds": baseline, "drex": drex,
                         "delta": {key: drex[key] - baseline[key] for key in keys}})
    result = fresh_report(plan, keys, real_data.SAFETY, compared, failed, pilot_token)
    return {**result, "ds_inference_device": args.ds_device}


def fresh_report(plan, keys, safety, compared, failed, pilot_token):
    sessions = []
    for session in plan["reserved_sessions"]:
        rows = [row for row in compared if row["session_id"] == session]
        sessions.append({"session_id": session,
                         "eligible_scenes": sum(row["session_id"] == session for row in plan["scenes"]),
                         "compared_scenes": len(rows),
                         "mean_delta": {key: mean(row["delta"][key] for row in rows) for key in keys} if rows else None})
    observed = [row for row in sessions if row["compared_scenes"]]
    differences = [row["delta"]["pdm_score"] for row in compared]
    return {"protocol": HOLDOUT_PROTOCOL, "prompt_variant": "baseline", "observation_variant": "action-effects",
            "observation_sha256": digest(plan), "reserved_scenes": plan["reserved_scenes"],
            "planned_scenes": len(plan["scenes"]), "compared_scenes": len(compared),
            "prediction_failures": failed, "eligibility_rejections": plan["rejected"],
            "teacher_checkpoint_sha256": plan["teacher_checkpoint_sha256"], "baseline_parity_token": pilot_token,
            "mean_delta": {key: mean(row["delta"][key] for row in compared) for key in keys} if compared else None,
            "mean_delta_equal_session": {
                key: mean(row["mean_delta"][key] for row in observed) for key in keys} if observed else None,
            "mean_pdm": {method: mean(row[method]["pdm_score"] for row in compared)
                         for method in ("ds", "drex")} if compared else None,
            "pdm_wins": sum(delta > 1e-8 for delta in differences),
            "pdm_losses": sum(delta < -1e-8 for delta in differences),
            "pdm_ties": sum(abs(delta) <= 1e-8 for delta in differences),
            "safety_regressions": {key: sum(row["delta"][key] < -1e-8 for row in compared) for key in safety},
            "sessions": sessions, "cases": compared}


def summarize_inputs(root, reference_report):
    reference = read(reference_report)
    by_token = {case["token"]: case for case in reference["cases"]}
    if (reference.get("observation_variant") != "gap-forecast" or reference.get("prompt_variant") != "baseline"
            or len(by_token) != reference["planned_scenes"]):
        raise RuntimeError("Expected the completed gap-forecast comparison")
    variants = []
    for variant in EXTRA_INPUT_VARIANTS:
        result = read(root / variant / "report.json")
        if (result["observation_sha256"] != reference["observation_sha256"]
                or result["observation_variant"] != variant or result["prompt_variant"] != "baseline"):
            raise RuntimeError(f"Mismatched comparison for {variant}")
        deltas = []
        for case in result["cases"]:
            previous = by_token[case["token"]]
            if previous["ds"] != case["ds"]:
                raise RuntimeError(f"DriveSuprim baseline changed for {case['token']}")
            deltas.append({key: case["drex"][key] - value for key, value in previous["drex"].items()})
        variants.append({"observation_variant": variant, "compared_scenes": result["compared_scenes"],
                         "prediction_failures": result["prediction_failures"], "mean_delta_vs_ds": result["mean_delta"],
                         "mean_delta_vs_gap_forecast": {key: mean(row[key] for row in deltas) for key in deltas[0]}
                         if deltas else None,
                         "pdm_better_vs_gap_forecast": sum(row["pdm_score"] > 1e-8 for row in deltas),
                         "pdm_worse_vs_gap_forecast": sum(row["pdm_score"] < -1e-8 for row in deltas),
                         "safety_regressions_vs_ds": result["safety_regressions"],
                         "safety_regressions_vs_gap_forecast": {
                             key: sum(row[key] < -1e-8 for row in deltas) for key in result["safety_regressions"]},
                         "report": f"{variant}/report.json"})
    history = read(root / "leader-history" / "history.json")
    return {"observation_sha256": reference["observation_sha256"], "planned_scenes": len(by_token),
            "history_available_scenes": sum(any(row["leader_history"]) for row in history["scenes"]),
            "gap_forecast_mean_delta_vs_ds": reference["mean_delta"], "variants": variants}


def on_astra(stage, args, payload=None):
    container = os.environ.get("DREX_ASTRA_CONTAINER", "egorov_drivesuprim")
    package = os.environ.get("DREX_ASTRA_PACKAGE", "/workspace/exp_v2/drex_transfer/navsim_train_v1")
    source = os.environ.get("DREX_STRAIGHT_SOURCE", "/workspace/exp_v2/drex_rollout_audit/rollout_audit_pool64_v1")
    repo = os.environ.get("DREX_ASTRA_REPO", "/workspace/DriveSuprim")
    control = os.environ.get("DREX_TRANSFER_CONTROL", str(Path.home() / ".ssh/drex-astra-transfer.sock"))
    host = os.environ.get("DREX_ASTRA_HOST", "egorov@93.180.62.99")
    port = os.environ.get("DREX_ASTRA_PORT", "27434")
    ds_device = os.environ.get("DREX_ASTRA_DS_DEVICE", "cuda")
    gpu = os.environ.get("DREX_ASTRA_DS_GPU", "0") if stage == "fresh-score" and ds_device == "cuda" else ""
    remote = ["docker", "exec", "-i", "-w", repo, container, "env", f"PYTHONPATH={source}/tools:{repo}",
              f"CUDA_VISIBLE_DEVICES={gpu}", "OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "MKL_NUM_THREADS=2",
              "python", "-B", "-c", Path(__file__).read_text(encoding="utf-8"), stage,
              "--package", package, "--source", source, "--repo", repo, "--limit", str(args.limit)]
    if stage in ("fresh-prepare", "fresh-score"):
        root = "/workspace/exp_v2/drivemem_memory"
        acquisition = os.environ.get("DREX_ASTRA_ACQUISITION", f"{root}/selector_data_expansion_v2")
        cache_root = os.environ.get("DREX_ASTRA_FRESH_CACHE", f"{root}/{HOLDOUT_PROTOCOL}")
        remote.extend(["--acquisition", acquisition, "--cache-root", cache_root])
    if stage == "fresh-score":
        remote.extend(["--ds-device", ds_device])
    finished = subprocess.run(["ssh", "-T", "-S", control, "-o", "BatchMode=yes", "-p", port, "--", host,
                               shlex.join(remote)], input=json.dumps(payload) if payload is not None else "",
                              text=True, encoding="utf-8", stdout=subprocess.PIPE)
    if finished.returncode:
        raise SystemExit(f"Astra stage {stage} failed with exit status {finished.returncode}. See stderr above.")
    return json.loads(finished.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict", "score", "history", "predict-suite",
                                          "fresh-prepare", "fresh-score", "bridge-fresh-prepare", "bridge-fresh-score",
                                          "bridge-prepare", "bridge-score", "bridge-history", "summarize-inputs"))
    parser.add_argument("--package", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--acquisition", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--ds-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--pilot-observations", type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS)
    parser.add_argument("--observation-variant", choices=OBSERVATION_VARIANTS)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.stage in ("prepare", "score", "history", "fresh-prepare", "fresh-score"):
        if None in (args.package, args.source, args.repo):
            parser.error("Astra stages require --package, --source and --repo")
        if args.stage in ("fresh-prepare", "fresh-score") and None in (args.acquisition, args.cache_root):
            parser.error("Fresh Astra stages require --acquisition and --cache-root")
        with redirect_stdout(sys.stderr):
            if args.stage == "prepare":
                result = prepare(args.package, args.source, args.repo, args.limit)
            elif args.stage == "history":
                result = export_history(args.package, args.source, args.repo, json.load(sys.stdin))
            elif args.stage == "fresh-prepare":
                result = fresh_prepare(args)
            elif args.stage == "fresh-score":
                result = fresh_score(args, json.load(sys.stdin))
            else:
                result = score(args.package, args.source, args.repo, json.load(sys.stdin))
        print(json.dumps(result, allow_nan=False))
        return 0
    if args.out is None:
        parser.error("--out must point to the experiment directory")
    if args.stage == "summarize-inputs":
        if args.reference_report is None:
            parser.error("--reference-report must point to the gap-forecast report")
        result = summarize_inputs(args.out, args.reference_report)
        write(args.out / "suite.json", result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return int(any(variant["prediction_failures"] for variant in result["variants"]))
    if args.stage in ("predict", "predict-suite") and (args.weights is None or not args.weights.is_dir()):
        parser.error("--weights must point to the Drex Max weights directory")
    if args.stage == "predict-suite":
        return int(predict_suite(args.out, args.weights))
    observations = args.out / "observations.json"
    predictions = args.out / "predictions.json"
    summary = args.out / "report.json"
    history_path = args.out / "history.json"
    if args.stage == "bridge-fresh-prepare":
        args.out.mkdir(parents=True, exist_ok=True)
        result = on_astra("fresh-prepare", args)
        if observations.exists() and read(observations) != result:
            parser.error("Saved observations do not match the frozen Astra cohort")
        if not observations.exists():
            write(observations, result)
        print(f"Eligible straight scenes: {len(result['scenes'])}/{result['reserved_scenes']}. Observations: {observations}")
        print("Eligibility rejections:", result["rejected"])
        if not result["scenes"]:
            report = {"protocol": HOLDOUT_PROTOCOL, "status": "no_eligible_scenes", "compared_scenes": 0,
                      "reserved_scenes": result["reserved_scenes"], "reserved_sessions": result["reserved_sessions"],
                      "eligibility_rejections": result["rejected"]}
            write(summary, report)
            return 3
        return 0
    if args.stage == "bridge-prepare":
        if observations.exists():
            parser.error("Use a new run name to prepare a different cohort")
        result = on_astra("prepare", args)
        args.out.mkdir(parents=True, exist_ok=True)
        write(observations, result)
        print(f"Prepared {len(result['scenes'])}/{args.limit} straight scenes. Observations: {observations}")
        print("Rejected before selection:", result["rejected"])
        return 0
    if args.stage == "bridge-history":
        plan = read(observations)
        if not history_path.exists():
            write(history_path, on_astra("history", args, plan))
        histories = histories_by_scene(plan, read(history_path))
        covered = sum(any(tracks) for tracks in histories.values())
        print(f"Past leader observations available in {covered}/{len(histories)} scenes. History: {history_path}")
        return 0
    if args.stage == "predict":
        if predictions.exists():
            parser.error("Use a new run name to write a different prediction file")
        history = read(history_path) if args.observation_variant == "leader-history" else None
        return int(predict(read(observations), args.weights, predictions,
                           args.prompt_variant or "baseline", args.observation_variant or "radar", history=history))
    plan, output = read(observations), read(predictions)
    if len(plan["scenes"]) != len(output["cases"]):
        parser.error("Run scoring after the GPU job has written all planned case results")
    if args.prompt_variant and output.get("prompt_variant", "baseline") != args.prompt_variant:
        parser.error("The saved predictions use a different prompt variant")
    if args.observation_variant and output.get("observation_variant", "radar") != args.observation_variant:
        parser.error("The saved predictions use a different observation variant")
    payload = {"plan": plan, "predictions": output}
    if output.get("observation_variant") == "leader-history":
        payload["history"] = read(history_path)
    if args.stage == "bridge-fresh-score":
        if args.pilot_observations is None:
            parser.error("--pilot-observations must point to the original TRAIN observations")
        pilot = read(args.pilot_observations)
        if pilot["protocol"] != PROTOCOL or pilot["anchor_identity_sha256"] != plan["anchor_identity_sha256"]:
            parser.error("Original TRAIN pilot anchor differs from the fresh cohort's reference")
        payload["pilot_token"] = pilot["scenes"][0]["token"]
    result = on_astra("fresh-score" if args.stage == "bridge-fresh-score" else "score", args, payload)
    write(summary, result)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
    print(f"Paired scores: {summary}")
    return int(result["compared_scenes"] == 0 or (args.stage == "bridge-fresh-score" and result["prediction_failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
