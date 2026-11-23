"""Matched straight-road pilot with portable observations and Astra PDM scoring."""

from __future__ import annotations

import argparse
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
OBSERVATION_VARIANTS = ("radar", "gap-forecast")


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


def straight_observation(cache):
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
        leaders.append({"gap_m": round(rear_x - front, 4), "speed_mps": round(max(0.0, vx), 4)})
    leaders.sort(key=lambda actor: actor["gap_m"])
    return {"speed_mps": round(speed, 4), "leaders": leaders}, None


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
    from collections import Counter
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


def build_plan(record, choose, prompt_variant="baseline", observation_variant="radar"):
    from .driver import request_for
    from .simulation import ACCELERATION, Action, Vehicle

    ego = Vehicle(0.0, record["speed_mps"])
    poses, decisions = [], []
    previous_action = Action.MAINTAIN
    previous_speed = None
    for step in range(0, POSES, ACTION_STEPS):
        elapsed = round(step * STEP, 9)
        lead = min(record["leaders"], key=lambda a: a["gap_m"] + elapsed * a["speed_mps"], default=None)
        radar = None
        if lead is not None:
            gap = lead["gap_m"] + elapsed * lead["speed_mps"] - ego.front_m
            closing_speed = ego.speed_mps - lead["speed_mps"]
            radar = {"range_m": round(gap, 2), "closing_speed_mps": round(closing_speed, 2)}
            if observation_variant == "gap-forecast":
                radar["gap_if_maintain_1s_m"] = round(gap - closing_speed, 2)
                radar["gap_if_maintain_2s_m"] = round(gap - 2 * closing_speed, 2)
        row = request_for({"ego_speed_mps": round(ego.speed_mps, 2), "previous_ego_speed_mps": previous_speed,
                           "speed_limit_mps": SPEED_CAP, "lead_radar": radar, "previous_action": previous_action.value})
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


def predict(plan, weights, out, prompt_variant="baseline", observation_variant="radar"):
    from .driver import DrexMax

    if plan["protocol"] != PROTOCOL or plan["speed_cap_mps"] != SPEED_CAP or not plan["scenes"]:
        raise ValueError("Unsupported or empty straight-scene package")
    started = perf_counter()
    choose = DrexMax(weights)
    report = {"observation_sha256": digest(plan), "prompt_variant": prompt_variant,
              "observation_variant": observation_variant, "model_path": str(weights),
              "model_load_wall_s": perf_counter() - started, "cases": []}
    for record in plan["scenes"]:
        try:
            outcome = build_plan(record, choose, prompt_variant, observation_variant)
        except (ValueError, RuntimeError) as exc:
            outcome = {"token": record["token"], "error": f"{type(exc).__name__}: {exc}"}
        report["cases"].append(outcome)
        write(out, report)
        print(f"{len(report['cases'])}/{len(plan['scenes'])} {record['token']}", flush=True)
    return any("error" in case for case in report["cases"])


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
    selected = [candidates[record["token"]][0] for record in plan["scenes"]]
    manifest, _, vocab, template, caches = bound_context(package, repo, selected)
    keys = ("pdm_score", *real_data.METRICS)
    compared, failed = [], []
    for record, prediction in zip(plan["scenes"], predictions["cases"]):
        token = record["token"]
        cache = caches[token]
        if cache["sha256"] != record["cache_sha256"]:
            raise RuntimeError(f"Metric cache changed for {token}")
        observation, reason = straight_observation(load_cache(cache))
        if reason is not None or observation != {k: record[k] for k in ("speed_mps", "leaders")}:
            raise RuntimeError(f"Current observation changed for {token}")
        if "error" in prediction:
            failed.append({"token": token, "error": prediction["error"]})
            continue
        sample = real_data.load_inputs(package, token, manifest, vocab)
        anchor = next(c for c in candidates[token] if c["candidate_local"] == sample["baseline_local"])
        if list(keys) != anchor["metric_keys"]:
            raise RuntimeError(f"DriveSuprim metric columns changed for {token}")
        poses = np.asarray(prediction["poses"], dtype=float)
        if (poses.shape != (POSES, 3) or not np.isfinite(poses).all() or np.any(poses[:, 1:] != 0)
                or np.any(np.diff(poses[:, 0], prepend=0) < 0) or poses[-1, 0] > SPEED_CAP * HORIZON + 1e-8):
            raise ValueError(f"Invalid straight trajectory for {token}")
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


def on_astra(stage, args, payload=None):
    container = os.environ.get("DREX_ASTRA_CONTAINER", "egorov_drivesuprim")
    package = os.environ.get("DREX_ASTRA_PACKAGE", "/workspace/exp_v2/drex_transfer/navsim_train_v1")
    source = os.environ.get("DREX_STRAIGHT_SOURCE", "/workspace/exp_v2/drex_rollout_audit/rollout_audit_pool64_v1")
    repo = os.environ.get("DREX_ASTRA_REPO", "/workspace/DriveSuprim")
    control = os.environ.get("DREX_TRANSFER_CONTROL", str(Path.home() / ".ssh/drex-astra-transfer.sock"))
    host = os.environ.get("DREX_ASTRA_HOST", "egorov@93.180.62.99")
    port = os.environ.get("DREX_ASTRA_PORT", "27434")
    remote = ["docker", "exec", "-i", "-w", repo, container, "env", f"PYTHONPATH={source}/tools:{repo}",
              "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "MKL_NUM_THREADS=2",
              "python", "-B", "-c", Path(__file__).read_text(encoding="utf-8"), stage,
              "--package", package, "--source", source, "--repo", repo, "--limit", str(args.limit)]
    finished = subprocess.run(["ssh", "-T", "-S", control, "-o", "BatchMode=yes", "-p", port, "--", host,
                               shlex.join(remote)], input=json.dumps(payload) if payload is not None else "",
                              text=True, encoding="utf-8", stdout=subprocess.PIPE, check=True)
    return json.loads(finished.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict", "score", "bridge-prepare", "bridge-score"))
    parser.add_argument("--package", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS)
    parser.add_argument("--observation-variant", choices=OBSERVATION_VARIANTS)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.stage in ("prepare", "score"):
        if None in (args.package, args.source, args.repo):
            parser.error("Astra stages require --package, --source and --repo")
        with redirect_stdout(sys.stderr):
            result = (prepare(args.package, args.source, args.repo, args.limit) if args.stage == "prepare" else
                      score(args.package, args.source, args.repo, json.load(sys.stdin)))
        print(json.dumps(result, allow_nan=False))
        return 0
    if args.out is None:
        parser.error("--out must point to the experiment directory")
    observations = args.out / "observations.json"
    predictions = args.out / "predictions.json"
    summary = args.out / "report.json"
    if args.stage == "bridge-prepare":
        if observations.exists():
            parser.error("Use a new run name to prepare a different cohort")
        result = on_astra("prepare", args)
        args.out.mkdir(parents=True, exist_ok=True)
        write(observations, result)
        print(f"Prepared {len(result['scenes'])}/{args.limit} straight scenes. Observations: {observations}")
        print("Rejected before selection:", result["rejected"])
        return 0
    if args.stage == "predict":
        if args.weights is None or not args.weights.is_dir():
            parser.error("--weights must point to the Drex Max weights directory")
        if predictions.exists():
            parser.error("Use a new run name to write a different prediction file")
        return int(predict(read(observations), args.weights, predictions,
                           args.prompt_variant or "baseline", args.observation_variant or "radar"))
    plan, output = read(observations), read(predictions)
    if len(plan["scenes"]) != len(output["cases"]):
        parser.error("Run scoring after the GPU job has written all planned case results")
    if args.prompt_variant and output.get("prompt_variant", "baseline") != args.prompt_variant:
        parser.error("The saved predictions use a different prompt variant")
    if args.observation_variant and output.get("observation_variant", "radar") != args.observation_variant:
        parser.error("The saved predictions use a different observation variant")
    result = on_astra("score", args, {"plan": plan, "predictions": output})
    write(summary, result)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
    print(f"Paired scores: {summary}")
    return int(result["compared_scenes"] == 0)


if __name__ == "__main__":
    raise SystemExit(main())
