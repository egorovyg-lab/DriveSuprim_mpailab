"""Offline fresh Drex A/C: freeze, predict on MSU270, score on Astra, validate locally."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
from importlib.util import find_spec
import json
from math import fsum, isclose, isfinite
from pathlib import Path, PurePosixPath
import platform
from random import Random
import re
from statistics import mean
import sys
from time import perf_counter
from typing import Any, Callable

from . import lead_brake_gap_bc as bc

night, straight = bc.night, bc.straight
EXPERIMENT = "corridor-preview-lead-brake-gap-conditional-fresh-v1"
ARMS = ("a", "c")
N_DRIVES, DRAW_SEED, SEED = 120, 20261125, 20261124
REPS, RANK, TOL = 20_000, 999, 1e-8
DRAW = "sha256-ascii-seed-colon-session-id-v1"
PDM, EP = ENDPOINTS = ("pdm_score", "ego_progress")
NC, TTC, DAC, DDC, TL, LK, HC = PROTECTED = night.BRAKE_GAP_PROTECTED
METRICS = (*ENDPOINTS, *PROTECTED)
SETUP = night.PREVIEW_SETUPS[night.BRAKE_GAP_CONTROL]
TIMES = [step / 10 for step in range(0, straight.POSES, straight.ACTION_STEPS)]
CAUSAL = ("speed_mps", "leaders", "corridor_objects", "ego_bounds")
SCENE = ("token", "session_id", "cache_sha256", *CAUSAL)
EXCLUSIONS = ("archive32", "reserve4", "other_prior")
CODE_FILES = ("fresh_conditional.py", "fresh_conditional.sh", "lead_brake_gap_bc.py", "navsim_night.py",
              "navsim_straight.py", "driver.py", "simulation.py", "camera.py")
POLICY = {"arms": {"a": "corridor-preview", "c": "corridor-preview+lead-brake-gap-when-closing"},
          "lead_decel_mps2": 2.0, "seed_per_arm": SEED, "draw": DRAW, "draw_seed": DRAW_SEED,
          "drives": N_DRIVES, "poses": [40, 3], "step_s": 0.1, "decisions": 8, "choice": "argmax",
          "max_length": 2048, "filter": "unmodified-straight_observation"}
OBJ = dict[str, Any]
ERRORS = (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, OverflowError, RecursionError)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact(value: Any, keys: tuple[str, ...], label: str) -> OBJ:
    need(isinstance(value, dict) and set(value) == set(keys), f"{label}: unexpected/missing fields")
    return value


def unique_json(items: list[tuple[str, Any]]) -> OBJ:
    obj = dict(items)
    need(len(obj) == len(items), "Duplicate JSON key")
    return obj


def pin(value: Any, label: str) -> str:
    need(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"{label}: SHA-256 required")
    return value


def read(path: Path, expected: str | None = None) -> tuple[Any, str]:
    data = path.read_bytes()
    actual = sha256(data).hexdigest()
    if expected is not None:
        need(actual == pin(expected, str(path)), f"Byte SHA-256 mismatch: {path}")
    return json.loads(data, object_pairs_hook=unique_json, parse_constant=lambda x: finite(x, "JSON")), actual


def finite(value: Any, label: str) -> float:
    need(type(value) in (float, int) and isfinite(value), f"{label}: finite numeric value required")
    return float(value)


def text(value: Any, label: str) -> str:
    need(isinstance(value, str) and bool(value.strip()), f"{label}: missing text")
    return value


def ident(value: Any, label: str, expr: str = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}") -> str:
    need(isinstance(value, str) and re.fullmatch(expr, value) is not None, f"{label}: invalid identifier")
    return value


def token_id(value: Any) -> str:
    return ident(value, "token", r"[0-9a-f]{16}")


def ids(values: Any, parse: Callable) -> set[str]:
    need(isinstance(values, list) and bool(values), "Nonempty ID list required")
    result = {parse(value) for value in values}
    need(len(values) == len(result), "Duplicate ID")
    return result


def code(revision: str, diff_sha: str) -> OBJ:
    return {"revision": ident(revision, "revision", r"[0-9a-f]{40}"), "scoped_diff_sha256": pin(diff_sha, "diff"),
            "files": {n: night.file_sha256(Path(__file__).with_name(n)) for n in CODE_FILES}}


def attached_code(expected: OBJ) -> None:
    need(code(expected["revision"], expected["scoped_diff_sha256"]) == expected, "Executing source identity mismatch")


def scene_record(row: Any) -> OBJ:
    exact(row, SCENE, "causal scene")
    token_id(row["token"])
    ident(row["session_id"], "session")
    pin(row["cache_sha256"], "cache")
    need(0.5 <= finite(row["speed_mps"], "initial speed") <= straight.SPEED_CAP, "Invalid initial speed")
    need(isinstance(row["leaders"], list), "Invalid initial leaders")
    for lead in row["leaders"]:
        exact(lead, ("gap_m", "speed_mps"), "initial leader")
        need(all(finite(v, "initial leader") >= 0 for v in lead.values()), "Invalid initial leader")
    bounds = exact(row["ego_bounds"], ("front_m", "rear_m"), "ego bounds")
    need(finite(bounds["front_m"], "ego front") > finite(bounds["rear_m"], "ego rear"), "Invalid ego footprint")
    need(isinstance(row["corridor_objects"], list), "Invalid corridor objects")
    numeric = ("x_m", "y_m", "heading_rad", "length_m", "width_m", "vx_mps", "vy_mps",
               "lateral_reach_m", "half_length_along_path_m")
    for actor in row["corridor_objects"]:
        exact(actor, ("track_token", "type", *numeric, "projected_lateral_overlap"), "initial corridor actor")
        ident(actor["track_token"], "track token")
        ident(actor["type"], "object type")
        for key in numeric:
            finite(actor[key], key)
        need(actor["length_m"] > 0 and actor["width_m"] > 0 and actor["lateral_reach_m"] > 0
             and actor["half_length_along_path_m"] > 0 and type(actor["projected_lateral_overlap"]) is bool,
             "Invalid corridor actor extent")
    return row


def binding(scene: OBJ) -> OBJ:
    return {**{k: scene[k] for k in SCENE[:3]},
            "observation_sha256": straight.digest({k: scene[k] for k in CAUSAL[:2]}),
            "geometry_sha256": straight.digest({k: scene[k] for k in CAUSAL[2:]})}


def select(pack: OBJ, known_archive: OBJ) -> OBJ:
    exact(pack, ("experiment_id", "dataset", "frame", "exclusions", "eligible_scenes"), "steward export")
    need(pack["experiment_id"] == EXPERIMENT, "Wrong steward export experiment")
    dataset = exact(pack["dataset"], ("source_id", "version", "sha256"), "dataset")
    ident(dataset["source_id"], "source")
    text(dataset["version"], "dataset version")
    pin(dataset["sha256"], "dataset")
    frame = exact(pack["frame"], ("frame_id", "universe_sha256", "adapter_sha256", "filter_sha256"), "frame")
    ident(frame["frame_id"], "frame ID")
    for name in ("universe_sha256", "adapter_sha256", "filter_sha256"):
        pin(frame[name], name)
    need(frame["filter_sha256"] == night.file_sha256(Path(straight.__file__)), "Not the pinned original straight filter")
    exclusions = exact(pack["exclusions"], EXCLUSIONS, "exclusions")
    all_tokens, all_sessions = set(), set()
    archived = set()
    for name in EXCLUSIONS:
        group = exact(exclusions[name], ("evidence_sha256", "tokens", "session_ids"), name)
        pin(group["evidence_sha256"], name)
        tokens = ids(group["tokens"], token_id)
        sessions = ids(group["session_ids"], lambda v: ident(v, name))
        if name == "archive32":
            need(len(sessions) == 32 and set(known_archive) <= tokens and set(known_archive.values()) <= sessions,
                 "32-drive exclusion inventory misses the verified archived subset or count")
            archived = sessions
        elif name == "reserve4":
            need(len(sessions) == 4 and not archived & sessions and not all_tokens & tokens,
                 "Four-drive prior reserve must be separately excluded")
        all_tokens.update(tokens)
        all_sessions.update(sessions)
    groups = defaultdict(list)
    seen = set()
    need(isinstance(pack["eligible_scenes"], list), "Missing eligible frame")
    for value in pack["eligible_scenes"]:
        row = scene_record(value)
        t, s = row["token"], row["session_id"]
        need(t not in seen, f"Duplicate frame token: {t}")
        need(t not in all_tokens and s not in all_sessions, f"Eligibility frame conflicts with exclusions: {t}/{s}")
        seen.add(t)
        groups[s].append(row)
    population = sorted(groups)
    need(len(population) >= N_DRIVES, f"Need {N_DRIVES} eligible source drives, got {len(population)}")
    chosen = sorted(sorted(population, key=lambda s: (sha256(f"{DRAW_SEED}:{s}".encode("ascii")).digest(), s))[:N_DRIVES])
    return {"eligible_drive_count": len(population), "eligible_drive_ids_sha256": straight.digest(population),
            "eligible_scenes_sha256": straight.digest(sorted(pack["eligible_scenes"], key=lambda r: r["token"])),
            "session_ids": chosen,
            "scenes": [r for s in chosen for r in sorted(groups[s], key=lambda r: r["token"])]}


def freeze(pack_path: Path, attestation_path: Path, attestation_sha: str, operator: str, smoke_path: Path,
           code_id: OBJ, attempt: str) -> OBJ:
    approval, approval_sha = read(attestation_path, attestation_sha)
    exact(approval, ("experiment_id", "steward", "independence_owner", "reviewed_by", "source_approval_ref",
                     "independence_approval_ref", "exclusion_review_ref", "preoutcome_review_ref",
                     "precision_decision_ref", "capacity_decision_ref", "max_scenes", "msu_wall_seconds",
                     "astra_wall_seconds", "export_sha256", "archive_score_sha256", "code"), "offline attestation")
    need(approval["experiment_id"] == EXPERIMENT and approval["reviewed_by"] == text(operator, "operator")
         and approval["code"] == code_id, "Attestation does not bind this operator/protocol/code")
    for key in ("steward", "independence_owner", "source_approval_ref", "independence_approval_ref",
                "exclusion_review_ref", "preoutcome_review_ref", "precision_decision_ref", "capacity_decision_ref"):
        text(approval[key], key)
    for key in ("max_scenes", "msu_wall_seconds", "astra_wall_seconds"):
        need(type(approval[key]) is int and approval[key] > 0, f"Missing positive approved limit: {key}")
    pack, pack_sha = read(pack_path, pin(approval["export_sha256"], "export pin"))
    smoke, smoke_sha = read(smoke_path, pin(approval["archive_score_sha256"], "archive score pin"))
    need(smoke["experiment_id"] == EXPERIMENT and smoke["stage"] == "archive-score"
         and smoke["status"] == "smoke_verified" and smoke["code"] == code_id
         and smoke["prior_comparison_sha256"] == bc.PRIOR_COMPARISON and smoke["token"] == bc.TARGET,
         "This code has no verified archived A score smoke")
    known = smoke["known_archive"]
    need(bc.TARGET in known and bc.OTHER in known and len(known) == 23 and len(set(known.values())) == 14,
         "The verified archived 23-token/14-drive subset is missing")
    chosen = select(pack, known)
    need(len(chosen["scenes"]) <= approval["max_scenes"], "Full frozen cohort exceeds approved capacity; do not redraw")
    return {"experiment_id": EXPERIMENT, "stage": "freeze", "attempt": attempt, "policy": POLICY, "code": code_id,
            "export_sha256": pack_sha, "attestation_sha256": approval_sha, "archive_score_sha256": smoke_sha,
            "archive_runtime": smoke["runtime"], "scorer": smoke["scorer"], "dataset": pack["dataset"],
            "frame": pack["frame"], "exclusion_inventory_sha256": straight.digest(pack["exclusions"]),
            "limits": {k: approval[k] for k in ("max_scenes", "msu_wall_seconds", "astra_wall_seconds")}, **chosen}


def load_plan(path: Path, sha: str) -> tuple[OBJ, str]:
    plan, plan_sha = read(path, sha)
    need(plan["experiment_id"] == EXPERIMENT and plan["stage"] == "freeze" and plan["policy"] == POLICY,
         "Wrong experiment/policy in freeze")
    attached_code(plan["code"])
    scenes = [scene_record(r) for r in plan["scenes"]]
    sessions = sorted({r["session_id"] for r in scenes})
    need(len(sessions) == N_DRIVES and sessions == plan["session_ids"]
         and len({r["token"] for r in scenes}) == len(scenes) <= plan["limits"]["max_scenes"]
         and scenes == sorted(scenes, key=lambda r: (r["session_id"], r["token"])), "Invalid frozen membership")
    return plan, plan_sha


def ordered_sha(request: OBJ) -> str:
    return sha256(night.request_json(request).encode("utf-8")).hexdigest()


def compact(case: OBJ, gates: list[OBJ]) -> OBJ:
    need(len(case["decisions"]) == len(gates) == len(TIMES), "Expected eight matched decisions and gates")
    return {"token": case["token"], "poses": case["poses"], "pose_sha256": straight.digest(case["poses"]),
             "decisions": [{"time_s": d["time_s"], "action": d["action"], "request_sha256": ordered_sha(d["request"]),
                            **g} for d, g in zip(case["decisions"], gates)]}


def make_pair(record: OBJ, model: Any, reset_seed: Callable) -> OBJ:
    scene = {k: record[k] for k in ("token", *CAUSAL)}
    result = {}
    for arm in ARMS:
        reset_seed(SEED)
        telemetry = []

        def request_for_arm(augmented, reason):
            conditional, audit = bc.counterfactual(augmented, reason, model.model)
            base = night.without_brake_gap(augmented)
            telemetry.append({"reason": reason, "base_sha256": ordered_sha(base), "aug_sha256": ordered_sha(augmented),
                              "base_tokens": (len(night.no_brake_gap_truncation(model.model, base))
                                              if reason == "closing" else audit["c_tokens"]),
                              "aug_tokens": audit["b_tokens"]})
            return conditional if arm == "c" else base

        if arm == "a":
            # Run the actual, untouched control. Replay its exact simulated states only for tokenizer parity.
            case = straight.build_plan(scene, model.choose, *SETUP)
            actions = iter(d["action"] for d in case["decisions"])
            replay = straight.build_plan(scene, lambda _: (next(actions), None), *SETUP,
                                         lead_brake_gap=True, brake_gap_request=request_for_arm)
            need(case["poses"] == replay["poses"] and len(case["decisions"]) == len(replay["decisions"]) == len(TIMES)
                 and all(
                night.request_json(a["request"]) == night.request_json(b["request"])
                for a, b in zip(case["decisions"], replay["decisions"])), "Baseline request/pose parity failed")
        else:
            case = straight.build_plan(scene, model.choose, *SETUP, lead_brake_gap=True,
                                       brake_gap_request=request_for_arm)
        night.check_brake_gap_pose_rows(case)
        result[arm] = compact(case, telemetry)
    return {"scene": binding(record), **result}


def check_pair(pair: OBJ, scene: OBJ) -> None:
    exact(pair, ("scene", *ARMS), "trajectory pair")
    t = scene["token"]
    need(pair["scene"] == binding(scene), f"Trajectory inputs differ from manifest: {t}")
    for arm in ARMS:
        case = exact(pair[arm], ("token", "poses", "pose_sha256", "decisions"), "trajectory")
        need(case["token"] == t and case["pose_sha256"] == straight.digest(case["poses"]), f"Pose binding: {t}/{arm}")
        night.check_brake_gap_pose_rows(case)
        trace = case["decisions"]
        need([d["time_s"] for d in trace] == TIMES, f"Expected eight decision times: {t}/{arm}")
        decisions = iter(trace)

        def check_gate(augmented, reason):
            step = next(decisions)
            exact(step, ("time_s", "action", "reason", "request_sha256", "base_sha256", "aug_sha256",
                         "base_tokens", "aug_tokens"), "decision receipt")
            base = night.without_brake_gap(augmented)
            request = augmented if arm == "c" and reason == "closing" else base
            need(reason == step["reason"] and ordered_sha(base) == step["base_sha256"]
                 and ordered_sha(augmented) == step["aug_sha256"] and ordered_sha(request) == step["request_sha256"]
                 and all(type(step[k]) is int and 0 < step[k] < 2048 for k in ("base_tokens", "aug_tokens")),
                 f"Gate/request replay mismatch: {t}/{arm}/{step['time_s']}")
            return request

        actions = iter(d["action"] for d in trace)
        replay = straight.build_plan({k: scene[k] for k in ("token", *CAUSAL)}, lambda _: (next(actions), None), *SETUP,
                                     lead_brake_gap=True, brake_gap_request=check_gate)
        need(replay["poses"] == case["poses"], f"Decision/pose replay mismatch: {t}/{arm}")
    first = [pair[a]["decisions"][0] for a in ARMS]
    need(all(first[0][k] == first[1][k] for k in ("reason", "base_sha256", "aug_sha256", "base_tokens", "aug_tokens")),
         f"First counterfactual states differ: {t}")


def budget(started: float, seconds: float) -> None:
    need(perf_counter() - started < seconds, "Approved wall-clock limit reached")


def predict_folder(out: Path, scenes: list[OBJ], lineage: OBJ, runtime: OBJ, reset_seed: Callable, load: Callable,
                   seconds: float) -> OBJ:
    started = perf_counter()
    budget(started, seconds)
    (out / "pairs").mkdir()
    night.write_brake_gap_artifact(out / "runtime.json", runtime)
    model = load()
    hashes = {}
    for scene in scenes:
        budget(started, seconds)
        token = scene["token"]
        print(f"Predicting {token}", flush=True)
        pair = make_pair(scene, model, reset_seed)
        check_pair(pair, scene)
        path = out / "pairs" / f"{token}.json"
        night.write_brake_gap_artifact(path, pair)
        hashes[token] = night.file_sha256(path)
        budget(started, seconds)
    result = {"experiment_id": EXPERIMENT, **lineage, "status": "predicted", "runtime": runtime, "pairs": hashes,
              "worker_wall_seconds": perf_counter() - started}
    night.write_brake_gap_artifact(out / "prediction.json", result)
    return result


def predicted(path: Path, expected_sha: str, scenes: list[OBJ]) -> tuple[OBJ, str]:
    header, actual_sha = read(path, expected_sha)
    tokens = {s["token"] for s in scenes}
    need(header["experiment_id"] == EXPERIMENT and header["status"] == "predicted" and set(header["pairs"]) == tokens
         and {p.name for p in (path.parent / "pairs").iterdir()} == {f"{t}.json" for t in tokens},
         "Prediction membership mismatch or incomplete prediction")
    return header, actual_sha


def matching_runtime(actual: OBJ, smoke: OBJ) -> None:
    keys = ("identity", "gpu", "gpu_capability", "seed_per_arm")
    need(all(actual[k] == smoke[k] for k in keys) and actual["seed_per_arm"] == SEED
         and actual["slurm_job_id"] != smoke["slurm_job_id"], "MSU runtime differs from archived smoke or reuses its job")


def raw(metrics: Any, label: str) -> OBJ:
    exact(metrics, METRICS, label)
    values = {k: finite(metrics[k], label + "." + k) for k in METRICS}
    need(all(0 <= v <= 1 for v in values.values()), f"Raw range violation: {label}")
    v = values
    pdm = v[NC] * v[DAC] * v[DDC] * v[TL] * (5*v[EP] + 5*v[TTC] + 2*v[LK] + 2*v[HC]) / 14
    need(isclose(v[PDM], pdm, abs_tol=TOL, rel_tol=0.0), f"Raw PDM formula mismatch: {label}")
    return {**v, PDM: pdm}


def score_folder(out: Path, scenes: list[OBJ], prediction: Path, pred: OBJ, pred_sha: str, score_identity: OBJ,
                 evaluate_scene: Callable, seconds: float, verify: Callable | None = None) -> OBJ:
    started = perf_counter()
    budget(started, seconds)
    (out / "pairs").mkdir()
    rows = {}
    for scene in scenes:
        budget(started, seconds)
        token = scene["token"]
        print(f"Scoring {token}", flush=True)
        case, case_sha = read(prediction.parent / "pairs" / f"{token}.json", pin(pred["pairs"][token], token))
        check_pair(case, scene)
        evaluate = evaluate_scene(scene)
        row = {"scene": binding(scene), "prediction_pair_sha256": case_sha,
               **{arm: raw(evaluate(case[arm]), f"{token}/{arm}") for arm in ARMS}}
        if verify is not None:
            verify(case, row)
        if case["a"]["pose_sha256"] == case["c"]["pose_sha256"]:
            need(row["a"] == row["c"], f"Same poses yielded different raw metrics: {token}")
        path = out / "pairs" / f"{token}.json"
        night.write_brake_gap_artifact(path, row)
        rows[token] = night.file_sha256(path)
        budget(started, seconds)
    return {"experiment_id": EXPERIMENT, "prediction_sha256": pred_sha, "runtime": pred["runtime"],
            "scorer": score_identity, "pairs": rows, "worker_wall_seconds": perf_counter() - started}


def read_signed_pair(path: Path, expected: str, scene: OBJ) -> tuple[OBJ, str]:
    pair, actual = read(path, pin(expected, scene["token"]))
    check_pair(pair, scene)
    return pair, actual


def joined(outcome: Path, outcome_sha: str, prediction: Path, pred: OBJ, pred_sha: str,
           plan: OBJ, plan_sha: str) -> list[OBJ]:
    scored, _ = read(outcome, outcome_sha)
    scenes = plan["scenes"]
    tokens = {row["token"] for row in scenes}
    need(scored["experiment_id"] == EXPERIMENT and scored["stage"] == "score" and scored["status"] == "scored"
         and scored["manifest_sha256"] == plan_sha and scored["prediction_sha256"] == pred_sha
         and scored["runtime"] == pred["runtime"] and scored["scorer"] == plan["scorer"]
         and set(scored["pairs"]) == tokens
         and {p.name for p in (outcome.parent / "pairs").iterdir()} == {f"{t}.json" for t in tokens},
         "Score membership, lineage or completion mismatch")
    rows = []
    for scene in scenes:
        token = scene["token"]
        pair, pair_sha = read_signed_pair(prediction.parent / "pairs" / f"{token}.json", pred["pairs"][token], scene)
        row, _ = read(outcome.parent / "pairs" / f"{token}.json", pin(scored["pairs"][token], token))
        exact(row, ("scene", "prediction_pair_sha256", *ARMS), "score pair")
        need(row["scene"] == pair["scene"] and row["prediction_pair_sha256"] == pair_sha,
             f"Scored inputs/trajectory mismatch: {token}")
        a, c = (raw(row[arm], f"{token}/{arm}") for arm in ARMS)
        if pair["a"]["pose_sha256"] == pair["c"]["pose_sha256"]:
            need(a == c, f"Same poses yielded different raw metrics: {token}")
        rows.append({"token": token, "session_id": scene["session_id"], "delta": {k: c[k] - a[k] for k in METRICS},
                     "raw_a": a, "raw_c": c, "distance_delta_m": pair["c"]["poses"][-1][0] - pair["a"]["poses"][-1][0],
                     "gates": {arm: Counter(d["reason"] for d in pair[arm]["decisions"]) for arm in ARMS}})
    return rows


def analyse(rows: list[OBJ]) -> OBJ:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["session_id"]].append(row)
    need(len(grouped) == N_DRIVES, "Analysis requires 120 complete drives")
    drive_ids = sorted(grouped)
    per_drive = {s: {"scenes": len(grouped[s]), "mean_delta": {k: mean(r["delta"][k] for r in grouped[s]) for k in METRICS},
                     "mean_distance_delta_m": mean(r["distance_delta_m"] for r in grouped[s])} for s in drive_ids}
    sums = [(len(grouped[s]), *(fsum(r["delta"][k] for r in grouped[s]) for k in ENDPOINTS)) for s in drive_ids]
    totals = [fsum(g[i] for g in sums) for i in (1, 2)]
    losses = [{"token": r["token"], "session_id": r["session_id"], "metric": k,
               "a": r["raw_a"][k], "c": r["raw_c"][k], "delta": r["delta"][k]}
              for r in rows for k in PROTECTED if r["delta"][k] < -TOL]
    wlt = {}
    for k in ENDPOINTS:
        w, l = sum(r["delta"][k] > TOL for r in rows), sum(r["delta"][k] < -TOL for r in rows)
        wlt[k] = {"wins": w, "losses": l, "ties": len(rows) - w - l}
    coverage = {a: {g: sum(r["gates"][a].get(g, 0) for r in rows) for g in bc.GATES} for a in ARMS}
    result = {"units": {"metric": "fraction; 0.01 = 1 pp", "distance": "meters"}, "scenes": len(rows),
              "drives": N_DRIVES, "mean_delta": {k: mean(r["delta"][k] for r in rows) for k in METRICS},
              "mean_distance_delta_m": mean(r["distance_delta_m"] for r in rows), "win_loss_tie": wlt,
               "per_drive": per_drive,
               "equal_drive_mean_delta": {k: mean(d["mean_delta"][k] for d in per_drive.values()) for k in METRICS},
               "leave_one_drive_out": {
                  s: {k: (v - g[i]) / (len(rows) - g[0]) for i, (k, v) in enumerate(zip(ENDPOINTS, totals), 1)}
                  for s, g in zip(drive_ids, sums)}, "gate_counts": coverage,
              "gate_closing_fraction": {a: coverage[a]["closing"] / (8 * len(rows)) for a in ARMS},
              "protected_losses": losses, "bootstrap": None}
    if losses:
        return {**result, "status": "rejected", "failed_gates": ["protected_regression"]}
    result["zero_events"] = {"upper_95": 1 - 0.05 ** (1 / N_DRIVES),
                             "conditional_on": "independent equal-risk drives and appropriate sampling; not safety or power"}
    degenerate = [k for k in ENDPOINTS if len({d["mean_delta"][k] for d in per_drive.values()}) == 1]
    if degenerate:
        return {**result, "status": "inconclusive", "failed_gates": ["constant_drive_means"], "constant": degenerate}
    samples = ([], [])
    rng = Random(SEED)
    for _ in range(REPS):
        draw = rng.choices(sums, k=N_DRIVES)
        n = sum(g[0] for g in draw)
        for i, series in enumerate(samples, 1):
            series.append(fsum(g[i] for g in draw) / n)
    lower = {k: sorted(values)[RANK] for k, values in zip(ENDPOINTS, samples)}
    thresholds = {PDM: 0.0, EP: -0.02}
    failed = [k for k in ENDPOINTS if lower[k] <= thresholds[k]]
    return {**result, "status": "rejected" if failed else "passed", "failed_gates": failed,
            "bootstrap": {"seed": SEED, "replicates": REPS, "nearest_rank_zero_based": RANK,
                          "lower": lower, "strictly_above": thresholds}}


def canonical(path: Path, directory: bool = False) -> Path:
    need(path.is_absolute() and path.resolve(strict=True) == path and (path.is_dir() if directory else path.is_file()),
         f"Explicit canonical {'directory' if directory else 'file'} required: {path}")
    return path


def cache_paths(path: Path, sha: str, root: Path, scenes: list[OBJ]) -> tuple[OBJ, str]:
    data, index_sha = read(canonical(path), sha)
    need(isinstance(data, dict) and set(data) == {r["token"] for r in scenes}, "Astra cache index membership mismatch")
    canonical(root, directory=True)
    result = {}
    for token, rel in data.items():
        parts = PurePosixPath(text(rel, "cache relative path"))
        need(not parts.is_absolute() and parts.as_posix() == rel and len(parts.parts) > 0
             and all(re.fullmatch(r"[A-Za-z0-9_.-]+", p) and p not in (".", "..") for p in parts.parts),
             f"Invalid Astra cache path: {token}")
        result[token] = root.joinpath(*parts.parts)
    need(len(set(result.values())) == len(result), "Astra cache index aliases tokens")
    return result, index_sha


def astra_evaluator(args: argparse.Namespace, scenes: list[OBJ], expected: OBJ) -> tuple[OBJ, Callable, OBJ]:
    from rollout_replay import config_core

    repo, tools = canonical(args.navsim_repo, True), canonical(args.tools, True)
    spec = find_spec("navsim")
    replay_spec = find_spec("rollout_replay")
    need(spec is not None and spec.submodule_search_locations is not None
         and [Path(p).resolve() for p in spec.submodule_search_locations] == [repo / "navsim"]
         and replay_spec is not None and replay_spec.origin is not None
         and Path(replay_spec.origin).resolve() == tools / "rollout_replay.py", "Astra PYTHONPATH does not match pins")
    template, template_sha = read(canonical(args.template), args.template_sha256)
    identity = {"config_sha256": straight.digest(config_core(template)), "navsim_sha256": night.tree_sha256(repo / "navsim"),
                "tools_sha256": night.tree_sha256(tools), "python": platform.python_version(),
                "versions": {n: version(n) for n in ("numpy", "hydra-core", "shapely", "nuplan-devkit")}}
    need(identity == expected, "Astra scorer/config/tools/Python differs from archived attempt-002")
    paths, index_sha = cache_paths(args.cache_index, args.cache_index_sha256, args.cache_root, scenes)

    def for_scene(scene):
        token = scene["token"]
        path = canonical(paths[token])
        need(night.file_sha256(path) == scene["cache_sha256"], f"Cache SHA-256 mismatch: {token}")
        # Only operator-approved trusted pickle caches; future tracks remain on this scorer host.
        info = {"path": path}
        cache = straight.load_cache(info)
        obs, reason = straight.straight_observation(cache)
        need(reason is None and obs == {k: scene[k] for k in CAUSAL[:2]}
             and night.corridor_objects(cache) == scene["corridor_objects"]
             and night.ego_bounds(cache) == scene["ego_bounds"], f"Causal export/cache mismatch: {token}: {reason}")
        return lambda case: straight.raw_score(info, straight.checked_straight_poses(case), template, METRICS)

    return identity, for_scene, {"template_sha256": template_sha, "cache_index_sha256": index_sha}


def archive_predict(args: argparse.Namespace, out: Path) -> None:
    archive, prior_path = canonical(args.archive, True), canonical(args.prior_attempt, True)
    need(archive.name == night.PROTOCOL and prior_path.name == "attempt-002"
         and prior_path.parent.name == night.BRAKE_GAP_PAIRED_EXPERIMENT, "Original archive/attempt-002 paths required")
    code_id = code(args.revision, args.diff_sha256)
    bound, plan, _ = night.brake_gap_archive(archive, verify_traces=True)
    prior = bc.prior_reference(prior_path, bound, plan)
    report, _ = read(prior_path / "comparison.json")
    need(straight.digest(report) == bc.PRIOR_COMPARISON, "Attempt-002 A report changed")
    record = next(s for s in plan["scenes"] if s["token"] == bc.TARGET)
    scene = scene_record({k: record[k] for k in SCENE})
    ref_a = next(r["a"] for r in report["cases"] if r["token"] == bc.TARGET)
    raw(ref_a, "attempt-002 reference A")
    runtime, reset = night.current_brake_gap_runtime(args)
    need(runtime["identity"] == prior["runtime_identity"], "MSU checkpoint/runtime differs from attempt-002")
    from .driver import DrexMax

    lineage = {"stage": "archive-predict", "code": code_id, "archive_scene": scene,
               "known_archive": {s["token"]: s["session_id"] for s in plan["scenes"]},
               "prior_comparison_sha256": bc.PRIOR_COMPARISON, "scorer": prior["scorer"], "reference_a": ref_a}
    predict_folder(out, [scene], lineage, runtime, reset, lambda: DrexMax(args.weights), 600)


def archive_score(args: argparse.Namespace, out: Path) -> None:
    header, _ = read(args.prediction, args.prediction_sha256)
    scene = scene_record(header["archive_scene"])
    pred, pred_sha = predicted(args.prediction, args.prediction_sha256, [scene])
    need(pred["stage"] == "archive-predict" and scene["token"] == bc.TARGET
         and pred["prior_comparison_sha256"] == bc.PRIOR_COMPARISON
         and len(pred["known_archive"]) == 23 and len(set(pred["known_archive"].values())) == 14,
         "Not the frozen archived smoke input")
    attached_code(pred["code"])
    scorer, evaluate, inputs = astra_evaluator(args, [scene], pred["scorer"])

    def verify(_case, row):
        night.check_brake_gap_metrics(row["a"], pred["reference_a"], bc.TARGET, "attempt-002 A")
        need(all(row["a"][k] == pred["reference_a"][k] for k in night.RISK_METRICS), "Archived A mismatch on NC/TTC")

    result = score_folder(out, [scene], args.prediction, pred, pred_sha, scorer, evaluate, 600, verify)
    night.write_brake_gap_artifact(out / "score.json", {**result, **inputs, "stage": "archive-score",
                                  "status": "smoke_verified", "code": pred["code"], "token": bc.TARGET,
                                  "known_archive": pred["known_archive"], "prior_comparison_sha256": bc.PRIOR_COMPARISON})


def require_prediction(args: argparse.Namespace, plan: OBJ, plan_sha: str) -> tuple[OBJ, str]:
    pred, pred_sha = predicted(args.prediction, args.prediction_sha256, plan["scenes"])
    need(pred["stage"] == "predict" and pred["manifest_sha256"] == plan_sha, "Prediction not from this frozen manifest")
    matching_runtime(pred["runtime"], plan["archive_runtime"])
    return pred, pred_sha


def execute(args: argparse.Namespace, out: Path) -> int:
    started = perf_counter()
    stage = args.stage
    if stage == "archive-predict":
        archive_predict(args, out)
    elif stage == "archive-score":
        archive_score(args, out)
    elif stage == "freeze":
        plan = freeze(args.export, args.attestation, args.attestation_sha256, args.operator, args.archive_score,
                      code(args.revision, args.diff_sha256), out.name)
        night.write_brake_gap_artifact(out / "manifest.json", plan)
    else:
        plan, plan_sha = load_plan(args.manifest, args.manifest_sha256)
        need(plan["attempt"] == out.name, "New stage must use the frozen attempt ID under its own role parent")
        if stage == "predict":
            runtime, reset = night.current_brake_gap_runtime(args)
            matching_runtime(runtime, plan["archive_runtime"])
            from .driver import DrexMax

            predict_folder(out, plan["scenes"], {"stage": "predict", "manifest_sha256": plan_sha}, runtime, reset,
                           lambda: DrexMax(args.weights), plan["limits"]["msu_wall_seconds"] - (perf_counter() - started))
        else:
            pred, pred_sha = require_prediction(args, plan, plan_sha)
            if stage == "score":
                scorer, evaluate, inputs = astra_evaluator(args, plan["scenes"], plan["scorer"])
                result = score_folder(out, plan["scenes"], args.prediction, pred, pred_sha, scorer, evaluate,
                                      plan["limits"]["astra_wall_seconds"] - (perf_counter() - started))
                night.write_brake_gap_artifact(out / "score.json", {**result, **inputs, "stage": stage, "status": "scored",
                                              "manifest_sha256": plan_sha})
            else:
                rows = joined(args.scores, args.scores_sha256, args.prediction, pred, pred_sha, plan, plan_sha)
                result = analyse(rows)
                night.write_brake_gap_artifact(out / "report.json", {**result, "experiment_id": EXPERIMENT,
                                              "manifest_sha256": plan_sha, "prediction_sha256": pred_sha,
                                              "score_sha256": args.scores_sha256, "cases": rows})
                print(result["status"], flush=True)
                return {"passed": 0, "rejected": 3, "inconclusive": 4}[result["status"]]
    print(f"{stage} complete: {out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subs = parser.add_subparsers(dest="stage", required=True)
    for stage, paths, strings in (
        ("archive-predict", "archive prior-attempt weights upstream", "revision diff-sha256"),
        ("archive-score", "prediction", "prediction-sha256"),
        ("freeze", "export attestation archive-score", "attestation-sha256 operator revision diff-sha256"),
        ("predict", "manifest weights upstream", "manifest-sha256"),
        ("score", "manifest prediction", "manifest-sha256 prediction-sha256"),
        ("validate", "manifest prediction scores", "manifest-sha256 prediction-sha256 scores-sha256"),
    ):
        sub = subs.add_parser(stage, allow_abbrev=False)
        if stage in ("archive-score", "score"):
            paths += " navsim-repo tools template cache-root cache-index"
            strings += " template-sha256 cache-index-sha256"
        sub.add_argument("--out", type=Path, required=True)
        for name in paths.split():
            sub.add_argument("--" + name, type=Path, required=True)
        for name in strings.split():
            sub.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    out = None
    try:
        need(args.out.is_absolute() and args.out.parent.resolve(strict=True) == args.out.parent
             and args.out.parent.is_dir() and re.fullmatch(r"attempt-[0-9]{3,}", args.out.name) is not None,
             "Output must be a new canonical absolute attempt-NNN under its separately allocated role parent")
        for name, path in vars(args).items():
            if isinstance(path, Path) and name != "out":
                resolved = canonical(path, directory=name in
                                     ("archive", "prior_attempt", "weights", "upstream", "navsim_repo", "tools", "cache_root"))
                protected = resolved if resolved.is_dir() else resolved.parent
                need(args.out != protected and protected not in args.out.parents and args.out not in protected.parents,
                     f"Output overlaps input scope: {name}")
            if name.endswith("_sha256"):
                pin(path, name)
        args.out.mkdir(mode=0o700)
        out = args.out
        return execute(args, out)
    except ERRORS as exc:
        failure = f"{type(exc).__name__}: {exc}"
        print(f"incomplete: {failure}", file=sys.stderr, flush=True)
        if out is not None:
            try:
                night.write_brake_gap_artifact(out / "failure.json",
                    {"experiment_id": EXPERIMENT, "stage": args.stage, "status": "incomplete",
                     "first_failed_condition": failure})
            except OSError as log_error:
                print(f"Could not record failure: {log_error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
