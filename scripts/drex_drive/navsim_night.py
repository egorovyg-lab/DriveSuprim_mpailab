"""Run a matched NAVSIM straight-road experiment over the full archived TRAIN inventory."""

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from statistics import mean
import json
import sys

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


def predict_night(root, weights, load=straight.load_drex):
    plan = straight.read(root / "observations.json")
    check_plan(plan)
    model = None
    for name, (prompt, observation, actions) in VARIANTS.items():
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


def archive_evaluator(args, plan):
    import numpy as np
    import real_data
    from rollout_replay import bound_context

    tokens = check_plan(plan)
    archive, candidates, source_id = straight.source_context(args.package, args.source)
    sessions = {row["token"]: row["session_id"] for row in archive}
    if plan["anchor_identity_sha256"] != source_id or any(
            sessions.get(row["token"]) != row["session_id"] for row in plan["scenes"]):
        raise RuntimeError("The selected scenes differ from the archived TRAIN inventory")
    manifest, _, vocab, template, caches = bound_context(args.package, args.repo, [candidates[t][0] for t in tokens])
    keys = ("pdm_score", *real_data.METRICS)

    def for_scene(record):
        token = record["token"]
        cache = caches[token]
        if cache["sha256"] != record["cache_sha256"]:
            raise RuntimeError(f"Metric cache changed for {token}")
        observation, reason = straight.straight_observation(straight.load_cache(cache))
        if reason is not None or observation != {k: record[k] for k in ("speed_mps", "leaders")}:
            raise RuntimeError(f"Current observation changed for {token}")
        sample = real_data.load_inputs(args.package, token, manifest, vocab)
        anchor = next(row for row in candidates[token] if row["candidate_local"] == sample["baseline_local"])
        if list(keys) != anchor["metric_keys"]:
            raise RuntimeError(f"DriveSuprim metric columns changed for {token}")
        baseline = straight.raw_score(cache, vocab[anchor["candidate_vocab"]], template, keys)
        if not np.allclose(list(baseline.values()), anchor["expected_raw"], atol=1e-8, rtol=1e-7):
            raise RuntimeError(f"DriveSuprim score parity failed for {token}")

        def evaluate(prediction):
            result = straight.raw_score(cache, straight.checked_straight_poses(prediction), template, keys)
            if not all(np.isfinite(list(result.values()))):
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


def on_astra(stage, args, payload):
    straight_code = Path(straight.__file__).read_text(encoding="utf-8")
    launcher = ("import sys, types\n"
                "module = types.ModuleType('_drex_night_straight')\n"
                "sys.modules[module.__name__] = module\n"
                f"exec(compile({straight_code!r}, 'navsim_straight.py', 'exec'), module.__dict__)\n"
                + Path(__file__).read_text(encoding="utf-8"))
    return straight.on_astra(stage, args, payload, source_code=launcher)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict", "score", "probes",
                                          "astra-night-prepare", "astra-night-score", "astra-night-probes"))
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
            remote = {"astra-night-prepare": prepare, "astra-night-score": score, "astra-night-probes": score_probes}
            result = remote[args.stage](args, payload)
        print(json.dumps(result, allow_nan=False))
        return 0

    if args.out is None:
        parser.error("--out is required")
    root = args.out
    plan_path = root / "observations.json"
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
    if args.stage == "predict":
        if args.weights is None or not args.weights.is_dir():
            parser.error("--weights must point to the installed Drex Max model")
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
    targets = make_probes(root, plan, report)
    result = on_astra("astra-night-probes", args, {"plan": plan, "targets": targets})
    output = root / "single-edits.json"
    straight.write(output, result)
    print(f"Scored {result['tested_edits']} single-command edits: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
