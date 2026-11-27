"""Closed archived-TRAIN B/C runner; use lead_brake_gap.sh bc-* on MSU270."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import json
from math import isfinite
from pathlib import Path
import re
from statistics import mean
import sys

if __package__:
    from . import navsim_night as night
else:
    import _drex_night as night

straight = night.straight
digest, read, write = straight.digest, straight.read, night.write_brake_gap_artifact
EXPERIMENT = "corridor-preview-lead-brake-gap-conditional-v1"
TARGET, OTHER = night.BRAKE_GAP_TARGET, night.COLLISION_TOKEN
FIELD, SEED, TOL = night.BRAKE_GAP_FIELD, night.BRAKE_GAP_PAIR_SEED, night.EPSILON
ARMS = ("b", "c")
GATES = ("closing", "zero", "opening", "no_leader")
PRIOR_COMPARISON = "30d1d364a321962c0744e46d180fa433472f1950a7b688e770de04cb89e2a2e5"
PRIOR_RUNTIME = "4856a1486e37fea029f28a73f0a103431db246a26fc83ae5a307279bb45cb3b1"
PRIOR_JOB = "301704"
PRIOR_RECEIPT = "852d7e685ee85fec35c7ac33665e9fba774a459e1a594c47d93fbc8f46a6b226"


def prior_reference(root, binding, plan):
    def original(path):
        if path.resolve(strict=True) != path or not path.is_file():
            raise RuntimeError(f"Noncanonical attempt-002 reference: {path}")
        return read(path)

    report, runtime, manifest, receipt, marker = (
        original(root / p) for p in ("comparison.json", "paired-pilot/runtime.json", "manifest.json",
                                     "paired-pilot/prediction-receipt.json", "paired-pilot/smoke.json"))
    tokens = {r["token"]: r["session_id"] for r in plan["scenes"]}
    if (digest(report) != PRIOR_COMPARISON or digest(runtime) != PRIOR_RUNTIME or digest(receipt) != PRIOR_RECEIPT
            or report["prediction_receipt_sha256"] != PRIOR_RECEIPT
            or runtime["slurm_job_id"] != PRIOR_JOB or runtime["seed_per_arm"] != SEED
            or digest(marker) != receipt["pilot_smoke_sha256"] or marker["runtime_sha256"] != PRIOR_RUNTIME
            or marker["status"] != "passed" or marker["attempt"] != str(root)
            or marker["manifest_sha256"] != digest(manifest)
            or manifest["attempt"] != str(root) or manifest["experiment_id"] != night.BRAKE_GAP_PAIRED_EXPERIMENT
            or manifest["binding"] != binding or report["status"] != "pass" or report["unscored_tokens"]
            or len(report["cases"]) != len(tokens)
            or {r["token"]: r["session_id"] for r in report["cases"]} != tokens
            or set(receipt["prediction_sha256"]["b"]) != set(tokens)
            or marker["first_pair_sha256"]["b"] != receipt["prediction_sha256"]["b"][TARGET]):
        raise RuntimeError("Frozen paired attempt-002 reference/lineage mismatch")
    # Optional exact trajectory comparison, never an inference or scoring input.
    trace_path = root / "paired-pilot" / "b" / f"{TARGET}.json"
    target_pose_sha = None
    if trace_path.exists() or trace_path.is_symlink():
        trace = original(trace_path)
        if trace["token"] != TARGET or digest(trace) != receipt["prediction_sha256"]["b"][TARGET]:
            raise RuntimeError("Historical B target trajectory identity mismatch")
        night.check_brake_gap_pose_rows(trace)
        target_pose_sha = digest(trace["poses"])
    return {"attempt": str(root), "comparison_sha256": PRIOR_COMPARISON, "runtime_sha256": PRIOR_RUNTIME,
            "receipt_sha256": PRIOR_RECEIPT, "manifest_sha256": digest(manifest), "job": PRIOR_JOB,
            "target_b_pose_sha256": target_pose_sha, "runtime_identity": runtime["identity"],
            "scorer": report["scorer"], "raw_b": {r["token"]: r["b"] for r in report["cases"]}}


def counterfactual(augmented, reason, tokenizer):
    fields = augmented["state"]["action_effects"]["forecast"]
    radar = augmented["state"]["lead_radar"]
    if (reason not in GATES or (radar is None) != (reason == "no_leader") or not fields
            or tuple(fields) != tuple(augmented["question"]["criteria"])
            or any((row[FIELD] is not None if radar is None else
                    type(row[FIELD]) not in (int, float) or not isfinite(row[FIELD])) for row in fields.values())):
        raise ValueError("Invalid always-on forecast or gate")
    stripped = night.without_brake_gap(augmented)
    conditional = augmented if reason == "closing" else stripped
    # Verify ordered JSON roundtrip with the only allowed edit: dropping each final forecast key.
    restored = json.loads(night.request_json(stripped))
    for action, row in restored["state"]["action_effects"]["forecast"].items():
        row[FIELD] = fields[action][FIELD]
    if night.request_json(restored) != night.request_json(augmented):
        raise RuntimeError("Counterfactual changed request ordering or non-gate fields")
    b_tokens = night.no_brake_gap_truncation(tokenizer, augmented)
    stripped_tokens = night.no_brake_gap_truncation(tokenizer, stripped)
    if b_tokens == stripped_tokens:
        raise RuntimeError("Drex tokenizer erased the brake-gap field")
    return conditional, {"reason": reason, "b_tokens": len(b_tokens),
                         "c_tokens": len(b_tokens) if reason == "closing" else len(stripped_tokens),
                         "b_request_sha256": digest(augmented), "c_request_sha256": digest(conditional)}


def stable_case_sha(case):
    return digest({**case, "decisions": [{k: v for k, v in d.items() if k != "wall_seconds"}
                                         for d in case["decisions"]]})


def validate_case(case, token, arm):
    night.check_brake_gap_pose_rows(case)
    if (case["token"] != token or [d["time_s"] for d in case["decisions"]]
            != [round(i * straight.STEP, 9) for i in range(0, straight.POSES, straight.ACTION_STEPS)]
            or len(case["gate"]) != 8):
        raise RuntimeError(f"Invalid decision/gate trace: {token}/{arm}")
    for decision, gate in zip(case["decisions"], case["gate"]):
        reason, request = gate["reason"], decision["request"]
        forecast = request["state"]["action_effects"]["forecast"]
        expected = arm == "b" or reason == "closing"
        if (reason not in GATES or any((FIELD in row) != expected for row in forecast.values())
                or digest(request) != gate[f"{arm}_request_sha256"]
                or any(not 0 < gate[f"{a}_tokens"] < 2048 for a in ARMS)):
            raise RuntimeError(f"Invalid gate telemetry: {token}/{arm}")


def predict(folder, records, chain, runtime, seed, load, scored_smoke=None):
    folder.mkdir()
    write(folder / "runtime.json", runtime)
    cases = {arm: {} for arm in ARMS}
    for arm in ARMS:
        (folder / arm).mkdir()
    model = load()
    for record in sorted(records, key=lambda r: r["token"] != TARGET):
        token = record["token"]
        # Only frozen initial observations, static geometry and causal simulated ego state.
        scene = {k: record[k] for k in ("token", "speed_mps", "leaders", "corridor_objects", "ego_bounds")}
        for arm in ARMS:
            gate = []

            def request_for_arm(augmented, reason):
                c, audit = counterfactual(augmented, reason, model.model)
                gate.append(audit)
                return augmented if arm == "b" else c

            seed(SEED)
            case = straight.build_plan(scene, model.choose, *night.PREVIEW_SETUPS[night.BRAKE_GAP_CONTROL],
                                       lead_brake_gap=True, brake_gap_request=request_for_arm)
            case["gate"] = gate
            validate_case(case, token, arm)
            write(folder / arm / f"{token}.json", case)
            cases[arm][token] = case
        if cases["b"][token]["gate"][0] != cases["c"][token]["gate"][0]:
            raise RuntimeError(f"Starting states differ: {token}")
        if token == TARGET and scored_smoke is not None:
            replay = {a: stable_case_sha(cases[a][token]) for a in ARMS}
            if replay != scored_smoke["target_replay_sha256"]:
                raise RuntimeError("Pilot first pair does not replay the separately scored B/C smoke")
        print(f"Predicted B/C: {token}", flush=True)
    write(folder / "prediction-receipt.json", prediction_receipt(chain, runtime, cases, scored_smoke))


def prediction_receipt(chain, runtime, cases, scored_smoke):
    return {"status": "predicted", **chain, "runtime_sha256": digest(runtime),
            "scored_smoke_sha256": digest(scored_smoke) if scored_smoke is not None else None,
            "prediction_sha256": {a: {t: digest(c) for t, c in cases[a].items()} for a in ARMS},
            "target_replay_sha256": {a: stable_case_sha(cases[a][TARGET]) for a in ARMS}}


def read_prediction(folder, records, chain, expected_runtime, scored_smoke=None):
    tokens = {r["token"] for r in records}
    cases = {a: {p.stem: read(p) for p in (folder / a).glob("*.json")} for a in ARMS}
    for arm in ARMS:
        if set(cases[arm]) != tokens:
            raise RuntimeError(f"Missing or extra prediction tokens: {arm}")
        for token, case in cases[arm].items():
            validate_case(case, token, arm)
    runtime = read(folder / "runtime.json")
    receipt = read(folder / "prediction-receipt.json")
    if (receipt != prediction_receipt(chain, runtime, cases, scored_smoke)
            or runtime["identity"] != expected_runtime or runtime["seed_per_arm"] != SEED
            or (scored_smoke is not None and receipt["target_replay_sha256"] != scored_smoke["target_replay_sha256"])):
        raise RuntimeError("Prediction receipt/runtime/smoke replay mismatch")
    return cases, receipt


def summary(rows):
    result = night.brake_gap_summary(rows)
    drives = defaultdict(list)
    for row in rows:
        drives[row["session_id"]].append(row)
    progress = [{"session_id": r["session_id"], "delta": {"pdm_score": r["delta"]["ego_progress"]},
                 "distance_delta_m": r["distance_delta_m"]} for r in rows]
    p = night.brake_gap_summary(progress)
    for suffix in ("by_session_pp", "equal_session_pp", "leave_one_session_out_pp", "bootstrap_95_pp"):
        result[f"ego_progress_{suffix}"] = p[f"pdm_{suffix}"]
    result["per_drive"] = {key: {"scenes": len(group),
                                 "delta_pp": {k: 100 * mean(r["delta"][k] for r in group)
                                              for k in rows[0]["delta"]},
                                 "mean_distance_delta_m": mean(r["distance_delta_m"] for r in group)}
                           for key, group in sorted(drives.items())}
    result["gate_coverage"] = {a: {g: sum(r["gate"][a].get(g, 0) for r in rows) for g in GATES} for a in ARMS}
    result["gate_coverage_fraction"] = {a: c["closing"] / sum(c.values()) for a, c in result["gate_coverage"].items()}
    return result


def score_pairs(plan, payload, scorer_identity, for_scene, result):
    smoke = payload["stage"] == "score-smoke"
    records = sorted((r for r in plan["scenes"] if not smoke or r["token"] == TARGET),
                     key=lambda r: (r["token"] != TARGET, r["token"] != OTHER))
    tokens = {r["token"] for r in records}
    result["unscored_tokens"] = [r["token"] for r in records]
    if not records or any(set(payload["pairs"][a]) != tokens for a in ARMS):
        raise RuntimeError("Exact B/C prediction membership required")
    if scorer_identity != payload["expected_scorer"]:
        raise RuntimeError("Astra scorer/config/tools/runtime differ from attempt-002")
    result["scorer"] = scorer_identity
    for record in records:
        token = record["token"]
        result["active_token"] = token
        b, c = (payload["pairs"][a][token] for a in ARMS)
        for case in (b, c):
            if case["token"] != token:
                raise RuntimeError("Wrong compact trajectory token")
            night.check_brake_gap_pose_rows(case)
        ds, evaluate = for_scene(record)
        night.check_brake_gap_metrics(ds, payload["raw_ds"][token], token, "DrSu")
        actual_b = evaluate(b)
        old_b = payload["raw_b"][token]
        night.check_brake_gap_metrics(actual_b, old_b, token, "attempt-002 B")
        if any(actual_b[k] != old_b[k] for k in night.RISK_METRICS):
            raise RuntimeError(f"Attempt-002 B NC/TTC mismatch: {token}")
        if token == TARGET and payload["old_target_pose_sha256"] is not None:
            if digest(b["poses"]) != payload["old_target_pose_sha256"]:
                raise RuntimeError("Regenerated B target trajectory differs from attempt-002")
        actual_c = actual_b if c["poses"] == b["poses"] else evaluate(c)
        night.check_brake_gap_metrics(actual_c, actual_c, token, "C schema")
        if not smoke and token == TARGET:
            for arm, actual in (("b", actual_b), ("c", actual_c)):
                night.check_brake_gap_metrics(actual, payload["smoke_raw"][arm], token, f"scored smoke {arm}")
        delta = {k: actual_c[k] - actual_b[k] for k in actual_b}
        row = {"token": token, "session_id": record["session_id"], "ds": ds, "b": actual_b, "c": actual_c,
               "delta": delta, "distance_delta_m": c["poses"][-1][0] - b["poses"][-1][0],
               "gate": {a: payload["pairs"][a][token]["gate"] for a in ARMS},
               "regressions": [k for k in night.BRAKE_GAP_PROTECTED if delta[k] < -TOL]}
        result["cases"].append(row)
        result["unscored_tokens"].remove(token)
    del result["active_token"]
    if smoke:
        result["status"] = "smoke_verified"
        return
    result["summary"] = summary(result["cases"])
    deltas = result["summary"]["mean_delta_pp"]
    failures = [{"token": r["token"], "reason": "protected_regression", "metrics": r["regressions"]}
                for r in result["cases"] if r["regressions"]]
    failures.extend({"token": r["token"], "reason": "target_nc_ttc"} for r in result["cases"]
                    if r["token"] in (TARGET, OTHER) and any(r["c"][k] != 1.0 for k in night.RISK_METRICS))
    if deltas["ego_progress"] < 1.0:
        failures.append({"reason": "mean_ego_progress_delta_below_1pp"})
    if deltas["pdm_score"] <= -2.0:
        failures.append({"reason": "mean_raw_pdm_delta_not_strictly_above_minus_2pp"})
    result.update(status="rejected" if failures else "pass", scientific_gate_failures=failures)


def scoring_report(args, payload):
    result = {"experiment_id": EXPERIMENT, "split": "archived TRAIN; diagnostic", "stage": payload["stage"],
              "status": "incomplete", "cases": []}
    try:
        plan = night.with_corridor(payload["plan"], payload["export"])
        result["unscored_tokens"] = [r["token"] for r in plan["scenes"]
                                     if payload["stage"] != "score-smoke" or r["token"] == TARGET]
        scorer = night.brake_gap_scorer_identity(args, plan)
        if scorer != payload["expected_scorer"]:
            raise RuntimeError("Scorer lineage changed before scoring")
        for_scene, _ = night.archive_evaluator(args, plan)
        score_pairs(plan, payload, scorer, for_scene, result)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, OverflowError) as exc:
        result["first_failed_condition"] = f"{type(exc).__name__}: {exc}"
    return result


def score_stage(root, stage, records, chain, prior, archive_payload, bridge):
    smoke = stage == "score-smoke"
    folder = root / ("bc-smoke" if smoke else "bc-pilot")
    output = root / ("smoke-score.json" if smoke else "comparison.json")
    target_records = [r for r in records if r["token"] == TARGET]
    result = {"experiment_id": EXPERIMENT, "stage": stage, "status": "incomplete", "cases": [],
              "unscored_tokens": [r["token"] for r in records if not smoke or r["token"] == TARGET], **chain}
    try:
        scored_smoke = None if smoke else require_scored_smoke(root, target_records, chain, prior)
        cases, receipt = read_prediction(folder, target_records if smoke else records, chain,
                                         prior["runtime_identity"], scored_smoke)
        compact = {a: {t: {"token": t, "poses": c["poses"], "gate": dict(Counter(g["reason"] for g in c["gate"]))}
                       for t, c in cases[a].items()} for a in ARMS}
        payload = {"stage": stage, "plan": archive_payload["plan"], "export": archive_payload["export"],
                   "pairs": compact, "expected_scorer": prior["scorer"], "raw_b": prior["raw_b"],
                   "old_target_pose_sha256": prior["target_b_pose_sha256"],
                   "raw_ds": {t: r["ds"] for t, r in archive_payload["reference"].items()},
                   "smoke_raw": None if smoke else scored_smoke["cases"][0]}
        remote = bridge(payload)
        status = remote["status"]
        pairs = remote["cases"]
        expected = {r["token"]: r["session_id"] for r in (target_records if smoke else records)}
        if (remote["experiment_id"] != EXPERIMENT or remote["stage"] != stage
                or status not in (("incomplete", "smoke_verified") if smoke else ("incomplete", "pass", "rejected"))
                or (status != "incomplete" and (remote["scorer"] != prior["scorer"] or remote["unscored_tokens"]
                    or len(pairs) != len(expected) or {r["token"]: r["session_id"] for r in pairs} != expected))):
            raise RuntimeError("Unexpected or partial Astra response")
        result.update(remote, **chain, prediction_receipt_sha256=digest(receipt),
                      target_replay_sha256=receipt["target_replay_sha256"],
                      scored_smoke_sha256=receipt["scored_smoke_sha256"])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, SystemExit, OverflowError) as exc:
        result["first_failed_condition"] = f"{type(exc).__name__}: {exc}"
    write(output, result)
    print(f"{result['status']}: {output}", flush=True)
    return {"smoke_verified": 0, "pass": 0, "rejected": 3}.get(result["status"], 2)


def require_scored_smoke(root, target_records, chain, prior):
    scored = read(root / "smoke-score.json")
    _, receipt = read_prediction(root / "bc-smoke", target_records, chain, prior["runtime_identity"])
    if (scored["status"] != "smoke_verified" or scored["experiment_id"] != EXPERIMENT
            or scored["stage"] != "score-smoke" or any(scored[k] != v for k, v in chain.items())
            or scored["scorer"] != prior["scorer"] or scored["unscored_tokens"]
            or len(scored["cases"]) != 1 or scored["cases"][0]["token"] != TARGET
            or scored["cases"][0]["session_id"] != target_records[0]["session_id"]
            or scored["prediction_receipt_sha256"] != digest(receipt)
            or scored["target_replay_sha256"] != receipt["target_replay_sha256"]):
        raise RuntimeError("Astra smoke has not scored and passed for this exact B/C prediction receipt")
    row = scored["cases"][0]
    night.check_brake_gap_metrics(row["b"], prior["raw_b"][TARGET], TARGET, "scored smoke B")
    night.check_brake_gap_metrics(row["c"], row["c"], TARGET, "scored smoke C")
    return scored


def run(args):
    root = args.out.resolve()
    archive = args.archive.resolve(strict=True)
    if (root != args.out.absolute() or root.parent.name != EXPERIMENT or root.parent.parent.name != "runs"
            or not re.fullmatch(r"attempt-[0-9]{3,}", root.name)
            or archive != root.parent.parent / night.PROTOCOL):
        raise RuntimeError("Use canonical, separate run/attempt and frozen TRAIN archive paths")
    source = Path(__file__).resolve()
    stage = args.stage
    binding, plan, payload = night.brake_gap_archive(archive, verify_traces=stage == "verify")
    prior = prior_reference(root.parent.parent / night.BRAKE_GAP_PAIRED_EXPERIMENT / "attempt-002", binding, plan)
    manifest = {"experiment_id": EXPERIMENT, "attempt": str(root), "archive": str(archive), "binding": binding,
                "reference": {k: v for k, v in prior.items() if k != "raw_b"},
                "code": {"git_head": args.git_head, "git_diff_sha256": args.git_diff_sha256,
                         "source_sha256": {name: night.file_sha256(source.with_name(name)) for name in
                                           (source.name, "navsim_night.py", "navsim_straight.py", "driver.py",
                                            "simulation.py", "lead_brake_gap.sh")}},
                "inference": {"selection": "argmax", "max_length": 2048, "seed_per_arm": SEED,
                              "lead_brake_mps2": straight.PREVIEW_LEAD_BRAKE_MPS2,
                              "c": "selected radar leader and unrounded closing speed strictly positive"}}
    if stage == "verify":
        root.mkdir(mode=0o700, parents=True)
        write(root / "manifest.json", manifest)
        return 0
    if read(root / "manifest.json") != manifest:
        raise RuntimeError("Attempt manifest, code or frozen inputs changed")
    chain = {"attempt": str(root), "manifest_sha256": digest(manifest)}
    if stage in ("score-smoke", "score"):
        return score_stage(root, stage, plan["scenes"], chain, prior, payload,
                           lambda p: night.on_astra("astra-score", args, p, extra_source=source))
    smoke = stage in ("smoke", "ready-smoke")
    target = [r for r in plan["scenes"] if r["token"] == TARGET]
    prior_smoke = None if smoke else require_scored_smoke(root, target, chain, prior)
    folder = root / ("bc-smoke" if smoke else "bc-pilot")
    if folder.exists() or folder.is_symlink():
        raise RuntimeError("This inference stage already exists; use a new attempt")
    if stage.startswith("ready-"):
        print(f"{stage}: artifact gates verified; runtime check will run on the GPU node", flush=True)
        return 0
    runtime, seed = night.current_brake_gap_runtime(args)
    if runtime["identity"] != prior["runtime_identity"]:
        raise RuntimeError("Current checkpoint/upstream/software differs from attempt-002")
    if not smoke and runtime["slurm_job_id"] == read(root / "bc-smoke" / "runtime.json")["slurm_job_id"]:
        raise RuntimeError("Smoke and pilot must be separate GPU allocations")
    from .driver import DrexMax

    predict(folder, target if smoke else plan["scenes"], chain, runtime, seed, lambda: DrexMax(args.weights), prior_smoke)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify", "ready-smoke", "smoke", "score-smoke",
                                          "ready-predict", "predict", "score", "astra-score"))
    for arg in ("out", "archive", "upstream", "weights", "repo", "package", "source"):
        parser.add_argument(f"--{arg}", type=Path)
    parser.add_argument("--git-head")
    parser.add_argument("--git-diff-sha256")
    parser.add_argument("--limit", type=int, default=8)  # Common Astra transport argument.
    args = parser.parse_args()
    if args.stage == "astra-score":
        payload = json.load(sys.stdin)
        with redirect_stdout(sys.stderr):
            result = scoring_report(args, payload)
        print(json.dumps(result, allow_nan=False))
        return 0
    if args.out is None or args.archive is None or not args.git_head or not args.git_diff_sha256:
        parser.error("Use the MSU wrapper with out/archive/code identity")
    try:
        return run(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, OverflowError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
