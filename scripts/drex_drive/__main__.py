"""Run a closed-loop episode and write a report and a replayable decision trace."""

import argparse
from collections.abc import Callable
import json
from math import ceil
from pathlib import Path
from time import perf_counter

from .driver import DrexMax, reference, request_for
from .simulation import Action, DECISION_STEPS, SCENARIOS, STEP_S, World


def run_episode(
    world: World,
    decide: Callable,
    record: Callable[[dict], None],
    max_ticks: int,
) -> dict:
    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")
    action = Action.WAIT
    previous_speed = None
    source = "driver"
    timing_s = []
    fault_count = 0
    emergency_ticks = 0
    fallback_ticks = 0
    min_gap = world.gap_m
    previous_acceleration = 0.0
    peak_jerk = 0.0
    outcome = "time_limit"

    for tick in range(max_ticks):
        if tick % DECISION_STEPS == 0:
            row = request_for(world.observe(action, previous_speed))
            previous_speed = world.ego.speed_mps
            start = perf_counter()
            source = "driver"
            error = None
            try:
                action, probabilities = decide(row)
                if not isinstance(action, Action):
                    raise ValueError(f"Unknown action {action!r}")
            except Exception as exc:
                action, probabilities = Action.STOP, None
                error = f"{type(exc).__name__}: {exc}"
                source = "fallback"
                fault_count += 1
            timing_s.append(perf_counter() - start)
            record({
                "event": "decision", "tick": world.tick, "sim_time_s": world.time_s,
                "request": row, "action": action.value, "source": source,
                "probabilities": probabilities, "error": error, "wall_seconds": timing_s[-1],
            })

        emergency, acceleration = world.step(action)
        emergency_ticks += int(emergency)
        fallback_ticks += int(source == "fallback" and not emergency)
        gap = world.gap_m
        if gap is not None:
            min_gap = gap if min_gap is None else min(min_gap, gap)
        peak_jerk = max(peak_jerk, abs(acceleration - previous_acceleration) / STEP_S)
        previous_acceleration = acceleration
        ended = world.outcome()
        record({
            "event": "step", "tick": world.tick, "sim_time_s": world.time_s,
            "action": action.value, "source": "emergency_brake" if emergency else source,
            "ego_front_m": round(world.ego.front_m, 4), "ego_speed_mps": round(world.ego.speed_mps, 4),
            "acceleration_mps2": round(acceleration, 4), "lead_gap_m": gap,
            "red_crossings": world.red_crossings, "outcome": ended,
        })
        if ended is not None:
            outcome = ended
            break

    ordered = sorted(timing_s)
    return {
        "outcome": outcome,
        "clean_finish": outcome == "finished" and not (world.red_crossings or emergency_ticks or fault_count),
        "sim_time_s": world.time_s,
        "ego_distance_m": round(world.ego.front_m, 3),
        "red_crossings": world.red_crossings,
        "emergency_brake_ticks": emergency_ticks,
        "fallback_ticks": fallback_ticks,
        "driver_control_fraction": 1 - (emergency_ticks + fallback_ticks) / (tick + 1),
        "decision_count": len(timing_s),
        "decision_errors": fault_count,
        "minimum_lead_gap_m": None if min_gap is None else round(min_gap, 3),
        "peak_abs_jerk_mps3": round(peak_jerk, 3),
        "decision_first_wall_s": timing_s[0],
        "decision_p95_wall_s": ordered[ceil(0.95 * len(ordered)) - 1],
        "decision_p99_wall_s": ordered[ceil(0.99 * len(ordered)) - 1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="route")
    parser.add_argument("--driver", choices=("drex", "reference"), default="drex")
    parser.add_argument("--model", type=Path, help="Path to locally prepared Drex Max v1 weights")
    parser.add_argument("--out", type=Path, required=True, help="New directory for the trace and report")
    parser.add_argument("--max-seconds", type=int, default=60, help="Limit in simulated seconds")
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.out.exists():
        parser.error("--out must point to a new directory")

    load_started = perf_counter()
    choose = reference
    model_path = None
    if args.driver == "drex":
        if args.model is None or not args.model.is_dir():
            parser.error("--driver drex requires --model pointing to the local weights directory")
        model_path = args.model.resolve()
        choose = DrexMax(model_path)
    load_seconds = perf_counter() - load_started

    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "trace.jsonl").open("x", encoding="utf-8", buffering=1) as trace:
        def record(event: dict) -> None:
            trace.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

        result = run_episode(World(SCENARIOS[args.scenario]), choose, record, round(args.max_seconds / STEP_S))
    report = {"scenario": args.scenario, "driver": args.driver, "mode": "synchronous",
              "model_path": str(model_path) if model_path is not None else None,
              "model_load_wall_s": load_seconds if model_path is not None else None, **result}
    payload = json.dumps(report, indent=2, allow_nan=False)
    (args.out / "report.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["clean_finish"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
