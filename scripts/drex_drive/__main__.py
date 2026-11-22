"""Run a closed-loop episode and write a report and a replayable decision trace."""

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import json
from math import ceil
from pathlib import Path
from time import perf_counter, sleep

from .driver import DrexMax, reference, request_for
from .realtime import LiveDriver, ask
from .simulation import Action, DECISION_PERIOD_S, DECISION_STEPS, SCENARIOS, STEP_S, World


def run_episode(
    world: World,
    decide: Callable,
    record: Callable[[dict], None],
    max_ticks: int,
    *,
    realtime: bool = False,
    warmup: bool = False,
    injected_timeouts: dict[int, int] | None = None,
) -> dict:
    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")
    if realtime:
        with ThreadPoolExecutor(max_workers=1) as worker:
            warmup_seconds = worker.submit(_warmup, world, decide).result() if warmup else None
            live = LiveDriver(decide, worker, record, injected_timeouts or {})
            report = _drive(world, decide, record, max_ticks, live)
    else:
        if injected_timeouts:
            raise ValueError("Injected timeouts require realtime mode")
        warmup_seconds = _warmup(world, decide) if warmup else None
        report = _drive(world, decide, record, max_ticks, None)
    return {"warmup_wall_s": warmup_seconds, **report}


def _warmup(world: World, decide) -> float:
    start = perf_counter()
    reply = ask(decide, request_for(world.observe(Action.WAIT, None)))
    if reply.error is not None:
        raise RuntimeError(f"Warmup failed: {reply.error}")
    return perf_counter() - start


def percentile(values: list[float], q: float) -> float | None:
    return sorted(values)[ceil(q * len(values)) - 1] if values else None


def parse_timeout(value: str) -> tuple[int, int]:
    try:
        at_ms, extra_ms = map(int, value.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected SIM_MS:EXTRA_WALL_MS") from exc
    period_ms = round(DECISION_PERIOD_S * 1000)
    if at_ms < 0 or at_ms % period_ms or extra_ms <= period_ms:
        raise argparse.ArgumentTypeError(f"SIM_MS must be a nonnegative multiple of {period_ms}. "
                                         f"EXTRA_WALL_MS must exceed {period_ms}")
    return at_ms // round(STEP_S * 1000), extra_ms


def _drive(world: World, decide, record, max_ticks: int, live: LiveDriver | None) -> dict:
    action = Action.WAIT
    previous_speed = None
    source = "driver"
    timing_s = []
    fault_count = 0
    emergency_ticks = 0
    fallback_ticks = 0
    startup_wait_ticks = 0
    fallback_activations = 0
    fallback_started_s = None
    fallback_recoveries_s = []
    tick_lag_s = []
    min_gap = world.gap_m
    previous_acceleration = 0.0
    peak_jerk = 0.0
    outcome = "time_limit"
    started = perf_counter()

    for tick in range(max_ticks):
        last_source = source
        if live is not None:
            scheduled_s = started + tick * STEP_S
            wait_s = scheduled_s - perf_counter()
            if wait_s > 0:
                sleep(wait_s)
            tick_lag_s.append(max(0.0, perf_counter() - scheduled_s))
            action, source = live.step(world)
        elif tick % DECISION_STEPS == 0:
            row = request_for(world.observe(action, previous_speed))
            previous_speed = world.ego.speed_mps
            start = perf_counter()
            reply = ask(decide, row)
            action = reply.action
            source = "fallback" if reply.error is not None else "driver"
            fault_count += int(reply.error is not None)
            timing_s.append(perf_counter() - start)
            record({
                "event": "decision", "tick": world.tick, "sim_time_s": world.time_s,
                "request": row, "action": action.value, "source": source,
                "probabilities": reply.probabilities, "error": reply.error, "wall_seconds": timing_s[-1],
            })

        emergency, acceleration = world.step(action)
        emergency_ticks += int(emergency)
        fallback_ticks += int(source == "fallback" and not emergency)
        startup_wait_ticks += int(source == "startup_wait" and not emergency)
        if source == "fallback" and last_source != "fallback":
            fallback_activations += 1
            fallback_started_s = perf_counter()
            record({"event": "fallback_started", "tick": tick})
        if source == "driver" and not emergency and fallback_started_s is not None:
            duration_s = perf_counter() - fallback_started_s
            fallback_recoveries_s.append(duration_s)
            fallback_started_s = None
            record({"event": "fallback_recovered", "tick": tick, "wall_seconds": duration_s})
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
            **({"tick_lag_wall_s": tick_lag_s[-1]} if live is not None else {}),
        })
        if ended is not None:
            outcome = ended
            break

    episode_seconds = perf_counter() - started
    live_stats = {}
    if live is not None:
        live_stats = live.finish(world.tick)
        timing_s = live.response_seconds
        fault_count = live.errors
        live_stats.update(
            decision_deadline_wall_s=DECISION_STEPS * STEP_S,
            decision_applied_p95_wall_s=percentile(live.applied_ages_s, 0.95),
            tick_lag_max_wall_s=max(tick_lag_s),
            tick_lag_p95_wall_s=percentile(tick_lag_s, 0.95),
        )
    safe_finish = outcome == "finished" and not (world.red_crossings or emergency_ticks or fault_count)
    injections = live_stats.get("injected_timeouts", [])
    if injections:
        live_stats["latency_stress_pass"] = (
            safe_finish
            and all(item["status"] == "discarded" for item in injections)
            and live_stats["decision_deadline_misses"] == fallback_activations
            == len(fallback_recoveries_s) == len(injections)
        )
    return {
        "outcome": outcome,
        "clean_finish": safe_finish and not fallback_activations,
        "sim_time_s": world.time_s,
        "episode_wall_s": episode_seconds,
        "ego_distance_m": round(world.ego.front_m, 3),
        "red_crossings": world.red_crossings,
        "emergency_brake_ticks": emergency_ticks,
        "fallback_ticks": fallback_ticks,
        "fallback_activations": fallback_activations,
        "fallback_recoveries_wall_s": fallback_recoveries_s,
        "startup_wait_ticks": startup_wait_ticks,
        "driver_control_fraction": 1 - (emergency_ticks + fallback_ticks + startup_wait_ticks) / (tick + 1),
        "decision_count": len(timing_s),
        "decision_errors": fault_count,
        "minimum_lead_gap_m": None if min_gap is None else round(min_gap, 3),
        "peak_abs_jerk_mps3": round(peak_jerk, 3),
        "decision_first_wall_s": timing_s[0] if timing_s else None,
        "decision_p95_wall_s": percentile(timing_s, 0.95),
        "decision_p99_wall_s": percentile(timing_s, 0.99),
        **live_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="route")
    parser.add_argument("--driver", choices=("drex", "reference"), default="drex")
    parser.add_argument("--mode", choices=("synchronous", "realtime"), default="synchronous")
    parser.add_argument("--warmup", action="store_true", help="Run one discarded model call before the episode")
    parser.add_argument("--model", type=Path, help="Path to locally prepared Drex Max v1 weights")
    parser.add_argument("--out", type=Path, required=True, help="New directory for the trace and report")
    parser.add_argument("--max-seconds", type=int, default=60, help="Limit in simulated seconds")
    parser.add_argument("--inject-timeout", type=parse_timeout, action="append", default=[],
                        metavar="SIM_MS:EXTRA_WALL_MS", help="Delay a selected realtime response after inference")
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.out.exists():
        parser.error("--out must point to a new directory")
    max_ticks = round(args.max_seconds / STEP_S)
    injected_timeouts = dict(args.inject_timeout)
    if injected_timeouts and args.mode != "realtime":
        parser.error("--inject-timeout requires --mode realtime")
    if len(injected_timeouts) != len(args.inject_timeout):
        parser.error("--inject-timeout times must be unique")
    if any(tick >= max_ticks for tick in injected_timeouts):
        parser.error("--inject-timeout must be scheduled before --max-seconds")

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

        result = run_episode(World(SCENARIOS[args.scenario]), choose, record, max_ticks,
                             realtime=args.mode == "realtime", warmup=args.warmup, injected_timeouts=injected_timeouts)
    report = {"scenario": args.scenario, "driver": args.driver, "mode": args.mode,
              "model_path": str(model_path) if model_path is not None else None,
              "model_load_wall_s": load_seconds if model_path is not None else None, **result}
    payload = json.dumps(report, indent=2, allow_nan=False)
    (args.out / "report.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.get("latency_stress_pass", report["clean_finish"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
