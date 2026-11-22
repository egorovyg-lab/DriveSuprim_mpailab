"""Run one inference at a time while the simulation advances on its own clock."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter, sleep

from .driver import request_for
from .simulation import Action, DECISION_PERIOD_S, DECISION_STEPS, STEP_S, World


@dataclass(frozen=True)
class Reply:
    action: Action
    probabilities: dict[str, float] | None
    error: str | None


def ask(decide, row: dict) -> Reply:
    try:
        action, probabilities = decide(row)
        if not isinstance(action, Action):
            raise ValueError(f"Unknown action {action!r}")
        return Reply(action, probabilities, None)
    except Exception as exc:
        return Reply(Action.STOP, None, f"{type(exc).__name__}: {exc}")


def timed_ask(decide, row: dict, extra_delay_ms: int) -> tuple[Reply, float]:
    reply = ask(decide, row)
    if extra_delay_ms:
        sleep(extra_delay_ms / 1000)
    return reply, perf_counter()


@dataclass
class Pending:
    tick: int
    submitted_s: float
    future: Future
    expired: bool = False

    @property
    def deadline_s(self) -> float:
        return self.submitted_s + DECISION_PERIOD_S


class LiveDriver:
    def __init__(self, decide, worker: ThreadPoolExecutor, record, injected_timeouts: dict[int, int]) -> None:
        self.decide = decide
        self.worker = worker
        self.record = record
        self.pending: Pending | None = None
        self.action = Action.WAIT
        self.source = "startup_wait"
        self.valid_until_tick = 0
        self.previous_speed = None
        self.errors = 0
        self.submitted = 0
        self.applied_ages_s: list[float] = []
        self.response_seconds: list[float] = []
        self.deadline_misses = 0
        self.busy_slots = 0
        self.late_replies = 0
        self.injected_timeouts = injected_timeouts
        self.injection_status = dict.fromkeys(injected_timeouts, "not_submitted")

    def step(self, world: World) -> tuple[Action, str]:
        pending = self.pending
        if pending is not None:
            ready = pending.future.done()
            now = perf_counter()
            if now >= pending.deadline_s and not pending.expired:
                pending.expired = True
                self.deadline_misses += 1
                self.action, self.source = Action.STOP, "fallback"
                if pending.tick in self.injection_status:
                    self.injection_status[pending.tick] = "expired"
                self.record({"event": "decision_expired", "request_tick": pending.tick, "tick": world.tick})
            if ready:
                self._take_reply(pending, world.tick, now)
                self.pending = None

        if self.source == "driver" and world.tick >= self.valid_until_tick:
            self.action, self.source = Action.STOP, "fallback"
            self.record({"event": "command_expired", "tick": world.tick})

        if world.tick % DECISION_STEPS == 0:
            if self.pending is None:
                row = request_for(world.observe(self.action, self.previous_speed))
                self.previous_speed = world.ego.speed_mps
                issued = perf_counter()
                extra_delay = self.injected_timeouts.get(world.tick, 0)
                future = self.worker.submit(timed_ask, self.decide, row, extra_delay)
                self.pending = Pending(world.tick, issued, future)
                if extra_delay:
                    self.injection_status[world.tick] = "submitted"
                self.submitted += 1
                self.record({"event": "decision_requested", "tick": world.tick, "sim_time_s": world.time_s,
                             "deadline_seconds": DECISION_PERIOD_S, "extra_delay_ms": extra_delay, "request": row})
            else:
                self.busy_slots += 1
                self.record({"event": "decision_slot_busy", "tick": world.tick, "request_tick": self.pending.tick})
        return self.action, self.source

    def _take_reply(self, pending: Pending, tick: int, polled_s: float) -> None:
        reply, completed_s = pending.future.result()
        age_s = max(polled_s, completed_s) - pending.submitted_s
        duration_s = completed_s - pending.submitted_s
        self.response_seconds.append(duration_s)
        self.errors += int(reply.error is not None)
        if pending.expired:
            self.late_replies += 1
            status = "late"
            if pending.tick in self.injection_status:
                self.injection_status[pending.tick] = "discarded"
        elif reply.error is not None:
            self.action, self.source = Action.STOP, "fallback"
            status = "error"
        else:
            self.action, self.source = reply.action, "driver"
            self.valid_until_tick = tick + DECISION_STEPS
            self.applied_ages_s.append(age_s)
            status = "applied"
        self.record({"event": "decision", "status": status, "tick": tick, "request_tick": pending.tick,
                     "action": reply.action.value, "probabilities": reply.probabilities, "error": reply.error,
                     "wall_seconds": duration_s, "age_seconds": age_s})

    def finish(self, tick: int) -> dict:
        unfinished = int(self.pending is not None and not self.pending.expired)
        if self.pending is not None:
            self.record({"event": "decision_end", "tick": tick, "request_tick": self.pending.tick,
                         "expired": self.pending.expired})
        return {"decision_count": self.submitted, "decision_replies": len(self.response_seconds),
                "decision_applied": len(self.applied_ages_s), "decision_deadline_misses": self.deadline_misses,
                "decision_slots_busy": self.busy_slots, "decision_late_replies": self.late_replies,
                "decision_unfinished_at_end": unfinished,
                "injected_timeouts": [
                    {"request_tick": tick, "sim_time_s": round(tick * STEP_S, 9),
                     "extra_delay_ms": self.injected_timeouts[tick], "status": status}
                    for tick, status in sorted(self.injection_status.items())
                ],
                "decision_deadline_miss_fraction": self.deadline_misses / (self.submitted - unfinished)
                if self.submitted > unfinished else None}
