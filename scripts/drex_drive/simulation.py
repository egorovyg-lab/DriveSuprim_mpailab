"""Longitudinal driving scenarios for the Drex Max prototype."""

from dataclasses import dataclass, field
from enum import Enum
from math import sqrt


STEP_S = 0.1
DECISION_STEPS = 5
DECISION_PERIOD_S = STEP_S * DECISION_STEPS
SPEED_LIMIT_MPS = 6.0
ACCEL_MPS2 = 2.0
BRAKE_MPS2 = 3.0
EMERGENCY_BRAKE_MPS2 = 7.0
LEAD_BRAKE_MPS2 = 3.0
LEAD_LENGTH_M = 4.5
GOAL_TOLERANCE_M = 2.0
STOPPED_MPS = 0.05
POSITION_EPSILON_M = 1e-8


class Action(str, Enum):
    PROCEED = "proceed"
    MAINTAIN = "maintain"
    SLOW = "slow"
    STOP = "stop"
    WAIT = "wait"


ACCELERATION = {
    Action.PROCEED: ACCEL_MPS2,
    Action.MAINTAIN: 0.0,
    Action.SLOW: -1.5,
    Action.STOP: -BRAKE_MPS2,
    Action.WAIT: -BRAKE_MPS2,
}


@dataclass(frozen=True)
class Scenario:
    destination_m: float
    stop_line_m: float | None = None
    green_after_s: float = 0.0
    lead_start_m: float | None = None
    lead_speed_mps: float = 4.0
    lead_brake_window_s: tuple[float, float] | None = None


SCENARIOS = {
    "route": Scenario(100.0, stop_line_m=46.0, green_after_s=17.0, lead_start_m=17.0),
    "red_light": Scenario(70.0, stop_line_m=31.0, green_after_s=11.0),
    "lead_brakes": Scenario(85.0, lead_start_m=15.0, lead_brake_window_s=(2.0, 7.0)),
}


@dataclass(frozen=True)
class Vehicle:
    front_m: float
    speed_mps: float

    def advance(self, acceleration: float, speed_limit: float) -> "Vehicle":
        speed = min(speed_limit, max(0.0, self.speed_mps + acceleration * STEP_S))
        moving_time = (speed - self.speed_mps) / acceleration if acceleration else 0.0
        distance = (self.speed_mps + speed) * 0.5 * moving_time + speed * (STEP_S - moving_time)
        return Vehicle(self.front_m + distance, speed)


@dataclass
class World:
    scenario: Scenario
    ego: Vehicle = field(default_factory=lambda: Vehicle(0.0, 0.0))
    lead: Vehicle | None = field(init=False)
    lead_braking: bool = field(default=False, init=False)
    tick: int = 0
    red_crossings: int = 0

    def __post_init__(self) -> None:
        start = self.scenario.lead_start_m
        self.lead = None if start is None else Vehicle(start, self.scenario.lead_speed_mps)

    @property
    def time_s(self) -> float:
        return round(self.tick * STEP_S, 9)

    @property
    def gap_m(self) -> float | None:
        return None if self.lead is None else self.lead.front_m - LEAD_LENGTH_M - self.ego.front_m

    def observe(self, previous: Action, previous_speed_mps: float | None) -> dict:
        stop_line = self.scenario.stop_line_m
        gap = self.gap_m
        return {
            "ego_speed_mps": round(self.ego.speed_mps, 2),
            "previous_ego_speed_mps": None if previous_speed_mps is None else round(previous_speed_mps, 2),
            "speed_limit_mps": SPEED_LIMIT_MPS,
            "destination_distance_m": round(self.scenario.destination_m - self.ego.front_m, 2),
            "previous_action": previous.value,
            "traffic_light": (
                None if stop_line is None or self.ego.front_m > stop_line + POSITION_EPSILON_M else {
                    "state": "red" if self.time_s < self.scenario.green_after_s else "green",
                    "stop_line_distance_m": round(max(0.0, stop_line - self.ego.front_m), 2),
                }
            ),
            "lead_vehicle": None if gap is None else {
                "gap_m": round(gap, 2),
                "speed_mps": round(self.lead.speed_mps, 2),
            },
        }

    def step(self, action: Action) -> tuple[bool, float]:
        previous = self.ego
        gap = self.gap_m
        emergency = False
        if self.lead is not None and previous.speed_mps > STOPPED_MPS:
            ego_distance = previous.speed_mps**2 / (2 * EMERGENCY_BRAKE_MPS2)
            lead_distance = self.lead.speed_mps**2 / (2 * LEAD_BRAKE_MPS2)
            # Leave clearance for one simulation tick before the next check.
            emergency = gap <= 0.5 + previous.speed_mps * STEP_S + max(0.0, ego_distance - lead_distance)

        acceleration = -EMERGENCY_BRAKE_MPS2 if emergency else ACCELERATION[action]
        self.ego = previous.advance(acceleration, SPEED_LIMIT_MPS)
        self._advance_lead()
        self.tick += 1

        line = self.scenario.stop_line_m
        if line is not None and previous.front_m <= line + POSITION_EPSILON_M < self.ego.front_m:
            if self.time_s <= self.scenario.green_after_s:
                self.red_crossings += 1
        return emergency, (self.ego.speed_mps - previous.speed_mps) / STEP_S

    def _advance_lead(self) -> None:
        if self.lead is None:
            return
        target = self.scenario.lead_speed_mps
        brake_window = self.scenario.lead_brake_window_s
        if brake_window is not None and brake_window[0] <= self.time_s < brake_window[1]:
            target = 0.0
        line = self.scenario.stop_line_m
        if line is not None and self.time_s < self.scenario.green_after_s:
            remaining = max(0.0, line - 2.0 - self.lead.front_m)
            target = min(target, sqrt(2 * LEAD_BRAKE_MPS2 * remaining))
        acceleration = max(-LEAD_BRAKE_MPS2, min(ACCEL_MPS2, (target - self.lead.speed_mps) / STEP_S))
        self.lead_braking = acceleration < -1e-6 or target == 0.0
        self.lead = self.lead.advance(acceleration, self.scenario.lead_speed_mps)

    def outcome(self) -> str | None:
        gap = self.gap_m
        remaining = self.scenario.destination_m - self.ego.front_m
        if gap is not None and gap <= 0.0:
            return "collision"
        if remaining < -GOAL_TOLERANCE_M:
            return "missed_destination"
        if remaining <= GOAL_TOLERANCE_M and self.ego.speed_mps <= STOPPED_MPS:
            return "finished"
        return None
