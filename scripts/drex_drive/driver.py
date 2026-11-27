"""Decision request and the official Drex Max adapter."""

from __future__ import annotations

from math import isclose, isfinite
from pathlib import Path

from .camera import render
from .simulation import ACCEL_MPS2, Action, BRAKE_MPS2, DECISION_PERIOD_S, GOAL_TOLERANCE_M, World


CRITERIA = {
    Action.PROCEED.value: "Accelerate towards the speed limit. Use this to start or resume moving.",
    Action.MAINTAIN.value: "Keep the current speed.",
    Action.SLOW.value: "Reduce speed gently at 1.5 m/s².",
    Action.STOP.value: "Brake at 3 m/s² down to a full stop.",
    Action.WAIT.value: "Remain stopped until you can proceed.",
}
STRUCTURED_OBSERVATIONS = ("state", "state-camera")
RADAR_OBSERVATIONS = ("camera-range", "camera-radar")
OBSERVATIONS = (*STRUCTURED_OBSERVATIONS, "camera", "camera-pair", *RADAR_OBSERVATIONS)


def request_for(state: dict) -> dict:
    return {
        "state": state,
        "question": {
            "type": "choice",
            "instructions": (
                f"Drive to the destination and stop within {GOAL_TOLERANCE_M:g} meters of it. "
                f"Your action applies for the next {DECISION_PERIOD_S:g} seconds. "
                "Keep a clear gap to the lead vehicle. Stop before a red light's line "
                "and resume when it is green and the way is clear. Choose the current action only."
            ),
            "criteria": CRITERIA.copy(),
        },
    }


def driving_request(world: World, previous: Action, previous_speed: float | None, observation: str,
                    images: list | None = None) -> dict:
    state = world.observe(previous, previous_speed)
    if observation not in STRUCTURED_OBSERVATIONS:
        del state["traffic_light"], state["lead_vehicle"]
    if observation in RADAR_OBSERVATIONS:
        gap = world.gap_m
        target = None
        if world.lead is not None and gap is not None:
            target = {"range_m": round(gap, 2)}
            if observation == "camera-radar":
                target["closing_speed_mps"] = round(world.ego.speed_mps - world.lead.speed_mps, 2)
        state["lead_radar"] = target
    row = request_for(state)
    if observation != "state":
        row["images"] = images if images is not None else [render(world)]
        if len(row["images"]) == 2:
            row["question"]["instructions"] += (
                f" Image 1 is the forward view {DECISION_PERIOD_S:g} seconds earlier. Image 2 is the current "
                "forward view. Compare the lead vehicle's apparent size and brake lights to judge whether "
                "the gap is closing. Read the current traffic signal and its white stop line from image 2."
            )
        else:
            row["question"]["instructions"] += (
                " The image is the current forward view. Look for the traffic signal, its white stop "
                "line and the lead vehicle in that view."
            )
    if observation in RADAR_OBSERVATIONS:
        row["question"]["instructions"] += (
            " When available, lead_radar reports the bumper-to-bumper range to the lead vehicle in your lane."
        )
    if observation == "camera-radar":
        row["question"]["instructions"] += " A positive closing_speed_mps means that range is shrinking."
    return row


class DrexMax:
    def __init__(self, weights: Path) -> None:
        import torch
        from drex_max.model import DecisionModel

        self.model = DecisionModel(weights, device="cuda:0")
        if self.model.readout.weight.dtype != torch.bfloat16:
            raise RuntimeError("Expected the BF16 Drex Max readout")
        self.verified_frame_counts: set[int] = set()

    def __call__(self, request: dict) -> tuple[Action, dict[str, float]]:
        choice, probabilities = self.choose(request)
        return Action(choice), probabilities

    def choose(self, request: dict) -> tuple[str, dict[str, float]]:
        frame_count = len(request.get("images", ()))
        if frame_count and frame_count not in self.verified_frame_counts:
            inputs = self.model.prepare([request], max_length=2048).inputs
            boxes = inputs.get("image_grid_thw")
            if "pixel_values" not in inputs or boxes is None or len(boxes) != frame_count:
                raise RuntimeError(f"Drex Max preprocessing did not produce camera pixel inputs for {frame_count} frames")
            self.verified_frame_counts.add(frame_count)
            del inputs, boxes
        raw = self.model.predict([request], batch_size=1, max_length=2048)[0]
        values = [float(value) for value in raw]
        keys = request["question"]["criteria"]
        if (len(values) != len(keys) or any(not isfinite(value) or not 0 <= value <= 1 for value in values)
                or not isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-5)):
            raise ValueError("Drex Max returned an invalid choice distribution")
        probabilities = dict(zip(keys, values))
        return max(probabilities, key=probabilities.__getitem__), probabilities


def reference(request: dict) -> tuple[Action, None]:
    """Deterministic comparison driver for local tests and scenario validation."""
    state = request["state"]
    speed = state["ego_speed_mps"]
    stopping_distance = speed**2 / (2 * BRAKE_MPS2)
    destination = state["destination_distance_m"]
    action = Action.PROCEED if speed < state["speed_limit_mps"] else Action.MAINTAIN
    light = state["traffic_light"]
    if light is not None and light["state"] == "red":
        next_speed = min(state["speed_limit_mps"], speed + ACCEL_MPS2 * DECISION_PERIOD_S)
        distance_if_proceed = next_speed * DECISION_PERIOD_S + next_speed**2 / (2 * BRAKE_MPS2)
        if light["stop_line_distance_m"] <= distance_if_proceed + 0.75:
            return (Action.STOP if speed else Action.WAIT), None
    if destination <= stopping_distance + speed * DECISION_PERIOD_S + 0.75:
        return (Action.STOP if speed else Action.WAIT), None
    lead = state["lead_vehicle"]
    if lead is not None:
        lead_speed = lead["speed_mps"]
        gap = lead["gap_m"]
        relative_stop = max(0.0, stopping_distance - lead_speed**2 / (2 * BRAKE_MPS2))
        if gap <= 2.5 + speed * DECISION_PERIOD_S + relative_stop:
            return (Action.STOP if speed else Action.WAIT), None
        if gap < 3.0 + 1.25 * speed:
            if speed > lead_speed + 0.2:
                return Action.SLOW, None
            if speed >= lead_speed - 0.2:
                return Action.MAINTAIN, None
    return action, None
