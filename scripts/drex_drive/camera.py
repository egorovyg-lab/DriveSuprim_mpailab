"""Deterministic pinhole rendering of the longitudinal simulator's road."""

from .simulation import DECISION_STEPS, World


FRAME_SIZE = (512, 320)
HORIZON_Y = 126
FOCAL_PIXELS = 340
CAMERA_HEIGHT_M = 1.5
CAMERA_BEHIND_FRONT_M = 1.5
LANE_HALF_WIDTH_M = 1.8


class CameraHistory:
    def __init__(self) -> None:
        self.frame = None
        self.tick: int | None = None

    def capture(self, world: World) -> list:
        frame = render(world)
        images = [self.frame, frame] if self.tick == world.tick - DECISION_STEPS else [frame]
        self.frame, self.tick = frame, world.tick
        return images


def _project(x_m: float, height_m: float, depth_m: float) -> tuple[int, int]:
    scale = FOCAL_PIXELS / depth_m
    return round(FRAME_SIZE[0] / 2 + x_m * scale), round(HORIZON_Y + (CAMERA_HEIGHT_M - height_m) * scale)


def _road_strip(draw, near: float, far: float, left: float, right: float, color: str) -> None:
    draw.polygon([_project(left, 0, far), _project(right, 0, far),
                  _project(right, 0, near), _project(left, 0, near)], fill=color)


def _signal(draw, depth: float, red: bool) -> None:
    x = 2.35
    housing = [_project(x - 0.25, 3.25, depth), _project(x + 0.25, 1.88, depth)]
    draw.line([_project(x, 0, depth), _project(x, 2.1, depth)], fill="#686867",
              width=max(1, round(FOCAL_PIXELS * 0.12 / depth)))
    draw.rectangle(housing, fill="#161a1c", outline="#111415")
    for height, color in [(2.98, "#ff2521" if red else "#553733"), (2.56, "#59502a"),
                          (2.14, "#2f4b40" if red else "#23f65c")]:
        draw.ellipse([_project(x - 0.16, height + 0.16, depth), _project(x + 0.16, height - 0.16, depth)],
                     fill=color)


def _lead_car(draw, depth: float, braking: bool) -> None:
    draw.polygon([_project(x, y, depth) for x, y in [(-0.93, 0.25), (-0.93, 0.95), (-0.65, 1.47),
                                                    (0.65, 1.47), (0.93, 0.95), (0.93, 0.25)]],
                 fill="#a4adc0", outline="#292e35")
    draw.polygon([_project(x, y, depth) for x, y in [(-0.54, 1.38), (0.54, 1.38),
                                                    (0.73, 0.96), (-0.73, 0.96)]], fill="#344557")
    draw.rectangle([_project(-0.9, 0.39, depth), _project(0.9, 0.24, depth)], fill="#4b5060")
    for x in (-0.71, 0.71):
        draw.rectangle([_project(x - 0.13, 0.74, depth), _project(x + 0.13, 0.59, depth)],
                       fill="#ff251c" if braking else "#7a3438")
        draw.rectangle([_project(x - 0.13, 0.28, depth), _project(x + 0.13, 0.04, depth)], fill="#20242a")


def render(world: World):
    from PIL import Image, ImageDraw

    frame = Image.new("RGB", FRAME_SIZE, "#b6cfdf")
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, HORIZON_Y, FRAME_SIZE[0], FRAME_SIZE[1]), fill="#7f9576")
    _road_strip(draw, 0.5, 180, -2.25, 2.25, "#55595e")
    for edge in (-LANE_HALF_WIDTH_M, LANE_HALF_WIDTH_M):
        _road_strip(draw, 0.5, 180, edge - 0.055, edge + 0.055, "#e8e3d8")

    stop_line = world.scenario.stop_line_m
    if stop_line is not None:
        depth = stop_line - world.ego.front_m + CAMERA_BEHIND_FRONT_M
        if 0.65 < depth <= 100:
            _road_strip(draw, depth, depth + 0.35, -LANE_HALF_WIDTH_M, LANE_HALF_WIDTH_M, "#f7f4ea")
        # The signal is beyond the line so it remains in frame at the stop position.
        if 0.2 < depth <= 100:
            _signal(draw, depth + 4.0, world.time_s < world.scenario.green_after_s)

    gap = world.gap_m
    if gap is not None and 0 <= gap <= 120:
        _lead_car(draw, gap + CAMERA_BEHIND_FRONT_M, world.lead_braking)
    return frame
