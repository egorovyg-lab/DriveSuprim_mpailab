"""Deterministic 2D diagram task. This is not NAVSIM or vehicle simulation."""
import json
import math
import random

SEED = 20261111
COUNTS = (2, 8, 16, 32)
FAMILIES = ("clear", "boundary", "obstacles", "mixed")
HALF_WIDTH, HALF_LENGTH = .7, 1.2
TASKS = (("text", "canonical"), ("text_image", "canonical"),
         ("image", "canonical"), ("image", "reversed"),
         ("image", "permuted"), ("image", "repeat"), ("blank", "canonical"))


def scene(index):
    rng = random.Random(SEED + 7919 * index)
    count, family = COUNTS[index % 4], FAMILIES[(index + index // 4) % 4]
    width = 6. if family in ("clear", "obstacles") else rng.choice((3., 4., 5.))
    obstacles = []
    if family in ("obstacles", "mixed"):
        for _ in range(rng.randint(1, 3)):
            x, y = rng.uniform(-width + 1, width - 1), rng.uniform(9, 22)
            w, h = rng.uniform(1, 2.4), rng.uniform(1, 2.5)
            obstacles.append([round(x-w/2, 2), round(y-h/2, 2), round(x+w/2, 2), round(y+h/2, 2)])
    candidates = [dict(id=0, points=[[0., 0.], [0., 0.]], progress=0.)]
    for i in range(1, count):
        x = rng.uniform(-4.5, 4.5) if family in ("clear", "obstacles") else rng.uniform(-7, 7)
        y = float(rng.choice((12, 16, 20, 24, 28)))
        points = [[0., 0.], [0., 4.], [round(x*.35, 2), round(y*.55, 2)], [round(x, 2), y]]
        candidates.append(dict(id=i, points=points, progress=y))
    return dict(index=index, family=family, road=[-width, width], obstacles=obstacles, candidates=candidates)


def intersects(a, b, rect):
    """Closed segment/AABB intersection, including boundary contact."""
    lo, hi = 0., 1.
    for dim in (0, 1):
        delta = b[dim] - a[dim]
        low, high = rect[dim], rect[dim+2]
        if abs(delta) < 1e-12:
            if not low <= a[dim] <= high:
                return False
        else:
            t0, t1 = (low-a[dim])/delta, (high-a[dim])/delta
            lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
            if lo > hi:
                return False
    return True


def legal(world, candidate):
    points = candidate["points"]
    if any(not world["road"][0]+HALF_WIDTH < p[0] < world["road"][1]-HALF_WIDTH for p in points):
        return False
    for x0, y0, x1, y1 in world["obstacles"]:
        expanded = (x0-HALF_WIDTH, y0-HALF_LENGTH, x1+HALF_WIDTH, y1+HALF_LENGTH)
        if any(intersects(a, b, expanded) for a, b in zip(points, points[1:])):
            return False
    return True


def grade(world, chosen):
    candidates = world["candidates"]
    safe = [c for c in candidates if legal(world, c)]
    best = max(c["progress"] for c in safe)
    c = candidates[chosen]
    ok = legal(world, c)
    return dict(safe=ok, optimal=bool(ok and abs(c["progress"]-best) < 1e-8),
                progress=c["progress"], safe_progress=c["progress"] if ok else 0.,
                best_progress=best, baseline_selected=chosen == 0)


def order(world, kind):
    ids = list(range(len(world["candidates"])))
    random.Random(SEED + 31 * world["index"]).shuffle(ids)
    if kind == "reversed":
        ids.reverse()
    elif kind == "permuted":
        random.Random(SEED + 31 * world["index"] + 11).shuffle(ids)
    elif kind in ("canonical", "repeat"):
        pass
    else:
        raise ValueError("Unknown option order")
    return ids


def hull(points):
    points = sorted(set(points))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for target, seq in ((lower, points), (upper, reversed(points))):
        for p in seq:
            while len(target) >= 2 and cross(target[-2], target[-1], p) <= 0:
                target.pop()
            target.append(p)
    return lower[:-1] + upper[:-1]


def board(world, ids, show_world=True):
    from PIL import Image, ImageDraw, ImageFont
    n = len(ids)
    cols = 2 if n <= 8 else (4 if n <= 16 else 8)
    rows = math.ceil(n/cols)
    panel = 256
    image = Image.new("RGB", (cols*panel, rows*panel), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    for j, pid in enumerate(ids):
        ox, oy = (j % cols)*panel, (j // cols)*panel
        def xy(p):
            # Magnify lateral distances so footprint/boundary contacts are legible.
            return (ox+128+p[0]*12., oy+219-p[1]*5.3)
        def rectangle(rect, **kwargs):
            x0, y0, x1, y1 = rect
            draw.rectangle([xy((x0, y1)), xy((x1, y0))], **kwargs)
        draw.rectangle((ox, oy, ox+255, oy+255), outline="#999999")
        draw.text((ox+8, oy+4), f"C{j:02d}", fill="black", font=font)
        if show_world:
            rectangle((world["road"][0], -4, world["road"][1], 32), fill="#dddddd", outline="black", width=2)
        c = world["candidates"][pid]
        for a, b in zip(c["points"], c["points"][1:]):
            swept = hull([(p[0]+dx, p[1]+dy) for p in (a, b)
                          for dx in (-HALF_WIDTH, HALF_WIDTH) for dy in (-HALF_LENGTH, HALF_LENGTH)])
            draw.polygon([xy(p) for p in swept], fill="#87dbe8")
        if show_world:
            for rect in world["obstacles"]:
                rectangle(rect, fill="#e94b4b", outline="#8b0000")
        draw.line([xy(p) for p in c["points"]], fill="#003bb5", width=2)
        cx, cy = xy((0, 0))
        draw.ellipse((cx-2, cy-2, cx+2, cy+2), fill="black")
        draw.text((ox+8, oy+236), f"Forward: {c['progress']:.0f} m", fill="black", font=font)
    return image


def request(world, mode, ids):
    """Explicit inference schema; evaluation labels/physical IDs are not forwarded."""
    criteria = {f"C{j:02d}": f"Candidate panel C{j:02d}; forward progress {world['candidates'][pid]['progress']:.0f} m."
                for j, pid in enumerate(ids)}
    state = dict(convention="Diagram task. x is lateral, y is forward, all distances in metres. Lateral distances are visually magnified; axes have different scales.",
                 footprint=dict(half_width=HALF_WIDTH, half_length=HALF_LENGTH),
                 legend="Grey is drivable road; red rectangles are obstacles. Cyan is the swept vehicle footprint; blue is its centre path. Touching a red obstacle or a road boundary is forbidden.")
    if mode in ("text", "text_image"):
        state.update(road_x_bounds=world["road"], obstacle_rectangles_xyxy=world["obstacles"],
            candidates=[dict(code=f"C{j:02d}", points=world["candidates"][pid]["points"],
                             forward_progress=world["candidates"][pid]["progress"]) for j, pid in enumerate(ids)])
    elif mode not in ("image", "blank"):
        raise ValueError("Unknown modality")
    row = dict(state=state, question=dict(type="choice", criteria=criteria,
        instructions="Choose a candidate whose entire cyan swept footprint stays strictly inside the grey road and does not touch a red obstacle. Among such candidates maximize forward progress. Several equally good candidates are allowed. Stopping at the black start point is permitted if no forward candidate is valid."))
    return row


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
