"""Read-only current-camera/FOV audit of a completed privileged geometry run.

Separate outputs only. No GPU, training, scoring, downloads or holdout images.
Cases are outcome-conditioned diagnostics, not an evaluation cohort.
"""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import platform

import numpy as np
import shapely
from PIL import Image, ImageDraw, __version__ as pillow_version

from navsim.agents import corridor_geometry as geometry
from navsim.agents import corridor_projection as projection
from navsim.planning.script import run_corridor_geometry as parent

REPO, read, write, sha = parent.REPO, parent.read, parent.write, parent.sha
MANIFESTS = frozenset(("identity.json", "source_config.json", "cache_index.json", "splits.json",
    "split_identity.json", "selection.json", "acquisition_plan.json", "expansion_protocol.json"))
REFERENCES = {
    "metadata": "https://github.com/OpenDriveLab/OpenScene/blob/main/DriveEngine/process_data/create_openscene_metadata.py",
    "camera": "https://github.com/OpenDriveLab/OpenScene/blob/main/DriveEngine/process_data/helpers/nuplan_cameras_utils.py",
}
NOTE = ("Diagnostic FOV upper bound, NOT verified visibility or camera-only performance. "
    "Flat current ego ground plane; source-defined OpenScene camera-to-ego extrinsics despite sensor2lidar names. "
    "No occlusion, road height or camera/LiDAR time-offset correction. Current-body points masked. "
    "Distortion applied to raw images; exact DriveSuprim three-camera crop coordinates, PIL preview resize only. "
    "Map geometry and outcome-conditioned cases are diagnostic labels only. Human projection review still required. "
    "No new scorer, training, image transfer or fresh holdout access.")


def edge_paths(path, value):
    base = path.parent
    if isinstance(value.get("source"), str):
        root = Path(value["source"])
        base = root if root.is_absolute() else base / root
    for name, expected in value.get("inputs", {}).items():
        location = Path(name)
        if isinstance(expected, dict):
            root = location if location.is_absolute() else path.parent / location
            for filename, digest in expected.items():
                child = Path(filename)
                yield (child if child.is_absolute() else root / child).resolve(), digest
        else:
            yield (location if location.is_absolute() else base / location).resolve(), expected
    for name, digest in value.get("artifacts", {}).items():
        yield (path.parent / name).resolve(), digest
    if path.name == "split_identity.json" and "splits" in value:
        yield path.parent / "splits.json", value["splits"]


def metadata_graph(bundle, expected):
    pending, seen, pinned = [(bundle / "identity.json", expected)], {}, {}
    while pending:
        path, digest = pending.pop()
        path = path.resolve()
        if path in seen:
            if seen[path][0] != digest:
                raise RuntimeError("Conflicting metadata hashes: " + str(path))
            continue
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError("Pinned projection metadata changed/missing: " + str(path))
        value = read(path)
        if not isinstance(value, dict):
            raise ValueError("Expected object metadata: " + str(path))
        seen[path] = (digest, value)
        if path.name == "cache_index.json":
            continue
        for child, child_hash in edge_paths(path, value):
            if not isinstance(child_hash, str):
                raise ValueError("Expected explicit metadata hash: " + str(child))
            if child in pinned and pinned[child] != child_hash:
                raise RuntimeError("Conflicting pinned inputs: " + str(child))
            pinned[child] = child_hash
            if child.name in MANIFESTS:
                pending.append((child, child_hash))
    return seen, pinned


def completed_run(root):
    identity, report = read(root / "identity.json"), read(root / "report.json")
    if report["status"] != "complete" or report.get("rollout_DAC_consistency_passed") is not True:
        raise ValueError("Complete DAC-consistent geometry run required")
    if report["identity_sha256"] != sha(root / "identity.json"):
        raise RuntimeError("Parent geometry identity hash mismatch")
    for name, digest in report["artifacts"].items():
        if sha(root / name) != digest:
            raise RuntimeError("Parent geometry artifact changed: " + name)
    for name, digest in identity["sources"].items():
        if sha(REPO / name) != digest:
            raise RuntimeError("Parent geometry code changed: " + name)
    for name, digest in identity["inputs"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError("Parent geometry bundle changed: " + name)
    parent.checked_packets(root, identity)
    marker = [Path(p) for p in identity["inputs"] if Path(p).name == "bundle_manifest.json"]
    if len(marker) != 1:
        raise ValueError("One frozen parent bundle required")
    bundle = marker[0].parent
    manifest = read(marker[0])
    if (sha(bundle / "identity.json") != manifest["identity_sha256"]
            or sha(bundle / "examples_index.json") != manifest["index_sha256"]):
        raise RuntimeError("Bundle membership changed")
    metadata = {t: r["metadata"] for t, r in read(bundle / "examples_index.json").items()}
    protected = set(read(bundle / "identity.json")["protected_sessions"])
    if (set(metadata) != set(identity["tokens"]) or any(m["role"] != "train" for m in metadata.values())
            or {m["session_id"] for m in metadata.values()} & protected):
        raise ValueError("Training/protected membership mismatch")
    decisions = [d for d in read(root / "decisions.json")["decisions"] if d["method"] == "rollout"]
    if len(decisions) != len(metadata) or {d["token"] for d in decisions} != set(metadata):
        raise ValueError("Incomplete/duplicate frozen rollout decisions")
    return identity, bundle, metadata, decisions


def select_cases(metadata, decisions):
    bad = sorted((d for d in decisions if not d["baseline_passed"]), key=lambda d: d["token"])
    good = {d["token"]: d for d in decisions if d["baseline_passed"]}
    cases, used = [], {d["token"] for d in bad}
    for d in bad:
        kind = "repair" if d["reason"] == "baseline_rejected_choose_first_valid" else "unrepairable"
        cases.append(dict(token=d["token"], kind=kind, decision=d, metadata=metadata[d["token"]]))
        sid = metadata[d["token"]]["session_id"]
        controls = sorted(t for t in good if t not in used and metadata[t]["session_id"] == sid)
        if controls:
            token = controls[0]
            cases.append(dict(token=token, kind="matched_control", matched_to=d["token"],
                decision=good[token], metadata=metadata[token]))
            used.add(token)
    return sorted(cases, key=lambda c: (c["kind"], c["token"]))


def resolve_records(bundle, metadata):
    seen, pinned = metadata_graph(bundle, sha(bundle / "identity.json"))
    records, configurations, log_hashes = {}, [], {}
    for path, (_, value) in seen.items():
        if "navsim_log_path" in value and "original_sensor_path" in value:
            config = value["agent"]["config"]
            if int(config["n_camera"]) != 3:
                raise ValueError("This audit implements the frozen three-camera policy only")
            configurations.append((value["navsim_log_path"], value["original_sensor_path"],
                int(config["camera_width"]), int(config["camera_height"])))
        # Decoder/export identities also contain abbreviated "records" without
        # timestamps/logs. Only dataset split/selection files define frames.
        frame_records = (value.get("records", []) + value.get("legacy_training_records", [])
            if path.name in ("splits.json", "selection.json") else [])
        for record in frame_records:
            token = record["token"]
            # Older metadata-only inventories may assign another role before
            # the accepted union was collected. They cannot define this audit's
            # membership: require an explicit training record and the verified
            # bundle's session, and never follow an old development/holdout row.
            if token not in metadata or record.get("role") not in ("memory", "fit", "train"):
                continue
            if record["session_id"] != metadata[token]["session_id"]:
                raise ValueError("Training token bound to another session: " + token)
            selected = {k: record[k] for k in ("token", "session_id", "log_name", "timestamp_us")}
            if token in records and records[token] != selected:
                raise ValueError("Conflicting selected frame metadata: " + token)
            records[token] = selected
        for log, digest in value.get("log_sha256", {}).items():
            if log in log_hashes and log_hashes[log] != digest:
                raise ValueError("Conflicting log hashes: " + log)
            log_hashes[log] = digest
    if len(set(configurations)) != 1:
        raise ValueError("One consistent pinned image/log/crop configuration required")
    return records, configurations[0], seen, pinned, log_hashes


def safe_image(root, relative):
    value = str(relative)
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value
            or path.suffix.lower() not in (".jpg", ".jpeg", ".png")):
        raise ValueError("Unsafe image path: " + value)
    target = (root / value).resolve()
    if root.resolve() not in target.parents:
        raise ValueError("Image escaped sensor root")
    return target


def initialize(root, out):
    identity, bundle, metadata, decisions = completed_run(root)
    records, config, manifests, pinned, log_hashes = resolve_records(bundle, metadata)
    logs, sensors = Path(config[0]).resolve(), Path(config[1]).resolve()
    for source in (root, bundle, logs, sensors):
        if source == out or source in out.parents or out in source.parents:
            raise ValueError("Separate projection output outside parent/bundle/data roots required")
    cases = select_cases(metadata, decisions)
    inputs = {str(root / n): sha(root / n) for n in ("identity.json", "report.json", "packet_index.json", "decisions.json")}
    inputs.update({str(p): digest for p, (digest, _) in manifests.items()})
    grouped = {}
    for case in cases:
        if case["token"] not in records:
            raise ValueError("Missing pinned source frame record: " + case["token"])
        record = records[case["token"]]
        case["record"] = record
        grouped.setdefault(record["log_name"], []).append(case)
    for log, group in sorted(grouped.items()):
        path = logs / (log + ".pkl")
        expected = pinned.get(path, log_hashes.get(log))
        if not expected or not path.is_file() or sha(path) != expected:
            raise RuntimeError("Missing/changed pinned source log: " + str(path))
        inputs[str(path)] = expected
        with path.open("rb") as stream:
            frames = pickle.load(stream)  # Existing trusted metadata; only current frame fields below.
        wanted = {c["token"]: c for c in group}
        found = set()
        for frame in frames:
            token = frame["token"]
            if token not in wanted:
                continue
            if token in found:
                raise ValueError("Duplicate current frame token: " + token)
            found.add(token)
            case = wanted[token]
            if int(frame["timestamp"]) != int(case["record"]["timestamp_us"]):
                raise ValueError("Selected current-frame timestamp changed: " + token)
            case["issues"] = []
            try:
                observation = projection.current_observation(frame)
                case["observation"] = observation
                case["images"] = {}
                for name in projection.CAMERAS:
                    target = safe_image(sensors, observation["cameras"][name]["data_path"])
                    if not target.is_file():
                        raise FileNotFoundError("Missing current image: " + str(target))
                    with Image.open(target) as image:
                        size = list(image.size)
                        if image.mode not in ("RGB", "RGBA"):
                            raise ValueError("Expected RGB current image: " + str(target))
                    digest = sha(target)
                    inputs[str(target)] = digest
                    case["images"][name] = dict(path=str(target), sha256=digest, size=size)
                projection.crop_boxes({n: v["size"] for n, v in case["images"].items()})
            except (KeyError, ValueError, OSError) as exc:
                case["issues"].append(str(exc))
        if found != set(wanted) or sha(path) != expected:
            raise RuntimeError("Source log token mismatch or concurrent change: " + log)
    sources = {name: sha(REPO / name) for name in (
        "navsim/agents/corridor_projection.py", "navsim/planning/script/run_corridor_projection.py",
        "scripts/drivesuprim/evaluation/run_corridor_projection.sh",
        "navsim/agents/drivesuprim/drivesuprim_features.py")}
    snapshot = dict(protocol="current_camera_corridor_projection_v1", parent=str(root),
        parent_identity_sha256=sha(root / "identity.json"), inputs=inputs, sources=sources, cases=cases,
        output_size=list(config[2:]), calibration_contract=projection.CALIBRATION_CONTRACT,
        references=REFERENCES, python=platform.python_version(), numpy=np.__version__, pillow=pillow_version,
        shapely=shapely.__version__,
        note=NOTE)
    marker = out / "identity.json"
    if marker.exists():
        if read(marker) != snapshot:
            raise RuntimeError("Projection inputs/code/runtime changed; use a new --out")
        snapshot = read(marker)  # Do not reserialize a hash-bound identity.
    else:
        if any(p.name != ".run.lock" for p in out.iterdir()):
            raise ValueError("Nonempty projection output without identity")
        write(marker, snapshot)
    plan = dict(status="planned", cases=len(cases), case_kinds=dict(Counter(c["kind"] for c in cases)),
        sessions=len({c["metadata"]["session_id"] for c in cases}),
        current_images=sum(len(c.get("images", {})) for c in cases),
        blocked_cases={c["token"]: c["issues"] for c in cases if c["issues"]},
        calibration_contract=projection.CALIBRATION_CONTRACT, output_size=snapshot["output_size"],
        additional_download_GB=0, new_training_runs=0, fresh_holdout_processed=0, note=NOTE)
    write(out / "plan.json", plan)
    return snapshot, plan


def audit_case(case, packet, output_size):
    observation = case["observation"]
    b = int(packet["baseline_position"])
    origin = packet["rollout_poses"][b, 0]
    r = projection.rotation(observation["ego_rotation"])
    yaw = np.arctan2(r[1, 0], r[0, 0])
    if (not np.allclose(origin[:2], observation["ego_translation"][:2], atol=1e-5, rtol=0)
            or abs(np.arctan2(np.sin(origin[2]-yaw), np.cos(origin[2]-yaw))) > 1e-5):
        raise ValueError("Current camera ego and geometry packet origin disagree")
    chosen = geometry.decide(packet, "rollout")["chosen"]
    if (int(packet["candidate_ids"][chosen]) != int(case["decision"]["candidate_id"])
            or int(packet["local_indices"][chosen]) != int(case["decision"]["candidate_local"])):
        raise ValueError("Geometry decision changed")
    positions = [b] if b == chosen else [b, chosen]
    xy = geometry.corners(packet["rollout_poses"][positions], packet["vehicle_dimensions"])
    points = projection.ground_ego_points(xy, observation)
    sizes = {name: info["size"] for name, info in case["images"].items()}
    projected = projection.panorama_projection(points, observation["cameras"], sizes, output_size)
    in_fov = np.logical_or.reduce([p["in_crop"] for p in projected.values()])
    body = projection.current_body_mask(xy, origin, packet["vehicle_dimensions"])
    available = in_fov & ~body
    inside = projection.road_membership(xy, geometry.decode_polygons(packet["polygon_bytes"], packet["polygon_offsets"]))
    actions = []
    for j, position in enumerate(positions):
        failures = ~inside[j]
        steps = np.flatnonzero(failures.any(axis=1))
        first = int(steps[0]) if len(steps) else -1
        actions.append(dict(action="baseline" if position == b else "chosen", candidate_id=int(packet["candidate_ids"][position]),
            failed_corner_samples=int(failures.sum()), projectable_failed_corner_samples=int((failures & available[j]).sum()),
            unknown_failed_corner_samples=int((failures & ~available[j]).sum()),
            first_failure_step=first, first_failure_s=first*float(packet["sampling"][1]) if first >= 0 else None,
            first_failure_has_projectable_corner=bool((failures[first] & available[j, first]).any()) if first >= 0 else None,
            all_corner_samples=int(failures.size), projectable_corner_samples=int(available[j].sum())))
    numeric = dict(world_xy=xy, ego_xyz=points, inside_road=inside, in_fov=in_fov,
        current_body_mask=body, projectable=available, positions=np.array(positions),
        **{"panorama_" + name: value["panorama"] for name, value in projected.items()},
        **{"in_crop_" + name: value["in_crop"] for name, value in projected.items()})
    return actions, numeric


def render(case, values, actions, size, path):
    images = {}
    for name, info in case["images"].items():
        if sha(Path(info["path"])) != info["sha256"]:
            raise RuntimeError("Current image changed: " + info["path"])
        with Image.open(info["path"]) as image:
            images[name] = image.convert("RGB")
    boxes = projection.crop_boxes({n: i.size for n, i in images.items()})
    cropped = [images[n].crop(boxes[n]) for n in projection.CAMERAS]
    panorama = Image.new("RGB", (sum(i.width for i in cropped), cropped[0].height))
    x = 0
    for image in cropped:
        panorama.paste(image, (x, 0))
        x += image.width
    panorama = panorama.resize(tuple(size), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size[0], size[1]+48), "#15202b")
    canvas.paste(panorama, (0, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), case["token"] + "  " + case["kind"] + "  DIAGNOSTIC / FLAT PLANE / OCCLUSION UNKNOWN", fill="white")
    draw.text((10, 25), "red=baseline  green=chosen  yellow cross=map-outside  gray=current-body unknown  FOV only", fill="white")
    offset = 0
    for name, image in zip(projection.CAMERAS, cropped):
        pixel = round(offset*size[0]/sum(i.width for i in cropped))
        if offset:
            draw.line((pixel, 48, pixel, size[1]+48), fill="#a0a0a0", width=1)
        draw.text((pixel+8, 54), name.upper(), fill="white", stroke_width=1, stroke_fill="black")
        offset += image.width
    for j, action in enumerate(actions):
        color = "#f05050" if action["action"] == "baseline" else "#40e080"
        steps = set(np.linspace(0, values["world_xy"].shape[1]-1, 7, dtype=int).tolist())
        if action["first_failure_step"] >= 0:
            steps.add(action["first_failure_step"])
        for name in projection.CAMERAS:
            uv, visible = values["panorama_" + name][j], values["in_crop_" + name][j]
            for step in sorted(steps):
                for a, z in projection.FOOTPRINT_EDGES:
                    if visible[step, a] and visible[step, z]:
                        p, q = uv[step, a] + [0, 48], uv[step, z] + [0, 48]
                        edge_color = "#a0a0a0" if values["current_body_mask"][j, step, [a, z]].any() else color
                        draw.line([tuple(p), tuple(q)], fill=edge_color, width=2)
                for corner in range(4):
                    if visible[step, corner] and not values["inside_road"][j, step, corner]:
                        px, py = uv[step, corner] + [0, 48]
                        point_color = "#a0a0a0" if values["current_body_mask"][j, step, corner] else "yellow"
                        draw.line((px-4, py-4, px+4, py+4), fill=point_color, width=2)
                        draw.line((px-4, py+4, px+4, py-4), fill=point_color, width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def execute(stage, root, out):
    snapshot, plan = initialize(root, out)
    if stage == "plan":
        return plan
    index_path = out / "render_index.json"
    index = read(index_path) if index_path.exists() else {}
    cases = snapshot["cases"]
    if set(index) - {c["token"] for c in cases}:
        raise ValueError("Unexpected projection case membership")
    for info in index.values():
        if info["identity_sha256"] != sha(out / "identity.json"):
            raise RuntimeError("Projection case identity changed")
        for name, digest in info["artifacts"].items():
            if not (out / name).is_file() or sha(out / name) != digest:
                raise RuntimeError("Committed projection artifact changed: " + name)
    if stage == "pilot":
        groups = [[c for c in cases if c["kind"] == k] for k in ("repair", "unrepairable", "matched_control")]
        cases = []
        while any(groups) and len(cases) < 4:
            for group in groups:
                if group and len(cases) < 4:
                    cases.append(group.pop(0))
    to_render = [] if stage == "report" else cases
    for case in to_render:
        token = case["token"]
        if token in index or case["issues"]:
            continue
        packet = parent.arrays(root / "packets" / (token + ".npz"))
        try:
            actions, numeric = audit_case(case, packet, snapshot["output_size"])
            record = dict(token=token, kind=case["kind"], session_id=case["metadata"]["session_id"], actions=actions)
            names = ["cases/"+token+".json", "cases/"+token+".npz", "preview/"+token+".png"]
            write(out / names[0], record)
            parent.save(out / names[1], **numeric)
            render(case, numeric, actions, snapshot["output_size"], out / names[2])
        except Exception as exc:
            write(out / "failure.json", dict(status="projection_failed", token=token, error=str(exc),
                identity_sha256=sha(out / "identity.json")))
            raise
        index[token] = dict(identity_sha256=sha(out / "identity.json"), artifacts={n: sha(out / n) for n in names})
        write(index_path, index)
        print(f"PROJECTION {len(index)}/{len(snapshot['cases'])} {token}", flush=True)
    records = [read(out / ("cases/"+t+".json")) for t in sorted(index)]
    failed_baselines = [r["actions"][0] for r in records if r["kind"] != "matched_control"]
    report = dict(status="complete" if len(index) == len(snapshot["cases"]) else "partial",
        selected_cases=len(snapshot["cases"]), rendered_cases=len(index),
        case_kinds=dict(Counter(r["kind"] for r in records)), blocked_cases=plan["blocked_cases"],
        failing_baselines_audited=len(failed_baselines),
        failing_baselines_with_any_projectable_failure=sum(a["projectable_failed_corner_samples"] > 0 for a in failed_baselines),
        failing_baselines_with_projectable_first_failure=sum(bool(a["first_failure_has_projectable_corner"]) for a in failed_baselines),
        failed_corner_samples=sum(a["failed_corner_samples"] for a in failed_baselines),
        projectable_failed_corner_samples=sum(a["projectable_failed_corner_samples"] for a in failed_baselines),
        human_projection_review_required=True, calibration_contract=projection.CALIBRATION_CONTRACT,
        additional_download_GB=0, new_training_runs=0, fresh_holdout_processed=0,
        identity_sha256=sha(out / "identity.json"), note=NOTE)
    write(out / "report.json", report)
    cards = ["<html><meta charset='utf-8'><title>Corridor current-camera audit</title><body>",
        "<h1>Current-camera corridor diagnostic</h1><p>"+html.escape(NOTE)+"</p>"]
    for r in records:
        token = html.escape(r["token"])
        cards.append(f"<h2>{token} {html.escape(r['kind'])}</h2><img style='width:100%;max-width:1600px' src='preview/{token}.png'>")
    cards.append("</body></html>")
    (out / "index.html").write_text("\n".join(cards), encoding="utf-8")
    return report


def main():
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory/experiments"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "pilot", "all", "report"))
    parser.add_argument("--geometry", type=Path, default=base / "corridor_geometry_v2")
    parser.add_argument("--out", type=Path, default=base / "corridor_projection_v1")
    args = parser.parse_args()
    root, out = args.geometry.resolve(), args.out.resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("Separate projection output required")
    with parent.locked(out):
        result = execute(args.stage, root, out)
    print(json.dumps(result, indent=2), flush=True)
    if args.stage != "plan" and result.get("blocked_cases"):
        raise SystemExit("Projection cases blocked; inspect plan.json. No automatic calibration fallback.")


if __name__ == "__main__":
    main()
