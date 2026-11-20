"""Audit projection coverage and attainable gain in the fixed [1, 2] s DAC window."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
from PIL import Image

from navsim.agents import corridor_geometry as geometry, corridor_projection as projection
from navsim.agents.drivemem import METRICS, SAFETY
from navsim.planning.script import run_corridor_projection as camera
from navsim.planning.script import run_drivemem_residual_memory as io

WINDOW = (1., 2.)
DAC_COLUMN = 1 + METRICS.index("drivable_area_compliance")
NOTE = "Privileged map and stored raw local PDM. Projection coverage is a FOV upper bound. Human review is diagnostic."


def current_cameras(bundle, metadata):
    records, config, manifests, pinned, log_hashes = camera.resolve_records(bundle, metadata)
    logs, sensors = (Path(path).resolve() for path in config[:2])
    groups = defaultdict(dict)
    observations, blocked = {}, {}
    inputs = {str(path): digest for path, (digest, _) in manifests.items()}
    for token in metadata:
        if token not in records:
            blocked[token] = "missing_frozen_frame_record"
            continue
        record = records[token]
        log = record["log_name"]
        if Path(log).name != log or not re.fullmatch(r"[A-Za-z0-9_.-]+", log):
            raise ValueError("Unsafe log name")
        groups[log][token] = record
    for log, wanted in sorted(groups.items()):
        path = (logs / (log + ".pkl")).resolve()
        expected = pinned.get(path, log_hashes.get(log))
        if not expected or not path.is_file():
            blocked.update({token: "missing_pinned_log" for token in wanted})
            continue
        if io.sha(path) != expected:
            raise ValueError(f"Frozen log changed: {path}")
        inputs[str(path)] = expected
        with path.open("rb") as stream:
            frames = pickle.load(stream)  # Existing trusted NAVSIM metadata.
        seen = set()
        for frame in frames:
            token = frame["token"]
            if token not in wanted:
                continue
            if token in seen or int(frame["timestamp"]) != int(wanted[token]["timestamp_us"]):
                raise ValueError(f"Frozen frame alignment failed: {token}")
            seen.add(token)
            try:
                observation = projection.current_observation(frame)
                images = {}
                for name in projection.CAMERAS:
                    image_path = camera.safe_image(sensors, observation["cameras"][name]["data_path"])
                    with Image.open(image_path) as image:
                        if image.mode not in ("RGB", "RGBA"):
                            raise ValueError("Expected RGB image")
                        images[name] = dict(path=str(image_path), size=list(image.size))
                projection.crop_boxes({name: image["size"] for name, image in images.items()})
                observations[token] = dict(observation=observation, images=images)
            except (KeyError, ValueError, OSError) as exc:
                blocked[token] = str(exc)
        blocked.update({token: "missing_current_frame" for token in set(wanted)-seen})
        if io.sha(path) != expected:
            raise ValueError(f"Log changed while reading: {path}")
    return observations, blocked, inputs, tuple(config[2:])


def geometry_and_projection(packet, camera_data, size):
    geometry.validate_packet(packet)
    poses = packet["rollout_poses"]
    times = np.arange(poses.shape[1]) * float(packet["sampling"][1])
    window = (times >= WINDOW[0]-1e-8) & (times <= WINDOW[1]+1e-8)
    if not window.any() or times[-1] < WINDOW[1]-1e-8:
        raise ValueError("Rollout does not cover the fixed DAC window")
    dimensions = packet["vehicle_dimensions"]
    xy = geometry.corners(poses, dimensions)
    polygons = geometry.decode_polygons(packet["polygon_bytes"], packet["polygon_offsets"])
    inside = projection.road_membership(xy, polygons)
    points = xy[:, window]
    origin = poses[int(packet["baseline_position"]), 0]
    body = projection.current_body_mask(points, origin, dimensions)
    result = dict(world_xy=points, inside_road=inside[:, window], current_body_mask=body,
                  in_fov=np.zeros(body.shape, dtype=bool))
    if camera_data is not None:
        observation = camera_data["observation"]
        rotation = projection.rotation(observation["ego_rotation"])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        if (not np.allclose(origin[:2], observation["ego_translation"][:2], atol=1e-5, rtol=0)
                or abs(np.arctan2(np.sin(origin[2]-yaw), np.cos(origin[2]-yaw))) > 1e-5):
            raise ValueError("Current camera and rollout origins disagree")
        xyz = projection.ground_ego_points(points, observation)
        sizes = {name: info["size"] for name, info in camera_data["images"].items()}
        views = projection.panorama_projection(xyz, observation["cameras"], sizes, size)
        for name, view in views.items():
            result["in_fov"] |= view["in_crop"]
            result["in_crop_" + name] = view["in_crop"]
            result["panorama_" + name] = view["panorama"]
    return times, window, inside, result


def choose(packet, inside, projectable):
    baseline = int(packet["baseline_position"])
    clear = inside.all(axis=(1, 2))
    eligible = clear & projectable.all(axis=(1, 2))
    witness = bool((~inside[baseline] & projectable[baseline]).any())
    order = np.argsort(-packet["selection_scores"], kind="stable")
    first = next((int(i) for i in order if clear[i]), baseline)
    covered = next((int(i) for i in order if eligible[i]), baseline)
    return (baseline if clear[baseline] else first, covered if witness else baseline, eligible, witness)


def render_case(out, token, camera_data, projection_data, baseline, replacement, size):
    positions = [baseline] if baseline == replacement else [baseline, replacement]
    values = {key: value[positions] for key, value in projection_data.items()}
    actions = []
    for offset, position in enumerate(positions):
        bad_steps = np.flatnonzero((~values["inside_road"][offset]).any(axis=1))
        actions.append(dict(action="baseline" if position == baseline else "chosen",
                            first_failure_step=int(bad_steps[0]) if len(bad_steps) else -1))
    for info in camera_data["images"].values():
        info["sha256"] = io.sha(info["path"])
    relative = f"preview/{token}.png"
    camera.render(dict(token=token, kind="1-2 s", **camera_data), values, actions, size, out / relative)
    return relative


def scene_row(token, session, packet, times, window, inside, data, labels, issue):
    baseline = int(packet["baseline_position"])
    fail = ~inside[baseline].all(axis=1)
    projectable = data["in_fov"] & ~data["current_body_mask"]
    map_choice, fov_choice, eligible, witness = choose(packet, data["inside_road"], projectable)
    delta = labels - labels[baseline]
    safety_columns = [j+1 for j in SAFETY]
    safe = (delta[:, safety_columns] >= -1e-8).all(axis=1)
    safe_eligible = eligible & safe & (labels[:, DAC_COLUMN] == 1)
    useful_oracle = bool(witness and np.any(safe_eligible & (delta[:, 0] >= .005 - 1e-8)))
    row = dict(token=token, session_id=session, baseline_id=int(packet["candidate_ids"][baseline]),
               full_dac_failure=bool(fail.any()), before_window=bool(fail[times < WINDOW[0]-1e-8].any()),
               in_window=bool(fail[window].any()), after_window=bool(fail[times > WINDOW[1]+1e-8].any()),
               first_failure_s=float(times[np.argmax(fail)]) if fail.any() else None,
               baseline_projectable_exit=witness, fov_eligible_useful_alternative=useful_oracle,
               baseline_window_points=int(projectable[baseline].size),
               baseline_projectable_points=int(projectable[baseline].sum()),
               baseline_body_points=int(data["current_body_mask"][baseline].sum()),
               baseline_outside_crop_points=int((~data["in_fov"][baseline] & ~data["current_body_mask"][baseline]).sum()),
               camera_issue=issue, baseline_pdm=float(labels[baseline, 0]))
    for name, choice in (("map", map_choice), ("fov", fov_choice)):
        row.update({f"{name}_id": int(packet["candidate_ids"][choice]),
                    f"{name}_replaced": choice != baseline,
                    f"{name}_gain_pp": float(100 * delta[choice, 0]),
                    f"{name}_harm": not bool(safe[choice]),
                    f"{name}_useful": bool(safe[choice] and delta[choice, 0] >= .005 - 1e-8 and fail[window].any()
                                           and np.all(data["inside_road"][choice]) and labels[choice, DAC_COLUMN] == 1)})
        for j, metric in enumerate(METRICS):
            row[f"{name}_delta_{metric}"] = float(delta[choice, j+1])
    row["safe_fov_oracle_gain_pp"] = float(100 * max(0., delta[safe_eligible, 0].max(initial=0.))) if witness else 0.
    return row, fov_choice


def summarize(frame, prefix):
    count = frame[f"{prefix}_useful"]
    values = frame[f"{prefix}_gain_pp"]
    return dict(replacements=int(frame[f"{prefix}_replaced"].sum()), useful_DAC_repairs=int(count.sum()),
                useful_sessions=int(frame.loc[count, "session_id"].nunique()),
                gain_macro_pp=float(values.groupby(frame.session_id).mean().mean()), gain_micro_pp=float(values.mean()),
                safety_harm=int(frame[f"{prefix}_harm"].sum()), pdm_losses=int((values < -1e-6).sum()))


def passes_gate(result):
    return (result["useful_DAC_repairs"] >= 5 and result["useful_sessions"] >= 3
            and result["gain_macro_pp"] > 0 and result["safety_harm"] == 0)


def run(root, out):
    root, out = root.resolve(), out.resolve()
    identity, bundle, metadata, decisions = camera.completed_run(root)
    records, blocked, bindings, size = current_cameras(bundle, metadata)
    for path in (root, bundle.resolve(), *(Path(p).resolve() for p in bindings)):
        if out == path or out.is_relative_to(path) or path.is_relative_to(out):
            raise ValueError("Output must be separate from frozen inputs")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", t) for t in metadata):
        raise ValueError("Unsafe scene token")
    examples, _ = camera.parent.bundles.load_bundle(bundle)
    by_token = {e["metadata"]["token"]: e for e in examples}
    index = io.read(root / "packet_index.json")
    if set(index) != set(metadata) or set(by_token) != set(metadata):
        raise ValueError("Incomplete frozen cohort")
    snapshot = dict(protocol="drivemem_dac_window_1_2_v1", geometry=str(root), window=list(WINDOW),
                    geometry_report_sha256=io.sha(root/"report.json"), metadata_bindings=bindings,
                    camera_inputs_digest=io.model.digest(dict(records=records, blocked=blocked, output_size=size)),
                    code_sha256={str(p): io.sha(p) for p in map(Path, (__file__, projection.__file__, camera.__file__))})
    controls = {c["token"] for c in camera.select_cases(metadata, decisions) if c["kind"] == "matched_control"}
    with io.locked(out):
        if not (out/"identity.json").exists() and any(p.name != ".run.lock" for p in out.iterdir()):
            raise ValueError("Nonempty output without audit identity")
        io.freeze(out/"identity.json", snapshot)
        if (out/"commit.json").exists():
            commit = io.read(out/"commit.json")
            io.files_checked(out, commit)
            for path, digest in commit["preview_inputs"].items():
                if io.sha(path) != digest:
                    raise ValueError("Reviewed camera image changed: " + path)
            return io.read(out/"summary.json")
        rows, points, review_rows, artifacts, preview_inputs = [], defaultdict(list), [], {}, {}
        saved_times = None
        for token in sorted(metadata):
            path = root/"packets"/(token+".npz")
            packet = camera.parent.arrays(path)
            example = by_token[token]
            local = example["label_indices"]
            ids = example["sample"]["candidate_ids"][local]
            if not np.array_equal(local, packet["local_indices"]) or not np.array_equal(ids, packet["candidate_ids"]):
                raise ValueError(f"Geometry and outcome candidate alignment failed: {token}")
            if int(example["sample"]["selected_local"]) != int(local[int(packet["baseline_position"])]):
                raise ValueError(f"Frozen baseline differs: {token}")
            camera_data = records.get(token)
            try:
                times, window, inside, data = geometry_and_projection(packet, camera_data, size)
            except ValueError as exc:
                if camera_data is None:
                    raise
                blocked[token] = str(exc)
                camera_data = None
                times, window, inside, data = geometry_and_projection(packet, None, size)
            labels = example["label_raw"]
            if saved_times is None:
                saved_times = times[window]
            elif not np.array_equal(saved_times, times[window]):
                raise ValueError("Sampling differs across frozen packets")
            if not np.array_equal(inside.all(axis=(1, 2)), labels[:, DAC_COLUMN] == 1):
                raise ValueError(f"Full DAC parity failed: {token}")
            row, choice = scene_row(token, metadata[token]["session_id"], packet, times, window, inside, data, labels, blocked.get(token, ""))
            rows.append(row)
            points["candidate_ids"].append(ids)
            points["inside_road"].append(data["inside_road"])
            points["in_fov"].append(data["in_fov"])
            points["under_current_body"].append(data["current_body_mask"])
            if camera_data and (row["full_dac_failure"] or token in controls):
                filename = render_case(out, token, camera_data, data, int(packet["baseline_position"]), choice, size)
                artifacts[filename] = io.sha(out/filename)
                preview_inputs.update({v["path"]: v["sha256"] for v in camera_data["images"].values()})
                if row["fov_replaced"]:
                    review_rows.append(dict(token=token, baseline_id=row["baseline_id"], replacement_id=row["fov_id"],
                                            preview=filename, preview_sha256=artifacts[filename],
                                            baseline_witness="unknown", replacement_corridor="unknown", note=""))
        frame = pd.DataFrame(rows)
        frame.to_csv(out/"per_scene.csv", index=False)
        cols = ["token", "baseline_id", "replacement_id", "preview", "preview_sha256", "baseline_witness",
                "replacement_corridor", "note"]
        pd.DataFrame(review_rows, columns=cols).to_csv(out/"review_template.csv", index=False)
        io.save(out/"window_points.npz", tokens=frame.token.to_numpy(dtype=str), times_s=saved_times,
                **{key: np.stack(values) for key, values in points.items()})
        reference = summarize(frame, "fov")
        report = dict(status="complete", scenes=len(frame), sessions=int(frame.session_id.nunique()), window_s=list(WINDOW),
                      full_baseline_DAC_failures=int(frame.full_dac_failure.sum()),
                      failure_times={key: int(frame[key].sum()) for key in ("before_window", "in_window", "after_window")},
                      baseline_projectable_exit=int(frame.baseline_projectable_exit.sum()), blocked_camera_scenes=blocked,
                      projectable_points=int((np.stack(points["in_fov"]) & ~np.stack(points["under_current_body"])).sum()),
                      total_window_points=int(np.stack(points["inside_road"]).size),
                      map_reference=summarize(frame, "map"), fov_reference=reference,
                      visible_review_pending=len(review_rows), fov_prescreen_passed=passes_gate(reference),
                      safe_fov_oracle_gain_macro_pp=float(frame.groupby("session_id").safe_fov_oracle_gain_pp.mean().mean()),
                      new_scorer_calls=0, new_training_runs=0, note=NOTE)
        io.write(out/"summary.json", report)
        for name in ("identity.json", "summary.json", "per_scene.csv", "review_template.csv", "window_points.npz"):
            artifacts[name] = io.sha(out/name)
        io.write(out/"commit.json", dict(files=artifacts, preview_inputs=preview_inputs))
        return report


def review(out, annotations):
    with io.locked(out):
        commit = io.read(out/"commit.json")
        io.files_checked(out, commit)
        for path, digest in commit["preview_inputs"].items():
            if io.sha(path) != digest:
                raise ValueError("Reviewed camera image changed: " + path)
        template = pd.read_csv(out/"review_template.csv", dtype=str, keep_default_na=False)
        labels = pd.read_csv(annotations, dtype=str, keep_default_na=False)
        keys = ["token", "baseline_id", "replacement_id", "preview", "preview_sha256"]
        if not template[keys].equals(labels[keys]):
            raise ValueError("Review must correspond to the frozen template and previews")
        for key in ("baseline_witness", "replacement_corridor"):
            if not labels[key].isin(["visible", "occluded", "ambiguous", "unknown"]).all():
                raise ValueError(f"Invalid visibility annotation in {key}")
        taken = labels.baseline_witness.eq("visible") & labels.replacement_corridor.eq("visible")
        frame = pd.read_csv(out/"per_scene.csv", dtype={"token": str})
        accepted = frame.token.isin(labels.loc[taken, "token"])
        for field in ("fov_replaced", "fov_useful", "fov_harm"):
            frame[field] &= accepted
        frame.loc[~accepted, "fov_gain_pp"] = 0.
        result = summarize(frame, "fov")
        summary = dict(status="reviewed", annotations_sha256=io.sha(annotations), reviewed_proposals=len(labels),
                       both_visible=int(taken.sum()), unresolved=int(
                           (labels.baseline_witness.eq("unknown") | labels.replacement_corridor.eq("unknown")).sum()),
                       reviewed_reference=result, proceed_to_verifier=passes_gate(result), note=NOTE)
        io.write(out/"review_report.json", summary)
        return summary


def main():
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory/experiments"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=base/"corridor_geometry_v2")
    parser.add_argument("--out", type=Path, default=base/"drivemem_dac_window_v1")
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    result = review(args.out, args.review) if args.review else run(args.geometry, args.out)
    compact = {k: v for k, v in result.items() if k != "blocked_camera_scenes"}
    if "blocked_camera_scenes" in result:
        compact["blocked_camera_scene_count"] = len(result["blocked_camera_scenes"])
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
