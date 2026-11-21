"""Prepare portable train-only geometry and frozen R34 weights on Astra."""
import argparse
import json
import os
from pathlib import Path
import re

import numpy as np

from navsim.agents import corridor_geometry as geom
from navsim.agents import corridor_projection as proj
from navsim.agents import drivemem_residual_memory as model
from navsim.planning.script import run_corridor_projection as camera
from navsim.planning.script import run_drivemem_dac_window as audit
from navsim.planning.script import run_drivemem_residual_memory as io

PREFIX = "model.teacher.model._backbone.image_encoder."
DAC = 1 + model.METRICS.index("drivable_area_compliance")
EDGE_M = .5


def r34_weights(path):
    import torch

    # The existing Astra model checkpoint is a trusted local Lightning file.
    # The portable payload is tensor-only NPZ, not a pickle checkpoint.
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    weights = {}
    for key, value in state.items():
        key = key.removeprefix("agent.")
        if key.startswith(PREFIX):
            name = key[len(PREFIX):]
            if not name.startswith(("layer4.", "fc.")):
                weights[name] = value.detach().cpu().numpy()
    if not all(key in weights for key in ("conv1.weight", "layer1.0.conv1.weight", "layer3.5.conv2.weight")):
        raise ValueError("Expected the frozen DriveSuprim R34 teacher encoder")
    return weights


def near_road_edge(xy, polygons, eligible):
    """Diagnostic near-edge negatives on candidates that pass the full DAC."""
    import shapely
    from shapely.geometry import Point
    from shapely.ops import unary_union

    result = np.zeros(xy.shape[:-1], dtype=bool)
    if eligible.any():
        boundary = unary_union([p if p.is_valid else p.buffer(0) for p in polygons]).boundary
        points = xy[eligible]
        if hasattr(shapely, "points"):
            distance = shapely.distance(shapely.points(points), boundary)
        else:
            distance = np.array([boundary.distance(Point(point)) for point in points])
        result[eligible] = distance <= EDGE_M
    return result


def packet_arrays(packet, example, observation, image_size):
    sample = example["sample"]
    ix = example["label_indices"]
    ids = sample["candidate_ids"][ix]
    base = int(packet["baseline_position"])
    if (not np.array_equal(ids, packet["candidate_ids"]) or not np.array_equal(ix, packet["local_indices"])
            or int(sample["selected_local"]) != int(ix[base])):
        raise ValueError("Packet, features and labels refer to different candidates")
    geom.validate_packet(packet)
    poses = packet["rollout_poses"]
    dims = packet["vehicle_dimensions"]
    dt = float(packet["sampling"][1])
    times = np.arange(poses.shape[1])*dt
    if len(times) != 41 or abs(float(times[-1])-4.) > 1e-8:
        raise ValueError("Expected the original 4 s / 10 Hz scorer sampling")
    xy = geom.corners(poses, dims)
    polygons = geom.decode_polygons(packet["polygon_bytes"], packet["polygon_offsets"])
    inside = proj.road_membership(xy, polygons)
    raw = np.asarray(example["label_raw"], dtype=np.float64)
    if not np.array_equal(inside.all((1, 2)), raw[:, DAC] == 1):
        raise ValueError("Stored raw DAC does not match full-horizon geometry")
    # Fixed t=.5..4 s: remove the near-body lead-in and retain later exits.
    use = times >= .5-1e-8
    xyz = xy[:, use]
    sizes = {name: info["size"] for name, info in observation["images"].items()}
    current = observation["observation"]
    proj.require_rollout_alignment(poses[base, 0], current)
    views = proj.panorama_projection(proj.ground_ego_points(xyz, current), current["cameras"], sizes, image_size)
    body = proj.current_body_mask(xyz, poses[base, 0], dims)
    # When cameras overlap, choose the projection furthest from its crop edge.
    # This choice depends only on calibration and trajectory geometry.
    boxes = proj.crop_boxes(sizes)
    widths = [boxes[name][2] - boxes[name][0] for name in proj.CAMERAS]
    bounds = np.r_[0., np.cumsum(widths)] * image_size[0] / sum(widths) - .5
    uv = np.zeros((*xyz.shape[:-1], 2), dtype=np.float32)
    view_id = np.full(xyz.shape[:-1], -1, dtype=np.int8)
    best_margin = np.full(view_id.shape, -np.inf)
    for j, name in enumerate(proj.CAMERAS):
        pix = views[name]["panorama"]
        margin = np.minimum.reduce((pix[..., 0]-bounds[j], bounds[j+1]-pix[..., 0],
                                    pix[..., 1]+.5, image_size[1]-.5-pix[..., 1]))
        take = views[name]["in_crop"] & ~body & (margin > best_margin)
        uv[take], view_id[take], best_margin[take] = pix[take], j, margin[take]
    # Represent footprint geometry in initial ego SE2, without world coordinates.
    x, y, heading = poses[base, 0]
    cosine, sine = np.cos(heading), np.sin(heading)
    ego_xy = (xyz - [x, y]) @ np.array([[cosine, -sine], [sine, cosine]])
    inputs = dict(candidate_ids=ids, local_indices=ix, baseline=np.array(base), order=np.argsort(
        -packet["selection_scores"], kind="stable"), uv=uv, view=view_id,
        ego_xy=ego_xy.astype(np.float32), times=times[use].astype(np.float32),
        decoder=sample["decoder_states"][ix].astype(np.float32),
        teacher_dac=sample["head_drivable_area_compliance"][ix].astype(np.float32))
    # Map queries are stored only on the label side of the bundle.
    near_edge = near_road_edge(xyz, polygons, (view_id >= 0) & (raw[:, DAC] == 1)[:, None, None])
    return inputs, dict(outside=~inside[:, use], near_edge=near_edge, raw=raw)


def prepare(root, checkpoint, out):
    root, out, checkpoint = root.resolve(), out.resolve(), checkpoint.resolve()
    if root.is_relative_to(out) or out.is_relative_to(root):
        raise ValueError("Use a separate output directory")
    _, bundle, metadata, _ = camera.completed_run(root)
    if out.is_relative_to(bundle) or bundle.is_relative_to(out):
        raise ValueError("Output and source bundle must be separate")
    observations, blocked, bindings, size = audit.current_cameras(bundle, metadata)
    if blocked:
        raise ValueError(f"Repair missing camera data before exporting the fixed cohort: {blocked}")
    index = io.read(root/"packet_index.json")
    examples, _ = camera.parent.bundles.load_bundle(bundle)
    lookup = {e["metadata"]["token"]: e for e in examples}
    if set(index) != set(metadata) or set(lookup) != set(metadata):
        raise ValueError("Incomplete frozen geometry")
    split = model.partition(list(metadata.values()))
    sources = dict(geometry_sha256=io.sha(root/"report.json"), checkpoint_sha256=io.sha(checkpoint),
                   image_size=list(size), groups=split, camera_bindings=bindings,
                   code={p.name: io.sha(p) for p in map(Path, (__file__, proj.__file__, audit.__file__))})
    with io.locked(out):
        io.freeze(out/"prepare_identity.json", sources)
        if (out/"manifest.json").exists():
            report = io.read(out/"manifest.json")
            io.files_checked(out, report)
            return report
        io.save(out/"r34.npz", **r34_weights(checkpoint))
        files = {"r34.npz": io.sha(out/"r34.npz")}
        image_hashes, rows = {}, []
        for token, meta in sorted(metadata.items()):
            if not re.fullmatch(r"[a-f0-9]{16}", token):
                raise ValueError(f"Unexpected training token {token}")
            packet = camera.parent.arrays(root/"packets"/(token+".npz"))
            obs = observations[token]
            features, labels = packet_arrays(packet, lookup[token], obs, size)
            images = {}
            for name in proj.CAMERAS:
                relative = obs["observation"]["cameras"][name]["data_path"]
                path = obs["images"][name]["path"]
                if path not in image_hashes:
                    image_hashes[path] = io.sha(path)
                images[name] = dict(relative=relative, sha256=image_hashes[path], size=obs["images"][name]["size"])
            for section, arrays in (("inputs", features), ("labels", labels)):
                name = f"{section}/{token}.npz"
                io.save(out/name, **arrays)
                files[name] = io.sha(out/name)
            rows.append(dict(token=token, session_id=meta["session_id"], images=images))
        report = dict(protocol="drivemem_spatial_dac_r34_v1", images=list(proj.CAMERAS), image_size=list(size),
                      groups=split, cohort=rows, files=files, weight_source_sha256=sources["checkpoint_sha256"])
        io.write(out/"manifest.json", report)
        return report


if __name__ == "__main__":
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=base/"drivemem_memory/experiments/corridor_geometry_v2")
    parser.add_argument("--checkpoint", type=Path, default=base/"model_ckpt/drivesuprim_r34.ckpt")
    parser.add_argument("--out", type=Path, default=base/"drivemem_spatial_dac_v1")
    args = parser.parse_args()
    result = prepare(args.geometry, args.checkpoint, args.out)
    print(json.dumps(dict(scenes=len(result["cohort"]), sessions=sum(map(len, result["groups"])),
                          artifact=str(args.out), archive_bytes=sum((args.out/n).stat().st_size for n in result["files"]))))
