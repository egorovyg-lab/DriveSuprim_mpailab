"""Privileged static geometry experiment on frozen training candidates.

prepare reads trusted caches in a converter; infer reads only primitive geometry
packets. Full PDM/traffic scoring and future human/actor fields are not used.
"""
import argparse
from contextlib import contextmanager
import html
from importlib import metadata as package_metadata
import json
import lzma
import os
from pathlib import Path
import pickle
import platform
import time

import numpy as np
import pandas as pd

from navsim.agents import corridor_geometry as geometry
from navsim.agents import selector_training_bundle as bundles
from navsim.agents.drivemem import METRICS, token_seed
from navsim.planning.script.run_drivemem_action_transfer import record, summary
from navsim.planning.script.run_drivemem_expanded_utility import REPO, read, write, sha

NOTE = ("Privileged current static-HD-map positive control, NOT camera-only comparable or a new deployable primary. "
    "No future actor or human trajectories enter packets/decisions. Cached PDM labels remain evaluation only. "
    "Rollout checks deliberately reproduce the geometry used by the DAC target: DAC agreement is a consistency "
    "check, not independent learned perception or safety evidence. Only sampled vehicle corners are checked. "
    "Collision, direction, lights and TTC are not certified. Training sample is archive-selected and inspected. "
    "Local PDM, not EPDMS. Fresh holdout unopened.")
MODES = ("raw", "rollout")


@contextmanager
def locked(out):
    out.mkdir(parents=True, exist_ok=True)
    handle = (out / ".run.lock").open("a+b")
    acquired = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0, 2)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise RuntimeError("Another corridor writer is using " + str(out)) from exc
        yield
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def source_hashes():
    names = ("navsim/agents/corridor_geometry.py", "navsim/planning/script/run_corridor_geometry.py",
        "scripts/drivesuprim/evaluation/run_corridor_geometry.sh", "navsim/evaluate/pdm_score.py",
        "navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py",
        "navsim/planning/simulation/planner/pdm_planner/simulation/batch_lqr.py",
        "navsim/planning/simulation/planner/pdm_planner/simulation/batch_kinematic_bicycle.py",
        "navsim/planning/simulation/planner/pdm_planner/observation/pdm_occupancy_map.py",
        "navsim/planning/simulation/planner/pdm_planner/utils/pdm_array_representation.py",
        "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py")
    return {name: sha(REPO / name) for name in names}


def save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def resolve_inputs(bundle, examples):
    parent = read(bundle / "identity.json")
    tokens = {e["metadata"]["token"] for e in examples}
    caches, configs, bound = {t: [] for t in tokens}, [], {}
    for name, digest in parent["inputs"].items():
        path = Path(name)
        if path.name not in ("cache_index.json", "source_config.json", "identity.json"):
            continue
        value = read(path)
        if sha(path) != digest:
            raise RuntimeError("Pinned geometry source changed: " + name)
        if path.name == "cache_index.json":
            bound[name] = digest
            for token in tokens & set(value):
                info = value[token]
                if not isinstance(info, dict) or set(("path", "sha256")) - set(info):
                    raise ValueError("Explicit cache file path/hash required: " + name)
                caches[token].append(dict(path=str(Path(info["path"]).resolve()), sha256=info["sha256"], index=name))
        elif "proposal_sampling" in value:
            cfg = value["proposal_sampling"]
            count, interval = int(cfg["num_poses"]), float(cfg["interval_length"])
            if count < 1 or interval <= 0 or not np.isfinite(interval):
                raise ValueError("Invalid pinned rollout sampling")
            configs.append((count, interval))
            bound[name] = digest
    missing = sorted(t for t, options in caches.items() if not options)
    if missing:
        raise ValueError("Missing pinned metric caches for training tokens: " + ",".join(missing[:10]))
    if not configs or len(set(configs)) != 1:
        raise ValueError("One consistent pinned proposal_sampling required")
    selected = {}
    for token, options in caches.items():
        # Resolve duplicates before viewing geometry or outcomes.
        options.sort(key=lambda item: (item["path"], item["index"]))
        chosen = options[0]
        if not Path(chosen["path"]).is_file() or sha(Path(chosen["path"])) != chosen["sha256"]:
            raise RuntimeError("Missing/changed selected metric cache: " + token)
        selected[token] = dict(**chosen, alternatives=len(options))
    return selected, configs[0], bound


def pilot_tokens(examples, count=8):
    groups = {}
    for e in examples:
        m = e["metadata"]
        groups.setdefault((m.get("map_name", "unknown"), m["session_id"]), []).append(m["token"])
    for values in groups.values():
        values.sort(key=lambda t: (token_seed(t, 20261110), t))
    result = []
    while len(result) < min(count, len(examples)):
        for key in sorted(groups):
            if groups[key] and len(result) < count:
                result.append(groups[key].pop(0))
    return result


def static_context(cache):
    # This is the entire cache-field allowlist. Never pass the cache to infer.
    return cache.ego_state, cache.drivable_area_map


def navsim_geometry(ego, road, trajectories, sampling):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    future_sampling = TrajectorySampling(num_poses=sampling[0], interval_length=sampling[1])
    path_sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    states = np.stack([get_trajectory_as_array(transform_trajectory(Trajectory(path, path_sampling), ego),
        future_sampling, ego.time_point) for path in trajectories])
    simulated = PDMSimulator(future_sampling).simulate_proposals(states, ego)
    indices = road.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
    polygons = [road[road.tokens[i]] for i in indices]
    vehicle = ego.car_footprint.vehicle_parameters
    return dict(raw_poses=states[..., StateIndex.STATE_SE2].copy(),
        rollout_poses=simulated[..., StateIndex.STATE_SE2].copy(),
        vehicle_dimensions=np.array([vehicle.length, vehicle.width, vehicle.rear_axle_to_center]),
        polygons=polygons)


def make_packet(cache, example, vocab, sampling, backend=navsim_geometry):
    sample, ix, meta = example["sample"], example["label_indices"], example["metadata"]
    if meta["role"] != "train":
        raise ValueError("Geometry converter accepts training scenes only")
    if vocab.shape[1:] != (40, 3) or not np.isfinite(vocab).all():
        raise ValueError("Expected frozen four-second vocabulary at 10 Hz")
    if not np.array_equal(ix, np.argsort(-sample["selection_scores"], kind="stable")[:16]):
        raise ValueError("Frozen top16 alignment changed")
    ego, road = static_context(cache)
    ids = sample["candidate_ids"][ix]
    physical = backend(ego, road, vocab[ids].copy(), sampling)
    data, offsets = geometry.encode_polygons(physical.pop("polygons"))
    packet = dict(candidate_ids=ids.copy(), local_indices=ix.copy(),
        baseline_position=np.array(int(np.flatnonzero(ix == int(sample["selected_local"]))[0])),
        selection_scores=sample["selection_scores"][ix].copy(), **physical,
        polygon_bytes=data, polygon_offsets=offsets, sampling=np.array(sampling))
    geometry.validate_packet(packet)
    return packet


def freeze_identity(out, bundle, examples, selected, sampling, bindings):
    import shapely
    try:
        nuplan_version = package_metadata.version("nuplan-devkit")
    except package_metadata.PackageNotFoundError:
        nuplan_version = "not-installed"
    identity = dict(protocol="privileged_static_corridor_v1", privileged=True, camera_only_comparable=False,
        inputs={str(bundle / n): sha(bundle / n) for n in ("identity.json", "bundle_manifest.json")},
        sources=source_hashes(), cache_indexes=bindings, cache_selection=selected, sampling=list(sampling),
        tokens=[e["metadata"]["token"] for e in examples], pilot_tokens=pilot_tokens(examples),
        python=platform.python_version(), numpy=np.__version__, shapely=shapely.__version__,
        nuplan_version=nuplan_version, note=NOTE)
    marker = out / "identity.json"
    if marker.exists() and read(marker) != identity:
        raise RuntimeError("Corridor inputs/code/runtime changed; use a new --out root")
    if out.exists() and not marker.exists() and any(p.name != ".run.lock" for p in out.iterdir()):
        raise ValueError("Nonempty corridor output without identity")
    write(marker, identity)
    return identity


def checked_packets(out, identity):
    index = read(out / "packet_index.json") if (out / "packet_index.json").exists() else {}
    if set(index) - set(identity["tokens"]):
        raise ValueError("Unexpected geometry packet tokens")
    for token, info in index.items():
        if (info["identity_sha256"] != sha(out / "identity.json")
                or sha(out / "packets" / (token + ".npz")) != info["sha256"]):
            raise RuntimeError("Committed geometry packet changed: " + token)
    return index


def prepare(out, identity, examples, vocab, limit=None):
    index = checked_packets(out, identity)
    lookup = {e["metadata"]["token"]: e for e in examples}
    order = identity["pilot_tokens"] + sorted(set(lookup) - set(identity["pilot_tokens"]))
    wanted = order if limit is None else identity["pilot_tokens"][:limit]
    started, new = time.monotonic(), 0
    for token in wanted:
        if token in index:
            continue
        info = identity["cache_selection"][token]
        if sha(Path(info["path"])) != info["sha256"]:
            raise RuntimeError("Pinned cache changed: " + token)
        with lzma.open(info["path"], "rb") as stream:
            cache = pickle.load(stream)  # Trusted, existing local NAVSIM cache only.
        packet = make_packet(cache, lookup[token], vocab, tuple(identity["sampling"]))
        del cache
        path = out / "packets" / (token + ".npz")
        save(path, **packet)
        index[token] = dict(sha256=sha(path), identity_sha256=sha(out / "identity.json"))
        write(out / "packet_index.json", index)
        new += 1
        print(f"CORRIDOR PACKET {len(index)}/{len(examples)} {token}", flush=True)
    return dict(prepared_scenes=len(index), selected_scenes=len(examples), newly_prepared=new,
        seconds=time.monotonic() - started, privileged=True, camera_only_comparable=False)


def infer(out, identity):
    index = checked_packets(out, identity)
    if not index:
        raise ValueError("No prepared geometry packets")
    decisions, checks = [], []
    for token in sorted(index):
        packet = arrays(out / "packets" / (token + ".npz"))
        for mode in MODES:
            result = geometry.decide(packet, mode)
            chosen = result["chosen"]
            decisions.append(dict(method=mode, token=token, candidate_local=int(packet["local_indices"][chosen]),
                candidate_id=int(packet["candidate_ids"][chosen]), reason=result["reason"],
                baseline_passed=bool(result["passed"][int(packet["baseline_position"])])))
            for j, local in enumerate(packet["local_indices"]):
                checks.append(dict(method=mode, token=token, candidate_local=int(local),
                    candidate_id=int(packet["candidate_ids"][j]), passed=bool(result["passed"][j]),
                    outside_samples=int(result["outside_samples"][j]), first_outside=int(result["first_outside"][j])))
    snapshot = dict(identity_sha256=sha(out / "identity.json"), packet_index_sha256=sha(out / "packet_index.json"),
                    decisions=decisions, checks=checks)
    path = out / "decisions.json"
    if path.exists() and read(path) != snapshot:
        raise RuntimeError("Frozen geometry membership/decisions changed; use a separate --out for pilot/all")
    write(path, snapshot)
    return snapshot


def evaluate(out, identity, examples):
    state = read(out / "decisions.json")
    if state["identity_sha256"] != sha(out / "identity.json") or state["packet_index_sha256"] != sha(out / "packet_index.json"):
        raise ValueError("Geometry decisions refer to another input snapshot")
    lookup = {e["metadata"]["token"]: e for e in examples}
    rows, labels = [], {}
    for d in state["decisions"]:
        e = lookup[d["token"]]
        ix, sample = e["label_indices"], dict(e["sample"], token=d["token"], session_id=e["metadata"]["session_id"])
        if d["candidate_local"] not in ix or int(sample["candidate_ids"][d["candidate_local"]]) != d["candidate_id"]:
            raise ValueError("Geometry selection outside frozen candidates")
        outcomes = {str(int(local)): {v: dict(zip(bundles.LABEL_KEYS, e["label_" + v][j].tolist()))
                    for v in ("raw", "filtered")} for j, local in enumerate(ix)}
        rows.append({**d, **record(sample, outcomes, d["candidate_local"])})
        labels[d["token"]] = {int(local): float(e["label_raw"][j, 1 + METRICS.index("drivable_area_compliance")]) for j, local in enumerate(ix)}
    checks = pd.DataFrame(state["checks"])
    checks["stored_raw_DAC"] = [labels[r.token][r.candidate_local] for r in checks.itertuples()]
    checks["matches_raw_DAC"] = checks.passed.astype(float) == checks.stored_raw_DAC
    stats = summary(rows, 20261110)
    result = []
    for mode, value in stats.items():
        q = pd.DataFrame([r for r in rows if r["method"] == mode])
        c = checks[checks.method == mode]
        result.append(dict(method=mode, gain_macro_pp=value["gain_macro_pp"], gain_micro_pp=value["gain_micro_pp"],
            gain_ci95_sessions=value["gain_ci95_sessions"], interventions=value["interventions"], harm=value["harm"],
            quality_degraded=value["quality_degraded"], worst_gain_pp=float(min(0, q.raw_gain_pp.min())),
            candidate_DAC_mismatches=int((~c.matches_raw_DAC).sum()),
            false_passes=int((c.passed & (c.stored_raw_DAC < 1)).sum()),
            false_rejections=int((~c.passed & (c.stored_raw_DAC == 1)).sum()),
            decision_reasons={str(k): int(v) for k, v in q.reason.value_counts().items()},
            component_degradations={m: int((q["raw_delta_" + m] < -1e-8).sum()) for m in METRICS}))
    for name, frame in (("per_scene.csv", pd.DataFrame(rows)), ("candidate_geometry.csv", checks)):
        temp = out / (name + ".tmp")
        frame.to_csv(temp, index=False)
        temp.replace(out / name)
    tokens = {d["token"] for d in state["decisions"]}
    rollout_consistent = all(r["candidate_DAC_mismatches"] == 0 for r in result if r["method"] == "rollout")
    report = dict(status="complete" if tokens == set(identity["tokens"]) else "pilot_complete",
        selected_scenes=len(examples), evaluated_scenes=len(tokens), sessions=len({lookup[t]["metadata"]["session_id"] for t in tokens}),
        privileged=True, camera_only_comparable=False, primary="baseline", rows=result,
        rollout_DAC_consistency_passed=rollout_consistent,
        interpretation="privileged_positive_control_only" if rollout_consistent else "stop_and_inspect_rollout_DAC_mismatch",
        geometry_rollouts=len(tokens) * 16, new_full_PDM_scorer_calls=0, new_training_runs=0,
        additional_download_GB=0, fresh_holdout_processed=0, identity_sha256=sha(out / "identity.json"),
        artifacts={n: sha(out / n) for n in ("packet_index.json", "decisions.json", "per_scene.csv", "candidate_geometry.csv")}, note=NOTE)
    write(out / "report.json", report)
    return report


def preview(out, identity):
    index = checked_packets(out, identity)
    for token in identity["pilot_tokens"]:
        if token not in index:
            continue
        packet = arrays(out / "packets" / (token + ".npz"))
        b = int(packet["baseline_position"])
        origin = packet["raw_poses"][b, 0, :2]
        radius = max(20., max(float(np.abs(packet[k][..., :2] - origin).max()) for k in ("raw_poses", "rollout_poses")) + 6.)
        polygons = geometry.decode_polygons(packet["polygon_bytes"], packet["polygon_offsets"])
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="900" viewBox="{-radius} {-radius} {2*radius} {2*radius}">',
                 f'<rect x="{-radius}" y="{-radius}" width="{2*radius}" height="{2*radius}" fill="white"/>']
        for polygon in polygons:
            for p in ([polygon] if polygon.geom_type == "Polygon" else polygon.geoms):
                paths = []
                for ring in [p.exterior, *p.interiors]:
                    xy = np.array(ring.coords) - origin
                    paths.append("M " + " L ".join(f"{x:.4f},{-y:.4f}" for x, y in xy) + " Z")
                parts.append('<path d="' + " ".join(paths) + '" fill="#eef1f3" fill-rule="evenodd" stroke="#b0b8c0" stroke-width="0.08"/>')
        for mode, color in (("raw", "#2860c0"), ("rollout", "#008050")):
            result = geometry.decide(packet, mode)
            for j in dict.fromkeys((b, result["chosen"])):
                xy = packet[mode + "_poses"][j, :, :2] - origin
                points = " ".join(f"{x:.4f},{-y:.4f}" for x, y in xy)
                parts.append(f'<polyline points="{points}" fill="none" stroke="{color if j != b else "#d03030"}" stroke-width="0.3"/>')
                footprints = geometry.corners(packet[mode + "_poses"][j], packet["vehicle_dimensions"]) - origin
                for step in np.linspace(0, len(footprints) - 1, 5, dtype=int):
                    points = " ".join(f"{x:.4f},{-y:.4f}" for x, y in footprints[step])
                    parts.append(f'<polygon points="{points}" fill="none" stroke="{color if j != b else "#d03030"}" stroke-opacity="0.5" stroke-width="0.12"/>')
        parts.append(f'<text x="{-radius+2}" y="{-radius+4}" font-size="2">{html.escape(token)} privileged static map</text></svg>')
        path = out / "preview" / (token + ".svg")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(parts), encoding="utf-8")
    return dict(previews=str(out / "preview"), note="Red=baseline; blue=raw choice; green=rollout choice. World XY relative to current ego; not camera rendering.")


def execute(a):
    bundle, out = a.bundle.resolve(), a.out.resolve()
    if out == bundle or out in bundle.parents or bundle in out.parents:
        raise ValueError("Separate geometry output required")
    examples, vocab = bundles.load_bundle(bundle)
    selected, sampling, bindings = resolve_inputs(bundle, examples)
    identity = freeze_identity(out, bundle, examples, selected, sampling, bindings)
    if a.stage == "plan":
        result = dict(status="planned", scenes=len(examples), pilot_scenes=len(identity["pilot_tokens"]),
            sampling=list(sampling), modes=list(MODES), privileged=True, camera_only_comparable=False,
            scenes_with_multiple_cache_options=sum(v["alternatives"] > 1 for v in selected.values()),
            note=NOTE)
        write(out / "plan.json", result)
    elif a.stage in ("prepare", "prepare-pilot"):
        result = prepare(out, identity, examples, vocab, 8 if a.stage == "prepare-pilot" else None)
    elif a.stage == "infer":
        snapshot = infer(out, identity)
        result = dict(frozen_decisions=len(snapshot["decisions"]), privileged=True)
    elif a.stage == "evaluate":
        checked_packets(out, identity)
        result = evaluate(out, identity, examples)
    elif a.stage == "preview":
        result = preview(out, identity)
    elif a.stage == "report":
        checked_packets(out, identity)
        result = read(out / "report.json")
        if result["identity_sha256"] != sha(out / "identity.json"):
            raise RuntimeError("Geometry report identity changed")
        for name, digest in result["artifacts"].items():
            if sha(out / name) != digest:
                raise RuntimeError("Geometry report artifact changed: " + name)
    else:
        prepare(out, identity, examples, vocab)
        infer(out, identity)
        result = evaluate(out, identity, examples)
        preview(out, identity)
    print(json.dumps(result, indent=2), flush=True)
    if a.stage in ("evaluate", "all") and result.get("rollout_DAC_consistency_passed") is False:
        raise SystemExit("Rollout DAC mismatch: report and packets preserved. Inspect the geometry before further experiments.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory"
    p.add_argument("stage", choices=("plan", "prepare-pilot", "prepare", "infer", "evaluate", "preview", "report", "all"))
    p.add_argument("--bundle", type=Path, default=base / "selector_training_bundle_v3_30gb")
    p.add_argument("--out", type=Path, default=base / "experiments/corridor_geometry_v1")
    a = p.parse_args()
    bundle, out = a.bundle.resolve(), a.out.resolve()
    if out == bundle or out in bundle.parents or bundle in out.parents:
        raise ValueError("Separate geometry output required")
    with locked(out):
        execute(a)


if __name__ == "__main__":
    main()
