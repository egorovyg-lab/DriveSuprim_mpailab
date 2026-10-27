"""Inventory, cache, export, train, and evaluate a small residual-memory pilot.

Query outcomes are read only by the evaluator, after predictions are frozen.
Calibration/test are reserved; `all` evaluates development only.
"""
import hashlib
import json
import lzma
import os
import pickle
import platform
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.drivemem import (
    METRICS, SAFETY, ROLES, session_id, split_sessions, spaced_records, token_seed,
    probabilities, training_candidate_indices, nearest_donors, memory_correction,
    corrected_choice, fit_ridge, ridge_correction,
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    repo = Path(__file__).resolve().parents[3]
    files = ["navsim/agents/drivemem.py", "navsim/planning/script/run_drivemem_memory.py",
             "navsim/agents/drivesuprim/drivesuprim_model.py", "navsim/agents/drivesuprim/drivesuprim_config.py",
             "navsim/agents/drivesuprim/drivesuprim_features.py", "navsim/agents/drivesuprim/ssl_meta_arch.py",
             "navsim/planning/script/run_drivemem_oracle_pilot.py",
             "navsim/common/dataloader.py", "navsim/common/dataclasses.py",
             "navsim/planning/training/dataset_ssl.py",
             "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml",
             "navsim/evaluate/pdm_score.py", "navsim/planning/metric_caching/metric_cache_processor.py",
             "navsim/planning/scenario_builder/navsim_scenario.py",
             "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
             "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py",
             "navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py",
             "navsim/traffic_agents_policies/log_replay_traffic_agents.py"]
    return {name: sha(repo / name) for name in files}


def identity(cfg):
    data = OmegaConf.to_container(cfg, resolve=True)
    settings = dict(data["drivemem"])
    settings.pop("stage")
    return {"schema_version": 1, "sources": source_hashes(), "settings": settings,
            "checkpoint_sha256": sha(cfg.agent.checkpoint_path),
            "agent": data["agent"], "scorer": data["scorer"], "simulator": data["simulator"],
            "train_test_split": data["train_test_split"],
            "proposal_sampling": data["proposal_sampling"], "traffic_agents": data["traffic_agents"],
            "traffic_agents_policy": data["traffic_agents_policy"],
            "navsim_log_path": data["navsim_log_path"], "original_sensor_path": data["original_sensor_path"],
            "map_root": os.environ.get("NUPLAN_MAPS_ROOT"), "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__}


def verify_identity(cfg, root):
    current = identity(cfg)
    if read_json(root / "identity.json") != current:
        raise RuntimeError("Code, checkpoint or configuration changed. Use a new DRIVEMEM_MEMORY_ROOT.")
    if (root / "splits.json").exists():
        expected = read_json(root / "split_identity.json")
        if expected != {"splits": sha(root / "splits.json"), "inventory": sha(root / "inventory.csv")}:
            raise RuntimeError("Frozen splits or inventory changed")


def inventory(cfg, root):
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig
    logs_path = Path(cfg.navsim_log_path)
    sensor_path = Path(cfg.original_sensor_path)
    if not logs_path.is_dir():
        raise FileNotFoundError(
            f"navtrain log directory not found: {logs_path}\n"
            f"OPENSCENE_DATA_ROOT={os.environ.get('OPENSCENE_DATA_ROOT')}\n"
            "Run run_drivemem_memory.sh paths. Set DRIVEMEM_LOGS_PATH if logs are stored elsewhere.")
    if not sensor_path.is_dir():
        print(f"Images are not present at {sensor_path}. Metadata inventory can proceed; "
              "GPU export will require the trainval images.", flush=True)
    if (root / "identity.json").exists():
        if read_json(root / "identity.json") != identity(cfg):
            if {p.name for p in root.iterdir()} == {"identity.json"}:
                print("Refreshing identity from an earlier failed preflight; no committed results exist.", flush=True)
            else:
                raise RuntimeError("Inventory identity changed; choose a new root")
        if (root / "split_identity.json").exists() and (root / "inventory_report.json").exists():
            verify_identity(cfg, root)
            print("Inventory already complete; using frozen splits.", flush=True)
            return
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "identity.json", identity(cfg))
    base_filter = instantiate(cfg.train_test_split.scene_filter)
    if base_filter.max_scenes is not None or base_filter.include_synthetic_scenes:
        raise ValueError("Inventory requires full ORIGINAL navtrain filter")
    repo = Path(__file__).resolve().parents[3]
    test_filter = OmegaConf.load(repo / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml")
    excluded = {session_id(str(name)) for name in test_filter.log_names}
    allowed = set(base_filter.log_names) if base_filter.log_names else None
    paths = sorted(p for p in Path(cfg.navsim_log_path).glob("*.pkl") if allowed is None or p.stem in allowed)
    records = []
    for number, path in enumerate(paths):
        sid = session_id(path.stem)
        if sid in excluded:
            continue
        scene_filter = instantiate(cfg.train_test_split.scene_filter)
        scene_filter.log_names = [path.stem]
        loader = SceneLoader(data_path=path.parent, original_sensor_path=None,
                             scene_filter=scene_filter, sensor_config=SensorConfig.build_no_sensors())
        for token, frames in loader.scene_frames_dicts.items():
            current = frames[scene_filter.num_history_frames - 1]
            records.append(dict(token=token, session_id=sid, log_name=path.stem,
                                timestamp_us=int(current["timestamp"]), map_name=str(current["map_location"])))
        if (number + 1) % 25 == 0:
            print(f"INVENTORY logs={number + 1}/{len(paths)} scenes={len(records)}", flush=True)
    if not records:
        raise RuntimeError("No eligible navtrain scenes")
    frame = pd.DataFrame(records).sort_values(["session_id", "timestamp_us", "log_name", "token"])
    duplicate_times = int(frame.duplicated(["session_id", "timestamp_us"]).sum())
    frame = frame.drop_duplicates(["session_id", "timestamp_us"])
    if frame.token.duplicated().any():
        raise RuntimeError("Duplicate tokens across sessions")
    frame.to_csv(root / "inventory.csv", index=False)
    groups = {sid: group.to_dict("records") for sid, group in frame.groupby("session_id", sort=True)}
    eligible = {}
    max_scenes = max(int(cfg.drivemem.scenes[r]) for r in ROLES)
    for sid, rows in groups.items():
        try:
            spaced_records(rows, max_scenes, float(cfg.drivemem.gap_s), token_seed(sid, int(cfg.drivemem.seed)))
            eligible[sid] = rows
        except ValueError:
            pass
    eligible_records = [r for rows in eligible.values() for r in rows]
    allocation = split_sessions(eligible_records, dict(cfg.drivemem.sessions), int(cfg.drivemem.seed))
    selected = []
    for role in ROLES:
        for sid in allocation[role]:
            rows = spaced_records(eligible[sid], int(cfg.drivemem.scenes[role]), float(cfg.drivemem.gap_s),
                                  token_seed(sid, int(cfg.drivemem.seed)))
            selected.extend([dict(row, role=role) for row in rows])
    write_json(root / "splits.json", {"allocation": allocation, "records": selected,
                                     "log_sha256": {log: sha(Path(cfg.navsim_log_path) / (log + ".pkl"))
                                                    for log in sorted({r["log_name"] for r in selected})},
                                     "navtest_sessions_excluded": sorted(excluded),
                                     "purpose": "module development; calibration independence from base pretraining unverified"})
    write_json(root / "split_identity.json", {"splits": sha(root / "splits.json"),
                                              "inventory": sha(root / "inventory.csv")})
    report = dict(available_scenes=len(frame), source_sessions=len(groups), eligible_sessions=len(eligible),
                  duplicate_session_times_removed=duplicate_times, selected_scenes=len(selected),
                  selected_sessions=sum(len(x) for x in allocation.values()),
                  roles={r: {"sessions": len(allocation[r]), "scenes": sum(x["role"] == r for x in selected)} for r in ROLES},
                  navsim_log_path=str(cfg.navsim_log_path), original_sensor_path=str(cfg.original_sensor_path),
                  existing_train_cache_metadata=str(Path(os.environ["NAVSIM_EXP_ROOT"]) / "metric_cache/train/ori/metadata"),
                  python=platform.python_version())
    write_json(root / "inventory_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def records(root):
    return read_json(root / "splits.json")["records"]


def build_cache(cfg, root):
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    if "NUPLAN_MAPS_ROOT" not in os.environ:
        raise RuntimeError("NUPLAN_MAPS_ROOT must point to the mounted maps")
    index_path = root / "cache_index.json"
    index = read_json(index_path) if index_path.exists() else {}
    processor = MetricCacheProcessor(str(root / "metric_cache"), False, instantiate(cfg.proposal_sampling))
    selected = records(root)
    # Load one log at a time, retaining only the preselected scenes.
    for log in sorted({r["log_name"] for r in selected}):
        if sha(Path(cfg.navsim_log_path) / (log + ".pkl")) != read_json(root / "splits.json")["log_sha256"][log]:
            raise RuntimeError(f"Input log changed: {log}")
        wanted = [r for r in selected if r["log_name"] == log]
        todo = []
        for row in wanted:
            token = row["token"]
            if token in index:
                if sha(index[token]["path"]) != index[token]["sha256"]:
                    raise RuntimeError(f"Cache changed: {token}")
            else:
                todo.append(row)
        if not todo:
            continue
        scene_filter = instantiate(cfg.train_test_split.scene_filter)
        scene_filter.log_names = [log]
        scene_filter.tokens = [r["token"] for r in todo]
        loader = SceneLoader(Path(cfg.navsim_log_path), None, scene_filter, sensor_config=SensorConfig.build_no_sensors())
        if set(loader.tokens) != {r["token"] for r in todo}:
            raise RuntimeError(f"Selected tokens missing from log: {log}")
        for row in todo:
            token = row["token"]
            scene = loader.get_scene_from_token(token)
            scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")
            # New dedicated cache, not a mixture with unknown previous training caches.
            cache = processor.compute_metric_cache(scenario)
            if cache.log_name != log or cache.timepoint.time_us != int(row["timestamp_us"]):
                raise RuntimeError(f"Generated cache identity mismatch: {token}")
            # A file not present in cache_index is incomplete/uncommitted and can
            # be regenerated. Commit the lzma pickle before committing its hash.
            cache.file_path.parent.mkdir(parents=True, exist_ok=True)
            temp = cache.file_path.with_name(cache.file_path.name + ".tmp")
            with lzma.open(temp, "wb") as stream:
                pickle.dump(cache, stream)
            os.replace(temp, cache.file_path)
            index[token] = {"path": str(cache.file_path), "sha256": sha(cache.file_path)}
            write_json(index_path, index)
            print(f"CACHE {len(index)}/{len(selected)} {token}", flush=True)


def load_cache(root, token):
    info = read_json(root / "cache_index.json")[token]
    if sha(info["path"]) != info["sha256"]:
        raise RuntimeError(f"Cache checksum mismatch: {token}")
    with lzma.open(info["path"], "rb") as stream:
        return pickle.load(stream)


def export(cfg, root):
    import torch
    from torch.utils.data import DataLoader
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.training.dataset_ssl import DatasetSSL
    from navsim.planning.script.run_drivemem_oracle_pilot import on_device
    if not Path(cfg.original_sensor_path).is_dir():
        raise FileNotFoundError(f"Trainval image directory not found: {cfg.original_sensor_path}. "
                                "Run run_drivemem_memory.sh paths; set DRIVEMEM_SENSOR_PATH if necessary.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()
    agent.to("cuda")
    if cfg.agent.config.inference.model != "teacher" or cfg.agent.config.refinement.num_refinement_stage != 1:
        raise ValueError("Initial memory pilot expects teacher and one refinement stage")
    out = root / "features"
    out.mkdir(exist_ok=True)
    index_path = root / "export_index.json"
    index = read_json(index_path) if index_path.exists() else {}
    capture = {}

    def hook(module, inputs, output):
        capture["embedding"] = output.mean(dim=(2, 3)).detach().cpu().numpy()[0].copy()

    handle = agent.model.teacher["model"].downscale_layer.register_forward_hook(hook)
    try:
        selected = records(root)
        for log in sorted({r["log_name"] for r in selected}):
            if sha(Path(cfg.navsim_log_path) / (log + ".pkl")) != read_json(root / "splits.json")["log_sha256"][log]:
                raise RuntimeError(f"Input log changed: {log}")
            wanted = [r for r in selected if r["log_name"] == log]
            todo = []
            for row in wanted:
                if row["token"] in index:
                    if sha(out / f"{row['token']}.npz") != index[row["token"]]:
                        raise RuntimeError(f"Feature checksum mismatch: {row['token']}")
                else:
                    todo.append(row)
            if not todo:
                continue
            scene_filter = instantiate(cfg.train_test_split.scene_filter)
            scene_filter.log_names = [log]
            scene_filter.tokens = [r["token"] for r in todo]
            loader = SceneLoader(Path(cfg.navsim_log_path), Path(cfg.original_sensor_path), scene_filter,
                                 sensor_config=agent.get_sensor_config())
            if set(loader.tokens) != {r["token"] for r in todo}:
                raise RuntimeError("Feature loader token mismatch")
            dataset = DatasetSSL(loader, agent.get_feature_builders(), agent.get_target_builders(),
                                 cfg=cfg.agent.config, cache_path=None, append_token_to_batch=True)
            for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
                batch = on_device(batch, "cuda")
                token = str(batch[2][0])
                with torch.no_grad():
                    agent._config.inference.export_candidates = False
                    first, _ = agent.forward(batch)
                    baseline = first["final_traj"].cpu().numpy()[0].copy()
                    del first
                    agent._config.inference.export_candidates = True
                    pred, _ = agent.forward(batch)
                stage = pred["refinement"][-1]
                ids = stage["indices_absolute"].cpu().numpy()[0]
                chosen = int(stage["final_selected_local_indices"].cpu().numpy()[0])
                vocab = pred["trajectory_vocab"].detach().cpu().numpy()
                scores = stage["final_selection_scores"].cpu().numpy()[0]
                if (len(ids) != 256 or len(set(ids.tolist())) != len(ids) or chosen != int(np.argmax(scores))
                        or not np.array_equal(vocab[ids], stage["trajs"].cpu().numpy()[0])):
                    raise RuntimeError(f"Candidate pool/selection mismatch: {token}")
                if not np.array_equal(baseline, vocab[ids[chosen]]) or not np.array_equal(baseline, pred["final_traj"].cpu().numpy()[0]):
                    raise RuntimeError(f"Capture changes baseline: {token}")
                vp = root / "vocab.npz"
                if not vp.exists():
                    save_npz(vp, vocab=vocab)
                    write_json(root / "vocab_identity.json", {"sha256": sha(vp)})
                else:
                    if sha(vp) != read_json(root / "vocab_identity.json")["sha256"]:
                        raise RuntimeError("Vocabulary checksum mismatch")
                    with np.load(vp, allow_pickle=False) as stored:
                        if not np.array_equal(vocab, stored["vocab"]):
                            raise RuntimeError("Active vocabulary changed")
                arrays = {"candidate_ids": ids, "selected_local": np.array(chosen),
                          "selection_scores": scores,
                          "scene_embedding": capture["embedding"],
                          "ego_status": torch.cat(batch[0]["status_feature"], dim=1).cpu().numpy()[0]}
                arrays.update({"head_" + k: v.cpu().numpy()[0] for k, v in stage["layer_results"][-1].items()})
                if any(not np.isfinite(v).all() for v in arrays.values()):
                    raise RuntimeError(f"Nonfinite export: {token}")
                save_npz(out / f"{token}.npz", **arrays)
                index[token] = sha(out / f"{token}.npz")
                write_json(index_path, index)
                print(f"EXPORT {len(index)}/{len(selected)} {token}", flush=True)
                del pred, stage, batch
    finally:
        handle.remove()


def sample(root, row):
    path = root / "features" / f"{row['token']}.npz"
    if sha(path) != read_json(root / "export_index.json")[row["token"]]:
        raise RuntimeError(f"Export checksum mismatch: {row['token']}")
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k].copy() for k in data.files}
    result.update(token=row["token"], session_id=row["session_id"], role=row["role"])
    return result


def evaluators(cfg, root):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
    simulator = instantiate(cfg.simulator)
    raw_scorer, filtered_scorer = instantiate(cfg.scorer), instantiate(cfg.scorer)
    raw_scorer._config.human_penalty_filter = False
    filtered_scorer._config.human_penalty_filter = True
    if cfg.traffic_agents != "non_reactive":
        raise ValueError("Pilot requires non-reactive replay")
    policy = instantiate(cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling)
    sampling = instantiate(cfg.agent.config.trajectory_sampling)
    with np.load(root / "vocab.npz", allow_pickle=False) as data:
        vocab = data["vocab"]

    def evaluate(token, candidate_id):
        result = {}
        states = None
        for name, scorer in (("raw", raw_scorer), ("filtered", filtered_scorer)):
            row, trajectory_states = pdm_score(load_cache(root, token), Trajectory(vocab[candidate_id], sampling),
                                              simulator.proposal_sampling, simulator, scorer, policy)
            row = row.iloc[0]
            values = {k: float(row[k]) for k in (*METRICS, "pdm_score", "multiplicative_metrics_prod")}
            values["weighted_metrics"] = np.array(row["weighted_metrics"], dtype=float).tolist()
            values["weighted_metrics_array"] = np.array(row["weighted_metrics_array"], dtype=float).tolist()
            if not np.isfinite([values[k] for k in (*METRICS, "pdm_score")]).all():
                raise RuntimeError("Nonfinite scorer output")
            mask = np.arange(len(values["weighted_metrics"])) != int(WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT)
            weighted, weights = np.array(values["weighted_metrics"]), np.array(values["weighted_metrics_array"])
            values["aggregate_without_two_frame"] = float(values["multiplicative_metrics_prod"]
                                                          * (weighted[mask] * weights[mask]).sum() / weights[mask].sum())
            if name == "raw":
                values["raw_progress_m"] = float(scorer._progress_raw[1])
                if not np.isclose(values["aggregate_without_two_frame"], values["pdm_score"], atol=1e-8, rtol=1e-7):
                    raise RuntimeError("Raw aggregation mismatch")
            else:
                values["raw_progress_m"] = result["raw"]["raw_progress_m"]
            # Values returned above are copied before another scorer call.
            result[name] = values
            if states is None:
                states = trajectory_states.copy()
            elif not np.allclose(states, trajectory_states, atol=1e-8, rtol=1e-7):
                raise RuntimeError("Raw/filter simulated trajectories differ")
        return result, states
    return evaluate, vocab


def label_indices(cfg, s):
    if s["role"] == "memory" and cfg.drivemem.memory_mode == "executed_only":
        return np.array([int(s["selected_local"])], dtype=np.int64)
    return training_candidate_indices(s, int(cfg.drivemem.label_top), int(cfg.drivemem.label_random), int(cfg.drivemem.seed))


def label(cfg, root):
    evaluate, _ = evaluators(cfg, root)
    out = root / "labels"
    out.mkdir(exist_ok=True)
    source = [r for r in records(root) if r["role"] in ("memory", "fit")]
    for n, row in enumerate(source):
        s = sample(root, row)
        indices = label_indices(cfg, s)
        done = out / f"{row['token']}.json"
        if done.exists():
            saved = read_json(done)
            if saved["local_indices"] != indices.tolist() or saved["feature_sha256"] != sha(root / "features" / f"{row['token']}.npz"):
                raise RuntimeError("Training label provenance mismatch")
            print(f"LABEL RESUME {n + 1}/{len(source)}", flush=True)
            continue
        values = []
        for index in indices:
            value, _ = evaluate(row["token"], int(s["candidate_ids"][index]))
            values.append(value)
        write_json(done, {"token": row["token"], "role": row["role"], "local_indices": indices.tolist(),
                          "values": values, "feature_sha256": sha(root / "features" / f"{row['token']}.npz")})
        print(f"LABEL {n + 1}/{len(source)} candidates={len(indices)} {row['token']}", flush=True)


def training_samples(cfg, root):
    result = []
    for row in records(root):
        if row["role"] not in ("memory", "fit"):
            continue
        s = sample(root, row)
        labels = read_json(root / "labels" / f"{row['token']}.json")
        indices = np.array(labels["local_indices"], dtype=int)
        if (labels["token"] != row["token"] or labels["role"] != row["role"]
                or labels["feature_sha256"] != sha(root / "features" / f"{row['token']}.npz")
                or not np.array_equal(indices, label_indices(cfg, s))):
            raise RuntimeError("Training labels are inconsistent with frozen features/roles")
        values = np.array([[v["raw"][k] for k in METRICS] for v in labels["values"]])
        if values.shape != (len(indices), len(METRICS)) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise RuntimeError("Invalid training outcomes")
        if row["role"] == "memory" and cfg.drivemem.memory_mode == "executed_only":
            mask = indices == int(s["selected_local"])
            indices, values = indices[mask], values[mask]
        s.update(label_local_indices=indices, label_values=values)
        result.append(s)
    return result


def fit(cfg, root):
    label_hashes = {p.name: sha(p) for p in sorted((root / "labels").glob("*.json"))}
    if (root / "fit.json").exists():
        previous = read_json(root / "fit.json")
        if previous["labels"] != label_hashes or previous["ridge_sha256"] != sha(root / "ridge.npz"):
            raise RuntimeError("Previously fitted model or labels changed")
        print("FIT RESUME: frozen model already exists", flush=True)
        return
    samples = training_samples(cfg, root)
    bank = [s for s in samples if s["role"] == "memory"]
    train = [s for s in samples if s["role"] == "fit"]
    with np.load(root / "vocab.npz", allow_pickle=False) as data:
        vocab = data["vocab"]
    # The parametric control sees all labels available to memory plus fit labels.
    model = fit_ridge(samples, vocab, int(cfg.drivemem.seed), float(cfg.drivemem.ridge_lambda))
    distances = [nearest_donors(s, bank, int(cfg.drivemem.neighbors), int(cfg.drivemem.seed))[1] for s in train]
    finite = [d for d in distances if np.isfinite(d)]
    if not finite:
        raise RuntimeError("No compatible donors for fit queries")
    threshold = float(np.quantile(finite, float(cfg.drivemem.threshold_quantile)))
    output = root / "ridge.npz"
    save_npz(output, **model)
    write_json(root / "fit.json", {"status": "complete", "ridge_sha256": sha(output),
                                   "labels": label_hashes, "similarity_threshold": threshold,
                                   "threshold_source": "fit nearest-neighbor distances only; no query outcomes",
                                   "memory_mode": str(cfg.drivemem.memory_mode),
                                   "training_scenes": len(samples), "bank_scenes": len(bank)})
    print(f"FIT complete bank={len(bank)} train={len(train)} threshold={threshold:.6f}", flush=True)


def choices(cfg, s, bank, vocab, ridge, threshold):
    ids, distance = nearest_donors(s, bank, int(cfg.drivemem.neighbors), int(cfg.drivemem.seed))
    random_ids, _ = nearest_donors(s, bank, int(cfg.drivemem.neighbors), int(cfg.drivemem.seed), random=True)
    base = int(s["selected_local"])
    sim = corrected_choice(s, memory_correction(s, bank, ids, vocab), float(cfg.drivemem.strength))
    random_choice = corrected_choice(s, memory_correction(s, bank, random_ids, vocab), float(cfg.drivemem.strength))
    no_memory = corrected_choice(s, ridge_correction(s, vocab, ridge), float(cfg.drivemem.strength))
    return {"baseline": base, "no_memory_ridge": no_memory, "random_memory": random_choice,
            "similarity_memory": sim, "similarity_threshold": sim if distance <= threshold else base}, {
                "nearest_distance": distance if np.isfinite(distance) else None,
                "donor_tokens": [str(bank[i]["token"]) for i in ids]}


def run_evaluation(cfg, root, role):
    fit_info = read_json(root / "fit.json")
    if sha(root / "ridge.npz") != fit_info["ridge_sha256"]:
        raise RuntimeError("Ridge model changed")
    for name, digest in fit_info["labels"].items():
        if sha(root / "labels" / name) != digest:
            raise RuntimeError("Training labels changed")
    # Only M/F labels enter model construction. C/T/V outcomes are never loaded here.
    bank = [s for s in training_samples(cfg, root) if s["role"] == "memory"]
    with np.load(root / "ridge.npz", allow_pickle=False) as data:
        ridge = {k: data[k].copy() for k in data.files}
    evaluate, vocab = evaluators(cfg, root)
    out = root / ("evaluation_" + role)
    out.mkdir(exist_ok=True)
    final_rows = []
    wanted = [r for r in records(root) if r["role"] == role]
    for number, row in enumerate(wanted):
        s = sample(root, row)
        # Freeze every method's decision before requesting any query outcome.
        decisions, retrieval = choices(cfg, s, bank, vocab, ridge, fit_info["similarity_threshold"])
        done = out / f"{row['token']}.json"
        if done.exists():
            saved = read_json(done)
            if saved["decisions"] != decisions or saved["ridge_sha256"] != sha(root / "ridge.npz"):
                raise RuntimeError("Evaluation decisions/model changed")
            final_rows.extend(saved["rows"])
            print(f"EVALUATE {role} RESUME {number + 1}/{len(wanted)}", flush=True)
            continue
        outcomes = {}
        for index in sorted(set(decisions.values())):
            outcomes[index], _ = evaluate(row["token"], int(s["candidate_ids"][index]))
        baseline = outcomes[decisions["baseline"]]
        repeated, _ = evaluate(row["token"], int(s["candidate_ids"][decisions["baseline"]]))
        for version in ("raw", "filtered"):
            for key in (*METRICS, "pdm_score", "aggregate_without_two_frame"):
                if not np.isclose(baseline[version][key], repeated[version][key], atol=1e-8, rtol=1e-7):
                    raise RuntimeError("Baseline repeat mismatch")
        rows = []
        for method, index in decisions.items():
            result = outcomes[index]
            entry = dict(token=row["token"], session_id=row["session_id"], role=role, method=method,
                         candidate_local=index, candidate_id=int(s["candidate_ids"][index]),
                         intervention=index != decisions["baseline"], **retrieval)
            for version in ("raw", "filtered"):
                base, value = baseline[version], result[version]
                entry[version + "_score"] = value["pdm_score"]
                entry[version + "_gain_pp"] = 100 * (value["pdm_score"] - base["pdm_score"])
                entry[version + "_harm"] = any(value[METRICS[k]] + 1e-8 < base[METRICS[k]] for k in SAFETY)
                entry[version + "_quality_degraded"] = value["pdm_score"] + 1e-8 < base["pdm_score"]
                entry[version + "_absolute_violation"] = any(value[METRICS[k]] < 1 - 1e-8 for k in SAFETY)
                for key in METRICS:
                    entry[version + "_" + key] = value[key]
            rows.append(entry)
        write_json(done, {"decisions": decisions, "retrieval": retrieval, "outcomes": outcomes,
                          "rows": rows, "ridge_sha256": sha(root / "ridge.npz")})
        final_rows.extend(rows)
        print(f"EVALUATE {role} {number + 1}/{len(wanted)} unique_actions={len(outcomes)}", flush=True)
    frame = pd.DataFrame(final_rows)
    frame.to_csv(out / "per_scene.csv", index=False)
    report = {"status": "complete", "role": role, "scenes": len(wanted),
              "sessions": len({r["session_id"] for r in wanted}), "methods": {},
              "memory_mode": str(cfg.drivemem.memory_mode),
              "note": "Local simulated metric; empirical harm only, no calibrated guarantee; all donor outcomes come from simulation."}
    for method, group in frame.groupby("method"):
        active = group[group.intervention]
        # Paired cluster bootstrap of gains, sampling sessions as blocks.
        by_session = group.groupby("session_id").raw_gain_pp.agg(["sum", "count"])
        rng = np.random.default_rng(int(cfg.drivemem.seed))
        pick = rng.integers(0, len(by_session), size=(2000, len(by_session)))
        boot = by_session["sum"].to_numpy()[pick].sum(axis=1) / by_session["count"].to_numpy()[pick].sum(axis=1)
        report["methods"][method] = {"raw_score": float(group.raw_score.mean()),
                                       "raw_gain_pp": float(group.raw_gain_pp.mean()),
                                       "filtered_score": float(group.filtered_score.mean()),
                                       "filtered_gain_pp": float(group.filtered_gain_pp.mean()),
                                       "gain_ci95_sessions": [float(x) for x in np.quantile(boot, [.025, .975])],
                                       "coverage": float(group.intervention.mean()),
                                       "interventions": len(active), "harm_events": int(active.raw_harm.sum()),
                                       "quality_degradation_events": int(active.raw_quality_degraded.sum()),
                                       "improved_scenes": int((group.raw_gain_pp > 1e-6).sum()),
                                       "harm_all_scenes": float(group.raw_harm.mean()),
                                       "harm_among_interventions": float(active.raw_harm.mean()) if len(active) else None,
                                       "filtered_harm_among_interventions": float(active.filtered_harm.mean()) if len(active) else None,
                                       "quality_degradation_among_interventions": float(active.raw_quality_degraded.mean()) if len(active) else None,
                                       "absolute_violation": float(group.raw_absolute_violation.mean())}
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


@hydra.main(config_path="config/pdm_scoring", config_name="default_drivemem_memory", version_base=None)
def main(cfg):
    root = Path(cfg.drivemem.root)
    stage = str(cfg.drivemem.stage)
    if cfg.drivemem.memory_mode not in ("enriched", "executed_only"):
        raise ValueError("memory_mode must be enriched or executed_only")
    if stage in ("inventory", "all"):
        inventory(cfg, root)
    else:
        verify_identity(cfg, root)
    if stage == "all":
        for fn in (build_cache, export, label, fit):
            fn(cfg, root)
        run_evaluation(cfg, root, "development")
    elif stage == "cache":
        build_cache(cfg, root)
    elif stage == "export":
        export(cfg, root)
    elif stage == "label":
        label(cfg, root)
    elif stage == "fit":
        fit(cfg, root)
    elif stage in ("evaluate", "test"):
        run_evaluation(cfg, root, "test" if stage == "test" else "development")
    elif stage == "report":
        print(json.dumps(read_json(root / "inventory_report.json"), indent=2))
        print(json.dumps(read_json(root / "evaluation_development/report.json"), indent=2))
    elif stage != "inventory":
        raise ValueError("Unknown stage")


if __name__ == "__main__":
    main()
