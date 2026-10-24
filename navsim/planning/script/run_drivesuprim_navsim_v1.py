"""Evaluate DriveSuprim with the paper's NAVSIM v1 inference coefficients.

Scoring/cache source: autonomousvision/navsim v1.1, commit
3e8291bfa89ff247231e0227778840cd0a036896. Uses the local dataset,
feature builders and simulator; no reactive or two-frame v2 scoring.
"""
import csv
import gc
import hashlib
import json
import logging
import lzma
import os
import pickle
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from navsim.common.dataclasses import Scene, SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.evaluate.navsim_v1.metric_cache_processor import MetricCacheProcessor
from navsim.evaluate.navsim_v1.pdm_enums import MultiMetricIndex, WeightedMetricIndex
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.training.dataset_ssl import DatasetSSL

LOG = logging.getLogger(__name__)
UPSTREAM = "autonomousvision/navsim@3e8291bfa89ff247231e0227778840cd0a036896"
CACHE_SCHEMA = "drivesuprim-navsim-v1.1-5s-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_hash(tokens):
    return hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest()


def cache_source_hashes():
    root = Path(__file__).resolve().parents[2] / "evaluate" / "navsim_v1"
    return {p.name: sha256(p) for p in sorted(root.glob("*.py"))}


def make_loader(cfg, scene_filter=None, sensor_config=None):
    return SceneLoader(
        original_sensor_path=Path(cfg.original_sensor_path) if sensor_config is not None else None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter if scene_filter is not None else instantiate(cfg.train_test_split.scene_filter),
        sensor_config=sensor_config if sensor_config is not None else SensorConfig.build_no_sensors(),
    )


def jobs_for(loader, cfg, **extra):
    return [
        {"cfg": cfg, "log_name": log_name, "tokens": sorted(tokens), **extra}
        for log_name, tokens in sorted(loader.get_tokens_list_per_log().items())
    ]


def cache_worker(args):
    cfg = args[0]["cfg"]
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = [a["log_name"] for a in args]
    scene_filter.tokens = [token for a in args for token in a["tokens"]]
    loader = make_loader(cfg, scene_filter)
    processor = MetricCacheProcessor(str(cfg.metric_cache_path), force_feature_computation=False)
    records = []
    for index, (token, scene_dict) in enumerate(sorted(loader.scene_frames_dicts.items())):
        LOG.info("NAVSIM v1 cache %s / %s: %s", index + 1, len(loader.tokens), token)
        scene = Scene.from_scene_dict_list(
            scene_dict, None,
            num_history_frames=scene_filter.num_history_frames,
            num_future_frames=scene_filter.num_future_frames,
            sensor_config=SensorConfig.build_no_sensors(),
        )
        scenario = NavSimScenario(
            scene,
            map_root=os.environ["NUPLAN_MAPS_ROOT"],
            map_version=os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"),
        )
        entry = processor.compute_metric_cache(scenario)
        records.append({"token": token, "path": str(entry.file_name)})
        gc.collect()
    return records


def read_cache_index(cfg, expected_tokens):
    root = Path(cfg.metric_cache_path)
    manifest_path = root / "navsim_v1_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("NAVSIM v1 cache is not ready. Run the launcher in its default mode, or NAVSIM_V1_MODE=cache.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CACHE_SCHEMA:
        raise RuntimeError("Cache schema mismatch; select a new NAVSIM_V1_CACHE directory.")
    if manifest.get("source_hashes") != cache_source_hashes():
        raise RuntimeError("Cache implementation changed; select a new NAVSIM_V1_CACHE directory.")
    if manifest.get("tokens_sha256") != token_hash(expected_tokens):
        raise RuntimeError("Cache token set differs from the selected split; select a new NAVSIM_V1_CACHE directory.")
    index_path = root / "metadata" / "navsim_v1.csv"
    if sha256(index_path) != manifest.get("index_sha256"):
        raise RuntimeError("Cache metadata checksum mismatch.")
    paths = {}
    with index_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            token = row["token"]
            if token in paths:
                raise RuntimeError("Duplicate cache token: " + token)
            paths[token] = row["file_name"]
    missing = set(expected_tokens) - set(paths)
    if missing:
        raise RuntimeError(f"Missing metric cache for {len(missing)} scenes; evaluation aborted.")
    absent_files = [p for p in paths.values() if not Path(p).is_file()]
    if absent_files:
        raise RuntimeError(f"Missing {len(absent_files)} cache files, first: {absent_files[0]}")
    return paths


def build_cache(cfg, loader):
    root = Path(cfg.metric_cache_path)
    if (root / "navsim_v1_manifest.json").is_file():
        read_cache_index(cfg, loader.tokens)
        LOG.info("Complete NAVSIM v1 cache already exists: %s scenes.", len(loader.tokens))
        return
    # A v2 cache must not be silently reused as v1.
    if root.exists() and any(root.glob("metadata/*.csv")):
        raise RuntimeError("Unmarked cache metadata exists here. Select a fresh NAVSIM_V1_CACHE directory.")
    worker = build_worker(cfg)
    records = worker_map(worker, cache_worker, jobs_for(loader, cfg))
    paths = {r["token"]: r["path"] for r in records}
    if set(paths) != set(loader.tokens) or len(records) != len(paths):
        raise RuntimeError("Cache coverage or uniqueness check failed; no completion manifest written.")
    index_path = root / "metadata" / "navsim_v1.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["token", "file_name"])
        writer.writeheader()
        writer.writerows({"token": token, "file_name": paths[token]} for token in sorted(paths))
    manifest = {
        "schema": CACHE_SCHEMA, "upstream": UPSTREAM,
        "source_hashes": cache_source_hashes(), "tokens_sha256": token_hash(paths),
        "scene_count": len(paths), "index_sha256": sha256(index_path),
        "observation_horizon_s": 5.0,
    }
    (root / "navsim_v1_manifest.json").write_text(json.dumps(manifest, indent=2))
    LOG.info("NAVSIM v1 cache complete: %s scenes. Path: %s", len(paths), root)


class PredictionModule(pl.LightningModule):
    def __init__(self, agent, config):
        super().__init__()
        self.agent = agent
        self.config = config

    def predict_step(self, batch, batch_idx):
        self.agent.eval()
        predictions, _ = self.agent.forward(batch)
        use_refine = self.config.refinement.use_multi_stage and not self.config.inference.use_first_stage_traj_in_infer
        poses = predictions["final_traj" if use_refine else "trajectory"].detach().cpu().numpy()
        return {
            token: {"trajectory": Trajectory(pose, self.config.trajectory_sampling)}
            for token, pose in zip(batch[2], poses)
        }


def score_worker(args):
    cfg = args[0]["cfg"]
    predictions = args[0]["predictions"]
    paths = args[0]["cache_paths"]
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    rows = []
    for token in sorted(token for a in args for token in a["tokens"]):
        row = {"token": token, "valid": False}
        try:
            with lzma.open(paths[token], "rb") as stream:
                cache = pickle.load(stream)
            # v1 TTC queries the full 4 s proposal plus up to 0.9 s.
            cache.observation[49]
            pred = transform_trajectory(predictions[token]["trajectory"], cache.ego_state)
            states = np.stack([
                get_trajectory_as_array(cache.trajectory, simulator.proposal_sampling, cache.ego_state.time_point),
                get_trajectory_as_array(pred, simulator.proposal_sampling, cache.ego_state.time_point),
            ])
            simulated = simulator.simulate_proposals(states, cache.ego_state)
            scores = scorer.score_proposals(
                simulated, cache.observation, cache.centerline, cache.route_lane_ids, cache.drivable_area_map,
            )
            row.update(
                valid=True,
                no_at_fault_collisions=float(scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, 1]),
                drivable_area_compliance=float(scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 1]),
                ego_progress=float(scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, 1]),
                time_to_collision_within_bound=float(scorer._weighted_metrics[WeightedMetricIndex.TTC, 1]),
                comfort=float(scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, 1]),
                pdm_score=float(scores[1]),
            )
            metrics = [row[key] for key in (
                "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                "time_to_collision_within_bound", "comfort", "pdm_score",
            )]
            if not np.isfinite(metrics).all():
                raise ValueError("Scorer produced a non-finite metric.")
        except Exception:
            row["valid"] = False
            LOG.error("NAVSIM v1 scoring failed for token %s:\n%s", token, traceback.format_exc())
        rows.append(row)
    return rows


def evaluate(cfg, loader, cache_paths):
    pl.seed_everything(0, workers=True)
    agent = instantiate(cfg.agent)
    agent.initialize()
    inference_loader = make_loader(cfg, sensor_config=agent.get_sensor_config())
    dataset = DatasetSSL(
        scene_loader=inference_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None, cfg=cfg.agent.config,
        force_cache_computation=False, append_token_to_batch=True,
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
    predictions = trainer.predict(PredictionModule(agent, cfg.agent.config), dataloader, return_predictions=True)
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, predictions)
        if dist.get_rank() != 0:
            return
    else:
        gathered = [predictions]
    merged = {}
    for batches in gathered:
        for batch in batches:
            for token, value in batch.items():
                if token in merged and not np.array_equal(merged[token]["trajectory"].poses, value["trajectory"].poses):
                    raise RuntimeError("DDP produced different predictions for duplicate token: " + token)
                merged[token] = value
    if set(merged) != set(loader.tokens):
        raise RuntimeError(f"Prediction coverage mismatch: {len(merged)} predictions, {len(loader.tokens)} scenes.")
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if cfg.agent.config.inference.save_pickle:
        with (output / "predictions_navsim_v1.pkl").open("wb") as stream:
            pickle.dump(merged, stream, protocol=pickle.HIGHEST_PROTOCOL)
    records = worker_map(
        build_worker(cfg), score_worker,
        jobs_for(loader, cfg, predictions=merged, cache_paths=cache_paths),
    )
    frame = pd.DataFrame(records).sort_values("token").reset_index(drop=True)
    if frame["token"].duplicated().any() or set(frame["token"]) != set(loader.tokens):
        raise RuntimeError("Scoring coverage or uniqueness mismatch.")
    valid = frame[frame["valid"]]
    columns = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
               "time_to_collision_within_bound", "comfort", "pdm_score"]
    if valid.empty:
        frame.to_csv(output / "failed_scenarios.csv", index=False)
        raise RuntimeError("All scenes failed; inspect failed_scenarios.csv and logs.")
    average = valid[columns].mean()
    result = pd.concat([frame, pd.DataFrame([{
        "token": "average_all_frames", "valid": bool(frame["valid"].all()), **average.to_dict(),
    }])], ignore_index=True)
    csv_path = output / (datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + ".csv")
    result.to_csv(csv_path, index=False)
    manifest = {
        "protocol": "NAVSIM v1 / PDMS", "upstream": UPSTREAM,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_scene_count": len(loader.tokens), "successful": len(valid),
        "failed": int((~frame["valid"]).sum()), "tokens_sha256": token_hash(loader.tokens),
        "checkpoint": str(cfg.agent.checkpoint_path), "checkpoint_sha256": sha256(cfg.agent.checkpoint_path),
        "vocab_sha256": sha256(cfg.agent.config.vocab_path),
        "cache_manifest_sha256": sha256(Path(cfg.metric_cache_path) / "navsim_v1_manifest.json"),
        "csv_sha256": sha256(csv_path),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "lightning": pl.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "seed": 0, "average": average.to_dict(), "complete": bool(frame["valid"].all()),
        "selection_coefficients": {"imi": 0.05, "NC": 0.5, "DAC": 0.5, "EP": 5.0, "TTC": 5.0, "C": 2.0, "lambda_avg": 8.0},
        "comfort_prediction_head": "history_comfort (the released checkpoint has no separate C head)",
    }
    root = Path(__file__).resolve().parents[3]
    tracked_sources = [
        root / "navsim/agents/drivesuprim/drivesuprim_model.py",
        root / "navsim/agents/drivesuprim/drivesuprim_config.py",
        Path(__file__),
        root / "navsim/planning/script/config/pdm_scoring/scorer/navsim_v1.yaml",
    ] + list((root / "navsim/evaluate/navsim_v1").glob("*.py"))
    manifest["source_sha256"] = {str(p.relative_to(root)): sha256(p) for p in tracked_sources}
    (output / "reproduction_manifest.json").write_text(json.dumps(manifest, indent=2))
    LOG.info(
        "Finished NAVSIM v1 evaluation. Successful: %s. Failed: %s. Average PDMS: %.12f (%.4f%%). CSV: %s",
        len(valid), manifest["failed"], average["pdm_score"], 100 * average["pdm_score"], csv_path,
    )
    if not frame["valid"].all():
        raise RuntimeError("Some scenes failed; this run is incomplete and must not be reported as reproduction.")


@hydra.main(config_path="config/pdm_scoring", config_name="default_run_pdm_score_gpu_ssl", version_base=None)
def main(cfg: DictConfig):
    build_logger(cfg)
    if cfg.agent.config.training or not cfg.agent.config.only_ori_input:
        raise ValueError("Evaluation requires training=false and only_ori_input=true.")
    if cfg.agent.config.inference.protocol != "navsim_v1":
        raise ValueError("Select agent.config.inference.protocol=navsim_v1.")
    if cfg.scorer._target_ != "navsim.evaluate.navsim_v1.pdm_scorer.PDMScorer":
        raise ValueError("Select scorer=navsim_v1.")
    if cfg.train_test_split.data_split != "test" or cfg.train_test_split.scene_filter.max_scenes is not None:
        raise ValueError("Reproduction requires the full navtest split.")
    loader = make_loader(cfg)
    if not loader.tokens:
        raise RuntimeError("No navtest scenes found; check OPENSCENE_DATA_ROOT.")
    LOG.info("NAVSIM v1: %s selected scenes. Cache: %s", len(loader.tokens), cfg.metric_cache_path)
    mode = cfg.navsim_v1_mode
    if mode == "cache":
        build_cache(cfg, loader)
    elif mode == "evaluate":
        cache_paths = read_cache_index(cfg, loader.tokens)
        evaluate(cfg, loader, cache_paths)
    else:
        raise ValueError("navsim_v1_mode must be cache or evaluate.")


if __name__ == "__main__":
    main()
