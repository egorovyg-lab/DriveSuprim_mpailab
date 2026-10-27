"""Export and score DriveSuprim candidates for a small NAVSIM v2 pilot.

This intentionally evaluates each candidate with the ordinary pdm_score path.
Its local oracle is NOT the full EPDMS oracle: two-frame comfort is absent.
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.evaluate.drivemem_oracle_utils import sample_logs, summarize_candidates, aggregate_report
import navsim.evaluate.drivemem_oracle_utils as oracle_utils
from navsim.planning.training.dataset_ssl import DatasetSSL

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu_ssl"
METRICS = (
    "no_at_fault_collisions", "drivable_area_compliance",
    "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
    "time_to_collision_within_bound", "lane_keeping", "history_comfort", "pdm_score",
)
SAFETY = (
    "no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
    "traffic_light_compliance", "time_to_collision_within_bound",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def git_info():
    def run(*args):
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short"),
                "diff_sha256": hashlib.sha256(run("diff", "--binary").encode()).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return {"error": "git information unavailable"}


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_npz(path, **arrays):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def on_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: on_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [on_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(on_device(v, device) for v in value)
    return value


def to_numpy(value):
    return value.detach().cpu().numpy()


def export(cfg, out):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for export. Run inside the GPU-enabled container.")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {out}; choose a new directory.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates").mkdir()
    ckpt = Path(cfg.agent.checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if cfg.agent.config.inference.model != "teacher":
        raise ValueError("Pilot is configured for teacher inference.")
    if not cfg.agent.config.refinement.use_multi_stage or cfg.agent.config.inference.use_first_stage_traj_in_infer:
        raise ValueError("Pilot requires the final refinement stage.")
    if cfg.traffic_agents != "non_reactive":
        raise ValueError("Pilot requires non_reactive traffic policy.")

    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()
    agent.to("cuda")
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(original_sensor_path=Path(cfg.original_sensor_path),
                               data_path=Path(cfg.navsim_log_path), scene_filter=scene_filter,
                               sensor_config=agent.get_sensor_config())
    sampling_info = {"method": "first_scenes"}
    selected_metadata = []
    if cfg.drivemem.get("sampling", "first") == "logs":
        if scene_filter.max_scenes is not None or scene_loader.synthetic_scenes:
            raise ValueError("Log sampling requires max_scenes=null and ORIGINAL scenes only")
        records = []
        for token, frames in scene_loader.scene_frames_dicts.items():
            current = frames[scene_filter.num_history_frames - 1]
            records.append({"token": token, "log_name": current["log_name"],
                            "timestamp_us": current["timestamp"]})
        selected_metadata, sampling_info = sample_logs(
            records, int(cfg.drivemem.n_logs), int(cfg.drivemem.scenes_per_log),
            float(cfg.drivemem.min_gap_s), int(cfg.drivemem.seed),
            list(cfg.drivemem.get("excluded_logs", [])))
        scene_loader.scene_frames_dicts = {item["token"]: scene_loader.scene_frames_dicts[item["token"]]
                                          for item in selected_metadata}
        print(f"Selected {len(selected_metadata)} scenes from {sampling_info['n_logs']} logs; "
              f"eligible logs: {sampling_info['eligible_logs']}", flush=True)
    cache = MetricCacheLoader(Path(cfg.metric_cache_path))
    missing = sorted(set(scene_loader.tokens) - set(cache.tokens))
    if missing:
        raise RuntimeError(f"Missing metric cache for {len(missing)} scenes: {missing[:10]}")
    if not scene_loader.tokens:
        raise RuntimeError("No scenes selected.")
    dataset = DatasetSSL(scene_loader=scene_loader, feature_builders=agent.get_feature_builders(),
                         target_builders=agent.get_target_builders(), cache_path=None,
                         cfg=cfg.agent.config, force_cache_computation=False, append_token_to_batch=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    manifest = {"schema_version": 1, "status": "export_in_progress", "git": git_info(),
                "pilot_script_sha256": sha256(Path(__file__)),
                "utils_sha256": sha256(Path(oracle_utils.__file__)),
                "sampling": sampling_info, "selected_metadata": selected_metadata,
                "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt),
                "metric_cache_path": str(cfg.metric_cache_path),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0), "tokens": [],
                "config": OmegaConf.to_container(cfg, resolve=True)}
    write_json(out / "manifest.json", manifest)
    actual_vocab = None
    t0 = time.monotonic()
    for n, batch in enumerate(loader):
        batch = on_device(batch, "cuda")
        tokens = batch[2]
        if len(tokens) != 1:
            raise RuntimeError("Expected exactly one token per batch.")
        token = str(tokens[0])
        if token in manifest["tokens"] or Path(token).name != token:
            raise RuntimeError(f"Duplicate or unsafe token: {token}")
        # Re-run the same input with capture disabled/enabled; selection must not change.
        with torch.no_grad():
            agent._config.inference.export_candidates = False
            original, _ = agent.forward(batch)
            original_traj = to_numpy(original["final_traj"])[0].copy()
            del original
            agent._config.inference.export_candidates = True
            pred, _ = agent.forward(batch)
        stage = pred["refinement"][-1]
        trajectories = to_numpy(stage["trajs"])[0]
        ids = to_numpy(stage["indices_absolute"])[0]
        scores = to_numpy(stage["final_selection_scores"])[0]
        selected = int(to_numpy(stage["final_selected_local_indices"])[0])
        final_traj = to_numpy(pred["final_traj"])[0]
        if not np.array_equal(original_traj, final_traj) or not np.array_equal(trajectories[selected], final_traj):
            raise RuntimeError(f"Trajectory changed when exporting, token={token}")
        if selected != int(np.argmax(scores)) or len(set(ids.tolist())) != len(ids):
            raise RuntimeError(f"Invalid selection or duplicate candidate IDs, token={token}")
        if not np.isfinite(scores).all() or not np.isfinite(trajectories).all():
            raise RuntimeError(f"Non-finite candidate data: {token}")
        vocab = to_numpy(pred["trajectory_vocab"])
        if not np.array_equal(vocab[ids], trajectories):
            raise RuntimeError(f"Candidate geometry differs from active vocabulary: {token}")
        if actual_vocab is None:
            actual_vocab = vocab.copy()
            write_npz(out / "vocab.npz", vocab=actual_vocab)
            manifest["vocab_sha256"] = sha256(out / "vocab.npz")
        elif not np.array_equal(actual_vocab, vocab):
            raise RuntimeError("Vocabulary changed between scenes.")
        last = stage["layer_results"][-1]
        heads = {"head_" + key: to_numpy(value)[0] for key, value in last.items()}
        if any(not np.isfinite(v).all() for v in heads.values()):
            raise RuntimeError(f"Non-finite head scores: {token}")
        write_npz(out / "candidates" / f"{token}.npz", candidate_ids=ids,
                  selection_scores=scores, selected_local=np.array(selected, dtype=np.int64),
                  coarse_selected=np.array(int(to_numpy(pred["selected_indices"])[0]), dtype=np.int64),
                  selected_trajectory=final_traj, **heads)
        manifest["tokens"].append(token)
        write_json(out / "manifest.json", manifest)
        print(f"EXPORT {n + 1}/{len(dataset)} token={token} selected={selected} candidates={len(ids)}", flush=True)
        del pred, stage, last, batch
    if set(manifest["tokens"]) != set(scene_loader.tokens):
        raise RuntimeError("Export token set differs from selected scene set.")
    manifest["status"] = "export_complete"
    manifest["export_seconds"] = time.monotonic() - t0
    write_json(out / "manifest.json", manifest)
    print(f"Export complete: {out}", flush=True)


def compare_rows(left, right, label):
    for key in METRICS:
        a, b = float(left[key]), float(right[key])
        if not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a, b, rtol=1e-7, atol=1e-8):
            raise RuntimeError(f"{label}: {key}: {a} != {b}")


def score(cfg, out):
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "export_complete":
        raise RuntimeError("Incomplete export; start a new run.")
    if sha256(cfg.agent.checkpoint_path) != manifest["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed since export.")
    if sha256(out / "vocab.npz") != manifest["vocab_sha256"]:
        raise RuntimeError("Vocabulary file changed since export.")
    if sha256(Path(__file__)) != manifest["pilot_script_sha256"]:
        raise RuntimeError("Pilot script changed since export.")
    if sha256(Path(oracle_utils.__file__)) != manifest.get("utils_sha256"):
        raise RuntimeError("Oracle utilities changed since export.")
    original_cfg = manifest["config"]
    current_cfg = OmegaConf.to_container(cfg, resolve=True)
    for key in ("agent", "simulator", "scorer", "traffic_agents", "traffic_agents_policy", "metric_cache_path", "proposal_sampling"):
        if original_cfg[key] != current_cfg[key]:
            raise RuntimeError(f"Configuration changed since export: {key}")
    if manifest["git"].get("commit") != git_info().get("commit") or manifest["git"].get("diff_sha256") != git_info().get("diff_sha256"):
        raise RuntimeError("Git commit or tracked diff changed since export.")
    limit = int(cfg.drivemem.candidate_limit)
    if limit < 1:
        raise ValueError("candidate_limit must be positive")
    results = out / f"scoring_k{limit}"
    results.mkdir(exist_ok=True)
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise RuntimeError("Simulator and scorer sampling differ")
    policy = instantiate(cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling)
    cache = MetricCacheLoader(Path(cfg.metric_cache_path))
    with np.load(out / "vocab.npz", allow_pickle=False) as data:
        vocab = data["vocab"]
    csv_path = str(cfg.drivemem.reference_csv or "")
    reference = None
    if csv_path:
        ref = pd.read_csv(csv_path)
        if "token" not in ref or ref["token"].duplicated().any():
            raise RuntimeError("Reference CSV has missing or duplicate tokens")
        reference = ref.set_index("token")
    scoring_identity = {"export_manifest_sha256": sha256(out / "manifest.json"),
                        "candidate_limit": limit, "reference_csv": csv_path,
                        "reference_sha256": sha256(csv_path) if csv_path else None}
    identity_path = results / "scoring_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != scoring_identity:
            raise RuntimeError("Existing scoring results belong to a different export/config/reference")
    else:
        if any(results.iterdir()):
            raise RuntimeError("Cannot resume a legacy or unidentified scoring directory")
        write_json(identity_path, scoring_identity)
    all_rows, summaries = [], []
    resumed = 0
    start = time.monotonic()
    for number, token in enumerate(manifest["tokens"]):
        done_path = results / f"{token}_done.json"
        candidate_path = out / "candidates" / f"{token}.npz"
        if done_path.exists():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if (done["candidate_sha256"] != sha256(candidate_path)
                    or done["states_sha256"] != sha256(results / f"{token}_states.npz")
                    or done["summary"]["token"] != token
                    or summarize_candidates(done["rows"]) != done["summary"]):
                raise RuntimeError(f"Corrupted completed scene: {token}")
            all_rows.extend(done["rows"])
            summaries.append(done["summary"])
            resumed += 1
            print(f"RESUME {number + 1}/{len(manifest['tokens'])} token={token}", flush=True)
            continue
        with np.load(out / "candidates" / f"{token}.npz", allow_pickle=False) as data:
            ids, selection = data["candidate_ids"], data["selection_scores"]
            selected = int(data["selected_local"])
            selected_trajectory = data["selected_trajectory"]
            predicted_nc = 1 / (1 + np.exp(-np.clip(data["head_no_at_fault_collisions"].astype(np.float64), -700, 700)))
            predicted_ttc = 1 / (1 + np.exp(-np.clip(data["head_time_to_collision_within_bound"].astype(np.float64), -700, 700)))
        if not np.array_equal(vocab[ids[selected]], selected_trajectory):
            raise RuntimeError(f"Stored baseline trajectory mismatch: {token}")
        # Stable score order; baseline is forcibly included even for ties.
        full_order = np.argsort(-selection, kind="stable")
        model_rank = np.empty(len(ids), dtype=np.int64)
        model_rank[full_order] = np.arange(1, len(ids) + 1)
        order = list(full_order[:min(limit, len(ids))])
        if selected not in order:
            order[-1] = selected
        order = [selected] + [int(i) for i in order if i != selected]

        def evaluate(index):
            # Reload for every candidate because pdm_score mutates observation.
            metric_cache = cache.get_from_token(token)
            traj = Trajectory(vocab[ids[index]], instantiate(cfg.agent.config.trajectory_sampling))
            result, states = pdm_score(metric_cache, traj, simulator.proposal_sampling, simulator, scorer, policy)
            row = result.iloc[0]
            values = {key: float(row[key]) for key in METRICS}
            if not all(np.isfinite(v) for v in values.values()) or not np.isfinite(states).all():
                raise RuntimeError(f"Non-finite scorer output: {token}, candidate={index}")
            return values, states, metric_cache

        baseline, baseline_states, meta = evaluate(selected)
        repeated, repeated_states, _ = evaluate(selected)
        compare_rows(baseline, repeated, f"Baseline repeat {token}")
        if not np.allclose(baseline_states, repeated_states, rtol=1e-7, atol=1e-8):
            raise RuntimeError(f"Baseline simulated states changed on repeat: {token}")
        if reference is not None:
            if token not in reference.index:
                raise RuntimeError(f"Token missing from reference CSV: {token}")
            # Reference CSV does not contain pre-aggregation pdm_score.
            for key in METRICS[:-1]:
                if key not in reference.columns:
                    raise RuntimeError(f"Missing column in reference CSV: {key}")
                expected = float(reference.loc[token, key])
                if not np.isclose(baseline[key], expected, rtol=1e-6, atol=1e-6):
                    raise RuntimeError(f"Reference mismatch {token} {key}: {baseline[key]} != {expected}")
        rows, states_out = [], []
        for index in order:
            if index == selected:
                values, states = baseline, baseline_states
            else:
                values, states, _ = evaluate(index)
            row = {"token": token, "log_name": meta.log_name, "start_time": float(meta.timepoint.time_s),
                   "candidate_local": int(index), "candidate_id": int(ids[index]),
                   "is_baseline": index == selected, "selection_score": float(selection[index]), **values}
            row.update(model_rank=int(model_rank[index]), predicted_nc=float(predicted_nc[index]),
                       predicted_ttc=float(predicted_ttc[index]))
            row["safe_vs_baseline"] = all(values[k] + 1e-8 >= baseline[k] for k in SAFETY)
            rows.append(row)
            states_out.append(states)
        # Re-evaluate baseline after all other candidates to detect state contamination.
        after, after_states, _ = evaluate(selected)
        compare_rows(baseline, after, f"Baseline after candidates {token}")
        if not np.allclose(baseline_states, after_states, rtol=1e-7, atol=1e-8):
            raise RuntimeError(f"Baseline states changed after candidates: {token}")
        summaries.append(summarize_candidates(rows))
        all_rows.extend(rows)
        write_npz(results / f"{token}_states.npz", candidate_local=np.asarray(order, dtype=np.int64),
                  simulated_states=np.stack(states_out))
        write_json(done_path, {"rows": rows, "summary": summaries[-1],
                               "candidate_sha256": sha256(candidate_path),
                               "states_sha256": sha256(results / f"{token}_states.npz")})
        print(f"SCORE {number + 1}/{len(manifest['tokens'])} token={token} "
              f"local_gain={summaries[-1]['oracle_gain_pp']:.4f} pp", flush=True)
    pd.DataFrame(all_rows).to_csv(results / "candidate_metrics.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(results / "scene_summary.csv", index=False)
    extra_report, log_summary = aggregate_report(summary)
    log_summary.to_csv(results / "log_summary.csv", index=False)
    report = {"status": "complete", "scenes": len(summary), "candidate_limit": limit,
              "reference_csv": csv_path or None, "reference_checked": reference is not None,
              "baseline_local_pdm_mean": float(summary.baseline_local_pdm.mean()),
              "oracle_local_pdm_mean": float(summary.oracle_local_pdm.mean()),
              "safe_oracle_local_pdm_mean": float(summary.safe_oracle_local_pdm.mean()),
              "oracle_gain_mean_pp": float(summary.oracle_gain_pp.mean()),
              "safe_oracle_gain_mean_pp": float(summary.safe_oracle_gain_pp.mean()),
              "scoring_seconds": time.monotonic() - start,
              "resumed_scenes": resumed, "sampling": manifest.get("sampling", {}),
              "note": "Local PDM metric without two-frame extended comfort; not full EPDMS."}
    report.update(extra_report)
    write_json(results / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    out = Path(cfg.drivemem.output_dir)
    if cfg.drivemem.stage == "export":
        export(cfg, out)
    elif cfg.drivemem.stage == "score":
        score(cfg, out)
    else:
        raise ValueError("stage must be export or score")


if __name__ == "__main__":
    main()
