"""Prospective, train-only feature/label collection for selector data expansion.

No historical features are overwritten. Holdout records never enter cache,
GPU export, or label loops. New features are captured together on a frozen
native candidate pool, with current repeat checks and predeclared functional
consistency budgets, before any of these new outcomes are evaluated.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "download"))
import prepare_selector_dataset as acquisition
from prepare_selector_dataset import read, write, sha
from navsim.agents.drivemem import METRICS, SAFETY, token_seed
from navsim.agents.drivemem_determinism import PROFILE, prepare_environment, configure, runtime, repeat_parity, failure_details
from navsim.agents.drivemem_functional_contract import CONTRACT, parity as functional_parity
from navsim.agents.drivemem_frozen_pool import apply_pool, describe_pool

FIELDS = ("pdm_score",) + METRICS


def save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def train_records(selection):
    records = selection["records"]
    if any(r["role"] not in ("train", "holdout") for r in records):
        raise ValueError("Unknown acquisition role")
    holdout = {r["session_id"] for r in records if r["role"] == "holdout"}
    train = [dict(r) for r in records if r["role"] == "train"]
    if not train or {r["session_id"] for r in train} & holdout:
        raise ValueError("Empty train set or train/holdout overlap")
    all_tokens = [r["token"] for r in records] + [r["token"] for r in selection["legacy_training_records"]]
    if len(set(all_tokens)) != len(all_tokens):
        raise ValueError("Duplicate new/legacy/holdout scene token")
    legacy = {r["session_id"] for r in selection["legacy_training_records"]}
    if holdout & legacy:
        raise ValueError("Fresh holdout overlaps legacy training")
    # Preserve broad city/session coverage in the pilot without reading outcomes.
    grouped = defaultdict(list)
    for r in train:
        grouped[(r.get("map_name", "unknown"), r["session_id"])].append(r)
    for group in grouped.values():
        group.sort(key=lambda r: (token_seed(r["token"], 20261109), r["token"]))
    ordered, keys = [], sorted(grouped)
    while any(grouped.values()):
        for key in keys:
            if grouped[key]:
                ordered.append(grouped[key].pop(0))
    # First scene per city, then the remaining interleaved sessions.
    first, rest, seen = [], [], set()
    for r in ordered:
        city = r.get("map_name", "unknown")
        if city not in seen:
            first.append(r)
            seen.add(city)
        else:
            rest.append(r)
    return first + rest


def initialize(data, out):
    plan, requests = acquisition.verify(data)
    selection = read(data / "selection.json")
    records = train_records(selection)
    source = Path(plan["options"]["source"])
    template = read(source / "identity.json")
    if not Path(template["agent"]["checkpoint_path"]).is_file():
        raise FileNotFoundError("Missing DriveSuprim checkpoint")
    if sha(Path(template["agent"]["checkpoint_path"])) != template["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from the original teacher")
    old_sources = template["sources"]
    for name, digest in old_sources.items():
        if sha(REPO / name) != digest:
            raise RuntimeError("Frozen teacher/scorer code changed: " + name)
    bindings = dict(old_sources)
    for name in ("navsim/planning/script/run_selector_collection.py", "navsim/agents/drivemem_determinism.py",
                 "navsim/agents/drivemem_frozen_pool.py", "navsim/agents/drivemem_functional_contract.py",
                 "navsim/agents/drivemem_numeric_contract.py"):
        bindings[name] = sha(REPO / name)
    identity = dict(schema=1, protocol="prospective_train_only_frozen_pool_v2", acquisition_root=str(data),
                    acquisition_sha256=sha(data / "acquisition_plan.json"), source_identity_sha256=sha(source / "identity.json"),
                    sources=bindings, numerical_profile=PROFILE, functional_contract=CONTRACT,
                    criterion_note="Existing functional budgets declared before this new collection; IMI drift diagnostic only. "
                                   "Native baseline/trajectory/top16 must match. Current frozen-pool repeats must be bitwise equal.",
                    python=platform.python_version(), numpy=np.__version__, selected_scenes=len(records),
                    legacy_scenes=len(selection["legacy_training_records"]),
                    train_sessions=len({r["session_id"] for r in records}),
                    projected_train_sessions=len({r["session_id"] for r in records + selection["legacy_training_records"]}),
                    holdout_sessions=len(selection["fresh_holdout_sessions"]),
                    checkpoint_sha256=template["checkpoint_sha256"])
    for root in (data, source, Path(plan["options"]["addition"]), Path(plan["options"]["inventory"]), Path(plan["options"]["validation"])):
        if out == root or out in root.parents or root in out.parents:
            raise ValueError("Separate collection root required")
    marker = out / "identity.json"
    if marker.exists():
        stored = read(marker)
        if {k: v for k, v in stored.items() if k != "artifacts"} != identity:
            raise RuntimeError("Collection code/runtime/inputs changed; use a new output root")
        verify_collection(out)
        return template, stored
    if out.exists() and any(out.iterdir()):
        raise ValueError("Nonempty collection root without identity")
    committed, status = acquisition.image_status(data, requests)
    if status["status"] != "complete":
        raise RuntimeError("Acquisition images incomplete; finish fetch first")
    # Retain hashes of train input images only. Do not load/score holdout scenes.
    train_paths = acquisition.request_images(records, Path(template["navsim_log_path"]),
                    Path(template["original_sensor_path"]), {}, set())
    if train_paths["archives"] or not set(train_paths["images"]) <= set(committed):
        raise RuntimeError("Train image manifest disagrees with completed acquisition")
    out.mkdir(parents=True, exist_ok=True)
    write(out / "splits.json", dict(records=records,
          log_sha256={Path(path).stem: digest for path, digest in train_paths["log_hashes"].items()},
          purpose="New train collection only; acquisition holdout deliberately absent"))
    with (out / "inventory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write(out / "split_identity.json", dict(splits=sha(out / "splits.json"), inventory=sha(out / "inventory.csv")))
    write(out / "train_images.json", dict(sensor_root=template["original_sensor_path"], images=train_paths["initial_images"]))
    write(out / "source_config.json", template)
    identity["artifacts"] = {name: sha(out / name) for name in ("splits.json", "inventory.csv", "split_identity.json", "train_images.json", "source_config.json")}
    write(marker, identity)
    print(json.dumps(dict(status="initialized", scenes=len(records), cities=dict(Counter(r.get("map_name", "unknown") for r in records)),
              holdout_processed=0, projected_total_train_scenes=identity["legacy_scenes"] + len(records)), indent=2), flush=True)
    return template, identity


def verify_collection(out):
    identity = read(out / "identity.json")
    acquisition.verify(Path(identity["acquisition_root"]))
    if sha(Path(identity["acquisition_root"]) / "acquisition_plan.json") != identity["acquisition_sha256"]:
        raise RuntimeError("Frozen acquisition changed")
    for name, digest in identity["sources"].items():
        if sha(REPO / name) != digest:
            raise RuntimeError("Frozen collection dependency changed: " + name)
    if identity["python"] != platform.python_version() or identity["numpy"] != np.__version__:
        raise RuntimeError("Collection Python/NumPy runtime changed")
    for name, digest in identity["artifacts"].items():
        if sha(out / name) != digest:
            raise RuntimeError("Frozen collection artifact changed: " + name)
    records = read(out / "splits.json")["records"]
    if any(r["role"] != "train" for r in records) or len(records) != identity["selected_scenes"]:
        raise ValueError("Train-only membership changed")
    return identity, records


def cfg_from_source(template):
    from omegaconf import OmegaConf
    keys = ("agent", "scorer", "simulator", "proposal_sampling", "traffic_agents", "traffic_agents_policy",
            "train_test_split", "navsim_log_path", "original_sensor_path")
    return OmegaConf.create({k: template[k] for k in keys})


def verify_train_images(out):
    manifest = read(out / "train_images.json")
    for number, (relative, info) in enumerate(manifest["images"].items(), 1):
        path = acquisition.checked_target(Path(manifest["sensor_root"]), relative)
        if not path.is_file() or path.stat().st_size != info["size"] or sha(path) != info["sha256"]:
            raise RuntimeError("Frozen train image missing/changed: " + str(path))
        if number % 2000 == 0:
            print(f"VERIFY TRAIN IMAGES {number}/{len(manifest['images'])}", flush=True)


def checked_index(out, folder, filename, records):
    index = read(out / filename) if (out / filename).exists() else {}
    expected = {r["token"] for r in records}
    if set(index) - expected:
        raise ValueError("Unexpected scene in " + filename)
    for token, entry in index.items():
        path = Path(entry["path"]) if isinstance(entry, dict) and "path" in entry else out / folder / (token + ".npz")
        if out not in path.resolve().parents:
            raise ValueError("Collection file escapes output root")
        digest = entry["sha256"] if isinstance(entry, dict) else entry
        if sha(path) != digest:
            raise RuntimeError("Committed collection file changed: " + str(path))
    return index


def checked_features(out, records):
    index = checked_index(out, "features", "export_index.json", records)
    native = checked_index(out, "native", "native_index.json", records)
    if index:
        from navsim.agents.drivemem_determinism import validate_runtime
        validate_runtime(out, index, read, sha)
        if sha(out / "vocab.npz") != read(out / "vocab_identity.json")["sha256"]:
            raise RuntimeError("Vocabulary changed")
        for token, entry in index.items():
            if entry["native_sha256"] != native.get(token):
                raise RuntimeError("Native snapshot/export binding changed")
    return index


def validate_sample(a, vocab, full=True):
    ids, scores, b = a["candidate_ids"], a["selection_scores"], int(a["selected_local"])
    if (ids.shape != (256,) or ids.dtype.kind not in "iu" or len(set(ids.tolist())) != 256
            or ids.min() < 0 or ids.max() >= len(vocab) or scores.shape != (256,)
            or b != int(np.argmax(scores)) or not np.array_equal(a["final_traj"], vocab[ids[b]])):
        raise ValueError("Invalid candidate pool/baseline/trajectory")
    heads = {"head_" + k for k in METRICS} | {"head_imi"}
    if {k for k in a if k.startswith("head_")} != heads or any(a[k].shape != (256,) for k in heads):
        raise ValueError("Invalid metric heads")
    if not all(np.isfinite(v).all() for v in a.values()):
        raise ValueError("Nonfinite exported feature")
    if full and (a["decoder_states"].shape != (256, 256) or a["scene_embedding"].ndim != 1 or a["ego_status"].ndim != 1):
        raise ValueError("Invalid decoder/context feature shape")


def commit_feature(out, token, native, unhooked, fresh, vocab, extra, index, diagnostics):
    validate_sample(native, vocab, full=False)
    validate_sample(fresh, vocab, full=False)
    repeat_parity(unhooked, fresh)
    functional_parity(native, fresh, vocab)
    merged = dict(fresh, **extra)
    validate_sample(merged, vocab)
    save(out / "features" / (token + ".npz"), **merged)
    index[token] = dict(sha256=sha(out / "features" / (token + ".npz")), native_sha256=sha(out / "native" / (token + ".npz")),
                       runtime_sha256=sha(out / "runtime.json"), repeat_bitwise_equal=True, natural_pool_passes=diagnostics)
    write(out / "export_index.json", index)


def export_features(cfg, out, records, limit):
    prepare_environment()
    import torch
    configure(torch)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. CPU cache/labels are independent; completed files preserved.")
    observed = runtime(torch)
    if (out / "failure.json").exists():
        raise RuntimeError("Previous parity/alignment failure recorded; inspect failure.json before continuing")
    if (out / "runtime.json").exists() and read(out / "runtime.json") != observed:
        raise RuntimeError("GPU/library runtime changed; preserve this collection and investigate")
    write(out / "runtime.json", observed)
    print(json.dumps(dict(gpu=observed["gpu_name"], torch=observed["torch"], profile=PROFILE)), flush=True)
    index = checked_features(out, records)
    native_index = read(out / "native_index.json") if (out / "native_index.json").exists() else {}
    if (out / "vocab.npz").exists() and sha(out / "vocab.npz") != read(out / "vocab_identity.json")["sha256"]:
        raise RuntimeError("Vocabulary changed")
    todo = [r for r in records if r["token"] not in index][:limit]
    if not todo:
        print("Feature export already complete", flush=True)
        return
    verify_train_images(out)
    from hydra.utils import instantiate
    from torch.utils.data import DataLoader
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.training.dataset_ssl import DatasetSSL
    from navsim.planning.script.run_drivemem_oracle_pilot import on_device
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()
    agent.to("cuda")
    if cfg.agent.config.inference.model != "teacher" or cfg.agent.config.refinement.num_refinement_stage != 1:
        raise ValueError("Teacher with one refinement stage required")
    agent._config.inference.export_candidates = True
    teacher = agent.model.teacher["model"]
    def snapshot(pred):
        stage = pred["refinement"][-1]
        a = dict(candidate_ids=stage["indices_absolute"].cpu().numpy()[0].copy(),
                 selected_local=stage["final_selected_local_indices"].cpu().numpy()[0].copy(),
                 selection_scores=stage["final_selection_scores"].cpu().numpy()[0].copy(),
                 final_traj=pred["final_traj"].cpu().numpy()[0].copy())
        a.update({"head_" + k: v.cpu().numpy()[0].copy() for k, v in stage["layer_results"][-1].items()})
        return a
    start = time.monotonic()
    for log in sorted({r["log_name"] for r in todo}):
        selected = [r for r in todo if r["log_name"] == log]
        sf = instantiate(cfg.train_test_split.scene_filter)
        sf.log_names, sf.tokens = [log], [r["token"] for r in selected]
        loader = SceneLoader(Path(cfg.navsim_log_path), Path(cfg.original_sensor_path), sf, sensor_config=agent.get_sensor_config())
        if set(loader.tokens) != set(sf.tokens):
            raise ValueError("Feature loader membership changed")
        dataset = DatasetSSL(loader, agent.get_feature_builders(), agent.get_target_builders(), cfg=cfg.agent.config,
                             cache_path=None, append_token_to_batch=True)
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
            batch = on_device(batch, "cuda")
            token = str(batch[2][0])
            with torch.no_grad():
                if token not in native_index:
                    pred, _ = agent.forward(batch)
                    native = snapshot(pred)
                    vocab = pred["trajectory_vocab"].detach().cpu().numpy().copy()
                    validate_sample(native, vocab, full=False)
                    if (out / "vocab.npz").exists():
                        if not np.array_equal(vocab, arrays(out / "vocab.npz")["vocab"]):
                            raise ValueError("Vocabulary changed")
                    else:
                        save(out / "vocab.npz", vocab=vocab)
                        write(out / "vocab_identity.json", dict(sha256=sha(out / "vocab.npz")))
                    save(out / "native" / (token + ".npz"), **native)
                    native_index[token] = sha(out / "native" / (token + ".npz"))
                    write(out / "native_index.json", native_index)
                    del pred
                native = arrays(out / "native" / (token + ".npz"))
                vocab = arrays(out / "vocab.npz")["vocab"]
                replay, capture, handles, fresh = dict(natural=[]), {}, [], None
                def coarse(module, inp):
                    replay["coarse"] = inp[0].detach()
                def pool(module, inp, output):
                    current = output["refinement"][0]["indices_absolute"].cpu().numpy()[0].copy()
                    replay["natural"].append(describe_pool(native["candidate_ids"], current))
                    return apply_pool(output, replay.pop("coarse"), native["candidate_ids"])
                def hidden(module, inp, output):
                    if not isinstance(output, list) or not output:
                        raise ValueError("Unexpected decoder output")
                    capture["hidden"] = output[-1].detach().clone()
                    capture["calls"] = capture.get("calls", 0) + 1
                def embedding(module, inp, output):
                    capture["embedding"] = output.mean(dim=(2, 3)).cpu().numpy()[0].copy()
                try:
                    handles.append(teacher._trajectory_head.heads["no_at_fault_collisions"].register_forward_pre_hook(coarse))
                    handles.append(teacher._trajectory_head.register_forward_hook(pool))
                    pred, _ = agent.forward(batch)
                    unhooked = snapshot(pred)
                    del pred
                    handles.append(teacher.downscale_layer.register_forward_hook(embedding))
                    handles.append(teacher._trajectory_offset_head.transformer_blocks[-1].register_forward_hook(hidden))
                    pred, _ = agent.forward(batch)
                    fresh = snapshot(pred)
                    if capture.get("calls") != 1 or len(replay["natural"]) != 2:
                        raise RuntimeError("Unexpected capture call count")
                    for name, head in teacher._trajectory_offset_head.multi_stage_heads[-1].items():
                        value = head(capture["hidden"]).squeeze(-1).cpu().numpy()[0]
                        if not np.allclose(value, fresh["head_" + name], atol=1e-5, rtol=1e-6):
                            raise RuntimeError("Decoder/head alignment failure: " + name)
                    extra = dict(decoder_states=capture["hidden"].cpu().numpy()[0].copy(),
                        scene_embedding=capture["embedding"], ego_status=torch.cat(batch[0]["status_feature"], dim=1).cpu().numpy()[0].copy())
                    commit_feature(out, token, native, unhooked, fresh, vocab, extra, index, replay["natural"])
                except Exception as exc:
                    detail = dict(status="export_failed", token=token, error=str(exc))
                    if fresh is not None:
                        detail["native_vs_capture"] = failure_details(native, fresh)
                    write(out / "failure.json", detail)
                    raise
                finally:
                    for handle in reversed(handles):
                        handle.remove()
                print(f"COLLECT EXPORT {len(index)}/{len(records)} {token}", flush=True)
                del pred, batch, capture, replay, extra
    print(json.dumps(dict(newly_exported_scenes=len(todo), seconds=time.monotonic() - start)), flush=True)


def load_sample(out, r, index):
    path = out / "features" / (r["token"] + ".npz")
    if sha(path) != index[r["token"]]["sha256"]:
        raise RuntimeError("Feature checksum changed")
    return dict(arrays(path), **{k: r[k] for k in ("token", "session_id", "role")})


def check_outcomes(values):
    for version in ("raw", "filtered"):
        v = np.asarray([values[version][k] for k in FIELDS], dtype=float)
        if not np.isfinite(v).all() or np.any((v < 0) | (v > 1)):
            raise ValueError("Invalid scorer outcomes")


def collect_labels(out, records, index, evaluate, limit):
    previous = out / "support_report.json"
    if previous.exists():
        for name, digest in read(previous)["labels"].items():
            if sha(out / "top16_labels" / name) != digest:
                raise RuntimeError("Previously completed training labels changed")
    done, new = 0, 0
    start = time.monotonic()
    for r in records:
        if r["role"] != "train":
            raise ValueError("Labels refuse non-training roles")
        s = load_sample(out, r, index)
        ix = np.argsort(-s["selection_scores"], kind="stable")[:16].tolist()
        b = int(s["selected_local"])
        if b not in ix:
            raise ValueError("Baseline outside top16")
        spec = dict(token=r["token"], session_id=r["session_id"], role="train", feature_sha256=index[r["token"]]["sha256"],
                    top16=ix, candidate_ids=s["candidate_ids"][ix].tolist())
        path = out / "top16_labels" / (r["token"] + ".json")
        state = read(path) if path.exists() else dict(**spec, outcomes={})
        if any(state.get(k) != v for k, v in spec.items()) or not set(state["outcomes"]) <= set(map(str, ix)):
            raise RuntimeError("Label provenance changed")
        for value in state["outcomes"].values():
            check_outcomes(value)
        if len(state["outcomes"]) == 16:
            done += 1
            continue
        if new >= limit:
            continue
        baseline, _ = evaluate(r["token"], int(s["candidate_ids"][b]))
        check_outcomes(baseline)
        if str(b) in state["outcomes"]:
            for version in ("raw", "filtered"):
                if not np.allclose([state["outcomes"][str(b)][version][k] for k in FIELDS],
                                   [baseline[version][k] for k in FIELDS], atol=1e-8, rtol=1e-7):
                    raise RuntimeError("Resumed baseline scorer changed")
        state["outcomes"][str(b)] = baseline
        write(path, state)
        for i in ix:
            if str(i) not in state["outcomes"]:
                value, _ = evaluate(r["token"], int(s["candidate_ids"][i]))
                check_outcomes(value)
                state["outcomes"][str(i)] = value
                write(path, state)
        done, new = done + 1, new + 1
        print(f"COLLECT TOP16 {done}/{len(records)} {r['token']}", flush=True)
    return dict(completed_scenes=done, newly_completed_scenes=new, seconds=time.monotonic() - start)


def report(out, identity, records):
    features = checked_features(out, records)
    cache = checked_index(out, "", "cache_index.json", records)
    completed, hashes, rows, events = [], {}, [], []
    for r in records:
        path = out / "top16_labels" / (r["token"] + ".json")
        if not path.exists():
            continue
        state = read(path)
        if r["token"] not in features or state["feature_sha256"] != features[r["token"]]["sha256"]:
            raise RuntimeError("Labels refer to changed features")
        s = load_sample(out, r, features)
        ix = np.argsort(-s["selection_scores"], kind="stable")[:16].tolist()
        if (state["top16"] != ix or any(state[k] != r[k] for k in ("token", "session_id", "role"))
                or state["candidate_ids"] != s["candidate_ids"][ix].tolist() or not set(state["outcomes"]) <= set(map(str, ix))):
            raise RuntimeError("Label membership/provenance changed")
        if len(state["outcomes"]) != 16:
            continue
        for value in state["outcomes"].values():
            check_outcomes(value)
        completed.append(r["token"])
        hashes[path.name] = sha(path)
        b = state["outcomes"][str(int(s["selected_local"]))]["raw"]
        rows.append(dict(token=r["token"], session_id=r["session_id"], map_name=r.get("map_name", "unknown"),
                         baseline=b["pdm_score"], oracle_gain_pp=100 * (max(v["raw"]["pdm_score"] for v in state["outcomes"].values()) - b["pdm_score"])))
        for i in ix:
            if i == int(s["selected_local"]):
                continue
            for j in SAFETY:
                key = METRICS[j]
                value = state["outcomes"][str(i)]["raw"][key]
                events.append(dict(metric=key, session_id=r["session_id"], token=r["token"], below_one=value < 1 - 1e-8,
                                   degraded=value < b[key] - 1e-8, improved=value > b[key] + 1e-8))
    support = {}
    for metric in (METRICS[j] for j in SAFETY):
        group = [r for r in events if r["metric"] == metric]
        support[metric] = dict(actions=len(group), below_one=sum(r["below_one"] for r in group),
            degraded=sum(r["degraded"] for r in group), improved=sum(r["improved"] for r in group),
            degradation_scenes=len({r["token"] for r in group if r["degraded"]}),
            degradation_sessions=len({r["session_id"] for r in group if r["degraded"]}),
            improvement_sessions=len({r["session_id"] for r in group if r["improved"]}))
    result = dict(status="complete" if len(completed) == len(records) else "partial", selected_scenes=len(records),
        cached_scenes=len(cache), exported_scenes=len(features), labelled_scenes=len(completed),
        missing_label_scenes=len(records) - len(completed), holdout_processed=0,
        projected_total_train_scenes=identity["legacy_scenes"] + len(records),
        ready_train_scenes=identity["legacy_scenes"] + len(completed), safety_support=support, labels=hashes,
        scene_summaries=rows, note="Training data collection only. Stored raw PDM; no reconstruction from filtered metrics. "
            "Candidate rows correlated; zero observed degradation is not calibrated safety. Local PDM, not full EPDMS.")
    previous = out / "support_report.json"
    if previous.exists():
        for name, digest in read(previous)["labels"].items():
            if sha(out / "top16_labels" / name) != digest:
                raise RuntimeError("Previously completed training labels changed")
    write(previous, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("labels", "scene_summaries")}, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory"
    p.add_argument("stage", choices=("init", "cache", "export-pilot", "export", "label-pilot", "label", "status", "report"))
    p.add_argument("--data", type=Path, default=base / "selector_data_expansion_v2")
    p.add_argument("--out", type=Path, default=base / "selector_collection_v2")
    a = p.parse_args()
    data, out = a.data.resolve(), a.out.resolve()
    if a.stage == "init":
        initialize(data, out)
        return
    identity, records = verify_collection(out)
    if Path(identity["acquisition_root"]) != data:
        raise ValueError("Collection belongs to another acquisition")
    template = read(out / "source_config.json")
    if sha(Path(template["agent"]["checkpoint_path"])) != identity["checkpoint_sha256"]:
        raise RuntimeError("Teacher checkpoint changed")
    if a.stage in ("status", "report"):
        report(out, identity, records)
        return
    cfg = cfg_from_source(template)
    if template.get("map_root") and os.environ.get("NUPLAN_MAPS_ROOT"):
        if Path(os.environ["NUPLAN_MAPS_ROOT"]).resolve() != Path(template["map_root"]).resolve():
            raise RuntimeError("Map root differs from original teacher/scorer configuration")
    if not os.environ.get("NUPLAN_MAPS_ROOT"):
        if not template.get("map_root"):
            raise RuntimeError("Set NUPLAN_MAPS_ROOT to the installed nuPlan maps")
        os.environ["NUPLAN_MAPS_ROOT"] = template["map_root"]
    if a.stage == "cache":
        from navsim.planning.script.run_drivemem_memory import build_cache
        build_cache(cfg, out)
    elif a.stage.startswith("export"):
        export_features(cfg, out, records, 4 if a.stage == "export-pilot" else len(records))
    else:
        index = checked_features(out, records)
        cache = checked_index(out, "", "cache_index.json", records)
        if len(index) != len(records) or len(cache) != len(records):
            raise RuntimeError("Complete cache and feature export before labelling")
        if sha(out / "vocab.npz") != read(out / "vocab_identity.json")["sha256"]:
            raise RuntimeError("Vocabulary changed")
        # Freeze all candidate pools/feature hashes before opening any new labels.
        spec = dict(collection_identity_sha256=sha(out / "identity.json"), export_index_sha256=sha(out / "export_index.json"),
                    cache_index_sha256=sha(out / "cache_index.json"), vocab_sha256=sha(out / "vocab.npz"))
        marker = out / "label_identity.json"
        if marker.exists() and read(marker) != spec:
            raise RuntimeError("Frozen label inputs changed")
        write(marker, spec)
        from navsim.planning.script.run_drivemem_memory import evaluators
        evaluate, _ = evaluators(cfg, out)
        timing = collect_labels(out, records, index, evaluate, 8 if a.stage == "label-pilot" else len(records))
        write(out / "last_label_timing.json", timing)
        print(json.dumps(timing, indent=2), flush=True)
    report(out, identity, records)


if __name__ == "__main__":
    main()
