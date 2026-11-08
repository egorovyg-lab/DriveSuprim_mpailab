"""Expand selector image data under an ADDITIONAL compressed-transfer cap.

Plan uses metadata, local image presence, and archive costs only. No outcomes,
model fitting, GPU work, or image-archive downloads occur during planning.
The manifest is for a new dataset; old collectors with fixed 32/80/152 quotas
must not be used on it. Labels/features are a subsequent collection stage.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from navsim.agents.drivemem import session_id, spaced_records, token_seed
from plan_drivemem_images import BASE, CAMERAS, MAPPING, MAPPING_SHA256, archive_entry, fetch_bytes, safe_image_path, sha
from fetch_drivemem_budget_images import checked_target, download, extract_images


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sources(args):
    roots = [args.source, args.addition, args.inventory, args.validation] + args.exclude_run
    bindings, splits = {}, {}
    for root in roots:
        split = read(root / "splits.json")
        frozen = read(root / "split_identity.json")
        if frozen != dict(splits=sha(root / "splits.json"), inventory=sha(root / "inventory.csv")):
            raise RuntimeError("Frozen split changed: " + str(root))
        for name in ("identity.json", "splits.json", "split_identity.json", "inventory.csv"):
            bindings[str(root / name)] = sha(root / name)
        splits[root] = split
    selection = read(args.addition / "label_selection.json")
    bindings[str(args.addition / "label_selection.json")] = sha(args.addition / "label_selection.json")
    addition = {r["token"]: r for r in splits[args.addition]["records"]}
    legacy = [dict(r) for r in splits[args.source]["records"] if r["role"] in ("memory", "fit")]
    for r in selection["records"]:
        actual = addition[r["token"]]
        if r["role"] not in ("memory", "fit") or any(r[k] != actual[k] for k in ("role", "session_id")):
            raise ValueError("Unexpected existing M/F selection")
        legacy.append(dict(actual))
    if not legacy or len({r["token"] for r in legacy}) != len(legacy):
        raise ValueError("Empty or duplicate legacy training set")
    train = {r["session_id"] for r in legacy}
    observed = {r["session_id"] for root in (args.source, args.validation)
                for r in splits[root]["records"] if r["role"] == "development"}
    if train & observed:
        raise ValueError("Training session overlaps inspected development")
    reserved = {r["session_id"] for split in splits.values() for r in split["records"]
                if r["role"] in ("calibration", "test")}
    conflicts = sorted(reserved & train)  # Already-used M/F can never become fresh holdout.
    excluded = observed | (reserved - train)
    for split in splits.values():
        excluded.update(split.get("navtest_sessions_excluded", []))
    for root in args.exclude_run:
        excluded.update(r["session_id"] for r in splits[root]["records"])
    if train & excluded:
        raise ValueError("Current training session conflicts with an explicit exclusion/navtest")
    identity = read(args.source / "identity.json")
    return bindings, legacy, train, excluded, identity, conflicts


def metadata(out):
    mapping_path, listing_path = out / "metadata/mapping.json", out / "metadata/archives.json"
    if mapping_path.exists():
        raw = mapping_path.read_bytes()
    else:
        raw = fetch_bytes(BASE + "/resolve/main/" + MAPPING + "?download=1")
        if hashlib.sha256(raw).hexdigest() != MAPPING_SHA256:
            raise RuntimeError("Official mapping changed")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_path.write_bytes(raw)
    if hashlib.sha256(raw).hexdigest() != MAPPING_SHA256:
        raise RuntimeError("Cached official mapping changed")
    if not listing_path.exists():
        url = "https://huggingface.co/api/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_trainval_camera?limit=1000"
        write(listing_path, json.loads(fetch_bytes(url)))
    reverse, entries, shards = {}, {}, set()
    for key, logs in json.loads(raw).items():
        match = re.fullmatch(r"openscene_sensor_trainval_(\d+)", key)
        if not match or not isinstance(logs, list):
            raise ValueError("Unexpected official mapping schema")
        shard = int(match[1])
        shards.add(shard)
        for log in logs:
            if log in reverse:
                raise ValueError("Ambiguous log/archive mapping")
            reverse[log] = shard
    for entry in read(listing_path):
        match = re.fullmatch(r"openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_(\d+)\.tgz", entry.get("path", ""))
        if match:
            shard = int(match[1])
            if shard in entries:
                raise ValueError("Duplicate official archive entry")
            entries[shard] = archive_entry(entry)
    if set(entries) != shards:
        raise RuntimeError("Incomplete official archive listing; no accurate budget possible")
    return reverse, entries


def candidate_rows(rows, legacy, excluded, installed, reverse, gap_s):
    past, used = defaultdict(list), {r["token"] for r in legacy}
    for r in legacy:
        past[r["session_id"]].append(int(r["timestamp_us"]))
    for times in past.values():
        times.sort()
    unique = {}
    gap = round(gap_s * 1e6)
    for r in rows:
        sid, t, log = r["session_id"], int(r["timestamp_us"]), r["log_name"]
        if sid != session_id(log):
            raise ValueError("Inventory session/log mismatch")
        if sid in excluded or r["token"] in used or log not in installed or log not in reverse:
            continue
        times = past.get(sid, [])
        i = bisect.bisect_left(times, t)
        if (i and t - times[i - 1] < gap) or (i < len(times) and times[i] - t < gap):
            continue
        key = (sid, t)
        item = dict(r, timestamp_us=t, shard=reverse[log])
        if key not in unique or (log, r["token"]) < (unique[key]["log_name"], unique[key]["token"]):
            unique[key] = item
    result = sorted(unique.values(), key=lambda r: (r["session_id"], r["timestamp_us"], r["token"]))
    if len({r["token"] for r in result}) != len(result):
        raise ValueError("Duplicate inventory tokens")
    return result


def load_frames(log_path):
    # These are the user's installed, trusted OpenScene metadata pickles.
    with log_path.open("rb") as f:
        frames = pickle.load(f)
    positions = {f["token"]: i for i, f in enumerate(frames)}
    if len(positions) != len(frames):
        raise ValueError("Duplicate frame tokens: " + str(log_path))
    return frames, positions


def camera_paths(frames, position, history=4):
    if position < history - 1:
        raise ValueError("Insufficient camera history")
    paths = set()
    for frame in frames[position - history + 1:position + 1]:
        cams = {k.lower(): v for k, v in frame["cams"].items() if k.lower() in CAMERAS}
        if set(cams) != CAMERAS:
            raise ValueError("Missing camera metadata")
        paths.update(safe_image_path(c["data_path"]) for c in cams.values())
    return paths


def complete_rgb_tokens(rows, logs_path, sensor_root, known_logs):
    complete = set()
    @lru_cache(maxsize=None)
    def present(relative):
        path = checked_target(sensor_root, relative)
        return path.is_file() and path.stat().st_size > 0
    groups = defaultdict(list)
    for r in rows:
        if r["log_name"] in known_logs:
            groups[r["log_name"]].append(r)
    for number, (log, group) in enumerate(sorted(groups.items()), 1):
        print(f"CHECK EXISTING RGB {number}/{len(groups)} {log}", flush=True)
        frames, positions = load_frames(logs_path / (log + ".pkl"))
        for r in group:
            if r["token"] not in positions:
                raise ValueError("Inventory token absent from source metadata")
            if all(present(p) for p in camera_paths(frames, positions[r["token"]])):
                complete.add(r["token"])
    return complete


def capacities(rows_by_session, active, free_tokens, legacy_counts, cap, gap_s):
    result = {}
    gap = round(gap_s * 1e6)
    for sid, rows in rows_by_session.items():
        last, count, limit = None, 0, cap - legacy_counts.get(sid, 0)
        for r in rows:
            if count >= limit:
                break
            if r["shard"] not in active and r["token"] not in free_tokens:
                continue
            t = r["timestamp_us"]
            if last is None or t - last >= gap:
                last, count = t, count + 1
        result[sid] = count
    return result


def acquire(rows, entries, cached, free_tokens, legacy_counts, budget, minimum, cap, target, gap):
    groups = defaultdict(list)
    affected = defaultdict(set)
    for r in rows:
        groups[r["session_id"]].append(r)
        affected[r["shard"]].add(r["session_id"])
    def capacity(active):
        return capacities(groups, active, free_tokens, legacy_counts, cap, gap)
    def extend(caps, active, shard):
        after = dict(caps)
        subset = {sid: groups[sid] for sid in affected[shard]}
        after.update(capacities(subset, active | {shard}, free_tokens, legacy_counts, cap, gap))
        return after
    def score(caps):
        fresh = [n for sid, n in caps.items() if sid not in legacy_counts]
        complete = [n for n in fresh if n >= minimum]
        partial = sum(n / minimum for n in fresh if n < minimum)
        return 1000 * min(len(complete), target) + sum(complete) + partial + .5 * sum(caps.get(sid, 0) for sid in legacy_counts)
    candidates = sorted({r["shard"] for r in rows} - set(cached))
    candidates = [i for i in candidates if entries[i]["size"] <= budget]
    initial = capacity(set(cached))
    initial_score = score(initial)
    first = sorted(candidates, key=lambda i: (-(score(extend(initial, set(cached), i)) - initial_score) / entries[i]["size"], i))[:6]
    solutions = []
    for start in [None] + first:
        print(f"SEARCH acquisition start={start} additional_cap={budget / 1e9:g} GB", flush=True)
        chosen = set() if start is None else {start}
        spent = sum(entries[i]["size"] for i in chosen)
        caps = capacity(set(cached) | chosen)
        while True:
            current = score(caps)
            trials = []
            for i in candidates:
                if i in chosen or spent + entries[i]["size"] > budget:
                    continue
                after = extend(caps, set(cached) | chosen, i)
                gain = score(after) - current
                if gain > 0:
                    trials.append((gain / entries[i]["size"], -entries[i]["size"], -i, i, after))
            if not trials:
                break
            _, _, _, i, caps = max(trials, key=lambda x: x[:3])
            chosen.add(i)
            spent += entries[i]["size"]
        solutions.append((score(caps), -spent, tuple(sorted(chosen)), caps))
    best = max(solutions, key=lambda x: (x[0], x[1], tuple(-i for i in x[2])))
    return set(best[2]), best[3], initial


def freeze_records(rows, caps, active, free_tokens, legacy_counts, args):
    fresh = sorted(sid for sid, n in caps.items() if sid not in legacy_counts and n >= args.min_scenes)
    if len(fresh) <= args.holdout_sessions:
        raise ValueError(f"Only {len(fresh)} new eligible sessions; need more than {args.holdout_sessions} to reserve holdout and expand training")
    order = np.random.default_rng(args.seed).permutation(fresh).tolist()
    holdout = set(order[:args.holdout_sessions])
    selected = []
    for sid in sorted(set(legacy_counts) | set(fresh)):
        n = args.min_scenes if sid in holdout else caps.get(sid, 0)
        if not n:
            continue
        available = [r for r in rows if r["session_id"] == sid and (r["shard"] in active or r["token"] in free_tokens)]
        pick = spaced_records(available, n, args.gap_s, token_seed(sid, args.seed))
        selected.extend(dict(r, role="holdout" if sid in holdout else "train",
                             session_origin="existing_training" if sid in legacy_counts else "new_session") for r in pick)
    return selected, fresh, holdout


def request_images(records, logs_path, sensor_root, entries, cached):
    requests, initial, log_hashes, archives = set(), {}, {}, {}
    for log in sorted({r["log_name"] for r in records}):
        path = logs_path / (log + ".pkl")
        before = sha(path)
        frames, positions = load_frames(path)
        for r in (r for r in records if r["log_name"] == log):
            if r["token"] not in positions:
                raise ValueError("Selected token absent from source log")
            current = positions[r["token"]]
            if int(frames[current]["timestamp"]) != int(r["timestamp_us"]):
                raise ValueError("Selected timestamp disagrees with source log")
            paths = camera_paths(frames, current)
            requests.update(paths)
            for image in sorted(paths):
                target = checked_target(sensor_root, image)
                if target.is_file() and target.stat().st_size > 0:
                    if image not in initial:
                        initial[image] = dict(size=target.stat().st_size, sha256=sha(target))
                else:
                    shard = r["shard"]
                    entry = archives.setdefault(shard, dict(entries[shard], shard=shard, cached=shard in cached, images=[]))
                    entry["images"].append(image)
        if sha(path) != before:
            raise RuntimeError("Source log changed while planning")
        log_hashes[str(path)] = before
    for entry in archives.values():
        entry["images"] = sorted(set(entry["images"]))
    return dict(sensor_root=str(sensor_root), images=sorted(requests), initial_images=initial,
                archives=[archives[i] for i in sorted(archives)], log_hashes=log_hashes,
                sensor_policy="Original agent: eight cameras, four past/current frames; no future images")


def options(args):
    return {k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v
            for k, v in vars(args).items() if k != "stage"}


def verify(out, args=None):
    plan = read(out / "acquisition_plan.json")
    if args is not None and plan["options"] != options(args):
        raise RuntimeError("Dataset planning options changed; use a new --out root")
    for path, digest in plan["inputs"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError("Frozen acquisition input changed: " + path)
    for relative, digest in plan["artifacts"].items():
        if sha(out / relative) != digest:
            raise RuntimeError("Frozen acquisition artifact changed: " + relative)
    requests = read(out / "image_requests.json")
    if sum(e["size"] for e in requests["archives"] if not e["cached"]) > plan["additional_budget_bytes"]:
        raise RuntimeError("Additional transfer cap exceeded")
    return plan, requests


def plan(args):
    out = args.out
    if (out / "acquisition_plan.json").exists():
        result, _ = verify(out, args)
        print(json.dumps(result["summary"], indent=2), flush=True)
        return result
    for root in (args.source, args.addition, args.inventory, args.validation, args.archive_dir) + tuple(args.exclude_run):
        if root == out or root in out.parents or out in root.parents:
            raise ValueError("New output must be separate from source/archive roots")
    if out.exists() and any(p.name != "metadata" for p in out.iterdir()):
        raise ValueError("Nonempty output without acquisition plan; preserve it and use a new root")
    bindings, legacy, train_sessions, excluded, identity, conflicts = sources(args)
    reverse, entries = metadata(out)
    logs_path, sensor_root = Path(identity["navsim_log_path"]), Path(identity["original_sensor_path"])
    installed = {p.stem for p in logs_path.glob("*.pkl")}
    with (args.inventory / "inventory.csv").open(encoding="utf-8", newline="") as stream:
        rows = candidate_rows(list(csv.DictReader(stream)), legacy, excluded, installed, reverse, args.gap_s)
    cached = {i for i, e in entries.items() if (args.archive_dir / Path(e["path"]).name).is_file()
              and (args.archive_dir / Path(e["path"]).name).stat().st_size == e["size"]}
    known_logs = {p.name for p in sensor_root.iterdir() if p.is_dir()} & installed if sensor_root.is_dir() else set()
    for root in (args.source, args.addition, args.inventory, args.validation):
        old_requests = root / "image_preparation/requests.json"
        if old_requests.exists():
            previous = read(old_requests)
            for log, paths in previous.get("images_by_log", {}).items():
                if any(checked_target(sensor_root, image).is_file() for image in paths):
                    known_logs.add(log)
    free_tokens = complete_rgb_tokens(rows, logs_path, sensor_root, known_logs)
    counts = Counter(r["session_id"] for r in legacy)
    budget = int(args.max_extra_gb * 1e9)
    chosen, caps, initial = acquire(rows, entries, cached, free_tokens, counts, budget,
                                    args.min_scenes, args.max_scenes, args.target_new_sessions, args.gap_s)
    records, fresh, holdout = freeze_records(rows, caps, cached | chosen, free_tokens, counts, args)
    requests = request_images(records, logs_path, sensor_root, entries, cached)
    charged = sum(e["size"] for e in requests["archives"] if not e["cached"])
    if charged > budget:
        raise RuntimeError("Internal acquisition budget violation")
    bindings.update(requests["log_hashes"])
    for name in ("download/prepare_selector_dataset.py", "download/plan_drivemem_images.py",
                 "download/fetch_drivemem_budget_images.py", "download/prepare_drivemem_navtrain.py", "navsim/agents/drivemem.py"):
        bindings[str(REPO / name)] = sha(REPO / name)
    new_train = [r for r in records if r["role"] == "train"]
    summary = dict(status="planned", image_archives_downloaded=False, additional_budget_GB=args.max_extra_gb,
        additional_compressed_GB=charged / 1e9, cached_shards=sorted(cached),
        selected_archives=[dict(shard=e["shard"], cached=e["cached"], compressed_GB=e["size"] / 1e9) for e in requests["archives"]],
        legacy_train_scenes=len(legacy), legacy_train_sessions=len(train_sessions),
        added_train_scenes=len(new_train), total_train_scenes=len(legacy) + len(new_train),
        total_train_sessions=len(train_sessions | {r["session_id"] for r in new_train}),
        new_training_sessions=len(set(fresh) - holdout), fresh_holdout_sessions=len(holdout),
        fresh_holdout_scenes=sum(r["role"] == "holdout" for r in records),
        target_new_sessions=args.target_new_sessions, target_met=len(fresh) >= args.target_new_sessions,
        new_sessions_shortfall=max(0, args.target_new_sessions - len(fresh)),
        complete_rgb_candidates_without_extraction=len(free_tokens),
        cached_new_eligible_sessions=sum(n >= args.min_scenes for sid, n in initial.items() if sid not in counts),
        required_images=len(requests["images"]), existing_images=len(requests["initial_images"]),
        future_training_candidate_evaluations=16 * len(new_train), future_raw_filtered_calls=32 * len(new_train),
        historical_role_conflicts=conflicts, protected_sessions=len(excluded),
        train_map_counts=dict(Counter(r.get("map_name", "unknown") for r in new_train)),
        note="Metadata/cost-selected, not representative. Greedy multistart heuristic, not global optimum. "
             "Cached archives credited by size for planning; SHA256 checked before extraction. "
             "Cap covers unique additional compressed files, excluding protocol overhead. "
             "Fresh holdout is session-separated for the selector; base-pretraining independence unverified. "
             "Images/manifests only: new features and top16 labels must still be collected.")
    write(out / "selection.json", dict(records=records, legacy_training_records=legacy,
                excluded_sessions=sorted(excluded), fresh_holdout_sessions=sorted(holdout)))
    write(out / "image_requests.json", requests)
    write(out / "capacity_report.json", summary)
    artifacts = {p: sha(out / p) for p in ("selection.json", "image_requests.json", "capacity_report.json", "metadata/mapping.json", "metadata/archives.json")}
    result = dict(schema=1, options=options(args), additional_budget_bytes=budget, inputs=bindings, artifacts=artifacts, summary=summary)
    write(out / "acquisition_plan.json", result)
    print(json.dumps(summary, indent=2), flush=True)
    return result


def image_status(out, requests):
    expected = set(requests["images"])
    committed = dict(requests["initial_images"])
    for entry in requests["archives"]:
        receipt = out / "receipts" / (Path(entry["path"]).name + ".json")
        if receipt.exists():
            saved = read(receipt)
            if saved["archive_sha256"] != entry["sha256"] or set(saved["images"]) != set(entry["images"]):
                raise RuntimeError("Image receipt changed")
            committed.update(saved["images"])
    for number, (relative, info) in enumerate(committed.items(), 1):
        path = checked_target(Path(requests["sensor_root"]), relative)
        if not path.is_file() or path.stat().st_size != info["size"] or sha(path) != info["sha256"]:
            raise RuntimeError("Committed image missing/changed: " + str(path))
        if number % 2000 == 0:
            print(f"VERIFY IMAGES {number}/{len(committed)}", flush=True)
    if set(committed) - expected:
        raise ValueError("Unexpected installed image")
    missing = sorted(expected - set(committed))
    return committed, dict(status="complete" if not missing else "partial", required_images=len(expected),
        verified_images=len(committed), missing_images=len(missing), missing_examples=missing[:8],
        note="Image preparation only; no new labels/features or training executed.")


def fetch(out, requests, archive_dir):
    image_status(out, requests)  # Validate all pre-existing/committed images before transfers.
    pending = [e for e in requests["archives"] if not (out / "receipts" / (Path(e["path"]).name + ".json")).exists()]
    # A vanished cached archive must NOT silently turn into extra network transfer.
    for e in pending:
        if e["cached"]:
            path = archive_dir / Path(e["path"]).name
            print("VERIFY CACHED " + str(path), flush=True)
            if not path.is_file() or path.stat().st_size != e["size"] or sha(path) != e["sha256"]:
                raise RuntimeError("Planned cached archive is missing/corrupt; no additional downloads started: " + str(path))
    for e in sorted(pending, key=lambda e: (not e["cached"], e["shard"])):
        archive = archive_dir / Path(e["path"]).name if e["cached"] else download(e, archive_dir)
        completed = extract_images(archive, Path(requests["sensor_root"]), set(e["images"]))
        if set(completed) != set(e["images"]):
            raise RuntimeError("Archive lacks requested images; archive preserved")
        write(out / "receipts" / (Path(e["path"]).name + ".json"), dict(archive_sha256=e["sha256"], images=completed))
        print(f"INSTALLED shard={e['shard']} images={len(completed)}; archive retained for reuse", flush=True)
    committed, report = image_status(out, requests)
    write(out / "installed_images.json", committed)
    write(out / "image_status.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path("/workspace/exp_v2/drivemem_memory")
    p.add_argument("stage", choices=("plan", "fetch", "status"))
    p.add_argument("--source", type=Path, default=base / "budget20gb_v1")
    p.add_argument("--addition", type=Path, default=base / "mf_new4_shard174_v1")
    p.add_argument("--inventory", type=Path, default=base / "pilot_v1")
    p.add_argument("--validation", type=Path, default=base / "validation_cached4_v1")
    p.add_argument("--exclude-run", type=Path, action="append", default=[])
    p.add_argument("--archive-dir", type=Path, default=Path("/workspace/exp_v2/drivemem_data/.downloads/drivemem_camera"))
    p.add_argument("--out", type=Path, default=base / "selector_data_expansion_v2")
    p.add_argument("--max-extra-gb", type=float, default=30.)
    p.add_argument("--target-new-sessions", type=int, default=16)
    p.add_argument("--holdout-sessions", type=int, default=4)
    p.add_argument("--min-scenes", type=int, default=16)
    p.add_argument("--max-scenes", type=int, default=64)
    p.add_argument("--gap-s", type=float, default=10.)
    p.add_argument("--seed", type=int, default=20261109)
    a = p.parse_args()
    for k, v in vars(a).items():
        if isinstance(v, Path):
            setattr(a, k, v.resolve())
    a.exclude_run = [x.resolve() for x in a.exclude_run]
    if (not np.isfinite(a.max_extra_gb) or not 0 < a.max_extra_gb <= 30
            or a.holdout_sessions < 1 or a.target_new_sessions <= a.holdout_sessions
            or not 1 <= a.min_scenes <= a.max_scenes or not np.isfinite(a.gap_s) or a.gap_s < 10):
        raise ValueError("Require <=30 GB additional, positive holdout, target>holdout, valid quotas, gap>=10s")
    if a.stage == "plan":
        plan(a)
    else:
        _, requests = verify(a.out, a)
        if a.stage == "fetch":
            fetch(a.out, requests, a.archive_dir)
        else:
            _, report = image_status(a.out, requests)
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
