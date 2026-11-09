"""Second image expansion: <=30 GB NEW archives, anchored on the full bundle.

Reuses frozen archive verification/extraction and collector manifests. Planning
uses metadata only; the existing four holdout sessions remain protected.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_selector_dataset as original

read, write, sha = original.read, original.write, original.sha


def parent_snapshot(previous_data, previous_bundle):
    previous, _ = original.verify(previous_data)
    selection = read(previous_data / "selection.json")
    identity, manifest = read(previous_bundle / "identity.json"), read(previous_bundle / "bundle_manifest.json")
    for name, key in (("identity.json", "identity_sha256"), ("examples_index.json", "index_sha256")):
        if sha(previous_bundle / name) != manifest[key]:
            raise RuntimeError("Parent bundle metadata changed")
    if identity["inputs"].get(str(previous_data / "selection.json")) != sha(previous_data / "selection.json"):
        raise RuntimeError("Parent selection/bundle mismatch")
    index = read(previous_bundle / "examples_index.json")
    anchors = selection["legacy_training_records"] + [r for r in selection["records"] if r["role"] == "train"]
    if len(anchors) != len(index) or {r["token"] for r in anchors} != set(index):
        raise ValueError("Full parent bundle must be used as spacing anchors")
    for r in anchors:
        m = index[r["token"]]["metadata"]
        if m["role"] != "train" or m["session_id"] != r["session_id"]:
            raise ValueError("Parent scene/session membership mismatch")
    protected = set(identity["protected_sessions"]) | set(selection["excluded_sessions"]) | set(selection["fresh_holdout_sessions"])
    if {r["session_id"] for r in anchors} & protected:
        raise ValueError("Parent training/protected session overlap")
    bindings = dict(previous["inputs"])
    for root, names in ((previous_data, ("acquisition_plan.json", "selection.json")),
                        (previous_bundle, ("bundle_manifest.json", "identity.json", "examples_index.json"))):
        bindings.update({str(root / name): sha(root / name) for name in names})
    return previous, anchors, protected, selection["fresh_holdout_sessions"], bindings


def freeze(rows, active, existing, minimum, cap, gap_s):
    groups = defaultdict(list)
    for r in rows:
        if r["shard"] in active:
            groups[r["session_id"]].append(r)
    records = []
    for sid, group in sorted(groups.items()):
        selected, last = [], None
        for r in sorted(group, key=lambda r: (int(r["timestamp_us"]), r["token"])):
            t = int(r["timestamp_us"])
            if len(selected) >= max(0, cap - existing.get(sid, 0)):
                break
            if last is None or t - last >= round(gap_s * 1e6):
                selected.append(dict(r, role="train", session_origin="existing_training" if sid in existing else "new_session"))
                last = t
        if sid in existing or len(selected) >= minimum:
            records.extend(selected)
    if not records:
        raise ValueError("No additional training scenes under this budget; no download started")
    return records


def verify(out, args=None):
    plan, requests = original.verify(out)
    if plan.get("protocol") != "parent_bundle_anchored_expansion30_v1":
        raise ValueError("Wrong expansion protocol")
    if args is not None:
        expected = original.options(args)
        if any(plan["options"][k] != value for k, value in expected.items()):
            raise RuntimeError("Expansion options changed; use a new --out root")
    return plan, requests


def plan(args):
    if not 0 < args.max_extra_gb <= 30 or args.min_scenes < 1 or args.max_scenes < args.min_scenes or args.gap_s < 10:
        raise ValueError("Require 0 < additional GB <= 30, valid quotas and gap >= 10s")
    if (args.out / "acquisition_plan.json").exists():
        result, _ = verify(args.out, args)
        print(json.dumps(result["summary"], indent=2), flush=True)
        return result
    previous, anchors, protected, holdout, bindings = parent_snapshot(args.previous_data, args.previous_bundle)
    source_roots = tuple(Path(previous["options"][k]) for k in ("source", "addition", "inventory", "validation"))
    for root in (args.previous_data, args.previous_bundle, args.archive_dir, *source_roots):
        if root == args.out or root in args.out.parents or args.out in root.parents:
            raise ValueError("Separate new acquisition root required")
    if args.out.exists() and any(p.name != "metadata" for p in args.out.iterdir()):
        raise ValueError("Nonempty output without plan; preserve it and use a new root")
    # Reuse exact cached manifests. No network is needed for planning.
    for name in ("mapping.json", "archives.json"):
        source = args.previous_data / "metadata" / name
        if not source.is_file():
            raise FileNotFoundError("Cached metadata required; no download is started")
        destination = args.out / "metadata" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha(destination) != sha(source):
            raise RuntimeError("New metadata cache differs from parent")
        if not destination.exists():
            shutil.copyfile(source, destination)
    reverse, entries = original.metadata(args.out)
    source = Path(previous["options"]["source"])
    template = read(source / "identity.json")
    logs_path, sensor_root = Path(template["navsim_log_path"]), Path(template["original_sensor_path"])
    installed = {p.stem for p in logs_path.glob("*.pkl")}
    inventory = Path(previous["options"]["inventory"]) / "inventory.csv"
    with inventory.open(encoding="utf-8", newline="") as stream:
        rows = original.candidate_rows(list(csv.DictReader(stream)), anchors, protected, installed, reverse, args.gap_s)
    cached = {i for i, e in entries.items() if (args.archive_dir / Path(e["path"]).name).is_file()
              and (args.archive_dir / Path(e["path"]).name).stat().st_size == e["size"]}
    existing = Counter(r["session_id"] for r in anchors)
    budget = int(args.max_extra_gb * 1e9)
    # Existing deterministic multistart greedy solver evaluates UNION coverage.
    # No holdout reassignment: all protected sessions were removed above.
    target = max(1, len({r["session_id"] for r in rows if r["session_id"] not in existing}))
    chosen, _, _ = original.acquire(rows, entries, cached, set(), existing, budget,
                                    args.min_scenes, args.max_scenes, target, args.gap_s)
    records = freeze(rows, cached | chosen, existing, args.min_scenes, args.max_scenes, args.gap_s)
    if {r["session_id"] for r in records} & protected:
        raise ValueError("New selection intersects protected sessions")
    requests = original.request_images(records, logs_path, sensor_root, entries, cached)
    charged = sum(e["size"] for e in requests["archives"] if not e["cached"])
    if charged > budget:
        raise RuntimeError("Additional archive budget exceeded")
    bindings.update(requests["log_hashes"])
    for path in (source / "identity.json", inventory, Path(__file__).resolve(),
                 REPO / "download/prepare_selector_dataset.py", REPO / "download/plan_drivemem_images.py",
                 REPO / "download/fetch_drivemem_budget_images.py", REPO / "navsim/agents/drivemem.py"):
        bindings[str(path)] = sha(path)
    new_sessions = {r["session_id"] for r in records} - set(existing)
    summary = dict(status="planned", additional_budget_GB=args.max_extra_gb, additional_compressed_GB=charged / 1e9,
        image_archives_downloaded=False, legacy_train_scenes=len(anchors), legacy_train_sessions=len(existing),
        added_train_scenes=len(records), total_train_scenes=len(anchors) + len(records),
        total_train_sessions=len(existing) + len(new_sessions), new_training_sessions=len(new_sessions),
        existing_session_extra_scenes=sum(r["session_id"] in existing for r in records),
        fresh_holdout_sessions=len(holdout), fresh_holdout_scenes=0, fresh_holdout_processed=0,
        cached_shards=sorted(cached),
        selected_archives=[dict(shard=e["shard"], cached=e["cached"], compressed_GB=e["size"] / 1e9) for e in requests["archives"]],
        required_images=len(requests["images"]), existing_images=len(requests["initial_images"]),
        future_training_candidate_evaluations=16 * len(records), future_raw_filtered_calls=32 * len(records),
        added_scenes_per_session=dict(Counter(r["session_id"] for r in records)),
        train_map_counts=dict(Counter(r.get("map_name", "unknown") for r in records)),
        note="Second additional-transfer budget, <=30 GB compressed unique archives excluding HTTP overhead. "
             "Multistart greedy heuristic prioritizes new eligible sessions; not proven globally optimal. "
             "Metadata-selected earliest feasible times; no outcomes used. Existing fresh holdout preserved "
             "and not downloaded again. Cached archives credited by size only; SHA checked before extraction. "
             "No features, labels or training yet; archive-selected sample is not representative.")
    write(args.out / "selection.json", dict(records=records, legacy_training_records=anchors,
          excluded_sessions=sorted(protected), fresh_holdout_sessions=sorted(holdout)))
    write(args.out / "image_requests.json", requests)
    write(args.out / "capacity_report.json", summary)
    options = original.options(args)
    # The unchanged collector reads these original source locations.
    options.update({k: previous["options"][k] for k in ("source", "addition", "inventory", "validation")})
    artifacts = {name: sha(args.out / name) for name in (
        "selection.json", "image_requests.json", "capacity_report.json", "metadata/mapping.json", "metadata/archives.json")}
    result = dict(schema=1, protocol="parent_bundle_anchored_expansion30_v1", options=options,
                  additional_budget_bytes=budget, inputs=bindings, artifacts=artifacts, summary=summary)
    write(args.out / "acquisition_plan.json", result)
    print(json.dumps(summary, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path("/workspace/exp_v2/drivemem_memory")
    p.add_argument("stage", choices=("plan", "fetch", "status"))
    p.add_argument("--previous-data", type=Path, default=base / "selector_data_expansion_v2")
    p.add_argument("--previous-bundle", type=Path, default=base / "selector_training_bundle_v2")
    p.add_argument("--archive-dir", type=Path, default=Path("/workspace/exp_v2/drivemem_data/.downloads/drivemem_camera"))
    p.add_argument("--out", type=Path, default=base / "selector_data_expansion_v3_30gb")
    p.add_argument("--max-extra-gb", type=float, default=30.)
    p.add_argument("--min-scenes", type=int, default=16)
    p.add_argument("--max-scenes", type=int, default=128)
    p.add_argument("--gap-s", type=float, default=10.)
    a = p.parse_args()
    for k, v in vars(a).items():
        if isinstance(v, Path):
            setattr(a, k, v.resolve())
    if a.stage == "plan":
        plan(a)
    else:
        _, requests = verify(a.out, a)
        if a.stage == "fetch":
            original.fetch(a.out, requests, a.archive_dir)
        else:
            _, report = original.image_status(a.out, requests)
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
