"""Append a completed train-only collection to an immutable parent bundle."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from navsim.agents import selector_training_bundle as bundle
from navsim.planning.script import run_selector_collection as collection
from navsim.planning.script.run_drivemem_expanded_utility import REPO, read, write, sha


def load_sources(parent, new_root):
    old, vocab = bundle.load_bundle(parent)
    parent_identity = read(parent / "identity.json")
    identity, records = collection.verify_collection(new_root)
    acquisition = Path(identity["acquisition_root"])
    plan, _ = collection.acquisition.verify(acquisition)
    if plan.get("protocol") != "parent_bundle_anchored_expansion30_v1" or Path(plan["options"]["previous_bundle"]) != parent:
        raise ValueError("Collection must extend this exact parent bundle")
    selection = read(acquisition / "selection.json")
    if ({r["token"] for r in selection["legacy_training_records"]} != {e["metadata"]["token"] for e in old}
            or len(selection["legacy_training_records"]) != len(old)):
        raise ValueError("Parent training membership changed")
    if (not records or any(r["role"] != "train" for r in records)
            or {r["token"] for r in records} != {r["token"] for r in selection["records"]}
            or len(records) != len(selection["records"])):
        raise ValueError("Train-only acquisition/collection membership mismatch")
    source = Path(plan["options"]["source"])
    template, config = read(source / "identity.json"), read(new_root / "source_config.json")
    keys = ("agent", "scorer", "simulator", "proposal_sampling", "traffic_agents", "traffic_agents_policy", "train_test_split",
            "navsim_log_path", "original_sensor_path", "map_root", "checkpoint_sha256")
    if any(template[k] != config[k] for k in keys):
        raise ValueError("Parent/new teacher or scorer configuration mismatch")
    if identity["numerical_profile"] != collection.PROFILE:
        raise ValueError("Numerical profile mismatch")
    if not np.array_equal(vocab, collection.arrays(new_root / "vocab.npz")["vocab"]):
        raise ValueError("Parent/new vocabulary mismatch")
    features = collection.checked_features(new_root, records)
    cache = collection.checked_index(new_root, "", "cache_index.json", records)
    support = read(new_root / "support_report.json")
    if (support["status"] != "complete" or support["holdout_processed"] != 0 or support["labelled_scenes"] != len(records)
            or len(features) != len(records) or len(cache) != len(records)
            or set(support["labels"]) != {r["token"] + ".json" for r in records}):
        raise ValueError("Complete train-only collection required")
    spec = dict(collection_identity_sha256=sha(new_root / "identity.json"), export_index_sha256=sha(new_root / "export_index.json"),
                cache_index_sha256=sha(new_root / "cache_index.json"), vocab_sha256=sha(new_root / "vocab.npz"))
    if read(new_root / "label_identity.json") != spec:
        raise RuntimeError("Frozen new label identity changed")
    bindings = dict(parent_identity["inputs"])
    def bind(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise RuntimeError("Frozen extension input changed: " + str(path))
        if str(path) in bindings and bindings[str(path)] != digest:
            raise RuntimeError("Parent binding changed")
        bindings[str(path)] = digest
    for name in ("bundle_manifest.json", "identity.json", "examples_index.json", "vocab.npz"):
        bind(parent / name)
    for name in ("identity.json", "source_config.json", "splits.json", "split_identity.json", "inventory.csv", "runtime.json",
                 "export_index.json", "native_index.json", "cache_index.json", "vocab.npz", "vocab_identity.json", "label_identity.json", "support_report.json"):
        bind(new_root / name)
    for name in ("acquisition_plan.json", "selection.json"):
        bind(acquisition / name)
    bind(Path(__file__).resolve())
    protected = set(parent_identity["protected_sessions"]) | set(selection["excluded_sessions"]) | set(selection["fresh_holdout_sessions"])
    new = []
    for r in records:
        s = collection.load_sample(new_root, r, features)
        path = new_root / "top16_labels" / (r["token"] + ".json")
        bind(path, support["labels"][path.name])
        state = read(path)
        ix = np.argsort(-s["selection_scores"], kind="stable")[:16].tolist()
        if (any(state[k] != r[k] for k in ("token", "session_id", "role")) or state["top16"] != ix
                or state["candidate_ids"] != s["candidate_ids"][ix].tolist()
                or state["feature_sha256"] != features[r["token"]]["sha256"]):
            raise ValueError("New label pool/provenance mismatch")
        bind(new_root / "features" / (r["token"] + ".npz"), features[r["token"]]["sha256"])
        bind(new_root / "native" / (r["token"] + ".npz"), features[r["token"]]["native_sha256"])
        new.append(bundle.example(s, state["outcomes"], vocab, dict(token=r["token"], session_id=r["session_id"], role="train",
              origin="expansion30", source_role="train", map_name=r.get("map_name", "unknown"), feature_protocol="prospective_native_replay_v2")))
    expected = {e["metadata"]["token"] for e in old} | {r["token"] for r in records}
    merged = bundle.merge_examples(old, new, expected, protected, plan["summary"]["total_train_sessions"])
    if len(merged) != plan["summary"]["total_train_scenes"]:
        raise ValueError("Expanded scene count disagrees with acquisition")
    return merged, vocab, bindings, protected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory"
    p.add_argument("stage", choices=("build", "audit"))
    p.add_argument("--parent", type=Path, default=base / "selector_training_bundle_v2")
    p.add_argument("--collection", type=Path, default=base / "selector_collection_v3_30gb")
    p.add_argument("--out", type=Path, default=base / "selector_training_bundle_v3_30gb")
    a = p.parse_args()
    parent, new_root, out = [x.resolve() for x in (a.parent, a.collection, a.out)]
    for root in (parent, new_root):
        if out == root or out in root.parents or root in out.parents:
            raise ValueError("Separate expanded bundle output required")
    if a.stage == "build":
        examples, vocab, inputs, protected = load_sources(parent, new_root)
        bundle.build(out, examples, vocab, inputs, protected)
    examples, _ = bundle.load_bundle(out)
    report, rows = bundle.audit(examples)
    write(out / "audit.json", report)
    write(out / "audit_per_scene.json", rows)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
