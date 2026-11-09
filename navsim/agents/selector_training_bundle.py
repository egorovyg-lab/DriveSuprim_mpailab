"""Immutable train-only union of legacy functional features and new collection.

NumPy only. No model/scorer calls and no holdout features/outcomes. Training
inputs and measured labels have separate fields in the bundle reader.
"""
from collections import Counter, defaultdict
from pathlib import Path
import platform
import re

import numpy as np

from navsim.agents.drivemem import METRICS, SAFETY
from navsim.agents.drivemem_action_transfer import pool
from navsim.agents.drivemem_decoder_utility import load_decoder_inputs
from navsim.planning.script import run_selector_collection as collection
from navsim.planning.script.run_drivemem_expanded_utility import REPO, read, write, save as save_npz, sha

INPUT_KEYS = ("candidate_ids", "selected_local", "selection_scores", "scene_embedding", "ego_status", "decoder_states") + tuple(
    "head_" + k for k in (*METRICS, "imi"))
LABEL_KEYS = ("pdm_score",) + METRICS
IMPROVEMENT_MIN, TIE_EPSILON = .005, .001


def save(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(path, **values)


def example(sample, outcomes, vocab, metadata):
    """Explicit feature allowlist: measured outcomes cannot become network inputs."""
    inputs = {k: np.asarray(sample[k]).copy() for k in INPUT_KEYS}
    ids, scores, b = inputs["candidate_ids"], inputs["selection_scores"], int(inputs["selected_local"])
    if (ids.shape != (256,) or ids.dtype.kind not in "iu" or len(set(ids.tolist())) != 256
            or ids.min() < 0 or ids.max() >= len(vocab) or scores.shape != (256,)
            or b != int(np.argmax(scores)) or inputs["decoder_states"].shape != (256, 256)):
        raise ValueError("Invalid frozen candidate/decoder alignment")
    if inputs["scene_embedding"].ndim != 1 or inputs["ego_status"].ndim != 1:
        raise ValueError("Invalid context shape")
    if any(inputs["head_" + k].shape != (256,) for k in (*METRICS, "imi")):
        raise ValueError("Invalid metric head shape")
    if not all(np.isfinite(a).all() for a in inputs.values()):
        raise ValueError("Nonfinite input")
    indices = pool(inputs)
    if b not in indices or any(str(int(i)) not in outcomes for i in indices):
        raise ValueError("Complete top16 outcomes including baseline required")
    labels = {}
    for version in ("raw", "filtered"):
        a = np.asarray([[outcomes[str(int(i))][version][k] for k in LABEL_KEYS] for i in indices], dtype=np.float64)
        if not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
            raise ValueError("Invalid stored PDM/component labels")
        labels[version] = a
    if metadata["role"] != "train":
        raise ValueError("Bundle accepts training examples only")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", metadata["token"]):
        raise ValueError("Unsafe scene token")
    return dict(sample=inputs, label_indices=indices.astype(np.int64), label_raw=labels["raw"],
                label_filtered=labels["filtered"], metadata=dict(metadata))


def merge_examples(legacy, new, expected_tokens, protected, expected_sessions):
    examples = legacy + new
    tokens = [e["metadata"]["token"] for e in examples]
    if len(set(tokens)) != len(tokens) or set(tokens) != set(expected_tokens):
        raise ValueError("Duplicate/missing/unexpected union scene tokens")
    sessions = {e["metadata"]["session_id"] for e in examples}
    if sessions & set(protected):
        raise ValueError("Training union intersects protected/holdout sessions")
    if len(sessions) != expected_sessions:
        raise ValueError("Union session count disagrees with acquisition")
    shapes = {tuple((k, e["sample"][k].shape) for k in INPUT_KEYS) for e in examples}
    if len(shapes) != 1:
        raise ValueError("Legacy/new feature dimensions differ")
    return sorted(examples, key=lambda e: (e["metadata"]["session_id"], e["metadata"]["token"]))


def load_sources(prior, addition, decoder, new_root):
    data = load_decoder_inputs(prior, addition, decoder)
    bank, fit, targets, vocab, source, _, _, bindings, _, _ = data
    bindings = dict(bindings)
    identity, records = collection.verify_collection(new_root)
    acq_root = Path(identity["acquisition_root"])
    plan, _ = collection.acquisition.verify(acq_root)
    selection = read(acq_root / "selection.json")
    legacy_metadata = {r["token"]: r for r in selection["legacy_training_records"]}
    new_metadata = {r["token"]: r for r in records}
    if set(legacy_metadata) != {s["token"] for s in bank + fit}:
        raise ValueError("Acquisition legacy membership differs from accepted M/F")
    if set(new_metadata) != {r["token"] for r in selection["records"] if r["role"] == "train"}:
        raise ValueError("Acquisition/new collector train membership differs")
    old_identity = read(source / "identity.json")
    new_config = read(new_root / "source_config.json")
    keys = ("agent", "scorer", "simulator", "proposal_sampling", "traffic_agents", "traffic_agents_policy", "train_test_split",
            "navsim_log_path", "original_sensor_path", "map_root", "checkpoint_sha256")
    if any(old_identity[k] != new_config[k] for k in keys):
        raise ValueError("Legacy/new teacher/scorer configuration mismatch")
    if read(decoder / "identity.json")["numerical_profile"] != identity["numerical_profile"]:
        raise ValueError("Legacy/new declared numerical profiles differ")
    new_vocab = collection.arrays(new_root / "vocab.npz")["vocab"]
    if not np.array_equal(vocab, new_vocab):
        raise ValueError("Legacy/new vocabularies differ")
    features = collection.checked_features(new_root, records)
    collection.checked_index(new_root, "", "cache_index.json", records)
    support = read(new_root / "support_report.json")
    if (support["status"] != "complete" or support["labelled_scenes"] != len(records)
            or len(features) != len(records) or support["holdout_processed"] != 0
            or set(support["labels"]) != {r["token"] + ".json" for r in records}):
        raise ValueError("Complete train-only new collection required")
    label_identity = dict(collection_identity_sha256=sha(new_root / "identity.json"), export_index_sha256=sha(new_root / "export_index.json"),
                          cache_index_sha256=sha(new_root / "cache_index.json"), vocab_sha256=sha(new_root / "vocab.npz"))
    if read(new_root / "label_identity.json") != label_identity:
        raise RuntimeError("Frozen new label identity changed")
    def bind(path, expected=None):
        value = sha(path)
        if expected is not None and value != expected:
            raise RuntimeError("Frozen union input changed: " + str(path))
        bindings[str(path)] = value
    for name in ("identity.json", "source_config.json", "splits.json", "split_identity.json", "inventory.csv", "runtime.json",
                 "export_index.json", "native_index.json", "cache_index.json", "vocab.npz", "vocab_identity.json", "label_identity.json", "support_report.json"):
        bind(new_root / name)
    bind(acq_root / "acquisition_plan.json")
    bind(acq_root / "selection.json")
    addition_tokens = {r["token"] for r in read(addition / "label_selection.json")["records"]}
    legacy, new = [], []
    for s in bank + fit:
        r = legacy_metadata[s["token"]]
        if r["session_id"] != s["session_id"] or r["role"] != s["role"]:
            raise ValueError("Legacy scene/session/role mismatch")
        origin_root = addition if s["token"] in addition_tokens else source
        bind(origin_root / "features" / (s["token"] + ".npz"))
        bind(decoder / "features" / origin_root.name / (s["token"] + ".npz"))
        legacy.append(example(s, targets[s["token"]], vocab, dict(token=s["token"], session_id=s["session_id"],
            role="train", origin="legacy", source_role=s["role"], map_name=r.get("map_name", "unknown"),
            feature_protocol="accepted_legacy_functional")))
    for r in records:
        s = collection.load_sample(new_root, r, features)
        path = new_root / "top16_labels" / (r["token"] + ".json")
        bind(path, support["labels"][path.name])
        state = read(path)
        ix = pool(s).tolist()
        if (any(state[k] != r[k] for k in ("token", "session_id", "role"))
                or state["top16"] != ix or state["candidate_ids"] != s["candidate_ids"][ix].tolist()
                or state["feature_sha256"] != features[r["token"]]["sha256"]):
            raise ValueError("New label pool/provenance mismatch")
        bind(new_root / "features" / (r["token"] + ".npz"), features[r["token"]]["sha256"])
        bind(new_root / "native" / (r["token"] + ".npz"), features[r["token"]]["native_sha256"])
        new.append(example(s, state["outcomes"], vocab, dict(token=r["token"], session_id=r["session_id"],
            role="train", origin="new", source_role="train", map_name=r.get("map_name", "unknown"),
            feature_protocol="prospective_native_replay_v2")))
    protected = set(selection["excluded_sessions"]) | set(selection["fresh_holdout_sessions"])
    merged = merge_examples(legacy, new, set(legacy_metadata) | set(new_metadata), protected, plan["summary"]["total_train_sessions"])
    return merged, vocab, bindings, protected


def audit(examples):
    scene_rows, events = [], []
    for e in examples:
        m, raw, ix = e["metadata"], e["label_raw"], e["label_indices"]
        base = list(ix).index(int(e["sample"]["selected_local"]))
        quality, components = raw[:, 0], raw[:, 1:]
        gain = quality - quality[base]
        harm = np.any(components[:, SAFETY] + 1e-8 < components[base, list(SAFETY)], axis=1)
        useful = (gain >= IMPROVEMENT_MIN) & ~harm
        scene_rows.append(dict(**m, baseline=float(quality[base]),
            oracle_gain_pp=100 * float(gain.max()), relative_safe_oracle_gain_pp=100 * float(gain[~harm].max()),
            useful_alternatives=int(useful.sum()), harmful_alternatives=int(harm.sum()),
            meaningful_ranking=bool(np.ptp(quality) > TIE_EPSILON), worst_regret_pp=100 * float(np.ptp(quality))))
        for j in SAFETY:
            for i in range(16):
                if i != base:
                    events.append(dict(metric=METRICS[j], token=m["token"], session_id=m["session_id"], origin=m["origin"],
                        below_one=bool(components[i, j] < 1 - 1e-8),
                        degraded=bool(components[i, j] < components[base, j] - 1e-8),
                        improved=bool(components[i, j] > components[base, j] + 1e-8)))
    def section(rows):
        sessions = defaultdict(list)
        for r in rows:
            sessions[r["session_id"]].append(r)
        return dict(scenes=len(rows), sessions=len(sessions),
            baseline_mean=float(np.mean([r["baseline"] for r in rows])),
            oracle_gain_micro_pp=float(np.mean([r["oracle_gain_pp"] for r in rows])),
            oracle_gain_macro_pp=float(np.mean([np.mean([r["oracle_gain_pp"] for r in group]) for group in sessions.values()])),
            relative_safe_oracle_gain_micro_pp=float(np.mean([r["relative_safe_oracle_gain_pp"] for r in rows])),
            scenes_with_useful_alternative=sum(r["useful_alternatives"] > 0 for r in rows),
            sessions_with_useful_alternative=len({r["session_id"] for r in rows if r["useful_alternatives"]}),
            meaningful_ranking_scenes=sum(r["meaningful_ranking"] for r in rows))
    result = dict(status="complete", **section(scene_rows),
        origins={origin: section([r for r in scene_rows if r["origin"] == origin]) for origin in sorted({r["origin"] for r in scene_rows})},
        safety_support={}, improvement_min=IMPROVEMENT_MIN, tie_epsilon=TIE_EPSILON,
        scenes_per_session=dict(Counter(r["session_id"] for r in scene_rows)),
        map_scenes=dict(Counter(r["map_name"] for r in scene_rows)),
        note="Train-label audit only. Candidate rows correlated. Relative-safe means no component worse than baseline, "
             "not absolute safety. Local PDM, not EPDMS. Old/new feature origins retained; origin session counts overlap. Fresh holdout unopened.")
    for metric in (METRICS[j] for j in SAFETY):
        group = [r for r in events if r["metric"] == metric]
        result["safety_support"][metric] = dict(actions=len(group), degraded=sum(r["degraded"] for r in group),
            improved=sum(r["improved"] for r in group), below_one=sum(r["below_one"] for r in group),
            degradation_scenes=len({r["token"] for r in group if r["degraded"]}),
            degradation_sessions=len({r["session_id"] for r in group if r["degraded"]}),
            improvement_sessions=len({r["session_id"] for r in group if r["improved"]}))
    return result, scene_rows


def build(out, examples, vocab, inputs, protected):
    sources = {name: sha(REPO / name) for name in (
        "navsim/agents/selector_training_bundle.py", "navsim/planning/script/run_selector_bundle.py")}
    identity = dict(schema=1, inputs=inputs, sources=sources, python=platform.python_version(), numpy=np.__version__,
                    counts=dict(scenes=len(examples), sessions=len({e["metadata"]["session_id"] for e in examples})),
                    protected_sessions=sorted(protected), protocol="train_only_union_raw_labels_v2")
    if (out / "identity.json").exists():
        if read(out / "identity.json") != identity:
            raise RuntimeError("Union inputs/code changed; use a new bundle root")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Nonempty bundle output without identity")
    write(out / "identity.json", identity)
    index = read(out / "examples_index.json") if (out / "examples_index.json").exists() else {}
    if set(index) - {e["metadata"]["token"] for e in examples}:
        raise ValueError("Unexpected bundled scenes")
    for n, e in enumerate(examples, 1):
        token = e["metadata"]["token"]
        path = out / "examples" / (token + ".npz")
        if token in index:
            if sha(path) != index[token]["sha256"] or index[token]["metadata"] != e["metadata"]:
                raise RuntimeError("Committed bundle example changed")
        else:
            save(path, **e["sample"], label_indices=e["label_indices"], label_raw=e["label_raw"], label_filtered=e["label_filtered"])
            index[token] = dict(sha256=sha(path), metadata=e["metadata"])
            write(out / "examples_index.json", index)
        if n % 100 == 0 or n == len(examples):
            print(f"BUNDLE {n}/{len(examples)}", flush=True)
    if (out / "vocab.npz").exists():
        with np.load(out / "vocab.npz", allow_pickle=False) as f:
            if not np.array_equal(vocab, f["vocab"]):
                raise RuntimeError("Bundled vocabulary changed")
    else:
        save(out / "vocab.npz", vocab=vocab)
    write(out / "bundle_manifest.json", dict(identity_sha256=sha(out / "identity.json"), index_sha256=sha(out / "examples_index.json"),
            vocab_sha256=sha(out / "vocab.npz"), counts=identity["counts"], label_keys=list(LABEL_KEYS), input_keys=list(INPUT_KEYS)))
    return identity


def load_bundle(out):
    manifest, identity = read(out / "bundle_manifest.json"), read(out / "identity.json")
    if any(sha(out / name) != manifest[key] for name, key in (
            ("identity.json", "identity_sha256"), ("examples_index.json", "index_sha256"), ("vocab.npz", "vocab_sha256"))):
        raise RuntimeError("Frozen bundle manifest changed")
    for name, digest in identity["sources"].items():
        if sha(REPO / name) != digest:
            raise RuntimeError("Frozen bundle reader changed")
    for name, digest in identity["inputs"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError("Frozen upstream bundle input changed: " + name)
    with np.load(out / "vocab.npz", allow_pickle=False) as f:
        vocab = f["vocab"].copy()
    examples = []
    for token, entry in read(out / "examples_index.json").items():
        path = out / "examples" / (token + ".npz")
        if sha(path) != entry["sha256"] or entry["metadata"]["token"] != token:
            raise RuntimeError("Bundled example changed")
        with np.load(path, allow_pickle=False) as f:
            sample = {k: f[k].copy() for k in INPUT_KEYS}
            examples.append(dict(sample=sample, label_indices=f["label_indices"].copy(), label_raw=f["label_raw"].copy(),
                                 label_filtered=f["label_filtered"].copy(), metadata=entry["metadata"]))
    if len(examples) != identity["counts"]["scenes"] or manifest["counts"] != identity["counts"]:
        raise RuntimeError("Frozen bundle scene counts changed")
    return merge_examples(examples, [], [e["metadata"]["token"] for e in examples], identity["protected_sessions"], identity["counts"]["sessions"]), vocab
