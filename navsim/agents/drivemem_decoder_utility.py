"""Additive decoder-feature ablation; all learner/policy math stays frozen."""
import numpy as np

from navsim.agents import drivemem_component_utility as original
from navsim.agents.drivemem_component_utility import attach, ridge_path, predict, choose, EPS, TOL, KEYS
from navsim.agents.drivemem_action_transfer import pool
from navsim.planning.script.run_drivemem_expanded_utility import load_inputs, load_sample, read, sha, REPO
from navsim.planning.script.run_drivemem_decoder_export import key, feature_path, status


def features(sample, vocab, bank=None, h=None, seed=20261027, random=False, shuffle=False):
    x, supported = original.features(sample, vocab, bank, h, seed, random, shuffle)
    latent = np.asarray(sample["decoder_states"], dtype=float)
    if latent.ndim != 2 or len(latent) != len(sample["candidate_ids"]) or not np.isfinite(latent).all():
        raise ValueError("Invalid decoder feature matrix")
    ix, b = pool(sample), int(sample["selected_local"])
    return np.column_stack([x, latent[ix], latent[ix] - latent[b]]), supported


def load_decoder_sample(root, record, index, decoder):
    s = load_sample(root, record, index)
    entry = read(decoder / "export_index.json")[key(root, s["token"])]
    path = feature_path(decoder, root, s["token"])
    if sha(path) != entry["sha256"] or entry["source_sha256"] != index[s["token"]]:
        raise RuntimeError("Decoder/source feature changed")
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data["candidate_ids"], s["candidate_ids"]):
            raise ValueError("Decoder candidate alignment changed")
        s["decoder_states"] = data["decoder_states"].copy()
    return s


def load_decoder_inputs(prior, addition, decoder):
    data = list(load_inputs(prior, addition))
    bank, fitting, _, _, source, validation, _, bindings, _, _ = data
    identity = read(decoder / "identity.json")
    exporter = identity.get("exporter_path", "navsim/planning/script/run_drivemem_decoder_export.py")
    if exporter not in ("navsim/planning/script/run_drivemem_decoder_export.py",
                        "navsim/planning/script/run_drivemem_decoder_replay_export.py"):
        raise RuntimeError("Unknown decoder export protocol")
    if sha(REPO / exporter) != identity["exporter_sha256"]:
        raise RuntimeError("Decoder exporter changed")
    if exporter.endswith("_replay_export.py") and sha(REPO / "navsim/agents/drivemem_frozen_pool.py") != identity.get("pool_helper_sha256"):
        raise RuntimeError("Frozen-pool replay helper changed")
    if identity["inputs"] != bindings:
        raise RuntimeError("Decoder sidecar belongs to different input snapshot")
    records = []
    for r in identity["records"]:
        from pathlib import Path
        if r["role"] not in ("memory", "fit", "development"):
            raise ValueError("Closed role in decoder export")
        records.append((Path(r["source"]), r))
    _, report = status(decoder, records, identity)
    if len(records) != 152 or report["status"] != "complete":
        raise RuntimeError("Complete 152-scene decoder export required")
    # Verify the previous component experiment so this remains a feature-only
    # ablation, rather than silently changing the underlying learner/policies.
    reference = prior.parent / "component_utility_v1"
    ref_identity = read(reference / "identity.json")
    for name, digest in ref_identity["sources"].items():
        if sha(REPO / name) != digest:
            raise RuntimeError("Frozen component algorithm changed")
    for p in (reference / "identity.json", reference / "report.json", decoder / "identity.json", decoder / "export_index.json"):
        bindings[str(p)] = sha(p)
    roots = {r["token"]: addition for r in read(addition / "label_selection.json")["records"]}
    indices = {root: read(root / "export_index.json") for root in (source, addition)}
    def augment(s):
        root = roots.get(s["token"], source)
        r = dict(token=s["token"], session_id=s["session_id"], role=s["role"])
        extra = load_decoder_sample(root, r, indices[root], decoder)
        return dict(s, decoder_states=extra["decoder_states"])
    data[0], data[1] = [augment(s) for s in bank], [augment(s) for s in fitting]
    return tuple(data)
