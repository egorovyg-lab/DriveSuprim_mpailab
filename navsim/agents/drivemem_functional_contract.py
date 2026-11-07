"""Post-audit functional acceptance; IMI drift is recorded, never thresholded."""
from pathlib import Path
import numpy as np

from navsim.agents.drivemem import METRICS
from navsim.agents.drivemem_determinism import PROFILE
from navsim.agents.drivemem_numeric_contract import audit, aggregate as numeric_aggregate

CONTRACT = dict(metric_logit_max_abs=5e-5, metric_sigmoid_max_abs=1e-5,
                score_atol=1e-5, score_rtol=1e-6, top16_set_exact=True,
                imi_drift="diagnostic_only_finite_required", selected_after_full_audit=True)
PILOT_TOKENS = ("e7832ec3cbcb5fe4", "fedf56bd69af5cec", "64f4a68010155184", "c5c16e45232453e3")


def pilot_order(records):
    if not set(PILOT_TOKENS).issubset({r["token"] for _, r in records}):
        raise ValueError("Functional pilot scenes absent from frozen cohort")
    priority = {token: i for i, token in enumerate(PILOT_TOKENS)}
    return sorted(records, key=lambda item: priority.get(item[1]["token"], len(priority)))


def parity(old, fresh, vocab):
    heads = {"head_" + name for name in METRICS} | {"head_imi"}
    for sample in (old, fresh):
        if {name for name in sample if name.startswith("head_")} != heads:
            raise RuntimeError("Functional export requires all eight metric heads and IMI")
        ids = np.asarray(sample["candidate_ids"])
        if ids.shape != (256,) or ids.dtype.kind not in "iu" or len(set(ids.tolist())) != 256:
            raise RuntimeError("Invalid functional candidate pool")
        if ids.min() < 0 or ids.max() >= len(vocab):
            raise RuntimeError("Invalid vocabulary ID")
        for name in heads | {"selection_scores"}:
            if np.asarray(sample[name]).shape != (256,):
                raise RuntimeError("Invalid functional vector: " + name)
    for name in ("candidate_ids", "selected_local"):
        if not np.array_equal(old[name], fresh[name]):
            raise RuntimeError("Functional export changed " + name)
    result = audit(old, fresh)  # Includes finite checks for IMI as well.
    if not np.allclose(old["selection_scores"], fresh["selection_scores"],
                       atol=CONTRACT["score_atol"], rtol=CONTRACT["score_rtol"]):
        raise RuntimeError("Functional export selection score drift")
    for name in heads - {"head_imi"}:
        row = result["fields"][name]
        if row["max_abs_diff"] > CONTRACT["metric_logit_max_abs"]:
            raise RuntimeError("Metric logit budget exceeded: " + name)
        if row["sigmoid_max_abs_diff"] > CONTRACT["metric_sigmoid_max_abs"]:
            raise RuntimeError("Metric sigmoid budget exceeded: " + name)
    if not result["top16_set_equal"]:
        raise RuntimeError("Functional export changed top16 membership")
    ids, b = fresh["candidate_ids"], int(fresh["selected_local"])
    if not 0 <= b < len(ids) or b != int(np.argmax(fresh["selection_scores"])):
        raise RuntimeError("Functional export changed selected action")
    if not np.array_equal(fresh["final_traj"], vocab[ids[b]]):
        raise RuntimeError("Functional export changed selected trajectory")
    return max(row["max_abs_diff"] for row in result["fields"].values())


def aggregate(index):
    result = numeric_aggregate(index)
    result["contract"] = CONTRACT
    result["imi_drift_is_diagnostic"] = True
    return result


def prior_audit_snapshot(root, bindings, records, repo, read, sha):
    """Bind the COMPLETE earlier audit, including failures and per-scene hashes."""
    identity, report = read(root / "identity.json"), read(root / "report.json")
    if identity["inputs"] != bindings or identity["profile"] != PROFILE:
        raise RuntimeError("Prior audit inputs/profile differ")
    if identity["runner"] != sha(repo / "navsim/planning/script/run_drivemem_decoder_numeric_audit.py"):
        raise RuntimeError("Prior audit runner changed")
    for name, digest in identity["helpers"].items():
        if sha(repo / name) != digest:
            raise RuntimeError("Prior audit dependency changed: " + name)
    expected = {rt.name + ":" + r["token"]: (str(rt), r["token"], r["role"]) for rt, r in records}
    actual = {Path(r["source"]).name + ":" + r["token"]: (r["source"], r["token"], r["role"]) for r in identity["records"]}
    if len(records) != 152 or len(identity["records"]) != 152 or actual != expected:
        raise RuntimeError("Prior audit cohort differs")
    index = read(root / "audit_index.json")
    if set(index) != set(expected) or report["status"] != "complete" or report["audited_scenes"] != 152 or report["required_scenes"] != 152:
        raise RuntimeError("Complete matching 152-scene prior audit required")
    runtime_path = root / "runtime.json"
    if read(runtime_path)["settings"] != PROFILE:
        raise RuntimeError("Prior audit runtime profile differs")
    runtime_digest = sha(runtime_path)
    files = {name: sha(root / name) for name in ("identity.json", "report.json", "audit_index.json", "runtime.json")}
    for k, digest in index.items():
        name = "scenes/" + k.replace(":", "_") + ".json"
        row = read(root / name)
        if sha(root / name) != digest or row["runtime_sha256"] != runtime_digest:
            raise RuntimeError("Prior audit scene/runtime changed: " + k)
        if row["token"] != expected[k][1] or row["role"] != expected[k][2] or row["source"] != expected[k][0]:
            raise RuntimeError("Prior audit scene identity changed")
        if (not row["repeat_bitwise_equal"] or not row["baseline_equal"] or not row["vocabulary_equal"] or
                not row["head_alignment"] or not all(row["head_alignment"].values()) or
                row["numeric_audit"] is None or not row["numeric_audit"]["top16_set_equal"]):
            raise RuntimeError("Prior audit has a functional invariant failure: " + k)
        files[name] = digest
    return dict(path=str(root), files=files, criteria_revised_after_audit=True)
