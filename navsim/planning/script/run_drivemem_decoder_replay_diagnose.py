"""Read-only, targeted probe of frozen/replayed/hooked numerical parity."""
import json
import os

import numpy as np

from navsim.planning.script import run_drivemem_decoder_replay_export as exporter
from navsim.planning.script.run_drivemem_decoder_diagnose import compare


def numeric_detail(reference, current, ids, atol, rtol):
    x, y = np.asarray(reference), np.asarray(current)
    result = dict(reference_shape=list(x.shape), current_shape=list(y.shape),
                  reference_dtype=str(x.dtype), current_dtype=str(y.dtype))
    if x.shape != y.shape or x.ndim != 1 or len(x) != len(ids):
        return dict(result, comparable=False)
    finite = np.isfinite(x) & np.isfinite(y)
    result.update(comparable=True, nonfinite_elements=int((~finite).sum()))
    if not finite.all():
        return result
    # Match np.allclose(reference, current): relative tolerance uses CURRENT.
    delta = y.astype(np.float64) - x.astype(np.float64)
    absolute = np.abs(delta)
    allowed = atol + rtol * np.abs(y.astype(np.float64))
    failed = absolute > allowed
    ratio = absolute / allowed
    worst = np.argsort(-ratio, kind="stable")[:5]
    result.update(allclose=bool(np.allclose(x, y, atol=atol, rtol=rtol)),
        elements=len(x), failed_elements=int(failed.sum()), max_abs_diff=float(absolute.max()),
        p99_abs_diff=float(np.quantile(absolute, .99)), mean_signed_diff=float(delta.mean()),
        max_tolerance_ratio=float(ratio.max()))
    if failed.any():
        result["worst_by_tolerance"] = [dict(vocab_id=int(ids[i]), reference=float(x[i]), current=float(y[i]),
            signed_diff=float(delta[i]), allowed_abs_diff=float(allowed[i]),
            tolerance_ratio=float(ratio[i])) for i in worst if failed[i]]
    return result


def detailed_compare(reference, current, vocab, label):
    result = compare(reference, current)
    # Full numeric details below replace the older short diff dictionary.
    result.pop("by_vocab_numeric_parity", None)
    result["comparison"] = label
    a, b = reference["candidate_ids"], current["candidate_ids"]
    ai, bi = {int(v): i for i, v in enumerate(a)}, {int(v): i for i, v in enumerate(b)}
    common = sorted(set(ai) & set(bi))
    details = {}
    if len(ai) == len(a) and len(bi) == len(b) and common:
        for field in ("selection_scores",) + tuple(k for k in reference if k.startswith("head_")):
            x = np.asarray(reference[field])[[ai[v] for v in common]]
            y = np.asarray(current[field])[[bi[v] for v in common]]
            details[field] = numeric_detail(x, y, common, exporter.ATOL, exporter.RTOL)
    result["numeric_details"] = details
    result["reference_selected_local"] = int(reference["selected_local"])
    result["current_selected_local"] = int(current["selected_local"])
    result["top16_order_equal"] = result["reference_top16"] == result["current_top16"]
    result["final_trajectory_matches_reference_action"] = bool(np.array_equal(
        current["final_traj"], vocab[int(a[int(reference["selected_local"])])]))
    # Centering and log-softmax are explanatory diagnostics, never acceptance gates.
    if "head_imi" in details and details["head_imi"].get("nonfinite_elements") == 0:
        x = np.asarray(reference["head_imi"], dtype=float)[[ai[v] for v in common]]
        y = np.asarray(current["head_imi"], dtype=float)[[bi[v] for v in common]]
        def log_softmax(z):
            shifted = z - z.max()
            return shifted - np.log(np.exp(shifted).sum())
        result["imi_diagnostics"] = dict(
            centered_max_abs_diff=float(np.max(np.abs((y - y.mean()) - (x - x.mean())))),
            log_softmax_max_abs_diff_on_common_pool=float(np.max(np.abs(log_softmax(y) - log_softmax(x)))),
            note="Descriptive only; no tolerance or parity condition changed.")
    return result


class ProbeDone(Exception):
    pass


def main(cfg):
    target = os.environ.get("DRIVEMEM_DECODER_DIAG_TOKEN", "").strip()
    originals = {name: getattr(exporter, name) for name in ("parity", "write", "save", "status")}
    original_stage = cfg.drivemem.stage
    calls, frozen = [], []
    token = [None]

    def checked_compare(reference, current, vocab, label):
        row = detailed_compare(reference, current, vocab, label)
        try:
            originals["parity"](reference, current, vocab)
            row["strict_parity"] = "passed"
        except RuntimeError as exc:
            row["strict_parity"] = str(exc)
        return row

    def probe(reference, current, vocab):
        if not calls:
            token[0] = reference["token"]
            if target and token[0] != target:
                raise RuntimeError("Diagnostic probed an unexpected token")
            frozen.append(reference)
            calls.append(checked_compare(reference, current, vocab, "frozen_vs_hooked_replay"))
            return 0.
        calls.append(checked_compare(reference, current, vocab, "unhooked_vs_hooked_replay"))
        calls.append(checked_compare(frozen[0], reference, vocab, "frozen_vs_unhooked_replay"))
        print(json.dumps(dict(status="diagnostic_only", token=token[0], atol=exporter.ATOL,
            rtol=exporter.RTOL, comparisons=calls,
            note="One pair of forwards, no retries. No sidecar/index/identity/report writes, fitting or scorer calls. Original parity gates unchanged."),
            indent=2, allow_nan=False), flush=True)
        raise ProbeDone()

    def targeted_status(out, records, identity):
        # Verify the ENTIRE real index before selecting a local read-only probe.
        index, report = originals["status"](out, records, identity)
        if target:
            selected = [(root, row) for root, row in records if row["token"] == target]
            if len(selected) != 1:
                raise ValueError("Diagnostic token must identify exactly one allowed M/F/development record")
            records[:] = selected  # identity was already checked against all 152 records.
            index = dict(index)
            root, row = selected[0]
            index.pop(exporter.key(root, row["token"]), None)  # Reprobe even if saved; never persisted.
        return index, report

    def no_save(*args, **kwargs):
        raise RuntimeError("Diagnostic unexpectedly reached artifact saving")

    exporter.parity, exporter.status = probe, targeted_status
    exporter.write, exporter.save = lambda *args, **kwargs: None, no_save
    cfg.drivemem.stage = "decoder-replay-export"
    try:
        exporter.main(cfg)
        if not calls:
            print("No pending scene probed; specify DRIVEMEM_DECODER_DIAG_TOKEN to inspect a saved scene.", flush=True)
    except ProbeDone:
        pass
    finally:
        for name, value in originals.items():
            setattr(exporter, name, value)
        cfg.drivemem.stage = original_stage


if __name__ == "__main__":
    import hydra
    hydra.main(config_path="config/pdm_scoring", config_name="default_drivemem_memory", version_base=None)(main)()
