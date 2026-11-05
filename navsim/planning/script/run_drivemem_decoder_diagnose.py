"""Probe the first pending decoder scene without modifying the frozen sidecar."""
import json
import numpy as np

from navsim.planning.script import run_drivemem_decoder_export as exporter


def compare(old, fresh):
    a, b = np.asarray(old["candidate_ids"]), np.asarray(fresh["candidate_ids"])
    aset, bset = set(map(int, a)), set(map(int, b))
    def ordered_ids(s):
        ix = np.argsort(-s["selection_scores"], kind="stable")[:16]
        return [int(i) for i in s["candidate_ids"][ix]]
    result = dict(reference="frozen_export" if "token" in old else "current_unhooked_pass",
        reference_count=len(a), current_count=len(b), ordered_ids_equal=np.array_equal(a, b),
        same_id_set=aset == bset, reference_unique=len(aset), current_unique=len(bset),
        missing_ids=sorted(aset - bset), added_ids=sorted(bset - aset),
        reference_selected_vocab=int(a[int(old["selected_local"])]),
        current_selected_vocab=int(b[int(fresh["selected_local"])]),
        reference_top16=ordered_ids(old), current_top16=ordered_ids(fresh))
    result["selected_vocab_equal"] = result["reference_selected_vocab"] == result["current_selected_vocab"]
    common = sorted(aset & bset)
    if common and len(aset) == len(a) and len(bset) == len(b):
        ai, bi = {int(v): i for i, v in enumerate(a)}, {int(v): i for i, v in enumerate(b)}
        diffs = {}
        for field in ("selection_scores",) + tuple(k for k in old if k.startswith("head_")):
            x, y = np.asarray(old[field])[[ai[v] for v in common]], np.asarray(fresh[field])[[bi[v] for v in common]]
            diffs[field] = dict(max_abs_diff=float(np.max(np.abs(x - y))),
                allclose=bool(np.allclose(x, y, atol=exporter.ATOL, rtol=exporter.RTOL)))
        result["common_ids"] = len(common)
        result["by_vocab_numeric_parity"] = diffs
    return result


class ProbeDone(Exception):
    pass


def main(cfg):
    calls = []
    old_parity, old_write, old_save = exporter.parity, exporter.write, exporter.save
    def probe(old, fresh, vocab):
        row = compare(old, fresh)
        try:
            old_parity(old, fresh, vocab)
            row["strict_parity"] = "passed"
        except RuntimeError as exc:
            row["strict_parity"] = str(exc)
        calls.append(row)
        if len(calls) == 2:
            print(json.dumps(dict(status="diagnostic_only", token=token[0], comparisons=calls,
                note="No sidecar/index/identity writes. Frozen export and current unhooked pass compared separately. No parity conditions relaxed."), indent=2), flush=True)
            raise ProbeDone()
        return 0.
    token = [None]
    def with_token(old, fresh, vocab):
        if "token" in old:
            token[0] = old["token"]
        return probe(old, fresh, vocab)
    def no_save(*args, **kwargs):
        raise RuntimeError("Diagnostic unexpectedly reached artifact save")
    exporter.parity = with_token
    exporter.write = lambda *a, **kw: None
    exporter.save = no_save
    cfg.drivemem.stage = "decoder-export"  # stage is excluded from identity.
    try:
        exporter.main(cfg)
        if not calls:
            print("No pending scene probed (sidecar may already be complete).", flush=True)
    except ProbeDone:
        pass
    finally:
        exporter.parity, exporter.write, exporter.save = old_parity, old_write, old_save


if __name__ == "__main__":
    import hydra
    hydra.main(config_path="config/pdm_scoring", config_name="default_drivemem_memory", version_base=None)(main)()
