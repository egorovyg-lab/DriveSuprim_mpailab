"""Finite, fixed old/new numerical budget; current repeats remain exact."""
import numpy as np

CONTRACT = dict(logit_max_abs=5e-5, sigmoid_max_abs=1e-5, imi_log_softmax_max_abs=1e-4,
                score_atol=1e-5, score_rtol=1e-6, top16_set_exact=True)


def audit(old, fresh):
    fields = {}
    names = ("selection_scores",) + tuple(k for k in old if k.startswith("head_"))
    for name in names:
        x, y = np.asarray(old[name], dtype=float), np.asarray(fresh[name], dtype=float)
        if x.shape != y.shape or x.ndim != 1 or not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise RuntimeError("Invalid numerical array: " + name)
        delta = np.abs(x - y)
        row = dict(max_abs_diff=float(delta.max()), legacy_failed_elements=int((delta > 1e-5 + 1e-6 * np.abs(y)).sum()))
        if name == "head_imi":
            def log_softmax(z):
                z = z - z.max()
                return z - np.log(np.exp(z).sum())
            row["log_softmax_max_abs_diff"] = float(np.max(np.abs(log_softmax(x) - log_softmax(y))))
        elif name.startswith("head_"):
            # Stable sigmoid without overflow for either sign.
            def sigmoid(z):
                return .5 * (1. + np.tanh(.5 * z))
            row["sigmoid_max_abs_diff"] = float(np.max(np.abs(sigmoid(x) - sigmoid(y))))
        fields[name] = row
    def top16(s):
        return set(map(int, s["candidate_ids"][np.argsort(-s["selection_scores"], kind="stable")[:16]]))
    return dict(fields=fields, top16_set_equal=top16(old) == top16(fresh))


def parity(old, fresh, vocab):
    for name in ("candidate_ids", "selected_local"):
        if not np.array_equal(old[name], fresh[name]):
            raise RuntimeError("Numeric export changed " + name)
    report = audit(old, fresh)
    for name, row in report["fields"].items():
        if name == "selection_scores":
            if not np.allclose(old[name], fresh[name], atol=CONTRACT["score_atol"], rtol=CONTRACT["score_rtol"]):
                raise RuntimeError("Numeric export selection score drift")
        elif row["max_abs_diff"] > CONTRACT["logit_max_abs"]:
            raise RuntimeError("Logit budget exceeded: " + name)
        if row.get("sigmoid_max_abs_diff", 0.) > CONTRACT["sigmoid_max_abs"]:
            raise RuntimeError("Sigmoid budget exceeded: " + name)
        if row.get("log_softmax_max_abs_diff", 0.) > CONTRACT["imi_log_softmax_max_abs"]:
            raise RuntimeError("IMI normalized budget exceeded")
    if not report["top16_set_equal"]:
        raise RuntimeError("Numeric export changed top16 membership")
    ids, b = fresh["candidate_ids"], int(fresh["selected_local"])
    if b != int(np.argmax(fresh["selection_scores"])) or not np.array_equal(fresh["final_traj"], vocab[ids[b]]):
        raise RuntimeError("Numeric export changed selected trajectory")
    return max(row["max_abs_diff"] for row in report["fields"].values())


def aggregate(index):
    reports = [entry["numeric_audit"] for entry in index.values()]
    names = sorted({name for report in reports for name in report["fields"]})
    return dict(contract=CONTRACT, audited_scenes=len(reports),
        scenes_outside_legacy_tolerance=sum(any(v["legacy_failed_elements"] for v in r["fields"].values()) for r in reports),
        field_maxima={name: {stat: max(r["fields"][name][stat] for r in reports)
                       for stat in reports[0]["fields"][name]} for name in names})
