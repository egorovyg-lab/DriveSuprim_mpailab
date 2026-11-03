"""Action-transfer ablations and a full-embedding ridge control. NumPy only."""
import numpy as np

from navsim.agents.drivemem import (
    METRICS, geometry, nearest_donors, probabilities, metric_selection_score, normalize_embedding,
)


def pool(sample):
    return np.argsort(-sample["selection_scores"], kind="stable")[:16]


def transfer(query, bank, vocab, h, contrast=False, seed=20261027, random=False, scene_only=False):
    if h <= 0:
        raise ValueError("Bandwidth must be positive")
    ids, distance = nearest_donors(query, bank, 5, seed, random=random)
    g = geometry(vocab[query["candidate_ids"]])
    terms, ess, support, residual_spread, entropy = [], [], [], [], []
    for i in ids:
        donor = bank[i]
        idx = donor["label_local_indices"]
        d = np.square(g[:, None] - geometry(vocab[donor["candidate_ids"][idx]])[None]).mean(axis=2)
        logits = -d / h
        w = np.exp(logits - logits.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        r = donor["label_values"] - probabilities(donor)[idx]
        term = np.broadcast_to(r.mean(axis=0), (len(g), len(METRICS))).copy() if scene_only else w @ r
        if contrast:
            term -= term[int(query["selected_local"])].copy()
        terms.append(term)
        residual_spread.append(float(r.std(axis=0).mean()))
        entropy.append(-(w * np.log(np.maximum(w, 1e-300))).sum(axis=1))
        ess.append(1 / np.square(w).sum(axis=1))
        support.append(d.min(axis=1))
    correction = np.mean(terms, axis=0) if terms else np.zeros((len(g), len(METRICS)))
    ix = pool(query)
    return correction, dict(donors=len(ids), donor_sessions=len({bank[i]["session_id"] for i in ids}),
        distance=float(distance) if np.isfinite(distance) else None,
        ess=float(np.mean(np.asarray(ess)[:, ix])) if ess else None,
        entropy=float(np.mean(np.asarray(entropy)[:, ix])) if entropy else None,
        donor_residual_action_std=float(np.mean(residual_spread)) if residual_spread else None,
        donor_disagreement=float(np.asarray(terms)[:, ix].std(axis=0).mean()) if terms else None,
        action_support_distance=float(np.mean(np.asarray(support)[:, ix])) if support else None,
        correction_abs=float(np.abs(correction[ix]).mean()),
        candidate_std=float(correction[ix].std(axis=0).mean()))


def choose(sample, correction, beta=.5):
    p = probabilities(sample)
    scores = sample["selection_scores"].astype(float) + metric_selection_score(np.clip(p + beta * correction, 0, 1)) - metric_selection_score(p)
    ix = pool(sample)
    base = int(sample["selected_local"])
    if base not in ix:
        raise ValueError("Baseline outside top-16")
    return base if scores[base] == scores[ix].max() else int(ix[np.argmax(scores[ix])])


def bandwidth_scale(bank, vocab, seed):
    values = []
    for query in bank:
        donors = [s for s in bank if s["session_id"] != query["session_id"]]
        ids, _ = nearest_donors(query, donors, 5, seed)
        if not ids:
            continue
        qg = geometry(vocab[query["candidate_ids"][pool(query)]])
        dg = np.concatenate([geometry(vocab[donors[i]["candidate_ids"][donors[i]["label_local_indices"]]]) for i in ids])
        d = np.square(qg[:, None] - dg[None]).mean(axis=2)
        positive = np.where(d > 1e-12, d, np.inf).min(axis=1)
        values.extend(positive[np.isfinite(positive)].tolist())
    if not values:
        raise ValueError("No positive cross-session action distances in M")
    return float(np.median(values))


def features(sample, vocab, indices):
    """Same context/action/interaction family as v1, retaining FULL visual embedding."""
    ego = sample["ego_status"].astype(float).copy()
    for start in range(0, len(ego), 8):
        ego[start + 4:start + 6] /= 10
        ego[start + 6:start + 8] /= 3
    c = np.r_[normalize_embedding(sample["scene_embedding"]), ego]
    g = geometry(vocab[sample["candidate_ids"][indices]])
    return np.column_stack([np.broadcast_to(c, (len(g), len(c))), g, probabilities(sample)[indices],
                            (c[None, :, None] * g[:, None, :]).reshape(len(g), -1)])


def ridge_path(samples, vocab, lambdas):
    """Weighted ridge with an unpenalized intercept, solved in sample space.

    Normalization uses training data only. One eigendecomposition per CV fold
    supplies the complete regularization path; labels have equal scene weight.
    """
    x = np.concatenate([features(s, vocab, s["label_local_indices"]) for s in samples])
    y = np.concatenate([s["label_values"] - probabilities(s)[s["label_local_indices"]] for s in samples])
    w = np.concatenate([np.full(len(s["label_local_indices"]), 1 / len(s["label_local_indices"])) for s in samples])
    w /= w.mean()
    mean, ym = np.average(x, axis=0, weights=w), np.average(y, axis=0, weights=w)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
    scale[scale < 1e-6] = 1
    a = (x - mean) / scale * np.sqrt(w[:, None])
    yc = (y - ym) * np.sqrt(w[:, None])
    eigen, u = np.linalg.eigh(a @ a.T)
    rhs = u.T @ yc
    models = {}
    for lam in lambdas:
        if lam <= 0:
            raise ValueError("Positive regularization required")
        coef = a.T @ (u @ (rhs / (np.maximum(eigen, 0)[:, None] + lam)))
        models[lam] = dict(mean=mean, scale=scale, ym=ym, coef=coef)
    return models


def ridge_predict(sample, vocab, model):
    ix = pool(sample)
    correction = np.zeros((len(sample["candidate_ids"]), len(METRICS)))
    correction[ix] = (features(sample, vocab, ix) - model["mean"]) / model["scale"] @ model["coef"] + model["ym"]
    return correction
