"""Direct candidate-vs-baseline utility, with a matched retrieval control.

NumPy only. No query outcomes are consumed by features or inference.
"""
import numpy as np

from navsim.agents.drivemem import geometry, nearest_donors, normalize_embedding, probabilities
from navsim.agents.drivemem_action_transfer import pool


def relative_geometry(sample, vocab):
    g = geometry(vocab[sample["candidate_ids"]])
    return g - g[int(sample["selected_local"])]


def attach_utility(sample, outcomes):
    ix = pool(sample)
    base = int(sample["selected_local"])
    if base not in ix or any(str(i) not in outcomes for i in ix):
        raise ValueError("Complete top16 including baseline required")
    scores = np.array([outcomes[str(i)]["raw"]["pdm_score"] for i in ix], dtype=float)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid PDM labels")
    return dict(sample, utility_indices=ix, utility_values=scores - scores[list(ix).index(base)])


def bandwidth(bank, vocab, seed):
    """M geometry only; fixed median positive nearest cross-session distance."""
    distances = []
    for query in bank:
        eligible = [s for s in bank if s["session_id"] != query["session_id"]]
        ids, _ = nearest_donors(query, eligible, 5, seed)
        if not ids:
            continue
        q = relative_geometry(query, vocab)[pool(query)]
        g = np.concatenate([relative_geometry(eligible[i], vocab)[pool(eligible[i])] for i in ids])
        d = np.square(q[:, None] - g[None]).mean(axis=2)
        near = np.where(d > 1e-12, d, np.inf).min(axis=1)
        distances.extend(near[np.isfinite(near)])
    if not distances:
        raise ValueError("No cross-session M geometry support")
    return float(np.median(distances))


def recall_features(query, bank, vocab, h, seed):
    if h <= 0 or any(s["role"] != "memory" for s in bank):
        raise ValueError("Positive bandwidth and M-only bank required")
    # Filter before retrieval as well as inside nearest_donors: all scenes of
    # the query session are excluded, including during M training.
    eligible = [s for s in bank if s["session_id"] != query["session_id"]]
    ids, _ = nearest_donors(query, eligible, 5, seed)
    q = relative_geometry(query, vocab)[pool(query)]
    values = np.zeros((len(q), 5, 3))
    for slot, i in enumerate(ids):
        donor = eligible[i]
        g = relative_geometry(donor, vocab)[donor["utility_indices"]]
        d = np.square(q[:, None] - g[None]).mean(axis=2)
        logits = -d / h
        w = np.exp(logits - logits.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        estimate = w @ donor["utility_values"]
        support = np.exp(-d.min(axis=1) / h)
        values[:, slot] = np.column_stack([estimate, support * estimate, support])
    pooled = values[:, :, 1].sum(axis=1) / (1 + values[:, :, 2].sum(axis=1))
    x = np.column_stack([values.reshape(len(q), -1), pooled])
    base = list(pool(query)).index(int(query["selected_local"]))
    return x - x[base]


def features(sample, vocab, bank=None, h=None, seed=20261027):
    ix = pool(sample)
    base = int(sample["selected_local"])
    p = probabilities(sample)
    g = geometry(vocab[sample["candidate_ids"]])
    score = np.asarray(sample["selection_scores"], dtype=float)
    relative = np.column_stack([g[ix] - g[base], p[ix] - p[base], score[ix] - score[base]])
    ego = np.asarray(sample["ego_status"], dtype=float).copy()
    for start in range(0, len(ego), 8):
        ego[start + 4:start + 6] /= 10
        ego[start + 6:start + 8] /= 3
    context = np.r_[normalize_embedding(sample["scene_embedding"]), ego, p[base], g[base]]
    x = np.column_stack([relative, (context[None, :, None] * relative[:, None]).reshape(len(ix), -1)])
    if bank is not None:
        x = np.column_stack([x, recall_features(sample, bank, vocab, h, seed)])
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features")
    return x


def ridge_path(xs, ys, lambdas):
    """Scene-equal ridge. RMS scale, no centering/intercept: f(baseline)=0."""
    x, y = np.concatenate(xs), np.concatenate(ys)
    w = np.concatenate([np.full(len(a), 1 / len(a)) for a in xs])
    w /= w.mean()
    scale = np.sqrt(np.average(x * x, axis=0, weights=w))
    scale[scale < 1e-6] = 1
    a = x / scale * np.sqrt(w[:, None])
    eigen, u = np.linalg.eigh(a @ a.T)
    rhs = u.T @ (y * np.sqrt(w))
    models = {}
    for lam in lambdas:
        if lam <= 0:
            raise ValueError("Positive regularization required")
        coef = a.T @ (u @ (rhs / (np.maximum(eigen, 0) + lam)))
        models[lam] = dict(scale=scale, coef=coef)
    return models


def predict(x, model):
    return x / model["scale"] @ model["coef"]


def choose(sample, prediction):
    ix = pool(sample)
    base = int(sample["selected_local"])
    prediction = np.asarray(prediction, dtype=float)
    if base not in ix or prediction.shape != ix.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid top16 predictions")
    return base if prediction.max() <= 0 else int(ix[np.argmax(prediction)])
