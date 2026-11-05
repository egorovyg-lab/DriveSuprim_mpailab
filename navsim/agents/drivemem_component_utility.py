"""Bounded absolute outcome regression and explicit component-preserving policies.

Targets are raw PDM and eight raw components. Outputs are scores, not calibrated
probabilities or confidence bounds. NumPy only; no query outcomes at inference.
"""
import copy
import numpy as np

from navsim.agents.drivemem import METRICS, SAFETY, nearest_donors, probabilities, token_seed
from navsim.agents.drivemem_action_transfer import pool, features as absolute_features
from navsim.agents.drivemem_pair_utility import relative_geometry

KEYS = ("pdm_score",) + METRICS
EPS = .001
TOL = 1e-8


def attach(sample, outcomes):
    ix = pool(sample)
    if int(sample["selected_local"]) not in ix:
        raise ValueError("Baseline outside top16")
    y = np.array([[outcomes[str(i)]["raw"][k] for k in KEYS] for i in ix], dtype=float)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("Invalid raw outcome labels")
    return dict(sample, outcome_values=y, memory_values=y.copy())


def shuffled_bank(bank, seed):
    """Shuffle whole outcome vectors within command groups; true targets stay intact."""
    result = copy.deepcopy(bank)
    commands = sorted({tuple((s["ego_status"][:4] > .5).tolist()) for s in bank})
    for command in commands:
        ids = sorted([i for i, s in enumerate(bank) if tuple((s["ego_status"][:4] > .5).tolist()) == command], key=lambda i: bank[i]["token"])
        y = np.concatenate([bank[i]["memory_values"] for i in ids])
        y = y[np.random.default_rng(token_seed(str(command), seed)).permutation(len(y))]
        start = 0
        for i in ids:
            n = len(bank[i]["memory_values"])
            result[i]["memory_values"] = y[start:start + n].copy()
            start += n
    return result


def recall(sample, bank, vocab, h, seed, random=False, shuffle=False):
    eligible = [s for s in bank if s["session_id"] != sample["session_id"]]
    if h <= 0 or any(s["role"] != "memory" for s in bank):
        raise ValueError("Positive h and M-only bank required")
    # Shuffle AFTER exclusion: a training M query's labels must not be moved
    # into another donor and accidentally become its own inference features.
    if shuffle and eligible:
        eligible = shuffled_bank(eligible, seed)
    ids, _ = nearest_donors(sample, eligible, 5, seed, random=random)
    ix = pool(sample)
    if not ids:
        return np.zeros((len(ix), 28)), False
    qg = relative_geometry(sample, vocab)[ix]
    terms, supports = [], []
    for i in ids:
        donor = eligible[i]
        g = relative_geometry(donor, vocab)[pool(donor)]
        d = np.square(qg[:, None] - g[None]).mean(axis=2)
        w = np.exp(-(d - d.min(axis=1, keepdims=True)) / h)
        w /= w.sum(axis=1, keepdims=True)
        terms.append(w @ donor["memory_values"])
        supports.append(np.exp(-d.min(axis=1) / h))
    terms, supports = np.array(terms), np.array(supports)
    weighted = (terms * supports[:, :, None]).sum(axis=0) / (1 + supports.sum(axis=0)[:, None])
    b = list(ix).index(int(sample["selected_local"]))
    return np.column_stack([weighted, weighted - weighted[b], terms.std(axis=0), supports.mean(axis=0)]), True


def features(sample, vocab, bank=None, h=None, seed=20261027, random=False, shuffle=False):
    # Absolute context is present at baseline, so baseline Q can vary by scene.
    ix = pool(sample)
    x = absolute_features(sample, vocab, ix)
    gap = sample["selection_scores"][ix] - sample["selection_scores"][int(sample["selected_local"])]
    x = np.column_stack([x, gap])
    supported = True
    if bank is not None:
        recalled, supported = recall(sample, bank, vocab, h, seed, random, shuffle)
        x = np.column_stack([x, recalled])
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features")
    return x, supported


def ridge_path(xs, ys, lambdas):
    """Scene-equal least squares on clipped outcome logits, with free intercept.

    This learns new absolute targets. Applying sigmoid to the old delta model
    would preserve its ranking and is NOT what is done here.
    """
    x, y = np.concatenate(xs), np.concatenate(ys)
    y = np.clip(y, EPS, 1 - EPS)
    y = np.log(y / (1 - y))
    w = np.concatenate([np.full(len(a), 1 / len(a)) for a in xs])
    w /= w.mean()
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
    scale[scale < 1e-6] = 1
    ym = np.average(y, axis=0, weights=w)
    a = (x - mean) / scale * np.sqrt(w[:, None])
    eigen, u = np.linalg.eigh(a @ a.T)
    rhs = u.T @ ((y - ym) * np.sqrt(w[:, None]))
    result = {}
    for lam in lambdas:
        if lam <= 0:
            raise ValueError("Positive regularization required")
        coef = a.T @ (u @ (rhs / (np.maximum(eigen, 0)[:, None] + lam)))
        result[lam] = dict(mean=mean, scale=scale, ym=ym, coef=coef)
    return result


def predict(x, model):
    z = (x - model["mean"]) / model["scale"] @ model["coef"] + model["ym"]
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def choose(sample, prediction, policy="guarded", supported=True):
    ix = pool(sample)
    b = list(ix).index(int(sample["selected_local"]))
    if prediction.shape != (len(ix), len(KEYS)) or not np.isfinite(prediction).all():
        raise ValueError("Invalid bounded prediction")
    if policy not in ("raw", "component", "guarded"):
        raise ValueError("Unknown policy")
    if not supported:
        return int(ix[b])
    allowed = np.ones(len(ix), dtype=bool)
    if policy != "raw":
        j = np.array(SAFETY) + 1
        allowed &= (prediction[:, j] >= prediction[b, j] - TOL).all(axis=1)
    if policy == "guarded":
        teacher = probabilities(sample)[ix][:, SAFETY]
        allowed &= (teacher >= teacher[b] - TOL).all(axis=1)
    allowed[b] = True
    quality = np.where(allowed, prediction[:, 0], -np.inf)
    if quality.max() <= prediction[b, 0] + TOL:
        return int(ix[b])
    return int(ix[np.argmax(quality)])
