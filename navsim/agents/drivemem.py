"""Small, auditable residual-memory baseline. NumPy only; no query outcomes."""
import hashlib
import re

import numpy as np

METRICS = (
    "no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
    "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound",
    "lane_keeping", "history_comfort",
)
SAFETY = (0, 1, 2, 3, 5)
ROLES = ("memory", "fit", "development", "calibration", "test")


def session_id(log_name):
    match = re.fullmatch(r"(.+_veh-[^_]+)_\d+_\d+", log_name)
    if not match:
        raise ValueError(f"Unrecognized log name; cannot infer session safely: {log_name}")
    return match.group(1)


def split_sessions(records, counts, seed):
    """Allocate whole source sessions before looking at cache/quality labels."""
    sessions = sorted({r["session_id"] for r in records})
    total = sum(int(counts[r]) for r in ROLES)
    if any(int(counts[r]) < 1 for r in ROLES) or total > len(sessions):
        raise ValueError(f"Need {total} eligible sessions, available {len(sessions)}; counts={counts}")
    shuffled = np.random.default_rng(seed).permutation(sessions).tolist()
    result, start = {}, 0
    for role in ROLES:
        result[role] = sorted(shuffled[start:start + int(counts[role])])
        start += int(counts[role])
    return result


def spaced_records(records, count, gap_s, seed):
    ordered = sorted(records, key=lambda r: (int(r["timestamp_us"]), r["token"]))
    times = np.array([int(r["timestamp_us"]) for r in ordered], dtype=np.int64)
    if not len(times):
        raise ValueError("No scenes")
    nxt = np.maximum(np.arange(len(times)) + 1,
                     np.searchsorted(times, times + int(round(gap_s * 1e6))))
    capacity = np.zeros(len(times) + 1, dtype=int)
    for i in range(len(times) - 1, -1, -1):
        capacity[i] = 1 + capacity[nxt[i]]
    if capacity[0] < count:
        raise ValueError("Session does not support requested temporal spacing")
    rng = np.random.default_rng(seed)
    selected, start = [], 0
    for remaining in range(count, 0, -1):
        options = np.flatnonzero(capacity[:-1] >= remaining)
        index = int(rng.choice(options[options >= start]))
        selected.append(ordered[index])
        start = int(nxt[index])
    return selected


def token_seed(token, seed):
    return int.from_bytes(hashlib.sha256((str(seed) + ":" + token).encode()).digest()[:8], "big")


def probabilities(sample):
    logits = np.stack([sample["head_" + k] for k in METRICS], axis=1).astype(np.float64)
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def geometry(poses):
    """Action compatibility descriptors in ego coordinates; never future truth."""
    poses = np.asarray(poses)
    indices = np.linspace(0, poses.shape[1] - 1, 4).astype(int)
    positions = (poses[:, indices, :2] / np.array([20., 10.])).reshape(len(poses), -1)
    heading = poses[:, -1, 2]
    origin = np.zeros((len(poses), 1, 2))
    distance = np.linalg.norm(np.diff(np.concatenate([origin, poses[:, :, :2]], axis=1), axis=1), axis=2).sum(axis=1)
    return np.column_stack([positions, np.sin(heading), np.cos(heading), distance / 40.])


def metric_selection_score(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return (.5 * np.log(p[:, 0]) + .5 * np.log(p[:, 1]) + .3 * np.log(p[:, 2])
            + .1 * np.log(p[:, 3])
            + 6 * np.log(5 * p[:, 4] + 5 * p[:, 5] + 2 * p[:, 6] + p[:, 7]))


def corrected_choice(sample, correction, strength):
    p = probabilities(sample)
    scores = sample["selection_scores"].astype(np.float64).copy()
    scores += metric_selection_score(np.clip(p + strength * correction, 0, 1)) - metric_selection_score(p)
    baseline = int(sample["selected_local"])
    # Preserve the real baseline in exact ties, including the no-op case.
    return baseline if scores[baseline] == scores.max() else int(np.argmax(scores))


def training_candidate_indices(sample, top, random_count, seed):
    order = np.argsort(-sample["selection_scores"], kind="stable")
    selected = int(sample["selected_local"])
    chosen = list(dict.fromkeys([selected] + order[:top].tolist()))
    remaining = np.array([int(i) for i in order if i not in chosen], dtype=int)
    if random_count:
        rng = np.random.default_rng(token_seed(str(sample["token"]), seed))
        chosen.extend(rng.choice(remaining, min(random_count, len(remaining)), replace=False).tolist())
    return np.array(chosen, dtype=np.int64)


def normalize_embedding(x):
    x = np.asarray(x, dtype=np.float64)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def nearest_donors(query, bank, k, seed, random=False):
    """Compatibility uses command and observed ego state, not scenario outcomes."""
    command = tuple((query["ego_status"][:4] > .5).tolist())
    possible, distances = [], []
    q = normalize_embedding(query["scene_embedding"])
    for i, donor in enumerate(bank):
        if donor["session_id"] == query["session_id"]:
            raise ValueError("Memory/query session overlap")
        if tuple((donor["ego_status"][:4] > .5).tolist()) != command:
            continue
        visual = np.square(q - normalize_embedding(donor["scene_embedding"])).sum()
        velocity = np.square((query["ego_status"][4:6] - donor["ego_status"][4:6]) / 5.).sum()
        possible.append(i)
        distances.append(float(visual + velocity))
    if not possible:
        return [], float("inf")
    order = np.argsort(distances, kind="stable")
    nearest = float(distances[int(order[0])])
    if random:
        order = np.random.default_rng(token_seed(str(query["token"]), seed)).permutation(len(possible))
    return [possible[int(i)] for i in order[:k]], nearest


def memory_correction(query, bank, donor_indices, vocab):
    q_geometry = geometry(vocab[query["candidate_ids"]])
    if not donor_indices:
        return np.zeros((len(q_geometry), len(METRICS)))
    corrections = []
    for index in donor_indices:
        donor = bank[index]
        action_indices = donor["label_local_indices"]
        donor_geometry = geometry(vocab[donor["candidate_ids"][action_indices]])
        residual = donor["label_values"] - probabilities(donor)[action_indices]
        distances = np.square(q_geometry[:, None, :] - donor_geometry[None, :, :]).mean(axis=2)
        # Stable softmax: every donor has equal total episode weight.
        logits = -distances / .25
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        corrections.append(weights @ residual)
    return np.mean(corrections, axis=0)


def context(sample, projection):
    embedding = normalize_embedding(sample["scene_embedding"]) @ projection
    ego = np.asarray(sample["ego_status"], dtype=float).copy()
    if len(ego) % 8:
        raise ValueError("Expected concatenated 8-value ego status vectors")
    for start in range(0, len(ego), 8):
        ego[start + 4:start + 6] /= 10.
        ego[start + 6:start + 8] /= 3.
    return np.concatenate([embedding, ego])


def design(sample, vocab, projection):
    p = probabilities(sample)
    g = geometry(vocab[sample["candidate_ids"]])
    c = context(sample, projection)
    repeated = np.broadcast_to(c, (len(g), len(c)))
    interactions = (repeated[:, :, None] * g[:, None, :]).reshape(len(g), -1)
    return np.column_stack([repeated, g, p, interactions])


def fit_ridge(samples, vocab, seed=20261027, regularization=10.):
    dim = len(samples[0]["scene_embedding"])
    projection = np.random.default_rng(seed).normal(size=(dim, min(12, dim))) / np.sqrt(dim)
    x, y, weights = [], [], []
    for sample in samples:
        idx = sample["label_local_indices"]
        x.append(design(sample, vocab, projection)[idx])
        y.append(sample["label_values"] - probabilities(sample)[idx])
        weights.extend([1 / len(idx)] * len(idx))  # equal scene weight
    x, y = np.concatenate(x), np.concatenate(y)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-6] = 1.
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    weights = np.asarray(weights)
    weights /= weights.mean()
    penalty = np.eye(z.shape[1]) * regularization
    penalty[0, 0] = 0
    coef = np.linalg.solve(z.T @ (z * weights[:, None]) + penalty, z.T @ (y * weights[:, None]))
    return dict(projection=projection, mean=mean, scale=scale, coef=coef)


def ridge_correction(sample, vocab, model):
    x = design(sample, vocab, model["projection"])
    z = np.column_stack([np.ones(len(x)), (x - model["mean"]) / model["scale"]])
    return z @ model["coef"]
