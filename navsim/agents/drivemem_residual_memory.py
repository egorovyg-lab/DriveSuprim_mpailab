"""Cross-session memory of a learned selector's out-of-group errors.

Input preparation and retrieval need NumPy only. Torch is imported only for
fitting/replay. Query labels are deliberately absent from the inference API.
"""
from dataclasses import dataclass, asdict
import hashlib
import itertools
import json

import numpy as np

from navsim.agents.drivemem import METRICS, SAFETY, geometry, probabilities, normalize_embedding

INPUT_KEYS = ("candidate_ids", "selected_local", "selection_scores", "scene_embedding",
              "ego_status", "decoder_states") + tuple("head_" + k for k in (*METRICS, "imi"))
LABEL_KEYS = ("pdm_score",) + METRICS
BLOCKS = ("context", "teacher", "geometry", "decoder")
CONTROL_SEEDS = tuple(range(20261113, 20261118))
NOTE = ("Exploratory OOF on inspected, archive-selected train sessions. Not independent validation, "
        "calibrated risk or full EPDMS. Candidate rows and overlapping training folds are correlated. "
        "Historical development and protected/fresh holdout are not evaluated.")


@dataclass(frozen=True)
class Config:
    width: int = 64
    epochs: int = 120
    learning_rate: float = .001
    weight_decay: float = .001
    seed: int = 20261113
    margin: float = .005
    neighbors: int = 5
    probes: int = 512
    bootstrap_reps: int = 10000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def seeded(value, seed):
    return int.from_bytes(hashlib.sha256((str(seed) + ":" + value).encode()).digest()[:8], "big")


def partition(metadata, seed=20261113):
    sessions = sorted({m["session_id"] for m in metadata},
                      key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    if len(sessions) < 8 or len(sessions) % 4:
        raise ValueError("Four equal groups require a multiple of four, at least eight sessions")
    return [sorted(sessions[i::4]) for i in range(4)]


def fit_specs(groups):
    """Six pair-group base fits and four triple-group base/corrector fits."""
    if len(groups) != 4 or len(set(sum(groups, []))) != sum(map(len, groups)):
        raise ValueError("Overlapping/invalid session groups")
    return [("base", tuple(c)) for n in (2, 3) for c in itertools.combinations(range(4), n)] + [
        ("corrector", tuple(i for i in range(4) if i != outer)) for outer in range(4)]


def input_entry(sample, vocab, metadata):
    """Return an outcome-free top64 view; identical features for all policies."""
    s = {k: np.asarray(sample[k]) for k in INPUT_KEYS}
    ids, scores = s["candidate_ids"], np.asarray(s["selection_scores"], dtype=float)
    b = int(s["selected_local"])
    if (metadata["role"] != "train" or ids.shape != (256,) or ids.dtype.kind not in "iu"
            or len(set(ids.tolist())) != 256 or ids.min() < 0 or ids.max() >= len(vocab)
            or scores.shape != (256,) or b != int(np.argmax(scores))
            or s["decoder_states"].shape != (256, 256)
            or any(s["head_" + k].shape != (256,) for k in (*METRICS, "imi"))
            or s["scene_embedding"].ndim != 1 or s["ego_status"].ndim != 1
            or not all(np.isfinite(a).all() for a in s.values())):
        raise ValueError("Invalid frozen training inputs/alignment")
    ix = np.argsort(-scores, kind="stable")[:64]
    base = int(np.flatnonzero(ix == b)[0])
    p, g = probabilities(s), geometry(vocab[ids])
    latent = np.asarray(s["decoder_states"], dtype=float)
    imi = np.asarray(s["head_imi"], dtype=float).copy()
    imi -= np.max(imi)
    imi -= np.log(np.exp(imi).sum())
    context = np.r_[normalize_embedding(s["scene_embedding"]), s["ego_status"]]
    parts = dict(decoder=np.column_stack([latent[ix], latent[ix] - latent[b]]),
                 geometry=np.column_stack([g[ix], g[ix] - g[b]]),
                 teacher=np.column_stack([p[ix], p[ix] - p[b], scores[ix], scores[ix] - scores[b],
                                          imi[ix], imi[ix] - imi[b]]),
                 context=np.broadcast_to(context, (64, len(context))))
    # Same order as trajectory_selector.inputs, with only the pool widened.
    slices, offset = {}, 0
    for name in ("decoder", "geometry", "teacher", "context"):
        slices[name] = (offset, offset + parts[name].shape[1])
        offset += parts[name].shape[1]
    x = np.column_stack([*(parts[k] for k in ("decoder", "geometry", "teacher", "context")), ix == b]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite candidate features")
    return dict(token=metadata["token"], session_id=metadata["session_id"], x=x,
                indices=ix.astype(np.int64), ids=ids[ix].astype(np.int64), baseline=base,
                blocks=slices, sampling_slice=metadata.get("sampling_slice", "unknown"))


def supervised(entry, raw):
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (64, 9) or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
        raise ValueError("Complete finite top64 raw labels in [0,1] required")
    q, b = raw[:, 0], entry["baseline"]
    harm = np.any(raw[:, 1:][:, SAFETY] + 1e-8 < raw[b, 1:][list(SAFETY)], axis=1)
    gain = q - q[b]
    return dict(entry, targets=dict(quality=q.astype(np.float32), gain=gain.astype(np.float32),
                                   good=((gain >= .005) & ~harm).astype(np.float32)))


def weights(entries):
    sessions = [e["session_id"] for e in entries]
    w = np.array([1 / sessions.count(s) for s in sessions], dtype=float)
    return w / w.sum()


def scaler(entries):
    x, w = np.stack([e["x"] for e in entries]).astype(float), weights(entries)
    mean = np.einsum("s,skd->d", w, x) / x.shape[1]
    std = np.sqrt(np.maximum(np.einsum("s,skd->d", w, (x - mean) ** 2) / x.shape[1], 0))
    std[std < 1e-6] = 1.
    return mean.astype(np.float32), std.astype(np.float32)


def train(entries, config, kind="base", residuals=None, progress=None):
    """Coherent-selector objective, or matched scalar residual correction."""
    import torch
    from torch.nn import functional as F
    from navsim.agents.trajectory_selector import Selector
    if kind not in ("base", "corrector") or not entries:
        raise ValueError("Invalid fit")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    mean, scale = scaler(entries)
    x = torch.from_numpy(np.clip((np.stack([e["x"] for e in entries]) - mean) / scale, -8, 8))
    base = torch.tensor([e["baseline"] for e in entries], dtype=torch.long)
    alternative = torch.ones(x.shape[:2], dtype=torch.bool).scatter_(1, base[:, None], False)
    w = torch.tensor(weights(entries), dtype=torch.float32)
    if kind == "base":
        q, gain, good = [torch.tensor(np.stack([e["targets"][k] for e in entries]), dtype=torch.float32)
                         for k in ("quality", "gain", "good")]
    else:
        gain = torch.tensor(np.stack([residuals[e["token"]] for e in entries]), dtype=torch.float32)
        if gain.shape != x.shape[:2] or not torch.isfinite(gain).all() or torch.any(gain.gather(1, base[:, None]) != 0):
            raise ValueError("Invalid cross-fitted residual targets")
    net = Selector(x.shape[-1], "mlp", config.width, 4)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = net(x, base)
        regression = F.smooth_l1_loss(pred["gain"], gain, reduction="none", beta=.05)
        if kind == "base":
            # Exactly trajectory_selector_v2's coherent gain/regret objective.
            prob = (pred["gain"] / .1).softmax(-1)
            regret = (prob * (q.max(1, keepdim=True).values - q)).sum(1)
            benefit = F.binary_cross_entropy_with_logits(pred["good_logit"], good, reduction="none")
            loss = ((5 * regret + ((5 * regression + .25 * benefit) * alternative).sum(1) / 63) * w).sum()
        else:
            loss = (((regression * alternative).sum(1) / 63) * w).sum()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 40 == 0 or epoch + 1 == config.epochs:
            item = dict(epoch=epoch + 1, loss=float(loss.detach()))
            history.append(item)
            if progress:
                progress(item)
    return dict(mean=mean, scale=scale, **{k: v.detach().numpy().copy() for k, v in net.state_dict().items()}), history


def predict(checkpoint, entries, config):
    """Only x and baseline are read: targets/outcomes cannot enter replay."""
    import torch
    from navsim.agents.trajectory_selector import Selector
    net = Selector(len(checkpoint["mean"]), "mlp", config.width, 4)
    net.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in checkpoint.items() if k not in ("mean", "scale")})
    net.eval()
    # Scene batches bound inference RAM and keep train/query APIs separate.
    result = []
    with torch.no_grad():
        for start in range(0, len(entries), 16):
            batch = entries[start:start + 16]
            x = np.clip((np.stack([e["x"] for e in batch]) - checkpoint["mean"]) / checkpoint["scale"], -8, 8)
            gain = net(torch.from_numpy(x), torch.tensor([e["baseline"] for e in batch]))["gain"].numpy()
            if not np.isfinite(gain).all() or np.any(gain[np.arange(len(batch)), [e["baseline"] for e in batch]] != 0):
                raise RuntimeError("Nonfinite prediction/nonzero baseline")
            result.extend(gain.astype(float))
    return np.asarray(result)


def choose(entry, gain, margin=.005):
    gain = np.asarray(gain, dtype=float)
    b = entry["baseline"]
    if gain.shape != (64,) or not np.isfinite(gain).all() or gain[b] != 0:
        raise ValueError("Invalid predicted gain/baseline")
    if gain.max() < margin:
        return b
    tied = np.flatnonzero(gain >= gain.max() - 1e-7)
    return b if b in tied else int(tied[np.argmin(entry["ids"][tied])])


def keys(entry, mean, scale):
    chunks = []
    for name in BLOCKS:
        a, b = entry["blocks"][name]
        chunks.append(np.clip((entry["x"][:, a:b].astype(float) - mean[a:b]) / scale[a:b], -8, 8)
                      / np.sqrt(4 * (b - a)))
    return np.column_stack(chunks)


class Memory:
    def __init__(self, entries, residuals, mean, scale):
        self.mean, self.scale = mean, scale
        records = []
        self.scene_entries = {e["token"]: e for e in entries}
        for e in entries:
            r = np.asarray(residuals[e["token"]], dtype=float)
            if r.shape != (64,) or not np.isfinite(r).all() or r[e["baseline"]] != 0:
                raise ValueError("Invalid donor residuals")
            z = keys(e, mean, scale)
            for p in range(64):
                if p != e["baseline"]:
                    records.append((e["session_id"], e["token"], int(e["ids"][p]), p, z[p], float(r[p])))
        records.sort(key=lambda row: row[:3])
        self.records = [dict(session_id=s, token=t, candidate_id=v, position=p) for s, t, v, p, _, _ in records]
        self.z = np.stack([r[4] for r in records])
        self.norm = np.einsum("ij,ij->i", self.z, self.z)
        self.residuals = np.array([r[5] for r in records])
        self.sessions = sorted({r[0] for r in records})
        self.by_session = {s: np.array([i for i, r in enumerate(records) if r[0] == s]) for s in self.sessions}

    def distances(self, query):
        """One nearest action per distinct eligible session. No residual read."""
        z = keys(query, self.mean, self.scale)
        # 64 x donor_count; never allocate query x donor x feature.
        d = np.maximum(np.einsum("ij,ij->i", z, z)[:, None] + self.norm - 2 * z @ self.z.T, 0)
        eligible = [s for s in self.sessions if s != query["session_id"]]
        index, distance = [], []
        for sid in eligible:
            ix = self.by_session[sid]
            pos = np.argmin(d[:, ix], axis=1)
            selected = ix[pos]
            index.append(selected)
            distance.append(d[np.arange(64), selected])
        if not eligible:
            return eligible, np.empty((64, 0), dtype=int), np.empty((64, 0))
        return eligible, np.asarray(index, dtype=int).T, np.asarray(distance, dtype=float).T

    def bandwidth(self, entries, config):
        # Round-robin per-session probes selected without consulting residuals.
        grouped = {}
        for e in entries:
            for p in range(64):
                if p != e["baseline"]:
                    grouped.setdefault(e["session_id"], []).append((e, p))
        for sid, rows in grouped.items():
            rows.sort(key=lambda a: seeded(a[0]["token"] + ":" + str(int(a[0]["ids"][a[1]])), config.seed))
        probes = [grouped[s][i] for i in range(max(map(len, grouped.values())))
                  for s in sorted(grouped) if i < len(grouped[s])][:config.probes]
        cache, values = {}, []
        for e, p in probes:
            if e["token"] not in cache:
                cache[e["token"]] = self.distances(e)[2]
            d = cache[e["token"]][p]
            positive = d[d > 1e-12]
            if len(positive):
                values.append(float(positive.min()))
        if not values:
            raise ValueError("No positive cross-session key distances for bandwidth")
        return float(np.median(values))

    def shuffled(self, seed):
        """Cross-session scene-vector permutation; preserve within-scene ranks."""
        groups = {}
        for e in self.scene_entries.values():
            groups.setdefault(e["session_id"], []).append(e)
        sessions = sorted(groups, key=lambda s: seeded(s, seed))
        ordered = [e for s in sessions for e in sorted(groups[s], key=lambda e: seeded(e["token"], seed))]
        shift = max(map(len, groups.values()))
        if shift * 2 > len(ordered):
            raise ValueError("Cannot permute scene vectors across sessions without changing the protocol")
        mapping = {e["token"]: ordered[(i + shift) % len(ordered)] for i, e in enumerate(ordered)}
        by_pair = {(r["token"], r["position"]): i for i, r in enumerate(self.records)}
        values, provenance = [], {}
        for r in self.records:
            source = mapping[r["token"]]
            if source["session_id"] == r["session_id"]:
                raise RuntimeError("Shuffle failed entire-session exclusion")
            target = self.scene_entries[r["token"]]
            rank = [p for p in range(64) if p != target["baseline"]].index(r["position"])
            p = [p for p in range(64) if p != source["baseline"]][rank]
            values.append(self.residuals[by_pair[(source["token"], p)]])
            provenance[r["token"]] = source["token"]
        return np.asarray(values), provenance

    def recall(self, query, h, config, mode="nearest", seed=None, values=None, distances=None):
        if not np.isfinite(h) or h <= 0 or mode not in ("nearest", "random"):
            raise ValueError("Invalid retrieval protocol")
        sessions, ix, d = self.distances(query) if distances is None else distances
        if query["session_id"] in sessions:
            raise RuntimeError("Own session present in donor distance table")
        k = min(config.neighbors, len(sessions))
        if not k:
            return np.zeros(64), dict(index=np.zeros((64, 0), dtype=int), distance=np.zeros((64, 0)), weight=np.zeros((64, 0)))
        if mode == "random":
            columns = np.random.default_rng(seeded(query["token"], seed)).permutation(len(sessions))[:k]
            columns = np.broadcast_to(columns, (64, k))
        else:
            columns = np.argsort(d, axis=1, kind="stable")[:, :k]
        idx, dist = np.take_along_axis(ix, columns, 1), np.take_along_axis(d, columns, 1)
        w = np.exp(-dist / h)
        r = self.residuals if values is None else np.asarray(values)
        if r.shape != self.residuals.shape or not np.isfinite(r).all():
            raise ValueError("Invalid shuffled residuals")
        result = (w * r[idx]).sum(1) / (1 + w.sum(1))
        result[query["baseline"]] = 0.
        w[query["baseline"]] = 0.
        return result, dict(index=idx, distance=dist, weight=w)


def config_dict(config):
    return asdict(config)
