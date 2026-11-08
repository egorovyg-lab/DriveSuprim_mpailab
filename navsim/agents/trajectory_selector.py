"""Small supervised top-16 selectors. No retrieval, RGB encoder, or query labels.

Network scores express relative preference. The benefit head is an uncalibrated
estimate of improvement without component degradation, not a safety guarantee.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents.drivemem import METRICS, SAFETY, geometry, normalize_embedding, probabilities
from navsim.agents.drivemem_action_transfer import pool


@dataclass(frozen=True)
class SelectorConfig:
    width: int = 64
    heads: int = 4
    epochs: int = 120
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    tie_epsilon: float = 0.001  # PDM units; 0.1 percentage points
    improvement_min: float = 0.005  # PDM units; 0.5 percentage points
    seed: int = 20261108

    def __post_init__(self):
        if self.width < 4 or self.heads < 1 or self.width % self.heads:
            raise ValueError("Width must be positive and divisible by heads")
        if self.epochs < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid training parameters")
        if not 0 <= self.tie_epsilon < self.improvement_min <= 1:
            raise ValueError("Require 0 <= tie epsilon < minimum improvement <= 1")


def inputs(sample, vocab):
    """Whitelist inference inputs. Neither outcome fields nor IDs become features."""
    ids = np.asarray(sample["candidate_ids"])
    scores = np.asarray(sample["selection_scores"], dtype=float)
    baseline = int(sample["selected_local"])
    latent = np.asarray(sample["decoder_states"], dtype=float)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(set(ids.tolist())) != len(ids)
            or len(ids) < 16 or ids.min() < 0 or ids.max() >= len(vocab)):
        raise ValueError("Invalid candidate vocabulary IDs")
    if (scores.shape != ids.shape or not np.isfinite(scores).all()
            or not 0 <= baseline < len(ids) or scores[baseline] != scores.max()):
        raise ValueError("Invalid baseline or selection scores")
    if latent.ndim != 2 or latent.shape[0] != len(ids):
        raise ValueError("Decoder/candidate alignment mismatch")
    ix = pool(sample)
    if baseline not in ix:
        raise ValueError("Baseline must be in frozen top16")
    b = int(np.flatnonzero(ix == baseline)[0])
    p = probabilities(sample)
    g = geometry(vocab[ids])
    ego = np.asarray(sample["ego_status"], dtype=float).reshape(-1)
    context = np.r_[normalize_embedding(sample["scene_embedding"]), ego]
    imi = np.asarray(sample["head_imi"], dtype=float)
    imi = imi - np.max(imi)
    imi = imi - np.log(np.exp(imi).sum())
    x = np.column_stack([
        latent[ix], latent[ix] - latent[baseline],
        g[ix], g[ix] - g[baseline], p[ix], p[ix] - p[baseline],
        scores[ix], scores[ix] - scores[baseline], imi[ix], imi[ix] - imi[baseline],
        np.broadcast_to(context, (len(ix), len(context))), ix == baseline,
    ]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite inference features")
    return x, ix.astype(np.int64), b


def targets(sample, outcomes, config):
    """Training-only targets; preserve the scorer's stored PDM, never reconstruct it."""
    if sample["role"] not in ("memory", "fit"):
        raise ValueError("Only M/F outcomes may train the selector")
    ix = pool(sample)
    b = list(ix).index(int(sample["selected_local"]))
    if any(str(int(i)) not in outcomes for i in ix):
        raise ValueError("Complete top16 labels required")
    values = np.asarray([[outcomes[str(int(i))]["raw"][k]
                          for k in ("pdm_score",) + METRICS] for i in ix], dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid stored PDM/component labels")
    quality = values[:, 0]
    gain = quality - quality[b]
    harm = np.any(values[:, 1:][:, SAFETY] + 1e-8 < values[b, 1:][list(SAFETY)], axis=1)
    good = (gain >= config.improvement_min) & ~harm
    good[b] = False
    return dict(quality=quality.astype(np.float32), gain=gain.astype(np.float32),
                good=good.astype(np.float32), harm=harm, components=values[:, 1:])


def session_weights(samples):
    sessions = [s["session_id"] for s in samples]
    weights = np.asarray([1 / sessions.count(sid) for sid in sessions], dtype=np.float64)
    return weights / weights.sum()


def fit_scaler(xs, weights):
    """Equal session mass; callers pass TRAIN folds only."""
    x = np.stack(xs).astype(np.float64)
    w = np.asarray(weights, dtype=float)
    mean = np.einsum("s,skd->d", w, x) / x.shape[1]
    variance = np.einsum("s,skd->d", w, (x - mean) ** 2) / x.shape[1]
    scale = np.sqrt(np.maximum(variance, 0))
    scale[scale < 1e-6] = 1
    return mean.astype(np.float32), scale.astype(np.float32)


def standardize(xs, mean, scale):
    return np.clip((np.asarray(xs, dtype=np.float32) - mean) / scale, -8, 8)


class Selector(nn.Module):
    """Shared candidate scorer or permutation-equivariant set attention block.

    The MLP's expansion is chosen to match the attention block's parameter count.
    There are no candidate-slot, vocabulary-ID, token, or session embeddings.
    """
    def __init__(self, input_dim, family, width=64, heads=4):
        super().__init__()
        if family not in ("mlp", "attention") or width % heads:
            raise ValueError("Invalid selector family/dimensions")
        self.family, self.width, self.heads = family, width, heads
        self.project = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
        self.norm = nn.LayerNorm(width)
        if family == "attention":
            self.qkv = nn.Linear(width, 3 * width)
            self.attention_out = nn.Linear(width, width)
            self.ff_norm = nn.LayerNorm(width)
            expansion = 2 * width
        else:
            expansion = 4 * width
        self.ff = nn.Sequential(nn.Linear(width, expansion), nn.GELU(), nn.Linear(expansion, width))
        self.readout = nn.Linear(width, 3)

    def forward(self, x, baseline, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if (not bool(mask.any(dim=1).all()) or not bool(mask.gather(1, baseline[:, None]).all())):
            raise ValueError("Every set needs a valid baseline")
        h = self.project(x.masked_fill(~mask[..., None], 0))
        if self.family == "attention":
            batch, count, _ = h.shape
            qkv = self.qkv(self.norm(h)).reshape(batch, count, 3, self.heads, self.width // self.heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attention = (q @ k.transpose(-2, -1)) / (self.width // self.heads) ** .5
            attention = attention.masked_fill(~mask[:, None, None, :], -torch.inf).softmax(-1)
            mixed = (attention @ v).transpose(1, 2).reshape(batch, count, self.width)
            h = h + self.attention_out(mixed)
            h = h + self.ff(self.ff_norm(h))
        else:
            h = h + self.ff(self.norm(h))
        out = self.readout(h)
        gain = out[..., 1] - out[..., 1].gather(1, baseline[:, None])
        return dict(rank=out[..., 0].masked_fill(~mask, -torch.inf),
                    gain=gain.masked_fill(~mask, 0), good_logit=out[..., 2].masked_fill(~mask, 0))


def ranking_loss(logits, quality, mask, epsilon):
    """Assign mass to ANY near-best action. All-equal sets have zero rank loss."""
    good = (quality >= quality.masked_fill(~mask, -torch.inf).max(1, keepdim=True).values - epsilon) & mask
    logp = logits.masked_fill(~mask, -torch.inf).log_softmax(-1)
    return -torch.logsumexp(logp.masked_fill(~good, -torch.inf), dim=1)


def train(xs, labels, samples, family, config, progress=None):
    if any(s["role"] not in ("memory", "fit") for s in samples):
        raise ValueError("Training refuses non-M/F samples")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    weights = session_weights(samples)
    mean, scale = fit_scaler(xs, weights)
    x = torch.from_numpy(standardize(xs, mean, scale))
    baseline = torch.tensor([s["baseline_position"] for s in samples], dtype=torch.long)
    mask = torch.ones(x.shape[:2], dtype=torch.bool)
    alternatives = mask.clone().scatter_(1, baseline[:, None], False)
    q, gain, good = [torch.tensor(np.stack([y[k] for y in labels]), dtype=torch.float32)
                     for k in ("quality", "gain", "good")]
    w = torch.tensor(weights, dtype=torch.float32)
    model = Selector(x.shape[-1], family, config.width, config.heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    model.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, baseline, mask)
        rank = ranking_loss(pred["rank"], q, mask, config.tie_epsilon)
        regression = F.smooth_l1_loss(pred["gain"], gain, reduction="none", beta=.05)
        benefit = F.binary_cross_entropy_with_logits(pred["good_logit"], good, reduction="none")
        denominator = alternatives.sum(1)
        aux = ((5 * regression + .25 * benefit) * alternatives).sum(1) / denominator
        loss = ((rank + aux) * w).sum()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if epoch in (0, config.epochs - 1):
            history.append(dict(epoch=epoch + 1, loss=float(loss.detach())))
        if progress is not None and (epoch + 1) % 40 == 0:
            progress(epoch + 1, float(loss.detach()))
    model.eval()
    return model, mean, scale, history


def predict(model, xs, baseline, mean, scale):
    model.eval()
    with torch.no_grad():
        result = model(torch.from_numpy(standardize(xs, mean, scale)), torch.tensor(baseline, dtype=torch.long))
    return {k: v.numpy() for k, v in result.items()}


def choose(rank, gain, good_logit, baseline, candidate_ids, threshold=None, improvement_min=.005):
    """All scores are predictions; thresholds confer no calibrated risk bound."""
    rank, gain, good_logit = [np.asarray(a, dtype=float) for a in (rank, gain, good_logit)]
    if not all(np.isfinite(a).all() for a in (rank, gain, good_logit)):
        raise ValueError("Nonfinite selector prediction")
    tied = np.flatnonzero(rank >= rank.max() - 1e-7)
    chosen = baseline if baseline in tied else int(tied[np.argmin(np.asarray(candidate_ids)[tied])])
    if threshold is not None and chosen != baseline:
        benefit = 1 / (1 + np.exp(-np.clip(good_logit[chosen], -700, 700)))
        if gain[chosen] < improvement_min or benefit < threshold:
            chosen = baseline
    return chosen


def checkpoint(model, mean, scale):
    return dict(mean=mean, scale=scale, **{k: v.detach().numpy() for k, v in model.state_dict().items()})


def restore(arrays, family, config):
    model = Selector(len(arrays["mean"]), family, config.width, config.heads)
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in arrays.items() if k not in ("mean", "scale")}, strict=True)
    model.eval()
    return model, arrays["mean"], arrays["scale"]
