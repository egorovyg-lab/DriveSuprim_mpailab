"""Bounded correction to the teacher score and five degradation classifiers.

All statistics are fit on TRAIN sessions. Class-weighted risk estimates and
seed disagreement are diagnostics, not calibrated probabilities/bounds.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents import trajectory_selector as original
from navsim.agents import trajectory_selector_v2 as data_adapter
from navsim.agents.drivemem import SAFETY

VARIANTS = ("linear", "mlp")
SEEDS = (20261108, 20261109, 20261110)
RISK_LIMITS = (.05, .15, .30)
MARGINS = (.005, .02)


@dataclass(frozen=True)
class CriticConfig:
    width: int = 64
    epochs: int = 120
    learning_rate: float = .001
    weight_decay: float = .01
    residual_cap: float = .1
    minimum_support_sessions: int = 2
    positive_weight_cap: float = 30.
    regression_weight: float = 10.
    optimism_weight: float = 4.
    harmful_weight: float = 4.
    correction_penalty: float = .1

    def __post_init__(self):
        values = (self.learning_rate, self.weight_decay, self.residual_cap, self.positive_weight_cap,
                  self.regression_weight, self.optimism_weight, self.harmful_weight, self.correction_penalty)
        if (not np.isfinite(values).all() or self.width < 4 or self.epochs < 1 or self.learning_rate <= 0
                or self.weight_decay < 0 or not 0 < self.residual_cap <= .5 or self.minimum_support_sessions < 1
                or min(self.positive_weight_cap, self.regression_weight) < 1
                or min(self.optimism_weight, self.harmful_weight, self.correction_penalty) < 0):
            raise ValueError("Invalid residual critic configuration")


def build_entries(examples, vocab, challenge_tokens, config):
    entries = data_adapter.entries_from_bundle(examples, vocab, challenge_tokens,
                original.SelectorConfig(epochs=config.epochs))
    lookup = {e["metadata"]["token"]: e for e in examples}
    for e in entries:
        raw = lookup[e["sample"]["token"]]["label_raw"]
        if raw.shape != (16, 9) or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
            raise ValueError("Invalid stored raw labels")
        components = raw[:, 1:][:, list(SAFETY)]
        e["targets"]["risk"] = (components + 1e-8 < components[e["baseline"]]).astype(np.float32)
        scores = np.asarray(e["sample"]["selection_scores"], dtype=np.float64)[e["indices"]]
        e["teacher_gap"] = (scores - scores[e["baseline"]]).astype(np.float32)
    return entries


def policies():
    result = dict(raw=dict(margin=0., risk_limit=None, uncertainty=False))
    for margin in MARGINS:
        name = f"gain_m{round(1000 * margin):03d}"
        result[name] = dict(margin=margin, risk_limit=None, uncertainty=True)
        for limit in RISK_LIMITS:
            result[f"risk_t{round(100 * limit):02d}_m{round(1000 * margin):03d}"] = dict(
                margin=margin, risk_limit=limit, uncertainty=True)
    return result


def fit_statistics(entries, config):
    if not entries or any(e["metadata"]["role"] != "train" for e in entries):
        raise ValueError("Statistics require training scenes only")
    w = original.session_weights([e["sample"] for e in entries])
    mean, scale = original.fit_scaler([e["x"] for e in entries], w)
    gap = np.stack([e["teacher_gap"] for e in entries]).astype(float)
    gain = np.stack([e["targets"]["gain"] for e in entries]).astype(float)
    risk = np.stack([e["targets"]["risk"] for e in entries]).astype(float)
    mask = np.ones(gap.shape, dtype=bool)
    mask[np.arange(len(entries)), [e["baseline"] for e in entries]] = False
    weight = w[:, None] * mask / mask.sum(1)[:, None]
    denominator = float((weight * gap ** 2).sum())
    slope = float(np.clip((weight * gap * gain).sum() / max(denominator, 1e-12), 0, 1))
    positive = (risk * weight[..., None]).sum((0, 1))
    positive_weight = np.clip((1 - positive) / np.maximum(positive, 1e-8), 1, config.positive_weight_cap)
    support = np.array([len({e["sample"]["session_id"] for e in entries
                             if np.any(e["targets"]["risk"][:, j])}) for j in range(5)], dtype=np.int64)
    # Smoothed, session-weighted training priors initialize classifiers.
    prior = np.clip(positive, 1e-3, 1 - 1e-3)
    return dict(mean=mean, scale=scale, prior_slope=np.array(slope),
                positive_weight=positive_weight.astype(np.float32), risk_support_sessions=support,
                risk_bias=np.log(prior / (1 - prior)).astype(np.float32))


class ResidualCritic(nn.Module):
    def __init__(self, input_dim, variant, config):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("Unknown critic variant")
        self.variant, self.cap = variant, config.residual_cap
        if variant == "mlp":
            self.project = nn.Sequential(nn.Linear(input_dim, config.width), nn.LayerNorm(config.width), nn.GELU())
            self.norm = nn.LayerNorm(config.width)
            self.ff = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(), nn.Linear(4 * config.width, config.width))
        self.readout = nn.Linear(input_dim if variant == "linear" else config.width, 6)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x, baseline, teacher_gap, prior_slope, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if not bool(mask.gather(1, baseline[:, None]).all()):
            raise ValueError("Valid baseline required")
        h = x.masked_fill(~mask[..., None], 0)
        if self.variant == "mlp":
            h = self.project(h)
            h = h + self.ff(self.norm(h))
        output = self.readout(h)
        correction = self.cap * output[..., 0].tanh()
        # Baseline correction is defined as zero. Subtracting another bounded
        # head value couples every action to baseline saturation and can suppress
        # good actions while trying to fit an unrepresentable catastrophic loss.
        correction = correction.scatter(1, baseline[:, None], 0.)
        prior = (prior_slope * teacher_gap.masked_fill(~mask, 0)).clamp(-1, 0)
        gain = prior + correction
        return dict(gain=gain.masked_fill(~mask, 0), correction=correction.masked_fill(~mask, 0),
                    risk_logits=output[..., 1:].masked_fill(~mask[..., None], 0))


def train(entries, variant, seed, config, progress=None):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    stats = fit_statistics(entries, config)
    weights = torch.tensor(original.session_weights([e["sample"] for e in entries]), dtype=torch.float32)
    x = torch.from_numpy(original.standardize([e["x"] for e in entries], stats["mean"], stats["scale"]))
    baseline = torch.tensor([e["baseline"] for e in entries], dtype=torch.long)
    gap = torch.tensor(np.stack([e["teacher_gap"] for e in entries]), dtype=torch.float32)
    target = torch.tensor(np.stack([e["targets"]["gain"] for e in entries]), dtype=torch.float32)
    risk = torch.tensor(np.stack([e["targets"]["risk"] for e in entries]), dtype=torch.float32)
    alternatives = torch.ones(x.shape[:2], dtype=torch.bool).scatter_(1, baseline[:, None], False)
    positive_weight = torch.from_numpy(stats["positive_weight"])
    net = ResidualCritic(x.shape[-1], variant, config)
    with torch.no_grad():
        net.readout.bias[1:].copy_(torch.from_numpy(stats["risk_bias"]))
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = net(x, baseline, gap, float(stats["prior_slope"]))
        error = prediction["gain"] - target
        regression = error.square() * (1 + config.optimism_weight * (error > 0)) * (
            1 + config.harmful_weight * risk.any(-1))
        classification = F.binary_cross_entropy_with_logits(prediction["risk_logits"], risk,
                             pos_weight=positive_weight, reduction="none").mean(-1)
        per_candidate = config.regression_weight * regression + classification + config.correction_penalty * prediction["correction"].square()
        loss = (((per_candidate * alternatives).sum(1) / alternatives.sum(1)) * weights).sum()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite critic loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 40 == 0 or epoch + 1 == config.epochs:
            item = dict(epoch=epoch + 1, loss=float(loss.detach()), gradient_norm=float(norm.detach()))
            history.append(item)
            if progress:
                progress(item)
    net.eval()
    return net, stats, history


def predict(net, entries, stats):
    # Explicit input whitelist: validation targets are never accessed here.
    x = torch.from_numpy(original.standardize([e["x"] for e in entries], stats["mean"], stats["scale"]))
    baseline = torch.tensor([e["baseline"] for e in entries], dtype=torch.long)
    gap = torch.tensor(np.stack([e["teacher_gap"] for e in entries]), dtype=torch.float32)
    net.eval()
    with torch.no_grad():
        result = {k: v.numpy() for k, v in net(x, baseline, gap, float(stats["prior_slope"])).items()}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise RuntimeError("Nonfinite critic prediction")
    if not np.array_equal(result["gain"][np.arange(len(entries)), baseline.numpy()], np.zeros(len(entries))):
        raise RuntimeError("Baseline gain must be exactly zero")
    if np.max(np.abs(result["correction"])) > net.cap + 1e-6:
        raise RuntimeError("Residual bound violated")
    return result


def checkpoint(net, stats):
    return dict(stats, **{k: v.detach().numpy() for k, v in net.state_dict().items()})


def restore(arrays, variant, config):
    names = ("mean", "scale", "prior_slope", "positive_weight", "risk_support_sessions", "risk_bias")
    stats = {k: arrays[k].copy() for k in names}
    net = ResidualCritic(len(stats["mean"]), variant, config)
    net.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in arrays.items() if k not in names}, strict=True)
    net.eval()
    return net, stats


def aggregate(predictions):
    gains = np.stack([p["gain"] for p in predictions]).astype(float)
    logits = np.stack([p["risk_logits"] for p in predictions]).astype(float)
    risk = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
    return dict(gain=gains.mean(0), gain_lower=gains.mean(0) - gains.std(0), risk=risk.max(0))


def choose(entry, prediction, row, policy, support, config):
    mean, lower, risk = (prediction[k][row] for k in ("gain", "gain_lower", "risk"))
    if not all(np.isfinite(a).all() for a in (mean, lower, risk)) or risk.shape != (len(mean), 5):
        raise ValueError("Invalid policy predictions")
    b = entry["baseline"]
    if mean[b] != 0 or lower[b] != 0:
        raise ValueError("Nonzero baseline prediction")
    eligible = (lower if policy["uncertainty"] else mean) >= policy["margin"]
    unsupported = False
    if policy["risk_limit"] is not None:
        unsupported = bool(np.any(np.asarray(support) < config.minimum_support_sessions))
        eligible &= (risk <= policy["risk_limit"]).all(-1)
        if unsupported:
            eligible[:] = False
    eligible[b] = True
    scores = np.where(eligible, mean, -np.inf)
    tied = np.flatnonzero(scores >= scores.max() - 1e-7)
    ids = entry["sample"]["candidate_ids"][entry["indices"]]
    chosen = b if b in tied else int(tied[np.argmin(ids[tied])])
    return chosen, dict(eligible_alternatives=int(eligible.sum()) - 1, support_blocked=unsupported,
        predicted_gain_pp=100 * float(mean[chosen]), gain_lower_pp=100 * float(lower[chosen]),
        predicted_risk_max=float(risk[chosen].max()) if chosen != b else 0.)
