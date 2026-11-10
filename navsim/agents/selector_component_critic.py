"""Eight supervised metric predictions with a fixed local-PDM readout.

Measured scores remain authoritative labels. The product of predicted components
is a surrogate score, not a calibrated expectation of their joint outcome.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents import trajectory_selector as original
from navsim.agents import trajectory_selector_v2 as adapter
from navsim.agents.drivemem import METRICS

VARIANTS = ("linear", "mlp")
SEEDS = (20261108, 20261109, 20261110)
FORMULA = dict(multiplicative=list(METRICS[:4]), weighted={
    "ego_progress": 5., "time_to_collision_within_bound": 5.,
    "lane_keeping": 2., "history_comfort": 2.}, extended_comfort=False)


def pdm(components):
    """Same expression for NumPy arrays and Torch tensors; last axis is METRICS.

    Progress must already be normalized, as in stored raw metric labels. This
    function does not re-normalize candidates together or apply a human filter.
    """
    if components.shape[-1] != 8:
        raise ValueError("Eight ordered PDM components required")
    c = components
    gate = c[..., 0] * c[..., 1] * c[..., 2] * c[..., 3]
    return gate * (5 * c[..., 4] + 5 * c[..., 5] + 2 * c[..., 6] + 2 * c[..., 7]) / 14


@dataclass(frozen=True)
class ComponentConfig:
    width: int = 64
    epochs: int = 120
    learning_rate: float = .001
    weight_decay: float = .01
    component_weight: float = 1.
    delta_weight: float = 2.
    quality_weight: float = 5.
    gain_weight: float = 10.
    optimism_weight: float = 4.
    correction_penalty: float = .001

    def __post_init__(self):
        values = (self.learning_rate, self.weight_decay, self.component_weight, self.delta_weight,
                  self.quality_weight, self.gain_weight, self.optimism_weight, self.correction_penalty)
        if (self.width < 4 or self.epochs < 1 or not np.isfinite(values).all()
                or self.learning_rate <= 0 or self.weight_decay < 0 or min(values[2:]) < 0
                or self.component_weight <= 0):
            raise ValueError("Invalid component critic configuration")


def build_entries(examples, vocab, challenge, config):
    entries = adapter.entries_from_bundle(examples, vocab, challenge, original.SelectorConfig(epochs=config.epochs))
    lookup = {e["metadata"]["token"]: e for e in examples}
    for e in entries:
        e["targets"]["components"] = lookup[e["sample"]["token"]]["label_raw"][:, 1:].astype(np.float32)
        e["teacher_logits"] = np.column_stack([e["sample"]["head_" + m][e["indices"]] for m in METRICS]).astype(np.float32)
    return entries


def formula_audit(entries):
    mismatches, maxima, total, filtered_disagreements = [], {}, 0, 0
    for e in entries:
        if e["metadata"]["role"] != "train":
            raise ValueError("Formula audit accepts training scenes only")
        values = np.array([[e["outcomes"][str(int(local))]["raw"][key]
                            for key in ("pdm_score",) + METRICS] for local in e["indices"]], dtype=float)
        if values.shape != (16, 9) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Invalid stored raw labels")
        reconstructed = pdm(values[:, 1:])
        errors = np.abs(reconstructed - values[:, 0])
        origin = e["metadata"]["origin"]
        maxima[origin] = max(maxima.get(origin, 0.), float(errors.max()))
        for j in np.flatnonzero(errors > 1e-8):
            mismatches.append(dict(token=e["sample"]["token"], origin=origin,
                candidate_local=int(e["indices"][j]), stored_score=float(values[j, 0]),
                formula_score=float(reconstructed[j]), absolute_error=float(errors[j])))
        for j, local in enumerate(e["indices"]):
            filtered = e["outcomes"][str(int(local))]["filtered"]
            filtered_disagreements += bool(np.any(np.abs(values[j, 1:] - np.array([filtered[m] for m in METRICS])) > 1e-8))
        total += len(values)
    report = dict(status="passed" if not mismatches else "failed", formula=FORMULA,
        candidate_rows=total, mismatches=len(mismatches), max_abs_error=max(maxima.values(), default=0.),
        max_abs_error_by_origin=maxima, raw_filtered_component_disagreements=filtered_disagreements,
        note="Only RAW labels determine acceptance. Filtered fields never replace stored raw PDM.")
    return report, mismatches


def fit_statistics(entries):
    if not entries or any(e["metadata"]["role"] != "train" for e in entries):
        raise ValueError("Statistics require training scenes only")
    weights = original.session_weights([e["sample"] for e in entries])
    mean, scale = original.fit_scaler([e["x"] for e in entries], weights)
    support = np.array([len({e["sample"]["session_id"] for e in entries
        if np.any(e["targets"]["components"][:, j] + 1e-8 < e["targets"]["components"][e["baseline"], j])})
        for j in range(8)], dtype=np.int64)
    return dict(mean=mean, scale=scale, degradation_support_sessions=support)


class ComponentCritic(nn.Module):
    def __init__(self, input_dim, variant, config):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("Unknown component critic variant")
        if variant == "mlp":
            self.project = nn.Sequential(nn.Linear(input_dim, config.width), nn.LayerNorm(config.width), nn.GELU())
            self.norm = nn.LayerNorm(config.width)
            self.ff = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(), nn.Linear(4 * config.width, config.width))
        self.variant = variant
        self.readout = nn.Linear(input_dim if variant == "linear" else config.width, 8)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x, baseline, teacher_logits, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if (teacher_logits.shape != (*x.shape[:2], 8) or mask.shape != x.shape[:2]
                or bool(((baseline < 0) | (baseline >= x.shape[1])).any())
                or not bool(mask.gather(1, baseline[:, None]).all())):
            raise ValueError("Valid baseline, mask and teacher alignment required")
        h = x.masked_fill(~mask[..., None], 0)
        if self.variant == "mlp":
            h = self.project(h)
            h = h + self.ff(self.norm(h))
        correction = self.readout(h).masked_fill(~mask[..., None], 0)
        logits = teacher_logits.masked_fill(~mask[..., None], 0) + correction
        components = logits.sigmoid().masked_fill(~mask[..., None], 0)
        score = pdm(components)
        gain = score - score.gather(1, baseline[:, None])
        return dict(logits=logits, correction=correction, components=components, score=score,
                    gain=gain.masked_fill(~mask, 0))


def train(entries, variant, seed, config, progress=None):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    stats = fit_statistics(entries)
    weights = torch.tensor(original.session_weights([e["sample"] for e in entries]), dtype=torch.float32)
    x, baseline, teacher = input_tensors(entries, stats)
    target = torch.tensor(np.stack([e["targets"]["components"] for e in entries]), dtype=torch.float32)
    quality = torch.tensor(np.stack([e["targets"]["quality"] for e in entries]), dtype=torch.float32)
    gain = torch.tensor(np.stack([e["targets"]["gain"] for e in entries]), dtype=torch.float32)
    alternatives = torch.ones(x.shape[:2], dtype=torch.bool).scatter_(1, baseline[:, None], False)
    target_delta = target - target.gather(1, baseline[:, None, None].expand(-1, 1, 8))
    net = ComponentCritic(x.shape[-1], variant, config)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = net(x, baseline, teacher)
        absolute = F.binary_cross_entropy_with_logits(pred["logits"], target, reduction="none").mean((1, 2))
        delta = pred["components"] - pred["components"].gather(1, baseline[:, None, None].expand(-1, 1, 8))
        pair_loss = ((delta - target_delta).square().mean(-1) * alternatives).sum(1) / alternatives.sum(1)
        quality_loss = (pred["score"] - quality).square().mean(1)
        error = pred["gain"] - gain
        gain_loss = ((error.square() * (1 + config.optimism_weight * (error > 0))) * alternatives).sum(1) / alternatives.sum(1)
        penalty = pred["correction"].square().mean((1, 2))
        loss = ((config.component_weight * absolute + config.delta_weight * pair_loss
            + config.quality_weight * quality_loss + config.gain_weight * gain_loss
            + config.correction_penalty * penalty) * weights).sum()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite component training loss")
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


def input_tensors(entries, stats):
    # Inference allowlist. No measured labels, support statistics or outcomes.
    return (torch.from_numpy(original.standardize([e["x"] for e in entries], stats["mean"], stats["scale"])),
            torch.tensor([e["baseline"] for e in entries], dtype=torch.long),
            torch.tensor(np.stack([e["teacher_logits"] for e in entries]), dtype=torch.float32))


def predict(net, entries, stats):
    net.eval()
    with torch.no_grad():
        outputs = net(*input_tensors(entries, stats))
    result = {k: outputs[k].numpy() for k in ("components", "score", "gain")}
    validate(result, entries)
    return result


def validate(prediction, entries):
    shapes = dict(components=(len(entries), 16, 8), score=(len(entries), 16), gain=(len(entries), 16))
    if any(prediction[k].shape != shape or not np.isfinite(prediction[k]).all() for k, shape in shapes.items()):
        raise RuntimeError("Invalid component prediction shape/value")
    c, score, gain = (prediction[k] for k in ("components", "score", "gain"))
    b = [e["baseline"] for e in entries]
    if np.any((c < 0) | (c > 1)) or np.any(gain[np.arange(len(entries)), b] != 0):
        raise RuntimeError("Component bounds or baseline gain violated")
    if not np.allclose(pdm(c), score, atol=1e-6, rtol=0):
        raise RuntimeError("Component formula consistency failed")
    if not np.allclose(score - score[np.arange(len(entries)), b, None], gain, atol=1e-6, rtol=0):
        raise RuntimeError("Component gain consistency failed")


def aggregate(predictions, entries):
    # Formula(mean components), not an unrelated scalar mean with a free gain.
    components = np.stack([p["components"] for p in predictions]).astype(float).mean(0)
    score = pdm(components)
    b = [e["baseline"] for e in entries]
    gain = score - score[np.arange(len(entries)), b, None]
    result = dict(components=components, score=score, gain=gain,
        gain_lower=gain - np.stack([p["gain"] for p in predictions]).astype(float).std(0))
    validate(result, entries)
    return result


def choose(entry, prediction, row, policy):
    if policy not in ("raw", "gain_m020"):
        raise ValueError("Unknown component selection policy")
    gain = prediction["gain"][row]
    eligible = (prediction["gain_lower"][row] if policy == "gain_m020" else gain) >= (.02 if policy == "gain_m020" else 0.)
    b = entry["baseline"]
    eligible[b] = True
    score = np.where(eligible, gain, -np.inf)
    tied = np.flatnonzero(score >= score.max() - 1e-7)
    ids = entry["sample"]["candidate_ids"][entry["indices"]]
    chosen = b if b in tied else int(tied[np.argmin(ids[tied])])
    return chosen


def checkpoint(net, stats):
    return dict(stats, **{k: v.detach().numpy() for k, v in net.state_dict().items()})


def restore(arrays, variant, config):
    names = ("mean", "scale", "degradation_support_sessions")
    stats = {k: arrays[k] for k in names}
    net = ComponentCritic(len(stats["mean"]), variant, config)
    net.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in arrays.items() if k not in names}, strict=True)
    net.eval()
    return net, stats
