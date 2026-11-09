"""Matched MLP experiment: legacy independent ranking vs coherent gain ranking.

Both networks have the same stem, initialization, parameter count, inputs, gain
and benefit readouts. In v2 the old independent rank row is unused: the gain
readout itself determines ordering. Benefit estimates are not calibrated risk.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents import trajectory_selector as original
from navsim.agents.drivemem import SAFETY
from navsim.agents.selector_training_bundle import LABEL_KEYS

FAMILIES = ("control_v1", "consistent_v2")
TEMPERATURE = .1
REGRET_WEIGHT = 5.
PROTOCOL = dict(temperature=TEMPERATURE, regret_weight=REGRET_WEIGHT, gain_weight=5., benefit_weight=.25,
                gain_beta=.05, independent_rank_readout_used=False)


class ConsistentSelector(original.Selector):
    def __init__(self, input_dim, width=64, heads=4):
        super().__init__(input_dim, "mlp", width, heads)

    def forward(self, x, baseline, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        result = super().forward(x, baseline, mask)
        result["rank"] = (result["gain"] / TEMPERATURE).masked_fill(~mask, -torch.inf)
        return result


def regret_loss(logits, quality, mask):
    """Expected observed PDM loss; a 0.9 loss costs more than a 0.002 loss."""
    probabilities = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    best = quality.masked_fill(~mask, -torch.inf).max(1, keepdim=True).values
    costs = (best - quality).masked_fill(~mask, 0)
    return (probabilities * costs).sum(1)


def entries_from_bundle(examples, vocab, challenge_tokens, config):
    entries = []
    for e in examples:
        meta = e["metadata"]
        if meta["role"] != "train":
            raise ValueError("Only bundled training scenes may fit")
        x, ix, b = original.inputs(e["sample"], vocab)
        if not np.array_equal(ix, e["label_indices"]):
            raise ValueError("Bundled labels and current top16 differ")
        raw = e["label_raw"]
        q = raw[:, 0]
        delta = q - q[b]
        harm = np.any(raw[:, 1:][:, SAFETY] + 1e-8 < raw[b, 1:][list(SAFETY)], axis=1)
        good = (delta >= config.improvement_min) & ~harm
        good[b] = False
        # Adapter for the frozen v1 trainer. Role has no feature/weight effect;
        # all scenes are train, and no donor retrieval occurs in this experiment.
        sample = dict(e["sample"], token=meta["token"], session_id=meta["session_id"], role="fit", baseline_position=b)
        outcomes = {str(int(local)): {version: {key: float(value) for key, value in zip(LABEL_KEYS, e["label_" + version][j])}
                                     for version in ("raw", "filtered")} for j, local in enumerate(ix)}
        entries.append(dict(sample=sample, metadata=meta, x=x, indices=ix, baseline=b, outcomes=outcomes,
            sampling_slice="challenge" if meta["token"] in challenge_tokens else "uniform",
            targets=dict(quality=q.astype(np.float32), gain=delta.astype(np.float32), good=good.astype(np.float32))))
    if not entries or len({e["x"].shape for e in entries}) != 1:
        raise ValueError("Empty/inconsistent selector training features")
    if len({e["metadata"]["token"] for e in entries}) != len(entries):
        raise ValueError("Duplicate training scenes")
    return sorted(entries, key=lambda e: (e["metadata"]["session_id"], e["metadata"]["token"]))


def train(entries, family, config, progress=None):
    xs, ys, samples = [e["x"] for e in entries], [e["targets"] for e in entries], [e["sample"] for e in entries]
    if family == "control_v1":
        return original.train(xs, ys, samples, "mlp", config, progress)
    if family != "consistent_v2" or any(s["role"] != "fit" for s in samples):
        raise ValueError("Invalid family or training membership")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    weights = original.session_weights(samples)
    mean, scale = original.fit_scaler(xs, weights)
    x = torch.from_numpy(original.standardize(xs, mean, scale))
    baseline = torch.tensor([s["baseline_position"] for s in samples], dtype=torch.long)
    mask = torch.ones(x.shape[:2], dtype=torch.bool)
    alternatives = mask.clone().scatter_(1, baseline[:, None], False)
    quality, gain, good = [torch.tensor(np.stack([y[k] for y in ys]), dtype=torch.float32) for k in ("quality", "gain", "good")]
    w = torch.tensor(weights, dtype=torch.float32)
    net = ConsistentSelector(x.shape[-1], config.width, config.heads)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    net.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = net(x, baseline, mask)
        regret = regret_loss(pred["rank"], quality, mask)
        regression = F.smooth_l1_loss(pred["gain"], gain, reduction="none", beta=.05)
        benefit = F.binary_cross_entropy_with_logits(pred["good_logit"], good, reduction="none")
        auxiliary = ((5 * regression + .25 * benefit) * alternatives).sum(1) / alternatives.sum(1)
        loss = ((REGRET_WEIGHT * regret + auxiliary) * w).sum()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite coherent-selector loss")
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimizer.step()
        if epoch in (0, config.epochs - 1):
            history.append(dict(epoch=epoch + 1, loss=float(loss.detach()),
                                expected_regret=float((regret.detach() * w).sum())))
        if progress is not None and (epoch + 1) % 40 == 0:
            progress(epoch + 1, float(loss.detach()))
    net.eval()
    return net, mean, scale, history


def predict(net, entries, mean, scale):
    return original.predict(net, [e["x"] for e in entries], [e["baseline"] for e in entries], mean, scale)


def restore(values, family, config):
    if family == "consistent_v2":
        net = ConsistentSelector(len(values["mean"]), config.width, config.heads)
    elif family == "control_v1":
        net = original.Selector(len(values["mean"]), "mlp", config.width, config.heads)
    else:
        raise ValueError("Unknown selector family")
    net.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in values.items() if k not in ("mean", "scale")}, strict=True)
    net.eval()
    return net, values["mean"], values["scale"]


def chosen_position(entry, prediction, row, family, threshold, config):
    ix, sample, b = entry["indices"], entry["sample"], entry["baseline"]
    position = original.choose(*(prediction[k][row] for k in ("rank", "gain", "good_logit")), b,
                               sample["candidate_ids"][ix], threshold, config.improvement_min)
    if family == "consistent_v2" and position != b and prediction["gain"][row, position] < -1e-7:
        raise RuntimeError("Coherent ranking selected a predicted negative-gain action")
    return position
