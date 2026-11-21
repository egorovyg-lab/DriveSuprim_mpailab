"""Frozen R34 probes for the DAC pilot. Projected points in, OOF decisions out."""
import argparse
from collections import OrderedDict
import csv
import json
from pathlib import Path
import re

import numpy as np

from navsim.agents import drivemem_residual_memory as model
from navsim.planning.script import run_drivemem_residual_memory as io

METHODS = ("aligned", "global", "shuffled")
NEEDED = .60
HIGH, LOW = .90, .10
SAMPLE = 256
STRIDE = 16
DAC = 1 + model.METRICS.index("drivable_area_compliance")
NOTE = ("Fixed train sessions, map labels are supervision/evaluation, no map at prediction. "
        "Current camera projection is a FOV bound; hidden points remain unknown. "
        "Scores and the observed-coverage replacement rule provide no calibrated risk bound.")


def read_manifest(data):
    m = io.read(data/"manifest.json")
    tokens = [x["token"] for x in m["cohort"]]
    expected = {f"{section}/{token}.npz" for section in ("inputs", "labels") for token in tokens} | {"r34.npz"}
    if (m["protocol"] != "drivemem_spatial_dac_r34_v1" or m["images"] != ["cam_l0", "cam_f0", "cam_r0"]
            or len(set(tokens)) != len(tokens) or any(not re.fullmatch("[0-9a-f]{16}", t) for t in tokens)
            or set(m["files"]) != expected or m["groups"] != model.partition(m["cohort"])):
        raise ValueError("Invalid frozen spatial DAC experiment manifest")
    return m


def read_arrays(root, manifest, name):
    path = root/name
    if io.sha(path) != manifest["files"][name]:
        raise ValueError("Portable artifact changed: " + name)
    return io.load_npz(path)


def frozen_encoder(weights, device):
    import torch
    from torchvision.models import resnet34

    network = resnet34(weights=None)
    names = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3")
    encoder = torch.nn.Sequential(OrderedDict((name, getattr(network, name)) for name in names))
    if set(encoder.state_dict()) != set(weights):
        raise ValueError("Frozen teacher keys differ from the expected R34 stages")
    encoder.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()}, strict=True)
    return encoder.eval().requires_grad_(False).to(device)


def camera_image(images, record, order, size):
    from PIL import Image

    crops = []
    for name in order:
        spec = record["images"][name]
        path = io.safe_child(images, spec["relative"])
        if io.sha(path) != spec["sha256"]:
            raise ValueError(f"Current camera bytes differ from Astra: {spec['relative']}")
        with Image.open(path) as photo:
            if list(photo.size) != spec["size"]:
                raise ValueError("Current camera dimensions changed")
            padding = 0 if name == "cam_f0" else 416
            if photo.width <= 2*padding or photo.height <= 56:
                raise ValueError("Image too small for the DriveSuprim crop")
            crops.append(photo.convert("RGB").crop((padding, 28, photo.width-padding, photo.height-28)))
    if len({p.height for p in crops}) != 1:
        raise ValueError("Camera crop heights differ")
    pano = Image.new("RGB", (sum(p.width for p in crops), crops[0].height))
    offset = 0
    for crop in crops:
        pano.paste(crop, (offset, 0))
        offset += crop.width
    # Fixed pixel-centre-aligned RGB resize for every arm of the pilot.
    interpolation = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(pano.resize(tuple(size), interpolation)).copy()


def export(data, images, out, manifest, indices, device):
    import PIL
    import torch
    import torch.nn.functional as F
    import torchvision

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Use a GPU allocation for export or pass --device cpu")
    if out.is_relative_to(images) or images.is_relative_to(out):
        raise ValueError("Feature output must be separate from frozen inputs")
    with io.locked(out):
        identity = dict(manifest=io.sha(data/"manifest.json"), code=io.sha(__file__),
                        device=device, torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                        gpu=torch.cuda.get_device_name() if device == "cuda" else None,
                        pillow=PIL.__version__)
        io.freeze(out/"feature_identity.json", identity)
        hashes = io.read(out/"features.json") if (out/"features.json").exists() else {}
        encoder = None
        for i in indices:
            row = manifest["cohort"][i]
            token = row["token"]
            name = f"features/{token}.npz"
            if name in hashes:
                if io.sha(out/name) != hashes[name]:
                    raise ValueError("Feature cache changed: " + name)
                continue
            if encoder is None:
                encoder = frozen_encoder(read_arrays(data, manifest, "r34.npz"), device)
            entry = read_arrays(data, manifest, f"inputs/{token}.npz")
            observe = entry["view"] >= 0
            pixels = camera_image(images, row, manifest["images"], manifest["image_size"])
            with torch.inference_mode():
                x = torch.from_numpy(pixels).permute(2, 0, 1)[None].to(device, torch.float32) / 255
                grid = encoder(x)
                dims = torch.tensor([grid.shape[-1]-1, grid.shape[-2]-1], device=device).clamp_min(1)
                points = torch.from_numpy(entry["uv"][observe]).to(device=device, dtype=grid.dtype)
                # ResNet padded stride-16 sampling centres are 0,16,... in image coordinates.
                points = 2 * points / (STRIDE*dims) - 1
                # Fixed 3x3 feature neighbourhood with the query at its centre.
                blurred = F.avg_pool2d(grid, 3, stride=1, padding=1, count_include_pad=False)
                if len(points):
                    aligned = F.grid_sample(blurred, points[None, :, None], align_corners=True,
                                            padding_mode="border")[0, :, :, 0].T
                else:
                    aligned = grid.new_empty((0, grid.shape[1]))
                pooled = blurred[0].mean((1, 2))
                aligned = aligned.to(torch.float16)
            if not torch.isfinite(aligned).all() or not torch.isfinite(pooled).all():
                raise ValueError("Nonfinite image features")
            io.save(out/name, aligned=aligned.cpu().numpy(),
                    global_pool=pooled.cpu().numpy().astype(np.float32))
            hashes[name] = io.sha(out/name)
            io.write(out/"features.json", hashes)
            if len(hashes) % 25 == 0 or len(indices) <= 4:
                print(f"FEATURES {len(hashes)}/{len(manifest['cohort'])}", flush=True)
        return dict(status="exported", scenes=len(hashes), total=len(manifest["cohort"]))


def features(entry, saved, token, method):
    mask = entry["view"] >= 0
    candidate, time, corner = np.nonzero(mask)
    patch = saved["aligned"]
    if method == "global":
        patch = np.broadcast_to(saved["global_pool"], patch.shape)
    elif method == "shuffled":
        perm = np.random.default_rng(model.seeded(token, 20261118)).permutation(len(patch))
        patch = patch[perm]
    elif method != "aligned":
        raise ValueError("Unknown probe")
    return np.column_stack((patch, entry["decoder"][candidate], entry["teacher_dac"][candidate],
                            entry["ego_xy"][mask], entry["uv"][mask], entry["times"][time],
                            np.eye(4, dtype=np.float32)[corner],
                            np.eye(3, dtype=np.float32)[entry["view"][mask]])).astype(np.float32)


def training_rows(targets, rng):
    """Include rare outsides plus inside points from the same training scene."""
    pos = np.flatnonzero(targets)
    neg = np.flatnonzero(~targets)
    cap = SAMPLE//2 if len(neg) else SAMPLE
    chosen_pos = rng.choice(pos, min(len(pos), cap), replace=False)
    chosen_neg = rng.choice(neg, min(len(neg), SAMPLE-len(chosen_pos)), replace=False)
    return np.sort(np.r_[chosen_pos, chosen_neg])


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def fit(matrix, labels, weights):
    """Deterministic L2 logistic probe with train-only normalization."""
    if not labels.any() or labels.all():
        raise ValueError("This training fold needs both inside and outside points")
    weights = weights / weights.sum()
    mean = weights @ matrix
    sd = np.sqrt(weights @ np.square(matrix-mean))
    sd = np.maximum(sd, .01)
    x = np.clip((matrix-mean)/sd, -6, 6)
    # Balanced BCE score for rare events. This score is not a probability estimate.
    factor = np.clip((weights @ ~labels) / (weights @ labels), 1., 20.)
    w = weights*np.where(labels, factor, 1.)
    w = (w/w.sum()).astype(np.float32)
    parameters = np.zeros(x.shape[1]+1, dtype=np.float32)
    # Fixed spectral upper bound on the logistic-loss curvature.
    cov = (x * w[:, None]).T @ x
    # Frobenius norm bounds the largest eigenvalue of the joint design Hessian.
    curvature_bound = np.sqrt(np.square(cov).sum() + 2*np.square(w @ x).sum() + 1)
    step = 1/(.25*curvature_bound+.01)
    momentum = parameters.copy()
    for k in range(160):
        error = w * (sigmoid(x @ momentum[:-1] + momentum[-1]) - labels)
        updated = momentum - step*np.r_[x.T @ error + .01*momentum[:-1], error.sum()]
        momentum = updated + (k/(k+3))*(updated-parameters)
        parameters = updated
    return dict(mean=mean, scale=sd, parameters=parameters)


def predict(params, x):
    return sigmoid(np.clip((x-params["mean"])/params["scale"], -6, 6) @
                   params["parameters"][:-1] + params["parameters"][-1])


def select(entry, score):
    """First DS alternative with observed support at baseline's risky steps."""
    baseline = int(entry["baseline"])
    observed = entry["view"] >= 0
    evidence = observed[baseline] & (score[baseline] >= HIGH)
    if not evidence.any():
        return baseline
    for j in entry["order"]:
        if j == baseline or observed[j].mean() < NEEDED:
            continue
        if (observed[j][evidence].all() and (score[j][observed[j]] <= LOW).all()):
            return int(j)
    return baseline


def precision_summary(truth, score):
    """Average precision over score tie groups, including all equal scores."""
    positives = int(truth.sum())
    if not positives:
        return dict(ap=None, positives=0, points=len(truth))
    order = np.argsort(-score, kind="stable")
    cut = np.r_[np.flatnonzero(np.diff(score[order])), len(order)-1]
    cumulative = np.cumsum(truth[order])[cut]
    ap = (np.diff(np.r_[0, cumulative]) * cumulative/(cut+1)).sum()/positives
    return dict(ap=float(ap), positives=positives, points=len(truth))


def study(data, out, manifest):
    index = io.read(out/"features.json")
    identity = io.read(out/"feature_identity.json")
    rows = manifest["cohort"]
    if identity["manifest"] != io.sha(data/"manifest.json") or identity["code"] != io.sha(__file__):
        raise ValueError("Feature export identity mismatch")
    files = {f"features/{r['token']}.npz" for r in rows}
    if set(index) != files:
        raise ValueError("Finish the full frozen feature export before fitting")
    io.files_checked(out, {"files": index})
    entries = {r["token"]: read_arrays(data, manifest, f"inputs/{r['token']}.npz") for r in rows}
    sessions = {r["token"]: r["session_id"] for r in rows}
    decisions = {}
    methods = (*METHODS, "teacher")
    for fold, held in enumerate(manifest["groups"]):
        train = [t for t in sessions if sessions[t] not in held]
        test = [t for t in sessions if sessions[t] in held]
        training_labels = {t: read_arrays(data, manifest, f"labels/{t}.npz")["outside"][entries[t]["view"] >= 0]
                           for t in train}
        samples = {t: training_rows(training_labels[t], np.random.default_rng(model.seeded(t, 20261118))) for t in train}
        seen = [t for t in train if len(samples[t])]
        if not seen:
            raise ValueError(f"No projected training points in fold {fold}")
        counts = {sid: sum(sessions[t] == sid for t in seen) for sid in {sessions[t] for t in seen}}
        sample_weight = np.concatenate([np.repeat(1/(counts[sessions[t]]*len(samples[t])), len(samples[t]))
                                        for t in seen]).astype(np.float32)
        target = np.concatenate([training_labels[t][samples[t]] for t in seen])
        report = out/f"predictions/fold{fold}.npz"
        marker = report.with_suffix(".json")
        if marker.exists():
            if io.read(marker) != dict(sha256=io.sha(report), feature_index=io.sha(out/"features.json")):
                raise ValueError("Cached fold predictions changed")
            scores = io.load_npz(report)
            if scores.pop("tokens").tolist() != test:
                raise ValueError("Cached fold does not match the frozen test sessions")
        else:
            scores = {}
            print(f"FOLD {fold} train={len(train)} test={len(test)} sampled_outside={target.sum()}", flush=True)
            for method in METHODS:
                x = np.concatenate([features(entries[t], io.load_npz(out/f"features/{t}.npz"), t, method)[samples[t]]
                                    for t in seen])
                params = fit(x, target, sample_weight)
                io.save(out/f"models/fold{fold}_{method}.npz", **params)
                del x
                scores[method] = np.stack([dense_score(entries[t], predict(params, features(
                    entries[t], io.load_npz(out/f"features/{t}.npz"), t, method))) for t in test])
            scores["teacher"] = np.stack([teacher_score(entries[t]) for t in test])
            io.save(report, tokens=np.asarray(test), **scores)
            io.write(marker, dict(sha256=io.sha(report), feature_index=io.sha(out/"features.json")))
        for j, token in enumerate(test):
            decisions[token] = {method: (scores[method][j], select(entries[token], scores[method][j])) for method in methods}
    # Evaluation only begins after decisions have been frozen for all scenes.
    destination = out/"decision_commit.json"
    io.freeze(destination, dict(folds=manifest["groups"], feature_index=io.sha(out/"features.json"),
                               rows={t: {m: int(v[1]) for m, v in d.items()} for t, d in decisions.items()}))
    return evaluate(data, out, manifest, entries, sessions, decisions)


def dense_score(entry, values):
    score = np.full(entry["view"].shape, np.nan, dtype=np.float32)
    score[entry["view"] >= 0] = values
    return score


def teacher_score(entry):
    return dense_score(entry, np.broadcast_to(sigmoid(-entry["teacher_dac"][:, None, None]),
                                             entry["view"].shape)[entry["view"] >= 0])


def evaluate(data, out, manifest, entries, sessions, decisions):
    methods = (*METHODS, "teacher")
    rows = []
    points = {method: [] for method in methods}
    visible_candidates = {method: [] for method in methods}
    for t, choices in decisions.items():
        entry = entries[t]
        base = int(entry["baseline"])
        labels = read_arrays(data, manifest, f"labels/{t}.npz")
        q = labels["raw"]
        mask = entry["view"] >= 0
        known = mask.any(axis=(1, 2))
        near = labels["near_edge"] & mask
        safe = (q[:, 1:][:, list(model.SAFETY)] >= q[base, 1:][list(model.SAFETY)]-1e-8).all(axis=1)
        repair = (q[base, DAC] < 1) & (q[:, DAC] == 1)
        repair_available = bool((repair & safe & (q[:, 0] > q[base, 0]+1e-8)).any())
        for method, (score, p) in choices.items():
            points[method].append((labels["outside"][mask], score[mask]))
            visible_candidates[method].append(((labels["outside"] & mask).any(axis=(1, 2))[known],
                                               np.where(mask, score, -np.inf).max(axis=(1, 2))[known]))
            delta = q[p]-q[base]
            row = dict(token=t, session_id=sessions[t], method=method, replaced=p != base,
                       baseline_flagged=bool(np.any(score[base] >= HIGH)),
                       baseline_false_flag=bool(q[base, DAC] == 1 and np.any(score[base] >= HIGH)),
                       baseline_DAC_failure=bool(q[base, DAC] < 1),
                       DAC_repair=bool(repair[p]),
                       missed_safe_PDM_repair=bool(repair_available and not (repair[p] and safe[p] and delta[0] > 1e-8)),
                       replacement_of_DAC_clear_baseline=bool(p != base and q[base, DAC] == 1),
                       baseline_projected_fraction=float(mask[base].mean()),
                       chosen_projected_fraction=float(mask[p].mean()),
                       edge_negative_points=int(near.sum()),
                       edge_false_flags=int((near & (score >= HIGH)).sum()),
                       baseline_rank=int(np.where(entry["order"] == base)[0][0])+1,
                       chosen_rank=int(np.where(entry["order"] == p)[0][0])+1,
                       candidate_id=int(entry["candidate_ids"][p]), gain_pp=float(100*delta[0]),
                       harm=bool(np.any(delta[1:][list(model.SAFETY)] < -1e-8)))
            row.update({k+"_delta": float(delta[j+1]) for j, k in enumerate(model.METRICS)})
            rows.append(row)
    with (out/"per_scene.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    results, vectors, session_rows = {}, {}, []
    count = len(set(sessions.values()))
    draw = np.random.default_rng(20261118).integers(0, count, (10000, count))
    for method, point in points.items():
        method_rows = [r for r in rows if r["method"] == method]
        by_session = {sid: [] for sid in set(sessions.values())}
        for row in method_rows:
            by_session[row["session_id"]].append(row)
        truth, score = map(np.concatenate, zip(*point))
        for sid, subset in sorted(by_session.items()):
            session_rows.append(dict(method=method, session_id=sid, scenes=len(subset),
                gain_mean_pp=float(np.mean([r["gain_pp"] for r in subset])),
                repairs=sum(r["DAC_repair"] for r in subset), harms=sum(r["harm"] for r in subset),
                replacements=sum(r["replaced"] for r in subset)))
        v = np.array([r["gain_mean_pp"] for r in session_rows if r["method"] == method])
        vectors[method] = v
        edge = sum(r["edge_negative_points"] for r in method_rows)
        edge_errors = sum(r["edge_false_flags"] for r in method_rows)
        results[method] = dict(point_detection=precision_summary(truth, score),
                               candidate_detection=precision_summary(*map(np.concatenate, zip(*visible_candidates[method]))),
                               edge_false_flag_rate=edge_errors/edge if edge else None,
                               edge_negative_points=edge,
                               gain_macro_pp=float(v.mean()),
                               gain_ci95_sessions=np.quantile(v[draw].mean(axis=1), [.025, .975]).tolist(),
                               gain_without_best_session_pp=float(np.sort(v)[:-1].mean()),
                               repairs=sum(r["DAC_repair"] for r in method_rows),
                               repair_sessions=len({r["session_id"] for r in method_rows if r["DAC_repair"]}),
                               missed_DAC_failures=sum(r["baseline_DAC_failure"] and not r["DAC_repair"] for r in method_rows),
                               missed_safe_PDM_repairs=sum(r["missed_safe_PDM_repair"] for r in method_rows),
                               interventions=sum(r["replaced"] for r in method_rows),
                               false_flags=sum(r["baseline_false_flag"] for r in method_rows),
                               replacements_of_DAC_clear_baseline=sum(r["replacement_of_DAC_clear_baseline"] for r in method_rows),
                               new_safety_harm=sum(r["harm"] for r in method_rows),
                               PDM_losses=sum(r["gain_pp"] < -1e-6 for r in method_rows))
    with (out/"per_session.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(session_rows[0]))
        writer.writeheader()
        writer.writerows(session_rows)
    paired = {}
    for control in ("global", "shuffled", "teacher"):
        delta = vectors["aligned"]-vectors[control]
        paired["aligned_minus_"+control] = dict(gain_macro_pp=float(delta.mean()),
            ci95_sessions=np.quantile(delta[draw].mean(axis=1), [.025, .975]).tolist(),
            sessions_improved=int((delta > 1e-6).sum()))
    def observed_candidate_ap(method):
        value = results[method]["candidate_detection"]["ap"]
        return value if value is not None else -np.inf

    aligned = results["aligned"]
    checks = dict(candidate_ap_above_both_spatial_controls=all(
                      observed_candidate_ap("aligned") > observed_candidate_ap(control)
                      for control in ("global", "shuffled")),
                  PDM_above_DS_and_both_spatial_controls=aligned["gain_macro_pp"] > 0 and all(
                      paired["aligned_minus_"+control]["gain_macro_pp"] > 0
                      for control in ("global", "shuffled")),
                  repairs_in_two_or_more_sessions=aligned["repair_sessions"] >= 2,
                  positive_without_best_session=aligned["gain_without_best_session_pp"] > 0,
                  no_new_safety_harm=aligned["new_safety_harm"] == 0)
    result = dict(status="complete", scenes=len(sessions), sessions=count,
                  methods=results, paired=paired, exploratory_checks=checks,
                  continue_spatial_pilot=all(checks.values()), note=NOTE)
    io.write(out/"report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("smoke", "export", "fit", "report"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--images", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    a = p.parse_args()
    data, out = a.data.resolve(), a.out.resolve()
    if data.is_relative_to(out) or out.is_relative_to(data):
        p.error("--out must be separate from --data")
    manifest = read_manifest(data)
    if a.stage in ("smoke", "export"):
        if a.images is None:
            p.error("--images is required for feature export")
        indices = range(min(4, len(manifest["cohort"]))) if a.stage == "smoke" else range(len(manifest["cohort"]))
        result = export(data, a.images.resolve(), out, manifest, indices, a.device)
    elif a.stage == "fit":
        with io.locked(out):
            result = study(data, out, manifest)
    else:
        result = io.read(out/"report.json")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
