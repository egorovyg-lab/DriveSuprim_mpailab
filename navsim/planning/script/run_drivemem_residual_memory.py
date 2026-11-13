"""Astra CPU top64 residual-memory study; no downloads, GPU or scorer calls."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from navsim.agents import drivemem_residual_memory as model

PROTOCOL = "drivemem_learned_residual_top64_v1"
METHODS = ("C0_DS", "C1_selector", "C2_memory", "C3_corrector") + tuple(
    f"C{c}_{seed}" for c in (4, 5) for seed in model.CONTROL_SEEDS)
SOURCES = ("navsim/agents/drivemem_residual_memory.py", "navsim/planning/script/run_drivemem_residual_memory.py",
           "navsim/agents/trajectory_selector.py", "navsim/agents/drivemem.py",
           "navsim/agents/drivemem_action_transfer.py",
           "scripts/drivesuprim/evaluation/run_drivemem_residual_memory.sh")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def freeze(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError("Frozen artifact differs: " + str(path))
    else:
        write(path, value)


def safe_child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes artifact root")
    return path


def files_checked(root, commit):
    for name, expected in commit["files"].items():
        if sha(safe_child(root, name)) != expected:
            raise RuntimeError("Committed artifact changed: " + str(root / name))


@contextmanager
def locked(root):
    root.mkdir(parents=True, exist_ok=True)
    f = (root / ".run.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0, 2)
            if not f.tell():
                f.write(b"0"); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        f.close()
        raise RuntimeError("Another DriveMem process holds " + str(root)) from exc
    try:
        yield
    finally:
        f.close()  # OS releases the lock; never delete a lock after a crash.


def runtime(threads):
    import torch
    torch.set_num_threads(threads)
    return dict(device="cpu", threads=torch.get_num_threads(), python=platform.python_version(),
                numpy=np.__version__, torch=torch.__version__, platform=platform.system(),
                blas_environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})


def peak_rss_mib():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(value / (1024 ** 2 if sys.platform == "darwin" else 1024))
    except ImportError:
        return None


def source_snapshot():
    return {name: sha(REPO / name) for name in SOURCES}


def bundle_metadata(root):
    manifest, identity, index = [read(root / n) for n in ("bundle_manifest.json", "identity.json", "examples_index.json")]
    for name, key in (("identity.json", "identity_sha256"), ("examples_index.json", "index_sha256"), ("vocab.npz", "vocab_sha256")):
        if sha(root / name) != manifest[key]:
            raise RuntimeError("Bundle manifest mismatch: " + name)
    if (manifest["counts"] != identity["counts"] or len(index) != identity["counts"]["scenes"]
            or len({e["metadata"]["session_id"] for e in index.values()}) != identity["counts"]["sessions"]
            or manifest["input_keys"] != list(model.INPUT_KEYS) or manifest["label_keys"] != list(model.LABEL_KEYS)):
        raise ValueError("Bundle counts/input/label schema mismatch")
    for token, e in index.items():
        m = e["metadata"]
        if (not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", token) or m["token"] != token
                or m["role"] != "train" or m["session_id"] in identity["protected_sessions"]
                or m.get("source_role") in ("development", "calibration", "test")):
            raise ValueError("Unprotected training-only cohort required")
    bindings = {str(root / n): sha(root / n) for n in ("bundle_manifest.json", "identity.json", "examples_index.json", "vocab.npz")}
    for name, expected in identity.get("sources", {}).items():
        if sha(safe_child(REPO, name)) != expected:
            raise RuntimeError("Frozen bundle source changed: " + name)
    for name, expected in identity.get("inputs", {}).items():
        if sha(Path(name)) != expected:
            raise RuntimeError("Frozen upstream input changed: " + name)
        bindings[name] = expected
    return identity, index, bindings


def label_paths(args):
    base = args.bundle.parent.parent
    # Deliberately bounded to the previous training-only table audit outputs.
    paths = [args.labels] if args.labels else sorted((base / "drex_rollout_audit").glob(
        "rollout_audit_table_*/artifacts/labels.json"))
    packages = [args.package_manifest] if args.package_manifest else [
        base / "drex_transfer/navsim_train_v1/package_manifest.json",
        base / "drex_transfer/navsim_train_v1/package/package_manifest.json"]
    return paths, packages


def match_labels(path, packages, index, bundle, explicit_request=None):
    request_path = explicit_request or path.parent.parent / "audit_request.json"
    request, report, identity = [read(p) for p in (request_path, path.parent / "report.json", path.parent / "identity.json")]
    labels = read(path)
    if (request["protocol"] != "table_top64_labels_v1" or request["metrics"] != list(model.LABEL_KEYS)
            or report["status"] != "complete" or report["scenes"] != len(index)
            or report["candidates"] != len(index) * 64 or report["labels_sha256"] != sha(path)
            or identity["request_sha256"] != sha(request_path)
            or identity["score_protocol"] != "single_candidate_raw_v1" or identity["human_penalty_filter"] is not False
            or labels["status"] != "complete" or labels["metrics"] != list(model.LABEL_KEYS)
            or labels["request_digest"] != model.digest(request)):
        raise ValueError("Incomplete/incompatible top64 label provenance")
    package_path = next((p for p in packages if p.is_file() and sha(p) == request["package_manifest_sha256"]), None)
    if package_path is None:
        raise FileNotFoundError("Matching package_manifest.json not found; pass --package-manifest (numeric metadata only)")
    package = read(package_path)
    if package["protocol"] != "portable_train_v1" or package["holdout_scenes"] != 0:
        raise ValueError("Training-only portable package required")
    for name in ("bundle_manifest.json", "identity.json", "examples_index.json", "vocab.npz"):
        if package["files"]["bundle/" + name]["sha256"] != sha(bundle / name):
            raise ValueError("Drex labels belong to a different feature bundle")
    for token, e in index.items():
        if package["files"]["bundle/examples/" + token + ".npz"]["sha256"] != e["sha256"]:
            raise ValueError("Portable candidate inputs changed")
    requests = {r["token"]: r["candidates"] for r in request["scenes"]}
    if set(requests) != set(index) or len(requests) != len(request["scenes"]):
        raise ValueError("Label request cohort mismatch")
    lookup = {}
    for row in labels["rows"]:
        key = (row["token"], int(row["candidate_local"]))
        if key in lookup or key[0] not in index:
            raise ValueError("Duplicate/unexpected label")
        lookup[key] = row
    if len(lookup) != len(index) * 64:
        raise ValueError("Complete top64 labels required")
    bindings = {str(p): sha(p) for p in (path, request_path, path.parent / "report.json", path.parent / "identity.json", package_path)}
    return lookup, requests, bindings


def collect(args, expected=(651, 32)):
    identity, index, bindings = bundle_metadata(args.bundle)
    if (len(index), identity["counts"]["sessions"]) != expected:
        raise ValueError(f"Expected {expected[0]} scenes/{expected[1]} sessions; no silent cohort reduction")
    paths, packages = label_paths(args)
    matches, errors = [], []
    for path in paths:
        try:
            matches.append((path, match_labels(path, packages, index, args.bundle, args.label_request)))
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            errors.append(dict(path=str(path), error=str(exc)))
    if not matches:
        raise FileNotFoundError("No matching complete Astra top64 labels. Set DRIVEMEM_RESIDUAL_LABELS / "
                                "--labels and, if needed, --package-manifest. Searched: " + str(paths) + "; errors=" + str(errors))
    if len({sha(p) for p, _ in matches}) != 1:
        raise ValueError("Multiple distinct matching label runs; specify --labels explicitly")
    path, (lookup, requests, extra) = matches[0]
    bindings.update(extra)
    with np.load(args.bundle / "vocab.npz", allow_pickle=False) as f:
        vocab = f["vocab"].copy()
    if vocab.ndim != 3 or vocab.shape[1:] != (40, 3) or not np.isfinite(vocab).all():
        raise ValueError("Expected finite four-second vocabulary")
    # Challenge tags are provenance only, never feature/selection conditions.
    challenge = set()
    for name in bindings:
        if Path(name).name == "label_selection.json":
            challenge.update(r["token"] for r in read(Path(name)).get("records", [])
                             if r.get("sampling", "").startswith("challenge"))
    entries, raws = [], {}
    for token in sorted(index):
        spec = index[token]
        example = args.bundle / "examples" / (token + ".npz")
        if sha(example) != spec["sha256"]:
            raise RuntimeError("Frozen example changed: " + token)
        arrays = load_npz(example)
        meta = dict(spec["metadata"], sampling_slice="challenge" if token in challenge else
                    ("uniform" if spec["metadata"].get("origin") in ("new", "expansion30") else "unknown"))
        entry = model.input_entry(arrays, vocab, meta)
        desired = [[int(j), int(v)] for j, v in zip(entry["indices"], entry["ids"])]
        if requests[token] != desired:
            raise ValueError("Frozen top64 request differs from current bundle: " + token)
        raw = []
        for j, v in desired:
            row = lookup[(token, j)]
            if int(row["candidate_vocab"]) != v:
                raise ValueError("Vocabulary/local ID label mismatch")
            raw.append(row["raw"])
        raw = np.asarray(raw, dtype=float)
        model.supervised(entry, raw)  # Validate the full nine-field raw schema.
        if (not np.array_equal(arrays["label_indices"], entry["indices"][:16])
                or not np.allclose(arrays["label_raw"], raw[:16], atol=1e-8, rtol=1e-7)):
            raise ValueError("Original top16 raw label parity failed: " + token)
        entries.append(entry); raws[token] = raw
    return entries, raws, dict(bindings=bindings, labels=str(path), original_index_sha256=sha(args.bundle / "examples_index.json"),
                               protected_sessions=identity["protected_sessions"])


def initialize(out, entries, provenance, config, threads, synthetic=False):
    groups = model.partition(entries, config.seed)
    identity = dict(protocol=PROTOCOL, config=model.config_dict(config), runtime=runtime(threads),
                    sources=source_snapshot(), provenance=provenance, groups=groups, methods=list(METHODS),
                    synthetic=synthetic, counts=dict(scenes=len(entries), sessions=sum(map(len, groups))), note=model.NOTE)
    if not (out / "identity.json").exists() and any(p.name not in (".run.lock", "failures", "failure.json") for p in out.iterdir()):
        raise ValueError("Nonempty output without identity; use a new root")
    freeze(out / "identity.json", identity)
    freeze(out / "split.json", dict(groups=groups, seed=config.seed, outcome_blind=True))
    return identity


def prepare(out, entries, raws):
    target = out / "data"
    if (target / "manifest.json").exists():
        commit = read(target / "manifest.json")
        files_checked(target, commit)
        return
    files, metadata = {}, []
    for e in entries:
        token = e["token"]
        inp, lab = target / "inputs" / (token + ".npz"), target / "labels" / (token + ".npz")
        save(inp, x=e["x"], indices=e["indices"], ids=e["ids"])
        save(lab, raw=raws[token])
        files[str(inp.relative_to(target)).replace("\\", "/")] = sha(inp)
        files[str(lab.relative_to(target)).replace("\\", "/")] = sha(lab)
        metadata.append({k: e[k] for k in ("token", "session_id", "baseline", "blocks", "sampling_slice")})
    write(target / "metadata.json", metadata)
    files["metadata.json"] = sha(target / "metadata.json")
    write(target / "manifest.json", dict(files=files, identity_sha256=sha(out / "identity.json")))


def prepared(out):
    commit = read(out / "data/manifest.json")
    if commit["identity_sha256"] != sha(out / "identity.json"):
        raise RuntimeError("Prepared data identity mismatch")
    files_checked(out / "data", commit)
    return [dict(m, **load_npz(out / "data/inputs" / (m["token"] + ".npz"))) for m in read(out / "data/metadata.json")]


def labels_for(out, entries):
    return {e["token"]: load_npz(out / "data/labels" / (e["token"] + ".npz"))["raw"] for e in entries}


def verify_root(out, config, threads):
    identity = read(out / "identity.json")
    if (identity["protocol"] != PROTOCOL or identity["config"] != model.config_dict(config)
            or identity["runtime"] != runtime(threads) or identity["sources"] != source_snapshot()):
        raise RuntimeError("Protocol/code/runtime changed; use a new output root")
    for name, expected in identity["provenance"].get("bindings", {}).items():
        if sha(Path(name)) != expected:
            raise RuntimeError("Bound source changed: " + name)
    if read(out / "split.json")["groups"] != identity["groups"]:
        raise RuntimeError("Session split changed")
    return identity


def fit_name(kind, groups):
    return kind + "_" + "".join(map(str, groups))


def fit_model(out, all_entries, groups, which, config, kind="base", residuals=None):
    sessions = {s for i in which for s in groups[i]}
    training = [e for e in all_entries if e["session_id"] in sessions]
    name = fit_name(kind, which)
    root = out / "fits" / name
    target_hash = None if residuals is None else hashlib.sha256(
        np.stack([residuals[e["token"]] for e in training]).astype("<f8").tobytes()).hexdigest()
    spec = dict(identity_sha256=sha(out / "identity.json"), data_sha256=sha(out / "data/manifest.json"),
                kind=kind, groups=list(which), sessions=sorted(sessions), tokens=[e["token"] for e in training],
                residual_target_sha256=target_hash)
    if (root / "commit.json").exists():
        commit = read(root / "commit.json")
        if commit["spec"] != spec:
            raise RuntimeError("Cached fit partition/targets changed")
        files_checked(root, commit)
        checkpoint = load_npz(root / "checkpoint.npz")
        replay = model.predict(checkpoint, training, config)
        if hashlib.sha256(replay.astype("<f8").tobytes()).hexdigest() != commit["replay_sha256"]:
            raise RuntimeError("Cached checkpoint prediction replay failed")
        return checkpoint
    raws = labels_for(out, training) if kind == "base" else None
    supervised = [model.supervised(e, raws[e["token"]]) for e in training] if raws is not None else training
    print(f"FIT {name} sessions={len(sessions)} scenes={len(training)} epochs={config.epochs}", flush=True)
    start = time.monotonic()
    def progress(item):
        print(f"  {name} epoch={item['epoch']} loss={item['loss']:.6f}", flush=True)
        write(out / "progress.json", dict(status="fitting", fit=name, **item))
    checkpoint, history = model.train(supervised, config, kind, residuals, progress)
    save(root / "checkpoint.npz", **checkpoint)
    before = model.predict(checkpoint, training, config)
    after = model.predict(load_npz(root / "checkpoint.npz"), training, config)
    if not np.array_equal(before, after):
        raise RuntimeError("Checkpoint replay differs")
    seconds = time.monotonic() - start
    write(root / "history.json", dict(epochs=history, seconds=seconds, process_peak_rss_mib=peak_rss_mib()))
    write(root / "commit.json", dict(spec=spec, files={n: sha(root / n) for n in ("checkpoint.npz", "history.json")},
             replay_sha256=hashlib.sha256(after.astype("<f8").tobytes()).hexdigest(), seconds=seconds))
    return checkpoint


def fold_commit(out, outer, entries, groups):
    path = out / "folds" / str(outer)
    if not (path / "commit.json").exists():
        return None
    c = read(path / "commit.json")
    tokens = [e["token"] for e in entries if e["session_id"] in groups[outer]]
    if (c["identity_sha256"] != sha(out / "identity.json") or c["tokens"] != tokens
            or c["outer"] != outer or c["methods"] != list(METHODS)
            or c["training_sessions"] != sorted(set(sum(groups, [])) - set(groups[outer]))):
        raise RuntimeError("Cached fold session/decision commitment mismatch")
    files_checked(path, c)
    which = tuple(i for i in range(4) if i != outer)
    expected_fits = {fit_name("base", pair) for pair in itertools.combinations(which, 2)} | {
        fit_name("base", which), fit_name("corrector", which)}
    if set(c["fit_commits"]) != expected_fits:
        raise RuntimeError("Incomplete fold fit provenance")
    for name, expected in c["fit_commits"].items():
        fit_root = out / "fits" / name
        if sha(fit_root / "commit.json") != expected:
            raise RuntimeError("Committed fit provenance changed")
        files_checked(fit_root, read(fit_root / "commit.json"))
    return c


def run_fold(out, entries, groups, outer, config):
    if fold_commit(out, outer, entries, groups):
        print(f"FOLD {outer} already committed", flush=True)
        return
    which = tuple(i for i in range(4) if i != outer)
    train_sessions = {s for i in which for s in groups[i]}
    training = [e for e in entries if e["session_id"] in train_sessions]
    query = [e for e in entries if e["session_id"] in groups[outer]]
    if {e["session_id"] for e in training} & {e["session_id"] for e in query}:
        raise RuntimeError("Whole query session leaked into training")
    residuals, sources = {}, {}
    raws = labels_for(out, training)  # Only the permitted outer-training sessions.
    for held in which:
        pair = tuple(i for i in which if i != held)
        checkpoint = fit_model(out, entries, groups, pair, config)
        donors = [e for e in training if e["session_id"] in groups[held]]
        pred = model.predict(checkpoint, donors, config)
        for e, p in zip(donors, pred):
            residuals[e["token"]] = model.supervised(e, raws[e["token"]])["targets"]["gain"].astype(float) - p
            sources[e["token"]] = dict(fit=fit_name("base", pair), training_sessions=sorted(s for i in pair for s in groups[i]))
            if e["session_id"] in sources[e["token"]]["training_sessions"]:
                raise RuntimeError("Own donor session used for residual prediction")
    base = fit_model(out, entries, groups, which, config)
    corrector = fit_model(out, entries, groups, which, config, "corrector", residuals)
    f, g = model.predict(base, query, config), model.predict(corrector, query, config)
    mean, scale = model.scaler(training)
    bank = model.Memory(training, residuals, mean, scale)
    h = bank.bandwidth(training, config)
    shuffled = {seed: bank.shuffled(seed) for seed in model.CONTROL_SEEDS}
    root = out / "folds" / str(outer)
    write(root / "donors.json", dict(records=bank.records, residual_sources=sources,
          training_tokens=[e["token"] for e in training], shuffle_sources={str(s): v[1] for s, v in shuffled.items()}, h=h))
    save(root / "residuals.npz", values=bank.residuals, mean=mean, scale=scale)
    predictions, positions, traces = [], [], {}
    for i, e in enumerate(query):
        ds = np.zeros(64)
        d = bank.distances(e)
        m, trace = bank.recall(e, h, config, distances=d)
        gains = [ds, f[i], f[i] + m, f[i] + g[i]]
        for name, value in trace.items():
            traces.setdefault("nearest_" + name, []).append(value)
        for seed in model.CONTROL_SEEDS:
            r, trace = bank.recall(e, h, config, "random", seed, distances=d)
            gains.append(f[i] + r)
            for name, value in trace.items():
                traces.setdefault(f"random_{seed}_" + name, []).append(value)
        for seed in model.CONTROL_SEEDS:
            m, _ = bank.recall(e, h, config, values=shuffled[seed][0], distances=d)
            gains.append(f[i] + m)
        gains = np.asarray(gains)
        if gains.shape != (len(METHODS), 64):
            raise RuntimeError("Policy coverage mismatch")
        predictions.append(gains)
        positions.append([model.choose(e, a, config.margin) for a in gains])
        if (i + 1) % 10 == 0 or i + 1 == len(query):
            print(f"FOLD {outer} decisions={i+1}/{len(query)}", flush=True)
            write(out / "progress.json", dict(status="deciding", outer=outer, scenes=i+1, total=len(query)))
    save(root / "donor_traces.npz", **{k: np.asarray(v) for k, v in traces.items()})
    save(root / "decisions.npz", positions=np.asarray(positions), gains=np.asarray(predictions))
    # All policies in this fold are committed before evaluator can read its labels.
    write(root / "commit.json", dict(identity_sha256=sha(out / "identity.json"), outer=outer,
          tokens=[e["token"] for e in query], methods=list(METHODS), training_sessions=sorted(train_sessions),
          fit_commits={name: sha(out / "fits" / name / "commit.json") for name in
                       [*(fit_name("base", pair) for pair in itertools.combinations(which, 2)),
                        fit_name("base", which), fit_name("corrector", which)]},
          files={n: sha(root / n) for n in ("donors.json", "residuals.npz", "donor_traces.npz", "decisions.npz")}))


def decision_rows(out, entries, groups, config):
    chosen = {}
    for outer in range(4):
        c = fold_commit(out, outer, entries, groups)
        if c is None:
            raise RuntimeError("All four OOF folds must commit before evaluation; labels remain unread")
        data = load_npz(out / "folds" / str(outer) / "decisions.npz")
        if data["positions"].shape != (len(c["tokens"]), len(METHODS)) or data["gains"].shape != (len(c["tokens"]), len(METHODS), 64):
            raise ValueError("Incomplete decision arrays")
        lookup = {e["token"]: e for e in entries}
        for token, pos, gains in zip(c["tokens"], data["positions"], data["gains"]):
            if token in chosen or not np.array_equal(pos, [model.choose(lookup[token], g, config.margin) for g in gains]):
                raise ValueError("Duplicate or non-replayable decisions")
            chosen[token] = (outer, pos, gains)
    if set(chosen) != {e["token"] for e in entries}:
        raise ValueError("OOF does not cover cohort exactly")
    return chosen


def aggregate(values, sids, sessions):
    return np.array([np.mean(values[np.asarray(sids) == s]) for s in sessions])


def effect_stats(vector, draws):
    vector = np.asarray(vector, dtype=float)
    interval = np.quantile(vector[draws].mean(1), [.025, .975])
    ordered = np.argsort(-vector, kind="stable")
    return dict(mean_pp=float(vector.mean()), ci95_sessions=interval.tolist(),
                leave_best_one_pp=float(np.delete(vector, ordered[:1]).mean()),
                leave_best_two_pp=float(np.delete(vector, ordered[:2]).mean()),
                sessions_positive=int((vector > 1e-6).sum()), sessions_negative=int((vector < -1e-6).sum()),
                sessions_improved_at_least_half_pp=int((vector >= .5).sum()))


def evaluate(out, entries, groups, config):
    chosen = decision_rows(out, entries, groups, config)  # No query labels before full commitment.
    commits = {str(i): sha(out / "folds" / str(i) / "commit.json") for i in range(4)}
    report_spec = dict(identity_sha256=sha(out / "identity.json"), data_sha256=sha(out / "data/manifest.json"), folds=commits)
    if (out / "report_commit.json").exists():
        commit = read(out / "report_commit.json")
        if commit["spec"] != report_spec:
            raise RuntimeError("Report inputs changed")
        files_checked(out, commit)
        return read(out / "summary.json")
    raws = labels_for(out, entries)
    rows = []
    for e in entries:
        raw = raws[e["token"]]
        b = e["baseline"]
        safe = np.all(raw[:, 1:][:, model.SAFETY] + 1e-8 >= raw[b, 1:][list(model.SAFETY)], axis=1)
        delta = raw[:, 0] - raw[b, 0]
        safe_best = float(delta[safe].max())
        outer, positions, gains = chosen[e["token"]]
        for j, method in enumerate(METHODS):
            p = int(positions[j])
            changes = raw[p, 1:] - raw[b, 1:]
            row = dict(token=e["token"], session_id=e["session_id"], outer=outer, method=method,
                       sampling_slice=e["sampling_slice"], candidate_local=int(e["indices"][p]), candidate_id=int(e["ids"][p]),
                       ds_rank=p+1, intervention=p != b, raw_score=float(raw[p, 0]), baseline_score=float(raw[b, 0]),
                       raw_gain_pp=float(100 * delta[p]), predicted_gain_pp=float(100 * gains[j, p]),
                       raw_harm=not bool(safe[p]), raw_quality_degraded=bool(delta[p] < -1e-8),
                       improved=bool(delta[p] > 1e-8), useful_repair=bool(delta[p] >= .005 and safe[p]),
                       missed_repair=bool(safe_best >= .005 and not (delta[p] >= .005 and safe[p])),
                       oracle_regret_pp=float(100 * (raw[:, 0].max() - raw[p, 0])),
                       safe_oracle_regret_pp=float(100 * max(0, safe_best - delta[p])),
                       oracle_gain_pp=float(100 * delta.max()), safe_oracle_gain_pp=100 * safe_best,
                       extra_top64_over_top16_pp=float(100 * (raw[:, 0].max() - raw[:16, 0].max())),
                       unique_tail_oracle=bool(raw[:, 0].max() > raw[:16, 0].max() + 1e-8))
            for k, m in enumerate(model.METRICS):
                row["raw_delta_" + m] = float(changes[k])
                row["raw_" + m] = float(raw[p, k+1])
            rows.append(row)
    import pandas as pd
    frame = pd.DataFrame(rows)
    # Compare actual selected actions, not only whether both differ from DS.
    control_rows = frame[frame.method == "C1_selector"].set_index("token")
    frame["gain_vs_C1_pp"] = frame.raw_gain_pp.to_numpy() - frame.token.map(control_rows.raw_gain_pp).to_numpy()
    frame["safety_worse_than_C1_choice"] = False
    for k, m in enumerate(model.METRICS):
        frame["delta_vs_C1_" + m] = frame["raw_" + m].to_numpy() - frame.token.map(control_rows["raw_" + m]).to_numpy()
        if k in model.SAFETY:
            frame["safety_worse_than_C1_choice"] |= frame["delta_vs_C1_" + m] < -1e-8
    sessions = sorted({e["session_id"] for e in entries})
    draws = np.random.default_rng(config.seed).integers(0, len(sessions), (config.bootstrap_reps, len(sessions)))
    save(out / "bootstrap_draws.npz", indices=draws)
    vectors, scene_vectors, methods, per_session = {}, {}, {}, []
    baseline_frame = frame[frame.method == "C1_selector"].set_index("token")
    tokens = [e["token"] for e in entries]
    for method in METHODS:
        q = frame[frame.method == method].set_index("token").loc[tokens]
        v = q.raw_gain_pp.to_numpy()
        sids = q.session_id.to_numpy()
        scene_vectors[method] = v
        vectors[method] = aggregate(v, sids, sessions)
        changed_vs_control = v - baseline_frame.loc[tokens].raw_gain_pp.to_numpy()
        new_harm = q.raw_harm.to_numpy() & ~baseline_frame.loc[tokens].raw_harm.to_numpy()
        info = effect_stats(vectors[method], draws)
        info.update(gain_micro_pp=float(v.mean()),
                    score_micro=float(q.raw_score.mean()), baseline_micro=float(q.baseline_score.mean()),
                    score_macro=float(aggregate(q.raw_score.to_numpy(), sids, sessions).mean()),
                    baseline_macro=float(aggregate(q.baseline_score.to_numpy(), sids, sessions).mean()),
                    interventions=int(q.intervention.sum()), improved_scenes=int(q.improved.sum()),
                    pdm_losses=int(q.raw_quality_degraded.sum()), safety_harm_scenes=int(q.raw_harm.sum()),
                    worst_scene_gain_pp=float(min(0, v.min())), gross_gain_sum_pp=float(v[v > 0].sum()),
                    gross_loss_sum_pp=float(v[v < 0].sum()), missed_repairs=int(q.missed_repair.sum()),
                    new_safety_harm_vs_C1=int(new_harm.sum()),
                    safety_worse_than_C1_choice=int(q.safety_worse_than_C1_choice.sum()),
                    new_pdm_losses_vs_C1=int((changed_vs_control < -1e-6).sum()),
                    pdm_improvements_vs_C1=int((changed_vs_control > 1e-6).sum()),
                    oracle_regret_mean_pp=float(q.oracle_regret_pp.mean()),
                    oracle_gain_macro_pp=float(aggregate(q.oracle_gain_pp.to_numpy(), sids, sessions).mean()),
                    extracted_oracle_fraction=float(vectors[method].mean() / max(1e-12, aggregate(q.oracle_gain_pp.to_numpy(), sids, sessions).mean())),
                    tail=dict(choices=int((q.ds_rank > 16).sum()), useful_repairs=int(((q.ds_rank > 16) & q.useful_repair).sum()),
                              losses=int(((q.ds_rank > 16) & q.raw_quality_degraded).sum()),
                              safety_harm=int(((q.ds_rank > 16) & q.raw_harm).sum())),
                    component_degradations={m: int((q["raw_delta_"+m] < -1e-8).sum()) for m in model.METRICS},
                    component_degraded_vs_C1_choice={m: int((q["delta_vs_C1_"+m] < -1e-8).sum()) for m in model.METRICS},
                    new_component_harm_vs_DS_not_in_C1={m: int(((q["raw_delta_"+m] < -1e-8) &
                        (baseline_frame.loc[tokens, "raw_delta_"+m] >= -1e-8)).sum()) for m in model.METRICS},
                    absolute_below_one={m: int((q["raw_"+m] < 1-1e-8).sum()) for m in model.METRICS},
                    slices={sl: dict(scenes=len(sub), gain_micro_pp=float(sub.raw_gain_pp.mean()),
                                      gain_macro_pp=float(sub.groupby("session_id").raw_gain_pp.mean().mean()))
                            for sl, sub in q.groupby("sampling_slice")})
        methods[method] = info
        for sid, gain in zip(sessions, vectors[method]):
            sub = q[q.session_id == sid]
            per_session.append(dict(method=method, session_id=sid, scenes=len(sub), raw_gain_pp=float(gain),
                                    interventions=int(sub.intervention.sum()), pdm_losses=int(sub.raw_quality_degraded.sum()),
                                    safety_harm=int(sub.raw_harm.sum())))
    for c in (4, 5):
        name = f"C{c}_seed_mean"
        vectors[name] = np.mean([vectors[f"C{c}_{s}"] for s in model.CONTROL_SEEDS], axis=0)
        scene_vectors[name] = np.mean([scene_vectors[f"C{c}_{s}"] for s in model.CONTROL_SEEDS], axis=0)
        methods[name] = effect_stats(vectors[name], draws)
        methods[name]["note"] = "Mean evaluation effect of five fixed policies; not a sixth deployable policy."
    comparisons = {"C2_minus_" + other: effect_stats(vectors["C2_memory"] - vectors[other], draws)
                   for other in ("C1_selector", "C3_corrector", "C4_seed_mean", "C5_seed_mean")}
    d = comparisons["C2_minus_C1_selector"]
    mem, control = methods["C2_memory"], methods["C1_selector"]
    criteria = dict(positive_over_DS=mem["mean_pp"] > 0,
                    incremental_at_least_quarter_pp=d["mean_pp"] >= .25,
                    all_paired_intervals_above_zero=all(c["ci95_sessions"][0] > 0 for c in comparisons.values()),
                    positive_without_best_one_two=d["leave_best_one_pp"] > 0 and d["leave_best_two_pp"] > 0,
                    three_sessions_half_pp=d["sessions_improved_at_least_half_pp"] >= 3,
                    no_new_safety_harm=mem["new_safety_harm_vs_C1"] == 0,
                    losses_not_more_than_C1=mem["pdm_losses"] <= control["pdm_losses"],
                    worst_loss_not_worse_than_C1=mem["worst_scene_gain_pp"] >= control["worst_scene_gain_pp"] - 1e-6)
    positive = all(criteria.values())
    oracle_rows = frame[frame.method == "C0_DS"]
    oracle = dict(top64_gain_macro_pp=float(oracle_rows.groupby("session_id").oracle_gain_pp.mean().mean()),
                  safe_top64_gain_macro_pp=float(oracle_rows.groupby("session_id").safe_oracle_gain_pp.mean().mean()),
                  strict_top64_over_top16_macro_pp=float(oracle_rows.groupby("session_id").extra_top64_over_top16_pp.mean().mean()),
                  scenes_with_strict_tail_reserve=int(oracle_rows.unique_tail_oracle.sum()),
                  scenes_with_useful_safe_repair=int((oracle_rows.safe_oracle_gain_pp >= .5).sum()))
    summary = dict(status="complete", protocol=PROTOCOL, scenes=len(entries), sessions=len(sessions),
                   primary="DS", research_candidate="C2_memory" if positive else None,
                   continue_memory_hypothesis=positive, criteria=criteria, methods=methods, paired=comparisons, oracle=oracle,
                   protected_holdout_processed=0, new_scorer_calls=0, new_drex_calls=0, note=model.NOTE)
    for path, df in ((out / "per_scene.csv", frame), (out / "per_session.csv", pd.DataFrame(per_session))):
        tmp = path.with_name(path.name + ".tmp")
        df.to_csv(tmp, index=False); tmp.replace(path)
    write(out / "paired_comparisons.json", comparisons)
    write(out / "summary.json", summary)
    names = ("per_scene.csv", "per_session.csv", "bootstrap_draws.npz", "paired_comparisons.json", "summary.json")
    write(out / "report_commit.json", dict(spec=report_spec, files={n: sha(out / n) for n in names}))
    return summary


def status(out):
    result = dict(status="not_planned", root=str(out))
    if (out / "identity.json").exists():
        identity = read(out / "identity.json")
        result.update(status="planned", counts=identity["counts"],
                      completed_fits=len(list((out / "fits").glob("*/commit.json"))), expected_fits=14,
                      completed_folds=len(list((out / "folds").glob("*/commit.json"))))
    if (out / "progress.json").exists():
        result["progress"] = read(out / "progress.json")
    if (out / "summary.json").exists() and (out / "report_commit.json").exists():
        result["status"] = "complete"
    if (out / "failure.json").exists():
        result["last_failure"] = read(out / "failure.json")
    return result


def synthetic():
    rng = np.random.default_rng(61)
    vocab = np.zeros((256, 40, 3), dtype=np.float32)
    vocab[:, :, 0] = np.linspace(0, 20, 40)
    vocab[:, :, 1] = np.arange(256)[:, None] / 256
    entries, raws = [], {}
    for sid in range(8):
        sample = dict(candidate_ids=np.arange(256), selected_local=np.array(0),
                      selection_scores=-np.arange(256) * .01, scene_embedding=rng.normal(size=4),
                      ego_status=rng.normal(size=8), decoder_states=rng.normal(0, .01, (256, 256)))
        sample.update({"head_" + m: rng.normal(0, .2, 256) for m in (*model.METRICS, "imi")})
        e = model.input_entry(sample, vocab, dict(token="t"+str(sid), session_id="s"+str(sid), role="train", sampling_slice="uniform"))
        raw = np.ones((64, 9)); raw[:, 0] = .6
        raw[3, 0], raw[5, 0], raw[5, 2] = .8, .1, 0.
        entries.append(e); raws[e["token"]] = raw
    return entries, raws


def smoke(parent):
    config = replace(model.Config(), width=8, epochs=2, probes=16, bootstrap_reps=100)
    with tempfile.TemporaryDirectory(prefix="drivemem_residual_smoke_", dir=parent) as name:
        out = Path(name)
        entries, raws = synthetic()
        initialize(out, entries, dict(bindings={}), config, 1, True)
        prepare(out, entries, raws)
        entries = prepared(out)
        groups = read(out / "split.json")["groups"]
        for i in range(4):
            run_fold(out, entries, groups, i, config)
        result = evaluate(out, entries, groups, config)
        for i in range(4):
            run_fold(out, entries, groups, i, config)
        assert evaluate(out, entries, groups, config) == result
        assert len(list((out / "fits").glob("*/commit.json"))) == 14
        return dict(status="smoke_passed", synthetic=True, fits=14, resumed=True, scenes=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inventory", "plan", "pilot", "all", "report", "status", "smoke"))
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2"))
    parser.add_argument("--bundle", type=Path, default=base / "drivemem_memory/selector_training_bundle_v3_30gb")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--label-request", type=Path)
    parser.add_argument("--package-manifest", type=Path)
    parser.add_argument("--out", type=Path, default=base / "drivemem_memory/experiments/residual_memory_top64_v1")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    config = model.Config()
    if args.stage == "status":
        print(json.dumps(status(args.out), indent=2)); return
    if args.stage == "smoke":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(smoke(args.out.parent), indent=2)); return
    if args.stage == "inventory":
        entries, _, provenance = collect(args)
        print(json.dumps(dict(status="ready", scenes=len(entries), sessions=len({e['session_id'] for e in entries}),
                              topk=64, candidate_labels=len(entries)*64, labels=provenance['labels'],
                              expected_fits=14, device="cpu", note=model.NOTE), indent=2)); return
    with locked(args.out):
        try:
            if args.stage == "plan":
                entries, raws, provenance = collect(args)
                identity = initialize(args.out, entries, provenance, config, args.threads)
                prepare(args.out, entries, raws)
                print(json.dumps(dict(status="planned", **identity["counts"], expected_fits=14, root=str(args.out)), indent=2))
                return
            identity = verify_root(args.out, config, args.threads)
            entries = prepared(args.out)
            groups = identity["groups"]
            if args.stage == "pilot":
                fit_model(args.out, entries, groups, (0, 1), config)
                info = read(args.out / "fits/base_01/commit.json")
                print(json.dumps(dict(status="pilot_complete", seconds=info["seconds"],
                                      process_peak_rss_mib=read(args.out / "fits/base_01/history.json")["process_peak_rss_mib"],
                                      evaluated_query_outcomes=0, note="Training fit only; cached for all. Full-run time is not guaranteed."), indent=2))
            elif args.stage == "all":
                for i in range(4):
                    run_fold(args.out, entries, groups, i, config)
                result = evaluate(args.out, entries, groups, config)
                write(args.out / "progress.json", dict(status="complete", scenes=len(entries)))
                print(json.dumps(dict(status=result["status"], primary=result["primary"],
                                      continue_memory_hypothesis=result["continue_memory_hypothesis"],
                                      paired=result["paired"], criteria=result["criteria"]), indent=2))
            else:
                print(json.dumps(evaluate(args.out, entries, groups, config), indent=2))
        except Exception as exc:
            failure = dict(time=datetime.now(timezone.utc).isoformat(), stage=args.stage, error=repr(exc), traceback=traceback.format_exc())
            write(args.out / "failure.json", failure)
            write(args.out / "failures" / (str(time.time_ns()) + ".json"), failure)
            raise


if __name__ == "__main__":
    main()
