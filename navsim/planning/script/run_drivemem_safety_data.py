"""Screen and label a frozen M/F-only expansion. Never train or read query labels."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.agents.drivemem import METRICS, SAFETY, probabilities, token_seed
from navsim.agents.drivemem_action_transfer import pool


def select_labels(samples, seed):
    result = []
    for sid in sorted({s["session_id"] for s in samples}):
        group = sorted([s for s in samples if s["session_id"] == sid], key=lambda s: s["token"])
        if len(group) != 16 or len({s["role"] for s in group}) != 1 or group[0]["role"] not in ("memory", "fit"):
            raise ValueError("Expected sixteen screening scenes per M/F session")
        order = np.random.default_rng(token_seed(sid, seed)).permutation(16)
        selected = {int(i): "uniform" for i in order[:4]}
        # Two degradation and two repair proxies. Component order rotates by
        # session, without query/label information. Scores are not probabilities.
        components = np.roll(np.array(SAFETY), token_seed(sid, seed) % len(SAFETY))
        for slot, kind in enumerate(("degrade", "repair", "degrade", "repair")):
            j = int(components[slot])
            scores = []
            for n, s in enumerate(group):
                p = probabilities(s)
                b = int(s["selected_local"])
                if b not in pool(s):
                    raise ValueError("Baseline outside top16")
                diff = p[pool(s), j] - p[b, j]
                score = max(0., -diff.min()) if kind == "degrade" else max(0., diff.max()) * (1 - p[b, j])
                scores.append((float(score), n))
            n = max((item for item in scores if item[1] not in selected), key=lambda t: (t[0], -t[1]))[1]
            selected[n] = "challenge_" + kind + "_" + METRICS[j]
        result.extend(dict(token=group[i]["token"], session_id=sid, role=group[i]["role"], sampling=tag)
                      for i, tag in sorted(selected.items()))
    return result


def collect(samples, selection, root, evaluate, read, write, sha, verify, limit):
    completed, evaluated = 0, 0
    start = time.monotonic()
    for r in selection:
        s = samples[r["token"]]
        path = root / "top16_labels" / (s["token"] + ".json")
        ix = pool(s).tolist()
        spec = dict(token=s["token"], session_id=s["session_id"], role=s["role"], sampling=r["sampling"],
                    feature_sha256=sha(root / "features" / (s["token"] + ".npz")), top16=ix)
        state = read(path) if path.exists() else dict(**spec, outcomes={})
        if any(state.get(k) != v for k, v in spec.items()) or not set(state["outcomes"]) <= {str(i) for i in ix}:
            raise RuntimeError("Label provenance changed")
        if len(state["outcomes"]) == 16:
            completed += 1
            continue
        if limit and evaluated >= limit:
            break
        b = int(s["selected_local"])
        # Reuse baseline if already saved on resume, verifying by a fresh repeat.
        baseline, _ = evaluate(s["token"], int(s["candidate_ids"][b]))
        if str(b) in state["outcomes"]:
            verify(state["outcomes"][str(b)], baseline)
        else:
            state["outcomes"][str(b)] = baseline
            write(path, state)
        for i in ix:
            if str(i) in state["outcomes"]:
                continue
            state["outcomes"][str(i)], _ = evaluate(s["token"], int(s["candidate_ids"][i]))
            write(path, state)
        evaluated += 1
        completed += 1
        print(f"TOP16 {completed}/{len(selection)} token={s['token']} sampling={r['sampling']}", flush=True)
    return dict(completed_scenes=completed, newly_completed_scenes=evaluated, seconds=time.monotonic() - start)


def support(samples, selection, root, read, sha):
    rows, missing, hashes, scenes = [], [], {}, []
    for r in selection:
        s = samples[r["token"]]
        path = root / "top16_labels" / (r["token"] + ".json")
        if not path.exists() or len(read(path)["outcomes"]) != 16:
            missing.append(r["token"])
            continue
        state = read(path)
        if (state["feature_sha256"] != sha(root / "features" / (s["token"] + ".npz")) or
                any(state.get(k) != r[k] for k in ("token", "session_id", "role", "sampling")) or state["top16"] != pool(s).tolist()):
            raise RuntimeError("Label provenance changed")
        values = state["outcomes"]
        baseline = values[str(int(s["selected_local"]))]["raw"]
        hashes[path.name] = sha(path)
        scores = [values[str(i)]["raw"]["pdm_score"] for i in pool(s)]
        scenes.append(dict(token=r["token"], role=r["role"], session_id=r["session_id"],
            slice="uniform" if r["sampling"] == "uniform" else "challenge",
            baseline=float(baseline["pdm_score"]), oracle_gain_pp=100 * (max(scores) - baseline["pdm_score"])))
        for i in pool(s):
            if int(i) == int(s["selected_local"]):
                continue
            for j in SAFETY:
                k = METRICS[j]
                v = float(values[str(i)]["raw"][k])
                if not np.isfinite(v) or not 0 <= v <= 1:
                    raise ValueError("Invalid scorer label")
                rows.append(dict(role=r["role"], slice="uniform" if r["sampling"] == "uniform" else "challenge",
                    metric=k, session_id=r["session_id"], token=r["token"], below_one=v < 1 - 1e-8,
                    degraded=v < baseline[k] - 1e-8, improved=v > baseline[k] + 1e-8))
    frame = pd.DataFrame(rows)
    if scenes:
        print("M/F BASELINE AND ACTION RESERVE (descriptive; uniform/challenge kept separate):")
        print(pd.DataFrame(scenes).groupby(["role", "slice"]).agg(scenes=("token", "size"),
              baseline=("baseline", "mean"), oracle_gain_pp=("oracle_gain_pp", "mean")).to_string())
    if len(frame):
        print(frame.groupby(["role", "slice", "metric"]).agg(actions=("token", "size"),
            below_one=("below_one", "sum"), degraded=("degraded", "sum"), improved=("improved", "sum")).to_string())
        print("\nSESSIONS WITH DEGRADATION:")
        print(frame[frame.degraded].groupby(["role", "slice", "metric", "session_id"]).size().to_string())
    return dict(status="complete" if not missing else "partial", selected_scenes=len(selection),
                completed_scenes=len(hashes), missing_tokens=missing, labels=hashes, scenes=scenes,
                note="M/F collection only. No fresh holdout, calibrated risk or model success established.")


def main(cfg):
    import os
    from navsim.planning.script.run_drivemem_memory import verify_identity, read_json as read, write_json as write, sha, sample, evaluators
    from navsim.planning.script.run_drivemem_development_oracle import verify_baseline
    root = Path(cfg.drivemem.root)
    verify_identity(cfg, root)
    stage = str(cfg.drivemem.stage).removeprefix("data-")
    if stage not in ("validate", "screen", "label", "support-report"):
        raise ValueError("Unknown collection stage")
    protocol = read(root / "expansion_protocol.json")
    repo = Path(__file__).resolve().parents[3]
    if sha(repo / "download/prepare_drivemem_mf_expansion.py") != protocol["preparer"]:
        raise RuntimeError("Frozen data preparer changed")
    for location, files in protocol["inputs"].items():
        for name, digest in files.items():
            if sha(Path(location) / name) != digest:
                raise RuntimeError("Collection source changed")
    if stage == "validate":
        print("Expansion identity and Hydra configuration: OK", flush=True)
        return
    manifest = dict(protocol_sha256=sha(root / "expansion_protocol.json"), export_index=sha(root / "export_index.json"),
        vocab=sha(root / "vocab.npz"), code=sha(Path(__file__)), splits=sha(root / "split_identity.json"),
        cache_index=sha(root / "cache_index.json"))
    records = read(root / "splits.json")["records"]
    samples = {r["token"]: sample(root, r) for r in records}
    if len(samples) != 160 or any(s["role"] not in ("memory", "fit") for s in samples.values()):
        raise ValueError("Expected 160 M/F-only screening records")
    selected = select_labels(list(samples.values()), protocol["seed"])
    if len(selected) != 80:
        raise ValueError("Expected exactly 80 label scenes")
    state = dict(identity=manifest, records=selected)
    marker = root / "label_selection.json"
    if marker.exists() and read(marker) != state:
        raise RuntimeError("Frozen screening/code/inputs changed; use a new dataset root")
    if not marker.exists() and stage != "screen":
        raise RuntimeError("Run screen first to freeze choices before labels")
    if stage == "screen":
        write(marker, state)
        print(json.dumps(dict(selected_scenes=80, uniform=40, challenge=40, candidate_evaluations=1280,
                              raw_filtered_calls=2560, note="Choices frozen without outcomes"), indent=2))
        return
    previous = root / "support_report.json"
    if previous.exists():
        for name, digest in read(previous)["labels"].items():
            if sha(root / "top16_labels" / name) != digest:
                raise RuntimeError("Previously committed complete labels changed")
    if stage == "label":
        limit = int(os.environ.get("DRIVEMEM_LABEL_LIMIT", "8"))
        if not 1 <= limit <= 80:
            raise ValueError("DRIVEMEM_LABEL_LIMIT must be between 1 and 80")
        evaluate, _ = evaluators(cfg, root)
        print(json.dumps(collect(samples, selected, root, evaluate, read, write, sha, verify_baseline, limit), indent=2))
    report = support(samples, selected, root, read, sha)
    write(root / "support_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("labels", "scenes")}, indent=2), flush=True)


if __name__ == "__main__":
    import hydra
    hydra.main(config_path="config/pdm_scoring", config_name="default_drivemem_memory", version_base=None)(main)()
