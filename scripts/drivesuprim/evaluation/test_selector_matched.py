from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import selector_matched as matched


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_bundle(root, members, parent=None):
    index = {}
    for token, sid, origin in members:
        path = root / "examples" / (token + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=np.ones((2, 3), dtype=np.float32), label_raw=np.ones((2, 9)))
        index[token] = dict(sha256=matched.sha(path), metadata=dict(token=token, session_id=sid, role="train", origin=origin))
    np.savez_compressed(root / "vocab.npz", vocab=np.zeros((2, 3, 3)))
    write(root / "examples_index.json", index)
    counts = dict(scenes=len(members), sessions=len({m[1] for m in members}))
    inputs = {} if parent is None else {str(parent / "bundle_manifest.json"): matched.sha(parent / "bundle_manifest.json")}
    write(root / "identity.json", dict(counts=counts, inputs=inputs, protected_sessions=["fresh"]))
    write(root / "bundle_manifest.json", dict(counts=counts, identity_sha256=matched.sha(root / "identity.json"),
          index_sha256=matched.sha(root / "examples_index.json"), vocab_sha256=matched.sha(root / "vocab.npz")))
    return index


def make_run(root, bundle, index, expanded=False):
    rows = []
    for method in matched.METHODS:
        for token, item in index.items():
            gain = 0.
            if expanded and method == "control_v1_raw":
                gain = {"a0": 2., "a1": 2., "c0": -50.}.get(token, 0.)
            row = dict(method=method, token=token, session_id=item["metadata"]["session_id"], origin=item["metadata"]["origin"],
                       slice="uniform", candidate_id=1 if gain else 0, raw_score=.8 + gain / 100, raw_gain_pp=gain,
                       oracle_score=.95, intervention=gain != 0, raw_harm=gain < -10, raw_quality_degraded=gain < 0,
                       predicted_gain_pp=1., benefit_estimate=.8)
            row.update({"raw_delta_" + m: -1. if gain < -10 and m == "drivable_area_compliance" else 0. for m in matched.METRICS})
            rows.append(row)
    root.mkdir()
    pd.DataFrame(rows).to_csv(root / "fit_cv_per_scene.csv", index=False)
    config = dict(epochs=120, seed=7)
    write(root / "identity.json", dict(config=config, inputs={str(bundle / name): matched.sha(bundle / name)
          for name in ("identity.json", "bundle_manifest.json")}, sources={"model": "frozen"}, threads=2,
          python="3.9", numpy="same", torch="same", platform="Linux", protocol={}, thresholds=[.5, .75, .9, 1.01]))
    write(root / "fit.json", dict(status="complete", config=config, fresh_holdout_processed=0,
          artifacts={"fit_cv_per_scene.csv": matched.sha(root / "fit_cv_per_scene.csv")},
          selected={f: dict(threshold=(1.01 if expanded else .9) if f == "consistent_v2" else (.9 if expanded else .75)) for f in matched.FAMILIES}))
    ordered = sorted(index, key=lambda t: (index[t]["metadata"]["session_id"], t))
    sessions = sorted({e["metadata"]["session_id"] for e in index.values()})
    for family in matched.FAMILIES:
        for sid in sessions:
            path = root / "cv" / (family + "_" + matched.hashlib.sha256(sid.encode()).hexdigest()[:16])
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path.with_suffix(".npz"), predictions=np.zeros((2, 3)))
            partition = dict(train_sessions=[s for s in sessions if s != sid], valid_session=sid,
                train_tokens=[t for t in ordered if index[t]["metadata"]["session_id"] != sid],
                valid_tokens=[t for t in ordered if index[t]["metadata"]["session_id"] == sid])
            write(path.with_suffix(".json"), dict(partition=partition, sha256=matched.sha(path.with_suffix(".npz"))))


def setup(base):
    old_bundle, new_bundle, reference, expanded = [base / n for n in ("old-bundle", "new-bundle", "reference", "expanded")]
    old = [("a0", "a", "legacy"), ("a1", "a", "legacy"), ("b0", "b", "new")]
    new = old + [("a-extra", "a", "expansion30"), ("c0", "c", "expansion30")]
    oi, ni = make_bundle(old_bundle, old), make_bundle(new_bundle, new, old_bundle)
    make_run(reference, old_bundle, oi)
    make_run(expanded, new_bundle, ni, expanded=True)
    return reference, expanded, old_bundle, new_bundle


class MatchedTests(unittest.TestCase):
    def test_identical_scene_comparison_separates_added_difficulty_and_is_read_only(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            base = Path(tmp).resolve()
            paths = setup(base)
            before = {str(p): matched.sha(p) for p in base.rglob("*") if p.is_file()}
            result = matched.compare(*paths)
            self.assertEqual(result["shared_examples_verified"], 3)
            shared = result["shared"]
            self.assertEqual(shared["expanded"]["methods"]["control_v1_raw"]["gain_macro_pp"], 1.)
            self.assertEqual(shared["paired"]["control_v1_raw"]["expanded_minus_reference_macro_pp"], 1.)
            self.assertEqual(result["added_new_session_scenes"]["scenes"], 1)
            self.assertEqual(result["added_old_session_scenes"]["scenes"], 1)
            self.assertEqual(result["added_scenes"]["methods"]["control_v1_raw"]["worst_loss_pp"], -50.)
            self.assertEqual(result["fresh_holdout_processed"], 0)
            self.assertEqual(result["historical_outcomes_read"], 0)
            self.assertEqual(before, {str(p): matched.sha(p) for p in base.rglob("*") if p.is_file()})

    def test_changed_shared_labels_detected_even_with_updated_archive_hash(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            paths = setup(Path(tmp).resolve())
            root = paths[3]
            path = root / "examples/a0.npz"
            with np.load(path, allow_pickle=False) as f:
                arrays = {k: f[k].copy() for k in f.files}
            arrays["label_raw"][0, 0] = .1
            np.savez_compressed(path, **arrays)
            index = matched.read(root / "examples_index.json")
            index["a0"]["sha256"] = matched.sha(path)
            write(root / "examples_index.json", index)
            manifest = matched.read(root / "bundle_manifest.json")
            manifest["index_sha256"] = matched.sha(root / "examples_index.json")
            write(root / "bundle_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Shared input/label values"):
                matched.compare(*paths)

    def test_missing_scene_rows_and_changed_csv_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            paths = setup(Path(tmp).resolve())
            root = paths[1]
            frame = pd.read_csv(root / "fit_cv_per_scene.csv").iloc[:-1]
            frame.to_csv(root / "fit_cv_per_scene.csv", index=False)
            with self.assertRaisesRegex(RuntimeError, "fitted artifact changed"):
                matched.compare(*paths)
            fit = matched.read(root / "fit.json")
            fit["artifacts"]["fit_cv_per_scene.csv"] = matched.sha(root / "fit_cv_per_scene.csv")
            write(root / "fit.json", fit)
            with self.assertRaisesRegex(ValueError, "Complete unique scene coverage"):
                matched.compare(*paths)

    def test_whole_session_exclusion_checks_new_scenes_in_old_sessions(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            paths = setup(Path(tmp).resolve())
            root = paths[1]
            fold = root / "cv" / ("control_v1_" + matched.hashlib.sha256(b"a").hexdigest()[:16] + ".json")
            state = matched.read(fold)
            state["partition"]["train_tokens"].append("a-extra")
            write(fold, state)
            with self.assertRaisesRegex(ValueError, "Whole-session CV"):
                matched.compare(*paths)

    def test_different_training_conditions_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            paths = setup(Path(tmp).resolve())
            root = paths[1]
            identity = matched.read(root / "identity.json")
            identity["threads"] = 4
            write(root / "identity.json", identity)
            with self.assertRaisesRegex(ValueError, "condition changed: threads"):
                matched.compare(*paths)

    def test_paired_baseline_and_oracle_must_match(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            paths = setup(Path(tmp).resolve())
            root = paths[1]
            frame = pd.read_csv(root / "fit_cv_per_scene.csv")
            frame.loc[0, "raw_score"] += .01
            frame.to_csv(root / "fit_cv_per_scene.csv", index=False)
            fit = matched.read(root / "fit.json")
            fit["artifacts"]["fit_cv_per_scene.csv"] = matched.sha(root / "fit_cv_per_scene.csv")
            write(root / "fit.json", fit)
            with self.assertRaisesRegex(ValueError, "Shared baseline/oracle"):
                matched.compare(*paths)


if __name__ == "__main__":
    unittest.main()
