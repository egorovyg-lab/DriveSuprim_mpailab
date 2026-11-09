import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from navsim.planning.script import run_selector_collection as collect


def fixture():
    vocab = np.arange(256 * 40 * 3, dtype=np.float32).reshape(256, 40, 3)
    a = dict(candidate_ids=np.arange(256), selected_local=np.array(0),
             selection_scores=np.linspace(2., 0., 256), final_traj=vocab[0].copy())
    a.update({"head_" + k: np.ones(256) for k in (*collect.METRICS, "imi")})
    extra = dict(decoder_states=np.zeros((256, 256), dtype=np.float32),
                 scene_embedding=np.zeros(8), ego_status=np.zeros(8))
    return a, extra, vocab


def record(token, sid="s", city="city", role="train"):
    return dict(token=token, session_id=sid, log_name=sid + "_00000_00100", timestamp_us=10,
                map_name=city, role=role)


def prepared_example(out, token="t", index=None):
    a, extra, vocab = fixture()
    collect.write(out / "runtime.json", dict(settings=collect.PROFILE))
    collect.save(out / "vocab.npz", vocab=vocab)
    collect.write(out / "vocab_identity.json", dict(sha256=collect.sha(out / "vocab.npz")))
    collect.save(out / "native" / (token + ".npz"), **a)
    native = collect.read(out / "native_index.json") if (out / "native_index.json").exists() else {}
    native[token] = collect.sha(out / "native" / (token + ".npz"))
    collect.write(out / "native_index.json", native)
    index = {} if index is None else index
    collect.commit_feature(out, token, a, copy.deepcopy(a), copy.deepcopy(a), vocab, extra, index, [])
    cache = out / "metric_cache" / (token + ".bin")
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(b"cache-fixture")
    old = collect.read(out / "cache_index.json") if (out / "cache_index.json").exists() else {}
    old[token] = dict(path=str(cache), sha256=collect.sha(cache))
    collect.write(out / "cache_index.json", old)
    return index


class CollectionTests(unittest.TestCase):
    def test_train_only_pilot_city_coverage_and_overlap_rejection(self):
        records = [record(str(i), "session" + str(i), "city" + str(i % 4)) for i in range(12)]
        records.append(record("holdout", "holdout", "city0", "holdout"))
        selection = dict(records=records, legacy_training_records=[record("old", "old")])
        ordered = collect.train_records(selection)
        self.assertEqual(len(ordered), 12)
        self.assertEqual(len({r["map_name"] for r in ordered[:4]}), 4)
        self.assertTrue(all(r["role"] == "train" for r in ordered))
        selection["legacy_training_records"][0]["session_id"] = "holdout"
        with self.assertRaisesRegex(ValueError, "holdout overlaps"):
            collect.train_records(selection)

    def test_export_commit_requires_current_repeat_and_native_functional_consistency(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp).resolve()
            index = prepared_example(out)
            self.assertEqual(collect.checked_features(out, [record("t")]), index)
            a, extra, vocab = fixture()
            changed = copy.deepcopy(a)
            changed["selection_scores"][1] += .001
            with self.assertRaisesRegex(RuntimeError, "bitwise"):
                collect.commit_feature(out, "bad", a, a, changed, vocab, extra, {}, [])
            changed = copy.deepcopy(a)
            changed["head_no_at_fault_collisions"] += .001
            with self.assertRaisesRegex(RuntimeError, "Metric logit"):
                collect.commit_feature(out, "bad", a, changed, changed, vocab, extra, {}, [])
            self.assertFalse((out / "features/bad.npz").exists())

    def test_feature_nan_alignment_and_native_link_are_rejected(self):
        a, extra, vocab = fixture()
        extra["decoder_states"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            collect.validate_sample(dict(a, **extra), vocab)
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp).resolve()
            prepared_example(out)
            collect.save(out / "native/t.npz", **dict(a, selection_scores=a["selection_scores"] + 1))
            collect.write(out / "native_index.json", dict(t=collect.sha(out / "native/t.npz")))
            with self.assertRaisesRegex(RuntimeError, "Native snapshot/export"):
                collect.checked_features(out, [record("t")])

    def test_label_resume_checks_baseline_preserves_actions_and_audits_DAC(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp).resolve()
            index = prepared_example(out)
            records = [record("t")]
            calls = []
            def score(token, candidate):
                calls.append(candidate)
                v = dict(pdm_score=.5 if candidate == 0 else .9, **{k: 1. for k in collect.METRICS})
                if candidate == 5:
                    v.update(drivable_area_compliance=0., pdm_score=0.)
                return dict(raw=v, filtered=v.copy()), None
            def interrupted(token, candidate):
                if len(calls) >= 3:
                    raise RuntimeError("interrupted")
                return score(token, candidate)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                collect.collect_labels(out, records, index, interrupted, 8)
            partial = collect.read(out / "top16_labels/t.json")
            self.assertEqual(len(partial["outcomes"]), 3)
            with redirect_stdout(io.StringIO()):
                timing = collect.collect_labels(out, records, index, score, 8)
                report = collect.report(out, dict(legacy_scenes=112), records)
            self.assertEqual(timing["completed_scenes"], 1)
            self.assertEqual(calls.count(0), 2)  # Baseline repeated on partial resume.
            self.assertEqual(calls.count(1), 1)
            self.assertEqual(report["projected_total_train_scenes"], 113)
            self.assertEqual(report["safety_support"]["drivable_area_compliance"]["degradation_sessions"], 1)
            with mock.patch.object(collect, "check_outcomes", wraps=collect.check_outcomes), redirect_stdout(io.StringIO()):
                result = collect.collect_labels(out, records, index, lambda *a: self.fail("already completed"), 8)
            self.assertEqual(result["newly_completed_scenes"], 0)
            state = collect.read(out / "top16_labels/t.json")
            state["outcomes"]["1"]["raw"]["pdm_score"] = .7
            collect.write(out / "top16_labels/t.json", state)
            with self.assertRaisesRegex(RuntimeError, "Previously completed"):
                collect.collect_labels(out, records, index, score, 8)

    def test_label_refuses_holdout_and_changed_baseline(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp).resolve()
            index = prepared_example(out)
            with self.assertRaisesRegex(ValueError, "non-training"):
                collect.collect_labels(out, [record("t", role="holdout")], index, lambda *a: self.fail(), 8)
            s = collect.load_sample(out, record("t"), index)
            values = dict(raw=dict(pdm_score=.5, **{k: 1. for k in collect.METRICS}))
            values["filtered"] = values["raw"].copy()
            collect.write(out / "top16_labels/t.json", dict(token="t", session_id="s", role="train",
                feature_sha256=index["t"]["sha256"], top16=list(range(16)), candidate_ids=list(range(16)), outcomes={"0": values}))
            different = copy.deepcopy(values)
            different["raw"]["pdm_score"] = .6
            with self.assertRaisesRegex(RuntimeError, "baseline scorer changed"):
                collect.collect_labels(out, [record("t")], index, lambda *a: (different, None), 8)

    def test_initializer_resumes_and_pins_train_only_membership(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            base = Path(tmp).resolve()
            data, source, out, images = [base / n for n in ("data", "source", "out", "images")]
            images.mkdir()
            checkpoint = base / "teacher.ckpt"
            checkpoint.write_bytes(b"teacher")
            template = dict(agent=dict(checkpoint_path=str(checkpoint)), checkpoint_sha256=collect.sha(checkpoint),
                            sources={}, navsim_log_path=str(base / "logs"), original_sensor_path=str(images))
            collect.write(source / "identity.json", template)
            plan = dict(options={k: str(base / k) for k in ("source", "addition", "inventory", "validation")})
            collect.write(data / "acquisition_plan.json", plan)
            selection = dict(records=[record("new", "train"), record("closed", "closed", role="holdout")],
                             legacy_training_records=[record("old", "old")], fresh_holdout_sessions=["closed"])
            collect.write(data / "selection.json", selection)
            image = images / "train.jpg"
            image.write_bytes(b"image")
            info = dict(size=image.stat().st_size, sha256=collect.sha(image))
            train_requests = dict(archives=[], images=["train.jpg"], initial_images={"train.jpg": info},
                                  log_hashes={str(base / "logs/train_00000_00100.pkl"): "log-hash"})
            with mock.patch.object(collect.acquisition, "verify", return_value=(plan, {})), \
                    mock.patch.object(collect.acquisition, "image_status", return_value=({"train.jpg": info}, {"status": "complete"})), \
                    mock.patch.object(collect.acquisition, "request_images", return_value=train_requests), redirect_stdout(io.StringIO()):
                _, identity = collect.initialize(data, out)
                self.assertEqual(collect.initialize(data, out)[1], identity)
                self.assertEqual([r["token"] for r in collect.read(out / "splits.json")["records"]], ["new"])
                collect.verify_train_images(out)
                image.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "train image"):
                    collect.verify_train_images(out)
                collect.write(out / "train_images.json", {})
                with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                    collect.verify_collection(out)

    def test_indices_reject_closed_tokens_and_external_cache_paths(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp).resolve()
            collect.write(out / "export_index.json", dict(holdout="digest"))
            with self.assertRaisesRegex(ValueError, "Unexpected scene"):
                collect.checked_index(out, "features", "export_index.json", [record("train")])
            collect.write(out / "cache_index.json", dict(train=dict(path=str(out.parent / "external.bin"), sha256="digest")))
            with self.assertRaisesRegex(ValueError, "escapes"):
                collect.checked_index(out, "", "cache_index.json", [record("train")])

    def test_cli_refuses_labelling_before_all_exports(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            base = Path(tmp).resolve()
            data, out = base / "data", base / "out"
            checkpoint = base / "teacher.ckpt"
            checkpoint.write_bytes(b"teacher")
            collect.write(out / "source_config.json", dict(agent=dict(checkpoint_path=str(checkpoint)), map_root=str(base / "maps")))
            identity = dict(acquisition_root=str(data), checkpoint_sha256=collect.sha(checkpoint))
            argv = ["collector", "label-pilot", "--data", str(data), "--out", str(out)]
            with mock.patch("sys.argv", argv), mock.patch.object(collect, "verify_collection", return_value=(identity, [record("train")])), \
                    mock.patch.object(collect, "cfg_from_source", return_value=object()), mock.patch.dict(collect.os.environ, {"NUPLAN_MAPS_ROOT": str(base / "maps")}):
                with self.assertRaisesRegex(RuntimeError, "Complete cache and feature export"):
                    collect.main()
            self.assertFalse((out / "label_identity.json").exists())


if __name__ == "__main__":
    unittest.main()
