import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from navsim.agents import selector_training_bundle as bundle


def fixture(token="t0", sid="s0", origin="legacy"):
    vocab = np.arange(256 * 40 * 3, dtype=np.float32).reshape(256, 40, 3)
    sample = dict(candidate_ids=np.arange(256), selected_local=np.array(0),
                  selection_scores=np.linspace(2, 0, 256), decoder_states=np.zeros((256, 256)),
                  scene_embedding=np.zeros(8), ego_status=np.zeros(8))
    sample.update({"head_" + k: np.zeros(256) for k in (*bundle.METRICS, "imi")})
    outcomes = {}
    for i in range(16):
        raw = dict(pdm_score=.5, **{k: 1. for k in bundle.METRICS})
        raw["no_at_fault_collisions"] = .5
        if i == 3:
            raw.update(pdm_score=1., no_at_fault_collisions=1.)
        if i == 5:
            raw.update(pdm_score=0., drivable_area_compliance=0.)
        outcomes[str(i)] = dict(raw=raw, filtered=raw.copy())
    metadata = dict(token=token, session_id=sid, origin=origin, role="train", source_role="fit", map_name="city",
                    feature_protocol="test")
    return sample, outcomes, vocab, metadata


class BundleTests(unittest.TestCase):
    def test_inputs_and_labels_are_separate_stored_scores_preserved(self):
        s, y, vocab, meta = fixture()
        s.update(utility_values=np.ones(16) * 999, outcome_values=np.ones((16, 9)) * 999)
        e = bundle.example(s, y, vocab, meta)
        self.assertEqual(set(e["sample"]), set(bundle.INPUT_KEYS))
        self.assertEqual(e["label_raw"][0, 0], .5)
        self.assertEqual(e["label_indices"].tolist(), list(range(16)))
        y["0"]["filtered"]["pdm_score"] = .7
        self.assertEqual(bundle.example(s, y, vocab, meta)["label_raw"][0, 0], .5)
        with self.assertRaisesRegex(ValueError, "training examples"):
            bundle.example(s, y, vocab, dict(meta, role="holdout"))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            bundle.example(s, y, vocab, dict(meta, token="../escape"))

    def test_missing_labels_bad_values_and_shape_mismatch_rejected(self):
        s, y, vocab, meta = fixture()
        del y["15"]
        with self.assertRaisesRegex(ValueError, "Complete top16"):
            bundle.example(s, y, vocab, meta)
        s, y, vocab, meta = fixture()
        y["0"]["raw"]["pdm_score"] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid stored"):
            bundle.example(s, y, vocab, meta)
        s, y, vocab, meta = fixture()
        s["decoder_states"] = np.zeros((255, 256))
        with self.assertRaisesRegex(ValueError, "alignment"):
            bundle.example(s, y, vocab, meta)

    def test_union_enforces_tokens_session_separation_and_shapes(self):
        a = bundle.example(*fixture())
        b = bundle.example(*fixture("t1", "s1", "new"))
        result = bundle.merge_examples([a], [b], {"t0", "t1"}, {"closed"}, 2)
        self.assertEqual(len(result), 2)
        with self.assertRaisesRegex(ValueError, "protected"):
            bundle.merge_examples([a], [b], {"t0", "t1"}, {"s1"}, 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            bundle.merge_examples([a], [a], {"t0"}, set(), 1)
        changed = copy.deepcopy(b)
        changed["sample"]["ego_status"] = np.zeros(16)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            bundle.merge_examples([a], [changed], {"t0", "t1"}, set(), 2)

    def test_audit_counts_independent_scenes_and_sessions(self):
        a = bundle.example(*fixture())
        b = bundle.example(*fixture("t1", "s1", "new"))
        audit, rows = bundle.audit([a, b])
        json.dumps(audit, allow_nan=False)
        self.assertEqual((audit["scenes"], audit["sessions"]), (2, 2))
        self.assertEqual(audit["scenes_with_useful_alternative"], 2)
        self.assertAlmostEqual(audit["oracle_gain_macro_pp"], 50.)
        dac = audit["safety_support"]["drivable_area_compliance"]
        self.assertEqual(dac["actions"], 30)
        self.assertEqual(dac["degraded"], 2)
        self.assertEqual(dac["degradation_scenes"], 2)
        self.assertEqual(dac["degradation_sessions"], 2)
        self.assertEqual(len(rows), 2)

    def test_session_macro_differs_from_scene_micro(self):
        examples = [bundle.example(*fixture("a0", "a")), bundle.example(*fixture("a1", "a")),
                    bundle.example(*fixture("b0", "b", "new"))]
        examples[-1]["label_raw"][:, 0] = 1.
        audit, _ = bundle.audit(examples)
        self.assertAlmostEqual(audit["oracle_gain_micro_pp"], 100 / 3)
        self.assertAlmostEqual(audit["oracle_gain_macro_pp"], 25.)

    def test_build_replays_and_hashes_upstream_and_bundled_examples(self):
        examples = [bundle.example(*fixture()), bundle.example(*fixture("t1", "s1", "new"))]
        vocab = fixture()[2]
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            upstream = root / "upstream.json"
            upstream.write_text("frozen", encoding="utf-8")
            out = root / "bundle"
            inputs = {str(upstream): bundle.sha(upstream)}
            with redirect_stdout(io.StringIO()):
                first = bundle.build(out, examples, vocab, inputs, {"holdout"})
                self.assertEqual(first, bundle.build(out, examples, vocab, inputs, {"holdout"}))
            loaded, table = bundle.load_bundle(out)
            np.testing.assert_array_equal(table, vocab)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(set(loaded[0]["sample"]), set(bundle.INPUT_KEYS))
            upstream.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "upstream"):
                bundle.load_bundle(out)
            upstream.write_text("frozen", encoding="utf-8")
            p = out / "examples/t0.npz"
            p.write_bytes(p.read_bytes() + b"changed")
            with self.assertRaisesRegex(RuntimeError, "example changed"):
                bundle.load_bundle(out)

    def test_source_loader_checks_new_pool_and_legacy_new_configuration(self):
        old_sample, old_outcomes, vocab, old_meta = fixture("old", "old-session")
        old_sample.update(token="old", session_id="old-session", role="fit")
        new_sample, new_outcomes, _, new_meta = fixture("new", "new-session", "new")
        new_sample.update(token="new", session_id="new-session", role="train")
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            prior, addition, decoder, source, new_root, acq = [root / n for n in ("prior", "addition", "decoder", "source", "new", "acq")]
            config_keys = ("agent", "scorer", "simulator", "proposal_sampling", "traffic_agents", "traffic_agents_policy", "train_test_split",
                           "navsim_log_path", "original_sensor_path", "map_root", "checkpoint_sha256")
            config = {k: "same" for k in config_keys}
            bundle.write(source / "identity.json", config)
            bundle.write(decoder / "identity.json", dict(numerical_profile={"profile": 1}))
            bundle.write(addition / "label_selection.json", dict(records=[]))
            for p in (source / "features/old.npz", decoder / "features/source/old.npz"):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"frozen-old-input")
            identity = dict(acquisition_root=str(acq), numerical_profile={"profile": 1})
            records = [dict(token="new", session_id="new-session", role="train", map_name="city")]
            selection = dict(legacy_training_records=[dict(token="old", session_id="old-session", role="fit", map_name="city")],
                             records=records, excluded_sessions=["closed"], fresh_holdout_sessions=["holdout"])
            bundle.write(acq / "selection.json", selection)
            plan = dict(summary=dict(total_train_sessions=2))
            bundle.write(acq / "acquisition_plan.json", plan)
            for name in ("identity.json", "splits.json", "split_identity.json", "inventory.csv", "runtime.json",
                         "export_index.json", "native_index.json", "cache_index.json", "vocab_identity.json"):
                bundle.write(new_root / name, identity if name == "identity.json" else {})
            bundle.write(new_root / "source_config.json", config)
            bundle.save(new_root / "vocab.npz", vocab=vocab)
            for folder in ("features", "native"):
                bundle.save(new_root / folder / "new.npz", **{k: new_sample[k] for k in bundle.INPUT_KEYS})
            index = {"new": dict(sha256=bundle.sha(new_root / "features/new.npz"), native_sha256=bundle.sha(new_root / "native/new.npz"))}
            state = dict(token="new", session_id="new-session", role="train", top16=list(range(16)), candidate_ids=list(range(16)),
                         feature_sha256=index["new"]["sha256"], outcomes=new_outcomes)
            bundle.write(new_root / "top16_labels/new.json", state)
            bundle.write(new_root / "support_report.json", dict(status="complete", labelled_scenes=1, holdout_processed=0,
                         labels={"new.json": bundle.sha(new_root / "top16_labels/new.json")}))
            bundle.write(new_root / "label_identity.json", dict(collection_identity_sha256=bundle.sha(new_root / "identity.json"),
                export_index_sha256=bundle.sha(new_root / "export_index.json"), cache_index_sha256=bundle.sha(new_root / "cache_index.json"),
                vocab_sha256=bundle.sha(new_root / "vocab.npz")))
            legacy = ([old_sample], [], {"old": old_outcomes}, vocab, source, root / "validation", root / "completion", {}, {}, 1)
            with mock.patch.object(bundle, "load_decoder_inputs", return_value=legacy), \
                    mock.patch.object(bundle.collection, "verify_collection", return_value=(identity, records)), \
                    mock.patch.object(bundle.collection.acquisition, "verify", return_value=(plan, {})), \
                    mock.patch.object(bundle.collection, "checked_features", return_value=index), \
                    mock.patch.object(bundle.collection, "checked_index", return_value={}), \
                    mock.patch.object(bundle.collection, "load_sample", return_value=new_sample):
                examples, table, bindings, protected = bundle.load_sources(prior, addition, decoder, new_root)
                self.assertEqual(len(examples), 2)
                self.assertEqual({e["metadata"]["origin"] for e in examples}, {"legacy", "new"})
                self.assertIn("holdout", protected)
                self.assertIn(str(new_root / "top16_labels/new.json"), bindings)
                config["scorer"] = "different"
                bundle.write(new_root / "source_config.json", config)
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    bundle.load_sources(prior, addition, decoder, new_root)


if __name__ == "__main__":
    unittest.main()
