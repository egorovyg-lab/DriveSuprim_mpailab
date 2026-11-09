import argparse
from contextlib import redirect_stdout
import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import shutil
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np

spec = importlib.util.spec_from_file_location("expand30", Path("download/prepare_selector_expand30.py"))
expand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expand)
from navsim.agents import selector_training_bundle as bundle
from navsim.planning.script import run_selector_extend_bundle as union
from test_selector_bundle import fixture


class Expand30Tests(unittest.TestCase):
    def setup_images(self, base):
        previous, parent, archives, pool, logs, sensors, source, inventory = [base / n for n in (
            "previous", "parent", "archives", "pool", "logs", "sensors", "source", "inventory")]
        for path in (archives, pool, logs, sensors, inventory):
            path.mkdir()
        checkpoint = base / "checkpoint.bin"
        checkpoint.write_bytes(b"frozen checkpoint")
        keys = ("agent", "scorer", "simulator", "proposal_sampling", "traffic_agents", "traffic_agents_policy", "train_test_split",
                "navsim_log_path", "original_sensor_path", "map_root", "checkpoint_sha256")
        config = {k: "same" for k in keys}
        config.update(agent=dict(checkpoint_path=str(checkpoint)), checkpoint_sha256=expand.sha(checkpoint), sources={},
                      navsim_log_path=str(logs), original_sensor_path=str(sensors))
        expand.write(source / "identity.json", config)
        names = ["olda_veh-0", "oldb_veh-1", "fresh_veh-2", "newa_veh-3", "newb_veh-4", "seen_veh-5"]
        rows, mapping, listing, archive_sources = [], {}, [], {}
        for shard, sid in enumerate(names):
            log = sid + "_00000_00100"
            frames = [dict(token=f"{sid}-f{i}", timestamp=i * 500_000,
                cams={cam: dict(data_path=f"{log}/{cam}/{i}.jpg") for cam in expand.original.CAMERAS}) for i in range(64)]
            (logs / (log + ".pkl")).write_bytes(pickle.dumps(frames))
            for i in (3, 23, 43, 63):
                rows.append(dict(token=frames[i]["token"], log_name=log, session_id=sid, timestamp_us=frames[i]["timestamp"], map_name="city"))
            archive = pool / f"openscene_sensor_trainval_camera_{shard}.tgz"
            with tarfile.open(archive, "w:gz") as tar:
                for f in frames:
                    for c in f["cams"].values():
                        payload = b"test image"
                        item = tarfile.TarInfo("openscene-v1.1/sensor_blobs/trainval/" + c["data_path"])
                        item.size = len(payload)
                        tar.addfile(item, io.BytesIO(payload))
            archive_sources[shard] = archive
            if shard < 3:
                shutil.copyfile(archive, archives / archive.name)
            mapping[f"openscene_sensor_trainval_{shard}"] = [log]
            listing.append(dict(path="openscene-v1.1/openscene_sensor_trainval_camera/" + archive.name,
                                size=archive.stat().st_size, lfs=dict(oid=expand.sha(archive))))
        raw_mapping = json.dumps(mapping).encode()
        for name, value in (("mapping.json", mapping), ("archives.json", listing)):
            expand.write(previous / "metadata" / name, value)
        # Use byte representation whose SHA is checked by the official parser.
        raw_mapping = (previous / "metadata/mapping.json").read_bytes()
        anchors = [dict(r, role="memory" if r["token"] == rows[0]["token"] else "train") for r in rows if r["session_id"] in names[:2]]
        holdout = dict(next(r for r in rows if r["session_id"] == names[2]), role="holdout")
        selection = dict(legacy_training_records=anchors[:1], records=anchors[1:] + [holdout],
                         excluded_sessions=[names[5]], fresh_holdout_sessions=[names[2]])
        expand.write(previous / "selection.json", selection)
        expand.write(previous / "image_requests.json", dict(archives=[]))
        with (inventory / "inventory.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        expand.write(parent / "examples_index.json", {r["token"]: dict(metadata=dict(token=r["token"], role="train", session_id=r["session_id"])) for r in anchors})
        expand.write(parent / "identity.json", dict(inputs={str(previous / "selection.json"): expand.sha(previous / "selection.json")},
                     protected_sessions=[names[2], names[5]]))
        expand.write(parent / "bundle_manifest.json", dict(identity_sha256=expand.sha(parent / "identity.json"),
                     index_sha256=expand.sha(parent / "examples_index.json")))
        options = dict(source=str(source), inventory=str(inventory), addition=str(base / "addition"), validation=str(base / "validation"))
        inputs = {str(p): expand.sha(p) for p in (source / "identity.json", inventory / "inventory.csv")}
        artifacts = {name: expand.sha(previous / name) for name in ("selection.json", "metadata/mapping.json", "metadata/archives.json")}
        expand.write(previous / "acquisition_plan.json", dict(options=options, inputs=inputs, artifacts=artifacts, additional_budget_bytes=0))
        args = argparse.Namespace(stage="plan", previous_data=previous, previous_bundle=parent, archive_dir=archives,
                                  out=base / "new", max_extra_gb=.001, min_scenes=2, max_scenes=6, gap_s=10.)
        return args, archive_sources, hashlib.sha256(raw_mapping).hexdigest()

    def test_plan_uses_all_parent_anchors_and_preserves_closed_sessions(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            args, archives, mapping_sha = self.setup_images(Path(tmp).resolve())
            with mock.patch.object(expand.original, "MAPPING_SHA256", mapping_sha), \
                    mock.patch.object(expand.original, "fetch_bytes", side_effect=AssertionError("no network")), redirect_stdout(io.StringIO()):
                result = expand.plan(args)
                self.assertEqual(result, expand.plan(args))
            report = result["summary"]
            self.assertEqual(report["legacy_train_scenes"], 8)
            self.assertEqual(report["added_train_scenes"], 8)
            self.assertEqual(report["new_training_sessions"], 2)
            self.assertEqual(report["existing_session_extra_scenes"], 0)
            self.assertLessEqual(report["additional_compressed_GB"], args.max_extra_gb)
            selection = expand.read(args.out / "selection.json")
            self.assertEqual({r["session_id"] for r in selection["records"]}, {"newa_veh-3", "newb_veh-4"})
            self.assertEqual(selection["fresh_holdout_sessions"], ["fresh_veh-2"])
            self.assertEqual({e["shard"] for e in expand.read(args.out / "image_requests.json")["archives"]}, {3, 4})
            self.assertEqual(len(list(args.archive_dir.glob("*.tgz"))), 3)
            args.max_scenes += 1
            with self.assertRaisesRegex(RuntimeError, "options changed"):
                expand.verify(args.out, args)

    def test_verified_fetch_and_unchanged_collector_init_are_compatible_and_resume(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            args, archives, mapping_sha = self.setup_images(Path(tmp).resolve())
            with mock.patch.object(expand.original, "MAPPING_SHA256", mapping_sha), redirect_stdout(io.StringIO()):
                expand.plan(args)
            _, requests = expand.verify(args.out, args)
            calls = []
            def download(entry, directory):
                calls.append(entry["shard"])
                target = directory / Path(entry["path"]).name
                shutil.copyfile(archives[entry["shard"]], target)
                self.assertEqual(expand.sha(target), entry["sha256"])
                return target
            with mock.patch.object(expand.original, "download", side_effect=download), redirect_stdout(io.StringIO()):
                self.assertEqual(expand.original.fetch(args.out, requests, args.archive_dir)["status"], "complete")
            self.assertEqual(set(calls), {3, 4})
            with mock.patch.object(expand.original, "download", side_effect=AssertionError("must resume")), redirect_stdout(io.StringIO()):
                self.assertEqual(expand.original.fetch(args.out, requests, args.archive_dir)["status"], "complete")
                root = Path(tmp).resolve() / "collection"
                _, identity = union.collection.initialize(args.out, root)
                _, records = union.collection.verify_collection(root)
            self.assertEqual(identity["legacy_scenes"], 8)
            self.assertEqual(len(records), 8)
            self.assertTrue(all(r["role"] == "train" for r in records))
            self.assertFalse({r["session_id"] for r in records} & {"fresh_veh-2", "seen_veh-5"})

    def test_corrupt_cached_archive_stops_before_additional_transfer(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            args, archives, mapping_sha = self.setup_images(Path(tmp).resolve())
            cached = args.archive_dir / archives[3].name
            shutil.copyfile(archives[3], cached)
            with mock.patch.object(expand.original, "MAPPING_SHA256", mapping_sha), redirect_stdout(io.StringIO()):
                expand.plan(args)
            cached.write_bytes(b"corrupt")
            _, requests = expand.verify(args.out, args)
            with mock.patch.object(expand.original, "download", side_effect=AssertionError("no transfer")), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "no additional downloads started"):
                    expand.original.fetch(args.out, requests, args.archive_dir)

    def test_unaffordable_or_over_budget_plan_does_not_publish_or_download(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            args, _, mapping_sha = self.setup_images(Path(tmp).resolve())
            args.max_extra_gb = 30.01
            with self.assertRaisesRegex(ValueError, "<= 30"):
                expand.plan(args)
            args.max_extra_gb = 1e-9
            with mock.patch.object(expand.original, "MAPPING_SHA256", mapping_sha), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "No additional"):
                    expand.plan(args)
            self.assertFalse((args.out / "acquisition_plan.json").exists())
            self.assertEqual(len(list(args.archive_dir.glob("*.tgz"))), 3)

    def test_joint_solver_does_not_double_count_overlapping_sessions(self):
        rows = []
        for shard, sid, offset in ((1, "a", 0), (2, "a", 20), (3, "b", 0)):
            rows += [dict(shard=shard, session_id=sid, token=str(shard) + str(t), timestamp_us=(t + offset) * 1_000_000) for t in (0, 10)]
        with redirect_stdout(io.StringIO()):
            chosen, _, _ = expand.original.acquire(rows, {i: dict(size=6) for i in (1, 2, 3)}, set(), set(), {}, 12, 2, 8, 99, 10.)
        records = expand.freeze(rows, chosen, {}, 2, 8, 10.)
        self.assertEqual(len({r["session_id"] for r in records}), 2)
        self.assertEqual(len(records), 4)

    def test_extend_bundle_preserves_parent_and_validates_new_labels(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            parent, new, acq, source, out = [root / n for n in ("parent", "new", "acq", "source", "union")]
            old_example = bundle.example(*fixture("old", "old-session"))
            sample, outcomes, vocab, _ = fixture("new", "new-session", "new")
            sample.update(token="new", session_id="new-session", role="train")
            config = {k: "same" for k in ("agent", "scorer", "simulator", "proposal_sampling", "traffic_agents", "traffic_agents_policy",
                      "train_test_split", "navsim_log_path", "original_sensor_path", "map_root", "checkpoint_sha256")}
            bundle.write(source / "identity.json", config)
            with redirect_stdout(io.StringIO()):
                bundle.build(parent, [old_example], vocab, {str(source / "identity.json"): bundle.sha(source / "identity.json")}, {"holdout"})
            parent_files = {str(p): bundle.sha(p) for p in parent.rglob("*") if p.is_file()}
            records = [dict(token="new", session_id="new-session", role="train", map_name="city")]
            selection = dict(legacy_training_records=[dict(token="old", session_id="old-session")], records=records,
                             excluded_sessions=["closed"], fresh_holdout_sessions=["holdout"])
            plan = dict(protocol="parent_bundle_anchored_expansion30_v1", options=dict(previous_bundle=str(parent), source=str(source)),
                        summary=dict(total_train_sessions=2, total_train_scenes=2))
            identity = dict(acquisition_root=str(acq), numerical_profile=union.collection.PROFILE)
            bundle.write(acq / "selection.json", selection)
            bundle.write(acq / "acquisition_plan.json", plan)
            for name in ("identity.json", "splits.json", "split_identity.json", "inventory.csv", "runtime.json", "export_index.json",
                         "native_index.json", "cache_index.json", "vocab_identity.json"):
                bundle.write(new / name, identity if name == "identity.json" else {})
            bundle.write(new / "source_config.json", config)
            bundle.save(new / "vocab.npz", vocab=vocab)
            for folder in ("features", "native"):
                bundle.save(new / folder / "new.npz", **{k: sample[k] for k in bundle.INPUT_KEYS})
            index = {"new": dict(sha256=bundle.sha(new / "features/new.npz"), native_sha256=bundle.sha(new / "native/new.npz"))}
            state = dict(token="new", session_id="new-session", role="train", top16=list(range(16)), candidate_ids=list(range(16)),
                         feature_sha256=index["new"]["sha256"], outcomes=outcomes)
            bundle.write(new / "top16_labels/new.json", state)
            support = dict(status="complete", holdout_processed=0, labelled_scenes=1, labels={"new.json": bundle.sha(new / "top16_labels/new.json")})
            bundle.write(new / "support_report.json", support)
            bundle.write(new / "label_identity.json", dict(collection_identity_sha256=bundle.sha(new / "identity.json"),
                export_index_sha256=bundle.sha(new / "export_index.json"), cache_index_sha256=bundle.sha(new / "cache_index.json"), vocab_sha256=bundle.sha(new / "vocab.npz")))
            with mock.patch.object(union.collection, "verify_collection", return_value=(identity, records)), \
                    mock.patch.object(union.collection.acquisition, "verify", return_value=(plan, {})), \
                    mock.patch.object(union.collection, "checked_features", return_value=index), \
                    mock.patch.object(union.collection, "checked_index", return_value={"new": {}}), \
                    mock.patch.object(union.collection, "load_sample", return_value=sample), redirect_stdout(io.StringIO()):
                examples, table, inputs, protected = union.load_sources(parent, new)
                bundle.build(out, examples, table, inputs, protected)
                loaded, _ = bundle.load_bundle(out)
                self.assertEqual({e["metadata"]["origin"] for e in loaded}, {"legacy", "expansion30"})
                old_loaded = next(e for e in loaded if e["metadata"]["token"] == "old")
                np.testing.assert_array_equal(old_loaded["label_raw"], old_example["label_raw"])
                self.assertEqual(parent_files, {str(p): bundle.sha(p) for p in parent.rglob("*") if p.is_file()})
                self.assertIn("holdout", protected)
                # Same immutable builder can resume the larger union.
                self.assertEqual(len(bundle.load_bundle(out)[0]), 2)
                state["candidate_ids"][0] = 255
                bundle.write(new / "top16_labels/new.json", state)
                support["labels"]["new.json"] = bundle.sha(new / "top16_labels/new.json")
                bundle.write(new / "support_report.json", support)
                with self.assertRaisesRegex(ValueError, "pool/provenance"):
                    union.load_sources(parent, new)


if __name__ == "__main__":
    unittest.main()
