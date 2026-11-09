from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import selector_scale as scale


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def row(sid, token, seconds, shard=1):
    return dict(session_id=sid, token=token, timestamp_us=int(seconds * 1e6),
                log_name=sid + "_00000_01000", shard=shard)


class ScaleTests(unittest.TestCase):
    def test_capacity_counts_sessions_once_respects_gap_and_cap(self):
        rows = [row("old", "a", 0), row("old", "duplicate", 0), row("old", "near", 5),
                row("old", "b", 10), row("new", "n", 0, 2)]
        self.assertEqual(scale.count_capacity(rows, {1}, {"old": 7}, 8, 10.), {"old": 1})
        self.assertEqual(scale.count_capacity(rows, {1, 2}, {"old": 7}, 9, 10.), {"old": 2, "new": 1})
        section = scale.capacity_section({"old": 2, "new": 16, "short": 15}, {"old": 7}, 16)
        self.assertEqual(section["additional_scenes"], 18)
        self.assertEqual(section["new_eligible_sessions"], 1)
        self.assertEqual(section["total_train_sessions_if_all_eligible_sessions_used"], 2)

    def setup_metadata(self, base):
        data, bundle, archives, source, inventory, logs = [base / n for n in ("data", "bundle", "archives", "source", "inventory", "logs")]
        archives.mkdir()
        logs.mkdir()
        old, fresh, closed, unseen = "old_veh-1", "new_veh-2", "closed_veh-3", "unseen_veh-4"
        anchors = [row(old, "old", 0), row(fresh, "already-new", 0)]
        selection = dict(legacy_training_records=[anchors[0]], records=[dict(anchors[1], role="train"),
                         dict(row(closed, "holdout", 0), role="holdout")],
                         excluded_sessions=[], fresh_holdout_sessions=[closed])
        write(data / "selection.json", selection)
        write(data / "image_requests.json", dict(archives=[]))
        for name in ("mapping.json", "archives.json"):
            write(data / "metadata" / name, {})
        write(bundle / "examples_index.json", {r["token"]: dict(metadata=dict(role="train", session_id=r["session_id"])) for r in anchors})
        write(bundle / "identity.json", dict(inputs={str(data / "selection.json"): scale.sha(data / "selection.json")}, protected_sessions=[closed]))
        write(bundle / "bundle_manifest.json", dict(identity_sha256=scale.sha(bundle / "identity.json"),
                                                    index_sha256=scale.sha(bundle / "examples_index.json")))
        write(source / "identity.json", dict(navsim_log_path=str(logs)))
        rows = [row(old, "near-old", 5), row(old, "old-far", 10), row(fresh, "near-new", 5),
                row(fresh, "new-far", 10), row(closed, "closed-far", 20)]
        rows += [row(unseen, "u" + str(i), i * 10, 2) for i in range(16)]
        path = inventory / "inventory.csv"
        path.parent.mkdir()
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = scale.csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for sid in (old, fresh, closed, unseen):
            (logs / (sid + "_00000_01000.pkl")).write_bytes(b"metadata not opened")
        (archives / "one.tgz").write_bytes(b"123")
        mapping = {sid + "_00000_01000": 2 if sid == unseen else 1 for sid in (old, fresh, closed, unseen)}
        entries = {1: dict(path="one.tgz", size=3), 2: dict(path="two.tgz", size=100)}
        inputs = {str(path): scale.sha(path), str(source / "identity.json"): scale.sha(source / "identity.json")}
        write(data / "acquisition_plan.json", dict(options=dict(source=str(source), inventory=str(inventory)),
              inputs=inputs, artifacts={"selection.json": scale.sha(data / "selection.json")}, additional_budget_bytes=0))
        return data, bundle, archives, mapping, entries

    def test_capacity_anchors_include_entire_bundle_and_holdout_stays_closed(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            data, bundle, archives, mapping, entries = self.setup_metadata(Path(tmp).resolve())
            before = {str(p): scale.sha(p) for p in Path(tmp).rglob("*") if p.is_file()}
            with mock.patch.object(scale.acquisition, "metadata", return_value=(mapping, entries)), \
                    mock.patch.object(scale.acquisition, "fetch_bytes", side_effect=AssertionError("no network")), \
                    mock.patch.object(scale.acquisition, "load_frames", side_effect=AssertionError("no features/images")):
                result = scale.capacity(data, bundle, archives)
            self.assertEqual(result["current_scenes"], 2)
            self.assertEqual(result["cached_capacity"]["additional_scenes"], 2)
            self.assertEqual(result["cached_capacity"]["new_eligible_sessions"], 0)
            self.assertEqual(result["additional_archive_options"][0]["additional_eligible_sessions"], 1)
            self.assertEqual(result["additional_archive_options"][0]["additional_eligible_scenes"], 16)
            self.assertEqual(result["additional_download_GB"], 0)
            self.assertEqual(result["fresh_holdout_sessions_preserved"], 1)
            self.assertEqual(before, {str(p): scale.sha(p) for p in Path(tmp).rglob("*") if p.is_file()})

    def test_capacity_rejects_missing_cache_and_changed_bundle_metadata(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            data, bundle, archives, _, _ = self.setup_metadata(Path(tmp).resolve())
            (data / "metadata/archives.json").unlink()
            with mock.patch.object(scale.acquisition, "fetch_bytes", side_effect=AssertionError("no network")):
                with self.assertRaisesRegex(FileNotFoundError, "never downloads"):
                    scale.capacity(data, bundle, archives)
            write(bundle / "examples_index.json", {})
            with self.assertRaisesRegex(RuntimeError, "bundle metadata changed"):
                scale.capacity(data, bundle, archives)

    def setup_runs(self, base):
        reference, sweep = base / "ref", base / "sweep"
        for epoch, root in zip(scale.DURATIONS, (reference, sweep / "epochs360", sweep / "epochs1000")):
            config = dict(epochs=epoch, seed=12, learning_rate=.001)
            write(root / "identity.json", dict(config=config, inputs={"same": "hash"}, sources={"model": "hash"}))
            (root / "model.npz").write_bytes(b"not opened, just hashed")
            cv = {family + "_" + mode: dict(gain_macro_pp=.1, harm=0, quality_degraded=0, interventions=5)
                  for family in scale.FAMILIES for mode in ("raw", "gated")}
            write(root / "fit.json", dict(status="complete", config=config, fresh_holdout_processed=0,
                  artifacts={"model.npz": scale.sha(root / "model.npz")}, cv_methods=cv,
                  selected={family: dict(threshold=.75, uniform_gain_macro_pp=.1) for family in scale.FAMILIES},
                  models={family: dict(history=[dict(epoch=epoch, loss=.01)]) for family in scale.FAMILIES}))
        return reference, sweep

    def test_duration_comparison_uses_only_training_reports_and_artifact_hashes(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            reference, sweep = self.setup_runs(Path(tmp))
            result = scale.compare(reference, sweep)
            self.assertEqual(len(result["rows"]), 6)
            self.assertEqual(result["historical_outcomes_read"], 0)
            write(sweep / "epochs1000/identity.json", dict(config=dict(epochs=1000, seed=13, learning_rate=.001)))
            with self.assertRaisesRegex(RuntimeError, "Complete train-only fit"):
                scale.compare(reference, sweep)

    def test_duration_comparison_rejects_different_data_and_tampered_weights(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            reference, sweep = self.setup_runs(Path(tmp))
            path = sweep / "epochs1000/identity.json"
            state = scale.read(path)
            state["inputs"] = {"other": "data"}
            write(path, state)
            with self.assertRaisesRegex(ValueError, "only in epoch"):
                scale.compare(reference, sweep)
            (reference / "model.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                scale.compare(reference, sweep)


if __name__ == "__main__":
    unittest.main()
