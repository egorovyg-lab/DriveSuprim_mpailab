import contextlib
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "download"))
import prepare_drivemem_new_mf as prep
from test_drivemem_safety_data import write, sha


class NewMFTests(unittest.TestCase):
    def test_roles_are_deterministic_and_membership_is_exact(self):
        a = prep.allocation(prep.EXPECTED, 20261027)
        self.assertEqual(len(a["memory"]), 2)
        self.assertEqual(len(a["fit"]), 2)
        self.assertFalse(set(a["memory"]) & set(a["fit"]))
        self.assertEqual(a, prep.allocation(sorted(prep.EXPECTED)[::-1], 20261027))
        with self.assertRaises(ValueError):
            prep.allocation(set(prep.EXPECTED) | {"unexpected"}, 1)

    def test_metadata_only_preparation_and_budget_refusal(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            source, inv, val, out = [root / n for n in ("source", "inv", "val", "out")]
            logs = root / "logs"
            logs.mkdir()
            rows = []
            names = []
            for i, sid in enumerate(sorted(prep.EXPECTED)):
                log = sid + "_00000_00999"
                names.append(log)
                (logs / (log + ".pkl")).write_bytes(b"metadata")
                rows.extend(dict(token=f"{i}-{j}", session_id=sid, log_name=log, timestamp_us=j * 20000000) for j in range(30))
            template = dict(settings=dict(seed=20261027), sources={}, navsim_log_path=str(logs))
            for r in (source, inv, val):
                write(r / "identity.json", template)
                write(r / "splits.json", dict(records=[], navtest_sessions_excluded=[]))
                with (r / "inventory.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                write(r / "split_identity.json", dict(splits=sha(r / "splits.json"), inventory=sha(r / "inventory.csv")))
            raw = json.dumps({"openscene_sensor_trainval_174": names}).encode()
            entry = dict(path="openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_174.tgz", size=8427621512, sha256="a" * 64)
            def fetch(url):
                return json.dumps([dict(path=entry["path"])]).encode() if "/api/" in url else raw
            with patch.object(prep, "fetch_bytes", side_effect=fetch) as network, patch.object(prep, "MAPPING_SHA256", hashlib.sha256(raw).hexdigest()), patch.object(prep, "archive_entry", return_value=entry), patch.object(prep, "collect_requests", return_value=dict(images=[], logs=[])), contextlib.redirect_stdout(io.StringIO()):
                prep.prepare(source, inv, val, out)
                state = prep.read(out / "expansion_plan.json")
                self.assertEqual(state["screening_scenes"], 64)
                self.assertEqual(state["label_scenes"], 32)
                self.assertEqual(state["raw_filtered_calls"], 1024)
                self.assertEqual(prep.read(out / "identity.json")["settings"]["sessions"]["fit"], 2)
                self.assertEqual(len(prep.read(out / "splits.json")["records"]), 64)
                before = network.call_count
                prep.prepare(source, inv, val, out)
                self.assertEqual(network.call_count, before)
                entry["size"] = 8500000001
                with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
                    prep.prepare(source, inv, val, root / "too_big")
                self.assertFalse((root / "too_big").exists())


if __name__ == "__main__":
    unittest.main()
