import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("download").resolve()))
from plan_drivemem_validation import inventory_candidates


class ValidationPlanTests(unittest.TestCase):
    def rows(self, sid, segment, times):
        return [dict(token=f"{sid}-{segment}-{i}", session_id=sid, log_name=f"{sid}_{segment}",
                     timestamp_us=int(t * 1e6)) for i, t in enumerate(times)]

    def test_locked_sessions_excluded_and_spacing_checked_within_log(self):
        locked, fresh, too_short = [f"2021.01.01.12.00.0{i}_veh-1" for i in range(3)]
        rows = self.rows(locked, "00000_00100", [0, 10, 20, 30])
        rows += self.rows(fresh, "00000_00100", [0, 10, 20, 30])
        rows += self.rows(too_short, "00000_00100", [0, 1, 2, 3])
        mapping = {"openscene_sensor_trainval_2": sorted({r["log_name"] for r in rows})}
        found = inventory_candidates(rows, mapping, {locked}, 4, 10, 1)
        self.assertEqual(set(found), {2})
        self.assertEqual(set(found[2]), {fresh})

    def test_segments_in_different_archives_do_not_create_false_capacity(self):
        sid = "2021.01.01.12.00.00_veh-1"
        rows = self.rows(sid, "00000_00020", [0, 10]) + self.rows(sid, "00020_00040", [20, 30])
        logs = sorted({r["log_name"] for r in rows})
        mapping = {f"openscene_sensor_trainval_{i}": [log] for i, log in enumerate(logs)}
        self.assertEqual(inventory_candidates(rows, mapping, set(), 4, 10, 1), {})
        mapping["openscene_sensor_trainval_3"] = [logs[0]]
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            inventory_candidates(rows, mapping, set(), 4, 10, 1)


if __name__ == "__main__":
    unittest.main()
