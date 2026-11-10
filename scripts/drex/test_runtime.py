from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import runtime


class RuntimeTests(unittest.TestCase):
    def test_manifest_pins_revision_and_excludes_code_and_assets(self):
        names = list(runtime.REQUIRED) + ["model-00001-of-00001.safetensors", "README.md", "code/run.py", "vision.png"]
        info = NS(sha="a" * 40, siblings=[NS(rfilename=n, size=10) for n in names])
        plan = runtime.manifest(info)
        self.assertEqual(plan["revision"], "a" * 40)
        self.assertFalse({"README.md", "vision.png", "code/run.py"} & {r["name"] for r in plan["files"]})
        self.assertEqual(plan["total_bytes"], len(plan["files"]) * 10)
        info.sha = "main"
        with self.assertRaisesRegex(ValueError, "revision"):
            runtime.manifest(info)

    def test_missing_files_and_unknown_sizes_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            runtime.manifest(NS(sha="a" * 40, siblings=[]))
        with self.assertRaisesRegex(ValueError, "size"):
            runtime.manifest(NS(sha="a" * 40, siblings=[NS(rfilename="config.json", size=None)]))

    def test_environment_lock_detects_changes(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as directory:
            p = Path(directory) / "lock"
            p.write_text("a==1\nb==2\n", encoding="utf-8")
            runtime.check_lock(p, "a==1\nb==2\n")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                runtime.check_lock(p, "a==2\nb==2\n")

    def test_resume_counts_only_missing_or_wrong_size_files(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as directory:
            p = Path(directory)
            rows = [dict(name="one", size=3), dict(name="two", size=4)]
            self.assertEqual(runtime.remaining_bytes(p, rows), 7)
            (p / "one").write_bytes(b"abc")
            self.assertEqual(runtime.remaining_bytes(p, rows), 4)
            (p / "one").write_bytes(b"a")
            self.assertEqual(runtime.remaining_bytes(p, rows), 7)


if __name__ == "__main__":
    unittest.main()
