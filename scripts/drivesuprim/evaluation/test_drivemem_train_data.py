import hashlib
import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("train_data", Path("download/prepare_drivemem_navtrain.py"))
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def archive(path, entries):
    with tarfile.open(path, "w:gz") as tar:
        for name, contents in entries:
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            tar.addfile(info, io.BytesIO(contents))


class MetadataTests(unittest.TestCase):
    def test_allowlist_flat_layout_resume_and_existing_test_unchanged(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent, Path.cwd().resolve())
            old = root / "navsim_logs/test/old.pkl"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"existing-test")
            source = root / "metadata.tgz"
            archive(source, [("openscene-v1.1/meta_datas/trainval/a.pkl", b"a"),
                             ("openscene-v1.1/meta_datas/trainval/b.pkl", b"b")])
            target = root / "navsim_logs/trainval"
            result = data.extract_logs(source, target, {"a"})
            self.assertEqual(set(result), {"a"})
            self.assertEqual((target / "a.pkl").read_bytes(), b"a")
            self.assertFalse((target / "b.pkl").exists())
            self.assertEqual(old.read_bytes(), b"existing-test")
            self.assertEqual(result, data.extract_logs(source, target, {"a"}))

    def test_different_existing_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent, Path.cwd().resolve())
            source, target = root / "metadata.tgz", root / "output"
            target.mkdir()
            (target / "a.pkl").write_bytes(b"different")
            archive(source, [("meta_datas/a.pkl", b"a")])
            with self.assertRaisesRegex(RuntimeError, "not overwritten"):
                data.extract_logs(source, target, {"a"})
            self.assertEqual((target / "a.pkl").read_bytes(), b"different")

    def test_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent, Path.cwd().resolve())
            source = root / "metadata.tgz"
            archive(source, [("../meta_datas/a.pkl", b"bad")])
            with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                data.extract_logs(source, root / "output", {"a"})

    def test_completed_partial_download_is_verified_and_promoted_without_network(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent, Path.cwd().resolve())
            contents = b"completed download fixture"
            partial = root / "openscene_metadata_trainval.tgz.part"
            partial.write_bytes(contents)
            with patch.object(data, "EXPECTED_SIZE", len(contents)), \
                    patch.object(data, "EXPECTED_SHA256", hashlib.sha256(contents).hexdigest()), \
                    patch.object(data.shutil, "which", return_value="curl"), \
                    patch.object(data.subprocess, "run") as request:
                result = data.download_archive(root)
            request.assert_not_called()
            self.assertEqual(result.read_bytes(), contents)
            self.assertFalse(partial.exists())


if __name__ == "__main__":
    unittest.main()
