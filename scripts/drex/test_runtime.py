from pathlib import Path
import os
import shutil
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest

import runtime


class SlurmRoutingTests(unittest.TestCase):
    def setUp(self):
        windows_bash = Path(r'C:\Program Files\Git\bin\bash.exe')
        self.bash = str(windows_bash) if windows_bash.exists() else shutil.which("bash")
        if not self.bash:
            self.skipTest("Bash is unavailable")
        self.tmp = tempfile.TemporaryDirectory(dir=str(Path.cwd()))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.envroot = self.root / "runtime"
        self.envroot.mkdir()
        self.log = self.root / "srun.log"
        self.calls = self.root / "python.log"
        self.script = Path(__file__).with_name("run_runtime.sh").resolve()
        self.env = dict(os.environ, DREX_RUNTIME_ROOT=self.posix(self.envroot),
                        DREX_TEST_LOG=self.posix(self.log), DREX_TEST_CALLS=self.posix(self.calls))
        for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURMD_NODENAME"):
            self.env.pop(name, None)
        self.stub("srun", 'printf "%s\\n" "$@" > "$DREX_TEST_LOG"\n')

    def posix(self, path):
        if os.name == "nt":
            return subprocess.check_output([self.bash, "-c", 'cygpath -u "$1"', "path", str(path)], text=True).strip()
        return str(path)

    def stub(self, name, body):
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8", newline="\n")
        path.chmod(0o755)
        return path

    def run_stage(self, *args):
        return subprocess.run([self.bash, "-c", 'export PATH="$1:$PATH"; shift; exec bash "$@"',
                               "runner", self.posix(self.bin), self.posix(self.script), *args],
                              env=self.env, text=True, capture_output=True, timeout=15)

    def test_gpu_and_prepare_submit_before_inspecting_login_python_or_missing_venv(self):
        for stage, limit in (("gpu-check", "00:05:00"), ("prepare", "00:30:00")):
            result = self.run_stage(stage)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = self.log.read_text().splitlines()
            self.assertIn("--gres=gpu:1", args)
            self.assertIn("--time=" + limit, args)
            self.assertIn("--login", args)
            self.assertFalse(any(arg.startswith("--immediate") for arg in args))
            self.assertEqual(args[-2:], ["--inside", stage])
            self.assertFalse((self.envroot / "venv").exists())

    def test_metadata_and_download_use_recorded_compute_node_without_gpu(self):
        (self.envroot / "setup_node.txt").write_text("cn29\n")
        for stage in ("model-plan", "download"):
            result = self.run_stage(stage)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = self.log.read_text().splitlines()
            self.assertIn("--nodelist=cn29", args)
            self.assertIn("--gres=none", args)
            self.assertNotIn("--gres=gpu:1", args)

    def test_internal_mode_requires_a_real_job_step(self):
        self.env["SLURM_JOB_ID"] = "100"  # salloc alone does not imply a compute-node shell.
        result = self.run_stage("--inside", "gpu-check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("actual Slurm job step", result.stderr)
        self.assertFalse(self.log.exists())

    def fake_setup(self):
        self.env.update(SLURM_JOB_ID="100", SLURM_STEP_ID="0", SLURMD_NODENAME="cn29")
        self.stub("flock", "exit 0\n")
        bootstrap = self.stub("bootstrap", "exit 0\n")
        self.env["DREX_BOOTSTRAP_PYTHON"] = self.posix(bootstrap)
        self.stub("git", 'if [[ "$*" == *"rev-parse HEAD"* ]]; then echo b932571901f531b2378b356fcab7d176048c2bcb; fi\n')
        (self.envroot / "upstream").mkdir()
        (self.envroot / "requirements.lock.txt").write_text("")
        fake_python = self.envroot / "venv/bin/python"
        fake_python.parent.mkdir(parents=True)
        fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$DREX_TEST_CALLS"\n'
                               'if [[ ${DREX_TEST_FAIL_CHECK:-} == 1 && "$*" == *"runtime.py check "* ]]; then exit 9; fi\n',
                               encoding="utf-8", newline="\n")
        fake_python.chmod(0o755)

    def test_prepare_in_allocation_runs_three_checks_without_nested_srun(self):
        self.fake_setup()
        result = self.run_stage("--inside", "prepare")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.log.exists())
        stages = [line.split("runtime.py ", 1)[1].split()[0] for line in self.calls.read_text().splitlines()
                  if "runtime.py " in line]
        self.assertEqual(stages, ["check", "gpu-check", "model-plan"])
        self.assertEqual((self.envroot / "setup_node.txt").read_text().strip(), "cn29")

    def test_prepare_stops_after_setup_failure(self):
        self.fake_setup()
        self.env["DREX_TEST_FAIL_CHECK"] = "1"
        result = self.run_stage("--inside", "prepare")
        self.assertEqual(result.returncode, 9, result.stderr)
        self.assertNotIn("runtime.py gpu-check", self.calls.read_text())
        self.assertNotIn("runtime.py model-plan", self.calls.read_text())
        self.assertFalse((self.envroot / "setup_node.txt").exists())


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
