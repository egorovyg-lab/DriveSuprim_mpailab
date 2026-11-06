import ast
import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import Mock, patch

import numpy as np

from navsim.agents import drivemem_determinism as profile
from navsim.agents import drivemem_decoder_utility as model
from navsim.planning.script import run_drivemem_decoder_replay_export as original
from navsim.planning.script import run_drivemem_decoder_deterministic_export as exporter

ENV = dict(PYTHONHASHSEED="20261027", CUBLAS_WORKSPACE_CONFIG=":4096:8", CUDA_VISIBLE_DEVICES="0")


def fake_torch():
    state = N(mode=False, warn=True, precision="high", flash=True, efficient=True, math=False)
    cuda_backend = N(matmul=N(allow_tf32=True),
        enable_flash_sdp=lambda b: setattr(state, "flash", b),
        enable_mem_efficient_sdp=lambda b: setattr(state, "efficient", b),
        enable_math_sdp=lambda b: setattr(state, "math", b),
        flash_sdp_enabled=lambda: state.flash,
        mem_efficient_sdp_enabled=lambda: state.efficient, math_sdp_enabled=lambda: state.math)
    def deterministic(mode, warn_only):
        state.mode, state.warn = mode, warn_only
    return N(__version__="2.0.1+cu118", version=N(cuda="11.8"),
        cuda=N(is_initialized=lambda: False, device_count=lambda: 1,
            get_device_properties=lambda i: N(name="GTX 1080", major=6, minor=1, total_memory=8 * 1024**3)),
        backends=N(cudnn=N(benchmark=True, deterministic=False, allow_tf32=True, version=lambda: 8700), cuda=cuda_backend),
        use_deterministic_algorithms=Mock(side_effect=deterministic),
        are_deterministic_algorithms_enabled=lambda: state.mode,
        is_deterministic_algorithms_warn_only_enabled=lambda: state.warn,
        set_float32_matmul_precision=lambda p: setattr(state, "precision", p),
        get_float32_matmul_precision=lambda: state.precision, manual_seed=Mock())


class DeterministicDecoderTests(unittest.TestCase):
    def test_profile_is_strict_configured_before_cuda_and_observed_in_runtime(self):
        torch = fake_torch()
        with patch.dict(os.environ, ENV), patch.object(profile.random, "seed") as pyseed, patch.object(profile.np.random, "seed") as npseed:
            profile.configure(torch)
            observed = profile.runtime(torch)
            self.assertEqual(observed["settings"], profile.PROFILE)
            self.assertEqual(observed["capability"], [6, 1])
            torch.use_deterministic_algorithms.assert_called_once_with(True, warn_only=False)
            torch.manual_seed.assert_called_once_with(profile.SEED)
            pyseed.assert_called_once_with(profile.SEED)
            npseed.assert_called_once_with(profile.SEED)
        torch.cuda.is_initialized = lambda: True
        with patch.dict(os.environ, ENV), self.assertRaisesRegex(RuntimeError, "before CUDA"):
            profile.configure(torch)

    def test_conflicting_environment_and_warn_only_are_rejected(self):
        for key, value in (("PYTHONHASHSEED", "1"), ("CUBLAS_WORKSPACE_CONFIG", ":16:8"), ("CUDA_VISIBLE_DEVICES", "0,1")):
            with patch.dict(os.environ, dict(ENV, **{key: value})), self.assertRaises(RuntimeError):
                profile.prepare_environment()
        torch = fake_torch()
        with patch.dict(os.environ, ENV), self.assertRaisesRegex(RuntimeError, "settings"):
            profile.runtime(torch)

    def test_bitwise_repeat_is_stricter_than_old_tolerance_without_relaxing_frozen_parity(self):
        vocab = np.zeros((16, 40, 3))
        sample = dict(candidate_ids=np.arange(16), selected_local=np.array(0),
            selection_scores=np.arange(16, 0, -1, dtype=np.float32), head_imi=np.ones(16, dtype=np.float32),
            final_traj=vocab[0])
        profile.repeat_parity(sample, copy.deepcopy(sample))
        changed = copy.deepcopy(sample)
        changed["head_imi"][4] = np.nextafter(np.float32(1), np.float32(2))
        exporter.parity(sample, changed, vocab)  # Old threshold would accept this tiny difference.
        with self.assertRaisesRegex(RuntimeError, "bitwise.*head_imi"):
            profile.repeat_parity(sample, changed)
        changed["head_imi"][4] += .001
        detail = profile.failure_details(sample, changed)
        self.assertEqual(detail["fields"]["head_imi"]["failed_elements"], 1)
        with self.assertRaisesRegex(RuntimeError, "head_imi"):
            exporter.parity(sample, changed, vocab)
        def functions(module):
            return {n.name: ast.dump(n) for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
        for name in ("parity", "wanted_records", "status"):
            self.assertEqual(functions(original)[name], functions(exporter)[name])

    def test_both_known_failed_scenes_are_in_first_pilot_without_dropping_other_records(self):
        tokens = ["first", profile.PILOT_TOKENS[1], "second", profile.PILOT_TOKENS[0], "last"]
        records = [(Path("source"), dict(token=t)) for t in tokens]
        ordered = profile.pilot_order(records)
        self.assertEqual(tuple(r["token"] for _, r in ordered[:2]), profile.PILOT_TOKENS)
        self.assertEqual({r["token"] for _, r in ordered}, set(tokens))
        with self.assertRaises(ValueError):
            profile.pilot_order(records[:2])

    def test_runtime_binding_rejects_mixed_old_scenes_or_metadata_changes(self):
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp:
            out = Path(tmp)
            profile.validate_runtime(out, {}, exporter.read, exporter.sha)
            with self.assertRaisesRegex(RuntimeError, "no runtime"):
                profile.validate_runtime(out, {"s": {}}, exporter.read, exporter.sha)
            exporter.write(out / "runtime.json", dict(settings=profile.PROFILE, gpu_name="GTX 1080"))
            index = {"s": dict(runtime_sha256=exporter.sha(out / "runtime.json"), repeat_bitwise_equal=True)}
            profile.validate_runtime(out, index, exporter.read, exporter.sha)
            with self.assertRaisesRegex(RuntimeError, "binding"):
                profile.validate_runtime(out, {"old": dict(sha256="old")}, exporter.read, exporter.sha)
            exporter.write(out / "runtime.json", dict(settings=profile.PROFILE, gpu_name="different"))
            with self.assertRaisesRegex(RuntimeError, "binding"):
                profile.validate_runtime(out, index, exporter.read, exporter.sha)

    def test_loader_verifies_new_exporter_profile_and_helper_hashes(self):
        name = "navsim/planning/script/run_drivemem_decoder_deterministic_export.py"
        identity = dict(exporter_path=name, exporter_sha256=exporter.sha(model.REPO / name),
            pool_helper_sha256=exporter.sha(model.REPO / "navsim/agents/drivemem_frozen_pool.py"),
            numerical_profile=copy.deepcopy(profile.PROFILE),
            numerical_helper_sha256=exporter.sha(model.REPO / "navsim/agents/drivemem_determinism.py"))
        self.assertEqual(model.verify_decoder_protocol(identity), name)
        identity["numerical_profile"]["warn_only"] = True
        with self.assertRaisesRegex(RuntimeError, "numerical protocol"):
            model.verify_decoder_protocol(identity)


if __name__ == "__main__":
    unittest.main()
