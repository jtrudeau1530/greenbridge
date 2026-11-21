"""TEST-18: a device without geometry or mesh shaders (MoltenVK, older GPUs) makes vk_harness exit 77; the
runners report that as SKIP, not as a failure."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
sys.path.insert(0, str(ROOT / "scripts"))
import check
import run_gpu_tests


class MeshCheckOnDeviceWithoutMeshShaders(unittest.TestCase):
    """check.py's ms_w64x_ballot with a fake vk_harness."""

    def run_mesh(self, code, stderr):
        def harness(cmd, **_kw):
            return subprocess.CompletedProcess(cmd, code, "", stderr)

        with tempfile.TemporaryDirectory() as tmp:
            for name in ("ms_w64x_ballot", "fs_passthru"):
                (Path(tmp) / f"{name}.spv").touch()
            with patch.object(check.subprocess, "run", side_effect=harness):
                return check.mesh_w64x_test(tmp, tmp, "ms_w64x_ballot")

    def test_missing_mesh_extension_skips(self):
        problem = self.run_mesh(77, "device: fake (subgroup 32..32)\nunsupported: no VK_EXT_mesh_shader\n")
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("no VK_EXT_mesh_shader", problem)

    def test_other_harness_errors_still_fail(self):
        problem = self.run_mesh(1, "FAILED vkCreateDevice -> -8\n")
        self.assertNotIsInstance(problem, check.Skip)
        self.assertTrue(problem)


class RunGpuTestsUnsupportedDetail(unittest.TestCase):
    def test_quotes_the_harness_reason(self):
        msg = "device: MoltenVK (subgroup 32..32)\nunsupported: no geometryShader"
        self.assertEqual(run_gpu_tests.unsupported_skip(msg), "SKIP (unsupported: no geometryShader)")

    def test_without_a_reason(self):
        self.assertEqual(run_gpu_tests.unsupported_skip(""), "SKIP (unsupported device)")

    def test_harness_and_runners_agree_on_the_exit_code(self):
        self.assertEqual(run_gpu_tests.UNSUPPORTED, check.UNSUPPORTED)
        harness = (ROOT / "runtime" / "test" / "vk_harness.cpp").read_text()
        for reason in ("unsupported: no VK_EXT_mesh_shader", "unsupported: no geometryShader"):
            self.assertIn(reason, harness)


if __name__ == "__main__":
    unittest.main()
