"""run_gpu_tests.py: a filtered run still builds the vertex shader a selected test draws with, and relative --build
/--work paths keep working for test programs that run in another working directory (vk_texture_test)."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gpu_tests


class DrawShaders(unittest.TestCase):
    def test_texture_sample_brings_vs_ids(self):
        self.assertEqual(run_gpu_tests.with_draw_shaders(["ps_texture_sample.asm"]),
                         ["ps_texture_sample.asm", "vs_ids.asm"])

    def test_flat_still_brings_vs_ids(self):
        self.assertEqual(run_gpu_tests.with_draw_shaders(["ps_flat.asm"]), ["ps_flat.asm", "vs_ids.asm"])

    def test_vs_ids_is_not_added_twice(self):
        files = ["ps_flat.asm", "ps_texture_sample.asm", "vs_ids.asm"]
        self.assertEqual(run_gpu_tests.with_draw_shaders(files), files)

    def test_other_shaders_add_nothing(self):
        self.assertEqual(run_gpu_tests.with_draw_shaders(["ps_ops.asm"]), ["ps_ops.asm"])

    def test_every_shader_named_exists(self):
        shaders = Path(run_gpu_tests.SHADER_DIR)
        for name, extra in run_gpu_tests.DRAWN_WITH.items():
            self.assertTrue((shaders / f"{name}.asm").is_file(), name)
            self.assertTrue((shaders / extra).is_file(), extra)


class RelativePaths(unittest.TestCase):
    def test_build_and_work_become_absolute(self):
        ext = ".exe" if os.name == "nt" else ""
        seen = {}

        def suite(a, work):
            seen["build"], seen["work"] = a.build, work
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp) / "build"
            build.mkdir()
            for tool in ("vk_compute_test", "vk_store_test", "vk_harness"):
                (build / (tool + ext)).touch()
            old = os.getcwd()
            os.chdir(tmp)
            try:
                with (patch.object(sys, "argv", ["run_gpu_tests.py", "--build", "build", "--work", "work"]),
                      patch.object(run_gpu_tests, "run_suite", side_effect=suite)):
                    self.assertEqual(run_gpu_tests.main(), 0)
            finally:
                os.chdir(old)
            self.assertEqual(Path(seen["build"]), build.resolve())
            self.assertEqual(Path(seen["work"]), (Path(tmp) / "work").resolve())


if __name__ == "__main__":
    unittest.main()
