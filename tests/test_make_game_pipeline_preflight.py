"""make_game.py: a missing vk_pipeline_check is reported in preflight, before a long shader build.

    python -I tests/test_make_game_pipeline_preflight.py

Standard library only. Every path is synthetic and every tool is mocked: no game files, GPU, Vulkan SDK or
build output is needed.
"""
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import make_game  # noqa: E402


class PipelinePreflightTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.work = os.path.join(self.root, "work")
        os.makedirs(self.work)
        # Satisfy every other prerequisite so only the checker under test can make preflight fail.
        for patcher in (mock.patch("sys.version_info", (3, 12, 0)),
                        mock.patch("make_game.tool_available", return_value=True),
                        mock.patch("shutil.which", return_value="/bin/tool"),
                        mock.patch("shutil.disk_usage", return_value=mock.Mock(free=500 * 2**30))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def assert_preflight_fails(self, stages, match):
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            with self.assertRaises(SystemExit) as ctx:
                make_game.preflight(stages, self.work)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn(match, err.getvalue())
        return err.getvalue()

    def test_pipelines_without_checker_fails_in_preflight(self):
        with mock.patch.object(make_game, "pipeline_check_exe", return_value=None):
            err = self.assert_preflight_fails(["shaders", "pipelines"], "vk_pipeline_check")
        self.assertIn("Install these first, then run the build again:", err)

    def test_main_does_not_start_shaders_when_checker_is_missing(self):
        game = os.path.join(self.root, "game")
        out = os.path.join(self.root, "out")
        os.makedirs(game)
        argv = ["make_game.py", "--game", game, "--work", self.work, "--out", out,
                "--stages", "shaders,pipelines"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(make_game, "below_normal"), \
                mock.patch.object(make_game, "check", return_value={"expected_exe": "game.exe"}), \
                mock.patch.object(make_game, "pipeline_check_exe", return_value=None), \
                mock.patch.object(make_game, "shaders") as shaders, \
                mock.patch.object(make_game, "pipelines") as pipelines, \
                mock.patch.object(make_game, "log"), \
                mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                make_game.main()
        self.assertEqual(ctx.exception.code, 1)
        shaders.assert_not_called()
        pipelines.assert_not_called()

    def test_checker_already_built_passes_preflight(self):
        with mock.patch.object(make_game, "pipeline_check_exe", return_value="/bin/vk_pipeline_check"):
            make_game.preflight(["pipelines"], self.work)

    def test_runtime_stage_builds_the_checker_so_it_is_not_required(self):
        # On Windows runtime produces vk_pipeline_check.exe, so a combined build must not fail preflight.
        with mock.patch("os.name", "nt"), \
                mock.patch.object(make_game, "visual_cpp_tools_ok", return_value=True), \
                mock.patch.object(make_game, "pipeline_check_exe", return_value=None):
            make_game.preflight(["runtime", "pipelines"], self.work)

    def test_missing_checker_is_listed_with_other_missing_prerequisites(self):
        with mock.patch("make_game.tool_available", return_value=False), \
                mock.patch.object(make_game, "pipeline_check_exe", return_value=None):
            err = self.assert_preflight_fails(["shaders", "pipelines"], "vk_pipeline_check")
        self.assertIn("llvm-mc (LLVM 21+ with the AMDGPU target)", err)
        self.assertEqual(err.count("vk_pipeline_check"), 1)


if __name__ == "__main__":
    unittest.main()
