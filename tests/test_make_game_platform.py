"""Platform-specific build stages, runtime packaging and vk_pipeline_check discovery.

    python -I tests/test_make_game_platform.py

No game data: stage selection uses mocks, and the packaging and discovery tests create synthetic files.
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

STAGE_FUNCS = ("shaders", "assets", "runtime", "pipelines")


class DefaultStagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.game = os.path.join(self.tmp.name, "game")
        os.makedirs(self.game)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, os_name, *args):
        argv = ["make_game.py", "--game", self.game, "--out", os.path.join(self.tmp.name, "out"), *args]
        mocks = {n: mock.MagicMock() for n in STAGE_FUNCS}
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(make_game.os, "name", os_name), \
                mock.patch.object(make_game, "below_normal"), \
                mock.patch.object(make_game, "check", return_value={"expected_exe": "game.exe"}), \
                mock.patch.object(make_game, "preflight") as preflight, \
                mock.patch.object(make_game, "log"), \
                mock.patch.multiple(make_game, **mocks):
            self.assertEqual(make_game.main(), 0)
        return preflight.call_args.args[0], {n: m.called for n, m in mocks.items()}

    def test_default_stages_leave_runtime_out_off_windows(self):
        stages, ran = self.run_main("posix")
        self.assertEqual(stages, ["check", "shaders", "assets", "pipelines"])
        self.assertEqual(ran, {"shaders": True, "assets": True, "runtime": False, "pipelines": True})

    def test_default_stages_keep_runtime_on_windows(self):
        stages, ran = self.run_main("nt")
        self.assertEqual(stages, make_game.STAGES)
        self.assertTrue(ran["runtime"])

    def test_explicit_full_stage_list_reaches_preflight_off_windows(self):
        stages, _ = self.run_main("posix", "--stages", ",".join(make_game.STAGES), "--wave64-emulate", "off")
        self.assertEqual(stages, make_game.STAGES)

    def test_explicit_stages_are_kept_as_given(self):
        stages, _ = self.run_main("posix", "--stages", "shaders,runtime")
        self.assertEqual(stages, ["shaders", "runtime"])


class PreflightTest(unittest.TestCase):
    def test_runtime_asked_for_off_windows_fails_before_any_work(self):
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as work, \
                mock.patch.object(make_game.os, "name", "posix"), \
                mock.patch("sys.stderr", err), \
                mock.patch.object(make_game.shutil, "disk_usage") as usage:
            with self.assertRaises(SystemExit) as ctx:
                make_game.preflight(["shaders", "runtime"], os.path.join(work, "w"))
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("runtime stage builds on Windows only", err.getvalue())
        usage.assert_not_called()
        self.assertFalse(os.path.exists(os.path.join(work, "w")))


class RuntimePackagingTest(unittest.TestCase):
    def setUp(self):
        self.root = self.enterContext(tempfile.TemporaryDirectory())
        self.build = os.path.join(self.root, "runtime", "build")
        self.game = os.path.join(self.root, "game")
        self.out = os.path.join(self.root, "out")
        os.makedirs(self.build)
        os.makedirs(self.game)
        self.write_artifact("gta6rt.exe", b"runtime")

    def write_artifact(self, name, contents):
        with open(os.path.join(self.build, name), "wb") as f:
            f.write(contents)

    def package(self):
        with mock.patch.object(make_game.os, "name", "nt"), \
                mock.patch.object(make_game, "ROOT", self.root), \
                mock.patch.object(make_game, "run"), \
                mock.patch.object(make_game, "log"):
            make_game.runtime(self.game, self.out, {"expected_exe": "game.exe"})

    def test_runtime_copies_executable_and_tls_host_together(self):
        self.write_artifact("gta6rt_tls_host.dll", b"TLS host")

        self.package()

        with open(os.path.join(self.out, "gta6rt.exe"), "rb") as f:
            self.assertEqual(f.read(), b"runtime")
        with open(os.path.join(self.out, "gta6rt_tls_host.dll"), "rb") as f:
            self.assertEqual(f.read(), b"TLS host")
        with open(os.path.join(self.out, "game_path.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), self.game)

    def test_runtime_rejects_missing_tls_host_before_copying(self):
        err = io.StringIO()
        with mock.patch("sys.stderr", err), self.assertRaises(SystemExit) as ctx:
            self.package()

        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("gta6rt_tls_host.dll", err.getvalue())
        self.assertFalse(os.path.exists(self.out))


class PipelineCheckExeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.build = os.path.join(self.tmp.name, "runtime", "build")
        os.makedirs(os.path.join(self.build, "linux"))
        self.root = mock.patch.object(make_game, "ROOT", self.tmp.name)
        self.root.start()

    def tearDown(self):
        self.root.stop()
        self.tmp.cleanup()

    def touch(self, *parts):
        p = os.path.join(self.build, *parts)
        open(p, "w").close()
        return p

    def exe(self, os_name):
        with mock.patch.object(make_game.os, "name", os_name):
            return make_game.pipeline_check_exe()

    def test_linux_prefers_the_readme_build_folder(self):
        self.assertIsNone(self.exe("posix"))
        plain = self.touch("vk_pipeline_check")
        self.assertEqual(self.exe("posix"), plain)
        readme = self.touch("linux", "vk_pipeline_check")
        self.assertEqual(self.exe("posix"), readme)

    def test_windows_finds_only_the_exe(self):
        self.touch("linux", "vk_pipeline_check")
        self.assertIsNone(self.exe("nt"))
        exe = self.touch("vk_pipeline_check.exe")
        self.assertEqual(self.exe("nt"), exe)

    def test_pipelines_off_windows_runs_the_linux_build(self):
        exe = self.touch("linux", "vk_pipeline_check")
        work = os.path.join(self.tmp.name, "work")
        os.makedirs(os.path.join(work, "shaders", "spirv"))
        open(os.path.join(work, "shaders", "spirv", "index.json"), "w").close()
        out = os.path.join(self.tmp.name, "out")
        with mock.patch.object(make_game.os, "name", "posix"), \
                mock.patch.object(make_game, "log"), \
                mock.patch.object(make_game, "run") as run:
            make_game.pipelines(work, out)
        self.assertEqual(run.call_args.args[0][0], exe)

    def test_pipelines_off_windows_without_the_build_says_how_to_build_it(self):
        err = io.StringIO()
        with mock.patch.object(make_game.os, "name", "posix"), mock.patch("sys.stderr", err):
            with self.assertRaises(SystemExit):
                make_game.pipelines(os.path.join(self.tmp.name, "work"), os.path.join(self.tmp.name, "out"))
        self.assertIn("cmake -S runtime -B runtime/build/linux", err.getvalue())


if __name__ == "__main__":
    unittest.main()
