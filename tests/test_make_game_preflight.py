"""Unit test suite covering make_game.py prerequisite checking, version string parsing, and toolchain detection.

    python -I tests/test_make_game_preflight.py
    python -m unittest discover -s tests -v

Tests use synthetic manifests, mock environments, and isolated temporary directories so they are deterministic,
fast, and contain zero game data.
"""
from collections import namedtuple
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import make_game  # noqa: E402

Usage = namedtuple("Usage", ["total", "used", "free"])


class PreflightTestBase(unittest.TestCase):
    def assert_fails(self, fn, *args, match=None, **kwargs):
        stderr_buf = io.StringIO()
        with mock.patch("sys.stderr", stderr_buf):
            with self.assertRaises(SystemExit) as ctx:
                fn(*args, **kwargs)
            self.assertEqual(ctx.exception.code, 1)
        err = stderr_buf.getvalue()
        if match:
            self.assertIn(match, err)
        return err


class VersionParsingAndMatchingTest(PreflightTestBase):
    def test_versions_loads_valid_manifest(self):
        manifests = make_game.versions()
        self.assertGreaterEqual(len(manifests), 1)
        v = manifests[0]
        self.assertIn("title", v)
        self.assertIn("version", v)
        self.assertIn("files", v)
        self.assertIn("expected_exe", v)
        self.assertEqual(v["version"], "1.0.2.1")
        self.assertEqual(v["expected_exe"], "game_scarlett_master.exe")

    def test_version_string_parsing_and_ordering(self):
        manifests = make_game.versions()
        parsed = [tuple(map(int, v["version"].split("."))) for v in manifests]
        self.assertIn((1, 0, 2, 1), parsed)
        self.assertGreater((1, 0, 2, 1), (1, 0, 2, 0))
        self.assertGreater((1, 0, 2, 0), (1, 0, 1, 9))
        self.assertGreater((2, 0, 0, 0), (1, 9, 9, 9))

    def test_versions_custom_directory_and_json_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            v1_path = os.path.join(tmp, "b_1.0.0.json")
            v2_path = os.path.join(tmp, "a_2.0.0.json")
            txt_path = os.path.join(tmp, "readme.txt")

            with open(v1_path, "w", encoding="utf-8") as f:
                json.dump({"title": "Custom B", "version": "1.0.0", "files": {}, "expected_exe": "b.exe"}, f)
            with open(v2_path, "w", encoding="utf-8") as f:
                json.dump({"title": "Custom A", "version": "2.0.0", "files": {}, "expected_exe": "a.exe"}, f)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("ignored")

            with mock.patch.object(make_game, "VERSIONS", tmp):
                loaded = make_game.versions()
                self.assertEqual(len(loaded), 2)
                # sorted by filename in os.listdir
                self.assertEqual(loaded[0]["version"], "2.0.0")
                self.assertEqual(loaded[1]["version"], "1.0.0")

    def test_match_version_success_on_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_game.versions()[0]
            for probe_name in make_game.PROBES:
                size = manifest["files"][probe_name]["size"]
                probe_path = os.path.join(tmp, probe_name)
                with open(probe_path, "wb") as f:
                    f.truncate(size)

            matched = make_game.match_version(tmp)
            self.assertIsNotNone(matched)
            self.assertEqual(matched["version"], manifest["version"])

    def test_match_version_missing_probe_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_game.versions()[0]
            # Create 2 probes, skip appdata0.rpf
            for probe_name in ("MicrosoftGame.Config", "Catalog000.bin"):
                size = manifest["files"][probe_name]["size"]
                probe_path = os.path.join(tmp, probe_name)
                with open(probe_path, "wb") as f:
                    f.truncate(size)

            self.assertIsNone(make_game.match_version(tmp))

    def test_match_version_wrong_probe_size_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_game.versions()[0]
            for probe_name in make_game.PROBES:
                size = manifest["files"][probe_name]["size"]
                probe_path = os.path.join(tmp, probe_name)
                with open(probe_path, "wb") as f:
                    f.truncate(size + 1 if probe_name == "MicrosoftGame.Config" else size)

            self.assertIsNone(make_game.match_version(tmp))

    def test_match_version_empty_directory_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(make_game.match_version(tmp))

    def test_load_version_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_game.versions()[0]
            for probe_name in make_game.PROBES:
                size = manifest["files"][probe_name]["size"]
                probe_path = os.path.join(tmp, probe_name)
                with open(probe_path, "wb") as f:
                    f.truncate(size)

            loaded = make_game.load_version(tmp)
            self.assertEqual(loaded["version"], manifest["version"])

    def test_load_version_failure_reports_supported_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            err = self.assert_fails(make_game.load_version, tmp,
                                    match="This doesn't look like a supported version of the game")
            self.assertIn("Supported:", err)
            self.assertIn("1.0.2.1", err)


class GameCheckTest(PreflightTestBase):
    def test_check_nonexistent_directory_fails(self):
        self.assert_fails(make_game.check, "nonexistent_dir_0xdeadbeef", full=False,
                          match="Game folder not found: nonexistent_dir_0xdeadbeef")

    def test_check_synthetic_game_all_files_present_with_exe(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = os.path.join(tmp, "manifests")
            game_dir = os.path.join(tmp, "game")
            os.makedirs(manifest_dir)
            os.makedirs(game_dir)

            v_info = {
                "title": "Synthetic Port Test",
                "version": "9.9.9.9",
                "expected_exe": "test_game.exe",
                "files": {
                    "MicrosoftGame.Config": {"size": 16},
                    "Catalog000.bin": {"size": 32},
                    "appdata0.rpf": {"size": 64},
                    "sub/data.bin": {"size": 128},
                },
            }
            with open(os.path.join(manifest_dir, "custom.json"), "w", encoding="utf-8") as f:
                json.dump(v_info, f)

            for rel, info in v_info["files"].items():
                p = os.path.join(game_dir, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f:
                    f.truncate(info["size"])

            exe_p = os.path.join(game_dir, v_info["expected_exe"])
            with open(exe_p, "wb") as f:
                f.truncate(256)

            stdout_buf = io.StringIO()
            with mock.patch.object(make_game, "VERSIONS", manifest_dir):
                with mock.patch("sys.stdout", stdout_buf):
                    res = make_game.check(game_dir, full=False)

            self.assertEqual(res["version"], "9.9.9.9")
            self.assertIn("all files present", stdout_buf.getvalue())
            self.assertIn("Executable found.", stdout_buf.getvalue())

    def test_check_missing_files_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = os.path.join(tmp, "manifests")
            game_dir = os.path.join(tmp, "game")
            os.makedirs(manifest_dir)
            os.makedirs(game_dir)

            v_info = {
                "title": "Synthetic Missing Test",
                "version": "9.9.9.9",
                "expected_exe": "test_game.exe",
                "files": {
                    "MicrosoftGame.Config": {"size": 16},
                    "Catalog000.bin": {"size": 32},
                    "appdata0.rpf": {"size": 64},
                    "missing.bin": {"size": 100},
                },
            }
            with open(os.path.join(manifest_dir, "custom.json"), "w", encoding="utf-8") as f:
                json.dump(v_info, f)

            for rel in ("MicrosoftGame.Config", "Catalog000.bin", "appdata0.rpf"):
                p = os.path.join(game_dir, rel)
                with open(p, "wb") as f:
                    f.truncate(v_info["files"][rel]["size"])

            with mock.patch.object(make_game, "VERSIONS", manifest_dir):
                self.assert_fails(make_game.check, game_dir, full=False,
                                  match="1 file(s) missing or different")

    def test_check_wrong_size_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = os.path.join(tmp, "manifests")
            game_dir = os.path.join(tmp, "game")
            os.makedirs(manifest_dir)
            os.makedirs(game_dir)

            v_info = {
                "title": "Synthetic Size Test",
                "version": "9.9.9.9",
                "expected_exe": "test_game.exe",
                "files": {
                    "MicrosoftGame.Config": {"size": 16},
                    "Catalog000.bin": {"size": 32},
                    "appdata0.rpf": {"size": 64},
                    "extra.bin": {"size": 100},
                },
            }
            with open(os.path.join(manifest_dir, "custom.json"), "w", encoding="utf-8") as f:
                json.dump(v_info, f)

            for rel, info in v_info["files"].items():
                p = os.path.join(game_dir, rel)
                with open(p, "wb") as f:
                    f.truncate(info["size"] + 5 if rel == "extra.bin" else info["size"])

            with mock.patch.object(make_game, "VERSIONS", manifest_dir):
                self.assert_fails(make_game.check, game_dir, full=False,
                                  match="1 file(s) missing or different")

    def test_check_missing_exe_logs_warning_and_proceeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = os.path.join(tmp, "manifests")
            game_dir = os.path.join(tmp, "game")
            os.makedirs(manifest_dir)
            os.makedirs(game_dir)

            v_info = {
                "title": "Synthetic No Exe",
                "version": "9.9.9.9",
                "expected_exe": "custom_loader.exe",
                "files": {
                    "MicrosoftGame.Config": {"size": 16},
                    "Catalog000.bin": {"size": 32},
                    "appdata0.rpf": {"size": 64},
                },
            }
            with open(os.path.join(manifest_dir, "custom.json"), "w", encoding="utf-8") as f:
                json.dump(v_info, f)

            for rel, info in v_info["files"].items():
                p = os.path.join(game_dir, rel)
                with open(p, "wb") as f:
                    f.truncate(info["size"])

            stdout_buf = io.StringIO()
            with mock.patch.object(make_game, "VERSIONS", manifest_dir):
                with mock.patch("sys.stdout", stdout_buf):
                    res = make_game.check(game_dir, full=False)

            self.assertEqual(res["version"], "9.9.9.9")
            self.assertIn("custom_loader.exe is not in the folder", stdout_buf.getvalue())


class ToolchainDetectionTest(PreflightTestBase):
    def test_tool_available_direct_file(self):
        self.assertTrue(make_game.tool_available(sys.executable))

    def test_tool_available_quoted_string(self):
        self.assertTrue(make_game.tool_available(f'"{sys.executable}"'))

    def test_tool_available_command_with_arguments(self):
        self.assertTrue(make_game.tool_available(f'"{sys.executable}" -I'))

    def test_tool_available_shutil_which_lookup(self):
        with mock.patch("shutil.which", return_value=r"C:\bin\custom_tool.exe"):
            self.assertTrue(make_game.tool_available("custom_tool"))

    def test_tool_available_nonexistent_command(self):
        self.assertFalse(make_game.tool_available("nonexistent_compiler_xyz_789"))

    def test_tool_available_empty_and_whitespace(self):
        self.assertFalse(make_game.tool_available(""))
        self.assertFalse(make_game.tool_available("   "))
        self.assertFalse(make_game.tool_available('""'))

    def test_need_success_returns_tool(self):
        self.assertEqual(make_game.need(sys.executable, "Install python"), sys.executable)

    def test_need_failure_exits_with_hint(self):
        self.assert_fails(make_game.need, "ghost_compiler", "Please install it.",
                          match="'ghost_compiler' was not found on PATH. Please install it.")


class PreflightPrerequisiteTest(PreflightTestBase):
    def setUp(self):
        super().setUp()
        # preflight runs `dotnet --list-sdks` (#223). Follow the test's faked shutil.which instead: the SDK counts as
        # installed exactly when the test says dotnet is on PATH, so no real dotnet is needed and none is consulted.
        patcher = mock.patch("make_game.dotnet_sdk_ok", side_effect=lambda: bool(shutil.which("dotnet")))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_preflight_success_all_tools_present_and_space_sufficient(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 11, 0)), \
                 mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("make_game.ensure_submodule", return_value=None), \
                 mock.patch("os.path.isfile", return_value=True), \
                 mock.patch("shutil.disk_usage", return_value=Usage(1000 * 2**30, 200 * 2**30, 800 * 2**30)):
                # runtime is refused before any check off Windows (tests/test_make_game_platform.py)
                stages = ["shaders", "assets"] + (["runtime"] if os.name == "nt" else [])
                make_game.preflight(stages, tmp_work, wav=True)

    def test_preflight_aggregates_multiple_missing_tools_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 11, 0)), \
                 mock.patch("shutil.which", return_value=None), \
                 mock.patch("make_game.tool_available", return_value=False), \
                 mock.patch("make_game.ensure_submodule", return_value=None), \
                 mock.patch("os.path.isfile", return_value=False), \
                 mock.patch("os.name", "nt"):
                err = self.assert_fails(make_game.preflight, ["shaders", "assets", "runtime"], tmp_work, wav=True,
                                        match="Install these first, then run the build again:")
                self.assertIn("ffmpeg (needed for --wav)", err)
                self.assertIn("llvm-mc (LLVM 21+ with the AMDGPU target)", err)
                self.assertIn("spirv-as:", err)
                self.assertIn("spirv-val:", err)
                self.assertIn(".NET 8 SDK", err)
                self.assertIn("Visual Studio 2022", err)

    def test_preflight_python_version_check(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 10, 8)), \
                 mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("make_game.ensure_submodule", return_value=None), \
                 mock.patch("os.path.isfile", return_value=True), \
                 mock.patch("shutil.disk_usage", return_value=Usage(1000 * 2**30, 100 * 2**30, 900 * 2**30)):
                self.assert_fails(make_game.preflight, ["shaders"], tmp_work, wav=False,
                                  match="Python 3.11 or newer:")

    def test_preflight_wav_without_assets_does_not_require_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            def mock_which(cmd):
                return None if cmd == "ffmpeg" else f"/bin/{cmd}"

            with mock.patch("sys.version_info", (3, 11, 0)), \
                 mock.patch("shutil.which", side_effect=mock_which), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("shutil.disk_usage", return_value=Usage(1000 * 2**30, 100 * 2**30, 900 * 2**30)):
                # wav=True with ["shaders"] stage only: ffmpeg not required
                make_game.preflight(["shaders"], tmp_work, wav=True)

    def test_preflight_assets_without_wav_does_not_require_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            def mock_which(cmd):
                return None if cmd == "ffmpeg" else f"/bin/{cmd}"

            with mock.patch("sys.version_info", (3, 11, 0)), \
                 mock.patch("shutil.which", side_effect=mock_which), \
                 mock.patch("make_game.ensure_submodule", return_value=None), \
                 mock.patch("shutil.disk_usage", return_value=Usage(1000 * 2**30, 100 * 2**30, 900 * 2**30)):
                # wav=False with ["assets"] stage: ffmpeg not required
                make_game.preflight(["assets"], tmp_work, wav=False)

    def test_preflight_disk_space_insufficient(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 11, 0)), \
                 mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("make_game.ensure_submodule", return_value=None), \
                 mock.patch("os.path.isfile", return_value=True), \
                 mock.patch("shutil.disk_usage", return_value=Usage(500 * 2**30, 485 * 2**30, 15 * 2**30)):
                # Shaders needs 30 GB, only 15 GB free
                self.assert_fails(make_game.preflight, ["shaders"], tmp_work, wav=False,
                                  match="15 GB free on the drive of")

    def test_preflight_disk_space_stage_accumulation(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 11, 0)), \
                 mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("make_game.ensure_submodule", return_value=None), \
                 mock.patch("os.path.isfile", return_value=True), \
                 mock.patch("shutil.disk_usage", return_value=Usage(500 * 2**30, 250 * 2**30, 250 * 2**30)):
                # shaders (30) + assets (300) = 330 GB needed, only 250 GB free
                err = self.assert_fails(make_game.preflight, ["shaders", "assets"], tmp_work, wav=False,
                                        match="about 330 GB is needed")
                self.assertIn("250 GB free", err)

    def test_ensure_submodule_present_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            core_dir = os.path.join(tmp, "third_party", "rpf9-explorer", "Core")
            os.makedirs(core_dir)
            with open(os.path.join(core_dir, "Core.cs"), "w") as f:
                f.write("// dummy")

            with mock.patch.object(make_game, "ROOT", tmp):
                make_game.ensure_submodule()

    def test_ensure_submodule_missing_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(make_game, "ROOT", tmp), \
                 mock.patch("shutil.which", return_value=None):
                self.assert_fails(make_game.ensure_submodule,
                                  match="third_party/rpf9-explorer is missing. Download the project with:")


class PathAndGameSelectionTest(PreflightTestBase):
    def test_choose_game_single_found(self):
        dummy_entry = ("D:\\TestGames\\Leonida", {"title": "Xbox Package", "version": "1.0.2.1"})
        with mock.patch("make_game.find_dumps", return_value=[dummy_entry]):
            chosen = make_game.choose_game()
            self.assertEqual(chosen, "D:\\TestGames\\Leonida")

    def test_choose_game_none_found_non_interactive(self):
        with mock.patch("make_game.find_dumps", return_value=[]), \
             mock.patch("sys.stdin.isatty", return_value=False):
            self.assert_fails(make_game.choose_game,
                              match="Your game files weren't found. Give the folder with --game.")

    def test_choose_game_multiple_found_non_interactive(self):
        dumps = [
            ("D:\\Games\\Copy1", {"title": "Copy 1", "version": "1.0.2.1"}),
            ("D:\\Games\\Copy2", {"title": "Copy 2", "version": "1.0.2.1"}),
        ]
        with mock.patch("make_game.find_dumps", return_value=dumps), \
             mock.patch("sys.stdin.isatty", return_value=False):
            self.assert_fails(make_game.choose_game,
                              match="More than one copy found: choose one with --game.")

    def test_default_out_matches_documents_folder(self):
        expected = os.path.join(os.path.expanduser("~"), "Documents", "Leonida")
        self.assertEqual(make_game.default_out(), expected)

    def test_constants_definitions(self):
        self.assertEqual(make_game.STAGES, ["check", "shaders", "assets", "runtime", "pipelines"])
        self.assertEqual(make_game.PROBES, ("MicrosoftGame.Config", "Catalog000.bin", "appdata0.rpf"))
        self.assertEqual(make_game.MIN_FREE_GB, {"shaders": 30, "assets": 300})

    def test_below_normal_priority_does_not_crash(self):
        make_game.below_normal()


if __name__ == "__main__":
    unittest.main()
