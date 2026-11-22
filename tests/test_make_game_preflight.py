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
            for name in make_game.PROBES:
                self.assertIn(f"Missing {name}.", err)


class GameCheckTest(PreflightTestBase):
    def test_check_nonexistent_directory_fails(self):
        self.assert_fails(make_game.check, "nonexistent_dir_0xdeadbeef", full=False,
                          match="Game folder not found: nonexistent_dir_0xdeadbeef")

    def test_check_option_reports_each_version_probe_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "MicrosoftGame.Config"), "wb") as config:
                config.write(b"test")
            os.mkdir(os.path.join(tmp, "Catalog000.bin"))
            manifests = [
                {"title": "Synthetic", "version": version, "files": {
                    "MicrosoftGame.Config": {"size": size},
                    "Catalog000.bin": {"size": 8},
                    "appdata0.rpf": {"size": 16},
                }}
                for version, size in [("1.0", 6), ("2.0", 10)]
            ]
            with mock.patch.object(make_game, "versions", return_value=manifests), \
                 mock.patch.object(sys, "argv", ["make_game.py", "--check", "--game", tmp]):
                err = self.assert_fails(make_game.main)

        self.assertIn("Synthetic 1.0\n    Wrong size for MicrosoftGame.Config (4 bytes, expected 6).", err)
        self.assertIn("Synthetic 2.0\n    Wrong size for MicrosoftGame.Config (4 bytes, expected 10).", err)
        self.assertEqual(err.count("Catalog000.bin is not a regular file."), 2)
        self.assertEqual(err.count("Missing appdata0.rpf."), 2)

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

    APPDATA_ENV = {"LOCALAPPDATA": r"C:\Users\u\AppData\Local", "APPDATA": r"C:\Users\u\AppData\Roaming"}

    def test_appdata_hint_names_the_folder_llvm_mc_is_in(self):
        with mock.patch("os.name", "nt"), mock.patch.dict(os.environ, self.APPDATA_ENV):
            for cmd, var in ((r"C:\Users\u\AppData\Local\Leonida\llvm-mc\llvm-mc.exe", "LOCALAPPDATA"),
                             (r'"c:\users\U\appdata\roaming\llvm\llvm-mc.exe"', "APPDATA")):
                with self.subTest(cmd=cmd):
                    hint = make_game.appdata_hint(cmd)
                    self.assertIn(f"LLVM_MC points into %{var}%", hint)
                    self.assertIn(r"%USERPROFILE%\.leonida\llvm-mc", hint)

    def test_appdata_hint_is_empty_outside_appdata(self):
        with mock.patch("os.name", "nt"), mock.patch.dict(os.environ, self.APPDATA_ENV):
            for cmd in ("llvm-mc", r"D:\Tools\llvm-mc\llvm-mc.exe", r"C:\Users\u\.leonida\llvm-mc\llvm-mc.exe",
                        r"C:\Users\u\AppData\LocalLow\llvm-mc.exe", "wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc"):
                with self.subTest(cmd=cmd):
                    self.assertEqual(make_game.appdata_hint(cmd), "")

    def test_appdata_hint_is_empty_off_windows(self):
        with mock.patch("os.name", "posix"), mock.patch.dict(os.environ, self.APPDATA_ENV):
            self.assertEqual(make_game.appdata_hint(r"C:\Users\u\AppData\Local\llvm-mc.exe"), "")

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
            with mock.patch("sys.version_info", (3, 12, 0)), \
                 mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("make_game.ensure_submodule", return_value=None), \
                 mock.patch("make_game.visual_cpp_tools_ok", return_value=True), \
                 mock.patch("shutil.disk_usage", return_value=Usage(1000 * 2**30, 200 * 2**30, 800 * 2**30)):
                # off Windows runtime checks the cross-build instead (tests/test_make_game_platform.py)
                stages = ["shaders", "assets"] + (["runtime"] if os.name == "nt" else [])
                make_game.preflight(stages, tmp_work, wav=True)

    def test_preflight_aggregates_multiple_missing_tools_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 12, 0)), \
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

    def test_preflight_explains_llvm_mc_hidden_in_appdata(self):
        env = {"LOCALAPPDATA": r"C:\Users\u\AppData\Local",
               "LLVM_MC": r"C:\Users\u\AppData\Local\Leonida\llvm-mc\llvm-mc.exe"}
        with tempfile.TemporaryDirectory() as tmp_work, \
             mock.patch.dict(os.environ, env), \
             mock.patch("os.name", "nt"), \
             mock.patch("sys.version_info", (3, 12, 0)), \
             mock.patch("shutil.which", return_value="/usr/bin/tool"), \
             mock.patch("make_game.tool_available", return_value=False):
            err = self.assert_fails(make_game.preflight, ["shaders"], tmp_work,
                                    match="llvm-mc (LLVM 21+ with the AMDGPU target)")
        self.assertIn("LLVM_MC points into %LOCALAPPDATA%", err)

    def test_preflight_rejects_visual_studio_installer_without_cpp_tools(self):
        with tempfile.TemporaryDirectory() as tmp_work, \
             mock.patch("os.name", "nt"), \
             mock.patch("os.path.isfile", return_value=True), \
             mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stdout=b"\r\n")):
            self.assert_fails(make_game.preflight, ["runtime"], tmp_work, match="Visual Studio 2022")

    def test_visual_cpp_tools_accepts_matching_installation(self):
        with mock.patch("os.path.isfile", return_value=True), \
             mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stdout=b"C:\\VS\r\n")) as run:
            self.assertTrue(make_game.visual_cpp_tools_ok())
        self.assertEqual(run.call_args.args[0][1:], ["-latest", "-products", "*", "-requires",
                                                   "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                                   "-property", "installationPath"])

    def test_preflight_python_version_check(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 11, 9)), \
                 mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("make_game.ensure_submodule", return_value=None), \
                 mock.patch("os.path.isfile", return_value=True), \
                 mock.patch("shutil.disk_usage", return_value=Usage(1000 * 2**30, 100 * 2**30, 900 * 2**30)):
                self.assert_fails(make_game.preflight, ["shaders"], tmp_work, wav=False,
                                  match="Python 3.12 or newer:")

    def test_preflight_wav_without_assets_does_not_require_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            def mock_which(cmd):
                return None if cmd == "ffmpeg" else f"/bin/{cmd}"

            with mock.patch("sys.version_info", (3, 12, 0)), \
                 mock.patch("shutil.which", side_effect=mock_which), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("shutil.disk_usage", return_value=Usage(1000 * 2**30, 100 * 2**30, 900 * 2**30)):
                # wav=True with ["shaders"] stage only: ffmpeg not required
                make_game.preflight(["shaders"], tmp_work, wav=True)

    def test_preflight_assets_without_wav_does_not_require_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            def mock_which(cmd):
                return None if cmd == "ffmpeg" else f"/bin/{cmd}"

            with mock.patch("sys.version_info", (3, 12, 0)), \
                 mock.patch("shutil.which", side_effect=mock_which), \
                 mock.patch("make_game.ensure_submodule", return_value=None), \
                 mock.patch("shutil.disk_usage", return_value=Usage(1000 * 2**30, 100 * 2**30, 900 * 2**30)):
                # wav=False with ["assets"] stage: ffmpeg not required
                make_game.preflight(["assets"], tmp_work, wav=False)

    def test_preflight_disk_space_insufficient(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 12, 0)), \
                 mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("make_game.ensure_submodule", return_value=None), \
                 mock.patch("os.path.isfile", return_value=True), \
                 mock.patch("shutil.disk_usage", return_value=Usage(500 * 2**30, 485 * 2**30, 15 * 2**30)):
                # Shaders needs 30 GB, only 15 GB free
                self.assert_fails(make_game.preflight, ["shaders"], tmp_work, wav=False,
                                  match="15 GB free on the drive of")

    def test_preflight_charges_space_only_when_shaders_need_to_be_rebuilt(self):
        cases = (
            ("current translation", {}, {}, False, True),
            ("matching emulation", {"spirv/wave64_emulate.txt": "on"}, {"emulate": True}, False, True),
            ("auto keeps emulation", {"spirv/wave64_emulate.txt": "on"}, {"keep_emulated": True}, False, True),
            ("missing disassembly marker", {"dxbc/disasm.done": None}, {}, False, False),
            ("old disassembly marker", {"dxbc/disasm.done": "shader_disasm.py finished"}, {}, False, False),
            ("changed LLVM", {"dxbc/disasm.done": "old llvm-mc"}, {}, False, False),
            ("stale translator", {"spirv/translation_sources.sha256": "stale"}, {}, False, False),
            ("old translation without fingerprint", {"spirv/translation_sources.sha256": None}, {}, False, False),
            ("needs emulation", {}, {"emulate": True}, False, False),
            ("explicit native mode", {"spirv/wave64_emulate.txt": "on"}, {}, False, False),
            ("interrupted unpack", {"dxbc/fssm_index.json": None}, {}, False, False),
            ("interrupted translation", {"spirv/index.json": None}, {}, False, False),
            ("old disassembly layout", {}, {}, True, False),
        )
        fingerprint = make_game.translation_sources_digest()
        usage = Usage(500 * 2**30, 485 * 2**30, 15 * 2**30)
        for name, changed, settings, legacy_layout, reusable in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as work:
                files = {
                    "dxbc/index.json": "{}",
                    "dxbc/fssm_index.json": "{}",
                    "dxbc/disasm.done": "test llvm-mc 21",
                    "spirv/index.json": "{}",
                    "spirv/translation_sources.sha256": fingerprint,
                    "spirv/wave64_emulate.txt": "off",
                    **changed,
                }
                shader_root = os.path.join(work, "shaders")
                for name, content in files.items():
                    if content is not None:
                        path = os.path.join(shader_root, name)
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, "w", encoding="utf-8") as handle:
                            handle.write(content + "\n")
                disasm_parts = ("disasm",) if legacy_layout else ("dxbc", "disasm")
                os.makedirs(os.path.join(shader_root, *disasm_parts))
                with mock.patch("sys.version_info", (3, 12, 0)), \
                     mock.patch("make_game.llvm_mc_signature", return_value="test llvm-mc 21"), \
                     mock.patch("make_game.tool_available", return_value=True), \
                     mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                     mock.patch("shutil.disk_usage", return_value=usage), \
                     mock.patch("sys.stdout", io.StringIO()):
                    if reusable:
                        make_game.preflight(["shaders"], work, **settings)
                    else:
                        self.assert_fails(make_game.preflight, ["shaders"], work,
                                          match="about 30 GB is needed", **settings)


    def test_preflight_disk_space_stage_accumulation(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 12, 0)), \
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


class GameDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.home = self.enterContext(tempfile.TemporaryDirectory())
        self.desktop = os.path.join(self.home, "Desktop")
        self.downloads = os.path.join(self.home, "Downloads")
        self.version = {"title": "Synthetic", "version": "1.0", "files": {
            "MicrosoftGame.Config": {"size": 3},
            "Catalog000.bin": {"size": 4},
            "appdata0.rpf": {"size": 5},
        }}

    def write_probes(self, folder):
        os.makedirs(folder)
        for name, info in self.version["files"].items():
            with open(os.path.join(folder, name), "wb") as file:
                file.truncate(info["size"])

    def discover(self, known=()):
        """find_dumps() over the synthetic home folder only; `known` is what the shell reports for Downloads,
        Desktop and Documents."""
        roots = (self.home, self.desktop, self.downloads, *known)
        isdir = os.path.isdir
        with mock.patch("os.path.expanduser", return_value=self.home), \
             mock.patch("os.path.isdir", side_effect=lambda path: path in roots and isdir(path)), \
             mock.patch.object(make_game, "known_folders", return_value=list(known)), \
             mock.patch.object(make_game, "versions", return_value=[self.version]):
            return make_game.find_dumps()

    def test_find_searches_known_folders_moved_into_onedrive(self):
        # OneDrive's Known Folder Move: Explorer's Desktop and Documents are ~\OneDrive\Desktop and
        # ~\OneDrive\Documents, two levels below the home folder, so the home root alone never reaches inside them.
        desktop = os.path.join(self.home, "OneDrive", "Desktop")
        documents = os.path.join(self.home, "OneDrive", "Documents")
        on_desktop = os.path.join(desktop, "GTA VI")
        in_documents = os.path.join(documents, "Games", "GTA VI")
        self.write_probes(on_desktop)
        self.write_probes(in_documents)

        self.assertEqual(self.discover(), [])
        self.assertEqual(self.discover(known=(desktop, documents)),
                         [(on_desktop, self.version), (in_documents, self.version)])

    def test_find_reports_a_copy_once_when_known_folders_are_not_moved(self):
        package = os.path.join(self.desktop, "Package")
        self.write_probes(package)
        self.assertEqual(self.discover(known=(self.downloads, self.desktop)), [(package, self.version)])

    def test_known_folders_from_the_shell(self):
        folders = make_game.known_folders()
        if os.name != "nt":
            self.assertEqual(folders, [])
            return
        self.assertEqual(len(folders), len(make_game.KNOWN_FOLDERS))
        for folder in folders:
            self.assertTrue(os.path.isabs(folder), folder)
        self.assertEqual(make_game.known_folders(), folders)   # the shell's buffers are freed, calls repeat cleanly

    def test_find_searches_below_rejected_folder_with_config(self):
        outer = os.path.join(self.desktop, "Backup")
        package = os.path.join(outer, "Package")
        self.write_probes(package)
        with open(os.path.join(outer, "MicrosoftGame.Config"), "wb") as config:
            config.write(b"abc")
        self.write_probes(os.path.join(outer, "Other", "TooDeep"))

        self.assertEqual(self.discover(), [(package, self.version)])

    def test_find_revisits_aliased_root_with_more_depth(self):
        target = os.path.join(self.downloads, "Outer", "Inner")
        package = os.path.join(self.desktop, "Package")
        self.write_probes(os.path.join(target, "Package"))
        self.write_probes(package)
        realpath = os.path.realpath

        def resolve(path):
            if path == self.desktop or path.startswith(self.desktop + os.sep):
                path = os.path.join(target, os.path.relpath(path, self.desktop))
            return realpath(path)

        with mock.patch("os.path.realpath", side_effect=resolve):
            self.assertEqual(self.discover(), [(package, self.version)])


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

    def default_out_in(self, platform, icloud_documents, earlier_build=False, call=None):
        """default_out() (or `call`) with a temporary home folder; icloud_documents makes the folder macOS keeps in
        iCloud Drive when Desktop & Documents sync is on (~/Library/Mobile Documents/com~apple~CloudDocs/Documents),
        earlier_build a ~/Documents/Leonida from before the change."""
        with tempfile.TemporaryDirectory() as home:
            if icloud_documents:
                os.makedirs(os.path.join(home, "Library", "Mobile Documents", "com~apple~CloudDocs", "Documents"))
            if earlier_build:
                os.makedirs(os.path.join(home, "Documents", "Leonida", "work"))
            with mock.patch.object(make_game.os.path, "expanduser", lambda p: home + p[1:] if p.startswith("~") else p), \
                    mock.patch.object(make_game.sys, "platform", platform):
                if call:
                    return [note.replace(home, "~") for note in call()]
                return os.path.relpath(make_game.default_out(), home)

    def test_icloud_note_names_an_earlier_build_in_documents(self):
        # A Mac that built into ~/Documents/Leonida before the default moved: say where that resumable build is.
        notes = self.default_out_in("darwin", True, earlier_build=True, call=make_game.icloud_default_notes)
        self.assertEqual(len(notes), 2, notes)
        self.assertIn("off iCloud", notes[0])
        self.assertIn(os.path.join("~", "Documents", "Leonida"), notes[1])
        self.assertIn(os.path.join("~", "Leonida") + " to resume", notes[1])
        self.assertIn("--out", notes[1])
        self.assertEqual(len(self.default_out_in("darwin", True, call=make_game.icloud_default_notes)), 1)

    def test_default_out_matches_documents_folder(self):
        for platform in ("win32", "linux", "darwin"):
            self.assertEqual(self.default_out_in(platform, False), os.path.join("Documents", "Leonida"), platform)

    def test_default_out_leaves_icloud_synced_documents_on_a_mac(self):
        # Before: with iCloud Desktop & Documents sync on, ~/Documents/Leonida (and its work folder, hundreds of GB
        # made from the game) went into iCloud Drive, which uploads it and can evict the work files from the disk.
        self.assertEqual(self.default_out_in("darwin", True), "Leonida")
        # Only macOS keeps Documents in iCloud Drive this way; elsewhere the folder means nothing.
        self.assertEqual(self.default_out_in("linux", True), os.path.join("Documents", "Leonida"))

    def test_constants_definitions(self):
        self.assertEqual(make_game.STAGES, ["check", "shaders", "assets", "runtime", "pipelines"])
        self.assertEqual(make_game.PROBES, ("MicrosoftGame.Config", "Catalog000.bin", "appdata0.rpf"))
        self.assertEqual(make_game.MIN_FREE_GB, {"shaders": 30, "assets": 300})

    def test_below_normal_priority_does_not_crash(self):
        make_game.below_normal()


class PreflightExportedAssetsSpaceTest(PreflightTestBase):
    """A rerun over finished asset exports doesn't need room for a second copy of them."""

    OGG = "awc_opus.py finished\n"
    XMA = "rpf9dump xma finished\n"

    def run_preflight(self, work, free_gb, wav=False, ffmpeg=True):
        usage = Usage(1000 * 2**30, (1000 - free_gb) * 2**30, free_gb * 2**30)

        def which(cmd):
            return None if cmd == "ffmpeg" and not ffmpeg else "/usr/bin/tool"

        # The other prerequisites are faked, so the result doesn't depend on what this machine has installed (a .NET
        # SDK, ffmpeg) and preflight always reaches the space check.
        with mock.patch("sys.version_info", (3, 12, 0)), \
             mock.patch("shutil.which", side_effect=which), \
             mock.patch("make_game.tool_available", return_value=True), \
             mock.patch("make_game.dotnet_sdk_ok", return_value=True), \
             mock.patch("make_game.ensure_submodule", return_value=None), \
             mock.patch("sys.stdout", io.StringIO()), \
             mock.patch("shutil.disk_usage", return_value=usage):
            make_game.preflight(["shaders", "assets"], work, wav=wav)

    def write_xma(self, work, decoded):
        """One XMA2 stream as the assets stage leaves it, with or without the .wav that ffmpeg decodes it to."""
        bank = os.path.join(work, "assets", "audio", "xma", "bank0")
        os.makedirs(bank, exist_ok=True)
        for name in ("123.xma.wav",) + (("123.wav",) if decoded else ()):
            with open(os.path.join(bank, name), "wb") as f:
                f.write(b"RIFF")

    def write_markers(self, work, ogg=OGG, xma=XMA):
        audio = os.path.join(work, "assets", "audio")
        os.makedirs(audio, exist_ok=True)
        for name, text in (("ogg.done", ogg), ("xma.done", xma)):
            if text is not None:
                with open(os.path.join(audio, name), "w", encoding="utf-8") as f:
                    f.write(text)

    def test_markers_written_by_assets_are_read_back(self):
        with tempfile.TemporaryDirectory() as work:
            audio = os.path.join(work, "assets", "audio")
            os.makedirs(audio)
            make_game.mark(os.path.join(audio, "ogg.done"), "awc_opus.py")
            make_game.mark(os.path.join(audio, "xma.done"), "rpf9dump xma")
            self.assertTrue(make_game.assets_exported(work))

    def test_exported_assets_need_only_the_shader_space(self):
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work)
            self.run_preflight(work, free_gb=220)  # 30 GB for the shaders, not 330
            err = self.assert_fails(self.run_preflight, work, free_gb=20, match="about 30 GB is needed")
            self.assertIn("20 GB free", err)

    def test_fresh_work_folder_still_needs_the_asset_space(self):
        with tempfile.TemporaryDirectory() as work:
            self.assertFalse(make_game.assets_exported(work))
            self.assert_fails(self.run_preflight, work, free_gb=220, match="about 330 GB is needed")

    def test_stopped_export_still_needs_the_asset_space(self):
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work, xma=None)  # stopped before the XMA2 step
            self.assertFalse(make_game.assets_exported(work))
            self.assert_fails(self.run_preflight, work, free_gb=220, match="about 330 GB is needed")

    def test_marker_with_other_contents_does_not_count(self):
        for ogg, xma in (("", self.XMA), (self.OGG, "something else\n"), (self.XMA, self.OGG)):
            with self.subTest(ogg=ogg, xma=xma), tempfile.TemporaryDirectory() as work:
                self.write_markers(work, ogg=ogg, xma=xma)
                self.assertFalse(make_game.assets_exported(work))

    def test_marker_that_is_a_folder_does_not_count(self):
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work, xma=None)
            os.makedirs(os.path.join(work, "assets", "audio", "xma.done"))
            self.assertFalse(make_game.assets_exported(work))

    def test_xma_streams_ffmpeg_will_decode_still_need_the_asset_space(self):
        # xma.done is written without ffmpeg too; once ffmpeg is on PATH every run decodes the streams with no .wav.
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work)
            self.write_xma(work, decoded=False)
            self.assert_fails(self.run_preflight, work, free_gb=220, match="about 330 GB is needed")

    def test_xma_streams_left_undecoded_without_ffmpeg_are_not_counted(self):
        # Without ffmpeg nothing decodes them, so the assets stage writes nothing more.
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work)
            self.write_xma(work, decoded=False)
            self.run_preflight(work, free_gb=220, ffmpeg=False)

    def test_decoded_xma_streams_need_only_the_shader_space(self):
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work)
            self.write_xma(work, decoded=True)
            self.run_preflight(work, free_gb=220)

    def test_pending_xma_lists_only_streams_without_a_wav(self):
        with tempfile.TemporaryDirectory() as work:
            xma = os.path.join(work, "assets", "audio", "xma")
            self.assertEqual(make_game.pending_xma(xma), [])  # no folder yet
            self.write_xma(work, decoded=True)
            self.assertEqual(make_game.pending_xma(xma), [])
            os.remove(os.path.join(xma, "bank0", "123.wav"))
            self.assertEqual(make_game.pending_xma(xma), [os.path.join(xma, "bank0", "123.xma.wav")])

    def test_wav_still_counts_the_asset_space(self):
        # The WAV decode writes several times the Ogg Opus size and has no marker of its own.
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work)
            self.assert_fails(self.run_preflight, work, free_gb=220, wav=True, match="about 330 GB is needed")


if __name__ == "__main__":
    unittest.main()
