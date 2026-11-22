"""LLVM_MC is one executable path or a whole command line, in every tool that reads it.

Windows runs llvm-mc through WSL (the official Windows LLVM build has no AMDGPU target):

    set LLVM_MC=wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc

    python -I tests/test_mc_command.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import mc_command  # noqa: E402


class SplitCommandTest(unittest.TestCase):
    def test_bare_name(self):
        self.assertEqual(mc_command.split_command("llvm-mc"), ["llvm-mc"])

    def test_empty(self):
        self.assertEqual(mc_command.split_command(""), [])
        self.assertEqual(mc_command.split_command("   "), [])

    def test_command_line(self):
        self.assertEqual(mc_command.split_command("wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc"),
                         ["wsl", "-d", "Ubuntu", "-e", "/usr/lib/llvm-21/bin/llvm-mc"])

    def test_quoted_arguments(self):
        if os.name == "nt":
            raw = '"C:\\Program Files\\LLVM\\bin\\llvm-mc.exe" --version'
            want = ["C:\\Program Files\\LLVM\\bin\\llvm-mc.exe", "--version"]
        else:
            raw = '"/opt/llvm 21/bin/llvm-mc" --version'
            want = ["/opt/llvm 21/bin/llvm-mc", "--version"]
        self.assertEqual(mc_command.split_command(raw), want)

    def test_path_with_spaces_is_kept_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "llvm mc")
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            self.assertEqual(mc_command.split_command(path), [path])
            self.assertEqual(mc_command.split_command(f'"{path}"'), [path])

    def test_path_with_spaces_and_no_extension_is_kept_whole(self):
        # C:\Program Files\LLVM\bin\llvm-mc names llvm-mc.exe on Windows. It is not a file under that exact
        # name, so it was split into C:\Program and Files\LLVM\bin\llvm-mc.
        raw = "C:\\Program Files\\LLVM\\bin\\llvm-mc"
        with mock.patch.object(mc_command.os, "name", "nt"), \
                mock.patch.object(mc_command.shutil, "which", return_value=raw + ".EXE") as which:
            self.assertEqual(mc_command.split_command(raw), [raw])
            which.assert_called_once_with(raw)

    @unittest.skipUnless(os.name == "nt", "Windows resolves a missing .exe")
    def test_real_path_with_spaces_and_no_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "llvm 21")
            os.mkdir(folder)
            with open(os.path.join(folder, "llvm-mc.exe"), "w", encoding="utf-8") as f:
                f.write("")
            path = os.path.join(folder, "llvm-mc")
            self.assertEqual(mc_command.split_command(path), [path])

    def test_command_line_is_still_split_when_its_first_word_is_on_path(self):
        # Only a value with a directory part is tried as one path: "wsl -d Ubuntu ..." stays a command line
        # whatever is on PATH, and a bare name is not looked up at all.
        with mock.patch.object(mc_command.shutil, "which", return_value="/usr/bin/x") as which:
            self.assertEqual(mc_command.split_command("wsl -d Ubuntu -e llvm-mc"),
                             ["wsl", "-d", "Ubuntu", "-e", "llvm-mc"])
            self.assertEqual(mc_command.split_command("llvm-mc"), ["llvm-mc"])
            which.assert_not_called()

    def test_missing_path_with_spaces_is_split_as_before(self):
        with mock.patch.object(mc_command.os, "name", "nt"), \
                mock.patch.object(mc_command.shutil, "which", return_value=None):
            self.assertEqual(mc_command.split_command('"C:\\no such\\llvm-mc" --version'),
                             ["C:\\no such\\llvm-mc", "--version"])

    def test_unbalanced_quote_is_one_clear_error(self):
        raw = '"unterminated --version'
        for platform in ("nt", "posix"):
            with self.subTest(platform=platform), mock.patch.object(mc_command.os, "name", platform):
                with self.assertRaises(SystemExit) as ctx:
                    mc_command.split_command(raw)
                self.assertIn("LLVM_MC", str(ctx.exception))
                self.assertIn("unterminated", str(ctx.exception))

    def test_mc_command_reads_the_environment(self):
        with mock.patch.dict(os.environ, {"LLVM_MC": "wsl -d Ubuntu -e /opt/llvm-21/bin/llvm-mc"}):
            self.assertEqual(mc_command.mc_command(),
                             ["wsl", "-d", "Ubuntu", "-e", "/opt/llvm-21/bin/llvm-mc"])

    def test_mc_command_falls_back_to_the_path_name(self):
        with mock.patch.dict(os.environ, {"LLVM_MC": ""}):
            self.assertEqual(mc_command.mc_command(), ["llvm-mc"])


if __name__ == "__main__":
    unittest.main()
