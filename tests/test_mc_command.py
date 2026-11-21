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

    def test_mc_command_reads_the_environment(self):
        with mock.patch.dict(os.environ, {"LLVM_MC": "wsl -d Ubuntu -e /opt/llvm-21/bin/llvm-mc"}):
            self.assertEqual(mc_command.mc_command(),
                             ["wsl", "-d", "Ubuntu", "-e", "/opt/llvm-21/bin/llvm-mc"])

    def test_mc_command_falls_back_to_the_path_name(self):
        with mock.patch.dict(os.environ, {"LLVM_MC": ""}):
            self.assertEqual(mc_command.mc_command(), ["llvm-mc"])


if __name__ == "__main__":
    unittest.main()
