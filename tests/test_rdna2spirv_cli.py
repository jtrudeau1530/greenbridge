"""rdna2spirv.py command line: --limit and --shard are checked before anything is translated or written.

    python -I tests/test_rdna2spirv_cli.py

Standard library only: every case is refused by argparse, so no llvm-mc, spirv-as or shader input is needed. No
game data.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
TOOL = os.path.join(ROOT, "tools", "rdna2spirv.py")

import rdna2spirv  # noqa: E402


class ArgumentTypes(unittest.TestCase):
    def test_limit(self):
        self.assertEqual(rdna2spirv._limit_arg("200"), 200)
        self.assertEqual(rdna2spirv._limit_arg("1"), 1)
        for text in ("0", "-1", "-200", "x", ""):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                rdna2spirv._limit_arg(text)

    def test_shard(self):
        self.assertEqual(rdna2spirv._shard_arg("0/8"), (0, 8))
        self.assertEqual(rdna2spirv._shard_arg("7/8"), (7, 8))
        self.assertEqual(rdna2spirv._shard_arg("0/1"), (0, 1))
        for text in ("0/0", "8/8", "9/8", "-1/8", "1/-8", "a/b", "0", "0/8/1", "", "/8"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                rdna2spirv._shard_arg(text)


class CommandLine(unittest.TestCase):
    def test_bad_limit_or_shard_is_refused_before_out_is_made(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "disasm")
            os.mkdir(src)
            with open(os.path.join(src, "ps_x.s"), "w", encoding="utf-8") as f:
                f.write("\ts_endpgm\n")
            for args, message in ((["--limit", "0"], "must be at least 1"),
                                  (["--limit", "-5"], "must be at least 1"),
                                  (["--shard", "0/0"], "need n >= 1"),
                                  (["--shard", "9/8"], "need n >= 1 and 0 <= i < n"),
                                  (["--shard", "a/b"], "expected i/n")):
                out = os.path.join(tmp, "spirv")
                with self.subTest(args=args):
                    p = subprocess.run([sys.executable, "-I", TOOL, src, "--out", out, *args],
                                       capture_output=True, text=True)
                    self.assertEqual(p.returncode, 2)
                    self.assertIn(message, p.stderr)
                    self.assertNotIn("Traceback", p.stderr)
                    self.assertFalse(os.path.exists(out))     # 9/8 used to write an empty index.part9.json


if __name__ == "__main__":
    unittest.main()
