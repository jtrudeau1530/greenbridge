"""rdna2spirv.py command-line validation and SPIR-V output tests.

    python -I tests/test_rdna2spirv_cli.py

Standard library only: the output tests substitute the SPIR-V tools and use synthetic input.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

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


class ValidatedOutput(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.src = root / "disasm" / "ps_fixture.s"
        self.src.parent.mkdir()
        self.src.write_text("\ts_endpgm\n", encoding="utf-8")
        (root / "ps_fixture.dxbc").write_bytes(b"DXBC" + bytes(28))
        self.out = root / "spirv"
        self.out.mkdir()
        self.module = self.out / "ps_fixture.spv"
        self.index = self.out / "index.json"

    def translate(self, *, reject_at=None):
        def spirv_tool(cmd, **kwargs):
            tool = cmd[0]
            if tool == "spirv-as":
                Path(cmd[3]).write_bytes(b"new module")
            elif tool == "spirv-val":
                self.assertEqual(Path(cmd[-1]).read_bytes(), b"new module")
                if reject_at is None:
                    self.assertFalse(self.module.exists(), "the binary was published before validation")
            else:
                self.fail(f"unexpected tool: {tool}")
            rejected = tool == reject_at
            return subprocess.CompletedProcess(cmd, int(rejected), "", "synthetic rejection" if rejected else "")

        output = io.StringIO()
        with mock.patch.object(sys, "argv", [TOOL, str(self.src), "--out", str(self.out)]), \
                mock.patch.object(rdna2spirv.subprocess, "run", side_effect=spirv_tool), redirect_stdout(output):
            rdna2spirv.main()
        return output.getvalue()

    def test_validator_rejection_does_not_publish_binary(self):
        self.assertIn("INVALID: synthetic rejection", self.translate(reject_at="spirv-val"))
        self.assertFalse(self.module.exists())
        self.assertEqual(json.loads(self.index.read_text(encoding="utf-8")), {})
        self.assertTrue((self.out / "ps_fixture.spvasm").is_file())
        self.assertEqual(list(self.out.glob("*.tmp")), [])

    def test_rejection_clears_retried_shader_but_preserves_unrelated_output(self):
        other = self.out / "cs_other.spv"
        other.write_bytes(b"unrelated module")
        previous = {"cs_other": {"stage": "cs"}}
        for tool in ("spirv-as", "spirv-val"):
            with self.subTest(tool=tool):
                self.module.write_bytes(b"old module")
                self.index.write_text(json.dumps({**previous, "ps_fixture": {"stage": "ps"}}), encoding="utf-8")
                self.assertIn("INVALID: synthetic rejection", self.translate(reject_at=tool))
                self.assertFalse(self.module.exists())
                self.assertEqual(json.loads(self.index.read_text(encoding="utf-8")), previous)
                self.assertEqual(other.read_bytes(), b"unrelated module")

    def test_validation_success_publishes_binary_and_metadata(self):
        self.assertIn("1 pass spirv-val", self.translate())
        self.assertEqual(self.module.read_bytes(), b"new module")
        self.assertIn("ps_fixture", json.loads(self.index.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
