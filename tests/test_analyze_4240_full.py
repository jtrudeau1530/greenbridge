"""analyze_4240_full: a rejected 42 40 00 00 block is counted and stepped over, not parsed for ever.

    python -I tests/test_analyze_4240_full.py

Everything is synthetic: an RPF9-like header, zlib streams and hand-built block headers. No game data is used.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "analyze_4240_full.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import analyze_4240_full as A  # noqa: E402

BLOCK_6330 = struct.pack("<I", 0x3063) + bytes(0x60 - 4)


def block_4240(words):
    """A 0x60-byte 42 40 00 00 block: word 0 is the magic, words 1..22 are slot words."""
    header = bytearray(0x60)
    struct.pack_into("<24I", header, 0, *words)
    return bytes(header)


def make_archive(path, *bodies):
    """RPF9-like header (stream start word at 4) followed by one zlib stream per body."""
    header = bytearray(0x780)
    header[0:4] = b"9FPR"
    struct.pack_into("<I", header, 4, 0x780)
    with open(path, "wb") as f:
        f.write(bytes(header) + b"".join(zlib.compress(b, 9) for b in bodies) + bytes(64))


class Analyze4240Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args, timeout=30):
        return subprocess.run([sys.executable, "-I", TOOL, *args], cwd=ROOT,
                              capture_output=True, text=True, timeout=timeout)

    def truncated_archive(self):
        path = os.path.join(self.root, "truncated.rpf")
        # A 63 30 block the walker steps over, then the 42 40 magic with only 8 bytes behind it.
        make_archive(path, BLOCK_6330 + struct.pack("<I", 0x4042) + bytes(8))
        return path

    def test_a_truncated_block_returns_instead_of_looping(self):
        # Before: the except branch did not advance pos, so the same word was parsed and rejected for ever
        # (measured: still running after 12 s under timeout).
        path = self.truncated_archive()
        p = self.run_tool(path, timeout=30)
        self.assertEqual(p.returncode, 0, p.stderr)
        stats = A.process_archive(path)
        self.assertEqual(stats["blocks_4240"], 0)
        self.assertEqual(stats["errors"], 1, "a rejected block must be counted")
        self.assertIn("first_error", stats)

    def test_a_valid_block_is_measured(self):
        path = os.path.join(self.root, "valid.rpf")
        slots = [0x4042, (0x30 << 8) | 0x01, (0x40 << 8) | 0x01] + [0] * 21
        make_archive(path, BLOCK_6330 + block_4240(slots))
        stats = A.process_archive(path)
        self.assertEqual(stats["errors"], 0)
        self.assertEqual(stats["blocks_6330"], 1)
        self.assertEqual(stats["blocks_4240"], 1)
        self.assertEqual(stats["slot_tag_counts"], {(1, 1): 1, (0, 0x42): 1, (2, 1): 1})
        # Slot 1 sits at 0x60 and slot 0 at 0x80; slot 2 shares 0x80, so it adds no stride.
        self.assertEqual(stats["strides"], {(1, 1, 0, 0x42, 0x20): 1})

    def test_missing_input_is_one_error_line(self):
        missing = os.path.join(self.root, "missing.rpf")
        p = self.run_tool(missing)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("does not exist", p.stderr)
        self.assertIn("missing.rpf", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertNotIn("Processing", p.stdout)

    def test_limit_below_one_is_refused(self):
        path = self.truncated_archive()
        for bad in ("0", "-1"):
            with self.subTest(limit=bad):
                p = self.run_tool(path, "--limit", bad)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertIn("at least 1", p.stderr)
                self.assertNotIn("Traceback", p.stderr)

    def test_json_in_a_missing_folder_is_created(self):
        path = self.truncated_archive()
        out = os.path.join(self.root, "no", "such", "stats.json")
        p = self.run_tool(path, "--json", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(len(doc), 1)
        self.assertEqual(doc[0]["errors"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
