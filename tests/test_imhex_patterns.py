"""Tests for the ImHex patterns in tools/imhex.

    python -I tests/test_imhex_patterns.py

Every byte here is built by the test: no game file, name or path is used. The RBF0 inputs are first checked with
tools/rbf0_index.py, the repository's own parser. When ImHex is installed, each pattern then runs on the same bytes
through ImHex's pattern-language CLI (`imhex --pl format ... -f json`) and the decoded fields are compared with the
parser's result. ImHex is found through LEONIDA_IMHEX, then PATH, then its default Windows install folder; without it
the ImHex half is skipped and the rest still runs.
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERNS = os.path.join(ROOT, "tools", "imhex")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rbf0_index  # noqa: E402


def find_imhex():
    candidates = [os.environ.get("LEONIDA_IMHEX"), shutil.which("imhex")]
    if os.name == "nt":
        candidates.append(os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "ImHex", "imhex.exe"))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


IMHEX = find_imhex()


# ---- synthetic inputs ------------------------------------------------------------------------------------------

def _name(name):
    return struct.pack("<H", len(name)) + name


def rbf0_stream(wide):
    """One resource with every record type of docs/rbf0.md section 3, then bytes past the root close."""
    q, d = ("Q", "d") if wide else ("I", "f")
    q, d, d3 = "<" + q, "<" + d, "<" + d * 3
    out = b"RBF0" + (rbf0_index.PROLOGUE if wide else b"")
    out += bytes([0, 0x00]) + _name(b"root") + struct.pack("<HHH", 0, 0, 1)
    out += bytes([1, 0x10]) + _name(b"count") + struct.pack(q, 7)                 # the root's one attribute
    out += bytes([2, 0x00]) + _name(b"item") + struct.pack("<HHH", 0, 0, 0)
    out += bytes([3, 0x40]) + _name(b"scale") + struct.pack(d, 1.5)
    out += bytes([4, 0x50]) + _name(b"pos") + struct.pack(d3, 1.0, -2.0, 0.25)
    out += bytes([5, 0x60]) + _name(b"label") + struct.pack("<H", 5) + b"probe"
    out += bytes([6, 0x20]) + _name(b"flag") + bytes([6, 0x30])                    # new, then reused
    out += bytes([3, 0x10]) + struct.pack(q, 9)                                   # reused with another type
    out += bytes([0xFD, 0xFF]) + struct.pack("<I", 5) + b"text\x00"
    out += b"\xff\xff"                                                            # close item
    out += b"\xff\xff"                                                            # close root: the end
    return out + b"\x00\x11\x22\x33"                                              # not part of the resource


def rpf9_archive(data_offset=0x100, size=0x140):
    head = b"9FPR" + struct.pack("<III", data_offset, 0x11, 0x22)
    toc = bytes((i * 37) & 0xFF for i in range(data_offset - len(head)))
    body = b"\x78\x9c" + bytes(size - data_offset - 2)
    return head + toc + body


def pragma_magic(pattern):
    """The bytes of `#pragma magic [ .. ] @ 0x00` in a pattern file."""
    with open(os.path.join(PATTERNS, pattern), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#pragma magic"):
                inside = line[line.index("[") + 1:line.index("]")]
                return bytes(int(b, 16) for b in inside.split())
    return None


# ---- tests -----------------------------------------------------------------------------------------------------

class InputsAndMagic(unittest.TestCase):
    """Runs everywhere: the synthetic inputs are valid RBF0 by the repository's parser, and the magics agree."""

    def test_synthetic_streams_parse(self):
        for wide in (True, False):
            r = rbf0_index.parse_rbf0(rbf0_stream(wide)[:-4])
            self.assertEqual(r["status"], "ok", r["error"])
            self.assertEqual(r["layout"], "wide-64" if wide else "narrow-32")
            self.assertEqual(r["elements"], 2)
            self.assertEqual(r["descriptors"], 7)

    def test_pattern_magics(self):
        self.assertEqual(pragma_magic("rbf0.hexpat"), rbf0_index.MAGIC)
        self.assertEqual(pragma_magic("rpf9.hexpat"), b"9FPR")  # RPF9 stored little-endian (docs/rpf.md)


@unittest.skipUnless(IMHEX, "ImHex is not installed (set LEONIDA_IMHEX to imhex.exe)")
class PatternsInImHex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_pattern(self, pattern, data):
        """Returns (decoded JSON or None, combined output)."""
        src = os.path.join(self.tmp.name, "input.bin")
        out = os.path.join(self.tmp.name, "out.json")
        if os.path.exists(out):
            os.remove(out)
        with open(src, "wb") as f:
            f.write(data)
        cmd = [IMHEX, "--pl", "format", "-i", src, "-p", os.path.join(PATTERNS, pattern), "-o", out, "-f", "json"]
        includes = os.path.join(os.path.dirname(IMHEX), "includes")
        if os.path.isdir(includes):
            cmd += ["-I", includes]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        log = p.stdout + p.stderr
        if not os.path.exists(out):
            return None, log
        with open(out, encoding="utf-8") as f:
            return json.load(f), log

    def check_rbf0(self, wide):
        data = rbf0_stream(wide)
        ref = rbf0_index.parse_rbf0(data[:-4])
        got, log = self.run_pattern("rbf0.hexpat", data)
        self.assertIsNotNone(got, log)
        self.assertEqual("prologue" in got["header"], wide)
        records = got["records"]

        names = [r["name"] for r in records if "name" in r]
        self.assertEqual(names, ref["descriptor_names"])
        self.assertEqual(sum(1 for r in records if r["index"] < 0xFD and r["type"] == 0x00), ref["elements"])
        ints = [r["value_int"] for r in records if "value_int" in r]
        self.assertEqual(ints, ref["ints"])
        floats = []
        for r in records:
            floats += [r["value_float"]] if "value_float" in r else r.get("value_vec3", [])
        self.assertEqual(floats, ref["floats"])
        strings = [r.get("value_string", r.get("text", "")) for r in records if "value_string" in r or "text" in r]
        self.assertEqual([s.split("%00")[0] for s in strings], ref["strings"])
        self.assertEqual([r["attribute_count"] for r in records if "attribute_count" in r], [1, 0])
        # The pattern stops where the root closes, as parse_rbf0 does: the 4 trailing bytes are not records.
        self.assertEqual(records[-1]["index"], 0xFF)
        self.assertEqual(sum(1 for r in records if r["index"] == 0xFF), 2)

    def test_rbf0_wide(self):
        self.check_rbf0(wide=True)

    def test_rbf0_narrow(self):
        self.check_rbf0(wide=False)

    def test_rpf9_header_and_opaque_toc(self):
        got, log = self.run_pattern("rpf9.hexpat", rpf9_archive(data_offset=0x100))
        self.assertIsNotNone(got, log)
        self.assertEqual(got["header"]["magic"], "9FPR")
        self.assertEqual(got["header"]["data_offset"], 0x100)
        self.assertEqual((got["header"]["u32_08"], got["header"]["u32_0c"]), (0x11, 0x22))
        self.assertEqual(len(got["header_rest_and_toc"]), 0x100 - 0x10)

    def test_rpf9_rejects_a_data_offset_past_the_end(self):
        got, log = self.run_pattern("rpf9.hexpat", rpf9_archive(data_offset=0x100, size=0x140)[:0x20])
        self.assertIsNone(got)
        self.assertIn("data offset is outside the file", log)


if __name__ == "__main__":
    unittest.main()
