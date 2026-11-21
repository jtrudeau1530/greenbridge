"""Regression: rpf9_carve must not overrun a short final read and skip trailing streams."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "rpf9_carve.py")


def make_archive(path):
    a = zlib.compress(b"A" * 1000)
    b = zlib.compress(b"B" * 2000)
    data = bytearray(0x40 + len(a) + len(b) + 4096)
    data[0:4] = b"RPF9"
    struct.pack_into("<I", data, 4, 0x40)
    data[0x40:0x40 + len(a)] = a
    data[0x40 + len(a):0x40 + len(a) + len(b)] = b
    with open(path, "wb") as f:
        f.write(data)
    return a, b


class CarveTrailingStreamTest(unittest.TestCase):
    def test_streams_at_eof_are_all_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            out = os.path.join(tmp, "out")
            a, b = make_archive(arc)
            p = subprocess.run([sys.executable, "-I", TOOL, arc, out, "--index-only"],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["files"], 2, doc["summary"])
            self.assertEqual([r["size"] for r in doc["files"]], [1000, 2000])
            self.assertEqual(doc["files"][0]["stored"], len(a))
            self.assertEqual(doc["files"][1]["stored"], len(b))
            self.assertEqual(doc["files"][1]["off"], 0x40 + len(a))


if __name__ == "__main__":
    unittest.main()
