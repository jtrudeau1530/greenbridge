"""navmesh_index must find the measured 0DRP page directory in an RPF9 stream and in carved files.

    python -I tests/test_navmesh_index.py

Everything is synthetic: a hand-built RPF9 header, zlib streams and a fake page. No game data is used.
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
TOOL = os.path.join(ROOT, "tools", "navmesh_index.py")
KEY = 0x8A2D3FF3DF663960
FIELD68 = 141492


def page(version=122, pages=3, key=KEY, field68=FIELD68):
    """A 0x80-byte synthetic page matching the measured layout: 0DRP at 16, key at 0x60."""
    b = bytearray(0x80)
    struct.pack_into("<Q", b, 8, 0x50004000)       # pointer-like word, unused by the index
    b[16:20] = b"0DRP"
    struct.pack_into("<I", b, 20, version)
    struct.pack_into("<I", b, 32, pages)
    struct.pack_into("<Q", b, 0x60, key)
    struct.pack_into("<I", b, 0x68, field68)
    return bytes(b)


def make_rpf(path):
    """RPF9-like header (start word at 4) followed by three zlib streams."""
    header = bytearray(0x780)
    header[0:4] = b"9FPR"                           # bytes of little-endian 'RPF9'
    struct.pack_into("<I", header, 4, 0x780)
    streams = [zlib.compress(page(), 9),
               zlib.compress(b"not a page directory" * 100, 9),
               zlib.compress(b"0DRP" + bytes(124), 9)]   # marker at 0, must not match
    with open(path, "wb") as f:
        f.write(bytes(header) + b"".join(streams) + bytes(64))
    return streams


class NavmeshIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        r = subprocess.run([sys.executable, "-I", TOOL, *args], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r

    def test_rpf_stream(self):
        rpf = os.path.join(self.root, "test.rpf")
        streams = make_rpf(rpf)
        out, csv_path = os.path.join(self.root, "index.json"), os.path.join(self.root, "index.csv")
        self.run_tool(rpf, "--out", out, "--csv", csv_path)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        s = doc["summary"]
        self.assertEqual(s["mode"], "rpf")
        self.assertEqual(s["streams"], 3)
        self.assertEqual(s["matched"], 1)
        self.assertEqual(s["versions"], {"122": 1})
        self.assertEqual(s["page_counts"], {"3": 1})
        self.assertEqual(s["keys_present"], 1)
        self.assertEqual(s["compressed_bytes"], sum(len(z) for z in streams))
        row = doc["pages"][0]
        self.assertEqual(row["offset"], 0x780)
        self.assertEqual(row["stored"], len(streams[0]))
        self.assertEqual(row["size"], 0x80)
        self.assertEqual(row["signature"], "0DRP@16")
        self.assertEqual(row["version"], 122)
        self.assertEqual(row["pages"], 3)
        self.assertEqual(row["key64"], KEY.to_bytes(8, "little").hex())
        self.assertEqual(row["u32_68"], FIELD68)
        with open(csv_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(lines[0], "source,offset,stored,size,signature,version,pages,key64,u32_68")
        self.assertEqual(len(lines), 2)
        self.assertIn("0DRP@16", lines[1])

    def test_directory(self):
        folder = os.path.join(self.root, "carve")
        os.makedirs(folder)
        with open(os.path.join(folder, "page.bin"), "wb") as f:
            f.write(page(pages=2))
        with open(os.path.join(folder, "other.bin"), "wb") as f:
            f.write(b"plain" * 100)
        with open(os.path.join(folder, "ignored.txt"), "wb") as f:
            f.write(page())
        out = os.path.join(self.root, "dir.json")
        self.run_tool(folder, "--out", out, "--pattern", "*.bin")
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["streams"], 2)
        self.assertEqual(doc["summary"]["matched"], 1)
        self.assertEqual(doc["summary"]["page_counts"], {"2": 1})
        row = doc["pages"][0]
        self.assertEqual(row["source"], "page.bin")
        self.assertIsNone(row["offset"])
        self.assertIsNone(row["stored"])
        self.assertEqual(row["size"], 0x80)

    def test_not_an_rpf(self):
        path = os.path.join(self.root, "not.rpf")
        with open(path, "wb") as f:
            f.write(b"nope" * 64)
        r = subprocess.run([sys.executable, "-I", TOOL, path, "--out", os.path.join(self.root, "x.json")],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not an RPF7/8/9 archive", r.stderr)


if __name__ == "__main__":
    unittest.main()
