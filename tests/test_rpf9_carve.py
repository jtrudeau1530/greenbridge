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


def carve(test, data):
    """Run rpf9_carve.py on `data`; return its index.json after checking it exited cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        arc = os.path.join(tmp, "x.rpf")
        out = os.path.join(tmp, "out")
        with open(arc, "wb") as f:
            f.write(data)
        p = subprocess.run([sys.executable, "-I", TOOL, arc, out, "--index-only"], capture_output=True, text=True)
        test.assertEqual(p.returncode, 0, p.stderr)
        with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
            return json.load(f)


class CarveEdgeTest(unittest.TestCase):
    def test_bink_signature_cut_off_by_end_of_file(self):
        # A truncated copy can end inside what looks like a Bink 2 header: its fields lie past the end of the file.
        a = zlib.compress(b"A" * 1000)
        data = bytearray(0x40) + a + b"KB2a" + bytes(8)
        data[0:4] = b"RPF9"
        struct.pack_into("<I", data, 4, 0x40)
        doc = carve(self, bytes(data))
        self.assertEqual(doc["summary"]["files"], 1, doc["summary"])
        self.assertEqual(doc["summary"]["unread_bytes"], 12)

    def test_empty_data_area(self):
        data = bytearray(0x40)
        data[0:4] = b"RPF9"
        struct.pack_into("<I", data, 4, 0x40)        # data starts at the end of the file
        doc = carve(self, bytes(data))
        self.assertEqual(doc["summary"]["files"], 0)
        self.assertEqual(doc["summary"]["span"], 0)

    def test_bad_range_arguments_fail_before_the_out_dir_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "x.rpf")
            make_archive(arc)
            for flag, value, message in (("--start", "-1", "must be 0 or more"),
                                         ("--limit", "0", "must be at least 1"),
                                         ("--limit", "-1", "must be at least 1")):
                out = os.path.join(tmp, "out")
                p = subprocess.run([sys.executable, "-I", TOOL, arc, out, "--index-only", flag, value],
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertIn(message, p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                self.assertFalse(os.path.exists(out))

    def test_hex_limit_is_still_accepted(self):
        # The new converter keeps the repo's decimal-or-0x convention for byte counts.
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "x.rpf")
            out = os.path.join(tmp, "out")
            make_archive(arc)
            p = subprocess.run([sys.executable, "-I", TOOL, arc, out, "--index-only", "--limit", "0x40"],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["summary"]["span"], 0x40)


if __name__ == "__main__":
    unittest.main()
