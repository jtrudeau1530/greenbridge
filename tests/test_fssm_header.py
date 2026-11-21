"""The FSSM pixel-input decoder reads documented record fields, the directory identity and the fp16 mask.

All blobs here are built by hand; no shader from the game is used.

    python -I tests/test_fssm_header.py
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TOOL = os.path.join(TOOLS, "fssm_header.py")
sys.path.insert(0, TOOLS)

import fssm_header  # noqa: E402


def blob(records=(), directory=(0x100, 0x20, 0x120), size=0x200, first=None):
    """Build a synthetic FSSM header.  records are complete 8-byte input records."""
    data = bytearray(size)
    data[0:4] = b"FSSM"
    struct.pack_into("<I", data, 0x120, directory[0])
    struct.pack_into("<I", data, 0x12C, directory[1])
    struct.pack_into("<I", data, 0x130, directory[2])
    for i, record in enumerate(records):
        start = 0x138 + 8 * i
        data[start:start + 8] = bytes(record)
    if first is not None:
        data[0x138] = first
    return bytes(data)


def record(semantic, mask, index, slot, flags, kind=6):
    return (kind, semantic, mask, index, 0, 0, slot, flags)


class PixelInputTest(unittest.TestCase):
    def test_two_records_keep_their_fields(self):
        b = blob([record(4, 0x7, 1, 9, 0x12), record(7, 0x3, 2, 10, 0x08)])
        self.assertEqual(fssm_header.pixel_inputs(b), [
            {"semantic": 4, "mask": 0x7, "index": 1, "slot": 9, "flags": 0x12},
            {"semantic": 7, "mask": 0x3, "index": 2, "slot": 10, "flags": 0x08},
        ])

    def test_table_stops_at_a_non_six_kind(self):
        b = blob([record(1, 0xF, 0, 3, 0), record(2, 0xF, 1, 4, 0, kind=5)])
        self.assertEqual(len(fssm_header.pixel_inputs(b)), 1)

    def test_no_table_when_first_byte_is_not_six(self):
        b = blob([record(1, 0xF, 0, 3, 0)], first=5)
        self.assertEqual(fssm_header.pixel_inputs(b), [])
        self.assertEqual(fssm_header.interpolants(b), (0, 0))

    def test_short_blob_has_no_records(self):
        self.assertEqual(fssm_header.pixel_inputs(b""), [])
        self.assertEqual(fssm_header.pixel_inputs(b"\x06" * 0x139), [])

    def test_fp16_mask_uses_table_position(self):
        b = blob([record(1, 0xF, 0, 0, 0x10),
                  record(2, 0xF, 1, 1, 0x00),
                  record(3, 0x3, 2, 2, 0x10)])
        self.assertEqual(fssm_header.interpolants(b), (3, 0b101))

    def test_fp16_mask_is_empty_without_the_flag(self):
        b = blob([record(1, 0xF, 0, 0, 0x02), record(2, 0xF, 1, 1, 0x08)])
        self.assertEqual(fssm_header.interpolants(b), (2, 0))


class DirectoryTest(unittest.TestCase):
    def test_identity_holds(self):
        self.assertTrue(fssm_header.directory_ok(blob(directory=(0x100, 0x20, 0x120))))

    def test_identity_breaks(self):
        self.assertFalse(fssm_header.directory_ok(blob(directory=(0x100, 0x20, 0x121))))
        self.assertFalse(fssm_header.directory_ok(blob(directory=(0x100, 0x20, 0xDEADBEEF))))

    def test_short_blob_is_not_ok(self):
        self.assertFalse(fssm_header.directory_ok(b"FSSM"))


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        fssm_dir = os.path.join(self.dir, "fssm")
        os.makedirs(fssm_dir)
        with open(os.path.join(fssm_dir, "a.fssm"), "wb") as f:
            f.write(blob([record(1, 0xF, 0, 0, 0x10), record(2, 0x3, 1, 1, 0)]))
        with open(os.path.join(fssm_dir, "b.fssm"), "wb") as f:
            f.write(blob(first=0))
        with open(os.path.join(self.dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump([{"file": "a.dxbc", "stage": "ps"}, {"file": "b.dxbc", "stage": "ps"}], f)
        self.fssm_dir = fssm_dir

    def tearDown(self):
        self.tmp.cleanup()

    def test_text_tally(self):
        p = subprocess.run([sys.executable, "-I", TOOL, self.fssm_dir], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("2 files, directory_ok 2, PS with table 1, entries 2", p.stdout)
        self.assertIn("a.fssm: stage=ps directory_ok=True entries=2 fp16=0x1", p.stdout)

    def test_json_tally(self):
        p = subprocess.run([sys.executable, "-I", TOOL, self.fssm_dir, "--json"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        doc = json.loads(p.stdout)
        self.assertEqual(doc["tally"]["ps_with_table"], 1)
        self.assertEqual(doc["tally"]["entries"], 2)


if __name__ == "__main__":
    unittest.main()
