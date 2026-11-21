"""Hand-built headings for tools/bvh_index.py. No game data is read.

    python -I tests/test_bvh_index.py
"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import bvh_index  # noqa: E402


def bvh_payload(n=10, tag=bvh_index.TAG, end_delta=0):
    b = bytearray(0x140 + n * 64)
    struct.pack_into("<I", b, 0x100, tag)
    struct.pack_into("<I", b, 0x104, n)
    end = 0x140 + n * 64 + end_delta
    struct.pack_into("<I", b, 0x20, end)
    struct.pack_into("<I", b, 0x28, end - 0x100)
    struct.pack_into("<6f", b, 0x110, -1.5, -2.0, -3.0, 4.5, 5.0, 6.0)
    struct.pack_into("<I", b, 0x130, 1234)
    struct.pack_into("<I", b, 0x134, 567)
    struct.pack_into("<i", b, 0x138, -89)
    b[0x140:0x180] = b"\x81" * 64
    return bytes(b)


def instance_payload():
    b = bytearray(208)
    for off, val in bvh_index.INSTANCE_WORDS:
        struct.pack_into("<Q", b, off, val)
    struct.pack_into("<Q", b, 0x40, 0x0123456789ABCDEF)
    struct.pack_into("<12f", b, 0x50, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    struct.pack_into("<I", b, 0xA0, 0x1234)
    struct.pack_into("<I", b, 0xA4, 0x00200010)
    return bytes(b)


def xml_payload():
    return b"\xef\xbb\xbf<?xml version=\"1.0\"?><SSetupData><deviceName>x</deviceName></SSetupData>"


def make_rpf(payloads):
    """An RPF9-shaped buffer: 4-byte magic, u32 data start at 4, then zlib streams separated by zeros."""
    body = b""
    for p in payloads:
        body += zlib.compress(p, 9) + b"\0" * 16
    head = bytearray(0x40)
    head[0:4] = b"9FPR"
    struct.pack_into("<I", head, 4, len(head))
    return bytes(head) + body


class ClassifyTest(unittest.TestCase):
    def test_bvh4(self):
        row = bvh_index.classify(bvh_payload(12), 0x140 + 12 * 64)
        self.assertEqual(row["kind"], "bvh4")
        self.assertEqual(row["node_count"], 12)
        self.assertEqual(row["root_offset"], 0x140 + 12 * 64 - 0x100)
        self.assertEqual(row["bounds_min"], [-1.5, -2.0, -3.0])
        self.assertEqual(row["bounds_max"], [4.5, 5.0, 6.0])
        self.assertEqual(row["count_130"], 1234)

    def test_bvh4_chunk(self):
        b = bytearray(65536)
        struct.pack_into("<I", b, 0x100, bvh_index.TAG)
        struct.pack_into("<I", b, 0x104, 1200)
        end = 0x140 + 1200 * 64
        struct.pack_into("<I", b, 0x20, end)
        struct.pack_into("<I", b, 0x28, end - 0x100)
        row = bvh_index.classify(bytes(b[:0x140]), 65536)
        self.assertEqual(row["kind"], "bvh4-chunk")
        self.assertTrue(row["chunked"])
        self.assertEqual(row["declared_bytes"], end)

    def test_bvh4_requires_the_size_formula(self):
        self.assertEqual(bvh_index.classify(bvh_payload(12, end_delta=64), 0x140 + 12 * 64 + 64)["kind"], "raw")
        self.assertEqual(bvh_index.classify(bvh_payload(12, tag=0x81020007), 0x140 + 12 * 64)["kind"], "raw")

    def test_instance208(self):
        row = bvh_index.classify(instance_payload(), 208)
        self.assertEqual(row["kind"], "instance208")
        self.assertEqual(row["hash"], "0123456789abcdef")
        self.assertEqual((row["hi16"], row["lo16"]), (0x20, 0x10))
        self.assertEqual(row["off_a0"], 0x1234)

    def test_descriptor(self):
        b = bytearray(160)
        struct.pack_into("<I", b, 8, 0x50000060)
        struct.pack_into("<I", b, 0x20, 0x50000090)
        self.assertEqual(bvh_index.classify(bytes(b), 160)["kind"], "descriptor")

    def test_xml(self):
        row = bvh_index.classify(xml_payload(), len(xml_payload()))
        self.assertEqual(row["kind"], "xml")
        self.assertFalse(row["fragment"])
        self.assertEqual(row["root_element"], "SSetupData")

    def test_xml_fragment(self):
        text = b"/americas/map/x_bvh.rpf</filename><Item/>" + b" " * 40
        row = bvh_index.classify(text, len(text))
        self.assertEqual(row["kind"], "xml")
        self.assertTrue(row["fragment"])


class ScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_streams_are_both_found(self):
        # The old carver advanced by the 1 MiB read size, so a small first stream hid the second one.
        payloads = [b"a" * 5000, bvh_payload(5)]
        path = os.path.join(self.dir, "two.rpf")
        with open(path, "wb") as f:
            f.write(make_rpf(payloads))
        rows, summary = bvh_index.scan_rpf(path)
        self.assertEqual(summary[0]["streams"], 2)
        self.assertEqual([r["kind"] for r in rows], ["raw", "bvh4"])
        self.assertEqual(rows[1]["node_count"], 5)
        self.assertEqual(summary[0]["inflated"], 5000 + 0x140 + 5 * 64)

    def test_index_outputs(self):
        path = os.path.join(self.dir, "pack.rpf")
        with open(path, "wb") as f:
            f.write(make_rpf([instance_payload(), bvh_payload(3), xml_payload()]))
        out = os.path.join(self.dir, "index.json")
        csv_path = os.path.join(self.dir, "index.csv")
        self.assertEqual(bvh_index.main([path, "--out", out, "--csv", csv_path]), 0)
        with open(out) as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["streams"], 3)
        self.assertEqual(doc["summary"]["kinds"], {"instance208": 1, "bvh4": 1, "xml": 1})
        self.assertNotIn("head", doc["streams"][0])
        with open(csv_path, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("signature", text.splitlines()[0])
        self.assertIn("bvh4:0x81020006", text)

    def test_directory_of_payloads(self):
        for name, payload in (("a.bin", bvh_payload(4)), ("b.bin", instance_payload()), ("c.bin", b"z" * 80)):
            with open(os.path.join(self.dir, name), "wb") as f:
                f.write(payload)
        rows, summary = bvh_index.scan_dir(self.dir)
        self.assertEqual({r["kind"] for r in rows}, {"bvh4", "instance208", "raw"})
        self.assertEqual(summary[0]["streams"], 3)

    def test_empty_and_short_archives_have_no_streams(self):
        for name, data in (("empty.rpf", b""), ("short.rpf", b"9FPR")):
            path = os.path.join(self.dir, name)
            with open(path, "wb") as f:
                f.write(data)
            rows, summary = bvh_index.scan_rpf(path)
            self.assertEqual(rows, [])
            self.assertEqual((summary[0]["streams"], summary[0]["span"]), (0, 0))

    def test_start_outside_the_archive_is_refused(self):
        path = os.path.join(self.dir, "one.rpf")
        with open(path, "wb") as f:
            f.write(make_rpf([bvh_payload(2)]))
        with self.assertRaises(ValueError):
            bvh_index.scan_rpf(path, start=1 << 20)

    def test_directory_keeps_good_archives_next_to_a_bad_one(self):
        with open(os.path.join(self.dir, "a_good.rpf"), "wb") as f:
            f.write(make_rpf([bvh_payload(2)]))
        with open(os.path.join(self.dir, "b_bad.rpf"), "wb") as f:
            f.write(b"9FPR" + struct.pack("<I", 1 << 20))   # data start past the end of the file
        rows, summaries = bvh_index.scan_dir(self.dir)
        self.assertEqual([r["kind"] for r in rows], ["bvh4"])
        self.assertEqual(len(summaries), 1)

    def test_cli_rejects_a_limit_below_one(self):
        path = os.path.join(self.dir, "one.rpf")
        with open(path, "wb") as f:
            f.write(make_rpf([bvh_payload(2)]))
        for value in ("0", "-100"):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                bvh_index.main([path, "--limit", value])
            self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
