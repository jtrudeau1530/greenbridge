"""Synthetic-header tests for tools/story_index.py.

    python -I tests/test_story_index.py

Every byte here is built by the test. No game file or game asset name is used.
"""
import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "story_index.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import story_index  # noqa: E402


def psin():
    data = bytearray(0x200)
    data[:4] = b"PSIN"
    tags = [b"LSCH", b"LENM", b"LSNS", b"LGID", b"LRSC", b"CHKS"]
    sizes = [0x40, 0x30, 0x28, 0x10, 0x18, 0x14]
    pos = 0x40
    for tag, size in zip(tags, sizes):
        data[pos:pos + 4] = tag
        data[pos + 4:pos + 8] = size.to_bytes(4, "big")
        pos += size
    return bytes(data)


def page():
    data = bytearray(0x60)
    struct.pack_into("<Q", data, 8, 0x50100420)
    data[16:20] = b"0DRP"
    struct.pack_into("<I", data, 0x14, 122)
    struct.pack_into("<Q", data, 0x20, 3)
    return bytes(data)


def rsc8():
    data = bytearray(0x40)
    data[:4] = b"RSC8"
    struct.pack_into("<I", data, 4, 8001)
    struct.pack_into("<I", data, 8, 0x1234)
    struct.pack_into("<I", data, 12, 1)
    return bytes(data)


def nested():
    data = bytearray(0x40)
    data[:4] = b"9FPR"
    struct.pack_into("<I", data, 4, 0x8A0)
    struct.pack_into("<I", data, 8, 0x810)
    struct.pack_into("<I", data, 12, 0x140)
    return bytes(data)


def rbf0():
    data = b"RBF0" + struct.pack("<H", 0xFFFB) + struct.pack("<I", 1) + struct.pack("<H", 21)
    data += b"test_schema\x00"
    return data + bytes(64 - len(data))


def chunks():
    return struct.pack("<III", 0x88560988, 8, 7) + bytes(52)


def clip():
    return struct.pack("<I", 3) + b"\x01\x01\x00\x00" + bytes(56)


def xml():
    data = b"\xef\xbb\xbf<?xml version=\"1.0\"?><X/>"
    return data + bytes(max(0, 64 - len(data)))


def raw():
    return b"\x11\x22\x33\x44" + bytes(60)


def zero():
    return bytes(64)


def text():
    return b"A" * 80


def bink():
    data = bytearray(0x40)
    data[:4] = b"KB2n"
    struct.pack_into("<I", data, 4, len(data) - 8)
    struct.pack_into("<I", data, 8, 1)
    struct.pack_into("<I", data, 0x14, 64)
    struct.pack_into("<I", data, 0x18, 64)
    return bytes(data)


PAYLOADS = [psin(), page(), rsc8(), nested(), rbf0(), chunks(), clip(), xml(), raw(), zero(), text(), bink()]
EXPECTED = ["psin", "page-directory", "rsc8", "rpf9-nested", "rbf0", "typehash", "clip-stream",
            "xml", "raw:11223344", "zero-prefix", "text", "bink2"]


class ClassifyTest(unittest.TestCase):
    def test_signatures(self):
        got = [story_index.classify(p, 0, len(p))["signature"] for p in PAYLOADS]
        self.assertEqual(got, EXPECTED)

    def test_psin_section_chain(self):
        chain = story_index.psin_sections(psin())
        self.assertEqual([t for t, _, _ in chain], ["LSCH", "LENM", "LSNS", "LGID", "LRSC", "CHKS"])
        self.assertEqual([o for _, o, _ in chain], [0x40, 0x80, 0xB0, 0xD8, 0xE8, 0x100])
        self.assertEqual([s for _, _, s in chain], [0x40, 0x30, 0x28, 0x10, 0x18, 0x14])
        r = story_index.classify(psin(), 0x100, 90)
        self.assertEqual((r["signature"], r["count"]), ("psin", 6))
        self.assertEqual(r["detail"], "sections=LSCH,LENM,LSNS,LGID,LRSC,CHKS")

    def test_psin_header_offset(self):
        data = bytearray(psin())
        data[4:8] = (0x80).to_bytes(4, "big")
        chain = story_index.psin_sections(bytes(data))
        self.assertEqual([t for t, _, _ in chain], ["LENM", "LSNS", "LGID", "LRSC", "CHKS"])
        self.assertEqual(chain[0][1], 0x80)

    def test_page_header(self):
        r = story_index.classify(page(), 0x200, 40)
        self.assertEqual((r["version"], r["count"], r["detail"]), (122, 3, "0DRP at +16"))

    def test_rsc8_header(self):
        r = story_index.classify(rsc8(), 0x300, 50)
        self.assertEqual((r["signature"], r["version"]), ("rsc8", 8001))
        self.assertEqual(r["detail"], "declared_size=4660 flags=1")

    def test_nested_rpf9_header(self):
        r = story_index.classify(nested(), 0x400, 60)
        self.assertEqual(r["signature"], "rpf9-nested")
        self.assertEqual(r["detail"], "data_off=2208 toc=2064")

    def test_xml(self):
        self.assertEqual(story_index.classify(xml(), 0x500, 70)["signature"], "xml")


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.rpf = os.path.join(self.root, "synthetic.rpf")
        header = bytearray(0x100)
        header[:4] = b"9FPR"
        struct.pack_into("<I", header, 4, len(header))
        body = bytearray()
        self.offsets = []
        for payload in PAYLOADS:
            self.offsets.append(len(header) + len(body))
            body += payload if payload[:3] == b"KB2" else zlib.compress(payload)
            body += bytes((-len(body)) % 16)
        body += bytes(64)
        with open(self.rpf, "wb") as f:
            f.write(header + body)
        self.span = len(body)

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_offsets_and_summary(self):
        rows, summary = story_index.scan_rpf(self.rpf)
        self.assertEqual([r["offset"] for r in rows], self.offsets)
        self.assertEqual([r["signature"] for r in rows], EXPECTED)
        self.assertEqual(summary["files"], len(PAYLOADS))
        self.assertEqual(summary["bink2_files"], 1)
        self.assertEqual(summary["unread_bytes"], 0)
        self.assertEqual(summary["span"], self.span)
        self.assertEqual(summary["inflated_bytes"], sum(len(p) for p in PAYLOADS))
        # A stored length must stop at the zlib end, not the end of the read block.
        zlibs = [r for r in rows if r["signature"] != "bink2"]
        self.assertEqual(sum(r["stored"] for r in zlibs), sum(len(zlib.compress(p)) for p in PAYLOADS[:-1]))
        self.assertEqual(rows[-1]["stored"], 0x40)
        self.assertGreater(summary["readable_fraction"], 0.5)

    def test_scan_limit(self):
        limit = self.offsets[1] - 0x100 - 1
        rows, summary = story_index.scan_rpf(self.rpf, limit=limit)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["signature"], "psin")
        self.assertEqual(summary["span"], limit)

    def test_csv_and_json(self):
        rows, summary = story_index.scan_rpf(self.rpf)
        csv_path = os.path.join(self.root, "out.csv")
        json_path = os.path.join(self.root, "out.json")
        story_index.write_output(csv_path, rows, summary)
        story_index.write_output(json_path, rows, summary)
        with open(csv_path, newline="", encoding="utf-8") as f:
            table = list(csv.reader(f))
        self.assertEqual(table[0], story_index.COLUMNS)
        self.assertEqual(len(table), len(PAYLOADS) + 1)
        self.assertEqual(table[1][3], "psin")
        self.assertEqual(table[1][5], "6")
        with open(json_path, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["files"], len(PAYLOADS))
        self.assertEqual(len(doc["files"]), len(PAYLOADS))

    def test_cli(self):
        out = os.path.join(self.root, "cli.json")
        p = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--quiet"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["signatures"]["psin"]["files"], 1)
        self.assertEqual(doc["summary"]["signatures"]["page-directory"]["files"], 1)
        self.assertEqual(doc["summary"]["signatures"]["rsc8"]["files"], 1)

    def test_directory(self):
        folder = os.path.join(self.root, "carved")
        os.makedirs(folder)
        for i, payload in enumerate((psin(), page(), rsc8())):
            with open(os.path.join(folder, f"{0x1000 + i * 0x100:012x}.bin"), "wb") as f:
                f.write(payload)
        rows, summary = story_index.scan_dir(folder)
        self.assertEqual([r["offset"] for r in rows], [0x1000, 0x1100, 0x1200])
        self.assertEqual([r["signature"] for r in rows], ["psin", "page-directory", "rsc8"])
        self.assertEqual(summary["mode"], "directory")


if __name__ == "__main__":
    unittest.main()
