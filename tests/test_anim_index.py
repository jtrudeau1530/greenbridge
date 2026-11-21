"""Synthetic-header tests for tools/anim_index.py.

    python -I tests/test_anim_index.py

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
TOOL = os.path.join(ROOT, "tools", "anim_index.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import anim_index  # noqa: E402


def move():
    data = bytearray(72)
    data[:4] = b"MoVE"
    struct.pack_into("<I", data, 4, 19)
    struct.pack_into("<I", data, 0x14, 852)
    return bytes(data)


def page():
    data = bytearray(0x60)
    struct.pack_into("<Q", data, 8, 0x50100420)
    data[16:20] = b"0DRP"
    struct.pack_into("<I", data, 0x14, 122)
    struct.pack_into("<Q", data, 0x20, 3)
    return bytes(data)


def names():
    data = b"pack:/alpha.clip\x00pack:/bravo.anim\x00"
    return data + bytes(64 - len(data))


def skeleton():
    data = b"SKEL_Spine\x00SKEL_Neck\x00"
    return data + bytes(64 - len(data))


def chunks():
    return struct.pack("<III", 0x88560988, 8, 7) + bytes(52)


def clip():
    return struct.pack("<I", 3) + b"\x01\x01\x00\x00" + bytes(56)


def rbf0():
    data = b"RBF0" + struct.pack("<H", 0xFFFB) + struct.pack("<I", 1) + struct.pack("<H", 21)
    data += b"rage__crRigSetManager\x00"
    return data + bytes(64 - len(data))


def raw():
    return b"\x11\x22\x33\x44" + bytes(60)


def text():
    data = b"<AnimData></AnimData>" * 4
    return data + bytes(max(0, 64 - len(data)))


def bink():
    data = bytearray(0x40)
    data[:4] = b"KB2n"
    struct.pack_into("<I", data, 4, len(data) - 8)
    struct.pack_into("<I", data, 8, 1)
    struct.pack_into("<I", data, 0x14, 64)
    struct.pack_into("<I", data, 0x18, 64)
    return bytes(data)


PAYLOADS = [move(), page(), names(), skeleton(), chunks(), clip(), rbf0(), raw(), text(), bink()]
EXPECTED = ["move", "page-directory", "name-table", "skeleton-names", "typehash", "clip-stream",
            "rbf0", "raw:11223344", "text", "bink2"]


class ClassifyTest(unittest.TestCase):
    def test_signatures(self):
        got = [anim_index.classify(p, 0, len(p))["signature"] for p in PAYLOADS]
        self.assertEqual(got, EXPECTED)

    def test_move_header(self):
        r = anim_index.classify(move(), 0x100, 90)
        self.assertEqual((r["version"], r["detail"]), (19, "size_hint=852"))

    def test_page_header(self):
        r = anim_index.classify(page(), 0x200, 40)
        self.assertEqual((r["version"], r["count"], r["detail"]), (122, 3, "0DRP at +16"))

    def test_names(self):
        r = anim_index.classify(names(), 0x300, 30)
        self.assertEqual((r["name"], r["count"]), ("pack:/alpha.clip", 2))

    def test_skeleton(self):
        r = anim_index.classify(skeleton(), 0x400, 20)
        self.assertEqual((r["name"], r["count"]), ("SKEL_Spine", 2))

    def test_typehash(self):
        r = anim_index.classify(chunks(), 0x500, 70)
        self.assertEqual((r["version"], r["count"], r["detail"]), (8, 7, "type=0x88560988"))

    def test_clip_header(self):
        r = anim_index.classify(clip(), 0x600, 70)
        self.assertEqual((r["version"], r["detail"]), (3, "24-byte header"))

    def test_rbf0(self):
        r = anim_index.classify(rbf0(), 0x700, 30)
        self.assertEqual((r["name"], r["detail"]), ("rage__crRigSetManager", "rig metadata"))


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
        rows, summary = anim_index.scan_rpf(self.rpf)
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
        rows, summary = anim_index.scan_rpf(self.rpf, limit=limit)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["signature"], "move")
        self.assertEqual(summary["span"], limit)

    def test_csv_and_json(self):
        rows, summary = anim_index.scan_rpf(self.rpf)
        csv_path = os.path.join(self.root, "out.csv")
        json_path = os.path.join(self.root, "out.json")
        anim_index.write_output(csv_path, rows, summary)
        anim_index.write_output(json_path, rows, summary)
        with open(csv_path, newline="", encoding="utf-8") as f:
            table = list(csv.reader(f))
        self.assertEqual(table[0], anim_index.COLUMNS)
        self.assertEqual(len(table), len(PAYLOADS) + 1)
        self.assertEqual(table[1][3], "move")
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
        self.assertEqual(doc["summary"]["signatures"]["move"]["files"], 1)
        self.assertEqual(doc["summary"]["signatures"]["page-directory"]["files"], 1)

    def test_directory(self):
        folder = os.path.join(self.root, "carved")
        os.makedirs(folder)
        for i, payload in enumerate((move(), page(), names())):
            with open(os.path.join(folder, f"{0x1000 + i * 0x100:012x}.bin"), "wb") as f:
                f.write(payload)
        rows, summary = anim_index.scan_dir(folder)
        self.assertEqual([r["offset"] for r in rows], [0x1000, 0x1100, 0x1200])
        self.assertEqual([r["signature"] for r in rows], ["move", "page-directory", "name-table"])
        self.assertEqual(summary["mode"], "directory")


if __name__ == "__main__":
    unittest.main()
