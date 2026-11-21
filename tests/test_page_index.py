"""Synthetic tests for tools/page_index.py. No game files are read or needed."""
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
TOOL = os.path.join(ROOT, "tools", "page_index.py")
BASE = 0x50000000


def header(size, q28=0, q30=0, q38=0, q40=0, counts=(0, 0, 0, 0), page_count=2):
    data = bytearray(size)
    data[0x10:0x14] = b"0DRP"
    struct.pack_into("<I", data, 0x14, 122)
    struct.pack_into("<Q", data, 0x20, page_count)
    for at, value in ((0x28, q28), (0x30, q30), (0x38, q38), (0x40, q40)):
        struct.pack_into("<Q", data, at, BASE + value if value else 0)
    for i, value in enumerate(counts):
        struct.pack_into("<H", data, 0x48 + 2 * i, value)
    return data


def put_q38(data, table, entries):
    for i, (hash_value, size, target) in enumerate(entries):
        struct.pack_into("<IIQ", data, table + 16 * i, hash_value, size, BASE + target)


def make_rpf(path, streams):
    data = bytearray(0x40)
    data[0:4] = b"RPF9"
    struct.pack_into("<I", data, 4, 0x40)
    offsets = []
    for stream in streams:
        offsets.append(len(data))
        data += zlib.compress(stream)
    with open(path, "wb") as f:
        f.write(data)
    return offsets


def run_tool(source, out, extra=()):
    proc = subprocess.run([sys.executable, "-I", TOOL, source, out, "--quiet", *extra],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr)


class PageIndexTest(unittest.TestCase):
    def test_simple_page_and_q28_raw(self):
        page = header(0x200, q28=0x180, q38=0x100, counts=(1, 0, 1, 0))
        page[0x60:0xA0] = b"\xaa" * 0x40
        put_q38(page, 0x100, [(0x1234, 0x40, 0x60)])
        struct.pack_into("<IIQQII", page, 0x180, 0x1234, 0x999, 0x300, BASE + 0x60, 0x10, 0x30000)
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            out = os.path.join(tmp, "pages.json")
            offsets = make_rpf(arc, [page])
            run_tool(arc, out)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["pages"], 1)
            self.assertEqual(doc["summary"]["complete_pages"], 1)
            self.assertEqual(doc["summary"]["q38_records"], 1)
            self.assertEqual(doc["summary"]["payload_at_0x60"], 1)
            page_doc = doc["pages"][0]
            self.assertEqual(page_doc["offset"], offsets[0])
            self.assertEqual(page_doc["stored"], len(zlib.compress(page)))
            self.assertEqual(page_doc["inflated"], 0x200)
            self.assertEqual(page_doc["version"], 122)
            self.assertEqual(page_doc["page_count"], 2)
            self.assertEqual(page_doc["tables"], {"q28": 0x180, "q30": None, "q38": 0x100, "q40": None})
            rec = page_doc["q38_records"][0]
            self.assertEqual((rec["hash"], rec["size"], rec["target"], rec["target_stream"]),
                             (0x1234, 0x40, 0x60, 0))
            q28 = page_doc["q28_records"][0]
            self.assertEqual((q28["hash_a"], q28["hash_b"], q28["field_08"], q28["target"],
                              q28["count_or_size"], q28["flags"]),
                             (0x1234, 0x999, 0x300, 0x60, 0x10, 0x30000))
            csv_path = os.path.join(tmp, "pages.csv")
            run_tool(arc, csv_path)
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["hash"], "0x00001234")
            self.assertEqual(rows[0]["target"], "0x60")
            self.assertEqual(rows[0]["target_stream"], "0")

    def test_continuation_stream_is_appended(self):
        page = header(0x100, q38=0xC0, counts=(0, 0, 1, 0))
        put_q38(page, 0xC0, [(0x5678, 0x80, 0x100)])
        cont = b"\xbb" * 0x100
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "cont.rpf")
            out = os.path.join(tmp, "pages.json")
            offsets = make_rpf(arc, [page, cont])
            run_tool(arc, out)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["pages"], 1)
            self.assertEqual(doc["summary"]["pages_with_continuations"], 1)
            page_doc = doc["pages"][0]
            self.assertTrue(page_doc["complete"])
            self.assertEqual(len(page_doc["continuations"]), 1)
            self.assertEqual(page_doc["continuations"][0]["offset"], offsets[1])
            self.assertEqual(page_doc["continuations"][0]["inflated"], 0x100)
            rec = page_doc["q38_records"][0]
            self.assertEqual((rec["target"], rec["size"], rec["target_stream"]), (0x100, 0x80, 1))
            self.assertEqual(page_doc["span"], 0x180)

    def test_q38_table_straddles_stream(self):
        page = header(0x100, q38=0xF8, counts=(0, 0, 2, 0))
        entries = [(0x1111, 0x10, 0x60), (0x2222, 0x20, 0x120)]
        table = b"".join(struct.pack("<IIQ", h, s, BASE + t) for h, s, t in entries)
        page[0xF8:0x100] = table[:8]
        page[0x60:0x70] = b"\xaa" * 0x10
        cont = bytearray(0x140)
        cont[:8] = table[8:16]
        cont[8:24] = table[16:32]
        cont[0x20:0x40] = b"\xcc" * 0x20
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "straddle.rpf")
            out = os.path.join(tmp, "pages.json")
            make_rpf(arc, [page, bytes(cont)])
            run_tool(arc, out)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["pages"], 1)
            page_doc = doc["pages"][0]
            self.assertTrue(page_doc["complete"])
            self.assertEqual(page_doc["tables"]["q38"], 0xF8)
            self.assertEqual([(r["hash"], r["size"], r["target"], r["target_stream"])
                              for r in page_doc["q38_records"]],
                             [(0x1111, 0x10, 0x60, 0), (0x2222, 0x20, 0x120, 1)])
            self.assertEqual(page_doc["span"], 0x140)

    def test_carve_directory_mode(self):
        page = header(0x100, q38=0xC0, counts=(0, 0, 1, 0))
        put_q38(page, 0xC0, [(0x9999, 0x20, 0x100)])
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "000000000040.bin"), "wb") as f:
                f.write(page)
            with open(os.path.join(tmp, "000000000200.bin"), "wb") as f:
                f.write(b"\xdd" * 0x40)
            out = os.path.join(tmp, "pages.json")
            run_tool(tmp, out)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["mode"], "directory")
            self.assertEqual(doc["summary"]["pages"], 1)
            page_doc = doc["pages"][0]
            self.assertEqual(page_doc["offset"], 0x40)
            self.assertIsNone(page_doc["stored"])
            self.assertEqual(page_doc["continuations"][0]["offset"], 0x200)
            self.assertEqual(page_doc["q38_records"][0]["target_stream"], 1)


if __name__ == "__main__":
    unittest.main()
