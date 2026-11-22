"""Synthetic-header tests for tools/restofworld_index.py.

    python -I tests/test_restofworld_index.py

Every byte is built by this test. No game file, asset name or payload is used.
"""
import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "restofworld_index.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import restofworld_index  # noqa: E402


def psin():
    sizes = [0x30, 0x10, 0x20, 8, 0x58, 0x14]
    tags = [b"LSCH", b"LENM", b"LSNS", b"LGID", b"LRSC", b"CHKS"]
    data = bytearray(0x20 + sum(sizes))
    data[:4] = b"PSIN"
    data[4:8] = (0x20).to_bytes(4, "big")
    struct.pack_into("<I", data, 8, 261)
    pos = 0x20
    for tag, size in zip(tags, sizes):
        data[pos:pos + 4] = tag
        data[pos + 4:pos + 8] = size.to_bytes(4, "big")
        if tag == b"LRSC":
            data[pos + 8:pos + 12] = b"0DRP"
            struct.pack_into("<I", data, pos + 12, 122)
            struct.pack_into("<I", data, pos + 8 + 0x0C, 7)
        pos += size
    return bytes(data)


def page():
    data = bytearray(0x60)
    struct.pack_into("<Q", data, 8, 0x50001000)
    data[0x10:0x14] = b"0DRP"
    struct.pack_into("<I", data, 0x14, 122)
    struct.pack_into("<Q", data, 0x20, 3)
    return bytes(data)


def page_at_zero():
    data = bytearray(0x60)
    data[:4] = b"0DRP"
    struct.pack_into("<I", data, 4, 122)
    struct.pack_into("<I", data, 0x0C, 7)
    return bytes(data)


def rsc8():
    data = bytearray(0x50)
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
    data += b"TestSchema\x00"
    return data + bytes(64 - len(data))


def bink():
    data = bytearray(0x40)
    data[:4] = b"KB2n"
    struct.pack_into("<I", data, 4, len(data) - 8)
    struct.pack_into("<I", data, 8, 1)
    struct.pack_into("<I", data, 0x14, 64)
    struct.pack_into("<I", data, 0x18, 64)
    return bytes(data)


def xml():
    data = b"\xef\xbb\xbf<?xml version=\"1.0\"?><X/>"
    return data + bytes(max(0, 64 - len(data)))


def respage():
    return b"\x42\x40\x00\x00" + bytes(60)


def zero():
    return bytes(64)


def typehash():
    return struct.pack("<IIII", 0x12345678, 8, 7, 0) + bytes(48)


def clip():
    return struct.pack("<I", 3) + b"\x01\x01\x00\x00" + bytes(56)


def raw():
    return b"\x11\x22\x33\x44" + bytes(60)


def text():
    return b"A" * 80


def awc():
    return b"AWC\x00" + struct.pack("<I", 1) + bytes(56)


def adat():
    return b"ADAT" + bytes(60)


def fssm():
    return b"FSSM" + struct.pack("<I", 2) + bytes(56)


def scrp():
    return b"SCRP" + struct.pack("<I", 3) + bytes(56)


def ysc():
    return b"YSC\x00" + bytes(60)


def rscc():
    return b"RSCC" + struct.pack("<I", 4) + bytes(56)


PAYLOADS = [psin(), page(), rsc8(), nested(), rbf0(), bink(), xml(), respage(),
            zero(), typehash(), clip(), raw(), text()]
EXPECTED = ["psin", "0drp", "rsc8", "rpf9-nested", "rbf0", "bink2", "xml", "respage",
            "zero-prefix", "typehash", "clip", "raw:11223344", "text"]


class ClassifyTest(unittest.TestCase):
    def test_signatures(self):
        got = [restofworld_index.classify(p, 0, len(p))["signature"] for p in PAYLOADS]
        self.assertEqual(got, EXPECTED)

    def test_psin_chain_and_complete_flag(self):
        chain = restofworld_index.psin_sections(psin())
        self.assertEqual([t for t, _, _ in chain],
                         ["LSCH", "LENM", "LSNS", "LGID", "LRSC", "CHKS"])
        self.assertEqual([o for _, o, _ in chain], [0x20, 0x50, 0x60, 0x80, 0x88, 0xE0])
        self.assertEqual([s for _, _, s in chain], [0x30, 0x10, 0x20, 8, 0x58, 0x14])
        row = restofworld_index.classify(psin(), 0x100, 90)
        self.assertEqual((row["signature"], row["count"]), ("psin", 6))
        self.assertIn("complete=1", row["detail"])
        self.assertIn("field8=261", row["detail"])

    def test_psin_truncated_is_partial(self):
        row = restofworld_index.classify(psin()[:-4], 0x100, 90)
        self.assertEqual(row["signature"], "psin")
        self.assertIn("complete=0", row["detail"])

    def test_page_header(self):
        row = restofworld_index.classify(page(), 0x200, 40)
        self.assertEqual((row["signature"], row["version"], row["count"], row["detail"]),
                         ("0drp", 122, 3, "0DRP@0x10"))

    def test_page_at_zero_uses_lrsc_count(self):
        row = restofworld_index.classify(page_at_zero(), 0x200, 40)
        self.assertEqual((row["signature"], row["version"], row["count"]), ("0drp", 122, 7))

    def test_rsc8_header(self):
        row = restofworld_index.classify(rsc8(), 0x300, 50)
        self.assertEqual((row["signature"], row["version"]), ("rsc8", 8001))
        self.assertEqual(row["detail"], "declared=4660;flags=1")

    def test_rpf9_header(self):
        row = restofworld_index.classify(nested(), 0x400, 60)
        self.assertEqual(row["signature"], "rpf9-nested")
        self.assertEqual(row["detail"], "data_off=2208;toc=2064")

    def test_rbf0_schema(self):
        row = restofworld_index.classify(rbf0(), 0x500, 70)
        self.assertEqual((row["signature"], row["detail"]), ("rbf0", "schema=TestSchema"))

    def test_bink_header(self):
        row = restofworld_index.classify(bink(), 0x600, 64)
        self.assertEqual((row["signature"], row["version"], row["count"], row["detail"]),
                         ("bink2", "n", 1, "64x64"))
        self.assertEqual(row["size"], 64)

    def test_other_container_magics(self):
        cases = [(awc(), "awc"), (adat(), "adat"), (fssm(), "fssm"),
                 (scrp(), "scrp"), (ysc(), "ysc"), (rscc(), "rscc")]
        for payload, expected in cases:
            self.assertEqual(restofworld_index.classify(payload, 0, len(payload))["signature"],
                             expected)


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
        rows, summary = restofworld_index.scan_rpf(self.rpf)
        self.assertEqual([r["offset"] for r in rows], self.offsets)
        self.assertEqual([r["signature"] for r in rows], EXPECTED)
        self.assertEqual(summary["files"], len(PAYLOADS))
        self.assertEqual(summary["bink2_files"], 1)
        self.assertEqual(summary["unread_bytes"], 0)
        self.assertEqual(summary["span"], self.span)
        self.assertEqual(summary["inflated_bytes"], sum(len(p) for p in PAYLOADS))
        zlibs = [r for r in rows if r["signature"] != "bink2"]
        self.assertEqual(sum(r["stored"] for r in zlibs),
                         sum(len(zlib.compress(p)) for p in PAYLOADS if p[:3] != b"KB2"))
        self.assertEqual(rows[-1]["stored"], len(zlib.compress(text())))
        self.assertGreater(summary["readable_fraction"], 0.5)
        self.assertEqual(summary["skipped_candidates"], 0)

    def test_scan_limit(self):
        limit = self.offsets[2] - 0x100 - 1
        rows, summary = restofworld_index.scan_rpf(self.rpf, limit=limit)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["signature"] for r in rows], ["psin", "0drp"])
        self.assertEqual(summary["span"], limit)

    def test_csv_and_json(self):
        rows, summary = restofworld_index.scan_rpf(self.rpf)
        csv_path = os.path.join(self.root, "out.csv")
        json_path = os.path.join(self.root, "out.json")
        restofworld_index.write_output(csv_path, rows, summary)
        restofworld_index.write_output(json_path, rows, summary)
        with open(csv_path, newline="", encoding="utf-8") as f:
            table = list(csv.reader(f))
        self.assertEqual(table[0], restofworld_index.COLUMNS)
        self.assertEqual(len(table), len(PAYLOADS) + 1)
        self.assertEqual(table[1][3], "psin")
        self.assertEqual(table[1][5], "6")
        with open(json_path, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["files"], len(PAYLOADS))
        self.assertEqual(len(doc["files"]), len(PAYLOADS))

    def test_bad_scan_arguments_fail_before_an_index_is_written(self):
        for flag, value, message in (("--start", "-1", "must be 0 or more"),
                                     ("--limit", "0", "must be at least 1"),
                                     ("--limit", "-100", "must be at least 1"),
                                     ("--max-inflate", "0", "must be at least 1")):
            out = os.path.join(self.root, "bad.json")
            proc = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--quiet", flag, value],
                                  capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertIn(message, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertFalse(os.path.exists(out))

    def test_missing_source_is_an_error_not_a_traceback(self):
        out = os.path.join(self.root, "missing.json")
        proc = subprocess.run([sys.executable, "-I", TOOL, os.path.join(self.root, "nope.rpf"), out, "--quiet"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("nope.rpf", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(os.path.exists(out))

    def test_unwritable_output_path_fails_before_the_scan(self):
        for out in (os.path.join(self.root, "no_such_folder", "x.json"), self.root):
            proc = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--quiet"],
                                  capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.root, "no_such_folder")))

    def test_cli(self):
        out = os.path.join(self.root, "cli.json")
        proc = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--quiet"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["signatures"]["psin"]["files"], 1)
        self.assertEqual(doc["summary"]["signatures"]["0drp"]["files"], 1)
        self.assertEqual(doc["summary"]["signatures"]["bink2"]["files"], 1)

    def test_directory_skips_index_and_reads_offsets(self):
        folder = os.path.join(self.root, "carved")
        os.makedirs(folder)
        for i, payload in enumerate((psin(), page(), page_at_zero(), rsc8())):
            with open(os.path.join(folder, f"{0x1000 + i * 0x100:012x}.bin"), "wb") as f:
                f.write(payload)
        with open(os.path.join(folder, "index.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        rows, summary = restofworld_index.scan_dir(folder)
        self.assertEqual([r["offset"] for r in rows], [0x1000, 0x1100, 0x1200, 0x1300])
        self.assertEqual([r["signature"] for r in rows], ["psin", "0drp", "0drp", "rsc8"])
        self.assertEqual(rows[2]["count"], 7)
        self.assertEqual(summary["mode"], "directory")
        self.assertEqual(summary["files"], 4)

    def test_directory_rejects_start_and_limit(self):
        folder = os.path.join(self.root, "carved_opts")
        os.makedirs(folder)
        with open(os.path.join(folder, f"{0x1000:012x}.bin"), "wb") as f:
            f.write(page())
        for extra in (["--start", "0"], ["--limit", "64"], ["--start", "0", "--limit", "64"]):
            out = os.path.join(self.root, "dir_bad.json")
            proc = subprocess.run([sys.executable, "-I", TOOL, folder, out, "--quiet", *extra],
                                  capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertIn("--start and --limit apply to a single archive, not a directory", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertFalse(os.path.exists(out))

    def test_directory_respects_max_inflate(self):
        folder = os.path.join(self.root, "carved_limit")
        os.makedirs(folder)
        small_bin = page()  # 96 bytes
        large_bin = psin()  # 244 bytes
        large_bk2 = bytearray(200)
        large_bk2[:len(bink())] = bink()
        struct.pack_into("<I", large_bk2, 4, len(large_bk2) - 8)
        with open(os.path.join(folder, f"{0x1000:012x}.bin"), "wb") as f:
            f.write(small_bin)
        with open(os.path.join(folder, f"{0x2000:012x}.bin"), "wb") as f:
            f.write(large_bin)
        with open(os.path.join(folder, f"{0x3000:012x}.bk2"), "wb") as f:
            f.write(large_bk2)

        rows, summary = restofworld_index.scan_dir(folder, max_inflate=100)
        self.assertEqual([r["signature"] for r in rows], ["0drp", "bink2"])
        self.assertEqual(summary["files"], 2)
        self.assertEqual(summary["skipped_candidates"], 1)

        out = os.path.join(self.root, "dir_limit.json")
        proc = subprocess.run([sys.executable, "-I", TOOL, folder, out, "--quiet", "--max-inflate", "100"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["files"], 2)
        self.assertEqual(doc["summary"]["skipped_candidates"], 1)
        self.assertEqual([r["signature"] for r in doc["files"]], ["0drp", "bink2"])

    def test_scan_rpf_closes_mmap_on_success_and_out_of_range_start(self):
        created = []
        real_mmap = restofworld_index.mmap.mmap

        def track_mmap(*args, **kwargs):
            m = real_mmap(*args, **kwargs)
            created.append(m)
            return m

        with open(self.rpf, "rb") as src:
            raw_bytes = src.read()
        rpf_ok = os.path.join(self.root, "ok.rpf")
        rpf_err = os.path.join(self.root, "err.rpf")
        for path in (rpf_ok, rpf_err):
            with open(path, "wb") as dst:
                dst.write(raw_bytes)

        try:
            with unittest.mock.patch.object(restofworld_index.mmap, "mmap", side_effect=track_mmap):
                restofworld_index.scan_rpf(rpf_ok)
                self.assertEqual(len(created), 1)
                self.assertTrue(created[-1].closed)
                os.remove(rpf_ok)
                self.assertFalse(os.path.exists(rpf_ok))

                with self.assertRaisesRegex(ValueError, "start offset is outside the archive"):
                    restofworld_index.scan_rpf(rpf_err, start=len(raw_bytes) + 1)
                self.assertEqual(len(created), 2)
                self.assertTrue(created[-1].closed)
                os.remove(rpf_err)
                self.assertFalse(os.path.exists(rpf_err))
        finally:
            for m in created:
                if not m.closed:
                    m.close()


if __name__ == "__main__":
    unittest.main()
