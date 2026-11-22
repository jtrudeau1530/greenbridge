"""navmesh_index must find the measured 0DRP page directory in an RPF9 stream and in carved files.

    python -I tests/test_navmesh_index.py

Everything is synthetic: a hand-built RPF9 header, zlib streams and a fake page. No game data is used.
"""
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "navmesh_index.py")
KEY = 0x8A2D3FF3DF663960
FIELD68 = 141492


def load_tool_module():
    spec = importlib.util.spec_from_file_location("navmesh_index_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    def test_out_and_csv_in_missing_folders_are_created(self):
        # Before: write_outputs opened both paths directly, so a missing folder ended a finished scan in a
        # FileNotFoundError traceback and lost the index.
        folder = os.path.join(self.root, "carve")
        os.makedirs(folder)
        with open(os.path.join(folder, "page.bin"), "wb") as f:
            f.write(page(pages=5))
        out = os.path.join(self.root, "no", "such", "index.json")
        csv_path = os.path.join(self.root, "also", "missing", "index.csv")
        self.run_tool(folder, "--out", out, "--csv", csv_path, "--pattern", "*.bin")
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["page_counts"], {"5": 1})
        with open(csv_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("0DRP@16", lines[1])

    def assert_output_collision(self, *args):
        r = subprocess.run([sys.executable, "-I", TOOL, *args], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("--out and --csv must refer to different files", r.stderr)

    def test_explicit_output_collision_before_scan(self):
        # A missing input would give a different error if the scan started.
        self.assert_output_collision("missing.rpf", "--out", "new.json", "--csv", "./new.json")
        self.assertFalse(os.path.exists(os.path.join(self.root, "new.json")))

    def test_default_output_collision_preserves_report(self):
        make_rpf(os.path.join(self.root, "nav.rpf"))
        report = os.path.join(self.root, "nav.navmesh-index.json")
        with open(report, "wb") as f:
            f.write(b"previous report")
        self.assert_output_collision("nav.rpf", "--csv", report)
        with open(report, "rb") as f:
            self.assertEqual(f.read(), b"previous report")

    def test_hard_link_output_collision(self):
        report = os.path.join(self.root, "out.json")
        with open(report, "wb") as f:
            f.write(b"previous report")
        try:
            os.link(report, os.path.join(self.root, "out.csv"))
        except OSError as exc:
            self.skipTest(f"cannot create a hard link: {exc}")
        self.assert_output_collision(self.root, "--out", "out.json", "--csv", "out.csv")
        with open(report, "rb") as f:
            self.assertEqual(f.read(), b"previous report")

    def test_not_an_rpf(self):
        path = os.path.join(self.root, "not.rpf")
        with open(path, "wb") as f:
            f.write(b"nope" * 64)
        r = subprocess.run([sys.executable, "-I", TOOL, path, "--out", os.path.join(self.root, "x.json")],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not an RPF7/8/9 archive", r.stderr)

    def test_limit_below_one_is_refused(self):
        path = os.path.join(self.root, "nav.rpf")
        make_rpf(path)
        for extra in (("--limit", "0"), ("--limit", "-100"), ("--start", "-1")):
            out = os.path.join(self.root, "x.json")
            r = subprocess.run([sys.executable, "-I", TOOL, path, "--out", out, *extra],
                               cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertFalse(os.path.exists(out))

    def test_directory_refuses_start_and_limit(self):
        folder = os.path.join(self.root, "carve")
        os.makedirs(folder)
        with open(os.path.join(folder, "page.bin"), "wb") as f:
            f.write(page())
        for extra in (("--start", "0"), ("--limit", "128"), ("--start", "0", "--limit", "128")):
            out = os.path.join(self.root, "dir_bad.json")
            r = subprocess.run([sys.executable, "-I", TOOL, folder, "--out", out, *extra],
                               cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("--start and --limit apply to a single archive, not a directory", r.stderr)
            self.assertFalse(os.path.exists(out))

    def test_archive_refuses_pattern(self):
        rpf = os.path.join(self.root, "nav.rpf")
        make_rpf(rpf)
        for pat in ("*.bin", "*.dat"):
            out = os.path.join(self.root, "rpf_bad.json")
            r = subprocess.run([sys.executable, "-I", TOOL, rpf, "--out", out, "--pattern", pat],
                               cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("--pattern applies to a directory, not an archive", r.stderr)
            self.assertFalse(os.path.exists(out))

    def test_scan_archive_closes_mmap_on_success(self):
        mod = load_tool_module()
        rpf = os.path.join(self.root, "close_ok.rpf")
        make_rpf(rpf)
        real_mmap = mod.mmap.mmap
        created = []

        def tracking_mmap(*args, **kwargs):
            m = real_mmap(*args, **kwargs)
            created.append(m)
            return m

        try:
            with mock.patch.object(mod.mmap, "mmap", side_effect=tracking_mmap):
                rows, summary = mod.scan_archive(rpf)
            self.assertEqual(len(rows), 1)
            self.assertEqual(summary["matched"], 1)
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].closed)
            os.remove(rpf)
            self.assertFalse(os.path.exists(rpf))
        finally:
            for m in created:
                if not m.closed:
                    m.close()

    def test_scan_archive_closes_mmap_on_start_out_of_range(self):
        mod = load_tool_module()
        rpf = os.path.join(self.root, "close_err.rpf")
        make_rpf(rpf)
        size = os.path.getsize(rpf)
        real_mmap = mod.mmap.mmap
        created = []

        def tracking_mmap(*args, **kwargs):
            m = real_mmap(*args, **kwargs)
            created.append(m)
            return m

        try:
            with mock.patch.object(mod.mmap, "mmap", side_effect=tracking_mmap):
                with self.assertRaises(SystemExit):
                    mod.scan_archive(rpf, start=size)
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].closed)
            os.remove(rpf)
            self.assertFalse(os.path.exists(rpf))
        finally:
            for m in created:
                if not m.closed:
                    m.close()


class MissingInputTest(unittest.TestCase):
    def test_missing_input_is_a_message_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "n.json")
            p = subprocess.run([sys.executable, "-I", TOOL, os.path.join(tmp, "missing.rpf"), "--out", out],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 1, p.stderr)
            self.assertIn("missing.rpf", p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertFalse(os.path.exists(out))


class LimitWindowTest(unittest.TestCase):
    """--limit must bound the bytes read, not just where signatures are searched."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rpf = os.path.join(self.tmp.name, "window.rpf")
        self.streams = make_rpf(self.rpf)
        self.mod = load_tool_module()

    def summary(self, limit):
        _rows, summary = self.mod.scan_archive(self.rpf, limit=limit)
        self.assertLessEqual(summary["compressed_bytes"], summary["span"])
        self.assertLessEqual(summary["readable_fraction"], 1.0)
        self.assertEqual(summary["compressed_bytes"] + summary["zero_padding"] + summary["unread_bytes"],
                         summary["span"])
        return summary

    def test_stream_crossing_the_limit_is_not_counted(self):
        # The zlib header is inside the window but its body is not: counting it would make the
        # compressed bytes exceed the span and readable_fraction greater than 1.
        summary = self.summary(10)
        self.assertEqual(summary["span"], 10)
        self.assertEqual(summary["streams"], 0)
        self.assertEqual(summary["compressed_bytes"], 0)

    def test_stream_ending_at_the_limit_is_counted(self):
        summary = self.summary(len(self.streams[0]))
        self.assertEqual(summary["span"], len(self.streams[0]))
        self.assertEqual(summary["streams"], 1)
        self.assertEqual(summary["compressed_bytes"], len(self.streams[0]))
        self.assertEqual(summary["readable_fraction"], 1.0)

    def test_window_too_short_for_a_signature_has_no_streams(self):
        summary = self.summary(1)
        self.assertEqual(summary["streams"], 0)
        self.assertEqual(summary["span"], 1)


if __name__ == "__main__":
    unittest.main()
