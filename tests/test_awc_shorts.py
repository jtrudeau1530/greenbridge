"""Tests for tools/awc_shorts.py, built from synthetic AWC data (no game files).

    python -I tests/test_awc_shorts.py
"""
import contextlib
import importlib.util
import io
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tool():
    path = os.path.join(ROOT, "tools", "awc_shorts.py")
    spec = importlib.util.spec_from_file_location("awc_shorts_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shorts = load_tool()


def tag(typ, offset, size):
    return (typ << 56) | (size << 28) | offset


def build_music_bank(toc, samples, rate=48000):
    """One 0x48 stream with one descriptor channel and one 2048-byte block."""
    desc = bytearray(28)
    struct.pack_into("<III", desc, 0, 1, 2048, 1)
    struct.pack_into("<II", desc, 12, 0x1234, samples)
    struct.pack_into("<H", desc, 22, rate)
    desc[24] = 0x0B
    data = bytearray(4096)
    struct.pack_into("<I", data, 4, 1)       # one subblock
    struct.pack_into("<H", data, 2048, 1)
    data[2050] = toc
    header = 16 + 4 + 2 * 8
    out = bytearray(b"ADAT" + struct.pack("<III", 0, 1, header))
    out += struct.pack("<I", 0x1234 | (2 << 29))
    out += struct.pack("<Q", tag(0x48, header, len(desc)))
    out += struct.pack("<Q", tag(0x55, header + len(desc), len(data)))
    return bytes(out + desc + data)


def ogg_head_and_last_granule(data):
    if data[:4] != b"OggS":
        raise AssertionError("not Ogg")
    p = 0
    granule = None
    first_body = None
    while p < len(data):
        if data[p:p + 4] != b"OggS":
            raise AssertionError("bad Ogg page")
        g, = struct.unpack_from("<q", data, p + 6)
        nseg = data[p + 26]
        body = p + 27 + nseg
        body_size = sum(data[p + 27:p + 27 + nseg])
        if first_body is None:
            first_body = body
        granule = g
        p = body + body_size
    if first_body is None:
        raise AssertionError("empty Ogg")
    return data[first_body + 9], granule


class ShortsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out")

    def write(self, name, toc, samples):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(build_music_bank(toc, samples))
        return path

    def test_stereo_selection_and_granule(self):
        path = self.write("stereo.awc", 0xFC, 480000)   # 5 s interleaved
        rows = shorts.export_bank(path, self.out, 4, 6)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["stream"], rows[0]["channels"]), ("00001234", 2))
        self.assertAlmostEqual(rows[0]["duration_s"], 5.0, places=6)
        with open(os.path.join(self.out, "stereo", "00001234.ogg"), "rb") as f:
            channels, granule = ogg_head_and_last_granule(f.read())
        self.assertEqual(channels, 2)
        self.assertEqual(granule, 960)   # the synthetic bank has one 20 ms packet

    def test_mono_selection_and_filter(self):
        path = self.write("mono.awc", 0x78, 480000)      # 10 s mono
        self.assertEqual(shorts.export_bank(path, self.out, 9, 11)[0]["channels"], 1)
        self.assertEqual(shorts.export_bank(path, self.out, 4, 6), [])

    def test_cli_writes_csv(self):
        path = self.write("cli.awc", 0xFC, 480000)
        argv = [path, self.out, "--min-seconds", "4", "--max-seconds", "6", "--jobs", "1"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(shorts.main(argv), 0)
        fields, rows = None, []
        with open(os.path.join(self.out, "shorts.csv"), encoding="utf-8") as f:
            import csv
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = list(reader)
        self.assertEqual(fields, list(shorts.FIELDS))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["channels"], "2")
        self.assertTrue(rows[0]["file"].endswith("00001234.ogg"))

    def test_bad_range(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(shorts.main(["x", self.out, "--min-seconds", "6", "--max-seconds", "5"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
