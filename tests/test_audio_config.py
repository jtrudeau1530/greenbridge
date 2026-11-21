"""Synthetic tests for tools/audio_config.py (no game data)."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "audio_config.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import audio_config as A  # noqa: E402


def make_record(config_id, duration_ms, hash_a=0x11111111, hash_b=0x22222222, tail=b"\x00" * 22):
    """One 46-byte record: duration, then the marker block at +4."""
    b = bytearray(46)
    struct.pack_into("<I", b, 0, duration_ms)
    b[4:6] = b"\x0c\x01"
    struct.pack_into("<H", b, 6, config_id)
    b[8:12] = b"\x02\x00\x0c\x00"
    struct.pack_into("<I", b, 12, hash_a)
    struct.pack_into("<I", b, 16, hash_b)
    b[20:24] = b"\x00\x00\x02\x23"
    b[24:24 + len(tail)] = tail
    return bytes(b)


class TestHash(unittest.TestCase):
    def test_joaat_vectors(self):
        self.assertEqual(A.joaat("test"), 0x3F75CCC1)
        self.assertEqual(A.joaat("Radio_Test"), 0xE663FCAE)
        self.assertEqual(A.joaat(""), 0)

    def test_normalized_name(self):
        self.assertEqual(A.normalized_name("Hello,  World!"), "hello_world")
        self.assertEqual(A.normalized_name("A.B-C"), "a_b_c")
        self.assertEqual(A.normalized_name("Already_ok"), "already_ok")


class TestScan(unittest.TestCase):
    def test_single_record(self):
        blob = b"\x00\x00\x00\x00" + make_record(0xB422, 264_091, 0x49FC8153, 0xDAA8AC2F)
        got = A.scan(blob)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["id"], 0xB422)
        self.assertEqual(got[0]["duration_ms"], 264_091)
        self.assertEqual(got[0]["duration_s"], 264.091)
        self.assertEqual(got[0]["hash_a"], 0x49FC8153)
        self.assertEqual(got[0]["hash_b"], 0xDAA8AC2F)
        self.assertEqual(got[0]["stream_id"], 0xDAA8AC2F & A.STREAM_MASK)
        self.assertEqual(got[0]["offset"], 8)

    def test_two_adjacent_records(self):
        blob = b"\x00" * 4 + make_record(0x1000, 100_000) + make_record(0x1001, 200_000)
        got = A.scan(blob)
        self.assertEqual([r["id"] for r in got], [0x1000, 0x1001])
        self.assertEqual([r["offset"] for r in got], [8, 54])

    def test_rejects_bad_marker_and_duration(self):
        wrong_marker = bytearray(make_record(0x1000, 100_000))
        wrong_marker[8] = 0
        low = make_record(0x1001, 1_000)
        high = make_record(0x1002, 3_000_000)
        blob = b"\x00" * 4 + bytes(wrong_marker) + low + high
        self.assertEqual(A.scan(blob), [])

    def test_truncated_record(self):
        # A partial record (marker present, third marker cut off) must not be reported.
        blob = b"\x00" * 4 + make_record(0x1000, 100_000)[:22]
        self.assertEqual(A.scan(blob), [])

    def test_join_across_streams(self):
        # A zlib block boundary can split a record. Split before the third marker ends.
        record = make_record(0x3000, 150_000)
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "00000001.bin"), "wb") as f:
                f.write(b"\x00" * 4 + record[:22])
            with open(os.path.join(tmp, "00000002.bin"), "wb") as f:
                f.write(record[22:])
            self.assertEqual(len(A.collect(tmp)), 0)
            self.assertEqual(len(A.collect(tmp, join=True)), 1)


class TestCli(unittest.TestCase):
    def test_json_and_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = "Test Song"
            hb = A.joaat(A.normalized_name(name))
            blob = b"\x00" * 4 + make_record(0x2000, 123_456, hash_b=hb)
            with open(os.path.join(tmp, "00000001.bin"), "wb") as f:
                f.write(blob)
            pool = os.path.join(tmp, "names.txt")
            with open(pool, "wb") as f:
                f.write(name.encode() + b"\x00Other\x00")
            p = subprocess.run([sys.executable, "-I", TOOL, tmp, "--json", "--names", pool],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            doc = json.loads(p.stdout)
            self.assertEqual(doc["summary"]["records"], 1)
            self.assertEqual(doc["summary"]["distinct_ids"], 1)
            self.assertEqual(doc["summary"]["named"], 1)
            self.assertEqual(doc["records"][0]["name"], name)
            self.assertEqual(doc["records"][0]["stream_id"], hb & A.STREAM_MASK)


if __name__ == "__main__":
    unittest.main()
