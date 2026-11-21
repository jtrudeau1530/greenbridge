"""Tests for tools/bink_index.py, built from synthetic Bink 2 headers (no game data).

    python -I tests/test_bink_index.py
"""
import csv
import importlib.util
import json
import os
import struct
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "bink_index.py")
    spec = importlib.util.spec_from_file_location("bink_index_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bink = load_module()


def make_stream(revision="n", frames=4, width=320, height=240, audio=0, body=256,
                fps_num=30, fps_den=1):
    """A plausible Bink 2 stream: fixed header, audio metadata, frame table, payload."""
    extra = b"\x00\x05\xe9\x87" if revision in "ijkn" else b""
    audio_blob = b""
    if audio:
        audio_blob += struct.pack("<I", 0x23A00) * audio       # max decoded sizes
        for _ in range(audio):
            audio_blob += struct.pack("<HH", 48000, 0x7000)   # rate, flags
        for track in range(audio):
            audio_blob += struct.pack("<I", track)            # track IDs
    total = 0x2C + len(extra) + len(audio_blob) + 4 * frames + body
    header = bytearray(0x2C)
    header[0:4] = b"KB2" + revision.encode()
    struct.pack_into("<10I", header, 4, total - 8, frames, body, frames,
                     width, height, fps_num, fps_den, 0x40013, audio)
    return bytes(header) + extra + audio_blob + bytes(4 * frames) + bytes(body)


class TestHeader(unittest.TestCase):
    def test_fields(self):
        row = bink.parse_header(make_stream(), 0)
        self.assertEqual(row["revision"], "n")
        self.assertEqual((row["width"], row["height"]), (320, 240))
        self.assertEqual(row["frames"], 4)
        self.assertEqual((row["fps_num"], row["fps_den"], row["fps"]), (30, 1, 30.0))
        self.assertEqual(row["size"], len(make_stream()))
        self.assertEqual(row["audio_tracks"], 0)

    def test_revision_j_and_audio_descriptors(self):
        row = bink.parse_header(make_stream("j", audio=2), 0)
        self.assertEqual(row["audio_tracks"], 2)
        self.assertEqual(row["audio_rates"], [48000, 48000])
        self.assertEqual(row["audio_flags"], ["0x7000", "0x7000"])
        self.assertEqual(row["audio_codecs"], ["binkaudio_dct", "binkaudio_dct"])
        self.assertEqual(row["audio_channels"], [2, 2])

    def test_truncated_file_is_skipped(self):
        data = make_stream()[:-100]
        self.assertIsNone(bink.parse_header(data, 0))

    def test_bad_fields_are_rejected(self):
        bad_fps = bytearray(make_stream())
        struct.pack_into("<I", bad_fps, 0x1C, 0)
        self.assertIsNone(bink.parse_header(bad_fps, 0))
        bad_width = bytearray(make_stream())
        struct.pack_into("<I", bad_width, 0x14, 0)
        self.assertIsNone(bink.parse_header(bad_width, 0))
        bad_audio = bytearray(make_stream())
        struct.pack_into("<I", bad_audio, 0x28, 999)
        self.assertIsNone(bink.parse_header(bad_audio, 0))


class TestScanAndWrite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        with open(os.path.join(self.dir, "one.bk2"), "wb") as fh:
            fh.write(make_stream(frames=7))
        with open(os.path.join(self.dir, "truncated.bk2"), "wb") as fh:
            fh.write(make_stream()[:-64])
        with open(os.path.join(self.dir, "junk.bin"), "wb") as fh:
            fh.write(b"not a movie" * 64)

    def tearDown(self):
        self.tmp.cleanup()

    def test_directory_scan_skips_bad_files(self):
        rows, summary = bink.scan_target(self.dir)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "one.bk2")
        self.assertEqual(rows[0]["frames"], 7)
        self.assertEqual(summary["streams"], 1)

    def test_csv_and_json(self):
        rows, summary = bink.scan_target(self.dir)
        csv_path = os.path.join(self.dir, "index.csv")
        json_path = os.path.join(self.dir, "index.json")
        bink.write_csv(csv_path, rows, probe=False)
        bink.write_json(json_path, self.dir, rows, summary, probe=False)
        with open(csv_path, encoding="utf-8", newline="") as fh:
            got = list(csv.DictReader(fh))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["revision"], "n")
        self.assertNotIn("decodes", got[0])
        with open(json_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertEqual(doc["summary"]["streams"], 1)
        self.assertEqual(doc["streams"][0]["width"], 320)
        self.assertNotIn("decodes", doc["streams"][0])


if __name__ == "__main__":
    unittest.main()
