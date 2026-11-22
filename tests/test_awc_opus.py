"""Regression tests for tools/awc_opus.py, built entirely from synthetic AWC data (no game files).

    python -I tests/test_awc_opus.py
"""
import contextlib
import importlib.util
import io
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_awc_opus():
    path = os.path.join(ROOT, "tools", "awc_opus.py")
    spec = importlib.util.spec_from_file_location("awc_opus_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


awc = load_awc_opus()


def read_wav(path):
    """Fields of the canonical 44-byte-header WAV that write_wav produces."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE" or data[12:16] != b"fmt " or data[36:40] != b"data":
        raise AssertionError("not a canonical RIFF/WAVE file")
    riff_size, = struct.unpack_from("<I", data, 4)
    fmt_size, fmt, channels, rate, byte_rate, block_align, bits = struct.unpack_from("<IHHIIHH", data, 16)
    data_size, = struct.unpack_from("<I", data, 40)
    return {
        "data": data,
        "riff_size": riff_size,
        "fmt_size": fmt_size,
        "format": fmt,
        "channels": channels,
        "rate": rate,
        "byte_rate": byte_rate,
        "block_align": block_align,
        "bits": bits,
        "data_size": data_size,
        "pcm": data[44:],
    }


def parse_ogg_pages(data):
    """All pages as dicts; also verifies that every stored CRC matches the page contents."""
    pages = []
    p = 0
    while p < len(data):
        if data[p:p + 4] != b"OggS":
            raise AssertionError("bad Ogg capture pattern at %d" % p)
        flags = data[p + 5]
        granule, = struct.unpack_from("<q", data, p + 6)
        serial, seq, crc = struct.unpack_from("<III", data, p + 14)
        nseg = data[p + 26]
        lacing = data[p + 27:p + 27 + nseg]
        body_size = sum(lacing)
        body_start = p + 27 + nseg
        body = data[body_start:body_start + body_size]
        page = bytearray(data[p:body_start + body_size])
        struct.pack_into("<I", page, 22, 0)
        if awc.ogg_crc(bytes(page)) != crc:
            raise AssertionError("bad Ogg CRC on page %d" % seq)
        pages.append({"flags": flags, "granule": granule, "serial": serial, "seq": seq,
                      "lacing": lacing, "body": body})
        p = body_start + body_size
    return pages


def tag(typ, offset, size):
    return (typ << 56) | (size << 28) | offset


def format_block(samples, rate, codec, channels):
    block = bytearray(32)
    struct.pack_into("<I", block, 0, samples)
    struct.pack_into("<H", block, 8, rate)
    block[28] = codec
    block[29] = channels
    return bytes(block)


def build_bank(streams):
    """A minimal little-endian AWC: ADAT header, stream table, two tags per stream, then fmt/data.

    streams: [(id, format bytes, data bytes)]. The format bytes are 32 bytes:
    u32 samples, u16 rate at +8, codec at +28, channels at +29.
    """
    n = len(streams)
    header_size = 16 + n * 4 + n * 2 * 8
    body = bytearray()
    entries = []
    for sid, fmt, data in streams:
        fmt_at = header_size + len(body)
        body += fmt
        data_at = header_size + len(body)
        body += data
        entries.append((sid, fmt_at, len(fmt), data_at, len(data)))
    out = bytearray(b"ADAT" + struct.pack("<III", 0, n, header_size))
    out += b"".join(struct.pack("<I", (sid & 0x1FFFFFFF) | (2 << 29))
                    for sid, _, _, _, _ in entries)
    for _, fmt_at, fmt_size, data_at, data_size in entries:
        out += struct.pack("<Q", tag(0x76, fmt_at, fmt_size))
        out += struct.pack("<Q", tag(0x55, data_at, data_size))
    if len(out) != header_size:
        raise AssertionError("stream table/tag layout does not match header_size")
    out += body
    return bytes(out)


def build_music_bank(tocs, samples, rate=48000):
    """A one-stream AWC with a 0x48 streamed-music descriptor and one 2048-byte block.

    tocs: one Opus TOC byte per descriptor channel, each in its own 2048-byte subblock.
    samples: the descriptor's interleaved sample count (per-channel frames * packet channels).
    """
    channels = len(tocs)
    desc = bytearray(12 + 16 * channels)
    struct.pack_into("<III", desc, 0, 1, 2048, channels)
    for i, toc in enumerate(tocs):
        base = 12 + 16 * i
        struct.pack_into("<II", desc, base, 0x100 + i, samples)
        struct.pack_into("<H", desc, base + 10, rate)
        desc[base + 12] = 0x0B
    data = bytearray(2048 + 2048 * channels)
    for i in range(channels):
        struct.pack_into("<I", data, 24 * i + 4, 1)   # one subblock
        struct.pack_into("<H", data, 2048 + 2048 * i, 1)
        data[2050 + 2048 * i] = tocs[i]
    header = 16 + 4 + 2 * 8
    out = bytearray(b"ADAT" + struct.pack("<III", 0, 1, header))
    out += struct.pack("<I", 0x20 | (2 << 29))
    out += struct.pack("<Q", tag(0x48, header, len(desc)))
    out += struct.pack("<Q", tag(0x55, header + len(desc), len(data)))
    return bytes(out + desc + data)


class WriteWavTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_pcm16_fields(self):
        pcm = bytes(i & 0xFF for i in range(400))
        path = self.path("pcm16.wav")
        awc.write_wav(path, pcm, 22050, 2)
        w = read_wav(path)
        self.assertEqual(w["riff_size"], 36 + len(pcm))
        self.assertEqual(w["fmt_size"], 16)
        self.assertEqual(w["format"], 1)
        self.assertEqual(w["channels"], 2)
        self.assertEqual(w["rate"], 22050)
        self.assertEqual(w["byte_rate"], 22050 * 2 * 2)
        self.assertEqual(w["block_align"], 2 * 2)
        self.assertEqual(w["bits"], 16)
        self.assertEqual(w["data_size"], len(pcm))
        self.assertEqual(w["pcm"], pcm)
        self.assertEqual(len(w["data"]), 44 + len(pcm))

    def test_float32_fields(self):
        pcm = struct.pack("<8f", -2.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -1.0)
        path = self.path("float32.wav")
        awc.write_wav(path, pcm, 44100, 2, float32=True)
        w = read_wav(path)
        self.assertEqual(w["riff_size"], 36 + len(pcm))
        self.assertEqual(w["fmt_size"], 16)
        self.assertEqual(w["format"], 3)
        self.assertEqual(w["channels"], 2)
        self.assertEqual(w["rate"], 44100)
        self.assertEqual(w["byte_rate"], 44100 * 2 * 4)
        self.assertEqual(w["block_align"], 2 * 4)
        self.assertEqual(w["bits"], 32)
        self.assertEqual(w["data_size"], len(pcm))
        self.assertEqual(w["pcm"], pcm)
        self.assertEqual(len(w["data"]), 44 + len(pcm))


class OggCrcTest(unittest.TestCase):
    # A fixed 32-byte page: 27-byte header (checksum field zero), one segment of four bytes, body b"test".
    PAGE = bytes.fromhex("4f67675300020000000000000000785634120000000000000000010474657374")

    def test_known_page_crc(self):
        self.assertEqual(len(self.PAGE), 32)
        self.assertEqual(self.PAGE[:4], b"OggS")
        self.assertEqual(awc.ogg_crc(self.PAGE), 0xF7708715)

    def test_payload_changes_crc(self):
        self.assertNotEqual(awc.ogg_crc(self.PAGE[:-1] + b"u"), 0xF7708715)


class OpusSamplesTest(unittest.TestCase):
    def test_toc_patterns(self):
        cases = [
            ("SILK 20 ms, code 0", bytes([0x08]), 960),         # config 1: 960-sample frame, one frame
            ("SILK 40 ms, code 0", bytes([0x10]), 1920),        # config 2: 1920-sample frame, one frame
            ("SILK 40 ms, code 1", bytes([0x11]), 3840),        # two 1920-sample frames
            ("CELT 10 ms, code 0", bytes([0x90]), 480),         # config 18: 480-sample frame, one frame
            ("CELT 20 ms, code 0", bytes([0x98]), 960),         # config 19: 960-sample frame, one frame
            ("CELT 10 ms, code 2", bytes([0x92]), 960),         # two 480-sample frames
            ("CELT 20 ms, code 3", bytes([0x9B, 0x03]), 2880),  # packet[1] & 0x3F == 3 frames of 960
        ]
        for name, packet, expected in cases:
            with self.subTest(name):
                self.assertEqual(awc.opus_samples(packet), expected)


class ExportBankTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out")

    def make_bank(self):
        pcm16 = bytes(i & 0xFF for i in range(100 * 2 * 2))
        floats = struct.pack("<50f", *[((i % 9) - 4) / 4.0 for i in range(50)])
        opus = bytearray(2 * 2048)
        packet_a = bytes([0x02])        # SILK 10 ms, code 2: 2 * 480 = 960 samples
        packet_b = bytes([0x0B, 0x02])  # SILK 20 ms, code 3: 2 * 960 = 1920 samples
        struct.pack_into("<H", opus, 0, len(packet_a))
        opus[2:2 + len(packet_a)] = packet_a
        struct.pack_into("<H", opus, 2048, len(packet_b))
        opus[2050:2050 + len(packet_b)] = packet_b
        bank = build_bank([
            (1, format_block(100, 22050, 0x00, 2), pcm16),
            (2, format_block(50, 48000, 0x02, 1), floats),
            (3, format_block(2880, 48000, 0x0B, 2), bytes(opus)),
        ])
        path = os.path.join(self.tmp.name, "synthetic.awc")
        with open(path, "wb") as f:
            f.write(bank)
        return path, pcm16, floats

    def test_recursive_cli_preserves_bank_paths(self):
        src = Path(self.tmp.name) / "banks"
        samples = {"north/shared": 11, "south/shared": 22, "flat": 33}
        fmt = format_block(1, 48000, 0, 1)
        for stem, sample in samples.items():
            bank = src / (stem + ".awc")
            bank.parent.mkdir(parents=True, exist_ok=True)
            bank.write_bytes(build_bank([(1, fmt, struct.pack("<h", sample))]))

        for jobs in (1, 2):
            with self.subTest(jobs=jobs):
                out = Path(self.out) / str(jobs)
                previous = out / "shared" / "00000001.wav"
                previous.parent.mkdir(parents=True)
                previous.write_bytes(b"previous export")
                subprocess.run([sys.executable, "-I", awc.__file__, src, out, "--jobs", str(jobs)],
                               check=True, capture_output=True)
                for stem, sample in samples.items():
                    self.assertEqual(read_wav(out / stem / "00000001.wav")["pcm"], struct.pack("<h", sample))
                self.assertEqual(previous.read_bytes(), b"previous export")

    def test_cli_bank_limits(self):
        src = Path(self.tmp.name) / "banks"
        src.mkdir()
        fmt = format_block(1, 48000, 0, 1)
        for stem in ("second", "first"):
            (src / (stem + ".awc")).write_bytes(build_bank([(1, fmt, struct.pack("<h", 42))]))

        for limit, expected in ((0, []), (1, ["first"]), (9, ["first", "second"])):
            with self.subTest(limit=limit):
                out = Path(self.out) / str(limit)
                result = subprocess.run([sys.executable, "-I", awc.__file__, src, out, "--limit", str(limit),
                                         "--jobs", "1"], check=True, capture_output=True, text=True)
                self.assertIn(f"{len(expected)} banks:", result.stdout)
                self.assertEqual(sorted(wav.parent.name for wav in out.glob("*/00000001.wav")), expected)

        result = subprocess.run([sys.executable, "-I", awc.__file__, src, self.out, "--limit", "-1"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--limit must be non-negative", result.stderr)
        self.assertFalse((Path(self.out) / "first").exists())

    def test_single_bank_cli_keeps_flat_output(self):
        bank, pcm, _ = self.make_bank()
        subprocess.run([sys.executable, "-I", awc.__file__, bank, self.out, "--jobs", "1"],
                       check=True, capture_output=True)
        self.assertEqual(read_wav(Path(self.out) / "synthetic" / "00000001.wav")["pcm"], pcm)

    def test_pcm16_and_float_and_opus(self):
        bank, pcm16, floats = self.make_bank()
        counts = awc.export_bank(bank, self.out)
        self.assertEqual(counts, {"opus": 1, "pcm": 2, "other": 0, "music": 0})
        od = os.path.join(self.out, "synthetic")

        with self.subTest("PCM16 stream -> WAV"):
            w = read_wav(os.path.join(od, "00000001.wav"))
            self.assertEqual(w["format"], 1)
            self.assertEqual(w["bits"], 16)
            self.assertEqual(w["rate"], 22050)
            self.assertEqual(w["channels"], 2)
            self.assertEqual(w["data_size"], 100 * 2 * 2)      # declared samples * channels * 2 bytes
            self.assertEqual(w["pcm"], pcm16)

        with self.subTest("float32 stream -> float WAV"):
            w = read_wav(os.path.join(od, "00000002.wav"))
            self.assertEqual(w["format"], 3)
            self.assertEqual(w["bits"], 32)
            self.assertEqual(w["rate"], 48000)
            self.assertEqual(w["channels"], 1)
            self.assertEqual(w["data_size"], 50 * 1 * 4)       # declared samples * channels * 4 bytes
            self.assertEqual(w["pcm"], floats)

        with self.subTest("Opus stream -> Ogg Opus"):
            with open(os.path.join(od, "00000003.ogg"), "rb") as f:
                data = f.read()
            pages = parse_ogg_pages(data)
            self.assertGreaterEqual(len(pages), 3)
            head = pages[0]
            self.assertEqual(head["seq"], 0)
            self.assertEqual(head["flags"] & 0x02, 0x02)       # beginning of stream
            self.assertEqual(head["body"][:8], b"OpusHead")
            version, channels, preskip, in_rate, gain, mapping = struct.unpack_from("<BBHIhB", head["body"], 8)
            self.assertEqual((version, channels, preskip, in_rate, gain, mapping), (1, 2, 0, 48000, 0, 0))
            tags = pages[1]
            self.assertEqual(tags["seq"], 1)
            self.assertEqual(tags["body"][:8], b"OpusTags")
            last = pages[-1]
            self.assertEqual(last["flags"] & 0x04, 0x04)       # end of stream
            self.assertEqual(last["granule"], 960 + 1920)      # both packet TOCs, exact end trimming
            self.assertGreater(last["granule"], 0)
            self.assertEqual(last["serial"], 3)


class StreamedMusicTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out")

    def export(self, tocs, samples):
        path = os.path.join(self.tmp.name, "streamed.awc")
        with open(path, "wb") as f:
            f.write(build_music_bank(tocs, samples))
        counts = awc.export_bank(path, self.out)
        return counts, os.path.join(self.out, "streamed")

    def test_stereo_packets_export_one_stereo_ogg(self):
        counts, od = self.export([0xFC], 1920)              # 2 x 960 interleaved samples
        self.assertEqual(counts, {"opus": 0, "pcm": 0, "other": 0, "music": 1})
        with open(os.path.join(od, "00000100.ogg"), "rb") as f:
            pages = parse_ogg_pages(f.read())
        version, channels, preskip, in_rate, gain, mapping = struct.unpack_from("<BBHIhB", pages[0]["body"], 8)
        self.assertEqual((version, channels, preskip, in_rate, gain, mapping), (1, 2, 0, 48000, 0, 0))
        self.assertEqual(pages[-1]["granule"], 960)

    def test_two_mono_groups_export_two_mono_oggs(self):
        counts, od = self.export([0x78, 0x78], 960)
        self.assertEqual(counts, {"opus": 0, "pcm": 0, "other": 0, "music": 2})
        for sid in ("00000100", "00000101"):
            with self.subTest(sid=sid):
                with open(os.path.join(od, sid + ".ogg"), "rb") as f:
                    pages = parse_ogg_pages(f.read())
                channels = pages[0]["body"][9]
                self.assertEqual(channels, 1)
                self.assertEqual(pages[-1]["granule"], 960)

    def test_two_stereo_groups_export_two_stereo_oggs(self):
        counts, od = self.export([0xFC, 0xFC], 1920)
        self.assertEqual(counts, {"opus": 0, "pcm": 0, "other": 0, "music": 2})
        for sid in ("00000100", "00000101"):
            with self.subTest(sid=sid):
                with open(os.path.join(od, sid + ".ogg"), "rb") as f:
                    pages = parse_ogg_pages(f.read())
                self.assertEqual(pages[0]["body"][9], 2)
                self.assertEqual(pages[-1]["granule"], 960)

    def test_music_channels_halves_interleaved_samples(self):
        path = os.path.join(self.tmp.name, "streamed.awc")
        with open(path, "wb") as f:
            f.write(build_music_bank([0xFC], 1920))
        with open(path, "rb") as f:
            d = f.read()
        _, music = awc.parse_awc(d)
        info = awc.music_channels(d, music[0][0], music[0][1])
        self.assertEqual((info[0][1], info[0][5]), (960, 2))

    def test_music_channels_metadata_only_still_reports_packet_channels(self):
        path = os.path.join(self.tmp.name, "streamed.awc")
        with open(path, "wb") as f:
            f.write(build_music_bank([0xFC], 1920))
        with open(path, "rb") as f:
            d = f.read()
        _, music = awc.parse_awc(d)
        info = awc.music_channels(d, music[0][0], music[0][1], collect_packets=False)
        self.assertEqual((info[0][1], info[0][4], info[0][5]), (960, [], 2))


class MalformedBankTest(unittest.TestCase):
    def test_truncated_pcm_bank_creates_no_outputs(self):
        pcm = b"\x01\x00" * 4
        fmt = format_block(4, 48000, 0, 1)
        bank = build_bank([(1, fmt, pcm), (2, fmt, pcm)])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "short.awc")
            with open(path, "wb") as f:
                f.write(bank[:-2])
            out = os.path.join(tmp, "out")
            with self.assertRaisesRegex(ValueError, "0x55.*outside"):
                awc.export_bank(path, out)
            self.assertFalse(os.path.exists(out))

    def test_bad_magic_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.awc")
            with open(path, "wb") as f:
                f.write(b"XATD" + b"\x00" * 60)
            with self.assertRaises(ValueError):
                awc.export_bank(path, os.path.join(tmp, "out"))

    def test_main_skips_bad_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.awc")
            with open(path, "wb") as f:
                f.write(b"XATD" + b"\x00" * 60)
            out = io.StringIO()
            # --jobs 1: the module is loaded under a private name, which worker processes cannot import
            argv = ["awc_opus.py", path, os.path.join(tmp, "out"), "--jobs", "1"]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
                self.assertEqual(awc.main(), 0)
            self.assertIn("skip bad.awc", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
