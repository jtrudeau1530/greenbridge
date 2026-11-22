"""Tests for tools/awc_manifest.py, built from synthetic AWC data only (no game files).

    python -I tests/test_awc_manifest.py
"""
import csv, io, json, os, struct, subprocess, sys, tempfile, unittest

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tools", "awc_manifest.py")
HEADER = ["bank", "stream", "codec", "codec_name", "channels", "sample_rate", "declared_samples",
          "duration_s", "streamed_music", "data_offset", "data_size", "bank_size", "notes"]


def fmt(samples, rate, codec, channels):
    block = bytearray(32)
    struct.pack_into("<I", block, 0, samples)
    struct.pack_into("<H", block, 8, rate)
    block[28], block[29] = codec, channels
    return bytes(block)


def music_blob(ids, samples, rate, toc=0xF8):
    """(descriptor, data) for one block of Opus channels. toc is the packet TOC byte."""
    desc = bytearray(12 + 16 * len(ids))
    struct.pack_into("<III", desc, 0, 1, 2048, len(ids))
    for i, cid in enumerate(ids):
        base = 12 + 16 * i
        struct.pack_into("<II", desc, base, cid, samples)
        struct.pack_into("<H", desc, base + 10, rate)
        desc[base + 12] = 0x0B
    data = bytearray(2048 + 2048 * len(ids))          # aligned record table, then channel blocks
    for i in range(len(ids)):
        struct.pack_into("<I", data, 24 * i + 4, 1)   # subblock count; skip stays 0
        struct.pack_into("<H", data, 2048 + 2048 * i, 1)
        data[2050 + 2048 * i] = toc                   # CELT 20 ms, one frame (stereo when bit 2 is set)
    return bytes(desc), bytes(data)


def build_bank(entries):
    """entries: [(id, [(tag type, blob), ...])]; fmt/data blobs follow the header."""
    ntags = sum(len(tags) for _, tags in entries)
    header = 16 + len(entries) * 4 + ntags * 8
    body, placed = bytearray(), []
    for sid, tags in entries:
        recs = []
        for typ, blob in tags:
            recs.append((typ, header + len(body), len(blob)))
            body += blob
        placed.append((sid, recs))
    out = bytearray(b"ADAT" + struct.pack("<III", 0, len(entries), header))
    out += b"".join(struct.pack("<I", (sid & 0x1FFFFFFF) | (len(recs) << 29))
                    for sid, recs in placed)
    out += b"".join(struct.pack("<Q", (typ << 56) | (size << 28) | at)
                    for _, recs in placed for typ, at, size in recs)
    if len(out) != header:
        raise AssertionError("bad synthetic header")
    return bytes(out + body)


def entries(with_music=True):
    opus = bytearray(2048)
    struct.pack_into("<H", opus, 0, 1)
    opus[2] = 0xF8
    out = [(0x10, [(0x76, fmt(4800, 48000, 0x00, 1)), (0x55, bytes(9600))]),
           (0x20, [(0x76, fmt(4410, 44100, 0x02, 2)), (0x55, bytes(35280))]),
           (0x30, [(0x76, fmt(96000, 48000, 0x0B, 2)), (0x55, bytes(opus))])]
    if with_music:
        desc, data = music_blob([0x40, 0x41], 48000, 48000)
        out.append((0x50, [(0x48, desc), (0x55, data)]))
    return out


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write_bank(self, name="alpha.awc", with_music=True):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(build_bank(entries(with_music)))
        return path

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True,
                              text=True, encoding="utf-8", timeout=60)

    def test_csv_rows_codecs_and_bad_bank_skip(self):
        bank = self.write_bank()
        with open(os.path.join(self.tmp.name, "bad.awc"), "wb") as f:
            f.write(b"XATD" + b"\x00" * 64)
        p = self.run_tool(self.tmp.name)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertIn("warning", p.stderr.lower())
        self.assertIn("bad.awc", p.stderr)
        self.assertIn("2 banks, 5 streams, 2 streamed-music channels", p.stderr)
        reader = csv.DictReader(io.StringIO(p.stdout))
        self.assertEqual(reader.fieldnames, HEADER)
        rows = {r["stream"]: r for r in reader}
        self.assertEqual(sorted(rows), ["00000010", "00000020", "00000030", "00000040", "00000041"])
        self.assertEqual({r["bank"] for r in rows.values()}, {"alpha.awc"})
        self.assertEqual(int(rows["00000010"]["bank_size"]), os.path.getsize(bank))
        for sid, codec, name in [("00000010", "0x00", "PCM16"),
                                 ("00000020", "0x02", "PCM float32"),
                                 ("00000030", "0x0B", "Opus")]:
            self.assertEqual((rows[sid]["codec"], rows[sid]["codec_name"]), (codec, name))
        for sid, channels, rate, samples, seconds in [("00000010", 1, 48000, 4800, 0.1),
                                                      ("00000020", 2, 44100, 4410, 0.1),
                                                      ("00000030", 2, 48000, 96000, 2.0),
                                                      ("00000040", 1, 48000, 48000, 1.0)]:
            row = rows[sid]
            self.assertEqual((row["channels"], row["sample_rate"], row["declared_samples"]),
                             (str(channels), str(rate), str(samples)))
            self.assertAlmostEqual(float(row["duration_s"]), seconds, places=6)
        self.assertEqual({s: rows[s]["streamed_music"] for s in ("00000010", "00000040", "00000041")},
                         {"00000010": "no", "00000040": "yes", "00000041": "yes"})
        self.assertEqual((int(rows["00000010"]["data_size"]), int(rows["00000040"]["data_size"])),
                         (9600, 6144))

    def test_json_out_is_inferred_from_extension(self):
        bank = self.write_bank(with_music=False)
        out = os.path.join(self.tmp.name, "nested", "manifest.json")
        p = self.run_tool(bank, "--out", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout, "")
        self.assertIn("1 banks, 3 streams, 0 streamed-music channels", p.stderr)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual((doc["bank_count"], doc["stream_count"]), (1, 3))
        self.assertEqual(doc["streamed_music_channel_count"], 0)
        self.assertEqual([r["stream"] for r in doc["streams"]], ["00000010", "00000020", "00000030"])
        self.assertIs(doc["streams"][0]["streamed_music"], False)
        self.assertAlmostEqual(doc["streams"][0]["duration_s"], 0.1, places=6)

    def test_recursive_report_distinguishes_banks_with_the_same_basename(self):
        for folder in ("left", "right"):
            os.mkdir(os.path.join(self.tmp.name, folder))
            self.write_bank(os.path.join(folder, "shared.awc"), with_music=False)
        p = self.run_tool(self.tmp.name, "--format", "json")
        self.assertEqual(p.returncode, 0, p.stderr)
        rows = json.loads(p.stdout)["streams"]
        self.assertEqual([row["bank"] for row in rows], ["left/shared.awc"] * 3 + ["right/shared.awc"] * 3)

    def test_stereo_streamed_music_uses_packet_channels_and_per_channel_samples(self):
        desc, data = music_blob([0x40], 1920, 48000, toc=0xFC)   # 2 x 960 interleaved samples
        path = os.path.join(self.tmp.name, "stereo.awc")
        with open(path, "wb") as f:
            f.write(build_bank([(0x50, [(0x48, desc), (0x55, data)])]))
        p = self.run_tool(path)
        self.assertEqual(p.returncode, 0, p.stderr)
        rows = {r["stream"]: r for r in csv.DictReader(io.StringIO(p.stdout))}
        row = rows["00000040"]
        self.assertEqual((row["channels"], row["declared_samples"]), ("2", "960"))
        self.assertAlmostEqual(float(row["duration_s"]), 0.02, places=6)
        self.assertEqual(row["streamed_music"], "yes")


if __name__ == "__main__":
    unittest.main(verbosity=2)
