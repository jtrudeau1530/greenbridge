"""Check Ogg Opus end trimming, chained audio, and WAV output paths.

    python -I tests/test_audio_wav.py

The decode tests use synthetic audio and skip when ffmpeg is not on PATH.
"""
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "audio_wav.py")
FFMPEG = shutil.which("ffmpeg")

sys.path.insert(0, os.path.join(ROOT, "tools"))
import audio_wav  # noqa: E402
import awc_opus  # noqa: E402


@unittest.skipUnless(FFMPEG, "ffmpeg is not on PATH")
class AudioWavTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.src = os.path.join(self.dir, "in", "one.ogg")
        os.makedirs(os.path.dirname(self.src))
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                        "sine=frequency=440:sample_rate=48000:duration=1",
                        "-ac", "1", "-c:a", "libopus", self.src], check=True)
        self.out = os.path.join(self.dir, "wav")
        self.dst = os.path.join(self.out, "one.wav")

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)

    def test_writes_exact_16_bit_mono_wav(self):
        p = self.run_tool(self.src, "--out", self.out, "--jobs", "1")
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(self.dst, "rb") as f:
            header = f.read(12)
        self.assertEqual(header[:4], b"RIFF")
        self.assertEqual(header[8:], b"WAVE")
        with wave.open(self.dst, "rb") as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getframerate(), 48000)
            self.assertEqual(w.getnframes(), 48000)

    def test_repackaged_stream_keeps_its_declared_end(self):
        # The first data page contains 50 silence packets (48,000 samples). The declared end falls before it.
        declared = 47877
        awc_opus.write_ogg_opus(self.src, [bytes.fromhex("f8fffe")] * 51, 1, 42, declared)
        self.assertEqual(audio_wav.ogg_output_samples(self.src), declared)
        p = self.run_tool(self.src, "--out", self.out)
        self.assertEqual(p.returncode, 0, p.stderr)
        with wave.open(self.dst, "rb") as w:
            self.assertEqual(w.getnframes(), declared)

    def test_chained_links_match_direct_ffmpeg_decode(self):
        for name, tones in [("unequal", [(440, 100), (880, 200)]), ("repeated", [(440, 100)] * 3)]:
            with self.subTest(name=name):
                links = []
                for serial, (hz, ms) in enumerate(tones, 1):
                    tone = f"sine=frequency={hz}:sample_rate=48000:duration={ms / 1000}"
                    link = subprocess.check_output([FFMPEG, "-v", "error", "-f", "lavfi", "-i", tone,
                                                    "-c:a", "libopus", "-fflags", "+bitexact", "-serial_offset",
                                                    str(serial), "-f", "ogg", "pipe:1"])
                    self.assertEqual(int.from_bytes(link[14:18], "little"), serial)
                    links.append(link)
                src = Path(self.dir, name + ".ogg")
                src.write_bytes(b"".join(links))
                reference = os.path.join(self.dir, "reference.wav")
                subprocess.run([FFMPEG, "-v", "error", "-y", "-i", src, "-c:a", "pcm_s16le", reference],
                               check=True, capture_output=True)
                with wave.open(reference, "rb") as w:
                    self.assertEqual(w.getnframes(), sum(ms for _, ms in tones) * 48)
                    pcm = w.readframes(w.getnframes())
                p = self.run_tool(src)
                self.assertEqual(p.returncode, 0, p.stderr)
                with wave.open(str(src.with_suffix(".wav")), "rb") as w:
                    self.assertEqual(w.getnframes() * 2, len(pcm))
                    self.assertTrue(w.readframes(w.getnframes()) == pcm, "decoded PCM differs")

    def test_second_run_skips_the_existing_wav(self):
        self.assertEqual(self.run_tool(self.src, "--out", self.out).returncode, 0)
        p = self.run_tool(self.src, "--out", self.out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("0 WAV written", p.stdout)
        self.assertIn("1 already present", p.stdout)

    def test_decode_preserves_an_existing_wav_at_the_old_temporary_name(self):
        old_tmp = self.dst + ".part.wav"
        os.makedirs(self.out)
        with open(old_tmp, "wb") as f:
            f.write(b"previous output")
        # This input would resume from old_tmp while the other input is decoded.
        shutil.copyfile(self.src, os.path.join(os.path.dirname(self.src), "one.wav.part.opus"))
        p = self.run_tool(os.path.dirname(self.src), "--out", self.out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("1 WAV written, 1 already present", p.stdout)
        with open(old_tmp, "rb") as f:
            self.assertEqual(f.read(), b"previous output")
        self.assertEqual(sorted(os.listdir(self.out)), ["one.wav", "one.wav.part.wav"])

    def test_missing_ffmpeg_is_an_explanation_not_a_traceback(self):
        p = self.run_tool(self.src, "--out", self.out, "--ffmpeg", os.path.join(self.dir, "no-ffmpeg"))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("ffmpeg", p.stderr.lower())
        self.assertNotIn("Traceback", p.stderr)


class GranuleTest(unittest.TestCase):
    def test_trim_requires_identifiable_matching_endpoints(self):
        # A 240-channel family-255 header occupies two laces. Every output maps to the same mono stream.
        head = b"OpusHead" + struct.pack("<BBHIhBBB", 1, 240, 312, 48000, 0, 255, 1, 0) + bytes(240)
        bos = awc_opus.ogg_page(10, 0, 0, [head], 2)
        eos = awc_opus.ogg_page(10, 2, 1272, [b"\xf8\xff\xfe"], 4)
        cases = [
            ("matching", bos, eos, 960),
            ("different serial", bos, awc_opus.ogg_page(11, 2, 1272, [b"x"], 4), 0),
            ("no EOS", bos, awc_opus.ogg_page(10, 2, 1272, [b"x"], 0), 0),
            ("no BOS", awc_opus.ogg_page(10, 0, 0, [head], 0), eos, 0),
            ("embedded signature", awc_opus.ogg_page(10, 0, 0, [b"x" + head], 2), eos, 0),
            ("short header", awc_opus.ogg_page(10, 0, 0, [head[:12]], 2), eos, 0),
            ("split across packets", awc_opus.ogg_page(10, 0, 0, [head[:12], head[12:]], 2), eos, 0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "sample.ogg")
            for name, first, last, expected in cases:
                with self.subTest(name=name):
                    path.write_bytes(first + last)
                    self.assertEqual(audio_wav.ogg_output_samples(path), expected)


# Stands in for ffmpeg: writes the input's file name into the output, so each WAV shows which file it came from.
FAKE_FFMPEG = """import os, sys
args = sys.argv[1:]
src = args[args.index("-i") + 1]
with open(args[-1], "wb") as f:
    f.write(b"FAKE:" + os.path.basename(src).encode())
"""


class SameStemTest(unittest.TestCase):
    """x.ogg and x.opus both decoded to x.wav: two jobs raced on one file and one result was lost."""

    def test_inputs_that_share_a_stem_keep_their_extension(self):
        items = [("d/x.ogg", "x"), ("d/x.opus", "x"), ("d/y.ogg", "y")]
        self.assertEqual(audio_wav.output_names(items),
                         ([("d/x.ogg", "x.ogg"), ("d/x.opus", "x.opus"), ("d/y.ogg", "y")], 2))

    def test_unique_names_are_unchanged(self):
        items = [("a/x.ogg", os.path.join("a", "x")), ("b/x.opus", os.path.join("b", "x")), ("y.opus", "y")]
        self.assertEqual(audio_wav.output_names(items), (items, 0))

    def test_a_name_that_still_clashes_gets_a_counter(self):
        # x.ogg keeps its extension (it clashes with x.opus) and then meets x.ogg.ogg's plain stem.
        items = [("x.ogg", "x"), ("x.ogg.ogg", "x.ogg"), ("x.opus", "x")]
        named, _ = audio_wav.output_names(items)
        self.assertEqual(named, [("x.ogg", "x.ogg"), ("x.ogg.ogg", "x.ogg~2"), ("x.opus", "x.opus")])

    def test_names_that_differ_only_in_case_clash_where_the_file_system_ignores_case(self):
        items = [("x.ogg", "x"), ("X.OPUS", "X")]
        named, renamed = audio_wav.output_names(items)
        if os.path.normcase("X") == os.path.normcase("x"):
            self.assertEqual((named, renamed), ([("x.ogg", "x.ogg"), ("X.OPUS", "X.OPUS")], 2))
        else:
            self.assertEqual((named, renamed), (items, 0))

    def test_output_names_follow_linked_directories_and_counter_collisions(self):
        with tempfile.TemporaryDirectory() as out:
            first = os.path.join(out, "first")
            os.mkdir(first)
            try:
                os.symlink(first, os.path.join(out, "second"), target_is_directory=True)
            except (OSError, NotImplementedError) as e:
                self.skipTest(f"directory symlinks unavailable: {e}")

            # The worker will create child/ later. Both logical paths lead to one physical WAV.
            a = os.path.join("first", "child", "track")
            b = os.path.join("second", "child", "track")
            c = os.path.join("independent", "child", "track")
            items = [(a + ".ogg", a), (b + ".ogg", b), (c + ".opus", c)]
            expected = [(a + ".ogg", a + ".ogg"), (b + ".ogg", b + ".ogg~2"), (c + ".opus", c)]
            self.assertEqual(audio_wav.output_names(items, out), (expected, 2))
            self.assertFalse(os.path.exists(os.path.join(first, "child")))
            os.mkdir(os.path.join(first, "child"))
            self.assertEqual(audio_wav.output_names(items, out), (expected, 2))

    def test_every_input_gets_its_own_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in")
            out = os.path.join(tmp, "out")
            os.makedirs(src)
            for name in ("x.ogg", "x.opus", "y.ogg"):
                with open(os.path.join(src, name), "wb") as f:
                    f.write(b"OggS" + b"\0" * 24 + b"OpusHead" + bytes(64))
            script = os.path.join(tmp, "fake_ffmpeg.py")
            with open(script, "w", encoding="utf-8") as f:
                f.write(FAKE_FFMPEG)
            if os.name == "nt":
                ffmpeg = os.path.join(tmp, "ffmpeg.cmd")
                with open(ffmpeg, "w", encoding="utf-8") as f:
                    f.write(f'@"{sys.executable}" "{script}" %*\n')
            else:
                ffmpeg = os.path.join(tmp, "ffmpeg")
                with open(ffmpeg, "w", encoding="utf-8") as f:
                    f.write(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
                os.chmod(ffmpeg, 0o755)

            run = [sys.executable, "-I", TOOL, src, "--out", out, "--jobs", "3", "--ffmpeg", ffmpeg]
            p = subprocess.run(run, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("3 WAV written", p.stdout)
            self.assertIn("2 file(s) share a name", p.stdout)
            got = {}
            for name in sorted(os.listdir(out)):
                with open(os.path.join(out, name), "rb") as f:
                    got[name] = f.read()
            self.assertEqual(got, {"x.ogg.wav": b"FAKE:x.ogg", "x.opus.wav": b"FAKE:x.opus", "y.wav": b"FAKE:y.ogg"})

            p = subprocess.run(run, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("0 WAV written, 3 already present", p.stdout)


if __name__ == "__main__":
    unittest.main()
