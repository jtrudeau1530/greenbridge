"""Decode a deterministic Ogg Opus to PCM WAV and pin the exact sample count.

    python -I tests/test_audio_wav.py

The test skips when ffmpeg is not on PATH; the tool needs it. It checks the WAV with the standard library's
wave module, and the exact 48,000 frames also pin the granule end-trim (ffmpeg can otherwise keep a partial frame).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "audio_wav.py")
FFMPEG = shutil.which("ffmpeg")


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


if __name__ == "__main__":
    unittest.main()
