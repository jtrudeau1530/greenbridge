"""Exit codes of a built rpf9dump. Set RPF9DUMP_DLL to its Release DLL to run it."""

import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


def awc_rpf(path):
    # "9FPR" archive holding one AWC bank: one nonstreamed XMA2 (codec 5) mono stream, id 0x1234, 64 data bytes.
    hlen = 16 + 4 + 2 * 8
    tag = lambda t, size, rel: struct.pack("<Q", (t << 56) | (size << 28) | rel)
    fmt = bytearray(32)
    struct.pack_into("<I", fmt, 0, 1000)
    struct.pack_into("<H", fmt, 8, 48000)
    fmt[28], fmt[29] = 5, 1
    bank = (b"ADAT" + struct.pack("<IIII", 0xFF000001, 1, hlen, (2 << 29) | 0x1234)
            + tag(0x76, 32, hlen) + tag(0x55, 64, hlen + 32) + bytes(fmt) + b"\x11" * 64)
    path.write_bytes(b"9FPR" + bytes(0x1000 - 4) + bank + bytes(0x1000))


def run(*args):
    return subprocess.run(["dotnet", os.environ["RPF9DUMP_DLL"], *map(str, args)], capture_output=True, text=True)


@unittest.skipUnless(os.environ.get("RPF9DUMP_DLL"), "requires a built rpf9dump")
class Rpf9dumpExitCodeTest(unittest.TestCase):
    def test_failed_xma_write_exits_nonzero_and_rerun_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "audio.rpf"
            awc_rpf(src)
            wav = root / "out" / "audio" / "xma" / "audio_0001_at_0000000000001000" / "00001234.xma.wav"
            blocker = Path(str(wav) + ".partial")
            blocker.mkdir(parents=True)          # the write of this one stream fails
            failed = run("xma", src, root / "out")
            self.assertNotEqual(failed.returncode, 0, failed.stdout + failed.stderr)
            self.assertFalse(wav.exists())
            blocker.rmdir()
            retry = run("xma", src, root / "out")
            self.assertEqual(retry.returncode, 0, retry.stdout + retry.stderr)
            self.assertEqual(wav.read_bytes()[:4], b"RIFF")
            self.assertEqual(wav.read_bytes()[-64:], b"\x11" * 64)


if __name__ == "__main__":
    unittest.main()
