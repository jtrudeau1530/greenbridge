"""Bad-input errors for the index/carve/audio tools: a message, not a traceback.

    python -I tests/test_bad_input_errors.py

All inputs are built here; no game file or asset is used.
"""
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(tool, *args):
    return subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", tool), *args],
                          capture_output=True, text=True)


class BadInputTest(unittest.TestCase):
    def assert_error(self, p, text):
        self.assertNotEqual(p.returncode, 0, p.stdout)
        self.assertIn(text.lower(), p.stderr.lower())
        self.assertNotIn("traceback", p.stderr.lower())

    def test_empty_archive_in_scan_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, "empty.rpf")
            open(empty, "wb").close()
            for tool in ("anim_index.py", "rbf0_index.py", "restofworld_index.py"):
                with self.subTest(tool=tool):
                    out = os.path.join(tmp, "out.json")
                    p = run(tool, empty, out)
                    self.assert_error(p, "empty file")
                    self.assertFalse(os.path.exists(out), "a failed run must not write an index")

    def test_empty_and_short_archive_in_carve(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, "empty.rpf")
            open(empty, "wb").close()
            self.assert_error(run("rpf9_carve.py", empty, os.path.join(tmp, "out")), "empty file")

            short = os.path.join(tmp, "short.rpf")
            with open(short, "wb") as f:
                f.write(b"RPF9")
            self.assert_error(run("rpf9_carve.py", short, os.path.join(tmp, "out")), "too short")

    def test_carve_start_at_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "empty-data.rpf")
            data = bytearray(0x40)
            data[0:4] = b"9FPR"
            struct.pack_into("<I", data, 4, 0x40)
            with open(arc, "wb") as f:
                f.write(data)
            p = run("rpf9_carve.py", arc, os.path.join(tmp, "out"), "--start", "64")
            self.assert_error(p, "outside the file")

    def test_carve_and_gamedata_index_reject_a_non_rpf9_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "notrpf.rpf")
            with open(arc, "wb") as f:
                f.write(b"MZ" + bytes(0x100))
            self.assert_error(run("rpf9_carve.py", arc, os.path.join(tmp, "out")), "not an RPF9 archive")
            self.assert_error(run("gamedata_index.py", arc), "not an RPF9 archive")

    def test_audio_config_with_no_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = os.path.join(tmp, "empty.rec")
            with open(bank, "wb") as f:
                f.write(bytes(64))
            self.assert_error(run("audio_config.py", bank), "no audio-config records")


if __name__ == "__main__":
    unittest.main()
