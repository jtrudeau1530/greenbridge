"""Input checks for tools/tile_probe.py, on a synthetic blob (no game data).

    python -I tests/test_tile_probe.py

The tool needs numpy, so the tests skip without it.
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "tile_probe.py")
HAVE_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAVE_NUMPY, "tile_probe.py needs numpy")
class TileProbeInputTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.blob = os.path.join(self.tmp.name, "blob.bin")
        with open(self.blob, "wb") as f:
            f.write(bytes(range(256)) * 64)  # 16 KiB: smaller blobs give no candidate width

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True, timeout=120)

    def test_missing_file_is_a_message_not_a_traceback(self):
        p = self.run_tool(os.path.join(self.tmp.name, "missing.bin"))
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("missing.bin", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_offset_outside_the_file_is_an_error(self):
        # Before: -5 quietly probed the last five bytes, the file size and more printed nothing, and both exited 0.
        for offset in ("-5", "16384", "99999"):
            with self.subTest(offset=offset):
                p = self.run_tool(self.blob, "--offset", offset)
                self.assertEqual(p.returncode, 1, p.stderr)
                self.assertIn("outside", p.stderr)
                self.assertEqual(p.stdout, "")

    def test_empty_file_is_an_error(self):
        empty = os.path.join(self.tmp.name, "empty.bin")
        open(empty, "wb").close()
        p = self.run_tool(empty)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("0 bytes", p.stderr)

    def test_offset_inside_the_file_still_probes(self):
        p = self.run_tool(self.blob, "--offset", "0")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("score", p.stdout)


if __name__ == "__main__":
    unittest.main()
