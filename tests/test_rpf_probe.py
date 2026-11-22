"""Tests for tools/rpf_probe.py, built from synthetic archive headers (no game data).

    python -I tests/test_rpf_probe.py
"""
import contextlib
import errno
import importlib.util
import io
import json
import os
import struct
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "rpf_probe.py")
    spec = importlib.util.spec_from_file_location("rpf_probe_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load_module()


def archive_bytes(version=b"4FPR", words=(0x10, 0x20, 0x30, 0x40), pad_to=None, tail=b""):
    """Little-endian 'RPF4' magic, four header words, then optional padding and payload."""
    data = version + struct.pack("<4I", *words)
    if pad_to is not None and len(data) < pad_to:
        data += b"\x00" * (pad_to - len(data))
    return data + tail


class EntropyTests(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(probe.entropy(b""), 0.0)

    def test_single_symbol_is_zero(self):
        self.assertEqual(probe.entropy(b"\x00" * 16), 0.0)

    def test_two_equally_frequent_symbols(self):
        self.assertEqual(probe.entropy(b"\x00\x01" * 8), 1.0)

    def test_all_byte_values_once(self):
        self.assertEqual(probe.entropy(bytes(range(256))), 8.0)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def write(self, data, name="a.rpf"):
        path = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_version_and_header_words(self):
        row = probe.probe(self.write(archive_bytes(words=(0x0, 0xFFFFFFFF, 0x1234, 0x10))))
        self.assertEqual(row["version"], "RPF4")
        self.assertEqual(row["header_words"], ["0x0", "0xffffffff", "0x1234", "0x10"])

    def test_natural_order_magic_is_not_a_version(self):
        self.assertEqual(probe.probe(self.write(b"RPF4" + b"\x00" * 16))["version"], None)

    def test_other_magic_has_no_version(self):
        self.assertEqual(probe.probe(self.write(b"DDS " + b"\x00" * 30))["version"], None)

    def test_short_file_has_no_words(self):
        row = probe.probe(self.write(b"4FPR" + b"\x00" * 4))
        self.assertEqual(row["version"], "RPF4")
        self.assertEqual(row["header_words"], [])
        self.assertEqual(row["head_entropy"], 0.0)
        self.assertFalse(row["looks_encrypted"])

    def test_zero_filled_body_is_not_flagged(self):
        row = probe.probe(self.write(archive_bytes(words=(0, 0, 0, 0), pad_to=1024)))
        self.assertEqual(row["head_entropy"], 0.0)
        self.assertFalse(row["looks_encrypted"])

    def test_dense_body_is_flagged(self):
        row = probe.probe(self.write(archive_bytes(tail=bytes(range(256)) * 16)))
        self.assertGreater(row["head_entropy"], 7.9)
        self.assertTrue(row["looks_encrypted"])

    def test_only_the_first_64_kib_is_read(self):
        path = self.write(archive_bytes(pad_to=0x10000, tail=bytes(range(256)) * 16))
        row = probe.probe(path)
        self.assertEqual(row["size"], 0x10000 + 4096)
        self.assertEqual(row["head_entropy"], 0.0)
        self.assertFalse(row["looks_encrypted"])

    def test_size_is_the_full_file(self):
        self.assertEqual(probe.probe(self.write(archive_bytes(pad_to=0x10000)))["size"], 0x10000)


class MainTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.game = os.path.join(self.tmp, "game")
        os.makedirs(self.game)
        self.out = os.path.join(self.tmp, "reports", "rpf.json")

    def write(self, rel, data):
        path = os.path.join(self.game, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def run_main(self):
        out = io.StringIO()
        with mock.patch("sys.argv", ["rpf_probe.py", self.game, "--out", self.out]), contextlib.redirect_stdout(out):
            self.assertEqual(probe.main(), 0)
        return out.getvalue()

    def rows(self):
        with open(self.out, encoding="utf-8") as f:
            return json.load(f)

    def test_collects_rpf_files_recursively_and_sorts_by_size(self):
        small = self.write("a.rpf", archive_bytes())
        large = self.write("sub/b.RPF", archive_bytes(pad_to=4096))
        self.write("sub/notes.txt", b"ignored")
        out = self.run_main()
        rows = self.rows()
        self.assertEqual([os.path.basename(r["file"]) for r in rows], ["b.RPF", "a.rpf"])
        self.assertEqual(os.path.normpath(rows[0]["file"]), os.path.normpath(large))
        self.assertEqual(os.path.normpath(rows[1]["file"]), os.path.normpath(small))
        self.assertIn("2 archives", out)
        self.assertIn("'RPF4': 2", out)

    def test_creates_the_output_directory(self):
        self.write("a.rpf", archive_bytes())
        self.run_main()
        self.assertTrue(os.path.isfile(self.out))

    def test_empty_game_directory(self):
        out = self.run_main()
        self.assertEqual(self.rows(), [])
        self.assertIn("0 archives", out)

    def test_failed_scan_preserves_the_existing_report(self):
        archive = self.write("a.rpf", archive_bytes())
        self.run_main()
        before = self.rows()
        missing = os.path.join(self.tmp, "missing")
        for source in (missing, archive):
            with self.subTest(source=source):
                err = io.StringIO()
                argv = ["rpf_probe.py", source, "--out", self.out]
                with mock.patch("sys.argv", argv), contextlib.redirect_stderr(err):
                    self.assertEqual(probe.main(), 2)
                self.assertIn(source, err.getvalue())
                self.assertEqual(self.rows(), before)

    def test_unreadable_subdirectory_preserves_the_existing_report(self):
        self.write("a.rpf", archive_bytes())
        blocked = os.path.dirname(self.write("blocked/b.rpf", archive_bytes()))
        self.run_main()
        before = self.rows()
        scandir = os.scandir

        def deny_subdirectory(path):
            if path == blocked:
                raise PermissionError(errno.EACCES, "Permission denied", path)
            return scandir(path)

        err = io.StringIO()
        argv = ["rpf_probe.py", self.game, "--out", self.out]
        with (mock.patch("sys.argv", argv), mock.patch.object(os, "scandir", side_effect=deny_subdirectory),
              contextlib.redirect_stderr(err)):
            self.assertEqual(probe.main(), 2)
        self.assertIn(blocked, err.getvalue())
        self.assertEqual(self.rows(), before)


if __name__ == "__main__":
    unittest.main()
