"""Tests for tools/world_census.py, built from synthetic 0x7a resources (no game data).

    python -I tests/test_world_census.py
"""
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WC = load("world_census_under_test", os.path.join(ROOT, "tools", "world_census.py"))
TWI = load("world_index_test_builder", os.path.join(ROOT, "tests", "test_world_index.py"))


def make_archive(path):
    """A small RPF9: normal map, zlib-split map, raw-split map, meta and a duplicate."""
    out = bytearray(0x40)
    out[:4] = b"9FPR"
    struct.pack_into("<I", out, 4, 0x40)

    def stream(blob):
        pos = (len(out) + 15) & ~15
        out.extend(b"\0" * (pos - len(out)))
        out.extend(zlib.compress(blob))

    def raw(blob):
        pos = (len(out) + 15) & ~15
        out.extend(b"\0" * (pos - len(out)))
        out.extend(blob)

    normal = TWI.make_resource(True)
    stream(normal)
    stream(TWI.make_resource(False))
    zsplit = TWI.make_resource(True)
    stream(zsplit[:0x150])
    stream(zsplit[0x150:])
    split = TWI.make_resource(True)
    stream(split[:0x150])
    raw(split[0x150:])
    stream(normal)
    with open(path, "wb") as f:
        f.write(out)


class WorldCensusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def run_census(self, shard_size=1 << 30, jobs=1):
        arc = os.path.join(self.dir, "sample.rpf")
        make_archive(arc)
        out = os.path.join(self.dir, "out")
        code = WC.main([arc, "--out", out, "--shard-size", str(shard_size), "--jobs", str(jobs)])
        with open(os.path.join(out, "summary.json"), encoding="utf-8") as f:
            return code, json.load(f)

    def test_whole_archive_counts_and_coverage(self):
        code, s = self.run_census()
        self.assertEqual(code, 0)
        self.assertEqual(s["maps"], 4)
        self.assertEqual(s["entities"], 4)
        self.assertTrue(s["complete"])
        self.assertTrue(s["census_exact"])
        self.assertEqual(s["coverage"]["accounted"], s["coverage"]["span"])
        self.assertGreater(s["coverage"]["raw_system"], 0)
        self.assertEqual(s["coverage"]["unread"], 0)

    def test_duplicates_are_reported(self):
        _, s = self.run_census()
        self.assertGreaterEqual(s["duplicates"]["content_extra_maps"], 1)
        self.assertGreaterEqual(s["duplicates"]["content_extra_entities"], 1)
        self.assertGreaterEqual(s["duplicates"]["name_groups"], 1)
        self.assertEqual(s["duplicate_offsets"], 0)

    def test_tiny_shards_do_not_double_count(self):
        code, s = self.run_census(shard_size=400, jobs=1)
        self.assertEqual(code, 0)
        self.assertEqual(s["maps"], 4)
        self.assertEqual(s["entities"], 4)
        self.assertEqual(s["duplicate_offsets"], 0)
        self.assertEqual(s["coverage"]["accounted"], s["coverage"]["span"])

    def test_required_len_detects_split_page(self):
        blob = TWI.make_resource(True)
        self.assertEqual(WC.required_len(blob), len(blob))
        self.assertGreater(WC.required_len(blob[:0x150]), 0x150)


class CensusArgumentTest(unittest.TestCase):
    """--shard-size 0 planned shards forever (memory grew without bound); --jobs 0 crashed in ProcessPoolExecutor."""

    def test_expected_shards_rejects_a_size_below_one(self):
        self.assertEqual(WC._expected_shards(0, 10, 4),
                         [{"index": 0, "start": 0, "end": 4}, {"index": 1, "start": 4, "end": 8},
                          {"index": 2, "start": 8, "end": 10}])

        class Counted(int):
            """Counts pos + shard_size, so a missing guard fails here instead of looping forever."""
            adds = 0

            def __radd__(self, other):
                Counted.adds += 1
                if Counted.adds > 1000:
                    raise RuntimeError("shard planning never finished")
                return other + int(self)

        for size in (0, -5):
            Counted.adds = 0
            with self.subTest(size=size), self.assertRaises(ValueError):
                WC._expected_shards(0, 10, Counted(size))

    def test_cli_rejects_shard_size_and_jobs_below_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "sample.rpf")
            make_archive(arc)
            out = os.path.join(tmp, "out")
            tool = os.path.join(ROOT, "tools", "world_census.py")
            for flag, value in (("--shard-size", "0"), ("--shard-size", "-1"), ("--jobs", "0")):
                with self.subTest(flag=flag, value=value):
                    # A subprocess with a timeout, so the old endless loop fails the test instead of hanging it.
                    p = subprocess.run([sys.executable, "-I", tool, arc, "--out", out, flag, value],
                                       capture_output=True, text=True, timeout=10)
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertIn("must be at least 1", p.stderr)
                    self.assertNotIn("Traceback", p.stderr)
            self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
