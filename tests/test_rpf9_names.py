"""Synthetic tests for tools/rpf9_names.py.

    python -I tests/test_rpf9_names.py

Every byte is built by the test; no game file or game asset name is used.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "rpf9_names.py")


def stream(text):
    """A zlib stream of at least 64 bytes so the scanner accepts it."""
    data = text + b" " * max(0, 64 - len(text))
    return zlib.compress(data)


def make_archive(path, magic=b"RPF9"):
    a = stream(b"prefix alpha.#dr suffix")
    b = stream(b"bravo.#ft")
    c = stream(b"charlie.#dd")
    d = stream(b"no resource names here")
    body = a + b + c + d
    data = bytearray(0x40 + len(body) + 4096)
    data[0:4] = magic
    struct.pack_into("<I", data, 4, 0x40)
    data[0x40:0x40 + len(body)] = body
    with open(path, "wb") as f:
        f.write(data)
    return [a, b, c, d]


class NamesTest(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args],
                              capture_output=True, text=True)

    def test_finds_names_and_counts_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            out = os.path.join(tmp, "index.json")
            make_archive(arc)
            p = self.run_tool(arc, out, "--quiet")
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            summary = doc["summary"]
            self.assertEqual(summary["streams"], 4, summary)
            self.assertEqual(summary["streams_with_names"], 3, summary)
            self.assertEqual(summary["unique_names"], 3, summary)
            self.assertEqual(summary["by_type"], {"dd": 1, "dr": 1, "ft": 1}, summary)
            names = sorted(n for row in doc["files"] for n in row["names"])
            self.assertEqual(names, ["alpha.#dr", "bravo.#ft", "charlie.#dd"])
            self.assertEqual(len(doc["files"]), 3)

    def test_accepts_reversed_magic(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            out = os.path.join(tmp, "index.json")
            make_archive(arc, magic=b"9FPR")
            p = self.run_tool(arc, out, "--quiet")
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["unique_names"], 3, doc["summary"])

    def test_types_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            out = os.path.join(tmp, "index.json")
            make_archive(arc)
            p = self.run_tool(arc, out, "--types", "ft", "--quiet")
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            names = [n for row in doc["files"] for n in row["names"]]
            self.assertEqual(names, ["bravo.#ft"])
            self.assertEqual(doc["summary"]["by_type"], {"ft": 1})

    def test_max_inflate_hides_late_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            out = os.path.join(tmp, "index.json")
            data = b"x" * 40 + b"late.#dr" + b"y" * 40
            a = zlib.compress(data)
            blob = bytearray(0x40 + len(a) + 4096)
            blob[0:4] = b"RPF9"
            struct.pack_into("<I", blob, 4, 0x40)
            blob[0x40:0x40 + len(a)] = a
            with open(arc, "wb") as f:
                f.write(blob)
            p = self.run_tool(arc, out, "--max-inflate", "32", "--quiet")
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["streams_with_names"], 0, doc["summary"])
            self.assertEqual(doc["files"], [])

    def test_invalid_arguments_do_not_write_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            make_archive(arc)
            cases = [
                ("--start", "-1"),
                ("--limit", "0"),
                ("--max-inflate", "0"),
                ("--types", "bad suffix"),
            ]
            for i, args in enumerate(cases):
                out = os.path.join(tmp, f"out{i}.json")
                p = self.run_tool(arc, out, *args)
                self.assertEqual(p.returncode, 2, (args, p.stdout, p.stderr))
                self.assertFalse(os.path.exists(out), args)

    def test_missing_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.json")
            p = self.run_tool(os.path.join(tmp, "missing.rpf"), out, "--quiet")
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
