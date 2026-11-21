"""Synthetic tests for tools/psin_index.py (no game data)."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "psin_index.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import psin_index as P  # noqa: E402


def make_section(tag, payload=b""):
    size = 8 + len(payload)
    return tag.encode("latin-1") + struct.pack(">I", size) + payload


def make_psin(sections, version=22, table_bytes=0, first=None):
    body = b"".join(make_section(tag, payload) for tag, payload in sections)
    first = 16 if first is None else first
    header = bytearray(max(first, 16))
    header[0:4] = b"PSIN"
    struct.pack_into(">I", header, 4, first)
    struct.pack_into("<I", header, 8, version)
    struct.pack_into("<I", header, 12, table_bytes)
    return bytes(header) + body


class TestParse(unittest.TestCase):
    def test_complete_record(self):
        blob = make_psin([
            ("LSCH", b"\x01\x02\x03"),
            ("LENM", b"\x00" * 8),
            ("LGID", b""),
            ("LRSC", b"0DRP\x7a\x00\x00\x00"),
            ("CHKS", b"\x00" * 12),
        ])
        got = P.parse(blob)
        self.assertTrue(got["complete"])
        self.assertEqual(got["version"], 22)
        self.assertEqual([s["tag"] for s in got["sections"]],
                         ["LSCH", "LENM", "LGID", "LRSC", "CHKS"])
        self.assertEqual(got["sections"][2]["size"], 8)
        self.assertEqual(got["sections"][0]["payload"], 3)
        self.assertEqual(got["rsc_version"], 122)
        self.assertEqual(got["checksum_size"], 0)
        self.assertEqual(got["checksum"], "00000000")

    def test_rsc_and_checksum_fields(self):
        blob = make_psin([
            ("LRSC", b"0DRP\x7a\x00\x00\x00"),
            ("CHKS", struct.pack(">I", 5392) + b"\x99\xfe\x66\x9f"),
        ])
        got = P.parse(blob)
        self.assertTrue(got["complete"])
        self.assertEqual(got["rsc_version"], 122)
        self.assertEqual(got["sections"][0]["rsc_magic"], "0DRP")
        self.assertEqual(got["checksum_size"], 5392)
        self.assertEqual(got["checksum"], "99fe669f")

    def test_not_psin(self):
        self.assertIsNone(P.parse(b"NOPE" + b"\x00" * 20))

    def test_first_section_outside_file(self):
        blob = make_psin([("LSCH", b"")], first=16)
        blob = blob[:4] + struct.pack(">I", 9999) + blob[8:]
        got = P.parse(blob)
        self.assertFalse(got["complete"])
        self.assertIn("outside", got["error"])

    def test_truncated_section(self):
        blob = make_psin([("LSCH", b"\x00" * 8)])
        got = P.parse(blob[:-2])
        self.assertFalse(got["complete"])

    def test_zero_size_section(self):
        blob = make_psin([("LSCH", b"")])
        blob = blob[:20] + struct.pack(">I", 0) + blob[24:]
        got = P.parse(blob)
        self.assertFalse(got["complete"])
        self.assertIn("size 0", got["error"])

    def test_trailing_bytes(self):
        blob = make_psin([("LSCH", b"")]) + b"junk"
        got = P.parse(blob)
        self.assertFalse(got["complete"])
        self.assertIn("ended at", got["error"])


class TestCli(unittest.TestCase):
    def test_json_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = make_psin([("LSCH", b""), ("CHKS", b"\x00" * 4)])
            with open(os.path.join(tmp, "00000001.bin"), "wb") as f:
                f.write(good)
            with open(os.path.join(tmp, "00000002.bin"), "wb") as f:
                f.write(b"not a psin")
            p = subprocess.run([sys.executable, "-I", TOOL, tmp, "--json"],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            doc = json.loads(p.stdout)
            self.assertEqual(doc["summary"]["files"], 1)
            self.assertEqual(doc["summary"]["complete"], 1)
            self.assertEqual(doc["summary"]["section_tags"], {"CHKS": 1, "LSCH": 1})


if __name__ == "__main__":
    unittest.main()
