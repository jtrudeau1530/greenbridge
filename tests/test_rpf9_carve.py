"""Regression: rpf9_carve must not overrun a short final read and skip trailing streams."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "rpf9_carve.py")


def make_archive(path):
    a = zlib.compress(b"A" * 1000)
    b = zlib.compress(b"B" * 2000)
    data = bytearray(0x40 + len(a) + len(b) + 4096)
    data[0:4] = b"9FPR"
    struct.pack_into("<I", data, 4, 0x40)
    data[0x40:0x40 + len(a)] = a
    data[0x40 + len(a):0x40 + len(a) + len(b)] = b
    with open(path, "wb") as f:
        f.write(data)
    return a, b


class CarveTrailingStreamTest(unittest.TestCase):
    def test_streams_at_eof_are_all_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            out = os.path.join(tmp, "out")
            a, b = make_archive(arc)
            p = subprocess.run([sys.executable, "-I", TOOL, arc, out, "--index-only"],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["files"], 2, doc["summary"])
            self.assertEqual([r["size"] for r in doc["files"]], [1000, 2000])
            self.assertEqual(doc["files"][0]["stored"], len(a))
            self.assertEqual(doc["files"][1]["stored"], len(b))
            self.assertEqual(doc["files"][1]["off"], 0x40 + len(a))


def carve(test, data):
    """Run rpf9_carve.py on `data`; return its index.json after checking it exited cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        arc = os.path.join(tmp, "x.rpf")
        out = os.path.join(tmp, "out")
        with open(arc, "wb") as f:
            f.write(data)
        p = subprocess.run([sys.executable, "-I", TOOL, arc, out, "--index-only"], capture_output=True, text=True)
        test.assertEqual(p.returncode, 0, p.stderr)
        with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
            return json.load(f)


class CarveEdgeTest(unittest.TestCase):
    def test_bink_signature_cut_off_by_end_of_file(self):
        # A truncated copy can end inside what looks like a Bink 2 header: its fields lie past the end of the file.
        a = zlib.compress(b"A" * 1000)
        data = bytearray(0x40) + a + b"KB2a" + bytes(8)
        data[0:4] = b"9FPR"
        struct.pack_into("<I", data, 4, 0x40)
        doc = carve(self, bytes(data))
        self.assertEqual(doc["summary"]["files"], 1, doc["summary"])
        self.assertEqual(doc["summary"]["unread_bytes"], 12)

    def test_empty_data_area(self):
        data = bytearray(0x40)
        data[0:4] = b"9FPR"
        struct.pack_into("<I", data, 4, 0x40)        # data starts at the end of the file
        doc = carve(self, bytes(data))
        self.assertEqual(doc["summary"]["files"], 0)
        self.assertEqual(doc["summary"]["span"], 0)

    def test_header_data_offset_past_eof_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "x.rpf")
            out = os.path.join(tmp, "out")
            data = bytearray(0x40)
            data[0:4] = b"9FPR"
            struct.pack_into("<I", data, 4, 0x1000)   # data offset past the end of the file
            with open(arc, "wb") as f:
                f.write(data)
            p = subprocess.run([sys.executable, "-I", TOOL, arc, out, "--index-only"],
                               capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("header data offset 4096", p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertFalse(os.path.exists(os.path.join(out, "index.json")))
            self.assertFalse(os.path.exists(out))

    def test_bad_range_arguments_fail_before_the_out_dir_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "x.rpf")
            make_archive(arc)
            for flag, value, message in (("--start", "-1", "must be 0 or more"),
                                         ("--limit", "0", "must be at least 1"),
                                         ("--limit", "-1", "must be at least 1")):
                out = os.path.join(tmp, "out")
                p = subprocess.run([sys.executable, "-I", TOOL, arc, out, "--index-only", flag, value],
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertIn(message, p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                self.assertFalse(os.path.exists(out))

    def test_non_rpf9_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "x.rpf")
            out = os.path.join(tmp, "out")
            with open(arc, "wb") as f:
                f.write(b"MZ" + bytes(0x3E) + zlib.compress(b"A" * 1000))
            p = subprocess.run([sys.executable, "-I", TOOL, arc, out, "--index-only"],
                               capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("x.rpf: not an RPF9 archive", p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertFalse(os.path.exists(out))

    def test_valid_magic_is_accepted_with_limit(self):
        # make_archive() writes the real on-disk magic; a plain run and --limit with the magic check both pass.
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "x.rpf")
            make_archive(arc)
            for extra in ([], ["--limit", "4096"]):
                out = os.path.join(tmp, "out" + str(len(extra)))
                p = subprocess.run([sys.executable, "-I", TOOL, arc, out, "--index-only", *extra],
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stderr)

    def test_hex_limit_is_still_accepted(self):
        # The new converter keeps the repo's decimal-or-0x convention for byte counts.
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "x.rpf")
            out = os.path.join(tmp, "out")
            make_archive(arc)
            p = subprocess.run([sys.executable, "-I", TOOL, arc, out, "--index-only", "--limit", "0x40"],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["summary"]["span"], 0x40)



def noisy(n, seed):
    """Bytes zlib can't shrink, so a stream's stored size is about n."""
    import hashlib
    out = bytearray()
    while len(out) < n:
        out += hashlib.sha256(bytes([seed]) + len(out).to_bytes(4, "little")).digest()
    return bytes(out[:n])


class LimitWindowTest(unittest.TestCase):
    def test_a_stream_crossing_the_limit_is_left_out(self):
        """--limit ending inside a stream: the stream doesn't decode inside the window, so it isn't counted as
        readable, and readable + zero + unread add up to the span (as story_index does since #712)."""
        first = zlib.compress(noisy(4000, 1))
        second = zlib.compress(noisy(4000, 2))
        data = bytearray(0x40 + 0x100) + first + second
        data[0:4] = b"9FPR"
        struct.pack_into("<I", data, 4, 0x40)
        limit = 0x100 + len(first) // 2
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "x.rpf")
            out = os.path.join(tmp, "out")
            with open(arc, "wb") as f:
                f.write(data)
            p = subprocess.run([sys.executable, "-I", TOOL, arc, out, "--index-only", "--limit", str(limit)],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                summary = json.load(f)["summary"]
        self.assertEqual(summary["span"], limit)
        self.assertEqual(summary["files"], 0)
        self.assertEqual(summary["compressed_bytes"] + summary["zero_padding"] + summary["unread_bytes"],
                         summary["span"])

if __name__ == "__main__":
    unittest.main()


def adat_archive(with_bank=True):
    """Config streams, a raw `METADATA` decoy, then (optionally) a zero-stream AWC bank and a stream after it."""
    cfg = [zlib.compress(b"config-%d " % i * 300) for i in range(2)]
    after = zlib.compress(b"after the bank " * 200)
    data = bytearray(b"9FPR" + struct.pack("<I", 0x40) + bytes(0x38))
    for c in cfg:
        data += c
    data += b"\x00" * 16 + b"<METADATA>" + bytes(6)   # ADAT inside METADATA, its "header" is text and zeros
    bank = len(data) + (-len(data)) % 0x10
    data += bytes(bank - len(data))
    if with_bank:
        data += b"ADAT" + struct.pack("<III", 0, 0, 16)   # flags 0, no streams, a 16-byte header: a valid bank
    data += bytes(64) + after + bytes(64)
    return bytes(data), bank


def carve_args(test, data, *args):
    with tempfile.TemporaryDirectory() as tmp:
        arc = os.path.join(tmp, "a.rpf")
        out = os.path.join(tmp, "out")
        with open(arc, "wb") as f:
            f.write(data)
        p = subprocess.run([sys.executable, "-I", TOOL, arc, out, "--index-only", *args], capture_output=True,
                           text=True)
        test.assertEqual(p.returncode, 0, p.stderr)
        with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
            return json.load(f), p.stdout


class CarveUntilAdatTest(unittest.TestCase):
    """--until-adat: the audio config at the start of audio.rpf, ended by the first real AWC bank."""

    def test_stops_at_the_first_valid_bank_not_at_metadata(self):
        data, bank = adat_archive()
        doc, out = carve_args(self, data, "--until-adat")
        self.assertEqual(doc["summary"]["first_adat_bank"], bank)
        self.assertEqual(doc["summary"]["files"], 2, "the stream after the bank must not be carved")
        self.assertEqual(doc["summary"]["span"], bank - 0x40)
        self.assertIn(f"first AWC bank at {bank:#x}", out)
        whole, _ = carve_args(self, data)
        self.assertEqual(whole["summary"]["files"], 3, "without the option the stream after the bank is carved")
        self.assertNotIn("first_adat_bank", whole["summary"])

    def test_no_bank_scans_the_whole_window(self):
        data, _ = adat_archive(with_bank=False)
        doc, out = carve_args(self, data, "--until-adat")
        self.assertIsNone(doc["summary"]["first_adat_bank"])
        self.assertEqual(doc["summary"]["files"], 3)
        self.assertIn("no AWC bank", out)

    def test_a_limit_before_the_bank_wins(self):
        data, bank = adat_archive()
        doc, _ = carve_args(self, data, "--until-adat", "--limit", str(0x20))
        self.assertIsNone(doc["summary"]["first_adat_bank"])
        self.assertEqual(doc["summary"]["span"], 0x20)
