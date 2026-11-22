"""Synthetic tests for tools/rbf0_encode.py.

    python -I tests/test_rbf0_encode.py

Every byte here is built by the test. No game file, game asset path or game name is used.
"""
import io
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rbf0_encode  # noqa: E402
import rbf0_tree  # noqa: E402

PROLOGUE = rbf0_tree.PROLOGUE


class Stream:
    """Builds an RBF0 record stream; a descriptor gets its name on first use, as the format requires."""

    def __init__(self, wide=True):
        self.wide = wide
        self.out = bytearray(b"RBF0" + (PROLOGUE if wide else b""))
        self.names = []

    def _rec(self, name, kind):
        if name in self.names:
            self.out += bytes((self.names.index(name), kind))
        else:
            self.names.append(name)
            raw = name.encode()
            self.out += bytes((len(self.names) - 1, kind)) + struct.pack("<H", len(raw)) + raw
        return self

    def open(self, name, attrs=0):
        self._rec(name, 0x00)
        self.out += struct.pack("<HHH", 0, 0, attrs)
        return self

    def int(self, name, v):
        self._rec(name, 0x10)
        self.out += struct.pack("<Q" if self.wide else "<I", v)
        return self

    def bool(self, name, v):
        return self._rec(name, 0x20 if v else 0x30)

    def float(self, name, v):
        self._rec(name, 0x40)
        self.out += struct.pack("<d" if self.wide else "<f", v)
        return self

    def vec3(self, name, *v):
        self._rec(name, 0x50)
        self.out += struct.pack("<ddd" if self.wide else "<fff", *v)
        return self

    def string(self, name, raw):
        self._rec(name, 0x60)
        self.out += struct.pack("<H", len(raw)) + raw
        return self

    def text(self, raw):
        self.out += b"\xfd\xff" + struct.pack("<I", len(raw)) + raw
        return self

    def raw(self, b):
        self.out += b
        return self

    def close(self):
        self.out += b"\xff\xff"
        return self

    def bytes(self):
        return bytes(self.out)


def every_record(wide):
    """Every record type, a descriptor reused with another type, nested elements and three text forms."""
    return (Stream(wide).open("root", attrs=2).int("id", 7).string("label", b"probe")
            .open("item", attrs=1).float("scale", 1.5).vec3("pos", 1.0, -2.0, 0.25)
            .bool("on", True).bool("on", False).int("scale", 9)            # "scale" reused as an integer
            .string("tail", b"nul\x00").text(b"body\x00").close()
            .open("item").text(b"no terminator").close()
            .open("item").text(b"two\x00\x00").close()
            .close().bytes())


def roundtrip(blob):
    doc = rbf0_tree.parse_resource(blob)
    assert doc["status"] == "ok", doc["error"]
    return rbf0_encode.encode(doc)


class RoundTrip(unittest.TestCase):
    def test_every_record_wide_and_narrow(self):
        for wide in (True, False):
            blob = every_record(wide)
            self.assertEqual(roundtrip(blob), blob, "wide" if wide else "narrow")

    def test_infinities_are_exact(self):
        for wide in (True, False):
            blob = (Stream(wide).open("r").float("f", math.inf).float("g", -math.inf)
                    .vec3("v", 1.0, math.inf, -math.inf).close().bytes())
            self.assertEqual(roundtrip(blob), blob)

    def test_narrow_float_keeps_its_bits(self):
        blob = Stream(False).open("r").float("f", 0.1).vec3("v", 1e-30, 3.4e38, -0.0).close().bytes()
        self.assertEqual(roundtrip(blob), blob)

    def test_largest_integers(self):
        self.assertEqual(roundtrip(Stream(True).open("r").int("i", 2**64 - 1).close().bytes())[-10:-2],
                         b"\xff" * 8)
        blob = Stream(False).open("r").int("i", 2**32 - 1).close().bytes()
        self.assertEqual(roundtrip(blob), blob)

    def test_through_json(self):
        """The rbf0_tree.py JSON report form: floats and integers survive json.dumps/loads exactly."""
        blob = every_record(True)
        doc = json.loads(json.dumps(rbf0_tree.parse_resource(blob), allow_nan=False))
        self.assertEqual(rbf0_encode.encode(doc), blob)

    def test_an_edited_tree_encodes(self):
        doc = rbf0_tree.parse_resource(every_record(True))
        doc["tree"]["attributes"][1]["value"] = "changed"
        del doc["tree"]["attributes"][1]["byte_length"]           # an edit drops the recorded length
        out = rbf0_encode.encode(doc)
        again = rbf0_tree.parse_resource(out)
        self.assertEqual(again["status"], "ok")
        self.assertEqual(again["tree"]["attributes"][1]["value"], "changed")


class Refusals(unittest.TestCase):
    def reason(self, blob=None, doc=None):
        doc = doc or rbf0_tree.parse_resource(blob)
        with self.assertRaises(rbf0_encode.EncodeError) as cm:
            rbf0_encode.encode(doc)
        return cm.exception.reason

    def test_nan(self):
        self.assertEqual(self.reason(Stream().open("r").float("f", math.nan).close().bytes()), "nan")
        self.assertEqual(self.reason(Stream().open("r").vec3("v", 0.0, math.nan, 1.0).close().bytes()), "nan")

    def test_invalid_utf8(self):
        self.assertEqual(self.reason(Stream().open("r").string("s", b"\xff\xfe").close().bytes()), "utf8")
        self.assertEqual(self.reason(Stream().open("r").text(b"caf\xe9\x00").close().bytes()), "utf8")
        bad_name = b"RBF0" + PROLOGUE + bytes((0, 0)) + struct.pack("<H", 2) + b"\xc3\x28" + bytes(6) + b"\xff\xff"
        self.assertEqual(self.reason(bad_name), "utf8")

    def test_incomplete_parses(self):
        whole = Stream().open("r").int("i", 1).close().bytes()
        self.assertEqual(self.reason(whole[:-3]), "status")              # truncated
        self.assertEqual(self.reason(whole + b"\x00\x00"), "status")     # bytes after the root closed

    def test_attribute_after_content(self):
        blob = Stream().open("r", attrs=1).text(b"t\x00").int("late", 1).close().bytes()
        doc = rbf0_tree.parse_resource(blob)
        self.assertTrue(doc["tree"].get("attribute_order_violation"))
        self.assertEqual(self.reason(doc=doc), "order")

    def test_descriptor_out_of_order(self):
        doc = rbf0_tree.parse_resource(every_record(True))
        doc["tree"]["attributes"][0]["descriptor"] = 40
        self.assertEqual(self.reason(doc=doc), "descriptor")

    def test_length_mismatch(self):
        doc = rbf0_tree.parse_resource(every_record(True))
        doc["tree"]["attributes"][1]["value"] = "longer than recorded"
        self.assertEqual(self.reason(doc=doc), "length")

    def test_integer_too_wide_for_narrow(self):
        doc = rbf0_tree.parse_resource(Stream(False).open("r").int("i", 5).close().bytes())
        doc["tree"]["content"][0]["value"] = 2**32
        self.assertEqual(self.reason(doc=doc), "value")

    def test_float32_overflow_and_direct_nan(self):
        doc = rbf0_tree.parse_resource(Stream(False).open("r").float("f", 1.5).close().bytes())
        doc["tree"]["content"][0]["value"] = 1e40
        self.assertEqual(self.reason(doc=doc), "value")
        doc["tree"]["content"][0]["value"] = math.nan
        self.assertEqual(self.reason(doc=doc), "nan")

    def test_bool_rejected_for_descriptor_and_attrs_declared(self):
        doc = rbf0_tree.parse_resource(every_record(True))
        doc["tree"]["attrs_declared"] = True
        self.assertEqual(self.reason(doc=doc), "value")
        doc = rbf0_tree.parse_resource(every_record(True))
        doc["tree"]["attributes"][0]["descriptor"] = True
        self.assertEqual(self.reason(doc=doc), "descriptor")

    def test_non_dict_doc_and_non_str_text(self):
        self.assertEqual(self.reason(doc="not-a-dict"), "value")
        doc = rbf0_tree.parse_resource(Stream(True).open("r").text(b"hi\x00").close().bytes())
        doc["tree"]["content"][0]["text"] = 123
        self.assertEqual(self.reason(doc=doc), "value")


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.carve = os.path.join(self.tmp.name, "carve")
        os.mkdir(self.carve)

    def put(self, name, data):
        with open(os.path.join(self.carve, name), "wb") as f:
            f.write(data)

    def run_main(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = rbf0_encode.main(list(argv))
        return code, buf.getvalue()

    def test_check_joins_continuations_and_counts(self):
        big = Stream().open("r")
        for i in range(800):
            big.string("s", b"x" * 100)
        big = big.close().bytes()
        self.assertGreater(len(big), 65536)
        self.put("00001000.bin", big[:65536])                          # a 64 KiB stream ...
        self.put("00012000.bin", big[65536:])                          # ... and its continuation
        self.put("00020000.bin", every_record(False))
        self.put("00030000.bin", Stream().open("r").float("f", math.nan).close().bytes())
        code, out = self.run_main("--check", self.carve)
        self.assertEqual(code, 0, out)
        self.assertIn("3 resource(s): 2 exact, 0 different, 1 refused", out)
        self.assertIn("refused (nan): 1", out)
        self.assertIn("narrow-32: 1 of 1 exact", out)

    def test_encode_report_to_new_directory(self):
        self.put("00000100.bin", every_record(True))
        docs, summary = rbf0_tree.decode_source(self.carve)
        report = os.path.join(self.tmp.name, "tree.json")
        rbf0_tree.write_json(report, docs, summary)
        out_dir = os.path.join(self.tmp.name, "out")
        code, out = self.run_main(report, out_dir)
        self.assertEqual(code, 0)
        self.assertIn("1 resource(s) written, 0 refused", out)
        with open(os.path.join(out_dir, "00000100.rbf"), "rb") as f:
            self.assertEqual(f.read(), every_record(True))
        with self.assertRaises(SystemExit):                            # never into a non-empty directory
            self.run_main(report, out_dir)

    def test_argument_errors(self):
        for argv in ([], ["--check", self.carve, "extra"], ["--check", self.carve, "--limit", "0"],
                     ["only-report.json"], ["--check", os.path.join(self.tmp.name, "missing")]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), redirect_stdout(io.StringIO()),                     redirect_stderr(io.StringIO()):
                rbf0_encode.main(argv)

    def test_malformed_reports_raise_clean_system_exit(self):
        cases = {
            "bad-json.json": "{not valid json",
            "list-root.json": "[]",
            "bad-resources.json": json.dumps({"tool": "rbf0_tree", "resources": "bad"}),
        }
        for filename, content in cases.items():
            report = os.path.join(self.tmp.name, filename)
            with open(report, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            out_dir = os.path.join(self.tmp.name, "out-" + filename)
            with self.subTest(filename=filename), self.assertRaises(SystemExit):
                self.run_main(report, out_dir)
            self.assertFalse(os.path.exists(out_dir))

    def test_write_all_handles_non_int_offset_and_non_dict_resource(self):
        doc = rbf0_tree.parse_resource(every_record(True))
        doc["offset"] = "0x100"
        report = os.path.join(self.tmp.name, "mixed.json")
        with open(report, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"tool": "rbf0_tree", "resources": [doc, "not-a-dict"]}, f)
        out_dir = os.path.join(self.tmp.name, "out-mixed")
        code, out = self.run_main(report, out_dir)
        self.assertEqual(code, 0, out)
        self.assertIn("1 resource(s) written, 1 refused", out)
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "00000000.rbf")))


if __name__ == "__main__":
    unittest.main()
