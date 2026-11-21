"""Synthetic-tree tests for tools/rbf0_tree.py.

    python -I tests/test_rbf0_tree.py

Every byte here is built by the test. No game file, game asset path or game name is used.
"""
import json
import os
import struct
import sys
import tempfile
import unittest
from xml.etree import ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rbf0_tree  # noqa: E402

WIDE = b"RBF0" + rbf0_tree.PROLOGUE
CLOSE = b"\xff\xff"


def _name(name):
    return struct.pack("<H", len(name)) + name


def _open(index, name=None, attrs=0):
    out = bytes([index, 0x00])
    if name is not None:
        out += _name(name)
    return out + struct.pack("<HHH", 0, 0, attrs)


def _value(index, kind, payload=b"", name=None):
    out = bytes([index, kind])
    if name is not None:
        out += _name(name)
    return out + payload


def _text(text, nul=True):
    raw = text + (b"\x00" if nul else b"")
    return b"\xfd\xff" + struct.pack("<I", len(raw)) + raw


def wide_blob():
    """Root -> Item(attrs=2) -> Child(text), then a value after the child."""
    b = bytearray(WIDE)
    b += _open(0, b"TestRoot")
    b += _open(1, b"Item", attrs=2)
    b += _value(2, 0x10, struct.pack("<Q", 7), b"count")
    b += _value(3, 0x60, struct.pack("<H", 5) + b"alpha", b"label")
    b += _open(4, b"Child")
    b += _text(b"body")
    b += CLOSE
    b += _value(5, 0x40, struct.pack("<d", 0.5), b"ratio")
    b += CLOSE
    b += _open(6, b"Empty")
    b += CLOSE
    b += CLOSE
    return bytes(b)


def narrow_blob():
    b = bytearray(b"RBF0")
    b += _open(0, b"TestRoot")
    b += _value(1, 0x10, struct.pack("<I", 7), b"count")
    b += _value(2, 0x40, struct.pack("<f", 0.5), b"ratio")
    b += CLOSE
    return bytes(b)


def other_blob():
    b = bytearray(WIDE)
    b += _open(0, b"OtherRoot")
    b += _value(1, 0x20, b"", b"flag")
    b += CLOSE
    return bytes(b)


class ParseTest(unittest.TestCase):
    def test_tree_offsets_and_nesting(self):
        blob = wide_blob()
        doc = rbf0_tree.parse_resource(blob)
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["root_name"], "TestRoot")
        self.assertEqual(doc["layout"], "wide-64")
        self.assertEqual(doc["size"], len(blob))
        self.assertEqual(doc["end"], len(blob))
        self.assertEqual((doc["counts"]["elements"], doc["counts"]["values"], doc["counts"]["texts"]),
                         (4, 3, 1))
        root = doc["tree"]
        self.assertEqual(root["offset"], 8)
        self.assertEqual(root["header_offset"], 20)
        self.assertEqual(root["end"], len(blob))
        item = root["content"][0]
        self.assertEqual(item["name"], "Item")
        self.assertEqual(item["offset"], 26)
        self.assertEqual(item["header_offset"], 34)
        self.assertEqual(item["depth"], 2)
        self.assertEqual(item["attrs_declared"], 2)
        self.assertEqual([a["name"] for a in item["attributes"]], ["count", "label"])
        self.assertEqual([a["value"] for a in item["attributes"]], [7, "alpha"])
        self.assertEqual([n["kind"] for n in item["content"]], ["element", "value"])
        child = item["content"][0]
        self.assertEqual(child["name"], "Child")
        self.assertEqual(child["depth"], 3)
        self.assertEqual(child["content"][0]["kind"], "text")
        self.assertEqual(child["content"][0]["depth"], 4)
        self.assertEqual(child["content"][0]["text"], "body")
        self.assertTrue(child["content"][0]["nul_terminated"])
        self.assertEqual(item["content"][1]["name"], "ratio")
        self.assertEqual(item["content"][1]["value"], 0.5)
        self.assertTrue(item["value_after_child"])
        self.assertEqual(doc["attribute_mismatches"], 0)
        self.assertEqual(doc["attribute_order_violations"], 0)
        self.assertEqual(doc["max_depth"], 3)  # elements only; the text child is at element depth 4

    def test_descriptor_table(self):
        doc = rbf0_tree.parse_resource(wide_blob())
        self.assertEqual(doc["descriptor_count"], 7)
        table = {d["index"]: d for d in doc["descriptors"]}
        self.assertEqual(table[0]["name"], "TestRoot")
        self.assertEqual(table[0]["first_type"], "open")
        self.assertEqual(table[2]["name"], "count")
        self.assertEqual(table[2]["first_type"], "int")
        self.assertEqual(table[3]["name"], "label")
        self.assertEqual(table[3]["first_type"], "string")
        self.assertEqual(table[5]["types"], {"float": 1})
        self.assertEqual(table[3]["uses"], 1)
        self.assertEqual(doc["distinct_descriptor_names"], 7)

    def test_descriptor_is_a_name_index_not_a_type(self):
        b = bytearray(WIDE)
        b += _open(0, b"TestRoot")
        b += _open(1, b"Item", attrs=2)
        b += _value(2, 0x10, struct.pack("<Q", 5), b"value")
        b += _value(2, 0x40, struct.pack("<d", 1.5))
        b += CLOSE
        b += CLOSE
        doc = rbf0_tree.parse_resource(bytes(b))
        self.assertEqual(doc["status"], "ok")
        desc = {d["index"]: d for d in doc["descriptors"]}[2]
        self.assertEqual(desc["name"], "value")
        self.assertEqual(desc["first_type"], "int")
        self.assertEqual(desc["types"], {"int": 1, "float": 1})
        item = doc["tree"]["content"][0]
        self.assertEqual([a["type"] for a in item["attributes"]], ["int", "float"])

    def test_narrow_widths(self):
        doc = rbf0_tree.parse_resource(narrow_blob())
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["layout"], "narrow-32")
        item = doc["tree"]
        self.assertEqual(item["content"][0]["value"], 7)
        self.assertEqual(item["content"][0]["size"], 4)
        self.assertEqual(item["content"][1]["value"], 0.5)
        self.assertEqual(item["content"][1]["size"], 4)

    def test_text_and_string_framing(self):
        doc = rbf0_tree.parse_resource(wide_blob())
        item = doc["tree"]["content"][0]
        label = item["attributes"][1]
        self.assertEqual(label["type"], "string")
        self.assertEqual(label["byte_length"], 5)
        self.assertFalse(label["nul_terminated"])
        self.assertEqual(label["payload_offset"] + label["size"], label["payload_offset"] + 5)
        text = item["content"][0]["content"][0]
        self.assertEqual(text["length"], 5)
        self.assertTrue(text["nul_terminated"])
        self.assertEqual(text["payload_offset"] + text["length"], text["offset"] + 2 + 4 + 5)

    def test_trailing_bytes(self):
        blob = wide_blob() + b"\x00\x00"
        doc = rbf0_tree.parse_resource(blob)
        self.assertEqual(doc["status"], "trailing")
        self.assertEqual(doc["tree"]["end"], len(blob) - 2)
        self.assertIn("after the root closed", doc["error"])

    def test_truncated(self):
        doc = rbf0_tree.parse_resource(wide_blob()[:30])
        self.assertEqual(doc["status"], "truncated")
        self.assertIn("truncated", doc["error"])

    def test_unknown_descriptor_fails_closed(self):
        b = WIDE + _open(0, b"TestRoot") + bytes([9, 0x10]) + struct.pack("<Q", 1) + CLOSE
        doc = rbf0_tree.parse_resource(b)
        self.assertEqual(doc["status"], "bad")
        self.assertIn("descriptor index 9 of 1", doc["error"])
        self.assertEqual(doc["descriptor_count"], 1)
        self.assertEqual(doc["tree"]["name"], "TestRoot")
        self.assertEqual(doc["tree"]["content"], [])
        self.assertEqual(doc["counts"].get("values", 0), 0)

    def test_unknown_record_type_fails_closed(self):
        b = WIDE + _open(0, b"TestRoot") + bytes([1, 0x99]) + _name(b"weird") + CLOSE
        doc = rbf0_tree.parse_resource(b)
        self.assertEqual(doc["status"], "bad")
        self.assertIn("unknown record type 0x99", doc["error"])
        self.assertEqual([d["name"] for d in doc["descriptors"]], ["TestRoot", "weird"])
        self.assertEqual(doc["tree"]["content"], [])

    def test_bad_element_word_fails_closed(self):
        b = WIDE + bytes([0, 0x00]) + _name(b"TestRoot") + struct.pack("<HHH", 1, 0, 0) + CLOSE
        doc = rbf0_tree.parse_resource(b)
        self.assertEqual(doc["status"], "bad")
        self.assertIn("word is not zero", doc["error"])
        self.assertIsNone(doc["tree"])

    def test_not_rbf0(self):
        with self.assertRaises(rbf0_tree.TreeError):
            rbf0_tree.parse_resource(b"PSIN" + bytes(16))


class JoinTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_join_at_64k(self):
        b = bytearray(WIDE)
        b += _open(0, b"BigRoot")
        filler = b"x" * 4000
        b += _value(1, 0x60, struct.pack("<H", len(filler)) + filler, b"filler")
        while len(b) < 70000:
            b += _value(1, 0x60, struct.pack("<H", len(filler)) + filler)
        b += CLOSE
        blob = bytes(b)
        self.assertGreater(len(blob), 65536)
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(blob[:65536])
        with open(os.path.join(self.root, "00010000.bin"), "wb") as f:
            f.write(blob[65536:])
        docs, summary = rbf0_tree.decode_source(self.root)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["chunks"], 2)
        self.assertEqual(docs[0]["size"], len(blob))
        self.assertEqual(docs[0]["status"], "ok")
        self.assertEqual(docs[0]["root_name"], "BigRoot")
        self.assertEqual(summary["resources"], 1)
        self.assertEqual(summary["ok"], 1)

    def test_single_resource_file(self):
        path = os.path.join(self.root, "resource.bin")
        with open(path, "wb") as f:
            f.write(wide_blob())
        docs, summary = rbf0_tree.decode_source(path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["root_name"], "TestRoot")
        self.assertEqual(summary["ok"], 1)


class OutputTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_json_and_xml(self):
        blob = bytearray(WIDE)
        blob += _open(0, b"TestRoot", attrs=1)
        blob += _value(1, 0x60, struct.pack("<H", 3) + b"a &", b"xmlns:xi")
        blob += _open(2, b"Text")
        blob += _text(b"a < b & c")
        blob += CLOSE
        blob += _open(3, b"Ctrl")
        blob += _text(b"a\x01b")
        blob += CLOSE
        blob += CLOSE
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(blob)
        out = os.path.join(self.root, "tree.json")
        xml_dir = os.path.join(self.root, "xml")
        self.assertEqual(rbf0_tree.main([self.root, out, "--xml", xml_dir, "--quiet"]), 0)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["resources"], 1)
        self.assertEqual(doc["summary"]["elements"], 3)
        self.assertEqual(doc["resources"][0]["tree"]["attributes"][0]["name"], "xmlns:xi")
        xml_path = os.path.join(xml_dir, "00000000.xml")
        self.assertTrue(os.path.exists(xml_path))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        self.assertEqual(root.tag, "rbf0")
        attr = root.find(".//attr")
        self.assertEqual(attr.get("name"), "xmlns:xi")
        self.assertEqual(attr.text, "a &")
        texts = root.findall(".//text")
        self.assertEqual(texts[0].text, "a < b & c")
        self.assertEqual(texts[1].text, "a\ufffdb")  # XML 1.0 control bytes are replaced, not emitted

    def test_root_filter_and_limit(self):
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(wide_blob())
        with open(os.path.join(self.root, "00001000.bin"), "wb") as f:
            f.write(other_blob())
        docs, summary = rbf0_tree.decode_source(self.root, roots=["OtherRoot"])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["root_name"], "OtherRoot")
        self.assertEqual(summary["roots"], {"OtherRoot": 1})
        docs, _ = rbf0_tree.decode_source(self.root, limit=1)
        self.assertEqual(len(docs), 1)

    def test_bad_resource_exit_code(self):
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(WIDE + _open(0, b"TestRoot") + bytes([9, 0x10]) + struct.pack("<Q", 1))
        out = os.path.join(self.root, "tree.json")
        self.assertEqual(rbf0_tree.main([self.root, out, "--quiet"]), 1)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["bad"], 1)


if __name__ == "__main__":
    unittest.main()
