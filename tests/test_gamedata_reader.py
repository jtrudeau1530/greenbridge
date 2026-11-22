"""gamedata_reader: the generic conversion of RAGE parser XML, root/type dispatch, streaming and the CLI (no game files).

    python -I tests/test_gamedata_reader.py

Every document is written here by hand, in the conventions docs/gamedata.md describes for common.rpf.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, TOOLS)
import gamedata_reader as R  # noqa: E402

DOC = b"""\xef\xbb\xbf<?xml version="1.0" encoding="UTF-8"?>
<CSyntheticRoot version="3">
  <enabled value="true"/>
  <count value="12"/>
  <mask value="0x1F"/>
  <scale value="1.5"/>
  <bias value="-2.5f"/>
  <limit value="inf"/>
  <label>synthetic_name</label>
  <empty/>
  <offset x="1.0" y="2.5" z="-3"/>
  <tint x="0.5" y="0.25" z="0" w="1"/>
  <ids content="short_array">
    1 2 3
    4 5
  </ids>
  <weights content="float_array">0.5 1.0 2.0</weights>
  <uvs content="vector2_array">0 1 2 3</uvs>
  <basis content="matrix34">
    1 0 0
    0 1 0
    0 0 1
    4 5 6
  </basis>
  <items>
    <Item type="CThing"><size value="2"/></Item>
    <Item type="NULL"/>
    <Item><size value="3"/></Item>
  </items>
  <lookup>
    <Item key="alpha"><v value="1"/></Item>
    <Item key="beta"><v value="2"/></Item>
  </lookup>
  <target ref="other_entry"/>
  <entry name="first" userFlags="3"><v value="1"/></entry>
  <entry name="second" userFlags="0"><v value="2"/></entry>
</CSyntheticRoot>
"""

WANT = {
    "@version": 3, "enabled": True, "count": 12, "mask": 31, "scale": 1.5, "bias": -2.5, "limit": float("inf"),
    "label": "synthetic_name", "empty": "", "offset": (1.0, 2.5, -3.0), "tint": (0.5, 0.25, 0.0, 1.0),
    "ids": [1, 2, 3, 4, 5], "weights": [0.5, 1, 2.0], "uvs": [(0, 1), (2, 3)],
    "basis": [(1, 0, 0), (0, 1, 0), (0, 0, 1), (4, 5, 6)],
    "items": [{"$type": "CThing", "size": 2}, None, {"size": 3}],
    "lookup": {"alpha": {"v": 1}, "beta": {"v": 2}},
    "target": {"$ref": "other_entry"},
    "entry": [{"v": 1, "@name": "first", "@userFlags": 3}, {"v": 2, "@name": "second", "@userFlags": 0}],
}


class Conventions(unittest.TestCase):
    def test_every_convention(self):
        reader = R.Reader(root_handlers={}, type_handlers={})
        tag, typ, value = reader.read(DOC)
        self.assertEqual((tag, typ), ("CSyntheticRoot", None))
        self.assertEqual(value, WANT)
        self.assertEqual(reader.types, {"CThing": 1})
        for kind in ("scalar", "vector", "array", "list", "map", "null", "object", "reference", "struct", "text",
                     "extra attributes"):
            self.assertGreater(reader.stats[kind], 0, kind)

    def test_streaming_equals_whole_tree(self):
        # read() converts each top-level field as it ends and clears it; the result is the whole-tree conversion
        reader = R.Reader(root_handlers={}, type_handlers={})
        whole = reader.convert(ET.fromstring(DOC[3:]))
        self.assertEqual(reader.read(DOC)[2], whole)

    def test_scalars(self):
        for text, want in (("true", True), ("false", False), ("7", 7), ("-7", -7), ("0xff", 255), ("1e3", 1000.0),
                           ("2.0f", 2.0), ("-inf", float("-inf")), ("abc", "abc"), ("1.2.3", "1.2.3")):
            self.assertEqual(R.parse_scalar(text), want, text)
            self.assertIs(type(R.parse_scalar(text)), type(want), text)

    def test_vector_array_must_be_whole(self):
        with self.assertRaises(ValueError):
            R.parse_content("vec3v_array", "1 2 3 4")

    def test_polymorphic_root_and_list_root(self):
        reader = R.Reader(root_handlers={}, type_handlers={})
        self.assertEqual(reader.read(b'<R type="CRootType"><a value="1"/></R>'), ("R", "CRootType",
                                                                                  {"$type": "CRootType", "a": 1}))
        self.assertEqual(reader.read(b'<R><Item value="1"/><Item value="2"/></R>')[2], [1, 2])
        self.assertEqual(reader.read(b'<R>leaf</R>')[2], "leaf")


class Dispatch(unittest.TestCase):
    def test_type_handler(self):
        calls = []

        def thing(el, reader):
            calls.append(el.tag)
            generic = reader.convert(el, use_type_handler=False)
            return ("thing", generic["size"])

        reader = R.Reader(root_handlers={}, type_handlers={"CThing": thing})
        value = reader.read(DOC)[2]
        self.assertEqual(value["items"], [("thing", 2), None, {"size": 3}])
        self.assertEqual(calls, ["Item"])
        self.assertEqual(reader.stats["type handler"], 1)

    def test_root_handler_gets_the_whole_root(self):
        def root(el, reader):
            return sorted({child.tag for child in el})

        reader = R.Reader(root_handlers={"CSyntheticRoot": root}, type_handlers={})
        tag, _, value = reader.read(DOC)
        self.assertEqual(tag, "CSyntheticRoot")
        self.assertIn("lookup", value)
        self.assertEqual(reader.stats["root handler"], 1)

    def test_decorators_register(self):
        saved_r, saved_t = dict(R.ROOT_HANDLERS), dict(R.TYPE_HANDLERS)
        try:
            @R.register_root("CSyntheticRoot")
            def _root(el, reader):
                return "root"

            @R.register_type("CThing")
            def _thing(el, reader):
                return "thing"

            self.assertEqual(R.Reader().read(DOC)[2], "root")
            self.assertEqual(R.Reader().read(b'<A><Item type="CThing"/></A>')[2], ["thing"])
        finally:
            R.ROOT_HANDLERS.clear(); R.ROOT_HANDLERS.update(saved_r)
            R.TYPE_HANDLERS.clear(); R.TYPE_HANDLERS.update(saved_t)


class Run(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = os.path.join(self.tmp.name, "xml")
        os.makedirs(self.dir)
        with open(os.path.join(self.dir, "a.xml"), "wb") as f:
            f.write(DOC)
        with open(os.path.join(self.dir, "b.xml"), "wb") as f:
            f.write(b'<COther><x value="1"/>')                  # truncated, as a tail-cut carve can be
        with open(os.path.join(self.dir, "c.xml"), "wb") as f:
            f.write(b'<COther><x value="2"/></COther>')

    def test_summary_counts(self):
        s = R.run([self.dir], reader=R.Reader(root_handlers={}, type_handlers={}))
        self.assertEqual((s["documents"], s["roots"], s["not_well_formed"]), (2, 2, 1))
        self.assertEqual(s["per_root_documents"], {"CSyntheticRoot": 1, "COther": 1})
        self.assertEqual(s["type_classes"], 1)

    def test_root_filter(self):
        s = R.run([self.dir], root_filter="COther", reader=R.Reader(root_handlers={}, type_handlers={}))
        self.assertEqual(s["documents"], 1)

    def test_cli_prints_counts_and_writes_json(self):
        out = os.path.join(self.tmp.name, "json", "nested")
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "gamedata_reader.py"), self.dir, "--out", out],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("2 documents read, 2 roots, 1 not well-formed", r.stdout)
        self.assertNotIn("synthetic_name", r.stdout)                # counts only, no values
        files = sorted(os.listdir(out))
        self.assertEqual(len(files), 2)
        with open(os.path.join(out, next(f for f in files if f.startswith("CSyntheticRoot"))), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["label"], "synthetic_name")

    def test_cli_missing_target(self):
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "gamedata_reader.py"),
                            os.path.join(self.tmp.name, "nope")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("does not exist", r.stderr)
        self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
