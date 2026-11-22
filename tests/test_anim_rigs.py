"""Synthetic tests for tools/anim_rigs.py.

    python -I tests/test_anim_rigs.py

Every byte here is built by the test: a fake RPF9 data area with two rig dictionaries and one clip dictionary
(the builders of tests/test_anim_clips.py), and a small rig-set JSON in the shape `anim_index.py --rigs` writes.
No game file or game asset name is used.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "anim_rigs.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anim_rigs  # noqa: E402
from anim_index import joaat32  # noqa: E402
from test_anim_clips import Page, two_sequence_dictionary, write_rpf  # noqa: E402

KIND_A, KIND_B = joaat32("kind_a"), joaat32("kind_b")


def rig_block(page, tables, version=2):
    """`RIG` + version, u32 length, 8 filler bytes, then each table {kind, 0xC + 8n, n, word} + n items."""
    body = bytearray(b"\x11" * 8)
    for kind, items in tables:
        body += struct.pack("<4I", kind, 0xC + 8 * len(items), len(items), 0x0300)
        body += b"".join(struct.pack("<II", h, 0x0300 + i) for i, h in enumerate(items))
    return page.alloc(MAGIC_V(version) + struct.pack("<I", len(body)) + bytes(body))


def MAGIC_V(version):
    return b"RIG" + bytes([version])


def rig_dictionary(name, rigs):
    """rigs: {rig path: [(kind, [item hashes])]}. The root's own +0x70 holds `RIG` + version like the real ones."""
    p = Page()
    pm = p.alloc(bytes(16))
    p.b[pm + 8], p.b[pm + 9] = 1, 0
    p.ptr(8, pm)
    p.string(0x38, name)
    struct.pack_into("<I", p.b, 0x68, joaat32(name))
    p.b[0x70:0x74] = MAGIC_V(2)
    items = []
    for path, tables in rigs.items():
        value = p.alloc(bytes(0x90))
        p.ptr(value + 0x20, rig_block(p, tables))
        items.append((joaat32(path), value))
    p.atmap(0x48, items)
    return bytes(p.b)


RIGS = {
    "src@one": {"anim/rig/x/a": [(KIND_A, [1, 2, 3])], "anim/rig/x/b": [(KIND_A, [4]), (KIND_B, [5, 6])]},
    "src@two": {"anim/rig/y/c": [(KIND_B, [7, 8])], "anim/rig/y/unused": [(KIND_A, [9])]},
}
RIGSETS = {"rbf0": {"rigsets": [
    {"key": "anim/rigset/ped/p", "entries": [
        {"source": "src@one", "rig": "anim/rig/x/a", "group": "pose"},
        {"source": "src@one", "rig": "anim/rig/x/b", "group": "deformation"},
        {"source": "src@two", "rig": "anim/rig/y/c", "group": "facial"}]},
    {"key": "anim/rigset/object/o", "entries": [
        {"source": "src@two", "rig": "anim/rig/x/a", "group": "pose"},       # rig is in the other dictionary
        {"source": "src@none", "rig": "anim/rig/z/q", "group": "pose"},      # no dictionary has this name
        {"source": "src@one", "rig": "anim/rig/x/missing", "group": "pose"}]},
]}}


class RigDictionaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rpf = os.path.join(self.tmp.name, "synthetic.rpf")
        self.rigs = os.path.join(self.tmp.name, "rigs.json")
        with open(self.rigs, "w", encoding="utf-8") as f:
            json.dump(RIGSETS, f)
        streams = [rig_dictionary(name, rigs) for name, rigs in RIGS.items()] + [two_sequence_dictionary()]
        self.offsets = write_rpf(self.rpf, streams)

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_reads_only_rig_dictionaries(self):
        dicts, summary = anim_rigs.scan(self.rpf)
        self.assertEqual([d["name"] for d in dicts], ["src@one", "src@two"])
        self.assertEqual([d["offset"] for d in dicts], self.offsets[:2])
        c = summary["counts"]
        self.assertEqual((c["rig_dictionaries"], c["rig_dictionary_hash_matches_name"], c["entries"],
                          c["entries_with_block"], c["entries_key_high_zero"], c["block_version:2"]), (2, 2, 4, 4, 4, 4))
        self.assertEqual((c["tables"], c[f"table_kind:0x{KIND_A:08x}"], c[f"table_kind:0x{KIND_B:08x}"]), (5, 3, 2))
        self.assertEqual((c[f"table_items:0x{KIND_A:08x}"], c[f"table_items:0x{KIND_B:08x}"]), (5, 4))

    def test_entry_keys_and_tables(self):
        dicts, _ = anim_rigs.scan(self.rpf)
        one = {e["key"]: e["tables"] for e in dicts[0]["entries"]}
        self.assertEqual(one, {joaat32("anim/rig/x/a"): [(KIND_A, 3)],
                               joaat32("anim/rig/x/b"): [(KIND_A, 1), (KIND_B, 2)]})

    def test_join_rbf0(self):
        dicts, _ = anim_rigs.scan(self.rpf)
        self.assertEqual(anim_rigs.join_rbf0(dicts, RIGSETS), {
            "distinct_rigs": 6, "in_another_dictionary": 1, "key_missing": 1, "keys": 4,
            "keys_named_by_an_entry": 3, "no_dictionary_named_source": 1, "resolved": 3})

    def test_bad_block_is_reported_not_raised(self):
        page = bytearray(rig_dictionary("src@bad", {"anim/rig/b": [(KIND_A, [1])]}))
        at = page.rfind(b"RIG\x02")
        page[at:at + 3] = b"XYZ"
        rpf = os.path.join(self.tmp.name, "bad.rpf")
        write_rpf(rpf, [bytes(page)])
        dicts, summary = anim_rigs.scan(rpf)
        self.assertEqual(summary["counts"]["rig_dictionaries_unparsed"], 1)
        self.assertIn("no RIG block", dicts[0]["error"])

    def test_cli_writes_counts_only(self):
        out = os.path.join(self.tmp.name, "out.json")
        r = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--rigs", self.rigs],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["rbf0_join"]["resolved"], 3)
        self.assertEqual([row["entries"] for row in doc["rig_dictionaries"]], [2, 2])
        self.assertNotIn("tables", json.dumps(doc["rig_dictionaries"]))


if __name__ == "__main__":
    unittest.main()
