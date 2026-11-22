"""Synthetic tests for tools/psin_names.py (no game data: every name and hash is invented)."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "psin_names.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psin_names as N  # noqa: E402
from test_psin_data import make_container, ptr  # noqa: E402
from test_psin_index import joaat  # noqa: E402


def mm64(name):
    return N.murmur3_x64_128(name.encode())[0]


ROOT_T = joaat("PropSet")
MODEL, KEY, MODELS, OTHER = joaat("model"), joaat("key"), joaat("models"), joaat("other")
MEMBERS = [
    (0x100, 0, 0x4A, 0, 0, 0),           # 0: element descriptor, u32 hash
    (MODEL, 0x00, 0x4A, 0, 0, 0),        # u32 name hash
    (OTHER, 0x04, 0x4D, 0, 0, 0),        # u32 hash that is not a listed name
    (KEY, 0x08, 0x4F, 0, 0, 0),          # u64 name key
    (MODELS, 0x10, 0x5C, 0, 0, 0),       # {u32 count, u32 capacity, ptr} -> u32 hash[]
]
SCHEMA = [(ROOT_T, 0x3, 0x20, 1, MEMBERS)]
NAMES = ["prop_crate_01", "prop_barrel_02", "Prop_Lamp_03"]


def sample(model="prop_crate_01", key="prop_crate_01", models=("prop_barrel_02", "prop_lamp_03"), other=0x12345678):
    root = bytearray(0x20)
    struct.pack_into("<IIQ", root, 0, joaat(model) if model else 0, other, mm64(key))
    struct.pack_into("<IIQ", root, 0x10, len(models), len(models), ptr(2))
    items = b"".join(struct.pack("<I", joaat(m)) for m in models)
    return make_container([(ROOT_T, bytes(root)), (0x4A, items)], SCHEMA)


class TestHashValues(unittest.TestCase):
    def test_every_hash_member_is_found_with_its_field(self):
        found = N.hash_values(sample())
        fields = [(s, f, t) for s, f, t, _ in found]
        self.assertEqual(fields, [(ROOT_T, MODEL, 0x4A), (ROOT_T, OTHER, 0x4D), (ROOT_T, KEY, 0x4F),
                                  (ROOT_T, MODELS, 0x4A), (ROOT_T, MODELS, 0x4A)])
        self.assertEqual(found[0][3], joaat("prop_crate_01"))
        self.assertEqual(found[2][3], mm64("prop_crate_01"))

    def test_not_a_container(self):
        with self.assertRaises(ValueError):
            N.hash_values(b"PSIN" + b"\0" * 12)


class TestDictionaries(unittest.TestCase):
    def test_u32_is_lower_cased_joaat_and_u64_covers_both_cases(self):
        d32, d64 = N.dictionaries({"Prop_Lamp_03"})
        self.assertIn(joaat("prop_lamp_03"), d32)
        self.assertIn(mm64("Prop_Lamp_03"), d64)
        self.assertIn(mm64("prop_lamp_03"), d64)

    def test_control_is_salted(self):
        d32, d64 = N.dictionaries({"prop_crate_01"}, N.CONTROL_SALT)
        self.assertNotIn(joaat("prop_crate_01"), d32)
        self.assertIn(joaat("prop_crate_01" + N.CONTROL_SALT), d32)
        self.assertEqual(set(d64.values()), {"prop_crate_01"})

    def test_names_from_rpf9_names_json_and_text(self):
        with tempfile.TemporaryDirectory() as d:
            j, t = os.path.join(d, "n.json"), os.path.join(d, "n.txt")
            with open(j, "w") as f:
                json.dump({"files": [{"names": ["prop_crate_01.#dr", "veh_frame.#ft"]}]}, f)
            with open(t, "w") as f:
                f.write("prop_lamp_03\n\n")
            self.assertEqual(N.load_names([j, t]), {"prop_crate_01", "veh_frame", "prop_lamp_03"})


class TestScan(unittest.TestCase):
    def folder(self, d, blobs):
        for i, b in enumerate(blobs):
            with open(os.path.join(d, f"{i:04x}.bin"), "wb") as f:
                f.write(b)
        return d

    def test_counts_and_fields(self):
        with tempfile.TemporaryDirectory() as d:
            self.folder(d, [sample(), sample(model=None, models=()), b"PSIN" + b"\0" * 12, b"not psin"])
            summary, table = N.scan(d, set(NAMES))
        self.assertEqual(summary["PSIN files"], 3)
        self.assertEqual(summary["containers"], 2)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["u32 values"], 6)
        self.assertEqual(summary["u32 resolved"], 3)
        self.assertEqual(summary["u64 resolved"], 2)
        self.assertEqual(summary.get("u32 control", 0) + summary.get("u64 control", 0), 0)
        self.assertEqual(summary["containers with a resolved name"], 2)
        rows = {r["field"]: r for r in table}
        self.assertEqual(rows[f"{MODELS:08x}"]["resolved"], 2)
        self.assertEqual(rows[f"{MODEL:08x}"]["zero"], 1)
        self.assertEqual(rows[f"{OTHER:08x}"]["resolved"], 0)
        self.assertEqual(table[0]["field"], f"{KEY:08x}")       # most resolved first, then by hash

    def test_control_counts_a_planted_salted_value(self):
        with tempfile.TemporaryDirectory() as d:
            self.folder(d, [sample(model="prop_crate_01" + N.CONTROL_SALT)])
            summary, _ = N.scan(d, set(NAMES))
        self.assertEqual(summary["u32 control"], 1)


class TestCli(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)

    def test_text_json_and_refs(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "c")
            os.mkdir(src)
            with open(os.path.join(src, "0001.bin"), "wb") as f:
                f.write(sample())
            names = os.path.join(d, "names.txt")
            with open(names, "w") as f:
                f.write("\n".join(NAMES))
            refs = os.path.join(d, "refs.tsv")
            r = self.run_tool(src, "--names", names, "--refs", refs)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("u32 resolved", r.stdout)
            with open(refs) as f:
                rows = [line.split("\t") for line in f.read().splitlines()]
            self.assertEqual(rows[0][0], "file")
            self.assertEqual(sorted(row[5] for row in rows[1:]),
                             ["Prop_Lamp_03", "prop_barrel_02", "prop_crate_01", "prop_crate_01"])
            r = self.run_tool(src, "--names", names, "--json")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(r.stdout)["summary"]["u64 resolved"], 1)

    def test_rejects_refs_aliasing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "one.bin")
            names = os.path.join(tmp, "names.txt")
            for path, data in ((source, sample()), (names, b"prop_crate_01\n")):
                with open(path, "wb") as f:
                    f.write(data)
            for destination in (source, names):
                with self.subTest(destination=destination):
                    with open(destination, "rb") as f:
                        original = f.read()
                    alias = os.path.join(tmp, ".", os.path.basename(destination))
                    result = self.run_tool(source, "--names", names, "--refs", alias)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("--refs must not overwrite an input", result.stderr)
                    with open(destination, "rb") as f:
                        self.assertEqual(f.read(), original)

    def test_missing_names_file_is_one_line(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_tool(d, "--names", os.path.join(d, "none.txt"))
        self.assertEqual(r.returncode, 2)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
