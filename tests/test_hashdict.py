"""Unit tests for tools/hashdict.py: Jenkins one-at-a-time (joaat) hashing, string harvesting,
lookup CLI, case-insensitivity, and collision handling.

    python -I tests/test_hashdict.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "hashdict.py")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "tools"))

import hashdict


class JenkinsHashTest(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(hashdict.joaat(""), 0x00000000)

    def test_known_reference_vectors(self):
        # Known Jenkins one-at-a-time reference vectors
        self.assertEqual(hashdict.joaat("a"), 0xCA2E9442)
        self.assertEqual(hashdict.joaat("apple"), 0x88F08EF3)
        self.assertEqual(hashdict.joaat("the quick brown fox jumps over the lazy dog"), 0x2D1643CE)
        # Canonical vehicle name hash documented in docs/design.md
        self.assertEqual(hashdict.joaat("adder"), 0xB779A091)

    def test_case_insensitivity(self):
        # joaat normalizes strings by lowercasing prior to hashing
        self.assertEqual(hashdict.joaat("ADDER"), 0xB779A091)
        self.assertEqual(hashdict.joaat("Adder"), 0xB779A091)
        self.assertEqual(hashdict.joaat("aDdEr"), 0xB779A091)
        self.assertEqual(
            hashdict.joaat("vehicle_model/car01.wft"),
            hashdict.joaat("VEHICLE_MODEL/CAR01.WFT"),
        )
        self.assertEqual(
            hashdict.joaat("Shader_Disasm:CodeBlock"),
            hashdict.joaat("shader_disasm:codeblock"),
        )

    def test_unsigned_32bit_range(self):
        test_strings = [
            "",
            "x",
            "test_token_123",
            "very_long_string_identifier_" * 10,
            "path/to/asset.bin",
        ]
        for s in test_strings:
            h = hashdict.joaat(s)
            self.assertIsInstance(h, int)
            self.assertGreaterEqual(h, 0)
            self.assertLessEqual(h, 0xFFFFFFFF)
            self.assertEqual(h & 0xFFFFFFFF, h)

    def test_latin1_encoding_and_replacement(self):
        # Latin-1 character in range 0-255
        self.assertEqual(hashdict.joaat("café"), hashdict.joaat("caf\xe9"))
        # Character outside Latin-1 range is replaced with '?'
        self.assertEqual(hashdict.joaat("caf\u20ac"), hashdict.joaat("caf?"))


class StringHarvestTest(unittest.TestCase):
    def test_harvest_basic_identifiers(self):
        data = b"\x00\x01first_identifier\x00second_identifier\xff"
        found = set()
        hashdict.harvest(data, found, minlen=3)
        self.assertIn("first_identifier", found)
        self.assertIn("second_identifier", found)

    def test_harvest_delimiters_split(self):
        # Strings are split by [./:\\] to extract nested token pieces
        data = b"\x00folder/sub_dir\\file_item.ext:asset_name\x00"
        found = set()
        hashdict.harvest(data, found, minlen=3)
        self.assertIn("folder/sub_dir\\file_item.ext:asset_name", found)
        self.assertIn("folder", found)
        self.assertIn("sub_dir", found)
        self.assertIn("file_item", found)
        self.assertIn("ext", found)
        self.assertIn("asset_name", found)

    def test_harvest_minlen_threshold(self):
        # Subparts shorter than minlen are filtered out
        data = b"\x00ab.cde.f.ghij\x00"
        found3 = set()
        hashdict.harvest(data, found3, minlen=3)
        self.assertNotIn("ab", found3)
        self.assertNotIn("f", found3)
        self.assertIn("cde", found3)
        self.assertIn("ghij", found3)

        found4 = set()
        hashdict.harvest(data, found4, minlen=4)
        self.assertNotIn("cde", found4)
        self.assertIn("ghij", found4)


class CollisionHandlingTest(unittest.TestCase):
    def test_case_collision_preserves_first_seen(self):
        # Case variants collide to identical hash; setdefault retains first entry
        d = {}
        for s in ["Adder", "adder", "ADDER"]:
            d.setdefault(f"{hashdict.joaat(s):08x}", s)
        self.assertEqual(len(d), 1)
        self.assertEqual(d["b779a091"], "Adder")

    def test_distinct_strings_collision_handling(self):
        # Two distinct synthetic strings producing the same 32-bit joaat hash (0x18c8836d)
        s1 = "str_85_83667"
        s2 = "str_63346_847129"
        self.assertNotEqual(s1, s2)
        h1 = hashdict.joaat(s1)
        h2 = hashdict.joaat(s2)
        self.assertEqual(h1, 0x18C8836D)
        self.assertEqual(h2, 0x18C8836D)

        d = {}
        d.setdefault(f"{h1:08x}", s1)
        d.setdefault(f"{h2:08x}", s2)
        # Collision does not overwrite the existing mapping
        self.assertEqual(len(d), 1)
        self.assertEqual(d["18c8836d"], s1)


class LookupCLITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.hashes_file = os.path.join(self.tmp.name, "hashes.json")
        self.mapping = {
            "b779a091": "adder",
            "88f08ef3": "apple",
            "18c8836d": "str_85_83667",
        }
        with open(self.hashes_file, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable, "-I", TOOL, *args],
            capture_output=True,
            text=True,
        )

    def test_lookup_known_hashes(self):
        p = self.run_tool("--lookup", self.hashes_file, "0xb779a091", "88f08ef3")
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = [line.strip() for line in p.stdout.strip().splitlines() if line.strip()]
        self.assertEqual(lines, [
            "0xb779a091 adder",
            "88f08ef3 apple",
        ])

    def test_lookup_unknown_hash_prints_question_mark(self):
        p = self.run_tool("--lookup", self.hashes_file, "0xdeadbeef")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("0xdeadbeef ?", p.stdout)

    def test_lookup_hex_formatting_variants(self):
        # Supports 0x prefix, uppercase, lowercase, and raw hex
        p = self.run_tool("--lookup", self.hashes_file, "0XB779A091", "0x18c8836d", "88F08EF3")
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = [line.strip() for line in p.stdout.strip().splitlines() if line.strip()]
        self.assertEqual(lines, [
            "0XB779A091 adder",
            "0x18c8836d str_85_83667",
            "88F08EF3 apple",
        ])


class CarveAndExtraCLITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.carve_dir = os.path.join(self.dir, "carve")
        os.makedirs(self.carve_dir)

        # 1. Archive in default --archives list ('common')
        common_dir = os.path.join(self.carve_dir, "common")
        os.makedirs(common_dir)
        common_index = {
            "files": [
                {"off": 0, "head": "00000000", "type": "bin"},
                {"off": 16, "head": "00000000", "type": "bink2"},
            ]
        }
        with open(os.path.join(common_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(common_index, f)
        with open(os.path.join(common_dir, f"{0:012x}.bin"), "wb") as f:
            f.write(b"\x00\x00common_archive_asset\x00\x00")
        with open(os.path.join(common_dir, f"{16:012x}.bin"), "wb") as f:
            f.write(b"\x00\x00bink2_skipped_asset\x00\x00")

        # 2. Archive not in default list, but with metadata head (PSIN: 5053494e)
        meta_dir = os.path.join(self.carve_dir, "custom_pack")
        os.makedirs(meta_dir)
        meta_index = {
            "files": [
                {"off": 32, "head": "5053494e1234", "type": "bin"},
            ]
        }
        with open(os.path.join(meta_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(meta_index, f)
        with open(os.path.join(meta_dir, f"{32:012x}.bin"), "wb") as f:
            f.write(b"\x00\x00pso_metadata_token\x00\x00")

        # 3. Extra directory with filenames
        self.extra_dir = os.path.join(self.dir, "extra_files")
        os.makedirs(self.extra_dir)
        with open(os.path.join(self.extra_dir, "model_prop.mesh"), "w") as f:
            f.write("")

        # 4. Extra standalone file
        self.extra_file = os.path.join(self.dir, "standalone.txt")
        with open(self.extra_file, "wb") as f:
            f.write(b"\x00standalone_symbol_name\x00")

        self.out_json = os.path.join(self.dir, "out_hashes.json")

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable, "-I", TOOL, *args],
            capture_output=True,
            text=True,
        )

    def test_build_dictionary_and_roundtrip_lookup(self):
        p = self.run_tool(
            self.carve_dir,
            "--out", self.out_json,
            "--extra", self.extra_dir, self.extra_file,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(os.path.isfile(self.out_json))

        with open(self.out_json, "r", encoding="utf-8") as f:
            d = json.load(f)

        # Confirm harvested strings are in dictionary under their joaat hex
        self.assertEqual(d.get(f"{hashdict.joaat('common_archive_asset'):08x}"), "common_archive_asset")
        self.assertEqual(d.get(f"{hashdict.joaat('pso_metadata_token'):08x}"), "pso_metadata_token")
        self.assertEqual(d.get(f"{hashdict.joaat('model_prop'):08x}"), "model_prop")
        self.assertEqual(d.get(f"{hashdict.joaat('standalone_symbol_name'):08x}"), "standalone_symbol_name")

        # Bink2 entries should be skipped
        self.assertNotIn(f"{hashdict.joaat('bink2_skipped_asset'):08x}", d)

        # Round-trip lookup via CLI
        lookup_hash = f"0x{hashdict.joaat('model_prop'):08x}"
        p_lookup = self.run_tool("--lookup", self.out_json, lookup_hash)
        self.assertEqual(p_lookup.returncode, 0, p_lookup.stderr)
        self.assertIn(f"{lookup_hash} model_prop", p_lookup.stdout)


if __name__ == "__main__":
    unittest.main()
