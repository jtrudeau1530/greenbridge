"""Synthetic tests for tools/world_names.py (no game data).

    python -I tests/test_world_names.py
"""
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tool():
    path = os.path.join(ROOT, "tools", "world_names.py")
    spec = importlib.util.spec_from_file_location("world_names_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WN = load_tool()


class HashTest(unittest.TestCase):
    def test_joaat_case_sensitive_field_names(self):
        # Confirmed by the 0x7a schema member tables.
        self.assertEqual(WN.joaat("CMapData", case_sensitive=True), 0xD3593FA6)
        self.assertEqual(WN.joaat("CEntityDef", case_sensitive=True), 0xCE501483)
        self.assertEqual(WN.joaat("archetypeName", case_sensitive=True), 0xA023A02C)
        self.assertEqual(WN.joaat("name", case_sensitive=True), 0xE6E17F14)
        self.assertEqual(WN.joaat("parent", case_sensitive=True), 0xA99C4651)

    def test_joaat_is_case_insensitive_by_default(self):
        self.assertEqual(WN.joaat("PLAYER"), WN.joaat("player"))
        self.assertNotEqual(WN.joaat("PLAYER", case_sensitive=True),
                            WN.joaat("player", case_sensitive=True))

    def test_murmur3_known_vectors(self):
        self.assertEqual(WN.murmur3_x64_128(b""), (0, 0))
        self.assertEqual(WN.murmur3_x64_128(b"hello"),
                         (0xCBD8A7B341BD9B02, 0x5B1E906A48AE1D19))
        self.assertEqual(WN.murmur3_x64_128(b"The quick brown fox jumps over the lazy dog"),
                         (0xE34BBC7BBC071B6C, 0x7A433CA9C49A9347))

    def test_key64_is_low_murmur_word(self):
        name = "synthetic_key_64"
        self.assertEqual(WN.hash64(name), WN.murmur3_x64_128(name.encode())[0])

    def test_u64_parse_and_format(self):
        self.assertEqual(WN.parse_u64(0x1122334455667788), 0x1122334455667788)
        self.assertEqual(WN.parse_u64("0x1122334455667788"), 0x1122334455667788)
        self.assertEqual(WN.parse_u64("1122334455667788"), 0x1122334455667788)
        self.assertEqual(WN.parse_u64(None), None)
        self.assertEqual(WN.format_u64(0xAB), "00000000000000ab")


class DictionaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_build_from_all_sources(self):
        manifest = self.write("manifest.json", json.dumps({
            "Items": [
                {"Name": "synthetic_prop.#dr"},
                {"Name": "synthetic_prop_lod.#ft"},
            ]
        }))
        scan = self.write("scan.json", json.dumps({
            "entries": [{"Path": "recovered/models/synthetic_cell_a._dr_at_0000000000000010.r9modeldata"}]
        }))
        names = self.write("names.txt", "synthetic_name_a\n# comment\nsynthetic_name_b\n")
        xml = self.write("common.xml",
                         "<root><filename>platform:/levels/synthetic/synthetic_cell_b.rpf</filename>"
                         "<name>SyntheticCellC</name></root>")
        dictionary = WN.build_dictionary([manifest], [scan], [names], [xml])
        counts = dictionary.counts()
        self.assertEqual(counts["exact_names"], 8)
        self.assertEqual(counts["exact_hashes"], 8)
        self.assertEqual(counts["exact_collisions"], 0)
        for name in ("synthetic_prop", "synthetic_prop_lod", "synthetic_cell_a",
                     "synthetic_name_a", "synthetic_name_b", "synthetic_cell_b",
                     "synthetic_cell_b.rpf", "SyntheticCellC"):
            self.assertIn(name, WN._unique_names(dictionary.names))

    def test_directory_manifest_discovery(self):
        models = os.path.join(self.dir, "assets", "archive", "models")
        os.makedirs(models)
        path = os.path.join(models, "manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"Items": [{"Name": "synthetic_dir_model.#dr"}]}, handle)
        dictionary = WN.build_dictionary([os.path.join(self.dir, "assets")])
        self.assertIn("synthetic_dir_model", WN._unique_names(dictionary.names))

    def test_collision_is_ambiguous_without_key64_and_resolved_with_it(self):
        dictionary = WN.NameDictionary()
        dictionary.add("PLAYER")
        dictionary.add("player")
        self.assertEqual(dictionary.counts()["exact_collisions"], 1)
        h = WN.joaat("player")
        ambiguous = dictionary.match(h)
        self.assertEqual(ambiguous["status"], "ambiguous")
        resolved = dictionary.match(h, WN.hash64("player"))
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(resolved["name"], "player")
        self.assertIn("PLAYER", resolved["rejected"])
        resolved_upper = dictionary.match(h, WN.hash64("PLAYER"))
        self.assertEqual(resolved_upper["name"], "PLAYER")

    def test_derived_lod_names_are_separate(self):
        dictionary = WN.NameDictionary()
        dictionary.add("synthetic_lod_base")
        self.assertEqual(dictionary.derive_lod(), 2)
        h = WN.joaat("synthetic_lod_base_lod")
        without = dictionary.match(h, WN.hash64("synthetic_lod_base_lod"))
        self.assertEqual(without["status"], "unresolved")
        with_derived = dictionary.match(h, WN.hash64("synthetic_lod_base_lod"),
                                        include_derived=True)
        self.assertEqual(with_derived["status"], "resolved")
        self.assertTrue(with_derived["derived"])
        self.assertEqual(with_derived["name"], "synthetic_lod_base_lod")

    def test_json_roundtrip(self):
        dictionary = WN.NameDictionary()
        dictionary.add("synthetic_roundtrip")
        dictionary.derive_lod()
        restored = WN.NameDictionary.from_json(json.loads(json.dumps(dictionary.to_json())))
        self.assertEqual(restored.counts()["exact_names"], 1)
        self.assertEqual(restored.counts()["derived_names"], 2)
        self.assertEqual(restored.match(WN.joaat("synthetic_roundtrip"))["name"],
                         "synthetic_roundtrip")


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.dictionary = WN.NameDictionary()
        for name in ("synthetic_map_cell", "synthetic_parent_cell", "synthetic_prop",
                     "synthetic_lod_base"):
            self.dictionary.add(name)
        self.dictionary.derive_lod()

    def test_map_rows(self):
        rows = [
            {"offset": 1, "map_name": WN.joaat("synthetic_map_cell"),
             "parent": WN.joaat("synthetic_parent_cell"),
             "map_name64": WN.format_u64(WN.hash64("synthetic_map_cell"))},
            {"offset": 2, "map_name": WN.joaat("synthetic_parent_cell"), "parent": 0},
            {"offset": 3, "kind": "meta"},
        ]
        stats = WN.resolve_maps(rows, self.dictionary)
        self.assertEqual(stats["rows"], 2)
        self.assertEqual(stats["name_present"], 2)
        self.assertEqual(stats["name_resolved"], 2)
        self.assertEqual(stats["name_verified"], 1)
        self.assertEqual(stats["parent_present"], 1)
        self.assertEqual(stats["parent_resolved"], 1)
        self.assertEqual(stats["parent_links"], 1)
        self.assertEqual(stats["parent_links_resolved"], 1)

    def test_map_64_mismatch_is_unconfirmed(self):
        rows = [{"map_name": WN.joaat("synthetic_map_cell"),
                 "map_name64": WN.format_u64(WN.hash64("synthetic_map_cell") + 1)}]
        stats = WN.resolve_maps(rows, self.dictionary)
        self.assertEqual(stats["name_resolved"], 0)
        self.assertEqual(stats["name_unconfirmed"], 1)

    def test_archetype_counts(self):
        exact = "synthetic_prop"
        derived = "synthetic_lod_base_lod"
        data = {
            "archetypes": {
                f"{WN.joaat(exact):08x}": 7,
                f"{WN.joaat(derived):08x}": 3,
                f"{WN.joaat('synthetic_missing'):08x}": 2,
            },
            "keys": {
                f"{WN.joaat(exact):08x}": WN.format_u64(WN.hash64(exact)),
                f"{WN.joaat(derived):08x}": WN.format_u64(WN.hash64(derived)),
            },
        }
        stats = WN.resolve_archetypes(data, self.dictionary, include_derived=True)
        self.assertEqual(stats["entities"], 12)
        self.assertEqual(stats["unique_hashes"], 3)
        self.assertEqual(stats["resolved_exact_entities"], 7)
        self.assertEqual(stats["resolved_derived_entities"], 3)
        self.assertEqual(stats["verified_entities"], 10)
        self.assertEqual(stats["unresolved_entities"], 2)

    def test_archetype_32_only_is_rejected_by_key64(self):
        dictionary = WN.NameDictionary()
        dictionary.add("PLAYER")
        dictionary.add("player")
        h = WN.joaat("player")
        stats = WN.resolve_archetypes(
            {"archetypes": {f"{h:08x}": 4}, "keys": {f"{h:08x}": WN.format_u64(WN.hash64("player"))}},
            dictionary)
        self.assertEqual(stats["resolved_exact_entities"], 4)
        self.assertEqual(stats["unconfirmed_entities"], 0)
        wrong = WN.resolve_archetypes(
            {"archetypes": {f"{h:08x}": 4}, "keys": {f"{h:08x}": WN.format_u64(WN.hash64("PLAYER") + 1)}},
            dictionary)
        self.assertEqual(wrong["unconfirmed_entities"], 4)


class CliTest(unittest.TestCase):
    def test_build_and_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "manifest.json")
            with open(manifest, "w", encoding="utf-8") as handle:
                json.dump({"Items": [{"Name": "synthetic_cli_prop.#dr"}]}, handle)
            dictionary = os.path.join(tmp, "names.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(WN.main(["build", "--manifest", manifest, "--out", dictionary]), 0)
            with open(dictionary, encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertEqual(data["counts"]["exact_names"], 1)

            maps = os.path.join(tmp, "maps.json")
            with open(maps, "w", encoding="utf-8") as handle:
                json.dump({"maps": [{"map_name": WN.joaat("synthetic_cli_prop")}]}, handle)
            report = os.path.join(tmp, "report.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(WN.main(["resolve", "--dict", dictionary, "--maps", maps,
                                          "--out", report]), 0)
            with open(report, encoding="utf-8") as handle:
                result = json.load(handle)
            self.assertEqual(result["maps"]["name_resolved"], 1)


if __name__ == "__main__":
    unittest.main()
