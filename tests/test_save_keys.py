"""Tests for tools/save_keys.py: the hash families, key collection from JSAV/ASAV/JSON, and dictionary resolution
(plain, zlib-wrapped and composed names). All names and containers are synthetic; no game data is used.

    python -I tests/test_save_keys.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import save_keys as sk  # noqa: E402
import saves  # noqa: E402


def key(name):
    return "$0x%08x" % sk.joaat(name.encode())


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sk.main(argv)
    return buf.getvalue()


class HashFamilies(unittest.TestCase):
    def test_reference_vectors(self):
        self.assertEqual(sk.joaat(b"a"), 0xCA2E9442)
        self.assertEqual(sk.joaat(b"adder"), 0xB779A091)
        self.assertNotEqual(sk.joaat(b"Adder"), sk.joaat(b"adder"))   # case-sensitive
        self.assertEqual(sk.murmur3_32(b"hello"), 0x248BFA47)
        self.assertEqual(sk.murmur3_x64_64(b"hello"), 0xCBD8A7B341BD9B02)
        self.assertEqual(sk.murmur3_x64_64(b"The quick brown fox jumps over the lazy dog"), 0xE34BBC7BBC071B6C)
        self.assertEqual(sk.fnv1a(b"a"), 0xE40C292C)
        self.assertEqual(sk.fnv1(b"a"), 0x050C5D7E)

    def test_incremental_matches_whole(self):
        self.assertEqual(sk.joaat_final(sk.joaat_step(sk.joaat_step(0, b"Left"), b"Right")), sk.joaat(b"LeftRight"))


class Resolve(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        names = ["WidgetBlock", "GadgetBlock", "Colour", "spinRate", "SprocketCount", "MysteryField"]
        self.names = names
        blocks, hashes = [], []
        for i, n in enumerate(["WidgetBlock", "GadgetBlock"]):
            h = sk.joaat(n.encode())
            hashes.append(h)
            body = {key("Colour"): i, key("spinRate"): 1.5, key("SprocketCount"): {"$x": 1, "$y": 2},
                    key("MysteryField"): 0}
            blocks.append(json.dumps({key("CodeGroup"): {"$0x%08x" % h: body}}, separators=(",", ":")).encode())
        self.jsav = os.path.join(self.dir, "SLOT")
        with open(self.jsav, "wb") as f:
            f.write(saves.build_jsav(3, 2, hashes, [1, 1], blocks))
        # dictionary: some names in clear, one inside a zlib stream, the parts of another as separate words
        self.dict = os.path.join(self.dir, "strings.bin")
        with open(self.dict, "wb") as f:
            f.write(b"\0\0WidgetBlock\0Colour\0spinRate\0\x01CodeGroup\0")
            f.write(zlib.compress(b"<Item>GadgetBlock</Item>"))
            f.write(b"\0Sprocket\0Count\0Sprocket\0Count\0")

    def tearDown(self):
        self.tmp.cleanup()

    def test_collect_counts_index_and_json_keys(self):
        keys = sk.collect_keys([self.jsav])
        # 2 block ids + group + 4 fields
        self.assertEqual(len(keys), 7)
        self.assertEqual(keys[sk.joaat(b"WidgetBlock")], 2)    # index entry + inner key
        self.assertIn("7 distinct keys", run(["collect", self.jsav]))

    def test_collect_from_extracted_folder(self):
        out = os.path.join(self.dir, "x")
        with contextlib.redirect_stdout(io.StringIO()):
            saves.extract(self.jsav, out)
        self.assertEqual(set(sk.collect_keys([out])), set(sk.collect_keys([self.jsav])))

    def test_resolve_counts_per_family(self):
        res = os.path.join(self.dir, "res.json")
        text = run(["resolve", self.jsav, "--dict", self.dict, "--out", res])
        line = {ln.split()[0]: int(ln.split()[1]) for ln in text.splitlines()[1:] if ln.startswith("  ")
                and ln.split()[1].isdigit()}
        self.assertEqual(line["joaat"], 5)          # WidgetBlock, GadgetBlock (zlib), Colour, spinRate, CodeGroup
        self.assertEqual(line["joaat_lower"], 0)
        self.assertEqual(line["fnv1a"], 0)
        with open(res, encoding="utf-8") as f:
            got = json.load(f)
        self.assertEqual(got["%08x" % sk.joaat(b"GadgetBlock")], ["GadgetBlock"])
        self.assertNotIn("%08x" % sk.joaat(b"MysteryField"), got)

    def test_compose_finds_two_word_names(self):
        text = run(["resolve", self.jsav, "--dict", self.dict, "--compose", "50"])
        self.assertIn("joaat, composed       1 more keys", text)       # SprocketCount from Sprocket + Count
        self.assertIn("total (joaat)         6 of 7 keys", text)


if __name__ == "__main__":
    unittest.main()
