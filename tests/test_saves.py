"""Tests for tools/saves.py: ASAV/JSAV/SLAYER1 parsing, the byte-exact JSAV rebuild, extract/pack, diff, and the
import into the offline XGameSave folder layout. Every container here is synthetic, built by the test itself; no real
save data is used or quoted.

    python -I tests/test_saves.py
"""
import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "saves.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import saves  # noqa: E402

CODE_GROUP = "$0x11111111"
SCRIPT_GROUP = "$0x22222222"


def block(group, h, body):
    """One synthetic block in the retail shape: {"<group>":{"$0x<hash>":{...}}}, compact JSON."""
    return json.dumps({group: {"$0x%08x" % h: body}}, separators=(",", ":")).encode()


def make_jsav(n_code=3, n_script=2, seed=0):
    hashes, blocks = [], []
    for i in range(n_code + n_script):
        h = 0x10000000 + seed * 0x100 + i
        group = CODE_GROUP if i < n_code else SCRIPT_GROUP
        body = {"$0xaaaa0000": i, "$0xbbbb0000": {"$x": 1.5, "$y": -2.25, "$z": 0.0},
                "$0xcccc0000": {"$content": "int_array", "$data": list(range(i + 1))}}
        hashes.append(h)
        blocks.append(block(group, h, body))
    return saves.build_jsav(3, n_code, hashes, [1] * len(blocks), blocks), hashes, blocks


def make_asav(kind=3, title="Slot title", slot="SLOT0", payload=b"\x0a\x00\x00\x00", ts=1700000000):
    d = bytearray(saves.ASAV_HDR)
    struct.pack_into("<4sIIIIIII", d, 0, b"ASAV", 1, kind, 1, 0, saves.ASAV_HDR, 0, saves.ASAV_HDR + len(payload))
    struct.pack_into("<Q", d, 0x20, ts)
    t = title.encode("utf-16le")
    d[0x28:0x28 + len(t)] = t
    s = slot.encode("ascii")
    d[0x128:0x128 + len(s)] = s
    struct.pack_into("<IIQ", d, 0x160, 0x12345678, 0, saves.ASAV_HDR + len(payload))
    return bytes(d) + payload


def make_slayer(name, inner):
    n = name.encode("ascii")
    return b"SLAYER1\0" + struct.pack("<II", 1, len(n)) + n + struct.pack("<Q", len(inner)) + inner


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def rd(path):
    with open(path, "rb") as f:
        return f.read()


def rd_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        fn(*args)
    return out.getvalue()


class JsavTest(unittest.TestCase):
    def test_layout_and_framing(self):
        d, hashes, blocks = make_jsav()
        h, es = saves.parse_jsav(d)
        self.assertEqual((h["version"], h["code_blocks"], h["script_blocks"]), (3, 3, 2))
        self.assertEqual(h["body_offset"] % 16, 0)
        self.assertEqual(len(d) % 16, 0)
        self.assertTrue(d[h["body_offset"]:].startswith(b'{"CodeSaveBlocks":[\r\n'))
        self.assertIn(b'],\r\n"ScriptSaveBlocks":[\r\n', d)
        self.assertTrue(d.rstrip(b"\0").endswith(b"]\r\n}"))
        self.assertEqual([int(e["hash"], 16) for e in es], hashes)
        self.assertEqual(saves.jsav_blocks(d, es), blocks)
        self.assertEqual({e["flags"] for e in es} | {e["reserved"] for e in es}, {1, 0})

    def test_body_is_one_json_document(self):
        d, _, _ = make_jsav()
        h, _ = saves.parse_jsav(d)
        doc = json.loads(d[h["body_offset"]:h["body_end"]])
        self.assertEqual(len(doc["CodeSaveBlocks"]), 3)
        self.assertEqual(len(doc["ScriptSaveBlocks"]), 2)

    def test_rebuild_is_byte_identical(self):
        d, _, _ = make_jsav(5, 4)
        h, es = saves.parse_jsav(d)
        again = saves.build_jsav(h["version"], h["code_blocks"], [int(e["hash"], 16) for e in es],
                                 [e["flags"] for e in es], saves.jsav_blocks(d, es))
        self.assertEqual(again, d)

    def test_edge_counts(self):
        for n_code, n_script in ((0, 2), (2, 0), (1, 1)):
            d, _, blocks = make_jsav(n_code, n_script)
            h, es = saves.parse_jsav(d)
            self.assertEqual(saves.jsav_blocks(d, es), blocks)
            json.loads(d[h["body_offset"]:h["body_end"]])

    def test_verify_cli(self):
        d, _, _ = make_jsav()
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "s.jsav")
            write(p, d)
            r = subprocess.run([sys.executable, "-I", TOOL, "verify", p], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("byte-identical", r.stdout)
            write(p, d[:-16] + b"x" * 16)          # damage the body
            r = subprocess.run([sys.executable, "-I", TOOL, "verify", p], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)

    def test_extract_edit_pack(self):
        d, _, _ = make_jsav()
        with tempfile.TemporaryDirectory() as tmp:
            src, out, packed = os.path.join(tmp, "s.jsav"), os.path.join(tmp, "x"), os.path.join(tmp, "p.jsav")
            write(src, d)
            quiet(saves.extract, src, out)
            idx = rd_json(os.path.join(out, "index.json"))
            self.assertEqual(len(idx["entries"]), 5)
            self.assertTrue(os.path.isfile(os.path.join(out, "save.pretty.json")))
            quiet(saves.pack, out, packed)
            self.assertEqual(rd(packed), d)
            # an edited block of a different length still packs to a valid file
            f = os.path.join(out, idx["entries"][1]["file"])
            write(f, block(CODE_GROUP, 0x10000001, {"$0xaaaa0000": 123456789}))
            quiet(saves.pack, out, packed)
            h, es = saves.parse_jsav(rd(packed))
            self.assertEqual(json.loads(saves.jsav_blocks(rd(packed), es)[1]),
                             {CODE_GROUP: {"$0x10000001": {"$0xaaaa0000": 123456789}}})

    def test_diff(self):
        a, _, _ = make_jsav(seed=0)
        h, es = saves.parse_jsav(a)
        blocks = saves.jsav_blocks(a, es)
        blocks[2] = blocks[2].replace(b'"$0xaaaa0000":2', b'"$0xaaaa0000":7')
        b = saves.build_jsav(3, 3, [int(e["hash"], 16) for e in es], [1] * 5, blocks)
        with tempfile.TemporaryDirectory() as tmp:
            pa, pb = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            write(pa, a)
            write(pb, b)
            out = quiet(saves.diff, pa, pb)
        self.assertIn("changed: code 10000002", out)
        self.assertIn("4 blocks identical", out)


class AsavTest(unittest.TestCase):
    def test_header_fields(self):
        d = make_asav(kind=2, title="Profile", slot="SettingsX", payload=b'{"$v": 2.0}')
        h = saves.parse_asav(d)
        self.assertEqual((h["version"], h["kind"], h["payload_offset"], h["field_1c"]), (1, 2, 0x170, len(d)))
        self.assertEqual((h["title"], h["slot_name"], h["checksum_160"]), ("Profile", "SettingsX", "12345678"))
        self.assertEqual(h["timestamp_utc"], "2023-11-14 22:13:20 UTC")
        self.assertEqual(saves.asav_payload(d, h), b'{"$v": 2.0}')

    def test_extract_text_payload(self):
        d = make_asav(kind=2, payload=b'{\r\n\t"$0x01020304": true\r\n}')
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "a"), os.path.join(tmp, "x")
            write(src, d)
            quiet(saves.extract, src, out)
            self.assertEqual(rd_json(os.path.join(out, "payload.json")), {"$0x01020304": True})
            self.assertEqual(rd(os.path.join(out, "header.bin")), d[:0x170])

    def test_rejects_other_magic(self):
        with self.assertRaises(ValueError):
            saves.parse_asav(b"JSAV" + bytes(0x200))


class SlayerTest(unittest.TestCase):
    def test_wrapper_and_unaligned_name(self):
        inner = make_asav(kind=1, title="Packed", slot="SLOT7") + os.urandom(64)
        for name in ("ABCD0000", "memory.dat"):          # 8 and 10 bytes: the second leaves the ASAV unaligned
            d = make_slayer(name, inner)
            h = saves.parse_slayer(d)
            self.assertEqual((h["name"], h["payload_size"]), (name, len(inner)))
            self.assertEqual(d[h["payload_offset"]:], inner)
            self.assertEqual(saves.parse_asav(d[h["payload_offset"]:])["slot_name"], "SLOT7")

    def test_info_reports_without_decoding(self):
        d = make_slayer("ABCD0000", make_asav(kind=1, title="Packed") + os.urandom(64))
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "packed.dat")
            write(p, d)
            out = quiet(saves.info, [p])
        self.assertIn("SLAYER1", out)
        self.assertIn("not decoded", out)


class ImportTest(unittest.TestCase):
    def test_encoding_matches_the_shim(self):
        # Same rule as encode_name() in runtime/shims/xgameruntime_services.cpp.
        self.assertEqual(saves.xgs_encode("slot 1/main"), "slot%201%2Fmain")
        self.assertEqual(saves.xgs_encode("world.dat"), "world.dat")
        self.assertEqual(saves.xgs_encode(".hidden"), "%2Ehidden")
        self.assertEqual(saves.xgs_encode("~tmp"), "%7Etmp")
        self.assertEqual(saves.xgs_encode("é"), "%C3%A9")

    def _folders(self, tmp):
        jsav, _, _ = make_jsav()
        story = os.path.join(tmp, "in", "story")
        write(os.path.join(story, "SLOT0"), jsav)
        write(os.path.join(story, "chives.dat"), make_slayer("SLOT0", make_asav(kind=1, title="T", slot="SLOT3")))
        prof = os.path.join(tmp, "in", "profile")
        write(os.path.join(prof, "Player"), make_asav(kind=3, title="Player", slot="Player"))
        return story, prof, jsav

    def test_import_layout_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            story, prof, jsav = self._folders(tmp)
            out = os.path.join(tmp, "saves")
            quiet(saves.import_saves, [story, prof, "--out", out])
            # slot name comes from the packed copy's ASAV header when there is one
            self.assertEqual(rd(os.path.join(out, "local-user", "SLOT3", "SLOT0")), jsav)
            self.assertTrue(os.path.isfile(os.path.join(out, "local-user", "Player", "Player")))
            m = rd_json(os.path.join(out, "import-manifest.json"))
            self.assertEqual(sorted(e["container"] for e in m["entries"]), ["Player", "SLOT3"])
            self.assertFalse(any(n.startswith("~") for _, _, fs in os.walk(out) for n in fs))

    def test_map_dry_run_force_and_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            story, prof, _ = self._folders(tmp)
            out = os.path.join(tmp, "saves")
            quiet(saves.import_saves, [story, "--out", out, "--map", "SLOT3=slot 0/story", "--dry-run"])
            self.assertFalse(os.path.exists(out))
            quiet(saves.import_saves, [story, "--out", out, "--map", "SLOT3=slot 0/story"])
            self.assertTrue(os.path.isfile(os.path.join(out, "local-user", "slot%200", "story")))
            with self.assertRaises(SystemExit):        # refuses to overwrite without --force
                quiet(saves.import_saves, [story, "--out", out, "--map", "SLOT3=slot 0/story"])
            quiet(saves.import_saves, [story, "--out", out, "--map", "SLOT3=slot 0/story", "--force"])
            with self.assertRaises(SystemExit):        # two payloads onto one blob (case-insensitive on Windows)
                quiet(saves.import_saves, [story, prof, "--out", out, "--force",
                                           "--map", "SLOT3=Same/blob", "--map", "Player=same/BLOB"])

    def test_folder_with_two_payloads_is_refused(self):
        # A folder holding a save and an older backup with the same signature must not import either silently.
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "ambiguous")
            jsav, _, _ = make_jsav()
            write(os.path.join(folder, "slot1"), jsav)
            write(os.path.join(folder, "slot1.bak"), jsav)
            out = os.path.join(tmp, "o")
            with self.assertRaises(SystemExit) as cm:
                quiet(saves.import_saves, [folder, "--out", out, "--dry-run"])
            msg = str(cm.exception)
            self.assertIn("slot1", msg)
            self.assertIn("slot1.bak", msg)
            self.assertFalse(os.path.exists(out))

    def test_folder_without_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, "empty")
            write(os.path.join(empty, "readme.txt"), b"nothing here")
            with self.assertRaises(SystemExit):
                quiet(saves.import_saves, [empty, "--out", os.path.join(tmp, "o")])


if __name__ == "__main__":
    unittest.main()
