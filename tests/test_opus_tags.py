"""Synthetic tests for tools/opus_tags.py (no game data: invented banks, names and subtitles)."""
import csv
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "opus_tags.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import awc_opus  # noqa: E402
import opus_tags as T  # noqa: E402

PACKET = bytes([0xF8]) + b"\x55" * 40          # CELT-only 20 ms mono frame code 0: 960 samples


def make_ogg(path, packets=200, stream_id=0x0ABCDEF1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    awc_opus.write_ogg_opus(path, [PACKET] * packets, 1, stream_id, packets * 960 - 100)


def read_tags(data):
    """(vendor, [comments]) of the OpusTags packet on page 1."""
    pages = list(T.pages(data))
    at, _ = pages[1]
    body = at + 27 + data[at + 26]
    n = struct.unpack_from("<I", data, body + 8)[0]
    vendor = data[body + 12:body + 12 + n]
    at = body + 12 + n
    count = struct.unpack_from("<I", data, at)[0]
    at += 4
    out = []
    for _ in range(count):
        k = struct.unpack_from("<I", data, at)[0]
        out.append(data[at + 4:at + 4 + k].decode("utf-8"))
        at += 4 + k
    return vendor, out


def crc_ok(data):
    for a, e in T.pages(data):
        page = bytearray(data[a:e])
        want = struct.unpack_from("<I", page, 22)[0]
        page[22:26] = b"\0\0\0\0"
        if awc_opus.ogg_crc(bytes(page)) != want:
            return False
    return True


def write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


NAMES = ["source", "bank", "stream_id", "name", "rule"]
DIALOGUE = ["wave", "key", "label", "speaker", "text_en", "text_de"]


class TestPacket(unittest.TestCase):
    def test_tags_packet_layout_and_empty_values(self):
        p = T.tags_packet([("TITLE", "wave_a"), ("SPEAKER", ""), ("SUBTITLE", "Grüß dich")])
        self.assertTrue(p.startswith(b"OpusTags" + struct.pack("<I", 8) + T.VENDOR))
        self.assertEqual(struct.unpack_from("<I", p, 20)[0], 2)
        self.assertIn("SUBTITLE=Grüß dich".encode(), p)

    def test_markup_is_removed(self):
        self.assertEqual(T.clean("~sl:0.0:1.5~Hold ~COLOR_RED~still~s~,~n~friend.  "), "Hold still , friend.")


class TestRetag(unittest.TestCase):
    def test_only_the_tags_page_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "b", "0ABCDEF1.ogg")
            make_ogg(path)
            with open(path, "rb") as f:
                old = f.read()
        new = T.retag(old, T.tags_packet([("TITLE", "x" * 3000)]))
        old_pages, new_pages = list(T.pages(old)), list(T.pages(new))
        self.assertEqual(len(old_pages), len(new_pages))
        self.assertEqual(old[:old_pages[1][0]], new[:new_pages[1][0]])
        self.assertEqual(old[old_pages[2][0]:], new[new_pages[2][0]:])
        self.assertTrue(crc_ok(new))
        self.assertEqual(read_tags(new), (T.VENDOR, ["TITLE=" + "x" * 3000]))

    def test_rejects_what_it_cannot_rewrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "b", "0ABCDEF1.ogg")
            make_ogg(path)
            with open(path, "rb") as f:
                data = f.read()
        with self.assertRaisesRegex(ValueError, "one page"):
            T.retag(data, b"OpusTags" + b"\0" * T.MAX_PACKET)
        with self.assertRaisesRegex(ValueError, "Ogg page"):
            T.retag(b"RIFF" + data[4:], T.tags_packet([]))
        first = next(T.pages(data))[1]
        with self.assertRaisesRegex(ValueError, "two Ogg pages"):
            T.retag(data[:first], T.tags_packet([]))


class TestFolder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.opus = os.path.join(d, "opus")
        make_ogg(os.path.join(self.opus, "story", "bank_01", "0ABCDEF1.ogg"))
        make_ogg(os.path.join(self.opus, "story", "bank_01", "00000123.ogg"), stream_id=0x123)
        make_ogg(os.path.join(self.opus, "story", "bank_02", "1FFFFFFF.ogg"), stream_id=0x1FFFFFFF)
        self.names = os.path.join(d, "names.csv")
        write_csv(self.names, NAMES, [
            {"source": "story", "bank": "bank_01", "stream_id": "0abcdef1", "name": "heistau_abcd_01", "rule": "dialogue"},
            {"source": "story", "bank": "bank_02", "stream_id": "1fffffff", "name": "door_slam_03", "rule": "family"},
            {"source": "other", "bank": "bank_02", "stream_id": "1fffffff", "name": "wrong_name", "rule": "family"},
        ])
        self.dialogue = os.path.join(d, "dialogue.csv")
        write_csv(self.dialogue, DIALOGUE, [{"wave": "heistau_abcd_01", "key": "HEISTAU_ABCD", "label": "HEIST_ZED_1",
                                             "speaker": "ZED", "text_en": "~z~Move.~n~Now.", "text_de": "Los."}])

    def tearDown(self):
        self.tmp.cleanup()

    def tags(self, *parts):
        with open(os.path.join(self.opus, *parts), "rb") as f:
            return read_tags(f.read())[1]

    def test_names_and_dialogue(self):
        names = T.load_names(self.names, "story")
        dialogue, langs = T.load_dialogue(self.dialogue)
        stats = T.tag_folder(self.opus, names, dialogue, langs)
        self.assertEqual((stats["ogg files"], stats["tagged"], stats["without a name"]), (3, 2, 1))
        self.assertEqual((stats["with a dialogue line"], stats["with a subtitle"]), (1, 1))
        self.assertEqual(self.tags("story", "bank_01", "0ABCDEF1.ogg"),
                         ["TITLE=heistau_abcd_01", "LABEL=HEIST_ZED_1", "SPEAKER=ZED", "SUBTITLE=Move. Now.",
                          "SUBTITLE_EN=Move. Now.", "SUBTITLE_DE=Los."])
        self.assertEqual(self.tags("story", "bank_02", "1FFFFFFF.ogg"), ["TITLE=door_slam_03"])
        self.assertEqual(self.tags("story", "bank_01", "00000123.ogg"), [])

    def test_duplicate_id_stream_gets_the_same_tags(self):
        # awc_opus.py names a bank's second stream with an id already used <id>~2 (export_stems); it has the same
        # id, so it is the same name and dialogue line.
        stem = awc_opus.export_stems([0x0ABCDEF1, 0x0ABCDEF1])[1]
        make_ogg(os.path.join(self.opus, "story", "bank_01", stem + ".ogg"))
        dialogue, langs = T.load_dialogue(self.dialogue)
        stats = T.tag_folder(self.opus, T.load_names(self.names, "story"), dialogue, langs)
        self.assertEqual((stats["ogg files"], stats["tagged"], stats["without a name"]), (4, 3, 1))
        self.assertEqual(self.tags("story", "bank_01", stem + ".ogg"), self.tags("story", "bank_01", "0ABCDEF1.ogg"))
        self.assertIn("TITLE=heistau_abcd_01", self.tags("story", "bank_01", stem + ".ogg"))

    def test_second_run_replaces_its_own_tags(self):
        names = T.load_names(self.names, "story")
        T.tag_folder(self.opus, names)
        path = os.path.join(self.opus, "story", "bank_02", "1FFFFFFF.ogg")
        with open(path, "rb") as f:
            once = f.read()
        T.tag_folder(self.opus, names)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), once)
        self.assertTrue(crc_ok(once))

    def test_dry_run_writes_nothing(self):
        path = os.path.join(self.opus, "story", "bank_02", "1FFFFFFF.ogg")
        with open(path, "rb") as f:
            before = f.read()
        stats = T.tag_folder(self.opus, T.load_names(self.names, "story"), dry_run=True)
        self.assertEqual(stats["tagged"], 2)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_cli(self):
        r = subprocess.run([sys.executable, "-I", TOOL, self.opus, "--names", self.names, "--source", "story",
                            "--dialogue", self.dialogue], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("with a subtitle", r.stdout)
        r = subprocess.run([sys.executable, "-I", TOOL, self.opus, "--names", self.names + ".missing"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
