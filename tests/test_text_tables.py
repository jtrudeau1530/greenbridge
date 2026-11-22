"""Tests for tools/text_tables.py, built from synthetic resources and conversation data (no game data).

    python -I tests/test_text_tables.py
"""
import contextlib
import csv
import importlib.util
import io
import json
import os
import struct
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "text_tables.py")
    spec = importlib.util.spec_from_file_location("text_tables_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = load_module()


def ptr(off):
    return struct.pack("<II", 0x50000000 | off, 0)


def make_resource(strings, pad=0):
    """A logical resource: 0DRP at +0x10, a descriptor, then hash table, index array and string blob.

    ``strings`` maps label -> text. ``pad`` adds bytes before the tables so they can cross a 64 KiB boundary.
    """
    labels = list(strings)
    blob = bytearray()
    blob_offsets = []
    for label in labels:
        blob_offsets.append(len(blob))
        blob += strings[label].encode("utf-8") + b"\0"
    head = bytearray(0x10) + b"0DRP" + bytes(0x0C)
    desc_at = len(head)
    desc_len = 8 + 4 + 4 + 8 + 4 + 4 + 8 + 4 + 4 + 8
    index_at = desc_at + desc_len + pad
    hash_at = index_at + 16 * len(labels)
    blob_at = hash_at + 16 * len(labels)
    desc = (ptr(index_at) + struct.pack("<II", 0, 0x10) + ptr(0) + b"\x01\x01\x00\x00"
            + struct.pack("<I", 16 * len(labels)) + ptr(hash_at) + struct.pack("<II", 0x10, len(blob))
            + ptr(blob_at))
    assert len(desc) == desc_len
    index = b"".join(struct.pack("<QQ", 0, off << 20) for off in blob_offsets)
    hashes = bytearray()
    for i, label in enumerate(labels):
        entry = bytearray(16)
        struct.pack_into("<II", entry, 0, T.joaat(label), 0)
        entry[8] = 1
        entry[11:14] = i.to_bytes(3, "little")
        hashes += entry
    return bytes(head + desc + bytes(pad) + index + hashes + blob)


def write(path, data):
    with open(path, "wb") as fh:
        fh.write(data)
    return path


EN = {"ZZ_LINE_A_1": "This is the first line and you know it", "ZZ_LINE_B_2": "What is that, the second?",
      "ZZ_MENU_TITLE": "the menu"}
DE = {"ZZ_LINE_A_1": "Das ist die erste Zeile und du weißt es", "ZZ_LINE_B_2": "Was ist das, die zweite?",
      "ZZ_MENU_TITLE": "das Menü"}
JA = {"ZZ_LINE_A_1": "これは最初の行です"}
CONV = b"head\x00ZZ_ROOTAU_ABCD\x00ZZ_LINE_A_1\x00ZZ_ROOTAU_ABCD\x00ZZ_LINE_B_2\x00ZZ_ROOTAU_ABCD\x00ZZ_LINE_A_1\x00"


class TestTables(unittest.TestCase):
    def test_decode(self):
        tables = list(T.string_tables(make_resource(EN)))
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0], {T.joaat(k): v for k, v in EN.items()})

    def test_bad_bounds_skipped(self):
        res = bytearray(make_resource(EN))
        at = res.index(b"\x01\x01\x00\x00") + 4
        struct.pack_into("<I", res, at, len(res) * 2)      # hash table larger than the resource
        self.assertEqual(list(T.string_tables(bytes(res))), [])

    def test_language(self):
        self.assertEqual(T.language(EN.values()), "en")
        self.assertEqual(T.language(DE.values()), "de")
        self.assertEqual(T.language(JA.values()), "ja")
        self.assertEqual(T.language(["Привет"]), "ru")
        self.assertEqual(T.language(["안녕하세요"]), "ko")
        self.assertEqual(T.language(["你好世界"]), "zh")

    def test_group(self):
        self.assertEqual(T.table_group(["BIL1_A_1", "BIL1_B_2", "BIL1_C_3", "OTHER"]), "BIL1")
        self.assertEqual(T.table_group(["AB_CD_1", "AB_CD_2", "AB_CD_3"]), "AB_CD")
        self.assertEqual(T.table_group(["A_1", "B_2"]), "")


class TestChains(unittest.TestCase):
    def carve(self, tmp, streams):
        """rpf9_carve.py layout: index.json plus one <offset>.bin per inflated stream."""
        files, off = [], 0x1000
        for data in streams:
            write(os.path.join(tmp, f"{off:012x}.bin"), data)
            files.append({"off": off, "stored": len(data), "size": len(data), "pre": "", "head": data[:16].hex()})
            off += 0x20000
        with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as fh:
            json.dump({"files": files, "summary": {}}, fh)

    def test_resource_across_64k_streams(self):
        res = make_resource(EN, pad=T.BLOCK)            # tables start past the first 64 KiB
        streams = [res[:T.BLOCK], res[T.BLOCK:], make_resource(DE), CONV]
        with tempfile.TemporaryDirectory() as tmp:
            self.carve(tmp, streams)
            scan = T.Scan()
            for label, off, data in T.blobs(tmp):
                scan.add(label, off, data)
        self.assertEqual(sorted(lang for _s, _o, lang, _t in scan.tables), ["de", "en"])
        self.assertEqual(scan.tables[0][1], 0x1000)

    def test_short_stream_ends_the_chain(self):
        rows = [{"off": 0, "size": 100}, {"off": 1, "size": 50}]
        data = {0: b"a" * 100, 1: b"b" * 50}
        self.assertEqual([o for o, _d in T._chains(rows, lambda r: data[r["off"]])], [0, 1])

    def test_new_resource_ends_the_chain(self):
        rows = [{"off": 0, "size": T.BLOCK}, {"off": 1, "size": 10}]
        data = {0: bytes(T.BLOCK), 1: bytes(0x10) + b"0DRP"}
        self.assertEqual([o for o, _d in T._chains(rows, lambda r: data[r["off"]])], [0, 1])


class TestDialogue(unittest.TestCase):
    def test_waves_and_text(self):
        scan = T.Scan()
        scan.add("a", 0, make_resource(EN))
        scan.add("b", 0, make_resource(DE))
        scan.add("c", 0, CONV)
        rows = list(scan.dialogue_rows())
        self.assertEqual([(r["wave"], r["label"], r["speaker"]) for r in rows],
                         [("zz_rootau_abcd_01", "ZZ_LINE_A_1", "A"), ("zz_rootau_abcd_02", "ZZ_LINE_B_2", "B")])
        self.assertEqual(rows[0]["text_en"], EN["ZZ_LINE_A_1"])
        self.assertEqual(rows[1]["text_de"], DE["ZZ_LINE_B_2"])
        self.assertEqual(scan.languages(), ["en", "de"])

    def test_lowercase_in_label(self):
        scan = T.Scan()
        scan.add("c", 0, b"ZZ_KEYAU_ABCD\x00ZZ_TIMELEFT_90s_1\x00")
        self.assertEqual(scan.pairs["ZZ_KEYAU_ABCD"], ["ZZ_TIMELEFT_90s_1"])


class TestCli(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            write(os.path.join(src, "en.bin"), make_resource(EN))
            write(os.path.join(src, "de.bin"), make_resource(DE))
            write(os.path.join(src, "conv.pso"), CONV)
            labels = os.path.join(tmp, "labels.json")
            with open(labels, "w", encoding="utf-8") as fh:
                json.dump({f"{T.joaat('zz_menu_title'):08x}": "zz_menu_title"}, fh)
            outs = {k: os.path.join(tmp, f"{k}.csv") for k in ("dialogue", "tables", "strings")}
            with contextlib.redirect_stdout(io.StringIO()) as log:
                self.assertEqual(T.main([src, "--labels", labels, "--dialogue", outs["dialogue"],
                                         "--tables", outs["tables"], "--strings", outs["strings"]]), 0)
            self.assertIn("2 string tables", log.getvalue())
            self.assertIn("1 dialogue keys, 2 lines", log.getvalue())
            with open(outs["tables"], encoding="utf-8", newline="") as fh:
                tables = list(csv.DictReader(fh))
            self.assertEqual({(t["lang"], t["strings"], t["resolved"], t["group"]) for t in tables},
                             {("en", "3", "3", "ZZ_LINE"), ("de", "3", "3", "ZZ_LINE")})  # 2 of 3 share ZZ_LINE
            with open(outs["strings"], encoding="utf-8", newline="") as fh:
                strings = list(csv.DictReader(fh))
            self.assertIn(("ZZ_MENU_TITLE", "das Menü"), {(s["label"], s["text"]) for s in strings})
            self.assertFalse([f for f in os.listdir(tmp) if f.startswith(".text_tables-")],
                             "staging files must not be left behind")

    def test_failed_later_write_leaves_earlier_reports_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            write(os.path.join(src, "conv.pso"), CONV)
            dialogue = os.path.join(tmp, "dialogue.csv")
            old = "OLD GOOD REPORT\n"
            with open(dialogue, "w", encoding="utf-8") as fh:
                fh.write(old)
            tables = os.path.join(tmp, "tables.csv")
            os.mkdir(tables)                        # a later output that cannot be written
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(T.main([src, "--dialogue", dialogue, "--tables", tables]), 2)
            self.assertIn("cannot write", err.getvalue())
            with open(dialogue, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), old)
            self.assertFalse([f for f in os.listdir(tmp) if f.startswith(".text_tables-")])

    @unittest.skipUnless(os.name == "nt", "Windows denies os.replace on a file held open by a reader")
    def test_reader_holding_a_later_report_keeps_all_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            write(os.path.join(src, "conv.pso"), CONV)
            dialogue = os.path.join(tmp, "dialogue.csv")
            tables = os.path.join(tmp, "tables.csv")
            old_dialogue, old_tables = "OLD DIALOGUE\n", "OLD TABLES\n"
            for path, text in ((dialogue, old_dialogue), (tables, old_tables)):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
            with open(tables, "rb"):       # a reader keeps the second output open
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(T.main([src, "--dialogue", dialogue, "--tables", tables]), 2)
            self.assertIn("cannot write", err.getvalue())
            with open(dialogue, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), old_dialogue)
            with open(tables, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), old_tables)
            self.assertFalse([f for f in os.listdir(tmp) if f.startswith(".text_tables-")],
                             "staging and backup files must not be left behind")

    def test_two_outputs_naming_one_file_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            write(os.path.join(src, "conv.pso"), CONV)
            out = os.path.join(tmp, "same.csv")
            with contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as cm:
                    T.main([src, "--dialogue", out, "--tables", out])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("distinct", err.getvalue())
            self.assertFalse(os.path.exists(out), "no report may be written before the check")

    def test_missing_input(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(T.main(["/nonexistent"]), 2)


if __name__ == "__main__":
    unittest.main()
