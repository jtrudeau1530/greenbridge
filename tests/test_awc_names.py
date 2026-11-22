"""Tests for tools/awc_names.py, built from synthetic AWC stream tables and words (no game data).

    python -I tests/test_awc_names.py
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
    path = os.path.join(ROOT, "tools", "awc_names.py")
    spec = importlib.util.spec_from_file_location("awc_names_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load_module()


def make_bank(names, flags=0, extra_ids=()):
    """A minimal AWC: header, the optional tables ``flags`` selects, and a stream table without tags."""
    ids = [A.joaat(n) & A.M29 for n in names] + list(extra_ids)
    n = len(ids)
    out = bytearray(b"ADAT" + struct.pack("<III", flags, n, 0))
    if flags & 0x400000:
        out += bytes(4)
    if flags & 0x10000:
        out += bytes(2 * n)
    if flags & 0x200000:
        out += bytes(4 * (n + 1))
    # a non-zero tag count in bits 31:29 must not leak into the id
    out += b"".join(struct.pack("<I", (1 << 29) | i) for i in ids)
    return bytes(out)


def write(path, data):
    with open(path, "wb" if isinstance(data, bytes) else "w") as fh:
        fh.write(data)
    return path


class TestJoaat(unittest.TestCase):
    def test_reference_values(self):
        # Jenkins one-at-a-time, lowercased (the same function as tools/hashdict.py)
        self.assertEqual(A.joaat("a"), 0xCA2E9442)
        self.assertEqual(A.joaat("ABC"), A.joaat("abc"))

    def test_incremental_state(self):
        base = A.joaat_state(b"foo_bar")
        self.assertEqual(A.joaat_finish(A.joaat_state(b"_01", base)), A.joaat("foo_bar_01"))

    def test_matches_hashdict(self):
        import importlib.util as u
        spec = u.spec_from_file_location("hashdict_ref", os.path.join(ROOT, "tools", "hashdict.py"))
        hd = u.module_from_spec(spec)
        spec.loader.exec_module(hd)
        for w in ("x", "Some_Name_01", "path/to/thing", "caf\xe9"):
            self.assertEqual(A.joaat(w), hd.joaat(w))


class TestBanks(unittest.TestCase):
    def test_flags_skip_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            for flags in (0, 0x400000, 0x10000, 0x200000, 0x610000):
                path = write(os.path.join(tmp, f"{flags:x}.awc"), make_bank(["a_01", "b_02"], flags))
                self.assertEqual(A.bank_ids(path), [A.joaat("a_01") & A.M29, A.joaat("b_02") & A.M29])

    def test_directory_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "sub"))
            write(os.path.join(tmp, "sub", "one.awc"), make_bank(["x"]))
            write(os.path.join(tmp, "readme.txt"), "not a bank")
            self.assertEqual(list(A.load_banks(tmp)), ["sub/one"])
            manifest = write(os.path.join(tmp, "audio_manifest.json"), json.dumps(
                {"src": "audio.rpf", "Count": 2, "Banks": [{"Path": "p/b.awc", "Streams": [{"Id": "%08X" % (A.joaat("x") & A.M29)}]},
                                {"Path": "p/unreadable.awc", "Streams": None}]}))
            self.assertEqual(A.load_banks(manifest), {"p/b.awc": [A.joaat("x") & A.M29], "p/unreadable.awc": []})

    def test_truncated_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(os.path.join(tmp, "t.awc"), make_bank(["a", "b"])[:-3])
            with self.assertRaises(ValueError):
                A.bank_ids(path)


class TestRules(unittest.TestCase):
    def index(self, banks):
        return A.Index({k: [A.joaat(n) & A.M29 for n in v] for k, v in banks.items()})

    def common_words(self, *tokens):
        # each token in MIN_TOKEN words so family bases built from them count as "wordy"
        return [f"{t}_filler{k}" for t in tokens for k in range(A.MIN_TOKEN)]

    def test_dialogue_rule_exact_word(self):
        d = A.Dictionary(["zz_rootau_abcd_01", "zz_rootau_abcd"])
        found = A.attack(d, d.bases(A.MIN_TOKEN), self.index({"b": ["zz_rootau_abcd_01"]}), "")
        self.assertEqual([v[:2] for v in found.values()], [("zz_rootau_abcd_01", "dialogue")])

    def test_family_needs_members_in_one_bank_and_a_verbatim_word(self):
        words = self.common_words("horn", "loop") + ["horn_loop_01"]
        d = A.Dictionary(words)
        bases = d.bases(A.MIN_TOKEN)
        self.assertIn("horn_loop", bases)
        # three members in one bank -> accepted (horn_loop_02/_03 are not words but extend the family)
        found = A.attack(d, bases, self.index({"b": ["horn_loop_01", "horn_loop_02", "horn_loop_03"]}), "")
        self.assertEqual(sorted(v[0] for v in found.values()), ["horn_loop_01", "horn_loop_02", "horn_loop_03"])
        # spread over banks -> no family anywhere
        found = A.attack(d, bases, self.index({"b1": ["horn_loop_01"], "b2": ["horn_loop_02"],
                                               "b3": ["horn_loop_03"]}), "")
        self.assertEqual(found, {})

    def test_family_without_verbatim_member_is_rejected(self):
        d = A.Dictionary(self.common_words("horn", "loop") + ["horn_loop_77"])
        found = A.attack(d, d.bases(A.MIN_TOKEN), self.index({"b": ["horn_loop_01", "horn_loop_02",
                                                                     "horn_loop_03"]}), "")
        self.assertEqual(found, {})

    def test_rare_token_base_is_not_wordy(self):
        d = A.Dictionary(self.common_words("horn") + ["horn_qqq_01"])
        self.assertNotIn("horn_qqq", d.bases(A.MIN_TOKEN))
        self.assertIn("horn_qqq", d.bases(1))

    def test_family_ok(self):
        self.assertTrue(A.family_ok(["a_01", "a_02"]))
        self.assertFalse(A.family_ok(["a_01", "a_03"]))
        self.assertTrue(A.family_ok(["a_01", "a_05", "a_09"]))
        self.assertFalse(A.family_ok(["a_01"]))

    def test_control_salt_changes_every_name(self):
        d = A.Dictionary(["zz_rootau_abcd_01"])
        found = A.attack(d, [], self.index({"b": ["zz_rootau_abcd_01"]}), A.SALT)
        self.assertEqual(found, {})

    def test_name_source_keeps_rules_with_quiet_controls(self):
        words = self.common_words("horn", "loop") + ["horn_loop_01", "zz_rootau_abcd_01"]
        d = A.Dictionary(words)
        index = self.index({"b": ["horn_loop_01", "horn_loop_02", "zz_rootau_abcd_01"]})
        found, stats = A.name_source(d, d.bases(A.MIN_TOKEN), index)
        self.assertEqual(stats["named"], 3)
        self.assertTrue(stats["dialogue_rule"])
        self.assertEqual(stats["family_min_size"], 2)
        self.assertEqual(stats["control_dialogue"] + stats["control_family"], 0)


class TestConversations(unittest.TestCase):
    def test_waves_per_label(self):
        blob = (b"junk\x00ZZ_ROOTAU_ABCD\x00ZZ_LINE_A\x00ZZ_ROOTAU_ABCD\x00ZZ_LINE_B\x00"
                b"ZZ_ROOTAU_ABCD\x00ZZ_LINE_A\x00YY_XAU_EFGH\x00YY_ONLY\x00")
        with tempfile.TemporaryDirectory() as tmp:
            write(os.path.join(tmp, "c.pso"), blob)
            waves = sorted(A.conversation_waves([tmp]))
        # a repeated label is one variation, not two
        self.assertEqual(waves, ["yy_xau_efgh_01", "zz_rootau_abcd_01", "zz_rootau_abcd_02"])


class TestCli(unittest.TestCase):
    def test_sources_with_same_basename_keep_separate_summaries_and_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources = []
            for parent, count in (("first", 1), ("second", 2)):
                folder = os.path.join(tmp, parent, "banks")
                os.makedirs(folder)
                sources.append(folder)
                write(os.path.join(folder, "one.awc"), make_bank(["zz_rootau_abcd_01"], extra_ids=range(count - 1)))
            words = write(os.path.join(tmp, "words.txt"), "zz_rootau_abcd_01\n")
            out = os.path.join(tmp, "names.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(A.main([*sources, "--words", words, "--out", out]), 0)
            with open(out, encoding="utf-8") as fh:
                doc = json.load(fh)
            self.assertEqual({k: v["streams"] for k, v in doc["summary"].items()},
                             {"first/banks": 1, "second/banks": 2})
            self.assertEqual([row["source"] for row in doc["streams"]], ["first/banks", "second/banks"])

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            banks = os.path.join(tmp, "banks")
            os.mkdir(banks)
            write(os.path.join(banks, "one.awc"), make_bank(["zz_rootau_abcd_01", "zz_rootau_abcd_02"],
                                                            extra_ids=[12345]))
            conv = write(os.path.join(tmp, "c.pso"),
                         b"ZZ_ROOTAU_ABCD\x00ZZ_A\x00ZZ_ROOTAU_ABCD\x00ZZ_B\x00")
            words = write(os.path.join(tmp, "words.txt"), "unrelated\nother_word\n")
            out = os.path.join(tmp, "names.csv")
            with contextlib.redirect_stdout(io.StringIO()) as log:
                self.assertEqual(A.main([banks, "--words", words, "--conversations", conv, "--out", out]), 0)
            self.assertIn("2 named", log.getvalue())
            with open(out, encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([(r["bank"], r["name"], r["rule"]) for r in rows],
                             [("one", "zz_rootau_abcd_01", "dialogue"), ("one", "zz_rootau_abcd_02", "dialogue")]
                             if A.joaat("zz_rootau_abcd_01") & A.M29 < A.joaat("zz_rootau_abcd_02") & A.M29 else
                             [("one", "zz_rootau_abcd_02", "dialogue"), ("one", "zz_rootau_abcd_01", "dialogue")])
            json_out = os.path.join(tmp, "names.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(A.main([banks, "--conversations", conv, "--out", json_out]), 0)
            with open(json_out, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["summary"]["banks"]["streams"], 3)

    def test_rejects_output_aliasing_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = write(os.path.join(tmp, "one.awc"), make_bank(["zz_rootau_abcd_01"]))
            original = b"ZZ_ROOTAU_ABCD\x00ZZ_A\x00"
            conv = write(os.path.join(tmp, "c.pso"), original)
            alias = os.path.join(tmp, ".", "c.pso")
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as cm:
                    A.main([bank, "--conversations", conv, "--out", alias])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("--out must not overwrite an input", err.getvalue())
            with open(conv, "rb") as f:
                self.assertEqual(f.read(), original)

    def test_hashdict_json_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(os.path.join(tmp, "hashes.json"), json.dumps({"0000abcd": "Some_Word"}))
            self.assertEqual(A.read_words(path), ["Some_Word"])

    def test_needs_a_dictionary(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            A.main([ROOT])

    def test_missing_input(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(A.main(["/nonexistent", "--words", "/nonexistent.txt"]), 2)


if __name__ == "__main__":
    unittest.main()
