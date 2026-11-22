"""Tests for tools/mem_diff.py, the FUZZ-5 memory differential fuzzer (standard library only; no game data, GPU or
llvm-mc).

    python -I tests/test_mem_diff.py
"""
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.dont_write_bytecode = True

import mem_diff as md  # noqa: E402


def program(lines, memory):
    p = md.Program()
    p.lines, p.memory = lines, dict(memory)
    return p


class Harness(unittest.TestCase):
    def test_self_test_passes(self):
        self.assertTrue(md.self_test())

    def test_generation_is_deterministic(self):
        a = md.generate_program(random.Random(42), 3)
        b = md.generate_program(random.Random(42), 3)
        self.assertEqual(a.lines, b.lines)
        self.assertEqual(a.memory, b.memory)

    def test_every_family_runs_clean(self):
        for family in sorted(md.GENERATORS):
            with self.subTest(family=family):
                counts, first = md.run_suite(60, seed=20261122, max_ops=2, families=[family])
                self.assertEqual(counts["mismatch"], 0, first)
                self.assertEqual(counts["harness-error"], 0, first)
                self.assertEqual(counts["skip"], {}, counts["skip"])
                self.assertEqual(counts["pass"], 60)


class Reference(unittest.TestCase):
    """The reference model on hand-made cases (the expected words are worked out by hand)."""

    def run_ref(self, lines, memory):
        ref = md.RefMachine(dict(memory))
        ref.run(lines)
        return ref

    def test_structured_buffer_address(self):
        # base 0x3000, stride 16, index 2, voffset 4, soffset 8, offset:4 -> 0x3000 + 8 + 4 + 32 + 4 = 0x3030
        ref = self.run_ref(["s_mov_b32 s0, 0x3000", "s_mov_b32 s1, 0x100000", "s_mov_b32 s2, 4", "s_mov_b32 s3, 0",
                            "v_mov_b32 v2, 2", "v_mov_b32 v3, 4",
                            "buffer_load_dword v4, v[2:3], s[0:3], 8 idxen offen offset:4"], {0x3030: 0xABCD})
        self.assertEqual(ref.v[4], 0xABCD)

    def test_s_buffer_load_masks_the_descriptor_flags(self):
        # V# word1 bits 31:16 (stride/flags) are not part of the base address.
        ref = self.run_ref(["s_mov_b32 s0, 0x4000", "s_mov_b32 s1, 0xFFFF0001", "s_mov_b32 s2, 0x100", "s_mov_b32 s3, 0",
                            "s_buffer_load_dword s8, s[0:3], 0x8"], {(1 << 32) + 0x4008: 0x55})
        self.assertEqual(ref.s[8], 0x55)

    def test_global_negative_offset_and_sign_extension(self):
        ref = self.run_ref(["v_mov_b32 v2, 0x5010", "v_mov_b32 v3, 0",
                            "global_load_sbyte v4, v[2:3], off offset:-15"], {0x5000: 0x0000_8000})
        self.assertEqual(ref.v[4], 0xFFFFFF80)       # byte 1 of the word at 0x5000, sign-extended

    def test_cmpswap_and_glc(self):
        memory = {0x6000: 7}
        lines = ["s_mov_b32 s0, 0x6000", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x10", "s_mov_b32 s3, 0",
                 "v_mov_b32 v4, 9", "v_mov_b32 v5, 7", "buffer_atomic_cmpswap v[4:5], off, s[0:3], 0 glc"]
        ref = self.run_ref(lines, memory)
        self.assertEqual((ref.mem[0x6000], ref.v[4]), (9, 7))


class GlobalOffsetRegression(unittest.TestCase):
    """llvm-mc prints a gfx10 global_* immediate as signed decimal ('offset:-16'). The translation must add -16, not
    sign-extend the printed value a second time (which reads 4 KiB lower)."""

    def test_negative_global_offset(self):
        memory = {0x8000 - 16: 0x11111111, 0x8000 - 16 - 0x1000: 0x22222222}
        for form in (["v_mov_b32 v2, 0x8000", "v_mov_b32 v3, 0", "global_load_dword v8, v[2:3], off offset:-16"],
                     ["s_mov_b32 s4, 0x8000", "s_mov_b32 s5, 0", "v_mov_b32 v2, 0",
                      "global_load_dword v8, v2, s[4:5] offset:-16"]):
            with self.subTest(form=form[-1]):
                status, detail = md.run_one("ps_mem_global_neg", program(
                    form + ["s_waitcnt vmcnt(0)", "v_mov_b32 v9, 0", "v_mov_b32 v10, 0", "v_mov_b32 v11, 0"], memory))
                self.assertEqual(status, "match", detail)

    def test_negative_global_store_offset(self):
        status, detail = md.run_one("ps_mem_global_neg_store", program(
            ["v_mov_b32 v2, 0x9000", "v_mov_b32 v3, 0", "v_mov_b32 v4, 0xC0FFEE",
             "global_store_dword v[2:3], v4, off offset:-2048", "v_mov_b32 v8, 0", "v_mov_b32 v9, 0",
             "v_mov_b32 v10, 0", "v_mov_b32 v11, 0"], {}))
        self.assertEqual(status, "match", detail)


if __name__ == "__main__":
    unittest.main()
