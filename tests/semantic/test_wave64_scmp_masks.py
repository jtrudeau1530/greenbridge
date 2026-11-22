"""Semantic tests for Wave64/Wave32 s_cmp_* on opposite mask halves and mask-vs-scalar operands.

Translates synthetic .s listings (no llvm-mc or game data required) with tools/rdna2spirv.py and
executes all 64 (wave64) or 32 (wave32) lanes via tests/semantic/spirv_mini_interp.run_workgroup.

Covers:
  - Wave64 s_cmp_eq_u32 / s_cmp_lg_u32 on opposite halves: vcc_lo vs vcc_hi, exec_lo vs exec_hi,
    and MASK_LO SGPR vs MASK_HI SGPR (equal halves, unequal halves matching on lane 0 with equal
    popcount, and single-bit mismatches).
  - Wave64 and Wave32 s_cmp_eq_u32 / s_cmp_lt_u32 / s_cmp_gt_i32 comparing a lane-mask half
    against arbitrary 32-bit scalar SGPRs and immediates (e.g. 0x80000003).
  - Wave64 s_cmp_eq_u64 / s_cmp_lg_u64 comparing a lane mask against an arbitrary 64-bit SGPR
    pair or non-0/-1 immediate.
  - Preservation of the fast wave_equal vote path for same-half and mask-vs-0/-1 equality compares.
"""
import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
# COMPUTE_PGM_RSRC2: 0x8090 -> 8 user SGPRs (s[0:3]=V#, s4=lo, s5=hi, s6=cmp_lo, s7=cmp_hi), TGID_X.
VSHARP = [0, 0, 0xFFFFFFFF, 0]


def make_fssm(threads):
    return bytes(0x138) + struct.pack("<4I", 0x8090, threads, 1, 1)


def translate_cs(name, asm, threads=64):
    """Translate a synthetic compute .s listing into SPIR-V assembly."""
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        s_path = os.path.join(tmp, "disasm", name + ".s")
        with open(s_path, "w", encoding="utf-8") as f:
            f.write(f"; {name} synthetic\n")
            for ln in asm.splitlines():
                st = ln.strip()
                if st and not st.startswith(";"):
                    f.write(f"\t{st}\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(make_fssm(threads))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(s_path)
            gc.collect()
        return spv


def run_cs(spv, threads, s4=0, s5=0, s6=0, s7=0):
    pc = VSHARP + [s4 & 0xFFFFFFFF, s5 & 0xFFFFFFFF, s6 & 0xFFFFFFFF, s7 & 0xFFFFFFFF] + [0] * 24
    return si.run_workgroup(
        spv, threads, threads,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc, memory={},
    )


def emit_store_scc(slot):
    """Emit RDNA2 lines that store SCC (1 or 0) to buffer dword `slot`."""
    return (
        f"s_cselect_b32 s20, 1, 0\n"
        f"v_mov_b32_e32 v4, s20\n"
        f"v_mov_b32_e32 v5, {slot * 4}\n"
        f"buffer_store_dword v4, v5, s[0:3], 0 offen\n"
    )


# Builds a wave64 lane mask in `vcc` whose bit i is ((s4 if i < 32 else s5) >> (i & 31)) & 1,
# and copies it into s[12:13] (MASK_LO=s12, MASK_HI=s13).
WAVE64_BUILD_VCC_AND_SGPR = """\
v_mov_b32_e32 v11, s5
v_cmp_lt_u32_e64 vcc, v0, 32
v_cndmask_b32_e64 v1, v11, s4, vcc
v_and_b32_e32 v2, 31, v0
v_lshrrev_b32_e32 v3, v2, v1
v_and_b32_e32 v3, 1, v3
v_cmp_ne_u32_e64 vcc, 0, v3
s_and_b64 s[12:13], vcc, exec
"""

# Same for wave32: builds `vcc` from s4 (lanes 0..31).
WAVE32_BUILD_VCC = """\
v_mov_b32_e32 v1, s4
v_and_b32_e32 v2, 31, v0
v_lshrrev_b32_e32 v3, v2, v1
v_and_b32_e32 v3, 1, v3
v_cmp_ne_u32_e64 vcc_lo, 0, v3
"""


def _s32(u):
    u &= 0xFFFFFFFF
    return u - 0x100000000 if u >= 0x80000000 else u


class Wave64OppositeHalvesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asm = (
            WAVE64_BUILD_VCC_AND_SGPR
            # 0: vcc_lo == vcc_hi
            + "s_cmp_eq_u32 vcc_lo, vcc_hi\n" + emit_store_scc(0)
            # 1: vcc_lo != vcc_hi
            + "s_cmp_lg_u32 vcc_lo, vcc_hi\n" + emit_store_scc(1)
            # 2: MASK_LO (s12) == MASK_HI (s13)
            + "s_cmp_eq_u32 s12, s13\n" + emit_store_scc(2)
            # 3: MASK_LO (s12) != MASK_HI (s13)
            + "s_cmp_lg_u32 s12, s13\n" + emit_store_scc(3)
            # 4 & 5: exec_lo == exec_hi / exec_lo != exec_hi (save/restore exec so store runs)
            + "s_mov_b64 s[14:15], exec\n"
            + "s_and_b64 exec, exec, vcc\n"
            + "s_cmp_eq_u32 exec_lo, exec_hi\n"
            + "s_cselect_b32 s21, 1, 0\n"
            + "s_cmp_lg_u32 exec_lo, exec_hi\n"
            + "s_cselect_b32 s22, 1, 0\n"
            + "s_mov_b64 exec, s[14:15]\n"
            + "v_mov_b32_e32 v4, s21\nv_mov_b32_e32 v5, 16\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "v_mov_b32_e32 v4, s22\nv_mov_b32_e32 v5, 20\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "s_endpgm\n"
        )
        cls.spv = translate_cs("cs_w64_opp_halves", asm, threads=64)

    def test_opposite_halves_equal_and_unequal(self):
        cases = [
            # Equal halves
            (0x80000003, 0x80000003),
            (0x00000000, 0x00000000),
            (0xFFFFFFFF, 0xFFFFFFFF),
            # Unequal halves that match on lane 0 AND have identical popcount (2)
            (0x00000003, 0x00000005),
            # Unequal halves with lane 0 off in both and identical popcount (1)
            (0x00000002, 0x80000000),
            # Unequal halves where one is all-zero
            (0x00000000, 0x80000000),
            (0x00000001, 0x00000002),
        ]
        for lo, hi in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                mem = run_cs(self.spv, 64, s4=lo, s5=hi)
                eq = int(lo == hi)
                ne = int(lo != hi)
                got = [mem[4 * k] for k in range(6)]
                self.assertEqual(got, [eq, ne, eq, ne, eq, ne])


class MaskVsScalarAndOrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asm_w64 = (
            WAVE64_BUILD_VCC_AND_SGPR
            # 0: vcc_lo == 0x80000003 (arbitrary immediate)
            + "s_cmp_eq_u32 vcc_lo, 0x80000003\n" + emit_store_scc(0)
            # 1: vcc_hi == s6 (arbitrary scalar SGPR)
            + "s_cmp_eq_u32 vcc_hi, s6\n" + emit_store_scc(1)
            # 2: s6 == s12 (scalar on left, MASK_LO on right)
            + "s_cmp_eq_u32 s6, s12\n" + emit_store_scc(2)
            # 3: vcc_lo < s6 (unsigned ordering)
            + "s_cmp_lt_u32 vcc_lo, s6\n" + emit_store_scc(3)
            # 4: s13 < s7 (MASK_HI unsigned ordering)
            + "s_cmp_lt_u32 s13, s7\n" + emit_store_scc(4)
            # 5: vcc_lo > 0 (signed i32 ordering: 0x80000003 is negative!)
            + "s_cmp_gt_i32 vcc_lo, 0\n" + emit_store_scc(5)
            # 6: vcc_hi > s7 (signed i32 ordering)
            + "s_cmp_gt_i32 vcc_hi, s7\n" + emit_store_scc(6)
            # 7: exec_hi < s7 (exec_hi ordering against scalar)
            + "s_mov_b64 s[14:15], exec\n"
            + "s_and_b64 exec, exec, vcc\n"
            + "s_cmp_lt_u32 exec_hi, s7\n"
            + "s_cselect_b32 s21, 1, 0\n"
            + "s_mov_b64 exec, s[14:15]\n"
            + "v_mov_b32_e32 v4, s21\nv_mov_b32_e32 v5, 28\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "s_endpgm\n"
        )
        cls.spv_w64 = translate_cs("cs_w64_mask_scalar", asm_w64, threads=64)

        asm_w32 = (
            WAVE32_BUILD_VCC
            # 0: vcc_lo == 0x80000003
            + "s_cmp_eq_u32 vcc_lo, 0x80000003\n" + emit_store_scc(0)
            # 1: vcc_lo < s6 (u32)
            + "s_cmp_lt_u32 vcc_lo, s6\n" + emit_store_scc(1)
            # 2: vcc_lo > 0 (i32)
            + "s_cmp_gt_i32 vcc_lo, 0\n" + emit_store_scc(2)
            # 3: s6 == vcc_lo (scalar on left)
            + "s_cmp_eq_u32 s6, vcc_lo\n" + emit_store_scc(3)
            + "s_endpgm\n"
        )
        cls.spv_w32 = translate_cs("cs_w32_mask_scalar", asm_w32, threads=32)

    def test_wave64_mask_vs_scalar_and_ordering(self):
        cases = [
            # (lo, hi, cmp_lo=s6, cmp_hi=s7)
            (0x80000003, 0x00000005, 0x80000003, 0x00000006),
            (0x80000003, 0x80000001, 0x80000004, 0x00000000),
            (0x00000005, 0x00000005, 0x00000005, 0x00000005),
            (0x00000001, 0xFFFFFFFE, 0x00000000, 0xFFFFFFFF),
        ]
        for lo, hi, s6, s7 in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), s6=hex(s6), s7=hex(s7)):
                mem = run_cs(self.spv_w64, 64, s4=lo, s5=hi, s6=s6, s7=s7)
                want = [
                    int(lo == 0x80000003),
                    int(hi == s6),
                    int(s6 == lo),
                    int(lo < s6),
                    int(hi < s7),
                    int(_s32(lo) > 0),
                    int(_s32(hi) > _s32(s7)),
                    int(hi < s7),
                ]
                self.assertEqual([mem[4 * k] for k in range(8)], want)

    def test_wave32_mask_vs_scalar_and_ordering(self):
        cases = [
            (0x80000003, 0x80000003),
            (0x80000003, 0x80000004),
            (0x00000005, 0x00000004),
            (0x00000000, 0x00000001),
        ]
        for lo, s6 in cases:
            with self.subTest(lo=hex(lo), s6=hex(s6)):
                mem = run_cs(self.spv_w32, 32, s4=lo, s6=s6)
                want = [
                    int(lo == 0x80000003),
                    int(lo < s6),
                    int(_s32(lo) > 0),
                    int(s6 == lo),
                ]
                self.assertEqual([mem[4 * k] for k in range(4)], want)


class Wave64U64MaskCompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asm = (
            WAVE64_BUILD_VCC_AND_SGPR
            # 0: s_cmp_eq_u64 vcc, s[6:7] (mask vs arbitrary 64-bit scalar pair)
            + "s_cmp_eq_u64 vcc, s[6:7]\n" + emit_store_scc(0)
            # 1: s_cmp_lg_u64 vcc, s[6:7]
            + "s_cmp_lg_u64 vcc, s[6:7]\n" + emit_store_scc(1)
            # 2: s_cmp_eq_u64 s[6:7], s[12:13] (scalar pair vs mask SGPR pair)
            + "s_cmp_eq_u64 s[6:7], s[12:13]\n" + emit_store_scc(2)
            # 3: s_cmp_eq_u64 vcc, 0x12345678 (mask vs non-0/-1 immediate)
            + "s_cmp_eq_u64 vcc, 0x12345678\n" + emit_store_scc(3)
            # 4: s_cmp_lg_u64 vcc, 0x12345678
            + "s_cmp_lg_u64 vcc, 0x12345678\n" + emit_store_scc(4)
            + "s_endpgm\n"
        )
        cls.spv = translate_cs("cs_w64_u64_mask", asm, threads=64)

    def test_u64_mask_vs_scalar_pair_and_imm(self):
        cases = [
            # Exact match with scalar pair
            (0x12345678, 0x9ABCDEF0, 0x12345678, 0x9ABCDEF0),
            # lo matches, hi differs (same lane 0 bit)
            (0x12345679, 0x00000001, 0x12345679, 0x00000002),
            # hi matches, lo differs
            (0x00000003, 0x80000001, 0x00000005, 0x80000001),
            # Exact match with 0x00000000_12345678 immediate
            (0x12345678, 0x00000000, 0x12345678, 0x00000000),
            # lo matches 0x12345678, hi non-zero -> != 0x12345678
            (0x12345678, 0x00000001, 0x00000000, 0x00000000),
        ]
        for lo, hi, s6, s7 in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), s6=hex(s6), s7=hex(s7)):
                mem = run_cs(self.spv, 64, s4=lo, s5=hi, s6=s6, s7=s7)
                eq_pair = int(lo == s6 and hi == s7)
                eq_imm = int(lo == 0x12345678 and hi == 0)
                want = [eq_pair, 1 - eq_pair, eq_pair, eq_imm, 1 - eq_imm]
                self.assertEqual([mem[4 * k] for k in range(5)], want)


class FastPathPreservationTests(unittest.TestCase):
    def test_same_half_and_zero_imm_keep_vote_fast_path(self):
        asm = (
            WAVE64_BUILD_VCC_AND_SGPR
            + "s_cmp_eq_u32 vcc_lo, 0\n"
            + "s_cmp_eq_u32 vcc_lo, exec_lo\n"
            + "s_cmp_lg_u32 vcc_hi, 0\n"
            + "s_cmp_eq_u64 vcc, exec\n"
            + "s_endpgm\n"
        )
        spv = translate_cs("cs_w64_fast_vote", asm, threads=64)
        self.assertIn("OpGroupNonUniformAll", spv)
        self.assertIn("OpGroupNonUniformAny", spv)
        self.assertNotIn("OpGroupNonUniformBitwiseOr", spv)

    def test_opposite_halves_and_scalar_use_wave_mask_not_vote(self):
        prologue = WAVE64_BUILD_VCC_AND_SGPR.replace("s_and_b64 s[12:13], vcc, exec", "s_mov_b64 s[12:13], vcc")
        for name, cmp_op in (
            ("opp", "s_cmp_eq_u32 vcc_lo, vcc_hi"),
            ("imm", "s_cmp_eq_u32 vcc_lo, 0x80000003"),
            ("ord", "s_cmp_lt_u32 vcc_lo, s6"),
            ("u64", "s_cmp_eq_u64 vcc, s[6:7]"),
            # -1 (all ones): a vote misses lanes that were never launched, whose bits are 0 in the mask
            ("ones32", "s_cmp_lg_u32 vcc_hi, -1"),
            ("ones64", "s_cmp_eq_u64 vcc, -1"),
        ):
            with self.subTest(case=name):
                spv = translate_cs(f"cs_w64_wm_{name}", prologue + cmp_op + "\ns_endpgm\n", threads=64)
                self.assertIn("OpGroupNonUniformBitwiseOr", spv)
                self.assertNotIn("OpGroupNonUniformAll", spv)
                self.assertNotIn("OpGroupNonUniformAny", spv)


if __name__ == "__main__":
    unittest.main()
