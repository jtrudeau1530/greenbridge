"""Semantic tests for RDNA2 scalar bit test and zero-counting instructions.

Covers:
  - s_bitcmp0_b32: SCC = ((S0 >> (S1 & 31)) & 1) == 0
  - s_bitcmp1_b32: SCC = ((S0 >> (S1 & 31)) & 1) == 1
  - s_bitcmp0_b64: SCC = ((S0_64 >> (S1 & 63)) & 1) == 0
  - s_bitcmp1_b64: SCC = ((S0_64 >> (S1 & 63)) & 1) == 1
  - s_bcnt0_i32_b32: D = count_zero_bits(S0), SCC = (D != 0)
  - s_bcnt0_i32_b64: D = count_zero_bits(S0_64), SCC = (D != 0)
  - s_ff0_i32_b32: D = find_first_zero_bit(S0), -1 if none
  - s_ff0_i32_b64: D = find_first_zero_bit(S0_64), -1 if none
  - Lane mask awareness: testing bits on lane-mask registers (exec / vcc)
  - SPIR-V translation validation via rdna2spirv.Gen
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def ref_bitcmp0_b32(s0: int, s1: int) -> int:
    bit = ((s0 & MASK32) >> (s1 & 31)) & 1
    return 1 if bit == 0 else 0


def ref_bitcmp1_b32(s0: int, s1: int) -> int:
    bit = ((s0 & MASK32) >> (s1 & 31)) & 1
    return 1 if bit == 1 else 0


def ref_bitcmp0_b64(s0_64: int, s1: int) -> int:
    bit = ((s0_64 & MASK64) >> (s1 & 63)) & 1
    return 1 if bit == 0 else 0


def ref_bitcmp1_b64(s0_64: int, s1: int) -> int:
    bit = ((s0_64 & MASK64) >> (s1 & 63)) & 1
    return 1 if bit == 1 else 0


def ref_bcnt0_b32(s0: int) -> tuple[int, int]:
    val = (s0 & MASK32) ^ MASK32
    count = bin(val).count("1")
    return count, 1 if count != 0 else 0


def ref_bcnt0_b64(s0_64: int) -> tuple[int, int]:
    val = (s0_64 & MASK64) ^ MASK64
    count = bin(val).count("1")
    return count, 1 if count != 0 else 0


def ref_ff0_b32(s0: int) -> int:
    val = (s0 & MASK32) ^ MASK32
    if val == 0:
        return -1
    return (val & -val).bit_length() - 1


def ref_ff0_b64(s0_64: int) -> int:
    val = (s0_64 & MASK64) ^ MASK64
    if val == 0:
        return -1
    return (val & -val).bit_length() - 1


class TestBitcmpAndBcnt0Semantics(unittest.TestCase):
    """Unit tests for reference models of bit test and zero-counting instructions."""

    def test_bitcmp_b32_all_bits(self):
        for bit_idx in range(32):
            pattern = 1 << bit_idx
            self.assertEqual(ref_bitcmp1_b32(pattern, bit_idx), 1)
            self.assertEqual(ref_bitcmp0_b32(pattern, bit_idx), 0)
            # Test inverted pattern
            self.assertEqual(ref_bitcmp1_b32(~pattern, bit_idx), 0)
            self.assertEqual(ref_bitcmp0_b32(~pattern, bit_idx), 1)

    def test_bitcmp_b64_all_bits(self):
        for bit_idx in range(64):
            pattern = 1 << bit_idx
            self.assertEqual(ref_bitcmp1_b64(pattern, bit_idx), 1)
            self.assertEqual(ref_bitcmp0_b64(pattern, bit_idx), 0)
            # Test inverted pattern
            self.assertEqual(ref_bitcmp1_b64(~pattern, bit_idx), 0)
            self.assertEqual(ref_bitcmp0_b64(~pattern, bit_idx), 1)

    def test_bitcmp_modulo_shift(self):
        # 32-bit: bits [31:5] are ignored (s1 & 31)
        self.assertEqual(ref_bitcmp1_b32(0x80000000, 31 + 64), 1)
        self.assertEqual(ref_bitcmp0_b32(0x80000000, 31 + 64), 0)
        # 64-bit: bits [31:6] are ignored (s1 & 63)
        self.assertEqual(ref_bitcmp1_b64(0x8000000000000000, 63 + 128), 1)
        self.assertEqual(ref_bitcmp0_b64(0x8000000000000000, 63 + 128), 0)

    def test_bcnt0_b32_boundaries(self):
        self.assertEqual(ref_bcnt0_b32(0), (32, 1))
        self.assertEqual(ref_bcnt0_b32(MASK32), (0, 0))
        self.assertEqual(ref_bcnt0_b32(0x55555555), (16, 1))
        self.assertEqual(ref_bcnt0_b32(0x0000FFFF), (16, 1))

    def test_bcnt0_b64_boundaries(self):
        self.assertEqual(ref_bcnt0_b64(0), (64, 1))
        self.assertEqual(ref_bcnt0_b64(MASK64), (0, 0))
        self.assertEqual(ref_bcnt0_b64(0x5555555555555555), (32, 1))
        self.assertEqual(ref_bcnt0_b64(0x00000000FFFFFFFF), (32, 1))

    def test_ff0_b32(self):
        self.assertEqual(ref_ff0_b32(MASK32), -1)
        self.assertEqual(ref_ff0_b32(0), 0)
        self.assertEqual(ref_ff0_b32(1), 1)
        self.assertEqual(ref_ff0_b32(0b111), 3)
        self.assertEqual(ref_ff0_b32(0x7FFFFFFF), 31)

    def test_ff0_b64(self):
        self.assertEqual(ref_ff0_b64(MASK64), -1)
        self.assertEqual(ref_ff0_b64(0), 0)
        self.assertEqual(ref_ff0_b64(1), 1)
        self.assertEqual(ref_ff0_b64(MASK32), 32)
        self.assertEqual(ref_ff0_b64(0x7FFFFFFFFFFFFFFF), 63)


class TestRdna2SpirvBitcmpEmission(unittest.TestCase):
    """Test SPIR-V code generation in rdna2spirv for bitcmp, bcnt0, ff0."""

    def test_s_bitcmp1_b32_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bitcmp1_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "imm", "v": 7},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpShiftRightLogical %uint", emitted)
        self.assertIn("OpBitwiseAnd %uint", emitted)
        self.assertIn("OpINotEqual %bool", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bitcmp0_b32_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bitcmp0_b32",
            "args": [
                {"k": "sgpr", "r": 2},
                {"k": "sgpr", "r": 3},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpShiftRightLogical %uint", emitted)
        self.assertIn("OpBitwiseAnd %uint", emitted)
        self.assertIn("OpIEqual %bool", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bitcmp1_b64_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bitcmp1_b64",
            "args": [
                {"k": "sgpr", "r": 4, "n": 2},
                {"k": "imm", "v": 45},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpUGreaterThanEqual %bool", emitted)
        self.assertIn("OpSelect %uint", emitted)
        self.assertIn("OpShiftRightLogical %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bitcmp1_b64_exec_lane_mask(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bitcmp1_b64",
            "args": [
                {"k": "special", "t": "exec"},
                {"k": "imm", "v": 16},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpGroupNonUniformBitwiseOr", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bcnt0_i32_b32_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bcnt0_i32_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "sgpr", "r": 1},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bcnt0_i32_b64_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bcnt0_i32_b64",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "sgpr", "r": 2, "n": 2},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpIAdd %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_ff0_i32_b32_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff0_i32_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "sgpr", "r": 1},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("FindILsb", emitted)

    def test_s_ff0_i32_b64_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff0_i32_b64",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "sgpr", "r": 2, "n": 2},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("FindILsb", emitted)
        self.assertIn("OpIAdd %uint", emitted)


if __name__ == "__main__":
    unittest.main()
