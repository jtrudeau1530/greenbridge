"""Semantic reference oracle and verification for RDNA2 64-bit inverted scalar logic.

Covers:
  - s_andn2_b64: D = S0 & (~S1)
  - s_orn2_b64:  D = S0 | (~S1)
  - s_nand_b64:  D = ~(S0 & S1)
  - s_nor_b64:   D = ~(S0 | S1)
  - s_xnor_b64:  D = ~(S0 ^ S1)
  - Condition code SCC: SCC = 1 if D != 0 else 0

Tests standard library bitwise arithmetic, truth table exhaustion, boundary patterns,
four-corner SCC zero/nonzero matrix, and non-commutative operand asymmetry.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def ref_andn2_b64(s0: int, s1: int) -> int:
    """RDNA2 s_andn2_b64: D = S0 & (~S1)."""
    return (s0 & (~s1 & MASK64)) & MASK64


def ref_orn2_b64(s0: int, s1: int) -> int:
    """RDNA2 s_orn2_b64: D = S0 | (~S1)."""
    return (s0 | (~s1 & MASK64)) & MASK64


def ref_nand_b64(s0: int, s1: int) -> int:
    """RDNA2 s_nand_b64: D = ~(S0 & S1)."""
    return (~(s0 & s1)) & MASK64


def ref_nor_b64(s0: int, s1: int) -> int:
    """RDNA2 s_nor_b64: D = ~(S0 | S1)."""
    return (~(s0 | s1)) & MASK64


def ref_xnor_b64(s0: int, s1: int) -> int:
    """RDNA2 s_xnor_b64: D = ~(S0 ^ S1)."""
    return (~(s0 ^ s1)) & MASK64


def ref_scc(val64: int) -> int:
    """RDNA2 SCC condition flag: 1 if 64-bit result != 0, else 0."""
    return 1 if (val64 & MASK64) != 0 else 0


REF_OPS = {
    "s_andn2_b64": ref_andn2_b64,
    "s_orn2_b64": ref_orn2_b64,
    "s_nand_b64": ref_nand_b64,
    "s_nor_b64": ref_nor_b64,
    "s_xnor_b64": ref_xnor_b64,
}


class SaluInvertedLogicOracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for 64-bit inverted scalar logic."""

    def test_truth_table_bit_by_bit(self):
        """Verify 1-bit truth table combinations across all 5 inverted operations."""
        # 1-bit truth tables (s0, s1):
        # andn2: 0,0->0; 0,1->0; 1,0->1; 1,1->0
        # orn2:  0,0->1; 0,1->0; 1,0->1; 1,1->1
        # nand:  0,0->1; 0,1->1; 1,0->1; 1,1->0
        # nor:   0,0->1; 0,1->0; 1,0->0; 1,1->0
        # xnor:  0,0->1; 0,1->0; 1,0->0; 1,1->1
        truth_tables = {
            "s_andn2_b64": {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 0},
            "s_orn2_b64":  {(0, 0): 1, (0, 1): 0, (1, 0): 1, (1, 1): 1},
            "s_nand_b64":  {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0},
            "s_nor_b64":   {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0},
            "s_xnor_b64":  {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1},
        }
        for op, table in truth_tables.items():
            fn = REF_OPS[op]
            for (b0, b1), expected_bit in table.items():
                with self.subTest(op=op, b0=b0, b1=b1):
                    # Broadcast bit to all 64 bits and test
                    s0 = MASK64 if b0 else 0
                    s1 = MASK64 if b1 else 0
                    want = MASK64 if expected_bit else 0
                    got = fn(s0, s1)
                    self.assertEqual(got, want)
                    self.assertEqual(ref_scc(got), 1 if expected_bit else 0)

    def test_symmetry_and_asymmetry(self):
        """Test commutativity for nand/nor/xnor and non-commutativity for andn2/orn2."""
        a = 0xF0F0F0F0_0F0F0F0F
        b = 0x33333333_CCCCCCCC
        # nand, nor, xnor are commutative
        for op in ("s_nand_b64", "s_nor_b64", "s_xnor_b64"):
            fn = REF_OPS[op]
            with self.subTest(op=op, prop="commutative"):
                self.assertEqual(fn(a, b), fn(b, a))

        # andn2, orn2 are non-commutative
        for op in ("s_andn2_b64", "s_orn2_b64"):
            fn = REF_OPS[op]
            with self.subTest(op=op, prop="non-commutative"):
                self.assertNotEqual(fn(a, b), fn(b, a))

    def test_boundary_values(self):
        """Test edge cases with 0, MASK64, MSB, LSB, and alternating bit patterns."""
        boundaries = [
            0x00000000_00000000,
            0xFFFFFFFF_FFFFFFFF,
            0x80000000_00000000,
            0x00000000_00000001,
            0x55555555_55555555,
            0xAAAAAAAA_AAAAAAAA,
            0x00000000_FFFFFFFF,
            0xFFFFFFFF_00000000,
            0x12345678_9ABCDEF0,
            0xEDCBA987_6543210F,
        ]
        for op, fn in REF_OPS.items():
            for s0 in boundaries:
                for s1 in boundaries:
                    with self.subTest(op=op, s0=hex(s0), s1=hex(s1)):
                        res = fn(s0, s1)
                        self.assertGreaterEqual(res, 0)
                        self.assertLessEqual(res, MASK64)
                        scc = ref_scc(res)
                        self.assertIn(scc, (0, 1))
                        self.assertEqual(scc, 1 if res != 0 else 0)

    def test_scc_four_corners(self):
        """Verify SCC accurately distinguishes all four (lo, hi) zero/nonzero pairs."""
        # 1. lo == 0, hi == 0 -> SCC == 0
        self.assertEqual(ref_scc(0x00000000_00000000), 0)
        # 2. lo != 0, hi == 0 -> SCC == 1
        self.assertEqual(ref_scc(0x00000000_00000001), 1)
        self.assertEqual(ref_scc(0x00000000_80000000), 1)
        # 3. lo == 0, hi != 0 -> SCC == 1 (crucial 64-bit high dword test)
        self.assertEqual(ref_scc(0x00000001_00000000), 1)
        self.assertEqual(ref_scc(0x80000000_00000000), 1)
        # 4. lo != 0, hi != 0 -> SCC == 1
        self.assertEqual(ref_scc(0x12345678_9ABCDEF0), 1)
        self.assertEqual(ref_scc(0xFFFFFFFF_FFFFFFFF), 1)

    def test_zero_result_edge_cases(self):
        """Verify inputs that explicitly evaluate to 0 clear SCC."""
        x = 0xA5A5A5A5_5A5A5A5A
        # andn2(x, x) = x & ~x = 0
        self.assertEqual(ref_andn2_b64(x, x), 0)
        self.assertEqual(ref_scc(ref_andn2_b64(x, x)), 0)

        # andn2(0, x) = 0 & ~x = 0
        self.assertEqual(ref_andn2_b64(0, x), 0)
        self.assertEqual(ref_scc(ref_andn2_b64(0, x)), 0)

        # orn2(0, MASK64) = 0 | ~MASK64 = 0 | 0 = 0
        self.assertEqual(ref_orn2_b64(0, MASK64), 0)
        self.assertEqual(ref_scc(ref_orn2_b64(0, MASK64)), 0)

        # nand(MASK64, MASK64) = ~(MASK64 & MASK64) = 0
        self.assertEqual(ref_nand_b64(MASK64, MASK64), 0)
        self.assertEqual(ref_scc(ref_nand_b64(MASK64, MASK64)), 0)

        # nor(x, ~x) = ~(x | ~x) = ~MASK64 = 0
        not_x = (~x) & MASK64
        self.assertEqual(ref_nor_b64(x, not_x), 0)
        self.assertEqual(ref_scc(ref_nor_b64(x, not_x)), 0)

        # nor(MASK64, 0) = ~(MASK64 | 0) = 0
        self.assertEqual(ref_nor_b64(MASK64, 0), 0)
        self.assertEqual(ref_scc(ref_nor_b64(MASK64, 0)), 0)

        # xnor(x, ~x) = ~(x ^ ~x) = ~MASK64 = 0
        self.assertEqual(ref_xnor_b64(x, not_x), 0)
        self.assertEqual(ref_scc(ref_xnor_b64(x, not_x)), 0)


if __name__ == "__main__":
    unittest.main()
