from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

# ==============================================================================
# RDNA2 SALU Architectural Constants
# ==============================================================================

MASK_U32: int = 0xFFFF_FFFF
MODULO_U32: int = 0x1_0000_0000
SIGN_BIT_U32: int = 0x8000_0000
MAX_U32: int = 0xFFFF_FFFF
NUM_SGPRS: int = 256


# ==============================================================================
# Reference Mathematical Oracles for s_add_co_u32 & s_sub_co_u32
# ==============================================================================

def oracle_s_add_co_u32(src0: int, src1: int) -> Tuple[int, int]:
    """Reference mathematical oracle for RDNA2 s_add_co_u32 / s_add_u32.

    Computes 32-bit unsigned addition:
        D = (S0 + S1) mod 2^32
        SCC = 1 if (S0 + S1 >= 2^32) else 0 (Carry Out)
    """
    s0 = src0 & MASK_U32
    s1 = src1 & MASK_U32
    raw_sum = s0 + s1
    res = raw_sum & MASK_U32
    carry = 1 if raw_sum >= MODULO_U32 else 0
    return res, carry


def oracle_s_sub_co_u32(src0: int, src1: int) -> Tuple[int, int]:
    """Reference mathematical oracle for RDNA2 s_sub_co_u32 / s_sub_u32.

    Computes 32-bit unsigned subtraction:
        D = (S0 - S1) mod 2^32
        SCC = 1 if (S0 < S1) else 0 (Borrow Out)
    """
    s0 = src0 & MASK_U32
    s1 = src1 & MASK_U32
    raw_diff = s0 - s1
    res = raw_diff & MASK_U32
    borrow = 1 if s0 < s1 else 0
    return res, borrow


# ==============================================================================
# Architectural Hardware Model
# ==============================================================================

@dataclass
class ScalarModel:
    """Architectural model representing the RDNA2 Scalar ALU (SALU) register file and status."""
    sgprs: List[int] = field(default_factory=lambda: [0] * NUM_SGPRS)
    scc: int = 0

    def read_sgpr(self, index: int) -> int:
        if not (0 <= index < NUM_SGPRS):
            raise IndexError(f"SGPR index {index} out of bounds [0, {NUM_SGPRS - 1}]")
        return self.sgprs[index] & MASK_U32

    def write_sgpr(self, index: int, val: int) -> None:
        if not (0 <= index < NUM_SGPRS):
            raise IndexError(f"SGPR index {index} out of bounds [0, {NUM_SGPRS - 1}]")
        self.sgprs[index] = val & MASK_U32

    def set_scc(self, val: int) -> None:
        self.scc = 1 if val else 0


# ==============================================================================
# Phase 1 Unit Tests: Reference Oracles & Mathematical Properties
# ==============================================================================

class TestReferenceOracleAddCoU32(unittest.TestCase):
    """Exhaustive mathematical verification for oracle_s_add_co_u32."""

    def test_zero_identities(self) -> None:
        self.assertEqual(oracle_s_add_co_u32(0, 0), (0, 0))
        self.assertEqual(oracle_s_add_co_u32(42, 0), (42, 0))
        self.assertEqual(oracle_s_add_co_u32(0, 1337), (1337, 0))
        self.assertEqual(oracle_s_add_co_u32(MAX_U32, 0), (MAX_U32, 0))

    def test_commutativity(self) -> None:
        vectors = [
            (0, 1),
            (100, 200),
            (MAX_U32, 1),
            (0x8000_0000, 0x8000_0000),
            (0x1234_5678, 0x9ABC_DEF0),
        ]
        for a, b in vectors:
            with self.subTest(a=a, b=b):
                res_ab, c_ab = oracle_s_add_co_u32(a, b)
                res_ba, c_ba = oracle_s_add_co_u32(b, a)
                self.assertEqual(res_ab, res_ba)
                self.assertEqual(c_ab, c_ba)

    def test_boundary_carry_transitions(self) -> None:
        # Just before carry
        self.assertEqual(oracle_s_add_co_u32(MAX_U32 - 1, 1), (MAX_U32, 0))
        self.assertEqual(oracle_s_add_co_u32(0x7FFF_FFFF, 0x8000_0000), (MAX_U32, 0))
        # Exact carry threshold
        self.assertEqual(oracle_s_add_co_u32(MAX_U32, 1), (0, 1))
        self.assertEqual(oracle_s_add_co_u32(1, MAX_U32), (0, 1))
        self.assertEqual(oracle_s_add_co_u32(MAX_U32, 2), (1, 1))
        # Both high operands
        self.assertEqual(oracle_s_add_co_u32(MAX_U32, MAX_U32), (MAX_U32 - 1, 1))
        self.assertEqual(oracle_s_add_co_u32(0x8000_0000, 0x8000_0000), (0, 1))
        self.assertEqual(oracle_s_add_co_u32(0x8000_0000, 0x8000_0001), (1, 1))


class TestReferenceOracleSubCoU32(unittest.TestCase):
    """Exhaustive mathematical verification for oracle_s_sub_co_u32."""

    def test_zero_identities(self) -> None:
        self.assertEqual(oracle_s_sub_co_u32(0, 0), (0, 0))
        self.assertEqual(oracle_s_sub_co_u32(42, 0), (42, 0))
        self.assertEqual(oracle_s_sub_co_u32(MAX_U32, 0), (MAX_U32, 0))

    def test_self_subtraction(self) -> None:
        values = [0, 1, 42, 0x7FFF_FFFF, 0x8000_0000, MAX_U32]
        for v in values:
            with self.subTest(v=v):
                res, borrow = oracle_s_sub_co_u32(v, v)
                self.assertEqual(res, 0)
                self.assertEqual(borrow, 0)

    def test_boundary_borrow_transitions(self) -> None:
        # Minimal borrow cases: 0 - 1
        self.assertEqual(oracle_s_sub_co_u32(0, 1), (MAX_U32, 1))
        self.assertEqual(oracle_s_sub_co_u32(0, MAX_U32), (1, 1))
        self.assertEqual(oracle_s_sub_co_u32(1, 2), (MAX_U32, 1))
        # Borrow from 0x8000_0000
        self.assertEqual(oracle_s_sub_co_u32(0x8000_0000, 1), (0x7FFF_FFFF, 0))
        self.assertEqual(oracle_s_sub_co_u32(0x7FFF_FFFF, 0x8000_0000), (MAX_U32, 1))
        # High boundary subtractions without borrow
        self.assertEqual(oracle_s_sub_co_u32(MAX_U32, 1), (MAX_U32 - 1, 0))
        self.assertEqual(oracle_s_sub_co_u32(MAX_U32, MAX_U32), (0, 0))

    def test_add_sub_inverse_relation(self) -> None:
        pairs = [
            (10, 5),
            (5, 10),
            (0, 100),
            (MAX_U32, 12345),
            (0x8000_0000, 0x1234_5678),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                diff, _ = oracle_s_sub_co_u32(a, b)
                recovered, _ = oracle_s_add_co_u32(diff, b)
                self.assertEqual(recovered, a & MASK_U32)


class TestArchitecturalModel(unittest.TestCase):
    """Verification of the ScalarModel register file and SCC status representation."""

    def test_sgpr_read_write(self) -> None:
        model = ScalarModel()
        self.assertEqual(model.read_sgpr(0), 0)
        model.write_sgpr(0, 0xDEAD_BEEF)
        self.assertEqual(model.read_sgpr(0), 0xDEAD_BEEF)
        # Verify 32-bit truncation
        model.write_sgpr(1, 0x1_1234_5678)
        self.assertEqual(model.read_sgpr(1), 0x1234_5678)

    def test_sgpr_bounds(self) -> None:
        model = ScalarModel()
        with self.assertRaises(IndexError):
            model.read_sgpr(-1)
        with self.assertRaises(IndexError):
            model.read_sgpr(NUM_SGPRS)
        with self.assertRaises(IndexError):
            model.write_sgpr(NUM_SGPRS + 10, 1)

    def test_scc_flag(self) -> None:
        model = ScalarModel()
        self.assertEqual(model.scc, 0)
        model.set_scc(1)
        self.assertEqual(model.scc, 1)
        model.set_scc(0)
        self.assertEqual(model.scc, 0)
        model.set_scc(42)
        self.assertEqual(model.scc, 1)


if __name__ == "__main__":
    unittest.main()
