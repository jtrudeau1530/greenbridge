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
# Reference Mathematical Oracles for s_addc_u32 & s_subb_u32
# ==============================================================================

def oracle_s_addc_u32(src0: int, src1: int, cin: int = 0) -> Tuple[int, int]:
    """Reference mathematical oracle for RDNA2 s_addc_u32 / s_add_co_ci_u32.

    Computes 32-bit unsigned addition with carry-in and carry-out:
        D = (S0 + S1 + SCC.in) mod 2^32
        SCC.out = 1 if (S0 + S1 + SCC.in >= 2^32) else 0
    """
    s0 = src0 & MASK_U32
    s1 = src1 & MASK_U32
    c = 1 if cin else 0
    raw_sum = s0 + s1 + c
    res = raw_sum & MASK_U32
    carry_out = 1 if raw_sum >= MODULO_U32 else 0
    return res, carry_out


def oracle_s_subb_u32(src0: int, src1: int, bin_val: int = 0) -> Tuple[int, int]:
    """Reference mathematical oracle for RDNA2 s_subb_u32 / s_sub_co_ci_u32.

    Computes 32-bit unsigned subtraction with borrow-in and borrow-out:
        D = (S0 - S1 - SCC.in) mod 2^32
        SCC.out = 1 if (S0 < S1 + SCC.in) else 0
    """
    s0 = src0 & MASK_U32
    s1 = src1 & MASK_U32
    b = 1 if bin_val else 0
    raw_diff = s0 - s1 - b
    res = raw_diff & MASK_U32
    borrow_out = 1 if s0 < (s1 + b) else 0
    return res, borrow_out


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
# Unit Tests: Reference Oracles & Mathematical Properties
# ==============================================================================

class TestReferenceOracleAddcU32(unittest.TestCase):
    """Exhaustive mathematical verification for oracle_s_addc_u32."""

    def test_zero_identities_without_carry_in(self) -> None:
        self.assertEqual(oracle_s_addc_u32(0, 0, cin=0), (0, 0))
        self.assertEqual(oracle_s_addc_u32(42, 0, cin=0), (42, 0))
        self.assertEqual(oracle_s_addc_u32(0, 1337, cin=0), (1337, 0))
        self.assertEqual(oracle_s_addc_u32(MAX_U32, 0, cin=0), (MAX_U32, 0))

    def test_zero_identities_with_carry_in(self) -> None:
        self.assertEqual(oracle_s_addc_u32(0, 0, cin=1), (1, 0))
        self.assertEqual(oracle_s_addc_u32(42, 0, cin=1), (43, 0))
        self.assertEqual(oracle_s_addc_u32(0, 1337, cin=1), (1338, 0))
        # MAX_U32 + 0 + 1 = 0 with carry out = 1
        self.assertEqual(oracle_s_addc_u32(MAX_U32, 0, cin=1), (0, 1))

    def test_commutativity(self) -> None:
        vectors = [
            (0, 1),
            (100, 200),
            (MAX_U32, 1),
            (0x8000_0000, 0x8000_0000),
            (0x1234_5678, 0x9ABC_DEF0),
        ]
        for cin in (0, 1):
            for a, b in vectors:
                with self.subTest(a=a, b=b, cin=cin):
                    res_ab, c_ab = oracle_s_addc_u32(a, b, cin)
                    res_ba, c_ba = oracle_s_addc_u32(b, a, cin)
                    self.assertEqual(res_ab, res_ba)
                    self.assertEqual(c_ab, c_ba)

    def test_boundary_carry_transitions(self) -> None:
        # MAX_U32 - 1 + 1 with cin=0 => MAX_U32, carry=0
        self.assertEqual(oracle_s_addc_u32(MAX_U32 - 1, 1, cin=0), (MAX_U32, 0))
        # MAX_U32 - 1 + 1 with cin=1 => 0, carry=1
        self.assertEqual(oracle_s_addc_u32(MAX_U32 - 1, 1, cin=1), (0, 1))
        # MAX_U32 + MAX_U32 + cin=1 => MAX_U32, carry=1
        self.assertEqual(oracle_s_addc_u32(MAX_U32, MAX_U32, cin=1), (MAX_U32, 1))


class TestReferenceOracleSubbU32(unittest.TestCase):
    """Exhaustive mathematical verification for oracle_s_subb_u32."""

    def test_zero_identities(self) -> None:
        self.assertEqual(oracle_s_subb_u32(0, 0, bin_val=0), (0, 0))
        self.assertEqual(oracle_s_subb_u32(42, 0, bin_val=0), (42, 0))
        self.assertEqual(oracle_s_subb_u32(42, 0, bin_val=1), (41, 0))
        self.assertEqual(oracle_s_subb_u32(0, 0, bin_val=1), (MAX_U32, 1))

    def test_self_subtraction(self) -> None:
        values = [0, 1, 42, 0x7FFF_FFFF, 0x8000_0000, MAX_U32]
        for v in values:
            with self.subTest(v=v):
                # v - v - 0 = 0, borrow 0
                res, borrow = oracle_s_subb_u32(v, v, bin_val=0)
                self.assertEqual(res, 0)
                self.assertEqual(borrow, 0)
                # v - v - 1 = MAX_U32, borrow 1
                res_b, borrow_b = oracle_s_subb_u32(v, v, bin_val=1)
                self.assertEqual(res_b, MAX_U32)
                self.assertEqual(borrow_b, 1)

    def test_boundary_borrow_transitions(self) -> None:
        self.assertEqual(oracle_s_subb_u32(0, 1, bin_val=0), (MAX_U32, 1))
        self.assertEqual(oracle_s_subb_u32(0, 1, bin_val=1), (MAX_U32 - 1, 1))
        self.assertEqual(oracle_s_subb_u32(1, 1, bin_val=1), (MAX_U32, 1))
        self.assertEqual(oracle_s_subb_u32(2, 1, bin_val=1), (0, 0))


class TestArchitecturalModel(unittest.TestCase):
    """Verification of the ScalarModel register file and SCC status representation."""

    def test_sgpr_read_write(self) -> None:
        model = ScalarModel()
        model.write_sgpr(5, 0x1234_5678)
        self.assertEqual(model.read_sgpr(5), 0x1234_5678)
        model.write_sgpr(6, 0x1_AAAA_BBBB)
        self.assertEqual(model.read_sgpr(6), 0xAAAA_BBBB)

    def test_sgpr_bounds(self) -> None:
        model = ScalarModel()
        with self.assertRaises(IndexError):
            model.read_sgpr(-1)
        with self.assertRaises(IndexError):
            model.read_sgpr(NUM_SGPRS)

    def test_scc_flag(self) -> None:
        model = ScalarModel()
        self.assertEqual(model.scc, 0)
        model.set_scc(1)
        self.assertEqual(model.scc, 1)
        model.set_scc(0)
        self.assertEqual(model.scc, 0)


if __name__ == "__main__":
    unittest.main()
