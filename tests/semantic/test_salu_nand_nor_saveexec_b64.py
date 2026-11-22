"""Semantic reference oracle and verification for RDNA2 s_nand_saveexec_b64 and s_nor_saveexec_b64.

RDNA2 SALU 64-bit SaveExec Instructions:
  s_nand_saveexec_b64:
    SDST = EXEC
    EXEC = ~(SSRC0 & EXEC) & 0xFFFFFFFFFFFFFFFF
    SCC  = 1 if EXEC != 0 else 0

  s_nor_saveexec_b64:
    SDST = EXEC
    EXEC = ~(SSRC0 | EXEC) & 0xFFFFFFFFFFFFFFFF
    SCC  = 1 if EXEC != 0 else 0

Phase 1: Reference mathematical oracle and architectural model.
"""
from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union, Dict, Any

# Bitwise 32-bit and 64-bit architectural masks
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
ALL64: int = 0xFFFF_FFFF_FFFF_FFFF

# IEEE 754 binary64 (double precision) architectural constants
SIGN_MASK_F64: int = 0x8000_0000_0000_0000
EXP_MASK_F64: int = 0x7FF0_0000_0000_0000
FRAC_MASK_F64: int = 0x000F_FFFF_FFFF_FFFF
QNAN_BIT_F64: int = 0x0008_0000_0000_0000
CANONICAL_QNAN_F64: int = 0x7FF8_0000_0000_0000

POS_ZERO_F64: int = 0x0000_0000_0000_0000
NEG_ZERO_F64: int = 0x8000_0000_0000_0000
POS_INF_F64: int = 0x7FF0_0000_0000_0000
NEG_INF_F64: int = 0xFFF0_0000_0000_0000


def words_to_u64(lo: int, hi: int) -> int:
    """Combine two 32-bit scalar register words (lo, hi) into a 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def u64_to_words(val: int) -> Tuple[int, int]:
    """Split a 64-bit integer into two 32-bit scalar register words (lo, hi)."""
    return val & MASK32, (val >> 32) & MASK32


def f64_to_bits(val: float) -> int:
    """Convert a Python float (IEEE 754 double) to 64-bit integer bits."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def bits_to_f64(bits: int) -> float:
    """Convert 64-bit integer bits to a Python float (IEEE 754 double)."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def popcount64(val: int) -> int:
    """Return the number of set bits in a 64-bit integer."""
    return bin(val & MASK64).count("1")


def find_first_one_64(val: int) -> int:
    """Find the lowest set bit index [0..63] in a 64-bit integer, or -1 if zero."""
    v = val & MASK64
    if v == 0:
        return -1
    return (v & -v).bit_length() - 1


# ==============================================================================
# Reference Mathematical Oracles
# ==============================================================================

def oracle_s_nand_b64(s0: int, s1: int) -> int:
    """Reference mathematical oracle for 64-bit bitwise NAND: D = ~(s0 & s1)."""
    return (~(s0 & s1)) & MASK64


def oracle_s_nor_b64(s0: int, s1: int) -> int:
    """Reference mathematical oracle for 64-bit bitwise NOR: D = ~(s0 | s1)."""
    return (~(s0 | s1)) & MASK64


def oracle_s_nand_saveexec_b64(ssrc0: int, old_exec: int) -> Tuple[int, int, int]:
    """Reference architectural oracle for RDNA2 s_nand_saveexec_b64.

    Semantics:
      saved_dst = old_exec
      new_exec  = ~(ssrc0 & old_exec) & 0xFFFFFFFFFFFFFFFF
      scc       = 1 if new_exec != 0 else 0

    Returns:
      (saved_dst, new_exec, scc)
    """
    saved_dst = old_exec & MASK64
    new_exec = (~(ssrc0 & old_exec)) & MASK64
    scc = 1 if new_exec != 0 else 0
    return saved_dst, new_exec, scc


def oracle_s_nor_saveexec_b64(ssrc0: int, old_exec: int) -> Tuple[int, int, int]:
    """Reference architectural oracle for RDNA2 s_nor_saveexec_b64.

    Semantics:
      saved_dst = old_exec
      new_exec  = ~(ssrc0 | old_exec) & 0xFFFFFFFFFFFFFFFF
      scc       = 1 if new_exec != 0 else 0

    Returns:
      (saved_dst, new_exec, scc)
    """
    saved_dst = old_exec & MASK64
    new_exec = (~(ssrc0 | old_exec)) & MASK64
    scc = 1 if new_exec != 0 else 0
    return saved_dst, new_exec, scc


# ==============================================================================
# Phase 1 Unit Tests: Oracle and Architectural Model
# ==============================================================================

class TestOracleAndModel(unittest.TestCase):
    """Verify bitwise mathematical properties and reference oracle definitions."""

    def test_truth_table_exhaustive(self) -> None:
        """Exhaustive 2-bit truth table verification for NAND and NOR."""
        # Bit combinations: (0,0), (0,1), (1,0), (1,1)
        # NAND truth table: ~(0&0)=1, ~(0&1)=1, ~(1&0)=1, ~(1&1)=0
        self.assertEqual(oracle_s_nand_b64(0, 0) & 1, 1)
        self.assertEqual(oracle_s_nand_b64(0, 1) & 1, 1)
        self.assertEqual(oracle_s_nand_b64(1, 0) & 1, 1)
        self.assertEqual(oracle_s_nand_b64(1, 1) & 1, 0)

        # NOR truth table: ~(0|0)=1, ~(0|1)=0, ~(1|0)=0, ~(1|1)=0
        self.assertEqual(oracle_s_nor_b64(0, 0) & 1, 1)
        self.assertEqual(oracle_s_nor_b64(0, 1) & 1, 0)
        self.assertEqual(oracle_s_nor_b64(1, 0) & 1, 0)
        self.assertEqual(oracle_s_nor_b64(1, 1) & 1, 0)

    def test_nand_identities(self) -> None:
        """Verify identity, complementation, and zero absorption for NAND."""
        val = 0x0123_4567_89AB_CDEF
        # NAND with 0 yields all ones (111...1)
        self.assertEqual(oracle_s_nand_b64(val, 0), ALL64)
        self.assertEqual(oracle_s_nand_b64(0, val), ALL64)
        # NAND with all ones yields bitwise NOT
        self.assertEqual(oracle_s_nand_b64(val, ALL64), (~val) & MASK64)
        self.assertEqual(oracle_s_nand_b64(ALL64, val), (~val) & MASK64)
        # NAND with itself yields bitwise NOT
        self.assertEqual(oracle_s_nand_b64(val, val), (~val) & MASK64)

    def test_nor_identities(self) -> None:
        """Verify identity, complementation, and zero absorption for NOR."""
        val = 0x0123_4567_89AB_CDEF
        # NOR with 0 yields bitwise NOT
        self.assertEqual(oracle_s_nor_b64(val, 0), (~val) & MASK64)
        self.assertEqual(oracle_s_nor_b64(0, val), (~val) & MASK64)
        # NOR with all ones yields 0
        self.assertEqual(oracle_s_nor_b64(val, ALL64), 0)
        self.assertEqual(oracle_s_nor_b64(ALL64, val), 0)
        # NOR with itself yields bitwise NOT
        self.assertEqual(oracle_s_nor_b64(val, val), (~val) & MASK64)

    def test_de_morgan_duality(self) -> None:
        """Verify De Morgan's laws: ~(A & B) == ~A | ~B and ~(A | B) == ~A & ~B."""
        patterns = [
            (0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555),
            (0xFFFF_0000_FFFF_0000, 0x0000_FFFF_0000_FFFF),
            (0x1234_5678_9ABC_DEF0, 0xFEDC_BA98_7654_3210),
            (0x8000_0000_0000_0000, 0x0000_0000_0000_0001),
        ]
        for a, b in patterns:
            not_a = (~a) & MASK64
            not_b = (~b) & MASK64
            # NAND duality: ~(A & B) == ~A | ~B
            self.assertEqual(oracle_s_nand_b64(a, b), (not_a | not_b) & MASK64)
            # NOR duality: ~(A | B) == ~A & ~B
            self.assertEqual(oracle_s_nor_b64(a, b), (not_a & not_b) & MASK64)

    def test_nand_saveexec_oracle_semantics(self) -> None:
        """Verify s_nand_saveexec_b64 preserves old EXEC in SDST and computes correct new EXEC and SCC."""
        old_exec = 0xFFFF_0000_FFFF_0000
        ssrc0 = 0xAAAA_AAAA_AAAA_AAAA
        saved, new_exec, scc = oracle_s_nand_saveexec_b64(ssrc0, old_exec)

        self.assertEqual(saved, old_exec)
        expected_new_exec = (~(ssrc0 & old_exec)) & MASK64
        self.assertEqual(new_exec, expected_new_exec)
        self.assertEqual(scc, 1 if expected_new_exec != 0 else 0)

        # Test case where new EXEC becomes 0:
        # ~(ssrc0 & old_exec) == 0 iff (ssrc0 & old_exec) == ALL64
        saved_zero, new_exec_zero, scc_zero = oracle_s_nand_saveexec_b64(ALL64, ALL64)
        self.assertEqual(saved_zero, ALL64)
        self.assertEqual(new_exec_zero, 0)
        self.assertEqual(scc_zero, 0)

    def test_nor_saveexec_oracle_semantics(self) -> None:
        """Verify s_nor_saveexec_b64 preserves old EXEC in SDST and computes correct new EXEC and SCC."""
        old_exec = 0x0000_0000_FFFF_FFFF
        ssrc0 = 0xFFFF_0000_0000_0000
        saved, new_exec, scc = oracle_s_nor_saveexec_b64(ssrc0, old_exec)

        self.assertEqual(saved, old_exec)
        expected_new_exec = (~(ssrc0 | old_exec)) & MASK64
        self.assertEqual(new_exec, expected_new_exec)
        self.assertEqual(scc, 1 if expected_new_exec != 0 else 0)

        # Test case where new EXEC becomes 0:
        # ~(ssrc0 | old_exec) == 0 whenever ssrc0 | old_exec == ALL64
        saved_zero, new_exec_zero, scc_zero = oracle_s_nor_saveexec_b64(0xFFFF_FFFF_0000_0000, 0x0000_0000_FFFF_FFFF)
        self.assertEqual(saved_zero, 0x0000_0000_FFFF_FFFF)
        self.assertEqual(new_exec_zero, 0)
        self.assertEqual(scc_zero, 0)

        # Test case where old_exec is 0 and ssrc0 is 0:
        # ~(0 | 0) = ALL64, so new EXEC is ALL64, SCC is 1
        saved_full, new_exec_full, scc_full = oracle_s_nor_saveexec_b64(0, 0)
        self.assertEqual(saved_full, 0)
        self.assertEqual(new_exec_full, ALL64)
        self.assertEqual(scc_full, 1)


if __name__ == "__main__":
    unittest.main()
