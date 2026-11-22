"""Semantic reference tests and architectural model for AMD RDNA2 s_xnor_saveexec_b64.

AMD RDNA2 (GFX10/GFX1030) Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 ISA, scalar instructions operate across 32-bit Scalar General-Purpose
  Registers (SGPRs), the 64-bit Execution Mask (EXEC / exec_lo, exec_hi), the Scalar
  Condition Code (SCC), and special system registers (VCC, M0).

  Instruction:
    s_xnor_saveexec_b64 sdst, ssrc0

  Architectural Operation:
    1. The incoming 64-bit EXEC register is saved to the destination scalar register:
         SDST = EXEC
    2. The new 64-bit EXEC mask is calculated as the bitwise equivalence (XNOR) of
       the source operand SSRC0 and the prior EXEC mask:
         NEW_EXEC = ~(SSRC0 ^ EXEC) & 0xFFFFFFFFFFFFFFFF
    3. The EXEC register is updated with the new mask:
         EXEC = NEW_EXEC
    4. The Scalar Condition Code (SCC) is updated based on whether the new EXEC mask
       has any active lanes enabled:
         SCC = 1 if (NEW_EXEC != 0) else 0

  Bitwise XNOR Truth Table:
    A | B | A XNOR B
    --+---+---------
    0 | 0 |    1
    0 | 1 |    0
    1 | 0 |    0
    1 | 1 |    1

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

import os
import random
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT64: int = 0x8000000000000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def split_dwords(val64: int) -> Tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def popcount64(val: int) -> int:
    """Count the number of set bits in an unsigned 64-bit integer."""
    return bin(val & MASK64).count("1")


def ref_xnor64(a: int, b: int) -> int:
    """Reference mathematical oracle for 64-bit bitwise XNOR: ~(a ^ b)."""
    return (~(a ^ b)) & MASK64


def ref_scc(val64: int) -> int:
    """RDNA2 SALU condition flag: SCC is 1 if 64-bit value is non-zero, else 0."""
    return 1 if (val64 & MASK64) != 0 else 0


def oracle_s_xnor_saveexec_b64(src0: int, exec_current: int) -> Tuple[int, int, int]:
    """Reference architectural oracle for s_xnor_saveexec_b64.

    Returns:
        (saved_exec, new_exec, scc)
    """
    saved_exec = exec_current & MASK64
    new_exec = ref_xnor64(src0, exec_current)
    scc = ref_scc(new_exec)
    return saved_exec, new_exec, scc


class TestSXnorSaveexecB64ModelAndOracle(unittest.TestCase):
    """Reference mathematical oracle and architectural model tests."""

    def test_truth_table_all_bits(self) -> None:
        """Verify the 1-bit truth table of XNOR across all 4 input combinations."""
        cases = [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (1, 1, 1),
        ]
        for b0, b1, expected in cases:
            with self.subTest(b0=b0, b1=b1):
                s0 = MASK64 if b0 else 0
                s1 = MASK64 if b1 else 0
                want = MASK64 if expected else 0
                got = ref_xnor64(s0, s1)
                self.assertEqual(got, want)
                self.assertEqual(ref_scc(got), 1 if expected else 0)

    def test_commutativity_symmetry(self) -> None:
        """Verify that 64-bit XNOR is strictly commutative."""
        rng = random.Random(0x1337)
        for _ in range(50):
            a = rng.getrandbits(64)
            b = rng.getrandbits(64)
            self.assertEqual(ref_xnor64(a, b), ref_xnor64(b, a))

    def test_algebraic_identities(self) -> None:
        """Verify core boolean algebraic identities of the 64-bit XNOR operator."""
        rng = random.Random(0xCAFE)
        for _ in range(50):
            x = rng.getrandbits(64)
            # Identity: x XNOR 0xFFFFFFFFFFFFFFFF == x
            self.assertEqual(ref_xnor64(x, MASK64), x)
            # Negation: x XNOR 0 == ~x
            self.assertEqual(ref_xnor64(x, 0), (~x) & MASK64)
            # Self-equivalence: x XNOR x == 0xFFFFFFFFFFFFFFFF
            self.assertEqual(ref_xnor64(x, x), MASK64)
            # Contradiction: x XNOR (~x) == 0
            self.assertEqual(ref_xnor64(x, (~x) & MASK64), 0)
            # Involution with XOR: (x XNOR y) XNOR 0 == x ^ y
            y = rng.getrandbits(64)
            self.assertEqual(ref_xnor64(ref_xnor64(x, y), 0), (x ^ y) & MASK64)

    def test_de_morgan_and_boolean_expansion(self) -> None:
        """Verify equivalence to canonical SOP form: (A & B) | (~A & ~B)."""
        rng = random.Random(0xBEEF)
        for _ in range(50):
            a = rng.getrandbits(64)
            b = rng.getrandbits(64)
            expanded = ((a & b) | ((~a & MASK64) & (~b & MASK64))) & MASK64
            self.assertEqual(ref_xnor64(a, b), expanded)

    def test_scc_flag_truth(self) -> None:
        """Verify SCC condition code behavior on exact zero and non-zero boundaries."""
        self.assertEqual(ref_scc(0), 0)
        self.assertEqual(ref_scc(1), 1)
        self.assertEqual(ref_scc(MASK64), 1)
        self.assertEqual(ref_scc(SIGN_BIT64), 1)
        for i in range(64):
            self.assertEqual(ref_scc(1 << i), 1)

    def test_saveexec_architectural_contract(self) -> None:
        """Verify the three-part architectural return contract of s_xnor_saveexec_b64."""
        test_cases = [
            (0x0000000000000000, 0x0000000000000000, 0x0000000000000000, MASK64, 1),
            (MASK64, MASK64, MASK64, MASK64, 1),
            (0x0000000000000000, MASK64, MASK64, 0x0000000000000000, 0),
            (MASK64, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0),
            (0xAAAAAAAAAAAAAAAA, 0x5555555555555555, 0x5555555555555555, 0x0000000000000000, 0),
            (0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, MASK64, 1),
            (0x1234567890ABCDEF, 0xFEDCBA0987654321, 0xFEDCBA0987654321,
             (~(0x1234567890ABCDEF ^ 0xFEDCBA0987654321)) & MASK64, 1),
        ]
        for src0, exec_curr, want_saved, want_new, want_scc in test_cases:
            with self.subTest(src0=hex(src0), exec_curr=hex(exec_curr)):
                saved, new_exec, scc = oracle_s_xnor_saveexec_b64(src0, exec_curr)
                self.assertEqual(saved, want_saved)
                self.assertEqual(new_exec, want_new)
                self.assertEqual(scc, want_scc)

    def test_walking_bits(self) -> None:
        """Verify single walking bit across all 64 lane positions."""
        for i in range(64):
            bit = 1 << i
            # When src0 == bit and exec == bit: bit matches bit -> XNOR bit is 1, all others 1
            saved, new_exec, scc = oracle_s_xnor_saveexec_b64(bit, bit)
            self.assertEqual(saved, bit)
            self.assertEqual(new_exec, MASK64)
            self.assertEqual(scc, 1)

            # When src0 == bit and exec == ~bit: bit position differs -> XNOR bit is 0, all others 0 -> new_exec == 0
            saved, new_exec, scc = oracle_s_xnor_saveexec_b64(bit, (~bit) & MASK64)
            self.assertEqual(saved, (~bit) & MASK64)
            self.assertEqual(new_exec, 0)
            self.assertEqual(scc, 0)

    def test_fuzz_random_vectors(self) -> None:
        """Fuzz testing with 200 random 64-bit bitmasks."""
        rng = random.Random(0x9999)
        for _ in range(200):
            src = rng.getrandbits(64)
            curr = rng.getrandbits(64)
            saved, new_exec, scc = oracle_s_xnor_saveexec_b64(src, curr)
            self.assertEqual(saved, curr)
            expected_new = (~(src ^ curr)) & MASK64
            self.assertEqual(new_exec, expected_new)
            self.assertEqual(scc, 1 if expected_new != 0 else 0)


if __name__ == "__main__":
    unittest.main()
