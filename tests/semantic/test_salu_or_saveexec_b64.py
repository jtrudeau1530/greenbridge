from __future__ import annotations

import math
import random
import struct
import sys
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

# ==============================================================================
# Architectural Constants for AMD RDNA2 / GCN Scalar ALU
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
SIGN_BIT64: int = 0x8000_0000_0000_0000
EXP_MASK64: int = 0x7FF0_0000_0000_0000
FRAC_MASK64: int = 0x000F_FFFF_FFFF_FFFF
CANONICAL_QNAN64: int = 0x7FF8_0000_0000_0000

ALL_LANES_ACTIVE_64: int = 0xFFFF_FFFF_FFFF_FFFF
NO_LANES_ACTIVE_64: int = 0x0000_0000_0000_0000
EVEN_LANES_64: int = 0x5555_5555_5555_5555
ODD_LANES_64: int = 0xAAAA_AAAA_AAAA_AAAA
LOWER_HALF_ACTIVE_64: int = 0x0000_0000_FFFF_FFFF
UPPER_HALF_ACTIVE_64: int = 0xFFFF_FFFF_0000_0000


# ==============================================================================
# Helper Conversions and Bitwise Primitives
# ==============================================================================

def u64(val: int) -> int:
    """Mask integer to architectural unsigned 64-bit bounds."""
    return val & MASK64


def u32(val: int) -> int:
    """Mask integer to architectural unsigned 32-bit bounds."""
    return val & MASK32


def split_u64_to_u32s(val: int) -> Tuple[int, int]:
    """Split 64-bit register/mask into (lo, hi) 32-bit words."""
    v = u64(val)
    lo = v & MASK32
    hi = (v >> 32) & MASK32
    return lo, hi


def combine_u32s_to_u64(lo: int, hi: int) -> int:
    """Combine (lo, hi) 32-bit words into a 64-bit integer."""
    return u64((u32(hi) << 32) | u32(lo))


def popcount64(val: int) -> int:
    """Return count of set bits (active execution lanes) in 64-bit mask."""
    return bin(u64(val)).count("1")


def f64_to_bits(val: float) -> int:
    """Convert a Python float to 64-bit IEEE 754 bitcast."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def bits_to_f64(bits: int) -> float:
    """Convert 64-bit integer bits to Python float."""
    return struct.unpack("<d", struct.pack("<Q", u64(bits)))[0]


# ==============================================================================
# Reference Mathematical Oracle for s_or_saveexec_b64
# ==============================================================================

def oracle_s_or_saveexec_b64(exec_in: int, src_val: int) -> Tuple[int, int, int]:
    """Reference mathematical oracle for AMD RDNA2 s_or_saveexec_b64.
    
    ISA Specification (AMD RDNA2 ISA: Scalar ALU Operations):
      Instruction: S_OR_SAVEEXEC_B64 SDEST, SSRC0
      Operation:
        SDEST = EXEC
        EXEC  = SSRC0 | EXEC
        SCC   = (EXEC != 0)
        
    Args:
      exec_in: The initial 64-bit execution mask (EXEC[63:0]).
      src_val: The 64-bit scalar operand from SSRC0.
      
    Returns:
      Tuple of (saved_sdst, new_exec, scc):
        saved_sdst: The previous EXEC value saved to SDEST.
        new_exec: The updated 64-bit EXEC mask (src_val | exec_in).
        scc: Scalar Condition Code, 1 if new_exec != 0 else 0.
    """
    saved_sdst = u64(exec_in)
    new_exec = u64(src_val | exec_in)
    scc = 1 if new_exec != 0 else 0
    return saved_sdst, new_exec, scc


# ==============================================================================
# Phase 1: Reference Mathematical Oracle & Architectural Model Tests
# ==============================================================================

class TestSOrSaveExecB64Oracle(unittest.TestCase):
    """Verifies mathematical invariants, identity laws, and RDNA2 oracle behavior."""

    def test_truth_table_per_bit(self) -> None:
        """Verify the 1-bit boolean truth table for OR logic across all lanes."""
        truth_table = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 1,
        }
        for (exec_bit, src_bit), expected_exec in truth_table.items():
            for bit_pos in (0, 1, 15, 31, 32, 47, 63):
                exec_in = exec_bit << bit_pos
                src_val = src_bit << bit_pos
                saved, new_exec, scc = oracle_s_or_saveexec_b64(exec_in, src_val)
                self.assertEqual(saved, exec_in)
                expected_val = expected_exec << bit_pos
                self.assertEqual(new_exec, expected_val)
                self.assertEqual(scc, 1 if expected_val != 0 else 0)

    def test_identity_zero(self) -> None:
        """OR with 0 leaves EXEC unchanged and saves original EXEC."""
        test_patterns = [
            NO_LANES_ACTIVE_64,
            ALL_LANES_ACTIVE_64,
            EVEN_LANES_64,
            ODD_LANES_64,
            LOWER_HALF_ACTIVE_64,
            UPPER_HALF_ACTIVE_64,
            0x1234_5678_9ABC_DEF0,
        ]
        for pat in test_patterns:
            saved, new_exec, scc = oracle_s_or_saveexec_b64(pat, 0)
            self.assertEqual(saved, pat)
            self.assertEqual(new_exec, pat)
            self.assertEqual(scc, 1 if pat != 0 else 0)

    def test_dominance_all_ones(self) -> None:
        """OR with ALL_LANES_ACTIVE forces new EXEC to all ones and SCC=1."""
        test_patterns = [
            NO_LANES_ACTIVE_64,
            ALL_LANES_ACTIVE_64,
            EVEN_LANES_64,
            ODD_LANES_64,
            0x0000_0000_0000_0001,
        ]
        for pat in test_patterns:
            saved, new_exec, scc = oracle_s_or_saveexec_b64(pat, ALL_LANES_ACTIVE_64)
            self.assertEqual(saved, pat)
            self.assertEqual(new_exec, ALL_LANES_ACTIVE_64)
            self.assertEqual(scc, 1)

    def test_superset_invariant(self) -> None:
        """New EXEC must always be a superset of original EXEC bits."""
        rng = random.Random(0x5A1E_0864)
        for _ in range(100):
            exec_in = rng.getrandbits(64)
            src_val = rng.getrandbits(64)
            saved, new_exec, scc = oracle_s_or_saveexec_b64(exec_in, src_val)
            self.assertEqual(saved, exec_in)
            # Invariant: all bits initially active in EXEC remain active
            self.assertEqual(new_exec & exec_in, exec_in)
            # Invariant: all bits active in SSRC0 become active
            self.assertEqual(new_exec & src_val, src_val)
            # Invariant: active lane count never decreases
            self.assertGreaterEqual(popcount64(new_exec), popcount64(exec_in))

    def test_idempotence(self) -> None:
        """When src == exec_in, new EXEC equals exec_in."""
        for pattern in [0x0, 0x1, 0x8000_0000_0000_0000, 0xDEAD_BEEF_CAFE_BABE, MASK64]:
            saved, new_exec, scc = oracle_s_or_saveexec_b64(pattern, pattern)
            self.assertEqual(saved, pattern)
            self.assertEqual(new_exec, pattern)
            self.assertEqual(scc, 1 if pattern != 0 else 0)

    def test_word_splitting_lo_hi(self) -> None:
        """Ensure 64-bit operation matches independent 32-bit (lo, hi) word operations."""
        rng = random.Random(0x3232_6464)
        for _ in range(50):
            exec_in = rng.getrandbits(64)
            src_val = rng.getrandbits(64)
            saved, new_exec, scc = oracle_s_or_saveexec_b64(exec_in, src_val)
            exec_lo, exec_hi = split_u64_to_u32s(exec_in)
            src_lo, src_hi = split_u64_to_u32s(src_val)
            expected_new_lo = exec_lo | src_lo
            expected_new_hi = exec_hi | src_hi
            new_lo, new_hi = split_u64_to_u32s(new_exec)
            self.assertEqual(new_lo, expected_new_lo)
            self.assertEqual(new_hi, expected_new_hi)
            self.assertEqual(combine_u32s_to_u64(new_lo, new_hi), new_exec)

    def test_scc_flag_truth_table(self) -> None:
        """SCC flag is 1 if and only if new EXEC has at least one active bit."""
        # 1. Both zero -> SCC == 0
        saved, new_exec, scc = oracle_s_or_saveexec_b64(0, 0)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)

        # 2. Exec non-zero, src zero -> SCC == 1
        saved, new_exec, scc = oracle_s_or_saveexec_b64(1, 0)
        self.assertEqual(scc, 1)

        # 3. Exec zero, src non-zero -> SCC == 1
        saved, new_exec, scc = oracle_s_or_saveexec_b64(0, 1 << 63)
        self.assertEqual(scc, 1)

        # 4. Both non-zero -> SCC == 1
        saved, new_exec, scc = oracle_s_or_saveexec_b64(EVEN_LANES_64, ODD_LANES_64)
        self.assertEqual(new_exec, MASK64)
        self.assertEqual(scc, 1)

    def test_walking_ones_oracle(self) -> None:
        """Test single-lane activations across all 64 lane positions."""
        for lane in range(64):
            single_bit = 1 << lane
            saved, new_exec, scc = oracle_s_or_saveexec_b64(0, single_bit)
            self.assertEqual(saved, 0)
            self.assertEqual(new_exec, single_bit)
            self.assertEqual(scc, 1)
            self.assertEqual(popcount64(new_exec), 1)


if __name__ == "__main__":
    unittest.main()
