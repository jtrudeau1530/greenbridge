"""Semantic interpreter tests and reference mathematical oracle for RDNA2 s_and_saveexec_b64.

Models the AMD RDNA2 / GFX10 / GCN Scalar Arithmetic Logic Unit (SALU) architectural
specification for:
    s_and_saveexec_b64 sdst, ssrc0

Architectural Semantics (SOP1 Encoding):
  1. Latched State Execution:
     sdst[63:0] = EXEC[63:0]
     EXEC[63:0] = ssrc0[63:0] & EXEC[63:0]
     SCC        = (EXEC[63:0] != 0) ? 1 : 0
  2. Operand Rules:
     - sdst: 64-bit destination SGPR pair (or EXEC/VCC special registers).
       Receives the previous (saved) value of the 64-bit EXEC mask.
     - ssrc0: 64-bit source operand (SGPR pair, literal constant, inline constant, or special register).
     - EXEC: 64-bit execution mask governing vector wavefront lane activation in Wave64 mode.
     - SCC: Scalar Condition Code bit in the STATUS register, updated to 1 if any lane in the
       new EXEC mask is non-zero, and 0 if all lanes are inactive.
  3. Register File Constraints:
     - 64-bit SGPR pairs must be 64-bit aligned (even SGPR index: s[0:1], s[2:3], etc.).
     - Destination latches previous EXEC before any potential register overwrite.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import os
import sys
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT64: int = 0x8000000000000000
ALL_LANES_64: int = 0xFFFFFFFFFFFFFFFF
NO_LANES_64: int = 0x0000000000000000
EVEN_LANES_64: int = 0x5555555555555555
ODD_LANES_64: int = 0xAAAAAAAAAAAAAAAA
LOW32_LANES_64: int = 0x00000000FFFFFFFF
HIGH32_LANES_64: int = 0xFFFFFFFF00000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def popcount64(val: int) -> int:
    """Return the number of active bits in a 64-bit integer."""
    return bin(val & MASK64).count("1")


def active_lanes_from_exec(exec_val: int) -> list[int]:
    """Extract a list of active lane indices [0..63] from a 64-bit EXEC mask."""
    val = exec_val & MASK64
    return [lane for lane in range(64) if (val & (1 << lane)) != 0]


def exec_from_active_lanes(lanes: list[int]) -> int:
    """Construct a 64-bit EXEC mask from a list of active lane indices [0..63]."""
    mask = 0
    for lane in lanes:
        if 0 <= lane < 64:
            mask |= (1 << lane)
        else:
            raise ValueError(f"Lane index {lane} out of 64-bit bounds [0..63]")
    return mask & MASK64


def ref_s_and_saveexec_b64(exec_mask: int, ssrc0: int) -> tuple[int, int, int]:
    """Pure mathematical reference oracle for RDNA2 s_and_saveexec_b64.

    Args:
        exec_mask: Current 64-bit execution mask.
        ssrc0: 64-bit source operand mask.

    Returns:
        tuple (sdst, new_exec, scc):
          - sdst: The saved prior EXEC mask (64-bit).
          - new_exec: The updated EXEC mask after bitwise AND with ssrc0 (64-bit).
          - scc: 1 if new_exec != 0, else 0.
    """
    exec_clean = exec_mask & MASK64
    src_clean = ssrc0 & MASK64
    sdst = exec_clean
    new_exec = exec_clean & src_clean
    scc = 1 if new_exec != 0 else 0
    return sdst, new_exec, scc


class TestSaluAndSaveexecB64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_oracle_identity_and_zero(self) -> None:
        """Test identity with all ones and clearing with zero mask."""
        # EXEC all lanes active, src all lanes active -> saved all, new all, SCC=1
        sdst, new_exec, scc = ref_s_and_saveexec_b64(ALL_LANES_64, ALL_LANES_64)
        self.assertEqual(sdst, ALL_LANES_64)
        self.assertEqual(new_exec, ALL_LANES_64)
        self.assertEqual(scc, 1)

        # EXEC all lanes, src zero -> saved all, new zero, SCC=0
        sdst, new_exec, scc = ref_s_and_saveexec_b64(ALL_LANES_64, NO_LANES_64)
        self.assertEqual(sdst, ALL_LANES_64)
        self.assertEqual(new_exec, NO_LANES_64)
        self.assertEqual(scc, 0)

        # EXEC zero, src all lanes -> saved zero, new zero, SCC=0
        sdst, new_exec, scc = ref_s_and_saveexec_b64(NO_LANES_64, ALL_LANES_64)
        self.assertEqual(sdst, NO_LANES_64)
        self.assertEqual(new_exec, NO_LANES_64)
        self.assertEqual(scc, 0)

    def test_oracle_scc_flag_transitions(self) -> None:
        """Verify SCC accurately reflects non-zero state of new_exec."""
        # Single bit mask yields SCC=1
        for lane in (0, 1, 31, 32, 62, 63):
            bit = 1 << lane
            sdst, new_exec, scc = ref_s_and_saveexec_b64(ALL_LANES_64, bit)
            self.assertEqual(sdst, ALL_LANES_64)
            self.assertEqual(new_exec, bit)
            self.assertEqual(scc, 1)

        # Disjoint masks yield new_exec=0 and SCC=0
        sdst, new_exec, scc = ref_s_and_saveexec_b64(EVEN_LANES_64, ODD_LANES_64)
        self.assertEqual(sdst, EVEN_LANES_64)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)

    def test_oracle_single_lane_masks(self) -> None:
        """Verify single active lane filtering across all 64 lane positions."""
        for lane in range(64):
            bit = 1 << lane
            sdst, new_exec, scc = ref_s_and_saveexec_b64(bit, bit)
            self.assertEqual(sdst, bit)
            self.assertEqual(new_exec, bit)
            self.assertEqual(scc, 1)

            # Different lane -> zero result
            other_bit = 1 << ((lane + 1) % 64)
            sdst2, new_exec2, scc2 = ref_s_and_saveexec_b64(bit, other_bit)
            self.assertEqual(sdst2, bit)
            self.assertEqual(new_exec2, 0)
            self.assertEqual(scc2, 0)

    def test_oracle_alternating_bitmasks(self) -> None:
        """Test alternating even/odd lane bitmasks."""
        sdst, new_exec, scc = ref_s_and_saveexec_b64(ALL_LANES_64, EVEN_LANES_64)
        self.assertEqual(sdst, ALL_LANES_64)
        self.assertEqual(new_exec, EVEN_LANES_64)
        self.assertEqual(scc, 1)
        self.assertEqual(popcount64(new_exec), 32)

        sdst, new_exec, scc = ref_s_and_saveexec_b64(EVEN_LANES_64, EVEN_LANES_64)
        self.assertEqual(sdst, EVEN_LANES_64)
        self.assertEqual(new_exec, EVEN_LANES_64)
        self.assertEqual(scc, 1)

    def test_oracle_split_join_dwords(self) -> None:
        """Verify 64-bit SGPR pair split/join helpers."""
        lo, hi = split_dwords(0x123456789ABCDEF0)
        self.assertEqual(lo, 0x9ABCDEF0)
        self.assertEqual(hi, 0x12345678)
        self.assertEqual(join_dwords(lo, hi), 0x123456789ABCDEF0)

    def test_oracle_active_lanes_helpers(self) -> None:
        """Test conversion between bitmask and active lane list."""
        lanes = [0, 7, 15, 31, 32, 47, 63]
        mask = exec_from_active_lanes(lanes)
        self.assertEqual(popcount64(mask), len(lanes))
        self.assertEqual(active_lanes_from_exec(mask), lanes)

        with self.assertRaises(ValueError):
            exec_from_active_lanes([64])

    def test_oracle_wave32_upper_lower_masks(self) -> None:
        """Verify behavior on lower 32-bit and upper 32-bit halves."""
        sdst, new_exec, scc = ref_s_and_saveexec_b64(LOW32_LANES_64, HIGH32_LANES_64)
        self.assertEqual(sdst, LOW32_LANES_64)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)

        sdst, new_exec, scc = ref_s_and_saveexec_b64(ALL_LANES_64, LOW32_LANES_64)
        self.assertEqual(sdst, ALL_LANES_64)
        self.assertEqual(new_exec, LOW32_LANES_64)
        self.assertEqual(scc, 1)

    def test_oracle_commutative_and_associative_properties(self) -> None:
        """Verify bitwise boolean algebraic properties."""
        a = 0xF0F0F0F00F0F0F0F
        b = 0xCCCCCCCC33333333
        _, res_ab, _ = ref_s_and_saveexec_b64(a, b)
        _, res_ba, _ = ref_s_and_saveexec_b64(b, a)
        self.assertEqual(res_ab, res_ba)


if __name__ == "__main__":
    unittest.main()
