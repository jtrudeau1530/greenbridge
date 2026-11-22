"""Semantic reference tests for RDNA2 64-bit vector conditional select (v_cndmask_b64).

RDNA2 Vector Arithmetic Logic Unit (VALU) architecture:
  In AMD RDNA2 (GFX10) instruction set architecture, vector registers (VGPRs) are 32-bit wide.
  Operations on 64-bit vector quantities (integers, double-precision floats, or 64-bit pointers)
  operate across aligned pairs of adjacent registers: v[d:d+1] from source pairs v[s0:s0+1] and
  v[s1:s1+1].

  v_cndmask_b64 conditionally selects 64-bit values per-lane based on a condition bit:
    For active lane i in the wavefront (0 <= i < WaveSize, WaveSize in {32, 64}):
      cond_bit = (condition_mask >> i) & 1
      dst[i]   = src1[i] if cond_bit != 0 else src0[i]

  Low-level dword decomposition:
    Because each 64-bit VGPR pair consists of low dword v[2k] and high dword v[2k+1]:
      dst_lo[i] = src1_lo[i] if cond_bit != 0 else src0_lo[i]
      dst_hi[i] = src1_hi[i] if cond_bit != 0 else src0_hi[i]
    Both halves are governed by the exact same per-lane condition bit without divergence between
    the low and high dwords.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

import math
import os
import random
import struct
import sys
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned word as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & (1 << 63)) else val


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def f64_to_u64(val: float) -> int:
    """Pack an IEEE-754 double precision float into an unsigned 64-bit integer."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def u64_to_f64(bits: int) -> float:
    """Unpack an unsigned 64-bit integer into an IEEE-754 double precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def ref_cndmask_b64(src0: int, src1: int, cond: int | bool) -> int:
    """Reference oracle for 64-bit conditional select (v_cndmask_b64) on a single lane.

    In RDNA2 ISA:
      D.u64 = cond ? S1.u64 : S0.u64
    Condition is true (non-zero) selects src1; false (zero) selects src0.
    """
    return (src1 if cond else src0) & MASK64


def ref_cndmask_b64_dwords(
    s0_lo: int, s0_hi: int, s1_lo: int, s1_hi: int, cond: int | bool
) -> tuple[int, int]:
    """Reference oracle for pair-wise 32-bit dword selection.

    Demonstrates that 64-bit vector conditional select operates identically to parallel
    32-bit conditional select on low and high dwords using the identical lane condition.
    """
    bit = 1 if cond else 0
    lo = (s1_lo if bit else s0_lo) & MASK32
    hi = (s1_hi if bit else s0_hi) & MASK32
    return lo, hi


def ref_cndmask_b64_decomposed(src0: int, src1: int, cond: int | bool) -> int:
    """Compute 64-bit conditional select by decomposing into 32-bit dwords and reassembling."""
    s0_lo, s0_hi = split_dwords(src0)
    s1_lo, s1_hi = split_dwords(src1)
    res_lo, res_hi = ref_cndmask_b64_dwords(s0_lo, s0_hi, s1_lo, s1_hi, cond)
    return join_dwords(res_lo, res_hi)


def ref_simd_cndmask_b64(
    src0_lanes: list[int],
    src1_lanes: list[int],
    mask: int,
    wave_size: int = 64,
    exec_mask: int | None = None,
    prev_dst: list[int] | None = None,
) -> list[int]:
    """Reference oracle for SIMD vector conditional select across a wavefront.

    Parameters:
      src0_lanes: 64-bit values for src0 across wave_size lanes.
      src1_lanes: 64-bit values for src1 across wave_size lanes.
      mask: condition bitmask (32-bit or 64-bit integer, bit i determines selection for lane i).
      wave_size: 32 (wave32) or 64 (wave64).
      exec_mask: optional execution mask; inactive lanes (exec_bit == 0) preserve prev_dst.
      prev_dst: previous destination values across lanes, preserved when lane is inactive.

    Returns:
      list of 64-bit unsigned integers representing the destination register pair across lanes.
    """
    if wave_size not in (32, 64):
        raise ValueError(f"Invalid wave_size: {wave_size}. Must be 32 or 64.")
    if len(src0_lanes) != wave_size or len(src1_lanes) != wave_size:
        raise ValueError(
            f"Expected {wave_size} lane inputs, got {len(src0_lanes)} and {len(src1_lanes)}"
        )
    if prev_dst is None:
        prev_dst = [0] * wave_size
    elif len(prev_dst) != wave_size:
        raise ValueError(f"Expected {wave_size} prev_dst entries, got {len(prev_dst)}")

    effective_exec = ((1 << wave_size) - 1) if exec_mask is None else (exec_mask & ((1 << wave_size) - 1))
    results: list[int] = []

    for i in range(wave_size):
        if not ((effective_exec >> i) & 1):
            results.append(prev_dst[i] & MASK64)
            continue
        cond_bit = (mask >> i) & 1
        val = ref_cndmask_b64(src0_lanes[i], src1_lanes[i], cond_bit)
        results.append(val)

    return results


class TestCndmaskB64ReferenceOracle(unittest.TestCase):
    """Unit tests for single-lane and SIMD 64-bit conditional select reference oracles."""

    def test_basic_truth_table(self) -> None:
        """Verify standard selector truth table: 0 selects src0, 1 selects src1."""
        val0 = 0x01234567_89ABCDEF
        val1 = 0xFEDCBA98_76543210
        self.assertEqual(ref_cndmask_b64(val0, val1, 0), val0)
        self.assertEqual(ref_cndmask_b64(val0, val1, False), val0)
        self.assertEqual(ref_cndmask_b64(val0, val1, 1), val1)
        self.assertEqual(ref_cndmask_b64(val0, val1, True), val1)
        self.assertEqual(ref_cndmask_b64(val0, val1, 42), val1)

    def test_64bit_distinct_dwords(self) -> None:
        """Test combinations where only the low, only the high, or both dwords differ."""
        cases = [
            # Only low dwords differ
            (0x11112222_33334444, 0x11112222_AAAABBBB),
            # Only high dwords differ
            (0xAAAA0000_12345678, 0xBBBB0000_12345678),
            # Both dwords differ completely
            (0x00000000_00000000, 0xFFFFFFFF_FFFFFFFF),
            (0x55555555_55555555, 0xAAAAAAAA_AAAAAAAA),
            (0x12345678_9ABCDEF0, 0x0FEDCBA9_87654321),
        ]
        for s0, s1 in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                self.assertEqual(ref_cndmask_b64(s0, s1, 0), s0)
                self.assertEqual(ref_cndmask_b64(s0, s1, 1), s1)
                self.assertEqual(
                    ref_cndmask_b64_decomposed(s0, s1, 0),
                    ref_cndmask_b64(s0, s1, 0),
                )
                self.assertEqual(
                    ref_cndmask_b64_decomposed(s0, s1, 1),
                    ref_cndmask_b64(s0, s1, 1),
                )

    def test_edge_bit_patterns(self) -> None:
        """Test boundary conditions, sign bits (bit 63, bit 31), and extreme integer limits."""
        patterns = [
            0,
            1,
            MASK64,
            1 << 63,
            (1 << 63) - 1,
            1 << 31,
            (1 << 31) - 1,
            1 << 32,
            MASK32,
            0x7FFFFFFF_FFFFFFFF,
            0x80000000_00000000,
            0xFFFFFFFF_00000000,
            0x00000000_FFFFFFFF,
        ]
        for i, p0 in enumerate(patterns):
            for j, p1 in enumerate(patterns):
                self.assertEqual(ref_cndmask_b64(p0, p1, 0), p0)
                self.assertEqual(ref_cndmask_b64(p0, p1, 1), p1)

    def test_float64_special_bitcasts(self) -> None:
        """Test 64-bit conditional select on IEEE-754 double precision bit patterns."""
        f_values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            math.pi,
            math.e,
            float("inf"),
            float("-inf"),
            float("nan"),
            1e-308,
            1e308,
            sys.float_info.min,
            sys.float_info.max,
        ]
        u64_patterns = [f64_to_u64(f) for f in f_values]

        for u0 in u64_patterns:
            for u1 in u64_patterns:
                self.assertEqual(ref_cndmask_b64(u0, u1, 0), u0)
                self.assertEqual(ref_cndmask_b64(u0, u1, 1), u1)

    def test_algebraic_identities(self) -> None:
        """Verify algebraic identities of 64-bit conditional selection."""
        rng = random.Random(1337)
        for _ in range(50):
            x = rng.getrandbits(64)
            y = rng.getrandbits(64)

            # Identity: selecting between identical values returns the value
            self.assertEqual(ref_cndmask_b64(x, x, 0), x)
            self.assertEqual(ref_cndmask_b64(x, x, 1), x)

            # Inversion: swapping operands and inverting condition bit yields identical result
            self.assertEqual(ref_cndmask_b64(x, y, 0), ref_cndmask_b64(y, x, 1))
            self.assertEqual(ref_cndmask_b64(x, y, 1), ref_cndmask_b64(y, x, 0))

            # Idempotence: re-selecting result with same condition returns itself
            r0 = ref_cndmask_b64(x, y, 0)
            self.assertEqual(ref_cndmask_b64(r0, y, 0), r0)
            r1 = ref_cndmask_b64(x, y, 1)
            self.assertEqual(ref_cndmask_b64(x, r1, 1), r1)

    def test_dword_decomposition_equivalence(self) -> None:
        """Verify bit-exact match between full 64-bit select and dword-wise decomposition."""
        rng = random.Random(42)
        for _ in range(200):
            s0 = rng.getrandbits(64)
            s1 = rng.getrandbits(64)
            for c in (0, 1):
                want = ref_cndmask_b64(s0, s1, c)
                got = ref_cndmask_b64_decomposed(s0, s1, c)
                self.assertEqual(want, got)
                s0_lo, s0_hi = split_dwords(s0)
                s1_lo, s1_hi = split_dwords(s1)
                lo, hi = ref_cndmask_b64_dwords(s0_lo, s0_hi, s1_lo, s1_hi, c)
                self.assertEqual(lo, want & MASK32)
                self.assertEqual(hi, (want >> 32) & MASK32)

    def test_simd_oracle_wave32(self) -> None:
        """Validate SIMD reference oracle across 32 lanes."""
        rng = random.Random(32)
        s0_lanes = [rng.getrandbits(64) for _ in range(32)]
        s1_lanes = [rng.getrandbits(64) for _ in range(32)]
        mask = 0b10101010_11001100_11110000_01010101

        res = ref_simd_cndmask_b64(s0_lanes, s1_lanes, mask, wave_size=32)
        self.assertEqual(len(res), 32)
        for lane in range(32):
            bit = (mask >> lane) & 1
            expected = s1_lanes[lane] if bit else s0_lanes[lane]
            self.assertEqual(res[lane], expected)

    def test_simd_oracle_wave64(self) -> None:
        """Validate SIMD reference oracle across 64 lanes."""
        rng = random.Random(64)
        s0_lanes = [rng.getrandbits(64) for _ in range(64)]
        s1_lanes = [rng.getrandbits(64) for _ in range(64)]
        mask = 0xCAFEBABEDEADBEEF

        res = ref_simd_cndmask_b64(s0_lanes, s1_lanes, mask, wave_size=64)
        self.assertEqual(len(res), 64)
        for lane in range(64):
            bit = (mask >> lane) & 1
            expected = s1_lanes[lane] if bit else s0_lanes[lane]
            self.assertEqual(res[lane], expected)


if __name__ == "__main__":
    unittest.main()
