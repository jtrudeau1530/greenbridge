from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_sad_u32,
v_sad_hi_u32, and v_mqsad_u32.

Architecture & Instruction Specifications (AMD RDNA2 / GFX10 Vector ALU):
- v_sad_u32 (VOP3 3-operand vector ALU instruction):
    Calculates the 32-bit unsigned sum of absolute differences with accumulation:
        diff = |src0.u32 - src1.u32|
        dst.u32 = (diff + src2.u32) mod 2^32
    Unsigned Semantics:
        The absolute difference is strictly unsigned:
        diff = (src0 >= src1) ? (src0 - src1) : (src1 - src0)
        Operands operate on unsigned 32-bit integers in [0, 2^32 - 1].
    Clamp modifier:
        When clamp is enabled on v_sad_u32, the result saturates to 0xFFFFFFFF on overflow:
        dst.u32 = min(0xFFFFFFFF, diff + src2.u32)

- v_sad_hi_u32:
    Computes the high 32-bit carry / overflow word of the full 64-bit SAD:
        diff = |src0.u32 - src1.u32|
        full_sum = diff + src2.u32
        dst.u32 = (full_sum >> 32) & 0xFFFFFFFF
    Since diff <= 2^32 - 1 and src2 <= 2^32 - 1, full_sum <= 2^33 - 2, so the high
    carry word is strictly in {0, 1}.
    Pairing (v_sad_u32, v_sad_hi_u32) synthesizes a full 64-bit precision sum of
    absolute differences with accumulation.

- v_mqsad_u32 / v_mqsad_u32_u8:
    Quad-byte masked sum of absolute differences with four full 32-bit accumulators:
    S0 is 64 bits (2 VGPRs), S1 is 32 bits (1 VGPR), S2 is 128 bits (4 VGPRs), D is 128 bits (4 VGPRs).
    For each dword k in 0..3:
        W_k = S0[8*k + 31 : 8*k] (32-bit window sliding across S0 byte-by-byte)
        D[k] = (S2[k] + sum_{i=0..3} |W_k.byte[i] - S1.byte[i]| (skipping if S1.byte[i] == 0 when masked)) mod 2^32.

Standard library Python only: unittest, dataclasses, enum, math, struct.
Single module scope, zero game data.
"""

import math
import struct
import unittest
from typing import Sequence, Tuple

# Architectural 32-bit and 64-bit masks
MASK8 = 0xFF
MASK16 = 0xFFFF
MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF
MASK128 = (1 << 128) - 1

CLAMP_MAX_U32 = MASK32


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x8000_0000) else val


def extract_byte(word32: int, byte_idx: int) -> int:
    """Extract byte at byte_idx (0..3) from a 32-bit word."""
    return (word32 >> (8 * (byte_idx & 3))) & MASK8


def byte_window_u32(val64: int, byte_offset: int) -> int:
    """Extract 32-bit window W_k at byte offset k from 64-bit val: bits [8*k+31 : 8*k]."""
    return (val64 >> (8 * byte_offset)) & MASK32


def ref_abs_diff_u32(s0: int, s1: int) -> int:
    """Calculate the strictly unsigned 32-bit absolute difference |s0 - s1|."""
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def ref_sad_u32(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    """Bit-accurate reference mathematical oracle for v_sad_u32.

    diff = |s0.u32 - s1.u32|
    result = (diff + s2.u32) mod 2^32 (or clamped to 0xFFFFFFFF on overflow).
    """
    diff = ref_abs_diff_u32(s0, s1)
    acc = s2 & MASK32
    full_sum = diff + acc
    if clamp:
        return min(CLAMP_MAX_U32, full_sum)
    return full_sum & MASK32


def ref_sad_hi_u32(s0: int, s1: int, s2: int) -> int:
    """Bit-accurate reference mathematical oracle for v_sad_hi_u32 (high 32-bit carry word).

    diff = |s0.u32 - s1.u32|
    full_sum = diff + s2.u32
    carry = (full_sum >> 32) & 0xFFFFFFFF
    """
    diff = ref_abs_diff_u32(s0, s1)
    acc = s2 & MASK32
    full_sum = diff + acc
    return (full_sum >> 32) & MASK32


def ref_quad_byte_sad(win: int, s1: int, *, masked: bool = True) -> int:
    """Compute sum of absolute differences across 4 bytes between win and s1.

    When masked is True, bytes where s1.byte[i] == 0 are skipped.
    """
    total = 0
    for i in range(4):
        b_win = extract_byte(win, i)
        b_s1 = extract_byte(s1, i)
        if masked and b_s1 == 0:
            continue
        total += abs(b_win - b_s1)
    return total


def ref_mqsad_u32(
    s0_64: int,
    s1_32: int,
    s2_128: Tuple[int, int, int, int],
    *,
    masked: bool = True,
) -> Tuple[int, int, int, int]:
    """Bit-accurate reference mathematical oracle for v_mqsad_u32.

    s0_64: 64-bit input source across two VGPRs
    s1_32: 32-bit template operand in one VGPR
    s2_128: 4x 32-bit accumulator dwords across four VGPRs
    Returns 4x 32-bit dwords destination tuple.
    """
    s0 = s0_64 & MASK64
    s1 = s1_32 & MASK32
    out = []
    for k in range(4):
        win = byte_window_u32(s0, k)
        sad = ref_quad_byte_sad(win, s1, masked=masked)
        acc = s2_128[k] & MASK32
        out.append((acc + sad) & MASK32)
    return (out[0], out[1], out[2], out[3])


class TestValuSadU32Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracles of v_sad_u32, v_sad_hi_u32, and v_mqsad_u32."""

    def test_abs_diff_u32_basic(self) -> None:
        self.assertEqual(ref_abs_diff_u32(10, 4), 6)
        self.assertEqual(ref_abs_diff_u32(4, 10), 6)
        self.assertEqual(ref_abs_diff_u32(0, 0), 0)
        self.assertEqual(ref_abs_diff_u32(MASK32, MASK32), 0)

    def test_abs_diff_u32_commutativity(self) -> None:
        pairs = [
            (0, MASK32),
            (1, 0x8000_0000),
            (0x1234_5678, 0x8765_4321),
            (0x7FFF_FFFF, 0x8000_0000),
            (0xDEAD_BEEF, 0xCAFE_BABE),
        ]
        for a, b in pairs:
            self.assertEqual(
                ref_abs_diff_u32(a, b),
                ref_abs_diff_u32(b, a),
                f"Commutativity failed for ({hex(a)}, {hex(b)})",
            )

    def test_abs_diff_u32_unsigned_vs_signed_distinction(self) -> None:
        # In signed arithmetic |0 - 0xFFFFFFFF| would be |-(-1)| = 1.
        # In unsigned arithmetic |0 - 0xFFFFFFFF| = 0xFFFFFFFF.
        self.assertEqual(ref_abs_diff_u32(0, MASK32), MASK32)
        self.assertEqual(ref_abs_diff_u32(MASK32, 0), MASK32)

    def test_triangle_inequality(self) -> None:
        triplets = [
            (100, 250, 400),
            (0, 0x5000_0000, 0xA000_0000),
            (0x1000_0000, 0x3000_0000, 0x7000_0000),
        ]
        for a, b, c in triplets:
            d_ac = ref_abs_diff_u32(a, c)
            d_ab = ref_abs_diff_u32(a, b)
            d_bc = ref_abs_diff_u32(b, c)
            self.assertLessEqual(d_ac, d_ab + d_bc)

    def test_sad_u32_basic_and_identity(self) -> None:
        # Identity: |a - a| + s2 == s2
        for val in [0, 1, 0x1234_5678, MASK32]:
            self.assertEqual(ref_sad_u32(val, val, 0x9999), 0x9999)
            self.assertEqual(ref_sad_u32(val, val, 0), 0)

    def test_sad_u32_wraparound_and_clamp(self) -> None:
        # diff = 0xFFFFFFFF, acc = 1 -> wraps to 0 without clamp, clamps to 0xFFFFFFFF with clamp
        self.assertEqual(ref_sad_u32(0, MASK32, 1, clamp=False), 0)
        self.assertEqual(ref_sad_u32(0, MASK32, 1, clamp=True), MASK32)

        # diff = 0xFFFFFFFF, acc = 0xFFFFFFFF -> wraps to 0xFFFFFFFE, clamps to 0xFFFFFFFF
        self.assertEqual(ref_sad_u32(0, MASK32, MASK32, clamp=False), 0xFFFF_FFFE)
        self.assertEqual(ref_sad_u32(0, MASK32, MASK32, clamp=True), MASK32)

        # non-overflow sum respects clamp
        self.assertEqual(ref_sad_u32(10, 5, 20, clamp=True), 25)

    def test_sad_hi_u32_carry_bounds(self) -> None:
        # Carry is strictly in {0, 1}
        self.assertEqual(ref_sad_hi_u32(10, 5, 20), 0)
        self.assertEqual(ref_sad_hi_u32(0, MASK32, 0), 0)
        self.assertEqual(ref_sad_hi_u32(0, MASK32, 1), 1)
        self.assertEqual(ref_sad_hi_u32(0, MASK32, MASK32), 1)

    def test_multi_precision_64bit_synthesis(self) -> None:
        # Full 64-bit SAD = (ref_sad_hi_u32 << 32) | ref_sad_u32
        test_cases = [
            (0, MASK32, 1, (1 << 32)),
            (0, MASK32, MASK32, (2 * MASK32)),
            (0x8000_0000, 0, 0x8000_0001, (1 << 32) + 1),
            (100, 50, 25, 75),
        ]
        for s0, s1, s2, expected_64 in test_cases:
            lo = ref_sad_u32(s0, s1, s2, clamp=False)
            hi = ref_sad_hi_u32(s0, s1, s2)
            combined_64 = (hi << 32) | lo
            self.assertEqual(combined_64, expected_64)

    def test_quad_byte_sad_masked_vs_unmasked(self) -> None:
        win = 0x0403_0201  # bytes: 1, 2, 3, 4
        s1 = 0x0400_0200   # bytes: 0, 2, 0, 4 (bytes 0 and 2 are 0)

        # Masked: skips byte 0 and byte 2 in s1
        # byte 0 (1 vs 0): skipped
        # byte 1 (2 vs 2): |2 - 2| = 0
        # byte 2 (3 vs 0): skipped
        # byte 3 (4 vs 4): |4 - 4| = 0
        self.assertEqual(ref_quad_byte_sad(win, s1, masked=True), 0)

        # Unmasked: includes all 4 bytes
        # byte 0: |1 - 0| = 1
        # byte 1: |2 - 2| = 0
        # byte 2: |3 - 0| = 3
        # byte 3: |4 - 4| = 0 -> total = 4
        self.assertEqual(ref_quad_byte_sad(win, s1, masked=False), 4)

    def test_mqsad_u32_sliding_window(self) -> None:
        # S0 = 0x0807_0605_0403_0201 (bytes: b0=1, b1=2, b2=3, b3=4, b4=5, b5=6, b6=7, b7=8)
        # Windows W_k (32-bit):
        # W_0 = bytes [1, 2, 3, 4] = 0x0403_0201
        # W_1 = bytes [2, 3, 4, 5] = 0x0504_0302
        # W_2 = bytes [3, 4, 5, 6] = 0x0605_0403
        # W_3 = bytes [4, 5, 6, 7] = 0x0706_0504
        s0_64 = 0x0807_0605_0403_0201
        self.assertEqual(byte_window_u32(s0_64, 0), 0x0403_0201)
        self.assertEqual(byte_window_u32(s0_64, 1), 0x0504_0302)
        self.assertEqual(byte_window_u32(s0_64, 2), 0x0605_0403)
        self.assertEqual(byte_window_u32(s0_64, 3), 0x0706_0504)

        s1_32 = 0x0403_0201  # exactly matches W_0
        s2_128 = (10, 20, 30, 40)
        res = ref_mqsad_u32(s0_64, s1_32, s2_128, masked=False)
        # For W_0: diff is 0, res[0] = 10 + 0 = 10
        self.assertEqual(res[0], 10)
        # For W_1: bytes are [2, 3, 4, 5] vs [1, 2, 3, 4] -> diff is 1 + 1 + 1 + 1 = 4 -> res[1] = 20 + 4 = 24
        self.assertEqual(res[1], 24)
        # For W_2: bytes are [3, 4, 5, 6] vs [1, 2, 3, 4] -> diff is 2 + 2 + 2 + 2 = 8 -> res[2] = 30 + 8 = 38
        self.assertEqual(res[2], 38)
        # For W_3: bytes are [4, 5, 6, 7] vs [1, 2, 3, 4] -> diff is 3 + 3 + 3 + 3 = 12 -> res[3] = 40 + 12 = 52
        self.assertEqual(res[3], 52)


if __name__ == "__main__":
    unittest.main()
