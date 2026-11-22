from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_med3 instructions.

Instructions:
  v_med3_f32 dst, src0, src1, src2 (VOP3)
  v_med3_i32 dst, src0, src1, src2 (VOP3)
  v_med3_u32 dst, src0, src1, src2 (VOP3)

Architecture:
  - RDNA2 Vector ALU (GFX10) three-operand median.
  - S0, S1, S2: 32-bit operands (VGPR, SGPR, literal, or inline constant).
  - D: 32-bit destination VGPR.
  - Modifiers (v_med3_f32):
      * Input modifiers on float sources: abs (|S|), neg (-S).
      * Output modifiers: clamp (clamps result to [0.0, 1.0]), omod (mul:2, mul:4, div:2).
  - IEEE 754 float semantics:
      * Signed zeros: -0.0 < +0.0.
      * NaN suppression (minNum/maxNum style clamp):
        - 1 NaN operand: returns min of the two non-NaN operands.
        - 2 NaN operands: returns the single non-NaN operand.
        - 3 NaN operands: returns quiet NaN (0x7FC00000).
        - Signaling NaNs are quieted on return.
"""

from fractions import Fraction
import itertools
import math
import os
import re
import struct
import sys
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

SIGN_MASK_F32 = 0x80000000
EXP_MASK_F32 = 0x7F800000
MANT_MASK_F32 = 0x007FFFFF

POS_ZERO_F32 = 0x00000000
NEG_ZERO_F32 = 0x80000000
POS_INF_F32 = 0x7F800000
NEG_INF_F32 = 0xFF800000
QNAN_F32 = 0x7FC00000
SNAN_F32 = 0x7F800001

ONE_F32 = 0x3F800000        # 1.0
HALF_F32 = 0x3F000000       # 0.5
TWO_F32 = 0x40000000        # 2.0
MAX_FINITE_F32 = 0x7F7FFFFF
MIN_NORMAL_F32 = 0x00800000
MIN_SUBNORMAL_F32 = 0x00000001
MAX_SUBNORMAL_F32 = 0x007FFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def bits_to_f32(bits: int) -> float:
    """Unpack an unsigned 32-bit integer word into an IEEE 754 single precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Pack an IEEE 754 single precision float into an unsigned 32-bit integer word."""
    if math.isnan(val):
        return QNAN_F32
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return NEG_INF_F32 if val < 0.0 else POS_INF_F32


def is_nan_f32(bits: int) -> bool:
    """Check if binary32 bit pattern represents any NaN."""
    return ((bits & EXP_MASK_F32) == EXP_MASK_F32) and ((bits & MANT_MASK_F32) != 0)


def is_inf_f32(bits: int) -> bool:
    """Check if binary32 bit pattern represents +/- infinity."""
    return ((bits & EXP_MASK_F32) == EXP_MASK_F32) and ((bits & MANT_MASK_F32) == 0)


def is_zero_f32(bits: int) -> bool:
    """Check if binary32 bit pattern represents +/- 0.0."""
    return (bits & ~SIGN_MASK_F32) == 0


def is_subnormal_f32(bits: int) -> bool:
    """Check if binary32 bit pattern represents a subnormal float."""
    return ((bits & EXP_MASK_F32) == 0) and ((bits & MANT_MASK_F32) != 0)


def is_normal_f32(bits: int) -> bool:
    """Check if binary32 bit pattern represents a normalized finite float."""
    exp = (bits & EXP_MASK_F32) >> 23
    return 1 <= exp <= 254


def ref_med3_i32(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    """Bit-accurate reference mathematical oracle for RDNA2 v_med3_i32."""
    vals = [s32(s0), s32(s1), s32(s2)]
    vals.sort()
    res = vals[1] & MASK32
    if clamp:
        res = max(0, min(1, s32(res))) & MASK32
    return res


def ref_med3_u32(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    """Bit-accurate reference mathematical oracle for RDNA2 v_med3_u32."""
    vals = [u32(s0), u32(s1), u32(s2)]
    vals.sort()
    res = vals[1] & MASK32
    if clamp:
        res = max(0, min(1, res)) & MASK32
    return res


def ref_med3_f32(
    s0_bits: int,
    s1_bits: int,
    s2_bits: int,
    *,
    clamp: bool = False,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    abs2: bool = False,
    neg2: bool = False,
    omod: str = "none",
) -> int:
    """Bit-accurate reference mathematical oracle for RDNA2 v_med3_f32."""
    raw = [s0_bits & MASK32, s1_bits & MASK32, s2_bits & MASK32]
    mods = [(abs0, neg0), (abs1, neg1), (abs2, neg2)]
    for i in range(3):
        a, n = mods[i]
        if a:
            raw[i] &= ~SIGN_MASK_F32
        if n:
            raw[i] ^= SIGN_MASK_F32

    nans = [is_nan_f32(b) for b in raw]
    nan_count = sum(nans)

    if nan_count == 3:
        # All operands NaN -> quiet NaN
        res_bits = raw[0] | 0x00400000
    elif nan_count == 2:
        # Exactly two NaNs -> return the single non-NaN operand
        non_nan_idx = nans.index(False)
        res_bits = raw[non_nan_idx]
    elif nan_count == 1:
        # Exactly one NaN -> on RDNA2 hardware, clamp(NaN, min(x, y), max(x, y)) yields min(x, y)
        non_nan_indices = [i for i, is_n in enumerate(nans) if not is_n]
        b0, b1 = raw[non_nan_indices[0]], raw[non_nan_indices[1]]
        f0, f1 = bits_to_f32(b0), bits_to_f32(b1)
        # RDNA2 orders -0.0 < +0.0
        k0 = (f0, math.copysign(1.0, f0))
        k1 = (f1, math.copysign(1.0, f1))
        res_bits = b0 if k0 <= k1 else b1
    else:
        # All finite / infinity operands: median ordered with -0.0 < +0.0
        items = []
        for b in raw:
            f = bits_to_f32(b)
            items.append((f, math.copysign(1.0, f), b))
        items.sort(key=lambda t: (t[0], t[1]))
        res_bits = items[1][2]

    # Apply omod and clamp if requested
    if is_nan_f32(res_bits):
        if clamp:
            return POS_ZERO_F32
        return res_bits | 0x00400000

    f_res = bits_to_f32(res_bits)
    if omod == "mul:2":
        f_res *= 2.0
    elif omod == "mul:4":
        f_res *= 4.0
    elif omod == "div:2":
        f_res *= 0.5

    if clamp:
        if math.isnan(f_res):
            return POS_ZERO_F32
        f_res = max(0.0, min(1.0, f_res))
        if f_res == 0.0 and math.copysign(1.0, f_res) < 0.0:
            f_res = 0.0

    return f32_to_bits(f_res)


def ref_med3(opcode: str, s0: int, s1: int, s2: int, **kwargs) -> int:
    """Dispatch reference median calculation by opcode."""
    if opcode == "v_med3_i32":
        return ref_med3_i32(s0, s1, s2, **kwargs)
    elif opcode == "v_med3_u32":
        return ref_med3_u32(s0, s1, s2, **kwargs)
    elif opcode == "v_med3_f32":
        return ref_med3_f32(s0, s1, s2, **kwargs)
    raise ValueError(f"Unknown opcode: {opcode}")


class TestValuMed3Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_i32_basic(self):
        self.assertEqual(ref_med3_i32(5, -10, 7), 5)
        self.assertEqual(ref_med3_i32(-10, -20, -15), -15 & MASK32)
        self.assertEqual(ref_med3_i32(0, 0, 0), 0)
        self.assertEqual(ref_med3_i32(0x7FFFFFFF, 0x80000000, 0), 0)
        self.assertEqual(ref_med3_i32(0xFFFFFFFF, 0, 1), 0)  # -1, 0, 1 -> 0

    def test_oracle_u32_basic(self):
        self.assertEqual(ref_med3_u32(5, 10, 7), 7)
        self.assertEqual(ref_med3_u32(0xFFFFFFFF, 0, 1), 1)  # 0, 1, 4294967295 -> 1
        self.assertEqual(ref_med3_u32(0x7FFFFFFF, 0x80000000, 0), 0x7FFFFFFF)
        self.assertEqual(ref_med3_u32(100, 200, 150), 150)

    def test_oracle_signed_vs_unsigned_divergence(self):
        # 0xFFFFFFFF (-1 signed, 4294967295 unsigned), 0, 10
        self.assertEqual(ref_med3_i32(0xFFFFFFFF, 0, 10), 0)
        self.assertEqual(ref_med3_u32(0xFFFFFFFF, 0, 10), 10)

        # 0x80000000 (-2147483648 signed, 2147483648 unsigned), 5, 7
        self.assertEqual(ref_med3_i32(0x80000000, 5, 7), 5)
        self.assertEqual(ref_med3_u32(0x80000000, 5, 7), 7)

    def test_oracle_permutation_invariance(self):
        triples_i32 = [
            (5, 0x80000000, 7),
            (0xFFFFFFFF, 0, 1),
            (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
            (10, 20, 30),
            (-5, -1, -10),
        ]
        for t in triples_i32:
            expected = ref_med3_i32(*t)
            for p in itertools.permutations(t):
                self.assertEqual(ref_med3_i32(*p), expected)

        for t in triples_i32:
            expected_u = ref_med3_u32(*t)
            for p in itertools.permutations(t):
                self.assertEqual(ref_med3_u32(*p), expected_u)

    def test_oracle_f32_finite_numbers(self):
        b1 = f32_to_bits(1.0)
        b2 = f32_to_bits(2.0)
        b3 = f32_to_bits(3.0)
        self.assertEqual(ref_med3_f32(b1, b2, b3), b2)
        self.assertEqual(ref_med3_f32(b3, b1, b2), b2)

        bn1 = f32_to_bits(-1.0)
        bn2 = f32_to_bits(-2.0)
        bn3 = f32_to_bits(-3.0)
        self.assertEqual(ref_med3_f32(bn1, bn2, bn3), bn2)

    def test_oracle_f32_signed_zeros(self):
        # -0.0 < +0.0 on RDNA2 hardware
        b_neg_zero = NEG_ZERO_F32
        b_pos_zero = POS_ZERO_F32
        b_one = f32_to_bits(1.0)
        b_neg_one = f32_to_bits(-1.0)

        # med3(-0, +0, 1.0) = +0
        self.assertEqual(ref_med3_f32(b_neg_zero, b_pos_zero, b_one), b_pos_zero)
        self.assertEqual(ref_med3_f32(b_one, b_neg_zero, b_pos_zero), b_pos_zero)

        # med3(-0, +0, -1.0) = -0
        self.assertEqual(ref_med3_f32(b_neg_zero, b_pos_zero, b_neg_one), b_neg_zero)
        self.assertEqual(ref_med3_f32(b_neg_one, b_pos_zero, b_neg_zero), b_neg_zero)

    def test_oracle_f32_infinities(self):
        b_pos_inf = POS_INF_F32
        b_neg_inf = NEG_INF_F32
        b_zero = POS_ZERO_F32
        self.assertEqual(ref_med3_f32(b_neg_inf, b_zero, b_pos_inf), b_zero)
        self.assertEqual(ref_med3_f32(b_neg_inf, b_neg_inf, b_pos_inf), b_neg_inf)
        self.assertEqual(ref_med3_f32(b_neg_inf, b_pos_inf, b_pos_inf), b_pos_inf)

    def test_oracle_f32_nan_suppression(self):
        b_nan = QNAN_F32
        b1 = f32_to_bits(1.0)
        b2 = f32_to_bits(2.0)

        # 1 NaN -> min of remaining
        self.assertEqual(ref_med3_f32(b_nan, b1, b2), b1)
        self.assertEqual(ref_med3_f32(b2, b_nan, b1), b1)

        # 2 NaNs -> return non-NaN operand
        self.assertEqual(ref_med3_f32(b_nan, b_nan, b2), b2)
        self.assertEqual(ref_med3_f32(b1, b_nan, b_nan), b1)

        # 3 NaNs -> quiet NaN
        res3 = ref_med3_f32(b_nan, SNAN_F32, b_nan)
        self.assertTrue(is_nan_f32(res3))

    def test_oracle_f32_modifiers(self):
        b2 = f32_to_bits(2.0)
        b4 = f32_to_bits(4.0)
        b6 = f32_to_bits(6.0)

        # abs modifier on negative
        bn4 = f32_to_bits(-4.0)
        self.assertEqual(ref_med3_f32(b2, bn4, b6, abs1=True), b4)

        # neg modifier on positive
        self.assertEqual(ref_med3_f32(b2, b4, b6, neg1=True), b2)  # 2.0, -4.0, 6.0 -> 2.0

        # clamp modifier
        b0p5 = f32_to_bits(0.5)
        b1p5 = f32_to_bits(1.5)
        b2p5 = f32_to_bits(2.5)
        self.assertEqual(ref_med3_f32(b0p5, b1p5, b2p5, clamp=True), ONE_F32)  # median 1.5 clamped to 1.0

        # omod modifiers
        b1 = f32_to_bits(1.0)
        self.assertEqual(ref_med3_f32(b1, b1, b1, omod="mul:2"), f32_to_bits(2.0))
        self.assertEqual(ref_med3_f32(b1, b1, b1, omod="mul:4"), f32_to_bits(4.0))
        self.assertEqual(ref_med3_f32(b1, b1, b1, omod="div:2"), f32_to_bits(0.5))


if __name__ == "__main__":
    unittest.main()
