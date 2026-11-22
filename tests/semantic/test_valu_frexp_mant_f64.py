from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_frexp_mant_f64.

Instruction:
  v_frexp_mant_f64 dst, src0 (VOP1 / VOP3)
  D.f64 = mantissa(S0.f64)

Architecture:
  - RDNA2 Vector ALU (GFX10) double-precision mantissa extraction.
  - S0: 64-bit IEEE 754 binary64 floating-point operand (VGPR pair, SGPR pair, or literal).
  - D:  64-bit IEEE 754 binary64 floating-point destination (VGPR pair).
  - Result:
      * For finite non-zero S0: returns mantissa in [0.5, 1.0) with sign of S0.
      * For +0.0 / -0.0: returns +0.0 / -0.0.
      * For +inf / -inf: returns +inf / -inf.
      * For NaN: returns quiet NaN (payload preserved/quieted).
      * Subnormal inputs are normalized in IEEE preserve mode before extracting mantissa.
  - Modifiers (VOP3):
      * Input modifiers on S0: abs (|S0|), neg (-S0).
      * Output modifiers: clamp (clamps result to [0.0, 1.0]), omod (mul:2, mul:4, div:2).
"""

from fractions import Fraction
import math
import os
import struct
import sys
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

SIGN_MASK_F64 = 0x8000000000000000
EXP_MASK_F64 = 0x7FF0000000000000
MANT_MASK_F64 = 0x000FFFFFFFFFFFFF
BIAS_F64 = 1023
FREXP_EXP_F64 = 1022  # Biased exponent for range [0.5, 1.0)

POS_ZERO_F64 = 0x0000000000000000
NEG_ZERO_F64 = 0x8000000000000000
POS_INF_F64 = 0x7FF0000000000000
NEG_INF_F64 = 0xFFF0000000000000
QNAN_F64 = 0x7FF8000000000000
SNAN_F64 = 0x7FF0000000000001

ONE_F64 = 0x3FF0000000000000        # 1.0
TWO_F64 = 0x4000000000000000        # 2.0
HALF_F64 = 0x3FE0000000000000       # 0.5
MAX_FINITE_F64 = 0x7FEFFFFFFFFFFFFF # ~1.7976931348623157e+308
MIN_NORMAL_F64 = 0x0010000000000000 # 2^-1022 (~2.2250738585072014e-308)
MAX_SUBNORMAL_F64 = 0x000FFFFFFFFFFFFF
MIN_SUBNORMAL_F64 = 0x0000000000000001 # 2^-1074 (~4.9406564584124654e-324)


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & (1 << 63)) else val


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def bits_to_f64(bits: int) -> float:
    """Unpack an unsigned 64-bit integer into an IEEE 754 double precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    """Pack an IEEE 754 double precision float into an unsigned 64-bit integer word."""
    if math.isnan(val):
        return QNAN_F64
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return NEG_INF_F64 if val < 0.0 else POS_INF_F64


def is_nan_f64(bits: int) -> bool:
    """Check if binary64 bit pattern represents any NaN."""
    return ((bits & EXP_MASK_F64) == EXP_MASK_F64) and ((bits & MANT_MASK_F64) != 0)


def is_inf_f64(bits: int) -> bool:
    """Check if binary64 bit pattern represents +/- infinity."""
    return ((bits & EXP_MASK_F64) == EXP_MASK_F64) and ((bits & MANT_MASK_F64) == 0)


def is_zero_f64(bits: int) -> bool:
    """Check if binary64 bit pattern represents +/- 0.0."""
    return (bits & ~SIGN_MASK_F64) == 0


def is_subnormal_f64(bits: int) -> bool:
    """Check if binary64 bit pattern represents a subnormal float."""
    return ((bits & EXP_MASK_F64) == 0) and ((bits & MANT_MASK_F64) != 0)


def is_normal_f64(bits: int) -> bool:
    """Check if binary64 bit pattern represents a normalized finite float."""
    exp = (bits & EXP_MASK_F64) >> 52
    return 1 <= exp <= 2046


def ref_frexp_mant_f64(
    src0_bits: int,
    *,
    clamp: bool = False,
    abs_src0: bool = False,
    neg_src0: bool = False,
    omod: str = "none",
) -> int:
    """Bit-accurate reference mathematical oracle for RDNA2 v_frexp_mant_f64.

    Extracts the normalized mantissa in [0.5, 1.0) with the sign of S0.
    """
    src0_bits &= MASK64

    if abs_src0:
        src0_bits &= ~SIGN_MASK_F64
    if neg_src0:
        src0_bits ^= SIGN_MASK_F64

    sign = (src0_bits >> 63) & 1
    exp_field = (src0_bits >> 52) & 0x7FF
    mant_field = src0_bits & MANT_MASK_F64

    # 1. NaN handling
    if exp_field == 0x7FF and mant_field != 0:
        if clamp:
            return POS_ZERO_F64
        return src0_bits | 0x0008000000000000

    # 2. Infinity handling
    if exp_field == 0x7FF and mant_field == 0:
        if clamp:
            return POS_ZERO_F64 if sign else ONE_F64
        return src0_bits

    # 3. Zero handling (+/- 0.0)
    if exp_field == 0 and mant_field == 0:
        if clamp and sign:
            return POS_ZERO_F64
        return src0_bits

    # 4. Extract mantissa bits
    if exp_field == 0:
        # Subnormal input: normalize mantissa
        F = mant_field
        shift_left = 52 - (F.bit_length() - 1)
        normalized_mant = (F << shift_left) & MANT_MASK_F64
    else:
        # Normal input: directly use mantissa bits
        normalized_mant = mant_field

    # Base exponent for [0.5, 1.0) is 1022 (unbiased -1)
    target_exp = FREXP_EXP_F64

    # Apply omod multiplier
    if omod == "mul:2":
        target_exp += 1
    elif omod == "mul:4":
        target_exp += 2
    elif omod == "div:2":
        target_exp -= 1

    res = (sign << 63) | (target_exp << 52) | normalized_mant

    if clamp:
        if sign:
            return POS_ZERO_F64
        if res > ONE_F64:
            return ONE_F64

    return res


def ref_frexp_mant_f64_math(src0_bits: int) -> int:
    """Cross-check using Python's math.frexp."""
    src0_bits &= MASK64
    if is_nan_f64(src0_bits):
        return src0_bits | 0x0008000000000000
    if is_inf_f64(src0_bits) or is_zero_f64(src0_bits):
        return src0_bits

    f = bits_to_f64(src0_bits)
    m, _ = math.frexp(f)
    return f64_to_bits(m)


def ref_frexp_mant_f64_fraction(src0_bits: int) -> int:
    """Exact rational mathematical cross-check using fractions.Fraction."""
    src0_bits &= MASK64
    if is_nan_f64(src0_bits):
        return src0_bits | 0x0008000000000000
    if is_inf_f64(src0_bits) or is_zero_f64(src0_bits):
        return src0_bits

    sign = (src0_bits >> 63) & 1
    exp_field = (src0_bits >> 52) & 0x7FF
    mant_field = src0_bits & MANT_MASK_F64

    if exp_field == 0:
        frac = Fraction(mant_field, 1 << 52) * (Fraction(2) ** -1022)
    else:
        frac = (Fraction(1) + Fraction(mant_field, 1 << 52)) * (Fraction(2) ** (exp_field - 1023))

    frac_abs = abs(frac)
    # Find power of two such that frac_abs * 2^p is in [0.5, 1.0)
    p = 0
    while frac_abs >= 1:
        frac_abs /= 2
        p -= 1
    while frac_abs < Fraction(1, 2):
        frac_abs *= 2
        p += 1

    res_float = float(frac_abs) if sign == 0 else -float(frac_abs)
    return f64_to_bits(res_float)


class TestValuFrexpMantF64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_oracle_normal_values(self):
        # 1.0 = 0.5 * 2^1 -> mantissa is 0.5
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64)), 0.5)
        # 1.5 = 0.75 * 2^1 -> mantissa is 0.75
        f_1p5 = f64_to_bits(1.5)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_1p5)), 0.75)
        # -4.0 = -0.5 * 2^3 -> mantissa is -0.5
        f_neg4 = f64_to_bits(-4.0)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_neg4)), -0.5)

    def test_oracle_powers_of_two(self):
        for p in range(-50, 50):
            val = math.ldexp(1.0, p)
            val_bits = f64_to_bits(val)
            self.assertEqual(bits_to_f64(ref_frexp_mant_f64(val_bits)), 0.5)

    def test_oracle_cross_check_math_frexp(self):
        test_floats = [
            0.125, 0.3333333333333333, 1.0, 2.5, 42.0, 1000.125,
            -7.5, -0.0625, 1.7976931348623157e+308, 2.2250738585072014e-308
        ]
        for f in test_floats:
            f_bits = f64_to_bits(f)
            oracle_res = ref_frexp_mant_f64(f_bits)
            math_res = ref_frexp_mant_f64_math(f_bits)
            self.assertEqual(oracle_res, math_res)

    def test_oracle_cross_check_fraction(self):
        for f in [1.5, 3.25, -0.75, 10.125, 0.03125]:
            f_bits = f64_to_bits(f)
            self.assertEqual(ref_frexp_mant_f64(f_bits), ref_frexp_mant_f64_fraction(f_bits))

    def test_oracle_special_values(self):
        # +0.0 and -0.0 preserved
        self.assertEqual(ref_frexp_mant_f64(POS_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_ZERO_F64), NEG_ZERO_F64)
        # +inf and -inf preserved
        self.assertEqual(ref_frexp_mant_f64(POS_INF_F64), POS_INF_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_INF_F64), NEG_INF_F64)
        # NaNs preserved and quieted
        self.assertTrue(is_nan_f64(ref_frexp_mant_f64(QNAN_F64)))
        self.assertTrue(is_nan_f64(ref_frexp_mant_f64(SNAN_F64)))

    def test_oracle_subnormal_normalization(self):
        # Smallest subnormal 2^-1074: normalized mantissa must be 0.5
        res_sub_min = ref_frexp_mant_f64(MIN_SUBNORMAL_F64)
        self.assertEqual(bits_to_f64(res_sub_min), 0.5)

        # Negative subnormal: normalized mantissa is -0.5
        res_neg_sub = ref_frexp_mant_f64(MIN_SUBNORMAL_F64 | SIGN_MASK_F64)
        self.assertEqual(bits_to_f64(res_neg_sub), -0.5)

    def test_oracle_modifiers(self):
        # abs modifier
        f_neg2 = f64_to_bits(-2.0)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_neg2, abs_src0=True)), 0.5)
        # neg modifier
        f_pos2 = f64_to_bits(2.0)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_pos2, neg_src0=True)), -0.5)
        # clamp modifier
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_neg2, clamp=True)), 0.0)
        # omod modifiers
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64, omod="mul:2")), 1.0)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64, omod="mul:4")), 2.0)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64, omod="div:2")), 0.25)


if __name__ == "__main__":
    unittest.main()
