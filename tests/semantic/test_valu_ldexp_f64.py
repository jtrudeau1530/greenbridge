from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_ldexp_f64.

Instruction:
  v_ldexp_f64 dst, src0, src1 (VOP3)
  D.f64 = S0.f64 * (2 ^ S1.i32)

Architecture:
  - RDNA2 Vector ALU (GFX10) double-precision load exponent.
  - S0: 64-bit IEEE 754 binary64 floating-point operand (VGPR pair, SGPR pair, or literal).
  - S1: 32-bit signed two's complement integer exponent (VGPR, SGPR, or literal).
  - D:  64-bit IEEE 754 binary64 floating-point destination (VGPR pair).
  - Modifiers:
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


def ref_ldexp_f64(
    src0_bits: int,
    src1_i32: int,
    *,
    clamp: bool = False,
    abs_src0: bool = False,
    neg_src0: bool = False,
    omod: str = "none",
) -> int:
    """Bit-accurate reference mathematical oracle for RDNA2 v_ldexp_f64.

    Computes: D = S0 * 2^S1
    Where S0 is IEEE 754 binary64 and S1 is signed 32-bit integer.
    """
    src0_bits &= MASK64
    scale = s32(src1_i32)

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
        # Quiet the NaN if signaling
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

    # 4. Extract mantissa and unbiased exponent
    if exp_field == 0:
        # Subnormal input
        F = mant_field
        E = -1022
        shift_left = 52 - (F.bit_length() - 1)
        F = F << shift_left
        E = E - shift_left
    else:
        # Normal input
        E = exp_field - BIAS_F64
        F = (1 << 52) | mant_field

    # Apply omod multiplier to exponent
    omod_exp_shift = 0
    if omod == "mul:2":
        omod_exp_shift = 1
    elif omod == "mul:4":
        omod_exp_shift = 2
    elif omod == "div:2":
        omod_exp_shift = -1

    E_scaled = E + scale + omod_exp_shift
    target_exp = E_scaled + BIAS_F64

    # 5. Overflow check
    if target_exp >= 2047:
        if clamp:
            return POS_ZERO_F64 if sign else ONE_F64
        return (sign << 63) | POS_INF_F64

    # 6. Normal float range
    if target_exp >= 1:
        res = (sign << 63) | (target_exp << 52) | (F & MANT_MASK_F64)
        if clamp:
            if sign:
                return POS_ZERO_F64
            if res > ONE_F64:
                return ONE_F64
        return res

    # 7. Subnormal range / Gradual underflow (round-to-nearest-even)
    shift_right = 1 - target_exp
    if shift_right > 55:
        # Complete underflow
        return POS_ZERO_F64 if (clamp and sign) else (sign << 63)

    q = F >> shift_right
    mask = (1 << shift_right) - 1
    half = 1 << (shift_right - 1)
    remainder = F & mask

    # Round to nearest even
    if remainder > half:
        q += 1
    elif remainder == half:
        if q & 1:
            q += 1

    if q >= (1 << 52):
        # Rounded up into normal range (min normal)
        res = (sign << 63) | (1 << 52)
    else:
        res = (sign << 63) | (q & MANT_MASK_F64)

    if clamp:
        if sign:
            return POS_ZERO_F64
        if res > ONE_F64:
            return ONE_F64
    return res


def ref_ldexp_f64_rational(src0_bits: int, src1_i32: int) -> int:
    """Exact rational mathematical oracle using fractions.Fraction."""
    src0_bits &= MASK64
    scale = s32(src1_i32)

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

    scaled_frac = frac * (Fraction(2) ** scale)
    scaled_float = float(scaled_frac)
    res_bits = f64_to_bits(scaled_float)
    if sign:
        res_bits |= SIGN_MASK_F64
    return res_bits


class TestValuLdexpF64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_oracle_normal_values(self):
        # 1.0 * 2^3 = 8.0
        res = ref_ldexp_f64(ONE_F64, 3)
        self.assertEqual(bits_to_f64(res), 8.0)
        # 1.5 * 2^2 = 6.0
        f_1p5 = f64_to_bits(1.5)
        self.assertEqual(bits_to_f64(ref_ldexp_f64(f_1p5, 2)), 6.0)
        # -4.0 * 2^-2 = -1.0
        f_neg4 = f64_to_bits(-4.0)
        self.assertEqual(bits_to_f64(ref_ldexp_f64(f_neg4, -2)), -1.0)

    def test_oracle_powers_of_two(self):
        for p in range(-50, 50):
            res = ref_ldexp_f64(ONE_F64, p)
            expected = math.ldexp(1.0, p)
            self.assertEqual(bits_to_f64(res), expected)

    def test_oracle_cross_check_math_ldexp(self):
        test_floats = [0.125, 0.3333333333333333, 1.0, 2.5, 42.0, 1000.125, -7.5, -0.0625]
        test_scales = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
        for f in test_floats:
            f_bits = f64_to_bits(f)
            for s in test_scales:
                oracle_res = bits_to_f64(ref_ldexp_f64(f_bits, s))
                expected = math.ldexp(f, s)
                self.assertEqual(oracle_res, expected)

    def test_oracle_cross_check_rational(self):
        for f in [1.5, 3.25, -0.75, 10.125]:
            f_bits = f64_to_bits(f)
            for s in [-4, -1, 0, 3, 7]:
                self.assertEqual(ref_ldexp_f64(f_bits, s), ref_ldexp_f64_rational(f_bits, s))

    def test_oracle_special_values(self):
        # +0.0 and -0.0 preserved
        self.assertEqual(ref_ldexp_f64(POS_ZERO_F64, 10), POS_ZERO_F64)
        self.assertEqual(ref_ldexp_f64(NEG_ZERO_F64, 10), NEG_ZERO_F64)
        # +inf and -inf preserved
        self.assertEqual(ref_ldexp_f64(POS_INF_F64, 5), POS_INF_F64)
        self.assertEqual(ref_ldexp_f64(NEG_INF_F64, 5), NEG_INF_F64)
        # NaNs preserved and quieted
        res_qnan = ref_ldexp_f64(QNAN_F64, 2)
        self.assertTrue(is_nan_f64(res_qnan))
        res_snan = ref_ldexp_f64(SNAN_F64, 2)
        self.assertTrue(is_nan_f64(res_snan))

    def test_oracle_subnormal_and_gradual_underflow(self):
        # Smallest normal shifted down into subnormal
        sub_bits = ref_ldexp_f64(MIN_NORMAL_F64, -1)
        self.assertTrue(is_subnormal_f64(sub_bits))
        self.assertEqual(sub_bits, 0x0008000000000000)
        # Subnormal shifted back to normal
        norm_bits = ref_ldexp_f64(sub_bits, 1)
        self.assertEqual(norm_bits, MIN_NORMAL_F64)
        # Complete underflow to zero
        zero_bits = ref_ldexp_f64(MIN_NORMAL_F64, -100)
        self.assertEqual(zero_bits, POS_ZERO_F64)

    def test_oracle_overflow(self):
        # Overflow to infinity
        res_overflow = ref_ldexp_f64(MAX_FINITE_F64, 2)
        self.assertEqual(res_overflow, POS_INF_F64)
        res_neg_overflow = ref_ldexp_f64(MAX_FINITE_F64 | SIGN_MASK_F64, 2)
        self.assertEqual(res_neg_overflow, NEG_INF_F64)

    def test_oracle_modifiers(self):
        # abs modifier
        f_neg2 = f64_to_bits(-2.0)
        self.assertEqual(bits_to_f64(ref_ldexp_f64(f_neg2, 1, abs_src0=True)), 4.0)
        # neg modifier
        f_pos2 = f64_to_bits(2.0)
        self.assertEqual(bits_to_f64(ref_ldexp_f64(f_pos2, 1, neg_src0=True)), -4.0)
        # clamp modifier
        self.assertEqual(bits_to_f64(ref_ldexp_f64(f_pos2, 2, clamp=True)), 1.0)
        self.assertEqual(bits_to_f64(ref_ldexp_f64(f_neg2, 1, clamp=True)), 0.0)
        # omod modifiers
        self.assertEqual(bits_to_f64(ref_ldexp_f64(ONE_F64, 0, omod="mul:2")), 2.0)
        self.assertEqual(bits_to_f64(ref_ldexp_f64(ONE_F64, 0, omod="mul:4")), 4.0)
        self.assertEqual(bits_to_f64(ref_ldexp_f64(ONE_F64, 0, omod="div:2")), 0.5)


if __name__ == "__main__":
    unittest.main()
