from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_frexp_f64.

Instructions:
  1. v_frexp_mant_f64 dst, src0 (VOP1 e32 / VOP3 e64)
     D.f64 = Mantissa(S0.f64)
     Decomposes S0 such that S0 = mantissa * (2 ^ exponent) with |mantissa| in [0.5, 1.0).
     For normal inputs, the IEEE 754 11-bit biased exponent field [62:52] is replaced with 1022
     (0x3FE, representing 2^-1), preserving the original sign bit (bit 63) and 52-bit fraction (bits [51:0]).
     Special values:
       * +/-0.0: returns +/-0.0 (sign bit preserved).
       * +/-INF: returns +/-INF unchanged.
       * NaN: returns NaN unchanged (quiet / signaling payload bits preserved).
       * Subnormals: normalized into [0.5, 1.0) with exponent set to 1022.

  2. v_frexp_exp_i32_f64 dst, src0 (VOP1 e32 / VOP3 e64)
     D.i32 = Exponent(S0.f64)
     Extracts the unbiased integer base-2 exponent of S0, returned as a signed 32-bit integer.
     For normal inputs, computes (biased_exponent - 1022).
     Special values:
       * +/-0.0: returns 0.
       * +/-INF: returns 0.
       * NaN: returns 0.
       * Subnormals: returns -1022 - leading_zeros.

Architecture:
  - RDNA2 Vector ALU (GFX10) double-precision floating-point extraction.
  - S0: 64-bit IEEE 754 binary64 operand (VGPR pair, SGPR pair, or literal / inline constant).
  - D (mantissa): 64-bit IEEE 754 binary64 destination (VGPR pair).
  - D (exponent): 32-bit signed integer destination (single VGPR).
  - Modifiers:
      * Input modifiers on S0: abs (|S0|), neg (-S0).
      * Output modifiers on mantissa: clamp (clamps to [0.0, 1.0]), omod (mul:2, mul:4, div:2).
"""

from fractions import Fraction
import math
import os
import re
import struct
import sys
import unittest
from dataclasses import dataclass
from enum import Enum, auto

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

SIGN_MASK_F64 = 0x8000000000000000
EXP_MASK_F64 = 0x7FF0000000000000
MANT_MASK_F64 = 0x000FFFFFFFFFFFFF
BIAS_F64 = 1023
EXP_SHIFT_F64 = 52

EXP_FREXP_BIASED_F64 = 1022  # 2^-1 has biased exponent 1022 in IEEE 754 double precision
EXP_FREXP_BITS_F64 = EXP_FREXP_BIASED_F64 << EXP_SHIFT_F64  # 0x3FE0000000000000 (0.5)

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
    exp = (bits & EXP_MASK_F64) >> EXP_SHIFT_F64
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

    Extracts significand normalized into [0.5, 1.0) with original sign preserved.
    """
    src0_bits &= MASK64
    if abs_src0:
        src0_bits &= ~SIGN_MASK_F64
    if neg_src0:
        src0_bits ^= SIGN_MASK_F64

    sign = src0_bits & SIGN_MASK_F64
    exp_field = (src0_bits >> EXP_SHIFT_F64) & 0x7FF
    mant_field = src0_bits & MANT_MASK_F64

    # 1. NaN handling
    if exp_field == 0x7FF and mant_field != 0:
        if clamp:
            return POS_ZERO_F64
        # Quiet signaling NaN if needed, preserve payload
        return src0_bits | 0x0008000000000000

    # 2. Infinity handling
    if exp_field == 0x7FF and mant_field == 0:
        if clamp:
            return POS_ZERO_F64 if sign != 0 else ONE_F64
        return src0_bits

    # 3. Zero handling
    if exp_field == 0 and mant_field == 0:
        if clamp and sign != 0:
            return POS_ZERO_F64
        return src0_bits

    # 4. Subnormal handling: normalized into [0.5, 1.0)
    if exp_field == 0 and mant_field != 0:
        lz = 52 - mant_field.bit_length()
        norm_mant = (mant_field << (lz + 1)) & MANT_MASK_F64
        res_bits = sign | EXP_FREXP_BITS_F64 | norm_mant
    else:
        # 5. Normal float: replace exponent with 1022 (0x3FE)
        res_bits = sign | EXP_FREXP_BITS_F64 | mant_field

    # Output modifiers
    if omod != "none":
        val = bits_to_f64(res_bits)
        if omod == "mul:2":
            val *= 2.0
        elif omod == "mul:4":
            val *= 4.0
        elif omod == "div:2":
            val *= 0.5
        res_bits = f64_to_bits(val)

    if clamp:
        val = bits_to_f64(res_bits)
        if math.isnan(val) or val < 0.0:
            res_bits = POS_ZERO_F64
        elif val > 1.0:
            res_bits = ONE_F64

    return res_bits


def ref_frexp_exp_i32_f64(
    src0_bits: int,
    *,
    abs_src0: bool = False,
    neg_src0: bool = False,
) -> int:
    """Bit-accurate reference mathematical oracle for RDNA2 v_frexp_exp_i32_f64.

    Extracts unbiased exponent as signed 32-bit two's-complement integer.
    Special values (+/-0, +/-inf, NaN) return 0.
    """
    src0_bits &= MASK64
    if abs_src0:
        src0_bits &= ~SIGN_MASK_F64
    if neg_src0:
        src0_bits ^= SIGN_MASK_F64

    exp_field = (src0_bits >> EXP_SHIFT_F64) & 0x7FF
    mant_field = src0_bits & MANT_MASK_F64

    # Special values: NaN, Inf, Zero return 0
    if exp_field == 0x7FF or (exp_field == 0 and mant_field == 0):
        return 0

    # Subnormal
    if exp_field == 0:
        lz = 52 - mant_field.bit_length()
        unbiased = -1022 - lz
        return u32(unbiased)

    # Normal float: unbiased = exp_field - 1022
    unbiased = exp_field - 1022
    return u32(unbiased)


def ref_frexp_reconstruct_f64(src0_bits: int) -> float:
    """Reconstruct S0 = mantissa * 2^exponent from frexp decomposition."""
    mant_bits = ref_frexp_mant_f64(src0_bits)
    exp_i32 = s32(ref_frexp_exp_i32_f64(src0_bits))
    m = bits_to_f64(mant_bits)
    return math.ldexp(m, exp_i32)


class TestValuFrexpF64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_oracle_normal_values(self):
        # 1.0 -> mantissa = 0.5, exp = 1
        m1 = ref_frexp_mant_f64(ONE_F64)
        e1 = ref_frexp_exp_i32_f64(ONE_F64)
        self.assertEqual(bits_to_f64(m1), 0.5)
        self.assertEqual(s32(e1), 1)

        # 2.0 -> mantissa = 0.5, exp = 2
        m2 = ref_frexp_mant_f64(TWO_F64)
        e2 = ref_frexp_exp_i32_f64(TWO_F64)
        self.assertEqual(bits_to_f64(m2), 0.5)
        self.assertEqual(s32(e2), 2)

        # 0.5 -> mantissa = 0.5, exp = 0
        m05 = ref_frexp_mant_f64(HALF_F64)
        e05 = ref_frexp_exp_i32_f64(HALF_F64)
        self.assertEqual(bits_to_f64(m05), 0.5)
        self.assertEqual(s32(e05), 0)

        # 1.5 -> mantissa = 0.75, exp = 1
        f_1p5 = f64_to_bits(1.5)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_1p5)), 0.75)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(f_1p5)), 1)

        # -4.0 -> mantissa = -0.5, exp = 3
        f_neg4 = f64_to_bits(-4.0)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_neg4)), -0.5)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(f_neg4)), 3)

    def test_oracle_powers_of_two(self):
        for p in range(-50, 50):
            val = math.ldexp(1.0, p)
            val_bits = f64_to_bits(val)
            m_bits = ref_frexp_mant_f64(val_bits)
            e_bits = ref_frexp_exp_i32_f64(val_bits)
            self.assertEqual(bits_to_f64(m_bits), 0.5)
            self.assertEqual(s32(e_bits), p + 1)

    def test_oracle_cross_check_math_frexp(self):
        test_floats = [0.125, 0.3333333333333333, 1.0, 2.5, 42.0, 1000.125, -7.5, -0.0625]
        for f in test_floats:
            f_bits = f64_to_bits(f)
            oracle_m = bits_to_f64(ref_frexp_mant_f64(f_bits))
            oracle_e = s32(ref_frexp_exp_i32_f64(f_bits))
            py_m, py_e = math.frexp(f)
            self.assertEqual(oracle_m, py_m)
            self.assertEqual(oracle_e, py_e)

    def test_oracle_exact_reconstruction(self):
        test_floats = [
            1.5, 3.25, -0.75, 10.125, -12345.6789, 1.0e-100, 1.0e100,
            bits_to_f64(MIN_NORMAL_F64),
            bits_to_f64(MAX_SUBNORMAL_F64),
            bits_to_f64(MIN_SUBNORMAL_F64),
        ]
        for f in test_floats:
            f_bits = f64_to_bits(f)
            reconstructed = ref_frexp_reconstruct_f64(f_bits)
            self.assertEqual(reconstructed, f)

    def test_oracle_special_values(self):
        # +0.0 and -0.0
        self.assertEqual(ref_frexp_mant_f64(POS_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_ZERO_F64), NEG_ZERO_F64)
        self.assertEqual(ref_frexp_exp_i32_f64(POS_ZERO_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(NEG_ZERO_F64), 0)

        # +inf and -inf
        self.assertEqual(ref_frexp_mant_f64(POS_INF_F64), POS_INF_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_INF_F64), NEG_INF_F64)
        self.assertEqual(ref_frexp_exp_i32_f64(POS_INF_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(NEG_INF_F64), 0)

        # NaNs
        res_qnan_m = ref_frexp_mant_f64(QNAN_F64)
        self.assertTrue(is_nan_f64(res_qnan_m))
        self.assertEqual(ref_frexp_exp_i32_f64(QNAN_F64), 0)

        res_snan_m = ref_frexp_mant_f64(SNAN_F64)
        self.assertTrue(is_nan_f64(res_snan_m))
        self.assertEqual(ref_frexp_exp_i32_f64(SNAN_F64), 0)

    def test_oracle_subnormal_normalization(self):
        # Max subnormal
        max_sub_m = ref_frexp_mant_f64(MAX_SUBNORMAL_F64)
        max_sub_e = ref_frexp_exp_i32_f64(MAX_SUBNORMAL_F64)
        py_m, py_e = math.frexp(bits_to_f64(MAX_SUBNORMAL_F64))
        self.assertEqual(bits_to_f64(max_sub_m), py_m)
        self.assertEqual(s32(max_sub_e), py_e)
        self.assertEqual(s32(max_sub_e), -1022)

        # Min subnormal
        min_sub_m = ref_frexp_mant_f64(MIN_SUBNORMAL_F64)
        min_sub_e = ref_frexp_exp_i32_f64(MIN_SUBNORMAL_F64)
        py_min_m, py_min_e = math.frexp(bits_to_f64(MIN_SUBNORMAL_F64))
        self.assertEqual(bits_to_f64(min_sub_m), py_min_m)
        self.assertEqual(s32(min_sub_e), py_min_e)
        self.assertEqual(s32(min_sub_e), -1073)

    def test_oracle_modifiers(self):
        f_neg2 = f64_to_bits(-2.0)
        # abs modifier on mantissa
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_neg2, abs_src0=True)), 0.5)
        # neg modifier on positive mantissa
        f_pos2 = f64_to_bits(2.0)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_pos2, neg_src0=True)), -0.5)
        # abs and neg on exponent (no effect on magnitude/value)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(f_neg2, abs_src0=True)), 2)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(f_pos2, neg_src0=True)), 2)

        # clamp modifier on mantissa
        # positive mantissa in [0.5, 1.0) is unchanged by clamp
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_pos2, clamp=True)), 0.5)
        # negative mantissa is clamped to +0.0
        self.assertEqual(ref_frexp_mant_f64(f_neg2, clamp=True), POS_ZERO_F64)

        # omod modifiers
        # 0.5 * 2 = 1.0
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64, omod="mul:2")), 1.0)
        # 0.5 * 4 = 2.0
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64, omod="mul:4")), 2.0)
        # 0.5 / 2 = 0.25
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64, omod="div:2")), 0.25)


if __name__ == "__main__":
    unittest.main()
