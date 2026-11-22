"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_frexp_exp_i32_f64.

Models IEEE 754-2008 / IEEE 754-2019 binary64 double-precision unbiased exponent extraction as
specified in the AMD RDNA2 / GFX10 Instruction Set Architecture:
  - v_frexp_exp_i32_f64: D.i = Exponent(S0.d)
    Extracts the unbiased integer exponent of S0.d such that S0.d = mantissa * 2^exponent with
    |mantissa| in [0.5, 1.0).
    The result is returned as a 32-bit signed integer (two's complement) written to a single 32-bit VGPR.

RDNA2 Double-Precision Exponent Extraction Rules:
  1. Register Operands:
     Source operand S0 is a 64-bit float occupying an aligned register pair v[s:s+1] or s[s:s+1],
     or a 64-bit immediate literal. Destination D is a single 32-bit vector register v[d].
  2. Normal Floats:
     In IEEE 754 double precision, exponent bias is 1023 and 0.5 has biased exponent 1022.
     For normal doubles (1 <= biased_exponent <= 2046):
       unbiased_exponent = biased_exponent - 1022
     Cross-checks identically with Python math.frexp(x)[1].
  3. Special Values:
     * +/-0.0: returns 0
     * +/-INF: returns 0
     * NaN (quiet or signaling): returns 0
  4. Subnormals (Denormals):
     In IEEE mode (default), subnormal numbers (biased_exponent == 0, fraction != 0) are normalized:
       unbiased_exponent = -1022 - leading_zeros(fraction)
     Under flush-to-zero (FTZ), subnormals return 0.
  5. VOP1 / VOP3 Encoding Forms & Modifiers:
     Available in VOP1 (e32) and VOP3 (e64) encodings.
     Input modifiers: abs (|src|) and neg (-src). Since the exponent depends only on magnitude,
     abs and neg preserve the computed exponent value.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_MASK_F64: int = 0x8000000000000000
EXP_MASK_F64: int = 0x7FF0000000000000
MANT_MASK_F64: int = 0x000FFFFFFFFFFFFF
EXP_SHIFT_F64: int = 52
EXP_BIAS_F64: int = 1023
EXP_FREXP_BIASED_F64: int = 1022  # 2^-1 has biased exponent 1022 (0x3FE) in double precision

# Canonical IEEE 754 binary64 bit patterns
POS_ZERO_F64: int = 0x0000000000000000
NEG_ZERO_F64: int = 0x8000000000000000
POS_ONE_F64: int = 0x3FF0000000000000       # 1.0 -> exp = 1
NEG_ONE_F64: int = 0xBFF0000000000000       # -1.0 -> exp = 1
HALF_F64: int = 0x3FE0000000000000          # 0.5 -> exp = 0
NEG_HALF_F64: int = 0xBFE0000000000000      # -0.5 -> exp = 0
POS_TWO_F64: int = 0x4000000000000000       # 2.0 -> exp = 2
POS_FOUR_F64: int = 0x4010000000000000      # 4.0 -> exp = 3
QUARTER_F64: int = 0x3FD0000000000000       # 0.25 -> exp = -1

DBL_MIN: int = 0x0010000000000000           # 2^-1022 -> exp = -1021
NEG_DBL_MIN: int = 0x8010000000000000       # -2^-1022 -> exp = -1021
DBL_MAX: int = 0x7FEFFFFFFFFFFFFF           # ~1.797e+308 -> exp = 1024
NEG_DBL_MAX: int = 0xFFEFFFFFFFFFFFFF
POS_INF_F64: int = 0x7FF0000000000000       # +infinity -> exp = 0
NEG_INF_F64: int = 0xFFF0000000000000       # -infinity -> exp = 0
QNAN_F64: int = 0x7FF8000000000000          # Quiet NaN -> exp = 0
NEG_QNAN_F64: int = 0xFFF8000000000000      # Negative quiet NaN -> exp = 0
SNAN_F64: int = 0x7FF0000000000001          # Signaling NaN -> exp = 0
SUBNORM_MIN_F64: int = 0x0000000000000001   # 2^-1074 -> exp = -1073
SUBNORM_MAX_F64: int = 0x000FFFFFFFFFFFFF   # (1 - 2^-52)*2^-1022 -> exp = -1022
NEG_SUBNORM_MIN_F64: int = 0x8000000000000001
NEG_SUBNORM_MAX_F64: int = 0x800FFFFFFFFFFFFF


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
    return val - (1 << 64) if (val & SIGN_MASK_F64) else val


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def bits_to_f64(bits: int) -> float:
    """Decode 64-bit unsigned integer to IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    """Encode double float to IEEE 754 double-precision 64-bit unsigned integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return POS_INF_F64 if val > 0 else NEG_INF_F64


def is_nan_f64(bits: int) -> bool:
    """True if 64-bit integer represents an IEEE 754 double NaN."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == EXP_MASK_F64 and (bits & MANT_MASK_F64) != 0


def is_subnormal_f64(bits: int) -> bool:
    """True if bits represent an IEEE 754 double subnormal number."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == 0 and (bits & MANT_MASK_F64) != 0


def apply_f64_modifiers(bits: int, neg: bool = False, abs_val: bool = False) -> int:
    """Applies VOP3 input modifiers: abs (|x|) first, then neg (-x)."""
    bits &= MASK64
    if abs_val:
        bits &= ~SIGN_MASK_F64
    if neg:
        bits ^= SIGN_MASK_F64
    return bits & MASK64


def ref_frexp_exp_i32_f64(s0_bits: int, ftz: bool = False) -> int:
    """Pure bitwise reference oracle for RDNA2 v_frexp_exp_i32_f64.

    Rules:
      - If NaN or +/-INF: returns 0.
      - If +/-0.0: returns 0.
      - If normal double: returns (biased_exp - 1022) as 32-bit two's complement.
      - If subnormal double: returns (-1022 - leading_zeros) under IEEE mode, or 0 under FTZ.
    """
    s0_bits &= MASK64
    exp_biased = (s0_bits & EXP_MASK_F64) >> EXP_SHIFT_F64
    frac = s0_bits & MANT_MASK_F64

    # Special case: +/-INF or NaN
    if exp_biased == 0x7FF:
        return 0

    # Special case: zero or subnormal
    if exp_biased == 0:
        if frac == 0 or ftz:
            return 0
        # In double precision, mantissa has 52 bits
        lz = 52 - frac.bit_length()
        unbiased = -1022 - lz
        return unbiased & MASK32

    # Normal double float
    unbiased = exp_biased - EXP_FREXP_BIASED_F64
    return unbiased & MASK32


def ref_frexp_exp_i32_ext(
    s0_bits: int,
    neg: bool = False,
    abs_val: bool = False,
    ftz: bool = False,
) -> int:
    """Reference oracle for v_frexp_exp_i32_f64 with input modifiers."""
    mod = apply_f64_modifiers(s0_bits, neg=neg, abs_val=abs_val)
    return ref_frexp_exp_i32_f64(mod, ftz=ftz)


class TestValuFrexpExpI32F64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_powers_of_two(self) -> None:
        self.assertEqual(s32(ref_frexp_exp_i32_f64(HALF_F64)), 0)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(POS_ONE_F64)), 1)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(POS_TWO_F64)), 2)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(POS_FOUR_F64)), 3)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(QUARTER_F64)), -1)

    def test_cross_check_math_frexp_normals(self) -> None:
        test_floats = [
            1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.141592653589793, -100.25,
            1e10, 1e-10, 1e300, 1e-300,
        ]
        for f in test_floats:
            with self.subTest(val=f):
                expected_exp = math.frexp(f)[1]
                bits = f64_to_bits(f)
                computed_exp = s32(ref_frexp_exp_i32_f64(bits))
                self.assertEqual(computed_exp, expected_exp)

    def test_zeros_return_zero(self) -> None:
        self.assertEqual(ref_frexp_exp_i32_f64(POS_ZERO_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(NEG_ZERO_F64), 0)

    def test_infinities_return_zero(self) -> None:
        self.assertEqual(ref_frexp_exp_i32_f64(POS_INF_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(NEG_INF_F64), 0)

    def test_nans_return_zero(self) -> None:
        self.assertEqual(ref_frexp_exp_i32_f64(QNAN_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(NEG_QNAN_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(SNAN_F64), 0)

    def test_subnormals_unbiased_exponent(self) -> None:
        # SUBNORM_MAX: leading bit is at 51, lz = 0 -> exp = -1022
        self.assertEqual(s32(ref_frexp_exp_i32_f64(SUBNORM_MAX_F64)), -1022)
        # SUBNORM_MIN: fraction is 1, lz = 51 -> exp = -1073
        self.assertEqual(s32(ref_frexp_exp_i32_f64(SUBNORM_MIN_F64)), -1073)

    def test_modifiers_abs_neg(self) -> None:
        # Sign does not alter exponent magnitude
        self.assertEqual(ref_frexp_exp_i32_ext(NEG_ONE_F64, abs_val=True), 1)
        self.assertEqual(ref_frexp_exp_i32_ext(POS_ONE_F64, neg=True), 1)
        self.assertEqual(ref_frexp_exp_i32_ext(POS_TWO_F64, neg=True, abs_val=True), 2)

    def test_ftz_mode(self) -> None:
        self.assertEqual(ref_frexp_exp_i32_f64(SUBNORM_MIN_F64, ftz=True), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(SUBNORM_MAX_F64, ftz=True), 0)
        # Normal floats unaffected by FTZ
        self.assertEqual(ref_frexp_exp_i32_f64(POS_ONE_F64, ftz=True), 1)


if __name__ == "__main__":
    unittest.main()
