"""Semantic tests and reference mathematical oracle for RDNA2 v_rsq_f16.

AMD RDNA2 VALU Instruction Set Architecture (GFX10 / GFX1030 / GFX1035):
  The v_rsq_f16 instruction computes the IEEE-754 binary16 (half-precision) reciprocal
  square root of a 16-bit floating-point operand:
      D.f16 = 1.0 / sqrt(S0.f16)

Mathematical & Architectural Semantics:
  - Exact powers of two and four:
      1.0     -> 1.0     (0x3C00 -> 0x3C00)
      4.0     -> 0.5     (0x4400 -> 0x3800)
      16.0    -> 0.25    (0x4C00 -> 0x3400)
      64.0    -> 0.125   (0x5400 -> 0x3000)
      0.25    -> 2.0     (0x3400 -> 0x4000)
      0.0625  -> 4.0     (0x2C00 -> 0x4400)
      0.015625-> 8.0     (0x2400 -> 0x4800)
  - Special floating-point values:
      +inf    -> +0.0    (IEEE-754 / GLSL 1/sqrt(+inf) = +0.0)
      +0.0    -> +inf    (Division by zero)
      -0.0    -> +inf    (RDNA2 VALU hardware maps -0.0 to +inf)
      qNaN    -> quiet NaN preserved
      sNaN    -> quieted NaN (quiet bit 9 set)
      negative finite (-x) -> Canonical Quiet NaN (square root of negative value)
      -inf    -> Canonical Quiet NaN
  - Subnormal numbers:
      Gradual underflow (FTZ=0): evaluated with full mathematical precision.
          1.0 / sqrt(2^-24) = 1.0 / 2^-12 = 2^12 = 4096.0 (0x6C00)
      Flush-to-zero (FTZ=1): subnormal input flushes to +0.0, yielding +inf.
  - VOP3 / VOP3P modifiers:
      abs: Clears sign bit of S0 prior to evaluation.
      neg: Toggles sign bit of S0 prior to evaluation.
      clamp: Clamps output value to [0.0, 1.0]; clamped NaN produces +0.0.
      omod: Output scaling (mul:2, mul:4, div:2) applied prior to clamp.

Standard library Python only (Python 3.9+ compatible, unittest, zero game data).
"""
from __future__ import annotations

import math
import struct
import unittest
from typing import Optional, Tuple

# IEEE-754 binary16 (half-precision) architectural constants
SIGN_MASK_16: int = 0x8000
EXP_MASK_16: int = 0x7C00
FRAC_MASK_16: int = 0x03FF
QNAN_BIT_16: int = 0x0200
CANONICAL_QNAN_16: int = 0x7E00

# Canonical binary16 encodings
H_POS_ZERO: int = 0x0000
H_NEG_ZERO: int = 0x8000
H_POS_INF: int = 0x7C00
H_NEG_INF: int = 0xFC00
H_QNAN: int = 0x7E00
H_SNAN: int = 0x7D00

H_ONE: int = 0x3C00          # 1.0
H_FOUR: int = 0x4400         # 4.0
H_SIXTEEN: int = 0x4C00      # 16.0
H_SIXTY_FOUR: int = 0x5400   # 64.0
H_QUARTER: int = 0x3400      # 0.25
H_HALF: int = 0x3800         # 0.5
H_TWO: int = 0x4000          # 2.0
H_EIGHT: int = 0x4800        # 8.0
H_0P0625: int = 0x2C00       # 0.0625 (1/16)
H_0P125: int = 0x3000        # 0.125 (1/8)
H_4096: int = 0x6C00         # 4096.0 (2^12)

H_MAX_NORMAL: int = 0x7BFF   # 65504.0
H_MIN_NORMAL: int = 0x0400   # 2^-14 (~6.10e-5)
H_MIN_SUBNORMAL: int = 0x0001  # 2^-24 (~5.96e-8)
H_MAX_SUBNORMAL: int = 0x03FF  # 1023 * 2^-24


def f16_to_bits(val: float) -> int:
    """Convert a Python float to 16-bit IEEE-754 half-precision integer bits."""
    if math.isnan(val):
        sign = 0x8000 if math.copysign(1.0, val) < 0.0 else 0x0000
        return sign | CANONICAL_QNAN_16
    if math.isinf(val):
        return H_POS_INF if val > 0.0 else H_NEG_INF
    if val >= 65520.0:
        return H_POS_INF
    if val <= -65520.0:
        return H_NEG_INF
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f16(bits: int) -> float:
    """Convert 16-bit integer bits to a Python float via binary16 format."""
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def split_f16_bits(bits: int) -> Tuple[int, int, int]:
    """Split 16-bit halfword into (sign, exponent, fraction) components."""
    bits &= 0xFFFF
    sign = (bits & SIGN_MASK_16) >> 15
    exponent = (bits & EXP_MASK_16) >> 10
    fraction = bits & FRAC_MASK_16
    return sign, exponent, fraction


def is_nan_f16(bits: int) -> bool:
    """Return True if 16-bit pattern represents an IEEE-754 NaN."""
    return ((bits & EXP_MASK_16) == EXP_MASK_16) and ((bits & FRAC_MASK_16) != 0)


def is_qnan_f16(bits: int) -> bool:
    """Return True if 16-bit pattern represents a Quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_16)


def is_snan_f16(bits: int) -> bool:
    """Return True if 16-bit pattern represents a Signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_16)


def is_inf_f16(bits: int) -> bool:
    """Return True if 16-bit pattern represents positive or negative infinity."""
    return ((bits & EXP_MASK_16) == EXP_MASK_16) and ((bits & FRAC_MASK_16) == 0)


def is_subnormal_f16(bits: int) -> bool:
    """Return True if 16-bit pattern represents a subnormal half-precision float."""
    return ((bits & EXP_MASK_16) == 0) and ((bits & FRAC_MASK_16) != 0)


def is_zero_f16(bits: int) -> bool:
    """Return True if 16-bit pattern represents positive or negative zero."""
    return (bits & ~SIGN_MASK_16) == 0


def apply_f16_input_modifiers(bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply VOP3 input modifiers to 16-bit halfword: abs clears sign bit; neg toggles."""
    res = bits & 0xFFFF
    if abs_mod:
        res &= ~SIGN_MASK_16
    if neg_mod:
        res ^= SIGN_MASK_16
    return res


def oracle_v_rsq_f16(
    src0_bits: int,
    abs_mod: bool = False,
    neg_mod: bool = False,
    clamp: bool = False,
    omod: str = "none",
    ftz: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_rsq_f16 (reciprocal square root).

    Computes D = 1.0 / sqrt(src0) in IEEE-754 binary16 arithmetic conforming
    to RDNA2 hardware semantics, including modifiers, FTZ mode, and saturation clamp.
    """
    s0 = apply_f16_input_modifiers(src0_bits, abs_mod=abs_mod, neg_mod=neg_mod)

    # 1. NaN handling
    if is_nan_f16(s0):
        if clamp:
            return H_POS_ZERO
        return (s0 | QNAN_BIT_16) & 0xFFFF

    # 2. Flush-to-zero (FTZ) for subnormals
    if ftz and is_subnormal_f16(s0):
        s0 = (s0 & SIGN_MASK_16)  # Flush fraction to zero

    # 3. Signed zero handling: 1.0 / sqrt(+/-0.0) -> +Infinity in RDNA2
    if is_zero_f16(s0):
        if clamp:
            return H_ONE
        return H_POS_INF

    # 4. Negative values (including -inf) produce Quiet NaN
    if bool(s0 & SIGN_MASK_16):
        if clamp:
            return H_POS_ZERO
        return CANONICAL_QNAN_16

    # 5. Positive infinity: 1.0 / sqrt(+inf) = +0.0
    if s0 == H_POS_INF:
        return H_POS_ZERO

    # 6. Finite positive operands
    val = bits_to_f16(s0)
    if val <= 0.0:
        res_float = float("inf")
    else:
        res_float = 1.0 / math.sqrt(val)

    # Apply output scaling modifiers
    if omod == "mul:2":
        res_float *= 2.0
    elif omod == "mul:4":
        res_float *= 4.0
    elif omod == "div:2":
        res_float *= 0.5

    # Apply clamp [0.0, 1.0]
    if clamp:
        if math.isnan(res_float) or res_float < 0.0:
            res_float = 0.0
        elif res_float > 1.0:
            res_float = 1.0

    return f16_to_bits(res_float)


class TestReferenceOracleRSQF16(unittest.TestCase):
    """Unit tests verifying the mathematical reference oracle for v_rsq_f16."""

    def test_exact_powers_of_four_and_two(self) -> None:
        table = [
            (H_ONE, H_ONE, "1.0 -> 1.0"),
            (H_FOUR, H_HALF, "4.0 -> 0.5"),
            (H_SIXTEEN, H_QUARTER, "16.0 -> 0.25"),
            (H_SIXTY_FOUR, H_0P125, "64.0 -> 0.125"),
            (H_QUARTER, H_TWO, "0.25 -> 2.0"),
            (H_0P0625, H_FOUR, "0.0625 -> 4.0"),
            (0x2400, H_EIGHT, "0.015625 -> 8.0"),
        ]
        for src, expected, desc in table:
            with self.subTest(msg=desc):
                self.assertEqual(oracle_v_rsq_f16(src), expected)

    def test_special_floating_point_boundaries(self) -> None:
        self.assertEqual(oracle_v_rsq_f16(H_POS_ZERO), H_POS_INF)
        self.assertEqual(oracle_v_rsq_f16(H_NEG_ZERO), H_POS_INF)
        self.assertEqual(oracle_v_rsq_f16(H_POS_INF), H_POS_ZERO)
        self.assertTrue(is_nan_f16(oracle_v_rsq_f16(H_NEG_INF)))

    def test_negative_values_produce_quiet_nan(self) -> None:
        for val in (-1.0, -4.0, -16.0, -0.25):
            bits = f16_to_bits(val)
            res = oracle_v_rsq_f16(bits)
            self.assertTrue(is_nan_f16(res))
            self.assertEqual(res, CANONICAL_QNAN_16)

    def test_nan_inputs_quietened(self) -> None:
        self.assertEqual(oracle_v_rsq_f16(H_QNAN), H_QNAN)
        res_snan = oracle_v_rsq_f16(H_SNAN)
        self.assertTrue(is_qnan_f16(res_snan))

    def test_input_modifiers(self) -> None:
        # abs(-4.0) -> +4.0 -> 0.5
        neg_four = f16_to_bits(-4.0)
        self.assertEqual(oracle_v_rsq_f16(neg_four, abs_mod=True), H_HALF)

        # neg(4.0) -> -4.0 -> NaN
        self.assertTrue(is_nan_f16(oracle_v_rsq_f16(H_FOUR, neg_mod=True)))

        # abs + neg of 4.0: abs first clears sign, then neg toggles -> -4.0 -> NaN
        self.assertTrue(is_nan_f16(oracle_v_rsq_f16(H_FOUR, abs_mod=True, neg_mod=True)))

    def test_clamp_and_omod_modifiers(self) -> None:
        # 0.25 -> 2.0; clamped -> 1.0
        self.assertEqual(oracle_v_rsq_f16(H_QUARTER, clamp=True), H_ONE)

        # 4.0 -> 0.5; omod mul:2 -> 1.0
        self.assertEqual(oracle_v_rsq_f16(H_FOUR, omod="mul:2"), H_ONE)
        # 4.0 -> 0.5; omod mul:4 -> 2.0
        self.assertEqual(oracle_v_rsq_f16(H_FOUR, omod="mul:4"), H_TWO)
        # 4.0 -> 0.5; omod div:2 -> 0.25
        self.assertEqual(oracle_v_rsq_f16(H_FOUR, omod="div:2"), H_QUARTER)

        # Clamped NaN -> +0.0
        self.assertEqual(oracle_v_rsq_f16(H_QNAN, clamp=True), H_POS_ZERO)

    def test_subnormal_handling(self) -> None:
        # 2^-24 subnormal min: 1 / sqrt(2^-24) = 4096.0
        self.assertEqual(oracle_v_rsq_f16(H_MIN_SUBNORMAL, ftz=False), H_4096)
        # Under FTZ, flushes to 0.0 -> +inf
        self.assertEqual(oracle_v_rsq_f16(H_MIN_SUBNORMAL, ftz=True), H_POS_INF)


if __name__ == "__main__":
    unittest.main()
