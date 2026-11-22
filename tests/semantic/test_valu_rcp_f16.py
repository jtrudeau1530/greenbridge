from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_rcp_f16.

Instruction:
  v_rcp_f16 dst, src0 (VOP1 / VOP3)
  D.f16 = 1.0 / S0.f16

Architecture:
  - RDNA2 Vector ALU (GFX10) half-precision floating-point reciprocal.
  - S0: 16-bit IEEE 754 binary16 float (halfword from VGPR, SGPR, or literal/inline const).
  - D:  16-bit IEEE 754 binary16 float (halfword in VGPR).
  - VOP1 preserves the untouched 16-bit half of the destination 32-bit VGPR.
  - VOP3 op_sel selects which half (lo [15:0] or hi [31:16]) is read or written.
  - Modifiers:
      * Input modifiers: abs (|S0|), neg (-S0).
      * Output modifiers: clamp (clamps result to [0.0, 1.0]).
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

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

# IEEE 754 binary16 architectural bit layout
SIGN_MASK_F16: int = 0x8000
EXP_MASK_F16: int = 0x7C00
MANT_MASK_F16: int = 0x03FF
QNAN_BIT_F16: int = 0x0200
BIAS_F16: int = 15

# Special half-precision bit patterns
POS_ZERO_F16: int = 0x0000
NEG_ZERO_F16: int = 0x8000
POS_INF_F16: int = 0x7C00
NEG_INF_F16: int = 0xFC00
QNAN_F16: int = 0x7E00
SNAN_F16: int = 0x7C01

# Standard reference floating-point constants in binary16
ONE_F16: int = 0x3C00            # 1.0
NEG_ONE_F16: int = 0xBC00        # -1.0
TWO_F16: int = 0x4000            # 2.0
NEG_TWO_F16: int = 0xC000        # -2.0
HALF_F16: int = 0x3800           # 0.5
NEG_HALF_F16: int = 0xB800       # -0.5
QUARTER_F16: int = 0x3400        # 0.25
NEG_QUARTER_F16: int = 0xB400    # -0.25
FOUR_F16: int = 0x4400           # 4.0
NEG_FOUR_F16: int = 0xC400       # -4.0
EIGHT_F16: int = 0x4800          # 8.0
THREE_F16: int = 0x4200          # 3.0
ONE_THIRD_F16: int = 0x3555      # 1/3 ~ 0.333251953125
ONE_AND_HALF_F16: int = 0x3E00   # 1.5
TWO_THIRDS_F16: int = 0x3955     # 2/3 ~ 0.66650390625

MAX_FINITE_F16: int = 0x7BFF     # 65504.0
MIN_NORMAL_F16: int = 0x0400     # 2^-14 (~6.103515625e-5)
MAX_SUBNORMAL_F16: int = 0x03FF  # ~6.09755516e-5
MIN_SUBNORMAL_F16: int = 0x0001  # 2^-24 (~5.9604644775e-8)


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s16(val: int) -> int:
    """Interpret a 16-bit unsigned halfword as signed two's complement integer."""
    val &= MASK16
    return val - (1 << 16) if (val & 0x8000) else val


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as signed two's complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def split_halves(val32: int) -> tuple[int, int]:
    """Split 32-bit dword into (low_half, high_half)."""
    val = val32 & MASK32
    return val & MASK16, (val >> 16) & MASK16


def join_halves(lo: int, hi: int) -> int:
    """Join (low_half, high_half) into a 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def bits_to_f16(bits: int) -> float:
    """Convert unsigned 16-bit bits into Python float via struct IEEE 754 half."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def f16_to_bits(val: float) -> int:
    """Convert Python float into unsigned 16-bit IEEE 754 half bits."""
    if math.isnan(val):
        return QNAN_F16
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return NEG_INF_F16 if val < 0.0 else POS_INF_F16


def is_nan_f16(bits: int) -> bool:
    """Check if binary16 bits represent any NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) != 0)


def is_qnan_f16(bits: int) -> bool:
    """Check if binary16 bits represent a Quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_snan_f16(bits: int) -> bool:
    """Check if binary16 bits represent a Signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """Check if binary16 bits represent +/- infinity."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) == 0)


def is_zero_f16(bits: int) -> bool:
    """Check if binary16 bits represent +/- 0.0."""
    return (bits & ~SIGN_MASK_F16) == 0


def is_subnormal_f16(bits: int) -> bool:
    """Check if binary16 bits represent a subnormal (denormalized) half float."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & MANT_MASK_F16) != 0)


def is_normal_f16(bits: int) -> bool:
    """Check if binary16 bits represent a normalized finite half float."""
    exp = (bits & EXP_MASK_F16) >> 10
    return 1 <= exp <= 30


def apply_f16_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP input modifiers: abs clears sign bit, neg toggles sign bit."""
    res = bits & MASK16
    if abs_mod:
        res &= ~SIGN_MASK_F16
    if neg_mod:
        res ^= SIGN_MASK_F16
    return res


def _rne_int(q: Fraction) -> int:
    """Round rational number to nearest integer using Round-to-Nearest-Even (RNE)."""
    n = q.numerator // q.denominator
    rem = q - n
    half = Fraction(1, 2)
    if rem > half or (rem == half and (n & 1)):
        n += 1
    return n


def f16_bits_to_fraction(bits: int) -> Fraction:
    """Convert binary16 bits to exact rational Fraction."""
    bits &= MASK16
    sign = -1 if (bits & SIGN_MASK_F16) else 1
    exp_field = (bits & EXP_MASK_F16) >> 10
    mant_field = bits & MANT_MASK_F16
    if exp_field == 0x1F:
        raise ValueError(f"Non-finite f16 bits {bits:#06x}")
    if exp_field == 0:
        return sign * Fraction(mant_field, 1 << 10) * (Fraction(2) ** -14)
    return sign * Fraction((1 << 10) | mant_field, 1 << 10) * (Fraction(2) ** (exp_field - 15))


def round_fraction_to_f16(val: Fraction, neg_zero: bool = False) -> int:
    """Round exact rational Fraction to IEEE 754 binary16 bits using RNE."""
    if val == 0:
        return NEG_ZERO_F16 if neg_zero else POS_ZERO_F16
    sign = SIGN_MASK_F16 if val < 0 else 0
    a = abs(val)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if e < -14:
        n = _rne_int(a / (Fraction(2) ** -24))
        if n == 0:
            return sign
        if n < (1 << 10):
            return sign | n
        return sign | (1 << 10)
    n = _rne_int(a / (Fraction(2) ** (e - 10)))
    if n == (1 << 11):
        n >>= 1
        e += 1
    if e > 15:
        return sign | POS_INF_F16
    return sign | ((e + 15) << 10) | (n & MANT_MASK_F16)


def clamp_f16(bits: int) -> int:
    """Apply RDNA2 clamp modifier [0.0, 1.0] to halfword result."""
    if is_nan_f16(bits):
        return POS_ZERO_F16
    if bits & SIGN_MASK_F16:
        return POS_ZERO_F16
    if is_inf_f16(bits):
        return ONE_F16
    if bits > ONE_F16:
        return ONE_F16
    return bits


def ref_rcp_f16(
    src0_bits: int,
    *,
    abs_src0: bool = False,
    neg_src0: bool = False,
    clamp: bool = False,
) -> int:
    """Bit-accurate reference mathematical oracle for RDNA2 v_rcp_f16.

    Computes: D = 1.0 / S0 in IEEE 754 binary16.
    """
    s0 = apply_f16_modifiers(src0_bits, abs_src0, neg_src0)

    # 1. NaN handling
    if is_nan_f16(s0):
        if clamp:
            return POS_ZERO_F16
        # Propagate quiet NaN with quiet bit set
        return s0 | QNAN_BIT_F16

    # 2. Zero handling: 1.0 / (+-0.0) = +-inf
    if is_zero_f16(s0):
        res = NEG_INF_F16 if (s0 & SIGN_MASK_F16) else POS_INF_F16
        return clamp_f16(res) if clamp else res

    # 3. Infinity handling: 1.0 / (+-inf) = +-0.0
    if is_inf_f16(s0):
        res = NEG_ZERO_F16 if (s0 & SIGN_MASK_F16) else POS_ZERO_F16
        return clamp_f16(res) if clamp else res

    # 4. Finite non-zero: calculate using exact rational reciprocal
    frac = f16_bits_to_fraction(s0)
    recip_frac = Fraction(1) / frac
    res = round_fraction_to_f16(recip_frac, neg_zero=bool(s0 & SIGN_MASK_F16))

    if clamp:
        res = clamp_f16(res)
    return res


def ref_rcp_f16_rational(
    src0_bits: int,
    *,
    abs_src0: bool = False,
    neg_src0: bool = False,
    clamp: bool = False,
) -> int:
    """Alternative rational mathematical oracle implementation."""
    return ref_rcp_f16(src0_bits, abs_src0=abs_src0, neg_src0=neg_src0, clamp=clamp)


class TestValuRcpF16ReferenceOracle(unittest.TestCase):
    def test_powers_of_two_reciprocals(self):
        cases = [
            (ONE_F16, ONE_F16),
            (TWO_F16, HALF_F16),
            (HALF_F16, TWO_F16),
            (FOUR_F16, QUARTER_F16),
            (QUARTER_F16, FOUR_F16),
            (NEG_ONE_F16, NEG_ONE_F16),
            (NEG_TWO_F16, NEG_HALF_F16),
            (NEG_HALF_F16, NEG_TWO_F16),
            (NEG_FOUR_F16, NEG_QUARTER_F16),
        ]
        for inp, expected in cases:
            got = ref_rcp_f16(inp)
            self.assertEqual(got, expected, f"Failed on input {inp:#06x}: expected {expected:#06x}, got {got:#06x}")

    def test_rational_fractions(self):
        got_third = ref_rcp_f16(THREE_F16)
        self.assertEqual(got_third, ONE_THIRD_F16)
        got_two_thirds = ref_rcp_f16(ONE_AND_HALF_F16)
        self.assertEqual(got_two_thirds, TWO_THIRDS_F16)

    def test_infinities_and_zeros(self):
        self.assertEqual(ref_rcp_f16(POS_ZERO_F16), POS_INF_F16)
        self.assertEqual(ref_rcp_f16(NEG_ZERO_F16), NEG_INF_F16)
        self.assertEqual(ref_rcp_f16(POS_INF_F16), POS_ZERO_F16)
        self.assertEqual(ref_rcp_f16(NEG_INF_F16), NEG_ZERO_F16)

    def test_clamping(self):
        self.assertEqual(ref_rcp_f16(HALF_F16, clamp=True), ONE_F16)
        self.assertEqual(ref_rcp_f16(TWO_F16, clamp=True), HALF_F16)
        self.assertEqual(ref_rcp_f16(NEG_TWO_F16, clamp=True), POS_ZERO_F16)
        self.assertEqual(ref_rcp_f16(POS_ZERO_F16, clamp=True), ONE_F16)
        self.assertEqual(ref_rcp_f16(NEG_ZERO_F16, clamp=True), POS_ZERO_F16)
        self.assertEqual(ref_rcp_f16(QNAN_F16, clamp=True), POS_ZERO_F16)

    def test_input_modifiers(self):
        # abs on negative input yields positive reciprocal
        self.assertEqual(ref_rcp_f16(NEG_TWO_F16, abs_src0=True), HALF_F16)
        # neg on positive input yields negative reciprocal
        self.assertEqual(ref_rcp_f16(TWO_F16, neg_src0=True), NEG_HALF_F16)
        # -|x| on negative yields negative
        self.assertEqual(ref_rcp_f16(NEG_TWO_F16, abs_src0=True, neg_src0=True), NEG_HALF_F16)

    def test_subnormal_inputs(self):
        # 1 / 2^-15 = 2^15 = 32768.0 = 0x7800
        sub_2m15 = 0x0200
        self.assertEqual(ref_rcp_f16(sub_2m15), 0x7800)
        # 1 / 2^-14 = 2^14 = 16384.0 = 0x7400
        self.assertEqual(ref_rcp_f16(MIN_NORMAL_F16), 0x7400)
        # Smallest subnormal 2^-24 overflows to infinity
        self.assertEqual(ref_rcp_f16(MIN_SUBNORMAL_F16), POS_INF_F16)


if __name__ == "__main__":
    unittest.main()
