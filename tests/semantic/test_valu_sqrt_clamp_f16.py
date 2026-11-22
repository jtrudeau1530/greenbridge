from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import List, Optional, Tuple, Union

# IEEE 754 binary16 (half precision) architectural constants
SIGN_MASK: int = 0x8000
EXP_MASK: int = 0x7C00
FRAC_MASK: int = 0x03FF
QNAN_BIT: int = 0x0200
CANONICAL_QNAN: int = 0x7E00

POS_ZERO: int = 0x0000
NEG_ZERO: int = 0x8000
POS_INF: int = 0x7C00
NEG_INF: int = 0xFC00

ONE_F16: int = 0x3C00
FOUR_F16: int = 0x4400
NINE_F16: int = 0x4880
TWO_F16: int = 0x4000
THREE_F16: int = 0x4200
HALF_F16: int = 0x3800
QUARTER_F16: int = 0x3400


def f16_to_bits(val: float) -> int:
    """Convert a Python float to 16-bit binary16 integer bits."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f16(bits: int) -> float:
    """Convert 16-bit binary16 integer bits to a Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def split_f16_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK) >> 15
    exponent = (bits & EXP_MASK) >> 10
    fraction = bits & FRAC_MASK
    return sign, exponent, fraction


def dword_to_f16_halves(dword: int) -> Tuple[int, int]:
    """Split a 32-bit VGPR dword into (lo, hi) 16-bit halfwords."""
    return dword & 0xFFFF, (dword >> 16) & 0xFFFF


def f16_halves_to_dword(lo: int, hi: int) -> int:
    """Combine two 16-bit halfwords (lo, hi) into a 32-bit VGPR dword."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def is_nan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision NaN."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) != 0)


def is_qnan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision Quiet NaN."""
    return is_nan(bits) and bool(bits & QNAN_BIT)


def is_snan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision Signaling NaN."""
    return is_nan(bits) and not bool(bits & QNAN_BIT)


def is_inf(bits: int) -> bool:
    """Return True if bits represent positive or negative infinity."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) == 0)


def is_subnormal(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormal) half-precision float."""
    return ((bits & EXP_MASK) == 0) and ((bits & FRAC_MASK) != 0)


def is_zero(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK) == 0


def apply_f16_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & 0xFFFF
    if abs_mod:
        res &= ~SIGN_MASK
    if neg_mod:
        res ^= SIGN_MASK
    return res


def ref_sqrt_clamp_f16(
    src0_bits: int,
    abs0: bool = False,
    neg0: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_sqrt_clamp_f16.

    Computes D = clamp(sqrt(src0), 0.0, 1.0) conforming to IEEE 754-2008 binary16
    and AMD RDNA2 ISA VALU clamping rules:
      - Clamping NaN results to +0.0 (0x0000).
      - Clamping negative square roots (invalid operation) to +0.0 (0x0000).
      - Clamping negative zero (-0.0) to +0.0 (0x0000).
      - Clamping results > 1.0 (including +inf) to +1.0 (0x3C00).
      - Preserving in-range values in [0.0, 1.0].
    """
    s0 = apply_f16_modifiers(src0_bits, abs0, neg0)

    # 1. NaN inputs clamp to +0.0
    if is_nan(s0):
        return POS_ZERO

    # 2. Negative inputs (excluding -0.0) produce NaN in IEEE 754, clamped to +0.0
    if (s0 & SIGN_MASK) != 0 and s0 != NEG_ZERO:
        return POS_ZERO

    # 3. Signed zeros: sqrt(+0) = +0, sqrt(-0) = -0; clamping forces -0.0 to +0.0
    if is_zero(s0):
        return POS_ZERO

    # 4. Positive infinity: sqrt(+inf) = +inf; clamped to 1.0
    if is_inf(s0):
        return ONE_F16

    # 5. Finite positive square root with [0.0, 1.0] saturation
    f0 = bits_to_f16(s0)
    res_f = math.sqrt(f0)
    if res_f > 1.0:
        return ONE_F16
    if res_f < 0.0:
        return POS_ZERO
    return f16_to_bits(res_f)


# ==============================================================================
# Unit Tests
# ==============================================================================

class TestSqrtClampF16ModelAndOracle(unittest.TestCase):
    """Unit tests for binary16 bit model, predicates, and clamped reference oracle."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 4.0, 9.0]
        for val in values:
            with self.subTest(val=val):
                bits = f16_to_bits(val)
                recovered = bits_to_f16(bits)
                self.assertEqual(val, recovered)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_classification_predicates(self) -> None:
        self.assertTrue(f16_to_bits(0.0) == POS_ZERO)
        self.assertTrue(f16_to_bits(-0.0) == NEG_ZERO)
        self.assertTrue(is_zero(POS_ZERO))
        self.assertTrue(is_zero(NEG_ZERO))
        self.assertFalse(is_zero(f16_to_bits(1.0)))

        self.assertTrue(is_inf(POS_INF))
        self.assertTrue(is_inf(NEG_INF))
        self.assertFalse(is_inf(POS_ZERO))

        snan_bits = 0x7C01
        qnan_bits = CANONICAL_QNAN
        self.assertTrue(is_nan(snan_bits))
        self.assertTrue(is_snan(snan_bits))
        self.assertFalse(is_qnan(snan_bits))

        self.assertTrue(is_nan(qnan_bits))
        self.assertTrue(is_qnan(qnan_bits))
        self.assertFalse(is_snan(qnan_bits))

        subnorm_bits = 0x0001
        self.assertTrue(is_subnormal(subnorm_bits))
        self.assertFalse(is_subnormal(POS_ZERO))

    def test_halfword_packing_vgpr(self) -> None:
        lo = 0x3C00  # 1.0
        hi = 0x4000  # 2.0
        dword = f16_halves_to_dword(lo, hi)
        self.assertEqual(dword, 0x40003C00)
        unpacked_lo, unpacked_hi = dword_to_f16_halves(dword)
        self.assertEqual(unpacked_lo, lo)
        self.assertEqual(unpacked_hi, hi)

    def test_exact_clamped_sqrt(self) -> None:
        # Values > 1.0 saturate to 1.0
        self.assertEqual(ref_sqrt_clamp_f16(FOUR_F16), ONE_F16)
        self.assertEqual(ref_sqrt_clamp_f16(NINE_F16), ONE_F16)
        self.assertEqual(ref_sqrt_clamp_f16(TWO_F16), ONE_F16)

        # Values <= 1.0 remain unmodified
        self.assertEqual(ref_sqrt_clamp_f16(ONE_F16), ONE_F16)
        self.assertEqual(ref_sqrt_clamp_f16(QUARTER_F16), HALF_F16)

    def test_signed_zero_rules(self) -> None:
        self.assertEqual(ref_sqrt_clamp_f16(POS_ZERO), POS_ZERO)
        # -0.0 clamps to +0.0
        self.assertEqual(ref_sqrt_clamp_f16(NEG_ZERO), POS_ZERO)

    def test_negative_inputs_clamped_to_zero(self) -> None:
        self.assertEqual(ref_sqrt_clamp_f16(0xBC00), POS_ZERO)  # -1.0
        self.assertEqual(ref_sqrt_clamp_f16(FOUR_F16 | SIGN_MASK), POS_ZERO)  # -4.0
        self.assertEqual(ref_sqrt_clamp_f16(NEG_INF), POS_ZERO)

    def test_nan_inputs_clamped_to_zero(self) -> None:
        self.assertEqual(ref_sqrt_clamp_f16(CANONICAL_QNAN), POS_ZERO)
        self.assertEqual(ref_sqrt_clamp_f16(0x7C05), POS_ZERO)  # SNaN

    def test_positive_inf_clamped_to_one(self) -> None:
        self.assertEqual(ref_sqrt_clamp_f16(POS_INF), ONE_F16)

    def test_input_modifiers_abs_neg(self) -> None:
        val_neg_four = FOUR_F16 | SIGN_MASK  # -4.0
        # abs on -4.0: sqrt(|-4.0|) = sqrt(4.0) = 2.0 -> clamped to 1.0
        res = ref_sqrt_clamp_f16(val_neg_four, abs0=True)
        self.assertEqual(res, ONE_F16)

        # neg on 0.25: -(0.25) = -0.25 -> clamped to +0.0
        res_neg = ref_sqrt_clamp_f16(QUARTER_F16, neg0=True)
        self.assertEqual(res_neg, POS_ZERO)


if __name__ == "__main__":
    unittest.main()
