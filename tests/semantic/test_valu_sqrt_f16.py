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


def oracle_v_sqrt_f16(
    src0_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_sqrt_f16 half-precision square root.

    Computes D = sqrt(src0) conforming to IEEE 754-2008 binary16
    with Round-to-Nearest-Even (RNE), applying input modifiers and clamp.
    """
    s0 = apply_f16_modifiers(src0_bits, abs0, neg0)

    # 1. NaN handling
    if is_nan(s0):
        if clamp:
            return POS_ZERO
        return s0 | QNAN_BIT

    # 2. Negative inputs (excluding negative zero)
    # In IEEE 754 and RDNA2, sqrt of negative non-zero produces Canonical Quiet NaN
    if (s0 & SIGN_MASK) != 0 and s0 != NEG_ZERO:
        return POS_ZERO if clamp else CANONICAL_QNAN

    # 3. Infinity handling (+inf -> +inf, -inf was caught above as < 0)
    if is_inf(s0):
        if clamp:
            return ONE_F16
        return POS_INF

    # 4. Zero handling (+0 -> +0, -0 -> -0)
    if is_zero(s0):
        if clamp:
            return POS_ZERO
        return s0

    # 5. Finite positive square root (normals and subnormals)
    f0 = bits_to_f16(s0)
    res_f = math.sqrt(f0)
    res = f16_to_bits(res_f)

    # 6. Clamp modifier ([0.0, 1.0])
    if clamp:
        if res_f < 0.0 or res == NEG_ZERO:
            res = POS_ZERO
        elif res_f > 1.0 or res == POS_INF:
            res = ONE_F16

    return res


# ==============================================================================
# Unit Tests
# ==============================================================================

class TestF16ModelAndOracle(unittest.TestCase):
    """Unit tests for binary16 bit model, predicates, and reference oracle."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 4.0, 9.0, 16.0]
        for val in values:
            with self.subTest(val=val):
                bits = f16_to_bits(val)
                recovered = bits_to_f16(bits)
                self.assertEqual(val, recovered)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_halfword_packing_vgpr(self) -> None:
        lo = 0x3C00  # 1.0
        hi = 0x4000  # 2.0
        dword = f16_halves_to_dword(lo, hi)
        self.assertEqual(dword, 0x40003C00)
        unpacked_lo, unpacked_hi = dword_to_f16_halves(dword)
        self.assertEqual(unpacked_lo, lo)
        self.assertEqual(unpacked_hi, hi)
        self.assertEqual(bits_to_f16(unpacked_lo), 1.0)
        self.assertEqual(bits_to_f16(unpacked_hi), 2.0)

    def test_classification_predicates(self) -> None:
        self.assertTrue(f16_to_bits(0.0) == POS_ZERO)
        self.assertTrue(f16_to_bits(-0.0) == NEG_ZERO)
        self.assertTrue(is_zero(POS_ZERO))
        self.assertTrue(is_zero(NEG_ZERO))
        self.assertFalse(is_zero(f16_to_bits(1.0)))

        self.assertTrue(is_inf(POS_INF))
        self.assertTrue(is_inf(NEG_INF))
        self.assertFalse(is_inf(POS_ZERO))

        snan_bits = 0x7C01  # exponent 0x1F, fraction != 0, quiet bit 9 is 0
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
        self.assertFalse(is_subnormal(f16_to_bits(1.0)))

    def test_exact_finite_sqrt(self) -> None:
        self.assertEqual(oracle_v_sqrt_f16(FOUR_F16), TWO_F16)
        self.assertEqual(oracle_v_sqrt_f16(NINE_F16), THREE_F16)
        self.assertEqual(oracle_v_sqrt_f16(ONE_F16), ONE_F16)
        self.assertEqual(oracle_v_sqrt_f16(QUARTER_F16), HALF_F16)

    def test_signed_zero_rules(self) -> None:
        # IEEE 754: sqrt(+0.0) = +0.0, sqrt(-0.0) = -0.0
        self.assertEqual(oracle_v_sqrt_f16(POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_v_sqrt_f16(NEG_ZERO), NEG_ZERO)

    def test_input_modifiers_abs_neg(self) -> None:
        val_neg_four = FOUR_F16 | SIGN_MASK  # -4.0
        # abs on -4.0: sqrt(|-4.0|) = sqrt(4.0) = 2.0
        res = oracle_v_sqrt_f16(val_neg_four, abs0=True)
        self.assertEqual(res, TWO_F16)

        # neg on 4.0: sqrt(-(4.0)) = sqrt(-4.0) = NaN
        res_neg = oracle_v_sqrt_f16(FOUR_F16, neg0=True)
        self.assertTrue(is_qnan(res_neg))
        self.assertEqual(res_neg, CANONICAL_QNAN)

        # abs and neg together: -|4.0| = -4.0 -> NaN
        res_neg_abs = oracle_v_sqrt_f16(FOUR_F16, abs0=True, neg0=True)
        self.assertEqual(res_neg_abs, CANONICAL_QNAN)

    def test_clamp_modifier(self) -> None:
        # sqrt(4.0) = 2.0 -> clamped to 1.0
        res_clamped_hi = oracle_v_sqrt_f16(FOUR_F16, clamp=True)
        self.assertEqual(res_clamped_hi, ONE_F16)

        # sqrt(0.25) = 0.5 -> within [0, 1], remains 0.5
        res_mid = oracle_v_sqrt_f16(QUARTER_F16, clamp=True)
        self.assertEqual(res_mid, HALF_F16)

        # sqrt(-4.0) with clamp produces +0.0
        res_nan_clamp = oracle_v_sqrt_f16(FOUR_F16 | SIGN_MASK, clamp=True)
        self.assertEqual(res_nan_clamp, POS_ZERO)

        # sqrt(-0.0) with clamp produces +0.0
        res_zero_clamp = oracle_v_sqrt_f16(NEG_ZERO, clamp=True)
        self.assertEqual(res_zero_clamp, POS_ZERO)

    def test_infinity_sqrt(self) -> None:
        self.assertEqual(oracle_v_sqrt_f16(POS_INF), POS_INF)
        self.assertEqual(oracle_v_sqrt_f16(NEG_INF), CANONICAL_QNAN)
        # Clamped +inf yields 1.0
        self.assertEqual(oracle_v_sqrt_f16(POS_INF, clamp=True), ONE_F16)

    def test_negative_operand_produces_qnan(self) -> None:
        self.assertEqual(oracle_v_sqrt_f16(0xBC00), CANONICAL_QNAN)  # -1.0
        self.assertEqual(oracle_v_sqrt_f16(0x8001), CANONICAL_QNAN)  # negative subnormal

    def test_nan_quietening_and_propagation(self) -> None:
        snan = 0x7C05  # Signaling NaN
        res = oracle_v_sqrt_f16(snan)
        self.assertTrue(is_qnan(res))
        self.assertEqual(res, snan | QNAN_BIT)


if __name__ == "__main__":
    unittest.main()
