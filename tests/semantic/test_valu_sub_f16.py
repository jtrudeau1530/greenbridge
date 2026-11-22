from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import List, Optional, Tuple, Union

# IEEE 754 binary16 (half precision) architectural constants
SIGN_MASK_F16: int = 0x8000
EXP_MASK_F16: int = 0x7C00
FRAC_MASK_F16: int = 0x03FF
QNAN_BIT_F16: int = 0x0200
CANONICAL_QNAN_F16: int = 0x7E00

POS_ZERO_F16: int = 0x0000
NEG_ZERO_F16: int = 0x8000
POS_INF_F16: int = 0x7C00
NEG_INF_F16: int = 0xFC00

ONE_F16: int = 0x3C00
HALF_F16: int = 0x3800
TWO_F16: int = 0x4000
MAX_FINITE_F16: int = 0x7BFF
MIN_NORMAL_F16: int = 0x0400
MIN_SUBNORMAL_F16: int = 0x0001
MAX_SUBNORMAL_F16: int = 0x03FF


def f16_to_bits(val: float) -> int:
    """Convert a Python float to 16-bit IEEE 754 half-precision integer bits."""
    if math.isnan(val):
        return CANONICAL_QNAN_F16
    if math.isinf(val):
        return POS_INF_F16 if val > 0 else NEG_INF_F16
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return POS_INF_F16 if val > 0 else NEG_INF_F16


def bits_to_f16(bits: int) -> float:
    """Convert 16-bit integer bits to a Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def split_f16_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK_F16) >> 15
    exponent = (bits & EXP_MASK_F16) >> 10
    fraction = bits & FRAC_MASK_F16
    return sign, exponent, fraction


def unpack_half_word(word: int, hi: bool) -> int:
    """Extract low (hi=False) or high (hi=True) 16-bit half from a 32-bit register word."""
    return ((word >> 16) if hi else word) & 0xFFFF


def pack_half_word(old_word: int, val16: int, hi: bool) -> int:
    """Merge 16-bit result into low or high half of 32-bit register word, preserving the other half."""
    val16 &= 0xFFFF
    if hi:
        return (old_word & 0x0000FFFF) | (val16 << 16)
    else:
        return (old_word & 0xFFFF0000) | val16


def is_nan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & FRAC_MASK_F16) != 0)


def is_qnan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision Quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_snan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision Signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """Return True if bits represent IEEE 754 half-precision positive or negative infinity."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & FRAC_MASK_F16) == 0)


def is_subnormal_f16(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormal) half-precision float."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & FRAC_MASK_F16) != 0)


def is_zero_f16(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK_F16) == 0


def apply_f16_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & 0xFFFF
    if abs_mod:
        res &= ~SIGN_MASK_F16
    if neg_mod:
        res ^= SIGN_MASK_F16
    return res


def oracle_v_sub_f16(
    src0_bits: int,
    src1_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_sub_f16 half-precision subtraction.

    Computes D = src0 - src1 conforming to IEEE 754-2008 half precision (binary16)
    with Round-to-Nearest-Even (RNE), applying input modifiers and clamp.
    """
    s0 = apply_f16_modifiers(src0_bits, abs0, neg0)
    s1 = apply_f16_modifiers(src1_bits, abs1, neg1)

    # 1. NaN handling
    if is_nan_f16(s0) or is_nan_f16(s1):
        if clamp:
            # Clamping NaN in RDNA2 produces +0.0
            return POS_ZERO_F16
        # Prefer quietening signaling NaN or propagating quiet NaN
        nan_bits = s0 if is_nan_f16(s0) else s1
        return nan_bits | QNAN_BIT_F16

    # 2. Infinity handling
    s0_inf = is_inf_f16(s0)
    s1_inf = is_inf_f16(s1)
    if s0_inf and s1_inf:
        s0_sign = s0 & SIGN_MASK_F16
        s1_sign = s1 & SIGN_MASK_F16
        # (+inf) - (+inf) or (-inf) - (-inf) -> Invalid operation -> Canonical Quiet NaN
        if s0_sign == s1_sign:
            return POS_ZERO_F16 if clamp else CANONICAL_QNAN_F16
        # (+inf) - (-inf) = +inf
        # (-inf) - (+inf) = -inf
        res = s0
        if clamp:
            return ONE_F16 if res == POS_INF_F16 else POS_ZERO_F16
        return res
    if s0_inf:
        if clamp:
            return ONE_F16 if s0 == POS_INF_F16 else POS_ZERO_F16
        return s0
    if s1_inf:
        # finite - (+inf) = -inf; finite - (-inf) = +inf
        res = NEG_INF_F16 if s1 == POS_INF_F16 else POS_INF_F16
        if clamp:
            return ONE_F16 if res == POS_INF_F16 else POS_ZERO_F16
        return res

    # 3. Signed zero handling
    if is_zero_f16(s0) and is_zero_f16(s1):
        s0_is_neg = (s0 == NEG_ZERO_F16)
        s1_is_neg = (s1 == NEG_ZERO_F16)
        # (+0) - (+0) = +0
        # (+0) - (-0) = +0
        # (-0) - (-0) = +0 (Round-to-Nearest-Even)
        # (-0) - (+0) = -0
        if s0_is_neg and not s1_is_neg:
            res = NEG_ZERO_F16
        else:
            res = POS_ZERO_F16
        if clamp and res == NEG_ZERO_F16:
            return POS_ZERO_F16
        return res

    # 4. Finite subtraction
    f0 = bits_to_f16(s0)
    f1 = bits_to_f16(s1)
    f_res = f0 - f1
    res = f16_to_bits(f_res)

    # In round-to-nearest-even, x - x produces +0.0
    if f_res == 0.0 and not (s0 == NEG_ZERO_F16 and s1 == POS_ZERO_F16):
        res = POS_ZERO_F16

    # 5. Clamp modifier ([0.0, 1.0])
    if clamp:
        if f_res < 0.0 or res == NEG_ZERO_F16:
            res = POS_ZERO_F16
        elif f_res > 1.0 or res == POS_INF_F16:
            res = ONE_F16

    return res


class TestF16ModelAndOracle(unittest.TestCase):
    """Unit tests for binary16 bit model, predicates, and reference mathematical oracle."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 65504.0, -65504.0]
        for val in values:
            with self.subTest(val=val):
                bits = f16_to_bits(val)
                recovered = bits_to_f16(bits)
                self.assertEqual(val, recovered)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_halfword_packing_and_unpacking(self) -> None:
        old_reg = 0x12345678
        val_lo = 0x3C00  # 1.0
        val_hi = 0x4000  # 2.0

        updated_lo = pack_half_word(old_reg, val_lo, hi=False)
        self.assertEqual(updated_lo & 0xFFFF, val_lo)
        self.assertEqual((updated_lo >> 16) & 0xFFFF, 0x1234)
        self.assertEqual(unpack_half_word(updated_lo, hi=False), val_lo)
        self.assertEqual(unpack_half_word(updated_lo, hi=True), 0x1234)

        updated_hi = pack_half_word(old_reg, val_hi, hi=True)
        self.assertEqual((updated_hi >> 16) & 0xFFFF, val_hi)
        self.assertEqual(updated_hi & 0xFFFF, 0x5678)
        self.assertEqual(unpack_half_word(updated_hi, hi=True), val_hi)
        self.assertEqual(unpack_half_word(updated_hi, hi=False), 0x5678)

    def test_classification_predicates(self) -> None:
        self.assertTrue(f16_to_bits(0.0) == POS_ZERO_F16)
        self.assertTrue(f16_to_bits(-0.0) == NEG_ZERO_F16)
        self.assertTrue(is_zero_f16(POS_ZERO_F16))
        self.assertTrue(is_zero_f16(NEG_ZERO_F16))
        self.assertFalse(is_zero_f16(f16_to_bits(1.0)))

        self.assertTrue(is_inf_f16(POS_INF_F16))
        self.assertTrue(is_inf_f16(NEG_INF_F16))
        self.assertFalse(is_inf_f16(POS_ZERO_F16))

        snan_bits = 0x7C01
        qnan_bits = CANONICAL_QNAN_F16
        self.assertTrue(is_nan_f16(snan_bits))
        self.assertTrue(is_snan_f16(snan_bits))
        self.assertFalse(is_qnan_f16(snan_bits))

        self.assertTrue(is_nan_f16(qnan_bits))
        self.assertTrue(is_qnan_f16(qnan_bits))
        self.assertFalse(is_snan_f16(qnan_bits))

        subnorm_bits = MIN_SUBNORMAL_F16
        self.assertTrue(is_subnormal_f16(subnorm_bits))
        self.assertFalse(is_subnormal_f16(POS_ZERO_F16))
        self.assertFalse(is_subnormal_f16(f16_to_bits(1.0)))

    def test_exact_finite_subtraction(self) -> None:
        c = oracle_v_sub_f16(f16_to_bits(3.75), f16_to_bits(1.5))
        self.assertEqual(bits_to_f16(c), 2.25)

        c2 = oracle_v_sub_f16(f16_to_bits(10.0), f16_to_bits(25.5))
        self.assertEqual(bits_to_f16(c2), -15.5)

    def test_signed_zero_rules(self) -> None:
        # IEEE 754: (+0) - (+0) = +0
        self.assertEqual(oracle_v_sub_f16(POS_ZERO_F16, POS_ZERO_F16), POS_ZERO_F16)
        # (-0) - (-0) = +0
        self.assertEqual(oracle_v_sub_f16(NEG_ZERO_F16, NEG_ZERO_F16), POS_ZERO_F16)
        # (+0) - (-0) = +0
        self.assertEqual(oracle_v_sub_f16(POS_ZERO_F16, NEG_ZERO_F16), POS_ZERO_F16)
        # (-0) - (+0) = -0
        self.assertEqual(oracle_v_sub_f16(NEG_ZERO_F16, POS_ZERO_F16), NEG_ZERO_F16)
        # x - x = +0
        self.assertEqual(oracle_v_sub_f16(f16_to_bits(42.0), f16_to_bits(42.0)), POS_ZERO_F16)

    def test_input_modifiers_abs_neg(self) -> None:
        val_pos = f16_to_bits(10.0)
        val_neg = f16_to_bits(-10.0)

        # abs on negative: |-10.0| - 5.0 = 5.0
        res = oracle_v_sub_f16(val_neg, f16_to_bits(5.0), abs0=True)
        self.assertEqual(bits_to_f16(res), 5.0)

        # neg on positive: -(10.0) - 5.0 = -15.0
        res = oracle_v_sub_f16(val_pos, f16_to_bits(5.0), neg0=True)
        self.assertEqual(bits_to_f16(res), -15.0)

        # abs and neg together: -|10.0| - 5.0 = -15.0; -|-10.0| - 5.0 = -15.0
        res1 = oracle_v_sub_f16(val_pos, f16_to_bits(5.0), abs0=True, neg0=True)
        res2 = oracle_v_sub_f16(val_neg, f16_to_bits(5.0), abs0=True, neg0=True)
        self.assertEqual(bits_to_f16(res1), -15.0)
        self.assertEqual(bits_to_f16(res2), -15.0)

    def test_clamp_modifier(self) -> None:
        # Clamped result > 1.0 becomes 1.0
        res_hi = oracle_v_sub_f16(f16_to_bits(2.5), f16_to_bits(0.5), clamp=True)
        self.assertEqual(res_hi, ONE_F16)

        # Clamped result < 0.0 becomes +0.0
        res_lo = oracle_v_sub_f16(f16_to_bits(0.5), f16_to_bits(1.5), clamp=True)
        self.assertEqual(res_lo, POS_ZERO_F16)

        # Clamped result within [0.0, 1.0] is unmodified
        res_mid = oracle_v_sub_f16(f16_to_bits(0.75), f16_to_bits(0.25), clamp=True)
        self.assertEqual(bits_to_f16(res_mid), 0.5)

    def test_infinity_subtraction(self) -> None:
        self.assertEqual(oracle_v_sub_f16(POS_INF_F16, f16_to_bits(1.0)), POS_INF_F16)
        self.assertEqual(oracle_v_sub_f16(NEG_INF_F16, f16_to_bits(1.0)), NEG_INF_F16)
        self.assertEqual(oracle_v_sub_f16(f16_to_bits(1.0), POS_INF_F16), NEG_INF_F16)
        self.assertEqual(oracle_v_sub_f16(f16_to_bits(1.0), NEG_INF_F16), POS_INF_F16)
        # (+inf) - (+inf) -> Canonical NaN
        self.assertEqual(oracle_v_sub_f16(POS_INF_F16, POS_INF_F16), CANONICAL_QNAN_F16)
        # (-inf) - (-inf) -> Canonical NaN
        self.assertEqual(oracle_v_sub_f16(NEG_INF_F16, NEG_INF_F16), CANONICAL_QNAN_F16)
        # (+inf) - (-inf) -> +inf
        self.assertEqual(oracle_v_sub_f16(POS_INF_F16, NEG_INF_F16), POS_INF_F16)
        # (-inf) - (+inf) -> -inf
        self.assertEqual(oracle_v_sub_f16(NEG_INF_F16, POS_INF_F16), NEG_INF_F16)

    def test_nan_quietening_and_propagation(self) -> None:
        snan = 0x7C05
        res = oracle_v_sub_f16(snan, f16_to_bits(1.0))
        self.assertTrue(is_qnan_f16(res))
        self.assertEqual(res, snan | QNAN_BIT_F16)


if __name__ == "__main__":
    unittest.main()
