from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import List, Optional, Tuple, Union

# IEEE 754 binary32 (single precision) architectural constants
SIGN_MASK: int = 0x8000_0000
EXP_MASK: int = 0x7F80_0000
FRAC_MASK: int = 0x007F_FFFF
QNAN_BIT: int = 0x0040_0000
CANONICAL_QNAN: int = 0x7FC0_0000

POS_ZERO: int = 0x0000_0000
NEG_ZERO: int = 0x8000_0000
POS_INF: int = 0x7F80_0000
NEG_INF: int = 0xFF80_0000

ONE_F32: int = 0x3F80_0000


def f32_to_bits(val: float) -> int:
    """Convert a Python float to 32-bit IEEE 754 single-precision integer bits."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_f32(bits: int) -> float:
    """Convert 32-bit integer bits to a Python float (IEEE 754 single precision)."""
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF_FFFF))[0]


def split_f32_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK) >> 31
    exponent = (bits & EXP_MASK) >> 23
    fraction = bits & FRAC_MASK
    return sign, exponent, fraction


def is_nan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 single-precision NaN."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) != 0)


def is_qnan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Quiet NaN."""
    return is_nan(bits) and bool(bits & QNAN_BIT)


def is_snan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Signaling NaN."""
    return is_nan(bits) and not bool(bits & QNAN_BIT)


def is_inf(bits: int) -> bool:
    """Return True if bits represent IEEE 754 positive or negative infinity."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) == 0)


def is_subnormal(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormal) single-precision float."""
    return ((bits & EXP_MASK) == 0) and ((bits & FRAC_MASK) != 0)


def is_zero(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK) == 0


def apply_f32_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & 0xFFFF_FFFF
    if abs_mod:
        res &= ~SIGN_MASK
    if neg_mod:
        res ^= SIGN_MASK
    return res


def oracle_v_sqrt_f32(
    src_bits: int,
    abs_mod: bool = False,
    neg_mod: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_sqrt_f32 single-precision square root.

    Computes D = sqrt(src) conforming to IEEE 754-2008 single precision (binary32)
    with Round-to-Nearest-Even (RNE), applying input modifiers and clamp.
    """
    s = apply_f32_modifiers(src_bits, abs_mod, neg_mod)

    # 1. NaN handling
    if is_nan(s):
        if clamp:
            return POS_ZERO
        return s | QNAN_BIT

    # 2. Signed zero handling: sqrt(+0) = +0, sqrt(-0) = -0
    if is_zero(s):
        if s == NEG_ZERO:
            return POS_ZERO if clamp else NEG_ZERO
        return POS_ZERO

    # 3. Negative values (< 0): Invalid operation -> Canonical Quiet NaN
    if bool(s & SIGN_MASK):
        return POS_ZERO if clamp else CANONICAL_QNAN

    # 4. Positive infinity: sqrt(+inf) = +inf
    if s == POS_INF:
        return ONE_F32 if clamp else POS_INF

    # 5. Finite positive values
    f_val = bits_to_f32(s)
    res_f = math.sqrt(f_val)
    res = f32_to_bits(res_f)

    # 6. Clamp modifier ([0.0, 1.0])
    if clamp:
        if res_f < 0.0 or res == NEG_ZERO:
            res = POS_ZERO
        elif res_f > 1.0 or res == POS_INF:
            res = ONE_F32

    return res


# ==============================================================================
# Unit Tests
# ==============================================================================

class TestF32ModelAndOracle(unittest.TestCase):
    """Unit tests for binary32 bit model, predicates, and reference oracle."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.25, 4.0, 1e-10, 1e10, 3.1415927]
        for val in values:
            with self.subTest(val=val):
                bits = f32_to_bits(val)
                recovered = bits_to_f32(bits)
                self.assertEqual(f32_to_bits(recovered), bits)
                if not math.isnan(val):
                    self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_classification_predicates(self) -> None:
        self.assertTrue(f32_to_bits(0.0) == POS_ZERO)
        self.assertTrue(f32_to_bits(-0.0) == NEG_ZERO)
        self.assertTrue(is_zero(POS_ZERO))
        self.assertTrue(is_zero(NEG_ZERO))
        self.assertFalse(is_zero(f32_to_bits(1.0)))

        self.assertTrue(is_inf(POS_INF))
        self.assertTrue(is_inf(NEG_INF))
        self.assertFalse(is_inf(POS_ZERO))

        snan_bits = 0x7F80_0001
        qnan_bits = CANONICAL_QNAN
        self.assertTrue(is_nan(snan_bits))
        self.assertTrue(is_snan(snan_bits))
        self.assertFalse(is_qnan(snan_bits))

        self.assertTrue(is_nan(qnan_bits))
        self.assertTrue(is_qnan(qnan_bits))
        self.assertFalse(is_snan(qnan_bits))

        subnorm_bits = 0x0000_0001
        self.assertTrue(is_subnormal(subnorm_bits))
        self.assertFalse(is_subnormal(POS_ZERO))
        self.assertFalse(is_subnormal(f32_to_bits(1.0)))

    def test_exact_finite_sqrt(self) -> None:
        c4 = oracle_v_sqrt_f32(f32_to_bits(4.0))
        self.assertEqual(bits_to_f32(c4), 2.0)

        c9 = oracle_v_sqrt_f32(f32_to_bits(9.0))
        self.assertEqual(bits_to_f32(c9), 3.0)

        c025 = oracle_v_sqrt_f32(f32_to_bits(0.25))
        self.assertEqual(bits_to_f32(c025), 0.5)

        c100 = oracle_v_sqrt_f32(f32_to_bits(100.0))
        self.assertEqual(bits_to_f32(c100), 10.0)

    def test_signed_zero_rules(self) -> None:
        self.assertEqual(oracle_v_sqrt_f32(POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_v_sqrt_f32(NEG_ZERO), NEG_ZERO)

    def test_negative_values_produce_canonical_qnan(self) -> None:
        self.assertEqual(oracle_v_sqrt_f32(f32_to_bits(-4.0)), CANONICAL_QNAN)
        self.assertEqual(oracle_v_sqrt_f32(NEG_INF), CANONICAL_QNAN)

    def test_input_modifiers_abs_neg(self) -> None:
        val_neg = f32_to_bits(-16.0)
        res_abs = oracle_v_sqrt_f32(val_neg, abs_mod=True)
        self.assertEqual(bits_to_f32(res_abs), 4.0)

        res_neg = oracle_v_sqrt_f32(val_neg, neg_mod=True)
        self.assertEqual(bits_to_f32(res_neg), 4.0)

        res_neg_pos = oracle_v_sqrt_f32(f32_to_bits(16.0), neg_mod=True)
        self.assertEqual(res_neg_pos, CANONICAL_QNAN)

        res_both = oracle_v_sqrt_f32(val_neg, abs_mod=True, neg_mod=True)
        self.assertEqual(res_both, CANONICAL_QNAN)

    def test_clamp_modifier(self) -> None:
        res_hi = oracle_v_sqrt_f32(f32_to_bits(4.0), clamp=True)
        self.assertEqual(res_hi, ONE_F32)

        res_mid = oracle_v_sqrt_f32(f32_to_bits(0.25), clamp=True)
        self.assertEqual(bits_to_f32(res_mid), 0.5)

        res_nz = oracle_v_sqrt_f32(NEG_ZERO, clamp=True)
        self.assertEqual(res_nz, POS_ZERO)

        res_neg = oracle_v_sqrt_f32(f32_to_bits(-4.0), clamp=True)
        self.assertEqual(res_neg, POS_ZERO)

    def test_infinity_sqrt(self) -> None:
        self.assertEqual(oracle_v_sqrt_f32(POS_INF), POS_INF)
        self.assertEqual(oracle_v_sqrt_f32(POS_INF, clamp=True), ONE_F32)
        self.assertEqual(oracle_v_sqrt_f32(NEG_INF), CANONICAL_QNAN)
        self.assertEqual(oracle_v_sqrt_f32(NEG_INF, clamp=True), POS_ZERO)

    def test_nan_quietening_and_propagation(self) -> None:
        snan = 0x7F80_1234
        res = oracle_v_sqrt_f32(snan)
        self.assertTrue(is_qnan(res))
        self.assertEqual(res, snan | QNAN_BIT)

        qnan = 0x7FC0_8765
        res_q = oracle_v_sqrt_f32(qnan)
        self.assertEqual(res_q, qnan)


if __name__ == "__main__":
    unittest.main()
