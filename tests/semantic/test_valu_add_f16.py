from __future__ import annotations

import math
import struct
import unittest
from typing import Tuple

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
MAX_NORMAL_F16: int = 0x7BFF
MIN_NORMAL_F16: int = 0x0400
MAX_SUBNORMAL_F16: int = 0x03FF
MIN_SUBNORMAL_F16: int = 0x0001


def f16_to_bits(val: float) -> int:
    """Convert a Python float to 16-bit integer bits (IEEE 754 binary16)."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f16(bits: int) -> float:
    """Convert 16-bit integer bits to a Python float (IEEE 754 binary16)."""
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def split_f16_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK) >> 15
    exponent = (bits & EXP_MASK) >> 10
    fraction = bits & FRAC_MASK
    return sign, exponent, fraction


def pack_f16_pair(lo: int, hi: int) -> int:
    """Pack two 16-bit values into a 32-bit register word."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def unpack_f16_pair(val32: int) -> Tuple[int, int]:
    """Unpack a 32-bit register word into two 16-bit half words (lo, hi)."""
    return val32 & 0xFFFF, (val32 >> 16) & 0xFFFF


def is_nan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 NaN."""
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


def oracle_v_add_f16(
    src0_bits: int,
    src1_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_add_f16 half-precision addition.
    
    Computes D = src0 + src1 conforming to IEEE 754-2008 half precision (binary16)
    with Round-to-Nearest-Even (RNE), applying input modifiers and clamp.
    """
    s0 = apply_f16_modifiers(src0_bits, abs0, neg0)
    s1 = apply_f16_modifiers(src1_bits, abs1, neg1)

    # 1. NaN handling
    if is_nan(s0) or is_nan(s1):
        if clamp:
            # Clamping NaN in RDNA2 produces +0.0
            return POS_ZERO
        # Prefer quietening signaling NaN or propagating quiet NaN
        nan_bits = s0 if is_nan(s0) else s1
        return nan_bits | QNAN_BIT

    # 2. Infinity handling
    s0_inf = is_inf(s0)
    s1_inf = is_inf(s1)
    if s0_inf and s1_inf:
        s0_sign = (s0 & SIGN_MASK)
        s1_sign = (s1 & SIGN_MASK)
        if s0_sign != s1_sign:
            # +inf + -inf -> Invalid operation -> Canonical Quiet NaN
            return POS_ZERO if clamp else CANONICAL_QNAN
        # Same signed infinities
        if clamp:
            return ONE_F16 if s0 == POS_INF else POS_ZERO
        return s0
    if s0_inf:
        if clamp:
            return ONE_F16 if s0 == POS_INF else POS_ZERO
        return s0
    if s1_inf:
        if clamp:
            return ONE_F16 if s1 == POS_INF else POS_ZERO
        return s1

    # 3. Signed zero handling
    if is_zero(s0) and is_zero(s1):
        s0_is_neg = (s0 == NEG_ZERO)
        s1_is_neg = (s1 == NEG_ZERO)
        if s0_is_neg and s1_is_neg:
            res = NEG_ZERO
        else:
            res = POS_ZERO
        if clamp and res == NEG_ZERO:
            return POS_ZERO
        return res

    # 4. Finite arithmetic
    f0 = bits_to_f16(s0)
    f1 = bits_to_f16(s1)
    f_sum = f0 + f1

    # Exact signed zero cancellation: (+x) + (-x) -> +0.0
    if f_sum == 0.0:
        return POS_ZERO

    try:
        raw_res = f16_to_bits(f_sum)
    except OverflowError:
        raw_res = POS_INF if f_sum > 0 else NEG_INF

    # 5. Clamp modifier handling [0.0, 1.0]
    if clamp:
        if is_inf(raw_res):
            return ONE_F16 if raw_res == POS_INF else POS_ZERO
        res_f = bits_to_f16(raw_res)
        if res_f < 0.0 or raw_res == NEG_ZERO:
            return POS_ZERO
        elif res_f > 1.0:
            return ONE_F16

    return raw_res


class TestF16ModelAndOracle(unittest.TestCase):
    """Unit tests for binary16 bit model, predicates, and reference oracle."""

    def test_round_trip_conversion(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 65504.0, -65504.0, 3.140625]
        for val in values:
            with self.subTest(val=val):
                bits = f16_to_bits(val)
                recovered = bits_to_f16(bits)
                self.assertEqual(val, recovered)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_word_packing_vgpr_pair(self) -> None:
        lo = 0x3C00  # 1.0
        hi = 0x4000  # 2.0
        packed = pack_f16_pair(lo, hi)
        self.assertEqual(packed, 0x40003C00)
        unpacked_lo, unpacked_hi = unpack_f16_pair(packed)
        self.assertEqual(unpacked_lo, lo)
        self.assertEqual(unpacked_hi, hi)

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
        self.assertFalse(is_subnormal(f16_to_bits(1.0)))

    def test_exact_finite_addition(self) -> None:
        c = oracle_v_add_f16(f16_to_bits(1.5), f16_to_bits(2.25))
        self.assertEqual(bits_to_f16(c), 3.75)

        c2 = oracle_v_add_f16(f16_to_bits(100.0), f16_to_bits(-25.5))
        self.assertEqual(bits_to_f16(c2), 74.5)

    def test_signed_zero_rules(self) -> None:
        # IEEE 754: (+0) + (+0) = +0
        self.assertEqual(oracle_v_add_f16(POS_ZERO, POS_ZERO), POS_ZERO)
        # (-0) + (-0) = -0
        self.assertEqual(oracle_v_add_f16(NEG_ZERO, NEG_ZERO), NEG_ZERO)
        # (+0) + (-0) = +0
        self.assertEqual(oracle_v_add_f16(POS_ZERO, NEG_ZERO), POS_ZERO)
        # (-0) + (+0) = +0
        self.assertEqual(oracle_v_add_f16(NEG_ZERO, POS_ZERO), POS_ZERO)
        # x + (-x) = +0
        self.assertEqual(oracle_v_add_f16(f16_to_bits(42.0), f16_to_bits(-42.0)), POS_ZERO)

    def test_input_modifiers_abs_neg(self) -> None:
        val_pos = f16_to_bits(10.0)
        val_neg = f16_to_bits(-10.0)

        # abs on negative: |-10.0| + 5.0 = 15.0
        res = oracle_v_add_f16(val_neg, f16_to_bits(5.0), abs0=True)
        self.assertEqual(bits_to_f16(res), 15.0)

        # neg on positive: -(10.0) + 5.0 = -5.0
        res = oracle_v_add_f16(val_pos, f16_to_bits(5.0), neg0=True)
        self.assertEqual(bits_to_f16(res), -5.0)

        # abs and neg together: -|10.0| + 5.0 = -5.0; -|-10.0| + 5.0 = -5.0
        res1 = oracle_v_add_f16(val_pos, f16_to_bits(5.0), abs0=True, neg0=True)
        res2 = oracle_v_add_f16(val_neg, f16_to_bits(5.0), abs0=True, neg0=True)
        self.assertEqual(bits_to_f16(res1), -5.0)
        self.assertEqual(bits_to_f16(res2), -5.0)

    def test_clamp_modifier(self) -> None:
        # Clamped result > 1.0 becomes 1.0
        res_hi = oracle_v_add_f16(f16_to_bits(0.75), f16_to_bits(0.5), clamp=True)
        self.assertEqual(res_hi, ONE_F16)

        # Clamped result < 0.0 becomes +0.0
        res_lo = oracle_v_add_f16(f16_to_bits(-1.5), f16_to_bits(0.5), clamp=True)
        self.assertEqual(res_lo, POS_ZERO)

        # Clamped result within [0.0, 1.0] is unmodified
        res_mid = oracle_v_add_f16(f16_to_bits(0.25), f16_to_bits(0.5), clamp=True)
        self.assertEqual(bits_to_f16(res_mid), 0.75)

    def test_infinity_addition(self) -> None:
        self.assertEqual(oracle_v_add_f16(POS_INF, f16_to_bits(1.0)), POS_INF)
        self.assertEqual(oracle_v_add_f16(NEG_INF, f16_to_bits(1.0)), NEG_INF)
        self.assertEqual(oracle_v_add_f16(POS_INF, POS_INF), POS_INF)
        self.assertEqual(oracle_v_add_f16(NEG_INF, NEG_INF), NEG_INF)
        # (+inf) + (-inf) -> Canonical NaN
        self.assertEqual(oracle_v_add_f16(POS_INF, NEG_INF), CANONICAL_QNAN)

    def test_nan_quietening_and_propagation(self) -> None:
        snan = 0x7C05
        res = oracle_v_add_f16(snan, f16_to_bits(1.0))
        self.assertTrue(is_qnan(res))
        self.assertEqual(res, snan | QNAN_BIT)


if __name__ == "__main__":
    unittest.main()
