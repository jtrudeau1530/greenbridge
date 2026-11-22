from __future__ import annotations

import math
import struct
import unittest
from fractions import Fraction
from typing import List, Optional, Tuple, Union

# IEEE 754 binary16 (half-precision) architectural constants
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


def f16_to_bits(val: float) -> int:
    """Convert a Python float to 16-bit IEEE 754 half-precision bits."""
    if math.isnan(val):
        return CANONICAL_QNAN_F16
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f16(bits: int) -> float:
    """Convert 16-bit integer bits to a Python float (IEEE 754 half precision)."""
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def pack_2xf16(lo: int, hi: int) -> int:
    """Pack two 16-bit halfwords into a 32-bit dword."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def unpack_2xf16(dword: int) -> Tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit halfwords (lo, hi)."""
    return dword & 0xFFFF, (dword >> 16) & 0xFFFF


def split_f16_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK_F16) >> 15
    exponent = (bits & EXP_MASK_F16) >> 10
    fraction = bits & FRAC_MASK_F16
    return sign, exponent, fraction


def is_nan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & FRAC_MASK_F16) != 0)


def is_qnan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_snan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """Return True if bits represent IEEE 754 positive or negative infinity."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & FRAC_MASK_F16) == 0)


def is_subnormal_f16(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormal) half-precision float."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & FRAC_MASK_F16) != 0)


def is_zero_f16(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK_F16) == 0


def apply_f16_modifiers(bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply VOP3P input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & 0xFFFF
    if abs_mod:
        res &= ~SIGN_MASK_F16
    if neg_mod:
        res ^= SIGN_MASK_F16
    return res


def oracle_add_f16(
    s0_bits: int,
    s1_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for scalar IEEE 754 half-precision addition with clamp."""
    s0 = apply_f16_modifiers(s0_bits, abs_mod=abs0, neg_mod=neg0)
    s1 = apply_f16_modifiers(s1_bits, abs_mod=abs1, neg_mod=neg1)

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
        s0_sign = (s0 & SIGN_MASK_F16)
        s1_sign = (s1 & SIGN_MASK_F16)
        if s0_sign != s1_sign:
            # (+inf) + (-inf) -> Invalid operation -> Canonical Quiet NaN
            return POS_ZERO_F16 if clamp else CANONICAL_QNAN_F16
        if clamp:
            return ONE_F16 if s0 == POS_INF_F16 else POS_ZERO_F16
        return s0
    if s0_inf:
        if clamp:
            return ONE_F16 if s0 == POS_INF_F16 else POS_ZERO_F16
        return s0
    if s1_inf:
        if clamp:
            return ONE_F16 if s1 == POS_INF_F16 else POS_ZERO_F16
        return s1

    # 3. Signed zero handling
    if is_zero_f16(s0) and is_zero_f16(s1):
        s0_is_neg = (s0 == NEG_ZERO_F16)
        s1_is_neg = (s1 == NEG_ZERO_F16)
        if s0_is_neg and s1_is_neg:
            res = NEG_ZERO_F16
        else:
            res = POS_ZERO_F16
        if clamp and res == NEG_ZERO_F16:
            return POS_ZERO_F16
        return res

    # 4. Finite addition
    f0 = bits_to_f16(s0)
    f1 = bits_to_f16(s1)
    f_res = f0 + f1
    res = f16_to_bits(f_res)

    # In round-to-nearest-even, x + (-x) produces +0.0
    if f_res == 0.0 and not (s0 == NEG_ZERO_F16 and s1 == NEG_ZERO_F16):
        res = POS_ZERO_F16

    # 5. Clamp modifier ([0.0, 1.0])
    if clamp:
        if f_res < 0.0 or res == NEG_ZERO_F16:
            res = POS_ZERO_F16
        elif f_res > 1.0 or res == POS_INF_F16:
            res = ONE_F16

    return res


def oracle_v_pk_add_f16(
    src0_dword: int,
    src1_dword: int,
    op_sel: Tuple[int, int] = (0, 0),
    op_sel_hi: Tuple[int, int] = (1, 1),
    neg_lo: Tuple[bool, bool] = (False, False),
    neg_hi: Tuple[bool, bool] = (False, False),
    abs_lo: Tuple[bool, bool] = (False, False),
    abs_hi: Tuple[bool, bool] = (False, False),
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_pk_add_f16 packed addition.
    
    Computes packed 2xf16 addition:
    lo = src0[op_sel[0]] + src1[op_sel[1]]
    hi = src0[op_sel_hi[0]] + src1[op_sel_hi[1]]
    applying neg, abs, and clamp modifiers.
    """
    s0_halves = unpack_2xf16(src0_dword)
    s1_halves = unpack_2xf16(src1_dword)

    s0_lo_in = s0_halves[op_sel[0]]
    s1_lo_in = s1_halves[op_sel[1]]
    res_lo = oracle_add_f16(
        s0_lo_in,
        s1_lo_in,
        abs0=abs_lo[0],
        neg0=neg_lo[0],
        abs1=abs_lo[1],
        neg1=neg_lo[1],
        clamp=clamp,
    )

    s0_hi_in = s0_halves[op_sel_hi[0]]
    s1_hi_in = s1_halves[op_sel_hi[1]]
    res_hi = oracle_add_f16(
        s0_hi_in,
        s1_hi_in,
        abs0=abs_hi[0],
        neg0=neg_hi[0],
        abs1=abs_hi[1],
        neg1=neg_hi[1],
        clamp=clamp,
    )

    return pack_2xf16(res_lo, res_hi)


# ==============================================================================
# Unit Tests
# ==============================================================================

class TestF16ModelAndOracle(unittest.TestCase):
    """Unit tests for binary16 / packed 2xf16 bit model and reference oracle."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 65504.0, -65504.0]
        for val in values:
            with self.subTest(val=val):
                bits = f16_to_bits(val)
                recovered = bits_to_f16(bits)
                self.assertEqual(val, recovered)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_dword_packing_and_unpacking(self) -> None:
        lo = f16_to_bits(1.5)
        hi = f16_to_bits(-3.0)
        packed = pack_2xf16(lo, hi)
        unpacked_lo, unpacked_hi = unpack_2xf16(packed)
        self.assertEqual(unpacked_lo, lo)
        self.assertEqual(unpacked_hi, hi)
        self.assertEqual(bits_to_f16(unpacked_lo), 1.5)
        self.assertEqual(bits_to_f16(unpacked_hi), -3.0)

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

        subnorm_bits = 0x0001
        self.assertTrue(is_subnormal_f16(subnorm_bits))
        self.assertFalse(is_subnormal_f16(POS_ZERO_F16))
        self.assertFalse(is_subnormal_f16(f16_to_bits(1.0)))

    def test_exact_finite_addition(self) -> None:
        # lo: 1.5 + 2.5 = 4.0; hi: 0.25 + 0.75 = 1.0
        s0 = pack_2xf16(f16_to_bits(1.5), f16_to_bits(0.25))
        s1 = pack_2xf16(f16_to_bits(2.5), f16_to_bits(0.75))
        res = oracle_v_pk_add_f16(s0, s1)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(bits_to_f16(r_lo), 4.0)
        self.assertEqual(bits_to_f16(r_hi), 1.0)

    def test_signed_zero_rules(self) -> None:
        self.assertEqual(oracle_add_f16(POS_ZERO_F16, POS_ZERO_F16), POS_ZERO_F16)
        self.assertEqual(oracle_add_f16(NEG_ZERO_F16, NEG_ZERO_F16), NEG_ZERO_F16)
        self.assertEqual(oracle_add_f16(POS_ZERO_F16, NEG_ZERO_F16), POS_ZERO_F16)
        self.assertEqual(oracle_add_f16(NEG_ZERO_F16, POS_ZERO_F16), POS_ZERO_F16)
        self.assertEqual(oracle_add_f16(f16_to_bits(5.0), f16_to_bits(-5.0)), POS_ZERO_F16)

    def test_input_modifiers_abs_neg(self) -> None:
        val_pos = f16_to_bits(4.0)
        val_neg = f16_to_bits(-4.0)
        s0 = pack_2xf16(val_neg, val_pos)
        s1 = pack_2xf16(f16_to_bits(2.0), f16_to_bits(2.0))

        # lo: |-4.0| + 2.0 = 6.0; hi: -(4.0) + 2.0 = -2.0
        res = oracle_v_pk_add_f16(
            s0, s1,
            abs_lo=(True, False),
            neg_hi=(True, False),
        )
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(bits_to_f16(r_lo), 6.0)
        self.assertEqual(bits_to_f16(r_hi), -2.0)

    def test_op_sel_cross_and_broadcast(self) -> None:
        # s0 = [lo: 10.0, hi: 20.0], s1 = [lo: 1.0, hi: 2.0]
        s0 = pack_2xf16(f16_to_bits(10.0), f16_to_bits(20.0))
        s1 = pack_2xf16(f16_to_bits(1.0), f16_to_bits(2.0))

        # Cross add: lo uses s0.hi + s1.lo (20 + 1 = 21), hi uses s0.lo + s1.hi (10 + 2 = 12)
        res_cross = oracle_v_pk_add_f16(s0, s1, op_sel=(1, 0), op_sel_hi=(0, 1))
        r_lo, r_hi = unpack_2xf16(res_cross)
        self.assertEqual(bits_to_f16(r_lo), 21.0)
        self.assertEqual(bits_to_f16(r_hi), 12.0)

        # Broadcast hi: lo and hi both use s0.hi + s1.hi (20 + 2 = 22)
        res_bcast = oracle_v_pk_add_f16(s0, s1, op_sel=(1, 1), op_sel_hi=(1, 1))
        b_lo, b_hi = unpack_2xf16(res_bcast)
        self.assertEqual(bits_to_f16(b_lo), 22.0)
        self.assertEqual(bits_to_f16(b_hi), 22.0)

    def test_clamp_modifier(self) -> None:
        # lo: 0.5 + 0.8 = 1.3 -> clamped to 1.0
        # hi: -1.0 + 0.4 = -0.6 -> clamped to 0.0
        s0 = pack_2xf16(f16_to_bits(0.5), f16_to_bits(-1.0))
        s1 = pack_2xf16(f16_to_bits(0.8), f16_to_bits(0.4))
        res = oracle_v_pk_add_f16(s0, s1, clamp=True)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(r_lo, ONE_F16)
        self.assertEqual(r_hi, POS_ZERO_F16)

    def test_infinity_and_nan_addition(self) -> None:
        self.assertEqual(oracle_add_f16(POS_INF_F16, f16_to_bits(1.0)), POS_INF_F16)
        self.assertEqual(oracle_add_f16(NEG_INF_F16, f16_to_bits(1.0)), NEG_INF_F16)
        self.assertEqual(oracle_add_f16(POS_INF_F16, NEG_INF_F16), CANONICAL_QNAN_F16)

        snan = 0x7C05
        res_nan = oracle_add_f16(snan, f16_to_bits(1.0))
        self.assertTrue(is_qnan_f16(res_nan))
        self.assertEqual(res_nan, snan | QNAN_BIT_F16)

        # Clamping NaN yields +0.0
        res_clamped_nan = oracle_add_f16(snan, f16_to_bits(1.0), clamp=True)
        self.assertEqual(res_clamped_nan, POS_ZERO_F16)


if __name__ == "__main__":
    unittest.main()
