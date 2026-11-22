"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_pk_fma_clamp_f16.

Architecture: AMD RDNA2 (GFX10 / GFX1030) VALU VOP3P.
Instruction:
    v_pk_fma_clamp_f16 dst, src0, src1, src2 [op_sel] [op_sel_hi] [neg_lo] [neg_hi]

Packed execution:
    dst.lo = clamp01(src0_sel_lo * src1_sel_lo + src2_sel_lo)
    dst.hi = clamp01(src0_sel_hi * src1_sel_hi + src2_sel_hi)

Clamp semantics in RDNA2 VALU:
    - Saturated output range is [0.0, 1.0].
    - Underflow / negative results (-0.0 or < 0.0) saturate to +0.0 (0x0000).
    - Overflow / results > 1.0 saturate to +1.0 (0x3C00).
    - Any NaN result (including invalid operations like 0 * inf or (+inf + -inf))
      saturates to +0.0 (0x0000).

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import enum
from fractions import Fraction
import math
import struct
from typing import List, Optional, Tuple, Union
import unittest

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF

# IEEE 754 binary16 bit constants
F16_POS_ZERO: int = 0x0000
F16_NEG_ZERO: int = 0x8000
F16_POS_ONE: int = 0x3C00
F16_NEG_ONE: int = 0xBC00
F16_POS_HALF: int = 0x3800
F16_NEG_HALF: int = 0xB800
F16_POS_TWO: int = 0x4000
F16_NEG_TWO: int = 0xC000
F16_POS_INF: int = 0x7C00
F16_NEG_INF: int = 0xFC00
F16_CANONICAL_QNAN: int = 0x7E00
F16_QUIET_BIT: int = 0x0200

F16_EXP_MASK: int = 0x7C00
F16_MANT_MASK: int = 0x03FF
F16_SIGN_MASK: int = 0x8000

F16_MIN_NORMAL: int = 0x0400    # 2^-14
F16_MAX_NORMAL: int = 0x7BFF    # 65504.0
F16_MIN_SUBNORMAL: int = 0x0001 # 2^-24
F16_MAX_SUBNORMAL: int = 0x03FF # 2^-14 * (1023/1024)

OLD_VGPR_SENTINEL: int = 0xDEADBEEF


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit range."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range."""
    return val & MASK32


def pack_f16x2(lo: int, hi: int) -> int:
    """Pack two 16-bit half words into a single 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def unpack_f16x2(packed: int) -> Tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit half words (lo, hi)."""
    return packed & MASK16, (packed >> 16) & MASK16


def f16_to_u16(val: float) -> int:
    """Convert float to IEEE 754 binary16 bits."""
    if math.isnan(val):
        return F16_CANONICAL_QNAN
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return F16_POS_INF if val > 0 else F16_NEG_INF


def u16_to_f16(bits: int) -> float:
    """Convert IEEE 754 binary16 bits to float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def is_f16_nan(bits: int) -> bool:
    """Return True if 16-bit pattern is an IEEE 754 NaN."""
    return ((bits & F16_EXP_MASK) == F16_EXP_MASK) and ((bits & F16_MANT_MASK) != 0)


def is_f16_snan(bits: int) -> bool:
    """Return True if 16-bit pattern is a signaling NaN."""
    return is_f16_nan(bits) and ((bits & F16_QUIET_BIT) == 0)


def is_f16_qnan(bits: int) -> bool:
    """Return True if 16-bit pattern is a quiet NaN."""
    return is_f16_nan(bits) and ((bits & F16_QUIET_BIT) != 0)


def is_f16_zero(bits: int) -> bool:
    """Return True if 16-bit pattern is +0.0 or -0.0."""
    return (bits & (F16_EXP_MASK | F16_MANT_MASK)) == 0


def is_f16_inf(bits: int) -> bool:
    """Return True if 16-bit pattern is +inf or -inf."""
    return ((bits & F16_EXP_MASK) == F16_EXP_MASK) and ((bits & F16_MANT_MASK) == 0)


def is_f16_subnormal(bits: int) -> bool:
    """Return True if 16-bit pattern is a subnormal float."""
    return ((bits & F16_EXP_MASK) == 0) and ((bits & F16_MANT_MASK) != 0)


def f16_sign(bits: int) -> int:
    """Return the sign bit of a 16-bit float (0 or 1)."""
    return (bits >> 15) & 1


def clamp01_f16(bits: int) -> int:
    """Apply RDNA2 VALU clamp [0.0, 1.0] saturation to binary16 bits."""
    if is_f16_nan(bits):
        return F16_POS_ZERO
    val = u16_to_f16(bits)
    if val < 0.0 or bits == F16_NEG_ZERO:
        return F16_POS_ZERO
    if val > 1.0 or bits == F16_POS_INF:
        return F16_POS_ONE
    return bits


def ref_fma_clamp_f16(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for scalar clamped fused multiply-add: clamp01(s0 * s1 + s2).

    Computes fused multiplication and addition conforming to IEEE 754-2008
    with infinite intermediate precision before rounding and RDNA2 [0.0, 1.0] clamping.
    """
    s0 &= MASK16
    s1 &= MASK16
    s2 &= MASK16

    # 1. NaN handling: Any NaN input clamped produces +0.0 in RDNA2 VALU
    if is_f16_nan(s0) or is_f16_nan(s1) or is_f16_nan(s2):
        return F16_POS_ZERO

    # 2. Check for invalid operation during multiply (0 * inf)
    s0_inf = is_f16_inf(s0)
    s1_inf = is_f16_inf(s1)
    s0_zero = is_f16_zero(s0)
    s1_zero = is_f16_zero(s1)

    if (s0_inf and s1_zero) or (s1_inf and s0_zero):
        return F16_POS_ZERO

    # 3. Check for invalid operation during addition (+inf + -inf)
    prod_sign = f16_sign(s0) ^ f16_sign(s1)
    prod_is_inf = s0_inf or s1_inf
    s2_inf = is_f16_inf(s2)

    if prod_is_inf and s2_inf:
        if prod_sign != f16_sign(s2):
            return F16_POS_ZERO

    # 4. Handle infinities that survive
    if prod_is_inf:
        # If product is +inf, result is +inf -> clamped to 1.0
        # If product is -inf, result is -inf -> clamped to +0.0
        return F16_POS_ONE if prod_sign == 0 else F16_POS_ZERO
    if s2_inf:
        return F16_POS_ONE if f16_sign(s2) == 0 else F16_POS_ZERO

    # 5. Exact arithmetic with Fraction for finite numbers
    f0 = Fraction(u16_to_f16(s0))
    f1 = Fraction(u16_to_f16(s1))
    f2 = Fraction(u16_to_f16(s2))
    exact_res = f0 * f1 + f2

    # Float conversion and IEEE 754 binary16 rounding
    raw_res_bits = f16_to_u16(float(exact_res))

    # Apply clamp
    return clamp01_f16(raw_res_bits)


def ref_v_pk_fma_clamp_f16(
    s0_dword: int,
    s1_dword: int,
    s2_dword: int,
    *,
    op_sel: Tuple[int, int, int] = (0, 0, 0),
    op_sel_hi: Tuple[int, int, int] = (1, 1, 1),
    neg_lo: Tuple[int, int, int] = (0, 0, 0),
    neg_hi: Tuple[int, int, int] = (0, 0, 0),
) -> int:
    """Reference mathematical oracle for packed dual-lane v_pk_fma_clamp_f16."""
    s0_lo_orig, s0_hi_orig = unpack_f16x2(s0_dword)
    s1_lo_orig, s1_hi_orig = unpack_f16x2(s1_dword)
    s2_lo_orig, s2_hi_orig = unpack_f16x2(s2_dword)

    # op_sel selects low calculation inputs
    s0_lo = s0_hi_orig if op_sel[0] else s0_lo_orig
    s1_lo = s1_hi_orig if op_sel[1] else s1_lo_orig
    s2_lo = s2_hi_orig if op_sel[2] else s2_lo_orig

    # op_sel_hi selects high calculation inputs
    s0_hi = s0_hi_orig if op_sel_hi[0] else s0_lo_orig
    s1_hi = s1_hi_orig if op_sel_hi[1] else s1_lo_orig
    s2_hi = s2_hi_orig if op_sel_hi[2] else s2_lo_orig

    # Apply neg_lo modifiers
    if neg_lo[0]:
        s0_lo ^= F16_SIGN_MASK
    if neg_lo[1]:
        s1_lo ^= F16_SIGN_MASK
    if neg_lo[2]:
        s2_lo ^= F16_SIGN_MASK

    # Apply neg_hi modifiers
    if neg_hi[0]:
        s0_hi ^= F16_SIGN_MASK
    if neg_hi[1]:
        s1_hi ^= F16_SIGN_MASK
    if neg_hi[2]:
        s2_hi ^= F16_SIGN_MASK

    res_lo = ref_fma_clamp_f16(s0_lo, s1_lo, s2_lo)
    res_hi = ref_fma_clamp_f16(s0_hi, s1_hi, s2_hi)

    return pack_f16x2(res_lo, res_hi)


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_basic_normal_fma_clamped(self) -> None:
        # lo: 0.5 * 0.5 + 0.25 = 0.50 (in range [0, 1])
        # hi: 0.25 * 2.0 + 0.125 = 0.625 (in range [0, 1])
        s0 = pack_f16x2(f16_to_u16(0.5), f16_to_u16(0.25))
        s1 = pack_f16x2(f16_to_u16(0.5), f16_to_u16(2.0))
        s2 = pack_f16x2(f16_to_u16(0.25), f16_to_u16(0.125))

        res = ref_v_pk_fma_clamp_f16(s0, s1, s2)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 0.50)
        self.assertAlmostEqual(u16_to_f16(res_hi), 0.625)

    def test_clamping_upper_and_lower_bounds(self) -> None:
        # lo: 2.0 * 2.0 + 1.0 = 5.0 -> clamp -> 1.0
        # hi: 2.0 * -1.0 + 0.5 = -1.5 -> clamp -> +0.0
        s0 = pack_f16x2(f16_to_u16(2.0), f16_to_u16(2.0))
        s1 = pack_f16x2(f16_to_u16(2.0), f16_to_u16(-1.0))
        s2 = pack_f16x2(f16_to_u16(1.0), f16_to_u16(0.5))

        res = ref_v_pk_fma_clamp_f16(s0, s1, s2)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertEqual(res_lo, F16_POS_ONE)
        self.assertEqual(res_hi, F16_POS_ZERO)

    def test_nan_clamping_produces_zero(self) -> None:
        # Clamped NaN in RDNA2 produces +0.0
        s0 = pack_f16x2(F16_CANONICAL_QNAN, f16_to_u16(1.0))
        s1 = pack_f16x2(f16_to_u16(1.0), F16_CANONICAL_QNAN)
        s2 = pack_f16x2(f16_to_u16(0.0), f16_to_u16(0.0))

        res = ref_v_pk_fma_clamp_f16(s0, s1, s2)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertEqual(res_lo, F16_POS_ZERO)
        self.assertEqual(res_hi, F16_POS_ZERO)

    def test_op_sel_and_negation_modifiers(self) -> None:
        # s0: [0.5, 2.0], s1: [0.5, 0.25], s2: [0.1, 0.2]
        s0 = pack_f16x2(f16_to_u16(0.5), f16_to_u16(2.0))
        s1 = pack_f16x2(f16_to_u16(0.5), f16_to_u16(0.25))
        s2 = pack_f16x2(f16_to_u16(0.1), f16_to_u16(0.2))

        # op_sel:(1, 1, 0) selects (s0.hi, s1.hi, s2.lo) -> 2.0 * 0.25 + 0.1 = 0.60
        # neg_lo:(1, 0, 0) negates s0 -> -2.0 * 0.25 + 0.1 = -0.40 -> clamp -> +0.0
        res = ref_v_pk_fma_clamp_f16(
            s0, s1, s2,
            op_sel=(1, 1, 0),
            neg_lo=(1, 0, 0),
        )
        res_lo, _ = unpack_f16x2(res)
        self.assertEqual(res_lo, F16_POS_ZERO)


if __name__ == "__main__":
    unittest.main()
