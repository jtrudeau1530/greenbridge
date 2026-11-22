from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_fma_f16.

Tests the half-precision fused multiply-add instruction:
    v_fma_f16 D, S0, S1, S2
Lowered semantics:
    D = (S0 * S1) + S2
with single-rounding precision per IEEE 754-2008 (binary16).
Supports input modifiers:
    - abs (|src|: clear sign bit)
    - neg (-src: invert sign bit)
Supports output modifiers:
    - clamp: saturates result to [0.0, 1.0], with NaN clamped to +0.0
    - omod: mul:2, mul:4, div:2
Architectural model:
    16-bit halfword registers mapped onto 32-bit VGPR/SGPR halves:
    [lo: bits 15..0, hi: bits 31..16].
"""

from enum import Enum, auto
from fractions import Fraction
import math
import struct
import sys
import unittest

# IEEE 754 binary16 bitwise architectural constants
F16_POS_ZERO = 0x0000
F16_NEG_ZERO = 0x8000
F16_ONE = 0x3C00
F16_NEG_ONE = 0xBC00
F16_TWO = 0x4000
F16_HALF = 0x3800
F16_POS_INF = 0x7C00
F16_NEG_INF = 0xFC00
F16_CANONICAL_QNAN = 0x7E00
F16_SNAN = 0x7C01
F16_MIN_SUBNORMAL = 0x0001       # 2^-24 ~= 5.96e-8
F16_MAX_SUBNORMAL = 0x03FF       # (1023/1024) * 2^-14
F16_MIN_NORMAL = 0x0400          # 2^-14 ~= 6.10e-5
F16_MAX_NORMAL = 0x7BFF          # 65504.0

SIGN_MASK_F16 = 0x8000
EXP_MASK_F16 = 0x7C00
MANT_MASK_F16 = 0x03FF
QNAN_BIT_F16 = 0x0200

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF


def f16_to_u16(val: float) -> int:
    """Pack Python float into unsigned 16-bit integer (IEEE 754 binary16)."""
    try:
        return struct.unpack("<H", struct.pack("<e", float(val)))[0]
    except OverflowError:
        return F16_POS_INF if val > 0 else F16_NEG_INF


def u16_to_f16(bits: int) -> float:
    """Unpack unsigned 16-bit integer into Python float (IEEE 754 binary16)."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def split_f16(bits: int) -> tuple[int, int, int]:
    """Split 16-bit word into (sign, 5-bit exponent, 10-bit mantissa)."""
    b = bits & MASK16
    return (b >> 15) & 1, (b >> 10) & 0x1F, b & MANT_MASK_F16


def is_nan_f16(bits: int) -> bool:
    """Check if 16-bit word is an IEEE 754 binary16 NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) != 0)


def is_snan_f16(bits: int) -> bool:
    """Check if 16-bit word is an IEEE 754 binary16 signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_qnan_f16(bits: int) -> bool:
    """Check if 16-bit word is an IEEE 754 binary16 quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """Check if 16-bit word is an IEEE 754 binary16 infinity (+/-)."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) == 0)


def is_zero_f16(bits: int) -> bool:
    """Check if 16-bit word is zero (+0.0 or -0.0)."""
    return (bits & 0x7FFF) == 0


def is_subnormal_f16(bits: int) -> bool:
    """Check if 16-bit word is an IEEE 754 binary16 subnormal (denormal)."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & MANT_MASK_F16) != 0)


def apply_src_mods_u16(bits: int, *, neg: bool = False, abs_: bool = False) -> int:
    """Apply bit-exact RDNA2 source modifiers (|src|, -src) to a 16-bit float bit pattern."""
    val = bits & MASK16
    if abs_:
        val &= ~SIGN_MASK_F16  # clear sign bit
    if neg:
        val ^= SIGN_MASK_F16   # invert sign bit
    return val


def apply_clamp_f16(bits: int) -> int:
    """Apply RDNA2 clamp modifier [0.0, 1.0] to float16, mapping NaNs to +0.0."""
    if is_nan_f16(bits):
        return F16_POS_ZERO
    val_f = u16_to_f16(bits)
    if val_f < 0.0:
        return F16_POS_ZERO
    if val_f > 1.0:
        return F16_ONE
    return bits & MASK16


def apply_omod_f16(bits: int, omod: int) -> int:
    """Apply RDNA2 output modifier: 0: none, 1: mul:2, 2: mul:4, 3: div:2."""
    if omod == 0:
        return bits & MASK16
    if is_nan_f16(bits):
        return bits & MASK16
    val = u16_to_f16(bits)
    if omod == 1:
        val *= 2.0
    elif omod == 2:
        val *= 4.0
    elif omod == 3:
        val *= 0.5
    else:
        raise ValueError(f"Invalid omod value: {omod}")
    return f16_to_u16(val)


def raw_fma_f16(u0: int, u1: int, u2: int) -> int:
    """Compute exact IEEE 754 half precision FMA (u0 * u1) + u2 with single rounding."""
    u0 &= MASK16
    u1 &= MASK16
    u2 &= MASK16

    # 1. NaN propagation: if any input is NaN, quiet and return
    for u in (u0, u1, u2):
        if is_snan_f16(u):
            return u | QNAN_BIT_F16
    for u in (u0, u1, u2):
        if is_qnan_f16(u):
            return u

    # 2. Invalid operations: Inf * 0 or 0 * Inf
    if (is_inf_f16(u0) and is_zero_f16(u1)) or (is_zero_f16(u0) and is_inf_f16(u1)):
        return F16_CANONICAL_QNAN

    # 3. Product is infinite
    prod_inf = is_inf_f16(u0) or is_inf_f16(u1)
    if prod_inf:
        prod_sign = ((u0 >> 15) ^ (u1 >> 15)) & 1
        if is_inf_f16(u2):
            u2_sign = (u2 >> 15) & 1
            if prod_sign != u2_sign:
                return F16_CANONICAL_QNAN
        return F16_NEG_INF if prod_sign else F16_POS_INF

    # 4. S2 is infinite
    if is_inf_f16(u2):
        return u2

    # 5. Product is zero
    if is_zero_f16(u0) or is_zero_f16(u1):
        prod_sign = ((u0 >> 15) ^ (u1 >> 15)) & 1
        if is_zero_f16(u2):
            u2_sign = (u2 >> 15) & 1
            if prod_sign == 1 and u2_sign == 1:
                return F16_NEG_ZERO
            return F16_POS_ZERO
        return u2

    # 6. Finite values: compute exact rational arithmetic with Fraction
    f0 = u16_to_f16(u0)
    f1 = u16_to_f16(u1)
    f2 = u16_to_f16(u2)
    frac_sum = Fraction(f0) * Fraction(f1) + Fraction(f2)

    if frac_sum == 0:
        return F16_POS_ZERO

    try:
        val_f = float(frac_sum)
        return f16_to_u16(val_f)
    except OverflowError:
        return F16_POS_INF if frac_sum > 0 else F16_NEG_INF


def ref_fma_f16(
    s0: float,
    s1: float,
    s2: float,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    abs0: bool = False,
    abs1: bool = False,
    abs2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> float:
    """Reference mathematical oracle for v_fma_f16 calculating (s0 * s1) + s2 in binary16."""
    u0 = apply_src_mods_u16(f16_to_u16(s0), neg=neg0, abs_=abs0)
    u1 = apply_src_mods_u16(f16_to_u16(s1), neg=neg1, abs_=abs1)
    u2 = apply_src_mods_u16(f16_to_u16(s2), neg=neg2, abs_=abs2)

    res_u16 = raw_fma_f16(u0, u1, u2)

    if omod != 0:
        res_u16 = apply_omod_f16(res_u16, omod)
    if clamp:
        res_u16 = apply_clamp_f16(res_u16)

    return u16_to_f16(res_u16)


def ref_fma_f16_u16(
    u0: int,
    u1: int,
    u2: int,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    abs0: bool = False,
    abs1: bool = False,
    abs2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> int:
    """Bit-exact reference mathematical oracle for v_fma_f16 using 16-bit integer words."""
    m0 = apply_src_mods_u16(u0, neg=neg0, abs_=abs0)
    m1 = apply_src_mods_u16(u1, neg=neg1, abs_=abs1)
    m2 = apply_src_mods_u16(u2, neg=neg2, abs_=abs2)

    res_u16 = raw_fma_f16(m0, m1, m2)

    if omod != 0:
        res_u16 = apply_omod_f16(res_u16, omod)
    if clamp:
        res_u16 = apply_clamp_f16(res_u16)

    return res_u16 & MASK16


class RegisterHalf:
    """Architectural model of a 16-bit halfword within a 32-bit register (lo or hi)."""

    def __init__(self, raw_u16: int = 0) -> None:
        self.raw = raw_u16 & MASK16

    @classmethod
    def from_f16(cls, val: float) -> RegisterHalf:
        return cls(f16_to_u16(val))

    @property
    def as_u16(self) -> int:
        return self.raw

    @property
    def as_f16(self) -> float:
        return u16_to_f16(self.raw)

    def set_u16(self, bits: int) -> None:
        self.raw = bits & MASK16

    def set_f16(self, val: float) -> None:
        self.set_u16(f16_to_u16(val))

    def __repr__(self) -> str:
        return f"RegisterHalf({self.raw:#06x}, f16={self.as_f16})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegisterHalf):
            return NotImplemented
        return self.raw == other.raw


class VGPRWord:
    """Architectural model of a 32-bit VGPR containing low and high 16-bit halfwords."""

    def __init__(self, dword: int = 0) -> None:
        self.lo = RegisterHalf(dword & MASK16)
        self.hi = RegisterHalf((dword >> 16) & MASK16)

    @property
    def dword(self) -> int:
        return (self.lo.as_u16) | ((self.hi.as_u16) << 16)

    @dword.setter
    def dword(self, val: int) -> None:
        self.lo.set_u16(val & MASK16)
        self.hi.set_u16((val >> 16) & MASK16)

    def __repr__(self) -> str:
        return f"VGPRWord(lo={self.lo.as_u16:#06x}, hi={self.hi.as_u16:#06x})"


class TestFmaF16MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural register models."""

    def test_basic_fma_identities(self):
        # 1.0 * 1.0 + 0.0 == 1.0
        self.assertEqual(ref_fma_f16(1.0, 1.0, 0.0), 1.0)
        # 2.0 * 3.0 + 4.0 == 10.0
        self.assertEqual(ref_fma_f16(2.0, 3.0, 4.0), 10.0)
        # -2.5 * 4.0 + 1.25 == -8.75
        self.assertEqual(ref_fma_f16(-2.5, 4.0, 1.25), -8.75)
        # 0.0 * 100.0 + 5.5 == 5.5
        self.assertEqual(ref_fma_f16(0.0, 100.0, 5.5), 5.5)

    def test_fused_precision_single_rounding(self):
        # FMA retains intermediate product precision without intermediate 16-bit rounding.
        # Let a = 1.0 + 2**-10 (0x3C01), b = 1.0 + 2**-10 (0x3C01), c = -(1.0 + 2**-9) (0xBC02)
        # (a * b) = 1.0 + 2**-9 + 2**-20
        # Unfused: (a * b) rounds to (1.0 + 2**-9) in f16, so (a * b) + c == 0.0
        # Fused: (a * b) + c retains 2**-20, which in f16 is subnormal 0x0010 != 0.0
        a = u16_to_f16(0x3C01)
        b = u16_to_f16(0x3C01)
        c = u16_to_f16(0xBC02)
        unfused = u16_to_f16(f16_to_u16(a * b)) + c
        fused = ref_fma_f16(a, b, c)
        self.assertEqual(unfused, 0.0)
        self.assertNotEqual(fused, 0.0)
        self.assertEqual(f16_to_u16(fused), 0x0010)

    def test_source_modifiers(self):
        # abs(|src|): clears sign bit
        self.assertEqual(ref_fma_f16(-3.5, 2.0, 1.0, abs0=True), 8.0)
        self.assertEqual(ref_fma_f16(3.5, -2.0, 1.0, abs1=True), 8.0)
        self.assertEqual(ref_fma_f16(3.5, 2.0, -1.0, abs2=True), 8.0)

        # neg(-src): inverts sign bit
        self.assertEqual(ref_fma_f16(3.5, 2.0, 1.0, neg0=True), -6.0)
        self.assertEqual(ref_fma_f16(3.5, 2.0, 1.0, neg1=True), -6.0)
        self.assertEqual(ref_fma_f16(3.5, 2.0, 1.0, neg2=True), 6.0)

        # combined neg and abs: -|src|
        self.assertEqual(ref_fma_f16(-3.5, 2.0, 1.0, neg0=True, abs0=True), -6.0)

    def test_output_modifiers_clamp_and_omod(self):
        # clamp saturates to [0.0, 1.0]
        self.assertEqual(ref_fma_f16(2.0, 3.0, 0.0, clamp=True), 1.0)
        self.assertEqual(ref_fma_f16(-2.0, 3.0, 0.0, clamp=True), 0.0)
        self.assertEqual(ref_fma_f16(0.5, 1.0, 0.25, clamp=True), 0.75)

        # omod: mul:2, mul:4, div:2
        self.assertEqual(ref_fma_f16(1.5, 2.0, 0.5, omod=1), 7.0)   # (3.5) * 2 = 7.0
        self.assertEqual(ref_fma_f16(1.5, 2.0, 0.5, omod=2), 14.0)  # (3.5) * 4 = 14.0
        self.assertEqual(ref_fma_f16(1.5, 2.0, 0.5, omod=3), 1.75)  # (3.5) * 0.5 = 1.75

        # omod combined with clamp: (omod applies before clamp)
        self.assertEqual(ref_fma_f16(0.125, 2.0, 0.0, omod=1, clamp=True), 0.5)
        self.assertEqual(ref_fma_f16(0.25, 2.0, 0.0, omod=1, clamp=True), 1.0)
        self.assertEqual(ref_fma_f16(0.5, 2.0, 0.0, omod=2, clamp=True), 1.0)

    def test_register_half_architecture(self):
        rh = RegisterHalf.from_f16(1.0)
        self.assertEqual(rh.as_f16, 1.0)
        self.assertEqual(rh.as_u16, F16_ONE)

        word = VGPRWord(0)
        word.lo.set_f16(1.0)
        word.hi.set_f16(2.0)
        self.assertEqual(word.lo.as_u16, F16_ONE)
        self.assertEqual(word.hi.as_u16, F16_TWO)
        self.assertEqual(word.dword, (F16_TWO << 16) | F16_ONE)

        # bit-exact oracle roundtrip
        u_res = ref_fma_f16_u16(F16_TWO, F16_HALF, F16_ONE)
        self.assertEqual(u_res, f16_to_u16(2.0))


if __name__ == "__main__":
    unittest.main()
