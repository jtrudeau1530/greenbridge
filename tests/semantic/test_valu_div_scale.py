from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_div_scale.

Tests the floating-point division pre-scaling instructions:
    v_div_scale_f32 D, sdst, S0, S1, S2
    v_div_scale_f64 D, sdst, S0, S1, S2

Lowered semantics (VOP3b dual-output):
    D = Scaled version of S0 (scaled by 2^k or 2^-k if required to prevent underflow/overflow)
    sdst (vcc) = 1 if scaling was applied, 0 if no scaling occurred
Where:
    S0 = operand to scale (typically denominator S1 or numerator S2)
    S1 = denominator
    S2 = numerator
For f32: scale factor is 2^64 or 2^-64.
For f64: scale factor is 2^128 or 2^-128.
Supports input modifiers:
    - abs (|src|: clear sign bit)
    - neg (-src: invert sign bit)
Supports output modifiers:
    - clamp: saturates result to [0.0, 1.0], with NaN clamped to +0.0
    - omod: mul:2, mul:4, div:2
Architectural model:
    f32: 32-bit registers (VGPR / SGPR)
    f64: 64-bit register pairs mapped onto aligned VGPR/SGPR pairs [lo: 32-bit dword, hi: 32-bit dword].
"""

from enum import Enum, auto
from fractions import Fraction
import math
import struct
import sys
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# IEEE 754 single precision constants
F32_POS_ZERO = 0x00000000
F32_NEG_ZERO = 0x80000000
F32_ONE = 0x3F800000
F32_NEG_ONE = 0xBF800000
F32_TWO = 0x40000000
F32_HALF = 0x3F000000
F32_POS_INF = 0x7F800000
F32_NEG_INF = 0xFF800000
F32_CANONICAL_QNAN = 0x7FC00000
F32_SNAN = 0x7F800001
F32_SUB_MIN = 0x00000001
F32_SUB_MAX = 0x007FFFFF
F32_MIN_NORMAL = 0x00800000
F32_MAX_NORMAL = 0x7F7FFFFF

# IEEE 754 double precision constants
F64_POS_ZERO = 0x0000000000000000
F64_NEG_ZERO = 0x8000000000000000
F64_ONE = 0x3FF0000000000000
F64_NEG_ONE = 0xBFF0000000000000
F64_TWO = 0x4000000000000000
F64_HALF = 0x3FE0000000000000
F64_POS_INF = 0x7FF0000000000000
F64_NEG_INF = 0xFFF0000000000000
F64_CANONICAL_QNAN = 0x7FF8000000000000
F64_SNAN = 0x7FF0000000000001
F64_SUB_MIN = 0x0000000000000001
F64_SUB_MAX = 0x000FFFFFFFFFFFFF
F64_MIN_NORMAL = 0x0010000000000000
F64_MAX_NORMAL = 0x7FEFFFFFFFFFFFFF

# Scaling constants
SCALE64_POS = 2.0 ** 64
SCALE64_NEG = 2.0 ** -64
SCALE128_POS = 2.0 ** 128
SCALE128_NEG = 2.0 ** -128

THRESH_F32_SMALL = 2.0 ** -88
THRESH_F32_LARGE = 2.0 ** 62
THRESH_F64_SMALL = 2.0 ** -768
THRESH_F64_LARGE = 2.0 ** 768


def f32_to_u32(val: float) -> int:
    """Pack Python float into unsigned 32-bit integer (IEEE 754 binary32)."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def u32_to_f32(bits: int) -> float:
    """Unpack unsigned 32-bit integer into Python float (IEEE 754 binary32)."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f64_to_u64(val: float) -> int:
    """Pack Python float into unsigned 64-bit integer (IEEE 754 binary64)."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def u64_to_f64(bits: int) -> float:
    """Unpack unsigned 64-bit integer into Python float (IEEE 754 binary64)."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def u64_to_dwords(val: int) -> tuple[int, int]:
    """Split 64-bit word into (low 32-bit dword, high 32-bit dword)."""
    return val & MASK32, (val >> 32) & MASK32


def dwords_to_u64(lo: int, hi: int) -> int:
    """Combine (low 32-bit dword, high 32-bit dword) into 64-bit word."""
    return (lo & MASK32) | ((hi & MASK32) << 32)


def is_f32_nan(bits: int) -> bool:
    return ((bits & 0x7F800000) == 0x7F800000) and ((bits & 0x007FFFFF) != 0)


def is_f32_snan(bits: int) -> bool:
    return is_f32_nan(bits) and ((bits & 0x00400000) == 0)


def is_f32_qnan(bits: int) -> bool:
    return is_f32_nan(bits) and ((bits & 0x00400000) != 0)


def quiet_f32(bits: int) -> int:
    return (bits & MASK32) | 0x00400000


def is_f64_nan(bits: int) -> bool:
    return ((bits & 0x7FF0000000000000) == 0x7FF0000000000000) and ((bits & 0x000FFFFFFFFFFFFF) != 0)


def is_f64_snan(bits: int) -> bool:
    return is_f64_nan(bits) and ((bits & 0x0008000000000000) == 0)


def is_f64_qnan(bits: int) -> bool:
    return is_f64_nan(bits) and ((bits & 0x0008000000000000) != 0)


def quiet_f64(bits: int) -> int:
    return (bits & MASK64) | 0x0008000000000000


def apply_src_mods_u32(val_u32: int, *, neg: bool = False, abs_: bool = False) -> int:
    bits = val_u32 & MASK32
    if abs_:
        bits &= ~F32_NEG_ZERO
    if neg:
        bits ^= F32_NEG_ZERO
    return bits


def apply_src_mods_u64(val_u64: int, *, neg: bool = False, abs_: bool = False) -> int:
    bits = val_u64 & MASK64
    if abs_:
        bits &= ~F64_NEG_ZERO
    if neg:
        bits ^= F64_NEG_ZERO
    return bits


def apply_clamp_f32(val: float) -> float:
    if math.isnan(val):
        return 0.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def apply_clamp_f64(val: float) -> float:
    if math.isnan(val):
        return 0.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def apply_omod_float(val: float, omod: int) -> float:
    if omod == 0:
        return val
    elif omod == 1:
        return val * 2.0
    elif omod == 2:
        return val * 4.0
    elif omod == 3:
        return val * 0.5
    else:
        raise ValueError(f"Invalid omod: {omod}")


def ref_div_scale_f32(
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
) -> tuple[float, int]:
    """Reference mathematical oracle for v_div_scale_f32.

    Returns (scaled_result, vcc_bit).
    """
    u0 = apply_src_mods_u32(f32_to_u32(s0), neg=neg0, abs_=abs0)
    u1 = apply_src_mods_u32(f32_to_u32(s1), neg=neg1, abs_=abs1)
    u2 = apply_src_mods_u32(f32_to_u32(s2), neg=neg2, abs_=abs2)

    if is_f32_nan(u0):
        res = u32_to_f32(quiet_f32(u0))
        if clamp:
            res = apply_clamp_f32(res)
        return res, 0

    if is_f32_nan(u1) or is_f32_nan(u2):
        res = u32_to_f32(u0)
        if clamp:
            res = apply_clamp_f32(res)
        return res, 0

    v0 = u32_to_f32(u0)
    v1 = u32_to_f32(u1)
    v2 = u32_to_f32(u2)

    abs_y = abs(v1)
    abs_x = abs(v2)

    scaled = False
    scale_factor = 1.0

    # Denominator check has priority
    if 0.0 < abs_y <= THRESH_F32_SMALL:
        scaled = True
        scale_factor = SCALE64_POS
    elif abs_y >= THRESH_F32_LARGE and math.isfinite(abs_y):
        scaled = True
        scale_factor = SCALE64_NEG
    elif 0.0 < abs_x <= THRESH_F32_SMALL:
        scaled = True
        scale_factor = SCALE64_POS
    elif abs_x >= THRESH_F32_LARGE and math.isfinite(abs_x):
        scaled = True
        scale_factor = SCALE64_NEG

    if scaled:
        res = v0 * scale_factor
        vcc = 1
    else:
        res = v0
        vcc = 0

    if omod != 0:
        res = apply_omod_float(res, omod)
    if clamp:
        res = apply_clamp_f32(res)

    return res, vcc


def ref_div_scale_f32_u32(
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
) -> tuple[int, int]:
    """Bit-exact reference oracle for v_div_scale_f32."""
    res_f, vcc = ref_div_scale_f32(
        u32_to_f32(u0),
        u32_to_f32(u1),
        u32_to_f32(u2),
        neg0=neg0,
        neg1=neg1,
        neg2=neg2,
        abs0=abs0,
        abs1=abs1,
        abs2=abs2,
        clamp=clamp,
        omod=omod,
    )
    return f32_to_u32(res_f), vcc


def ref_div_scale_f64(
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
) -> tuple[float, int]:
    """Reference mathematical oracle for v_div_scale_f64.

    Returns (scaled_result, vcc_bit).
    """
    u0 = apply_src_mods_u64(f64_to_u64(s0), neg=neg0, abs_=abs0)
    u1 = apply_src_mods_u64(f64_to_u64(s1), neg=neg1, abs_=abs1)
    u2 = apply_src_mods_u64(f64_to_u64(s2), neg=neg2, abs_=abs2)

    if is_f64_nan(u0):
        res = u64_to_f64(quiet_f64(u0))
        if clamp:
            res = apply_clamp_f64(res)
        return res, 0

    if is_f64_nan(u1) or is_f64_nan(u2):
        res = u64_to_f64(u0)
        if clamp:
            res = apply_clamp_f64(res)
        return res, 0

    v0 = u64_to_f64(u0)
    v1 = u64_to_f64(u1)
    v2 = u64_to_f64(u2)

    abs_y = abs(v1)
    abs_x = abs(v2)

    scaled = False
    scale_factor = 1.0

    if 0.0 < abs_y <= THRESH_F64_SMALL:
        scaled = True
        scale_factor = SCALE128_POS
    elif abs_y >= THRESH_F64_LARGE and math.isfinite(abs_y):
        scaled = True
        scale_factor = SCALE128_NEG
    elif 0.0 < abs_x <= THRESH_F64_SMALL:
        scaled = True
        scale_factor = SCALE128_POS
    elif abs_x >= THRESH_F64_LARGE and math.isfinite(abs_x):
        scaled = True
        scale_factor = SCALE128_NEG

    if scaled:
        res = v0 * scale_factor
        vcc = 1
    else:
        res = v0
        vcc = 0

    if omod != 0:
        res = apply_omod_float(res, omod)
    if clamp:
        res = apply_clamp_f64(res)

    return res, vcc


def ref_div_scale_f64_u64(
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
) -> tuple[int, int]:
    """Bit-exact reference oracle for v_div_scale_f64."""
    res_f, vcc = ref_div_scale_f64(
        u64_to_f64(u0),
        u64_to_f64(u1),
        u64_to_f64(u2),
        neg0=neg0,
        neg1=neg1,
        neg2=neg2,
        abs0=abs0,
        abs1=abs1,
        abs2=abs2,
        clamp=clamp,
        omod=omod,
    )
    return f64_to_u64(res_f), vcc


class RegisterPair:
    """Architectural model of an aligned 64-bit register pair (e.g. v[0:1] or s[0:1])."""

    def __init__(self, lo: int = 0, hi: int = 0) -> None:
        self.lo = lo & MASK32
        self.hi = hi & MASK32

    @classmethod
    def from_u64(cls, val: int) -> RegisterPair:
        lo, hi = u64_to_dwords(val)
        return cls(lo, hi)

    @classmethod
    def from_f64(cls, val: float) -> RegisterPair:
        return cls.from_u64(f64_to_u64(val))

    @property
    def as_u64(self) -> int:
        return dwords_to_u64(self.lo, self.hi)

    @property
    def as_f64(self) -> float:
        return u64_to_f64(self.as_u64)

    def set_u64(self, val: int) -> None:
        self.lo, self.hi = u64_to_dwords(val)

    def set_f64(self, val: float) -> None:
        self.set_u64(f64_to_u64(val))

    def __repr__(self) -> str:
        return f"RegisterPair(lo={self.lo:#010x}, hi={self.hi:#010x}, f64={self.as_f64})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegisterPair):
            return NotImplemented
        return self.lo == other.lo and self.hi == other.hi


class TestDivScaleReferenceOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural models."""

    def test_f32_normal_no_scale(self):
        # Normal inputs: 10.0 / 2.0 -> no scale required
        res, vcc = ref_div_scale_f32(2.0, 2.0, 10.0)
        self.assertEqual(res, 2.0)
        self.assertEqual(vcc, 0)

        # Scaling numerator: 10.0 / 2.0 -> 10.0 unchanged
        res_num, vcc_num = ref_div_scale_f32(10.0, 2.0, 10.0)
        self.assertEqual(res_num, 10.0)
        self.assertEqual(vcc_num, 0)

    def test_f32_scale_up_small_denom(self):
        # Very small denominator (e.g. 2^-100): scale up by 2^64
        small_denom = 2.0 ** -100
        res, vcc = ref_div_scale_f32(small_denom, small_denom, 1.0)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, small_denom * SCALE64_POS)

    def test_f32_scale_down_large_denom(self):
        # Very large denominator (e.g. 2^70): scale down by 2^-64
        large_denom = 2.0 ** 70
        res, vcc = ref_div_scale_f32(large_denom, large_denom, 1.0)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, large_denom * SCALE64_NEG)

    def test_f64_normal_no_scale(self):
        # Normal inputs: 100.0 / 4.0
        res, vcc = ref_div_scale_f64(4.0, 4.0, 100.0)
        self.assertEqual(res, 4.0)
        self.assertEqual(vcc, 0)

    def test_f64_scale_up_small_denom(self):
        # Very small f64 denom (e.g. 2^-800): scale up by 2^128
        small_denom = 2.0 ** -800
        res, vcc = ref_div_scale_f64(small_denom, small_denom, 1.0)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, small_denom * SCALE128_POS)

    def test_f64_scale_down_large_denom(self):
        # Very large f64 denom (e.g. 2^800): scale down by 2^-128
        large_denom = 2.0 ** 800
        res, vcc = ref_div_scale_f64(large_denom, large_denom, 1.0)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, large_denom * SCALE128_NEG)

    def test_source_modifiers(self):
        # abs and neg
        res, vcc = ref_div_scale_f32(-2.0, 2.0, 10.0, abs0=True)
        self.assertEqual(res, 2.0)
        self.assertEqual(vcc, 0)

        res_neg, _ = ref_div_scale_f32(2.0, 2.0, 10.0, neg0=True)
        self.assertEqual(res_neg, -2.0)

    def test_output_modifiers(self):
        # clamp saturates to [0.0, 1.0]
        res, _ = ref_div_scale_f32(5.0, 2.0, 10.0, clamp=True)
        self.assertEqual(res, 1.0)

        res_neg, _ = ref_div_scale_f32(-5.0, 2.0, 10.0, clamp=True)
        self.assertEqual(res_neg, 0.0)

        # omod mul:2
        res_omod, _ = ref_div_scale_f32(3.0, 2.0, 10.0, omod=1)
        self.assertEqual(res_omod, 6.0)

    def test_register_pair_model(self):
        rp = RegisterPair.from_f64(1.0)
        self.assertEqual(rp.as_f64, 1.0)
        self.assertEqual(rp.as_u64, F64_ONE)
        self.assertEqual(rp.lo, 0x00000000)
        self.assertEqual(rp.hi, 0x3FF00000)


if __name__ == "__main__":
    unittest.main()
