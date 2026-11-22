from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_fma_f64.

Tests the double-precision fused multiply-add instruction:
    v_fma_f64 D, S0, S1, S2
Lowered semantics:
    D = (S0 * S1) + S2
with single-rounding precision per IEEE 754-2008 (binary64).
Supports input modifiers:
    - abs (|src|: clear sign bit)
    - neg (-src: invert sign bit)
Supports output modifiers:
    - clamp: saturates result to [0.0, 1.0], with NaN clamped to +0.0
    - omod: mul:2, mul:4, div:2
Architectural model:
    64-bit floating point registers mapped onto aligned VGPR/SGPR pairs:
    [lo: 32-bit dword, hi: 32-bit dword].
"""

from fractions import Fraction
import math
import struct
import sys
import unittest

# IEEE 754 double precision bitwise constants
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

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def f64_to_u64(val: float) -> int:
    """Pack Python float (IEEE 754 binary64) into unsigned 64-bit integer."""
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


def apply_src_mods_u64(val_u64: int, *, neg: bool = False, abs_: bool = False) -> int:
    """Apply bit-exact RDNA2 source modifiers (|src|, -src) to a 64-bit float bit pattern."""
    bits = val_u64 & MASK64
    if abs_:
        bits &= ~F64_NEG_ZERO  # clear sign bit
    if neg:
        bits ^= F64_NEG_ZERO   # invert sign bit
    return bits


def apply_clamp_f64(val: float) -> float:
    """Apply RDNA2 clamp modifier [0.0, 1.0] to float64, mapping NaNs to +0.0."""
    if math.isnan(val):
        return 0.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def apply_omod_f64(val: float, omod: int) -> float:
    """Apply RDNA2 output modifier: 0: none, 1: mul:2, 2: mul:4, 3: div:2."""
    if omod == 0:
        return val
    elif omod == 1:
        return val * 2.0
    elif omod == 2:
        return val * 4.0
    elif omod == 3:
        return val * 0.5
    else:
        raise ValueError(f"Invalid omod value: {omod}")


def raw_fma_f64(v0: float, v1: float, v2: float) -> float:
    """Compute exact IEEE 754 double precision FMA (v0 * v1) + v2."""
    if math.isnan(v0) or math.isnan(v1) or math.isnan(v2):
        return math.nan

    if (math.isinf(v0) and v1 == 0.0) or (v0 == 0.0 and math.isinf(v1)):
        return math.nan

    p_inf = math.isinf(v0) or math.isinf(v1)
    if p_inf:
        p_sign = -1.0 if (math.copysign(1.0, v0) * math.copysign(1.0, v1) < 0) else 1.0
        if math.isinf(v2):
            v2_sign = math.copysign(1.0, v2)
            if p_sign != v2_sign:
                return math.nan
        return math.copysign(float("inf"), p_sign)

    if math.isinf(v2):
        return v2

    if v0 == 0.0 or v1 == 0.0:
        prod_sign = -1.0 if (math.copysign(1.0, v0) * math.copysign(1.0, v1) < 0) else 1.0
        if v2 == 0.0:
            v2_sign = math.copysign(1.0, v2)
            if prod_sign < 0 and v2_sign < 0:
                return -0.0
            return 0.0
        return v2

    frac = Fraction(v0) * Fraction(v1) + Fraction(v2)
    if frac == 0:
        return 0.0
    try:
        return float(frac)
    except OverflowError:
        return math.copysign(float("inf"), 1.0 if frac > 0 else -1.0)


def ref_fma_f64(
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
    """Reference mathematical oracle for v_fma_f64 calculating (s0 * s1) + s2 in binary64."""
    u0 = apply_src_mods_u64(f64_to_u64(s0), neg=neg0, abs_=abs0)
    u1 = apply_src_mods_u64(f64_to_u64(s1), neg=neg1, abs_=abs1)
    u2 = apply_src_mods_u64(f64_to_u64(s2), neg=neg2, abs_=abs2)

    v0 = u64_to_f64(u0)
    v1 = u64_to_f64(u1)
    v2 = u64_to_f64(u2)

    res = raw_fma_f64(v0, v1, v2)

    if omod != 0:
        res = apply_omod_f64(res, omod)
    if clamp:
        res = apply_clamp_f64(res)

    return res


def ref_fma_f64_u64(
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
    """Bit-exact reference mathematical oracle for v_fma_f64 using 64-bit integer words."""
    res_f = ref_fma_f64(
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
    return f64_to_u64(res_f)


class RegisterPair:
    """Architectural model of a 64-bit register pair (e.g. v[0:1] or s[0:1])."""

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


class TestFmaF64MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural register models."""

    def test_basic_fma_identities(self):
        # 1.0 * 1.0 + 0.0 == 1.0
        self.assertEqual(ref_fma_f64(1.0, 1.0, 0.0), 1.0)
        # 2.0 * 3.0 + 4.0 == 10.0
        self.assertEqual(ref_fma_f64(2.0, 3.0, 4.0), 10.0)
        # -2.5 * 4.0 + 1.25 == -8.75
        self.assertEqual(ref_fma_f64(-2.5, 4.0, 1.25), -8.75)
        # 0.0 * 100.0 + 5.5 == 5.5
        self.assertEqual(ref_fma_f64(0.0, 100.0, 5.5), 5.5)

    def test_fused_precision_single_rounding(self):
        # FMA retains full intermediate product precision without intermediate rounding.
        # Let a = 1.0 + 2**-52, b = 1.0 - 2**-52, c = -1.0
        # (a * b) = 1.0 - 2**-104
        # Unfused: (a * b) rounds to 1.0, so (a * b) - 1.0 == 0.0
        # Fused: (a * b) - 1.0 retains -2**-104, rounding to -2**-104 != 0.0
        a = 1.0 + (2.0 ** -52)
        b = 1.0 - (2.0 ** -52)
        c = -1.0
        unfused = (a * b) + c
        fused = ref_fma_f64(a, b, c)
        self.assertEqual(unfused, 0.0)
        self.assertNotEqual(fused, 0.0)
        self.assertEqual(fused, -(2.0 ** -104))

    def test_source_modifiers(self):
        # abs(|src|): clears sign bit
        self.assertEqual(ref_fma_f64(-3.5, 2.0, 1.0, abs0=True), 8.0)
        self.assertEqual(ref_fma_f64(3.5, -2.0, 1.0, abs1=True), 8.0)
        self.assertEqual(ref_fma_f64(3.5, 2.0, -1.0, abs2=True), 8.0)

        # neg(-src): inverts sign bit
        self.assertEqual(ref_fma_f64(3.5, 2.0, 1.0, neg0=True), -6.0)
        self.assertEqual(ref_fma_f64(3.5, 2.0, 1.0, neg1=True), -6.0)
        self.assertEqual(ref_fma_f64(3.5, 2.0, 1.0, neg2=True), 6.0)

        # combined neg and abs: -|src|
        self.assertEqual(ref_fma_f64(-3.5, 2.0, 1.0, neg0=True, abs0=True), -6.0)

    def test_output_modifiers_clamp_and_omod(self):
        # clamp saturates to [0.0, 1.0]
        self.assertEqual(ref_fma_f64(2.0, 3.0, 0.0, clamp=True), 1.0)
        self.assertEqual(ref_fma_f64(-2.0, 3.0, 0.0, clamp=True), 0.0)
        self.assertEqual(ref_fma_f64(0.5, 1.0, 0.25, clamp=True), 0.75)

        # omod: mul:2, mul:4, div:2
        self.assertEqual(ref_fma_f64(1.5, 2.0, 0.5, omod=1), 7.0)   # (3.5) * 2 = 7.0
        self.assertEqual(ref_fma_f64(1.5, 2.0, 0.5, omod=2), 14.0)  # (3.5) * 4 = 14.0
        self.assertEqual(ref_fma_f64(1.5, 2.0, 0.5, omod=3), 1.75)  # (3.5) * 0.5 = 1.75

        # omod combined with clamp: (omod applies before clamp)
        self.assertEqual(ref_fma_f64(0.2, 2.0, 0.0, omod=1, clamp=True), 0.8)
        self.assertEqual(ref_fma_f64(0.4, 2.0, 0.0, omod=2, clamp=True), 1.0)

    def test_register_pair_architecture(self):
        rp = RegisterPair.from_f64(1.0)
        self.assertEqual(rp.as_f64, 1.0)
        self.assertEqual(rp.as_u64, F64_ONE)
        self.assertEqual(rp.lo, 0x00000000)
        self.assertEqual(rp.hi, 0x3FF00000)

        rp.set_f64(-2.0)
        self.assertEqual(rp.as_f64, -2.0)
        self.assertEqual(rp.hi, 0xC0000000)

        # bit-exact oracle roundtrip
        u_res = ref_fma_f64_u64(F64_TWO, F64_HALF, F64_ONE)
        self.assertEqual(u_res, f64_to_u64(2.0))


if __name__ == "__main__":
    unittest.main()
