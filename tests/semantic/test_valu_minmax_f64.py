"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_min_f64 and v_max_f64.

Models IEEE 754-2008 / IEEE 754-2019 binary64 double-precision minNum/maxNum and AMD RDNA2 / GFX10
Vector Arithmetic Logic Unit (VALU) architectural semantics:
  - v_min_f64: D.d = min(S0.d, S1.d)
  - v_max_f64: D.d = max(S0.d, S1.d)

RDNA2 Double-Precision Floating-Point ISA Rules:
  1. Register Pair Allocation:
     Double-precision 64-bit values occupy aligned pairs of 32-bit vector registers:
     v[d:d+1] where low 32 bits reside in v[d] and high 32 bits in v[d+1].
  2. NaN Suppression (IEEE 754-2008 / 2019 minNum / maxNum):
     If one operand is NaN and the other is a valid floating-point number, the valid number is
     returned unchanged. If both operands are NaN, a quiet NaN is returned.
  3. Signed Zero Ordering (-0.0 < +0.0):
     RDNA2 orders -0.0 strictly less than +0.0:
       min(+0.0, -0.0) = -0.0, min(-0.0, +0.0) = -0.0
       max(+0.0, -0.0) = +0.0, max(-0.0, +0.0) = +0.0
  4. Infinities & Subnormals:
     -INF < all finite numbers < +INF.
     Subnormal double values are preserved without flushing to zero in IEEE mode.
  5. VOP3 Encoding Modifiers:
     Input modifiers: abs (|x|) clears sign bit 63; neg (-x) inverts sign bit 63.
     When both abs and neg are set, neg(abs(x)) clears then sets sign bit (forcing negative).
     Output modifier: clamp clamps final result to [0.0, 1.0]. Clamping negative values or -0.0
     yields +0.0. Clamping NaN returns NaN unchanged.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_MASK_F64: int = 0x8000000000000000
EXP_MASK_F64: int = 0x7FF0000000000000
MANT_MASK_F64: int = 0x000FFFFFFFFFFFFF
QUIET_NAN_BIT_F64: int = 0x0008000000000000

# Canonical IEEE 754 binary64 bit patterns
POS_ZERO_F64: int = 0x0000000000000000
NEG_ZERO_F64: int = 0x8000000000000000
POS_ONE_F64: int = 0x3FF0000000000000       # 1.0
NEG_ONE_F64: int = 0xBFF0000000000000       # -1.0
POS_TWO_F64: int = 0x4000000000000000       # 2.0
NEG_TWO_F64: int = 0xC000000000000000       # -2.0
HALF_F64: int = 0x3FE0000000000000          # 0.5
NEG_HALF_F64: int = 0xBFE0000000000000      # -0.5
POS_THREE_F64: int = 0x4008000000000000     # 3.0
POS_TEN_F64: int = 0x4024000000000000       # 10.0
NEG_TEN_F64: int = 0xC024000000000000       # -10.0

DBL_MIN: int = 0x0010000000000000           # Smallest positive normal: 2^-1022
NEG_DBL_MIN: int = 0x8010000000000000       # -2^-1022
DBL_MAX: int = 0x7FEFFFFFFFFFFFFF           # (2 - 2^-52) * 2^1023
NEG_DBL_MAX: int = 0xFFEFFFFFFFFFFFFF
POS_INF_F64: int = 0x7FF0000000000000       # +infinity
NEG_INF_F64: int = 0xFFF0000000000000       # -infinity
QNAN_F64: int = 0x7FF8000000000000          # Canonical quiet NaN
NEG_QNAN_F64: int = 0xFFF8000000000000      # Negative quiet NaN
QNAN_PAYLOAD_A: int = 0x7FF8000000000001    # Quiet NaN with payload A
QNAN_PAYLOAD_B: int = 0xFFF9000000000002    # Quiet NaN with payload B
SNAN_F64: int = 0x7FF0000000000001          # Signaling NaN (MSB of mantissa is 0, payload != 0)
SUBNORM_MIN_F64: int = 0x0000000000000001   # Smallest subnormal: 2^-1074
SUBNORM_MAX_F64: int = 0x000FFFFFFFFFFFFF   # Largest positive subnormal: (1 - 2^-52) * 2^-1022
NEG_SUBNORM_MIN_F64: int = 0x8000000000000001
NEG_SUBNORM_MAX_F64: int = 0x800FFFFFFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned word as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & SIGN_MASK_F64) else val


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def bits_to_f64(bits: int) -> float:
    """Decode 64-bit unsigned integer to IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    """Encode double float to IEEE 754 double-precision 64-bit unsigned integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return POS_INF_F64 if val > 0 else NEG_INF_F64


def is_nan_f64(bits: int) -> bool:
    """True if 64-bit integer represents an IEEE 754 double NaN."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == EXP_MASK_F64 and (bits & MANT_MASK_F64) != 0


def is_qnan_f64(bits: int) -> bool:
    """True if bits represent a quiet NaN (MSB of mantissa bit 51 is 1)."""
    bits &= MASK64
    return is_nan_f64(bits) and bool(bits & QUIET_NAN_BIT_F64)


def is_snan_f64(bits: int) -> bool:
    """True if bits represent a signaling NaN (MSB of mantissa bit 51 is 0)."""
    bits &= MASK64
    return is_nan_f64(bits) and not bool(bits & QUIET_NAN_BIT_F64)


def is_subnormal_f64(bits: int) -> bool:
    """True if bits represent an IEEE 754 double subnormal number."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == 0 and (bits & MANT_MASK_F64) != 0


def zero_ordered_key(v: float) -> tuple[float, float]:
    """Sort key ordering -0.0 strictly below +0.0."""
    return (v, math.copysign(1.0, v))


def ref_min_f64(s0: int, s1: int) -> int:
    """Pure bitwise reference oracle for RDNA2 v_min_f64.

    Rules:
      - If one operand is NaN, return the non-NaN operand unchanged.
      - If both operands are NaN, return canonical quiet NaN (0x7FF8000000000000).
      - If non-NaN values, order -0.0 < +0.0.
    """
    s0 &= MASK64
    s1 &= MASK64
    nan0 = is_nan_f64(s0)
    nan1 = is_nan_f64(s1)
    if nan0 and nan1:
        return QNAN_F64
    if nan0:
        return s1
    if nan1:
        return s0

    f0 = bits_to_f64(s0)
    f1 = bits_to_f64(s1)
    k0 = zero_ordered_key(f0)
    k1 = zero_ordered_key(f1)
    return s0 if k0 <= k1 else s1


def ref_max_f64(s0: int, s1: int) -> int:
    """Pure bitwise reference oracle for RDNA2 v_max_f64.

    Rules:
      - If one operand is NaN, return the non-NaN operand unchanged.
      - If both operands are NaN, return canonical quiet NaN (0x7FF8000000000000).
      - If non-NaN values, order -0.0 < +0.0.
    """
    s0 &= MASK64
    s1 &= MASK64
    nan0 = is_nan_f64(s0)
    nan1 = is_nan_f64(s1)
    if nan0 and nan1:
        return QNAN_F64
    if nan0:
        return s1
    if nan1:
        return s0

    f0 = bits_to_f64(s0)
    f1 = bits_to_f64(s1)
    k0 = zero_ordered_key(f0)
    k1 = zero_ordered_key(f1)
    return s0 if k0 >= k1 else s1


def apply_f64_modifiers(bits: int, neg: bool = False, abs_val: bool = False) -> int:
    """Applies VOP3 input modifiers: abs (|x|) first, then neg (-x)."""
    bits &= MASK64
    if abs_val:
        bits &= ~SIGN_MASK_F64
    if neg:
        bits ^= SIGN_MASK_F64
    return bits & MASK64


def apply_f64_clamp(bits: int) -> int:
    """Applies VOP3 clamp modifier: clamps result to [0.0, 1.0]."""
    bits &= MASK64
    if is_nan_f64(bits):
        return bits
    val = bits_to_f64(bits)
    if val < 0.0 or (val == 0.0 and (bits & SIGN_MASK_F64)):
        return POS_ZERO_F64
    if val > 1.0:
        return POS_ONE_F64
    return bits


def ref_min_f64_ext(
    s0: int,
    s1: int,
    neg0: bool = False,
    abs0: bool = False,
    neg1: bool = False,
    abs1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for v_min_f64 with input and output modifiers."""
    mod0 = apply_f64_modifiers(s0, neg=neg0, abs_val=abs0)
    mod1 = apply_f64_modifiers(s1, neg=neg1, abs_val=abs1)
    res = ref_min_f64(mod0, mod1)
    return apply_f64_clamp(res) if clamp else res


def ref_max_f64_ext(
    s0: int,
    s1: int,
    neg0: bool = False,
    abs0: bool = False,
    neg1: bool = False,
    abs1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for v_max_f64 with input and output modifiers."""
    mod0 = apply_f64_modifiers(s0, neg=neg0, abs_val=abs0)
    mod1 = apply_f64_modifiers(s1, neg=neg1, abs_val=abs1)
    res = ref_max_f64(mod0, mod1)
    return apply_f64_clamp(res) if clamp else res


class TestValuMinMaxF64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_basic_float_min_max(self) -> None:
        self.assertEqual(ref_min_f64(POS_ONE_F64, POS_TWO_F64), POS_ONE_F64)
        self.assertEqual(ref_min_f64(POS_TWO_F64, POS_ONE_F64), POS_ONE_F64)
        self.assertEqual(ref_max_f64(POS_ONE_F64, POS_TWO_F64), POS_TWO_F64)
        self.assertEqual(ref_max_f64(POS_TWO_F64, POS_ONE_F64), POS_TWO_F64)
        self.assertEqual(ref_min_f64(NEG_TWO_F64, NEG_ONE_F64), NEG_TWO_F64)
        self.assertEqual(ref_max_f64(NEG_TWO_F64, NEG_ONE_F64), NEG_ONE_F64)

    def test_signed_zeros_ordering(self) -> None:
        self.assertEqual(ref_min_f64(POS_ZERO_F64, NEG_ZERO_F64), NEG_ZERO_F64)
        self.assertEqual(ref_min_f64(NEG_ZERO_F64, POS_ZERO_F64), NEG_ZERO_F64)
        self.assertEqual(ref_max_f64(POS_ZERO_F64, NEG_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(ref_max_f64(NEG_ZERO_F64, POS_ZERO_F64), POS_ZERO_F64)

    def test_infinities_ordering(self) -> None:
        self.assertEqual(ref_min_f64(NEG_INF_F64, POS_INF_F64), NEG_INF_F64)
        self.assertEqual(ref_max_f64(NEG_INF_F64, POS_INF_F64), POS_INF_F64)
        self.assertEqual(ref_min_f64(DBL_MAX, POS_INF_F64), DBL_MAX)
        self.assertEqual(ref_max_f64(DBL_MAX, POS_INF_F64), POS_INF_F64)
        self.assertEqual(ref_min_f64(NEG_DBL_MAX, NEG_INF_F64), NEG_INF_F64)
        self.assertEqual(ref_max_f64(NEG_DBL_MAX, NEG_INF_F64), NEG_DBL_MAX)

    def test_subnormals_preservation(self) -> None:
        self.assertTrue(is_subnormal_f64(SUBNORM_MIN_F64))
        self.assertTrue(is_subnormal_f64(SUBNORM_MAX_F64))
        self.assertEqual(ref_min_f64(SUBNORM_MIN_F64, SUBNORM_MAX_F64), SUBNORM_MIN_F64)
        self.assertEqual(ref_max_f64(SUBNORM_MIN_F64, SUBNORM_MAX_F64), SUBNORM_MAX_F64)
        self.assertEqual(ref_min_f64(POS_ZERO_F64, SUBNORM_MIN_F64), POS_ZERO_F64)
        self.assertEqual(ref_max_f64(POS_ZERO_F64, SUBNORM_MIN_F64), SUBNORM_MIN_F64)

    def test_nan_suppression_and_propagation(self) -> None:
        self.assertEqual(ref_min_f64(QNAN_F64, POS_ONE_F64), POS_ONE_F64)
        self.assertEqual(ref_min_f64(POS_ONE_F64, QNAN_F64), POS_ONE_F64)
        self.assertEqual(ref_max_f64(QNAN_F64, POS_ONE_F64), POS_ONE_F64)
        self.assertEqual(ref_max_f64(POS_ONE_F64, QNAN_F64), POS_ONE_F64)
        self.assertEqual(ref_min_f64(QNAN_PAYLOAD_A, QNAN_PAYLOAD_B), QNAN_F64)
        self.assertEqual(ref_max_f64(QNAN_PAYLOAD_A, QNAN_PAYLOAD_B), QNAN_F64)

    def test_modifiers_abs_neg(self) -> None:
        self.assertEqual(apply_f64_modifiers(NEG_ONE_F64, abs_val=True), POS_ONE_F64)
        self.assertEqual(apply_f64_modifiers(POS_ONE_F64, neg=True), NEG_ONE_F64)
        self.assertEqual(apply_f64_modifiers(POS_ONE_F64, neg=True, abs_val=True), NEG_ONE_F64)
        self.assertEqual(ref_min_f64_ext(POS_TWO_F64, POS_ONE_F64, neg0=True), NEG_TWO_F64)
        self.assertEqual(ref_max_f64_ext(NEG_TWO_F64, NEG_TEN_F64, abs0=True), POS_TWO_F64)

    def test_clamp_modifier(self) -> None:
        self.assertEqual(apply_f64_clamp(NEG_TWO_F64), POS_ZERO_F64)
        self.assertEqual(apply_f64_clamp(NEG_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(apply_f64_clamp(POS_TWO_F64), POS_ONE_F64)
        self.assertEqual(apply_f64_clamp(HALF_F64), HALF_F64)
        self.assertEqual(apply_f64_clamp(QNAN_F64), QNAN_F64)

    def test_split_join_dwords(self) -> None:
        lo, hi = split_dwords(POS_ONE_F64)
        self.assertEqual(lo, 0x00000000)
        self.assertEqual(hi, 0x3FF00000)
        self.assertEqual(join_dwords(lo, hi), POS_ONE_F64)


if __name__ == "__main__":
    unittest.main()
