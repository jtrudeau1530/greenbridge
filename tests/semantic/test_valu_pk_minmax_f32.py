from __future__ import annotations

import math
import struct
import unittest
from typing import Tuple

# ==============================================================================
# IEEE 754 binary32 (Single-Precision) Architectural Constants
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF

SIGN_MASK32: int = 0x8000_0000
EXP_MASK32: int = 0x7F80_0000
FRAC_MASK32: int = 0x007F_FFFF
QNAN_BIT32: int = 0x0040_0000

# Canonical and special binary32 values
POS_ZERO: int = 0x0000_0000          # +0.0
NEG_ZERO: int = 0x8000_0000          # -0.0
POS_ONE: int = 0x3F80_0000           # +1.0
NEG_ONE: int = 0xBF80_0000           # -1.0
POS_HALF: int = 0x3F00_0000          # +0.5
NEG_HALF: int = 0xBF00_0000          # -0.5
POS_TWO: int = 0x4000_0000           # +2.0
NEG_TWO: int = 0xC000_0000           # -2.0
POS_INF: int = 0x7F80_0000           # +inf
NEG_INF: int = 0xFF80_0000           # -inf
QNAN_CANONICAL: int = 0x7FC0_0000    # Canonical quiet NaN
SNAN_DEFAULT: int = 0x7F80_0001      # Signaling NaN default
NEG_QNAN: int = 0xFFC0_0000          # Negative quiet NaN

# Subnormal (denormal) binary32 constants
POS_MIN_SUBNORM: int = 0x0000_0001   # +2^-149, smallest positive subnormal
POS_MAX_SUBNORM: int = 0x007F_FFFF   # largest positive subnormal
NEG_MIN_SUBNORM: int = 0x8000_0001   # -2^-149, smallest negative subnormal
NEG_MAX_SUBNORM: int = 0x807F_FFFF   # largest negative subnormal

# Normal binary32 boundaries
POS_MIN_NORMAL: int = 0x0080_0000    # +2^-126, smallest positive normal
POS_MAX_NORMAL: int = 0x7F7F_FFFF    # largest finite binary32
NEG_MAX_NORMAL: int = 0xFF7F_FFFF    # largest negative finite binary32


# ==============================================================================
# Bit Manipulation and Conversion Utilities
# ==============================================================================

def f32_to_bits(val: float) -> int:
    """Convert a Python float to an IEEE 754 binary32 32-bit integer."""
    if math.isnan(val):
        return QNAN_CANONICAL
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def bits_to_f32(bits: int) -> float:
    """Convert an IEEE 754 binary32 32-bit integer to a Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def unpack_pk_f32(packed: int) -> Tuple[int, int]:
    """Unpack a 64-bit register pair into two 32-bit dwords (lo, hi)."""
    lo = packed & MASK32
    hi = (packed >> 32) & MASK32
    return lo, hi


def pack_pk_f32(lo: int, hi: int) -> int:
    """Pack two 32-bit dwords (lo, hi) into a single 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def is_nan32(bits: int) -> bool:
    """Check if 32-bit word represents an IEEE 754 binary32 NaN."""
    return ((bits & EXP_MASK32) == EXP_MASK32) and ((bits & FRAC_MASK32) != 0)


def is_qnan32(bits: int) -> bool:
    """Check if 32-bit word represents a Quiet NaN."""
    return is_nan32(bits) and bool(bits & QNAN_BIT32)


def is_snan32(bits: int) -> bool:
    """Check if 32-bit word represents a Signaling NaN."""
    return is_nan32(bits) and not bool(bits & QNAN_BIT32)


def is_inf32(bits: int) -> bool:
    """Check if 32-bit word represents positive or negative infinity."""
    return ((bits & EXP_MASK32) == EXP_MASK32) and ((bits & FRAC_MASK32) == 0)


def is_subnormal32(bits: int) -> bool:
    """Check if 32-bit word represents a subnormal (denormal) single-precision float."""
    return ((bits & EXP_MASK32) == 0) and ((bits & FRAC_MASK32) != 0)


def is_zero32(bits: int) -> bool:
    """Check if 32-bit word represents positive or negative zero."""
    return (bits & ~SIGN_MASK32) == 0


def apply_f32_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply single-precision float input modifiers (abs, neg)."""
    res = bits & MASK32
    if abs_mod:
        res &= ~SIGN_MASK32
    if neg_mod:
        res ^= SIGN_MASK32
    return res


# ==============================================================================
# Reference Mathematical Oracles for Scalar & Packed f32 Min/Max
# ==============================================================================

def clamp_f32(bits: int) -> int:
    """Clamp a 32-bit float to [0.0, 1.0]. NaN clamps to +0.0."""
    if is_nan32(bits):
        return POS_ZERO
    f = bits_to_f32(bits)
    if f < 0.0 or bits == NEG_ZERO:
        return POS_ZERO
    if f > 1.0 or bits == POS_INF:
        return POS_ONE
    return bits & MASK32


def oracle_min_f32(
    s0: int,
    s1: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for RDNA2 scalar single-precision min (IEEE 754-2008 / GLSL NMin).

    Rules:
    - NaN suppression: if one operand is NaN and the other is a number, return the number.
    - If both operands are NaN, return quiet NaN (canonical quiet NaN).
    - Signed zero tie-breaking: -0.0 < +0.0 (min(-0.0, +0.0) == -0.0).
    - Subnormal numbers are preserved.
    - If clamp=True, output clamped to [0.0, 1.0].
    """
    a = apply_f32_modifiers(s0, abs0, neg0)
    b = apply_f32_modifiers(s1, abs1, neg1)

    a_nan = is_nan32(a)
    b_nan = is_nan32(b)

    if a_nan and b_nan:
        res = a | QNAN_BIT32
    elif a_nan:
        res = b
    elif b_nan:
        res = a
    else:
        a_is_neg_zero = (a == NEG_ZERO)
        b_is_neg_zero = (b == NEG_ZERO)
        a_is_pos_zero = (a == POS_ZERO)
        b_is_pos_zero = (b == POS_ZERO)

        if (a_is_neg_zero and b_is_pos_zero) or (a_is_pos_zero and b_is_neg_zero):
            res = NEG_ZERO
        elif (a_is_neg_zero and b_is_neg_zero):
            res = NEG_ZERO
        elif (a_is_pos_zero and b_is_pos_zero):
            res = POS_ZERO
        else:
            fa = bits_to_f32(a)
            fb = bits_to_f32(b)
            if fa < fb:
                res = a
            elif fb < fa:
                res = b
            else:
                res = a if (a & SIGN_MASK32) else b

    if clamp:
        res = clamp_f32(res)
    return res & MASK32


def oracle_max_f32(
    s0: int,
    s1: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for RDNA2 scalar single-precision max (IEEE 754-2008 / GLSL NMax).

    Rules:
    - NaN suppression: if one operand is NaN and the other is a number, return the number.
    - If both operands are NaN, return quiet NaN.
    - Signed zero tie-breaking: -0.0 < +0.0 (max(-0.0, +0.0) == +0.0).
    - Subnormal numbers are preserved.
    - If clamp=True, output clamped to [0.0, 1.0].
    """
    a = apply_f32_modifiers(s0, abs0, neg0)
    b = apply_f32_modifiers(s1, abs1, neg1)

    a_nan = is_nan32(a)
    b_nan = is_nan32(b)

    if a_nan and b_nan:
        res = a | QNAN_BIT32
    elif a_nan:
        res = b
    elif b_nan:
        res = a
    else:
        a_is_neg_zero = (a == NEG_ZERO)
        b_is_neg_zero = (b == NEG_ZERO)
        a_is_pos_zero = (a == POS_ZERO)
        b_is_pos_zero = (b == POS_ZERO)

        if (a_is_neg_zero and b_is_pos_zero) or (a_is_pos_zero and b_is_neg_zero):
            res = POS_ZERO
        elif (a_is_neg_zero and b_is_neg_zero):
            res = NEG_ZERO
        elif (a_is_pos_zero and b_is_pos_zero):
            res = POS_ZERO
        else:
            fa = bits_to_f32(a)
            fb = bits_to_f32(b)
            if fa > fb:
                res = a
            elif fb > fa:
                res = b
            else:
                res = b if not (b & SIGN_MASK32) else a

    if clamp:
        res = clamp_f32(res)
    return res & MASK32


def oracle_pk_min_f32(
    s0_u64: int,
    s1_u64: int,
    *,
    op_sel: Tuple[int, int, int, int] = (0, 0, 1, 1),
    abs0_lo: bool = False,
    neg0_lo: bool = False,
    abs1_lo: bool = False,
    neg1_lo: bool = False,
    abs0_hi: bool = False,
    neg0_hi: bool = False,
    abs1_hi: bool = False,
    neg1_hi: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for RDNA2 packed single-precision v_pk_min_f32."""
    s0_words = unpack_pk_f32(s0_u64)
    s1_words = unpack_pk_f32(s1_u64)

    in0_lo = s0_words[op_sel[0]]
    in1_lo = s1_words[op_sel[1]]
    in0_hi = s0_words[op_sel[2]]
    in1_hi = s1_words[op_sel[3]]

    out_lo = oracle_min_f32(
        in0_lo, in1_lo,
        abs0=abs0_lo, neg0=neg0_lo, abs1=abs1_lo, neg1=neg1_lo,
        clamp=clamp,
    )
    out_hi = oracle_min_f32(
        in0_hi, in1_hi,
        abs0=abs0_hi, neg0=neg0_hi, abs1=abs1_hi, neg1=neg1_hi,
        clamp=clamp,
    )

    return pack_pk_f32(out_lo, out_hi)


def oracle_pk_max_f32(
    s0_u64: int,
    s1_u64: int,
    *,
    op_sel: Tuple[int, int, int, int] = (0, 0, 1, 1),
    abs0_lo: bool = False,
    neg0_lo: bool = False,
    abs1_lo: bool = False,
    neg1_lo: bool = False,
    abs0_hi: bool = False,
    neg0_hi: bool = False,
    abs1_hi: bool = False,
    neg1_hi: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for RDNA2 packed single-precision v_pk_max_f32."""
    s0_words = unpack_pk_f32(s0_u64)
    s1_words = unpack_pk_f32(s1_u64)

    in0_lo = s0_words[op_sel[0]]
    in1_lo = s1_words[op_sel[1]]
    in0_hi = s0_words[op_sel[2]]
    in1_hi = s1_words[op_sel[3]]

    out_lo = oracle_max_f32(
        in0_lo, in1_lo,
        abs0=abs0_lo, neg0=neg0_lo, abs1=abs1_lo, neg1=neg1_lo,
        clamp=clamp,
    )
    out_hi = oracle_max_f32(
        in0_hi, in1_hi,
        abs0=abs0_hi, neg0=neg0_hi, abs1=abs1_hi, neg1=neg1_hi,
        clamp=clamp,
    )

    return pack_pk_f32(out_lo, out_hi)


# ==============================================================================
# Commit 1 Unit Tests
# ==============================================================================

class TestF32ModelAndOracle(unittest.TestCase):
    """Unit tests for single-precision bit representation, predicates, and reference oracles."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1048576.0, -1048576.0]
        for val in values:
            with self.subTest(val=val):
                bits = f32_to_bits(val)
                recovered = bits_to_f32(bits)
                self.assertEqual(val, recovered)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_word_packing_and_unpacking(self) -> None:
        lo = 0x12345678
        hi = 0x9ABCDEF0
        packed = pack_pk_f32(lo, hi)
        self.assertEqual(packed, (hi << 32) | lo)
        rec_lo, rec_hi = unpack_pk_f32(packed)
        self.assertEqual(rec_lo, lo)
        self.assertEqual(rec_hi, hi)

    def test_classification_predicates(self) -> None:
        self.assertTrue(is_zero32(POS_ZERO))
        self.assertTrue(is_zero32(NEG_ZERO))
        self.assertFalse(is_zero32(POS_ONE))

        self.assertTrue(is_inf32(POS_INF))
        self.assertTrue(is_inf32(NEG_INF))
        self.assertFalse(is_inf32(POS_ZERO))
        self.assertFalse(is_inf32(QNAN_CANONICAL))

        self.assertTrue(is_nan32(QNAN_CANONICAL))
        self.assertTrue(is_qnan32(QNAN_CANONICAL))
        self.assertFalse(is_snan32(QNAN_CANONICAL))

        self.assertTrue(is_nan32(SNAN_DEFAULT))
        self.assertTrue(is_snan32(SNAN_DEFAULT))
        self.assertFalse(is_qnan32(SNAN_DEFAULT))

        self.assertTrue(is_subnormal32(POS_MIN_SUBNORM))
        self.assertTrue(is_subnormal32(POS_MAX_SUBNORM))
        self.assertFalse(is_subnormal32(POS_ZERO))
        self.assertFalse(is_subnormal32(POS_MIN_NORMAL))

    def test_exact_finite_min_max(self) -> None:
        self.assertEqual(oracle_min_f32(POS_ONE, POS_TWO), POS_ONE)
        self.assertEqual(oracle_max_f32(POS_ONE, POS_TWO), POS_TWO)

        self.assertEqual(oracle_min_f32(NEG_ONE, POS_HALF), NEG_ONE)
        self.assertEqual(oracle_max_f32(NEG_ONE, POS_HALF), POS_HALF)

    def test_signed_zero_tie_breaking(self) -> None:
        self.assertEqual(oracle_min_f32(NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(oracle_min_f32(POS_ZERO, NEG_ZERO), NEG_ZERO)

        self.assertEqual(oracle_max_f32(NEG_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_max_f32(POS_ZERO, NEG_ZERO), POS_ZERO)

        self.assertEqual(oracle_min_f32(NEG_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(oracle_max_f32(POS_ZERO, POS_ZERO), POS_ZERO)

    def test_nan_suppression_and_propagation(self) -> None:
        self.assertEqual(oracle_min_f32(QNAN_CANONICAL, POS_TWO), POS_TWO)
        self.assertEqual(oracle_min_f32(POS_TWO, QNAN_CANONICAL), POS_TWO)
        self.assertEqual(oracle_max_f32(QNAN_CANONICAL, POS_TWO), POS_TWO)
        self.assertEqual(oracle_max_f32(POS_TWO, QNAN_CANONICAL), POS_TWO)

        both_min = oracle_min_f32(QNAN_CANONICAL, SNAN_DEFAULT)
        self.assertTrue(is_qnan32(both_min))
        both_max = oracle_max_f32(SNAN_DEFAULT, QNAN_CANONICAL)
        self.assertTrue(is_qnan32(both_max))

    def test_modifiers_and_clamp(self) -> None:
        res_abs = oracle_min_f32(NEG_TWO, POS_ONE, abs0=True)
        self.assertEqual(res_abs, POS_ONE)

        res_neg = oracle_max_f32(POS_TWO, NEG_ONE, neg0=True)
        self.assertEqual(res_neg, NEG_ONE)

        self.assertEqual(oracle_max_f32(POS_TWO, POS_HALF, clamp=True), POS_ONE)
        self.assertEqual(oracle_min_f32(NEG_ONE, NEG_TWO, clamp=True), POS_ZERO)

    def test_packed_oracles_basic(self) -> None:
        s0 = pack_pk_f32(NEG_ONE, POS_TWO)
        s1 = pack_pk_f32(POS_ONE, POS_HALF)

        pk_min = oracle_pk_min_f32(s0, s1)
        exp_min = pack_pk_f32(NEG_ONE, POS_HALF)
        self.assertEqual(pk_min, exp_min)

        pk_max = oracle_pk_max_f32(s0, s1)
        exp_max = pack_pk_f32(POS_ONE, POS_TWO)
        self.assertEqual(pk_max, exp_max)


if __name__ == "__main__":
    unittest.main()
