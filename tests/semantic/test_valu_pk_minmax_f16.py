from __future__ import annotations

import math
import struct
import unittest
from typing import Tuple

# ==============================================================================
# IEEE 754 binary16 (Half-Precision) Architectural Constants
# ==============================================================================

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF

SIGN_MASK16: int = 0x8000
EXP_MASK16: int = 0x7C00
FRAC_MASK16: int = 0x03FF
QNAN_BIT16: int = 0x0200

# Canonical and special binary16 values
POS_ZERO: int = 0x0000          # +0.0
NEG_ZERO: int = 0x8000          # -0.0
POS_ONE: int = 0x3C00           # +1.0
NEG_ONE: int = 0xBC00           # -1.0
POS_HALF: int = 0x3800          # +0.5
NEG_HALF: int = 0xB800          # -0.5
POS_TWO: int = 0x4000           # +2.0
NEG_TWO: int = 0xC000           # -2.0
POS_INF: int = 0x7C00           # +inf
NEG_INF: int = 0xFC00           # -inf
QNAN_CANONICAL: int = 0x7E00    # Canonical quiet NaN
SNAN_DEFAULT: int = 0x7D00      # Signaling NaN default
NEG_QNAN: int = 0xFE00          # Negative quiet NaN

# Subnormal (denormal) binary16 constants
POS_MIN_SUBNORM: int = 0x0001   # +2^-24, smallest positive subnormal
POS_MAX_SUBNORM: int = 0x03FF   # largest positive subnormal
NEG_MIN_SUBNORM: int = 0x8001   # -2^-24, smallest negative subnormal
NEG_MAX_SUBNORM: int = 0x83FF   # largest negative subnormal

# Normal binary16 boundaries
POS_MIN_NORMAL: int = 0x0400    # +2^-14, smallest positive normal
POS_MAX_NORMAL: int = 0x7BFF    # +65504.0, largest finite binary16
NEG_MAX_NORMAL: int = 0xFBFF    # -65504.0, largest negative finite binary16


# ==============================================================================
# Bit Manipulation and Conversion Utilities
# ==============================================================================

def f16_to_bits(val: float) -> int:
    """Convert a Python float to an IEEE 754 binary16 16-bit integer."""
    if math.isnan(val):
        return QNAN_CANONICAL
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def bits_to_f16(bits: int) -> float:
    """Convert an IEEE 754 binary16 16-bit integer to a Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def unpack_pk_f16(packed: int) -> Tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit half words (lo, hi)."""
    lo = packed & MASK16
    hi = (packed >> 16) & MASK16
    return lo, hi


def pack_pk_f16(lo: int, hi: int) -> int:
    """Pack two 16-bit half words (lo, hi) into a single 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def is_nan16(bits: int) -> bool:
    """Check if 16-bit word represents an IEEE 754 binary16 NaN."""
    return ((bits & EXP_MASK16) == EXP_MASK16) and ((bits & FRAC_MASK16) != 0)


def is_qnan16(bits: int) -> bool:
    """Check if 16-bit word represents a Quiet NaN."""
    return is_nan16(bits) and bool(bits & QNAN_BIT16)


def is_snan16(bits: int) -> bool:
    """Check if 16-bit word represents a Signaling NaN."""
    return is_nan16(bits) and not bool(bits & QNAN_BIT16)


def is_inf16(bits: int) -> bool:
    """Check if 16-bit word represents positive or negative infinity."""
    return ((bits & EXP_MASK16) == EXP_MASK16) and ((bits & FRAC_MASK16) == 0)


def is_subnormal16(bits: int) -> bool:
    """Check if 16-bit word represents a subnormal (denormal) half-float."""
    return ((bits & EXP_MASK16) == 0) and ((bits & FRAC_MASK16) != 0)


def is_zero16(bits: int) -> bool:
    """Check if 16-bit word represents positive or negative zero."""
    return (bits & ~SIGN_MASK16) == 0


def apply_f16_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply half-float input modifiers (abs, neg)."""
    res = bits & MASK16
    if abs_mod:
        res &= ~SIGN_MASK16
    if neg_mod:
        res ^= SIGN_MASK16
    return res


# ==============================================================================
# Reference Mathematical Oracles for Scalar & Packed f16 Min/Max
# ==============================================================================

def clamp_f16(bits: int) -> int:
    """Clamp a 16-bit half-float to [0.0, 1.0]. NaN clamps to +0.0."""
    if is_nan16(bits):
        return POS_ZERO
    f = bits_to_f16(bits)
    if f < 0.0 or bits == NEG_ZERO:
        return POS_ZERO
    if f > 1.0 or bits == POS_INF:
        return POS_ONE
    return bits & MASK16


def oracle_min_f16(
    s0: int,
    s1: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for RDNA2 scalar v_min_f16 (IEEE 754-2008 / GLSL NMin).

    Rules:
    - NaN suppression: if one operand is NaN and the other is a number, return the number.
    - If both operands are NaN, return quiet NaN (s0 quietened if NaN).
    - Signed zero tie-breaking: -0.0 < +0.0 (min(-0.0, +0.0) == -0.0).
    - Subnormal numbers are preserved.
    - If clamp=True, output clamped to [0.0, 1.0].
    """
    a = apply_f16_modifiers(s0, abs0, neg0)
    b = apply_f16_modifiers(s1, abs1, neg1)

    a_nan = is_nan16(a)
    b_nan = is_nan16(b)

    if a_nan and b_nan:
        res = a | QNAN_BIT16
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
            fa = bits_to_f16(a)
            fb = bits_to_f16(b)
            if fa < fb:
                res = a
            elif fb < fa:
                res = b
            else:
                # Equal finite values: preserve sign or default to a
                res = a if (a & SIGN_MASK16) else b

    if clamp:
        res = clamp_f16(res)
    return res & MASK16


def oracle_max_f16(
    s0: int,
    s1: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for RDNA2 scalar v_max_f16 (IEEE 754-2008 / GLSL NMax).

    Rules:
    - NaN suppression: if one operand is NaN and the other is a number, return the number.
    - If both operands are NaN, return quiet NaN (s0 quietened if NaN).
    - Signed zero tie-breaking: -0.0 < +0.0 (max(-0.0, +0.0) == +0.0).
    - Subnormal numbers are preserved.
    - If clamp=True, output clamped to [0.0, 1.0].
    """
    a = apply_f16_modifiers(s0, abs0, neg0)
    b = apply_f16_modifiers(s1, abs1, neg1)

    a_nan = is_nan16(a)
    b_nan = is_nan16(b)

    if a_nan and b_nan:
        res = a | QNAN_BIT16
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
            fa = bits_to_f16(a)
            fb = bits_to_f16(b)
            if fa > fb:
                res = a
            elif fb > fa:
                res = b
            else:
                # Equal finite values: prefer positive sign
                res = b if not (b & SIGN_MASK16) else a

    if clamp:
        res = clamp_f16(res)
    return res & MASK16


def oracle_pk_min_f16(
    s0_u32: int,
    s1_u32: int,
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
    """Reference oracle for RDNA2 packed v_pk_min_f16 (VOP3P).

    Selects halves based on op_sel:
    - op_sel[0]: src0 half for dest.lo (0=lo, 1=hi)
    - op_sel[1]: src1 half for dest.lo (0=lo, 1=hi)
    - op_sel[2]: src0 half for dest.hi (0=lo, 1=hi)
    - op_sel[3]: src1 half for dest.hi (0=lo, 1=hi)
    Computes min for lo and hi independently, packs into 32-bit dword.
    """
    s0_halves = unpack_pk_f16(s0_u32)
    s1_halves = unpack_pk_f16(s1_u32)

    in0_lo = s0_halves[op_sel[0]]
    in1_lo = s1_halves[op_sel[1]]
    in0_hi = s0_halves[op_sel[2]]
    in1_hi = s1_halves[op_sel[3]]

    out_lo = oracle_min_f16(
        in0_lo, in1_lo,
        abs0=abs0_lo, neg0=neg0_lo, abs1=abs1_lo, neg1=neg1_lo,
        clamp=clamp,
    )
    out_hi = oracle_min_f16(
        in0_hi, in1_hi,
        abs0=abs0_hi, neg0=neg0_hi, abs1=abs1_hi, neg1=neg1_hi,
        clamp=clamp,
    )

    return pack_pk_f16(out_lo, out_hi)


def oracle_pk_max_f16(
    s0_u32: int,
    s1_u32: int,
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
    """Reference oracle for RDNA2 packed v_pk_max_f16 (VOP3P).

    Selects halves based on op_sel, computes max for lo and hi independently.
    """
    s0_halves = unpack_pk_f16(s0_u32)
    s1_halves = unpack_pk_f16(s1_u32)

    in0_lo = s0_halves[op_sel[0]]
    in1_lo = s1_halves[op_sel[1]]
    in0_hi = s0_halves[op_sel[2]]
    in1_hi = s1_halves[op_sel[3]]

    out_lo = oracle_max_f16(
        in0_lo, in1_lo,
        abs0=abs0_lo, neg0=neg0_lo, abs1=abs1_lo, neg1=neg1_lo,
        clamp=clamp,
    )
    out_hi = oracle_max_f16(
        in0_hi, in1_hi,
        abs0=abs0_hi, neg0=neg0_hi, abs1=abs1_hi, neg1=neg1_hi,
        clamp=clamp,
    )

    return pack_pk_f16(out_lo, out_hi)


# ==============================================================================
# Commit 1 Unit Tests
# ==============================================================================

class TestF16ModelAndOracle(unittest.TestCase):
    """Unit tests for half-precision bit representation, predicates, and reference oracles."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 65504.0, -65504.0]
        for val in values:
            with self.subTest(val=val):
                bits = f16_to_bits(val)
                recovered = bits_to_f16(bits)
                self.assertEqual(val, recovered)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_word_packing_and_unpacking(self) -> None:
        lo = 0x1234
        hi = 0x5678
        packed = pack_pk_f16(lo, hi)
        self.assertEqual(packed, 0x56781234)
        rec_lo, rec_hi = unpack_pk_f16(packed)
        self.assertEqual(rec_lo, lo)
        self.assertEqual(rec_hi, hi)

    def test_classification_predicates(self) -> None:
        self.assertTrue(is_zero16(POS_ZERO))
        self.assertTrue(is_zero16(NEG_ZERO))
        self.assertFalse(is_zero16(POS_ONE))

        self.assertTrue(is_inf16(POS_INF))
        self.assertTrue(is_inf16(NEG_INF))
        self.assertFalse(is_inf16(POS_ZERO))
        self.assertFalse(is_inf16(QNAN_CANONICAL))

        self.assertTrue(is_nan16(QNAN_CANONICAL))
        self.assertTrue(is_qnan16(QNAN_CANONICAL))
        self.assertFalse(is_snan16(QNAN_CANONICAL))

        self.assertTrue(is_nan16(SNAN_DEFAULT))
        self.assertTrue(is_snan16(SNAN_DEFAULT))
        self.assertFalse(is_qnan16(SNAN_DEFAULT))

        self.assertTrue(is_subnormal16(POS_MIN_SUBNORM))
        self.assertTrue(is_subnormal16(POS_MAX_SUBNORM))
        self.assertFalse(is_subnormal16(POS_ZERO))
        self.assertFalse(is_subnormal16(POS_MIN_NORMAL))

    def test_exact_finite_min_max(self) -> None:
        # min(1.0, 2.0) == 1.0, max(1.0, 2.0) == 2.0
        self.assertEqual(oracle_min_f16(POS_ONE, POS_TWO), POS_ONE)
        self.assertEqual(oracle_max_f16(POS_ONE, POS_TWO), POS_TWO)

        # min(-1.0, 0.5) == -1.0, max(-1.0, 0.5) == 0.5
        self.assertEqual(oracle_min_f16(NEG_ONE, POS_HALF), NEG_ONE)
        self.assertEqual(oracle_max_f16(NEG_ONE, POS_HALF), POS_HALF)

    def test_signed_zero_tie_breaking(self) -> None:
        # min(-0.0, +0.0) == -0.0
        self.assertEqual(oracle_min_f16(NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(oracle_min_f16(POS_ZERO, NEG_ZERO), NEG_ZERO)

        # max(-0.0, +0.0) == +0.0
        self.assertEqual(oracle_max_f16(NEG_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_max_f16(POS_ZERO, NEG_ZERO), POS_ZERO)

        # min(-0.0, -0.0) == -0.0; max(+0.0, +0.0) == +0.0
        self.assertEqual(oracle_min_f16(NEG_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(oracle_max_f16(POS_ZERO, POS_ZERO), POS_ZERO)

    def test_nan_suppression_and_propagation(self) -> None:
        # If one operand is NaN, non-NaN operand is selected
        self.assertEqual(oracle_min_f16(QNAN_CANONICAL, POS_TWO), POS_TWO)
        self.assertEqual(oracle_min_f16(POS_TWO, QNAN_CANONICAL), POS_TWO)
        self.assertEqual(oracle_max_f16(QNAN_CANONICAL, POS_TWO), POS_TWO)
        self.assertEqual(oracle_max_f16(POS_TWO, QNAN_CANONICAL), POS_TWO)

        # If both operands are NaN, quiet NaN is returned
        both_min = oracle_min_f16(QNAN_CANONICAL, SNAN_DEFAULT)
        self.assertTrue(is_qnan16(both_min))
        both_max = oracle_max_f16(SNAN_DEFAULT, QNAN_CANONICAL)
        self.assertTrue(is_qnan16(both_max))

    def test_modifiers_and_clamp(self) -> None:
        # abs modifier: min(|-2.0|, 1.0) == min(2.0, 1.0) == 1.0
        res_abs = oracle_min_f16(NEG_TWO, POS_ONE, abs0=True)
        self.assertEqual(res_abs, POS_ONE)

        # neg modifier: max(-(2.0), -1.0) == max(-2.0, -1.0) == -1.0
        res_neg = oracle_max_f16(POS_TWO, NEG_ONE, neg0=True)
        self.assertEqual(res_neg, NEG_ONE)

        # clamp modifier: clamps to [0.0, 1.0]
        # max(2.0, 0.5) clamped -> 1.0
        self.assertEqual(oracle_max_f16(POS_TWO, POS_HALF, clamp=True), POS_ONE)
        # min(-1.0, -2.0) clamped -> +0.0
        self.assertEqual(oracle_min_f16(NEG_ONE, NEG_TWO, clamp=True), POS_ZERO)

    def test_packed_oracles_basic(self) -> None:
        # pack: s0 = (hi=2.0, lo=-1.0), s1 = (hi=0.5, lo=1.0)
        s0 = pack_pk_f16(NEG_ONE, POS_TWO)
        s1 = pack_pk_f16(POS_ONE, POS_HALF)

        # v_pk_min_f16:
        # lo: min(-1.0, 1.0) = -1.0
        # hi: min(2.0, 0.5) = 0.5
        pk_min = oracle_pk_min_f16(s0, s1)
        exp_min = pack_pk_f16(NEG_ONE, POS_HALF)
        self.assertEqual(pk_min, exp_min)

        # v_pk_max_f16:
        # lo: max(-1.0, 1.0) = 1.0
        # hi: max(2.0, 0.5) = 2.0
        pk_max = oracle_pk_max_f16(s0, s1)
        exp_max = pack_pk_f16(POS_ONE, POS_TWO)
        self.assertEqual(pk_max, exp_max)


if __name__ == "__main__":
    unittest.main()
