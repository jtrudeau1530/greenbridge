"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_rsq_f64.

Models IEEE 754-2008 / IEEE 754-2019 binary64 double-precision reciprocal square root
and AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics:
  D.d = 1.0 / sqrt(S0.d)

RDNA2 Double-Precision Floating-Point ISA Rules:
  1. Register Pair Allocation:
     Double-precision 64-bit values occupy aligned pairs of 32-bit vector registers:
     v[d:d+1] where low 32 bits reside in v[d] and high 32 bits in v[d+1].
  2. Reciprocal Square Root Operation:
     Computes the IEEE 754 reciprocal square root of S0.
     - Positive finite numbers: 1.0 / sqrt(S0). Exact powers of 4 and 2 yield exact results:
       rsq(1.0) = 1.0, rsq(4.0) = 0.5, rsq(16.0) = 0.25, rsq(0.25) = 2.0, rsq(0.0625) = 4.0.
     - Zeros: rsq(+0.0) = +Infinity, rsq(-0.0) = +Infinity.
     - Positive Infinity: rsq(+Infinity) = +0.0.
     - Negative numbers & -Infinity: square root of negative is invalid, producing Quiet NaN.
     - NaNs: Quiet NaN propagates; Signaling NaN is quieted.
  3. Modifiers (VOP3 encoding):
     - Input modifiers: abs (|S0|) clears sign bit 63; neg (-S0) toggles sign bit 63.
     - Output modifier: clamp saturates result to [0.0, 1.0]. Clamping negative/NaN yields +0.0.
  4. Subnormals & Flush-to-Zero:
     Under IEEE mode, subnormal double values are computed with full precision.
     Under flush-to-zero (FTZ), subnormal inputs flush to signed zero, yielding +Infinity.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import math
import struct
import unittest
from typing import Tuple

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
SIGN_MASK: int = 0x8000_0000_0000_0000
EXP_MASK: int = 0x7FF0_0000_0000_0000
FRAC_MASK: int = 0x000F_FFFF_FFFF_FFFF
QNAN_BIT: int = 0x0008_0000_0000_0000
CANONICAL_QNAN: int = 0x7FF8_0000_0000_0000

# Canonical IEEE 754 binary64 bit patterns
POS_ZERO: int = 0x0000_0000_0000_0000
NEG_ZERO: int = 0x8000_0000_0000_0000
POS_INF: int = 0x7FF0_0000_0000_0000
NEG_INF: int = 0xFFF0_0000_0000_0000

ONE_F64: int = 0x3FF0_0000_0000_0000          # 1.0
FOUR_F64: int = 0x4010_0000_0000_0000         # 4.0
SIXTEEN_F64: int = 0x4030_0000_0000_0000      # 16.0
SIXTY_FOUR_F64: int = 0x4050_0000_0000_0000   # 64.0
QUARTER_F64: int = 0x3FD0_0000_0000_0000      # 0.25
SIXTEENTH_F64: int = 0x3FB0_0000_0000_0000    # 0.0625
HALF_F64: int = 0x3FE0_0000_0000_0000         # 0.5
TWO_F64: int = 0x4000_0000_0000_0000          # 2.0
THREE_F64: int = 0x4008_0000_0000_0000        # 3.0
NINE_F64: int = 0x4022_0000_0000_0000         # 9.0
HUNDRED_F64: int = 0x4059_0000_0000_0000      # 100.0

NEG_ONE_F64: int = 0xBFF0_0000_0000_0000      # -1.0
NEG_FOUR_F64: int = 0xC010_0000_0000_0000     # -4.0
NEG_QUARTER_F64: int = 0xBFD0_0000_0000_0000  # -0.25

DBL_MIN: int = 0x0010_0000_0000_0000          # Smallest positive normal: 2^-1022
DBL_MAX: int = 0x7FEF_FFFF_FFFF_FFFF          # Largest finite positive float
SUBNORM_MIN: int = 0x0000_0000_0000_0001      # Smallest positive subnormal: 2^-1074
SUBNORM_MAX: int = 0x000F_FFFF_FFFF_FFFF      # Largest positive subnormal


def f64_to_bits(val: float) -> int:
    """Convert Python float (IEEE 754 double) to 64-bit unsigned integer bits."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def bits_to_f64(bits: int) -> float:
    """Convert 64-bit unsigned integer bits to Python float (IEEE 754 double)."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def split_f64_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK) >> 63
    exponent = (bits & EXP_MASK) >> 52
    fraction = bits & FRAC_MASK
    return sign, exponent, fraction


def words_to_f64_bits(lo: int, hi: int) -> int:
    """Combine two 32-bit VGPR words (lo, hi) into a 64-bit integer representation."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def f64_bits_to_words(bits: int) -> Tuple[int, int]:
    """Split a 64-bit integer into two 32-bit VGPR words (lo, hi)."""
    lo = bits & MASK32
    hi = (bits >> 32) & MASK32
    return lo, hi


def is_nan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 NaN."""
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
    """Return True if bits represent a subnormal (denormal) double-precision float."""
    return ((bits & EXP_MASK) == 0) and ((bits & FRAC_MASK) != 0)


def is_zero(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK) == 0


def apply_f64_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & MASK64
    if abs_mod:
        res &= ~SIGN_MASK
    if neg_mod:
        res ^= SIGN_MASK
    return res


def oracle_v_rsq_f64(
    src_bits: int,
    abs_src: bool = False,
    neg_src: bool = False,
    clamp: bool = False,
    flush_subnormals: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_rsq_f64.

    Computes D = 1.0 / sqrt(src) according to IEEE 754-2008 / RDNA2 VALU rules:
      - abs_src / neg_src input modifiers applied before root calculation.
      - Flush-to-zero flushes subnormal input to signed zero.
      - NaN input produces Quiet NaN (or +0.0 if clamp is enabled).
      - Negative non-zero and -Infinity inputs produce Canonical Quiet NaN (or +0.0 if clamp is enabled).
      - +0.0 and -0.0 produce +Infinity (clamped to 1.0 if clamp is enabled).
      - +Infinity produces +0.0.
      - Finite positive numbers produce 1.0 / sqrt(x), clamped to [0.0, 1.0] if clamp is enabled.
    """
    s = apply_f64_modifiers(src_bits, abs_src, neg_src)

    # 1. Subnormal flush-to-zero mode
    if flush_subnormals and is_subnormal(s):
        s = NEG_ZERO if (s & SIGN_MASK) else POS_ZERO

    # 2. NaN inputs propagate quieted NaN
    if is_nan(s):
        if clamp:
            return POS_ZERO
        return s | QNAN_BIT

    # 3. Negative non-zero or negative infinity yields Quiet NaN
    if (s & SIGN_MASK) and not is_zero(s):
        if clamp:
            return POS_ZERO
        return CANONICAL_QNAN

    # 4. Zero inputs (+0.0, -0.0) produce +Infinity
    if is_zero(s):
        if clamp:
            return ONE_F64
        return POS_INF

    # 5. Positive Infinity yields +0.0
    if s == POS_INF:
        return POS_ZERO

    # 6. Finite positive calculation
    val = bits_to_f64(s)
    res_val = 1.0 / math.sqrt(val)

    if clamp:
        if res_val < 0.0:
            res_val = 0.0
        elif res_val > 1.0:
            res_val = 1.0

    return f64_to_bits(res_val)


class TestF64ModelAndOracle(unittest.TestCase):
    """Unit tests for IEEE 754 binary64 bit functions and the reference v_rsq_f64 oracle."""

    def test_exact_powers_of_four_and_two(self) -> None:
        """Exact square roots and reciprocals must be bit-exact in binary64."""
        self.assertEqual(oracle_v_rsq_f64(ONE_F64), ONE_F64)
        self.assertEqual(oracle_v_rsq_f64(FOUR_F64), HALF_F64)
        self.assertEqual(oracle_v_rsq_f64(SIXTEEN_F64), QUARTER_F64)
        self.assertEqual(oracle_v_rsq_f64(SIXTY_FOUR_F64), f64_to_bits(0.125))
        self.assertEqual(oracle_v_rsq_f64(QUARTER_F64), TWO_F64)
        self.assertEqual(oracle_v_rsq_f64(SIXTEENTH_F64), FOUR_F64)

    def test_positive_finite_non_integers(self) -> None:
        """Test reciprocal square root for general positive floating point values."""
        res_nine = bits_to_f64(oracle_v_rsq_f64(NINE_F64))
        self.assertAlmostEqual(res_nine, 1.0 / 3.0, places=15)

        res_hundred = bits_to_f64(oracle_v_rsq_f64(HUNDRED_F64))
        self.assertAlmostEqual(res_hundred, 0.1, places=15)

        res_two = bits_to_f64(oracle_v_rsq_f64(TWO_F64))
        self.assertAlmostEqual(res_two, 1.0 / math.sqrt(2.0), places=15)

    def test_zero_inputs_produce_positive_infinity(self) -> None:
        """Both +0.0 and -0.0 yield +Infinity in AMD RDNA2 RSQ semantics."""
        self.assertEqual(oracle_v_rsq_f64(POS_ZERO), POS_INF)
        self.assertEqual(oracle_v_rsq_f64(NEG_ZERO), POS_INF)

    def test_positive_infinity_yields_positive_zero(self) -> None:
        """1.0 / sqrt(+Inf) yields +0.0."""
        self.assertEqual(oracle_v_rsq_f64(POS_INF), POS_ZERO)

    def test_negative_inputs_produce_canonical_qnan(self) -> None:
        """Square root of negative real numbers produces Quiet NaN."""
        self.assertEqual(oracle_v_rsq_f64(NEG_ONE_F64), CANONICAL_QNAN)
        self.assertEqual(oracle_v_rsq_f64(NEG_FOUR_F64), CANONICAL_QNAN)
        self.assertEqual(oracle_v_rsq_f64(NEG_QUARTER_F64), CANONICAL_QNAN)
        self.assertEqual(oracle_v_rsq_f64(NEG_INF), CANONICAL_QNAN)

    def test_nan_inputs_propagate_quieted(self) -> None:
        """NaN operands propagate quieted NaN."""
        qnan_in = CANONICAL_QNAN | 0x1234
        snan_in = POS_INF | 0x0001_0000  # Signaling NaN
        self.assertTrue(is_qnan(oracle_v_rsq_f64(qnan_in)))
        self.assertTrue(is_qnan(oracle_v_rsq_f64(snan_in)))

    def test_input_modifiers_abs_and_neg(self) -> None:
        """Input modifiers abs and neg transform operand before root calculation."""
        # abs(-4.0) -> +4.0 -> rsq gives 0.5
        self.assertEqual(oracle_v_rsq_f64(NEG_FOUR_F64, abs_src=True), HALF_F64)
        # neg(4.0) -> -4.0 -> rsq gives NaN
        self.assertEqual(oracle_v_rsq_f64(FOUR_F64, neg_src=True), CANONICAL_QNAN)
        # neg(-4.0) -> +4.0 -> rsq gives 0.5
        self.assertEqual(oracle_v_rsq_f64(NEG_FOUR_F64, neg_src=True), HALF_F64)
        # abs and neg together on positive: neg(abs(4.0)) -> -4.0 -> NaN
        self.assertEqual(oracle_v_rsq_f64(FOUR_F64, abs_src=True, neg_src=True), CANONICAL_QNAN)

    def test_clamp_modifier(self) -> None:
        """Clamp modifier saturates result to [0.0, 1.0]."""
        # 0.25 -> 2.0 -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f64(QUARTER_F64, clamp=True), ONE_F64)
        # 4.0 -> 0.5 -> within [0.0, 1.0] -> unchanged
        self.assertEqual(oracle_v_rsq_f64(FOUR_F64, clamp=True), HALF_F64)
        # +0.0 -> +inf -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f64(POS_ZERO, clamp=True), ONE_F64)
        # Negative -> NaN -> clamped to +0.0
        self.assertEqual(oracle_v_rsq_f64(NEG_FOUR_F64, clamp=True), POS_ZERO)
        # NaN -> clamped to +0.0
        self.assertEqual(oracle_v_rsq_f64(CANONICAL_QNAN, clamp=True), POS_ZERO)

    def test_subnormal_handling(self) -> None:
        """Subnormal precision preservation in IEEE mode and flush in FTZ mode."""
        sub = SUBNORM_MAX
        res_ieee = oracle_v_rsq_f64(sub, flush_subnormals=False)
        self.assertFalse(is_nan(res_ieee))
        self.assertFalse(is_inf(res_ieee))
        self.assertGreater(bits_to_f64(res_ieee), 1.0)

        # In FTZ mode, subnormal flushes to +0.0, so rsq gives +Infinity
        res_ftz = oracle_v_rsq_f64(sub, flush_subnormals=True)
        self.assertEqual(res_ftz, POS_INF)

    def test_word_conversion_helpers(self) -> None:
        """Splitting and joining 32-bit dwords must roundtrip 64-bit values accurately."""
        lo, hi = f64_bits_to_words(FOUR_F64)
        rejoined = words_to_f64_bits(lo, hi)
        self.assertEqual(rejoined, FOUR_F64)

        sign, exp, frac = split_f64_bits(FOUR_F64)
        self.assertEqual(sign, 0)
        self.assertEqual(exp, 1025)
        self.assertEqual(frac, 0)


if __name__ == "__main__":
    unittest.main()
