from __future__ import annotations

from fractions import Fraction
import math
import struct
import unittest

# IEEE 754 binary64 (double precision) architectural constants
SIGN_MASK_F64: int = 0x8000_0000_0000_0000
EXP_MASK_F64: int = 0x7FF0_0000_0000_0000
FRAC_MASK_F64: int = 0x000F_FFFF_FFFF_FFFF
QNAN_BIT_F64: int = 0x0008_0000_0000_0000
CANONICAL_QNAN_F64: int = 0x7FF8_0000_0000_0000

POS_ZERO_F64: int = 0x0000_0000_0000_0000
NEG_ZERO_F64: int = 0x8000_0000_0000_0000
POS_INF_F64: int = 0x7FF0_0000_0000_0000
NEG_INF_F64: int = 0xFFF0_0000_0000_0000

ONE_F64: int = 0x3FF0_0000_0000_0000
# Largest binary64 strictly below 1.0: 1.0 - 2^-53
LARGEST_BELOW_ONE_F64: int = 0x3FEF_FFFF_FFFF_FFFF


def f64_to_bits(val: float) -> int:
    """Convert a Python float (IEEE 754 double) to 64-bit integer bits."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def bits_to_f64(bits: int) -> float:
    """Convert 64-bit integer bits to a Python float (IEEE 754 double)."""
    return struct.unpack("<d", struct.pack("<Q", bits & 0xFFFF_FFFF_FFFF_FFFF))[0]


def split_f64_bits(bits: int) -> tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK_F64) >> 63
    exponent = (bits & EXP_MASK_F64) >> 52
    fraction = bits & FRAC_MASK_F64
    return sign, exponent, fraction


def words_to_f64_bits(lo: int, hi: int) -> int:
    """Combine two 32-bit VGPR words (lo, hi) into a 64-bit integer representation."""
    return ((hi & 0xFFFF_FFFF) << 32) | (lo & 0xFFFF_FFFF)


def f64_bits_to_words(bits: int) -> tuple[int, int]:
    """Split a 64-bit integer into two 32-bit VGPR words (lo, hi)."""
    lo = bits & 0xFFFF_FFFF
    hi = (bits >> 32) & 0xFFFF_FFFF
    return lo, hi


def is_nan_f64(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 NaN."""
    return ((bits & EXP_MASK_F64) == EXP_MASK_F64) and ((bits & FRAC_MASK_F64) != 0)


def is_qnan_f64(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Quiet NaN."""
    return is_nan_f64(bits) and bool(bits & QNAN_BIT_F64)


def is_snan_f64(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Signaling NaN."""
    return is_nan_f64(bits) and not bool(bits & QNAN_BIT_F64)


def is_inf_f64(bits: int) -> bool:
    """Return True if bits represent IEEE 754 positive or negative infinity."""
    return ((bits & EXP_MASK_F64) == EXP_MASK_F64) and ((bits & FRAC_MASK_F64) == 0)


def is_subnormal_f64(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormal) double-precision float."""
    return ((bits & EXP_MASK_F64) == 0) and ((bits & FRAC_MASK_F64) != 0)


def is_zero_f64(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK_F64) == 0


def apply_f64_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & 0xFFFF_FFFF_FFFF_FFFF
    if abs_mod:
        res &= ~SIGN_MASK_F64
    if neg_mod:
        res ^= SIGN_MASK_F64
    return res


def f64_to_fraction(bits: int) -> Fraction:
    """Decode finite IEEE 754 binary64 bit pattern into an exact Fraction."""
    bits &= 0xFFFF_FFFF_FFFF_FFFF
    exp = (bits & EXP_MASK_F64) >> 52
    if exp == 0x7FF:
        raise ValueError(f"non-finite binary64 {bits:#018x}")
    sign = -1 if (bits & SIGN_MASK_F64) else 1
    mant = bits & FRAC_MASK_F64
    if exp == 0:
        return Fraction(0) if mant == 0 else Fraction(sign * mant, 1 << 1074)
    return Fraction(sign * ((1 << 52) | mant)) * (Fraction(2) ** (exp - 1075))


def fraction_to_f64_bits(q: Fraction) -> int:
    """Round an exact Fraction to IEEE 754 binary64 (round-to-nearest-even)."""
    if q == 0:
        return POS_ZERO_F64
    sgn = SIGN_MASK_F64 if q < 0 else 0
    a = abs(q)
    e = max(a.numerator.bit_length() - a.denominator.bit_length() - 1, -1022)
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    while Fraction(2) ** e > a and e > -1022:
        e -= 1
    if e == -1022 and a < Fraction(2) ** -1022:
        m = a / (Fraction(2) ** -1074)
        n = m.numerator // m.denominator
        rem = m - n
        if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
            n += 1
        if n >= (1 << 52):
            return sgn | (1 << 52)
        return sgn | n
    else:
        m = a / (Fraction(2) ** (e - 52))
        n = m.numerator // m.denominator
        rem = m - n
        if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
            n += 1
        if n >= (1 << 53):
            n >>= 1
            e += 1
        if e > 1023:
            return sgn | POS_INF_F64
        biased_exp = e + 1023
        return sgn | (biased_exp << 52) | (n - (1 << 52))


def exact_fract_f64_fraction(bits: int) -> Fraction | None:
    """Exact x - floor(x) as Fraction in [0, 1), or None for non-finite values."""
    if is_nan_f64(bits) or is_inf_f64(bits):
        return None
    q = f64_to_fraction(bits)
    return q - math.floor(q)


def oracle_v_fract_f64(
    src0_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_fract_f64.

    Computes D.f64 = S0.f64 - floor(S0.f64) under IEEE 754-2008 double precision:
    - Input modifiers: abs applied before neg (-|s0|).
    - Signed zero: fract(-0.0) -> +0.0; fract(+0.0) -> +0.0.
    - Exact integers (1.0, -1.0, 2.0, etc.) -> +0.0.
    - Infinities: fract(+-inf) -> Canonical Quiet NaN.
    - NaNs: signaling NaNs are quieted; quiet NaNs preserve payload.
    - Rounding boundary: a result that rounds up to 1.0 (tiny negative numbers)
      is clamped to 0x3FEFFFFFFFFFFFFF (largest binary64 below 1.0), matching
      hardware behavior documented in docs/design.md.
    - Clamp modifier: clamps result to [0.0, 1.0]. Under clamp, NaN produces +0.0.
    """
    s0 = apply_f64_modifiers(src0_bits, abs_mod=abs0, neg_mod=neg0)

    # 1. NaN handling
    if is_nan_f64(s0):
        if clamp:
            return POS_ZERO_F64
        return s0 | QNAN_BIT_F64

    # 2. Infinity handling
    if is_inf_f64(s0):
        if clamp:
            return POS_ZERO_F64
        return CANONICAL_QNAN_F64

    # 3. Signed zero handling
    if is_zero_f64(s0):
        return POS_ZERO_F64

    # 4. Exact fraction calculation
    frac = exact_fract_f64_fraction(s0)
    if frac is None or frac == 0:
        return POS_ZERO_F64

    res = fraction_to_f64_bits(frac)

    # 5. Hardware clamp: any result that rounded up to 1.0 becomes largest double < 1.0
    if res == ONE_F64:
        res = LARGEST_BELOW_ONE_F64

    # 6. Clamp modifier ([0.0, 1.0])
    if clamp:
        # Since frac is strictly in [0, 1), finite result is already in [0, 1]
        pass

    return res


# ==============================================================================
# Unit Tests: Reference Oracle and Architectural Model
# ==============================================================================

class TestValuFractF64ModelAndOracle(unittest.TestCase):
    """Unit tests for binary64 bit model, predicates, and reference oracle."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 1e-100, 1e100, math.pi]
        for val in values:
            with self.subTest(val=val):
                bits = f64_to_bits(val)
                recovered = bits_to_f64(bits)
                if math.isnan(val):
                    self.assertTrue(math.isnan(recovered))
                else:
                    self.assertEqual(val, recovered)
                    self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_word_packing_vgpr_pair(self) -> None:
        val = 1234567.89012345
        bits = f64_to_bits(val)
        lo, hi = f64_bits_to_words(bits)
        self.assertEqual(lo, bits & 0xFFFF_FFFF)
        self.assertEqual(hi, (bits >> 32) & 0xFFFF_FFFF)
        combined = words_to_f64_bits(lo, hi)
        self.assertEqual(combined, bits)
        self.assertEqual(bits_to_f64(combined), val)

    def test_classification_predicates(self) -> None:
        self.assertTrue(f64_to_bits(0.0) == POS_ZERO_F64)
        self.assertTrue(f64_to_bits(-0.0) == NEG_ZERO_F64)
        self.assertTrue(is_zero_f64(POS_ZERO_F64))
        self.assertTrue(is_zero_f64(NEG_ZERO_F64))
        self.assertFalse(is_zero_f64(f64_to_bits(1.0)))

        self.assertTrue(is_inf_f64(POS_INF_F64))
        self.assertTrue(is_inf_f64(NEG_INF_F64))
        self.assertFalse(is_inf_f64(POS_ZERO_F64))

        snan_bits = 0x7FF0_0000_0000_0001
        qnan_bits = CANONICAL_QNAN_F64
        self.assertTrue(is_nan_f64(snan_bits))
        self.assertTrue(is_snan_f64(snan_bits))
        self.assertFalse(is_qnan_f64(snan_bits))

        self.assertTrue(is_nan_f64(qnan_bits))
        self.assertTrue(is_qnan_f64(qnan_bits))
        self.assertFalse(is_snan_f64(qnan_bits))

        subnorm_bits = 0x0000_0000_0000_0001
        self.assertTrue(is_subnormal_f64(subnorm_bits))
        self.assertFalse(is_subnormal_f64(POS_ZERO_F64))
        self.assertFalse(is_subnormal_f64(f64_to_bits(1.0)))

    def test_exact_finite_fraction(self) -> None:
        # fract(1.25) == 0.25
        res = oracle_v_fract_f64(f64_to_bits(1.25))
        self.assertEqual(bits_to_f64(res), 0.25)

        # fract(3.75) == 0.75
        res = oracle_v_fract_f64(f64_to_bits(3.75))
        self.assertEqual(bits_to_f64(res), 0.75)

        # fract(-0.25) == 0.75
        res = oracle_v_fract_f64(f64_to_bits(-0.25))
        self.assertEqual(bits_to_f64(res), 0.75)

        # fract(-1.25) == 0.75
        res = oracle_v_fract_f64(f64_to_bits(-1.25))
        self.assertEqual(bits_to_f64(res), 0.75)

        # fract(-0.75) == 0.25
        res = oracle_v_fract_f64(f64_to_bits(-0.75))
        self.assertEqual(bits_to_f64(res), 0.25)

    def test_signed_zero_and_exact_integers(self) -> None:
        # fract(+0.0) -> +0.0
        self.assertEqual(oracle_v_fract_f64(POS_ZERO_F64), POS_ZERO_F64)
        # fract(-0.0) -> +0.0 (not -0.0)
        self.assertEqual(oracle_v_fract_f64(NEG_ZERO_F64), POS_ZERO_F64)

        for integer_val in [1.0, -1.0, 2.0, -2.0, 10.0, -10.0, 1024.0, -1024.0, 65536.0]:
            with self.subTest(integer_val=integer_val):
                res = oracle_v_fract_f64(f64_to_bits(integer_val))
                self.assertEqual(res, POS_ZERO_F64)

    def test_rounding_to_one_clamped_to_largest_below_one(self) -> None:
        # Tiny negative inputs whose exact x - floor(x) rounds to 1.0 in binary64
        # must clamp to LARGEST_BELOW_ONE_F64 (0x3FEFFFFFFFFFFFFF)
        tiny_negatives = [
            -1e-16,
            -1e-18,
            -1e-25,
            -2.0 ** -54,
            -2.0 ** -60,
            -2.0 ** -100,
            -2.0 ** -1074,  # negative min subnormal
        ]
        for val in tiny_negatives:
            with self.subTest(val=val):
                bits = f64_to_bits(val)
                res = oracle_v_fract_f64(bits)
                self.assertEqual(res, LARGEST_BELOW_ONE_F64)

    def test_input_modifiers(self) -> None:
        # abs: fract(|-1.25|) = fract(1.25) = 0.25
        res_abs = oracle_v_fract_f64(f64_to_bits(-1.25), abs0=True)
        self.assertEqual(bits_to_f64(res_abs), 0.25)

        # neg: fract(-(1.25)) = fract(-1.25) = 0.75
        res_neg = oracle_v_fract_f64(f64_to_bits(1.25), neg0=True)
        self.assertEqual(bits_to_f64(res_neg), 0.75)

        # abs before neg: -|1.25| = -1.25 -> 0.75; -|-1.25| = -1.25 -> 0.75
        res_neg_abs1 = oracle_v_fract_f64(f64_to_bits(1.25), abs0=True, neg0=True)
        res_neg_abs2 = oracle_v_fract_f64(f64_to_bits(-1.25), abs0=True, neg0=True)
        self.assertEqual(bits_to_f64(res_neg_abs1), 0.75)
        self.assertEqual(bits_to_f64(res_neg_abs2), 0.75)

    def test_clamp_modifier(self) -> None:
        # Normal positive fractions stay unchanged
        res = oracle_v_fract_f64(f64_to_bits(1.25), clamp=True)
        self.assertEqual(bits_to_f64(res), 0.25)

        # NaN with clamp becomes +0.0
        res_nan = oracle_v_fract_f64(CANONICAL_QNAN_F64, clamp=True)
        self.assertEqual(res_nan, POS_ZERO_F64)

        # Inf with clamp becomes +0.0
        res_inf = oracle_v_fract_f64(POS_INF_F64, clamp=True)
        self.assertEqual(res_inf, POS_ZERO_F64)

    def test_infinities_and_nans(self) -> None:
        # fract(+-inf) -> Canonical Quiet NaN
        self.assertEqual(oracle_v_fract_f64(POS_INF_F64), CANONICAL_QNAN_F64)
        self.assertEqual(oracle_v_fract_f64(NEG_INF_F64), CANONICAL_QNAN_F64)

        # Signaling NaN quietened
        snan = 0x7FF0_0000_1234_5678
        res_snan = oracle_v_fract_f64(snan)
        self.assertTrue(is_qnan_f64(res_snan))
        self.assertEqual(res_snan, snan | QNAN_BIT_F64)

        # Quiet NaN payload preserved
        qnan = 0x7FF8_0000_ABCD_EF01
        res_qnan = oracle_v_fract_f64(qnan)
        self.assertEqual(res_qnan, qnan)


if __name__ == "__main__":
    unittest.main()
