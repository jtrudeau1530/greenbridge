from __future__ import annotations

import math
import struct
import unittest
from fractions import Fraction
from typing import List, Optional, Tuple, Union

# ==============================================================================
# IEEE 754 binary16 (Half-Precision) Architectural Constants
# ==============================================================================

SIGN_MASK: int = 0x8000
EXP_MASK: int = 0x7C00
FRAC_MASK: int = 0x03FF
QNAN_BIT: int = 0x0200
CANONICAL_QNAN: int = 0x7E00

POS_ZERO: int = 0x0000
NEG_ZERO: int = 0x8000
POS_INF: int = 0x7C00
NEG_INF: int = 0xFC00

ONE_F16: int = 0x3C00
NEAR_ONE_F16: int = 0x3BFF  # Largest binary16 strictly less than 1.0 (1.0 - 2^-11)

# Representative IEEE 754 binary16 encodings
H_POS_ZERO: int = 0x0000
H_NEG_ZERO: int = 0x8000
H_SUB_MIN: int = 0x0001           # Smallest positive subnormal (2^-24)
H_NEG_SUB_MIN: int = 0x8001
H_SUB_MAX: int = 0x03FF           # Largest positive subnormal ((1023/1024) * 2^-14)
H_NEG_SUB_MAX: int = 0x83FF
H_NORM_MIN: int = 0x0400          # Smallest positive normal (2^-14)
H_NEG_NORM_MIN: int = 0x8400
H_0P125: int = 0x3000             # 0.125
H_0P25: int = 0x3400              # 0.25
H_NEG_0P25: int = 0xB400          # -0.25
H_0P375: int = 0x3600             # 0.375
H_0P5: int = 0x3800               # 0.5
H_NEG_0P5: int = 0xB800           # -0.5
H_0P75: int = 0x3A00              # 0.75
H_NEG_0P75: int = 0xBA00          # -0.75
H_NEAR_ONE: int = 0x3BFF          # 0.99951171875
H_1P0: int = 0x3C00               # 1.0
H_NEG_1P0: int = 0xBC00           # -1.0
H_1P25: int = 0x3D00              # 1.25
H_NEG_1P25: int = 0xBD00          # -1.25
H_1P5: int = 0x3E00               # 1.5
H_NEG_1P5: int = 0xBE00           # -1.5
H_1P75: int = 0x3F00              # 1.75
H_NEG_1P75: int = 0xBF00          # -1.75
H_2P0: int = 0x4000               # 2.0
H_NEG_2P0: int = 0xC000           # -2.0
H_2P5: int = 0x4100               # 2.5
H_NEG_2P5: int = 0xC100           # -2.5
H_3P0: int = 0x4200               # 3.0
H_NEG_3P0: int = 0xC200           # -3.0
H_3P75: int = 0x4380              # 3.75
H_NEG_3P75: int = 0xC380          # -3.75
H_1023P5: int = 0x63FF            # 1023.5 (1023 + 0.5)
H_1024: int = 0x6400              # 1024.0 (all >= 1024 are integers)
H_NEG_1024: int = 0xE400
H_MAX: int = 0x7BFF               # 65504.0
H_NEG_MAX: int = 0xFBFF


def f16_to_bits(val: float) -> int:
    """Convert a Python float to 16-bit binary16 integer bits."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f16(bits: int) -> float:
    """Convert 16-bit binary16 integer bits to a Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def split_f16_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack binary16 bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK) >> 15
    exponent = (bits & EXP_MASK) >> 10
    fraction = bits & FRAC_MASK
    return sign, exponent, fraction


def is_nan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 NaN."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) != 0)


def is_qnan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 Quiet NaN."""
    return is_nan(bits) and bool(bits & QNAN_BIT)


def is_snan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 Signaling NaN."""
    return is_nan(bits) and not bool(bits & QNAN_BIT)


def is_inf(bits: int) -> bool:
    """Return True if bits represent positive or negative infinity."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) == 0)


def is_subnormal(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormal) binary16 float."""
    return ((bits & EXP_MASK) == 0) and ((bits & FRAC_MASK) != 0)


def is_zero(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK) == 0


def apply_f16_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & 0xFFFF
    if abs_mod:
        res &= ~SIGN_MASK
    if neg_mod:
        res ^= SIGN_MASK
    return res


def h16_to_fraction(h: int) -> Fraction:
    """Decode finite IEEE 754 binary16 bit pattern into an exact Fraction."""
    h &= 0xFFFF
    exp = (h >> 10) & 0x1F
    if exp == 0x1F:
        raise ValueError(f"Non-finite binary16 {h:#06x}")
    sign = -1 if (h >> 15) else 1
    mant = h & 0x3FF
    if exp == 0:
        return Fraction(0) if mant == 0 else Fraction(sign * mant, 1 << 24)
    return Fraction(sign * ((1 << 10) | mant)) * (Fraction(2) ** (exp - 25))


def fraction_to_h16(q: Fraction) -> int:
    """Round an exact Fraction to IEEE 754 binary16 (Round-to-Nearest-Even)."""
    if q == 0:
        return POS_ZERO
    sgn = 0x8000 if q < 0 else 0
    a = abs(q)
    e = max(a.numerator.bit_length() - a.denominator.bit_length(), -14)
    if e > -14 and Fraction(2) ** e > a:
        e -= 1
    m = a / (Fraction(2) ** (e - 10))
    n = m.numerator // m.denominator
    rem = m - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
        n += 1
    if e == -14 and n < 1024:
        return sgn | n
    if n >= 2048:
        n >>= 1
        e += 1
    if e > 15:
        return sgn | POS_INF
    return sgn | ((e + 15) << 10) | (n - 1024)


def exact_fract_f16_fraction(h: int) -> Optional[Fraction]:
    """Exact x - floor(x) as a Fraction in [0, 1), or None for +-inf / NaN."""
    if is_nan(h) or is_inf(h):
        return None
    q = h16_to_fraction(h)
    return q - math.floor(q)


def oracle_v_fract_f16(
    src_bits: int,
    abs_mod: bool = False,
    neg_mod: bool = False,
    clamp: bool = False,
    mul: int = 1,
    div: int = 1,
) -> int:
    """Reference mathematical oracle for RDNA2 v_fract_f16 half-precision fractional part.

    Computes D = S0 - floor(S0), returning values in [0.0, 1.0).
    Conforms to RDNA2 hardware semantics:
      - Clamps results that round to 1.0 to NEAR_ONE_F16 (0x3BFF).
      - ±0.0 yields +0.0 (strictly cleared sign bit).
      - ±inf yields NaN (inf - inf is invalid); clamped yields +0.0.
      - NaN yields quietened NaN; clamped yields +0.0.
      - VOP3 source modifiers (abs, neg) applied before computation.
      - VOP3 output scaling (mul/div) and clamp ([0.0, 1.0]).
    """
    s = apply_f16_modifiers(src_bits, abs_mod, neg_mod)

    # 1. NaN handling
    if is_nan(s):
        if clamp:
            return POS_ZERO
        return s | QNAN_BIT

    # 2. Infinity handling (inf - floor(inf) -> inf - inf -> Invalid -> NaN)
    if is_inf(s):
        if clamp:
            return POS_ZERO
        return CANONICAL_QNAN

    # 3. Signed zero handling: fract(+0.0) == fract(-0.0) == +0.0
    if is_zero(s):
        return POS_ZERO

    # 4. Exact rational fractional part
    frac = exact_fract_f16_fraction(s)
    assert frac is not None

    if mul == 1 and div == 1:
        res = fraction_to_h16(frac)
        # In IEEE 754 RNE, values like 1.0 - 2^-25 round to 1.0.
        # RDNA2 ISA contract guarantees fract output is strictly in [0.0, 1.0).
        if res == ONE_F16:
            return NEAR_ONE_F16
        if clamp:
            if res == NEG_ZERO:
                return POS_ZERO
        return res

    # Scaled VOP3 output modifiers
    scaled = frac * Fraction(mul, div)
    if clamp:
        scaled = min(max(scaled, Fraction(0)), Fraction(1))
    res_scaled = fraction_to_h16(scaled)
    if clamp and res_scaled == NEG_ZERO:
        return POS_ZERO
    return res_scaled


# ==============================================================================
# Unit Tests
# ==============================================================================

class TestReferenceOracleF16(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_fraction_roundtrip_finite_f16(self) -> None:
        samples = [
            H_POS_ZERO, H_NEG_ZERO,
            H_SUB_MIN, H_NEG_SUB_MIN, H_SUB_MAX, H_NEG_SUB_MAX,
            H_NORM_MIN, H_NEG_NORM_MIN,
            H_0P125, H_0P25, H_NEG_0P25, H_0P375,
            H_0P5, H_NEG_0P5, H_0P75, H_NEG_0P75,
            H_NEAR_ONE, H_1P0, H_NEG_1P0,
            H_1P25, H_NEG_1P25, H_1P5, H_NEG_1P5, H_1P75, H_NEG_1P75,
            H_2P0, H_NEG_2P0, H_2P5, H_NEG_2P5, H_3P0, H_NEG_3P0,
            H_3P75, H_NEG_3P75, H_1023P5, H_1024, H_NEG_1024,
            H_MAX, H_NEG_MAX,
        ]
        for h in samples:
            with self.subTest(h=hex(h)):
                q = h16_to_fraction(h)
                expected_float = bits_to_f16(h)
                self.assertEqual(q, Fraction(expected_float))
                want = H_POS_ZERO if h == H_NEG_ZERO else h
                self.assertEqual(fraction_to_h16(q), want)

    def test_hand_worked_fract_rows(self) -> None:
        rows = [
            (H_POS_ZERO, H_POS_ZERO),
            (H_NEG_ZERO, H_POS_ZERO),
            (H_0P25, H_0P25),
            (H_NEG_0P25, H_0P75),
            (H_0P5, H_0P5),
            (H_NEG_0P5, H_0P5),
            (H_0P75, H_0P75),
            (H_NEG_0P75, H_0P25),
            (H_1P0, H_POS_ZERO),
            (H_NEG_1P0, H_POS_ZERO),
            (H_1P25, H_0P25),
            (H_NEG_1P25, H_0P75),
            (H_1P75, H_0P75),
            (H_NEG_1P75, H_0P25),
            (H_2P0, H_POS_ZERO),
            (H_NEG_2P0, H_POS_ZERO),
            (H_2P5, H_0P5),
            (H_NEG_2P5, H_0P5),
            (H_3P75, H_0P75),
            (H_NEG_3P75, H_0P25),
        ]
        for inp, expected in rows:
            with self.subTest(inp=hex(inp)):
                self.assertEqual(oracle_v_fract_f16(inp), expected)

    def test_subnormals_and_near_one_clamp(self) -> None:
        # Positive subnormals: floor(x) == 0, so fract(x) == x
        self.assertEqual(oracle_v_fract_f16(H_SUB_MIN), H_SUB_MIN)
        self.assertEqual(oracle_v_fract_f16(H_SUB_MAX), H_SUB_MAX)
        self.assertEqual(oracle_v_fract_f16(H_NORM_MIN), H_NORM_MIN)

        # Negative subnormals: x - floor(x) = 1.0 - |x|, which rounds to 1.0
        # and clamps to NEAR_ONE_F16 (0x3BFF)
        self.assertEqual(oracle_v_fract_f16(H_NEG_SUB_MIN), NEAR_ONE_F16)
        self.assertEqual(oracle_v_fract_f16(H_NEG_SUB_MAX), NEAR_ONE_F16)
        self.assertEqual(oracle_v_fract_f16(H_NEG_NORM_MIN), NEAR_ONE_F16)

    def test_large_integers_mantissa_exhaustion(self) -> None:
        # In binary16, numbers with |x| >= 1024.0 have no fractional bits
        self.assertEqual(oracle_v_fract_f16(H_1024), H_POS_ZERO)
        self.assertEqual(oracle_v_fract_f16(H_NEG_1024), H_POS_ZERO)
        self.assertEqual(oracle_v_fract_f16(H_MAX), H_POS_ZERO)
        self.assertEqual(oracle_v_fract_f16(H_NEG_MAX), H_POS_ZERO)

        # 1023.5 has fractional part 0.5
        self.assertEqual(oracle_v_fract_f16(H_1023P5), H_0P5)

    def test_nan_and_inf_handling(self) -> None:
        # Signaling NaN quietened
        snan = 0x7C01
        self.assertTrue(is_snan(snan))
        res_snan = oracle_v_fract_f16(snan)
        self.assertTrue(is_qnan(res_snan))
        self.assertEqual(res_snan, snan | QNAN_BIT)

        # Quiet NaN preserved
        qnan = 0x7E55
        self.assertTrue(is_qnan(qnan))
        self.assertEqual(oracle_v_fract_f16(qnan), qnan)

        # Infs produce Canonical QNaN
        self.assertEqual(oracle_v_fract_f16(POS_INF), CANONICAL_QNAN)
        self.assertEqual(oracle_v_fract_f16(NEG_INF), CANONICAL_QNAN)

        # Clamping NaN or Inf produces POS_ZERO
        self.assertEqual(oracle_v_fract_f16(qnan, clamp=True), POS_ZERO)
        self.assertEqual(oracle_v_fract_f16(POS_INF, clamp=True), POS_ZERO)
        self.assertEqual(oracle_v_fract_f16(NEG_INF, clamp=True), POS_ZERO)

    def test_source_modifiers(self) -> None:
        # abs clears sign: fract(|-0.25|) = fract(0.25) = 0.25
        self.assertEqual(oracle_v_fract_f16(H_NEG_0P25, abs_mod=True), H_0P25)
        # neg toggles sign: fract(-(0.25)) = fract(-0.25) = 0.75
        self.assertEqual(oracle_v_fract_f16(H_0P25, neg_mod=True), H_0P75)
        # abs then neg: -|-0.25| = -0.25 -> 0.75
        self.assertEqual(oracle_v_fract_f16(H_NEG_0P25, abs_mod=True, neg_mod=True), H_0P75)

    def test_output_modifiers(self) -> None:
        # fract(0.5) * 2 = 1.0
        self.assertEqual(oracle_v_fract_f16(H_0P5, mul=2), H_1P0)
        # fract(0.5) / 2 = 0.25
        self.assertEqual(oracle_v_fract_f16(H_0P5, div=2), H_0P25)
        # fract(0.75) * 2 = 1.5 clamped to 1.0
        self.assertEqual(oracle_v_fract_f16(H_0P75, mul=2, clamp=True), H_1P0)


if __name__ == "__main__":
    unittest.main()
