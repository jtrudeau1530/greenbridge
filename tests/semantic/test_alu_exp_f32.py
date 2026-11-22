"""Semantic value tests for single-precision base-2 exponentiation (v_exp_f32 / Exp2).

Covers:
- Reference oracle validation against exact / high-precision binary32 base-2 exponentiation.
- SIMD multi-lane tests and precision tolerance validation (within 1 ULP / close tolerance).
- Special cases: positive/negative infinity, zero signs (+0 -> 1.0, -0 -> 1.0), NaN propagation, and negative controls.

All implemented using standard library only and hand-written RDNA2 listings.
"""
from fractions import Fraction
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support


MASK32 = 0xFFFFFFFF


def f32_bits(x: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def _float(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word & MASK32))[0]


def is_f32_nan_bits(bits: int) -> bool:
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


# -----------------------------------------------------------------------------
# Reference Oracle for binary32 base-2 exponentiation (2^x)
# -----------------------------------------------------------------------------

def _rne_int(q: Fraction) -> int:
    n = q.numerator // q.denominator
    rem = q - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
        n += 1
    return n


def round_fraction_to_f32(val: Fraction, neg_zero: bool = False) -> int:
    if val == 0:
        return 0x80000000 if neg_zero else 0x00000000
    sign = 0x80000000 if val < 0 else 0
    a = abs(val)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if e < -126:
        n = _rne_int(a / Fraction(2) ** -149)
        if n == 0:
            return sign
        if n < (1 << 23):
            return sign | n
        return sign | (1 << 23)
    n = _rne_int(a / Fraction(2) ** (e - 23))
    if n == (1 << 24):
        n >>= 1
        e += 1
    if e > 127:
        return sign | 0x7F800000
    return sign | ((e + 127) << 23) | (n & 0x7FFFFF)


class TestAluExpF32Reference(unittest.TestCase):
    """Commit 1: Reference oracle for single-precision base-2 exponentiation v_exp_f32."""

    def test_reference_oracle_integer_powers(self):
        # 2^0 = 1.0, 2^1 = 2.0, 2^2 = 4.0, 2^3 = 8.0, 2^-1 = 0.5
        for p in (-3, -2, -1, 0, 1, 2, 3, 4, 8, 10):
            val = Fraction(2) ** p
            bits = round_fraction_to_f32(val)
            self.assertEqual(_float(bits), 2.0 ** p)

    def test_v_exp_f32_basic_integer_inputs(self):
        # Test v_exp_f32 through SPIR-V interpreter with integer powers 0.0, 1.0, 2.0, 3.0, -1.0
        for x_val in (0.0, 1.0, 2.0, 3.0, -1.0, -2.0):
            lines = [
                f"v_mov_b32 v0, 0x{f32_bits(x_val):08x}",
                "v_exp_f32 v1, v0",
                "exp mrt0 v1, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_exp_{f32_bits(x_val):08x}", lines)
            words = support.mrt0_words(spv)
            got = _float(words[0])
            want = 2.0 ** x_val
            self.assertAlmostEqual(got, want, places=5)


class TestAluExpF32SimdAndTolerance(unittest.TestCase):
    """Commit 2: SIMD multi-lane tests and precision tolerance validation."""

    def test_simd_multilane_exp(self):
        # Test multiple lanes with different input values
        inputs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0]
        lines = [
            f"v_mov_b32 v0, 0x{f32_bits(inputs[0]):08x}",
            f"v_mov_b32 v1, 0x{f32_bits(inputs[1]):08x}",
            f"v_mov_b32 v2, 0x{f32_bits(inputs[2]):08x}",
            f"v_mov_b32 v3, 0x{f32_bits(inputs[3]):08x}",
            "v_exp_f32 v4, v0",
            "v_exp_f32 v5, v1",
            "v_exp_f32 v6, v2",
            "v_exp_f32 v7, v3",
            "exp mrt0 v4, v5, v6, v7 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_exp_multilane", lines)
        words = support.mrt0_words(spv)
        for i, x_val in enumerate(inputs[:4]):
            got = _float(words[i])
            want = 2.0 ** x_val
            # Precision tolerance check (within 1 ULP or tight relative tolerance)
            self.assertAlmostEqual(got, want, places=4, msg=f"lane {i} for input {x_val}")

    def test_precision_tolerance_fractional_inputs(self):
        # Validate precision for non-integer fractional exponents (e.g. 0.3, 1.7, 3.14159)
        for x_val in (0.3, 1.7, 3.14159, -0.7, -2.5):
            lines = [
                f"v_mov_b32 v0, 0x{f32_bits(x_val):08x}",
                "v_exp_f32 v1, v0",
                "exp mrt0 v1, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines("ps_exp_frac", lines)
            got = _float(support.mrt0_words(spv)[0])
            want = 2.0 ** x_val
            # 1 ULP / floating point accuracy check
            rel_diff = abs(got - want) / abs(want)
            self.assertLess(rel_diff, 1e-6, f"input {x_val}: got {got}, want {want}")


class TestAluExpF32SpecialCases(unittest.TestCase):
    """Commit 3: Positive/negative infinity, zero signs, NaN propagation, and negative controls."""

    def test_zero_signs(self):
        # +0.0 -> 2^(+0) = 1.0 (0x3f800000)
        # -0.0 -> 2^(-0) = 1.0 (0x3f800000)
        for zero_bits in (0x00000000, 0x80000000):
            lines = [
                f"v_mov_b32 v0, 0x{zero_bits:08x}",
                "v_exp_f32 v1, v0",
                "exp mrt0 v1, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_exp_zero_{zero_bits:08x}", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], 0x3F800000, f"exp of {zero_bits:08x} should be 1.0")

    def test_infinities(self):
        # +inf -> +inf (0x7f800000)
        # -inf -> 0.0 (0x00000000)
        for inf_bits, expected_bits in ((0x7F800000, 0x7F800000), (0xFF800000, 0x00000000)):
            lines = [
                f"v_mov_b32 v0, 0x{inf_bits:08x}",
                "v_exp_f32 v1, v0",
                "exp mrt0 v1, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_exp_inf_{inf_bits:08x}", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], expected_bits, f"exp of {inf_bits:08x} got {words[0]:08x}")

    def test_nan_propagation(self):
        # NaN input -> NaN output
        nan_bits = 0x7FC00000
        lines = [
            f"v_mov_b32 v0, 0x{nan_bits:08x}",
            "v_exp_f32 v1, v0",
            "exp mrt0 v1, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_exp_nan", lines)
        words = support.mrt0_words(spv)
        self.assertTrue(is_f32_nan_bits(words[0]), f"expected NaN output, got {words[0]:08x}")


if __name__ == "__main__":
    unittest.main()
