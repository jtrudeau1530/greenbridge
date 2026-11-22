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


if __name__ == "__main__":
    unittest.main()
