"""Semantic reference oracle and interpreter tests for RDNA2 packed half-precision dot products:
  - v_dot2acc_f16_f16: D.f16 = S0.f16[0] * S1.f16[0] + S0.f16[1] * S1.f16[1] + S2.f16

AMD RDNA2 VALU Instruction Set Architecture (GFX10):
  The v_dot2acc_f16_f16 instruction computes the dot product of two packed pairs of IEEE-754
  binary16 (half-precision) floating-point numbers from 32-bit vector registers S0 and S1,
  and adds an accumulator S2 into the scalar 16-bit result.
  S0 contains two 16-bit half-precision floats: S0.x (bits [15:0]) and S0.y (bits [31:16]).
  S1 contains two 16-bit half-precision floats: S1.x (bits [15:0]) and S1.y (bits [31:16]).
  S2 contains a 16-bit half-precision accumulator in bits [15:0] (or selected by op_sel).
  Result D is a 16-bit float written to bits [15:0] of the destination register.

Formula:
  D = S0.x * S1.x + S0.y * S1.y + S2

Modifiers:
  - clamp: Saturates the floating-point result to [0.0, 1.0]. Values < 0.0 or -0.0 clamp to
    +0.0, values > 1.0 clamp to 1.0.
  - op_sel / op_sel_hi: Selects which 16-bit half-word of each source is read for the low/high
    product terms and accumulator.
  - neg_lo / neg_hi: Negates the chosen half-words prior to multiplication and addition.

Standard library only: Python 3.9+ compatible, zero game data, zero proprietary assets.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Bit masks
MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# Canonical IEEE-754 binary16 encodings
H_POS_ZERO = 0x0000        # +0.0
H_NEG_ZERO = 0x8000        # -0.0
H_SUB_MIN = 0x0001         # +2^-24 (~5.96046e-8, smallest positive subnormal)
H_NEG_SUB_MIN = 0x8001     # -2^-24
H_SUB_MAX = 0x03FF         # +1023 * 2^-24 (~6.09756e-5, largest subnormal)
H_NEG_SUB_MAX = 0x83FF     # -1023 * 2^-24
H_NORM_MIN = 0x0400        # +2^-14 (~6.10352e-5, smallest positive normal)
H_NEG_NORM_MIN = 0x8400    # -2^-14
H_ONE = 0x3C00             # +1.0
H_NEG_ONE = 0xBC00         # -1.0
H_HALF = 0x3800            # +0.5
H_NEG_HALF = 0xB800        # -0.5
H_TWO = 0x4000             # +2.0
H_THREE = 0x4200           # +3.0
H_FOUR = 0x4400            # +4.0
H_TEN = 0x4900             # +10.0
H_NORM_MAX = 0x7BFF        # +65504.0 (largest finite f16)
H_NEG_NORM_MAX = 0xFBFF    # -65504.0
H_POS_INF = 0x7C00         # +Infinity
H_NEG_INF = 0xFC00         # -Infinity
H_QNAN = 0x7E00            # Quiet NaN
H_SNAN = 0x7D00            # Signaling NaN


def f16_bits_to_float(bits: int) -> float:
    """Convert a 16-bit unsigned integer to an IEEE-754 Python float."""
    bits &= MASK16
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def float_to_f16_bits(val: float) -> int:
    """Convert an IEEE-754 Python float to a 16-bit unsigned integer."""
    if math.isnan(val):
        sign = 0x8000 if math.copysign(1.0, val) < 0 else 0x0000
        return sign | H_QNAN
    if math.isinf(val):
        return H_POS_INF if val > 0 else H_NEG_INF
    # Overflow boundary in IEEE-754 binary16: >= 65520.0 rounds to infinity
    if val >= 65520.0:
        return H_POS_INF
    if val <= -65520.0:
        return H_NEG_INF
    return struct.unpack("<H", struct.pack("<e", val))[0]


def pack_f16x2(lo: int, hi: int) -> int:
    """Pack two 16-bit halfwords into a 32-bit dword."""
    return (lo & MASK16) | ((hi & MASK16) << 16)


def unpack_f16x2(dword: int) -> tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit halfwords (lo, hi)."""
    dword &= MASK32
    return dword & MASK16, (dword >> 16) & MASK16


def is_f16_subnormal(bits: int) -> bool:
    """Check if 16-bit word represents a subnormal float (exp == 0, frac != 0)."""
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    return exp == 0 and frac != 0


def is_f16_nan(bits: int) -> bool:
    """Check if 16-bit word represents any NaN (exp == 31, frac != 0)."""
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    return exp == 0x1F and frac != 0


def is_f16_inf(bits: int) -> bool:
    """Check if 16-bit word represents +/- infinity (exp == 31, frac == 0)."""
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    return exp == 0x1F and frac == 0


def is_f16_zero(bits: int) -> bool:
    """Check if 16-bit word represents +/- 0.0."""
    return (bits & 0x7FFF) == 0


def clamp_f16(val: float) -> float:
    """Apply RDNA VALU float clamp modifier saturating to [0.0, 1.0]."""
    if math.isnan(val):
        return 0.0
    if val < 0.0 or (math.isinf(val) and val < 0):
        return 0.0
    if val > 1.0 or (math.isinf(val) and val > 0):
        return 1.0
    return val


def ref_dot2acc_f16(
    s0: int,
    s1: int,
    s2_acc: int = 0,
    clamp: bool = False,
    op_sel: tuple[int, int, int] = (0, 0, 0),
    op_sel_hi: tuple[int, int, int] = (1, 1, 1),
    neg_lo: tuple[int, int, int] = (0, 0, 0),
    neg_hi: tuple[int, int, int] = (0, 0, 0),
    flush_subnormals: bool = False,
) -> int:
    """Pure-Python reference oracle for RDNA2 v_dot2acc_f16_f16.

    Formula: D = S0.x * S1.x + S0.y * S1.y + S2
    """
    s0_halves = unpack_f16x2(s0)
    s1_halves = unpack_f16x2(s1)

    # First product term: S0.x * S1.x
    h0_x = s0_halves[op_sel[0] & 1]
    h1_x = s1_halves[op_sel[1] & 1]
    if flush_subnormals and is_f16_subnormal(h0_x):
        h0_x = h0_x & 0x8000
    if flush_subnormals and is_f16_subnormal(h1_x):
        h1_x = h1_x & 0x8000

    f0_x = f16_bits_to_float(h0_x)
    f1_x = f16_bits_to_float(h1_x)
    if neg_lo[0]:
        f0_x = -f0_x
    if neg_lo[1]:
        f1_x = -f1_x

    # Second product term: S0.y * S1.y
    h0_y = s0_halves[op_sel_hi[0] & 1]
    h1_y = s1_halves[op_sel_hi[1] & 1]
    if flush_subnormals and is_f16_subnormal(h0_y):
        h0_y = h0_y & 0x8000
    if flush_subnormals and is_f16_subnormal(h1_y):
        h1_y = h1_y & 0x8000

    f0_y = f16_bits_to_float(h0_y)
    f1_y = f16_bits_to_float(h1_y)
    if neg_hi[0]:
        f0_y = -f0_y
    if neg_hi[1]:
        f1_y = -f1_y

    # Accumulator S2
    s2_bits = s2_acc & MASK16
    if flush_subnormals and is_f16_subnormal(s2_bits):
        s2_bits = s2_bits & 0x8000

    f2 = f16_bits_to_float(s2_bits)
    if len(neg_lo) > 2 and neg_lo[2]:
        f2 = -f2

    # Multiplication terms
    p0 = f0_x * f1_x
    p1 = f0_y * f1_y

    dot = p0 + p1 + f2

    if clamp:
        dot = clamp_f16(dot)

    res_bits = float_to_f16_bits(dot)
    if flush_subnormals and is_f16_subnormal(res_bits):
        res_bits = res_bits & 0x8000
    return res_bits


class TestDot2AccF16ReferenceOracle(unittest.TestCase):
    """Exhaustive validation of reference mathematical semantics for v_dot2acc_f16_f16."""

    def test_float_conversion_roundtrip(self):
        """Sanity check f16 <-> float conversions for representative constants."""
        cases = [
            (H_POS_ZERO, 0.0),
            (H_NEG_ZERO, -0.0),
            (H_ONE, 1.0),
            (H_NEG_ONE, -1.0),
            (H_HALF, 0.5),
            (H_TWO, 2.0),
            (H_THREE, 3.0),
            (H_FOUR, 4.0),
            (H_TEN, 10.0),
            (H_NORM_MAX, 65504.0),
            (H_NEG_NORM_MAX, -65504.0),
            (H_POS_INF, float("inf")),
            (H_NEG_INF, float("-inf")),
        ]
        for bits, val in cases:
            with self.subTest(bits=hex(bits), val=val):
                converted = f16_bits_to_float(bits)
                if math.isinf(val):
                    self.assertTrue(math.isinf(converted))
                    self.assertEqual(math.copysign(1.0, converted), math.copysign(1.0, val))
                else:
                    self.assertEqual(converted, val)
                    self.assertEqual(math.copysign(1.0, converted), math.copysign(1.0, val))
                re_bits = float_to_f16_bits(converted)
                self.assertEqual(re_bits, bits)

    def test_basic_dot2acc_calculation(self):
        """Verify normal vectors dot product calculation with accumulator."""
        # [1.0, 2.0] . [3.0, 4.0] + 10.0 = 1*3 + 2*4 + 10 = 21.0
        s0 = pack_f16x2(H_ONE, H_TWO)
        s1 = pack_f16x2(H_THREE, H_FOUR)
        s2 = H_TEN
        res = ref_dot2acc_f16(s0, s1, s2)
        self.assertEqual(f16_bits_to_float(res), 21.0)

        # [0.5, 0.5] . [0.5, 0.5] + 0.5 = 0.25 + 0.25 + 0.5 = 1.0
        s0 = pack_f16x2(H_HALF, H_HALF)
        s1 = pack_f16x2(H_HALF, H_HALF)
        s2 = H_HALF
        res = ref_dot2acc_f16(s0, s1, s2)
        self.assertEqual(res, H_ONE)

    def test_orthogonal_vectors_with_accumulator(self):
        """Dot product of orthogonal vectors yields S2 exactly."""
        # [1.0, 0.0] . [0.0, 1.0] + 4.0 = 0.0 + 4.0 = 4.0
        s0 = pack_f16x2(H_ONE, H_POS_ZERO)
        s1 = pack_f16x2(H_POS_ZERO, H_ONE)
        s2 = H_FOUR
        res = ref_dot2acc_f16(s0, s1, s2)
        self.assertEqual(res, H_FOUR)

        # [2.0, -1.0] . [1.0, 2.0] + 3.0 = (2.0 - 2.0) + 3.0 = 3.0
        s0 = pack_f16x2(H_TWO, H_NEG_ONE)
        s1 = pack_f16x2(H_ONE, H_TWO)
        s2 = H_THREE
        res = ref_dot2acc_f16(s0, s1, s2)
        self.assertEqual(res, H_THREE)

    def test_zero_accumulator(self):
        """Zero accumulator reduces to simple dot product."""
        s0 = pack_f16x2(H_ONE, H_TWO)
        s1 = pack_f16x2(H_THREE, H_FOUR)
        res = ref_dot2acc_f16(s0, s1, H_POS_ZERO)
        self.assertEqual(f16_bits_to_float(res), 11.0)

    def test_zero_multiplicands(self):
        """Zero multiplicands return accumulator intact."""
        s0 = pack_f16x2(H_POS_ZERO, H_POS_ZERO)
        s1 = pack_f16x2(H_POS_ZERO, H_POS_ZERO)
        s2 = H_TEN
        res = ref_dot2acc_f16(s0, s1, s2)
        self.assertEqual(res, H_TEN)

    def test_signed_zero_rules_in_dot2acc(self):
        """Signed zero multiplication and addition semantics."""
        # (+0.0 * +1.0) + (+0.0 * +1.0) + +0.0 = +0.0
        s0 = pack_f16x2(H_POS_ZERO, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_POS_ZERO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_POS_ZERO)

        # (-0.0 * +1.0) + (-0.0 * +1.0) + -0.0 = -0.0 + -0.0 + -0.0 = -0.0
        s0 = pack_f16x2(H_NEG_ZERO, H_NEG_ZERO)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_NEG_ZERO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_NEG_ZERO)

        # (-0.0 * +1.0) + (-0.0 * +1.0) + +0.0 = -0.0 + -0.0 + +0.0 = +0.0
        s0 = pack_f16x2(H_NEG_ZERO, H_NEG_ZERO)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_POS_ZERO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_POS_ZERO)

        # (+0.0 * +1.0) + (-0.0 * +1.0) + +0.0 = +0.0 + -0.0 + +0.0 = +0.0
        s0 = pack_f16x2(H_POS_ZERO, H_NEG_ZERO)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_POS_ZERO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_POS_ZERO)

        # Exact cancellation: (-1.0 * 1.0) + (1.0 * 1.0) + +0.0 = +0.0
        s0 = pack_f16x2(H_NEG_ONE, H_ONE)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_POS_ZERO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_POS_ZERO)

    def test_subnormal_inputs_gradual_underflow(self):
        """Gradual underflow and subnormal operand handling."""
        # Subnormal min * 1.0 + 0 + 0 = Subnormal min
        s0 = pack_f16x2(H_SUB_MIN, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        s2 = H_POS_ZERO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_SUB_MIN)

        # Two subnormals summed with subnormal accumulator
        # 2^-24 + 2^-24 + 2^-24 = 3 * 2^-24 (0x0003)
        s0 = pack_f16x2(H_SUB_MIN, H_SUB_MIN)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_SUB_MIN
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), 0x0003)

    def test_infinity_propagation_and_indeterminate_forms(self):
        """Infinity arithmetic and indeterminate forms in dot2acc."""
        # [+inf, 0] . [1.0, 0] + 2.0 = +inf + 2.0 = +inf
        s0 = pack_f16x2(H_POS_INF, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        s2 = H_TWO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_POS_INF)

        # [-inf, 0] . [1.0, 0] + 2.0 = -inf + 2.0 = -inf
        s0 = pack_f16x2(H_NEG_INF, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        s2 = H_TWO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_NEG_INF)

        # (+inf) + (-inf) + 0 = NaN
        s0 = pack_f16x2(H_POS_INF, H_NEG_INF)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_POS_ZERO
        self.assertTrue(math.isnan(f16_bits_to_float(ref_dot2acc_f16(s0, s1, s2))))

        # 0.0 * inf + 0 + 0 = NaN
        s0 = pack_f16x2(H_POS_ZERO, H_POS_ZERO)
        s1 = pack_f16x2(H_POS_INF, H_POS_ZERO)
        s2 = H_POS_ZERO
        self.assertTrue(math.isnan(f16_bits_to_float(ref_dot2acc_f16(s0, s1, s2))))

    def test_source_selection_op_sel_and_negation(self):
        """op_sel and negation modifiers correctly permute and invert inputs."""
        s0 = pack_f16x2(H_ONE, H_TWO)
        s1 = pack_f16x2(H_THREE, H_FOUR)
        s2 = H_TEN
        # Default: 1*3 + 2*4 + 10 = 21.0
        self.assertEqual(f16_bits_to_float(ref_dot2acc_f16(s0, s1, s2)), 21.0)

        # Swapped S1 halves: op_sel=(0, 1, 0), op_sel_hi=(1, 0, 1) -> 1*4 + 2*3 + 10 = 20.0
        res = ref_dot2acc_f16(s0, s1, s2, op_sel=(0, 1, 0), op_sel_hi=(1, 0, 1))
        self.assertEqual(f16_bits_to_float(res), 20.0)

        # Negate accumulator: 1*3 + 2*4 - 10 = 1.0
        res_neg_acc = ref_dot2acc_f16(s0, s1, s2, neg_lo=(0, 0, 1))
        self.assertEqual(f16_bits_to_float(res_neg_acc), 1.0)

    def test_overflow_to_infinity(self):
        """Values exceeding maximum finite f16 overflow to infinity."""
        # 65504 * 1.0 + 65504 * 1.0 + 0 = 131008 -> +inf
        s0 = pack_f16x2(H_NORM_MAX, H_NORM_MAX)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_POS_ZERO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_POS_INF)


if __name__ == "__main__":
    unittest.main()
