"""Semantic value tests for RDNA2 single-precision transcendental VALU operations:
  - v_cos_f32:  D.f32 = cos(S0.f32 * 2*pi)   (operand in turns / revolutions)
  - v_sin_f32:  D.f32 = sin(S0.f32 * 2*pi)   (operand in turns / revolutions)

RDNA2 takes trigonometric inputs in units of revolutions (turns), reducing to [-0.5, 0.5]
before multiplying by 2*pi. This test suite verifies plain-Python IEEE-754 / RDNA2 reference
oracles, Pythagorean identity invariants (sin^2 + cos^2 ~= 1.0), periodic turn reduction,
and end-to-end SPIR-V shader translation and execution via spirv_mini_interp.

Standard library Python only; synthetic hand-written listings only (no game data).
"""
from __future__ import annotations

import gc
import math
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
TWO_PI = 6.283185307179586476925286766559

# Canonical binary32 IEEE-754 bit patterns.
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
HALF = 0x3F000000          # 0.5
NEG_HALF = 0xBF000000      # -0.5
ONE = 0x3F800000           # 1.0
NEG_ONE = 0xBF800000       # -1.0
TWO = 0x40000000           # 2.0
NEG_TWO = 0xC0000000       # -2.0
POS_INF = 0x7F800000       # +inf
NEG_INF = 0xFF800000       # -inf
QNAN = 0x7FC00000          # canonical positive quiet NaN
NEG_QNAN = 0xFFC00000      # canonical negative quiet NaN
SNAN = 0x7F800001          # signaling NaN


def f32_to_bits(val: float) -> int:
    """Pack Python float to 32-bit unsigned integer IEEE-754 representation."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def bits_to_f32(bits: int) -> float:
    """Unpack 32-bit unsigned integer to Python single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_f32_nan_bits(bits: int) -> bool:
    """Check if 32-bit word encodes an IEEE-754 NaN."""
    b = bits & MASK32
    return (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF) != 0


def round_even(x: float) -> float:
    """Round to nearest integer, breaking ties to even (IEEE-754 roundTiesToEven)."""
    return float(round(x))


def ref_turn_reduce(turns: float) -> float:
    """Reduce turn count to [-0.5, 0.5] as performed by RDNA2 hardware and rdna2spirv.

    For finite inputs, turns - round_even(turns) reduces exactly in binary32.
    Signed zeros (+0.0 and -0.0) are preserved.
    """
    if math.isnan(turns) or math.isinf(turns):
        return float("nan")
    if turns == 0.0:
        return turns
    return turns - round_even(turns)


def ref_cos_f32(turns_bits: int) -> int:
    """RDNA2 v_cos_f32 reference oracle: D.f32 = cos(turns * 2*pi)."""
    if is_f32_nan_bits(turns_bits):
        return QNAN
    val = bits_to_f32(turns_bits)
    if math.isinf(val):
        return QNAN
    if val == 0.0:
        return ONE
    t = ref_turn_reduce(val)
    rad = t * TWO_PI
    res = math.cos(rad)
    return f32_to_bits(res)


def ref_sin_f32(turns_bits: int) -> int:
    """RDNA2 v_sin_f32 reference oracle: D.f32 = sin(turns * 2*pi)."""
    if is_f32_nan_bits(turns_bits):
        return QNAN
    val = bits_to_f32(turns_bits)
    if math.isinf(val):
        return QNAN
    if turns_bits == NEG_ZERO:
        return NEG_ZERO
    if turns_bits == POS_ZERO:
        return POS_ZERO
    t = ref_turn_reduce(val)
    rad = t * TWO_PI
    res = math.sin(rad)
    if res == 0.0 and math.copysign(1.0, val) < 0:
        return NEG_ZERO
    return f32_to_bits(res)


class TestTranscendentalOracles(unittest.TestCase):
    """Test suite validating the RDNA2 transcendental reference oracles."""

    def test_cos_cardinal_angles(self):
        # 0 turns = 0 rad -> cos = 1.0
        self.assertEqual(ref_cos_f32(POS_ZERO), ONE)
        self.assertEqual(ref_cos_f32(NEG_ZERO), ONE)
        # 0.25 turns = pi/2 rad -> cos = 0.0
        c025 = bits_to_f32(ref_cos_f32(f32_to_bits(0.25)))
        self.assertAlmostEqual(c025, 0.0, places=6)
        # 0.5 turns = pi rad -> cos = -1.0
        c05 = bits_to_f32(ref_cos_f32(f32_to_bits(0.5)))
        self.assertAlmostEqual(c05, -1.0, places=6)
        # 0.75 turns = 3pi/2 rad -> cos = 0.0
        c075 = bits_to_f32(ref_cos_f32(f32_to_bits(0.75)))
        self.assertAlmostEqual(c075, 0.0, places=6)
        # 1.0 turns = 2pi rad -> cos = 1.0
        c10 = bits_to_f32(ref_cos_f32(f32_to_bits(1.0)))
        self.assertAlmostEqual(c10, 1.0, places=6)

    def test_sin_cardinal_angles(self):
        # 0.0 turns -> sin = 0.0; -0.0 turns -> sin = -0.0
        self.assertEqual(ref_sin_f32(POS_ZERO), POS_ZERO)
        self.assertEqual(ref_sin_f32(NEG_ZERO), NEG_ZERO)
        # 0.25 turns = pi/2 rad -> sin = 1.0
        s025 = bits_to_f32(ref_sin_f32(f32_to_bits(0.25)))
        self.assertAlmostEqual(s025, 1.0, places=6)
        # 0.5 turns = pi rad -> sin = 0.0
        s05 = bits_to_f32(ref_sin_f32(f32_to_bits(0.5)))
        self.assertAlmostEqual(s05, 0.0, places=6)
        # 0.75 turns = 3pi/2 rad -> sin = -1.0
        s075 = bits_to_f32(ref_sin_f32(f32_to_bits(0.75)))
        self.assertAlmostEqual(s075, -1.0, places=6)
        # 1.0 turns = 2pi rad -> sin = 0.0
        s10 = bits_to_f32(ref_sin_f32(f32_to_bits(1.0)))
        self.assertAlmostEqual(s10, 0.0, places=6)

    def test_pythagorean_identity(self):
        # sin^2(x) + cos^2(x) == 1.0 across continuous spectrum of angles
        for i in range(-50, 51):
            turns = i * 0.125 + 0.03125
            t_bits = f32_to_bits(turns)
            s = bits_to_f32(ref_sin_f32(t_bits))
            c = bits_to_f32(ref_cos_f32(t_bits))
            pyth = s * s + c * c
            self.assertAlmostEqual(pyth, 1.0, delta=1e-5, msg=f"Pythagorean identity failed at turns={turns}")

    def test_turn_periodicity_and_symmetry(self):
        # Invariance under whole integer turn offsets: f(x + k) == f(x)
        # Using dyadic fractions where fractional part is exact in binary32:
        dyadic_angles = [0.125, 0.25, 0.375, 0.625, 0.875]
        for a in dyadic_angles:
            for k in (1, 2, 5, 10, 50, 100):
                base_c = bits_to_f32(ref_cos_f32(f32_to_bits(a)))
                shift_c = bits_to_f32(ref_cos_f32(f32_to_bits(a + float(k))))
                self.assertAlmostEqual(base_c, shift_c, delta=1e-5)

                base_s = bits_to_f32(ref_sin_f32(f32_to_bits(a)))
                shift_s = bits_to_f32(ref_sin_f32(f32_to_bits(a + float(k))))
                self.assertAlmostEqual(base_s, shift_s, delta=1e-5)

        # Symmetry: cos(-x) == cos(x), sin(-x) == -sin(x)
        angles = [0.1, 0.23, 0.45, 0.67, 0.89]
        for a in angles:
            pos_bits = f32_to_bits(a)
            neg_bits = f32_to_bits(-a)
            self.assertAlmostEqual(bits_to_f32(ref_cos_f32(pos_bits)), bits_to_f32(ref_cos_f32(neg_bits)), delta=1e-6)
            self.assertAlmostEqual(bits_to_f32(ref_sin_f32(pos_bits)), -bits_to_f32(ref_sin_f32(neg_bits)), delta=1e-6)


class TestTrigTranslationAndExecution(unittest.TestCase):
    """End-to-end translation and execution of v_cos_f32 and v_sin_f32."""

    def test_v_cos_and_v_sin_pixel_shader_execution(self):
        test_turns = [0.0, 0.25, 0.5, 0.75]
        for t in test_turns:
            tb = f32_to_bits(t)
            lines = [
                f"v_mov_b32 v0, 0x{tb:08x}",
                "v_cos_f32 v1, v0",
                "v_sin_f32 v2, v0",
                "exp mrt0 v1, v2, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_trig_{tb:08x}", lines)
            words = support.mrt0_words(spv)
            got_cos = bits_to_f32(words[0])
            got_sin = bits_to_f32(words[1])
            expected_cos = bits_to_f32(ref_cos_f32(tb))
            expected_sin = bits_to_f32(ref_sin_f32(tb))
            self.assertAlmostEqual(got_cos, expected_cos, delta=1e-5)
            self.assertAlmostEqual(got_sin, expected_sin, delta=1e-5)

    def test_large_turns_reduction_in_shader(self):
        for large_turn in (100.3, 254.51):
            tb = f32_to_bits(large_turn)
            lines = [
                f"v_mov_b32 v0, 0x{tb:08x}",
                "v_sin_f32 v1, v0",
                "v_cos_f32 v2, v0",
                "exp mrt0 v1, v2, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_large_trig_{tb:08x}", lines)
            words = support.mrt0_words(spv)
            got_sin = bits_to_f32(words[0])
            got_cos = bits_to_f32(words[1])
            expected_sin = bits_to_f32(ref_sin_f32(tb))
            expected_cos = bits_to_f32(ref_cos_f32(tb))
            self.assertAlmostEqual(got_sin, expected_sin, delta=1e-5)
            self.assertAlmostEqual(got_cos, expected_cos, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
