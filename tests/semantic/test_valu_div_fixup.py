"""Semantic interpreter tests for AMD RDNA2 floating-point quotient fixup: v_div_fixup_f32.

RDNA2 VALU floating-point division fixup instruction:
  v_div_fixup_f32  vdst, s0, s1, s2

Arguments:
  s0: quotient estimate (computed from preliminary reciprocal / Newton-Raphson sequence)
  s1: denominator (y in x / y)
  s2: numerator (x in x / y)

Exact AMD RDNA2 fixup truth table:
  1. If s1 is NaN, return quiet NaN (quiet_f32(s1)).
  2. If s2 is NaN, return quiet NaN (quiet_f32(s2)).
  3. If s1 == 0 and s2 == 0, return NaN (canonical QNaN: 0x7FC00000).
  4. If s1 == 0 and s2 != 0, return sign(s1 ^ s2) * inf.
  5. If s1 == +-inf and s2 == +-inf, return NaN (canonical QNaN: 0x7FC00000).
  6. If s1 == +-inf and s2 != +-inf, return sign(s1 ^ s2) * 0.0.
  7. If s2 == 0, return sign(s1 ^ s2) * 0.0.
  8. Otherwise return s0.

Zero game data; standard library Python only; Python 3.9+ compatible.
"""
from __future__ import annotations

import math
import struct
import unittest

MASK32 = 0xFFFFFFFF

# IEEE-754 binary32 bit constants
F32_POS_ZERO = 0x00000000
F32_NEG_ZERO = 0x80000000
F32_POS_INF = 0x7F800000
F32_NEG_INF = 0xFF800000
F32_QNAN = 0x7FC00000
F32_NEG_QNAN = 0xFFC00000
F32_SNAN = 0x7F800001
F32_NEG_SNAN = 0xFF800001

F32_ONE = 0x3F800000
F32_NEG_ONE = 0xBF800000
F32_TWO = 0x40000000
F32_HALF = 0x3F000000
F32_NORM_MIN = 0x00800000
F32_NORM_MAX = 0x7F7FFFFF
F32_SUB_MIN = 0x00000001
F32_SUB_MAX = 0x007FFFFF

OLD_VGPR_SENTINEL = 0xDEADBEEF


def f32_to_bits(val: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return F32_NEG_INF if val < 0 else F32_POS_INF


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_sign(bits: int) -> int:
    return (bits >> 31) & 1


def is_f32_nan(bits: int) -> bool:
    return ((bits & 0x7F800000) == 0x7F800000) and ((bits & 0x007FFFFF) != 0)


def is_f32_snan(bits: int) -> bool:
    return is_f32_nan(bits) and ((bits & 0x00400000) == 0)


def is_f32_qnan(bits: int) -> bool:
    return is_f32_nan(bits) and ((bits & 0x00400000) != 0)


def is_f32_zero(bits: int) -> bool:
    return (bits & 0x7FFFFFFF) == 0


def is_f32_inf(bits: int) -> bool:
    return (bits & 0x7FFFFFFF) == 0x7F800000


def is_f32_subnormal(bits: int) -> bool:
    return ((bits & 0x7F800000) == 0) and ((bits & 0x007FFFFF) != 0)


def is_f32_normal(bits: int) -> bool:
    exp = (bits >> 23) & 0xFF
    return 0 < exp < 255


def quiet_f32(bits: int) -> int:
    """Ensure quiet NaN bit (bit 22) is set on single-precision NaN while preserving payload."""
    if not is_f32_nan(bits):
        return F32_QNAN
    return (bits & MASK32) | 0x00400000


def ref_v_div_fixup_f32(s0: int, s1: int, s2: int) -> int:
    """Exact IEEE single-precision division fixup reference oracle for RDNA2 v_div_fixup_f32.

    s0 = quotient estimate
    s1 = denominator
    s2 = numerator
    """
    s0 &= MASK32
    s1 &= MASK32
    s2 &= MASK32

    sign_xor = f32_sign(s1) ^ f32_sign(s2)
    sign_bit = 0x80000000 if sign_xor else 0x00000000

    if is_f32_nan(s1):
        return quiet_f32(s1)
    if is_f32_nan(s2):
        return quiet_f32(s2)
    if is_f32_zero(s1) and is_f32_zero(s2):
        return F32_QNAN
    if is_f32_zero(s1):
        return sign_bit | F32_POS_INF
    if is_f32_inf(s1) and is_f32_inf(s2):
        return F32_QNAN
    if is_f32_inf(s1):
        return sign_bit | F32_POS_ZERO
    if is_f32_zero(s2):
        return sign_bit | F32_POS_ZERO
    return s0


class WaveDivFixupInterpreter:
    """Multi-lane SIMD wave reference interpreter modeling RDNA2 Wave32/Wave64 VALU execution."""

    def __init__(self, wave_size: int = 32) -> None:
        self.wave_size = wave_size
        self.exec_mask = (1 << wave_size) - 1
        self.vgprs = [[OLD_VGPR_SENTINEL] * 256 for _ in range(wave_size)]

    def set_lane_vgpr(self, lane: int, reg: int, val: int) -> None:
        self.vgprs[lane][reg] = val & MASK32

    def get_lane_vgpr(self, lane: int, reg: int) -> int:
        return self.vgprs[lane][reg] & MASK32

    def v_div_fixup_f32(self, vdst: int, s0_reg: int, s1_reg: int, s2_reg: int) -> None:
        """Executes v_div_fixup_f32 across all lanes conditioned on the exec mask."""
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.get_lane_vgpr(lane, s0_reg)
                s1 = self.get_lane_vgpr(lane, s1_reg)
                s2 = self.get_lane_vgpr(lane, s2_reg)
                self.vgprs[lane][vdst] = ref_v_div_fixup_f32(s0, s1, s2)


class TestF32Classification(unittest.TestCase):
    def test_zero_classification(self):
        self.assertTrue(is_f32_zero(F32_POS_ZERO))
        self.assertTrue(is_f32_zero(F32_NEG_ZERO))
        self.assertFalse(is_f32_zero(F32_ONE))
        self.assertEqual(f32_sign(F32_POS_ZERO), 0)
        self.assertEqual(f32_sign(F32_NEG_ZERO), 1)

    def test_inf_classification(self):
        self.assertTrue(is_f32_inf(F32_POS_INF))
        self.assertTrue(is_f32_inf(F32_NEG_INF))
        self.assertFalse(is_f32_inf(F32_QNAN))
        self.assertEqual(f32_sign(F32_POS_INF), 0)
        self.assertEqual(f32_sign(F32_NEG_INF), 1)

    def test_nan_classification_and_quieting(self):
        self.assertTrue(is_f32_nan(F32_QNAN))
        self.assertTrue(is_f32_nan(F32_SNAN))
        self.assertTrue(is_f32_qnan(F32_QNAN))
        self.assertTrue(is_f32_snan(F32_SNAN))
        self.assertFalse(is_f32_qnan(F32_SNAN))
        self.assertFalse(is_f32_snan(F32_QNAN))

        quieted = quiet_f32(F32_SNAN)
        self.assertTrue(is_f32_qnan(quieted))
        self.assertEqual(quieted, F32_SNAN | 0x00400000)

        # NaN payload preservation on quieting
        custom_snan = 0x7F800042
        self.assertTrue(is_f32_snan(custom_snan))
        custom_qnan = quiet_f32(custom_snan)
        self.assertTrue(is_f32_qnan(custom_qnan))
        self.assertEqual(custom_qnan & 0x003FFFFF, 0x00000042)


class TestDivFixupF32ZeroAndInfCases(unittest.TestCase):
    def test_zero_divided_by_zero_produces_qnan(self):
        for s1 in (F32_POS_ZERO, F32_NEG_ZERO):
            for s2 in (F32_POS_ZERO, F32_NEG_ZERO):
                with self.subTest(s1=hex(s1), s2=hex(s2)):
                    res = ref_v_div_fixup_f32(F32_ONE, s1, s2)
                    self.assertTrue(is_f32_qnan(res))
                    self.assertEqual(res, F32_QNAN)

    def test_finite_divided_by_zero_produces_signed_inf(self):
        test_cases = [
            (F32_POS_ZERO, F32_ONE, F32_POS_INF),       # (+1) / (+0) = +inf
            (F32_NEG_ZERO, F32_ONE, F32_NEG_INF),       # (+1) / (-0) = -inf
            (F32_POS_ZERO, F32_NEG_ONE, F32_NEG_INF),   # (-1) / (+0) = -inf
            (F32_NEG_ZERO, F32_NEG_ONE, F32_POS_INF),   # (-1) / (-0) = +inf
            (F32_POS_ZERO, F32_TWO, F32_POS_INF),       # (+2) / (+0) = +inf
            (F32_NEG_ZERO, F32_TWO, F32_NEG_INF),       # (+2) / (-0) = -inf
        ]
        for s1, s2, expected in test_cases:
            with self.subTest(s1=hex(s1), s2=hex(s2)):
                res = ref_v_div_fixup_f32(0x12345678, s1, s2)
                self.assertEqual(res, expected)

    def test_inf_divided_by_inf_produces_qnan(self):
        for s1 in (F32_POS_INF, F32_NEG_INF):
            for s2 in (F32_POS_INF, F32_NEG_INF):
                with self.subTest(s1=hex(s1), s2=hex(s2)):
                    res = ref_v_div_fixup_f32(F32_ONE, s1, s2)
                    self.assertTrue(is_f32_qnan(res))
                    self.assertEqual(res, F32_QNAN)

    def test_finite_divided_by_inf_produces_signed_zero(self):
        test_cases = [
            (F32_POS_INF, F32_ONE, F32_POS_ZERO),       # (+1) / (+inf) = +0.0
            (F32_NEG_INF, F32_ONE, F32_NEG_ZERO),       # (+1) / (-inf) = -0.0
            (F32_POS_INF, F32_NEG_ONE, F32_NEG_ZERO),   # (-1) / (+inf) = -0.0
            (F32_NEG_INF, F32_NEG_ONE, F32_POS_ZERO),   # (-1) / (-inf) = +0.0
            (F32_POS_INF, F32_TWO, F32_POS_ZERO),       # (+2) / (+inf) = +0.0
            (F32_NEG_INF, F32_TWO, F32_NEG_ZERO),       # (+2) / (-inf) = -0.0
        ]
        for s1, s2, expected in test_cases:
            with self.subTest(s1=hex(s1), s2=hex(s2)):
                res = ref_v_div_fixup_f32(0xCAFEBABE, s1, s2)
                self.assertEqual(res, expected)

    def test_zero_divided_by_finite_produces_signed_zero(self):
        test_cases = [
            (F32_ONE, F32_POS_ZERO, F32_POS_ZERO),       # (+0.0) / (+1) = +0.0
            (F32_NEG_ONE, F32_POS_ZERO, F32_NEG_ZERO),   # (+0.0) / (-1) = -0.0
            (F32_ONE, F32_NEG_ZERO, F32_NEG_ZERO),       # (-0.0) / (+1) = -0.0
            (F32_NEG_ONE, F32_NEG_ZERO, F32_POS_ZERO),   # (-0.0) / (-1) = +0.0
            (F32_TWO, F32_POS_ZERO, F32_POS_ZERO),       # (+0.0) / (+2) = +0.0
            (f32_to_bits(-5.0), F32_POS_ZERO, F32_NEG_ZERO),
        ]
        for s1, s2, expected in test_cases:
            with self.subTest(s1=hex(s1), s2=hex(s2)):
                res = ref_v_div_fixup_f32(0x99887766, s1, s2)
                self.assertEqual(res, expected)


class TestDivFixupF32NaNCases(unittest.TestCase):
    def test_denominator_nan_propagates_quiet_nan(self):
        snan = 0x7F800007
        qnan = 0x7FC00010
        # s1 is sNaN: quiets s1
        res_snan = ref_v_div_fixup_f32(F32_ONE, snan, F32_ONE)
        self.assertTrue(is_f32_qnan(res_snan))
        self.assertEqual(res_snan, snan | 0x00400000)

        # s1 is qNaN: returns qNaN
        res_qnan = ref_v_div_fixup_f32(F32_ONE, qnan, F32_ONE)
        self.assertEqual(res_qnan, qnan)

    def test_numerator_nan_propagates_quiet_nan_when_denom_not_nan(self):
        snan = 0x7F80002A
        qnan = 0x7FC00055
        res_snan = ref_v_div_fixup_f32(F32_ONE, F32_ONE, snan)
        self.assertTrue(is_f32_qnan(res_snan))
        self.assertEqual(res_snan, snan | 0x00400000)

        res_qnan = ref_v_div_fixup_f32(F32_ONE, F32_ONE, qnan)
        self.assertEqual(res_qnan, qnan)

    def test_denominator_nan_takes_precedence_over_numerator_nan(self):
        s1_nan = 0x7F800111
        s2_nan = 0x7F800222
        res = ref_v_div_fixup_f32(F32_ONE, s1_nan, s2_nan)
        self.assertEqual(res, s1_nan | 0x00400000)


class TestDivFixupF32NormalPassThrough(unittest.TestCase):
    def test_regular_quotient_estimate_passes_through_intact(self):
        # When neither denominator nor numerator is 0, inf, or NaN, s0 is returned directly
        s0_estimates = [
            f32_to_bits(1.0),
            f32_to_bits(0.5),
            f32_to_bits(0.25),
            f32_to_bits(-3.14159),
            0x12345678,
            0xAABBCCDD,
        ]
        s1_denom = f32_to_bits(3.0)
        s2_num = f32_to_bits(6.0)

        for s0 in s0_estimates:
            with self.subTest(s0=hex(s0)):
                res = ref_v_div_fixup_f32(s0, s1_denom, s2_num)
                self.assertEqual(res, s0)


class TestDivFixupF32SimdWaveExecution(unittest.TestCase):
    def test_wave32_divergence_under_exec_mask(self):
        wave = WaveDivFixupInterpreter(wave_size=32)
        # Enable only even lanes (0, 2, 4, ...)
        wave.exec_mask = 0x55555555

        vdst, s0_reg, s1_reg, s2_reg = 0, 1, 2, 3
        for lane in range(32):
            wave.set_lane_vgpr(lane, s0_reg, F32_HALF)
            # Even lanes do 1 / 0 -> +inf; odd lanes do 0 / 1 -> 0.0
            if lane % 2 == 0:
                wave.set_lane_vgpr(lane, s1_reg, F32_POS_ZERO)
                wave.set_lane_vgpr(lane, s2_reg, F32_ONE)
            else:
                wave.set_lane_vgpr(lane, s1_reg, F32_ONE)
                wave.set_lane_vgpr(lane, s2_reg, F32_POS_ZERO)
            wave.set_lane_vgpr(lane, vdst, OLD_VGPR_SENTINEL)

        wave.v_div_fixup_f32(vdst, s0_reg, s1_reg, s2_reg)

        for lane in range(32):
            val = wave.get_lane_vgpr(lane, vdst)
            if lane % 2 == 0:
                self.assertEqual(val, F32_POS_INF, f"Active lane {lane} failed")
            else:
                self.assertEqual(val, OLD_VGPR_SENTINEL, f"Inactive lane {lane} mutated")

    def test_wave64_execution(self):
        wave = WaveDivFixupInterpreter(wave_size=64)
        wave.exec_mask = 0x00000000FFFFFFFF  # lower 32 lanes active, upper 32 inactive

        vdst, s0_reg, s1_reg, s2_reg = 10, 11, 12, 13
        for lane in range(64):
            wave.set_lane_vgpr(lane, s0_reg, F32_ONE)
            wave.set_lane_vgpr(lane, s1_reg, F32_POS_ZERO)
            wave.set_lane_vgpr(lane, s2_reg, F32_POS_ZERO)  # 0 / 0 -> QNaN
            wave.set_lane_vgpr(lane, vdst, 0x11223344)

        wave.v_div_fixup_f32(vdst, s0_reg, s1_reg, s2_reg)

        for lane in range(64):
            val = wave.get_lane_vgpr(lane, vdst)
            if lane < 32:
                self.assertEqual(val, F32_QNAN)
            else:
                self.assertEqual(val, 0x11223344)


if __name__ == "__main__":
    unittest.main()
