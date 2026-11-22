"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cmp_class.

Models IEEE 754-2008 / IEEE 754-2019 floating-point classification and AMD RDNA2 / GFX10
Vector Comparison (VOPC / VOP3) architectural semantics:
  - v_cmp_class_f32: D.mask[lane] = (class(src0.f32) & src1.u32) != 0
  - v_cmp_class_f64: D.mask[lane] = (class(src0.f64) & src1.u32) != 0

RDNA2 Floating-Point Classification Bit Layout (10-bit bitmask):
  - Bit 0 (0x001): Signaling NaN (sNaN)
  - Bit 1 (0x002): Quiet NaN (qNaN)
  - Bit 2 (0x004): Negative Infinity (-INF)
  - Bit 3 (0x008): Negative Normal
  - Bit 4 (0x010): Negative Subnormal / Denormal
  - Bit 5 (0x020): Negative Zero (-0.0)
  - Bit 6 (0x040): Positive Zero (+0.0)
  - Bit 7 (0x080): Positive Subnormal / Denormal
  - Bit 8 (0x100): Positive Normal
  - Bit 9 (0x200): Positive Infinity (+INF)

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import math
import struct
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

# Float32 masks and constants
SIGN_MASK_F32: int = 0x80000000
EXP_MASK_F32: int = 0x7F800000
MANT_MASK_F32: int = 0x007FFFFF
QUIET_NAN_BIT_F32: int = 0x00400000

POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_ONE_F32: int = 0x3F800000
NEG_ONE_F32: int = 0xBF800000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001
DENORM_MIN_F32: int = 0x00000001
DENORM_MAX_F32: int = 0x007FFFFF
NEG_DENORM_MIN_F32: int = 0x80000001
NEG_DENORM_MAX_F32: int = 0x807FFFFF
FLT_MIN_F32: int = 0x00800000
FLT_MAX_F32: int = 0x7F7FFFFF
NEG_FLT_MIN_F32: int = 0x80800000
NEG_FLT_MAX_F32: int = 0xFF7FFFFF

# Float64 masks and constants
SIGN_MASK_F64: int = 0x8000000000000000
EXP_MASK_F64: int = 0x7FF0000000000000
MANT_MASK_F64: int = 0x000FFFFFFFFFFFFF
QUIET_NAN_BIT_F64: int = 0x0008000000000000

POS_ZERO_F64: int = 0x0000000000000000
NEG_ZERO_F64: int = 0x8000000000000000
POS_ONE_F64: int = 0x3FF0000000000000
NEG_ONE_F64: int = 0xBFF0000000000000
POS_INF_F64: int = 0x7FF0000000000000
NEG_INF_F64: int = 0xFFF0000000000000
QNAN_F64: int = 0x7FF8000000000000
SNAN_F64: int = 0x7FF0000000000001
DENORM_MIN_F64: int = 0x0000000000000001
DENORM_MAX_F64: int = 0x000FFFFFFFFFFFFF
NEG_DENORM_MIN_F64: int = 0x8000000000000001
NEG_DENORM_MAX_F64: int = 0x800FFFFFFFFFFFFF
DBL_MIN_F64: int = 0x0010000000000000
DBL_MAX_F64: int = 0x7FEFFFFFFFFFFFFF
NEG_DBL_MIN_F64: int = 0x8010000000000000
NEG_DBL_MAX_F64: int = 0xFFEFFFFFFFFFFFFF

# 10 IEEE 754 Floating-Point Class Bit Flags
CLASS_SNAN: int = 1 << 0          # 0x001
CLASS_QNAN: int = 1 << 1          # 0x002
CLASS_NEG_INF: int = 1 << 2       # 0x004
CLASS_NEG_NORMAL: int = 1 << 3    # 0x008
CLASS_NEG_DENORMAL: int = 1 << 4  # 0x010
CLASS_NEG_ZERO: int = 1 << 5      # 0x020
CLASS_POS_ZERO: int = 1 << 6      # 0x040
CLASS_POS_DENORMAL: int = 1 << 7  # 0x080
CLASS_POS_NORMAL: int = 1 << 8    # 0x100
CLASS_POS_INF: int = 1 << 9       # 0x200

# Canonical composite class masks
CLASS_MASK_NAN: int = CLASS_SNAN | CLASS_QNAN
CLASS_MASK_INF: int = CLASS_NEG_INF | CLASS_POS_INF
CLASS_MASK_ZERO: int = CLASS_NEG_ZERO | CLASS_POS_ZERO
CLASS_MASK_DENORMAL: int = CLASS_NEG_DENORMAL | CLASS_POS_DENORMAL
CLASS_MASK_NORMAL: int = CLASS_NEG_NORMAL | CLASS_POS_NORMAL
CLASS_MASK_FINITE: int = CLASS_MASK_ZERO | CLASS_MASK_DENORMAL | CLASS_MASK_NORMAL
CLASS_MASK_ALL: int = (1 << 10) - 1  # 0x3FF


def apply_f32_modifiers(bits: int, neg: bool = False, abs_val: bool = False) -> int:
    """Applies VOP3 input modifiers: abs (|x|) first, then neg (-x)."""
    bits &= MASK32
    if abs_val:
        bits &= ~SIGN_MASK_F32
    if neg:
        bits ^= SIGN_MASK_F32
    return bits & MASK32


def apply_f64_modifiers(bits: int, neg: bool = False, abs_val: bool = False) -> int:
    """Applies VOP3 input modifiers: abs (|x|) first, then neg (-x)."""
    bits &= MASK64
    if abs_val:
        bits &= ~SIGN_MASK_F64
    if neg:
        bits ^= SIGN_MASK_F64
    return bits & MASK64


def classify_f32(bits: int) -> int:
    """Classifies an IEEE 754 binary32 float and returns a single class bit."""
    u = bits & MASK32
    sign = (u & SIGN_MASK_F32) != 0
    exp_val = (u & EXP_MASK_F32) >> 23
    mant = u & MANT_MASK_F32

    if exp_val == 0xFF:
        if mant == 0:
            return CLASS_NEG_INF if sign else CLASS_POS_INF
        is_quiet = (mant & QUIET_NAN_BIT_F32) != 0
        return CLASS_QNAN if is_quiet else CLASS_SNAN

    if exp_val == 0:
        if mant == 0:
            return CLASS_NEG_ZERO if sign else CLASS_POS_ZERO
        return CLASS_NEG_DENORMAL if sign else CLASS_POS_DENORMAL

    return CLASS_NEG_NORMAL if sign else CLASS_POS_NORMAL


def classify_f64(bits: int) -> int:
    """Classifies an IEEE 754 binary64 double and returns a single class bit."""
    u = bits & MASK64
    sign = (u & SIGN_MASK_F64) != 0
    exp_val = (u & EXP_MASK_F64) >> 52
    mant = u & MANT_MASK_F64

    if exp_val == 0x7FF:
        if mant == 0:
            return CLASS_NEG_INF if sign else CLASS_POS_INF
        is_quiet = (mant & QUIET_NAN_BIT_F64) != 0
        return CLASS_QNAN if is_quiet else CLASS_SNAN

    if exp_val == 0:
        if mant == 0:
            return CLASS_NEG_ZERO if sign else CLASS_POS_ZERO
        return CLASS_NEG_DENORMAL if sign else CLASS_POS_DENORMAL

    return CLASS_NEG_NORMAL if sign else CLASS_POS_NORMAL


def ref_cmp_class_f32(src0_bits: int, mask: int, neg: bool = False, abs_val: bool = False) -> bool:
    """Reference oracle for v_cmp_class_f32."""
    val = apply_f32_modifiers(src0_bits, neg=neg, abs_val=abs_val)
    cls = classify_f32(val)
    return bool(cls & (mask & MASK32))


def ref_cmp_class_f64(src0_bits: int, mask: int, neg: bool = False, abs_val: bool = False) -> bool:
    """Reference oracle for v_cmp_class_f64."""
    val = apply_f64_modifiers(src0_bits, neg=neg, abs_val=abs_val)
    cls = classify_f64(val)
    return bool(cls & (mask & MASK32))


class TestValuCmpClassOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_classify_all_f32_classes(self) -> None:
        expected = [
            (SNAN_F32, CLASS_SNAN),
            (QNAN_F32, CLASS_QNAN),
            (NEG_INF_F32, CLASS_NEG_INF),
            (NEG_ONE_F32, CLASS_NEG_NORMAL),
            (NEG_DENORM_MIN_F32, CLASS_NEG_DENORMAL),
            (NEG_ZERO_F32, CLASS_NEG_ZERO),
            (POS_ZERO_F32, CLASS_POS_ZERO),
            (DENORM_MIN_F32, CLASS_POS_DENORMAL),
            (POS_ONE_F32, CLASS_POS_NORMAL),
            (POS_INF_F32, CLASS_POS_INF),
        ]
        for bits, cls in expected:
            with self.subTest(bits=f"{bits:#010x}", expected=f"{cls:#05x}"):
                self.assertEqual(classify_f32(bits), cls)

    def test_classify_all_f64_classes(self) -> None:
        expected = [
            (SNAN_F64, CLASS_SNAN),
            (QNAN_F64, CLASS_QNAN),
            (NEG_INF_F64, CLASS_NEG_INF),
            (NEG_ONE_F64, CLASS_NEG_NORMAL),
            (NEG_DENORM_MIN_F64, CLASS_NEG_DENORMAL),
            (NEG_ZERO_F64, CLASS_NEG_ZERO),
            (POS_ZERO_F64, CLASS_POS_ZERO),
            (DENORM_MIN_F64, CLASS_POS_DENORMAL),
            (POS_ONE_F64, CLASS_POS_NORMAL),
            (POS_INF_F64, CLASS_POS_INF),
        ]
        for bits, cls in expected:
            with self.subTest(bits=f"{bits:#018x}", expected=f"{cls:#05x}"):
                self.assertEqual(classify_f64(bits), cls)

    def test_ref_cmp_class_f32_composite_masks(self) -> None:
        self.assertTrue(ref_cmp_class_f32(POS_ZERO_F32, CLASS_MASK_ZERO))
        self.assertTrue(ref_cmp_class_f32(NEG_ZERO_F32, CLASS_MASK_ZERO))
        self.assertFalse(ref_cmp_class_f32(POS_ONE_F32, CLASS_MASK_ZERO))

        self.assertTrue(ref_cmp_class_f32(SNAN_F32, CLASS_MASK_NAN))
        self.assertTrue(ref_cmp_class_f32(QNAN_F32, CLASS_MASK_NAN))
        self.assertFalse(ref_cmp_class_f32(POS_INF_F32, CLASS_MASK_NAN))

        self.assertTrue(ref_cmp_class_f32(POS_INF_F32, CLASS_MASK_INF))
        self.assertTrue(ref_cmp_class_f32(NEG_INF_F32, CLASS_MASK_INF))
        self.assertFalse(ref_cmp_class_f32(QNAN_F32, CLASS_MASK_INF))

    def test_ref_cmp_class_f64_composite_masks(self) -> None:
        self.assertTrue(ref_cmp_class_f64(POS_ZERO_F64, CLASS_MASK_ZERO))
        self.assertTrue(ref_cmp_class_f64(NEG_ZERO_F64, CLASS_MASK_ZERO))
        self.assertFalse(ref_cmp_class_f64(POS_ONE_F64, CLASS_MASK_ZERO))

        self.assertTrue(ref_cmp_class_f64(SNAN_F64, CLASS_MASK_NAN))
        self.assertTrue(ref_cmp_class_f64(QNAN_F64, CLASS_MASK_NAN))
        self.assertFalse(ref_cmp_class_f64(POS_INF_F64, CLASS_MASK_NAN))

    def test_input_modifiers_f32(self) -> None:
        # abs(-1.0) -> +1.0 (matches POS_NORMAL)
        self.assertTrue(ref_cmp_class_f32(NEG_ONE_F32, CLASS_POS_NORMAL, abs_val=True))
        self.assertFalse(ref_cmp_class_f32(NEG_ONE_F32, CLASS_NEG_NORMAL, abs_val=True))

        # neg(+1.0) -> -1.0 (matches NEG_NORMAL)
        self.assertTrue(ref_cmp_class_f32(POS_ONE_F32, CLASS_NEG_NORMAL, neg=True))
        self.assertFalse(ref_cmp_class_f32(POS_ONE_F32, CLASS_POS_NORMAL, neg=True))

        # -|-0.0| -> -0.0 (matches NEG_ZERO)
        self.assertTrue(ref_cmp_class_f32(NEG_ZERO_F32, CLASS_NEG_ZERO, neg=True, abs_val=True))
        self.assertFalse(ref_cmp_class_f32(NEG_ZERO_F32, CLASS_POS_ZERO, neg=True, abs_val=True))

    def test_input_modifiers_f64(self) -> None:
        # abs(-INF) -> +INF
        self.assertTrue(ref_cmp_class_f64(NEG_INF_F64, CLASS_POS_INF, abs_val=True))
        self.assertFalse(ref_cmp_class_f64(NEG_INF_F64, CLASS_NEG_INF, abs_val=True))

        # neg(+INF) -> -INF
        self.assertTrue(ref_cmp_class_f64(POS_INF_F64, CLASS_NEG_INF, neg=True))
        self.assertFalse(ref_cmp_class_f64(POS_INF_F64, CLASS_POS_INF, neg=True))


if __name__ == "__main__":
    unittest.main()
