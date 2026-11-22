"""Semantic value tests for single-precision square root v_sqrt_f32."""
from __future__ import annotations
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

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def f32(x: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def f32_bits(x: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_f32_nan_bits(bits: int) -> bool:
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def ref_sqrt(x: float) -> float:
    fx = f32(x)
    bits = f32_bits(fx)
    sign = (bits >> 31) & 1
    if sign == 1 and fx != 0.0:
        return float("nan")
    if fx == 0.0:
        return -0.0 if math.copysign(1.0, fx) < 0 else 0.0
    if math.isnan(fx):
        return fx
    if math.isinf(fx):
        return float("inf")
    return f32(math.sqrt(fx))


class TestValuSqrtF32Oracle(unittest.TestCase):
    def test_oracle_finite_positive(self):
        for v in (0.0, 1.0, 4.0, 9.0, 16.0, 2.0, 0.25, 1e-10, 1e10):
            res = ref_sqrt(v)
            self.assertEqual(res, math.sqrt(v))

    def test_oracle_special_cases(self):
        self.assertEqual(f32_bits(ref_sqrt(0.0)), 0x00000000)
        self.assertEqual(f32_bits(ref_sqrt(-0.0)), 0x80000000)
        self.assertEqual(f32_bits(ref_sqrt(float("inf"))), 0x7F800000)
        self.assertTrue(is_f32_nan_bits(f32_bits(ref_sqrt(-1.0))))
