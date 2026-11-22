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


def run_sqrt(val_bits: int) -> int:
    lines = [
        f"v_mov_b32 v1, {val_bits:#x}",
        "v_sqrt_f32 v3, v1",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        with open(os.path.join(tmp, "disasm", "test.s"), "w", encoding="utf-8") as f:
            f.write("; test\n" + "".join(f"\t{ln}\n" for ln in lines))
        with open(os.path.join(tmp, "test.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(os.path.join(tmp, "disasm", "test.s"))
        res = si.run_fragment(spv)
        return list(res["%out_mrt0"])[3]


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


class TestValuSqrtF32SIMDAndPrecision(unittest.TestCase):
    def test_simd_multilane_execution(self):
        inputs = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0]
        for val in inputs:
            vbits = f32_bits(f32(val))
            got_bits = run_sqrt(vbits)
            expected = f32(math.sqrt(val))
            self.assertEqual(bits_f32(got_bits), expected)

    def test_precision_tolerance(self):
        for val in (2.0, 3.0, 5.0, 10.0, 0.1, 1000.0):
            vbits = f32_bits(f32(val))
            got_bits = run_sqrt(vbits)
            got = bits_f32(got_bits)
            expected = math.sqrt(val)
            diff = abs(got - expected)
            self.assertLessEqual(diff, 1e-6 * expected)
