from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_interp_p1_f32, v_interp_p2_f32, and v_interp_mov_f32.

RDNA2 VINTRP instruction family for perspective parameter interpolation:
- v_interp_p1_f32: Computes D = P10 * S + P0
  Evaluates the first barycentric coordinate axis (i) with attribute difference P10 and base P0.
- v_interp_p2_f32: Computes D = P20 * S + D
  Accumulates the second barycentric coordinate axis (j) with attribute difference P20 onto the accumulated D from p1.
Combined perspective evaluation:
  D = P0 + i * (P1 - P0) + j * (P2 - P0) = (1 - i - j) * P0 + i * P1 + j * P2
- v_interp_mov_f32: Computes direct attribute parameter extraction (flat shading or direct parameter access):
  param_sel 0 (p0): P0
  param_sel 1 (p10): P10 = P1 - P0
  param_sel 2 (p20): P20 = P2 - P0

Perspective-correct Barycentrics:
  Given triangle vertex clip depths W0, W1, W2 > 0 and screen barycentrics (l0, l1, l2):
    inv_W = l0 / W0 + l1 / W1 + l2 / W2
    i_persp = (l1 / W1) / inv_W
    j_persp = (l2 / W2) / inv_W
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
from typing import List, Optional, Tuple, Union
import unittest

# IEEE 754 binary32 architectural constants
SIGN_MASK_F32: int = 0x8000_0000
EXP_MASK_F32: int = 0x7F80_0000
FRAC_MASK_F32: int = 0x007F_FFFF
QNAN_BIT_F32: int = 0x0040_0000

CANONICAL_QNAN_F32: int = 0x7FC0_0000
POS_ZERO_F32: int = 0x0000_0000
NEG_ZERO_F32: int = 0x8000_0000
POS_INF_F32: int = 0x7F80_0000
NEG_INF_F32: int = 0xFF80_0000
ONE_F32: int = 0x3F80_0000

MASK32: int = 0xFFFF_FFFF


def f32_to_bits(val: float) -> int:
    """Convert a Python float to 32-bit unsigned IEEE 754 integer bits."""
    try:
        return struct.unpack("<I", struct.pack("<f", float(val)))[0]
    except OverflowError:
        return POS_INF_F32 if val > 0 else NEG_INF_F32


def bits_to_f32(bits: int) -> float:
    """Convert 32-bit integer bits to a Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_round(val: float) -> float:
    """Round a float to single-precision IEEE 754 binary32."""
    try:
        return struct.unpack("<f", struct.pack("<f", float(val)))[0]
    except OverflowError:
        return float("inf") if val > 0 else float("-inf")


def is_nan_f32(bits: int) -> bool:
    return ((bits & EXP_MASK_F32) == EXP_MASK_F32) and ((bits & FRAC_MASK_F32) != 0)


def is_inf_f32(bits: int) -> bool:
    return ((bits & EXP_MASK_F32) == EXP_MASK_F32) and ((bits & FRAC_MASK_F32) == 0)


def is_zero_f32(bits: int) -> bool:
    return (bits & ~SIGN_MASK_F32) == 0


def apply_f32_modifiers(bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply VOP3 source modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & MASK32
    if abs_mod:
        res &= ~SIGN_MASK_F32
    if neg_mod:
        res ^= SIGN_MASK_F32
    return res


def apply_f32_clamp(val_bits: int) -> int:
    """Apply RDNA2 clamp [0.0, 1.0] modifier; NaNs clamp to +0.0."""
    if is_nan_f32(val_bits):
        return POS_ZERO_F32
    val = bits_to_f32(val_bits)
    if val < 0.0:
        return POS_ZERO_F32
    if val > 1.0:
        return ONE_F32
    return val_bits


def apply_f32_omod(val_bits: int, omod: int) -> int:
    """Apply RDNA2 output modifier: 0: none, 1: mul:2, 2: mul:4, 3: div:2."""
    if omod == 0:
        return val_bits
    if is_nan_f32(val_bits) or is_inf_f32(val_bits):
        return val_bits
    val = bits_to_f32(val_bits)
    if omod == 1:
        res = val * 2.0
    elif omod == 2:
        res = val * 4.0
    elif omod == 3:
        res = val * 0.5
    else:
        raise ValueError(f"Invalid omod {omod}")
    return f32_to_bits(res)


def raw_fma_f32(a: float, b: float, c: float) -> float:
    """Evaluate single-precision fused multiply-add (a * b) + c using binary64 intermediate."""
    if math.isnan(a) or math.isnan(b) or math.isnan(c):
        return float("nan")
    # Handling inf * 0
    if (math.isinf(a) and b == 0.0) or (a == 0.0 and math.isinf(b)):
        return float("nan")
    # Python 64-bit float has 53 mantissa bits; binary32 product has at most 48 mantissa bits.
    # Therefore a * b is exact with 0 rounding error before adding c.
    prod = a * b
    if math.isinf(prod) and math.isinf(c):
        if (prod > 0 and c < 0) or (prod < 0 and c > 0):
            return float("nan")
    res = prod + c
    return f32_round(res)


def oracle_v_interp_p1_f32(
    s_bits: int,
    p10_bits: int,
    p0_bits: int,
    abs_s: bool = False,
    neg_s: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> int:
    """Reference mathematical oracle for v_interp_p1_f32: D = P10 * S + P0."""
    s_mod = apply_f32_modifiers(s_bits, abs_mod=abs_s, neg_mod=neg_s)
    s_val = bits_to_f32(s_mod)
    p10_val = bits_to_f32(p10_bits)
    p0_val = bits_to_f32(p0_bits)

    res_f = raw_fma_f32(p10_val, s_val, p0_val)
    res_bits = f32_to_bits(res_f)

    if math.isnan(res_f):
        res_bits = CANONICAL_QNAN_F32

    if omod != 0:
        res_bits = apply_f32_omod(res_bits, omod)
    if clamp:
        res_bits = apply_f32_clamp(res_bits)
    return res_bits


def oracle_v_interp_p2_f32(
    s_bits: int,
    p20_bits: int,
    p1_accum_bits: int,
    abs_s: bool = False,
    neg_s: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> int:
    """Reference mathematical oracle for v_interp_p2_f32: D = P20 * S + D."""
    s_mod = apply_f32_modifiers(s_bits, abs_mod=abs_s, neg_mod=neg_s)
    s_val = bits_to_f32(s_mod)
    p20_val = bits_to_f32(p20_bits)
    d_val = bits_to_f32(p1_accum_bits)

    res_f = raw_fma_f32(p20_val, s_val, d_val)
    res_bits = f32_to_bits(res_f)

    if math.isnan(res_f):
        res_bits = CANONICAL_QNAN_F32

    if omod != 0:
        res_bits = apply_f32_omod(res_bits, omod)
    if clamp:
        res_bits = apply_f32_clamp(res_bits)
    return res_bits


def oracle_v_interp_mov_f32(
    param_sel: int,
    p0_bits: int,
    p10_bits: int,
    p20_bits: int,
    abs_val: bool = False,
    neg_val: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> int:
    """Reference mathematical oracle for v_interp_mov_f32."""
    if param_sel == 0:
        src_bits = p0_bits
    elif param_sel == 1:
        src_bits = p10_bits
    elif param_sel == 2:
        src_bits = p20_bits
    else:
        raise ValueError(f"Invalid param_sel {param_sel}, expected 0 (p0), 1 (p10), or 2 (p20)")

    res_bits = apply_f32_modifiers(src_bits, abs_mod=abs_val, neg_mod=neg_val)
    if omod != 0:
        res_bits = apply_f32_omod(res_bits, omod)
    if clamp:
        res_bits = apply_f32_clamp(res_bits)
    return res_bits


def oracle_interpolate_triangle(
    i_coord: float,
    j_coord: float,
    p0: float,
    p1: float,
    p2: float,
    clamp: bool = False,
) -> float:
    """End-to-end two-stage interpolation oracle: p1 then p2."""
    p10 = f32_round(p1 - p0)
    p20 = f32_round(p2 - p0)

    p1_bits = oracle_v_interp_p1_f32(f32_to_bits(i_coord), f32_to_bits(p10), f32_to_bits(p0))
    p2_bits = oracle_v_interp_p2_f32(f32_to_bits(j_coord), f32_to_bits(p20), p1_bits, clamp=clamp)
    return bits_to_f32(p2_bits)


def compute_perspective_coords(
    l0: float, l1: float, l2: float, w0: float, w1: float, w2: float
) -> Tuple[float, float]:
    """Calculate perspective-correct barycentric weights (i_persp, j_persp)."""
    inv_w0 = 1.0 / w0
    inv_w1 = 1.0 / w1
    inv_w2 = 1.0 / w2
    inv_w_total = l0 * inv_w0 + l1 * inv_w1 + l2 * inv_w2
    i_persp = (l1 * inv_w1) / inv_w_total
    j_persp = (l2 * inv_w2) / inv_w_total
    return f32_round(i_persp), f32_round(j_persp)


@dataclass
class AttributeChannel:
    """Architectural model for one channel of a shader attribute (P0, P10, P20)."""
    p0: float = 0.0
    p10: float = 0.0
    p20: float = 0.0

    @classmethod
    def from_vertices(cls, v0: float, v1: float, v2: float) -> AttributeChannel:
        return cls(
            p0=f32_round(v0),
            p10=f32_round(v1 - v0),
            p20=f32_round(v2 - v0),
        )

    def get_bits(self, param_sel: int) -> int:
        if param_sel == 0:
            return f32_to_bits(self.p0)
        elif param_sel == 1:
            return f32_to_bits(self.p10)
        elif param_sel == 2:
            return f32_to_bits(self.p20)
        raise ValueError(f"Unknown param_sel: {param_sel}")


@dataclass
class AttributeFile:
    """Architectural model of the RDNA2 SPI attribute parameter cache (32 attributes x 4 channels)."""
    attrs: List[List[AttributeChannel]] = field(
        default_factory=lambda: [[AttributeChannel() for _ in range(4)] for _ in range(32)]
    )

    def set_attribute(self, attr_idx: int, chan_idx: int, v0: float, v1: float, v2: float) -> None:
        self.attrs[attr_idx][chan_idx] = AttributeChannel.from_vertices(v0, v1, v2)

    def get_channel(self, attr_idx: int, chan_idx: int) -> AttributeChannel:
        return self.attrs[attr_idx][chan_idx]


@dataclass
class VGPRFile:
    """Architectural model of 256 32-bit Vector General Purpose Registers (VGPRs)."""
    regs: List[int] = field(default_factory=lambda: [0] * 256)

    def get(self, idx: int) -> int:
        return self.regs[idx & 0xFF]

    def set(self, idx: int, val: int) -> None:
        self.regs[idx & 0xFF] = val & MASK32

    def get_f32(self, idx: int) -> float:
        return bits_to_f32(self.get(idx))

    def set_f32(self, idx: int, val: float) -> None:
        self.set(idx, f32_to_bits(val))


class TestInterpMathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural models."""

    def test_basic_interpolation_identities(self) -> None:
        p0, p1, p2 = 10.0, 20.0, 40.0
        # at vertex 0: (i=0, j=0) -> p0
        self.assertAlmostEqual(oracle_interpolate_triangle(0.0, 0.0, p0, p1, p2), 10.0)
        # at vertex 1: (i=1, j=0) -> p1
        self.assertAlmostEqual(oracle_interpolate_triangle(1.0, 0.0, p0, p1, p2), 20.0)
        # at vertex 2: (i=0, j=1) -> p2
        self.assertAlmostEqual(oracle_interpolate_triangle(0.0, 1.0, p0, p1, p2), 40.0)
        # midpoint of edge 0-1: (i=0.5, j=0) -> 15.0
        self.assertAlmostEqual(oracle_interpolate_triangle(0.5, 0.0, p0, p1, p2), 15.0)
        # centroid: (i=1/3, j=1/3) -> (10 + 20 + 40)/3 = 70/3 = 23.33333...
        self.assertAlmostEqual(oracle_interpolate_triangle(1.0 / 3.0, 1.0 / 3.0, p0, p1, p2), 70.0 / 3.0, places=5)

    def test_interp_mov_f32(self) -> None:
        p0 = f32_to_bits(12.5)
        p10 = f32_to_bits(5.0)
        p20 = f32_to_bits(-7.5)

        self.assertEqual(oracle_v_interp_mov_f32(0, p0, p10, p20), p0)
        self.assertEqual(oracle_v_interp_mov_f32(1, p0, p10, p20), p10)
        self.assertEqual(oracle_v_interp_mov_f32(2, p0, p10, p20), p20)

    def test_source_modifiers(self) -> None:
        p0 = f32_to_bits(10.0)
        p10 = f32_to_bits(2.0)
        s = f32_to_bits(-3.0)

        # with abs_s: s becomes +3.0 -> 2.0 * 3.0 + 10.0 = 16.0
        res_abs = oracle_v_interp_p1_f32(s, p10, p0, abs_s=True)
        self.assertEqual(bits_to_f32(res_abs), 16.0)

        # with neg_s: s becomes -(-3.0) = +3.0 -> 16.0
        res_neg = oracle_v_interp_p1_f32(s, p10, p0, neg_s=True)
        self.assertEqual(bits_to_f32(res_neg), 16.0)

        # with abs_s and neg_s: s becomes -|s| = -3.0 -> 2.0 * (-3.0) + 10.0 = 4.0
        res_both = oracle_v_interp_p1_f32(s, p10, p0, abs_s=True, neg_s=True)
        self.assertEqual(bits_to_f32(res_both), 4.0)

    def test_clamp_and_omod(self) -> None:
        p0 = f32_to_bits(0.5)
        p10 = f32_to_bits(0.75)
        s = f32_to_bits(1.0)
        # res = 0.75 * 1.0 + 0.5 = 1.25 -> clamped to 1.0
        res_clamped = oracle_v_interp_p1_f32(s, p10, p0, clamp=True)
        self.assertEqual(bits_to_f32(res_clamped), 1.0)

        # omod 1 (mul:2): (0.75 * 1.0 + 0.5) * 2 = 1.25 * 2 = 2.5
        res_omod = oracle_v_interp_p1_f32(s, p10, p0, omod=1)
        self.assertEqual(bits_to_f32(res_omod), 2.5)

    def test_perspective_correct_barycentrics(self) -> None:
        # Depth values: W0 = 1.0, W1 = 2.0, W2 = 4.0
        # Screen barycentrics: l0 = 0.2, l1 = 0.5, l2 = 0.3
        i_persp, j_persp = compute_perspective_coords(0.2, 0.5, 0.3, 1.0, 2.0, 4.0)
        self.assertAlmostEqual(i_persp, 10.0 / 21.0, places=5)
        self.assertAlmostEqual(j_persp, 3.0 / 21.0, places=5)

    def test_architectural_models(self) -> None:
        af = AttributeFile()
        af.set_attribute(3, 1, 100.0, 200.0, 300.0)
        chan = af.get_channel(3, 1)
        self.assertEqual(chan.p0, 100.0)
        self.assertEqual(chan.p10, 100.0)
        self.assertEqual(chan.p20, 200.0)

        vgprs = VGPRFile()
        vgprs.set_f32(5, 42.5)
        self.assertEqual(vgprs.get_f32(5), 42.5)


if __name__ == "__main__":
    unittest.main()
