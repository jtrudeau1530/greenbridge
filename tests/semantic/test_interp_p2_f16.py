from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_interp_p2_f16.

RDNA2 VINTRP instruction family:
    v_interp_p2_f16 vD, vSrc(J), attrN.chan [high] [clamp]

Instruction semantics:
    Evaluates phase 2 of perspective parameter interpolation for half-precision float attributes:
        D = p1_accum + J * (P2 - P0)   (or P20 * J + p1_accum)
    where:
        - `p1_accum`: f32 accumulator input (produced by v_interp_p1ll_f16 or v_interp_p1lv_f16),
          stored in destination register vD.
        - `J`: barycentric coordinate j (typically in vSrc / v1).
        - `P20` = P2 - P0: slope parameter for attribute N channel chan (p20 parameter in LDS/parameter cache).
          If `high` source modifier is set: selects high 16-bit half [31:16] of the 32-bit attribute word.
          Otherwise: selects low 16-bit half [15:0] of the 32-bit attribute word.
        - The fused multiply-add accumulation is performed in floating-point and rounded once to binary16 (f16).
        - If `clamp`: clamps the resulting f16 to [0.0, 1.0], mapping NaNs to +0.0.
        - Destination write:
          If `high` destination modifier (or op_sel destination bit) is set:
              writes the 16-bit result into vD[31:16], strictly preserving vD[15:0].
          Otherwise (default low half):
              writes the 16-bit result into vD[15:0], strictly preserving vD[31:16] (per SHADER-430 / gfx1036).

Zero game data; standard library Python only; Python 3.9+ compatible.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import sys
from typing import List, Optional, Tuple, Union
import unittest

# IEEE 754 binary16 (half-precision) constants
H_POS_ZERO: int = 0x0000
H_NEG_ZERO: int = 0x8000
H_ONE: int = 0x3C00
H_NEG_ONE: int = 0xBC00
H_TWO: int = 0x4000
H_HALF: int = 0x3800
H_POS_INF: int = 0x7C00
H_NEG_INF: int = 0xFC00
H_QNAN: int = 0x7E00
H_SNAN: int = 0x7C01
H_SUB_MIN: int = 0x0001
H_SUB_MAX: int = 0x03FF
H_MIN_NORMAL: int = 0x0400
H_MAX_NORMAL: int = 0x7BFF

# IEEE 754 binary32 constants
F32_POS_ZERO: int = 0x0000_0000
F32_NEG_ZERO: int = 0x8000_0000
F32_POS_INF: int = 0x7F80_0000
F32_NEG_INF: int = 0xFF80_0000
F32_QNAN: int = 0x7FC0_0000

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF


def f16_to_u16(val: float) -> int:
    """Pack Python float into IEEE 754 binary16 unsigned 16-bit int."""
    try:
        return struct.unpack("<H", struct.pack("<e", float(val)))[0]
    except OverflowError:
        return H_POS_INF if val > 0 else H_NEG_INF


def u16_to_f16(bits: int) -> float:
    """Unpack IEEE 754 binary16 unsigned 16-bit int into Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def f32_to_u32(val: float) -> int:
    """Pack Python float into IEEE 754 binary32 unsigned 32-bit int."""
    try:
        return struct.unpack("<I", struct.pack("<f", float(val)))[0]
    except OverflowError:
        return F32_POS_INF if val > 0 else F32_NEG_INF


def u32_to_f32(bits: int) -> float:
    """Unpack IEEE 754 binary32 unsigned 32-bit int into Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def pack_f16_pair(lo_u16: int, hi_u16: int) -> int:
    """Pack two 16-bit words into a single 32-bit dword [hi:lo]."""
    return (lo_u16 & MASK16) | ((hi_u16 & MASK16) << 16)


def unpack_f16_pair(val_u32: int) -> tuple[int, int]:
    """Unpack a 32-bit dword into (low 16-bit word, high 16-bit word)."""
    return val_u32 & MASK16, (val_u32 >> 16) & MASK16


def is_nan_f16(bits: int) -> bool:
    return ((bits & 0x7C00) == 0x7C00) and ((bits & 0x03FF) != 0)


def is_nan_f32(bits: int) -> bool:
    return ((bits & 0x7F80_0000) == 0x7F80_0000) and ((bits & 0x007F_FFFF) != 0)


def apply_clamp_f16(bits_u16: int) -> int:
    """Apply RDNA2 clamp modifier [0.0, 1.0] to half-precision float, mapping NaNs to +0.0."""
    val = u16_to_f16(bits_u16)
    if math.isnan(val) or bits_u16 == H_NEG_ZERO:
        return H_POS_ZERO
    if val < 0.0:
        return H_POS_ZERO
    if val > 1.0:
        return H_ONE
    return f16_to_u16(val)


def apply_f32_modifiers(bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply VOP3 source modifiers (abs, neg) on 32-bit float."""
    res = bits & MASK32
    if abs_mod:
        res &= ~0x8000_0000
    if neg_mod:
        res ^= 0x8000_0000
    return res


def ref_interp_p2_f16(
    j_coord_f32: float,
    p20_word_u32: int,
    p1_accum_f32: float,
    *,
    high_src: bool = False,
    high_dst: bool = False,
    old_vdst_u32: int = 0,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for v_interp_p2_f16.

    Computes:
        res = p1_accum + j * p20
    rounded to f16 (binary16), optionally clamped to [0.0, 1.0], and merged
    into old_vdst_u32 at low half [15:0] or high half [31:16], strictly preserving the untouched half.
    """
    lo_src, hi_src = unpack_f16_pair(p20_word_u32)
    selected_u16 = hi_src if high_src else lo_src
    p20_val = u16_to_f16(selected_u16)

    # Accumulate interpolation product
    if math.isnan(j_coord_f32) or math.isnan(p20_val) or math.isnan(p1_accum_f32):
        res_u16 = H_QNAN
    elif (math.isinf(j_coord_f32) and p20_val == 0.0) or (j_coord_f32 == 0.0 and math.isinf(p20_val)):
        res_u16 = H_QNAN
    else:
        prod = j_coord_f32 * p20_val
        if (math.isinf(prod) and math.isinf(p1_accum_f32)) and ((prod > 0 and p1_accum_f32 < 0) or (prod < 0 and p1_accum_f32 > 0)):
            res_u16 = H_QNAN
        else:
            accum = prod + p1_accum_f32
            res_u16 = f16_to_u16(accum)

    if clamp:
        res_u16 = apply_clamp_f16(res_u16)

    old_lo, old_hi = unpack_f16_pair(old_vdst_u32)
    if high_dst:
        return pack_f16_pair(old_lo, res_u16)
    else:
        return pack_f16_pair(res_u16, old_hi)


class AttributeBuffer:
    """Architectural model of the LDS/Parameter interpolation attribute cache."""

    def __init__(self) -> None:
        # data: [param 0..2][attr 0..31][chan 0..3] -> 32-bit uint
        # param 0: p0, param 1: p10, param 2: p20
        self.data = [[[0] * 4 for _ in range(32)] for _ in range(3)]

    def write_param_channel(self, param: int, attr: int, chan: int, val_u32: int) -> None:
        if not (0 <= param <= 2):
            raise ValueError(f"Invalid param index: {param}")
        if not (0 <= attr <= 31):
            raise ValueError(f"Invalid attribute index: {attr}")
        if not (0 <= chan <= 3):
            raise ValueError(f"Invalid channel index: {chan}")
        self.data[param][attr][chan] = val_u32 & MASK32

    def read_param_channel(self, param: int, attr: int, chan: int) -> int:
        if not (0 <= param <= 2):
            raise ValueError(f"Invalid param index: {param}")
        if not (0 <= attr <= 31):
            raise ValueError(f"Invalid attribute index: {attr}")
        if not (0 <= chan <= 3):
            raise ValueError(f"Invalid channel index: {chan}")
        return self.data[param][attr][chan]


class InterpolationRegisterFile:
    """Architectural model of 256 32-bit Vector General Purpose Registers (VGPRs)."""

    def __init__(self) -> None:
        self.vgprs = [0] * 256

    def read_vgpr_u32(self, idx: int) -> int:
        return self.vgprs[idx & 0xFF]

    def write_vgpr_u32(self, idx: int, val: int) -> None:
        self.vgprs[idx & 0xFF] = val & MASK32

    def read_vgpr_f32(self, idx: int) -> float:
        return u32_to_f32(self.vgprs[idx & 0xFF])

    def write_vgpr_f32(self, idx: int, val: float) -> None:
        self.vgprs[idx & 0xFF] = f32_to_u32(val)

    def read_vgpr_f16(self, idx: int, high: bool = False) -> float:
        lo, hi = unpack_f16_pair(self.vgprs[idx & 0xFF])
        return u16_to_f16(hi if high else lo)

    def write_vgpr_f16(self, idx: int, val_u16: int, high: bool = False) -> None:
        old_lo, old_hi = unpack_f16_pair(self.vgprs[idx & 0xFF])
        if high:
            self.vgprs[idx & 0xFF] = pack_f16_pair(old_lo, val_u16)
        else:
            self.vgprs[idx & 0xFF] = pack_f16_pair(val_u16, old_hi)


class TestInterpP2F16MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural models."""

    def test_basic_interpolation_accumulation(self):
        # j = 0.5, p1_accum = 2.0, p20 = 4.0 -> 2.0 + 0.5 * 4.0 = 4.0
        p20_word = pack_f16_pair(f16_to_u16(4.0), f16_to_u16(10.0))
        res = ref_interp_p2_f16(0.5, p20_word, 2.0, high_src=False, high_dst=False, old_vdst_u32=0)
        lo, hi = unpack_f16_pair(res)
        self.assertEqual(u16_to_f16(lo), 4.0)
        self.assertEqual(hi, 0)

    def test_high_source_selection(self):
        # p20_word: lo = 4.0, hi = 10.0. high_src=True chooses 10.0
        # j = 0.5, p1_accum = 3.0 -> 3.0 + 0.5 * 10.0 = 8.0
        p20_word = pack_f16_pair(f16_to_u16(4.0), f16_to_u16(10.0))
        res = ref_interp_p2_f16(0.5, p20_word, 3.0, high_src=True, high_dst=False, old_vdst_u32=0)
        lo, _ = unpack_f16_pair(res)
        self.assertEqual(u16_to_f16(lo), 8.0)

    def test_destination_half_preservation_gfx1036(self):
        # SHADER-430: v_interp_p2_f16 preserves the untouched 16-bit half of vdst
        canary_old = 0xDEADBEEF
        p20_word = pack_f16_pair(f16_to_u16(2.0), f16_to_u16(0.0))
        # j = 0.0, p1_accum = 1.0 -> 1.0 (0x3C00)
        res_lo = ref_interp_p2_f16(0.0, p20_word, 1.0, high_src=False, high_dst=False, old_vdst_u32=canary_old)
        # Low half modified, high half 0xDEAD preserved
        self.assertEqual(res_lo, 0xDEAD3C00)

        # High half modified, low half 0xBEEF preserved
        res_hi = ref_interp_p2_f16(0.0, p20_word, 1.0, high_src=False, high_dst=True, old_vdst_u32=canary_old)
        self.assertEqual(res_hi, 0x3C00BEEF)

    def test_clamp_modifier(self):
        p20_word = pack_f16_pair(f16_to_u16(2.0), 0)
        # Extrapolating beyond 1.0: j = 1.0, p1_accum = 1.5 -> 3.5 clamped to 1.0
        res_high_clamp = ref_interp_p2_f16(1.0, p20_word, 1.5, clamp=True)
        self.assertEqual(unpack_f16_pair(res_high_clamp)[0], H_ONE)

        # Extrapolating below 0.0: j = -2.0, p1_accum = 1.0 -> -3.0 clamped to +0.0
        res_low_clamp = ref_interp_p2_f16(-2.0, p20_word, 1.0, clamp=True)
        self.assertEqual(unpack_f16_pair(res_low_clamp)[0], H_POS_ZERO)

    def test_attribute_buffer_and_register_file(self):
        ab = AttributeBuffer()
        ab.write_param_channel(2, 4, 1, 0x12345678)
        self.assertEqual(ab.read_param_channel(2, 4, 1), 0x12345678)

        rf = InterpolationRegisterFile()
        rf.write_vgpr_u32(10, 0xAABBCCDD)
        rf.write_vgpr_f16(10, 0x3C00, high=False)
        self.assertEqual(rf.read_vgpr_u32(10), 0xAABB3C00)


if __name__ == "__main__":
    unittest.main()
