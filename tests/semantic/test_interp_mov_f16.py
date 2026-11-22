from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_interp_mov_f16.

Tests the half-precision parameter interpolation attribute move instruction:
    v_interp_mov_f16 vD, param, attrN.chan [high] [clamp]

Lowered semantics:
    Moves a 16-bit float attribute from parameter source (p0, p10, p20)
    corresponding to attribute N and channel (x, y, z, w).
    Supports:
      - `high` source modifier: selects high 16-bit half (bits [31:16]) or low 16-bit half (bits [15:0])
        of the packed 32-bit attribute channel word.
      - `high` destination modifier: selects whether to write into bits [31:16] or bits [15:0] of vD,
        strictly preserving the other 16-bit half of the 32-bit destination VGPR.
      - `clamp` modifier: clamps the 16-bit float result to [0.0, 1.0], mapping NaNs to +0.0.

Architectural model:
    Pixel shader VINTRP pipeline where attributes are packed per channel.
    Flat copying of provoking vertex (p0) or slope parameters (p10, p20).
"""

import math
import struct
import sys
import unittest

# IEEE 754 half-precision (binary16) constants
H_POS_ZERO = 0x0000
H_NEG_ZERO = 0x8000
H_ONE = 0x3C00
H_NEG_ONE = 0xBC00
H_TWO = 0x4000
H_HALF = 0x3800
H_POS_INF = 0x7C00
H_NEG_INF = 0xFC00
H_QNAN = 0x7E00
H_SNAN = 0x7C01
H_SUB_MIN = 0x0001
H_SUB_MAX = 0x03FF
H_MIN_NORMAL = 0x0400
H_MAX_NORMAL = 0x7BFF

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF


def f16_to_u16(val: float) -> int:
    """Pack Python float into IEEE 754 binary16 (half-precision) unsigned 16-bit int."""
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return H_POS_INF if val > 0 else H_NEG_INF


def u16_to_f16(bits: int) -> float:
    """Unpack IEEE 754 binary16 unsigned 16-bit int into Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def pack_f16_pair(lo_u16: int, hi_u16: int) -> int:
    """Pack two 16-bit words into a single 32-bit dword [hi:lo]."""
    return (lo_u16 & MASK16) | ((hi_u16 & MASK16) << 16)


def unpack_f16_pair(val_u32: int) -> tuple[int, int]:
    """Unpack a 32-bit dword into (low 16-bit word, high 16-bit word)."""
    return val_u32 & MASK16, (val_u32 >> 16) & MASK16


def apply_clamp_f16(bits_u16: int) -> int:
    """Apply RDNA2 clamp modifier [0.0, 1.0] to half-precision float, mapping NaNs to +0.0."""
    val = u16_to_f16(bits_u16)
    if math.isnan(val):
        return H_POS_ZERO
    if val < 0.0:
        return H_POS_ZERO
    if val > 1.0:
        return H_ONE
    return f16_to_u16(val)


def ref_interp_mov_f16(
    attr_word_u32: int,
    *,
    high_src: bool = False,
    high_dst: bool = False,
    old_dst_u32: int = 0,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for v_interp_mov_f16.

    Selects 16-bit source half from attr_word_u32, optionally clamps, and merges
    into old_dst_u32 at either high or low half, strictly preserving the untouched half.
    """
    lo_src, hi_src = unpack_f16_pair(attr_word_u32)
    selected_u16 = hi_src if high_src else lo_src

    if clamp:
        result_u16 = apply_clamp_f16(selected_u16)
    else:
        result_u16 = selected_u16

    old_lo, old_hi = unpack_f16_pair(old_dst_u32)
    if high_dst:
        return pack_f16_pair(old_lo, result_u16)
    else:
        return pack_f16_pair(result_u16, old_hi)


class AttributeBuffer:
    """Architectural model of the LDS/Parameter interpolation attribute cache.

    Maintains 32 attributes x 4 channels (X, Y, Z, W) x 3 parameters (p0, p10, p20).
    """

    def __init__(self) -> None:
        # storage: [param_type 0..2][attr 0..31][chan 0..3] -> 32-bit uint
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
    """Register file for 16-bit and 32-bit VGPR operations."""

    def __init__(self) -> None:
        self.vgprs = [0] * 256

    def read_vgpr_u32(self, idx: int) -> int:
        return self.vgprs[idx]

    def write_vgpr_u32(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32

    def read_vgpr_f16(self, idx: int, high: bool = False) -> float:
        lo, hi = unpack_f16_pair(self.vgprs[idx])
        return u16_to_f16(hi if high else lo)

    def write_vgpr_f16(self, idx: int, val_u16: int, high: bool = False) -> None:
        old_lo, old_hi = unpack_f16_pair(self.vgprs[idx])
        if high:
            self.vgprs[idx] = pack_f16_pair(old_lo, val_u16)
        else:
            self.vgprs[idx] = pack_f16_pair(val_u16, old_hi)


class TestInterpMovF16MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural models."""

    def test_basic_low_and_high_source_selection(self):
        # Attribute packed word with lo = 1.5 (0x3E00), hi = -2.0 (0xC000)
        attr_word = pack_f16_pair(0x3E00, 0xC000)

        # Move low source half to low destination half
        res_lo = ref_interp_mov_f16(attr_word, high_src=False, high_dst=False, old_dst_u32=0)
        self.assertEqual(res_lo, 0x00003E00)

        # Move high source half to low destination half
        res_hi = ref_interp_mov_f16(attr_word, high_src=True, high_dst=False, old_dst_u32=0)
        self.assertEqual(res_hi, 0x0000C000)

    def test_destination_half_preservation(self):
        # Pre-existing destination value
        canary_old = 0xAAAA5555  # hi: 0xAAAA, lo: 0x5555
        new_val_u16 = 0x3C00     # 1.0

        # Writing to low half preserves high half
        res_write_lo = ref_interp_mov_f16(new_val_u16, high_src=False, high_dst=False, old_dst_u32=canary_old)
        self.assertEqual(res_write_lo, 0xAAAA3C00)

        # Writing to high half preserves low half
        res_write_hi = ref_interp_mov_f16(new_val_u16, high_src=False, high_dst=True, old_dst_u32=canary_old)
        self.assertEqual(res_write_hi, 0x3C005555)

    def test_clamp_modifier(self):
        # 2.0 (0x4000) clamped to 1.0 (0x3C00)
        res_clamp_pos = ref_interp_mov_f16(0x4000, clamp=True)
        self.assertEqual(res_clamp_pos, H_ONE)

        # -1.0 (0xBC00) clamped to +0.0 (0x0000)
        res_clamp_neg = ref_interp_mov_f16(0xBC00, clamp=True)
        self.assertEqual(res_clamp_neg, H_POS_ZERO)

        # NaN clamped to +0.0
        res_clamp_nan = ref_interp_mov_f16(H_QNAN, clamp=True)
        self.assertEqual(res_clamp_nan, H_POS_ZERO)

    def test_attribute_buffer_storage(self):
        buf = AttributeBuffer()
        # Write p0, attr 5, chan Y (1)
        buf.write_param_channel(0, 5, 1, 0x12345678)
        self.assertEqual(buf.read_param_channel(0, 5, 1), 0x12345678)

        # Other channels remain 0
        self.assertEqual(buf.read_param_channel(0, 5, 0), 0)
        self.assertEqual(buf.read_param_channel(1, 5, 1), 0)


if __name__ == "__main__":
    unittest.main()
