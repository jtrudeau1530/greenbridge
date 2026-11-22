from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_pack_b32_u16.

AMD RDNA VALU Instruction Set Architecture (ISA):
    v_pack_b32_u16 D, S0, S1 [op_sel:[s0, s1]] [clamp]
Operation:
    Packs two 16-bit integer values into a 32-bit vector register (VGPR).
    dst[15:0]  = half_word(S0, op_sel[0]) [clamped to [0, 0xFFFF] if clamp]
    dst[31:16] = half_word(S1, op_sel[1]) [clamped to [0, 0xFFFF] if clamp]

Modifiers and Semantics:
    - op_sel:[s0, s1]:
        s0 = 0: select low half-word (bits 15:0) of S0
        s0 = 1: select high half-word (bits 31:16) of S0
        s1 = 0: select low half-word (bits 15:0) of S1
        s1 = 1: select high half-word (bits 31:16) of S1
    - Half-word Truncation:
        When clamp is not asserted, bits outside the selected 16-bit window are
        discarded (masked to 0xFFFF).
    - Clamping / Saturation:
        When clamp is asserted on integer packing, values outside [0, 65535] saturate:
          - Unsigned saturation: values > 0xFFFF saturate to 0xFFFF.
          - Signed saturation: negative values saturate to 0, values > 0xFFFF saturate to 0xFFFF.

Architectural model:
    32-bit vector registers (VGPRs) and scalar registers (SGPRs).
    Unlike sub-dword writes, v_pack_b32_u16 overwrites all 32 bits of the destination VGPR.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import struct
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFF_FFFF
MASK16: int = 0xFFFF
SIGN16: int = 0x8000
MAX_U16: int = 0xFFFF


def s32(x: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    x &= MASK32
    return x - (1 << 32) if x & 0x80000000 else x


def ref_extract_half(val32: int, hi: bool) -> int:
    """Extract low 16 bits (hi=False) or high 16 bits (hi=True)."""
    val32 &= MASK32
    return ((val32 >> 16) if hi else val32) & MASK16


def ref_clamp_u16(val: int, signed: bool = False) -> int:
    """Clamp integer value to unsigned 16-bit range [0, 65535]."""
    if signed:
        s = s32(val)
        if s < 0:
            return 0
        if s > MAX_U16:
            return MAX_U16
        return s & MASK16
    u = val & MASK32
    return min(u, MAX_U16)


def pack_b32_u16(lo_half: int, hi_half: int) -> int:
    """Pack two 16-bit halves into an unsigned 32-bit dword."""
    return ((hi_half & MASK16) << 16) | (lo_half & MASK16)


def oracle_v_pack_b32_u16(
    src0_bits: int,
    src1_bits: int,
    op_sel: Tuple[int, int] = (0, 0),
    clamp: bool = False,
    signed: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_pack_b32_u16."""
    s0 = src0_bits & MASK32
    s1 = src1_bits & MASK32

    raw0 = (s0 >> 16) if op_sel[0] else s0
    raw1 = (s1 >> 16) if op_sel[1] else s1

    if clamp:
        h0 = ref_clamp_u16(raw0, signed=signed)
        h1 = ref_clamp_u16(raw1, signed=signed)
    else:
        h0 = raw0 & MASK16
        h1 = raw1 & MASK16

    return pack_b32_u16(h0, h1)


class TestValuPackB32U16ModelAndOracle(unittest.TestCase):
    """Architectural model and reference mathematical oracle tests for v_pack_b32_u16."""

    def test_basic_packing_identity(self) -> None:
        res = oracle_v_pack_b32_u16(0x00001234, 0x00005678)
        self.assertEqual(res, 0x56781234)

    def test_op_sel_permutations(self) -> None:
        s0 = 0xA1B2C3D4
        s1 = 0xE5F60718
        # op_sel (0, 0): s0 low (C3D4), s1 low (0718)
        self.assertEqual(oracle_v_pack_b32_u16(s0, s1, op_sel=(0, 0)), 0x0718C3D4)
        # op_sel (1, 0): s0 high (A1B2), s1 low (0718)
        self.assertEqual(oracle_v_pack_b32_u16(s0, s1, op_sel=(1, 0)), 0x0718A1B2)
        # op_sel (0, 1): s0 low (C3D4), s1 high (E5F6)
        self.assertEqual(oracle_v_pack_b32_u16(s0, s1, op_sel=(0, 1)), 0xE5F6C3D4)
        # op_sel (1, 1): s0 high (A1B2), s1 high (E5F6)
        self.assertEqual(oracle_v_pack_b32_u16(s0, s1, op_sel=(1, 1)), 0xE5F6A1B2)

    def test_half_word_truncation_without_clamp(self) -> None:
        res = oracle_v_pack_b32_u16(0xDEADBEEF, 0xCAFEBABE, op_sel=(0, 0), clamp=False)
        self.assertEqual(res, (0xBABE << 16) | 0xBEEF)

    def test_unsigned_clamping_saturation(self) -> None:
        # Values exceeding 0xFFFF clamp to 0xFFFF
        res = oracle_v_pack_b32_u16(0x00020000, 0x00018000, op_sel=(0, 0), clamp=True, signed=False)
        self.assertEqual(res, 0xFFFFFFFF)

        res_partial = oracle_v_pack_b32_u16(0x00000042, 0x00050000, op_sel=(0, 0), clamp=True, signed=False)
        self.assertEqual(res_partial, 0xFFFF0042)

    def test_signed_clamping_saturation(self) -> None:
        # Negative 32-bit values clamp to 0, values > 0xFFFF clamp to 0xFFFF
        neg_val = 0xFFFFFFFF  # -1 in s32
        large_val = 0x00020000  # 131072
        res = oracle_v_pack_b32_u16(neg_val, large_val, clamp=True, signed=True)
        self.assertEqual(res, 0xFFFF0000)

        res2 = oracle_v_pack_b32_u16(large_val, neg_val, clamp=True, signed=True)
        self.assertEqual(res2, 0x0000FFFF)

    def test_dword_overwrite_semantics(self) -> None:
        res = oracle_v_pack_b32_u16(0, 0)
        self.assertEqual(res, 0)
        self.assertEqual(ref_extract_half(res, False), 0)
        self.assertEqual(ref_extract_half(res, True), 0)


if __name__ == "__main__":
    unittest.main()
