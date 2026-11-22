from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_pack_b32_i16.

Tests the vector 16-bit integer packing instruction:
    v_pack_b32_i16 D, S0, S1 [op_sel:[s0, s1, 0, 0]] [clamp]
Lowered semantics:
    dst[15:0]  = half_word(S0, op_sel[0])  # 16-bit signed integer channel 0
    dst[31:16] = half_word(S1, op_sel[1])  # 16-bit signed integer channel 1
with full bit-exact two's complement signed integer preservation.
Supports input modifiers per 16-bit integer source:
    - abs (|src|: absolute value in 16-bit signed range [-32768, 32767])
    - neg (-src: two's complement negation in 16-bit signed range)
Operand selection:
    - op_sel[0] selects low half (0: bits [15:0]) or high half (1: bits [31:16]) of S0
    - op_sel[1] selects low half (0: bits [15:0]) or high half (1: bits [31:16]) of S1
Clamp modifier:
    - clamp saturates each 16-bit signed channel to [-32768, 32767].
Architectural model:
    32-bit vector registers (VGPRs) and scalar registers (SGPRs).
    Overwrites all 32 bits of the destination VGPR: dst = (ch1 << 16) | (ch0 & 0xFFFF).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import unittest
from typing import List, Optional, Sequence, Tuple, Union

# Architectural bitwise masks and constants
MASK32: int = 0xFFFF_FFFF
MASK16: int = 0xFFFF
SIGN16: int = 0x8000
ABS16: int = 0x7FFF

INT16_MIN: int = -32768
INT16_MAX: int = 32767
UINT16_MAX: int = 65535


def s16(val: int) -> int:
    """Interpret 16-bit integer representation as signed two's-complement [-32768, 32767]."""
    val &= MASK16
    return val - 0x10000 if (val & SIGN16) else val


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit word [0, 65535]."""
    return val & MASK16


def extract_half(val32: int, high_half: bool) -> int:
    """Extract low 16 bits (high_half=False) or high 16 bits (high_half=True)."""
    return ((val32 >> 16) & MASK16) if high_half else (val32 & MASK16)


def apply_i16_modifiers(
    bits16: int,
    abs_mod: bool = False,
    neg_mod: bool = False,
    clamp: bool = False,
) -> int:
    """Apply RDNA2 VOP3 source modifiers (abs, neg) and clamp to a 16-bit integer word.

    In signed two's-complement arithmetic:
    - abs computes |s16(val)|. For INT16_MIN (-32768):
        without clamp, wraps to -32768 (0x8000) in 16-bit two's complement;
        with clamp, saturates to INT16_MAX (32767, 0x7FFF).
    - neg computes two's complement negation: -val (mod 65536).
    - clamp saturates result to [INT16_MIN, INT16_MAX].
    """
    val = s16(bits16)
    if abs_mod:
        val = abs(val)
        if clamp and val > INT16_MAX:
            val = INT16_MAX
    if neg_mod:
        val = -val
    if clamp:
        val = max(INT16_MIN, min(INT16_MAX, val))
    return val & MASK16


def pack_b32_i16(lo_half: int, hi_half: int) -> int:
    """Pack two 16-bit halves into an unsigned 32-bit dword."""
    return (((hi_half & MASK16) << 16) | (lo_half & MASK16)) & MASK32


def oracle_v_pack_b32_i16(
    src0_bits: int,
    src1_bits: int,
    op_sel: Tuple[int, int] = (0, 0),
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference architectural oracle for RDNA2 v_pack_b32_i16 instruction.

    Packs half_word(src0, op_sel[0]) into dst[15:0] and
          half_word(src1, op_sel[1]) into dst[31:16],
    applying respective source modifiers and optional channel clamping.
    """
    h0 = extract_half(src0_bits, bool(op_sel[0]))
    h1 = extract_half(src1_bits, bool(op_sel[1]))

    h0_mod = apply_i16_modifiers(h0, abs_mod=abs0, neg_mod=neg0, clamp=clamp)
    h1_mod = apply_i16_modifiers(h1, abs_mod=abs1, neg_mod=neg1, clamp=clamp)

    return pack_b32_i16(h0_mod, h1_mod)


class TestPackB32I16ModelAndOracle(unittest.TestCase):
    """Architectural model and mathematical reference oracle tests for v_pack_b32_i16."""

    def test_basic_packing_default_op_sel(self) -> None:
        # Pack 100 into low half and -200 into high half
        s0 = 100
        s1 = (-200) & MASK16
        res = oracle_v_pack_b32_i16(s0, s1, op_sel=(0, 0))
        expected = (s1 << 16) | s0
        self.assertEqual(res, expected)
        self.assertEqual(s16(extract_half(res, False)), 100)
        self.assertEqual(s16(extract_half(res, True)), -200)

    def test_op_sel_permutations(self) -> None:
        # s0 has 10 in low half, 20 in high half
        # s1 has -30 in low half, -40 in high half
        s0 = (20 << 16) | (10 & MASK16)
        s1 = (((-40) & MASK16) << 16) | ((-30) & MASK16)

        # op_sel: (0, 0) -> s0[15:0]=10, s1[15:0]=-30
        res00 = oracle_v_pack_b32_i16(s0, s1, (0, 0))
        self.assertEqual(extract_half(res00, False), 10)
        self.assertEqual(s16(extract_half(res00, True)), -30)

        # op_sel: (1, 0) -> s0[31:16]=20, s1[15:0]=-30
        res10 = oracle_v_pack_b32_i16(s0, s1, (1, 0))
        self.assertEqual(extract_half(res10, False), 20)
        self.assertEqual(s16(extract_half(res10, True)), -30)

        # op_sel: (0, 1) -> s0[15:0]=10, s1[31:16]=-40
        res01 = oracle_v_pack_b32_i16(s0, s1, (0, 1))
        self.assertEqual(extract_half(res01, False), 10)
        self.assertEqual(s16(extract_half(res01, True)), -40)

        # op_sel: (1, 1) -> s0[31:16]=20, s1[31:16]=-40
        res11 = oracle_v_pack_b32_i16(s0, s1, (1, 1))
        self.assertEqual(extract_half(res11, False), 20)
        self.assertEqual(s16(extract_half(res11, True)), -40)

    def test_signed_integer_ranges(self) -> None:
        # Test boundary values: INT16_MIN (-32768 = 0x8000), INT16_MAX (32767 = 0x7FFF), -1 (0xFFFF), 0
        min_i16 = 0x8000
        max_i16 = 0x7FFF
        minus_one = 0xFFFF
        zero = 0x0000

        res1 = oracle_v_pack_b32_i16(min_i16, max_i16)
        self.assertEqual(res1, (max_i16 << 16) | min_i16)
        self.assertEqual(s16(extract_half(res1, False)), INT16_MIN)
        self.assertEqual(s16(extract_half(res1, True)), INT16_MAX)

        res2 = oracle_v_pack_b32_i16(minus_one, zero)
        self.assertEqual(res2, (zero << 16) | minus_one)
        self.assertEqual(s16(extract_half(res2, False)), -1)
        self.assertEqual(s16(extract_half(res2, True)), 0)

    def test_source_modifiers_abs_and_neg(self) -> None:
        # abs modifier on negative values
        self.assertEqual(apply_i16_modifiers((-500) & MASK16, abs_mod=True), 500)
        self.assertEqual(apply_i16_modifiers(500, abs_mod=True), 500)

        # neg modifier on positive and negative values
        self.assertEqual(s16(apply_i16_modifiers(500, neg_mod=True)), -500)
        self.assertEqual(s16(apply_i16_modifiers((-500) & MASK16, neg_mod=True)), 500)

        # Both abs and neg (-|x|)
        self.assertEqual(s16(apply_i16_modifiers(500, abs_mod=True, neg_mod=True)), -500)
        self.assertEqual(s16(apply_i16_modifiers((-500) & MASK16, abs_mod=True, neg_mod=True)), -500)

        # Apply via oracle
        res = oracle_v_pack_b32_i16(50, (-100) & MASK16, neg0=True, abs1=True)
        self.assertEqual(s16(extract_half(res, False)), -50)
        self.assertEqual(s16(extract_half(res, True)), 100)

    def test_clamp_modifier_saturation(self) -> None:
        # abs(-32768) without clamp wraps to -32768 (0x8000)
        self.assertEqual(s16(apply_i16_modifiers(0x8000, abs_mod=True, clamp=False)), INT16_MIN)
        # abs(-32768) with clamp saturates to 32767 (0x7FFF)
        self.assertEqual(s16(apply_i16_modifiers(0x8000, abs_mod=True, clamp=True)), INT16_MAX)

        res_clamped = oracle_v_pack_b32_i16(0x8000, 0x8000, abs0=True, abs1=True, clamp=True)
        self.assertEqual(res_clamped, (0x7FFF << 16) | 0x7FFF)

    def test_full_dword_packing_and_extraction(self) -> None:
        # Packing random bit patterns round-trips exactly
        s0 = 0x1234
        s1 = 0xABCD
        res = oracle_v_pack_b32_i16(s0, s1)
        self.assertEqual(res, 0xABCD1234)
        self.assertEqual(extract_half(res, False), 0x1234)
        self.assertEqual(extract_half(res, True), 0xABCD)


if __name__ == "__main__":
    unittest.main()
