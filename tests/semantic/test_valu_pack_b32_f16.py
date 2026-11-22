from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_pack_b32_f16.

Tests the vector 16-bit float packing instruction:
    v_pack_b32_f16 D, S0, S1 [op_sel:[s0, s1, 0, 0]]
Lowered semantics:
    dst[15:0]  = half_word(S0, op_sel[0])
    dst[31:16] = half_word(S1, op_sel[1])
with full bit-exact IEEE 754-2008 binary16 half-precision preservation.
Supports input modifiers per half-float source:
    - abs (|src|: clear sign bit 15)
    - neg (-src: invert sign bit 15)
Operand selection:
    - op_sel[0] selects low half (0) or high half (1) of S0
    - op_sel[1] selects low half (0) or high half (1) of S1
Architectural model:
    32-bit vector registers (VGPRs) and scalar registers (SGPRs).
    Unlike sub-dword writes, v_pack_b32_f16 overwrites all 32 bits of the destination register.
"""

import math
import struct
import unittest
from typing import Tuple

# IEEE 754-2008 binary16 architectural bitwise masks & constants
MASK32: int = 0xFFFF_FFFF
MASK16: int = 0xFFFF
SIGN16: int = 0x8000
ABS16: int = 0x7FFF
EXP_MASK16: int = 0x7C00
FRAC_MASK16: int = 0x03FF
QNAN_BIT16: int = 0x0200
CANONICAL_QNAN16: int = 0x7E00

POS_ZERO: int = 0x0000              # +0.0
NEG_ZERO: int = 0x8000              # -0.0
MIN_SUBNORMAL: int = 0x0001         # 2^-24 (smallest positive subnormal)
MAX_SUBNORMAL: int = 0x03FF         # largest positive subnormal
MIN_SUBNORMAL_NEG: int = 0x8001     # smallest negative subnormal
MAX_SUBNORMAL_NEG: int = 0x83FF     # largest negative subnormal
MIN_NORMAL: int = 0x0400            # 2^-14 (smallest positive normal)
HALF: int = 0x3800                  # 0.5
ONE: int = 0x3C00                   # 1.0
TWO: int = 0x4000                   # 2.0
FOUR: int = 0x4400                  # 4.0
MAX_FINITE: int = 0x7BFF            # 65504.0
NEG_ONE: int = 0xBC00               # -1.0
NEG_TWO: int = 0xC000               # -2.0
MIN_FINITE_NEG: int = 0xFBFF        # -65504.0
POS_INF: int = 0x7C00               # +Infinity
NEG_INF: int = 0xFC00               # -Infinity
POS_QNAN: int = 0x7E00              # Quiet NaN
NEG_QNAN: int = 0xFE00
POS_SNAN: int = 0x7C01              # Signaling NaN
NEG_SNAN: int = 0xFC01


def f16_to_bits(val: float) -> int:
    """Pack Python float into IEEE 754 binary16 unsigned 16-bit integer."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f16(bits: int) -> float:
    """Unpack unsigned 16-bit integer into Python float (IEEE 754 binary16)."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def split_f16_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack 16-bit word into (sign, exponent, fraction) components."""
    sign = (bits & SIGN16) >> 15
    exponent = (bits & EXP_MASK16) >> 10
    fraction = bits & FRAC_MASK16
    return sign, exponent, fraction


def is_nan16(bits: int) -> bool:
    """Return True if 16-bit word represents an IEEE 754 NaN."""
    return ((bits & EXP_MASK16) == EXP_MASK16) and ((bits & FRAC_MASK16) != 0)


def is_qnan16(bits: int) -> bool:
    """Return True if 16-bit word represents an IEEE 754 Quiet NaN."""
    return is_nan16(bits) and bool(bits & QNAN_BIT16)


def is_snan16(bits: int) -> bool:
    """Return True if 16-bit word represents an IEEE 754 Signaling NaN."""
    return is_nan16(bits) and not bool(bits & QNAN_BIT16)


def is_inf16(bits: int) -> bool:
    """Return True if 16-bit word represents positive or negative infinity."""
    return ((bits & EXP_MASK16) == EXP_MASK16) and ((bits & FRAC_MASK16) == 0)


def is_subnormal16(bits: int) -> bool:
    """Return True if 16-bit word represents a subnormal (denormal) half float."""
    return ((bits & EXP_MASK16) == 0) and ((bits & FRAC_MASK16) != 0)


def is_zero16(bits: int) -> bool:
    """Return True if 16-bit word represents +0.0 or -0.0."""
    return (bits & ~SIGN16) == 0


def extract_half(val32: int, high_half: bool) -> int:
    """Extract low 16 bits (high_half=False) or high 16 bits (high_half=True)."""
    return ((val32 >> 16) & MASK16) if high_half else (val32 & MASK16)


def apply_f16_modifiers(bits16: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply RDNA2 source modifiers (|src|, -src) to a 16-bit half float word.
    
    In RDNA2 ISA, abs clears sign bit (bit 15) first, then neg inverts sign bit (bit 15).
    """
    res = bits16 & MASK16
    if abs_mod:
        res &= ABS16
    if neg_mod:
        res ^= SIGN16
    return res


def pack_b32_f16(lo_half: int, hi_half: int) -> int:
    """Pack two 16-bit halves into an unsigned 32-bit dword."""
    return ((hi_half & MASK16) << 16) | (lo_half & MASK16)


def oracle_v_pack_b32_f16(
    src0_bits: int,
    src1_bits: int,
    op_sel: Tuple[int, int] = (0, 0),
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
) -> int:
    """Reference architectural oracle for RDNA2 v_pack_b32_f16 instruction.
    
    Packs half_word(src0, op_sel[0]) into dst[15:0] and
          half_word(src1, op_sel[1]) into dst[31:16],
    applying respective source modifiers.
    """
    h0 = extract_half(src0_bits, bool(op_sel[0]))
    h1 = extract_half(src1_bits, bool(op_sel[1]))

    h0_mod = apply_f16_modifiers(h0, abs_mod=abs0, neg_mod=neg0)
    h1_mod = apply_f16_modifiers(h1, abs_mod=abs1, neg_mod=neg1)

    return pack_b32_f16(h0_mod, h1_mod)


class TestPackB32F16ModelAndOracle(unittest.TestCase):
    """Architectural model and mathematical reference oracle tests for v_pack_b32_f16."""

    def test_basic_packing_default_op_sel(self) -> None:
        # Pack 1.0 (0x3C00) in low half and 2.0 (0x4000) in high half
        s0 = ONE
        s1 = TWO
        res = oracle_v_pack_b32_f16(s0, s1, op_sel=(0, 0))
        expected = (TWO << 16) | ONE
        self.assertEqual(res, expected)
        self.assertEqual(extract_half(res, False), ONE)
        self.assertEqual(extract_half(res, True), TWO)

    def test_op_sel_permutations(self) -> None:
        # s0 has 1.0 in low half and 0.5 in high half
        # s1 has 2.0 in low half and 4.0 in high half
        s0 = (HALF << 16) | ONE
        s1 = (FOUR << 16) | TWO

        # op_sel: [0, 0] -> s0[15:0], s1[15:0] => 1.0, 2.0
        self.assertEqual(oracle_v_pack_b32_f16(s0, s1, (0, 0)), (TWO << 16) | ONE)

        # op_sel: [1, 0] -> s0[31:16], s1[15:0] => 0.5, 2.0
        self.assertEqual(oracle_v_pack_b32_f16(s0, s1, (1, 0)), (TWO << 16) | HALF)

        # op_sel: [0, 1] -> s0[15:0], s1[31:16] => 1.0, 4.0
        self.assertEqual(oracle_v_pack_b32_f16(s0, s1, (0, 1)), (FOUR << 16) | ONE)

        # op_sel: [1, 1] -> s0[31:16], s1[31:16] => 0.5, 4.0
        self.assertEqual(oracle_v_pack_b32_f16(s0, s1, (1, 1)), (FOUR << 16) | HALF)

    def test_source_modifiers_abs_and_neg(self) -> None:
        # abs modifier clears sign bit
        self.assertEqual(apply_f16_modifiers(NEG_ONE, abs_mod=True), ONE)
        self.assertEqual(apply_f16_modifiers(ONE, abs_mod=True), ONE)

        # neg modifier inverts sign bit
        self.assertEqual(apply_f16_modifiers(ONE, neg_mod=True), NEG_ONE)
        self.assertEqual(apply_f16_modifiers(NEG_ONE, neg_mod=True), ONE)

        # abs then neg (-|x|) sets sign bit regardless of input sign
        self.assertEqual(apply_f16_modifiers(ONE, abs_mod=True, neg_mod=True), NEG_ONE)
        self.assertEqual(apply_f16_modifiers(NEG_ONE, abs_mod=True, neg_mod=True), NEG_ONE)

        # Apply modifiers via oracle
        res = oracle_v_pack_b32_f16(NEG_ONE, TWO, abs0=True, neg1=True)
        self.assertEqual(res, (NEG_TWO << 16) | ONE)

    def test_signed_zeros_preservation(self) -> None:
        # Pack +0.0 and -0.0
        res = oracle_v_pack_b32_f16(POS_ZERO, NEG_ZERO)
        self.assertEqual(res, (NEG_ZERO << 16) | POS_ZERO)
        self.assertEqual(extract_half(res, False), POS_ZERO)
        self.assertEqual(extract_half(res, True), NEG_ZERO)

        # Negate +0.0 -> -0.0
        res_neg = oracle_v_pack_b32_f16(POS_ZERO, NEG_ZERO, neg0=True, abs1=True)
        self.assertEqual(res_neg, (POS_ZERO << 16) | NEG_ZERO)

    def test_infinities_preservation(self) -> None:
        res = oracle_v_pack_b32_f16(POS_INF, NEG_INF)
        self.assertEqual(res, (NEG_INF << 16) | POS_INF)
        self.assertTrue(is_inf16(extract_half(res, False)))
        self.assertTrue(is_inf16(extract_half(res, True)))

    def test_nan_preservation_and_payloads(self) -> None:
        qnan_payload = EXP_MASK16 | QNAN_BIT16 | 0x0055
        snan_payload = EXP_MASK16 | 0x00AA
        res = oracle_v_pack_b32_f16(qnan_payload, snan_payload)
        self.assertEqual(extract_half(res, False), qnan_payload)
        self.assertEqual(extract_half(res, True), snan_payload)
        self.assertTrue(is_qnan16(extract_half(res, False)))
        self.assertTrue(is_snan16(extract_half(res, True)))

    def test_subnormals_preservation(self) -> None:
        res = oracle_v_pack_b32_f16(MIN_SUBNORMAL, MAX_SUBNORMAL)
        self.assertEqual(res, (MAX_SUBNORMAL << 16) | MIN_SUBNORMAL)
        self.assertTrue(is_subnormal16(extract_half(res, False)))
        self.assertTrue(is_subnormal16(extract_half(res, True)))


if __name__ == "__main__":
    unittest.main()
