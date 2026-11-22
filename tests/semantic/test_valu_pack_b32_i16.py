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



# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Reg:
    """Represents an architectural 32-bit register (VGPR or SGPR)."""
    index: int
    reg_type: RegType

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index {self.index} out of range [0, 255]")

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class Literal32:
    """Represents a 32-bit integer literal or inline constant."""
    value: int

    def __post_init__(self) -> None:
        if not (-0x8000_0000 <= self.value <= 0xFFFF_FFFF):
            raise ValueError(f"Literal32 value {self.value} out of 32-bit range")

    @classmethod
    def from_i16(cls, val: int) -> Literal32:
        return cls(s16(val) & MASK32)

    @property
    def name(self) -> str:
        return f"0x{self.value & MASK32:08X}"


@dataclass(frozen=True)
class VOP3Operand:
    """Typed expression operand with optional input modifiers (abs, neg)."""
    source: Union[Reg, Literal32]
    abs_mod: bool = False
    neg_mod: bool = False

    def disassemble(self) -> str:
        s = self.source.name
        if self.abs_mod:
            s = f"|{s}|"
        if self.neg_mod:
            s = f"-{s}"
        return s


class Opcode(Enum):
    V_PACK_B32_I16 = "v_pack_b32_i16"


@dataclass(frozen=True)
class VPackB32I16Instruction:
    """Symbolic AST node representing a RDNA2 v_pack_b32_i16 instruction."""
    dst: Reg
    src0: VOP3Operand
    src1: VOP3Operand
    op_sel: Tuple[int, int] = (0, 0)
    clamp: bool = False
    opcode: Opcode = Opcode.V_PACK_B32_I16

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")
        if self.op_sel[0] not in (0, 1) or self.op_sel[1] not in (0, 1):
            raise ValueError(f"op_sel values must be 0 or 1, got {self.op_sel}")

    def disassemble(self) -> str:
        s = f"{self.opcode.value} {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}"
        modifiers = []
        if self.op_sel != (0, 0):
            modifiers.append(f"op_sel:[{self.op_sel[0]},{self.op_sel[1]},0]")
        if self.clamp:
            modifiers.append("clamp")
        if modifiers:
            s += " " + " ".join(modifiers)
        return s


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_creation_and_disassembly(self) -> None:
        v0 = Reg(0, RegType.VGPR)
        s5 = Reg(5, RegType.SGPR)
        self.assertEqual(v0.name, "v0")
        self.assertEqual(s5.name, "s5")

    def test_register_bounds_validation(self) -> None:
        with self.assertRaises(ValueError):
            Reg(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(256, RegType.VGPR)

    def test_literal_operand(self) -> None:
        lit = Literal32(0x1234)
        self.assertEqual(lit.name, "0x00001234")
        lit_i16 = Literal32.from_i16(-50)
        self.assertEqual(s16(lit_i16.value & MASK16), -50)

    def test_operand_modifiers_disassembly(self) -> None:
        v1 = Reg(1, RegType.VGPR)
        op_plain = VOP3Operand(v1)
        self.assertEqual(op_plain.disassemble(), "v1")

        op_abs = VOP3Operand(v1, abs_mod=True)
        self.assertEqual(op_abs.disassemble(), "|v1|")

        op_neg = VOP3Operand(v1, neg_mod=True)
        self.assertEqual(op_neg.disassemble(), "-v1")

        op_both = VOP3Operand(v1, abs_mod=True, neg_mod=True)
        self.assertEqual(op_both.disassemble(), "-|v1|")

    def test_instruction_disassembly(self) -> None:
        dst = Reg(0, RegType.VGPR)
        src0 = VOP3Operand(Reg(1, RegType.VGPR))
        src1 = VOP3Operand(Reg(2, RegType.VGPR), neg_mod=True)

        inst_default = VPackB32I16Instruction(dst=dst, src0=src0, src1=src1)
        self.assertEqual(inst_default.disassemble(), "v_pack_b32_i16 v0, v1, -v2")

        inst_opsel = VPackB32I16Instruction(dst=dst, src0=src0, src1=src1, op_sel=(1, 0), clamp=True)
        self.assertEqual(inst_opsel.disassemble(), "v_pack_b32_i16 v0, v1, -v2 op_sel:[1,0,0] clamp")

    def test_instruction_validation(self) -> None:
        dst_sgpr = Reg(0, RegType.SGPR)
        src = VOP3Operand(Reg(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VPackB32I16Instruction(dst=dst_sgpr, src0=src, src1=src)

        with self.assertRaises(ValueError):
            VPackB32I16Instruction(dst=Reg(0, RegType.VGPR), src0=src, src1=src, op_sel=(2, 0))


if __name__ == "__main__":
    unittest.main()

