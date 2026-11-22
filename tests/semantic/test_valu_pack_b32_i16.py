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



# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single wavefront lane."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def get_vgpr(self, index: int) -> int:
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        return self.vgprs[index] & MASK32

    def set_vgpr(self, index: int, val: int) -> None:
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        self.vgprs[index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        self.sgprs[index] = val & MASK32


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_pack_b32_i16."""

    def resolve_operand(self, op: VOP3Operand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.value & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VPackB32I16Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)

        # Computation phase
        result = oracle_v_pack_b32_i16(
            src0_bits=s0_raw,
            src1_bits=s1_raw,
            op_sel=instr.op_sel,
            abs0=instr.src0.abs_mod,
            neg0=instr.src0.neg_mod,
            abs1=instr.src1.abs_mod,
            neg1=instr.src1.neg_mod,
            clamp=instr.clamp,
        )

        # Write-back phase
        state.set_vgpr(instr.dst.index, result)
        return result


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_basic_step_vgprs(self) -> None:
        self.state.set_vgpr(1, 100)
        self.state.set_vgpr(2, (-200) & MASK16)
        instr = VPackB32I16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        expected = ((((-200) & MASK16) << 16) | 100) & MASK32
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_vgpr(0), expected)

    def test_step_with_sgpr_and_literal(self) -> None:
        # SGPR 4 has [high: 500, low: -300]
        self.state.set_sgpr(4, (500 << 16) | ((-300) & MASK16))
        # Literal operand is -1234
        instr = VPackB32I16Instruction(
            dst=Reg(5, RegType.VGPR),
            src0=VOP3Operand(Reg(4, RegType.SGPR)),
            src1=VOP3Operand(Literal32.from_i16(-1234)),
            op_sel=(1, 0),
        )
        # src0 op_sel 1 extracts high half 500; src1 op_sel 0 extracts low half -1234
        res = self.interp.step(instr, self.state)
        expected = ((((-1234) & MASK16) << 16) | 500) & MASK32
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_vgpr(5), expected)

    def test_step_with_modifiers(self) -> None:
        self.state.set_vgpr(1, (-75) & MASK16)
        self.state.set_vgpr(2, 120)
        instr = VPackB32I16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR), abs_mod=True),
            src1=VOP3Operand(Reg(2, RegType.VGPR), neg_mod=True),
        )
        res = self.interp.step(instr, self.state)
        expected = ((((-120) & MASK16) << 16) | 75) & MASK32
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_vgpr(0), expected)

    def test_step_with_clamp(self) -> None:
        # abs(-32768) with clamp saturates to 32767
        self.state.set_vgpr(1, 0x8000)
        self.state.set_vgpr(2, 0x8000)
        instr = VPackB32I16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR), abs_mod=True),
            src1=VOP3Operand(Reg(2, RegType.VGPR), abs_mod=True),
            clamp=True,
        )
        res = self.interp.step(instr, self.state)
        expected = (0x7FFF << 16) | 0x7FFF
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_vgpr(0), expected)

    def test_deterministic_repeatability(self) -> None:
        self.state.set_vgpr(10, 0x12345678)
        self.state.set_vgpr(11, 0x9ABCDEF0)
        instr = VPackB32I16Instruction(
            dst=Reg(12, RegType.VGPR),
            src0=VOP3Operand(Reg(10, RegType.VGPR)),
            src1=VOP3Operand(Reg(11, RegType.VGPR)),
            op_sel=(1, 0),
        )
        res0 = self.interp.step(instr, self.state)
        for _ in range(50):
            res_n = self.interp.step(instr, self.state)
            self.assertEqual(res0, res_n)



# ==============================================================================
# SIMD Multi-Lane Execution Engine and EXEC Masking
# ==============================================================================

@dataclass
class SIMDState:
    """Multi-lane SIMD machine state representing Wave32 or Wave64."""
    num_lanes: int
    exec_mask: Optional[int] = None
    vgprs: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {self.num_lanes}")
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        if self.exec_mask is None:
            self.exec_mask = (1 << self.num_lanes) - 1

    def get_vgpr(self, lane: int, index: int) -> int:
        return self.vgprs[lane][index] & MASK32

    def set_vgpr(self, lane: int, index: int, val: int) -> None:
        self.vgprs[lane][index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VOP3Operand, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(lane, op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.value & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VPackB32I16Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * state.num_lanes

        # Atomic read phase across all lanes
        s0_vals = [
            self.resolve_lane_operand(instr.src0, state, lane)
            for lane in range(state.num_lanes)
        ]
        s1_vals = [
            self.resolve_lane_operand(instr.src1, state, lane)
            for lane in range(state.num_lanes)
        ]

        # Execution phase
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = oracle_v_pack_b32_i16(
                    src0_bits=s0_vals[lane],
                    src1_bits=s1_vals[lane],
                    op_sel=instr.op_sel,
                    abs0=instr.src0.abs_mod,
                    neg0=instr.src0.neg_mod,
                    abs1=instr.src1.abs_mod,
                    neg1=instr.src1.neg_mod,
                    clamp=instr.clamp,
                )
                state.set_vgpr(lane, instr.dst.index, res)
                results[lane] = res

        return results


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    CANARY_OLD: int = 0xDEADBEEF

    def setUp(self) -> None:
        self.simd_interp = SIMDInterpreter()

    def test_wave32_full_exec(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr(lane, 1, lane + 1)
            state.set_vgpr(lane, 2, (-(lane + 1)) & MASK16)

        instr = VPackB32I16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            op_sel=(0, 0),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(32):
            expected = (((-(lane + 1)) & MASK16) << 16) | (lane + 1)
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr(lane, 1, lane * 10)
            state.set_vgpr(lane, 2, lane * 20)

        instr = VPackB32I16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            expected = (((lane * 20) & MASK16) << 16) | ((lane * 10) & MASK16)
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_divergence_exec_masking(self) -> None:
        mask = 0xAAAA_AAAA_5555_5555
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 100)
            state.set_vgpr(lane, 2, 200)

        instr = VPackB32I16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            is_active = bool((mask >> lane) & 1)
            if is_active:
                expected = (200 << 16) | 100
                self.assertEqual(results[lane], expected)
                self.assertEqual(state.get_vgpr(lane, 0), expected)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_split_halves_masking(self) -> None:
        # Lower 32 lanes active, upper 32 inactive
        mask = 0x0000_0000_FFFF_FFFF
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 300)
            state.set_vgpr(lane, 2, 400)

        instr = VPackB32I16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), (400 << 16) | 300)
        for lane in range(32, 64):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_single_active_lane(self) -> None:
        mask = 1 << 42
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 55)
            state.set_vgpr(lane, 2, 66)

        instr = VPackB32I16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(64):
            if lane == 42:
                self.assertEqual(state.get_vgpr(lane, 0), (66 << 16) | 55)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_zero_exec_mask_preserves_all(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)

        instr = VPackB32I16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Literal32(111)),
            src1=VOP3Operand(Literal32(222)),
        )
        results = self.simd_interp.step(instr, state)
        self.assertEqual(results, [None] * 64)
        for lane in range(64):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_uniform_scalar_broadcast(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        state.set_sgpr(8, (888 << 16) | 777)
        for lane in range(32):
            state.set_vgpr(lane, 1, lane + 10)

        # src0 is SGPR 8 (uniform broadcast), src1 is per-lane VGPR 1
        instr = VPackB32I16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(8, RegType.SGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            op_sel=(0, 0),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), ((lane + 10) << 16) | 777)



# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src0(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, (400 << 16) | 100)  # v0 = [400, 100]
        state.set_vgpr(1, (200 << 16) | 300)  # v1 = [200, 300]
        # v0 = pack(v0[15:0], v1[15:0]) -> low half 100, high half 300
        instr = VPackB32I16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            op_sel=(0, 0),
        )
        res = self.scalar_interp.step(instr, state)
        expected = (300 << 16) | 100
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(0), expected)
        # v1 untouched
        self.assertEqual(state.get_vgpr(1), (200 << 16) | 300)

    def test_dst_alias_src1(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, (200 << 16) | 300)  # v0 = [200, 300]
        state.set_vgpr(1, (400 << 16) | 100)  # v1 = [400, 100]
        # v0 = pack(v1[15:0], v0[31:16]) -> low half 100, high half 200
        instr = VPackB32I16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
            op_sel=(0, 1),
        )
        res = self.scalar_interp.step(instr, state)
        expected = (200 << 16) | 100
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(0), expected)
        # v1 untouched
        self.assertEqual(state.get_vgpr(1), (400 << 16) | 100)

    def test_dst_alias_both_srcs_in_place_swap(self) -> None:
        state = ScalarMachineState()
        original = (400 << 16) | 100  # high: 400, low: 100
        state.set_vgpr(0, original)
        # In-place swap halves of v0: low gets v0[31:16] (400), high gets v0[15:0] (100)
        instr = VPackB32I16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
            op_sel=(1, 0),
        )
        res = self.scalar_interp.step(instr, state)
        expected = (100 << 16) | 400
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(0), expected)

    def test_dst_alias_both_srcs_duplicate_half(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(3, (400 << 16) | 100)
        # Duplicate low half into both channels
        instr_lo = VPackB32I16Instruction(
            dst=Reg(3, RegType.VGPR),
            src0=VOP3Operand(Reg(3, RegType.VGPR)),
            src1=VOP3Operand(Reg(3, RegType.VGPR)),
            op_sel=(0, 0),
        )
        res_lo = self.scalar_interp.step(instr_lo, state)
        self.assertEqual(res_lo, (100 << 16) | 100)

        # Duplicate high half into both channels
        state.set_vgpr(3, (400 << 16) | 100)
        instr_hi = VPackB32I16Instruction(
            dst=Reg(3, RegType.VGPR),
            src0=VOP3Operand(Reg(3, RegType.VGPR)),
            src1=VOP3Operand(Reg(3, RegType.VGPR)),
            op_sel=(1, 1),
        )
        res_hi = self.scalar_interp.step(instr_hi, state)
        self.assertEqual(res_hi, (400 << 16) | 400)

    def test_simd_multi_lane_aliased_swap(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr(lane, 0, ((lane + 0x100) << 16) | (lane & 0xFFFF))

        # Each lane swaps its own v0 halves in place
        instr = VPackB32I16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
            op_sel=(1, 0),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            expected = ((lane & 0xFFFF) << 16) | ((lane + 0x100) & 0xFFFF)
            self.assertEqual(state.get_vgpr(lane, 0), expected)


if __name__ == "__main__":
    unittest.main()




