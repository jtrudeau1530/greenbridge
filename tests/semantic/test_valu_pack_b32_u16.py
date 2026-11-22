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
    """Represents a 32-bit literal integer constant."""
    bits: int

    @property
    def name(self) -> str:
        return f"0x{self.bits & MASK32:08X}"


@dataclass(frozen=True)
class VOP3Operand:
    """Typed expression operand (Register or Literal32)."""
    source: Union[Reg, Literal32]

    def disassemble(self) -> str:
        return self.source.name


class Opcode(Enum):
    V_PACK_B32_U16 = "v_pack_b32_u16"


@dataclass(frozen=True)
class VPackB32U16Instruction:
    """Symbolic AST node representing a RDNA2 v_pack_b32_u16 instruction."""
    dst: Reg
    src0: VOP3Operand
    src1: VOP3Operand
    op_sel: Tuple[int, int] = (0, 0)
    clamp: bool = False
    signed: bool = False
    opcode: Opcode = Opcode.V_PACK_B32_U16

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")
        if self.op_sel[0] not in (0, 1) or self.op_sel[1] not in (0, 1):
            raise ValueError(f"op_sel values must be 0 or 1, got {self.op_sel}")

    def disassemble(self) -> str:
        s = f"{self.opcode.value} {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}"
        if self.op_sel != (0, 0):
            s += f" op_sel:[{self.op_sel[0]},{self.op_sel[1]}]"
        if self.clamp:
            s += " clamp"
        return s


class TestValuPackB32U16SymbolicAST(unittest.TestCase):
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
        lit = Literal32(0x12345678)
        self.assertEqual(lit.name, "0x12345678")

    def test_instruction_disassembly(self) -> None:
        dst = Reg(0, RegType.VGPR)
        src0 = VOP3Operand(Reg(1, RegType.VGPR))
        src1 = VOP3Operand(Reg(2, RegType.SGPR))

        inst_default = VPackB32U16Instruction(dst=dst, src0=src0, src1=src1)
        self.assertEqual(inst_default.disassemble(), "v_pack_b32_u16 v0, v1, s2")

        inst_opsel = VPackB32U16Instruction(dst=dst, src0=src0, src1=src1, op_sel=(1, 0))
        self.assertEqual(inst_opsel.disassemble(), "v_pack_b32_u16 v0, v1, s2 op_sel:[1,0]")

        inst_clamp = VPackB32U16Instruction(dst=dst, src0=src0, src1=src1, clamp=True)
        self.assertEqual(inst_clamp.disassemble(), "v_pack_b32_u16 v0, v1, s2 clamp")

        inst_both = VPackB32U16Instruction(dst=dst, src0=src0, src1=src1, op_sel=(1, 1), clamp=True)
        self.assertEqual(inst_both.disassemble(), "v_pack_b32_u16 v0, v1, s2 op_sel:[1,1] clamp")

    def test_instruction_validation(self) -> None:
        dst_sgpr = Reg(0, RegType.SGPR)
        src = VOP3Operand(Reg(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VPackB32U16Instruction(dst=dst_sgpr, src0=src, src1=src)

        with self.assertRaises(ValueError):
            VPackB32U16Instruction(dst=Reg(0, RegType.VGPR), src0=src, src1=src, op_sel=(2, 0))
        with self.assertRaises(ValueError):
            VPackB32U16Instruction(dst=Reg(0, RegType.VGPR), src0=src, src1=src, op_sel=(0, -1))


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single wavefront lane."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index] & MASK32

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32


class ScalarVALUInterpreter:
    """Deterministic scalar instruction interpreter for v_pack_b32_u16."""

    def resolve_operand(self, op: VOP3Operand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.bits & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VPackB32U16Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)

        # Computation phase
        result = oracle_v_pack_b32_u16(
            src0_bits=s0_raw,
            src1_bits=s1_raw,
            op_sel=instr.op_sel,
            clamp=instr.clamp,
            signed=instr.signed,
        )

        # Write-back phase
        state.set_vgpr(instr.dst.index, result)
        return result


class TestValuPackB32U16ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_basic_step_vgprs(self) -> None:
        self.state.set_vgpr(1, 0x1111)
        self.state.set_vgpr(2, 0x2222)
        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x22221111)
        self.assertEqual(self.state.get_vgpr(0), 0x22221111)

    def test_step_sgpr_and_literal(self) -> None:
        self.state.set_sgpr(3, 0xAABBCCDD)
        instr = VPackB32U16Instruction(
            dst=Reg(4, RegType.VGPR),
            src0=VOP3Operand(Reg(3, RegType.SGPR)),
            src1=VOP3Operand(Literal32(0x55667788)),
            op_sel=(1, 0),
        )
        # s0 high = 0xAABB, literal low = 0x7788
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x7788AABB)
        self.assertEqual(self.state.get_vgpr(4), 0x7788AABB)

    def test_step_with_clamp(self) -> None:
        self.state.set_vgpr(1, 0x00020000)
        self.state.set_vgpr(2, 0x00000050)
        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            clamp=True,
        )
        res = self.interp.step(instr, self.state)
        # s0 saturated to 0xFFFF, s1 unclamped 0x0050
        self.assertEqual(res, 0x0050FFFF)
        self.assertEqual(self.state.get_vgpr(0), 0x0050FFFF)

    def test_step_op_sel_extraction(self) -> None:
        self.state.set_vgpr(1, 0x12345678)
        self.state.set_vgpr(2, 0x9ABCDEF0)
        instr = VPackB32U16Instruction(
            dst=Reg(3, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            op_sel=(0, 1),
        )
        # s0 low = 0x5678, s1 high = 0x9ABC
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x9ABC5678)
        self.assertEqual(self.state.get_vgpr(3), 0x9ABC5678)

    def test_deterministic_repeatability(self) -> None:
        self.state.set_vgpr(10, 0xDEADBEEF)
        self.state.set_vgpr(11, 0xCAFEBABE)
        instr = VPackB32U16Instruction(
            dst=Reg(12, RegType.VGPR),
            src0=VOP3Operand(Reg(10, RegType.VGPR)),
            src1=VOP3Operand(Reg(11, RegType.VGPR)),
            op_sel=(1, 0),
        )
        res0 = self.interp.step(instr, self.state)
        for _ in range(50):
            res_n = self.interp.step(instr, self.state)
            self.assertEqual(res0, res_n)

    def test_unaffected_registers_preserved(self) -> None:
        self.state.set_vgpr(1, 0x1111)
        self.state.set_vgpr(2, 0x2222)
        self.state.set_vgpr(3, 0x33334444)
        self.state.set_sgpr(5, 0x55556666)
        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        self.interp.step(instr, self.state)
        self.assertEqual(self.state.get_vgpr(3), 0x33334444)
        self.assertEqual(self.state.get_sgpr(5), 0x55556666)


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
            return op.source.bits & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VPackB32U16Instruction, state: SIMDState) -> List[Optional[int]]:
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
                res = oracle_v_pack_b32_u16(
                    src0_bits=s0_vals[lane],
                    src1_bits=s1_vals[lane],
                    op_sel=instr.op_sel,
                    clamp=instr.clamp,
                    signed=instr.signed,
                )
                state.set_vgpr(lane, instr.dst.index, res)
                results[lane] = res

        return results


class TestValuPackB32U16SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    CANARY_OLD: int = 0xDEADBEEF

    def setUp(self) -> None:
        self.simd_interp = SIMDInterpreter()

    def test_wave32_full_exec(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr(lane, 1, lane)
            state.set_vgpr(lane, 2, lane * 3)

        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            op_sel=(0, 0),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(32):
            expected = ((lane * 3 & MASK16) << 16) | (lane & MASK16)
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr(lane, 1, (lane << 16) | 0x1000)
            state.set_vgpr(lane, 2, (lane << 16) | 0x2000)

        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            op_sel=(1, 1),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            expected = ((lane & MASK16) << 16) | (lane & MASK16)
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_divergence_exec_masking(self) -> None:
        mask = 0xAAAA_AAAA_5555_5555
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0x1111)
            state.set_vgpr(lane, 2, 0x2222)

        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            is_active = bool((mask >> lane) & 1)
            if is_active:
                expected = 0x22221111
                self.assertEqual(results[lane], expected)
                self.assertEqual(state.get_vgpr(lane, 0), expected)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_split_halves_masking(self) -> None:
        mask = 0x0000_0000_FFFF_FFFF
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0x3333)
            state.set_vgpr(lane, 2, 0x4444)

        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), 0x44443333)
        for lane in range(32, 64):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_single_lane_active(self) -> None:
        target_lane = 17
        mask = 1 << target_lane
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0x7777)
            state.set_vgpr(lane, 2, 0x8888)

        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            if lane == target_lane:
                self.assertEqual(results[lane], 0x88887777)
                self.assertEqual(state.get_vgpr(lane, 0), 0x88887777)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_all_lanes_disabled(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0x1234)
            state.set_vgpr(lane, 2, 0x5678)

        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            self.assertIsNone(results[lane])
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_shared_sgpr_across_divergent_lanes(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x0000_FFFF)
        state.set_sgpr(2, 0x50006000)
        for lane in range(32):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, lane)

        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.SGPR)),
            op_sel=(0, 1),
        )
        # src0 extracts low half (lane), src1 extracts high half of SGPR2 (0x5000)
        self.simd_interp.step(instr, state)
        for lane in range(16):
            expected = (0x5000 << 16) | (lane & MASK16)
            self.assertEqual(state.get_vgpr(lane, 0), expected)
        for lane in range(16, 32):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestValuPackB32U16RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing hazards and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_dst_aliases_src0(self) -> None:
        # v0 = v_pack_b32_u16 v0, v1
        self.state.set_vgpr(0, 0xAAAA1111)
        self.state.set_vgpr(1, 0xBBBB2222)
        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            op_sel=(0, 0),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x22221111)
        self.assertEqual(self.state.get_vgpr(0), 0x22221111)
        # v1 remains unchanged
        self.assertEqual(self.state.get_vgpr(1), 0xBBBB2222)

    def test_dst_aliases_src1(self) -> None:
        # v1 = v_pack_b32_u16 v0, v1
        self.state.set_vgpr(0, 0xAAAA1111)
        self.state.set_vgpr(1, 0xBBBB2222)
        instr = VPackB32U16Instruction(
            dst=Reg(1, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            op_sel=(0, 0),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x22221111)
        self.assertEqual(self.state.get_vgpr(1), 0x22221111)
        # v0 remains unchanged
        self.assertEqual(self.state.get_vgpr(0), 0xAAAA1111)

    def test_dst_aliases_both_src0_and_src1_swap_halves(self) -> None:
        # In-place half-word swap: v0 = v_pack_b32_u16 v0, v0 op_sel:[1,0]
        # Low half becomes old high half (0x1234), High half becomes old low half (0x5678)
        self.state.set_vgpr(0, 0x12345678)
        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
            op_sel=(1, 0),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x56781234)
        self.assertEqual(self.state.get_vgpr(0), 0x56781234)

    def test_inplace_self_duplicate_low(self) -> None:
        # v0 = v_pack_b32_u16 v0, v0 op_sel:[0,0]
        self.state.set_vgpr(0, 0x12345678)
        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
            op_sel=(0, 0),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x56785678)
        self.assertEqual(self.state.get_vgpr(0), 0x56785678)

    def test_inplace_self_duplicate_high(self) -> None:
        # v0 = v_pack_b32_u16 v0, v0 op_sel:[1,1]
        self.state.set_vgpr(0, 0x12345678)
        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
            op_sel=(1, 1),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x12341234)
        self.assertEqual(self.state.get_vgpr(0), 0x12341234)

    def test_aliased_clamping_inplace(self) -> None:
        # v0 = v_pack_b32_u16 v0, v1 clamp
        self.state.set_vgpr(0, 0x00030000)  # > 0xFFFF
        self.state.set_vgpr(1, 0x00000042)
        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            clamp=True,
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x0042FFFF)
        self.assertEqual(self.state.get_vgpr(0), 0x0042FFFF)

    def test_simd_multi_lane_aliasing(self) -> None:
        simd = SIMDInterpreter()
        state = SIMDState(num_lanes=64)
        for lane in range(64):
            state.set_vgpr(lane, 0, (lane << 16) | 0x00AA)
            state.set_vgpr(lane, 1, 0xBB00 | lane)

        instr = VPackB32U16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            op_sel=(1, 0),
        )
        # v0 receives low half = v0.hi (lane), high half = v1.lo (0xBB00 | lane)
        simd.step(instr, state)
        for lane in range(64):
            expected = (((0xBB00 | lane) & MASK16) << 16) | (lane & MASK16)
            self.assertEqual(state.get_vgpr(lane, 0), expected)
            self.assertEqual(state.get_vgpr(lane, 1), 0xBB00 | lane)






if __name__ == "__main__":
    unittest.main()
