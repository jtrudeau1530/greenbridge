from __future__ import annotations

import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

# ==============================================================================
# RDNA2 SALU Architectural Constants
# ==============================================================================

MASK_U32: int = 0xFFFF_FFFF
MODULO_U32: int = 0x1_0000_0000
SIGN_BIT_U32: int = 0x8000_0000
MAX_U32: int = 0xFFFF_FFFF
NUM_SGPRS: int = 256


# ==============================================================================
# Reference Mathematical Oracles for s_add_co_u32 & s_sub_co_u32
# ==============================================================================

def oracle_s_add_co_u32(src0: int, src1: int) -> Tuple[int, int]:
    """Reference mathematical oracle for RDNA2 s_add_co_u32 / s_add_u32.

    Computes 32-bit unsigned addition:
        D = (S0 + S1) mod 2^32
        SCC = 1 if (S0 + S1 >= 2^32) else 0 (Carry Out)
    """
    s0 = src0 & MASK_U32
    s1 = src1 & MASK_U32
    raw_sum = s0 + s1
    res = raw_sum & MASK_U32
    carry = 1 if raw_sum >= MODULO_U32 else 0
    return res, carry


def oracle_s_sub_co_u32(src0: int, src1: int) -> Tuple[int, int]:
    """Reference mathematical oracle for RDNA2 s_sub_co_u32 / s_sub_u32.

    Computes 32-bit unsigned subtraction:
        D = (S0 - S1) mod 2^32
        SCC = 1 if (S0 < S1) else 0 (Borrow Out)
    """
    s0 = src0 & MASK_U32
    s1 = src1 & MASK_U32
    raw_diff = s0 - s1
    res = raw_diff & MASK_U32
    borrow = 1 if s0 < s1 else 0
    return res, borrow


# ==============================================================================
# Architectural Hardware Model
# ==============================================================================

@dataclass
class ScalarModel:
    """Architectural model representing the RDNA2 Scalar ALU (SALU) register file and status."""
    sgprs: List[int] = field(default_factory=lambda: [0] * NUM_SGPRS)
    scc: int = 0

    def read_sgpr(self, index: int) -> int:
        if not (0 <= index < NUM_SGPRS):
            raise IndexError(f"SGPR index {index} out of bounds [0, {NUM_SGPRS - 1}]")
        return self.sgprs[index] & MASK_U32

    def write_sgpr(self, index: int, val: int) -> None:
        if not (0 <= index < NUM_SGPRS):
            raise IndexError(f"SGPR index {index} out of bounds [0, {NUM_SGPRS - 1}]")
        self.sgprs[index] = val & MASK_U32

    def set_scc(self, val: int) -> None:
        self.scc = 1 if val else 0


# ==============================================================================
# Phase 1 Unit Tests: Reference Oracles & Mathematical Properties
# ==============================================================================

class TestReferenceOracleAddCoU32(unittest.TestCase):
    """Exhaustive mathematical verification for oracle_s_add_co_u32."""

    def test_zero_identities(self) -> None:
        self.assertEqual(oracle_s_add_co_u32(0, 0), (0, 0))
        self.assertEqual(oracle_s_add_co_u32(42, 0), (42, 0))
        self.assertEqual(oracle_s_add_co_u32(0, 1337), (1337, 0))
        self.assertEqual(oracle_s_add_co_u32(MAX_U32, 0), (MAX_U32, 0))

    def test_commutativity(self) -> None:
        vectors = [
            (0, 1),
            (100, 200),
            (MAX_U32, 1),
            (0x8000_0000, 0x8000_0000),
            (0x1234_5678, 0x9ABC_DEF0),
        ]
        for a, b in vectors:
            with self.subTest(a=a, b=b):
                res_ab, c_ab = oracle_s_add_co_u32(a, b)
                res_ba, c_ba = oracle_s_add_co_u32(b, a)
                self.assertEqual(res_ab, res_ba)
                self.assertEqual(c_ab, c_ba)

    def test_boundary_carry_transitions(self) -> None:
        # Just before carry
        self.assertEqual(oracle_s_add_co_u32(MAX_U32 - 1, 1), (MAX_U32, 0))
        self.assertEqual(oracle_s_add_co_u32(0x7FFF_FFFF, 0x8000_0000), (MAX_U32, 0))
        # Exact carry threshold
        self.assertEqual(oracle_s_add_co_u32(MAX_U32, 1), (0, 1))
        self.assertEqual(oracle_s_add_co_u32(1, MAX_U32), (0, 1))
        self.assertEqual(oracle_s_add_co_u32(MAX_U32, 2), (1, 1))
        # Both high operands
        self.assertEqual(oracle_s_add_co_u32(MAX_U32, MAX_U32), (MAX_U32 - 1, 1))
        self.assertEqual(oracle_s_add_co_u32(0x8000_0000, 0x8000_0000), (0, 1))
        self.assertEqual(oracle_s_add_co_u32(0x8000_0000, 0x8000_0001), (1, 1))


class TestReferenceOracleSubCoU32(unittest.TestCase):
    """Exhaustive mathematical verification for oracle_s_sub_co_u32."""

    def test_zero_identities(self) -> None:
        self.assertEqual(oracle_s_sub_co_u32(0, 0), (0, 0))
        self.assertEqual(oracle_s_sub_co_u32(42, 0), (42, 0))
        self.assertEqual(oracle_s_sub_co_u32(MAX_U32, 0), (MAX_U32, 0))

    def test_self_subtraction(self) -> None:
        values = [0, 1, 42, 0x7FFF_FFFF, 0x8000_0000, MAX_U32]
        for v in values:
            with self.subTest(v=v):
                res, borrow = oracle_s_sub_co_u32(v, v)
                self.assertEqual(res, 0)
                self.assertEqual(borrow, 0)

    def test_boundary_borrow_transitions(self) -> None:
        # Minimal borrow cases: 0 - 1
        self.assertEqual(oracle_s_sub_co_u32(0, 1), (MAX_U32, 1))
        self.assertEqual(oracle_s_sub_co_u32(0, MAX_U32), (1, 1))
        self.assertEqual(oracle_s_sub_co_u32(1, 2), (MAX_U32, 1))
        # Borrow from 0x8000_0000
        self.assertEqual(oracle_s_sub_co_u32(0x8000_0000, 1), (0x7FFF_FFFF, 0))
        self.assertEqual(oracle_s_sub_co_u32(0x7FFF_FFFF, 0x8000_0000), (MAX_U32, 1))
        # High boundary subtractions without borrow
        self.assertEqual(oracle_s_sub_co_u32(MAX_U32, 1), (MAX_U32 - 1, 0))
        self.assertEqual(oracle_s_sub_co_u32(MAX_U32, MAX_U32), (0, 0))

    def test_add_sub_inverse_relation(self) -> None:
        pairs = [
            (10, 5),
            (5, 10),
            (0, 100),
            (MAX_U32, 12345),
            (0x8000_0000, 0x1234_5678),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                diff, _ = oracle_s_sub_co_u32(a, b)
                recovered, _ = oracle_s_add_co_u32(diff, b)
                self.assertEqual(recovered, a & MASK_U32)


class TestArchitecturalModel(unittest.TestCase):
    """Verification of the ScalarModel register file and SCC status representation."""

    def test_sgpr_read_write(self) -> None:
        model = ScalarModel()
        self.assertEqual(model.read_sgpr(0), 0)
        model.write_sgpr(0, 0xDEAD_BEEF)
        self.assertEqual(model.read_sgpr(0), 0xDEAD_BEEF)
        # Verify 32-bit truncation
        model.write_sgpr(1, 0x1_1234_5678)
        self.assertEqual(model.read_sgpr(1), 0x1234_5678)

    def test_sgpr_bounds(self) -> None:
        model = ScalarModel()
        with self.assertRaises(IndexError):
            model.read_sgpr(-1)
        with self.assertRaises(IndexError):
            model.read_sgpr(NUM_SGPRS)
        with self.assertRaises(IndexError):
            model.write_sgpr(NUM_SGPRS + 10, 1)

    def test_scc_flag(self) -> None:
        model = ScalarModel()
        self.assertEqual(model.scc, 0)
        model.set_scc(1)
        self.assertEqual(model.scc, 1)
        model.set_scc(0)
        self.assertEqual(model.scc, 0)
        model.set_scc(42)
        self.assertEqual(model.scc, 1)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class SOPOpcode(Enum):
    """SOP2 scalar ALU opcodes."""
    S_ADD_CO_U32 = "s_add_co_u32"
    S_SUB_CO_U32 = "s_sub_co_u32"
    S_ADD_U32 = "s_add_u32"
    S_SUB_U32 = "s_sub_u32"


@dataclass(frozen=True)
class SGPROperand:
    """Represents a 32-bit Scalar General-Purpose Register operand."""
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index < NUM_SGPRS):
            raise ValueError(f"SGPR index {self.index} out of range [0, {NUM_SGPRS - 1}]")

    @property
    def name(self) -> str:
        return f"s{self.index}"


@dataclass(frozen=True)
class LiteralU32:
    """Represents an arbitrary 32-bit literal constant in the instruction stream."""
    value: int

    def __post_init__(self) -> None:
        if not (0 <= self.value <= MAX_U32):
            raise ValueError(f"Literal constant 0x{self.value:X} out of range [0, 0x{MAX_U32:X}]")

    @property
    def name(self) -> str:
        return f"0x{self.value:08X}"


@dataclass(frozen=True)
class InlineConstant:
    """Represents an inline integer constant (-16..64) encoded directly in operand field."""
    value: int

    def __post_init__(self) -> None:
        if not (-16 <= self.value <= 64):
            raise ValueError(f"Inline integer constant {self.value} out of range [-16, 64]")

    @property
    def name(self) -> str:
        return f"{self.value}"


SOP2Operand = Union[SGPROperand, LiteralU32, InlineConstant]


@dataclass(frozen=True)
class SAddSubCoU32Instruction:
    """Symbolic AST node representing a 32-bit SALU add/sub with carry instruction."""
    opcode: SOPOpcode
    dst: SGPROperand
    src0: SOP2Operand
    src1: SOP2Operand

    def __post_init__(self) -> None:
        if not isinstance(self.dst, SGPROperand):
            raise TypeError(f"Destination operand must be SGPROperand, got {type(self.dst).__name__}")
        if self.opcode not in (
            SOPOpcode.S_ADD_CO_U32,
            SOPOpcode.S_SUB_CO_U32,
            SOPOpcode.S_ADD_U32,
            SOPOpcode.S_SUB_U32,
        ):
            raise ValueError(f"Invalid opcode for SAddSubCoU32Instruction: {self.opcode}")

        # Hardware constraint: at most one 32-bit literal constant allowed in SOP2 encoding
        literal_count = sum(1 for src in (self.src0, self.src1) if isinstance(src, LiteralU32))
        if literal_count > 1:
            raise ValueError("SOP2 instructions cannot encode more than one 32-bit literal constant")

    def disassemble(self) -> str:
        return f"{self.opcode.value} {self.dst.name}, {self.src0.name}, {self.src1.name}"


class TestSymbolicAST(unittest.TestCase):
    """Verification of symbolic AST node typing, bounds, constraints, and disassembly."""

    def test_sgpr_operand_valid(self) -> None:
        op = SGPROperand(0)
        self.assertEqual(op.name, "s0")
        op_max = SGPROperand(NUM_SGPRS - 1)
        self.assertEqual(op_max.name, f"s{NUM_SGPRS - 1}")

    def test_sgpr_operand_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            SGPROperand(-1)
        with self.assertRaises(ValueError):
            SGPROperand(NUM_SGPRS)

    def test_literal_u32_valid(self) -> None:
        lit = LiteralU32(0x1234_5678)
        self.assertEqual(lit.name, "0x12345678")
        self.assertEqual(lit.value, 0x1234_5678)

    def test_literal_u32_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            LiteralU32(-1)
        with self.assertRaises(ValueError):
            LiteralU32(0x1_0000_0000)

    def test_inline_constant_valid(self) -> None:
        for val in (-16, 0, 1, 64):
            c = InlineConstant(val)
            self.assertEqual(c.name, str(val))
            self.assertEqual(c.value, val)

    def test_inline_constant_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            InlineConstant(-17)
        with self.assertRaises(ValueError):
            InlineConstant(65)

    def test_instruction_disassembly(self) -> None:
        instr_add = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_ADD_CO_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(1),
            src1=SGPROperand(2),
        )
        self.assertEqual(instr_add.disassemble(), "s_add_co_u32 s0, s1, s2")

        instr_sub = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_SUB_CO_U32,
            dst=SGPROperand(4),
            src0=SGPROperand(5),
            src1=LiteralU32(0xDEADBEEF),
        )
        self.assertEqual(instr_sub.disassemble(), "s_sub_co_u32 s4, s5, 0xDEADBEEF")

        instr_inline = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_ADD_U32,
            dst=SGPROperand(10),
            src0=SGPROperand(11),
            src1=InlineConstant(1),
        )
        self.assertEqual(instr_inline.disassemble(), "s_add_u32 s10, s11, 1")

    def test_instruction_invalid_dst(self) -> None:
        with self.assertRaises(TypeError):
            SAddSubCoU32Instruction(  # type: ignore[arg-type]
                opcode=SOPOpcode.S_ADD_CO_U32,
                dst=LiteralU32(42),  # type: ignore[arg-type]
                src0=SGPROperand(0),
                src1=SGPROperand(1),
            )

    def test_instruction_multiple_literals_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SAddSubCoU32Instruction(
                opcode=SOPOpcode.S_ADD_CO_U32,
                dst=SGPROperand(0),
                src0=LiteralU32(0x100),
                src1=LiteralU32(0x200),
            )


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 SALU instructions."""

    def resolve_operand(self, op: SOP2Operand, state: ScalarModel) -> int:
        if isinstance(op, SGPROperand):
            return state.read_sgpr(op.index)
        elif isinstance(op, LiteralU32):
            return op.value & MASK_U32
        elif isinstance(op, InlineConstant):
            return op.value & MASK_U32
        else:
            raise TypeError(f"Unsupported operand type: {type(op).__name__}")

    def step(self, instr: SAddSubCoU32Instruction, state: ScalarModel) -> Tuple[int, int]:
        # Atomic read phase
        s0_val = self.resolve_operand(instr.src0, state)
        s1_val = self.resolve_operand(instr.src1, state)

        # Computation phase
        if instr.opcode in (SOPOpcode.S_ADD_CO_U32, SOPOpcode.S_ADD_U32):
            res, cc = oracle_s_add_co_u32(s0_val, s1_val)
        elif instr.opcode in (SOPOpcode.S_SUB_CO_U32, SOPOpcode.S_SUB_U32):
            res, cc = oracle_s_sub_co_u32(s0_val, s1_val)
        else:
            raise ValueError(f"Unsupported opcode: {instr.opcode}")

        # Update condition code and destination SGPR
        state.set_scc(cc)
        state.write_sgpr(instr.dst.index, res)
        return res, cc

    def run_program(
        self, program: List[SAddSubCoU32Instruction], state: ScalarModel
    ) -> List[Tuple[int, int]]:
        results: List[Tuple[int, int]] = []
        for instr in program:
            results.append(self.step(instr, state))
        return results


class TestScalarInterpreter(unittest.TestCase):
    """Verification of deterministic scalar instruction interpretation."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarModel()

    def test_scalar_add_execution(self) -> None:
        self.state.write_sgpr(1, 100)
        self.state.write_sgpr(2, 250)
        instr = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_ADD_CO_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(1),
            src1=SGPROperand(2),
        )
        res, cc = self.interp.step(instr, self.state)
        self.assertEqual(res, 350)
        self.assertEqual(cc, 0)
        self.assertEqual(self.state.read_sgpr(0), 350)
        self.assertEqual(self.state.scc, 0)

    def test_scalar_add_carry_generation(self) -> None:
        self.state.write_sgpr(1, MAX_U32)
        self.state.write_sgpr(2, 5)
        instr = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_ADD_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(1),
            src1=SGPROperand(2),
        )
        res, cc = self.interp.step(instr, self.state)
        self.assertEqual(res, 4)
        self.assertEqual(cc, 1)
        self.assertEqual(self.state.read_sgpr(0), 4)
        self.assertEqual(self.state.scc, 1)

    def test_scalar_sub_execution(self) -> None:
        self.state.write_sgpr(1, 500)
        self.state.write_sgpr(2, 200)
        instr = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_SUB_CO_U32,
            dst=SGPROperand(3),
            src0=SGPROperand(1),
            src1=SGPROperand(2),
        )
        res, cc = self.interp.step(instr, self.state)
        self.assertEqual(res, 300)
        self.assertEqual(cc, 0)
        self.assertEqual(self.state.read_sgpr(3), 300)
        self.assertEqual(self.state.scc, 0)

    def test_scalar_sub_borrow_generation(self) -> None:
        self.state.write_sgpr(1, 10)
        self.state.write_sgpr(2, 20)
        instr = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_SUB_U32,
            dst=SGPROperand(3),
            src0=SGPROperand(1),
            src1=SGPROperand(2),
        )
        res, cc = self.interp.step(instr, self.state)
        self.assertEqual(res, MAX_U32 - 9)
        self.assertEqual(cc, 1)
        self.assertEqual(self.state.read_sgpr(3), MAX_U32 - 9)
        self.assertEqual(self.state.scc, 1)

    def test_scalar_with_literal_and_inline_constants(self) -> None:
        self.state.write_sgpr(1, 0x1000)
        # Add literal
        instr_lit = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_ADD_CO_U32,
            dst=SGPROperand(2),
            src0=SGPROperand(1),
            src1=LiteralU32(0x2000),
        )
        res_lit, cc_lit = self.interp.step(instr_lit, self.state)
        self.assertEqual(res_lit, 0x3000)
        self.assertEqual(cc_lit, 0)

        # Sub inline constant
        instr_inline = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_SUB_CO_U32,
            dst=SGPROperand(3),
            src0=SGPROperand(2),
            src1=InlineConstant(1),
        )
        res_in, cc_in = self.interp.step(instr_inline, self.state)
        self.assertEqual(res_in, 0x2FFF)
        self.assertEqual(cc_in, 0)

    def test_scalar_program_chain(self) -> None:
        # Program: s0 = 0; s0 += 10; s0 += 20; s0 -= 5
        program = [
            SAddSubCoU32Instruction(
                opcode=SOPOpcode.S_ADD_CO_U32,
                dst=SGPROperand(0),
                src0=SGPROperand(0),
                src1=InlineConstant(10),
            ),
            SAddSubCoU32Instruction(
                opcode=SOPOpcode.S_ADD_CO_U32,
                dst=SGPROperand(0),
                src0=SGPROperand(0),
                src1=InlineConstant(20),
            ),
            SAddSubCoU32Instruction(
                opcode=SOPOpcode.S_SUB_CO_U32,
                dst=SGPROperand(0),
                src0=SGPROperand(0),
                src1=InlineConstant(5),
            ),
        ]
        results = self.interp.run_program(program, self.state)
        self.assertEqual(results, [(10, 0), (30, 0), (25, 0)])
        self.assertEqual(self.state.read_sgpr(0), 25)


# ==============================================================================
# SIMD Multi-Lane Execution Engine and 32/64-Bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDMultiLaneState:
    """Architectural state for a wavefront (Wave32 or Wave64) with shared SALU."""
    wave_size: int = 32
    exec_mask: int = field(init=False)
    scalar_model: ScalarModel = field(default_factory=ScalarModel)
    vgprs: List[List[int]] = field(init=False)

    def __post_init__(self) -> None:
        if self.wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {self.wave_size}")
        self.exec_mask = (1 << self.wave_size) - 1
        self.vgprs = [[0] * 256 for _ in range(self.wave_size)]

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane {lane} out of range [0, {self.wave_size - 1}]")
        return bool(self.exec_mask & (1 << lane))

    def read_lane_vgpr(self, lane: int, reg_idx: int) -> int:
        return self.vgprs[lane][reg_idx] & MASK_U32

    def write_lane_vgpr(self, lane: int, reg_idx: int, val: int) -> None:
        self.vgprs[lane][reg_idx] = val & MASK_U32


class SIMDExecutionEngine:
    """Execution engine testing interaction between shared SALU and divergent SIMD lanes."""

    def __init__(self) -> None:
        self.scalar_interp = ScalarInterpreter()

    def step_salu(
        self, instr: SAddSubCoU32Instruction, simd_state: SIMDMultiLaneState
    ) -> Tuple[int, int]:
        """SALU execution is uniform and unconditional across the wavefront."""
        return self.scalar_interp.step(instr, simd_state.scalar_model)

    def vector_add_scalar_broadcast(
        self, dst_vgpr: int, src_vgpr: int, src_sgpr: int, simd_state: SIMDMultiLaneState
    ) -> List[Optional[int]]:
        """Simulate VALU lane operation consuming scalar SGPR result under EXEC mask."""
        scalar_val = simd_state.scalar_model.read_sgpr(src_sgpr)
        results: List[Optional[int]] = [None] * simd_state.wave_size
        for lane in range(simd_state.wave_size):
            if simd_state.is_lane_active(lane):
                lane_val = simd_state.read_lane_vgpr(lane, src_vgpr)
                out_val, _ = oracle_s_add_co_u32(lane_val, scalar_val)
                simd_state.write_lane_vgpr(lane, dst_vgpr, out_val)
                results[lane] = out_val
        return results


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Verification of SIMD multi-lane divergence and 32/64-bit EXEC masking interaction."""

    def setUp(self) -> None:
        self.engine = SIMDExecutionEngine()

    def test_salu_unconditional_execution_with_zero_exec_mask(self) -> None:
        # SALU must execute even when EXEC is 0 (all lanes masked off)
        for wave_size in (32, 64):
            with self.subTest(wave_size=wave_size):
                state = SIMDMultiLaneState(wave_size=wave_size)
                state.exec_mask = 0  # Inactive wavefront
                state.scalar_model.write_sgpr(1, 100)
                state.scalar_model.write_sgpr(2, 200)

                instr = SAddSubCoU32Instruction(
                    opcode=SOPOpcode.S_ADD_CO_U32,
                    dst=SGPROperand(0),
                    src0=SGPROperand(1),
                    src1=SGPROperand(2),
                )
                res, cc = self.engine.step_salu(instr, state)
                self.assertEqual(res, 300)
                self.assertEqual(cc, 0)
                # Ensure SGPR and SCC were updated unconditionally
                self.assertEqual(state.scalar_model.read_sgpr(0), 300)
                self.assertEqual(state.scalar_model.scc, 0)

    def test_salu_broadcast_to_wave32(self) -> None:
        state = SIMDMultiLaneState(wave_size=32)
        # Set divergent EXEC: only even lanes active (0x55555555)
        state.exec_mask = 0x5555_5555
        # Set up lane VGPRs: v0[lane] = lane * 10
        for lane in range(32):
            state.write_lane_vgpr(lane, 0, lane * 10)
            state.write_lane_vgpr(lane, 1, 0xEEEE_EEEE)  # sentinel initial dst

        # Compute scalar offset via SALU: s0 = 1000 + 24
        state.scalar_model.write_sgpr(2, 1000)
        salu_instr = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_ADD_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(2),
            src1=InlineConstant(24),
        )
        self.engine.step_salu(salu_instr, state)
        self.assertEqual(state.scalar_model.read_sgpr(0), 1024)

        # Broadcast s0 to active lanes: v1[lane] = v0[lane] + s0
        results = self.engine.vector_add_scalar_broadcast(dst_vgpr=1, src_vgpr=0, src_sgpr=0, simd_state=state)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], (lane * 10) + 1024)
                self.assertEqual(state.read_lane_vgpr(lane, 1), (lane * 10) + 1024)
            else:
                self.assertIsNone(results[lane])
                # Inactive lane must remain untouched
                self.assertEqual(state.read_lane_vgpr(lane, 1), 0xEEEE_EEEE)

    def test_salu_broadcast_to_wave64(self) -> None:
        state = SIMDMultiLaneState(wave_size=64)
        # Lower 32 lanes active, upper 32 lanes masked out
        state.exec_mask = 0x0000_0000_FFFF_FFFF

        for lane in range(64):
            state.write_lane_vgpr(lane, 5, 500)
            state.write_lane_vgpr(lane, 6, 0x1111_1111)

        # SALU computes scalar reduction/subtraction: s4 = 100 - 30
        state.scalar_model.write_sgpr(10, 100)
        salu_sub = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_SUB_CO_U32,
            dst=SGPROperand(4),
            src0=SGPROperand(10),
            src1=InlineConstant(30),
        )
        self.engine.step_salu(salu_sub, state)
        self.assertEqual(state.scalar_model.read_sgpr(4), 70)
        self.assertEqual(state.scalar_model.scc, 0)

        # Vector lanes consume s4
        results = self.engine.vector_add_scalar_broadcast(dst_vgpr=6, src_vgpr=5, src_sgpr=4, simd_state=state)
        for lane in range(64):
            if lane < 32:
                self.assertEqual(results[lane], 570)
                self.assertEqual(state.read_lane_vgpr(lane, 6), 570)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.read_lane_vgpr(lane, 6), 0x1111_1111)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasingAndPreservation(unittest.TestCase):
    """Verification of atomic read-before-write semantics and register aliasing."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarModel()

    def test_in_place_accumulator_dst_equals_src0(self) -> None:
        # s0 = s0 + s1
        self.state.write_sgpr(0, 100)
        self.state.write_sgpr(1, 50)
        instr_add = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_ADD_CO_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(0),
            src1=SGPROperand(1),
        )
        res, cc = self.interp.step(instr_add, self.state)
        self.assertEqual(res, 150)
        self.assertEqual(cc, 0)
        self.assertEqual(self.state.read_sgpr(0), 150)
        self.assertEqual(self.state.read_sgpr(1), 50)  # src1 preserved

        # s0 = s0 - s1
        instr_sub = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_SUB_CO_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(0),
            src1=SGPROperand(1),
        )
        res_sub, cc_sub = self.interp.step(instr_sub, self.state)
        self.assertEqual(res_sub, 100)
        self.assertEqual(cc_sub, 0)
        self.assertEqual(self.state.read_sgpr(0), 100)

    def test_in_place_accumulator_dst_equals_src1(self) -> None:
        # s1 = s0 + s1
        self.state.write_sgpr(0, 200)
        self.state.write_sgpr(1, 80)
        instr_add = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_ADD_CO_U32,
            dst=SGPROperand(1),
            src0=SGPROperand(0),
            src1=SGPROperand(1),
        )
        res, cc = self.interp.step(instr_add, self.state)
        self.assertEqual(res, 280)
        self.assertEqual(self.state.read_sgpr(1), 280)
        self.assertEqual(self.state.read_sgpr(0), 200)  # src0 preserved

        # s1 = s0 - s1 => 200 - 280 = -80 (mod 2^32), borrow = 1
        instr_sub = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_SUB_CO_U32,
            dst=SGPROperand(1),
            src0=SGPROperand(0),
            src1=SGPROperand(1),
        )
        res_sub, cc_sub = self.interp.step(instr_sub, self.state)
        self.assertEqual(res_sub, (200 - 280) & MASK_U32)
        self.assertEqual(cc_sub, 1)
        self.assertEqual(self.state.read_sgpr(1), (200 - 280) & MASK_U32)
        self.assertEqual(self.state.read_sgpr(0), 200)

    def test_self_referential_source_aliasing(self) -> None:
        # s2 = s0 + s0 (doubling)
        self.state.write_sgpr(0, 0x4000_0000)
        instr_double = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_ADD_CO_U32,
            dst=SGPROperand(2),
            src0=SGPROperand(0),
            src1=SGPROperand(0),
        )
        res_double, cc_double = self.interp.step(instr_double, self.state)
        self.assertEqual(res_double, 0x8000_0000)
        self.assertEqual(cc_double, 0)
        self.assertEqual(self.state.read_sgpr(0), 0x4000_0000)  # Preserved

        # Doubling with carry: 0x8000_0000 + 0x8000_0000 = 0, carry = 1
        self.state.write_sgpr(0, 0x8000_0000)
        res_double2, cc_double2 = self.interp.step(instr_double, self.state)
        self.assertEqual(res_double2, 0)
        self.assertEqual(cc_double2, 1)

        # s3 = s0 - s0 (always 0, borrow = 0)
        instr_zero = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_SUB_CO_U32,
            dst=SGPROperand(3),
            src0=SGPROperand(0),
            src1=SGPROperand(0),
        )
        res_zero, cc_zero = self.interp.step(instr_zero, self.state)
        self.assertEqual(res_zero, 0)
        self.assertEqual(cc_zero, 0)
        self.assertEqual(self.state.read_sgpr(3), 0)

    def test_tri_aliasing_dst_src0_src1(self) -> None:
        # In-place self-addition: s0 = s0 + s0
        self.state.write_sgpr(0, 0x0000_0008)
        instr_tri_add = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_ADD_CO_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(0),
            src1=SGPROperand(0),
        )
        res, cc = self.interp.step(instr_tri_add, self.state)
        self.assertEqual(res, 16)
        self.assertEqual(cc, 0)
        self.assertEqual(self.state.read_sgpr(0), 16)

        # In-place self-subtraction: s0 = s0 - s0
        instr_tri_sub = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_SUB_CO_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(0),
            src1=SGPROperand(0),
        )
        res_sub, cc_sub = self.interp.step(instr_tri_sub, self.state)
        self.assertEqual(res_sub, 0)
        self.assertEqual(cc_sub, 0)
        self.assertEqual(self.state.read_sgpr(0), 0)

    def test_unreferenced_register_preservation(self) -> None:
        # Populate all SGPRs with distinct sentinel values
        for i in range(NUM_SGPRS):
            self.state.write_sgpr(i, (i * 0x0101_0101) & MASK_U32)

        # Perform operation modifying only s42
        instr = SAddSubCoU32Instruction(
            opcode=SOPOpcode.S_ADD_CO_U32,
            dst=SGPROperand(42),
            src0=SGPROperand(10),
            src1=InlineConstant(7),
        )
        self.interp.step(instr, self.state)

        # Verify all registers except s42 maintain exact original value
        for i in range(NUM_SGPRS):
            if i == 42:
                expected = ((10 * 0x0101_0101) + 7) & MASK_U32
                self.assertEqual(self.state.read_sgpr(i), expected)
            else:
                self.assertEqual(self.state.read_sgpr(i), (i * 0x0101_0101) & MASK_U32)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class SALUStatusFlags:
    """Architectural and algebraic condition flags for SALU operations."""
    carry_or_borrow: int
    zero: bool
    sign: bool
    signed_overflow: bool
    parity_even: bool


def evaluate_salu_flags(
    opcode: SOPOpcode, src0: int, src1: int, result: int, cc: int
) -> SALUStatusFlags:
    """Evaluate architectural condition codes and algebraic status flags."""
    s0 = src0 & MASK_U32
    s1 = src1 & MASK_U32
    res = result & MASK_U32

    zero = (res == 0)
    sign = bool(res & SIGN_BIT_U32)

    is_add = opcode in (SOPOpcode.S_ADD_CO_U32, SOPOpcode.S_ADD_U32)
    if is_add:
        # Addition overflow occurs when operands have same sign, but result has different sign
        signed_overflow = bool((~(s0 ^ s1) & (s0 ^ res)) & SIGN_BIT_U32)
    else:
        # Subtraction overflow occurs when operands have different signs, and result sign differs from s0
        signed_overflow = bool(((s0 ^ s1) & (s0 ^ res)) & SIGN_BIT_U32)

    parity_even = (bin(res).count("1") % 2 == 0)

    return SALUStatusFlags(
        carry_or_borrow=cc,
        zero=zero,
        sign=sign,
        signed_overflow=signed_overflow,
        parity_even=parity_even,
    )


def simulate_64bit_add(a64: int, b64: int) -> Tuple[int, int]:
    """Simulate 64-bit addition via s_add_co_u32 (lo) and s_addc_u32 (hi with SCC carry)."""
    a_lo, a_hi = a64 & MASK_U32, (a64 >> 32) & MASK_U32
    b_lo, b_hi = b64 & MASK_U32, (b64 >> 32) & MASK_U32

    # Step 1: low dword s_add_co_u32
    r_lo, carry = oracle_s_add_co_u32(a_lo, b_lo)

    # Step 2: high dword s_addc_u32 (add with carry)
    r_hi_temp = a_hi + b_hi + carry
    r_hi = r_hi_temp & MASK_U32
    carry_out = 1 if r_hi_temp >= MODULO_U32 else 0

    res64 = (r_hi << 32) | r_lo
    return res64, carry_out


def simulate_64bit_sub(a64: int, b64: int) -> Tuple[int, int]:
    """Simulate 64-bit subtraction via s_sub_co_u32 (lo) and s_subb_u32 (hi with SCC borrow)."""
    a_lo, a_hi = a64 & MASK_U32, (a64 >> 32) & MASK_U32
    b_lo, b_hi = b64 & MASK_U32, (b64 >> 32) & MASK_U32

    # Step 1: low dword s_sub_co_u32
    r_lo, borrow = oracle_s_sub_co_u32(a_lo, b_lo)

    # Step 2: high dword s_subb_u32 (sub with borrow)
    diff_hi = a_hi - b_hi - borrow
    r_hi = diff_hi & MASK_U32
    borrow_out = 1 if a_hi < (b_hi + borrow) else 0

    res64 = (r_hi << 32) | r_lo
    return res64, borrow_out


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Verification of condition codes, signed overflow, and multi-word ripple carry/borrow."""

    def test_signed_overflow_vs_unsigned_carry_addition(self) -> None:
        # Case 1: 0x7FFF_FFFF + 1 = 0x8000_0000 (signed overflow, no carry)
        res, cc = oracle_s_add_co_u32(0x7FFF_FFFF, 1)
        flags = evaluate_salu_flags(SOPOpcode.S_ADD_CO_U32, 0x7FFF_FFFF, 1, res, cc)
        self.assertEqual(res, 0x8000_0000)
        self.assertEqual(cc, 0)
        self.assertTrue(flags.signed_overflow)
        self.assertTrue(flags.sign)
        self.assertFalse(flags.zero)

        # Case 2: 0xFFFF_FFFF + 1 = 0 (carry, no signed overflow)
        res2, cc2 = oracle_s_add_co_u32(MAX_U32, 1)
        flags2 = evaluate_salu_flags(SOPOpcode.S_ADD_CO_U32, MAX_U32, 1, res2, cc2)
        self.assertEqual(res2, 0)
        self.assertEqual(cc2, 1)
        self.assertFalse(flags2.signed_overflow)
        self.assertFalse(flags2.sign)
        self.assertTrue(flags2.zero)

        # Case 3: 0x8000_0000 + 0x8000_0000 = 0 (both signed overflow AND carry)
        res3, cc3 = oracle_s_add_co_u32(0x8000_0000, 0x8000_0000)
        flags3 = evaluate_salu_flags(SOPOpcode.S_ADD_CO_U32, 0x8000_0000, 0x8000_0000, res3, cc3)
        self.assertEqual(res3, 0)
        self.assertEqual(cc3, 1)
        self.assertTrue(flags3.signed_overflow)
        self.assertTrue(flags3.zero)

    def test_signed_overflow_vs_unsigned_borrow_subtraction(self) -> None:
        # Case 1: 0x8000_0000 - 1 = 0x7FFF_FFFF (signed overflow, no borrow)
        res, cc = oracle_s_sub_co_u32(0x8000_0000, 1)
        flags = evaluate_salu_flags(SOPOpcode.S_SUB_CO_U32, 0x8000_0000, 1, res, cc)
        self.assertEqual(res, 0x7FFF_FFFF)
        self.assertEqual(cc, 0)
        self.assertTrue(flags.signed_overflow)
        self.assertFalse(flags.sign)

        # Case 2: 0 - 1 = 0xFFFF_FFFF (borrow, no signed overflow)
        res2, cc2 = oracle_s_sub_co_u32(0, 1)
        flags2 = evaluate_salu_flags(SOPOpcode.S_SUB_CO_U32, 0, 1, res2, cc2)
        self.assertEqual(res2, MAX_U32)
        self.assertEqual(cc2, 1)
        self.assertFalse(flags2.signed_overflow)
        self.assertTrue(flags2.sign)

        # Case 3: 0x7FFF_FFFF - 0x8000_0000 = 0xFFFF_FFFF (both signed overflow AND borrow)
        res3, cc3 = oracle_s_sub_co_u32(0x7FFF_FFFF, 0x8000_0000)
        flags3 = evaluate_salu_flags(SOPOpcode.S_SUB_CO_U32, 0x7FFF_FFFF, 0x8000_0000, res3, cc3)
        self.assertEqual(res3, MAX_U32)
        self.assertEqual(cc3, 1)
        self.assertTrue(flags3.signed_overflow)
        self.assertTrue(flags3.sign)

    def test_64bit_ripple_carry_multi_word_addition(self) -> None:
        test_pairs = [
            (0x0000_0000_FFFF_FFFF, 0x0000_0000_0000_0001),
            (0x1234_5678_FFFF_FFFF, 0x0000_0000_0000_0001),
            (0x7FFF_FFFF_FFFF_FFFF, 0x0000_0000_0000_0001),
            (0xFFFF_FFFF_FFFF_FFFF, 0x0000_0000_0000_0001),
            (0xAAAA_AAAA_5555_5555, 0x5555_5555_AAAA_AAAA),
        ]
        for a, b in test_pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                res64, carry_out = simulate_64bit_add(a, b)
                expected_sum = (a + b) & 0xFFFF_FFFF_FFFF_FFFF
                expected_carry = 1 if (a + b) >= (1 << 64) else 0
                self.assertEqual(res64, expected_sum)
                self.assertEqual(carry_out, expected_carry)

    def test_64bit_ripple_borrow_multi_word_subtraction(self) -> None:
        test_pairs = [
            (0x0000_0001_0000_0000, 0x0000_0000_0000_0001),
            (0x1000_0000_0000_0000, 0x0000_0000_0000_0001),
            (0x0000_0000_0000_0000, 0x0000_0000_0000_0001),
            (0x1234_5678_0000_0000, 0x0000_0000_0000_0005),
            (0xFFFF_FFFF_FFFF_FFFF, 0x0000_0000_0000_0001),
        ]
        for a, b in test_pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                res64, borrow_out = simulate_64bit_sub(a, b)
                expected_diff = (a - b) & 0xFFFF_FFFF_FFFF_FFFF
                expected_borrow = 1 if a < b else 0
                self.assertEqual(res64, expected_diff)
                self.assertEqual(borrow_out, expected_borrow)


# ==============================================================================
# IEEE 754 Binary32 Boundary Extremes and Subnormal Bitfield Manipulation
# ==============================================================================

F32_POS_ZERO: int = 0x0000_0000
F32_NEG_ZERO: int = 0x8000_0000
F32_POS_ONE: int = 0x3F80_0000
F32_NEG_ONE: int = 0xBF80_0000
F32_POS_INF: int = 0x7F80_0000
F32_NEG_INF: int = 0xFF80_0000
F32_CANONICAL_QNAN: int = 0x7FC0_0000
F32_SNAN: int = 0x7F80_0001
F32_MIN_SUBNORMAL: int = 0x0000_0001
F32_MAX_SUBNORMAL: int = 0x007F_FFFF
F32_MIN_NORMAL: int = 0x0080_0000
F32_MAX_NORMAL: int = 0x7F7F_FFFF

F32_EXP_MASK: int = 0x7F80_0000
F32_FRAC_MASK: int = 0x007F_FFFF
F32_QNAN_BIT: int = 0x0040_0000


def u32_to_f32(bits: int) -> float:
    """Reinterpret unsigned 32-bit integer as IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK_U32))[0]


def f32_to_u32(val: float) -> int:
    """Reinterpret IEEE 754 single-precision float as unsigned 32-bit integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def is_f32_nan(bits: int) -> bool:
    """Check if bit pattern corresponds to IEEE 754 single-precision NaN."""
    return ((bits & F32_EXP_MASK) == F32_EXP_MASK) and ((bits & F32_FRAC_MASK) != 0)


def is_f32_subnormal(bits: int) -> bool:
    """Check if bit pattern corresponds to IEEE 754 single-precision subnormal."""
    return ((bits & F32_EXP_MASK) == 0) and ((bits & F32_FRAC_MASK) != 0)


def is_f32_inf(bits: int) -> bool:
    """Check if bit pattern corresponds to IEEE 754 single-precision infinity."""
    return ((bits & F32_EXP_MASK) == F32_EXP_MASK) and ((bits & F32_FRAC_MASK) == 0)


class TestIEEE754BitwiseBoundariesAndSubnormals(unittest.TestCase):
    """Verification of SALU integer arithmetic on IEEE 754 bit representations."""

    def test_subnormal_to_normal_boundary_transition(self) -> None:
        # Incrementing largest subnormal gives smallest positive normal
        res, carry = oracle_s_add_co_u32(F32_MAX_SUBNORMAL, 1)
        self.assertEqual(res, F32_MIN_NORMAL)
        self.assertEqual(carry, 0)
        self.assertTrue(is_f32_subnormal(F32_MAX_SUBNORMAL))
        self.assertFalse(is_f32_subnormal(res))
        self.assertAlmostEqual(u32_to_f32(res), 1.1754943508222875e-38, places=44)

        # Decrementing smallest normal gives largest subnormal
        res_sub, borrow = oracle_s_sub_co_u32(F32_MIN_NORMAL, 1)
        self.assertEqual(res_sub, F32_MAX_SUBNORMAL)
        self.assertEqual(borrow, 0)
        self.assertTrue(is_f32_subnormal(res_sub))

    def test_subnormal_zero_boundary(self) -> None:
        # Smallest subnormal - 1 gives +0.0
        res, borrow = oracle_s_sub_co_u32(F32_MIN_SUBNORMAL, 1)
        self.assertEqual(res, F32_POS_ZERO)
        self.assertEqual(borrow, 0)

        # +0.0 - Smallest subnormal gives underflow/borrow
        res_under, borrow_under = oracle_s_sub_co_u32(F32_POS_ZERO, F32_MIN_SUBNORMAL)
        self.assertEqual(res_under, MAX_U32)
        self.assertEqual(borrow_under, 1)

    def test_biased_exponent_scaling_arithmetic(self) -> None:
        # 1.0f is 0x3F80_0000. Adding 1 to the exponent field (0x0080_0000) produces 2.0f (0x4000_0000)
        exp_step = 0x0080_0000
        res_double, carry = oracle_s_add_co_u32(F32_POS_ONE, exp_step)
        self.assertEqual(res_double, 0x4000_0000)
        self.assertEqual(carry, 0)
        self.assertEqual(u32_to_f32(res_double), 2.0)

        # Decrementing exponent field of 1.0f produces 0.5f (0x3F00_0000)
        res_half, borrow = oracle_s_sub_co_u32(F32_POS_ONE, exp_step)
        self.assertEqual(res_half, 0x3F00_0000)
        self.assertEqual(borrow, 0)
        self.assertEqual(u32_to_f32(res_half), 0.5)

    def test_nan_payload_arithmetic(self) -> None:
        # Modifying NaN payload via SALU addition
        res, carry = oracle_s_add_co_u32(F32_CANONICAL_QNAN, 0x0000_0007)
        self.assertEqual(res, 0x7FC0_0007)
        self.assertEqual(carry, 0)
        self.assertTrue(is_f32_nan(res))
        # Ensure quiet bit is preserved
        self.assertTrue(bool(res & F32_QNAN_BIT))

    def test_infinity_boundary_transitions(self) -> None:
        # Adding 1 to +infinity turns it into a signaling NaN pattern (0x7F800001)
        res_snan, carry = oracle_s_add_co_u32(F32_POS_INF, 1)
        self.assertEqual(res_snan, F32_SNAN)
        self.assertEqual(carry, 0)
        self.assertTrue(is_f32_nan(res_snan))

        # Subtracting 1 from +infinity turns it into max normal float (0x7F7FFFFF)
        res_max_norm, borrow = oracle_s_sub_co_u32(F32_POS_INF, 1)
        self.assertEqual(res_max_norm, F32_MAX_NORMAL)
        self.assertEqual(borrow, 0)
        self.assertFalse(is_f32_nan(res_max_norm))
        self.assertFalse(is_f32_inf(res_max_norm))


# ==============================================================================
# Negative Controls, Syntax Rejection, and Parser Validation
# ==============================================================================

def parse_salu_operand(token: str) -> SOP2Operand:
    """Parse a single assembly operand token into a typed SOP2Operand."""
    token = token.strip()
    if not token:
        raise ValueError("Operand token cannot be empty")

    if token.startswith("s") and token[1:].isdigit():
        idx = int(token[1:])
        return SGPROperand(idx)

    if token.startswith("0x") or token.startswith("0X"):
        val = int(token, 16)
        return LiteralU32(val)

    try:
        val = int(token)
        if -16 <= val <= 64:
            return InlineConstant(val)
        return LiteralU32(val)
    except ValueError:
        pass

    raise ValueError(f"Unrecognized operand syntax: '{token}'")


def parse_salu_instruction(asm_line: str) -> SAddSubCoU32Instruction:
    """Parse a SALU assembly instruction string into an AST node."""
    line = asm_line.strip()
    if not line:
        raise ValueError("Empty instruction line")

    parts = line.split(None, 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid instruction format: '{asm_line}'")

    op_str, args_str = parts
    try:
        opcode = SOPOpcode(op_str)
    except ValueError:
        raise ValueError(f"Unsupported opcode: '{op_str}'")

    arg_tokens = [arg.strip() for arg in args_str.split(",")]
    if len(arg_tokens) != 3:
        raise ValueError(f"Expected 3 comma-separated operands, got {len(arg_tokens)}: '{args_str}'")

    dst_token = arg_tokens[0]
    if not (dst_token.startswith("s") and dst_token[1:].isdigit()):
        raise TypeError(f"Destination must be an SGPR, got '{dst_token}'")
    dst = SGPROperand(int(dst_token[1:]))

    src0 = parse_salu_operand(arg_tokens[1])
    src1 = parse_salu_operand(arg_tokens[2])

    return SAddSubCoU32Instruction(opcode=opcode, dst=dst, src0=src0, src1=src1)


class TestNegativeControlsAndSyntaxRejection(unittest.TestCase):
    """Verification that invalid opcodes, illegal syntax, and malformed inputs are rejected."""

    def test_parser_valid_instructions(self) -> None:
        instr1 = parse_salu_instruction("s_add_co_u32 s0, s1, s2")
        self.assertEqual(instr1.opcode, SOPOpcode.S_ADD_CO_U32)
        self.assertEqual(instr1.dst, SGPROperand(0))
        self.assertEqual(instr1.src0, SGPROperand(1))
        self.assertEqual(instr1.src1, SGPROperand(2))

        instr2 = parse_salu_instruction("s_sub_co_u32 s10, s20, 0x1234")
        self.assertEqual(instr2.opcode, SOPOpcode.S_SUB_CO_U32)
        self.assertEqual(instr2.dst, SGPROperand(10))
        self.assertEqual(instr2.src0, SGPROperand(20))
        self.assertEqual(instr2.src1, LiteralU32(0x1234))

        instr3 = parse_salu_instruction("s_add_u32 s5, s6, 1")
        self.assertEqual(instr3.src1, InlineConstant(1))

    def test_parser_unsupported_opcodes(self) -> None:
        invalid_opcodes = [
            "v_add_f32 s0, s1, s2",
            "s_mul_u32 s0, s1, s2",
            "s_invalid_op s0, s1, s2",
            "s_add_co_i32 s0, s1, s2",
        ]
        for line in invalid_opcodes:
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    parse_salu_instruction(line)

    def test_parser_invalid_operands_and_vgpr_rejection(self) -> None:
        # SALU cannot take VGPRs as operands
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_add_co_u32 s0, v1, s2")
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_add_co_u32 s0, s1, v2")
        # SALU destination cannot be VGPR
        with self.assertRaises(TypeError):
            parse_salu_instruction("s_add_co_u32 v0, s1, s2")
        # SALU destination cannot be literal or inline constant
        with self.assertRaises(TypeError):
            parse_salu_instruction("s_add_co_u32 42, s1, s2")

    def test_parser_sgpr_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_add_co_u32 s256, s0, s1")
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_add_co_u32 s0, s500, s1")

    def test_parser_multiple_literals_rejection(self) -> None:
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_add_co_u32 s0, 0x1000, 0x2000")


# ==============================================================================
# Mutation Detection Engine
# ==============================================================================

class TestMutationDetectionEngine(unittest.TestCase):
    """Mutation testing: verifies that all intentional mutants are killed by test vectors."""

    def test_mutant_add_sub_killed(self) -> None:
        # Mutant: replaces addition with subtraction
        def mutant(s0: int, s1: int) -> Tuple[int, int]:
            return (s0 - s1) & MASK_U32, 1 if s0 < s1 else 0

        # Vector (10, 5) detects mutant
        gold = oracle_s_add_co_u32(10, 5)
        mut = mutant(10, 5)
        self.assertNotEqual(gold, mut)

    def test_mutant_add_xor_killed(self) -> None:
        # Mutant: replaces addition with bitwise XOR
        def mutant(s0: int, s1: int) -> Tuple[int, int]:
            return (s0 ^ s1) & MASK_U32, 0

        gold = oracle_s_add_co_u32(1, 1)
        mut = mutant(1, 1)
        self.assertNotEqual(gold, mut)

    def test_mutant_add_no_carry_killed(self) -> None:
        # Mutant: carry is always 0
        def mutant(s0: int, s1: int) -> Tuple[int, int]:
            return (s0 + s1) & MASK_U32, 0

        gold = oracle_s_add_co_u32(MAX_U32, 1)
        mut = mutant(MAX_U32, 1)
        self.assertNotEqual(gold, mut)

    def test_mutant_add_off_by_one_carry_killed(self) -> None:
        # Mutant: carry only when > 0x1_0000_0000 (misses exact threshold)
        def mutant(s0: int, s1: int) -> Tuple[int, int]:
            raw = s0 + s1
            return raw & MASK_U32, 1 if raw > MODULO_U32 else 0

        gold = oracle_s_add_co_u32(MAX_U32, 1)
        mut = mutant(MAX_U32, 1)
        self.assertEqual(gold[1], 1)
        self.assertEqual(mut[1], 0)
        self.assertNotEqual(gold, mut)

    def test_mutant_sub_no_borrow_killed(self) -> None:
        # Mutant: borrow is always 0
        def mutant(s0: int, s1: int) -> Tuple[int, int]:
            return (s0 - s1) & MASK_U32, 0

        gold = oracle_s_sub_co_u32(0, 1)
        mut = mutant(0, 1)
        self.assertNotEqual(gold, mut)

    def test_mutant_sub_strict_underflow_borrow_killed(self) -> None:
        # Mutant: borrow when s0 <= s1 (incorrectly flags borrow on equality)
        def mutant(s0: int, s1: int) -> Tuple[int, int]:
            return (s0 - s1) & MASK_U32, 1 if s0 <= s1 else 0

        gold = oracle_s_sub_co_u32(5, 5)
        mut = mutant(5, 5)
        self.assertEqual(gold[1], 0)
        self.assertEqual(mut[1], 1)
        self.assertNotEqual(gold, mut)

    def test_mutant_truncate_31bit_killed(self) -> None:
        # Mutant: truncates to 31 bits instead of 32 bits
        def mutant(s0: int, s1: int) -> Tuple[int, int]:
            return (s0 + s1) & 0x7FFF_FFFF, 0

        gold = oracle_s_add_co_u32(0x8000_0000, 0)
        mut = mutant(0x8000_0000, 0)
        self.assertNotEqual(gold, mut)


if __name__ == "__main__":
    unittest.main()
