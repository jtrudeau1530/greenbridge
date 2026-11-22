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
# Reference Mathematical Oracles for s_addc_u32 & s_subb_u32
# ==============================================================================

def oracle_s_addc_u32(src0: int, src1: int, cin: int = 0) -> Tuple[int, int]:
    """Reference mathematical oracle for RDNA2 s_addc_u32 / s_add_co_ci_u32.

    Computes 32-bit unsigned addition with carry-in and carry-out:
        D = (S0 + S1 + SCC.in) mod 2^32
        SCC.out = 1 if (S0 + S1 + SCC.in >= 2^32) else 0
    """
    s0 = src0 & MASK_U32
    s1 = src1 & MASK_U32
    c = 1 if cin else 0
    raw_sum = s0 + s1 + c
    res = raw_sum & MASK_U32
    carry_out = 1 if raw_sum >= MODULO_U32 else 0
    return res, carry_out


def oracle_s_subb_u32(src0: int, src1: int, bin_val: int = 0) -> Tuple[int, int]:
    """Reference mathematical oracle for RDNA2 s_subb_u32 / s_sub_co_ci_u32.

    Computes 32-bit unsigned subtraction with borrow-in and borrow-out:
        D = (S0 - S1 - SCC.in) mod 2^32
        SCC.out = 1 if (S0 < S1 + SCC.in) else 0
    """
    s0 = src0 & MASK_U32
    s1 = src1 & MASK_U32
    b = 1 if bin_val else 0
    raw_diff = s0 - s1 - b
    res = raw_diff & MASK_U32
    borrow_out = 1 if s0 < (s1 + b) else 0
    return res, borrow_out


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
# Unit Tests: Reference Oracles & Mathematical Properties
# ==============================================================================

class TestReferenceOracleAddcU32(unittest.TestCase):
    """Exhaustive mathematical verification for oracle_s_addc_u32."""

    def test_zero_identities_without_carry_in(self) -> None:
        self.assertEqual(oracle_s_addc_u32(0, 0, cin=0), (0, 0))
        self.assertEqual(oracle_s_addc_u32(42, 0, cin=0), (42, 0))
        self.assertEqual(oracle_s_addc_u32(0, 1337, cin=0), (1337, 0))
        self.assertEqual(oracle_s_addc_u32(MAX_U32, 0, cin=0), (MAX_U32, 0))

    def test_zero_identities_with_carry_in(self) -> None:
        self.assertEqual(oracle_s_addc_u32(0, 0, cin=1), (1, 0))
        self.assertEqual(oracle_s_addc_u32(42, 0, cin=1), (43, 0))
        self.assertEqual(oracle_s_addc_u32(0, 1337, cin=1), (1338, 0))
        # MAX_U32 + 0 + 1 = 0 with carry out = 1
        self.assertEqual(oracle_s_addc_u32(MAX_U32, 0, cin=1), (0, 1))

    def test_commutativity(self) -> None:
        vectors = [
            (0, 1),
            (100, 200),
            (MAX_U32, 1),
            (0x8000_0000, 0x8000_0000),
            (0x1234_5678, 0x9ABC_DEF0),
        ]
        for cin in (0, 1):
            for a, b in vectors:
                with self.subTest(a=a, b=b, cin=cin):
                    res_ab, c_ab = oracle_s_addc_u32(a, b, cin)
                    res_ba, c_ba = oracle_s_addc_u32(b, a, cin)
                    self.assertEqual(res_ab, res_ba)
                    self.assertEqual(c_ab, c_ba)

    def test_boundary_carry_transitions(self) -> None:
        # MAX_U32 - 1 + 1 with cin=0 => MAX_U32, carry=0
        self.assertEqual(oracle_s_addc_u32(MAX_U32 - 1, 1, cin=0), (MAX_U32, 0))
        # MAX_U32 - 1 + 1 with cin=1 => 0, carry=1
        self.assertEqual(oracle_s_addc_u32(MAX_U32 - 1, 1, cin=1), (0, 1))
        # MAX_U32 + MAX_U32 + cin=1 => MAX_U32, carry=1
        self.assertEqual(oracle_s_addc_u32(MAX_U32, MAX_U32, cin=1), (MAX_U32, 1))


class TestReferenceOracleSubbU32(unittest.TestCase):
    """Exhaustive mathematical verification for oracle_s_subb_u32."""

    def test_zero_identities(self) -> None:
        self.assertEqual(oracle_s_subb_u32(0, 0, bin_val=0), (0, 0))
        self.assertEqual(oracle_s_subb_u32(42, 0, bin_val=0), (42, 0))
        self.assertEqual(oracle_s_subb_u32(42, 0, bin_val=1), (41, 0))
        self.assertEqual(oracle_s_subb_u32(0, 0, bin_val=1), (MAX_U32, 1))

    def test_self_subtraction(self) -> None:
        values = [0, 1, 42, 0x7FFF_FFFF, 0x8000_0000, MAX_U32]
        for v in values:
            with self.subTest(v=v):
                # v - v - 0 = 0, borrow 0
                res, borrow = oracle_s_subb_u32(v, v, bin_val=0)
                self.assertEqual(res, 0)
                self.assertEqual(borrow, 0)
                # v - v - 1 = MAX_U32, borrow 1
                res_b, borrow_b = oracle_s_subb_u32(v, v, bin_val=1)
                self.assertEqual(res_b, MAX_U32)
                self.assertEqual(borrow_b, 1)

    def test_boundary_borrow_transitions(self) -> None:
        self.assertEqual(oracle_s_subb_u32(0, 1, bin_val=0), (MAX_U32, 1))
        self.assertEqual(oracle_s_subb_u32(0, 1, bin_val=1), (MAX_U32 - 1, 1))
        self.assertEqual(oracle_s_subb_u32(1, 1, bin_val=1), (MAX_U32, 1))
        self.assertEqual(oracle_s_subb_u32(2, 1, bin_val=1), (0, 0))


class TestArchitecturalModel(unittest.TestCase):
    """Verification of the ScalarModel register file and SCC status representation."""

    def test_sgpr_read_write(self) -> None:
        model = ScalarModel()
        model.write_sgpr(5, 0x1234_5678)
        self.assertEqual(model.read_sgpr(5), 0x1234_5678)
        model.write_sgpr(6, 0x1_AAAA_BBBB)
        self.assertEqual(model.read_sgpr(6), 0xAAAA_BBBB)

    def test_sgpr_bounds(self) -> None:
        model = ScalarModel()
        with self.assertRaises(IndexError):
            model.read_sgpr(-1)
        with self.assertRaises(IndexError):
            model.read_sgpr(NUM_SGPRS)

    def test_scc_flag(self) -> None:
        model = ScalarModel()
        self.assertEqual(model.scc, 0)
        model.set_scc(1)
        self.assertEqual(model.scc, 1)
        model.set_scc(0)
        self.assertEqual(model.scc, 0)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class SOPOpcode(Enum):
    """SOP2 scalar ALU opcodes with carry-in and carry-out."""
    S_ADDC_U32 = "s_addc_u32"
    S_SUBB_U32 = "s_subb_u32"
    S_ADD_CO_CI_U32 = "s_add_co_ci_u32"
    S_SUB_CO_CI_U32 = "s_sub_co_ci_u32"


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
class SAddSubbCoCiU32Instruction:
    """Symbolic AST node representing a 32-bit SALU add/sub with carry-in and carry-out."""
    opcode: SOPOpcode
    dst: SGPROperand
    src0: SOP2Operand
    src1: SOP2Operand

    def __post_init__(self) -> None:
        if not isinstance(self.dst, SGPROperand):
            raise TypeError(f"Destination operand must be SGPROperand, got {type(self.dst).__name__}")
        if self.opcode not in (
            SOPOpcode.S_ADDC_U32,
            SOPOpcode.S_SUBB_U32,
            SOPOpcode.S_ADD_CO_CI_U32,
            SOPOpcode.S_SUB_CO_CI_U32,
        ):
            raise ValueError(f"Invalid opcode for SAddSubbCoCiU32Instruction: {self.opcode}")

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
        instr_add = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_ADDC_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(1),
            src1=SGPROperand(2),
        )
        self.assertEqual(instr_add.disassemble(), "s_addc_u32 s0, s1, s2")

        instr_sub = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_SUBB_U32,
            dst=SGPROperand(4),
            src0=SGPROperand(5),
            src1=LiteralU32(0xDEADBEEF),
        )
        self.assertEqual(instr_sub.disassemble(), "s_subb_u32 s4, s5, 0xDEADBEEF")

    def test_instruction_invalid_dst(self) -> None:
        with self.assertRaises(TypeError):
            SAddSubbCoCiU32Instruction(  # type: ignore[arg-type]
                opcode=SOPOpcode.S_ADDC_U32,
                dst=LiteralU32(42),  # type: ignore[arg-type]
                src0=SGPROperand(0),
                src1=SGPROperand(1),
            )

    def test_instruction_multiple_literals_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SAddSubbCoCiU32Instruction(
                opcode=SOPOpcode.S_ADDC_U32,
                dst=SGPROperand(0),
                src0=LiteralU32(0x100),
                src1=LiteralU32(0x200),
            )


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 SALU instructions with carry/borrow in."""

    def resolve_operand(self, op: SOP2Operand, state: ScalarModel) -> int:
        if isinstance(op, SGPROperand):
            return state.read_sgpr(op.index)
        elif isinstance(op, LiteralU32):
            return op.value & MASK_U32
        elif isinstance(op, InlineConstant):
            return op.value & MASK_U32
        else:
            raise TypeError(f"Unsupported operand type: {type(op).__name__}")

    def step(self, instr: SAddSubbCoCiU32Instruction, state: ScalarModel) -> Tuple[int, int]:
        # Atomic read phase
        s0_val = self.resolve_operand(instr.src0, state)
        s1_val = self.resolve_operand(instr.src1, state)
        cin = state.scc

        # Computation phase
        if instr.opcode in (SOPOpcode.S_ADDC_U32, SOPOpcode.S_ADD_CO_CI_U32):
            res, cc_out = oracle_s_addc_u32(s0_val, s1_val, cin=cin)
        elif instr.opcode in (SOPOpcode.S_SUBB_U32, SOPOpcode.S_SUB_CO_CI_U32):
            res, cc_out = oracle_s_subb_u32(s0_val, s1_val, bin_val=cin)
        else:
            raise ValueError(f"Unsupported opcode: {instr.opcode}")

        # Update condition code and destination SGPR
        state.set_scc(cc_out)
        state.write_sgpr(instr.dst.index, res)
        return res, cc_out

    def run_program(
        self, program: List[SAddSubbCoCiU32Instruction], state: ScalarModel
    ) -> List[Tuple[int, int]]:
        results: List[Tuple[int, int]] = []
        for instr in program:
            results.append(self.step(instr, state))
        return results


class TestScalarInterpreter(unittest.TestCase):
    """Verification of deterministic scalar instruction interpretation with SCC carry-in/out."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarModel()

    def test_scalar_addc_with_zero_scc(self) -> None:
        self.state.set_scc(0)
        self.state.write_sgpr(1, 100)
        self.state.write_sgpr(2, 200)
        instr = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_ADDC_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(1),
            src1=SGPROperand(2),
        )
        res, cc = self.interp.step(instr, self.state)
        self.assertEqual(res, 300)
        self.assertEqual(cc, 0)
        self.assertEqual(self.state.read_sgpr(0), 300)
        self.assertEqual(self.state.scc, 0)

    def test_scalar_addc_with_one_scc(self) -> None:
        self.state.set_scc(1)
        self.state.write_sgpr(1, 100)
        self.state.write_sgpr(2, 200)
        instr = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_ADDC_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(1),
            src1=SGPROperand(2),
        )
        res, cc = self.interp.step(instr, self.state)
        self.assertEqual(res, 301)
        self.assertEqual(cc, 0)
        self.assertEqual(self.state.read_sgpr(0), 301)
        self.assertEqual(self.state.scc, 0)

    def test_scalar_addc_carry_out_propagation(self) -> None:
        self.state.set_scc(1)
        self.state.write_sgpr(1, MAX_U32)
        self.state.write_sgpr(2, 0)
        instr = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_ADDC_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(1),
            src1=SGPROperand(2),
        )
        res, cc = self.interp.step(instr, self.state)
        self.assertEqual(res, 0)
        self.assertEqual(cc, 1)
        self.assertEqual(self.state.read_sgpr(0), 0)
        self.assertEqual(self.state.scc, 1)

    def test_scalar_subb_with_zero_scc(self) -> None:
        self.state.set_scc(0)
        self.state.write_sgpr(1, 50)
        self.state.write_sgpr(2, 20)
        instr = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_SUBB_U32,
            dst=SGPROperand(3),
            src0=SGPROperand(1),
            src1=SGPROperand(2),
        )
        res, cc = self.interp.step(instr, self.state)
        self.assertEqual(res, 30)
        self.assertEqual(cc, 0)
        self.assertEqual(self.state.read_sgpr(3), 30)
        self.assertEqual(self.state.scc, 0)

    def test_scalar_subb_with_one_scc(self) -> None:
        self.state.set_scc(1)
        self.state.write_sgpr(1, 50)
        self.state.write_sgpr(2, 20)
        instr = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_SUBB_U32,
            dst=SGPROperand(3),
            src0=SGPROperand(1),
            src1=SGPROperand(2),
        )
        res, cc = self.interp.step(instr, self.state)
        self.assertEqual(res, 29)
        self.assertEqual(cc, 0)
        self.assertEqual(self.state.read_sgpr(3), 29)
        self.assertEqual(self.state.scc, 0)

    def test_scalar_subb_borrow_out_propagation(self) -> None:
        self.state.set_scc(1)
        self.state.write_sgpr(1, 10)
        self.state.write_sgpr(2, 10)
        instr = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_SUBB_U32,
            dst=SGPROperand(3),
            src0=SGPROperand(1),
            src1=SGPROperand(2),
        )
        res, cc = self.interp.step(instr, self.state)
        self.assertEqual(res, MAX_U32)
        self.assertEqual(cc, 1)
        self.assertEqual(self.state.read_sgpr(3), MAX_U32)
        self.assertEqual(self.state.scc, 1)


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
        self, instr: SAddSubbCoCiU32Instruction, simd_state: SIMDMultiLaneState
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
                out_val, _ = oracle_s_addc_u32(lane_val, scalar_val, cin=0)
                simd_state.write_lane_vgpr(lane, dst_vgpr, out_val)
                results[lane] = out_val
        return results


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Verification of SIMD multi-lane divergence and 32/64-bit EXEC masking interaction."""

    def setUp(self) -> None:
        self.engine = SIMDExecutionEngine()

    def test_salu_unconditional_execution_with_zero_exec_mask(self) -> None:
        for wave_size in (32, 64):
            with self.subTest(wave_size=wave_size):
                state = SIMDMultiLaneState(wave_size=wave_size)
                state.exec_mask = 0  # Inactive wavefront
                state.scalar_model.set_scc(1)
                state.scalar_model.write_sgpr(1, 100)
                state.scalar_model.write_sgpr(2, 200)

                instr = SAddSubbCoCiU32Instruction(
                    opcode=SOPOpcode.S_ADDC_U32,
                    dst=SGPROperand(0),
                    src0=SGPROperand(1),
                    src1=SGPROperand(2),
                )
                res, cc = self.engine.step_salu(instr, state)
                self.assertEqual(res, 301)
                self.assertEqual(cc, 0)
                # SGPR and SCC must update despite exec_mask == 0
                self.assertEqual(state.scalar_model.read_sgpr(0), 301)
                self.assertEqual(state.scalar_model.scc, 0)

    def test_salu_broadcast_to_wave32(self) -> None:
        state = SIMDMultiLaneState(wave_size=32)
        state.exec_mask = 0x5555_5555  # Even lanes active
        for lane in range(32):
            state.write_lane_vgpr(lane, 0, lane * 20)
            state.write_lane_vgpr(lane, 1, 0x9999_9999)

        state.scalar_model.set_scc(1)
        state.scalar_model.write_sgpr(2, 500)
        salu_instr = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_ADDC_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(2),
            src1=InlineConstant(9),
        )
        self.engine.step_salu(salu_instr, state)
        # 500 + 9 + 1 = 510
        self.assertEqual(state.scalar_model.read_sgpr(0), 510)

        results = self.engine.vector_add_scalar_broadcast(dst_vgpr=1, src_vgpr=0, src_sgpr=0, simd_state=state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], (lane * 20) + 510)
                self.assertEqual(state.read_lane_vgpr(lane, 1), (lane * 20) + 510)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.read_lane_vgpr(lane, 1), 0x9999_9999)

    def test_salu_broadcast_to_wave64(self) -> None:
        state = SIMDMultiLaneState(wave_size=64)
        state.exec_mask = 0xFFFF_0000_FFFF_0000
        for lane in range(64):
            state.write_lane_vgpr(lane, 3, 100)
            state.write_lane_vgpr(lane, 4, 0x1111_1111)

        state.scalar_model.set_scc(1)
        state.scalar_model.write_sgpr(8, 200)
        salu_subb = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_SUBB_U32,
            dst=SGPROperand(5),
            src0=SGPROperand(8),
            src1=InlineConstant(50),
        )
        self.engine.step_salu(salu_subb, state)
        # 200 - 50 - 1 = 149
        self.assertEqual(state.scalar_model.read_sgpr(5), 149)
        self.assertEqual(state.scalar_model.scc, 0)

        results = self.engine.vector_add_scalar_broadcast(dst_vgpr=4, src_vgpr=3, src_sgpr=5, simd_state=state)
        for lane in range(64):
            if (state.exec_mask & (1 << lane)) != 0:
                self.assertEqual(results[lane], 249)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.read_lane_vgpr(lane, 4), 0x1111_1111)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasingAndPreservation(unittest.TestCase):
    """Verification of atomic read-before-write semantics and register aliasing with carry/borrow."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarModel()

    def test_in_place_accumulator_dst_equals_src0(self) -> None:
        # s0 = s0 + s1 + cin
        self.state.set_scc(1)
        self.state.write_sgpr(0, 100)
        self.state.write_sgpr(1, 50)
        instr_add = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_ADDC_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(0),
            src1=SGPROperand(1),
        )
        res, cc = self.interp.step(instr_add, self.state)
        # 100 + 50 + 1 = 151
        self.assertEqual(res, 151)
        self.assertEqual(cc, 0)
        self.assertEqual(self.state.read_sgpr(0), 151)
        self.assertEqual(self.state.read_sgpr(1), 50)  # src1 preserved

        # s0 = s0 - s1 - bin
        self.state.set_scc(1)
        instr_sub = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_SUBB_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(0),
            src1=SGPROperand(1),
        )
        res_sub, cc_sub = self.interp.step(instr_sub, self.state)
        # 151 - 50 - 1 = 100
        self.assertEqual(res_sub, 100)
        self.assertEqual(cc_sub, 0)
        self.assertEqual(self.state.read_sgpr(0), 100)

    def test_in_place_accumulator_dst_equals_src1(self) -> None:
        # s1 = s0 + s1 + cin
        self.state.set_scc(1)
        self.state.write_sgpr(0, 200)
        self.state.write_sgpr(1, 80)
        instr_add = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_ADDC_U32,
            dst=SGPROperand(1),
            src0=SGPROperand(0),
            src1=SGPROperand(1),
        )
        res, cc = self.interp.step(instr_add, self.state)
        # 200 + 80 + 1 = 281
        self.assertEqual(res, 281)
        self.assertEqual(self.state.read_sgpr(1), 281)
        self.assertEqual(self.state.read_sgpr(0), 200)  # src0 preserved

    def test_self_referential_source_aliasing(self) -> None:
        # s2 = s0 + s0 + cin (doubling + 1)
        self.state.set_scc(1)
        self.state.write_sgpr(0, 0x4000_0000)
        instr_double = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_ADDC_U32,
            dst=SGPROperand(2),
            src0=SGPROperand(0),
            src1=SGPROperand(0),
        )
        res_double, cc_double = self.interp.step(instr_double, self.state)
        self.assertEqual(res_double, 0x8000_0001)
        self.assertEqual(cc_double, 0)
        self.assertEqual(self.state.read_sgpr(0), 0x4000_0000)

        # s3 = s0 - s0 - bin
        # If bin == 1, s0 - s0 - 1 = -1 (0xFFFFFFFF) with borrow = 1
        self.state.set_scc(1)
        instr_self_sub = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_SUBB_U32,
            dst=SGPROperand(3),
            src0=SGPROperand(0),
            src1=SGPROperand(0),
        )
        res_sub, cc_sub = self.interp.step(instr_self_sub, self.state)
        self.assertEqual(res_sub, MAX_U32)
        self.assertEqual(cc_sub, 1)

    def test_tri_aliasing_dst_src0_src1(self) -> None:
        # s0 = s0 + s0 + cin
        self.state.set_scc(1)
        self.state.write_sgpr(0, 15)
        instr_tri_add = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_ADDC_U32,
            dst=SGPROperand(0),
            src0=SGPROperand(0),
            src1=SGPROperand(0),
        )
        res, cc = self.interp.step(instr_tri_add, self.state)
        # 15 + 15 + 1 = 31
        self.assertEqual(res, 31)
        self.assertEqual(cc, 0)
        self.assertEqual(self.state.read_sgpr(0), 31)

    def test_unreferenced_register_preservation(self) -> None:
        for i in range(NUM_SGPRS):
            self.state.write_sgpr(i, (i * 0x0202_0202) & MASK_U32)

        self.state.set_scc(1)
        instr = SAddSubbCoCiU32Instruction(
            opcode=SOPOpcode.S_ADDC_U32,
            dst=SGPROperand(77),
            src0=SGPROperand(10),
            src1=InlineConstant(3),
        )
        self.interp.step(instr, self.state)

        for i in range(NUM_SGPRS):
            if i == 77:
                expected = ((10 * 0x0202_0202) + 3 + 1) & MASK_U32
                self.assertEqual(self.state.read_sgpr(i), expected)
            else:
                self.assertEqual(self.state.read_sgpr(i), (i * 0x0202_0202) & MASK_U32)


# ==============================================================================
# Condition Codes, Status Flags, and Extended Precision Propagation
# ==============================================================================

@dataclass(frozen=True)
class SALUStatusFlags:
    """Architectural condition codes and algebraic status flags."""
    carry_or_borrow: int
    zero: bool
    sign: bool
    signed_overflow: bool
    parity_even: bool


def evaluate_salu_co_ci_flags(
    opcode: SOPOpcode, src0: int, src1: int, cin: int, result: int, cc_out: int
) -> SALUStatusFlags:
    """Evaluate condition codes including signed overflow for carry-in operations."""
    s0 = src0 & MASK_U32
    s1 = src1 & MASK_U32
    c = 1 if cin else 0
    res = result & MASK_U32

    zero = (res == 0)
    sign = bool(res & SIGN_BIT_U32)

    # Convert to signed 32-bit integer for signed overflow detection
    def to_signed(val: int) -> int:
        return val - MODULO_U32 if (val & SIGN_BIT_U32) else val

    s0_signed = to_signed(s0)
    s1_signed = to_signed(s1)

    is_add = opcode in (SOPOpcode.S_ADDC_U32, SOPOpcode.S_ADD_CO_CI_U32)
    if is_add:
        exact_sum = s0_signed + s1_signed + c
        signed_overflow = not (-(1 << 31) <= exact_sum < (1 << 31))
    else:
        exact_diff = s0_signed - s1_signed - c
        signed_overflow = not (-(1 << 31) <= exact_diff < (1 << 31))

    parity_even = (bin(res).count("1") % 2 == 0)

    return SALUStatusFlags(
        carry_or_borrow=cc_out,
        zero=zero,
        sign=sign,
        signed_overflow=signed_overflow,
        parity_even=parity_even,
    )


def simulate_multi_dword_add(words_a: List[int], words_b: List[int]) -> Tuple[List[int], int]:
    """Chain 32-bit additions across N dwords using s_addc_u32."""
    assert len(words_a) == len(words_b)
    res_words: List[int] = []
    carry = 0
    for a, b in zip(words_a, words_b):
        res, carry = oracle_s_addc_u32(a, b, cin=carry)
        res_words.append(res)
    return res_words, carry


def simulate_multi_dword_sub(words_a: List[int], words_b: List[int]) -> Tuple[List[int], int]:
    """Chain 32-bit subtractions across N dwords using s_subb_u32."""
    assert len(words_a) == len(words_b)
    res_words: List[int] = []
    borrow = 0
    for a, b in zip(words_a, words_b):
        res, borrow = oracle_s_subb_u32(a, b, bin_val=borrow)
        res_words.append(res)
    return res_words, borrow


class TestConditionCodesAndExtendedPrecisionPropagation(unittest.TestCase):
    """Verification of condition codes and multi-dword ripple carry/borrow propagation."""

    def test_signed_overflow_evaluation(self) -> None:
        # 0x7FFF_FFFF + 0 + cin=1 => 0x8000_0000 (signed overflow)
        res, cc = oracle_s_addc_u32(0x7FFF_FFFF, 0, cin=1)
        flags = evaluate_salu_co_ci_flags(SOPOpcode.S_ADDC_U32, 0x7FFF_FFFF, 0, 1, res, cc)
        self.assertEqual(res, 0x8000_0000)
        self.assertEqual(cc, 0)
        self.assertTrue(flags.signed_overflow)
        self.assertTrue(flags.sign)

        # 0x8000_0000 - 0 - bin=1 => 0x7FFF_FFFF (signed overflow)
        res_s, cc_s = oracle_s_subb_u32(0x8000_0000, 0, bin_val=1)
        flags_s = evaluate_salu_co_ci_flags(SOPOpcode.S_SUBB_U32, 0x8000_0000, 0, 1, res_s, cc_s)
        self.assertEqual(res_s, 0x7FFF_FFFF)
        self.assertEqual(cc_s, 0)
        self.assertTrue(flags_s.signed_overflow)
        self.assertFalse(flags_s.sign)

    def test_96bit_ripple_carry_addition(self) -> None:
        # A = 0x00000000_FFFFFFFF_FFFFFFFF, B = 0x00000000_00000000_00000001
        words_a = [MAX_U32, MAX_U32, 0]
        words_b = [1, 0, 0]
        res_words, carry_out = simulate_multi_dword_add(words_a, words_b)
        self.assertEqual(res_words, [0, 0, 1])
        self.assertEqual(carry_out, 0)

    def test_96bit_ripple_borrow_subtraction(self) -> None:
        # A = 0x00000001_00000000_00000000, B = 0x00000000_00000000_00000001
        words_a = [0, 0, 1]
        words_b = [1, 0, 0]
        res_words, borrow_out = simulate_multi_dword_sub(words_a, words_b)
        self.assertEqual(res_words, [MAX_U32, MAX_U32, 0])
        self.assertEqual(borrow_out, 0)

    def test_128bit_ripple_arithmetic_against_bigint(self) -> None:
        vectors_128 = [
            (0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF, 1),
            ((1 << 96) - 1, 1),
            (0x1234_5678_9ABC_DEF0_1357_9BDF_0246_8ACE, 0xEDCB_A987_6543_210F_ECCA_8642_FD75_3190),
        ]
        for a128, b128 in vectors_128:
            with self.subTest(a=hex(a128), b=hex(b128)):
                words_a = [(a128 >> (i * 32)) & MASK_U32 for i in range(4)]
                words_b = [(b128 >> (i * 32)) & MASK_U32 for i in range(4)]

                # Addition
                res_add, c_out = simulate_multi_dword_add(words_a, words_b)
                sum_int = sum(w << (i * 32) for i, w in enumerate(res_add))
                self.assertEqual(sum_int, (a128 + b128) & ((1 << 128) - 1))
                self.assertEqual(c_out, 1 if (a128 + b128) >= (1 << 128) else 0)

                # Subtraction
                res_sub, b_out = simulate_multi_dword_sub(words_a, words_b)
                diff_int = sum(w << (i * 32) for i, w in enumerate(res_sub))
                self.assertEqual(diff_int, (a128 - b128) & ((1 << 128) - 1))
                self.assertEqual(b_out, 1 if a128 < b128 else 0)


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

    def test_subnormal_to_normal_transition_via_carry_in(self) -> None:
        # F32_MAX_SUBNORMAL + 0 + cin=1 => F32_MIN_NORMAL
        res, carry = oracle_s_addc_u32(F32_MAX_SUBNORMAL, 0, cin=1)
        self.assertEqual(res, F32_MIN_NORMAL)
        self.assertEqual(carry, 0)
        self.assertTrue(is_f32_subnormal(F32_MAX_SUBNORMAL))
        self.assertFalse(is_f32_subnormal(res))
        self.assertAlmostEqual(u32_to_f32(res), 1.1754943508222875e-38, places=44)

        # F32_MIN_NORMAL - 0 - bin=1 => F32_MAX_SUBNORMAL
        res_sub, borrow = oracle_s_subb_u32(F32_MIN_NORMAL, 0, bin_val=1)
        self.assertEqual(res_sub, F32_MAX_SUBNORMAL)
        self.assertEqual(borrow, 0)
        self.assertTrue(is_f32_subnormal(res_sub))

    def test_subnormal_zero_boundary_with_borrow_in(self) -> None:
        # Smallest subnormal - 0 - bin=1 gives +0.0
        res, borrow = oracle_s_subb_u32(F32_MIN_SUBNORMAL, 0, bin_val=1)
        self.assertEqual(res, F32_POS_ZERO)
        self.assertEqual(borrow, 0)

        # +0.0 - 0 - bin=1 gives 0xFFFFFFFF with borrow = 1
        res_under, borrow_under = oracle_s_subb_u32(F32_POS_ZERO, 0, bin_val=1)
        self.assertEqual(res_under, MAX_U32)
        self.assertEqual(borrow_under, 1)

    def test_biased_exponent_scaling_with_carry_in(self) -> None:
        exp_step = 0x0080_0000
        # 1.0f (0x3F80_0000) + (exp_step - 1) + cin=1 => 2.0f (0x4000_0000)
        res_double, carry = oracle_s_addc_u32(F32_POS_ONE, exp_step - 1, cin=1)
        self.assertEqual(res_double, 0x4000_0000)
        self.assertEqual(carry, 0)
        self.assertEqual(u32_to_f32(res_double), 2.0)

        # 1.0f - (exp_step - 1) - bin=1 => 0.5f (0x3F00_0000)
        res_half, borrow = oracle_s_subb_u32(F32_POS_ONE, exp_step - 1, bin_val=1)
        self.assertEqual(res_half, 0x3F00_0000)
        self.assertEqual(borrow, 0)
        self.assertEqual(u32_to_f32(res_half), 0.5)

    def test_nan_payload_arithmetic_with_carry_in(self) -> None:
        # Adding cin=1 to quiet NaN
        res, carry = oracle_s_addc_u32(F32_CANONICAL_QNAN, 0, cin=1)
        self.assertEqual(res, F32_CANONICAL_QNAN + 1)
        self.assertEqual(carry, 0)
        self.assertTrue(is_f32_nan(res))
        self.assertTrue(bool(res & F32_QNAN_BIT))

    def test_infinity_boundary_transitions_with_carry_in(self) -> None:
        # +infinity + 0 + cin=1 => signaling NaN (0x7F800001)
        res_snan, carry = oracle_s_addc_u32(F32_POS_INF, 0, cin=1)
        self.assertEqual(res_snan, F32_SNAN)
        self.assertEqual(carry, 0)
        self.assertTrue(is_f32_nan(res_snan))

        # +infinity - 0 - bin=1 => max normal float (0x7F7FFFFF)
        res_max_norm, borrow = oracle_s_subb_u32(F32_POS_INF, 0, bin_val=1)
        self.assertEqual(res_max_norm, F32_MAX_NORMAL)
        self.assertEqual(borrow, 0)
        self.assertFalse(is_f32_nan(res_max_norm))
        self.assertFalse(is_f32_inf(res_max_norm))


if __name__ == "__main__":
    unittest.main()
