"""Semantic reference oracle and verification for RDNA2 s_nand_saveexec_b64 and s_nor_saveexec_b64.

RDNA2 SALU 64-bit SaveExec Instructions:
  s_nand_saveexec_b64:
    SDST = EXEC
    EXEC = ~(SSRC0 & EXEC) & 0xFFFFFFFFFFFFFFFF
    SCC  = 1 if EXEC != 0 else 0

  s_nor_saveexec_b64:
    SDST = EXEC
    EXEC = ~(SSRC0 | EXEC) & 0xFFFFFFFFFFFFFFFF
    SCC  = 1 if EXEC != 0 else 0

Phase 1: Reference mathematical oracle and architectural model.
"""
from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union, Dict, Any

# Bitwise 32-bit and 64-bit architectural masks
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
ALL64: int = 0xFFFF_FFFF_FFFF_FFFF

# IEEE 754 binary64 (double precision) architectural constants
SIGN_MASK_F64: int = 0x8000_0000_0000_0000
EXP_MASK_F64: int = 0x7FF0_0000_0000_0000
FRAC_MASK_F64: int = 0x000F_FFFF_FFFF_FFFF
QNAN_BIT_F64: int = 0x0008_0000_0000_0000
CANONICAL_QNAN_F64: int = 0x7FF8_0000_0000_0000

POS_ZERO_F64: int = 0x0000_0000_0000_0000
NEG_ZERO_F64: int = 0x8000_0000_0000_0000
POS_INF_F64: int = 0x7FF0_0000_0000_0000
NEG_INF_F64: int = 0xFFF0_0000_0000_0000


def words_to_u64(lo: int, hi: int) -> int:
    """Combine two 32-bit scalar register words (lo, hi) into a 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def u64_to_words(val: int) -> Tuple[int, int]:
    """Split a 64-bit integer into two 32-bit scalar register words (lo, hi)."""
    return val & MASK32, (val >> 32) & MASK32


def f64_to_bits(val: float) -> int:
    """Convert a Python float (IEEE 754 double) to 64-bit integer bits."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def bits_to_f64(bits: int) -> float:
    """Convert 64-bit integer bits to a Python float (IEEE 754 double)."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def popcount64(val: int) -> int:
    """Return the number of set bits in a 64-bit integer."""
    return bin(val & MASK64).count("1")


def find_first_one_64(val: int) -> int:
    """Find the lowest set bit index [0..63] in a 64-bit integer, or -1 if zero."""
    v = val & MASK64
    if v == 0:
        return -1
    return (v & -v).bit_length() - 1


# ==============================================================================
# Reference Mathematical Oracles
# ==============================================================================

def oracle_s_nand_b64(s0: int, s1: int) -> int:
    """Reference mathematical oracle for 64-bit bitwise NAND: D = ~(s0 & s1)."""
    return (~(s0 & s1)) & MASK64


def oracle_s_nor_b64(s0: int, s1: int) -> int:
    """Reference mathematical oracle for 64-bit bitwise NOR: D = ~(s0 | s1)."""
    return (~(s0 | s1)) & MASK64


def oracle_s_nand_saveexec_b64(ssrc0: int, old_exec: int) -> Tuple[int, int, int]:
    """Reference architectural oracle for RDNA2 s_nand_saveexec_b64.

    Semantics:
      saved_dst = old_exec
      new_exec  = ~(ssrc0 & old_exec) & 0xFFFFFFFFFFFFFFFF
      scc       = 1 if new_exec != 0 else 0

    Returns:
      (saved_dst, new_exec, scc)
    """
    saved_dst = old_exec & MASK64
    new_exec = (~(ssrc0 & old_exec)) & MASK64
    scc = 1 if new_exec != 0 else 0
    return saved_dst, new_exec, scc


def oracle_s_nor_saveexec_b64(ssrc0: int, old_exec: int) -> Tuple[int, int, int]:
    """Reference architectural oracle for RDNA2 s_nor_saveexec_b64.

    Semantics:
      saved_dst = old_exec
      new_exec  = ~(ssrc0 | old_exec) & 0xFFFFFFFFFFFFFFFF
      scc       = 1 if new_exec != 0 else 0

    Returns:
      (saved_dst, new_exec, scc)
    """
    saved_dst = old_exec & MASK64
    new_exec = (~(ssrc0 | old_exec)) & MASK64
    scc = 1 if new_exec != 0 else 0
    return saved_dst, new_exec, scc


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class Opcode(str, Enum):
    S_NAND_SAVEEXEC_B64 = "s_nand_saveexec_b64"
    S_NOR_SAVEEXEC_B64 = "s_nor_saveexec_b64"


class RegType(Enum):
    SGPR = auto()
    EXEC = auto()
    VCC = auto()
    SCC = auto()


@dataclass(frozen=True)
class Reg64:
    """Represents a 64-bit architectural register (SGPR pair, EXEC, or VCC)."""
    reg_type: RegType
    index: int = 0

    def __post_init__(self) -> None:
        if self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 254):
                raise ValueError(f"SGPR pair base index {self.index} out of range [0, 254]")
        elif self.reg_type in (RegType.EXEC, RegType.VCC, RegType.SCC):
            if self.index != 0:
                raise ValueError(f"Special register {self.reg_type.name} does not use non-zero index")

    @property
    def lo_reg(self) -> int:
        return self.index

    @property
    def hi_reg(self) -> int:
        return self.index + 1

    @property
    def name(self) -> str:
        if self.reg_type == RegType.SGPR:
            return f"s[{self.index}:{self.index + 1}]"
        elif self.reg_type == RegType.EXEC:
            return "exec"
        elif self.reg_type == RegType.VCC:
            return "vcc"
        elif self.reg_type == RegType.SCC:
            return "scc"
        return f"unk_{self.reg_type.name}"


@dataclass(frozen=True)
class Literal64:
    """Represents a 64-bit unsigned literal constant or bitmask."""
    value: int

    def __post_init__(self) -> None:
        if not (0 <= self.value <= MASK64):
            raise ValueError(f"Literal64 value {self.value:#x} out of 64-bit range")

    @property
    def name(self) -> str:
        return f"{self.value:#018x}"


Operand64 = Union[Reg64, Literal64]


@dataclass(frozen=True)
class SaveExecInstruction:
    """Symbolic AST node representing a 64-bit saveexec instruction."""
    opcode: Opcode
    sdst: Reg64
    ssrc0: Operand64

    def __post_init__(self) -> None:
        if self.sdst.reg_type not in (RegType.SGPR, RegType.EXEC):
            raise TypeError(f"Destination sdst must be SGPR or EXEC, got {self.sdst.reg_type}")
        if isinstance(self.ssrc0, Reg64):
            if self.ssrc0.reg_type not in (RegType.SGPR, RegType.EXEC, RegType.VCC):
                raise TypeError(f"Source ssrc0 register must be SGPR, EXEC, or VCC, got {self.ssrc0.reg_type}")

    def disassemble(self) -> str:
        return f"{self.opcode.value} {self.sdst.name}, {self.ssrc0.name}"


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural scalar register file and status registers for a wavefront."""
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)
    exec_mask: int = field(default=ALL64)
    vcc: int = 0
    scc: int = 0

    def get_sgpr64(self, index: int) -> int:
        return words_to_u64(self.sgprs[index], self.sgprs[index + 1])

    def set_sgpr64(self, index: int, val: int) -> None:
        lo, hi = u64_to_words(val)
        self.sgprs[index] = lo
        self.sgprs[index + 1] = hi

    def read_reg64(self, reg: Reg64) -> int:
        if reg.reg_type == RegType.SGPR:
            return self.get_sgpr64(reg.index)
        elif reg.reg_type == RegType.EXEC:
            return self.exec_mask
        elif reg.reg_type == RegType.VCC:
            return self.vcc
        elif reg.reg_type == RegType.SCC:
            return self.scc
        raise ValueError(f"Unknown register type: {reg.reg_type}")

    def write_reg64(self, reg: Reg64, val: int) -> None:
        if reg.reg_type == RegType.SGPR:
            self.set_sgpr64(reg.index, val)
        elif reg.reg_type == RegType.EXEC:
            self.exec_mask = val & MASK64
        elif reg.reg_type == RegType.VCC:
            self.vcc = val & MASK64
        elif reg.reg_type == RegType.SCC:
            self.scc = val & 1
        else:
            raise ValueError(f"Unknown register type: {reg.reg_type}")


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 saveexec operations."""

    def resolve_operand(self, op: Operand64, state: ScalarMachineState) -> int:
        if isinstance(op, Reg64):
            return state.read_reg64(op)
        elif isinstance(op, Literal64):
            return op.value
        raise TypeError(f"Unsupported operand type: {type(op)}")

    def step(self, instr: SaveExecInstruction, state: ScalarMachineState) -> Tuple[int, int, int]:
        # Atomic read phase
        s0 = self.resolve_operand(instr.ssrc0, state)
        old_exec = state.exec_mask

        # Computation phase
        if instr.opcode == Opcode.S_NAND_SAVEEXEC_B64:
            saved, new_exec, scc = oracle_s_nand_saveexec_b64(s0, old_exec)
        elif instr.opcode == Opcode.S_NOR_SAVEEXEC_B64:
            saved, new_exec, scc = oracle_s_nor_saveexec_b64(s0, old_exec)
        else:
            raise ValueError(f"Unsupported opcode: {instr.opcode}")

        # Write-back phase
        state.write_reg64(instr.sdst, saved)
        state.exec_mask = new_exec
        state.scc = scc

        return saved, new_exec, scc


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDWaveState:
    """Architectural state for a SIMD wavefront (Wave32 or Wave64)."""
    num_lanes: int = 64
    exec_mask: Optional[int] = None
    vgprs: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)
    vcc: int = 0
    scc: int = 0

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {self.num_lanes}")
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        if self.exec_mask is None:
            self.exec_mask = (1 << self.num_lanes) - 1
        else:
            lane_mask = (1 << self.num_lanes) - 1
            self.exec_mask &= lane_mask

    def get_vgpr(self, lane: int, reg_idx: int) -> int:
        return self.vgprs[lane][reg_idx]

    def set_vgpr(self, lane: int, reg_idx: int, val: int) -> None:
        self.vgprs[lane][reg_idx] = val & MASK32

    def get_sgpr64(self, idx: int) -> int:
        return words_to_u64(self.sgprs[idx], self.sgprs[idx + 1])

    def set_sgpr64(self, idx: int, val: int) -> None:
        lo, hi = u64_to_words(val)
        self.sgprs[idx] = lo
        self.sgprs[idx + 1] = hi

    def is_lane_active(self, lane: int) -> bool:
        return bool(self.exec_mask & (1 << lane))


class SIMDExecutionEngine:
    """Simulates wave execution across divergent SIMD lanes governed by EXEC masking."""

    def resolve_operand(self, op: Operand64, state: SIMDWaveState) -> int:
        if isinstance(op, Reg64):
            if op.reg_type == RegType.SGPR:
                return state.get_sgpr64(op.index)
            elif op.reg_type == RegType.EXEC:
                return state.exec_mask
            elif op.reg_type == RegType.VCC:
                return state.vcc
            elif op.reg_type == RegType.SCC:
                return state.scc
        elif isinstance(op, Literal64):
            return op.value
        raise TypeError(f"Unsupported operand type: {type(op)}")

    def execute_saveexec(self, instr: SaveExecInstruction, state: SIMDWaveState) -> Tuple[int, int, int]:
        s0 = self.resolve_operand(instr.ssrc0, state)
        old_exec = state.exec_mask
        lane_mask = (1 << state.num_lanes) - 1

        if instr.opcode == Opcode.S_NAND_SAVEEXEC_B64:
            saved = old_exec
            new_exec = (~(s0 & old_exec)) & lane_mask
            scc = 1 if new_exec != 0 else 0
        elif instr.opcode == Opcode.S_NOR_SAVEEXEC_B64:
            saved = old_exec
            new_exec = (~(s0 | old_exec)) & lane_mask
            scc = 1 if new_exec != 0 else 0
        else:
            raise ValueError(f"Unsupported opcode: {instr.opcode}")

        if instr.sdst.reg_type == RegType.SGPR:
            state.set_sgpr64(instr.sdst.index, saved)
        elif instr.sdst.reg_type == RegType.EXEC:
            pass  # Overwritten by new_exec below

        state.exec_mask = new_exec
        state.scc = scc
        return saved, new_exec, scc

    def execute_valu_mov(
        self,
        dst_vgpr: int,
        src_val: Union[int, List[int]],
        state: SIMDWaveState,
    ) -> List[bool]:
        """Simulate masked VALU write: only lanes active in state.exec_mask are updated."""
        written = [False] * state.num_lanes
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                val = src_val[lane] if isinstance(src_val, list) else src_val
                state.set_vgpr(lane, dst_vgpr, val)
                written[lane] = True
        return written


# ==============================================================================
# Phase 1 Unit Tests: Oracle and Architectural Model
# ==============================================================================

class TestOracleAndModel(unittest.TestCase):
    """Verify bitwise mathematical properties and reference oracle definitions."""

    def test_truth_table_exhaustive(self) -> None:
        """Exhaustive 2-bit truth table verification for NAND and NOR."""
        # Bit combinations: (0,0), (0,1), (1,0), (1,1)
        # NAND truth table: ~(0&0)=1, ~(0&1)=1, ~(1&0)=1, ~(1&1)=0
        self.assertEqual(oracle_s_nand_b64(0, 0) & 1, 1)
        self.assertEqual(oracle_s_nand_b64(0, 1) & 1, 1)
        self.assertEqual(oracle_s_nand_b64(1, 0) & 1, 1)
        self.assertEqual(oracle_s_nand_b64(1, 1) & 1, 0)

        # NOR truth table: ~(0|0)=1, ~(0|1)=0, ~(1|0)=0, ~(1|1)=0
        self.assertEqual(oracle_s_nor_b64(0, 0) & 1, 1)
        self.assertEqual(oracle_s_nor_b64(0, 1) & 1, 0)
        self.assertEqual(oracle_s_nor_b64(1, 0) & 1, 0)
        self.assertEqual(oracle_s_nor_b64(1, 1) & 1, 0)

    def test_nand_identities(self) -> None:
        """Verify identity, complementation, and zero absorption for NAND."""
        val = 0x0123_4567_89AB_CDEF
        # NAND with 0 yields all ones (111...1)
        self.assertEqual(oracle_s_nand_b64(val, 0), ALL64)
        self.assertEqual(oracle_s_nand_b64(0, val), ALL64)
        # NAND with all ones yields bitwise NOT
        self.assertEqual(oracle_s_nand_b64(val, ALL64), (~val) & MASK64)
        self.assertEqual(oracle_s_nand_b64(ALL64, val), (~val) & MASK64)
        # NAND with itself yields bitwise NOT
        self.assertEqual(oracle_s_nand_b64(val, val), (~val) & MASK64)

    def test_nor_identities(self) -> None:
        """Verify identity, complementation, and zero absorption for NOR."""
        val = 0x0123_4567_89AB_CDEF
        # NOR with 0 yields bitwise NOT
        self.assertEqual(oracle_s_nor_b64(val, 0), (~val) & MASK64)
        self.assertEqual(oracle_s_nor_b64(0, val), (~val) & MASK64)
        # NOR with all ones yields 0
        self.assertEqual(oracle_s_nor_b64(val, ALL64), 0)
        self.assertEqual(oracle_s_nor_b64(ALL64, val), 0)
        # NOR with itself yields bitwise NOT
        self.assertEqual(oracle_s_nor_b64(val, val), (~val) & MASK64)

    def test_de_morgan_duality(self) -> None:
        """Verify De Morgan's laws: ~(A & B) == ~A | ~B and ~(A | B) == ~A & ~B."""
        patterns = [
            (0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555),
            (0xFFFF_0000_FFFF_0000, 0x0000_FFFF_0000_FFFF),
            (0x1234_5678_9ABC_DEF0, 0xFEDC_BA98_7654_3210),
            (0x8000_0000_0000_0000, 0x0000_0000_0000_0001),
        ]
        for a, b in patterns:
            not_a = (~a) & MASK64
            not_b = (~b) & MASK64
            # NAND duality: ~(A & B) == ~A | ~B
            self.assertEqual(oracle_s_nand_b64(a, b), (not_a | not_b) & MASK64)
            # NOR duality: ~(A | B) == ~A & ~B
            self.assertEqual(oracle_s_nor_b64(a, b), (not_a & not_b) & MASK64)

    def test_nand_saveexec_oracle_semantics(self) -> None:
        """Verify s_nand_saveexec_b64 preserves old EXEC in SDST and computes correct new EXEC and SCC."""
        old_exec = 0xFFFF_0000_FFFF_0000
        ssrc0 = 0xAAAA_AAAA_AAAA_AAAA
        saved, new_exec, scc = oracle_s_nand_saveexec_b64(ssrc0, old_exec)

        self.assertEqual(saved, old_exec)
        expected_new_exec = (~(ssrc0 & old_exec)) & MASK64
        self.assertEqual(new_exec, expected_new_exec)
        self.assertEqual(scc, 1 if expected_new_exec != 0 else 0)

        # Test case where new EXEC becomes 0:
        # ~(ssrc0 & old_exec) == 0 iff (ssrc0 & old_exec) == ALL64
        saved_zero, new_exec_zero, scc_zero = oracle_s_nand_saveexec_b64(ALL64, ALL64)
        self.assertEqual(saved_zero, ALL64)
        self.assertEqual(new_exec_zero, 0)
        self.assertEqual(scc_zero, 0)

    def test_nor_saveexec_oracle_semantics(self) -> None:
        """Verify s_nor_saveexec_b64 preserves old EXEC in SDST and computes correct new EXEC and SCC."""
        old_exec = 0x0000_0000_FFFF_FFFF
        ssrc0 = 0xFFFF_0000_0000_0000
        saved, new_exec, scc = oracle_s_nor_saveexec_b64(ssrc0, old_exec)

        self.assertEqual(saved, old_exec)
        expected_new_exec = (~(ssrc0 | old_exec)) & MASK64
        self.assertEqual(new_exec, expected_new_exec)
        self.assertEqual(scc, 1 if expected_new_exec != 0 else 0)

        # Test case where new EXEC becomes 0:
        # ~(ssrc0 | old_exec) == 0 whenever ssrc0 | old_exec == ALL64
        saved_zero, new_exec_zero, scc_zero = oracle_s_nor_saveexec_b64(0xFFFF_FFFF_0000_0000, 0x0000_0000_FFFF_FFFF)
        self.assertEqual(saved_zero, 0x0000_0000_FFFF_FFFF)
        self.assertEqual(new_exec_zero, 0)
        self.assertEqual(scc_zero, 0)

        # Test case where old_exec is 0 and ssrc0 is 0:
        # ~(0 | 0) = ALL64, so new EXEC is ALL64, SCC is 1
        saved_full, new_exec_full, scc_full = oracle_s_nor_saveexec_b64(0, 0)
        self.assertEqual(saved_full, 0)
        self.assertEqual(new_exec_full, ALL64)
        self.assertEqual(scc_full, 1)


# ==============================================================================
# Phase 2 Unit Tests: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class TestSymbolicAST(unittest.TestCase):
    """Verify symbolic AST construction, typed operand expressions, and disassembly."""

    def test_reg64_properties(self) -> None:
        s0_1 = Reg64(RegType.SGPR, 0)
        self.assertEqual(s0_1.lo_reg, 0)
        self.assertEqual(s0_1.hi_reg, 1)
        self.assertEqual(s0_1.name, "s[0:1]")

        exec_reg = Reg64(RegType.EXEC)
        self.assertEqual(exec_reg.name, "exec")

        vcc_reg = Reg64(RegType.VCC)
        self.assertEqual(vcc_reg.name, "vcc")

    def test_literal64_properties(self) -> None:
        lit = Literal64(0x1234_5678_9ABC_DEF0)
        self.assertEqual(lit.value, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(lit.name, "0x123456789abcdef0")

    def test_ast_instruction_disassembly(self) -> None:
        instr_nand = SaveExecInstruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B64,
            sdst=Reg64(RegType.SGPR, 4),
            ssrc0=Reg64(RegType.SGPR, 6),
        )
        self.assertEqual(instr_nand.disassemble(), "s_nand_saveexec_b64 s[4:5], s[6:7]")

        instr_nor_lit = SaveExecInstruction(
            opcode=Opcode.S_NOR_SAVEEXEC_B64,
            sdst=Reg64(RegType.SGPR, 0),
            ssrc0=Literal64(0x0000_FFFF_0000_FFFF),
        )
        self.assertEqual(instr_nor_lit.disassemble(), "s_nor_saveexec_b64 s[0:1], 0x0000ffff0000ffff")

        instr_nor_vcc = SaveExecInstruction(
            opcode=Opcode.S_NOR_SAVEEXEC_B64,
            sdst=Reg64(RegType.EXEC),
            ssrc0=Reg64(RegType.VCC),
        )
        self.assertEqual(instr_nor_vcc.disassemble(), "s_nor_saveexec_b64 exec, vcc")

    def test_ast_immutability_and_hash(self) -> None:
        instr1 = SaveExecInstruction(Opcode.S_NAND_SAVEEXEC_B64, Reg64(RegType.SGPR, 0), Reg64(RegType.SGPR, 2))
        instr2 = SaveExecInstruction(Opcode.S_NAND_SAVEEXEC_B64, Reg64(RegType.SGPR, 0), Reg64(RegType.SGPR, 2))
        self.assertEqual(instr1, instr2)
        s = {instr1, instr2}
        self.assertEqual(len(s), 1)


# ==============================================================================
# Phase 3 Unit Tests: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class TestScalarInterpreter(unittest.TestCase):
    """Verify deterministic execution and state transitions of ScalarInterpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_nand_saveexec_step(self) -> None:
        # Initial: EXEC = 0xFFFF_FFFF_0000_0000, s[2:3] = 0xAAAA_AAAA_AAAA_AAAA
        self.state.exec_mask = 0xFFFF_FFFF_0000_0000
        self.state.set_sgpr64(2, 0xAAAA_AAAA_AAAA_AAAA)

        instr = SaveExecInstruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B64,
            sdst=Reg64(RegType.SGPR, 4),
            ssrc0=Reg64(RegType.SGPR, 2),
        )

        saved, new_exec, scc = self.interp.step(instr, self.state)

        # Saved destination s[4:5] must receive old EXEC
        self.assertEqual(saved, 0xFFFF_FFFF_0000_0000)
        self.assertEqual(self.state.get_sgpr64(4), 0xFFFF_FFFF_0000_0000)

        # New EXEC = ~(0xAAAA_AAAA_AAAA_AAAA & 0xFFFF_FFFF_0000_0000) = ~(0xAAAA_AAAA_0000_0000)
        expected_new_exec = (~(0xAAAA_AAAA_0000_0000)) & MASK64
        self.assertEqual(new_exec, expected_new_exec)
        self.assertEqual(self.state.exec_mask, expected_new_exec)
        self.assertEqual(scc, 1)
        self.assertEqual(self.state.scc, 1)

    def test_nor_saveexec_step(self) -> None:
        # Initial: EXEC = 0x0000_0000_FFFF_FFFF, literal = 0x0000_FFFF_0000_0000
        self.state.exec_mask = 0x0000_0000_FFFF_FFFF

        instr = SaveExecInstruction(
            opcode=Opcode.S_NOR_SAVEEXEC_B64,
            sdst=Reg64(RegType.SGPR, 0),
            ssrc0=Literal64(0x0000_FFFF_0000_0000),
        )

        saved, new_exec, scc = self.interp.step(instr, self.state)

        # Saved destination s[0:1] must receive old EXEC
        self.assertEqual(saved, 0x0000_0000_FFFF_FFFF)
        self.assertEqual(self.state.get_sgpr64(0), 0x0000_0000_FFFF_FFFF)

        # New EXEC = ~(0x0000_FFFF_0000_0000 | 0x0000_0000_FFFF_FFFF) = ~(0x0000_FFFF_FFFF_FFFF)
        expected_new_exec = (~(0x0000_FFFF_FFFF_FFFF)) & MASK64
        self.assertEqual(new_exec, expected_new_exec)
        self.assertEqual(self.state.exec_mask, expected_new_exec)
        self.assertEqual(scc, 1)
        self.assertEqual(self.state.scc, 1)

    def test_step_determinism(self) -> None:
        """Run identical step repeatedly to guarantee bit-level determinism."""
        instr = SaveExecInstruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B64,
            sdst=Reg64(RegType.SGPR, 6),
            ssrc0=Literal64(0x0123_4567_89AB_CDEF),
        )

        results = []
        for _ in range(5):
            st = ScalarMachineState(exec_mask=0xF0F0_F0F0_F0F0_F0F0)
            res = self.interp.step(instr, st)
            results.append((res, st.get_sgpr64(6), st.exec_mask, st.scc))

        for r in results[1:]:
            self.assertEqual(r, results[0])

    def test_chained_saveexec_sequence(self) -> None:
        """Sequential saveexec pipeline verifying state accumulation."""
        self.state.exec_mask = ALL64
        # 1. NOR with 0x0000_0000_FFFF_FFFF -> new_exec = ~(ALL64 | ...) = 0
        instr1 = SaveExecInstruction(
            opcode=Opcode.S_NOR_SAVEEXEC_B64,
            sdst=Reg64(RegType.SGPR, 10),
            ssrc0=Literal64(0x0000_0000_FFFF_FFFF),
        )
        saved1, exec1, scc1 = self.interp.step(instr1, self.state)
        self.assertEqual(saved1, ALL64)
        self.assertEqual(exec1, 0)
        self.assertEqual(scc1, 0)

        # 2. NAND with 0x5555_5555_5555_5555 -> new_exec = ~(... & 0) = ALL64
        instr2 = SaveExecInstruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B64,
            sdst=Reg64(RegType.SGPR, 12),
            ssrc0=Literal64(0x5555_5555_5555_5555),
        )
        saved2, exec2, scc2 = self.interp.step(instr2, self.state)
        self.assertEqual(saved2, 0)
        self.assertEqual(exec2, ALL64)
        self.assertEqual(scc2, 1)


# ==============================================================================
# Phase 4 Unit Tests: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class TestSIMDDivergenceAndMasking(unittest.TestCase):
    """Verify SIMD lane masking, multi-lane divergence, and inactive lane preservation."""

    def setUp(self) -> None:
        self.engine = SIMDExecutionEngine()

    def test_wave64_nand_divergence(self) -> None:
        state = SIMDWaveState(num_lanes=64)
        # Initialize VGPR 5 to a baseline pattern across all lanes
        for lane in range(64):
            state.set_vgpr(lane, 5, 0xDEAD0000 + lane)

        # Set initial EXEC to alternating pattern
        state.exec_mask = 0x5555_5555_5555_5555  # Even lanes active
        # SSRC0 mask = 0x3333_3333_3333_3333 (pairs of lanes)
        instr = SaveExecInstruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B64,
            sdst=Reg64(RegType.SGPR, 8),
            ssrc0=Literal64(0x3333_3333_3333_3333),
        )

        saved, new_exec, scc = self.engine.execute_saveexec(instr, state)
        self.assertEqual(saved, 0x5555_5555_5555_5555)

        # Expected new EXEC: ~(0x3333_3333_3333_3333 & 0x5555_5555_5555_5555)
        # 0x3333... & 0x5555... = 0x1111_1111_1111_1111
        # ~(0x1111...) = 0xEEEE_EEEE_EEEE_EEEE
        expected_new_exec = 0xEEEE_EEEE_EEEE_EEEE
        self.assertEqual(new_exec, expected_new_exec)
        self.assertEqual(state.exec_mask, expected_new_exec)
        self.assertEqual(scc, 1)

        # Execute VALU write to VGPR 5 with new value 0xBEEF0000 + lane
        written = self.engine.execute_valu_mov(5, [0xBEEF0000 + l for l in range(64)], state)

        # Verify that only lanes enabled in new_exec received the new value!
        for lane in range(64):
            if expected_new_exec & (1 << lane):
                self.assertTrue(written[lane])
                self.assertEqual(state.get_vgpr(lane, 5), 0xBEEF0000 + lane)
            else:
                self.assertFalse(written[lane])
                self.assertEqual(state.get_vgpr(lane, 5), 0xDEAD0000 + lane)

    def test_wave64_nor_half_wave_divergence(self) -> None:
        state = SIMDWaveState(num_lanes=64)
        for lane in range(64):
            state.set_vgpr(lane, 12, 0xCAFE0000 + lane)

        # Old EXEC is lower 32 lanes only
        state.exec_mask = 0x0000_0000_FFFF_FFFF
        # SSRC0 covers upper 32 lanes
        instr = SaveExecInstruction(
            opcode=Opcode.S_NOR_SAVEEXEC_B64,
            sdst=Reg64(RegType.SGPR, 16),
            ssrc0=Literal64(0xFFFF_FFFF_0000_0000),
        )

        saved, new_exec, scc = self.engine.execute_saveexec(instr, state)
        self.assertEqual(saved, 0x0000_0000_FFFF_FFFF)
        # ~(0xFFFF_FFFF_0000_0000 | 0x0000_0000_FFFF_FFFF) = ~(ALL64) = 0
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)

        # Since all lanes turned off, VALU write should touch 0 lanes
        written = self.engine.execute_valu_mov(12, 0x12345678, state)
        self.assertFalse(any(written))
        for lane in range(64):
            self.assertEqual(state.get_vgpr(lane, 12), 0xCAFE0000 + lane)

    def test_wave32_divergence_and_masking(self) -> None:
        state = SIMDWaveState(num_lanes=32)
        for lane in range(32):
            state.set_vgpr(lane, 7, 0xAA00 + lane)

        state.exec_mask = 0x0000_FFFF  # lanes 0..15
        instr = SaveExecInstruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B64,
            sdst=Reg64(RegType.SGPR, 2),
            ssrc0=Literal64(0x00FF_00FF),
        )

        saved, new_exec, scc = self.engine.execute_saveexec(instr, state)
        self.assertEqual(saved, 0x0000_FFFF)
        # s0 & old_exec in 32-bit: 0x00FF & 0xFFFF = 0x00FF
        # ~(0x00FF) & 0xFFFFFFFF = 0xFFFF_FF00
        self.assertEqual(new_exec, 0xFFFF_FF00)
        self.assertEqual(scc, 1)

        written = self.engine.execute_valu_mov(7, 0xBBBB, state)
        for lane in range(32):
            if lane < 8:
                self.assertFalse(written[lane])
                self.assertEqual(state.get_vgpr(lane, 7), 0xAA00 + lane)
            else:
                self.assertTrue(written[lane])
                self.assertEqual(state.get_vgpr(lane, 7), 0xBBBB)


if __name__ == "__main__":
    unittest.main()
