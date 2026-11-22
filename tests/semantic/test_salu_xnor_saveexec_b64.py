"""Semantic reference tests and architectural model for AMD RDNA2 s_xnor_saveexec_b64.

AMD RDNA2 (GFX10/GFX1030) Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 ISA, scalar instructions operate across 32-bit Scalar General-Purpose
  Registers (SGPRs), the 64-bit Execution Mask (EXEC / exec_lo, exec_hi), the Scalar
  Condition Code (SCC), and special system registers (VCC, M0).

  Instruction:
    s_xnor_saveexec_b64 sdst, ssrc0

  Architectural Operation:
    1. The incoming 64-bit EXEC register is saved to the destination scalar register:
         SDST = EXEC
    2. The new 64-bit EXEC mask is calculated as the bitwise equivalence (XNOR) of
       the source operand SSRC0 and the prior EXEC mask:
         NEW_EXEC = ~(SSRC0 ^ EXEC) & 0xFFFFFFFFFFFFFFFF
    3. The EXEC register is updated with the new mask:
         EXEC = NEW_EXEC
    4. The Scalar Condition Code (SCC) is updated based on whether the new EXEC mask
       has any active lanes enabled:
         SCC = 1 if (NEW_EXEC != 0) else 0

  Bitwise XNOR Truth Table:
    A | B | A XNOR B
    --+---+---------
    0 | 0 |    1
    0 | 1 |    0
    1 | 0 |    0
    1 | 1 |    1

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import random
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT64: int = 0x8000000000000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def split_dwords(val64: int) -> Tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def popcount64(val: int) -> int:
    """Count the number of set bits in an unsigned 64-bit integer."""
    return bin(val & MASK64).count("1")


def ref_xnor64(a: int, b: int) -> int:
    """Reference mathematical oracle for 64-bit bitwise XNOR: ~(a ^ b)."""
    return (~(a ^ b)) & MASK64


def ref_scc(val64: int) -> int:
    """RDNA2 SALU condition flag: SCC is 1 if 64-bit value is non-zero, else 0."""
    return 1 if (val64 & MASK64) != 0 else 0


def oracle_s_xnor_saveexec_b64(src0: int, exec_current: int) -> Tuple[int, int, int]:
    """Reference architectural oracle for s_xnor_saveexec_b64.

    Returns:
        (saved_exec, new_exec, scc)
    """
    saved_exec = exec_current & MASK64
    new_exec = ref_xnor64(src0, exec_current)
    scc = ref_scc(new_exec)
    return saved_exec, new_exec, scc


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    """Register file domain for RDNA2 ISA."""
    SGPR = auto()
    VGPR = auto()
    SPECIAL = auto()


class SpecialReg(Enum):
    """Architectural special registers relevant to SALU execution."""
    EXEC = auto()
    VCC = auto()
    SCC = auto()
    NULL = auto()


@dataclass(frozen=True)
class RegPair:
    """Represents a 64-bit aligned register pair, e.g. s[4:5] or v[0:1]."""
    index: int
    reg_type: RegType = RegType.SGPR

    def __post_init__(self) -> None:
        if self.reg_type not in (RegType.SGPR, RegType.VGPR):
            raise ValueError(f"Invalid reg_type: {self.reg_type}")
        if self.index < 0:
            raise ValueError(f"Register index must be non-negative, got {self.index}")
        if self.index % 2 != 0:
            raise ValueError(f"64-bit register pair must be even-aligned, got {self.index}")

    @property
    def lo_reg(self) -> int:
        return self.index

    @property
    def hi_reg(self) -> int:
        return self.index + 1

    @property
    def name(self) -> str:
        prefix = "s" if self.reg_type == RegType.SGPR else "v"
        return f"{prefix}[{self.index}:{self.index + 1}]"


@dataclass(frozen=True)
class SpecialRegOperand:
    """Represents an architectural special register operand like exec or vcc."""
    reg: SpecialReg

    @property
    def name(self) -> str:
        return self.reg.name.lower()


@dataclass(frozen=True)
class Literal64:
    """Represents a 64-bit constant literal scalar operand."""
    bits: int

    def __post_init__(self) -> None:
        if not (0 <= self.bits <= MASK64):
            raise ValueError(f"Literal64 out of 64-bit unsigned bounds: {self.bits}")

    @property
    def name(self) -> str:
        return f"0x{self.bits:016x}"


@dataclass(frozen=True)
class SALUSourceOperand:
    """Typed source operand for scalar arithmetic instructions."""
    source: Union[RegPair, SpecialRegOperand, Literal64]

    def disassemble(self) -> str:
        return self.source.name


@dataclass(frozen=True)
class SXnorSaveexecB64Instruction:
    """Symbolic AST node representing the s_xnor_saveexec_b64 instruction."""
    dst: Union[RegPair, SpecialRegOperand]
    src0: SALUSourceOperand

    def __post_init__(self) -> None:
        if isinstance(self.dst, RegPair):
            if self.dst.reg_type != RegType.SGPR:
                raise TypeError(f"SALU destination must be SGPR pair, got {self.dst.reg_type}")
        elif isinstance(self.dst, SpecialRegOperand):
            if self.dst.reg not in (SpecialReg.EXEC, SpecialReg.VCC):
                raise TypeError(f"Invalid special register destination: {self.dst.reg}")
        else:
            raise TypeError(f"Unsupported destination operand type: {type(self.dst)}")

    def disassemble(self) -> str:
        dst_str = self.dst.name if hasattr(self.dst, "name") else str(self.dst)
        return f"s_xnor_saveexec_b64 {dst_str}, {self.src0.disassemble()}"


def parse_salu_instruction(line: str) -> SXnorSaveexecB64Instruction:
    """Parse a single assembly line into an SXnorSaveexecB64Instruction AST node."""
    line = line.strip()
    if not line.startswith("s_xnor_saveexec_b64"):
        raise ValueError(f"Not an s_xnor_saveexec_b64 instruction: {line}")
    payload = line[len("s_xnor_saveexec_b64"):].strip()
    parts = [p.strip() for p in payload.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected 2 operands (dst, src0), got: {line}")
    dst_part, src0_part = parts

    # Parse destination
    if dst_part.lower() == "exec":
        dst: Union[RegPair, SpecialRegOperand] = SpecialRegOperand(SpecialReg.EXEC)
    elif dst_part.lower() == "vcc":
        dst = SpecialRegOperand(SpecialReg.VCC)
    elif dst_part.startswith("s[") and dst_part.endswith("]"):
        idx_str = dst_part[2:-1]
        lo, hi = map(int, idx_str.split(":"))
        if hi != lo + 1:
            raise ValueError(f"Invalid 64-bit pair range: {dst_part}")
        dst = RegPair(index=lo, reg_type=RegType.SGPR)
    else:
        raise ValueError(f"Unrecognized destination operand: {dst_part}")

    # Parse src0
    if src0_part.lower() == "exec":
        src_op = SALUSourceOperand(SpecialRegOperand(SpecialReg.EXEC))
    elif src0_part.lower() == "vcc":
        src_op = SALUSourceOperand(SpecialRegOperand(SpecialReg.VCC))
    elif src0_part.startswith("s[") and src0_part.endswith("]"):
        idx_str = src0_part[2:-1]
        lo, hi = map(int, idx_str.split(":"))
        if hi != lo + 1:
            raise ValueError(f"Invalid 64-bit pair range: {src0_part}")
        src_op = SALUSourceOperand(RegPair(index=lo, reg_type=RegType.SGPR))
    elif src0_part.startswith("0x") or src0_part.isdigit() or (src0_part.startswith("-") and src0_part[1:].isdigit()):
        val = int(src0_part, 0) & MASK64
        src_op = SALUSourceOperand(Literal64(bits=val))
    else:
        raise ValueError(f"Unrecognized src0 operand: {src0_part}")

    return SXnorSaveexecB64Instruction(dst=dst, src0=src_op)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register state for RDNA2 SALU execution."""
    sgprs: List[int] = field(default_factory=lambda: [0] * 128)
    exec_mask: int = MASK64
    scc: int = 0
    vcc: int = 0

    def get_sgpr64(self, index: int) -> int:
        """Read 64-bit value from aligned SGPR pair s[index:index+1]."""
        if index < 0 or index + 1 >= len(self.sgprs):
            raise IndexError(f"SGPR index out of range: {index}")
        return join_dwords(self.sgprs[index], self.sgprs[index + 1])

    def set_sgpr64(self, index: int, val: int) -> None:
        """Write 64-bit value into aligned SGPR pair s[index:index+1]."""
        if index < 0 or index + 1 >= len(self.sgprs):
            raise IndexError(f"SGPR index out of range: {index}")
        lo, hi = split_dwords(val)
        self.sgprs[index] = lo
        self.sgprs[index + 1] = hi


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for s_xnor_saveexec_b64."""

    def resolve_source(self, op: SALUSourceOperand, state: ScalarMachineState) -> int:
        """Resolve a 64-bit source operand against machine state."""
        if isinstance(op.source, RegPair):
            return state.get_sgpr64(op.source.index)
        elif isinstance(op.source, SpecialRegOperand):
            if op.source.reg == SpecialReg.EXEC:
                return state.exec_mask & MASK64
            elif op.source.reg == SpecialReg.VCC:
                return state.vcc & MASK64
            elif op.source.reg == SpecialReg.NULL:
                return 0
            else:
                raise ValueError(f"Unsupported source special register: {op.source.reg}")
        elif isinstance(op.source, Literal64):
            return op.source.bits & MASK64
        else:
            raise TypeError(f"Unknown source operand type: {type(op.source)}")

    def step(self, instr: SXnorSaveexecB64Instruction, state: ScalarMachineState) -> Tuple[int, int, int]:
        """Execute one s_xnor_saveexec_b64 instruction deterministically."""
        # 1. Atomic read phase
        src0_val = self.resolve_source(instr.src0, state)
        incoming_exec = state.exec_mask & MASK64

        # 2. Mathematical evaluation
        saved_exec, new_exec, scc = oracle_s_xnor_saveexec_b64(src0_val, incoming_exec)

        # 3. Write-back phase
        # Destination register write
        if isinstance(instr.dst, RegPair):
            state.set_sgpr64(instr.dst.index, saved_exec)
        elif isinstance(instr.dst, SpecialRegOperand):
            if instr.dst.reg == SpecialReg.VCC:
                state.vcc = saved_exec
            elif instr.dst.reg == SpecialReg.EXEC:
                # If destination is EXEC, EXEC will be overwritten by new_exec in final state
                pass

        # EXEC and SCC update
        state.exec_mask = new_exec
        state.scc = scc

        return saved_exec, new_exec, scc


class TestSXnorSaveexecB64ModelAndOracle(unittest.TestCase):
    """Reference mathematical oracle and architectural model tests."""

    def test_truth_table_all_bits(self) -> None:
        """Verify the 1-bit truth table of XNOR across all 4 input combinations."""
        cases = [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (1, 1, 1),
        ]
        for b0, b1, expected in cases:
            with self.subTest(b0=b0, b1=b1):
                s0 = MASK64 if b0 else 0
                s1 = MASK64 if b1 else 0
                want = MASK64 if expected else 0
                got = ref_xnor64(s0, s1)
                self.assertEqual(got, want)
                self.assertEqual(ref_scc(got), 1 if expected else 0)

    def test_commutativity_symmetry(self) -> None:
        """Verify that 64-bit XNOR is strictly commutative."""
        rng = random.Random(0x1337)
        for _ in range(50):
            a = rng.getrandbits(64)
            b = rng.getrandbits(64)
            self.assertEqual(ref_xnor64(a, b), ref_xnor64(b, a))

    def test_algebraic_identities(self) -> None:
        """Verify core boolean algebraic identities of the 64-bit XNOR operator."""
        rng = random.Random(0xCAFE)
        for _ in range(50):
            x = rng.getrandbits(64)
            # Identity: x XNOR 0xFFFFFFFFFFFFFFFF == x
            self.assertEqual(ref_xnor64(x, MASK64), x)
            # Negation: x XNOR 0 == ~x
            self.assertEqual(ref_xnor64(x, 0), (~x) & MASK64)
            # Self-equivalence: x XNOR x == 0xFFFFFFFFFFFFFFFF
            self.assertEqual(ref_xnor64(x, x), MASK64)
            # Contradiction: x XNOR (~x) == 0
            self.assertEqual(ref_xnor64(x, (~x) & MASK64), 0)
            # Involution with XOR: (x XNOR y) XNOR 0 == x ^ y
            y = rng.getrandbits(64)
            self.assertEqual(ref_xnor64(ref_xnor64(x, y), 0), (x ^ y) & MASK64)

    def test_de_morgan_and_boolean_expansion(self) -> None:
        """Verify equivalence to canonical SOP form: (A & B) | (~A & ~B)."""
        rng = random.Random(0xBEEF)
        for _ in range(50):
            a = rng.getrandbits(64)
            b = rng.getrandbits(64)
            expanded = ((a & b) | ((~a & MASK64) & (~b & MASK64))) & MASK64
            self.assertEqual(ref_xnor64(a, b), expanded)

    def test_scc_flag_truth(self) -> None:
        """Verify SCC condition code behavior on exact zero and non-zero boundaries."""
        self.assertEqual(ref_scc(0), 0)
        self.assertEqual(ref_scc(1), 1)
        self.assertEqual(ref_scc(MASK64), 1)
        self.assertEqual(ref_scc(SIGN_BIT64), 1)
        for i in range(64):
            self.assertEqual(ref_scc(1 << i), 1)

    def test_saveexec_architectural_contract(self) -> None:
        """Verify the three-part architectural return contract of s_xnor_saveexec_b64."""
        test_cases = [
            (0x0000000000000000, 0x0000000000000000, 0x0000000000000000, MASK64, 1),
            (MASK64, MASK64, MASK64, MASK64, 1),
            (0x0000000000000000, MASK64, MASK64, 0x0000000000000000, 0),
            (MASK64, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0),
            (0xAAAAAAAAAAAAAAAA, 0x5555555555555555, 0x5555555555555555, 0x0000000000000000, 0),
            (0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, MASK64, 1),
            (0x1234567890ABCDEF, 0xFEDCBA0987654321, 0xFEDCBA0987654321,
             (~(0x1234567890ABCDEF ^ 0xFEDCBA0987654321)) & MASK64, 1),
        ]
        for src0, exec_curr, want_saved, want_new, want_scc in test_cases:
            with self.subTest(src0=hex(src0), exec_curr=hex(exec_curr)):
                saved, new_exec, scc = oracle_s_xnor_saveexec_b64(src0, exec_curr)
                self.assertEqual(saved, want_saved)
                self.assertEqual(new_exec, want_new)
                self.assertEqual(scc, want_scc)

    def test_walking_bits(self) -> None:
        """Verify single walking bit across all 64 lane positions."""
        for i in range(64):
            bit = 1 << i
            # When src0 == bit and exec == bit: bit matches bit -> XNOR bit is 1, all others 1
            saved, new_exec, scc = oracle_s_xnor_saveexec_b64(bit, bit)
            self.assertEqual(saved, bit)
            self.assertEqual(new_exec, MASK64)
            self.assertEqual(scc, 1)

            # When src0 == bit and exec == ~bit: bit position differs -> XNOR bit is 0, all others 0 -> new_exec == 0
            saved, new_exec, scc = oracle_s_xnor_saveexec_b64(bit, (~bit) & MASK64)
            self.assertEqual(saved, (~bit) & MASK64)
            self.assertEqual(new_exec, 0)
            self.assertEqual(scc, 0)

    def test_fuzz_random_vectors(self) -> None:
        """Fuzz testing with 200 random 64-bit bitmasks."""
        rng = random.Random(0x9999)
        for _ in range(200):
            src = rng.getrandbits(64)
            curr = rng.getrandbits(64)
            saved, new_exec, scc = oracle_s_xnor_saveexec_b64(src, curr)
            self.assertEqual(saved, curr)
            expected_new = (~(src ^ curr)) & MASK64
            self.assertEqual(new_exec, expected_new)
            self.assertEqual(scc, 1 if expected_new != 0 else 0)


class TestSymbolicAST(unittest.TestCase):
    """Symbolic AST opcode definitions, expression typing, and parser tests."""

    def test_ast_construction_sgpr_operands(self) -> None:
        """Construct AST node with SGPR pair operands."""
        dst = RegPair(index=4, reg_type=RegType.SGPR)
        src0 = SALUSourceOperand(RegPair(index=2, reg_type=RegType.SGPR))
        instr = SXnorSaveexecB64Instruction(dst=dst, src0=src0)
        self.assertEqual(instr.dst, dst)
        self.assertEqual(instr.src0, src0)
        self.assertEqual(instr.disassemble(), "s_xnor_saveexec_b64 s[4:5], s[2:3]")

    def test_ast_construction_literal_operand(self) -> None:
        """Construct AST node with a 64-bit literal constant."""
        dst = RegPair(index=0, reg_type=RegType.SGPR)
        lit = Literal64(bits=0x123456789ABCDEF0)
        instr = SXnorSaveexecB64Instruction(dst=dst, src0=SALUSourceOperand(lit))
        self.assertEqual(instr.disassemble(), "s_xnor_saveexec_b64 s[0:1], 0x123456789abcdef0")

    def test_ast_construction_special_registers(self) -> None:
        """Construct AST node using special registers exec and vcc."""
        dst = SpecialRegOperand(SpecialReg.EXEC)
        src = SALUSourceOperand(SpecialRegOperand(SpecialReg.EXEC))
        instr = SXnorSaveexecB64Instruction(dst=dst, src0=src)
        self.assertEqual(instr.disassemble(), "s_xnor_saveexec_b64 exec, exec")

    def test_ast_immutability(self) -> None:
        """Ensure AST nodes are immutable frozen dataclasses."""
        dst = RegPair(index=8, reg_type=RegType.SGPR)
        src = SALUSourceOperand(RegPair(index=6, reg_type=RegType.SGPR))
        instr = SXnorSaveexecB64Instruction(dst=dst, src0=src)
        with self.assertRaises(Exception):
            instr.dst = RegPair(index=10, reg_type=RegType.SGPR)  # type: ignore

    def test_regpair_alignment_enforcement(self) -> None:
        """Verify that odd SGPR indices are rejected for 64-bit register pairs."""
        with self.assertRaises(ValueError):
            RegPair(index=1, reg_type=RegType.SGPR)
        with self.assertRaises(ValueError):
            RegPair(index=3, reg_type=RegType.SGPR)
        p = RegPair(index=2, reg_type=RegType.SGPR)
        self.assertEqual(p.lo_reg, 2)
        self.assertEqual(p.hi_reg, 3)

    def test_invalid_destination_rejection(self) -> None:
        """Verify that non-SGPR destinations raise TypeError."""
        vgpr_dst = RegPair(index=0, reg_type=RegType.VGPR)
        src = SALUSourceOperand(RegPair(index=2, reg_type=RegType.SGPR))
        with self.assertRaises(TypeError):
            SXnorSaveexecB64Instruction(dst=vgpr_dst, src0=src)

    def test_parser_roundtrip(self) -> None:
        """Verify that assembly parser parses and disassembles correctly."""
        lines = [
            "s_xnor_saveexec_b64 s[4:5], s[2:3]",
            "s_xnor_saveexec_b64 s[0:1], exec",
            "s_xnor_saveexec_b64 exec, s[8:9]",
            "s_xnor_saveexec_b64 s[10:11], 0x0000000000000000",
            "s_xnor_saveexec_b64 s[12:13], 0xffffffffffffffff",
        ]
        for line in lines:
            with self.subTest(line=line):
                ast_node = parse_salu_instruction(line)
                self.assertEqual(ast_node.disassemble(), line)

    def test_parser_syntax_errors(self) -> None:
        """Verify parser rejections on malformed instructions."""
        bad_lines = [
            "v_add_f64 s[4:5], s[2:3]",
            "s_xnor_saveexec_b64 s[4:5]",
            "s_xnor_saveexec_b64 s[1:2], s[2:3]",
            "s_xnor_saveexec_b64 s[4:5], s[2:3], s[0:1]",
            "s_xnor_saveexec_b64 s[4:6], s[2:3]",
        ]
        for bad in bad_lines:
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, TypeError)):
                    parse_salu_instruction(bad)


class TestScalarInterpreter(unittest.TestCase):
    """Deterministic scalar instruction interpreter tests."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_single_step_execution(self) -> None:
        """Verify basic scalar instruction execution and register updates."""
        # Setup initial state
        self.state.exec_mask = 0xFFFFFFFF00000000
        self.state.set_sgpr64(2, 0xAAAAAAAAAAAAAAAA)
        self.state.scc = 0

        instr = parse_salu_instruction("s_xnor_saveexec_b64 s[4:5], s[2:3]")
        saved, new_exec, scc = self.interp.step(instr, self.state)

        # Saved EXEC should be previous EXEC
        self.assertEqual(saved, 0xFFFFFFFF00000000)
        self.assertEqual(self.state.get_sgpr64(4), 0xFFFFFFFF00000000)

        # New EXEC should be ~(0xAAAAAAAAAAAAAAAA ^ 0xFFFFFFFF00000000)
        expected_new = (~(0xAAAAAAAAAAAAAAAA ^ 0xFFFFFFFF00000000)) & MASK64
        self.assertEqual(new_exec, expected_new)
        self.assertEqual(self.state.exec_mask, expected_new)

        # SCC should be non-zero (1)
        self.assertEqual(scc, 1)
        self.assertEqual(self.state.scc, 1)

    def test_zero_result_scc_cleared(self) -> None:
        """Verify SCC is 0 when new EXEC becomes all zeros."""
        # When ssrc0 == ~EXEC, ~(ssrc0 ^ EXEC) == ~(~EXEC ^ EXEC) == ~(-1) == 0
        incoming = 0x1234567890ABCDEF
        self.state.exec_mask = incoming
        self.state.set_sgpr64(0, (~incoming) & MASK64)
        self.state.scc = 1  # Pre-set SCC to 1 to verify transition to 0

        instr = parse_salu_instruction("s_xnor_saveexec_b64 s[6:7], s[0:1]")
        saved, new_exec, scc = self.interp.step(instr, self.state)

        self.assertEqual(saved, incoming)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)
        self.assertEqual(self.state.exec_mask, 0)
        self.assertEqual(self.state.scc, 0)

    def test_literal_operand_execution(self) -> None:
        """Verify execution with 64-bit literal constant."""
        self.state.exec_mask = MASK64
        instr = parse_salu_instruction("s_xnor_saveexec_b64 s[8:9], 0x5555555555555555")
        saved, new_exec, scc = self.interp.step(instr, self.state)

        self.assertEqual(saved, MASK64)
        self.assertEqual(self.state.get_sgpr64(8), MASK64)
        self.assertEqual(new_exec, 0x5555555555555555)
        self.assertEqual(self.state.exec_mask, 0x5555555555555555)
        self.assertEqual(scc, 1)

    def test_special_reg_vcc_source(self) -> None:
        """Verify reading from VCC special register."""
        self.state.vcc = 0x0123456789ABCDEF
        self.state.exec_mask = MASK64
        instr = parse_salu_instruction("s_xnor_saveexec_b64 s[10:11], vcc")
        saved, new_exec, scc = self.interp.step(instr, self.state)

        self.assertEqual(saved, MASK64)
        self.assertEqual(new_exec, 0x0123456789ABCDEF)
        self.assertEqual(self.state.exec_mask, 0x0123456789ABCDEF)
        self.assertEqual(scc, 1)

    def test_sequential_instruction_chaining(self) -> None:
        """Verify a chain of saveexec instructions accurately passes EXEC state."""
        self.state.exec_mask = 0x00000000FFFFFFFF
        self.state.set_sgpr64(0, 0x00000000FFFFFFFF)
        self.state.set_sgpr64(2, 0xFFFFFFFFFFFFFFFF)

        # Step 1: s[4:5] = EXEC (0x00000000FFFFFFFF); EXEC = ~(s[0:1] ^ EXEC) = ~0 = MASK64
        instr1 = parse_salu_instruction("s_xnor_saveexec_b64 s[4:5], s[0:1]")
        s1, e1, c1 = self.interp.step(instr1, self.state)
        self.assertEqual(s1, 0x00000000FFFFFFFF)
        self.assertEqual(e1, MASK64)
        self.assertEqual(c1, 1)

        # Step 2: s[6:7] = EXEC (MASK64); EXEC = ~(s[2:3] ^ EXEC) = ~(MASK64 ^ MASK64) = MASK64
        instr2 = parse_salu_instruction("s_xnor_saveexec_b64 s[6:7], s[2:3]")
        s2, e2, c2 = self.interp.step(instr2, self.state)
        self.assertEqual(s2, MASK64)
        self.assertEqual(e2, MASK64)
        self.assertEqual(c2, 1)

    def test_determinism_repeatability(self) -> None:
        """Ensure repeating same step on identical state produces identical output."""
        instr = parse_salu_instruction("s_xnor_saveexec_b64 s[0:1], s[2:3]")
        for _ in range(5):
            st = ScalarMachineState()
            st.exec_mask = 0x1122334455667788
            st.set_sgpr64(2, 0x8877665544332211)
            saved, new_exec, scc = self.interp.step(instr, st)
            self.assertEqual(saved, 0x1122334455667788)
            self.assertEqual(new_exec, (~(0x8877665544332211 ^ 0x1122334455667788)) & MASK64)
            self.assertEqual(scc, 1)


if __name__ == "__main__":
    unittest.main()
