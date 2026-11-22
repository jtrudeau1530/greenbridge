from __future__ import annotations

"""Semantic reference oracle and architectural model for RDNA2 s_andn2_saveexec_b64.

Instruction specification:
    s_andn2_saveexec_b64 SDST, SSRC0 (SOP1 format)
    
Architecture:
    - Target: AMD RDNA2 / GFX10 Scalar ALU (SALU)
    - SDST: 64-bit scalar destination register pair (e.g. s[0:1])
    - SSRC0: 64-bit scalar source operand (SGPR pair, literal constant, or inline constant)
    - Implicit operand / destination: EXEC (64-bit execution mask register)
    - Implicit destination flag: SCC (Scalar Condition Code, bit 0 of status register)

Operation:
    SDST = EXEC
    EXEC = SSRC0 & ~EXEC
    SCC = 1 if (EXEC != 0) else 0

Wavefront control-flow contract:
    - Saves current active lane execution mask to SDST.
    - Updates EXEC with lanes that are set in SSRC0 and were previously INACTIVE (~EXEC).
    - SCC reflects whether any lane in the new EXEC mask is active.
    - Used in compiler lowering of conditional branching (if-else inversion, loop exit, divergence).
"""

import re
import sys
import unittest
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def split_dwords(val64: int) -> Tuple[int, int]:
    """Split 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into 64-bit unsigned integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def ref_andn2_b64(s0: int, s1: int) -> int:
    """Bit-exact reference for 64-bit AND-NOT operation: D = S0 & (~S1)."""
    return (s0 & (~s1 & MASK64)) & MASK64


def ref_scc(val64: int) -> int:
    """RDNA2 SCC condition flag update: 1 if 64-bit result is non-zero, else 0."""
    return 1 if (val64 & MASK64) != 0 else 0


def ref_andn2_saveexec_b64(ssrc0: int, current_exec: int) -> Tuple[int, int, int]:
    """Bit-exact reference mathematical oracle for RDNA2 s_andn2_saveexec_b64.
    
    Returns:
        (saved_exec, new_exec, scc)
        - saved_exec: previous 64-bit EXEC written to SDST
        - new_exec: updated 64-bit EXEC written to EXEC register
        - scc: updated 1-bit SCC condition code (1 if new_exec != 0 else 0)
    """
    saved_exec = current_exec & MASK64
    new_exec = ref_andn2_b64(ssrc0, saved_exec)
    scc = ref_scc(new_exec)
    return saved_exec, new_exec, scc


class RegisterPair:
    """Architectural model of an aligned 64-bit scalar register pair s[N:N+1]."""

    def __init__(self, lo: int = 0, hi: int = 0) -> None:
        self.lo = lo & MASK32
        self.hi = hi & MASK32

    @classmethod
    def from_u64(cls, val: int) -> RegisterPair:
        lo, hi = split_dwords(val)
        return cls(lo, hi)

    @property
    def as_u64(self) -> int:
        return join_dwords(self.lo, self.hi)

    def set_u64(self, val: int) -> None:
        self.lo, self.hi = split_dwords(val)

    def __repr__(self) -> str:
        return f"RegisterPair(lo={self.lo:#010x}, hi={self.hi:#010x}, u64={self.as_u64:#018x})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegisterPair):
            return NotImplemented
        return self.lo == other.lo and self.hi == other.hi


class ExecRegister:
    """Architectural model of the 64-bit wave64 EXEC execution mask register."""

    def __init__(self, val: int = MASK64) -> None:
        self.val = val & MASK64

    @property
    def lo(self) -> int:
        return self.val & MASK32

    @property
    def hi(self) -> int:
        return (self.val >> 32) & MASK32

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < 64):
            raise IndexError(f"Lane index {lane} out of bounds [0, 63]")
        return bool(self.val & (1 << lane))

    def active_lane_count(self) -> int:
        return bin(self.val).count("1")

    def __repr__(self) -> str:
        return f"ExecRegister(val={self.val:#018x}, active_lanes={self.active_lane_count()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExecRegister):
            return NotImplemented
        return self.val == other.val


class SccRegister:
    """Architectural model of the 1-bit Scalar Condition Code (SCC) status flag."""

    def __init__(self, value: int = 0) -> None:
        self.value = 1 if value else 0

    def update(self, result_u64: int) -> None:
        self.value = 1 if (result_u64 & MASK64) != 0 else 0

    def __repr__(self) -> str:
        return f"SccRegister({self.value})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SccRegister):
            return NotImplemented
        return self.value == other.value


class TestSaluAndn2SaveexecB64Oracle(unittest.TestCase):
    """Unit test suite for the reference mathematical oracle and architectural models."""

    def test_1bit_truth_table_exhaustive(self):
        """Verify the complete truth table of AND-NOT across all bit lanes.
        
        Truth table for S0 & ~S1:
        S0=0, S1=0 -> 0 & ~0 = 0 & 1 = 0
        S0=0, S1=1 -> 0 & ~1 = 0 & 0 = 0
        S0=1, S1=0 -> 1 & ~0 = 1 & 1 = 1
        S0=1, S1=1 -> 1 & ~1 = 1 & 0 = 0
        """
        table = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 1,
            (1, 1): 0,
        }
        for (b0, b1), expected_bit in table.items():
            for bit_pos in range(64):
                s0 = (1 << bit_pos) if b0 else 0
                exec_val = (1 << bit_pos) if b1 else 0
                saved, new_exec, scc = ref_andn2_saveexec_b64(s0, exec_val)
                expected_new_exec = (1 << bit_pos) if expected_bit else 0
                self.assertEqual(saved, exec_val)
                self.assertEqual(new_exec, expected_new_exec)
                self.assertEqual(scc, 1 if expected_bit else 0)

    def test_de_morgan_duality(self):
        """Verify De Morgan's equivalence: A & ~B == ~(~A | B)."""
        test_patterns = [
            (0x123456789ABCDEF0, 0x0FEDCBA987654321),
            (0xAAAAAAAAAAAAAAAA, 0x5555555555555555),
            (0xFFFFFFFF00000000, 0x00000000FFFFFFFF),
            (0x0000000000000000, 0xFFFFFFFFFFFFFFFF),
            (0xFFFFFFFFFFFFFFFF, 0x0000000000000000),
        ]
        for a, b in test_patterns:
            direct = ref_andn2_b64(a, b)
            de_morgan = (~((~a & MASK64) | b)) & MASK64
            self.assertEqual(direct, de_morgan)

    def test_non_commutativity(self):
        """Verify that S0 & ~S1 != S1 & ~S0 for asymmetric inputs."""
        a = 0xF0F0F0F0F0F0F0F0
        b = 0x3333333333333333
        res_ab = ref_andn2_b64(a, b)
        res_ba = ref_andn2_b64(b, a)
        self.assertNotEqual(res_ab, res_ba)
        # Specifically:
        # A & ~B = 0xF0... & 0xCC... = 0xC0C0C0C0C0C0C0C0
        # B & ~A = 0x33... & 0x0F... = 0x0303030303030303
        self.assertEqual(res_ab, 0xC0C0C0C0C0C0C0C0)
        self.assertEqual(res_ba, 0x0303030303030303)

    def test_self_annihilation_and_nullification(self):
        """Verify edge case mathematical identities."""
        patterns = [
            0x0000000000000000,
            0xFFFFFFFFFFFFFFFF,
            0x5555555555555555,
            0x123456789ABCDEF0,
        ]
        for x in patterns:
            # A & ~A == 0
            self.assertEqual(ref_andn2_b64(x, x), 0)
            self.assertEqual(ref_scc(ref_andn2_b64(x, x)), 0)
            # 0 & ~A == 0
            self.assertEqual(ref_andn2_b64(0, x), 0)
            # A & ~0 == A
            self.assertEqual(ref_andn2_b64(x, 0), x)
            # A & ~MASK64 == 0
            self.assertEqual(ref_andn2_b64(x, MASK64), 0)

    def test_saveexec_contract(self):
        """Verify that saved_exec rigorously captures old EXEC before modification."""
        test_execs = [
            0x0000000000000000,
            0x0000000000000001,
            0x8000000000000000,
            0xFFFFFFFF00000000,
            0x00000000FFFFFFFF,
            0x5555555555555555,
            0xFFFFFFFFFFFFFFFF,
        ]
        for old_exec in test_execs:
            for ssrc0 in [0, MASK64, 0xAAAAAAAAAAAAAAAA, old_exec]:
                saved, new_exec, scc = ref_andn2_saveexec_b64(ssrc0, old_exec)
                self.assertEqual(saved, old_exec)
                self.assertEqual(new_exec, ref_andn2_b64(ssrc0, old_exec))
                self.assertEqual(scc, 1 if new_exec != 0 else 0)

    def test_scc_four_quadrants(self):
        """Verify SCC accurately distinguishes lo-dword and hi-dword non-zero states."""
        # 1. both lo and hi zero -> SCC = 0
        self.assertEqual(ref_scc(0x0000000000000000), 0)
        # 2. lo non-zero, hi zero -> SCC = 1
        self.assertEqual(ref_scc(0x0000000000000001), 1)
        self.assertEqual(ref_scc(0x0000000080000000), 1)
        # 3. lo zero, hi non-zero -> SCC = 1
        self.assertEqual(ref_scc(0x0000000100000000), 1)
        self.assertEqual(ref_scc(0x8000000000000000), 1)
        # 4. both lo and hi non-zero -> SCC = 1
        self.assertEqual(ref_scc(0x123456789ABCDEF0), 1)
        self.assertEqual(ref_scc(0xFFFFFFFFFFFFFFFF), 1)

    def test_architectural_register_pair_and_exec(self):
        """Verify RegisterPair, ExecRegister, and SccRegister modeling integrity."""
        rp = RegisterPair.from_u64(0x123456789ABCDEF0)
        self.assertEqual(rp.lo, 0x9ABCDEF0)
        self.assertEqual(rp.hi, 0x12345678)
        self.assertEqual(rp.as_u64, 0x123456789ABCDEF0)

        rp.set_u64(0xAAAAAAAAAAAAAAAA)
        self.assertEqual(rp.lo, 0xAAAAAAAA)
        self.assertEqual(rp.hi, 0xAAAAAAAA)

        exec_reg = ExecRegister(0x0000000000000003)
        self.assertTrue(exec_reg.is_lane_active(0))
        self.assertTrue(exec_reg.is_lane_active(1))
        self.assertFalse(exec_reg.is_lane_active(2))
        self.assertEqual(exec_reg.active_lane_count(), 2)

        scc = SccRegister()
        self.assertEqual(scc.value, 0)
        scc.update(0x0000000100000000)
        self.assertEqual(scc.value, 1)
        scc.update(0x0)
        self.assertEqual(scc.value, 0)


# --- Symbolic AST Definitions and Typed Expressions ---

class OperandKind(Enum):
    SGPR_PAIR = auto()     # 64-bit SGPR pair s[N:N+1]
    SGPR_SINGLE = auto()   # 32-bit SGPR sN (invalid for 64-bit SOP1, used in type checking)
    LITERAL_U64 = auto()   # 64-bit literal constant
    INLINE_CONST = auto()  # Small inline integer constant (-16 to 64)
    SPECIAL_REG = auto()   # Special 64-bit register (e.g. exec, vcc)


@dataclass
class Operand:
    """Symbolic operand for RDNA2 SALU instructions."""
    kind: OperandKind
    reg_index: Optional[int] = None
    literal_val: Optional[int] = None
    name: Optional[str] = None

    def validate(self, is_dst: bool = False) -> None:
        if is_dst:
            if self.kind != OperandKind.SGPR_PAIR:
                raise TypeError(f"Destination must be a 64-bit SGPR pair, got {self.kind}")
            if self.reg_index is None or not (0 <= self.reg_index <= 104):
                raise ValueError(f"Invalid SGPR pair index: {self.reg_index}")
            if self.reg_index % 2 != 0:
                raise ValueError(f"64-bit SGPR pair must be even-aligned, got s[{self.reg_index}:{self.reg_index+1}]")
        else:
            if self.kind == OperandKind.SGPR_PAIR:
                if self.reg_index is None or not (0 <= self.reg_index <= 104):
                    raise ValueError(f"Invalid SGPR pair index: {self.reg_index}")
                if self.reg_index % 2 != 0:
                    raise ValueError(f"64-bit SGPR pair must be even-aligned, got s[{self.reg_index}:{self.reg_index+1}]")
            elif self.kind == OperandKind.LITERAL_U64:
                if self.literal_val is None or not (0 <= self.literal_val <= MASK64):
                    raise ValueError(f"Literal 64-bit value out of range: {self.literal_val}")
            elif self.kind == OperandKind.INLINE_CONST:
                if self.literal_val is None:
                    raise ValueError("Inline constant value required")
            elif self.kind == OperandKind.SPECIAL_REG:
                if self.name not in ("exec", "vcc"):
                    raise ValueError(f"Unsupported special register: {self.name}")
            elif self.kind == OperandKind.SGPR_SINGLE:
                raise TypeError("Single 32-bit SGPR cannot be used directly as 64-bit operand")

    def to_asm(self) -> str:
        if self.kind == OperandKind.SGPR_PAIR:
            return f"s[{self.reg_index}:{self.reg_index + 1}]"
        elif self.kind == OperandKind.SGPR_SINGLE:
            return f"s{self.reg_index}"
        elif self.kind == OperandKind.LITERAL_U64:
            return f"{self.literal_val:#018x}"
        elif self.kind == OperandKind.INLINE_CONST:
            return str(self.literal_val)
        elif self.kind == OperandKind.SPECIAL_REG:
            return str(self.name)
        return "unknown"


@dataclass
class InstructionAST:
    """Symbolic Abstract Syntax Tree node for s_andn2_saveexec_b64."""
    opcode: str
    dst: Operand
    src0: Operand

    def validate(self) -> None:
        if self.opcode != "s_andn2_saveexec_b64":
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        self.dst.validate(is_dst=True)
        self.src0.validate(is_dst=False)

    def to_asm(self) -> str:
        return f"{self.opcode} {self.dst.to_asm()}, {self.src0.to_asm()}"


RE_PAIR = re.compile(r"^s\[(\d+):(\d+)\]$")
RE_SINGLE = re.compile(r"^s(\d+)$")
RE_HEX = re.compile(r"^0x[0-9a-fA-F]+$")
RE_INT = re.compile(r"^-?\d+$")


def parse_sop1_andn2_saveexec_b64(asm: str) -> InstructionAST:
    """Parse assembly string into symbolic InstructionAST."""
    line = asm.strip()
    if not line:
        raise ValueError("Empty assembly statement")
    tokens = line.split(None, 1)
    if len(tokens) != 2:
        raise ValueError(f"Invalid instruction syntax: '{line}'")
    opcode, operands_str = tokens
    if opcode != "s_andn2_saveexec_b64":
        raise ValueError(f"Unexpected opcode: '{opcode}', expected 's_andn2_saveexec_b64'")

    parts = [p.strip() for p in operands_str.split(",")]
    if len(parts) != 2:
        raise ValueError(f"s_andn2_saveexec_b64 requires exactly 2 operands (dst, src0), got {len(parts)}")

    # Parse dst
    m_dst = RE_PAIR.match(parts[0])
    if not m_dst:
        m_s = RE_SINGLE.match(parts[0])
        if m_s:
            dst = Operand(OperandKind.SGPR_SINGLE, reg_index=int(m_s.group(1)))
        else:
            raise ValueError(f"Invalid destination format: '{parts[0]}'")
    else:
        lo, hi = int(m_dst.group(1)), int(m_dst.group(2))
        if hi != lo + 1:
            raise ValueError(f"Destination SGPR pair must be contiguous s[N:N+1], got s[{lo}:{hi}]")
        dst = Operand(OperandKind.SGPR_PAIR, reg_index=lo)

    # Parse src0
    src_str = parts[1]
    m_src_pair = RE_PAIR.match(src_str)
    if m_src_pair:
        lo, hi = int(m_src_pair.group(1)), int(m_src_pair.group(2))
        if hi != lo + 1:
            raise ValueError(f"Source SGPR pair must be contiguous s[N:N+1], got s[{lo}:{hi}]")
        src0 = Operand(OperandKind.SGPR_PAIR, reg_index=lo)
    elif src_str.lower() in ("exec", "vcc"):
        src0 = Operand(OperandKind.SPECIAL_REG, name=src_str.lower())
    elif RE_HEX.match(src_str):
        val = int(src_str, 16) & MASK64
        src0 = Operand(OperandKind.LITERAL_U64, literal_val=val)
    elif RE_INT.match(src_str):
        val = int(src_str)
        if -16 <= val <= 64:
            src0 = Operand(OperandKind.INLINE_CONST, literal_val=val)
        else:
            src0 = Operand(OperandKind.LITERAL_U64, literal_val=val & MASK64)
    elif RE_SINGLE.match(src_str):
        src0 = Operand(OperandKind.SGPR_SINGLE, reg_index=int(RE_SINGLE.match(src_str).group(1)))
    else:
        raise ValueError(f"Unrecognized source operand: '{src_str}'")

    ast = InstructionAST(opcode=opcode, dst=dst, src0=src0)
    ast.validate()
    return ast


class TestSaluAndn2SaveexecB64SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_parse_valid_sgpr_operands(self):
        ast = parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[0:1], s[2:3]")
        self.assertEqual(ast.opcode, "s_andn2_saveexec_b64")
        self.assertEqual(ast.dst.kind, OperandKind.SGPR_PAIR)
        self.assertEqual(ast.dst.reg_index, 0)
        self.assertEqual(ast.src0.kind, OperandKind.SGPR_PAIR)
        self.assertEqual(ast.src0.reg_index, 2)
        self.assertEqual(ast.to_asm(), "s_andn2_saveexec_b64 s[0:1], s[2:3]")

    def test_parse_literal_and_inline_constants(self):
        ast_lit = parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[4:5], 0x123456789ABCDEF0")
        self.assertEqual(ast_lit.src0.kind, OperandKind.LITERAL_U64)
        self.assertEqual(ast_lit.src0.literal_val, 0x123456789ABCDEF0)

        ast_inline = parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[6:7], -1")
        self.assertEqual(ast_inline.src0.kind, OperandKind.INLINE_CONST)
        self.assertEqual(ast_inline.src0.literal_val, -1)

    def test_parse_special_registers(self):
        ast = parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[8:9], exec")
        self.assertEqual(ast.src0.kind, OperandKind.SPECIAL_REG)
        self.assertEqual(ast.src0.name, "exec")
        self.assertEqual(ast.to_asm(), "s_andn2_saveexec_b64 s[8:9], exec")

    def test_ast_validation_rejects_unaligned_pair(self):
        with self.assertRaises(ValueError):
            parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[1:2], s[2:3]")

    def test_ast_validation_rejects_single_sgpr_dst(self):
        with self.assertRaises(TypeError):
            parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s0, s[2:3]")

    def test_ast_validation_rejects_non_contiguous_pair(self):
        with self.assertRaises(ValueError):
            parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[0:2], s[2:3]")

    def test_ast_structural_equality_and_repr(self):
        ast1 = parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[0:1], s[2:3]")
        ast2 = parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[0:1], s[2:3]")
        self.assertEqual(ast1, ast2)
        self.assertIn("s_andn2_saveexec_b64", repr(ast1))


# --- Deterministic Scalar Instruction Interpreter ---

class ScalarSaluInterpreter:
    """Deterministic architectural interpreter for RDNA2 SALU instructions."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.num_sgprs = num_sgprs
        self.sgprs = [0] * num_sgprs
        self.exec: int = MASK64
        self.scc: int = 0
        self.cycle_count: int = 0

    def get_sgpr_u32(self, idx: int) -> int:
        if not (0 <= idx < self.num_sgprs):
            raise IndexError(f"SGPR index out of range: s{idx}")
        return self.sgprs[idx] & MASK32

    def set_sgpr_u32(self, idx: int, val: int) -> None:
        if not (0 <= idx < self.num_sgprs):
            raise IndexError(f"SGPR index out of range: s{idx}")
        self.sgprs[idx] = val & MASK32

    def get_sgpr_u64(self, idx: int) -> int:
        if not (0 <= idx <= self.num_sgprs - 2):
            raise IndexError(f"SGPR pair index out of range: s[{idx}:{idx+1}]")
        lo = self.get_sgpr_u32(idx)
        hi = self.get_sgpr_u32(idx + 1)
        return join_dwords(lo, hi)

    def set_sgpr_u64(self, idx: int, val: int) -> None:
        if not (0 <= idx <= self.num_sgprs - 2):
            raise IndexError(f"SGPR pair index out of range: s[{idx}:{idx+1}]")
        lo, hi = split_dwords(val)
        self.set_sgpr_u32(idx, lo)
        self.set_sgpr_u32(idx + 1, hi)

    def resolve_src0(self, op: Operand) -> int:
        if op.kind == OperandKind.SGPR_PAIR:
            assert op.reg_index is not None
            return self.get_sgpr_u64(op.reg_index)
        elif op.kind == OperandKind.LITERAL_U64:
            assert op.literal_val is not None
            return op.literal_val & MASK64
        elif op.kind == OperandKind.INLINE_CONST:
            assert op.literal_val is not None
            return op.literal_val & MASK64
        elif op.kind == OperandKind.SPECIAL_REG:
            if op.name == "exec":
                return self.exec & MASK64
            elif op.name == "vcc":
                # VCC can be modeled as SGPR[104:105] or independent
                return self.get_sgpr_u64(104) if self.num_sgprs >= 106 else 0
            else:
                raise ValueError(f"Unknown special register: {op.name}")
        else:
            raise TypeError(f"Cannot resolve operand of kind {op.kind}")

    def execute(self, ast: InstructionAST) -> None:
        """Deterministically execute one instruction AST."""
        ast.validate()
        if ast.opcode == "s_andn2_saveexec_b64":
            src0_val = self.resolve_src0(ast.src0)
            saved_exec = self.exec & MASK64
            new_exec = ref_andn2_b64(src0_val, saved_exec)
            scc_val = ref_scc(new_exec)

            # Destination write occurs
            assert ast.dst.reg_index is not None
            self.set_sgpr_u64(ast.dst.reg_index, saved_exec)

            # State registers updated
            self.exec = new_exec
            self.scc = scc_val
            self.cycle_count += 1
        else:
            raise NotImplementedError(f"Interpreter does not implement opcode {ast.opcode}")

    def execute_asm(self, asm_str: str) -> None:
        ast = parse_sop1_andn2_saveexec_b64(asm_str)
        self.execute(ast)


class TestSaluAndn2SaveexecB64ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def test_interp_basic_execution(self):
        interp = ScalarSaluInterpreter()
        interp.exec = 0xFFFFFFFF00000000
        interp.set_sgpr_u64(2, 0x123456789ABCDEF0)

        interp.execute_asm("s_andn2_saveexec_b64 s[0:1], s[2:3]")
        # SDST receives old EXEC
        self.assertEqual(interp.get_sgpr_u64(0), 0xFFFFFFFF00000000)
        # EXEC = s[2:3] & ~old_exec = 0x123456789ABCDEF0 & 0x00000000FFFFFFFF = 0x000000009ABCDEF0
        self.assertEqual(interp.exec, 0x000000009ABCDEF0)
        # Nonzero new EXEC -> SCC = 1
        self.assertEqual(interp.scc, 1)

    def test_interp_with_literal_constant(self):
        interp = ScalarSaluInterpreter()
        interp.exec = 0x5555555555555555
        interp.execute_asm("s_andn2_saveexec_b64 s[4:5], 0xAAAAAAAAAAAAAAAA")
        self.assertEqual(interp.get_sgpr_u64(4), 0x5555555555555555)
        # 0xAAAA... & ~0x5555... = 0xAAAA... & 0xAAAA... = 0xAAAAAAAAAAAAAAAA
        self.assertEqual(interp.exec, 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(interp.scc, 1)

    def test_interp_with_inline_constant(self):
        interp = ScalarSaluInterpreter()
        interp.exec = 0x00000000FFFFFFFF
        interp.execute_asm("s_andn2_saveexec_b64 s[6:7], -1")
        self.assertEqual(interp.get_sgpr_u64(6), 0x00000000FFFFFFFF)
        # -1 (MASK64) & ~0x00000000FFFFFFFF = 0xFFFFFFFF00000000
        self.assertEqual(interp.exec, 0xFFFFFFFF00000000)
        self.assertEqual(interp.scc, 1)

    def test_interp_with_special_reg_exec(self):
        interp = ScalarSaluInterpreter()
        interp.exec = 0x123456789ABCDEF0
        interp.execute_asm("s_andn2_saveexec_b64 s[8:9], exec")
        # SDST receives old EXEC
        self.assertEqual(interp.get_sgpr_u64(8), 0x123456789ABCDEF0)
        # EXEC = old_exec & ~old_exec = 0
        self.assertEqual(interp.exec, 0x0)
        # Zero new EXEC -> SCC = 0
        self.assertEqual(interp.scc, 0)

    def test_interp_sequential_state_flow(self):
        interp = ScalarSaluInterpreter()
        interp.exec = 0xF0F0F0F0F0F0F0F0
        interp.set_sgpr_u64(0, 0x0F0F0F0F0F0F0F0F)
        interp.set_sgpr_u64(2, 0x00FF00FF00FF00FF)

        # Step 1: saveexec with s[0:1]
        interp.execute_asm("s_andn2_saveexec_b64 s[4:5], s[0:1]")
        self.assertEqual(interp.get_sgpr_u64(4), 0xF0F0F0F0F0F0F0F0)
        self.assertEqual(interp.exec, 0x0F0F0F0F0F0F0F0F)

        # Step 2: saveexec with s[2:3]
        interp.execute_asm("s_andn2_saveexec_b64 s[6:7], s[2:3]")
        self.assertEqual(interp.get_sgpr_u64(6), 0x0F0F0F0F0F0F0F0F)
        # new_exec = 0x00FF00FF00FF00FF & ~0x0F0F0F0F0F0F0F0F = 0x00F000F000F000F0
        self.assertEqual(interp.exec, 0x00F000F000F000F0)
        self.assertEqual(interp.scc, 1)

    def test_interp_unmodified_registers_preserved(self):
        interp = ScalarSaluInterpreter()
        for i in range(interp.num_sgprs):
            interp.set_sgpr_u32(i, 0xDEAD0000 | i)

        interp.exec = 0xCAFEBABEDEADBEEF
        interp.execute_asm("s_andn2_saveexec_b64 s[10:11], 0")

        # Destination s[10:11] updated
        self.assertEqual(interp.get_sgpr_u64(10), 0xCAFEBABEDEADBEEF)

        # All other SGPRs must remain unchanged
        for i in range(interp.num_sgprs):
            if i not in (10, 11):
                self.assertEqual(interp.get_sgpr_u32(i), 0xDEAD0000 | i)


# --- SIMD Multi-Lane Divergence and EXEC Masking Engine ---

class WaveEngine:
    """Multi-lane SIMD wave execution engine modeling RDNA2 wave32 / wave64."""

    def __init__(self, wave_size: int = 64, num_vgprs: int = 256) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave size: {wave_size}")
        self.wave_size = wave_size
        self.num_vgprs = num_vgprs
        self.vgprs = [[0] * wave_size for _ in range(num_vgprs)]
        self.salu = ScalarSaluInterpreter()
        self.mask_limit = MASK32 if wave_size == 32 else MASK64
        self.salu.exec = self.mask_limit

    @property
    def exec_mask(self) -> int:
        return self.salu.exec & self.mask_limit

    @exec_mask.setter
    def exec_mask(self, val: int) -> None:
        self.salu.exec = val & self.mask_limit

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            return False
        return bool(self.exec_mask & (1 << lane))

    def get_vgpr_lane(self, reg: int, lane: int) -> int:
        return self.vgprs[reg][lane] & MASK32

    def set_vgpr_lane(self, reg: int, lane: int, val: int) -> None:
        self.vgprs[reg][lane] = val & MASK32

    def valu_add_u32(self, dst_vgpr: int, src_vgpr: int, addend: int) -> None:
        """VALU instruction: updates dst_vgpr only in currently ACTIVE lanes."""
        for lane in range(self.wave_size):
            if self.is_lane_active(lane):
                old = self.get_vgpr_lane(src_vgpr, lane)
                self.set_vgpr_lane(dst_vgpr, lane, (old + addend) & MASK32)

    def valu_set_u32(self, dst_vgpr: int, val: int) -> None:
        """VALU instruction: sets constant in currently ACTIVE lanes."""
        for lane in range(self.wave_size):
            if self.is_lane_active(lane):
                self.set_vgpr_lane(dst_vgpr, lane, val & MASK32)


class TestSaluAndn2SaveexecB64SIMDMultiLane(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_divergent_if_else_execution(self):
        """Simulate RDNA2 shader conditional branching pattern:
        if (cond) {
            v0 += 10
        } else {
            v0 += 20
        }
        Compiler lowering:
            s_and_saveexec_b64 s[saved], cond_mask
            v_add_u32 v0, v0, 10
            s_andn2_saveexec_b64 s[then], s[saved]
            v_add_u32 v0, v0, 20
            s_mov_b64 exec, s[saved]
        """
        engine = WaveEngine(wave_size=64)
        # Initially only lower 32 lanes active
        initial_exec = 0x00000000FFFFFFFF
        engine.exec_mask = initial_exec

        # Condition mask: odd lanes within active range
        cond_mask = 0xAAAAAAAAAAAAAAAA

        # Step 1: then-branch -> EXEC = old_exec & cond_mask
        # s_and_saveexec_b64 s[0:1], cond_mask
        saved_exec = engine.exec_mask
        engine.salu.set_sgpr_u64(0, saved_exec)
        engine.exec_mask = saved_exec & cond_mask
        # Then-branch VALU action
        engine.valu_add_u32(dst_vgpr=0, src_vgpr=0, addend=10)

        # Step 2: else-branch -> EXEC = s[0:1] & ~EXEC
        # s_andn2_saveexec_b64 s[2:3], s[0:1]
        engine.salu.execute_asm("s_andn2_saveexec_b64 s[2:3], s[0:1]")
        # Verify s[2:3] saved the then-branch mask
        self.assertEqual(engine.salu.get_sgpr_u64(2), initial_exec & cond_mask)
        # Verify new EXEC is else-branch mask
        expected_else_mask = initial_exec & (~cond_mask & MASK64)
        self.assertEqual(engine.exec_mask, expected_else_mask)
        # Else-branch VALU action
        engine.valu_add_u32(dst_vgpr=0, src_vgpr=0, addend=20)

        # Step 3: restore original EXEC
        engine.exec_mask = engine.salu.get_sgpr_u64(0)

        # Verification across all 64 lanes
        for lane in range(64):
            val = engine.get_vgpr_lane(0, lane)
            if lane < 32:
                if lane % 2 == 1:
                    # Odd lanes took then-branch
                    self.assertEqual(val, 10, f"Lane {lane} expected then-branch result")
                else:
                    # Even lanes took else-branch
                    self.assertEqual(val, 20, f"Lane {lane} expected else-branch result")
            else:
                # Lanes >= 32 were inactive throughout
                self.assertEqual(val, 0, f"Lane {lane} was inactive, must be 0")

    def test_simd_inactive_lane_preservation(self):
        """Verify that inactive lanes under EXEC are untouched before and after saveexec."""
        engine = WaveEngine(wave_size=64)
        # Pre-fill all lanes of v1 with distinctive values
        for lane in range(64):
            engine.set_vgpr_lane(1, lane, 0x1000 + lane)

        # Set EXEC to only lanes 0, 1, 2
        engine.exec_mask = 0x7
        # Invert EXEC against mask 0xF (lanes 0..3)
        # new_exec = 0xF & ~0x7 = 0x8 (only lane 3)
        engine.salu.set_sgpr_u64(4, 0xF)
        engine.salu.execute_asm("s_andn2_saveexec_b64 s[6:7], s[4:5]")

        # Run VALU on v1
        engine.valu_add_u32(dst_vgpr=1, src_vgpr=1, addend=0x5000)

        # Only lane 3 was modified
        self.assertEqual(engine.get_vgpr_lane(1, 3), 0x1000 + 3 + 0x5000)
        # All other lanes strictly unchanged
        for lane in range(64):
            if lane != 3:
                self.assertEqual(engine.get_vgpr_lane(1, lane), 0x1000 + lane)

    def test_simd_all_active_vs_none_active(self):
        engine = WaveEngine(wave_size=64)
        engine.exec_mask = MASK64
        # Invert against MASK64: MASK64 & ~MASK64 = 0
        engine.salu.execute_asm("s_andn2_saveexec_b64 s[0:1], -1")
        self.assertEqual(engine.exec_mask, 0)
        self.assertEqual(engine.salu.scc, 0)
        for lane in range(64):
            self.assertFalse(engine.is_lane_active(lane))

        # Now EXEC is 0, invert against MASK64: MASK64 & ~0 = MASK64
        engine.salu.execute_asm("s_andn2_saveexec_b64 s[2:3], -1")
        self.assertEqual(engine.exec_mask, MASK64)
        self.assertEqual(engine.salu.scc, 1)
        for lane in range(64):
            self.assertTrue(engine.is_lane_active(lane))

    def test_simd_wave32_masking(self):
        engine = WaveEngine(wave_size=32)
        self.assertEqual(engine.exec_mask, MASK32)
        # Upper 32 bits should never be set in wave32 mode
        engine.salu.set_sgpr_u64(0, 0xFFFFFFFF00000000)
        engine.salu.execute_asm("s_andn2_saveexec_b64 s[2:3], s[0:1]")
        # 0xFFFFFFFF00000000 & ~0x00000000FFFFFFFF = 0xFFFFFFFF00000000
        # In wave32 mode, active lanes are clamped to 32 bits
        self.assertEqual(engine.exec_mask, 0)
        self.assertFalse(engine.is_lane_active(32))


# --- Register Aliasing and In-Place Operand Preservation ---

class TestSaluAndn2SaveexecB64RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_in_place_dst_src0_aliasing_read_before_write(self):
        """Verify sdst == ssrc0: SSRC0 must be read BEFORE SDST is overwritten.
        
        Instruction: s_andn2_saveexec_b64 s[0:1], s[0:1]
        Let S0 = 0xAAAAAAAAAAAAAAAA, EXEC = 0x5555555555555555.
        Hardware behavior:
            saved_exec = EXEC = 0x5555555555555555
            new_exec = S0 & ~saved_exec = 0xAAAA... & ~0x5555... = 0xAAAAAAAAAAAAAAAA
            s[0:1] = saved_exec = 0x5555555555555555
            EXEC = new_exec = 0xAAAAAAAAAAAAAAAA
        If SDST was written before SSRC0 was read:
            new_exec would compute saved_exec & ~saved_exec = 0 (BUG!).
        """
        interp = ScalarSaluInterpreter()
        s0_val = 0xAAAAAAAAAAAAAAAA
        exec_val = 0x5555555555555555
        interp.set_sgpr_u64(0, s0_val)
        interp.exec = exec_val

        interp.execute_asm("s_andn2_saveexec_b64 s[0:1], s[0:1]")

        # Destination receives the OLD EXEC
        self.assertEqual(interp.get_sgpr_u64(0), exec_val)
        # EXEC receives (S0 & ~EXEC) using the OLD S0
        self.assertEqual(interp.exec, s0_val)
        self.assertEqual(interp.scc, 1)

    def test_in_place_aliasing_with_complement(self):
        """Verify in-place aliasing when S0 is identical to EXEC.
        
        S0 = 0x123456789ABCDEF0, EXEC = 0x123456789ABCDEF0.
        new_exec = S0 & ~EXEC = 0.
        s[2:3] gets 0x123456789ABCDEF0.
        EXEC gets 0.
        SCC gets 0.
        """
        interp = ScalarSaluInterpreter()
        val = 0x123456789ABCDEF0
        interp.set_sgpr_u64(2, val)
        interp.exec = val

        interp.execute_asm("s_andn2_saveexec_b64 s[2:3], s[2:3]")
        self.assertEqual(interp.get_sgpr_u64(2), val)
        self.assertEqual(interp.exec, 0)
        self.assertEqual(interp.scc, 0)

    def test_chained_saveexec_register_reuse(self):
        """Verify re-use of saved EXEC in multi-level nested control flow."""
        interp = ScalarSaluInterpreter()
        initial_exec = 0xFFFFFFFFFFFFFFFF
        interp.exec = initial_exec

        # First level branching: s[0:1] saves initial_exec
        interp.set_sgpr_u64(4, 0x00000000FFFFFFFF)
        interp.execute_asm("s_andn2_saveexec_b64 s[0:1], s[4:5]")
        # s[0:1] = 0xFFFFFFFFFFFFFFFF
        # EXEC = 0x00000000FFFFFFFF & ~0xFFFFFFFFFFFFFFFF = 0
        self.assertEqual(interp.get_sgpr_u64(0), initial_exec)
        self.assertEqual(interp.exec, 0)

        # Second level branching: invert using the saved EXEC from s[0:1]
        interp.execute_asm("s_andn2_saveexec_b64 s[2:3], s[0:1]")
        # s[2:3] = 0 (the old EXEC)
        # EXEC = s[0:1] & ~0 = 0xFFFFFFFFFFFFFFFF & ~0 = 0xFFFFFFFFFFFFFFFF
        self.assertEqual(interp.get_sgpr_u64(2), 0)
        self.assertEqual(interp.exec, initial_exec)
        self.assertEqual(interp.scc, 1)

    def test_high_register_indices_aliasing(self):
        """Verify aliasing at the upper boundary of SGPR file (s[104:105])."""
        interp = ScalarSaluInterpreter()
        interp.set_sgpr_u64(104, 0xCAFEBABE00000000)
        interp.exec = 0x00000000DEADBEEF

        interp.execute_asm("s_andn2_saveexec_b64 s[104:105], s[104:105]")
        self.assertEqual(interp.get_sgpr_u64(104), 0x00000000DEADBEEF)
        self.assertEqual(interp.exec, 0xCAFEBABE00000000)
        self.assertEqual(interp.scc, 1)


# --- Condition Codes, Status Flags, and Carry/Borrow Independence ---

class TestSaluAndn2SaveexecB64StatusFlags(unittest.TestCase):
    """Test suite for SCC condition codes, status flags, and carry/borrow independence."""

    def test_scc_generation_zero_vs_nonzero(self):
        """Verify SCC is strictly 0 for zero result, and 1 for any nonzero result."""
        interp = ScalarSaluInterpreter()

        # Zero result test: S0 & ~EXEC = 0
        interp.exec = 0x123456789ABCDEF0
        interp.set_sgpr_u64(0, 0x123456789ABCDEF0)
        interp.execute_asm("s_andn2_saveexec_b64 s[2:3], s[0:1]")
        self.assertEqual(interp.exec, 0)
        self.assertEqual(interp.scc, 0)

        # Nonzero result test: S0 & ~EXEC != 0
        interp.set_sgpr_u64(0, 0x1)
        interp.execute_asm("s_andn2_saveexec_b64 s[4:5], s[0:1]")
        self.assertEqual(interp.exec, 1)
        self.assertEqual(interp.scc, 1)

    def test_scc_boundary_bits(self):
        """Verify each boundary bit position (0, 31, 32, 63) properly asserts SCC."""
        boundary_bits = [0, 31, 32, 63]
        for bit in boundary_bits:
            interp = ScalarSaluInterpreter()
            interp.exec = 0  # all ~EXEC bits are 1
            mask = 1 << bit
            interp.set_sgpr_u64(0, mask)
            interp.execute_asm("s_andn2_saveexec_b64 s[2:3], s[0:1]")
            self.assertEqual(interp.exec, mask)
            self.assertEqual(interp.scc, 1, f"Bit {bit} failed to assert SCC")

    def test_carry_borrow_vcc_independence(self):
        """Verify that s_andn2_saveexec_b64 does not touch or corrupt VCC / carry flags."""
        interp = ScalarSaluInterpreter()
        # Set VCC registers s[104:105] to an identifiable canary
        vcc_canary = 0xFEEDFACECAFEBEEF
        interp.set_sgpr_u64(104, vcc_canary)

        interp.exec = 0xFFFFFFFFFFFFFFFF
        interp.set_sgpr_u64(0, 0x00000000FFFFFFFF)
        interp.execute_asm("s_andn2_saveexec_b64 s[2:3], s[0:1]")

        # VCC remains untouched
        self.assertEqual(interp.get_sgpr_u64(104), vcc_canary)

    def test_conditional_branch_emulation_scc_and_execz(self):
        """Verify SCC and EXECZ equivalence for conditional skipping of empty waves."""
        interp = ScalarSaluInterpreter()

        # Case 1: All lanes disabled -> EXEC == 0, SCC == 0
        interp.exec = MASK64
        interp.execute_asm("s_andn2_saveexec_b64 s[0:1], -1")
        # In RDNA2, compiler emits s_cbranch_execz or s_cbranch_scc0 to skip dead waves
        branch_execz = (interp.exec == 0)
        branch_scc0 = (interp.scc == 0)
        self.assertTrue(branch_execz)
        self.assertTrue(branch_scc0)
        self.assertEqual(branch_execz, branch_scc0)

        # Case 2: At least one lane enabled -> EXEC != 0, SCC == 1
        interp.execute_asm("s_andn2_saveexec_b64 s[2:3], 1")
        branch_execnz = (interp.exec != 0)
        branch_scc1 = (interp.scc == 1)
        self.assertTrue(branch_execnz)
        self.assertTrue(branch_scc1)
        self.assertEqual(branch_execnz, branch_scc1)

    def test_scc_multi_cycle_toggling(self):
        """Verify SCC state transitions cleanly between consecutive instructions."""
        interp = ScalarSaluInterpreter()
        states = [
            (MASK64, MASK64, 0, 0),  # new_exec = 0, scc = 0
            (MASK64, 0, MASK64, 1),  # new_exec = MASK64, scc = 1
            (0, MASK64, 0, 0),       # new_exec = 0, scc = 0
            (1, 0, 1, 1),            # new_exec = 1, scc = 1
        ]
        for s0, exec_in, expected_exec, expected_scc in states:
            interp.exec = exec_in
            interp.set_sgpr_u64(0, s0)
            interp.execute_asm("s_andn2_saveexec_b64 s[2:3], s[0:1]")
            self.assertEqual(interp.exec, expected_exec)
            self.assertEqual(interp.scc, expected_scc)


# --- IEEE 754 Boundary Extremes, Subnormals, and NaN Handling ---

class TestSaluAndn2SaveexecB64IEEE754Boundaries(unittest.TestCase):
    """Test suite for IEEE 754 float/double boundary masks, subnormals, and NaN handling."""

    # IEEE 754 double-precision bit patterns
    F64_POS_ZERO = 0x0000000000000000
    F64_NEG_ZERO = 0x8000000000000000
    F64_POS_INF = 0x7FF0000000000000
    F64_NEG_INF = 0xFFF0000000000000
    F64_CANONICAL_QNAN = 0x7FF8000000000000
    F64_SNAN = 0x7FF0000000000001
    F64_MIN_SUBNORMAL = 0x0000000000000001
    F64_MAX_SUBNORMAL = 0x000FFFFFFFFFFFFF
    F64_MIN_NORMAL = 0x0010000000000000
    F64_MAX_NORMAL = 0x7FEFFFFFFFFFFFFF

    def test_bitwise_transparency_on_ieee754_bit_patterns(self):
        """Verify s_andn2_saveexec_b64 preserves IEEE 754 bit patterns bit-for-bit.
        
        Shader scalar logic must not treat float bit patterns as numeric floats
        (no flushing of subnormals, no canonicalizing NaNs, no sign-bit destruction).
        """
        ieee_patterns = [
            self.F64_POS_ZERO,
            self.F64_NEG_ZERO,
            self.F64_POS_INF,
            self.F64_NEG_INF,
            self.F64_CANONICAL_QNAN,
            self.F64_SNAN,
            self.F64_MIN_SUBNORMAL,
            self.F64_MAX_SUBNORMAL,
            self.F64_MIN_NORMAL,
            self.F64_MAX_NORMAL,
        ]
        interp = ScalarSaluInterpreter()
        for p in ieee_patterns:
            # Set EXEC to 0 (all inverted bits in ~EXEC are 1)
            interp.exec = 0
            interp.set_sgpr_u64(0, p)
            interp.execute_asm("s_andn2_saveexec_b64 s[2:3], s[0:1]")
            # S0 & ~0 == S0 exactly
            self.assertEqual(interp.exec, p, f"Failed bitwise transparency on {p:#018x}")
            self.assertEqual(interp.get_sgpr_u64(2), 0)
            self.assertEqual(interp.scc, 1 if p != 0 else 0)

    def test_float_comparison_mask_divergence_filtering(self):
        """Simulate RDNA2 shader routine filtering exceptional float lanes:
        Lanes evaluate floating-point values.
        v_cmp_class_f64 produces mask of lanes that are NaN or Subnormal.
        s_andn2_saveexec_b64 s[nan_saved], s[all_lanes]
        Masks out NaN/subnormal lanes so normal calculation proceeds only on valid lanes.
        """
        engine = WaveEngine(wave_size=64)
        # All 64 lanes active initially
        engine.exec_mask = MASK64

        # Assume lanes 5, 12, 33, 47 contain NaNs or subnormals
        exceptional_lanes = {5, 12, 33, 47}
        exceptional_mask = 0
        for lane in exceptional_lanes:
            exceptional_mask |= (1 << lane)

        # Set up v0 with initial values
        for lane in range(64):
            engine.set_vgpr_lane(0, lane, 100)

        # Invert EXEC against exceptional mask:
        # EXEC_new = MASK64 & ~exceptional_mask (all lanes EXCEPT exceptional lanes)
        engine.salu.set_sgpr_u64(0, MASK64)
        # We set EXEC to exceptional_mask first (as if v_cmp wrote EXEC)
        engine.exec_mask = exceptional_mask
        # s_andn2_saveexec_b64 s[2:3], s[0:1]
        # s[2:3] = exceptional_mask
        # EXEC = MASK64 & ~exceptional_mask
        engine.salu.execute_asm("s_andn2_saveexec_b64 s[2:3], s[0:1]")

        self.assertEqual(engine.salu.get_sgpr_u64(2), exceptional_mask)
        expected_valid_mask = MASK64 & ~exceptional_mask
        self.assertEqual(engine.exec_mask, expected_valid_mask)
        self.assertEqual(engine.salu.scc, 1)

        # Normal VALU operation
        engine.valu_add_u32(dst_vgpr=0, src_vgpr=0, addend=50)

        # Verify only valid lanes were updated
        for lane in range(64):
            val = engine.get_vgpr_lane(0, lane)
            if lane in exceptional_lanes:
                self.assertEqual(val, 100, f"Exceptional lane {lane} should not have run")
            else:
                self.assertEqual(val, 150, f"Valid lane {lane} should have updated to 150")

    def test_ieee754_qnan_vs_snan_mask_inversion(self):
        """Verify distinct bit positions between QNaN and SNaN in mask logic."""
        interp = ScalarSaluInterpreter()
        # QNaN: bit 51 set; SNaN: bit 51 clear, bit 0 set
        qnan_mask = self.F64_CANONICAL_QNAN
        snan_mask = self.F64_SNAN

        interp.exec = qnan_mask
        interp.set_sgpr_u64(0, snan_mask)
        interp.execute_asm("s_andn2_saveexec_b64 s[2:3], s[0:1]")

        # new_exec = snan_mask & ~qnan_mask
        expected_new = (self.F64_SNAN & (~self.F64_CANONICAL_QNAN & MASK64)) & MASK64
        self.assertEqual(interp.exec, expected_new)
        # Since SNaN has bit 0 set and QNaN bit 0 clear:
        self.assertEqual(interp.exec & 0x1, 0x1)
        self.assertEqual(interp.scc, 1)


# --- Mutants for Mutation Testing ---

def mutant_and_instead_of_andn2(s0: int, exec_val: int) -> Tuple[int, int, int]:
    """Mutant A: computes s0 & exec_val instead of s0 & ~exec_val."""
    saved_exec = exec_val & MASK64
    new_exec = (s0 & saved_exec) & MASK64
    return saved_exec, new_exec, ref_scc(new_exec)


def mutant_inverted_operands(s0: int, exec_val: int) -> Tuple[int, int, int]:
    """Mutant B: computes ~s0 & exec_val instead of s0 & ~exec_val."""
    saved_exec = exec_val & MASK64
    new_exec = ((~s0 & MASK64) & saved_exec) & MASK64
    return saved_exec, new_exec, ref_scc(new_exec)


def mutant_no_save_exec(interp: ScalarSaluInterpreter, ast: InstructionAST) -> None:
    """Mutant C: forgets to write old EXEC to SDST register."""
    src0_val = interp.resolve_src0(ast.src0)
    saved_exec = interp.exec & MASK64
    new_exec = ref_andn2_b64(src0_val, saved_exec)
    # Skip writing saved_exec to ast.dst
    interp.exec = new_exec
    interp.scc = ref_scc(new_exec)


def mutant_write_before_read(interp: ScalarSaluInterpreter, ast: InstructionAST) -> None:
    """Mutant D: overwrites SDST before resolving SSRC0 (breaks aliased instructions)."""
    saved_exec = interp.exec & MASK64
    assert ast.dst.reg_index is not None
    interp.set_sgpr_u64(ast.dst.reg_index, saved_exec)
    # Now resolving src0 sees saved_exec if src0 aliased dst!
    src0_val = interp.resolve_src0(ast.src0)
    new_exec = ref_andn2_b64(src0_val, saved_exec)
    interp.exec = new_exec
    interp.scc = ref_scc(new_exec)


def mutant_exec_low32_only(s0: int, exec_val: int) -> Tuple[int, int, int]:
    """Mutant E: only updates low 32 bits of EXEC, leaving high 32 bits untouched."""
    saved_exec = exec_val & MASK64
    full_new = ref_andn2_b64(s0, saved_exec)
    new_exec = (saved_exec & (MASK32 << 32)) | (full_new & MASK32)
    return saved_exec, new_exec, ref_scc(new_exec)


def mutant_scc_inverted(s0: int, exec_val: int) -> Tuple[int, int, int]:
    """Mutant F: inverts SCC logic (1 if new_exec == 0 else 0)."""
    saved_exec, new_exec, scc = ref_andn2_saveexec_b64(s0, exec_val)
    return saved_exec, new_exec, 0 if scc else 1


# --- Negative Controls, Syntax Rejection, and Mutation Detection ---

class TestSaluAndn2SaveexecB64NegativeControlsAndMutations(unittest.TestCase):
    """Test suite for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_opcode(self):
        with self.assertRaises(ValueError):
            parse_sop1_andn2_saveexec_b64("s_and_saveexec_b64 s[0:1], s[2:3]")

    def test_syntax_rejection_insufficient_operands(self):
        with self.assertRaises(ValueError):
            parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[0:1]")

    def test_syntax_rejection_excess_operands(self):
        with self.assertRaises(ValueError):
            parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[0:1], s[2:3], s[4:5]")

    def test_syntax_rejection_single_sgpr_destination(self):
        with self.assertRaises(TypeError):
            parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s0, s[2:3]")

    def test_syntax_rejection_non_contiguous_registers(self):
        with self.assertRaises(ValueError):
            parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[0:2], s[2:3]")
        with self.assertRaises(ValueError):
            parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[0:1], s[2:4]")

    def test_syntax_rejection_out_of_bounds_registers(self):
        with self.assertRaises(ValueError):
            parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[106:107], s[0:1]")
        with self.assertRaises(ValueError):
            parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[0:1], s[106:107]")

    def test_syntax_rejection_malformed_tokens(self):
        with self.assertRaises(ValueError):
            parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64")
        with self.assertRaises(ValueError):
            parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[0:1], @invalid")

    def test_mutation_detection_and_instead_of_andn2(self):
        s0 = 0xF0F0F0F0F0F0F0F0
        old_exec = 0xF0F0F0F0F0F0F0F0
        # Correct: S0 & ~EXEC = 0
        _, correct_new, _ = ref_andn2_saveexec_b64(s0, old_exec)
        # Mutant: S0 & EXEC = 0xF0F0F0F0F0F0F0F0
        _, mutant_new, _ = mutant_and_instead_of_andn2(s0, old_exec)
        self.assertEqual(correct_new, 0)
        self.assertNotEqual(correct_new, mutant_new)

    def test_mutation_detection_inverted_operands(self):
        s0 = 0xFFFFFFFF00000000
        old_exec = 0x0000000000000000
        # Correct: S0 & ~0 = S0
        _, correct_new, _ = ref_andn2_saveexec_b64(s0, old_exec)
        # Mutant: ~S0 & 0 = 0
        _, mutant_new, _ = mutant_inverted_operands(s0, old_exec)
        self.assertEqual(correct_new, s0)
        self.assertNotEqual(correct_new, mutant_new)

    def test_mutation_detection_no_save_exec(self):
        interp = ScalarSaluInterpreter()
        interp.exec = 0xCAFEBABE12345678
        ast = parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[0:1], 0")
        mutant_no_save_exec(interp, ast)
        # Mutant fails to update s[0:1] with old EXEC
        self.assertNotEqual(interp.get_sgpr_u64(0), 0xCAFEBABE12345678)

    def test_mutation_detection_write_before_read(self):
        interp = ScalarSaluInterpreter()
        interp.exec = 0x5555555555555555
        interp.set_sgpr_u64(0, 0xAAAAAAAAAAAAAAAA)
        ast = parse_sop1_andn2_saveexec_b64("s_andn2_saveexec_b64 s[0:1], s[0:1]")
        mutant_write_before_read(interp, ast)
        # Mutant computes saved_exec & ~saved_exec = 0 instead of S0 & ~saved_exec
        self.assertEqual(interp.exec, 0)
        # Correct execution gives 0xAAAAAAAAAAAAAAAA
        correct_interp = ScalarSaluInterpreter()
        correct_interp.exec = 0x5555555555555555
        correct_interp.set_sgpr_u64(0, 0xAAAAAAAAAAAAAAAA)
        correct_interp.execute(ast)
        self.assertEqual(correct_interp.exec, 0xAAAAAAAAAAAAAAAA)
        self.assertNotEqual(interp.exec, correct_interp.exec)

    def test_mutation_detection_exec_low32_only(self):
        s0 = 0xFFFFFFFF00000000
        old_exec = 0x0000000000000000
        _, correct_new, _ = ref_andn2_saveexec_b64(s0, old_exec)
        _, mutant_new, _ = mutant_exec_low32_only(s0, old_exec)
        self.assertEqual(correct_new, 0xFFFFFFFF00000000)
        self.assertEqual(mutant_new, 0x0000000000000000)
        self.assertNotEqual(correct_new, mutant_new)

    def test_mutation_detection_scc_inverted(self):
        s0 = 0x0
        old_exec = 0x0
        _, _, correct_scc = ref_andn2_saveexec_b64(s0, old_exec)
        _, _, mutant_scc = mutant_scc_inverted(s0, old_exec)
        self.assertEqual(correct_scc, 0)
        self.assertEqual(mutant_scc, 1)
        self.assertNotEqual(correct_scc, mutant_scc)


if __name__ == "__main__":
    unittest.main()
