from __future__ import annotations

"""Semantic reference oracle and architectural model for RDNA2 s_nand_saveexec_b64.

Instruction specification:
    s_nand_saveexec_b64 SDST, SSRC0 (SOP1 format)
    
Architecture:
    - Target: AMD RDNA2 / GFX10 Scalar ALU (SALU)
    - SDST: 64-bit scalar destination register pair (e.g. s[0:1])
    - SSRC0: 64-bit scalar source operand (SGPR pair, literal constant, or inline constant)
    - Implicit operand / destination: EXEC (64-bit execution mask register)
    - Implicit destination flag: SCC (Scalar Condition Code, bit 0 of status register)

Operation:
    SDST = EXEC
    EXEC = ~(SSRC0 & EXEC)
    SCC = 1 if (EXEC != 0) else 0

Wavefront control-flow contract:
    - Saves current active lane execution mask to SDST.
    - Updates EXEC with the logical NAND of SSRC0 and previous EXEC.
    - SCC reflects whether any lane in the new EXEC mask is active.
    - De Morgan duality: ~(S0 & EXEC) == (~S0 | ~EXEC).
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


def ref_nand_b64(s0: int, s1: int) -> int:
    """Bit-exact reference for 64-bit NAND operation: D = ~(S0 & S1)."""
    return (~(s0 & s1)) & MASK64


def ref_scc(val64: int) -> int:
    """RDNA2 SCC condition flag update: 1 if 64-bit result is non-zero, else 0."""
    return 1 if (val64 & MASK64) != 0 else 0


def ref_nand_saveexec_b64(ssrc0: int, current_exec: int) -> Tuple[int, int, int]:
    """Bit-exact reference mathematical oracle for RDNA2 s_nand_saveexec_b64.
    
    Returns:
        (saved_exec, new_exec, scc)
        - saved_exec: previous 64-bit EXEC written to SDST
        - new_exec: updated 64-bit EXEC written to EXEC register
        - scc: updated 1-bit SCC condition code (1 if new_exec != 0 else 0)
    """
    saved_exec = current_exec & MASK64
    new_exec = ref_nand_b64(ssrc0, saved_exec)
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


class TestSaluNandSaveexecB64Oracle(unittest.TestCase):
    """Unit test suite for the reference mathematical oracle and architectural models."""

    def test_1bit_truth_table_exhaustive(self):
        """Verify the complete truth table of NAND across all bit lanes.
        
        Truth table for ~(S0 & S1):
        S0=0, S1=0 -> ~(0 & 0) = ~0 = 1
        S0=0, S1=1 -> ~(0 & 1) = ~0 = 1
        S0=1, S1=0 -> ~(1 & 0) = ~0 = 1
        S0=1, S1=1 -> ~(1 & 1) = ~1 = 0
        """
        table = {
            (0, 0): 1,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 0,
        }
        for (b0, b1), expected_bit in table.items():
            for bit_pos in range(64):
                s0 = (1 << bit_pos) if b0 else 0
                exec_val = (1 << bit_pos) if b1 else 0
                saved, new_exec, scc = ref_nand_saveexec_b64(s0, exec_val)
                # For bit positions other than bit_pos, s0=0, exec=0 -> ~(0&0) = 1
                # To isolate 1-bit behavior, check the specific bit_pos:
                got_bit = (new_exec >> bit_pos) & 1
                self.assertEqual(saved, exec_val)
                self.assertEqual(got_bit, expected_bit)
                self.assertEqual(scc, 1 if new_exec != 0 else 0)

    def test_de_morgan_duality(self):
        """Verify De Morgan's equivalence: ~(A & B) == (~A | ~B)."""
        test_patterns = [
            (0x123456789ABCDEF0, 0x0FEDCBA987654321),
            (0xAAAAAAAAAAAAAAAA, 0x5555555555555555),
            (0xFFFFFFFF00000000, 0x00000000FFFFFFFF),
            (0x0000000000000000, 0xFFFFFFFFFFFFFFFF),
            (0xFFFFFFFFFFFFFFFF, 0x0000000000000000),
        ]
        for a, b in test_patterns:
            direct = ref_nand_b64(a, b)
            de_morgan = ((~a & MASK64) | (~b & MASK64)) & MASK64
            self.assertEqual(direct, de_morgan)

    def test_commutativity(self):
        """Verify that ~(S0 & S1) == ~(S1 & S0) (NAND is commutative)."""
        a = 0xF0F0F0F0F0F0F0F0
        b = 0x3333333333333333
        res_ab = ref_nand_b64(a, b)
        res_ba = ref_nand_b64(b, a)
        self.assertEqual(res_ab, res_ba)

    def test_boundary_identities(self):
        """Verify edge case mathematical identities."""
        patterns = [
            0x0000000000000000,
            0xFFFFFFFFFFFFFFFF,
            0x5555555555555555,
            0x123456789ABCDEF0,
        ]
        for x in patterns:
            # ~(x & 0) == ~0 == MASK64
            self.assertEqual(ref_nand_b64(x, 0), MASK64)
            # ~(x & MASK64) == ~x
            self.assertEqual(ref_nand_b64(x, MASK64), (~x) & MASK64)
            # ~(x & x) == ~x
            self.assertEqual(ref_nand_b64(x, x), (~x) & MASK64)

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
                saved, new_exec, scc = ref_nand_saveexec_b64(ssrc0, old_exec)
                self.assertEqual(saved, old_exec)
                self.assertEqual(new_exec, ref_nand_b64(ssrc0, old_exec))
                self.assertEqual(scc, 1 if new_exec != 0 else 0)

    def test_scc_four_quadrants(self):
        """Verify SCC accurately reflects non-zero state."""
        # 1. zero result: only when S0 == MASK64 and EXEC == MASK64
        self.assertEqual(ref_scc(0x0000000000000000), 0)
        # 2. non-zero in lo, hi, or both
        self.assertEqual(ref_scc(0x0000000000000001), 1)
        self.assertEqual(ref_scc(0x0000000100000000), 1)
        self.assertEqual(ref_scc(0xFFFFFFFFFFFFFFFF), 1)

    def test_architectural_models(self):
        """Verify RegisterPair, ExecRegister, and SccRegister integrity."""
        rp = RegisterPair.from_u64(0x123456789ABCDEF0)
        self.assertEqual(rp.as_u64, 0x123456789ABCDEF0)

        exec_reg = ExecRegister(0x3)
        self.assertTrue(exec_reg.is_lane_active(0))
        self.assertTrue(exec_reg.is_lane_active(1))
        self.assertFalse(exec_reg.is_lane_active(2))

        scc = SccRegister()
        scc.update(0x1)
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
    """Symbolic Abstract Syntax Tree node for s_nand_saveexec_b64."""
    opcode: str
    dst: Operand
    src0: Operand

    def validate(self) -> None:
        if self.opcode != "s_nand_saveexec_b64":
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        self.dst.validate(is_dst=True)
        self.src0.validate(is_dst=False)

    def to_asm(self) -> str:
        return f"{self.opcode} {self.dst.to_asm()}, {self.src0.to_asm()}"


RE_PAIR = re.compile(r"^s\[(\d+):(\d+)\]$")
RE_SINGLE = re.compile(r"^s(\d+)$")
RE_HEX = re.compile(r"^0x[0-9a-fA-F]+$")
RE_INT = re.compile(r"^-?\d+$")


def parse_sop1_nand_saveexec_b64(asm: str) -> InstructionAST:
    """Parse assembly string into symbolic InstructionAST."""
    line = asm.strip()
    if not line:
        raise ValueError("Empty assembly statement")
    tokens = line.split(None, 1)
    if len(tokens) != 2:
        raise ValueError(f"Invalid instruction syntax: '{line}'")
    opcode, operands_str = tokens
    if opcode != "s_nand_saveexec_b64":
        raise ValueError(f"Unexpected opcode: '{opcode}', expected 's_nand_saveexec_b64'")

    parts = [p.strip() for p in operands_str.split(",")]
    if len(parts) != 2:
        raise ValueError(f"s_nand_saveexec_b64 requires exactly 2 operands (dst, src0), got {len(parts)}")

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


class TestSaluNandSaveexecB64SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_parse_valid_sgpr_operands(self):
        ast = parse_sop1_nand_saveexec_b64("s_nand_saveexec_b64 s[0:1], s[2:3]")
        self.assertEqual(ast.opcode, "s_nand_saveexec_b64")
        self.assertEqual(ast.dst.kind, OperandKind.SGPR_PAIR)
        self.assertEqual(ast.dst.reg_index, 0)
        self.assertEqual(ast.src0.kind, OperandKind.SGPR_PAIR)
        self.assertEqual(ast.src0.reg_index, 2)
        self.assertEqual(ast.to_asm(), "s_nand_saveexec_b64 s[0:1], s[2:3]")

    def test_parse_literal_and_inline_constants(self):
        ast_lit = parse_sop1_nand_saveexec_b64("s_nand_saveexec_b64 s[4:5], 0x123456789ABCDEF0")
        self.assertEqual(ast_lit.src0.kind, OperandKind.LITERAL_U64)
        self.assertEqual(ast_lit.src0.literal_val, 0x123456789ABCDEF0)

        ast_inline = parse_sop1_nand_saveexec_b64("s_nand_saveexec_b64 s[6:7], -1")
        self.assertEqual(ast_inline.src0.kind, OperandKind.INLINE_CONST)
        self.assertEqual(ast_inline.src0.literal_val, -1)

    def test_parse_special_registers(self):
        ast = parse_sop1_nand_saveexec_b64("s_nand_saveexec_b64 s[8:9], exec")
        self.assertEqual(ast.src0.kind, OperandKind.SPECIAL_REG)
        self.assertEqual(ast.src0.name, "exec")
        self.assertEqual(ast.to_asm(), "s_nand_saveexec_b64 s[8:9], exec")

    def test_ast_validation_rejects_unaligned_pair(self):
        with self.assertRaises(ValueError):
            parse_sop1_nand_saveexec_b64("s_nand_saveexec_b64 s[1:2], s[2:3]")

    def test_ast_validation_rejects_single_sgpr_dst(self):
        with self.assertRaises(TypeError):
            parse_sop1_nand_saveexec_b64("s_nand_saveexec_b64 s0, s[2:3]")

    def test_ast_validation_rejects_non_contiguous_pair(self):
        with self.assertRaises(ValueError):
            parse_sop1_nand_saveexec_b64("s_nand_saveexec_b64 s[0:2], s[2:3]")

    def test_ast_structural_equality_and_repr(self):
        ast1 = parse_sop1_nand_saveexec_b64("s_nand_saveexec_b64 s[0:1], s[2:3]")
        ast2 = parse_sop1_nand_saveexec_b64("s_nand_saveexec_b64 s[0:1], s[2:3]")
        self.assertEqual(ast1, ast2)
        self.assertIn("s_nand_saveexec_b64", repr(ast1))


if __name__ == "__main__":
    unittest.main()
