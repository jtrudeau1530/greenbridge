from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_dot8_int opcodes.

This module provides an architectural reference model and semantic test suite for RDNA2
VALU packed 4-bit integer dot product instructions:
    - v_dot8_i32_i4 (VOP3/VOP3P): Signed 4-bit integer 8-vector dot product:
        D.i32 = S2.i32 + sum(S0.i4[i] * S1.i4[i] for i in 0..7) mod 2^32
    - v_dot8_u32_u4 (VOP3/VOP3P): Unsigned 4-bit integer 8-vector dot product:
        D.u32 = S2.u32 + sum(S0.u4[i] * S1.u4[i] for i in 0..7) mod 2^32

ISA Specifications (RDNA2 / GFX10.3+):
    - Each 32-bit dword S0 and S1 contains eight packed 4-bit nibbles:
        Nibble 0: bits [3:0]
        Nibble 1: bits [7:4]
        Nibble 2: bits [11:8]
        Nibble 3: bits [15:12]
        Nibble 4: bits [19:16]
        Nibble 5: bits [23:20]
        Nibble 6: bits [27:24]
        Nibble 7: bits [31:28]
    - In v_dot8_i32_i4, each nibble is treated as signed 2's complement int4 in [-8, +7].
      Each component product S0.i4[i] * S1.i4[i] is in [-56, +64].
      The eight products sum to [-448, +512], which is added to 32-bit accumulator S2
      and wrapped modulo 2^32 (or clamped to [-2^31, 2^31 - 1] if clamp is asserted).
    - In v_dot8_u32_u4, each nibble is treated as unsigned uint4 in [0, 15].
      Each component product S0.u4[i] * S1.u4[i] is in [0, 225].
      The eight products sum to [0, 1800], which is added to 32-bit accumulator S2
      and wrapped modulo 2^32 (or clamped to [0, 0xFFFFFFFF] if clamp is asserted).

Standard library Python only; zero third-party dependencies.
Python 3.9+ compatible.
"""

import struct
from typing import Tuple
import unittest

# Bitmask and architectural boundary constants
MASK4: int = 0xF
MASK32: int = 0xFFFF_FFFF
SIGN4_BIT: int = 0x8
INT32_MIN: int = -0x8000_0000
INT32_MAX: int = 0x7FFF_FFFF
UINT32_MAX: int = 0xFFFF_FFFF


def unpack_u4(dword: int) -> Tuple[int, int, int, int, int, int, int, int]:
    """Unpack a 32-bit dword into eight unsigned 4-bit integers (nibble 0 to 7)."""
    val = dword & MASK32
    return (
        val & MASK4,
        (val >> 4) & MASK4,
        (val >> 8) & MASK4,
        (val >> 12) & MASK4,
        (val >> 16) & MASK4,
        (val >> 20) & MASK4,
        (val >> 24) & MASK4,
        (val >> 28) & MASK4,
    )


def pack_u4(n0: int, n1: int, n2: int, n3: int, n4: int, n5: int, n6: int, n7: int) -> int:
    """Pack eight unsigned 4-bit integers into a 32-bit dword."""
    return (
        (n0 & MASK4)
        | ((n1 & MASK4) << 4)
        | ((n2 & MASK4) << 8)
        | ((n3 & MASK4) << 12)
        | ((n4 & MASK4) << 16)
        | ((n5 & MASK4) << 20)
        | ((n6 & MASK4) << 24)
        | ((n7 & MASK4) << 28)
    ) & MASK32


def unpack_i4(dword: int) -> Tuple[int, int, int, int, int, int, int, int]:
    """Unpack a 32-bit dword into eight signed 4-bit integers (-8 to +7)."""
    raw_nibbles = unpack_u4(dword)
    return tuple((n - 16) if (n & SIGN4_BIT) else n for n in raw_nibbles)  # type: ignore[return-value]


def pack_i4(n0: int, n1: int, n2: int, n3: int, n4: int, n5: int, n6: int, n7: int) -> int:
    """Pack eight signed 4-bit integers into a 32-bit dword."""
    return pack_u4(
        n0 & MASK4,
        n1 & MASK4,
        n2 & MASK4,
        n3 & MASK4,
        n4 & MASK4,
        n5 & MASK4,
        n6 & MASK4,
        n7 & MASK4,
    )


def ref_dot8_u32_u4(s0: int, s1: int, s2: int = 0, *, clamp: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 unsigned 4-bit dot product (v_dot8_u32_u4).

    Calculates:
        D = (s2 + sum(s0[i] * s1[i] for i in 0..7))
    Modulo 2^32 without clamp; saturated to 0xFFFFFFFF if clamp is set.
    Returns unsigned 32-bit integer in [0, 0xFFFFFFFF].
    """
    n0 = unpack_u4(s0)
    n1 = unpack_u4(s1)
    prod_sum = sum(x * y for x, y in zip(n0, n1))
    acc = s2 & MASK32
    total = acc + prod_sum

    if clamp:
        return min(total, UINT32_MAX)
    return total & MASK32


def ref_dot8_i32_i4(s0: int, s1: int, s2: int = 0, *, clamp: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 signed 4-bit dot product (v_dot8_i32_i4).

    Calculates:
        D = (s2 + sum(s0.i4[i] * s1.i4[i] for i in 0..7))
    where S2 is interpreted as signed 32-bit integer.
    Modulo 2^32 without clamp; saturated to [-2^31, 2^31 - 1] if clamp is set.
    Returns unsigned 32-bit bit pattern representing the resulting dword.
    """
    n0 = unpack_i4(s0)
    n1 = unpack_i4(s1)
    prod_sum = sum(x * y for x, y in zip(n0, n1))

    # Interpret s2 as signed 32-bit integer
    s2_u32 = s2 & MASK32
    s2_i32 = (s2_u32 - 0x1_0000_0000) if (s2_u32 & 0x8000_0000) else s2_u32

    total_i64 = s2_i32 + prod_sum

    if clamp:
        clamped = max(INT32_MIN, min(total_i64, INT32_MAX))
        return clamped & MASK32
    return total_i64 & MASK32


class TestValuDot8IntMathematicalOracle(unittest.TestCase):
    """Architectural reference mathematical oracle unit tests for v_dot8_int."""

    def test_u4_basic_vectors(self) -> None:
        # Sum 0*1 + 1*2 + 2*3 + 3*4 + 4*5 + 5*6 + 6*7 + 7*8
        # = 0 + 2 + 6 + 12 + 20 + 30 + 42 + 56 = 168
        s0 = pack_u4(0, 1, 2, 3, 4, 5, 6, 7)
        s1 = pack_u4(1, 2, 3, 4, 5, 6, 7, 8)
        self.assertEqual(ref_dot8_u32_u4(s0, s1, 0), 168)
        # With accumulator = 500 -> 668
        self.assertEqual(ref_dot8_u32_u4(s0, s1, 500), 668)

    def test_u4_zero_operands(self) -> None:
        s0 = 0x0000_0000
        s1 = 0x1234_5678
        self.assertEqual(ref_dot8_u32_u4(s0, s1, 0x1234_ABCD), 0x1234_ABCD)
        self.assertEqual(ref_dot8_u32_u4(s1, s0, 0xCAFE_BABE), 0xCAFE_BABE)

    def test_u4_maximum_product(self) -> None:
        # 8 * (15 * 15) = 8 * 225 = 1800
        s0 = 0xFFFF_FFFF
        s1 = 0xFFFF_FFFF
        self.assertEqual(ref_dot8_u32_u4(s0, s1, 0), 1800)
        self.assertEqual(ref_dot8_u32_u4(s0, s1, 200), 2000)

    def test_u4_overflow_wrap_around(self) -> None:
        # Wrap around 2^32: (0xFFFFFFFF - 1800 + 1) + 1800 = 0
        s0 = 0xFFFF_FFFF
        s1 = 0xFFFF_FFFF
        acc = (MASK32 - 1800 + 1) & MASK32
        self.assertEqual(ref_dot8_u32_u4(s0, s1, acc), 0)

        # Overflow by 1
        acc_plus1 = (acc + 1) & MASK32
        self.assertEqual(ref_dot8_u32_u4(s0, s1, acc_plus1), 1)

    def test_u4_clamp_saturation(self) -> None:
        s0 = 0xFFFF_FFFF
        s1 = 0xFFFF_FFFF
        acc = 0xFFFF_FFF0
        # Without clamp, wraps around
        wrapped = ref_dot8_u32_u4(s0, s1, acc, clamp=False)
        self.assertLess(wrapped, 0xFFFF_FFF0)
        # With clamp, saturates to UINT32_MAX
        clamped = ref_dot8_u32_u4(s0, s1, acc, clamp=True)
        self.assertEqual(clamped, UINT32_MAX)

    def test_i4_basic_vectors(self) -> None:
        s0 = pack_i4(1, 2, 3, 4, -1, -2, -3, -4)
        s1 = pack_i4(1, 1, 1, 1, 1, 1, 1, 1)
        # Sum: 1+2+3+4 + (-1)+(-2)+(-3)+(-4) = 0
        self.assertEqual(ref_dot8_i32_i4(s0, s1, 0), 0)
        self.assertEqual(ref_dot8_i32_i4(s0, s1, 77), 77)

    def test_i4_negative_operands(self) -> None:
        # All negative: [-1, -2, -3, -4, -1, -2, -3, -4]
        s0 = pack_i4(-1, -2, -3, -4, -1, -2, -3, -4)
        s1 = pack_i4(2, 2, 2, 2, 2, 2, 2, 2)
        # Sum = 2 * (-20) = -40
        expected = (-40) & MASK32
        self.assertEqual(ref_dot8_i32_i4(s0, s1, 0), expected)

    def test_i4_extremes(self) -> None:
        # Max positive product: 8 * ((-8) * (-8)) = 8 * 64 = +512
        s_min = pack_i4(-8, -8, -8, -8, -8, -8, -8, -8)
        self.assertEqual(ref_dot8_i32_i4(s_min, s_min, 0), 512)

        # Max negative product: 8 * ((-8) * 7) = 8 * (-56) = -448
        s_max = pack_i4(7, 7, 7, 7, 7, 7, 7, 7)
        expected_neg = (-448) & MASK32
        self.assertEqual(ref_dot8_i32_i4(s_min, s_max, 0), expected_neg)

    def test_i4_signed_overflow_and_clamp(self) -> None:
        s_min = pack_i4(-8, -8, -8, -8, -8, -8, -8, -8)
        # Adding +512 to INT32_MAX without clamp wraps to negative
        res_wrap = ref_dot8_i32_i4(s_min, s_min, INT32_MAX, clamp=False)
        self.assertTrue(bool(res_wrap & 0x8000_0000))

        # With clamp, saturates to INT32_MAX
        res_clamp_pos = ref_dot8_i32_i4(s_min, s_min, INT32_MAX, clamp=True)
        self.assertEqual(res_clamp_pos, INT32_MAX & MASK32)

        # Negative saturation: INT32_MIN + (-448) saturates to INT32_MIN
        s_max = pack_i4(7, 7, 7, 7, 7, 7, 7, 7)
        res_clamp_neg = ref_dot8_i32_i4(s_min, s_max, INT32_MIN, clamp=True)
        self.assertEqual(res_clamp_neg, INT32_MIN & MASK32)

    def test_divergence_between_signed_and_unsigned(self) -> None:
        # Nibble 0xF is 15 in unsigned; -1 in signed
        # S0 = [15, 0, ...], S1 = [1, 0, ...]
        s0 = pack_u4(15, 0, 0, 0, 0, 0, 0, 0)
        s1 = pack_u4(1, 0, 0, 0, 0, 0, 0, 0)
        u_res = ref_dot8_u32_u4(s0, s1, 0)
        i_res = ref_dot8_i32_i4(s0, s1, 0)
        self.assertEqual(u_res, 15)
        self.assertEqual(i_res, 0xFFFF_FFFF)  # -1 in 32-bit dword
        self.assertNotEqual(u_res, i_res)


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from enum import Enum, auto
import re
from typing import Optional, Union, List


class Opcode(Enum):
    """VALU opcode enumerations for packed 4-bit integer dot products."""
    V_DOT8_I32_I4 = "v_dot8_i32_i4"
    V_DOT8_U32_U4 = "v_dot8_u32_u4"
    V_MOV_B32 = "v_mov_b32_e32"


class OperandKind(Enum):
    """Operand storage classification."""
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()
    INLINE_INT = auto()


class TypeKind(Enum):
    """Data representation types for operands and expressions."""
    I32 = "i32"
    U32 = "u32"
    I4 = "i4"
    U4 = "u4"


@dataclass(frozen=True)
class ASTOperand:
    """Symbolic instruction operand representation."""
    kind: OperandKind
    reg_num: int = 0
    literal_value: int = 0
    type_kind: TypeKind = TypeKind.U32

    def to_asm(self) -> str:
        if self.kind == OperandKind.VGPR:
            return f"v{self.reg_num}"
        elif self.kind == OperandKind.SGPR:
            return f"s{self.reg_num}"
        elif self.kind == OperandKind.LITERAL:
            if self.literal_value >= 0x1000:
                return f"0x{self.literal_value:X}"
            return str(self.literal_value)
        elif self.kind == OperandKind.INLINE_INT:
            return str(self.literal_value)
        return "?"


@dataclass(frozen=True)
class ASTInstruction:
    """Symbolic instruction AST representation."""
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand
    src2: Optional[ASTOperand] = None
    clamp: bool = False

    def to_asm(self) -> str:
        parts = [self.opcode.value, self.dst.to_asm(), self.src0.to_asm(), self.src1.to_asm()]
        if self.src2 is not None:
            parts.append(self.src2.to_asm())
        res = f"{parts[0]} {', '.join(parts[1:])}"
        if self.clamp:
            res += " clamp"
        return res


@dataclass
class TypedExpr:
    """Base class for typed symbolic expressions."""
    type_kind: TypeKind


@dataclass
class VariableExpr(TypedExpr):
    """Symbolic variable representing a register source."""
    name: str


@dataclass
class LiteralExpr(TypedExpr):
    """Symbolic literal value."""
    val: int


@dataclass
class Dot8ComponentProductExpr(TypedExpr):
    """Component product of nibble i: src0.nibble[i] * src1.nibble[i]."""
    nibble_index: int
    left: TypedExpr
    right: TypedExpr


@dataclass
class Dot8Expr(TypedExpr):
    """Full dot8 expression: accumulator + sum(component products), with optional clamp."""
    opcode: Opcode
    src0: TypedExpr
    src1: TypedExpr
    acc: TypedExpr
    clamp: bool = False


def parse_operand(token: str, default_type: TypeKind = TypeKind.U32) -> ASTOperand:
    """Parse an assembly operand token into an ASTOperand."""
    token = token.strip()
    if token.startswith("v") and token[1:].isdigit():
        return ASTOperand(kind=OperandKind.VGPR, reg_num=int(token[1:]), type_kind=default_type)
    elif token.startswith("s") and token[1:].isdigit():
        return ASTOperand(kind=OperandKind.SGPR, reg_num=int(token[1:]), type_kind=default_type)
    else:
        val = int(token, 0)
        if -16 <= val <= 64:
            return ASTOperand(kind=OperandKind.INLINE_INT, literal_value=val, type_kind=default_type)
        return ASTOperand(kind=OperandKind.LITERAL, literal_value=val, type_kind=default_type)


def parse_instruction(line: str) -> ASTInstruction:
    """Parse a single assembly instruction string into an ASTInstruction."""
    line = line.strip()
    if not line or line.startswith("//") or line.startswith(";"):
        raise ValueError(f"Empty or comment line cannot be parsed: '{line}'")

    clamp = "clamp" in line.split()
    line_clean = re.sub(r"\bclamp\b", "", line).strip()

    tokens = line_clean.split(None, 1)
    op_name = tokens[0].lower()
    operands_str = tokens[1] if len(tokens) > 1 else ""

    opcode = None
    for op in Opcode:
        if op.value == op_name:
            opcode = op
            break
    if opcode is None:
        raise NotImplementedError(f"Unsupported or unrecognized opcode: '{op_name}'")

    op_parts = [p.strip() for p in operands_str.split(",") if p.strip()]

    if opcode == Opcode.V_MOV_B32:
        if len(op_parts) != 2:
            raise ValueError(f"v_mov_b32 expects 2 operands, got {len(op_parts)}")
        dst = parse_operand(op_parts[0])
        src0 = parse_operand(op_parts[1])
        return ASTInstruction(opcode=opcode, dst=dst, src0=src0, src1=src0, clamp=clamp)

    if opcode == Opcode.V_DOT8_I32_I4:
        elem_type = TypeKind.I4
        acc_type = TypeKind.I32
    else:
        elem_type = TypeKind.U4
        acc_type = TypeKind.U32

    if len(op_parts) == 3:
        dst = parse_operand(op_parts[0], acc_type)
        src0 = parse_operand(op_parts[1], elem_type)
        src1 = parse_operand(op_parts[2], elem_type)
        src2 = ASTOperand(kind=OperandKind.INLINE_INT, literal_value=0, type_kind=acc_type)
        return ASTInstruction(opcode=opcode, dst=dst, src0=src0, src1=src1, src2=src2, clamp=clamp)
    elif len(op_parts) == 4:
        dst = parse_operand(op_parts[0], acc_type)
        src0 = parse_operand(op_parts[1], elem_type)
        src1 = parse_operand(op_parts[2], elem_type)
        src2 = parse_operand(op_parts[3], acc_type)
        return ASTInstruction(opcode=opcode, dst=dst, src0=src0, src1=src1, src2=src2, clamp=clamp)
    else:
        raise ValueError(f"{opcode.value} expects 3 or 4 operands, got {len(op_parts)}: '{line}'")


class TestValuDot8IntSymbolicAST(unittest.TestCase):
    """Symbolic AST definitions and typed expression unit tests for v_dot8_int."""

    def test_ast_parse_v_dot8_i32_i4(self) -> None:
        instr = parse_instruction("v_dot8_i32_i4 v0, v1, v2, v3")
        self.assertEqual(instr.opcode, Opcode.V_DOT8_I32_I4)
        self.assertEqual(instr.dst, ASTOperand(OperandKind.VGPR, reg_num=0, type_kind=TypeKind.I32))
        self.assertEqual(instr.src0, ASTOperand(OperandKind.VGPR, reg_num=1, type_kind=TypeKind.I4))
        self.assertEqual(instr.src1, ASTOperand(OperandKind.VGPR, reg_num=2, type_kind=TypeKind.I4))
        self.assertEqual(instr.src2, ASTOperand(OperandKind.VGPR, reg_num=3, type_kind=TypeKind.I32))
        self.assertFalse(instr.clamp)

    def test_ast_parse_v_dot8_u32_u4_three_operands(self) -> None:
        instr = parse_instruction("v_dot8_u32_u4 v5, s2, v3")
        self.assertEqual(instr.opcode, Opcode.V_DOT8_U32_U4)
        self.assertEqual(instr.dst, ASTOperand(OperandKind.VGPR, reg_num=5, type_kind=TypeKind.U32))
        self.assertEqual(instr.src0, ASTOperand(OperandKind.SGPR, reg_num=2, type_kind=TypeKind.U4))
        self.assertEqual(instr.src1, ASTOperand(OperandKind.VGPR, reg_num=3, type_kind=TypeKind.U4))
        self.assertEqual(instr.src2, ASTOperand(OperandKind.INLINE_INT, literal_value=0, type_kind=TypeKind.U32))

    def test_ast_parse_with_clamp(self) -> None:
        instr = parse_instruction("v_dot8_u32_u4 v10, v11, v12, v13 clamp")
        self.assertEqual(instr.opcode, Opcode.V_DOT8_U32_U4)
        self.assertEqual(instr.dst.reg_num, 10)
        self.assertTrue(instr.clamp)

    def test_ast_roundtrip_formatting(self) -> None:
        asm = "v_dot8_i32_i4 v4, v1, v2, v3 clamp"
        instr = parse_instruction(asm)
        self.assertEqual(instr.to_asm(), asm)

    def test_ast_typed_expression_nodes(self) -> None:
        src0_expr = VariableExpr(TypeKind.I4, "v1")
        src1_expr = VariableExpr(TypeKind.I4, "v2")
        acc_expr = VariableExpr(TypeKind.I32, "v3")

        expr = Dot8Expr(
            type_kind=TypeKind.I32,
            opcode=Opcode.V_DOT8_I32_I4,
            src0=src0_expr,
            src1=src1_expr,
            acc=acc_expr,
            clamp=True,
        )
        self.assertEqual(expr.type_kind, TypeKind.I32)
        self.assertTrue(expr.clamp)


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

@dataclass
class ScalarRegisterFile:
    """Scalar and vector register storage for architectural simulation."""
    vgprs: list[int] = field(default_factory=lambda: [0] * 256)
    sgprs: list[int] = field(default_factory=lambda: [0] * 106)
    m0: int = 0
    vcc: int = 0

    def read_operand(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.VGPR:
            if not (0 <= op.reg_num < 256):
                raise IndexError(f"VGPR index out of bounds: v{op.reg_num}")
            return self.vgprs[op.reg_num] & MASK32
        elif op.kind == OperandKind.SGPR:
            if not (0 <= op.reg_num < 106):
                raise IndexError(f"SGPR index out of bounds: s{op.reg_num}")
            return self.sgprs[op.reg_num] & MASK32
        elif op.kind in (OperandKind.LITERAL, OperandKind.INLINE_INT):
            return op.literal_value & MASK32
        raise ValueError(f"Unknown operand kind: {op.kind}")

    def write_vgpr(self, reg_num: int, val: int) -> None:
        if not (0 <= reg_num < 256):
            raise IndexError(f"VGPR index out of bounds: v{reg_num}")
        self.vgprs[reg_num] = val & MASK32

    def write_sgpr(self, reg_num: int, val: int) -> None:
        if not (0 <= reg_num < 106):
            raise IndexError(f"SGPR index out of bounds: s{reg_num}")
        self.sgprs[reg_num] = val & MASK32


class Dot8IntScalarInterpreter:
    """Deterministic instruction interpreter for VALU dot8 operations."""

    def __init__(self, rf: Optional[ScalarRegisterFile] = None) -> None:
        self.rf = rf if rf is not None else ScalarRegisterFile()

    def step(self, instr: ASTInstruction) -> None:
        if instr.dst.kind != OperandKind.VGPR:
            raise ValueError(f"VALU instruction destination must be a VGPR, got {instr.dst.kind}")

        if instr.opcode == Opcode.V_MOV_B32:
            src_val = self.rf.read_operand(instr.src0)
            self.rf.write_vgpr(instr.dst.reg_num, src_val)
            return

        s0 = self.rf.read_operand(instr.src0)
        s1 = self.rf.read_operand(instr.src1)
        s2 = self.rf.read_operand(instr.src2) if instr.src2 is not None else 0

        if instr.opcode == Opcode.V_DOT8_I32_I4:
            result = ref_dot8_i32_i4(s0, s1, s2, clamp=instr.clamp)
        elif instr.opcode == Opcode.V_DOT8_U32_U4:
            result = ref_dot8_u32_u4(s0, s1, s2, clamp=instr.clamp)
        else:
            raise NotImplementedError(f"Unsupported opcode execution: {instr.opcode}")

        self.rf.write_vgpr(instr.dst.reg_num, result)

    def execute_asm(self, asm_lines: list[str]) -> None:
        for line in asm_lines:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith(";"):
                continue
            instr = parse_instruction(line)
            self.step(instr)


class TestValuDot8IntScalarInterpreter(unittest.TestCase):
    """Deterministic scalar interpreter tests for v_dot8_int."""

    def test_scalar_v_dot8_u32_u4_execution(self) -> None:
        interp = Dot8IntScalarInterpreter()
        s0 = pack_u4(1, 2, 3, 4, 1, 2, 3, 4)
        s1 = pack_u4(1, 1, 1, 1, 2, 2, 2, 2)
        # Sum: 1+2+3+4 + 2+4+6+8 = 10 + 20 = 30; acc = 70 -> 100
        interp.rf.write_vgpr(1, s0)
        interp.rf.write_vgpr(2, s1)
        interp.rf.write_vgpr(3, 70)

        interp.execute_asm(["v_dot8_u32_u4 v0, v1, v2, v3"])
        self.assertEqual(interp.rf.vgprs[0], 100)

    def test_scalar_v_dot8_i32_i4_execution(self) -> None:
        interp = Dot8IntScalarInterpreter()
        s0 = pack_i4(-2, -2, -2, -2, 3, 3, 3, 3)
        s1 = pack_i4(1, 1, 1, 1, 1, 1, 1, 1)
        # Sum: 4*(-2) + 4*(3) = -8 + 12 = 4
        interp.rf.write_vgpr(1, s0)
        interp.rf.write_vgpr(2, s1)
        interp.rf.write_vgpr(3, 0)

        interp.execute_asm(["v_dot8_i32_i4 v0, v1, v2, v3"])
        self.assertEqual(interp.rf.vgprs[0], 4)

    def test_scalar_sgpr_and_inline_operands(self) -> None:
        interp = Dot8IntScalarInterpreter()
        interp.rf.write_sgpr(4, pack_u4(1, 1, 1, 1, 1, 1, 1, 1))
        interp.rf.write_vgpr(1, pack_u4(2, 2, 2, 2, 2, 2, 2, 2))

        # 8 * (1 * 2) = 16; acc = 15 -> 31
        interp.execute_asm(["v_dot8_u32_u4 v0, s4, v1, 15"])
        self.assertEqual(interp.rf.vgprs[0], 31)

    def test_scalar_clamp_execution(self) -> None:
        interp = Dot8IntScalarInterpreter()
        s_min = pack_i4(-8, -8, -8, -8, -8, -8, -8, -8)  # 8 * 64 = +512
        interp.rf.write_vgpr(1, s_min)
        interp.rf.write_vgpr(2, s_min)
        interp.rf.write_vgpr(3, INT32_MAX)

        interp.execute_asm(["v_dot8_i32_i4 v0, v1, v2, v3 clamp"])
        self.assertEqual(interp.rf.vgprs[0], INT32_MAX & MASK32)


# ---------------------------------------------------------------------------
# SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking Model
# ---------------------------------------------------------------------------

class SIMDWaveState:
    """Wavefront SIMD execution state supporting Wave32 and Wave64 execution modes."""

    def __init__(self, wave_size: int = 32) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave_size {wave_size}; RDNA2 supports only Wave32 and Wave64")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: list[int] = [0] * 106

    def read_operand_for_lane(self, op: ASTOperand, lane: int) -> int:
        if op.kind == OperandKind.VGPR:
            if not (0 <= op.reg_num < 256):
                raise IndexError(f"VGPR index out of bounds: v{op.reg_num}")
            return self.vgprs[lane][op.reg_num] & MASK32
        elif op.kind == OperandKind.SGPR:
            if not (0 <= op.reg_num < 106):
                raise IndexError(f"SGPR index out of bounds: s{op.reg_num}")
            return self.sgprs[op.reg_num] & MASK32
        elif op.kind in (OperandKind.LITERAL, OperandKind.INLINE_INT):
            return op.literal_value & MASK32
        raise ValueError(f"Unknown operand kind: {op.kind}")

    def write_vgpr_lane(self, lane: int, reg_num: int, val: int) -> None:
        if not (0 <= reg_num < 256):
            raise IndexError(f"VGPR index out of bounds: v{reg_num}")
        self.vgprs[lane][reg_num] = val & MASK32

    def step(self, instr: ASTInstruction) -> None:
        if instr.dst.kind != OperandKind.VGPR:
            raise ValueError(f"VALU instruction destination must be a VGPR, got {instr.dst.kind}")

        dst_reg = instr.dst.reg_num
        for lane in range(self.wave_size):
            if not (self.exec_mask & (1 << lane)):
                continue

            if instr.opcode == Opcode.V_MOV_B32:
                src_val = self.read_operand_for_lane(instr.src0, lane)
                self.write_vgpr_lane(lane, dst_reg, src_val)
                continue

            s0 = self.read_operand_for_lane(instr.src0, lane)
            s1 = self.read_operand_for_lane(instr.src1, lane)
            s2 = self.read_operand_for_lane(instr.src2, lane) if instr.src2 is not None else 0

            if instr.opcode == Opcode.V_DOT8_I32_I4:
                res = ref_dot8_i32_i4(s0, s1, s2, clamp=instr.clamp)
            elif instr.opcode == Opcode.V_DOT8_U32_U4:
                res = ref_dot8_u32_u4(s0, s1, s2, clamp=instr.clamp)
            else:
                raise NotImplementedError(f"Unsupported opcode execution: {instr.opcode}")

            self.write_vgpr_lane(lane, dst_reg, res)

    def execute_asm(self, asm_lines: list[str]) -> None:
        for line in asm_lines:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith(";"):
                continue
            instr = parse_instruction(line)
            self.step(instr)


class TestValuDot8IntSIMDMultiLane(unittest.TestCase):
    """SIMD multi-lane divergence and 32/64-bit EXEC masking unit tests for v_dot8_int."""

    def test_wave32_uniform_execution(self) -> None:
        wave = SIMDWaveState(wave_size=32)
        s0 = pack_u4(1, 1, 1, 1, 1, 1, 1, 1)
        s1 = pack_u4(2, 2, 2, 2, 2, 2, 2, 2)
        for lane in range(32):
            wave.write_vgpr_lane(lane, 1, s0)
            wave.write_vgpr_lane(lane, 2, s1)
            wave.write_vgpr_lane(lane, 3, 50)

        # 8 * 2 = 16; 50 + 16 = 66
        wave.execute_asm(["v_dot8_u32_u4 v0, v1, v2, v3"])
        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][0], 66)

    def test_wave64_uniform_execution(self) -> None:
        wave = SIMDWaveState(wave_size=64)
        s0 = pack_i4(1, -1, 1, -1, 1, -1, 1, -1)
        s1 = pack_i4(3, 3, 3, 3, 3, 3, 3, 3)
        # Sum = 4*(3) + 4*(-3) = 0
        for lane in range(64):
            wave.write_vgpr_lane(lane, 1, s0)
            wave.write_vgpr_lane(lane, 2, s1)
            wave.write_vgpr_lane(lane, 3, 42)

        wave.execute_asm(["v_dot8_i32_i4 v0, v1, v2, v3"])
        for lane in range(64):
            self.assertEqual(wave.vgprs[lane][0], 42)

    def test_wave32_divergent_alternating_exec_mask(self) -> None:
        wave = SIMDWaveState(wave_size=32)
        wave.exec_mask = 0x5555_5555
        s0 = pack_u4(1, 1, 1, 1, 1, 1, 1, 1)
        s1 = pack_u4(3, 3, 3, 3, 3, 3, 3, 3)

        for lane in range(32):
            wave.write_vgpr_lane(lane, 0, 0xCAFE_BABE)
            wave.write_vgpr_lane(lane, 1, s0)
            wave.write_vgpr_lane(lane, 2, s1)

        # Active even lanes: 8 * 3 = 24
        # Inactive odd lanes: preserved sentinel 0xCAFE_BABE
        wave.execute_asm(["v_dot8_u32_u4 v0, v1, v2, 0"])
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[lane][0], 24)
            else:
                self.assertEqual(wave.vgprs[lane][0], 0xCAFE_BABE)

    def test_wave64_divergent_high_half_exec_mask(self) -> None:
        wave = SIMDWaveState(wave_size=64)
        wave.exec_mask = 0xFFFF_FFFF_0000_0000
        for lane in range(64):
            wave.write_vgpr_lane(lane, 0, 0x1234_5678)
            wave.write_vgpr_lane(lane, 1, pack_u4(1, 2, 1, 2, 1, 2, 1, 2))
            wave.write_vgpr_lane(lane, 2, pack_u4(1, 1, 1, 1, 1, 1, 1, 1))

        # 4*1 + 4*2 = 12
        wave.execute_asm(["v_dot8_u32_u4 v0, v1, v2, 0"])
        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][0], 0x1234_5678)
        for lane in range(32, 64):
            self.assertEqual(wave.vgprs[lane][0], 12)

    def test_wave_exec_mask_zero_preservation(self) -> None:
        wave = SIMDWaveState(wave_size=32)
        wave.exec_mask = 0x0000_0000
        for lane in range(32):
            wave.write_vgpr_lane(lane, 0, 0x5555_AAAA)
            wave.write_vgpr_lane(lane, 1, 0xFFFF_FFFF)
            wave.write_vgpr_lane(lane, 2, 0xFFFF_FFFF)

        wave.execute_asm(["v_dot8_u32_u4 v0, v1, v2, 0"])
        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][0], 0x5555_AAAA)

    def test_divergent_per_lane_vector_data(self) -> None:
        wave = SIMDWaveState(wave_size=32)
        for lane in range(32):
            # Lane ID in lowest nibble
            wave.write_vgpr_lane(lane, 1, pack_u4(lane & 0xF, 1, 1, 1, 1, 1, 1, 1))
            wave.write_vgpr_lane(lane, 2, pack_u4(2, 1, 1, 1, 1, 1, 1, 1))
            wave.write_vgpr_lane(lane, 3, 10)

        # (lane & 0xF) * 2 + 7 * (1 * 1) + 10 = 2 * (lane & 0xF) + 17
        wave.execute_asm(["v_dot8_u32_u4 v0, v1, v2, v3"])
        for lane in range(32):
            expected = 2 * (lane & 0xF) + 17
            self.assertEqual(wave.vgprs[lane][0], expected)


# ---------------------------------------------------------------------------
# Register Aliasing and In-Place Operand Preservation Tests
# ---------------------------------------------------------------------------

class TestValuDot8IntRegisterAliasing(unittest.TestCase):
    """Register aliasing, self-overwrite, and in-place operand preservation unit tests."""

    def test_aliasing_dst_equals_src0(self) -> None:
        interp = Dot8IntScalarInterpreter()
        initial_v1 = pack_u4(1, 2, 3, 4, 1, 2, 3, 4)
        v2 = pack_u4(2, 2, 2, 2, 2, 2, 2, 2)
        interp.rf.write_vgpr(1, initial_v1)
        interp.rf.write_vgpr(2, v2)

        # 2 * (1+2+3+4+1+2+3+4) = 2 * 20 = 40
        interp.execute_asm(["v_dot8_u32_u4 v1, v1, v2, 0"])
        self.assertEqual(interp.rf.vgprs[1], 40)

    def test_aliasing_dst_equals_src1(self) -> None:
        interp = Dot8IntScalarInterpreter()
        v1 = pack_i4(-1, 1, -1, 1, -1, 1, -1, 1)
        v2 = pack_i4(2, 3, 2, 3, 2, 3, 2, 3)
        interp.rf.write_vgpr(1, v1)
        interp.rf.write_vgpr(2, v2)

        # 4 * ((-1)*2 + 1*3) = 4 * (-2 + 3) = 4 * 1 = 4
        interp.execute_asm(["v_dot8_i32_i4 v2, v1, v2, 0"])
        self.assertEqual(interp.rf.vgprs[2], 4)

    def test_aliasing_dst_equals_src2_accumulator(self) -> None:
        interp = Dot8IntScalarInterpreter()
        v1 = pack_u4(1, 1, 1, 1, 1, 1, 1, 1)
        v2 = pack_u4(2, 2, 2, 2, 2, 2, 2, 2)
        interp.rf.write_vgpr(1, v1)
        interp.rf.write_vgpr(2, v2)
        interp.rf.write_vgpr(3, 500)

        # 8 * 2 = 16; 500 + 16 = 516
        interp.execute_asm(["v_dot8_u32_u4 v3, v1, v2, v3"])
        self.assertEqual(interp.rf.vgprs[3], 516)

    def test_aliasing_dst_equals_src0_and_src1_squared(self) -> None:
        interp = Dot8IntScalarInterpreter()
        # Vector squared norm: v1 . v1
        v1 = pack_u4(1, 2, 1, 2, 1, 2, 1, 2)
        interp.rf.write_vgpr(1, v1)

        # 4*(1^2) + 4*(2^2) = 4 + 16 = 20; acc = 10 -> 30
        interp.execute_asm(["v_dot8_u32_u4 v1, v1, v1, 10"])
        self.assertEqual(interp.rf.vgprs[1], 30)

    def test_full_aliasing_dst_src0_src1_src2(self) -> None:
        interp = Dot8IntScalarInterpreter()
        # Single register v0 serves as dst, src0, src1, and src2
        v0 = pack_u4(1, 1, 1, 1, 1, 1, 1, 1)  # 0x11111111
        interp.rf.write_vgpr(0, v0)

        # Sum = 8 * 1 = 8; acc = 0x11111111 -> 0x11111119
        interp.execute_asm(["v_dot8_u32_u4 v0, v0, v0, v0"])
        self.assertEqual(interp.rf.vgprs[0], 0x1111_1119)

    def test_simd_multi_lane_aliased_accumulation(self) -> None:
        wave = SIMDWaveState(wave_size=32)
        wave.exec_mask = 0x5555_5555
        for lane in range(32):
            wave.write_vgpr_lane(lane, 0, 100)
            wave.write_vgpr_lane(lane, 1, pack_u4(1, 1, 1, 1, 1, 1, 1, 1))
            wave.write_vgpr_lane(lane, 2, pack_u4(2, 2, 2, 2, 2, 2, 2, 2))

        # Active lanes: 100 + 8*2 = 116; Inactive: 100
        wave.execute_asm(["v_dot8_u32_u4 v0, v1, v2, v0"])
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[lane][0], 116)
            else:
                self.assertEqual(wave.vgprs[lane][0], 100)


if __name__ == "__main__":
    unittest.main()
