from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_dot4_int opcodes.

This module provides an architectural reference model and semantic test suite for RDNA2
VALU packed 8-bit integer dot product instructions:
    - v_dot4_i32_i8 (VOP3/VOP3P): Signed 8-bit integer 4-vector dot product:
        D.i32 = S2.i32 + sum(S0.i8[i] * S1.i8[i] for i in 0..3) mod 2^32
    - v_dot4_u32_u8 (VOP3/VOP3P): Unsigned 8-bit integer 4-vector dot product:
        D.u32 = S2.u32 + sum(S0.u8[i] * S1.u8[i] for i in 0..3) mod 2^32
    - v_dot4acc_u32_u8: Accumulating unsigned 8-bit integer dot product with explicit accumulator:
        D.u32 = S2.u32 + sum(S0.u8[i] * S1.u8[i] for i in 0..3) mod 2^32

ISA Specifications (RDNA2 / GFX10.3+):
    - Each 32-bit dword S0 and S1 contains four packed 8-bit integers:
        Byte 0: bits [7:0]
        Byte 1: bits [15:8]
        Byte 2: bits [23:16]
        Byte 3: bits [31:24]
    - In v_dot4_i32_i8, each byte is treated as signed 2's complement int8 in [-128, +127].
      Each component product S0.i8[i] * S1.i8[i] is in [-16256, +16384].
      The four products sum to [-65024, +65536], which is added to 32-bit accumulator S2
      and wrapped modulo 2^32 (or clamped to [-2^31, 2^31 - 1] if clamp is asserted).
    - In v_dot4_u32_u8, each byte is treated as unsigned uint8 in [0, 255].
      Each component product S0.u8[i] * S1.u8[i] is in [0, 65025].
      The four products sum to [0, 260100], which is added to 32-bit accumulator S2
      and wrapped modulo 2^32 (or clamped to [0, 0xFFFFFFFF] if clamp is asserted).

Standard library Python only; zero third-party dependencies.
Python 3.9+ compatible.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import re
import struct
from typing import Optional, Union, List, Tuple
import unittest

# Bitmask and architectural boundary constants
MASK8: int = 0xFF
MASK32: int = 0xFFFF_FFFF
SIGN8_BIT: int = 0x80
INT32_MIN: int = -0x8000_0000
INT32_MAX: int = 0x7FFF_FFFF
UINT32_MAX: int = 0xFFFF_FFFF


def unpack_u8(dword: int) -> tuple[int, int, int, int]:
    """Unpack a 32-bit dword into four unsigned 8-bit integers (byte0, byte1, byte2, byte3)."""
    val = dword & MASK32
    return (
        val & MASK8,
        (val >> 8) & MASK8,
        (val >> 16) & MASK8,
        (val >> 24) & MASK8,
    )


def pack_u8(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four unsigned 8-bit integers into a 32-bit dword."""
    return ((b0 & MASK8) | ((b1 & MASK8) << 8) | ((b2 & MASK8) << 16) | ((b3 & MASK8) << 24)) & MASK32


def unpack_i8(dword: int) -> tuple[int, int, int, int]:
    """Unpack a 32-bit dword into four signed 8-bit integers (-128 to +127)."""
    raw_bytes = unpack_u8(dword)
    return tuple((b - 256) if (b & SIGN8_BIT) else b for b in raw_bytes)  # type: ignore[return-value]


def pack_i8(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four signed 8-bit integers into a 32-bit dword."""
    return pack_u8(b0 & MASK8, b1 & MASK8, b2 & MASK8, b3 & MASK8)


def ref_dot4_u32_u8(s0: int, s1: int, s2: int = 0, *, clamp: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 unsigned 8-bit dot product (v_dot4_u32_u8).

    Calculates:
        D = (s2 + sum(s0[i] * s1[i] for i in 0..3))
    Modulo 2^32 without clamp; saturated to 0xFFFFFFFF if clamp is set.
    Returns unsigned 32-bit integer in [0, 0xFFFFFFFF].
    """
    b0 = unpack_u8(s0)
    b1 = unpack_u8(s1)
    prod_sum = sum(x * y for x, y in zip(b0, b1))
    acc = s2 & MASK32
    total = acc + prod_sum

    if clamp:
        return min(total, UINT32_MAX)
    return total & MASK32


def ref_dot4acc_u32_u8(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    """Reference mathematical oracle for v_dot4acc_u32_u8 accumulating form."""
    return ref_dot4_u32_u8(s0, s1, s2, clamp=clamp)


def ref_dot4_i32_i8(s0: int, s1: int, s2: int = 0, *, clamp: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 signed 8-bit dot product (v_dot4_i32_i8).

    Calculates:
        D = (s2 + sum(s0.i8[i] * s1.i8[i] for i in 0..3))
    where S2 is interpreted as signed 32-bit integer.
    Modulo 2^32 without clamp; saturated to [-2^31, 2^31 - 1] if clamp is set.
    Returns unsigned 32-bit bit pattern representing the resulting dword.
    """
    b0 = unpack_i8(s0)
    b1 = unpack_i8(s1)
    prod_sum = sum(x * y for x, y in zip(b0, b1))

    # Interpret s2 as signed 32-bit integer
    s2_u32 = s2 & MASK32
    s2_i32 = (s2_u32 - 0x1_0000_0000) if (s2_u32 & 0x8000_0000) else s2_u32

    total_i64 = s2_i32 + prod_sum

    if clamp:
        clamped = max(INT32_MIN, min(total_i64, INT32_MAX))
        return clamped & MASK32
    return total_i64 & MASK32


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class Opcode(Enum):
    """VALU opcode enumerations for packed integer dot products."""
    V_DOT4_I32_I8 = "v_dot4_i32_i8"
    V_DOT4_U32_U8 = "v_dot4_u32_u8"
    V_DOT4ACC_U32_U8 = "v_dot4acc_u32_u8"
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
    I8 = "i8"
    U8 = "u8"


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
class Dot4ComponentProductExpr(TypedExpr):
    """Component product of byte i: src0.byte[i] * src1.byte[i]."""
    byte_index: int
    left: TypedExpr
    right: TypedExpr


@dataclass
class Dot4Expr(TypedExpr):
    """Full dot4 expression: accumulator + sum(component products), with optional clamp."""
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
        # Literal or inline constant
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

    # v_dot4 instructions:
    if opcode == Opcode.V_DOT4_I32_I8:
        elem_type = TypeKind.I8
        acc_type = TypeKind.I32
    else:
        elem_type = TypeKind.U8
        acc_type = TypeKind.U32

    if len(op_parts) == 3:
        # 3-operand form without accumulator: implicit acc = 0
        dst = parse_operand(op_parts[0], acc_type)
        src0 = parse_operand(op_parts[1], elem_type)
        src1 = parse_operand(op_parts[2], elem_type)
        src2 = ASTOperand(kind=OperandKind.INLINE_INT, literal_value=0, type_kind=acc_type)
        return ASTInstruction(opcode=opcode, dst=dst, src0=src0, src1=src1, src2=src2, clamp=clamp)
    elif len(op_parts) == 4:
        # 4-operand form with explicit accumulator
        dst = parse_operand(op_parts[0], acc_type)
        src0 = parse_operand(op_parts[1], elem_type)
        src1 = parse_operand(op_parts[2], elem_type)
        src2 = parse_operand(op_parts[3], acc_type)
        return ASTInstruction(opcode=opcode, dst=dst, src0=src0, src1=src1, src2=src2, clamp=clamp)
    else:
        raise ValueError(f"{opcode.value} expects 3 or 4 operands, got {len(op_parts)}: '{line}'")


# ---------------------------------------------------------------------------
# Unit Tests: Phase 1 & Phase 2
# ---------------------------------------------------------------------------

class TestValuDot4IntMathematicalOracle(unittest.TestCase):
    """Architectural reference mathematical oracle unit tests for v_dot4_int."""

    def test_u8_basic_vectors(self) -> None:
        # 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        s0 = pack_u8(1, 2, 3, 4)
        s1 = pack_u8(5, 6, 7, 8)
        self.assertEqual(ref_dot4_u32_u8(s0, s1, 0), 70)
        # With accumulator = 1000 -> 1070
        self.assertEqual(ref_dot4_u32_u8(s0, s1, 1000), 1070)
        self.assertEqual(ref_dot4acc_u32_u8(s0, s1, 1000), 1070)

    def test_u8_zero_operands(self) -> None:
        s0 = 0x0000_0000
        s1 = 0x1234_5678
        self.assertEqual(ref_dot4_u32_u8(s0, s1, 0xDEAD_BEEF), 0xDEAD_BEEF)
        self.assertEqual(ref_dot4_u32_u8(s1, s0, 0xCAFE_BABE), 0xCAFE_BABE)

    def test_u8_maximum_product(self) -> None:
        # 4 * (255 * 255) = 4 * 65025 = 260100 (0x3F804)
        s0 = 0xFFFF_FFFF
        s1 = 0xFFFF_FFFF
        self.assertEqual(ref_dot4_u32_u8(s0, s1, 0), 260100)
        self.assertEqual(ref_dot4_u32_u8(s0, s1, 100), 260200)

    def test_u8_overflow_wrap_around(self) -> None:
        # Wrap around 2^32: (0xFFFFFFFF - 260100 + 1) + 260100 = 0
        s0 = 0xFFFF_FFFF
        s1 = 0xFFFF_FFFF
        acc = (MASK32 - 260100 + 1) & MASK32
        self.assertEqual(ref_dot4_u32_u8(s0, s1, acc), 0)

        # Overflow by 1
        acc_plus1 = (acc + 1) & MASK32
        self.assertEqual(ref_dot4_u32_u8(s0, s1, acc_plus1), 1)

    def test_u8_clamp_saturation(self) -> None:
        s0 = 0xFFFF_FFFF
        s1 = 0xFFFF_FFFF
        acc = 0xFFFF_0000
        # Without clamp, wraps around
        wrapped = ref_dot4_u32_u8(s0, s1, acc, clamp=False)
        self.assertLess(wrapped, 0xFFFF_0000)
        # With clamp, saturates to UINT32_MAX
        clamped = ref_dot4_u32_u8(s0, s1, acc, clamp=True)
        self.assertEqual(clamped, UINT32_MAX)

    def test_i8_basic_vectors(self) -> None:
        # 1*5 + 2*6 + 3*7 + 4*8 = 70
        s0 = pack_i8(1, 2, 3, 4)
        s1 = pack_i8(5, 6, 7, 8)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 70)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 500), 570)

    def test_i8_negative_operands(self) -> None:
        # (-1)*5 + 2*(-6) + (-3)*(-7) + (-4)*8 = -5 + -12 + 21 + -32 = -28
        s0 = pack_i8(-1, 2, -3, -4)
        s1 = pack_i8(5, -6, -7, 8)
        expected = (-28) & MASK32
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), expected)

    def test_i8_extremes(self) -> None:
        # Max positive product: 4 * ((-128) * (-128)) = 4 * 16384 = 65536
        s_min = pack_i8(-128, -128, -128, -128)
        self.assertEqual(ref_dot4_i32_i8(s_min, s_min, 0), 65536)

        # Max negative product: 4 * ((-128) * 127) = 4 * (-16256) = -65024
        s_max = pack_i8(127, 127, 127, 127)
        expected_neg = (-65024) & MASK32
        self.assertEqual(ref_dot4_i32_i8(s_min, s_max, 0), expected_neg)

    def test_i8_signed_overflow_and_clamp(self) -> None:
        s_min = pack_i8(-128, -128, -128, -128)
        # Adding to INT32_MAX without clamp wraps to negative
        res_wrap = ref_dot4_i32_i8(s_min, s_min, INT32_MAX, clamp=False)
        self.assertTrue(bool(res_wrap & 0x8000_0000))

        # With clamp, saturates to INT32_MAX
        res_clamp_pos = ref_dot4_i32_i8(s_min, s_min, INT32_MAX, clamp=True)
        self.assertEqual(res_clamp_pos, INT32_MAX & MASK32)

        # Negative saturation: INT32_MIN + (-65024) saturates to INT32_MIN
        s_max = pack_i8(127, 127, 127, 127)
        res_clamp_neg = ref_dot4_i32_i8(s_min, s_max, INT32_MIN, clamp=True)
        self.assertEqual(res_clamp_neg, INT32_MIN & MASK32)

    def test_divergence_between_signed_and_unsigned(self) -> None:
        # 0xFF as unsigned is 255; as signed is -1
        # S0 = [255, 0, 0, 0], S1 = [1, 0, 0, 0]
        # Unsigned: 255 * 1 = 255
        # Signed: (-1) * 1 = -1 (0xFFFFFFFF)
        s0 = pack_u8(255, 0, 0, 0)
        s1 = pack_u8(1, 0, 0, 0)
        u_res = ref_dot4_u32_u8(s0, s1, 0)
        i_res = ref_dot4_i32_i8(s0, s1, 0)
        self.assertEqual(u_res, 255)
        self.assertEqual(i_res, 0xFFFF_FFFF)
        self.assertNotEqual(u_res, i_res)


class TestValuDot4IntSymbolicAST(unittest.TestCase):
    """Symbolic AST definitions and typed expression unit tests for v_dot4_int."""

    def test_ast_parse_v_dot4_i32_i8(self) -> None:
        instr = parse_instruction("v_dot4_i32_i8 v0, v1, v2, v3")
        self.assertEqual(instr.opcode, Opcode.V_DOT4_I32_I8)
        self.assertEqual(instr.dst, ASTOperand(OperandKind.VGPR, reg_num=0, type_kind=TypeKind.I32))
        self.assertEqual(instr.src0, ASTOperand(OperandKind.VGPR, reg_num=1, type_kind=TypeKind.I8))
        self.assertEqual(instr.src1, ASTOperand(OperandKind.VGPR, reg_num=2, type_kind=TypeKind.I8))
        self.assertEqual(instr.src2, ASTOperand(OperandKind.VGPR, reg_num=3, type_kind=TypeKind.I32))
        self.assertFalse(instr.clamp)

    def test_ast_parse_v_dot4_u32_u8_three_operands(self) -> None:
        instr = parse_instruction("v_dot4_u32_u8 v5, s2, v3")
        self.assertEqual(instr.opcode, Opcode.V_DOT4_U32_U8)
        self.assertEqual(instr.dst, ASTOperand(OperandKind.VGPR, reg_num=5, type_kind=TypeKind.U32))
        self.assertEqual(instr.src0, ASTOperand(OperandKind.SGPR, reg_num=2, type_kind=TypeKind.U8))
        self.assertEqual(instr.src1, ASTOperand(OperandKind.VGPR, reg_num=3, type_kind=TypeKind.U8))
        # Default accumulator is literal 0
        self.assertEqual(instr.src2, ASTOperand(OperandKind.INLINE_INT, literal_value=0, type_kind=TypeKind.U32))

    def test_ast_parse_v_dot4acc_u32_u8_with_clamp(self) -> None:
        instr = parse_instruction("v_dot4acc_u32_u8 v10, v11, v12, v10 clamp")
        self.assertEqual(instr.opcode, Opcode.V_DOT4ACC_U32_U8)
        self.assertEqual(instr.dst.reg_num, 10)
        self.assertEqual(instr.src0.reg_num, 11)
        self.assertEqual(instr.src1.reg_num, 12)
        self.assertEqual(instr.src2.reg_num, 10)
        self.assertTrue(instr.clamp)

    def test_ast_parse_literal_and_inline_operands(self) -> None:
        instr = parse_instruction("v_dot4_u32_u8 v0, 0x01020304, v1, 100")
        self.assertEqual(instr.src0.kind, OperandKind.LITERAL)
        self.assertEqual(instr.src0.literal_value, 0x01020304)
        self.assertEqual(instr.src2.kind, OperandKind.LITERAL)
        self.assertEqual(instr.src2.literal_value, 100)

    def test_ast_roundtrip_formatting(self) -> None:
        asm = "v_dot4_i32_i8 v4, v1, v2, v3 clamp"
        instr = parse_instruction(asm)
        reformatted = instr.to_asm()
        self.assertEqual(reformatted, asm)

    def test_ast_typed_expression_nodes(self) -> None:
        src0_expr = VariableExpr(TypeKind.I8, "v1")
        src1_expr = VariableExpr(TypeKind.I8, "v2")
        acc_expr = VariableExpr(TypeKind.I32, "v3")

        expr = Dot4Expr(
            type_kind=TypeKind.I32,
            opcode=Opcode.V_DOT4_I32_I8,
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


class Dot4IntScalarInterpreter:
    """Deterministic instruction interpreter for VALU dot4 operations."""

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

        if instr.opcode == Opcode.V_DOT4_I32_I8:
            result = ref_dot4_i32_i8(s0, s1, s2, clamp=instr.clamp)
        elif instr.opcode in (Opcode.V_DOT4_U32_U8, Opcode.V_DOT4ACC_U32_U8):
            result = ref_dot4_u32_u8(s0, s1, s2, clamp=instr.clamp)
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


class TestValuDot4IntScalarInterpreter(unittest.TestCase):
    """Deterministic scalar interpreter tests for v_dot4_int."""

    def test_scalar_v_dot4_u32_u8_execution(self) -> None:
        interp = Dot4IntScalarInterpreter()
        # Set v1 = [1, 2, 3, 4], v2 = [5, 6, 7, 8], v3 = 100
        interp.rf.write_vgpr(1, pack_u8(1, 2, 3, 4))
        interp.rf.write_vgpr(2, pack_u8(5, 6, 7, 8))
        interp.rf.write_vgpr(3, 100)

        interp.execute_asm(["v_dot4_u32_u8 v0, v1, v2, v3"])
        # Expected: 70 + 100 = 170
        self.assertEqual(interp.rf.vgprs[0], 170)

    def test_scalar_v_dot4_i32_i8_execution(self) -> None:
        interp = Dot4IntScalarInterpreter()
        # Set v1 = [-1, 2, -3, -4], v2 = [5, -6, -7, 8], v3 = 0
        interp.rf.write_vgpr(1, pack_i8(-1, 2, -3, -4))
        interp.rf.write_vgpr(2, pack_i8(5, -6, -7, 8))
        interp.rf.write_vgpr(3, 0)

        interp.execute_asm(["v_dot4_i32_i8 v0, v1, v2, v3"])
        expected = (-28) & MASK32
        self.assertEqual(interp.rf.vgprs[0], expected)

    def test_scalar_v_dot4acc_u32_u8_accumulation_chain(self) -> None:
        interp = Dot4IntScalarInterpreter()
        interp.rf.write_vgpr(1, pack_u8(1, 1, 1, 1))
        interp.rf.write_vgpr(2, pack_u8(10, 20, 30, 40))  # sum = 100
        interp.rf.write_vgpr(0, 50)  # Initial accumulator

        # Step 1: v0 = 50 + 100 = 150
        interp.execute_asm(["v_dot4acc_u32_u8 v0, v1, v2, v0"])
        self.assertEqual(interp.rf.vgprs[0], 150)

        # Step 2: v0 = 150 + 100 = 250
        interp.execute_asm(["v_dot4acc_u32_u8 v0, v1, v2, v0"])
        self.assertEqual(interp.rf.vgprs[0], 250)

    def test_scalar_sgpr_and_inline_operands(self) -> None:
        interp = Dot4IntScalarInterpreter()
        interp.rf.write_sgpr(5, pack_u8(2, 2, 2, 2))
        interp.rf.write_vgpr(1, pack_u8(3, 3, 3, 3))

        # v_dot4_u32_u8 v0, s5, v1, 10
        # 4 * (2 * 3) = 24; 24 + 10 = 34
        interp.execute_asm(["v_dot4_u32_u8 v0, s5, v1, 10"])
        self.assertEqual(interp.rf.vgprs[0], 34)

    def test_scalar_clamp_execution(self) -> None:
        interp = Dot4IntScalarInterpreter()
        # Max positive product
        s_min = pack_i8(-128, -128, -128, -128)
        interp.rf.write_vgpr(1, s_min)
        interp.rf.write_vgpr(2, s_min)
        interp.rf.write_vgpr(3, INT32_MAX)

        interp.execute_asm(["v_dot4_i32_i8 v0, v1, v2, v3 clamp"])
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
            # Check if current lane is active in EXEC mask
            if not (self.exec_mask & (1 << lane)):
                continue

            if instr.opcode == Opcode.V_MOV_B32:
                src_val = self.read_operand_for_lane(instr.src0, lane)
                self.write_vgpr_lane(lane, dst_reg, src_val)
                continue

            s0 = self.read_operand_for_lane(instr.src0, lane)
            s1 = self.read_operand_for_lane(instr.src1, lane)
            s2 = self.read_operand_for_lane(instr.src2, lane) if instr.src2 is not None else 0

            if instr.opcode == Opcode.V_DOT4_I32_I8:
                res = ref_dot4_i32_i8(s0, s1, s2, clamp=instr.clamp)
            elif instr.opcode in (Opcode.V_DOT4_U32_U8, Opcode.V_DOT4ACC_U32_U8):
                res = ref_dot4_u32_u8(s0, s1, s2, clamp=instr.clamp)
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


class TestValuDot4IntSIMDMultiLane(unittest.TestCase):
    """SIMD multi-lane divergence and 32/64-bit EXEC masking unit tests for v_dot4_int."""

    def test_wave32_uniform_execution(self) -> None:
        wave = SIMDWaveState(wave_size=32)
        s0 = pack_u8(1, 2, 3, 4)
        s1 = pack_u8(5, 6, 7, 8)
        # Uniform vector setup across all 32 lanes
        for lane in range(32):
            wave.write_vgpr_lane(lane, 1, s0)
            wave.write_vgpr_lane(lane, 2, s1)
            wave.write_vgpr_lane(lane, 3, 100)

        wave.execute_asm(["v_dot4_u32_u8 v0, v1, v2, v3"])
        # All active lanes should have computed 70 + 100 = 170
        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][0], 170)

    def test_wave64_uniform_execution(self) -> None:
        wave = SIMDWaveState(wave_size=64)
        s0 = pack_i8(2, 2, 2, 2)
        s1 = pack_i8(3, 3, 3, 3)
        for lane in range(64):
            wave.write_vgpr_lane(lane, 1, s0)
            wave.write_vgpr_lane(lane, 2, s1)
            wave.write_vgpr_lane(lane, 3, 0)

        wave.execute_asm(["v_dot4_i32_i8 v0, v1, v2, v3"])
        for lane in range(64):
            self.assertEqual(wave.vgprs[lane][0], 24)

    def test_wave32_divergent_alternating_exec_mask(self) -> None:
        wave = SIMDWaveState(wave_size=32)
        # EXEC mask with alternating active lanes (even lanes active, odd inactive)
        wave.exec_mask = 0x5555_5555
        s0 = pack_u8(1, 1, 1, 1)
        s1 = pack_u8(2, 2, 2, 2)

        # Pre-fill destination v0 with sentinel value 0xDEADBEEF
        for lane in range(32):
            wave.write_vgpr_lane(lane, 0, 0xDEAD_BEEF)
            wave.write_vgpr_lane(lane, 1, s0)
            wave.write_vgpr_lane(lane, 2, s1)

        wave.execute_asm(["v_dot4_u32_u8 v0, v1, v2, 0"])
        # Active even lanes: 4 * (1 * 2) = 8
        # Inactive odd lanes: preserved sentinel 0xDEAD_BEEF
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[lane][0], 8, f"Lane {lane} active failed")
            else:
                self.assertEqual(wave.vgprs[lane][0], 0xDEAD_BEEF, f"Lane {lane} inactive modified")

    def test_wave64_divergent_high_half_exec_mask(self) -> None:
        wave = SIMDWaveState(wave_size=64)
        # Enable only upper 32 lanes: lanes [32..63]
        wave.exec_mask = 0xFFFF_FFFF_0000_0000
        for lane in range(64):
            wave.write_vgpr_lane(lane, 0, 0x1111_1111)
            wave.write_vgpr_lane(lane, 1, pack_u8(4, 3, 2, 1))
            wave.write_vgpr_lane(lane, 2, pack_u8(1, 2, 3, 4))

        wave.execute_asm(["v_dot4_u32_u8 v0, v1, v2, 5"])
        # 4*1 + 3*2 + 2*3 + 1*4 = 4 + 6 + 6 + 4 = 20; 20 + 5 = 25
        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][0], 0x1111_1111)
        for lane in range(32, 64):
            self.assertEqual(wave.vgprs[lane][0], 25)

    def test_wave_exec_mask_zero_preservation(self) -> None:
        wave = SIMDWaveState(wave_size=32)
        wave.exec_mask = 0x0000_0000
        for lane in range(32):
            wave.write_vgpr_lane(lane, 0, 0x9999_9999)
            wave.write_vgpr_lane(lane, 1, 0xFFFF_FFFF)
            wave.write_vgpr_lane(lane, 2, 0xFFFF_FFFF)

        wave.execute_asm(["v_dot4_u32_u8 v0, v1, v2, 0"])
        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][0], 0x9999_9999)

    def test_divergent_per_lane_vector_data(self) -> None:
        wave = SIMDWaveState(wave_size=32)
        # Give each lane distinct inputs based on its lane ID
        for lane in range(32):
            wave.write_vgpr_lane(lane, 1, pack_u8(lane, 1, 1, 1))
            wave.write_vgpr_lane(lane, 2, pack_u8(2, 2, 2, 2))
            wave.write_vgpr_lane(lane, 3, 10)

        wave.execute_asm(["v_dot4_u32_u8 v0, v1, v2, v3"])
        for lane in range(32):
            # (lane * 2) + (1 * 2) + (1 * 2) + (1 * 2) + 10 = 2*lane + 6 + 10 = 2*lane + 16
            expected = (2 * lane) + 16
            self.assertEqual(wave.vgprs[lane][0], expected)


# ---------------------------------------------------------------------------
# Register Aliasing and In-Place Operand Preservation Tests
# ---------------------------------------------------------------------------

class TestValuDot4IntRegisterAliasing(unittest.TestCase):
    """Register aliasing, self-overwrite, and in-place operand preservation unit tests."""

    def test_aliasing_dst_equals_src0(self) -> None:
        interp = Dot4IntScalarInterpreter()
        # v1 is both destination and src0
        initial_v1 = pack_u8(1, 2, 3, 4)
        v2 = pack_u8(10, 20, 30, 40)
        interp.rf.write_vgpr(1, initial_v1)
        interp.rf.write_vgpr(2, v2)

        # 1*10 + 2*20 + 3*30 + 4*40 = 10 + 40 + 90 + 160 = 300
        interp.execute_asm(["v_dot4_u32_u8 v1, v1, v2, 0"])
        self.assertEqual(interp.rf.vgprs[1], 300)

    def test_aliasing_dst_equals_src1(self) -> None:
        interp = Dot4IntScalarInterpreter()
        # v2 is both destination and src1
        v1 = pack_i8(-2, 3, -4, 5)
        v2 = pack_i8(6, -7, 8, -9)
        interp.rf.write_vgpr(1, v1)
        interp.rf.write_vgpr(2, v2)

        # (-2)*6 + 3*(-7) + (-4)*8 + 5*(-9) = -12 - 21 - 32 - 45 = -110
        expected = (-110) & MASK32
        interp.execute_asm(["v_dot4_i32_i8 v2, v1, v2, 0"])
        self.assertEqual(interp.rf.vgprs[2], expected)

    def test_aliasing_dst_equals_src2_accumulator(self) -> None:
        interp = Dot4IntScalarInterpreter()
        # v3 is accumulator and destination: in-place accumulation
        v1 = pack_u8(2, 2, 2, 2)
        v2 = pack_u8(3, 3, 3, 3)
        initial_acc = 500
        interp.rf.write_vgpr(1, v1)
        interp.rf.write_vgpr(2, v2)
        interp.rf.write_vgpr(3, initial_acc)

        # 4 * (2 * 3) = 24; 500 + 24 = 524
        interp.execute_asm(["v_dot4acc_u32_u8 v3, v1, v2, v3"])
        self.assertEqual(interp.rf.vgprs[3], 524)

    def test_aliasing_dst_equals_src0_and_src1_squared(self) -> None:
        interp = Dot4IntScalarInterpreter()
        # v1 is dst, src0, and src1: computes self dot-product (squared vector norm)
        v1 = pack_u8(3, 4, 5, 6)
        interp.rf.write_vgpr(1, v1)

        # 3^2 + 4^2 + 5^2 + 6^2 = 9 + 16 + 25 + 36 = 86; accumulator = 14 -> 100
        interp.execute_asm(["v_dot4_u32_u8 v1, v1, v1, 14"])
        self.assertEqual(interp.rf.vgprs[1], 100)

    def test_full_aliasing_dst_src0_src1_src2(self) -> None:
        interp = Dot4IntScalarInterpreter()
        # Single register v0 serves as dst, src0, src1, and src2
        # Let v0 initially be packed bytes [2, 2, 2, 2] = 0x02020202
        v0 = pack_u8(2, 2, 2, 2)
        interp.rf.write_vgpr(0, v0)

        # Products sum = 4 * (2 * 2) = 16
        # Accumulator value is the entire 32-bit dword 0x02020202
        # Expected = 0x02020202 + 16 = 0x02020212
        interp.execute_asm(["v_dot4acc_u32_u8 v0, v0, v0, v0"])
        self.assertEqual(interp.rf.vgprs[0], 0x0202_0212)

    def test_simd_multi_lane_aliased_accumulation(self) -> None:
        wave = SIMDWaveState(wave_size=32)
        # Even lanes active, odd inactive
        wave.exec_mask = 0x5555_5555
        for lane in range(32):
            wave.write_vgpr_lane(lane, 0, 1000)  # accumulator & destination
            wave.write_vgpr_lane(lane, 1, pack_u8(1, 2, 3, 4))
            wave.write_vgpr_lane(lane, 2, pack_u8(1, 1, 1, 1))

        # Active lanes: sum = 1+2+3+4 = 10; 1000 + 10 = 1010
        # Inactive lanes: remain 1000
        wave.execute_asm(["v_dot4acc_u32_u8 v0, v1, v2, v0"])
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[lane][0], 1010)
            else:
                self.assertEqual(wave.vgprs[lane][0], 1000)

    def test_sequential_accumulation_pipeline(self) -> None:
        interp = Dot4IntScalarInterpreter()
        # Initialize accumulator v0 to 0
        interp.rf.write_vgpr(0, 0)
        interp.rf.write_vgpr(1, pack_u8(1, 1, 1, 1))
        interp.rf.write_vgpr(2, pack_u8(2, 2, 2, 2))
        interp.rf.write_vgpr(3, pack_u8(3, 3, 3, 3))

        # Pipeline:
        # 1: v0 += v1 . v2 -> 0 + 4*(1*2) = 8
        # 2: v0 += v2 . v3 -> 8 + 4*(2*3) = 8 + 24 = 32
        # 3: v0 += v1 . v3 -> 32 + 4*(1*3) = 32 + 12 = 44
        pipeline = [
            "v_dot4acc_u32_u8 v0, v1, v2, v0",
            "v_dot4acc_u32_u8 v0, v2, v3, v0",
            "v_dot4acc_u32_u8 v0, v1, v3, v0",
        ]
        interp.execute_asm(pipeline)
        self.assertEqual(interp.rf.vgprs[0], 44)


# ---------------------------------------------------------------------------
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dot4StatusFlags:
    """Architectural status flags and condition codes for dot4 execution."""
    zero: bool
    negative: bool
    carry_out: bool
    overflow: bool
    saturated: bool


def compute_dot4_u32_status_flags(s0: int, s1: int, s2: int, *, clamp: bool = False) -> Dot4StatusFlags:
    """Compute status flags and condition codes for unsigned dot4 (v_dot4_u32_u8)."""
    b0 = unpack_u8(s0)
    b1 = unpack_u8(s1)
    prod_sum = sum(x * y for x, y in zip(b0, b1))
    acc = s2 & MASK32
    unclamped_sum = acc + prod_sum

    res = min(unclamped_sum, UINT32_MAX) if clamp else (unclamped_sum & MASK32)

    zero = (res == 0)
    negative = bool(res & 0x8000_0000)
    carry_out = unclamped_sum > UINT32_MAX
    overflow = carry_out
    saturated = clamp and (unclamped_sum > UINT32_MAX)

    return Dot4StatusFlags(
        zero=zero,
        negative=negative,
        carry_out=carry_out,
        overflow=overflow,
        saturated=saturated,
    )


def compute_dot4_i32_status_flags(s0: int, s1: int, s2: int, *, clamp: bool = False) -> Dot4StatusFlags:
    """Compute status flags and condition codes for signed dot4 (v_dot4_i32_i8)."""
    b0 = unpack_i8(s0)
    b1 = unpack_i8(s1)
    prod_sum = sum(x * y for x, y in zip(b0, b1))

    s2_u32 = s2 & MASK32
    s2_i32 = (s2_u32 - 0x1_0000_0000) if (s2_u32 & 0x8000_0000) else s2_u32

    unclamped_total = s2_i32 + prod_sum
    clamped_total = max(INT32_MIN, min(unclamped_total, INT32_MAX))

    res = (clamped_total if clamp else unclamped_total) & MASK32

    zero = (res == 0)
    negative = bool(res & 0x8000_0000)

    # 32-bit unsigned carry from the adder:
    adder_u32 = (s2 & MASK32) + (prod_sum & MASK32)
    carry_out = adder_u32 > MASK32

    # Signed overflow occurs if total falls outside 32-bit signed integer range
    overflow = unclamped_total < INT32_MIN or unclamped_total > INT32_MAX
    saturated = clamp and overflow

    return Dot4StatusFlags(
        zero=zero,
        negative=negative,
        carry_out=carry_out,
        overflow=overflow,
        saturated=saturated,
    )


def evaluate_vcc_predicate(results: list[int], predicate_fn) -> int:
    """Evaluate a per-lane condition across wavefront lanes, packing into a VCC mask."""
    vcc = 0
    for lane, val in enumerate(results):
        if predicate_fn(val):
            vcc |= (1 << lane)
    return vcc


class TestValuDot4IntConditionCodesAndStatusFlags(unittest.TestCase):
    """Status flags, condition codes, and carry/borrow generation unit tests for v_dot4_int."""

    def test_zero_flag_generation(self) -> None:
        # Zero operand dot product
        flags_zero = compute_dot4_u32_status_flags(0, 0, 0)
        self.assertTrue(flags_zero.zero)
        self.assertFalse(flags_zero.negative)
        self.assertFalse(flags_zero.carry_out)

        # Non-zero result
        flags_nonzero = compute_dot4_u32_status_flags(pack_u8(1, 0, 0, 0), pack_u8(1, 0, 0, 0), 0)
        self.assertFalse(flags_nonzero.zero)

        # Signed cancellation to zero: product = +70, accumulator = -70
        s0 = pack_i8(1, 2, 3, 4)
        s1 = pack_i8(5, 6, 7, 8)  # dot = 70
        acc = (-70) & MASK32
        flags_cancel = compute_dot4_i32_status_flags(s0, s1, acc)
        self.assertTrue(flags_cancel.zero)

    def test_negative_flag_generation(self) -> None:
        # Signed negative product: -28
        s0 = pack_i8(-1, 2, -3, -4)
        s1 = pack_i8(5, -6, -7, 8)
        flags_neg = compute_dot4_i32_status_flags(s0, s1, 0)
        self.assertTrue(flags_neg.negative)
        self.assertFalse(flags_neg.zero)

        # Unsigned large value with bit 31 set: e.g. 0x80000000
        flags_u_high = compute_dot4_u32_status_flags(0, 0, 0x8000_0000)
        self.assertTrue(flags_u_high.negative)

    def test_unsigned_carry_out_generation(self) -> None:
        # Accumulator near UINT32_MAX, adding 260100 causes carry_out
        s0 = 0xFFFF_FFFF
        s1 = 0xFFFF_FFFF  # product sum = 260100
        flags_carry = compute_dot4_u32_status_flags(s0, s1, 0xFFFF_FFFF)
        self.assertTrue(flags_carry.carry_out)
        self.assertTrue(flags_carry.overflow)

        # No carry out when sum fits in 32 bits
        flags_no_carry = compute_dot4_u32_status_flags(s0, s1, 1000)
        self.assertFalse(flags_no_carry.carry_out)
        self.assertFalse(flags_no_carry.overflow)

    def test_signed_overflow_generation(self) -> None:
        # Adding positive 65536 to INT32_MAX causes signed overflow
        s_min = pack_i8(-128, -128, -128, -128)  # (-128)^2 * 4 = 65536
        flags_pos_overflow = compute_dot4_i32_status_flags(s_min, s_min, INT32_MAX)
        self.assertTrue(flags_pos_overflow.overflow)
        self.assertFalse(flags_pos_overflow.saturated)  # clamp was False

        # Adding negative -65024 to INT32_MIN causes signed underflow
        s_max = pack_i8(127, 127, 127, 127)
        flags_neg_underflow = compute_dot4_i32_status_flags(s_min, s_max, INT32_MIN)
        self.assertTrue(flags_neg_underflow.overflow)

        # No overflow during ordinary operations
        flags_normal = compute_dot4_i32_status_flags(s_min, s_max, 0)
        self.assertFalse(flags_normal.overflow)

    def test_saturation_flag_generation(self) -> None:
        # Unsigned clamp saturation
        flags_u_sat = compute_dot4_u32_status_flags(0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, clamp=True)
        self.assertTrue(flags_u_sat.saturated)

        # Unsigned clamp without saturation
        flags_u_unsat = compute_dot4_u32_status_flags(0xFFFF_FFFF, 0xFFFF_FFFF, 100, clamp=True)
        self.assertFalse(flags_u_unsat.saturated)

        # Signed clamp saturation
        s_min = pack_i8(-128, -128, -128, -128)
        flags_i_sat = compute_dot4_i32_status_flags(s_min, s_min, INT32_MAX, clamp=True)
        self.assertTrue(flags_i_sat.saturated)

    def test_vcc_comparison_predicate_generation(self) -> None:
        # Multi-lane comparison generating VCC mask
        wave = SIMDWaveState(wave_size=32)
        for lane in range(32):
            wave.write_vgpr_lane(lane, 1, pack_u8(lane, 0, 0, 0))
            wave.write_vgpr_lane(lane, 2, pack_u8(1, 0, 0, 0))
            wave.write_vgpr_lane(lane, 3, 0)

        # Each lane computes: lane * 1 = lane
        wave.execute_asm(["v_dot4_u32_u8 v0, v1, v2, v3"])
        lane_results = [wave.vgprs[lane][0] for lane in range(32)]

        # Compare result >= 16: lanes 16..31 should be 1 in VCC
        vcc_ge_16 = evaluate_vcc_predicate(lane_results, lambda v: v >= 16)
        expected_mask = 0xFFFF_0000
        self.assertEqual(vcc_ge_16, expected_mask)


# ---------------------------------------------------------------------------
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling Tests
# ---------------------------------------------------------------------------

# IEEE 754 binary32 architectural bit patterns for cross-format validation
FLOAT_POS_ZERO: int = 0x0000_0000
FLOAT_NEG_ZERO: int = 0x8000_0000
FLOAT_ONE: int = 0x3F80_0000
FLOAT_NEG_ONE: int = 0xBF80_0000
FLOAT_POS_INF: int = 0x7F80_0000
FLOAT_NEG_INF: int = 0xFF80_0000
FLOAT_CANONICAL_QNAN: int = 0x7FC0_0000
FLOAT_SNAN: int = 0x7F80_0001
FLOAT_MIN_SUBNORMAL: int = 0x0000_0001
FLOAT_MAX_SUBNORMAL: int = 0x007F_FFFF
FLOAT_MAX_NORMAL: int = 0x7F7F_FFFF


class TestValuDot4IntIEEE754BoundaryExtremes(unittest.TestCase):
    """IEEE 754 floating-point boundary extremes, subnormals, and NaN handling tests for integer dot4."""

    def test_ieee754_qnan_snan_treated_as_raw_bytes(self) -> None:
        # FLOAT_CANONICAL_QNAN = 0x7FC00000 -> bytes: [0x00, 0x00, 0xC0, 0x7F]
        # In unsigned u8: [0, 0, 192, 127]
        # In signed i8:   [0, 0, -64, 127]
        # Multiplied by identity vector [1, 1, 1, 1] = 0x01010101
        ident = 0x0101_0101

        # Unsigned dot4: 0*1 + 0*1 + 192*1 + 127*1 = 319
        u_res = ref_dot4_u32_u8(FLOAT_CANONICAL_QNAN, ident, 0)
        self.assertEqual(u_res, 319)

        # Signed dot4: 0*1 + 0*1 + (-64)*1 + 127*1 = 63
        i_res = ref_dot4_i32_i8(FLOAT_CANONICAL_QNAN, ident, 0)
        self.assertEqual(i_res, 63)

        # Signaling NaN: 0x7F800001 -> bytes: [0x01, 0x00, 0x80, 0x7F]
        # Unsigned: 1*1 + 0*1 + 128*1 + 127*1 = 256
        snan_u = ref_dot4_u32_u8(FLOAT_SNAN, ident, 0)
        self.assertEqual(snan_u, 256)

        # Signed: 1*1 + 0*1 + (-128)*1 + 127*1 = 0
        snan_i = ref_dot4_i32_i8(FLOAT_SNAN, ident, 0)
        self.assertEqual(snan_i, 0)

    def test_ieee754_infinities_integer_dot_product(self) -> None:
        # FLOAT_POS_INF = 0x7F800000 -> bytes: [0x00, 0x00, 0x80, 0x7F]
        # FLOAT_NEG_INF = 0xFF800000 -> bytes: [0x00, 0x00, 0x80, 0xFF]
        ident = 0x0101_0101

        # Unsigned POS_INF dot [1,1,1,1] = 128 + 127 = 255
        self.assertEqual(ref_dot4_u32_u8(FLOAT_POS_INF, ident, 0), 255)

        # Unsigned NEG_INF dot [1,1,1,1] = 128 + 255 = 383
        self.assertEqual(ref_dot4_u32_u8(FLOAT_NEG_INF, ident, 0), 383)

        # Signed POS_INF dot [1,1,1,1] = (-128) + 127 = -1 (0xFFFFFFFF)
        self.assertEqual(ref_dot4_i32_i8(FLOAT_POS_INF, ident, 0), 0xFFFF_FFFF)

        # Signed NEG_INF dot [1,1,1,1] = (-128) + (-1) = -129
        self.assertEqual(ref_dot4_i32_i8(FLOAT_NEG_INF, ident, 0), (-129) & MASK32)

    def test_ieee754_subnormals_preservation_without_ftz(self) -> None:
        # In floating-point units with Flush-To-Zero (FTZ), subnormals are forced to 0.
        # In VALU integer dot4, subnormals MUST be preserved as their exact bit patterns!
        # FLOAT_MIN_SUBNORMAL = 0x00000001 -> bytes: [1, 0, 0, 0]
        # Multiplied by [5, 0, 0, 0]
        s0 = FLOAT_MIN_SUBNORMAL
        s1 = 0x0000_0005  # bytes: [5, 0, 0, 0]
        res = ref_dot4_u32_u8(s0, s1, 0)
        self.assertEqual(res, 5)

        # FLOAT_MAX_SUBNORMAL = 0x007FFFFF -> bytes: [0xFF, 0xFF, 0x7F, 0x00]
        # Unsigned: [255, 255, 127, 0] . [1, 1, 1, 1] = 255 + 255 + 127 + 0 = 637
        res_max_sub = ref_dot4_u32_u8(FLOAT_MAX_SUBNORMAL, 0x0101_0101, 0)
        self.assertEqual(res_max_sub, 637)

    def test_ieee754_negative_zero_sign_bit_differentiation(self) -> None:
        # In IEEE 754, +0.0 == -0.0.
        # But in integer dot4:
        # FLOAT_POS_ZERO = 0x00000000 -> bytes: [0, 0, 0, 0]
        # FLOAT_NEG_ZERO = 0x80000000 -> bytes: [0, 0, 0, 0x80]
        multiplier = 0x0101_0101  # [1, 1, 1, 1]

        pos_zero_res = ref_dot4_u32_u8(FLOAT_POS_ZERO, multiplier, 0)
        neg_zero_res = ref_dot4_u32_u8(FLOAT_NEG_ZERO, multiplier, 0)

        # Unsigned: +0.0 gives 0, -0.0 gives 128
        self.assertEqual(pos_zero_res, 0)
        self.assertEqual(neg_zero_res, 128)
        self.assertNotEqual(pos_zero_res, neg_zero_res)

        # Signed: +0.0 gives 0, -0.0 gives -128
        pos_zero_i = ref_dot4_i32_i8(FLOAT_POS_ZERO, multiplier, 0)
        neg_zero_i = ref_dot4_i32_i8(FLOAT_NEG_ZERO, multiplier, 0)
        self.assertEqual(pos_zero_i, 0)
        self.assertEqual(neg_zero_i, (-128) & MASK32)

    def test_ieee754_float_max_normal_byte_interaction(self) -> None:
        # FLOAT_MAX_NORMAL = 0x7F7FFFFF -> bytes: [0xFF, 0xFF, 0x7F, 0x7F]
        # Unsigned: [255, 255, 127, 127]
        # Dot with itself: 255^2 + 255^2 + 127^2 + 127^2 = 65025 + 65025 + 16129 + 16129 = 162308
        res = ref_dot4_u32_u8(FLOAT_MAX_NORMAL, FLOAT_MAX_NORMAL, 0)
        self.assertEqual(res, 162308)

        # Signed: [-1, -1, 127, 127]
        # Dot with itself: (-1)^2 + (-1)^2 + 127^2 + 127^2 = 1 + 1 + 16129 + 16129 = 32260
        res_signed = ref_dot4_i32_i8(FLOAT_MAX_NORMAL, FLOAT_MAX_NORMAL, 0)
        self.assertEqual(res_signed, 32260)

    def test_integer_extreme_min_max_signed_matrix(self) -> None:
        # Extreme signed matrix:
        # S_MIN = [-128, -128, -128, -128] = 0x80808080
        # S_MAX = [ 127,  127,  127,  127] = 0x7F7F7F7F
        s_min = 0x8080_8080
        s_max = 0x7F7F_7F7F

        # MIN . MIN = 4 * 16384 = +65536
        self.assertEqual(ref_dot4_i32_i8(s_min, s_min, 0), 65536)

        # MAX . MAX = 4 * 16129 = +64516
        self.assertEqual(ref_dot4_i32_i8(s_max, s_max, 0), 64516)

        # MIN . MAX = 4 * (-16256) = -65024
        self.assertEqual(ref_dot4_i32_i8(s_min, s_max, 0), (-65024) & MASK32)

    def test_integer_extreme_unsigned_overflow_and_saturation(self) -> None:
        s_max_u = 0xFFFF_FFFF  # [255, 255, 255, 255]
        # 4 * (255 * 255) = 260100
        # Wrap at UINT32_MAX
        res_wrap = ref_dot4_u32_u8(s_max_u, s_max_u, UINT32_MAX, clamp=False)
        self.assertEqual(res_wrap, 260099)

        # Saturation at UINT32_MAX
        res_sat = ref_dot4_u32_u8(s_max_u, s_max_u, UINT32_MAX, clamp=True)
        self.assertEqual(res_sat, UINT32_MAX)


# ---------------------------------------------------------------------------
# Negative Controls, Syntax Rejection, and Mutation Detection Tests
# ---------------------------------------------------------------------------

class TestValuDot4IntNegativeControls(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection unit tests."""

    def test_rejection_of_sgpr_as_valu_destination(self) -> None:
        interp = Dot4IntScalarInterpreter()
        # VALU cannot write to an SGPR destination
        with self.assertRaises(ValueError):
            interp.execute_asm(["v_dot4_u32_u8 s0, v1, v2, v3"])

    def test_rejection_of_out_of_bounds_registers(self) -> None:
        interp = Dot4IntScalarInterpreter()
        # VGPR >= 256
        with self.assertRaises(IndexError):
            interp.rf.read_operand(ASTOperand(kind=OperandKind.VGPR, reg_num=256))
        # SGPR >= 106
        with self.assertRaises(IndexError):
            interp.rf.read_operand(ASTOperand(kind=OperandKind.SGPR, reg_num=106))

    def test_rejection_of_unrecognized_opcode(self) -> None:
        with self.assertRaises(NotImplementedError):
            parse_instruction("v_dot4_unknown_op v0, v1, v2, v3")

    def test_rejection_of_malformed_instruction_syntax(self) -> None:
        # Too few operands
        with self.assertRaises(ValueError):
            parse_instruction("v_dot4_u32_u8 v0, v1")
        # Empty instruction
        with self.assertRaises(ValueError):
            parse_instruction("")

    def test_rejection_of_invalid_wave_size(self) -> None:
        # Only wave32 and wave64 are valid in RDNA2
        with self.assertRaises(ValueError):
            SIMDWaveState(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWaveState(wave_size=128)

    def test_mutation_bit_flip_sensitivity(self) -> None:
        # Every bit flip in S0 or S1 should produce a distinct result for non-zero multiplier
        s0 = pack_u8(10, 20, 30, 40)
        s1 = pack_u8(1, 2, 3, 4)
        acc = 100
        baseline = ref_dot4_u32_u8(s0, s1, acc)

        # Perturb each of the 32 bits of s0 individually
        for bit_pos in range(32):
            s0_mutated = s0 ^ (1 << bit_pos)
            mutated_res = ref_dot4_u32_u8(s0_mutated, s1, acc)
            self.assertNotEqual(
                mutated_res,
                baseline,
                f"Bit flip at bit {bit_pos} failed to perturb output",
            )

    def test_mutation_byte_endianness_divergence(self) -> None:
        # Swapping byte 0 and byte 1 in s0 must produce divergent dot product when s1 has different multipliers
        s0 = pack_u8(10, 20, 30, 40)
        s1 = pack_u8(1, 5, 1, 1)  # byte 0 weight = 1, byte 1 weight = 5
        base_dot = ref_dot4_u32_u8(s0, s1, 0)

        # Mutate endianness: swap byte 0 and byte 1
        s0_swapped = pack_u8(20, 10, 30, 40)
        mutated_dot = ref_dot4_u32_u8(s0_swapped, s1, 0)
        self.assertNotEqual(base_dot, mutated_dot)

    def test_mutation_signedness_divergence_detection(self) -> None:
        # Treating negative i8 values as u8 must strongly diverge
        s0 = pack_i8(-1, -2, -3, -4)
        s1 = pack_i8(1, 1, 1, 1)

        signed_res = ref_dot4_i32_i8(s0, s1, 0)
        unsigned_res = ref_dot4_u32_u8(s0, s1, 0)

        # Signed: (-1) + (-2) + (-3) + (-4) = -10 (0xFFFFFFF6)
        # Unsigned: 255 + 254 + 253 + 252 = 1014
        self.assertEqual(signed_res, (-10) & MASK32)
        self.assertEqual(unsigned_res, 1014)
        self.assertNotEqual(signed_res, unsigned_res)


if __name__ == "__main__":
    unittest.main()
