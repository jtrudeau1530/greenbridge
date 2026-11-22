"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_mul_hi_u32 and v_mul_hi_i32.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  The v_mul_hi_u32 and v_mul_hi_i32 instructions compute the high 32 bits of a 64-bit
  integer product of two 32-bit integer operands:
    v_mul_hi_u32 v[d], src0, src1 : D = (unsigned(src0) * unsigned(src1)) >> 32
    v_mul_hi_i32 v[d], src0, src1 : D = (signed(src0) * signed(src1)) >> 32

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
import enum
import struct
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
SIGN32 = 0x80000000

INT32_MIN = -0x80000000
INT32_MAX = 0x7FFFFFFF
UINT32_MAX = 0xFFFFFFFF

OLD_VGPR_SENTINEL = 0xDEADBEEF


def to_u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer [0, 2^32 - 1]."""
    return val & MASK32


def to_i32(val: int) -> int:
    """Interpret 32-bit integer as signed two's complement integer [-2^31, 2^31 - 1]."""
    u = val & MASK32
    return u - 0x100000000 if u & SIGN32 else u


def to_u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit integer [0, 2^64 - 1]."""
    return val & MASK64


def to_i64(val: int) -> int:
    """Interpret 64-bit integer as signed two's complement integer [-2^63, 2^63 - 1]."""
    u = val & MASK64
    return u - 0x10000000000000000 if u & 0x8000000000000000 else u


def ref_mul_hi_u32(src0: int, src1: int) -> int:
    """Reference mathematical oracle for RDNA2 unsigned 32-bit high multiply v_mul_hi_u32.

    Computes:
      D.u32 = ((src0.u32 * src1.u32) >> 32) & 0xFFFFFFFF
    """
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    prod64 = u0 * u1
    return (prod64 >> 32) & MASK32


def ref_mul_hi_i32(src0: int, src1: int) -> int:
    """Reference mathematical oracle for RDNA2 signed 32-bit high multiply v_mul_hi_i32.

    Computes:
      D.i32 = ((src0.i32 * src1.i32) >> 32) & 0xFFFFFFFF
    under signed two's-complement arithmetic.
    """
    i0 = to_i32(src0)
    i1 = to_i32(src1)
    prod_signed = i0 * i1
    prod_bits = prod_signed & MASK64
    return (prod_bits >> 32) & MASK32


def ref_mul_pair_u32(src0: int, src1: int) -> tuple[int, int]:
    """Decomposed low and high dword product for unsigned 32-bit multiplication.

    Returns (lo, hi) such that:
      (hi << 32) | lo == (src0.u32 * src1.u32)
    """
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    prod = u0 * u1
    lo = prod & MASK32
    hi = (prod >> 32) & MASK32
    return lo, hi


def ref_mul_pair_i32(src0: int, src1: int) -> tuple[int, int]:
    """Decomposed low and high dword product for signed 32-bit multiplication.

    Returns (lo, hi) representing the 64-bit signed two's complement product.
    """
    i0 = to_i32(src0)
    i1 = to_i32(src1)
    prod = i0 * i1
    prod_bits = prod & MASK64
    lo = prod_bits & MASK32
    hi = (prod_bits >> 32) & MASK32
    return lo, hi


class VALURegisterFile:
    """Architectural model for RDNA2 VALU 32-bit vector register file."""

    def __init__(self, num_vgprs: int = 256) -> None:
        self.num_vgprs = num_vgprs
        self.regs = [0] * num_vgprs

    def get_vgpr(self, idx: int) -> int:
        if not (0 <= idx < self.num_vgprs):
            raise IndexError(f"VGPR index out of range: {idx}")
        return self.regs[idx] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < self.num_vgprs):
            raise IndexError(f"VGPR index out of range: {idx}")
        self.regs[idx] = val & MASK32


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions for v_mul_hi
# ==============================================================================

class Opcode(enum.Enum):
    V_MUL_HI_U32 = "v_mul_hi_u32"
    V_MUL_HI_I32 = "v_mul_hi_i32"


class OperandKind(enum.Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"
    LITERAL = "literal"
    INLINE_INT = "inline_int"


@dataclass
class ASTOperand:
    kind: OperandKind
    reg_idx: int = 0
    imm_val: int = 0

    def format_asm(self) -> str:
        if self.kind == OperandKind.VGPR:
            return f"v{self.reg_idx}"
        elif self.kind == OperandKind.SGPR:
            return f"s{self.reg_idx}"
        elif self.kind == OperandKind.LITERAL:
            return f"0x{self.imm_val:08x}"
        elif self.kind == OperandKind.INLINE_INT:
            return str(self.imm_val)
        return "unknown"


@dataclass
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand

    def validate(self) -> None:
        if self.opcode not in (Opcode.V_MUL_HI_U32, Opcode.V_MUL_HI_I32):
            raise ValueError(f"Invalid opcode for v_mul_hi AST: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR:
            raise ValueError(f"Destination must be a VGPR, got {self.dst.kind}")
        if not (0 <= self.dst.reg_idx <= 255):
            raise ValueError(f"Destination VGPR index out of range: {self.dst.reg_idx}")
        for name, src in [("src0", self.src0), ("src1", self.src1)]:
            if src.kind == OperandKind.VGPR:
                if not (0 <= src.reg_idx <= 255):
                    raise ValueError(f"{name} VGPR index out of range: {src.reg_idx}")
            elif src.kind == OperandKind.SGPR:
                if not (0 <= src.reg_idx <= 105):
                    raise ValueError(f"{name} SGPR index out of range: {src.reg_idx}")

    def format_asm(self) -> str:
        return f"{self.opcode.value} {self.dst.format_asm()}, {self.src0.format_asm()}, {self.src1.format_asm()}"


class IntegerExpression(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, env: dict[str, int]) -> int:
        pass


@dataclass
class IntConstant(IntegerExpression):
    val: int

    def evaluate(self, env: dict[str, int]) -> int:
        return self.val & MASK32


@dataclass
class IntVariable(IntegerExpression):
    name: str

    def evaluate(self, env: dict[str, int]) -> int:
        if self.name not in env:
            raise KeyError(f"Variable {self.name} not found in environment")
        return env[self.name] & MASK32


@dataclass
class MulHiExpr(IntegerExpression):
    opcode: Opcode
    lhs: IntegerExpression
    rhs: IntegerExpression

    def evaluate(self, env: dict[str, int]) -> int:
        v_lhs = self.lhs.evaluate(env)
        v_rhs = self.rhs.evaluate(env)
        if self.opcode == Opcode.V_MUL_HI_U32:
            return ref_mul_hi_u32(v_lhs, v_rhs)
        elif self.opcode == Opcode.V_MUL_HI_I32:
            return ref_mul_hi_i32(v_lhs, v_rhs)
        raise ValueError(f"Unsupported opcode in MulHiExpr: {self.opcode}")


def instruction_to_expression(inst: ASTInstruction) -> MulHiExpr:
    """Translate an ASTInstruction to a typed symbolic expression tree."""
    inst.validate()

    def build_operand_expr(op: ASTOperand, prefix: str) -> IntegerExpression:
        if op.kind in (OperandKind.VGPR, OperandKind.SGPR):
            return IntVariable(f"{prefix}_{op.reg_idx}")
        elif op.kind in (OperandKind.LITERAL, OperandKind.INLINE_INT):
            return IntConstant(op.imm_val)
        raise ValueError(f"Unsupported operand kind: {op.kind}")

    lhs = build_operand_expr(inst.src0, "src0")
    rhs = build_operand_expr(inst.src1, "src1")
    return MulHiExpr(opcode=inst.opcode, lhs=lhs, rhs=rhs)


class TestVMulHiOracleAndModel(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_mul_hi_u32_basic(self) -> None:
        # 0x80000000 * 2 = 0x100000000 -> hi = 1
        self.assertEqual(ref_mul_hi_u32(0x80000000, 2), 1)

        # 0x10000 * 0x10000 = 0x100000000 -> hi = 1
        self.assertEqual(ref_mul_hi_u32(0x10000, 0x10000), 1)

        # Small products fit in 32 bits -> hi = 0
        self.assertEqual(ref_mul_hi_u32(100, 200), 0)
        self.assertEqual(ref_mul_hi_u32(0xFFFF, 0xFFFF), 0)

    def test_mul_hi_u32_extremes(self) -> None:
        # 0 * anything = 0
        self.assertEqual(ref_mul_hi_u32(0, UINT32_MAX), 0)
        self.assertEqual(ref_mul_hi_u32(UINT32_MAX, 0), 0)

        # UINT32_MAX * 1 -> hi = 0
        self.assertEqual(ref_mul_hi_u32(UINT32_MAX, 1), 0)

        # UINT32_MAX * 2 = 0x1FFFFFFFE -> hi = 1
        self.assertEqual(ref_mul_hi_u32(UINT32_MAX, 2), 1)

        # UINT32_MAX * UINT32_MAX = 0xFFFFFFFE_00000001 -> hi = 0xFFFFFFFE
        self.assertEqual(ref_mul_hi_u32(UINT32_MAX, UINT32_MAX), 0xFFFFFFFE)

    def test_mul_hi_i32_basic(self) -> None:
        # (+3) * (+4) = +12 -> hi = 0
        self.assertEqual(ref_mul_hi_i32(3, 4), 0)

        # (-1) * (-1) = +1 -> 64-bit 0x00000000_00000001 -> hi = 0
        self.assertEqual(ref_mul_hi_i32(0xFFFFFFFF, 0xFFFFFFFF), 0)

        # (-1) * (+1) = -1 -> 64-bit 0xFFFFFFFF_FFFFFFFF -> hi = 0xFFFFFFFF
        self.assertEqual(ref_mul_hi_i32(0xFFFFFFFF, 1), 0xFFFFFFFF)

        # (+10) * (-5) = -50 -> 64-bit negative -> hi = 0xFFFFFFFF
        self.assertEqual(ref_mul_hi_i32(10, to_u32(-5)), 0xFFFFFFFF)

    def test_mul_hi_i32_extremes(self) -> None:
        # INT32_MIN (-2^31) * 2 = -2^32 -> 64-bit 0xFFFFFFFF_00000000 -> hi = 0xFFFFFFFF
        self.assertEqual(ref_mul_hi_i32(0x80000000, 2), 0xFFFFFFFF)

        # INT32_MIN * -1 = +2^31 (0x80000000) -> fits in 32-bit positive -> hi = 0
        self.assertEqual(ref_mul_hi_i32(0x80000000, 0xFFFFFFFF), 0)

        # INT32_MIN * INT32_MIN = (-2^31) * (-2^31) = +2^62 = 0x40000000_00000000 -> hi = 0x40000000
        self.assertEqual(ref_mul_hi_i32(0x80000000, 0x80000000), 0x40000000)

        # INT32_MAX (2^31 - 1) * INT32_MAX = (2^31 - 1)^2 = 2^62 - 2^32 + 1
        # 64-bit = 0x3FFFFFFF_00000001 -> hi = 0x3FFFFFFF
        self.assertEqual(ref_mul_hi_i32(INT32_MAX, INT32_MAX), 0x3FFFFFFF)

        # INT32_MIN * INT32_MAX = (-2^31) * (2^31 - 1) = -2^62 + 2^31
        # 64-bit = 0xC0000000_80000000 -> hi = 0xC0000000
        self.assertEqual(ref_mul_hi_i32(0x80000000, INT32_MAX), 0xC0000000)

    def test_mul_pair_decomposition(self) -> None:
        s0 = 0x12345678
        s1 = 0x9ABCDEF0

        # Unsigned pair
        lo_u, hi_u = ref_mul_pair_u32(s0, s1)
        full_u = (hi_u << 32) | lo_u
        self.assertEqual(full_u, (s0 & MASK32) * (s1 & MASK32))

        # Signed pair
        lo_i, hi_i = ref_mul_pair_i32(s0, s1)
        full_i = (hi_i << 32) | lo_i
        expected_i = to_u64(to_i32(s0) * to_i32(s1))
        self.assertEqual(full_i, expected_i)

    def test_register_file_access(self) -> None:
        rf = VALURegisterFile(num_vgprs=64)
        rf.set_vgpr(5, 0x12345678)
        self.assertEqual(rf.get_vgpr(5), 0x12345678)

        with self.assertRaises(IndexError):
            rf.get_vgpr(64)
        with self.assertRaises(IndexError):
            rf.set_vgpr(-1, 0)


class TestSymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST definitions and typed expressions."""

    def test_ast_creation_and_formatting(self) -> None:
        dst = ASTOperand(kind=OperandKind.VGPR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR, reg_idx=1)
        src1 = ASTOperand(kind=OperandKind.SGPR, reg_idx=2)

        inst_u = ASTInstruction(opcode=Opcode.V_MUL_HI_U32, dst=dst, src0=src0, src1=src1)
        inst_u.validate()
        self.assertEqual(inst_u.format_asm(), "v_mul_hi_u32 v0, v1, s2")

        inst_i = ASTInstruction(opcode=Opcode.V_MUL_HI_I32, dst=dst, src0=src0, src1=src1)
        inst_i.validate()
        self.assertEqual(inst_i.format_asm(), "v_mul_hi_i32 v0, v1, s2")

    def test_typed_expression_evaluation(self) -> None:
        c1 = IntConstant(0x80000000)
        c2 = IntConstant(2)

        expr_u = MulHiExpr(opcode=Opcode.V_MUL_HI_U32, lhs=c1, rhs=c2)
        self.assertEqual(expr_u.evaluate({}), 1)

        expr_i = MulHiExpr(opcode=Opcode.V_MUL_HI_I32, lhs=c1, rhs=c2)
        # -2^31 * 2 = -2^32 -> hi = 0xFFFFFFFF
        self.assertEqual(expr_i.evaluate({}), 0xFFFFFFFF)

    def test_instruction_to_expression(self) -> None:
        dst = ASTOperand(kind=OperandKind.VGPR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR, reg_idx=1)
        src1 = ASTOperand(kind=OperandKind.LITERAL, imm_val=UINT32_MAX)
        inst = ASTInstruction(opcode=Opcode.V_MUL_HI_U32, dst=dst, src0=src0, src1=src1)

        expr = instruction_to_expression(inst)
        env = {"src0_1": UINT32_MAX}
        self.assertEqual(expr.evaluate(env), 0xFFFFFFFE)

    def test_ast_validation_errors(self) -> None:
        # Invalid destination kind
        bad_dst = ASTInstruction(
            opcode=Opcode.V_MUL_HI_U32,
            dst=ASTOperand(OperandKind.SGPR, reg_idx=0),
            src0=ASTOperand(OperandKind.VGPR, reg_idx=1),
            src1=ASTOperand(OperandKind.VGPR, reg_idx=2),
        )
        with self.assertRaises(ValueError):
            bad_dst.validate()

        # Destination VGPR out of bounds
        oob_dst = ASTInstruction(
            opcode=Opcode.V_MUL_HI_U32,
            dst=ASTOperand(OperandKind.VGPR, reg_idx=256),
            src0=ASTOperand(OperandKind.VGPR, reg_idx=1),
            src1=ASTOperand(OperandKind.VGPR, reg_idx=2),
        )
        with self.assertRaises(ValueError):
            oob_dst.validate()


if __name__ == "__main__":
    unittest.main()
