"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_rcp_f64.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  Operations on 64-bit double-precision floating-point quantities (binary64, IEEE 754)
  operate across aligned pairs of adjacent registers: v[d:d+1] from source operand
  v[s0:s0+1] (or scalar SGPR pair s[k:k+1], literal 64-bit constant, or inline constant).

  Instruction:
    v_rcp_f64 v[d:d+1], src0 [clamp] [omod]

  Reciprocal definition:
    D.f64 = 1.0 / S0.f64

  IEEE-754 binary64 encoding:
    - 1 sign bit (bit 63)
    - 11 exponent bits with bias 1023 (bits 62:52)
    - 52 mantissa / fraction bits (bits 51:0)

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
import enum
import math
import struct
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# IEEE-754 binary64 bit constants
F64_POS_ZERO = 0x0000000000000000
F64_NEG_ZERO = 0x8000000000000000
F64_POS_INF = 0x7FF0000000000000
F64_NEG_INF = 0xFFF0000000000000
F64_CANONICAL_QNAN = 0x7FF8000000000000
F64_QUIET_BIT = 0x0008000000000000

F64_EXP_MASK = 0x7FF0000000000000
F64_MANT_MASK = 0x000FFFFFFFFFFFFF
F64_SIGN_MASK = 0x8000000000000000

F64_ONE = 0x3FF0000000000000          # 1.0
F64_NEG_ONE = 0xBFF0000000000000      # -1.0
F64_TWO = 0x4000000000000000          # 2.0
F64_HALF = 0x3FE0000000000000         # 0.5
F64_FOUR = 0x4010000000000000         # 4.0
F64_QUARTER = 0x3FD0000000000000      # 0.25
F64_MIN_NORMAL = 0x0010000000000000   # 2^-1022
F64_MAX_NORMAL = 0x7FEFFFFFFFFFFFFF   # (2 - 2^-52) * 2^1023
F64_MIN_SUBNORMAL = 0x0000000000000001  # 2^-1074
F64_MAX_SUBNORMAL = 0x000FFFFFFFFFFFFF

OLD_VGPR_SENTINEL = 0xDEADBEEF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def f64_to_u64(val: float) -> int:
    """Pack an IEEE-754 double precision float into an unsigned 64-bit integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return F64_NEG_INF if val < 0 else F64_POS_INF


def u64_to_f64(bits: int) -> float:
    """Unpack an unsigned 64-bit integer into an IEEE-754 double precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def is_f64_nan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents an IEEE-754 NaN."""
    return ((bits & F64_EXP_MASK) == F64_EXP_MASK) and ((bits & F64_MANT_MASK) != 0)


def is_f64_snan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a signaling NaN."""
    return is_f64_nan(bits) and ((bits & F64_QUIET_BIT) == 0)


def is_f64_qnan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a quiet NaN."""
    return is_f64_nan(bits) and ((bits & F64_QUIET_BIT) != 0)


def is_f64_zero(bits: int) -> bool:
    """Return True if the 64-bit pattern represents +0.0 or -0.0."""
    return (bits & (F64_EXP_MASK | F64_MANT_MASK)) == 0


def is_f64_inf(bits: int) -> bool:
    """Return True if the 64-bit pattern represents +inf or -inf."""
    return ((bits & F64_EXP_MASK) == F64_EXP_MASK) and ((bits & F64_MANT_MASK) == 0)


def is_f64_subnormal(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a subnormal/denormal float."""
    return ((bits & F64_EXP_MASK) == 0) and ((bits & F64_MANT_MASK) != 0)


def is_f64_normal(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a normalized finite float."""
    exp = (bits >> 52) & 0x7FF
    return 0 < exp < 0x7FF


def f64_sign(bits: int) -> int:
    """Return the sign bit (0 or 1)."""
    return (bits >> 63) & 1


def quiet_f64(bits: int) -> int:
    """Force an IEEE-754 double NaN to quiet by setting the quiet bit."""
    return (bits & MASK64) | F64_QUIET_BIT


def apply_input_modifiers_f64(bits: int, *, neg: bool = False, abs_val: bool = False) -> int:
    """Apply VOP3 input modifiers (abs, neg).

    In RDNA2 ISA:
      When both abs and neg are set, abs is evaluated before neg:
      i.e. neg(abs(x)) forces the sign bit to 1 (negative).
    """
    res = bits & MASK64
    if abs_val:
        res &= ~F64_SIGN_MASK
    if neg:
        res ^= F64_SIGN_MASK
    return res


def apply_f64_clamp(bits: int) -> int:
    """Clamp double precision value to [0.0, 1.0]. NaN clamps to +0.0."""
    if is_f64_nan(bits):
        return F64_POS_ZERO
    val = u64_to_f64(bits)
    if val < 0.0 or is_f64_zero(bits):
        return F64_POS_ZERO
    elif val > 1.0 or is_f64_inf(bits):
        return F64_ONE
    return bits & MASK64


def ref_rcp_f64(
    src0_bits: int,
    *,
    clamp: bool = False,
    src0_neg: bool = False,
    src0_abs: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 double-precision reciprocal v_rcp_f64.

    Evaluates:
      D.f64 = 1.0 / S0.f64
    accounting for IEEE-754 binary64 rules, input modifiers (neg, abs), and clamp [0.0, 1.0].
    """
    s0 = apply_input_modifiers_f64(src0_bits, neg=src0_neg, abs_val=src0_abs)
    sign = f64_sign(s0)

    # NaN input yields quiet NaN
    if is_f64_nan(s0):
        res = quiet_f64(s0)
    # 1.0 / +-0.0 = +-Inf
    elif is_f64_zero(s0):
        res = F64_NEG_INF if sign else F64_POS_INF
    # 1.0 / +-Inf = +-0.0
    elif is_f64_inf(s0):
        res = F64_NEG_ZERO if sign else F64_POS_ZERO
    else:
        val0 = u64_to_f64(s0)
        try:
            recip = 1.0 / val0
            res = f64_to_u64(recip)
            if is_f64_zero(res):
                res = F64_NEG_ZERO if sign else F64_POS_ZERO
            elif is_f64_inf(res):
                res = F64_NEG_INF if sign else F64_POS_INF
        except OverflowError:
            res = F64_NEG_INF if sign else F64_POS_INF

    if clamp:
        res = apply_f64_clamp(res)

    return res & MASK64


def ref_rcp_f64_dwords(
    s0_lo: int,
    s0_hi: int,
    *,
    clamp: bool = False,
    src0_neg: bool = False,
    src0_abs: bool = False,
) -> tuple[int, int]:
    """Decomposed dword interface for v_rcp_f64."""
    s0 = join_dwords(s0_lo, s0_hi)
    res = ref_rcp_f64(s0, clamp=clamp, src0_neg=src0_neg, src0_abs=src0_abs)
    return split_dwords(res)


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

    def get_vgpr_pair(self, idx: int) -> int:
        if not (0 <= idx < self.num_vgprs - 1):
            raise IndexError(f"VGPR pair index out of range: v[{idx}:{idx+1}]")
        lo = self.get_vgpr(idx)
        hi = self.get_vgpr(idx + 1)
        return join_dwords(lo, hi)

    def set_vgpr_pair(self, idx: int, val64: int) -> None:
        if not (0 <= idx < self.num_vgprs - 1):
            raise IndexError(f"VGPR pair index out of range: v[{idx}:{idx+1}]")
        lo, hi = split_dwords(val64)
        self.set_vgpr(idx, lo)
        self.set_vgpr(idx + 1, hi)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions for v_rcp_f64
# ==============================================================================

class OperandKind(enum.Enum):
    VGPR_PAIR = "vgpr_pair"
    SGPR_PAIR = "sgpr_pair"
    LITERAL_64 = "literal_64"
    INLINE_IMM = "inline_imm"


@dataclass
class ASTOperand:
    kind: OperandKind
    reg_idx: int = 0
    imm_val: int = 0
    neg: bool = False
    abs_val: bool = False

    def format_asm(self) -> str:
        if self.kind == OperandKind.VGPR_PAIR:
            s = f"v[{self.reg_idx}:{self.reg_idx + 1}]"
        elif self.kind == OperandKind.SGPR_PAIR:
            s = f"s[{self.reg_idx}:{self.reg_idx + 1}]"
        elif self.kind == OperandKind.LITERAL_64:
            s = f"0x{self.imm_val:016x}"
        elif self.kind == OperandKind.INLINE_IMM:
            s = f"{self.imm_val}"
        else:
            s = "unknown"
        if self.abs_val:
            s = f"|{s}|"
        if self.neg:
            s = f"-{s}"
        return s


@dataclass
class ASTInstruction:
    opcode: str
    dst: ASTOperand
    src0: ASTOperand
    clamp: bool = False
    omod: int = 0

    def validate(self) -> None:
        if self.opcode != "v_rcp_f64":
            raise ValueError(f"Invalid opcode for v_rcp_f64 AST: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR_PAIR:
            raise ValueError(f"Destination must be a VGPR pair, got {self.dst.kind}")
        if not (0 <= self.dst.reg_idx <= 254):
            raise ValueError(f"Destination VGPR pair index out of range: {self.dst.reg_idx}")
        if self.dst.neg or self.dst.abs_val:
            raise ValueError("Destination operand cannot have neg or abs modifiers")
        if self.src0.kind == OperandKind.VGPR_PAIR:
            if not (0 <= self.src0.reg_idx <= 254):
                raise ValueError(f"src0 VGPR pair index out of range: {self.src0.reg_idx}")
        elif self.src0.kind == OperandKind.SGPR_PAIR:
            if not (0 <= self.src0.reg_idx <= 104):
                raise ValueError(f"src0 SGPR pair index out of range: {self.src0.reg_idx}")

    def format_asm(self) -> str:
        s = f"{self.opcode} {self.dst.format_asm()}, {self.src0.format_asm()}"
        if self.clamp:
            s += " clamp"
        return s


class F64Expression(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, env: dict[str, int]) -> int:
        pass


@dataclass
class F64Constant(F64Expression):
    bits: int

    def evaluate(self, env: dict[str, int]) -> int:
        return self.bits & MASK64


@dataclass
class F64Variable(F64Expression):
    name: str

    def evaluate(self, env: dict[str, int]) -> int:
        if self.name not in env:
            raise KeyError(f"Variable {self.name} not found in environment")
        return env[self.name] & MASK64


@dataclass
class F64UnaryOp(F64Expression):
    op: str  # "abs", "neg", "neg_abs"
    operand: F64Expression

    def evaluate(self, env: dict[str, int]) -> int:
        val = self.operand.evaluate(env)
        if self.op == "abs":
            return apply_input_modifiers_f64(val, abs_val=True)
        elif self.op == "neg":
            return apply_input_modifiers_f64(val, neg=True)
        elif self.op == "neg_abs":
            return apply_input_modifiers_f64(val, neg=True, abs_val=True)
        raise ValueError(f"Unknown unary operator: {self.op}")


@dataclass
class F64Rcp(F64Expression):
    operand: F64Expression
    clamp: bool = False

    def evaluate(self, env: dict[str, int]) -> int:
        val = self.operand.evaluate(env)
        return ref_rcp_f64(val, clamp=self.clamp)


def instruction_to_expression(inst: ASTInstruction) -> F64Rcp:
    """Translate an ASTInstruction to a typed symbolic expression tree."""
    inst.validate()

    def build_operand_expr(op: ASTOperand, prefix: str) -> F64Expression:
        if op.kind in (OperandKind.VGPR_PAIR, OperandKind.SGPR_PAIR):
            expr: F64Expression = F64Variable(f"{prefix}_{op.reg_idx}")
        elif op.kind in (OperandKind.LITERAL_64, OperandKind.INLINE_IMM):
            expr = F64Constant(op.imm_val)
        else:
            raise ValueError(f"Unsupported operand kind: {op.kind}")

        if op.abs_val and op.neg:
            expr = F64UnaryOp("neg_abs", expr)
        elif op.abs_val:
            expr = F64UnaryOp("abs", expr)
        elif op.neg:
            expr = F64UnaryOp("neg", expr)
        return expr

    src_expr = build_operand_expr(inst.src0, "src0")
    return F64Rcp(operand=src_expr, clamp=inst.clamp)


class TestSymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST definitions and typed expressions."""

    def test_ast_creation_and_formatting(self) -> None:
        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2)
        inst = ASTInstruction(opcode="v_rcp_f64", dst=dst, src0=src0)
        inst.validate()
        self.assertEqual(inst.format_asm(), "v_rcp_f64 v[0:1], v[2:3]")

        # With clamp and modifiers
        src0_mod = ASTOperand(kind=OperandKind.SGPR_PAIR, reg_idx=4, neg=True, abs_val=True)
        inst_clamp = ASTInstruction(opcode="v_rcp_f64", dst=dst, src0=src0_mod, clamp=True)
        inst_clamp.validate()
        self.assertEqual(inst_clamp.format_asm(), "v_rcp_f64 v[0:1], -|s[4:5]| clamp")

    def test_typed_expression_evaluation(self) -> None:
        c1 = F64Constant(F64_TWO)
        rcp_expr = F64Rcp(operand=c1)
        res = rcp_expr.evaluate({})
        self.assertEqual(u64_to_f64(res), 0.5)

        # Unary modifier
        neg_c1 = F64UnaryOp("neg", c1)
        res_neg = F64Rcp(operand=neg_c1).evaluate({})
        self.assertEqual(u64_to_f64(res_neg), -0.5)

    def test_instruction_to_expression(self) -> None:
        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2, neg=True)
        inst = ASTInstruction(opcode="v_rcp_f64", dst=dst, src0=src0, clamp=False)

        expr = instruction_to_expression(inst)
        env = {"src0_2": F64_FOUR}
        res = expr.evaluate(env)
        # 1.0 / (-4.0) = -0.25
        self.assertEqual(u64_to_f64(res), -0.25)

    def test_ast_validation_errors(self) -> None:
        # Invalid opcode
        bad_op = ASTInstruction(
            opcode="v_mul_f64",
            dst=ASTOperand(OperandKind.VGPR_PAIR, 0),
            src0=ASTOperand(OperandKind.VGPR_PAIR, 2),
        )
        with self.assertRaises(ValueError):
            bad_op.validate()

        # Destination cannot be SGPR
        bad_dst = ASTInstruction(
            opcode="v_rcp_f64",
            dst=ASTOperand(OperandKind.SGPR_PAIR, 0),
            src0=ASTOperand(OperandKind.VGPR_PAIR, 2),
        )
        with self.assertRaises(ValueError):
            bad_dst.validate()

        # Destination out of bounds
        oob_dst = ASTInstruction(
            opcode="v_rcp_f64",
            dst=ASTOperand(OperandKind.VGPR_PAIR, 256),
            src0=ASTOperand(OperandKind.VGPR_PAIR, 2),
        )
        with self.assertRaises(ValueError):
            oob_dst.validate()


class TestVRcpF64ReferenceOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_basic_reciprocal(self) -> None:
        # 1.0 / 1.0 = 1.0
        r = ref_rcp_f64(F64_ONE)
        self.assertEqual(u64_to_f64(r), 1.0)

        # 1.0 / 2.0 = 0.5
        r = ref_rcp_f64(F64_TWO)
        self.assertEqual(u64_to_f64(r), 0.5)

        # 1.0 / 0.5 = 2.0
        r = ref_rcp_f64(F64_HALF)
        self.assertEqual(u64_to_f64(r), 2.0)

        # 1.0 / 4.0 = 0.25
        r = ref_rcp_f64(F64_FOUR)
        self.assertEqual(u64_to_f64(r), 0.25)

        # 1.0 / -4.0 = -0.25
        r = ref_rcp_f64(f64_to_u64(-4.0))
        self.assertEqual(u64_to_f64(r), -0.25)

    def test_zero_reciprocal_rules(self) -> None:
        # 1.0 / +0.0 = +Inf
        self.assertEqual(ref_rcp_f64(F64_POS_ZERO), F64_POS_INF)
        # 1.0 / -0.0 = -Inf
        self.assertEqual(ref_rcp_f64(F64_NEG_ZERO), F64_NEG_INF)

    def test_inf_reciprocal_rules(self) -> None:
        # 1.0 / +Inf = +0.0
        self.assertEqual(ref_rcp_f64(F64_POS_INF), F64_POS_ZERO)
        # 1.0 / -Inf = -0.0
        self.assertEqual(ref_rcp_f64(F64_NEG_INF), F64_NEG_ZERO)

    def test_nan_reciprocal_rules(self) -> None:
        # QNaN propagated with quiet bit
        qnan = F64_CANONICAL_QNAN
        self.assertTrue(is_f64_nan(ref_rcp_f64(qnan)))
        self.assertTrue(is_f64_qnan(ref_rcp_f64(qnan)))

        # SNaN quieted
        snan = (F64_POS_INF | 0x0001000000000000) & ~F64_QUIET_BIT
        r_snan = ref_rcp_f64(snan)
        self.assertTrue(is_f64_nan(r_snan))
        self.assertTrue(is_f64_qnan(r_snan))

    def test_input_modifiers(self) -> None:
        # neg: 1.0 / neg(2.0) = -0.5
        r = ref_rcp_f64(F64_TWO, src0_neg=True)
        self.assertEqual(u64_to_f64(r), -0.5)

        # abs: 1.0 / abs(-2.0) = 0.5
        r = ref_rcp_f64(f64_to_u64(-2.0), src0_abs=True)
        self.assertEqual(u64_to_f64(r), 0.5)

        # neg + abs: neg(abs(-2.0)) = -2.0 -> 1.0 / -2.0 = -0.5
        r = ref_rcp_f64(f64_to_u64(-2.0), src0_neg=True, src0_abs=True)
        self.assertEqual(u64_to_f64(r), -0.5)

    def test_clamp_modifier(self) -> None:
        # In range [0.0, 1.0]: 1.0 / 2.0 = 0.5 -> 0.5
        r = ref_rcp_f64(F64_TWO, clamp=True)
        self.assertEqual(u64_to_f64(r), 0.5)

        # > 1.0 clamped to 1.0: 1.0 / 0.5 = 2.0 -> 1.0
        r = ref_rcp_f64(F64_HALF, clamp=True)
        self.assertEqual(r, F64_ONE)

        # +Inf clamped to 1.0: 1.0 / +0.0 = +Inf -> 1.0
        r = ref_rcp_f64(F64_POS_ZERO, clamp=True)
        self.assertEqual(r, F64_ONE)

        # Negative clamped to +0.0: 1.0 / -2.0 = -0.5 -> +0.0
        r = ref_rcp_f64(f64_to_u64(-2.0), clamp=True)
        self.assertEqual(r, F64_POS_ZERO)

        # NaN clamped to +0.0
        r = ref_rcp_f64(F64_CANONICAL_QNAN, clamp=True)
        self.assertEqual(r, F64_POS_ZERO)

    def test_dword_decomposition(self) -> None:
        s0 = f64_to_u64(12.5)
        lo, hi = split_dwords(s0)
        r_lo, r_hi = ref_rcp_f64_dwords(lo, hi)
        res = join_dwords(r_lo, r_hi)
        self.assertEqual(u64_to_f64(res), 0.08)

    def test_register_file_pair_access(self) -> None:
        rf = VALURegisterFile(num_vgprs=16)
        val = 0x01234567_89ABCDEF
        rf.set_vgpr_pair(4, val)

        self.assertEqual(rf.get_vgpr(4), 0x89ABCDEF)
        self.assertEqual(rf.get_vgpr(5), 0x01234567)
        self.assertEqual(rf.get_vgpr_pair(4), val)

        with self.assertRaises(IndexError):
            rf.get_vgpr_pair(15)


if __name__ == "__main__":
    unittest.main()
