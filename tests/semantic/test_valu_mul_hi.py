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


# ==============================================================================
# Deterministic Scalar Instruction Interpreter for v_mul_hi
# ==============================================================================

def parse_asm_operand(tok: str) -> ASTOperand:
    tok = tok.strip()
    if tok.startswith("v") and tok[1:].isdigit():
        return ASTOperand(kind=OperandKind.VGPR, reg_idx=int(tok[1:]))
    elif tok.startswith("s") and tok[1:].isdigit():
        return ASTOperand(kind=OperandKind.SGPR, reg_idx=int(tok[1:]))
    elif tok.startswith("0x") or tok.startswith("0X"):
        return ASTOperand(kind=OperandKind.LITERAL, imm_val=int(tok, 16) & MASK32)
    else:
        try:
            val = int(tok)
            return ASTOperand(kind=OperandKind.INLINE_INT, imm_val=val & MASK32)
        except ValueError:
            raise ValueError(f"Unrecognized assembly operand token: {tok}")


def parse_asm_line(line: str) -> ASTInstruction:
    clean = line.split(";")[0].split("//")[0].strip()
    tokens = [t for t in clean.replace(",", " ").split() if t]
    if not tokens:
        raise ValueError("Empty assembly line")
    op_str = tokens[0]
    if op_str == "v_mul_hi_u32":
        opcode = Opcode.V_MUL_HI_U32
    elif op_str == "v_mul_hi_i32":
        opcode = Opcode.V_MUL_HI_I32
    else:
        raise ValueError(f"Expected opcode v_mul_hi_u32 or v_mul_hi_i32, got {op_str}")

    if len(tokens) != 4:
        raise ValueError(f"{op_str} requires 3 operands (dst, src0, src1), got {len(tokens) - 1}: {tokens[1:]}")

    dst = parse_asm_operand(tokens[1])
    src0 = parse_asm_operand(tokens[2])
    src1 = parse_asm_operand(tokens[3])
    inst = ASTInstruction(opcode=opcode, dst=dst, src0=src0, src1=src1)
    inst.validate()
    return inst


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 VALU v_mul_hi."""

    def __init__(self, num_vgprs: int = 256, num_sgprs: int = 106) -> None:
        self.vgpr = VALURegisterFile(num_vgprs=num_vgprs)
        self.num_sgprs = num_sgprs
        self.sgprs = [0] * num_sgprs
        self.vcc = 0
        self.scc = 0

    def get_sgpr(self, idx: int) -> int:
        if not (0 <= idx < self.num_sgprs):
            raise IndexError(f"SGPR index out of range: {idx}")
        return self.sgprs[idx] & MASK32

    def set_sgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < self.num_sgprs):
            raise IndexError(f"SGPR index out of range: {idx}")
        self.sgprs[idx] = val & MASK32

    def resolve_operand(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.vgpr.get_vgpr(op.reg_idx)
        elif op.kind == OperandKind.SGPR:
            return self.get_sgpr(op.reg_idx)
        elif op.kind in (OperandKind.LITERAL, OperandKind.INLINE_INT):
            return op.imm_val & MASK32
        raise ValueError(f"Unknown operand kind: {op.kind}")

    def execute_ast(self, inst: ASTInstruction) -> int:
        inst.validate()
        s0 = self.resolve_operand(inst.src0)
        s1 = self.resolve_operand(inst.src1)
        if inst.opcode == Opcode.V_MUL_HI_U32:
            res = ref_mul_hi_u32(s0, s1)
        elif inst.opcode == Opcode.V_MUL_HI_I32:
            res = ref_mul_hi_i32(s0, s1)
        else:
            raise ValueError(f"Unsupported opcode: {inst.opcode}")
        self.vgpr.set_vgpr(inst.dst.reg_idx, res)
        return res

    def execute_asm(self, line: str) -> int:
        inst = parse_asm_line(line)
        return self.execute_ast(inst)


# ==============================================================================
# SIMD Wavefront Execution Model (Wave32 / Wave64) for v_mul_hi
# ==============================================================================

def ref_simd_mul_hi(
    opcode: Opcode,
    src0_lanes: list[int],
    src1_lanes: list[int],
    wave_size: int = 64,
    exec_mask: int | None = None,
    prev_dst: list[int] | None = None,
) -> list[int]:
    """Reference SIMD execution oracle across wavefront lanes."""
    if wave_size not in (32, 64):
        raise ValueError(f"Unsupported wave size: {wave_size}")
    if len(src0_lanes) != wave_size or len(src1_lanes) != wave_size:
        raise ValueError(f"Operand length must match wave_size ({wave_size})")

    active_mask = (1 << wave_size) - 1 if exec_mask is None else (exec_mask & ((1 << wave_size) - 1))
    fallback = [0] * wave_size if prev_dst is None else prev_dst
    if len(fallback) != wave_size:
        raise ValueError(f"prev_dst length must match wave_size ({wave_size})")

    output = list(fallback)
    fn = ref_mul_hi_u32 if opcode == Opcode.V_MUL_HI_U32 else ref_mul_hi_i32
    for lane in range(wave_size):
        if (active_mask >> lane) & 1:
            output[lane] = fn(src0_lanes[lane], src1_lanes[lane])
    return output


class SIMDWavefront:
    """SIMD wavefront simulator modeling multi-lane execution and EXEC masking."""

    def __init__(self, wave_size: int = 64, num_vgprs: int = 256) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self.num_vgprs = num_vgprs
        self.lanes = [VALURegisterFile(num_vgprs=num_vgprs) for _ in range(wave_size)]
        self.exec_mask = (1 << wave_size) - 1
        self.vcc = 0
        self.sgprs = [0] * 106

    def store_vgpr_wave(self, reg_idx: int, values: list[int]) -> None:
        if len(values) != self.wave_size:
            raise ValueError(f"Value count ({len(values)}) does not match wave_size ({self.wave_size})")
        for lane in range(self.wave_size):
            self.lanes[lane].set_vgpr(reg_idx, values[lane])

    def load_vgpr_wave(self, reg_idx: int) -> list[int]:
        return [self.lanes[lane].get_vgpr(reg_idx) for lane in range(self.wave_size)]

    def step_v_mul_hi(
        self,
        opcode: Opcode,
        dst_reg: int,
        src0_reg: int,
        src1_reg: int,
    ) -> list[int]:
        active_mask = self.exec_mask & ((1 << self.wave_size) - 1)
        fn = ref_mul_hi_u32 if opcode == Opcode.V_MUL_HI_U32 else ref_mul_hi_i32
        for lane in range(self.wave_size):
            if (active_mask >> lane) & 1:
                s0 = self.lanes[lane].get_vgpr(src0_reg)
                s1 = self.lanes[lane].get_vgpr(src1_reg)
                res = fn(s0, s1)
                self.lanes[lane].set_vgpr(dst_reg, res)
        return self.load_vgpr_wave(dst_reg)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Isolation for v_mul_hi
# ==============================================================================

@dataclass
class MulHiStatusFlags:
    zero: bool = False
    negative: bool = False
    overflow_32: bool = False


def compute_mul_hi_flags(opcode: Opcode, s0: int, s1: int, hi: int) -> MulHiStatusFlags:
    flags = MulHiStatusFlags()
    flags.zero = (hi == 0)
    flags.negative = bool((hi & SIGN32) != 0)

    if opcode == Opcode.V_MUL_HI_U32:
        # For unsigned, product overflows 32 bits if high dword is non-zero
        flags.overflow_32 = (hi != 0)
    elif opcode == Opcode.V_MUL_HI_I32:
        # For signed, product overflows 32 bits if 64-bit product doesn't fit in signed 32-bit:
        # which means hi is not a pure sign-extension of lo.
        lo, _ = ref_mul_pair_i32(s0, s1)
        expected_hi = 0xFFFFFFFF if (lo & SIGN32) else 0
        flags.overflow_32 = (hi != expected_hi)

    return flags


def ref_cmp_u32(op: str, s0: int, s1: int) -> bool:
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    if op in ("eq", "=="):
        return u0 == u1
    elif op in ("ne", "!="):
        return u0 != u1
    elif op in ("lt", "<"):
        return u0 < u1
    elif op in ("le", "<="):
        return u0 <= u1
    elif op in ("gt", ">"):
        return u0 > u1
    elif op in ("ge", ">="):
        return u0 >= u1
    raise ValueError(f"Unknown comparison operator: {op}")


def ref_cmp_i32(op: str, s0: int, s1: int) -> bool:
    i0 = to_i32(s0)
    i1 = to_i32(s1)
    if op in ("eq", "=="):
        return i0 == i1
    elif op in ("ne", "!="):
        return i0 != i1
    elif op in ("lt", "<"):
        return i0 < i1
    elif op in ("le", "<="):
        return i0 <= i1
    elif op in ("gt", ">"):
        return i0 > i1
    elif op in ("ge", ">="):
        return i0 >= i1
    raise ValueError(f"Unknown comparison operator: {op}")


def generate_condition_mask(
    op: str, is_signed: bool, s0_lanes: list[int], s1_lanes: list[int]
) -> int:
    mask = 0
    fn = ref_cmp_i32 if is_signed else ref_cmp_u32
    for lane, (s0, s1) in enumerate(zip(s0_lanes, s1_lanes)):
        if fn(op, s0, s1):
            mask |= (1 << lane)
    return mask


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


class TestScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def test_execute_v_mul_hi_u32_vgpr_sources(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr(1, 0x80000000)
        interp.vgpr.set_vgpr(2, 2)

        dst = ASTOperand(kind=OperandKind.VGPR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR, reg_idx=1)
        src1 = ASTOperand(kind=OperandKind.VGPR, reg_idx=2)
        inst = ASTInstruction(opcode=Opcode.V_MUL_HI_U32, dst=dst, src0=src0, src1=src1)

        res = interp.execute_ast(inst)
        self.assertEqual(res, 1)
        self.assertEqual(interp.vgpr.get_vgpr(0), 1)

    def test_execute_v_mul_hi_i32_vgpr_sources(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr(1, to_u32(-10))
        interp.vgpr.set_vgpr(2, to_u32(20))

        dst = ASTOperand(kind=OperandKind.VGPR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR, reg_idx=1)
        src1 = ASTOperand(kind=OperandKind.VGPR, reg_idx=2)
        inst = ASTInstruction(opcode=Opcode.V_MUL_HI_I32, dst=dst, src0=src0, src1=src1)

        res = interp.execute_ast(inst)
        # -200 -> negative 64-bit product -> hi dword = 0xFFFFFFFF
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(interp.vgpr.get_vgpr(0), 0xFFFFFFFF)

    def test_execute_sgpr_and_literal_sources(self) -> None:
        interp = ScalarInterpreter()
        interp.set_sgpr(4, 0x10000)
        interp.vgpr.set_vgpr(2, 0x20000)

        # v0 = v_mul_hi_u32 v2, s4 -> (0x20000 * 0x10000) >> 32 = 0x200000000 >> 32 = 2
        res = interp.execute_asm("v_mul_hi_u32 v0, v2, s4")
        self.assertEqual(res, 2)
        self.assertEqual(interp.vgpr.get_vgpr(0), 2)

        # literal multiplication
        res_lit = interp.execute_asm("v_mul_hi_u32 v1, v0, 0x80000000")
        # 2 * 0x80000000 = 0x100000000 -> hi = 1
        self.assertEqual(res_lit, 1)
        self.assertEqual(interp.vgpr.get_vgpr(1), 1)

    def test_execute_i32_assembly(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr(5, 0x80000000)  # -2^31
        interp.vgpr.set_vgpr(6, 0x80000000)  # -2^31

        # (-2^31) * (-2^31) = +2^62 = 0x40000000_00000000 -> hi = 0x40000000
        res = interp.execute_asm("v_mul_hi_i32 v0, v5, v6")
        self.assertEqual(res, 0x40000000)
        self.assertEqual(interp.vgpr.get_vgpr(0), 0x40000000)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_active_u32(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        s0_vals = [0x10000000 * (i + 1) for i in range(32)]
        s1_vals = [0x10 for _ in range(32)]
        sim.store_vgpr_wave(1, s0_vals)
        sim.store_vgpr_wave(2, s1_vals)

        res = sim.step_v_mul_hi(Opcode.V_MUL_HI_U32, 0, 1, 2)
        for i in range(32):
            expected = ref_mul_hi_u32(s0_vals[i], s1_vals[i])
            self.assertEqual(res[i], expected)

    def test_wave64_all_active_i32(self) -> None:
        sim = SIMDWavefront(wave_size=64)
        s0_vals = [to_u32(-1000000 * (i + 1)) for i in range(64)]
        s1_vals = [to_u32(2000) for _ in range(64)]
        sim.store_vgpr_wave(1, s0_vals)
        sim.store_vgpr_wave(2, s1_vals)

        res = sim.step_v_mul_hi(Opcode.V_MUL_HI_I32, 0, 1, 2)
        for i in range(64):
            expected = ref_mul_hi_i32(s0_vals[i], s1_vals[i])
            self.assertEqual(res[i], expected)

    def test_wave32_divergent_exec_mask(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        sim.store_vgpr_wave(0, [OLD_VGPR_SENTINEL] * 32)
        s0_vals = [0x80000000] * 32
        s1_vals = [2] * 32
        sim.store_vgpr_wave(1, s0_vals)
        sim.store_vgpr_wave(2, s1_vals)

        # Alternating active lanes: 0b0101...
        sim.exec_mask = 0x55555555
        res = sim.step_v_mul_hi(Opcode.V_MUL_HI_U32, 0, 1, 2)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(res[lane], 1)
            else:
                self.assertEqual(res[lane], OLD_VGPR_SENTINEL)

    def test_wave64_divergent_exec_mask(self) -> None:
        sim = SIMDWavefront(wave_size=64)
        sim.store_vgpr_wave(0, [OLD_VGPR_SENTINEL] * 64)
        s0_vals = [0xFFFFFFFF] * 64
        s1_vals = [0xFFFFFFFF] * 64
        sim.store_vgpr_wave(1, s0_vals)
        sim.store_vgpr_wave(2, s1_vals)

        # Lower 32 lanes active, upper 32 inactive
        sim.exec_mask = 0x00000000_FFFFFFFF
        res = sim.step_v_mul_hi(Opcode.V_MUL_HI_U32, 0, 1, 2)

        for lane in range(32):
            self.assertEqual(res[lane], 0xFFFFFFFE)
        for lane in range(32, 64):
            self.assertEqual(res[lane], OLD_VGPR_SENTINEL)

    def test_wave32_upper_exec_bits_ignored(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        sim.store_vgpr_wave(0, [OLD_VGPR_SENTINEL] * 32)
        sim.store_vgpr_wave(1, [0x80000000] * 32)
        sim.store_vgpr_wave(2, [4] * 32)

        # Upper 32 bits should not affect wave32
        sim.exec_mask = 0xDEADBEEF_FFFFFFFF
        res = sim.step_v_mul_hi(Opcode.V_MUL_HI_U32, 0, 1, 2)
        for lane in range(32):
            self.assertEqual(res[lane], 2)

    def test_simd_oracle_validation(self) -> None:
        wave_size = 64
        s0_vals = [(i * 0x04000000 + 0x12345) & MASK32 for i in range(wave_size)]
        s1_vals = [((63 - i) * 0x02000000 + 0x6789A) & MASK32 for i in range(wave_size)]
        exec_mask = 0xAAAAAAAAAAAAAAAA
        prev_dst = [0xCAFE0000 | i for i in range(wave_size)]

        expected = ref_simd_mul_hi(
            Opcode.V_MUL_HI_U32, s0_vals, s1_vals, wave_size=wave_size, exec_mask=exec_mask, prev_dst=prev_dst
        )

        sim = SIMDWavefront(wave_size=wave_size)
        sim.store_vgpr_wave(0, prev_dst)
        sim.store_vgpr_wave(1, s0_vals)
        sim.store_vgpr_wave(2, s1_vals)
        sim.exec_mask = exec_mask

        actual = sim.step_v_mul_hi(Opcode.V_MUL_HI_U32, 0, 1, 2)
        self.assertEqual(actual, expected)


class TestRegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_inplace_dst_equals_src0_u32(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr(0, 0x80000000)
        interp.vgpr.set_vgpr(1, 4)

        # v0 = v_mul_hi_u32 v0, v1 -> (0x80000000 * 4) >> 32 = 0x200000000 >> 32 = 2
        res = interp.execute_asm("v_mul_hi_u32 v0, v0, v1")
        self.assertEqual(res, 2)
        self.assertEqual(interp.vgpr.get_vgpr(0), 2)
        # v1 must remain untouched
        self.assertEqual(interp.vgpr.get_vgpr(1), 4)

    def test_inplace_dst_equals_src1_u32(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr(0, 0x80000000)
        interp.vgpr.set_vgpr(1, 6)

        # v1 = v_mul_hi_u32 v0, v1 -> (0x80000000 * 6) >> 32 = 3
        res = interp.execute_asm("v_mul_hi_u32 v1, v0, v1")
        self.assertEqual(res, 3)
        self.assertEqual(interp.vgpr.get_vgpr(1), 3)
        # v0 must remain untouched
        self.assertEqual(interp.vgpr.get_vgpr(0), 0x80000000)

    def test_inplace_self_squaring_u32(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr(2, UINT32_MAX)

        # v2 = v_mul_hi_u32 v2, v2 -> (0xFFFFFFFF)^2 >> 32 = 0xFFFFFFFE
        res = interp.execute_asm("v_mul_hi_u32 v2, v2, v2")
        self.assertEqual(res, 0xFFFFFFFE)
        self.assertEqual(interp.vgpr.get_vgpr(2), 0xFFFFFFFE)

    def test_inplace_dst_equals_src0_i32(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr(3, to_u32(-2))
        interp.vgpr.set_vgpr(4, 10)

        # v3 = v_mul_hi_i32 v3, v4 -> (-2 * 10) = -20 -> hi = 0xFFFFFFFF
        res = interp.execute_asm("v_mul_hi_i32 v3, v3, v4")
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(interp.vgpr.get_vgpr(3), 0xFFFFFFFF)
        self.assertEqual(interp.vgpr.get_vgpr(4), 10)

    def test_inplace_self_squaring_i32(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr(5, 0x80000000)  # -2^31

        # v5 = v_mul_hi_i32 v5, v5 -> (-2^31)^2 = 2^62 = 0x40000000_00000000 -> hi = 0x40000000
        res = interp.execute_asm("v_mul_hi_i32 v5, v5, v5")
        self.assertEqual(res, 0x40000000)
        self.assertEqual(interp.vgpr.get_vgpr(5), 0x40000000)

    def test_simd_inplace_divergent_mask(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        init_vals = [0x80000000 for _ in range(32)]
        multiplier_vals = [i + 1 for i in range(32)]
        sim.store_vgpr_wave(0, init_vals)
        sim.store_vgpr_wave(1, multiplier_vals)

        # Active on even lanes only
        sim.exec_mask = 0x55555555
        res = sim.step_v_mul_hi(Opcode.V_MUL_HI_U32, 0, 0, 1)

        for lane in range(32):
            if lane % 2 == 0:
                expected = (0x80000000 * (lane + 1)) >> 32
                self.assertEqual(res[lane], expected)
            else:
                self.assertEqual(res[lane], init_vals[lane])


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Test suite for condition codes, status flags, and carry/borrow isolation."""

    def test_status_flags_zero_and_negative(self) -> None:
        # High word is 0
        r0 = ref_mul_hi_u32(100, 200)
        f0 = compute_mul_hi_flags(Opcode.V_MUL_HI_U32, 100, 200, r0)
        self.assertTrue(f0.zero)
        self.assertFalse(f0.negative)
        self.assertFalse(f0.overflow_32)

        # High word has sign bit set
        r1 = ref_mul_hi_i32(to_u32(-10), 2)
        f1 = compute_mul_hi_flags(Opcode.V_MUL_HI_I32, to_u32(-10), 2, r1)
        self.assertEqual(r1, 0xFFFFFFFF)
        self.assertFalse(f1.zero)
        self.assertTrue(f1.negative)
        self.assertFalse(f1.overflow_32)  # -20 fits in signed 32-bit!

    def test_status_flags_overflow_32(self) -> None:
        # Unsigned overflow 32 bits
        r_u = ref_mul_hi_u32(0x80000000, 2)
        f_u = compute_mul_hi_flags(Opcode.V_MUL_HI_U32, 0x80000000, 2, r_u)
        self.assertEqual(r_u, 1)
        self.assertTrue(f_u.overflow_32)

        # Signed overflow 32 bits: 2^30 * 4 = 2^32, does not fit in signed 32-bit (overflows)
        r_i = ref_mul_hi_i32(0x40000000, 4)
        f_i = compute_mul_hi_flags(Opcode.V_MUL_HI_I32, 0x40000000, 4, r_i)
        self.assertEqual(r_i, 1)
        self.assertTrue(f_i.overflow_32)

    def test_vcc_and_scc_preserved(self) -> None:
        interp = ScalarInterpreter()
        interp.vcc = 0x12345678_9ABCDEF0
        interp.scc = 1
        interp.vgpr.set_vgpr(1, 0x80000000)
        interp.vgpr.set_vgpr(2, 2)

        interp.execute_asm("v_mul_hi_u32 v0, v1, v2")
        self.assertEqual(interp.vcc, 0x12345678_9ABCDEF0)
        self.assertEqual(interp.scc, 1)

        sim = SIMDWavefront(wave_size=32)
        sim.vcc = 0xCAFEBABE
        sim.store_vgpr_wave(1, [0x80000000] * 32)
        sim.store_vgpr_wave(2, [2] * 32)
        sim.step_v_mul_hi(Opcode.V_MUL_HI_U32, 0, 1, 2)
        self.assertEqual(sim.vcc, 0xCAFEBABE)

    def test_condition_mask_generation(self) -> None:
        # Generate condition mask from high-word comparisons
        hi_words = [0, 1, 0xFFFFFFFF, 0x40000000]
        zeros = [0, 0, 0, 0]

        # Equal to zero -> lanes 0
        mask_eq = generate_condition_mask("==", False, hi_words, zeros)
        self.assertEqual(mask_eq, 0b0001)

        # Greater than zero (unsigned) -> lanes 1, 2, 3
        mask_gt_u = generate_condition_mask(">", False, hi_words, zeros)
        self.assertEqual(mask_gt_u, 0b1110)

        # Less than zero (signed) -> lane 2 (0xFFFFFFFF = -1)
        mask_lt_i = generate_condition_mask("<", True, hi_words, zeros)
        self.assertEqual(mask_lt_i, 0b0100)


class TestBoundaryExtremesAndBitPatterns(unittest.TestCase):
    """Test suite for boundary extremes, signed/unsigned wrap, and bit patterns."""

    def test_unsigned_powers_of_two_boundaries(self) -> None:
        # Exactly 2^32: high word = 1, low word = 0
        self.assertEqual(ref_mul_hi_u32(1 << 16, 1 << 16), 1)
        self.assertEqual(ref_mul_hi_u32(1 << 31, 2), 1)

        # 2^31 * 2^31 = 2^62 -> high word = 2^30 = 0x40000000
        self.assertEqual(ref_mul_hi_u32(1 << 31, 1 << 31), 0x40000000)

        # 2^31 - 1 with 2 -> (2^32 - 2) < 2^32 -> high word = 0
        self.assertEqual(ref_mul_hi_u32(0x7FFFFFFF, 2), 0)

    def test_unsigned_checkerboard_bit_patterns(self) -> None:
        p0 = 0x55555555
        p1 = 0xAAAAAAAA
        prod = p0 * p1
        expected_hi = (prod >> 32) & MASK32
        self.assertEqual(ref_mul_hi_u32(p0, p1), expected_hi)

    def test_signed_boundary_negation_and_wrap(self) -> None:
        # INT32_MIN (-2^31) multiplied by -1 gives +2^31 (0x80000000)
        # As 64-bit signed integer, this is +0x00000000_80000000 -> high word = 0
        self.assertEqual(ref_mul_hi_i32(0x80000000, 0xFFFFFFFF), 0)

        # INT32_MIN multiplied by 1 gives -2^31
        # As 64-bit signed integer, sign-extended to 0xFFFFFFFF_80000000 -> high word = 0xFFFFFFFF
        self.assertEqual(ref_mul_hi_i32(0x80000000, 1), 0xFFFFFFFF)

        # INT32_MIN * INT32_MIN = +2^62 = 0x40000000_00000000 -> high word = 0x40000000
        self.assertEqual(ref_mul_hi_i32(0x80000000, 0x80000000), 0x40000000)

        # INT32_MIN * INT32_MAX = (-2^31) * (2^31 - 1) = -2^62 + 2^31 = 0xC0000000_80000000
        self.assertEqual(ref_mul_hi_i32(0x80000000, 0x7FFFFFFF), 0xC0000000)

        # INT32_MAX * INT32_MAX = (2^31 - 1)^2 = 2^62 - 2^32 + 1 = 0x3FFFFFFF_00000001
        self.assertEqual(ref_mul_hi_i32(0x7FFFFFFF, 0x7FFFFFFF), 0x3FFFFFFF)

    def test_signed_checkerboard_patterns(self) -> None:
        p0 = 0x55555555  # positive
        p1 = 0xAAAAAAAA  # negative in signed two's-complement
        expected_hi = ((to_i32(p0) * to_i32(p1) & MASK64) >> 32) & MASK32
        self.assertEqual(ref_mul_hi_i32(p0, p1), expected_hi)

    def test_64bit_product_reconstruction_sweep(self) -> None:
        # Sweep edge cases and verify high-low word reconstruction
        test_values = [
            0, 1, 2, 0x7FFF, 0xFFFF, 0x10000, 0x7FFFFFFF, 0x80000000,
            0x80000001, 0xFFFFFFFE, 0xFFFFFFFF
        ]
        for a in test_values:
            for b in test_values[:4]:
                # Unsigned reconstruction
                lo_u, hi_u = ref_mul_pair_u32(a, b)
                reconstructed_u = (hi_u << 32) | lo_u
                self.assertEqual(reconstructed_u, a * b)

                # Signed reconstruction
                lo_i, hi_i = ref_mul_pair_i32(a, b)
                reconstructed_i = (hi_i << 32) | lo_i
                expected_i = (to_i32(a) * to_i32(b)) & MASK64
                self.assertEqual(reconstructed_i, expected_i)


# ==============================================================================
# Mutants for Mutation Detection
# ==============================================================================

def mutant_low_word_instead_of_high(opcode: Opcode, s0: int, s1: int) -> int:
    """Mutant 1: Returns the low 32 bits instead of high 32 bits."""
    if opcode == Opcode.V_MUL_HI_U32:
        return (s0 * s1) & MASK32
    else:
        return (to_i32(s0) * to_i32(s1)) & MASK32


def mutant_unsigned_for_signed(opcode: Opcode, s0: int, s1: int) -> int:
    """Mutant 2: Executes unsigned multiplication even when signed is requested."""
    return ref_mul_hi_u32(s0, s1)


def mutant_signed_for_unsigned(opcode: Opcode, s0: int, s1: int) -> int:
    """Mutant 3: Executes signed multiplication even when unsigned is requested."""
    return ref_mul_hi_i32(s0, s1)


def mutant_shift_31(opcode: Opcode, s0: int, s1: int) -> int:
    """Mutant 4: Shifts by 31 bits instead of 32."""
    if opcode == Opcode.V_MUL_HI_U32:
        return ((s0 * s1) >> 31) & MASK32
    else:
        return (((to_i32(s0) * to_i32(s1)) & MASK64) >> 31) & MASK32


def mutant_shift_33(opcode: Opcode, s0: int, s1: int) -> int:
    """Mutant 5: Shifts by 33 bits instead of 32."""
    if opcode == Opcode.V_MUL_HI_U32:
        return ((s0 * s1) >> 33) & MASK32
    else:
        return (((to_i32(s0) * to_i32(s1)) & MASK64) >> 33) & MASK32


def mutant_addition_operator(opcode: Opcode, s0: int, s1: int) -> int:
    """Mutant 6: Replaces multiplication with addition."""
    return (s0 + s1) & MASK32


class TestNegativeControlsAndMutation(unittest.TestCase):
    """Test suite for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_empty_lines(self) -> None:
        with self.assertRaises(ValueError):
            parse_asm_line("")
        with self.assertRaises(ValueError):
            parse_asm_line("   ; comment only")

    def test_syntax_rejection_invalid_opcode(self) -> None:
        for bad_op in ("v_mul_lo_u32", "v_mul_f32", "v_add_u32", "bad_opcode"):
            with self.assertRaises(ValueError):
                parse_asm_line(f"{bad_op} v0, v1, v2")

    def test_syntax_rejection_operand_count(self) -> None:
        with self.assertRaises(ValueError):
            parse_asm_line("v_mul_hi_u32 v0, v1")
        with self.assertRaises(ValueError):
            parse_asm_line("v_mul_hi_u32 v0, v1, v2, v3")

    def test_syntax_rejection_dst_sgpr(self) -> None:
        with self.assertRaises(ValueError):
            parse_asm_line("v_mul_hi_u32 s0, v1, v2")

    def test_syntax_rejection_out_of_bounds_registers(self) -> None:
        with self.assertRaises(ValueError):
            parse_asm_line("v_mul_hi_u32 v256, v1, v2")
        with self.assertRaises(ValueError):
            parse_asm_line("v_mul_hi_u32 v0, s110, v2")

    def test_syntax_rejection_malformed_tokens(self) -> None:
        with self.assertRaises(ValueError):
            parse_asm_line("v_mul_hi_u32 v0, bad_token_xyz, v2")

    def test_mutation_detects_low_word_instead_of_high(self) -> None:
        s0 = 0x80000000
        s1 = 4
        # (0x80000000 * 4) = 0x200000000 -> hi = 2, lo = 0
        correct = ref_mul_hi_u32(s0, s1)
        sabotaged = mutant_low_word_instead_of_high(Opcode.V_MUL_HI_U32, s0, s1)
        self.assertEqual(correct, 2)
        self.assertEqual(sabotaged, 0)
        self.assertNotEqual(sabotaged, correct)

    def test_mutation_detects_signedness_confusion(self) -> None:
        # -1 * -1 = +1: signed hi = 0, unsigned hi = 0xFFFFFFFE
        s0 = 0xFFFFFFFF
        s1 = 0xFFFFFFFF
        correct_i = ref_mul_hi_i32(s0, s1)
        sabotaged_i = mutant_unsigned_for_signed(Opcode.V_MUL_HI_I32, s0, s1)
        self.assertEqual(correct_i, 0)
        self.assertEqual(sabotaged_i, 0xFFFFFFFE)
        self.assertNotEqual(sabotaged_i, correct_i)

        correct_u = ref_mul_hi_u32(s0, s1)
        sabotaged_u = mutant_signed_for_unsigned(Opcode.V_MUL_HI_U32, s0, s1)
        self.assertEqual(correct_u, 0xFFFFFFFE)
        self.assertEqual(sabotaged_u, 0)
        self.assertNotEqual(sabotaged_u, correct_u)

    def test_mutation_detects_shift_displacement_error(self) -> None:
        s0 = 0x80000000
        s1 = 2
        # Product = 0x100000000 (1 << 32)
        correct = ref_mul_hi_u32(s0, s1)
        self.assertEqual(correct, 1)

        # Shift 31 gives 2
        sabotaged_31 = mutant_shift_31(Opcode.V_MUL_HI_U32, s0, s1)
        self.assertEqual(sabotaged_31, 2)
        self.assertNotEqual(sabotaged_31, correct)

        # Shift 33 gives 0
        sabotaged_33 = mutant_shift_33(Opcode.V_MUL_HI_U32, s0, s1)
        self.assertEqual(sabotaged_33, 0)
        self.assertNotEqual(sabotaged_33, correct)

    def test_mutation_detects_addition_instead_of_multiplication(self) -> None:
        s0 = 0x10000
        s1 = 0x10000
        correct = ref_mul_hi_u32(s0, s1)
        sabotaged = mutant_addition_operator(Opcode.V_MUL_HI_U32, s0, s1)
        self.assertEqual(correct, 1)
        self.assertEqual(sabotaged, 0x20000)
        self.assertNotEqual(sabotaged, correct)

    def test_mutation_detects_exec_mask_leakage(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        sim.store_vgpr_wave(0, [OLD_VGPR_SENTINEL] * 32)
        sim.store_vgpr_wave(1, [0x80000000] * 32)
        sim.store_vgpr_wave(2, [2] * 32)

        # Only lane 0 active
        sim.exec_mask = 0x00000001
        res = sim.step_v_mul_hi(Opcode.V_MUL_HI_U32, 0, 1, 2)

        self.assertEqual(res[0], 1)
        for lane in range(1, 32):
            self.assertEqual(res[lane], OLD_VGPR_SENTINEL, f"Inactive lane {lane} was mutated!")


if __name__ == "__main__":
    unittest.main()
