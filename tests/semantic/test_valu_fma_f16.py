from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_fma_f16.

Tests the half-precision fused multiply-add instruction:
    v_fma_f16 D, S0, S1, S2
Lowered semantics:
    D = (S0 * S1) + S2
with single-rounding precision per IEEE 754-2008 (binary16).
Supports input modifiers:
    - abs (|src|: clear sign bit)
    - neg (-src: invert sign bit)
Supports output modifiers:
    - clamp: saturates result to [0.0, 1.0], with NaN clamped to +0.0
    - omod: mul:2, mul:4, div:2
Architectural model:
    16-bit halfword registers mapped onto 32-bit VGPR/SGPR halves:
    [lo: bits 15..0, hi: bits 31..16].
"""

from dataclasses import dataclass
from enum import Enum, auto
from fractions import Fraction
import math
import struct
import sys
import unittest

# IEEE 754 binary16 bitwise architectural constants
F16_POS_ZERO = 0x0000
F16_NEG_ZERO = 0x8000
F16_ONE = 0x3C00
F16_NEG_ONE = 0xBC00
F16_TWO = 0x4000
F16_HALF = 0x3800
F16_POS_INF = 0x7C00
F16_NEG_INF = 0xFC00
F16_CANONICAL_QNAN = 0x7E00
F16_SNAN = 0x7C01
F16_MIN_SUBNORMAL = 0x0001       # 2^-24 ~= 5.96e-8
F16_MAX_SUBNORMAL = 0x03FF       # (1023/1024) * 2^-14
F16_MIN_NORMAL = 0x0400          # 2^-14 ~= 6.10e-5
F16_MAX_NORMAL = 0x7BFF          # 65504.0

SIGN_MASK_F16 = 0x8000
EXP_MASK_F16 = 0x7C00
MANT_MASK_F16 = 0x03FF
QNAN_BIT_F16 = 0x0200

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF


def f16_to_u16(val: float) -> int:
    """Pack Python float into unsigned 16-bit integer (IEEE 754 binary16)."""
    try:
        return struct.unpack("<H", struct.pack("<e", float(val)))[0]
    except OverflowError:
        return F16_POS_INF if val > 0 else F16_NEG_INF


def u16_to_f16(bits: int) -> float:
    """Unpack unsigned 16-bit integer into Python float (IEEE 754 binary16)."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def split_f16(bits: int) -> tuple[int, int, int]:
    """Split 16-bit word into (sign, 5-bit exponent, 10-bit mantissa)."""
    b = bits & MASK16
    return (b >> 15) & 1, (b >> 10) & 0x1F, b & MANT_MASK_F16


def is_nan_f16(bits: int) -> bool:
    """Check if 16-bit word is an IEEE 754 binary16 NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) != 0)


def is_snan_f16(bits: int) -> bool:
    """Check if 16-bit word is an IEEE 754 binary16 signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_qnan_f16(bits: int) -> bool:
    """Check if 16-bit word is an IEEE 754 binary16 quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """Check if 16-bit word is an IEEE 754 binary16 infinity (+/-)."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) == 0)


def is_zero_f16(bits: int) -> bool:
    """Check if 16-bit word is zero (+0.0 or -0.0)."""
    return (bits & 0x7FFF) == 0


def is_subnormal_f16(bits: int) -> bool:
    """Check if 16-bit word is an IEEE 754 binary16 subnormal (denormal)."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & MANT_MASK_F16) != 0)


def apply_src_mods_u16(bits: int, *, neg: bool = False, abs_: bool = False) -> int:
    """Apply bit-exact RDNA2 source modifiers (|src|, -src) to a 16-bit float bit pattern."""
    val = bits & MASK16
    if abs_:
        val &= ~SIGN_MASK_F16  # clear sign bit
    if neg:
        val ^= SIGN_MASK_F16   # invert sign bit
    return val


def apply_clamp_f16(bits: int) -> int:
    """Apply RDNA2 clamp modifier [0.0, 1.0] to float16, mapping NaNs to +0.0."""
    if is_nan_f16(bits):
        return F16_POS_ZERO
    val_f = u16_to_f16(bits)
    if val_f < 0.0:
        return F16_POS_ZERO
    if val_f > 1.0:
        return F16_ONE
    return bits & MASK16


def apply_omod_f16(bits: int, omod: int) -> int:
    """Apply RDNA2 output modifier: 0: none, 1: mul:2, 2: mul:4, 3: div:2."""
    if omod == 0:
        return bits & MASK16
    if is_nan_f16(bits):
        return bits & MASK16
    val = u16_to_f16(bits)
    if omod == 1:
        val *= 2.0
    elif omod == 2:
        val *= 4.0
    elif omod == 3:
        val *= 0.5
    else:
        raise ValueError(f"Invalid omod value: {omod}")
    return f16_to_u16(val)


def raw_fma_f16(u0: int, u1: int, u2: int) -> int:
    """Compute exact IEEE 754 half precision FMA (u0 * u1) + u2 with single rounding."""
    u0 &= MASK16
    u1 &= MASK16
    u2 &= MASK16

    # 1. NaN propagation: if any input is NaN, quiet and return
    for u in (u0, u1, u2):
        if is_snan_f16(u):
            return u | QNAN_BIT_F16
    for u in (u0, u1, u2):
        if is_qnan_f16(u):
            return u

    # 2. Invalid operations: Inf * 0 or 0 * Inf
    if (is_inf_f16(u0) and is_zero_f16(u1)) or (is_zero_f16(u0) and is_inf_f16(u1)):
        return F16_CANONICAL_QNAN

    # 3. Product is infinite
    prod_inf = is_inf_f16(u0) or is_inf_f16(u1)
    if prod_inf:
        prod_sign = ((u0 >> 15) ^ (u1 >> 15)) & 1
        if is_inf_f16(u2):
            u2_sign = (u2 >> 15) & 1
            if prod_sign != u2_sign:
                return F16_CANONICAL_QNAN
        return F16_NEG_INF if prod_sign else F16_POS_INF

    # 4. S2 is infinite
    if is_inf_f16(u2):
        return u2

    # 5. Product is zero
    if is_zero_f16(u0) or is_zero_f16(u1):
        prod_sign = ((u0 >> 15) ^ (u1 >> 15)) & 1
        if is_zero_f16(u2):
            u2_sign = (u2 >> 15) & 1
            if prod_sign == 1 and u2_sign == 1:
                return F16_NEG_ZERO
            return F16_POS_ZERO
        return u2

    # 6. Finite values: compute exact rational arithmetic with Fraction
    f0 = u16_to_f16(u0)
    f1 = u16_to_f16(u1)
    f2 = u16_to_f16(u2)
    frac_sum = Fraction(f0) * Fraction(f1) + Fraction(f2)

    if frac_sum == 0:
        return F16_POS_ZERO

    try:
        val_f = float(frac_sum)
        return f16_to_u16(val_f)
    except OverflowError:
        return F16_POS_INF if frac_sum > 0 else F16_NEG_INF


def ref_fma_f16(
    s0: float,
    s1: float,
    s2: float,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    abs0: bool = False,
    abs1: bool = False,
    abs2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> float:
    """Reference mathematical oracle for v_fma_f16 calculating (s0 * s1) + s2 in binary16."""
    u0 = apply_src_mods_u16(f16_to_u16(s0), neg=neg0, abs_=abs0)
    u1 = apply_src_mods_u16(f16_to_u16(s1), neg=neg1, abs_=abs1)
    u2 = apply_src_mods_u16(f16_to_u16(s2), neg=neg2, abs_=abs2)

    res_u16 = raw_fma_f16(u0, u1, u2)

    if omod != 0:
        res_u16 = apply_omod_f16(res_u16, omod)
    if clamp:
        res_u16 = apply_clamp_f16(res_u16)

    return u16_to_f16(res_u16)


def ref_fma_f16_u16(
    u0: int,
    u1: int,
    u2: int,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    abs0: bool = False,
    abs1: bool = False,
    abs2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> int:
    """Bit-exact reference mathematical oracle for v_fma_f16 using 16-bit integer words."""
    m0 = apply_src_mods_u16(u0, neg=neg0, abs_=abs0)
    m1 = apply_src_mods_u16(u1, neg=neg1, abs_=abs1)
    m2 = apply_src_mods_u16(u2, neg=neg2, abs_=abs2)

    res_u16 = raw_fma_f16(m0, m1, m2)

    if omod != 0:
        res_u16 = apply_omod_f16(res_u16, omod)
    if clamp:
        res_u16 = apply_clamp_f16(res_u16)

    return res_u16 & MASK16


class RegisterHalf:
    """Architectural model of a 16-bit halfword within a 32-bit register (lo or hi)."""

    def __init__(self, raw_u16: int = 0) -> None:
        self.raw = raw_u16 & MASK16

    @classmethod
    def from_f16(cls, val: float) -> RegisterHalf:
        return cls(f16_to_u16(val))

    @property
    def as_u16(self) -> int:
        return self.raw

    @property
    def as_f16(self) -> float:
        return u16_to_f16(self.raw)

    def set_u16(self, bits: int) -> None:
        self.raw = bits & MASK16

    def set_f16(self, val: float) -> None:
        self.set_u16(f16_to_u16(val))

    def __repr__(self) -> str:
        return f"RegisterHalf({self.raw:#06x}, f16={self.as_f16})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegisterHalf):
            return NotImplemented
        return self.raw == other.raw


class VGPRWord:
    """Architectural model of a 32-bit VGPR containing low and high 16-bit halfwords."""

    def __init__(self, dword: int = 0) -> None:
        self.lo = RegisterHalf(dword & MASK16)
        self.hi = RegisterHalf((dword >> 16) & MASK16)

    @property
    def dword(self) -> int:
        return (self.lo.as_u16) | ((self.hi.as_u16) << 16)

    @dword.setter
    def dword(self, val: int) -> None:
        self.lo.set_u16(val & MASK16)
        self.hi.set_u16((val >> 16) & MASK16)

    def __repr__(self) -> str:
        return f"VGPRWord(lo={self.lo.as_u16:#06x}, hi={self.hi.as_u16:#06x})"


class TestFmaF16MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural register models."""

    def test_basic_fma_identities(self):
        # 1.0 * 1.0 + 0.0 == 1.0
        self.assertEqual(ref_fma_f16(1.0, 1.0, 0.0), 1.0)
        # 2.0 * 3.0 + 4.0 == 10.0
        self.assertEqual(ref_fma_f16(2.0, 3.0, 4.0), 10.0)
        # -2.5 * 4.0 + 1.25 == -8.75
        self.assertEqual(ref_fma_f16(-2.5, 4.0, 1.25), -8.75)
        # 0.0 * 100.0 + 5.5 == 5.5
        self.assertEqual(ref_fma_f16(0.0, 100.0, 5.5), 5.5)

    def test_fused_precision_single_rounding(self):
        # FMA retains intermediate product precision without intermediate 16-bit rounding.
        # Let a = 1.0 + 2**-10 (0x3C01), b = 1.0 + 2**-10 (0x3C01), c = -(1.0 + 2**-9) (0xBC02)
        # (a * b) = 1.0 + 2**-9 + 2**-20
        # Unfused: (a * b) rounds to (1.0 + 2**-9) in f16, so (a * b) + c == 0.0
        # Fused: (a * b) + c retains 2**-20, which in f16 is subnormal 0x0010 != 0.0
        a = u16_to_f16(0x3C01)
        b = u16_to_f16(0x3C01)
        c = u16_to_f16(0xBC02)
        unfused = u16_to_f16(f16_to_u16(a * b)) + c
        fused = ref_fma_f16(a, b, c)
        self.assertEqual(unfused, 0.0)
        self.assertNotEqual(fused, 0.0)
        self.assertEqual(f16_to_u16(fused), 0x0010)

    def test_source_modifiers(self):
        # abs(|src|): clears sign bit
        self.assertEqual(ref_fma_f16(-3.5, 2.0, 1.0, abs0=True), 8.0)
        self.assertEqual(ref_fma_f16(3.5, -2.0, 1.0, abs1=True), 8.0)
        self.assertEqual(ref_fma_f16(3.5, 2.0, -1.0, abs2=True), 8.0)

        # neg(-src): inverts sign bit
        self.assertEqual(ref_fma_f16(3.5, 2.0, 1.0, neg0=True), -6.0)
        self.assertEqual(ref_fma_f16(3.5, 2.0, 1.0, neg1=True), -6.0)
        self.assertEqual(ref_fma_f16(3.5, 2.0, 1.0, neg2=True), 6.0)

        # combined neg and abs: -|src|
        self.assertEqual(ref_fma_f16(-3.5, 2.0, 1.0, neg0=True, abs0=True), -6.0)

    def test_output_modifiers_clamp_and_omod(self):
        # clamp saturates to [0.0, 1.0]
        self.assertEqual(ref_fma_f16(2.0, 3.0, 0.0, clamp=True), 1.0)
        self.assertEqual(ref_fma_f16(-2.0, 3.0, 0.0, clamp=True), 0.0)
        self.assertEqual(ref_fma_f16(0.5, 1.0, 0.25, clamp=True), 0.75)

        # omod: mul:2, mul:4, div:2
        self.assertEqual(ref_fma_f16(1.5, 2.0, 0.5, omod=1), 7.0)   # (3.5) * 2 = 7.0
        self.assertEqual(ref_fma_f16(1.5, 2.0, 0.5, omod=2), 14.0)  # (3.5) * 4 = 14.0
        self.assertEqual(ref_fma_f16(1.5, 2.0, 0.5, omod=3), 1.75)  # (3.5) * 0.5 = 1.75

        # omod combined with clamp: (omod applies before clamp)
        self.assertEqual(ref_fma_f16(0.125, 2.0, 0.0, omod=1, clamp=True), 0.5)
        self.assertEqual(ref_fma_f16(0.25, 2.0, 0.0, omod=1, clamp=True), 1.0)
        self.assertEqual(ref_fma_f16(0.5, 2.0, 0.0, omod=2, clamp=True), 1.0)

    def test_register_half_architecture(self):
        rh = RegisterHalf.from_f16(1.0)
        self.assertEqual(rh.as_f16, 1.0)
        self.assertEqual(rh.as_u16, F16_ONE)

        word = VGPRWord(0)
        word.lo.set_f16(1.0)
        word.hi.set_f16(2.0)
        self.assertEqual(word.lo.as_u16, F16_ONE)
        self.assertEqual(word.hi.as_u16, F16_TWO)
        self.assertEqual(word.dword, (F16_TWO << 16) | F16_ONE)

        # bit-exact oracle roundtrip
        u_res = ref_fma_f16_u16(F16_TWO, F16_HALF, F16_ONE)
        self.assertEqual(u_res, f16_to_u16(2.0))


class Opcode(Enum):
    V_FMA_F16 = "v_fma_f16"


class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    INLINE_CONST = auto()
    LITERAL_CONST = auto()


class HalfSel(Enum):
    LO = 0  # bits 15:0 (.l)
    HI = 1  # bits 31:16 (.h)

    def to_asm(self) -> str:
        return "l" if self == HalfSel.LO else "h"


class TypeKind(Enum):
    FLOAT16 = "f16"
    UINT16 = "u16"
    UINT32 = "u32"


class OMod(Enum):
    NONE = 0
    MUL2 = 1
    MUL4 = 2
    DIV2 = 3

    def to_asm(self) -> str:
        if self == OMod.MUL2:
            return "mul:2"
        elif self == OMod.MUL4:
            return "mul:4"
        elif self == OMod.DIV2:
            return "div:2"
        return ""


class ASTOperand:
    """Symbolic AST operand representing a 16-bit register half, constant, or literal with modifiers."""

    def __init__(
        self,
        kind: OperandKind,
        index: int = 0,
        half: HalfSel = HalfSel.LO,
        value: float | int | None = None,
        *,
        neg: bool = False,
        abs_: bool = False,
        type_kind: TypeKind = TypeKind.FLOAT16,
    ) -> None:
        self.kind = kind
        self.index = index
        self.half = half
        self.value = value
        self.neg = neg
        self.abs_ = abs_
        self.type_kind = type_kind

    @classmethod
    def vgpr(
        cls,
        index: int,
        half: HalfSel = HalfSel.LO,
        *,
        neg: bool = False,
        abs_: bool = False,
    ) -> ASTOperand:
        return cls(OperandKind.VGPR, index=index, half=half, neg=neg, abs_=abs_)

    @classmethod
    def sgpr(
        cls,
        index: int,
        half: HalfSel = HalfSel.LO,
        *,
        neg: bool = False,
        abs_: bool = False,
    ) -> ASTOperand:
        return cls(OperandKind.SGPR, index=index, half=half, neg=neg, abs_=abs_)

    @classmethod
    def inline_const(cls, value: float, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.INLINE_CONST, value=value, neg=neg, abs_=abs_)

    @classmethod
    def literal_const(cls, value: float, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.LITERAL_CONST, value=value, neg=neg, abs_=abs_)

    def disasm(self) -> str:
        if self.kind == OperandKind.VGPR:
            body = f"v{self.index}.{self.half.to_asm()}"
        elif self.kind == OperandKind.SGPR:
            body = f"s{self.index}.{self.half.to_asm()}"
        elif self.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            body = str(self.value)
        else:
            raise ValueError(f"Unknown operand kind: {self.kind}")

        if self.abs_:
            body = f"|{body}|"
        if self.neg:
            body = f"-{body}"
        return body

    def __repr__(self) -> str:
        return f"ASTOperand({self.disasm()}, type={self.type_kind.value})"


class ASTInstruction:
    """Symbolic AST instruction representation for v_fma_f16."""

    def __init__(
        self,
        opcode: Opcode,
        dst: ASTOperand,
        src0: ASTOperand,
        src1: ASTOperand,
        src2: ASTOperand,
        *,
        clamp: bool = False,
        omod: OMod = OMod.NONE,
    ) -> None:
        self.opcode = opcode
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.clamp = clamp
        self.omod = omod

    def validate(self) -> None:
        if self.opcode != Opcode.V_FMA_F16:
            raise ValueError(f"Invalid opcode for v_fma_f16 AST: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR:
            raise TypeError("Destination must be a VGPR")
        if self.dst.type_kind != TypeKind.FLOAT16:
            raise TypeError(f"Destination type must be FLOAT16, got {self.dst.type_kind}")
        if not (0 <= self.dst.index <= 255):
            raise ValueError(f"Destination VGPR index out of range: {self.dst.index}")
        if self.dst.neg or self.dst.abs_:
            raise ValueError("Destination operand cannot have neg/abs modifiers")

        for i, src in enumerate((self.src0, self.src1, self.src2)):
            if src.type_kind != TypeKind.FLOAT16:
                raise TypeError(f"Source operand {i} type must be FLOAT16, got {src.type_kind}")
            if src.kind == OperandKind.VGPR and not (0 <= src.index <= 255):
                raise ValueError(f"Source {i} VGPR index out of range: {src.index}")
            if src.kind == OperandKind.SGPR and not (0 <= src.index <= 105):
                raise ValueError(f"Source {i} SGPR index out of range: {src.index}")

    def disasm(self) -> str:
        parts = [
            self.opcode.value,
            f"{self.dst.disasm()},",
            f"{self.src0.disasm()},",
            f"{self.src1.disasm()},",
            self.src2.disasm(),
        ]
        if self.omod != OMod.NONE:
            parts.append(self.omod.to_asm())
        if self.clamp:
            parts.append("clamp")
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"ASTInstruction({self.disasm()})"


class TypedExpr:
    """Base class for typed symbolic expressions."""

    def __init__(self, type_kind: TypeKind = TypeKind.FLOAT16) -> None:
        self.type_kind = type_kind

    def evaluate(self, env: dict[str, float]) -> float:
        raise NotImplementedError

    def simplify(self) -> TypedExpr:
        return self


class VariableExpr(TypedExpr):
    """Named variable in symbolic expression tree (e.g. representing a register value)."""

    def __init__(self, name: str, type_kind: TypeKind = TypeKind.FLOAT16) -> None:
        super().__init__(type_kind)
        self.name = name

    def evaluate(self, env: dict[str, float]) -> float:
        if self.name not in env:
            raise KeyError(f"Variable {self.name} not found in evaluation environment")
        return env[self.name]

    def __repr__(self) -> str:
        return f"Var({self.name})"


class ConstantExpr(TypedExpr):
    """Constant literal in symbolic expression tree."""

    def __init__(self, value: float, type_kind: TypeKind = TypeKind.FLOAT16) -> None:
        super().__init__(type_kind)
        self.value = value

    def evaluate(self, env: dict[str, float]) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Const({self.value})"


class UnaryOpExpr(TypedExpr):
    """Unary modifier applied to a typed expression ('neg' or 'abs')."""

    def __init__(self, op: str, expr: TypedExpr, type_kind: TypeKind = TypeKind.FLOAT16) -> None:
        super().__init__(type_kind)
        if op not in ("neg", "abs"):
            raise ValueError(f"Unknown unary operator: {op}")
        self.op = op
        self.expr = expr

    def evaluate(self, env: dict[str, float]) -> float:
        val = self.expr.evaluate(env)
        if self.op == "abs":
            return math.fabs(val)
        elif self.op == "neg":
            return -val
        return val

    def simplify(self) -> TypedExpr:
        sub = self.expr.simplify()
        if isinstance(sub, ConstantExpr):
            if self.op == "abs":
                return ConstantExpr(math.fabs(sub.value), self.type_kind)
            elif self.op == "neg":
                return ConstantExpr(-sub.value, self.type_kind)
        return UnaryOpExpr(self.op, sub, self.type_kind)

    def __repr__(self) -> str:
        return f"{self.op}({self.expr})"


class FMA16Expr(TypedExpr):
    """Symbolic FMA ternary typed expression: (src0 * src1) + src2 with modifiers."""

    def __init__(
        self,
        src0: TypedExpr,
        src1: TypedExpr,
        src2: TypedExpr,
        *,
        clamp: bool = False,
        omod: OMod = OMod.NONE,
        type_kind: TypeKind = TypeKind.FLOAT16,
    ) -> None:
        super().__init__(type_kind)
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.clamp = clamp
        self.omod = omod

    def evaluate(self, env: dict[str, float]) -> float:
        v0 = self.src0.evaluate(env)
        v1 = self.src1.evaluate(env)
        v2 = self.src2.evaluate(env)
        return ref_fma_f16(v0, v1, v2, clamp=self.clamp, omod=self.omod.value)

    def simplify(self) -> TypedExpr:
        s0 = self.src0.simplify()
        s1 = self.src1.simplify()
        s2 = self.src2.simplify()
        if isinstance(s0, ConstantExpr) and isinstance(s1, ConstantExpr) and isinstance(s2, ConstantExpr):
            res = ref_fma_f16(s0.value, s1.value, s2.value, clamp=self.clamp, omod=self.omod.value)
            return ConstantExpr(res, self.type_kind)
        return FMA16Expr(s0, s1, s2, clamp=self.clamp, omod=self.omod, type_kind=self.type_kind)

    def __repr__(self) -> str:
        return f"FMA16({self.src0}, {self.src1}, {self.src2}, clamp={self.clamp}, omod={self.omod.name})"


class TestFmaF16SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_validation(self):
        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(1, HalfSel.LO),
            src1=ASTOperand.vgpr(2, HalfSel.HI),
            src2=ASTOperand.vgpr(3, HalfSel.LO),
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_fma_f16 v0.l, v1.l, v2.h, v3.l")

    def test_ast_with_modifiers_and_disasm(self):
        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(1, HalfSel.HI, neg=True, abs_=True),
            src1=ASTOperand.sgpr(2, HalfSel.LO),
            src2=ASTOperand.inline_const(1.0),
            clamp=True,
            omod=OMod.MUL2,
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_fma_f16 v0.l, -|v1.h|, s2.l, 1.0 mul:2 clamp")

    def test_ast_validation_errors(self):
        # Invalid destination kind
        bad_dst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(TypeError):
            bad_dst.validate()

        # Destination has modifiers (illegal in RDNA2)
        bad_dst_mod = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, neg=True),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            bad_dst_mod.validate()

        # Destination out of bounds
        bad_index = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(256),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            bad_index.validate()

    def test_typed_expressions_and_evaluation(self):
        # Symbolic: (a * b) + c
        expr = FMA16Expr(
            VariableExpr("a"),
            VariableExpr("b"),
            VariableExpr("c"),
        )
        env = {"a": 2.5, "b": 4.0, "c": -3.0}
        self.assertEqual(expr.evaluate(env), 7.0)

    def test_typed_expression_constant_folding(self):
        # FMA(Const(2.0), Const(3.0), Const(4.0)) -> Const(10.0)
        c_expr = FMA16Expr(
            ConstantExpr(2.0),
            ConstantExpr(3.0),
            ConstantExpr(4.0),
        )
        folded = c_expr.simplify()
        self.assertIsInstance(folded, ConstantExpr)
        self.assertEqual(folded.value, 10.0)

        # Unary folding
        u_expr = UnaryOpExpr("neg", ConstantExpr(5.5))
        u_folded = u_expr.simplify()
        self.assertIsInstance(u_folded, ConstantExpr)
        self.assertEqual(u_folded.value, -5.5)


class ScalarRegisterFile:
    """Register file for a single scalar lane with 256 32-bit VGPRs and 106 32-bit SGPRs."""

    def __init__(self) -> None:
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106

    def read_vgpr_u32(self, idx: int) -> int:
        return self.vgprs[idx]

    def write_vgpr_u32(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32

    def read_sgpr_u32(self, idx: int) -> int:
        return self.sgprs[idx]

    def write_sgpr_u32(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32

    def read_vgpr_half_u16(self, idx: int, half: HalfSel = HalfSel.LO) -> int:
        dword = self.vgprs[idx]
        return (dword >> 16) & MASK16 if half == HalfSel.HI else (dword & MASK16)

    def write_vgpr_half_u16(self, idx: int, half: HalfSel, val: int) -> None:
        current = self.vgprs[idx]
        u16 = val & MASK16
        if half == HalfSel.HI:
            self.vgprs[idx] = (current & 0x0000FFFF) | (u16 << 16)
        else:
            self.vgprs[idx] = (current & 0xFFFF0000) | u16

    def read_sgpr_half_u16(self, idx: int, half: HalfSel = HalfSel.LO) -> int:
        dword = self.sgprs[idx]
        return (dword >> 16) & MASK16 if half == HalfSel.HI else (dword & MASK16)

    def write_sgpr_half_u16(self, idx: int, half: HalfSel, val: int) -> None:
        current = self.sgprs[idx]
        u16 = val & MASK16
        if half == HalfSel.HI:
            self.sgprs[idx] = (current & 0x0000FFFF) | (u16 << 16)
        else:
            self.sgprs[idx] = (current & 0xFFFF0000) | u16

    def read_vgpr_half_f16(self, idx: int, half: HalfSel = HalfSel.LO) -> float:
        return u16_to_f16(self.read_vgpr_half_u16(idx, half))

    def write_vgpr_half_f16(self, idx: int, half: HalfSel, val: float) -> None:
        self.write_vgpr_half_u16(idx, half, f16_to_u16(val))

    def read_sgpr_half_f16(self, idx: int, half: HalfSel = HalfSel.LO) -> float:
        return u16_to_f16(self.read_sgpr_half_u16(idx, half))

    def write_sgpr_half_f16(self, idx: int, half: HalfSel, val: float) -> None:
        self.write_sgpr_half_u16(idx, half, f16_to_u16(val))


class FmaF16ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_fma_f16."""

    def __init__(self, reg_file: ScalarRegisterFile | None = None) -> None:
        self.reg_file = reg_file if reg_file is not None else ScalarRegisterFile()

    def resolve_source_u16(self, src: ASTOperand) -> int:
        if src.kind == OperandKind.VGPR:
            val_u16 = self.reg_file.read_vgpr_half_u16(src.index, src.half)
        elif src.kind == OperandKind.SGPR:
            val_u16 = self.reg_file.read_sgpr_half_u16(src.index, src.half)
        elif src.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            val_u16 = f16_to_u16(float(src.value)) if src.value is not None else F16_POS_ZERO
        else:
            raise ValueError(f"Unsupported operand kind: {src.kind}")

        return apply_src_mods_u16(val_u16, neg=src.neg, abs_=src.abs_)

    def execute(self, inst: ASTInstruction) -> None:
        inst.validate()
        u0 = self.resolve_source_u16(inst.src0)
        u1 = self.resolve_source_u16(inst.src1)
        u2 = self.resolve_source_u16(inst.src2)

        res_u16 = ref_fma_f16_u16(u0, u1, u2, clamp=inst.clamp, omod=inst.omod.value)
        self.reg_file.write_vgpr_half_u16(inst.dst.index, inst.dst.half, res_u16)


class TestFmaF16ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = FmaF16ScalarInterpreter(self.rf)

    def test_scalar_interpreter_vgprs(self):
        # v0.l = v1.l * v2.h + v3.l
        self.rf.write_vgpr_half_f16(1, HalfSel.LO, 3.0)
        self.rf.write_vgpr_half_f16(2, HalfSel.HI, 4.0)
        self.rf.write_vgpr_half_f16(3, HalfSel.LO, 5.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(1, HalfSel.LO),
            src1=ASTOperand.vgpr(2, HalfSel.HI),
            src2=ASTOperand.vgpr(3, HalfSel.LO),
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_half_f16(0, HalfSel.LO), 17.0)

    def test_scalar_interpreter_sgpr_and_constants(self):
        # v0.h = s0.l * v1.l + inline_const(10.0)
        self.rf.write_sgpr_half_f16(0, HalfSel.LO, 2.5)
        self.rf.write_vgpr_half_f16(1, HalfSel.LO, 4.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.HI),
            src0=ASTOperand.sgpr(0, HalfSel.LO),
            src1=ASTOperand.vgpr(1, HalfSel.LO),
            src2=ASTOperand.inline_const(10.0),
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_half_f16(0, HalfSel.HI), 20.0)

    def test_scalar_interpreter_modifiers(self):
        # v0.l = -|v1.l| * v2.l + 5.0 with clamp
        # v1.l = -2.0 -> abs -> 2.0 -> neg -> -2.0
        # -2.0 * 3.0 + 5.0 = -1.0 -> clamp -> 0.0
        self.rf.write_vgpr_half_f16(1, HalfSel.LO, -2.0)
        self.rf.write_vgpr_half_f16(2, HalfSel.LO, 3.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(1, HalfSel.LO, neg=True, abs_=True),
            src1=ASTOperand.vgpr(2, HalfSel.LO),
            src2=ASTOperand.inline_const(5.0),
            clamp=True,
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_half_f16(0, HalfSel.LO), 0.0)

    def test_scalar_interpreter_sibling_half_preservation(self):
        # Packing canary in v0.h
        self.rf.write_vgpr_half_u16(0, HalfSel.HI, 0xABCD)
        self.rf.write_vgpr_half_f16(1, HalfSel.LO, 2.0)
        self.rf.write_vgpr_half_f16(2, HalfSel.LO, 2.0)
        self.rf.write_vgpr_half_f16(3, HalfSel.LO, 2.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(1, HalfSel.LO),
            src1=ASTOperand.vgpr(2, HalfSel.LO),
            src2=ASTOperand.vgpr(3, HalfSel.LO),
        )
        self.interp.execute(inst)
        # v0.l updated to 6.0
        self.assertEqual(self.rf.read_vgpr_half_f16(0, HalfSel.LO), 6.0)
        # v0.h strictly preserved
        self.assertEqual(self.rf.read_vgpr_half_u16(0, HalfSel.HI), 0xABCD)

    def test_scalar_interpreter_deterministic_repeatability(self):
        self.rf.write_vgpr_half_f16(1, HalfSel.LO, 1.25)
        self.rf.write_vgpr_half_f16(2, HalfSel.HI, 3.5)
        self.rf.write_vgpr_half_f16(3, HalfSel.LO, -2.125)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(1, HalfSel.LO),
            src1=ASTOperand.vgpr(2, HalfSel.HI),
            src2=ASTOperand.vgpr(3, HalfSel.LO),
        )
        self.interp.execute(inst)
        res1 = self.rf.read_vgpr_half_u16(0, HalfSel.LO)

        self.interp.execute(inst)
        res2 = self.rf.read_vgpr_half_u16(0, HalfSel.LO)
        self.assertEqual(res1, res2)


class SIMDWaveState:
    """Architectural model of a SIMD wavefront (Wave32 or Wave64) with EXEC masking."""

    def __init__(self, wave_size: int = 64) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self._exec_mask = (1 << wave_size) - 1
        self.vgprs = [[0] * 256 for _ in range(wave_size)]
        self.sgprs = [0] * 106

    @property
    def exec_mask(self) -> int:
        return self._exec_mask

    @exec_mask.setter
    def exec_mask(self, val: int) -> None:
        mask_limit = (1 << self.wave_size) - 1
        self._exec_mask = val & mask_limit

    @property
    def exec_lo(self) -> int:
        return self._exec_mask & MASK32

    @exec_lo.setter
    def exec_lo(self, val: int) -> None:
        self._exec_mask = (self._exec_mask & ~MASK32) | (val & MASK32)

    @property
    def exec_hi(self) -> int:
        return (self._exec_mask >> 32) & MASK32

    @exec_hi.setter
    def exec_hi(self, val: int) -> None:
        if self.wave_size == 32:
            raise ValueError("Cannot set exec_hi on Wave32")
        self._exec_mask = (self._exec_mask & MASK32) | ((val & MASK32) << 32)

    def set_lane_vgpr_half_u16(self, lane: int, idx: int, half: HalfSel, val: int) -> None:
        current = self.vgprs[lane][idx]
        u16 = val & MASK16
        if half == HalfSel.HI:
            self.vgprs[lane][idx] = (current & 0x0000FFFF) | (u16 << 16)
        else:
            self.vgprs[lane][idx] = (current & 0xFFFF0000) | u16

    def get_lane_vgpr_half_u16(self, lane: int, idx: int, half: HalfSel) -> int:
        dword = self.vgprs[lane][idx]
        return (dword >> 16) & MASK16 if half == HalfSel.HI else (dword & MASK16)

    def set_lane_vgpr_half_f16(self, lane: int, idx: int, half: HalfSel, val: float) -> None:
        self.set_lane_vgpr_half_u16(lane, idx, half, f16_to_u16(val))

    def get_lane_vgpr_half_f16(self, lane: int, idx: int, half: HalfSel) -> float:
        return u16_to_f16(self.get_lane_vgpr_half_u16(lane, idx, half))

    def set_sgpr_half_u16(self, idx: int, half: HalfSel, val: int) -> None:
        current = self.sgprs[idx]
        u16 = val & MASK16
        if half == HalfSel.HI:
            self.sgprs[idx] = (current & 0x0000FFFF) | (u16 << 16)
        else:
            self.sgprs[idx] = (current & 0xFFFF0000) | u16

    def get_sgpr_half_u16(self, idx: int, half: HalfSel) -> int:
        dword = self.sgprs[idx]
        return (dword >> 16) & MASK16 if half == HalfSel.HI else (dword & MASK16)

    def set_sgpr_half_f16(self, idx: int, half: HalfSel, val: float) -> None:
        self.set_sgpr_half_u16(idx, half, f16_to_u16(val))

    def get_sgpr_half_f16(self, idx: int, half: HalfSel) -> float:
        return u16_to_f16(self.get_sgpr_half_u16(idx, half))

    def _resolve_operand_u16(self, src: ASTOperand, lane: int) -> int:
        if src.kind == OperandKind.VGPR:
            val_u16 = self.get_lane_vgpr_half_u16(lane, src.index, src.half)
        elif src.kind == OperandKind.SGPR:
            val_u16 = self.get_sgpr_half_u16(src.index, src.half)
        elif src.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            val_u16 = f16_to_u16(float(src.value)) if src.value is not None else F16_POS_ZERO
        else:
            raise ValueError(f"Unsupported operand kind: {src.kind}")

        return apply_src_mods_u16(val_u16, neg=src.neg, abs_=src.abs_)

    def execute(self, inst: ASTInstruction) -> None:
        inst.validate()
        for lane in range(self.wave_size):
            if (self._exec_mask >> lane) & 1:
                u0 = self._resolve_operand_u16(inst.src0, lane)
                u1 = self._resolve_operand_u16(inst.src1, lane)
                u2 = self._resolve_operand_u16(inst.src2, lane)
                res_u16 = ref_fma_f16_u16(u0, u1, u2, clamp=inst.clamp, omod=inst.omod.value)
                self.set_lane_vgpr_half_u16(lane, inst.dst.index, inst.dst.half, res_u16)


class TestFmaF16SIMDMultiLane(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    CANARY_OLD = 0xBEEF

    def test_wave32_full_exec(self):
        wave = SIMDWaveState(wave_size=32)
        wave.exec_mask = 0xFFFFFFFF
        for lane in range(32):
            wave.set_lane_vgpr_half_f16(lane, 1, HalfSel.LO, float(lane))
            wave.set_lane_vgpr_half_f16(lane, 2, HalfSel.LO, 2.0)
            wave.set_lane_vgpr_half_f16(lane, 3, HalfSel.LO, 1.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(1, HalfSel.LO),
            src1=ASTOperand.vgpr(2, HalfSel.LO),
            src2=ASTOperand.vgpr(3, HalfSel.LO),
        )
        wave.execute(inst)

        for lane in range(32):
            expected = float(lane * 2 + 1)
            self.assertEqual(wave.get_lane_vgpr_half_f16(lane, 0, HalfSel.LO), expected)

    def test_wave64_full_exec(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_lo = 0xFFFFFFFF
        wave.exec_hi = 0xFFFFFFFF
        for lane in range(64):
            wave.set_lane_vgpr_half_f16(lane, 1, HalfSel.LO, float(lane) * 0.5)
            wave.set_lane_vgpr_half_f16(lane, 2, HalfSel.LO, 4.0)
            wave.set_lane_vgpr_half_f16(lane, 3, HalfSel.LO, -10.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(1, HalfSel.LO),
            src1=ASTOperand.vgpr(2, HalfSel.LO),
            src2=ASTOperand.vgpr(3, HalfSel.LO),
        )
        wave.execute(inst)

        for lane in range(64):
            expected = float(lane * 0.5) * 4.0 - 10.0
            self.assertEqual(wave.get_lane_vgpr_half_f16(lane, 0, HalfSel.LO), expected)

    def test_wave64_divergence_exec_masking(self):
        wave = SIMDWaveState(wave_size=64)
        # Even lanes in low 32, odd lanes in high 32
        wave.exec_lo = 0x55555555
        wave.exec_hi = 0xAAAAAAAA

        for lane in range(64):
            wave.set_lane_vgpr_half_u16(lane, 0, HalfSel.LO, self.CANARY_OLD)
            wave.set_lane_vgpr_half_f16(lane, 1, HalfSel.LO, 3.0)
            wave.set_lane_vgpr_half_f16(lane, 2, HalfSel.LO, 5.0)
            wave.set_lane_vgpr_half_f16(lane, 3, HalfSel.LO, 2.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(1, HalfSel.LO),
            src1=ASTOperand.vgpr(2, HalfSel.LO),
            src2=ASTOperand.vgpr(3, HalfSel.LO),
        )
        wave.execute(inst)

        for lane in range(64):
            is_active = bool((wave.exec_mask >> lane) & 1)
            if is_active:
                self.assertEqual(wave.get_lane_vgpr_half_f16(lane, 0, HalfSel.LO), 17.0)
            else:
                self.assertEqual(wave.get_lane_vgpr_half_u16(lane, 0, HalfSel.LO), self.CANARY_OLD)

    def test_wave64_halves_masking(self):
        wave = SIMDWaveState(wave_size=64)
        # Only low 32 lanes active
        wave.exec_lo = 0xFFFFFFFF
        wave.exec_hi = 0x00000000

        for lane in range(64):
            wave.set_lane_vgpr_half_u16(lane, 0, HalfSel.LO, self.CANARY_OLD)
            wave.set_lane_vgpr_half_f16(lane, 1, HalfSel.LO, 2.0)
            wave.set_lane_vgpr_half_f16(lane, 2, HalfSel.LO, 2.0)
            wave.set_lane_vgpr_half_f16(lane, 3, HalfSel.LO, 2.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(1, HalfSel.LO),
            src1=ASTOperand.vgpr(2, HalfSel.LO),
            src2=ASTOperand.vgpr(3, HalfSel.LO),
        )
        wave.execute(inst)

        for lane in range(32):
            self.assertEqual(wave.get_lane_vgpr_half_f16(lane, 0, HalfSel.LO), 6.0)
        for lane in range(32, 64):
            self.assertEqual(wave.get_lane_vgpr_half_u16(lane, 0, HalfSel.LO), self.CANARY_OLD)

    def test_single_active_lane(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_mask = 1 << 17  # only lane 17

        for lane in range(64):
            wave.set_lane_vgpr_half_u16(lane, 0, HalfSel.LO, self.CANARY_OLD)
            wave.set_lane_vgpr_half_f16(lane, 1, HalfSel.LO, 7.0)
            wave.set_lane_vgpr_half_f16(lane, 2, HalfSel.LO, 3.0)
            wave.set_lane_vgpr_half_f16(lane, 3, HalfSel.LO, 1.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(1, HalfSel.LO),
            src1=ASTOperand.vgpr(2, HalfSel.LO),
            src2=ASTOperand.vgpr(3, HalfSel.LO),
        )
        wave.execute(inst)

        for lane in range(64):
            if lane == 17:
                self.assertEqual(wave.get_lane_vgpr_half_f16(lane, 0, HalfSel.LO), 22.0)
            else:
                self.assertEqual(wave.get_lane_vgpr_half_u16(lane, 0, HalfSel.LO), self.CANARY_OLD)

    def test_zero_exec_preserves_all_lanes(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_mask = 0

        for lane in range(64):
            wave.set_lane_vgpr_half_u16(lane, 0, HalfSel.LO, self.CANARY_OLD)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.inline_const(1.0),
            src1=ASTOperand.inline_const(1.0),
            src2=ASTOperand.inline_const(1.0),
        )
        wave.execute(inst)

        for lane in range(64):
            self.assertEqual(wave.get_lane_vgpr_half_u16(lane, 0, HalfSel.LO), self.CANARY_OLD)


class TestFmaF16RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = FmaF16ScalarInterpreter(self.rf)

    def test_dst_alias_src0(self) -> None:
        self.rf.write_vgpr_half_f16(0, HalfSel.LO, 3.0)
        self.rf.write_vgpr_half_f16(1, HalfSel.LO, 4.0)
        self.rf.write_vgpr_half_f16(2, HalfSel.LO, 5.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(0, HalfSel.LO),
            src1=ASTOperand.vgpr(1, HalfSel.LO),
            src2=ASTOperand.vgpr(2, HalfSel.LO),
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_half_f16(0, HalfSel.LO), 17.0)
        # Verify unaliased sources remain untouched
        self.assertEqual(self.rf.read_vgpr_half_f16(1, HalfSel.LO), 4.0)
        self.assertEqual(self.rf.read_vgpr_half_f16(2, HalfSel.LO), 5.0)

    def test_dst_alias_src1(self) -> None:
        self.rf.write_vgpr_half_f16(0, HalfSel.LO, 4.0)
        self.rf.write_vgpr_half_f16(1, HalfSel.LO, 3.0)
        self.rf.write_vgpr_half_f16(2, HalfSel.LO, 5.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(1, HalfSel.LO),
            src1=ASTOperand.vgpr(0, HalfSel.LO),
            src2=ASTOperand.vgpr(2, HalfSel.LO),
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_half_f16(0, HalfSel.LO), 17.0)
        self.assertEqual(self.rf.read_vgpr_half_f16(1, HalfSel.LO), 3.0)
        self.assertEqual(self.rf.read_vgpr_half_f16(2, HalfSel.LO), 5.0)

    def test_dst_alias_src2_accumulator(self) -> None:
        # In-place accumulation MAC: v0 = (v1 * v2) + v0
        self.rf.write_vgpr_half_f16(0, HalfSel.LO, 10.0)
        self.rf.write_vgpr_half_f16(1, HalfSel.LO, 2.5)
        self.rf.write_vgpr_half_f16(2, HalfSel.LO, 4.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(1, HalfSel.LO),
            src1=ASTOperand.vgpr(2, HalfSel.LO),
            src2=ASTOperand.vgpr(0, HalfSel.LO),
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_half_f16(0, HalfSel.LO), 20.0)
        self.assertEqual(self.rf.read_vgpr_half_f16(1, HalfSel.LO), 2.5)
        self.assertEqual(self.rf.read_vgpr_half_f16(2, HalfSel.LO), 4.0)

    def test_dst_alias_all_sources(self) -> None:
        # v0 = (v0 * v0) + v0
        self.rf.write_vgpr_half_f16(0, HalfSel.LO, 2.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(0, HalfSel.LO),
            src1=ASTOperand.vgpr(0, HalfSel.LO),
            src2=ASTOperand.vgpr(0, HalfSel.LO),
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_half_f16(0, HalfSel.LO), 6.0)

    def test_inplace_iterative_accumulation(self) -> None:
        # 50 iterations: v0 = (v0 * 1.0) + 0.25
        self.rf.write_vgpr_half_f16(0, HalfSel.LO, 0.0)
        self.rf.write_vgpr_half_f16(1, HalfSel.LO, 1.0)
        self.rf.write_vgpr_half_f16(2, HalfSel.LO, 0.25)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(0, HalfSel.LO),
            src1=ASTOperand.vgpr(1, HalfSel.LO),
            src2=ASTOperand.vgpr(2, HalfSel.LO),
        )
        for _ in range(50):
            self.interp.execute(inst)

        self.assertEqual(self.rf.read_vgpr_half_f16(0, HalfSel.LO), 12.5)

    def test_sibling_halfword_preservation_under_aliasing(self) -> None:
        # Reading high half as source, writing low half as destination
        # v0.l = (v0.h * 2.0) + 1.0
        self.rf.write_vgpr_half_f16(0, HalfSel.HI, 5.0)
        self.rf.write_vgpr_half_f16(0, HalfSel.LO, 0.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(0, HalfSel.HI),
            src1=ASTOperand.inline_const(2.0),
            src2=ASTOperand.inline_const(1.0),
        )
        self.interp.execute(inst)
        # v0.l updated to 11.0
        self.assertEqual(self.rf.read_vgpr_half_f16(0, HalfSel.LO), 11.0)
        # v0.h must remain untouched 5.0
        self.assertEqual(self.rf.read_vgpr_half_f16(0, HalfSel.HI), 5.0)

    def test_high_half_destination_sibling_preservation(self) -> None:
        # Reading low half as source, writing high half as destination
        # v0.h = (v0.l * 3.0) + 2.0
        self.rf.write_vgpr_half_f16(0, HalfSel.LO, 4.0)
        self.rf.write_vgpr_half_f16(0, HalfSel.HI, 0.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.HI),
            src0=ASTOperand.vgpr(0, HalfSel.LO),
            src1=ASTOperand.inline_const(3.0),
            src2=ASTOperand.inline_const(2.0),
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_half_f16(0, HalfSel.HI), 14.0)
        self.assertEqual(self.rf.read_vgpr_half_f16(0, HalfSel.LO), 4.0)

    def test_simd_multi_lane_aliasing(self) -> None:
        wave = SIMDWaveState(wave_size=32)
        wave.exec_mask = 0xFFFFFFFF
        for lane in range(32):
            wave.set_lane_vgpr_half_f16(lane, 0, HalfSel.LO, float(lane))

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.vgpr(0, HalfSel.LO),
            src1=ASTOperand.inline_const(2.0),
            src2=ASTOperand.inline_const(1.0),
        )
        wave.execute(inst)

        for lane in range(32):
            expected = float(lane) * 2.0 + 1.0
            self.assertEqual(wave.get_lane_vgpr_half_f16(lane, 0, HalfSel.LO), expected)


@dataclass(frozen=True)
class FPStatusFlags:
    """Architectural floating-point condition codes and IEEE 754 status flags."""
    invalid: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False
    zero: bool = False
    negative: bool = False
    denormal: bool = False
    infinite: bool = False
    nan: bool = False


def evaluate_v_fma_f16_flags(
    u0: int,
    u1: int,
    u2: int,
    res_u16: int,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    abs0: bool = False,
    abs1: bool = False,
    abs2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> FPStatusFlags:
    """Evaluate IEEE 754 architectural flags and condition codes for v_fma_f16."""
    s0 = apply_src_mods_u16(u0, neg=neg0, abs_=abs0)
    s1 = apply_src_mods_u16(u1, neg=neg1, abs_=abs1)
    s2 = apply_src_mods_u16(u2, neg=neg2, abs_=abs2)

    s0_nan = is_nan_f16(s0)
    s1_nan = is_nan_f16(s1)
    s2_nan = is_nan_f16(s2)
    s0_snan = is_snan_f16(s0)
    s1_snan = is_snan_f16(s1)
    s2_snan = is_snan_f16(s2)
    s0_inf = is_inf_f16(s0)
    s1_inf = is_inf_f16(s1)
    s2_inf = is_inf_f16(s2)

    invalid = False
    if s0_snan or s1_snan or s2_snan:
        invalid = True
    elif (s0_inf and is_zero_f16(s1)) or (is_zero_f16(s0) and s1_inf):
        invalid = True
    elif (s0_inf or s1_inf) and s2_inf:
        prod_sign = ((s0 >> 15) ^ (s1 >> 15)) & 1
        s2_sign = (s2 >> 15) & 1
        if prod_sign != s2_sign:
            invalid = True

    nan_flag = is_nan_f16(res_u16)
    inf_flag = is_inf_f16(res_u16)
    zero_flag = is_zero_f16(res_u16)
    neg_flag = bool(res_u16 & SIGN_MASK_F16)
    denorm_flag = (
        is_subnormal_f16(s0)
        or is_subnormal_f16(s1)
        or is_subnormal_f16(s2)
        or is_subnormal_f16(res_u16)
    )

    overflow = False
    if not (s0_nan or s1_nan or s2_nan or s0_inf or s1_inf or s2_inf) and inf_flag:
        overflow = True

    underflow = False
    if is_subnormal_f16(res_u16):
        underflow = True

    inexact = False
    if overflow:
        inexact = True
    elif not (s0_nan or s1_nan or s2_nan or s0_inf or s1_inf or s2_inf or nan_flag or inf_flag):
        f0 = u16_to_f16(s0)
        f1 = u16_to_f16(s1)
        f2 = u16_to_f16(s2)
        exact_math = Fraction(f0) * Fraction(f1) + Fraction(f2)
        if omod == 1:
            exact_math *= 2
        elif omod == 2:
            exact_math *= 4
        elif omod == 3:
            exact_math /= 2
        if clamp:
            exact_math = min(Fraction(1), max(Fraction(0), exact_math))
        res_f = u16_to_f16(res_u16)
        if exact_math != Fraction(res_f):
            inexact = True

    return FPStatusFlags(
        invalid=invalid,
        overflow=overflow,
        underflow=underflow,
        inexact=inexact,
        zero=zero_flag,
        negative=neg_flag,
        denormal=denorm_flag,
        infinite=inf_flag,
        nan=nan_flag,
    )


def compute_f16_mantissa_fma_carry(u0: int, u1: int, u2: int) -> tuple[int, int]:
    """Demonstrate IEEE 754 mantissa alignment and carry-out generation during FMA."""
    _, exp0, frac0 = split_f16(u0)
    _, exp1, frac1 = split_f16(u1)
    _, exp2, frac2 = split_f16(u2)

    m0 = (1 << 10) | frac0 if exp0 != 0 else frac0
    m1 = (1 << 10) | frac1 if exp1 != 0 else frac1
    m2 = (1 << 10) | frac2 if exp2 != 0 else frac2

    m_prod = m0 * m1
    exp_prod = (exp0 if exp0 != 0 else 1) + (exp1 if exp1 != 0 else 1) - 15
    e2 = exp2 if exp2 != 0 else 1

    m2_shifted = m2 << 10
    if exp_prod > e2:
        shift = min(exp_prod - e2, 30)
        m2_aligned = m2_shifted >> shift
        prod_aligned = m_prod
    else:
        shift = min(e2 - exp_prod, 30)
        prod_aligned = m_prod >> shift
        m2_aligned = m2_shifted

    mant_sum = prod_aligned + m2_aligned
    carry_out = 1 if mant_sum >= (1 << 21) else 0
    return mant_sum, carry_out


class TestFmaF16ConditionCodesAndStatusFlags(unittest.TestCase):
    """Test suite for status flags, condition codes, and carry generation."""

    def test_exact_finite_flags(self):
        u0 = f16_to_u16(1.0)
        u1 = f16_to_u16(2.0)
        u2 = f16_to_u16(0.5)
        res = ref_fma_f16_u16(u0, u1, u2)
        flags = evaluate_v_fma_f16_flags(u0, u1, u2, res)
        self.assertFalse(flags.invalid)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.underflow)
        self.assertFalse(flags.inexact)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.denormal)
        self.assertFalse(flags.infinite)
        self.assertFalse(flags.nan)

    def test_inexact_flag(self):
        # 1.0009765625 * 1.0009765625 + 0.0 has inexact remainder 2^-20
        u0 = 0x3C01
        u1 = 0x3C01
        u2 = 0x0000
        res = ref_fma_f16_u16(u0, u1, u2)
        flags = evaluate_v_fma_f16_flags(u0, u1, u2, res)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.underflow)

    def test_overflow_flag(self):
        # 60000.0 * 2.0 + 0.0 overflows max normal f16 (65504.0) to +inf
        u0 = f16_to_u16(60000.0)
        u1 = f16_to_u16(2.0)
        u2 = 0x0000
        res = ref_fma_f16_u16(u0, u1, u2)
        flags = evaluate_v_fma_f16_flags(u0, u1, u2, res)
        self.assertTrue(flags.overflow)
        self.assertTrue(flags.inexact)
        self.assertTrue(flags.infinite)
        self.assertFalse(flags.invalid)

    def test_underflow_flag(self):
        # Smallest subnormal result
        u0 = 0x3C01
        u1 = 0x3C01
        u2 = 0xBC02
        res = ref_fma_f16_u16(u0, u1, u2)
        self.assertEqual(res, 0x0010)
        flags = evaluate_v_fma_f16_flags(u0, u1, u2, res)
        self.assertTrue(flags.underflow)
        self.assertTrue(flags.denormal)

    def test_invalid_operation_flags(self):
        # Signaling NaN
        u_snan = F16_SNAN
        res1 = ref_fma_f16_u16(u_snan, F16_ONE, F16_POS_ZERO)
        flags1 = evaluate_v_fma_f16_flags(u_snan, F16_ONE, F16_POS_ZERO, res1)
        self.assertTrue(flags1.invalid)
        self.assertTrue(flags1.nan)

        # Inf * 0
        res2 = ref_fma_f16_u16(F16_POS_INF, F16_POS_ZERO, F16_ONE)
        flags2 = evaluate_v_fma_f16_flags(F16_POS_INF, F16_POS_ZERO, F16_ONE, res2)
        self.assertTrue(flags2.invalid)
        self.assertTrue(flags2.nan)

        # (+inf) + (-inf)
        res3 = ref_fma_f16_u16(F16_POS_INF, F16_ONE, F16_NEG_INF)
        flags3 = evaluate_v_fma_f16_flags(F16_POS_INF, F16_ONE, F16_NEG_INF, res3)
        self.assertTrue(flags3.invalid)
        self.assertTrue(flags3.nan)

    def test_zero_and_negative_condition_codes(self):
        # Negative zero
        res_neg0 = ref_fma_f16_u16(F16_NEG_ZERO, F16_ONE, F16_NEG_ZERO)
        flags_neg0 = evaluate_v_fma_f16_flags(F16_NEG_ZERO, F16_ONE, F16_NEG_ZERO, res_neg0)
        self.assertTrue(flags_neg0.zero)
        self.assertTrue(flags_neg0.negative)

        # Positive zero
        res_pos0 = ref_fma_f16_u16(F16_POS_ZERO, F16_ONE, F16_POS_ZERO)
        flags_pos0 = evaluate_v_fma_f16_flags(F16_POS_ZERO, F16_ONE, F16_POS_ZERO, res_pos0)
        self.assertTrue(flags_pos0.zero)
        self.assertFalse(flags_pos0.negative)

    def test_mantissa_carry_generation(self):
        # 1.0 * 1.0 + 1.0 -> mantissas: (1 << 20) + (1 << 20) = (1 << 21) -> carry out = 1
        _, carry1 = compute_f16_mantissa_fma_carry(F16_ONE, F16_ONE, F16_ONE)
        self.assertEqual(carry1, 1)

        # 1.0 * 1.0 + 0.25 -> mantissas sum < (1 << 21) -> carry out = 0
        _, carry0 = compute_f16_mantissa_fma_carry(F16_ONE, F16_ONE, f16_to_u16(0.25))
        self.assertEqual(carry0, 0)


class TestFmaF16IEEE754BoundaryExtremes(unittest.TestCase):
    """Test suite for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_addition_without_carry(self):
        sub1 = 0x0001
        sub2 = 0x0002
        res = ref_fma_f16_u16(sub1, F16_ONE, sub2)
        self.assertEqual(res, 0x0003)
        self.assertTrue(is_subnormal_f16(res))

    def test_subnormal_to_normal_transition(self):
        max_sub = 0x03FF  # largest subnormal (1023/1024) * 2^-14
        min_sub = 0x0001  # smallest subnormal 2^-24
        # (max_sub * 1.0) + min_sub = 0x0400 = smallest normal (2^-14)
        res = ref_fma_f16_u16(max_sub, F16_ONE, min_sub)
        self.assertEqual(res, 0x0400)
        self.assertFalse(is_subnormal_f16(res))
        _, exp, frac = split_f16(res)
        self.assertEqual(exp, 1)
        self.assertEqual(frac, 0)

    def test_subnormal_gradual_underflow(self):
        min_normal = F16_MIN_NORMAL  # 0x0400 = 2^-14
        res = ref_fma_f16_u16(min_normal, F16_HALF, F16_POS_ZERO)
        self.assertEqual(res, 0x0200)  # 2^-15 (subnormal)
        self.assertTrue(is_subnormal_f16(res))

    def test_subnormal_cancellation(self):
        sub = 0x0123
        neg_sub = sub | SIGN_MASK_F16
        res = ref_fma_f16_u16(sub, F16_ONE, neg_sub)
        self.assertEqual(res, F16_POS_ZERO)

    def test_nan_signaling_to_quiet_conversion(self):
        snan = 0x7C05  # fraction != 0, quiet bit 9 is 0
        self.assertTrue(is_snan_f16(snan))
        res = ref_fma_f16_u16(snan, F16_ONE, F16_POS_ZERO)
        self.assertTrue(is_qnan_f16(res))
        self.assertEqual(res, snan | QNAN_BIT_F16)

    def test_nan_payload_preservation(self):
        payload = 0x01AB
        qnan = EXP_MASK_F16 | payload | QNAN_BIT_F16
        res = ref_fma_f16_u16(qnan, F16_TWO, F16_ONE)
        self.assertEqual(res, qnan)

    def test_nan_with_modifiers(self):
        qnan = 0x7E01
        # abs should clear sign bit
        res_abs = ref_fma_f16_u16(qnan | SIGN_MASK_F16, F16_ONE, F16_POS_ZERO, abs0=True)
        self.assertEqual(res_abs & SIGN_MASK_F16, 0)
        self.assertTrue(is_qnan_f16(res_abs))

        # neg should toggle sign bit
        res_neg = ref_fma_f16_u16(qnan, F16_ONE, F16_POS_ZERO, neg0=True)
        self.assertEqual(res_neg & SIGN_MASK_F16, SIGN_MASK_F16)

    def test_nan_with_clamp(self):
        qnan = F16_CANONICAL_QNAN
        res = ref_fma_f16_u16(qnan, F16_ONE, F16_ONE, clamp=True)
        self.assertEqual(res, F16_POS_ZERO)

    def test_infinity_arithmetic_boundaries(self):
        self.assertEqual(ref_fma_f16_u16(F16_POS_INF, f16_to_u16(100.0), F16_POS_ZERO), F16_POS_INF)
        self.assertEqual(ref_fma_f16_u16(F16_NEG_INF, f16_to_u16(100.0), F16_POS_ZERO), F16_NEG_INF)
        self.assertEqual(ref_fma_f16_u16(F16_POS_INF, F16_ONE, F16_NEG_INF), F16_CANONICAL_QNAN)
        self.assertEqual(ref_fma_f16_u16(F16_NEG_INF, F16_ONE, F16_POS_INF), F16_CANONICAL_QNAN)

        # Infinity with clamp
        self.assertEqual(ref_fma_f16_u16(F16_POS_INF, F16_ONE, F16_POS_ZERO, clamp=True), F16_ONE)
        self.assertEqual(ref_fma_f16_u16(F16_NEG_INF, F16_ONE, F16_POS_ZERO, clamp=True), F16_POS_ZERO)

    def test_round_to_nearest_even_ties(self):
        # In IEEE 754 binary16 (10 bits significand):
        # 1.0 has bits 0x3C00 (even LSB = 0)
        # 1.0 + 2^-10 has bits 0x3C01 (odd LSB = 1)
        # 2^-11 is exactly halfway between representable numbers.
        half_ulp = 2.0 ** -11
        one = 1.0
        one_plus_ulp = u16_to_f16(0x3C01)

        # Case 1: 1.0 (even LSB) + 2^-11: tie rounds to even (1.0)
        res1 = ref_fma_f16(one, 1.0, half_ulp)
        self.assertEqual(f16_to_u16(res1), F16_ONE)

        # Case 2: (1.0 + 2^-10) (odd LSB) + 2^-11: tie rounds to even (1.0 + 2 * 2^-10) = 0x3C02
        res2 = ref_fma_f16(one_plus_ulp, 1.0, half_ulp)
        self.assertEqual(f16_to_u16(res2), 0x3C02)


class TestFmaF16NegativeControlsAndMutations(unittest.TestCase):
    """Test suite for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_opcode(self):
        class FakeOpcode(Enum):
            V_ADD_F16 = "v_add_f16"

        bad_inst = ASTInstruction(
            opcode=FakeOpcode.V_ADD_F16,  # type: ignore
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            bad_inst.validate()

    def test_syntax_rejection_bad_dst_kind(self):
        bad_inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(TypeError):
            bad_inst.validate()

    def test_syntax_rejection_bad_dst_type(self):
        bad_inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand(OperandKind.VGPR, index=0, type_kind=TypeKind.UINT16),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(TypeError):
            bad_inst.validate()

    def test_syntax_rejection_dst_modifiers(self):
        bad_inst_neg = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, neg=True),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            bad_inst_neg.validate()

        bad_inst_abs = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, abs_=True),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            bad_inst_abs.validate()

    def test_syntax_rejection_oob_indices(self):
        bad_vgpr = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(256),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            bad_vgpr.validate()

        bad_sgpr = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.sgpr(106),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            bad_sgpr.validate()

    def test_syntax_rejection_bad_src_type(self):
        bad_src = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand(OperandKind.VGPR, index=1, type_kind=TypeKind.UINT32),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(TypeError):
            bad_src.validate()

    def test_mutation_unfused_intermediate_rounding_detected(self):
        # Mutated logic: unfused multiplication followed by addition
        def mutated_unfused_fma(u0: int, u1: int, u2: int) -> int:
            f0 = u16_to_f16(u0)
            f1 = u16_to_f16(u1)
            f2 = u16_to_f16(u2)
            # Intermediate rounding to binary16
            prod_rounded = u16_to_f16(f16_to_u16(f0 * f1))
            return f16_to_u16(prod_rounded + f2)

        u0 = 0x3C01
        u1 = 0x3C01
        u2 = 0xBC02

        oracle_result = ref_fma_f16_u16(u0, u1, u2)
        mutated_result = mutated_unfused_fma(u0, u1, u2)

        # Oracle preserves the 2^-20 subnormal (0x0010), while mutated unfused flushes to 0.0
        self.assertEqual(oracle_result, 0x0010)
        self.assertEqual(mutated_result, 0x0000)
        self.assertNotEqual(oracle_result, mutated_result)

    def test_mutation_addition_subtraction_flip_detected(self):
        def mutated_fma_sub(u0: int, u1: int, u2: int) -> int:
            f0 = u16_to_f16(u0)
            f1 = u16_to_f16(u1)
            f2 = u16_to_f16(u2)
            frac_diff = Fraction(f0) * Fraction(f1) - Fraction(f2)
            return f16_to_u16(float(frac_diff))

        u0 = f16_to_u16(2.0)
        u1 = f16_to_u16(3.0)
        u2 = f16_to_u16(4.0)

        oracle_result = ref_fma_f16_u16(u0, u1, u2)      # 2 * 3 + 4 = 10.0
        mutated_result = mutated_fma_sub(u0, u1, u2)     # 2 * 3 - 4 = 2.0
        self.assertEqual(u16_to_f16(oracle_result), 10.0)
        self.assertEqual(u16_to_f16(mutated_result), 2.0)
        self.assertNotEqual(oracle_result, mutated_result)

    def test_mutation_dropped_clamp_detected(self):
        def mutated_clamp(bits: int) -> int:
            return bits  # mutated: no clamp

        u0 = f16_to_u16(2.0)
        u1 = f16_to_u16(3.0)
        u2 = f16_to_u16(0.0)

        oracle_result = ref_fma_f16_u16(u0, u1, u2, clamp=True)
        mutated_result = mutated_clamp(ref_fma_f16_u16(u0, u1, u2, clamp=False))

        self.assertEqual(u16_to_f16(oracle_result), 1.0)
        self.assertEqual(u16_to_f16(mutated_result), 6.0)
        self.assertNotEqual(oracle_result, mutated_result)

    def test_mutation_dropped_omod_detected(self):
        def mutated_omod(bits: int, omod: int) -> int:
            return bits  # mutated: dropped omod

        u0 = f16_to_u16(1.5)
        u1 = f16_to_u16(2.0)
        u2 = f16_to_u16(0.5)

        oracle_result = ref_fma_f16_u16(u0, u1, u2, omod=1)
        mutated_result = mutated_omod(ref_fma_f16_u16(u0, u1, u2, omod=0), 1)

        self.assertEqual(u16_to_f16(oracle_result), 7.0)
        self.assertEqual(u16_to_f16(mutated_result), 3.5)
        self.assertNotEqual(oracle_result, mutated_result)

    def test_mutation_sign_modifier_dropped_detected(self):
        def mutated_apply_mods(bits: int, *, neg: bool = False, abs_: bool = False) -> int:
            return bits  # mutated: ignores neg/abs

        raw = f16_to_u16(3.5)
        oracle_mod = apply_src_mods_u16(raw, neg=True)
        mutated_mod = mutated_apply_mods(raw, neg=True)

        self.assertEqual(u16_to_f16(oracle_mod), -3.5)
        self.assertEqual(u16_to_f16(mutated_mod), 3.5)
        self.assertNotEqual(oracle_mod, mutated_mod)

    def test_mutation_subnormal_flush_detected(self):
        def mutated_ftz_fma(u0: int, u1: int, u2: int) -> int:
            res = ref_fma_f16_u16(u0, u1, u2)
            if is_subnormal_f16(res):
                return F16_POS_ZERO  # mutated: flush to zero
            return res

        u0 = 0x3C01
        u1 = 0x3C01
        u2 = 0xBC02

        oracle_result = ref_fma_f16_u16(u0, u1, u2)
        mutated_result = mutated_ftz_fma(u0, u1, u2)

        self.assertEqual(oracle_result, 0x0010)
        self.assertEqual(mutated_result, 0x0000)
        self.assertNotEqual(oracle_result, mutated_result)

    def test_negative_control_canary_corruption(self):
        wave = SIMDWaveState(wave_size=32)
        wave.exec_mask = 0x00000001  # only lane 0 active
        canary = 0xCAFE
        for lane in range(32):
            wave.set_lane_vgpr_half_u16(lane, 0, HalfSel.LO, canary)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F16,
            dst=ASTOperand.vgpr(0, HalfSel.LO),
            src0=ASTOperand.inline_const(2.0),
            src1=ASTOperand.inline_const(3.0),
            src2=ASTOperand.inline_const(4.0),
        )
        wave.execute(inst)

        self.assertEqual(wave.get_lane_vgpr_half_f16(0, 0, HalfSel.LO), 10.0)
        for lane in range(1, 32):
            self.assertEqual(
                wave.get_lane_vgpr_half_u16(lane, 0, HalfSel.LO),
                canary,
                f"Inactive lane {lane} was corrupted",
            )


if __name__ == "__main__":
    unittest.main()
