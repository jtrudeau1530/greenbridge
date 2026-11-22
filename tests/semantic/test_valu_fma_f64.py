from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_fma_f64.

Tests the double-precision fused multiply-add instruction:
    v_fma_f64 D, S0, S1, S2
Lowered semantics:
    D = (S0 * S1) + S2
with single-rounding precision per IEEE 754-2008 (binary64).
Supports input modifiers:
    - abs (|src|: clear sign bit)
    - neg (-src: invert sign bit)
Supports output modifiers:
    - clamp: saturates result to [0.0, 1.0], with NaN clamped to +0.0
    - omod: mul:2, mul:4, div:2
Architectural model:
    64-bit floating point registers mapped onto aligned VGPR/SGPR pairs:
    [lo: 32-bit dword, hi: 32-bit dword].
"""

from enum import Enum, auto
from fractions import Fraction
import math
import struct
import sys
import unittest

# IEEE 754 double precision bitwise constants
F64_POS_ZERO = 0x0000000000000000
F64_NEG_ZERO = 0x8000000000000000
F64_ONE = 0x3FF0000000000000
F64_NEG_ONE = 0xBFF0000000000000
F64_TWO = 0x4000000000000000
F64_HALF = 0x3FE0000000000000
F64_POS_INF = 0x7FF0000000000000
F64_NEG_INF = 0xFFF0000000000000
F64_CANONICAL_QNAN = 0x7FF8000000000000
F64_SNAN = 0x7FF0000000000001
F64_SUB_MIN = 0x0000000000000001
F64_SUB_MAX = 0x000FFFFFFFFFFFFF
F64_MIN_NORMAL = 0x0010000000000000
F64_MAX_NORMAL = 0x7FEFFFFFFFFFFFFF

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def f64_to_u64(val: float) -> int:
    """Pack Python float (IEEE 754 binary64) into unsigned 64-bit integer."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def u64_to_f64(bits: int) -> float:
    """Unpack unsigned 64-bit integer into Python float (IEEE 754 binary64)."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def u64_to_dwords(val: int) -> tuple[int, int]:
    """Split 64-bit word into (low 32-bit dword, high 32-bit dword)."""
    return val & MASK32, (val >> 32) & MASK32


def dwords_to_u64(lo: int, hi: int) -> int:
    """Combine (low 32-bit dword, high 32-bit dword) into 64-bit word."""
    return (lo & MASK32) | ((hi & MASK32) << 32)


def apply_src_mods_u64(val_u64: int, *, neg: bool = False, abs_: bool = False) -> int:
    """Apply bit-exact RDNA2 source modifiers (|src|, -src) to a 64-bit float bit pattern."""
    bits = val_u64 & MASK64
    if abs_:
        bits &= ~F64_NEG_ZERO  # clear sign bit
    if neg:
        bits ^= F64_NEG_ZERO   # invert sign bit
    return bits


def apply_clamp_f64(val: float) -> float:
    """Apply RDNA2 clamp modifier [0.0, 1.0] to float64, mapping NaNs to +0.0."""
    if math.isnan(val):
        return 0.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def apply_omod_f64(val: float, omod: int) -> float:
    """Apply RDNA2 output modifier: 0: none, 1: mul:2, 2: mul:4, 3: div:2."""
    if omod == 0:
        return val
    elif omod == 1:
        return val * 2.0
    elif omod == 2:
        return val * 4.0
    elif omod == 3:
        return val * 0.5
    else:
        raise ValueError(f"Invalid omod value: {omod}")


def raw_fma_f64(v0: float, v1: float, v2: float) -> float:
    """Compute exact IEEE 754 double precision FMA (v0 * v1) + v2."""
    if math.isnan(v0) or math.isnan(v1) or math.isnan(v2):
        return math.nan

    if (math.isinf(v0) and v1 == 0.0) or (v0 == 0.0 and math.isinf(v1)):
        return math.nan

    p_inf = math.isinf(v0) or math.isinf(v1)
    if p_inf:
        p_sign = -1.0 if (math.copysign(1.0, v0) * math.copysign(1.0, v1) < 0) else 1.0
        if math.isinf(v2):
            v2_sign = math.copysign(1.0, v2)
            if p_sign != v2_sign:
                return math.nan
        return math.copysign(float("inf"), p_sign)

    if math.isinf(v2):
        return v2

    if v0 == 0.0 or v1 == 0.0:
        prod_sign = -1.0 if (math.copysign(1.0, v0) * math.copysign(1.0, v1) < 0) else 1.0
        if v2 == 0.0:
            v2_sign = math.copysign(1.0, v2)
            if prod_sign < 0 and v2_sign < 0:
                return -0.0
            return 0.0
        return v2

    frac = Fraction(v0) * Fraction(v1) + Fraction(v2)
    if frac == 0:
        return 0.0
    try:
        return float(frac)
    except OverflowError:
        return math.copysign(float("inf"), 1.0 if frac > 0 else -1.0)


def ref_fma_f64(
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
    """Reference mathematical oracle for v_fma_f64 calculating (s0 * s1) + s2 in binary64."""
    u0 = apply_src_mods_u64(f64_to_u64(s0), neg=neg0, abs_=abs0)
    u1 = apply_src_mods_u64(f64_to_u64(s1), neg=neg1, abs_=abs1)
    u2 = apply_src_mods_u64(f64_to_u64(s2), neg=neg2, abs_=abs2)

    v0 = u64_to_f64(u0)
    v1 = u64_to_f64(u1)
    v2 = u64_to_f64(u2)

    res = raw_fma_f64(v0, v1, v2)

    if omod != 0:
        res = apply_omod_f64(res, omod)
    if clamp:
        res = apply_clamp_f64(res)

    return res


def ref_fma_f64_u64(
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
    """Bit-exact reference mathematical oracle for v_fma_f64 using 64-bit integer words."""
    res_f = ref_fma_f64(
        u64_to_f64(u0),
        u64_to_f64(u1),
        u64_to_f64(u2),
        neg0=neg0,
        neg1=neg1,
        neg2=neg2,
        abs0=abs0,
        abs1=abs1,
        abs2=abs2,
        clamp=clamp,
        omod=omod,
    )
    return f64_to_u64(res_f)


class RegisterPair:
    """Architectural model of a 64-bit register pair (e.g. v[0:1] or s[0:1])."""

    def __init__(self, lo: int = 0, hi: int = 0) -> None:
        self.lo = lo & MASK32
        self.hi = hi & MASK32

    @classmethod
    def from_u64(cls, val: int) -> RegisterPair:
        lo, hi = u64_to_dwords(val)
        return cls(lo, hi)

    @classmethod
    def from_f64(cls, val: float) -> RegisterPair:
        return cls.from_u64(f64_to_u64(val))

    @property
    def as_u64(self) -> int:
        return dwords_to_u64(self.lo, self.hi)

    @property
    def as_f64(self) -> float:
        return u64_to_f64(self.as_u64)

    def set_u64(self, val: int) -> None:
        self.lo, self.hi = u64_to_dwords(val)

    def set_f64(self, val: float) -> None:
        self.set_u64(f64_to_u64(val))

    def __repr__(self) -> str:
        return f"RegisterPair(lo={self.lo:#010x}, hi={self.hi:#010x}, f64={self.as_f64})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegisterPair):
            return NotImplemented
        return self.lo == other.lo and self.hi == other.hi


class TestFmaF64MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural register models."""

    def test_basic_fma_identities(self):
        # 1.0 * 1.0 + 0.0 == 1.0
        self.assertEqual(ref_fma_f64(1.0, 1.0, 0.0), 1.0)
        # 2.0 * 3.0 + 4.0 == 10.0
        self.assertEqual(ref_fma_f64(2.0, 3.0, 4.0), 10.0)
        # -2.5 * 4.0 + 1.25 == -8.75
        self.assertEqual(ref_fma_f64(-2.5, 4.0, 1.25), -8.75)
        # 0.0 * 100.0 + 5.5 == 5.5
        self.assertEqual(ref_fma_f64(0.0, 100.0, 5.5), 5.5)

    def test_fused_precision_single_rounding(self):
        # FMA retains full intermediate product precision without intermediate rounding.
        # Let a = 1.0 + 2**-52, b = 1.0 - 2**-52, c = -1.0
        # (a * b) = 1.0 - 2**-104
        # Unfused: (a * b) rounds to 1.0, so (a * b) - 1.0 == 0.0
        # Fused: (a * b) - 1.0 retains -2**-104, rounding to -2**-104 != 0.0
        a = 1.0 + (2.0 ** -52)
        b = 1.0 - (2.0 ** -52)
        c = -1.0
        unfused = (a * b) + c
        fused = ref_fma_f64(a, b, c)
        self.assertEqual(unfused, 0.0)
        self.assertNotEqual(fused, 0.0)
        self.assertEqual(fused, -(2.0 ** -104))

    def test_source_modifiers(self):
        # abs(|src|): clears sign bit
        self.assertEqual(ref_fma_f64(-3.5, 2.0, 1.0, abs0=True), 8.0)
        self.assertEqual(ref_fma_f64(3.5, -2.0, 1.0, abs1=True), 8.0)
        self.assertEqual(ref_fma_f64(3.5, 2.0, -1.0, abs2=True), 8.0)

        # neg(-src): inverts sign bit
        self.assertEqual(ref_fma_f64(3.5, 2.0, 1.0, neg0=True), -6.0)
        self.assertEqual(ref_fma_f64(3.5, 2.0, 1.0, neg1=True), -6.0)
        self.assertEqual(ref_fma_f64(3.5, 2.0, 1.0, neg2=True), 6.0)

        # combined neg and abs: -|src|
        self.assertEqual(ref_fma_f64(-3.5, 2.0, 1.0, neg0=True, abs0=True), -6.0)

    def test_output_modifiers_clamp_and_omod(self):
        # clamp saturates to [0.0, 1.0]
        self.assertEqual(ref_fma_f64(2.0, 3.0, 0.0, clamp=True), 1.0)
        self.assertEqual(ref_fma_f64(-2.0, 3.0, 0.0, clamp=True), 0.0)
        self.assertEqual(ref_fma_f64(0.5, 1.0, 0.25, clamp=True), 0.75)

        # omod: mul:2, mul:4, div:2
        self.assertEqual(ref_fma_f64(1.5, 2.0, 0.5, omod=1), 7.0)   # (3.5) * 2 = 7.0
        self.assertEqual(ref_fma_f64(1.5, 2.0, 0.5, omod=2), 14.0)  # (3.5) * 4 = 14.0
        self.assertEqual(ref_fma_f64(1.5, 2.0, 0.5, omod=3), 1.75)  # (3.5) * 0.5 = 1.75

        # omod combined with clamp: (omod applies before clamp)
        self.assertEqual(ref_fma_f64(0.2, 2.0, 0.0, omod=1, clamp=True), 0.8)
        self.assertEqual(ref_fma_f64(0.4, 2.0, 0.0, omod=2, clamp=True), 1.0)

    def test_register_pair_architecture(self):
        rp = RegisterPair.from_f64(1.0)
        self.assertEqual(rp.as_f64, 1.0)
        self.assertEqual(rp.as_u64, F64_ONE)
        self.assertEqual(rp.lo, 0x00000000)
        self.assertEqual(rp.hi, 0x3FF00000)

        rp.set_f64(-2.0)
        self.assertEqual(rp.as_f64, -2.0)
        self.assertEqual(rp.hi, 0xC0000000)

        # bit-exact oracle roundtrip
        u_res = ref_fma_f64_u64(F64_TWO, F64_HALF, F64_ONE)
        self.assertEqual(u_res, f64_to_u64(2.0))


class Opcode(Enum):
    V_FMA_F64 = "v_fma_f64"


class OperandKind(Enum):
    VGPR_PAIR = auto()
    SGPR_PAIR = auto()
    INLINE_CONST = auto()
    LITERAL_CONST = auto()


class TypeKind(Enum):
    FLOAT64 = "f64"
    UINT64 = "u64"
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
    """Symbolic AST operand representing a register pair, constant, or literal with modifiers."""

    def __init__(
        self,
        kind: OperandKind,
        index: int = 0,
        value: float | int | None = None,
        *,
        neg: bool = False,
        abs_: bool = False,
        type_kind: TypeKind = TypeKind.FLOAT64,
    ) -> None:
        self.kind = kind
        self.index = index
        self.value = value
        self.neg = neg
        self.abs_ = abs_
        self.type_kind = type_kind

    @classmethod
    def vgpr(cls, index: int, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.VGPR_PAIR, index=index, neg=neg, abs_=abs_)

    @classmethod
    def sgpr(cls, index: int, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.SGPR_PAIR, index=index, neg=neg, abs_=abs_)

    @classmethod
    def inline_const(cls, value: float, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.INLINE_CONST, value=value, neg=neg, abs_=abs_)

    @classmethod
    def literal_const(cls, value: float, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.LITERAL_CONST, value=value, neg=neg, abs_=abs_)

    def disasm(self) -> str:
        if self.kind == OperandKind.VGPR_PAIR:
            body = f"v[{self.index}:{self.index + 1}]"
        elif self.kind == OperandKind.SGPR_PAIR:
            body = f"s[{self.index}:{self.index + 1}]"
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
    """Symbolic AST instruction representation for v_fma_f64."""

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
        if self.opcode != Opcode.V_FMA_F64:
            raise ValueError(f"Invalid opcode for v_fma_f64 AST: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR_PAIR:
            raise TypeError("Destination must be a VGPR pair")
        if self.dst.type_kind != TypeKind.FLOAT64:
            raise TypeError(f"Destination type must be FLOAT64, got {self.dst.type_kind}")
        if not (0 <= self.dst.index <= 254):
            raise ValueError(f"Destination VGPR pair index out of range: {self.dst.index}")
        if self.dst.neg or self.dst.abs_:
            raise ValueError("Destination operand cannot have neg/abs modifiers")

        for i, src in enumerate((self.src0, self.src1, self.src2)):
            if src.type_kind != TypeKind.FLOAT64:
                raise TypeError(f"Source operand {i} type must be FLOAT64, got {src.type_kind}")
            if src.kind == OperandKind.VGPR_PAIR and not (0 <= src.index <= 254):
                raise ValueError(f"Source {i} VGPR pair index out of range: {src.index}")
            if src.kind == OperandKind.SGPR_PAIR and not (0 <= src.index <= 104):
                raise ValueError(f"Source {i} SGPR pair index out of range: {src.index}")

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

    def __init__(self, type_kind: TypeKind = TypeKind.FLOAT64) -> None:
        self.type_kind = type_kind

    def evaluate(self, env: dict[str, float]) -> float:
        raise NotImplementedError

    def simplify(self) -> TypedExpr:
        return self


class VariableExpr(TypedExpr):
    """Named variable in symbolic expression tree (e.g. representing a register value)."""

    def __init__(self, name: str, type_kind: TypeKind = TypeKind.FLOAT64) -> None:
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

    def __init__(self, value: float, type_kind: TypeKind = TypeKind.FLOAT64) -> None:
        super().__init__(type_kind)
        self.value = value

    def evaluate(self, env: dict[str, float]) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Const({self.value})"


class UnaryOpExpr(TypedExpr):
    """Unary modifier applied to a typed expression ('neg' or 'abs')."""

    def __init__(self, op: str, expr: TypedExpr, type_kind: TypeKind = TypeKind.FLOAT64) -> None:
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


class FMAExpr(TypedExpr):
    """Symbolic FMA ternary typed expression: (src0 * src1) + src2 with modifiers."""

    def __init__(
        self,
        src0: TypedExpr,
        src1: TypedExpr,
        src2: TypedExpr,
        *,
        clamp: bool = False,
        omod: OMod = OMod.NONE,
        type_kind: TypeKind = TypeKind.FLOAT64,
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
        return ref_fma_f64(v0, v1, v2, clamp=self.clamp, omod=self.omod.value)

    def simplify(self) -> TypedExpr:
        s0 = self.src0.simplify()
        s1 = self.src1.simplify()
        s2 = self.src2.simplify()
        if isinstance(s0, ConstantExpr) and isinstance(s1, ConstantExpr) and isinstance(s2, ConstantExpr):
            res = ref_fma_f64(s0.value, s1.value, s2.value, clamp=self.clamp, omod=self.omod.value)
            return ConstantExpr(res, self.type_kind)
        return FMAExpr(s0, s1, s2, clamp=self.clamp, omod=self.omod, type_kind=self.type_kind)

    def __repr__(self) -> str:
        return f"FMA({self.src0}, {self.src1}, {self.src2}, clamp={self.clamp}, omod={self.omod.name})"


class TestFmaF64SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_validation(self):
        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_fma_f64 v[0:1], v[2:3], v[4:5], v[6:7]")

    def test_ast_with_modifiers_and_disasm(self):
        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2, neg=True, abs_=True),
            src1=ASTOperand.sgpr(4),
            src2=ASTOperand.inline_const(1.0),
            clamp=True,
            omod=OMod.MUL2,
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_fma_f64 v[0:1], -|v[2:3]|, s[4:5], 1.0 mul:2 clamp")

    def test_ast_validation_errors(self):
        # Invalid destination kind
        bad_dst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        with self.assertRaises(TypeError):
            bad_dst.validate()

        # Destination has modifiers (illegal in RDNA2)
        bad_dst_mod = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0, neg=True),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        with self.assertRaises(ValueError):
            bad_dst_mod.validate()

        # Destination out of bounds
        bad_index = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(256),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        with self.assertRaises(ValueError):
            bad_index.validate()

    def test_typed_expressions_and_evaluation(self):
        # Symbolic: (a * b) + c
        expr = FMAExpr(
            VariableExpr("a"),
            VariableExpr("b"),
            VariableExpr("c"),
        )
        env = {"a": 2.5, "b": 4.0, "c": -3.0}
        self.assertEqual(expr.evaluate(env), 7.0)

    def test_typed_expression_constant_folding(self):
        # FMA(Const(2.0), Const(3.0), Const(4.0)) -> Const(10.0)
        c_expr = FMAExpr(
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
    """Register file for a single scalar lane with 256 VGPRs and 106 SGPRs."""

    def __init__(self) -> None:
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106

    def read_vgpr(self, idx: int) -> int:
        return self.vgprs[idx]

    def write_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32

    def read_sgpr(self, idx: int) -> int:
        return self.sgprs[idx]

    def write_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32

    def read_vgpr_pair_u64(self, idx: int) -> int:
        return dwords_to_u64(self.vgprs[idx], self.vgprs[idx + 1])

    def write_vgpr_pair_u64(self, idx: int, val: int) -> None:
        lo, hi = u64_to_dwords(val)
        self.vgprs[idx] = lo
        self.vgprs[idx + 1] = hi

    def read_vgpr_pair_f64(self, idx: int) -> float:
        return u64_to_f64(self.read_vgpr_pair_u64(idx))

    def write_vgpr_pair_f64(self, idx: int, val: float) -> None:
        self.write_vgpr_pair_u64(idx, f64_to_u64(val))

    def read_sgpr_pair_u64(self, idx: int) -> int:
        return dwords_to_u64(self.sgprs[idx], self.sgprs[idx + 1])

    def write_sgpr_pair_u64(self, idx: int, val: int) -> None:
        lo, hi = u64_to_dwords(val)
        self.sgprs[idx] = lo
        self.sgprs[idx + 1] = hi

    def read_sgpr_pair_f64(self, idx: int) -> float:
        return u64_to_f64(self.read_sgpr_pair_u64(idx))

    def write_sgpr_pair_f64(self, idx: int, val: float) -> None:
        self.write_sgpr_pair_u64(idx, f64_to_u64(val))


class FmaF64ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_fma_f64."""

    def __init__(self, reg_file: ScalarRegisterFile | None = None) -> None:
        self.reg_file = reg_file if reg_file is not None else ScalarRegisterFile()

    def resolve_source_f64(self, src: ASTOperand) -> float:
        if src.kind == OperandKind.VGPR_PAIR:
            val = self.reg_file.read_vgpr_pair_f64(src.index)
        elif src.kind == OperandKind.SGPR_PAIR:
            val = self.reg_file.read_sgpr_pair_f64(src.index)
        elif src.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            val = float(src.value) if src.value is not None else 0.0
        else:
            raise ValueError(f"Unsupported operand kind: {src.kind}")

        # Apply source modifiers
        u = apply_src_mods_u64(f64_to_u64(val), neg=src.neg, abs_=src.abs_)
        return u64_to_f64(u)

    def execute(self, inst: ASTInstruction) -> None:
        inst.validate()
        s0 = self.resolve_source_f64(inst.src0)
        s1 = self.resolve_source_f64(inst.src1)
        s2 = self.resolve_source_f64(inst.src2)

        res = ref_fma_f64(s0, s1, s2, clamp=inst.clamp, omod=inst.omod.value)
        self.reg_file.write_vgpr_pair_f64(inst.dst.index, res)


class TestFmaF64ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = FmaF64ScalarInterpreter(self.rf)

    def test_scalar_interpreter_vgprs(self):
        # v[0:1] = v[2:3] * v[4:5] + v[6:7]
        self.rf.write_vgpr_pair_f64(2, 3.0)
        self.rf.write_vgpr_pair_f64(4, 4.0)
        self.rf.write_vgpr_pair_f64(6, 5.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_pair_f64(0), 17.0)

    def test_scalar_interpreter_sgpr_and_constants(self):
        # v[0:1] = s[0:1] * v[2:3] + inline_const(10.0)
        self.rf.write_sgpr_pair_f64(0, 2.5)
        self.rf.write_vgpr_pair_f64(2, 4.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.sgpr(0),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.inline_const(10.0),
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_pair_f64(0), 20.0)

    def test_scalar_interpreter_modifiers(self):
        # v[0:1] = -|v[2:3]| * v[4:5] + 5.0 with clamp
        # v[2:3] = -2.0 -> abs -> 2.0 -> neg -> -2.0
        # -2.0 * 3.0 + 5.0 = -1.0 -> clamp -> 0.0
        self.rf.write_vgpr_pair_f64(2, -2.0)
        self.rf.write_vgpr_pair_f64(4, 3.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2, neg=True, abs_=True),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.inline_const(5.0),
            clamp=True,
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_pair_f64(0), 0.0)

    def test_scalar_interpreter_deterministic_repeatability(self):
        # Verify that repeating operations produces identical bit patterns
        self.rf.write_vgpr_pair_f64(2, 1.23456789012345)
        self.rf.write_vgpr_pair_f64(4, 9.87654321098765)
        self.rf.write_vgpr_pair_f64(6, -3.14159265358979)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        self.interp.execute(inst)
        res1 = self.rf.read_vgpr_pair_u64(0)

        self.interp.execute(inst)
        res2 = self.rf.read_vgpr_pair_u64(0)
        self.assertEqual(res1, res2)


if __name__ == "__main__":
    unittest.main()
