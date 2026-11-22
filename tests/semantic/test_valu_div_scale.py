from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_div_scale.

Tests the floating-point division pre-scaling instructions:
    v_div_scale_f32 D, sdst, S0, S1, S2
    v_div_scale_f64 D, sdst, S0, S1, S2

Lowered semantics (VOP3b dual-output):
    D = Scaled version of S0 (scaled by 2^k or 2^-k if required to prevent underflow/overflow)
    sdst (vcc) = 1 if scaling was applied, 0 if no scaling occurred
Where:
    S0 = operand to scale (typically denominator S1 or numerator S2)
    S1 = denominator
    S2 = numerator
For f32: scale factor is 2^64 or 2^-64.
For f64: scale factor is 2^128 or 2^-128.
Supports input modifiers:
    - abs (|src|: clear sign bit)
    - neg (-src: invert sign bit)
Supports output modifiers:
    - clamp: saturates result to [0.0, 1.0], with NaN clamped to +0.0
    - omod: mul:2, mul:4, div:2
Architectural model:
    f32: 32-bit registers (VGPR / SGPR)
    f64: 64-bit register pairs mapped onto aligned VGPR/SGPR pairs [lo: 32-bit dword, hi: 32-bit dword].
"""

from enum import Enum, auto
from fractions import Fraction
import math
import struct
import sys
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# IEEE 754 single precision constants
F32_POS_ZERO = 0x00000000
F32_NEG_ZERO = 0x80000000
F32_ONE = 0x3F800000
F32_NEG_ONE = 0xBF800000
F32_TWO = 0x40000000
F32_HALF = 0x3F000000
F32_POS_INF = 0x7F800000
F32_NEG_INF = 0xFF800000
F32_CANONICAL_QNAN = 0x7FC00000
F32_SNAN = 0x7F800001
F32_SUB_MIN = 0x00000001
F32_SUB_MAX = 0x007FFFFF
F32_MIN_NORMAL = 0x00800000
F32_MAX_NORMAL = 0x7F7FFFFF

# IEEE 754 double precision constants
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

# Scaling constants
SCALE64_POS = 2.0 ** 64
SCALE64_NEG = 2.0 ** -64
SCALE128_POS = 2.0 ** 128
SCALE128_NEG = 2.0 ** -128

THRESH_F32_SMALL = 2.0 ** -88
THRESH_F32_LARGE = 2.0 ** 62
THRESH_F64_SMALL = 2.0 ** -768
THRESH_F64_LARGE = 2.0 ** 768


def f32_to_u32(val: float) -> int:
    """Pack Python float into unsigned 32-bit integer (IEEE 754 binary32)."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def u32_to_f32(bits: int) -> float:
    """Unpack unsigned 32-bit integer into Python float (IEEE 754 binary32)."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f64_to_u64(val: float) -> int:
    """Pack Python float into unsigned 64-bit integer (IEEE 754 binary64)."""
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


def is_f32_nan(bits: int) -> bool:
    return ((bits & 0x7F800000) == 0x7F800000) and ((bits & 0x007FFFFF) != 0)


def is_f32_snan(bits: int) -> bool:
    return is_f32_nan(bits) and ((bits & 0x00400000) == 0)


def is_f32_qnan(bits: int) -> bool:
    return is_f32_nan(bits) and ((bits & 0x00400000) != 0)


def quiet_f32(bits: int) -> int:
    return (bits & MASK32) | 0x00400000


def is_f64_nan(bits: int) -> bool:
    return ((bits & 0x7FF0000000000000) == 0x7FF0000000000000) and ((bits & 0x000FFFFFFFFFFFFF) != 0)


def is_f64_snan(bits: int) -> bool:
    return is_f64_nan(bits) and ((bits & 0x0008000000000000) == 0)


def is_f64_qnan(bits: int) -> bool:
    return is_f64_nan(bits) and ((bits & 0x0008000000000000) != 0)


def quiet_f64(bits: int) -> int:
    return (bits & MASK64) | 0x0008000000000000


def apply_src_mods_u32(val_u32: int, *, neg: bool = False, abs_: bool = False) -> int:
    bits = val_u32 & MASK32
    if abs_:
        bits &= ~F32_NEG_ZERO
    if neg:
        bits ^= F32_NEG_ZERO
    return bits


def apply_src_mods_u64(val_u64: int, *, neg: bool = False, abs_: bool = False) -> int:
    bits = val_u64 & MASK64
    if abs_:
        bits &= ~F64_NEG_ZERO
    if neg:
        bits ^= F64_NEG_ZERO
    return bits


def apply_clamp_f32(val: float) -> float:
    if math.isnan(val):
        return 0.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def apply_clamp_f64(val: float) -> float:
    if math.isnan(val):
        return 0.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def apply_omod_float(val: float, omod: int) -> float:
    if omod == 0:
        return val
    elif omod == 1:
        return val * 2.0
    elif omod == 2:
        return val * 4.0
    elif omod == 3:
        return val * 0.5
    else:
        raise ValueError(f"Invalid omod: {omod}")


def ref_div_scale_f32(
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
) -> tuple[float, int]:
    """Reference mathematical oracle for v_div_scale_f32.

    Returns (scaled_result, vcc_bit).
    """
    u0 = apply_src_mods_u32(f32_to_u32(s0), neg=neg0, abs_=abs0)
    u1 = apply_src_mods_u32(f32_to_u32(s1), neg=neg1, abs_=abs1)
    u2 = apply_src_mods_u32(f32_to_u32(s2), neg=neg2, abs_=abs2)

    if is_f32_nan(u0):
        res = u32_to_f32(quiet_f32(u0))
        if clamp:
            res = apply_clamp_f32(res)
        return res, 0

    if is_f32_nan(u1) or is_f32_nan(u2):
        res = u32_to_f32(u0)
        if clamp:
            res = apply_clamp_f32(res)
        return res, 0

    v0 = u32_to_f32(u0)
    v1 = u32_to_f32(u1)
    v2 = u32_to_f32(u2)

    abs_y = abs(v1)
    abs_x = abs(v2)

    scaled = False
    scale_factor = 1.0

    # Denominator check has priority
    if 0.0 < abs_y <= THRESH_F32_SMALL:
        scaled = True
        scale_factor = SCALE64_POS
    elif abs_y >= THRESH_F32_LARGE and math.isfinite(abs_y):
        scaled = True
        scale_factor = SCALE64_NEG
    elif 0.0 < abs_x <= THRESH_F32_SMALL:
        scaled = True
        scale_factor = SCALE64_POS
    elif abs_x >= THRESH_F32_LARGE and math.isfinite(abs_x):
        scaled = True
        scale_factor = SCALE64_NEG

    if scaled:
        res = v0 * scale_factor
        vcc = 1
    else:
        res = v0
        vcc = 0

    if omod != 0:
        res = apply_omod_float(res, omod)
    if clamp:
        res = apply_clamp_f32(res)

    return res, vcc


def ref_div_scale_f32_u32(
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
) -> tuple[int, int]:
    """Bit-exact reference oracle for v_div_scale_f32."""
    res_f, vcc = ref_div_scale_f32(
        u32_to_f32(u0),
        u32_to_f32(u1),
        u32_to_f32(u2),
        neg0=neg0,
        neg1=neg1,
        neg2=neg2,
        abs0=abs0,
        abs1=abs1,
        abs2=abs2,
        clamp=clamp,
        omod=omod,
    )
    return f32_to_u32(res_f), vcc


def ref_div_scale_f64(
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
) -> tuple[float, int]:
    """Reference mathematical oracle for v_div_scale_f64.

    Returns (scaled_result, vcc_bit).
    """
    u0 = apply_src_mods_u64(f64_to_u64(s0), neg=neg0, abs_=abs0)
    u1 = apply_src_mods_u64(f64_to_u64(s1), neg=neg1, abs_=abs1)
    u2 = apply_src_mods_u64(f64_to_u64(s2), neg=neg2, abs_=abs2)

    if is_f64_nan(u0):
        res = u64_to_f64(quiet_f64(u0))
        if clamp:
            res = apply_clamp_f64(res)
        return res, 0

    if is_f64_nan(u1) or is_f64_nan(u2):
        res = u64_to_f64(u0)
        if clamp:
            res = apply_clamp_f64(res)
        return res, 0

    v0 = u64_to_f64(u0)
    v1 = u64_to_f64(u1)
    v2 = u64_to_f64(u2)

    abs_y = abs(v1)
    abs_x = abs(v2)

    scaled = False
    scale_factor = 1.0

    if 0.0 < abs_y <= THRESH_F64_SMALL:
        scaled = True
        scale_factor = SCALE128_POS
    elif abs_y >= THRESH_F64_LARGE and math.isfinite(abs_y):
        scaled = True
        scale_factor = SCALE128_NEG
    elif 0.0 < abs_x <= THRESH_F64_SMALL:
        scaled = True
        scale_factor = SCALE128_POS
    elif abs_x >= THRESH_F64_LARGE and math.isfinite(abs_x):
        scaled = True
        scale_factor = SCALE128_NEG

    if scaled:
        res = v0 * scale_factor
        vcc = 1
    else:
        res = v0
        vcc = 0

    if omod != 0:
        res = apply_omod_float(res, omod)
    if clamp:
        res = apply_clamp_f64(res)

    return res, vcc


def ref_div_scale_f64_u64(
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
) -> tuple[int, int]:
    """Bit-exact reference oracle for v_div_scale_f64."""
    res_f, vcc = ref_div_scale_f64(
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
    return f64_to_u64(res_f), vcc


class RegisterPair:
    """Architectural model of an aligned 64-bit register pair (e.g. v[0:1] or s[0:1])."""

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


class TestDivScaleReferenceOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural models."""

    def test_f32_normal_no_scale(self):
        # Normal inputs: 10.0 / 2.0 -> no scale required
        res, vcc = ref_div_scale_f32(2.0, 2.0, 10.0)
        self.assertEqual(res, 2.0)
        self.assertEqual(vcc, 0)

        # Scaling numerator: 10.0 / 2.0 -> 10.0 unchanged
        res_num, vcc_num = ref_div_scale_f32(10.0, 2.0, 10.0)
        self.assertEqual(res_num, 10.0)
        self.assertEqual(vcc_num, 0)

    def test_f32_scale_up_small_denom(self):
        # Very small denominator (e.g. 2^-100): scale up by 2^64
        small_denom = 2.0 ** -100
        res, vcc = ref_div_scale_f32(small_denom, small_denom, 1.0)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, small_denom * SCALE64_POS)

    def test_f32_scale_down_large_denom(self):
        # Very large denominator (e.g. 2^70): scale down by 2^-64
        large_denom = 2.0 ** 70
        res, vcc = ref_div_scale_f32(large_denom, large_denom, 1.0)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, large_denom * SCALE64_NEG)

    def test_f64_normal_no_scale(self):
        # Normal inputs: 100.0 / 4.0
        res, vcc = ref_div_scale_f64(4.0, 4.0, 100.0)
        self.assertEqual(res, 4.0)
        self.assertEqual(vcc, 0)

    def test_f64_scale_up_small_denom(self):
        # Very small f64 denom (e.g. 2^-800): scale up by 2^128
        small_denom = 2.0 ** -800
        res, vcc = ref_div_scale_f64(small_denom, small_denom, 1.0)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, small_denom * SCALE128_POS)

    def test_f64_scale_down_large_denom(self):
        # Very large f64 denom (e.g. 2^800): scale down by 2^-128
        large_denom = 2.0 ** 800
        res, vcc = ref_div_scale_f64(large_denom, large_denom, 1.0)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, large_denom * SCALE128_NEG)

    def test_source_modifiers(self):
        # abs and neg
        res, vcc = ref_div_scale_f32(-2.0, 2.0, 10.0, abs0=True)
        self.assertEqual(res, 2.0)
        self.assertEqual(vcc, 0)

        res_neg, _ = ref_div_scale_f32(2.0, 2.0, 10.0, neg0=True)
        self.assertEqual(res_neg, -2.0)

    def test_output_modifiers(self):
        # clamp saturates to [0.0, 1.0]
        res, _ = ref_div_scale_f32(5.0, 2.0, 10.0, clamp=True)
        self.assertEqual(res, 1.0)

        res_neg, _ = ref_div_scale_f32(-5.0, 2.0, 10.0, clamp=True)
        self.assertEqual(res_neg, 0.0)

        # omod mul:2
        res_omod, _ = ref_div_scale_f32(3.0, 2.0, 10.0, omod=1)
        self.assertEqual(res_omod, 6.0)

    def test_register_pair_model(self):
        rp = RegisterPair.from_f64(1.0)
        self.assertEqual(rp.as_f64, 1.0)
        self.assertEqual(rp.as_u64, F64_ONE)
        self.assertEqual(rp.lo, 0x00000000)
        self.assertEqual(rp.hi, 0x3FF00000)


import abc
from dataclasses import dataclass


class Opcode(Enum):
    V_DIV_SCALE_F32 = "v_div_scale_f32"
    V_DIV_SCALE_F64 = "v_div_scale_f64"


class OperandKind(Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"
    VGPR_PAIR = "vgpr_pair"
    SGPR_PAIR = "sgpr_pair"
    VCC = "vcc"
    LITERAL = "literal"
    INLINE_CONST = "inline_const"


class OMod(Enum):
    NONE = 0
    MUL2 = 1
    MUL4 = 2
    DIV2 = 3


class TypeKind(Enum):
    F32 = "f32"
    F64 = "f64"


@dataclass
class ASTOperand:
    kind: OperandKind
    reg_idx: int = 0
    imm_val: float = 0.0
    neg: bool = False
    abs_: bool = False

    @classmethod
    def vgpr(cls, idx: int, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.VGPR, reg_idx=idx, neg=neg, abs_=abs_)

    @classmethod
    def sgpr(cls, idx: int, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.SGPR, reg_idx=idx, neg=neg, abs_=abs_)

    @classmethod
    def vgpr_pair(cls, idx: int, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.VGPR_PAIR, reg_idx=idx, neg=neg, abs_=abs_)

    @classmethod
    def sgpr_pair(cls, idx: int, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.SGPR_PAIR, reg_idx=idx, neg=neg, abs_=abs_)

    @classmethod
    def vcc(cls) -> ASTOperand:
        return cls(OperandKind.VCC)

    @classmethod
    def inline_const(cls, val: float, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.INLINE_CONST, imm_val=val, neg=neg, abs_=abs_)

    def format_asm(self) -> str:
        if self.kind == OperandKind.VGPR:
            s = f"v{self.reg_idx}"
        elif self.kind == OperandKind.SGPR:
            s = f"s{self.reg_idx}"
        elif self.kind == OperandKind.VGPR_PAIR:
            s = f"v[{self.reg_idx}:{self.reg_idx + 1}]"
        elif self.kind == OperandKind.SGPR_PAIR:
            s = f"s[{self.reg_idx}:{self.reg_idx + 1}]"
        elif self.kind == OperandKind.VCC:
            s = "vcc"
        elif self.kind == OperandKind.INLINE_CONST:
            s = f"{self.imm_val}"
        else:
            s = f"0x{int(self.imm_val):x}"
        if self.abs_:
            s = f"|{s}|"
        if self.neg:
            s = f"-{s}"
        return s


@dataclass
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    sdst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand
    src2: ASTOperand
    clamp: bool = False
    omod: OMod = OMod.NONE

    def validate(self) -> None:
        if not isinstance(self.opcode, Opcode):
            raise TypeError(f"Invalid opcode type: {type(self.opcode)}")

        is_f64 = (self.opcode == Opcode.V_DIV_SCALE_F64)
        expected_dst_kind = OperandKind.VGPR_PAIR if is_f64 else OperandKind.VGPR
        if self.dst.kind != expected_dst_kind:
            raise TypeError(f"{self.opcode.value} destination must be {expected_dst_kind.value}, got {self.dst.kind.value}")

        if self.dst.neg or self.dst.abs_:
            raise ValueError("Destination operand cannot have neg or abs modifiers")

        max_vgpr = 254 if is_f64 else 255
        if not (0 <= self.dst.reg_idx <= max_vgpr):
            raise ValueError(f"Destination register index out of range: {self.dst.reg_idx}")

        if self.sdst.kind not in (OperandKind.VCC, OperandKind.SGPR, OperandKind.SGPR_PAIR):
            raise TypeError(f"Scalar destination must be VCC or SGPR, got {self.sdst.kind.value}")

        for name, src in [("src0", self.src0), ("src1", self.src1), ("src2", self.src2)]:
            if src.kind in (OperandKind.VGPR, OperandKind.VGPR_PAIR):
                limit = 254 if src.kind == OperandKind.VGPR_PAIR else 255
                if not (0 <= src.reg_idx <= limit):
                    raise ValueError(f"{name} VGPR index out of range: {src.reg_idx}")
            elif src.kind in (OperandKind.SGPR, OperandKind.SGPR_PAIR):
                limit = 104 if src.kind == OperandKind.SGPR_PAIR else 105
                if not (0 <= src.reg_idx <= limit):
                    raise ValueError(f"{name} SGPR index out of range: {src.reg_idx}")

    def format_asm(self) -> str:
        s = f"{self.opcode.value} {self.dst.format_asm()}, {self.sdst.format_asm()}, {self.src0.format_asm()}, {self.src1.format_asm()}, {self.src2.format_asm()}"
        if self.clamp:
            s += " clamp"
        if self.omod == OMod.MUL2:
            s += " mul:2"
        elif self.omod == OMod.MUL4:
            s += " mul:4"
        elif self.omod == OMod.DIV2:
            s += " div:2"
        return s


class TypedExpr(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, env: dict[str, float]) -> tuple[float, int]:
        pass


@dataclass
class VariableExpr(TypedExpr):
    name: str

    def evaluate(self, env: dict[str, float]) -> tuple[float, int]:
        if self.name not in env:
            raise KeyError(f"Variable {self.name} not found in environment")
        return env[self.name], 0


@dataclass
class ConstantExpr(TypedExpr):
    val: float

    def evaluate(self, env: dict[str, float]) -> tuple[float, int]:
        return self.val, 0


@dataclass
class UnaryOpExpr(TypedExpr):
    op: str  # "neg", "abs"
    sub_expr: TypedExpr

    def evaluate(self, env: dict[str, float]) -> tuple[float, int]:
        val, _ = self.sub_expr.evaluate(env)
        if self.op == "abs":
            return abs(val), 0
        elif self.op == "neg":
            return -val, 0
        raise ValueError(f"Unknown unary op: {self.op}")


@dataclass
class DivScaleExpr(TypedExpr):
    is_f64: bool
    s0: TypedExpr
    s1: TypedExpr
    s2: TypedExpr
    clamp: bool = False
    omod: int = 0

    def evaluate(self, env: dict[str, float]) -> tuple[float, int]:
        v0, _ = self.s0.evaluate(env)
        v1, _ = self.s1.evaluate(env)
        v2, _ = self.s2.evaluate(env)
        if self.is_f64:
            return ref_div_scale_f64(v0, v1, v2, clamp=self.clamp, omod=self.omod)
        else:
            return ref_div_scale_f32(v0, v1, v2, clamp=self.clamp, omod=self.omod)


class TestSymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_f32_formatting(self):
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        inst.validate()
        self.assertEqual(inst.format_asm(), "v_div_scale_f32 v0, vcc, v1, v1, v2")

    def test_ast_f64_formatting_with_modifiers(self):
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr_pair(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr_pair(2, neg=True, abs_=True),
            src1=ASTOperand.vgpr_pair(2),
            src2=ASTOperand.sgpr_pair(4),
            clamp=True,
            omod=OMod.MUL2,
        )
        inst.validate()
        self.assertEqual(inst.format_asm(), "v_div_scale_f64 v[0:1], vcc, -|v[2:3]|, v[2:3], s[4:5] clamp mul:2")

    def test_typed_expression_evaluation(self):
        expr = DivScaleExpr(
            is_f64=False,
            s0=VariableExpr("y"),
            s1=VariableExpr("y"),
            s2=VariableExpr("x"),
        )
        # Normal inputs: 10.0 / 2.0 -> no scale
        res, vcc = expr.evaluate({"y": 2.0, "x": 10.0})
        self.assertEqual(res, 2.0)
        self.assertEqual(vcc, 0)

        # Small denominator -> scale up by 2^64
        small = 2.0 ** -100
        res_scaled, vcc_scaled = expr.evaluate({"y": small, "x": 1.0})
        self.assertEqual(vcc_scaled, 1)
        self.assertAlmostEqual(res_scaled, small * (2.0 ** 64))

    def test_ast_validation_errors(self):
        # f32 with pair dst
        with self.assertRaises(TypeError):
            bad_inst = ASTInstruction(
                opcode=Opcode.V_DIV_SCALE_F32,
                dst=ASTOperand.vgpr_pair(0),
                sdst=ASTOperand.vcc(),
                src0=ASTOperand.vgpr(1),
                src1=ASTOperand.vgpr(1),
                src2=ASTOperand.vgpr(2),
            )
            bad_inst.validate()

        # Destination with neg modifier
        with self.assertRaises(ValueError):
            bad_dst = ASTInstruction(
                opcode=Opcode.V_DIV_SCALE_F32,
                dst=ASTOperand.vgpr(0, neg=True),
                sdst=ASTOperand.vcc(),
                src0=ASTOperand.vgpr(1),
                src1=ASTOperand.vgpr(1),
                src2=ASTOperand.vgpr(2),
            )
            bad_dst.validate()

        # Destination out of bounds
        with self.assertRaises(ValueError):
            bad_oob = ASTInstruction(
                opcode=Opcode.V_DIV_SCALE_F32,
                dst=ASTOperand.vgpr(256),
                sdst=ASTOperand.vcc(),
                src0=ASTOperand.vgpr(1),
                src1=ASTOperand.vgpr(1),
                src2=ASTOperand.vgpr(2),
            )
            bad_oob.validate()


class ScalarRegisterFile:
    """Architectural register file for scalar execution with 256 VGPRs and 106 SGPRs."""

    def __init__(self) -> None:
        self.vgpr = [0] * 256
        self.sgpr = [0] * 106
        self.vcc = 0

    def get_vgpr_u32(self, idx: int) -> int:
        return self.vgpr[idx] & MASK32

    def set_vgpr_u32(self, idx: int, val: int) -> None:
        self.vgpr[idx] = val & MASK32

    def get_sgpr_u32(self, idx: int) -> int:
        return self.sgpr[idx] & MASK32

    def set_sgpr_u32(self, idx: int, val: int) -> None:
        self.sgpr[idx] = val & MASK32

    def get_vgpr_f32(self, idx: int) -> float:
        return u32_to_f32(self.get_vgpr_u32(idx))

    def set_vgpr_f32(self, idx: int, val: float) -> None:
        self.set_vgpr_u32(idx, f32_to_u32(val))

    def get_vgpr_pair_u64(self, idx: int) -> int:
        return dwords_to_u64(self.get_vgpr_u32(idx), self.get_vgpr_u32(idx + 1))

    def set_vgpr_pair_u64(self, idx: int, val: int) -> None:
        lo, hi = u64_to_dwords(val)
        self.set_vgpr_u32(idx, lo)
        self.set_vgpr_u32(idx + 1, hi)

    def get_vgpr_pair_f64(self, idx: int) -> float:
        return u64_to_f64(self.get_vgpr_pair_u64(idx))

    def set_vgpr_pair_f64(self, idx: int, val: float) -> None:
        self.set_vgpr_pair_u64(idx, f64_to_u64(val))

    def get_sgpr_pair_u64(self, idx: int) -> int:
        return dwords_to_u64(self.get_sgpr_u32(idx), self.get_sgpr_u32(idx + 1))

    def set_sgpr_pair_u64(self, idx: int, val: int) -> None:
        lo, hi = u64_to_dwords(val)
        self.set_sgpr_u32(idx, lo)
        self.set_sgpr_u32(idx + 1, hi)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_div_scale."""

    def __init__(self, rf: ScalarRegisterFile | None = None) -> None:
        self.rf = rf if rf is not None else ScalarRegisterFile()

    def resolve_f32(self, op: ASTOperand) -> float:
        if op.kind == OperandKind.VGPR:
            val = self.rf.get_vgpr_f32(op.reg_idx)
        elif op.kind == OperandKind.SGPR:
            val = u32_to_f32(self.rf.get_sgpr_u32(op.reg_idx))
        elif op.kind == OperandKind.INLINE_CONST:
            val = op.imm_val
        elif op.kind == OperandKind.LITERAL:
            val = u32_to_f32(int(op.imm_val))
        else:
            raise TypeError(f"Invalid operand kind for f32: {op.kind}")

        u = apply_src_mods_u32(f32_to_u32(val), neg=op.neg, abs_=op.abs_)
        return u32_to_f32(u)

    def resolve_f64(self, op: ASTOperand) -> float:
        if op.kind == OperandKind.VGPR_PAIR:
            val = self.rf.get_vgpr_pair_f64(op.reg_idx)
        elif op.kind == OperandKind.SGPR_PAIR:
            val = u64_to_f64(self.rf.get_sgpr_pair_u64(op.reg_idx))
        elif op.kind == OperandKind.INLINE_CONST:
            val = op.imm_val
        elif op.kind == OperandKind.LITERAL:
            val = u64_to_f64(int(op.imm_val))
        else:
            raise TypeError(f"Invalid operand kind for f64: {op.kind}")

        u = apply_src_mods_u64(f64_to_u64(val), neg=op.neg, abs_=op.abs_)
        return u64_to_f64(u)

    def step(self, inst: ASTInstruction) -> tuple[float, int]:
        inst.validate()
        omod_code = inst.omod.value

        if inst.opcode == Opcode.V_DIV_SCALE_F32:
            s0 = self.resolve_f32(inst.src0)
            s1 = self.resolve_f32(inst.src1)
            s2 = self.resolve_f32(inst.src2)
            res, vcc_bit = ref_div_scale_f32(s0, s1, s2, clamp=inst.clamp, omod=omod_code)

            # Write-back destination
            self.rf.set_vgpr_f32(inst.dst.reg_idx, res)

            # Write-back scalar destination (sdst)
            if inst.sdst.kind == OperandKind.VCC:
                self.rf.vcc = vcc_bit
            elif inst.sdst.kind == OperandKind.SGPR:
                self.rf.set_sgpr_u32(inst.sdst.reg_idx, vcc_bit)
            elif inst.sdst.kind == OperandKind.SGPR_PAIR:
                self.rf.set_sgpr_pair_u64(inst.sdst.reg_idx, vcc_bit)

            return res, vcc_bit

        elif inst.opcode == Opcode.V_DIV_SCALE_F64:
            s0 = self.resolve_f64(inst.src0)
            s1 = self.resolve_f64(inst.src1)
            s2 = self.resolve_f64(inst.src2)
            res, vcc_bit = ref_div_scale_f64(s0, s1, s2, clamp=inst.clamp, omod=omod_code)

            # Write-back destination
            self.rf.set_vgpr_pair_f64(inst.dst.reg_idx, res)

            # Write-back scalar destination
            if inst.sdst.kind == OperandKind.VCC:
                self.rf.vcc = vcc_bit
            elif inst.sdst.kind == OperandKind.SGPR:
                self.rf.set_sgpr_u32(inst.sdst.reg_idx, vcc_bit)
            elif inst.sdst.kind == OperandKind.SGPR_PAIR:
                self.rf.set_sgpr_pair_u64(inst.sdst.reg_idx, vcc_bit)

            return res, vcc_bit

        else:
            raise NotImplementedError(f"Opcode {inst.opcode} not implemented")


class TestScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def test_scalar_f32_normal_step(self):
        interp = ScalarInterpreter()
        interp.rf.set_vgpr_f32(1, 2.0)
        interp.rf.set_vgpr_f32(2, 10.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        res, vcc = interp.step(inst)
        self.assertEqual(res, 2.0)
        self.assertEqual(vcc, 0)
        self.assertEqual(interp.rf.get_vgpr_f32(0), 2.0)
        self.assertEqual(interp.rf.vcc, 0)

    def test_scalar_f32_scale_up_small_denom(self):
        interp = ScalarInterpreter()
        small = 2.0 ** -100
        interp.rf.set_vgpr_f32(1, small)
        interp.rf.set_vgpr_f32(2, 1.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        res, vcc = interp.step(inst)
        self.assertEqual(vcc, 1)
        self.assertEqual(interp.rf.vcc, 1)
        self.assertAlmostEqual(res, small * (2.0 ** 64))
        self.assertAlmostEqual(interp.rf.get_vgpr_f32(0), small * (2.0 ** 64))

    def test_scalar_f64_scale_down_large_denom(self):
        interp = ScalarInterpreter()
        large = 2.0 ** 800
        interp.rf.set_vgpr_pair_f64(2, large)
        interp.rf.set_vgpr_pair_f64(4, 1.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr_pair(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr_pair(2),
            src1=ASTOperand.vgpr_pair(2),
            src2=ASTOperand.vgpr_pair(4),
        )
        res, vcc = interp.step(inst)
        self.assertEqual(vcc, 1)
        self.assertEqual(interp.rf.vcc, 1)
        self.assertAlmostEqual(res, large * (2.0 ** -128))
        self.assertAlmostEqual(interp.rf.get_vgpr_pair_f64(0), large * (2.0 ** -128))

    def test_scalar_determinism_50_runs(self):
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        results = []
        for _ in range(50):
            interp = ScalarInterpreter()
            interp.rf.set_vgpr_f32(1, 3.14159)
            interp.rf.set_vgpr_f32(2, 2.71828)
            res, vcc = interp.step(inst)
            results.append((res, vcc))

        self.assertTrue(all(r == results[0] for r in results))


class SIMDWaveState:
    """Architectural SIMD Wavefront state supporting Wave32 and Wave64 with EXEC masking."""

    def __init__(self, wave_size: int = 64) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self.vgprs = [[0] * 256 for _ in range(wave_size)]
        self.sgprs = [0] * 106
        self.exec_mask = (1 << wave_size) - 1
        self.vcc = 0

    def get_lane_vgpr_u32(self, lane: int, reg: int) -> int:
        return self.vgprs[lane][reg] & MASK32

    def set_lane_vgpr_u32(self, lane: int, reg: int, val: int) -> None:
        self.vgprs[lane][reg] = val & MASK32

    def get_lane_vgpr_f32(self, lane: int, reg: int) -> float:
        return u32_to_f32(self.get_lane_vgpr_u32(lane, reg))

    def set_lane_vgpr_f32(self, lane: int, reg: int, val: float) -> None:
        self.set_lane_vgpr_u32(lane, reg, f32_to_u32(val))

    def get_lane_vgpr_pair_u64(self, lane: int, reg: int) -> int:
        return dwords_to_u64(self.get_lane_vgpr_u32(lane, reg), self.get_lane_vgpr_u32(lane, reg + 1))

    def set_lane_vgpr_pair_u64(self, lane: int, reg: int, val: int) -> None:
        lo, hi = u64_to_dwords(val)
        self.set_lane_vgpr_u32(lane, reg, lo)
        self.set_lane_vgpr_u32(lane, reg + 1, hi)

    def get_lane_vgpr_pair_f64(self, lane: int, reg: int) -> float:
        return u64_to_f64(self.get_lane_vgpr_pair_u64(lane, reg))

    def set_lane_vgpr_pair_f64(self, lane: int, reg: int, val: float) -> None:
        self.set_lane_vgpr_pair_u64(lane, reg, f64_to_u64(val))

    def get_sgpr_u32(self, reg: int) -> int:
        return self.sgprs[reg] & MASK32

    def set_sgpr_u32(self, reg: int, val: int) -> None:
        self.sgprs[reg] = val & MASK32

    def get_sgpr_pair_u64(self, reg: int) -> int:
        return dwords_to_u64(self.get_sgpr_u32(reg), self.get_sgpr_u32(reg + 1))

    def set_sgpr_pair_u64(self, reg: int, val: int) -> None:
        lo, hi = u64_to_dwords(val)
        self.set_sgpr_u32(reg, lo)
        self.set_sgpr_u32(reg + 1, hi)


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def __init__(self, wave: SIMDWaveState | None = None) -> None:
        self.wave = wave if wave is not None else SIMDWaveState()

    def resolve_lane_f32(self, lane: int, op: ASTOperand) -> float:
        if op.kind == OperandKind.VGPR:
            val = self.wave.get_lane_vgpr_f32(lane, op.reg_idx)
        elif op.kind == OperandKind.SGPR:
            val = u32_to_f32(self.wave.get_sgpr_u32(op.reg_idx))
        elif op.kind == OperandKind.INLINE_CONST:
            val = op.imm_val
        elif op.kind == OperandKind.LITERAL:
            val = u32_to_f32(int(op.imm_val))
        else:
            raise TypeError(f"Invalid operand kind for f32: {op.kind}")

        u = apply_src_mods_u32(f32_to_u32(val), neg=op.neg, abs_=op.abs_)
        return u32_to_f32(u)

    def resolve_lane_f64(self, lane: int, op: ASTOperand) -> float:
        if op.kind == OperandKind.VGPR_PAIR:
            val = self.wave.get_lane_vgpr_pair_f64(lane, op.reg_idx)
        elif op.kind == OperandKind.SGPR_PAIR:
            val = u64_to_f64(self.wave.get_sgpr_pair_u64(op.reg_idx))
        elif op.kind == OperandKind.INLINE_CONST:
            val = op.imm_val
        elif op.kind == OperandKind.LITERAL:
            val = u64_to_f64(int(op.imm_val))
        else:
            raise TypeError(f"Invalid operand kind for f64: {op.kind}")

        u = apply_src_mods_u64(f64_to_u64(val), neg=op.neg, abs_=op.abs_)
        return u64_to_f64(u)

    def step(self, inst: ASTInstruction) -> tuple[list[float | None], int]:
        inst.validate()
        wave = self.wave
        active_mask = wave.exec_mask & ((1 << wave.wave_size) - 1)
        omod_code = inst.omod.value

        results: list[float | None] = [None] * wave.wave_size
        vcc_out = wave.vcc  # Retain untouched bits for inactive lanes

        if inst.opcode == Opcode.V_DIV_SCALE_F32:
            # Latch phase
            s0_vals = [self.resolve_lane_f32(l, inst.src0) for l in range(wave.wave_size)]
            s1_vals = [self.resolve_lane_f32(l, inst.src1) for l in range(wave.wave_size)]
            s2_vals = [self.resolve_lane_f32(l, inst.src2) for l in range(wave.wave_size)]

            # Execution phase
            for lane in range(wave.wave_size):
                if (active_mask >> lane) & 1:
                    res, vcc_bit = ref_div_scale_f32(
                        s0_vals[lane], s1_vals[lane], s2_vals[lane],
                        clamp=inst.clamp, omod=omod_code
                    )
                    wave.set_lane_vgpr_f32(lane, inst.dst.reg_idx, res)
                    results[lane] = res
                    if vcc_bit:
                        vcc_out |= (1 << lane)
                    else:
                        vcc_out &= ~(1 << lane)

        elif inst.opcode == Opcode.V_DIV_SCALE_F64:
            # Latch phase
            s0_vals = [self.resolve_lane_f64(l, inst.src0) for l in range(wave.wave_size)]
            s1_vals = [self.resolve_lane_f64(l, inst.src1) for l in range(wave.wave_size)]
            s2_vals = [self.resolve_lane_f64(l, inst.src2) for l in range(wave.wave_size)]

            # Execution phase
            for lane in range(wave.wave_size):
                if (active_mask >> lane) & 1:
                    res, vcc_bit = ref_div_scale_f64(
                        s0_vals[lane], s1_vals[lane], s2_vals[lane],
                        clamp=inst.clamp, omod=omod_code
                    )
                    wave.set_lane_vgpr_pair_f64(lane, inst.dst.reg_idx, res)
                    results[lane] = res
                    if vcc_bit:
                        vcc_out |= (1 << lane)
                    else:
                        vcc_out &= ~(1 << lane)

        # Update scalar destination
        wave.vcc = vcc_out
        if inst.sdst.kind == OperandKind.SGPR:
            wave.set_sgpr_u32(inst.sdst.reg_idx, vcc_out & MASK32)
        elif inst.sdst.kind == OperandKind.SGPR_PAIR:
            wave.set_sgpr_pair_u64(inst.sdst.reg_idx, vcc_out & MASK64)

        return results, vcc_out


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_full_exec(self):
        wave = SIMDWaveState(wave_size=32)
        interp = SIMDInterpreter(wave)
        for l in range(32):
            wave.set_lane_vgpr_f32(l, 1, float(l + 1))
            wave.set_lane_vgpr_f32(l, 2, 100.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        res, vcc = interp.step(inst)
        self.assertEqual(vcc, 0)
        for l in range(32):
            self.assertEqual(res[l], float(l + 1))
            self.assertEqual(wave.get_lane_vgpr_f32(l, 0), float(l + 1))

    def test_wave32_divergent_mask(self):
        wave = SIMDWaveState(wave_size=32)
        # Even lanes active (0x55555555)
        wave.exec_mask = 0x55555555
        interp = SIMDInterpreter(wave)
        sentinel = 999.0
        for l in range(32):
            wave.set_lane_vgpr_f32(l, 0, sentinel)
            # small denom in lane 0 and 2 -> scale up
            val = (2.0 ** -100) if (l % 2 == 0) else 10.0
            wave.set_lane_vgpr_f32(l, 1, val)
            wave.set_lane_vgpr_f32(l, 2, 1.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        res, vcc = interp.step(inst)
        # All even lanes were scaled -> vcc bit set for even lanes
        self.assertEqual(vcc, 0x55555555)
        for l in range(32):
            if l % 2 == 0:
                self.assertIsNotNone(res[l])
                self.assertAlmostEqual(wave.get_lane_vgpr_f32(l, 0), (2.0 ** -100) * (2.0 ** 64))
            else:
                self.assertIsNone(res[l])
                self.assertEqual(wave.get_lane_vgpr_f32(l, 0), sentinel)

    def test_wave64_full_and_divergent(self):
        wave = SIMDWaveState(wave_size=64)
        # Lower 32 active, upper 32 inactive
        wave.exec_mask = 0x00000000FFFFFFFF
        interp = SIMDInterpreter(wave)
        sentinel_pair = 12345.0
        for l in range(64):
            wave.set_lane_vgpr_pair_f64(l, 0, sentinel_pair)
            wave.set_lane_vgpr_pair_f64(l, 2, 2.0)
            wave.set_lane_vgpr_pair_f64(l, 4, 10.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr_pair(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr_pair(2),
            src1=ASTOperand.vgpr_pair(2),
            src2=ASTOperand.vgpr_pair(4),
        )
        res, vcc = interp.step(inst)
        self.assertEqual(vcc, 0)
        for l in range(64):
            if l < 32:
                self.assertEqual(wave.get_lane_vgpr_pair_f64(l, 0), 2.0)
            else:
                self.assertEqual(wave.get_lane_vgpr_pair_f64(l, 0), sentinel_pair)

    def test_uniform_sgpr_broadcast(self):
        wave = SIMDWaveState(wave_size=32)
        interp = SIMDInterpreter(wave)
        wave.set_sgpr_u32(10, f32_to_u32(2.0 ** -100))
        for l in range(32):
            wave.set_lane_vgpr_f32(l, 2, 5.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.sgpr(10),
            src1=ASTOperand.sgpr(10),
            src2=ASTOperand.vgpr(2),
        )
        res, vcc = interp.step(inst)
        self.assertEqual(vcc, 0xFFFFFFFF)
        for l in range(32):
            self.assertAlmostEqual(wave.get_lane_vgpr_f32(l, 0), (2.0 ** -100) * (2.0 ** 64))


class TestRegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_inplace_dst_equals_src0_and_src1_f32(self):
        interp = ScalarInterpreter()
        small = 2.0 ** -100
        interp.rf.set_vgpr_f32(1, small)
        interp.rf.set_vgpr_f32(2, 5.0)

        # v1 = v_div_scale_f32 v1, vcc, v1, v1, v2
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(1),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        res, vcc = interp.step(inst)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, small * (2.0 ** 64))
        self.assertAlmostEqual(interp.rf.get_vgpr_f32(1), small * (2.0 ** 64))
        # src2 (v2) must remain intact
        self.assertEqual(interp.rf.get_vgpr_f32(2), 5.0)

    def test_inplace_dst_equals_src0_and_src2_f32(self):
        interp = ScalarInterpreter()
        interp.rf.set_vgpr_f32(1, 2.0)
        interp.rf.set_vgpr_f32(2, 10.0)

        # v2 = v_div_scale_f32 v2, vcc, v2, v1, v2
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(2),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        res, vcc = interp.step(inst)
        self.assertEqual(vcc, 0)
        self.assertEqual(res, 10.0)
        self.assertEqual(interp.rf.get_vgpr_f32(2), 10.0)
        # src1 (v1) must remain intact
        self.assertEqual(interp.rf.get_vgpr_f32(1), 2.0)

    def test_inplace_f64_self_aliasing(self):
        interp = ScalarInterpreter()
        small = 2.0 ** -800
        interp.rf.set_vgpr_pair_f64(0, small)

        # v[0:1] = v_div_scale_f64 v[0:1], vcc, v[0:1], v[0:1], v[0:1]
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr_pair(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr_pair(0),
            src1=ASTOperand.vgpr_pair(0),
            src2=ASTOperand.vgpr_pair(0),
        )
        res, vcc = interp.step(inst)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, small * (2.0 ** 128))
        self.assertAlmostEqual(interp.rf.get_vgpr_pair_f64(0), small * (2.0 ** 128))

    def test_f64_partially_overlapping_register_pairs(self):
        interp = ScalarInterpreter()
        # s0 in v[0:1], s1 in v[2:3], dst in v[1:2] (overlaps v1 and v2)
        v0_raw = 0xAAAA5555
        v3_raw = 0x3333CCCC
        interp.rf.set_vgpr_u32(0, v0_raw)
        interp.rf.set_vgpr_u32(1, 0x11111111)
        interp.rf.set_vgpr_u32(2, 0x22222222)
        interp.rf.set_vgpr_u32(3, v3_raw)

        s0_val = interp.rf.get_vgpr_pair_f64(0)
        s1_val = interp.rf.get_vgpr_pair_f64(2)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr_pair(1),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr_pair(0),
            src1=ASTOperand.vgpr_pair(0),
            src2=ASTOperand.vgpr_pair(2),
        )
        res, vcc = interp.step(inst)

        # v[1:2] must hold result
        self.assertEqual(interp.rf.get_vgpr_pair_f64(1), res)
        # Boundary dwords v[0] and v[3] must remain untouched!
        self.assertEqual(interp.rf.get_vgpr_u32(0), v0_raw)
        self.assertEqual(interp.rf.get_vgpr_u32(3), v3_raw)

    def test_simd_inplace_divergent_mask(self):
        wave = SIMDWaveState(wave_size=32)
        # Odd lanes active
        wave.exec_mask = 0xAAAAAAAA
        interp = SIMDInterpreter(wave)

        small = 2.0 ** -100
        for l in range(32):
            wave.set_lane_vgpr_f32(l, 0, float(l + 1))
            wave.set_lane_vgpr_f32(l, 1, small)
            wave.set_lane_vgpr_f32(l, 2, 1.0)

        # v0 = v_div_scale_f32 v0, vcc, v1, v1, v2
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        res, vcc = interp.step(inst)
        self.assertEqual(vcc, 0xAAAAAAAA)
        for l in range(32):
            if l % 2 == 1:
                # Active lane: updated to scaled value
                self.assertAlmostEqual(wave.get_lane_vgpr_f32(l, 0), small * (2.0 ** 64))
            else:
                # Inactive lane: retained original float(l + 1)
                self.assertEqual(wave.get_lane_vgpr_f32(l, 0), float(l + 1))


if __name__ == "__main__":
    unittest.main()




