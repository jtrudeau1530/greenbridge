"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_mul_f16.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  Operations on 16-bit half-precision floating-point quantities (binary16, IEEE 754)
  operate on 16-bit packed or scalar sub-words:
    - VOP2: v_mul_f16 v[d], src0, v[s1]
    - VOP3: v_mul_f16 v[d], src0, src1 [clamp] [omod] [op_sel]

  IEEE 754-2008 binary16 encoding:
    - 1 sign bit (bit 15)
    - 5 exponent bits with bias 15 (bits 14:10)
    - 10 mantissa / fraction bits (bits 9:0)

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
import math
import re
import struct
import unittest
from fractions import Fraction
from typing import Tuple

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF

# IEEE 754 binary16 bit constants
SIGN_MASK_F16 = 0x8000
EXP_MASK_F16 = 0x7C00
MANT_MASK_F16 = 0x03FF
QNAN_BIT_F16 = 0x0200
CANONICAL_QNAN_F16 = 0x7E00
SNAN_F16 = 0x7D00

POS_ZERO_F16 = 0x0000
NEG_ZERO_F16 = 0x8000
POS_INF_F16 = 0x7C00
NEG_INF_F16 = 0xFC00

POS_ONE_F16 = 0x3C00          # 1.0
NEG_ONE_F16 = 0xBC00          # -1.0
POS_TWO_F16 = 0x4000          # 2.0
NEG_TWO_F16 = 0xC000          # -2.0
POS_HALF_F16 = 0x3800         # 0.5
NEG_HALF_F16 = 0xB800         # -0.5

MAX_FINITE_F16 = 0x7BFF       # 65504.0
MIN_NORMAL_F16 = 0x0400       # 2^-14 (~6.1035e-5)
MAX_SUBNORMAL_F16 = 0x03FF    # ~6.0975e-5
MIN_SUBNORMAL_F16 = 0x0001    # 2^-24 (~5.9605e-8)

DEFAULT_VGPR_SENTINEL = 0xA55A5AA5


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit range [0, 65535]."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def f16_to_bits(val: float) -> int:
    """Pack a Python float into an IEEE 754 half-precision 16-bit unsigned integer."""
    if math.isnan(val):
        return CANONICAL_QNAN_F16
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return NEG_INF_F16 if val < 0.0 else POS_INF_F16


def bits_to_f16(bits: int) -> float:
    """Unpack an unsigned 16-bit integer into a Python float via IEEE 754 half-precision."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def split_f16(bits: int) -> Tuple[int, int, int]:
    """Split 16-bit float into (sign, exponent, mantissa)."""
    b = bits & MASK16
    sign = (b >> 15) & 1
    exp = (b >> 10) & 0x1F
    mant = b & 0x03FF
    return sign, exp, mant


def is_nan_f16(bits: int) -> bool:
    """Return True if 16-bit pattern represents an IEEE 754 NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) != 0)


def is_snan_f16(bits: int) -> bool:
    """Return True if 16-bit pattern represents a signaling NaN."""
    return is_nan_f16(bits) and ((bits & QNAN_BIT_F16) == 0)


def is_qnan_f16(bits: int) -> bool:
    """Return True if 16-bit pattern represents a quiet NaN."""
    return is_nan_f16(bits) and ((bits & QNAN_BIT_F16) != 0)


def is_inf_f16(bits: int) -> bool:
    """Return True if 16-bit pattern represents +inf or -inf."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) == 0)


def is_zero_f16(bits: int) -> bool:
    """Return True if 16-bit pattern represents +0.0 or -0.0."""
    return (bits & ~SIGN_MASK_F16) == 0


def is_subnormal_f16(bits: int) -> bool:
    """Return True if 16-bit pattern represents a subnormal (denormal) float."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & MANT_MASK_F16) != 0)


def is_normal_f16(bits: int) -> bool:
    """Return True if 16-bit pattern represents a normalized finite float."""
    exp = (bits >> 10) & 0x1F
    return 1 <= exp <= 30


def apply_f16_modifiers(bits: int, *, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & MASK16
    if abs_mod:
        res &= ~SIGN_MASK_F16
    if neg_mod:
        res ^= SIGN_MASK_F16
    return res


def ref_mul_f16(
    s0_bits: int,
    s1_bits: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
    omod: str = "none",
) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_mul_f16 half-precision multiplication.

    Computes: D = s0 * s1 conforming to IEEE 754-2008 binary16, applying
    input modifiers, output clamp, and omod multipliers.
    """
    s0 = apply_f16_modifiers(s0_bits, abs_mod=abs0, neg_mod=neg0)
    s1 = apply_f16_modifiers(s1_bits, abs_mod=abs1, neg_mod=neg1)

    # 1. NaN propagation: if any operand is NaN, produce quieted NaN
    if is_nan_f16(s0) or is_nan_f16(s1):
        if clamp:
            # RDNA2 architecture rule: Clamping NaN produces +0.0
            return POS_ZERO_F16
        nan_bits = s0 if is_nan_f16(s0) else s1
        return (nan_bits | QNAN_BIT_F16) & MASK16

    sign = ((s0 >> 15) ^ (s1 >> 15)) & 1

    # 2. Multiplication of 0 by Infinity is Invalid Operation -> Canonical QNaN
    if (is_zero_f16(s0) and is_inf_f16(s1)) or (is_inf_f16(s0) and is_zero_f16(s1)):
        if clamp:
            return POS_ZERO_F16
        return CANONICAL_QNAN_F16

    # 3. Infinity * Finite -> Infinity
    if is_inf_f16(s0) or is_inf_f16(s1):
        res = (sign << 15) | POS_INF_F16
        if clamp:
            # Clamping [0.0, 1.0]: -inf -> 0.0, +inf -> 1.0
            return POS_ZERO_F16 if sign else POS_ONE_F16
        return res

    # 4. Zero * Finite -> Zero
    if is_zero_f16(s0) or is_zero_f16(s1):
        res = (sign << 15) | POS_ZERO_F16
        if clamp:
            # -0.0 clamped to [0.0, 1.0] produces +0.0
            return POS_ZERO_F16
        return res

    # 5. Finite non-zero arithmetic via IEEE 754 half precision
    val0 = bits_to_f16(s0)
    val1 = bits_to_f16(s1)
    prod = val0 * val1

    # Apply omod multiplier
    if omod == "mul:2":
        prod *= 2.0
    elif omod == "mul:4":
        prod *= 4.0
    elif omod == "div:2":
        prod *= 0.5

    # Apply clamp to [0.0, 1.0]
    if clamp:
        if math.isnan(prod) or prod < 0.0:
            return POS_ZERO_F16
        if prod > 1.0:
            return POS_ONE_F16

    return f16_to_bits(prod)


def ref_mul_f16_rational(s0_bits: int, s1_bits: int) -> int:
    """Exact rational multiplication cross-check for finite binary16 operands."""
    s0 = s0_bits & MASK16
    s1 = s1_bits & MASK16
    if is_nan_f16(s0) or is_nan_f16(s1):
        return CANONICAL_QNAN_F16
    if (is_zero_f16(s0) and is_inf_f16(s1)) or (is_inf_f16(s0) and is_zero_f16(s1)):
        return CANONICAL_QNAN_F16
    sign = ((s0 >> 15) ^ (s1 >> 15)) & 1
    if is_inf_f16(s0) or is_inf_f16(s1):
        return (sign << 15) | POS_INF_F16
    if is_zero_f16(s0) or is_zero_f16(s1):
        return (sign << 15) | POS_ZERO_F16

    def decode_fraction(b: int) -> Fraction:
        exp = (b >> 10) & 0x1F
        mant = b & 0x03FF
        if exp == 0:
            # Subnormal: (-1)^sign * 2^-14 * (mant / 1024)
            return Fraction(mant, 1024 * (1 << 14))
        # Normal: (-1)^sign * 2^(exp - 15) * (1 + mant / 1024)
        sig = Fraction(1024 + mant, 1024)
        p = exp - 15
        if p >= 0:
            return sig * (1 << p)
        return sig / (1 << (-p))

    frac_prod = decode_fraction(s0) * decode_fraction(s1)
    val = float(frac_prod)
    if sign:
        val = -val
    return f16_to_bits(val)


class RegisterFile:
    """Architectural register model supporting 32-bit VGPR/SGPR storage and 16-bit halfword access."""

    def __init__(self, num_vgprs: int = 64, num_sgprs: int = 32):
        self.vgprs = [DEFAULT_VGPR_SENTINEL] * num_vgprs
        self.sgprs = [0] * num_sgprs

    def get_vgpr(self, idx: int) -> int:
        return self.vgprs[idx] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32

    def get_vgpr_half(self, idx: int, hi: bool) -> int:
        dword = self.vgprs[idx]
        return ((dword >> 16) if hi else dword) & MASK16

    def set_vgpr_half(self, idx: int, hi: bool, val: int) -> None:
        val16 = val & MASK16
        old = self.vgprs[idx]
        if hi:
            self.vgprs[idx] = (old & 0x0000FFFF) | (val16 << 16)
        else:
            self.vgprs[idx] = (old & 0xFFFF0000) | val16

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs[idx] & MASK32

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32

    def get_sgpr_half(self, idx: int, hi: bool) -> int:
        dword = self.sgprs[idx]
        return ((dword >> 16) if hi else dword) & MASK16

    def set_sgpr_half(self, idx: int, hi: bool, val: int) -> None:
        val16 = val & MASK16
        old = self.sgprs[idx]
        if hi:
            self.sgprs[idx] = (old & 0x0000FFFF) | (val16 << 16)
        else:
            self.sgprs[idx] = (old & 0xFFFF0000) | val16


class DataType(enum.Enum):
    """Data types supported by the symbolic VALU AST."""
    F16 = "f16"
    F32 = "f32"
    I32 = "i32"
    U32 = "u32"


class OperandKind(enum.Enum):
    """Operand categories for RDNA2 VALU instructions."""
    VGPR = "vgpr"
    SGPR = "sgpr"
    LITERAL = "literal"
    INLINE_FLOAT = "inline_float"


@dataclass
class ASTOperand:
    """Symbolic typed operand with modifier attributes."""
    kind: OperandKind
    reg_index: int = 0
    hi: bool = False
    abs_mod: bool = False
    neg_mod: bool = False
    literal_val: int = 0
    data_type: DataType = DataType.F16

    def to_asm(self) -> str:
        if self.kind == OperandKind.VGPR:
            base = f"v{self.reg_index}" + (".h" if self.hi else ".l")
        elif self.kind == OperandKind.SGPR:
            base = f"s{self.reg_index}" + (".h" if self.hi else ".l")
        elif self.kind == OperandKind.INLINE_FLOAT:
            base = f"{bits_to_f16(self.literal_val)}"
        elif self.kind == OperandKind.LITERAL:
            base = f"0x{self.literal_val:04X}"
        else:
            base = "?"

        if self.abs_mod:
            base = f"|{base}|"
        if self.neg_mod:
            base = f"-{base}"
        return base


@dataclass
class ASTInstruction:
    """Symbolic AST node for v_mul_f16 instruction with typed validation."""
    opcode: str
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand
    clamp: bool = False
    omod: str = "none"

    def validate(self) -> None:
        if self.opcode != "v_mul_f16":
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR:
            raise TypeError("Destination must be a VGPR")
        if self.dst.abs_mod or self.dst.neg_mod:
            raise ValueError("Destination cannot have input modifiers (abs/neg)")
        if self.omod not in ("none", "mul:2", "mul:4", "div:2"):
            raise ValueError(f"Invalid omod specifier: {self.omod}")

    def to_asm(self) -> str:
        parts = [f"{self.opcode} {self.dst.to_asm()}, {self.src0.to_asm()}, {self.src1.to_asm()}"]
        if self.clamp:
            parts.append("clamp")
        if self.omod != "none":
            parts.append(self.omod)
        return " ".join(parts)


def parse_v_mul_f16_asm(asm_line: str) -> ASTInstruction:
    """Parse a single assembly instruction line into an ASTInstruction node."""
    line = asm_line.strip()
    match = re.match(r"^v_mul_f16\s+([^,]+),\s*([^,]+),\s*(\S+)(.*)$", line)
    if not match:
        raise ValueError(f"Cannot parse v_mul_f16 assembly: {asm_line}")

    dst_str, src0_str, src1_str, trailing = match.groups()
    clamp = "clamp" in trailing
    omod = "none"
    for candidate in ("mul:2", "mul:4", "div:2"):
        if candidate in trailing:
            omod = candidate

    def parse_operand(op_str: str) -> ASTOperand:
        s = op_str.strip()
        neg_mod = False
        abs_mod = False

        if s.startswith("-"):
            neg_mod = True
            s = s[1:].strip()

        if s.startswith("|") and s.endswith("|"):
            abs_mod = True
            s = s[1:-1].strip()

        # Check VGPR or SGPR
        vgpr_match = re.match(r"^v(\d+)(\.([lh]))?$", s)
        if vgpr_match:
            idx = int(vgpr_match.group(1))
            hi = vgpr_match.group(3) == "h"
            return ASTOperand(kind=OperandKind.VGPR, reg_index=idx, hi=hi, abs_mod=abs_mod, neg_mod=neg_mod)

        sgpr_match = re.match(r"^s(\d+)(\.([lh]))?$", s)
        if sgpr_match:
            idx = int(sgpr_match.group(1))
            hi = sgpr_match.group(3) == "h"
            return ASTOperand(kind=OperandKind.SGPR, reg_index=idx, hi=hi, abs_mod=abs_mod, neg_mod=neg_mod)

        # Hex literal
        if s.startswith("0x") or s.startswith("0X"):
            val = int(s, 16)
            return ASTOperand(kind=OperandKind.LITERAL, literal_val=val, abs_mod=abs_mod, neg_mod=neg_mod)

        # Float constant
        try:
            val_f = float(s)
            val_bits = f16_to_bits(val_f)
            return ASTOperand(kind=OperandKind.INLINE_FLOAT, literal_val=val_bits, abs_mod=abs_mod, neg_mod=neg_mod)
        except ValueError:
            pass

        raise ValueError(f"Unrecognized operand format: {op_str}")

    dst_op = parse_operand(dst_str)
    src0_op = parse_operand(src0_str)
    src1_op = parse_operand(src1_str)

    ast = ASTInstruction(opcode="v_mul_f16", dst=dst_op, src0=src0_op, src1=src1_op, clamp=clamp, omod=omod)
    ast.validate()
    return ast


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 v_mul_f16."""

    def __init__(self, num_vgprs: int = 64, num_sgprs: int = 32):
        self.reg_file = RegisterFile(num_vgprs=num_vgprs, num_sgprs=num_sgprs)

    def resolve_operand(self, op: ASTOperand) -> int:
        """Resolve an AST operand to its raw 16-bit binary representation."""
        if op.kind == OperandKind.VGPR:
            return self.reg_file.get_vgpr_half(op.reg_index, hi=op.hi)
        elif op.kind == OperandKind.SGPR:
            return self.reg_file.get_sgpr_half(op.reg_index, hi=op.hi)
        elif op.kind in (OperandKind.LITERAL, OperandKind.INLINE_FLOAT):
            return op.literal_val & MASK16
        raise ValueError(f"Cannot resolve operand: {op}")

    def execute_ast(self, inst: ASTInstruction) -> int:
        """Execute a validated AST instruction and commit result to destination VGPR."""
        inst.validate()
        raw_s0 = self.resolve_operand(inst.src0)
        raw_s1 = self.resolve_operand(inst.src1)

        res = ref_mul_f16(
            raw_s0,
            raw_s1,
            abs0=inst.src0.abs_mod,
            neg0=inst.src0.neg_mod,
            abs1=inst.src1.abs_mod,
            neg1=inst.src1.neg_mod,
            clamp=inst.clamp,
            omod=inst.omod,
        )

        self.reg_file.set_vgpr_half(inst.dst.reg_index, hi=inst.dst.hi, val=res)
        return res

    def execute_asm(self, asm_line: str) -> int:
        """Parse assembly string and execute on the scalar interpreter."""
        inst = parse_v_mul_f16_asm(asm_line)
        return self.execute_ast(inst)


class TestValuMulF16Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_oracle_normal_multiplication(self) -> None:
        # 2.0 * 3.0 = 6.0
        f_2 = f16_to_bits(2.0)
        f_3 = f16_to_bits(3.0)
        res = ref_mul_f16(f_2, f_3)
        self.assertEqual(bits_to_f16(res), 6.0)

        # -1.5 * 4.0 = -6.0
        f_neg1p5 = f16_to_bits(-1.5)
        f_4 = f16_to_bits(4.0)
        res_neg = ref_mul_f16(f_neg1p5, f_4)
        self.assertEqual(bits_to_f16(res_neg), -6.0)

    def test_oracle_cross_check_rational(self) -> None:
        test_pairs = [
            (f16_to_bits(1.5), f16_to_bits(2.5)),
            (f16_to_bits(-0.75), f16_to_bits(8.0)),
            (f16_to_bits(0.0625), f16_to_bits(16.0)),
            (f16_to_bits(3.0), f16_to_bits(0.333251953125)),
        ]
        for s0, s1 in test_pairs:
            self.assertEqual(ref_mul_f16(s0, s1), ref_mul_f16_rational(s0, s1))

    def test_oracle_zero_and_infinity_invariants(self) -> None:
        # 0.0 * inf = NaN
        self.assertEqual(ref_mul_f16(POS_ZERO_F16, POS_INF_F16), CANONICAL_QNAN_F16)
        self.assertEqual(ref_mul_f16(NEG_ZERO_F16, POS_INF_F16), CANONICAL_QNAN_F16)
        self.assertEqual(ref_mul_f16(POS_ZERO_F16, NEG_INF_F16), CANONICAL_QNAN_F16)

        # 0.0 * 5.0 = 0.0 (sign rules)
        f_5 = f16_to_bits(5.0)
        self.assertEqual(ref_mul_f16(POS_ZERO_F16, f_5), POS_ZERO_F16)
        self.assertEqual(ref_mul_f16(NEG_ZERO_F16, f_5), NEG_ZERO_F16)

        # inf * -2.0 = -inf
        self.assertEqual(ref_mul_f16(POS_INF_F16, NEG_TWO_F16), NEG_INF_F16)
        self.assertEqual(ref_mul_f16(NEG_INF_F16, NEG_TWO_F16), POS_INF_F16)

    def test_oracle_nan_propagation(self) -> None:
        # QNaN propagation
        qnan = CANONICAL_QNAN_F16
        res = ref_mul_f16(qnan, POS_TWO_F16)
        self.assertTrue(is_qnan_f16(res))

        # SNaN quietened
        res_snan = ref_mul_f16(SNAN_F16, POS_ONE_F16)
        self.assertTrue(is_qnan_f16(res_snan))

    def test_oracle_subnormal_gradual_underflow(self) -> None:
        # Subnormal * 2.0
        sub = MIN_SUBNORMAL_F16
        res = ref_mul_f16(sub, POS_TWO_F16)
        self.assertEqual(res, 0x0002)

    def test_oracle_modifiers(self) -> None:
        # abs and neg modifiers
        f_neg2 = f16_to_bits(-2.0)
        f_3 = f16_to_bits(3.0)
        # abs(s0) * s1 -> 2.0 * 3.0 = 6.0
        self.assertEqual(bits_to_f16(ref_mul_f16(f_neg2, f_3, abs0=True)), 6.0)
        # neg(s0) * s1 -> -(-2.0) * 3.0 = 6.0
        self.assertEqual(bits_to_f16(ref_mul_f16(f_neg2, f_3, neg0=True)), 6.0)

        # clamp modifier
        self.assertEqual(bits_to_f16(ref_mul_f16(f_neg2, f_3, clamp=True)), 0.0)
        self.assertEqual(bits_to_f16(ref_mul_f16(POS_TWO_F16, POS_TWO_F16, clamp=True)), 1.0)
        self.assertEqual(ref_mul_f16(CANONICAL_QNAN_F16, POS_TWO_F16, clamp=True), POS_ZERO_F16)

        # omod modifiers
        self.assertEqual(bits_to_f16(ref_mul_f16(POS_ONE_F16, POS_ONE_F16, omod="mul:2")), 2.0)
        self.assertEqual(bits_to_f16(ref_mul_f16(POS_ONE_F16, POS_ONE_F16, omod="mul:4")), 4.0)
        self.assertEqual(bits_to_f16(ref_mul_f16(POS_ONE_F16, POS_ONE_F16, omod="div:2")), 0.5)

    def test_register_file_halfword_isolation(self) -> None:
        rf = RegisterFile(num_vgprs=4)
        rf.set_vgpr(0, 0x12345678)
        self.assertEqual(rf.get_vgpr_half(0, hi=False), 0x5678)
        self.assertEqual(rf.get_vgpr_half(0, hi=True), 0x1234)

        rf.set_vgpr_half(0, hi=False, val=0xBEEF)
        self.assertEqual(rf.get_vgpr(0), 0x1234BEEF)

        rf.set_vgpr_half(0, hi=True, val=0xCAFE)
        self.assertEqual(rf.get_vgpr(0), 0xCAFEBEEF)


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for AST opcode definitions, type checking, and expression serialization."""

    def test_ast_to_asm_roundtrip(self) -> None:
        inst = ASTInstruction(
            opcode="v_mul_f16",
            dst=ASTOperand(OperandKind.VGPR, reg_index=0, hi=False),
            src0=ASTOperand(OperandKind.VGPR, reg_index=1, hi=True, neg_mod=True),
            src1=ASTOperand(OperandKind.SGPR, reg_index=2, hi=False, abs_mod=True),
            clamp=True,
            omod="mul:2",
        )
        asm_str = inst.to_asm()
        self.assertEqual(asm_str, "v_mul_f16 v0.l, -v1.h, |s2.l| clamp mul:2")

        parsed = parse_v_mul_f16_asm(asm_str)
        self.assertEqual(parsed.dst.reg_index, 0)
        self.assertFalse(parsed.dst.hi)
        self.assertEqual(parsed.src0.reg_index, 1)
        self.assertTrue(parsed.src0.hi)
        self.assertTrue(parsed.src0.neg_mod)
        self.assertEqual(parsed.src1.reg_index, 2)
        self.assertFalse(parsed.src1.hi)
        self.assertTrue(parsed.src1.abs_mod)
        self.assertTrue(parsed.clamp)
        self.assertEqual(parsed.omod, "mul:2")

    def test_parser_modifiers_and_literals(self) -> None:
        inst = parse_v_mul_f16_asm("v_mul_f16 v2.h, 0x3C00, 2.0 div:2")
        self.assertEqual(inst.dst.reg_index, 2)
        self.assertTrue(inst.dst.hi)
        self.assertEqual(inst.src0.literal_val, 0x3C00)
        self.assertEqual(inst.src1.literal_val, POS_TWO_F16)
        self.assertEqual(inst.omod, "div:2")

    def test_ast_validation_errors(self) -> None:
        # Invalid opcode
        with self.assertRaises(ValueError):
            bad = ASTInstruction(
                opcode="v_add_f16",
                dst=ASTOperand(OperandKind.VGPR, 0),
                src0=ASTOperand(OperandKind.VGPR, 1),
                src1=ASTOperand(OperandKind.VGPR, 2),
            )
            bad.validate()

        # Non-VGPR destination
        with self.assertRaises(TypeError):
            bad_dst = ASTInstruction(
                opcode="v_mul_f16",
                dst=ASTOperand(OperandKind.SGPR, 0),
                src0=ASTOperand(OperandKind.VGPR, 1),
                src1=ASTOperand(OperandKind.VGPR, 2),
            )
            bad_dst.validate()

        # Destination with input modifiers
        with self.assertRaises(ValueError):
            bad_mod = ASTInstruction(
                opcode="v_mul_f16",
                dst=ASTOperand(OperandKind.VGPR, 0, neg_mod=True),
                src0=ASTOperand(OperandKind.VGPR, 1),
                src1=ASTOperand(OperandKind.VGPR, 2),
            )
            bad_mod.validate()

        # Invalid omod
        with self.assertRaises(ValueError):
            bad_omod = ASTInstruction(
                opcode="v_mul_f16",
                dst=ASTOperand(OperandKind.VGPR, 0),
                src0=ASTOperand(OperandKind.VGPR, 1),
                src1=ASTOperand(OperandKind.VGPR, 2),
                omod="bad_omod",
            )
            bad_omod.validate()


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def test_scalar_vgpr_execution(self) -> None:
        interp = ScalarInterpreter()
        interp.reg_file.set_vgpr_half(1, hi=False, val=f16_to_bits(2.5))
        interp.reg_file.set_vgpr_half(2, hi=False, val=f16_to_bits(4.0))

        res = interp.execute_asm("v_mul_f16 v0.l, v1.l, v2.l")
        self.assertEqual(bits_to_f16(res), 10.0)
        self.assertEqual(interp.reg_file.get_vgpr_half(0, hi=False), f16_to_bits(10.0))

    def test_scalar_sgpr_and_inline_execution(self) -> None:
        interp = ScalarInterpreter()
        interp.reg_file.set_sgpr_half(3, hi=True, val=f16_to_bits(-3.0))

        res = interp.execute_asm("v_mul_f16 v4.h, s3.h, 2.0")
        self.assertEqual(bits_to_f16(res), -6.0)
        self.assertEqual(interp.reg_file.get_vgpr_half(4, hi=True), f16_to_bits(-6.0))

    def test_scalar_destination_halfword_preservation(self) -> None:
        interp = ScalarInterpreter()
        # Initialize v5 with distinctive pattern 0xAAAA5555
        interp.reg_file.set_vgpr(5, 0xAAAA5555)
        interp.reg_file.set_vgpr_half(1, hi=False, val=POS_ONE_F16)
        interp.reg_file.set_vgpr_half(2, hi=False, val=POS_TWO_F16)

        # Write to v5.l, v5.h must remain 0xAAAA
        interp.execute_asm("v_mul_f16 v5.l, v1.l, v2.l")
        self.assertEqual(interp.reg_file.get_vgpr_half(5, hi=True), 0xAAAA)
        self.assertEqual(interp.reg_file.get_vgpr_half(5, hi=False), POS_TWO_F16)

        # Write to v5.h, v5.l must remain POS_TWO_F16
        interp.execute_asm("v_mul_f16 v5.h, v1.l, 2.0")
        self.assertEqual(interp.reg_file.get_vgpr_half(5, hi=False), POS_TWO_F16)
        self.assertEqual(interp.reg_file.get_vgpr_half(5, hi=True), POS_TWO_F16)


if __name__ == "__main__":
    unittest.main()
