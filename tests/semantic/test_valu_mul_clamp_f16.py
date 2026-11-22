"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_mul_clamp_f16.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  Operations on 16-bit half-precision floating-point quantities (binary16, IEEE 754)
  with clamp saturation enforce clamping of the multiplication result to [0.0, 1.0]:
    - VOP3: v_mul_f16 v[d], src0, src1 clamp [omod] [op_sel]
    - Pseudocode: D.f16 = clamp(S0.f16 * S1.f16, 0.0, 1.0)

  RDNA2 Clamping Semantics:
    - If S0 * S1 < 0.0 or negative signed zero: clamps to +0.0 (0x0000).
    - If S0 * S1 > 1.0 or +infinity: clamps to +1.0 (0x3C00).
    - If S0 * S1 is NaN (including 0 * inf or NaN operand): clamping forces result to +0.0 (0x0000).
    - Subnormal results in (0.0, 1.0] are preserved with gradual underflow.

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
from typing import List, Optional, Tuple

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

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


def ref_mul_clamp_f16(
    s0_bits: int,
    s1_bits: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    omod: str = "none",
) -> int:
    """Reference mathematical oracle for RDNA2 v_mul_f16 with clamp saturation to [0.0, 1.0]."""
    s0 = apply_f16_modifiers(s0_bits, abs_mod=abs0, neg_mod=neg0)
    s1 = apply_f16_modifiers(s1_bits, abs_mod=abs1, neg_mod=neg1)

    # NaN input clamped produces +0.0
    if is_nan_f16(s0) or is_nan_f16(s1):
        return POS_ZERO_F16

    sign = ((s0 >> 15) ^ (s1 >> 15)) & 1

    # 0 * inf is invalid operation -> NaN -> clamped to +0.0
    if (is_zero_f16(s0) and is_inf_f16(s1)) or (is_inf_f16(s0) and is_zero_f16(s1)):
        return POS_ZERO_F16

    # Infinity * finite
    if is_inf_f16(s0) or is_inf_f16(s1):
        return POS_ZERO_F16 if sign else POS_ONE_F16

    # Zero * finite -> 0.0 -> clamped to +0.0 (even if product is -0.0)
    if is_zero_f16(s0) or is_zero_f16(s1):
        return POS_ZERO_F16

    # Finite non-zero multiplication
    val0 = bits_to_f16(s0)
    val1 = bits_to_f16(s1)
    prod = val0 * val1

    if omod == "mul:2":
        prod *= 2.0
    elif omod == "mul:4":
        prod *= 4.0
    elif omod == "div:2":
        prod *= 0.5

    # Clamp saturation to [0.0, 1.0]
    if math.isnan(prod) or prod <= 0.0:
        return POS_ZERO_F16
    if prod >= 1.0:
        return POS_ONE_F16

    return f16_to_bits(prod)


def ref_mul_clamp_f16_rational(s0_bits: int, s1_bits: int) -> int:
    """Exact rational multiplication cross-check with [0.0, 1.0] clamping."""
    s0 = s0_bits & MASK16
    s1 = s1_bits & MASK16
    if is_nan_f16(s0) or is_nan_f16(s1):
        return POS_ZERO_F16
    if (is_zero_f16(s0) and is_inf_f16(s1)) or (is_inf_f16(s0) and is_zero_f16(s1)):
        return POS_ZERO_F16
    sign = ((s0 >> 15) ^ (s1 >> 15)) & 1
    if sign:
        return POS_ZERO_F16
    if is_inf_f16(s0) or is_inf_f16(s1):
        return POS_ONE_F16
    if is_zero_f16(s0) or is_zero_f16(s1):
        return POS_ZERO_F16

    def decode_fraction(b: int) -> Fraction:
        exp = (b >> 10) & 0x1F
        mant = b & 0x03FF
        if exp == 0:
            return Fraction(mant, 1024 * (1 << 14))
        sig = Fraction(1024 + mant, 1024)
        p = exp - 15
        if p >= 0:
            return sig * (1 << p)
        return sig / (1 << (-p))

    frac_prod = decode_fraction(s0) * decode_fraction(s1)
    if frac_prod <= 0:
        return POS_ZERO_F16
    if frac_prod >= 1:
        return POS_ONE_F16

    return f16_to_bits(float(frac_prod))


def ref_simd_mul_clamp_f16(
    s0_wave: List[int],
    s1_wave: List[int],
    *,
    wave_size: int = 32,
    exec_mask: int = 0xFFFFFFFF,
    prev_dst: Optional[List[int]] = None,
    omod: str = "none",
) -> List[int]:
    """Reference multi-lane SIMD vector clamped multiplication under EXEC mask."""
    result: List[int] = []
    for lane in range(wave_size):
        mask_bit = (exec_mask >> lane) & 1
        if mask_bit:
            s0 = s0_wave[lane]
            s1 = s1_wave[lane]
            val = ref_mul_clamp_f16(s0, s1, omod=omod)
            result.append(val)
        else:
            old_val = prev_dst[lane] if prev_dst is not None else 0
            result.append(old_val & MASK16)
    return result


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


class TestValuMulClampF16Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle with clamp saturation."""

    def test_oracle_normal_in_range(self) -> None:
        # 0.5 * 0.5 = 0.25 in [0.0, 1.0]
        s0 = POS_HALF_F16
        s1 = POS_HALF_F16
        res = ref_mul_clamp_f16(s0, s1)
        self.assertEqual(bits_to_f16(res), 0.25)

    def test_oracle_clamp_upper_saturation(self) -> None:
        # 2.0 * 3.0 = 6.0 -> clamped to 1.0
        f_2 = f16_to_bits(2.0)
        f_3 = f16_to_bits(3.0)
        res = ref_mul_clamp_f16(f_2, f_3)
        self.assertEqual(res, POS_ONE_F16)

    def test_oracle_clamp_lower_saturation(self) -> None:
        # -2.0 * 3.0 = -6.0 -> clamped to +0.0
        f_neg2 = f16_to_bits(-2.0)
        f_3 = f16_to_bits(3.0)
        res = ref_mul_clamp_f16(f_neg2, f_3)
        self.assertEqual(res, POS_ZERO_F16)

    def test_oracle_nan_clamping_to_zero(self) -> None:
        # Clamping NaN in RDNA2 produces +0.0
        self.assertEqual(ref_mul_clamp_f16(CANONICAL_QNAN_F16, POS_HALF_F16), POS_ZERO_F16)
        self.assertEqual(ref_mul_clamp_f16(SNAN_F16, POS_ONE_F16), POS_ZERO_F16)
        self.assertEqual(ref_mul_clamp_f16(POS_ZERO_F16, POS_INF_F16), POS_ZERO_F16)

    def test_oracle_infinity_clamping(self) -> None:
        # +inf * +2.0 = +inf -> clamped to 1.0
        self.assertEqual(ref_mul_clamp_f16(POS_INF_F16, POS_TWO_F16), POS_ONE_F16)
        # -inf * +2.0 = -inf -> clamped to +0.0
        self.assertEqual(ref_mul_clamp_f16(NEG_INF_F16, POS_TWO_F16), POS_ZERO_F16)

    def test_oracle_cross_check_rational(self) -> None:
        test_pairs = [
            (f16_to_bits(0.5), f16_to_bits(0.75)),
            (f16_to_bits(0.125), f16_to_bits(4.0)),
            (f16_to_bits(2.5), f16_to_bits(3.0)),
            (f16_to_bits(0.0625), f16_to_bits(0.5)),
        ]
        for s0, s1 in test_pairs:
            self.assertEqual(ref_mul_clamp_f16(s0, s1), ref_mul_clamp_f16_rational(s0, s1))


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
    """Symbolic AST node for v_mul_f16 clamp instruction."""
    opcode: str
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand
    clamp: bool = True
    omod: str = "none"

    def validate(self) -> None:
        if self.opcode not in ("v_mul_f16", "v_mul_clamp_f16"):
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR:
            raise TypeError("Destination must be a VGPR")
        if self.dst.abs_mod or self.dst.neg_mod:
            raise ValueError("Destination cannot have input modifiers (abs/neg)")
        if not self.clamp:
            raise ValueError("v_mul_clamp_f16 must have clamp modifier enabled")
        if self.omod not in ("none", "mul:2", "mul:4", "div:2"):
            raise ValueError(f"Invalid omod specifier: {self.omod}")

    def to_asm(self) -> str:
        parts = [f"{self.opcode} {self.dst.to_asm()}, {self.src0.to_asm()}, {self.src1.to_asm()}"]
        if self.clamp:
            parts.append("clamp")
        if self.omod != "none":
            parts.append(self.omod)
        return " ".join(parts)


def parse_v_mul_clamp_f16_asm(asm_line: str) -> ASTInstruction:
    """Parse a single assembly instruction line into an ASTInstruction node."""
    line = asm_line.strip()
    match = re.match(r"^(v_mul_f16|v_mul_clamp_f16)\s+([^,]+),\s*([^,]+),\s*(\S+)(.*)$", line)
    if not match:
        raise ValueError(f"Cannot parse v_mul_clamp_f16 assembly: {asm_line}")

    op, dst_str, src0_str, src1_str, trailing = match.groups()
    clamp = True if op == "v_mul_clamp_f16" else False
    omod = "none"
    trailing_tokens = trailing.strip().split()
    for tok in trailing_tokens:
        if tok == "clamp":
            clamp = True
        elif tok in ("mul:2", "mul:4", "div:2"):
            omod = tok
        else:
            raise ValueError(f"Invalid instruction modifier or token: {tok}")

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

        if s.startswith("0x") or s.startswith("0X"):
            val = int(s, 16)
            return ASTOperand(kind=OperandKind.LITERAL, literal_val=val, abs_mod=abs_mod, neg_mod=neg_mod)

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

        parsed = parse_v_mul_clamp_f16_asm(asm_str)
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

    def test_ast_validation_errors(self) -> None:
        # Invalid opcode
        with self.assertRaises(ValueError):
            bad = ASTInstruction(
                opcode="v_sub_f16",
                dst=ASTOperand(OperandKind.VGPR, 0),
                src0=ASTOperand(OperandKind.VGPR, 1),
                src1=ASTOperand(OperandKind.VGPR, 2),
                clamp=True,
            )
            bad.validate()

        # Missing clamp
        with self.assertRaises(ValueError):
            bad_clamp = ASTInstruction(
                opcode="v_mul_f16",
                dst=ASTOperand(OperandKind.VGPR, 0),
                src0=ASTOperand(OperandKind.VGPR, 1),
                src1=ASTOperand(OperandKind.VGPR, 2),
                clamp=False,
            )
            bad_clamp.validate()


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 v_mul_clamp_f16."""

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

        res = ref_mul_clamp_f16(
            raw_s0,
            raw_s1,
            abs0=inst.src0.abs_mod,
            neg0=inst.src0.neg_mod,
            abs1=inst.src1.abs_mod,
            neg1=inst.src1.neg_mod,
            omod=inst.omod,
        )

        self.reg_file.set_vgpr_half(inst.dst.reg_index, hi=inst.dst.hi, val=res)
        return res

    def execute_asm(self, asm_line: str) -> int:
        """Parse assembly string and execute on the scalar interpreter."""
        inst = parse_v_mul_clamp_f16_asm(asm_line)
        return self.execute_ast(inst)


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def test_scalar_clamped_multiplication(self) -> None:
        interp = ScalarInterpreter()
        interp.reg_file.set_vgpr_half(1, hi=False, val=f16_to_bits(0.5))
        interp.reg_file.set_vgpr_half(2, hi=False, val=f16_to_bits(0.5))

        res = interp.execute_asm("v_mul_f16 v0.l, v1.l, v2.l clamp")
        self.assertEqual(bits_to_f16(res), 0.25)
        self.assertEqual(interp.reg_file.get_vgpr_half(0, hi=False), f16_to_bits(0.25))

    def test_scalar_clamp_saturation_upper(self) -> None:
        interp = ScalarInterpreter()
        interp.reg_file.set_vgpr_half(1, hi=False, val=f16_to_bits(4.0))
        interp.reg_file.set_vgpr_half(2, hi=False, val=f16_to_bits(2.0))

        res = interp.execute_asm("v_mul_f16 v0.l, v1.l, v2.l clamp")
        self.assertEqual(res, POS_ONE_F16)
        self.assertEqual(interp.reg_file.get_vgpr_half(0, hi=False), POS_ONE_F16)

    def test_scalar_destination_halfword_preservation(self) -> None:
        interp = ScalarInterpreter()
        interp.reg_file.set_vgpr(5, 0xBEEF5555)
        interp.reg_file.set_vgpr_half(1, hi=False, val=f16_to_bits(0.5))
        interp.reg_file.set_vgpr_half(2, hi=False, val=f16_to_bits(0.5))

        # Write to v5.l, v5.h must remain 0xBEEF
        interp.execute_asm("v_mul_f16 v5.l, v1.l, v2.l clamp")
        self.assertEqual(interp.reg_file.get_vgpr_half(5, hi=True), 0xBEEF)
        self.assertEqual(interp.reg_file.get_vgpr_half(5, hi=False), f16_to_bits(0.25))


class SIMDWavefront:
    """Vectorized SIMD wavefront model supporting Wave32/Wave64 divergence and EXEC masking."""

    def __init__(self, wave_size: int = 32):
        if wave_size not in (32, 64):
            raise ValueError("Wave size must be 32 or 64")
        self.wave_size = wave_size
        self.exec_mask = MASK32 if wave_size == 32 else MASK64
        self.lanes = [RegisterFile() for _ in range(wave_size)]

    def is_lane_active(self, lane: int) -> bool:
        """Return True if the specified lane index is active under the current EXEC mask."""
        return bool((self.exec_mask >> lane) & 1)

    def load_vgpr_half_wave(self, reg_idx: int, hi: bool) -> List[int]:
        """Read 16-bit values from all lanes in the wavefront for the specified VGPR halfword."""
        return [self.lanes[i].get_vgpr_half(reg_idx, hi=hi) for i in range(self.wave_size)]

    def store_vgpr_half_wave(self, reg_idx: int, hi: bool, values: List[int]) -> None:
        """Store 16-bit values to all lanes in the wavefront for the specified VGPR halfword."""
        for i in range(min(len(values), self.wave_size)):
            self.lanes[i].set_vgpr_half(reg_idx, hi=hi, val=values[i])

    def step_v_mul_clamp_f16(
        self,
        dst_reg: int,
        dst_hi: bool,
        src0_reg: int,
        src0_hi: bool,
        src1_reg: int,
        src1_hi: bool,
        omod: str = "none",
    ) -> List[int]:
        """Execute a SIMD v_mul_clamp_f16 step across all lanes adhering to EXEC masking."""
        results: List[int] = []
        for lane in range(self.wave_size):
            if self.is_lane_active(lane):
                s0 = self.lanes[lane].get_vgpr_half(src0_reg, hi=src0_hi)
                s1 = self.lanes[lane].get_vgpr_half(src1_reg, hi=src1_hi)
                res = ref_mul_clamp_f16(s0, s1, omod=omod)
                self.lanes[lane].set_vgpr_half(dst_reg, hi=dst_hi, val=res)
                results.append(res)
            else:
                old_val = self.lanes[lane].get_vgpr_half(dst_reg, hi=dst_hi)
                results.append(old_val)
        return results


class TestSIMDExecution(unittest.TestCase):
    """Unit tests for SIMD wavefront multi-lane execution, divergence, and EXEC masking."""

    def test_wave32_uniform_execution(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        s0_vals = [f16_to_bits(0.5)] * 32
        s1_vals = [f16_to_bits(float(i) * 0.1) for i in range(32)]
        sim.store_vgpr_half_wave(1, hi=False, values=s0_vals)
        sim.store_vgpr_half_wave(2, hi=False, values=s1_vals)

        results = sim.step_v_mul_clamp_f16(dst_reg=0, dst_hi=False, src0_reg=1, src0_hi=False, src1_reg=2, src1_hi=False)
        for i in range(32):
            expected = min(max(0.5 * (float(i) * 0.1), 0.0), 1.0)
            self.assertAlmostEqual(bits_to_f16(results[i]), expected, places=3)
            self.assertAlmostEqual(bits_to_f16(sim.lanes[i].get_vgpr_half(0, hi=False)), expected, places=3)

    def test_wave32_divergence_mask(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        prev_dst = [0x5555] * 32
        sim.store_vgpr_half_wave(0, hi=False, values=prev_dst)

        s0_vals = [f16_to_bits(0.5)] * 32
        s1_vals = [f16_to_bits(0.5)] * 32
        sim.store_vgpr_half_wave(1, hi=False, values=s0_vals)
        sim.store_vgpr_half_wave(2, hi=False, values=s1_vals)

        sim.exec_mask = 0xAAAAAAAA
        results = sim.step_v_mul_clamp_f16(dst_reg=0, dst_hi=False, src0_reg=1, src0_hi=False, src1_reg=2, src1_hi=False)

        for i in range(32):
            if i % 2 == 1:
                self.assertEqual(bits_to_f16(results[i]), 0.25)
                self.assertEqual(bits_to_f16(sim.lanes[i].get_vgpr_half(0, hi=False)), 0.25)
            else:
                self.assertEqual(results[i], 0x5555)
                self.assertEqual(sim.lanes[i].get_vgpr_half(0, hi=False), 0x5555)

    def test_wave64_divergence_and_oracle_cross_check(self) -> None:
        sim = SIMDWavefront(wave_size=64)
        exec_mask = 0x00000000FFFFFFFF
        sim.exec_mask = exec_mask

        prev_dst = [u16(0x7000 + i) for i in range(64)]
        sim.store_vgpr_half_wave(0, hi=True, values=prev_dst)

        s0_vals = [f16_to_bits(float(i + 1) * 0.1) for i in range(64)]
        s1_vals = [f16_to_bits(0.5) for _ in range(64)]
        sim.store_vgpr_half_wave(3, hi=False, values=s0_vals)
        sim.store_vgpr_half_wave(4, hi=True, values=s1_vals)

        expected = ref_simd_mul_clamp_f16(
            s0_vals, s1_vals, wave_size=64, exec_mask=exec_mask, prev_dst=prev_dst
        )

        results = sim.step_v_mul_clamp_f16(dst_reg=0, dst_hi=True, src0_reg=3, src0_hi=False, src1_reg=4, src1_hi=True)
        self.assertEqual(results, expected)


class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_inplace_dst_equals_src0(self) -> None:
        interp = ScalarInterpreter()
        interp.reg_file.set_vgpr_half(0, hi=False, val=f16_to_bits(0.5))
        interp.reg_file.set_vgpr_half(1, hi=False, val=f16_to_bits(0.5))

        res = interp.execute_asm("v_mul_f16 v0.l, v0.l, v1.l clamp")
        self.assertEqual(bits_to_f16(res), 0.25)
        self.assertEqual(bits_to_f16(interp.reg_file.get_vgpr_half(0, hi=False)), 0.25)
        self.assertEqual(bits_to_f16(interp.reg_file.get_vgpr_half(1, hi=False)), 0.5)

    def test_inplace_dst_equals_src1(self) -> None:
        interp = ScalarInterpreter()
        interp.reg_file.set_vgpr_half(0, hi=False, val=f16_to_bits(0.5))
        interp.reg_file.set_vgpr_half(1, hi=False, val=f16_to_bits(0.25))

        res = interp.execute_asm("v_mul_f16 v1.l, v0.l, v1.l clamp")
        self.assertEqual(bits_to_f16(res), 0.125)
        self.assertEqual(bits_to_f16(interp.reg_file.get_vgpr_half(1, hi=False)), 0.125)
        self.assertEqual(bits_to_f16(interp.reg_file.get_vgpr_half(0, hi=False)), 0.5)

    def test_inplace_self_squaring(self) -> None:
        interp = ScalarInterpreter()
        interp.reg_file.set_vgpr_half(2, hi=False, val=f16_to_bits(0.5))

        # 0.5 * 0.5 = 0.25
        res = interp.execute_asm("v_mul_f16 v2.l, v2.l, v2.l clamp")
        self.assertEqual(bits_to_f16(res), 0.25)
        self.assertEqual(bits_to_f16(interp.reg_file.get_vgpr_half(2, hi=False)), 0.25)

    def test_subword_intra_register_crossover(self) -> None:
        interp = ScalarInterpreter()
        interp.reg_file.set_vgpr_half(3, hi=False, val=f16_to_bits(0.5))
        interp.reg_file.set_vgpr_half(3, hi=True, val=f16_to_bits(0.5))

        # v3.h = v3.l * v3.h clamp -> 0.25 into v3.h, v3.l remains 0.5
        res = interp.execute_asm("v_mul_f16 v3.h, v3.l, v3.h clamp")
        self.assertEqual(bits_to_f16(res), 0.25)
        self.assertEqual(bits_to_f16(interp.reg_file.get_vgpr_half(3, hi=True)), 0.25)
        self.assertEqual(bits_to_f16(interp.reg_file.get_vgpr_half(3, hi=False)), 0.5)

    def test_successive_inplace_chain(self) -> None:
        interp = ScalarInterpreter()
        interp.reg_file.set_vgpr_half(0, hi=False, val=f16_to_bits(0.5))

        # Squaring twice: 0.5 -> 0.25 -> 0.0625
        interp.execute_asm("v_mul_f16 v0.l, v0.l, v0.l clamp")
        interp.execute_asm("v_mul_f16 v0.l, v0.l, v0.l clamp")
        self.assertEqual(bits_to_f16(interp.reg_file.get_vgpr_half(0, hi=False)), 0.0625)


@dataclass
class StatusFlags:
    """Architectural floating-point status flags and condition codes with clamp detection."""
    invalid: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False
    zero: bool = False
    negative: bool = False
    denormal: bool = False
    clamped_low: bool = False
    clamped_high: bool = False


def compute_f16_mantissa_product_carry(s0_bits: int, s1_bits: int) -> Tuple[int, int]:
    """Compute product of normalized significands and carry-out bit."""
    _, exp0, mant0 = split_f16(s0_bits)
    _, exp1, mant1 = split_f16(s1_bits)

    sig0 = (1024 | mant0) if exp0 > 0 else mant0
    sig1 = (1024 | mant1) if exp1 > 0 else mant1

    raw_prod = sig0 * sig1
    carry = 1 if raw_prod >= (1 << 21) else 0
    return raw_prod, carry


def evaluate_v_mul_clamp_f16_flags(s0_bits: int, s1_bits: int, res_bits: int) -> StatusFlags:
    """Evaluate condition codes and status flags for v_mul_clamp_f16."""
    flags = StatusFlags()
    s0 = s0_bits & MASK16
    s1 = s1_bits & MASK16
    res = res_bits & MASK16

    if is_snan_f16(s0) or is_snan_f16(s1):
        flags.invalid = True
    elif (is_zero_f16(s0) and is_inf_f16(s1)) or (is_inf_f16(s0) and is_zero_f16(s1)):
        flags.invalid = True

    if res == POS_ZERO_F16:
        flags.zero = True
        if not (is_zero_f16(s0) or is_zero_f16(s1)):
            flags.clamped_low = True

    if res == POS_ONE_F16:
        flags.clamped_high = True

    if (res >> 15) & 1:
        flags.negative = True

    if is_subnormal_f16(s0) or is_subnormal_f16(s1) or is_subnormal_f16(res):
        flags.denormal = True

    return flags


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and mantissa product carry."""

    def test_exact_in_range_flags(self) -> None:
        s0 = f16_to_bits(0.5)
        s1 = f16_to_bits(0.5)
        res = ref_mul_clamp_f16(s0, s1)
        flags = evaluate_v_mul_clamp_f16_flags(s0, s1, res)
        self.assertFalse(flags.invalid)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.clamped_low)
        self.assertFalse(flags.clamped_high)

    def test_clamp_low_flags(self) -> None:
        s0 = f16_to_bits(-2.0)
        s1 = f16_to_bits(3.0)
        res = ref_mul_clamp_f16(s0, s1)
        self.assertEqual(res, POS_ZERO_F16)
        flags = evaluate_v_mul_clamp_f16_flags(s0, s1, res)
        self.assertTrue(flags.zero)
        self.assertTrue(flags.clamped_low)
        self.assertFalse(flags.negative)

    def test_clamp_high_flags(self) -> None:
        s0 = f16_to_bits(2.0)
        s1 = f16_to_bits(4.0)
        res = ref_mul_clamp_f16(s0, s1)
        self.assertEqual(res, POS_ONE_F16)
        flags = evaluate_v_mul_clamp_f16_flags(s0, s1, res)
        self.assertTrue(flags.clamped_high)
        self.assertFalse(flags.zero)

    def test_mantissa_product_carry(self) -> None:
        f_1p5 = f16_to_bits(1.5)
        _, carry1 = compute_f16_mantissa_product_carry(f_1p5, f_1p5)
        self.assertEqual(carry1, 1)


if __name__ == "__main__":
    unittest.main()
