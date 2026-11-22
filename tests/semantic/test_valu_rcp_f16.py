from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_rcp_f16.

Instruction:
  v_rcp_f16 dst, src0 (VOP1 / VOP3)
  D.f16 = 1.0 / S0.f16

Architecture:
  - RDNA2 Vector ALU (GFX10) half-precision floating-point reciprocal.
  - S0: 16-bit IEEE 754 binary16 float (halfword from VGPR, SGPR, or literal/inline const).
  - D:  16-bit IEEE 754 binary16 float (halfword in VGPR).
  - VOP1 preserves the untouched 16-bit half of the destination 32-bit VGPR.
  - VOP3 op_sel selects which half (lo [15:0] or hi [31:16]) is read or written.
  - Modifiers:
      * Input modifiers: abs (|S0|), neg (-S0).
      * Output modifiers: clamp (clamps result to [0.0, 1.0]).
"""

from fractions import Fraction
import math
import os
import re
import struct
import sys
import unittest
from dataclasses import dataclass
from enum import Enum, auto

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

# IEEE 754 binary16 architectural bit layout
SIGN_MASK_F16: int = 0x8000
EXP_MASK_F16: int = 0x7C00
MANT_MASK_F16: int = 0x03FF
QNAN_BIT_F16: int = 0x0200
BIAS_F16: int = 15

# Special half-precision bit patterns
POS_ZERO_F16: int = 0x0000
NEG_ZERO_F16: int = 0x8000
POS_INF_F16: int = 0x7C00
NEG_INF_F16: int = 0xFC00
QNAN_F16: int = 0x7E00
SNAN_F16: int = 0x7C01

# Standard reference floating-point constants in binary16
ONE_F16: int = 0x3C00            # 1.0
NEG_ONE_F16: int = 0xBC00        # -1.0
TWO_F16: int = 0x4000            # 2.0
NEG_TWO_F16: int = 0xC000        # -2.0
HALF_F16: int = 0x3800           # 0.5
NEG_HALF_F16: int = 0xB800       # -0.5
QUARTER_F16: int = 0x3400        # 0.25
NEG_QUARTER_F16: int = 0xB400    # -0.25
FOUR_F16: int = 0x4400           # 4.0
NEG_FOUR_F16: int = 0xC400       # -4.0
EIGHT_F16: int = 0x4800          # 8.0
THREE_F16: int = 0x4200          # 3.0
ONE_THIRD_F16: int = 0x3555      # 1/3 ~ 0.333251953125
ONE_AND_HALF_F16: int = 0x3E00   # 1.5
TWO_THIRDS_F16: int = 0x3955     # 2/3 ~ 0.66650390625

MAX_FINITE_F16: int = 0x7BFF     # 65504.0
MIN_NORMAL_F16: int = 0x0400     # 2^-14 (~6.103515625e-5)
MAX_SUBNORMAL_F16: int = 0x03FF  # ~6.09755516e-5
MIN_SUBNORMAL_F16: int = 0x0001  # 2^-24 (~5.9604644775e-8)


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s16(val: int) -> int:
    """Interpret a 16-bit unsigned halfword as signed two's complement integer."""
    val &= MASK16
    return val - (1 << 16) if (val & 0x8000) else val


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as signed two's complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def split_halves(val32: int) -> tuple[int, int]:
    """Split 32-bit dword into (low_half, high_half)."""
    val = val32 & MASK32
    return val & MASK16, (val >> 16) & MASK16


def join_halves(lo: int, hi: int) -> int:
    """Join (low_half, high_half) into a 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def bits_to_f16(bits: int) -> float:
    """Convert unsigned 16-bit bits into Python float via struct IEEE 754 half."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def f16_to_bits(val: float) -> int:
    """Convert Python float into unsigned 16-bit IEEE 754 half bits."""
    if math.isnan(val):
        return QNAN_F16
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return NEG_INF_F16 if val < 0.0 else POS_INF_F16


def is_nan_f16(bits: int) -> bool:
    """Check if binary16 bits represent any NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) != 0)


def is_qnan_f16(bits: int) -> bool:
    """Check if binary16 bits represent a Quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_snan_f16(bits: int) -> bool:
    """Check if binary16 bits represent a Signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """Check if binary16 bits represent +/- infinity."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) == 0)


def is_zero_f16(bits: int) -> bool:
    """Check if binary16 bits represent +/- 0.0."""
    return (bits & ~SIGN_MASK_F16) == 0


def is_subnormal_f16(bits: int) -> bool:
    """Check if binary16 bits represent a subnormal (denormalized) half float."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & MANT_MASK_F16) != 0)


def is_normal_f16(bits: int) -> bool:
    """Check if binary16 bits represent a normalized finite half float."""
    exp = (bits & EXP_MASK_F16) >> 10
    return 1 <= exp <= 30


def apply_f16_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP input modifiers: abs clears sign bit, neg toggles sign bit."""
    res = bits & MASK16
    if abs_mod:
        res &= ~SIGN_MASK_F16
    if neg_mod:
        res ^= SIGN_MASK_F16
    return res


def _rne_int(q: Fraction) -> int:
    """Round rational number to nearest integer using Round-to-Nearest-Even (RNE)."""
    n = q.numerator // q.denominator
    rem = q - n
    half = Fraction(1, 2)
    if rem > half or (rem == half and (n & 1)):
        n += 1
    return n


def f16_bits_to_fraction(bits: int) -> Fraction:
    """Convert binary16 bits to exact rational Fraction."""
    bits &= MASK16
    sign = -1 if (bits & SIGN_MASK_F16) else 1
    exp_field = (bits & EXP_MASK_F16) >> 10
    mant_field = bits & MANT_MASK_F16
    if exp_field == 0x1F:
        raise ValueError(f"Non-finite f16 bits {bits:#06x}")
    if exp_field == 0:
        return sign * Fraction(mant_field, 1 << 10) * (Fraction(2) ** -14)
    return sign * Fraction((1 << 10) | mant_field, 1 << 10) * (Fraction(2) ** (exp_field - 15))


def round_fraction_to_f16(val: Fraction, neg_zero: bool = False) -> int:
    """Round exact rational Fraction to IEEE 754 binary16 bits using RNE."""
    if val == 0:
        return NEG_ZERO_F16 if neg_zero else POS_ZERO_F16
    sign = SIGN_MASK_F16 if val < 0 else 0
    a = abs(val)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if e < -14:
        n = _rne_int(a / (Fraction(2) ** -24))
        if n == 0:
            return sign
        if n < (1 << 10):
            return sign | n
        return sign | (1 << 10)
    n = _rne_int(a / (Fraction(2) ** (e - 10)))
    if n == (1 << 11):
        n >>= 1
        e += 1
    if e > 15:
        return sign | POS_INF_F16
    return sign | ((e + 15) << 10) | (n & MANT_MASK_F16)


def clamp_f16(bits: int) -> int:
    """Apply RDNA2 clamp modifier [0.0, 1.0] to halfword result."""
    if is_nan_f16(bits):
        return POS_ZERO_F16
    if bits & SIGN_MASK_F16:
        return POS_ZERO_F16
    if is_inf_f16(bits):
        return ONE_F16
    if bits > ONE_F16:
        return ONE_F16
    return bits


def ref_rcp_f16(
    src0_bits: int,
    *,
    abs_src0: bool = False,
    neg_src0: bool = False,
    clamp: bool = False,
) -> int:
    """Bit-accurate reference mathematical oracle for RDNA2 v_rcp_f16.

    Computes: D = 1.0 / S0 in IEEE 754 binary16.
    """
    s0 = apply_f16_modifiers(src0_bits, abs_src0, neg_src0)

    # 1. NaN handling
    if is_nan_f16(s0):
        if clamp:
            return POS_ZERO_F16
        # Propagate quiet NaN with quiet bit set
        return s0 | QNAN_BIT_F16

    # 2. Zero handling: 1.0 / (+-0.0) = +-inf
    if is_zero_f16(s0):
        res = NEG_INF_F16 if (s0 & SIGN_MASK_F16) else POS_INF_F16
        return clamp_f16(res) if clamp else res

    # 3. Infinity handling: 1.0 / (+-inf) = +-0.0
    if is_inf_f16(s0):
        res = NEG_ZERO_F16 if (s0 & SIGN_MASK_F16) else POS_ZERO_F16
        return clamp_f16(res) if clamp else res

    # 4. Finite non-zero: calculate using exact rational reciprocal
    frac = f16_bits_to_fraction(s0)
    recip_frac = Fraction(1) / frac
    res = round_fraction_to_f16(recip_frac, neg_zero=bool(s0 & SIGN_MASK_F16))

    if clamp:
        res = clamp_f16(res)
    return res


def ref_rcp_f16_rational(
    src0_bits: int,
    *,
    abs_src0: bool = False,
    neg_src0: bool = False,
    clamp: bool = False,
) -> int:
    """Alternative rational mathematical oracle implementation."""
    return ref_rcp_f16(src0_bits, abs_src0=abs_src0, neg_src0=neg_src0, clamp=clamp)


class TestValuRcpF16ReferenceOracle(unittest.TestCase):
    def test_powers_of_two_reciprocals(self):
        cases = [
            (ONE_F16, ONE_F16),
            (TWO_F16, HALF_F16),
            (HALF_F16, TWO_F16),
            (FOUR_F16, QUARTER_F16),
            (QUARTER_F16, FOUR_F16),
            (NEG_ONE_F16, NEG_ONE_F16),
            (NEG_TWO_F16, NEG_HALF_F16),
            (NEG_HALF_F16, NEG_TWO_F16),
            (NEG_FOUR_F16, NEG_QUARTER_F16),
        ]
        for inp, expected in cases:
            got = ref_rcp_f16(inp)
            self.assertEqual(got, expected, f"Failed on input {inp:#06x}: expected {expected:#06x}, got {got:#06x}")

    def test_rational_fractions(self):
        got_third = ref_rcp_f16(THREE_F16)
        self.assertEqual(got_third, ONE_THIRD_F16)
        got_two_thirds = ref_rcp_f16(ONE_AND_HALF_F16)
        self.assertEqual(got_two_thirds, TWO_THIRDS_F16)

    def test_infinities_and_zeros(self):
        self.assertEqual(ref_rcp_f16(POS_ZERO_F16), POS_INF_F16)
        self.assertEqual(ref_rcp_f16(NEG_ZERO_F16), NEG_INF_F16)
        self.assertEqual(ref_rcp_f16(POS_INF_F16), POS_ZERO_F16)
        self.assertEqual(ref_rcp_f16(NEG_INF_F16), NEG_ZERO_F16)

    def test_clamping(self):
        self.assertEqual(ref_rcp_f16(HALF_F16, clamp=True), ONE_F16)
        self.assertEqual(ref_rcp_f16(TWO_F16, clamp=True), HALF_F16)
        self.assertEqual(ref_rcp_f16(NEG_TWO_F16, clamp=True), POS_ZERO_F16)
        self.assertEqual(ref_rcp_f16(POS_ZERO_F16, clamp=True), ONE_F16)
        self.assertEqual(ref_rcp_f16(NEG_ZERO_F16, clamp=True), POS_ZERO_F16)
        self.assertEqual(ref_rcp_f16(QNAN_F16, clamp=True), POS_ZERO_F16)

    def test_input_modifiers(self):
        # abs on negative input yields positive reciprocal
        self.assertEqual(ref_rcp_f16(NEG_TWO_F16, abs_src0=True), HALF_F16)
        # neg on positive input yields negative reciprocal
        self.assertEqual(ref_rcp_f16(TWO_F16, neg_src0=True), NEG_HALF_F16)
        # -|x| on negative yields negative
        self.assertEqual(ref_rcp_f16(NEG_TWO_F16, abs_src0=True, neg_src0=True), NEG_HALF_F16)

    def test_subnormal_inputs(self):
        # 1 / 2^-15 = 2^15 = 32768.0 = 0x7800
        sub_2m15 = 0x0200
        self.assertEqual(ref_rcp_f16(sub_2m15), 0x7800)
        # 1 / 2^-14 = 2^14 = 16384.0 = 0x7400
        self.assertEqual(ref_rcp_f16(MIN_NORMAL_F16), 0x7400)
        # Smallest subnormal 2^-24 overflows to infinity
        self.assertEqual(ref_rcp_f16(MIN_SUBNORMAL_F16), POS_INF_F16)



class DataType(Enum):
    """Data types supported by RDNA2 VALU half-precision instructions."""
    F16 = auto()
    U16 = auto()
    I16 = auto()
    F32 = auto()
    U32 = auto()
    I32 = auto()


class OperandKind(Enum):
    """Classification of instruction operands."""
    VGPR_HALF = auto()
    VGPR_FULL = auto()
    SGPR_HALF = auto()
    SGPR_FULL = auto()
    LITERAL_F16 = auto()
    INLINE_CONST = auto()


@dataclass
class Operand:
    """Typed operand representation in symbolic AST."""
    kind: OperandKind
    data_type: DataType
    reg_index: int = 0
    is_hi: bool = False
    literal_val: int = 0
    abs_mod: bool = False
    neg_mod: bool = False

    def validate(self) -> None:
        if self.kind in (OperandKind.VGPR_HALF, OperandKind.VGPR_FULL):
            if not (0 <= self.reg_index < 256):
                raise ValueError(f"VGPR index {self.reg_index} out of valid range [0, 256)")
        elif self.kind in (OperandKind.SGPR_HALF, OperandKind.SGPR_FULL):
            if not (0 <= self.reg_index < 106):
                raise ValueError(f"SGPR index {self.reg_index} out of valid range [0, 106)")
        if (self.abs_mod or self.neg_mod) and self.data_type not in (DataType.F16, DataType.F32):
            raise TypeError("Floating-point modifiers abs/neg can only be applied to float types")

    def to_asm(self) -> str:
        s = ""
        suffix = ".h" if self.is_hi else ".l"
        if self.kind == OperandKind.VGPR_HALF:
            s = f"v{self.reg_index}{suffix}"
        elif self.kind == OperandKind.VGPR_FULL:
            s = f"v{self.reg_index}"
        elif self.kind == OperandKind.SGPR_HALF:
            s = f"s{self.reg_index}{suffix}"
        elif self.kind == OperandKind.SGPR_FULL:
            s = f"s{self.reg_index}"
        elif self.kind == OperandKind.LITERAL_F16:
            s = f"{self.literal_val:#06x}"
        elif self.kind == OperandKind.INLINE_CONST:
            s = f"{self.literal_val}"

        if self.abs_mod:
            s = f"|{s}|"
        if self.neg_mod:
            s = f"-{s}"
        return s


@dataclass
class InstructionAST:
    """Symbolic AST node for v_rcp_f16."""
    opcode: str
    dst: Operand
    src0: Operand
    clamp: bool = False
    op_sel: tuple[int, int] = (0, 0)

    def validate(self) -> None:
        if self.opcode != "v_rcp_f16":
            raise ValueError(f"Unsupported opcode '{self.opcode}', expected 'v_rcp_f16'")
        if self.dst.kind not in (OperandKind.VGPR_HALF, OperandKind.VGPR_FULL):
            raise TypeError("Destination must be a VGPR operand")
        if self.dst.data_type != DataType.F16:
            raise TypeError("Destination data type must be DataType.F16")
        if self.dst.abs_mod or self.dst.neg_mod:
            raise ValueError("Destination operand does not support input modifiers (abs/neg)")
        if self.src0.data_type != DataType.F16:
            raise TypeError("Source operand must be DataType.F16")
        if self.op_sel[0] not in (0, 1) or self.op_sel[1] not in (0, 1):
            raise ValueError(f"op_sel values must be 0 or 1, got {self.op_sel}")
        self.dst.validate()
        self.src0.validate()

    def to_asm(self) -> str:
        tokens = [f"{self.opcode} {self.dst.to_asm()}, {self.src0.to_asm()}"]
        if self.clamp:
            tokens.append("clamp")
        if self.op_sel != (0, 0):
            tokens.append(f"op_sel:[{self.op_sel[0]},{self.op_sel[1]}]")
        return " ".join(tokens)


def parse_vop_rcp_f16(asm_line: str) -> InstructionAST:
    """Parse assembly string into an InstructionAST."""
    line = asm_line.strip()
    if ";" in line:
        line = line.split(";", 1)[0].strip()
    if "//" in line:
        line = line.split("//", 1)[0].strip()
    if not line:
        raise ValueError("Empty assembly line")

    parts = line.split(None, 1)
    opcode = parts[0]
    if opcode != "v_rcp_f16":
        raise ValueError(f"Expected opcode 'v_rcp_f16', got '{opcode}'")
    if len(parts) < 2:
        raise ValueError(f"Incomplete instruction: {line}")

    operands_text = parts[1]
    clamp = False
    op_sel = [0, 0]

    m_opsel = re.search(r"op_sel:\[\s*([01])\s*,\s*([01])\s*\]", operands_text)
    if m_opsel:
        op_sel = [int(m_opsel.group(1)), int(m_opsel.group(2))]
        operands_text = operands_text[:m_opsel.start()] + operands_text[m_opsel.end():]

    tokens = [t.strip() for t in operands_text.split(",") if t.strip()]
    if len(tokens) < 2:
        raise ValueError(f"v_rcp_f16 requires destination and source operands: {line}")

    dst_str = tokens[0]
    src_rest = tokens[1].split()
    src0_str = src_rest[0]

    for modifier in src_rest[1:]:
        if modifier.lower() == "clamp":
            clamp = True
        elif modifier.lower().startswith("op_sel"):
            pass
        else:
            raise ValueError(f"Unrecognized modifier '{modifier}' in '{line}'")

    m_dst = re.match(r"^v(\d+)(\.([lh]))?$", dst_str)
    if not m_dst:
        raise ValueError(f"Invalid VGPR destination operand '{dst_str}'")
    d_reg = int(m_dst.group(1))
    d_hi = False
    if m_dst.group(3):
        d_hi = (m_dst.group(3) == "h")
        op_sel[0] = 1 if d_hi else 0
    elif op_sel[0]:
        d_hi = True
    dst_op = Operand(
        kind=OperandKind.VGPR_HALF if m_dst.group(3) else OperandKind.VGPR_FULL,
        data_type=DataType.F16,
        reg_index=d_reg,
        is_hi=d_hi,
    )

    s0_str = src0_str.strip()
    neg_s0 = False
    abs_s0 = False
    if s0_str.startswith("-"):
        neg_s0 = True
        s0_str = s0_str[1:].strip()
    if s0_str.startswith("|") and s0_str.endswith("|"):
        abs_s0 = True
        s0_str = s0_str[1:-1].strip()

    m_vsrc = re.match(r"^v(\d+)(\.([lh]))?$", s0_str)
    m_ssrc = re.match(r"^s(\d+)(\.([lh]))?$", s0_str)

    if m_vsrc:
        s_reg = int(m_vsrc.group(1))
        s_hi = (m_vsrc.group(3) == "h") if m_vsrc.group(3) else bool(op_sel[1])
        if m_vsrc.group(3):
            op_sel[1] = 1 if s_hi else 0
        src0_op = Operand(
            kind=OperandKind.VGPR_HALF if m_vsrc.group(3) else OperandKind.VGPR_FULL,
            data_type=DataType.F16,
            reg_index=s_reg,
            is_hi=s_hi,
            abs_mod=abs_s0,
            neg_mod=neg_s0,
        )
    elif m_ssrc:
        s_reg = int(m_ssrc.group(1))
        s_hi = (m_ssrc.group(3) == "h") if m_ssrc.group(3) else bool(op_sel[1])
        if m_ssrc.group(3):
            op_sel[1] = 1 if s_hi else 0
        src0_op = Operand(
            kind=OperandKind.SGPR_HALF if m_ssrc.group(3) else OperandKind.SGPR_FULL,
            data_type=DataType.F16,
            reg_index=s_reg,
            is_hi=s_hi,
            abs_mod=abs_s0,
            neg_mod=neg_s0,
        )
    else:
        try:
            if s0_str.startswith("0x") or s0_str.startswith("0X"):
                lit_bits = int(s0_str, 16) & MASK16
            else:
                lit_bits = f16_to_bits(float(s0_str))
            src0_op = Operand(
                kind=OperandKind.LITERAL_F16,
                data_type=DataType.F16,
                literal_val=lit_bits,
                abs_mod=abs_s0,
                neg_mod=neg_s0,
            )
        except ValueError:
            raise ValueError(f"Unrecognized src0 operand format: '{s0_str}'")

    ast = InstructionAST(
        opcode=opcode,
        dst=dst_op,
        src0=src0_op,
        clamp=clamp,
        op_sel=(op_sel[0], op_sel[1]),
    )
    ast.validate()
    return ast


class TestValuRcpF16AST(unittest.TestCase):
    def test_ast_construction_and_validation(self):
        dst = Operand(kind=OperandKind.VGPR_FULL, data_type=DataType.F16, reg_index=0)
        src = Operand(kind=OperandKind.VGPR_FULL, data_type=DataType.F16, reg_index=1, abs_mod=True)
        ast = InstructionAST(opcode="v_rcp_f16", dst=dst, src0=src, clamp=True)
        ast.validate()
        self.assertEqual(ast.opcode, "v_rcp_f16")
        self.assertTrue(ast.clamp)

    def test_parse_assembly_forms(self):
        ast1 = parse_vop_rcp_f16("v_rcp_f16 v0, v1")
        self.assertEqual(ast1.dst.reg_index, 0)
        self.assertEqual(ast1.src0.reg_index, 1)
        self.assertFalse(ast1.clamp)
        self.assertEqual(ast1.op_sel, (0, 0))

        ast2 = parse_vop_rcp_f16("v_rcp_f16 v2, -|v3| clamp")
        self.assertEqual(ast2.dst.reg_index, 2)
        self.assertEqual(ast2.src0.reg_index, 3)
        self.assertTrue(ast2.src0.abs_mod)
        self.assertTrue(ast2.src0.neg_mod)
        self.assertTrue(ast2.clamp)

        ast3 = parse_vop_rcp_f16("v_rcp_f16 v0, v1 op_sel:[1,0]")
        self.assertEqual(ast3.op_sel, (1, 0))
        self.assertTrue(ast3.dst.is_hi)

        ast4 = parse_vop_rcp_f16("v_rcp_f16 v5.h, s2.l")
        self.assertEqual(ast4.dst.reg_index, 5)
        self.assertTrue(ast4.dst.is_hi)
        self.assertEqual(ast4.src0.reg_index, 2)
        self.assertFalse(ast4.src0.is_hi)
        self.assertEqual(ast4.op_sel, (1, 0))

        ast5 = parse_vop_rcp_f16("v_rcp_f16 v0, 2.0")
        self.assertEqual(ast5.src0.kind, OperandKind.LITERAL_F16)
        self.assertEqual(ast5.src0.literal_val, TWO_F16)

    def test_invalid_ast_rejection(self):
        # Invalid opcode
        bad_dst = Operand(kind=OperandKind.VGPR_FULL, data_type=DataType.F16, reg_index=0)
        bad_src = Operand(kind=OperandKind.VGPR_FULL, data_type=DataType.F16, reg_index=1)
        with self.assertRaises(ValueError):
            InstructionAST(opcode="v_add_f16", dst=bad_dst, src0=bad_src).validate()

        # Destination with modifier
        bad_dst_mod = Operand(kind=OperandKind.VGPR_FULL, data_type=DataType.F16, reg_index=0, neg_mod=True)
        with self.assertRaises(ValueError):
            InstructionAST(opcode="v_rcp_f16", dst=bad_dst_mod, src0=bad_src).validate()

        # SGPR as destination
        sgpr_dst = Operand(kind=OperandKind.SGPR_FULL, data_type=DataType.F16, reg_index=0)
        with self.assertRaises(TypeError):
            InstructionAST(opcode="v_rcp_f16", dst=sgpr_dst, src0=bad_src).validate()

        # VGPR out of bounds
        oob_dst = Operand(kind=OperandKind.VGPR_FULL, data_type=DataType.F16, reg_index=300)
        with self.assertRaises(ValueError):
            InstructionAST(opcode="v_rcp_f16", dst=oob_dst, src0=bad_src).validate()


if __name__ == "__main__":
    unittest.main()

