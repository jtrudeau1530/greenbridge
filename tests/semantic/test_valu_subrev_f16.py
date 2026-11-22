from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_subrev_f16.

Instruction:
  v_subrev_f16 dst, src0, src1 (VOP2 / VOP3)
  D.f16 = S1.f16 - S0.f16

Architecture:
  - RDNA2 Vector ALU (GFX10) half-precision floating-point reverse subtraction.
  - Computes the difference S1 - S0 conforming to IEEE 754-2008 binary16.
  - In VOP2 format:
      * S0: 16-bit float operand (VGPR, SGPR, or 16-bit inline constant / literal).
      * S1: 16-bit float operand (VGPR).
      * D:  16-bit float destination VGPR (writes low 16 bits, preserves high 16 bits).
  - In VOP3 format:
      * Input modifiers: abs (|S|), neg (-S) on both src0 and src1.
      * Output modifiers: clamp (clamps result to [0.0, 1.0]), omod (mul:2, mul:4, div:2).
      * op_sel: selects low or high 16-bit half for src0, src1, and dst independently.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
import math
import re
import struct
from typing import Optional, Union
import unittest

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

SIGN_MASK_F16: int = 0x8000
EXP_MASK_F16: int = 0x7C00
MANT_MASK_F16: int = 0x03FF
QNAN_BIT_F16: int = 0x0200
BIAS_F16: int = 15

POS_ZERO_F16: int = 0x0000
NEG_ZERO_F16: int = 0x8000
POS_INF_F16: int = 0x7C00
NEG_INF_F16: int = 0xFC00
CANONICAL_QNAN_F16: int = 0x7E00
SNAN_DEFAULT_F16: int = 0x7C01

ONE_F16: int = 0x3C00          # 1.0
TWO_F16: int = 0x4000          # 2.0
THREE_F16: int = 0x4200        # 3.0
FOUR_F16: int = 0x4400         # 4.0
HALF_F16: int = 0x3800         # 0.5
QUARTER_F16: int = 0x3400      # 0.25
NEG_ONE_F16: int = 0xBC00      # -1.0
NEG_TWO_F16: int = 0xC000      # -2.0
NEG_HALF_F16: int = 0xB800     # -0.5

MAX_FINITE_F16: int = 0x7BFF   # 65504.0
NEG_MAX_FINITE_F16: int = 0xFBFF  # -65504.0
MIN_NORMAL_F16: int = 0x0400   # 2^-14 (~6.1035e-5)
MAX_SUBNORMAL_F16: int = 0x03FF # ~6.0975e-5
MIN_SUBNORMAL_F16: int = 0x0001 # 2^-24 (~5.9605e-8)


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s16(val: int) -> int:
    """Interpret a 16-bit unsigned value as signed two's complement integer."""
    val &= MASK16
    return val - (1 << 16) if (val & 0x8000) else val


def bits_to_f16(bits: int) -> float:
    """Convert unsigned 16-bit integer bits to Python float (IEEE 754 binary16)."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def f16_to_bits(val: float) -> int:
    """Convert Python float to unsigned 16-bit integer bits (IEEE 754 binary16 RNE)."""
    if math.isnan(val):
        return CANONICAL_QNAN_F16
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return NEG_INF_F16 if val < 0.0 else POS_INF_F16


def split_f16_bits(bits: int) -> tuple[int, int, int]:
    """Unpack 16-bit float into (sign, exponent, fraction) components."""
    bits &= MASK16
    sign = (bits & SIGN_MASK_F16) >> 15
    exponent = (bits & EXP_MASK_F16) >> 10
    fraction = bits & MANT_MASK_F16
    return sign, exponent, fraction


def is_nan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) != 0)


def is_qnan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_snan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """Return True if bits represent IEEE 754 positive or negative infinity."""
    return (bits & 0x7FFF) == 0x7C00


def is_zero_f16(bits: int) -> bool:
    """Return True if bits represent positive or negative zero (+0.0 or -0.0)."""
    return (bits & ~SIGN_MASK_F16) == 0


def is_subnormal_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision subnormal (denormal)."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & MANT_MASK_F16) != 0)


def is_normal_f16(bits: int) -> bool:
    """Return True if bits represent a normalized finite half-precision float."""
    exp = (bits & EXP_MASK_F16) >> 10
    return 1 <= exp <= 30


def pack_half(lo: int, hi: int) -> int:
    """Pack two 16-bit halfwords into a 32-bit register word."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def unpack_half(word: int) -> tuple[int, int]:
    """Unpack a 32-bit register word into (lo, hi) 16-bit halfwords."""
    return word & MASK16, (word >> 16) & MASK16


def extract_half(word: int, hi_sel: bool) -> int:
    """Extract 16-bit halfword from 32-bit word using high-select boolean flag."""
    return (word >> 16) & MASK16 if hi_sel else word & MASK16


def insert_half(old_word: int, val16: int, dst_hi: bool) -> int:
    """Insert 16-bit halfword into low or high half of 32-bit word, preserving other half."""
    val16 &= MASK16
    old_word &= MASK32
    if dst_hi:
        return (val16 << 16) | (old_word & 0x0000FFFF)
    else:
        return (old_word & 0xFFFF0000) | val16


def apply_f16_input_mods(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & MASK16
    if abs_mod:
        res &= ~SIGN_MASK_F16
    if neg_mod:
        res ^= SIGN_MASK_F16
    return res


def apply_f16_output_mods(bits: int, clamp: bool, omod: str = "none") -> int:
    """Apply VOP3 output modifiers: omod scaling and clamp to [0.0, 1.0]."""
    bits &= MASK16
    if is_nan_f16(bits):
        # Clamping NaN in RDNA2 produces +0.0
        return POS_ZERO_F16 if clamp else bits

    # Apply omod scaling if not NaN and not infinity
    if not is_inf_f16(bits) and omod != "none":
        val = bits_to_f16(bits)
        if omod == "mul:2":
            val *= 2.0
        elif omod == "mul:4":
            val *= 4.0
        elif omod == "div:2":
            val *= 0.5
        bits = f16_to_bits(val)

    if clamp:
        if is_inf_f16(bits):
            return ONE_F16 if bits == POS_INF_F16 else POS_ZERO_F16
        val = bits_to_f16(bits)
        if val < 0.0 or is_zero_f16(bits):
            return POS_ZERO_F16
        elif val > 1.0:
            return ONE_F16
        return bits

    return bits


def ref_subrev_f16(
    src0_bits: int,
    src1_bits: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
    omod: str = "none",
) -> int:
    """Reference mathematical oracle for RDNA2 v_subrev_f16 half-precision reverse subtraction.

    Operation: D = src1 - src0 (S1 - S0)
    Conforms strictly to IEEE 754-2008 binary16 Round-to-Nearest-Even (RNE)
    with RDNA2 special values, NaN quietening, and input/output modifiers.
    """
    s0 = apply_f16_input_mods(src0_bits, abs0, neg0)
    s1 = apply_f16_input_mods(src1_bits, abs1, neg1)

    # 1. NaN handling
    s0_nan = is_nan_f16(s0)
    s1_nan = is_nan_f16(s1)
    if s0_nan or s1_nan:
        if clamp:
            return POS_ZERO_F16
        nan_bits = s0 if s0_nan else s1
        return nan_bits | QNAN_BIT_F16

    # 2. Infinity handling
    s0_inf = is_inf_f16(s0)
    s1_inf = is_inf_f16(s1)
    if s0_inf and s1_inf:
        s0_sign = s0 & SIGN_MASK_F16
        s1_sign = s1 & SIGN_MASK_F16
        # (+inf) - (+inf) or (-inf) - (-inf) -> Invalid operation -> Canonical Quiet NaN
        if s0_sign == s1_sign:
            return POS_ZERO_F16 if clamp else CANONICAL_QNAN_F16
        # (+inf) - (-inf) = +inf; (-inf) - (+inf) = -inf
        res_inf = s1
        return apply_f16_output_mods(res_inf, clamp, omod)
    if s1_inf:
        return apply_f16_output_mods(s1, clamp, omod)
    if s0_inf:
        # finite - (+inf) = -inf; finite - (-inf) = +inf
        res_inf = NEG_INF_F16 if s0 == POS_INF_F16 else POS_INF_F16
        return apply_f16_output_mods(res_inf, clamp, omod)

    # 3. Finite numbers
    val0 = bits_to_f16(s0)
    val1 = bits_to_f16(s1)
    diff = val1 - val0

    # Signed zero preservation:
    # (+0.0) - (+0.0) == +0.0
    # (-0.0) - (-0.0) == +0.0
    # (+0.0) - (-0.0) == +0.0
    # (-0.0) - (+0.0) == -0.0
    if diff == 0.0:
        if val1 == 0.0 and val0 == 0.0:
            if (s1 & SIGN_MASK_F16) and not (s0 & SIGN_MASK_F16):
                res_zero = NEG_ZERO_F16
            else:
                res_zero = POS_ZERO_F16
        else:
            res_zero = POS_ZERO_F16
        return apply_f16_output_mods(res_zero, clamp, omod)

    raw_res = f16_to_bits(diff)
    return apply_f16_output_mods(raw_res, clamp, omod)


def f16_bits_to_fraction(bits: int) -> Fraction:
    """Convert a finite binary16 bit pattern to an exact rational Fraction."""
    sign, exp, mant = split_f16_bits(bits)
    if exp == 0:
        # Subnormal: 0.mant * 2^-14 = mant * 2^-24
        val = Fraction(mant, 1 << 24)
    else:
        # Normalized: 1.mant * 2^(exp - 15) = (1024 + mant) * 2^(exp - 25)
        num = (1 << 10) | mant
        shift = exp - 25
        if shift >= 0:
            val = Fraction(num << shift, 1)
        else:
            val = Fraction(num, 1 << (-shift))
    return -val if sign else val


def ref_subrev_f16_rational(
    src0_bits: int,
    src1_bits: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Independent exact rational arithmetic cross-check oracle for v_subrev_f16.

    Uses fractions.Fraction to compute the exact difference S1 - S0 without intermediate
    floating-point inaccuracy, then applies bit-exact IEEE 754 half-precision RNE rounding.
    """
    s0 = apply_f16_input_mods(src0_bits, abs0, neg0)
    s1 = apply_f16_input_mods(src1_bits, abs1, neg1)

    if is_nan_f16(s0) or is_nan_f16(s1):
        if clamp:
            return POS_ZERO_F16
        nan_bits = s0 if is_nan_f16(s0) else s1
        return nan_bits | QNAN_BIT_F16

    if is_inf_f16(s0) or is_inf_f16(s1):
        return ref_subrev_f16(src0_bits, src1_bits, abs0=abs0, neg0=neg0, abs1=abs1, neg1=neg1, clamp=clamp)

    frac0 = f16_bits_to_fraction(s0)
    frac1 = f16_bits_to_fraction(s1)
    diff_frac = frac1 - frac0

    if diff_frac == 0:
        if (s1 & SIGN_MASK_F16) and not (s0 & SIGN_MASK_F16):
            res_zero = NEG_ZERO_F16
        else:
            res_zero = POS_ZERO_F16
        return apply_f16_output_mods(res_zero, clamp)

    sign = 1 if diff_frac < 0 else 0
    abs_diff = abs(diff_frac)

    # Check overflow beyond max finite
    max_finite_val = Fraction(65504, 1)
    if abs_diff >= Fraction(65520, 1):
        res_inf = NEG_INF_F16 if sign else POS_INF_F16
        return apply_f16_output_mods(res_inf, clamp)

    # Normal vs subnormal bounds
    min_normal_val = Fraction(1, 1 << 14)
    if abs_diff < min_normal_val:
        # Subnormal representation: mant = abs_diff * 2^24 rounded
        scaled = abs_diff * (1 << 24)
        floor_mant = scaled.numerator // scaled.denominator
        rem = scaled - floor_mant
        half_one = Fraction(1, 2)
        if rem > half_one or (rem == half_one and (floor_mant % 2 == 1)):
            floor_mant += 1
        if floor_mant >= (1 << 10):
            res_bits = (sign << 15) | (1 << 10)  # promoted to min normal
        else:
            res_bits = (sign << 15) | floor_mant
        return apply_f16_output_mods(res_bits, clamp)

    # Normalized float: find exponent
    val_float = float(abs_diff)
    exp = int(math.floor(math.log2(val_float))) + 15
    exp = max(1, min(30, exp))

    scaled = abs_diff * Fraction(1 << (25 - exp), 1)
    num = scaled.numerator // scaled.denominator
    rem = scaled - num
    half_one = Fraction(1, 2)
    if rem > half_one or (rem == half_one and (num % 2 == 1)):
        num += 1

    if num >= (1 << 11):
        num >>= 1
        exp += 1

    if exp >= 31:
        res_inf = NEG_INF_F16 if sign else POS_INF_F16
        return apply_f16_output_mods(res_inf, clamp)

    mant = num & MANT_MASK_F16
    res_bits = (sign << 15) | (exp << 10) | mant
    return apply_f16_output_mods(res_bits, clamp)


# ============================================================================
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ============================================================================

class Opcode(Enum):
    V_SUBREV_F16 = "v_subrev_f16"


class Encoding(Enum):
    VOP2 = "vop2"
    VOP3 = "vop3"


class RegKind(Enum):
    VGPR = auto()
    SGPR = auto()


class DataType(Enum):
    F16 = auto()
    F32 = auto()
    I16 = auto()
    I32 = auto()
    U16 = auto()
    U32 = auto()


@dataclass(frozen=True)
class Register:
    kind: RegKind
    index: int
    hi: bool = False  # True: high 16-bit half (bits 16..31); False: low (0..15)

    def validate(self) -> None:
        if self.kind == RegKind.VGPR:
            if not (0 <= self.index < 256):
                raise ValueError(f"VGPR index {self.index} out of valid range [0, 255].")
        elif self.kind == RegKind.SGPR:
            if not (0 <= self.index < 106):
                raise ValueError(f"SGPR index {self.index} out of valid range [0, 105].")

    def __str__(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        suffix = ".h" if self.hi else ""
        return f"{prefix}{self.index}{suffix}"


@dataclass(frozen=True)
class Operand:
    reg: Optional[Register] = None
    imm_val: Optional[Union[float, int]] = None
    abs_mod: bool = False
    neg_mod: bool = False
    dtype: DataType = DataType.F16

    def is_register(self) -> bool:
        return self.reg is not None

    def is_literal(self) -> bool:
        return self.imm_val is not None

    def validate(self) -> None:
        if self.reg is None and self.imm_val is None:
            raise ValueError("Operand must specify either register or immediate value.")
        if self.reg is not None and self.imm_val is not None:
            raise ValueError("Operand cannot specify both register and immediate value.")
        if self.reg is not None:
            self.reg.validate()
        if self.dtype != DataType.F16:
            raise TypeError(f"v_subrev_f16 expects F16 operand data type, got {self.dtype}.")

    def __str__(self) -> str:
        if self.reg is not None:
            base = str(self.reg)
        elif isinstance(self.imm_val, float):
            base = f"{self.imm_val}"
        else:
            base = f"{self.imm_val:#x}"
        if self.abs_mod:
            base = f"|{base}|"
        if self.neg_mod:
            base = f"-{base}"
        return base


@dataclass
class InstructionModifiers:
    clamp: bool = False
    omod: str = "none"  # "none", "mul:2", "mul:4", "div:2"
    op_sel: tuple[bool, bool, bool] = (False, False, False)  # (src0_hi, src1_hi, dst_hi)

    def validate(self, encoding: Encoding) -> None:
        if encoding == Encoding.VOP2:
            if self.clamp:
                raise ValueError("VOP2 encoding does not support clamp modifier.")
            if self.omod != "none":
                raise ValueError(f"VOP2 encoding does not support omod modifier {self.omod}.")
            if any(self.op_sel):
                raise ValueError("VOP2 encoding does not support op_sel modifier.")
        if self.omod not in ("none", "mul:2", "mul:4", "div:2"):
            raise ValueError(f"Invalid omod value: {self.omod}.")


@dataclass
class InstructionAST:
    opcode: Opcode
    dst: Register
    src0: Operand
    src1: Operand
    mods: InstructionModifiers = field(default_factory=InstructionModifiers)
    encoding: Encoding = Encoding.VOP2

    def validate(self) -> None:
        if self.opcode != Opcode.V_SUBREV_F16:
            raise ValueError(f"Unsupported opcode: {self.opcode}.")
        if self.dst.kind != RegKind.VGPR:
            raise TypeError("VALU destination register must be a VGPR.")
        self.dst.validate()
        self.src0.validate()
        self.src1.validate()
        if self.encoding == Encoding.VOP2:
            if self.src0.abs_mod or self.src0.neg_mod or self.src1.abs_mod or self.src1.neg_mod:
                raise ValueError("VOP2 encoding does not support input modifiers (abs/neg).")
        self.mods.validate(self.encoding)

    def to_asm(self) -> str:
        self.validate()
        suffix = "_e64" if self.encoding == Encoding.VOP3 else "_e32"
        op_name = f"{self.opcode.value}{suffix}"
        dst_str = str(self.dst)
        src0_str = str(self.src0)
        src1_str = str(self.src1)
        parts = [f"{op_name} {dst_str}, {src0_str}, {src1_str}"]
        if self.mods.clamp:
            parts.append("clamp")
        if self.mods.omod != "none":
            parts.append(self.mods.omod)
        if any(self.mods.op_sel):
            sel_str = f"op_sel:[{int(self.mods.op_sel[0])},{int(self.mods.op_sel[1])},{int(self.mods.op_sel[2])}]"
            parts.append(sel_str)
        return " ".join(parts)


def parse_subrev_f16_asm(asm_text: str) -> InstructionAST:
    """Parse a single assembly instruction line into an InstructionAST."""
    tokens = asm_text.strip().split()
    if not tokens:
        raise ValueError("Empty assembly instruction.")

    op_token = tokens[0].rstrip(",")
    if "v_subrev_f16" not in op_token:
        raise ValueError(f"Expected v_subrev_f16 opcode, got: {op_token}")

    is_e64 = "_e64" in op_token or any(
        t in ("clamp", "mul:2", "mul:4", "div:2") or t.startswith("op_sel")
        for t in tokens
    )
    encoding = Encoding.VOP3 if is_e64 else Encoding.VOP2

    rest = " ".join(tokens[1:])
    operands_raw = [
        o.strip()
        for o in re.split(r",|\s+", rest)
        if o.strip()
        and not o.startswith("clamp")
        and not o.startswith("mul:")
        and not o.startswith("div:")
        and not o.startswith("op_sel")
    ]

    if len(operands_raw) < 3:
        raise ValueError(f"v_subrev_f16 requires at least 3 operands (dst, src0, src1), got: {operands_raw}")

    dst_tok = operands_raw[0]
    dst_hi = dst_tok.endswith(".h")
    dst_idx = int(re.search(r"v(\d+)", dst_tok).group(1))
    dst = Register(RegKind.VGPR, dst_idx, hi=dst_hi)

    def parse_operand(tok: str) -> Operand:
        abs_m = (tok.startswith("|") and tok.endswith("|")) or ("-|" in tok)
        neg_m = tok.startswith("-")
        clean = tok.replace("-", "").replace("|", "")

        if clean.startswith("v"):
            hi = clean.endswith(".h")
            idx = int(re.search(r"v(\d+)", clean).group(1))
            return Operand(reg=Register(RegKind.VGPR, idx, hi=hi), abs_mod=abs_m, neg_mod=neg_m, dtype=DataType.F16)
        elif clean.startswith("s"):
            hi = clean.endswith(".h")
            idx = int(re.search(r"s(\d+)", clean).group(1))
            return Operand(reg=Register(RegKind.SGPR, idx, hi=hi), abs_mod=abs_m, neg_mod=neg_m, dtype=DataType.F16)
        else:
            if "." in clean or "e" in clean.lower():
                val = float(clean)
            elif clean.startswith("0x") or clean.startswith("0X"):
                val = int(clean, 16)
            else:
                val = int(clean)
            return Operand(imm_val=val, abs_mod=abs_m, neg_mod=neg_m, dtype=DataType.F16)

    src0 = parse_operand(operands_raw[1])
    src1 = parse_operand(operands_raw[2])

    clamp_m = "clamp" in tokens
    omod_m = "none"
    for t in tokens:
        if t in ("mul:2", "mul:4", "div:2"):
            omod_m = t

    op_sel = (False, False, False)
    for t in tokens:
        if t.startswith("op_sel:"):
            m = re.search(r"op_sel:\[(\d+),(\d+),(\d+)\]", t)
            if m:
                op_sel = (bool(int(m.group(1))), bool(int(m.group(2))), bool(int(m.group(3))))

    mods = InstructionModifiers(clamp=clamp_m, omod=omod_m, op_sel=op_sel)
    ast = InstructionAST(
        opcode=Opcode.V_SUBREV_F16,
        dst=dst,
        src0=src0,
        src1=src1,
        mods=mods,
        encoding=encoding,
    )
    ast.validate()
    return ast


# ============================================================================
# Phase 3: Deterministic Scalar Instruction Interpreter
# ============================================================================

class ScalarMachineState:
    """Deterministic scalar machine state representing VGPRs, SGPRs, and status registers."""

    def __init__(self) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.vcc: int = 0

    def get_vgpr(self, index: int) -> int:
        if not (0 <= index < 256):
            raise IndexError(f"VGPR index {index} out of bounds.")
        return self.vgprs[index]

    def set_vgpr(self, index: int, val: int) -> None:
        if not (0 <= index < 256):
            raise IndexError(f"VGPR index {index} out of bounds.")
        self.vgprs[index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        if not (0 <= index < 106):
            raise IndexError(f"SGPR index {index} out of bounds.")
        return self.sgprs[index]

    def set_sgpr(self, index: int, val: int) -> None:
        if not (0 <= index < 106):
            raise IndexError(f"SGPR index {index} out of bounds.")
        self.sgprs[index] = val & MASK32

    def read_reg16(self, reg: Register, op_sel_hi: bool = False) -> int:
        """Read a 16-bit halfword from a VGPR or SGPR."""
        use_hi = reg.hi or op_sel_hi
        word = self.get_vgpr(reg.index) if reg.kind == RegKind.VGPR else self.get_sgpr(reg.index)
        return extract_half(word, use_hi)

    def write_vgpr16(self, index: int, val16: int, hi: bool = False) -> None:
        """Write 16-bit halfword into VGPR, preserving the opposite 16-bit half."""
        old_word = self.get_vgpr(index)
        new_word = insert_half(old_word, val16, dst_hi=hi)
        self.set_vgpr(index, new_word)

    def eval_operand(self, op: Operand, op_sel_hi: bool = False) -> int:
        """Evaluate an operand to an unsigned 16-bit bit pattern, applying abs/neg."""
        if op.is_register():
            assert op.reg is not None
            raw = self.read_reg16(op.reg, op_sel_hi=op_sel_hi)
        elif op.is_literal():
            assert op.imm_val is not None
            if isinstance(op.imm_val, float):
                raw = f16_to_bits(op.imm_val)
            else:
                raw = op.imm_val & MASK16
        else:
            raise ValueError("Invalid operand.")
        return apply_f16_input_mods(raw, op.abs_mod, op.neg_mod)


class ScalarVALUInterpreter:
    """Deterministic scalar instruction interpreter for v_subrev_f16."""

    def step(self, instr: Union[InstructionAST, str], state: ScalarMachineState) -> int:
        """Execute a single v_subrev_f16 instruction against scalar machine state."""
        if isinstance(instr, str):
            ast = parse_subrev_f16_asm(instr)
        else:
            ast = instr
        ast.validate()

        src0_hi = ast.mods.op_sel[0]
        src1_hi = ast.mods.op_sel[1]
        dst_hi = ast.mods.op_sel[2] or ast.dst.hi

        s0 = state.eval_operand(ast.src0, op_sel_hi=src0_hi)
        s1 = state.eval_operand(ast.src1, op_sel_hi=src1_hi)

        # In reverse subtraction: D = S1 - S0
        res16 = ref_subrev_f16(s0, s1, clamp=ast.mods.clamp, omod=ast.mods.omod)
        state.write_vgpr16(ast.dst.index, res16, hi=dst_hi)
        return res16


# ============================================================================
# Phase 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ============================================================================

class SIMDWaveState:
    """SIMD wavefront state modeling Wave32/Wave64 execution, multi-lane VGPRs, and EXEC masks."""

    def __init__(self, wave_size: int = 64, exec_mask: Optional[int] = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave_size: {wave_size}. Must be 32 or 64.")
        self.wave_size: int = wave_size
        self.exec_mask: int = ((1 << wave_size) - 1) if exec_mask is None else exec_mask
        self.vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: list[int] = [0] * 106
        self.vcc: int = 0

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane {lane} out of range for wave_size {self.wave_size}.")
        return bool((self.exec_mask >> lane) & 1)

    def get_lane_vgpr(self, lane: int, index: int) -> int:
        return self.vgprs[lane][index]

    def set_lane_vgpr(self, lane: int, index: int, val: int) -> None:
        self.vgprs[lane][index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index]

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32

    def read_lane_reg16(self, lane: int, reg: Register, op_sel_hi: bool = False) -> int:
        use_hi = reg.hi or op_sel_hi
        word = self.get_lane_vgpr(lane, reg.index) if reg.kind == RegKind.VGPR else self.get_sgpr(reg.index)
        return extract_half(word, use_hi)

    def write_lane_vgpr16(self, lane: int, index: int, val16: int, hi: bool = False) -> None:
        old_word = self.get_lane_vgpr(lane, index)
        new_word = insert_half(old_word, val16, dst_hi=hi)
        self.set_lane_vgpr(lane, index, new_word)

    def eval_lane_operand(self, lane: int, op: Operand, op_sel_hi: bool = False) -> int:
        if op.is_register():
            assert op.reg is not None
            raw = self.read_lane_reg16(lane, op.reg, op_sel_hi=op_sel_hi)
        elif op.is_literal():
            assert op.imm_val is not None
            if isinstance(op.imm_val, float):
                raw = f16_to_bits(op.imm_val)
            else:
                raw = op.imm_val & MASK16
        else:
            raise ValueError("Invalid operand.")
        return apply_f16_input_mods(raw, op.abs_mod, op.neg_mod)


class SIMDVALUInterpreter:
    """SIMD wavefront instruction interpreter for v_subrev_f16 under EXEC masking."""

    def step(self, instr: Union[InstructionAST, str], state: SIMDWaveState) -> list[Optional[int]]:
        """Step instruction across all wavefront lanes according to state.exec_mask."""
        if isinstance(instr, str):
            ast = parse_subrev_f16_asm(instr)
        else:
            ast = instr
        ast.validate()

        src0_hi = ast.mods.op_sel[0]
        src1_hi = ast.mods.op_sel[1]
        dst_hi = ast.mods.op_sel[2] or ast.dst.hi

        results: list[Optional[int]] = [None] * state.wave_size
        for lane in range(state.wave_size):
            if not state.is_lane_active(lane):
                continue
            s0 = state.eval_lane_operand(lane, ast.src0, op_sel_hi=src0_hi)
            s1 = state.eval_lane_operand(lane, ast.src1, op_sel_hi=src1_hi)
            res16 = ref_subrev_f16(s0, s1, clamp=ast.mods.clamp, omod=ast.mods.omod)
            state.write_lane_vgpr16(lane, ast.dst.index, res16, hi=dst_hi)
            results[lane] = res16
        return results


# ============================================================================
# Phase 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ============================================================================

@dataclass
class F16StatusFlags:
    """IEEE 754 and RDNA2 floating-point exception flags and condition codes."""
    invalid: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False
    zero: bool = False
    negative: bool = False
    denormal: bool = False
    infinite: bool = False
    nan: bool = False


def evaluate_v_subrev_f16_flags(s0_bits: int, s1_bits: int, res_bits: int) -> F16StatusFlags:
    """Evaluate architectural status flags for v_subrev_f16 (D = S1 - S0)."""
    flags = F16StatusFlags()

    s0_nan = is_nan_f16(s0_bits)
    s1_nan = is_nan_f16(s1_bits)
    s0_snan = is_snan_f16(s0_bits)
    s1_snan = is_snan_f16(s1_bits)
    s0_inf = is_inf_f16(s0_bits)
    s1_inf = is_inf_f16(s1_bits)

    # 1. Invalid operation
    if s0_snan or s1_snan:
        flags.invalid = True
    elif s0_inf and s1_inf:
        # (+inf) - (+inf) or (-inf) - (-inf)
        if (s0_bits & SIGN_MASK_F16) == (s1_bits & SIGN_MASK_F16):
            flags.invalid = True

    # 2. Denormal operand or result
    if is_subnormal_f16(s0_bits) or is_subnormal_f16(s1_bits) or is_subnormal_f16(res_bits):
        flags.denormal = True

    # 3. Result classification
    if is_nan_f16(res_bits):
        flags.nan = True
    elif is_inf_f16(res_bits):
        flags.infinite = True
        # If inputs were finite, overflow occurred
        if not s0_inf and not s1_inf and not s0_nan and not s1_nan:
            flags.overflow = True
            flags.inexact = True
    elif is_zero_f16(res_bits):
        flags.zero = True
    else:
        # Check inexactness via rational fraction
        if not s0_nan and not s1_nan and not s0_inf and not s1_inf:
            frac0 = f16_bits_to_fraction(s0_bits)
            frac1 = f16_bits_to_fraction(s1_bits)
            exact_diff = frac1 - frac0
            res_frac = f16_bits_to_fraction(res_bits)
            if exact_diff != res_frac:
                flags.inexact = True
            if is_subnormal_f16(res_bits) and flags.inexact:
                flags.underflow = True

    flags.negative = bool(res_bits & SIGN_MASK_F16)
    return flags


def compute_f16_significand_borrow(s0_bits: int, s1_bits: int) -> tuple[int, int, int]:
    """Simulate hardware significand alignment and borrow generation for S1 - S0.

    Returns:
        (aligned_mant0, aligned_mant1, borrow_out)
    """
    _, exp0, mant0 = split_f16_bits(s0_bits)
    _, exp1, mant1 = split_f16_bits(s1_bits)

    # Add hidden bit for normalized floats
    sig0 = (1 << 10) | mant0 if exp0 > 0 else mant0
    sig1 = (1 << 10) | mant1 if exp1 > 0 else mant1

    eff_exp0 = exp0 if exp0 > 0 else 1
    eff_exp1 = exp1 if exp1 > 0 else 1

    if eff_exp1 >= eff_exp0:
        shift = min(15, eff_exp1 - eff_exp0)
        aligned0 = sig0 >> shift
        aligned1 = sig1
    else:
        shift = min(15, eff_exp0 - eff_exp1)
        aligned0 = sig0
        aligned1 = sig1 >> shift

    diff = aligned1 - aligned0
    borrow_out = 1 if diff < 0 else 0
    return aligned0, aligned1, borrow_out


# ============================================================================
# Unit Tests
# ============================================================================

class TestValuSubrevF16Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model of v_subrev_f16."""

    def test_basic_normal_subtraction(self) -> None:
        # 3.0 - 1.0 = 2.0 (S1=3.0, S0=1.0)
        res = ref_subrev_f16(ONE_F16, THREE_F16)
        self.assertEqual(res, TWO_F16)
        self.assertEqual(bits_to_f16(res), 2.0)

        # 4.0 - 1.5 = 2.5
        f_1p5 = f16_to_bits(1.5)
        res = ref_subrev_f16(f_1p5, FOUR_F16)
        self.assertEqual(bits_to_f16(res), 2.5)

        # 0.5 - 2.0 = -1.5
        res = ref_subrev_f16(TWO_F16, HALF_F16)
        self.assertEqual(bits_to_f16(res), -1.5)

    def test_reverse_subtraction_asymmetry(self) -> None:
        # S1 - S0 != S0 - S1
        res1 = ref_subrev_f16(ONE_F16, THREE_F16)   # 3.0 - 1.0 = 2.0
        res2 = ref_subrev_f16(THREE_F16, ONE_F16)   # 1.0 - 3.0 = -2.0
        self.assertEqual(bits_to_f16(res1), 2.0)
        self.assertEqual(bits_to_f16(res2), -2.0)
        self.assertNotEqual(res1, res2)

    def test_cross_check_with_rational_oracle(self) -> None:
        test_values = [
            f16_to_bits(x)
            for x in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 7.5, 12.25, -0.5, -1.0, -2.5]
        ]
        for s0 in test_values:
            for s1 in test_values:
                ref_res = ref_subrev_f16(s0, s1)
                rat_res = ref_subrev_f16_rational(s0, s1)
                self.assertEqual(ref_res, rat_res, f"Mismatch for s0={s0:#06x}, s1={s1:#06x}")

    def test_signed_zeros(self) -> None:
        # (+0.0) - (+0.0) == +0.0
        self.assertEqual(ref_subrev_f16(POS_ZERO_F16, POS_ZERO_F16), POS_ZERO_F16)
        # (-0.0) - (-0.0) == +0.0
        self.assertEqual(ref_subrev_f16(NEG_ZERO_F16, NEG_ZERO_F16), POS_ZERO_F16)
        # (+0.0) - (-0.0) == +0.0
        self.assertEqual(ref_subrev_f16(NEG_ZERO_F16, POS_ZERO_F16), POS_ZERO_F16)
        # (-0.0) - (+0.0) == -0.0
        self.assertEqual(ref_subrev_f16(POS_ZERO_F16, NEG_ZERO_F16), NEG_ZERO_F16)

    def test_infinity_arithmetic(self) -> None:
        # (+inf) - (+inf) = Canonical Quiet NaN
        self.assertEqual(ref_subrev_f16(POS_INF_F16, POS_INF_F16), CANONICAL_QNAN_F16)
        # (-inf) - (-inf) = Canonical Quiet NaN
        self.assertEqual(ref_subrev_f16(NEG_INF_F16, NEG_INF_F16), CANONICAL_QNAN_F16)
        # (+inf) - (-inf) = +inf
        self.assertEqual(ref_subrev_f16(NEG_INF_F16, POS_INF_F16), POS_INF_F16)
        # (-inf) - (+inf) = -inf
        self.assertEqual(ref_subrev_f16(POS_INF_F16, NEG_INF_F16), NEG_INF_F16)
        # finite - (+inf) = -inf
        self.assertEqual(ref_subrev_f16(POS_INF_F16, ONE_F16), NEG_INF_F16)
        # finite - (-inf) = +inf
        self.assertEqual(ref_subrev_f16(NEG_INF_F16, ONE_F16), POS_INF_F16)
        # (+inf) - finite = +inf
        self.assertEqual(ref_subrev_f16(ONE_F16, POS_INF_F16), POS_INF_F16)
        # (-inf) - finite = -inf
        self.assertEqual(ref_subrev_f16(ONE_F16, NEG_INF_F16), NEG_INF_F16)

    def test_nan_handling_and_quieting(self) -> None:
        # Quiet NaN in src0
        res = ref_subrev_f16(CANONICAL_QNAN_F16, ONE_F16)
        self.assertTrue(is_qnan_f16(res))

        # Quiet NaN in src1
        res = ref_subrev_f16(ONE_F16, CANONICAL_QNAN_F16)
        self.assertTrue(is_qnan_f16(res))

        # Signaling NaN is quieted
        res_snan = ref_subrev_f16(SNAN_DEFAULT_F16, ONE_F16)
        self.assertTrue(is_qnan_f16(res_snan))
        self.assertTrue(res_snan & QNAN_BIT_F16)

    def test_subnormal_arithmetic(self) -> None:
        # Subnormal subtraction
        res = ref_subrev_f16(MIN_SUBNORMAL_F16, MAX_SUBNORMAL_F16)
        self.assertEqual(res, MAX_SUBNORMAL_F16 - MIN_SUBNORMAL_F16)
        # Min normal minus smallest subnormal
        res2 = ref_subrev_f16(MIN_SUBNORMAL_F16, MIN_NORMAL_F16)
        self.assertEqual(res2, MAX_SUBNORMAL_F16)

    def test_finite_overflow(self) -> None:
        # 65504.0 - (-65504.0) = +inf
        res = ref_subrev_f16(NEG_MAX_FINITE_F16, MAX_FINITE_F16)
        self.assertEqual(res, POS_INF_F16)
        # -65504.0 - 65504.0 = -inf
        res2 = ref_subrev_f16(MAX_FINITE_F16, NEG_MAX_FINITE_F16)
        self.assertEqual(res2, NEG_INF_F16)

    def test_input_modifiers(self) -> None:
        # abs(s0): S1 - |S0| -> 3.0 - |-1.0| = 2.0
        res = ref_subrev_f16(NEG_ONE_F16, THREE_F16, abs0=True)
        self.assertEqual(bits_to_f16(res), 2.0)

        # neg(s0): S1 - (-S0) -> 3.0 - (-1.0) = 4.0
        res = ref_subrev_f16(ONE_F16, THREE_F16, neg0=True)
        self.assertEqual(bits_to_f16(res), 4.0)

        # abs(s1): |S1| - S0 -> |-3.0| - 1.0 = 2.0
        f_neg3 = f16_to_bits(-3.0)
        res = ref_subrev_f16(ONE_F16, f_neg3, abs1=True)
        self.assertEqual(bits_to_f16(res), 2.0)

        # neg(s1): (-S1) - S0 -> (-3.0) - 1.0 = -4.0
        res = ref_subrev_f16(ONE_F16, THREE_F16, neg1=True)
        self.assertEqual(bits_to_f16(res), -4.0)

        # combined -|x|
        res = ref_subrev_f16(ONE_F16, THREE_F16, abs1=True, neg1=True)
        self.assertEqual(bits_to_f16(res), -4.0)

    def test_output_modifiers(self) -> None:
        # clamp: negative result becomes +0.0
        res_neg = ref_subrev_f16(TWO_F16, ONE_F16, clamp=True)  # 1.0 - 2.0 = -1.0 -> 0.0
        self.assertEqual(res_neg, POS_ZERO_F16)

        # clamp: result > 1.0 becomes 1.0
        res_pos = ref_subrev_f16(ONE_F16, THREE_F16, clamp=True)  # 3.0 - 1.0 = 2.0 -> 1.0
        self.assertEqual(res_pos, ONE_F16)

        # clamp: NaN becomes +0.0
        res_nan = ref_subrev_f16(CANONICAL_QNAN_F16, ONE_F16, clamp=True)
        self.assertEqual(res_nan, POS_ZERO_F16)

        # omod mul:2
        res_m2 = ref_subrev_f16(ONE_F16, TWO_F16, omod="mul:2")  # (2.0 - 1.0) * 2 = 2.0
        self.assertEqual(bits_to_f16(res_m2), 2.0)

        # omod div:2
        res_d2 = ref_subrev_f16(ONE_F16, TWO_F16, omod="div:2")  # (2.0 - 1.0) / 2 = 0.5
        self.assertEqual(bits_to_f16(res_d2), 0.5)


class TestValuSubrevF16SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions for v_subrev_f16."""

    def test_ast_valid_creation_vop2_and_vop3(self) -> None:
        # VOP2
        ast_vop2 = InstructionAST(
            opcode=Opcode.V_SUBREV_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Operand(reg=Register(RegKind.VGPR, 1)),
            src1=Operand(reg=Register(RegKind.VGPR, 2)),
            encoding=Encoding.VOP2,
        )
        ast_vop2.validate()
        self.assertEqual(ast_vop2.to_asm(), "v_subrev_f16_e32 v0, v1, v2")

        # VOP3
        ast_vop3 = InstructionAST(
            opcode=Opcode.V_SUBREV_F16,
            dst=Register(RegKind.VGPR, 0, hi=True),
            src0=Operand(reg=Register(RegKind.VGPR, 1), abs_mod=True, neg_mod=True),
            src1=Operand(reg=Register(RegKind.SGPR, 2)),
            mods=InstructionModifiers(clamp=True, omod="mul:2", op_sel=(False, True, True)),
            encoding=Encoding.VOP3,
        )
        ast_vop3.validate()
        self.assertEqual(ast_vop3.to_asm(), "v_subrev_f16_e64 v0.h, -|v1|, s2 clamp mul:2 op_sel:[0,1,1]")

    def test_ast_to_asm_formatting(self) -> None:
        ast = InstructionAST(
            opcode=Opcode.V_SUBREV_F16,
            dst=Register(RegKind.VGPR, 5),
            src0=Operand(imm_val=1.5),
            src1=Operand(reg=Register(RegKind.VGPR, 10)),
            mods=InstructionModifiers(clamp=True),
            encoding=Encoding.VOP3,
        )
        self.assertEqual(ast.to_asm(), "v_subrev_f16_e64 v5, 1.5, v10 clamp")

    def test_asm_parser_vop2(self) -> None:
        ast = parse_subrev_f16_asm("v_subrev_f16_e32 v0, v1, v2")
        self.assertEqual(ast.opcode, Opcode.V_SUBREV_F16)
        self.assertEqual(ast.dst, Register(RegKind.VGPR, 0))
        self.assertEqual(ast.src0.reg, Register(RegKind.VGPR, 1))
        self.assertEqual(ast.src1.reg, Register(RegKind.VGPR, 2))
        self.assertEqual(ast.encoding, Encoding.VOP2)
        self.assertFalse(ast.mods.clamp)

    def test_asm_parser_vop3_modifiers_and_op_sel(self) -> None:
        ast = parse_subrev_f16_asm("v_subrev_f16_e64 v3.h, -|v1|, v2 clamp mul:4 op_sel:[1,0,1]")
        self.assertEqual(ast.encoding, Encoding.VOP3)
        self.assertTrue(ast.dst.hi)
        self.assertEqual(ast.dst.index, 3)
        self.assertTrue(ast.src0.abs_mod)
        self.assertTrue(ast.src0.neg_mod)
        self.assertTrue(ast.mods.clamp)
        self.assertEqual(ast.mods.omod, "mul:4")
        self.assertEqual(ast.mods.op_sel, (True, False, True))

    def test_ast_validation_rejects_sgpr_dst(self) -> None:
        ast = InstructionAST(
            opcode=Opcode.V_SUBREV_F16,
            dst=Register(RegKind.SGPR, 0),
            src0=Operand(reg=Register(RegKind.VGPR, 1)),
            src1=Operand(reg=Register(RegKind.VGPR, 2)),
        )
        with self.assertRaises(TypeError):
            ast.validate()

    def test_ast_validation_rejects_out_of_bounds_registers(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256).validate()
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 106).validate()

    def test_ast_validation_rejects_vop2_with_clamp_or_mods(self) -> None:
        ast_clamp = InstructionAST(
            opcode=Opcode.V_SUBREV_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Operand(reg=Register(RegKind.VGPR, 1)),
            src1=Operand(reg=Register(RegKind.VGPR, 2)),
            mods=InstructionModifiers(clamp=True),
            encoding=Encoding.VOP2,
        )
        with self.assertRaises(ValueError):
            ast_clamp.validate()

        ast_neg = InstructionAST(
            opcode=Opcode.V_SUBREV_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Operand(reg=Register(RegKind.VGPR, 1), neg_mod=True),
            src1=Operand(reg=Register(RegKind.VGPR, 2)),
            encoding=Encoding.VOP2,
        )
        with self.assertRaises(ValueError):
            ast_neg.validate()

    def test_ast_validation_rejects_wrong_data_type(self) -> None:
        op = Operand(reg=Register(RegKind.VGPR, 1), dtype=DataType.F32)
        with self.assertRaises(TypeError):
            op.validate()


class TestValuSubrevF16ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter for v_subrev_f16."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_scalar_step_basic_vgpr_sources(self) -> None:
        # Set v1 = 1.0 (low half), v2 = 3.0 (low half), canary in high half
        self.state.set_vgpr(0, 0xDEAD0000)
        self.state.set_vgpr(1, pack_half(ONE_F16, 0x1234))
        self.state.set_vgpr(2, pack_half(THREE_F16, 0x5678))

        res = self.interp.step("v_subrev_f16_e32 v0, v1, v2", self.state)
        self.assertEqual(res, TWO_F16)
        v0_val = self.state.get_vgpr(0)
        lo, hi = unpack_half(v0_val)
        self.assertEqual(lo, TWO_F16)
        self.assertEqual(hi, 0xDEAD)

    def test_scalar_step_sgpr_and_literal_sources(self) -> None:
        # S0 is literal 1.5, S1 is VGPR with 4.5 -> D = 4.5 - 1.5 = 3.0
        self.state.set_vgpr(1, f16_to_bits(4.5))
        res = self.interp.step("v_subrev_f16_e64 v0, 1.5, v1", self.state)
        self.assertEqual(bits_to_f16(res), 3.0)
        self.assertEqual(bits_to_f16(self.state.read_reg16(Register(RegKind.VGPR, 0))), 3.0)

        # S0 is SGPR with 0.5, S1 is VGPR with 2.0 -> D = 2.0 - 0.5 = 1.5
        self.state.set_sgpr(5, HALF_F16)
        self.state.set_vgpr(2, TWO_F16)
        res2 = self.interp.step("v_subrev_f16_e32 v0, s5, v2", self.state)
        self.assertEqual(bits_to_f16(res2), 1.5)

    def test_scalar_step_vop3_input_modifiers(self) -> None:
        # S0 = -1.0, S1 = 2.0. With -|s0|, S0 becomes -1.0. S1 - (-1.0) = 3.0
        self.state.set_sgpr(0, NEG_ONE_F16)
        self.state.set_vgpr(1, TWO_F16)
        res = self.interp.step("v_subrev_f16_e64 v0, -|s0|, v1", self.state)
        self.assertEqual(bits_to_f16(res), 3.0)

    def test_scalar_step_vop3_clamp_and_omod(self) -> None:
        # S0 = 3.0, S1 = 1.0 -> 1.0 - 3.0 = -2.0. Clamped to 0.0
        self.state.set_vgpr(1, THREE_F16)
        self.state.set_vgpr(2, ONE_F16)
        res = self.interp.step("v_subrev_f16_e64 v0, v1, v2 clamp", self.state)
        self.assertEqual(res, POS_ZERO_F16)

        # S0 = 0.5, S1 = 1.5 -> (1.5 - 0.5) * 4 = 4.0
        self.state.set_vgpr(1, HALF_F16)
        self.state.set_vgpr(2, f16_to_bits(1.5))
        res2 = self.interp.step("v_subrev_f16_e64 v0, v1, v2 mul:4", self.state)
        self.assertEqual(bits_to_f16(res2), 4.0)

    def test_scalar_step_op_sel_halves(self) -> None:
        # v1 has lo=0.25, hi=1.0. v2 has lo=2.0, hi=5.0
        self.state.set_vgpr(1, pack_half(QUARTER_F16, ONE_F16))
        self.state.set_vgpr(2, pack_half(TWO_F16, f16_to_bits(5.0)))
        self.state.set_vgpr(0, pack_half(0x1111, 0x2222))

        # op_sel:[1, 0, 1] reads v1.h (1.0) and v2.l (2.0), writes to v0.h:
        # D = 2.0 - 1.0 = 1.0 into v0.h. v0.l preserves 0x1111.
        res = self.interp.step("v_subrev_f16_e64 v0.h, v1, v2 op_sel:[1,0,1]", self.state)
        self.assertEqual(res, ONE_F16)
        v0_val = self.state.get_vgpr(0)
        lo, hi = unpack_half(v0_val)
        self.assertEqual(lo, 0x1111)
        self.assertEqual(hi, ONE_F16)

    def test_scalar_step_determinism(self) -> None:
        s_a = ScalarMachineState()
        s_b = ScalarMachineState()
        for s in (s_a, s_b):
            s.set_vgpr(1, pack_half(ONE_F16, TWO_F16))
            s.set_vgpr(2, pack_half(THREE_F16, FOUR_F16))
            self.interp.step("v_subrev_f16_e32 v0, v1, v2", s)
        self.assertEqual(s_a.vgprs, s_b.vgprs)


class TestValuSubrevF16SIMDExecMasking(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking for v_subrev_f16."""

    def setUp(self) -> None:
        self.simd_interp = SIMDVALUInterpreter()

    def test_wave32_all_lanes_active(self) -> None:
        state = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF)
        for lane in range(32):
            state.set_lane_vgpr(lane, 1, ONE_F16)                    # S0 = 1.0
            state.set_lane_vgpr(lane, 2, f16_to_bits(float(lane)))   # S1 = lane
            state.set_lane_vgpr(lane, 0, 0)

        # D = S1 - S0 = lane - 1.0
        results = self.simd_interp.step("v_subrev_f16_e32 v0, v1, v2", state)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            expected = f16_to_bits(float(lane) - 1.0)
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.read_lane_reg16(lane, Register(RegKind.VGPR, 0)), expected)

    def test_wave64_all_lanes_active(self) -> None:
        state = SIMDWaveState(wave_size=64, exec_mask=0xFFFFFFFFFFFFFFFF)
        for lane in range(64):
            state.set_lane_vgpr(lane, 1, f16_to_bits(0.5))
            state.set_lane_vgpr(lane, 2, f16_to_bits(float(lane) * 0.5))

        results = self.simd_interp.step("v_subrev_f16_e32 v0, v1, v2", state)
        for lane in range(64):
            expected = f16_to_bits(float(lane) * 0.5 - 0.5)
            self.assertEqual(results[lane], expected)

    def test_wave32_partial_exec_mask_preserves_inactive_lanes(self) -> None:
        # Alternating lanes active: mask = 0x55555555 (lanes 0, 2, 4, ...)
        exec_mask = 0x55555555
        state = SIMDWaveState(wave_size=32, exec_mask=exec_mask)
        canary = 0xCAFE
        for lane in range(32):
            state.set_lane_vgpr(lane, 0, pack_half(canary, canary))
            state.set_lane_vgpr(lane, 1, ONE_F16)
            state.set_lane_vgpr(lane, 2, FOUR_F16)  # 4.0 - 1.0 = 3.0

        results = self.simd_interp.step("v_subrev_f16_e32 v0, v1, v2", state)
        for lane in range(32):
            lo, hi = unpack_half(state.get_lane_vgpr(lane, 0))
            if (exec_mask >> lane) & 1:
                self.assertEqual(results[lane], THREE_F16)
                self.assertEqual(lo, THREE_F16)
                self.assertEqual(hi, canary)  # opposite half preserved
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(lo, canary)  # inactive lane untouched
                self.assertEqual(hi, canary)

    def test_wave64_high_dword_mask(self) -> None:
        # Lanes 0..31 inactive, lanes 32..63 active
        exec_mask = 0xFFFFFFFF00000000
        state = SIMDWaveState(wave_size=64, exec_mask=exec_mask)
        for lane in range(64):
            state.set_lane_vgpr(lane, 0, 0x12345678)
            state.set_lane_vgpr(lane, 1, HALF_F16)
            state.set_lane_vgpr(lane, 2, TWO_F16)  # 2.0 - 0.5 = 1.5

        results = self.simd_interp.step("v_subrev_f16_e32 v0, v1, v2", state)
        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(state.get_lane_vgpr(lane, 0), 0x12345678)
        for lane in range(32, 64):
            self.assertEqual(bits_to_f16(results[lane]), 1.5)
            self.assertEqual(extract_half(state.get_lane_vgpr(lane, 0), hi_sel=False), f16_to_bits(1.5))

    def test_simd_sgpr_scalar_broadcast(self) -> None:
        state = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF)
        # s4 holds scalar constant 10.0 (broadcast across all lanes as S0)
        state.set_sgpr(4, f16_to_bits(10.0))
        for lane in range(32):
            state.set_lane_vgpr(lane, 1, f16_to_bits(float(lane) + 10.0))  # S1 = lane + 10.0

        # D = S1 - S0 = (lane + 10.0) - 10.0 = lane
        results = self.simd_interp.step("v_subrev_f16_e32 v0, s4, v1", state)
        for lane in range(32):
            self.assertEqual(bits_to_f16(results[lane]), float(lane))


class TestValuSubrevF16RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation for v_subrev_f16."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarVALUInterpreter()
        self.simd_interp = SIMDVALUInterpreter()

    def test_dst_alias_src0(self) -> None:
        # v0 = 1.0, v1 = 3.0. Instr: v_subrev_f16 v0, v0, v1 -> v0 = 3.0 - 1.0 = 2.0
        state = ScalarMachineState()
        state.set_vgpr(0, pack_half(ONE_F16, 0xAAAA))
        state.set_vgpr(1, pack_half(THREE_F16, 0xBBBB))

        res = self.scalar_interp.step("v_subrev_f16_e32 v0, v0, v1", state)
        self.assertEqual(res, TWO_F16)
        lo, hi = unpack_half(state.get_vgpr(0))
        self.assertEqual(lo, TWO_F16)
        self.assertEqual(hi, 0xAAAA)  # opposite half preserved
        self.assertEqual(state.get_vgpr(1), pack_half(THREE_F16, 0xBBBB))  # src1 untouched

    def test_dst_alias_src1(self) -> None:
        # v0 = 1.0, v1 = 3.0. Instr: v_subrev_f16 v1, v0, v1 -> v1 = 3.0 - 1.0 = 2.0
        state = ScalarMachineState()
        state.set_vgpr(0, pack_half(ONE_F16, 0xAAAA))
        state.set_vgpr(1, pack_half(THREE_F16, 0xBBBB))

        res = self.scalar_interp.step("v_subrev_f16_e32 v1, v0, v1", state)
        self.assertEqual(res, TWO_F16)
        lo, hi = unpack_half(state.get_vgpr(1))
        self.assertEqual(lo, TWO_F16)
        self.assertEqual(hi, 0xBBBB)
        self.assertEqual(state.get_vgpr(0), pack_half(ONE_F16, 0xAAAA))

    def test_dst_alias_src0_and_src1_self_subtraction(self) -> None:
        # Self-subtraction: v0 = 5.0. v_subrev_f16 v0, v0, v0 -> 5.0 - 5.0 = +0.0
        state = ScalarMachineState()
        f_5p0 = f16_to_bits(5.0)
        state.set_vgpr(0, pack_half(f_5p0, 0xCCCC))

        res = self.scalar_interp.step("v_subrev_f16_e32 v0, v0, v0", state)
        self.assertEqual(res, POS_ZERO_F16)
        lo, hi = unpack_half(state.get_vgpr(0))
        self.assertEqual(lo, POS_ZERO_F16)
        self.assertEqual(hi, 0xCCCC)

    def test_intra_dword_half_word_aliasing_write_hi(self) -> None:
        # v0 holds lo=1.0, hi=4.0.
        # Instr: v_subrev_f16_e64 v0.h, v0, v0.h
        # S0 reads v0.l (1.0), S1 reads v0.h (4.0).
        # D writes to v0.h = 4.0 - 1.0 = 3.0. v0.l must remain 1.0.
        state = ScalarMachineState()
        state.set_vgpr(0, pack_half(ONE_F16, FOUR_F16))

        ast = InstructionAST(
            opcode=Opcode.V_SUBREV_F16,
            dst=Register(RegKind.VGPR, 0, hi=True),
            src0=Operand(reg=Register(RegKind.VGPR, 0, hi=False)),
            src1=Operand(reg=Register(RegKind.VGPR, 0, hi=True)),
            encoding=Encoding.VOP3,
        )
        res = self.scalar_interp.step(ast, state)
        self.assertEqual(res, THREE_F16)
        lo, hi = unpack_half(state.get_vgpr(0))
        self.assertEqual(lo, ONE_F16)    # preserved
        self.assertEqual(hi, THREE_F16)  # updated

    def test_intra_dword_half_word_aliasing_write_lo(self) -> None:
        # v0 holds lo=1.0, hi=4.0.
        # Instr: v_subrev_f16_e64 v0, v0.h, v0
        # S0 reads v0.h (4.0), S1 reads v0.l (1.0).
        # D writes to v0.l = 1.0 - 4.0 = -3.0. v0.h must remain 4.0.
        state = ScalarMachineState()
        state.set_vgpr(0, pack_half(ONE_F16, FOUR_F16))

        ast = InstructionAST(
            opcode=Opcode.V_SUBREV_F16,
            dst=Register(RegKind.VGPR, 0, hi=False),
            src0=Operand(reg=Register(RegKind.VGPR, 0, hi=True)),
            src1=Operand(reg=Register(RegKind.VGPR, 0, hi=False)),
            encoding=Encoding.VOP3,
        )
        res = self.scalar_interp.step(ast, state)
        self.assertEqual(bits_to_f16(res), -3.0)
        lo, hi = unpack_half(state.get_vgpr(0))
        self.assertEqual(lo, f16_to_bits(-3.0))
        self.assertEqual(hi, FOUR_F16)  # preserved

    def test_simd_multi_lane_aliasing(self) -> None:
        state = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF)
        for lane in range(32):
            state.set_lane_vgpr(lane, 0, f16_to_bits(float(lane) * 2.0))
            state.set_lane_vgpr(lane, 1, f16_to_bits(100.0))

        # In-place update: v0 = v1 - v0 = 100.0 - lane * 2.0
        self.simd_interp.step("v_subrev_f16_e32 v0, v0, v1", state)
        for lane in range(32):
            expected = f16_to_bits(100.0 - float(lane) * 2.0)
            self.assertEqual(extract_half(state.get_lane_vgpr(lane, 0), hi_sel=False), expected)


class TestValuSubrevF16ConditionCodesAndFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_exact_finite_flags(self) -> None:
        # S0 = 1.0, S1 = 3.0 -> D = 2.0 (exact)
        s0 = ONE_F16
        s1 = THREE_F16
        res = ref_subrev_f16(s0, s1)
        flags = evaluate_v_subrev_f16_flags(s0, s1, res)
        self.assertFalse(flags.invalid)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.underflow)
        self.assertFalse(flags.inexact)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.denormal)
        self.assertFalse(flags.infinite)
        self.assertFalse(flags.nan)

    def test_inexact_flag(self) -> None:
        # 1.0 - 0.0001 in f16 is inexact
        s0 = f16_to_bits(0.0001)
        s1 = ONE_F16
        res = ref_subrev_f16(s0, s1)
        flags = evaluate_v_subrev_f16_flags(s0, s1, res)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.invalid)

    def test_overflow_flag(self) -> None:
        # MAX_FINITE - (-MAX_FINITE) overflows to +inf
        s0 = NEG_MAX_FINITE_F16
        s1 = MAX_FINITE_F16
        res = ref_subrev_f16(s0, s1)
        self.assertEqual(res, POS_INF_F16)
        flags = evaluate_v_subrev_f16_flags(s0, s1, res)
        self.assertTrue(flags.overflow)
        self.assertTrue(flags.infinite)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.invalid)

    def test_underflow_and_denorm_flags(self) -> None:
        # Subnormal subtraction
        s0 = MIN_SUBNORMAL_F16
        s1 = MAX_SUBNORMAL_F16
        res = ref_subrev_f16(s0, s1)
        flags = evaluate_v_subrev_f16_flags(s0, s1, res)
        self.assertTrue(flags.denormal)
        self.assertFalse(flags.overflow)

    def test_invalid_operation_flag(self) -> None:
        # (+inf) - (+inf) is invalid
        res = ref_subrev_f16(POS_INF_F16, POS_INF_F16)
        flags = evaluate_v_subrev_f16_flags(POS_INF_F16, POS_INF_F16, res)
        self.assertTrue(flags.invalid)
        self.assertTrue(flags.nan)

        # Signaling NaN triggers invalid operation
        res2 = ref_subrev_f16(SNAN_DEFAULT_F16, ONE_F16)
        flags2 = evaluate_v_subrev_f16_flags(SNAN_DEFAULT_F16, ONE_F16, res2)
        self.assertTrue(flags2.invalid)
        self.assertTrue(flags2.nan)

    def test_zero_and_negative_flags(self) -> None:
        # (-0.0) - (+0.0) = -0.0
        res_nz = ref_subrev_f16(POS_ZERO_F16, NEG_ZERO_F16)
        flags_nz = evaluate_v_subrev_f16_flags(POS_ZERO_F16, NEG_ZERO_F16, res_nz)
        self.assertTrue(flags_nz.zero)
        self.assertTrue(flags_nz.negative)

        # (+0.0) - (-0.0) = +0.0
        res_pz = ref_subrev_f16(NEG_ZERO_F16, POS_ZERO_F16)
        flags_pz = evaluate_v_subrev_f16_flags(NEG_ZERO_F16, POS_ZERO_F16, res_pz)
        self.assertTrue(flags_pz.zero)
        self.assertFalse(flags_pz.negative)

        # 1.0 - 3.0 = -2.0
        res_neg = ref_subrev_f16(THREE_F16, ONE_F16)
        flags_neg = evaluate_v_subrev_f16_flags(THREE_F16, ONE_F16, res_neg)
        self.assertFalse(flags_neg.zero)
        self.assertTrue(flags_neg.negative)

    def test_significand_borrow_generation(self) -> None:
        # S1=1.0, S0=2.0 -> S1 < S0 so borrow_out = 1
        _, _, borrow1 = compute_f16_significand_borrow(TWO_F16, ONE_F16)
        self.assertEqual(borrow1, 1)

        # S1=3.0, S0=1.0 -> S1 > S0 so borrow_out = 0
        _, _, borrow0 = compute_f16_significand_borrow(ONE_F16, THREE_F16)
        self.assertEqual(borrow0, 0)


class TestValuSubrevF16IEEE754Boundaries(unittest.TestCase):
    """Exhaustive IEEE 754-2008 boundary extremes, subnormals, and NaN handling tests for v_subrev_f16."""

    def test_signed_zero_combinations(self) -> None:
        # All 4 permutations of signed zero operands
        # S1 - S0:
        # (+0.0) - (+0.0) -> +0.0
        self.assertEqual(ref_subrev_f16(POS_ZERO_F16, POS_ZERO_F16), POS_ZERO_F16)
        # (+0.0) - (-0.0) -> +0.0
        self.assertEqual(ref_subrev_f16(NEG_ZERO_F16, POS_ZERO_F16), POS_ZERO_F16)
        # (-0.0) - (+0.0) -> -0.0
        self.assertEqual(ref_subrev_f16(POS_ZERO_F16, NEG_ZERO_F16), NEG_ZERO_F16)
        # (-0.0) - (-0.0) -> +0.0
        self.assertEqual(ref_subrev_f16(NEG_ZERO_F16, NEG_ZERO_F16), POS_ZERO_F16)

    def test_subnormal_boundary_transitions(self) -> None:
        # MIN_NORMAL = 0x0400 (2^-14). Subtract MIN_SUBNORMAL = 0x0001 (2^-24).
        # Expected result is MAX_SUBNORMAL = 0x03FF.
        res = ref_subrev_f16(MIN_SUBNORMAL_F16, MIN_NORMAL_F16)
        self.assertEqual(res, MAX_SUBNORMAL_F16)
        self.assertTrue(is_subnormal_f16(res))

        # Reversing: MIN_SUBNORMAL - MIN_NORMAL = -MAX_SUBNORMAL
        res_rev = ref_subrev_f16(MIN_NORMAL_F16, MIN_SUBNORMAL_F16)
        self.assertEqual(res_rev, MAX_SUBNORMAL_F16 | SIGN_MASK_F16)
        self.assertTrue(is_subnormal_f16(res_rev))

    def test_subnormal_precision_and_gradual_underflow(self) -> None:
        # Subtraction of two close normalized numbers that results in a subnormal
        # 1.0009765625 (0x3C01) - 1.0 (0x3C00) = 2^-10 in normal range,
        # but near MIN_NORMAL: 0x0401 - 0x0400 = 0x0001 (MIN_SUBNORMAL)
        val_a = 0x0401
        val_b = 0x0400
        res = ref_subrev_f16(val_b, val_a)
        self.assertEqual(res, MIN_SUBNORMAL_F16)
        self.assertTrue(is_subnormal_f16(res))

    def test_infinity_edge_cases(self) -> None:
        # (+inf) - (+inf) -> Canonical Quiet NaN
        self.assertEqual(ref_subrev_f16(POS_INF_F16, POS_INF_F16), CANONICAL_QNAN_F16)
        # (-inf) - (-inf) -> Canonical Quiet NaN
        self.assertEqual(ref_subrev_f16(NEG_INF_F16, NEG_INF_F16), CANONICAL_QNAN_F16)
        # (+inf) - (-inf) -> +inf
        self.assertEqual(ref_subrev_f16(NEG_INF_F16, POS_INF_F16), POS_INF_F16)
        # (-inf) - (+inf) -> -inf
        self.assertEqual(ref_subrev_f16(POS_INF_F16, NEG_INF_F16), NEG_INF_F16)

        # Infinities with modifiers: S1 - (-|S0|): (+inf) - (-|+inf|) = (+inf) - (-inf) = +inf
        res_mod = ref_subrev_f16(POS_INF_F16, POS_INF_F16, abs0=True, neg0=True)
        self.assertEqual(res_mod, POS_INF_F16)

    def test_nan_quieting_and_payload_preservation(self) -> None:
        # Signaling NaNs must have quiet bit (bit 9) set
        snan_payload = 0x7C55
        self.assertTrue(is_snan_f16(snan_payload))
        res_s0 = ref_subrev_f16(snan_payload, ONE_F16)
        self.assertTrue(is_qnan_f16(res_s0))
        self.assertEqual(res_s0 & QNAN_BIT_F16, QNAN_BIT_F16)

        res_s1 = ref_subrev_f16(ONE_F16, snan_payload)
        self.assertTrue(is_qnan_f16(res_s1))
        self.assertEqual(res_s1 & QNAN_BIT_F16, QNAN_BIT_F16)

        # Quiet NaN payload preserved
        qnan_payload = 0x7EAA
        res_q = ref_subrev_f16(qnan_payload, ONE_F16)
        self.assertTrue(is_qnan_f16(res_q))

    def test_nan_clamping_to_zero(self) -> None:
        # In RDNA2, clamping any NaN result produces +0.0
        qnan = CANONICAL_QNAN_F16
        snan = SNAN_DEFAULT_F16
        self.assertEqual(ref_subrev_f16(qnan, ONE_F16, clamp=True), POS_ZERO_F16)
        self.assertEqual(ref_subrev_f16(ONE_F16, qnan, clamp=True), POS_ZERO_F16)
        self.assertEqual(ref_subrev_f16(snan, ONE_F16, clamp=True), POS_ZERO_F16)
        self.assertEqual(ref_subrev_f16(POS_INF_F16, POS_INF_F16, clamp=True), POS_ZERO_F16)

    def test_max_finite_overflow_boundaries(self) -> None:
        # 65504.0 - (-1.0) = 65505.0 -> rounds to 65504.0 (even tie)
        res_near = ref_subrev_f16(NEG_ONE_F16, MAX_FINITE_F16)
        self.assertEqual(res_near, MAX_FINITE_F16)

        # 65504.0 - (-16.0) -> overflows to +inf
        f_neg16 = f16_to_bits(-16.0)
        res_over = ref_subrev_f16(f_neg16, MAX_FINITE_F16)
        self.assertEqual(res_over, POS_INF_F16)

    def test_subnormal_with_modifiers(self) -> None:
        # S0 = -MIN_SUBNORMAL, S1 = MIN_SUBNORMAL. With abs0: S1 - |-S0| = 0
        s0 = MIN_SUBNORMAL_F16 | SIGN_MASK_F16
        s1 = MIN_SUBNORMAL_F16
        res = ref_subrev_f16(s0, s1, abs0=True)
        self.assertEqual(res, POS_ZERO_F16)


class TestValuSubrevF16NegativeControls(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and invalid AST inputs for v_subrev_f16."""

    def test_syntax_rejection_empty_and_bad_opcode(self) -> None:
        with self.assertRaises(ValueError):
            parse_subrev_f16_asm("")
        with self.assertRaises(ValueError):
            parse_subrev_f16_asm("v_add_f16_e32 v0, v1, v2")
        with self.assertRaises(ValueError):
            parse_subrev_f16_asm("v_subrev_f16_e32 v0, v1")

    def test_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1).validate()
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256).validate()
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, -1).validate()
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 106).validate()

    def test_invalid_destination_kind(self) -> None:
        # VALU destination cannot be SGPR
        ast = InstructionAST(
            opcode=Opcode.V_SUBREV_F16,
            dst=Register(RegKind.SGPR, 0),
            src0=Operand(reg=Register(RegKind.VGPR, 1)),
            src1=Operand(reg=Register(RegKind.VGPR, 2)),
        )
        with self.assertRaises(TypeError):
            ast.validate()

    def test_invalid_vop2_modifiers(self) -> None:
        # VOP2 cannot have clamp, omod, op_sel, or input modifiers
        ast_clamp = InstructionAST(
            opcode=Opcode.V_SUBREV_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Operand(reg=Register(RegKind.VGPR, 1)),
            src1=Operand(reg=Register(RegKind.VGPR, 2)),
            mods=InstructionModifiers(clamp=True),
            encoding=Encoding.VOP2,
        )
        with self.assertRaises(ValueError):
            ast_clamp.validate()

        ast_omod = InstructionAST(
            opcode=Opcode.V_SUBREV_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Operand(reg=Register(RegKind.VGPR, 1)),
            src1=Operand(reg=Register(RegKind.VGPR, 2)),
            mods=InstructionModifiers(omod="mul:2"),
            encoding=Encoding.VOP2,
        )
        with self.assertRaises(ValueError):
            ast_omod.validate()

        ast_op_sel = InstructionAST(
            opcode=Opcode.V_SUBREV_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Operand(reg=Register(RegKind.VGPR, 1)),
            src1=Operand(reg=Register(RegKind.VGPR, 2)),
            mods=InstructionModifiers(op_sel=(True, False, False)),
            encoding=Encoding.VOP2,
        )
        with self.assertRaises(ValueError):
            ast_op_sel.validate()

    def test_state_out_of_bounds_access(self) -> None:
        scalar = ScalarMachineState()
        with self.assertRaises(IndexError):
            scalar.get_vgpr(256)
        with self.assertRaises(IndexError):
            scalar.get_sgpr(106)
        with self.assertRaises(IndexError):
            scalar.set_vgpr(256, 0)
        with self.assertRaises(IndexError):
            scalar.set_sgpr(106, 0)

    def test_simd_invalid_wave_size_and_lanes(self) -> None:
        with self.assertRaises(ValueError):
            SIMDWaveState(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWaveState(wave_size=128)
        simd = SIMDWaveState(wave_size=32)
        with self.assertRaises(IndexError):
            simd.is_lane_active(32)


class TestValuSubrevF16MutationDetection(unittest.TestCase):
    """Mutation tests: verifies that intentional bugs/mutations in subrev semantics are caught."""

    def test_mutation_non_reversed_subtraction(self) -> None:
        # Mutation: computing S0 - S1 instead of S1 - S0
        def mutated_sub(s0: int, s1: int) -> int:
            return ref_subrev_f16(s1, s0)  # reversed operands

        s0 = ONE_F16    # 1.0
        s1 = THREE_F16  # 3.0
        # True v_subrev_f16: 3.0 - 1.0 = 2.0
        # Mutated: 1.0 - 3.0 = -2.0
        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_sub(s0, s1), TWO_F16)

    def test_mutation_signed_zero_blindness(self) -> None:
        # Mutation: ignoring signed zero difference (-0.0) - (+0.0) -> producing +0.0
        def mutated_zero_sub(s0: int, s1: int) -> int:
            res = ref_subrev_f16(s0, s1)
            if res == NEG_ZERO_F16:
                return POS_ZERO_F16
            return res

        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_zero_sub(POS_ZERO_F16, NEG_ZERO_F16), NEG_ZERO_F16)

    def test_mutation_infinity_cancellation(self) -> None:
        # Mutation: (+inf) - (+inf) produces 0.0 instead of NaN
        def mutated_inf_sub(s0: int, s1: int) -> int:
            if s0 == POS_INF_F16 and s1 == POS_INF_F16:
                return POS_ZERO_F16
            return ref_subrev_f16(s0, s1)

        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_inf_sub(POS_INF_F16, POS_INF_F16), CANONICAL_QNAN_F16)

    def test_mutation_clamp_nan_bug(self) -> None:
        # Mutation: clamping NaN returns NaN instead of +0.0
        def mutated_clamp_nan(s0: int, s1: int) -> int:
            return ref_subrev_f16(s0, s1, clamp=False)

        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_clamp_nan(CANONICAL_QNAN_F16, ONE_F16), POS_ZERO_F16)

    def test_mutation_exec_mask_leak(self) -> None:
        # Mutation: inactive lane leaks write to destination
        simd = SIMDWaveState(wave_size=32, exec_mask=0)  # all inactive
        canary = 0xBEEF
        simd.set_lane_vgpr(0, 0, pack_half(canary, canary))

        interp = SIMDVALUInterpreter()
        interp.step("v_subrev_f16_e32 v0, v1, v2", simd)
        # Canary must remain intact
        self.assertEqual(simd.get_lane_vgpr(0, 0), pack_half(canary, canary))

        # Mutated write unconditionally overwrites
        simd.set_lane_vgpr(0, 0, pack_half(ONE_F16, canary))
        with self.assertRaises(AssertionError):
            self.assertEqual(simd.get_lane_vgpr(0, 0), pack_half(canary, canary))

    def test_mutation_modifier_order(self) -> None:
        # Mutation: applying neg then abs (which makes -|x| always positive | -x | = +x)
        def mutated_mods(bits: int, abs_mod: bool, neg_mod: bool) -> int:
            res = bits
            if neg_mod:
                res ^= SIGN_MASK_F16
            if abs_mod:
                res &= ~SIGN_MASK_F16
            return res

        # For -|1.0|, correct is -1.0 (sign bit set)
        correct = apply_f16_input_mods(ONE_F16, abs_mod=True, neg_mod=True)
        mutated = mutated_mods(ONE_F16, abs_mod=True, neg_mod=True)
        self.assertEqual(correct, NEG_ONE_F16)
        self.assertNotEqual(correct, mutated)


if __name__ == "__main__":
    unittest.main()
