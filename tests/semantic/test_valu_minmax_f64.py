"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_min_f64 and v_max_f64.

Models IEEE 754-2008 / IEEE 754-2019 binary64 double-precision minNum/maxNum and AMD RDNA2 / GFX10
Vector Arithmetic Logic Unit (VALU) architectural semantics:
  - v_min_f64: D.d = min(S0.d, S1.d)
  - v_max_f64: D.d = max(S0.d, S1.d)

RDNA2 Double-Precision Floating-Point ISA Rules:
  1. Register Pair Allocation:
     Double-precision 64-bit values occupy aligned pairs of 32-bit vector registers:
     v[d:d+1] where low 32 bits reside in v[d] and high 32 bits in v[d+1].
  2. NaN Suppression (IEEE 754-2008 / 2019 minNum / maxNum):
     If one operand is NaN and the other is a valid floating-point number, the valid number is
     returned unchanged. If both operands are NaN, a quiet NaN is returned.
  3. Signed Zero Ordering (-0.0 < +0.0):
     RDNA2 orders -0.0 strictly less than +0.0:
       min(+0.0, -0.0) = -0.0, min(-0.0, +0.0) = -0.0
       max(+0.0, -0.0) = +0.0, max(-0.0, +0.0) = +0.0
  4. Infinities & Subnormals:
     -INF < all finite numbers < +INF.
     Subnormal double values are preserved without flushing to zero in IEEE mode.
  5. VOP3 Encoding Modifiers:
     Input modifiers: abs (|x|) clears sign bit 63; neg (-x) inverts sign bit 63.
     When both abs and neg are set, neg(abs(x)) clears then sets sign bit (forcing negative).
     Output modifier: clamp clamps final result to [0.0, 1.0]. Clamping negative values or -0.0
     yields +0.0. Clamping NaN returns NaN unchanged.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_MASK_F64: int = 0x8000000000000000
EXP_MASK_F64: int = 0x7FF0000000000000
MANT_MASK_F64: int = 0x000FFFFFFFFFFFFF
QUIET_NAN_BIT_F64: int = 0x0008000000000000

# Canonical IEEE 754 binary64 bit patterns
POS_ZERO_F64: int = 0x0000000000000000
NEG_ZERO_F64: int = 0x8000000000000000
POS_ONE_F64: int = 0x3FF0000000000000       # 1.0
NEG_ONE_F64: int = 0xBFF0000000000000       # -1.0
POS_TWO_F64: int = 0x4000000000000000       # 2.0
NEG_TWO_F64: int = 0xC000000000000000       # -2.0
HALF_F64: int = 0x3FE0000000000000          # 0.5
NEG_HALF_F64: int = 0xBFE0000000000000      # -0.5
POS_THREE_F64: int = 0x4008000000000000     # 3.0
POS_TEN_F64: int = 0x4024000000000000       # 10.0
NEG_TEN_F64: int = 0xC024000000000000       # -10.0

DBL_MIN: int = 0x0010000000000000           # Smallest positive normal: 2^-1022
NEG_DBL_MIN: int = 0x8010000000000000       # -2^-1022
DBL_MAX: int = 0x7FEFFFFFFFFFFFFF           # (2 - 2^-52) * 2^1023
NEG_DBL_MAX: int = 0xFFEFFFFFFFFFFFFF
POS_INF_F64: int = 0x7FF0000000000000       # +infinity
NEG_INF_F64: int = 0xFFF0000000000000       # -infinity
QNAN_F64: int = 0x7FF8000000000000          # Canonical quiet NaN
NEG_QNAN_F64: int = 0xFFF8000000000000      # Negative quiet NaN
QNAN_PAYLOAD_A: int = 0x7FF8000000000001    # Quiet NaN with payload A
QNAN_PAYLOAD_B: int = 0xFFF9000000000002    # Quiet NaN with payload B
SNAN_F64: int = 0x7FF0000000000001          # Signaling NaN (MSB of mantissa is 0, payload != 0)
SUBNORM_MIN_F64: int = 0x0000000000000001   # Smallest subnormal: 2^-1074
SUBNORM_MAX_F64: int = 0x000FFFFFFFFFFFFF   # Largest positive subnormal: (1 - 2^-52) * 2^-1022
NEG_SUBNORM_MIN_F64: int = 0x8000000000000001
NEG_SUBNORM_MAX_F64: int = 0x800FFFFFFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned word as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & SIGN_MASK_F64) else val


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def bits_to_f64(bits: int) -> float:
    """Decode 64-bit unsigned integer to IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    """Encode double float to IEEE 754 double-precision 64-bit unsigned integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return POS_INF_F64 if val > 0 else NEG_INF_F64


def is_nan_f64(bits: int) -> bool:
    """True if 64-bit integer represents an IEEE 754 double NaN."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == EXP_MASK_F64 and (bits & MANT_MASK_F64) != 0


def is_qnan_f64(bits: int) -> bool:
    """True if bits represent a quiet NaN (MSB of mantissa bit 51 is 1)."""
    bits &= MASK64
    return is_nan_f64(bits) and bool(bits & QUIET_NAN_BIT_F64)


def is_snan_f64(bits: int) -> bool:
    """True if bits represent a signaling NaN (MSB of mantissa bit 51 is 0)."""
    bits &= MASK64
    return is_nan_f64(bits) and not bool(bits & QUIET_NAN_BIT_F64)


def is_subnormal_f64(bits: int) -> bool:
    """True if bits represent an IEEE 754 double subnormal number."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == 0 and (bits & MANT_MASK_F64) != 0


def zero_ordered_key(v: float) -> tuple[float, float]:
    """Sort key ordering -0.0 strictly below +0.0."""
    return (v, math.copysign(1.0, v))


def ref_min_f64(s0: int, s1: int) -> int:
    """Pure bitwise reference oracle for RDNA2 v_min_f64.

    Rules:
      - If one operand is NaN, return the non-NaN operand unchanged.
      - If both operands are NaN, return canonical quiet NaN (0x7FF8000000000000).
      - If non-NaN values, order -0.0 < +0.0.
    """
    s0 &= MASK64
    s1 &= MASK64
    nan0 = is_nan_f64(s0)
    nan1 = is_nan_f64(s1)
    if nan0 and nan1:
        return QNAN_F64
    if nan0:
        return s1
    if nan1:
        return s0

    f0 = bits_to_f64(s0)
    f1 = bits_to_f64(s1)
    k0 = zero_ordered_key(f0)
    k1 = zero_ordered_key(f1)
    return s0 if k0 <= k1 else s1


def ref_max_f64(s0: int, s1: int) -> int:
    """Pure bitwise reference oracle for RDNA2 v_max_f64.

    Rules:
      - If one operand is NaN, return the non-NaN operand unchanged.
      - If both operands are NaN, return canonical quiet NaN (0x7FF8000000000000).
      - If non-NaN values, order -0.0 < +0.0.
    """
    s0 &= MASK64
    s1 &= MASK64
    nan0 = is_nan_f64(s0)
    nan1 = is_nan_f64(s1)
    if nan0 and nan1:
        return QNAN_F64
    if nan0:
        return s1
    if nan1:
        return s0

    f0 = bits_to_f64(s0)
    f1 = bits_to_f64(s1)
    k0 = zero_ordered_key(f0)
    k1 = zero_ordered_key(f1)
    return s0 if k0 >= k1 else s1


def apply_f64_modifiers(bits: int, neg: bool = False, abs_val: bool = False) -> int:
    """Applies VOP3 input modifiers: abs (|x|) first, then neg (-x)."""
    bits &= MASK64
    if abs_val:
        bits &= ~SIGN_MASK_F64
    if neg:
        bits ^= SIGN_MASK_F64
    return bits & MASK64


def apply_f64_clamp(bits: int) -> int:
    """Applies VOP3 clamp modifier: clamps result to [0.0, 1.0]."""
    bits &= MASK64
    if is_nan_f64(bits):
        return bits
    val = bits_to_f64(bits)
    if val < 0.0 or (val == 0.0 and (bits & SIGN_MASK_F64)):
        return POS_ZERO_F64
    if val > 1.0:
        return POS_ONE_F64
    return bits


def ref_min_f64_ext(
    s0: int,
    s1: int,
    neg0: bool = False,
    abs0: bool = False,
    neg1: bool = False,
    abs1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for v_min_f64 with input and output modifiers."""
    mod0 = apply_f64_modifiers(s0, neg=neg0, abs_val=abs0)
    mod1 = apply_f64_modifiers(s1, neg=neg1, abs_val=abs1)
    res = ref_min_f64(mod0, mod1)
    return apply_f64_clamp(res) if clamp else res


def ref_max_f64_ext(
    s0: int,
    s1: int,
    neg0: bool = False,
    abs0: bool = False,
    neg1: bool = False,
    abs1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for v_max_f64 with input and output modifiers."""
    mod0 = apply_f64_modifiers(s0, neg=neg0, abs_val=abs0)
    mod1 = apply_f64_modifiers(s1, neg=neg1, abs_val=abs1)
    res = ref_max_f64(mod0, mod1)
    return apply_f64_clamp(res) if clamp else res


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    """Register file domain."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    """Symbolic register operand representing an aligned 64-bit pair or 32-bit register."""
    kind: RegKind
    index: int
    count: int = 2

    def __post_init__(self) -> None:
        if self.index < 0 or self.index >= 256:
            raise ValueError(f"Register index out of range [0, 255]: {self.index}")
        if self.count not in (1, 2):
            raise ValueError(f"Unsupported register count (must be 1 or 2): {self.count}")
        if self.count == 2 and self.index + 1 >= 256:
            raise ValueError(f"64-bit register pair exceeds register file boundary: {self.index}")

    @property
    def is_vgpr(self) -> bool:
        return self.kind == RegKind.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.kind == RegKind.SGPR

    def format_asm(self) -> str:
        if self.count == 2:
            return f"{self.kind.value}[{self.index}:{self.index + 1}]"
        return f"{self.kind.value}{self.index}"


@dataclasses.dataclass(frozen=True)
class Immediate:
    """Symbolic 64-bit immediate operand (float or bit pattern)."""
    bits: int

    def __init__(self, val: int | float) -> None:
        if isinstance(val, float):
            b = f64_to_bits(val)
        else:
            b = val & MASK64
        object.__setattr__(self, "bits", b)

    @property
    def as_float(self) -> float:
        return bits_to_f64(self.bits)

    def format_asm(self) -> str:
        return f"{self.bits:#018x}"


@dataclasses.dataclass(frozen=True)
class OperandModifiers:
    """Input modifiers for VALU floating-point instructions (abs and neg)."""
    neg: bool = False
    abs_val: bool = False

    def apply(self, bits: int) -> int:
        return apply_f64_modifiers(bits, neg=self.neg, abs_val=self.abs_val)

    def format_asm(self, base_str: str) -> str:
        res = base_str
        if self.abs_val:
            res = f"|{res}|"
        if self.neg:
            res = f"-{res}"
        return res


@dataclasses.dataclass(frozen=True)
class InstructionModifiers:
    """Output modifiers for VALU instructions (clamp)."""
    clamp: bool = False

    def apply(self, bits: int) -> int:
        return apply_f64_clamp(bits) if self.clamp else (bits & MASK64)

    def format_asm(self) -> str:
        return " clamp" if self.clamp else ""


class Opcode(enum.Enum):
    """VALU 64-bit floating-point opcodes."""
    V_MIN_F64 = "v_min_f64"
    V_MAX_F64 = "v_max_f64"


class Encoding(enum.Enum):
    """Instruction encoding format."""
    VOP2 = "e32"
    VOP3 = "e64"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a double-precision VALU instruction."""
    opcode: Opcode
    dst: Register
    src0: Register | Immediate
    src1: Register | Immediate
    src0_mods: OperandModifiers = dataclasses.field(default_factory=OperandModifiers)
    src1_mods: OperandModifiers = dataclasses.field(default_factory=OperandModifiers)
    inst_mods: InstructionModifiers = dataclasses.field(default_factory=InstructionModifiers)
    encoding: Encoding = Encoding.VOP3

    def __post_init__(self) -> None:
        if not self.dst.is_vgpr:
            raise ValueError(f"VALU instruction destination must be a VGPR: {self.dst}")
        if self.dst.count != 2:
            raise ValueError(f"VALU f64 instruction destination must be a 64-bit pair (count=2): {self.dst}")
        # VOP2 (e32) encoding does not support input modifiers or clamp
        has_mods = self.src0_mods.neg or self.src0_mods.abs_val or self.src1_mods.neg or self.src1_mods.abs_val or self.inst_mods.clamp
        if self.encoding == Encoding.VOP2 and has_mods:
            raise ValueError("VOP2 encoding does not support input modifiers or clamp (requires VOP3/e64)")

    def to_assembly(self) -> str:
        s0_str = self.src0.format_asm()
        s1_str = self.src1.format_asm()
        s0_fmt = self.src0_mods.format_asm(s0_str)
        s1_fmt = self.src1_mods.format_asm(s1_str)
        dst_str = self.dst.format_asm()
        clamp_str = self.inst_mods.format_asm()
        enc_suffix = f"_{self.encoding.value}" if self.encoding == Encoding.VOP2 else ""
        return f"{self.opcode.value}{enc_suffix} {dst_str}, {s0_fmt}, {s1_fmt}{clamp_str}"


@dataclasses.dataclass
class Program:
    """Symbolic sequence of VALU instructions."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarVALUInterpreter:
    """Deterministic scalar interpreter for GFX10/RDNA2 VALU f64 instructions."""

    def __init__(self) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.trace: list[str] = []

    def read_reg_u32(self, kind: RegKind, idx: int) -> int:
        if idx < 0 or (kind == RegKind.VGPR and idx >= 256) or (kind == RegKind.SGPR and idx >= 106):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        return (self.vgprs[idx] if kind == RegKind.VGPR else self.sgprs[idx]) & MASK32

    def write_reg_u32(self, kind: RegKind, idx: int, val: int) -> None:
        if idx < 0 or (kind == RegKind.VGPR and idx >= 256) or (kind == RegKind.SGPR and idx >= 106):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[idx] = val32
        else:
            self.sgprs[idx] = val32

    def read_reg_u64(self, kind: RegKind, idx: int) -> int:
        lo = self.read_reg_u32(kind, idx)
        hi = self.read_reg_u32(kind, idx + 1)
        return join_dwords(lo, hi)

    def write_reg_u64(self, kind: RegKind, idx: int, val: int) -> None:
        lo, hi = split_dwords(val)
        self.write_reg_u32(kind, idx, lo)
        self.write_reg_u32(kind, idx + 1, hi)

    def read_operand(self, op: Register | Immediate, mods: OperandModifiers) -> int:
        if isinstance(op, Immediate):
            raw = op.bits
        elif isinstance(op, Register):
            raw = self.read_reg_u64(op.kind, op.index)
        else:
            raise TypeError(f"Unsupported operand type: {type(op)}")
        return mods.apply(raw)

    def step(self, inst: Instruction) -> int:
        """Executes a single VALU instruction deterministically."""
        # Latch all input operands before writing to destination (read-before-write semantics)
        s0_val = self.read_operand(inst.src0, inst.src0_mods)
        s1_val = self.read_operand(inst.src1, inst.src1_mods)

        if inst.opcode == Opcode.V_MIN_F64:
            res_raw = ref_min_f64(s0_val, s1_val)
        elif inst.opcode == Opcode.V_MAX_F64:
            res_raw = ref_max_f64(s0_val, s1_val)
        else:
            raise NotImplementedError(f"Unsupported opcode in scalar interpreter: {inst.opcode}")

        final_res = inst.inst_mods.apply(res_raw)
        self.write_reg_u64(inst.dst.kind, inst.dst.index, final_res)
        self.trace.append(f"{inst.to_assembly()} => {final_res:#018x}")
        return final_res

    def run_program(self, prog: Program) -> list[int]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


class TestValuMinMaxF64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_basic_float_min_max(self) -> None:
        self.assertEqual(ref_min_f64(POS_ONE_F64, POS_TWO_F64), POS_ONE_F64)
        self.assertEqual(ref_min_f64(POS_TWO_F64, POS_ONE_F64), POS_ONE_F64)
        self.assertEqual(ref_max_f64(POS_ONE_F64, POS_TWO_F64), POS_TWO_F64)
        self.assertEqual(ref_max_f64(POS_TWO_F64, POS_ONE_F64), POS_TWO_F64)
        self.assertEqual(ref_min_f64(NEG_TWO_F64, NEG_ONE_F64), NEG_TWO_F64)
        self.assertEqual(ref_max_f64(NEG_TWO_F64, NEG_ONE_F64), NEG_ONE_F64)

    def test_signed_zeros_ordering(self) -> None:
        self.assertEqual(ref_min_f64(POS_ZERO_F64, NEG_ZERO_F64), NEG_ZERO_F64)
        self.assertEqual(ref_min_f64(NEG_ZERO_F64, POS_ZERO_F64), NEG_ZERO_F64)
        self.assertEqual(ref_max_f64(POS_ZERO_F64, NEG_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(ref_max_f64(NEG_ZERO_F64, POS_ZERO_F64), POS_ZERO_F64)

    def test_infinities_ordering(self) -> None:
        self.assertEqual(ref_min_f64(NEG_INF_F64, POS_INF_F64), NEG_INF_F64)
        self.assertEqual(ref_max_f64(NEG_INF_F64, POS_INF_F64), POS_INF_F64)
        self.assertEqual(ref_min_f64(DBL_MAX, POS_INF_F64), DBL_MAX)
        self.assertEqual(ref_max_f64(DBL_MAX, POS_INF_F64), POS_INF_F64)
        self.assertEqual(ref_min_f64(NEG_DBL_MAX, NEG_INF_F64), NEG_INF_F64)
        self.assertEqual(ref_max_f64(NEG_DBL_MAX, NEG_INF_F64), NEG_DBL_MAX)

    def test_subnormals_preservation(self) -> None:
        self.assertTrue(is_subnormal_f64(SUBNORM_MIN_F64))
        self.assertTrue(is_subnormal_f64(SUBNORM_MAX_F64))
        self.assertEqual(ref_min_f64(SUBNORM_MIN_F64, SUBNORM_MAX_F64), SUBNORM_MIN_F64)
        self.assertEqual(ref_max_f64(SUBNORM_MIN_F64, SUBNORM_MAX_F64), SUBNORM_MAX_F64)
        self.assertEqual(ref_min_f64(POS_ZERO_F64, SUBNORM_MIN_F64), POS_ZERO_F64)
        self.assertEqual(ref_max_f64(POS_ZERO_F64, SUBNORM_MIN_F64), SUBNORM_MIN_F64)

    def test_nan_suppression_and_propagation(self) -> None:
        self.assertEqual(ref_min_f64(QNAN_F64, POS_ONE_F64), POS_ONE_F64)
        self.assertEqual(ref_min_f64(POS_ONE_F64, QNAN_F64), POS_ONE_F64)
        self.assertEqual(ref_max_f64(QNAN_F64, POS_ONE_F64), POS_ONE_F64)
        self.assertEqual(ref_max_f64(POS_ONE_F64, QNAN_F64), POS_ONE_F64)
        self.assertEqual(ref_min_f64(QNAN_PAYLOAD_A, QNAN_PAYLOAD_B), QNAN_F64)
        self.assertEqual(ref_max_f64(QNAN_PAYLOAD_A, QNAN_PAYLOAD_B), QNAN_F64)

    def test_modifiers_abs_neg(self) -> None:
        self.assertEqual(apply_f64_modifiers(NEG_ONE_F64, abs_val=True), POS_ONE_F64)
        self.assertEqual(apply_f64_modifiers(POS_ONE_F64, neg=True), NEG_ONE_F64)
        self.assertEqual(apply_f64_modifiers(POS_ONE_F64, neg=True, abs_val=True), NEG_ONE_F64)
        self.assertEqual(ref_min_f64_ext(POS_TWO_F64, POS_ONE_F64, neg0=True), NEG_TWO_F64)
        self.assertEqual(ref_max_f64_ext(NEG_TWO_F64, NEG_TEN_F64, abs0=True), POS_TWO_F64)

    def test_clamp_modifier(self) -> None:
        self.assertEqual(apply_f64_clamp(NEG_TWO_F64), POS_ZERO_F64)
        self.assertEqual(apply_f64_clamp(NEG_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(apply_f64_clamp(POS_TWO_F64), POS_ONE_F64)
        self.assertEqual(apply_f64_clamp(HALF_F64), HALF_F64)
        self.assertEqual(apply_f64_clamp(QNAN_F64), QNAN_F64)

    def test_split_join_dwords(self) -> None:
        lo, hi = split_dwords(POS_ONE_F64)
        self.assertEqual(lo, 0x00000000)
        self.assertEqual(hi, 0x3FF00000)
        self.assertEqual(join_dwords(lo, hi), POS_ONE_F64)


class TestValuMinMaxF64AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_construction_and_formatting(self) -> None:
        v0 = Register(RegKind.VGPR, 0, count=2)
        v2 = Register(RegKind.VGPR, 2, count=2)
        s4 = Register(RegKind.SGPR, 4, count=2)
        self.assertTrue(v0.is_vgpr)
        self.assertFalse(v0.is_sgpr)
        self.assertTrue(s4.is_sgpr)
        self.assertEqual(v0.format_asm(), "v[0:1]")
        self.assertEqual(v2.format_asm(), "v[2:3]")
        self.assertEqual(s4.format_asm(), "s[4:5]")

    def test_immediate_construction(self) -> None:
        imm_f = Immediate(1.0)
        imm_b = Immediate(0x7FF0000000000000)
        self.assertEqual(imm_f.bits, POS_ONE_F64)
        self.assertEqual(imm_f.as_float, 1.0)
        self.assertEqual(imm_b.bits, POS_INF_F64)

    def test_operand_modifiers_formatting(self) -> None:
        mods_none = OperandModifiers()
        mods_abs = OperandModifiers(abs_val=True)
        mods_neg = OperandModifiers(neg=True)
        mods_both = OperandModifiers(neg=True, abs_val=True)

        self.assertEqual(mods_none.format_asm("v[0:1]"), "v[0:1]")
        self.assertEqual(mods_abs.format_asm("v[0:1]"), "|v[0:1]|")
        self.assertEqual(mods_neg.format_asm("v[0:1]"), "-v[0:1]")
        self.assertEqual(mods_both.format_asm("v[0:1]"), "-|v[0:1]|")

    def test_instruction_formatting(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_MIN_F64,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2, count=2),
            src1=Register(RegKind.VGPR, 4, count=2),
            src0_mods=OperandModifiers(neg=True, abs_val=True),
            inst_mods=InstructionModifiers(clamp=True),
        )
        self.assertEqual(inst.to_assembly(), "v_min_f64 v[0:1], -|v[2:3]|, v[4:5] clamp")

        inst_e32 = Instruction(
            opcode=Opcode.V_MAX_F64,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2, count=2),
            src1=Register(RegKind.VGPR, 4, count=2),
            encoding=Encoding.VOP2,
        )
        self.assertEqual(inst_e32.to_assembly(), "v_max_f64_e32 v[0:1], v[2:3], v[4:5]")

    def test_instruction_validation(self) -> None:
        # Invalid destination kind (must be VGPR)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_MIN_F64,
                dst=Register(RegKind.SGPR, 0, count=2),
                src0=Register(RegKind.VGPR, 2, count=2),
                src1=Register(RegKind.VGPR, 4, count=2),
            )

        # Invalid destination count (must be 2 for 64-bit float)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_MIN_F64,
                dst=Register(RegKind.VGPR, 0, count=1),
                src0=Register(RegKind.VGPR, 2, count=2),
                src1=Register(RegKind.VGPR, 4, count=2),
            )

        # VOP2 cannot take modifiers
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_MIN_F64,
                dst=Register(RegKind.VGPR, 0, count=2),
                src0=Register(RegKind.VGPR, 2, count=2),
                src1=Register(RegKind.VGPR, 4, count=2),
                src0_mods=OperandModifiers(neg=True),
                encoding=Encoding.VOP2,
            )

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.V_MIN_F64,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2, count=2),
            src1=Register(RegKind.VGPR, 4, count=2),
        ))
        prog.append(Instruction(
            opcode=Opcode.V_MAX_F64,
            dst=Register(RegKind.VGPR, 6, count=2),
            src0=Register(RegKind.VGPR, 0, count=2),
            src1=Immediate(1.0),
            inst_mods=InstructionModifiers(clamp=True),
        ))
        expected = (
            "v_min_f64 v[0:1], v[2:3], v[4:5]\n"
            f"v_max_f64 v[6:7], v[0:1], {POS_ONE_F64:#018x} clamp"
        )
        self.assertEqual(prog.to_assembly(), expected)


class TestValuMinMaxF64ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar VALU instruction interpreter."""

    def test_scalar_register_read_write(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u64(RegKind.VGPR, 0, POS_ONE_F64)
        interp.write_reg_u64(RegKind.SGPR, 4, NEG_TWO_F64)

        self.assertEqual(interp.read_reg_u64(RegKind.VGPR, 0), POS_ONE_F64)
        self.assertEqual(interp.read_reg_u64(RegKind.SGPR, 4), NEG_TWO_F64)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0x00000000)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 1), 0x3FF00000)

    def test_scalar_min_execution_vgpr_sources(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u64(RegKind.VGPR, 2, POS_TWO_F64)
        interp.write_reg_u64(RegKind.VGPR, 4, POS_ONE_F64)

        inst = Instruction(
            opcode=Opcode.V_MIN_F64,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2, count=2),
            src1=Register(RegKind.VGPR, 4, count=2),
        )
        res = interp.step(inst)
        self.assertEqual(res, POS_ONE_F64)
        self.assertEqual(interp.read_reg_u64(RegKind.VGPR, 0), POS_ONE_F64)

    def test_scalar_max_execution_sgpr_sources(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u64(RegKind.VGPR, 2, NEG_TWO_F64)
        interp.write_reg_u64(RegKind.SGPR, 0, NEG_ONE_F64)

        inst = Instruction(
            opcode=Opcode.V_MAX_F64,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2, count=2),
            src1=Register(RegKind.SGPR, 0, count=2),
        )
        res = interp.step(inst)
        self.assertEqual(res, NEG_ONE_F64)
        self.assertEqual(interp.read_reg_u64(RegKind.VGPR, 0), NEG_ONE_F64)

    def test_scalar_immediate_operand(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u64(RegKind.VGPR, 2, POS_TWO_F64)

        inst = Instruction(
            opcode=Opcode.V_MIN_F64,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2, count=2),
            src1=Immediate(3.0),
        )
        res = interp.step(inst)
        self.assertEqual(res, POS_TWO_F64)

    def test_scalar_modifiers_abs_neg_clamp(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u64(RegKind.VGPR, 2, NEG_TWO_F64)
        interp.write_reg_u64(RegKind.VGPR, 4, POS_THREE_F64)

        # min(|-2.0|, 3.0) clamp -> min(2.0, 3.0) = 2.0 clamped to [0,1] = 1.0
        inst = Instruction(
            opcode=Opcode.V_MIN_F64,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2, count=2),
            src1=Register(RegKind.VGPR, 4, count=2),
            src0_mods=OperandModifiers(abs_val=True),
            inst_mods=InstructionModifiers(clamp=True),
        )
        res = interp.step(inst)
        self.assertEqual(res, POS_ONE_F64)

    def test_scalar_program_execution_chain(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u64(RegKind.VGPR, 2, POS_TEN_F64)     # 10.0
        interp.write_reg_u64(RegKind.VGPR, 4, POS_TWO_F64)     # 2.0
        interp.write_reg_u64(RegKind.VGPR, 6, HALF_F64)        # 0.5

        prog = Program()
        # v[0:1] = min(v[2:3], v[4:5]) = min(10.0, 2.0) = 2.0
        prog.append(Instruction(
            opcode=Opcode.V_MIN_F64,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2, count=2),
            src1=Register(RegKind.VGPR, 4, count=2),
        ))
        # v[8:9] = max(v[0:1], v[6:7]) = max(2.0, 0.5) = 2.0
        prog.append(Instruction(
            opcode=Opcode.V_MAX_F64,
            dst=Register(RegKind.VGPR, 8, count=2),
            src0=Register(RegKind.VGPR, 0, count=2),
            src1=Register(RegKind.VGPR, 6, count=2),
        ))

        results = interp.run_program(prog)
        self.assertEqual(results, [POS_TWO_F64, POS_TWO_F64])
        self.assertEqual(interp.read_reg_u64(RegKind.VGPR, 0), POS_TWO_F64)
        self.assertEqual(interp.read_reg_u64(RegKind.VGPR, 8), POS_TWO_F64)
        self.assertEqual(len(interp.trace), 2)


if __name__ == "__main__":
    unittest.main()
