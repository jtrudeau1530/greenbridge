from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

# ==============================================================================
# IEEE 754 binary16 (Half-Precision) Architectural Constants
# ==============================================================================

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF

SIGN_MASK16: int = 0x8000
EXP_MASK16: int = 0x7C00
FRAC_MASK16: int = 0x03FF
QNAN_BIT16: int = 0x0200

# Canonical and special binary16 values
POS_ZERO: int = 0x0000          # +0.0
NEG_ZERO: int = 0x8000          # -0.0
POS_ONE: int = 0x3C00           # +1.0
NEG_ONE: int = 0xBC00           # -1.0
POS_HALF: int = 0x3800          # +0.5
NEG_HALF: int = 0xB800          # -0.5
POS_TWO: int = 0x4000           # +2.0
NEG_TWO: int = 0xC000           # -2.0
POS_INF: int = 0x7C00           # +inf
NEG_INF: int = 0xFC00           # -inf
QNAN_CANONICAL: int = 0x7E00    # Canonical quiet NaN
SNAN_DEFAULT: int = 0x7D00      # Signaling NaN default
NEG_QNAN: int = 0xFE00          # Negative quiet NaN

# Subnormal (denormal) binary16 constants
POS_MIN_SUBNORM: int = 0x0001   # +2^-24, smallest positive subnormal
POS_MAX_SUBNORM: int = 0x03FF   # largest positive subnormal
NEG_MIN_SUBNORM: int = 0x8001   # -2^-24, smallest negative subnormal
NEG_MAX_SUBNORM: int = 0x83FF   # largest negative subnormal

# Normal binary16 boundaries
POS_MIN_NORMAL: int = 0x0400    # +2^-14, smallest positive normal
POS_MAX_NORMAL: int = 0x7BFF    # +65504.0, largest finite binary16
NEG_MAX_NORMAL: int = 0xFBFF    # -65504.0, largest negative finite binary16


# ==============================================================================
# Bit Manipulation and Conversion Utilities
# ==============================================================================

def f16_to_bits(val: float) -> int:
    """Convert a Python float to an IEEE 754 binary16 16-bit integer."""
    if math.isnan(val):
        return QNAN_CANONICAL
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def bits_to_f16(bits: int) -> float:
    """Convert an IEEE 754 binary16 16-bit integer to a Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def unpack_pk_f16(packed: int) -> Tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit half words (lo, hi)."""
    lo = packed & MASK16
    hi = (packed >> 16) & MASK16
    return lo, hi


def pack_pk_f16(lo: int, hi: int) -> int:
    """Pack two 16-bit half words (lo, hi) into a single 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def is_nan16(bits: int) -> bool:
    """Check if 16-bit word represents an IEEE 754 binary16 NaN."""
    return ((bits & EXP_MASK16) == EXP_MASK16) and ((bits & FRAC_MASK16) != 0)


def is_qnan16(bits: int) -> bool:
    """Check if 16-bit word represents a Quiet NaN."""
    return is_nan16(bits) and bool(bits & QNAN_BIT16)


def is_snan16(bits: int) -> bool:
    """Check if 16-bit word represents a Signaling NaN."""
    return is_nan16(bits) and not bool(bits & QNAN_BIT16)


def is_inf16(bits: int) -> bool:
    """Check if 16-bit word represents positive or negative infinity."""
    return ((bits & EXP_MASK16) == EXP_MASK16) and ((bits & FRAC_MASK16) == 0)


def is_subnormal16(bits: int) -> bool:
    """Check if 16-bit word represents a subnormal (denormal) half-float."""
    return ((bits & EXP_MASK16) == 0) and ((bits & FRAC_MASK16) != 0)


def is_zero16(bits: int) -> bool:
    """Check if 16-bit word represents positive or negative zero."""
    return (bits & ~SIGN_MASK16) == 0


def apply_f16_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply half-float input modifiers (abs, neg)."""
    res = bits & MASK16
    if abs_mod:
        res &= ~SIGN_MASK16
    if neg_mod:
        res ^= SIGN_MASK16
    return res


# ==============================================================================
# Reference Mathematical Oracles for Scalar & Packed f16 Min/Max
# ==============================================================================

def clamp_f16(bits: int) -> int:
    """Clamp a 16-bit half-float to [0.0, 1.0]. NaN clamps to +0.0."""
    if is_nan16(bits):
        return POS_ZERO
    f = bits_to_f16(bits)
    if f < 0.0 or bits == NEG_ZERO:
        return POS_ZERO
    if f > 1.0 or bits == POS_INF:
        return POS_ONE
    return bits & MASK16


def oracle_min_f16(
    s0: int,
    s1: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for RDNA2 scalar v_min_f16 (IEEE 754-2008 / GLSL NMin).

    Rules:
    - NaN suppression: if one operand is NaN and the other is a number, return the number.
    - If both operands are NaN, return quiet NaN (s0 quietened if NaN).
    - Signed zero tie-breaking: -0.0 < +0.0 (min(-0.0, +0.0) == -0.0).
    - Subnormal numbers are preserved.
    - If clamp=True, output clamped to [0.0, 1.0].
    """
    a = apply_f16_modifiers(s0, abs0, neg0)
    b = apply_f16_modifiers(s1, abs1, neg1)

    a_nan = is_nan16(a)
    b_nan = is_nan16(b)

    if a_nan and b_nan:
        res = a | QNAN_BIT16
    elif a_nan:
        res = b
    elif b_nan:
        res = a
    else:
        a_is_neg_zero = (a == NEG_ZERO)
        b_is_neg_zero = (b == NEG_ZERO)
        a_is_pos_zero = (a == POS_ZERO)
        b_is_pos_zero = (b == POS_ZERO)

        if (a_is_neg_zero and b_is_pos_zero) or (a_is_pos_zero and b_is_neg_zero):
            res = NEG_ZERO
        elif (a_is_neg_zero and b_is_neg_zero):
            res = NEG_ZERO
        elif (a_is_pos_zero and b_is_pos_zero):
            res = POS_ZERO
        else:
            fa = bits_to_f16(a)
            fb = bits_to_f16(b)
            if fa < fb:
                res = a
            elif fb < fa:
                res = b
            else:
                # Equal finite values: preserve sign or default to a
                res = a if (a & SIGN_MASK16) else b

    if clamp:
        res = clamp_f16(res)
    return res & MASK16


def oracle_max_f16(
    s0: int,
    s1: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for RDNA2 scalar v_max_f16 (IEEE 754-2008 / GLSL NMax).

    Rules:
    - NaN suppression: if one operand is NaN and the other is a number, return the number.
    - If both operands are NaN, return quiet NaN (s0 quietened if NaN).
    - Signed zero tie-breaking: -0.0 < +0.0 (max(-0.0, +0.0) == +0.0).
    - Subnormal numbers are preserved.
    - If clamp=True, output clamped to [0.0, 1.0].
    """
    a = apply_f16_modifiers(s0, abs0, neg0)
    b = apply_f16_modifiers(s1, abs1, neg1)

    a_nan = is_nan16(a)
    b_nan = is_nan16(b)

    if a_nan and b_nan:
        res = a | QNAN_BIT16
    elif a_nan:
        res = b
    elif b_nan:
        res = a
    else:
        a_is_neg_zero = (a == NEG_ZERO)
        b_is_neg_zero = (b == NEG_ZERO)
        a_is_pos_zero = (a == POS_ZERO)
        b_is_pos_zero = (b == POS_ZERO)

        if (a_is_neg_zero and b_is_pos_zero) or (a_is_pos_zero and b_is_neg_zero):
            res = POS_ZERO
        elif (a_is_neg_zero and b_is_neg_zero):
            res = NEG_ZERO
        elif (a_is_pos_zero and b_is_pos_zero):
            res = POS_ZERO
        else:
            fa = bits_to_f16(a)
            fb = bits_to_f16(b)
            if fa > fb:
                res = a
            elif fb > fa:
                res = b
            else:
                # Equal finite values: prefer positive sign
                res = b if not (b & SIGN_MASK16) else a

    if clamp:
        res = clamp_f16(res)
    return res & MASK16


def oracle_pk_min_f16(
    s0_u32: int,
    s1_u32: int,
    *,
    op_sel: Tuple[int, int, int, int] = (0, 0, 1, 1),
    abs0_lo: bool = False,
    neg0_lo: bool = False,
    abs1_lo: bool = False,
    neg1_lo: bool = False,
    abs0_hi: bool = False,
    neg0_hi: bool = False,
    abs1_hi: bool = False,
    neg1_hi: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for RDNA2 packed v_pk_min_f16 (VOP3P).

    Selects halves based on op_sel:
    - op_sel[0]: src0 half for dest.lo (0=lo, 1=hi)
    - op_sel[1]: src1 half for dest.lo (0=lo, 1=hi)
    - op_sel[2]: src0 half for dest.hi (0=lo, 1=hi)
    - op_sel[3]: src1 half for dest.hi (0=lo, 1=hi)
    Computes min for lo and hi independently, packs into 32-bit dword.
    """
    s0_halves = unpack_pk_f16(s0_u32)
    s1_halves = unpack_pk_f16(s1_u32)

    in0_lo = s0_halves[op_sel[0]]
    in1_lo = s1_halves[op_sel[1]]
    in0_hi = s0_halves[op_sel[2]]
    in1_hi = s1_halves[op_sel[3]]

    out_lo = oracle_min_f16(
        in0_lo, in1_lo,
        abs0=abs0_lo, neg0=neg0_lo, abs1=abs1_lo, neg1=neg1_lo,
        clamp=clamp,
    )
    out_hi = oracle_min_f16(
        in0_hi, in1_hi,
        abs0=abs0_hi, neg0=neg0_hi, abs1=abs1_hi, neg1=neg1_hi,
        clamp=clamp,
    )

    return pack_pk_f16(out_lo, out_hi)


def oracle_pk_max_f16(
    s0_u32: int,
    s1_u32: int,
    *,
    op_sel: Tuple[int, int, int, int] = (0, 0, 1, 1),
    abs0_lo: bool = False,
    neg0_lo: bool = False,
    abs1_lo: bool = False,
    neg1_lo: bool = False,
    abs0_hi: bool = False,
    neg0_hi: bool = False,
    abs1_hi: bool = False,
    neg1_hi: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for RDNA2 packed v_pk_max_f16 (VOP3P).

    Selects halves based on op_sel, computes max for lo and hi independently.
    """
    s0_halves = unpack_pk_f16(s0_u32)
    s1_halves = unpack_pk_f16(s1_u32)

    in0_lo = s0_halves[op_sel[0]]
    in1_lo = s1_halves[op_sel[1]]
    in0_hi = s0_halves[op_sel[2]]
    in1_hi = s1_halves[op_sel[3]]

    out_lo = oracle_max_f16(
        in0_lo, in1_lo,
        abs0=abs0_lo, neg0=neg0_lo, abs1=abs1_lo, neg1=neg1_lo,
        clamp=clamp,
    )
    out_hi = oracle_max_f16(
        in0_hi, in1_hi,
        abs0=abs0_hi, neg0=neg0_hi, abs1=abs1_hi, neg1=neg1_hi,
        clamp=clamp,
    )

    return pack_pk_f16(out_lo, out_hi)


# ==============================================================================
# Commit 1 Unit Tests
# ==============================================================================

class TestF16ModelAndOracle(unittest.TestCase):
    """Unit tests for half-precision bit representation, predicates, and reference oracles."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 65504.0, -65504.0]
        for val in values:
            with self.subTest(val=val):
                bits = f16_to_bits(val)
                recovered = bits_to_f16(bits)
                self.assertEqual(val, recovered)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_word_packing_and_unpacking(self) -> None:
        lo = 0x1234
        hi = 0x5678
        packed = pack_pk_f16(lo, hi)
        self.assertEqual(packed, 0x56781234)
        rec_lo, rec_hi = unpack_pk_f16(packed)
        self.assertEqual(rec_lo, lo)
        self.assertEqual(rec_hi, hi)

    def test_classification_predicates(self) -> None:
        self.assertTrue(is_zero16(POS_ZERO))
        self.assertTrue(is_zero16(NEG_ZERO))
        self.assertFalse(is_zero16(POS_ONE))

        self.assertTrue(is_inf16(POS_INF))
        self.assertTrue(is_inf16(NEG_INF))
        self.assertFalse(is_inf16(POS_ZERO))
        self.assertFalse(is_inf16(QNAN_CANONICAL))

        self.assertTrue(is_nan16(QNAN_CANONICAL))
        self.assertTrue(is_qnan16(QNAN_CANONICAL))
        self.assertFalse(is_snan16(QNAN_CANONICAL))

        self.assertTrue(is_nan16(SNAN_DEFAULT))
        self.assertTrue(is_snan16(SNAN_DEFAULT))
        self.assertFalse(is_qnan16(SNAN_DEFAULT))

        self.assertTrue(is_subnormal16(POS_MIN_SUBNORM))
        self.assertTrue(is_subnormal16(POS_MAX_SUBNORM))
        self.assertFalse(is_subnormal16(POS_ZERO))
        self.assertFalse(is_subnormal16(POS_MIN_NORMAL))

    def test_exact_finite_min_max(self) -> None:
        # min(1.0, 2.0) == 1.0, max(1.0, 2.0) == 2.0
        self.assertEqual(oracle_min_f16(POS_ONE, POS_TWO), POS_ONE)
        self.assertEqual(oracle_max_f16(POS_ONE, POS_TWO), POS_TWO)

        # min(-1.0, 0.5) == -1.0, max(-1.0, 0.5) == 0.5
        self.assertEqual(oracle_min_f16(NEG_ONE, POS_HALF), NEG_ONE)
        self.assertEqual(oracle_max_f16(NEG_ONE, POS_HALF), POS_HALF)

    def test_signed_zero_tie_breaking(self) -> None:
        # min(-0.0, +0.0) == -0.0
        self.assertEqual(oracle_min_f16(NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(oracle_min_f16(POS_ZERO, NEG_ZERO), NEG_ZERO)

        # max(-0.0, +0.0) == +0.0
        self.assertEqual(oracle_max_f16(NEG_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_max_f16(POS_ZERO, NEG_ZERO), POS_ZERO)

        # min(-0.0, -0.0) == -0.0; max(+0.0, +0.0) == +0.0
        self.assertEqual(oracle_min_f16(NEG_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(oracle_max_f16(POS_ZERO, POS_ZERO), POS_ZERO)

    def test_nan_suppression_and_propagation(self) -> None:
        # If one operand is NaN, non-NaN operand is selected
        self.assertEqual(oracle_min_f16(QNAN_CANONICAL, POS_TWO), POS_TWO)
        self.assertEqual(oracle_min_f16(POS_TWO, QNAN_CANONICAL), POS_TWO)
        self.assertEqual(oracle_max_f16(QNAN_CANONICAL, POS_TWO), POS_TWO)
        self.assertEqual(oracle_max_f16(POS_TWO, QNAN_CANONICAL), POS_TWO)

        # If both operands are NaN, quiet NaN is returned
        both_min = oracle_min_f16(QNAN_CANONICAL, SNAN_DEFAULT)
        self.assertTrue(is_qnan16(both_min))
        both_max = oracle_max_f16(SNAN_DEFAULT, QNAN_CANONICAL)
        self.assertTrue(is_qnan16(both_max))

    def test_modifiers_and_clamp(self) -> None:
        # abs modifier: min(|-2.0|, 1.0) == min(2.0, 1.0) == 1.0
        res_abs = oracle_min_f16(NEG_TWO, POS_ONE, abs0=True)
        self.assertEqual(res_abs, POS_ONE)

        # neg modifier: max(-(2.0), -1.0) == max(-2.0, -1.0) == -1.0
        res_neg = oracle_max_f16(POS_TWO, NEG_ONE, neg0=True)
        self.assertEqual(res_neg, NEG_ONE)

        # clamp modifier: clamps to [0.0, 1.0]
        # max(2.0, 0.5) clamped -> 1.0
        self.assertEqual(oracle_max_f16(POS_TWO, POS_HALF, clamp=True), POS_ONE)
        # min(-1.0, -2.0) clamped -> +0.0
        self.assertEqual(oracle_min_f16(NEG_ONE, NEG_TWO, clamp=True), POS_ZERO)

    def test_packed_oracles_basic(self) -> None:
        # pack: s0 = (hi=2.0, lo=-1.0), s1 = (hi=0.5, lo=1.0)
        s0 = pack_pk_f16(NEG_ONE, POS_TWO)
        s1 = pack_pk_f16(POS_ONE, POS_HALF)

        # v_pk_min_f16:
        # lo: min(-1.0, 1.0) = -1.0
        # hi: min(2.0, 0.5) = 0.5
        pk_min = oracle_pk_min_f16(s0, s1)
        exp_min = pack_pk_f16(NEG_ONE, POS_HALF)
        self.assertEqual(pk_min, exp_min)

        # v_pk_max_f16:
        # lo: max(-1.0, 1.0) = 1.0
        # hi: max(2.0, 0.5) = 2.0
        pk_max = oracle_pk_max_f16(s0, s1)
        exp_max = pack_pk_f16(POS_ONE, POS_TWO)
        self.assertEqual(pk_max, exp_max)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegKind(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Reg:
    """Represents a 32-bit architectural register (VGPR or SGPR)."""
    index: int
    kind: RegKind = RegKind.VGPR

    def __post_init__(self) -> None:
        if self.kind == RegKind.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index {self.index} out of range [0, 255]")
        elif self.kind == RegKind.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index {self.index} out of range [0, 105]")
        else:
            raise TypeError(f"Unknown RegKind: {self.kind}")

    @property
    def name(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class Literal32:
    """Represents a 32-bit packed literal or inline constant."""
    bits: int

    def __post_init__(self) -> None:
        if not (0 <= self.bits <= 0xFFFF_FFFF):
            raise ValueError(f"Literal32 bits {self.bits:#x} out of range [0, 0xFFFFFFFF]")

    @property
    def lo_bits(self) -> int:
        return self.bits & MASK16

    @property
    def hi_bits(self) -> int:
        return (self.bits >> 16) & MASK16

    @property
    def name(self) -> str:
        return f"{self.bits:#010x}"


class Opcode(Enum):
    V_PK_MIN_F16 = "v_pk_min_f16"
    V_PK_MAX_F16 = "v_pk_max_f16"


@dataclass(frozen=True)
class VOP3POperand:
    """VOP3P packed operand with per-half input modifiers and op_sel."""
    source: Union[Reg, Literal32]
    neg_lo: bool = False
    neg_hi: bool = False
    abs_lo: bool = False
    abs_hi: bool = False
    op_sel: Tuple[int, int] = (0, 1)  # (lo_select, hi_select), where 0=lo, 1=hi

    def __post_init__(self) -> None:
        if self.op_sel not in ((0, 0), (0, 1), (1, 0), (1, 1)):
            raise ValueError(f"Invalid op_sel {self.op_sel}, elements must be 0 or 1")

    def disassemble(self) -> str:
        s = self.source.name
        mods = []
        if self.abs_lo and self.abs_hi:
            s = f"|{s}|"
        elif self.abs_lo:
            mods.append("abs_lo")
        elif self.abs_hi:
            mods.append("abs_hi")

        if self.neg_lo and self.neg_hi:
            s = f"-{s}"
        elif self.neg_lo:
            mods.append("neg_lo")
        elif self.neg_hi:
            mods.append("neg_hi")

        if self.op_sel != (0, 1):
            mods.append(f"op_sel:[{self.op_sel[0]},{self.op_sel[1]}]")

        if mods:
            return f"{s} ({' '.join(mods)})"
        return s


@dataclass(frozen=True)
class VPkMinMaxF16Instruction:
    """Symbolic AST node for RDNA2 v_pk_min_f16 and v_pk_max_f16 instructions."""
    opcode: Opcode
    dst: Reg
    src0: VOP3POperand
    src1: VOP3POperand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.kind != RegKind.VGPR:
            raise TypeError(f"Destination of VALU packed instruction must be VGPR, got {self.dst.kind}")

    def disassemble(self) -> str:
        s = f"{self.opcode.value} {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}"
        if self.clamp:
            s += " clamp"
        return s


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_reg_validation_and_naming(self) -> None:
        v0 = Reg(0, RegKind.VGPR)
        self.assertEqual(v0.name, "v0")
        v255 = Reg(255, RegKind.VGPR)
        self.assertEqual(v255.name, "v255")

        s0 = Reg(0, RegKind.SGPR)
        self.assertEqual(s0.name, "s0")
        s105 = Reg(105, RegKind.SGPR)
        self.assertEqual(s105.name, "s105")

        with self.assertRaises(ValueError):
            Reg(256, RegKind.VGPR)
        with self.assertRaises(ValueError):
            Reg(-1, RegKind.VGPR)
        with self.assertRaises(ValueError):
            Reg(106, RegKind.SGPR)

    def test_literal32_representation(self) -> None:
        lit = Literal32(0x3C004000)
        self.assertEqual(lit.lo_bits, 0x4000)
        self.assertEqual(lit.hi_bits, 0x3C00)
        self.assertEqual(lit.name, "0x3c004000")

        with self.assertRaises(ValueError):
            Literal32(0x1_0000_0000)
        with self.assertRaises(ValueError):
            Literal32(-1)

    def test_operand_disassembly_modifiers(self) -> None:
        v1 = Reg(1, RegKind.VGPR)
        op_plain = VOP3POperand(v1)
        self.assertEqual(op_plain.disassemble(), "v1")

        op_both_abs = VOP3POperand(v1, abs_lo=True, abs_hi=True)
        self.assertEqual(op_both_abs.disassemble(), "|v1|")

        op_both_neg = VOP3POperand(v1, neg_lo=True, neg_hi=True)
        self.assertEqual(op_both_neg.disassemble(), "-v1")

        op_custom = VOP3POperand(v1, neg_lo=True, op_sel=(1, 0))
        self.assertIn("neg_lo", op_custom.disassemble())
        self.assertIn("op_sel:[1,0]", op_custom.disassemble())

        with self.assertRaises(ValueError):
            VOP3POperand(v1, op_sel=(0, 2))

    def test_instruction_construction_and_validation(self) -> None:
        dst = Reg(0, RegKind.VGPR)
        src0 = VOP3POperand(Reg(1, RegKind.VGPR))
        src1 = VOP3POperand(Reg(2, RegKind.SGPR))

        inst_min = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MIN_F16,
            dst=dst,
            src0=src0,
            src1=src1,
            clamp=False,
        )
        self.assertEqual(inst_min.disassemble(), "v_pk_min_f16 v0, v1, s2")

        inst_max_clamp = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MAX_F16,
            dst=dst,
            src0=src0,
            src1=src1,
            clamp=True,
        )
        self.assertEqual(inst_max_clamp.disassemble(), "v_pk_max_f16 v0, v1, s2 clamp")

        # Destination must be VGPR
        with self.assertRaises(TypeError):
            VPkMinMaxF16Instruction(
                opcode=Opcode.V_PK_MIN_F16,
                dst=Reg(0, RegKind.SGPR),
                src0=src0,
                src1=src1,
            )

    def test_structural_equality(self) -> None:
        dst = Reg(4, RegKind.VGPR)
        src0 = VOP3POperand(Reg(5, RegKind.VGPR))
        src1 = VOP3POperand(Literal32(0x3C003C00))
        i1 = VPkMinMaxF16Instruction(Opcode.V_PK_MIN_F16, dst, src0, src1, clamp=True)
        i2 = VPkMinMaxF16Instruction(Opcode.V_PK_MIN_F16, dst, src0, src1, clamp=True)
        self.assertEqual(i1, i2)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for single-lane scalar interpretation."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)
    vcc: int = 0

    def get_reg(self, reg: Reg) -> int:
        if reg.kind == RegKind.VGPR:
            return self.vgprs[reg.index] & MASK32
        elif reg.kind == RegKind.SGPR:
            return self.sgprs[reg.index] & MASK32
        raise TypeError(f"Unknown RegKind: {reg.kind}")

    def set_reg(self, reg: Reg, val: int) -> None:
        if reg.kind == RegKind.VGPR:
            self.vgprs[reg.index] = val & MASK32
        elif reg.kind == RegKind.SGPR:
            self.sgprs[reg.index] = val & MASK32
        else:
            raise TypeError(f"Unknown RegKind: {reg.kind}")


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for packed half-precision min/max."""

    def resolve_operand(self, op: VOP3POperand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg):
            return state.get_reg(op.source)
        elif isinstance(op.source, Literal32):
            return op.source.bits
        raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VPkMinMaxF16Instruction, state: ScalarMachineState) -> int:
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)

        # Build op_sel tuple: (s0.lo_sel, s1.lo_sel, s0.hi_sel, s1.hi_sel)
        op_sel = (
            instr.src0.op_sel[0],
            instr.src1.op_sel[0],
            instr.src0.op_sel[1],
            instr.src1.op_sel[1],
        )

        if instr.opcode == Opcode.V_PK_MIN_F16:
            res = oracle_pk_min_f16(
                s0_raw, s1_raw,
                op_sel=op_sel,
                abs0_lo=instr.src0.abs_lo, neg0_lo=instr.src0.neg_lo,
                abs1_lo=instr.src1.abs_lo, neg1_lo=instr.src1.neg_lo,
                abs0_hi=instr.src0.abs_hi, neg0_hi=instr.src0.neg_hi,
                abs1_hi=instr.src1.abs_hi, neg1_hi=instr.src1.neg_hi,
                clamp=instr.clamp,
            )
        elif instr.opcode == Opcode.V_PK_MAX_F16:
            res = oracle_pk_max_f16(
                s0_raw, s1_raw,
                op_sel=op_sel,
                abs0_lo=instr.src0.abs_lo, neg0_lo=instr.src0.neg_lo,
                abs1_lo=instr.src1.abs_lo, neg1_lo=instr.src1.neg_lo,
                abs0_hi=instr.src0.abs_hi, neg0_hi=instr.src0.neg_hi,
                abs1_hi=instr.src1.abs_hi, neg1_hi=instr.src1.neg_hi,
                clamp=instr.clamp,
            )
        else:
            raise ValueError(f"Unsupported opcode: {instr.opcode}")

        state.set_reg(instr.dst, res)
        return res


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_scalar_vgpr_vgpr_pk_min(self) -> None:
        # v1: (hi=2.0, lo=-1.0), v2: (hi=0.5, lo=1.0)
        self.state.set_reg(Reg(1, RegKind.VGPR), pack_pk_f16(NEG_ONE, POS_TWO))
        self.state.set_reg(Reg(2, RegKind.VGPR), pack_pk_f16(POS_ONE, POS_HALF))

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MIN_F16,
            dst=Reg(0, RegKind.VGPR),
            src0=VOP3POperand(Reg(1, RegKind.VGPR)),
            src1=VOP3POperand(Reg(2, RegKind.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        # min: lo = min(-1.0, 1.0) = -1.0; hi = min(2.0, 0.5) = 0.5
        expected = pack_pk_f16(NEG_ONE, POS_HALF)
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_reg(Reg(0, RegKind.VGPR)), expected)

    def test_scalar_vgpr_sgpr_pk_max(self) -> None:
        # v0: (hi=-2.0, lo=0.5), s4: (hi=1.0, lo=1.0)
        self.state.set_reg(Reg(0, RegKind.VGPR), pack_pk_f16(POS_HALF, NEG_TWO))
        self.state.set_reg(Reg(4, RegKind.SGPR), pack_pk_f16(POS_ONE, POS_ONE))

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MAX_F16,
            dst=Reg(3, RegKind.VGPR),
            src0=VOP3POperand(Reg(0, RegKind.VGPR)),
            src1=VOP3POperand(Reg(4, RegKind.SGPR)),
        )
        res = self.interp.step(instr, self.state)
        # max: lo = max(0.5, 1.0) = 1.0; hi = max(-2.0, 1.0) = 1.0
        expected = pack_pk_f16(POS_ONE, POS_ONE)
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_reg(Reg(3, RegKind.VGPR)), expected)

    def test_scalar_literal_and_modifiers(self) -> None:
        # v5: (hi=-2.0, lo=-0.5)
        # src0: abs on both -> (hi=2.0, lo=0.5)
        # src1: literal (hi=1.0, lo=1.0)
        self.state.set_reg(Reg(5, RegKind.VGPR), pack_pk_f16(NEG_HALF, NEG_TWO))

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MIN_F16,
            dst=Reg(6, RegKind.VGPR),
            src0=VOP3POperand(Reg(5, RegKind.VGPR), abs_lo=True, abs_hi=True),
            src1=VOP3POperand(Literal32(pack_pk_f16(POS_ONE, POS_ONE))),
        )
        res = self.interp.step(instr, self.state)
        # min: lo = min(0.5, 1.0) = 0.5; hi = min(2.0, 1.0) = 1.0
        expected = pack_pk_f16(POS_HALF, POS_ONE)
        self.assertEqual(res, expected)

    def test_scalar_clamp_mode(self) -> None:
        # v1: (hi=2.0, lo=-1.0), v2: (hi=3.0, lo=-2.0)
        # pk_max: lo = max(-1.0, -2.0) = -1.0 -> clamp -> +0.0
        #         hi = max(2.0, 3.0) = 3.0 -> clamp -> 1.0
        self.state.set_reg(Reg(1, RegKind.VGPR), pack_pk_f16(NEG_ONE, POS_TWO))
        self.state.set_reg(Reg(2, RegKind.VGPR), pack_pk_f16(NEG_TWO, f16_to_bits(3.0)))

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MAX_F16,
            dst=Reg(0, RegKind.VGPR),
            src0=VOP3POperand(Reg(1, RegKind.VGPR)),
            src1=VOP3POperand(Reg(2, RegKind.VGPR)),
            clamp=True,
        )
        res = self.interp.step(instr, self.state)
        expected = pack_pk_f16(POS_ZERO, POS_ONE)
        self.assertEqual(res, expected)

    def test_scalar_determinism(self) -> None:
        runs = []
        for _ in range(50):
            st = ScalarMachineState()
            st.set_reg(Reg(0, RegKind.VGPR), 0x3C004000)
            st.set_reg(Reg(1, RegKind.VGPR), 0x42003800)
            instr = VPkMinMaxF16Instruction(
                opcode=Opcode.V_PK_MIN_F16,
                dst=Reg(2, RegKind.VGPR),
                src0=VOP3POperand(Reg(0, RegKind.VGPR)),
                src1=VOP3POperand(Reg(1, RegKind.VGPR)),
            )
            runs.append(self.interp.step(instr, st))
        self.assertTrue(all(r == runs[0] for r in runs))


# ==============================================================================
# SIMD Multi-Lane Execution Engine and EXEC Masking
# ==============================================================================

@dataclass
class SIMDState:
    """Multi-lane SIMD machine state representing Wave32 or Wave64."""
    num_lanes: int
    exec_mask: Optional[int] = None
    vgprs: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)
    vcc: int = 0

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {self.num_lanes}")
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        if self.exec_mask is None:
            self.exec_mask = (1 << self.num_lanes) - 1

    def get_lane_reg(self, lane: int, reg: Reg) -> int:
        if reg.kind == RegKind.VGPR:
            return self.vgprs[lane][reg.index] & MASK32
        elif reg.kind == RegKind.SGPR:
            return self.sgprs[reg.index] & MASK32
        raise TypeError(f"Unknown RegKind: {reg.kind}")

    def set_lane_vgpr(self, lane: int, reg: Reg, val: int) -> None:
        if reg.kind != RegKind.VGPR:
            raise TypeError(f"Cannot set lane register for non-VGPR: {reg.kind}")
        self.vgprs[lane][reg.index] = val & MASK32

    def set_sgpr(self, reg: Reg, val: int) -> None:
        if reg.kind != RegKind.SGPR:
            raise TypeError(f"Cannot set SGPR on VGPR: {reg.kind}")
        self.sgprs[reg.index] = val & MASK32


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VOP3POperand, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, Reg):
            return state.get_lane_reg(lane, op.source)
        elif isinstance(op.source, Literal32):
            return op.source.bits
        raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VPkMinMaxF16Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * state.num_lanes

        # Atomic read phase across all lanes
        s0_vals = [
            self.resolve_lane_operand(instr.src0, state, lane)
            for lane in range(state.num_lanes)
        ]
        s1_vals = [
            self.resolve_lane_operand(instr.src1, state, lane)
            for lane in range(state.num_lanes)
        ]

        op_sel = (
            instr.src0.op_sel[0],
            instr.src1.op_sel[0],
            instr.src0.op_sel[1],
            instr.src1.op_sel[1],
        )

        # Execution phase under EXEC mask
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                if instr.opcode == Opcode.V_PK_MIN_F16:
                    res = oracle_pk_min_f16(
                        s0_vals[lane], s1_vals[lane],
                        op_sel=op_sel,
                        abs0_lo=instr.src0.abs_lo, neg0_lo=instr.src0.neg_lo,
                        abs1_lo=instr.src1.abs_lo, neg1_lo=instr.src1.neg_lo,
                        abs0_hi=instr.src0.abs_hi, neg0_hi=instr.src0.neg_hi,
                        abs1_hi=instr.src1.abs_hi, neg1_hi=instr.src1.neg_hi,
                        clamp=instr.clamp,
                    )
                elif instr.opcode == Opcode.V_PK_MAX_F16:
                    res = oracle_pk_max_f16(
                        s0_vals[lane], s1_vals[lane],
                        op_sel=op_sel,
                        abs0_lo=instr.src0.abs_lo, neg0_lo=instr.src0.neg_lo,
                        abs1_lo=instr.src1.abs_lo, neg1_lo=instr.src1.neg_lo,
                        abs0_hi=instr.src0.abs_hi, neg0_hi=instr.src0.neg_hi,
                        abs1_hi=instr.src1.abs_hi, neg1_hi=instr.src1.neg_hi,
                        clamp=instr.clamp,
                    )
                else:
                    raise ValueError(f"Unsupported opcode: {instr.opcode}")

                state.set_lane_vgpr(lane, instr.dst, res)
                results[lane] = res

        return results


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD execution, lane divergence, and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_wave32_full_exec(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            v0_val = pack_pk_f16(f16_to_bits(float(lane)), f16_to_bits(float(lane) * 2.0))
            v1_val = pack_pk_f16(f16_to_bits(10.0), f16_to_bits(15.0))
            state.set_lane_vgpr(lane, Reg(0, RegKind.VGPR), v0_val)
            state.set_lane_vgpr(lane, Reg(1, RegKind.VGPR), v1_val)

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MIN_F16,
            dst=Reg(2, RegKind.VGPR),
            src0=VOP3POperand(Reg(0, RegKind.VGPR)),
            src1=VOP3POperand(Reg(1, RegKind.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            lo_exp = f16_to_bits(min(float(lane), 10.0))
            hi_exp = f16_to_bits(min(float(lane) * 2.0, 15.0))
            expected = pack_pk_f16(lo_exp, hi_exp)
            self.assertEqual(state.get_lane_reg(lane, Reg(2, RegKind.VGPR)), expected)

    def test_wave32_divergent_masking(self) -> None:
        # Alternating mask: even lanes active (0x55555555)
        sentinel = 0xDEADBEEF
        state = SIMDState(num_lanes=32, exec_mask=0x5555_5555)
        for lane in range(32):
            state.set_lane_vgpr(lane, Reg(0, RegKind.VGPR), pack_pk_f16(POS_ONE, POS_TWO))
            state.set_lane_vgpr(lane, Reg(1, RegKind.VGPR), pack_pk_f16(POS_TWO, POS_ONE))
            state.set_lane_vgpr(lane, Reg(2, RegKind.VGPR), sentinel)

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MAX_F16,
            dst=Reg(2, RegKind.VGPR),
            src0=VOP3POperand(Reg(0, RegKind.VGPR)),
            src1=VOP3POperand(Reg(1, RegKind.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertIsNotNone(results[lane])
                self.assertEqual(
                    state.get_lane_reg(lane, Reg(2, RegKind.VGPR)),
                    pack_pk_f16(POS_TWO, POS_TWO),
                )
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_reg(lane, Reg(2, RegKind.VGPR)), sentinel)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_lane_vgpr(lane, Reg(0, RegKind.VGPR), pack_pk_f16(POS_ONE, NEG_ONE))
            state.set_lane_vgpr(lane, Reg(1, RegKind.VGPR), pack_pk_f16(POS_TWO, NEG_TWO))

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MIN_F16,
            dst=Reg(0, RegKind.VGPR),
            src0=VOP3POperand(Reg(0, RegKind.VGPR)),
            src1=VOP3POperand(Reg(1, RegKind.VGPR)),
        )
        self.interp.step(instr, state)
        for lane in range(64):
            # min(1.0, 2.0) = 1.0; min(-1.0, -2.0) = -2.0
            self.assertEqual(
                state.get_lane_reg(lane, Reg(0, RegKind.VGPR)),
                pack_pk_f16(POS_ONE, NEG_TWO),
            )

    def test_wave64_half_disabled(self) -> None:
        # Lower 32 lanes active, upper 32 lanes inactive
        sentinel = 0xCAFEBABE
        state = SIMDState(num_lanes=64, exec_mask=0x0000_0000_FFFF_FFFF)
        for lane in range(64):
            state.set_lane_vgpr(lane, Reg(0, RegKind.VGPR), pack_pk_f16(POS_HALF, POS_ONE))
            state.set_lane_vgpr(lane, Reg(1, RegKind.VGPR), pack_pk_f16(POS_ONE, POS_HALF))
            state.set_lane_vgpr(lane, Reg(3, RegKind.VGPR), sentinel)

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MAX_F16,
            dst=Reg(3, RegKind.VGPR),
            src0=VOP3POperand(Reg(0, RegKind.VGPR)),
            src1=VOP3POperand(Reg(1, RegKind.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(64):
            if lane < 32:
                self.assertIsNotNone(results[lane])
                self.assertEqual(
                    state.get_lane_reg(lane, Reg(3, RegKind.VGPR)),
                    pack_pk_f16(POS_ONE, POS_ONE),
                )
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_reg(lane, Reg(3, RegKind.VGPR)), sentinel)

    def test_wave_exec_all_zero(self) -> None:
        sentinel = 0xAA55AA55
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)
        for lane in range(32):
            state.set_lane_vgpr(lane, Reg(0, RegKind.VGPR), sentinel)

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MIN_F16,
            dst=Reg(0, RegKind.VGPR),
            src0=VOP3POperand(Literal32(pack_pk_f16(POS_ONE, POS_ONE))),
            src1=VOP3POperand(Literal32(pack_pk_f16(POS_TWO, POS_TWO))),
        )
        results = self.interp.step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(state.get_lane_reg(lane, Reg(0, RegKind.VGPR)), sentinel)

    def test_simd_sgpr_uniform_broadcast(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        state.set_sgpr(Reg(4, RegKind.SGPR), pack_pk_f16(POS_TWO, POS_TWO))
        for lane in range(32):
            state.set_lane_vgpr(lane, Reg(0, RegKind.VGPR), pack_pk_f16(POS_ONE, POS_ONE))

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MAX_F16,
            dst=Reg(1, RegKind.VGPR),
            src0=VOP3POperand(Reg(0, RegKind.VGPR)),
            src1=VOP3POperand(Reg(4, RegKind.SGPR)),
        )
        self.interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(
                state.get_lane_reg(lane, Reg(1, RegKind.VGPR)),
                pack_pk_f16(POS_TWO, POS_TWO),
            )


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src0(self) -> None:
        state = ScalarMachineState()
        # v0: (hi=2.0, lo=-1.0), v1: (hi=0.5, lo=1.0)
        state.set_reg(Reg(0, RegKind.VGPR), pack_pk_f16(NEG_ONE, POS_TWO))
        state.set_reg(Reg(1, RegKind.VGPR), pack_pk_f16(POS_ONE, POS_HALF))

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MIN_F16,
            dst=Reg(0, RegKind.VGPR),
            src0=VOP3POperand(Reg(0, RegKind.VGPR)),
            src1=VOP3POperand(Reg(1, RegKind.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        expected = pack_pk_f16(NEG_ONE, POS_HALF)
        self.assertEqual(res, expected)
        self.assertEqual(state.get_reg(Reg(0, RegKind.VGPR)), expected)
        # src1 (v1) must remain untouched
        self.assertEqual(state.get_reg(Reg(1, RegKind.VGPR)), pack_pk_f16(POS_ONE, POS_HALF))

    def test_dst_alias_src1(self) -> None:
        state = ScalarMachineState()
        state.set_reg(Reg(0, RegKind.VGPR), pack_pk_f16(NEG_ONE, POS_TWO))
        state.set_reg(Reg(1, RegKind.VGPR), pack_pk_f16(POS_ONE, POS_HALF))

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MIN_F16,
            dst=Reg(1, RegKind.VGPR),
            src0=VOP3POperand(Reg(0, RegKind.VGPR)),
            src1=VOP3POperand(Reg(1, RegKind.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        expected = pack_pk_f16(NEG_ONE, POS_HALF)
        self.assertEqual(res, expected)
        self.assertEqual(state.get_reg(Reg(1, RegKind.VGPR)), expected)
        # src0 (v0) must remain untouched
        self.assertEqual(state.get_reg(Reg(0, RegKind.VGPR)), pack_pk_f16(NEG_ONE, POS_TWO))

    def test_dst_alias_both_srcs(self) -> None:
        state = ScalarMachineState()
        val = pack_pk_f16(POS_HALF, NEG_TWO)
        state.set_reg(Reg(0, RegKind.VGPR), val)

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MAX_F16,
            dst=Reg(0, RegKind.VGPR),
            src0=VOP3POperand(Reg(0, RegKind.VGPR)),
            src1=VOP3POperand(Reg(0, RegKind.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, val)
        self.assertEqual(state.get_reg(Reg(0, RegKind.VGPR)), val)

    def test_inplace_iterative_accumulation(self) -> None:
        state = ScalarMachineState()
        # Initial: large values (100.0, 100.0)
        state.set_reg(Reg(0, RegKind.VGPR), pack_pk_f16(f16_to_bits(100.0), f16_to_bits(100.0)))

        # Repeatedly compute min against descending values
        for i in range(50):
            step_val = float(50 - i)
            state.set_reg(Reg(1, RegKind.VGPR), pack_pk_f16(f16_to_bits(step_val), f16_to_bits(step_val * 2.0)))
            instr = VPkMinMaxF16Instruction(
                opcode=Opcode.V_PK_MIN_F16,
                dst=Reg(0, RegKind.VGPR),
                src0=VOP3POperand(Reg(0, RegKind.VGPR)),
                src1=VOP3POperand(Reg(1, RegKind.VGPR)),
            )
            self.scalar_interp.step(instr, state)

        # After 50 iterations: lo should be 1.0, hi should be 2.0
        final_val = state.get_reg(Reg(0, RegKind.VGPR))
        lo, hi = unpack_pk_f16(final_val)
        self.assertEqual(lo, f16_to_bits(1.0))
        self.assertEqual(hi, f16_to_bits(2.0))

    def test_inplace_cross_half_swizzle_alias(self) -> None:
        # Cross-half swizzle: v0 = v_pk_min_f16(v0.hi, v0.lo)
        state = ScalarMachineState()
        # lo = 5.0, hi = 2.0
        state.set_reg(Reg(0, RegKind.VGPR), pack_pk_f16(f16_to_bits(5.0), f16_to_bits(2.0)))

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MIN_F16,
            dst=Reg(0, RegKind.VGPR),
            src0=VOP3POperand(Reg(0, RegKind.VGPR), op_sel=(0, 0)),  # select lo for both
            src1=VOP3POperand(Reg(0, RegKind.VGPR), op_sel=(1, 1)),  # select hi for both
        )
        res = self.scalar_interp.step(instr, state)
        # min(5.0, 2.0) = 2.0 for both lo and hi
        expected = pack_pk_f16(f16_to_bits(2.0), f16_to_bits(2.0))
        self.assertEqual(res, expected)
        self.assertEqual(state.get_reg(Reg(0, RegKind.VGPR)), expected)

    def test_simd_multi_lane_aliasing(self) -> None:
        # SIMD in-place aliasing under EXEC mask
        sentinel = 0x55AA55AA
        state = SIMDState(num_lanes=32, exec_mask=0x0000_FFFF)  # lower 16 lanes active
        for lane in range(32):
            state.set_lane_vgpr(lane, Reg(0, RegKind.VGPR), pack_pk_f16(POS_ONE, POS_ONE))
            state.set_lane_vgpr(lane, Reg(1, RegKind.VGPR), pack_pk_f16(POS_TWO, POS_TWO))
            state.set_lane_vgpr(lane, Reg(2, RegKind.VGPR), sentinel)

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MAX_F16,
            dst=Reg(0, RegKind.VGPR),
            src0=VOP3POperand(Reg(0, RegKind.VGPR)),
            src1=VOP3POperand(Reg(1, RegKind.VGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            if lane < 16:
                # Active: updated in place to max(1.0, 2.0) = 2.0
                self.assertEqual(
                    state.get_lane_reg(lane, Reg(0, RegKind.VGPR)),
                    pack_pk_f16(POS_TWO, POS_TWO),
                )
            else:
                # Inactive: untouched
                self.assertEqual(
                    state.get_lane_reg(lane, Reg(0, RegKind.VGPR)),
                    pack_pk_f16(POS_ONE, POS_ONE),
                )
            # Reg 1 and Reg 2 remain unchanged
            self.assertEqual(state.get_lane_reg(lane, Reg(1, RegKind.VGPR)), pack_pk_f16(POS_TWO, POS_TWO))
            self.assertEqual(state.get_lane_reg(lane, Reg(2, RegKind.VGPR)), sentinel)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class PackedConditionCodes:
    """Architectural comparison condition codes for packed binary16 min/max."""
    equal_lo: bool
    less_lo: bool
    greater_lo: bool
    unordered_lo: bool
    equal_hi: bool
    less_hi: bool
    greater_hi: bool
    unordered_hi: bool
    zero_lo: bool
    zero_hi: bool
    negative_lo: bool
    negative_hi: bool


@dataclass(frozen=True)
class PackedFPStatusFlags:
    """IEEE 754 floating-point status flags for packed half-precision operations."""
    invalid: bool       # e.g., sNaN operand
    denormal: bool      # e.g., subnormal operand or result
    inexact: bool = False
    overflow: bool = False
    underflow: bool = False


def _eval_half_relation(a: int, b: int, res: int) -> Tuple[bool, bool, bool, bool, bool, bool]:
    """Helper to evaluate relation for a single half (equal, less, greater, unordered, zero, neg)."""
    nan_a = is_nan16(a)
    nan_b = is_nan16(b)
    if nan_a or nan_b:
        unordered = True
        equal = False
        less = False
        greater = False
    else:
        unordered = False
        fa = bits_to_f16(a)
        fb = bits_to_f16(b)
        a_is_neg_zero = (a == NEG_ZERO)
        b_is_neg_zero = (b == NEG_ZERO)
        a_is_pos_zero = (a == POS_ZERO)
        b_is_pos_zero = (b == POS_ZERO)

        if (a_is_neg_zero and b_is_pos_zero):
            equal = False
            less = True
            greater = False
        elif (a_is_pos_zero and b_is_neg_zero):
            equal = False
            less = False
            greater = True
        elif (a_is_neg_zero and b_is_neg_zero) or (a_is_pos_zero and b_is_pos_zero):
            equal = True
            less = False
            greater = False
        else:
            equal = (fa == fb)
            less = (fa < fb)
            greater = (fa > fb)

    zero = is_zero16(res)
    neg = bool(res & SIGN_MASK16)
    return equal, less, greater, unordered, zero, neg


def evaluate_pk_condition_codes(
    s0_u32: int,
    s1_u32: int,
    res_u32: int,
    op_sel: Tuple[int, int, int, int] = (0, 0, 1, 1),
) -> PackedConditionCodes:
    """Evaluate condition codes for packed operations across lo and hi halves."""
    s0_halves = unpack_pk_f16(s0_u32)
    s1_halves = unpack_pk_f16(s1_u32)
    res_halves = unpack_pk_f16(res_u32)

    eq_lo, lt_lo, gt_lo, un_lo, z_lo, neg_lo = _eval_half_relation(
        s0_halves[op_sel[0]], s1_halves[op_sel[1]], res_halves[0]
    )
    eq_hi, lt_hi, gt_hi, un_hi, z_hi, neg_hi = _eval_half_relation(
        s0_halves[op_sel[2]], s1_halves[op_sel[3]], res_halves[1]
    )

    return PackedConditionCodes(
        equal_lo=eq_lo, less_lo=lt_lo, greater_lo=gt_lo, unordered_lo=un_lo,
        equal_hi=eq_hi, less_hi=lt_hi, greater_hi=gt_hi, unordered_hi=un_hi,
        zero_lo=z_lo, zero_hi=z_hi, negative_lo=neg_lo, negative_hi=neg_hi,
    )


def evaluate_pk_status_flags(
    s0_u32: int,
    s1_u32: int,
    res_u32: int,
    op_sel: Tuple[int, int, int, int] = (0, 0, 1, 1),
) -> PackedFPStatusFlags:
    """Evaluate IEEE 754 architectural flags for packed binary16 min/max."""
    s0_halves = unpack_pk_f16(s0_u32)
    s1_halves = unpack_pk_f16(s1_u32)
    res_halves = unpack_pk_f16(res_u32)

    in0_lo = s0_halves[op_sel[0]]
    in1_lo = s1_halves[op_sel[1]]
    in0_hi = s0_halves[op_sel[2]]
    in1_hi = s1_halves[op_sel[3]]

    # Invalid operation triggered by any signaling NaN input
    invalid = (
        is_snan16(in0_lo) or is_snan16(in1_lo) or
        is_snan16(in0_hi) or is_snan16(in1_hi)
    )

    # Denormal flag triggered by any subnormal input or result
    denormal = (
        is_subnormal16(in0_lo) or is_subnormal16(in1_lo) or
        is_subnormal16(in0_hi) or is_subnormal16(in1_hi) or
        is_subnormal16(res_halves[0]) or is_subnormal16(res_halves[1])
    )

    return PackedFPStatusFlags(
        invalid=invalid,
        denormal=denormal,
        inexact=False,
        overflow=False,
        underflow=False,
    )


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow isolation."""

    def test_condition_codes_ordering(self) -> None:
        # lo: 1.0 < 2.0 (less_lo=True), hi: 3.0 > 0.5 (greater_hi=True)
        s0 = pack_pk_f16(POS_ONE, f16_to_bits(3.0))
        s1 = pack_pk_f16(POS_TWO, POS_HALF)
        res = oracle_pk_min_f16(s0, s1)

        cc = evaluate_pk_condition_codes(s0, s1, res)
        self.assertTrue(cc.less_lo)
        self.assertFalse(cc.greater_lo)
        self.assertFalse(cc.equal_lo)
        self.assertFalse(cc.unordered_lo)

        self.assertTrue(cc.greater_hi)
        self.assertFalse(cc.less_hi)
        self.assertFalse(cc.equal_hi)
        self.assertFalse(cc.unordered_hi)

    def test_condition_codes_unordered_nan(self) -> None:
        # lo: NaN vs 1.0 -> unordered_lo = True
        # hi: 2.0 == 2.0 -> equal_hi = True
        s0 = pack_pk_f16(QNAN_CANONICAL, POS_TWO)
        s1 = pack_pk_f16(POS_ONE, POS_TWO)
        res = oracle_pk_min_f16(s0, s1)

        cc = evaluate_pk_condition_codes(s0, s1, res)
        self.assertTrue(cc.unordered_lo)
        self.assertFalse(cc.less_lo)
        self.assertFalse(cc.greater_lo)
        self.assertFalse(cc.equal_lo)

        self.assertTrue(cc.equal_hi)
        self.assertFalse(cc.unordered_hi)

    def test_condition_codes_signed_zeros(self) -> None:
        # lo: -0.0 vs +0.0 -> -0.0 < +0.0, so less_lo=True, zero_lo=True, negative_lo=True
        # hi: +0.0 vs -0.0 -> +0.0 > -0.0, so greater_hi=True
        s0 = pack_pk_f16(NEG_ZERO, POS_ZERO)
        s1 = pack_pk_f16(POS_ZERO, NEG_ZERO)
        res = oracle_pk_min_f16(s0, s1)

        cc = evaluate_pk_condition_codes(s0, s1, res)
        self.assertTrue(cc.less_lo)
        self.assertTrue(cc.zero_lo)
        self.assertTrue(cc.negative_lo)

        self.assertTrue(cc.greater_hi)
        self.assertTrue(cc.zero_hi)
        self.assertTrue(cc.negative_hi)  # min(+0.0, -0.0) == -0.0

    def test_status_flags_snan_invalid(self) -> None:
        # Signaling NaN in lo triggers invalid flag
        s0 = pack_pk_f16(SNAN_DEFAULT, POS_ONE)
        s1 = pack_pk_f16(POS_TWO, POS_TWO)
        res = oracle_pk_min_f16(s0, s1)

        flags = evaluate_pk_status_flags(s0, s1, res)
        self.assertTrue(flags.invalid)
        self.assertFalse(flags.denormal)

    def test_status_flags_subnormal_denormal(self) -> None:
        # Subnormal input triggers denormal flag
        s0 = pack_pk_f16(POS_MIN_SUBNORM, POS_ONE)
        s1 = pack_pk_f16(POS_ONE, POS_ONE)
        res = oracle_pk_min_f16(s0, s1)

        flags = evaluate_pk_status_flags(s0, s1, res)
        self.assertFalse(flags.invalid)
        self.assertTrue(flags.denormal)

    def test_carry_borrow_isolation(self) -> None:
        # In RDNA2, packed VALU min/max instructions do not produce carry/borrow or alter VCC.
        interp = ScalarInterpreter()
        state = ScalarMachineState()
        state.vcc = 0x12345678  # pre-existing VCC condition code

        state.set_reg(Reg(0, RegKind.VGPR), pack_pk_f16(POS_ONE, POS_TWO))
        state.set_reg(Reg(1, RegKind.VGPR), pack_pk_f16(POS_TWO, POS_ONE))

        instr = VPkMinMaxF16Instruction(
            opcode=Opcode.V_PK_MIN_F16,
            dst=Reg(2, RegKind.VGPR),
            src0=VOP3POperand(Reg(0, RegKind.VGPR)),
            src1=VOP3POperand(Reg(1, RegKind.VGPR)),
        )
        interp.step(instr, state)
        # VCC must remain strictly preserved
        self.assertEqual(state.vcc, 0x12345678)

        # Same for SIMD execution
        simd_interp = SIMDInterpreter()
        simd_state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        simd_state.vcc = 0x87654321
        simd_interp.step(instr, simd_state)
        self.assertEqual(simd_state.vcc, 0x87654321)


# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, infinities, and NaNs."""

    def test_subnormal_boundaries_and_ordering(self) -> None:
        # Subnormal vs normal transition
        self.assertEqual(oracle_min_f16(POS_MAX_SUBNORM, POS_MIN_NORMAL), POS_MAX_SUBNORM)
        self.assertEqual(oracle_max_f16(POS_MAX_SUBNORM, POS_MIN_NORMAL), POS_MIN_NORMAL)

        # Negative subnormal vs negative normal
        self.assertEqual(oracle_min_f16(NEG_MAX_SUBNORM, NEG_MIN_SUBNORM), NEG_MAX_SUBNORM)
        self.assertEqual(oracle_max_f16(NEG_MAX_SUBNORM, NEG_MIN_SUBNORM), NEG_MIN_SUBNORM)

        # Subnormal vs zero
        self.assertEqual(oracle_min_f16(POS_ZERO, POS_MIN_SUBNORM), POS_ZERO)
        self.assertEqual(oracle_max_f16(POS_ZERO, POS_MIN_SUBNORM), POS_MIN_SUBNORM)
        self.assertEqual(oracle_min_f16(NEG_ZERO, NEG_MIN_SUBNORM), NEG_MIN_SUBNORM)
        self.assertEqual(oracle_max_f16(NEG_ZERO, NEG_MIN_SUBNORM), NEG_ZERO)

    def test_max_finite_and_infinity_boundaries(self) -> None:
        # Max positive normal vs +inf
        self.assertEqual(oracle_min_f16(POS_MAX_NORMAL, POS_INF), POS_MAX_NORMAL)
        self.assertEqual(oracle_max_f16(POS_MAX_NORMAL, POS_INF), POS_INF)

        # Max negative normal vs -inf
        self.assertEqual(oracle_min_f16(NEG_MAX_NORMAL, NEG_INF), NEG_INF)
        self.assertEqual(oracle_max_f16(NEG_MAX_NORMAL, NEG_INF), NEG_MAX_NORMAL)

        # +inf vs -inf
        self.assertEqual(oracle_min_f16(POS_INF, NEG_INF), NEG_INF)
        self.assertEqual(oracle_max_f16(POS_INF, NEG_INF), POS_INF)

    def test_nan_suppression_against_infinities_and_subnormals(self) -> None:
        qnan_custom = 0x7E55
        snan_custom = 0x7D33

        # NaN vs +inf: +inf is selected!
        self.assertEqual(oracle_min_f16(qnan_custom, POS_INF), POS_INF)
        self.assertEqual(oracle_max_f16(qnan_custom, POS_INF), POS_INF)

        # NaN vs -inf: -inf is selected!
        self.assertEqual(oracle_min_f16(snan_custom, NEG_INF), NEG_INF)
        self.assertEqual(oracle_max_f16(snan_custom, NEG_INF), NEG_INF)

        # NaN vs subnormal: subnormal is selected!
        self.assertEqual(oracle_min_f16(qnan_custom, POS_MIN_SUBNORM), POS_MIN_SUBNORM)
        self.assertEqual(oracle_max_f16(qnan_custom, POS_MIN_SUBNORM), POS_MIN_SUBNORM)

    def test_both_operands_nan_variants(self) -> None:
        qnan1 = 0x7E12
        qnan2 = 0x7E34
        res_min = oracle_min_f16(qnan1, qnan2)
        self.assertTrue(is_qnan16(res_min))

        snan = 0x7D44
        res_max = oracle_max_f16(snan, qnan1)
        self.assertTrue(is_qnan16(res_max))

    def test_packed_asymmetric_extremes(self) -> None:
        # lo: NaN vs subnormal -> subnormal selected
        # hi: +inf vs -inf -> min selects -inf, max selects +inf
        s0 = pack_pk_f16(QNAN_CANONICAL, POS_INF)
        s1 = pack_pk_f16(POS_MAX_SUBNORM, NEG_INF)

        res_min = oracle_pk_min_f16(s0, s1)
        self.assertEqual(res_min, pack_pk_f16(POS_MAX_SUBNORM, NEG_INF))

        res_max = oracle_pk_max_f16(s0, s1)
        self.assertEqual(res_max, pack_pk_f16(POS_MAX_SUBNORM, POS_INF))

    def test_clamp_extremes_interaction(self) -> None:
        # Positive subnormal (within [0.0, 1.0]) is preserved under clamp
        self.assertEqual(clamp_f16(POS_MIN_SUBNORM), POS_MIN_SUBNORM)
        self.assertEqual(clamp_f16(POS_MAX_SUBNORM), POS_MAX_SUBNORM)

        # Negative subnormal clamps to +0.0
        self.assertEqual(clamp_f16(NEG_MIN_SUBNORM), POS_ZERO)
        self.assertEqual(clamp_f16(NEG_MAX_SUBNORM), POS_ZERO)

        # +inf clamps to 1.0
        self.assertEqual(clamp_f16(POS_INF), POS_ONE)

        # -inf clamps to +0.0
        self.assertEqual(clamp_f16(NEG_INF), POS_ZERO)

        # NaN clamps to +0.0
        self.assertEqual(clamp_f16(QNAN_CANONICAL), POS_ZERO)
        self.assertEqual(clamp_f16(SNAN_DEFAULT), POS_ZERO)


# ==============================================================================
# Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestNegativeControlsAndMutationDetection(unittest.TestCase):
    """Negative controls, architectural syntax rejection, and mutation detection tests."""

    def test_syntax_rejection_registers(self) -> None:
        with self.assertRaises(ValueError):
            Reg(256, RegKind.VGPR)
        with self.assertRaises(ValueError):
            Reg(-1, RegKind.VGPR)
        with self.assertRaises(ValueError):
            Reg(106, RegKind.SGPR)
        with self.assertRaises(ValueError):
            Reg(-1, RegKind.SGPR)

    def test_syntax_rejection_instruction_dst(self) -> None:
        # Cannot assign SGPR as destination of packed VALU instruction
        with self.assertRaises(TypeError):
            VPkMinMaxF16Instruction(
                opcode=Opcode.V_PK_MIN_F16,
                dst=Reg(0, RegKind.SGPR),
                src0=VOP3POperand(Reg(1, RegKind.VGPR)),
                src1=VOP3POperand(Reg(2, RegKind.VGPR)),
            )

    def test_syntax_rejection_op_sel(self) -> None:
        with self.assertRaises(ValueError):
            VOP3POperand(Reg(0, RegKind.VGPR), op_sel=(0, 2))
        with self.assertRaises(ValueError):
            VOP3POperand(Reg(0, RegKind.VGPR), op_sel=(-1, 1))

    def test_syntax_rejection_literal_bounds(self) -> None:
        with self.assertRaises(ValueError):
            Literal32(-1)
        with self.assertRaises(ValueError):
            Literal32(0x1_0000_0000)

    def test_syntax_rejection_simd_lanes(self) -> None:
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_syntax_rejection_lane_reg_type(self) -> None:
        state = SIMDState(num_lanes=32)
        with self.assertRaises(TypeError):
            state.set_lane_vgpr(0, Reg(0, RegKind.SGPR), 0)

    def test_mutation_detection_opcode_inversion(self) -> None:
        # Sabotaging min to max must be detected
        s0 = pack_pk_f16(POS_ONE, POS_TWO)
        s1 = pack_pk_f16(POS_HALF, POS_HALF)

        correct_min = oracle_pk_min_f16(s0, s1)
        mutated_max = oracle_pk_max_f16(s0, s1)

        self.assertNotEqual(correct_min, mutated_max)
        self.assertEqual(correct_min, pack_pk_f16(POS_HALF, POS_HALF))
        self.assertEqual(mutated_max, pack_pk_f16(POS_ONE, POS_TWO))

    def test_mutation_detection_signed_zero_tiebreak(self) -> None:
        # Sabotaging signed zero tiebreak: min(-0.0, +0.0) must be -0.0, not +0.0
        got = oracle_min_f16(NEG_ZERO, POS_ZERO)
        self.assertEqual(got, NEG_ZERO)
        self.assertNotEqual(got, POS_ZERO)

        # max(-0.0, +0.0) must be +0.0, not -0.0
        got_max = oracle_max_f16(NEG_ZERO, POS_ZERO)
        self.assertEqual(got_max, POS_ZERO)
        self.assertNotEqual(got_max, NEG_ZERO)

    def test_mutation_detection_nan_suppression(self) -> None:
        # Sabotaging NaN suppression to propagate NaN must be detected
        got = oracle_min_f16(QNAN_CANONICAL, POS_TWO)
        self.assertEqual(got, POS_TWO)
        self.assertFalse(is_nan16(got))

    def test_mutation_detection_clamp_bypass(self) -> None:
        # Sabotaging clamp (omitting clamp on 2.0) must be detected
        val = POS_TWO  # 2.0
        clamped = clamp_f16(val)
        self.assertEqual(clamped, POS_ONE)  # clamped to 1.0
        self.assertNotEqual(clamped, val)

    def test_mutation_detection_op_sel_tampering(self) -> None:
        # s0: lo=1.0, hi=2.0; s1: lo=1.5, hi=3.0
        s0 = pack_pk_f16(POS_ONE, POS_TWO)
        s1 = pack_pk_f16(f16_to_bits(1.5), f16_to_bits(3.0))

        # Standard op_sel: lo gets s0.lo (1.0) vs s1.lo (1.5) -> 1.0
        #                  hi gets s0.hi (2.0) vs s1.hi (3.0) -> 2.0
        std_res = oracle_pk_min_f16(s0, s1, op_sel=(0, 0, 1, 1))

        # Tampered op_sel: swap s0 halves
        #                  lo gets s0.hi (2.0) vs s1.lo (1.5) -> 1.5
        #                  hi gets s0.lo (1.0) vs s1.hi (3.0) -> 1.0
        tampered_res = oracle_pk_min_f16(s0, s1, op_sel=(1, 0, 0, 1))

        self.assertNotEqual(std_res, tampered_res)
        self.assertEqual(std_res, pack_pk_f16(POS_ONE, POS_TWO))
        self.assertEqual(tampered_res, pack_pk_f16(f16_to_bits(1.5), POS_ONE))


if __name__ == "__main__":
    unittest.main()
