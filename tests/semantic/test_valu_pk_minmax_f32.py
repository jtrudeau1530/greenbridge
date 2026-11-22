from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

# ==============================================================================
# IEEE 754 binary32 (Single-Precision) Architectural Constants
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF

SIGN_MASK32: int = 0x8000_0000
EXP_MASK32: int = 0x7F80_0000
FRAC_MASK32: int = 0x007F_FFFF
QNAN_BIT32: int = 0x0040_0000

# Canonical and special binary32 values
POS_ZERO: int = 0x0000_0000          # +0.0
NEG_ZERO: int = 0x8000_0000          # -0.0
POS_ONE: int = 0x3F80_0000           # +1.0
NEG_ONE: int = 0xBF80_0000           # -1.0
POS_HALF: int = 0x3F00_0000          # +0.5
NEG_HALF: int = 0xBF00_0000          # -0.5
POS_TWO: int = 0x4000_0000           # +2.0
NEG_TWO: int = 0xC000_0000           # -2.0
POS_INF: int = 0x7F80_0000           # +inf
NEG_INF: int = 0xFF80_0000           # -inf
QNAN_CANONICAL: int = 0x7FC0_0000    # Canonical quiet NaN
SNAN_DEFAULT: int = 0x7F80_0001      # Signaling NaN default
NEG_QNAN: int = 0xFFC0_0000          # Negative quiet NaN

# Subnormal (denormal) binary32 constants
POS_MIN_SUBNORM: int = 0x0000_0001   # +2^-149, smallest positive subnormal
POS_MAX_SUBNORM: int = 0x007F_FFFF   # largest positive subnormal
NEG_MIN_SUBNORM: int = 0x8000_0001   # -2^-149, smallest negative subnormal
NEG_MAX_SUBNORM: int = 0x807F_FFFF   # largest negative subnormal

# Normal binary32 boundaries
POS_MIN_NORMAL: int = 0x0080_0000    # +2^-126, smallest positive normal
POS_MAX_NORMAL: int = 0x7F7F_FFFF    # largest finite binary32
NEG_MAX_NORMAL: int = 0xFF7F_FFFF    # largest negative finite binary32


# ==============================================================================
# Bit Manipulation and Conversion Utilities
# ==============================================================================

def f32_to_bits(val: float) -> int:
    """Convert a Python float to an IEEE 754 binary32 32-bit integer."""
    if math.isnan(val):
        return QNAN_CANONICAL
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def bits_to_f32(bits: int) -> float:
    """Convert an IEEE 754 binary32 32-bit integer to a Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def unpack_pk_f32(packed: int) -> Tuple[int, int]:
    """Unpack a 64-bit register pair into two 32-bit dwords (lo, hi)."""
    lo = packed & MASK32
    hi = (packed >> 32) & MASK32
    return lo, hi


def pack_pk_f32(lo: int, hi: int) -> int:
    """Pack two 32-bit dwords (lo, hi) into a single 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def is_nan32(bits: int) -> bool:
    """Check if 32-bit word represents an IEEE 754 binary32 NaN."""
    return ((bits & EXP_MASK32) == EXP_MASK32) and ((bits & FRAC_MASK32) != 0)


def is_qnan32(bits: int) -> bool:
    """Check if 32-bit word represents a Quiet NaN."""
    return is_nan32(bits) and bool(bits & QNAN_BIT32)


def is_snan32(bits: int) -> bool:
    """Check if 32-bit word represents a Signaling NaN."""
    return is_nan32(bits) and not bool(bits & QNAN_BIT32)


def is_inf32(bits: int) -> bool:
    """Check if 32-bit word represents positive or negative infinity."""
    return ((bits & EXP_MASK32) == EXP_MASK32) and ((bits & FRAC_MASK32) == 0)


def is_subnormal32(bits: int) -> bool:
    """Check if 32-bit word represents a subnormal (denormal) single-precision float."""
    return ((bits & EXP_MASK32) == 0) and ((bits & FRAC_MASK32) != 0)


def is_zero32(bits: int) -> bool:
    """Check if 32-bit word represents positive or negative zero."""
    return (bits & ~SIGN_MASK32) == 0


def apply_f32_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply single-precision float input modifiers (abs, neg)."""
    res = bits & MASK32
    if abs_mod:
        res &= ~SIGN_MASK32
    if neg_mod:
        res ^= SIGN_MASK32
    return res


# ==============================================================================
# Reference Mathematical Oracles for Scalar & Packed f32 Min/Max
# ==============================================================================

def clamp_f32(bits: int) -> int:
    """Clamp a 32-bit float to [0.0, 1.0]. NaN clamps to +0.0."""
    if is_nan32(bits):
        return POS_ZERO
    f = bits_to_f32(bits)
    if f < 0.0 or bits == NEG_ZERO:
        return POS_ZERO
    if f > 1.0 or bits == POS_INF:
        return POS_ONE
    return bits & MASK32


def oracle_min_f32(
    s0: int,
    s1: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for RDNA2 scalar single-precision min (IEEE 754-2008 / GLSL NMin).

    Rules:
    - NaN suppression: if one operand is NaN and the other is a number, return the number.
    - If both operands are NaN, return quiet NaN (canonical quiet NaN).
    - Signed zero tie-breaking: -0.0 < +0.0 (min(-0.0, +0.0) == -0.0).
    - Subnormal numbers are preserved.
    - If clamp=True, output clamped to [0.0, 1.0].
    """
    a = apply_f32_modifiers(s0, abs0, neg0)
    b = apply_f32_modifiers(s1, abs1, neg1)

    a_nan = is_nan32(a)
    b_nan = is_nan32(b)

    if a_nan and b_nan:
        res = a | QNAN_BIT32
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
            fa = bits_to_f32(a)
            fb = bits_to_f32(b)
            if fa < fb:
                res = a
            elif fb < fa:
                res = b
            else:
                res = a if (a & SIGN_MASK32) else b

    if clamp:
        res = clamp_f32(res)
    return res & MASK32


def oracle_max_f32(
    s0: int,
    s1: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference oracle for RDNA2 scalar single-precision max (IEEE 754-2008 / GLSL NMax).

    Rules:
    - NaN suppression: if one operand is NaN and the other is a number, return the number.
    - If both operands are NaN, return quiet NaN.
    - Signed zero tie-breaking: -0.0 < +0.0 (max(-0.0, +0.0) == +0.0).
    - Subnormal numbers are preserved.
    - If clamp=True, output clamped to [0.0, 1.0].
    """
    a = apply_f32_modifiers(s0, abs0, neg0)
    b = apply_f32_modifiers(s1, abs1, neg1)

    a_nan = is_nan32(a)
    b_nan = is_nan32(b)

    if a_nan and b_nan:
        res = a | QNAN_BIT32
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
            fa = bits_to_f32(a)
            fb = bits_to_f32(b)
            if fa > fb:
                res = a
            elif fb > fa:
                res = b
            else:
                res = b if not (b & SIGN_MASK32) else a

    if clamp:
        res = clamp_f32(res)
    return res & MASK32


def oracle_pk_min_f32(
    s0_u64: int,
    s1_u64: int,
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
    """Reference oracle for RDNA2 packed single-precision v_pk_min_f32."""
    s0_words = unpack_pk_f32(s0_u64)
    s1_words = unpack_pk_f32(s1_u64)

    in0_lo = s0_words[op_sel[0]]
    in1_lo = s1_words[op_sel[1]]
    in0_hi = s0_words[op_sel[2]]
    in1_hi = s1_words[op_sel[3]]

    out_lo = oracle_min_f32(
        in0_lo, in1_lo,
        abs0=abs0_lo, neg0=neg0_lo, abs1=abs1_lo, neg1=neg1_lo,
        clamp=clamp,
    )
    out_hi = oracle_min_f32(
        in0_hi, in1_hi,
        abs0=abs0_hi, neg0=neg0_hi, abs1=abs1_hi, neg1=neg1_hi,
        clamp=clamp,
    )

    return pack_pk_f32(out_lo, out_hi)


def oracle_pk_max_f32(
    s0_u64: int,
    s1_u64: int,
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
    """Reference oracle for RDNA2 packed single-precision v_pk_max_f32."""
    s0_words = unpack_pk_f32(s0_u64)
    s1_words = unpack_pk_f32(s1_u64)

    in0_lo = s0_words[op_sel[0]]
    in1_lo = s1_words[op_sel[1]]
    in0_hi = s0_words[op_sel[2]]
    in1_hi = s1_words[op_sel[3]]

    out_lo = oracle_max_f32(
        in0_lo, in1_lo,
        abs0=abs0_lo, neg0=neg0_lo, abs1=abs1_lo, neg1=neg1_lo,
        clamp=clamp,
    )
    out_hi = oracle_max_f32(
        in0_hi, in1_hi,
        abs0=abs0_hi, neg0=neg0_hi, abs1=abs1_hi, neg1=neg1_hi,
        clamp=clamp,
    )

    return pack_pk_f32(out_lo, out_hi)


# ==============================================================================
# Commit 1 Unit Tests
# ==============================================================================

class TestF32ModelAndOracle(unittest.TestCase):
    """Unit tests for single-precision bit representation, predicates, and reference oracles."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1048576.0, -1048576.0]
        for val in values:
            with self.subTest(val=val):
                bits = f32_to_bits(val)
                recovered = bits_to_f32(bits)
                self.assertEqual(val, recovered)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_word_packing_and_unpacking(self) -> None:
        lo = 0x12345678
        hi = 0x9ABCDEF0
        packed = pack_pk_f32(lo, hi)
        self.assertEqual(packed, (hi << 32) | lo)
        rec_lo, rec_hi = unpack_pk_f32(packed)
        self.assertEqual(rec_lo, lo)
        self.assertEqual(rec_hi, hi)

    def test_classification_predicates(self) -> None:
        self.assertTrue(is_zero32(POS_ZERO))
        self.assertTrue(is_zero32(NEG_ZERO))
        self.assertFalse(is_zero32(POS_ONE))

        self.assertTrue(is_inf32(POS_INF))
        self.assertTrue(is_inf32(NEG_INF))
        self.assertFalse(is_inf32(POS_ZERO))
        self.assertFalse(is_inf32(QNAN_CANONICAL))

        self.assertTrue(is_nan32(QNAN_CANONICAL))
        self.assertTrue(is_qnan32(QNAN_CANONICAL))
        self.assertFalse(is_snan32(QNAN_CANONICAL))

        self.assertTrue(is_nan32(SNAN_DEFAULT))
        self.assertTrue(is_snan32(SNAN_DEFAULT))
        self.assertFalse(is_qnan32(SNAN_DEFAULT))

        self.assertTrue(is_subnormal32(POS_MIN_SUBNORM))
        self.assertTrue(is_subnormal32(POS_MAX_SUBNORM))
        self.assertFalse(is_subnormal32(POS_ZERO))
        self.assertFalse(is_subnormal32(POS_MIN_NORMAL))

    def test_exact_finite_min_max(self) -> None:
        self.assertEqual(oracle_min_f32(POS_ONE, POS_TWO), POS_ONE)
        self.assertEqual(oracle_max_f32(POS_ONE, POS_TWO), POS_TWO)

        self.assertEqual(oracle_min_f32(NEG_ONE, POS_HALF), NEG_ONE)
        self.assertEqual(oracle_max_f32(NEG_ONE, POS_HALF), POS_HALF)

    def test_signed_zero_tie_breaking(self) -> None:
        self.assertEqual(oracle_min_f32(NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(oracle_min_f32(POS_ZERO, NEG_ZERO), NEG_ZERO)

        self.assertEqual(oracle_max_f32(NEG_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_max_f32(POS_ZERO, NEG_ZERO), POS_ZERO)

        self.assertEqual(oracle_min_f32(NEG_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(oracle_max_f32(POS_ZERO, POS_ZERO), POS_ZERO)

    def test_nan_suppression_and_propagation(self) -> None:
        self.assertEqual(oracle_min_f32(QNAN_CANONICAL, POS_TWO), POS_TWO)
        self.assertEqual(oracle_min_f32(POS_TWO, QNAN_CANONICAL), POS_TWO)
        self.assertEqual(oracle_max_f32(QNAN_CANONICAL, POS_TWO), POS_TWO)
        self.assertEqual(oracle_max_f32(POS_TWO, QNAN_CANONICAL), POS_TWO)

        both_min = oracle_min_f32(QNAN_CANONICAL, SNAN_DEFAULT)
        self.assertTrue(is_qnan32(both_min))
        both_max = oracle_max_f32(SNAN_DEFAULT, QNAN_CANONICAL)
        self.assertTrue(is_qnan32(both_max))

    def test_modifiers_and_clamp(self) -> None:
        res_abs = oracle_min_f32(NEG_TWO, POS_ONE, abs0=True)
        self.assertEqual(res_abs, POS_ONE)

        res_neg = oracle_max_f32(POS_TWO, NEG_ONE, neg0=True)
        self.assertEqual(res_neg, NEG_ONE)

        self.assertEqual(oracle_max_f32(POS_TWO, POS_HALF, clamp=True), POS_ONE)
        self.assertEqual(oracle_min_f32(NEG_ONE, NEG_TWO, clamp=True), POS_ZERO)

    def test_packed_oracles_basic(self) -> None:
        s0 = pack_pk_f32(NEG_ONE, POS_TWO)
        s1 = pack_pk_f32(POS_ONE, POS_HALF)

        pk_min = oracle_pk_min_f32(s0, s1)
        exp_min = pack_pk_f32(NEG_ONE, POS_HALF)
        self.assertEqual(pk_min, exp_min)

        pk_max = oracle_pk_max_f32(s0, s1)
        exp_max = pack_pk_f32(POS_ONE, POS_TWO)
        self.assertEqual(pk_max, exp_max)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegKind(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class RegPair:
    """Represents a 64-bit architectural register pair (e.g. v[0:1] or s[4:5])."""
    index: int
    kind: RegKind = RegKind.VGPR

    def __post_init__(self) -> None:
        if self.kind == RegKind.VGPR:
            if not (0 <= self.index <= 254):
                raise ValueError(f"VGPR pair base index {self.index} out of range [0, 254]")
        elif self.kind == RegKind.SGPR:
            if not (0 <= self.index <= 104):
                raise ValueError(f"SGPR pair base index {self.index} out of range [0, 104]")
        else:
            raise TypeError(f"Unknown RegKind: {self.kind}")

    @property
    def lo_reg(self) -> int:
        return self.index

    @property
    def hi_reg(self) -> int:
        return self.index + 1

    @property
    def name(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        return f"{prefix}[{self.index}:{self.index + 1}]"


@dataclass(frozen=True)
class Literal64:
    """Represents a 64-bit packed literal or inline constant pair."""
    bits: int

    def __post_init__(self) -> None:
        if not (0 <= self.bits <= 0xFFFF_FFFF_FFFF_FFFF):
            raise ValueError(f"Literal64 bits {self.bits:#x} out of range [0, 0xFFFFFFFFFFFFFFFF]")

    @property
    def lo_bits(self) -> int:
        return self.bits & MASK32

    @property
    def hi_bits(self) -> int:
        return (self.bits >> 32) & MASK32

    @property
    def name(self) -> str:
        return f"{self.bits:#018x}"


class Opcode(Enum):
    V_PK_MIN_F32 = "v_pk_min_f32"
    V_PK_MAX_F32 = "v_pk_max_f32"


@dataclass(frozen=True)
class VOP3POperand:
    """VOP3P packed operand with per-half input modifiers and op_sel."""
    source: Union[RegPair, Literal64]
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
class VPkMinMaxF32Instruction:
    """Symbolic AST node for RDNA2 v_pk_min_f32 and v_pk_max_f32 instructions."""
    opcode: Opcode
    dst: RegPair
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

    def test_regpair_validation_and_naming(self) -> None:
        v0 = RegPair(0, RegKind.VGPR)
        self.assertEqual(v0.name, "v[0:1]")
        self.assertEqual(v0.lo_reg, 0)
        self.assertEqual(v0.hi_reg, 1)

        s4 = RegPair(4, RegKind.SGPR)
        self.assertEqual(s4.name, "s[4:5]")

        with self.assertRaises(ValueError):
            RegPair(255, RegKind.VGPR)
        with self.assertRaises(ValueError):
            RegPair(-1, RegKind.VGPR)
        with self.assertRaises(ValueError):
            RegPair(105, RegKind.SGPR)

    def test_literal64_representation(self) -> None:
        lit = Literal64(0x3F80000040000000)
        self.assertEqual(lit.lo_bits, 0x40000000)
        self.assertEqual(lit.hi_bits, 0x3F800000)

        with self.assertRaises(ValueError):
            Literal64(0x1_0000_0000_0000_0000)
        with self.assertRaises(ValueError):
            Literal64(-1)

    def test_operand_disassembly_modifiers(self) -> None:
        v1 = RegPair(2, RegKind.VGPR)
        op_plain = VOP3POperand(v1)
        self.assertEqual(op_plain.disassemble(), "v[2:3]")

        op_both_abs = VOP3POperand(v1, abs_lo=True, abs_hi=True)
        self.assertEqual(op_both_abs.disassemble(), "|v[2:3]|")

        op_both_neg = VOP3POperand(v1, neg_lo=True, neg_hi=True)
        self.assertEqual(op_both_neg.disassemble(), "-v[2:3]")

        op_custom = VOP3POperand(v1, neg_lo=True, op_sel=(1, 0))
        self.assertIn("neg_lo", op_custom.disassemble())
        self.assertIn("op_sel:[1,0]", op_custom.disassemble())

        with self.assertRaises(ValueError):
            VOP3POperand(v1, op_sel=(0, 2))

    def test_instruction_construction_and_validation(self) -> None:
        dst = RegPair(0, RegKind.VGPR)
        src0 = VOP3POperand(RegPair(2, RegKind.VGPR))
        src1 = VOP3POperand(RegPair(4, RegKind.SGPR))

        inst_min = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MIN_F32,
            dst=dst,
            src0=src0,
            src1=src1,
            clamp=False,
        )
        self.assertEqual(inst_min.disassemble(), "v_pk_min_f32 v[0:1], v[2:3], s[4:5]")

        inst_max_clamp = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MAX_F32,
            dst=dst,
            src0=src0,
            src1=src1,
            clamp=True,
        )
        self.assertEqual(inst_max_clamp.disassemble(), "v_pk_max_f32 v[0:1], v[2:3], s[4:5] clamp")

        # Destination must be VGPR
        with self.assertRaises(TypeError):
            VPkMinMaxF32Instruction(
                opcode=Opcode.V_PK_MIN_F32,
                dst=RegPair(0, RegKind.SGPR),
                src0=src0,
                src1=src1,
            )

    def test_structural_equality(self) -> None:
        dst = RegPair(4, RegKind.VGPR)
        src0 = VOP3POperand(RegPair(6, RegKind.VGPR))
        src1 = VOP3POperand(Literal64(0x3F8000003F800000))
        i1 = VPkMinMaxF32Instruction(Opcode.V_PK_MIN_F32, dst, src0, src1, clamp=True)
        i2 = VPkMinMaxF32Instruction(Opcode.V_PK_MIN_F32, dst, src0, src1, clamp=True)
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

    def get_reg64(self, reg: RegPair) -> int:
        if reg.kind == RegKind.VGPR:
            lo = self.vgprs[reg.index] & MASK32
            hi = self.vgprs[reg.index + 1] & MASK32
            return pack_pk_f32(lo, hi)
        elif reg.kind == RegKind.SGPR:
            lo = self.sgprs[reg.index] & MASK32
            hi = self.sgprs[reg.index + 1] & MASK32
            return pack_pk_f32(lo, hi)
        raise TypeError(f"Unknown RegKind: {reg.kind}")

    def set_reg64(self, reg: RegPair, val: int) -> None:
        lo, hi = unpack_pk_f32(val)
        if reg.kind == RegKind.VGPR:
            self.vgprs[reg.index] = lo
            self.vgprs[reg.index + 1] = hi
        elif reg.kind == RegKind.SGPR:
            self.sgprs[reg.index] = lo
            self.sgprs[reg.index + 1] = hi
        else:
            raise TypeError(f"Unknown RegKind: {reg.kind}")


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for packed single-precision min/max."""

    def resolve_operand(self, op: VOP3POperand, state: ScalarMachineState) -> int:
        if isinstance(op.source, RegPair):
            return state.get_reg64(op.source)
        elif isinstance(op.source, Literal64):
            return op.source.bits
        raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VPkMinMaxF32Instruction, state: ScalarMachineState) -> int:
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)

        op_sel = (
            instr.src0.op_sel[0],
            instr.src1.op_sel[0],
            instr.src0.op_sel[1],
            instr.src1.op_sel[1],
        )

        if instr.opcode == Opcode.V_PK_MIN_F32:
            res = oracle_pk_min_f32(
                s0_raw, s1_raw,
                op_sel=op_sel,
                abs0_lo=instr.src0.abs_lo, neg0_lo=instr.src0.neg_lo,
                abs1_lo=instr.src1.abs_lo, neg1_lo=instr.src1.neg_lo,
                abs0_hi=instr.src0.abs_hi, neg0_hi=instr.src0.neg_hi,
                abs1_hi=instr.src1.abs_hi, neg1_hi=instr.src1.neg_hi,
                clamp=instr.clamp,
            )
        elif instr.opcode == Opcode.V_PK_MAX_F32:
            res = oracle_pk_max_f32(
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

        state.set_reg64(instr.dst, res)
        return res


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_scalar_vgpr_vgpr_pk_min(self) -> None:
        # v[2:3]: (hi=2.0, lo=-1.0), v[4:5]: (hi=0.5, lo=1.0)
        self.state.set_reg64(RegPair(2, RegKind.VGPR), pack_pk_f32(NEG_ONE, POS_TWO))
        self.state.set_reg64(RegPair(4, RegKind.VGPR), pack_pk_f32(POS_ONE, POS_HALF))

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MIN_F32,
            dst=RegPair(0, RegKind.VGPR),
            src0=VOP3POperand(RegPair(2, RegKind.VGPR)),
            src1=VOP3POperand(RegPair(4, RegKind.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        expected = pack_pk_f32(NEG_ONE, POS_HALF)
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_reg64(RegPair(0, RegKind.VGPR)), expected)

    def test_scalar_vgpr_sgpr_pk_max(self) -> None:
        # v[0:1]: (hi=-2.0, lo=0.5), s[4:5]: (hi=1.0, lo=1.0)
        self.state.set_reg64(RegPair(0, RegKind.VGPR), pack_pk_f32(POS_HALF, NEG_TWO))
        self.state.set_reg64(RegPair(4, RegKind.SGPR), pack_pk_f32(POS_ONE, POS_ONE))

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MAX_F32,
            dst=RegPair(6, RegKind.VGPR),
            src0=VOP3POperand(RegPair(0, RegKind.VGPR)),
            src1=VOP3POperand(RegPair(4, RegKind.SGPR)),
        )
        res = self.interp.step(instr, self.state)
        expected = pack_pk_f32(POS_ONE, POS_ONE)
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_reg64(RegPair(6, RegKind.VGPR)), expected)

    def test_scalar_literal_and_modifiers(self) -> None:
        # v[0:1]: (hi=-2.0, lo=-0.5)
        # src0: abs on both -> (hi=2.0, lo=0.5)
        # src1: literal (hi=1.0, lo=1.0)
        self.state.set_reg64(RegPair(0, RegKind.VGPR), pack_pk_f32(NEG_HALF, NEG_TWO))

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MIN_F32,
            dst=RegPair(4, RegKind.VGPR),
            src0=VOP3POperand(RegPair(0, RegKind.VGPR), abs_lo=True, abs_hi=True),
            src1=VOP3POperand(Literal64(pack_pk_f32(POS_ONE, POS_ONE))),
        )
        res = self.interp.step(instr, self.state)
        expected = pack_pk_f32(POS_HALF, POS_ONE)
        self.assertEqual(res, expected)

    def test_scalar_clamp_mode(self) -> None:
        # v[0:1]: (hi=2.0, lo=-1.0), v[2:3]: (hi=3.0, lo=-2.0)
        self.state.set_reg64(RegPair(0, RegKind.VGPR), pack_pk_f32(NEG_ONE, POS_TWO))
        self.state.set_reg64(RegPair(2, RegKind.VGPR), pack_pk_f32(NEG_TWO, f32_to_bits(3.0)))

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MAX_F32,
            dst=RegPair(4, RegKind.VGPR),
            src0=VOP3POperand(RegPair(0, RegKind.VGPR)),
            src1=VOP3POperand(RegPair(2, RegKind.VGPR)),
            clamp=True,
        )
        res = self.interp.step(instr, self.state)
        expected = pack_pk_f32(POS_ZERO, POS_ONE)
        self.assertEqual(res, expected)

    def test_scalar_determinism(self) -> None:
        runs = []
        for _ in range(50):
            st = ScalarMachineState()
            st.set_reg64(RegPair(0, RegKind.VGPR), 0x3F80000040000000)
            st.set_reg64(RegPair(2, RegKind.VGPR), 0x404000003F000000)
            instr = VPkMinMaxF32Instruction(
                opcode=Opcode.V_PK_MIN_F32,
                dst=RegPair(4, RegKind.VGPR),
                src0=VOP3POperand(RegPair(0, RegKind.VGPR)),
                src1=VOP3POperand(RegPair(2, RegKind.VGPR)),
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

    def get_lane_reg64(self, lane: int, reg: RegPair) -> int:
        if reg.kind == RegKind.VGPR:
            lo = self.vgprs[lane][reg.index] & MASK32
            hi = self.vgprs[lane][reg.index + 1] & MASK32
            return pack_pk_f32(lo, hi)
        elif reg.kind == RegKind.SGPR:
            lo = self.sgprs[reg.index] & MASK32
            hi = self.sgprs[reg.index + 1] & MASK32
            return pack_pk_f32(lo, hi)
        raise TypeError(f"Unknown RegKind: {reg.kind}")

    def set_lane_reg64(self, lane: int, reg: RegPair, val: int) -> None:
        if reg.kind != RegKind.VGPR:
            raise TypeError(f"Cannot set lane register for non-VGPR: {reg.kind}")
        lo, hi = unpack_pk_f32(val)
        self.vgprs[lane][reg.index] = lo
        self.vgprs[lane][reg.index + 1] = hi

    def set_sgpr64(self, reg: RegPair, val: int) -> None:
        if reg.kind != RegKind.SGPR:
            raise TypeError(f"Cannot set SGPR on VGPR: {reg.kind}")
        lo, hi = unpack_pk_f32(val)
        self.sgprs[reg.index] = lo
        self.sgprs[reg.index + 1] = hi


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VOP3POperand, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, RegPair):
            return state.get_lane_reg64(lane, op.source)
        elif isinstance(op.source, Literal64):
            return op.source.bits
        raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VPkMinMaxF32Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * state.num_lanes

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

        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                if instr.opcode == Opcode.V_PK_MIN_F32:
                    res = oracle_pk_min_f32(
                        s0_vals[lane], s1_vals[lane],
                        op_sel=op_sel,
                        abs0_lo=instr.src0.abs_lo, neg0_lo=instr.src0.neg_lo,
                        abs1_lo=instr.src1.abs_lo, neg1_lo=instr.src1.neg_lo,
                        abs0_hi=instr.src0.abs_hi, neg0_hi=instr.src0.neg_hi,
                        abs1_hi=instr.src1.abs_hi, neg1_hi=instr.src1.neg_hi,
                        clamp=instr.clamp,
                    )
                elif instr.opcode == Opcode.V_PK_MAX_F32:
                    res = oracle_pk_max_f32(
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

                state.set_lane_reg64(lane, instr.dst, res)
                results[lane] = res

        return results


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD execution, lane divergence, and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_wave32_full_exec(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            v0_val = pack_pk_f32(f32_to_bits(float(lane)), f32_to_bits(float(lane) * 2.0))
            v1_val = pack_pk_f32(f32_to_bits(10.0), f32_to_bits(15.0))
            state.set_lane_reg64(lane, RegPair(0, RegKind.VGPR), v0_val)
            state.set_lane_reg64(lane, RegPair(2, RegKind.VGPR), v1_val)

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MIN_F32,
            dst=RegPair(4, RegKind.VGPR),
            src0=VOP3POperand(RegPair(0, RegKind.VGPR)),
            src1=VOP3POperand(RegPair(2, RegKind.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            lo_exp = f32_to_bits(min(float(lane), 10.0))
            hi_exp = f32_to_bits(min(float(lane) * 2.0, 15.0))
            expected = pack_pk_f32(lo_exp, hi_exp)
            self.assertEqual(state.get_lane_reg64(lane, RegPair(4, RegKind.VGPR)), expected)

    def test_wave32_divergent_masking(self) -> None:
        sentinel = 0xDEADBEEFCAFEBABE
        state = SIMDState(num_lanes=32, exec_mask=0x5555_5555)
        for lane in range(32):
            state.set_lane_reg64(lane, RegPair(0, RegKind.VGPR), pack_pk_f32(POS_ONE, POS_TWO))
            state.set_lane_reg64(lane, RegPair(2, RegKind.VGPR), pack_pk_f32(POS_TWO, POS_ONE))
            state.set_lane_reg64(lane, RegPair(4, RegKind.VGPR), sentinel)

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MAX_F32,
            dst=RegPair(4, RegKind.VGPR),
            src0=VOP3POperand(RegPair(0, RegKind.VGPR)),
            src1=VOP3POperand(RegPair(2, RegKind.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertIsNotNone(results[lane])
                self.assertEqual(
                    state.get_lane_reg64(lane, RegPair(4, RegKind.VGPR)),
                    pack_pk_f32(POS_TWO, POS_TWO),
                )
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_reg64(lane, RegPair(4, RegKind.VGPR)), sentinel)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_lane_reg64(lane, RegPair(0, RegKind.VGPR), pack_pk_f32(POS_ONE, NEG_ONE))
            state.set_lane_reg64(lane, RegPair(2, RegKind.VGPR), pack_pk_f32(POS_TWO, NEG_TWO))

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MIN_F32,
            dst=RegPair(0, RegKind.VGPR),
            src0=VOP3POperand(RegPair(0, RegKind.VGPR)),
            src1=VOP3POperand(RegPair(2, RegKind.VGPR)),
        )
        self.interp.step(instr, state)
        for lane in range(64):
            self.assertEqual(
                state.get_lane_reg64(lane, RegPair(0, RegKind.VGPR)),
                pack_pk_f32(POS_ONE, NEG_TWO),
            )

    def test_wave64_half_disabled(self) -> None:
        sentinel = 0xCAFEBABEDEADBEEF
        state = SIMDState(num_lanes=64, exec_mask=0x0000_0000_FFFF_FFFF)
        for lane in range(64):
            state.set_lane_reg64(lane, RegPair(0, RegKind.VGPR), pack_pk_f32(POS_HALF, POS_ONE))
            state.set_lane_reg64(lane, RegPair(2, RegKind.VGPR), pack_pk_f32(POS_ONE, POS_HALF))
            state.set_lane_reg64(lane, RegPair(6, RegKind.VGPR), sentinel)

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MAX_F32,
            dst=RegPair(6, RegKind.VGPR),
            src0=VOP3POperand(RegPair(0, RegKind.VGPR)),
            src1=VOP3POperand(RegPair(2, RegKind.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(64):
            if lane < 32:
                self.assertIsNotNone(results[lane])
                self.assertEqual(
                    state.get_lane_reg64(lane, RegPair(6, RegKind.VGPR)),
                    pack_pk_f32(POS_ONE, POS_ONE),
                )
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_reg64(lane, RegPair(6, RegKind.VGPR)), sentinel)

    def test_wave_exec_all_zero(self) -> None:
        sentinel = 0xAA55AA5555AA55AA
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)
        for lane in range(32):
            state.set_lane_reg64(lane, RegPair(0, RegKind.VGPR), sentinel)

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MIN_F32,
            dst=RegPair(0, RegKind.VGPR),
            src0=VOP3POperand(Literal64(pack_pk_f32(POS_ONE, POS_ONE))),
            src1=VOP3POperand(Literal64(pack_pk_f32(POS_TWO, POS_TWO))),
        )
        results = self.interp.step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(state.get_lane_reg64(lane, RegPair(0, RegKind.VGPR)), sentinel)

    def test_simd_sgpr_uniform_broadcast(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        state.set_sgpr64(RegPair(4, RegKind.SGPR), pack_pk_f32(POS_TWO, POS_TWO))
        for lane in range(32):
            state.set_lane_reg64(lane, RegPair(0, RegKind.VGPR), pack_pk_f32(POS_ONE, POS_ONE))

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MAX_F32,
            dst=RegPair(2, RegKind.VGPR),
            src0=VOP3POperand(RegPair(0, RegKind.VGPR)),
            src1=VOP3POperand(RegPair(4, RegKind.SGPR)),
        )
        self.interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(
                state.get_lane_reg64(lane, RegPair(2, RegKind.VGPR)),
                pack_pk_f32(POS_TWO, POS_TWO),
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
        # v[0:1]: (hi=2.0, lo=-1.0), v[2:3]: (hi=0.5, lo=1.0)
        state.set_reg64(RegPair(0, RegKind.VGPR), pack_pk_f32(NEG_ONE, POS_TWO))
        state.set_reg64(RegPair(2, RegKind.VGPR), pack_pk_f32(POS_ONE, POS_HALF))

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MIN_F32,
            dst=RegPair(0, RegKind.VGPR),
            src0=VOP3POperand(RegPair(0, RegKind.VGPR)),
            src1=VOP3POperand(RegPair(2, RegKind.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        expected = pack_pk_f32(NEG_ONE, POS_HALF)
        self.assertEqual(res, expected)
        self.assertEqual(state.get_reg64(RegPair(0, RegKind.VGPR)), expected)
        # src1 (v[2:3]) must remain untouched
        self.assertEqual(state.get_reg64(RegPair(2, RegKind.VGPR)), pack_pk_f32(POS_ONE, POS_HALF))

    def test_dst_alias_src1(self) -> None:
        state = ScalarMachineState()
        state.set_reg64(RegPair(0, RegKind.VGPR), pack_pk_f32(NEG_ONE, POS_TWO))
        state.set_reg64(RegPair(2, RegKind.VGPR), pack_pk_f32(POS_ONE, POS_HALF))

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MIN_F32,
            dst=RegPair(2, RegKind.VGPR),
            src0=VOP3POperand(RegPair(0, RegKind.VGPR)),
            src1=VOP3POperand(RegPair(2, RegKind.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        expected = pack_pk_f32(NEG_ONE, POS_HALF)
        self.assertEqual(res, expected)
        self.assertEqual(state.get_reg64(RegPair(2, RegKind.VGPR)), expected)
        # src0 (v[0:1]) must remain untouched
        self.assertEqual(state.get_reg64(RegPair(0, RegKind.VGPR)), pack_pk_f32(NEG_ONE, POS_TWO))

    def test_dst_alias_both_srcs(self) -> None:
        state = ScalarMachineState()
        val = pack_pk_f32(POS_HALF, NEG_TWO)
        state.set_reg64(RegPair(0, RegKind.VGPR), val)

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MAX_F32,
            dst=RegPair(0, RegKind.VGPR),
            src0=VOP3POperand(RegPair(0, RegKind.VGPR)),
            src1=VOP3POperand(RegPair(0, RegKind.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, val)
        self.assertEqual(state.get_reg64(RegPair(0, RegKind.VGPR)), val)

    def test_inplace_iterative_accumulation(self) -> None:
        state = ScalarMachineState()
        state.set_reg64(RegPair(0, RegKind.VGPR), pack_pk_f32(f32_to_bits(100.0), f32_to_bits(100.0)))

        for i in range(50):
            step_val = float(50 - i)
            state.set_reg64(RegPair(2, RegKind.VGPR), pack_pk_f32(f32_to_bits(step_val), f32_to_bits(step_val * 2.0)))
            instr = VPkMinMaxF32Instruction(
                opcode=Opcode.V_PK_MIN_F32,
                dst=RegPair(0, RegKind.VGPR),
                src0=VOP3POperand(RegPair(0, RegKind.VGPR)),
                src1=VOP3POperand(RegPair(2, RegKind.VGPR)),
            )
            self.scalar_interp.step(instr, state)

        final_val = state.get_reg64(RegPair(0, RegKind.VGPR))
        lo, hi = unpack_pk_f32(final_val)
        self.assertEqual(lo, f32_to_bits(1.0))
        self.assertEqual(hi, f32_to_bits(2.0))

    def test_inplace_cross_half_swizzle_alias(self) -> None:
        state = ScalarMachineState()
        state.set_reg64(RegPair(0, RegKind.VGPR), pack_pk_f32(f32_to_bits(5.0), f32_to_bits(2.0)))

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MIN_F32,
            dst=RegPair(0, RegKind.VGPR),
            src0=VOP3POperand(RegPair(0, RegKind.VGPR), op_sel=(0, 0)),
            src1=VOP3POperand(RegPair(0, RegKind.VGPR), op_sel=(1, 1)),
        )
        res = self.scalar_interp.step(instr, state)
        expected = pack_pk_f32(f32_to_bits(2.0), f32_to_bits(2.0))
        self.assertEqual(res, expected)
        self.assertEqual(state.get_reg64(RegPair(0, RegKind.VGPR)), expected)

    def test_simd_multi_lane_aliasing(self) -> None:
        sentinel = 0x55AA55AA55AA55AA
        state = SIMDState(num_lanes=32, exec_mask=0x0000_FFFF)
        for lane in range(32):
            state.set_lane_reg64(lane, RegPair(0, RegKind.VGPR), pack_pk_f32(POS_ONE, POS_ONE))
            state.set_lane_reg64(lane, RegPair(2, RegKind.VGPR), pack_pk_f32(POS_TWO, POS_TWO))
            state.set_lane_reg64(lane, RegPair(4, RegKind.VGPR), sentinel)

        instr = VPkMinMaxF32Instruction(
            opcode=Opcode.V_PK_MAX_F32,
            dst=RegPair(0, RegKind.VGPR),
            src0=VOP3POperand(RegPair(0, RegKind.VGPR)),
            src1=VOP3POperand(RegPair(2, RegKind.VGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            if lane < 16:
                self.assertEqual(
                    state.get_lane_reg64(lane, RegPair(0, RegKind.VGPR)),
                    pack_pk_f32(POS_TWO, POS_TWO),
                )
            else:
                self.assertEqual(
                    state.get_lane_reg64(lane, RegPair(0, RegKind.VGPR)),
                    pack_pk_f32(POS_ONE, POS_ONE),
                )
            self.assertEqual(state.get_lane_reg64(lane, RegPair(2, RegKind.VGPR)), pack_pk_f32(POS_TWO, POS_TWO))
            self.assertEqual(state.get_lane_reg64(lane, RegPair(4, RegKind.VGPR)), sentinel)


if __name__ == "__main__":
    unittest.main()
