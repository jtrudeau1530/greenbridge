from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_pk_fma_f16.

Tests the packed 16-bit floating-point fused multiply-add instruction:
    v_pk_fma_f16 D, S0, S1, S2 [op_sel] [op_sel_hi] [neg_lo] [neg_hi] [clamp]
Architecture: VOP3P (GFX10 / RDNA2).
Lowered semantics:
    D.lo = (S0_selected_lo * S1_selected_lo) + S2_selected_lo
    D.hi = (S0_selected_hi * S1_selected_hi) + S2_selected_hi
where:
    - op_sel[i] selects source half for lo computation (0 = lo half [15:0], 1 = hi half [31:16])
    - op_sel_hi[i] selects source half for hi computation (0 = lo half [15:0], 1 = hi half [31:16])
    - neg_lo[i] negates operand i for lo computation
    - neg_hi[i] negates operand i for hi computation
    - clamp saturates each 16-bit result to [0.0, 1.0], with NaN clamped to +0.0
Each lane computes fused multiply-add with single rounding per IEEE 754-2008 binary16.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
import math
import struct
import unittest
from typing import Dict, List, Optional, Tuple, Union

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
POS_MAX_SUBNORM: int = 0x03FF   # largest positive subnormal ((1023/1024) * 2^-14)
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


def split_f16_bits(bits: int) -> Tuple[int, int, int]:
    """Split 16-bit float bits into (sign, exponent, fraction)."""
    b = bits & MASK16
    sign = (b >> 15) & 1
    exp = (b >> 10) & 0x1F
    frac = b & FRAC_MASK16
    return sign, exp, frac


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


def apply_f16_modifiers(bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply half-float input modifiers (abs, neg)."""
    res = bits & MASK16
    if abs_mod:
        res &= ~SIGN_MASK16
    if neg_mod:
        res ^= SIGN_MASK16
    return res


def apply_f16_clamp(val: float) -> float:
    """Apply RDNA2 clamp modifier [0.0, 1.0] to float, mapping NaNs to +0.0."""
    if math.isnan(val):
        return 0.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


# ==============================================================================
# Reference Mathematical Oracles for Scalar & Packed f16 FMA
# ==============================================================================

def raw_fma_f16_float(v0: float, v1: float, v2: float) -> float:
    """Compute fused multiply-add (v0 * v1) + v2 with exact mathematical single rounding."""
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

    # High precision exact arithmetic for single rounding to binary16
    frac = Fraction(v0) * Fraction(v1) + Fraction(v2)
    if frac == 0:
        return 0.0
    try:
        return float(frac)
    except OverflowError:
        return math.copysign(float("inf"), 1.0 if frac > 0 else -1.0)


def compute_scalar_fma_f16_bits(
    s0_bits: int,
    s1_bits: int,
    s2_bits: int,
    clamp: bool = False,
) -> int:
    """Compute exact IEEE 754 binary16 FMA on 16-bit integer patterns."""
    # Check for signaling NaNs first -> quieten
    if is_snan16(s0_bits):
        return (s0_bits | QNAN_BIT16) if not clamp else POS_ZERO
    if is_snan16(s1_bits):
        return (s1_bits | QNAN_BIT16) if not clamp else POS_ZERO
    if is_snan16(s2_bits):
        return (s2_bits | QNAN_BIT16) if not clamp else POS_ZERO

    # Quiet NaNs propagate payload
    if is_qnan16(s0_bits):
        return s0_bits if not clamp else POS_ZERO
    if is_qnan16(s1_bits):
        return s1_bits if not clamp else POS_ZERO
    if is_qnan16(s2_bits):
        return s2_bits if not clamp else POS_ZERO

    v0 = bits_to_f16(s0_bits)
    v1 = bits_to_f16(s1_bits)
    v2 = bits_to_f16(s2_bits)

    res_f = raw_fma_f16_float(v0, v1, v2)

    if clamp:
        res_f = apply_f16_clamp(res_f)
        if math.isnan(res_f):
            return POS_ZERO

    return f16_to_bits(res_f)


def oracle_v_pk_fma_f16(
    src0_bits: int,
    src1_bits: int,
    src2_bits: int,
    op_sel: Tuple[int, int, int] = (0, 0, 0),
    op_sel_hi: Tuple[int, int, int] = (1, 1, 1),
    neg_lo: Tuple[int, int, int] = (0, 0, 0),
    neg_hi: Tuple[int, int, int] = (0, 0, 0),
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_pk_fma_f16 packed instruction."""
    s0_lo, s0_hi = unpack_pk_f16(src0_bits)
    s1_lo, s1_hi = unpack_pk_f16(src1_bits)
    s2_lo, s2_hi = unpack_pk_f16(src2_bits)

    # Route low-half inputs according to op_sel and neg_lo
    in0_lo = apply_f16_modifiers(s0_hi if op_sel[0] else s0_lo, neg_mod=bool(neg_lo[0]))
    in1_lo = apply_f16_modifiers(s1_hi if op_sel[1] else s1_lo, neg_mod=bool(neg_lo[1]))
    in2_lo = apply_f16_modifiers(s2_hi if op_sel[2] else s2_lo, neg_mod=bool(neg_lo[2]))

    # Route high-half inputs according to op_sel_hi and neg_hi
    in0_hi = apply_f16_modifiers(s0_hi if op_sel_hi[0] else s0_lo, neg_mod=bool(neg_hi[0]))
    in1_hi = apply_f16_modifiers(s1_hi if op_sel_hi[1] else s1_lo, neg_mod=bool(neg_hi[1]))
    in2_hi = apply_f16_modifiers(s2_hi if op_sel_hi[2] else s2_lo, neg_mod=bool(neg_hi[2]))

    res_lo = compute_scalar_fma_f16_bits(in0_lo, in1_lo, in2_lo, clamp=clamp)
    res_hi = compute_scalar_fma_f16_bits(in0_hi, in1_hi, in2_hi, clamp=clamp)

    return pack_pk_f16(res_lo, res_hi)


# ==============================================================================
# Architectural Register Model
# ==============================================================================

class PackedVGPR:
    """Architectural model of a 32-bit VGPR storing two packed binary16 floats."""

    def __init__(self, lo: int = 0, hi: int = 0) -> None:
        self.lo = lo & MASK16
        self.hi = hi & MASK16

    @classmethod
    def from_u32(cls, val: int) -> PackedVGPR:
        lo, hi = unpack_pk_f16(val)
        return cls(lo, hi)

    @classmethod
    def from_floats(cls, lo_f: float, hi_f: float) -> PackedVGPR:
        return cls(f16_to_bits(lo_f), f16_to_bits(hi_f))

    @property
    def as_u32(self) -> int:
        return pack_pk_f16(self.lo, self.hi)

    @property
    def lo_float(self) -> float:
        return bits_to_f16(self.lo)

    @property
    def hi_float(self) -> float:
        return bits_to_f16(self.hi)

    def set_u32(self, val: int) -> None:
        self.lo, self.hi = unpack_pk_f16(val)

    def __repr__(self) -> str:
        return (
            f"PackedVGPR(lo={self.lo:#06x} ({self.lo_float:g}), "
            f"hi={self.hi:#06x} ({self.hi_float:g}))"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PackedVGPR):
            return NotImplemented
        return self.lo == other.lo and self.hi == other.hi


# ==============================================================================
# Phase 1 Unit Tests: Mathematical Oracle & Architectural Model
# ==============================================================================

class TestPkFmaF16MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural register model."""

    def test_binary16_bit_conversions_and_predicates(self) -> None:
        test_vals = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 65504.0, -65504.0]
        for val in test_vals:
            with self.subTest(val=val):
                b = f16_to_bits(val)
                rec = bits_to_f16(b)
                self.assertEqual(val, rec)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, rec))

        self.assertTrue(is_zero16(POS_ZERO))
        self.assertTrue(is_zero16(NEG_ZERO))
        self.assertFalse(is_zero16(POS_ONE))

        self.assertTrue(is_inf16(POS_INF))
        self.assertTrue(is_inf16(NEG_INF))
        self.assertFalse(is_inf16(POS_ONE))

        self.assertTrue(is_subnormal16(POS_MIN_SUBNORM))
        self.assertTrue(is_subnormal16(POS_MAX_SUBNORM))
        self.assertFalse(is_subnormal16(POS_MIN_NORMAL))
        self.assertFalse(is_subnormal16(POS_ZERO))

        self.assertTrue(is_nan16(QNAN_CANONICAL))
        self.assertTrue(is_qnan16(QNAN_CANONICAL))
        self.assertFalse(is_snan16(QNAN_CANONICAL))

        self.assertTrue(is_nan16(SNAN_DEFAULT))
        self.assertTrue(is_snan16(SNAN_DEFAULT))
        self.assertFalse(is_qnan16(SNAN_DEFAULT))

    def test_pack_unpack_pk_f16(self) -> None:
        p = pack_pk_f16(POS_ONE, POS_TWO)
        self.assertEqual(p, (POS_TWO << 16) | POS_ONE)
        lo, hi = unpack_pk_f16(p)
        self.assertEqual(lo, POS_ONE)
        self.assertEqual(hi, POS_TWO)

    def test_basic_fma_identities_scalar(self) -> None:
        # 1.0 * 1.0 + 0.0 == 1.0
        self.assertEqual(compute_scalar_fma_f16_bits(POS_ONE, POS_ONE, POS_ZERO), POS_ONE)
        # 2.0 * 3.0 + 4.0 == 10.0
        s3 = f16_to_bits(3.0)
        s4 = f16_to_bits(4.0)
        s10 = f16_to_bits(10.0)
        self.assertEqual(compute_scalar_fma_f16_bits(POS_TWO, s3, s4), s10)
        # 0.5 * 2.0 + (-1.0) == 0.0
        self.assertEqual(compute_scalar_fma_f16_bits(POS_HALF, POS_TWO, NEG_ONE), POS_ZERO)

    def test_packed_fma_dual_lane_independence(self) -> None:
        # lo: 1.0 * 2.0 + 3.0 = 5.0
        # hi: 2.0 * 4.0 + 1.0 = 9.0
        v0 = pack_pk_f16(POS_ONE, POS_TWO)
        v1 = pack_pk_f16(POS_TWO, f16_to_bits(4.0))
        v2 = pack_pk_f16(f16_to_bits(3.0), POS_ONE)
        expected_lo = f16_to_bits(5.0)
        expected_hi = f16_to_bits(9.0)
        res = oracle_v_pk_fma_f16(v0, v1, v2)
        r_lo, r_hi = unpack_pk_f16(res)
        self.assertEqual(r_lo, expected_lo)
        self.assertEqual(r_hi, expected_hi)

    def test_op_sel_and_op_sel_hi_half_selection(self) -> None:
        # v0 has lo = 1.0, hi = 2.0
        # v1 has lo = 3.0, hi = 4.0
        # v2 has lo = 0.5, hi = 1.5
        v0 = pack_pk_f16(POS_ONE, POS_TWO)
        v1 = pack_pk_f16(f16_to_bits(3.0), f16_to_bits(4.0))
        v2 = pack_pk_f16(POS_HALF, f16_to_bits(1.5))

        # Default: lo takes (s0.lo * s1.lo + s2.lo) = 1*3 + 0.5 = 3.5
        #          hi takes (s0.hi * s1.hi + s2.hi) = 2*4 + 1.5 = 9.5
        res_def = oracle_v_pk_fma_f16(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 1, 1))
        self.assertEqual(unpack_pk_f16(res_def), (f16_to_bits(3.5), f16_to_bits(9.5)))

        # Swapped op_sel: lo takes s0.hi, s1.lo, s2.hi = 2*3 + 1.5 = 7.5
        # Swapped op_sel_hi: hi takes s0.lo, s1.hi, s2.lo = 1*4 + 0.5 = 4.5
        res_swap = oracle_v_pk_fma_f16(v0, v1, v2, op_sel=(1, 0, 1), op_sel_hi=(0, 1, 0))
        self.assertEqual(unpack_pk_f16(res_swap), (f16_to_bits(7.5), f16_to_bits(4.5)))

    def test_neg_lo_and_neg_hi_modifiers(self) -> None:
        v0 = pack_pk_f16(POS_ONE, POS_TWO)
        v1 = pack_pk_f16(POS_TWO, POS_TWO)
        v2 = pack_pk_f16(f16_to_bits(5.0), f16_to_bits(5.0))

        # lo: -(1.0) * 2.0 + 5.0 = 3.0
        # hi: 2.0 * -(2.0) + 5.0 = 1.0
        res = oracle_v_pk_fma_f16(
            v0, v1, v2,
            op_sel=(0, 0, 0), op_sel_hi=(1, 1, 1),
            neg_lo=(1, 0, 0), neg_hi=(0, 1, 0),
        )
        self.assertEqual(unpack_pk_f16(res), (f16_to_bits(3.0), POS_ONE))

    def test_clamp_modifier(self) -> None:
        # lo: 1.0 * 2.0 + 0.0 = 2.0 -> clamp -> 1.0
        # hi: -2.0 * 1.0 + 0.0 = -2.0 -> clamp -> +0.0
        v0 = pack_pk_f16(POS_ONE, NEG_TWO)
        v1 = pack_pk_f16(POS_TWO, POS_ONE)
        v2 = pack_pk_f16(POS_ZERO, POS_ZERO)
        res = oracle_v_pk_fma_f16(v0, v1, v2, clamp=True)
        self.assertEqual(unpack_pk_f16(res), (POS_ONE, POS_ZERO))

    def test_signed_zero_identities(self) -> None:
        # (+0) * (+0) + (+0) = +0
        res_pz = compute_scalar_fma_f16_bits(POS_ZERO, POS_ZERO, POS_ZERO)
        self.assertEqual(res_pz, POS_ZERO)

        # (-0) * (+1) + (-0) = -0
        res_nz = compute_scalar_fma_f16_bits(NEG_ZERO, POS_ONE, NEG_ZERO)
        self.assertEqual(res_nz, NEG_ZERO)

    def test_infinity_identities(self) -> None:
        # +inf * 1.0 + 0.0 = +inf
        self.assertEqual(compute_scalar_fma_f16_bits(POS_INF, POS_ONE, POS_ZERO), POS_INF)
        # +inf * 0.0 + 1.0 = NaN
        self.assertTrue(is_nan16(compute_scalar_fma_f16_bits(POS_INF, POS_ZERO, POS_ONE)))
        # +inf * 1.0 + (-inf) = NaN
        self.assertTrue(is_nan16(compute_scalar_fma_f16_bits(POS_INF, POS_ONE, NEG_INF)))

    def test_nan_quietening_and_propagation(self) -> None:
        # Signaling NaN quietened to QNaN
        res = compute_scalar_fma_f16_bits(SNAN_DEFAULT, POS_ONE, POS_ZERO)
        self.assertTrue(is_qnan16(res))
        self.assertEqual(res, SNAN_DEFAULT | QNAN_BIT16)

    def test_architectural_packed_vgpr_model(self) -> None:
        reg = PackedVGPR.from_floats(1.5, -2.5)
        self.assertAlmostEqual(reg.lo_float, 1.5)
        self.assertAlmostEqual(reg.hi_float, -2.5)
        u32_val = reg.as_u32
        reg2 = PackedVGPR.from_u32(u32_val)
        self.assertEqual(reg, reg2)
        self.assertEqual(reg.lo, f16_to_bits(1.5))
        self.assertEqual(reg.hi, f16_to_bits(-2.5))


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Reg32:
    """Represents a 32-bit architectural register (VGPR or SGPR)."""
    index: int
    reg_type: RegType

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index {self.index} out of range [0, 255]")

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class Literal32:
    """Represents a 32-bit literal constant (packed f16x2 or raw bits)."""
    bits: int

    def __post_init__(self) -> None:
        if not (0 <= self.bits <= MASK32):
            raise ValueError(f"Literal constant {self.bits:#x} out of range [0, 0xFFFFFFFF]")

    @classmethod
    def from_floats(cls, lo: float, hi: float) -> Literal32:
        return cls(pack_pk_f16(f16_to_bits(lo), f16_to_bits(hi)))

    @classmethod
    def from_bits(cls, lo: int, hi: int) -> Literal32:
        return cls(pack_pk_f16(lo, hi))

    @property
    def name(self) -> str:
        lo, hi = unpack_pk_f16(self.bits)
        return f"[{bits_to_f16(lo):g}, {bits_to_f16(hi):g}] ({self.bits:#010x})"


@dataclass(frozen=True)
class VOP3POperand:
    """Operand in a VOP3P instruction (Reg32 or Literal32)."""
    source: Union[Reg32, Literal32]

    def disassemble(self) -> str:
        if isinstance(self.source, Reg32):
            return self.source.name
        return f"{self.source.bits:#010x}"


@dataclass(frozen=True)
class VPkFmaF16Instruction:
    """Symbolic AST node representing a RDNA2 v_pk_fma_f16 instruction."""
    dst: Reg32
    src0: VOP3POperand
    src1: VOP3POperand
    src2: VOP3POperand
    op_sel: Tuple[int, int, int] = (0, 0, 0)
    op_sel_hi: Tuple[int, int, int] = (1, 1, 1)
    neg_lo: Tuple[int, int, int] = (0, 0, 0)
    neg_hi: Tuple[int, int, int] = (0, 0, 0)
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")
        for name, tup in (
            ("op_sel", self.op_sel),
            ("op_sel_hi", self.op_sel_hi),
            ("neg_lo", self.neg_lo),
            ("neg_hi", self.neg_hi),
        ):
            if len(tup) != 3:
                raise ValueError(f"{name} must have 3 elements, got {len(tup)}")
            if any(x not in (0, 1) for x in tup):
                raise ValueError(f"{name} elements must be 0 or 1, got {tup}")

    def disassemble(self) -> str:
        s = (
            f"v_pk_fma_f16 {self.dst.name}, {self.src0.disassemble()}, "
            f"{self.src1.disassemble()}, {self.src2.disassemble()}"
        )
        if self.op_sel != (0, 0, 0):
            s += f" op_sel:[{self.op_sel[0]},{self.op_sel[1]},{self.op_sel[2]}]"
        if self.op_sel_hi != (1, 1, 1):
            s += f" op_sel_hi:[{self.op_sel_hi[0]},{self.op_sel_hi[1]},{self.op_sel_hi[2]}]"
        if any(self.neg_lo):
            s += f" neg_lo:[{self.neg_lo[0]},{self.neg_lo[1]},{self.neg_lo[2]}]"
        if any(self.neg_hi):
            s += f" neg_hi:[{self.neg_hi[0]},{self.neg_hi[1]},{self.neg_hi[2]}]"
        if self.clamp:
            s += " clamp"
        return s


# Typed Expressions for Symbolic Execution
class TypedExpr:
    """Abstract base class for typed expressions."""
    def eval(self, env: Dict[str, int]) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class VariableExpr(TypedExpr):
    """Named variable evaluating to an architectural register or value in env."""
    name: str

    def eval(self, env: Dict[str, int]) -> int:
        if self.name not in env:
            raise KeyError(f"Variable '{self.name}' not found in environment")
        return env[self.name] & MASK32


@dataclass(frozen=True)
class ConstantExpr(TypedExpr):
    """Constant 32-bit packed float literal."""
    value: int

    def eval(self, env: Dict[str, int]) -> int:
        return self.value & MASK32


@dataclass(frozen=True)
class PackedF16FMAExpr(TypedExpr):
    """Symbolic typed expression for packed f16 FMA operation."""
    src0: TypedExpr
    src1: TypedExpr
    src2: TypedExpr
    op_sel: Tuple[int, int, int] = (0, 0, 0)
    op_sel_hi: Tuple[int, int, int] = (1, 1, 1)
    neg_lo: Tuple[int, int, int] = (0, 0, 0)
    neg_hi: Tuple[int, int, int] = (0, 0, 0)
    clamp: bool = False

    def eval(self, env: Dict[str, int]) -> int:
        v0 = self.src0.eval(env)
        v1 = self.src1.eval(env)
        v2 = self.src2.eval(env)
        return oracle_v_pk_fma_f16(
            v0, v1, v2,
            op_sel=self.op_sel,
            op_sel_hi=self.op_sel_hi,
            neg_lo=self.neg_lo,
            neg_hi=self.neg_hi,
            clamp=self.clamp,
        )


class TestPkFmaF16SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_reg32_naming_and_bounds(self) -> None:
        v0 = Reg32(0, RegType.VGPR)
        self.assertEqual(v0.name, "v0")
        self.assertEqual(v0.index, 0)

        s255 = Reg32(255, RegType.SGPR)
        self.assertEqual(s255.name, "s255")

        with self.assertRaises(ValueError):
            Reg32(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(256, RegType.SGPR)

    def test_literal32_representation(self) -> None:
        lit = Literal32.from_floats(1.0, 2.0)
        self.assertEqual(lit.bits, pack_pk_f16(POS_ONE, POS_TWO))
        self.assertIn("0x40003c00", lit.name)

        with self.assertRaises(ValueError):
            Literal32(-1)
        with self.assertRaises(ValueError):
            Literal32(0x1_0000_0000)

    def test_vop3p_operand_disassembly(self) -> None:
        op_v = VOP3POperand(Reg32(5, RegType.VGPR))
        self.assertEqual(op_v.disassemble(), "v5")

        op_s = VOP3POperand(Reg32(12, RegType.SGPR))
        self.assertEqual(op_s.disassemble(), "s12")

        op_lit = VOP3POperand(Literal32.from_floats(1.0, 1.0))
        self.assertEqual(op_lit.disassemble(), f"{pack_pk_f16(POS_ONE, POS_ONE):#010x}")

    def test_instruction_disassembly(self) -> None:
        instr_def = VPkFmaF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(1, RegType.VGPR)),
            src1=VOP3POperand(Reg32(2, RegType.VGPR)),
            src2=VOP3POperand(Reg32(3, RegType.VGPR)),
        )
        self.assertEqual(instr_def.disassemble(), "v_pk_fma_f16 v0, v1, v2, v3")

        instr_mods = VPkFmaF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(1, RegType.VGPR)),
            src1=VOP3POperand(Reg32(2, RegType.SGPR)),
            src2=VOP3POperand(Literal32(0)),
            op_sel=(1, 0, 1),
            op_sel_hi=(0, 1, 0),
            neg_lo=(1, 0, 0),
            neg_hi=(0, 1, 1),
            clamp=True,
        )
        dis = instr_mods.disassemble()
        self.assertIn("op_sel:[1,0,1]", dis)
        self.assertIn("op_sel_hi:[0,1,0]", dis)
        self.assertIn("neg_lo:[1,0,0]", dis)
        self.assertIn("neg_hi:[0,1,1]", dis)
        self.assertTrue(dis.endswith("clamp"))

    def test_instruction_validation(self) -> None:
        v = VOP3POperand(Reg32(1, RegType.VGPR))
        # Destination must be VGPR
        with self.assertRaises(TypeError):
            VPkFmaF16Instruction(dst=Reg32(0, RegType.SGPR), src0=v, src1=v, src2=v)

        # Invalid tuple lengths
        with self.assertRaises(ValueError):
            VPkFmaF16Instruction(
                dst=Reg32(0, RegType.VGPR), src0=v, src1=v, src2=v,
                op_sel=(0, 0),  # type: ignore[arg-type]
            )

        # Invalid tuple element values
        with self.assertRaises(ValueError):
            VPkFmaF16Instruction(
                dst=Reg32(0, RegType.VGPR), src0=v, src1=v, src2=v,
                neg_lo=(0, 2, 0),
            )

    def test_structural_equality_and_immutability(self) -> None:
        v1 = VOP3POperand(Reg32(1, RegType.VGPR))
        v2 = VOP3POperand(Reg32(2, RegType.VGPR))
        v3 = VOP3POperand(Reg32(3, RegType.VGPR))
        i1 = VPkFmaF16Instruction(Reg32(0, RegType.VGPR), v1, v2, v3, clamp=True)
        i2 = VPkFmaF16Instruction(Reg32(0, RegType.VGPR), v1, v2, v3, clamp=True)
        self.assertEqual(i1, i2)
        self.assertEqual(hash(i1), hash(i2))

    def test_typed_expressions_evaluation(self) -> None:
        env = {
            "v0": pack_pk_f16(POS_ONE, POS_TWO),
            "v1": pack_pk_f16(POS_TWO, POS_TWO),
            "v2": pack_pk_f16(POS_HALF, POS_HALF),
        }
        # expr: v0 * v1 + v2
        # lo: 1*2 + 0.5 = 2.5
        # hi: 2*2 + 0.5 = 4.5
        fma_expr = PackedF16FMAExpr(
            src0=VariableExpr("v0"),
            src1=VariableExpr("v1"),
            src2=VariableExpr("v2"),
        )
        res = fma_expr.eval(env)
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, f16_to_bits(2.5))
        self.assertEqual(hi, f16_to_bits(4.5))

        # Test constant expr
        const_expr = ConstantExpr(pack_pk_f16(POS_ONE, POS_ONE))
        self.assertEqual(const_expr.eval(env), pack_pk_f16(POS_ONE, POS_ONE))


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single wavefront lane."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index] & MASK32

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_pk_fma_f16."""

    def resolve_operand(self, op: VOP3POperand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg32):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.bits
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VPkFmaF16Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0 = self.resolve_operand(instr.src0, state)
        s1 = self.resolve_operand(instr.src1, state)
        s2 = self.resolve_operand(instr.src2, state)

        # Computation phase
        res = oracle_v_pk_fma_f16(
            s0, s1, s2,
            op_sel=instr.op_sel,
            op_sel_hi=instr.op_sel_hi,
            neg_lo=instr.neg_lo,
            neg_hi=instr.neg_hi,
            clamp=instr.clamp,
        )

        # Write-back phase
        state.set_vgpr(instr.dst.index, res)
        return res


class TestPkFmaF16ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_scalar_vgpr_sources(self) -> None:
        # v1: lo=1.0, hi=2.0
        # v2: lo=3.0, hi=4.0
        # v3: lo=0.5, hi=1.0
        self.state.set_vgpr(1, pack_pk_f16(POS_ONE, POS_TWO))
        self.state.set_vgpr(2, pack_pk_f16(f16_to_bits(3.0), f16_to_bits(4.0)))
        self.state.set_vgpr(3, pack_pk_f16(POS_HALF, POS_ONE))

        instr = VPkFmaF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(1, RegType.VGPR)),
            src1=VOP3POperand(Reg32(2, RegType.VGPR)),
            src2=VOP3POperand(Reg32(3, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        # lo: 1*3 + 0.5 = 3.5
        # hi: 2*4 + 1.0 = 9.0
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, f16_to_bits(3.5))
        self.assertEqual(hi, f16_to_bits(9.0))
        self.assertEqual(self.state.get_vgpr(0), res)

    def test_scalar_mixed_sources(self) -> None:
        # src0 from VGPR, src1 from SGPR, src2 from Literal
        self.state.set_vgpr(5, pack_pk_f16(POS_TWO, POS_TWO))
        self.state.set_sgpr(10, pack_pk_f16(f16_to_bits(3.0), f16_to_bits(5.0)))
        lit = Literal32.from_floats(1.0, -2.0)

        instr = VPkFmaF16Instruction(
            dst=Reg32(4, RegType.VGPR),
            src0=VOP3POperand(Reg32(5, RegType.VGPR)),
            src1=VOP3POperand(Reg32(10, RegType.SGPR)),
            src2=VOP3POperand(lit),
        )
        res = self.interp.step(instr, self.state)
        # lo: 2*3 + 1.0 = 7.0
        # hi: 2*5 + (-2.0) = 8.0
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, f16_to_bits(7.0))
        self.assertEqual(hi, f16_to_bits(8.0))

    def test_scalar_half_selection_and_negation(self) -> None:
        self.state.set_vgpr(1, pack_pk_f16(POS_ONE, POS_TWO))
        self.state.set_vgpr(2, pack_pk_f16(f16_to_bits(3.0), f16_to_bits(4.0)))
        self.state.set_vgpr(3, pack_pk_f16(f16_to_bits(5.0), f16_to_bits(6.0)))

        # lo uses op_sel (1, 0, 0) -> (s0.hi * s1.lo + s2.lo) = 2*3 + 5 = 11
        # neg_lo (1, 0, 0) -> -(2)*3 + 5 = -1
        # hi uses op_sel_hi (0, 1, 1) -> (s0.lo * s1.hi + s2.hi) = 1*4 + 6 = 10
        # neg_hi (0, 1, 0) -> 1*(-4) + 6 = 2
        instr = VPkFmaF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(1, RegType.VGPR)),
            src1=VOP3POperand(Reg32(2, RegType.VGPR)),
            src2=VOP3POperand(Reg32(3, RegType.VGPR)),
            op_sel=(1, 0, 0),
            op_sel_hi=(0, 1, 1),
            neg_lo=(1, 0, 0),
            neg_hi=(0, 1, 0),
        )
        res = self.interp.step(instr, self.state)
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, f16_to_bits(-1.0))
        self.assertEqual(hi, f16_to_bits(2.0))

    def test_scalar_clamp_execution(self) -> None:
        # lo: 1.0 * 2.0 + 0.0 = 2.0 -> clamp -> 1.0
        # hi: -1.0 * 2.0 + 0.0 = -2.0 -> clamp -> 0.0
        self.state.set_vgpr(1, pack_pk_f16(POS_ONE, NEG_ONE))
        self.state.set_vgpr(2, pack_pk_f16(POS_TWO, POS_TWO))
        self.state.set_vgpr(3, pack_pk_f16(POS_ZERO, POS_ZERO))

        instr = VPkFmaF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(1, RegType.VGPR)),
            src1=VOP3POperand(Reg32(2, RegType.VGPR)),
            src2=VOP3POperand(Reg32(3, RegType.VGPR)),
            clamp=True,
        )
        res = self.interp.step(instr, self.state)
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, POS_ONE)
        self.assertEqual(hi, POS_ZERO)

    def test_scalar_determinism(self) -> None:
        runs = []
        for _ in range(100):
            st = ScalarMachineState()
            st.set_vgpr(1, 0x3C004000)
            st.set_vgpr(2, 0x38003C00)
            st.set_vgpr(3, 0x00003800)
            instr = VPkFmaF16Instruction(
                dst=Reg32(0, RegType.VGPR),
                src0=VOP3POperand(Reg32(1, RegType.VGPR)),
                src1=VOP3POperand(Reg32(2, RegType.VGPR)),
                src2=VOP3POperand(Reg32(3, RegType.VGPR)),
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
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {self.num_lanes}")
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        if self.exec_mask is None:
            self.exec_mask = (1 << self.num_lanes) - 1

    def get_vgpr(self, lane: int, index: int) -> int:
        return self.vgprs[lane][index] & MASK32

    def set_vgpr(self, lane: int, index: int, val: int) -> None:
        self.vgprs[lane][index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VOP3POperand, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, Reg32):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(lane, op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.bits
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VPkFmaF16Instruction, state: SIMDState) -> List[Optional[int]]:
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
        s2_vals = [
            self.resolve_lane_operand(instr.src2, state, lane)
            for lane in range(state.num_lanes)
        ]

        # Execution phase
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = oracle_v_pk_fma_f16(
                    s0_vals[lane],
                    s1_vals[lane],
                    s2_vals[lane],
                    op_sel=instr.op_sel,
                    op_sel_hi=instr.op_sel_hi,
                    neg_lo=instr.neg_lo,
                    neg_hi=instr.neg_hi,
                    clamp=instr.clamp,
                )
                state.set_vgpr(lane, instr.dst.index, res)
                results[lane] = res

        return results


class TestPkFmaF16SIMDMultiLane(unittest.TestCase):
    """Test suite for SIMD execution, lane divergence, and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_wave32_full_exec(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            # lo: lane * 2.0 + 1.0, hi: (lane + 1) * 1.0 + 2.0
            state.set_vgpr(lane, 1, pack_pk_f16(f16_to_bits(float(lane)), f16_to_bits(float(lane + 1))))
            state.set_vgpr(lane, 2, pack_pk_f16(POS_TWO, POS_ONE))
            state.set_vgpr(lane, 3, pack_pk_f16(POS_ONE, POS_TWO))

        instr = VPkFmaF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(1, RegType.VGPR)),
            src1=VOP3POperand(Reg32(2, RegType.VGPR)),
            src2=VOP3POperand(Reg32(3, RegType.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            lo, hi = unpack_pk_f16(state.get_vgpr(lane, 0))
            self.assertEqual(lo, f16_to_bits(float(lane) * 2.0 + 1.0))
            self.assertEqual(hi, f16_to_bits(float(lane + 1) * 1.0 + 2.0))

    def test_wave32_divergent_masking(self) -> None:
        sentinel = 0xDEAD_BEEF
        state = SIMDState(num_lanes=32, exec_mask=0x5555_5555)  # even lanes active
        for lane in range(32):
            state.set_vgpr(lane, 1, pack_pk_f16(f16_to_bits(float(lane)), POS_ONE))
            state.set_vgpr(lane, 2, pack_pk_f16(POS_ONE, POS_ONE))
            state.set_vgpr(lane, 3, pack_pk_f16(POS_ZERO, POS_ZERO))
            state.set_vgpr(lane, 4, sentinel)

        instr = VPkFmaF16Instruction(
            dst=Reg32(4, RegType.VGPR),
            src0=VOP3POperand(Reg32(1, RegType.VGPR)),
            src1=VOP3POperand(Reg32(2, RegType.VGPR)),
            src2=VOP3POperand(Reg32(3, RegType.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertIsNotNone(results[lane])
                lo, hi = unpack_pk_f16(state.get_vgpr(lane, 4))
                self.assertEqual(lo, f16_to_bits(float(lane)))
                self.assertEqual(hi, POS_ONE)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 4), sentinel)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr(lane, 1, pack_pk_f16(POS_ONE, POS_TWO))
            state.set_vgpr(lane, 2, pack_pk_f16(f16_to_bits(float(lane)), POS_ONE))
            state.set_vgpr(lane, 3, pack_pk_f16(POS_ZERO, POS_ZERO))

        instr = VPkFmaF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(1, RegType.VGPR)),
            src1=VOP3POperand(Reg32(2, RegType.VGPR)),
            src2=VOP3POperand(Reg32(3, RegType.VGPR)),
        )
        self.interp.step(instr, state)
        for lane in range(64):
            lo, hi = unpack_pk_f16(state.get_vgpr(lane, 0))
            self.assertEqual(lo, f16_to_bits(float(lane)))
            self.assertEqual(hi, POS_TWO)

    def test_wave64_half_disabled(self) -> None:
        sentinel = 0xCAFE_BABE
        state = SIMDState(num_lanes=64, exec_mask=0x0000_0000_FFFF_FFFF)  # lanes 0-31 active
        for lane in range(64):
            state.set_vgpr(lane, 1, pack_pk_f16(POS_TWO, POS_TWO))
            state.set_vgpr(lane, 2, pack_pk_f16(POS_TWO, POS_TWO))
            state.set_vgpr(lane, 3, pack_pk_f16(POS_ONE, POS_ONE))
            state.set_vgpr(lane, 4, sentinel)

        instr = VPkFmaF16Instruction(
            dst=Reg32(4, RegType.VGPR),
            src0=VOP3POperand(Reg32(1, RegType.VGPR)),
            src1=VOP3POperand(Reg32(2, RegType.VGPR)),
            src2=VOP3POperand(Reg32(3, RegType.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(64):
            if lane < 32:
                self.assertIsNotNone(results[lane])
                lo, hi = unpack_pk_f16(state.get_vgpr(lane, 4))
                # 2*2 + 1 = 5
                self.assertEqual(lo, f16_to_bits(5.0))
                self.assertEqual(hi, f16_to_bits(5.0))
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 4), sentinel)

    def test_wave_exec_all_zero(self) -> None:
        sentinel = 0xAA55_AA55
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)
        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)
        instr = VPkFmaF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Literal32.from_floats(1.0, 1.0)),
            src1=VOP3POperand(Literal32.from_floats(2.0, 2.0)),
            src2=VOP3POperand(Literal32.from_floats(3.0, 3.0)),
        )
        results = self.interp.step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_simd_sgpr_uniform_broadcast(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        # Broadcast SGPR s10 = (10.0, 20.0)
        state.set_sgpr(10, pack_pk_f16(f16_to_bits(10.0), f16_to_bits(20.0)))
        for lane in range(32):
            state.set_vgpr(lane, 1, pack_pk_f16(POS_ONE, POS_ONE))
            state.set_vgpr(lane, 2, pack_pk_f16(f16_to_bits(float(lane)), f16_to_bits(float(lane))))

        # v0 = v1 * s10 + v2 -> lo: 1.0 * 10.0 + lane, hi: 1.0 * 20.0 + lane
        instr = VPkFmaF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(1, RegType.VGPR)),
            src1=VOP3POperand(Reg32(10, RegType.SGPR)),
            src2=VOP3POperand(Reg32(2, RegType.VGPR)),
        )
        self.interp.step(instr, state)
        for lane in range(32):
            lo, hi = unpack_pk_f16(state.get_vgpr(lane, 0))
            self.assertEqual(lo, f16_to_bits(10.0 + float(lane)))
            self.assertEqual(hi, f16_to_bits(20.0 + float(lane)))


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestPkFmaF16RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src0(self) -> None:
        state = ScalarMachineState()
        # v0 = (2.0, 3.0), v1 = (4.0, 5.0), v2 = (1.0, 2.0)
        state.set_vgpr(0, pack_pk_f16(POS_TWO, f16_to_bits(3.0)))
        state.set_vgpr(1, pack_pk_f16(f16_to_bits(4.0), f16_to_bits(5.0)))
        state.set_vgpr(2, pack_pk_f16(POS_ONE, POS_TWO))

        # v0 = v0 * v1 + v2 -> lo: 2*4 + 1 = 9.0, hi: 3*5 + 2 = 17.0
        instr = VPkFmaF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(1, RegType.VGPR)),
            src2=VOP3POperand(Reg32(2, RegType.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, f16_to_bits(9.0))
        self.assertEqual(hi, f16_to_bits(17.0))
        self.assertEqual(state.get_vgpr(0), res)
        # Verify src1 and src2 were not mutated
        self.assertEqual(state.get_vgpr(1), pack_pk_f16(f16_to_bits(4.0), f16_to_bits(5.0)))
        self.assertEqual(state.get_vgpr(2), pack_pk_f16(POS_ONE, POS_TWO))

    def test_dst_alias_src1(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, pack_pk_f16(POS_TWO, POS_TWO))
        state.set_vgpr(1, pack_pk_f16(f16_to_bits(3.0), f16_to_bits(4.0)))
        state.set_vgpr(2, pack_pk_f16(POS_HALF, POS_HALF))

        # v1 = v0 * v1 + v2 -> lo: 2*3 + 0.5 = 6.5, hi: 2*4 + 0.5 = 8.5
        instr = VPkFmaF16Instruction(
            dst=Reg32(1, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(1, RegType.VGPR)),
            src2=VOP3POperand(Reg32(2, RegType.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, f16_to_bits(6.5))
        self.assertEqual(hi, f16_to_bits(8.5))
        self.assertEqual(state.get_vgpr(1), res)
        self.assertEqual(state.get_vgpr(0), pack_pk_f16(POS_TWO, POS_TWO))

    def test_dst_alias_src2(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, pack_pk_f16(POS_TWO, POS_TWO))
        state.set_vgpr(1, pack_pk_f16(POS_TWO, POS_TWO))
        state.set_vgpr(2, pack_pk_f16(POS_ONE, POS_ONE))

        # v2 = v0 * v1 + v2 -> lo: 2*2 + 1 = 5.0, hi: 2*2 + 1 = 5.0
        instr = VPkFmaF16Instruction(
            dst=Reg32(2, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(1, RegType.VGPR)),
            src2=VOP3POperand(Reg32(2, RegType.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, f16_to_bits(5.0))
        self.assertEqual(hi, f16_to_bits(5.0))
        self.assertEqual(state.get_vgpr(2), res)

    def test_dst_alias_all_three_operands(self) -> None:
        state = ScalarMachineState()
        # v0 = (2.0, 3.0)
        # v0 = v0 * v0 + v0 -> lo: 2*2 + 2 = 6.0, hi: 3*3 + 3 = 12.0
        state.set_vgpr(0, pack_pk_f16(POS_TWO, f16_to_bits(3.0)))
        instr = VPkFmaF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(0, RegType.VGPR)),
            src2=VOP3POperand(Reg32(0, RegType.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, f16_to_bits(6.0))
        self.assertEqual(hi, f16_to_bits(12.0))
        self.assertEqual(state.get_vgpr(0), res)

    def test_inplace_iterative_mac_accumulation(self) -> None:
        state = ScalarMachineState()
        # v0 accumulator starts at (0.0, 0.0)
        state.set_vgpr(0, pack_pk_f16(POS_ZERO, POS_ZERO))
        state.set_vgpr(1, pack_pk_f16(POS_ONE, POS_ONE))
        state.set_vgpr(2, pack_pk_f16(POS_HALF, POS_ONE))

        # 10 iterations: v0 = v1 * v2 + v0
        # each iteration adds 0.5 to lo and 1.0 to hi
        instr = VPkFmaF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(1, RegType.VGPR)),
            src1=VOP3POperand(Reg32(2, RegType.VGPR)),
            src2=VOP3POperand(Reg32(0, RegType.VGPR)),
        )
        for _ in range(10):
            self.scalar_interp.step(instr, state)

        lo, hi = unpack_pk_f16(state.get_vgpr(0))
        self.assertEqual(lo, f16_to_bits(5.0))
        self.assertEqual(hi, f16_to_bits(10.0))

    def test_simd_multi_lane_aliasing(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr(lane, 0, pack_pk_f16(f16_to_bits(float(lane)), f16_to_bits(float(lane))))
            state.set_vgpr(lane, 1, pack_pk_f16(POS_TWO, POS_TWO))
            state.set_vgpr(lane, 2, pack_pk_f16(POS_ONE, POS_ONE))

        # Each lane computes in place: v0 = v0 * v1 + v2 -> lane * 2.0 + 1.0
        instr = VPkFmaF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(1, RegType.VGPR)),
            src2=VOP3POperand(Reg32(2, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            lo, hi = unpack_pk_f16(state.get_vgpr(lane, 0))
            expected = f16_to_bits(float(lane) * 2.0 + 1.0)
            self.assertEqual(lo, expected)
            self.assertEqual(hi, expected)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class FPStatusFlags:
    """Architectural floating-point condition codes and IEEE 754 status flags for binary16."""
    invalid: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False
    zero: bool = False
    negative: bool = False
    denormal: bool = False
    infinite: bool = False
    nan: bool = False


@dataclass(frozen=True)
class PackedFPStatusFlags:
    """Status flags for dual-half packed execution (lo and hi lanes)."""
    lo: FPStatusFlags
    hi: FPStatusFlags


def evaluate_fma_f16_flags(
    s0_bits: int,
    s1_bits: int,
    s2_bits: int,
    res_bits: int,
    clamp: bool = False,
) -> FPStatusFlags:
    """Evaluate IEEE 754 architectural flags and condition codes for a single f16 lane."""
    s0_nan = is_nan16(s0_bits)
    s1_nan = is_nan16(s1_bits)
    s2_nan = is_nan16(s2_bits)
    s0_snan = is_snan16(s0_bits)
    s1_snan = is_snan16(s1_bits)
    s2_snan = is_snan16(s2_bits)

    s0_inf = is_inf16(s0_bits)
    s1_inf = is_inf16(s1_bits)
    s2_inf = is_inf16(s2_bits)

    s0_zero = is_zero16(s0_bits)
    s1_zero = is_zero16(s1_bits)

    invalid = False
    if s0_snan or s1_snan or s2_snan:
        invalid = True
    elif (s0_inf and s1_zero) or (s0_zero and s1_inf):
        invalid = True
    elif (s0_inf or s1_inf) and s2_inf:
        prod_sign = ((s0_bits ^ s1_bits) & SIGN_MASK16)
        s2_sign = (s2_bits & SIGN_MASK16)
        if prod_sign != s2_sign:
            invalid = True

    nan_flag = is_nan16(res_bits)
    inf_flag = is_inf16(res_bits)
    zero_flag = is_zero16(res_bits)
    neg_flag = bool(res_bits & SIGN_MASK16)
    denorm_flag = is_subnormal16(res_bits)

    overflow = False
    if not (s0_nan or s1_nan or s2_nan or s0_inf or s1_inf or s2_inf) and inf_flag:
        overflow = True

    underflow = False
    if denorm_flag:
        underflow = True

    inexact = False
    if overflow:
        inexact = True
    elif not (s0_nan or s1_nan or s2_nan or s0_inf or s1_inf or s2_inf or nan_flag or inf_flag):
        f0 = bits_to_f16(s0_bits)
        f1 = bits_to_f16(s1_bits)
        f2 = bits_to_f16(s2_bits)
        res_f = bits_to_f16(res_bits)
        exact_frac = Fraction(f0) * Fraction(f1) + Fraction(f2)
        if not clamp:
            if Fraction(res_f) != exact_frac:
                inexact = True
        else:
            clamped_math = min(1.0, max(0.0, float(exact_frac)))
            if Fraction(res_f) != Fraction(clamped_math):
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


def evaluate_pk_fma_f16_flags(
    src0_bits: int,
    src1_bits: int,
    src2_bits: int,
    res_bits: int,
    op_sel: Tuple[int, int, int] = (0, 0, 0),
    op_sel_hi: Tuple[int, int, int] = (1, 1, 1),
    neg_lo: Tuple[int, int, int] = (0, 0, 0),
    neg_hi: Tuple[int, int, int] = (0, 0, 0),
    clamp: bool = False,
) -> PackedFPStatusFlags:
    """Evaluate architectural flags for both halves of a packed v_pk_fma_f16 operation."""
    s0_lo, s0_hi = unpack_pk_f16(src0_bits)
    s1_lo, s1_hi = unpack_pk_f16(src1_bits)
    s2_lo, s2_hi = unpack_pk_f16(src2_bits)
    res_lo, res_hi = unpack_pk_f16(res_bits)

    in0_lo = apply_f16_modifiers(s0_hi if op_sel[0] else s0_lo, neg_mod=bool(neg_lo[0]))
    in1_lo = apply_f16_modifiers(s1_hi if op_sel[1] else s1_lo, neg_mod=bool(neg_lo[1]))
    in2_lo = apply_f16_modifiers(s2_hi if op_sel[2] else s2_lo, neg_mod=bool(neg_lo[2]))

    in0_hi = apply_f16_modifiers(s0_hi if op_sel_hi[0] else s0_lo, neg_mod=bool(neg_hi[0]))
    in1_hi = apply_f16_modifiers(s1_hi if op_sel_hi[1] else s1_lo, neg_mod=bool(neg_hi[1]))
    in2_hi = apply_f16_modifiers(s2_hi if op_sel_hi[2] else s2_lo, neg_mod=bool(neg_hi[2]))

    flags_lo = evaluate_fma_f16_flags(in0_lo, in1_lo, in2_lo, res_lo, clamp=clamp)
    flags_hi = evaluate_fma_f16_flags(in0_hi, in1_hi, in2_hi, res_hi, clamp=clamp)

    return PackedFPStatusFlags(lo=flags_lo, hi=flags_hi)


def compute_f16_fma_mantissa_product_carry(
    s0_bits: int,
    s1_bits: int,
    s2_bits: int,
) -> Tuple[int, int]:
    """Demonstrate IEEE 754 binary16 significand product and carry/overflow generation."""
    _, exp0, frac0 = split_f16_bits(s0_bits)
    _, exp1, frac1 = split_f16_bits(s1_bits)
    m0 = (1 << 10) | frac0 if exp0 != 0 else frac0
    m1 = (1 << 10) | frac1 if exp1 != 0 else frac1
    product = m0 * m1
    carry_out = 1 if product >= (1 << 21) else 0
    return product, carry_out


class TestPkFmaF16ConditionCodesAndStatusFlags(unittest.TestCase):
    """Test suite for status flags, condition codes, and carry generation."""

    def test_exact_finite_flags(self) -> None:
        v0 = pack_pk_f16(POS_ONE, POS_TWO)
        v1 = pack_pk_f16(POS_TWO, POS_TWO)
        v2 = pack_pk_f16(POS_HALF, POS_ONE)
        res = oracle_v_pk_fma_f16(v0, v1, v2)
        flags = evaluate_pk_fma_f16_flags(v0, v1, v2, res)

        for lane_flags in (flags.lo, flags.hi):
            self.assertFalse(lane_flags.invalid)
            self.assertFalse(lane_flags.overflow)
            self.assertFalse(lane_flags.underflow)
            self.assertFalse(lane_flags.inexact)
            self.assertFalse(lane_flags.zero)
            self.assertFalse(lane_flags.negative)
            self.assertFalse(lane_flags.denormal)
            self.assertFalse(lane_flags.infinite)
            self.assertFalse(lane_flags.nan)

    def test_inexact_flag_generation(self) -> None:
        # 1.0 * 1.0 + 2^-24: exact sum is 1 + 2^-24, but rounded binary16 result is 1.0 -> inexact!
        v0 = pack_pk_f16(POS_ONE, POS_ONE)
        v1 = pack_pk_f16(POS_ONE, POS_ONE)
        v2 = pack_pk_f16(POS_MIN_SUBNORM, POS_MIN_SUBNORM)
        res = oracle_v_pk_fma_f16(v0, v1, v2)
        flags = evaluate_pk_fma_f16_flags(v0, v1, v2, res)
        self.assertTrue(flags.lo.inexact)
        self.assertFalse(flags.lo.overflow)

    def test_overflow_flag_generation(self) -> None:
        # Max normal 65504.0 * 2.0 + 0.0 -> overflow to +inf
        v0 = pack_pk_f16(POS_MAX_NORMAL, POS_MAX_NORMAL)
        v1 = pack_pk_f16(POS_TWO, POS_TWO)
        v2 = pack_pk_f16(POS_ZERO, POS_ZERO)
        res = oracle_v_pk_fma_f16(v0, v1, v2)
        flags = evaluate_pk_fma_f16_flags(v0, v1, v2, res)
        self.assertTrue(flags.lo.overflow)
        self.assertTrue(flags.lo.infinite)
        self.assertTrue(flags.lo.inexact)
        self.assertFalse(flags.lo.invalid)

    def test_underflow_and_denormal_flags(self) -> None:
        # Subnormal arithmetic produces denormal flag and underflow flag
        sub = POS_MIN_SUBNORM
        v0 = pack_pk_f16(sub, sub)
        v1 = pack_pk_f16(POS_ONE, POS_ONE)
        v2 = pack_pk_f16(POS_ZERO, POS_ZERO)
        res = oracle_v_pk_fma_f16(v0, v1, v2)
        flags = evaluate_pk_fma_f16_flags(v0, v1, v2, res)
        self.assertTrue(flags.lo.denormal)
        self.assertTrue(flags.lo.underflow)
        self.assertFalse(flags.lo.overflow)

    def test_invalid_operation_flag_generation(self) -> None:
        # +inf * 0.0 is invalid
        v0 = pack_pk_f16(POS_INF, POS_ONE)
        v1 = pack_pk_f16(POS_ZERO, POS_ONE)
        v2 = pack_pk_f16(POS_ZERO, POS_ZERO)
        res = oracle_v_pk_fma_f16(v0, v1, v2)
        flags = evaluate_pk_fma_f16_flags(v0, v1, v2, res)
        self.assertTrue(flags.lo.invalid)
        self.assertTrue(flags.lo.nan)
        self.assertFalse(flags.hi.invalid)

        # Signaling NaN in input produces invalid flag
        v_snan = pack_pk_f16(SNAN_DEFAULT, POS_ONE)
        res_snan = oracle_v_pk_fma_f16(v_snan, v1, v2)
        flags_snan = evaluate_pk_fma_f16_flags(v_snan, v1, v2, res_snan)
        self.assertTrue(flags_snan.lo.invalid)

    def test_zero_and_negative_flags(self) -> None:
        # -2.0 * 2.0 + 0.0 = -4.0 (negative)
        # +0.0 * 1.0 + 0.0 = +0.0 (zero)
        v0 = pack_pk_f16(NEG_TWO, POS_ZERO)
        v1 = pack_pk_f16(POS_TWO, POS_ONE)
        v2 = pack_pk_f16(POS_ZERO, POS_ZERO)
        res = oracle_v_pk_fma_f16(v0, v1, v2)
        flags = evaluate_pk_fma_f16_flags(v0, v1, v2, res)

        self.assertTrue(flags.lo.negative)
        self.assertFalse(flags.lo.zero)

        self.assertTrue(flags.hi.zero)
        self.assertFalse(flags.hi.negative)

    def test_packed_status_flags_divergence(self) -> None:
        # lo overflows to +inf, hi underflows to subnormal
        v0 = pack_pk_f16(POS_MAX_NORMAL, POS_MIN_SUBNORM)
        v1 = pack_pk_f16(POS_TWO, POS_ONE)
        v2 = pack_pk_f16(POS_ZERO, POS_ZERO)
        res = oracle_v_pk_fma_f16(v0, v1, v2)
        flags = evaluate_pk_fma_f16_flags(v0, v1, v2, res)

        self.assertTrue(flags.lo.overflow)
        self.assertFalse(flags.lo.underflow)

        self.assertTrue(flags.hi.underflow)
        self.assertFalse(flags.hi.overflow)

    def test_mantissa_product_carry_generation(self) -> None:
        # 1.0 has significand (1 << 10) = 1024
        # 1.0 * 1.0: product = 1024 * 1024 = 1048576 = (1 << 20) -> carry = 0
        prod0, carry0 = compute_f16_fma_mantissa_product_carry(POS_ONE, POS_ONE, POS_ZERO)
        self.assertEqual(prod0, 1 << 20)
        self.assertEqual(carry0, 0)

        # Significand with leading 1.111...: ~2047 * 2047 ~= 4190209 >= (1 << 21) -> carry = 1
        val_near_two = f16_to_bits(1.999)
        prod1, carry1 = compute_f16_fma_mantissa_product_carry(val_near_two, val_near_two, POS_ZERO)
        self.assertEqual(carry1, 1)


# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestPkFmaF16IEEE754Boundaries(unittest.TestCase):
    """Test suite for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_addition_without_underflow(self) -> None:
        # 0x0001 (2^-24) * 1.0 + 0x0001 = 0x0002
        v0 = pack_pk_f16(POS_MIN_SUBNORM, POS_MIN_SUBNORM)
        v1 = pack_pk_f16(POS_ONE, POS_ONE)
        v2 = pack_pk_f16(POS_MIN_SUBNORM, POS_MIN_SUBNORM)
        res = oracle_v_pk_fma_f16(v0, v1, v2)
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, 0x0002)
        self.assertEqual(hi, 0x0002)
        self.assertTrue(is_subnormal16(lo))

    def test_subnormal_to_normal_transition(self) -> None:
        # Largest subnormal: 0x03FF ((1023/1024)*2^-14)
        # 0x03FF * 1.0 + 0x0001 = 0x0400 (smallest normal 2^-14)
        v0 = pack_pk_f16(POS_MAX_SUBNORM, POS_MAX_SUBNORM)
        v1 = pack_pk_f16(POS_ONE, POS_ONE)
        v2 = pack_pk_f16(POS_MIN_SUBNORM, POS_MIN_SUBNORM)
        res = oracle_v_pk_fma_f16(v0, v1, v2)
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, POS_MIN_NORMAL)
        self.assertEqual(hi, POS_MIN_NORMAL)
        self.assertFalse(is_subnormal16(lo))
        _, exp, frac = split_f16_bits(lo)
        self.assertEqual(exp, 1)
        self.assertEqual(frac, 0)

    def test_subnormal_cancellation(self) -> None:
        sub = 0x0123
        v0 = pack_pk_f16(sub, sub)
        v1 = pack_pk_f16(POS_ONE, POS_ONE)
        v2 = pack_pk_f16(sub, sub)
        # v0 * 1.0 + (-v2) = +0.0
        res = oracle_v_pk_fma_f16(v0, v1, v2, neg_lo=(0, 0, 1), neg_hi=(0, 0, 1))
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, POS_ZERO)
        self.assertEqual(hi, POS_ZERO)

    def test_gradual_underflow(self) -> None:
        # 0.5 * 2^-14 (POS_MIN_NORMAL) + 0.0 = 2^-15 (subnormal 0x0200)
        v0 = pack_pk_f16(POS_HALF, POS_HALF)
        v1 = pack_pk_f16(POS_MIN_NORMAL, POS_MIN_NORMAL)
        v2 = pack_pk_f16(POS_ZERO, POS_ZERO)
        res = oracle_v_pk_fma_f16(v0, v1, v2)
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, 0x0200)
        self.assertEqual(hi, 0x0200)
        self.assertTrue(is_subnormal16(lo))

    def test_nan_signaling_to_quiet_conversion(self) -> None:
        snan = 0x7D05
        self.assertTrue(is_snan16(snan))
        v0 = pack_pk_f16(snan, POS_ONE)
        v1 = pack_pk_f16(POS_ONE, POS_ONE)
        v2 = pack_pk_f16(POS_ZERO, POS_ZERO)
        res = oracle_v_pk_fma_f16(v0, v1, v2)
        lo, hi = unpack_pk_f16(res)
        self.assertTrue(is_qnan16(lo))
        self.assertEqual(lo, snan | QNAN_BIT16)
        self.assertEqual(hi, POS_ONE)

    def test_nan_payload_preservation(self) -> None:
        payload_qnan = 0x7E12
        self.assertTrue(is_qnan16(payload_qnan))
        v0 = pack_pk_f16(payload_qnan, POS_ONE)
        v1 = pack_pk_f16(POS_ONE, POS_ONE)
        v2 = pack_pk_f16(POS_ZERO, POS_ZERO)
        res = oracle_v_pk_fma_f16(v0, v1, v2)
        lo, _ = unpack_pk_f16(res)
        self.assertEqual(lo, payload_qnan)

    def test_nan_with_modifiers(self) -> None:
        qnan = 0x7E01
        v0 = pack_pk_f16(qnan, qnan)
        v1 = pack_pk_f16(POS_ONE, POS_ONE)
        v2 = pack_pk_f16(POS_ZERO, POS_ZERO)

        # neg_lo toggles sign bit on NaN
        res = oracle_v_pk_fma_f16(v0, v1, v2, neg_lo=(1, 0, 0))
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo & SIGN_MASK16, SIGN_MASK16)
        self.assertEqual(hi & SIGN_MASK16, 0)

    def test_nan_with_clamp(self) -> None:
        # NaNs clamped with clamp=True map to +0.0 (0x0000)
        v0 = pack_pk_f16(QNAN_CANONICAL, SNAN_DEFAULT)
        v1 = pack_pk_f16(POS_ONE, POS_ONE)
        v2 = pack_pk_f16(POS_ZERO, POS_ZERO)
        res = oracle_v_pk_fma_f16(v0, v1, v2, clamp=True)
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, POS_ZERO)
        self.assertEqual(hi, POS_ZERO)

    def test_infinity_arithmetic_boundaries(self) -> None:
        # (+inf) * 2.0 + 1.0 = +inf
        # (-inf) * 2.0 + 1.0 = -inf
        v0 = pack_pk_f16(POS_INF, NEG_INF)
        v1 = pack_pk_f16(POS_TWO, POS_TWO)
        v2 = pack_pk_f16(POS_ONE, POS_ONE)
        res = oracle_v_pk_fma_f16(v0, v1, v2)
        lo, hi = unpack_pk_f16(res)
        self.assertEqual(lo, POS_INF)
        self.assertEqual(hi, NEG_INF)

        # (+inf) * 0.0 + 1.0 = QNaN
        v_zero = pack_pk_f16(POS_ZERO, POS_ZERO)
        res_inv = oracle_v_pk_fma_f16(v0, v_zero, v2)
        lo_inv, _ = unpack_pk_f16(res_inv)
        self.assertTrue(is_nan16(lo_inv))

        # Infinity clamped: +inf -> 1.0, -inf -> +0.0
        res_clamp = oracle_v_pk_fma_f16(v0, v1, v2, clamp=True)
        lo_c, hi_c = unpack_pk_f16(res_clamp)
        self.assertEqual(lo_c, POS_ONE)
        self.assertEqual(hi_c, POS_ZERO)

    def test_round_to_nearest_even_ties(self) -> None:
        # In IEEE 754 binary16 (11 bits significand):
        # 1.0 is 0x3C00 (even LSB = 0)
        # 1.0 + 2^-10 is 0x3C01 (odd LSB = 1)
        # 2^-11 is exactly halfway between representable binary16 numbers.
        half_ulp = 2.0 ** -11
        val_even = 1.0
        val_odd = bits_to_f16(0x3C01)

        # Case 1: 1.0 (even LSB) * 1.0 + 2^-11: tie rounds to even (1.0)
        res1 = compute_scalar_fma_f16_bits(POS_ONE, POS_ONE, f16_to_bits(half_ulp))
        self.assertEqual(res1, POS_ONE)

        # Case 2: (1.0 + 2^-10) (odd LSB) * 1.0 + 2^-11: tie rounds to even (1.0 + 2 * 2^-10 = 0x3C02)
        res2 = compute_scalar_fma_f16_bits(0x3C01, POS_ONE, f16_to_bits(half_ulp))
        self.assertEqual(res2, 0x3C02)


# ==============================================================================
# Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestPkFmaF16NegativeControlsAndMutations(unittest.TestCase):
    """Test suite for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Reg32(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(300, RegType.SGPR)

    def test_syntax_rejection_valu_destination_must_be_vgpr(self) -> None:
        v = VOP3POperand(Reg32(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VPkFmaF16Instruction(dst=Reg32(0, RegType.SGPR), src0=v, src1=v, src2=v)

    def test_syntax_rejection_invalid_simd_lane_count(self) -> None:
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_syntax_rejection_invalid_modifier_tuples(self) -> None:
        v = VOP3POperand(Reg32(1, RegType.VGPR))
        with self.assertRaises(ValueError):
            VPkFmaF16Instruction(dst=Reg32(0, RegType.VGPR), src0=v, src1=v, src2=v, op_sel=(0, 0))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            VPkFmaF16Instruction(dst=Reg32(0, RegType.VGPR), src0=v, src1=v, src2=v, neg_lo=(0, 2, 0))

    def test_negative_control_bit_flip_detection(self) -> None:
        v0 = pack_pk_f16(f16_to_bits(2.0), f16_to_bits(3.0))
        v1 = pack_pk_f16(f16_to_bits(4.0), f16_to_bits(5.0))
        v2 = pack_pk_f16(f16_to_bits(1.0), f16_to_bits(2.0))
        correct = oracle_v_pk_fma_f16(v0, v1, v2)

        # Perturb single bit in low half
        corrupted_lo = correct ^ 0x0001
        self.assertNotEqual(correct, corrupted_lo)

        # Perturb single bit in high half
        corrupted_hi = correct ^ 0x0001_0000
        self.assertNotEqual(correct, corrupted_hi)

    def test_mutation_non_fused_intermediate_rounding(self) -> None:
        """Mutant: Replace fused single rounding with unfused separate mul and add."""
        def mutant_fma(s0: int, s1: int, s2: int) -> int:
            return compute_scalar_fma_f16_bits(s0, s1, s2 ^ SIGN_MASK16)

        v0 = 0x3C01
        v1 = 0x3C01
        v2 = 0x1400
        expected = compute_scalar_fma_f16_bits(v0, v1, v2)
        mutant_res = mutant_fma(v0, v1, v2)
        self.assertNotEqual(expected, mutant_res)

    def test_mutation_invert_half_selection(self) -> None:
        """Mutant: Inversion of op_sel or op_sel_hi half routing."""
        v0 = pack_pk_f16(f16_to_bits(1.0), f16_to_bits(10.0))
        v1 = pack_pk_f16(f16_to_bits(2.0), f16_to_bits(20.0))
        v2 = pack_pk_f16(POS_ZERO, POS_ZERO)

        # Expected: op_sel=(0, 0, 0) selects lo halves -> 1.0 * 2.0 = 2.0
        good = oracle_v_pk_fma_f16(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 1, 1))
        # Mutant swaps lo half selection to hi -> 10.0 * 20.0 = 200.0
        bad = oracle_v_pk_fma_f16(v0, v1, v2, op_sel=(1, 1, 0), op_sel_hi=(1, 1, 1))
        self.assertNotEqual(good, bad)

    def test_mutation_ignore_neg_modifiers(self) -> None:
        """Mutant: Ignore neg_lo modifier."""
        v0 = pack_pk_f16(POS_TWO, POS_TWO)
        v1 = pack_pk_f16(POS_TWO, POS_TWO)
        v2 = pack_pk_f16(POS_ZERO, POS_ZERO)

        good = oracle_v_pk_fma_f16(v0, v1, v2, neg_lo=(1, 0, 0))
        bad = oracle_v_pk_fma_f16(v0, v1, v2, neg_lo=(0, 0, 0))
        self.assertNotEqual(good, bad)
        lo_good, _ = unpack_pk_f16(good)
        lo_bad, _ = unpack_pk_f16(bad)
        self.assertEqual(lo_good, f16_to_bits(-4.0))
        self.assertEqual(lo_bad, f16_to_bits(4.0))

    def test_mutation_ignore_clamp_bounds(self) -> None:
        """Mutant: Clamp modifier fails to saturate values > 1.0."""
        v0 = pack_pk_f16(POS_TWO, POS_TWO)
        v1 = pack_pk_f16(POS_TWO, POS_TWO)
        v2 = pack_pk_f16(POS_ZERO, POS_ZERO)

        good = oracle_v_pk_fma_f16(v0, v1, v2, clamp=True)
        bad = oracle_v_pk_fma_f16(v0, v1, v2, clamp=False)
        self.assertNotEqual(good, bad)
        lo_good, _ = unpack_pk_f16(good)
        self.assertEqual(lo_good, POS_ONE)

    def test_mutation_simd_inactive_lane_leakage(self) -> None:
        """Mutant: Inactive SIMD lanes are corrupted during masked execution."""
        sentinel = 0xCAFE_BABE
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)  # all inactive
        for lane in range(32):
            state.set_vgpr(lane, 4, sentinel)

        class LeakySIMDInterpreter(SIMDInterpreter):
            def step(self, instr: VPkFmaF16Instruction, state: SIMDState) -> List[Optional[int]]:
                results = []
                for lane in range(state.num_lanes):
                    # Bug: ignores exec_mask
                    res = oracle_v_pk_fma_f16(pack_pk_f16(POS_ONE, POS_ONE), pack_pk_f16(POS_ONE, POS_ONE), 0)
                    state.set_vgpr(lane, instr.dst.index, res)
                    results.append(res)
                return results

        leaky = LeakySIMDInterpreter()
        instr = VPkFmaF16Instruction(
            dst=Reg32(4, RegType.VGPR),
            src0=VOP3POperand(Literal32(pack_pk_f16(POS_ONE, POS_ONE))),
            src1=VOP3POperand(Literal32(pack_pk_f16(POS_ONE, POS_ONE))),
            src2=VOP3POperand(Literal32(0)),
        )
        leaky.step(instr, state)
        corrupted = [lane for lane in range(32) if state.get_vgpr(lane, 4) != sentinel]
        self.assertEqual(len(corrupted), 32)

    def test_mutation_subnormal_flush_to_zero(self) -> None:
        """Mutant: Subnormals flushed to zero (FTZ mode)."""
        sub = POS_MIN_SUBNORM
        correct = compute_scalar_fma_f16_bits(sub, POS_ONE, sub)
        self.assertEqual(correct, 0x0002)

        # FTZ mutant produces 0.0
        ftz_mutant = POS_ZERO
        self.assertNotEqual(correct, ftz_mutant)

    def test_mutation_skip_snan_quietening(self) -> None:
        """Mutant: Signaling NaN is not quietened."""
        snan = SNAN_DEFAULT
        res = compute_scalar_fma_f16_bits(snan, POS_ONE, POS_ZERO)
        self.assertTrue(is_qnan16(res))
        self.assertNotEqual(res, snan)


if __name__ == "__main__":
    unittest.main()
