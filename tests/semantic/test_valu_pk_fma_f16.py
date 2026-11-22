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


if __name__ == "__main__":
    unittest.main()
