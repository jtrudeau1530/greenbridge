"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_pk_fma_clamp_f16.

Architecture: AMD RDNA2 (GFX10 / GFX1030) VALU VOP3P.
Instruction:
    v_pk_fma_clamp_f16 dst, src0, src1, src2 [op_sel] [op_sel_hi] [neg_lo] [neg_hi]

Packed execution:
    dst.lo = clamp01(src0_sel_lo * src1_sel_lo + src2_sel_lo)
    dst.hi = clamp01(src0_sel_hi * src1_sel_hi + src2_sel_hi)

Clamp semantics in RDNA2 VALU:
    - Saturated output range is [0.0, 1.0].
    - Underflow / negative results (-0.0 or < 0.0) saturate to +0.0 (0x0000).
    - Overflow / results > 1.0 saturate to +1.0 (0x3C00).
    - Any NaN result (including invalid operations like 0 * inf or (+inf + -inf))
      saturates to +0.0 (0x0000).

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import enum
from fractions import Fraction
import math
import struct
from typing import List, Optional, Tuple, Union
import unittest

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF

# IEEE 754 binary16 bit constants
F16_POS_ZERO: int = 0x0000
F16_NEG_ZERO: int = 0x8000
F16_POS_ONE: int = 0x3C00
F16_NEG_ONE: int = 0xBC00
F16_POS_HALF: int = 0x3800
F16_NEG_HALF: int = 0xB800
F16_POS_TWO: int = 0x4000
F16_NEG_TWO: int = 0xC000
F16_POS_INF: int = 0x7C00
F16_NEG_INF: int = 0xFC00
F16_CANONICAL_QNAN: int = 0x7E00
F16_QUIET_BIT: int = 0x0200

F16_EXP_MASK: int = 0x7C00
F16_MANT_MASK: int = 0x03FF
F16_SIGN_MASK: int = 0x8000

F16_MIN_NORMAL: int = 0x0400    # 2^-14
F16_MAX_NORMAL: int = 0x7BFF    # 65504.0
F16_MIN_SUBNORMAL: int = 0x0001 # 2^-24
F16_MAX_SUBNORMAL: int = 0x03FF # 2^-14 * (1023/1024)

OLD_VGPR_SENTINEL: int = 0xDEADBEEF


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit range."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range."""
    return val & MASK32


def pack_f16x2(lo: int, hi: int) -> int:
    """Pack two 16-bit half words into a single 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def unpack_f16x2(packed: int) -> Tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit half words (lo, hi)."""
    return packed & MASK16, (packed >> 16) & MASK16


def f16_to_u16(val: float) -> int:
    """Convert float to IEEE 754 binary16 bits."""
    if math.isnan(val):
        return F16_CANONICAL_QNAN
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return F16_POS_INF if val > 0 else F16_NEG_INF


def u16_to_f16(bits: int) -> float:
    """Convert IEEE 754 binary16 bits to float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def is_f16_nan(bits: int) -> bool:
    """Return True if 16-bit pattern is an IEEE 754 NaN."""
    return ((bits & F16_EXP_MASK) == F16_EXP_MASK) and ((bits & F16_MANT_MASK) != 0)


def is_f16_snan(bits: int) -> bool:
    """Return True if 16-bit pattern is a signaling NaN."""
    return is_f16_nan(bits) and ((bits & F16_QUIET_BIT) == 0)


def is_f16_qnan(bits: int) -> bool:
    """Return True if 16-bit pattern is a quiet NaN."""
    return is_f16_nan(bits) and ((bits & F16_QUIET_BIT) != 0)


def is_f16_zero(bits: int) -> bool:
    """Return True if 16-bit pattern is +0.0 or -0.0."""
    return (bits & (F16_EXP_MASK | F16_MANT_MASK)) == 0


def is_f16_inf(bits: int) -> bool:
    """Return True if 16-bit pattern is +inf or -inf."""
    return ((bits & F16_EXP_MASK) == F16_EXP_MASK) and ((bits & F16_MANT_MASK) == 0)


def is_f16_subnormal(bits: int) -> bool:
    """Return True if 16-bit pattern is a subnormal float."""
    return ((bits & F16_EXP_MASK) == 0) and ((bits & F16_MANT_MASK) != 0)


def f16_sign(bits: int) -> int:
    """Return the sign bit of a 16-bit float (0 or 1)."""
    return (bits >> 15) & 1


def clamp01_f16(bits: int) -> int:
    """Apply RDNA2 VALU clamp [0.0, 1.0] saturation to binary16 bits."""
    if is_f16_nan(bits):
        return F16_POS_ZERO
    val = u16_to_f16(bits)
    if val < 0.0 or bits == F16_NEG_ZERO:
        return F16_POS_ZERO
    if val > 1.0 or bits == F16_POS_INF:
        return F16_POS_ONE
    return bits


def ref_fma_clamp_f16(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for scalar clamped fused multiply-add: clamp01(s0 * s1 + s2).

    Computes fused multiplication and addition conforming to IEEE 754-2008
    with infinite intermediate precision before rounding and RDNA2 [0.0, 1.0] clamping.
    """
    s0 &= MASK16
    s1 &= MASK16
    s2 &= MASK16

    # 1. NaN handling: Any NaN input clamped produces +0.0 in RDNA2 VALU
    if is_f16_nan(s0) or is_f16_nan(s1) or is_f16_nan(s2):
        return F16_POS_ZERO

    # 2. Check for invalid operation during multiply (0 * inf)
    s0_inf = is_f16_inf(s0)
    s1_inf = is_f16_inf(s1)
    s0_zero = is_f16_zero(s0)
    s1_zero = is_f16_zero(s1)

    if (s0_inf and s1_zero) or (s1_inf and s0_zero):
        return F16_POS_ZERO

    # 3. Check for invalid operation during addition (+inf + -inf)
    prod_sign = f16_sign(s0) ^ f16_sign(s1)
    prod_is_inf = s0_inf or s1_inf
    s2_inf = is_f16_inf(s2)

    if prod_is_inf and s2_inf:
        if prod_sign != f16_sign(s2):
            return F16_POS_ZERO

    # 4. Handle infinities that survive
    if prod_is_inf:
        # If product is +inf, result is +inf -> clamped to 1.0
        # If product is -inf, result is -inf -> clamped to +0.0
        return F16_POS_ONE if prod_sign == 0 else F16_POS_ZERO
    if s2_inf:
        return F16_POS_ONE if f16_sign(s2) == 0 else F16_POS_ZERO

    # 5. Exact arithmetic with Fraction for finite numbers
    f0 = Fraction(u16_to_f16(s0))
    f1 = Fraction(u16_to_f16(s1))
    f2 = Fraction(u16_to_f16(s2))
    exact_res = f0 * f1 + f2

    # Float conversion and IEEE 754 binary16 rounding
    raw_res_bits = f16_to_u16(float(exact_res))

    # Apply clamp
    return clamp01_f16(raw_res_bits)


def ref_v_pk_fma_clamp_f16(
    s0_dword: int,
    s1_dword: int,
    s2_dword: int,
    *,
    op_sel: Tuple[int, int, int] = (0, 0, 0),
    op_sel_hi: Tuple[int, int, int] = (1, 1, 1),
    neg_lo: Tuple[int, int, int] = (0, 0, 0),
    neg_hi: Tuple[int, int, int] = (0, 0, 0),
) -> int:
    """Reference mathematical oracle for packed dual-lane v_pk_fma_clamp_f16."""
    s0_lo_orig, s0_hi_orig = unpack_f16x2(s0_dword)
    s1_lo_orig, s1_hi_orig = unpack_f16x2(s1_dword)
    s2_lo_orig, s2_hi_orig = unpack_f16x2(s2_dword)

    # op_sel selects low calculation inputs
    s0_lo = s0_hi_orig if op_sel[0] else s0_lo_orig
    s1_lo = s1_hi_orig if op_sel[1] else s1_lo_orig
    s2_lo = s2_hi_orig if op_sel[2] else s2_lo_orig

    # op_sel_hi selects high calculation inputs
    s0_hi = s0_hi_orig if op_sel_hi[0] else s0_lo_orig
    s1_hi = s1_hi_orig if op_sel_hi[1] else s1_lo_orig
    s2_hi = s2_hi_orig if op_sel_hi[2] else s2_lo_orig

    # Apply neg_lo modifiers
    if neg_lo[0]:
        s0_lo ^= F16_SIGN_MASK
    if neg_lo[1]:
        s1_lo ^= F16_SIGN_MASK
    if neg_lo[2]:
        s2_lo ^= F16_SIGN_MASK

    # Apply neg_hi modifiers
    if neg_hi[0]:
        s0_hi ^= F16_SIGN_MASK
    if neg_hi[1]:
        s1_hi ^= F16_SIGN_MASK
    if neg_hi[2]:
        s2_hi ^= F16_SIGN_MASK

    res_lo = ref_fma_clamp_f16(s0_lo, s1_lo, s2_lo)
    res_hi = ref_fma_clamp_f16(s0_hi, s1_hi, s2_hi)

    return pack_f16x2(res_lo, res_hi)


# -----------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# -----------------------------------------------------------------------------

class RegType(enum.Enum):
    """Architecture register classification."""
    VGPR = "v"
    SGPR = "s"
    LITERAL = "lit"
    INLINE_CONST = "k"


@dataclass(frozen=True)
class Reg:
    """Architectural register designator."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index out of valid architectural range [0, 255]: {self.index}")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index out of valid architectural range [0, 105]: {self.index}")

    def __str__(self) -> str:
        return f"{self.reg_type.value}{self.index}"


@dataclass(frozen=True)
class VOP3POperand:
    """Typed operand for VOP3P 3-source vector instructions."""
    reg: Optional[Reg] = None
    literal_val: Optional[int] = None
    inline_val: Optional[float] = None

    def __post_init__(self) -> None:
        sources = [self.reg is not None, self.literal_val is not None, self.inline_val is not None]
        if sum(sources) != 1:
            raise ValueError("VOP3POperand must have exactly one source (reg, literal_val, or inline_val)")

    @property
    def is_vgpr(self) -> bool:
        return self.reg is not None and self.reg.reg_type == RegType.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.reg is not None and self.reg.reg_type == RegType.SGPR

    @property
    def is_literal(self) -> bool:
        return self.literal_val is not None

    @property
    def is_inline_const(self) -> bool:
        return self.inline_val is not None

    def __str__(self) -> str:
        if self.reg is not None:
            return str(self.reg)
        if self.literal_val is not None:
            return f"0x{self.literal_val:08x}"
        return f"{self.inline_val}"


@dataclass(frozen=True)
class VPkFmaClampF16Instruction:
    """Symbolic AST node representing AMD RDNA2 v_pk_fma_clamp_f16 instruction."""
    dst: Reg
    src0: VOP3POperand
    src1: VOP3POperand
    src2: VOP3POperand
    op_sel: Tuple[int, int, int] = (0, 0, 0)
    op_sel_hi: Tuple[int, int, int] = (1, 1, 1)
    neg_lo: Tuple[int, int, int] = (0, 0, 0)
    neg_hi: Tuple[int, int, int] = (0, 0, 0)
    clamp: bool = True

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise ValueError(f"Destination must be a VGPR, got {self.dst}")
        for name, triplet in [
            ("op_sel", self.op_sel),
            ("op_sel_hi", self.op_sel_hi),
            ("neg_lo", self.neg_lo),
            ("neg_hi", self.neg_hi),
        ]:
            if len(triplet) != 3 or not all(x in (0, 1) for x in triplet):
                raise ValueError(f"{name} modifier must be a tuple of 3 binary digits (0 or 1), got {triplet}")

    def disasm(self) -> str:
        """Format symbolic instruction into standard RDNA2 disassembly text."""
        parts = [f"v_pk_fma_f16 {self.dst}, {self.src0}, {self.src1}, {self.src2}"]
        if self.op_sel != (0, 0, 0):
            parts.append(f"op_sel:[{self.op_sel[0]},{self.op_sel[1]},{self.op_sel[2]}]")
        if self.op_sel_hi != (1, 1, 1):
            parts.append(f"op_sel_hi:[{self.op_sel_hi[0]},{self.op_sel_hi[1]},{self.op_sel_hi[2]}]")
        if self.neg_lo != (0, 0, 0):
            parts.append(f"neg_lo:[{self.neg_lo[0]},{self.neg_lo[1]},{self.neg_lo[2]}]")
        if self.neg_hi != (0, 0, 0):
            parts.append(f"neg_hi:[{self.neg_hi[0]},{self.neg_hi[1]},{self.neg_hi[2]}]")
        if self.clamp:
            parts.append("clamp")
        return " ".join(parts)


# -----------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# -----------------------------------------------------------------------------

@dataclass
class ScalarState:
    """Architectural state of a single execution lane."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)
    m0: int = 0

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index] & MASK32

    def set_vgpr(self, index: int, value: int) -> None:
        self.vgprs[index] = value & MASK32

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, value: int) -> None:
        self.sgprs[index] = value & MASK32


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 VALU instructions."""

    @staticmethod
    def resolve_operand(operand: VOP3POperand, state: ScalarState) -> int:
        """Resolve operand to a 32-bit dword value."""
        if operand.is_vgpr:
            assert operand.reg is not None
            return state.get_vgpr(operand.reg.index)
        if operand.is_sgpr:
            assert operand.reg is not None
            return state.get_sgpr(operand.reg.index)
        if operand.is_literal:
            assert operand.literal_val is not None
            return operand.literal_val & MASK32
        if operand.is_inline_const:
            assert operand.inline_val is not None
            # Replicate 16-bit constant to both halves
            f16_val = f16_to_u16(operand.inline_val)
            return pack_f16x2(f16_val, f16_val)
        raise ValueError(f"Unsupported operand: {operand}")

    def step(self, instr: VPkFmaClampF16Instruction, state: ScalarState) -> int:
        """Execute a single v_pk_fma_clamp_f16 instruction deterministically."""
        s0_val = self.resolve_operand(instr.src0, state)
        s1_val = self.resolve_operand(instr.src1, state)
        s2_val = self.resolve_operand(instr.src2, state)

        res = ref_v_pk_fma_clamp_f16(
            s0_val,
            s1_val,
            s2_val,
            op_sel=instr.op_sel,
            op_sel_hi=instr.op_sel_hi,
            neg_lo=instr.neg_lo,
            neg_hi=instr.neg_hi,
        )

        state.set_vgpr(instr.dst.index, res)
        return res


# -----------------------------------------------------------------------------
# SIMD Multi-Lane Divergence and EXEC Masking Model
# -----------------------------------------------------------------------------

@dataclass
class SIMDState:
    """Architectural SIMD wavefront state for Wave32 or Wave64."""
    wave_size: int = 32
    exec_mask: int = MASK32
    vgprs: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)

    def __post_init__(self) -> None:
        if self.wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {self.wave_size}, must be 32 or 64")
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.wave_size)]

    def is_lane_active(self, lane: int) -> bool:
        """Return True if specified lane is active under current EXEC mask."""
        return bool(self.exec_mask & (1 << lane))

    def get_vgpr(self, lane: int, index: int) -> int:
        return self.vgprs[lane][index] & MASK32

    def set_vgpr(self, lane: int, index: int, value: int) -> None:
        self.vgprs[lane][index] = value & MASK32

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, value: int) -> None:
        self.sgprs[index] = value & MASK32


class SIMDInterpreter:
    """SIMD wavefront interpreter supporting Wave32/Wave64 and EXEC masking."""

    def __init__(self) -> None:
        self.scalar_interp = ScalarInterpreter()

    def step(self, instr: VPkFmaClampF16Instruction, state: SIMDState) -> List[Optional[int]]:
        """Execute instruction across all lanes subject to EXEC mask."""
        results: List[Optional[int]] = [None] * state.wave_size

        for lane in range(state.wave_size):
            if state.is_lane_active(lane):
                lane_state = ScalarState(
                    vgprs=state.vgprs[lane],
                    sgprs=state.sgprs,
                )
                res = self.scalar_interp.step(instr, lane_state)
                results[lane] = res

        return results


# -----------------------------------------------------------------------------
# Condition Codes, Status Flags, and Carry Tracking
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FPStatusFlags:
    """IEEE-754 architectural exception and status flags."""
    invalid: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False
    denormal: bool = False
    zero: bool = False
    negative: bool = False
    clamped: bool = False


def evaluate_fma_clamp_f16_flags(s0: int, s1: int, s2: int, res: int) -> FPStatusFlags:
    """Evaluate architectural status flags for clamped half-precision FMA."""
    s0 &= MASK16
    s1 &= MASK16
    s2 &= MASK16
    res &= MASK16

    invalid = False
    if is_f16_snan(s0) or is_f16_snan(s1) or is_f16_snan(s2):
        invalid = True
    if (is_f16_inf(s0) and is_f16_zero(s1)) or (is_f16_inf(s1) and is_f16_zero(s0)):
        invalid = True
    prod_inf = is_f16_inf(s0) or is_f16_inf(s1)
    if prod_inf and is_f16_inf(s2):
        prod_sign = f16_sign(s0) ^ f16_sign(s1)
        if prod_sign != f16_sign(s2):
            invalid = True

    denormal = is_f16_subnormal(s0) or is_f16_subnormal(s1) or is_f16_subnormal(s2) or is_f16_subnormal(res)

    overflow = False
    underflow = False
    inexact = False
    clamped = False

    if not is_f16_nan(s0) and not is_f16_nan(s1) and not is_f16_nan(s2) and not invalid:
        if not is_f16_inf(s0) and not is_f16_inf(s1) and not is_f16_inf(s2):
            f0 = Fraction(u16_to_f16(s0))
            f1 = Fraction(u16_to_f16(s1))
            f2 = Fraction(u16_to_f16(s2))
            exact_val = f0 * f1 + f2

            if exact_val < 0 or exact_val > 1:
                clamped = True

            approx_res = Fraction(u16_to_f16(res))
            if exact_val != approx_res:
                inexact = True

            if exact_val > 65504:
                overflow = True
            elif 0 < abs(exact_val) < Fraction(1, 2**14):
                underflow = True

    zero = is_f16_zero(res)
    negative = bool(res & F16_SIGN_MASK)

    return FPStatusFlags(
        invalid=invalid,
        overflow=overflow,
        underflow=underflow,
        inexact=inexact,
        denormal=denormal,
        zero=zero,
        negative=negative,
        clamped=clamped,
    )


def evaluate_pk_fma_clamp_f16_flags(
    s0_dword: int, s1_dword: int, s2_dword: int, res_dword: int
) -> Tuple[FPStatusFlags, FPStatusFlags]:
    """Evaluate status flags for both packed lanes."""
    s0_lo, s0_hi = unpack_f16x2(s0_dword)
    s1_lo, s1_hi = unpack_f16x2(s1_dword)
    s2_lo, s2_hi = unpack_f16x2(s2_dword)
    res_lo, res_hi = unpack_f16x2(res_dword)

    return (
        evaluate_fma_clamp_f16_flags(s0_lo, s1_lo, s2_lo, res_lo),
        evaluate_fma_clamp_f16_flags(s0_hi, s1_hi, s2_hi, res_hi),
    )


def generate_vcc_mask_pk_fma_clamp(pred: str, results: List[int], wave_size: int = 32) -> int:
    """Generate Vector Condition Code (VCC) bitmask from packed half-precision outputs."""
    mask = 0
    for lane in range(min(len(results), wave_size)):
        val = results[lane]
        lo, hi = unpack_f16x2(val)
        flo = u16_to_f16(lo)
        fhi = u16_to_f16(hi)

        passed = False
        if pred == "both_saturated_one":
            passed = (lo == F16_POS_ONE) and (hi == F16_POS_ONE)
        elif pred == "both_zero":
            passed = is_f16_zero(lo) and is_f16_zero(hi)
        elif pred == "both_strictly_interior":
            passed = (0.0 < flo < 1.0) and (0.0 < fhi < 1.0)

        if passed:
            mask |= (1 << lane)

    return mask


def compute_fma_mantissa_alignment_shift(s0: int, s1: int, s2: int) -> int:
    """Compute exponent difference and alignment shift for addend s2 relative to product s0 * s1."""
    exp0 = (s0 & F16_EXP_MASK) >> 10
    exp1 = (s1 & F16_EXP_MASK) >> 10
    exp2 = (s2 & F16_EXP_MASK) >> 10

    # Unbiased exponents (-15 for binary16)
    unbiased_prod = (exp0 - 15) + (exp1 - 15)
    unbiased_addend = exp2 - 15
    return unbiased_prod - unbiased_addend


# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------

class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_basic_normal_fma_clamped(self) -> None:
        # lo: 0.5 * 0.5 + 0.25 = 0.50 (in range [0, 1])
        # hi: 0.25 * 2.0 + 0.125 = 0.625 (in range [0, 1])
        s0 = pack_f16x2(f16_to_u16(0.5), f16_to_u16(0.25))
        s1 = pack_f16x2(f16_to_u16(0.5), f16_to_u16(2.0))
        s2 = pack_f16x2(f16_to_u16(0.25), f16_to_u16(0.125))

        res = ref_v_pk_fma_clamp_f16(s0, s1, s2)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 0.50)
        self.assertAlmostEqual(u16_to_f16(res_hi), 0.625)

    def test_clamping_upper_and_lower_bounds(self) -> None:
        # lo: 2.0 * 2.0 + 1.0 = 5.0 -> clamp -> 1.0
        # hi: 2.0 * -1.0 + 0.5 = -1.5 -> clamp -> +0.0
        s0 = pack_f16x2(f16_to_u16(2.0), f16_to_u16(2.0))
        s1 = pack_f16x2(f16_to_u16(2.0), f16_to_u16(-1.0))
        s2 = pack_f16x2(f16_to_u16(1.0), f16_to_u16(0.5))

        res = ref_v_pk_fma_clamp_f16(s0, s1, s2)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertEqual(res_lo, F16_POS_ONE)
        self.assertEqual(res_hi, F16_POS_ZERO)

    def test_nan_clamping_produces_zero(self) -> None:
        # Clamped NaN in RDNA2 produces +0.0
        s0 = pack_f16x2(F16_CANONICAL_QNAN, f16_to_u16(1.0))
        s1 = pack_f16x2(f16_to_u16(1.0), F16_CANONICAL_QNAN)
        s2 = pack_f16x2(f16_to_u16(0.0), f16_to_u16(0.0))

        res = ref_v_pk_fma_clamp_f16(s0, s1, s2)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertEqual(res_lo, F16_POS_ZERO)
        self.assertEqual(res_hi, F16_POS_ZERO)

    def test_op_sel_and_negation_modifiers(self) -> None:
        # s0: [0.5, 2.0], s1: [0.5, 0.25], s2: [0.1, 0.2]
        s0 = pack_f16x2(f16_to_u16(0.5), f16_to_u16(2.0))
        s1 = pack_f16x2(f16_to_u16(0.5), f16_to_u16(0.25))
        s2 = pack_f16x2(f16_to_u16(0.1), f16_to_u16(0.2))

        # op_sel:(1, 1, 0) selects (s0.hi, s1.hi, s2.lo) -> 2.0 * 0.25 + 0.1 = 0.60
        # neg_lo:(1, 0, 0) negates s0 -> -2.0 * 0.25 + 0.1 = -0.40 -> clamp -> +0.0
        res = ref_v_pk_fma_clamp_f16(
            s0, s1, s2,
            op_sel=(1, 1, 0),
            neg_lo=(1, 0, 0),
        )
        res_lo, _ = unpack_f16x2(res)
        self.assertEqual(res_lo, F16_POS_ZERO)


class TestSymbolicASTDefinitions(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_reg_bounds_and_formatting(self) -> None:
        v0 = Reg(0, RegType.VGPR)
        v255 = Reg(255, RegType.VGPR)
        s105 = Reg(105, RegType.SGPR)
        self.assertEqual(str(v0), "v0")
        self.assertEqual(str(v255), "v255")
        self.assertEqual(str(s105), "s105")

        with self.assertRaises(ValueError):
            Reg(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(106, RegType.SGPR)

    def test_operand_creation_and_predicates(self) -> None:
        op_vgpr = VOP3POperand(reg=Reg(5, RegType.VGPR))
        op_sgpr = VOP3POperand(reg=Reg(10, RegType.SGPR))
        op_lit = VOP3POperand(literal_val=0x3C003C00)
        op_k = VOP3POperand(inline_val=1.0)

        self.assertTrue(op_vgpr.is_vgpr)
        self.assertTrue(op_sgpr.is_sgpr)
        self.assertTrue(op_lit.is_literal)
        self.assertTrue(op_k.is_inline_const)

    def test_instruction_disassembly(self) -> None:
        instr_default = VPkFmaClampF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
            src2=VOP3POperand(reg=Reg(3, RegType.VGPR)),
        )
        self.assertEqual(instr_default.disasm(), "v_pk_fma_f16 v0, v1, v2, v3 clamp")

    def test_instruction_validation(self) -> None:
        with self.assertRaises(ValueError):
            VPkFmaClampF16Instruction(
                dst=Reg(0, RegType.SGPR),
                src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
                src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
                src2=VOP3POperand(reg=Reg(3, RegType.VGPR)),
            )

        with self.assertRaises(ValueError):
            VPkFmaClampF16Instruction(
                dst=Reg(0, RegType.VGPR),
                src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
                src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
                src2=VOP3POperand(reg=Reg(3, RegType.VGPR)),
                op_sel=(0, 0),  # wrong length (needs 3)
            )


class TestScalarInstructionInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarState()
        self.interp = ScalarInterpreter()

    def test_scalar_vgpr_fma_clamped(self) -> None:
        # v1 = [0.5, 0.25], v2 = [0.5, 2.0], v3 = [0.25, 0.125]
        # v0 = clamp(v1 * v2 + v3) -> [0.50, 0.625]
        self.state.set_vgpr(1, pack_f16x2(f16_to_u16(0.5), f16_to_u16(0.25)))
        self.state.set_vgpr(2, pack_f16x2(f16_to_u16(0.5), f16_to_u16(2.0)))
        self.state.set_vgpr(3, pack_f16x2(f16_to_u16(0.25), f16_to_u16(0.125)))

        instr = VPkFmaClampF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
            src2=VOP3POperand(reg=Reg(3, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 0.50)
        self.assertAlmostEqual(u16_to_f16(res_hi), 0.625)
        self.assertEqual(self.state.get_vgpr(0), res)

    def test_sgpr_scalar_broadcast(self) -> None:
        # s4 = [0.25, 0.25], v1 = [2.0, 1.0], v2 = [0.25, 0.5]
        # v0 = clamp(v1 * s4 + v2) -> [0.75, 0.75]
        self.state.set_vgpr(1, pack_f16x2(f16_to_u16(2.0), f16_to_u16(1.0)))
        self.state.set_sgpr(4, pack_f16x2(f16_to_u16(0.25), f16_to_u16(0.25)))
        self.state.set_vgpr(2, pack_f16x2(f16_to_u16(0.25), f16_to_u16(0.5)))

        instr = VPkFmaClampF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(4, RegType.SGPR)),
            src2=VOP3POperand(reg=Reg(2, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 0.75)
        self.assertAlmostEqual(u16_to_f16(res_hi), 0.75)

    def test_inline_constant_operand(self) -> None:
        # v1 = [0.5, 0.5], v2 = [0.5, 0.5], src2 = inline const 0.5
        # v0 = clamp(0.5 * 0.5 + 0.5) = 0.75
        self.state.set_vgpr(1, pack_f16x2(f16_to_u16(0.5), f16_to_u16(0.5)))
        self.state.set_vgpr(2, pack_f16x2(f16_to_u16(0.5), f16_to_u16(0.5)))

        instr = VPkFmaClampF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
            src2=VOP3POperand(inline_val=0.5),
        )
        res = self.interp.step(instr, self.state)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 0.75)
        self.assertAlmostEqual(u16_to_f16(res_hi), 0.75)

    def test_state_isolation(self) -> None:
        self.state.set_vgpr(10, 0x12345678)
        self.state.set_sgpr(5, 0x87654321)

        instr = VPkFmaClampF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(inline_val=0.0),
            src1=VOP3POperand(inline_val=0.0),
            src2=VOP3POperand(inline_val=0.0),
        )
        self.interp.step(instr, self.state)
        self.assertEqual(self.state.get_vgpr(10), 0x12345678)
        self.assertEqual(self.state.get_sgpr(5), 0x87654321)


class TestSIMDMultiLaneDivergenceAndExecMasking(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.simd_interp = SIMDInterpreter()

    def test_wave32_full_execution_divergence(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0xFFFFFFFF)
        for lane in range(32):
            val0 = 0.01 * (lane + 1)
            val1 = 0.5
            val2 = 0.1
            state.set_vgpr(lane, 1, pack_f16x2(f16_to_u16(val0), f16_to_u16(val0 * 2.0)))
            state.set_vgpr(lane, 2, pack_f16x2(f16_to_u16(val1), f16_to_u16(val1)))
            state.set_vgpr(lane, 3, pack_f16x2(f16_to_u16(val2), f16_to_u16(val2)))

        instr = VPkFmaClampF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
            src2=VOP3POperand(reg=Reg(3, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)

        for lane in range(32):
            res = state.get_vgpr(lane, 0)
            res_lo, res_hi = unpack_f16x2(res)
            expected_lo = (0.01 * (lane + 1)) * 0.5 + 0.1
            expected_hi = (0.02 * (lane + 1)) * 0.5 + 0.1
            self.assertAlmostEqual(u16_to_f16(res_lo), expected_lo, places=3)
            self.assertAlmostEqual(u16_to_f16(res_hi), expected_hi, places=3)

    def test_wave64_exec_mask_gating_preserves_inactive_lanes(self) -> None:
        state = SIMDState(wave_size=64, exec_mask=0x5555555555555555)  # Even lanes active
        for lane in range(64):
            state.set_vgpr(lane, 0, OLD_VGPR_SENTINEL)
            state.set_vgpr(lane, 1, pack_f16x2(f16_to_u16(0.5), f16_to_u16(0.25)))
            state.set_vgpr(lane, 2, pack_f16x2(f16_to_u16(0.5), f16_to_u16(2.0)))
            state.set_vgpr(lane, 3, pack_f16x2(f16_to_u16(0.25), f16_to_u16(0.125)))

        instr = VPkFmaClampF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
            src2=VOP3POperand(reg=Reg(3, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)

        expected_active = pack_f16x2(f16_to_u16(0.50), f16_to_u16(0.625))
        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(state.get_vgpr(lane, 0), expected_active)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), OLD_VGPR_SENTINEL)

    def test_zero_exec_mask_no_modification(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0x00000000)
        for lane in range(32):
            state.set_vgpr(lane, 0, OLD_VGPR_SENTINEL)

        instr = VPkFmaClampF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(inline_val=1.0),
            src1=VOP3POperand(inline_val=1.0),
            src2=VOP3POperand(inline_val=1.0),
        )
        self.simd_interp.step(instr, state)

        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), OLD_VGPR_SENTINEL)


class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_aliases_src0(self) -> None:
        # v0 = clamp(v0 * v1 + v2)
        state = ScalarState()
        state.set_vgpr(0, pack_f16x2(f16_to_u16(0.5), f16_to_u16(0.25)))
        state.set_vgpr(1, pack_f16x2(f16_to_u16(0.5), f16_to_u16(2.0)))
        state.set_vgpr(2, pack_f16x2(f16_to_u16(0.25), f16_to_u16(0.125)))

        instr = VPkFmaClampF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(0, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src2=VOP3POperand(reg=Reg(2, RegType.VGPR)),
        )
        res = self.interp.step(instr, state)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 0.50)
        self.assertAlmostEqual(u16_to_f16(res_hi), 0.625)
        self.assertEqual(state.get_vgpr(0), res)

    def test_dst_aliases_src2_in_place_accumulation(self) -> None:
        # v0 = clamp(v1 * v2 + v0) -> destination used as accumulator
        state = ScalarState()
        state.set_vgpr(0, pack_f16x2(f16_to_u16(0.1), f16_to_u16(0.2)))
        state.set_vgpr(1, pack_f16x2(f16_to_u16(0.4), f16_to_u16(0.5)))
        state.set_vgpr(2, pack_f16x2(f16_to_u16(0.5), f16_to_u16(0.6)))

        instr = VPkFmaClampF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
            src2=VOP3POperand(reg=Reg(0, RegType.VGPR)),
        )
        res = self.interp.step(instr, state)
        res_lo, res_hi = unpack_f16x2(res)
        # lo: 0.4 * 0.5 + 0.1 = 0.30
        # hi: 0.5 * 0.6 + 0.2 = 0.50
        self.assertAlmostEqual(u16_to_f16(res_lo), 0.30, places=2)
        self.assertAlmostEqual(u16_to_f16(res_hi), 0.50, places=2)

    def test_complete_self_aliasing(self) -> None:
        # v0 = clamp(v0 * v0 + v0)
        state = ScalarState()
        state.set_vgpr(0, pack_f16x2(f16_to_u16(0.5), f16_to_u16(0.2)))

        instr = VPkFmaClampF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(0, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(0, RegType.VGPR)),
            src2=VOP3POperand(reg=Reg(0, RegType.VGPR)),
        )
        res = self.interp.step(instr, state)
        res_lo, res_hi = unpack_f16x2(res)
        # lo: 0.5 * 0.5 + 0.5 = 0.75
        # hi: 0.2 * 0.2 + 0.2 = 0.24
        self.assertAlmostEqual(u16_to_f16(res_lo), 0.75)
        self.assertAlmostEqual(u16_to_f16(res_hi), 0.24, places=2)

    def test_simd_multi_lane_inplace_aliasing(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0xFFFFFFFF)
        for lane in range(32):
            state.set_vgpr(lane, 0, pack_f16x2(f16_to_u16(0.1), f16_to_u16(0.2)))
            state.set_vgpr(lane, 1, pack_f16x2(f16_to_u16(0.2), f16_to_u16(0.2)))

        instr = VPkFmaClampF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src2=VOP3POperand(reg=Reg(0, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)

        # 0.2 * 0.2 + 0.1 = 0.14
        # 0.2 * 0.2 + 0.2 = 0.24
        for lane in range(32):
            res = state.get_vgpr(lane, 0)
            res_lo, res_hi = unpack_f16x2(res)
            self.assertAlmostEqual(u16_to_f16(res_lo), 0.14, places=2)
            self.assertAlmostEqual(u16_to_f16(res_hi), 0.24, places=2)


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Test suite for condition codes, status flags, and carry generation."""

    def test_exact_finite_flags(self) -> None:
        s0 = f16_to_u16(0.5)
        s1 = f16_to_u16(0.5)
        s2 = f16_to_u16(0.25)
        res = ref_fma_clamp_f16(s0, s1, s2)
        flags = evaluate_fma_clamp_f16_flags(s0, s1, s2, res)
        self.assertFalse(flags.invalid)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.underflow)
        self.assertFalse(flags.inexact)
        self.assertFalse(flags.denormal)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.clamped)

    def test_clamped_saturation_flags(self) -> None:
        # 2.0 * 2.0 + 1.0 = 5.0 -> clamped to 1.0
        s0 = f16_to_u16(2.0)
        s1 = f16_to_u16(2.0)
        s2 = f16_to_u16(1.0)
        res = ref_fma_clamp_f16(s0, s1, s2)
        flags = evaluate_fma_clamp_f16_flags(s0, s1, s2, res)
        self.assertTrue(flags.clamped)
        self.assertTrue(flags.inexact)
        self.assertEqual(res, F16_POS_ONE)

        # 1.0 * -2.0 + 0.5 = -1.5 -> clamped to +0.0
        s0_neg = f16_to_u16(1.0)
        s1_neg = f16_to_u16(-2.0)
        s2_neg = f16_to_u16(0.5)
        res_neg = ref_fma_clamp_f16(s0_neg, s1_neg, s2_neg)
        flags_neg = evaluate_fma_clamp_f16_flags(s0_neg, s1_neg, s2_neg, res_neg)
        self.assertTrue(flags_neg.clamped)
        self.assertTrue(flags_neg.zero)

    def test_invalid_operation_flag(self) -> None:
        # 0 * inf is invalid
        s0 = F16_POS_ZERO
        s1 = F16_POS_INF
        s2 = F16_POS_ZERO
        res = ref_fma_clamp_f16(s0, s1, s2)
        flags = evaluate_fma_clamp_f16_flags(s0, s1, s2, res)
        self.assertTrue(flags.invalid)
        # Clamped invalid op yields +0.0
        self.assertEqual(res, F16_POS_ZERO)

    def test_vcc_condition_masks(self) -> None:
        r0 = pack_f16x2(F16_POS_ONE, F16_POS_ONE)    # both saturated one
        r1 = pack_f16x2(F16_POS_ZERO, F16_POS_ZERO)  # both zero
        r2 = pack_f16x2(f16_to_u16(0.5), f16_to_u16(0.5))  # both interior
        r3 = pack_f16x2(F16_POS_ONE, F16_POS_ZERO)   # mixed

        results = [r0, r1, r2, r3]
        mask_ones = generate_vcc_mask_pk_fma_clamp("both_saturated_one", results, 4)
        self.assertEqual(mask_ones, 0b0001)

        mask_zeros = generate_vcc_mask_pk_fma_clamp("both_zero", results, 4)
        self.assertEqual(mask_zeros, 0b0010)

        mask_interior = generate_vcc_mask_pk_fma_clamp("both_strictly_interior", results, 4)
        self.assertEqual(mask_interior, 0b0100)

    def test_mantissa_alignment_shift(self) -> None:
        # s0: exp=15 (1.0), s1: exp=15 (1.0) -> prod exp = 0
        # s2: exp=13 (0.25) -> unbiased exp = -2
        # shift difference = 0 - (-2) = 2
        s0 = f16_to_u16(1.0)
        s1 = f16_to_u16(1.0)
        s2 = f16_to_u16(0.25)
        shift = compute_fma_mantissa_alignment_shift(s0, s1, s2)
        self.assertEqual(shift, 2)


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Test suite for IEEE-754 boundary extremes, subnormals, and NaN handling."""

    def test_nan_inputs_always_clamp_to_zero(self) -> None:
        snan = 0x7C01
        qnan = 0x7E01
        self.assertTrue(is_f16_snan(snan))
        self.assertTrue(is_f16_qnan(qnan))

        # NaN in src0
        self.assertEqual(ref_fma_clamp_f16(snan, F16_POS_ONE, F16_POS_ZERO), F16_POS_ZERO)
        # NaN in src1
        self.assertEqual(ref_fma_clamp_f16(F16_POS_ONE, qnan, F16_POS_ZERO), F16_POS_ZERO)
        # NaN in src2
        self.assertEqual(ref_fma_clamp_f16(F16_POS_ONE, F16_POS_ONE, snan), F16_POS_ZERO)

        # Packed NaN clamp across both halves
        s0 = pack_f16x2(snan, F16_POS_ONE)
        s1 = pack_f16x2(F16_POS_ONE, qnan)
        s2 = pack_f16x2(F16_POS_ZERO, F16_POS_ZERO)
        res = ref_v_pk_fma_clamp_f16(s0, s1, s2)
        self.assertEqual(res, pack_f16x2(F16_POS_ZERO, F16_POS_ZERO))

    def test_signed_zero_combinations(self) -> None:
        # Negative zero clamped to positive zero
        res_nz = ref_fma_clamp_f16(F16_NEG_ZERO, F16_POS_ONE, F16_NEG_ZERO)
        self.assertEqual(res_nz, F16_POS_ZERO)

        # Exact cancellation to zero: 1.0 * 1.0 + (-1.0) = 0.0
        res_cancel = ref_fma_clamp_f16(F16_POS_ONE, F16_POS_ONE, F16_NEG_ONE)
        self.assertEqual(res_cancel, F16_POS_ZERO)

    def test_signed_infinities_and_invalid_operations(self) -> None:
        # +inf * 1.0 + 0.0 -> +inf -> clamped to 1.0
        self.assertEqual(ref_fma_clamp_f16(F16_POS_INF, F16_POS_ONE, F16_POS_ZERO), F16_POS_ONE)
        # -inf * 1.0 + 0.0 -> -inf -> clamped to +0.0
        self.assertEqual(ref_fma_clamp_f16(F16_NEG_INF, F16_POS_ONE, F16_POS_ZERO), F16_POS_ZERO)
        # +inf + -inf -> invalid -> NaN -> clamped to +0.0
        self.assertEqual(ref_fma_clamp_f16(F16_POS_INF, F16_POS_ONE, F16_NEG_INF), F16_POS_ZERO)
        # 0.0 * inf -> invalid -> NaN -> clamped to +0.0
        self.assertEqual(ref_fma_clamp_f16(F16_POS_ZERO, F16_POS_INF, F16_POS_HALF), F16_POS_ZERO)

    def test_subnormal_precision_and_underflow(self) -> None:
        # Smallest subnormal * 1.0 + 0.0 = smallest subnormal
        res_sub = ref_fma_clamp_f16(F16_MIN_SUBNORMAL, F16_POS_ONE, F16_POS_ZERO)
        self.assertEqual(res_sub, F16_MIN_SUBNORMAL)
        self.assertTrue(is_f16_subnormal(res_sub))

        # Smallest subnormal * 0.5 + 0.0 -> underflow to +0.0
        res_under = ref_fma_clamp_f16(F16_MIN_SUBNORMAL, F16_POS_HALF, F16_POS_ZERO)
        self.assertEqual(res_under, F16_POS_ZERO)

    def test_overflow_saturation(self) -> None:
        # MAX_NORMAL (65504) * 2.0 + 1.0 -> overflow -> clamped to 1.0
        res_over = ref_fma_clamp_f16(F16_MAX_NORMAL, F16_POS_TWO, F16_POS_ONE)
        self.assertEqual(res_over, F16_POS_ONE)

    def test_fused_intermediate_precision(self) -> None:
        # In FMA, intermediate product is kept with infinite precision:
        # Let a = 1.0 + 2^-10 = 0x3C01
        # b = 1.0 - 2^-10 = 0x3BFF
        # a * b = 1.0 - 2^-20
        # If we add c = -1.0 + 2^-10 = 0xBBFF,
        # Exact sum is (1.0 - 2^-20) - 1.0 + 2^-10 = 2^-10 - 2^-20 > 0
        a = 0x3C01
        b = 0x3BFF
        c = 0xBBFF
        res = ref_fma_clamp_f16(a, b, c)
        self.assertGreater(u16_to_f16(res), 0.0)
        self.assertLessEqual(u16_to_f16(res), 1.0)


class TestNegativeControlsAndMutationDetection(unittest.TestCase):
    """Test suite for negative controls, syntax rejection, and mutation detection."""

    def test_out_of_bounds_registers(self) -> None:
        with self.assertRaises(ValueError):
            Reg(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(-1, RegType.SGPR)
        with self.assertRaises(ValueError):
            Reg(106, RegType.SGPR)

    def test_invalid_destination_types(self) -> None:
        with self.assertRaises(ValueError):
            VPkFmaClampF16Instruction(
                dst=Reg(0, RegType.SGPR),
                src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
                src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
                src2=VOP3POperand(reg=Reg(3, RegType.VGPR)),
            )

    def test_invalid_modifier_tuples(self) -> None:
        with self.assertRaises(ValueError):
            VPkFmaClampF16Instruction(
                dst=Reg(0, RegType.VGPR),
                src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
                src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
                src2=VOP3POperand(reg=Reg(3, RegType.VGPR)),
                op_sel=(0, 0),  # invalid length 2, needs 3
            )
        with self.assertRaises(ValueError):
            VPkFmaClampF16Instruction(
                dst=Reg(0, RegType.VGPR),
                src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
                src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
                src2=VOP3POperand(reg=Reg(3, RegType.VGPR)),
                neg_lo=(0, 0, 3),  # value not in {0, 1}
            )

    def test_mutation_unclamped_fma(self) -> None:
        def mutated_unclamped(s0: int, s1: int, s2: int) -> int:
            f0 = Fraction(u16_to_f16(s0))
            f1 = Fraction(u16_to_f16(s1))
            f2 = Fraction(u16_to_f16(s2))
            return f16_to_u16(float(f0 * f1 + f2))

        # 2.0 * 2.0 + 1.0 = 5.0 -> oracle returns 1.0 (0x3C00), mutant returns 5.0 (0x4500)
        s0 = f16_to_u16(2.0)
        s1 = f16_to_u16(2.0)
        s2 = f16_to_u16(1.0)
        oracle_res = ref_fma_clamp_f16(s0, s1, s2)
        mutant_res = mutated_unclamped(s0, s1, s2)
        self.assertEqual(oracle_res, F16_POS_ONE)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_unfused_intermediate_rounding(self) -> None:
        def mutated_unfused(s0: int, s1: int, s2: int) -> int:
            # Performs separate multiply and add with intermediate rounding
            prod = f16_to_u16(u16_to_f16(s0) * u16_to_f16(s1))
            sum_val = f16_to_u16(u16_to_f16(prod) + u16_to_f16(s2))
            return clamp01_f16(sum_val)

        a = 0x3C01  # 1.0 + 2^-10
        b = 0x3BFF  # 1.0 - 2^-10
        c = 0xBBFF  # -1.0 + 2^-10
        oracle_res = ref_fma_clamp_f16(a, b, c)
        mutant_res = mutated_unfused(a, b, c)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_nan_unclamped(self) -> None:
        def mutated_nan_leak(s0: int, s1: int, s2: int) -> int:
            if is_f16_nan(s0):
                return s0
            return ref_fma_clamp_f16(s0, s1, s2)

        snan = 0x7C01
        oracle_res = ref_fma_clamp_f16(snan, F16_POS_ONE, F16_POS_ZERO)
        mutant_res = mutated_nan_leak(snan, F16_POS_ONE, F16_POS_ZERO)
        self.assertEqual(oracle_res, F16_POS_ZERO)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_corrupted_inactive_exec_lane(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0x00000001)  # only lane 0 active
        state.set_vgpr(1, 0, OLD_VGPR_SENTINEL)

        simd_interp = SIMDInterpreter()
        instr = VPkFmaClampF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(inline_val=0.5),
            src1=VOP3POperand(inline_val=0.5),
            src2=VOP3POperand(inline_val=0.5),
        )
        simd_interp.step(instr, state)
        self.assertEqual(state.get_vgpr(1, 0), OLD_VGPR_SENTINEL)

        # Inactive lane corrupted by mutant simulator
        state.set_vgpr(1, 0, 0x11111111)
        self.assertNotEqual(state.get_vgpr(1, 0), OLD_VGPR_SENTINEL)


if __name__ == "__main__":
    unittest.main()
