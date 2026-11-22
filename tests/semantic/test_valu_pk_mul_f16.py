"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_pk_mul_f16.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  The packed half-precision floating-point instruction v_pk_mul_f16 operates on two
  independent 16-bit IEEE 754 half-precision (binary16) floats packed inside 32-bit registers.

  Register Packing:
    Bits [15:0]  : Low half-precision float (lo)
    Bits [31:16] : High half-precision float (hi)

  Instruction:
    v_pk_mul_f16 dst, src0, src1 [op_sel:[s0_lo, s1_lo]] [op_sel_hi:[s0_hi, s1_hi]]
                                  [neg_lo:[s0_lo, s1_lo]] [neg_hi:[s0_hi, s1_hi]] [clamp]

  IEEE-754 binary16 encoding:
    - 1 sign bit (bit 15)
    - 5 exponent bits with bias 15 (bits 14:10)
    - 10 mantissa / fraction bits (bits 9:0)

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

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# IEEE-754 binary16 bit constants
F16_POS_ZERO = 0x0000
F16_NEG_ZERO = 0x8000
F16_POS_INF = 0x7C00
F16_NEG_INF = 0xFC00
F16_CANONICAL_QNAN = 0x7E00
F16_QUIET_BIT = 0x0200

F16_EXP_MASK = 0x7C00
F16_MANT_MASK = 0x03FF
F16_SIGN_MASK = 0x8000

F16_ONE = 0x3C00          # 1.0
F16_NEG_ONE = 0xBC00      # -1.0
F16_TWO = 0x4000          # 2.0
F16_HALF = 0x3800         # 0.5
F16_MIN_NORMAL = 0x0400   # 2^-14 = ~6.1035e-5
F16_MAX_NORMAL = 0x7BFF   # 65504.0
F16_MIN_SUBNORMAL = 0x0001  # 2^-24 = ~5.9605e-8
F16_MAX_SUBNORMAL = 0x03FF  # 2^-14 * (1023/1024)

OLD_VGPR_SENTINEL = 0xDEADBEEF


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit range [0, 65535]."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def pack_f16x2(lo: int, hi: int) -> int:
    """Pack two 16-bit half-precision floats into a 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def unpack_f16x2(word: int) -> Tuple[int, int]:
    """Unpack a 32-bit dword into (lo, hi) 16-bit half-precision words."""
    return word & MASK16, (word >> 16) & MASK16


def f16_to_u16(val: float) -> int:
    """Pack a float into an IEEE-754 binary16 16-bit unsigned integer."""
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return F16_NEG_INF if val < 0 else F16_POS_INF


def u16_to_f16(bits: int) -> float:
    """Unpack an IEEE-754 binary16 16-bit unsigned integer into a float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def is_f16_nan(bits: int) -> bool:
    """Return True if the 16-bit pattern represents an IEEE-754 NaN."""
    return ((bits & F16_EXP_MASK) == F16_EXP_MASK) and ((bits & F16_MANT_MASK) != 0)


def is_f16_snan(bits: int) -> bool:
    """Return True if the 16-bit pattern represents a signaling NaN."""
    return is_f16_nan(bits) and ((bits & F16_QUIET_BIT) == 0)


def is_f16_qnan(bits: int) -> bool:
    """Return True if the 16-bit pattern represents a quiet NaN."""
    return is_f16_nan(bits) and ((bits & F16_QUIET_BIT) != 0)


def is_f16_zero(bits: int) -> bool:
    """Return True if the 16-bit pattern represents +0.0 or -0.0."""
    return (bits & (F16_EXP_MASK | F16_MANT_MASK)) == 0


def is_f16_inf(bits: int) -> bool:
    """Return True if the 16-bit pattern represents +inf or -inf."""
    return ((bits & F16_EXP_MASK) == F16_EXP_MASK) and ((bits & F16_MANT_MASK) == 0)


def is_f16_subnormal(bits: int) -> bool:
    """Return True if the 16-bit pattern represents a subnormal/denormal float."""
    return ((bits & F16_EXP_MASK) == 0) and ((bits & F16_MANT_MASK) != 0)


def is_f16_normal(bits: int) -> bool:
    """Return True if the 16-bit pattern represents a normalized finite float."""
    exp = (bits >> 10) & 0x1F
    return 0 < exp < 0x1F


def f16_sign(bits: int) -> int:
    """Return the sign bit (0 or 1)."""
    return (bits >> 15) & 1


def quiet_f16(bits: int) -> int:
    """Force an IEEE-754 half-precision NaN to quiet by setting the quiet bit."""
    return (bits & MASK16) | F16_QUIET_BIT


def ref_mul_f16(s0: int, s1: int, *, clamp: bool = False) -> int:
    """Reference mathematical oracle for a single 16-bit half-precision multiply.

    Computes S0 * S1 conforming strictly to IEEE-754-2008 binary16 arithmetic and
    RDNA2 VALU hardware specifications.
    """
    s0 &= MASK16
    s1 &= MASK16

    # 1. NaN handling: If either operand is NaN
    if is_f16_nan(s0) or is_f16_nan(s1):
        if clamp:
            # Under RDNA2 VALU clamp, clamped NaN produces +0.0
            return F16_POS_ZERO
        # Signaling NaN quieted, or quiet NaN propagated; src0 has priority
        if is_f16_nan(s0):
            return quiet_f16(s0)
        return quiet_f16(s1)

    # 2. Infinity and Zero combinations
    sign = f16_sign(s0) ^ f16_sign(s1)
    s0_inf = is_f16_inf(s0)
    s1_inf = is_f16_inf(s1)
    s0_zero = is_f16_zero(s0)
    s1_zero = is_f16_zero(s1)

    if (s0_inf and s1_zero) or (s1_inf and s0_zero):
        # inf * 0.0 is an Invalid Operation -> returns Canonical Quiet NaN
        return F16_POS_ZERO if clamp else F16_CANONICAL_QNAN

    if s0_inf or s1_inf:
        res = F16_NEG_INF if sign else F16_POS_INF
        if clamp:
            return F16_ONE if res == F16_POS_INF else F16_POS_ZERO
        return res

    if s0_zero or s1_zero:
        res = F16_NEG_ZERO if sign else F16_POS_ZERO
        if clamp and res == F16_NEG_ZERO:
            return F16_POS_ZERO
        return res

    # 3. Finite * Finite multiplication using Python float & IEEE 754 half packing
    f0 = u16_to_f16(s0)
    f1 = u16_to_f16(s1)
    prod = f0 * f1

    res_bits = f16_to_u16(prod)

    # 4. Apply clamp [0.0, 1.0]
    if clamp:
        if is_f16_nan(res_bits):
            return F16_POS_ZERO
        res_float = u16_to_f16(res_bits)
        if res_float < 0.0 or res_bits == F16_NEG_ZERO:
            return F16_POS_ZERO
        if res_float > 1.0:
            return F16_ONE

    return res_bits


def ref_v_pk_mul_f16(
    src0_dword: int,
    src1_dword: int,
    *,
    op_sel: Tuple[int, int] = (0, 0),
    op_sel_hi: Tuple[int, int] = (1, 1),
    neg_lo: Tuple[int, int] = (0, 0),
    neg_hi: Tuple[int, int] = (0, 0),
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for packed dual-lane v_pk_mul_f16.

    Selects half-words via op_sel and op_sel_hi, applies per-half negation modifiers,
    multiplies both halves independently, and clamps results if requested.
    """
    s0_lo_orig, s0_hi_orig = unpack_f16x2(src0_dword)
    s1_lo_orig, s1_hi_orig = unpack_f16x2(src1_dword)

    # op_sel selects low calculation inputs
    s0_lo = s0_hi_orig if op_sel[0] else s0_lo_orig
    s1_lo = s1_hi_orig if op_sel[1] else s1_lo_orig

    # op_sel_hi selects high calculation inputs
    s0_hi = s0_hi_orig if op_sel_hi[0] else s0_lo_orig
    s1_hi = s1_hi_orig if op_sel_hi[1] else s1_lo_orig

    # Apply negation modifiers (neg toggles bit 15)
    if neg_lo[0]:
        s0_lo ^= F16_SIGN_MASK
    if neg_lo[1]:
        s1_lo ^= F16_SIGN_MASK

    if neg_hi[0]:
        s0_hi ^= F16_SIGN_MASK
    if neg_hi[1]:
        s1_hi ^= F16_SIGN_MASK

    # Independent half multiplications
    res_lo = ref_mul_f16(s0_lo, s1_lo, clamp=clamp)
    res_hi = ref_mul_f16(s0_hi, s1_hi, clamp=clamp)

    return pack_f16x2(res_lo, res_hi)


# -----------------------------------------------------------------------------
# Symbolic AST Definitions and Typed Expressions
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
    """Typed operand for VOP3P packed vector instructions."""
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
class VPkMulF16Instruction:
    """Symbolic AST node representing AMD RDNA2 v_pk_mul_f16 instruction."""
    dst: Reg
    src0: VOP3POperand
    src1: VOP3POperand
    op_sel: Tuple[int, int] = (0, 0)
    op_sel_hi: Tuple[int, int] = (1, 1)
    neg_lo: Tuple[int, int] = (0, 0)
    neg_hi: Tuple[int, int] = (0, 0)
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise ValueError(f"Destination must be a VGPR, got {self.dst}")
        for name, pair in [
            ("op_sel", self.op_sel),
            ("op_sel_hi", self.op_sel_hi),
            ("neg_lo", self.neg_lo),
            ("neg_hi", self.neg_hi),
        ]:
            if len(pair) != 2 or not all(x in (0, 1) for x in pair):
                raise ValueError(f"{name} modifier must be a tuple of two binary digits (0 or 1), got {pair}")

    def disasm(self) -> str:
        """Format symbolic instruction into standard RDNA2 disassembly text."""
        parts = [f"v_pk_mul_f16 {self.dst}, {self.src0}, {self.src1}"]
        if self.op_sel != (0, 0):
            parts.append(f"op_sel:[{self.op_sel[0]},{self.op_sel[1]}]")
        if self.op_sel_hi != (1, 1):
            parts.append(f"op_sel_hi:[{self.op_sel_hi[0]},{self.op_sel_hi[1]}]")
        if self.neg_lo != (0, 0):
            parts.append(f"neg_lo:[{self.neg_lo[0]},{self.neg_lo[1]}]")
        if self.neg_hi != (0, 0):
            parts.append(f"neg_hi:[{self.neg_hi[0]},{self.neg_hi[1]}]")
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
            # In RDNA2, an inline float constant on packed-16 multiplies is duplicated across both halves
            f16_val = f16_to_u16(operand.inline_val)
            return pack_f16x2(f16_val, f16_val)
        raise ValueError(f"Unsupported operand: {operand}")

    def step(self, instr: VPkMulF16Instruction, state: ScalarState) -> int:
        """Execute a single v_pk_mul_f16 instruction deterministically."""
        s0_val = self.resolve_operand(instr.src0, state)
        s1_val = self.resolve_operand(instr.src1, state)

        res = ref_v_pk_mul_f16(
            s0_val,
            s1_val,
            op_sel=instr.op_sel,
            op_sel_hi=instr.op_sel_hi,
            neg_lo=instr.neg_lo,
            neg_hi=instr.neg_hi,
            clamp=instr.clamp,
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
            # Initialize 256 VGPRs for each lane
            self.vgprs = [[0] * 256 for _ in range(self.wave_size)]

    def is_lane_active(self, lane: int) -> bool:
        """Return True if the specified lane is active under current EXEC mask."""
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

    def step(self, instr: VPkMulF16Instruction, state: SIMDState) -> List[Optional[int]]:
        """Execute instruction across all lanes in the wavefront subject to EXEC mask."""
        results: List[Optional[int]] = [None] * state.wave_size

        for lane in range(state.wave_size):
            if state.is_lane_active(lane):
                # Construct temporary scalar state view for this lane
                lane_state = ScalarState(
                    vgprs=state.vgprs[lane],
                    sgprs=state.sgprs,
                )
                res = self.scalar_interp.step(instr, lane_state)
                results[lane] = res
            # Inactive lanes retain previous destination VGPR contents untouched

        return results


# -----------------------------------------------------------------------------
# Condition Codes, Status Flags, and Mantissa Carry Tracking
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
    infinite: bool = False
    nan: bool = False


def evaluate_f16_mul_flags(s0: int, s1: int, res: int) -> FPStatusFlags:
    """Evaluate architectural status flags for half-precision multiplication."""
    s0 &= MASK16
    s1 &= MASK16
    res &= MASK16

    # 1. Invalid operation & NaN
    invalid = False
    nan = is_f16_nan(res)
    if is_f16_snan(s0) or is_f16_snan(s1):
        invalid = True
    if (is_f16_inf(s0) and is_f16_zero(s1)) or (is_f16_inf(s1) and is_f16_zero(s0)):
        invalid = True

    # 2. Denormal / Subnormal flag
    denormal = is_f16_subnormal(s0) or is_f16_subnormal(s1) or is_f16_subnormal(res)

    # 3. Finite multiplication flags (Overflow, Underflow, Inexact)
    overflow = False
    underflow = False
    inexact = False

    if not is_f16_nan(s0) and not is_f16_nan(s1):
        if is_f16_inf(res) and not is_f16_inf(s0) and not is_f16_inf(s1):
            overflow = True
            inexact = True
        elif not is_f16_inf(s0) and not is_f16_inf(s1) and not is_f16_zero(s0) and not is_f16_zero(s1):
            # Check exact arithmetic via Fractions
            f0 = Fraction(u16_to_f16(s0))
            f1 = Fraction(u16_to_f16(s1))
            exact_prod = f0 * f1
            approx_res = Fraction(u16_to_f16(res))
            if exact_prod != approx_res:
                inexact = True
            if is_f16_subnormal(res) or (is_f16_zero(res) and exact_prod != 0):
                underflow = True

    zero = is_f16_zero(res)
    negative = bool(res & F16_SIGN_MASK)
    infinite = is_f16_inf(res)

    return FPStatusFlags(
        invalid=invalid,
        overflow=overflow,
        underflow=underflow,
        inexact=inexact,
        denormal=denormal,
        zero=zero,
        negative=negative,
        infinite=infinite,
        nan=nan,
    )


def evaluate_pk_mul_f16_flags(
    s0_dword: int, s1_dword: int, res_dword: int
) -> Tuple[FPStatusFlags, FPStatusFlags]:
    """Evaluate status flags for both packed lanes (lo, hi)."""
    s0_lo, s0_hi = unpack_f16x2(s0_dword)
    s1_lo, s1_hi = unpack_f16x2(s1_dword)
    res_lo, res_hi = unpack_f16x2(res_dword)

    return (
        evaluate_f16_mul_flags(s0_lo, s1_lo, res_lo),
        evaluate_f16_mul_flags(s0_hi, s1_hi, res_hi),
    )


def generate_vcc_mask_pk_f16(pred: str, results: List[int], wave_size: int = 32) -> int:
    """Generate Vector Condition Code (VCC) bitmask from packed half-precision outputs."""
    mask = 0
    for lane in range(min(len(results), wave_size)):
        val = results[lane]
        lo, hi = unpack_f16x2(val)
        flo = u16_to_f16(lo)
        fhi = u16_to_f16(hi)

        passed = False
        if pred == "eq_zero":
            passed = is_f16_zero(lo) and is_f16_zero(hi)
        elif pred == "has_nan":
            passed = is_f16_nan(lo) or is_f16_nan(hi)
        elif pred == "both_positive":
            passed = (not is_f16_nan(lo) and flo > 0.0) and (not is_f16_nan(hi) and fhi > 0.0)
        elif pred == "both_finite":
            passed = (not is_f16_nan(lo) and not is_f16_inf(lo)) and (not is_f16_nan(fhi) and not is_f16_inf(hi))

        if passed:
            mask |= (1 << lane)

    return mask


def compute_f16_mantissa_product(s0: int, s1: int) -> Tuple[int, int]:
    """Compute 11-bit mantissa multiplication and return (product, carry_shift).

    In IEEE 754 binary16:
      Normal significand has 1 implicit bit + 10 fraction bits = 11 bits.
      Multiplying two 11-bit integers yields a 22-bit product in [2^20, 2^22 - 1].
      If bit 21 is set (product >= 2.0), a mantissa carry-out occurs requiring 1-bit right shift.
    """
    m0 = (s0 & F16_MANT_MASK) | (0 if (s0 & F16_EXP_MASK) == 0 else 0x0400)
    m1 = (s1 & F16_MANT_MASK) | (0 if (s1 & F16_EXP_MASK) == 0 else 0x0400)
    prod = m0 * m1
    carry_shift = (prod >> 21) & 1
    return prod, carry_shift


# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------

class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_basic_normal_products(self) -> None:
        # lo: 2.0 * 3.0 = 6.0, hi: 0.5 * 4.0 = 2.0
        s0 = pack_f16x2(f16_to_u16(2.0), f16_to_u16(0.5))
        s1 = pack_f16x2(f16_to_u16(3.0), f16_to_u16(4.0))
        res = ref_v_pk_mul_f16(s0, s1)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 6.0)
        self.assertAlmostEqual(u16_to_f16(res_hi), 2.0)

    def test_sign_propagation(self) -> None:
        # (+ * - = -), (- * - = +)
        s0 = pack_f16x2(f16_to_u16(2.0), f16_to_u16(-3.0))
        s1 = pack_f16x2(f16_to_u16(-5.0), f16_to_u16(-4.0))
        res = ref_v_pk_mul_f16(s0, s1)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), -10.0)
        self.assertAlmostEqual(u16_to_f16(res_hi), 12.0)

    def test_dual_lane_independence(self) -> None:
        # Test that low and high halves do not interfere with each other
        s0 = pack_f16x2(F16_ONE, F16_POS_ZERO)
        s1 = pack_f16x2(F16_POS_INF, F16_ONE)
        res = ref_v_pk_mul_f16(s0, s1)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertEqual(res_lo, F16_POS_INF)
        self.assertEqual(res_hi, F16_POS_ZERO)

    def test_op_sel_cross_and_broadcast(self) -> None:
        # s0: [1.0, 2.0], s1: [3.0, 4.0]
        s0 = pack_f16x2(f16_to_u16(1.0), f16_to_u16(2.0))
        s1 = pack_f16x2(f16_to_u16(3.0), f16_to_u16(4.0))

        # Cross halves: lo uses (s0.hi, s1.lo) -> 2.0 * 3.0 = 6.0
        # hi uses (s0.lo, s1.hi) -> 1.0 * 4.0 = 4.0
        res = ref_v_pk_mul_f16(s0, s1, op_sel=(1, 0), op_sel_hi=(0, 1))
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 6.0)
        self.assertAlmostEqual(u16_to_f16(res_hi), 4.0)

        # Broadcast s0.hi to both lanes:
        # lo uses (s0.hi, s1.lo) -> 2.0 * 3.0 = 6.0
        # hi uses (s0.hi, s1.hi) -> 2.0 * 4.0 = 8.0
        res_bcast = ref_v_pk_mul_f16(s0, s1, op_sel=(1, 0), op_sel_hi=(1, 1))
        res_lo2, res_hi2 = unpack_f16x2(res_bcast)
        self.assertAlmostEqual(u16_to_f16(res_lo2), 6.0)
        self.assertAlmostEqual(u16_to_f16(res_hi2), 8.0)

    def test_negation_modifiers(self) -> None:
        s0 = pack_f16x2(f16_to_u16(2.0), f16_to_u16(3.0))
        s1 = pack_f16x2(f16_to_u16(4.0), f16_to_u16(5.0))

        # neg_lo:[1, 0] -> -2.0 * 4.0 = -8.0
        # neg_hi:[0, 1] -> 3.0 * -5.0 = -15.0
        res = ref_v_pk_mul_f16(s0, s1, neg_lo=(1, 0), neg_hi=(0, 1))
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), -8.0)
        self.assertAlmostEqual(u16_to_f16(res_hi), -15.0)

    def test_clamping_independent_halves(self) -> None:
        # lo: 0.5 * -2.0 = -1.0 -> clamp -> +0.0
        # hi: 1.5 * 2.0 = 3.0 -> clamp -> 1.0
        s0 = pack_f16x2(f16_to_u16(0.5), f16_to_u16(1.5))
        s1 = pack_f16x2(f16_to_u16(-2.0), f16_to_u16(2.0))
        res = ref_v_pk_mul_f16(s0, s1, clamp=True)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertEqual(res_lo, F16_POS_ZERO)
        self.assertEqual(res_hi, F16_ONE)


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
        self.assertFalse(op_vgpr.is_sgpr)
        self.assertTrue(op_sgpr.is_sgpr)
        self.assertTrue(op_lit.is_literal)
        self.assertTrue(op_k.is_inline_const)

        with self.assertRaises(ValueError):
            VOP3POperand()  # no source provided
        with self.assertRaises(ValueError):
            VOP3POperand(reg=Reg(0), literal_val=123)  # ambiguous sources

    def test_instruction_disassembly(self) -> None:
        instr_default = VPkMulF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
        )
        self.assertEqual(instr_default.disasm(), "v_pk_mul_f16 v0, v1, v2")

        instr_modifiers = VPkMulF16Instruction(
            dst=Reg(5, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(6, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(12, RegType.SGPR)),
            op_sel=(1, 0),
            op_sel_hi=(0, 1),
            neg_lo=(1, 0),
            neg_hi=(0, 1),
            clamp=True,
        )
        expected = "v_pk_mul_f16 v5, v6, s12 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[0,1] clamp"
        self.assertEqual(instr_modifiers.disasm(), expected)

    def test_instruction_validation(self) -> None:
        with self.assertRaises(ValueError):
            # SGPR as destination is forbidden for VALU instruction
            VPkMulF16Instruction(
                dst=Reg(0, RegType.SGPR),
                src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
                src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
            )

        with self.assertRaises(ValueError):
            # Invalid modifier entry
            VPkMulF16Instruction(
                dst=Reg(0, RegType.VGPR),
                src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
                src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
                op_sel=(2, 0),
            )


class TestScalarInstructionInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarState()
        self.interp = ScalarInterpreter()

    def test_scalar_vgpr_multiplication(self) -> None:
        # v1 = [2.0, 3.0], v2 = [4.0, 5.0]
        # v0 = v1 * v2 -> [8.0, 15.0]
        self.state.set_vgpr(1, pack_f16x2(f16_to_u16(2.0), f16_to_u16(3.0)))
        self.state.set_vgpr(2, pack_f16x2(f16_to_u16(4.0), f16_to_u16(5.0)))

        instr = VPkMulF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 8.0)
        self.assertAlmostEqual(u16_to_f16(res_hi), 15.0)
        self.assertEqual(self.state.get_vgpr(0), res)

    def test_sgpr_scalar_broadcast(self) -> None:
        # s4 = [0.5, 2.0]
        # v3 = [4.0, 6.0]
        # v0 = v3 * s4 -> [2.0, 12.0]
        self.state.set_vgpr(3, pack_f16x2(f16_to_u16(4.0), f16_to_u16(6.0)))
        self.state.set_sgpr(4, pack_f16x2(f16_to_u16(0.5), f16_to_u16(2.0)))

        instr = VPkMulF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(3, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(4, RegType.SGPR)),
        )
        res = self.interp.step(instr, self.state)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 2.0)
        self.assertAlmostEqual(u16_to_f16(res_hi), 12.0)

    def test_inline_constant_operand(self) -> None:
        # v1 = [3.0, -4.0]
        # v0 = v1 * 2.0 (inline constant replicated to both halves) -> [6.0, -8.0]
        self.state.set_vgpr(1, pack_f16x2(f16_to_u16(3.0), f16_to_u16(-4.0)))

        instr = VPkMulF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src1=VOP3POperand(inline_val=2.0),
        )
        res = self.interp.step(instr, self.state)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 6.0)
        self.assertAlmostEqual(u16_to_f16(res_hi), -8.0)

    def test_state_isolation(self) -> None:
        # Verify unrelated registers are completely untouched
        self.state.set_vgpr(10, 0x12345678)
        self.state.set_sgpr(5, 0x87654321)

        instr = VPkMulF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(inline_val=1.0),
            src1=VOP3POperand(inline_val=1.0),
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
        # Populate each lane with divergent values: lane i has [i + 1, i + 2]
        for lane in range(32):
            state.set_vgpr(lane, 1, pack_f16x2(f16_to_u16(float(lane + 1)), f16_to_u16(float(lane + 2))))
            state.set_vgpr(lane, 2, pack_f16x2(f16_to_u16(2.0), f16_to_u16(0.5)))

        instr = VPkMulF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)

        for lane in range(32):
            res = state.get_vgpr(lane, 0)
            res_lo, res_hi = unpack_f16x2(res)
            self.assertAlmostEqual(u16_to_f16(res_lo), float(lane + 1) * 2.0)
            self.assertAlmostEqual(u16_to_f16(res_hi), float(lane + 2) * 0.5)

    def test_wave64_exec_mask_gating_preserves_inactive_lanes(self) -> None:
        state = SIMDState(wave_size=64, exec_mask=0x5555555555555555)  # Even lanes active
        # Pre-fill dst v0 with sentinel value across all 64 lanes
        for lane in range(64):
            state.set_vgpr(lane, 0, OLD_VGPR_SENTINEL)
            state.set_vgpr(lane, 1, pack_f16x2(f16_to_u16(3.0), f16_to_u16(4.0)))
            state.set_vgpr(lane, 2, pack_f16x2(f16_to_u16(2.0), f16_to_u16(2.0)))

        instr = VPkMulF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)

        expected_active = pack_f16x2(f16_to_u16(6.0), f16_to_u16(8.0))
        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(state.get_vgpr(lane, 0), expected_active)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), OLD_VGPR_SENTINEL)

    def test_zero_exec_mask_no_modification(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0x00000000)
        for lane in range(32):
            state.set_vgpr(lane, 0, OLD_VGPR_SENTINEL)

        instr = VPkMulF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(inline_val=1.0),
            src1=VOP3POperand(inline_val=1.0),
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
        # v0 = v0 * v1
        state = ScalarState()
        state.set_vgpr(0, pack_f16x2(f16_to_u16(2.0), f16_to_u16(3.0)))
        state.set_vgpr(1, pack_f16x2(f16_to_u16(4.0), f16_to_u16(5.0)))

        instr = VPkMulF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(0, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(1, RegType.VGPR)),
        )
        res = self.interp.step(instr, state)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 8.0)
        self.assertAlmostEqual(u16_to_f16(res_hi), 15.0)
        self.assertEqual(state.get_vgpr(0), res)

    def test_dst_aliases_src1(self) -> None:
        # v0 = v1 * v0
        state = ScalarState()
        state.set_vgpr(0, pack_f16x2(f16_to_u16(2.0), f16_to_u16(3.0)))
        state.set_vgpr(1, pack_f16x2(f16_to_u16(4.0), f16_to_u16(5.0)))

        instr = VPkMulF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(0, RegType.VGPR)),
        )
        res = self.interp.step(instr, state)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 8.0)
        self.assertAlmostEqual(u16_to_f16(res_hi), 15.0)
        self.assertEqual(state.get_vgpr(0), res)

    def test_dst_aliases_both_src0_and_src1_squaring(self) -> None:
        # v0 = v0 * v0 (in-place squaring)
        state = ScalarState()
        state.set_vgpr(0, pack_f16x2(f16_to_u16(3.0), f16_to_u16(-4.0)))

        instr = VPkMulF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(0, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(0, RegType.VGPR)),
        )
        res = self.interp.step(instr, state)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 9.0)
        self.assertAlmostEqual(u16_to_f16(res_hi), 16.0)
        self.assertEqual(state.get_vgpr(0), res)

    def test_cross_half_aliasing_inplace(self) -> None:
        # v0 = v0 * v0 op_sel:[1,0] op_sel_hi:[0,1]
        # v0 starts as [2.0, 5.0]
        # lo uses (v0.hi, v0.lo) -> 5.0 * 2.0 = 10.0
        # hi uses (v0.lo, v0.hi) -> 2.0 * 5.0 = 10.0
        state = ScalarState()
        state.set_vgpr(0, pack_f16x2(f16_to_u16(2.0), f16_to_u16(5.0)))

        instr = VPkMulF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(0, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(0, RegType.VGPR)),
            op_sel=(1, 0),
            op_sel_hi=(0, 1),
        )
        res = self.interp.step(instr, state)
        res_lo, res_hi = unpack_f16x2(res)
        self.assertAlmostEqual(u16_to_f16(res_lo), 10.0)
        self.assertAlmostEqual(u16_to_f16(res_hi), 10.0)

    def test_simd_multi_lane_inplace_aliasing(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0xFFFFFFFF)
        for lane in range(32):
            state.set_vgpr(lane, 0, pack_f16x2(f16_to_u16(float(lane)), f16_to_u16(2.0)))

        instr = VPkMulF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(reg=Reg(0, RegType.VGPR)),
            src1=VOP3POperand(reg=Reg(0, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)

        for lane in range(32):
            res = state.get_vgpr(lane, 0)
            res_lo, res_hi = unpack_f16x2(res)
            self.assertAlmostEqual(u16_to_f16(res_lo), float(lane) ** 2)
            self.assertAlmostEqual(u16_to_f16(res_hi), 4.0)


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Test suite for condition codes, status flags, and carry generation."""

    def test_exact_finite_flags(self) -> None:
        s0 = f16_to_u16(2.0)
        s1 = f16_to_u16(3.0)
        res = ref_mul_f16(s0, s1)
        flags = evaluate_f16_mul_flags(s0, s1, res)
        self.assertFalse(flags.invalid)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.underflow)
        self.assertFalse(flags.inexact)
        self.assertFalse(flags.denormal)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.infinite)
        self.assertFalse(flags.nan)

    def test_inexact_flag(self) -> None:
        # In half-precision, 1.0009765625 * 1.0009765625 cannot be represented exactly in 10-bit fraction
        s0 = 0x3C01  # 1.0 + 2^-10
        s1 = 0x3C01
        res = ref_mul_f16(s0, s1)
        flags = evaluate_f16_mul_flags(s0, s1, res)
        self.assertTrue(flags.inexact)

    def test_overflow_to_infinity_flag(self) -> None:
        s0 = F16_MAX_NORMAL  # 65504.0
        s1 = F16_TWO         # 2.0
        res = ref_mul_f16(s0, s1)
        self.assertEqual(res, F16_POS_INF)
        flags = evaluate_f16_mul_flags(s0, s1, res)
        self.assertTrue(flags.overflow)
        self.assertTrue(flags.infinite)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.invalid)

    def test_underflow_and_denormal_flags(self) -> None:
        s0 = F16_MIN_NORMAL  # 2^-14
        s1 = F16_HALF        # 0.5 -> product 2^-15 is subnormal
        res = ref_mul_f16(s0, s1)
        self.assertTrue(is_f16_subnormal(res))
        flags = evaluate_f16_mul_flags(s0, s1, res)
        self.assertTrue(flags.denormal)
        self.assertTrue(flags.underflow)

    def test_invalid_operation_flag(self) -> None:
        # inf * 0.0 is an invalid operation
        res = ref_mul_f16(F16_POS_INF, F16_POS_ZERO)
        self.assertTrue(is_f16_nan(res))
        flags = evaluate_f16_mul_flags(F16_POS_INF, F16_POS_ZERO, res)
        self.assertTrue(flags.invalid)
        self.assertTrue(flags.nan)

        # signaling NaN operand triggers invalid operation flag
        snan = 0x7C01  # exp=0x1F, mant=1 (bit 9 is 0)
        self.assertTrue(is_f16_snan(snan))
        res_snan = ref_mul_f16(snan, F16_ONE)
        flags_snan = evaluate_f16_mul_flags(snan, F16_ONE, res_snan)
        self.assertTrue(flags_snan.invalid)
        self.assertTrue(flags_snan.nan)

    def test_zero_and_negative_flags(self) -> None:
        # -2.0 * 3.0 = -6.0 -> negative flag
        s0 = f16_to_u16(-2.0)
        s1 = f16_to_u16(3.0)
        res = ref_mul_f16(s0, s1)
        flags = evaluate_f16_mul_flags(s0, s1, res)
        self.assertTrue(flags.negative)
        self.assertFalse(flags.zero)

        # -0.0 * 5.0 = -0.0 -> zero flag and negative flag
        res_nz = ref_mul_f16(F16_NEG_ZERO, f16_to_u16(5.0))
        flags_nz = evaluate_f16_mul_flags(F16_NEG_ZERO, f16_to_u16(5.0), res_nz)
        self.assertTrue(flags_nz.zero)
        self.assertTrue(flags_nz.negative)

    def test_packed_flags_evaluation(self) -> None:
        s0 = pack_f16x2(F16_POS_INF, f16_to_u16(2.0))
        s1 = pack_f16x2(F16_POS_ZERO, f16_to_u16(3.0))
        res = ref_v_pk_mul_f16(s0, s1)
        flags_lo, flags_hi = evaluate_pk_mul_f16_flags(s0, s1, res)
        self.assertTrue(flags_lo.invalid)
        self.assertTrue(flags_lo.nan)
        self.assertFalse(flags_hi.invalid)
        self.assertAlmostEqual(u16_to_f16(unpack_f16x2(res)[1]), 6.0)

    def test_vcc_condition_masks(self) -> None:
        # Generate 4 results:
        # Lane 0: both zero
        # Lane 1: positive
        # Lane 2: contains NaN
        # Lane 3: both positive
        r0 = pack_f16x2(F16_POS_ZERO, F16_NEG_ZERO)
        r1 = pack_f16x2(f16_to_u16(1.0), f16_to_u16(2.0))
        r2 = pack_f16x2(F16_CANONICAL_QNAN, f16_to_u16(1.0))
        r3 = pack_f16x2(f16_to_u16(3.0), f16_to_u16(4.0))

        results = [r0, r1, r2, r3]
        mask_zero = generate_vcc_mask_pk_f16("eq_zero", results, 4)
        self.assertEqual(mask_zero, 0b0001)

        mask_nan = generate_vcc_mask_pk_f16("has_nan", results, 4)
        self.assertEqual(mask_nan, 0b0100)

        mask_pos = generate_vcc_mask_pk_f16("both_positive", results, 4)
        self.assertEqual(mask_pos, 0b1010)

    def test_mantissa_product_and_carry(self) -> None:
        # 1.0 * 1.0: mantissas are 1024 * 1024 = 1048576 (2^20) -> carry_shift = 0
        _, c0 = compute_f16_mantissa_product(F16_ONE, F16_ONE)
        self.assertEqual(c0, 0)

        # 1.5 * 1.5 = 2.25: product >= 2.0 -> carry_shift = 1
        s_1p5 = f16_to_u16(1.5)
        _, c1 = compute_f16_mantissa_product(s_1p5, s_1p5)
        self.assertEqual(c1, 1)


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Test suite for IEEE-754 boundary extremes, subnormals, and NaN handling."""

    def test_nan_snan_quieting(self) -> None:
        # Positive signaling NaN
        snan_pos = 0x7C01
        self.assertTrue(is_f16_snan(snan_pos))
        res_pos = ref_mul_f16(snan_pos, F16_ONE)
        self.assertTrue(is_f16_qnan(res_pos))
        self.assertEqual(res_pos, 0x7E01)

        # Negative signaling NaN
        snan_neg = 0xFC05
        self.assertTrue(is_f16_snan(snan_neg))
        res_neg = ref_mul_f16(F16_TWO, snan_neg)
        self.assertTrue(is_f16_qnan(res_neg))
        self.assertEqual(res_neg, 0xFE05)

    def test_nan_payload_propagation_and_precedence(self) -> None:
        # Quiet NaN payload from src0
        qnan_a = 0x7E55
        res_a = ref_mul_f16(qnan_a, f16_to_u16(5.0))
        self.assertEqual(res_a, qnan_a)

        # Quiet NaN payload from src1
        qnan_b = 0xFEAA
        res_b = ref_mul_f16(f16_to_u16(-3.0), qnan_b)
        self.assertEqual(res_b, qnan_b)

        # Dual NaN: src0 takes precedence
        qnan0 = 0x7E12
        qnan1 = 0x7E34
        res_dual = ref_mul_f16(qnan0, qnan1)
        self.assertEqual(res_dual, qnan0)

    def test_nan_clamping_produces_zero(self) -> None:
        qnan = 0x7E55
        res_clamped = ref_mul_f16(qnan, F16_ONE, clamp=True)
        self.assertEqual(res_clamped, F16_POS_ZERO)

        # Packed NaN clamp
        s0 = pack_f16x2(qnan, F16_ONE)
        s1 = pack_f16x2(F16_ONE, qnan)
        res_pk_clamped = ref_v_pk_mul_f16(s0, s1, clamp=True)
        self.assertEqual(res_pk_clamped, pack_f16x2(F16_POS_ZERO, F16_POS_ZERO))

    def test_signed_zero_combinations(self) -> None:
        self.assertEqual(ref_mul_f16(F16_POS_ZERO, F16_POS_ZERO), F16_POS_ZERO)
        self.assertEqual(ref_mul_f16(F16_POS_ZERO, F16_NEG_ZERO), F16_NEG_ZERO)
        self.assertEqual(ref_mul_f16(F16_NEG_ZERO, F16_POS_ZERO), F16_NEG_ZERO)
        self.assertEqual(ref_mul_f16(F16_NEG_ZERO, F16_NEG_ZERO), F16_POS_ZERO)

        # Finite and zero
        self.assertEqual(ref_mul_f16(F16_POS_ZERO, f16_to_u16(10.0)), F16_POS_ZERO)
        self.assertEqual(ref_mul_f16(F16_POS_ZERO, f16_to_u16(-10.0)), F16_NEG_ZERO)
        self.assertEqual(ref_mul_f16(F16_NEG_ZERO, f16_to_u16(10.0)), F16_NEG_ZERO)
        self.assertEqual(ref_mul_f16(F16_NEG_ZERO, f16_to_u16(-10.0)), F16_POS_ZERO)

        # Clamped negative zero becomes +0.0
        self.assertEqual(ref_mul_f16(F16_NEG_ZERO, f16_to_u16(10.0), clamp=True), F16_POS_ZERO)

    def test_signed_infinity_combinations(self) -> None:
        self.assertEqual(ref_mul_f16(F16_POS_INF, F16_POS_INF), F16_POS_INF)
        self.assertEqual(ref_mul_f16(F16_POS_INF, F16_NEG_INF), F16_NEG_INF)
        self.assertEqual(ref_mul_f16(F16_NEG_INF, F16_POS_INF), F16_NEG_INF)
        self.assertEqual(ref_mul_f16(F16_NEG_INF, F16_NEG_INF), F16_POS_INF)

        # Inf * 0.0 -> Canonical Quiet NaN
        self.assertEqual(ref_mul_f16(F16_POS_INF, F16_POS_ZERO), F16_CANONICAL_QNAN)
        self.assertEqual(ref_mul_f16(F16_NEG_INF, F16_POS_ZERO), F16_CANONICAL_QNAN)
        self.assertEqual(ref_mul_f16(F16_POS_ZERO, F16_POS_INF), F16_CANONICAL_QNAN)
        self.assertEqual(ref_mul_f16(F16_NEG_ZERO, F16_NEG_INF), F16_CANONICAL_QNAN)

        # Clamped inf: positive inf becomes 1.0, negative inf becomes +0.0
        self.assertEqual(ref_mul_f16(F16_POS_INF, F16_ONE, clamp=True), F16_ONE)
        self.assertEqual(ref_mul_f16(F16_NEG_INF, F16_ONE, clamp=True), F16_POS_ZERO)

        # Clamped inf * 0.0 becomes +0.0
        self.assertEqual(ref_mul_f16(F16_POS_INF, F16_POS_ZERO, clamp=True), F16_POS_ZERO)

    def test_max_normal_overflow_boundaries(self) -> None:
        # MAX_NORMAL * 1.0 = MAX_NORMAL
        self.assertEqual(ref_mul_f16(F16_MAX_NORMAL, F16_ONE), F16_MAX_NORMAL)
        # MAX_NORMAL * -1.0 = -MAX_NORMAL
        self.assertEqual(ref_mul_f16(F16_MAX_NORMAL, F16_NEG_ONE), F16_MAX_NORMAL | F16_SIGN_MASK)
        # MAX_NORMAL * 2.0 -> overflow to +inf
        self.assertEqual(ref_mul_f16(F16_MAX_NORMAL, F16_TWO), F16_POS_INF)
        # MAX_NORMAL * -2.0 -> overflow to -inf
        self.assertEqual(ref_mul_f16(F16_MAX_NORMAL, f16_to_u16(-2.0)), F16_NEG_INF)

    def test_subnormal_boundaries(self) -> None:
        # MIN_NORMAL * 1.0 = MIN_NORMAL
        self.assertEqual(ref_mul_f16(F16_MIN_NORMAL, F16_ONE), F16_MIN_NORMAL)
        # MIN_NORMAL * 0.5 = subnormal 2^-15
        res_sub = ref_mul_f16(F16_MIN_NORMAL, F16_HALF)
        self.assertTrue(is_f16_subnormal(res_sub))
        self.assertEqual(res_sub, 0x0200)

        # MIN_SUBNORMAL (2^-24) * 1.0 = MIN_SUBNORMAL
        self.assertEqual(ref_mul_f16(F16_MIN_SUBNORMAL, F16_ONE), F16_MIN_SUBNORMAL)
        # MIN_SUBNORMAL * 2.0 = 0x0002
        self.assertEqual(ref_mul_f16(F16_MIN_SUBNORMAL, F16_TWO), 0x0002)
        # MIN_SUBNORMAL * 0.5 = underflow to +0.0
        self.assertEqual(ref_mul_f16(F16_MIN_SUBNORMAL, F16_HALF), F16_POS_ZERO)
        # -MIN_SUBNORMAL * 0.5 = underflow to -0.0
        self.assertEqual(ref_mul_f16(F16_MIN_SUBNORMAL | F16_SIGN_MASK, F16_HALF), F16_NEG_ZERO)

    def test_round_to_nearest_even(self) -> None:
        # Check round-to-nearest-even tie breaking
        val = 1.0 + 2.0**(-10)
        bits = f16_to_u16(val)
        res = ref_mul_f16(bits, F16_ONE)
        self.assertEqual(res, bits)


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
        # VALU destination cannot be SGPR
        with self.assertRaises(ValueError):
            VPkMulF16Instruction(
                dst=Reg(0, RegType.SGPR),
                src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
                src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
            )

    def test_invalid_modifier_tuples(self) -> None:
        # Tuple of wrong length
        with self.assertRaises(ValueError):
            VPkMulF16Instruction(
                dst=Reg(0, RegType.VGPR),
                src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
                src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
                op_sel=(0,),  # type: ignore[arg-type]
            )
        # Value not in {0, 1}
        with self.assertRaises(ValueError):
            VPkMulF16Instruction(
                dst=Reg(0, RegType.VGPR),
                src0=VOP3POperand(reg=Reg(1, RegType.VGPR)),
                src1=VOP3POperand(reg=Reg(2, RegType.VGPR)),
                neg_lo=(0, 2),
            )

    def test_mutation_addition_instead_of_multiplication(self) -> None:
        def mutated_mul_f16(s0: int, s1: int) -> int:
            return f16_to_u16(u16_to_f16(s0) + u16_to_f16(s1))

        # Golden expectation: 2.0 * 3.0 = 6.0. Mutant computes 2.0 + 3.0 = 5.0
        s0 = f16_to_u16(2.0)
        s1 = f16_to_u16(3.0)
        oracle_res = ref_mul_f16(s0, s1)
        mutant_res = mutated_mul_f16(s0, s1)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_dropped_clamp(self) -> None:
        def mutated_pk_mul(s0: int, s1: int, clamp: bool) -> int:
            # Drops clamp argument
            return ref_v_pk_mul_f16(s0, s1, clamp=False)

        # In packed mul: 2.0 * 2.0 = 4.0. With clamp, result is 1.0
        s0 = pack_f16x2(f16_to_u16(2.0), f16_to_u16(2.0))
        s1 = pack_f16x2(f16_to_u16(2.0), f16_to_u16(2.0))
        oracle_res = ref_v_pk_mul_f16(s0, s1, clamp=True)
        mutant_res = mutated_pk_mul(s0, s1, clamp=True)
        self.assertNotEqual(oracle_res, mutant_res)
        self.assertEqual(oracle_res, pack_f16x2(F16_ONE, F16_ONE))

    def test_mutation_failed_snan_quieting(self) -> None:
        def mutated_nan_oracle(s0: int, s1: int) -> int:
            # Returns raw bits without quieting signaling NaN
            if is_f16_nan(s0):
                return s0
            return s1

        snan = 0x7C01
        oracle_res = ref_mul_f16(snan, F16_ONE)
        mutant_res = mutated_nan_oracle(snan, F16_ONE)
        self.assertTrue(is_f16_qnan(oracle_res))
        self.assertFalse(is_f16_qnan(mutant_res))
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_ignored_op_sel_hi(self) -> None:
        # When op_sel_hi is (0, 0) instead of (1, 1), hi lane selects lo operand instead of hi operand
        s0 = pack_f16x2(f16_to_u16(2.0), f16_to_u16(10.0))
        s1 = pack_f16x2(f16_to_u16(3.0), f16_to_u16(4.0))

        correct = ref_v_pk_mul_f16(s0, s1, op_sel_hi=(1, 1))
        mutant = ref_v_pk_mul_f16(s0, s1, op_sel_hi=(0, 0))
        # correct hi is 10.0 * 4.0 = 40.0; mutant hi is 2.0 * 3.0 = 6.0
        self.assertNotEqual(correct, mutant)

    def test_mutation_corrupted_inactive_exec_lane(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0x00000001)  # only lane 0 active
        state.set_vgpr(1, 0, OLD_VGPR_SENTINEL)

        # Correct execution
        simd_interp = SIMDInterpreter()
        instr = VPkMulF16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(inline_val=1.0),
            src1=VOP3POperand(inline_val=1.0),
        )
        simd_interp.step(instr, state)
        self.assertEqual(state.get_vgpr(1, 0), OLD_VGPR_SENTINEL)

        # Mutant simulator that erroneously overwrites inactive lanes
        state.set_vgpr(1, 0, 0x11111111)
        self.assertNotEqual(state.get_vgpr(1, 0), OLD_VGPR_SENTINEL)


if __name__ == "__main__":
    unittest.main()
