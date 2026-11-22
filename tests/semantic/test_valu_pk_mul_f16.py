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
import math
import struct
from typing import Optional, Tuple, Union
import unittest

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF

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
# Unit Tests: Phase 1 & Phase 2
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


if __name__ == "__main__":
    unittest.main()
