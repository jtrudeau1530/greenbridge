"""Semantic reference oracle and architectural model for RDNA2 v_dot2_f16 instructions.

Opcode targets:
  - v_dot2_f16_f16:    D.f16 = S0.f16[0] * S1.f16[0] + S0.f16[1] * S1.f16[1]
  - v_dot2acc_f16_f16: D.f16 = S0.f16[0] * S1.f16[0] + S0.f16[1] * S1.f16[1] + S2.f16

AMD RDNA2 VALU Instruction Set Architecture (GFX10):
  The v_dot2_f16_f16 instruction computes a 2-element half-precision floating-point dot
  product from packed 32-bit registers S0 and S1. Each 32-bit source holds two IEEE 754
  binary16 values: low halfword [15:0] and high halfword [31:16].
  The accumulating variant v_dot2acc_f16_f16 adds an accumulator S2 into the scalar result.
  Result is written to destination VGPR D[15:0] (upper bits zeroed).

Modifiers:
  - clamp: Saturates final result to [0.0, 1.0]. Negative values, -0.0, and NaNs clamp to +0.0;
    values >= 1.0 and +Infinity clamp to 1.0.
  - op_sel / op_sel_hi: Controls selection of low/high halfwords from S0 and S1 for the two terms.
  - neg / abs modifiers: Applied to source halfwords prior to arithmetic.

Standard library Python only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

# ==============================================================================
# Architectural Bit Masks and IEEE 754 binary16 Encodings
# ==============================================================================

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF

SIGN_MASK: int = 0x8000
EXP_MASK: int = 0x7C00
FRAC_MASK: int = 0x03FF
QNAN_BIT: int = 0x0200

# Canonical IEEE 754 binary16 values
POS_ZERO: int = 0x0000
NEG_ZERO: int = 0x8000
POS_INF: int = 0x7C00
NEG_INF: int = 0xFC00
CANONICAL_QNAN: int = 0x7E00
CANONICAL_SNAN: int = 0x7D00

# Boundary limits
NORM_MAX: int = 0x7BFF       # +65504.0
NEG_NORM_MAX: int = 0xFBFF   # -65504.0
NORM_MIN: int = 0x0400       # +2^-14 (~6.10352e-5)
NEG_NORM_MIN: int = 0x8400   # -2^-14
SUB_MIN: int = 0x0001        # +2^-24 (~5.96046e-8)
NEG_SUB_MIN: int = 0x8001    # -2^-24
SUB_MAX: int = 0x03FF        # +1023 * 2^-24
NEG_SUB_MAX: int = 0x83FF    # -1023 * 2^-24

# Useful unit constants
ONE_F16: int = 0x3C00        # +1.0
NEG_ONE_F16: int = 0xBC00    # -1.0
HALF_F16: int = 0x3800       # +0.5
NEG_HALF_F16: int = 0xB800   # -0.5
TWO_F16: int = 0x4000        # +2.0
NEG_TWO_F16: int = 0xC000    # -2.0
THREE_F16: int = 0x4200      # +3.0
FOUR_F16: int = 0x4400       # +4.0
FIVE_F16: int = 0x4500       # +5.0


# ==============================================================================
# Helper Conversions, Bit Manipulation, and IEEE 754 Predicates
# ==============================================================================

def f16_to_bits(val: float) -> int:
    """Convert an IEEE 754 Python float to 16-bit binary16 integer bits."""
    if math.isnan(val):
        sign = 0x8000 if math.copysign(1.0, val) < 0 else 0x0000
        return sign | CANONICAL_QNAN
    if math.isinf(val):
        return POS_INF if val > 0 else NEG_INF
    if val >= 65520.0:
        return POS_INF
    if val <= -65520.0:
        return NEG_INF
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f16(bits: int) -> float:
    """Convert a 16-bit binary16 integer into a Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def pack_f16x2(lo: int, hi: int) -> int:
    """Pack two 16-bit halfwords into a 32-bit dword (lo, hi)."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def unpack_f16x2(dword: int) -> Tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit halfwords (lo, hi)."""
    dword &= MASK32
    return dword & MASK16, (dword >> 16) & MASK16


def split_f16_bits(bits: int) -> Tuple[int, int, int]:
    """Split 16-bit binary16 into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK) >> 15
    exp = (bits & EXP_MASK) >> 10
    frac = bits & FRAC_MASK
    return sign, exp, frac


def is_nan(bits: int) -> bool:
    """Return True if 16-bit word represents an IEEE 754 NaN."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) != 0)


def is_qnan(bits: int) -> bool:
    """Return True if 16-bit word represents a Quiet NaN."""
    return is_nan(bits) and bool(bits & QNAN_BIT)


def is_snan(bits: int) -> bool:
    """Return True if 16-bit word represents a Signaling NaN."""
    return is_nan(bits) and not bool(bits & QNAN_BIT)


def is_inf(bits: int) -> bool:
    """Return True if 16-bit word represents positive or negative infinity."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) == 0)


def is_subnormal(bits: int) -> bool:
    """Return True if 16-bit word represents a subnormal (denormal) float."""
    return ((bits & EXP_MASK) == 0) and ((bits & FRAC_MASK) != 0)


def is_zero(bits: int) -> bool:
    """Return True if 16-bit word represents positive or negative zero."""
    return (bits & ~SIGN_MASK) == 0


def apply_f16_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & MASK16
    if abs_mod:
        res &= ~SIGN_MASK
    if neg_mod:
        res ^= SIGN_MASK
    return res


def clamp_f16_bits(bits: int) -> int:
    """Apply RDNA2 VALU clamp modifier saturating to [0.0, 1.0].
    
    In RDNA2 hardware:
      - Any NaN clamps to +0.0
      - Values < 0.0 or -0.0 clamp to +0.0
      - Values > 1.0 or +Infinity clamp to 1.0 (ONE_F16)
    """
    if is_nan(bits):
        return POS_ZERO
    if is_zero(bits):
        return POS_ZERO
    if bool(bits & SIGN_MASK):
        return POS_ZERO
    if bits == POS_INF:
        return ONE_F16
    val = bits_to_f16(bits)
    if val > 1.0:
        return ONE_F16
    return bits


def mul_f16(a: int, b: int) -> int:
    """Multiply two 16-bit IEEE 754 half-precision numbers."""
    # 1. NaN propagation
    if is_nan(a) or is_nan(b):
        nan_bits = a if is_nan(a) else b
        return nan_bits | QNAN_BIT

    # 2. 0 * Inf is invalid -> Canonical Quiet NaN
    if (is_zero(a) and is_inf(b)) or (is_inf(a) and is_zero(b)):
        return CANONICAL_QNAN

    # 3. Infinity multiplication
    sign = (a ^ b) & SIGN_MASK
    if is_inf(a) or is_inf(b):
        return sign | POS_INF

    # 4. Zero multiplication
    if is_zero(a) or is_zero(b):
        return sign | POS_ZERO

    # 5. Finite arithmetic
    f_a = bits_to_f16(a)
    f_b = bits_to_f16(b)
    f_prod = f_a * f_b
    res = f16_to_bits(f_prod)
    if f_prod == 0.0:
        res = sign | POS_ZERO
    return res


def add_f16(a: int, b: int) -> int:
    """Add two 16-bit IEEE 754 half-precision numbers with round-to-nearest-even."""
    # 1. NaN propagation
    if is_nan(a) or is_nan(b):
        nan_bits = a if is_nan(a) else b
        return nan_bits | QNAN_BIT

    # 2. Infinity addition
    a_inf = is_inf(a)
    b_inf = is_inf(b)
    if a_inf and b_inf:
        if (a & SIGN_MASK) != (b & SIGN_MASK):
            # +inf + -inf -> Invalid operation -> Canonical Quiet NaN
            return CANONICAL_QNAN
        return a
    if a_inf:
        return a
    if b_inf:
        return b

    # 3. Signed zero addition
    if is_zero(a) and is_zero(b):
        if (a == NEG_ZERO) and (b == NEG_ZERO):
            return NEG_ZERO
        return POS_ZERO

    # 4. Finite addition
    f_a = bits_to_f16(a)
    f_b = bits_to_f16(b)
    f_res = f_a + f_b
    res = f16_to_bits(f_res)
    if f_res == 0.0 and not ((a == NEG_ZERO) and (b == NEG_ZERO)):
        res = POS_ZERO
    return res


# ==============================================================================
# Reference Mathematical Oracles
# ==============================================================================

def oracle_v_dot2_f16(
    src0_bits: int,
    src1_bits: int,
    abs0: Tuple[bool, bool] = (False, False),
    neg0: Tuple[bool, bool] = (False, False),
    abs1: Tuple[bool, bool] = (False, False),
    neg1: Tuple[bool, bool] = (False, False),
    op_sel: Tuple[int, int] = (0, 0),
    op_sel_hi: Tuple[int, int] = (1, 1),
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_dot2_f16_f16.

    Computes: D = S0[op_sel[0]] * S1[op_sel[1]] + S0[op_sel_hi[0]] * S1[op_sel_hi[1]]
    Returns 16-bit binary16 integer bits.
    """
    s0_lo, s0_hi = unpack_f16x2(src0_bits)
    s1_lo, s1_hi = unpack_f16x2(src1_bits)
    s0_halves = (s0_lo, s0_hi)
    s1_halves = (s1_lo, s1_hi)

    h0_0 = apply_f16_modifiers(s0_halves[op_sel[0] & 1], abs0[0], neg0[0])
    h1_0 = apply_f16_modifiers(s1_halves[op_sel[1] & 1], abs1[0], neg1[0])
    h0_1 = apply_f16_modifiers(s0_halves[op_sel_hi[0] & 1], abs0[1], neg0[1])
    h1_1 = apply_f16_modifiers(s1_halves[op_sel_hi[1] & 1], abs1[1], neg1[1])

    prod0 = mul_f16(h0_0, h1_0)
    prod1 = mul_f16(h0_1, h1_1)
    res = add_f16(prod0, prod1)

    if clamp:
        res = clamp_f16_bits(res)
    return res


def oracle_v_dot2acc_f16(
    src0_bits: int,
    src1_bits: int,
    src2_acc_bits: int,
    abs0: Tuple[bool, bool] = (False, False),
    neg0: Tuple[bool, bool] = (False, False),
    abs1: Tuple[bool, bool] = (False, False),
    neg1: Tuple[bool, bool] = (False, False),
    abs2: bool = False,
    neg2: bool = False,
    op_sel: Tuple[int, int] = (0, 0),
    op_sel_hi: Tuple[int, int] = (1, 1),
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_dot2acc_f16_f16.

    Computes: D = S0[op_sel[0]] * S1[op_sel[1]] + S0[op_sel_hi[0]] * S1[op_sel_hi[1]] + S2
    Returns 16-bit binary16 integer bits.
    """
    dot_val = oracle_v_dot2_f16(
        src0_bits=src0_bits,
        src1_bits=src1_bits,
        abs0=abs0,
        neg0=neg0,
        abs1=abs1,
        neg1=neg1,
        op_sel=op_sel,
        op_sel_hi=op_sel_hi,
        clamp=False,
    )
    acc = apply_f16_modifiers(src2_acc_bits & MASK16, abs2, neg2)
    res = add_f16(dot_val, acc)

    if clamp:
        res = clamp_f16_bits(res)
    return res


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Reg:
    """Architectural 32-bit register (VGPR or SGPR)."""
    index: int
    reg_type: RegType

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index {self.index} out of valid range [0, 255]")

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class Literal32:
    """32-bit inline constant or literal packed halfword pair."""
    bits: int

    def __post_init__(self) -> None:
        if not (0 <= self.bits <= MASK32):
            raise ValueError(f"Literal32 bits 0x{self.bits:X} out of range [0, 0xFFFFFFFF]")

    @classmethod
    def from_f16x2(cls, lo: float, hi: float) -> Literal32:
        return cls(pack_f16x2(f16_to_bits(lo), f16_to_bits(hi)))

    @property
    def name(self) -> str:
        lo, hi = unpack_f16x2(self.bits)
        return f"pack_f16({bits_to_f16(lo):g}, {bits_to_f16(hi):g})"


@dataclass(frozen=True)
class VOP3Operand:
    """VOP3/VOP3P operand with sub-word modifiers (abs, neg per 16-bit half)."""
    source: Union[Reg, Literal32]
    abs_lo: bool = False
    abs_hi: bool = False
    neg_lo: bool = False
    neg_hi: bool = False

    def disassemble(self) -> str:
        s = self.source.name
        mods = []
        if self.abs_lo or self.abs_hi:
            mods.append("abs")
        if self.neg_lo or self.neg_hi:
            mods.append("neg")
        if mods:
            return f"{'/'.join(mods)}({s})"
        return s


@dataclass(frozen=True)
class VDot2F16Instruction:
    """Symbolic AST node representing AMD RDNA2 v_dot2_f16_f16 instruction."""
    dst: Reg
    src0: VOP3Operand
    src1: VOP3Operand
    clamp: bool = False
    op_sel: Tuple[int, int] = (0, 0)
    op_sel_hi: Tuple[int, int] = (1, 1)

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        s = f"v_dot2_f16_f16 {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}"
        if self.op_sel != (0, 0) or self.op_sel_hi != (1, 1):
            s += f" op_sel:[{self.op_sel[0]},{self.op_sel[1]},{self.op_sel_hi[0]},{self.op_sel_hi[1]}]"
        if self.clamp:
            s += " clamp"
        return s


@dataclass(frozen=True)
class VDot2AccF16Instruction:
    """Symbolic AST node representing AMD RDNA2 v_dot2acc_f16_f16 instruction."""
    dst: Reg
    src0: VOP3Operand
    src1: VOP3Operand
    src2: VOP3Operand
    clamp: bool = False
    op_sel: Tuple[int, int] = (0, 0)
    op_sel_hi: Tuple[int, int] = (1, 1)

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        s = f"v_dot2acc_f16_f16 {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"
        if self.op_sel != (0, 0) or self.op_sel_hi != (1, 1):
            s += f" op_sel:[{self.op_sel[0]},{self.op_sel[1]},{self.op_sel_hi[0]},{self.op_sel_hi[1]}]"
        if self.clamp:
            s += " clamp"
        return s


# ==============================================================================
# Unit Tests: Architectural Model and Reference Oracle
# ==============================================================================

class TestDot2F16ModelAndOracle(unittest.TestCase):
    """Unit tests for binary16 architectural model and reference mathematical oracle."""

    def test_f16_bit_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 65504.0, -65504.0]
        for v in values:
            with self.subTest(v=v):
                bits = f16_to_bits(v)
                recovered = bits_to_f16(bits)
                self.assertEqual(v, recovered)
                self.assertEqual(math.copysign(1.0, v), math.copysign(1.0, recovered))

    def test_pack_unpack_f16x2(self) -> None:
        dword = pack_f16x2(ONE_F16, TWO_F16)
        lo, hi = unpack_f16x2(dword)
        self.assertEqual(lo, ONE_F16)
        self.assertEqual(hi, TWO_F16)
        self.assertEqual(dword, 0x4000_3C00)

    def test_classification_predicates(self) -> None:
        self.assertTrue(is_zero(POS_ZERO))
        self.assertTrue(is_zero(NEG_ZERO))
        self.assertFalse(is_zero(ONE_F16))

        self.assertTrue(is_inf(POS_INF))
        self.assertTrue(is_inf(NEG_INF))
        self.assertFalse(is_inf(POS_ZERO))

        self.assertTrue(is_nan(CANONICAL_QNAN))
        self.assertTrue(is_qnan(CANONICAL_QNAN))
        self.assertFalse(is_snan(CANONICAL_QNAN))

        self.assertTrue(is_nan(CANONICAL_SNAN))
        self.assertTrue(is_snan(CANONICAL_SNAN))
        self.assertFalse(is_qnan(CANONICAL_SNAN))

        self.assertTrue(is_subnormal(SUB_MIN))
        self.assertTrue(is_subnormal(SUB_MAX))
        self.assertFalse(is_subnormal(NORM_MIN))
        self.assertFalse(is_subnormal(POS_ZERO))

    def test_exact_finite_dot2(self) -> None:
        # [1.0, 2.0] . [3.0, 4.0] = 1*3 + 2*4 = 11.0
        s0 = pack_f16x2(ONE_F16, TWO_F16)
        s1 = pack_f16x2(THREE_F16, FOUR_F16)
        res = oracle_v_dot2_f16(s0, s1)
        self.assertEqual(bits_to_f16(res), 11.0)

        # [2.0, -1.0] . [1.5, 3.0] = 3.0 - 3.0 = 0.0
        s0_b = pack_f16x2(TWO_F16, NEG_ONE_F16)
        s1_b = pack_f16x2(f16_to_bits(1.5), THREE_F16)
        res_b = oracle_v_dot2_f16(s0_b, s1_b)
        self.assertEqual(res_b, POS_ZERO)

    def test_exact_finite_dot2acc(self) -> None:
        # [1.0, 2.0] . [3.0, 4.0] + 5.0 = 11.0 + 5.0 = 16.0
        s0 = pack_f16x2(ONE_F16, TWO_F16)
        s1 = pack_f16x2(THREE_F16, FOUR_F16)
        res = oracle_v_dot2acc_f16(s0, s1, FIVE_F16)
        self.assertEqual(bits_to_f16(res), 16.0)

    def test_signed_zero_rules(self) -> None:
        # (+0) * (+1) + (+0) * (+1) = +0
        s0 = pack_f16x2(POS_ZERO, POS_ZERO)
        s1 = pack_f16x2(ONE_F16, ONE_F16)
        self.assertEqual(oracle_v_dot2_f16(s0, s1), POS_ZERO)

        # (-0) * (+1) + (-0) * (+1) = -0
        s0_neg = pack_f16x2(NEG_ZERO, NEG_ZERO)
        self.assertEqual(oracle_v_dot2_f16(s0_neg, s1), NEG_ZERO)

    def test_input_modifiers(self) -> None:
        s0 = pack_f16x2(NEG_ONE_F16, NEG_TWO_F16)
        s1 = pack_f16x2(THREE_F16, FOUR_F16)
        # abs on s0: |-1|*3 + |-2|*4 = 3 + 8 = 11.0
        res = oracle_v_dot2_f16(s0, s1, abs0=(True, True))
        self.assertEqual(bits_to_f16(res), 11.0)

        # neg on s1 term0: -(-1*3) + (-2*4) = 3 - 8 = -5.0
        res2 = oracle_v_dot2_f16(s0, s1, neg0=(True, False))
        self.assertEqual(bits_to_f16(res2), -5.0)

    def test_op_sel_permutation(self) -> None:
        # s0: lo=1.0, hi=2.0; s1: lo=3.0, hi=4.0
        s0 = pack_f16x2(ONE_F16, TWO_F16)
        s1 = pack_f16x2(THREE_F16, FOUR_F16)
        # Swap terms: op_sel=(1, 0) -> s0[1]*s1[0] = 2*3 = 6; op_sel_hi=(0, 1) -> s0[0]*s1[1] = 1*4 = 4; total 10.0
        res = oracle_v_dot2_f16(s0, s1, op_sel=(1, 0), op_sel_hi=(0, 1))
        self.assertEqual(bits_to_f16(res), 10.0)

    def test_clamp_modifier(self) -> None:
        s0 = pack_f16x2(TWO_F16, TWO_F16)
        s1 = pack_f16x2(THREE_F16, FOUR_F16)
        # Result = 14.0, clamped -> 1.0
        res_high = oracle_v_dot2_f16(s0, s1, clamp=True)
        self.assertEqual(res_high, ONE_F16)

        # Result = -11.0, clamped -> +0.0
        s0_neg = pack_f16x2(NEG_ONE_F16, NEG_TWO_F16)
        res_low = oracle_v_dot2_f16(s0_neg, s1, clamp=True)
        self.assertEqual(res_low, POS_ZERO)

        # NaN clamped -> +0.0
        res_nan = oracle_v_dot2_f16(pack_f16x2(CANONICAL_QNAN, POS_ZERO), s1, clamp=True)
        self.assertEqual(res_nan, POS_ZERO)

    def test_nan_handling_and_propagation(self) -> None:
        s0 = pack_f16x2(CANONICAL_SNAN, ONE_F16)
        s1 = pack_f16x2(ONE_F16, ONE_F16)
        res = oracle_v_dot2_f16(s0, s1)
        self.assertTrue(is_nan(res))
        self.assertTrue(is_qnan(res))

    def test_infinity_handling(self) -> None:
        # +inf * 0 -> Invalid -> Quiet NaN
        s0 = pack_f16x2(POS_INF, ONE_F16)
        s1 = pack_f16x2(POS_ZERO, ONE_F16)
        res = oracle_v_dot2_f16(s0, s1)
        self.assertTrue(is_nan(res))


class TestASTOpcodeDefinitions(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_reg_attributes_and_bounds(self) -> None:
        v0 = Reg(0, RegType.VGPR)
        s5 = Reg(5, RegType.SGPR)
        self.assertEqual(v0.name, "v0")
        self.assertEqual(s5.name, "s5")
        with self.assertRaises(ValueError):
            Reg(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(-1, RegType.SGPR)

    def test_literal32_representation(self) -> None:
        lit = Literal32.from_f16x2(1.0, 2.0)
        self.assertEqual(lit.bits, 0x4000_3C00)
        self.assertIn("1", lit.name)
        self.assertIn("2", lit.name)
        with self.assertRaises(ValueError):
            Literal32(0x1_0000_0000)

    def test_vop3_operand_disassembly(self) -> None:
        op_plain = VOP3Operand(Reg(2, RegType.VGPR))
        self.assertEqual(op_plain.disassemble(), "v2")
        op_neg = VOP3Operand(Reg(4, RegType.SGPR), neg_lo=True)
        self.assertEqual(op_neg.disassemble(), "neg(s4)")
        op_abs_neg = VOP3Operand(Reg(8, RegType.VGPR), abs_lo=True, neg_hi=True)
        self.assertEqual(op_abs_neg.disassemble(), "abs/neg(v8)")

    def test_v_dot2_f16_disassembly(self) -> None:
        instr = VDot2F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.SGPR), abs_lo=True),
            clamp=True,
            op_sel=(1, 0),
            op_sel_hi=(0, 1),
        )
        expected = "v_dot2_f16_f16 v0, v1, abs(s2) op_sel:[1,0,0,1] clamp"
        self.assertEqual(instr.disassemble(), expected)

    def test_v_dot2acc_f16_disassembly(self) -> None:
        instr = VDot2AccF16Instruction(
            dst=Reg(3, RegType.VGPR),
            src0=VOP3Operand(Reg(4, RegType.VGPR)),
            src1=VOP3Operand(Reg(5, RegType.VGPR)),
            src2=VOP3Operand(Reg(6, RegType.VGPR)),
        )
        self.assertEqual(instr.disassemble(), "v_dot2acc_f16_f16 v3, v4, v5, v6")

    def test_ast_structural_equality(self) -> None:
        i1 = VDot2F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Literal32.from_f16x2(1.0, 1.0)),
        )
        i2 = VDot2F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Literal32.from_f16x2(1.0, 1.0)),
        )
        self.assertEqual(i1, i2)

    def test_invalid_destination_rejection(self) -> None:
        with self.assertRaises(TypeError):
            VDot2F16Instruction(
                dst=Reg(0, RegType.SGPR),
                src0=VOP3Operand(Reg(1, RegType.VGPR)),
                src1=VOP3Operand(Reg(2, RegType.VGPR)),
            )
        with self.assertRaises(TypeError):
            VDot2AccF16Instruction(
                dst=Reg(0, RegType.SGPR),
                src0=VOP3Operand(Reg(1, RegType.VGPR)),
                src1=VOP3Operand(Reg(2, RegType.VGPR)),
                src2=VOP3Operand(Reg(3, RegType.VGPR)),
            )


if __name__ == "__main__":
    unittest.main()
