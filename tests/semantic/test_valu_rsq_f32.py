"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_rsq_f32.

Models IEEE 754-2008 / IEEE 754-2019 binary32 single-precision reciprocal square root
and AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics:
  D.f = 1.0 / sqrt(S0.f)

RDNA2 Single-Precision Floating-Point ISA Rules:
  1. Register Allocation:
     Single-precision 32-bit values occupy individual 32-bit vector registers v[d].
  2. Reciprocal Square Root Operation:
     Computes the IEEE 754 reciprocal square root of S0:
     - Positive finite numbers: 1.0 / sqrt(S0). Exact powers of 4 and 2 yield exact results:
       rsq(1.0) = 1.0, rsq(4.0) = 0.5, rsq(16.0) = 0.25, rsq(0.25) = 2.0, rsq(0.0625) = 4.0.
     - Zeros: rsq(+0.0) = +Infinity, rsq(-0.0) = +Infinity.
     - Positive Infinity: rsq(+Infinity) = +0.0.
     - Negative numbers & -Infinity: square root of negative is invalid, producing Quiet NaN.
     - NaNs: Quiet NaN propagates; Signaling NaN is quieted.
  3. Modifiers (VOP3/VOP1 encoding):
     - Input modifiers: abs (|S0|) clears sign bit 31; neg (-S0) toggles sign bit 31.
     - Output modifier: clamp saturates result to [0.0, 1.0]. Clamping negative/NaN yields +0.0.
  4. Subnormals & Flush-to-Zero:
     Under IEEE mode, subnormal single-precision values are computed with full precision.
     Under flush-to-zero (FTZ), subnormal inputs flush to signed zero, yielding +Infinity.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFF_FFFF
SIGN_MASK_F32: int = 0x8000_0000
EXP_MASK_F32: int = 0x7F80_0000
MANT_MASK_F32: int = 0x007F_FFFF
QNAN_BIT_F32: int = 0x0040_0000
CANONICAL_QNAN_F32: int = 0x7FC0_0000

# Canonical IEEE 754 binary32 bit patterns
POS_ZERO_F32: int = 0x0000_0000
NEG_ZERO_F32: int = 0x8000_0000
POS_INF_F32: int = 0x7F80_0000
NEG_INF_F32: int = 0xFF80_0000

ONE_F32: int = 0x3F80_0000          # 1.0
FOUR_F32: int = 0x4080_0000         # 4.0
SIXTEEN_F32: int = 0x4180_0000      # 16.0
SIXTY_FOUR_F32: int = 0x4280_0000   # 64.0
QUARTER_F32: int = 0x3E80_0000      # 0.25
SIXTEENTH_F32: int = 0x3D80_0000    # 0.0625
HALF_F32: int = 0x3F00_0000         # 0.5
TWO_F32: int = 0x4000_0000          # 2.0
THREE_F32: int = 0x4040_0000        # 3.0
NINE_F32: int = 0x4110_0000         # 9.0
HUNDRED_F32: int = 0x42C8_0000      # 100.0

NEG_ONE_F32: int = 0xBF80_0000      # -1.0
NEG_FOUR_F32: int = 0xC080_0000     # -4.0
NEG_QUARTER_F32: int = 0xBE80_0000  # -0.25

FLT_MIN: int = 0x0080_0000          # Smallest positive normal: 2^-126
FLT_MAX: int = 0x7F7F_FFFF          # Largest finite positive float: (2 - 2^-23) * 2^127
SUBNORM_MIN_F32: int = 0x0000_0001  # Smallest positive subnormal: 2^-149
SUBNORM_MAX_F32: int = 0x007F_FFFF  # Largest positive subnormal: (1 - 2^-23) * 2^-126


def f32_to_bits(val: float) -> int:
    """Convert Python float to 32-bit unsigned IEEE 754 binary32 representation."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_f32(bits: int) -> float:
    """Convert 32-bit unsigned integer bits to Python float (IEEE 754 single precision)."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def split_f32_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK_F32) >> 31
    exponent = (bits & EXP_MASK_F32) >> 23
    fraction = bits & MANT_MASK_F32
    return sign, exponent, fraction


def is_nan_f32(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 single-precision NaN."""
    return ((bits & EXP_MASK_F32) == EXP_MASK_F32) and ((bits & MANT_MASK_F32) != 0)


def is_qnan_f32(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Quiet NaN."""
    return is_nan_f32(bits) and bool(bits & QNAN_BIT_F32)


def is_snan_f32(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Signaling NaN."""
    return is_nan_f32(bits) and not bool(bits & QNAN_BIT_F32)


def is_inf_f32(bits: int) -> bool:
    """Return True if bits represent IEEE 754 positive or negative infinity."""
    return ((bits & EXP_MASK_F32) == EXP_MASK_F32) and ((bits & MANT_MASK_F32) == 0)


def is_subnormal_f32(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormal) single-precision float."""
    return ((bits & EXP_MASK_F32) == 0) and ((bits & MANT_MASK_F32) != 0)


def is_zero_f32(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK_F32) == 0


def apply_f32_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP3/VOP1 input modifiers: abs clears sign bit 31; neg toggles sign bit 31."""
    res = bits & MASK32
    if abs_mod:
        res &= ~SIGN_MASK_F32
    if neg_mod:
        res ^= SIGN_MASK_F32
    return res


def oracle_v_rsq_f32(
    src_bits: int,
    abs_src: bool = False,
    neg_src: bool = False,
    clamp: bool = False,
    flush_subnormals: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_rsq_f32.

    Computes D = 1.0 / sqrt(src) according to IEEE 754-2008 / RDNA2 VALU rules:
      - abs_src / neg_src input modifiers applied before root calculation.
      - Flush-to-zero flushes subnormal input to signed zero.
      - NaN input produces Quiet NaN (or +0.0 if clamp is enabled).
      - Negative non-zero and -Infinity inputs produce Canonical Quiet NaN (or +0.0 if clamp is enabled).
      - +0.0 and -0.0 produce +Infinity (clamped to 1.0 if clamp is enabled).
      - +Infinity produces +0.0.
      - Finite positive numbers produce 1.0 / sqrt(x), clamped to [0.0, 1.0] if clamp is enabled.
    """
    s = apply_f32_modifiers(src_bits, abs_src, neg_src)

    # 1. Subnormal flush-to-zero mode
    if flush_subnormals and is_subnormal_f32(s):
        s = NEG_ZERO_F32 if (s & SIGN_MASK_F32) else POS_ZERO_F32

    # 2. NaN inputs propagate quieted NaN
    if is_nan_f32(s):
        if clamp:
            return POS_ZERO_F32
        return s | QNAN_BIT_F32

    # 3. Negative non-zero or negative infinity yields Quiet NaN
    if (s & SIGN_MASK_F32) and not is_zero_f32(s):
        if clamp:
            return POS_ZERO_F32
        return CANONICAL_QNAN_F32

    # 4. Zero inputs (+0.0, -0.0) produce +Infinity
    if is_zero_f32(s):
        if clamp:
            return ONE_F32
        return POS_INF_F32

    # 5. Positive Infinity yields +0.0
    if s == POS_INF_F32:
        return POS_ZERO_F32

    # 6. Finite positive calculation
    val = bits_to_f32(s)
    res_val = 1.0 / math.sqrt(val)

    if clamp:
        if res_val < 0.0:
            res_val = 0.0
        elif res_val > 1.0:
            res_val = 1.0

    return f32_to_bits(res_val)


class TestF32ModelAndOracle(unittest.TestCase):
    """Unit tests for IEEE 754 binary32 bit functions and the reference v_rsq_f32 oracle."""

    def test_exact_powers_of_four_and_two(self) -> None:
        """Exact square roots and reciprocals must be bit-exact in binary32."""
        self.assertEqual(oracle_v_rsq_f32(ONE_F32), ONE_F32)
        self.assertEqual(oracle_v_rsq_f32(FOUR_F32), HALF_F32)
        self.assertEqual(oracle_v_rsq_f32(SIXTEEN_F32), QUARTER_F32)
        self.assertEqual(oracle_v_rsq_f32(SIXTY_FOUR_F32), f32_to_bits(0.125))
        self.assertEqual(oracle_v_rsq_f32(QUARTER_F32), TWO_F32)
        self.assertEqual(oracle_v_rsq_f32(SIXTEENTH_F32), FOUR_F32)

    def test_positive_finite_non_integers(self) -> None:
        """Test reciprocal square root for general positive floating point values."""
        res_nine = bits_to_f32(oracle_v_rsq_f32(NINE_F32))
        self.assertAlmostEqual(res_nine, 1.0 / 3.0, places=6)

        res_hundred = bits_to_f32(oracle_v_rsq_f32(HUNDRED_F32))
        self.assertAlmostEqual(res_hundred, 0.1, places=6)

        res_two = bits_to_f32(oracle_v_rsq_f32(TWO_F32))
        self.assertAlmostEqual(res_two, 1.0 / math.sqrt(2.0), places=6)

    def test_zero_inputs_produce_positive_infinity(self) -> None:
        """Both +0.0 and -0.0 yield +Infinity in AMD RDNA2 RSQ semantics."""
        self.assertEqual(oracle_v_rsq_f32(POS_ZERO_F32), POS_INF_F32)
        self.assertEqual(oracle_v_rsq_f32(NEG_ZERO_F32), POS_INF_F32)

    def test_positive_infinity_yields_positive_zero(self) -> None:
        """1.0 / sqrt(+Inf) yields +0.0."""
        self.assertEqual(oracle_v_rsq_f32(POS_INF_F32), POS_ZERO_F32)

    def test_negative_inputs_produce_canonical_qnan(self) -> None:
        """Square root of negative real numbers produces Quiet NaN."""
        self.assertEqual(oracle_v_rsq_f32(NEG_ONE_F32), CANONICAL_QNAN_F32)
        self.assertEqual(oracle_v_rsq_f32(NEG_FOUR_F32), CANONICAL_QNAN_F32)
        self.assertEqual(oracle_v_rsq_f32(NEG_QUARTER_F32), CANONICAL_QNAN_F32)
        self.assertEqual(oracle_v_rsq_f32(NEG_INF_F32), CANONICAL_QNAN_F32)

    def test_nan_inputs_propagate_quieted(self) -> None:
        """NaN operands propagate quieted NaN."""
        qnan_in = CANONICAL_QNAN_F32 | 0x1234
        snan_in = POS_INF_F32 | 0x0001_0000  # Signaling NaN
        self.assertTrue(is_qnan_f32(oracle_v_rsq_f32(qnan_in)))
        self.assertTrue(is_qnan_f32(oracle_v_rsq_f32(snan_in)))

    def test_input_modifiers_abs_and_neg(self) -> None:
        """Input modifiers abs and neg transform operand before root calculation."""
        # abs(-4.0) -> +4.0 -> rsq gives 0.5
        self.assertEqual(oracle_v_rsq_f32(NEG_FOUR_F32, abs_src=True), HALF_F32)
        # neg(4.0) -> -4.0 -> rsq gives NaN
        self.assertEqual(oracle_v_rsq_f32(FOUR_F32, neg_src=True), CANONICAL_QNAN_F32)
        # neg(-4.0) -> +4.0 -> rsq gives 0.5
        self.assertEqual(oracle_v_rsq_f32(NEG_FOUR_F32, neg_src=True), HALF_F32)
        # abs and neg together on positive: neg(abs(4.0)) -> -4.0 -> NaN
        self.assertEqual(oracle_v_rsq_f32(FOUR_F32, abs_src=True, neg_src=True), CANONICAL_QNAN_F32)

    def test_clamp_modifier(self) -> None:
        """Clamp modifier saturates result to [0.0, 1.0]."""
        # 0.25 -> 2.0 -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f32(QUARTER_F32, clamp=True), ONE_F32)
        # 4.0 -> 0.5 -> within [0.0, 1.0] -> unchanged
        self.assertEqual(oracle_v_rsq_f32(FOUR_F32, clamp=True), HALF_F32)
        # +0.0 -> +inf -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f32(POS_ZERO_F32, clamp=True), ONE_F32)
        # Negative -> NaN -> clamped to +0.0
        self.assertEqual(oracle_v_rsq_f32(NEG_FOUR_F32, clamp=True), POS_ZERO_F32)
        # NaN -> clamped to +0.0
        self.assertEqual(oracle_v_rsq_f32(CANONICAL_QNAN_F32, clamp=True), POS_ZERO_F32)

    def test_subnormal_handling(self) -> None:
        """Subnormal precision preservation in IEEE mode and flush in FTZ mode."""
        sub = SUBNORM_MAX_F32
        res_ieee = oracle_v_rsq_f32(sub, flush_subnormals=False)
        self.assertFalse(is_nan_f32(res_ieee))
        self.assertFalse(is_inf_f32(res_ieee))
        self.assertGreater(bits_to_f32(res_ieee), 1.0)

        # In FTZ mode, subnormal flushes to +0.0, so rsq gives +Infinity
        res_ftz = oracle_v_rsq_f32(sub, flush_subnormals=True)
        self.assertEqual(res_ftz, POS_INF_F32)

    def test_bit_split_helper(self) -> None:
        """Verify binary32 field extraction."""
        sign, exp, frac = split_f32_bits(FOUR_F32)
        self.assertEqual(sign, 0)
        self.assertEqual(exp, 129)
        self.assertEqual(frac, 0)


# ============================================================================
# Section 2: Symbolic AST Opcode Definitions and Typed Expressions
# ============================================================================

class RegType(Enum):
    """Architectural register file identifier."""
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class RegRef:
    """A single 32-bit vector or scalar register (v[d] or s[s])."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index {self.index} out of valid range [0, 255]")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index {self.index} out of valid range [0, 105]")
        else:
            raise ValueError(f"Invalid register type for RegRef: {self.reg_type}")

    def __str__(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class Literal32:
    """A 32-bit immediate constant bit pattern."""
    bits: int

    def __post_init__(self) -> None:
        if not (0 <= self.bits <= MASK32):
            raise ValueError(f"Literal32 bits {hex(self.bits)} out of range [0, 2^32-1]")

    @classmethod
    def from_float(cls, val: float) -> Literal32:
        return cls(f32_to_bits(val))

    def to_float(self) -> float:
        return bits_to_f32(self.bits)

    def __str__(self) -> str:
        return f"0x{self.bits:08x}"


@dataclass(frozen=True)
class VOP1Operand:
    """A VOP1/VOP3 source operand with optional input modifiers (abs and neg)."""
    reg: Optional[RegRef] = None
    literal: Optional[Literal32] = None
    abs_mod: bool = False
    neg_mod: bool = False

    def __post_init__(self) -> None:
        if (self.reg is None) == (self.literal is None):
            raise ValueError("VOP1Operand must specify exactly one of reg or literal")

    @classmethod
    def from_vgpr(cls, index: int, abs_mod: bool = False, neg_mod: bool = False) -> VOP1Operand:
        return cls(reg=RegRef(index, RegType.VGPR), abs_mod=abs_mod, neg_mod=neg_mod)

    @classmethod
    def from_sgpr(cls, index: int, abs_mod: bool = False, neg_mod: bool = False) -> VOP1Operand:
        return cls(reg=RegRef(index, RegType.SGPR), abs_mod=abs_mod, neg_mod=neg_mod)

    @classmethod
    def from_lit(cls, bits: int, abs_mod: bool = False, neg_mod: bool = False) -> VOP1Operand:
        return cls(literal=Literal32(bits), abs_mod=abs_mod, neg_mod=neg_mod)

    def __str__(self) -> str:
        core = str(self.reg) if self.reg is not None else str(self.literal)
        if self.abs_mod:
            core = f"|{core}|"
        if self.neg_mod:
            core = f"-{core}"
        return core


@dataclass(frozen=True)
class VRSQF32Instruction:
    """Symbolic AST node for RDNA2 VALU v_rsq_f32 instruction."""
    dst: RegRef
    src0: VOP1Operand
    clamp: bool = False
    flush_denorms: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise ValueError(f"Destination must be a VGPR, got {self.dst.reg_type}")

    def disasm(self) -> str:
        """Render GCN/RDNA assembly representation."""
        res = f"v_rsq_f32 {self.dst}, {self.src0}"
        if self.clamp:
            res += " clamp"
        return res


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_reg_ref_creation_and_bounds(self) -> None:
        """Test register reference construction, indexing, and boundary validation."""
        v = RegRef(0, RegType.VGPR)
        self.assertEqual(v.index, 0)
        self.assertEqual(str(v), "v0")

        s = RegRef(10, RegType.SGPR)
        self.assertEqual(s.index, 10)
        self.assertEqual(str(s), "s10")

        # Boundary checks
        RegRef(255, RegType.VGPR)
        RegRef(105, RegType.SGPR)
        with self.assertRaises(ValueError):
            RegRef(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegRef(106, RegType.SGPR)
        with self.assertRaises(ValueError):
            RegRef(-1, RegType.VGPR)

    def test_literal32_creation(self) -> None:
        """Test 32-bit immediate literal operand."""
        lit = Literal32(ONE_F32)
        self.assertEqual(lit.to_float(), 1.0)
        self.assertEqual(str(lit), "0x3f800000")

        lit2 = Literal32.from_float(4.0)
        self.assertEqual(lit2.bits, FOUR_F32)

        with self.assertRaises(ValueError):
            Literal32(-1)
        with self.assertRaises(ValueError):
            Literal32(1 << 32)

    def test_vop1_operand_formatting(self) -> None:
        """Test formatting of VOP1/VOP3 operands with input modifiers."""
        op_plain = VOP1Operand.from_vgpr(4)
        self.assertEqual(str(op_plain), "v4")

        op_abs = VOP1Operand.from_vgpr(4, abs_mod=True)
        self.assertEqual(str(op_abs), "|v4|")

        op_neg = VOP1Operand.from_vgpr(4, neg_mod=True)
        self.assertEqual(str(op_neg), "-v4")

        op_neg_abs = VOP1Operand.from_vgpr(4, abs_mod=True, neg_mod=True)
        self.assertEqual(str(op_neg_abs), "-|v4|")

        op_sgpr = VOP1Operand.from_sgpr(0)
        self.assertEqual(str(op_sgpr), "s0")

        op_lit = VOP1Operand.from_lit(ONE_F32)
        self.assertEqual(str(op_lit), "0x3f800000")

        with self.assertRaises(ValueError):
            VOP1Operand(reg=None, literal=None)

    def test_instruction_disasm(self) -> None:
        """Test disassembly output of VRSQF32Instruction."""
        instr_basic = VRSQF32Instruction(
            dst=RegRef(0, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(2),
        )
        self.assertEqual(instr_basic.disasm(), "v_rsq_f32 v0, v2")

        instr_mod = VRSQF32Instruction(
            dst=RegRef(4, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(6, abs_mod=True, neg_mod=True),
            clamp=True,
        )
        self.assertEqual(instr_mod.disasm(), "v_rsq_f32 v4, -|v6| clamp")

        # Destination must be VGPR
        with self.assertRaises(ValueError):
            VRSQF32Instruction(
                dst=RegRef(0, RegType.SGPR),
                src0=VOP1Operand.from_vgpr(2),
            )


if __name__ == "__main__":
    unittest.main()
