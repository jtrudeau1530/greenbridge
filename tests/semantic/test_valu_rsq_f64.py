"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_rsq_f64.

Models IEEE 754-2008 / IEEE 754-2019 binary64 double-precision reciprocal square root
and AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics:
  D.d = 1.0 / sqrt(S0.d)

RDNA2 Double-Precision Floating-Point ISA Rules:
  1. Register Pair Allocation:
     Double-precision 64-bit values occupy aligned pairs of 32-bit vector registers:
     v[d:d+1] where low 32 bits reside in v[d] and high 32 bits in v[d+1].
  2. Reciprocal Square Root Operation:
     Computes the IEEE 754 reciprocal square root of S0.
     - Positive finite numbers: 1.0 / sqrt(S0). Exact powers of 4 and 2 yield exact results:
       rsq(1.0) = 1.0, rsq(4.0) = 0.5, rsq(16.0) = 0.25, rsq(0.25) = 2.0, rsq(0.0625) = 4.0.
     - Zeros: rsq(+0.0) = +Infinity, rsq(-0.0) = +Infinity.
     - Positive Infinity: rsq(+Infinity) = +0.0.
     - Negative numbers & -Infinity: square root of negative is invalid, producing Quiet NaN.
     - NaNs: Quiet NaN propagates; Signaling NaN is quieted.
  3. Modifiers (VOP3 encoding):
     - Input modifiers: abs (|S0|) clears sign bit 63; neg (-S0) toggles sign bit 63.
     - Output modifier: clamp saturates result to [0.0, 1.0]. Clamping negative/NaN yields +0.0.
  4. Subnormals & Flush-to-Zero:
     Under IEEE mode, subnormal double values are computed with full precision.
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
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
SIGN_MASK: int = 0x8000_0000_0000_0000
EXP_MASK: int = 0x7FF0_0000_0000_0000
FRAC_MASK: int = 0x000F_FFFF_FFFF_FFFF
QNAN_BIT: int = 0x0008_0000_0000_0000
CANONICAL_QNAN: int = 0x7FF8_0000_0000_0000

# Canonical IEEE 754 binary64 bit patterns
POS_ZERO: int = 0x0000_0000_0000_0000
NEG_ZERO: int = 0x8000_0000_0000_0000
POS_INF: int = 0x7FF0_0000_0000_0000
NEG_INF: int = 0xFFF0_0000_0000_0000

ONE_F64: int = 0x3FF0_0000_0000_0000          # 1.0
FOUR_F64: int = 0x4010_0000_0000_0000         # 4.0
SIXTEEN_F64: int = 0x4030_0000_0000_0000      # 16.0
SIXTY_FOUR_F64: int = 0x4050_0000_0000_0000   # 64.0
QUARTER_F64: int = 0x3FD0_0000_0000_0000      # 0.25
SIXTEENTH_F64: int = 0x3FB0_0000_0000_0000    # 0.0625
HALF_F64: int = 0x3FE0_0000_0000_0000         # 0.5
TWO_F64: int = 0x4000_0000_0000_0000          # 2.0
THREE_F64: int = 0x4008_0000_0000_0000        # 3.0
NINE_F64: int = 0x4022_0000_0000_0000         # 9.0
HUNDRED_F64: int = 0x4059_0000_0000_0000      # 100.0

NEG_ONE_F64: int = 0xBFF0_0000_0000_0000      # -1.0
NEG_FOUR_F64: int = 0xC010_0000_0000_0000     # -4.0
NEG_QUARTER_F64: int = 0xBFD0_0000_0000_0000  # -0.25

DBL_MIN: int = 0x0010_0000_0000_0000          # Smallest positive normal: 2^-1022
DBL_MAX: int = 0x7FEF_FFFF_FFFF_FFFF          # Largest finite positive float
SUBNORM_MIN: int = 0x0000_0000_0000_0001      # Smallest positive subnormal: 2^-1074
SUBNORM_MAX: int = 0x000F_FFFF_FFFF_FFFF      # Largest positive subnormal


def f64_to_bits(val: float) -> int:
    """Convert Python float (IEEE 754 double) to 64-bit unsigned integer bits."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def bits_to_f64(bits: int) -> float:
    """Convert 64-bit unsigned integer bits to Python float (IEEE 754 double)."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def split_f64_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK) >> 63
    exponent = (bits & EXP_MASK) >> 52
    fraction = bits & FRAC_MASK
    return sign, exponent, fraction


def words_to_f64_bits(lo: int, hi: int) -> int:
    """Combine two 32-bit VGPR words (lo, hi) into a 64-bit integer representation."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def f64_bits_to_words(bits: int) -> Tuple[int, int]:
    """Split a 64-bit integer into two 32-bit VGPR words (lo, hi)."""
    lo = bits & MASK32
    hi = (bits >> 32) & MASK32
    return lo, hi


def is_nan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 NaN."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) != 0)


def is_qnan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Quiet NaN."""
    return is_nan(bits) and bool(bits & QNAN_BIT)


def is_snan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Signaling NaN."""
    return is_nan(bits) and not bool(bits & QNAN_BIT)


def is_inf(bits: int) -> bool:
    """Return True if bits represent IEEE 754 positive or negative infinity."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) == 0)


def is_subnormal(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormal) double-precision float."""
    return ((bits & EXP_MASK) == 0) and ((bits & FRAC_MASK) != 0)


def is_zero(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK) == 0


def apply_f64_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & MASK64
    if abs_mod:
        res &= ~SIGN_MASK
    if neg_mod:
        res ^= SIGN_MASK
    return res


def oracle_v_rsq_f64(
    src_bits: int,
    abs_src: bool = False,
    neg_src: bool = False,
    clamp: bool = False,
    flush_subnormals: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_rsq_f64.

    Computes D = 1.0 / sqrt(src) according to IEEE 754-2008 / RDNA2 VALU rules:
      - abs_src / neg_src input modifiers applied before root calculation.
      - Flush-to-zero flushes subnormal input to signed zero.
      - NaN input produces Quiet NaN (or +0.0 if clamp is enabled).
      - Negative non-zero and -Infinity inputs produce Canonical Quiet NaN (or +0.0 if clamp is enabled).
      - +0.0 and -0.0 produce +Infinity (clamped to 1.0 if clamp is enabled).
      - +Infinity produces +0.0.
      - Finite positive numbers produce 1.0 / sqrt(x), clamped to [0.0, 1.0] if clamp is enabled.
    """
    s = apply_f64_modifiers(src_bits, abs_src, neg_src)

    # 1. Subnormal flush-to-zero mode
    if flush_subnormals and is_subnormal(s):
        s = NEG_ZERO if (s & SIGN_MASK) else POS_ZERO

    # 2. NaN inputs propagate quieted NaN
    if is_nan(s):
        if clamp:
            return POS_ZERO
        return s | QNAN_BIT

    # 3. Negative non-zero or negative infinity yields Quiet NaN
    if (s & SIGN_MASK) and not is_zero(s):
        if clamp:
            return POS_ZERO
        return CANONICAL_QNAN

    # 4. Zero inputs (+0.0, -0.0) produce +Infinity
    if is_zero(s):
        if clamp:
            return ONE_F64
        return POS_INF

    # 5. Positive Infinity yields +0.0
    if s == POS_INF:
        return POS_ZERO

    # 6. Finite positive calculation
    val = bits_to_f64(s)
    res_val = 1.0 / math.sqrt(val)

    if clamp:
        if res_val < 0.0:
            res_val = 0.0
        elif res_val > 1.0:
            res_val = 1.0

    return f64_to_bits(res_val)


class TestF64ModelAndOracle(unittest.TestCase):
    """Unit tests for IEEE 754 binary64 bit functions and the reference v_rsq_f64 oracle."""

    def test_exact_powers_of_four_and_two(self) -> None:
        """Exact square roots and reciprocals must be bit-exact in binary64."""
        self.assertEqual(oracle_v_rsq_f64(ONE_F64), ONE_F64)
        self.assertEqual(oracle_v_rsq_f64(FOUR_F64), HALF_F64)
        self.assertEqual(oracle_v_rsq_f64(SIXTEEN_F64), QUARTER_F64)
        self.assertEqual(oracle_v_rsq_f64(SIXTY_FOUR_F64), f64_to_bits(0.125))
        self.assertEqual(oracle_v_rsq_f64(QUARTER_F64), TWO_F64)
        self.assertEqual(oracle_v_rsq_f64(SIXTEENTH_F64), FOUR_F64)

    def test_positive_finite_non_integers(self) -> None:
        """Test reciprocal square root for general positive floating point values."""
        res_nine = bits_to_f64(oracle_v_rsq_f64(NINE_F64))
        self.assertAlmostEqual(res_nine, 1.0 / 3.0, places=15)

        res_hundred = bits_to_f64(oracle_v_rsq_f64(HUNDRED_F64))
        self.assertAlmostEqual(res_hundred, 0.1, places=15)

        res_two = bits_to_f64(oracle_v_rsq_f64(TWO_F64))
        self.assertAlmostEqual(res_two, 1.0 / math.sqrt(2.0), places=15)

    def test_zero_inputs_produce_positive_infinity(self) -> None:
        """Both +0.0 and -0.0 yield +Infinity in AMD RDNA2 RSQ semantics."""
        self.assertEqual(oracle_v_rsq_f64(POS_ZERO), POS_INF)
        self.assertEqual(oracle_v_rsq_f64(NEG_ZERO), POS_INF)

    def test_positive_infinity_yields_positive_zero(self) -> None:
        """1.0 / sqrt(+Inf) yields +0.0."""
        self.assertEqual(oracle_v_rsq_f64(POS_INF), POS_ZERO)

    def test_negative_inputs_produce_canonical_qnan(self) -> None:
        """Square root of negative real numbers produces Quiet NaN."""
        self.assertEqual(oracle_v_rsq_f64(NEG_ONE_F64), CANONICAL_QNAN)
        self.assertEqual(oracle_v_rsq_f64(NEG_FOUR_F64), CANONICAL_QNAN)
        self.assertEqual(oracle_v_rsq_f64(NEG_QUARTER_F64), CANONICAL_QNAN)
        self.assertEqual(oracle_v_rsq_f64(NEG_INF), CANONICAL_QNAN)

    def test_nan_inputs_propagate_quieted(self) -> None:
        """NaN operands propagate quieted NaN."""
        qnan_in = CANONICAL_QNAN | 0x1234
        snan_in = POS_INF | 0x0001_0000  # Signaling NaN
        self.assertTrue(is_qnan(oracle_v_rsq_f64(qnan_in)))
        self.assertTrue(is_qnan(oracle_v_rsq_f64(snan_in)))

    def test_input_modifiers_abs_and_neg(self) -> None:
        """Input modifiers abs and neg transform operand before root calculation."""
        # abs(-4.0) -> +4.0 -> rsq gives 0.5
        self.assertEqual(oracle_v_rsq_f64(NEG_FOUR_F64, abs_src=True), HALF_F64)
        # neg(4.0) -> -4.0 -> rsq gives NaN
        self.assertEqual(oracle_v_rsq_f64(FOUR_F64, neg_src=True), CANONICAL_QNAN)
        # neg(-4.0) -> +4.0 -> rsq gives 0.5
        self.assertEqual(oracle_v_rsq_f64(NEG_FOUR_F64, neg_src=True), HALF_F64)
        # abs and neg together on positive: neg(abs(4.0)) -> -4.0 -> NaN
        self.assertEqual(oracle_v_rsq_f64(FOUR_F64, abs_src=True, neg_src=True), CANONICAL_QNAN)

    def test_clamp_modifier(self) -> None:
        """Clamp modifier saturates result to [0.0, 1.0]."""
        # 0.25 -> 2.0 -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f64(QUARTER_F64, clamp=True), ONE_F64)
        # 4.0 -> 0.5 -> within [0.0, 1.0] -> unchanged
        self.assertEqual(oracle_v_rsq_f64(FOUR_F64, clamp=True), HALF_F64)
        # +0.0 -> +inf -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f64(POS_ZERO, clamp=True), ONE_F64)
        # Negative -> NaN -> clamped to +0.0
        self.assertEqual(oracle_v_rsq_f64(NEG_FOUR_F64, clamp=True), POS_ZERO)
        # NaN -> clamped to +0.0
        self.assertEqual(oracle_v_rsq_f64(CANONICAL_QNAN, clamp=True), POS_ZERO)

    def test_subnormal_handling(self) -> None:
        """Subnormal precision preservation in IEEE mode and flush in FTZ mode."""
        sub = SUBNORM_MAX
        res_ieee = oracle_v_rsq_f64(sub, flush_subnormals=False)
        self.assertFalse(is_nan(res_ieee))
        self.assertFalse(is_inf(res_ieee))
        self.assertGreater(bits_to_f64(res_ieee), 1.0)

        # In FTZ mode, subnormal flushes to +0.0, so rsq gives +Infinity
        res_ftz = oracle_v_rsq_f64(sub, flush_subnormals=True)
        self.assertEqual(res_ftz, POS_INF)

    def test_word_conversion_helpers(self) -> None:
        """Splitting and joining 32-bit dwords must roundtrip 64-bit values accurately."""
        lo, hi = f64_bits_to_words(FOUR_F64)
        rejoined = words_to_f64_bits(lo, hi)
        self.assertEqual(rejoined, FOUR_F64)

        sign, exp, frac = split_f64_bits(FOUR_F64)
        self.assertEqual(sign, 0)
        self.assertEqual(exp, 1025)
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
class RegPair:
    """An aligned 64-bit pair of 32-bit registers (v[d:d+1] or s[s:s+1])."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index <= 254):
                raise ValueError(f"VGPR index {self.index} out of valid range [0, 254] for a 64-bit pair")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 104):
                raise ValueError(f"SGPR index {self.index} out of valid range [0, 104] for a 64-bit pair")
        else:
            raise ValueError(f"Invalid register type for RegPair: {self.reg_type}")

    @property
    def low(self) -> int:
        """Low 32-bit dword register index."""
        return self.index

    @property
    def high(self) -> int:
        """High 32-bit dword register index."""
        return self.index + 1

    def __str__(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}[{self.index}:{self.index + 1}]"


@dataclass(frozen=True)
class Literal64:
    """A 64-bit immediate constant bit pattern."""
    bits: int

    def __post_init__(self) -> None:
        if not (0 <= self.bits <= MASK64):
            raise ValueError(f"Literal64 bits {hex(self.bits)} out of range [0, 2^64-1]")

    @classmethod
    def from_float(cls, val: float) -> Literal64:
        return cls(f64_to_bits(val))

    def to_float(self) -> float:
        return bits_to_f64(self.bits)

    def __str__(self) -> str:
        return f"0x{self.bits:016x}"


@dataclass(frozen=True)
class VOP3Operand:
    """A VOP3 source operand with optional input modifiers (abs and neg)."""
    reg_pair: Optional[RegPair] = None
    literal: Optional[Literal64] = None
    abs_mod: bool = False
    neg_mod: bool = False

    def __post_init__(self) -> None:
        if (self.reg_pair is None) == (self.literal is None):
            raise ValueError("VOP3Operand must specify exactly one of reg_pair or literal")

    @classmethod
    def from_vgpr(cls, index: int, abs_mod: bool = False, neg_mod: bool = False) -> VOP3Operand:
        return cls(reg_pair=RegPair(index, RegType.VGPR), abs_mod=abs_mod, neg_mod=neg_mod)

    @classmethod
    def from_sgpr(cls, index: int, abs_mod: bool = False, neg_mod: bool = False) -> VOP3Operand:
        return cls(reg_pair=RegPair(index, RegType.SGPR), abs_mod=abs_mod, neg_mod=neg_mod)

    @classmethod
    def from_lit(cls, bits: int, abs_mod: bool = False, neg_mod: bool = False) -> VOP3Operand:
        return cls(literal=Literal64(bits), abs_mod=abs_mod, neg_mod=neg_mod)

    def __str__(self) -> str:
        core = str(self.reg_pair) if self.reg_pair is not None else str(self.literal)
        if self.abs_mod:
            core = f"|{core}|"
        if self.neg_mod:
            core = f"-{core}"
        return core


@dataclass(frozen=True)
class VRSQF64Instruction:
    """Symbolic AST node for RDNA2 VALU v_rsq_f64 instruction."""
    dst: RegPair
    src0: VOP3Operand
    clamp: bool = False
    flush_denorms: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise ValueError(f"Destination must be a VGPR pair, got {self.dst.reg_type}")

    def disasm(self) -> str:
        """Render GCN/RDNA assembly representation."""
        res = f"v_rsq_f64 {self.dst}, {self.src0}"
        if self.clamp:
            res += " clamp"
        return res


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_reg_pair_creation_and_bounds(self) -> None:
        """Test register pair construction, indexing, and boundary validation."""
        v = RegPair(0, RegType.VGPR)
        self.assertEqual(v.low, 0)
        self.assertEqual(v.high, 1)
        self.assertEqual(str(v), "v[0:1]")

        s = RegPair(10, RegType.SGPR)
        self.assertEqual(s.low, 10)
        self.assertEqual(s.high, 11)
        self.assertEqual(str(s), "s[10:11]")

        # Boundary checks
        RegPair(254, RegType.VGPR)
        RegPair(104, RegType.SGPR)
        with self.assertRaises(ValueError):
            RegPair(255, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegPair(105, RegType.SGPR)
        with self.assertRaises(ValueError):
            RegPair(-1, RegType.VGPR)

    def test_literal64_creation(self) -> None:
        """Test 64-bit immediate literal operand."""
        lit = Literal64(0x3FF0_0000_0000_0000)
        self.assertEqual(lit.to_float(), 1.0)
        self.assertEqual(str(lit), "0x3ff0000000000000")

        lit2 = Literal64.from_float(4.0)
        self.assertEqual(lit2.bits, FOUR_F64)

        with self.assertRaises(ValueError):
            Literal64(-1)
        with self.assertRaises(ValueError):
            Literal64(1 << 64)

    def test_vop3_operand_formatting(self) -> None:
        """Test formatting of VOP3 operands with input modifiers."""
        op_plain = VOP3Operand.from_vgpr(4)
        self.assertEqual(str(op_plain), "v[4:5]")

        op_abs = VOP3Operand.from_vgpr(4, abs_mod=True)
        self.assertEqual(str(op_abs), "|v[4:5]|")

        op_neg = VOP3Operand.from_vgpr(4, neg_mod=True)
        self.assertEqual(str(op_neg), "-v[4:5]")

        op_neg_abs = VOP3Operand.from_vgpr(4, abs_mod=True, neg_mod=True)
        self.assertEqual(str(op_neg_abs), "-|v[4:5]|")

        op_sgpr = VOP3Operand.from_sgpr(0)
        self.assertEqual(str(op_sgpr), "s[0:1]")

        op_lit = VOP3Operand.from_lit(ONE_F64)
        self.assertEqual(str(op_lit), "0x3ff0000000000000")

        with self.assertRaises(ValueError):
            VOP3Operand(reg_pair=None, literal=None)

    def test_instruction_disasm(self) -> None:
        """Test disassembly output of VRSQF64Instruction."""
        instr_basic = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
        )
        self.assertEqual(instr_basic.disasm(), "v_rsq_f64 v[0:1], v[2:3]")

        instr_mod = VRSQF64Instruction(
            dst=RegPair(4, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(6, abs_mod=True, neg_mod=True),
            clamp=True,
        )
        self.assertEqual(instr_mod.disasm(), "v_rsq_f64 v[4:5], -|v[6:7]| clamp")

        # Destination must be VGPR
        with self.assertRaises(ValueError):
            VRSQF64Instruction(
                dst=RegPair(0, RegType.SGPR),
                src0=VOP3Operand.from_vgpr(2),
            )


# ============================================================================
# Section 3: Deterministic Scalar Instruction Interpreter
# ============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register state of an AMD RDNA2 compute unit single thread."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)
    pc: int = 0
    cycle_count: int = 0

    def read_vgpr(self, index: int) -> int:
        return self.vgprs[index] & MASK32

    def write_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = val & MASK32

    def read_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def write_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32

    def read_vgpr_pair(self, index: int) -> int:
        lo = self.read_vgpr(index)
        hi = self.read_vgpr(index + 1)
        return words_to_f64_bits(lo, hi)

    def write_vgpr_pair(self, index: int, val64: int) -> None:
        lo, hi = f64_bits_to_words(val64)
        self.write_vgpr(index, lo)
        self.write_vgpr(index + 1, hi)

    def read_sgpr_pair(self, index: int) -> int:
        lo = self.read_sgpr(index)
        hi = self.read_sgpr(index + 1)
        return words_to_f64_bits(lo, hi)

    def write_sgpr_pair(self, index: int, val64: int) -> None:
        lo, hi = f64_bits_to_words(val64)
        self.write_sgpr(index, lo)
        self.write_sgpr(index + 1, hi)

    def read_operand_raw_bits(self, op: VOP3Operand) -> int:
        if op.literal is not None:
            return op.literal.bits
        assert op.reg_pair is not None
        if op.reg_pair.reg_type == RegType.VGPR:
            return self.read_vgpr_pair(op.reg_pair.index)
        elif op.reg_pair.reg_type == RegType.SGPR:
            return self.read_sgpr_pair(op.reg_pair.index)
        raise ValueError(f"Unknown register type {op.reg_pair.reg_type}")


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 VALU instructions."""

    def __init__(self, state: Optional[ScalarMachineState] = None) -> None:
        self.state = state if state is not None else ScalarMachineState()

    def execute(self, instr: VRSQF64Instruction) -> int:
        """Step a single VRSQF64Instruction against scalar state."""
        raw_bits = self.state.read_operand_raw_bits(instr.src0)
        res_bits = oracle_v_rsq_f64(
            raw_bits,
            abs_src=instr.src0.abs_mod,
            neg_src=instr.src0.neg_mod,
            clamp=instr.clamp,
            flush_subnormals=instr.flush_denorms,
        )
        self.state.write_vgpr_pair(instr.dst.index, res_bits)
        self.state.pc += 4
        self.state.cycle_count += 1
        return res_bits


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction execution."""

    def test_vgpr_to_vgpr_execution(self) -> None:
        """Execute v_rsq_f64 reading from VGPR pair and writing to VGPR pair."""
        state = ScalarMachineState()
        # Set v[2:3] = 4.0
        state.write_vgpr_pair(2, FOUR_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, HALF_F64)
        self.assertEqual(state.read_vgpr_pair(0), HALF_F64)
        self.assertEqual(state.read_vgpr_pair(2), FOUR_F64)
        self.assertEqual(state.pc, 4)
        self.assertEqual(state.cycle_count, 1)

    def test_sgpr_source_execution(self) -> None:
        """Execute v_rsq_f64 with SGPR pair as source operand."""
        state = ScalarMachineState()
        # Set s[4:5] = 16.0
        state.write_sgpr_pair(4, SIXTEEN_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_sgpr(4),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, QUARTER_F64)
        self.assertEqual(state.read_vgpr_pair(0), QUARTER_F64)

    def test_literal_source_execution(self) -> None:
        """Execute v_rsq_f64 with immediate 64-bit constant literal."""
        state = ScalarMachineState()
        instr = VRSQF64Instruction(
            dst=RegPair(10, RegType.VGPR),
            src0=VOP3Operand.from_lit(QUARTER_F64),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, TWO_F64)
        self.assertEqual(state.read_vgpr_pair(10), TWO_F64)

    def test_input_modifiers_in_interpreter(self) -> None:
        """Execute v_rsq_f64 with abs and neg input modifiers via interpreter."""
        state = ScalarMachineState()
        state.write_vgpr_pair(6, NEG_FOUR_F64)

        # abs(-4.0) -> +4.0 -> rsq gives 0.5
        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(6, abs_mod=True),
        )
        interp = ScalarInterpreter(state)
        self.assertEqual(interp.execute(instr), HALF_F64)
        self.assertEqual(state.read_vgpr_pair(0), HALF_F64)

    def test_clamp_modifier_in_interpreter(self) -> None:
        """Execute v_rsq_f64 with clamp modifier via interpreter."""
        state = ScalarMachineState()
        state.write_vgpr_pair(2, QUARTER_F64)  # rsq(0.25) = 2.0 -> clamped to 1.0

        instr = VRSQF64Instruction(
            dst=RegPair(8, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
            clamp=True,
        )
        interp = ScalarInterpreter(state)
        self.assertEqual(interp.execute(instr), ONE_F64)
        self.assertEqual(state.read_vgpr_pair(8), ONE_F64)

    def test_register_isolation(self) -> None:
        """Verify that writing to destination does not modify unaddressed registers."""
        state = ScalarMachineState()
        state.write_vgpr(0, 0x1111_1111)
        state.write_vgpr(1, 0x2222_2222)
        state.write_vgpr(2, 0x3333_3333)
        state.write_vgpr(3, 0x4444_4444)
        state.write_vgpr_pair(4, FOUR_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(2, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(4),
        )
        interp = ScalarInterpreter(state)
        interp.execute(instr)

        # v[0:1] must remain untouched
        self.assertEqual(state.read_vgpr(0), 0x1111_1111)
        self.assertEqual(state.read_vgpr(1), 0x2222_2222)
        # v[2:3] has updated result
        self.assertEqual(state.read_vgpr_pair(2), HALF_F64)


# ============================================================================
# Section 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ============================================================================

@dataclass
class SIMDState:
    """Architectural state for a wavefront (Wave32 or Wave64) with EXEC masking."""
    num_lanes: int = 64
    exec_mask: int = MASK64
    lanes: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {self.num_lanes}")
        if not self.lanes:
            self.lanes = [[0] * 256 for _ in range(self.num_lanes)]
        if self.num_lanes == 32:
            self.exec_mask &= MASK32

    def is_lane_active(self, lane_idx: int) -> bool:
        """True if the lane's bit is enabled in the wavefront EXEC mask."""
        return bool(self.exec_mask & (1 << lane_idx))

    def read_lane_vgpr(self, lane_idx: int, reg_idx: int) -> int:
        return self.lanes[lane_idx][reg_idx] & MASK32

    def write_lane_vgpr(self, lane_idx: int, reg_idx: int, val: int) -> None:
        self.lanes[lane_idx][reg_idx] = val & MASK32

    def read_lane_vgpr_pair(self, lane_idx: int, reg_idx: int) -> int:
        lo = self.read_lane_vgpr(lane_idx, reg_idx)
        hi = self.read_lane_vgpr(lane_idx, reg_idx + 1)
        return words_to_f64_bits(lo, hi)

    def write_lane_vgpr_pair(self, lane_idx: int, reg_idx: int, val64: int) -> None:
        lo, hi = f64_bits_to_words(val64)
        self.write_lane_vgpr(lane_idx, reg_idx, lo)
        self.write_lane_vgpr(lane_idx, reg_idx + 1, hi)

    def read_sgpr_pair(self, reg_idx: int) -> int:
        lo = self.sgprs[reg_idx] & MASK32
        hi = self.sgprs[reg_idx + 1] & MASK32
        return words_to_f64_bits(lo, hi)

    def write_sgpr_pair(self, reg_idx: int, val64: int) -> None:
        lo, hi = f64_bits_to_words(val64)
        self.sgprs[reg_idx] = lo
        self.sgprs[reg_idx + 1] = hi

    def read_lane_operand_bits(self, lane_idx: int, op: VOP3Operand) -> int:
        if op.literal is not None:
            return op.literal.bits
        assert op.reg_pair is not None
        if op.reg_pair.reg_type == RegType.VGPR:
            return self.read_lane_vgpr_pair(lane_idx, op.reg_pair.index)
        elif op.reg_pair.reg_type == RegType.SGPR:
            return self.read_sgpr_pair(op.reg_pair.index)
        raise ValueError(f"Unknown register type {op.reg_pair.reg_type}")


class SIMDInterpreter:
    """SIMD wavefront execution engine modeling divergent multi-lane execution."""

    def __init__(self, state: Optional[SIMDState] = None) -> None:
        self.state = state if state is not None else SIMDState()

    def execute(self, instr: VRSQF64Instruction) -> List[Optional[int]]:
        """Execute VRSQF64Instruction across all active lanes conditioned on EXEC mask."""
        results: List[Optional[int]] = []
        for lane in range(self.state.num_lanes):
            if not self.state.is_lane_active(lane):
                results.append(None)
                continue
            raw_bits = self.state.read_lane_operand_bits(lane, instr.src0)
            res = oracle_v_rsq_f64(
                raw_bits,
                abs_src=instr.src0.abs_mod,
                neg_src=instr.src0.neg_mod,
                clamp=instr.clamp,
                flush_subnormals=instr.flush_denorms,
            )
            self.state.write_lane_vgpr_pair(lane, instr.dst.index, res)
            results.append(res)
        return results


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_full_execution(self) -> None:
        """Wave32 wavefront executes across all 32 lanes with divergent per-lane inputs."""
        simd = SIMDState(num_lanes=32, exec_mask=MASK32)
        # Populate each lane with a distinct power of 4 or known square
        test_inputs = [
            ONE_F64, FOUR_F64, SIXTEEN_F64, SIXTY_FOUR_F64,
            QUARTER_F64, SIXTEENTH_F64, POS_ZERO, POS_INF,
        ] * 4  # 32 inputs
        for lane, inp in enumerate(test_inputs):
            simd.write_lane_vgpr_pair(lane, 2, inp)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        self.assertEqual(len(res), 32)
        for lane, inp in enumerate(test_inputs):
            expected = oracle_v_rsq_f64(inp)
            self.assertEqual(res[lane], expected)
            self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), expected)

    def test_wave64_full_execution(self) -> None:
        """Wave64 wavefront executes across all 64 lanes."""
        simd = SIMDState(num_lanes=64, exec_mask=MASK64)
        for lane in range(64):
            simd.write_lane_vgpr_pair(lane, 4, FOUR_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(6, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(4),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        self.assertEqual(len(res), 64)
        for lane in range(64):
            self.assertEqual(res[lane], HALF_F64)
            self.assertEqual(simd.read_lane_vgpr_pair(lane, 6), HALF_F64)

    def test_checkerboard_divergence_even_lanes(self) -> None:
        """EXEC mask enables even lanes only; odd lanes must remain completely unmodified."""
        exec_even = 0x5555_5555_5555_5555
        simd = SIMDState(num_lanes=64, exec_mask=exec_even)
        SENTINEL = 0xDEAD_BEEF_CAFE_BABE

        for lane in range(64):
            simd.write_lane_vgpr_pair(lane, 0, SENTINEL)
            simd.write_lane_vgpr_pair(lane, 2, FOUR_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(res[lane], HALF_F64)
                self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), HALF_F64)
            else:
                self.assertIsNone(res[lane])
                self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), SENTINEL)

    def test_wave_half_divergence(self) -> None:
        """EXEC mask enables lower 32 lanes (half-wave) only."""
        exec_lo32 = 0x0000_0000_FFFF_FFFF
        simd = SIMDState(num_lanes=64, exec_mask=exec_lo32)
        SENTINEL = 0x1234_5678_9ABC_DEF0

        for lane in range(64):
            simd.write_lane_vgpr_pair(lane, 0, SENTINEL)
            simd.write_lane_vgpr_pair(lane, 2, SIXTEEN_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        for lane in range(32):
            self.assertEqual(res[lane], QUARTER_F64)
            self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), QUARTER_F64)

        for lane in range(32, 64):
            self.assertIsNone(res[lane])
            self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), SENTINEL)

    def test_single_lane_active(self) -> None:
        """EXEC mask enables exactly one lane (lane 17)."""
        target_lane = 17
        simd = SIMDState(num_lanes=64, exec_mask=(1 << target_lane))
        SENTINEL = 0xAAAA_BBBB_CCCC_DDDD

        for lane in range(64):
            simd.write_lane_vgpr_pair(lane, 0, SENTINEL)
            simd.write_lane_vgpr_pair(lane, 2, QUARTER_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        for lane in range(64):
            if lane == target_lane:
                self.assertEqual(res[lane], TWO_F64)
                self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), TWO_F64)
            else:
                self.assertIsNone(res[lane])
                self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), SENTINEL)

    def test_zero_exec_mask_noop(self) -> None:
        """An EXEC mask of 0 is a complete no-op across all registers and lanes."""
        simd = SIMDState(num_lanes=64, exec_mask=0)
        SENTINEL = 0x5555_AAAA_5555_AAAA

        for lane in range(64):
            simd.write_lane_vgpr_pair(lane, 0, SENTINEL)
            simd.write_lane_vgpr_pair(lane, 2, FOUR_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        self.assertTrue(all(r is None for r in res))
        for lane in range(64):
            self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), SENTINEL)

    def test_scalar_broadcast_to_active_simd_lanes(self) -> None:
        """Source operand from SGPR pair is broadcast identically across all active lanes."""
        simd = SIMDState(num_lanes=64, exec_mask=MASK64)
        simd.write_sgpr_pair(10, SIXTEEN_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(4, RegType.VGPR),
            src0=VOP3Operand.from_sgpr(10),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        for lane in range(64):
            self.assertEqual(res[lane], QUARTER_F64)
            self.assertEqual(simd.read_lane_vgpr_pair(lane, 4), QUARTER_F64)


if __name__ == "__main__":
    unittest.main()
