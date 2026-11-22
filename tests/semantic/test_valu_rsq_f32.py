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

    def read_operand_raw_bits(self, op: VOP1Operand) -> int:
        if op.literal is not None:
            return op.literal.bits
        assert op.reg is not None
        if op.reg.reg_type == RegType.VGPR:
            return self.read_vgpr(op.reg.index)
        elif op.reg.reg_type == RegType.SGPR:
            return self.read_sgpr(op.reg.index)
        raise ValueError(f"Unknown register type {op.reg.reg_type}")


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 VALU instructions."""

    def __init__(self, state: Optional[ScalarMachineState] = None) -> None:
        self.state = state if state is not None else ScalarMachineState()

    def execute(self, instr: VRSQF32Instruction) -> int:
        """Step a single VRSQF32Instruction against scalar state."""
        raw_bits = self.state.read_operand_raw_bits(instr.src0)
        res_bits = oracle_v_rsq_f32(
            raw_bits,
            abs_src=instr.src0.abs_mod,
            neg_src=instr.src0.neg_mod,
            clamp=instr.clamp,
            flush_subnormals=instr.flush_denorms,
        )
        self.state.write_vgpr(instr.dst.index, res_bits)
        self.state.pc += 4
        self.state.cycle_count += 1
        return res_bits


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction execution."""

    def test_vgpr_to_vgpr_execution(self) -> None:
        """Execute v_rsq_f32 reading from VGPR and writing to VGPR."""
        state = ScalarMachineState()
        # Set v2 = 4.0
        state.write_vgpr(2, FOUR_F32)

        instr = VRSQF32Instruction(
            dst=RegRef(0, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(2),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, HALF_F32)
        self.assertEqual(state.read_vgpr(0), HALF_F32)
        self.assertEqual(state.read_vgpr(2), FOUR_F32)
        self.assertEqual(state.pc, 4)
        self.assertEqual(state.cycle_count, 1)

    def test_sgpr_source_execution(self) -> None:
        """Execute v_rsq_f32 with SGPR as source operand."""
        state = ScalarMachineState()
        # Set s4 = 16.0
        state.write_sgpr(4, SIXTEEN_F32)

        instr = VRSQF32Instruction(
            dst=RegRef(0, RegType.VGPR),
            src0=VOP1Operand.from_sgpr(4),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, QUARTER_F32)
        self.assertEqual(state.read_vgpr(0), QUARTER_F32)

    def test_literal_source_execution(self) -> None:
        """Execute v_rsq_f32 with immediate 32-bit constant literal."""
        state = ScalarMachineState()
        instr = VRSQF32Instruction(
            dst=RegRef(10, RegType.VGPR),
            src0=VOP1Operand.from_lit(QUARTER_F32),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, TWO_F32)
        self.assertEqual(state.read_vgpr(10), TWO_F32)

    def test_input_modifiers_in_interpreter(self) -> None:
        """Execute v_rsq_f32 with abs and neg input modifiers via interpreter."""
        state = ScalarMachineState()
        state.write_vgpr(6, NEG_FOUR_F32)

        # abs(-4.0) -> +4.0 -> rsq gives 0.5
        instr = VRSQF32Instruction(
            dst=RegRef(0, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(6, abs_mod=True),
        )
        interp = ScalarInterpreter(state)
        self.assertEqual(interp.execute(instr), HALF_F32)
        self.assertEqual(state.read_vgpr(0), HALF_F32)

    def test_clamp_modifier_in_interpreter(self) -> None:
        """Execute v_rsq_f32 with clamp modifier via interpreter."""
        state = ScalarMachineState()
        state.write_vgpr(2, QUARTER_F32)  # rsq(0.25) = 2.0 -> clamped to 1.0

        instr = VRSQF32Instruction(
            dst=RegRef(8, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(2),
            clamp=True,
        )
        interp = ScalarInterpreter(state)
        self.assertEqual(interp.execute(instr), ONE_F32)
        self.assertEqual(state.read_vgpr(8), ONE_F32)

    def test_register_isolation(self) -> None:
        """Verify that writing to destination does not modify unaddressed registers."""
        state = ScalarMachineState()
        state.write_vgpr(0, 0x1111_1111)
        state.write_vgpr(1, 0x2222_2222)
        state.write_vgpr(2, FOUR_F32)

        instr = VRSQF32Instruction(
            dst=RegRef(1, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(2),
        )
        interp = ScalarInterpreter(state)
        interp.execute(instr)

        # v0 must remain untouched
        self.assertEqual(state.read_vgpr(0), 0x1111_1111)
        # v1 has updated result
        self.assertEqual(state.read_vgpr(1), HALF_F32)


# ============================================================================
# Section 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ============================================================================

@dataclass
class SIMDState:
    """Architectural state for a wavefront (Wave32 or Wave64) with EXEC masking."""
    num_lanes: int = 64
    exec_mask: int = 0xFFFF_FFFF_FFFF_FFFF
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

    def read_sgpr(self, reg_idx: int) -> int:
        return self.sgprs[reg_idx] & MASK32

    def write_sgpr(self, reg_idx: int, val: int) -> None:
        self.sgprs[reg_idx] = val & MASK32

    def read_lane_operand_bits(self, lane_idx: int, op: VOP1Operand) -> int:
        if op.literal is not None:
            return op.literal.bits
        assert op.reg is not None
        if op.reg.reg_type == RegType.VGPR:
            return self.read_lane_vgpr(lane_idx, op.reg.index)
        elif op.reg.reg_type == RegType.SGPR:
            return self.read_sgpr(op.reg.index)
        raise ValueError(f"Unknown register type {op.reg.reg_type}")


class SIMDInterpreter:
    """SIMD wavefront execution engine modeling divergent multi-lane execution."""

    def __init__(self, state: Optional[SIMDState] = None) -> None:
        self.state = state if state is not None else SIMDState()

    def execute(self, instr: VRSQF32Instruction) -> List[Optional[int]]:
        """Execute VRSQF32Instruction across all active lanes conditioned on EXEC mask."""
        results: List[Optional[int]] = []
        for lane in range(self.state.num_lanes):
            if not self.state.is_lane_active(lane):
                results.append(None)
                continue
            raw_bits = self.state.read_lane_operand_bits(lane, instr.src0)
            res = oracle_v_rsq_f32(
                raw_bits,
                abs_src=instr.src0.abs_mod,
                neg_src=instr.src0.neg_mod,
                clamp=instr.clamp,
                flush_subnormals=instr.flush_denorms,
            )
            self.state.write_lane_vgpr(lane, instr.dst.index, res)
            results.append(res)
        return results


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_full_execution(self) -> None:
        """Wave32 wavefront executes across all 32 lanes with divergent per-lane inputs."""
        simd = SIMDState(num_lanes=32, exec_mask=MASK32)
        test_inputs = [
            ONE_F32, FOUR_F32, SIXTEEN_F32, SIXTY_FOUR_F32,
            QUARTER_F32, SIXTEENTH_F32, POS_ZERO_F32, POS_INF_F32,
        ] * 4  # 32 inputs
        for lane, inp in enumerate(test_inputs):
            simd.write_lane_vgpr(lane, 2, inp)

        instr = VRSQF32Instruction(
            dst=RegRef(0, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        self.assertEqual(len(res), 32)
        for lane, inp in enumerate(test_inputs):
            expected = oracle_v_rsq_f32(inp)
            self.assertEqual(res[lane], expected)
            self.assertEqual(simd.read_lane_vgpr(lane, 0), expected)

    def test_wave64_full_execution(self) -> None:
        """Wave64 wavefront executes across all 64 lanes."""
        simd = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            simd.write_lane_vgpr(lane, 4, FOUR_F32)

        instr = VRSQF32Instruction(
            dst=RegRef(6, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(4),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        self.assertEqual(len(res), 64)
        for lane in range(64):
            self.assertEqual(res[lane], HALF_F32)
            self.assertEqual(simd.read_lane_vgpr(lane, 6), HALF_F32)

    def test_checkerboard_divergence_even_lanes(self) -> None:
        """EXEC mask enables even lanes only; odd lanes must remain completely unmodified."""
        exec_even = 0x5555_5555_5555_5555
        simd = SIMDState(num_lanes=64, exec_mask=exec_even)
        SENTINEL = 0xDEAD_BEEF

        for lane in range(64):
            simd.write_lane_vgpr(lane, 0, SENTINEL)
            simd.write_lane_vgpr(lane, 2, FOUR_F32)

        instr = VRSQF32Instruction(
            dst=RegRef(0, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(res[lane], HALF_F32)
                self.assertEqual(simd.read_lane_vgpr(lane, 0), HALF_F32)
            else:
                self.assertIsNone(res[lane])
                self.assertEqual(simd.read_lane_vgpr(lane, 0), SENTINEL)

    def test_wave_half_divergence(self) -> None:
        """EXEC mask enables lower 32 lanes (half-wave) only."""
        exec_lo32 = 0x0000_0000_FFFF_FFFF
        simd = SIMDState(num_lanes=64, exec_mask=exec_lo32)
        SENTINEL = 0x1234_5678

        for lane in range(64):
            simd.write_lane_vgpr(lane, 0, SENTINEL)
            simd.write_lane_vgpr(lane, 2, SIXTEEN_F32)

        instr = VRSQF32Instruction(
            dst=RegRef(0, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        for lane in range(32):
            self.assertEqual(res[lane], QUARTER_F32)
            self.assertEqual(simd.read_lane_vgpr(lane, 0), QUARTER_F32)

        for lane in range(32, 64):
            self.assertIsNone(res[lane])
            self.assertEqual(simd.read_lane_vgpr(lane, 0), SENTINEL)

    def test_single_lane_active(self) -> None:
        """EXEC mask enables exactly one lane (lane 17)."""
        target_lane = 17
        simd = SIMDState(num_lanes=64, exec_mask=(1 << target_lane))
        SENTINEL = 0xAAAA_BBBB

        for lane in range(64):
            simd.write_lane_vgpr(lane, 0, SENTINEL)
            simd.write_lane_vgpr(lane, 2, QUARTER_F32)

        instr = VRSQF32Instruction(
            dst=RegRef(0, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        for lane in range(64):
            if lane == target_lane:
                self.assertEqual(res[lane], TWO_F32)
                self.assertEqual(simd.read_lane_vgpr(lane, 0), TWO_F32)
            else:
                self.assertIsNone(res[lane])
                self.assertEqual(simd.read_lane_vgpr(lane, 0), SENTINEL)

    def test_zero_exec_mask_noop(self) -> None:
        """An EXEC mask of 0 is a complete no-op across all registers and lanes."""
        simd = SIMDState(num_lanes=64, exec_mask=0)
        SENTINEL = 0x5555_AAAA

        for lane in range(64):
            simd.write_lane_vgpr(lane, 0, SENTINEL)
            simd.write_lane_vgpr(lane, 2, FOUR_F32)

        instr = VRSQF32Instruction(
            dst=RegRef(0, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        self.assertTrue(all(r is None for r in res))
        for lane in range(64):
            self.assertEqual(simd.read_lane_vgpr(lane, 0), SENTINEL)

    def test_scalar_broadcast_to_active_simd_lanes(self) -> None:
        """Source operand from SGPR is broadcast identically across all active lanes."""
        simd = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        simd.write_sgpr(10, SIXTEEN_F32)

        instr = VRSQF32Instruction(
            dst=RegRef(4, RegType.VGPR),
            src0=VOP1Operand.from_sgpr(10),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        for lane in range(64):
            self.assertEqual(res[lane], QUARTER_F32)
            self.assertEqual(simd.read_lane_vgpr(lane, 4), QUARTER_F32)


# ============================================================================
# Section 5: Register Aliasing and In-Place Operand Preservation
# ============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_complete_in_place_aliasing_scalar(self) -> None:
        """In-place execution where destination and source are identical (v0 = rsq(v0))."""
        state = ScalarMachineState()
        state.write_vgpr(0, FOUR_F32)

        instr = VRSQF32Instruction(
            dst=RegRef(0, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(0),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, HALF_F32)
        self.assertEqual(state.read_vgpr(0), HALF_F32)

    def test_successive_in_place_evaluations(self) -> None:
        """Two successive in-place rsq evaluations: rsq(rsq(16.0)) == rsq(0.25) == 2.0."""
        state = ScalarMachineState()
        state.write_vgpr(4, SIXTEEN_F32)

        instr = VRSQF32Instruction(
            dst=RegRef(4, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(4),
        )
        interp = ScalarInterpreter(state)

        # Step 1: rsq(16.0) -> 0.25
        r1 = interp.execute(instr)
        self.assertEqual(r1, QUARTER_F32)
        self.assertEqual(state.read_vgpr(4), QUARTER_F32)

        # Step 2: rsq(0.25) -> 2.0
        r2 = interp.execute(instr)
        self.assertEqual(r2, TWO_F32)
        self.assertEqual(state.read_vgpr(4), TWO_F32)

    def test_in_place_simd_multi_lane(self) -> None:
        """Wavefront executing in-place v0 = rsq(v0) with lane divergence."""
        simd = SIMDState(num_lanes=64, exec_mask=0x5555_5555_5555_5555)  # even lanes
        for lane in range(64):
            simd.write_lane_vgpr(lane, 0, FOUR_F32)

        instr = VRSQF32Instruction(
            dst=RegRef(0, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(0),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(res[lane], HALF_F32)
                self.assertEqual(simd.read_lane_vgpr(lane, 0), HALF_F32)
            else:
                self.assertIsNone(res[lane])
                self.assertEqual(simd.read_lane_vgpr(lane, 0), FOUR_F32)

    def test_in_place_negative_to_nan(self) -> None:
        """In-place replacement of negative float by Canonical QNaN."""
        state = ScalarMachineState()
        state.write_vgpr(6, NEG_FOUR_F32)

        instr = VRSQF32Instruction(
            dst=RegRef(6, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(6),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, CANONICAL_QNAN_F32)
        self.assertEqual(state.read_vgpr(6), CANONICAL_QNAN_F32)


# ============================================================================
# Section 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ============================================================================

@dataclass
class FPStatusFlags:
    """IEEE 754 / RDNA2 floating-point exception sticky flags."""
    invalid: bool = False       # IOC: Invalid operation (negative operand, sNaN)
    div_by_zero: bool = False   # DZC: Division by zero (+0.0, -0.0 input)
    overflow: bool = False      # OFC: Overflow
    underflow: bool = False     # UFC: Underflow
    inexact: bool = False       # IXC: Inexact transcendental/rounding result

    def accumulate(self, other: FPStatusFlags) -> None:
        """Accumulate (bitwise OR) flags into hardware sticky status register."""
        self.invalid |= other.invalid
        self.div_by_zero |= other.div_by_zero
        self.overflow |= other.overflow
        self.underflow |= other.underflow
        self.inexact |= other.inexact

    def to_bits(self) -> int:
        """Encode flags into hardware status register bitmask."""
        bits = 0
        if self.invalid:
            bits |= 1 << 0
        if self.div_by_zero:
            bits |= 1 << 1
        if self.overflow:
            bits |= 1 << 2
        if self.underflow:
            bits |= 1 << 3
        if self.inexact:
            bits |= 1 << 4
        return bits


def compute_v_rsq_f32_flags(
    src_bits: int,
    abs_src: bool = False,
    neg_src: bool = False,
    clamp: bool = False,
    flush_subnormals: bool = False,
) -> Tuple[int, FPStatusFlags]:
    """Compute v_rsq_f32 result along with its architectural FP exception flags."""
    flags = FPStatusFlags()
    s = apply_f32_modifiers(src_bits, abs_src, neg_src)

    # Subnormal input under FTZ
    if flush_subnormals and is_subnormal_f32(s):
        flags.underflow = True
        s = NEG_ZERO_F32 if (s & SIGN_MASK_F32) else POS_ZERO_F32

    # Signaling NaN input raises Invalid
    if is_snan_f32(s):
        flags.invalid = True
        res = POS_ZERO_F32 if clamp else (s | QNAN_BIT_F32)
        return res, flags

    # Quiet NaN input propagates without Invalid
    if is_nan_f32(s):
        res = POS_ZERO_F32 if clamp else (s | QNAN_BIT_F32)
        return res, flags

    # Negative non-zero and -Infinity input raises Invalid
    if (s & SIGN_MASK_F32) and not is_zero_f32(s):
        flags.invalid = True
        res = POS_ZERO_F32 if clamp else CANONICAL_QNAN_F32
        return res, flags

    # Zero input raises Division-by-Zero
    if is_zero_f32(s):
        flags.div_by_zero = True
        res = ONE_F32 if clamp else POS_INF_F32
        return res, flags

    # Positive Infinity
    if s == POS_INF_F32:
        return POS_ZERO_F32, flags

    # Positive finite numbers
    val = bits_to_f32(s)
    res_val = 1.0 / math.sqrt(val)

    # Exact powers of 4
    is_exact = False
    if val > 0.0:
        log2_val = math.log2(val)
        if log2_val.is_integer() and (int(log2_val) % 2 == 0):
            is_exact = True

    if not is_exact:
        flags.inexact = True

    if clamp:
        if res_val < 0.0:
            res_val = 0.0
        elif res_val > 1.0:
            res_val = 1.0

    res_bits = f32_to_bits(res_val)
    if is_subnormal_f32(res_bits):
        flags.underflow = True

    return res_bits, flags


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for condition codes, IEEE 754 status flags, and VCC condition masks."""

    def test_invalid_flag_negative_inputs(self) -> None:
        """Negative operands raise the Invalid Operation (IOC) status flag."""
        _, flags_neg4 = compute_v_rsq_f32_flags(NEG_FOUR_F32)
        self.assertTrue(flags_neg4.invalid)
        self.assertFalse(flags_neg4.div_by_zero)

        _, flags_neginf = compute_v_rsq_f32_flags(NEG_INF_F32)
        self.assertTrue(flags_neginf.invalid)

    def test_invalid_flag_signaling_nan(self) -> None:
        """Signaling NaN operand raises the Invalid Operation (IOC) status flag."""
        snan = POS_INF_F32 | 0x0001_0000
        _, flags = compute_v_rsq_f32_flags(snan)
        self.assertTrue(flags.invalid)

    def test_quiet_nan_does_not_raise_invalid(self) -> None:
        """Quiet NaN operand propagates without asserting Invalid Operation."""
        _, flags = compute_v_rsq_f32_flags(CANONICAL_QNAN_F32)
        self.assertFalse(flags.invalid)

    def test_div_by_zero_flag_zeros(self) -> None:
        """Both +0.0 and -0.0 raise Division by Zero (DZC)."""
        _, flags_pos0 = compute_v_rsq_f32_flags(POS_ZERO_F32)
        self.assertTrue(flags_pos0.div_by_zero)
        self.assertFalse(flags_pos0.invalid)

        _, flags_neg0 = compute_v_rsq_f32_flags(NEG_ZERO_F32)
        self.assertTrue(flags_neg0.div_by_zero)

    def test_inexact_flag_non_perfect_squares(self) -> None:
        """Non-exact reciprocal square roots (2.0, 3.0, 10.0) assert Inexact (IXC)."""
        _, flags_two = compute_v_rsq_f32_flags(TWO_F32)
        self.assertTrue(flags_two.inexact)

        _, flags_three = compute_v_rsq_f32_flags(THREE_F32)
        self.assertTrue(flags_three.inexact)

    def test_exact_roots_do_not_raise_inexact(self) -> None:
        """Exact roots of powers of 4 (1.0, 4.0, 16.0, 0.25, 0.0625) do not assert Inexact."""
        _, flags_1 = compute_v_rsq_f32_flags(ONE_F32)
        self.assertFalse(flags_1.inexact)

        _, flags_4 = compute_v_rsq_f32_flags(FOUR_F32)
        self.assertFalse(flags_4.inexact)

        _, flags_16 = compute_v_rsq_f32_flags(SIXTEEN_F32)
        self.assertFalse(flags_16.inexact)

        _, flags_quarter = compute_v_rsq_f32_flags(QUARTER_F32)
        self.assertFalse(flags_quarter.inexact)

        _, flags_sixteenth = compute_v_rsq_f32_flags(SIXTEENTH_F32)
        self.assertFalse(flags_sixteenth.inexact)

    def test_cumulative_sticky_flags_accumulation(self) -> None:
        """Verify that sticky status register correctly accumulates flags across instructions."""
        sticky = FPStatusFlags()

        # Step 1: exact operation -> no flags
        _, f1 = compute_v_rsq_f32_flags(FOUR_F32)
        sticky.accumulate(f1)
        self.assertEqual(sticky.to_bits(), 0)

        # Step 2: division by zero -> div_by_zero flag (bit 1)
        _, f2 = compute_v_rsq_f32_flags(POS_ZERO_F32)
        sticky.accumulate(f2)
        self.assertEqual(sticky.to_bits(), 1 << 1)

        # Step 3: negative operand -> invalid flag (bit 0)
        _, f3 = compute_v_rsq_f32_flags(NEG_FOUR_F32)
        sticky.accumulate(f3)
        self.assertEqual(sticky.to_bits(), (1 << 0) | (1 << 1))

        # Step 4: inexact operation -> inexact flag (bit 4)
        _, f4 = compute_v_rsq_f32_flags(TWO_F32)
        sticky.accumulate(f4)
        self.assertEqual(sticky.to_bits(), (1 << 0) | (1 << 1) | (1 << 4))

    def test_vcc_condition_masks(self) -> None:
        """Compute condition mask across SIMD wave lanes (emulating V_CMP / VCC)."""
        simd = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            if lane % 4 == 0:
                simd.write_lane_vgpr(lane, 0, POS_ZERO_F32)  # -> +inf
            elif lane % 4 == 1:
                simd.write_lane_vgpr(lane, 0, POS_INF_F32)   # -> +0.0
            elif lane % 4 == 2:
                simd.write_lane_vgpr(lane, 0, NEG_FOUR_F32)  # -> NaN
            else:
                simd.write_lane_vgpr(lane, 0, FOUR_F32)  # -> 0.5 (positive finite)

        instr = VRSQF32Instruction(
            dst=RegRef(2, RegType.VGPR),
            src0=VOP1Operand.from_vgpr(0),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        vcc_zero = 0
        vcc_inf = 0
        vcc_nan = 0
        vcc_finite_pos = 0

        for lane, r in enumerate(res):
            assert r is not None
            if is_zero_f32(r):
                vcc_zero |= (1 << lane)
            elif is_inf_f32(r):
                vcc_inf |= (1 << lane)
            elif is_nan_f32(r):
                vcc_nan |= (1 << lane)
            else:
                vcc_finite_pos |= (1 << lane)

        self.assertEqual(vcc_zero | vcc_inf | vcc_nan | vcc_finite_pos, 0xFFFF_FFFF_FFFF_FFFF)
        self.assertEqual(vcc_zero & vcc_inf, 0)
        self.assertEqual(vcc_zero & vcc_nan, 0)
        self.assertEqual(vcc_inf & vcc_nan, 0)


# ============================================================================
# Section 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ============================================================================

class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_flt_min_normal_boundary(self) -> None:
        """rsq(2^-126) = 2^63, exact power of 2 representation in binary32."""
        res_bits = oracle_v_rsq_f32(FLT_MIN)
        self.assertFalse(is_nan_f32(res_bits))
        self.assertFalse(is_inf_f32(res_bits))
        expected_val = 2.0 ** 63
        self.assertAlmostEqual(bits_to_f32(res_bits), expected_val, delta=expected_val * 1e-6)

    def test_flt_max_normal_boundary(self) -> None:
        """rsq(FLT_MAX) produces small finite positive normal float."""
        res_bits = oracle_v_rsq_f32(FLT_MAX)
        self.assertFalse(is_nan_f32(res_bits))
        self.assertFalse(is_inf_f32(res_bits))
        res_val = bits_to_f32(res_bits)
        self.assertGreater(res_val, 0.0)
        self.assertLess(res_val, 1e-18)

    def test_subnorm_min_gradual_underflow(self) -> None:
        """rsq(2^-149) computes finite float under IEEE-754 gradual underflow."""
        res_bits = oracle_v_rsq_f32(SUBNORM_MIN_F32, flush_subnormals=False)
        self.assertFalse(is_nan_f32(res_bits))
        self.assertFalse(is_inf_f32(res_bits))
        self.assertGreater(bits_to_f32(res_bits), 2.0 ** 63)

    def test_subnorm_max_gradual_underflow(self) -> None:
        """Largest positive subnormal computes without NaN under IEEE mode."""
        res_bits = oracle_v_rsq_f32(SUBNORM_MAX_F32, flush_subnormals=False)
        self.assertFalse(is_nan_f32(res_bits))
        self.assertFalse(is_inf_f32(res_bits))
        self.assertGreater(bits_to_f32(res_bits), 2.0 ** 63)

    def test_subnormal_flush_to_zero_mode(self) -> None:
        """Under FTZ mode, both smallest and largest subnormals flush to 0, producing +Inf."""
        res_min_ftz = oracle_v_rsq_f32(SUBNORM_MIN_F32, flush_subnormals=True)
        self.assertEqual(res_min_ftz, POS_INF_F32)

        res_max_ftz = oracle_v_rsq_f32(SUBNORM_MAX_F32, flush_subnormals=True)
        self.assertEqual(res_max_ftz, POS_INF_F32)

    def test_qnan_payload_preservation(self) -> None:
        """Quiet NaN payloads are preserved and have the quiet bit asserted."""
        payload_a = 0x7FC0_1234
        res_a = oracle_v_rsq_f32(payload_a)
        self.assertEqual(res_a, payload_a)

        payload_b = 0x7FE0_5678
        res_b = oracle_v_rsq_f32(payload_b)
        self.assertEqual(res_b, payload_b)

    def test_snan_quieting(self) -> None:
        """Signaling NaN is converted to a Quiet NaN with quiet bit set."""
        snan = 0x7F80_0001
        res = oracle_v_rsq_f32(snan)
        self.assertTrue(is_qnan_f32(res))
        self.assertEqual(res, snan | QNAN_BIT_F32)

    def test_negative_subnormals_yield_qnan(self) -> None:
        """Negative subnormals represent negative numbers and yield Canonical QNaN."""
        neg_sub_min = SUBNORM_MIN_F32 | SIGN_MASK_F32
        neg_sub_max = SUBNORM_MAX_F32 | SIGN_MASK_F32
        self.assertEqual(oracle_v_rsq_f32(neg_sub_min), CANONICAL_QNAN_F32)
        self.assertEqual(oracle_v_rsq_f32(neg_sub_max), CANONICAL_QNAN_F32)

    def test_monotonic_decrease_across_normals(self) -> None:
        """Reciprocal square root is strictly monotonically decreasing for positive inputs."""
        inputs = [
            0.0001, 0.01, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0, 100.0, 10000.0, 1e6, 1e12,
        ]
        prev_res = math.inf
        for inp in inputs:
            curr_res = bits_to_f32(oracle_v_rsq_f32(f32_to_bits(inp)))
            self.assertLess(curr_res, prev_res)
            prev_res = curr_res

    def test_clamp_saturation_at_boundaries(self) -> None:
        """Clamp modifier saturates extreme values to [0.0, 1.0]."""
        # SUBNORM_MIN_F32 produces >> 1.0 -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f32(SUBNORM_MIN_F32, clamp=True), ONE_F32)
        # FLT_MIN produces 2^63 >> 1.0 -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f32(FLT_MIN, clamp=True), ONE_F32)
        # FLT_MAX produces ~5.4e-20 -> within [0.0, 1.0] -> unchanged
        self.assertEqual(oracle_v_rsq_f32(FLT_MAX, clamp=True), oracle_v_rsq_f32(FLT_MAX, clamp=False))
        # +0.0 produces +inf -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f32(POS_ZERO_F32, clamp=True), ONE_F32)
        # -0.0 produces +inf -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f32(NEG_ZERO_F32, clamp=True), ONE_F32)


if __name__ == "__main__":
    unittest.main()
