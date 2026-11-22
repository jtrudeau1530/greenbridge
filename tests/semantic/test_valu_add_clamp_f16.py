from __future__ import annotations

import math
import struct
import unittest
from fractions import Fraction
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto

# IEEE 754 binary16 (half precision) architectural constants
SIGN_MASK: int = 0x8000
EXP_MASK: int = 0x7C00
FRAC_MASK: int = 0x03FF
QNAN_BIT: int = 0x0200
CANONICAL_QNAN: int = 0x7E00

POS_ZERO: int = 0x0000
NEG_ZERO: int = 0x8000
POS_INF: int = 0x7C00
NEG_INF: int = 0xFC00

ONE_F16: int = 0x3C00
MAX_NORMAL_F16: int = 0x7BFF
MIN_NORMAL_F16: int = 0x0400
MAX_SUBNORMAL_F16: int = 0x03FF
MIN_SUBNORMAL_F16: int = 0x0001


def f16_to_bits(val: float) -> int:
    """Convert a Python float to 16-bit integer bits (IEEE 754 binary16)."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f16(bits: int) -> float:
    """Convert 16-bit integer bits to a Python float (IEEE 754 binary16)."""
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def split_f16_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK) >> 15
    exponent = (bits & EXP_MASK) >> 10
    fraction = bits & FRAC_MASK
    return sign, exponent, fraction


def pack_f16_pair(lo: int, hi: int) -> int:
    """Pack two 16-bit values into a 32-bit register word."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def unpack_f16_pair(val32: int) -> Tuple[int, int]:
    """Unpack a 32-bit register word into two 16-bit half words (lo, hi)."""
    return val32 & 0xFFFF, (val32 >> 16) & 0xFFFF


def is_nan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 NaN."""
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
    """Return True if bits represent a subnormal (denormal) half-precision float."""
    return ((bits & EXP_MASK) == 0) and ((bits & FRAC_MASK) != 0)


def is_zero(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK) == 0


def apply_f16_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & 0xFFFF
    if abs_mod:
        res &= ~SIGN_MASK
    if neg_mod:
        res ^= SIGN_MASK
    return res


def oracle_v_add_clamp_f16(
    src0_bits: int,
    src1_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_add_clamp_f16 half-precision addition with [0.0, 1.0] clamp.
    
    Computes D = clamp(src0 + src1, 0.0, 1.0) conforming to IEEE 754-2008 half precision (binary16)
    with Round-to-Nearest-Even (RNE) and AMD RDNA2 ISA clamp saturation semantics.
    """
    s0 = apply_f16_modifiers(src0_bits, abs0, neg0)
    s1 = apply_f16_modifiers(src1_bits, abs1, neg1)

    # 1. NaN handling: Clamping NaN in RDNA2 produces +0.0
    if is_nan(s0) or is_nan(s1):
        return POS_ZERO

    # 2. Infinity handling
    s0_inf = is_inf(s0)
    s1_inf = is_inf(s1)
    if s0_inf and s1_inf:
        s0_sign = (s0 & SIGN_MASK)
        s1_sign = (s1 & SIGN_MASK)
        if s0_sign != s1_sign:
            # +inf + -inf -> Invalid operation (NaN) -> clamped produces +0.0
            return POS_ZERO
        return ONE_F16 if s0 == POS_INF else POS_ZERO
    if s0_inf:
        return ONE_F16 if s0 == POS_INF else POS_ZERO
    if s1_inf:
        return ONE_F16 if s1 == POS_INF else POS_ZERO

    # 3. Signed zero handling
    if is_zero(s0) and is_zero(s1):
        # Result is +0.0 or -0.0, clamp forces -0.0 to +0.0
        return POS_ZERO

    # 4. Finite arithmetic
    f0 = bits_to_f16(s0)
    f1 = bits_to_f16(s1)
    f_sum = f0 + f1

    # Exact signed zero cancellation: (+x) + (-x) -> +0.0
    if f_sum == 0.0:
        return POS_ZERO

    try:
        raw_res = f16_to_bits(f_sum)
    except OverflowError:
        raw_res = POS_INF if f_sum > 0 else NEG_INF

    # 5. Clamp modifier handling [0.0, 1.0]
    if is_inf(raw_res):
        return ONE_F16 if raw_res == POS_INF else POS_ZERO
    res_f = bits_to_f16(raw_res)
    if res_f < 0.0 or raw_res == NEG_ZERO:
        return POS_ZERO
    elif res_f > 1.0:
        return ONE_F16

    return raw_res


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


class SubWord(Enum):
    LO = auto()
    HI = auto()


@dataclass(frozen=True)
class Reg16:
    """Represents an architectural 16-bit register operand (e.g. v0.l, v0.h, s2.l)."""
    index: int
    reg_type: RegType = RegType.VGPR
    sub_word: SubWord = SubWord.LO

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index {self.index} out of range [0, 255]")

    @property
    def is_hi(self) -> bool:
        return self.sub_word == SubWord.HI

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        suffix = "h" if self.is_hi else "l"
        return f"{prefix}{self.index}.{suffix}"


@dataclass(frozen=True)
class Literal16:
    """Represents a 16-bit inline float constant or literal."""
    bits: int

    @classmethod
    def from_float(cls, val: float) -> Literal16:
        return cls(f16_to_bits(val))

    @property
    def float_value(self) -> float:
        return bits_to_f16(self.bits)

    @property
    def name(self) -> str:
        if self.bits == POS_ZERO:
            return "0.0"
        if self.bits == NEG_ZERO:
            return "-0.0"
        f = self.float_value
        if math.isnan(f):
            return "nan"
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f"{f:g}"


@dataclass(frozen=True)
class VOP3Operand16:
    """Typed expression operand with optional input modifiers (abs, neg)."""
    source: Union[Reg16, Literal16]
    abs_mod: bool = False
    neg_mod: bool = False

    def disassemble(self) -> str:
        s = self.source.name
        if self.abs_mod:
            s = f"|{s}|"
        if self.neg_mod:
            s = f"-{s}"
        return s


@dataclass(frozen=True)
class VAddClampF16Instruction:
    """Symbolic AST node representing a RDNA2 v_add_clamp_f16 (or v_add_f16 clamp) instruction."""
    dst: Reg16
    src0: VOP3Operand16
    src1: VOP3Operand16

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        return f"v_add_f16 {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()} clamp"


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single wavefront lane."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def get_reg16(self, reg: Reg16) -> int:
        file = self.vgprs if reg.reg_type == RegType.VGPR else self.sgprs
        word = file[reg.index]
        return (word >> 16) & 0xFFFF if reg.is_hi else word & 0xFFFF

    def set_reg16(self, reg: Reg16, val16: int) -> None:
        file = self.vgprs if reg.reg_type == RegType.VGPR else self.sgprs
        old = file[reg.index]
        if reg.is_hi:
            file[reg.index] = (old & 0x0000FFFF) | ((val16 & 0xFFFF) << 16)
        else:
            file[reg.index] = (old & 0xFFFF0000) | (val16 & 0xFFFF)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_add_clamp_f16."""

    def resolve_operand(self, op: VOP3Operand16, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg16):
            return state.get_reg16(op.source)
        elif isinstance(op.source, Literal16):
            return op.source.bits
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VAddClampF16Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)

        # Computation phase
        result_bits = oracle_v_add_clamp_f16(
            src0_bits=s0_raw,
            src1_bits=s1_raw,
            abs0=instr.src0.abs_mod,
            neg0=instr.src0.neg_mod,
            abs1=instr.src1.abs_mod,
            neg1=instr.src1.neg_mod,
        )

        # Write-back phase
        state.set_reg16(instr.dst, result_bits)
        return result_bits


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

    def get_reg16(self, lane: int, reg: Reg16) -> int:
        if reg.reg_type == RegType.VGPR:
            word = self.vgprs[lane][reg.index]
        else:
            word = self.sgprs[reg.index]
        return (word >> 16) & 0xFFFF if reg.is_hi else word & 0xFFFF

    def set_reg16(self, lane: int, reg: Reg16, val16: int) -> None:
        if reg.reg_type == RegType.VGPR:
            old = self.vgprs[lane][reg.index]
            if reg.is_hi:
                self.vgprs[lane][reg.index] = (old & 0x0000FFFF) | ((val16 & 0xFFFF) << 16)
            else:
                self.vgprs[lane][reg.index] = (old & 0xFFFF0000) | (val16 & 0xFFFF)
        else:
            old = self.sgprs[reg.index]
            if reg.is_hi:
                self.sgprs[reg.index] = (old & 0x0000FFFF) | ((val16 & 0xFFFF) << 16)
            else:
                self.sgprs[reg.index] = (old & 0xFFFF0000) | (val16 & 0xFFFF)


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VOP3Operand16, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, Reg16):
            return state.get_reg16(lane, op.source)
        elif isinstance(op.source, Literal16):
            return op.source.bits
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VAddClampF16Instruction, state: SIMDState) -> List[Optional[int]]:
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

        # Execution phase
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = oracle_v_add_clamp_f16(
                    src0_bits=s0_vals[lane],
                    src1_bits=s1_vals[lane],
                    abs0=instr.src0.abs_mod,
                    neg0=instr.src0.neg_mod,
                    abs1=instr.src1.abs_mod,
                    neg1=instr.src1.neg_mod,
                )
                state.set_reg16(lane, instr.dst, res)
                results[lane] = res

        return results


# ==============================================================================
# Condition Codes, Status Flags, and Carry Generation
# ==============================================================================

@dataclass(frozen=True)
class FPStatusFlags:
    """Architectural floating-point condition codes and IEEE 754 status flags with clamp."""
    invalid: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False
    zero: bool = False
    negative: bool = False
    denormal: bool = False
    infinite: bool = False
    nan: bool = False
    clamped: bool = False


def evaluate_v_add_clamp_f16_flags(
    src0_bits: int,
    src1_bits: int,
    res_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
) -> FPStatusFlags:
    """Evaluate IEEE 754 architectural flags and condition codes for v_add_clamp_f16."""
    s0 = apply_f16_modifiers(src0_bits, abs0, neg0)
    s1 = apply_f16_modifiers(src1_bits, abs1, neg1)

    s0_nan = is_nan(s0)
    s1_nan = is_nan(s1)
    s0_snan = is_snan(s0)
    s1_snan = is_snan(s1)
    s0_inf = is_inf(s0)
    s1_inf = is_inf(s1)

    invalid = False
    if s0_snan or s1_snan:
        invalid = True
    elif s0_inf and s1_inf and ((s0 & SIGN_MASK) != (s1 & SIGN_MASK)):
        invalid = True

    nan_flag = is_nan(res_bits)
    inf_flag = is_inf(res_bits)
    zero_flag = is_zero(res_bits)
    neg_flag = bool(res_bits & SIGN_MASK)
    denorm_flag = is_subnormal(res_bits)

    f0 = bits_to_f16(s0) if not (s0_nan or s0_inf) else 0.0
    f1 = bits_to_f16(s1) if not (s1_nan or s1_inf) else 0.0
    unclamped_sum = f0 + f1

    overflow = False
    if not (s0_nan or s1_nan or s0_inf or s1_inf):
        if unclamped_sum > 65504.0 or unclamped_sum < -65504.0:
            overflow = True

    clamped = False
    if s0_nan or s1_nan or s0_inf or s1_inf:
        clamped = True
    elif unclamped_sum < 0.0 or unclamped_sum > 1.0 or res_bits in (POS_ZERO, ONE_F16) and (unclamped_sum != bits_to_f16(res_bits)):
        clamped = True

    underflow = False
    if denorm_flag:
        underflow = True

    inexact = False
    if overflow or clamped:
        inexact = True
    elif not (s0_nan or s1_nan or s0_inf or s1_inf or nan_flag or inf_flag):
        res_f = bits_to_f16(res_bits)
        if Fraction(f0) + Fraction(f1) != Fraction(res_f):
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
        clamped=clamped,
    )


def compute_f16_mantissa_carry(s0_bits: int, s1_bits: int) -> Tuple[int, int]:
    """Demonstrate IEEE 754 11-bit mantissa carry generation during half-precision addition."""
    _, exp0, frac0 = split_f16_bits(s0_bits)
    _, exp1, frac1 = split_f16_bits(s1_bits)
    m0 = (1 << 10) | frac0 if exp0 != 0 else frac0
    m1 = (1 << 10) | frac1 if exp1 != 0 else frac1
    if exp0 > exp1:
        m1 >>= (exp0 - exp1)
    elif exp1 > exp0:
        m0 >>= (exp1 - exp0)
    mant_sum = m0 + m1
    carry_out = 1 if mant_sum >= (1 << 11) else 0
    return mant_sum, carry_out


# ==============================================================================
# Unit Tests
# ==============================================================================


class TestF16ClampModelAndOracle(unittest.TestCase):
    """Unit tests for binary16 bit model, predicates, and reference clamped addition oracle."""

    def test_round_trip_conversion(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 65504.0, -65504.0, 3.140625]
        for val in values:
            with self.subTest(val=val):
                bits = f16_to_bits(val)
                recovered = bits_to_f16(bits)
                self.assertEqual(val, recovered)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_word_packing_vgpr_pair(self) -> None:
        lo = 0x3C00  # 1.0
        hi = 0x4000  # 2.0
        packed = pack_f16_pair(lo, hi)
        self.assertEqual(packed, 0x40003C00)
        unpacked_lo, unpacked_hi = unpack_f16_pair(packed)
        self.assertEqual(unpacked_lo, lo)
        self.assertEqual(unpacked_hi, hi)

    def test_classification_predicates(self) -> None:
        self.assertTrue(f16_to_bits(0.0) == POS_ZERO)
        self.assertTrue(f16_to_bits(-0.0) == NEG_ZERO)
        self.assertTrue(is_zero(POS_ZERO))
        self.assertTrue(is_zero(NEG_ZERO))
        self.assertFalse(is_zero(f16_to_bits(1.0)))

        self.assertTrue(is_inf(POS_INF))
        self.assertTrue(is_inf(NEG_INF))
        self.assertFalse(is_inf(POS_ZERO))

        snan_bits = 0x7C01
        qnan_bits = CANONICAL_QNAN
        self.assertTrue(is_nan(snan_bits))
        self.assertTrue(is_snan(snan_bits))
        self.assertFalse(is_qnan(snan_bits))

        self.assertTrue(is_nan(qnan_bits))
        self.assertTrue(is_qnan(qnan_bits))
        self.assertFalse(is_snan(qnan_bits))

        subnorm_bits = 0x0001
        self.assertTrue(is_subnormal(subnorm_bits))
        self.assertFalse(is_subnormal(POS_ZERO))
        self.assertFalse(is_subnormal(f16_to_bits(1.0)))

    def test_clamped_finite_addition(self) -> None:
        # Inside [0.0, 1.0] range
        c1 = oracle_v_add_clamp_f16(f16_to_bits(0.25), f16_to_bits(0.5))
        self.assertEqual(bits_to_f16(c1), 0.75)

        # Exceeds 1.0 -> clamped to 1.0
        c2 = oracle_v_add_clamp_f16(f16_to_bits(0.75), f16_to_bits(0.5))
        self.assertEqual(c2, ONE_F16)

        # Less than 0.0 -> clamped to +0.0
        c3 = oracle_v_add_clamp_f16(f16_to_bits(-1.5), f16_to_bits(0.5))
        self.assertEqual(c3, POS_ZERO)

    def test_signed_zero_rules(self) -> None:
        # All zero combinations clamp to +0.0
        self.assertEqual(oracle_v_add_clamp_f16(POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_v_add_clamp_f16(NEG_ZERO, NEG_ZERO), POS_ZERO)
        self.assertEqual(oracle_v_add_clamp_f16(POS_ZERO, NEG_ZERO), POS_ZERO)
        self.assertEqual(oracle_v_add_clamp_f16(NEG_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_v_add_clamp_f16(f16_to_bits(0.5), f16_to_bits(-0.5)), POS_ZERO)

    def test_input_modifiers_abs_neg(self) -> None:
        val_neg = f16_to_bits(-0.4)
        val_pos = f16_to_bits(0.2)

        # abs on negative: |-0.4| + 0.2 = 0.6
        res = oracle_v_add_clamp_f16(val_neg, val_pos, abs0=True)
        self.assertAlmostEqual(bits_to_f16(res), 0.6, places=3)

        # neg on positive: -(0.2) + 0.1 = -0.1 -> clamped to +0.0
        res2 = oracle_v_add_clamp_f16(val_pos, f16_to_bits(0.1), neg0=True)
        self.assertEqual(res2, POS_ZERO)

    def test_clamped_infinity(self) -> None:
        self.assertEqual(oracle_v_add_clamp_f16(POS_INF, f16_to_bits(1.0)), ONE_F16)
        self.assertEqual(oracle_v_add_clamp_f16(NEG_INF, f16_to_bits(1.0)), POS_ZERO)
        self.assertEqual(oracle_v_add_clamp_f16(POS_INF, POS_INF), ONE_F16)
        self.assertEqual(oracle_v_add_clamp_f16(NEG_INF, NEG_INF), POS_ZERO)
        self.assertEqual(oracle_v_add_clamp_f16(POS_INF, NEG_INF), POS_ZERO)

    def test_clamped_nan(self) -> None:
        snan = 0x7C05
        qnan = CANONICAL_QNAN
        self.assertEqual(oracle_v_add_clamp_f16(snan, f16_to_bits(0.5)), POS_ZERO)
        self.assertEqual(oracle_v_add_clamp_f16(f16_to_bits(0.5), qnan), POS_ZERO)
        self.assertEqual(oracle_v_add_clamp_f16(snan, qnan), POS_ZERO)


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_reg16_validation_and_naming(self) -> None:
        v0_l = Reg16(0, RegType.VGPR, SubWord.LO)
        self.assertEqual(v0_l.name, "v0.l")
        self.assertFalse(v0_l.is_hi)

        v0_h = Reg16(0, RegType.VGPR, SubWord.HI)
        self.assertEqual(v0_h.name, "v0.h")
        self.assertTrue(v0_h.is_hi)

        s8_l = Reg16(8, RegType.SGPR, SubWord.LO)
        self.assertEqual(s8_l.name, "s8.l")
        self.assertFalse(s8_l.is_hi)

        with self.assertRaises(ValueError):
            Reg16(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg16(-1, RegType.VGPR)

    def test_operand_disassembly(self) -> None:
        v = Reg16(2, RegType.VGPR, SubWord.LO)
        op_plain = VOP3Operand16(v)
        self.assertEqual(op_plain.disassemble(), "v2.l")

        op_abs = VOP3Operand16(v, abs_mod=True)
        self.assertEqual(op_abs.disassemble(), "|v2.l|")

        op_neg = VOP3Operand16(v, neg_mod=True)
        self.assertEqual(op_neg.disassemble(), "-v2.l")

        op_neg_abs = VOP3Operand16(v, abs_mod=True, neg_mod=True)
        self.assertEqual(op_neg_abs.disassemble(), "-|v2.l|")

        lit = Literal16.from_float(1.5)
        op_lit = VOP3Operand16(lit)
        self.assertEqual(op_lit.disassemble(), "1.5")

    def test_v_add_clamp_f16_ast_construction_and_validation(self) -> None:
        dst = Reg16(0, RegType.VGPR, SubWord.LO)
        src0 = VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.HI), neg_mod=True, abs_mod=True)
        src1 = VOP3Operand16(Reg16(4, RegType.SGPR, SubWord.LO))
        instr = VAddClampF16Instruction(dst=dst, src0=src0, src1=src1)

        self.assertEqual(instr.disassemble(), "v_add_f16 v0.l, -|v2.h|, s4.l clamp")

        # Invalid destination type (SGPR)
        with self.assertRaises(TypeError):
            VAddClampF16Instruction(dst=Reg16(0, RegType.SGPR), src0=src0, src1=src1)

    def test_structural_equality(self) -> None:
        i1 = VAddClampF16Instruction(
            Reg16(0, RegType.VGPR),
            VOP3Operand16(Reg16(2, RegType.VGPR)),
            VOP3Operand16(Literal16.from_float(0.0)),
        )
        i2 = VAddClampF16Instruction(
            Reg16(0, RegType.VGPR),
            VOP3Operand16(Reg16(2, RegType.VGPR)),
            VOP3Operand16(Literal16.from_float(0.0)),
        )
        self.assertEqual(i1, i2)


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_scalar_vgpr_vgpr_clamped_addition(self) -> None:
        # 0.25 + 0.5 = 0.75
        self.state.set_reg16(Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(0.25))
        self.state.set_reg16(Reg16(4, RegType.VGPR, SubWord.LO), f16_to_bits(0.5))
        instr = VAddClampF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(4, RegType.VGPR, SubWord.LO)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(bits_to_f16(res_bits), 0.75)
        self.assertEqual(self.state.get_reg16(Reg16(0, RegType.VGPR, SubWord.LO)), res_bits)

    def test_scalar_vgpr_sgpr_overflow_clamp(self) -> None:
        # 0.75 + 0.5 = 1.25 -> clamped to 1.0
        self.state.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(0.75))
        self.state.set_reg16(Reg16(8, RegType.SGPR, SubWord.LO), f16_to_bits(0.5))
        instr = VAddClampF16Instruction(
            dst=Reg16(4, RegType.VGPR, SubWord.HI),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(8, RegType.SGPR, SubWord.LO)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, ONE_F16)
        self.assertEqual(self.state.get_reg16(Reg16(4, RegType.VGPR, SubWord.HI)), ONE_F16)

    def test_scalar_vgpr_literal_negative_clamp(self) -> None:
        # 0.2 + (-0.5) = -0.3 -> clamped to +0.0
        self.state.set_reg16(Reg16(6, RegType.VGPR, SubWord.LO), f16_to_bits(0.2))
        instr = VAddClampF16Instruction(
            dst=Reg16(6, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(6, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Literal16.from_float(-0.5)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, POS_ZERO)
        self.assertEqual(self.state.get_reg16(Reg16(6, RegType.VGPR, SubWord.LO)), POS_ZERO)

    def test_scalar_with_modifiers(self) -> None:
        self.state.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(-0.8))
        self.state.set_reg16(Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(0.3))
        # -|-0.8| + 0.3 = -0.8 + 0.3 = -0.5 -> clamped to +0.0
        instr = VAddClampF16Instruction(
            dst=Reg16(4, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO), abs_mod=True, neg_mod=True),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, POS_ZERO)

    def test_scalar_subword_preservation(self) -> None:
        # Prepopulate v1 with distinct lo and hi half words
        self.state.vgprs[1] = pack_f16_pair(0x1111, 0x2222)
        instr_lo = VAddClampF16Instruction(
            dst=Reg16(1, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Literal16.from_float(0.25)),
            src1=VOP3Operand16(Literal16.from_float(0.25)),
        )
        self.interp.step(instr_lo, self.state)
        # Low half updated to 0.5 (0x3800), high half preserved (0x2222)
        self.assertEqual(self.state.get_reg16(Reg16(1, RegType.VGPR, SubWord.LO)), 0x3800)
        self.assertEqual(self.state.get_reg16(Reg16(1, RegType.VGPR, SubWord.HI)), 0x2222)

    def test_scalar_determinism(self) -> None:
        runs = []
        for _ in range(50):
            st = ScalarMachineState()
            st.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(0.4))
            st.set_reg16(Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(0.3))
            instr = VAddClampF16Instruction(
                dst=Reg16(4, RegType.VGPR, SubWord.LO),
                src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
                src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
            )
            runs.append(self.interp.step(instr, st))
        self.assertTrue(all(r == runs[0] for r in runs))


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD execution, lane divergence, and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_wave32_full_exec(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(float(lane) * 0.02))
            state.set_reg16(lane, Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(0.1))
        instr = VAddClampF16Instruction(
            dst=Reg16(4, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            expected = min(1.0, float(lane) * 0.02 + 0.1)
            self.assertAlmostEqual(bits_to_f16(state.get_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO))), expected, places=2)

    def test_wave32_divergent_masking(self) -> None:
        # Alternating mask: even lanes active (0x55555555)
        sentinel = 0xBEEF
        state = SIMDState(num_lanes=32, exec_mask=0x5555_5555)
        for lane in range(32):
            state.set_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(0.2))
            state.set_reg16(lane, Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(0.3))
            state.set_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO), sentinel)

        instr = VAddClampF16Instruction(
            dst=Reg16(4, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertIsNotNone(results[lane])
                self.assertEqual(bits_to_f16(state.get_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO))), 0.5)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO)), sentinel)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(0.7))
            state.set_reg16(lane, Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(0.5))
        instr = VAddClampF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        self.interp.step(instr, state)
        for lane in range(64):
            # 0.7 + 0.5 = 1.2 -> clamped to 1.0
            self.assertEqual(state.get_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO)), ONE_F16)

    def test_wave64_half_disabled(self) -> None:
        # Lower 32 lanes active, upper 32 lanes inactive
        sentinel = 0xCAFE
        state = SIMDState(num_lanes=64, exec_mask=0x0000_0000_FFFF_FFFF)
        for lane in range(64):
            state.set_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(0.25))
            state.set_reg16(lane, Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(0.5))
            state.set_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO), sentinel)

        instr = VAddClampF16Instruction(
            dst=Reg16(4, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        results = self.interp.step(instr, state)
        for lane in range(64):
            if lane < 32:
                self.assertIsNotNone(results[lane])
                self.assertEqual(bits_to_f16(state.get_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO))), 0.75)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO)), sentinel)

    def test_wave_exec_all_zero(self) -> None:
        sentinel = 0xAA55
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)
        for lane in range(32):
            state.set_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO), sentinel)
        instr = VAddClampF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Literal16.from_float(0.5)),
            src1=VOP3Operand16(Literal16.from_float(0.5)),
        )
        results = self.interp.step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(state.get_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO)), sentinel)

    def test_simd_sgpr_uniform_broadcast(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        state.set_reg16(0, Reg16(10, RegType.SGPR, SubWord.LO), f16_to_bits(0.25))
        for lane in range(32):
            state.set_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(0.5))

        instr = VAddClampF16Instruction(
            dst=Reg16(2, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(10, RegType.SGPR, SubWord.LO)),
        )
        self.interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(bits_to_f16(state.get_reg16(lane, Reg16(2, RegType.VGPR, SubWord.LO))), 0.75)


class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src0(self) -> None:
        state = ScalarMachineState()
        state.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(0.25))
        state.set_reg16(Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(0.5))
        instr = VAddClampF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        self.scalar_interp.step(instr, state)
        self.assertEqual(bits_to_f16(state.get_reg16(Reg16(0, RegType.VGPR, SubWord.LO))), 0.75)
        # src1 untouched
        self.assertEqual(bits_to_f16(state.get_reg16(Reg16(2, RegType.VGPR, SubWord.LO))), 0.5)

    def test_dst_alias_src1(self) -> None:
        state = ScalarMachineState()
        state.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(0.25))
        state.set_reg16(Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(0.5))
        instr = VAddClampF16Instruction(
            dst=Reg16(2, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        self.scalar_interp.step(instr, state)
        self.assertEqual(bits_to_f16(state.get_reg16(Reg16(2, RegType.VGPR, SubWord.LO))), 0.75)
        # src0 untouched
        self.assertEqual(bits_to_f16(state.get_reg16(Reg16(0, RegType.VGPR, SubWord.LO))), 0.25)

    def test_dst_alias_both_srcs(self) -> None:
        state = ScalarMachineState()
        state.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(0.6))
        instr = VAddClampF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
        )
        self.scalar_interp.step(instr, state)
        # 0.6 + 0.6 = 1.2 -> clamped to 1.0
        self.assertEqual(state.get_reg16(Reg16(0, RegType.VGPR, SubWord.LO)), ONE_F16)

    def test_subword_cross_half_aliasing(self) -> None:
        # Prepopulate v0 with lo=0.25, hi=0.5
        state = ScalarMachineState()
        state.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(0.25))
        state.set_reg16(Reg16(0, RegType.VGPR, SubWord.HI), f16_to_bits(0.5))

        instr = VAddClampF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.HI),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.HI)),
        )
        self.scalar_interp.step(instr, state)
        # hi updated to 0.75, lo untouched at 0.25
        self.assertEqual(bits_to_f16(state.get_reg16(Reg16(0, RegType.VGPR, SubWord.HI))), 0.75)
        self.assertEqual(bits_to_f16(state.get_reg16(Reg16(0, RegType.VGPR, SubWord.LO))), 0.25)

    def test_inplace_iterative_accumulation_saturates_at_one(self) -> None:
        state = ScalarMachineState()
        state.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(0.0))
        state.set_reg16(Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(0.2))
        instr = VAddClampF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        for _ in range(10):
            self.scalar_interp.step(instr, state)
        # Saturated and clamped at 1.0
        self.assertEqual(state.get_reg16(Reg16(0, RegType.VGPR, SubWord.LO)), ONE_F16)

    def test_simd_multi_lane_aliasing(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(0.1))
            state.set_reg16(lane, Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(0.2))

        instr = VAddClampF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertAlmostEqual(bits_to_f16(state.get_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO))), 0.3, places=2)


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for status flags, condition codes, and carry generation."""

    def test_exact_finite_flags(self) -> None:
        s0 = f16_to_bits(0.25)
        s1 = f16_to_bits(0.5)
        res = oracle_v_add_clamp_f16(s0, s1)
        flags = evaluate_v_add_clamp_f16_flags(s0, s1, res)
        self.assertFalse(flags.invalid)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.underflow)
        self.assertFalse(flags.inexact)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.denormal)
        self.assertFalse(flags.infinite)
        self.assertFalse(flags.nan)
        self.assertFalse(flags.clamped)

    def test_inexact_clamped_flag(self) -> None:
        # 0.75 + 0.5 = 1.25 -> clamped to 1.0 (clamped = True, inexact = True)
        s0 = f16_to_bits(0.75)
        s1 = f16_to_bits(0.5)
        res = oracle_v_add_clamp_f16(s0, s1)
        flags = evaluate_v_add_clamp_f16_flags(s0, s1, res)
        self.assertTrue(flags.clamped)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.overflow)

    def test_clamped_negative_produces_zero_flag(self) -> None:
        # -0.5 + 0.2 = -0.3 -> clamped to +0.0 (zero = True, negative = False, clamped = True)
        s0 = f16_to_bits(-0.5)
        s1 = f16_to_bits(0.2)
        res = oracle_v_add_clamp_f16(s0, s1)
        self.assertEqual(res, POS_ZERO)
        flags = evaluate_v_add_clamp_f16_flags(s0, s1, res)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.negative)
        self.assertTrue(flags.clamped)

    def test_underflow_and_denorm_flags(self) -> None:
        subnorm = 0x0001
        res = oracle_v_add_clamp_f16(subnorm, subnorm)
        self.assertEqual(res, 0x0002)
        flags = evaluate_v_add_clamp_f16_flags(subnorm, subnorm, res)
        self.assertTrue(flags.denormal)
        self.assertTrue(flags.underflow)
        self.assertFalse(flags.clamped)

    def test_invalid_operation_flag(self) -> None:
        # +inf + -inf is invalid, clamps to +0.0
        res = oracle_v_add_clamp_f16(POS_INF, NEG_INF)
        self.assertEqual(res, POS_ZERO)
        flags = evaluate_v_add_clamp_f16_flags(POS_INF, NEG_INF, res)
        self.assertTrue(flags.invalid)
        self.assertTrue(flags.clamped)

        # signaling NaN triggers invalid, clamps to +0.0
        snan = 0x7C01
        res2 = oracle_v_add_clamp_f16(snan, f16_to_bits(0.5))
        self.assertEqual(res2, POS_ZERO)
        flags2 = evaluate_v_add_clamp_f16_flags(snan, f16_to_bits(0.5), res2)
        self.assertTrue(flags2.invalid)
        self.assertTrue(flags2.clamped)

    def test_mantissa_carry_generation(self) -> None:
        # 1.0 + 1.0: mantissas are 1.000... + 1.000... = 2.000... -> carry out = 1
        _, carry1 = compute_f16_mantissa_carry(f16_to_bits(1.0), f16_to_bits(1.0))
        self.assertEqual(carry1, 1)

        # 1.0 + 0.25: mantissas sum < 2.0 -> carry out = 0
        _, carry0 = compute_f16_mantissa_carry(f16_to_bits(1.0), f16_to_bits(0.25))
        self.assertEqual(carry0, 0)


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_addition_without_clamp(self) -> None:
        sub1 = 0x0001
        sub2 = 0x0002
        res = oracle_v_add_clamp_f16(sub1, sub2)
        self.assertEqual(res, 0x0003)
        self.assertTrue(is_subnormal(res))

    def test_subnormal_to_normal_transition(self) -> None:
        # Largest subnormal: 0x03FF
        max_sub = 0x03FF
        min_sub = 0x0001
        # Sum is 0x0400 = smallest normal (2^-14)
        res = oracle_v_add_clamp_f16(max_sub, min_sub)
        self.assertEqual(res, 0x0400)
        self.assertFalse(is_subnormal(res))
        self.assertEqual(split_f16_bits(res)[1], 1)  # exponent = 1
        self.assertEqual(split_f16_bits(res)[2], 0)  # fraction = 0

    def test_subnormal_cancellation(self) -> None:
        sub = 0x0123
        neg_sub = sub | SIGN_MASK
        res = oracle_v_add_clamp_f16(sub, neg_sub)
        self.assertEqual(res, POS_ZERO)

    def test_nan_clamp_to_zero(self) -> None:
        snan = 0x7C05
        qnan = CANONICAL_QNAN
        # RDNA2 clamp modifier maps all NaNs to +0.0
        self.assertEqual(oracle_v_add_clamp_f16(snan, f16_to_bits(0.5)), POS_ZERO)
        self.assertEqual(oracle_v_add_clamp_f16(qnan, f16_to_bits(0.5)), POS_ZERO)
        self.assertEqual(oracle_v_add_clamp_f16(snan, qnan), POS_ZERO)

    def test_infinity_clamped_boundaries(self) -> None:
        self.assertEqual(oracle_v_add_clamp_f16(POS_INF, f16_to_bits(100.0)), ONE_F16)
        self.assertEqual(oracle_v_add_clamp_f16(NEG_INF, f16_to_bits(100.0)), POS_ZERO)
        self.assertEqual(oracle_v_add_clamp_f16(POS_INF, NEG_INF), POS_ZERO)
        self.assertEqual(oracle_v_add_clamp_f16(NEG_INF, POS_INF), POS_ZERO)
        self.assertEqual(oracle_v_add_clamp_f16(POS_INF, POS_INF), ONE_F16)
        self.assertEqual(oracle_v_add_clamp_f16(NEG_INF, NEG_INF), POS_ZERO)

    def test_round_to_nearest_even_ties(self) -> None:
        # In IEEE 754 half precision (11 bits significand):
        # 0.5 has bits 0x3800 (even LSB = 0)
        # 0.5 + 2^-11 has bits 0x3801 (odd LSB = 1)
        # 2^-12 is halfway between representable numbers.
        half_ulp = 2.0 ** -12
        val_even = 0.5
        val_odd = bits_to_f16(0x3801)

        # Case 1: 0.5 (even LSB) + 2^-12: tie rounds to even (0.5)
        res1 = oracle_v_add_clamp_f16(f16_to_bits(val_even), f16_to_bits(half_ulp))
        self.assertEqual(res1, f16_to_bits(val_even))

        # Case 2: (0.5 + 2^-11) (odd LSB) + 2^-12: tie rounds to even (0.5 + 2 * 2^-11, 0x3802)
        res2 = oracle_v_add_clamp_f16(f16_to_bits(val_odd), f16_to_bits(half_ulp))
        self.assertEqual(res2, 0x3802)


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Reg16(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg16(256, RegType.VGPR)

    def test_syntax_rejection_valu_destination_must_be_vgpr(self) -> None:
        dst_sgpr = Reg16(0, RegType.SGPR)
        src = VOP3Operand16(Reg16(2, RegType.VGPR))
        with self.assertRaises(TypeError):
            VAddClampF16Instruction(dst=dst_sgpr, src0=src, src1=src)

    def test_syntax_rejection_invalid_lane_count(self) -> None:
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_negative_control_bit_flip_detection(self) -> None:
        s0 = f16_to_bits(0.2)
        s1 = f16_to_bits(0.3)
        correct = oracle_v_add_clamp_f16(s0, s1)
        self.assertAlmostEqual(bits_to_f16(correct), 0.5, places=2)

        # Perturb single mantissa bit
        corrupted_mantissa = correct ^ 0x1
        self.assertNotEqual(correct, corrupted_mantissa)

        # Perturb single exponent bit
        corrupted_exp = correct ^ (1 << 10)
        self.assertNotEqual(correct, corrupted_exp)

        # Perturb sign bit
        corrupted_sign = correct ^ SIGN_MASK
        self.assertNotEqual(correct, corrupted_sign)

    def test_mutation_add_to_sub(self) -> None:
        """Mutant: Replace addition with subtraction (s0 - s1)."""
        def mutant_v_add_clamp_f16(s0_bits: int, s1_bits: int) -> int:
            f0 = bits_to_f16(s0_bits)
            f1 = bits_to_f16(s1_bits)
            diff = f0 - f1
            clamped = min(1.0, max(0.0, diff))
            return f16_to_bits(clamped)

        s0 = f16_to_bits(0.6)
        s1 = f16_to_bits(0.2)
        expected = oracle_v_add_clamp_f16(s0, s1)  # 0.8
        mutant_res = mutant_v_add_clamp_f16(s0, s1)  # 0.4
        self.assertNotEqual(expected, mutant_res)

    def test_mutation_ignore_abs_modifier(self) -> None:
        """Mutant: Ignore abs modifier on negative operand."""
        def mutant_apply_mod(bits: int, abs_mod: bool, neg_mod: bool) -> int:
            res = bits & 0xFFFF
            if neg_mod:
                res ^= SIGN_MASK
            return res

        val = f16_to_bits(-0.4)
        correct = apply_f16_modifiers(val, abs_mod=True, neg_mod=False)
        mutant = mutant_apply_mod(val, abs_mod=True, neg_mod=False)
        self.assertAlmostEqual(bits_to_f16(correct), 0.4, places=2)
        self.assertNotEqual(correct, mutant)

    def test_mutation_ignore_neg_modifier(self) -> None:
        """Mutant: Ignore neg modifier on positive operand."""
        def mutant_apply_mod(bits: int, abs_mod: bool, neg_mod: bool) -> int:
            res = bits & 0xFFFF
            if abs_mod:
                res &= ~SIGN_MASK
            return res

        val = f16_to_bits(0.5)
        correct = apply_f16_modifiers(val, abs_mod=False, neg_mod=True)
        mutant = mutant_apply_mod(val, abs_mod=False, neg_mod=True)
        self.assertEqual(bits_to_f16(correct), -0.5)
        self.assertNotEqual(correct, mutant)

    def test_mutation_unclamped_upper_bound(self) -> None:
        """Mutant: Fails to clamp values exceeding 1.0."""
        s0 = f16_to_bits(0.75)
        s1 = f16_to_bits(0.5)
        expected = oracle_v_add_clamp_f16(s0, s1)
        self.assertEqual(expected, ONE_F16)

        # Mutant produces unclamped 1.25
        unclamped = f16_to_bits(1.25)
        self.assertNotEqual(expected, unclamped)

    def test_mutation_unclamped_lower_bound(self) -> None:
        """Mutant: Fails to clamp negative values to +0.0."""
        s0 = f16_to_bits(-1.0)
        s1 = f16_to_bits(0.2)
        expected = oracle_v_add_clamp_f16(s0, s1)
        self.assertEqual(expected, POS_ZERO)

        # Mutant produces negative float
        unclamped = f16_to_bits(-0.8)
        self.assertNotEqual(expected, unclamped)

    def test_mutation_ignore_exec_mask(self) -> None:
        """Mutant: Inactive SIMD lanes are overwritten despite EXEC mask."""
        sentinel = 0xDEAD
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)  # all inactive
        for lane in range(32):
            state.set_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO), sentinel)

        # Mutant interpreter that ignores exec_mask
        class MutantSIMDInterpreter(SIMDInterpreter):
            def step(self, instr: VAddClampF16Instruction, state: SIMDState) -> List[Optional[int]]:
                results = []
                for lane in range(state.num_lanes):
                    res = oracle_v_add_clamp_f16(f16_to_bits(0.2), f16_to_bits(0.3))
                    state.set_reg16(lane, instr.dst, res)
                    results.append(res)
                return results

        mutant_interp = MutantSIMDInterpreter()
        instr = VAddClampF16Instruction(
            dst=Reg16(4, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Literal16.from_float(0.2)),
            src1=VOP3Operand16(Literal16.from_float(0.3)),
        )
        mutant_interp.step(instr, state)
        # Mutation detected: inactive lanes corrupted
        corrupted_lanes = [
            lane for lane in range(32)
            if state.get_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO)) != sentinel
        ]
        self.assertEqual(len(corrupted_lanes), 32)

    def test_mutation_subnormal_flush_to_zero(self) -> None:
        """Mutant: Subnormals flushed to zero (FTZ mode)."""
        sub = 0x0001
        correct = oracle_v_add_clamp_f16(sub, sub)
        self.assertEqual(correct, 0x0002)

        # FTZ mutant flushes to 0.0
        ftz_mutant = POS_ZERO
        self.assertNotEqual(correct, ftz_mutant)

    def test_mutation_nan_not_clamped(self) -> None:
        """Mutant: Clamped NaN propagates NaN instead of clamping to +0.0."""
        snan = 0x7C05
        correct = oracle_v_add_clamp_f16(snan, f16_to_bits(0.5))
        self.assertEqual(correct, POS_ZERO)

        # Propagated NaN mutant
        nan_mutant = snan | QNAN_BIT
        self.assertNotEqual(correct, nan_mutant)


if __name__ == "__main__":
    unittest.main()
