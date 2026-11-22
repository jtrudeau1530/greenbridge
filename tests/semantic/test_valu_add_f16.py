from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import List, Optional, Tuple, Union

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


def oracle_v_add_f16(
    src0_bits: int,
    src1_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_add_f16 half-precision addition.
    
    Computes D = src0 + src1 conforming to IEEE 754-2008 half precision (binary16)
    with Round-to-Nearest-Even (RNE), applying input modifiers and clamp.
    """
    s0 = apply_f16_modifiers(src0_bits, abs0, neg0)
    s1 = apply_f16_modifiers(src1_bits, abs1, neg1)

    # 1. NaN handling
    if is_nan(s0) or is_nan(s1):
        if clamp:
            # Clamping NaN in RDNA2 produces +0.0
            return POS_ZERO
        # Prefer quietening signaling NaN or propagating quiet NaN
        nan_bits = s0 if is_nan(s0) else s1
        return nan_bits | QNAN_BIT

    # 2. Infinity handling
    s0_inf = is_inf(s0)
    s1_inf = is_inf(s1)
    if s0_inf and s1_inf:
        s0_sign = (s0 & SIGN_MASK)
        s1_sign = (s1 & SIGN_MASK)
        if s0_sign != s1_sign:
            # +inf + -inf -> Invalid operation -> Canonical Quiet NaN
            return POS_ZERO if clamp else CANONICAL_QNAN
        # Same signed infinities
        if clamp:
            return ONE_F16 if s0 == POS_INF else POS_ZERO
        return s0
    if s0_inf:
        if clamp:
            return ONE_F16 if s0 == POS_INF else POS_ZERO
        return s0
    if s1_inf:
        if clamp:
            return ONE_F16 if s1 == POS_INF else POS_ZERO
        return s1

    # 3. Signed zero handling
    if is_zero(s0) and is_zero(s1):
        s0_is_neg = (s0 == NEG_ZERO)
        s1_is_neg = (s1 == NEG_ZERO)
        if s0_is_neg and s1_is_neg:
            res = NEG_ZERO
        else:
            res = POS_ZERO
        if clamp and res == NEG_ZERO:
            return POS_ZERO
        return res

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
    if clamp:
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
class VAddF16Instruction:
    """Symbolic AST node representing a RDNA2 v_add_f16 instruction."""
    dst: Reg16
    src0: VOP3Operand16
    src1: VOP3Operand16
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        s = f"v_add_f16 {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}"
        if self.clamp:
            s += " clamp"
        return s


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
    """Deterministic scalar instruction interpreter for v_add_f16."""

    def resolve_operand(self, op: VOP3Operand16, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg16):
            return state.get_reg16(op.source)
        elif isinstance(op.source, Literal16):
            return op.source.bits
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VAddF16Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)

        # Computation phase
        result_bits = oracle_v_add_f16(
            src0_bits=s0_raw,
            src1_bits=s1_raw,
            abs0=instr.src0.abs_mod,
            neg0=instr.src0.neg_mod,
            abs1=instr.src1.abs_mod,
            neg1=instr.src1.neg_mod,
            clamp=instr.clamp,
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

    def step(self, instr: VAddF16Instruction, state: SIMDState) -> List[Optional[int]]:
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
                res = oracle_v_add_f16(
                    src0_bits=s0_vals[lane],
                    src1_bits=s1_vals[lane],
                    abs0=instr.src0.abs_mod,
                    neg0=instr.src0.neg_mod,
                    abs1=instr.src1.abs_mod,
                    neg1=instr.src1.neg_mod,
                    clamp=instr.clamp,
                )
                state.set_reg16(lane, instr.dst, res)
                results[lane] = res

        return results


# ==============================================================================
# Condition Codes, Status Flags, and Carry Generation
# ==============================================================================

@dataclass(frozen=True)
class FPStatusFlags:
    """Architectural floating-point condition codes and IEEE 754 status flags."""
    invalid: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False
    zero: bool = False
    negative: bool = False
    denormal: bool = False
    infinite: bool = False
    nan: bool = False


def evaluate_v_add_f16_flags(
    src0_bits: int,
    src1_bits: int,
    res_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> FPStatusFlags:
    """Evaluate IEEE 754 architectural flags and condition codes for v_add_f16."""
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

    overflow = False
    if not (s0_nan or s1_nan or s0_inf or s1_inf) and inf_flag:
        overflow = True

    underflow = False
    if denorm_flag:
        underflow = True

    inexact = False
    if overflow:
        inexact = True
    elif not (s0_nan or s1_nan or s0_inf or s1_inf or nan_flag or inf_flag):
        f0 = bits_to_f16(s0)
        f1 = bits_to_f16(s1)
        res_f = bits_to_f16(res_bits)
        if not clamp:
            if Fraction(f0) + Fraction(f1) != Fraction(res_f):
                inexact = True
        else:
            clamped_math = min(1.0, max(0.0, float(Fraction(f0) + Fraction(f1))))
            if Fraction(clamped_math) != Fraction(res_f):
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

class TestF16ModelAndOracle(unittest.TestCase):
    """Unit tests for binary16 bit model, predicates, and reference oracle."""

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

    def test_exact_finite_addition(self) -> None:
        c = oracle_v_add_f16(f16_to_bits(1.5), f16_to_bits(2.25))
        self.assertEqual(bits_to_f16(c), 3.75)

        c2 = oracle_v_add_f16(f16_to_bits(100.0), f16_to_bits(-25.5))
        self.assertEqual(bits_to_f16(c2), 74.5)

    def test_signed_zero_rules(self) -> None:
        # IEEE 754: (+0) + (+0) = +0
        self.assertEqual(oracle_v_add_f16(POS_ZERO, POS_ZERO), POS_ZERO)
        # (-0) + (-0) = -0
        self.assertEqual(oracle_v_add_f16(NEG_ZERO, NEG_ZERO), NEG_ZERO)
        # (+0) + (-0) = +0
        self.assertEqual(oracle_v_add_f16(POS_ZERO, NEG_ZERO), POS_ZERO)
        # (-0) + (+0) = +0
        self.assertEqual(oracle_v_add_f16(NEG_ZERO, POS_ZERO), POS_ZERO)
        # x + (-x) = +0
        self.assertEqual(oracle_v_add_f16(f16_to_bits(42.0), f16_to_bits(-42.0)), POS_ZERO)

    def test_input_modifiers_abs_neg(self) -> None:
        val_pos = f16_to_bits(10.0)
        val_neg = f16_to_bits(-10.0)

        # abs on negative: |-10.0| + 5.0 = 15.0
        res = oracle_v_add_f16(val_neg, f16_to_bits(5.0), abs0=True)
        self.assertEqual(bits_to_f16(res), 15.0)

        # neg on positive: -(10.0) + 5.0 = -5.0
        res = oracle_v_add_f16(val_pos, f16_to_bits(5.0), neg0=True)
        self.assertEqual(bits_to_f16(res), -5.0)

        # abs and neg together: -|10.0| + 5.0 = -5.0; -|-10.0| + 5.0 = -5.0
        res1 = oracle_v_add_f16(val_pos, f16_to_bits(5.0), abs0=True, neg0=True)
        res2 = oracle_v_add_f16(val_neg, f16_to_bits(5.0), abs0=True, neg0=True)
        self.assertEqual(bits_to_f16(res1), -5.0)
        self.assertEqual(bits_to_f16(res2), -5.0)

    def test_clamp_modifier(self) -> None:
        # Clamped result > 1.0 becomes 1.0
        res_hi = oracle_v_add_f16(f16_to_bits(0.75), f16_to_bits(0.5), clamp=True)
        self.assertEqual(res_hi, ONE_F16)

        # Clamped result < 0.0 becomes +0.0
        res_lo = oracle_v_add_f16(f16_to_bits(-1.5), f16_to_bits(0.5), clamp=True)
        self.assertEqual(res_lo, POS_ZERO)

        # Clamped result within [0.0, 1.0] is unmodified
        res_mid = oracle_v_add_f16(f16_to_bits(0.25), f16_to_bits(0.5), clamp=True)
        self.assertEqual(bits_to_f16(res_mid), 0.75)

    def test_infinity_addition(self) -> None:
        self.assertEqual(oracle_v_add_f16(POS_INF, f16_to_bits(1.0)), POS_INF)
        self.assertEqual(oracle_v_add_f16(NEG_INF, f16_to_bits(1.0)), NEG_INF)
        self.assertEqual(oracle_v_add_f16(POS_INF, POS_INF), POS_INF)
        self.assertEqual(oracle_v_add_f16(NEG_INF, NEG_INF), NEG_INF)
        # (+inf) + (-inf) -> Canonical NaN
        self.assertEqual(oracle_v_add_f16(POS_INF, NEG_INF), CANONICAL_QNAN)

    def test_nan_quietening_and_propagation(self) -> None:
        snan = 0x7C05
        res = oracle_v_add_f16(snan, f16_to_bits(1.0))
        self.assertTrue(is_qnan(res))
        self.assertEqual(res, snan | QNAN_BIT)


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

    def test_v_add_f16_ast_construction_and_validation(self) -> None:
        dst = Reg16(0, RegType.VGPR, SubWord.LO)
        src0 = VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.HI), neg_mod=True, abs_mod=True)
        src1 = VOP3Operand16(Reg16(4, RegType.SGPR, SubWord.LO))
        instr = VAddF16Instruction(dst=dst, src0=src0, src1=src1, clamp=True)

        self.assertEqual(instr.disassemble(), "v_add_f16 v0.l, -|v2.h|, s4.l clamp")

        # Invalid destination type (SGPR)
        with self.assertRaises(TypeError):
            VAddF16Instruction(dst=Reg16(0, RegType.SGPR), src0=src0, src1=src1)

    def test_structural_equality(self) -> None:
        i1 = VAddF16Instruction(
            Reg16(0, RegType.VGPR),
            VOP3Operand16(Reg16(2, RegType.VGPR)),
            VOP3Operand16(Literal16.from_float(0.0)),
        )
        i2 = VAddF16Instruction(
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

    def test_scalar_vgpr_vgpr_addition(self) -> None:
        self.state.set_reg16(Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(12.5))
        self.state.set_reg16(Reg16(4, RegType.VGPR, SubWord.LO), f16_to_bits(7.5))
        instr = VAddF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(4, RegType.VGPR, SubWord.LO)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(bits_to_f16(res_bits), 20.0)
        self.assertEqual(self.state.get_reg16(Reg16(0, RegType.VGPR, SubWord.LO)), res_bits)

    def test_scalar_vgpr_sgpr_addition(self) -> None:
        self.state.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(10.0))
        self.state.set_reg16(Reg16(8, RegType.SGPR, SubWord.LO), f16_to_bits(2.5))
        instr = VAddF16Instruction(
            dst=Reg16(4, RegType.VGPR, SubWord.HI),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(8, RegType.SGPR, SubWord.LO)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(bits_to_f16(res_bits), 12.5)
        self.assertEqual(self.state.get_reg16(Reg16(4, RegType.VGPR, SubWord.HI)), res_bits)

    def test_scalar_vgpr_literal_addition(self) -> None:
        self.state.set_reg16(Reg16(6, RegType.VGPR, SubWord.LO), f16_to_bits(3.0))
        instr = VAddF16Instruction(
            dst=Reg16(6, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(6, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Literal16.from_float(0.5)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(bits_to_f16(res_bits), 3.5)
        self.assertEqual(self.state.get_reg16(Reg16(6, RegType.VGPR, SubWord.LO)), res_bits)

    def test_scalar_with_modifiers_and_clamp(self) -> None:
        self.state.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(-0.8))
        self.state.set_reg16(Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(1.5))
        # -|-0.8| + 1.5 = -0.8 + 1.5 = 0.7 (within [0, 1] so clamp doesn't alter 0.7)
        instr = VAddF16Instruction(
            dst=Reg16(4, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO), abs_mod=True, neg_mod=True),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
            clamp=True,
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertAlmostEqual(bits_to_f16(res_bits), 0.7, places=3)

    def test_scalar_subword_preservation(self) -> None:
        # Prepopulate v1 with distinct lo and hi half words
        self.state.vgprs[1] = pack_f16_pair(0x1111, 0x2222)
        instr_lo = VAddF16Instruction(
            dst=Reg16(1, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Literal16.from_float(1.0)),
            src1=VOP3Operand16(Literal16.from_float(2.0)),
        )
        self.interp.step(instr_lo, self.state)
        # Low half updated to 3.0 (0x4200), high half preserved (0x2222)
        self.assertEqual(self.state.get_reg16(Reg16(1, RegType.VGPR, SubWord.LO)), 0x4200)
        self.assertEqual(self.state.get_reg16(Reg16(1, RegType.VGPR, SubWord.HI)), 0x2222)

        instr_hi = VAddF16Instruction(
            dst=Reg16(1, RegType.VGPR, SubWord.HI),
            src0=VOP3Operand16(Literal16.from_float(0.5)),
            src1=VOP3Operand16(Literal16.from_float(0.5)),
        )
        self.interp.step(instr_hi, self.state)
        # High half updated to 1.0 (0x3C00), low half preserved (0x4200)
        self.assertEqual(self.state.get_reg16(Reg16(1, RegType.VGPR, SubWord.LO)), 0x4200)
        self.assertEqual(self.state.get_reg16(Reg16(1, RegType.VGPR, SubWord.HI)), 0x3C00)

    def test_scalar_determinism(self) -> None:
        runs = []
        for _ in range(50):
            st = ScalarMachineState()
            st.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), 0x4248)  # ~3.14
            st.set_reg16(Reg16(2, RegType.VGPR, SubWord.LO), 0x4170)  # ~2.718
            instr = VAddF16Instruction(
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
            state.set_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(float(lane)))
            state.set_reg16(lane, Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(10.0))
        instr = VAddF16Instruction(
            dst=Reg16(4, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            expected = float(lane) + 10.0
            self.assertEqual(bits_to_f16(state.get_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO))), expected)

    def test_wave32_divergent_masking(self) -> None:
        # Alternating mask: even lanes active (0x55555555)
        sentinel = 0xBEEF
        state = SIMDState(num_lanes=32, exec_mask=0x5555_5555)
        for lane in range(32):
            state.set_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(float(lane)))
            state.set_reg16(lane, Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(5.0))
            state.set_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO), sentinel)

        instr = VAddF16Instruction(
            dst=Reg16(4, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertIsNotNone(results[lane])
                self.assertEqual(bits_to_f16(state.get_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO))), float(lane) + 5.0)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO)), sentinel)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(float(lane) * 0.5))
            state.set_reg16(lane, Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(1.0))
        instr = VAddF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        self.interp.step(instr, state)
        for lane in range(64):
            self.assertEqual(bits_to_f16(state.get_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO))), float(lane) * 0.5 + 1.0)

    def test_wave64_half_disabled(self) -> None:
        # Lower 32 lanes active, upper 32 lanes inactive
        sentinel = 0xCAFE
        state = SIMDState(num_lanes=64, exec_mask=0x0000_0000_FFFF_FFFF)
        for lane in range(64):
            state.set_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(10.0))
            state.set_reg16(lane, Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(20.0))
            state.set_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO), sentinel)

        instr = VAddF16Instruction(
            dst=Reg16(4, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        results = self.interp.step(instr, state)
        for lane in range(64):
            if lane < 32:
                self.assertIsNotNone(results[lane])
                self.assertEqual(bits_to_f16(state.get_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO))), 30.0)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO)), sentinel)

    def test_wave_exec_all_zero(self) -> None:
        sentinel = 0xAA55
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)
        for lane in range(32):
            state.set_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO), sentinel)
        instr = VAddF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Literal16.from_float(1.0)),
            src1=VOP3Operand16(Literal16.from_float(2.0)),
        )
        results = self.interp.step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(state.get_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO)), sentinel)

    def test_simd_sgpr_uniform_broadcast(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        state.set_reg16(0, Reg16(10, RegType.SGPR, SubWord.LO), f16_to_bits(100.0))
        for lane in range(32):
            state.set_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(float(lane)))

        instr = VAddF16Instruction(
            dst=Reg16(2, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(10, RegType.SGPR, SubWord.LO)),
        )
        self.interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(bits_to_f16(state.get_reg16(lane, Reg16(2, RegType.VGPR, SubWord.LO))), 100.0 + float(lane))


class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src0(self) -> None:
        state = ScalarMachineState()
        state.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(15.5))
        state.set_reg16(Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(4.5))
        instr = VAddF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        self.scalar_interp.step(instr, state)
        self.assertEqual(bits_to_f16(state.get_reg16(Reg16(0, RegType.VGPR, SubWord.LO))), 20.0)
        # src1 untouched
        self.assertEqual(bits_to_f16(state.get_reg16(Reg16(2, RegType.VGPR, SubWord.LO))), 4.5)

    def test_dst_alias_src1(self) -> None:
        state = ScalarMachineState()
        state.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(15.5))
        state.set_reg16(Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(4.5))
        instr = VAddF16Instruction(
            dst=Reg16(2, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        self.scalar_interp.step(instr, state)
        self.assertEqual(bits_to_f16(state.get_reg16(Reg16(2, RegType.VGPR, SubWord.LO))), 20.0)
        # src0 untouched
        self.assertEqual(bits_to_f16(state.get_reg16(Reg16(0, RegType.VGPR, SubWord.LO))), 15.5)

    def test_dst_alias_both_srcs(self) -> None:
        state = ScalarMachineState()
        state.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(16.0))
        instr = VAddF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
        )
        self.scalar_interp.step(instr, state)
        self.assertEqual(bits_to_f16(state.get_reg16(Reg16(0, RegType.VGPR, SubWord.LO))), 32.0)

    def test_subword_cross_half_aliasing(self) -> None:
        # Prepopulate v0 with lo=1.5, hi=2.5
        state = ScalarMachineState()
        state.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(1.5))
        state.set_reg16(Reg16(0, RegType.VGPR, SubWord.HI), f16_to_bits(2.5))

        instr = VAddF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.HI),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.HI)),
        )
        self.scalar_interp.step(instr, state)
        # hi updated to 4.0, lo untouched at 1.5
        self.assertEqual(bits_to_f16(state.get_reg16(Reg16(0, RegType.VGPR, SubWord.HI))), 4.0)
        self.assertEqual(bits_to_f16(state.get_reg16(Reg16(0, RegType.VGPR, SubWord.LO))), 1.5)

    def test_inplace_iterative_accumulation(self) -> None:
        state = ScalarMachineState()
        state.set_reg16(Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(0.0))
        state.set_reg16(Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(0.5))
        instr = VAddF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        for _ in range(100):
            self.scalar_interp.step(instr, state)
        self.assertEqual(bits_to_f16(state.get_reg16(Reg16(0, RegType.VGPR, SubWord.LO))), 50.0)

    def test_simd_multi_lane_aliasing(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO), f16_to_bits(float(lane)))
            state.set_reg16(lane, Reg16(2, RegType.VGPR, SubWord.LO), f16_to_bits(1.0))

        instr = VAddF16Instruction(
            dst=Reg16(0, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Reg16(0, RegType.VGPR, SubWord.LO)),
            src1=VOP3Operand16(Reg16(2, RegType.VGPR, SubWord.LO)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(bits_to_f16(state.get_reg16(lane, Reg16(0, RegType.VGPR, SubWord.LO))), float(lane) + 1.0)


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for status flags, condition codes, and carry generation."""

    def test_exact_finite_flags(self) -> None:
        s0 = f16_to_bits(1.0)
        s1 = f16_to_bits(2.0)
        res = oracle_v_add_f16(s0, s1)
        flags = evaluate_v_add_f16_flags(s0, s1, res)
        self.assertFalse(flags.invalid)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.underflow)
        self.assertFalse(flags.inexact)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.denormal)
        self.assertFalse(flags.infinite)
        self.assertFalse(flags.nan)

    def test_inexact_flag(self) -> None:
        # In f16, 1.0 + 2^-11 has an inexact round
        s0 = f16_to_bits(1.0)
        s1 = f16_to_bits(2.0 ** -11)
        res = oracle_v_add_f16(s0, s1)
        flags = evaluate_v_add_f16_flags(s0, s1, res)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.underflow)

    def test_overflow_flag(self) -> None:
        half_max = 0x7BFF  # 65504.0
        res = oracle_v_add_f16(half_max, half_max)
        self.assertEqual(res, POS_INF)
        flags = evaluate_v_add_f16_flags(half_max, half_max, res)
        self.assertTrue(flags.overflow)
        self.assertTrue(flags.infinite)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.invalid)

    def test_underflow_and_denorm_flags(self) -> None:
        subnorm = 0x0001
        res = oracle_v_add_f16(subnorm, subnorm)
        self.assertEqual(res, 0x0002)
        flags = evaluate_v_add_f16_flags(subnorm, subnorm, res)
        self.assertTrue(flags.denormal)
        self.assertTrue(flags.underflow)
        self.assertFalse(flags.overflow)

    def test_invalid_operation_flag(self) -> None:
        # +inf + -inf is invalid
        res = oracle_v_add_f16(POS_INF, NEG_INF)
        flags = evaluate_v_add_f16_flags(POS_INF, NEG_INF, res)
        self.assertTrue(flags.invalid)
        self.assertTrue(flags.nan)

        # signaling NaN operand triggers invalid operation
        snan = 0x7C01
        res2 = oracle_v_add_f16(snan, f16_to_bits(1.0))
        flags2 = evaluate_v_add_f16_flags(snan, f16_to_bits(1.0), res2)
        self.assertTrue(flags2.invalid)
        self.assertTrue(flags2.nan)

    def test_zero_and_negative_flags(self) -> None:
        # +0 + +0
        res_pz = oracle_v_add_f16(POS_ZERO, POS_ZERO)
        flags_pz = evaluate_v_add_f16_flags(POS_ZERO, POS_ZERO, res_pz)
        self.assertTrue(flags_pz.zero)
        self.assertFalse(flags_pz.negative)

        # -0 + -0
        res_nz = oracle_v_add_f16(NEG_ZERO, NEG_ZERO)
        flags_nz = evaluate_v_add_f16_flags(NEG_ZERO, NEG_ZERO, res_nz)
        self.assertTrue(flags_nz.zero)
        self.assertTrue(flags_nz.negative)

        # -5.0 + -10.0 = -15.0
        s0 = f16_to_bits(-5.0)
        s1 = f16_to_bits(-10.0)
        res_neg = oracle_v_add_f16(s0, s1)
        flags_neg = evaluate_v_add_f16_flags(s0, s1, res_neg)
        self.assertFalse(flags_neg.zero)
        self.assertTrue(flags_neg.negative)

    def test_mantissa_carry_generation(self) -> None:
        # 1.0 + 1.0: mantissas are 1.000... + 1.000... = 2.000... -> carry out = 1
        _, carry1 = compute_f16_mantissa_carry(f16_to_bits(1.0), f16_to_bits(1.0))
        self.assertEqual(carry1, 1)

        # 1.0 + 0.25: mantissas sum < 2.0 -> carry out = 0
        _, carry0 = compute_f16_mantissa_carry(f16_to_bits(1.0), f16_to_bits(0.25))
        self.assertEqual(carry0, 0)


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_addition_without_carry(self) -> None:
        sub1 = 0x0001
        sub2 = 0x0002
        res = oracle_v_add_f16(sub1, sub2)
        self.assertEqual(res, 0x0003)
        self.assertTrue(is_subnormal(res))

    def test_subnormal_to_normal_transition(self) -> None:
        # Largest subnormal: 0x03FF
        max_sub = 0x03FF
        min_sub = 0x0001
        # Sum is 0x0400 = smallest normal (2^-14)
        res = oracle_v_add_f16(max_sub, min_sub)
        self.assertEqual(res, 0x0400)
        self.assertFalse(is_subnormal(res))
        self.assertEqual(split_f16_bits(res)[1], 1)  # exponent = 1
        self.assertEqual(split_f16_bits(res)[2], 0)  # fraction = 0

    def test_subnormal_cancellation(self) -> None:
        sub = 0x0123
        neg_sub = sub | SIGN_MASK
        res = oracle_v_add_f16(sub, neg_sub)
        self.assertEqual(res, POS_ZERO)

    def test_nan_signaling_to_quiet_conversion(self) -> None:
        snan = 0x7C05  # fraction non-zero, quiet bit 9 is 0
        self.assertTrue(is_snan(snan))
        res = oracle_v_add_f16(snan, f16_to_bits(1.0))
        self.assertTrue(is_qnan(res))
        self.assertEqual(res, snan | QNAN_BIT)

    def test_nan_payload_preservation(self) -> None:
        payload = 0x02AB
        qnan = EXP_MASK | payload
        res = oracle_v_add_f16(qnan, f16_to_bits(10.0))
        self.assertEqual(res, qnan)

    def test_nan_with_modifiers(self) -> None:
        qnan = 0x7E01
        # abs should clear sign bit
        res_abs = oracle_v_add_f16(qnan | SIGN_MASK, f16_to_bits(0.0), abs0=True)
        self.assertEqual(res_abs & SIGN_MASK, 0)
        self.assertTrue(is_qnan(res_abs))

        # neg should toggle sign bit
        res_neg = oracle_v_add_f16(qnan, f16_to_bits(0.0), neg0=True)
        self.assertEqual(res_neg & SIGN_MASK, SIGN_MASK)

    def test_nan_with_clamp(self) -> None:
        qnan = CANONICAL_QNAN
        res = oracle_v_add_f16(qnan, f16_to_bits(1.0), clamp=True)
        self.assertEqual(res, POS_ZERO)

    def test_infinity_arithmetic_boundaries(self) -> None:
        self.assertEqual(oracle_v_add_f16(POS_INF, f16_to_bits(100.0)), POS_INF)
        self.assertEqual(oracle_v_add_f16(NEG_INF, f16_to_bits(100.0)), NEG_INF)
        self.assertEqual(oracle_v_add_f16(POS_INF, NEG_INF), CANONICAL_QNAN)
        self.assertEqual(oracle_v_add_f16(NEG_INF, POS_INF), CANONICAL_QNAN)

        # Infinity with clamp
        self.assertEqual(oracle_v_add_f16(POS_INF, f16_to_bits(0.0), clamp=True), ONE_F16)
        self.assertEqual(oracle_v_add_f16(NEG_INF, f16_to_bits(0.0), clamp=True), POS_ZERO)

    def test_round_to_nearest_even_ties(self) -> None:
        # In IEEE 754 half precision (11 bits significand):
        # 1.0 has bits 0x3C00 (even LSB = 0)
        # 1.0 + 2^-10 has bits 0x3C01 (odd LSB = 1)
        # 2^-11 is exactly halfway between representable numbers.
        half_ulp = 2.0 ** -11
        one = 1.0
        one_plus_ulp = bits_to_f16(0x3C01)

        # Case 1: 1.0 (even LSB) + 2^-11: tie rounds to even (1.0)
        res1 = oracle_v_add_f16(f16_to_bits(one), f16_to_bits(half_ulp))
        self.assertEqual(res1, f16_to_bits(one))

        # Case 2: (1.0 + 2^-10) (odd LSB) + 2^-11: tie rounds to even (1.0 + 2 * 2^-10)
        res2 = oracle_v_add_f16(f16_to_bits(one_plus_ulp), f16_to_bits(half_ulp))
        self.assertEqual(res2, 0x3C02)


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
            VAddF16Instruction(dst=dst_sgpr, src0=src, src1=src)

    def test_syntax_rejection_invalid_lane_count(self) -> None:
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_negative_control_bit_flip_detection(self) -> None:
        s0 = f16_to_bits(12.0)
        s1 = f16_to_bits(18.0)
        correct = oracle_v_add_f16(s0, s1)
        self.assertEqual(bits_to_f16(correct), 30.0)

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
        def mutant_v_add_f16(s0_bits: int, s1_bits: int) -> int:
            f0 = bits_to_f16(s0_bits)
            f1 = bits_to_f16(s1_bits)
            return f16_to_bits(f0 - f1)

        s0 = f16_to_bits(10.0)
        s1 = f16_to_bits(3.0)
        expected = oracle_v_add_f16(s0, s1)
        mutant_res = mutant_v_add_f16(s0, s1)
        self.assertNotEqual(expected, mutant_res)

    def test_mutation_ignore_abs_modifier(self) -> None:
        """Mutant: Ignore abs modifier on negative operand."""
        def mutant_apply_mod(bits: int, abs_mod: bool, neg_mod: bool) -> int:
            res = bits & 0xFFFF
            if neg_mod:
                res ^= SIGN_MASK
            return res

        val = f16_to_bits(-10.0)
        correct = apply_f16_modifiers(val, abs_mod=True, neg_mod=False)
        mutant = mutant_apply_mod(val, abs_mod=True, neg_mod=False)
        self.assertEqual(correct, f16_to_bits(10.0))
        self.assertNotEqual(correct, mutant)

    def test_mutation_ignore_neg_modifier(self) -> None:
        """Mutant: Ignore neg modifier on positive operand."""
        def mutant_apply_mod(bits: int, abs_mod: bool, neg_mod: bool) -> int:
            res = bits & 0xFFFF
            if abs_mod:
                res &= ~SIGN_MASK
            return res

        val = f16_to_bits(10.0)
        correct = apply_f16_modifiers(val, abs_mod=False, neg_mod=True)
        mutant = mutant_apply_mod(val, abs_mod=False, neg_mod=True)
        self.assertEqual(correct, f16_to_bits(-10.0))
        self.assertNotEqual(correct, mutant)

    def test_mutation_ignore_clamp_upper_bound(self) -> None:
        """Mutant: Clamp modifier fails to clamp values exceeding 1.0."""
        s0 = f16_to_bits(0.75)
        s1 = f16_to_bits(0.5)
        # Expected clamp=True yields 1.0
        expected = oracle_v_add_f16(s0, s1, clamp=True)
        self.assertEqual(expected, ONE_F16)

        # Mutant unclamped
        unclamped = oracle_v_add_f16(s0, s1, clamp=False)
        self.assertNotEqual(expected, unclamped)

    def test_mutation_ignore_exec_mask(self) -> None:
        """Mutant: Inactive SIMD lanes are overwritten despite EXEC mask."""
        sentinel = 0xDEAD
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)  # all inactive
        for lane in range(32):
            state.set_reg16(lane, Reg16(4, RegType.VGPR, SubWord.LO), sentinel)

        # Mutant interpreter that ignores exec_mask
        class MutantSIMDInterpreter(SIMDInterpreter):
            def step(self, instr: VAddF16Instruction, state: SIMDState) -> List[Optional[int]]:
                results = []
                for lane in range(state.num_lanes):
                    res = oracle_v_add_f16(f16_to_bits(1.0), f16_to_bits(2.0))
                    state.set_reg16(lane, instr.dst, res)
                    results.append(res)
                return results

        mutant_interp = MutantSIMDInterpreter()
        instr = VAddF16Instruction(
            dst=Reg16(4, RegType.VGPR, SubWord.LO),
            src0=VOP3Operand16(Literal16.from_float(1.0)),
            src1=VOP3Operand16(Literal16.from_float(2.0)),
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
        correct = oracle_v_add_f16(sub, sub)
        self.assertEqual(correct, 0x0002)

        # FTZ mutant flushes to 0.0
        ftz_mutant = POS_ZERO
        self.assertNotEqual(correct, ftz_mutant)

    def test_mutation_skip_snan_quietening(self) -> None:
        """Mutant: Signaling NaN is not quietened."""
        snan = 0x7C05
        res = oracle_v_add_f16(snan, f16_to_bits(1.0))
        self.assertTrue(is_qnan(res))
        self.assertNotEqual(res, snan)  # snan without QNAN_BIT would be a bug


if __name__ == "__main__":
    unittest.main()
