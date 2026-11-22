from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import List, Optional, Tuple, Union

# IEEE 754 binary16 (half-precision) architectural constants
SIGN_MASK_F16: int = 0x8000
EXP_MASK_F16: int = 0x7C00
FRAC_MASK_F16: int = 0x03FF
QNAN_BIT_F16: int = 0x0200
CANONICAL_QNAN_F16: int = 0x7E00

POS_ZERO_F16: int = 0x0000
NEG_ZERO_F16: int = 0x8000
POS_INF_F16: int = 0x7C00
NEG_INF_F16: int = 0xFC00
ONE_F16: int = 0x3C00
HALF_F16: int = 0x3800
TWO_F16: int = 0x4000


def f16_to_bits(val: float) -> int:
    """Convert a Python float to 16-bit IEEE 754 half-precision bits."""
    if math.isnan(val):
        return CANONICAL_QNAN_F16
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return POS_INF_F16 if val > 0 else NEG_INF_F16


def bits_to_f16(bits: int) -> float:
    """Convert 16-bit integer bits to a Python float (IEEE 754 half precision)."""
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def pack_2xf16(lo: int, hi: int) -> int:
    """Pack two 16-bit halfwords into a 32-bit dword."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def unpack_2xf16(dword: int) -> Tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit halfwords (lo, hi)."""
    return dword & 0xFFFF, (dword >> 16) & 0xFFFF


def split_f16_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK_F16) >> 15
    exponent = (bits & EXP_MASK_F16) >> 10
    fraction = bits & FRAC_MASK_F16
    return sign, exponent, fraction


def is_nan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & FRAC_MASK_F16) != 0)


def is_qnan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_snan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """Return True if bits represent IEEE 754 positive or negative infinity."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & FRAC_MASK_F16) == 0)


def is_subnormal_f16(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormal) half-precision float."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & FRAC_MASK_F16) != 0)


def is_zero_f16(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK_F16) == 0


def apply_f16_modifiers(bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply VOP3P input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & 0xFFFF
    if abs_mod:
        res &= ~SIGN_MASK_F16
    if neg_mod:
        res ^= SIGN_MASK_F16
    return res


def oracle_add_f16(
    s0_bits: int,
    s1_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for scalar IEEE 754 half-precision addition with clamp."""
    s0 = apply_f16_modifiers(s0_bits, abs_mod=abs0, neg_mod=neg0)
    s1 = apply_f16_modifiers(s1_bits, abs_mod=abs1, neg_mod=neg1)

    # 1. NaN handling
    if is_nan_f16(s0) or is_nan_f16(s1):
        if clamp:
            # Clamping NaN in RDNA2 produces +0.0
            return POS_ZERO_F16
        # Prefer quietening signaling NaN or propagating quiet NaN
        nan_bits = s0 if is_nan_f16(s0) else s1
        return nan_bits | QNAN_BIT_F16

    # 2. Infinity handling
    s0_inf = is_inf_f16(s0)
    s1_inf = is_inf_f16(s1)
    if s0_inf and s1_inf:
        s0_sign = (s0 & SIGN_MASK_F16)
        s1_sign = (s1 & SIGN_MASK_F16)
        if s0_sign != s1_sign:
            # (+inf) + (-inf) -> Invalid operation -> Canonical Quiet NaN
            return POS_ZERO_F16 if clamp else CANONICAL_QNAN_F16
        if clamp:
            return ONE_F16 if s0 == POS_INF_F16 else POS_ZERO_F16
        return s0
    if s0_inf:
        if clamp:
            return ONE_F16 if s0 == POS_INF_F16 else POS_ZERO_F16
        return s0
    if s1_inf:
        if clamp:
            return ONE_F16 if s1 == POS_INF_F16 else POS_ZERO_F16
        return s1

    # 3. Signed zero handling
    if is_zero_f16(s0) and is_zero_f16(s1):
        s0_is_neg = (s0 == NEG_ZERO_F16)
        s1_is_neg = (s1 == NEG_ZERO_F16)
        if s0_is_neg and s1_is_neg:
            res = NEG_ZERO_F16
        else:
            res = POS_ZERO_F16
        if clamp and res == NEG_ZERO_F16:
            return POS_ZERO_F16
        return res

    # 4. Finite addition
    f0 = bits_to_f16(s0)
    f1 = bits_to_f16(s1)
    f_res = f0 + f1
    res = f16_to_bits(f_res)

    # In round-to-nearest-even, x + (-x) produces +0.0
    if f_res == 0.0 and not (s0 == NEG_ZERO_F16 and s1 == NEG_ZERO_F16):
        res = POS_ZERO_F16

    # 5. Clamp modifier ([0.0, 1.0])
    if clamp:
        if f_res < 0.0 or res == NEG_ZERO_F16:
            res = POS_ZERO_F16
        elif f_res > 1.0 or res == POS_INF_F16:
            res = ONE_F16

    return res


def oracle_v_pk_add_f16(
    src0_dword: int,
    src1_dword: int,
    op_sel: Tuple[int, int] = (0, 0),
    op_sel_hi: Tuple[int, int] = (1, 1),
    neg_lo: Tuple[bool, bool] = (False, False),
    neg_hi: Tuple[bool, bool] = (False, False),
    abs_lo: Tuple[bool, bool] = (False, False),
    abs_hi: Tuple[bool, bool] = (False, False),
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_pk_add_f16 packed addition.
    
    Computes packed 2xf16 addition:
    lo = src0[op_sel[0]] + src1[op_sel[1]]
    hi = src0[op_sel_hi[0]] + src1[op_sel_hi[1]]
    applying neg, abs, and clamp modifiers.
    """
    s0_halves = unpack_2xf16(src0_dword)
    s1_halves = unpack_2xf16(src1_dword)

    s0_lo_in = s0_halves[op_sel[0]]
    s1_lo_in = s1_halves[op_sel[1]]
    res_lo = oracle_add_f16(
        s0_lo_in,
        s1_lo_in,
        abs0=abs_lo[0],
        neg0=neg_lo[0],
        abs1=abs_lo[1],
        neg1=neg_lo[1],
        clamp=clamp,
    )

    s0_hi_in = s0_halves[op_sel_hi[0]]
    s1_hi_in = s1_halves[op_sel_hi[1]]
    res_hi = oracle_add_f16(
        s0_hi_in,
        s1_hi_in,
        abs0=abs_hi[0],
        neg0=neg_hi[0],
        abs1=abs_hi[1],
        neg1=neg_hi[1],
        clamp=clamp,
    )

    return pack_2xf16(res_lo, res_hi)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Reg32:
    """Represents an architectural 32-bit register (VGPR or SGPR)."""
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
    """Represents a 32-bit literal or inline constant holding packed 2xf16 values."""
    dword: int

    @classmethod
    def from_floats(cls, lo: float, hi: float) -> Literal32:
        return cls(pack_2xf16(f16_to_bits(lo), f16_to_bits(hi)))

    @property
    def lo_float(self) -> float:
        return bits_to_f16(self.dword & 0xFFFF)

    @property
    def hi_float(self) -> float:
        return bits_to_f16((self.dword >> 16) & 0xFFFF)

    @property
    def name(self) -> str:
        return f"0x{self.dword:08x}"


@dataclass(frozen=True)
class VOP3POperand:
    """Typed VOP3P operand with op_sel, neg, and abs modifiers."""
    source: Union[Reg32, Literal32]
    op_sel_lo: int = 0  # 0 for lo half, 1 for hi half
    op_sel_hi: int = 1  # 0 for lo half, 1 for hi half
    neg_lo: bool = False
    neg_hi: bool = False
    abs_lo: bool = False
    abs_hi: bool = False

    def __post_init__(self) -> None:
        if self.op_sel_lo not in (0, 1) or self.op_sel_hi not in (0, 1):
            raise ValueError(f"Invalid op_sel values ({self.op_sel_lo}, {self.op_sel_hi}), must be 0 or 1")

    def disassemble(self) -> str:
        s = self.source.name
        mods = []
        if self.abs_lo or self.abs_hi:
            mods.append(f"abs:[{int(self.abs_lo)},{int(self.abs_hi)}]")
        if self.neg_lo or self.neg_hi:
            mods.append(f"neg:[{int(self.neg_lo)},{int(self.neg_hi)}]")
        if self.op_sel_lo != 0 or self.op_sel_hi != 1:
            mods.append(f"op_sel:[{self.op_sel_lo},{self.op_sel_hi}]")
        if mods:
            return f"{s} {' '.join(mods)}"
        return s


@dataclass(frozen=True)
class VPkAddF16Instruction:
    """Symbolic AST node representing a RDNA2 v_pk_add_f16 VOP3P instruction."""
    dst: Reg32
    src0: VOP3POperand
    src1: VOP3POperand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        s = f"v_pk_add_f16 {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}"
        if self.clamp:
            s += " clamp"
        return s


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single wavefront lane (256 VGPRs, 256 SGPRs)."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index] & 0xFFFF_FFFF

    def set_vgpr(self, index: int, value: int) -> None:
        self.vgprs[index] = value & 0xFFFF_FFFF

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & 0xFFFF_FFFF

    def set_sgpr(self, index: int, value: int) -> None:
        self.sgprs[index] = value & 0xFFFF_FFFF


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_pk_add_f16."""

    def resolve_operand(self, op: VOP3POperand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg32):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.dword
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VPkAddF16Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)

        # Computation phase
        result_dword = oracle_v_pk_add_f16(
            src0_dword=s0_raw,
            src1_dword=s1_raw,
            op_sel=(instr.src0.op_sel_lo, instr.src1.op_sel_lo),
            op_sel_hi=(instr.src0.op_sel_hi, instr.src1.op_sel_hi),
            neg_lo=(instr.src0.neg_lo, instr.src1.neg_lo),
            neg_hi=(instr.src0.neg_hi, instr.src1.neg_hi),
            abs_lo=(instr.src0.abs_lo, instr.src1.abs_lo),
            abs_hi=(instr.src0.abs_hi, instr.src1.abs_hi),
            clamp=instr.clamp,
        )

        # Write-back phase
        state.set_vgpr(instr.dst.index, result_dword)
        return result_dword


# ==============================================================================
# SIMD Multi-Lane Execution Engine and EXEC Masking
# ==============================================================================

@dataclass
class SIMDState:
    """Multi-lane SIMD machine state representing Wave32 or Wave64 execution."""
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

    def get_vgpr(self, lane: int, index: int) -> int:
        return self.vgprs[lane][index] & 0xFFFF_FFFF

    def set_vgpr(self, lane: int, index: int, value: int) -> None:
        self.vgprs[lane][index] = value & 0xFFFF_FFFF

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & 0xFFFF_FFFF

    def set_sgpr(self, index: int, value: int) -> None:
        self.sgprs[index] = value & 0xFFFF_FFFF


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VOP3POperand, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, Reg32):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(lane, op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.dword
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VPkAddF16Instruction, state: SIMDState) -> List[Optional[int]]:
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
                res = oracle_v_pk_add_f16(
                    src0_dword=s0_vals[lane],
                    src1_dword=s1_vals[lane],
                    op_sel=(instr.src0.op_sel_lo, instr.src1.op_sel_lo),
                    op_sel_hi=(instr.src0.op_sel_hi, instr.src1.op_sel_hi),
                    neg_lo=(instr.src0.neg_lo, instr.src1.neg_lo),
                    neg_hi=(instr.src0.neg_hi, instr.src1.neg_hi),
                    abs_lo=(instr.src0.abs_lo, instr.src1.abs_lo),
                    abs_hi=(instr.src0.abs_hi, instr.src1.abs_hi),
                    clamp=instr.clamp,
                )
                state.set_vgpr(lane, instr.dst.index, res)
                results[lane] = res

        return results


# ==============================================================================
# Condition Codes, Status Flags, and Carry Generation
# ==============================================================================

@dataclass(frozen=True)
class FPStatusFlags16:
    """Architectural floating-point condition codes and IEEE 754 status flags for 16-bit float."""
    invalid: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False
    zero: bool = False
    negative: bool = False
    denormal: bool = False
    infinite: bool = False
    nan: bool = False


@dataclass(frozen=True)
class PackedFPStatusFlags16:
    """Architectural status flags for both packed 16-bit halves."""
    lo: FPStatusFlags16
    hi: FPStatusFlags16


def evaluate_f16_flags(
    s0_bits: int,
    s1_bits: int,
    res_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> FPStatusFlags16:
    """Evaluate IEEE 754 architectural flags and condition codes for a single f16 lane."""
    s0 = apply_f16_modifiers(s0_bits, abs_mod=abs0, neg_mod=neg0)
    s1 = apply_f16_modifiers(s1_bits, abs_mod=abs1, neg_mod=neg1)

    s0_nan = is_nan_f16(s0)
    s1_nan = is_nan_f16(s1)
    s0_snan = is_snan_f16(s0)
    s1_snan = is_snan_f16(s1)
    s0_inf = is_inf_f16(s0)
    s1_inf = is_inf_f16(s1)

    invalid = False
    if s0_snan or s1_snan:
        invalid = True
    elif s0_inf and s1_inf and ((s0 & SIGN_MASK_F16) != (s1 & SIGN_MASK_F16)):
        invalid = True

    nan_flag = is_nan_f16(res_bits)
    inf_flag = is_inf_f16(res_bits)
    zero_flag = is_zero_f16(res_bits)
    neg_flag = bool(res_bits & SIGN_MASK_F16)
    denorm_flag = is_subnormal_f16(res_bits)

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

    return FPStatusFlags16(
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


def evaluate_v_pk_add_f16_flags(
    src0_dword: int,
    src1_dword: int,
    res_dword: int,
    op_sel: Tuple[int, int] = (0, 0),
    op_sel_hi: Tuple[int, int] = (1, 1),
    neg_lo: Tuple[bool, bool] = (False, False),
    neg_hi: Tuple[bool, bool] = (False, False),
    abs_lo: Tuple[bool, bool] = (False, False),
    abs_hi: Tuple[bool, bool] = (False, False),
    clamp: bool = False,
) -> PackedFPStatusFlags16:
    """Evaluate packed status flags for both low and high 16-bit halves."""
    s0_halves = unpack_2xf16(src0_dword)
    s1_halves = unpack_2xf16(src1_dword)
    res_halves = unpack_2xf16(res_dword)

    lo_flags = evaluate_f16_flags(
        s0_halves[op_sel[0]],
        s1_halves[op_sel[1]],
        res_halves[0],
        abs0=abs_lo[0],
        neg0=neg_lo[0],
        abs1=abs_lo[1],
        neg1=neg_lo[1],
        clamp=clamp,
    )

    hi_flags = evaluate_f16_flags(
        s0_halves[op_sel_hi[0]],
        s1_halves[op_sel_hi[1]],
        res_halves[1],
        abs0=abs_hi[0],
        neg0=neg_hi[0],
        abs1=abs_hi[1],
        neg1=neg_hi[1],
        clamp=clamp,
    )

    return PackedFPStatusFlags16(lo=lo_flags, hi=hi_flags)


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
    """Unit tests for binary16 / packed 2xf16 bit model and reference oracle."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 65504.0, -65504.0]
        for val in values:
            with self.subTest(val=val):
                bits = f16_to_bits(val)
                recovered = bits_to_f16(bits)
                self.assertEqual(val, recovered)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_dword_packing_and_unpacking(self) -> None:
        lo = f16_to_bits(1.5)
        hi = f16_to_bits(-3.0)
        packed = pack_2xf16(lo, hi)
        unpacked_lo, unpacked_hi = unpack_2xf16(packed)
        self.assertEqual(unpacked_lo, lo)
        self.assertEqual(unpacked_hi, hi)
        self.assertEqual(bits_to_f16(unpacked_lo), 1.5)
        self.assertEqual(bits_to_f16(unpacked_hi), -3.0)

    def test_classification_predicates(self) -> None:
        self.assertTrue(f16_to_bits(0.0) == POS_ZERO_F16)
        self.assertTrue(f16_to_bits(-0.0) == NEG_ZERO_F16)
        self.assertTrue(is_zero_f16(POS_ZERO_F16))
        self.assertTrue(is_zero_f16(NEG_ZERO_F16))
        self.assertFalse(is_zero_f16(f16_to_bits(1.0)))

        self.assertTrue(is_inf_f16(POS_INF_F16))
        self.assertTrue(is_inf_f16(NEG_INF_F16))
        self.assertFalse(is_inf_f16(POS_ZERO_F16))

        snan_bits = 0x7C01
        qnan_bits = CANONICAL_QNAN_F16
        self.assertTrue(is_nan_f16(snan_bits))
        self.assertTrue(is_snan_f16(snan_bits))
        self.assertFalse(is_qnan_f16(snan_bits))

        self.assertTrue(is_nan_f16(qnan_bits))
        self.assertTrue(is_qnan_f16(qnan_bits))
        self.assertFalse(is_snan_f16(qnan_bits))

        subnorm_bits = 0x0001
        self.assertTrue(is_subnormal_f16(subnorm_bits))
        self.assertFalse(is_subnormal_f16(POS_ZERO_F16))
        self.assertFalse(is_subnormal_f16(f16_to_bits(1.0)))

    def test_exact_finite_addition(self) -> None:
        # lo: 1.5 + 2.5 = 4.0; hi: 0.25 + 0.75 = 1.0
        s0 = pack_2xf16(f16_to_bits(1.5), f16_to_bits(0.25))
        s1 = pack_2xf16(f16_to_bits(2.5), f16_to_bits(0.75))
        res = oracle_v_pk_add_f16(s0, s1)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(bits_to_f16(r_lo), 4.0)
        self.assertEqual(bits_to_f16(r_hi), 1.0)

    def test_signed_zero_rules(self) -> None:
        self.assertEqual(oracle_add_f16(POS_ZERO_F16, POS_ZERO_F16), POS_ZERO_F16)
        self.assertEqual(oracle_add_f16(NEG_ZERO_F16, NEG_ZERO_F16), NEG_ZERO_F16)
        self.assertEqual(oracle_add_f16(POS_ZERO_F16, NEG_ZERO_F16), POS_ZERO_F16)
        self.assertEqual(oracle_add_f16(NEG_ZERO_F16, POS_ZERO_F16), POS_ZERO_F16)
        self.assertEqual(oracle_add_f16(f16_to_bits(5.0), f16_to_bits(-5.0)), POS_ZERO_F16)

    def test_input_modifiers_abs_neg(self) -> None:
        val_pos = f16_to_bits(4.0)
        val_neg = f16_to_bits(-4.0)
        s0 = pack_2xf16(val_neg, val_pos)
        s1 = pack_2xf16(f16_to_bits(2.0), f16_to_bits(2.0))

        # lo: |-4.0| + 2.0 = 6.0; hi: -(4.0) + 2.0 = -2.0
        res = oracle_v_pk_add_f16(
            s0, s1,
            abs_lo=(True, False),
            neg_hi=(True, False),
        )
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(bits_to_f16(r_lo), 6.0)
        self.assertEqual(bits_to_f16(r_hi), -2.0)

    def test_op_sel_cross_and_broadcast(self) -> None:
        # s0 = [lo: 10.0, hi: 20.0], s1 = [lo: 1.0, hi: 2.0]
        s0 = pack_2xf16(f16_to_bits(10.0), f16_to_bits(20.0))
        s1 = pack_2xf16(f16_to_bits(1.0), f16_to_bits(2.0))

        # Cross add: lo uses s0.hi + s1.lo (20 + 1 = 21), hi uses s0.lo + s1.hi (10 + 2 = 12)
        res_cross = oracle_v_pk_add_f16(s0, s1, op_sel=(1, 0), op_sel_hi=(0, 1))
        r_lo, r_hi = unpack_2xf16(res_cross)
        self.assertEqual(bits_to_f16(r_lo), 21.0)
        self.assertEqual(bits_to_f16(r_hi), 12.0)

        # Broadcast hi: lo and hi both use s0.hi + s1.hi (20 + 2 = 22)
        res_bcast = oracle_v_pk_add_f16(s0, s1, op_sel=(1, 1), op_sel_hi=(1, 1))
        b_lo, b_hi = unpack_2xf16(res_bcast)
        self.assertEqual(bits_to_f16(b_lo), 22.0)
        self.assertEqual(bits_to_f16(b_hi), 22.0)

    def test_clamp_modifier(self) -> None:
        # lo: 0.5 + 0.8 = 1.3 -> clamped to 1.0
        # hi: -1.0 + 0.4 = -0.6 -> clamped to 0.0
        s0 = pack_2xf16(f16_to_bits(0.5), f16_to_bits(-1.0))
        s1 = pack_2xf16(f16_to_bits(0.8), f16_to_bits(0.4))
        res = oracle_v_pk_add_f16(s0, s1, clamp=True)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(r_lo, ONE_F16)
        self.assertEqual(r_hi, POS_ZERO_F16)

    def test_infinity_and_nan_addition(self) -> None:
        self.assertEqual(oracle_add_f16(POS_INF_F16, f16_to_bits(1.0)), POS_INF_F16)
        self.assertEqual(oracle_add_f16(NEG_INF_F16, f16_to_bits(1.0)), NEG_INF_F16)
        self.assertEqual(oracle_add_f16(POS_INF_F16, NEG_INF_F16), CANONICAL_QNAN_F16)

        snan = 0x7C05
        res_nan = oracle_add_f16(snan, f16_to_bits(1.0))
        self.assertTrue(is_qnan_f16(res_nan))
        self.assertEqual(res_nan, snan | QNAN_BIT_F16)

        # Clamping NaN yields +0.0
        res_clamped_nan = oracle_add_f16(snan, f16_to_bits(1.0), clamp=True)
        self.assertEqual(res_clamped_nan, POS_ZERO_F16)


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_reg32_validation_and_naming(self) -> None:
        v0 = Reg32(0, RegType.VGPR)
        self.assertEqual(v0.name, "v0")
        self.assertEqual(v0.index, 0)

        s15 = Reg32(15, RegType.SGPR)
        self.assertEqual(s15.name, "s15")
        self.assertEqual(s15.index, 15)

        with self.assertRaises(ValueError):
            Reg32(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(-1, RegType.VGPR)

    def test_literal32_representation(self) -> None:
        lit = Literal32.from_floats(1.0, 2.0)
        self.assertAlmostEqual(lit.lo_float, 1.0)
        self.assertAlmostEqual(lit.hi_float, 2.0)
        self.assertTrue(lit.name.startswith("0x"))

    def test_vop3p_operand_disassembly(self) -> None:
        v1 = Reg32(1, RegType.VGPR)
        op_plain = VOP3POperand(v1)
        self.assertEqual(op_plain.disassemble(), "v1")

        op_mods = VOP3POperand(v1, op_sel_lo=1, op_sel_hi=0, neg_lo=True, abs_hi=True)
        self.assertIn("abs:[0,1]", op_mods.disassemble())
        self.assertIn("neg:[1,0]", op_mods.disassemble())
        self.assertIn("op_sel:[1,0]", op_mods.disassemble())

        with self.assertRaises(ValueError):
            VOP3POperand(v1, op_sel_lo=2)

    def test_v_pk_add_f16_instruction_ast(self) -> None:
        dst = Reg32(0, RegType.VGPR)
        src0 = VOP3POperand(Reg32(2, RegType.VGPR), neg_lo=True)
        src1 = VOP3POperand(Reg32(4, RegType.SGPR))
        instr = VPkAddF16Instruction(dst=dst, src0=src0, src1=src1, clamp=True)
        dis = instr.disassemble()
        self.assertTrue(dis.startswith("v_pk_add_f16 v0, v2 neg:[1,0], s4 clamp"))

        # Destination must be VGPR
        with self.assertRaises(TypeError):
            VPkAddF16Instruction(dst=Reg32(0, RegType.SGPR), src0=src0, src1=src1)

    def test_structural_equality(self) -> None:
        i1 = VPkAddF16Instruction(
            Reg32(0, RegType.VGPR),
            VOP3POperand(Reg32(1, RegType.VGPR)),
            VOP3POperand(Literal32(0)),
        )
        i2 = VPkAddF16Instruction(
            Reg32(0, RegType.VGPR),
            VOP3POperand(Reg32(1, RegType.VGPR)),
            VOP3POperand(Literal32(0)),
        )
        self.assertEqual(i1, i2)


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_scalar_vgpr_vgpr_addition(self) -> None:
        # v2 = [lo: 2.0, hi: 3.0], v4 = [lo: 0.5, hi: 1.5]
        s0 = pack_2xf16(f16_to_bits(2.0), f16_to_bits(3.0))
        s1 = pack_2xf16(f16_to_bits(0.5), f16_to_bits(1.5))
        self.state.set_vgpr(2, s0)
        self.state.set_vgpr(4, s1)

        instr = VPkAddF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(2, RegType.VGPR)),
            src1=VOP3POperand(Reg32(4, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(bits_to_f16(r_lo), 2.5)
        self.assertEqual(bits_to_f16(r_hi), 4.5)
        self.assertEqual(self.state.get_vgpr(0), res)

    def test_scalar_vgpr_sgpr_addition(self) -> None:
        # v1 = [10.0, 20.0], s8 = [1.0, 2.0]
        self.state.set_vgpr(1, pack_2xf16(f16_to_bits(10.0), f16_to_bits(20.0)))
        self.state.set_sgpr(8, pack_2xf16(f16_to_bits(1.0), f16_to_bits(2.0)))

        instr = VPkAddF16Instruction(
            dst=Reg32(3, RegType.VGPR),
            src0=VOP3POperand(Reg32(1, RegType.VGPR)),
            src1=VOP3POperand(Reg32(8, RegType.SGPR)),
        )
        res = self.interp.step(instr, self.state)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(bits_to_f16(r_lo), 11.0)
        self.assertEqual(bits_to_f16(r_hi), 22.0)
        self.assertEqual(self.state.get_vgpr(3), res)

    def test_scalar_vgpr_literal_addition(self) -> None:
        self.state.set_vgpr(5, pack_2xf16(f16_to_bits(5.0), f16_to_bits(15.0)))
        lit = Literal32.from_floats(2.0, 3.0)

        instr = VPkAddF16Instruction(
            dst=Reg32(6, RegType.VGPR),
            src0=VOP3POperand(Reg32(5, RegType.VGPR)),
            src1=VOP3POperand(lit),
        )
        res = self.interp.step(instr, self.state)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(bits_to_f16(r_lo), 7.0)
        self.assertEqual(bits_to_f16(r_hi), 18.0)
        self.assertEqual(self.state.get_vgpr(6), res)

    def test_scalar_with_modifiers_and_clamp(self) -> None:
        # v0 = [lo: -0.5, hi: 1.2], v2 = [lo: 0.2, hi: 0.3]
        # lo: -|-0.5| + 0.2 = -0.5 + 0.2 = -0.3 -> clamped to +0.0
        # hi: 1.2 + 0.3 = 1.5 -> clamped to 1.0
        self.state.set_vgpr(0, pack_2xf16(f16_to_bits(-0.5), f16_to_bits(1.2)))
        self.state.set_vgpr(2, pack_2xf16(f16_to_bits(0.2), f16_to_bits(0.3)))

        instr = VPkAddF16Instruction(
            dst=Reg32(4, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR), abs_lo=True, neg_lo=True),
            src1=VOP3POperand(Reg32(2, RegType.VGPR)),
            clamp=True,
        )
        res = self.interp.step(instr, self.state)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(r_lo, POS_ZERO_F16)
        self.assertEqual(r_hi, ONE_F16)

    def test_scalar_determinism(self) -> None:
        runs = []
        for _ in range(30):
            st = ScalarMachineState()
            st.set_vgpr(0, pack_2xf16(0x3C00, 0x4000))
            st.set_vgpr(1, pack_2xf16(0x3800, 0x4200))
            instr = VPkAddF16Instruction(
                dst=Reg32(2, RegType.VGPR),
                src0=VOP3POperand(Reg32(0, RegType.VGPR)),
                src1=VOP3POperand(Reg32(1, RegType.VGPR)),
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
            s0 = pack_2xf16(f16_to_bits(float(lane)), f16_to_bits(float(lane) * 2.0))
            s1 = pack_2xf16(f16_to_bits(1.0), f16_to_bits(2.0))
            state.set_vgpr(lane, 0, s0)
            state.set_vgpr(lane, 1, s1)

        instr = VPkAddF16Instruction(
            dst=Reg32(2, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(1, RegType.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            r_lo, r_hi = unpack_2xf16(state.get_vgpr(lane, 2))
            self.assertEqual(bits_to_f16(r_lo), float(lane) + 1.0)
            self.assertEqual(bits_to_f16(r_hi), float(lane) * 2.0 + 2.0)

    def test_wave32_divergent_masking(self) -> None:
        sentinel = 0xDEAD_BEEF
        state = SIMDState(num_lanes=32, exec_mask=0x5555_5555)  # even lanes active
        for lane in range(32):
            s0 = pack_2xf16(f16_to_bits(float(lane)), f16_to_bits(10.0))
            s1 = pack_2xf16(f16_to_bits(1.0), f16_to_bits(5.0))
            state.set_vgpr(lane, 0, s0)
            state.set_vgpr(lane, 1, s1)
            state.set_vgpr(lane, 2, sentinel)

        instr = VPkAddF16Instruction(
            dst=Reg32(2, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(1, RegType.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertIsNotNone(results[lane])
                r_lo, r_hi = unpack_2xf16(state.get_vgpr(lane, 2))
                self.assertEqual(bits_to_f16(r_lo), float(lane) + 1.0)
                self.assertEqual(bits_to_f16(r_hi), 15.0)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 2), sentinel)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr(lane, 0, pack_2xf16(f16_to_bits(float(lane)), f16_to_bits(0.5)))
            state.set_vgpr(lane, 1, pack_2xf16(f16_to_bits(2.0), f16_to_bits(1.5)))

        instr = VPkAddF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(1, RegType.VGPR)),
        )
        self.interp.step(instr, state)
        for lane in range(64):
            r_lo, r_hi = unpack_2xf16(state.get_vgpr(lane, 0))
            self.assertEqual(bits_to_f16(r_lo), float(lane) + 2.0)
            self.assertEqual(bits_to_f16(r_hi), 2.0)

    def test_wave64_half_disabled(self) -> None:
        sentinel = 0xCAFE_BABE
        state = SIMDState(num_lanes=64, exec_mask=0x0000_0000_FFFF_FFFF)  # lower 32 lanes
        for lane in range(64):
            state.set_vgpr(lane, 0, pack_2xf16(f16_to_bits(4.0), f16_to_bits(6.0)))
            state.set_vgpr(lane, 1, pack_2xf16(f16_to_bits(1.0), f16_to_bits(2.0)))
            state.set_vgpr(lane, 2, sentinel)

        instr = VPkAddF16Instruction(
            dst=Reg32(2, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(1, RegType.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(64):
            if lane < 32:
                self.assertIsNotNone(results[lane])
                r_lo, r_hi = unpack_2xf16(state.get_vgpr(lane, 2))
                self.assertEqual(bits_to_f16(r_lo), 5.0)
                self.assertEqual(bits_to_f16(r_hi), 8.0)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 2), sentinel)

    def test_wave_exec_all_zero(self) -> None:
        sentinel = 0x1234_5678
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)
        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)

        instr = VPkAddF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Literal32.from_floats(1.0, 1.0)),
            src1=VOP3POperand(Literal32.from_floats(2.0, 2.0)),
        )
        results = self.interp.step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_simd_sgpr_uniform_broadcast(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        state.set_sgpr(5, pack_2xf16(f16_to_bits(100.0), f16_to_bits(200.0)))
        for lane in range(32):
            state.set_vgpr(lane, 0, pack_2xf16(f16_to_bits(float(lane)), f16_to_bits(float(lane))))

        instr = VPkAddF16Instruction(
            dst=Reg32(1, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(5, RegType.SGPR)),
        )
        self.interp.step(instr, state)
        for lane in range(32):
            r_lo, r_hi = unpack_2xf16(state.get_vgpr(lane, 1))
            self.assertEqual(bits_to_f16(r_lo), 100.0 + float(lane))
            self.assertEqual(bits_to_f16(r_hi), 200.0 + float(lane))


class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src0(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, pack_2xf16(f16_to_bits(1.5), f16_to_bits(2.5)))
        state.set_vgpr(1, pack_2xf16(f16_to_bits(0.5), f16_to_bits(1.0)))

        instr = VPkAddF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(1, RegType.VGPR)),
        )
        self.scalar_interp.step(instr, state)
        r_lo, r_hi = unpack_2xf16(state.get_vgpr(0))
        self.assertEqual(bits_to_f16(r_lo), 2.0)
        self.assertEqual(bits_to_f16(r_hi), 3.5)
        # src1 untouched
        s1_lo, s1_hi = unpack_2xf16(state.get_vgpr(1))
        self.assertEqual(bits_to_f16(s1_lo), 0.5)
        self.assertEqual(bits_to_f16(s1_hi), 1.0)

    def test_dst_alias_src1(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, pack_2xf16(f16_to_bits(1.5), f16_to_bits(2.5)))
        state.set_vgpr(1, pack_2xf16(f16_to_bits(0.5), f16_to_bits(1.0)))

        instr = VPkAddF16Instruction(
            dst=Reg32(1, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(1, RegType.VGPR)),
        )
        self.scalar_interp.step(instr, state)
        r_lo, r_hi = unpack_2xf16(state.get_vgpr(1))
        self.assertEqual(bits_to_f16(r_lo), 2.0)
        self.assertEqual(bits_to_f16(r_hi), 3.5)
        # src0 untouched
        s0_lo, s0_hi = unpack_2xf16(state.get_vgpr(0))
        self.assertEqual(bits_to_f16(s0_lo), 1.5)
        self.assertEqual(bits_to_f16(s0_hi), 2.5)

    def test_dst_alias_both_srcs(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, pack_2xf16(f16_to_bits(3.0), f16_to_bits(4.0)))

        instr = VPkAddF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(0, RegType.VGPR)),
        )
        self.scalar_interp.step(instr, state)
        r_lo, r_hi = unpack_2xf16(state.get_vgpr(0))
        self.assertEqual(bits_to_f16(r_lo), 6.0)
        self.assertEqual(bits_to_f16(r_hi), 8.0)

    def test_inplace_iterative_accumulation(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, pack_2xf16(f16_to_bits(0.0), f16_to_bits(0.0)))
        state.set_vgpr(1, pack_2xf16(f16_to_bits(0.5), f16_to_bits(1.0)))

        instr = VPkAddF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(1, RegType.VGPR)),
        )
        for _ in range(40):
            self.scalar_interp.step(instr, state)
        r_lo, r_hi = unpack_2xf16(state.get_vgpr(0))
        self.assertEqual(bits_to_f16(r_lo), 20.0)
        self.assertEqual(bits_to_f16(r_hi), 40.0)

    def test_non_aliased_registers_preserved(self) -> None:
        state = ScalarMachineState()
        sentinel_v2 = 0x1111_2222
        sentinel_v3 = 0x3333_4444
        state.set_vgpr(0, pack_2xf16(f16_to_bits(1.0), f16_to_bits(1.0)))
        state.set_vgpr(1, pack_2xf16(f16_to_bits(2.0), f16_to_bits(2.0)))
        state.set_vgpr(2, sentinel_v2)
        state.set_vgpr(3, sentinel_v3)

        instr = VPkAddF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(1, RegType.VGPR)),
        )
        self.scalar_interp.step(instr, state)
        self.assertEqual(state.get_vgpr(2), sentinel_v2)
        self.assertEqual(state.get_vgpr(3), sentinel_v3)

    def test_simd_multi_lane_aliasing(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr(lane, 0, pack_2xf16(f16_to_bits(float(lane)), f16_to_bits(1.0)))
            state.set_vgpr(lane, 1, pack_2xf16(f16_to_bits(2.0), f16_to_bits(3.0)))

        instr = VPkAddF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Reg32(0, RegType.VGPR)),
            src1=VOP3POperand(Reg32(1, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            r_lo, r_hi = unpack_2xf16(state.get_vgpr(lane, 0))
            self.assertEqual(bits_to_f16(r_lo), float(lane) + 2.0)
            self.assertEqual(bits_to_f16(r_hi), 4.0)


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for status flags, condition codes, and carry generation."""

    def test_exact_finite_flags(self) -> None:
        s0 = pack_2xf16(f16_to_bits(1.0), f16_to_bits(2.0))
        s1 = pack_2xf16(f16_to_bits(2.0), f16_to_bits(3.0))
        res = oracle_v_pk_add_f16(s0, s1)
        flags = evaluate_v_pk_add_f16_flags(s0, s1, res)

        self.assertFalse(flags.lo.invalid)
        self.assertFalse(flags.lo.overflow)
        self.assertFalse(flags.lo.underflow)
        self.assertFalse(flags.lo.inexact)
        self.assertFalse(flags.lo.zero)
        self.assertFalse(flags.lo.negative)

        self.assertFalse(flags.hi.invalid)
        self.assertFalse(flags.hi.overflow)
        self.assertFalse(flags.hi.underflow)
        self.assertFalse(flags.hi.inexact)
        self.assertFalse(flags.hi.zero)
        self.assertFalse(flags.hi.negative)

    def test_inexact_flag(self) -> None:
        # In half-precision, 1.0 + 2^-11 has a fractional bit beyond 10-bit precision
        ulp_half = 2.0 ** -11
        s0 = pack_2xf16(f16_to_bits(1.0), f16_to_bits(0.0))
        s1 = pack_2xf16(f16_to_bits(ulp_half), f16_to_bits(0.0))
        res = oracle_v_pk_add_f16(s0, s1)
        flags = evaluate_v_pk_add_f16_flags(s0, s1, res)
        self.assertTrue(flags.lo.inexact)
        self.assertFalse(flags.hi.inexact)

    def test_overflow_flag(self) -> None:
        max_f16 = 0x7BFF  # 65504.0
        s0 = pack_2xf16(max_f16, f16_to_bits(1.0))
        s1 = pack_2xf16(max_f16, f16_to_bits(1.0))
        res = oracle_v_pk_add_f16(s0, s1)
        flags = evaluate_v_pk_add_f16_flags(s0, s1, res)
        self.assertTrue(flags.lo.overflow)
        self.assertTrue(flags.lo.infinite)
        self.assertTrue(flags.lo.inexact)
        self.assertFalse(flags.hi.overflow)

    def test_underflow_and_denorm_flags(self) -> None:
        subnorm = 0x0001
        s0 = pack_2xf16(subnorm, f16_to_bits(1.0))
        s1 = pack_2xf16(subnorm, f16_to_bits(1.0))
        res = oracle_v_pk_add_f16(s0, s1)
        flags = evaluate_v_pk_add_f16_flags(s0, s1, res)
        self.assertTrue(flags.lo.denormal)
        self.assertTrue(flags.lo.underflow)
        self.assertFalse(flags.hi.denormal)

    def test_invalid_operation_flag(self) -> None:
        s0 = pack_2xf16(POS_INF_F16, 0x7C01)  # lo: +inf, hi: signaling NaN
        s1 = pack_2xf16(NEG_INF_F16, f16_to_bits(1.0))
        res = oracle_v_pk_add_f16(s0, s1)
        flags = evaluate_v_pk_add_f16_flags(s0, s1, res)
        self.assertTrue(flags.lo.invalid)  # +inf + -inf
        self.assertTrue(flags.hi.invalid)  # signaling NaN

    def test_zero_and_negative_flags(self) -> None:
        s0 = pack_2xf16(POS_ZERO_F16, NEG_ZERO_F16)
        s1 = pack_2xf16(POS_ZERO_F16, NEG_ZERO_F16)
        res = oracle_v_pk_add_f16(s0, s1)
        flags = evaluate_v_pk_add_f16_flags(s0, s1, res)
        self.assertTrue(flags.lo.zero)
        self.assertFalse(flags.lo.negative)
        self.assertTrue(flags.hi.zero)
        self.assertTrue(flags.hi.negative)

    def test_mantissa_carry_generation(self) -> None:
        # 1.0 + 1.0: mantissas 1.000 + 1.000 = 2.000 -> carry_out = 1
        _, carry1 = compute_f16_mantissa_carry(f16_to_bits(1.0), f16_to_bits(1.0))
        self.assertEqual(carry1, 1)

        # 1.0 + 0.25: mantissas sum < 2.0 -> carry_out = 0
        _, carry0 = compute_f16_mantissa_carry(f16_to_bits(1.0), f16_to_bits(0.25))
        self.assertEqual(carry0, 0)


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_addition_without_carry(self) -> None:
        sub1 = 0x0001
        sub2 = 0x0002
        s0 = pack_2xf16(sub1, sub2)
        s1 = pack_2xf16(sub2, sub1)
        res = oracle_v_pk_add_f16(s0, s1)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(r_lo, 0x0003)
        self.assertEqual(r_hi, 0x0003)
        self.assertTrue(is_subnormal_f16(r_lo))
        self.assertTrue(is_subnormal_f16(r_hi))

    def test_subnormal_to_normal_transition(self) -> None:
        # Max subnormal: 0x03FF; min subnormal: 0x0001
        # Sum = 0x0400 = min normal (2^-14)
        max_sub = 0x03FF
        min_sub = 0x0001
        s0 = pack_2xf16(max_sub, min_sub)
        s1 = pack_2xf16(min_sub, max_sub)
        res = oracle_v_pk_add_f16(s0, s1)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(r_lo, 0x0400)
        self.assertEqual(r_hi, 0x0400)
        self.assertFalse(is_subnormal_f16(r_lo))
        self.assertEqual(split_f16_bits(r_lo)[1], 1)  # exponent = 1
        self.assertEqual(split_f16_bits(r_lo)[2], 0)  # fraction = 0

    def test_subnormal_cancellation(self) -> None:
        sub = 0x0042
        neg_sub = sub | SIGN_MASK_F16
        s0 = pack_2xf16(sub, neg_sub)
        s1 = pack_2xf16(neg_sub, sub)
        res = oracle_v_pk_add_f16(s0, s1)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(r_lo, POS_ZERO_F16)
        self.assertEqual(r_hi, POS_ZERO_F16)

    def test_nan_signaling_to_quiet_conversion(self) -> None:
        snan = 0x7C05  # signaling NaN: exp = 0x1F, fraction != 0, quiet bit 9 is 0
        self.assertTrue(is_snan_f16(snan))
        s0 = pack_2xf16(snan, f16_to_bits(1.0))
        s1 = pack_2xf16(f16_to_bits(2.0), snan)
        res = oracle_v_pk_add_f16(s0, s1)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertTrue(is_qnan_f16(r_lo))
        self.assertTrue(is_qnan_f16(r_hi))
        self.assertEqual(r_lo, snan | QNAN_BIT_F16)
        self.assertEqual(r_hi, snan | QNAN_BIT_F16)

    def test_nan_payload_preservation(self) -> None:
        payload = 0x7E35  # Quiet NaN with payload 0x035
        s0 = pack_2xf16(payload, f16_to_bits(0.0))
        s1 = pack_2xf16(f16_to_bits(5.0), payload)
        res = oracle_v_pk_add_f16(s0, s1)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(r_lo, payload)
        self.assertEqual(r_hi, payload)

    def test_nan_with_modifiers(self) -> None:
        qnan = 0x7E01
        # abs should clear sign bit
        s0 = pack_2xf16(qnan | SIGN_MASK_F16, qnan)
        s1 = pack_2xf16(f16_to_bits(0.0), f16_to_bits(0.0))
        res = oracle_v_pk_add_f16(s0, s1, abs_lo=(True, False), neg_hi=(True, False))
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(r_lo & SIGN_MASK_F16, 0)
        self.assertEqual(r_hi & SIGN_MASK_F16, SIGN_MASK_F16)

    def test_nan_with_clamp(self) -> None:
        qnan = CANONICAL_QNAN_F16
        s0 = pack_2xf16(qnan, qnan)
        s1 = pack_2xf16(f16_to_bits(1.0), f16_to_bits(2.0))
        res = oracle_v_pk_add_f16(s0, s1, clamp=True)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(r_lo, POS_ZERO_F16)
        self.assertEqual(r_hi, POS_ZERO_F16)

    def test_infinity_arithmetic_boundaries(self) -> None:
        s0 = pack_2xf16(POS_INF_F16, NEG_INF_F16)
        s1 = pack_2xf16(POS_INF_F16, NEG_INF_F16)
        res = oracle_v_pk_add_f16(s0, s1)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertEqual(r_lo, POS_INF_F16)
        self.assertEqual(r_hi, NEG_INF_F16)

        # (+inf) + (-inf) -> Canonical NaN
        s_cross0 = pack_2xf16(POS_INF_F16, NEG_INF_F16)
        s_cross1 = pack_2xf16(NEG_INF_F16, POS_INF_F16)
        res_cross = oracle_v_pk_add_f16(s_cross0, s_cross1)
        rc_lo, rc_hi = unpack_2xf16(res_cross)
        self.assertEqual(rc_lo, CANONICAL_QNAN_F16)
        self.assertEqual(rc_hi, CANONICAL_QNAN_F16)

        # Infinity with clamp
        res_clamped = oracle_v_pk_add_f16(s0, s1, clamp=True)
        rcl_lo, rcl_hi = unpack_2xf16(res_clamped)
        self.assertEqual(rcl_lo, ONE_F16)
        self.assertEqual(rcl_hi, POS_ZERO_F16)

    def test_round_to_nearest_even_ties(self) -> None:
        # 1.0 is 0x3C00 (even LSB)
        # 1.0 + 2^-10 is 0x3C01 (odd LSB)
        # 2^-11 is halfway between representable numbers
        half_ulp = 2.0 ** -11
        one = 1.0
        one_plus_ulp = bits_to_f16(0x3C01)

        # Case 1: 1.0 (even LSB) + 2^-11: tie rounds to even (1.0 = 0x3C00)
        res1 = oracle_add_f16(f16_to_bits(one), f16_to_bits(half_ulp))
        self.assertEqual(res1, 0x3C00)

        # Case 2: (1.0 + 2^-10) (odd LSB) + 2^-11: tie rounds to even (1.0 + 2^-9 = 0x3C02)
        res2 = oracle_add_f16(f16_to_bits(one_plus_ulp), f16_to_bits(half_ulp))
        self.assertEqual(res2, 0x3C02)

    def test_dual_half_independence(self) -> None:
        # lo is NaN, hi is exact finite addition
        s0 = pack_2xf16(CANONICAL_QNAN_F16, f16_to_bits(3.0))
        s1 = pack_2xf16(f16_to_bits(1.0), f16_to_bits(4.0))
        res = oracle_v_pk_add_f16(s0, s1)
        r_lo, r_hi = unpack_2xf16(res)
        self.assertTrue(is_qnan_f16(r_lo))
        self.assertEqual(bits_to_f16(r_hi), 7.0)


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_negative_control_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Reg32(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(-5, RegType.SGPR)
        with self.assertRaises(ValueError):
            Reg32(300, RegType.SGPR)

    def test_negative_control_sgpr_destination_rejected(self) -> None:
        with self.assertRaises(TypeError):
            VPkAddF16Instruction(
                dst=Reg32(0, RegType.SGPR),
                src0=VOP3POperand(Reg32(1, RegType.VGPR)),
                src1=VOP3POperand(Reg32(2, RegType.VGPR)),
            )

    def test_negative_control_invalid_op_sel(self) -> None:
        v = Reg32(0, RegType.VGPR)
        with self.assertRaises(ValueError):
            VOP3POperand(v, op_sel_lo=-1)
        with self.assertRaises(ValueError):
            VOP3POperand(v, op_sel_hi=2)

    def test_negative_control_invalid_simd_lane_count(self) -> None:
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=48)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_negative_control_unsupported_operand_type(self) -> None:
        state = ScalarMachineState()
        dummy_op = VOP3POperand(source=None)  # type: ignore
        with self.assertRaises(TypeError):
            self.interp.resolve_operand(dummy_op, state)

    def test_mutation_detection_subtraction_vs_addition(self) -> None:
        s0 = pack_2xf16(f16_to_bits(10.0), f16_to_bits(20.0))
        s1 = pack_2xf16(f16_to_bits(4.0), f16_to_bits(5.0))
        oracle_res = oracle_v_pk_add_f16(s0, s1)

        # Mutate to subtraction
        mutated_lo = f16_to_bits(10.0 - 4.0)
        mutated_hi = f16_to_bits(20.0 - 5.0)
        mutated_res = pack_2xf16(mutated_lo, mutated_hi)
        self.assertNotEqual(oracle_res, mutated_res)

    def test_mutation_detection_clamp_omission(self) -> None:
        # Results exceed 1.0
        s0 = pack_2xf16(f16_to_bits(0.8), f16_to_bits(0.9))
        s1 = pack_2xf16(f16_to_bits(0.5), f16_to_bits(0.5))
        clamped_oracle = oracle_v_pk_add_f16(s0, s1, clamp=True)
        unclamped = oracle_v_pk_add_f16(s0, s1, clamp=False)
        self.assertNotEqual(clamped_oracle, unclamped)
        c_lo, c_hi = unpack_2xf16(clamped_oracle)
        self.assertEqual(c_lo, ONE_F16)
        self.assertEqual(c_hi, ONE_F16)

    def test_mutation_detection_op_sel_swapping(self) -> None:
        s0 = pack_2xf16(f16_to_bits(1.0), f16_to_bits(50.0))
        s1 = pack_2xf16(f16_to_bits(2.0), f16_to_bits(100.0))
        res_default = oracle_v_pk_add_f16(s0, s1, op_sel=(0, 0), op_sel_hi=(1, 1))
        res_swapped = oracle_v_pk_add_f16(s0, s1, op_sel=(1, 1), op_sel_hi=(0, 0))
        self.assertNotEqual(res_default, res_swapped)

    def test_mutation_detection_negation_omission(self) -> None:
        s0 = pack_2xf16(f16_to_bits(5.0), f16_to_bits(5.0))
        s1 = pack_2xf16(f16_to_bits(2.0), f16_to_bits(2.0))
        res_neg = oracle_v_pk_add_f16(s0, s1, neg_lo=(True, False))
        res_no_neg = oracle_v_pk_add_f16(s0, s1, neg_lo=(False, False))
        self.assertNotEqual(res_neg, res_no_neg)
        r_lo, _ = unpack_2xf16(res_neg)
        self.assertEqual(bits_to_f16(r_lo), -3.0)  # -5.0 + 2.0 = -3.0

    def test_mutation_detection_nan_quietening_omission(self) -> None:
        snan = 0x7C03
        res = oracle_add_f16(snan, f16_to_bits(1.0))
        # Omitting quietening keeps snan untouched, which must diverge from oracle
        self.assertNotEqual(res, snan)
        self.assertEqual(res, snan | QNAN_BIT_F16)

    def test_mutation_detection_inactive_lane_write(self) -> None:
        sentinel = 0x55AA_55AA
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0001)  # only lane 0 active
        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)

        instr = VPkAddF16Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3POperand(Literal32.from_floats(1.0, 1.0)),
            src1=VOP3POperand(Literal32.from_floats(1.0, 1.0)),
        )
        self.simd_interp.step(instr, state)
        self.assertNotEqual(state.get_vgpr(0, 0), sentinel)
        # Inactive lanes must remain exactly sentinel
        for lane in range(1, 32):
            self.assertEqual(state.get_vgpr(lane, 0), sentinel)


if __name__ == "__main__":
    unittest.main()
