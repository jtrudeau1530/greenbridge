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
    return struct.unpack("<H", struct.pack("<e", val))[0]


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


if __name__ == "__main__":
    unittest.main()
