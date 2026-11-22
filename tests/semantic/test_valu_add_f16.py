from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
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


if __name__ == "__main__":
    unittest.main()
