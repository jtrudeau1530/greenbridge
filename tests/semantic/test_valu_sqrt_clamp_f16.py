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
FOUR_F16: int = 0x4400
NINE_F16: int = 0x4880
TWO_F16: int = 0x4000
THREE_F16: int = 0x4200
HALF_F16: int = 0x3800
QUARTER_F16: int = 0x3400


def f16_to_bits(val: float) -> int:
    """Convert a Python float to 16-bit binary16 integer bits."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f16(bits: int) -> float:
    """Convert 16-bit binary16 integer bits to a Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def split_f16_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK) >> 15
    exponent = (bits & EXP_MASK) >> 10
    fraction = bits & FRAC_MASK
    return sign, exponent, fraction


def dword_to_f16_halves(dword: int) -> Tuple[int, int]:
    """Split a 32-bit VGPR dword into (lo, hi) 16-bit halfwords."""
    return dword & 0xFFFF, (dword >> 16) & 0xFFFF


def f16_halves_to_dword(lo: int, hi: int) -> int:
    """Combine two 16-bit halfwords (lo, hi) into a 32-bit VGPR dword."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def is_nan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision NaN."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) != 0)


def is_qnan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision Quiet NaN."""
    return is_nan(bits) and bool(bits & QNAN_BIT)


def is_snan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision Signaling NaN."""
    return is_nan(bits) and not bool(bits & QNAN_BIT)


def is_inf(bits: int) -> bool:
    """Return True if bits represent positive or negative infinity."""
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


def ref_sqrt_clamp_f16(
    src0_bits: int,
    abs0: bool = False,
    neg0: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_sqrt_clamp_f16.

    Computes D = clamp(sqrt(src0), 0.0, 1.0) conforming to IEEE 754-2008 binary16
    and AMD RDNA2 ISA VALU clamping rules:
      - Clamping NaN results to +0.0 (0x0000).
      - Clamping negative square roots (invalid operation) to +0.0 (0x0000).
      - Clamping negative zero (-0.0) to +0.0 (0x0000).
      - Clamping results > 1.0 (including +inf) to +1.0 (0x3C00).
      - Preserving in-range values in [0.0, 1.0].
    """
    s0 = apply_f16_modifiers(src0_bits, abs0, neg0)

    # 1. NaN inputs clamp to +0.0
    if is_nan(s0):
        return POS_ZERO

    # 2. Negative inputs (excluding -0.0) produce NaN in IEEE 754, clamped to +0.0
    if (s0 & SIGN_MASK) != 0 and s0 != NEG_ZERO:
        return POS_ZERO

    # 3. Signed zeros: sqrt(+0) = +0, sqrt(-0) = -0; clamping forces -0.0 to +0.0
    if is_zero(s0):
        return POS_ZERO

    # 4. Positive infinity: sqrt(+inf) = +inf; clamped to 1.0
    if is_inf(s0):
        return ONE_F16

    # 5. Finite positive square root with [0.0, 1.0] saturation
    f0 = bits_to_f16(s0)
    res_f = math.sqrt(f0)
    if res_f > 1.0:
        return ONE_F16
    if res_f < 0.0:
        return POS_ZERO
    return f16_to_bits(res_f)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


class HalfSelect(Enum):
    LO = 0  # bits [15:0]
    HI = 1  # bits [31:16]


@dataclass(frozen=True)
class RegHalf:
    """Represents an architectural 16-bit half-register (e.g., v0.l, v0.h, s4.l)."""
    index: int
    reg_type: RegType
    half: HalfSelect = HalfSelect.LO

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index {self.index} out of range [0, 255]")

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        suffix = ".l" if self.half == HalfSelect.LO else ".h"
        return f"{prefix}{self.index}{suffix}"


@dataclass(frozen=True)
class Literal16:
    """Represents a 16-bit literal or inline float constant."""
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
class VOP3Operand:
    """Typed expression operand with optional input modifiers (abs, neg)."""
    source: Union[RegHalf, Literal16]
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
class VSqrtClampF16Instruction:
    """Symbolic AST node representing a RDNA2 v_sqrt_clamp_f16 instruction."""
    dst: RegHalf
    src0: VOP3Operand
    clamp: bool = True

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        s = f"v_sqrt_f16 {self.dst.name}, {self.src0.disassemble()}"
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

    def get_vgpr16(self, index: int, half: HalfSelect) -> int:
        dword = self.vgprs[index]
        lo, hi = dword_to_f16_halves(dword)
        return hi if half == HalfSelect.HI else lo

    def set_vgpr16(self, index: int, half: HalfSelect, bits: int) -> None:
        dword = self.vgprs[index]
        lo, hi = dword_to_f16_halves(dword)
        if half == HalfSelect.HI:
            self.vgprs[index] = f16_halves_to_dword(lo, bits & 0xFFFF)
        else:
            self.vgprs[index] = f16_halves_to_dword(bits & 0xFFFF, hi)

    def get_sgpr16(self, index: int, half: HalfSelect) -> int:
        dword = self.sgprs[index]
        lo, hi = dword_to_f16_halves(dword)
        return hi if half == HalfSelect.HI else lo

    def set_sgpr16(self, index: int, half: HalfSelect, bits: int) -> None:
        dword = self.sgprs[index]
        lo, hi = dword_to_f16_halves(dword)
        if half == HalfSelect.HI:
            self.sgprs[index] = f16_halves_to_dword(lo, bits & 0xFFFF)
        else:
            self.sgprs[index] = f16_halves_to_dword(bits & 0xFFFF, hi)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_sqrt_clamp_f16."""

    def resolve_operand(self, op: VOP3Operand, state: ScalarMachineState) -> int:
        if isinstance(op.source, RegHalf):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr16(op.source.index, op.source.half)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr16(op.source.index, op.source.half)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal16):
            return op.source.bits
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VSqrtClampF16Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)

        # Computation phase with clamp saturation
        result_bits = ref_sqrt_clamp_f16(
            src0_bits=s0_raw,
            abs0=instr.src0.abs_mod,
            neg0=instr.src0.neg_mod,
        )

        # Write-back phase
        state.set_vgpr16(instr.dst.index, instr.dst.half, result_bits)
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

    def get_vgpr16(self, lane: int, index: int, half: HalfSelect) -> int:
        dword = self.vgprs[lane][index]
        lo, hi = dword_to_f16_halves(dword)
        return hi if half == HalfSelect.HI else lo

    def set_vgpr16(self, lane: int, index: int, half: HalfSelect, bits: int) -> None:
        dword = self.vgprs[lane][index]
        lo, hi = dword_to_f16_halves(dword)
        if half == HalfSelect.HI:
            self.vgprs[lane][index] = f16_halves_to_dword(lo, bits & 0xFFFF)
        else:
            self.vgprs[lane][index] = f16_halves_to_dword(bits & 0xFFFF, hi)

    def get_sgpr16(self, index: int, half: HalfSelect) -> int:
        dword = self.sgprs[index]
        lo, hi = dword_to_f16_halves(dword)
        return hi if half == HalfSelect.HI else lo

    def set_sgpr16(self, index: int, half: HalfSelect, bits: int) -> None:
        dword = self.sgprs[index]
        lo, hi = dword_to_f16_halves(dword)
        if half == HalfSelect.HI:
            self.sgprs[index] = f16_halves_to_dword(lo, bits & 0xFFFF)
        else:
            self.sgprs[index] = f16_halves_to_dword(bits & 0xFFFF, hi)


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VOP3Operand, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, RegHalf):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr16(lane, op.source.index, op.source.half)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr16(op.source.index, op.source.half)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal16):
            return op.source.bits
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VSqrtClampF16Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * state.num_lanes
        s0_vals = [
            self.resolve_lane_operand(instr.src0, state, lane)
            for lane in range(state.num_lanes)
        ]

        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = ref_sqrt_clamp_f16(
                    src0_bits=s0_vals[lane],
                    abs0=instr.src0.abs_mod,
                    neg0=instr.src0.neg_mod,
                )
                state.set_vgpr16(lane, instr.dst.index, instr.dst.half, res)
                results[lane] = res

        return results


# ==============================================================================
# Unit Tests
# ==============================================================================

class TestSqrtClampF16ModelAndOracle(unittest.TestCase):
    """Unit tests for binary16 bit model, predicates, and clamped reference oracle."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 4.0, 9.0]
        for val in values:
            with self.subTest(val=val):
                bits = f16_to_bits(val)
                recovered = bits_to_f16(bits)
                self.assertEqual(val, recovered)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

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

    def test_halfword_packing_vgpr(self) -> None:
        lo = 0x3C00  # 1.0
        hi = 0x4000  # 2.0
        dword = f16_halves_to_dword(lo, hi)
        self.assertEqual(dword, 0x40003C00)
        unpacked_lo, unpacked_hi = dword_to_f16_halves(dword)
        self.assertEqual(unpacked_lo, lo)
        self.assertEqual(unpacked_hi, hi)

    def test_exact_clamped_sqrt(self) -> None:
        # Values > 1.0 saturate to 1.0
        self.assertEqual(ref_sqrt_clamp_f16(FOUR_F16), ONE_F16)
        self.assertEqual(ref_sqrt_clamp_f16(NINE_F16), ONE_F16)
        self.assertEqual(ref_sqrt_clamp_f16(TWO_F16), ONE_F16)

        # Values <= 1.0 remain unmodified
        self.assertEqual(ref_sqrt_clamp_f16(ONE_F16), ONE_F16)
        self.assertEqual(ref_sqrt_clamp_f16(QUARTER_F16), HALF_F16)

    def test_signed_zero_rules(self) -> None:
        self.assertEqual(ref_sqrt_clamp_f16(POS_ZERO), POS_ZERO)
        # -0.0 clamps to +0.0
        self.assertEqual(ref_sqrt_clamp_f16(NEG_ZERO), POS_ZERO)

    def test_negative_inputs_clamped_to_zero(self) -> None:
        self.assertEqual(ref_sqrt_clamp_f16(0xBC00), POS_ZERO)  # -1.0
        self.assertEqual(ref_sqrt_clamp_f16(FOUR_F16 | SIGN_MASK), POS_ZERO)  # -4.0
        self.assertEqual(ref_sqrt_clamp_f16(NEG_INF), POS_ZERO)

    def test_nan_inputs_clamped_to_zero(self) -> None:
        self.assertEqual(ref_sqrt_clamp_f16(CANONICAL_QNAN), POS_ZERO)
        self.assertEqual(ref_sqrt_clamp_f16(0x7C05), POS_ZERO)  # SNaN

    def test_positive_inf_clamped_to_one(self) -> None:
        self.assertEqual(ref_sqrt_clamp_f16(POS_INF), ONE_F16)

    def test_input_modifiers_abs_neg(self) -> None:
        val_neg_four = FOUR_F16 | SIGN_MASK  # -4.0
        # abs on -4.0: sqrt(|-4.0|) = sqrt(4.0) = 2.0 -> clamped to 1.0
        res = ref_sqrt_clamp_f16(val_neg_four, abs0=True)
        self.assertEqual(res, ONE_F16)

        # neg on 0.25: -(0.25) = -0.25 -> clamped to +0.0
        res_neg = ref_sqrt_clamp_f16(QUARTER_F16, neg0=True)
        self.assertEqual(res_neg, POS_ZERO)


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_reghalf_validation_and_naming(self) -> None:
        v0_l = RegHalf(0, RegType.VGPR, HalfSelect.LO)
        self.assertEqual(v0_l.name, "v0.l")
        self.assertEqual(v0_l.index, 0)
        self.assertEqual(v0_l.half, HalfSelect.LO)

        v0_h = RegHalf(0, RegType.VGPR, HalfSelect.HI)
        self.assertEqual(v0_h.name, "v0.h")

        s8_l = RegHalf(8, RegType.SGPR, HalfSelect.LO)
        self.assertEqual(s8_l.name, "s8.l")
        s8_h = RegHalf(8, RegType.SGPR, HalfSelect.HI)
        self.assertEqual(s8_h.name, "s8.h")

        with self.assertRaises(ValueError):
            RegHalf(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegHalf(-1, RegType.VGPR)

    def test_operand_disassembly(self) -> None:
        v = RegHalf(2, RegType.VGPR, HalfSelect.LO)
        op_plain = VOP3Operand(v)
        self.assertEqual(op_plain.disassemble(), "v2.l")

        op_abs = VOP3Operand(v, abs_mod=True)
        self.assertEqual(op_abs.disassemble(), "|v2.l|")

        op_neg = VOP3Operand(v, neg_mod=True)
        self.assertEqual(op_neg.disassemble(), "-v2.l")

        op_neg_abs = VOP3Operand(v, abs_mod=True, neg_mod=True)
        self.assertEqual(op_neg_abs.disassemble(), "-|v2.l|")

        lit = Literal16.from_float(4.0)
        op_lit = VOP3Operand(lit)
        self.assertEqual(op_lit.disassemble(), "4")

    def test_v_sqrt_clamp_f16_ast_construction_and_validation(self) -> None:
        dst = RegHalf(0, RegType.VGPR, HalfSelect.LO)
        src0 = VOP3Operand(RegHalf(2, RegType.VGPR, HalfSelect.HI), abs_mod=True)
        instr = VSqrtClampF16Instruction(dst=dst, src0=src0, clamp=True)

        self.assertEqual(instr.disassemble(), "v_sqrt_f16 v0.l, |v2.h| clamp")

        # Invalid destination type (SGPR)
        with self.assertRaises(TypeError):
            VSqrtClampF16Instruction(dst=RegHalf(0, RegType.SGPR), src0=src0)

    def test_structural_equality(self) -> None:
        i1 = VSqrtClampF16Instruction(
            RegHalf(0, RegType.VGPR, HalfSelect.LO),
            VOP3Operand(RegHalf(2, RegType.VGPR, HalfSelect.HI)),
        )
        i2 = VSqrtClampF16Instruction(
            RegHalf(0, RegType.VGPR, HalfSelect.LO),
            VOP3Operand(RegHalf(2, RegType.VGPR, HalfSelect.HI)),
        )
        self.assertEqual(i1, i2)


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_scalar_vgpr_to_vgpr_lo(self) -> None:
        self.state.set_vgpr16(2, HalfSelect.LO, QUARTER_F16)
        instr = VSqrtClampF16Instruction(
            dst=RegHalf(0, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(2, RegType.VGPR, HalfSelect.LO)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, HALF_F16)
        self.assertEqual(self.state.get_vgpr16(0, HalfSelect.LO), HALF_F16)

    def test_scalar_vgpr_hi_preserves_lo(self) -> None:
        self.state.vgprs[0] = 0x5555AAAA
        self.state.set_vgpr16(4, HalfSelect.LO, QUARTER_F16)
        instr = VSqrtClampF16Instruction(
            dst=RegHalf(0, RegType.VGPR, HalfSelect.HI),
            src0=VOP3Operand(RegHalf(4, RegType.VGPR, HalfSelect.LO)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, HALF_F16)
        self.assertEqual(self.state.get_vgpr16(0, HalfSelect.HI), HALF_F16)
        self.assertEqual(self.state.get_vgpr16(0, HalfSelect.LO), 0xAAAA)

    def test_scalar_sgpr_source(self) -> None:
        self.state.set_sgpr16(8, HalfSelect.LO, ONE_F16)
        instr = VSqrtClampF16Instruction(
            dst=RegHalf(1, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(8, RegType.SGPR, HalfSelect.LO)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, ONE_F16)
        self.assertEqual(self.state.get_vgpr16(1, HalfSelect.LO), ONE_F16)

    def test_scalar_literal_source(self) -> None:
        instr = VSqrtClampF16Instruction(
            dst=RegHalf(3, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(Literal16.from_float(0.25)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, HALF_F16)
        self.assertEqual(self.state.get_vgpr16(3, HalfSelect.LO), HALF_F16)

    def test_scalar_clamping_saturation(self) -> None:
        # 4.0 saturates to 1.0
        self.state.set_vgpr16(5, HalfSelect.LO, FOUR_F16)
        instr = VSqrtClampF16Instruction(
            dst=RegHalf(6, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(5, RegType.VGPR, HalfSelect.LO)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, ONE_F16)

        # -4.0 saturates to +0.0
        self.state.set_vgpr16(7, HalfSelect.LO, FOUR_F16 | SIGN_MASK)
        instr2 = VSqrtClampF16Instruction(
            dst=RegHalf(8, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(7, RegType.VGPR, HalfSelect.LO)),
        )
        res_bits2 = self.interp.step(instr2, self.state)
        self.assertEqual(res_bits2, POS_ZERO)

    def test_scalar_determinism(self) -> None:
        runs = []
        for _ in range(50):
            st = ScalarMachineState()
            st.set_vgpr16(0, HalfSelect.LO, QUARTER_F16)
            instr = VSqrtClampF16Instruction(
                dst=RegHalf(1, RegType.VGPR, HalfSelect.LO),
                src0=VOP3Operand(RegHalf(0, RegType.VGPR, HalfSelect.LO)),
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
            val = 0.25 if lane % 2 == 0 else 4.0
            state.set_vgpr16(lane, 0, HalfSelect.LO, f16_to_bits(val))

        instr = VSqrtClampF16Instruction(
            dst=RegHalf(2, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(0, RegType.VGPR, HalfSelect.LO)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            expected = HALF_F16 if lane % 2 == 0 else ONE_F16
            self.assertEqual(state.get_vgpr16(lane, 2, HalfSelect.LO), expected)

    def test_wave32_divergent_masking(self) -> None:
        sentinel = 0xDEAD
        state = SIMDState(num_lanes=32, exec_mask=0x5555_5555)
        for lane in range(32):
            state.set_vgpr16(lane, 0, HalfSelect.LO, QUARTER_F16)
            state.set_vgpr16(lane, 2, HalfSelect.LO, sentinel)

        instr = VSqrtClampF16Instruction(
            dst=RegHalf(2, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(0, RegType.VGPR, HalfSelect.LO)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertIsNotNone(results[lane])
                self.assertEqual(state.get_vgpr16(lane, 2, HalfSelect.LO), HALF_F16)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr16(lane, 2, HalfSelect.LO), sentinel)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr16(lane, 0, HalfSelect.LO, FOUR_F16)

        instr = VSqrtClampF16Instruction(
            dst=RegHalf(1, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(0, RegType.VGPR, HalfSelect.LO)),
        )
        self.interp.step(instr, state)
        for lane in range(64):
            # Clamped to 1.0
            self.assertEqual(state.get_vgpr16(lane, 1, HalfSelect.LO), ONE_F16)

    def test_wave64_half_disabled(self) -> None:
        sentinel = 0xBEEF
        state = SIMDState(num_lanes=64, exec_mask=0x0000_0000_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr16(lane, 0, HalfSelect.LO, QUARTER_F16)
            state.set_vgpr16(lane, 2, HalfSelect.LO, sentinel)

        instr = VSqrtClampF16Instruction(
            dst=RegHalf(2, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(0, RegType.VGPR, HalfSelect.LO)),
        )
        results = self.interp.step(instr, state)
        for lane in range(64):
            if lane < 32:
                self.assertIsNotNone(results[lane])
                self.assertEqual(state.get_vgpr16(lane, 2, HalfSelect.LO), HALF_F16)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr16(lane, 2, HalfSelect.LO), sentinel)

    def test_wave_exec_all_zero(self) -> None:
        sentinel = 0xAA55
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)
        for lane in range(32):
            state.set_vgpr16(lane, 0, HalfSelect.LO, sentinel)

        instr = VSqrtClampF16Instruction(
            dst=RegHalf(0, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(Literal16.from_float(0.25)),
        )
        results = self.interp.step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(state.get_vgpr16(lane, 0, HalfSelect.LO), sentinel)

    def test_simd_sgpr_uniform_broadcast(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        state.set_sgpr16(10, HalfSelect.LO, QUARTER_F16)

        instr = VSqrtClampF16Instruction(
            dst=RegHalf(2, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(10, RegType.SGPR, HalfSelect.LO)),
        )
        self.interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr16(lane, 2, HalfSelect.LO), HALF_F16)


class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src0_lo_to_lo(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr16(0, HalfSelect.LO, QUARTER_F16)
        instr = VSqrtClampF16Instruction(
            dst=RegHalf(0, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(0, RegType.VGPR, HalfSelect.LO)),
        )
        self.scalar_interp.step(instr, state)
        self.assertEqual(state.get_vgpr16(0, HalfSelect.LO), HALF_F16)

    def test_dst_alias_src0_hi_to_hi(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr16(0, HalfSelect.HI, FOUR_F16)
        instr = VSqrtClampF16Instruction(
            dst=RegHalf(0, RegType.VGPR, HalfSelect.HI),
            src0=VOP3Operand(RegHalf(0, RegType.VGPR, HalfSelect.HI)),
        )
        self.scalar_interp.step(instr, state)
        # Clamped to 1.0
        self.assertEqual(state.get_vgpr16(0, HalfSelect.HI), ONE_F16)

    def test_dst_hi_src_lo_cross_half_alias(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr16(0, HalfSelect.LO, QUARTER_F16)
        state.set_vgpr16(0, HalfSelect.HI, 0x1234)
        instr = VSqrtClampF16Instruction(
            dst=RegHalf(0, RegType.VGPR, HalfSelect.HI),
            src0=VOP3Operand(RegHalf(0, RegType.VGPR, HalfSelect.LO)),
        )
        self.scalar_interp.step(instr, state)
        self.assertEqual(state.get_vgpr16(0, HalfSelect.HI), HALF_F16)
        self.assertEqual(state.get_vgpr16(0, HalfSelect.LO), QUARTER_F16)

    def test_dst_lo_src_hi_cross_half_alias(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr16(0, HalfSelect.HI, FOUR_F16)
        state.set_vgpr16(0, HalfSelect.LO, 0x5678)
        instr = VSqrtClampF16Instruction(
            dst=RegHalf(0, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(0, RegType.VGPR, HalfSelect.HI)),
        )
        self.scalar_interp.step(instr, state)
        self.assertEqual(state.get_vgpr16(0, HalfSelect.LO), ONE_F16)
        self.assertEqual(state.get_vgpr16(0, HalfSelect.HI), FOUR_F16)

    def test_inplace_iterative_evolution(self) -> None:
        # Starting with 0.0625 -> sqrt is 0.25 -> sqrt is 0.5
        state = ScalarMachineState()
        state.set_vgpr16(0, HalfSelect.LO, f16_to_bits(0.0625))
        instr = VSqrtClampF16Instruction(
            dst=RegHalf(0, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(0, RegType.VGPR, HalfSelect.LO)),
        )
        self.scalar_interp.step(instr, state)
        self.assertEqual(state.get_vgpr16(0, HalfSelect.LO), QUARTER_F16)
        self.scalar_interp.step(instr, state)
        self.assertEqual(state.get_vgpr16(0, HalfSelect.LO), HALF_F16)

    def test_simd_multi_lane_aliasing(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr16(lane, 0, HalfSelect.LO, QUARTER_F16)
        instr = VSqrtClampF16Instruction(
            dst=RegHalf(0, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(0, RegType.VGPR, HalfSelect.LO)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr16(lane, 0, HalfSelect.LO), HALF_F16)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
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
    clamped: bool = False


def evaluate_v_sqrt_clamp_f16_flags(
    src0_bits: int,
    res_bits: int,
    abs0: bool = False,
    neg0: bool = False,
) -> FPStatusFlags:
    """Evaluate IEEE 754 architectural flags and condition codes for v_sqrt_clamp_f16."""
    s0 = apply_f16_modifiers(src0_bits, abs0, neg0)
    s0_nan = is_nan(s0)
    s0_snan = is_snan(s0)
    s0_inf = is_inf(s0)
    s0_neg = bool(s0 & SIGN_MASK) and (s0 != NEG_ZERO)

    invalid = False
    if s0_snan or (s0_neg and not s0_nan):
        invalid = True

    nan_flag = is_nan(res_bits)
    inf_flag = is_inf(res_bits)
    zero_flag = is_zero(res_bits)
    neg_flag = bool(res_bits & SIGN_MASK)
    denorm_flag = is_subnormal(res_bits)

    clamped = False
    if s0_nan or s0_neg or (s0_inf and not bool(s0 & SIGN_MASK)):
        clamped = True
    elif not zero_flag:
        f0 = bits_to_f16(s0)
        res_unclamped = math.sqrt(f0)
        if res_unclamped > 1.0 or res_unclamped < 0.0:
            clamped = True

    inexact = False
    if clamped:
        inexact = True
    elif not (s0_nan or s0_inf or nan_flag or inf_flag or zero_flag):
        f0 = bits_to_f16(s0)
        res_f = bits_to_f16(res_bits)
        if Fraction(res_f) * Fraction(res_f) != Fraction(f0):
            inexact = True

    return FPStatusFlags(
        invalid=invalid,
        overflow=False,
        underflow=False,
        inexact=inexact,
        zero=zero_flag,
        negative=neg_flag,
        denormal=denorm_flag,
        infinite=inf_flag,
        nan=nan_flag,
        clamped=clamped,
    )


def compute_f16_root_borrow_carry(s0_bits: int) -> Tuple[int, int]:
    """Demonstrate binary radix-2 digit-recurrence square root trial subtraction."""
    _, exp0, frac0 = split_f16_bits(s0_bits)
    m = (1 << 10) | frac0 if exp0 != 0 else frac0
    rem = m
    trial_sub = (1 << 10)
    borrow = 1 if rem < trial_sub else 0
    root = 1 if borrow == 0 else 0
    return root, borrow


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for status flags, condition codes, and carry/borrow generation."""

    def test_exact_finite_flags(self) -> None:
        s0 = QUARTER_F16
        res = ref_sqrt_clamp_f16(s0)
        flags = evaluate_v_sqrt_clamp_f16_flags(s0, res)
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

    def test_inexact_flag(self) -> None:
        # 0.5 -> sqrt(0.5) is irrational, inexact
        s0 = HALF_F16
        res = ref_sqrt_clamp_f16(s0)
        flags = evaluate_v_sqrt_clamp_f16_flags(s0, res)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.invalid)

    def test_invalid_operation_flag(self) -> None:
        # Negative operand (-4.0) triggers invalid operation, but clamps to +0.0
        s0_neg = FOUR_F16 | SIGN_MASK
        res_neg = ref_sqrt_clamp_f16(s0_neg)
        self.assertEqual(res_neg, POS_ZERO)
        flags_neg = evaluate_v_sqrt_clamp_f16_flags(s0_neg, res_neg)
        self.assertTrue(flags_neg.invalid)
        self.assertTrue(flags_neg.clamped)
        self.assertTrue(flags_neg.zero)

        # Signaling NaN triggers invalid operation
        snan = 0x7C01
        res_snan = ref_sqrt_clamp_f16(snan)
        flags_snan = evaluate_v_sqrt_clamp_f16_flags(snan, res_snan)
        self.assertTrue(flags_snan.invalid)
        self.assertTrue(flags_snan.clamped)

    def test_clamped_flag_saturation(self) -> None:
        # 4.0 saturates to 1.0 -> clamped flag set
        s0 = FOUR_F16
        res = ref_sqrt_clamp_f16(s0)
        self.assertEqual(res, ONE_F16)
        flags = evaluate_v_sqrt_clamp_f16_flags(s0, res)
        self.assertTrue(flags.clamped)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.invalid)

    def test_zero_and_negative_flags(self) -> None:
        # +0.0
        res_pz = ref_sqrt_clamp_f16(POS_ZERO)
        flags_pz = evaluate_v_sqrt_clamp_f16_flags(POS_ZERO, res_pz)
        self.assertTrue(flags_pz.zero)
        self.assertFalse(flags_pz.negative)

        # -0.0 clamps to +0.0, so negative flag is False
        res_nz = ref_sqrt_clamp_f16(NEG_ZERO)
        flags_nz = evaluate_v_sqrt_clamp_f16_flags(NEG_ZERO, res_nz)
        self.assertTrue(flags_nz.zero)
        self.assertFalse(flags_nz.negative)

    def test_root_borrow_carry_generation(self) -> None:
        root1, borrow1 = compute_f16_root_borrow_carry(ONE_F16)
        self.assertEqual(borrow1, 0)
        self.assertEqual(root1, 1)

        root0, borrow0 = compute_f16_root_borrow_carry(0x0001)
        self.assertEqual(borrow0, 1)
        self.assertEqual(root0, 0)


if __name__ == "__main__":
    unittest.main()
