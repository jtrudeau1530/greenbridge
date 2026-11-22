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


def oracle_v_sqrt_f16(
    src0_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_sqrt_f16 half-precision square root.

    Computes D = sqrt(src0) conforming to IEEE 754-2008 binary16
    with Round-to-Nearest-Even (RNE), applying input modifiers and clamp.
    """
    s0 = apply_f16_modifiers(src0_bits, abs0, neg0)

    # 1. NaN handling
    if is_nan(s0):
        if clamp:
            return POS_ZERO
        return s0 | QNAN_BIT

    # 2. Negative inputs (excluding negative zero)
    # In IEEE 754 and RDNA2, sqrt of negative non-zero produces Canonical Quiet NaN
    if (s0 & SIGN_MASK) != 0 and s0 != NEG_ZERO:
        return POS_ZERO if clamp else CANONICAL_QNAN

    # 3. Infinity handling (+inf -> +inf, -inf was caught above as < 0)
    if is_inf(s0):
        if clamp:
            return ONE_F16
        return POS_INF

    # 4. Zero handling (+0 -> +0, -0 -> -0)
    if is_zero(s0):
        if clamp:
            return POS_ZERO
        return s0

    # 5. Finite positive square root (normals and subnormals)
    f0 = bits_to_f16(s0)
    res_f = math.sqrt(f0)
    res = f16_to_bits(res_f)

    # 6. Clamp modifier ([0.0, 1.0])
    if clamp:
        if res_f < 0.0 or res == NEG_ZERO:
            res = POS_ZERO
        elif res_f > 1.0 or res == POS_INF:
            res = ONE_F16

    return res


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
class VSqrtF16Instruction:
    """Symbolic AST node representing a RDNA2 v_sqrt_f16 instruction."""
    dst: RegHalf
    src0: VOP3Operand
    clamp: bool = False

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
    """Deterministic scalar instruction interpreter for v_sqrt_f16."""

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

    def step(self, instr: VSqrtF16Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)

        # Computation phase
        result_bits = oracle_v_sqrt_f16(
            src0_bits=s0_raw,
            abs0=instr.src0.abs_mod,
            neg0=instr.src0.neg_mod,
            clamp=instr.clamp,
        )

        # Write-back phase
        state.set_vgpr16(instr.dst.index, instr.dst.half, result_bits)
        return result_bits


# ==============================================================================
# Unit Tests
# ==============================================================================

class TestF16ModelAndOracle(unittest.TestCase):
    """Unit tests for binary16 bit model, predicates, and reference oracle."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 4.0, 9.0, 16.0]
        for val in values:
            with self.subTest(val=val):
                bits = f16_to_bits(val)
                recovered = bits_to_f16(bits)
                self.assertEqual(val, recovered)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_halfword_packing_vgpr(self) -> None:
        lo = 0x3C00  # 1.0
        hi = 0x4000  # 2.0
        dword = f16_halves_to_dword(lo, hi)
        self.assertEqual(dword, 0x40003C00)
        unpacked_lo, unpacked_hi = dword_to_f16_halves(dword)
        self.assertEqual(unpacked_lo, lo)
        self.assertEqual(unpacked_hi, hi)
        self.assertEqual(bits_to_f16(unpacked_lo), 1.0)
        self.assertEqual(bits_to_f16(unpacked_hi), 2.0)

    def test_classification_predicates(self) -> None:
        self.assertTrue(f16_to_bits(0.0) == POS_ZERO)
        self.assertTrue(f16_to_bits(-0.0) == NEG_ZERO)
        self.assertTrue(is_zero(POS_ZERO))
        self.assertTrue(is_zero(NEG_ZERO))
        self.assertFalse(is_zero(f16_to_bits(1.0)))

        self.assertTrue(is_inf(POS_INF))
        self.assertTrue(is_inf(NEG_INF))
        self.assertFalse(is_inf(POS_ZERO))

        snan_bits = 0x7C01  # exponent 0x1F, fraction != 0, quiet bit 9 is 0
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

    def test_exact_finite_sqrt(self) -> None:
        self.assertEqual(oracle_v_sqrt_f16(FOUR_F16), TWO_F16)
        self.assertEqual(oracle_v_sqrt_f16(NINE_F16), THREE_F16)
        self.assertEqual(oracle_v_sqrt_f16(ONE_F16), ONE_F16)
        self.assertEqual(oracle_v_sqrt_f16(QUARTER_F16), HALF_F16)

    def test_signed_zero_rules(self) -> None:
        # IEEE 754: sqrt(+0.0) = +0.0, sqrt(-0.0) = -0.0
        self.assertEqual(oracle_v_sqrt_f16(POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_v_sqrt_f16(NEG_ZERO), NEG_ZERO)

    def test_input_modifiers_abs_neg(self) -> None:
        val_neg_four = FOUR_F16 | SIGN_MASK  # -4.0
        # abs on -4.0: sqrt(|-4.0|) = sqrt(4.0) = 2.0
        res = oracle_v_sqrt_f16(val_neg_four, abs0=True)
        self.assertEqual(res, TWO_F16)

        # neg on 4.0: sqrt(-(4.0)) = sqrt(-4.0) = NaN
        res_neg = oracle_v_sqrt_f16(FOUR_F16, neg0=True)
        self.assertTrue(is_qnan(res_neg))
        self.assertEqual(res_neg, CANONICAL_QNAN)

        # abs and neg together: -|4.0| = -4.0 -> NaN
        res_neg_abs = oracle_v_sqrt_f16(FOUR_F16, abs0=True, neg0=True)
        self.assertEqual(res_neg_abs, CANONICAL_QNAN)

    def test_clamp_modifier(self) -> None:
        # sqrt(4.0) = 2.0 -> clamped to 1.0
        res_clamped_hi = oracle_v_sqrt_f16(FOUR_F16, clamp=True)
        self.assertEqual(res_clamped_hi, ONE_F16)

        # sqrt(0.25) = 0.5 -> within [0, 1], remains 0.5
        res_mid = oracle_v_sqrt_f16(QUARTER_F16, clamp=True)
        self.assertEqual(res_mid, HALF_F16)

        # sqrt(-4.0) with clamp produces +0.0
        res_nan_clamp = oracle_v_sqrt_f16(FOUR_F16 | SIGN_MASK, clamp=True)
        self.assertEqual(res_nan_clamp, POS_ZERO)

        # sqrt(-0.0) with clamp produces +0.0
        res_zero_clamp = oracle_v_sqrt_f16(NEG_ZERO, clamp=True)
        self.assertEqual(res_zero_clamp, POS_ZERO)

    def test_infinity_sqrt(self) -> None:
        self.assertEqual(oracle_v_sqrt_f16(POS_INF), POS_INF)
        self.assertEqual(oracle_v_sqrt_f16(NEG_INF), CANONICAL_QNAN)
        # Clamped +inf yields 1.0
        self.assertEqual(oracle_v_sqrt_f16(POS_INF, clamp=True), ONE_F16)

    def test_negative_operand_produces_qnan(self) -> None:
        self.assertEqual(oracle_v_sqrt_f16(0xBC00), CANONICAL_QNAN)  # -1.0
        self.assertEqual(oracle_v_sqrt_f16(0x8001), CANONICAL_QNAN)  # negative subnormal

    def test_nan_quietening_and_propagation(self) -> None:
        snan = 0x7C05  # Signaling NaN
        res = oracle_v_sqrt_f16(snan)
        self.assertTrue(is_qnan(res))
        self.assertEqual(res, snan | QNAN_BIT)


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

    def test_v_sqrt_f16_ast_construction_and_validation(self) -> None:
        dst = RegHalf(0, RegType.VGPR, HalfSelect.LO)
        src0 = VOP3Operand(RegHalf(2, RegType.VGPR, HalfSelect.HI), abs_mod=True)
        instr = VSqrtF16Instruction(dst=dst, src0=src0, clamp=True)

        self.assertEqual(instr.disassemble(), "v_sqrt_f16 v0.l, |v2.h| clamp")

        # Invalid destination type (SGPR)
        with self.assertRaises(TypeError):
            VSqrtF16Instruction(dst=RegHalf(0, RegType.SGPR), src0=src0)

    def test_structural_equality(self) -> None:
        i1 = VSqrtF16Instruction(
            RegHalf(0, RegType.VGPR, HalfSelect.LO),
            VOP3Operand(RegHalf(2, RegType.VGPR, HalfSelect.HI)),
        )
        i2 = VSqrtF16Instruction(
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
        self.state.set_vgpr16(2, HalfSelect.LO, FOUR_F16)
        instr = VSqrtF16Instruction(
            dst=RegHalf(0, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(2, RegType.VGPR, HalfSelect.LO)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, TWO_F16)
        self.assertEqual(self.state.get_vgpr16(0, HalfSelect.LO), TWO_F16)

    def test_scalar_vgpr_hi_preserves_lo(self) -> None:
        # Prepopulate v0 with 0xAAAA in lo, 0x5555 in hi
        self.state.vgprs[0] = 0x5555AAAA
        self.state.set_vgpr16(4, HalfSelect.LO, NINE_F16)
        instr = VSqrtF16Instruction(
            dst=RegHalf(0, RegType.VGPR, HalfSelect.HI),
            src0=VOP3Operand(RegHalf(4, RegType.VGPR, HalfSelect.LO)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, THREE_F16)
        # Hi half updated to THREE_F16, Lo half preserved (0xAAAA)
        self.assertEqual(self.state.get_vgpr16(0, HalfSelect.HI), THREE_F16)
        self.assertEqual(self.state.get_vgpr16(0, HalfSelect.LO), 0xAAAA)

    def test_scalar_sgpr_source(self) -> None:
        self.state.set_sgpr16(8, HalfSelect.LO, ONE_F16)
        instr = VSqrtF16Instruction(
            dst=RegHalf(1, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(8, RegType.SGPR, HalfSelect.LO)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, ONE_F16)
        self.assertEqual(self.state.get_vgpr16(1, HalfSelect.LO), ONE_F16)

    def test_scalar_literal_source(self) -> None:
        instr = VSqrtF16Instruction(
            dst=RegHalf(3, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(Literal16.from_float(0.25)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, HALF_F16)
        self.assertEqual(self.state.get_vgpr16(3, HalfSelect.LO), HALF_F16)

    def test_scalar_with_modifiers_and_clamp(self) -> None:
        self.state.set_vgpr16(5, HalfSelect.LO, FOUR_F16 | SIGN_MASK)  # -4.0
        # abs clears sign bit, then sqrt(4.0) = 2.0, clamp limits to 1.0
        instr = VSqrtF16Instruction(
            dst=RegHalf(6, RegType.VGPR, HalfSelect.LO),
            src0=VOP3Operand(RegHalf(5, RegType.VGPR, HalfSelect.LO), abs_mod=True),
            clamp=True,
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, ONE_F16)
        self.assertEqual(self.state.get_vgpr16(6, HalfSelect.LO), ONE_F16)

    def test_scalar_determinism(self) -> None:
        runs = []
        for _ in range(50):
            st = ScalarMachineState()
            st.set_vgpr16(0, HalfSelect.LO, 0x4880)  # 9.0
            instr = VSqrtF16Instruction(
                dst=RegHalf(1, RegType.VGPR, HalfSelect.LO),
                src0=VOP3Operand(RegHalf(0, RegType.VGPR, HalfSelect.LO)),
            )
            runs.append(self.interp.step(instr, st))
        self.assertTrue(all(r == runs[0] for r in runs))


if __name__ == "__main__":
    unittest.main()
