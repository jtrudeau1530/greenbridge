from __future__ import annotations

from dataclasses import dataclass, field, field
from enum import Enum, auto
from fractions import Fraction
import math
import struct
from typing import Union
import unittest

# IEEE 754 binary64 (double precision) architectural constants
SIGN_MASK_F64: int = 0x8000_0000_0000_0000
EXP_MASK_F64: int = 0x7FF0_0000_0000_0000
FRAC_MASK_F64: int = 0x000F_FFFF_FFFF_FFFF
QNAN_BIT_F64: int = 0x0008_0000_0000_0000
CANONICAL_QNAN_F64: int = 0x7FF8_0000_0000_0000

POS_ZERO_F64: int = 0x0000_0000_0000_0000
NEG_ZERO_F64: int = 0x8000_0000_0000_0000
POS_INF_F64: int = 0x7FF0_0000_0000_0000
NEG_INF_F64: int = 0xFFF0_0000_0000_0000

ONE_F64: int = 0x3FF0_0000_0000_0000
# Largest binary64 strictly below 1.0: 1.0 - 2^-53
LARGEST_BELOW_ONE_F64: int = 0x3FEF_FFFF_FFFF_FFFF


def f64_to_bits(val: float) -> int:
    """Convert a Python float (IEEE 754 double) to 64-bit integer bits."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def bits_to_f64(bits: int) -> float:
    """Convert 64-bit integer bits to a Python float (IEEE 754 double)."""
    return struct.unpack("<d", struct.pack("<Q", bits & 0xFFFF_FFFF_FFFF_FFFF))[0]


def split_f64_bits(bits: int) -> tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK_F64) >> 63
    exponent = (bits & EXP_MASK_F64) >> 52
    fraction = bits & FRAC_MASK_F64
    return sign, exponent, fraction


def words_to_f64_bits(lo: int, hi: int) -> int:
    """Combine two 32-bit VGPR words (lo, hi) into a 64-bit integer representation."""
    return ((hi & 0xFFFF_FFFF) << 32) | (lo & 0xFFFF_FFFF)


def f64_bits_to_words(bits: int) -> tuple[int, int]:
    """Split a 64-bit integer into two 32-bit VGPR words (lo, hi)."""
    lo = bits & 0xFFFF_FFFF
    hi = (bits >> 32) & 0xFFFF_FFFF
    return lo, hi


def is_nan_f64(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 NaN."""
    return ((bits & EXP_MASK_F64) == EXP_MASK_F64) and ((bits & FRAC_MASK_F64) != 0)


def is_qnan_f64(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Quiet NaN."""
    return is_nan_f64(bits) and bool(bits & QNAN_BIT_F64)


def is_snan_f64(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Signaling NaN."""
    return is_nan_f64(bits) and not bool(bits & QNAN_BIT_F64)


def is_inf_f64(bits: int) -> bool:
    """Return True if bits represent IEEE 754 positive or negative infinity."""
    return ((bits & EXP_MASK_F64) == EXP_MASK_F64) and ((bits & FRAC_MASK_F64) == 0)


def is_subnormal_f64(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormal) double-precision float."""
    return ((bits & EXP_MASK_F64) == 0) and ((bits & FRAC_MASK_F64) != 0)


def is_zero_f64(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK_F64) == 0


def apply_f64_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & 0xFFFF_FFFF_FFFF_FFFF
    if abs_mod:
        res &= ~SIGN_MASK_F64
    if neg_mod:
        res ^= SIGN_MASK_F64
    return res


def f64_to_fraction(bits: int) -> Fraction:
    """Decode finite IEEE 754 binary64 bit pattern into an exact Fraction."""
    bits &= 0xFFFF_FFFF_FFFF_FFFF
    exp = (bits & EXP_MASK_F64) >> 52
    if exp == 0x7FF:
        raise ValueError(f"non-finite binary64 {bits:#018x}")
    sign = -1 if (bits & SIGN_MASK_F64) else 1
    mant = bits & FRAC_MASK_F64
    if exp == 0:
        return Fraction(0) if mant == 0 else Fraction(sign * mant, 1 << 1074)
    return Fraction(sign * ((1 << 52) | mant)) * (Fraction(2) ** (exp - 1075))


def fraction_to_f64_bits(q: Fraction) -> int:
    """Round an exact Fraction to IEEE 754 binary64 (round-to-nearest-even)."""
    if q == 0:
        return POS_ZERO_F64
    sgn = SIGN_MASK_F64 if q < 0 else 0
    a = abs(q)
    e = max(a.numerator.bit_length() - a.denominator.bit_length() - 1, -1022)
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    while Fraction(2) ** e > a and e > -1022:
        e -= 1
    if e == -1022 and a < Fraction(2) ** -1022:
        m = a / (Fraction(2) ** -1074)
        n = m.numerator // m.denominator
        rem = m - n
        if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
            n += 1
        if n >= (1 << 52):
            return sgn | (1 << 52)
        return sgn | n
    else:
        m = a / (Fraction(2) ** (e - 52))
        n = m.numerator // m.denominator
        rem = m - n
        if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
            n += 1
        if n >= (1 << 53):
            n >>= 1
            e += 1
        if e > 1023:
            return sgn | POS_INF_F64
        biased_exp = e + 1023
        return sgn | (biased_exp << 52) | (n - (1 << 52))


def exact_fract_f64_fraction(bits: int) -> Fraction | None:
    """Exact x - floor(x) as Fraction in [0, 1), or None for non-finite values."""
    if is_nan_f64(bits) or is_inf_f64(bits):
        return None
    q = f64_to_fraction(bits)
    return q - math.floor(q)


def oracle_v_fract_f64(
    src0_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_fract_f64.

    Computes D.f64 = S0.f64 - floor(S0.f64) under IEEE 754-2008 double precision:
    - Input modifiers: abs applied before neg (-|s0|).
    - Signed zero: fract(-0.0) -> +0.0; fract(+0.0) -> +0.0.
    - Exact integers (1.0, -1.0, 2.0, etc.) -> +0.0.
    - Infinities: fract(+-inf) -> Canonical Quiet NaN.
    - NaNs: signaling NaNs are quieted; quiet NaNs preserve payload.
    - Rounding boundary: a result that rounds up to 1.0 (tiny negative numbers)
      is clamped to 0x3FEFFFFFFFFFFFFF (largest binary64 below 1.0), matching
      hardware behavior documented in docs/design.md.
    - Clamp modifier: clamps result to [0.0, 1.0]. Under clamp, NaN produces +0.0.
    """
    s0 = apply_f64_modifiers(src0_bits, abs_mod=abs0, neg_mod=neg0)

    # 1. NaN handling
    if is_nan_f64(s0):
        if clamp:
            return POS_ZERO_F64
        return s0 | QNAN_BIT_F64

    # 2. Infinity handling
    if is_inf_f64(s0):
        if clamp:
            return POS_ZERO_F64
        return CANONICAL_QNAN_F64

    # 3. Signed zero handling
    if is_zero_f64(s0):
        return POS_ZERO_F64

    # 4. Exact fraction calculation
    frac = exact_fract_f64_fraction(s0)
    if frac is None or frac == 0:
        return POS_ZERO_F64

    res = fraction_to_f64_bits(frac)

    # 5. Hardware clamp: any result that rounded up to 1.0 becomes largest double < 1.0
    if res == ONE_F64:
        res = LARGEST_BELOW_ONE_F64

    # 6. Clamp modifier ([0.0, 1.0])
    if clamp:
        # Since frac is strictly in [0, 1), finite result is already in [0, 1]
        pass

    return res


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


class Opcode(Enum):
    V_FRACT_F64 = auto()


@dataclass(frozen=True)
class RegPair:
    """Represents an architectural 64-bit register pair (e.g., v[0:1] or s[4:5])."""
    index: int
    reg_type: RegType

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 254):
            raise ValueError(f"Register pair base index {self.index} out of range [0, 254]")

    @property
    def lo_reg(self) -> int:
        return self.index

    @property
    def hi_reg(self) -> int:
        return self.index + 1

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}[{self.index}:{self.index + 1}]"


@dataclass(frozen=True)
class Literal64:
    """Represents a 64-bit literal or inline float constant."""
    bits: int

    @classmethod
    def from_float(cls, val: float) -> Literal64:
        return cls(f64_to_bits(val))

    @property
    def float_value(self) -> float:
        return bits_to_f64(self.bits)

    @property
    def name(self) -> str:
        if self.bits == POS_ZERO_F64:
            return "0.0"
        if self.bits == NEG_ZERO_F64:
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
    source: Union[RegPair, Literal64]
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
class VFractF64Instruction:
    """Symbolic AST node representing a RDNA2 v_fract_f64 instruction."""
    dst: RegPair
    src0: VOP3Operand
    clamp: bool = False
    opcode: Opcode = Opcode.V_FRACT_F64

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        s = f"v_fract_f64 {self.dst.name}, {self.src0.disassemble()}"
        if self.clamp:
            s += " clamp"
        return s


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single wavefront lane."""
    vgprs: list[int] = field(default_factory=lambda: [0] * 256)
    sgprs: list[int] = field(default_factory=lambda: [0] * 256)

    def get_vgpr64(self, index: int) -> int:
        return words_to_f64_bits(self.vgprs[index], self.vgprs[index + 1])

    def set_vgpr64(self, index: int, bits: int) -> None:
        lo, hi = f64_bits_to_words(bits)
        self.vgprs[index] = lo
        self.vgprs[index + 1] = hi

    def get_sgpr64(self, index: int) -> int:
        return words_to_f64_bits(self.sgprs[index], self.sgprs[index + 1])

    def set_sgpr64(self, index: int, bits: int) -> None:
        lo, hi = f64_bits_to_words(bits)
        self.sgprs[index] = lo
        self.sgprs[index + 1] = hi


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_fract_f64."""

    def resolve_operand(self, op: VOP3Operand, state: ScalarMachineState) -> int:
        if isinstance(op.source, RegPair):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr64(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr64(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal64):
            return op.source.bits
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VFractF64Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)

        # Computation phase
        result_bits = oracle_v_fract_f64(
            src0_bits=s0_raw,
            abs0=instr.src0.abs_mod,
            neg0=instr.src0.neg_mod,
            clamp=instr.clamp,
        )

        # Write-back phase
        state.set_vgpr64(instr.dst.index, result_bits)
        return result_bits


# ==============================================================================
# Unit Tests: Reference Oracle and Architectural Model
# ==============================================================================

class TestValuFractF64ModelAndOracle(unittest.TestCase):
    """Unit tests for binary64 bit model, predicates, and reference oracle."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 1e-100, 1e100, math.pi]
        for val in values:
            with self.subTest(val=val):
                bits = f64_to_bits(val)
                recovered = bits_to_f64(bits)
                if math.isnan(val):
                    self.assertTrue(math.isnan(recovered))
                else:
                    self.assertEqual(val, recovered)
                    self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_word_packing_vgpr_pair(self) -> None:
        val = 1234567.89012345
        bits = f64_to_bits(val)
        lo, hi = f64_bits_to_words(bits)
        self.assertEqual(lo, bits & 0xFFFF_FFFF)
        self.assertEqual(hi, (bits >> 32) & 0xFFFF_FFFF)
        combined = words_to_f64_bits(lo, hi)
        self.assertEqual(combined, bits)
        self.assertEqual(bits_to_f64(combined), val)

    def test_classification_predicates(self) -> None:
        self.assertTrue(f64_to_bits(0.0) == POS_ZERO_F64)
        self.assertTrue(f64_to_bits(-0.0) == NEG_ZERO_F64)
        self.assertTrue(is_zero_f64(POS_ZERO_F64))
        self.assertTrue(is_zero_f64(NEG_ZERO_F64))
        self.assertFalse(is_zero_f64(f64_to_bits(1.0)))

        self.assertTrue(is_inf_f64(POS_INF_F64))
        self.assertTrue(is_inf_f64(NEG_INF_F64))
        self.assertFalse(is_inf_f64(POS_ZERO_F64))

        snan_bits = 0x7FF0_0000_0000_0001
        qnan_bits = CANONICAL_QNAN_F64
        self.assertTrue(is_nan_f64(snan_bits))
        self.assertTrue(is_snan_f64(snan_bits))
        self.assertFalse(is_qnan_f64(snan_bits))

        self.assertTrue(is_nan_f64(qnan_bits))
        self.assertTrue(is_qnan_f64(qnan_bits))
        self.assertFalse(is_snan_f64(qnan_bits))

        subnorm_bits = 0x0000_0000_0000_0001
        self.assertTrue(is_subnormal_f64(subnorm_bits))
        self.assertFalse(is_subnormal_f64(POS_ZERO_F64))
        self.assertFalse(is_subnormal_f64(f64_to_bits(1.0)))

    def test_exact_finite_fraction(self) -> None:
        # fract(1.25) == 0.25
        res = oracle_v_fract_f64(f64_to_bits(1.25))
        self.assertEqual(bits_to_f64(res), 0.25)

        # fract(3.75) == 0.75
        res = oracle_v_fract_f64(f64_to_bits(3.75))
        self.assertEqual(bits_to_f64(res), 0.75)

        # fract(-0.25) == 0.75
        res = oracle_v_fract_f64(f64_to_bits(-0.25))
        self.assertEqual(bits_to_f64(res), 0.75)

        # fract(-1.25) == 0.75
        res = oracle_v_fract_f64(f64_to_bits(-1.25))
        self.assertEqual(bits_to_f64(res), 0.75)

        # fract(-0.75) == 0.25
        res = oracle_v_fract_f64(f64_to_bits(-0.75))
        self.assertEqual(bits_to_f64(res), 0.25)

    def test_signed_zero_and_exact_integers(self) -> None:
        # fract(+0.0) -> +0.0
        self.assertEqual(oracle_v_fract_f64(POS_ZERO_F64), POS_ZERO_F64)
        # fract(-0.0) -> +0.0 (not -0.0)
        self.assertEqual(oracle_v_fract_f64(NEG_ZERO_F64), POS_ZERO_F64)

        for integer_val in [1.0, -1.0, 2.0, -2.0, 10.0, -10.0, 1024.0, -1024.0, 65536.0]:
            with self.subTest(integer_val=integer_val):
                res = oracle_v_fract_f64(f64_to_bits(integer_val))
                self.assertEqual(res, POS_ZERO_F64)

    def test_rounding_to_one_clamped_to_largest_below_one(self) -> None:
        # Tiny negative inputs whose exact x - floor(x) rounds to 1.0 in binary64
        # must clamp to LARGEST_BELOW_ONE_F64 (0x3FEFFFFFFFFFFFFF)
        tiny_negatives = [
            -1e-16,
            -1e-18,
            -1e-25,
            -2.0 ** -54,
            -2.0 ** -60,
            -2.0 ** -100,
            -2.0 ** -1074,  # negative min subnormal
        ]
        for val in tiny_negatives:
            with self.subTest(val=val):
                bits = f64_to_bits(val)
                res = oracle_v_fract_f64(bits)
                self.assertEqual(res, LARGEST_BELOW_ONE_F64)

    def test_input_modifiers(self) -> None:
        # abs: fract(|-1.25|) = fract(1.25) = 0.25
        res_abs = oracle_v_fract_f64(f64_to_bits(-1.25), abs0=True)
        self.assertEqual(bits_to_f64(res_abs), 0.25)

        # neg: fract(-(1.25)) = fract(-1.25) = 0.75
        res_neg = oracle_v_fract_f64(f64_to_bits(1.25), neg0=True)
        self.assertEqual(bits_to_f64(res_neg), 0.75)

        # abs before neg: -|1.25| = -1.25 -> 0.75; -|-1.25| = -1.25 -> 0.75
        res_neg_abs1 = oracle_v_fract_f64(f64_to_bits(1.25), abs0=True, neg0=True)
        res_neg_abs2 = oracle_v_fract_f64(f64_to_bits(-1.25), abs0=True, neg0=True)
        self.assertEqual(bits_to_f64(res_neg_abs1), 0.75)
        self.assertEqual(bits_to_f64(res_neg_abs2), 0.75)

    def test_clamp_modifier(self) -> None:
        # Normal positive fractions stay unchanged
        res = oracle_v_fract_f64(f64_to_bits(1.25), clamp=True)
        self.assertEqual(bits_to_f64(res), 0.25)

        # NaN with clamp becomes +0.0
        res_nan = oracle_v_fract_f64(CANONICAL_QNAN_F64, clamp=True)
        self.assertEqual(res_nan, POS_ZERO_F64)

        # Inf with clamp becomes +0.0
        res_inf = oracle_v_fract_f64(POS_INF_F64, clamp=True)
        self.assertEqual(res_inf, POS_ZERO_F64)

    def test_infinities_and_nans(self) -> None:
        # fract(+-inf) -> Canonical Quiet NaN
        self.assertEqual(oracle_v_fract_f64(POS_INF_F64), CANONICAL_QNAN_F64)
        self.assertEqual(oracle_v_fract_f64(NEG_INF_F64), CANONICAL_QNAN_F64)

        # Signaling NaN quietened
        snan = 0x7FF0_0000_1234_5678
        res_snan = oracle_v_fract_f64(snan)
        self.assertTrue(is_qnan_f64(res_snan))
        self.assertEqual(res_snan, snan | QNAN_BIT_F64)

        # Quiet NaN payload preserved
        qnan = 0x7FF8_0000_ABCD_EF01
        res_qnan = oracle_v_fract_f64(qnan)
        self.assertEqual(res_qnan, qnan)


class TestValuFractF64SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_regpair_validation_and_naming(self) -> None:
        v0 = RegPair(0, RegType.VGPR)
        self.assertEqual(v0.name, "v[0:1]")
        self.assertEqual(v0.lo_reg, 0)
        self.assertEqual(v0.hi_reg, 1)

        s8 = RegPair(8, RegType.SGPR)
        self.assertEqual(s8.name, "s[8:9]")
        self.assertEqual(s8.lo_reg, 8)
        self.assertEqual(s8.hi_reg, 9)

        with self.assertRaises(ValueError):
            RegPair(255, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegPair(-1, RegType.VGPR)

    def test_operand_disassembly(self) -> None:
        v = RegPair(2, RegType.VGPR)
        op_plain = VOP3Operand(v)
        self.assertEqual(op_plain.disassemble(), "v[2:3]")

        op_abs = VOP3Operand(v, abs_mod=True)
        self.assertEqual(op_abs.disassemble(), "|v[2:3]|")

        op_neg = VOP3Operand(v, neg_mod=True)
        self.assertEqual(op_neg.disassemble(), "-v[2:3]")

        op_neg_abs = VOP3Operand(v, abs_mod=True, neg_mod=True)
        self.assertEqual(op_neg_abs.disassemble(), "-|v[2:3]|")

        lit = Literal64.from_float(1.5)
        op_lit = VOP3Operand(lit)
        self.assertEqual(op_lit.disassemble(), "1.5")

    def test_v_fract_f64_ast_construction_and_validation(self) -> None:
        dst = RegPair(0, RegType.VGPR)
        src0 = VOP3Operand(RegPair(2, RegType.VGPR), neg_mod=True, abs_mod=True)
        instr = VFractF64Instruction(dst=dst, src0=src0, clamp=True)

        self.assertEqual(instr.disassemble(), "v_fract_f64 v[0:1], -|v[2:3]| clamp")

        # Invalid destination type (SGPR)
        with self.assertRaises(TypeError):
            VFractF64Instruction(dst=RegPair(0, RegType.SGPR), src0=src0)

    def test_structural_equality(self) -> None:
        i1 = VFractF64Instruction(
            RegPair(0, RegType.VGPR),
            VOP3Operand(RegPair(2, RegType.VGPR)),
        )
        i2 = VFractF64Instruction(
            RegPair(0, RegType.VGPR),
            VOP3Operand(RegPair(2, RegType.VGPR)),
        )
        self.assertEqual(i1, i2)


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_scalar_vgpr_fraction(self) -> None:
        self.state.set_vgpr64(2, f64_to_bits(12.25))
        instr = VFractF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand(RegPair(2, RegType.VGPR)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(bits_to_f64(res_bits), 0.25)
        self.assertEqual(self.state.get_vgpr64(0), res_bits)

    def test_scalar_sgpr_fraction(self) -> None:
        self.state.set_sgpr64(8, f64_to_bits(-1.25))
        instr = VFractF64Instruction(
            dst=RegPair(4, RegType.VGPR),
            src0=VOP3Operand(RegPair(8, RegType.SGPR)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(bits_to_f64(res_bits), 0.75)
        self.assertEqual(self.state.get_vgpr64(4), res_bits)

    def test_scalar_literal_fraction(self) -> None:
        instr = VFractF64Instruction(
            dst=RegPair(6, RegType.VGPR),
            src0=VOP3Operand(Literal64.from_float(5.625)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(bits_to_f64(res_bits), 0.625)
        self.assertEqual(self.state.get_vgpr64(6), res_bits)

    def test_scalar_with_modifiers_and_clamp(self) -> None:
        self.state.set_vgpr64(0, f64_to_bits(-1.75))
        # abs: |-1.75| = 1.75 -> fract is 0.75
        instr = VFractF64Instruction(
            dst=RegPair(4, RegType.VGPR),
            src0=VOP3Operand(RegPair(0, RegType.VGPR), abs_mod=True),
            clamp=True,
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(bits_to_f64(res_bits), 0.75)

    def test_scalar_determinism(self) -> None:
        runs = []
        for _ in range(50):
            st = ScalarMachineState()
            st.set_vgpr64(0, 0x400921FB54442D18)  # pi
            instr = VFractF64Instruction(
                dst=RegPair(4, RegType.VGPR),
                src0=VOP3Operand(RegPair(0, RegType.VGPR)),
            )
            runs.append(self.interp.step(instr, st))
        self.assertTrue(all(r == runs[0] for r in runs))


if __name__ == "__main__":
    unittest.main()
