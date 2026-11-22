from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import List, Optional, Tuple, Union

# IEEE 754 binary16 (half precision) architectural constants
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
MAX_FINITE_F16: int = 0x7BFF
MIN_NORMAL_F16: int = 0x0400
MIN_SUBNORMAL_F16: int = 0x0001
MAX_SUBNORMAL_F16: int = 0x03FF


def f16_to_bits(val: float) -> int:
    """Convert a Python float to 16-bit IEEE 754 half-precision integer bits."""
    if math.isnan(val):
        return CANONICAL_QNAN_F16
    if math.isinf(val):
        return POS_INF_F16 if val > 0 else NEG_INF_F16
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return POS_INF_F16 if val > 0 else NEG_INF_F16


def bits_to_f16(bits: int) -> float:
    """Convert 16-bit integer bits to a Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def split_f16_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK_F16) >> 15
    exponent = (bits & EXP_MASK_F16) >> 10
    fraction = bits & FRAC_MASK_F16
    return sign, exponent, fraction


def unpack_half_word(word: int, hi: bool) -> int:
    """Extract low (hi=False) or high (hi=True) 16-bit half from a 32-bit register word."""
    return ((word >> 16) if hi else word) & 0xFFFF


def pack_half_word(old_word: int, val16: int, hi: bool) -> int:
    """Merge 16-bit result into low or high half of 32-bit register word, preserving the other half."""
    val16 &= 0xFFFF
    if hi:
        return (old_word & 0x0000FFFF) | (val16 << 16)
    else:
        return (old_word & 0xFFFF0000) | val16


def is_nan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & FRAC_MASK_F16) != 0)


def is_qnan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision Quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_snan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision Signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """Return True if bits represent IEEE 754 half-precision positive or negative infinity."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & FRAC_MASK_F16) == 0)


def is_subnormal_f16(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormal) half-precision float."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & FRAC_MASK_F16) != 0)


def is_zero_f16(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK_F16) == 0


def apply_f16_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & 0xFFFF
    if abs_mod:
        res &= ~SIGN_MASK_F16
    if neg_mod:
        res ^= SIGN_MASK_F16
    return res


def oracle_v_sub_f16(
    src0_bits: int,
    src1_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_sub_f16 half-precision subtraction.

    Computes D = src0 - src1 conforming to IEEE 754-2008 half precision (binary16)
    with Round-to-Nearest-Even (RNE), applying input modifiers and clamp.
    """
    s0 = apply_f16_modifiers(src0_bits, abs0, neg0)
    s1 = apply_f16_modifiers(src1_bits, abs1, neg1)

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
        s0_sign = s0 & SIGN_MASK_F16
        s1_sign = s1 & SIGN_MASK_F16
        # (+inf) - (+inf) or (-inf) - (-inf) -> Invalid operation -> Canonical Quiet NaN
        if s0_sign == s1_sign:
            return POS_ZERO_F16 if clamp else CANONICAL_QNAN_F16
        # (+inf) - (-inf) = +inf
        # (-inf) - (+inf) = -inf
        res = s0
        if clamp:
            return ONE_F16 if res == POS_INF_F16 else POS_ZERO_F16
        return res
    if s0_inf:
        if clamp:
            return ONE_F16 if s0 == POS_INF_F16 else POS_ZERO_F16
        return s0
    if s1_inf:
        # finite - (+inf) = -inf; finite - (-inf) = +inf
        res = NEG_INF_F16 if s1 == POS_INF_F16 else POS_INF_F16
        if clamp:
            return ONE_F16 if res == POS_INF_F16 else POS_ZERO_F16
        return res

    # 3. Signed zero handling
    if is_zero_f16(s0) and is_zero_f16(s1):
        s0_is_neg = (s0 == NEG_ZERO_F16)
        s1_is_neg = (s1 == NEG_ZERO_F16)
        # (+0) - (+0) = +0
        # (+0) - (-0) = +0
        # (-0) - (-0) = +0 (Round-to-Nearest-Even)
        # (-0) - (+0) = -0
        if s0_is_neg and not s1_is_neg:
            res = NEG_ZERO_F16
        else:
            res = POS_ZERO_F16
        if clamp and res == NEG_ZERO_F16:
            return POS_ZERO_F16
        return res

    # 4. Finite subtraction
    f0 = bits_to_f16(s0)
    f1 = bits_to_f16(s1)
    f_res = f0 - f1
    res = f16_to_bits(f_res)

    # In round-to-nearest-even, x - x produces +0.0
    if f_res == 0.0 and not (s0 == NEG_ZERO_F16 and s1 == POS_ZERO_F16):
        res = POS_ZERO_F16

    # 5. Clamp modifier ([0.0, 1.0])
    if clamp:
        if f_res < 0.0 or res == NEG_ZERO_F16:
            res = POS_ZERO_F16
        elif f_res > 1.0 or res == POS_INF_F16:
            res = ONE_F16

    return res


class TestF16ModelAndOracle(unittest.TestCase):
    """Unit tests for binary16 bit model, predicates, and reference mathematical oracle."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 65504.0, -65504.0]
        for val in values:
            with self.subTest(val=val):
                bits = f16_to_bits(val)
                recovered = bits_to_f16(bits)
                self.assertEqual(val, recovered)
                self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_halfword_packing_and_unpacking(self) -> None:
        old_reg = 0x12345678
        val_lo = 0x3C00  # 1.0
        val_hi = 0x4000  # 2.0

        updated_lo = pack_half_word(old_reg, val_lo, hi=False)
        self.assertEqual(updated_lo & 0xFFFF, val_lo)
        self.assertEqual((updated_lo >> 16) & 0xFFFF, 0x1234)
        self.assertEqual(unpack_half_word(updated_lo, hi=False), val_lo)
        self.assertEqual(unpack_half_word(updated_lo, hi=True), 0x1234)

        updated_hi = pack_half_word(old_reg, val_hi, hi=True)
        self.assertEqual((updated_hi >> 16) & 0xFFFF, val_hi)
        self.assertEqual(updated_hi & 0xFFFF, 0x5678)
        self.assertEqual(unpack_half_word(updated_hi, hi=True), val_hi)
        self.assertEqual(unpack_half_word(updated_hi, hi=False), 0x5678)

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

        subnorm_bits = MIN_SUBNORMAL_F16
        self.assertTrue(is_subnormal_f16(subnorm_bits))
        self.assertFalse(is_subnormal_f16(POS_ZERO_F16))
        self.assertFalse(is_subnormal_f16(f16_to_bits(1.0)))

    def test_exact_finite_subtraction(self) -> None:
        c = oracle_v_sub_f16(f16_to_bits(3.75), f16_to_bits(1.5))
        self.assertEqual(bits_to_f16(c), 2.25)

        c2 = oracle_v_sub_f16(f16_to_bits(10.0), f16_to_bits(25.5))
        self.assertEqual(bits_to_f16(c2), -15.5)

    def test_signed_zero_rules(self) -> None:
        # IEEE 754: (+0) - (+0) = +0
        self.assertEqual(oracle_v_sub_f16(POS_ZERO_F16, POS_ZERO_F16), POS_ZERO_F16)
        # (-0) - (-0) = +0
        self.assertEqual(oracle_v_sub_f16(NEG_ZERO_F16, NEG_ZERO_F16), POS_ZERO_F16)
        # (+0) - (-0) = +0
        self.assertEqual(oracle_v_sub_f16(POS_ZERO_F16, NEG_ZERO_F16), POS_ZERO_F16)
        # (-0) - (+0) = -0
        self.assertEqual(oracle_v_sub_f16(NEG_ZERO_F16, POS_ZERO_F16), NEG_ZERO_F16)
        # x - x = +0
        self.assertEqual(oracle_v_sub_f16(f16_to_bits(42.0), f16_to_bits(42.0)), POS_ZERO_F16)

    def test_input_modifiers_abs_neg(self) -> None:
        val_pos = f16_to_bits(10.0)
        val_neg = f16_to_bits(-10.0)

        # abs on negative: |-10.0| - 5.0 = 5.0
        res = oracle_v_sub_f16(val_neg, f16_to_bits(5.0), abs0=True)
        self.assertEqual(bits_to_f16(res), 5.0)

        # neg on positive: -(10.0) - 5.0 = -15.0
        res = oracle_v_sub_f16(val_pos, f16_to_bits(5.0), neg0=True)
        self.assertEqual(bits_to_f16(res), -15.0)

        # abs and neg together: -|10.0| - 5.0 = -15.0; -|-10.0| - 5.0 = -15.0
        res1 = oracle_v_sub_f16(val_pos, f16_to_bits(5.0), abs0=True, neg0=True)
        res2 = oracle_v_sub_f16(val_neg, f16_to_bits(5.0), abs0=True, neg0=True)
        self.assertEqual(bits_to_f16(res1), -15.0)
        self.assertEqual(bits_to_f16(res2), -15.0)

    def test_clamp_modifier(self) -> None:
        # Clamped result > 1.0 becomes 1.0
        res_hi = oracle_v_sub_f16(f16_to_bits(2.5), f16_to_bits(0.5), clamp=True)
        self.assertEqual(res_hi, ONE_F16)

        # Clamped result < 0.0 becomes +0.0
        res_lo = oracle_v_sub_f16(f16_to_bits(0.5), f16_to_bits(1.5), clamp=True)
        self.assertEqual(res_lo, POS_ZERO_F16)

        # Clamped result within [0.0, 1.0] is unmodified
        res_mid = oracle_v_sub_f16(f16_to_bits(0.75), f16_to_bits(0.25), clamp=True)
        self.assertEqual(bits_to_f16(res_mid), 0.5)

    def test_infinity_subtraction(self) -> None:
        self.assertEqual(oracle_v_sub_f16(POS_INF_F16, f16_to_bits(1.0)), POS_INF_F16)
        self.assertEqual(oracle_v_sub_f16(NEG_INF_F16, f16_to_bits(1.0)), NEG_INF_F16)
        self.assertEqual(oracle_v_sub_f16(f16_to_bits(1.0), POS_INF_F16), NEG_INF_F16)
        self.assertEqual(oracle_v_sub_f16(f16_to_bits(1.0), NEG_INF_F16), POS_INF_F16)
        # (+inf) - (+inf) -> Canonical NaN
        self.assertEqual(oracle_v_sub_f16(POS_INF_F16, POS_INF_F16), CANONICAL_QNAN_F16)
        # (-inf) - (-inf) -> Canonical NaN
        self.assertEqual(oracle_v_sub_f16(NEG_INF_F16, NEG_INF_F16), CANONICAL_QNAN_F16)
        # (+inf) - (-inf) -> +inf
        self.assertEqual(oracle_v_sub_f16(POS_INF_F16, NEG_INF_F16), POS_INF_F16)
        # (-inf) - (+inf) -> -inf
        self.assertEqual(oracle_v_sub_f16(NEG_INF_F16, POS_INF_F16), NEG_INF_F16)

    def test_nan_quietening_and_propagation(self) -> None:
        snan = 0x7C05
        res = oracle_v_sub_f16(snan, f16_to_bits(1.0))
        self.assertTrue(is_qnan_f16(res))
        self.assertEqual(res, snan | QNAN_BIT_F16)



# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Reg:
    """Represents an architectural 16-bit register operand inside a 32-bit register."""
    index: int
    reg_type: RegType
    hi: bool = False  # False = low 16 bits (.l), True = high 16 bits (.h)

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index {self.index} out of range [0, 255]")

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        half = "h" if self.hi else "l"
        return f"{prefix}{self.index}.{half}"


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
        if self.bits == POS_ZERO_F16:
            return "0.0"
        if self.bits == NEG_ZERO_F16:
            return "-0.0"
        f = self.float_value
        if math.isnan(f):
            return "nan"
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f"{f:g}"


@dataclass(frozen=True)
class VOP3Operand16:
    """Typed expression operand with optional input modifiers (abs, neg) and op_sel."""
    source: Union[Reg, Literal16]
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
class VSubF16Instruction:
    """Symbolic AST node representing a RDNA2 v_sub_f16 instruction."""
    dst: Reg
    src0: VOP3Operand16
    src1: VOP3Operand16
    clamp: bool = False
    is_vop3: bool = True

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        if not self.is_vop3 and not self.clamp and not self.dst.hi and not self.src0.abs_mod and not self.src0.neg_mod and not self.src1.abs_mod and not self.src1.neg_mod:
            # VOP2 format
            dst_str = f"v{self.dst.index}"
            src0_str = f"v{self.src0.source.index}" if isinstance(self.src0.source, Reg) and self.src0.source.reg_type == RegType.VGPR else self.src0.source.name
            src1_str = f"v{self.src1.source.index}" if isinstance(self.src1.source, Reg) and self.src1.source.reg_type == RegType.VGPR else self.src1.source.name
            return f"v_sub_f16_e32 {dst_str}, {src0_str}, {src1_str}"
        s = f"v_sub_f16_e64 {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}"
        if self.clamp:
            s += " clamp"
        return s


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_reg_validation_and_naming(self) -> None:
        v0_l = Reg(0, RegType.VGPR, hi=False)
        self.assertEqual(v0_l.name, "v0.l")
        self.assertEqual(v0_l.index, 0)
        self.assertFalse(v0_l.hi)

        v0_h = Reg(0, RegType.VGPR, hi=True)
        self.assertEqual(v0_h.name, "v0.h")
        self.assertTrue(v0_h.hi)

        s5_l = Reg(5, RegType.SGPR, hi=False)
        self.assertEqual(s5_l.name, "s5.l")

        with self.assertRaises(ValueError):
            Reg(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(-1, RegType.VGPR)

    def test_operand_disassembly(self) -> None:
        v = Reg(2, RegType.VGPR, hi=True)
        op_plain = VOP3Operand16(v)
        self.assertEqual(op_plain.disassemble(), "v2.h")

        op_abs = VOP3Operand16(v, abs_mod=True)
        self.assertEqual(op_abs.disassemble(), "|v2.h|")

        op_neg = VOP3Operand16(v, neg_mod=True)
        self.assertEqual(op_neg.disassemble(), "-v2.h")

        op_neg_abs = VOP3Operand16(v, abs_mod=True, neg_mod=True)
        self.assertEqual(op_neg_abs.disassemble(), "-|v2.h|")

        lit = Literal16.from_float(1.5)
        op_lit = VOP3Operand16(lit)
        self.assertEqual(op_lit.disassemble(), "1.5")

    def test_v_sub_f16_ast_construction_and_validation(self) -> None:
        dst = Reg(0, RegType.VGPR, hi=False)
        src0 = VOP3Operand16(Reg(2, RegType.VGPR, hi=False), neg_mod=True, abs_mod=True)
        src1 = VOP3Operand16(Reg(4, RegType.SGPR, hi=True))
        instr = VSubF16Instruction(dst=dst, src0=src0, src1=src1, clamp=True)

        self.assertEqual(instr.disassemble(), "v_sub_f16_e64 v0.l, -|v2.l|, s4.h clamp")

        # Invalid destination type (SGPR)
        with self.assertRaises(TypeError):
            VSubF16Instruction(dst=Reg(0, RegType.SGPR), src0=src0, src1=src1)

    def test_vop2_disassembly(self) -> None:
        dst = Reg(0, RegType.VGPR, hi=False)
        src0 = VOP3Operand16(Reg(1, RegType.VGPR, hi=False))
        src1 = VOP3Operand16(Reg(2, RegType.VGPR, hi=False))
        instr = VSubF16Instruction(dst=dst, src0=src0, src1=src1, is_vop3=False)
        self.assertEqual(instr.disassemble(), "v_sub_f16_e32 v0, v1, v2")

    def test_structural_equality(self) -> None:
        i1 = VSubF16Instruction(
            Reg(0, RegType.VGPR, hi=False),
            VOP3Operand16(Reg(2, RegType.VGPR, hi=False)),
            VOP3Operand16(Literal16.from_float(0.0)),
        )
        i2 = VSubF16Instruction(
            Reg(0, RegType.VGPR, hi=False),
            VOP3Operand16(Reg(2, RegType.VGPR, hi=False)),
            VOP3Operand16(Literal16.from_float(0.0)),
        )
        self.assertEqual(i1, i2)



# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single wavefront lane."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def get_vgpr16(self, index: int, hi: bool = False) -> int:
        return unpack_half_word(self.vgprs[index], hi)

    def set_vgpr16(self, index: int, bits16: int, hi: bool = False) -> None:
        self.vgprs[index] = pack_half_word(self.vgprs[index], bits16, hi)

    def get_sgpr16(self, index: int, hi: bool = False) -> int:
        return unpack_half_word(self.sgprs[index], hi)

    def set_sgpr16(self, index: int, bits16: int, hi: bool = False) -> None:
        self.sgprs[index] = pack_half_word(self.sgprs[index], bits16, hi)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_sub_f16."""

    def resolve_operand(self, op: VOP3Operand16, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr16(op.source.index, op.source.hi)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr16(op.source.index, op.source.hi)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal16):
            return op.source.bits
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VSubF16Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)

        # Computation phase
        result_bits = oracle_v_sub_f16(
            src0_bits=s0_raw,
            src1_bits=s1_raw,
            abs0=instr.src0.abs_mod,
            neg0=instr.src0.neg_mod,
            abs1=instr.src1.abs_mod,
            neg1=instr.src1.neg_mod,
            clamp=instr.clamp,
        )

        # Write-back phase (preserves non-targeted half of destination VGPR)
        state.set_vgpr16(instr.dst.index, result_bits, instr.dst.hi)
        return result_bits


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_scalar_vgpr_vgpr_subtraction(self) -> None:
        self.state.set_vgpr16(2, f16_to_bits(12.5), hi=False)
        self.state.set_vgpr16(4, f16_to_bits(7.5), hi=False)
        instr = VSubF16Instruction(
            dst=Reg(0, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Reg(2, RegType.VGPR, hi=False)),
            src1=VOP3Operand16(Reg(4, RegType.VGPR, hi=False)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(bits_to_f16(res_bits), 5.0)
        self.assertEqual(self.state.get_vgpr16(0, hi=False), res_bits)

    def test_scalar_vgpr_sgpr_subtraction(self) -> None:
        self.state.set_vgpr16(0, f16_to_bits(100.0), hi=False)
        self.state.set_sgpr16(8, f16_to_bits(25.0), hi=True)
        instr = VSubF16Instruction(
            dst=Reg(4, RegType.VGPR, hi=True),
            src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
            src1=VOP3Operand16(Reg(8, RegType.SGPR, hi=True)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(bits_to_f16(res_bits), 75.0)
        self.assertEqual(self.state.get_vgpr16(4, hi=True), res_bits)

    def test_scalar_vgpr_literal_subtraction(self) -> None:
        self.state.set_vgpr16(6, f16_to_bits(3.0), hi=False)
        instr = VSubF16Instruction(
            dst=Reg(6, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Reg(6, RegType.VGPR, hi=False)),
            src1=VOP3Operand16(Literal16.from_float(0.5)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(bits_to_f16(res_bits), 2.5)
        self.assertEqual(self.state.get_vgpr16(6, hi=False), res_bits)

    def test_scalar_with_modifiers_and_clamp(self) -> None:
        self.state.set_vgpr16(0, f16_to_bits(1.8), hi=False)
        self.state.set_vgpr16(2, f16_to_bits(0.5), hi=False)
        # 1.8 - 0.5 = 1.3 -> clamped to 1.0
        instr = VSubF16Instruction(
            dst=Reg(4, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
            src1=VOP3Operand16(Reg(2, RegType.VGPR, hi=False)),
            clamp=True,
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, ONE_F16)

    def test_scalar_determinism(self) -> None:
        runs = []
        for _ in range(50):
            st = ScalarMachineState()
            st.set_vgpr16(0, f16_to_bits(3.140625), hi=False)
            st.set_vgpr16(2, f16_to_bits(2.71875), hi=False)
            instr = VSubF16Instruction(
                dst=Reg(4, RegType.VGPR, hi=False),
                src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
                src1=VOP3Operand16(Reg(2, RegType.VGPR, hi=False)),
            )
            runs.append(self.interp.step(instr, st))
        self.assertTrue(all(r == runs[0] for r in runs))



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

    def get_vgpr16(self, lane: int, index: int, hi: bool = False) -> int:
        return unpack_half_word(self.vgprs[lane][index], hi)

    def set_vgpr16(self, lane: int, index: int, bits16: int, hi: bool = False) -> None:
        self.vgprs[lane][index] = pack_half_word(self.vgprs[lane][index], bits16, hi)

    def get_sgpr16(self, index: int, hi: bool = False) -> int:
        return unpack_half_word(self.sgprs[index], hi)

    def set_sgpr16(self, index: int, bits16: int, hi: bool = False) -> None:
        self.sgprs[index] = pack_half_word(self.sgprs[index], bits16, hi)


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VOP3Operand16, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr16(lane, op.source.index, op.source.hi)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr16(op.source.index, op.source.hi)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal16):
            return op.source.bits
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VSubF16Instruction, state: SIMDState) -> List[Optional[int]]:
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
                res = oracle_v_sub_f16(
                    src0_bits=s0_vals[lane],
                    src1_bits=s1_vals[lane],
                    abs0=instr.src0.abs_mod,
                    neg0=instr.src0.neg_mod,
                    abs1=instr.src1.abs_mod,
                    neg1=instr.src1.neg_mod,
                    clamp=instr.clamp,
                )
                state.set_vgpr16(lane, instr.dst.index, res, instr.dst.hi)
                results[lane] = res

        return results

    def step_packed(self, dst_idx: int, src0_idx: int, src1_idx: int, state: SIMDState, clamp: bool = False) -> None:
        """Model packed half-precision execution (v_pk_sub_f16) across lanes."""
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                w0 = state.vgprs[lane][src0_idx]
                w1 = state.vgprs[lane][src1_idx]
                lo0 = unpack_half_word(w0, hi=False)
                hi0 = unpack_half_word(w0, hi=True)
                lo1 = unpack_half_word(w1, hi=False)
                hi1 = unpack_half_word(w1, hi=True)
                res_lo = oracle_v_sub_f16(lo0, lo1, clamp=clamp)
                res_hi = oracle_v_sub_f16(hi0, hi1, clamp=clamp)
                state.vgprs[lane][dst_idx] = (res_hi << 16) | res_lo


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD execution, lane divergence, and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_wave32_full_exec(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr16(lane, 0, f16_to_bits(float(lane) + 10.0), hi=False)
            state.set_vgpr16(lane, 2, f16_to_bits(10.0), hi=False)
        instr = VSubF16Instruction(
            dst=Reg(4, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
            src1=VOP3Operand16(Reg(2, RegType.VGPR, hi=False)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            expected = float(lane)
            self.assertEqual(bits_to_f16(state.get_vgpr16(lane, 4, hi=False)), expected)

    def test_wave32_divergent_masking(self) -> None:
        # Alternating mask: even lanes active (0x55555555)
        sentinel = 0xAA55
        state = SIMDState(num_lanes=32, exec_mask=0x5555_5555)
        for lane in range(32):
            state.set_vgpr16(lane, 0, f16_to_bits(float(lane) + 5.0), hi=False)
            state.set_vgpr16(lane, 2, f16_to_bits(5.0), hi=False)
            state.set_vgpr16(lane, 4, sentinel, hi=False)

        instr = VSubF16Instruction(
            dst=Reg(4, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
            src1=VOP3Operand16(Reg(2, RegType.VGPR, hi=False)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertIsNotNone(results[lane])
                self.assertEqual(bits_to_f16(state.get_vgpr16(lane, 4, hi=False)), float(lane))
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr16(lane, 4, hi=False), sentinel)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr16(lane, 0, f16_to_bits(float(lane) * 2.0 + 1.0), hi=True)
            state.set_vgpr16(lane, 2, f16_to_bits(1.0), hi=True)
        instr = VSubF16Instruction(
            dst=Reg(0, RegType.VGPR, hi=True),
            src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=True)),
            src1=VOP3Operand16(Reg(2, RegType.VGPR, hi=True)),
        )
        self.interp.step(instr, state)
        for lane in range(64):
            self.assertEqual(bits_to_f16(state.get_vgpr16(lane, 0, hi=True)), float(lane) * 2.0)

    def test_wave64_half_disabled(self) -> None:
        # Lower 32 lanes active, upper 32 lanes inactive
        sentinel = 0xBEEF
        state = SIMDState(num_lanes=64, exec_mask=0x0000_0000_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr16(lane, 0, f16_to_bits(30.0), hi=False)
            state.set_vgpr16(lane, 2, f16_to_bits(10.0), hi=False)
            state.set_vgpr16(lane, 4, sentinel, hi=False)

        instr = VSubF16Instruction(
            dst=Reg(4, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
            src1=VOP3Operand16(Reg(2, RegType.VGPR, hi=False)),
        )
        results = self.interp.step(instr, state)
        for lane in range(64):
            if lane < 32:
                self.assertIsNotNone(results[lane])
                self.assertEqual(bits_to_f16(state.get_vgpr16(lane, 4, hi=False)), 20.0)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr16(lane, 4, hi=False), sentinel)

    def test_wave_exec_all_zero(self) -> None:
        sentinel = 0x1234
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)
        for lane in range(32):
            state.set_vgpr16(lane, 0, sentinel, hi=False)
        instr = VSubF16Instruction(
            dst=Reg(0, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Literal16.from_float(5.0)),
            src1=VOP3Operand16(Literal16.from_float(2.0)),
        )
        results = self.interp.step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(state.get_vgpr16(lane, 0, hi=False), sentinel)

    def test_simd_sgpr_uniform_broadcast(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        state.set_sgpr16(10, f16_to_bits(10.0), hi=False)
        for lane in range(32):
            state.set_vgpr16(lane, 0, f16_to_bits(100.0 + float(lane)), hi=False)

        instr = VSubF16Instruction(
            dst=Reg(2, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
            src1=VOP3Operand16(Reg(10, RegType.SGPR, hi=False)),
        )
        self.interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(bits_to_f16(state.get_vgpr16(lane, 2, hi=False)), 90.0 + float(lane))

    def test_simd_packed_pk_sub_f16_execution(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            w0 = (f16_to_bits(20.0 + lane) << 16) | f16_to_bits(10.0 + lane)
            w1 = (f16_to_bits(5.0) << 16) | f16_to_bits(2.0)
            state.vgprs[lane][0] = w0
            state.vgprs[lane][1] = w1

        self.interp.step_packed(dst_idx=2, src0_idx=0, src1_idx=1, state=state)
        for lane in range(32):
            res_word = state.vgprs[lane][2]
            res_lo = unpack_half_word(res_word, hi=False)
            res_hi = unpack_half_word(res_word, hi=True)
            self.assertEqual(bits_to_f16(res_lo), 8.0 + lane)
            self.assertEqual(bits_to_f16(res_hi), 15.0 + lane)



# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src0(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr16(0, f16_to_bits(15.5), hi=False)
        state.set_vgpr16(2, f16_to_bits(4.5), hi=False)
        instr = VSubF16Instruction(
            dst=Reg(0, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
            src1=VOP3Operand16(Reg(2, RegType.VGPR, hi=False)),
        )
        self.scalar_interp.step(instr, state)
        self.assertEqual(bits_to_f16(state.get_vgpr16(0, hi=False)), 11.0)
        # src1 untouched
        self.assertEqual(bits_to_f16(state.get_vgpr16(2, hi=False)), 4.5)

    def test_dst_alias_src1(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr16(0, f16_to_bits(15.5), hi=False)
        state.set_vgpr16(2, f16_to_bits(4.5), hi=False)
        instr = VSubF16Instruction(
            dst=Reg(2, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
            src1=VOP3Operand16(Reg(2, RegType.VGPR, hi=False)),
        )
        self.scalar_interp.step(instr, state)
        self.assertEqual(bits_to_f16(state.get_vgpr16(2, hi=False)), 11.0)
        # src0 untouched
        self.assertEqual(bits_to_f16(state.get_vgpr16(0, hi=False)), 15.5)

    def test_dst_alias_both_srcs(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr16(0, f16_to_bits(16.0), hi=False)
        instr = VSubF16Instruction(
            dst=Reg(0, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
            src1=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
        )
        self.scalar_interp.step(instr, state)
        self.assertEqual(bits_to_f16(state.get_vgpr16(0, hi=False)), 0.0)
        self.assertEqual(state.get_vgpr16(0, hi=False), POS_ZERO_F16)

    def test_inplace_iterative_subtraction(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr16(0, f16_to_bits(50.0), hi=False)
        state.set_vgpr16(2, f16_to_bits(0.5), hi=False)
        instr = VSubF16Instruction(
            dst=Reg(0, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
            src1=VOP3Operand16(Reg(2, RegType.VGPR, hi=False)),
        )
        for _ in range(60):
            self.scalar_interp.step(instr, state)
        self.assertEqual(bits_to_f16(state.get_vgpr16(0, hi=False)), 20.0)

    def test_halfword_aliasing_same_vgpr(self) -> None:
        # v0 holds two halves: lo = 5.0, hi = 20.0
        state = ScalarMachineState()
        state.set_vgpr16(0, f16_to_bits(5.0), hi=False)
        state.set_vgpr16(0, f16_to_bits(20.0), hi=True)

        # v0.l = v0.h - v0.l = 20.0 - 5.0 = 15.0
        instr = VSubF16Instruction(
            dst=Reg(0, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=True)),
            src1=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(bits_to_f16(res), 15.0)
        self.assertEqual(bits_to_f16(state.get_vgpr16(0, hi=False)), 15.0)
        # High half must remain intact at 20.0
        self.assertEqual(bits_to_f16(state.get_vgpr16(0, hi=True)), 20.0)

        # Now v0.h = v0.h - v0.l = 20.0 - 15.0 = 5.0
        instr2 = VSubF16Instruction(
            dst=Reg(0, RegType.VGPR, hi=True),
            src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=True)),
            src1=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
        )
        res2 = self.scalar_interp.step(instr2, state)
        self.assertEqual(bits_to_f16(res2), 5.0)
        self.assertEqual(bits_to_f16(state.get_vgpr16(0, hi=True)), 5.0)
        # Low half must remain intact at 15.0
        self.assertEqual(bits_to_f16(state.get_vgpr16(0, hi=False)), 15.0)

    def test_simd_multi_lane_aliasing(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr16(lane, 0, f16_to_bits(float(lane) + 10.0), hi=False)
            state.set_vgpr16(lane, 2, f16_to_bits(1.0), hi=False)

        instr = VSubF16Instruction(
            dst=Reg(0, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Reg(0, RegType.VGPR, hi=False)),
            src1=VOP3Operand16(Reg(2, RegType.VGPR, hi=False)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(bits_to_f16(state.get_vgpr16(lane, 0, hi=False)), float(lane) + 9.0)



# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class FP16StatusFlags:
    """Architectural floating-point condition codes and IEEE 754 status flags for binary16."""
    invalid: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False
    zero: bool = False
    negative: bool = False
    denormal: bool = False
    infinite: bool = False
    nan: bool = False


def evaluate_v_sub_f16_flags(
    src0_bits: int,
    src1_bits: int,
    res_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> FP16StatusFlags:
    """Evaluate IEEE 754 architectural flags and condition codes for v_sub_f16."""
    s0 = apply_f16_modifiers(src0_bits, abs0, neg0)
    s1 = apply_f16_modifiers(src1_bits, abs1, neg1)

    s0_nan = is_nan_f16(s0)
    s1_nan = is_nan_f16(s1)
    s0_snan = is_snan_f16(s0)
    s1_snan = is_snan_f16(s1)
    s0_inf = is_inf_f16(s0)
    s1_inf = is_inf_f16(s1)

    invalid = False
    if s0_snan or s1_snan:
        invalid = True
    elif s0_inf and s1_inf and ((s0 & SIGN_MASK_F16) == (s1 & SIGN_MASK_F16)):
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
            if Fraction(f0) - Fraction(f1) != Fraction(res_f):
                inexact = True
        else:
            clamped_math = min(1.0, max(0.0, float(Fraction(f0) - Fraction(f1))))
            if Fraction(clamped_math) != Fraction(res_f):
                inexact = True

    return FP16StatusFlags(
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


def compute_f16_mantissa_borrow(s0_bits: int, s1_bits: int) -> Tuple[int, int]:
    """Demonstrate IEEE 754 11-bit mantissa borrow generation during subtraction."""
    _, exp0, frac0 = split_f16_bits(s0_bits)
    _, exp1, frac1 = split_f16_bits(s1_bits)
    m0 = (1 << 10) | frac0 if exp0 != 0 else frac0
    m1 = (1 << 10) | frac1 if exp1 != 0 else frac1
    if exp0 > exp1:
        m1 >>= (exp0 - exp1)
    elif exp1 > exp0:
        m0 >>= (exp1 - exp0)
    borrow_out = 1 if m0 < m1 else 0
    diff = m0 - m1
    return diff, borrow_out


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for status flags, condition codes, and borrow generation."""

    def test_exact_finite_flags(self) -> None:
        s0 = f16_to_bits(3.0)
        s1 = f16_to_bits(1.0)
        res = oracle_v_sub_f16(s0, s1)
        flags = evaluate_v_sub_f16_flags(s0, s1, res)
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
        s0 = f16_to_bits(1.0)
        # Small value that causes inexact round-to-nearest
        s1 = f16_to_bits(0.00006)
        res = oracle_v_sub_f16(s0, s1)
        flags = evaluate_v_sub_f16_flags(s0, s1, res)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.underflow)

    def test_overflow_flag(self) -> None:
        # MAX_FINITE - (-MAX_FINITE) = 65504 - (-65504) overflows to +inf
        s0 = MAX_FINITE_F16
        s1 = MAX_FINITE_F16 | SIGN_MASK_F16
        res = oracle_v_sub_f16(s0, s1)
        self.assertEqual(res, POS_INF_F16)
        flags = evaluate_v_sub_f16_flags(s0, s1, res)
        self.assertTrue(flags.overflow)
        self.assertTrue(flags.infinite)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.invalid)

    def test_underflow_and_denorm_flags(self) -> None:
        # Subnormal subtraction
        subnorm = MIN_SUBNORMAL_F16 * 2
        res = oracle_v_sub_f16(subnorm, MIN_SUBNORMAL_F16)
        self.assertEqual(res, MIN_SUBNORMAL_F16)
        flags = evaluate_v_sub_f16_flags(subnorm, MIN_SUBNORMAL_F16, res)
        self.assertTrue(flags.denormal)
        self.assertTrue(flags.underflow)
        self.assertFalse(flags.overflow)

    def test_invalid_operation_flag(self) -> None:
        # (+inf) - (+inf) is invalid
        res = oracle_v_sub_f16(POS_INF_F16, POS_INF_F16)
        flags = evaluate_v_sub_f16_flags(POS_INF_F16, POS_INF_F16, res)
        self.assertTrue(flags.invalid)
        self.assertTrue(flags.nan)

        # signaling NaN operand triggers invalid operation
        snan = 0x7C01
        res2 = oracle_v_sub_f16(snan, f16_to_bits(1.0))
        flags2 = evaluate_v_sub_f16_flags(snan, f16_to_bits(1.0), res2)
        self.assertTrue(flags2.invalid)
        self.assertTrue(flags2.nan)

    def test_zero_and_negative_flags(self) -> None:
        # (+0) - (+0)
        res_pz = oracle_v_sub_f16(POS_ZERO_F16, POS_ZERO_F16)
        flags_pz = evaluate_v_sub_f16_flags(POS_ZERO_F16, POS_ZERO_F16, res_pz)
        self.assertTrue(flags_pz.zero)
        self.assertFalse(flags_pz.negative)

        # (-0) - (+0) = -0
        res_nz = oracle_v_sub_f16(NEG_ZERO_F16, POS_ZERO_F16)
        flags_nz = evaluate_v_sub_f16_flags(NEG_ZERO_F16, POS_ZERO_F16, res_nz)
        self.assertTrue(flags_nz.zero)
        self.assertTrue(flags_nz.negative)

        # 5.0 - 15.0 = -10.0
        s0 = f16_to_bits(5.0)
        s1 = f16_to_bits(15.0)
        res_neg = oracle_v_sub_f16(s0, s1)
        flags_neg = evaluate_v_sub_f16_flags(s0, s1, res_neg)
        self.assertFalse(flags_neg.zero)
        self.assertTrue(flags_neg.negative)

    def test_mantissa_borrow_generation(self) -> None:
        # 1.0 - 1.5: m0 = 1024, m1 = 1536 -> borrow = 1
        _, borrow1 = compute_f16_mantissa_borrow(f16_to_bits(1.0), f16_to_bits(1.5))
        self.assertEqual(borrow1, 1)

        # 1.5 - 1.0: m0 = 1536, m1 = 1024 -> borrow = 0
        _, borrow0 = compute_f16_mantissa_borrow(f16_to_bits(1.5), f16_to_bits(1.0))
        self.assertEqual(borrow0, 0)



# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Deep verification of IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_subtraction_without_borrow(self) -> None:
        sub1 = MIN_SUBNORMAL_F16 * 3
        sub2 = MIN_SUBNORMAL_F16
        res = oracle_v_sub_f16(sub1, sub2)
        self.assertEqual(res, MIN_SUBNORMAL_F16 * 2)
        self.assertTrue(is_subnormal_f16(res))

    def test_normal_to_subnormal_transition(self) -> None:
        # Min normal (0x0400 = 2^-14) minus smallest subnormal (0x0001 = 2^-24)
        # Yields largest subnormal (0x03FF)
        res = oracle_v_sub_f16(MIN_NORMAL_F16, MIN_SUBNORMAL_F16)
        self.assertEqual(res, MAX_SUBNORMAL_F16)
        self.assertTrue(is_subnormal_f16(res))
        self.assertEqual(split_f16_bits(res)[1], 0)  # exponent = 0
        self.assertEqual(split_f16_bits(res)[2], 0x03FF)  # fraction = max subnormal

    def test_subnormal_cancellation(self) -> None:
        sub = 0x0000_0123
        res = oracle_v_sub_f16(sub, sub)
        self.assertEqual(res, POS_ZERO_F16)

    def test_nan_signaling_to_quiet_conversion(self) -> None:
        snan = 0x7C05  # fraction non-zero, quiet bit 9 is 0
        self.assertTrue(is_snan_f16(snan))
        res = oracle_v_sub_f16(snan, f16_to_bits(1.0))
        self.assertTrue(is_qnan_f16(res))
        self.assertEqual(res, snan | QNAN_BIT_F16)

    def test_nan_payload_preservation(self) -> None:
        payload = 0x0255
        qnan = EXP_MASK_F16 | payload
        res = oracle_v_sub_f16(qnan, f16_to_bits(10.0))
        self.assertEqual(res, qnan)

    def test_nan_with_modifiers(self) -> None:
        qnan = 0x7E01
        # abs should clear sign bit
        res_abs = oracle_v_sub_f16(qnan | SIGN_MASK_F16, f16_to_bits(0.0), abs0=True)
        self.assertEqual(res_abs & SIGN_MASK_F16, 0)
        self.assertTrue(is_qnan_f16(res_abs))

        # neg should toggle sign bit
        res_neg = oracle_v_sub_f16(qnan, f16_to_bits(0.0), neg0=True)
        self.assertEqual(res_neg & SIGN_MASK_F16, SIGN_MASK_F16)

    def test_nan_with_clamp(self) -> None:
        qnan = CANONICAL_QNAN_F16
        res = oracle_v_sub_f16(qnan, f16_to_bits(1.0), clamp=True)
        self.assertEqual(res, POS_ZERO_F16)

    def test_infinity_subtraction_boundaries(self) -> None:
        self.assertEqual(oracle_v_sub_f16(POS_INF_F16, f16_to_bits(100.0)), POS_INF_F16)
        self.assertEqual(oracle_v_sub_f16(NEG_INF_F16, f16_to_bits(100.0)), NEG_INF_F16)
        self.assertEqual(oracle_v_sub_f16(f16_to_bits(100.0), POS_INF_F16), NEG_INF_F16)
        self.assertEqual(oracle_v_sub_f16(f16_to_bits(100.0), NEG_INF_F16), POS_INF_F16)

        self.assertEqual(oracle_v_sub_f16(POS_INF_F16, POS_INF_F16), CANONICAL_QNAN_F16)
        self.assertEqual(oracle_v_sub_f16(NEG_INF_F16, NEG_INF_F16), CANONICAL_QNAN_F16)
        self.assertEqual(oracle_v_sub_f16(POS_INF_F16, NEG_INF_F16), POS_INF_F16)
        self.assertEqual(oracle_v_sub_f16(NEG_INF_F16, POS_INF_F16), NEG_INF_F16)

        # Infinity with clamp
        self.assertEqual(oracle_v_sub_f16(POS_INF_F16, f16_to_bits(0.0), clamp=True), ONE_F16)
        self.assertEqual(oracle_v_sub_f16(NEG_INF_F16, f16_to_bits(0.0), clamp=True), POS_ZERO_F16)

    def test_round_to_nearest_even_ties(self) -> None:
        # In IEEE 754 half precision (10 bits fraction):
        # 1.0 has bits 0x3C00 (fraction 0, LSB even)
        # 1.0 + 2^-10 has bits 0x3C01 (fraction 1, LSB odd)
        # 2^-11 is halfway between representable numbers.
        half_ulp = 2.0 ** -11
        one = 1.0
        one_plus_ulp = bits_to_f16(0x3C01)

        # Case 1: (1.0 + 2^-10) - 2^-11: tie rounds to even (1.0)
        res1 = oracle_v_sub_f16(f16_to_bits(one_plus_ulp), f16_to_bits(half_ulp))
        self.assertEqual(res1, f16_to_bits(one))

    def test_finite_overflow_to_infinity(self) -> None:
        # 65504.0 - (-65504.0) overflows to +inf
        self.assertEqual(oracle_v_sub_f16(MAX_FINITE_F16, MAX_FINITE_F16 | SIGN_MASK_F16), POS_INF_F16)
        # -65504.0 - 65504.0 overflows to -inf
        self.assertEqual(oracle_v_sub_f16(MAX_FINITE_F16 | SIGN_MASK_F16, MAX_FINITE_F16), NEG_INF_F16)

    def test_extreme_finite_boundaries(self) -> None:
        self.assertEqual(oracle_v_sub_f16(MAX_FINITE_F16, MAX_FINITE_F16), POS_ZERO_F16)
        self.assertEqual(oracle_v_sub_f16(MAX_FINITE_F16 | SIGN_MASK_F16, MAX_FINITE_F16 | SIGN_MASK_F16), POS_ZERO_F16)



# ==============================================================================
# Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

def parse_vop_sub_f16(asm: str) -> VSubF16Instruction:
    """Parse a v_sub_f16 instruction assembly string with validation."""
    tokens = asm.replace(",", " ").split()
    if not tokens:
        raise ValueError("Empty assembly string")
    opcode = tokens[0]
    if opcode not in ("v_sub_f16", "v_sub_f16_e32", "v_sub_f16_e64"):
        raise ValueError(f"Unrecognized opcode: {opcode}")

    is_vop3 = opcode == "v_sub_f16_e64"
    operands = tokens[1:]
    clamp = "clamp" in operands
    if clamp:
        operands.remove("clamp")

    if len(operands) != 3:
        raise ValueError(f"Expected 3 operands (dst, src0, src1), got {len(operands)}")

    dst_str, src0_str, src1_str = operands

    def parse_reg(r_str: str) -> Reg:
        r = r_str.strip()
        hi = False
        if "." in r:
            parts = r.split(".")
            r = parts[0]
            hi = parts[1] == "h"
        prefix = r[0]
        if prefix not in ("v", "s"):
            raise ValueError(f"Invalid register prefix: {prefix}")
        reg_type = RegType.VGPR if prefix == "v" else RegType.SGPR
        idx = int(r[1:])
        return Reg(idx, reg_type, hi)

    def parse_op(op_str: str) -> VOP3Operand16:
        op_str = op_str.strip()
        abs_mod = False
        neg_mod = False
        if op_str.startswith("-"):
            neg_mod = True
            op_str = op_str[1:]
        if op_str.startswith("|") and op_str.endswith("|"):
            abs_mod = True
            op_str = op_str[1:-1]
        try:
            val = float(op_str)
            return VOP3Operand16(Literal16.from_float(val), abs_mod=abs_mod, neg_mod=neg_mod)
        except ValueError:
            return VOP3Operand16(parse_reg(op_str), abs_mod=abs_mod, neg_mod=neg_mod)

    dst_reg = parse_reg(dst_str)
    src0_op = parse_op(src0_str)
    src1_op = parse_op(src1_str)
    return VSubF16Instruction(dst=dst_reg, src0=src0_op, src1=src1_op, clamp=clamp, is_vop3=is_vop3)


def mutant_swap_operands(src0: int, src1: int) -> int:
    """Mutant A: computes reverse subtraction src1 - src0 instead of src0 - src1."""
    return oracle_v_sub_f16(src1, src0)


def mutant_ignore_abs_modifier(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Mutant B: ignores abs modifier."""
    res = bits & 0xFFFF
    if neg_mod:
        res ^= SIGN_MASK_F16
    return res


def mutant_ignore_neg_modifier(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Mutant C: ignores neg modifier."""
    res = bits & 0xFFFF
    if abs_mod:
        res &= ~SIGN_MASK_F16
    return res


def mutant_flush_subnormal(src0: int, src1: int) -> int:
    """Mutant D: flushes subnormals to zero prematurely."""
    res = oracle_v_sub_f16(src0, src1)
    if is_subnormal_f16(res):
        return POS_ZERO_F16
    return res


def mutant_skip_snan_quietening(src0: int, src1: int) -> int:
    """Mutant E: fails to quiet signaling NaN."""
    if is_snan_f16(src0):
        return src0
    if is_snan_f16(src1):
        return src1
    return oracle_v_sub_f16(src0, src1)


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection for v_sub_f16."""

    def test_syntax_rejection_invalid_opcode(self) -> None:
        with self.assertRaises(ValueError):
            parse_vop_sub_f16("v_add_f16 v0, v1, v2")

    def test_syntax_rejection_insufficient_operands(self) -> None:
        with self.assertRaises(ValueError):
            parse_vop_sub_f16("v_sub_f16 v0, v1")

    def test_syntax_rejection_out_of_range_registers(self) -> None:
        with self.assertRaises(ValueError):
            parse_vop_sub_f16("v_sub_f16 v256, v1, v2")
        with self.assertRaises(ValueError):
            parse_vop_sub_f16("v_sub_f16 v0, s256, v2")

    def test_syntax_rejection_sgpr_as_destination(self) -> None:
        with self.assertRaises(TypeError):
            parse_vop_sub_f16("v_sub_f16 s0, v1, v2")

    def test_single_bit_flip_detection(self) -> None:
        s0 = f16_to_bits(15.0)
        s1 = f16_to_bits(5.0)
        correct = oracle_v_sub_f16(s0, s1)
        self.assertEqual(bits_to_f16(correct), 10.0)

        # Corrupt single fraction bit
        corrupted_frac = correct ^ 0x1
        self.assertNotEqual(correct, corrupted_frac)

        # Corrupt single exponent bit
        corrupted_exp = correct ^ (1 << 10)
        self.assertNotEqual(correct, corrupted_exp)

        # Corrupt sign bit
        corrupted_sign = correct ^ SIGN_MASK_F16
        self.assertNotEqual(correct, corrupted_sign)

    def test_mutation_swap_operands(self) -> None:
        s0 = f16_to_bits(20.0)
        s1 = f16_to_bits(5.0)
        expected = oracle_v_sub_f16(s0, s1)
        mutant = mutant_swap_operands(s0, s1)
        self.assertEqual(bits_to_f16(expected), 15.0)
        self.assertEqual(bits_to_f16(mutant), -15.0)
        self.assertNotEqual(expected, mutant)

    def test_mutation_ignore_abs(self) -> None:
        val = f16_to_bits(-10.0)
        correct = apply_f16_modifiers(val, abs_mod=True, neg_mod=False)
        mutant = mutant_ignore_abs_modifier(val, abs_mod=True, neg_mod=False)
        self.assertEqual(correct, f16_to_bits(10.0))
        self.assertNotEqual(correct, mutant)

    def test_mutation_ignore_neg(self) -> None:
        val = f16_to_bits(10.0)
        correct = apply_f16_modifiers(val, abs_mod=False, neg_mod=True)
        mutant = mutant_ignore_neg_modifier(val, abs_mod=False, neg_mod=True)
        self.assertEqual(correct, f16_to_bits(-10.0))
        self.assertNotEqual(correct, mutant)

    def test_mutation_flush_subnormal(self) -> None:
        sub = MIN_SUBNORMAL_F16 * 2
        correct = oracle_v_sub_f16(sub, MIN_SUBNORMAL_F16)
        mutant = mutant_flush_subnormal(sub, MIN_SUBNORMAL_F16)
        self.assertTrue(is_subnormal_f16(correct))
        self.assertEqual(mutant, POS_ZERO_F16)
        self.assertNotEqual(correct, mutant)

    def test_mutation_skip_snan_quietening(self) -> None:
        snan = 0x7C05
        correct = oracle_v_sub_f16(snan, f16_to_bits(1.0))
        mutant = mutant_skip_snan_quietening(snan, f16_to_bits(1.0))
        self.assertTrue(is_qnan_f16(correct))
        self.assertTrue(is_snan_f16(mutant))
        self.assertNotEqual(correct, mutant)

    def test_mutation_exec_mask_leak(self) -> None:
        sentinel = 0xCAFE
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)
        for lane in range(32):
            state.set_vgpr16(lane, 4, sentinel, hi=False)

        class MutantSIMDInterpreter(SIMDInterpreter):
            def step(self, instr: VSubF16Instruction, state: SIMDState) -> List[Optional[int]]:
                results = []
                for lane in range(state.num_lanes):
                    # Bug: ignores exec_mask and corrupts lane
                    res = oracle_v_sub_f16(f16_to_bits(10.0), f16_to_bits(5.0))
                    state.set_vgpr16(lane, instr.dst.index, res, instr.dst.hi)
                    results.append(res)
                return results

        mutant = MutantSIMDInterpreter()
        instr = VSubF16Instruction(
            dst=Reg(4, RegType.VGPR, hi=False),
            src0=VOP3Operand16(Literal16.from_float(10.0)),
            src1=VOP3Operand16(Literal16.from_float(5.0)),
        )
        mutant.step(instr, state)
        corrupted = [l for l in range(32) if state.get_vgpr16(l, 4, hi=False) != sentinel]
        self.assertEqual(len(corrupted), 32)


if __name__ == "__main__":
    unittest.main()
