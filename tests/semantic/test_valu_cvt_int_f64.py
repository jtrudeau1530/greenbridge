from __future__ import annotations

import math
import struct
import unittest
from typing import Tuple

# ==============================================================================
# Architectural Constants for IEEE 754 Binary64 and 32-bit Integer Conversions
# ==============================================================================

SIGN_MASK_64: int = 0x8000_0000_0000_0000
EXP_MASK_64: int = 0x7FF0_0000_0000_0000
FRAC_MASK_64: int = 0x000F_FFFF_FFFF_FFFF
QNAN_BIT_64: int = 0x0008_0000_0000_0000
CANONICAL_QNAN_64: int = 0x7FF8_0000_0000_0000

POS_ZERO_64: int = 0x0000_0000_0000_0000
NEG_ZERO_64: int = 0x8000_0000_0000_0000
POS_INF_64: int = 0x7FF0_0000_0000_0000
NEG_INF_64: int = 0xFFF0_0000_0000_0000
POS_ONE_64: int = 0x3FF0_0000_0000_0000

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF

INT32_MIN: int = -0x8000_0000
INT32_MAX: int = 0x7FFF_FFFF
UINT32_MAX: int = 0xFFFF_FFFF


def f64_to_bits(val: float) -> int:
    """Convert a Python float (IEEE 754 binary64) to an unsigned 64-bit integer."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def bits_to_f64(bits: int) -> float:
    """Convert an unsigned 64-bit integer to a Python float (IEEE 754 binary64)."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def split_f64_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK_64) >> 63
    exponent = (bits & EXP_MASK_64) >> 52
    fraction = bits & FRAC_MASK_64
    return sign, exponent, fraction


def words_to_f64_bits(lo: int, hi: int) -> int:
    """Combine two 32-bit VGPR words (lo, hi) into a 64-bit integer representation."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def f64_bits_to_words(bits: int) -> Tuple[int, int]:
    """Split a 64-bit integer into two 32-bit VGPR words (lo, hi)."""
    lo = bits & MASK32
    hi = (bits >> 32) & MASK32
    return lo, hi


def i32_to_u32(val: int) -> int:
    """Convert a signed 32-bit integer to an unsigned 32-bit word representation."""
    return val & MASK32


def u32_to_i32(val: int) -> int:
    """Interpret an unsigned 32-bit word as a signed two's complement 32-bit integer."""
    val = val & MASK32
    return val if val < 0x8000_0000 else val - 0x1_0000_0000


def is_nan_f64(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 double NaN."""
    return ((bits & EXP_MASK_64) == EXP_MASK_64) and ((bits & FRAC_MASK_64) != 0)


def is_qnan_f64(bits: int) -> bool:
    """Return True if bits represent a Quiet NaN."""
    return is_nan_f64(bits) and bool(bits & QNAN_BIT_64)


def is_snan_f64(bits: int) -> bool:
    """Return True if bits represent a Signaling NaN."""
    return is_nan_f64(bits) and not bool(bits & QNAN_BIT_64)


def is_inf_f64(bits: int) -> bool:
    """Return True if bits represent positive or negative infinity."""
    return ((bits & EXP_MASK_64) == EXP_MASK_64) and ((bits & FRAC_MASK_64) == 0)


def is_subnormal_f64(bits: int) -> bool:
    """Return True if bits represent a subnormal double-precision float."""
    return ((bits & EXP_MASK_64) == 0) and ((bits & FRAC_MASK_64) != 0)


def is_zero_f64(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK_64) == 0


def apply_f64_modifiers(bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & MASK64
    if abs_mod:
        res &= ~SIGN_MASK_64
    if neg_mod:
        res ^= SIGN_MASK_64
    return res


def apply_f64_clamp(bits: int) -> int:
    """Apply VOP3 destination clamp [0.0, 1.0] modifier on f64 results."""
    if is_nan_f64(bits):
        return POS_ZERO_64
    if bits == NEG_ZERO_64:
        return POS_ZERO_64
    val = bits_to_f64(bits)
    if val < 0.0:
        return POS_ZERO_64
    if val > 1.0 or bits == POS_INF_64:
        return POS_ONE_64
    return bits


# ==============================================================================
# Reference Mathematical Oracles for Integer <-> Float64 Conversions
# ==============================================================================

def oracle_v_cvt_f64_i32(src0_i32: int, clamp: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_f64_i32.
    
    Converts a 32-bit signed integer to a 64-bit IEEE 754 float.
    Because binary64 has 53 bits of mantissa, all 32-bit integers are converted exactly.
    If clamp is True, the result is clamped to [0.0, 1.0].
    """
    s_val = u32_to_i32(src0_i32)
    f_val = float(s_val)
    res_bits = f64_to_bits(f_val)
    if clamp:
        res_bits = apply_f64_clamp(res_bits)
    return res_bits


def oracle_v_cvt_f64_u32(src0_u32: int, clamp: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_f64_u32.
    
    Converts a 32-bit unsigned integer to a 64-bit IEEE 754 float.
    All 32-bit unsigned integers are converted exactly without precision loss.
    If clamp is True, the result is clamped to [0.0, 1.0].
    """
    u_val = src0_u32 & MASK32
    f_val = float(u_val)
    res_bits = f64_to_bits(f_val)
    if clamp:
        res_bits = apply_f64_clamp(res_bits)
    return res_bits


def oracle_v_cvt_i32_f64(src0_bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_i32_f64.
    
    Converts an IEEE 754 double precision float to a 32-bit signed integer (i32).
    Applies VOP3 input modifiers (abs, neg).
    Rounding mode: Round-toward-zero (truncate).
    NaN inputs produce 0.
    Values exceeding INT32_MAX or +inf saturate to INT32_MAX (0x7FFFFFFF).
    Values below INT32_MIN or -inf saturate to INT32_MIN (0x80000000).
    """
    s0 = apply_f64_modifiers(src0_bits, abs_mod, neg_mod)
    if is_nan_f64(s0):
        return 0
    if is_zero_f64(s0):
        return 0
    if s0 == POS_INF_64:
        return INT32_MAX
    if s0 == NEG_INF_64:
        return i32_to_u32(INT32_MIN)

    val = bits_to_f64(s0)
    trunc_val = int(math.trunc(val))
    if trunc_val >= INT32_MAX:
        return INT32_MAX
    if trunc_val <= INT32_MIN:
        return i32_to_u32(INT32_MIN)
    return i32_to_u32(trunc_val)


def oracle_v_cvt_u32_f64(src0_bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_u32_f64.
    
    Converts an IEEE 754 double precision float to a 32-bit unsigned integer (u32).
    Applies VOP3 input modifiers (abs, neg).
    Rounding mode: Round-toward-zero (truncate).
    NaN inputs produce 0.
    Negative values (including -inf, -0.0, and any finite negative float) clamp to 0.
    Values exceeding UINT32_MAX or +inf saturate to UINT32_MAX (0xFFFFFFFF).
    """
    s0 = apply_f64_modifiers(src0_bits, abs_mod, neg_mod)
    if is_nan_f64(s0):
        return 0
    if is_zero_f64(s0):
        return 0
    if (s0 & SIGN_MASK_64) != 0:
        return 0
    if s0 == POS_INF_64:
        return UINT32_MAX

    val = bits_to_f64(s0)
    trunc_val = int(math.trunc(val))
    if trunc_val >= UINT32_MAX:
        return UINT32_MAX
    if trunc_val <= 0:
        return 0
    return trunc_val & MASK32



# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union


class Opcode(Enum):
    V_CVT_F64_I32 = auto()
    V_CVT_F64_U32 = auto()
    V_CVT_I32_F64 = auto()
    V_CVT_U32_F64 = auto()


class RegKind(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Register:
    """Architectural register or register pair."""
    kind: RegKind
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index {self.index} out of valid range [0, 255]")
        if self.count not in (1, 2):
            raise ValueError(f"Register count must be 1 (32-bit) or 2 (64-bit), got {self.count}")
        if self.index + self.count > 256:
            raise ValueError(f"Register {self.kind.name} {self.index} with count {self.count} exceeds index 255")

    @property
    def name(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        if self.count == 1:
            return f"{prefix}{self.index}"
        return f"{prefix}[{self.index}:{self.index + 1}]"


@dataclass(frozen=True)
class Literal32:
    """32-bit integer literal or inline constant."""
    bits: int

    @property
    def as_uint(self) -> int:
        return self.bits & MASK32

    @property
    def as_int(self) -> int:
        return u32_to_i32(self.bits)

    @property
    def name(self) -> str:
        return str(self.as_int)


@dataclass(frozen=True)
class Literal64:
    """64-bit double-precision float literal or inline constant."""
    bits: int

    @classmethod
    def from_float(cls, val: float) -> Literal64:
        return cls(f64_to_bits(val))

    @property
    def float_value(self) -> float:
        return bits_to_f64(self.bits)

    @property
    def name(self) -> str:
        if self.bits == POS_ZERO_64:
            return "0.0"
        if self.bits == NEG_ZERO_64:
            return "-0.0"
        f = self.float_value
        if math.isnan(f):
            return "nan"
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f"{f:g}"


@dataclass(frozen=True)
class VOP1Operand:
    """Typed source operand with optional VOP3 modifiers (abs, neg)."""
    source: Union[Register, Literal32, Literal64]
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
class VCvtInstruction:
    """Symbolic AST node for RDNA2 integer <-> float64 conversion instructions."""
    opcode: Opcode
    dst: Register
    src0: VOP1Operand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.kind != RegKind.VGPR:
            raise TypeError(f"VALU instruction destination must be a VGPR, got {self.dst.kind}")

        if self.opcode in (Opcode.V_CVT_F64_I32, Opcode.V_CVT_F64_U32):
            if self.dst.count != 2:
                raise ValueError(f"{self.opcode.name} destination must be a 64-bit VGPR pair, got count={self.dst.count}")
            if isinstance(self.src0.source, Register) and self.src0.source.count != 1:
                raise ValueError(f"{self.opcode.name} source register must be 32-bit (count=1), got count={self.src0.source.count}")
            if isinstance(self.src0.source, Literal64):
                raise TypeError(f"{self.opcode.name} source cannot be Literal64")
            if self.src0.abs_mod or self.src0.neg_mod:
                raise ValueError(f"{self.opcode.name} does not support float input modifiers on integer source")

        elif self.opcode in (Opcode.V_CVT_I32_F64, Opcode.V_CVT_U32_F64):
            if self.dst.count != 1:
                raise ValueError(f"{self.opcode.name} destination must be a 32-bit VGPR (count=1), got count={self.dst.count}")
            if isinstance(self.src0.source, Register) and self.src0.source.count != 2:
                raise ValueError(f"{self.opcode.name} source register must be a 64-bit pair (count=2), got count={self.src0.source.count}")
            if isinstance(self.src0.source, Literal32):
                raise TypeError(f"{self.opcode.name} source cannot be Literal32")
            if self.clamp:
                raise ValueError(f"{self.opcode.name} cannot clamp integer destination")

    def disassemble(self) -> str:
        op_name = self.opcode.name.lower()
        s = f"{op_name} {self.dst.name}, {self.src0.disassemble()}"
        if self.clamp:
            s += " clamp"
        return s


# ==============================================================================
# Unit Tests: Mathematical Model and Reference Oracles
# ==============================================================================

class TestVCvtIntF64ModelAndOracle(unittest.TestCase):
    """Unit tests for binary64 bit manipulation, integer conversion, and reference oracles."""

    def test_f64_bit_packing_and_predicates(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 42.5, -123.75, math.pi, 1e-100, 1e100]
        for v in values:
            bits = f64_to_bits(v)
            self.assertEqual(bits_to_f64(bits), v)
            lo, hi = f64_bits_to_words(bits)
            self.assertEqual(words_to_f64_bits(lo, hi), bits)

        self.assertTrue(is_zero_f64(POS_ZERO_64))
        self.assertTrue(is_zero_f64(NEG_ZERO_64))
        self.assertFalse(is_zero_f64(POS_ONE_64))

        self.assertTrue(is_inf_f64(POS_INF_64))
        self.assertTrue(is_inf_f64(NEG_INF_64))
        self.assertFalse(is_inf_f64(POS_ZERO_64))

        self.assertTrue(is_nan_f64(CANONICAL_QNAN_64))
        self.assertTrue(is_qnan_f64(CANONICAL_QNAN_64))
        self.assertFalse(is_snan_f64(CANONICAL_QNAN_64))

        snan = 0x7FF0_0000_0000_0001
        self.assertTrue(is_nan_f64(snan))
        self.assertTrue(is_snan_f64(snan))
        self.assertFalse(is_qnan_f64(snan))

        subnorm = 0x0000_0000_0000_0001
        self.assertTrue(is_subnormal_f64(subnorm))
        self.assertFalse(is_subnormal_f64(POS_ZERO_64))

    def test_i32_u32_conversions(self) -> None:
        self.assertEqual(i32_to_u32(0), 0)
        self.assertEqual(i32_to_u32(-1), 0xFFFF_FFFF)
        self.assertEqual(i32_to_u32(INT32_MIN), 0x8000_0000)
        self.assertEqual(i32_to_u32(INT32_MAX), 0x7FFF_FFFF)

        self.assertEqual(u32_to_i32(0), 0)
        self.assertEqual(u32_to_i32(0xFFFF_FFFF), -1)
        self.assertEqual(u32_to_i32(0x8000_0000), INT32_MIN)
        self.assertEqual(u32_to_i32(0x7FFF_FFFF), INT32_MAX)

    def test_oracle_v_cvt_f64_i32_exact(self) -> None:
        test_cases = [
            (0, POS_ZERO_64),
            (1, POS_ONE_64),
            (-1, f64_to_bits(-1.0)),
            (42, f64_to_bits(42.0)),
            (-1337, f64_to_bits(-1337.0)),
            (INT32_MAX, f64_to_bits(2147483647.0)),
            (i32_to_u32(INT32_MIN), f64_to_bits(-2147483648.0)),
        ]
        for src, expected in test_cases:
            with self.subTest(src=src):
                res = oracle_v_cvt_f64_i32(src)
                self.assertEqual(res, expected)
                self.assertEqual(bits_to_f64(res), bits_to_f64(expected))

    def test_oracle_v_cvt_f64_i32_clamp(self) -> None:
        # Clamping negative integer to [0.0, 1.0] produces +0.0
        self.assertEqual(oracle_v_cvt_f64_i32(-50, clamp=True), POS_ZERO_64)
        self.assertEqual(oracle_v_cvt_f64_i32(i32_to_u32(INT32_MIN), clamp=True), POS_ZERO_64)
        # Clamping 0 produces +0.0
        self.assertEqual(oracle_v_cvt_f64_i32(0, clamp=True), POS_ZERO_64)
        # Clamping 1 produces 1.0
        self.assertEqual(oracle_v_cvt_f64_i32(1, clamp=True), POS_ONE_64)
        # Clamping integer > 1 produces 1.0
        self.assertEqual(oracle_v_cvt_f64_i32(2, clamp=True), POS_ONE_64)
        self.assertEqual(oracle_v_cvt_f64_i32(INT32_MAX, clamp=True), POS_ONE_64)

    def test_oracle_v_cvt_f64_u32_exact(self) -> None:
        test_cases = [
            (0, POS_ZERO_64),
            (1, POS_ONE_64),
            (255, f64_to_bits(255.0)),
            (65535, f64_to_bits(65535.0)),
            (0x8000_0000, f64_to_bits(2147483648.0)),
            (UINT32_MAX, f64_to_bits(4294967295.0)),
        ]
        for src, expected in test_cases:
            with self.subTest(src=src):
                res = oracle_v_cvt_f64_u32(src)
                self.assertEqual(res, expected)
                self.assertEqual(bits_to_f64(res), bits_to_f64(expected))

    def test_oracle_v_cvt_f64_u32_clamp(self) -> None:
        self.assertEqual(oracle_v_cvt_f64_u32(0, clamp=True), POS_ZERO_64)
        self.assertEqual(oracle_v_cvt_f64_u32(1, clamp=True), POS_ONE_64)
        self.assertEqual(oracle_v_cvt_f64_u32(2, clamp=True), POS_ONE_64)
        self.assertEqual(oracle_v_cvt_f64_u32(UINT32_MAX, clamp=True), POS_ONE_64)

    def test_oracle_v_cvt_i32_f64_finite_and_trunc(self) -> None:
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(0.0)), 0)
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(-0.0)), 0)
        # Truncation towards zero
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(2.0)), 2)
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(2.1)), 2)
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(2.999)), 2)
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(-2.0)), i32_to_u32(-2))
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(-2.1)), i32_to_u32(-2))
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(-2.999)), i32_to_u32(-2))

    def test_oracle_v_cvt_i32_f64_saturation_and_nan(self) -> None:
        # NaN returns 0
        self.assertEqual(oracle_v_cvt_i32_f64(CANONICAL_QNAN_64), 0)
        self.assertEqual(oracle_v_cvt_i32_f64(0x7FF0_0000_1234_5678), 0)

        # Infinities
        self.assertEqual(oracle_v_cvt_i32_f64(POS_INF_64), INT32_MAX)
        self.assertEqual(oracle_v_cvt_i32_f64(NEG_INF_64), i32_to_u32(INT32_MIN))

        # Saturation thresholds
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(2147483647.0)), INT32_MAX)
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(2147483647.9)), INT32_MAX)
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(2147483648.0)), INT32_MAX)
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(1e30)), INT32_MAX)

        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(-2147483648.0)), i32_to_u32(INT32_MIN))
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(-2147483648.9)), i32_to_u32(INT32_MIN))
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(-2147483649.0)), i32_to_u32(INT32_MIN))
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(-1e30)), i32_to_u32(INT32_MIN))

    def test_oracle_v_cvt_i32_f64_modifiers(self) -> None:
        # abs modifier
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(-55.7), abs_mod=True), 55)
        # neg modifier
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(42.3), neg_mod=True), i32_to_u32(-42))
        # abs + neg modifier: -|-12.3| = -12.3 -> -12
        self.assertEqual(oracle_v_cvt_i32_f64(f64_to_bits(-12.3), abs_mod=True, neg_mod=True), i32_to_u32(-12))

    def test_oracle_v_cvt_u32_f64_finite_and_trunc(self) -> None:
        self.assertEqual(oracle_v_cvt_u32_f64(f64_to_bits(0.0)), 0)
        self.assertEqual(oracle_v_cvt_u32_f64(f64_to_bits(-0.0)), 0)
        self.assertEqual(oracle_v_cvt_u32_f64(f64_to_bits(5.0)), 5)
        self.assertEqual(oracle_v_cvt_u32_f64(f64_to_bits(5.8)), 5)
        self.assertEqual(oracle_v_cvt_u32_f64(f64_to_bits(100.99)), 100)

    def test_oracle_v_cvt_u32_f64_saturation_and_nan(self) -> None:
        # NaN returns 0
        self.assertEqual(oracle_v_cvt_u32_f64(CANONICAL_QNAN_64), 0)

        # Negative floats clamp to 0
        self.assertEqual(oracle_v_cvt_u32_f64(f64_to_bits(-0.5)), 0)
        self.assertEqual(oracle_v_cvt_u32_f64(f64_to_bits(-100.0)), 0)
        self.assertEqual(oracle_v_cvt_u32_f64(NEG_INF_64), 0)

        # Positive infinity and overflow saturate to UINT32_MAX
        self.assertEqual(oracle_v_cvt_u32_f64(POS_INF_64), UINT32_MAX)
        self.assertEqual(oracle_v_cvt_u32_f64(f64_to_bits(4294967295.0)), UINT32_MAX)
        self.assertEqual(oracle_v_cvt_u32_f64(f64_to_bits(4294967295.9)), UINT32_MAX)
        self.assertEqual(oracle_v_cvt_u32_f64(f64_to_bits(4294967296.0)), UINT32_MAX)
        self.assertEqual(oracle_v_cvt_u32_f64(f64_to_bits(1e30)), UINT32_MAX)

    def test_oracle_v_cvt_u32_f64_modifiers(self) -> None:
        # abs converts negative to positive before conversion
        self.assertEqual(oracle_v_cvt_u32_f64(f64_to_bits(-123.45), abs_mod=True), 123)
        # neg converts positive to negative -> clamps to 0
        self.assertEqual(oracle_v_cvt_u32_f64(f64_to_bits(50.0), neg_mod=True), 0)


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_creation_and_bounds(self) -> None:
        v0 = Register(RegKind.VGPR, 0, count=1)
        self.assertEqual(v0.name, "v0")
        v_pair = Register(RegKind.VGPR, 2, count=2)
        self.assertEqual(v_pair.name, "v[2:3]")

        s8 = Register(RegKind.SGPR, 8, count=1)
        self.assertEqual(s8.name, "s8")
        s_pair = Register(RegKind.SGPR, 10, count=2)
        self.assertEqual(s_pair.name, "s[10:11]")

        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 0, count=3)
        with self.assertRaises(ValueError):
            # Index 255 with count 2 spills past 255
            Register(RegKind.VGPR, 255, count=2)

    def test_literal_properties(self) -> None:
        lit32 = Literal32(0xFFFF_FFFF)
        self.assertEqual(lit32.as_uint, 0xFFFF_FFFF)
        self.assertEqual(lit32.as_int, -1)
        self.assertEqual(lit32.name, "-1")

        lit64 = Literal64.from_float(3.14)
        self.assertAlmostEqual(lit64.float_value, 3.14)
        self.assertEqual(lit64.name, "3.14")

    def test_operand_modifiers_disassembly(self) -> None:
        v_pair = Register(RegKind.VGPR, 4, count=2)
        op_plain = VOP1Operand(v_pair)
        self.assertEqual(op_plain.disassemble(), "v[4:5]")

        op_abs = VOP1Operand(v_pair, abs_mod=True)
        self.assertEqual(op_abs.disassemble(), "|v[4:5]|")

        op_neg = VOP1Operand(v_pair, neg_mod=True)
        self.assertEqual(op_neg.disassemble(), "-v[4:5]")

        op_neg_abs = VOP1Operand(v_pair, abs_mod=True, neg_mod=True)
        self.assertEqual(op_neg_abs.disassemble(), "-|v[4:5]|")

    def test_instruction_construction_all_opcodes(self) -> None:
        # v_cvt_f64_i32
        i1 = VCvtInstruction(
            opcode=Opcode.V_CVT_F64_I32,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=VOP1Operand(Register(RegKind.VGPR, 4, count=1)),
            clamp=True,
        )
        self.assertEqual(i1.disassemble(), "v_cvt_f64_i32 v[0:1], v4 clamp")

        # v_cvt_f64_u32
        i2 = VCvtInstruction(
            opcode=Opcode.V_CVT_F64_U32,
            dst=Register(RegKind.VGPR, 2, count=2),
            src0=VOP1Operand(Register(RegKind.SGPR, 6, count=1)),
        )
        self.assertEqual(i2.disassemble(), "v_cvt_f64_u32 v[2:3], s6")

        # v_cvt_i32_f64
        i3 = VCvtInstruction(
            opcode=Opcode.V_CVT_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=VOP1Operand(Register(RegKind.VGPR, 2, count=2), abs_mod=True, neg_mod=True),
        )
        self.assertEqual(i3.disassemble(), "v_cvt_i32_f64 v0, -|v[2:3]|")

        # v_cvt_u32_f64
        i4 = VCvtInstruction(
            opcode=Opcode.V_CVT_U32_F64,
            dst=Register(RegKind.VGPR, 10, count=1),
            src0=VOP1Operand(Literal64.from_float(10.5)),
        )
        self.assertEqual(i4.disassemble(), "v_cvt_u32_f64 v10, 10.5")

    def test_instruction_validation_errors(self) -> None:
        # Non-VGPR destination
        with self.assertRaises(TypeError):
            VCvtInstruction(
                opcode=Opcode.V_CVT_F64_I32,
                dst=Register(RegKind.SGPR, 0, count=2),
                src0=VOP1Operand(Register(RegKind.VGPR, 2, count=1)),
            )

        # Count mismatch: v_cvt_f64_i32 requires dst count=2
        with self.assertRaises(ValueError):
            VCvtInstruction(
                opcode=Opcode.V_CVT_F64_I32,
                dst=Register(RegKind.VGPR, 0, count=1),
                src0=VOP1Operand(Register(RegKind.VGPR, 2, count=1)),
            )

        # Count mismatch: v_cvt_i32_f64 requires dst count=1
        with self.assertRaises(ValueError):
            VCvtInstruction(
                opcode=Opcode.V_CVT_I32_F64,
                dst=Register(RegKind.VGPR, 0, count=2),
                src0=VOP1Operand(Register(RegKind.VGPR, 2, count=2)),
            )

        # Clamp invalid on integer destination
        with self.assertRaises(ValueError):
            VCvtInstruction(
                opcode=Opcode.V_CVT_I32_F64,
                dst=Register(RegKind.VGPR, 0, count=1),
                src0=VOP1Operand(Register(RegKind.VGPR, 2, count=2)),
                clamp=True,
            )

    def test_structural_equality(self) -> None:
        inst_a = VCvtInstruction(
            opcode=Opcode.V_CVT_F64_U32,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=VOP1Operand(Literal32(42)),
        )
        inst_b = VCvtInstruction(
            opcode=Opcode.V_CVT_F64_U32,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=VOP1Operand(Literal32(42)),
        )
        self.assertEqual(inst_a, inst_b)


if __name__ == "__main__":
    unittest.main()
