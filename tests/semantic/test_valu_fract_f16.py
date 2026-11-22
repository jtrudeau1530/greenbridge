from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import List, Optional, Tuple, Union

# ==============================================================================
# IEEE 754 binary16 (Half-Precision) Architectural Constants
# ==============================================================================

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
NEAR_ONE_F16: int = 0x3BFF  # Largest binary16 strictly less than 1.0 (1.0 - 2^-11)

# Representative IEEE 754 binary16 encodings
H_POS_ZERO: int = 0x0000
H_NEG_ZERO: int = 0x8000
H_SUB_MIN: int = 0x0001           # Smallest positive subnormal (2^-24)
H_NEG_SUB_MIN: int = 0x8001
H_SUB_MAX: int = 0x03FF           # Largest positive subnormal ((1023/1024) * 2^-14)
H_NEG_SUB_MAX: int = 0x83FF
H_NORM_MIN: int = 0x0400          # Smallest positive normal (2^-14)
H_NEG_NORM_MIN: int = 0x8400
H_0P125: int = 0x3000             # 0.125
H_0P25: int = 0x3400              # 0.25
H_NEG_0P25: int = 0xB400          # -0.25
H_0P375: int = 0x3600             # 0.375
H_0P5: int = 0x3800               # 0.5
H_NEG_0P5: int = 0xB800           # -0.5
H_0P75: int = 0x3A00              # 0.75
H_NEG_0P75: int = 0xBA00          # -0.75
H_NEAR_ONE: int = 0x3BFF          # 0.99951171875
H_1P0: int = 0x3C00               # 1.0
H_NEG_1P0: int = 0xBC00           # -1.0
H_1P25: int = 0x3D00              # 1.25
H_NEG_1P25: int = 0xBD00          # -1.25
H_1P5: int = 0x3E00               # 1.5
H_NEG_1P5: int = 0xBE00           # -1.5
H_1P75: int = 0x3F00              # 1.75
H_NEG_1P75: int = 0xBF00          # -1.75
H_2P0: int = 0x4000               # 2.0
H_NEG_2P0: int = 0xC000           # -2.0
H_2P5: int = 0x4100               # 2.5
H_NEG_2P5: int = 0xC100           # -2.5
H_3P0: int = 0x4200               # 3.0
H_NEG_3P0: int = 0xC200           # -3.0
H_3P75: int = 0x4380              # 3.75
H_NEG_3P75: int = 0xC380          # -3.75
H_1023P5: int = 0x63FF            # 1023.5 (1023 + 0.5)
H_1024: int = 0x6400              # 1024.0 (all >= 1024 are integers)
H_NEG_1024: int = 0xE400
H_MAX: int = 0x7BFF               # 65504.0
H_NEG_MAX: int = 0xFBFF


def f16_to_bits(val: float) -> int:
    """Convert a Python float to 16-bit binary16 integer bits."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f16(bits: int) -> float:
    """Convert 16-bit binary16 integer bits to a Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def split_f16_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack binary16 bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK) >> 15
    exponent = (bits & EXP_MASK) >> 10
    fraction = bits & FRAC_MASK
    return sign, exponent, fraction


def is_nan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 NaN."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) != 0)


def is_qnan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 Quiet NaN."""
    return is_nan(bits) and bool(bits & QNAN_BIT)


def is_snan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 Signaling NaN."""
    return is_nan(bits) and not bool(bits & QNAN_BIT)


def is_inf(bits: int) -> bool:
    """Return True if bits represent positive or negative infinity."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) == 0)


def is_subnormal(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormal) binary16 float."""
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


def h16_to_fraction(h: int) -> Fraction:
    """Decode finite IEEE 754 binary16 bit pattern into an exact Fraction."""
    h &= 0xFFFF
    exp = (h >> 10) & 0x1F
    if exp == 0x1F:
        raise ValueError(f"Non-finite binary16 {h:#06x}")
    sign = -1 if (h >> 15) else 1
    mant = h & 0x3FF
    if exp == 0:
        return Fraction(0) if mant == 0 else Fraction(sign * mant, 1 << 24)
    return Fraction(sign * ((1 << 10) | mant)) * (Fraction(2) ** (exp - 25))


def fraction_to_h16(q: Fraction) -> int:
    """Round an exact Fraction to IEEE 754 binary16 (Round-to-Nearest-Even)."""
    if q == 0:
        return POS_ZERO
    sgn = 0x8000 if q < 0 else 0
    a = abs(q)
    e = max(a.numerator.bit_length() - a.denominator.bit_length(), -14)
    if e > -14 and Fraction(2) ** e > a:
        e -= 1
    m = a / (Fraction(2) ** (e - 10))
    n = m.numerator // m.denominator
    rem = m - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
        n += 1
    if e == -14 and n < 1024:
        return sgn | n
    if n >= 2048:
        n >>= 1
        e += 1
    if e > 15:
        return sgn | POS_INF
    return sgn | ((e + 15) << 10) | (n - 1024)


def exact_fract_f16_fraction(h: int) -> Optional[Fraction]:
    """Exact x - floor(x) as a Fraction in [0, 1), or None for +-inf / NaN."""
    if is_nan(h) or is_inf(h):
        return None
    q = h16_to_fraction(h)
    return q - math.floor(q)


def oracle_v_fract_f16(
    src_bits: int,
    abs_mod: bool = False,
    neg_mod: bool = False,
    clamp: bool = False,
    mul: int = 1,
    div: int = 1,
) -> int:
    """Reference mathematical oracle for RDNA2 v_fract_f16 half-precision fractional part.

    Computes D = S0 - floor(S0), returning values in [0.0, 1.0).
    Conforms to RDNA2 hardware semantics:
      - Clamps results that round to 1.0 to NEAR_ONE_F16 (0x3BFF).
      - ±0.0 yields +0.0 (strictly cleared sign bit).
      - ±inf yields NaN (inf - inf is invalid); clamped yields +0.0.
      - NaN yields quietened NaN; clamped yields +0.0.
      - VOP3 source modifiers (abs, neg) applied before computation.
      - VOP3 output scaling (mul/div) and clamp ([0.0, 1.0]).
    """
    s = apply_f16_modifiers(src_bits, abs_mod, neg_mod)

    # 1. NaN handling
    if is_nan(s):
        if clamp:
            return POS_ZERO
        return s | QNAN_BIT

    # 2. Infinity handling (inf - floor(inf) -> inf - inf -> Invalid -> NaN)
    if is_inf(s):
        if clamp:
            return POS_ZERO
        return CANONICAL_QNAN

    # 3. Signed zero handling: fract(+0.0) == fract(-0.0) == +0.0
    if is_zero(s):
        return POS_ZERO

    # 4. Exact rational fractional part
    frac = exact_fract_f16_fraction(s)
    assert frac is not None

    if mul == 1 and div == 1:
        res = fraction_to_h16(frac)
        # In IEEE 754 RNE, values like 1.0 - 2^-25 round to 1.0.
        # RDNA2 ISA contract guarantees fract output is strictly in [0.0, 1.0).
        if res == ONE_F16:
            return NEAR_ONE_F16
        if clamp:
            if res == NEG_ZERO:
                return POS_ZERO
        return res

    # Scaled VOP3 output modifiers
    scaled = frac * Fraction(mul, div)
    if clamp:
        scaled = min(max(scaled, Fraction(0)), Fraction(1))
    res_scaled = fraction_to_h16(scaled)
    if clamp and res_scaled == NEG_ZERO:
        return POS_ZERO
    return res_scaled


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


class RegHalf(Enum):
    LO = 0
    HI = 1


@dataclass(frozen=True)
class Reg16:
    """Represents an architectural 16-bit register operand (e.g., v0.l, v0.h, s2.l)."""
    index: int
    reg_type: RegType
    half: RegHalf = RegHalf.LO

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index {self.index} out of range [0, 255]")

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        suffix = ".l" if self.half == RegHalf.LO else ".h"
        return f"{prefix}{self.index}{suffix}"


@dataclass(frozen=True)
class Literal16:
    """Represents a 16-bit literal or inline float/integer constant."""
    bits: int

    def __post_init__(self) -> None:
        if not (0 <= self.bits <= 0xFFFF):
            raise ValueError(f"16-bit literal bits {self.bits:#x} out of range [0, 0xFFFF]")

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
class VOP1Operand:
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
class VFractF16Instruction:
    """Symbolic AST node representing a RDNA2 v_fract_f16 instruction."""
    dst: Reg16
    src: VOP1Operand
    clamp: bool = False
    mul: int = 1
    div: int = 1

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")
        if self.mul not in (1, 2, 4):
            raise ValueError(f"Unsupported mul modifier: {self.mul}")
        if self.div not in (1, 2):
            raise ValueError(f"Unsupported div modifier: {self.div}")

    def disassemble(self) -> str:
        is_vop3 = (
            self.clamp
            or self.mul != 1
            or self.div != 1
            or self.src.abs_mod
            or self.src.neg_mod
            or self.dst.half != RegHalf.LO
            or (isinstance(self.src.source, Reg16) and self.src.source.half != RegHalf.LO)
        )
        mnemonic = "v_fract_f16_e64" if is_vop3 else "v_fract_f16_e32"
        s = f"{mnemonic} {self.dst.name}, {self.src.disassemble()}"
        if self.mul != 1:
            s += f" mul:{self.mul}"
        if self.div != 1:
            s += f" div:{self.div}"
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

    def get_vgpr16(self, index: int, half: RegHalf) -> int:
        dword = self.vgprs[index] & 0xFFFF_FFFF
        return ((dword >> 16) & 0xFFFF) if half == RegHalf.HI else (dword & 0xFFFF)

    def set_vgpr16(self, index: int, half: RegHalf, bits: int) -> None:
        dword = self.vgprs[index] & 0xFFFF_FFFF
        bits16 = bits & 0xFFFF
        if half == RegHalf.HI:
            self.vgprs[index] = (dword & 0x0000_FFFF) | (bits16 << 16)
        else:
            self.vgprs[index] = (dword & 0xFFFF_0000) | bits16

    def get_sgpr16(self, index: int, half: RegHalf) -> int:
        dword = self.sgprs[index] & 0xFFFF_FFFF
        return ((dword >> 16) & 0xFFFF) if half == RegHalf.HI else (dword & 0xFFFF)

    def set_sgpr16(self, index: int, half: RegHalf, bits: int) -> None:
        dword = self.sgprs[index] & 0xFFFF_FFFF
        bits16 = bits & 0xFFFF
        if half == RegHalf.HI:
            self.sgprs[index] = (dword & 0x0000_FFFF) | (bits16 << 16)
        else:
            self.sgprs[index] = (dword & 0xFFFF_0000) | bits16


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_fract_f16."""

    def resolve_operand(self, op: VOP1Operand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg16):
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

    def step(self, instr: VFractF16Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s_raw = self.resolve_operand(instr.src, state)

        # Computation phase
        result_bits = oracle_v_fract_f16(
            src_bits=s_raw,
            abs_mod=instr.src.abs_mod,
            neg_mod=instr.src.neg_mod,
            clamp=instr.clamp,
            mul=instr.mul,
            div=instr.div,
        )

        # Write-back phase (preserving untouched half of destination dword)
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

    def get_vgpr16(self, lane: int, index: int, half: RegHalf) -> int:
        dword = self.vgprs[lane][index] & 0xFFFF_FFFF
        return ((dword >> 16) & 0xFFFF) if half == RegHalf.HI else (dword & 0xFFFF)

    def set_vgpr16(self, lane: int, index: int, half: RegHalf, bits: int) -> None:
        dword = self.vgprs[lane][index] & 0xFFFF_FFFF
        bits16 = bits & 0xFFFF
        if half == RegHalf.HI:
            self.vgprs[lane][index] = (dword & 0x0000_FFFF) | (bits16 << 16)
        else:
            self.vgprs[lane][index] = (dword & 0xFFFF_0000) | bits16

    def get_sgpr16(self, index: int, half: RegHalf) -> int:
        dword = self.sgprs[index] & 0xFFFF_FFFF
        return ((dword >> 16) & 0xFFFF) if half == RegHalf.HI else (dword & 0xFFFF)

    def set_sgpr16(self, index: int, half: RegHalf, bits: int) -> None:
        dword = self.sgprs[index] & 0xFFFF_FFFF
        bits16 = bits & 0xFFFF
        if half == RegHalf.HI:
            self.sgprs[index] = (dword & 0x0000_FFFF) | (bits16 << 16)
        else:
            self.sgprs[index] = (dword & 0xFFFF_0000) | bits16


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VOP1Operand, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, Reg16):
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

    def step(self, instr: VFractF16Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * state.num_lanes

        # Atomic read phase across all lanes
        src_vals = [
            self.resolve_lane_operand(instr.src, state, lane)
            for lane in range(state.num_lanes)
        ]

        # Execution and selective write-back phase
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = oracle_v_fract_f16(
                    src_bits=src_vals[lane],
                    abs_mod=instr.src.abs_mod,
                    neg_mod=instr.src.neg_mod,
                    clamp=instr.clamp,
                    mul=instr.mul,
                    div=instr.div,
                )
                state.set_vgpr16(lane, instr.dst.index, instr.dst.half, res)
                results[lane] = res

        return results


# ==============================================================================
# Unit Tests
# ==============================================================================

class TestReferenceOracleF16(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_fraction_roundtrip_finite_f16(self) -> None:
        samples = [
            H_POS_ZERO, H_NEG_ZERO,
            H_SUB_MIN, H_NEG_SUB_MIN, H_SUB_MAX, H_NEG_SUB_MAX,
            H_NORM_MIN, H_NEG_NORM_MIN,
            H_0P125, H_0P25, H_NEG_0P25, H_0P375,
            H_0P5, H_NEG_0P5, H_0P75, H_NEG_0P75,
            H_NEAR_ONE, H_1P0, H_NEG_1P0,
            H_1P25, H_NEG_1P25, H_1P5, H_NEG_1P5, H_1P75, H_NEG_1P75,
            H_2P0, H_NEG_2P0, H_2P5, H_NEG_2P5, H_3P0, H_NEG_3P0,
            H_3P75, H_NEG_3P75, H_1023P5, H_1024, H_NEG_1024,
            H_MAX, H_NEG_MAX,
        ]
        for h in samples:
            with self.subTest(h=hex(h)):
                q = h16_to_fraction(h)
                expected_float = bits_to_f16(h)
                self.assertEqual(q, Fraction(expected_float))
                want = H_POS_ZERO if h == H_NEG_ZERO else h
                self.assertEqual(fraction_to_h16(q), want)

    def test_hand_worked_fract_rows(self) -> None:
        rows = [
            (H_POS_ZERO, H_POS_ZERO),
            (H_NEG_ZERO, H_POS_ZERO),
            (H_0P25, H_0P25),
            (H_NEG_0P25, H_0P75),
            (H_0P5, H_0P5),
            (H_NEG_0P5, H_0P5),
            (H_0P75, H_0P75),
            (H_NEG_0P75, H_0P25),
            (H_1P0, H_POS_ZERO),
            (H_NEG_1P0, H_POS_ZERO),
            (H_1P25, H_0P25),
            (H_NEG_1P25, H_0P75),
            (H_1P75, H_0P75),
            (H_NEG_1P75, H_0P25),
            (H_2P0, H_POS_ZERO),
            (H_NEG_2P0, H_POS_ZERO),
            (H_2P5, H_0P5),
            (H_NEG_2P5, H_0P5),
            (H_3P75, H_0P75),
            (H_NEG_3P75, H_0P25),
        ]
        for inp, expected in rows:
            with self.subTest(inp=hex(inp)):
                self.assertEqual(oracle_v_fract_f16(inp), expected)

    def test_subnormals_and_near_one_clamp(self) -> None:
        # Positive subnormals: floor(x) == 0, so fract(x) == x
        self.assertEqual(oracle_v_fract_f16(H_SUB_MIN), H_SUB_MIN)
        self.assertEqual(oracle_v_fract_f16(H_SUB_MAX), H_SUB_MAX)
        self.assertEqual(oracle_v_fract_f16(H_NORM_MIN), H_NORM_MIN)

        # Negative subnormals: x - floor(x) = 1.0 - |x|, which rounds to 1.0
        # and clamps to NEAR_ONE_F16 (0x3BFF)
        self.assertEqual(oracle_v_fract_f16(H_NEG_SUB_MIN), NEAR_ONE_F16)
        self.assertEqual(oracle_v_fract_f16(H_NEG_SUB_MAX), NEAR_ONE_F16)
        self.assertEqual(oracle_v_fract_f16(H_NEG_NORM_MIN), NEAR_ONE_F16)

    def test_large_integers_mantissa_exhaustion(self) -> None:
        # In binary16, numbers with |x| >= 1024.0 have no fractional bits
        self.assertEqual(oracle_v_fract_f16(H_1024), H_POS_ZERO)
        self.assertEqual(oracle_v_fract_f16(H_NEG_1024), H_POS_ZERO)
        self.assertEqual(oracle_v_fract_f16(H_MAX), H_POS_ZERO)
        self.assertEqual(oracle_v_fract_f16(H_NEG_MAX), H_POS_ZERO)

        # 1023.5 has fractional part 0.5
        self.assertEqual(oracle_v_fract_f16(H_1023P5), H_0P5)

    def test_nan_and_inf_handling(self) -> None:
        # Signaling NaN quietened
        snan = 0x7C01
        self.assertTrue(is_snan(snan))
        res_snan = oracle_v_fract_f16(snan)
        self.assertTrue(is_qnan(res_snan))
        self.assertEqual(res_snan, snan | QNAN_BIT)

        # Quiet NaN preserved
        qnan = 0x7E55
        self.assertTrue(is_qnan(qnan))
        self.assertEqual(oracle_v_fract_f16(qnan), qnan)

        # Infs produce Canonical QNaN
        self.assertEqual(oracle_v_fract_f16(POS_INF), CANONICAL_QNAN)
        self.assertEqual(oracle_v_fract_f16(NEG_INF), CANONICAL_QNAN)

        # Clamping NaN or Inf produces POS_ZERO
        self.assertEqual(oracle_v_fract_f16(qnan, clamp=True), POS_ZERO)
        self.assertEqual(oracle_v_fract_f16(POS_INF, clamp=True), POS_ZERO)
        self.assertEqual(oracle_v_fract_f16(NEG_INF, clamp=True), POS_ZERO)

    def test_source_modifiers(self) -> None:
        # abs clears sign: fract(|-0.25|) = fract(0.25) = 0.25
        self.assertEqual(oracle_v_fract_f16(H_NEG_0P25, abs_mod=True), H_0P25)
        # neg toggles sign: fract(-(0.25)) = fract(-0.25) = 0.75
        self.assertEqual(oracle_v_fract_f16(H_0P25, neg_mod=True), H_0P75)
        # abs then neg: -|-0.25| = -0.25 -> 0.75
        self.assertEqual(oracle_v_fract_f16(H_NEG_0P25, abs_mod=True, neg_mod=True), H_0P75)

    def test_output_modifiers(self) -> None:
        # fract(0.5) * 2 = 1.0
        self.assertEqual(oracle_v_fract_f16(H_0P5, mul=2), H_1P0)
        # fract(0.5) / 2 = 0.25
        self.assertEqual(oracle_v_fract_f16(H_0P5, div=2), H_0P25)
        # fract(0.75) * 2 = 1.5 clamped to 1.0
        self.assertEqual(oracle_v_fract_f16(H_0P75, mul=2, clamp=True), H_1P0)


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_reg16_properties_and_bounds(self) -> None:
        v0_lo = Reg16(0, RegType.VGPR, RegHalf.LO)
        self.assertEqual(v0_lo.name, "v0.l")
        v5_hi = Reg16(5, RegType.VGPR, RegHalf.HI)
        self.assertEqual(v5_hi.name, "v5.h")
        s2_lo = Reg16(2, RegType.SGPR, RegHalf.LO)
        self.assertEqual(s2_lo.name, "s2.l")

        with self.assertRaises(ValueError):
            Reg16(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg16(256, RegType.VGPR)

    def test_literal16_properties(self) -> None:
        lit_zero = Literal16(POS_ZERO)
        self.assertEqual(lit_zero.name, "0.0")
        lit_neg_zero = Literal16(NEG_ZERO)
        self.assertEqual(lit_neg_zero.name, "-0.0")
        lit_half = Literal16.from_float(0.5)
        self.assertEqual(lit_half.bits, H_0P5)
        self.assertEqual(lit_half.name, "0.5")

        with self.assertRaises(ValueError):
            Literal16(0x10000)

    def test_vop1_operand_disassembly(self) -> None:
        src_raw = VOP1Operand(Reg16(1, RegType.VGPR, RegHalf.LO))
        self.assertEqual(src_raw.disassemble(), "v1.l")

        src_abs = VOP1Operand(Reg16(1, RegType.VGPR, RegHalf.LO), abs_mod=True)
        self.assertEqual(src_abs.disassemble(), "|v1.l|")

        src_neg_abs = VOP1Operand(Reg16(1, RegType.VGPR, RegHalf.LO), abs_mod=True, neg_mod=True)
        self.assertEqual(src_neg_abs.disassemble(), "-|v1.l|")

    def test_instruction_disassembly(self) -> None:
        # Simple VOP1 e32 form
        instr_e32 = VFractF16Instruction(
            dst=Reg16(0, RegType.VGPR, RegHalf.LO),
            src=VOP1Operand(Reg16(1, RegType.VGPR, RegHalf.LO)),
        )
        self.assertEqual(instr_e32.disassemble(), "v_fract_f16_e32 v0.l, v1.l")

        # VOP3 e64 form with clamp and high half
        instr_e64 = VFractF16Instruction(
            dst=Reg16(0, RegType.VGPR, RegHalf.HI),
            src=VOP1Operand(Reg16(1, RegType.VGPR, RegHalf.LO), abs_mod=True, neg_mod=True),
            clamp=True,
            mul=2,
        )
        self.assertEqual(instr_e64.disassemble(), "v_fract_f16_e64 v0.h, -|v1.l| mul:2 clamp")

    def test_instruction_type_enforcement(self) -> None:
        with self.assertRaises(TypeError):
            VFractF16Instruction(
                dst=Reg16(0, RegType.SGPR, RegHalf.LO),
                src=VOP1Operand(Reg16(1, RegType.VGPR, RegHalf.LO)),
            )

        with self.assertRaises(ValueError):
            VFractF16Instruction(
                dst=Reg16(0, RegType.VGPR, RegHalf.LO),
                src=VOP1Operand(Reg16(1, RegType.VGPR, RegHalf.LO)),
                mul=3,
            )


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_scalar_vgpr_low_and_high_half(self) -> None:
        # v1 has low half = -1.75, high half = 0.75
        self.state.vgprs[1] = (H_0P75 << 16) | H_NEG_1P75

        # v0 starts with background pattern 0xDEADBEEF
        self.state.vgprs[0] = 0xDEAD_BEEF

        # Step 1: Read v1.l (-1.75 -> 0.25) and write to v0.l
        instr1 = VFractF16Instruction(
            dst=Reg16(0, RegType.VGPR, RegHalf.LO),
            src=VOP1Operand(Reg16(1, RegType.VGPR, RegHalf.LO)),
        )
        res1 = self.interp.step(instr1, self.state)
        self.assertEqual(res1, H_0P25)
        # v0.l is H_0P25, v0.h should be unchanged (0xDEAD)
        self.assertEqual(self.state.vgprs[0], (0xDEAD << 16) | H_0P25)

        # Step 2: Read v1.h (0.75 -> 0.75) and write to v0.h
        instr2 = VFractF16Instruction(
            dst=Reg16(0, RegType.VGPR, RegHalf.HI),
            src=VOP1Operand(Reg16(1, RegType.VGPR, RegHalf.HI)),
        )
        res2 = self.interp.step(instr2, self.state)
        self.assertEqual(res2, H_0P75)
        self.assertEqual(self.state.vgprs[0], (H_0P75 << 16) | H_0P25)

    def test_scalar_sgpr_operand(self) -> None:
        self.state.set_sgpr16(4, RegHalf.LO, H_NEG_0P25)
        instr = VFractF16Instruction(
            dst=Reg16(2, RegType.VGPR, RegHalf.LO),
            src=VOP1Operand(Reg16(4, RegType.SGPR, RegHalf.LO)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, H_0P75)
        self.assertEqual(self.state.get_vgpr16(2, RegHalf.LO), H_0P75)

    def test_scalar_literal_operand(self) -> None:
        instr = VFractF16Instruction(
            dst=Reg16(5, RegType.VGPR, RegHalf.LO),
            src=VOP1Operand(Literal16.from_float(1.5)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, H_0P5)
        self.assertEqual(self.state.get_vgpr16(5, RegHalf.LO), H_0P5)

    def test_scalar_modifiers_abs_neg_clamp_mul_div(self) -> None:
        self.state.set_vgpr16(10, RegHalf.LO, H_0P75)
        # -|0.75| = -0.75 -> fract(-0.75) = 0.25 -> mul:2 -> 0.5
        instr = VFractF16Instruction(
            dst=Reg16(11, RegType.VGPR, RegHalf.LO),
            src=VOP1Operand(Reg16(10, RegType.VGPR, RegHalf.LO), abs_mod=True, neg_mod=True),
            mul=2,
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, H_0P5)
        self.assertEqual(self.state.get_vgpr16(11, RegHalf.LO), H_0P5)

    def test_scalar_determinism(self) -> None:
        runs = []
        for _ in range(50):
            st = ScalarMachineState()
            st.set_vgpr16(1, RegHalf.LO, 0x3C10)  # ~1.015625
            instr = VFractF16Instruction(
                dst=Reg16(2, RegType.VGPR, RegHalf.LO),
                src=VOP1Operand(Reg16(1, RegType.VGPR, RegHalf.LO)),
            )
            runs.append(self.interp.step(instr, st))
        self.assertTrue(all(r == runs[0] for r in runs))


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.simd_interp = SIMDInterpreter()

    def test_simd_wave32_all_lanes_active(self) -> None:
        state = SIMDState(num_lanes=32)
        # Populate lane i with float value (i + 0.25)
        for lane in range(32):
            val = float(lane) + 0.25
            state.set_vgpr16(lane, 1, RegHalf.LO, f16_to_bits(val))

        instr = VFractF16Instruction(
            dst=Reg16(0, RegType.VGPR, RegHalf.LO),
            src=VOP1Operand(Reg16(1, RegType.VGPR, RegHalf.LO)),
        )
        res = self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(res[lane], H_0P25)
            self.assertEqual(state.get_vgpr16(lane, 0, RegHalf.LO), H_0P25)

    def test_simd_wave64_alternating_exec_mask(self) -> None:
        exec_mask = 0x5555_5555_5555_5555  # Even lanes active, odd lanes inactive
        state = SIMDState(num_lanes=64, exec_mask=exec_mask)
        sentinel = 0xCAFE

        for lane in range(64):
            # Input is 2.5 on all lanes
            state.set_vgpr16(lane, 1, RegHalf.LO, H_2P5)
            # Destination initialized to sentinel
            state.set_vgpr16(lane, 0, RegHalf.LO, sentinel)

        instr = VFractF16Instruction(
            dst=Reg16(0, RegType.VGPR, RegHalf.LO),
            src=VOP1Operand(Reg16(1, RegType.VGPR, RegHalf.LO)),
        )
        res = self.simd_interp.step(instr, state)

        for lane in range(64):
            if (exec_mask & (1 << lane)) != 0:
                self.assertEqual(res[lane], H_0P5)
                self.assertEqual(state.get_vgpr16(lane, 0, RegHalf.LO), H_0P5)
            else:
                self.assertIsNone(res[lane])
                self.assertEqual(state.get_vgpr16(lane, 0, RegHalf.LO), sentinel)

    def test_simd_uniform_sgpr_broadcast_across_lanes(self) -> None:
        state = SIMDState(num_lanes=32)
        state.set_sgpr16(6, RegHalf.LO, H_NEG_1P25)  # -1.25 -> 0.75

        instr = VFractF16Instruction(
            dst=Reg16(3, RegType.VGPR, RegHalf.LO),
            src=VOP1Operand(Reg16(6, RegType.SGPR, RegHalf.LO)),
        )
        res = self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(res[lane], H_0P75)
            self.assertEqual(state.get_vgpr16(lane, 3, RegHalf.LO), H_0P75)

    def test_simd_all_lanes_inactive_exec_zero(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x0)
        sentinel = 0xBEEF
        for lane in range(32):
            state.set_vgpr16(lane, 2, RegHalf.LO, sentinel)

        instr = VFractF16Instruction(
            dst=Reg16(2, RegType.VGPR, RegHalf.LO),
            src=VOP1Operand(Literal16.from_float(0.75)),
        )
        res = self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertIsNone(res[lane])
            self.assertEqual(state.get_vgpr16(lane, 2, RegHalf.LO), sentinel)

    def test_simd_high_half_destination_preservation(self) -> None:
        state = SIMDState(num_lanes=32)
        for lane in range(32):
            # vgpr0 dword has low half 0x1234, high half sentinel
            state.vgprs[lane][0] = 0x5555_1234
            state.set_vgpr16(lane, 1, RegHalf.LO, H_1P5)  # 1.5 -> 0.5

        instr = VFractF16Instruction(
            dst=Reg16(0, RegType.VGPR, RegHalf.HI),
            src=VOP1Operand(Reg16(1, RegType.VGPR, RegHalf.LO)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr16(lane, 0, RegHalf.HI), H_0P5)
            # Low half must remain 0x1234
            self.assertEqual(state.get_vgpr16(lane, 0, RegHalf.LO), 0x1234)


class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src_same_half(self) -> None:
        # In-place execution: v0.l = fract(v0.l)
        state = ScalarMachineState()
        state.set_vgpr16(0, RegHalf.LO, H_1P75)  # 1.75 -> 0.75
        state.set_vgpr16(0, RegHalf.HI, 0xABCD)  # untouched high half

        instr = VFractF16Instruction(
            dst=Reg16(0, RegType.VGPR, RegHalf.LO),
            src=VOP1Operand(Reg16(0, RegType.VGPR, RegHalf.LO)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, H_0P75)
        self.assertEqual(state.get_vgpr16(0, RegHalf.LO), H_0P75)
        self.assertEqual(state.get_vgpr16(0, RegHalf.HI), 0xABCD)

    def test_dst_alias_src_cross_half_low_to_high(self) -> None:
        # v0.h = fract(v0.l)
        state = ScalarMachineState()
        state.set_vgpr16(0, RegHalf.LO, H_NEG_0P25)  # -0.25 -> 0.75
        state.set_vgpr16(0, RegHalf.HI, 0x1111)

        instr = VFractF16Instruction(
            dst=Reg16(0, RegType.VGPR, RegHalf.HI),
            src=VOP1Operand(Reg16(0, RegType.VGPR, RegHalf.LO)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, H_0P75)
        # v0.h is updated to 0.75, v0.l remains untouched (-0.25)
        self.assertEqual(state.get_vgpr16(0, RegHalf.HI), H_0P75)
        self.assertEqual(state.get_vgpr16(0, RegHalf.LO), H_NEG_0P25)

    def test_dst_alias_src_cross_half_high_to_low(self) -> None:
        # v0.l = fract(v0.h)
        state = ScalarMachineState()
        state.set_vgpr16(0, RegHalf.HI, H_NEG_1P75)  # -1.75 -> 0.25
        state.set_vgpr16(0, RegHalf.LO, 0x2222)

        instr = VFractF16Instruction(
            dst=Reg16(0, RegType.VGPR, RegHalf.LO),
            src=VOP1Operand(Reg16(0, RegType.VGPR, RegHalf.HI)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, H_0P25)
        self.assertEqual(state.get_vgpr16(0, RegHalf.LO), H_0P25)
        self.assertEqual(state.get_vgpr16(0, RegHalf.HI), H_NEG_1P75)

    def test_inplace_iterative_fract_stepping(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr16(0, RegHalf.LO, H_3P75)  # 3.75
        instr = VFractF16Instruction(
            dst=Reg16(0, RegType.VGPR, RegHalf.LO),
            src=VOP1Operand(Reg16(0, RegType.VGPR, RegHalf.LO)),
        )
        # First iteration: 3.75 -> 0.75
        self.scalar_interp.step(instr, state)
        self.assertEqual(state.get_vgpr16(0, RegHalf.LO), H_0P75)

        # Subsequent iterations: 0.75 -> 0.75 (idempotent fixed point)
        for _ in range(50):
            self.scalar_interp.step(instr, state)
        self.assertEqual(state.get_vgpr16(0, RegHalf.LO), H_0P75)

    def test_simd_multi_lane_aliasing(self) -> None:
        state = SIMDState(num_lanes=32)
        for lane in range(32):
            state.set_vgpr16(lane, 4, RegHalf.LO, f16_to_bits(float(lane) + 0.5))

        instr = VFractF16Instruction(
            dst=Reg16(4, RegType.VGPR, RegHalf.LO),
            src=VOP1Operand(Reg16(4, RegType.VGPR, RegHalf.LO)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr16(lane, 4, RegHalf.LO), H_0P5)

    def test_simd_cross_half_aliasing(self) -> None:
        state = SIMDState(num_lanes=32)
        for lane in range(32):
            # Low half has negative offset (-1.25 -> 0.75)
            state.set_vgpr16(lane, 0, RegHalf.LO, H_NEG_1P25)
            state.set_vgpr16(lane, 0, RegHalf.HI, 0x5555)

        instr = VFractF16Instruction(
            dst=Reg16(0, RegType.VGPR, RegHalf.HI),
            src=VOP1Operand(Reg16(0, RegType.VGPR, RegHalf.LO)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr16(lane, 0, RegHalf.HI), H_0P75)
            self.assertEqual(state.get_vgpr16(lane, 0, RegHalf.LO), H_NEG_1P25)


if __name__ == "__main__":
    unittest.main()
