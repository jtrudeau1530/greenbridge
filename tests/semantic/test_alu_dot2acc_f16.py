"""Semantic reference oracle and interpreter tests for RDNA2 packed half-precision dot products:
  - v_dot2acc_f16_f16: D.f16 = S0.f16[0] * S1.f16[0] + S0.f16[1] * S1.f16[1] + S2.f16

AMD RDNA2 VALU Instruction Set Architecture (GFX10):
  The v_dot2acc_f16_f16 instruction computes the dot product of two packed pairs of IEEE-754
  binary16 (half-precision) floating-point numbers from 32-bit vector registers S0 and S1,
  and adds an accumulator S2 into the scalar 16-bit result.
  S0 contains two 16-bit half-precision floats: S0.x (bits [15:0]) and S0.y (bits [31:16]).
  S1 contains two 16-bit half-precision floats: S1.x (bits [15:0]) and S1.y (bits [31:16]).
  S2 contains a 16-bit half-precision accumulator in bits [15:0] (or selected by op_sel).
  Result D is a 16-bit float written to bits [15:0] of the destination register.

Formula:
  D = S0.x * S1.x + S0.y * S1.y + S2

Modifiers:
  - clamp: Saturates the floating-point result to [0.0, 1.0]. Values < 0.0 or -0.0 clamp to
    +0.0, values > 1.0 clamp to 1.0.
  - op_sel / op_sel_hi: Selects which 16-bit half-word of each source is read for the low/high
    product terms and accumulator.
  - neg_lo / neg_hi: Negates the chosen half-words prior to multiplication and addition.

Standard library only: Python 3.9+ compatible, zero game data, zero proprietary assets.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Bit masks
MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# Canonical IEEE-754 binary16 encodings
H_POS_ZERO = 0x0000        # +0.0
H_NEG_ZERO = 0x8000        # -0.0
H_SUB_MIN = 0x0001         # +2^-24 (~5.96046e-8, smallest positive subnormal)
H_NEG_SUB_MIN = 0x8001     # -2^-24
H_SUB_MAX = 0x03FF         # +1023 * 2^-24 (~6.09756e-5, largest subnormal)
H_NEG_SUB_MAX = 0x83FF     # -1023 * 2^-24
H_NORM_MIN = 0x0400        # +2^-14 (~6.10352e-5, smallest positive normal)
H_NEG_NORM_MIN = 0x8400    # -2^-14
H_ONE = 0x3C00             # +1.0
H_NEG_ONE = 0xBC00         # -1.0
H_HALF = 0x3800            # +0.5
H_NEG_HALF = 0xB800        # -0.5
H_TWO = 0x4000             # +2.0
H_THREE = 0x4200           # +3.0
H_FOUR = 0x4400            # +4.0
H_TEN = 0x4900             # +10.0
H_NORM_MAX = 0x7BFF        # +65504.0 (largest finite f16)
H_NEG_NORM_MAX = 0xFBFF    # -65504.0
H_POS_INF = 0x7C00         # +Infinity
H_NEG_INF = 0xFC00         # -Infinity
H_QNAN = 0x7E00            # Quiet NaN
H_SNAN = 0x7D00            # Signaling NaN


def f16_bits_to_float(bits: int) -> float:
    """Convert a 16-bit unsigned integer to an IEEE-754 Python float."""
    bits &= MASK16
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def float_to_f16_bits(val: float) -> int:
    """Convert an IEEE-754 Python float to a 16-bit unsigned integer."""
    if math.isnan(val):
        sign = 0x8000 if math.copysign(1.0, val) < 0 else 0x0000
        return sign | H_QNAN
    if math.isinf(val):
        return H_POS_INF if val > 0 else H_NEG_INF
    # Overflow boundary in IEEE-754 binary16: >= 65520.0 rounds to infinity
    if val >= 65520.0:
        return H_POS_INF
    if val <= -65520.0:
        return H_NEG_INF
    return struct.unpack("<H", struct.pack("<e", val))[0]


def pack_f16x2(lo: int, hi: int) -> int:
    """Pack two 16-bit halfwords into a 32-bit dword."""
    return (lo & MASK16) | ((hi & MASK16) << 16)


def unpack_f16x2(dword: int) -> tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit halfwords (lo, hi)."""
    dword &= MASK32
    return dword & MASK16, (dword >> 16) & MASK16


def is_f16_subnormal(bits: int) -> bool:
    """Check if 16-bit word represents a subnormal float (exp == 0, frac != 0)."""
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    return exp == 0 and frac != 0


def is_f16_nan(bits: int) -> bool:
    """Check if 16-bit word represents any NaN (exp == 31, frac != 0)."""
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    return exp == 0x1F and frac != 0


def is_f16_qnan(bits: int) -> bool:
    """Check if 16-bit word represents a quiet NaN (exp == 31, bit 9 == 1)."""
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    return exp == 0x1F and bool(frac & 0x200)


def is_f16_snan(bits: int) -> bool:
    """Check if 16-bit word represents a signaling NaN (exp == 31, frac != 0, bit 9 == 0)."""
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    return exp == 0x1F and (frac != 0) and not bool(frac & 0x200)


def is_f16_inf(bits: int) -> bool:
    """Check if 16-bit word represents +/- infinity (exp == 31, frac == 0)."""
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    return exp == 0x1F and frac == 0


def is_f16_zero(bits: int) -> bool:
    """Check if 16-bit word represents +/- 0.0."""
    return (bits & 0x7FFF) == 0


def clamp_f16(val: float) -> float:
    """Apply RDNA VALU float clamp modifier saturating to [0.0, 1.0]."""
    if math.isnan(val):
        return 0.0
    if val < 0.0 or (math.isinf(val) and val < 0) or (val == 0.0 and math.copysign(1.0, val) < 0.0):
        return 0.0
    if val > 1.0 or (math.isinf(val) and val > 0):
        return 1.0
    return val


def ref_dot2acc_f16(
    s0: int,
    s1: int,
    s2_acc: int = 0,
    clamp: bool = False,
    op_sel: tuple[int, int, int] = (0, 0, 0),
    op_sel_hi: tuple[int, int, int] = (1, 1, 1),
    neg_lo: tuple[int, int, int] = (0, 0, 0),
    neg_hi: tuple[int, int, int] = (0, 0, 0),
    flush_subnormals: bool = False,
) -> int:
    """Pure-Python reference oracle for RDNA2 v_dot2acc_f16_f16.

    Formula: D = S0.x * S1.x + S0.y * S1.y + S2
    """
    s0_halves = unpack_f16x2(s0)
    s1_halves = unpack_f16x2(s1)

    # First product term: S0.x * S1.x
    h0_x = s0_halves[op_sel[0] & 1]
    h1_x = s1_halves[op_sel[1] & 1]
    if flush_subnormals and is_f16_subnormal(h0_x):
        h0_x = h0_x & 0x8000
    if flush_subnormals and is_f16_subnormal(h1_x):
        h1_x = h1_x & 0x8000

    f0_x = f16_bits_to_float(h0_x)
    f1_x = f16_bits_to_float(h1_x)
    if neg_lo[0]:
        f0_x = -f0_x
    if neg_lo[1]:
        f1_x = -f1_x

    # Second product term: S0.y * S1.y
    h0_y = s0_halves[op_sel_hi[0] & 1]
    h1_y = s1_halves[op_sel_hi[1] & 1]
    if flush_subnormals and is_f16_subnormal(h0_y):
        h0_y = h0_y & 0x8000
    if flush_subnormals and is_f16_subnormal(h1_y):
        h1_y = h1_y & 0x8000

    f0_y = f16_bits_to_float(h0_y)
    f1_y = f16_bits_to_float(h1_y)
    if neg_hi[0]:
        f0_y = -f0_y
    if neg_hi[1]:
        f1_y = -f1_y

    # Accumulator S2
    s2_bits = s2_acc & MASK16
    if flush_subnormals and is_f16_subnormal(s2_bits):
        s2_bits = s2_bits & 0x8000

    f2 = f16_bits_to_float(s2_bits)
    if len(neg_lo) > 2 and neg_lo[2]:
        f2 = -f2

    # Multiplication terms
    p0 = f0_x * f1_x
    p1 = f0_y * f1_y

    dot = p0 + p1 + f2

    if clamp:
        dot = clamp_f16(dot)

    res_bits = float_to_f16_bits(dot)
    if flush_subnormals and is_f16_subnormal(res_bits):
        res_bits = res_bits & 0x8000
    return res_bits


class TestDot2AccF16ReferenceOracle(unittest.TestCase):
    """Exhaustive validation of reference mathematical semantics for v_dot2acc_f16_f16."""

    def test_float_conversion_roundtrip(self):
        """Sanity check f16 <-> float conversions for representative constants."""
        cases = [
            (H_POS_ZERO, 0.0),
            (H_NEG_ZERO, -0.0),
            (H_ONE, 1.0),
            (H_NEG_ONE, -1.0),
            (H_HALF, 0.5),
            (H_TWO, 2.0),
            (H_THREE, 3.0),
            (H_FOUR, 4.0),
            (H_TEN, 10.0),
            (H_NORM_MAX, 65504.0),
            (H_NEG_NORM_MAX, -65504.0),
            (H_POS_INF, float("inf")),
            (H_NEG_INF, float("-inf")),
        ]
        for bits, val in cases:
            with self.subTest(bits=hex(bits), val=val):
                converted = f16_bits_to_float(bits)
                if math.isinf(val):
                    self.assertTrue(math.isinf(converted))
                    self.assertEqual(math.copysign(1.0, converted), math.copysign(1.0, val))
                else:
                    self.assertEqual(converted, val)
                    self.assertEqual(math.copysign(1.0, converted), math.copysign(1.0, val))
                re_bits = float_to_f16_bits(converted)
                self.assertEqual(re_bits, bits)

    def test_basic_dot2acc_calculation(self):
        """Verify normal vectors dot product calculation with accumulator."""
        # [1.0, 2.0] . [3.0, 4.0] + 10.0 = 1*3 + 2*4 + 10 = 21.0
        s0 = pack_f16x2(H_ONE, H_TWO)
        s1 = pack_f16x2(H_THREE, H_FOUR)
        s2 = H_TEN
        res = ref_dot2acc_f16(s0, s1, s2)
        self.assertEqual(f16_bits_to_float(res), 21.0)

        # [0.5, 0.5] . [0.5, 0.5] + 0.5 = 0.25 + 0.25 + 0.5 = 1.0
        s0 = pack_f16x2(H_HALF, H_HALF)
        s1 = pack_f16x2(H_HALF, H_HALF)
        s2 = H_HALF
        res = ref_dot2acc_f16(s0, s1, s2)
        self.assertEqual(res, H_ONE)

    def test_orthogonal_vectors_with_accumulator(self):
        """Dot product of orthogonal vectors yields S2 exactly."""
        # [1.0, 0.0] . [0.0, 1.0] + 4.0 = 0.0 + 4.0 = 4.0
        s0 = pack_f16x2(H_ONE, H_POS_ZERO)
        s1 = pack_f16x2(H_POS_ZERO, H_ONE)
        s2 = H_FOUR
        res = ref_dot2acc_f16(s0, s1, s2)
        self.assertEqual(res, H_FOUR)

        # [2.0, -1.0] . [1.0, 2.0] + 3.0 = (2.0 - 2.0) + 3.0 = 3.0
        s0 = pack_f16x2(H_TWO, H_NEG_ONE)
        s1 = pack_f16x2(H_ONE, H_TWO)
        s2 = H_THREE
        res = ref_dot2acc_f16(s0, s1, s2)
        self.assertEqual(res, H_THREE)

    def test_zero_accumulator(self):
        """Zero accumulator reduces to simple dot product."""
        s0 = pack_f16x2(H_ONE, H_TWO)
        s1 = pack_f16x2(H_THREE, H_FOUR)
        res = ref_dot2acc_f16(s0, s1, H_POS_ZERO)
        self.assertEqual(f16_bits_to_float(res), 11.0)

    def test_zero_multiplicands(self):
        """Zero multiplicands return accumulator intact."""
        s0 = pack_f16x2(H_POS_ZERO, H_POS_ZERO)
        s1 = pack_f16x2(H_POS_ZERO, H_POS_ZERO)
        s2 = H_TEN
        res = ref_dot2acc_f16(s0, s1, s2)
        self.assertEqual(res, H_TEN)

    def test_signed_zero_rules_in_dot2acc(self):
        """Signed zero multiplication and addition semantics."""
        # (+0.0 * +1.0) + (+0.0 * +1.0) + +0.0 = +0.0
        s0 = pack_f16x2(H_POS_ZERO, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_POS_ZERO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_POS_ZERO)

        # (-0.0 * +1.0) + (-0.0 * +1.0) + -0.0 = -0.0 + -0.0 + -0.0 = -0.0
        s0 = pack_f16x2(H_NEG_ZERO, H_NEG_ZERO)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_NEG_ZERO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_NEG_ZERO)

        # (-0.0 * +1.0) + (-0.0 * +1.0) + +0.0 = -0.0 + -0.0 + +0.0 = +0.0
        s0 = pack_f16x2(H_NEG_ZERO, H_NEG_ZERO)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_POS_ZERO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_POS_ZERO)

        # (+0.0 * +1.0) + (-0.0 * +1.0) + +0.0 = +0.0 + -0.0 + +0.0 = +0.0
        s0 = pack_f16x2(H_POS_ZERO, H_NEG_ZERO)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_POS_ZERO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_POS_ZERO)

        # Exact cancellation: (-1.0 * 1.0) + (1.0 * 1.0) + +0.0 = +0.0
        s0 = pack_f16x2(H_NEG_ONE, H_ONE)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_POS_ZERO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_POS_ZERO)

    def test_subnormal_inputs_gradual_underflow(self):
        """Gradual underflow and subnormal operand handling."""
        # Subnormal min * 1.0 + 0 + 0 = Subnormal min
        s0 = pack_f16x2(H_SUB_MIN, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        s2 = H_POS_ZERO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_SUB_MIN)

        # Two subnormals summed with subnormal accumulator
        # 2^-24 + 2^-24 + 2^-24 = 3 * 2^-24 (0x0003)
        s0 = pack_f16x2(H_SUB_MIN, H_SUB_MIN)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_SUB_MIN
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), 0x0003)

    def test_infinity_propagation_and_indeterminate_forms(self):
        """Infinity arithmetic and indeterminate forms in dot2acc."""
        # [+inf, 0] . [1.0, 0] + 2.0 = +inf + 2.0 = +inf
        s0 = pack_f16x2(H_POS_INF, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        s2 = H_TWO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_POS_INF)

        # [-inf, 0] . [1.0, 0] + 2.0 = -inf + 2.0 = -inf
        s0 = pack_f16x2(H_NEG_INF, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        s2 = H_TWO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_NEG_INF)

        # (+inf) + (-inf) + 0 = NaN
        s0 = pack_f16x2(H_POS_INF, H_NEG_INF)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_POS_ZERO
        self.assertTrue(math.isnan(f16_bits_to_float(ref_dot2acc_f16(s0, s1, s2))))

        # 0.0 * inf + 0 + 0 = NaN
        s0 = pack_f16x2(H_POS_ZERO, H_POS_ZERO)
        s1 = pack_f16x2(H_POS_INF, H_POS_ZERO)
        s2 = H_POS_ZERO
        self.assertTrue(math.isnan(f16_bits_to_float(ref_dot2acc_f16(s0, s1, s2))))

    def test_source_selection_op_sel_and_negation(self):
        """op_sel and negation modifiers correctly permute and invert inputs."""
        s0 = pack_f16x2(H_ONE, H_TWO)
        s1 = pack_f16x2(H_THREE, H_FOUR)
        s2 = H_TEN
        # Default: 1*3 + 2*4 + 10 = 21.0
        self.assertEqual(f16_bits_to_float(ref_dot2acc_f16(s0, s1, s2)), 21.0)

        # Swapped S1 halves: op_sel=(0, 1, 0), op_sel_hi=(1, 0, 1) -> 1*4 + 2*3 + 10 = 20.0
        res = ref_dot2acc_f16(s0, s1, s2, op_sel=(0, 1, 0), op_sel_hi=(1, 0, 1))
        self.assertEqual(f16_bits_to_float(res), 20.0)

        # Negate accumulator: 1*3 + 2*4 - 10 = 1.0
        res_neg_acc = ref_dot2acc_f16(s0, s1, s2, neg_lo=(0, 0, 1))
        self.assertEqual(f16_bits_to_float(res_neg_acc), 1.0)

    def test_overflow_to_infinity(self):
        """Values exceeding maximum finite f16 overflow to infinity."""
        # 65504 * 1.0 + 65504 * 1.0 + 0 = 131008 -> +inf
        s0 = pack_f16x2(H_NORM_MAX, H_NORM_MAX)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = H_POS_ZERO
        self.assertEqual(ref_dot2acc_f16(s0, s1, s2), H_POS_INF)


class SimdWavefront:
    """RDNA2 SIMD Wavefront state holding VGPRs, SGPRs, and execution masks."""

    def __init__(self, wave_size: int = 64, initial_exec: int = 0xFFFFFFFFFFFFFFFF) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave size: {wave_size}")
        self.wave_size = wave_size
        self.vgpr: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgpr: list[int] = [0] * 106
        self.mask_all = (1 << wave_size) - 1
        self.exec_mask: int = initial_exec & self.mask_all

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & self.mask_all

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def set_vgpr(self, lane: int, reg: int, val: int) -> None:
        if not (0 <= reg < 256):
            raise IndexError(f"VGPR index {reg} out of range [0, 255]")
        self.vgpr[lane][reg] = val & MASK32

    def get_vgpr(self, lane: int, reg: int) -> int:
        if not (0 <= reg < 256):
            raise IndexError(f"VGPR index {reg} out of range [0, 255]")
        return self.vgpr[lane][reg]

    def set_sgpr(self, reg: int, val: int) -> None:
        if not (0 <= reg < 106):
            raise IndexError(f"SGPR index {reg} out of range [0, 105]")
        self.sgpr[reg] = val & MASK32

    def get_sgpr(self, reg: int) -> int:
        if not (0 <= reg < 106):
            raise IndexError(f"SGPR index {reg} out of range [0, 105]")
        return self.sgpr[reg]


class Dot2AccInterpreter:
    """Instruction interpreter for RDNA2 packed half-precision dot products with accumulation.

    Handles v_mov_b32, s_mov_b32, s_mov_b64 exec, v_dot2_f16_f16, and v_dot2acc_f16_f16.
    Respects SIMD lane execution masks, register aliasing, and source/destination modifiers.
    """

    def __init__(self, wave_size: int = 64, flush_subnormals: bool = False) -> None:
        self.wave = SimdWavefront(wave_size=wave_size)
        self.flush_subnormals = flush_subnormals

    def parse_reg(self, token: str) -> tuple[str, int]:
        token = token.strip().lower()
        if token.startswith("v") and token[1:].isdigit():
            idx = int(token[1:])
            if 0 <= idx < 256:
                return "vgpr", idx
            raise IndexError(f"VGPR index out of range: {token}")
        if token.startswith("s") and token[1:].isdigit():
            idx = int(token[1:])
            if 0 <= idx < 106:
                return "sgpr", idx
            raise IndexError(f"SGPR index out of range: {token}")
        if token in ("exec", "exec_lo"):
            return "exec_lo", 0
        if token == "exec_hi":
            return "exec_hi", 0
        raise ValueError(f"Unrecognized register operand: {token}")

    def parse_operand(self, token: str, lane: int) -> int:
        token = token.strip().rstrip(",")
        if not token:
            raise ValueError("Empty operand token")
        lower = token.lower()
        if lower.startswith("v") and lower[1:].isdigit():
            reg = int(lower[1:])
            return self.wave.get_vgpr(lane, reg)
        if lower.startswith("s") and lower[1:].isdigit():
            reg = int(lower[1:])
            return self.wave.get_sgpr(reg)
        if lower == "exec_lo":
            return self.wave.exec_mask & MASK32
        if lower == "exec_hi":
            return (self.wave.exec_mask >> 32) & MASK32
        if lower.startswith("0x") or lower.startswith("-0x"):
            return int(token, 16) & MASK32
        try:
            return int(token) & MASK32
        except ValueError:
            raise ValueError(f"Invalid operand token: {token}")

    def execute_line(self, line: str) -> None:
        line = line.split(";", 1)[0].split("//", 1)[0].strip()
        if not line:
            return

        parts = line.split(None, 1)
        opcode = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""

        # Tokenize preserving brackets [a,b,c]
        tokens: list[str] = []
        buf: list[str] = []
        bracket = 0
        for ch in args_str:
            if ch == "[":
                bracket += 1
                buf.append(ch)
            elif ch == "]":
                bracket = max(0, bracket - 1)
                buf.append(ch)
            elif ch in (",", " ", "\t") and bracket == 0:
                if buf:
                    tokens.append("".join(buf).strip())
                    buf = []
            else:
                buf.append(ch)
        if buf:
            tokens.append("".join(buf).strip())
        tokens = [t for t in tokens if t]

        if opcode in ("v_mov_b32", "v_mov_b32_e32", "v_mov_b32_e64"):
            if len(tokens) < 2:
                raise ValueError(f"{opcode} requires dst and src")
            dst_type, dst_reg = self.parse_reg(tokens[0])
            if dst_type != "vgpr":
                raise ValueError(f"v_mov destination must be VGPR, got {tokens[0]}")
            for lane in range(self.wave.wave_size):
                if self.wave.is_lane_active(lane):
                    val = self.parse_operand(tokens[1], lane)
                    self.wave.set_vgpr(lane, dst_reg, val)

        elif opcode in ("s_mov_b32", "s_mov_b64"):
            if len(tokens) < 2:
                raise ValueError(f"{opcode} requires dst and src")
            dst_token = tokens[0].lower()
            if dst_token == "exec":
                src_val = int(tokens[1], 16) if (tokens[1].startswith("0x") or tokens[1].startswith("-0x")) else int(tokens[1])
                self.wave.set_exec(src_val)
            elif dst_token == "exec_lo":
                src_val = self.parse_operand(tokens[1], 0)
                new_exec = (self.wave.exec_mask & (MASK32 << 32)) | (src_val & MASK32)
                self.wave.set_exec(new_exec)
            elif dst_token == "exec_hi":
                src_val = self.parse_operand(tokens[1], 0)
                new_exec = (self.wave.exec_mask & MASK32) | ((src_val & MASK32) << 32)
                self.wave.set_exec(new_exec)
            else:
                dst_type, dst_reg = self.parse_reg(tokens[0])
                if dst_type != "sgpr":
                    raise ValueError(f"s_mov destination must be SGPR, got {tokens[0]}")
                val = self.parse_operand(tokens[1], 0)
                self.wave.set_sgpr(dst_reg, val)

        elif opcode in ("v_dot2acc_f16_f16", "v_dot2_f16_f16"):
            is_acc = opcode == "v_dot2acc_f16_f16"
            min_operands = 4 if is_acc else 3
            reg_tokens: list[str] = []
            modifiers: list[str] = []
            for t in tokens:
                t_lower = t.lower()
                if (
                    t_lower == "clamp"
                    or t_lower.startswith("op_sel:")
                    or t_lower.startswith("op_sel_hi:")
                    or t_lower.startswith("neg_lo:")
                    or t_lower.startswith("neg_hi:")
                ):
                    modifiers.append(t_lower)
                else:
                    reg_tokens.append(t)

            if len(reg_tokens) < min_operands:
                raise ValueError(f"{opcode} requires at least {min_operands} operands, got {len(reg_tokens)}")

            dst_type, dst_reg = self.parse_reg(reg_tokens[0])
            if dst_type != "vgpr":
                raise ValueError(f"{opcode} destination must be VGPR, got {reg_tokens[0]}")

            clamp = "clamp" in modifiers
            op_sel = (0, 0, 0)
            op_sel_hi = (1, 1, 1)
            neg_lo = (0, 0, 0)
            neg_hi = (0, 0, 0)

            def parse_bracket_tuple(mod_str: str) -> tuple[int, ...]:
                inside = mod_str.split(":", 1)[1].strip("[]")
                return tuple(int(x.strip()) for x in inside.split(",") if x.strip())

            for m in modifiers:
                if m.startswith("op_sel:"):
                    vals = parse_bracket_tuple(m)
                    if len(vals) == 2:
                        op_sel = (vals[0], vals[1], 0)
                    elif len(vals) >= 3:
                        op_sel = (vals[0], vals[1], vals[2])
                elif m.startswith("op_sel_hi:"):
                    vals = parse_bracket_tuple(m)
                    if len(vals) == 2:
                        op_sel_hi = (vals[0], vals[1], 1)
                    elif len(vals) >= 3:
                        op_sel_hi = (vals[0], vals[1], vals[2])
                elif m.startswith("neg_lo:"):
                    vals = parse_bracket_tuple(m)
                    if len(vals) == 2:
                        neg_lo = (vals[0], vals[1], 0)
                    elif len(vals) >= 3:
                        neg_lo = (vals[0], vals[1], vals[2])
                elif m.startswith("neg_hi:"):
                    vals = parse_bracket_tuple(m)
                    if len(vals) == 2:
                        neg_hi = (vals[0], vals[1], 0)
                    elif len(vals) >= 3:
                        neg_hi = (vals[0], vals[1], vals[2])

            for lane in range(self.wave.wave_size):
                if not self.wave.is_lane_active(lane):
                    continue

                s0_val = self.parse_operand(reg_tokens[1], lane)
                s1_val = self.parse_operand(reg_tokens[2], lane)
                s2_val = self.parse_operand(reg_tokens[3], lane) if is_acc else 0

                res_f16 = ref_dot2acc_f16(
                    s0=s0_val,
                    s1=s1_val,
                    s2_acc=s2_val,
                    clamp=clamp,
                    op_sel=op_sel,
                    op_sel_hi=op_sel_hi,
                    neg_lo=neg_lo,
                    neg_hi=neg_hi,
                    flush_subnormals=self.flush_subnormals,
                )
                old_dst = self.wave.get_vgpr(lane, dst_reg)
                new_dst = (old_dst & 0xFFFF0000) | (res_f16 & MASK16)
                self.wave.set_vgpr(lane, dst_reg, new_dst)

        elif opcode in ("s_endpgm", "s_nop"):
            pass
        else:
            raise ValueError(f"Unsupported instruction opcode: {opcode}")

    def execute_program(self, lines: list[str]) -> None:
        for ln in lines:
            self.execute_line(ln)


class TestAccumulatorChainingAndClamping(unittest.TestCase):
    """Semantic tests for accumulator chaining, clamping saturation, and SIMD divergence."""

    def test_sequential_vector_dot_product_chaining(self):
        """Sequential accumulation of 4-element vectors across two v_dot2acc instructions."""
        # Vector A: [1.0, 2.0, 3.0, 4.0]
        # Vector B: [0.5, 1.5, 2.0, 0.5]
        # Expected dot product: 1*0.5 + 2*1.5 + 3*2.0 + 4*0.5 = 0.5 + 3.0 + 6.0 + 2.0 = 11.5
        v_a01 = pack_f16x2(H_ONE, H_TWO)
        v_b01 = pack_f16x2(H_HALF, float_to_f16_bits(1.5))
        v_a23 = pack_f16x2(H_THREE, H_FOUR)
        v_b23 = pack_f16x2(H_TWO, H_HALF)

        interp = Dot2AccInterpreter()
        program = [
            f"v_mov_b32 v0, {v_a01:#x}",
            f"v_mov_b32 v1, {v_b01:#x}",
            f"v_mov_b32 v2, {v_a23:#x}",
            f"v_mov_b32 v3, {v_b23:#x}",
            "v_mov_b32 v4, 0x0000",   # Accumulator initialized to +0.0
            "v_dot2acc_f16_f16 v4, v0, v1, v4",
            "v_dot2acc_f16_f16 v4, v2, v3, v4",
        ]
        interp.execute_program(program)

        # Check lane 0
        final_word = interp.wave.get_vgpr(0, 4)
        final_f16 = final_word & MASK16
        self.assertEqual(f16_bits_to_float(final_f16), 11.5)

    def test_8_element_vector_accumulation_chain(self):
        """Sequential accumulation of 8-element vectors partitioned into 4 dwords."""
        # Vector A: [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        # Vector B: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        # Sum of A = 9.0; Total = 9.0 * 2.0 = 18.0
        a_floats = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        b_floats = [2.0] * 8
        a_words = [pack_f16x2(float_to_f16_bits(a_floats[2 * i]), float_to_f16_bits(a_floats[2 * i + 1])) for i in range(4)]
        b_words = [pack_f16x2(float_to_f16_bits(b_floats[2 * i]), float_to_f16_bits(b_floats[2 * i + 1])) for i in range(4)]

        interp = Dot2AccInterpreter()
        program = [
            f"v_mov_b32 v0, {a_words[0]:#x}",
            f"v_mov_b32 v1, {b_words[0]:#x}",
            f"v_mov_b32 v2, {a_words[1]:#x}",
            f"v_mov_b32 v3, {b_words[1]:#x}",
            f"v_mov_b32 v4, {a_words[2]:#x}",
            f"v_mov_b32 v5, {b_words[2]:#x}",
            f"v_mov_b32 v6, {a_words[3]:#x}",
            f"v_mov_b32 v7, {b_words[3]:#x}",
            "v_mov_b32 v8, 0x0000",   # acc = 0
            "v_dot2acc_f16_f16 v8, v0, v1, v8",
            "v_dot2acc_f16_f16 v8, v2, v3, v8",
            "v_dot2acc_f16_f16 v8, v4, v5, v8",
            "v_dot2acc_f16_f16 v8, v6, v7, v8",
        ]
        interp.execute_program(program)

        final_f16 = interp.wave.get_vgpr(0, 8) & MASK16
        self.assertEqual(f16_bits_to_float(final_f16), 18.0)

    def test_16_element_vector_dot_product_chain(self):
        """Sequential accumulation of 16-element vectors across 8 dot2acc steps."""
        interp = Dot2AccInterpreter()
        # All elements 0.5, dot with 0.5 -> each pair gives 0.25 + 0.25 = 0.5
        # 8 pairs * 0.5 = 4.0
        pair = pack_f16x2(H_HALF, H_HALF)
        program = [
            f"v_mov_b32 v0, {pair:#x}",
            f"v_mov_b32 v1, {pair:#x}",
            "v_mov_b32 v2, 0x0000",
        ]
        for _ in range(8):
            program.append("v_dot2acc_f16_f16 v2, v0, v1, v2")
        interp.execute_program(program)

        final_f16 = interp.wave.get_vgpr(0, 2) & MASK16
        self.assertEqual(f16_bits_to_float(final_f16), 4.0)

    def test_iterative_accumulation_loop_simulation(self):
        """Simulation of an iterative FIR filter tap or recurrent state accumulation."""
        interp = Dot2AccInterpreter()
        # Initial state acc = 1.0
        # Each step: acc = 0.5 * 1.0 + 0.25 * 2.0 + acc = 0.5 + 0.5 + acc = 1.0 + acc
        s0 = pack_f16x2(H_HALF, float_to_f16_bits(0.25))
        s1 = pack_f16x2(H_ONE, H_TWO)
        program = [
            f"v_mov_b32 v0, {s0:#x}",
            f"v_mov_b32 v1, {s1:#x}",
            f"v_mov_b32 v2, {H_ONE:#x}",  # acc = 1.0
        ]
        for _ in range(5):
            program.append("v_dot2acc_f16_f16 v2, v0, v1, v2")
        interp.execute_program(program)

        # After 5 steps: 1.0 + 5 * 1.0 = 6.0
        final_f16 = interp.wave.get_vgpr(0, 2) & MASK16
        self.assertEqual(f16_bits_to_float(final_f16), 6.0)

    def test_clamp_saturation_negative_cases(self):
        """Verify clamp saturation modifier clamps negative results to +0.0."""
        interp = Dot2AccInterpreter()
        # Strongly negative: -10 * 2 + 0 + 0 = -20 -> clamped to +0.0
        s0 = pack_f16x2(float_to_f16_bits(-10.0), H_POS_ZERO)
        s1 = pack_f16x2(H_TWO, H_POS_ZERO)
        program = [
            f"v_mov_b32 v0, {s0:#x}",
            f"v_mov_b32 v1, {s1:#x}",
            "v_mov_b32 v2, 0x0000",
            "v_dot2acc_f16_f16 v2, v0, v1, v2 clamp",
        ]
        interp.execute_program(program)
        self.assertEqual(interp.wave.get_vgpr(0, 2) & MASK16, H_POS_ZERO)

        # Negative infinity clamps to +0.0
        s0_inf = pack_f16x2(H_NEG_INF, H_POS_ZERO)
        program_inf = [
            f"v_mov_b32 v0, {s0_inf:#x}",
            f"v_mov_b32 v1, {s1:#x}",
            "v_mov_b32 v2, 0x0000",
            "v_dot2acc_f16_f16 v2, v0, v1, v2 clamp",
        ]
        interp.execute_program(program_inf)
        self.assertEqual(interp.wave.get_vgpr(0, 2) & MASK16, H_POS_ZERO)

        # Negative zero clamps to +0.0
        s0_nzero = pack_f16x2(H_NEG_ZERO, H_NEG_ZERO)
        program_nzero = [
            f"v_mov_b32 v0, {s0_nzero:#x}",
            f"v_mov_b32 v1, {pack_f16x2(H_ONE, H_ONE):#x}",
            f"v_mov_b32 v2, {H_NEG_ZERO:#x}",
            "v_dot2acc_f16_f16 v2, v0, v1, v2 clamp",
        ]
        interp.execute_program(program_nzero)
        self.assertEqual(interp.wave.get_vgpr(0, 2) & MASK16, H_POS_ZERO)

    def test_clamp_saturation_above_one_cases(self):
        """Verify clamp saturation modifier clamps results > 1.0 to 1.0 (0x3C00)."""
        interp = Dot2AccInterpreter()
        # 1.5 * 1.0 + 0 + 0 = 1.5 -> clamped to 1.0
        s0 = pack_f16x2(float_to_f16_bits(1.5), H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        program = [
            f"v_mov_b32 v0, {s0:#x}",
            f"v_mov_b32 v1, {s1:#x}",
            "v_mov_b32 v2, 0x0000",
            "v_dot2acc_f16_f16 v2, v0, v1, v2 clamp",
        ]
        interp.execute_program(program)
        self.assertEqual(interp.wave.get_vgpr(0, 2) & MASK16, H_ONE)

        # Positive infinity clamps to 1.0
        s0_pinf = pack_f16x2(H_POS_INF, H_POS_ZERO)
        program_pinf = [
            f"v_mov_b32 v0, {s0_pinf:#x}",
            f"v_mov_b32 v1, {s1:#x}",
            "v_mov_b32 v2, 0x0000",
            "v_dot2acc_f16_f16 v2, v0, v1, v2 clamp",
        ]
        interp.execute_program(program_pinf)
        self.assertEqual(interp.wave.get_vgpr(0, 2) & MASK16, H_ONE)

    def test_clamp_saturation_in_range_invariance(self):
        """Values in [0.0, 1.0] are preserved with clamp modifier."""
        interp = Dot2AccInterpreter()
        test_values = [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0]
        for val in test_values:
            val_f16 = float_to_f16_bits(val)
            s0 = pack_f16x2(val_f16, H_POS_ZERO)
            s1 = pack_f16x2(H_ONE, H_POS_ZERO)
            program = [
                f"v_mov_b32 v0, {s0:#x}",
                f"v_mov_b32 v1, {s1:#x}",
                "v_mov_b32 v2, 0x0000",
                "v_dot2acc_f16_f16 v2, v0, v1, v2 clamp",
            ]
            interp.execute_program(program)
            self.assertEqual(interp.wave.get_vgpr(0, 2) & MASK16, val_f16)

    def test_intermediate_clamping_vs_final_clamping(self):
        """Intermediate clamping restricts accumulator to [0.0, 1.0] at each step."""
        # Step 1: 0.8 + 0.5 = 1.3
        # Step 2: -0.5
        # If clamped at Step 1: acc becomes 1.0, then Step 2 yields 1.0 - 0.5 = 0.5.
        # If unclamped at Step 1: acc is 1.3, then Step 2 yields 1.3 - 0.5 = 0.8.
        s0 = pack_f16x2(float_to_f16_bits(0.8), H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        acc1 = float_to_f16_bits(0.5)

        s2 = pack_f16x2(float_to_f16_bits(-0.5), H_POS_ZERO)
        s3 = pack_f16x2(H_ONE, H_POS_ZERO)

        interp_clamped = Dot2AccInterpreter()
        prog_clamped = [
            f"v_mov_b32 v0, {s0:#x}",
            f"v_mov_b32 v1, {s1:#x}",
            f"v_mov_b32 v2, {acc1:#x}",
            "v_dot2acc_f16_f16 v2, v0, v1, v2 clamp",  # clamps 1.3 to 1.0
            f"v_mov_b32 v3, {s2:#x}",
            f"v_mov_b32 v4, {s3:#x}",
            "v_dot2acc_f16_f16 v2, v3, v4, v2",        # 1.0 - 0.5 = 0.5
        ]
        interp_clamped.execute_program(prog_clamped)
        self.assertAlmostEqual(f16_bits_to_float(interp_clamped.wave.get_vgpr(0, 2) & MASK16), 0.5)

        interp_unclamped = Dot2AccInterpreter()
        prog_unclamped = [
            f"v_mov_b32 v0, {s0:#x}",
            f"v_mov_b32 v1, {s1:#x}",
            f"v_mov_b32 v2, {acc1:#x}",
            "v_dot2acc_f16_f16 v2, v0, v1, v2",        # stays 1.3
            f"v_mov_b32 v3, {s2:#x}",
            f"v_mov_b32 v4, {s3:#x}",
            "v_dot2acc_f16_f16 v2, v3, v4, v2",        # 1.3 - 0.5 = 0.8
        ]
        interp_unclamped.execute_program(prog_unclamped)
        self.assertAlmostEqual(f16_bits_to_float(interp_unclamped.wave.get_vgpr(0, 2) & MASK16), 0.8, delta=1e-3)

    def test_multi_lane_simd_exec_mask_divergence(self):
        """Active lanes update destination VGPR; inactive lanes preserve destination register."""
        interp = Dot2AccInterpreter(wave_size=64)
        # Seed all lanes with distinct values
        for lane in range(64):
            interp.wave.set_vgpr(lane, 0, pack_f16x2(H_ONE, H_ONE))
            interp.wave.set_vgpr(lane, 1, pack_f16x2(H_TWO, H_THREE))
            interp.wave.set_vgpr(lane, 2, 0x1234)
            interp.wave.set_vgpr(lane, 3, 0xDEADBEEF)  # destination register seed

        # Set exec mask to alternate lanes: 0x5555555555555555 (even lanes active, odd inactive)
        exec_mask = 0x5555555555555555
        interp.wave.set_exec(exec_mask)

        # Run dot2acc into v3: 1*2 + 1*3 + 0x1234(f16)
        interp.execute_line("v_dot2acc_f16_f16 v3, v0, v1, v2")

        expected_f16 = ref_dot2acc_f16(pack_f16x2(H_ONE, H_ONE), pack_f16x2(H_TWO, H_THREE), 0x1234)

        for lane in range(64):
            val = interp.wave.get_vgpr(lane, 3)
            if (exec_mask >> lane) & 1:
                # Active lane: lower half updated, upper half preserved from 0xDEADBEEF
                self.assertEqual(val & MASK16, expected_f16)
                self.assertEqual((val >> 16) & MASK16, 0xDEAD)
            else:
                # Inactive lane: preserved completely
                self.assertEqual(val, 0xDEADBEEF)

    def test_simd_divergent_control_flow_branches(self):
        """Simulate divergent branches across Wave64 lanes executing independent dot2acc."""
        interp = Dot2AccInterpreter(wave_size=64)
        for lane in range(64):
            interp.wave.set_vgpr(lane, 5, 0x00000000)

        # Branch A: Low 32 lanes active (exec = 0x00000000FFFFFFFF)
        interp.wave.set_exec(0x00000000FFFFFFFF)
        for lane in range(32):
            interp.wave.set_vgpr(lane, 0, pack_f16x2(H_ONE, H_ONE))
            interp.wave.set_vgpr(lane, 1, pack_f16x2(H_ONE, H_ONE))
            interp.wave.set_vgpr(lane, 2, H_ONE)
        interp.execute_line("v_dot2acc_f16_f16 v5, v0, v1, v2")  # 1*1 + 1*1 + 1 = 3.0

        # Branch B: High 32 lanes active (exec = 0xFFFFFFFF00000000)
        interp.wave.set_exec(0xFFFFFFFF00000000)
        for lane in range(32, 64):
            interp.wave.set_vgpr(lane, 0, pack_f16x2(H_TWO, H_TWO))
            interp.wave.set_vgpr(lane, 1, pack_f16x2(H_TWO, H_TWO))
            interp.wave.set_vgpr(lane, 2, H_TWO)
        interp.execute_line("v_dot2acc_f16_f16 v5, v0, v1, v2")  # 2*2 + 2*2 + 2 = 10.0

        # Verify low 32 lanes got 3.0 and high 32 lanes got 10.0
        for lane in range(32):
            self.assertEqual(f16_bits_to_float(interp.wave.get_vgpr(lane, 5) & MASK16), 3.0)
        for lane in range(32, 64):
            self.assertEqual(f16_bits_to_float(interp.wave.get_vgpr(lane, 5) & MASK16), 10.0)

    def test_wave32_and_wave64_execution(self):
        """Verify interpreter works for both Wave32 and Wave64 sizes."""
        for wsize in (32, 64):
            interp = Dot2AccInterpreter(wave_size=wsize)
            self.assertEqual(interp.wave.wave_size, wsize)
            self.assertEqual(interp.wave.exec_mask, (1 << wsize) - 1)
            interp.wave.set_vgpr(0, 0, pack_f16x2(H_ONE, H_ONE))
            interp.wave.set_vgpr(0, 1, pack_f16x2(H_HALF, H_HALF))
            interp.wave.set_vgpr(0, 2, H_HALF)
            interp.execute_line("v_dot2acc_f16_f16 v3, v0, v1, v2")
            self.assertEqual(f16_bits_to_float(interp.wave.get_vgpr(0, 3) & MASK16), 1.5)


class TestNanPropagationSubnormalsAndNegativeControls(unittest.TestCase):
    """Validation of NaN propagation, subnormal gradual underflow and FTZ, signed zeros, and negative controls."""

    def test_qnan_propagation_from_each_operand_position(self):
        """Quiet NaN in any source component propagates to NaN result."""
        # NaN in S0.x
        s0_x = pack_f16x2(H_QNAN, H_ONE)
        s1 = pack_f16x2(H_ONE, H_ONE)
        res = ref_dot2acc_f16(s0_x, s1, H_ONE)
        self.assertTrue(is_f16_nan(res))
        self.assertTrue(is_f16_qnan(res))

        # NaN in S0.y
        s0_y = pack_f16x2(H_ONE, H_QNAN)
        res = ref_dot2acc_f16(s0_y, s1, H_ONE)
        self.assertTrue(is_f16_nan(res))
        self.assertTrue(is_f16_qnan(res))

        # NaN in S1.x
        s1_x = pack_f16x2(H_QNAN, H_ONE)
        res = ref_dot2acc_f16(pack_f16x2(H_ONE, H_ONE), s1_x, H_ONE)
        self.assertTrue(is_f16_nan(res))
        self.assertTrue(is_f16_qnan(res))

        # NaN in S1.y
        s1_y = pack_f16x2(H_ONE, H_QNAN)
        res = ref_dot2acc_f16(pack_f16x2(H_ONE, H_ONE), s1_y, H_ONE)
        self.assertTrue(is_f16_nan(res))
        self.assertTrue(is_f16_qnan(res))

        # NaN in S2 accumulator
        res = ref_dot2acc_f16(pack_f16x2(H_ONE, H_ONE), pack_f16x2(H_ONE, H_ONE), H_QNAN)
        self.assertTrue(is_f16_nan(res))
        self.assertTrue(is_f16_qnan(res))

    def test_snan_quieting_and_propagation(self):
        """Signaling NaN in any operand produces a quieted NaN."""
        for snan in (H_SNAN, 0x7C01, 0x7CFF):
            s0 = pack_f16x2(snan, H_ONE)
            s1 = pack_f16x2(H_ONE, H_ONE)
            res = ref_dot2acc_f16(s0, s1, H_POS_ZERO)
            self.assertTrue(is_f16_nan(res))
            self.assertTrue(is_f16_qnan(res))

            # In accumulator S2
            res_acc = ref_dot2acc_f16(s1, s1, snan)
            self.assertTrue(is_f16_nan(res_acc))
            self.assertTrue(is_f16_qnan(res_acc))

    def test_indeterminate_forms(self):
        """Indeterminate arithmetic produces NaN."""
        # 0.0 * inf = NaN
        s0 = pack_f16x2(H_POS_ZERO, H_ONE)
        s1 = pack_f16x2(H_POS_INF, H_ONE)
        res = ref_dot2acc_f16(s0, s1, H_POS_ZERO)
        self.assertTrue(is_f16_nan(res))

        # (-0.0) * inf = NaN
        s0_n = pack_f16x2(H_NEG_ZERO, H_ONE)
        res_n = ref_dot2acc_f16(s0_n, s1, H_POS_ZERO)
        self.assertTrue(is_f16_nan(res_n))

        # (+inf) + (-inf) = NaN
        s0_inf = pack_f16x2(H_POS_INF, H_NEG_INF)
        s1_ones = pack_f16x2(H_ONE, H_ONE)
        res_inf = ref_dot2acc_f16(s0_inf, s1_ones, H_POS_ZERO)
        self.assertTrue(is_f16_nan(res_inf))

    def test_nan_with_clamp_modifier(self):
        """Under graphics saturation clamp, NaN clamps to +0.0."""
        s0 = pack_f16x2(H_QNAN, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        res = ref_dot2acc_f16(s0, s1, H_POS_ZERO, clamp=True)
        self.assertEqual(res, H_POS_ZERO)

        # NaN in accumulator with clamp
        res_acc = ref_dot2acc_f16(pack_f16x2(H_ONE, H_ONE), pack_f16x2(H_ONE, H_ONE), H_QNAN, clamp=True)
        self.assertEqual(res_acc, H_POS_ZERO)

    def test_subnormal_gradual_underflow_precision(self):
        """In gradual underflow mode, subnormal precision is maintained."""
        # 2^-24 + 2^-24 + 2^-24 = 3 * 2^-24
        s0 = pack_f16x2(H_SUB_MIN, H_SUB_MIN)
        s1 = pack_f16x2(H_ONE, H_ONE)
        res = ref_dot2acc_f16(s0, s1, H_SUB_MIN, flush_subnormals=False)
        self.assertEqual(res, 0x0003)

        # Accumulation loop of 10 subnormal steps
        interp = Dot2AccInterpreter(flush_subnormals=False)
        interp.wave.set_vgpr(0, 0, pack_f16x2(H_SUB_MIN, H_POS_ZERO))
        interp.wave.set_vgpr(0, 1, pack_f16x2(H_ONE, H_POS_ZERO))
        interp.wave.set_vgpr(0, 2, 0x0000)
        for _ in range(10):
            interp.execute_line("v_dot2acc_f16_f16 v2, v0, v1, v2")
        self.assertEqual(interp.wave.get_vgpr(0, 2) & MASK16, 10)

    def test_subnormal_boundary_transition_to_normal(self):
        """Subnormal accumulation crossing into the normal float range."""
        # Largest subnormal (1023 * 2^-24) + Smallest subnormal (1 * 2^-24) = 1024 * 2^-24 = 2^-14 (0x0400, smallest normal)
        s0 = pack_f16x2(H_SUB_MAX, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        s2 = H_SUB_MIN
        res = ref_dot2acc_f16(s0, s1, s2, flush_subnormals=False)
        self.assertEqual(res, H_NORM_MIN)
        self.assertFalse(is_f16_subnormal(res))

    def test_flush_to_zero_subnormal_inputs(self):
        """FTZ mode flushes subnormal inputs to signed zero before arithmetic."""
        s0 = pack_f16x2(H_SUB_MIN, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        # In FTZ, H_SUB_MIN becomes +0.0 -> result is +0.0
        res = ref_dot2acc_f16(s0, s1, H_POS_ZERO, flush_subnormals=True)
        self.assertEqual(res, H_POS_ZERO)

        # Negative subnormal input flushes to -0.0
        s0_neg = pack_f16x2(H_NEG_SUB_MIN, H_NEG_ZERO)
        s1_ones = pack_f16x2(H_ONE, H_ONE)
        res_neg = ref_dot2acc_f16(s0_neg, s1_ones, H_NEG_ZERO, flush_subnormals=True)
        self.assertEqual(res_neg, H_NEG_ZERO)

        # Subnormal accumulator flushes to zero
        s2_sub = H_SUB_MAX
        res_acc = ref_dot2acc_f16(pack_f16x2(H_POS_ZERO, H_POS_ZERO), pack_f16x2(H_ONE, H_ONE), s2_sub, flush_subnormals=True)
        self.assertEqual(res_acc, H_POS_ZERO)

    def test_flush_to_zero_subnormal_results(self):
        """FTZ mode flushes subnormal products/sums to zero."""
        # Product that results in subnormal: 2^-14 * 0.5 = 2^-15 (0x0200)
        s0 = pack_f16x2(H_NORM_MIN, H_POS_ZERO)
        s1 = pack_f16x2(H_HALF, H_POS_ZERO)
        res_ftz = ref_dot2acc_f16(s0, s1, H_POS_ZERO, flush_subnormals=True)
        self.assertEqual(res_ftz, H_POS_ZERO)

        res_no_ftz = ref_dot2acc_f16(s0, s1, H_POS_ZERO, flush_subnormals=False)
        self.assertEqual(res_no_ftz, 0x0200)
        self.assertTrue(is_f16_subnormal(res_no_ftz))

    def test_comprehensive_signed_zero_matrix(self):
        """Test exhaustive combinations of signed zeros in dot2acc."""
        for z0 in (H_POS_ZERO, H_NEG_ZERO):
            for z1 in (H_POS_ZERO, H_NEG_ZERO):
                for z2 in (H_POS_ZERO, H_NEG_ZERO):
                    s0 = pack_f16x2(z0, z1)
                    s1 = pack_f16x2(H_ONE, H_ONE)
                    res = ref_dot2acc_f16(s0, s1, z2)
                    self.assertTrue(is_f16_zero(res))
                    # Result is negative zero if and only if all summed terms are negative zero
                    all_neg = (z0 == H_NEG_ZERO) and (z1 == H_NEG_ZERO) and (z2 == H_NEG_ZERO)
                    expected = H_NEG_ZERO if all_neg else H_POS_ZERO
                    self.assertEqual(res, expected, f"Failed for z0={hex(z0)}, z1={hex(z1)}, z2={hex(z2)}")

    def test_negative_out_of_bounds_registers(self):
        """Accessing out of bounds VGPR or SGPR indices raises IndexError."""
        interp = Dot2AccInterpreter()
        with self.assertRaises(IndexError):
            interp.wave.get_vgpr(0, 256)
        with self.assertRaises(IndexError):
            interp.wave.set_vgpr(0, -1, 0)
        with self.assertRaises(IndexError):
            interp.wave.get_sgpr(106)
        with self.assertRaises(IndexError):
            interp.wave.set_sgpr(200, 0)
        with self.assertRaises(IndexError):
            interp.parse_reg("v256")
        with self.assertRaises(IndexError):
            interp.parse_reg("s106")

    def test_negative_invalid_destinations_and_opcodes(self):
        """Invalid destination types and opcodes raise appropriate exceptions."""
        interp = Dot2AccInterpreter()
        # VALU destination cannot be SGPR
        with self.assertRaises(ValueError):
            interp.execute_line("v_dot2acc_f16_f16 s0, v1, v2, v3")
        with self.assertRaises(ValueError):
            interp.execute_line("v_mov_b32 s0, v1")
        with self.assertRaises(ValueError):
            interp.execute_line("s_mov_b32 v0, s1")
        # Unsupported opcode
        with self.assertRaises(ValueError):
            interp.execute_line("v_unknown_opcode v0, v1, v2")
        # Insufficient operands
        with self.assertRaises(ValueError):
            interp.execute_line("v_dot2acc_f16_f16 v0, v1, v2")

    def test_negative_malformed_tokens(self):
        """Malformed or empty operand tokens raise ValueError."""
        interp = Dot2AccInterpreter()
        with self.assertRaises(ValueError):
            interp.parse_operand("", 0)
        with self.assertRaises(ValueError):
            interp.parse_operand("invalid_token_xyz", 0)
        with self.assertRaises(ValueError):
            interp.parse_reg("not_a_register")

    def test_negative_invalid_wave_size(self):
        """Unsupported wavefront sizes raise ValueError."""
        with self.assertRaises(ValueError):
            SimdWavefront(wave_size=16)
        with self.assertRaises(ValueError):
            SimdWavefront(wave_size=128)




if __name__ == "__main__":
    unittest.main()
