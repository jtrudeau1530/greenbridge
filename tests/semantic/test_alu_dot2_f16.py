"""Semantic reference oracle and interpreter tests for RDNA2 packed half-precision dot products:
  - v_dot2_f16_f16:    D.f16 = S0.f16[0] * S1.f16[0] + S0.f16[1] * S1.f16[1]
  - v_dot2acc_f16_f16: D.f16 = S0.f16[0] * S1.f16[0] + S0.f16[1] * S1.f16[1] + S2.f16

AMD RDNA2 VALU Instruction Set Architecture (GFX10):
  The v_dot2_f16_f16 instruction computes the dot product of two packed pairs of IEEE-754
  binary16 (half-precision) floating-point numbers from 32-bit vector registers S0 and S1.
  S0 contains two 16-bit half-precision floats: S0.x (bits [15:0]) and S0.y (bits [31:16]).
  S1 contains two 16-bit half-precision floats: S1.x (bits [15:0]) and S1.y (bits [31:16]).
  The accumulating variant v_dot2acc_f16_f16 (or v_dot2c_f16_f16) adds an accumulator S2
  into the scalar 16-bit result.

Modifiers:
  - clamp: Saturates the floating-point result to [0.0, 1.0]. Negative values and -0.0 clamp to
    +0.0, values > 1.0 clamp to 1.0.
  - op_sel / op_sel_hi: Selects which 16-bit half-word of each source is read for the low/high
    product terms.
  - neg_lo / neg_hi: Negates the chosen half-words prior to multiplication.

Standard library only: Python 3.9+ compatible, zero game data, zero proprietary assets.
"""
from __future__ import annotations

import math
import re
import struct
import unittest

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
    # Overflow boundary in IEEE-754 binary16: 65520.0 rounds to infinity
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


def clamp_f16(val: float) -> float:
    """Apply RDNA VALU float clamp modifier saturating to [0.0, 1.0]."""
    if math.isnan(val):
        return 0.0
    if val < 0.0 or math.isinf(val) and val < 0:
        return 0.0
    if val > 1.0 or math.isinf(val) and val > 0:
        return 1.0
    return val


def ref_dot2_f16(
    s0: int,
    s1: int,
    clamp: bool = False,
    op_sel: tuple[int, int] = (0, 0),
    op_sel_hi: tuple[int, int] = (1, 1),
    neg_lo: tuple[int, int] = (0, 0),
    neg_hi: tuple[int, int] = (0, 0),
) -> int:
    """Pure-Python reference oracle for RDNA2 v_dot2_f16_f16.

    Formula: D = S0.x * S1.x + S0.y * S1.y
    """
    s0_halves = unpack_f16x2(s0)
    s1_halves = unpack_f16x2(s1)

    # First product term (low half-word selection by default)
    h0_x = s0_halves[op_sel[0] & 1]
    h1_x = s1_halves[op_sel[1] & 1]
    f0_x = f16_bits_to_float(h0_x)
    f1_x = f16_bits_to_float(h1_x)
    if neg_lo[0]:
        f0_x = -f0_x
    if neg_lo[1]:
        f1_x = -f1_x

    # Second product term (high half-word selection by default)
    h0_y = s0_halves[op_sel_hi[0] & 1]
    h1_y = s1_halves[op_sel_hi[1] & 1]
    f0_y = f16_bits_to_float(h0_y)
    f1_y = f16_bits_to_float(h1_y)
    if neg_hi[0]:
        f0_y = -f0_y
    if neg_hi[1]:
        f1_y = -f1_y

    p0 = f0_x * f1_x
    p1 = f0_y * f1_y
    dot = p0 + p1

    if clamp:
        dot = clamp_f16(dot)

    return float_to_f16_bits(dot)


def ref_dot2acc_f16(
    s0: int,
    s1: int,
    s2_acc: int = 0,
    clamp: bool = False,
    op_sel: tuple[int, int, int] = (0, 0, 0),
    op_sel_hi: tuple[int, int, int] = (1, 1, 1),
    neg_lo: tuple[int, int, int] = (0, 0, 0),
    neg_hi: tuple[int, int, int] = (0, 0, 0),
) -> int:
    """Pure-Python reference oracle for RDNA2 v_dot2acc_f16_f16.

    Formula: D = S0.x * S1.x + S0.y * S1.y + S2
    """
    s0_halves = unpack_f16x2(s0)
    s1_halves = unpack_f16x2(s1)

    h0_x = s0_halves[op_sel[0] & 1]
    h1_x = s1_halves[op_sel[1] & 1]
    f0_x = f16_bits_to_float(h0_x)
    f1_x = f16_bits_to_float(h1_x)
    if neg_lo[0]:
        f0_x = -f0_x
    if neg_lo[1]:
        f1_x = -f1_x

    h0_y = s0_halves[op_sel_hi[0] & 1]
    h1_y = s1_halves[op_sel_hi[1] & 1]
    f0_y = f16_bits_to_float(h0_y)
    f1_y = f16_bits_to_float(h1_y)
    if neg_hi[0]:
        f0_y = -f0_y
    if neg_hi[1]:
        f1_y = -f1_y

    p0 = f0_x * f1_x
    p1 = f0_y * f1_y

    # S2 accumulator
    s2_bits = s2_acc & MASK16
    f2 = f16_bits_to_float(s2_bits)
    if len(neg_lo) > 2 and neg_lo[2]:
        f2 = -f2

    dot = p0 + p1 + f2

    if clamp:
        dot = clamp_f16(dot)

    return float_to_f16_bits(dot)


class TestDot2F16ReferenceOracle(unittest.TestCase):
    """Exhaustive validation of reference mathematical semantics for v_dot2_f16_f16."""

    def test_float_conversion_roundtrip(self):
        """Sanity check f16 <-> float conversions for representative constants."""
        cases = [
            (H_POS_ZERO, 0.0),
            (H_NEG_ZERO, -0.0),
            (H_ONE, 1.0),
            (H_NEG_ONE, -1.0),
            (H_HALF, 0.5),
            (H_TWO, 2.0),
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

    def test_basic_normal_dot_products(self):
        """Standard normal vectors dot product calculation."""
        # [1.0, 2.0] . [3.0, 4.0] = 1*3 + 2*4 = 11.0
        s0 = pack_f16x2(H_ONE, H_TWO)
        s1 = pack_f16x2(H_THREE, H_FOUR)
        res = ref_dot2_f16(s0, s1)
        self.assertEqual(f16_bits_to_float(res), 11.0)

        # [0.5, 0.5] . [0.5, 0.5] = 0.25 + 0.25 = 0.5
        s0 = pack_f16x2(H_HALF, H_HALF)
        s1 = pack_f16x2(H_HALF, H_HALF)
        res = ref_dot2_f16(s0, s1)
        self.assertEqual(res, H_HALF)

    def test_orthogonal_vectors(self):
        """Dot product of orthogonal vectors yields zero."""
        # [1.0, 0.0] . [0.0, 1.0] = 0.0
        s0 = pack_f16x2(H_ONE, H_POS_ZERO)
        s1 = pack_f16x2(H_POS_ZERO, H_ONE)
        res = ref_dot2_f16(s0, s1)
        self.assertEqual(res, H_POS_ZERO)

        # [2.0, -1.0] . [1.0, 2.0] = 2.0 - 2.0 = 0.0
        s0 = pack_f16x2(H_TWO, H_NEG_ONE)
        s1 = pack_f16x2(H_ONE, H_TWO)
        res = ref_dot2_f16(s0, s1)
        self.assertEqual(res, H_POS_ZERO)

    def test_signed_zeros(self):
        """Signed zero multiplication and addition semantics."""
        # [+0.0, +0.0] . [+0.0, +0.0] = +0.0
        s0 = pack_f16x2(H_POS_ZERO, H_POS_ZERO)
        s1 = pack_f16x2(H_POS_ZERO, H_POS_ZERO)
        self.assertEqual(ref_dot2_f16(s0, s1), H_POS_ZERO)

        # [-0.0, -0.0] . [+1.0, +1.0] = -0.0 + -0.0 = -0.0
        s0 = pack_f16x2(H_NEG_ZERO, H_NEG_ZERO)
        s1 = pack_f16x2(H_ONE, H_ONE)
        self.assertEqual(ref_dot2_f16(s0, s1), H_NEG_ZERO)

        # [+0.0, -0.0] . [+1.0, +1.0] = +0.0 + -0.0 = +0.0
        s0 = pack_f16x2(H_POS_ZERO, H_NEG_ZERO)
        s1 = pack_f16x2(H_ONE, H_ONE)
        self.assertEqual(ref_dot2_f16(s0, s1), H_POS_ZERO)

    def test_subnormal_operands(self):
        """Gradual underflow and subnormal operand handling."""
        # Subnormal min * 1.0 + 0 = Subnormal min
        s0 = pack_f16x2(H_SUB_MIN, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        self.assertEqual(ref_dot2_f16(s0, s1), H_SUB_MIN)

        # Two subnormals summed
        s0 = pack_f16x2(H_SUB_MIN, H_SUB_MIN)
        s1 = pack_f16x2(H_ONE, H_ONE)
        # 2^-24 + 2^-24 = 2 * 2^-24 = 2^-23 (0x0002)
        self.assertEqual(ref_dot2_f16(s0, s1), 0x0002)

    def test_infinities(self):
        """Infinity arithmetic and indeterminate form NaN generation."""
        # [+inf, 0] . [1.0, 0] = +inf
        s0 = pack_f16x2(H_POS_INF, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        self.assertEqual(ref_dot2_f16(s0, s1), H_POS_INF)

        # [-inf, 0] . [1.0, 0] = -inf
        s0 = pack_f16x2(H_NEG_INF, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        self.assertEqual(ref_dot2_f16(s0, s1), H_NEG_INF)

        # [+inf, -inf] . [1.0, 1.0] = +inf + -inf = NaN
        s0 = pack_f16x2(H_POS_INF, H_NEG_INF)
        s1 = pack_f16x2(H_ONE, H_ONE)
        self.assertTrue(math.isnan(f16_bits_to_float(ref_dot2_f16(s0, s1))))

        # [0.0, 0.0] . [+inf, 0.0] = 0 * inf = NaN
        s0 = pack_f16x2(H_POS_ZERO, H_POS_ZERO)
        s1 = pack_f16x2(H_POS_INF, H_POS_ZERO)
        self.assertTrue(math.isnan(f16_bits_to_float(ref_dot2_f16(s0, s1))))

    def test_nan_propagation(self):
        """Quiet and signaling NaNs propagate to NaN result."""
        for nan_enc in (H_QNAN, H_SNAN):
            s0 = pack_f16x2(nan_enc, H_ONE)
            s1 = pack_f16x2(H_ONE, H_ONE)
            res = ref_dot2_f16(s0, s1)
            self.assertTrue(math.isnan(f16_bits_to_float(res)))

            s0 = pack_f16x2(H_ONE, nan_enc)
            s1 = pack_f16x2(H_ONE, H_ONE)
            res = ref_dot2_f16(s0, s1)
            self.assertTrue(math.isnan(f16_bits_to_float(res)))

    def test_overflow_to_infinity(self):
        """Values exceeding maximum finite f16 overflow to infinity."""
        # 65504 * 2 + 0 = 131008 -> +inf
        s0 = pack_f16x2(H_NORM_MAX, H_POS_ZERO)
        s1 = pack_f16x2(H_TWO, H_POS_ZERO)
        self.assertEqual(ref_dot2_f16(s0, s1), H_POS_INF)

        # -65504 * 2 + 0 = -131008 -> -inf
        s0 = pack_f16x2(H_NEG_NORM_MAX, H_POS_ZERO)
        s1 = pack_f16x2(H_TWO, H_POS_ZERO)
        self.assertEqual(ref_dot2_f16(s0, s1), H_NEG_INF)

    def test_clamp_modifier(self):
        """Clamp saturates results to [0.0, 1.0]."""
        # Negative result clamps to +0.0
        s0 = pack_f16x2(H_NEG_ONE, H_POS_ZERO)
        s1 = pack_f16x2(H_TWO, H_POS_ZERO)
        self.assertEqual(ref_dot2_f16(s0, s1, clamp=True), H_POS_ZERO)

        # Result > 1.0 clamps to 1.0
        s0 = pack_f16x2(H_TWO, H_TWO)
        s1 = pack_f16x2(H_TWO, H_TWO)
        self.assertEqual(ref_dot2_f16(s0, s1, clamp=True), H_ONE)

        # Result in [0.0, 1.0] stays intact
        s0 = pack_f16x2(H_HALF, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        self.assertEqual(ref_dot2_f16(s0, s1, clamp=True), H_HALF)

        # +inf clamps to 1.0, -inf clamps to 0.0
        s0 = pack_f16x2(H_POS_INF, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        self.assertEqual(ref_dot2_f16(s0, s1, clamp=True), H_ONE)

        s0 = pack_f16x2(H_NEG_INF, H_POS_ZERO)
        s1 = pack_f16x2(H_ONE, H_POS_ZERO)
        self.assertEqual(ref_dot2_f16(s0, s1, clamp=True), H_POS_ZERO)

    def test_source_modifiers(self):
        """op_sel and negation modifiers correctly permute and invert inputs."""
        # Swap x and y halves using op_sel
        s0 = pack_f16x2(H_ONE, H_TWO)
        s1 = pack_f16x2(H_THREE, H_FOUR)
        # Default: 1*3 + 2*4 = 11
        self.assertEqual(f16_bits_to_float(ref_dot2_f16(s0, s1)), 11.0)
        # Swapped S1 halves: op_sel=(0, 1), op_sel_hi=(1, 0) -> 1*4 + 2*3 = 10
        res = ref_dot2_f16(s0, s1, op_sel=(0, 1), op_sel_hi=(1, 0))
        self.assertEqual(f16_bits_to_float(res), 10.0)

        # Negate first product term
        res_neg = ref_dot2_f16(s0, s1, neg_lo=(1, 0))
        # -1*3 + 2*4 = 5.0
        self.assertEqual(f16_bits_to_float(res_neg), 5.0)

    def test_accumulating_variant(self):
        """v_dot2acc_f16_f16 adds the third source accumulator S2."""
        # 1*3 + 2*4 + 10 = 21.0
        s0 = pack_f16x2(H_ONE, H_TWO)
        s1 = pack_f16x2(H_THREE, H_FOUR)
        s2 = float_to_f16_bits(10.0)
        res = ref_dot2acc_f16(s0, s1, s2)
        self.assertEqual(f16_bits_to_float(res), 21.0)


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
        self.vgpr[lane][reg & 0xFF] = val & MASK32

    def get_vgpr(self, lane: int, reg: int) -> int:
        return self.vgpr[lane][reg & 0xFF]

    def set_sgpr(self, reg: int, val: int) -> None:
        if not (0 <= reg < 106):
            raise IndexError(f"SGPR index {reg} out of range")
        self.sgpr[reg] = val & MASK32

    def get_sgpr(self, reg: int) -> int:
        if not (0 <= reg < 106):
            raise IndexError(f"SGPR index {reg} out of range")
        return self.sgpr[reg]


class Dot2Interpreter:
    """Instruction interpreter for RDNA2 packed half-precision dot products.

    Handles v_mov_b32, s_mov_b32, s_mov_b64 exec, v_dot2_f16_f16, and v_dot2acc_f16_f16.
    Respects SIMD lane execution masks, register aliasing, and source/destination modifiers.
    """

    def __init__(self, wave_size: int = 64) -> None:
        self.wave = SimdWavefront(wave_size=wave_size)

    def parse_reg(self, token: str) -> tuple[str, int]:
        token = token.strip().lower()
        if token.startswith("v") and token[1:].isdigit():
            return "vgpr", int(token[1:])
        if token.startswith("s") and token[1:].isdigit():
            return "sgpr", int(token[1:])
        raise ValueError(f"Unrecognized register name: {token}")

    def parse_imm(self, token: str) -> int:
        token = token.strip()
        if token.startswith("0x") or token.startswith("-0x"):
            return int(token, 16) & MASK32
        try:
            return int(token) & MASK32
        except ValueError:
            f_val = float(token)
            return float_to_f16_bits(f_val)

    def parse_imm64(self, token: str) -> int:
        token = token.strip()
        if token.startswith("0x") or token.startswith("-0x"):
            return int(token, 16) & MASK64
        return int(token) & MASK64

    def read_operand(self, token: str, lane: int) -> int:
        token = token.strip()
        if token.startswith("v") and token[1:].isdigit():
            return self.wave.get_vgpr(lane, int(token[1:]))
        if token.startswith("s") and token[1:].isdigit():
            return self.wave.get_sgpr(int(token[1:]))
        return self.parse_imm(token)

    def execute_line(self, line: str) -> None:
        line = line.split(";")[0].strip()
        if not line:
            return
        parts = line.split(None, 1)
        mnemonic = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        modifiers = re.findall(
            r"(?:op_sel(?:_hi)?:\[[0-9, ]+\]|neg_(?:lo|hi):\[[0-9, ]+\]|clamp)",
            rest,
            re.IGNORECASE,
        )
        remainder = re.sub(
            r"(?:op_sel(?:_hi)?:\[[0-9, ]+\]|neg_(?:lo|hi):\[[0-9, ]+\]|clamp)",
            "",
            rest,
            flags=re.IGNORECASE,
        )
        tokens = [t.strip() for t in remainder.split(",") if t.strip()]

        clamp = any(m.lower() == "clamp" for m in modifiers)
        op_sel = (0, 0, 0)
        op_sel_hi = (1, 1, 1)
        neg_lo = (0, 0, 0)
        neg_hi = (0, 0, 0)

        for m in modifiers:
            m_low = m.lower()
            if m_low.startswith("op_sel:[") and m_low.endswith("]"):
                bits = [int(b.strip()) for b in m_low[8:-1].split(",") if b.strip()]
                op_sel = tuple(bits + [0] * (3 - len(bits)))
            elif m_low.startswith("op_sel_hi:[") and m_low.endswith("]"):
                bits = [int(b.strip()) for b in m_low[11:-1].split(",") if b.strip()]
                op_sel_hi = tuple(bits + [1] * (3 - len(bits)))

        if mnemonic == "v_mov_b32":
            dst_reg = int(tokens[0][1:])
            for lane in range(self.wave.wave_size):
                if self.wave.is_lane_active(lane):
                    val = self.read_operand(tokens[1], lane)
                    self.wave.set_vgpr(lane, dst_reg, val)
        elif mnemonic == "s_mov_b32":
            dst_reg = int(tokens[0][1:])
            val = self.read_operand(tokens[1], 0)
            self.wave.set_sgpr(dst_reg, val)
        elif mnemonic == "s_mov_b64" and tokens[0].lower() == "exec":
            mask = self.parse_imm64(tokens[1])
            self.wave.set_exec(mask)
        elif mnemonic in ("v_dot2_f16_f16", "v_dot2_f16_f16_e64"):
            dst_reg = int(tokens[0][1:])
            s0_tok = tokens[1]
            s1_tok = tokens[2]
            for lane in range(self.wave.wave_size):
                if self.wave.is_lane_active(lane):
                    s0_val = self.read_operand(s0_tok, lane)
                    s1_val = self.read_operand(s1_tok, lane)
                    res = ref_dot2_f16(
                        s0_val,
                        s1_val,
                        clamp=clamp,
                        op_sel=op_sel[:2],
                        op_sel_hi=op_sel_hi[:2],
                        neg_lo=neg_lo[:2],
                        neg_hi=neg_hi[:2],
                    )
                    self.wave.set_vgpr(lane, dst_reg, res)
        elif mnemonic in ("v_dot2acc_f16_f16", "v_dot2c_f16_f16", "v_dot2acc_f16_f16_e64"):
            dst_reg = int(tokens[0][1:])
            s0_tok = tokens[1]
            s1_tok = tokens[2]
            has_s2 = len(tokens) >= 4
            s2_tok = tokens[3] if has_s2 else None
            for lane in range(self.wave.wave_size):
                if self.wave.is_lane_active(lane):
                    s0_val = self.read_operand(s0_tok, lane)
                    s1_val = self.read_operand(s1_tok, lane)
                    s2_val = self.read_operand(s2_tok, lane) if has_s2 else self.wave.get_vgpr(lane, dst_reg)
                    res = ref_dot2acc_f16(
                        s0_val,
                        s1_val,
                        s2_acc=s2_val,
                        clamp=clamp,
                        op_sel=op_sel,
                        op_sel_hi=op_sel_hi,
                        neg_lo=neg_lo,
                        neg_hi=neg_hi,
                    )
                    self.wave.set_vgpr(lane, dst_reg, res)
        else:
            raise NotImplementedError(f"Interpreter opcode not implemented: {mnemonic}")

    def execute_program(self, lines: list[str]) -> None:
        for line in lines:
            self.execute_line(line)


class TestDot2AccInterpreter(unittest.TestCase):
    """Semantic interpreter tests for v_dot2_f16_f16 and accumulating v_dot2acc_f16_f16."""

    def test_v_dot2_interpreter_execution(self):
        interp = Dot2Interpreter(wave_size=64)
        s0 = pack_f16x2(H_ONE, H_TWO)
        s1 = pack_f16x2(H_THREE, H_FOUR)
        interp.execute_program([
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            "v_dot2_f16_f16 v0, v1, v2",
        ])
        for lane in range(64):
            self.assertEqual(f16_bits_to_float(interp.wave.get_vgpr(lane, 0)), 11.0)

    def test_v_dot2acc_explicit_s2_vgpr(self):
        interp = Dot2Interpreter(wave_size=64)
        s0 = pack_f16x2(H_ONE, H_HALF)
        s1 = pack_f16x2(H_FOUR, H_TWO)
        s2 = float_to_f16_bits(5.0)
        interp.execute_program([
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            f"v_mov_b32 v3, {s2:#x}",
            "v_dot2acc_f16_f16 v0, v1, v2, v3",
        ])
        # 1*4 + 0.5*2 + 5.0 = 4 + 1 + 5 = 10.0
        for lane in range(64):
            self.assertEqual(f16_bits_to_float(interp.wave.get_vgpr(lane, 0)), 10.0)

    def test_v_dot2acc_explicit_s2_sgpr(self):
        interp = Dot2Interpreter(wave_size=64)
        s0 = pack_f16x2(H_ONE, H_ONE)
        s1 = pack_f16x2(H_TWO, H_TWO)
        s2 = float_to_f16_bits(1.5)
        interp.execute_program([
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            f"s_mov_b32 s4, {s2:#x}",
            "v_dot2acc_f16_f16 v0, v1, v2, s4",
        ])
        # 1*2 + 1*2 + 1.5 = 5.5
        for lane in range(64):
            self.assertEqual(f16_bits_to_float(interp.wave.get_vgpr(lane, 0)), 5.5)

    def test_v_dot2acc_implicit_in_place_accumulator(self):
        interp = Dot2Interpreter(wave_size=64)
        s0 = pack_f16x2(H_HALF, H_HALF)
        s1 = pack_f16x2(H_TWO, H_TWO)
        init_acc = float_to_f16_bits(7.0)
        interp.execute_program([
            f"v_mov_b32 v0, {init_acc:#x}",
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            "v_dot2acc_f16_f16 v0, v1, v2",
        ])
        # 0.5*2 + 0.5*2 + 7.0 = 2.0 + 7.0 = 9.0
        for lane in range(64):
            self.assertEqual(f16_bits_to_float(interp.wave.get_vgpr(lane, 0)), 9.0)

    def test_v_dot2acc_chained_accumulation(self):
        interp = Dot2Interpreter(wave_size=64)
        # Vector 1: [1.0, 1.0] . [1.0, 1.0] = 2.0
        # Vector 2: [2.0, 2.0] . [1.0, 1.0] = 4.0 -> total 6.0
        # Vector 3: [0.5, 0.5] . [2.0, 2.0] = 2.0 -> total 8.0
        v1 = pack_f16x2(H_ONE, H_ONE)
        v2 = pack_f16x2(H_TWO, H_TWO)
        v3 = pack_f16x2(H_HALF, H_HALF)
        interp.execute_program([
            "v_mov_b32 v0, 0",
            f"v_mov_b32 v1, {v1:#x}",
            f"v_mov_b32 v2, {v2:#x}",
            f"v_mov_b32 v3, {v3:#x}",
            "v_dot2acc_f16_f16 v0, v1, v1",
            "v_dot2acc_f16_f16 v0, v2, v1",
            "v_dot2acc_f16_f16 v0, v3, v2",
        ])
        for lane in range(64):
            self.assertEqual(f16_bits_to_float(interp.wave.get_vgpr(lane, 0)), 8.0)

    def test_v_dot2acc_clamp_modifier(self):
        interp = Dot2Interpreter(wave_size=64)
        s0 = pack_f16x2(H_ONE, H_ONE)
        s1 = pack_f16x2(H_ONE, H_ONE)
        s2 = float_to_f16_bits(5.0)
        # 1*1 + 1*1 + 5 = 7.0 clamped -> 1.0
        interp.execute_program([
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            f"v_mov_b32 v3, {s2:#x}",
            "v_dot2acc_f16_f16 v0, v1, v2, v3 clamp",
        ])
        for lane in range(64):
            self.assertEqual(interp.wave.get_vgpr(lane, 0), H_ONE)

        # Clamping negative sum to 0.0
        s0_neg = pack_f16x2(H_NEG_ONE, H_NEG_ONE)
        interp.execute_program([
            f"v_mov_b32 v1, {s0_neg:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            "v_mov_b32 v3, 0",
            "v_dot2acc_f16_f16 v0, v1, v2, v3 clamp",
        ])
        # -1*1 + -1*1 + 0 = -2.0 clamped -> 0.0
        for lane in range(64):
            self.assertEqual(interp.wave.get_vgpr(lane, 0), H_POS_ZERO)

    def test_v_dot2acc_op_sel_reordering(self):
        interp = Dot2Interpreter(wave_size=64)
        s0 = pack_f16x2(H_ONE, H_TWO)
        s1 = pack_f16x2(H_THREE, H_FOUR)
        # op_sel:[1, 0] op_sel_hi:[0, 1]
        # Low term: S0.y * S1.x = 2 * 3 = 6
        # High term: S0.x * S1.y = 1 * 4 = 4
        # Total = 10.0
        interp.execute_program([
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            "v_mov_b32 v3, 0",
            "v_dot2acc_f16_f16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]",
        ])
        for lane in range(64):
            self.assertEqual(f16_bits_to_float(interp.wave.get_vgpr(lane, 0)), 10.0)


class TestWavefrontSimdExecution(unittest.TestCase):
    """Multi-lane SIMD wavefront execution tests for v_dot2acc_f16_f16."""

    def test_full_wavefront_uniform_execution(self):
        interp = Dot2Interpreter(wave_size=64)
        s0 = pack_f16x2(H_ONE, H_TWO)
        s1 = pack_f16x2(H_HALF, H_HALF)
        s2 = float_to_f16_bits(2.5)
        # 1*0.5 + 2*0.5 + 2.5 = 0.5 + 1.0 + 2.5 = 4.0
        interp.execute_program([
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            f"v_mov_b32 v3, {s2:#x}",
            "v_dot2acc_f16_f16 v0, v1, v2, v3",
        ])
        for lane in range(64):
            self.assertEqual(f16_bits_to_float(interp.wave.get_vgpr(lane, 0)), 4.0)

    def test_full_wavefront_divergent_operands(self):
        interp = Dot2Interpreter(wave_size=64)
        # Seed per-lane differing values
        for lane in range(64):
            f_x = float(lane % 8)
            f_y = float(lane // 8)
            h_x = float_to_f16_bits(f_x)
            h_y = float_to_f16_bits(f_y)
            interp.wave.set_vgpr(lane, 1, pack_f16x2(h_x, h_y))
            interp.wave.set_vgpr(lane, 2, pack_f16x2(H_ONE, H_TWO))
            interp.wave.set_vgpr(lane, 3, float_to_f16_bits(1.0))

        interp.execute_line("v_dot2acc_f16_f16 v0, v1, v2, v3")

        for lane in range(64):
            f_x = float(lane % 8)
            f_y = float(lane // 8)
            expected = f_x * 1.0 + f_y * 2.0 + 1.0
            actual = f16_bits_to_float(interp.wave.get_vgpr(lane, 0))
            self.assertEqual(actual, expected, f"Mismatch at lane {lane}")

    def test_wavefront_active_vs_inactive_exec_mask(self):
        interp = Dot2Interpreter(wave_size=64)
        # Sentinel in v0
        for lane in range(64):
            interp.wave.set_vgpr(lane, 0, 0xDEADBEEF)
            interp.wave.set_vgpr(lane, 1, pack_f16x2(H_ONE, H_ONE))
            interp.wave.set_vgpr(lane, 2, pack_f16x2(H_ONE, H_ONE))
            interp.wave.set_vgpr(lane, 3, H_POS_ZERO)

        # Alternating mask: even lanes active (bit 0, 2, 4...), odd lanes inactive
        mask = 0x5555555555555555
        interp.execute_line(f"s_mov_b64 exec, {mask:#x}")
        interp.execute_line("v_dot2acc_f16_f16 v0, v1, v2, v3")

        for lane in range(64):
            if lane % 2 == 0:
                # Active: 1*1 + 1*1 + 0 = 2.0
                self.assertEqual(f16_bits_to_float(interp.wave.get_vgpr(lane, 0)), 2.0)
            else:
                # Inactive: must preserve 0xDEADBEEF untouched!
                self.assertEqual(interp.wave.get_vgpr(lane, 0), 0xDEADBEEF)

    def test_wavefront_all_lanes_inactive(self):
        interp = Dot2Interpreter(wave_size=64)
        for lane in range(64):
            interp.wave.set_vgpr(lane, 0, 0x12345678)
            interp.wave.set_vgpr(lane, 1, pack_f16x2(H_ONE, H_ONE))
            interp.wave.set_vgpr(lane, 2, pack_f16x2(H_ONE, H_ONE))

        interp.execute_line("s_mov_b64 exec, 0")
        interp.execute_line("v_dot2_f16_f16 v0, v1, v2")

        for lane in range(64):
            self.assertEqual(interp.wave.get_vgpr(lane, 0), 0x12345678)

    def test_wavefront_single_lane_active(self):
        interp = Dot2Interpreter(wave_size=64)
        for lane in range(64):
            interp.wave.set_vgpr(lane, 0, 0xCAFEBABE)
            interp.wave.set_vgpr(lane, 1, pack_f16x2(H_TWO, H_TWO))
            interp.wave.set_vgpr(lane, 2, pack_f16x2(H_HALF, H_HALF))

        # Only lane 42 active: 1 << 42
        interp.execute_line(f"s_mov_b64 exec, {1 << 42:#x}")
        interp.execute_line("v_dot2_f16_f16 v0, v1, v2")

        for lane in range(64):
            if lane == 42:
                # 2*0.5 + 2*0.5 = 2.0
                self.assertEqual(f16_bits_to_float(interp.wave.get_vgpr(lane, 0)), 2.0)
            else:
                self.assertEqual(interp.wave.get_vgpr(lane, 0), 0xCAFEBABE)

    def test_wavefront_wave32_and_wave64_divergence(self):
        # Wave32 test
        w32 = Dot2Interpreter(wave_size=32)
        s0 = pack_f16x2(H_ONE, H_ONE)
        s1 = pack_f16x2(H_THREE, H_THREE)
        w32.execute_program([
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            "v_dot2_f16_f16 v0, v1, v2",
        ])
        for lane in range(32):
            self.assertEqual(f16_bits_to_float(w32.wave.get_vgpr(lane, 0)), 6.0)

        # Wave64 lower vs upper halves
        w64 = Dot2Interpreter(wave_size=64)
        for lane in range(64):
            w64.wave.set_vgpr(lane, 0, 0)
            w64.wave.set_vgpr(lane, 1, pack_f16x2(H_ONE, H_ONE))
            w64.wave.set_vgpr(lane, 2, pack_f16x2(H_TWO, H_TWO))
            w64.wave.set_vgpr(lane, 3, float_to_f16_bits(float(lane)))

        # Lower half executes: 0x00000000FFFFFFFF
        w64.execute_line("s_mov_b64 exec, 0x00000000FFFFFFFF")
        w64.execute_line("v_dot2acc_f16_f16 v0, v1, v2, v3")

        for lane in range(64):
            if lane < 32:
                expected = 1.0 * 2.0 + 1.0 * 2.0 + float(lane)
                self.assertEqual(f16_bits_to_float(w64.wave.get_vgpr(lane, 0)), expected)
            else:
                self.assertEqual(w64.wave.get_vgpr(lane, 0), 0)

    def test_wavefront_mixed_edge_cases_across_lanes(self):
        interp = Dot2Interpreter(wave_size=64)
        test_lanes = [
            (0, pack_f16x2(H_POS_ZERO, H_POS_ZERO), pack_f16x2(H_POS_ZERO, H_POS_ZERO), H_POS_ZERO, H_POS_ZERO),
            (1, pack_f16x2(H_SUB_MIN, H_POS_ZERO), pack_f16x2(H_ONE, H_POS_ZERO), H_POS_ZERO, H_SUB_MIN),
            (2, pack_f16x2(H_POS_INF, H_POS_ZERO), pack_f16x2(H_ONE, H_POS_ZERO), H_POS_ZERO, H_POS_INF),
            (3, pack_f16x2(H_QNAN, H_POS_ZERO), pack_f16x2(H_ONE, H_POS_ZERO), H_POS_ZERO, H_QNAN),
            (4, pack_f16x2(H_NORM_MAX, H_POS_ZERO), pack_f16x2(H_TWO, H_POS_ZERO), H_POS_ZERO, H_POS_INF),
            (5, pack_f16x2(H_NEG_ONE, H_NEG_ONE), pack_f16x2(H_ONE, H_ONE), H_POS_ZERO, float_to_f16_bits(-2.0)),
        ]
        for lane, s0, s1, s2, _ in test_lanes:
            interp.wave.set_vgpr(lane, 1, s0)
            interp.wave.set_vgpr(lane, 2, s1)
            interp.wave.set_vgpr(lane, 3, s2)

        interp.execute_line("v_dot2acc_f16_f16 v0, v1, v2, v3")

        for lane, s0, s1, s2, exp in test_lanes:
            actual = interp.wave.get_vgpr(lane, 0)
            if exp == H_QNAN:
                self.assertTrue(math.isnan(f16_bits_to_float(actual)))
            else:
                self.assertEqual(actual, exp, f"Edge case mismatch at lane {lane}")


if __name__ == "__main__":
    unittest.main()
