"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_trig_f64.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  Operations on 64-bit double-precision floating-point quantities (binary64, IEEE 754)
  operate across aligned pairs of adjacent registers: v[d:d+1] from source operand
  v[s0:s0+1] (or scalar SGPR pair s[k:k+1], literal 64-bit constants, or inline constants).

  Instruction:
    v_trig_f64 v[d:d+1], src0 [clamp]
    (with trigonometric function / argument evaluation in units of 2*pi revolutions / turns)

  Mathematical Specification:
    1. Angle Domain:
       Input src0 is specified in cycles / revolutions (turns), where 1.0 turn = 2*pi radians.
       The hardware reduces the input argument modulo 1.0 to the fundamental domain [-0.5, 0.5]
       using IEEE roundTiesToEven:
         reduced_turns = turns - round_even(turns)
       This exact range reduction ensures zero transcendental error drift across multi-turn rotations.
    2. Primary Evaluation:
       The primary trigonometric function evaluates:
         radians = reduced_turns * 2*pi
         D.f64 = sin(radians)    (or cos(radians) in dual / cos-mode)
    3. IEEE 754 & RDNA2 Rules:
       - Signed Zero:
           sin(+0.0) = +0.0,  sin(-0.0) = -0.0
           cos(+0.0) = 1.0,   cos(-0.0) = 1.0
       - Infinities:
           sin(+-inf) = Canonical QNaN
           cos(+-inf) = Canonical QNaN
       - NaNs:
           NaN operands are quieted and propagated as Canonical Quiet NaNs.
       - Subnormals:
           Denormal / subnormal inputs are preserved without flushing in IEEE compliance mode.
       - Cardinal Angles (in turns):
           turns = 0.00  -> sin =  0.0, cos =  1.0
           turns = 0.25  -> sin =  1.0, cos =  0.0  (pi/2 rad)
           turns = 0.50  -> sin =  0.0, cos = -1.0  (pi rad)
           turns = 0.75  -> sin = -1.0, cos =  0.0  (3*pi/2 rad)
       - VOP3 Input Modifiers:
           abs (|x|) clears sign bit 63.
           neg (-x) toggles sign bit 63.
           neg(abs(x)) forces the sign bit to 1 (negative).
       - Output Clamp:
           clamp forces result into [0.0, 1.0]. Negative results or -0.0 clamp to +0.0.
           In RDNA2 VALU, clamping NaN yields +0.0.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

import enum
import math
import struct
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
TWO_PI: float = 6.283185307179586476925286766559

# IEEE-754 binary64 bit constants
F64_POS_ZERO: int = 0x0000000000000000
F64_NEG_ZERO: int = 0x8000000000000000
F64_POS_INF: int = 0x7FF0000000000000
F64_NEG_INF: int = 0xFFF0000000000000
F64_CANONICAL_QNAN: int = 0x7FF8000000000000
F64_QUIET_BIT: int = 0x0008000000000000

F64_EXP_MASK: int = 0x7FF0000000000000
F64_MANT_MASK: int = 0x000FFFFFFFFFFFFF
F64_SIGN_MASK: int = 0x8000000000000000

F64_ONE: int = 0x3FF0000000000000          # 1.0
F64_NEG_ONE: int = 0xBFF0000000000000      # -1.0
F64_TWO: int = 0x4000000000000000          # 2.0
F64_HALF: int = 0x3FE0000000000000         # 0.5
F64_QUARTER: int = 0x3FD0000000000000      # 0.25 (1/4 turn = 90 deg = pi/2 rad)
F64_NEG_QUARTER: int = 0xBFD0000000000000  # -0.25 (-1/4 turn = -90 deg = -pi/2 rad)
F64_THREE_QUARTERS: int = 0x3FE8000000000000  # 0.75 (3/4 turn = 270 deg = 3*pi/2 rad)

F64_MIN_NORMAL: int = 0x0010000000000000   # 2^-1022
F64_MAX_NORMAL: int = 0x7FEFFFFFFFFFFFFF   # (2 - 2^-52) * 2^1023
F64_MIN_SUBNORMAL: int = 0x0000000000000001  # 2^-1074
F64_MAX_SUBNORMAL: int = 0x000FFFFFFFFFFFFF

OLD_VGPR_SENTINEL: int = 0xDEADBEEF


class TrigMode(enum.Enum):
    SIN = 0
    COS = 1


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def f64_to_u64(val: float) -> int:
    """Pack an IEEE-754 double precision float into an unsigned 64-bit integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return F64_NEG_INF if val < 0 else F64_POS_INF


def u64_to_f64(bits: int) -> float:
    """Unpack an unsigned 64-bit integer into an IEEE-754 double precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def is_f64_nan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents an IEEE-754 NaN."""
    return ((bits & F64_EXP_MASK) == F64_EXP_MASK) and ((bits & F64_MANT_MASK) != 0)


def is_f64_snan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a signaling NaN."""
    return is_f64_nan(bits) and ((bits & F64_QUIET_BIT) == 0)


def is_f64_qnan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a quiet NaN."""
    return is_f64_nan(bits) and ((bits & F64_QUIET_BIT) != 0)


def is_f64_zero(bits: int) -> bool:
    """Return True if the 64-bit pattern represents +0.0 or -0.0."""
    return (bits & (F64_EXP_MASK | F64_MANT_MASK)) == 0


def is_f64_inf(bits: int) -> bool:
    """Return True if the 64-bit pattern represents +inf or -inf."""
    return ((bits & F64_EXP_MASK) == F64_EXP_MASK) and ((bits & F64_MANT_MASK) == 0)


def is_f64_subnormal(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a subnormal/denormal float."""
    return ((bits & F64_EXP_MASK) == 0) and ((bits & F64_MANT_MASK) != 0)


def is_f64_normal(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a normalized finite float."""
    exp = (bits >> 52) & 0x7FF
    return 0 < exp < 0x7FF


def f64_sign(bits: int) -> int:
    """Return the sign bit (0 or 1)."""
    return (bits >> 63) & 1


def quiet_f64(bits: int) -> int:
    """Force an IEEE-754 double NaN to quiet by setting the quiet bit."""
    return (bits & MASK64) | F64_QUIET_BIT


def round_even(x: float) -> float:
    """Round float to nearest integer, breaking ties to even."""
    return float(round(x))


def ref_turn_reduce_f64(turns: float) -> float:
    """Reduce turn count to [-0.5, 0.5] turns using roundTiesToEven."""
    if math.isnan(turns) or math.isinf(turns):
        return float("nan")
    if turns == 0.0:
        return turns
    return turns - round_even(turns)


def apply_input_modifiers_f64(bits: int, *, neg: bool = False, abs_val: bool = False) -> int:
    """Apply VOP3 input modifiers (abs, neg).

    In RDNA2 ISA:
      When both abs and neg are set, abs is evaluated before neg:
      i.e. neg(abs(x)) forces the sign bit to 1 (negative).
    """
    res = bits & MASK64
    if abs_val:
        res &= ~F64_SIGN_MASK
    if neg:
        res ^= F64_SIGN_MASK
    return res


def apply_f64_clamp(bits: int) -> int:
    """Apply VOP3 clamp modifier [0.0, 1.0]. NaN or negatives clamp to +0.0."""
    if is_f64_nan(bits):
        return F64_POS_ZERO
    val = u64_to_f64(bits)
    if val < 0.0 or is_f64_zero(bits):
        return F64_POS_ZERO
    elif val > 1.0:
        return F64_ONE
    return bits & MASK64


def ref_trig_f64(
    src0_bits: int,
    *,
    mode: TrigMode = TrigMode.SIN,
    clamp: bool = False,
    src0_neg: bool = False,
    src0_abs: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 double-precision v_trig_f64.

    Evaluates:
      D.f64 = trig(src0 * 2*pi)
    in units of revolutions / turns with exact [-0.5, 0.5] range reduction.
    """
    s0 = apply_input_modifiers_f64(src0_bits, neg=src0_neg, abs_val=src0_abs)

    # NaN input produces canonical quiet NaN
    if is_f64_nan(s0):
        res = quiet_f64(s0)
    # Infinities produce canonical quiet NaN
    elif is_f64_inf(s0):
        res = F64_CANONICAL_QNAN
    # Zero input rules
    elif is_f64_zero(s0):
        if mode == TrigMode.SIN:
            res = F64_NEG_ZERO if f64_sign(s0) else F64_POS_ZERO
        else:
            res = F64_ONE
    else:
        turns = u64_to_f64(s0)
        reduced = ref_turn_reduce_f64(turns)

        if mode == TrigMode.SIN:
            if reduced == 0.0:
                res = F64_NEG_ZERO if f64_sign(s0) else F64_POS_ZERO
            elif reduced == 0.25:
                res = F64_ONE
            elif reduced == -0.25:
                res = F64_NEG_ONE
            elif abs(reduced) == 0.5:
                res = F64_POS_ZERO
            else:
                rad = reduced * TWO_PI
                res = f64_to_u64(math.sin(rad))
                if is_f64_zero(res):
                    res = F64_NEG_ZERO if reduced < 0 else F64_POS_ZERO
        else:  # COS
            if reduced == 0.0:
                res = F64_ONE
            elif abs(reduced) == 0.25:
                res = F64_POS_ZERO
            elif abs(reduced) == 0.5:
                res = F64_NEG_ONE
            else:
                rad = reduced * TWO_PI
                res = f64_to_u64(math.cos(rad))

    if clamp:
        res = apply_f64_clamp(res)

    return res & MASK64


def ref_trig_f64_dwords(
    s0_lo: int,
    s0_hi: int,
    *,
    mode: TrigMode = TrigMode.SIN,
    clamp: bool = False,
    src0_neg: bool = False,
    src0_abs: bool = False,
) -> tuple[int, int]:
    """Decomposed dword interface for v_trig_f64."""
    s0 = join_dwords(s0_lo, s0_hi)
    res = ref_trig_f64(
        s0,
        mode=mode,
        clamp=clamp,
        src0_neg=src0_neg,
        src0_abs=src0_abs,
    )
    return split_dwords(res)


class VALURegisterFile:
    """Architectural model for RDNA2 VALU 32-bit vector register file."""

    def __init__(self, num_vgprs: int = 256) -> None:
        self.num_vgprs = num_vgprs
        self.regs = [0] * num_vgprs

    def get_vgpr(self, idx: int) -> int:
        if not (0 <= idx < self.num_vgprs):
            raise IndexError(f"VGPR index out of range: {idx}")
        return self.regs[idx] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < self.num_vgprs):
            raise IndexError(f"VGPR index out of range: {idx}")
        self.regs[idx] = val & MASK32

    def get_vgpr_pair(self, idx: int) -> int:
        if not (0 <= idx < self.num_vgprs - 1):
            raise IndexError(f"VGPR pair index out of range: v[{idx}:{idx+1}]")
        lo = self.get_vgpr(idx)
        hi = self.get_vgpr(idx + 1)
        return join_dwords(lo, hi)

    def set_vgpr_pair(self, idx: int, val64: int) -> None:
        if not (0 <= idx < self.num_vgprs - 1):
            raise IndexError(f"VGPR pair index out of range: v[{idx}:{idx+1}]")
        lo, hi = split_dwords(val64)
        self.set_vgpr(idx, lo)
        self.set_vgpr(idx + 1, hi)


class TestVTrigF64ReferenceOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_cardinal_angles_sin(self) -> None:
        # 0.0 turn -> sin(0) = 0.0
        self.assertEqual(ref_trig_f64(F64_POS_ZERO, mode=TrigMode.SIN), F64_POS_ZERO)
        # 0.25 turn -> sin(pi/2) = 1.0
        self.assertEqual(ref_trig_f64(F64_QUARTER, mode=TrigMode.SIN), F64_ONE)
        # 0.50 turn -> sin(pi) = 0.0
        self.assertEqual(ref_trig_f64(F64_HALF, mode=TrigMode.SIN), F64_POS_ZERO)
        # 0.75 turn -> sin(3*pi/2) = -1.0
        self.assertEqual(ref_trig_f64(F64_THREE_QUARTERS, mode=TrigMode.SIN), F64_NEG_ONE)
        # 1.00 turn -> sin(2*pi) = 0.0
        self.assertEqual(ref_trig_f64(F64_ONE, mode=TrigMode.SIN), F64_POS_ZERO)

    def test_cardinal_angles_cos(self) -> None:
        # 0.0 turn -> cos(0) = 1.0
        self.assertEqual(ref_trig_f64(F64_POS_ZERO, mode=TrigMode.COS), F64_ONE)
        # 0.25 turn -> cos(pi/2) = 0.0
        self.assertEqual(ref_trig_f64(F64_QUARTER, mode=TrigMode.COS), F64_POS_ZERO)
        # 0.50 turn -> cos(pi) = -1.0
        self.assertEqual(ref_trig_f64(F64_HALF, mode=TrigMode.COS), F64_NEG_ONE)
        # 0.75 turn -> cos(3*pi/2) = 0.0
        self.assertEqual(ref_trig_f64(F64_THREE_QUARTERS, mode=TrigMode.COS), F64_POS_ZERO)
        # 1.00 turn -> cos(2*pi) = 1.0
        self.assertEqual(ref_trig_f64(F64_ONE, mode=TrigMode.COS), F64_ONE)

    def test_signed_zeros(self) -> None:
        self.assertEqual(ref_trig_f64(F64_POS_ZERO, mode=TrigMode.SIN), F64_POS_ZERO)
        self.assertEqual(ref_trig_f64(F64_NEG_ZERO, mode=TrigMode.SIN), F64_NEG_ZERO)
        self.assertEqual(ref_trig_f64(F64_POS_ZERO, mode=TrigMode.COS), F64_ONE)
        self.assertEqual(ref_trig_f64(F64_NEG_ZERO, mode=TrigMode.COS), F64_ONE)

    def test_infinities_produce_qnan(self) -> None:
        self.assertEqual(ref_trig_f64(F64_POS_INF, mode=TrigMode.SIN), F64_CANONICAL_QNAN)
        self.assertEqual(ref_trig_f64(F64_NEG_INF, mode=TrigMode.SIN), F64_CANONICAL_QNAN)
        self.assertEqual(ref_trig_f64(F64_POS_INF, mode=TrigMode.COS), F64_CANONICAL_QNAN)
        self.assertEqual(ref_trig_f64(F64_NEG_INF, mode=TrigMode.COS), F64_CANONICAL_QNAN)

    def test_nan_propagation(self) -> None:
        snan = 0x7FF0000000000001
        res_sin = ref_trig_f64(snan, mode=TrigMode.SIN)
        self.assertTrue(is_f64_qnan(res_sin))
        res_cos = ref_trig_f64(snan, mode=TrigMode.COS)
        self.assertTrue(is_f64_qnan(res_cos))

    def test_input_modifiers(self) -> None:
        # neg: sin(-0.25 turn) = -1.0; with neg modifier: sin(-(-0.25)) = sin(0.25) = 1.0
        r = ref_trig_f64(F64_NEG_QUARTER, mode=TrigMode.SIN, src0_neg=True)
        self.assertEqual(r, F64_ONE)

        # abs: sin(|-0.25|) = sin(0.25) = 1.0
        r = ref_trig_f64(F64_NEG_QUARTER, mode=TrigMode.SIN, src0_abs=True)
        self.assertEqual(r, F64_ONE)

        # abs + neg: neg(abs(0.25)) = -0.25 -> sin(-0.25) = -1.0
        r = ref_trig_f64(F64_QUARTER, mode=TrigMode.SIN, src0_abs=True, src0_neg=True)
        self.assertEqual(r, F64_NEG_ONE)

    def test_clamp_modifier(self) -> None:
        # In range [0.0, 1.0]: sin(0.125 turn) = sin(pi/4) ~= 0.7071
        r = ref_trig_f64(f64_to_u64(0.125), mode=TrigMode.SIN, clamp=True)
        self.assertAlmostEqual(u64_to_f64(r), math.sin(math.pi / 4), places=7)

        # Negative clamped to +0.0: sin(-0.25) = -1.0 -> clamped to +0.0
        r = ref_trig_f64(F64_NEG_QUARTER, mode=TrigMode.SIN, clamp=True)
        self.assertEqual(r, F64_POS_ZERO)

        # NaN clamped to +0.0
        r = ref_trig_f64(F64_POS_INF, mode=TrigMode.SIN, clamp=True)
        self.assertEqual(r, F64_POS_ZERO)

    def test_turn_periodicity(self) -> None:
        # Any value x and x + 100.0 turns should yield the exact same result
        base_turns = 0.1738294
        for k in (1.0, 10.0, 1000.0, -50.0):
            t1 = f64_to_u64(base_turns)
            t2 = f64_to_u64(base_turns + k)
            self.assertAlmostEqual(
                u64_to_f64(ref_trig_f64(t1, mode=TrigMode.SIN)),
                u64_to_f64(ref_trig_f64(t2, mode=TrigMode.SIN)),
                places=9,
            )

    def test_pythagorean_identity(self) -> None:
        # sin^2(x) + cos^2(x) == 1.0 for arbitrary turn values
        for angle in (0.05, 0.1337, 0.33333333, -0.42, 12.3456):
            bits = f64_to_u64(angle)
            s = u64_to_f64(ref_trig_f64(bits, mode=TrigMode.SIN))
            c = u64_to_f64(ref_trig_f64(bits, mode=TrigMode.COS))
            self.assertAlmostEqual(s * s + c * c, 1.0, places=12)

    def test_dword_decomposition(self) -> None:
        s0 = f64_to_u64(0.25)
        s0_lo, s0_hi = split_dwords(s0)
        lo, hi = ref_trig_f64_dwords(s0_lo, s0_hi, mode=TrigMode.SIN)
        res64 = join_dwords(lo, hi)
        self.assertEqual(res64, F64_ONE)

    def test_register_file_pair_access(self) -> None:
        rf = VALURegisterFile(num_vgprs=16)
        val = 0x01234567_89ABCDEF
        rf.set_vgpr_pair(4, val)

        self.assertEqual(rf.get_vgpr(4), 0x89ABCDEF)
        self.assertEqual(rf.get_vgpr(5), 0x01234567)
        self.assertEqual(rf.get_vgpr_pair(4), val)

        with self.assertRaises(IndexError):
            rf.get_vgpr_pair(15)


if __name__ == "__main__":
    unittest.main()
