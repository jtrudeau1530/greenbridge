from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_subrev_f16.

Instruction:
  v_subrev_f16 dst, src0, src1 (VOP2 / VOP3)
  D.f16 = S1.f16 - S0.f16

Architecture:
  - RDNA2 Vector ALU (GFX10) half-precision floating-point reverse subtraction.
  - Computes the difference S1 - S0 conforming to IEEE 754-2008 binary16.
  - In VOP2 format:
      * S0: 16-bit float operand (VGPR, SGPR, or 16-bit inline constant / literal).
      * S1: 16-bit float operand (VGPR).
      * D:  16-bit float destination VGPR (writes low 16 bits, preserves high 16 bits).
  - In VOP3 format:
      * Input modifiers: abs (|S|), neg (-S) on both src0 and src1.
      * Output modifiers: clamp (clamps result to [0.0, 1.0]), omod (mul:2, mul:4, div:2).
      * op_sel: selects low or high 16-bit half for src0, src1, and dst independently.
"""

from fractions import Fraction
import math
import struct
import unittest

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

SIGN_MASK_F16: int = 0x8000
EXP_MASK_F16: int = 0x7C00
MANT_MASK_F16: int = 0x03FF
QNAN_BIT_F16: int = 0x0200
BIAS_F16: int = 15

POS_ZERO_F16: int = 0x0000
NEG_ZERO_F16: int = 0x8000
POS_INF_F16: int = 0x7C00
NEG_INF_F16: int = 0xFC00
CANONICAL_QNAN_F16: int = 0x7E00
SNAN_DEFAULT_F16: int = 0x7C01

ONE_F16: int = 0x3C00          # 1.0
TWO_F16: int = 0x4000          # 2.0
THREE_F16: int = 0x4200        # 3.0
FOUR_F16: int = 0x4400         # 4.0
HALF_F16: int = 0x3800         # 0.5
QUARTER_F16: int = 0x3400      # 0.25
NEG_ONE_F16: int = 0xBC00      # -1.0
NEG_TWO_F16: int = 0xC000      # -2.0
NEG_HALF_F16: int = 0xB800     # -0.5

MAX_FINITE_F16: int = 0x7BFF   # 65504.0
NEG_MAX_FINITE_F16: int = 0xFBFF  # -65504.0
MIN_NORMAL_F16: int = 0x0400   # 2^-14 (~6.1035e-5)
MAX_SUBNORMAL_F16: int = 0x03FF # ~6.0975e-5
MIN_SUBNORMAL_F16: int = 0x0001 # 2^-24 (~5.9605e-8)


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s16(val: int) -> int:
    """Interpret a 16-bit unsigned value as signed two's complement integer."""
    val &= MASK16
    return val - (1 << 16) if (val & 0x8000) else val


def bits_to_f16(bits: int) -> float:
    """Convert unsigned 16-bit integer bits to Python float (IEEE 754 binary16)."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def f16_to_bits(val: float) -> int:
    """Convert Python float to unsigned 16-bit integer bits (IEEE 754 binary16 RNE)."""
    if math.isnan(val):
        return CANONICAL_QNAN_F16
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return NEG_INF_F16 if val < 0.0 else POS_INF_F16


def split_f16_bits(bits: int) -> tuple[int, int, int]:
    """Unpack 16-bit float into (sign, exponent, fraction) components."""
    bits &= MASK16
    sign = (bits & SIGN_MASK_F16) >> 15
    exponent = (bits & EXP_MASK_F16) >> 10
    fraction = bits & MANT_MASK_F16
    return sign, exponent, fraction


def is_nan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) != 0)


def is_qnan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_snan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """Return True if bits represent IEEE 754 positive or negative infinity."""
    return (bits & 0x7FFF) == 0x7C00


def is_zero_f16(bits: int) -> bool:
    """Return True if bits represent positive or negative zero (+0.0 or -0.0)."""
    return (bits & ~SIGN_MASK_F16) == 0


def is_subnormal_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision subnormal (denormal)."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & MANT_MASK_F16) != 0)


def is_normal_f16(bits: int) -> bool:
    """Return True if bits represent a normalized finite half-precision float."""
    exp = (bits & EXP_MASK_F16) >> 10
    return 1 <= exp <= 30


def pack_half(lo: int, hi: int) -> int:
    """Pack two 16-bit halfwords into a 32-bit register word."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def unpack_half(word: int) -> tuple[int, int]:
    """Unpack a 32-bit register word into (lo, hi) 16-bit halfwords."""
    return word & MASK16, (word >> 16) & MASK16


def extract_half(word: int, hi_sel: bool) -> int:
    """Extract 16-bit halfword from 32-bit word using high-select boolean flag."""
    return (word >> 16) & MASK16 if hi_sel else word & MASK16


def insert_half(old_word: int, val16: int, dst_hi: bool) -> int:
    """Insert 16-bit halfword into low or high half of 32-bit word, preserving other half."""
    val16 &= MASK16
    old_word &= MASK32
    if dst_hi:
        return (val16 << 16) | (old_word & 0x0000FFFF)
    else:
        return (old_word & 0xFFFF0000) | val16


def apply_f16_input_mods(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & MASK16
    if abs_mod:
        res &= ~SIGN_MASK_F16
    if neg_mod:
        res ^= SIGN_MASK_F16
    return res


def apply_f16_output_mods(bits: int, clamp: bool, omod: str = "none") -> int:
    """Apply VOP3 output modifiers: omod scaling and clamp to [0.0, 1.0]."""
    bits &= MASK16
    if is_nan_f16(bits):
        # Clamping NaN in RDNA2 produces +0.0
        return POS_ZERO_F16 if clamp else bits

    # Apply omod scaling if not NaN and not infinity
    if not is_inf_f16(bits) and omod != "none":
        val = bits_to_f16(bits)
        if omod == "mul:2":
            val *= 2.0
        elif omod == "mul:4":
            val *= 4.0
        elif omod == "div:2":
            val *= 0.5
        bits = f16_to_bits(val)

    if clamp:
        if is_inf_f16(bits):
            return ONE_F16 if bits == POS_INF_F16 else POS_ZERO_F16
        val = bits_to_f16(bits)
        if val < 0.0 or is_zero_f16(bits):
            return POS_ZERO_F16
        elif val > 1.0:
            return ONE_F16
        return bits

    return bits


def ref_subrev_f16(
    src0_bits: int,
    src1_bits: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
    omod: str = "none",
) -> int:
    """Reference mathematical oracle for RDNA2 v_subrev_f16 half-precision reverse subtraction.

    Operation: D = src1 - src0 (S1 - S0)
    Conforms strictly to IEEE 754-2008 binary16 Round-to-Nearest-Even (RNE)
    with RDNA2 special values, NaN quietening, and input/output modifiers.
    """
    s0 = apply_f16_input_mods(src0_bits, abs0, neg0)
    s1 = apply_f16_input_mods(src1_bits, abs1, neg1)

    # 1. NaN handling
    s0_nan = is_nan_f16(s0)
    s1_nan = is_nan_f16(s1)
    if s0_nan or s1_nan:
        if clamp:
            return POS_ZERO_F16
        nan_bits = s0 if s0_nan else s1
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
        # (+inf) - (-inf) = +inf; (-inf) - (+inf) = -inf
        res_inf = s1
        return apply_f16_output_mods(res_inf, clamp, omod)
    if s1_inf:
        return apply_f16_output_mods(s1, clamp, omod)
    if s0_inf:
        # finite - (+inf) = -inf; finite - (-inf) = +inf
        res_inf = NEG_INF_F16 if s0 == POS_INF_F16 else POS_INF_F16
        return apply_f16_output_mods(res_inf, clamp, omod)

    # 3. Finite numbers
    val0 = bits_to_f16(s0)
    val1 = bits_to_f16(s1)
    diff = val1 - val0

    # Signed zero preservation:
    # (+0.0) - (+0.0) == +0.0
    # (-0.0) - (-0.0) == +0.0
    # (+0.0) - (-0.0) == +0.0
    # (-0.0) - (+0.0) == -0.0
    if diff == 0.0:
        if val1 == 0.0 and val0 == 0.0:
            if (s1 & SIGN_MASK_F16) and not (s0 & SIGN_MASK_F16):
                res_zero = NEG_ZERO_F16
            else:
                res_zero = POS_ZERO_F16
        else:
            res_zero = POS_ZERO_F16
        return apply_f16_output_mods(res_zero, clamp, omod)

    raw_res = f16_to_bits(diff)
    return apply_f16_output_mods(raw_res, clamp, omod)


def f16_bits_to_fraction(bits: int) -> Fraction:
    """Convert a finite binary16 bit pattern to an exact rational Fraction."""
    sign, exp, mant = split_f16_bits(bits)
    if exp == 0:
        # Subnormal: 0.mant * 2^-14 = mant * 2^-24
        val = Fraction(mant, 1 << 24)
    else:
        # Normalized: 1.mant * 2^(exp - 15) = (1024 + mant) * 2^(exp - 25)
        num = (1 << 10) | mant
        shift = exp - 25
        if shift >= 0:
            val = Fraction(num << shift, 1)
        else:
            val = Fraction(num, 1 << (-shift))
    return -val if sign else val


def ref_subrev_f16_rational(
    src0_bits: int,
    src1_bits: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Independent exact rational arithmetic cross-check oracle for v_subrev_f16.

    Uses fractions.Fraction to compute the exact difference S1 - S0 without intermediate
    floating-point inaccuracy, then applies bit-exact IEEE 754 half-precision RNE rounding.
    """
    s0 = apply_f16_input_mods(src0_bits, abs0, neg0)
    s1 = apply_f16_input_mods(src1_bits, abs1, neg1)

    if is_nan_f16(s0) or is_nan_f16(s1):
        if clamp:
            return POS_ZERO_F16
        nan_bits = s0 if is_nan_f16(s0) else s1
        return nan_bits | QNAN_BIT_F16

    if is_inf_f16(s0) or is_inf_f16(s1):
        return ref_subrev_f16(src0_bits, src1_bits, abs0=abs0, neg0=neg0, abs1=abs1, neg1=neg1, clamp=clamp)

    frac0 = f16_bits_to_fraction(s0)
    frac1 = f16_bits_to_fraction(s1)
    diff_frac = frac1 - frac0

    if diff_frac == 0:
        if (s1 & SIGN_MASK_F16) and not (s0 & SIGN_MASK_F16):
            res_zero = NEG_ZERO_F16
        else:
            res_zero = POS_ZERO_F16
        return apply_f16_output_mods(res_zero, clamp)

    sign = 1 if diff_frac < 0 else 0
    abs_diff = abs(diff_frac)

    # Check overflow beyond max finite
    max_finite_val = Fraction(65504, 1)
    if abs_diff >= Fraction(65520, 1):
        res_inf = NEG_INF_F16 if sign else POS_INF_F16
        return apply_f16_output_mods(res_inf, clamp)

    # Normal vs subnormal bounds
    min_normal_val = Fraction(1, 1 << 14)
    if abs_diff < min_normal_val:
        # Subnormal representation: mant = abs_diff * 2^24 rounded
        scaled = abs_diff * (1 << 24)
        floor_mant = scaled.numerator // scaled.denominator
        rem = scaled - floor_mant
        half_one = Fraction(1, 2)
        if rem > half_one or (rem == half_one and (floor_mant % 2 == 1)):
            floor_mant += 1
        if floor_mant >= (1 << 10):
            res_bits = (sign << 15) | (1 << 10)  # promoted to min normal
        else:
            res_bits = (sign << 15) | floor_mant
        return apply_f16_output_mods(res_bits, clamp)

    # Normalized float: find exponent
    val_float = float(abs_diff)
    exp = int(math.floor(math.log2(val_float))) + 15
    exp = max(1, min(30, exp))

    scaled = abs_diff * Fraction(1 << (25 - exp), 1)
    num = scaled.numerator // scaled.denominator
    rem = scaled - num
    half_one = Fraction(1, 2)
    if rem > half_one or (rem == half_one and (num % 2 == 1)):
        num += 1

    if num >= (1 << 11):
        num >>= 1
        exp += 1

    if exp >= 31:
        res_inf = NEG_INF_F16 if sign else POS_INF_F16
        return apply_f16_output_mods(res_inf, clamp)

    mant = num & MANT_MASK_F16
    res_bits = (sign << 15) | (exp << 10) | mant
    return apply_f16_output_mods(res_bits, clamp)


class TestValuSubrevF16Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model of v_subrev_f16."""

    def test_basic_normal_subtraction(self) -> None:
        # 3.0 - 1.0 = 2.0 (S1=3.0, S0=1.0)
        res = ref_subrev_f16(ONE_F16, THREE_F16)
        self.assertEqual(res, TWO_F16)
        self.assertEqual(bits_to_f16(res), 2.0)

        # 4.0 - 1.5 = 2.5
        f_1p5 = f16_to_bits(1.5)
        res = ref_subrev_f16(f_1p5, FOUR_F16)
        self.assertEqual(bits_to_f16(res), 2.5)

        # 0.5 - 2.0 = -1.5
        res = ref_subrev_f16(TWO_F16, HALF_F16)
        self.assertEqual(bits_to_f16(res), -1.5)

    def test_reverse_subtraction_asymmetry(self) -> None:
        # S1 - S0 != S0 - S1
        res1 = ref_subrev_f16(ONE_F16, THREE_F16)   # 3.0 - 1.0 = 2.0
        res2 = ref_subrev_f16(THREE_F16, ONE_F16)   # 1.0 - 3.0 = -2.0
        self.assertEqual(bits_to_f16(res1), 2.0)
        self.assertEqual(bits_to_f16(res2), -2.0)
        self.assertNotEqual(res1, res2)

    def test_cross_check_with_rational_oracle(self) -> None:
        test_values = [
            f16_to_bits(x)
            for x in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 7.5, 12.25, -0.5, -1.0, -2.5]
        ]
        for s0 in test_values:
            for s1 in test_values:
                ref_res = ref_subrev_f16(s0, s1)
                rat_res = ref_subrev_f16_rational(s0, s1)
                self.assertEqual(ref_res, rat_res, f"Mismatch for s0={s0:#06x}, s1={s1:#06x}")

    def test_signed_zeros(self) -> None:
        # (+0.0) - (+0.0) == +0.0
        self.assertEqual(ref_subrev_f16(POS_ZERO_F16, POS_ZERO_F16), POS_ZERO_F16)
        # (-0.0) - (-0.0) == +0.0
        self.assertEqual(ref_subrev_f16(NEG_ZERO_F16, NEG_ZERO_F16), POS_ZERO_F16)
        # (+0.0) - (-0.0) == +0.0
        self.assertEqual(ref_subrev_f16(NEG_ZERO_F16, POS_ZERO_F16), POS_ZERO_F16)
        # (-0.0) - (+0.0) == -0.0
        self.assertEqual(ref_subrev_f16(POS_ZERO_F16, NEG_ZERO_F16), NEG_ZERO_F16)

    def test_infinity_arithmetic(self) -> None:
        # (+inf) - (+inf) = Canonical Quiet NaN
        self.assertEqual(ref_subrev_f16(POS_INF_F16, POS_INF_F16), CANONICAL_QNAN_F16)
        # (-inf) - (-inf) = Canonical Quiet NaN
        self.assertEqual(ref_subrev_f16(NEG_INF_F16, NEG_INF_F16), CANONICAL_QNAN_F16)
        # (+inf) - (-inf) = +inf
        self.assertEqual(ref_subrev_f16(NEG_INF_F16, POS_INF_F16), POS_INF_F16)
        # (-inf) - (+inf) = -inf
        self.assertEqual(ref_subrev_f16(POS_INF_F16, NEG_INF_F16), NEG_INF_F16)
        # finite - (+inf) = -inf
        self.assertEqual(ref_subrev_f16(POS_INF_F16, ONE_F16), NEG_INF_F16)
        # finite - (-inf) = +inf
        self.assertEqual(ref_subrev_f16(NEG_INF_F16, ONE_F16), POS_INF_F16)
        # (+inf) - finite = +inf
        self.assertEqual(ref_subrev_f16(ONE_F16, POS_INF_F16), POS_INF_F16)
        # (-inf) - finite = -inf
        self.assertEqual(ref_subrev_f16(ONE_F16, NEG_INF_F16), NEG_INF_F16)

    def test_nan_handling_and_quieting(self) -> None:
        # Quiet NaN in src0
        res = ref_subrev_f16(CANONICAL_QNAN_F16, ONE_F16)
        self.assertTrue(is_qnan_f16(res))

        # Quiet NaN in src1
        res = ref_subrev_f16(ONE_F16, CANONICAL_QNAN_F16)
        self.assertTrue(is_qnan_f16(res))

        # Signaling NaN is quieted
        res_snan = ref_subrev_f16(SNAN_DEFAULT_F16, ONE_F16)
        self.assertTrue(is_qnan_f16(res_snan))
        self.assertTrue(res_snan & QNAN_BIT_F16)

    def test_subnormal_arithmetic(self) -> None:
        # Subnormal subtraction
        res = ref_subrev_f16(MIN_SUBNORMAL_F16, MAX_SUBNORMAL_F16)
        self.assertEqual(res, MAX_SUBNORMAL_F16 - MIN_SUBNORMAL_F16)
        # Min normal minus smallest subnormal
        res2 = ref_subrev_f16(MIN_SUBNORMAL_F16, MIN_NORMAL_F16)
        self.assertEqual(res2, MAX_SUBNORMAL_F16)

    def test_finite_overflow(self) -> None:
        # 65504.0 - (-65504.0) = +inf
        res = ref_subrev_f16(NEG_MAX_FINITE_F16, MAX_FINITE_F16)
        self.assertEqual(res, POS_INF_F16)
        # -65504.0 - 65504.0 = -inf
        res2 = ref_subrev_f16(MAX_FINITE_F16, NEG_MAX_FINITE_F16)
        self.assertEqual(res2, NEG_INF_F16)

    def test_input_modifiers(self) -> None:
        # abs(s0): S1 - |S0| -> 3.0 - |-1.0| = 2.0
        res = ref_subrev_f16(NEG_ONE_F16, THREE_F16, abs0=True)
        self.assertEqual(bits_to_f16(res), 2.0)

        # neg(s0): S1 - (-S0) -> 3.0 - (-1.0) = 4.0
        res = ref_subrev_f16(ONE_F16, THREE_F16, neg0=True)
        self.assertEqual(bits_to_f16(res), 4.0)

        # abs(s1): |S1| - S0 -> |-3.0| - 1.0 = 2.0
        f_neg3 = f16_to_bits(-3.0)
        res = ref_subrev_f16(ONE_F16, f_neg3, abs1=True)
        self.assertEqual(bits_to_f16(res), 2.0)

        # neg(s1): (-S1) - S0 -> (-3.0) - 1.0 = -4.0
        res = ref_subrev_f16(ONE_F16, THREE_F16, neg1=True)
        self.assertEqual(bits_to_f16(res), -4.0)

        # combined -|x|
        res = ref_subrev_f16(ONE_F16, THREE_F16, abs1=True, neg1=True)
        self.assertEqual(bits_to_f16(res), -4.0)

    def test_output_modifiers(self) -> None:
        # clamp: negative result becomes +0.0
        res_neg = ref_subrev_f16(TWO_F16, ONE_F16, clamp=True)  # 1.0 - 2.0 = -1.0 -> 0.0
        self.assertEqual(res_neg, POS_ZERO_F16)

        # clamp: result > 1.0 becomes 1.0
        res_pos = ref_subrev_f16(ONE_F16, THREE_F16, clamp=True)  # 3.0 - 1.0 = 2.0 -> 1.0
        self.assertEqual(res_pos, ONE_F16)

        # clamp: NaN becomes +0.0
        res_nan = ref_subrev_f16(CANONICAL_QNAN_F16, ONE_F16, clamp=True)
        self.assertEqual(res_nan, POS_ZERO_F16)

        # omod mul:2
        res_m2 = ref_subrev_f16(ONE_F16, TWO_F16, omod="mul:2")  # (2.0 - 1.0) * 2 = 2.0
        self.assertEqual(bits_to_f16(res_m2), 2.0)

        # omod div:2
        res_d2 = ref_subrev_f16(ONE_F16, TWO_F16, omod="div:2")  # (2.0 - 1.0) / 2 = 0.5
        self.assertEqual(bits_to_f16(res_d2), 0.5)


if __name__ == "__main__":
    unittest.main()
