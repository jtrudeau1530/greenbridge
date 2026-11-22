"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_mul_f64.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  Operations on 64-bit double-precision floating-point quantities (binary64, IEEE 754)
  operate across aligned pairs of adjacent registers: v[d:d+1] from source operands
  v[s0:s0+1] and v[s1:s1+1] (or scalar SGPR pairs s[k:k+1], literal 64-bit constants,
  or inline constants).

  Instruction:
    v_mul_f64 v[d:d+1], src0, src1 [clamp] [omod]

  IEEE-754 binary64 encoding:
    - 1 sign bit (bit 63)
    - 11 exponent bits with bias 1023 (bits 62:52)
    - 52 mantissa / fraction bits (bits 51:0)

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

import math
import struct
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# IEEE-754 binary64 bit constants
F64_POS_ZERO = 0x0000000000000000
F64_NEG_ZERO = 0x8000000000000000
F64_POS_INF = 0x7FF0000000000000
F64_NEG_INF = 0xFFF0000000000000
F64_CANONICAL_QNAN = 0x7FF8000000000000
F64_QUIET_BIT = 0x0008000000000000

F64_EXP_MASK = 0x7FF0000000000000
F64_MANT_MASK = 0x000FFFFFFFFFFFFF
F64_SIGN_MASK = 0x8000000000000000

F64_ONE = 0x3FF0000000000000          # 1.0
F64_NEG_ONE = 0xBFF0000000000000      # -1.0
F64_TWO = 0x4000000000000000          # 2.0
F64_HALF = 0x3FE0000000000000         # 0.5
F64_MIN_NORMAL = 0x0010000000000000   # 2^-1022
F64_MAX_NORMAL = 0x7FEFFFFFFFFFFFFF   # (2 - 2^-52) * 2^1023
F64_MIN_SUBNORMAL = 0x0000000000000001  # 2^-1074
F64_MAX_SUBNORMAL = 0x000FFFFFFFFFFFFF

OLD_VGPR_SENTINEL = 0xDEADBEEF


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


def ref_mul_f64(
    src0_bits: int,
    src1_bits: int,
    *,
    clamp: bool = False,
    src0_neg: bool = False,
    src0_abs: bool = False,
    src1_neg: bool = False,
    src1_abs: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 double-precision multiplication v_mul_f64.

    Evaluates:
      D.f64 = src0 * src1
    accounting for IEEE-754 binary64 rules, input modifiers (neg, abs), and clamp [0.0, 1.0].
    """
    s0 = apply_input_modifiers_f64(src0_bits, neg=src0_neg, abs_val=src0_abs)
    s1 = apply_input_modifiers_f64(src1_bits, neg=src1_neg, abs_val=src1_abs)

    sign_res = f64_sign(s0) ^ f64_sign(s1)

    # NaN propagation
    if is_f64_nan(s0):
        res = quiet_f64(s0)
    elif is_f64_nan(s1):
        res = quiet_f64(s1)
    # 0 * Inf -> Canonical QNaN
    elif (is_f64_zero(s0) and is_f64_inf(s1)) or (is_f64_inf(s0) and is_f64_zero(s1)):
        res = F64_CANONICAL_QNAN
    # Inf * anything non-zero -> Inf with expected sign
    elif is_f64_inf(s0) or is_f64_inf(s1):
        res = F64_NEG_INF if sign_res else F64_POS_INF
    # Zero * anything finite -> Zero with expected sign
    elif is_f64_zero(s0) or is_f64_zero(s1):
        res = F64_NEG_ZERO if sign_res else F64_POS_ZERO
    else:
        # Standard finite multiplication
        val0 = u64_to_f64(s0)
        val1 = u64_to_f64(s1)
        prod = val0 * val1
        res = f64_to_u64(prod)
        if is_f64_zero(res):
            res = F64_NEG_ZERO if sign_res else F64_POS_ZERO
        elif is_f64_inf(res):
            res = F64_NEG_INF if sign_res else F64_POS_INF

    # Clamp modifier: clamps result to [0.0, 1.0].
    # In RDNA2 VALU: if result is NaN, clamp produces +0.0.
    if clamp:
        if is_f64_nan(res):
            res = F64_POS_ZERO
        else:
            val = u64_to_f64(res)
            if val < 0.0 or is_f64_zero(res):
                res = F64_POS_ZERO
            elif val > 1.0:
                res = F64_ONE

    return res & MASK64


def ref_mul_f64_dwords(
    s0_lo: int,
    s0_hi: int,
    s1_lo: int,
    s1_hi: int,
    *,
    clamp: bool = False,
    src0_neg: bool = False,
    src0_abs: bool = False,
    src1_neg: bool = False,
    src1_abs: bool = False,
) -> tuple[int, int]:
    """Decomposed dword interface for v_mul_f64."""
    s0 = join_dwords(s0_lo, s0_hi)
    s1 = join_dwords(s1_lo, s1_hi)
    res = ref_mul_f64(
        s0,
        s1,
        clamp=clamp,
        src0_neg=src0_neg,
        src0_abs=src0_abs,
        src1_neg=src1_neg,
        src1_abs=src1_abs,
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


class TestVMulF64ReferenceOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_basic_multiplication(self) -> None:
        # 2.0 * 3.5 = 7.0
        r = ref_mul_f64(f64_to_u64(2.0), f64_to_u64(3.5))
        self.assertEqual(u64_to_f64(r), 7.0)

        # -1.5 * 4.0 = -6.0
        r = ref_mul_f64(f64_to_u64(-1.5), f64_to_u64(4.0))
        self.assertEqual(u64_to_f64(r), -6.0)

        # -2.5 * -8.0 = 20.0
        r = ref_mul_f64(f64_to_u64(-2.5), f64_to_u64(-8.0))
        self.assertEqual(u64_to_f64(r), 20.0)

    def test_zero_rules(self) -> None:
        # +0.0 * +0.0 = +0.0
        self.assertEqual(ref_mul_f64(F64_POS_ZERO, F64_POS_ZERO), F64_POS_ZERO)
        # +0.0 * -0.0 = -0.0
        self.assertEqual(ref_mul_f64(F64_POS_ZERO, F64_NEG_ZERO), F64_NEG_ZERO)
        # -0.0 * -0.0 = +0.0
        self.assertEqual(ref_mul_f64(F64_NEG_ZERO, F64_NEG_ZERO), F64_POS_ZERO)
        # +0.0 * 5.0 = +0.0
        self.assertEqual(ref_mul_f64(F64_POS_ZERO, f64_to_u64(5.0)), F64_POS_ZERO)
        # -0.0 * 5.0 = -0.0
        self.assertEqual(ref_mul_f64(F64_NEG_ZERO, f64_to_u64(5.0)), F64_NEG_ZERO)

    def test_inf_and_zero_rules(self) -> None:
        # 0 * Inf -> Canonical QNaN
        self.assertEqual(ref_mul_f64(F64_POS_ZERO, F64_POS_INF), F64_CANONICAL_QNAN)
        self.assertEqual(ref_mul_f64(F64_NEG_ZERO, F64_POS_INF), F64_CANONICAL_QNAN)
        self.assertEqual(ref_mul_f64(F64_POS_INF, F64_POS_ZERO), F64_CANONICAL_QNAN)

        # Inf * Inf -> Inf with XOR sign
        self.assertEqual(ref_mul_f64(F64_POS_INF, F64_POS_INF), F64_POS_INF)
        self.assertEqual(ref_mul_f64(F64_POS_INF, F64_NEG_INF), F64_NEG_INF)
        self.assertEqual(ref_mul_f64(F64_NEG_INF, F64_NEG_INF), F64_POS_INF)

        # Inf * finite non-zero -> Inf with XOR sign
        self.assertEqual(ref_mul_f64(F64_POS_INF, f64_to_u64(2.0)), F64_POS_INF)
        self.assertEqual(ref_mul_f64(F64_POS_INF, f64_to_u64(-2.0)), F64_NEG_INF)

    def test_input_modifiers(self) -> None:
        # neg: 2.0 * neg(3.0) = -6.0
        r = ref_mul_f64(f64_to_u64(2.0), f64_to_u64(3.0), src1_neg=True)
        self.assertEqual(u64_to_f64(r), -6.0)

        # abs: -2.0 * abs(-3.0) = -6.0
        r = ref_mul_f64(f64_to_u64(-2.0), f64_to_u64(-3.0), src1_abs=True)
        self.assertEqual(u64_to_f64(r), -6.0)

        # abs + neg on src0: neg(abs(-2.0)) = -2.0; * 3.0 = -6.0
        r = ref_mul_f64(f64_to_u64(-2.0), f64_to_u64(3.0), src0_abs=True, src0_neg=True)
        self.assertEqual(u64_to_f64(r), -6.0)

    def test_clamp_modifier(self) -> None:
        # In range [0.0, 1.0]: 0.5 * 1.5 = 0.75
        r = ref_mul_f64(f64_to_u64(0.5), f64_to_u64(1.5), clamp=True)
        self.assertEqual(u64_to_f64(r), 0.75)

        # > 1.0 clamped to 1.0: 2.0 * 3.0 = 6.0 -> 1.0
        r = ref_mul_f64(f64_to_u64(2.0), f64_to_u64(3.0), clamp=True)
        self.assertEqual(r, F64_ONE)

        # < 0.0 clamped to +0.0: -2.0 * 3.0 = -6.0 -> +0.0
        r = ref_mul_f64(f64_to_u64(-2.0), f64_to_u64(3.0), clamp=True)
        self.assertEqual(r, F64_POS_ZERO)

        # NaN clamped to +0.0
        r = ref_mul_f64(F64_POS_ZERO, F64_POS_INF, clamp=True)
        self.assertEqual(r, F64_POS_ZERO)

    def test_dword_decomposition(self) -> None:
        s0 = f64_to_u64(12.345)
        s1 = f64_to_u64(67.89)
        s0_lo, s0_hi = split_dwords(s0)
        s1_lo, s1_hi = split_dwords(s1)

        lo, hi = ref_mul_f64_dwords(s0_lo, s0_hi, s1_lo, s1_hi)
        res64 = join_dwords(lo, hi)
        expected = ref_mul_f64(s0, s1)
        self.assertEqual(res64, expected)

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
