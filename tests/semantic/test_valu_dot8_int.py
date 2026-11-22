from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_dot8_int opcodes.

This module provides an architectural reference model and semantic test suite for RDNA2
VALU packed 4-bit integer dot product instructions:
    - v_dot8_i32_i4 (VOP3/VOP3P): Signed 4-bit integer 8-vector dot product:
        D.i32 = S2.i32 + sum(S0.i4[i] * S1.i4[i] for i in 0..7) mod 2^32
    - v_dot8_u32_u4 (VOP3/VOP3P): Unsigned 4-bit integer 8-vector dot product:
        D.u32 = S2.u32 + sum(S0.u4[i] * S1.u4[i] for i in 0..7) mod 2^32

ISA Specifications (RDNA2 / GFX10.3+):
    - Each 32-bit dword S0 and S1 contains eight packed 4-bit nibbles:
        Nibble 0: bits [3:0]
        Nibble 1: bits [7:4]
        Nibble 2: bits [11:8]
        Nibble 3: bits [15:12]
        Nibble 4: bits [19:16]
        Nibble 5: bits [23:20]
        Nibble 6: bits [27:24]
        Nibble 7: bits [31:28]
    - In v_dot8_i32_i4, each nibble is treated as signed 2's complement int4 in [-8, +7].
      Each component product S0.i4[i] * S1.i4[i] is in [-56, +64].
      The eight products sum to [-448, +512], which is added to 32-bit accumulator S2
      and wrapped modulo 2^32 (or clamped to [-2^31, 2^31 - 1] if clamp is asserted).
    - In v_dot8_u32_u4, each nibble is treated as unsigned uint4 in [0, 15].
      Each component product S0.u4[i] * S1.u4[i] is in [0, 225].
      The eight products sum to [0, 1800], which is added to 32-bit accumulator S2
      and wrapped modulo 2^32 (or clamped to [0, 0xFFFFFFFF] if clamp is asserted).

Standard library Python only; zero third-party dependencies.
Python 3.9+ compatible.
"""

import struct
from typing import Tuple
import unittest

# Bitmask and architectural boundary constants
MASK4: int = 0xF
MASK32: int = 0xFFFF_FFFF
SIGN4_BIT: int = 0x8
INT32_MIN: int = -0x8000_0000
INT32_MAX: int = 0x7FFF_FFFF
UINT32_MAX: int = 0xFFFF_FFFF


def unpack_u4(dword: int) -> Tuple[int, int, int, int, int, int, int, int]:
    """Unpack a 32-bit dword into eight unsigned 4-bit integers (nibble 0 to 7)."""
    val = dword & MASK32
    return (
        val & MASK4,
        (val >> 4) & MASK4,
        (val >> 8) & MASK4,
        (val >> 12) & MASK4,
        (val >> 16) & MASK4,
        (val >> 20) & MASK4,
        (val >> 24) & MASK4,
        (val >> 28) & MASK4,
    )


def pack_u4(n0: int, n1: int, n2: int, n3: int, n4: int, n5: int, n6: int, n7: int) -> int:
    """Pack eight unsigned 4-bit integers into a 32-bit dword."""
    return (
        (n0 & MASK4)
        | ((n1 & MASK4) << 4)
        | ((n2 & MASK4) << 8)
        | ((n3 & MASK4) << 12)
        | ((n4 & MASK4) << 16)
        | ((n5 & MASK4) << 20)
        | ((n6 & MASK4) << 24)
        | ((n7 & MASK4) << 28)
    ) & MASK32


def unpack_i4(dword: int) -> Tuple[int, int, int, int, int, int, int, int]:
    """Unpack a 32-bit dword into eight signed 4-bit integers (-8 to +7)."""
    raw_nibbles = unpack_u4(dword)
    return tuple((n - 16) if (n & SIGN4_BIT) else n for n in raw_nibbles)  # type: ignore[return-value]


def pack_i4(n0: int, n1: int, n2: int, n3: int, n4: int, n5: int, n6: int, n7: int) -> int:
    """Pack eight signed 4-bit integers into a 32-bit dword."""
    return pack_u4(
        n0 & MASK4,
        n1 & MASK4,
        n2 & MASK4,
        n3 & MASK4,
        n4 & MASK4,
        n5 & MASK4,
        n6 & MASK4,
        n7 & MASK4,
    )


def ref_dot8_u32_u4(s0: int, s1: int, s2: int = 0, *, clamp: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 unsigned 4-bit dot product (v_dot8_u32_u4).

    Calculates:
        D = (s2 + sum(s0[i] * s1[i] for i in 0..7))
    Modulo 2^32 without clamp; saturated to 0xFFFFFFFF if clamp is set.
    Returns unsigned 32-bit integer in [0, 0xFFFFFFFF].
    """
    n0 = unpack_u4(s0)
    n1 = unpack_u4(s1)
    prod_sum = sum(x * y for x, y in zip(n0, n1))
    acc = s2 & MASK32
    total = acc + prod_sum

    if clamp:
        return min(total, UINT32_MAX)
    return total & MASK32


def ref_dot8_i32_i4(s0: int, s1: int, s2: int = 0, *, clamp: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 signed 4-bit dot product (v_dot8_i32_i4).

    Calculates:
        D = (s2 + sum(s0.i4[i] * s1.i4[i] for i in 0..7))
    where S2 is interpreted as signed 32-bit integer.
    Modulo 2^32 without clamp; saturated to [-2^31, 2^31 - 1] if clamp is set.
    Returns unsigned 32-bit bit pattern representing the resulting dword.
    """
    n0 = unpack_i4(s0)
    n1 = unpack_i4(s1)
    prod_sum = sum(x * y for x, y in zip(n0, n1))

    # Interpret s2 as signed 32-bit integer
    s2_u32 = s2 & MASK32
    s2_i32 = (s2_u32 - 0x1_0000_0000) if (s2_u32 & 0x8000_0000) else s2_u32

    total_i64 = s2_i32 + prod_sum

    if clamp:
        clamped = max(INT32_MIN, min(total_i64, INT32_MAX))
        return clamped & MASK32
    return total_i64 & MASK32


class TestValuDot8IntMathematicalOracle(unittest.TestCase):
    """Architectural reference mathematical oracle unit tests for v_dot8_int."""

    def test_u4_basic_vectors(self) -> None:
        # Sum 0*1 + 1*2 + 2*3 + 3*4 + 4*5 + 5*6 + 6*7 + 7*8
        # = 0 + 2 + 6 + 12 + 20 + 30 + 42 + 56 = 168
        s0 = pack_u4(0, 1, 2, 3, 4, 5, 6, 7)
        s1 = pack_u4(1, 2, 3, 4, 5, 6, 7, 8)
        self.assertEqual(ref_dot8_u32_u4(s0, s1, 0), 168)
        # With accumulator = 500 -> 668
        self.assertEqual(ref_dot8_u32_u4(s0, s1, 500), 668)

    def test_u4_zero_operands(self) -> None:
        s0 = 0x0000_0000
        s1 = 0x1234_5678
        self.assertEqual(ref_dot8_u32_u4(s0, s1, 0x1234_ABCD), 0x1234_ABCD)
        self.assertEqual(ref_dot8_u32_u4(s1, s0, 0xCAFE_BABE), 0xCAFE_BABE)

    def test_u4_maximum_product(self) -> None:
        # 8 * (15 * 15) = 8 * 225 = 1800
        s0 = 0xFFFF_FFFF
        s1 = 0xFFFF_FFFF
        self.assertEqual(ref_dot8_u32_u4(s0, s1, 0), 1800)
        self.assertEqual(ref_dot8_u32_u4(s0, s1, 200), 2000)

    def test_u4_overflow_wrap_around(self) -> None:
        # Wrap around 2^32: (0xFFFFFFFF - 1800 + 1) + 1800 = 0
        s0 = 0xFFFF_FFFF
        s1 = 0xFFFF_FFFF
        acc = (MASK32 - 1800 + 1) & MASK32
        self.assertEqual(ref_dot8_u32_u4(s0, s1, acc), 0)

        # Overflow by 1
        acc_plus1 = (acc + 1) & MASK32
        self.assertEqual(ref_dot8_u32_u4(s0, s1, acc_plus1), 1)

    def test_u4_clamp_saturation(self) -> None:
        s0 = 0xFFFF_FFFF
        s1 = 0xFFFF_FFFF
        acc = 0xFFFF_FFF0
        # Without clamp, wraps around
        wrapped = ref_dot8_u32_u4(s0, s1, acc, clamp=False)
        self.assertLess(wrapped, 0xFFFF_FFF0)
        # With clamp, saturates to UINT32_MAX
        clamped = ref_dot8_u32_u4(s0, s1, acc, clamp=True)
        self.assertEqual(clamped, UINT32_MAX)

    def test_i4_basic_vectors(self) -> None:
        s0 = pack_i4(1, 2, 3, 4, -1, -2, -3, -4)
        s1 = pack_i4(1, 1, 1, 1, 1, 1, 1, 1)
        # Sum: 1+2+3+4 + (-1)+(-2)+(-3)+(-4) = 0
        self.assertEqual(ref_dot8_i32_i4(s0, s1, 0), 0)
        self.assertEqual(ref_dot8_i32_i4(s0, s1, 77), 77)

    def test_i4_negative_operands(self) -> None:
        # All negative: [-1, -2, -3, -4, -1, -2, -3, -4]
        s0 = pack_i4(-1, -2, -3, -4, -1, -2, -3, -4)
        s1 = pack_i4(2, 2, 2, 2, 2, 2, 2, 2)
        # Sum = 2 * (-20) = -40
        expected = (-40) & MASK32
        self.assertEqual(ref_dot8_i32_i4(s0, s1, 0), expected)

    def test_i4_extremes(self) -> None:
        # Max positive product: 8 * ((-8) * (-8)) = 8 * 64 = +512
        s_min = pack_i4(-8, -8, -8, -8, -8, -8, -8, -8)
        self.assertEqual(ref_dot8_i32_i4(s_min, s_min, 0), 512)

        # Max negative product: 8 * ((-8) * 7) = 8 * (-56) = -448
        s_max = pack_i4(7, 7, 7, 7, 7, 7, 7, 7)
        expected_neg = (-448) & MASK32
        self.assertEqual(ref_dot8_i32_i4(s_min, s_max, 0), expected_neg)

    def test_i4_signed_overflow_and_clamp(self) -> None:
        s_min = pack_i4(-8, -8, -8, -8, -8, -8, -8, -8)
        # Adding +512 to INT32_MAX without clamp wraps to negative
        res_wrap = ref_dot8_i32_i4(s_min, s_min, INT32_MAX, clamp=False)
        self.assertTrue(bool(res_wrap & 0x8000_0000))

        # With clamp, saturates to INT32_MAX
        res_clamp_pos = ref_dot8_i32_i4(s_min, s_min, INT32_MAX, clamp=True)
        self.assertEqual(res_clamp_pos, INT32_MAX & MASK32)

        # Negative saturation: INT32_MIN + (-448) saturates to INT32_MIN
        s_max = pack_i4(7, 7, 7, 7, 7, 7, 7, 7)
        res_clamp_neg = ref_dot8_i32_i4(s_min, s_max, INT32_MIN, clamp=True)
        self.assertEqual(res_clamp_neg, INT32_MIN & MASK32)

    def test_divergence_between_signed_and_unsigned(self) -> None:
        # Nibble 0xF is 15 in unsigned; -1 in signed
        # S0 = [15, 0, ...], S1 = [1, 0, ...]
        s0 = pack_u4(15, 0, 0, 0, 0, 0, 0, 0)
        s1 = pack_u4(1, 0, 0, 0, 0, 0, 0, 0)
        u_res = ref_dot8_u32_u4(s0, s1, 0)
        i_res = ref_dot8_i32_i4(s0, s1, 0)
        self.assertEqual(u_res, 15)
        self.assertEqual(i_res, 0xFFFF_FFFF)  # -1 in 32-bit dword
        self.assertNotEqual(u_res, i_res)


if __name__ == "__main__":
    unittest.main()
