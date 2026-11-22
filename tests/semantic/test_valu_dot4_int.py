from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_dot4_int opcodes.

This module provides an architectural reference model and semantic test suite for RDNA2
VALU packed 8-bit integer dot product instructions:
    - v_dot4_i32_i8 (VOP3/VOP3P): Signed 8-bit integer 4-vector dot product:
        D.i32 = S2.i32 + sum(S0.i8[i] * S1.i8[i] for i in 0..3) mod 2^32
    - v_dot4_u32_u8 (VOP3/VOP3P): Unsigned 8-bit integer 4-vector dot product:
        D.u32 = S2.u32 + sum(S0.u8[i] * S1.u8[i] for i in 0..3) mod 2^32
    - v_dot4acc_u32_u8: Accumulating unsigned 8-bit integer dot product with explicit accumulator:
        D.u32 = S2.u32 + sum(S0.u8[i] * S1.u8[i] for i in 0..3) mod 2^32

ISA Specifications (RDNA2 / GFX10.3+):
    - Each 32-bit dword S0 and S1 contains four packed 8-bit integers:
        Byte 0: bits [7:0]
        Byte 1: bits [15:8]
        Byte 2: bits [23:16]
        Byte 3: bits [31:24]
    - In v_dot4_i32_i8, each byte is treated as signed 2's complement int8 in [-128, +127].
      Each component product S0.i8[i] * S1.i8[i] is in [-16256, +16384].
      The four products sum to [-65024, +65536], which is added to 32-bit accumulator S2
      and wrapped modulo 2^32 (or clamped to [-2^31, 2^31 - 1] if clamp is asserted).
    - In v_dot4_u32_u8, each byte is treated as unsigned uint8 in [0, 255].
      Each component product S0.u8[i] * S1.u8[i] is in [0, 65025].
      The four products sum to [0, 260100], which is added to 32-bit accumulator S2
      and wrapped modulo 2^32 (or clamped to [0, 0xFFFFFFFF] if clamp is asserted).

Standard library Python only; zero third-party dependencies.
Python 3.9+ compatible.
"""

import struct
import unittest

# Bitmask and architectural boundary constants
MASK8: int = 0xFF
MASK32: int = 0xFFFF_FFFF
SIGN8_BIT: int = 0x80
INT32_MIN: int = -0x8000_0000
INT32_MAX: int = 0x7FFF_FFFF
UINT32_MAX: int = 0xFFFF_FFFF


def unpack_u8(dword: int) -> tuple[int, int, int, int]:
    """Unpack a 32-bit dword into four unsigned 8-bit integers (byte0, byte1, byte2, byte3)."""
    val = dword & MASK32
    return (
        val & MASK8,
        (val >> 8) & MASK8,
        (val >> 16) & MASK8,
        (val >> 24) & MASK8,
    )


def pack_u8(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four unsigned 8-bit integers into a 32-bit dword."""
    return ((b0 & MASK8) | ((b1 & MASK8) << 8) | ((b2 & MASK8) << 16) | ((b3 & MASK8) << 24)) & MASK32


def unpack_i8(dword: int) -> tuple[int, int, int, int]:
    """Unpack a 32-bit dword into four signed 8-bit integers (-128 to +127)."""
    raw_bytes = unpack_u8(dword)
    return tuple((b - 256) if (b & SIGN8_BIT) else b for b in raw_bytes)  # type: ignore[return-value]


def pack_i8(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four signed 8-bit integers into a 32-bit dword."""
    return pack_u8(b0 & MASK8, b1 & MASK8, b2 & MASK8, b3 & MASK8)


def ref_dot4_u32_u8(s0: int, s1: int, s2: int = 0, *, clamp: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 unsigned 8-bit dot product (v_dot4_u32_u8).

    Calculates:
        D = (s2 + sum(s0[i] * s1[i] for i in 0..3))
    Modulo 2^32 without clamp; saturated to 0xFFFFFFFF if clamp is set.
    Returns unsigned 32-bit integer in [0, 0xFFFFFFFF].
    """
    b0 = unpack_u8(s0)
    b1 = unpack_u8(s1)
    prod_sum = sum(x * y for x, y in zip(b0, b1))
    acc = s2 & MASK32
    total = acc + prod_sum

    if clamp:
        return min(total, UINT32_MAX)
    return total & MASK32


def ref_dot4acc_u32_u8(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    """Reference mathematical oracle for v_dot4acc_u32_u8 accumulating form."""
    return ref_dot4_u32_u8(s0, s1, s2, clamp=clamp)


def ref_dot4_i32_i8(s0: int, s1: int, s2: int = 0, *, clamp: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 signed 8-bit dot product (v_dot4_i32_i8).

    Calculates:
        D = (s2 + sum(s0.i8[i] * s1.i8[i] for i in 0..3))
    where S2 is interpreted as signed 32-bit integer.
    Modulo 2^32 without clamp; saturated to [-2^31, 2^31 - 1] if clamp is set.
    Returns unsigned 32-bit bit pattern representing the resulting dword.
    """
    b0 = unpack_i8(s0)
    b1 = unpack_i8(s1)
    prod_sum = sum(x * y for x, y in zip(b0, b1))

    # Interpret s2 as signed 32-bit integer
    s2_u32 = s2 & MASK32
    s2_i32 = (s2_u32 - 0x1_0000_0000) if (s2_u32 & 0x8000_0000) else s2_u32

    total_i64 = s2_i32 + prod_sum

    if clamp:
        clamped = max(INT32_MIN, min(total_i64, INT32_MAX))
        return clamped & MASK32
    return total_i64 & MASK32


class TestValuDot4IntMathematicalOracle(unittest.TestCase):
    """Architectural reference mathematical oracle unit tests for v_dot4_int."""

    def test_u8_basic_vectors(self) -> None:
        # 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        s0 = pack_u8(1, 2, 3, 4)
        s1 = pack_u8(5, 6, 7, 8)
        self.assertEqual(ref_dot4_u32_u8(s0, s1, 0), 70)
        # With accumulator = 1000 -> 1070
        self.assertEqual(ref_dot4_u32_u8(s0, s1, 1000), 1070)
        self.assertEqual(ref_dot4acc_u32_u8(s0, s1, 1000), 1070)

    def test_u8_zero_operands(self) -> None:
        s0 = 0x0000_0000
        s1 = 0x1234_5678
        self.assertEqual(ref_dot4_u32_u8(s0, s1, 0xDEAD_BEEF), 0xDEAD_BEEF)
        self.assertEqual(ref_dot4_u32_u8(s1, s0, 0xCAFE_BABE), 0xCAFE_BABE)

    def test_u8_maximum_product(self) -> None:
        # 4 * (255 * 255) = 4 * 65025 = 260100 (0x3F804)
        s0 = 0xFFFF_FFFF
        s1 = 0xFFFF_FFFF
        self.assertEqual(ref_dot4_u32_u8(s0, s1, 0), 260100)
        self.assertEqual(ref_dot4_u32_u8(s0, s1, 100), 260200)

    def test_u8_overflow_wrap_around(self) -> None:
        # Wrap around 2^32: (0xFFFFFFFF - 260100 + 1) + 260100 = 0
        s0 = 0xFFFF_FFFF
        s1 = 0xFFFF_FFFF
        acc = (MASK32 - 260100 + 1) & MASK32
        self.assertEqual(ref_dot4_u32_u8(s0, s1, acc), 0)

        # Overflow by 1
        acc_plus1 = (acc + 1) & MASK32
        self.assertEqual(ref_dot4_u32_u8(s0, s1, acc_plus1), 1)

    def test_u8_clamp_saturation(self) -> None:
        s0 = 0xFFFF_FFFF
        s1 = 0xFFFF_FFFF
        acc = 0xFFFF_0000
        # Without clamp, wraps around
        wrapped = ref_dot4_u32_u8(s0, s1, acc, clamp=False)
        self.assertLess(wrapped, 0xFFFF_0000)
        # With clamp, saturates to UINT32_MAX
        clamped = ref_dot4_u32_u8(s0, s1, acc, clamp=True)
        self.assertEqual(clamped, UINT32_MAX)

    def test_i8_basic_vectors(self) -> None:
        # 1*5 + 2*6 + 3*7 + 4*8 = 70
        s0 = pack_i8(1, 2, 3, 4)
        s1 = pack_i8(5, 6, 7, 8)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 70)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 500), 570)

    def test_i8_negative_operands(self) -> None:
        # (-1)*5 + 2*(-6) + (-3)*(-7) + (-4)*8 = -5 + -12 + 21 + -32 = -28
        s0 = pack_i8(-1, 2, -3, -4)
        s1 = pack_i8(5, -6, -7, 8)
        expected = (-28) & MASK32
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), expected)

    def test_i8_extremes(self) -> None:
        # Max positive product: 4 * ((-128) * (-128)) = 4 * 16384 = 65536
        s_min = pack_i8(-128, -128, -128, -128)
        self.assertEqual(ref_dot4_i32_i8(s_min, s_min, 0), 65536)

        # Max negative product: 4 * ((-128) * 127) = 4 * (-16256) = -65024
        s_max = pack_i8(127, 127, 127, 127)
        expected_neg = (-65024) & MASK32
        self.assertEqual(ref_dot4_i32_i8(s_min, s_max, 0), expected_neg)

    def test_i8_signed_overflow_and_clamp(self) -> None:
        s_min = pack_i8(-128, -128, -128, -128)
        # Adding to INT32_MAX without clamp wraps to negative
        res_wrap = ref_dot4_i32_i8(s_min, s_min, INT32_MAX, clamp=False)
        self.assertTrue(bool(res_wrap & 0x8000_0000))

        # With clamp, saturates to INT32_MAX
        res_clamp_pos = ref_dot4_i32_i8(s_min, s_min, INT32_MAX, clamp=True)
        self.assertEqual(res_clamp_pos, INT32_MAX & MASK32)

        # Negative saturation: INT32_MIN + (-65024) saturates to INT32_MIN
        s_max = pack_i8(127, 127, 127, 127)
        res_clamp_neg = ref_dot4_i32_i8(s_min, s_max, INT32_MIN, clamp=True)
        self.assertEqual(res_clamp_neg, INT32_MIN & MASK32)

    def test_divergence_between_signed_and_unsigned(self) -> None:
        # 0xFF as unsigned is 255; as signed is -1
        # S0 = [255, 0, 0, 0], S1 = [1, 0, 0, 0]
        # Unsigned: 255 * 1 = 255
        # Signed: (-1) * 1 = -1 (0xFFFFFFFF)
        s0 = pack_u8(255, 0, 0, 0)
        s1 = pack_u8(1, 0, 0, 0)
        u_res = ref_dot4_u32_u8(s0, s1, 0)
        i_res = ref_dot4_i32_i8(s0, s1, 0)
        self.assertEqual(u_res, 255)
        self.assertEqual(i_res, 0xFFFF_FFFF)
        self.assertNotEqual(u_res, i_res)


if __name__ == "__main__":
    unittest.main()
