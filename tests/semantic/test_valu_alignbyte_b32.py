"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_alignbyte_b32.

Models AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics
for the byte alignment / funnel shift instruction:
  v_alignbyte_b32 dst, src0, src1, src2

RDNA2 VALU Instruction Specification:
  v_alignbyte_b32 D.u, S0.u, S1.u, S2.u
  Operands:
    - D:  32-bit VGPR destination
    - S0: 32-bit register (VGPR/SGPR) or immediate, providing high 32 bits (bytes 4..7)
    - S1: 32-bit register (VGPR/SGPR) or immediate, providing low 32 bits (bytes 0..3)
    - S2: 32-bit register (VGPR/SGPR) or inline constant shift amount (in bytes)

  Mathematical Operation:
    concat = ((S0 & 0xFFFFFFFF) << 32) | (S1 & 0xFFFFFFFF)
    shift_bytes = S2 & 3
    D = (concat >> (shift_bytes * 8)) & 0xFFFFFFFF

  Byte Selection Mapping:
    Let S1 = [B0, B1, B2, B3] and S0 = [B4, B5, B6, B7] (in little-endian byte order).
    - shift_bytes == 0: D = [B0, B1, B2, B3] = S1
    - shift_bytes == 1: D = [B1, B2, B3, B4] = (S1 >> 8)  | ((S0 & 0xFF) << 24)
    - shift_bytes == 2: D = [B2, B3, B4, B5] = (S1 >> 16) | ((S0 & 0xFFFF) << 16)
    - shift_bytes == 3: D = [B3, B4, B5, B6] = (S1 >> 24) | ((S0 & 0xFFFFFF) << 8)

  Key Architectural Properties:
    1. Only the low 2 bits of S2 (S2 & 3) determine the byte shift count.
       Shift counts >= 4, negative numbers, and high bits are strictly masked modulo 4.
    2. Byte B7 (the highest byte of S0) is never part of the destination for any valid shift.
    3. Common compiler idioms:
       - S0 == S1: 32-bit byte-rotate right: rotr32(S1, (S2 & 3) * 8).
       - S0 == 0:  logical shift right of S1 by (S2 & 3) * 8 bits.
       - S1 == 0:  logical shift left of S0 low bytes to high positions.
    4. Bitwise integrity: All bit patterns within bytes are preserved exactly with zero mutation.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import os
import struct
import sys
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT32: int = 0x80000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def get_byte(val32: int, byte_idx: int) -> int:
    """Extract byte at index byte_idx (0..3) from 32-bit word."""
    return (val32 >> ((byte_idx & 3) * 8)) & 0xFF


def set_byte(val32: int, byte_idx: int, byte_val: int) -> int:
    """Insert byte_val at byte_idx (0..3) in 32-bit word."""
    shift = (byte_idx & 3) * 8
    mask = ~(0xFF << shift) & MASK32
    return (val32 & mask) | ((byte_val & 0xFF) << shift)


def pack_bytes(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four 8-bit bytes into a 32-bit unsigned little-endian integer."""
    return ((b3 & 0xFF) << 24) | ((b2 & 0xFF) << 16) | ((b1 & 0xFF) << 8) | (b0 & 0xFF)


def unpack_bytes(val32: int) -> tuple[int, int, int, int]:
    """Unpack a 32-bit unsigned integer into four 8-bit bytes (b0, b1, b2, b3)."""
    val = val32 & MASK32
    return val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF


def ref_alignbyte_b32(src0: int, src1: int, src2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_alignbyte_b32.

    Concatenates {src0, src1} into a 64-bit word with src0 as the high dword and
    src1 as the low dword, then right shifts logically by (src2 & 3) bytes (0, 8, 16, or 24 bits)
    and extracts the low 32 bits.
    """
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    shift_bytes = src2 & 3
    concat = (s0 << 32) | s1
    res = (concat >> (shift_bytes * 8)) & MASK32
    return res


def ref_alignbyte_by_bytes(src0: int, src1: int, shift: int) -> int:
    """Alternative architectural byte-indexed oracle for verification and cross-checking."""
    b_lo = unpack_bytes(src1)
    b_hi = unpack_bytes(src0)
    eight_bytes = b_lo + b_hi  # 8 bytes: [0..7]
    s = shift & 3
    return pack_bytes(
        eight_bytes[s],
        eight_bytes[s + 1],
        eight_bytes[s + 2],
        eight_bytes[s + 3],
    )


class TestValuAlignbyteB32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_shift_zero_returns_src1(self) -> None:
        src0 = 0xDEADBEEF
        src1 = 0xCAFEBABE
        self.assertEqual(ref_alignbyte_b32(src0, src1, 0), src1)
        self.assertEqual(ref_alignbyte_by_bytes(src0, src1, 0), src1)

    def test_shift_one_byte(self) -> None:
        # S1: bytes [0x11, 0x22, 0x33, 0x44] -> 0x44332211
        # S0: bytes [0x55, 0x66, 0x77, 0x88] -> 0x88776655
        # Expected: bytes [0x22, 0x33, 0x44, 0x55] -> 0x55443322
        src1 = 0x44332211
        src0 = 0x88776655
        expected = 0x55443322
        self.assertEqual(ref_alignbyte_b32(src0, src1, 1), expected)
        self.assertEqual(ref_alignbyte_by_bytes(src0, src1, 1), expected)

    def test_shift_two_bytes(self) -> None:
        src1 = 0x44332211
        src0 = 0x88776655
        expected = 0x66554433
        self.assertEqual(ref_alignbyte_b32(src0, src1, 2), expected)
        self.assertEqual(ref_alignbyte_by_bytes(src0, src1, 2), expected)

    def test_shift_three_bytes(self) -> None:
        src1 = 0x44332211
        src0 = 0x88776655
        expected = 0x77665544
        self.assertEqual(ref_alignbyte_b32(src0, src1, 3), expected)
        self.assertEqual(ref_alignbyte_by_bytes(src0, src1, 3), expected)

    def test_shift_masking_modulo_four(self) -> None:
        src1 = 0x12345678
        src0 = 0x9ABCDEF0
        # src2 values with same low 2 bits must yield identical results
        for base_shift in (0, 1, 2, 3):
            ref = ref_alignbyte_b32(src0, src1, base_shift)
            for mult in (4, 8, 12, 16, 32, 64, 100, 256, 1024, 0xFFFFFFFC):
                test_shift = mult + base_shift
                self.assertEqual(ref_alignbyte_b32(src0, src1, test_shift), ref)

    def test_negative_shift_masking(self) -> None:
        src1 = 0xAABBCCDD
        src0 = 0x11223344
        # -1 in two's complement has low 2 bits 0b11 == 3
        self.assertEqual(ref_alignbyte_b32(src0, src1, -1), ref_alignbyte_b32(src0, src1, 3))
        # -2 has low 2 bits 0b10 == 2
        self.assertEqual(ref_alignbyte_b32(src0, src1, -2), ref_alignbyte_b32(src0, src1, 2))
        # -3 has low 2 bits 0b01 == 1
        self.assertEqual(ref_alignbyte_b32(src0, src1, -3), ref_alignbyte_b32(src0, src1, 1))
        # -4 has low 2 bits 0b00 == 0
        self.assertEqual(ref_alignbyte_b32(src0, src1, -4), ref_alignbyte_b32(src0, src1, 0))

    def test_rotate_idiom_src0_equals_src1(self) -> None:
        # When src0 == src1, alignbyte is 32-bit rotate right by (shift & 3) bytes
        val = 0x12345678
        # shift 0: 0x12345678
        self.assertEqual(ref_alignbyte_b32(val, val, 0), val)
        # shift 1: byte 0x78 moves to high byte -> 0x78123456
        self.assertEqual(ref_alignbyte_b32(val, val, 1), 0x78123456)
        # shift 2: bytes 0x5678 move to high -> 0x56781234
        self.assertEqual(ref_alignbyte_b32(val, val, 2), 0x56781234)
        # shift 3: bytes 0x345678 move to high -> 0x34567812
        self.assertEqual(ref_alignbyte_b32(val, val, 3), 0x34567812)

    def test_zero_src0_logical_shift_right(self) -> None:
        val = 0x89ABCDEF
        self.assertEqual(ref_alignbyte_b32(0, val, 0), val)
        self.assertEqual(ref_alignbyte_b32(0, val, 1), val >> 8)
        self.assertEqual(ref_alignbyte_b32(0, val, 2), val >> 16)
        self.assertEqual(ref_alignbyte_b32(0, val, 3), val >> 24)

    def test_zero_src1_logical_shift_left(self) -> None:
        val = 0x89ABCDEF
        self.assertEqual(ref_alignbyte_b32(val, 0, 0), 0)
        self.assertEqual(ref_alignbyte_b32(val, 0, 1), (val & 0xFF) << 24)
        self.assertEqual(ref_alignbyte_b32(val, 0, 2), (val & 0xFFFF) << 16)
        self.assertEqual(ref_alignbyte_b32(val, 0, 3), (val & 0xFFFFFF) << 8)

    def test_highest_byte_of_src0_never_included(self) -> None:
        # Highest byte of S0 is 0xFF, rest zeros
        src0 = 0xFF000000
        src1 = 0x00000000
        for s in range(8):
            # 0xFF should NEVER appear in result
            res = ref_alignbyte_b32(src0, src1, s)
            self.assertEqual(res, 0)

    def test_pack_unpack_bytes_roundtrip(self) -> None:
        val = 0xDEADBEEF
        b0, b1, b2, b3 = unpack_bytes(val)
        self.assertEqual(b0, 0xEF)
        self.assertEqual(b1, 0xBE)
        self.assertEqual(b2, 0xAD)
        self.assertEqual(b3, 0xDE)
        self.assertEqual(pack_bytes(b0, b1, b2, b3), val)


if __name__ == "__main__":
    unittest.main()
