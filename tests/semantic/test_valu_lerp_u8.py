from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_lerp_u8.

AMD RDNA VALU Instruction Set Architecture:
  v_lerp_u8 dst, src0, src1, src2

Operation:
  Performs parallel byte-wise linear interpolation (rounded unsigned average)
  across four packed 8-bit unsigned byte lanes within 32-bit dwords:
    For each byte lane i in {0, 1, 2, 3}:
      s0_i = (src0 >> (8 * i)) & 0xFF
      s1_i = (src1 >> (8 * i)) & 0xFF
      s2_i = (src2 >> (8 * i)) & 0x01   # only bit 0 of each byte lane contributes
      sum_i = s0_i + s1_i + s2_i        # range: 0 .. 511 (9-bit intermediate)
      dst_i = (sum_i >> 1) & 0xFF       # range: 0 .. 255 (8-bit bounded result)

Mathematical and Architectural Properties:
  - Range isolation: max sum 255 + 255 + 1 = 511; 511 >> 1 = 255 (0xFF).
    Result is strictly bound within [0, 255] for every byte lane. No cross-byte
    carry or overflow can occur.
  - Rounding control:
    - s2 bit 0 = 0: floor / truncation of the unsigned average.
    - s2 bit 0 = 1: ceiling / round-half-up of the unsigned average.
  - Identity property:
    - When s0_i == s1_i == k, ((k + k + r) >> 1) == (2k + r) >> 1 == k for both r in {0, 1}.
  - Symmetry / Commutativity:
    - src0 and src1 are interchangeable: v_lerp_u8(a, b, c) == v_lerp_u8(b, a, c).
  - Upper bit discard:
    - Bits [7:1] of each byte lane in src2 are strictly ignored by hardware and translator.
"""

from typing import List, Optional, Tuple, Union
import unittest

# Architectural bitmasks and constants
MASK8: int = 0xFF
MASK32: int = 0xFFFF_FFFF
BYTE_SHIFTS: Tuple[int, int, int, int] = (0, 8, 16, 24)


def unpack_bytes_u32(val: int) -> Tuple[int, int, int, int]:
    """Unpack a 32-bit unsigned word into 4 byte lanes (b0, b1, b2, b3)."""
    val &= MASK32
    return (
        val & MASK8,
        (val >> 8) & MASK8,
        (val >> 16) & MASK8,
        (val >> 24) & MASK8,
    )


def pack_bytes_u32(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four 8-bit unsigned integers into a 32-bit unsigned word."""
    return (
        (b0 & MASK8)
        | ((b1 & MASK8) << 8)
        | ((b2 & MASK8) << 16)
        | ((b3 & MASK8) << 24)
    ) & MASK32


def ref_lerp_u8_byte(s0: int, s1: int, s2_round: int) -> int:
    """Compute single byte-lane linear interpolation: ((s0 + s1 + (s2 & 1)) >> 1) & 0xFF.

    The intermediate sum ranges from 0 to 511. Right shifting by 1 produces
    a value in [0, 255] with strict lane isolation and no carry propagation.
    """
    b0 = s0 & MASK8
    b1 = s1 & MASK8
    r = s2_round & 0x01
    return ((b0 + b1 + r) >> 1) & MASK8


def oracle_v_lerp_u8(src0: int, src1: int, src2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_lerp_u8 32-bit packed word interpolation."""
    b0_0, b0_1, b0_2, b0_3 = unpack_bytes_u32(src0)
    b1_0, b1_1, b1_2, b1_3 = unpack_bytes_u32(src1)
    b2_0, b2_1, b2_2, b2_3 = unpack_bytes_u32(src2)

    r0 = ref_lerp_u8_byte(b0_0, b1_0, b2_0)
    r1 = ref_lerp_u8_byte(b0_1, b1_1, b2_1)
    r2 = ref_lerp_u8_byte(b0_2, b1_2, b2_2)
    r3 = ref_lerp_u8_byte(b0_3, b1_3, b2_3)

    return pack_bytes_u32(r0, r1, r2, r3)


# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestVLerpU8ModelAndOracle(unittest.TestCase):
    """Unit tests for RDNA2 v_lerp_u8 architectural bit model and reference oracle."""

    def test_byte_unpack_pack_roundtrip(self) -> None:
        patterns = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0xA55A3C7E,
            0x01020304,
            0x80402010,
        ]
        for pat in patterns:
            with self.subTest(pattern=hex(pat)):
                b0, b1, b2, b3 = unpack_bytes_u32(pat)
                repacked = pack_bytes_u32(b0, b1, b2, b3)
                self.assertEqual(repacked, pat)

    def test_reference_known_vectors(self) -> None:
        # 0xFF and 0xFF with round=1 -> 0xFF
        self.assertEqual(oracle_v_lerp_u8(0xFF, 0xFF, 1), 0xFF)
        # 0x01 and 0x02 with round=0 -> 1 ((1+2+0)>>1 == 1)
        self.assertEqual(oracle_v_lerp_u8(0x01, 0x02, 0), 0x01)
        # 0x01 and 0x02 with round=1 -> 2 ((1+2+1)>>1 == 2)
        self.assertEqual(oracle_v_lerp_u8(0x01, 0x02, 1), 0x02)
        # Multi-lane distinct bytes
        s0 = 0x10203040
        s1 = 0x30405060
        # lane 0: (0x40+0x60)>>1 = (64+96)>>1 = 80 = 0x50
        # lane 1: (0x30+0x50)>>1 = (48+80)>>1 = 64 = 0x40
        # lane 2: (0x20+0x40)>>1 = (32+64)>>1 = 48 = 0x30
        # lane 3: (0x10+0x30)>>1 = (16+48)>>1 = 32 = 0x20
        self.assertEqual(oracle_v_lerp_u8(s0, s1, 0), 0x20304050)

    def test_rounding_control(self) -> None:
        # Odd sum: 3 + 4 = 7. Round 0 -> 3; Round 1 -> 4
        s0 = 0x03030303
        s1 = 0x04040404
        self.assertEqual(oracle_v_lerp_u8(s0, s1, 0x00000000), 0x03030303)
        self.assertEqual(oracle_v_lerp_u8(s0, s1, 0x01010101), 0x04040404)

        # Mixed round bits per lane
        s2 = pack_bytes_u32(0, 1, 0, 1)
        expected = pack_bytes_u32(3, 4, 3, 4)
        self.assertEqual(oracle_v_lerp_u8(s0, s1, s2), expected)

    def test_s2_upper_bits_discarded(self) -> None:
        s0 = 0x10203040
        s1 = 0x10203040
        # s2 has bits [7:1] set (0xFE in every lane), but bit 0 is 0
        s2_upper = 0xFEFEFEFE
        self.assertEqual(oracle_v_lerp_u8(s0, s1, s2_upper), s0)

        # s2 has bits [7:0] set (0xFF in every lane), bit 0 is 1
        s2_all = 0xFFFFFFFF
        self.assertEqual(oracle_v_lerp_u8(s0, s1, s2_all), oracle_v_lerp_u8(s0, s1, 0x01010101))

    def test_commutativity_src0_src1(self) -> None:
        pairs = [
            (0x00000000, 0xFFFFFFFF, 0x01010101),
            (0x12345678, 0x9ABCDEF0, 0x00000000),
            (0x55555555, 0xAAAAAAAA, 0xFFFFFFFF),
            (0xFEEDBEEF, 0x01020304, 0x01000100),
        ]
        for s0, s1, s2 in pairs:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                self.assertEqual(oracle_v_lerp_u8(s0, s1, s2), oracle_v_lerp_u8(s1, s0, s2))

    def test_identity_property(self) -> None:
        for val in [0x00000000, 0x7F7F7F7F, 0x80808080, 0xFFFFFFFF, 0x12345678]:
            with self.subTest(val=hex(val)):
                self.assertEqual(oracle_v_lerp_u8(val, val, 0x00000000), val)
                self.assertEqual(oracle_v_lerp_u8(val, val, 0x01010101), val)

    def test_no_cross_lane_bleed(self) -> None:
        # Lane 0 reaches max sum 511 (255 + 255 + 1). Does it bleed into Lane 1?
        s0 = 0x000000FF
        s1 = 0x000000FF
        s2 = 0x00000001
        res = oracle_v_lerp_u8(s0, s1, s2)
        self.assertEqual(res, 0x000000FF)
        b0, b1, b2, b3 = unpack_bytes_u32(res)
        self.assertEqual(b0, 0xFF)
        self.assertEqual(b1, 0x00)
        self.assertEqual(b2, 0x00)
        self.assertEqual(b3, 0x00)


if __name__ == "__main__":
    unittest.main()
