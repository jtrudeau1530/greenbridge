from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_perm_b32.

Tests the vector byte permutation instruction:
    v_perm_b32 D, S0, S1, S2
Lowered semantics:
    Destination byte i (for i in 0..3) is chosen based on selector byte i (S2[8*i+7 : 8*i]):
    - 0..3: selects byte 0..3 of S1 (bits [7:0], [15:8], [23:16], [31:24])
    - 4..7: selects byte 0..3 of S0 (bits [7:0], [15:8], [23:16], [31:24])
    - 8: sign-extension of S1 bit 31 (0xFF if negative, else 0x00)
    - 9: sign-extension of S0 bit 15 (0xFF if negative, else 0x00)
    - 10: sign-extension of S0 bit 31 (0xFF if negative, else 0x00)
    - 11: sign-extension replicate (0xFF if S1 bit 7 else 0x00)
    - 12: constant 0x00
    - >= 13: constant 0xFF

Architectural model:
    32-bit vector and scalar registers (VGPRs/SGPRs) with per-byte lane slicing,
    selector decoding, and bit-exact execution. Standard library only; zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

# ==============================================================================
# Architectural Constants
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
BYTE_MASK: int = 0xFF
BYTE_WIDTH: int = 8

# Selector codes
SEL_SRC1_BYTE0: int = 0
SEL_SRC1_BYTE1: int = 1
SEL_SRC1_BYTE2: int = 2
SEL_SRC1_BYTE3: int = 3

SEL_SRC0_BYTE0: int = 4
SEL_SRC0_BYTE1: int = 5
SEL_SRC0_BYTE2: int = 6
SEL_SRC0_BYTE3: int = 7

SEL_SIGN_EXT_SRC1_B3: int = 8   # S1[31] replicated
SEL_SIGN_EXT_SRC0_B1: int = 9   # S0[15] replicated
SEL_SIGN_EXT_SRC0_B3: int = 10  # S0[31] replicated
SEL_SIGN_EXT_SRC1_B0: int = 11  # S1[7] replicated
SEL_ZERO: int = 12              # Constant 0x00
SEL_ONES_MIN: int = 13          # Constant 0xFF (>= 13)


# ==============================================================================
# Bit Manipulation and Byte Extraction Utilities
# ==============================================================================

def extract_byte(val: int, idx: int) -> int:
    """Extract byte idx (0..3) from an unsigned 32-bit integer."""
    if not 0 <= idx <= 3:
        raise ValueError(f"Byte index out of range [0, 3]: {idx}")
    return (val >> (idx * BYTE_WIDTH)) & BYTE_MASK


def pack_bytes(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack 4 unsigned bytes (b0..b3) into an unsigned 32-bit integer."""
    return ((b0 & BYTE_MASK) |
            ((b1 & BYTE_MASK) << 8) |
            ((b2 & BYTE_MASK) << 16) |
            ((b3 & BYTE_MASK) << 24)) & MASK32


def make_selector(b0: int, b1: int, b2: int, b3: int) -> int:
    """Construct a 32-bit selector word from 4 individual byte selector values."""
    return pack_bytes(b0, b1, b2, b3)


def permute_single_byte(src0: int, src1: int, sel_code: int) -> int:
    """Evaluate a single byte according to RDNA2 v_perm_b32 hardware semantics."""
    sel = sel_code & BYTE_MASK
    if 0 <= sel <= 3:
        return extract_byte(src1, sel)
    elif 4 <= sel <= 7:
        return extract_byte(src0, sel - 4)
    elif sel == SEL_SIGN_EXT_SRC1_B3:
        return 0xFF if bool(src1 & 0x8000_0000) else 0x00
    elif sel == SEL_SIGN_EXT_SRC0_B1:
        return 0xFF if bool(src0 & 0x0000_8000) else 0x00
    elif sel == SEL_SIGN_EXT_SRC0_B3:
        return 0xFF if bool(src0 & 0x8000_0000) else 0x00
    elif sel == SEL_SIGN_EXT_SRC1_B0:
        return 0xFF if bool(src1 & 0x0000_0080) else 0x00
    elif sel == SEL_ZERO:
        return 0x00
    else:  # sel >= SEL_ONES_MIN (13..255)
        return 0xFF


def oracle_v_perm_b32(src0: int, src1: int, selector: int) -> int:
    """Reference mathematical oracle for RDNA2 v_perm_b32.
    
    src0: upper 4 bytes pool (indices 4..7)
    src1: lower 4 bytes pool (indices 0..3)
    selector: 32-bit control word with 4 selector bytes
    """
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    sel = selector & MASK32

    out_b0 = permute_single_byte(s0, s1, extract_byte(sel, 0))
    out_b1 = permute_single_byte(s0, s1, extract_byte(sel, 1))
    out_b2 = permute_single_byte(s0, s1, extract_byte(sel, 2))
    out_b3 = permute_single_byte(s0, s1, extract_byte(sel, 3))

    return pack_bytes(out_b0, out_b1, out_b2, out_b3)


# ==============================================================================
# Architectural Register Models
# ==============================================================================

class VGPRWord:
    """Architectural model of a 32-bit vector register (VGPR)."""

    def __init__(self, value: int = 0) -> None:
        self.raw: int = value & MASK32

    @property
    def u32(self) -> int:
        return self.raw

    @u32.setter
    def u32(self, val: int) -> None:
        self.raw = val & MASK32

    @property
    def i32(self) -> int:
        return struct.unpack("<i", struct.pack("<I", self.raw))[0]

    @property
    def is_negative(self) -> bool:
        return bool(self.raw & 0x8000_0000)

    def get_byte(self, idx: int) -> int:
        return extract_byte(self.raw, idx)

    def set_byte(self, idx: int, b: int) -> None:
        if not 0 <= idx <= 3:
            raise ValueError(f"Byte index out of range: {idx}")
        shift = idx * 8
        mask = ~(0xFF << shift) & MASK32
        self.raw = (self.raw & mask) | ((b & 0xFF) << shift)

    def __repr__(self) -> str:
        return f"VGPRWord(0x{self.raw:08X})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VGPRWord):
            return self.raw == other.raw
        if isinstance(other, int):
            return self.raw == (other & MASK32)
        return NotImplemented


class PermSelectorWord:
    """Architectural model of a 32-bit v_perm_b32 selector word."""

    def __init__(self, raw: int = 0) -> None:
        self.raw: int = raw & MASK32

    @classmethod
    def from_bytes(cls, b0: int, b1: int, b2: int, b3: int) -> PermSelectorWord:
        return cls(pack_bytes(b0, b1, b2, b3))

    @property
    def sel0(self) -> int:
        return extract_byte(self.raw, 0)

    @property
    def sel1(self) -> int:
        return extract_byte(self.raw, 1)

    @property
    def sel2(self) -> int:
        return extract_byte(self.raw, 2)

    @property
    def sel3(self) -> int:
        return extract_byte(self.raw, 3)

    @classmethod
    def identity_src1(cls) -> PermSelectorWord:
        """Picks bytes 0, 1, 2, 3 directly from src1."""
        return cls.from_bytes(0, 1, 2, 3)

    @classmethod
    def identity_src0(cls) -> PermSelectorWord:
        """Picks bytes 0, 1, 2, 3 directly from src0 (selectors 4, 5, 6, 7)."""
        return cls.from_bytes(4, 5, 6, 7)

    @classmethod
    def bswap_src1(cls) -> PermSelectorWord:
        """Reverses byte order of src1."""
        return cls.from_bytes(3, 2, 1, 0)

    @classmethod
    def bswap_src0(cls) -> PermSelectorWord:
        """Reverses byte order of src0."""
        return cls.from_bytes(7, 6, 5, 4)

    @classmethod
    def broadcast_byte(cls, src_byte_idx: int) -> PermSelectorWord:
        """Replicates a single source byte into all 4 destination byte positions."""
        return cls.from_bytes(src_byte_idx, src_byte_idx, src_byte_idx, src_byte_idx)


# ==============================================================================
# Unit Tests: Reference Oracle and Architectural Model
# ==============================================================================

class TestPermB32ReferenceOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural models."""

    def test_identity_src1_selection(self):
        src0 = 0xAABBCCDD
        src1 = 0x11223344
        sel = PermSelectorWord.identity_src1().raw
        self.assertEqual(oracle_v_perm_b32(src0, src1, sel), 0x11223344)

    def test_identity_src0_selection(self):
        src0 = 0xAABBCCDD
        src1 = 0x11223344
        sel = PermSelectorWord.identity_src0().raw
        self.assertEqual(oracle_v_perm_b32(src0, src1, sel), 0xAABBCCDD)

    def test_cross_source_interleaving(self):
        # Pick byte 0 of src1, byte 0 of src0, byte 1 of src1, byte 1 of src0
        # src1: [44, 33, 22, 11] -> byte 0=0x44, byte 1=0x33
        # src0: [DD, CC, BB, AA] -> byte 0=0xDD, byte 1=0xCC
        # Expected result bytes: [44, DD, 33, CC] -> 0xCC33DD44
        src0 = 0xAABBCCDD
        src1 = 0x11223344
        sel = make_selector(0, 4, 1, 5)
        self.assertEqual(oracle_v_perm_b32(src0, src1, sel), 0xCC33DD44)

    def test_endianness_bswap(self):
        src1 = 0x12345678
        src0 = 0x9ABCDEF0
        # Reverse src1: [78, 56, 34, 12] -> [12, 34, 56, 78]
        sel_rev1 = PermSelectorWord.bswap_src1().raw
        self.assertEqual(oracle_v_perm_b32(src0, src1, sel_rev1), 0x78563412)

        # Reverse src0
        sel_rev0 = PermSelectorWord.bswap_src0().raw
        self.assertEqual(oracle_v_perm_b32(src0, src1, sel_rev0), 0xF0DEBC9A)

    def test_byte_broadcast(self):
        src1 = 0x01020304  # byte 0 = 0x04
        sel = PermSelectorWord.broadcast_byte(0).raw
        self.assertEqual(oracle_v_perm_b32(0, src1, sel), 0x04040404)

        src0 = 0x88776655  # byte 7 (idx 3 of src0) = 0x88
        sel7 = PermSelectorWord.broadcast_byte(7).raw
        self.assertEqual(oracle_v_perm_b32(src0, 0, sel7), 0x88888888)

    def test_constant_zero_fill(self):
        src0 = 0xFFFFFFFF
        src1 = 0xFFFFFFFF
        sel = make_selector(SEL_ZERO, SEL_ZERO, SEL_ZERO, SEL_ZERO)
        self.assertEqual(oracle_v_perm_b32(src0, src1, sel), 0x00000000)

        # Mix zero fill with selected bytes
        sel_mix = make_selector(0, SEL_ZERO, 4, SEL_ZERO)
        res = oracle_v_perm_b32(0xAABBCCDD, 0x11223344, sel_mix)
        # byte 0: src1[0] = 0x44, byte 1: 0, byte 2: src0[0] = 0xDD, byte 3: 0
        self.assertEqual(res, 0x00DD0044)

    def test_constant_ones_fill(self):
        src0 = 0x00000000
        src1 = 0x00000000
        # Code 13
        sel13 = make_selector(13, 14, 15, 255)
        self.assertEqual(oracle_v_perm_b32(src0, src1, sel13), 0xFFFFFFFF)

        # Mix ones fill with selected bytes
        sel_mix = make_selector(0, 13, 4, 255)
        res = oracle_v_perm_b32(0x12345678, 0x9ABCDEF0, sel_mix)
        # byte 0: src1[0]=0xF0, byte 1: 0xFF, byte 2: src0[0]=0x78, byte 3: 0xFF
        self.assertEqual(res, 0xFF78FFF0)

    def test_sign_extension_modes(self):
        # SEL_SIGN_EXT_SRC1_B3 (8): sign of S1 bit 31
        s1_pos = 0x7FFF_FFFF  # bit 31 is 0
        s1_neg = 0x8000_0000  # bit 31 is 1
        sel8 = make_selector(8, 8, 8, 8)
        self.assertEqual(oracle_v_perm_b32(0, s1_pos, sel8), 0x00000000)
        self.assertEqual(oracle_v_perm_b32(0, s1_neg, sel8), 0xFFFFFFFF)

        # SEL_SIGN_EXT_SRC0_B1 (9): sign of S0 bit 15
        s0_pos15 = 0x0000_7FFF  # bit 15 is 0
        s0_neg15 = 0x0000_8000  # bit 15 is 1
        sel9 = make_selector(9, 9, 9, 9)
        self.assertEqual(oracle_v_perm_b32(s0_pos15, 0, sel9), 0x00000000)
        self.assertEqual(oracle_v_perm_b32(s0_neg15, 0, sel9), 0xFFFFFFFF)

        # SEL_SIGN_EXT_SRC0_B3 (10): sign of S0 bit 31
        s0_pos31 = 0x7FFF_FFFF
        s0_neg31 = 0x8000_0000
        sel10 = make_selector(10, 10, 10, 10)
        self.assertEqual(oracle_v_perm_b32(s0_pos31, 0, sel10), 0x00000000)
        self.assertEqual(oracle_v_perm_b32(s0_neg31, 0, sel10), 0xFFFFFFFF)

    def test_vgpr_word_model(self):
        vw = VGPRWord(0x12345678)
        self.assertEqual(vw.u32, 0x12345678)
        self.assertEqual(vw.get_byte(0), 0x78)
        self.assertEqual(vw.get_byte(1), 0x56)
        self.assertEqual(vw.get_byte(2), 0x34)
        self.assertEqual(vw.get_byte(3), 0x12)

        vw.set_byte(0, 0xAA)
        self.assertEqual(vw.u32, 0x123456AA)
        vw.set_byte(3, 0xFF)
        self.assertEqual(vw.u32, 0xFF3456AA)
        self.assertTrue(vw.is_negative)

    def test_selector_word_model(self):
        sw = PermSelectorWord.from_bytes(1, 2, 3, 4)
        self.assertEqual(sw.sel0, 1)
        self.assertEqual(sw.sel1, 2)
        self.assertEqual(sw.sel2, 3)
        self.assertEqual(sw.sel3, 4)
        self.assertEqual(sw.raw, 0x04030201)


if __name__ == "__main__":
    unittest.main()
