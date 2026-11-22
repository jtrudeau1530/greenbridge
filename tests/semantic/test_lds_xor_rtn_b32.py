from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_xor_rtn_b32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 32-bit Bitwise XOR with Return:
Performs an atomic bitwise XOR on 32-bit LDS memory and returns the original value.

Syntax:
    ds_xor_rtn_b32 vdst, vaddr, vdata0 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    LDS[dword_index] = (old_val ^ vdata0) & 0xFFFFFFFF
    vdst = old_val
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK_32: int = 0xFFFF_FFFF          # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK_32
    return val - (1 << 32) if (val & 0x8000_0000) else val


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0..31) for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_xor_rtn_b32(current_mem: int, data: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_xor_rtn_b32.

    Returns:
        (old_val, new_val)
    """
    old_val = u32(current_mem)
    data_masked = u32(data)
    new_val = old_val ^ data_masked
    return old_val, new_val


class LDSModel:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        if capacity_dwords <= 0 or (capacity_dwords & (capacity_dwords - 1)) != 0:
            raise ValueError(f"capacity_dwords must be a power of two, got {capacity_dwords}")
        self.capacity_dwords = capacity_dwords
        self.memory = [0] * capacity_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        self.memory[idx] = u32(val)

    def atomic_xor_b32(self, vaddr: int, data: int, offset: int = 0) -> Tuple[int, int]:
        """Execute atomic XOR with return on LDS dword.

        Returns (old_val, new_val).
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        curr = self.memory[idx]
        old_val, new_val = oracle_ds_xor_rtn_b32(curr, data)
        self.memory[idx] = new_val
        return old_val, new_val


class TestLdsXorRtnB32ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and LDS architectural model."""

    def setUp(self) -> None:
        self.lds = LDSModel()

    def test_xor_identity_zero(self) -> None:
        """XOR with zero preserves the original value and returns it."""
        addr = 0x40
        self.lds.write_dword(addr, 0x1234_5678)
        old_val, new_val = self.lds.atomic_xor_b32(addr, 0x0000_0000)
        self.assertEqual(old_val, 0x1234_5678)
        self.assertEqual(new_val, 0x1234_5678)
        self.assertEqual(self.lds.read_dword(addr), 0x1234_5678)

    def test_xor_all_ones_inversion(self) -> None:
        """XOR with all-ones inverts all 32 bits."""
        addr = 0x80
        self.lds.write_dword(addr, 0xAAAA_5555)
        old_val, new_val = self.lds.atomic_xor_b32(addr, 0xFFFF_FFFF)
        self.assertEqual(old_val, 0xAAAA_5555)
        self.assertEqual(new_val, 0x5555_AAAA)
        self.assertEqual(self.lds.read_dword(addr), 0x5555_AAAA)

    def test_xor_self_annihilation(self) -> None:
        """XORing a value with itself yields zero."""
        addr = 0x100
        val = 0xCAFE_BABE
        self.lds.write_dword(addr, val)
        old_val, new_val = self.lds.atomic_xor_b32(addr, val)
        self.assertEqual(old_val, val)
        self.assertEqual(new_val, 0)
        self.assertEqual(self.lds.read_dword(addr), 0)

    def test_xor_involution_roundtrip(self) -> None:
        """Applying the same XOR operand twice restores the original value: (A ^ B) ^ B == A."""
        addr = 0x200
        initial = 0xDEAD_BEEF
        key = 0x1337_C0DE
        self.lds.write_dword(addr, initial)

        # First XOR
        old1, new1 = self.lds.atomic_xor_b32(addr, key)
        self.assertEqual(old1, initial)
        self.assertEqual(new1, initial ^ key)

        # Second XOR with same key restores initial
        old2, new2 = self.lds.atomic_xor_b32(addr, key)
        self.assertEqual(old2, initial ^ key)
        self.assertEqual(new2, initial)
        self.assertEqual(self.lds.read_dword(addr), initial)

    def test_offset_calculation_and_wrap(self) -> None:
        """Offset is 16-bit unsigned and added to vaddr modulo 2^32."""
        base = 0x1000
        offset = 0x24
        eff = calc_lds_effective_address(base, offset)
        self.assertEqual(eff, 0x1024)

        # 16-bit offset masking
        eff_masked = calc_lds_effective_address(base, 0x1_0004)
        self.assertEqual(eff_masked, 0x1004)

        # 32-bit address wraparound
        eff_wrap = calc_lds_effective_address(0xFFFF_FFF0, 0x0020)
        self.assertEqual(eff_wrap, 0x0000_0010)

    def test_dword_alignment_truncation(self) -> None:
        """Low 2 bits of effective address are ignored for dword indexing."""
        idx0 = calc_lds_dword_index(0x400)
        idx1 = calc_lds_dword_index(0x401)
        idx2 = calc_lds_dword_index(0x402)
        idx3 = calc_lds_dword_index(0x403)
        self.assertEqual(idx0, 0x100)
        self.assertEqual(idx1, 0x100)
        self.assertEqual(idx2, 0x100)
        self.assertEqual(idx3, 0x100)

    def test_bank_distribution(self) -> None:
        """32 consecutive dwords map to 32 distinct LDS banks [0..31]."""
        banks = [calc_lds_bank(i * 4) for i in range(32)]
        self.assertEqual(banks, list(range(32)))
        # Bank 32 wraps to 0
        self.assertEqual(calc_lds_bank(32 * 4), 0)

    def test_lds_capacity_wrap(self) -> None:
        """LDS access wraps modulo LDS capacity (default 64 KiB = 16384 dwords)."""
        addr_base = 0x10
        addr_wrapped = addr_base + LDS_SIZE_BYTES
        idx_base = calc_lds_dword_index(addr_base)
        idx_wrapped = calc_lds_dword_index(addr_wrapped)
        self.assertEqual(idx_base, idx_wrapped)

    def test_bit_patterns_walking_ones_and_zeros(self) -> None:
        """Test bitwise XOR across all 32 bit positions with walking ones and walking zeros."""
        addr = 0x300
        for bit in range(32):
            one = 1 << bit
            zero_inv = u32(~one)
            self.lds.write_dword(addr, 0)
            old, new = self.lds.atomic_xor_b32(addr, one)
            self.assertEqual(old, 0)
            self.assertEqual(new, one)

            old2, new2 = self.lds.atomic_xor_b32(addr, zero_inv)
            self.assertEqual(old2, one)
            self.assertEqual(new2, MASK_32)


if __name__ == "__main__":
    unittest.main()
