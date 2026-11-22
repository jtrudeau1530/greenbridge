"""Semantic reference oracle and architectural model for ds_condxchg32_rtn_b32.

RDNA2 / GFX10 Local Data Share (LDS) Conditional Exchange:
Performs a 32-bit atomic compare-and-exchange operation on LDS memory.
Syntax:
    ds_condxchg32_rtn_b32 vdst, vaddr, vdata0, vdata1 [offset:offset16]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    if old_val == vdata0:
        LDS[dword_index] = vdata1
    vdst = old_val
"""
from __future__ import annotations

import unittest
from typing import Tuple

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent banks
MASK_32: int = 0xFFFF_FFFF          # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask


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
    """Determine the LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_condxchg32(
    current_mem: int,
    cmp_val: int,
    new_val: int,
) -> Tuple[int, int, bool]:
    """Pure mathematical reference oracle for ds_condxchg32_rtn_b32.
    
    Returns:
        (returned_old_val, new_mem_val, exchanged)
    """
    old_val = current_mem & MASK_32
    cmp_masked = cmp_val & MASK_32
    new_masked = new_val & MASK_32

    if old_val == cmp_masked:
        return old_val, new_masked, True
    return old_val, old_val, False


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
        self.memory[idx] = val & MASK_32

    def atomic_condxchg32(
        self,
        vaddr: int,
        cmp_val: int,
        new_val: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        """Execute atomic conditional exchange on LDS dword."""
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        curr = self.memory[idx]
        old_val, new_mem, exchanged = oracle_ds_condxchg32(curr, cmp_val, new_val)
        self.memory[idx] = new_mem
        return old_val, new_mem, exchanged


class TestLdsCondxchg32ReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_successful_exchange_when_equal(self) -> None:
        old, mem, swapped = oracle_ds_condxchg32(0x12345678, 0x12345678, 0x9ABCDEF0)
        self.assertTrue(swapped)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(mem, 0x9ABCDEF0)

    def test_failed_exchange_when_not_equal(self) -> None:
        old, mem, swapped = oracle_ds_condxchg32(0x11112222, 0x33334444, 0x99990000)
        self.assertFalse(swapped)
        self.assertEqual(old, 0x11112222)
        self.assertEqual(mem, 0x11112222)

    def test_effective_address_with_immediate_offset(self) -> None:
        eff = calc_lds_effective_address(0x1000, 0x0024)
        self.assertEqual(eff, 0x1024)
        self.assertEqual(calc_lds_dword_index(eff), 0x1024 >> 2)

    def test_dword_alignment_and_bit_truncation(self) -> None:
        # Addresses differing only in byte offset within a dword map to the same dword
        eff0 = calc_lds_effective_address(0x200)
        eff1 = calc_lds_effective_address(0x201)
        eff2 = calc_lds_effective_address(0x202)
        eff3 = calc_lds_effective_address(0x203)
        self.assertEqual(calc_lds_dword_index(eff0), 0x80)
        self.assertEqual(calc_lds_dword_index(eff1), 0x80)
        self.assertEqual(calc_lds_dword_index(eff2), 0x80)
        self.assertEqual(calc_lds_dword_index(eff3), 0x80)

    def test_modulo_wrapping_at_capacity_boundary(self) -> None:
        capacity_bytes = LDS_SIZE_DWORDS * 4
        eff = calc_lds_effective_address(capacity_bytes + 0x10)
        idx = calc_lds_dword_index(eff)
        self.assertEqual(idx, 0x10 >> 2)

    def test_32bit_address_overflow_wrapping(self) -> None:
        eff = calc_lds_effective_address(0xFFFF_FFFE, 0x0004)
        self.assertEqual(eff, 0x0000_0002)
        self.assertEqual(calc_lds_dword_index(eff), 0)

    def test_bank_index_distribution(self) -> None:
        for bank in range(32):
            addr = bank * 4
            self.assertEqual(calc_lds_bank(addr), bank)

    def test_lds_model_state_update_on_match(self) -> None:
        lds = LDSModel()
        addr = 0x400
        lds.write_dword(addr, 0xCAFEBABE)
        old, mem, swapped = lds.atomic_condxchg32(addr, 0xCAFEBABE, 0xDEADBEEF, offset=0)
        self.assertTrue(swapped)
        self.assertEqual(old, 0xCAFEBABE)
        self.assertEqual(mem, 0xDEADBEEF)
        self.assertEqual(lds.read_dword(addr), 0xDEADBEEF)

    def test_lds_model_state_preservation_on_mismatch(self) -> None:
        lds = LDSModel()
        addr = 0x800
        lds.write_dword(addr, 0x5555AAAA)
        # Attempt swap at addr with mismatched compare value
        old, mem, swapped = lds.atomic_condxchg32(addr, 0x11112222, 0xFFFFFFFF, offset=0)
        self.assertFalse(swapped)
        self.assertEqual(old, 0x5555AAAA)
        self.assertEqual(mem, 0x5555AAAA)
        # Check original addr was preserved
        self.assertEqual(lds.read_dword(addr), 0x5555AAAA)


if __name__ == "__main__":
    unittest.main()
