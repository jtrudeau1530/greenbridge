from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_min_rtn_u32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 32-bit Unsigned Minimum with Return:
Performs a 32-bit atomic unsigned minimum operation on LDS storage and returns
the previous value in memory to the destination VGPR.

Syntax:
    ds_min_rtn_u32 vdst, vaddr, vdata0 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    LDS[dword_index] = min(old_val, vdata0)   # Unsigned 32-bit comparison
    vdst = old_val

Standard library Python only; zero third-party dependencies; zero game data.
"""

import math
import struct
import unittest
from typing import Dict, List, Optional, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
MASK32: int = 0xFFFFFFFF
UINT32_MIN: int = 0x00000000
UINT32_MAX: int = 0xFFFFFFFF
UINT32_MID: int = 0x80000000
UINT32_HIGH_BIT: int = 0x80000000

DEFAULT_LDS_SIZE_BYTES: int = 65536   # 64 KiB default capacity
DEFAULT_LDS_NUM_DWORDS: int = 16384  # 16,384 32-bit dwords
NUM_LDS_BANKS: int = 32              # 32 memory banks
OFFSET16_MASK: int = 0xFFFF          # 16-bit immediate byte offset mask


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = DEFAULT_LDS_NUM_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0 to 31) for an effective address."""
    return (eff_addr >> 2) % NUM_LDS_BANKS


def oracle_min_u32(a: int, b: int) -> int:
    """Pure reference unsigned 32-bit minimum."""
    ua = a & MASK32
    ub = b & MASK32
    return ua if ua < ub else ub


def ref_ds_min_rtn_u32(current_mem: int, candidate: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_min_rtn_u32.

    Computes:
        old_val = current_mem & 0xFFFFFFFF
        new_mem = min(old_val, candidate) in unsigned 32-bit domain

    Returns:
        (old_val, new_mem)
    """
    old_val = u32(current_mem)
    cand = u32(candidate)
    new_mem = oracle_min_u32(old_val, cand)
    return old_val, new_mem


class LDSMemory:
    """Architectural simulation model of AMD RDNA2 Local Data Share (LDS)."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of 2, got {size_bytes}")
        self.size_bytes: int = size_bytes
        self.num_dwords: int = size_bytes // 4
        self.dwords: List[int] = [0] * self.num_dwords

    def get_dword_index(self, byte_addr: int, offset: int = 0) -> int:
        """Compute effective wrapped dword index, dropping bottom 2 unaligned bits."""
        eff_addr = calc_lds_effective_address(byte_addr, offset)
        return calc_lds_dword_index(eff_addr, self.num_dwords)

    def get_bank(self, dword_idx: int) -> int:
        """Compute memory bank index for the given dword (0 to 31)."""
        return dword_idx % NUM_LDS_BANKS

    def read_dword(self, dword_idx: int) -> int:
        """Read a 32-bit dword from LDS."""
        return self.dwords[dword_idx & (self.num_dwords - 1)]

    def write_dword(self, dword_idx: int, val: int) -> None:
        """Write a 32-bit dword into LDS."""
        self.dwords[dword_idx & (self.num_dwords - 1)] = val & MASK32

    def read_by_byte_addr(self, byte_addr: int, offset: int = 0) -> int:
        """Read a 32-bit dword using effective byte address."""
        idx = self.get_dword_index(byte_addr, offset)
        return self.read_dword(idx)

    def write_by_byte_addr(self, byte_addr: int, val: int, offset: int = 0) -> None:
        """Write a 32-bit dword using effective byte address."""
        idx = self.get_dword_index(byte_addr, offset)
        self.write_dword(idx, val)

    def atomic_min_rtn_u32(self, byte_addr: int, val: int, offset: int = 0) -> int:
        """Execute atomic unsigned minimum at (byte_addr + offset), returning previous value."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        new_val = oracle_min_u32(old_val, val)
        self.dwords[idx] = new_val
        return old_val


class TestDSMinRtnU32OracleAndModel(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural LDS model."""

    def test_oracle_basic_min(self) -> None:
        self.assertEqual(oracle_min_u32(10, 20), 10)
        self.assertEqual(oracle_min_u32(20, 10), 10)
        self.assertEqual(oracle_min_u32(50, 50), 50)
        self.assertEqual(oracle_min_u32(0, 0), 0)

    def test_oracle_unsigned_ordering(self) -> None:
        # In unsigned comparison, 0 is strictly smaller than 0xFFFFFFFF
        self.assertEqual(oracle_min_u32(UINT32_MIN, UINT32_MAX), UINT32_MIN)
        self.assertEqual(oracle_min_u32(UINT32_MAX, UINT32_MIN), UINT32_MIN)
        # 0x7FFFFFFF vs 0x80000000
        self.assertEqual(oracle_min_u32(0x7FFFFFFF, 0x80000000), 0x7FFFFFFF)
        # 0x80000000 vs 0xFFFFFFFF
        self.assertEqual(oracle_min_u32(0x80000000, 0xFFFFFFFF), 0x80000000)

    def test_ref_ds_min_rtn_u32(self) -> None:
        old_val, new_mem = ref_ds_min_rtn_u32(100, 40)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_mem, 40)

        # When candidate is larger, memory remains unchanged
        old_val, new_mem = ref_ds_min_rtn_u32(40, 999)
        self.assertEqual(old_val, 40)
        self.assertEqual(new_mem, 40)

    def test_lds_memory_init_and_bounds(self) -> None:
        lds = LDSMemory(65536)
        self.assertEqual(lds.size_bytes, 65536)
        self.assertEqual(lds.num_dwords, 16384)
        self.assertEqual(len(lds.dwords), 16384)
        for i in range(10):
            self.assertEqual(lds.read_dword(i), 0)

        # Invalid sizes
        with self.assertRaises(ValueError):
            LDSMemory(0)
        with self.assertRaises(ValueError):
            LDSMemory(-1024)
        with self.assertRaises(ValueError):
            LDSMemory(50000)  # Not a power of 2

    def test_lds_addressing_and_alignment(self) -> None:
        lds = LDSMemory(65536)
        # Bytes 0, 1, 2, 3 all access dword 0
        self.assertEqual(lds.get_dword_index(0), 0)
        self.assertEqual(lds.get_dword_index(1), 0)
        self.assertEqual(lds.get_dword_index(2), 0)
        self.assertEqual(lds.get_dword_index(3), 0)
        # Byte 4 accesses dword 1
        self.assertEqual(lds.get_dword_index(4), 1)
        # Offset 4 on byte 0 accesses dword 1
        self.assertEqual(lds.get_dword_index(0, offset=4), 1)
        # Offset 8 on byte 4 accesses dword 3
        self.assertEqual(lds.get_dword_index(4, offset=8), 3)

        # Capacity wrapping
        self.assertEqual(lds.get_dword_index(65536), 0)
        self.assertEqual(lds.get_dword_index(65536 + 12), 3)

        # 32-bit address overflow wrapping
        self.assertEqual(lds.get_dword_index(0xFFFFFFFF, offset=1), 0)

    def test_lds_bank_distribution(self) -> None:
        lds = LDSMemory(65536)
        for dword in range(64):
            expected_bank = dword % 32
            self.assertEqual(lds.get_bank(dword), expected_bank)
            self.assertEqual(calc_lds_bank(dword * 4), expected_bank)

    def test_lds_atomic_min_rtn_u32_execution(self) -> None:
        lds = LDSMemory(65536)
        # Initial write
        lds.write_by_byte_addr(0x100, 0x1000)
        self.assertEqual(lds.read_by_byte_addr(0x100), 0x1000)

        # Atomic min with smaller candidate: returns 0x1000, memory becomes 0x0500
        old = lds.atomic_min_rtn_u32(0x100, 0x0500)
        self.assertEqual(old, 0x1000)
        self.assertEqual(lds.read_by_byte_addr(0x100), 0x0500)

        # Atomic min with larger candidate: returns 0x0500, memory remains 0x0500
        old2 = lds.atomic_min_rtn_u32(0x100, 0x0800)
        self.assertEqual(old2, 0x0500)
        self.assertEqual(lds.read_by_byte_addr(0x100), 0x0500)

        # Atomic min with offset
        old3 = lds.atomic_min_rtn_u32(0x0FC, 0x0200, offset=4)
        self.assertEqual(old3, 0x0500)
        self.assertEqual(lds.read_by_byte_addr(0x100), 0x0200)


if __name__ == "__main__":
    unittest.main()
