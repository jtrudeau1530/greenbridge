from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_max_rtn_u64.

RDNA2 / GFX10 Local Data Share (LDS) Atomic Unsigned 64-bit Integer Maximum with Return:
Atomically compares an unsigned 64-bit integer operand with the 64-bit value in LDS memory,
computes the unsigned maximum, updates LDS with the maximum, and returns the original
64-bit value from LDS memory to the destination VGPR pair.

Syntax:
    ds_max_rtn_u64 v[dst_lo:dst_hi], vaddr, v[data_lo:data_hi] [offset:offset16]

Architectural Semantics:
    effective_byte_address = (vaddr + (offset16 & 0xFFFF)) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_qwords - 1)
    old_val = LDS_QWORD[qword_index]
    new_val = max(old_val, vdata_u64)
    LDS_QWORD[qword_index] = new_val
    vdst_lo = old_val & 0xFFFFFFFF
    vdst_hi = (old_val >> 32) & 0xFFFFFFFF

Unsigned 64-bit Integer Properties:
    - Range: [0, 2^64 - 1] ([0, 18446744073709551615])
    - UINT64_MIN = 0
    - UINT64_MAX = 0xFFFFFFFFFFFFFFFF
    - Unsigned comparison: 0xFFFFFFFFFFFFFFFF > 0, 0x8000000000000000 > 0x7FFFFFFFFFFFFFFF
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_SIZE_QWORDS: int = 8192         # 8,192 64-bit qwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask

UINT64_MIN: int = 0
UINT64_MAX: int = 0xFFFF_FFFF_FFFF_FFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword [0, 2^64 - 1]."""
    return val & MASK64


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = LDS_SIZE_QWORDS) -> int:
    """Convert an effective byte address to an aligned LDS 64-bit qword index.

    LDS 64-bit qwords are 8-byte aligned; the low 3 bits are ignored by hardware.
    The index wraps modulo the power-of-two qword capacity.
    """
    return (eff_addr >> 3) & (capacity_qwords - 1)


def calc_lds_qword_banks(eff_addr: int) -> Tuple[int, int]:
    """Determine the two LDS memory bank indices (0..31) accessed by a 64-bit qword."""
    dword_idx = (eff_addr >> 2) & (LDS_SIZE_DWORDS - 1)
    bank_lo = dword_idx % LDS_BANK_COUNT
    bank_hi = (dword_idx + 1) % LDS_BANK_COUNT
    return bank_lo, bank_hi


def ref_ds_max_rtn_u64(
    current_mem: int,
    data_val: int,
) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_max_rtn_u64.

    Computes:
        old_val = current_mem & MASK64
        new_val = max(old_val, data_val & MASK64)

    Returns:
        (old_val, new_val)
    """
    old_val = u64(current_mem)
    data_u = u64(data_val)
    new_val = max(old_val, data_u)
    return old_val, new_val


class LDSMemory64:
    """Architectural memory model representing LDS with 64-bit atomic access support."""

    def __init__(self, size_bytes: int = LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"size_bytes must be a positive power of two, got {size_bytes}")
        self.size_bytes = size_bytes
        self.num_dwords = size_bytes // 4
        self.num_qwords = size_bytes // 8
        self.dwords: List[int] = [0] * self.num_dwords

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit unsigned integer from LDS at 8-byte aligned address."""
        dword_idx = ((eff_addr >> 3) << 1) & (self.num_dwords - 1)
        lo = self.dwords[dword_idx]
        hi = self.dwords[dword_idx + 1]
        return lo | (hi << 32)

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit unsigned integer to LDS at 8-byte aligned address."""
        dword_idx = ((eff_addr >> 3) << 1) & (self.num_dwords - 1)
        v = u64(val)
        self.dwords[dword_idx] = v & MASK32
        self.dwords[dword_idx + 1] = (v >> 32) & MASK32

    def atomic_max_rtn_u64(
        self,
        vaddr: int,
        vdata: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic unsigned 64-bit maximum with return on LDS.

        Returns:
            (old_val, new_val)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        curr = self.read_qword(eff_addr)
        old_val, new_val = ref_ds_max_rtn_u64(curr, vdata)
        self.write_qword(eff_addr, new_val)
        return old_val, new_val


class TestLDSMaxRtnU64OracleAndModel(unittest.TestCase):
    """Phase 1: Unit tests for unsigned 64-bit integer oracle and LDS architectural model."""

    def test_oracle_unsigned_basic_ordering(self) -> None:
        old_val, new_val = ref_ds_max_rtn_u64(10, 20)
        self.assertEqual(old_val, 10)
        self.assertEqual(new_val, 20)

        old_val2, new_val2 = ref_ds_max_rtn_u64(500, 250)
        self.assertEqual(old_val2, 500)
        self.assertEqual(new_val2, 500)

    def test_oracle_high_bit_unsigned_semantics(self) -> None:
        # 0x8000000000000000 > 0x7FFFFFFFFFFFFFFF in unsigned 64-bit
        val_high = 0x8000000000000000
        val_mid = 0x7FFFFFFFFFFFFFFF
        old_val, new_val = ref_ds_max_rtn_u64(val_mid, val_high)
        self.assertEqual(old_val, val_mid)
        self.assertEqual(new_val, val_high)

    def test_oracle_uint64_extremes(self) -> None:
        # UINT64_MAX vs 0
        old_val, new_val = ref_ds_max_rtn_u64(0, UINT64_MAX)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, UINT64_MAX)

        # UINT64_MAX vs UINT64_MAX - 1
        old_val2, new_val2 = ref_ds_max_rtn_u64(UINT64_MAX - 1, UINT64_MAX)
        self.assertEqual(old_val2, UINT64_MAX - 1)
        self.assertEqual(new_val2, UINT64_MAX)

    def test_oracle_equal_operands(self) -> None:
        val = 0x1122334455667788
        old_val, new_val = ref_ds_max_rtn_u64(val, val)
        self.assertEqual(old_val, val)
        self.assertEqual(new_val, val)

    def test_lds_memory_qword_read_write(self) -> None:
        lds = LDSMemory64(size_bytes=4096)
        val = 0xAABBCCDDEEFF0011
        lds.write_qword(0x20, val)
        self.assertEqual(lds.read_qword(0x20), val)

    def test_lds_qword_alignment_drop_low_bits(self) -> None:
        lds = LDSMemory64(size_bytes=4096)
        val = 0x123456789ABCDEF0
        lds.write_qword(0x40, val)
        # Bytes 0x40 through 0x47 all align to qword starting at 0x40
        for off in range(8):
            self.assertEqual(lds.read_qword(0x40 + off), val)

    def test_lds_memory_capacity_wrapping(self) -> None:
        lds = LDSMemory64(size_bytes=4096)
        val = 0xFEEDFACECAFEBABE
        lds.write_qword(0, val)
        self.assertEqual(lds.read_qword(4096), val)
        self.assertEqual(lds.read_qword(8192), val)

    def test_lds_atomic_max_rtn_u64(self) -> None:
        lds = LDSMemory64(size_bytes=4096)
        addr = 0x80
        lds.write_qword(addr, 1000)

        # max(1000, 5000) -> returns 1000, new is 5000
        old_val, new_val = lds.atomic_max_rtn_u64(addr, 5000)
        self.assertEqual(old_val, 1000)
        self.assertEqual(new_val, 5000)
        self.assertEqual(lds.read_qword(addr), 5000)

        # max(5000, 2000) -> returns 5000, new remains 5000
        old_val2, new_val2 = lds.atomic_max_rtn_u64(addr, 2000)
        self.assertEqual(old_val2, 5000)
        self.assertEqual(new_val2, 5000)
        self.assertEqual(lds.read_qword(addr), 5000)

    def test_lds_qword_bank_distribution(self) -> None:
        # A 64-bit access spans two consecutive dword banks
        b_lo, b_hi = calc_lds_qword_banks(0)
        self.assertEqual((b_lo, b_hi), (0, 1))

        b_lo2, b_hi2 = calc_lds_qword_banks(124)  # dword 31
        self.assertEqual((b_lo2, b_hi2), (31, 0))  # wraps around 32 banks


if __name__ == "__main__":
    unittest.main()
