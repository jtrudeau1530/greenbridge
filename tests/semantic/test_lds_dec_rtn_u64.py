from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_dec_rtn_u64.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 64-bit Decrement with Return:
Performs a 64-bit atomic wrapping decrement on LDS memory and returns the original 64-bit value.

Syntax:
    ds_dec_rtn_u64 vdst, vaddr, vdata0 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + (offset16 & 0xFFFF)) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_qwords - 1)
    old_val = LDS[qword_index]
    if old_val == 0 or old_val > vdata0:
        new_mem = vdata0
    else:
        new_mem = (old_val - 1) & 0xFFFFFFFFFFFFFFFF
    LDS[qword_index] = new_mem
    vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Capacity: 64 KiB default capacity (8,192 64-bit qwords / 16,384 32-bit dwords).
  - Addressing: Base byte address in VGPR vaddr + 16-bit unsigned immediate offset (offset:offset16).
  - Alignment: 64-bit qword aligned; low 3 bits of byte address are truncated (eff_addr >> 3).
  - Wrap Condition: Unsigned 64-bit comparison:
      If old_val == 0 or old_val > vdata0: new_mem = vdata0
      Else: new_mem = (old_val - 1) mod 2^64.
  - Return: The pre-operation 64-bit value is returned to VGPR pair vdst (v[dst:dst+1]).
"""

from dataclasses import dataclass
from enum import Enum, auto
import math
import struct
from typing import Dict, List, Optional, Tuple, Union
import unittest

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536          # 64 KiB capacity
LDS_SIZE_DWORDS: int = 16384         # 16,384 32-bit dwords
LDS_SIZE_QWORDS: int = 8192          # 8,192 64-bit qwords
LDS_BANK_COUNT: int = 32             # 32 memory banks
MASK_32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
MASK_64: int = 0xFFFF_FFFF_FFFF_FFFF # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF          # 16-bit offset mask
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
DEFAULT_NUM_LANES: int = 64

# IEEE 754 Double-Precision Boundary Constants
POS_ZERO_F64: int = 0x0000000000000000
NEG_ZERO_F64: int = 0x8000000000000000
POS_INF_F64: int = 0x7FF0000000000000
NEG_INF_F64: int = 0xFFF0000000000000
QNAN_F64: int = 0x7FF8000000000000
SNAN_F64: int = 0x7FF0000000000001
MIN_NORMAL_F64: int = 0x0010000000000000
MIN_SUBNORMAL_F64: int = 0x0000000000000001
MAX_SUBNORMAL_F64: int = 0x000FFFFFFFFFFFFF
MAX_NORMAL_F64: int = 0x7FEFFFFFFFFFFFFF
ONE_F64: int = 0x3FF0000000000000


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK_64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as signed two's-complement integer."""
    val &= MASK_64
    return val - (1 << 64) if (val & (1 << 63)) else val


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def split_u64_to_u32s(val: int) -> Tuple[int, int]:
    """Split 64-bit unsigned integer into (lo, hi) 32-bit words."""
    val &= MASK_64
    return val & MASK_32, (val >> 32) & MASK_32


def combine_u32s_to_u64(lo: int, hi: int) -> int:
    """Combine (lo, hi) 32-bit words into a 64-bit unsigned integer."""
    return ((hi & MASK_32) << 32) | (lo & MASK_32)


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate byte offset.
    Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = LDS_SIZE_QWORDS) -> int:
    """Convert an effective byte address to an aligned LDS 64-bit qword index.

    LDS 64-bit addresses are qword-aligned; the low 3 bits are ignored by hardware.
    The index wraps modulo the power-of-two qword capacity.
    """
    return (eff_addr >> 3) & (capacity_qwords - 1)


def ref_ds_dec_u64(old_val: int, vdata0: int) -> int:
    """Pure reference mathematical formula for ds_dec_u64.

    If old_val == 0 or old_val > vdata0:
        new_val = vdata0
    else:
        new_val = (old_val - 1) & 0xFFFFFFFFFFFFFFFF
    """
    a = u64(old_val)
    b = u64(vdata0)
    if a == 0 or a > b:
        return b
    return u64(a - 1)


def oracle_ds_dec_rtn_u64(
    current_mem: int,
    wrap_limit: int,
) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_dec_rtn_u64.

    Computes:
        old_val = current_mem & 0xFFFFFFFFFFFFFFFF
        new_mem = ref_ds_dec_u64(old_val, wrap_limit)

    Returns:
        (old_val, new_mem)
    """
    old_val = u64(current_mem)
    new_mem = ref_ds_dec_u64(old_val, wrap_limit)
    return old_val, new_mem


class LDSModel64:
    """Architectural memory model representing the Local Data Share scratchpad with 64-bit access."""

    def __init__(self, capacity_qwords: int = LDS_SIZE_QWORDS) -> None:
        if capacity_qwords <= 0 or (capacity_qwords & (capacity_qwords - 1)) != 0:
            raise ValueError(f"capacity_qwords must be a positive power of two, got {capacity_qwords}")
        self.capacity_qwords = capacity_qwords
        self.memory: List[int] = [0] * capacity_qwords

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword from LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        return self.memory[idx]

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit qword to LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        self.memory[idx] = u64(val)

    def atomic_dec_rtn_u64(
        self,
        vaddr: int,
        wrap_limit: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 64-bit wrapping decrement with return on LDS.

        Returns:
            Tuple[int, int]: (old_val, new_mem)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        curr = self.memory[idx]
        old_val, new_mem = oracle_ds_dec_rtn_u64(curr, wrap_limit)
        self.memory[idx] = new_mem
        return old_val, new_mem


def bytearray_oracle_ds_dec_rtn_u64(
    lds_bytes: bytearray,
    vaddr: int,
    wrap_limit: int,
    offset: int = 0,
    lds_size_bytes: int = LDS_SIZE_BYTES,
) -> Tuple[int, int]:
    """Reference bytearray mathematical oracle for RDNA2 ds_dec_rtn_u64.

    Computes effective address, reads previous 64-bit qword, applies wrapping
    decrement logic, writes updated 64-bit qword back, and returns (old_val, new_mem).
    """
    eff_addr = calc_lds_effective_address(vaddr, offset) & (lds_size_bytes - 1)
    aligned_addr = eff_addr & ~7  # qword aligned (drop low 3 bits)
    old_val = struct.unpack_from("<Q", lds_bytes, aligned_addr)[0]
    new_mem = ref_ds_dec_u64(old_val, wrap_limit)
    struct.pack_into("<Q", lds_bytes, aligned_addr, new_mem)
    return old_val, new_mem


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSModel64(capacity_qwords=1024)
        self.lds_bytes = bytearray(LDS_SIZE_BYTES)

    def test_normal_decrement_u64(self) -> None:
        self.mem.write_qword(10 * 8, 5)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=80, wrap_limit=10, offset=0)
        self.assertEqual(old_val, 5)
        self.assertEqual(new_val, 4)
        self.assertEqual(self.mem.read_qword(10 * 8), 4)

        # Cross 32-bit boundary decrement: 0x1_0000_0000 -> 0x0_FFFF_FFFF
        self.mem.write_qword(10 * 8, 0x1_0000_0000)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=80, wrap_limit=0x5_0000_0000, offset=0)
        self.assertEqual(old_val, 0x1_0000_0000)
        self.assertEqual(new_val, 0x0_FFFF_FFFF)
        self.assertEqual(self.mem.read_qword(10 * 8), 0x0_FFFF_FFFF)

    def test_wrap_at_zero_u64(self) -> None:
        bound = 0x1234_5678_9ABC_DEF0
        self.mem.write_qword(20 * 8, 0)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=160, wrap_limit=bound, offset=0)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, bound)
        self.assertEqual(self.mem.read_qword(20 * 8), bound)

    def test_wrap_strictly_greater_than_bound_u64(self) -> None:
        bound = 0x1000_0000_0000_0000
        val_greater = 0xF000_0000_0000_0000
        self.mem.write_qword(30 * 8, val_greater)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=240, wrap_limit=bound, offset=0)
        self.assertEqual(old_val, val_greater)
        self.assertEqual(new_val, bound)
        self.assertEqual(self.mem.read_qword(30 * 8), bound)

        # MASK_64 wraps to bound
        self.mem.write_qword(30 * 8, MASK_64)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=240, wrap_limit=bound, offset=0)
        self.assertEqual(old_val, MASK_64)
        self.assertEqual(new_val, bound)
        self.assertEqual(self.mem.read_qword(30 * 8), bound)

    def test_decrement_when_equal_to_bound_u64(self) -> None:
        bound = 0xCAFE_BABE_DEAD_BEEF
        self.mem.write_qword(40 * 8, bound)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=320, wrap_limit=bound, offset=0)
        self.assertEqual(old_val, bound)
        self.assertEqual(new_val, bound - 1)
        self.assertEqual(self.mem.read_qword(40 * 8), bound - 1)

    def test_boundary_bound_zero_u64(self) -> None:
        self.mem.write_qword(5 * 8, 0)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=40, wrap_limit=0, offset=0)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, 0)

        self.mem.write_qword(5 * 8, 1)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=40, wrap_limit=0, offset=0)
        self.assertEqual(old_val, 1)
        self.assertEqual(new_val, 0)

        self.mem.write_qword(5 * 8, MASK_64)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=40, wrap_limit=0, offset=0)
        self.assertEqual(old_val, MASK_64)
        self.assertEqual(new_val, 0)

    def test_boundary_bound_max_u64(self) -> None:
        self.mem.write_qword(6 * 8, 0)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=48, wrap_limit=MASK_64, offset=0)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, MASK_64)

        self.mem.write_qword(6 * 8, MASK_64)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=48, wrap_limit=MASK_64, offset=0)
        self.assertEqual(old_val, MASK_64)
        self.assertEqual(new_val, MASK_64 - 1)

    def test_countdown_cycle_u64(self) -> None:
        bound = 3
        idx = 15
        self.mem.write_qword(idx * 8, bound)

        expected_sequence = [
            (3, 2),
            (2, 1),
            (1, 0),
            (0, 3),
            (3, 2),
        ]
        for exp_old, exp_new in expected_sequence:
            old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=idx * 8, wrap_limit=bound)
            self.assertEqual(old_val, exp_old)
            self.assertEqual(new_val, exp_new)
            self.assertEqual(self.mem.read_qword(idx * 8), exp_new)

    def test_effective_address_and_qword_alignment(self) -> None:
        # vaddr = 0x1000, offset = 16 -> eff = 0x1010 -> qword index = (0x1010 >> 3) & 1023 = 514 & 1023
        self.mem.write_qword(0x1010, 50)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=0x1000, wrap_limit=100, offset=16)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_val, 49)
        self.assertEqual(self.mem.read_qword(0x1010), 49)

        # Low 3 bits of byte address ignored (offset 7 -> eff 0x1007 -> aligned to 0x1000)
        self.mem.write_qword(0x1000, 25)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=0x1000, wrap_limit=100, offset=7)
        self.assertEqual(old_val, 25)
        self.assertEqual(new_val, 24)
        self.assertEqual(self.mem.read_qword(0x1000), 24)

    def test_bytearray_oracle_equivalence_u64(self) -> None:
        addr = 0x200
        val64 = 0x1122_3344_5566_7788
        struct.pack_into("<Q", self.lds_bytes, addr, val64)
        old_val, new_val = bytearray_oracle_ds_dec_rtn_u64(self.lds_bytes, vaddr=addr, wrap_limit=0x1000)
        # val64 > 0x1000 -> wraps to 0x1000
        self.assertEqual(old_val, val64)
        self.assertEqual(new_val, 0x1000)
        self.assertEqual(struct.unpack_from("<Q", self.lds_bytes, addr)[0], 0x1000)


if __name__ == "__main__":
    unittest.main()
