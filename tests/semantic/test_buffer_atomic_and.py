"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_and.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory atomic bitwise AND:
  buffer_atomic_and vdata, vaddr, srsrc, soffset [offset:N] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR containing the 32-bit dword value to bitwise AND atomically. If 'glc' is enabled,
       the pre-operation dword value at memory is returned to vdata.
     - vaddr: VGPR containing the byte or index offset (when offen or idxen is enabled).
     - srsrc: 128-bit Buffer Resource Descriptor (4 SGPRs s[r:r+3]):
         Dword 0: Base address low [31:0]
         Dword 1: Base address high [15:0], STRIDE [31:16]
         Dword 2: NUM_RECORDS (buffer limit / capacity in bytes)
         Dword 3: Format, element size, resource type
     - soffset: Scalar SGPR or immediate byte offset.
     - inst_offset: 12-bit unsigned immediate byte offset [0, 4095].
  2. Addressing and Bounds Checking:
     effective_offset = soffset + inst_offset + (vaddr if offen else 0)
     If effective_offset + 4 > NUM_RECORDS:
       Access is Out-Of-Bounds (OOB).
       The atomic operation is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to vdata.
  3. Atomic Operation:
     For in-bounds accesses:
       old_val = memory[effective_address]
       new_val = (old_val & (and_val & 0xFFFFFFFF)) & 0xFFFFFFFF
       memory[effective_address] = new_val
       if glc: vdata = old_val
  4. Inactive Lanes (EXEC mask):
     Lanes disabled by EXEC mask never touch memory and do not modify their VGPRs.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
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


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


@dataclasses.dataclass(frozen=True)
class BufferResourceDescriptor:
    """AMD GFX10 128-bit Buffer Resource Descriptor (V#)."""
    base_address: int
    stride: int
    num_records: int
    format_flags: int = 0

    @classmethod
    def from_dwords(cls, d0: int, d1: int, d2: int, d3: int) -> BufferResourceDescriptor:
        base_lo = d0 & MASK32
        base_hi = d1 & 0xFFFF
        base = (base_hi << 32) | base_lo
        stride = (d1 >> 16) & 0xFFFF
        num_records = d2 & MASK32
        return cls(base_address=base, stride=stride, num_records=num_records, format_flags=d3)

    def to_dwords(self) -> tuple[int, int, int, int]:
        d0 = self.base_address & MASK32
        d1 = ((self.base_address >> 32) & 0xFFFF) | ((self.stride & 0xFFFF) << 16)
        d2 = self.num_records & MASK32
        d3 = self.format_flags & MASK32
        return d0, d1, d2, d3


class BufferMemory:
    """Byte-addressable memory buffer simulating GPU global / buffer device memory."""

    def __init__(self, size: int = 65536) -> None:
        self.data: bytearray = bytearray(size)

    def read_u32(self, addr: int) -> int:
        if addr < 0 or addr + 4 > len(self.data):
            return 0
        return struct.unpack_from("<I", self.data, addr)[0]

    def write_u32(self, addr: int, val: int) -> None:
        if addr < 0 or addr + 4 > len(self.data):
            return
        struct.pack_into("<I", self.data, addr, val & MASK32)


def ref_buffer_atomic_and(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    and_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_and.

    Returns:
      (return_val, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS
    if effective_offset + 4 > desc.num_records:
        # Out-of-bounds: memory untouched, returns 0 if glc
        return (0 if glc else (and_val & MASK32)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u32(phys_addr)
    new_val = (old_val & (and_val & MASK32)) & MASK32
    mem.write_u32(phys_addr, new_val)

    ret_val = old_val if glc else (and_val & MASK32)
    return ret_val, True


class TestBufferAtomicAndOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_atomic_and_in_bounds(self) -> None:
        # Pre-seed address 100 with 0xFF00FF00
        self.mem.write_u32(100, 0xFF00FF00)
        ret, in_bounds = ref_buffer_atomic_and(
            self.mem, self.desc, soffset=0, vaddr=100, and_val=0xF0F0F0F0, glc=False
        )
        self.assertTrue(in_bounds)
        # Memory should now hold 0xFF00FF00 & 0xF0F0F0F0 = 0xF000F000
        self.assertEqual(self.mem.read_u32(100), 0xF000F000)

    def test_atomic_and_glc_returns_pre_op_value(self) -> None:
        self.mem.write_u32(200, 0x12345678)
        ret, in_bounds = ref_buffer_atomic_and(
            self.mem, self.desc, soffset=0, vaddr=200, and_val=0x00FF00FF, glc=True
        )
        self.assertTrue(in_bounds)
        # glc returns old value (0x12345678)
        self.assertEqual(ret, 0x12345678)
        # memory updated to 0x00340078
        self.assertEqual(self.mem.read_u32(200), 0x00340078)

    def test_out_of_bounds_rejection(self) -> None:
        # num_records is 1024; offset 1024 is out of bounds
        self.mem.write_u32(1024, 0xAAAAAAAA)
        ret, in_bounds = ref_buffer_atomic_and(
            self.mem, self.desc, soffset=0, vaddr=1024, and_val=0x55555555, glc=True
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u32(1024), 0xAAAAAAAA)

    def test_bitwise_and_identities(self) -> None:
        # Identity: X & 0xFFFFFFFF == X
        self.mem.write_u32(0, 0xDEADBEEF)
        ret, in_bounds = ref_buffer_atomic_and(
            self.mem, self.desc, soffset=0, vaddr=0, and_val=0xFFFFFFFF, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xDEADBEEF)
        self.assertEqual(self.mem.read_u32(0), 0xDEADBEEF)

        # Zero annihilator: X & 0 == 0
        ret, in_bounds = ref_buffer_atomic_and(
            self.mem, self.desc, soffset=0, vaddr=0, and_val=0, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xDEADBEEF)
        self.assertEqual(self.mem.read_u32(0), 0)

        # Idempotence: 0 & 0 == 0
        ret, in_bounds = ref_buffer_atomic_and(
            self.mem, self.desc, soffset=0, vaddr=0, and_val=0x1234, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u32(0), 0)

    def test_inst_offset_and_soffset_combination(self) -> None:
        # Total offset = soffset(64) + inst_offset(16) + vaddr(20) = 100
        self.mem.write_u32(100, 0x0F0F0F0F)
        ret, in_bounds = ref_buffer_atomic_and(
            self.mem, self.desc, soffset=64, vaddr=20, and_val=0x03030303, inst_offset=16, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x0F0F0F0F)
        self.assertEqual(self.mem.read_u32(100), 0x03030303)

    def test_descriptor_to_from_dwords_roundtrip(self) -> None:
        desc = BufferResourceDescriptor(
            base_address=0x00007FFF12340000,
            stride=64,
            num_records=65536,
            format_flags=0x12345678,
        )
        d0, d1, d2, d3 = desc.to_dwords()
        restored = BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)
        self.assertEqual(desc.base_address, restored.base_address)
        self.assertEqual(desc.stride, restored.stride)
        self.assertEqual(desc.num_records, restored.num_records)
        self.assertEqual(desc.format_flags, restored.format_flags)


if __name__ == "__main__":
    unittest.main()
