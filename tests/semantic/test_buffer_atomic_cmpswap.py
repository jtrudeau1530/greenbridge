"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_cmpswap.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory atomic compare-and-swap:
  buffer_atomic_cmpswap vdata, vaddr, srsrc, soffset [offset:N] [offen] [idxen] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR pair v[d:d+1] containing 64 bits:
         vdata[0]: 32-bit new value to store if comparison matches (src).
         vdata[1]: 32-bit expected value to compare against memory (cmp).
       If 'glc' is enabled, the pre-operation dword value at memory is returned to vdata[0].
       vdata[1] is preserved and not modified by hardware writeback.
     - vaddr: VGPR containing byte or index offset (when offen or idxen is enabled).
     - srsrc: 128-bit Buffer Resource Descriptor (4 SGPRs s[r:r+3]):
         Dword 0: Base address low [31:0]
         Dword 1: Base address high [15:0], STRIDE [31:16]
         Dword 2: NUM_RECORDS (buffer limit / capacity in bytes)
         Dword 3: Format, element size, resource type
     - soffset: Scalar SGPR or immediate byte offset.
     - inst_offset: 12-bit unsigned immediate byte offset [0, 4095].
  2. Addressing and Bounds Checking:
     effective_offset = (soffset & 0xFFFFFFFF) + (inst_offset & 0xFFF) + (vaddr & 0xFFFFFFFF if offen else 0)
     If effective_offset + 4 > NUM_RECORDS:
       Access is Out-Of-Bounds (OOB).
       The atomic cmpswap is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to vdata[0].
  3. Atomic Operation:
     For in-bounds accesses:
       old_val = memory[effective_address]
       if old_val == vdata[1]:
           memory[effective_address] = vdata[0]
           swapped = True
       else:
           swapped = False
       if glc:
           vdata[0] = old_val
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


def ref_buffer_atomic_cmpswap(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    src_val: int,
    cmp_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_cmpswap.

    Returns:
      (return_val, was_swapped, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS
    if effective_offset + 4 > desc.num_records:
        # Out-of-bounds: memory untouched, returns 0 if glc else src_val
        return (0 if glc else (src_val & MASK32)), False, False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u32(phys_addr)
    u_cmp = cmp_val & MASK32
    u_src = src_val & MASK32

    swapped = (old_val == u_cmp)
    if swapped:
        mem.write_u32(phys_addr, u_src)

    ret_val = old_val if glc else u_src
    return ret_val, swapped, True


class TestBufferAtomicCmpswapOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_successful_cmpswap_in_bounds(self) -> None:
        # Memory contains 0x11223344; compare against 0x11223344, swap in 0x55667788
        self.mem.write_u32(64, 0x11223344)
        ret, swapped, in_bounds = ref_buffer_atomic_cmpswap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=64,
            src_val=0x55667788,
            cmp_val=0x11223344,
            glc=False,
        )
        self.assertTrue(in_bounds)
        self.assertTrue(swapped)
        self.assertEqual(self.mem.read_u32(64), 0x55667788)
        self.assertEqual(ret, 0x55667788)

    def test_failed_cmpswap_when_not_equal(self) -> None:
        # Memory contains 0x1000; cmp is 0x2000 (mismatch); memory unchanged
        self.mem.write_u32(64, 0x1000)
        ret, swapped, in_bounds = ref_buffer_atomic_cmpswap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=64,
            src_val=0x3000,
            cmp_val=0x2000,
            glc=False,
        )
        self.assertTrue(in_bounds)
        self.assertFalse(swapped)
        self.assertEqual(self.mem.read_u32(64), 0x1000)
        self.assertEqual(ret, 0x3000)

    def test_cmpswap_glc_returns_pre_op_value_on_match(self) -> None:
        self.mem.write_u32(128, 42)
        ret, swapped, in_bounds = ref_buffer_atomic_cmpswap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=128,
            src_val=99,
            cmp_val=42,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertTrue(swapped)
        self.assertEqual(ret, 42)
        self.assertEqual(self.mem.read_u32(128), 99)

    def test_cmpswap_glc_returns_pre_op_value_on_mismatch(self) -> None:
        self.mem.write_u32(128, 42)
        ret, swapped, in_bounds = ref_buffer_atomic_cmpswap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=128,
            src_val=99,
            cmp_val=100,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertFalse(swapped)
        self.assertEqual(ret, 42)
        self.assertEqual(self.mem.read_u32(128), 42)

    def test_out_of_bounds_rejection(self) -> None:
        # num_records is 1024; offset 1024 is out of bounds
        self.mem.write_u32(1024, 0xCAFE)
        ret, swapped, in_bounds = ref_buffer_atomic_cmpswap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1024,
            src_val=0xBEEF,
            cmp_val=0xCAFE,
            glc=True,
        )
        self.assertFalse(in_bounds)
        self.assertFalse(swapped)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u32(1024), 0xCAFE)

    def test_inst_offset_and_soffset_combination(self) -> None:
        # Total offset = soffset(128) + inst_offset(32) + vaddr(40) = 200
        self.mem.write_u32(200, 500)
        ret, swapped, in_bounds = ref_buffer_atomic_cmpswap(
            self.mem,
            self.desc,
            soffset=128,
            vaddr=40,
            src_val=600,
            cmp_val=500,
            inst_offset=32,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertTrue(swapped)
        self.assertEqual(ret, 500)
        self.assertEqual(self.mem.read_u32(200), 600)

    def test_wrapping_32bit_integer_address_and_data(self) -> None:
        self.mem.write_u32(0, 0xFFFFFFFF)
        ret, swapped, in_bounds = ref_buffer_atomic_cmpswap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=0,
            src_val=0,
            cmp_val=0xFFFFFFFF,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertTrue(swapped)
        self.assertEqual(ret, 0xFFFFFFFF)
        self.assertEqual(self.mem.read_u32(0), 0)

    def test_signed_two_complement_values(self) -> None:
        # -1 in two's complement is 0xFFFFFFFF
        self.mem.write_u32(16, u32(-1))
        ret, swapped, in_bounds = ref_buffer_atomic_cmpswap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=16,
            src_val=u32(-2),
            cmp_val=s32(-1),
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertTrue(swapped)
        self.assertEqual(ret, 0xFFFFFFFF)
        self.assertEqual(self.mem.read_u32(16), u32(-2))

    def test_identity_swap(self) -> None:
        self.mem.write_u32(48, 0x12345678)
        ret, swapped, in_bounds = ref_buffer_atomic_cmpswap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=48,
            src_val=0x12345678,
            cmp_val=0x12345678,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertTrue(swapped)
        self.assertEqual(ret, 0x12345678)
        self.assertEqual(self.mem.read_u32(48), 0x12345678)

    def test_descriptor_roundtrip(self) -> None:
        desc = BufferResourceDescriptor(
            base_address=0x00007FFF00000000,
            stride=16,
            num_records=65536,
            format_flags=0x80,
        )
        d0, d1, d2, d3 = desc.to_dwords()
        restored = BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)
        self.assertEqual(desc, restored)


if __name__ == "__main__":
    unittest.main()
