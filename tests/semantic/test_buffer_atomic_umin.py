from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_umin.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory unsigned atomic minimum:
  buffer_atomic_umin vdata, vaddr, srsrc, soffset [offset:N] [offen] [idxen] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR containing the 32-bit dword value to compare atomically. If 'glc' is enabled,
       the pre-operation dword value at memory is returned to vdata.
     - vaddr: VGPR containing the byte or index offset (when offen or idxen is enabled).
     - srsrc: 128-bit Buffer Resource Descriptor (4 SGPRs s[r:r+3]):
         Dword 0: Base address low [31:0]
         Dword 1: Base address high [15:0], STRIDE [31:16]
         Dword 2: NUM_RECORDS (buffer capacity in bytes)
         Dword 3: Format, element size, resource type
     - soffset: Scalar SGPR or immediate byte offset.
     - inst_offset: 12-bit unsigned immediate byte offset [0, 4095].
  2. Addressing and Bounds Checking:
     effective_offset = soffset + inst_offset + (vaddr if offen else 0)
     If effective_offset + 4 > NUM_RECORDS:
       Access is Out-Of-Bounds (OOB).
       The atomic operation is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to vdata.
  3. Atomic Unsigned Minimum Operation:
     For in-bounds accesses:
       old_val = memory[effective_address]
       new_val = min(unsigned32(old_val), unsigned32(vdata))
       memory[effective_address] = unsigned32(new_val)
       if glc: vdata = old_val
  4. Inactive Lanes (EXEC mask):
     Lanes disabled by EXEC mask never touch memory and do not modify their VGPRs.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import dataclasses
import enum
import struct
import sys
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT32: int = 0x80000000
UINT32_MAX: int = 0xFFFFFFFF
UINT32_MIN: int = 0x00000000


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


def ref_buffer_atomic_umin(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    cmp_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference mathematical oracle for RDNA2 buffer_atomic_umin.

    Computes:
      effective_offset = soffset + inst_offset + (vaddr if offen else 0)
      if effective_offset + 4 <= desc.num_records:
        old_val = mem[effective_address]
        new_val = min(u32(old_val), u32(cmp_val))
        mem[effective_address] = new_val
        return (old_val if glc else cmp_val), True
      else:
        return (0 if glc else cmp_val), False
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS
    if effective_offset + 4 > desc.num_records:
        # Out-of-bounds: memory untouched, returns 0 if glc
        return (0 if glc else (cmp_val & MASK32)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u32(phys_addr)
    min_val = min(u32(old_val), u32(cmp_val))
    mem.write_u32(phys_addr, min_val)

    ret_val = old_val if glc else (cmp_val & MASK32)
    return ret_val, True


class TestBufferAtomicUminOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_unsigned_min_basic(self) -> None:
        # Initial memory has 20, cmp with 10 -> min is 10
        self.mem.write_u32(0, 20)
        ret, ok = ref_buffer_atomic_umin(self.mem, self.desc, soffset=0, vaddr=0, cmp_val=10, glc=True)
        self.assertTrue(ok)
        self.assertEqual(ret, 20)
        self.assertEqual(self.mem.read_u32(0), 10)

        # Compare with 30 -> min is still 10
        ret2, ok2 = ref_buffer_atomic_umin(self.mem, self.desc, soffset=0, vaddr=0, cmp_val=30, glc=True)
        self.assertTrue(ok2)
        self.assertEqual(ret2, 10)
        self.assertEqual(self.mem.read_u32(0), 10)

    def test_unsigned_vs_signed_difference(self) -> None:
        # Key distinction between smin and umin:
        # 0xFFFFFFFF is -1 in signed, but 4,294,967,295 in unsigned!
        # Comparing 10 vs 0xFFFFFFFF:
        # smin(10, 0xFFFFFFFF) = 0xFFFFFFFF (-1)
        # umin(10, 0xFFFFFFFF) = 10
        self.mem.write_u32(4, 10)
        ret, ok = ref_buffer_atomic_umin(self.mem, self.desc, soffset=0, vaddr=4, cmp_val=0xFFFFFFFF, glc=True)
        self.assertTrue(ok)
        self.assertEqual(ret, 10)
        self.assertEqual(self.mem.read_u32(4), 10)

        # 0x80000000 (2^31) vs 5:
        # In unsigned: 5 < 0x80000000 -> min is 5
        self.mem.write_u32(8, 0x80000000)
        ret3, ok3 = ref_buffer_atomic_umin(self.mem, self.desc, soffset=0, vaddr=8, cmp_val=5, glc=True)
        self.assertTrue(ok3)
        self.assertEqual(ret3, 0x80000000)
        self.assertEqual(self.mem.read_u32(8), 5)

    def test_unsigned_min_extremes(self) -> None:
        # 0 vs UINT32_MAX -> min is 0
        self.mem.write_u32(12, UINT32_MAX)
        ret, ok = ref_buffer_atomic_umin(self.mem, self.desc, soffset=0, vaddr=12, cmp_val=UINT32_MIN, glc=True)
        self.assertTrue(ok)
        self.assertEqual(ret, UINT32_MAX)
        self.assertEqual(self.mem.read_u32(12), UINT32_MIN)

    def test_oob_suppression_and_glc_zero(self) -> None:
        # Buffer capacity is 1024, access at 1024 -> OOB
        self.mem.write_u32(100, 42)
        ret, ok = ref_buffer_atomic_umin(self.mem, self.desc, soffset=0, vaddr=1024, cmp_val=5, glc=True)
        self.assertFalse(ok)
        self.assertEqual(ret, 0)  # glc on OOB returns 0


if __name__ == "__main__":
    unittest.main()
