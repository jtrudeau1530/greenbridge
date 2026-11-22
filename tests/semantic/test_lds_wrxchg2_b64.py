from __future__ import annotations

import struct
import unittest
from typing import Dict, List, Optional, Tuple, Union

# LDS Architectural Constants for RDNA2 (GFX10/GFX10.3)
LDS_DEFAULT_SIZE_BYTES: int = 65536  # 64 KiB default Workgroup LDS allocation
LDS_DEFAULT_SIZE_DWORDS: int = 16384
DWORD_BYTES: int = 4
QWORD_BYTES: int = 8
ST64_MULTIPLIER: int = 64
UNIT_B64_BYTES: int = 8
UNIT_B64_ST64_BYTES: int = 8 * ST64_MULTIPLIER  # 512 bytes

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF


def split_u64_to_u32s(val: int) -> Tuple[int, int]:
    """Split 64-bit unsigned integer into (lo, hi) 32-bit words."""
    val &= MASK64
    return val & MASK32, (val >> 32) & MASK32


def combine_u32s_to_u64(lo: int, hi: int) -> int:
    """Combine (lo, hi) 32-bit words into a 64-bit unsigned integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def compute_lds_address(
    vaddr: int,
    offset: int,
    is_st64: bool = False,
    lds_size_bytes: int = LDS_DEFAULT_SIZE_BYTES,
) -> int:
    """Compute wrapped byte address in LDS for ds_wrxchg2_b64.

    Address formula: (vaddr + offset * unit) & (lds_size_bytes - 1)
    where unit is 8 bytes for standard ds_wrxchg2_b64, or 512 bytes for ds_wrxchg2st64_b64.
    """
    unit = UNIT_B64_ST64_BYTES if is_st64 else UNIT_B64_BYTES
    effective = (vaddr + offset * unit) & (lds_size_bytes - 1)
    return effective


def oracle_ds_wrxchg2_b64(
    lds_mem: bytearray,
    vaddr: int,
    data0: int,
    data1: int,
    offset0: int = 0,
    offset1: int = 0,
    is_st64: bool = False,
) -> Tuple[int, int]:
    """Reference mathematical oracle for RDNA2 ds_wrxchg2_b64.

    Simulates atomic exchange of two 64-bit elements in LDS memory.
    Returns (old_data0, old_data1).
    When addr0 == addr1, slot 0 exchange occurs first, followed by slot 1.
    """
    lds_bytes = len(lds_mem)
    addr0 = compute_lds_address(vaddr, offset0, is_st64, lds_bytes)
    addr1 = compute_lds_address(vaddr, offset1, is_st64, lds_bytes)

    # 4-byte dword alignment mask standard in DS addressing
    addr0 &= ~3
    addr1 &= ~3

    # Slot 0 atomic exchange
    addr0_next = (addr0 + 4) & (lds_bytes - 1)
    old0_lo = struct.unpack_from("<I", lds_mem, addr0)[0]
    old0_hi = struct.unpack_from("<I", lds_mem, addr0_next)[0]
    old0 = combine_u32s_to_u64(old0_lo, old0_hi)

    d0_lo, d0_hi = split_u64_to_u32s(data0)
    struct.pack_into("<I", lds_mem, addr0, d0_lo)
    struct.pack_into("<I", lds_mem, addr0_next, d0_hi)

    # Slot 1 atomic exchange
    addr1_next = (addr1 + 4) & (lds_bytes - 1)
    old1_lo = struct.unpack_from("<I", lds_mem, addr1)[0]
    old1_hi = struct.unpack_from("<I", lds_mem, addr1_next)[0]
    old1 = combine_u32s_to_u64(old1_lo, old1_hi)

    d1_lo, d1_hi = split_u64_to_u32s(data1)
    struct.pack_into("<I", lds_mem, addr1, d1_lo)
    struct.pack_into("<I", lds_mem, addr1_next, d1_hi)

    return old0, old1


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.lds = bytearray(LDS_DEFAULT_SIZE_BYTES)

    def test_basic_exchange_identity(self) -> None:
        # Seed memory at byte 0x40 and 0x48
        struct.pack_into("<Q", self.lds, 0x40, 0x1111_2222_3333_4444)
        struct.pack_into("<Q", self.lds, 0x48, 0x5555_6666_7777_8888)

        old0, old1 = oracle_ds_wrxchg2_b64(
            self.lds,
            vaddr=0x40,
            data0=0xAAAA_BBBB_CCCC_DDDD,
            data1=0xEEEE_FFFF_0000_1111,
            offset0=0,
            offset1=1,  # 0x40 + 1 * 8 = 0x48
            is_st64=False,
        )

        self.assertEqual(old0, 0x1111_2222_3333_4444)
        self.assertEqual(old1, 0x5555_6666_7777_8888)

        # Verify new memory values
        mem0 = struct.unpack_from("<Q", self.lds, 0x40)[0]
        mem1 = struct.unpack_from("<Q", self.lds, 0x48)[0]
        self.assertEqual(mem0, 0xAAAA_BBBB_CCCC_DDDD)
        self.assertEqual(mem1, 0xEEEE_FFFF_0000_1111)

    def test_offset_scaling_unit_8(self) -> None:
        # offset0=3 -> +24 bytes, offset1=7 -> +56 bytes
        vaddr = 0x100
        addr0 = vaddr + 3 * 8
        addr1 = vaddr + 7 * 8
        struct.pack_into("<Q", self.lds, addr0, 0x1234_5678_9ABC_DEF0)
        struct.pack_into("<Q", self.lds, addr1, 0xFEDC_BA98_7654_3210)

        old0, old1 = oracle_ds_wrxchg2_b64(
            self.lds,
            vaddr=vaddr,
            data0=0x0102_0304_0506_0708,
            data1=0x0807_0605_0403_0201,
            offset0=3,
            offset1=7,
            is_st64=False,
        )

        self.assertEqual(old0, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(old1, 0xFEDC_BA98_7654_3210)
        self.assertEqual(struct.unpack_from("<Q", self.lds, addr0)[0], 0x0102_0304_0506_0708)
        self.assertEqual(struct.unpack_from("<Q", self.lds, addr1)[0], 0x0807_0605_0403_0201)

    def test_st64_stride_scaling_unit_512(self) -> None:
        # st64 form scales offsets by 512 bytes (8 * 64)
        vaddr = 0x80
        addr0 = vaddr + 0 * 512  # 0x80
        addr1 = vaddr + 1 * 512  # 0x280
        struct.pack_into("<Q", self.lds, addr0, 0xCAFE_BABE_0000_0001)
        struct.pack_into("<Q", self.lds, addr1, 0xDEAD_BEEF_0000_0002)

        old0, old1 = oracle_ds_wrxchg2_b64(
            self.lds,
            vaddr=vaddr,
            data0=0x1111_1111_1111_1111,
            data1=0x2222_2222_2222_2222,
            offset0=0,
            offset1=1,
            is_st64=True,
        )

        self.assertEqual(old0, 0xCAFE_BABE_0000_0001)
        self.assertEqual(old1, 0xDEAD_BEEF_0000_0002)
        self.assertEqual(struct.unpack_from("<Q", self.lds, addr0)[0], 0x1111_1111_1111_1111)
        self.assertEqual(struct.unpack_from("<Q", self.lds, addr1)[0], 0x2222_2222_2222_2222)

    def test_lds_memory_wraparound_boundary(self) -> None:
        # Wrap around 64 KiB LDS boundary
        vaddr = LDS_DEFAULT_SIZE_BYTES - 8  # 0xFFF8
        offset0 = 0  # 0xFFF8
        offset1 = 1  # 0xFFF8 + 8 = 0x10000 -> wraps to 0x0000
        struct.pack_into("<Q", self.lds, 0xFFF8, 0x9999_8888_7777_6666)
        struct.pack_into("<Q", self.lds, 0x0000, 0x3333_2222_1111_0000)

        old0, old1 = oracle_ds_wrxchg2_b64(
            self.lds,
            vaddr=vaddr,
            data0=0xAAAA_AAAA_AAAA_AAAA,
            data1=0xBBBB_BBBB_BBBB_BBBB,
            offset0=offset0,
            offset1=offset1,
            is_st64=False,
        )

        self.assertEqual(old0, 0x9999_8888_7777_6666)
        self.assertEqual(old1, 0x3333_2222_1111_0000)
        self.assertEqual(struct.unpack_from("<Q", self.lds, 0xFFF8)[0], 0xAAAA_AAAA_AAAA_AAAA)
        self.assertEqual(struct.unpack_from("<Q", self.lds, 0x0000)[0], 0xBBBB_BBBB_BBBB_BBBB)

    def test_same_address_collision_sequential_semantics(self) -> None:
        # Both offset0 and offset1 target the exact same address
        vaddr = 0x200
        addr = vaddr + 2 * 8  # 0x210
        struct.pack_into("<Q", self.lds, addr, 0x0000_0000_FEED_FACE)

        d0 = 0x1111_2222_3333_4444
        d1 = 0x5555_6666_7777_8888
        old0, old1 = oracle_ds_wrxchg2_b64(
            self.lds,
            vaddr=vaddr,
            data0=d0,
            data1=d1,
            offset0=2,
            offset1=2,
            is_st64=False,
        )

        # Slot 0 reads initial memory (0xFEEDFACE) and writes d0
        self.assertEqual(old0, 0x0000_0000_FEED_FACE)
        # Slot 1 reads what slot 0 just wrote (d0) and writes d1
        self.assertEqual(old1, d0)
        # Final memory holds d1
        self.assertEqual(struct.unpack_from("<Q", self.lds, addr)[0], d1)

    def test_dword_little_endian_layout(self) -> None:
        # Check that dwords are stored in little-endian order: low dword at addr, high dword at addr+4
        val = 0x1234_5678_9ABC_DEF0
        d_lo, d_hi = split_u64_to_u32s(val)
        self.assertEqual(d_lo, 0x9ABC_DEF0)
        self.assertEqual(d_hi, 0x1234_5678)

        oracle_ds_wrxchg2_b64(self.lds, vaddr=0x20, data0=val, data1=0, offset0=0, offset1=1)
        stored_lo = struct.unpack_from("<I", self.lds, 0x20)[0]
        stored_hi = struct.unpack_from("<I", self.lds, 0x24)[0]
        self.assertEqual(stored_lo, 0x9ABC_DEF0)
        self.assertEqual(stored_hi, 0x1234_5678)

    def test_boundary_bit_patterns(self) -> None:
        for val0, val1 in [
            (0, MASK64),
            (MASK64, 0),
            (0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA),
            (0x8000_0000_0000_0000, 0x7FFF_FFFF_FFFF_FFFF),
        ]:
            with self.subTest(val0=hex(val0), val1=hex(val1)):
                old0, old1 = oracle_ds_wrxchg2_b64(
                    self.lds, vaddr=0x10, data0=val0, data1=val1, offset0=0, offset1=1
                )
                self.assertEqual(struct.unpack_from("<Q", self.lds, 0x10)[0], val0)
                self.assertEqual(struct.unpack_from("<Q", self.lds, 0x18)[0], val1)


if __name__ == "__main__":
    unittest.main()
