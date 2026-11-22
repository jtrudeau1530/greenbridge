from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_add_sub_src2_u32.

Opcodes:
  - ds_add_src2_u32: Atomic unsigned 32-bit addition with secondary operand in LDS:
      tmp_a = LDS[A]
      tmp_b = LDS[B]
      LDS[A] = (tmp_a + tmp_b) & 0xFFFFFFFF
  - ds_sub_src2_u32: Atomic unsigned 32-bit subtraction with secondary operand in LDS:
      tmp_a = LDS[A]
      tmp_b = LDS[B]
      LDS[A] = (tmp_a - tmp_b) & 0xFFFFFFFF

Architecture & Specification:
  - AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture.
  - Scratchpad memory shared across all lanes of a workgroup.
  - Addressing:
      Byte address computed from VGPR vaddr.
      offset0 (8-bit immediate) and offset1 (8-bit immediate) specify offsets.
      Address A = (vaddr + offset0 * 4) & 0xFFFFFFFF.
      Address B = (vaddr + offset1 * 4) & 0xFFFFFFFF.
      Dword indices are obtained via (addr >> 2) & (lds_dwords - 1).
  - Non-returning: Results are written in-place to LDS; VGPRs remain unchanged.
"""

import sys
import unittest

MASK32 = 0xFFFFFFFF
DEFAULT_LDS_DWORDS = 16384  # 64 KiB default capacity in dwords (16384 * 4 bytes)
DEFAULT_NUM_LANES = 64     # Wave64 default width
DEFAULT_NUM_VGPRS = 256    # Standard GFX10 vector register file size


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def ref_ds_add_src2_u32(val_a: int, val_b: int) -> int:
    """Reference ISA calculation for ds_add_src2_u32.

    LDS[A] = (LDS[A] + LDS[B]) mod 2^32.
    """
    return u32(u32(val_a) + u32(val_b))


def ref_ds_sub_src2_u32(val_a: int, val_b: int) -> int:
    """Reference ISA calculation for ds_sub_src2_u32.

    LDS[A] = (LDS[A] - LDS[B]) mod 2^32.
    """
    return u32(u32(val_a) - u32(val_b))


def calc_lds_src2_addresses(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    lds_dwords: int = DEFAULT_LDS_DWORDS,
) -> tuple[int, int, int, int]:
    """Calculate byte addresses and dword indices for address A and address B.

    Address A = (vaddr + offset0 * 4) & MASK32
    Address B = (vaddr + offset1 * 4) & MASK32
    Dword index = (addr >> 2) & (lds_dwords - 1)

    Returns:
        tuple[int, int, int, int]: (byte_addr_a, byte_addr_b, dword_idx_a, dword_idx_b)
    """
    byte_addr_a = u32(vaddr + (offset0 * 4))
    byte_addr_b = u32(vaddr + (offset1 * 4))
    dword_idx_a = (byte_addr_a >> 2) & (lds_dwords - 1)
    dword_idx_b = (byte_addr_b >> 2) & (lds_dwords - 1)
    return byte_addr_a, byte_addr_b, dword_idx_a, dword_idx_b


class TestReferenceModelAndOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model."""

    def test_add_src2_basic_addition(self) -> None:
        self.assertEqual(ref_ds_add_src2_u32(0, 0), 0)
        self.assertEqual(ref_ds_add_src2_u32(100, 250), 350)
        self.assertEqual(ref_ds_add_src2_u32(1, 0xFFFE), 0xFFFF)
        self.assertEqual(ref_ds_add_src2_u32(0x12345678, 0x10000001), 0x22345679)

    def test_add_src2_overflow_wrapping(self) -> None:
        # Wrap around 2^32
        self.assertEqual(ref_ds_add_src2_u32(MASK32, 1), 0)
        self.assertEqual(ref_ds_add_src2_u32(MASK32, 2), 1)
        self.assertEqual(ref_ds_add_src2_u32(0x80000000, 0x80000000), 0)
        self.assertEqual(ref_ds_add_src2_u32(0xFFFFFFF0, 0x20), 0x10)

    def test_sub_src2_basic_subtraction(self) -> None:
        self.assertEqual(ref_ds_sub_src2_u32(0, 0), 0)
        self.assertEqual(ref_ds_sub_src2_u32(500, 200), 300)
        self.assertEqual(ref_ds_sub_src2_u32(100, 100), 0)
        self.assertEqual(ref_ds_sub_src2_u32(0x50000000, 0x10000000), 0x40000000)

    def test_sub_src2_underflow_wrapping(self) -> None:
        # Borrow wrap around 2^32
        self.assertEqual(ref_ds_sub_src2_u32(0, 1), MASK32)
        self.assertEqual(ref_ds_sub_src2_u32(0, 5), MASK32 - 4)
        self.assertEqual(ref_ds_sub_src2_u32(10, 20), MASK32 - 9)
        self.assertEqual(ref_ds_sub_src2_u32(0x80000000, 0x80000001), MASK32)

    def test_address_calculation_and_alignment(self) -> None:
        byte_a, byte_b, idx_a, idx_b = calc_lds_src2_addresses(
            vaddr=0x2000, offset0=3, offset1=7, lds_dwords=2048
        )
        self.assertEqual(byte_a, 0x200C)
        self.assertEqual(byte_b, 0x201C)
        self.assertEqual(idx_a, (0x200C >> 2) & 2047)
        self.assertEqual(idx_b, (0x201C >> 2) & 2047)

    def test_self_referencing_aliasing(self) -> None:
        # When A == B:
        # add doubles the value
        self.assertEqual(ref_ds_add_src2_u32(50, 50), 100)
        self.assertEqual(ref_ds_add_src2_u32(0x80000000, 0x80000000), 0)
        # sub zeroes the value
        self.assertEqual(ref_ds_sub_src2_u32(50, 50), 0)
        self.assertEqual(ref_ds_sub_src2_u32(MASK32, MASK32), 0)


if __name__ == "__main__":
    unittest.main()
