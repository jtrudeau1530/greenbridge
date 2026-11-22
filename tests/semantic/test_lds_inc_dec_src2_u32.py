from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_inc_dec_src2_u32.

Opcodes:
  - ds_inc_src2_u32: Atomic wrapping 32-bit increment using LDS secondary operand LDS[B].
      tmp_a = LDS[A]
      tmp_b = LDS[B]
      LDS[A] = 0 if tmp_a >= tmp_b else (tmp_a + 1) & 0xFFFFFFFF
  - ds_dec_src2_u32: Atomic wrapping 32-bit decrement using LDS secondary operand LDS[B].
      tmp_a = LDS[A]
      tmp_b = LDS[B]
      LDS[A] = tmp_b if (tmp_a == 0 or tmp_a > tmp_b) else (tmp_a - 1) & 0xFFFFFFFF
  - ds_add_src2_u32: Atomic 32-bit addition of LDS[B] into LDS[A].
      tmp_a = LDS[A]
      tmp_b = LDS[B]
      LDS[A] = (tmp_a + tmp_b) & 0xFFFFFFFF
  - ds_sub_src2_u32: Atomic 32-bit subtraction of LDS[B] from LDS[A].
      tmp_a = LDS[A]
      tmp_b = LDS[B]
      LDS[A] = (tmp_a - tmp_b) & 0xFFFFFFFF

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing:
      Base address in bytes from VGPR vaddr.
      offset0 (8-bit immediate) and offset1 (8-bit immediate) specify address offsets.
      Address A = (vaddr + offset0 * 4) or byte offset.
      Address B = secondary address in LDS.
      Dword indices are calculated by dropping low 2 bits and wrapping to LDS dwords.
  - Non-returning: Vector general-purpose registers (VGPRs) are not modified.
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


def ref_ds_inc_src2_u32(val_a: int, val_b: int) -> int:
    """Reference ISA calculation for ds_inc_src2_u32.

    If current memory value at A (val_a) is greater than or equal to wrap
    threshold at B (val_b), reset to 0. Otherwise, increment by 1.
    """
    a = u32(val_a)
    b = u32(val_b)
    if a >= b:
        return 0
    return u32(a + 1)


def ref_ds_dec_src2_u32(val_a: int, val_b: int) -> int:
    """Reference ISA calculation for ds_dec_src2_u32.

    If current memory value at A (val_a) is 0 or strictly exceeds the wrap
    threshold at B (val_b), wrap to val_b. Otherwise, decrement by 1.
    """
    a = u32(val_a)
    b = u32(val_b)
    if a == 0 or a > b:
        return b
    return u32(a - 1)


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

    def test_inc_src2_normal_increment(self) -> None:
        self.assertEqual(ref_ds_inc_src2_u32(0, 5), 1)
        self.assertEqual(ref_ds_inc_src2_u32(1, 5), 2)
        self.assertEqual(ref_ds_inc_src2_u32(3, 5), 4)
        self.assertEqual(ref_ds_inc_src2_u32(4, 5), 5)

    def test_inc_src2_wrap_equal_and_greater(self) -> None:
        # Equal to threshold wraps to 0
        self.assertEqual(ref_ds_inc_src2_u32(5, 5), 0)
        # Greater than threshold wraps to 0
        self.assertEqual(ref_ds_inc_src2_u32(6, 5), 0)
        self.assertEqual(ref_ds_inc_src2_u32(100, 5), 0)
        self.assertEqual(ref_ds_inc_src2_u32(MASK32, 5), 0)

    def test_inc_src2_boundary_zero_and_max(self) -> None:
        # When limit is 0, any unsigned value >= 0 wraps to 0
        self.assertEqual(ref_ds_inc_src2_u32(0, 0), 0)
        self.assertEqual(ref_ds_inc_src2_u32(1, 0), 0)
        # When limit is MASK32
        self.assertEqual(ref_ds_inc_src2_u32(0, MASK32), 1)
        self.assertEqual(ref_ds_inc_src2_u32(MASK32 - 1, MASK32), MASK32)
        self.assertEqual(ref_ds_inc_src2_u32(MASK32, MASK32), 0)

    def test_dec_src2_normal_decrement(self) -> None:
        self.assertEqual(ref_ds_dec_src2_u32(5, 5), 4)
        self.assertEqual(ref_ds_dec_src2_u32(4, 5), 3)
        self.assertEqual(ref_ds_dec_src2_u32(2, 5), 1)
        self.assertEqual(ref_ds_dec_src2_u32(1, 5), 0)

    def test_dec_src2_wrap_zero_and_greater(self) -> None:
        # 0 wraps to limit b
        self.assertEqual(ref_ds_dec_src2_u32(0, 5), 5)
        # Greater than limit wraps to limit b
        self.assertEqual(ref_ds_dec_src2_u32(6, 5), 5)
        self.assertEqual(ref_ds_dec_src2_u32(100, 5), 5)
        self.assertEqual(ref_ds_dec_src2_u32(MASK32, 5), 5)

    def test_dec_src2_boundary_zero_and_max(self) -> None:
        self.assertEqual(ref_ds_dec_src2_u32(0, 0), 0)
        self.assertEqual(ref_ds_dec_src2_u32(1, 0), 0)
        self.assertEqual(ref_ds_dec_src2_u32(0, MASK32), MASK32)
        self.assertEqual(ref_ds_dec_src2_u32(MASK32, MASK32), MASK32 - 1)

    def test_add_src2_u32_arithmetic(self) -> None:
        self.assertEqual(ref_ds_add_src2_u32(10, 20), 30)
        self.assertEqual(ref_ds_add_src2_u32(0, 0), 0)
        self.assertEqual(ref_ds_add_src2_u32(MASK32, 1), 0)
        self.assertEqual(ref_ds_add_src2_u32(MASK32, 5), 4)
        self.assertEqual(ref_ds_add_src2_u32(0x80000000, 0x80000000), 0)

    def test_sub_src2_u32_arithmetic(self) -> None:
        self.assertEqual(ref_ds_sub_src2_u32(30, 10), 20)
        self.assertEqual(ref_ds_sub_src2_u32(10, 10), 0)
        self.assertEqual(ref_ds_sub_src2_u32(0, 1), MASK32)
        self.assertEqual(ref_ds_sub_src2_u32(0, 5), MASK32 - 4)
        self.assertEqual(ref_ds_sub_src2_u32(0x80000000, 1), 0x7FFFFFFF)

    def test_address_calculation_and_alignment(self) -> None:
        byte_a, byte_b, idx_a, idx_b = calc_lds_src2_addresses(
            vaddr=0x1000, offset0=2, offset1=4, lds_dwords=1024
        )
        self.assertEqual(byte_a, 0x1008)
        self.assertEqual(byte_b, 0x1010)
        self.assertEqual(idx_a, (0x1008 >> 2) & 1023)
        self.assertEqual(idx_b, (0x1010 >> 2) & 1023)

    def test_self_referencing_aliasing_oracle(self) -> None:
        # Same memory location for both operands: val_a == val_b == 4
        # inc: val_a >= val_b (4 >= 4) -> 0
        self.assertEqual(ref_ds_inc_src2_u32(4, 4), 0)
        # dec: val_a == 4, val_b == 4 -> 4 - 1 = 3
        self.assertEqual(ref_ds_dec_src2_u32(4, 4), 3)
        # dec: val_a == 0, val_b == 0 -> 0
        self.assertEqual(ref_ds_dec_src2_u32(0, 0), 0)
        # add: 4 + 4 = 8
        self.assertEqual(ref_ds_add_src2_u32(4, 4), 8)
        # sub: 4 - 4 = 0
        self.assertEqual(ref_ds_sub_src2_u32(4, 4), 0)


if __name__ == "__main__":
    unittest.main()
