from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_add_src2_u32.

Opcode:
  - ds_add_src2_u32: Atomic 32-bit addition of LDS secondary operand LDS[B] into LDS[A].
      tmp_a = LDS[A]
      tmp_b = LDS[B]
      LDS[A] = (tmp_a + tmp_b) & 0xFFFFFFFF

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing:
      Base address in bytes from VGPR vaddr.
      offset0 (8-bit immediate) and offset1 (8-bit immediate) specify address offsets in dwords (offset * 4).
      Address A = (vaddr + offset0 * 4).
      Address B = (vaddr + offset1 * 4).
      Dword indices are calculated by dropping low 2 bits and wrapping to LDS capacity.
  - Non-returning: Vector general-purpose registers (VGPRs) are not modified.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import struct
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
DEFAULT_LDS_DWORDS = 16384  # 64 KiB default capacity in dwords (16384 * 4 bytes)
DEFAULT_NUM_LANES = 64     # Wave64 default width
DEFAULT_NUM_VGPRS = 256    # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS = 106    # Standard GFX10 scalar register file size

# IEEE 754 Single-Precision Boundary Constants
POS_ZERO_F32 = 0x00000000
NEG_ZERO_F32 = 0x80000000
POS_INF_F32 = 0x7F800000
NEG_INF_F32 = 0xFF800000
QNAN_F32 = 0x7FC00000
SNAN_F32 = 0x7F800001
MIN_NORMAL_F32 = 0x00800000
MIN_SUBNORMAL_F32 = 0x00000001
MAX_SUBNORMAL_F32 = 0x007FFFFF
MAX_NORMAL_F32 = 0x7F7FFFFF
ONE_F32 = 0x3F800000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def bits_to_f32(bits: int) -> float:
    """Interpret 32-bit unsigned integer as IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Pack IEEE 754 single-precision float into 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def is_f32_nan(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 NaN."""
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def is_f32_subnormal(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 subnormal float."""
    return (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0


def ref_ds_add_src2_u32(val_a: int, val_b: int) -> int:
    """Reference ISA calculation for ds_add_src2_u32.

    LDS[A] = (LDS[A] + LDS[B]) mod 2^32.
    """
    return u32(u32(val_a) + u32(val_b))


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


# ==============================================================================
# Unit Tests
# ==============================================================================

class TestReferenceModelAndOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model."""

    def test_basic_addition(self) -> None:
        self.assertEqual(ref_ds_add_src2_u32(10, 20), 30)
        self.assertEqual(ref_ds_add_src2_u32(100, 250), 350)
        self.assertEqual(ref_ds_add_src2_u32(0x12345678, 0x11111111), 0x23456789)

    def test_addition_with_zero(self) -> None:
        self.assertEqual(ref_ds_add_src2_u32(0, 0), 0)
        self.assertEqual(ref_ds_add_src2_u32(42, 0), 42)
        self.assertEqual(ref_ds_add_src2_u32(0, 42), 42)

    def test_addition_wrap_around_overflow(self) -> None:
        self.assertEqual(ref_ds_add_src2_u32(MASK32, 1), 0)
        self.assertEqual(ref_ds_add_src2_u32(MASK32, 5), 4)
        self.assertEqual(ref_ds_add_src2_u32(0x80000000, 0x80000000), 0)
        self.assertEqual(ref_ds_add_src2_u32(MASK32, MASK32), MASK32 - 1)

    def test_address_calculation_and_alignment(self) -> None:
        byte_a, byte_b, idx_a, idx_b = calc_lds_src2_addresses(
            vaddr=0x1000, offset0=2, offset1=4, lds_dwords=1024
        )
        self.assertEqual(byte_a, 0x1008)
        self.assertEqual(byte_b, 0x1010)
        self.assertEqual(idx_a, (0x1008 >> 2) & 1023)
        self.assertEqual(idx_b, (0x1010 >> 2) & 1023)

    def test_self_referencing_aliasing_oracle(self) -> None:
        # A and B point to the same value -> doubling
        self.assertEqual(ref_ds_add_src2_u32(4, 4), 8)
        self.assertEqual(ref_ds_add_src2_u32(0x80000000, 0x80000000), 0)


if __name__ == "__main__":
    unittest.main()
