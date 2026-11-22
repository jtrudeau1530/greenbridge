"""Semantic interpreter tests for LDS atomic bitwise OR operations (ds_or_b32, ds_or_rtn_b32, ds_or_b64, ds_or_rtn_b64).

Standard library only; no GPU, no LLVM, no external tools, no game files.

Run directly:
    python -I tests/semantic/test_lds_atomic_or.py
Or via unittest discovery:
    python -m unittest discover -s tests/semantic -v

Background & Specification Rules:
  * AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
    - LDS is an on-chip, software-managed scratchpad memory shared across
      all work-items (threads/lanes) in a workgroup.
    - Single 32-bit Atomic OR:
        - `ds_or_b32 vaddr, vdata [offset:N]`:
            Non-returning form:
              eff_byte = (vaddr + offset) & 0xFFFFFFFF
              dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
              LDS[dword_idx] = (LDS[dword_idx] | vdata) & 0xFFFFFFFF
            The pre-atomic memory contents are discarded.
        - `ds_or_rtn_b32 vdst, vaddr, vdata [offset:N]`:
            Returning form:
              eff_byte = (vaddr + offset) & 0xFFFFFFFF
              dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
              old_val = LDS[dword_idx]
              LDS[dword_idx] = (old_val | vdata) & 0xFFFFFFFF
              vdst = old_val
            Preserves the old value before the OR into destination VGPR vdst.
    - 64-bit Atomic OR:
        - `ds_or_b64 vaddr, v[data_lo:data_hi] [offset:N]`:
            Non-returning 64-bit bitwise OR on two consecutive 32-bit dwords (low dword first):
              eff_byte = (vaddr + offset) & 0xFFFFFFFF
              dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
              LDS[dword_idx]   |= vdata_lo
              LDS[dword_idx+1] |= vdata_hi
        - `ds_or_rtn_b64 v[dst_lo:dst_hi], vaddr, v[data_lo:data_hi] [offset:N]`:
            Returning form: writes prior 64 bits into destination VGPR pair v[dst_lo:dst_hi].
    - Active lane masking:
        - LDS instructions are vector operations governed by the execution mask (EXEC).
        - Inactive lanes do not mutate LDS storage and leave destination VGPRs unmodified.
    - Hardware wrapping & alignment:
        - Byte addresses are shifted right by 2, dropping unaligned low bits [1:0].
        - The resulting dword index is masked with (lds_dwords - 1), wrapping out-of-bounds
          offsets within the allocated power-of-two LDS capacity.
    - Mathematical properties of bitwise OR:
        - Identity: x | 0 == x.
        - Dominant / Saturation element: x | 0xFFFFFFFF == 0xFFFFFFFF.
        - Idempotence: x | x == x.
        - Commutative & Associative: (x | y) == (y | x), (x | y) | z == x | (y | z).
        - Monotonicity: (x | y) & x == x (bits only transition 0 -> 1, never 1 -> 0).
        - Bitmask union: fundamental for flag aggregation, multi-lane barrier signals,
          and sparse occupancy tracking.
"""
from __future__ import annotations

import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
DEFAULT_LDS_DWORDS = 16384  # 64 KiB default capacity in dwords


class UnsupportedConstruct(NotImplementedError):
    """The module uses an unmodelled SPIR-V opcode, type, or storage class."""


class WorkgroupBoundsError(ValueError):
    """Workgroup dimension or memory bounds violation."""


def reference_atomic_or_b32(current_val: int, or_mask: int) -> tuple[int, int]:
    """Reference 32-bit atomic OR: returns (old_val, updated_mem_val)."""
    old = current_val & MASK32
    new = (old | or_mask) & MASK32
    return old, new


def reference_atomic_or_b64(current_val: int, or_mask: int) -> tuple[int, int]:
    """Reference 64-bit atomic OR: returns (old_val, updated_mem_val)."""
    old = current_val & MASK64
    new = (old | or_mask) & MASK64
    return old, new


def split_u64_to_u32(val: int) -> tuple[int, int]:
    """Splits a 64-bit unsigned integer into (low_u32, high_u32)."""
    val = val & MASK64
    return val & MASK32, (val >> 32) & MASK32


def combine_u32_to_u64(lo: int, hi: int) -> int:
    """Combines two 32-bit unsigned integers (low_u32, high_u32) into a 64-bit unsigned integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def calc_lds_byte_offset(
    vaddr: int,
    byte_offset: int = 0,
    lds_dwords: int = DEFAULT_LDS_DWORDS,
) -> tuple[int, int, int]:
    """Reference ISA calculation for single LDS address: (effective_byte, dword_idx, byte_in_dword)."""
    vaddr = vaddr & MASK32
    byte_offset = byte_offset & 0xFFFF  # 16-bit unsigned offset in RDNA2 DS instructions
    eff_byte = (vaddr + byte_offset) & MASK32
    dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
    byte_in_dword = eff_byte & 3
    return eff_byte, dword_idx, byte_in_dword


def calc_lds_dual_offsets(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    lds_dwords: int = DEFAULT_LDS_DWORDS,
    st64: bool = False,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Reference ISA calculation for dual LDS addresses.

    Returns:
        ((eff_byte0, dword_idx0), (eff_byte1, dword_idx1))
    """
    vaddr = vaddr & MASK32
    unit = 256 if st64 else 4
    off0_bytes = (offset0 & 0xFF) * unit
    off1_bytes = (offset1 & 0xFF) * unit
    eff_byte0 = (vaddr + off0_bytes) & MASK32
    eff_byte1 = (vaddr + off1_bytes) & MASK32
    dword_idx0 = (eff_byte0 >> 2) & (lds_dwords - 1)
    dword_idx1 = (eff_byte1 >> 2) & (lds_dwords - 1)
    return (eff_byte0, dword_idx0), (eff_byte1, dword_idx1)


class TestAtomicOrReferenceMath(unittest.TestCase):
    """Verifies reference bitwise OR properties, algebraic laws, and edge cases."""

    def test_identity_zero_mask_b32(self):
        cases = [0, 1, 0x12345678, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                old, new = reference_atomic_or_b32(val, 0)
                self.assertEqual(old, val)
                self.assertEqual(new, val)

    def test_saturation_all_ones_b32(self):
        cases = [0, 1, 0x55555555, 0xAAAAAAAA, 0x80000000, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                old, new = reference_atomic_or_b32(val, MASK32)
                self.assertEqual(old, val)
                self.assertEqual(new, MASK32)

    def test_idempotence_b32(self):
        cases = [0, 0x10203040, 0x55AA55AA, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                old, new = reference_atomic_or_b32(val, val)
                self.assertEqual(old, val)
                self.assertEqual(new, val)

    def test_commutativity_and_associativity_b32(self):
        a, b = 0x3F2A1C8E, 0x9D4E7B01
        _, ab = reference_atomic_or_b32(a, b)
        _, ba = reference_atomic_or_b32(b, a)
        self.assertEqual(ab, ba)

        c = 0x55AA33CC
        _, step_ab = reference_atomic_or_b32(a, b)
        _, ab_c = reference_atomic_or_b32(step_ab, c)

        _, step_bc = reference_atomic_or_b32(b, c)
        _, a_bc = reference_atomic_or_b32(a, step_bc)
        self.assertEqual(ab_c, a_bc)

    def test_monotonicity_b32(self):
        # Bitwise OR can only set bits from 0 to 1, never clears bits
        cases = [
            (0, 0x00000001),
            (0x10000000, 0x00000001),
            (0x0F0F0F0F, 0xF0F0F0F0),
            (0x7FFFFFFF, 0x80000000),
        ]
        for initial, mask in cases:
            with self.subTest(initial=hex(initial), mask=hex(mask)):
                old, new = reference_atomic_or_b32(initial, mask)
                self.assertEqual(old, initial)
                # All bits of initial must still be present in new
                self.assertEqual(new & initial, initial)
                # All bits of mask must be present in new
                self.assertEqual(new & mask, mask)
                self.assertGreaterEqual(bin(new).count("1"), bin(initial).count("1"))

    def test_disjoint_bit_patterns_b32(self):
        # Alternating bit patterns
        old, new = reference_atomic_or_b32(0x55555555, 0xAAAAAAAA)
        self.assertEqual(old, 0x55555555)
        self.assertEqual(new, MASK32)

        # Upper and lower half-words
        old, new = reference_atomic_or_b32(0xFFFF0000, 0x0000FFFF)
        self.assertEqual(old, 0xFFFF0000)
        self.assertEqual(new, MASK32)

        # Nibble interleaving
        old, new = reference_atomic_or_b32(0x0F0F0F0F, 0xF0F0F0F0)
        self.assertEqual(old, 0x0F0F0F0F)
        self.assertEqual(new, MASK32)

    def test_sequential_bit_accumulation_b32(self):
        # Accumulate 32 individual bit flags across 32 sequential atomic operations
        current = 0
        for bit in range(32):
            flag = 1 << bit
            old, current = reference_atomic_or_b32(current, flag)
            self.assertEqual(old & flag, 0)
            self.assertEqual(current & flag, flag)
        self.assertEqual(current, MASK32)

    def test_sparse_bit_combinations_b32(self):
        # Non-contiguous bits
        val = 0
        steps = [1 << 3, 1 << 7, 1 << 15, 1 << 31]
        expected = (1 << 3) | (1 << 7) | (1 << 15) | (1 << 31)
        for s in steps:
            _, val = reference_atomic_or_b32(val, s)
        self.assertEqual(val, expected)

    def test_reference_atomic_or_b64_properties(self):
        init = 0x0123456789ABCDEF
        mask = 0xFEDCBA9876543210
        old, mem = reference_atomic_or_b64(init, mask)
        self.assertEqual(old, init)
        self.assertEqual(mem, init | mask)

        # 64-bit zero identity
        old_id, mem_id = reference_atomic_or_b64(init, 0)
        self.assertEqual(old_id, init)
        self.assertEqual(mem_id, init)

        # 64-bit all-ones saturation
        old_sat, mem_sat = reference_atomic_or_b64(init, MASK64)
        self.assertEqual(old_sat, init)
        self.assertEqual(mem_sat, MASK64)

        # 64-bit idempotence
        old_idem, mem_idem = reference_atomic_or_b64(init, init)
        self.assertEqual(old_idem, init)
        self.assertEqual(mem_idem, init)

    def test_reference_atomic_or_b64_cross_dword_patterns(self):
        # Low dword only mutated
        init = 0xAAAAAAAA_00000000
        mask = 0x00000000_FFFFFFFF
        _, res = reference_atomic_or_b64(init, mask)
        self.assertEqual(res, 0xAAAAAAAA_FFFFFFFF)

        # High dword only mutated
        init = 0x00000000_55555555
        mask = 0xFFFFFFFF_00000000
        _, res = reference_atomic_or_b64(init, mask)
        self.assertEqual(res, 0xFFFFFFFF_55555555)

        # Cross dword boundaries
        lo_word = 0x00000000_80000000  # bit 31
        hi_word = 0x00000001_00000000  # bit 32
        _, res = reference_atomic_or_b64(lo_word, hi_word)
        self.assertEqual(res, 0x00000001_80000000)

    def test_split_and_combine_u64_u32(self):
        v64 = 0xFEDCBA9876543210
        lo, hi = split_u64_to_u32(v64)
        self.assertEqual(lo, 0x76543210)
        self.assertEqual(hi, 0xFEDCBA98)
        self.assertEqual(combine_u32_to_u64(lo, hi), v64)


class TestLdsAddressingAndOffsets(unittest.TestCase):
    """Verifies single and dual LDS byte offsets, dword truncation, st64 strides, and wrapping."""

    def test_aligned_dword_addressing(self):
        lds_dwords = 256
        for dword in (0, 1, 2, 16, 64, 127, 255):
            byte_addr = dword * 4
            eff, dword_idx, byte_pos = calc_lds_byte_offset(byte_addr, 0, lds_dwords)
            self.assertEqual(eff, byte_addr)
            self.assertEqual(dword_idx, dword)
            self.assertEqual(byte_pos, 0)

    def test_unaligned_byte_truncation(self):
        lds_dwords = 128
        base_aligned = 64  # dword 16
        for b in (0, 1, 2, 3):
            eff, dword_idx, byte_pos = calc_lds_byte_offset(base_aligned, b, lds_dwords)
            self.assertEqual(eff, base_aligned + b)
            self.assertEqual(dword_idx, 16)
            self.assertEqual(byte_pos, b)

    def test_offset_16bit_masking(self):
        lds_dwords = 512
        # RDNA2 DS instruction offset field is 16-bit unsigned (0x0000 - 0xFFFF)
        eff, dword_idx, byte_pos = calc_lds_byte_offset(0, 0x10008, lds_dwords)
        self.assertEqual(eff, 8)
        self.assertEqual(dword_idx, 2)
        self.assertEqual(byte_pos, 0)

    def test_dual_offset_unit_strides(self):
        (eff0, idx0), (eff1, idx1) = calc_lds_dual_offsets(0, 3, 10, lds_dwords=1024, st64=False)
        self.assertEqual(eff0, 12)
        self.assertEqual(idx0, 3)
        self.assertEqual(eff1, 40)
        self.assertEqual(idx1, 10)

    def test_dual_offset_st64_stride(self):
        (eff0, idx0), (eff1, idx1) = calc_lds_dual_offsets(0, 2, 5, lds_dwords=1024, st64=True)
        self.assertEqual(eff0, 512)
        self.assertEqual(idx0, 128)
        self.assertEqual(eff1, 1280)
        self.assertEqual(idx1, 320)

    def test_boundary_power_of_two_wrapping(self):
        lds_dwords = 64  # 256 bytes total
        # Byte address 256 wraps to dword 0
        _, idx0, _ = calc_lds_byte_offset(256, 0, lds_dwords)
        self.assertEqual(idx0, 0)
        # Byte address 260 wraps to dword 1
        _, idx1, _ = calc_lds_byte_offset(256, 4, lds_dwords)
        self.assertEqual(idx1, 1)
        # 32-bit unsigned overflow: 0xFFFFFFFC + 4 = 0x00000000 -> dword 0
        _, idx_wrap, _ = calc_lds_byte_offset(0xFFFFFFFC, 4, lds_dwords)
        self.assertEqual(idx_wrap, 0)


if __name__ == "__main__":
    unittest.main()
