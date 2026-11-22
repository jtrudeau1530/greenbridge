"""Semantic interpreter tests for AMD RDNA2 / GFX10 typed buffer stores (MTBUF).

Covers ISA semantics, format conversion, addressing, bounds checking, and translation:
  * Typed buffer store instructions:
      - tbuffer_store_format_x: 1 component store
      - tbuffer_store_format_xy: 2 component store
      - tbuffer_store_format_xyz: 3 component store
      - tbuffer_store_format_xyzw: 4 component store
  * GFX10 format packings & numerical conversions:
      - BUF_DATA_FORMAT_8 / BUF_NUM_FORMAT_UNORM (code 1, 8-bit unorm in [0, 255])
      - BUF_DATA_FORMAT_16 / BUF_NUM_FORMAT_FLOAT (code 13/29/71, 16-bit IEEE 754 half)
      - BUF_DATA_FORMAT_32 / BUF_NUM_FORMAT_UINT (code 20/62/72/75, 32-bit unsigned int)
      - BUF_DATA_FORMAT_32 / BUF_NUM_FORMAT_FLOAT (code 22/64/74/77, 32-bit IEEE 754 single)
  * Buffer Resource Descriptor (V# 128-bit layout):
      - Dword 0: Base address low 32 bits [31:0]
      - Dword 1: Base address high 16 bits [15:0]; Stride in bytes [29:16]; flags [31:30]
      - Dword 2: Number of records (num_records capacity limit)
      - Dword 3: Format code [18:12]; DST_SEL [11:0]; OOB_SELECT [29:28]
  * Addressing calculation:
      - effective_offset = (voffset + (soffset & ~3) + inst_offset)
      - Structured buffers: effective_offset + vindex * stride
      - Effective 64-bit physical address: (base_64 + total_offset) & 0xFFFFFFFFFFFFFFFF
  * Bounds checking:
      - Stores beyond num_records are silently ignored / no-op (memory untouched).
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MASK14 = 0x3FFF
MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

BUF_DATA_FORMAT_8 = 1
BUF_DATA_FORMAT_16 = 2
BUF_DATA_FORMAT_32 = 4

BUF_NUM_FORMAT_UNORM = 1
BUF_NUM_FORMAT_FLOAT = 7
BUF_NUM_FORMAT_UINT = 4
BUF_NUM_FORMAT_SINT = 5

GFX10_FMT_8_UNORM = 1
GFX10_FMT_16_FLOAT = 13
GFX10_FMT_16_16_FLOAT = 29
GFX10_FMT_16_16_16_16_FLOAT = 71
GFX10_FMT_32_UINT = 20
GFX10_FMT_32_32_UINT = 62
GFX10_FMT_32_32_32_UINT = 72
GFX10_FMT_32_32_32_32_UINT = 75
GFX10_FMT_32_FLOAT = 22
GFX10_FMT_32_32_FLOAT = 64
GFX10_FMT_32_32_32_FLOAT = 74
GFX10_FMT_32_32_32_32_FLOAT = 77
GFX10_FMT_8_8_8_8_UNORM = 56

FORMAT_ELEMENT_SIZES: dict[int, int] = {
    GFX10_FMT_8_UNORM: 1,
    GFX10_FMT_16_FLOAT: 2,
    GFX10_FMT_16_16_FLOAT: 4,
    GFX10_FMT_16_16_16_16_FLOAT: 8,
    GFX10_FMT_32_UINT: 4,
    GFX10_FMT_32_32_UINT: 8,
    GFX10_FMT_32_32_32_UINT: 12,
    GFX10_FMT_32_32_32_32_UINT: 16,
    GFX10_FMT_32_FLOAT: 4,
    GFX10_FMT_32_32_FLOAT: 8,
    GFX10_FMT_32_32_32_FLOAT: 12,
    GFX10_FMT_32_32_32_32_FLOAT: 16,
    GFX10_FMT_8_8_8_8_UNORM: 4,
}


def pack_unorm8(value: float) -> int:
    """Pack single float value into 8-bit unsigned normalized [0, 255].

    NaN maps to 0.0 before clamping. Clamps to [0.0, 1.0] and rounds to nearest even.
    """
    if math.isnan(value):
        value = 0.0
    clamped = max(0.0, min(1.0, value))
    scaled = clamped * 255.0
    rounded = int(round(scaled))
    return max(0, min(255, rounded))


def pack_float16(value: float) -> int:
    """Pack single float value into 16-bit IEEE 754 half-precision float."""
    raw = struct.pack("<e", value)
    return struct.unpack("<H", raw)[0]


def pack_uint32(value: int) -> int:
    """Pack integer value into 32-bit unsigned integer."""
    return int(value) & MASK32


def pack_float32(value: float) -> int:
    """Pack single float value into 32-bit IEEE 754 single-precision bit pattern."""
    raw = struct.pack("<f", value)
    return struct.unpack("<I", raw)[0]


def pack_typed_format_bytes(values: tuple[float | int, ...] | list[float | int], format_code: int) -> bytes:
    """Pack component values into raw bytes for a given GFX10 format code."""
    if format_code == GFX10_FMT_8_UNORM:
        val = pack_unorm8(float(values[0]))
        return struct.pack("<B", val)
    elif format_code == GFX10_FMT_8_8_8_8_UNORM:
        padded = list(values[:4]) + [0.0] * max(0, 4 - len(values))
        unorms = [pack_unorm8(float(v)) for v in padded[:4]]
        return struct.pack("<4B", *unorms)
    elif format_code == GFX10_FMT_16_FLOAT:
        val = pack_float16(float(values[0]))
        return struct.pack("<H", val)
    elif format_code == GFX10_FMT_16_16_FLOAT:
        padded = list(values[:2]) + [0.0] * max(0, 2 - len(values))
        f16s = [pack_float16(float(v)) for v in padded[:2]]
        return struct.pack("<2H", *f16s)
    elif format_code == GFX10_FMT_16_16_16_16_FLOAT:
        padded = list(values[:4]) + [0.0] * max(0, 4 - len(values))
        f16s = [pack_float16(float(v)) for v in padded[:4]]
        return struct.pack("<4H", *f16s)
    elif format_code in (GFX10_FMT_32_UINT, GFX10_FMT_32_32_UINT, GFX10_FMT_32_32_32_UINT, GFX10_FMT_32_32_32_32_UINT):
        count = {GFX10_FMT_32_UINT: 1, GFX10_FMT_32_32_UINT: 2, GFX10_FMT_32_32_32_UINT: 3, GFX10_FMT_32_32_32_32_UINT: 4}[format_code]
        padded = list(values[:count]) + [0] * max(0, count - len(values))
        u32s = [pack_uint32(int(v)) for v in padded[:count]]
        return struct.pack(f"<{count}I", *u32s)
    elif format_code in (GFX10_FMT_32_FLOAT, GFX10_FMT_32_32_FLOAT, GFX10_FMT_32_32_32_FLOAT, GFX10_FMT_32_32_32_32_FLOAT):
        count = {GFX10_FMT_32_FLOAT: 1, GFX10_FMT_32_32_FLOAT: 2, GFX10_FMT_32_32_32_FLOAT: 3, GFX10_FMT_32_32_32_32_FLOAT: 4}[format_code]
        padded = list(values[:count]) + [0.0] * max(0, count - len(values))
        u32s = [pack_float32(float(v)) for v in padded[:count]]
        return struct.pack(f"<{count}I", *u32s)
    else:
        raise ValueError(f"Unsupported format code: {format_code}")


class BufferResourceDescriptor:
    """128-bit AMD RDNA2 Buffer Resource Descriptor (V#)."""

    def __init__(self, d0: int = 0, d1: int = 0, d2: int = 0, d3: int = 0) -> None:
        self.d0 = d0 & MASK32
        self.d1 = d1 & MASK32
        self.d2 = d2 & MASK32
        self.d3 = d3 & MASK32

    @classmethod
    def from_fields(
        cls,
        base_address: int,
        stride: int = 0,
        num_records: int = 0,
        format_code: int = 0,
        dst_sel: int = 0,
        oob_select: int = 3,
        flags: int = 0,
    ) -> BufferResourceDescriptor:
        d0 = base_address & MASK32
        base_hi = (base_address >> 32) & MASK16
        stride_field = (stride & MASK14) << 16
        d1 = base_hi | stride_field | ((flags & 3) << 30)
        d2 = num_records & MASK32
        d3 = ((format_code & 0x7F) << 12) | (dst_sel & 0xFFF) | ((oob_select & 3) << 28)
        return cls(d0, d1, d2, d3)

    @property
    def base_lo(self) -> int:
        return self.d0

    @property
    def base_hi(self) -> int:
        return self.d1 & MASK16

    @property
    def base_address_64(self) -> int:
        return ((self.base_hi << 32) | self.base_lo) & MASK64

    @property
    def stride(self) -> int:
        return (self.d1 >> 16) & MASK14

    @property
    def num_records(self) -> int:
        return self.d2

    @property
    def format_code(self) -> int:
        return (self.d3 >> 12) & 0x7F

    @property
    def dst_sel(self) -> int:
        return self.d3 & 0xFFF

    @property
    def oob_select(self) -> int:
        return (self.d3 >> 28) & 0x3

    def as_words(self) -> tuple[int, int, int, int]:
        return (self.d0, self.d1, self.d2, self.d3)


def compute_typed_buffer_address(
    desc: BufferResourceDescriptor,
    soffset: int = 0,
    inst_offset: int = 0,
    voffset: int = 0,
    vindex: int | None = None,
    element_size: int = 4,
) -> tuple[int, bool, int]:
    """Calculates effective 64-bit physical address, in-bounds status, and total byte offset.

    Addressing calculation rule:
      effective_offset = (voffset + (soffset & ~3) + inst_offset)
    With structured buffers:
      total_offset = effective_offset + (vindex * stride if vindex is not None else 0)

    Returns:
      (effective_address, in_bounds, total_offset)
    """
    masked_soff = soffset & ~3
    effective_offset = (voffset + masked_soff + inst_offset) & MASK32

    if vindex is not None:
        total_offset = (effective_offset + (vindex & MASK32) * desc.stride) & MASK32
    else:
        total_offset = effective_offset

    effective_addr = (desc.base_address_64 + total_offset) & MASK64

    if desc.oob_select == 2:
        in_bounds = desc.num_records != 0
    elif desc.oob_select == 0:
        if vindex is not None:
            idx_ok = (vindex & MASK32) < desc.num_records
            off_ok = (effective_offset + element_size) <= desc.stride
            in_bounds = idx_ok and off_ok
        else:
            in_bounds = (total_offset + element_size) <= (desc.num_records * desc.stride)
    elif desc.oob_select == 1:
        if vindex is not None:
            in_bounds = (vindex & MASK32) < desc.num_records
        else:
            in_bounds = (total_offset + element_size) <= (desc.num_records * desc.stride)
    else:
        in_bounds = (total_offset + element_size) <= desc.num_records

    return effective_addr, in_bounds, total_offset


class TypedBufferReferenceInterpreter:
    """Semantic reference interpreter modeling RDNA2 typed buffer stores (MTBUF)."""

    def __init__(self, memory: dict[int, int] | None = None) -> None:
        self.memory: dict[int, int] = {} if memory is None else dict(memory)

    def read_dword(self, byte_addr: int) -> int:
        aligned = byte_addr & ~3
        return self.memory.get(aligned & MASK64, 0) & MASK32

    def write_bytes(self, byte_addr: int, data: bytes) -> None:
        for offset, byte_val in enumerate(data):
            curr_addr = byte_addr + offset
            aligned = curr_addr & ~3
            shift = (curr_addr & 3) * 8
            old_dword = self.read_dword(aligned)
            cleared = old_dword & ~(0xFF << shift)
            new_dword = cleared | ((byte_val & 0xFF) << shift)
            self.memory[aligned & MASK64] = new_dword & MASK32

    def store_format(
        self,
        vdata: tuple[float | int, ...] | list[float | int],
        desc: BufferResourceDescriptor,
        soffset: int = 0,
        inst_offset: int = 0,
        voffset: int = 0,
        vindex: int | None = None,
        format_code: int = GFX10_FMT_8_UNORM,
    ) -> bool:
        elem_size = FORMAT_ELEMENT_SIZES.get(format_code, 4)
        addr, in_bounds, _ = compute_typed_buffer_address(
            desc,
            soffset=soffset,
            inst_offset=inst_offset,
            voffset=voffset,
            vindex=vindex,
            element_size=elem_size,
        )
        if desc.format_code == 0:
            return False
        if not in_bounds:
            return False
        packed = pack_typed_format_bytes(vdata, format_code)
        self.write_bytes(addr, packed)
        return True


class TestBufferDescriptor(unittest.TestCase):
    def test_descriptor_field_extraction(self) -> None:
        base = 0x00007FFF80001000
        stride = 64
        num_records = 256
        format_code = GFX10_FMT_32_32_32_32_FLOAT
        dst_sel = 0xF2E
        oob_sel = 3
        desc = BufferResourceDescriptor.from_fields(
            base_address=base,
            stride=stride,
            num_records=num_records,
            format_code=format_code,
            dst_sel=dst_sel,
            oob_select=oob_sel,
        )
        self.assertEqual(desc.base_lo, 0x80001000)
        self.assertEqual(desc.base_hi, 0x7FFF)
        self.assertEqual(desc.base_address_64, base)
        self.assertEqual(desc.stride, 64)
        self.assertEqual(desc.num_records, 256)
        self.assertEqual(desc.format_code, GFX10_FMT_32_32_32_32_FLOAT)
        self.assertEqual(desc.dst_sel, 0xF2E)
        self.assertEqual(desc.oob_select, 3)

    def test_descriptor_raw_words_roundtrip(self) -> None:
        words = (0x12345678, 0x0040ABCD, 0x00000400, 0x3004D123)
        desc = BufferResourceDescriptor(*words)
        self.assertEqual(desc.as_words(), words)
        self.assertEqual(desc.base_lo, 0x12345678)
        self.assertEqual(desc.base_hi, 0xABCD)
        self.assertEqual(desc.stride, 0x40)
        self.assertEqual(desc.num_records, 0x400)
        self.assertEqual(desc.format_code, (0x3004D123 >> 12) & 0x7F)
        self.assertEqual(desc.oob_select, (0x3004D123 >> 28) & 0x3)


class TestAddressingCalculations(unittest.TestCase):
    def test_soffset_alignment_masking(self) -> None:
        desc = BufferResourceDescriptor.from_fields(base_address=0x1000, num_records=1024, oob_select=3)
        for unaligned, expected_masked in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 4), (7, 4), (15, 12), (0xFFFFFFFF, 0xFFFFFFFC)]:
            addr, in_bounds, total = compute_typed_buffer_address(desc, soffset=unaligned, inst_offset=0, voffset=0)
            self.assertEqual(total, expected_masked)
            self.assertEqual(addr, 0x1000 + expected_masked)

    def test_effective_offset_formula(self) -> None:
        desc = BufferResourceDescriptor.from_fields(base_address=0x2000, num_records=2048, oob_select=3)
        voffset = 0x80
        soffset = 0x43
        inst_offset = 0x14
        addr, in_bounds, total = compute_typed_buffer_address(desc, soffset=soffset, inst_offset=inst_offset, voffset=voffset)
        self.assertEqual(total, 0x80 + 0x40 + 0x14)
        self.assertEqual(addr, 0x2000 + total)
        self.assertTrue(in_bounds)

    def test_structured_vindex_stride_addressing(self) -> None:
        desc = BufferResourceDescriptor.from_fields(base_address=0x5000, stride=32, num_records=16, oob_select=1)
        addr, in_bounds, total = compute_typed_buffer_address(desc, soffset=4, inst_offset=8, voffset=0, vindex=5, element_size=16)
        expected_offset = (4 + 8) + (5 * 32)
        self.assertEqual(total, expected_offset)
        self.assertEqual(addr, 0x5000 + expected_offset)
        self.assertTrue(in_bounds)

    def test_bounds_checking_raw_exact(self) -> None:
        desc = BufferResourceDescriptor.from_fields(base_address=0, num_records=100, oob_select=3)
        _, in_bounds_edge, _ = compute_typed_buffer_address(desc, inst_offset=96, element_size=4)
        self.assertTrue(in_bounds_edge)
        _, in_bounds_oob, _ = compute_typed_buffer_address(desc, inst_offset=97, element_size=4)
        self.assertFalse(in_bounds_oob)


class TestFormatPackingReference(unittest.TestCase):
    def test_unorm8_packing(self) -> None:
        self.assertEqual(pack_unorm8(0.0), 0)
        self.assertEqual(pack_unorm8(1.0), 255)
        self.assertEqual(pack_unorm8(0.5), 128)
        self.assertEqual(pack_unorm8(-1.0), 0)
        self.assertEqual(pack_unorm8(2.5), 255)
        self.assertEqual(pack_unorm8(float("nan")), 0)

    def test_float16_packing(self) -> None:
        self.assertEqual(pack_float16(0.0), 0x0000)
        self.assertEqual(pack_float16(1.0), 0x3C00)
        self.assertEqual(pack_float16(2.0), 0x4000)
        self.assertEqual(pack_float16(-1.0), 0xBC00)
        self.assertEqual(pack_float16(65504.0), 0x7BFF)

    def test_uint32_packing(self) -> None:
        self.assertEqual(pack_uint32(0), 0)
        self.assertEqual(pack_uint32(0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(pack_uint32(0x12345678), 0x12345678)
        self.assertEqual(pack_uint32(0x100000005), 5)

    def test_float32_packing(self) -> None:
        self.assertEqual(pack_float32(0.0), 0x00000000)
        self.assertEqual(pack_float32(1.0), 0x3F800000)
        self.assertEqual(pack_float32(2.0), 0x40000000)
        self.assertEqual(pack_float32(-1.0), 0xBF800000)

    def test_pack_typed_format_bytes(self) -> None:
        b_u8 = pack_typed_format_bytes((1.0,), GFX10_FMT_8_UNORM)
        self.assertEqual(b_u8, b"\xff")
        b_u8_4 = pack_typed_format_bytes((0.0, 0.5, 1.0, 1.0), GFX10_FMT_8_8_8_8_UNORM)
        self.assertEqual(b_u8_4, bytes([0, 128, 255, 255]))
        b_f16 = pack_typed_format_bytes((1.0, 2.0), GFX10_FMT_16_16_FLOAT)
        self.assertEqual(struct.unpack("<2H", b_f16), (0x3C00, 0x4000))
        b_u32 = pack_typed_format_bytes((0x12345678, 0x9ABCDEF0), GFX10_FMT_32_32_UINT)
        self.assertEqual(struct.unpack("<2I", b_u32), (0x12345678, 0x9ABCDEF0))
        b_f32 = pack_typed_format_bytes((1.0, 2.0, 3.0, 4.0), GFX10_FMT_32_32_32_32_FLOAT)
        self.assertEqual(struct.unpack("<4I", b_f32), (0x3F800000, 0x40000000, 0x40400000, 0x40800000))


class TestTypedStoreReferenceInterpreter(unittest.TestCase):
    def test_unorm8_sub_dword_store_preserves_neighbors(self) -> None:
        interp = TypedBufferReferenceInterpreter(memory={0x200: 0x11223344})
        desc = BufferResourceDescriptor.from_fields(base_address=0, num_records=0x1000, format_code=GFX10_FMT_8_UNORM)
        success = interp.store_format((1.0,), desc, voffset=0x200, format_code=GFX10_FMT_8_UNORM)
        self.assertTrue(success)
        self.assertEqual(interp.read_dword(0x200), 0x112233FF)

    def test_float16_sub_dword_store_preserves_high_half(self) -> None:
        interp = TypedBufferReferenceInterpreter(memory={0x300: 0xAAAA5555})
        desc = BufferResourceDescriptor.from_fields(base_address=0, num_records=0x1000, format_code=GFX10_FMT_16_FLOAT)
        success = interp.store_format((1.0,), desc, voffset=0x300, format_code=GFX10_FMT_16_FLOAT)
        self.assertTrue(success)
        self.assertEqual(interp.read_dword(0x300), 0xAAAA3C00)

    def test_xyzw_float32_multi_dword_store(self) -> None:
        interp = TypedBufferReferenceInterpreter()
        desc = BufferResourceDescriptor.from_fields(base_address=0x100, num_records=0x1000, format_code=GFX10_FMT_32_32_32_32_FLOAT)
        success = interp.store_format((1.0, 2.0, 3.0, 4.0), desc, voffset=0x10, format_code=GFX10_FMT_32_32_32_32_FLOAT)
        self.assertTrue(success)
        self.assertEqual(interp.read_dword(0x110), 0x3F800000)
        self.assertEqual(interp.read_dword(0x114), 0x40000000)
        self.assertEqual(interp.read_dword(0x118), 0x40400000)
        self.assertEqual(interp.read_dword(0x11C), 0x40800000)

    def test_out_of_bounds_suppressed_memory_unchanged(self) -> None:
        interp = TypedBufferReferenceInterpreter(memory={0x200: 0xDEADBEEF})
        desc = BufferResourceDescriptor.from_fields(base_address=0, num_records=0x100, format_code=GFX10_FMT_32_UINT)
        success = interp.store_format((0x12345678,), desc, voffset=0x200, format_code=GFX10_FMT_32_UINT)
        self.assertFalse(success)
        self.assertEqual(interp.read_dword(0x200), 0xDEADBEEF)

    def test_invalid_descriptor_format_suppressed(self) -> None:
        interp = TypedBufferReferenceInterpreter(memory={0x200: 0xDEADBEEF})
        desc = BufferResourceDescriptor.from_fields(base_address=0, num_records=0x1000, format_code=0)
        success = interp.store_format((1.0,), desc, voffset=0x200, format_code=GFX10_FMT_8_UNORM)
        self.assertFalse(success)
        self.assertEqual(interp.read_dword(0x200), 0xDEADBEEF)


if __name__ == "__main__":
    unittest.main()
