from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_smin.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory signed atomic minimum:
  buffer_atomic_smin vdata, vaddr, srsrc, soffset [offset:N] [offen] [idxen] [glc] [slc]

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
  3. Atomic Signed Minimum Operation:
     For in-bounds accesses:
       old_val = memory[effective_address]
       new_val = min(signed32(old_val), signed32(vdata))
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
INT32_MIN: int = -2147483648
INT32_MAX: int = 2147483647

# Boundary constants
U32_INT32_MIN: int = 0x80000000
U32_INT32_MAX: int = 0x7FFFFFFF
U32_MINUS_ONE: int = 0xFFFFFFFF
U32_ZERO: int = 0x00000000


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


def ref_buffer_atomic_smin(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    cmp_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference mathematical oracle for RDNA2 buffer_atomic_smin.

    Computes:
      effective_offset = soffset + inst_offset + (vaddr if offen else 0)
      if effective_offset + 4 <= desc.num_records:
        old_val = mem[effective_address]
        new_val = min(s32(old_val), s32(cmp_val))
        mem[effective_address] = u32(new_val)
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

    phys_addr = (desc.base_address + effective_offset) & 0xFFFFFFFFFFFFFFFF
    old_val = mem.read_u32(phys_addr)
    min_val = min(s32(old_val), s32(cmp_val))
    new_val = u32(min_val)
    mem.write_u32(phys_addr, new_val)

    ret_val = old_val if glc else (cmp_val & MASK32)
    return ret_val, True


class RegKind(enum.Enum):
    """Register file domain."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    """Symbolic register operand."""
    kind: RegKind
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        if self.index < 0 or self.index >= 256:
            raise ValueError(f"Register index out of range [0, 255]: {self.index}")
        if self.count not in (1, 2, 4):
            raise ValueError(f"Unsupported register count: {self.count}")
        if self.index + self.count > 256:
            raise ValueError(f"Register range exceeds register file boundary: {self.index}")

    @property
    def is_vgpr(self) -> bool:
        return self.kind == RegKind.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.kind == RegKind.SGPR

    def format_asm(self) -> str:
        if self.count > 1:
            return f"{self.kind.value}[{self.index}:{self.index + self.count - 1}]"
        return f"{self.kind.value}{self.index}"


@dataclasses.dataclass(frozen=True)
class BufferAtomicModifiers:
    """Modifiers and options for buffer atomic instructions."""
    offset: int = 0
    glc: bool = False
    slc: bool = False
    offen: bool = True
    idxen: bool = False

    def __post_init__(self) -> None:
        if self.offset < 0 or self.offset > 4095:
            raise ValueError(f"Immediate inst_offset must be in range [0, 4095]: {self.offset}")


class Opcode(enum.Enum):
    """Buffer atomic opcodes."""
    BUFFER_ATOMIC_SMIN = "buffer_atomic_smin"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_smin instruction."""
    opcode: Opcode
    vdata: Register
    vaddr: Register
    srsrc: Register
    soffset: Register | int
    mods: BufferAtomicModifiers = dataclasses.field(default_factory=BufferAtomicModifiers)

    def __post_init__(self) -> None:
        if not self.vdata.is_vgpr or self.vdata.count != 1:
            raise ValueError(f"vdata must be a single 32-bit VGPR: {self.vdata}")
        if not self.vaddr.is_vgpr:
            raise ValueError(f"vaddr must be a VGPR: {self.vaddr}")
        if not self.srsrc.is_sgpr or self.srsrc.count != 4:
            raise ValueError(f"srsrc must be a 128-bit SGPR group (count=4): {self.srsrc}")
        if isinstance(self.soffset, Register) and not self.soffset.is_sgpr:
            raise ValueError(f"soffset register must be an SGPR: {self.soffset}")

    def to_assembly(self) -> str:
        vd_str = self.vdata.format_asm()
        va_str = self.vaddr.format_asm()
        sr_str = self.srsrc.format_asm()
        so_str = self.soffset.format_asm() if isinstance(self.soffset, Register) else str(self.soffset)

        mod_parts: list[str] = []
        if self.mods.offen:
            mod_parts.append("offen")
        if self.mods.idxen:
            mod_parts.append("idxen")
        if self.mods.offset != 0:
            mod_parts.append(f"offset:{self.mods.offset}")
        if self.mods.glc:
            mod_parts.append("glc")
        if self.mods.slc:
            mod_parts.append("slc")

        mods_str = (" " + " ".join(mod_parts)) if mod_parts else ""
        return f"{self.opcode.value} {vd_str}, {va_str}, {sr_str}, {so_str}{mods_str}"


@dataclasses.dataclass
class Program:
    """Symbolic sequence of buffer instructions."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


class TestBufferAtomicSminAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_definition(self) -> None:
        self.assertEqual(Opcode.BUFFER_ATOMIC_SMIN.value, "buffer_atomic_smin")

    def test_register_operand_validation(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        self.assertEqual(v0.format_asm(), "v0")
        s_group = Register(RegKind.SGPR, 4, count=4)
        self.assertEqual(s_group.format_asm(), "s[4:7]")

        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 4, count=3)

    def test_instruction_construction_and_formatting(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8),
            mods=BufferAtomicModifiers(offen=True, offset=64, glc=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_smin v0, v1, s[4:7], s8 offen offset:64 glc",
        )

    def test_instruction_with_immediate_soffset(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN,
            vdata=Register(RegKind.VGPR, 2),
            vaddr=Register(RegKind.VGPR, 3),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=False, slc=True),
        )
        self.assertEqual(inst.to_assembly(), "buffer_atomic_smin v2, v3, s[0:3], 0 slc")

    def test_program_assembly(self) -> None:
        prog = Program()
        inst1 = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
        )
        prog.append(inst1)
        self.assertEqual(prog.to_assembly(), inst1.to_assembly())


class TestBufferAtomicSminOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_signed_min_positive_numbers(self) -> None:
        # Initial memory has 20, cmp with 10 -> min is 10
        self.mem.write_u32(0, 20)
        ret, ok = ref_buffer_atomic_smin(self.mem, self.desc, soffset=0, vaddr=0, cmp_val=10, glc=True)
        self.assertTrue(ok)
        self.assertEqual(ret, 20)
        self.assertEqual(self.mem.read_u32(0), 10)

        # Compare with 30 -> min is still 10
        ret2, ok2 = ref_buffer_atomic_smin(self.mem, self.desc, soffset=0, vaddr=0, cmp_val=30, glc=True)
        self.assertTrue(ok2)
        self.assertEqual(ret2, 10)
        self.assertEqual(self.mem.read_u32(0), 10)

    def test_signed_min_negative_numbers(self) -> None:
        # Initial memory has 5, cmp with -10 (0xFFFFFFF6) -> min is -10
        self.mem.write_u32(4, 5)
        cmp_neg = u32(-10)
        ret, ok = ref_buffer_atomic_smin(self.mem, self.desc, soffset=0, vaddr=4, cmp_val=cmp_neg, glc=True)
        self.assertTrue(ok)
        self.assertEqual(ret, 5)
        self.assertEqual(s32(self.mem.read_u32(4)), -10)

        # Compare -10 with -20 -> min is -20
        cmp_neg20 = u32(-20)
        ret2, ok2 = ref_buffer_atomic_smin(self.mem, self.desc, soffset=0, vaddr=4, cmp_val=cmp_neg20, glc=True)
        self.assertTrue(ok2)
        self.assertEqual(s32(ret2), -10)
        self.assertEqual(s32(self.mem.read_u32(4)), -20)

    def test_signed_min_extremes(self) -> None:
        # INT32_MAX (0x7FFFFFFF) vs INT32_MIN (0x80000000) -> min is INT32_MIN
        self.mem.write_u32(8, U32_INT32_MAX)
        ret, ok = ref_buffer_atomic_smin(self.mem, self.desc, soffset=0, vaddr=8, cmp_val=U32_INT32_MIN, glc=True)
        self.assertTrue(ok)
        self.assertEqual(ret, U32_INT32_MAX)
        self.assertEqual(self.mem.read_u32(8), U32_INT32_MIN)

    def test_signed_min_with_zero(self) -> None:
        # 0 vs -1 -> min is -1
        self.mem.write_u32(12, 0)
        ret, ok = ref_buffer_atomic_smin(self.mem, self.desc, soffset=0, vaddr=12, cmp_val=U32_MINUS_ONE, glc=True)
        self.assertTrue(ok)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u32(12), U32_MINUS_ONE)

    def test_oob_suppression_and_glc_zero(self) -> None:
        # Buffer num_records is 1024, access at 1024 (1024 + 4 > 1024) -> OOB
        self.mem.write_u32(100, 42)
        ret, ok = ref_buffer_atomic_smin(self.mem, self.desc, soffset=0, vaddr=1024, cmp_val=-5, glc=True)
        self.assertFalse(ok)
        self.assertEqual(ret, 0)  # glc on OOB returns 0


if __name__ == "__main__":
    unittest.main()
