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


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

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
    BUFFER_ATOMIC_CMPSWAP = "buffer_atomic_cmpswap"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_cmpswap instruction."""
    opcode: Opcode
    vdata: Register
    vaddr: Register
    srsrc: Register
    soffset: Register | int
    mods: BufferAtomicModifiers = dataclasses.field(default_factory=BufferAtomicModifiers)

    def __post_init__(self) -> None:
        if not self.vdata.is_vgpr or self.vdata.count != 2:
            raise ValueError(f"vdata must be a 64-bit VGPR pair (count=2): {self.vdata}")
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


class TestBufferAtomicCmpswapAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_construction(self) -> None:
        v0 = Register(RegKind.VGPR, 0, count=1)
        v_pair = Register(RegKind.VGPR, 2, count=2)
        s_desc = Register(RegKind.SGPR, 8, count=4)

        self.assertEqual(v0.format_asm(), "v0")
        self.assertEqual(v_pair.format_asm(), "v[2:3]")
        self.assertEqual(s_desc.format_asm(), "s[8:11]")

    def test_instruction_assembly_formatting(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(offset=16, glc=True, slc=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_cmpswap v[0:1], v2, s[4:7], s8 offen offset:16 glc slc",
        )

    def test_immediate_soffset_formatting(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 4, count=2),
            vaddr=Register(RegKind.VGPR, 6, count=1),
            srsrc=Register(RegKind.SGPR, 12, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_cmpswap v[4:5], v6, s[12:15], 0 offen",
        )

    def test_vdata_must_be_vgpr_count_2(self) -> None:
        # Invalid: count=1
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )
        # Invalid: SGPR
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
                vdata=Register(RegKind.SGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

    def test_srsrc_must_be_sgpr_count_4(self) -> None:
        # Invalid: count=2
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=2),
                soffset=0,
            )
        # Invalid: VGPR
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.VGPR, 4, count=4),
                soffset=0,
            )

    def test_soffset_register_validation(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=Register(RegKind.VGPR, 8, count=1),
            )

    def test_immediate_offset_bounds(self) -> None:
        with self.assertRaises(ValueError):
            BufferAtomicModifiers(offset=4096)
        with self.assertRaises(ValueError):
            BufferAtomicModifiers(offset=-1)

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
                mods=BufferAtomicModifiers(glc=True),
            )
        )
        expected = "buffer_atomic_cmpswap v[0:1], v2, s[4:7], 0 offen glc"
        self.assertEqual(prog.to_assembly(), expected)


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarBufferInterpreter:
    """Deterministic single-lane scalar interpreter for buffer instructions."""

    def __init__(self, mem_size: int = 65536) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.mem: BufferMemory = BufferMemory(mem_size)
        self.trace: list[str] = []

    def read_reg_u32(self, kind: RegKind, idx: int) -> int:
        if kind == RegKind.VGPR:
            return self.vgprs[idx] & MASK32
        return self.sgprs[idx] & MASK32

    def write_reg_u32(self, kind: RegKind, idx: int, val: int) -> None:
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[idx] = val32
        else:
            self.sgprs[idx] = val32

    def set_buffer_descriptor(self, idx: int, desc: BufferResourceDescriptor) -> None:
        d0, d1, d2, d3 = desc.to_dwords()
        self.write_reg_u32(RegKind.SGPR, idx, d0)
        self.write_reg_u32(RegKind.SGPR, idx + 1, d1)
        self.write_reg_u32(RegKind.SGPR, idx + 2, d2)
        self.write_reg_u32(RegKind.SGPR, idx + 3, d3)

    def get_buffer_descriptor(self, idx: int) -> BufferResourceDescriptor:
        d0 = self.read_reg_u32(RegKind.SGPR, idx)
        d1 = self.read_reg_u32(RegKind.SGPR, idx + 1)
        d2 = self.read_reg_u32(RegKind.SGPR, idx + 2)
        d3 = self.read_reg_u32(RegKind.SGPR, idx + 3)
        return BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)

    def step(self, inst: Instruction) -> tuple[int, bool, bool]:
        """Executes a single buffer_atomic_cmpswap instruction deterministically."""
        desc = self.get_buffer_descriptor(inst.srsrc.index)
        soffset = (
            self.read_reg_u32(inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        src_val = self.read_reg_u32(RegKind.VGPR, inst.vdata.index)
        cmp_val = self.read_reg_u32(RegKind.VGPR, inst.vdata.index + 1)

        ret_val, swapped, in_bounds = ref_buffer_atomic_cmpswap(
            self.mem,
            desc,
            soffset=soffset,
            vaddr=vaddr,
            src_val=src_val,
            cmp_val=cmp_val,
            inst_offset=inst.mods.offset,
            glc=inst.mods.glc,
            offen=inst.mods.offen,
        )

        if inst.mods.glc:
            self.write_reg_u32(RegKind.VGPR, inst.vdata.index, ret_val)

        self.trace.append(
            f"{inst.to_assembly()} => ret={ret_val:#010x}, swapped={swapped}, in_bounds={in_bounds}"
        )
        return ret_val, swapped, in_bounds

    def run_program(self, prog: Program) -> list[tuple[int, bool, bool]]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


class TestBufferAtomicCmpswapScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_scalar_successful_cmpswap(self) -> None:
        self.interp.mem.write_u32(64, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 200)  # src
        self.interp.write_reg_u32(RegKind.VGPR, 1, 100)  # cmp
        self.interp.write_reg_u32(RegKind.VGPR, 2, 64)   # vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertTrue(swapped)
        self.assertEqual(self.interp.mem.read_u32(64), 200)
        # Without glc, registers remain untouched
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 200)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 1), 100)

    def test_scalar_failed_cmpswap(self) -> None:
        self.interp.mem.write_u32(64, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 200)  # src
        self.interp.write_reg_u32(RegKind.VGPR, 1, 999)  # cmp (mismatch)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 64)   # vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertFalse(swapped)
        self.assertEqual(self.interp.mem.read_u32(64), 100)

    def test_scalar_cmpswap_glc_writeback_match(self) -> None:
        self.interp.mem.write_u32(128, 500)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 750)  # src
        self.interp.write_reg_u32(RegKind.VGPR, 1, 500)  # cmp (match)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 128)  # vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertTrue(swapped)
        self.assertEqual(ret, 500)
        # vdata[0] receives pre-op value (500)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 500)
        # vdata[1] is preserved (500)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 1), 500)
        self.assertEqual(self.interp.mem.read_u32(128), 750)

    def test_scalar_cmpswap_glc_writeback_mismatch(self) -> None:
        self.interp.mem.write_u32(128, 500)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 750)  # src
        self.interp.write_reg_u32(RegKind.VGPR, 1, 123)  # cmp (mismatch)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 128)  # vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertFalse(swapped)
        self.assertEqual(ret, 500)
        # vdata[0] receives pre-op value (500)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 500)
        # vdata[1] is preserved (123)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 1), 123)
        self.assertEqual(self.interp.mem.read_u32(128), 500)

    def test_scalar_oob_suppression(self) -> None:
        self.interp.mem.write_u32(1024, 777)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 999)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 777)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 1024)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertFalse(in_bounds)
        self.assertFalse(swapped)
        self.assertEqual(ret, 0)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 0)
        self.assertEqual(self.interp.mem.read_u32(1024), 777)

    def test_scalar_program_chain(self) -> None:
        self.interp.mem.write_u32(0, 10)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 20)  # src
        self.interp.write_reg_u32(RegKind.VGPR, 1, 10)  # cmp
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)   # vaddr

        self.interp.write_reg_u32(RegKind.VGPR, 4, 30)  # src2
        self.interp.write_reg_u32(RegKind.VGPR, 5, 20)  # cmp2

        prog = Program()
        prog.append(
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
                mods=BufferAtomicModifiers(glc=False),
            )
        )
        prog.append(
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
                vdata=Register(RegKind.VGPR, 4, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
                mods=BufferAtomicModifiers(glc=False),
            )
        )

        results = self.interp.run_program(prog)
        self.assertTrue(results[0][1])  # 1st swapped: 10 -> 20
        self.assertTrue(results[1][1])  # 2nd swapped: 20 -> 30
        self.assertEqual(self.interp.mem.read_u32(0), 30)
        self.assertEqual(len(self.interp.trace), 2)


# ---------------------------------------------------------------------------
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ---------------------------------------------------------------------------

class SIMDBufferWavefrontInterpreter:
    """Multi-lane SIMD interpreter modeling Wave32 and Wave64 execution for buffer atomics."""

    def __init__(self, wave_size: int = 64, mem_size: int = 65536) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {wave_size} (must be 32 or 64)")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: list[int] = [0] * 106
        self.mem: BufferMemory = BufferMemory(mem_size)

    def set_exec_mask(self, mask: int) -> None:
        valid_mask = (1 << self.wave_size) - 1
        self.exec_mask = mask & valid_mask

    def read_lane_reg_u32(self, lane: int, kind: RegKind, idx: int) -> int:
        if kind == RegKind.VGPR:
            return self.vgprs[lane][idx] & MASK32
        return self.sgprs[idx] & MASK32

    def write_lane_reg_u32(self, lane: int, kind: RegKind, idx: int, val: int) -> None:
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[lane][idx] = val32
        else:
            self.sgprs[idx] = val32

    def set_buffer_descriptor(self, idx: int, desc: BufferResourceDescriptor) -> None:
        d0, d1, d2, d3 = desc.to_dwords()
        self.write_lane_reg_u32(0, RegKind.SGPR, idx, d0)
        self.write_lane_reg_u32(0, RegKind.SGPR, idx + 1, d1)
        self.write_lane_reg_u32(0, RegKind.SGPR, idx + 2, d2)
        self.write_lane_reg_u32(0, RegKind.SGPR, idx + 3, d3)

    def get_buffer_descriptor(self, idx: int) -> BufferResourceDescriptor:
        d0 = self.read_lane_reg_u32(0, RegKind.SGPR, idx)
        d1 = self.read_lane_reg_u32(0, RegKind.SGPR, idx + 1)
        d2 = self.read_lane_reg_u32(0, RegKind.SGPR, idx + 2)
        d3 = self.read_lane_reg_u32(0, RegKind.SGPR, idx + 3)
        return BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)

    def step(self, inst: Instruction) -> list[tuple[int | None, bool, bool]]:
        """Executes a buffer_atomic_cmpswap instruction across all lanes in the wavefront."""
        desc = self.get_buffer_descriptor(inst.srsrc.index)
        soffset = (
            self.read_lane_reg_u32(0, inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        # 1. Latch inputs for each lane
        lane_inputs: list[tuple[int, int, int]] = []
        for lane in range(self.wave_size):
            src_val = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vdata.index)
            cmp_val = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vdata.index + 1)
            vaddr = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index)
            lane_inputs.append((src_val, cmp_val, vaddr))

        # 2. Serialize atomic operations across active lanes in ascending order
        lane_results: list[tuple[int | None, bool, bool]] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append((None, False, False))
                continue

            src_val, cmp_val, vaddr = lane_inputs[lane]
            ret_val, swapped, in_bounds = ref_buffer_atomic_cmpswap(
                self.mem,
                desc,
                soffset=soffset,
                vaddr=vaddr,
                src_val=src_val,
                cmp_val=cmp_val,
                inst_offset=inst.mods.offset,
                glc=inst.mods.glc,
                offen=inst.mods.offen,
            )

            if inst.mods.glc:
                self.write_lane_reg_u32(lane, RegKind.VGPR, inst.vdata.index, ret_val)

            lane_results.append((ret_val, swapped, in_bounds))

        return lane_results


class TestBufferAtomicCmpswapSIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=65536,
        )

    def test_wave32_full_active_independent_locations(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)

        # Each lane targets 4 * lane
        for lane in range(32):
            addr = lane * 4
            simd.mem.write_u32(addr, 100 + lane)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 500 + lane)  # src
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 100 + lane)  # cmp (match)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, addr)        # vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        for lane in range(32):
            ret, swapped, in_bounds = results[lane]
            self.assertTrue(in_bounds)
            self.assertTrue(swapped)
            self.assertEqual(ret, 100 + lane)
            self.assertEqual(simd.mem.read_u32(lane * 4), 500 + lane)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 100 + lane)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 1), 100 + lane)

    def test_wave64_full_active_independent_locations(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)

        for lane in range(64):
            addr = lane * 4
            simd.mem.write_u32(addr, 1000 + lane)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 2000 + lane)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 1000 + lane)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        for lane in range(64):
            ret, swapped, in_bounds = results[lane]
            self.assertTrue(in_bounds)
            self.assertTrue(swapped)
            self.assertEqual(ret, 1000 + lane)
            self.assertEqual(simd.mem.read_u32(lane * 4), 2000 + lane)

    def test_exec_mask_inactive_lanes_suppressed(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        # Only lanes 0..3 active
        simd.set_exec_mask(0x0000000F)

        for lane in range(32):
            addr = lane * 4
            simd.mem.write_u32(addr, 50)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 99)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 50)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        for lane in range(4):
            self.assertTrue(results[lane][1])
            self.assertEqual(simd.mem.read_u32(lane * 4), 99)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 50)

        for lane in range(4, 32):
            self.assertIsNone(results[lane][0])
            self.assertEqual(simd.mem.read_u32(lane * 4), 50)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 99)

    def test_wavefront_colliding_address_serialization(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0x00000007)  # Lanes 0, 1, 2 active

        target_addr = 64
        simd.mem.write_u32(target_addr, 100)

        # Lane 0 expects 100 -> writes 200
        simd.write_lane_reg_u32(0, RegKind.VGPR, 0, 200)
        simd.write_lane_reg_u32(0, RegKind.VGPR, 1, 100)
        simd.write_lane_reg_u32(0, RegKind.VGPR, 2, target_addr)

        # Lane 1 expects 200 -> writes 300
        simd.write_lane_reg_u32(1, RegKind.VGPR, 0, 300)
        simd.write_lane_reg_u32(1, RegKind.VGPR, 1, 200)
        simd.write_lane_reg_u32(1, RegKind.VGPR, 2, target_addr)

        # Lane 2 expects 100 (stale!) -> fails
        simd.write_lane_reg_u32(2, RegKind.VGPR, 0, 400)
        simd.write_lane_reg_u32(2, RegKind.VGPR, 1, 100)
        simd.write_lane_reg_u32(2, RegKind.VGPR, 2, target_addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        # Lane 0 succeeded
        self.assertTrue(results[0][1])
        self.assertEqual(results[0][0], 100)
        # Lane 1 succeeded (serialized after lane 0)
        self.assertTrue(results[1][1])
        self.assertEqual(results[1][0], 200)
        # Lane 2 failed (memory was already 300)
        self.assertFalse(results[2][1])
        self.assertEqual(results[2][0], 300)

        # Final memory is 300
        self.assertEqual(simd.mem.read_u32(target_addr), 300)

    def test_partial_exec_masks_checkerboard(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0x55555555)  # Even lanes active

        for lane in range(32):
            simd.mem.write_u32(lane * 4, 10)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 20)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 10)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, lane * 4)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        simd.step(inst)

        for lane in range(32):
            expected = 20 if (lane % 2 == 0) else 10
            self.assertEqual(simd.mem.read_u32(lane * 4), expected)

    def test_exec_mask_zero_quiescent(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0)

        simd.mem.write_u32(0, 42)
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        self.assertEqual(len(results), 64)
        for r in results:
            self.assertIsNone(r[0])
            self.assertFalse(r[1])
        self.assertEqual(simd.mem.read_u32(0), 42)


# ---------------------------------------------------------------------------
# Register Aliasing and In-Place Operand Preservation
# ---------------------------------------------------------------------------

class TestBufferAtomicCmpswapAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_aliasing_vdata0_equals_vaddr_with_glc(self) -> None:
        # v0 is vdata[0] (src) and vaddr
        # Memory at address 64 contains 1000
        self.interp.mem.write_u32(64, 1000)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 64)    # src = 64, vaddr = 64
        self.interp.write_reg_u32(RegKind.VGPR, 1, 1000)  # cmp = 1000

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertTrue(swapped)
        # Memory at 64 was swapped to src (64)
        self.assertEqual(self.interp.mem.read_u32(64), 64)
        # v0 received pre-op value (1000)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 1000)
        # v1 (cmp) is preserved intact (1000)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 1), 1000)

    def test_aliasing_vdata1_equals_vaddr_with_glc(self) -> None:
        # v1 is vdata[1] (cmp) and vaddr
        # Address is 32; memory has 500
        self.interp.mem.write_u32(32, 500)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 999)   # src = 999
        self.interp.write_reg_u32(RegKind.VGPR, 1, 32)    # cmp = 32, vaddr = 32

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        # 500 != 32, so swap fails
        self.assertFalse(swapped)
        self.assertEqual(self.interp.mem.read_u32(32), 500)
        # v0 receives pre-op value (500)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 500)
        # v1 is preserved intact (32)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 1), 32)

    def test_vdata1_preserved_under_glc_writeback(self) -> None:
        self.interp.mem.write_u32(100, 0xAAAA5555)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x11112222)  # src
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0xAAAA5555)  # cmp
        self.interp.write_reg_u32(RegKind.VGPR, 2, 100)         # vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)

        # v0 receives pre-op value
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 0xAAAA5555)
        # v1 remains unchanged
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 1), 0xAAAA5555)

    def test_unrelated_vgprs_preserved(self) -> None:
        for idx in range(10, 20):
            self.interp.write_reg_u32(RegKind.VGPR, idx, 0x77000000 + idx)

        self.interp.write_reg_u32(RegKind.VGPR, 0, 10)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)

        for idx in range(10, 20):
            self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, idx), 0x77000000 + idx)

    def test_sgpr_descriptor_and_offset_preserved(self) -> None:
        d0, d1, d2, d3 = self.desc.to_dwords()
        self.interp.write_reg_u32(RegKind.SGPR, 8, 16)  # soffset

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)

        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 4), d0)
        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 5), d1)
        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 6), d2)
        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 7), d3)
        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 8), 16)

    def test_simd_aliasing_with_divergent_exec_mask(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        exec_mask = 0x0000000F  # Lanes 0..3 active
        simd.set_exec_mask(exec_mask)

        # Pre-seed memory: each lane has initial value 1000 + lane
        for lane in range(32):
            addr = lane * 4
            simd.mem.write_u32(addr, 1000 + lane)
            # Aliasing: vaddr = v0 = addr
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, addr)
            # cmp matches initial value
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 1000 + lane)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        simd.step(inst)

        # Active lanes:
        # Memory updated to src (addr); v0 receives pre-op value (1000 + lane); v1 preserved
        for lane in range(4):
            addr = lane * 4
            self.assertEqual(simd.mem.read_u32(addr), addr)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 1000 + lane)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 1), 1000 + lane)

        # Inactive lanes:
        # Memory untouched (1000 + lane); v0 untouched (addr); v1 untouched
        for lane in range(4, 32):
            addr = lane * 4
            self.assertEqual(simd.mem.read_u32(addr), 1000 + lane)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), addr)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 1), 1000 + lane)


# ---------------------------------------------------------------------------
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class AtomicCmpswapFlags:
    """Architectural condition codes and status flags for buffer_atomic_cmpswap."""
    success: bool
    zero: bool
    negative: bool
    diff: int
    borrow: bool
    carry: bool
    overflow: bool
    unaligned_addr: bool
    out_of_bounds: bool


def evaluate_cmpswap_flags(
    old_val: int,
    cmp_val: int,
    src_val: int,
    eff_addr: int,
    num_records: int,
    swapped: bool,
) -> AtomicCmpswapFlags:
    """Evaluate condition codes, status flags, and arithmetic carry/borrow for cmpswap."""
    u_old = old_val & MASK32
    u_cmp = cmp_val & MASK32
    u_src = src_val & MASK32

    # Zero flag: returned old value is zero
    zero = (u_old == 0)

    # Negative flag: MSB of returned old value is set
    negative = bool(u_old & SIGN_BIT32)

    # Unsigned difference and borrow against comparison value
    diff = (u_old - u_cmp) & MASK32
    borrow = u_old < u_cmp

    # Carry generation on new value write
    carry = ((u_old + u_src) > MASK32) if swapped else False

    # Signed comparison overflow for (old_val - cmp_val)
    s_old = (u_old ^ SIGN_BIT32) - SIGN_BIT32
    s_cmp = (u_cmp ^ SIGN_BIT32) - SIGN_BIT32
    s_diff = s_old - s_cmp
    overflow = s_diff < -0x80000000 or s_diff > 0x7FFFFFFF

    # Unaligned address check (low 2 bits must be zero for 4-byte dword alignment)
    unaligned = bool(eff_addr & 0x3)

    # Bounds check against descriptor limit
    oob = (eff_addr + 4) > num_records

    return AtomicCmpswapFlags(
        success=swapped,
        zero=zero,
        negative=negative,
        diff=diff,
        borrow=borrow,
        carry=carry,
        overflow=overflow,
        unaligned_addr=unaligned,
        out_of_bounds=oob,
    )


class TestBufferAtomicCmpswapFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_exact_match_flags(self) -> None:
        flags = evaluate_cmpswap_flags(
            old_val=0x1234,
            cmp_val=0x1234,
            src_val=0x5678,
            eff_addr=0x100,
            num_records=1024,
            swapped=True,
        )
        self.assertTrue(flags.success)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertEqual(flags.diff, 0)
        self.assertFalse(flags.borrow)
        self.assertFalse(flags.carry)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.unaligned_addr)
        self.assertFalse(flags.out_of_bounds)

    def test_zero_flag_on_zero_return(self) -> None:
        flags = evaluate_cmpswap_flags(
            old_val=0,
            cmp_val=0,
            src_val=100,
            eff_addr=0x200,
            num_records=1024,
            swapped=True,
        )
        self.assertTrue(flags.success)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.negative)
        self.assertEqual(flags.diff, 0)
        self.assertFalse(flags.borrow)

    def test_negative_flag_when_msb_set(self) -> None:
        flags = evaluate_cmpswap_flags(
            old_val=0x80000001,
            cmp_val=0x10,
            src_val=0x20,
            eff_addr=0x300,
            num_records=1024,
            swapped=False,
        )
        self.assertFalse(flags.success)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.negative)
        self.assertFalse(flags.borrow)

    def test_borrow_generation_when_old_below_cmp(self) -> None:
        flags = evaluate_cmpswap_flags(
            old_val=10,
            cmp_val=25,
            src_val=50,
            eff_addr=0x400,
            num_records=1024,
            swapped=False,
        )
        self.assertFalse(flags.success)
        self.assertTrue(flags.borrow)
        self.assertEqual(flags.diff, (10 - 25) & MASK32)

    def test_no_borrow_when_old_above_cmp(self) -> None:
        flags = evaluate_cmpswap_flags(
            old_val=50,
            cmp_val=25,
            src_val=100,
            eff_addr=0x500,
            num_records=1024,
            swapped=False,
        )
        self.assertFalse(flags.success)
        self.assertFalse(flags.borrow)
        self.assertEqual(flags.diff, 25)

    def test_signed_overflow_flag(self) -> None:
        # s_old = 0x7FFFFFFF (max positive), s_cmp = -10 (negative)
        # diff = 0x7FFFFFFF - (-10) overflows positive 32-bit signed range
        flags = evaluate_cmpswap_flags(
            old_val=0x7FFFFFFF,
            cmp_val=s32(-10),
            src_val=0,
            eff_addr=0x100,
            num_records=1024,
            swapped=False,
        )
        self.assertFalse(flags.success)
        self.assertTrue(flags.overflow)

    def test_unaligned_address_detection(self) -> None:
        flags_unaligned = evaluate_cmpswap_flags(
            old_val=10,
            cmp_val=10,
            src_val=20,
            eff_addr=0x102,  # Not 4-byte aligned
            num_records=1024,
            swapped=True,
        )
        self.assertTrue(flags_unaligned.unaligned_addr)

        flags_aligned = evaluate_cmpswap_flags(
            old_val=10,
            cmp_val=10,
            src_val=20,
            eff_addr=0x104,
            num_records=1024,
            swapped=True,
        )
        self.assertFalse(flags_aligned.unaligned_addr)

    def test_out_of_bounds_detection(self) -> None:
        flags_oob = evaluate_cmpswap_flags(
            old_val=0,
            cmp_val=0,
            src_val=0,
            eff_addr=1024,  # eff_addr + 4 > 1024
            num_records=1024,
            swapped=False,
        )
        self.assertTrue(flags_oob.out_of_bounds)

        flags_in_bounds = evaluate_cmpswap_flags(
            old_val=0,
            cmp_val=0,
            src_val=0,
            eff_addr=1020,  # 1020 + 4 == 1024 (in bounds)
            num_records=1024,
            swapped=True,
        )
        self.assertFalse(flags_in_bounds.out_of_bounds)

    def test_carry_generation_on_write(self) -> None:
        # Carry flag is set when old + src overflows 32 bits on successful swap
        flags_carry = evaluate_cmpswap_flags(
            old_val=0xFFFFFFFF,
            cmp_val=0xFFFFFFFF,
            src_val=1,
            eff_addr=0x100,
            num_records=1024,
            swapped=True,
        )
        self.assertTrue(flags_carry.carry)

        flags_no_carry = evaluate_cmpswap_flags(
            old_val=10,
            cmp_val=10,
            src_val=20,
            eff_addr=0x100,
            num_records=1024,
            swapped=True,
        )
        self.assertFalse(flags_no_carry.carry)


# ---------------------------------------------------------------------------
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ---------------------------------------------------------------------------

F32_POS_ZERO: int = 0x00000000
F32_NEG_ZERO: int = 0x80000000
F32_POS_INF: int = 0x7F800000
F32_NEG_INF: int = 0xFF800000
F32_QUIET_NAN: int = 0x7FC00000
F32_SIGNALING_NAN: int = 0x7F800001
F32_MIN_SUBNORMAL: int = 0x00000001
F32_MAX_SUBNORMAL: int = 0x007FFFFF
F32_MIN_NORMAL: int = 0x00800000
F32_MAX_NORMAL: int = 0x7F7FFFFF


def is_f32_nan(bits: int) -> bool:
    """Check if 32-bit pattern represents an IEEE 754 NaN."""
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    return exp == 0xFF and mant != 0


def is_f32_subnormal(bits: int) -> bool:
    """Check if 32-bit pattern represents a denormalized/subnormal float."""
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    return exp == 0 and mant != 0


class TestBufferAtomicCmpswapIEEE754(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_bitwise_exact_zeros_do_not_match(self) -> None:
        # In float math +0.0 == -0.0, but in atomic cmpswap bitwise equality is required.
        addr = 64
        self.interp.mem.write_u32(addr, F32_POS_ZERO)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x1234)        # src
        self.interp.write_reg_u32(RegKind.VGPR, 1, F32_NEG_ZERO)  # cmp (-0.0)
        self.interp.write_reg_u32(RegKind.VGPR, 2, addr)          # vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        # +0.0 does NOT bitwise match -0.0
        self.assertFalse(swapped)
        self.assertEqual(ret, F32_POS_ZERO)
        self.assertEqual(self.interp.mem.read_u32(addr), F32_POS_ZERO)

    def test_bitwise_exact_zeros_match_when_identical(self) -> None:
        addr = 64
        self.interp.mem.write_u32(addr, F32_NEG_ZERO)
        self.interp.write_reg_u32(RegKind.VGPR, 0, F32_POS_ZERO)  # src
        self.interp.write_reg_u32(RegKind.VGPR, 1, F32_NEG_ZERO)  # cmp (-0.0)
        self.interp.write_reg_u32(RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertTrue(swapped)
        self.assertEqual(ret, F32_NEG_ZERO)
        self.assertEqual(self.interp.mem.read_u32(addr), F32_POS_ZERO)

    def test_quiet_nan_identical_payload_matches(self) -> None:
        nan_val = 0x7FC01234
        self.assertTrue(is_f32_nan(nan_val))
        addr = 128
        self.interp.mem.write_u32(addr, nan_val)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0xCAFE)
        self.interp.write_reg_u32(RegKind.VGPR, 1, nan_val)
        self.interp.write_reg_u32(RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertTrue(swapped)
        self.assertEqual(ret, nan_val)
        self.assertEqual(self.interp.mem.read_u32(addr), 0xCAFE)

    def test_quiet_nan_differing_payloads_do_not_match(self) -> None:
        nan1 = 0x7FC00001
        nan2 = 0x7FC00002
        addr = 128
        self.interp.mem.write_u32(addr, nan1)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x9999)
        self.interp.write_reg_u32(RegKind.VGPR, 1, nan2)
        self.interp.write_reg_u32(RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertFalse(swapped)
        self.assertEqual(ret, nan1)
        self.assertEqual(self.interp.mem.read_u32(addr), nan1)

    def test_signaling_nan_preserved_without_quietening(self) -> None:
        self.assertTrue(is_f32_nan(F32_SIGNALING_NAN))
        addr = 200
        self.interp.mem.write_u32(addr, 0x1111)
        self.interp.write_reg_u32(RegKind.VGPR, 0, F32_SIGNALING_NAN)  # src
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x1111)             # cmp
        self.interp.write_reg_u32(RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertTrue(swapped)
        # Signaling NaN stored bit-for-bit, bit 22 remains 0 (signaling)
        stored = self.interp.mem.read_u32(addr)
        self.assertEqual(stored, F32_SIGNALING_NAN)
        self.assertEqual(stored & 0x00400000, 0)

    def test_infinities_distinct_and_preserved(self) -> None:
        addr = 256
        self.interp.mem.write_u32(addr, F32_POS_INF)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 1, F32_NEG_INF)
        self.interp.write_reg_u32(RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        # +inf != -inf
        self.assertFalse(swapped)
        self.assertEqual(self.interp.mem.read_u32(addr), F32_POS_INF)

        # Now compare with +inf -> matches
        self.interp.write_reg_u32(RegKind.VGPR, 0, F32_NEG_INF)
        self.interp.write_reg_u32(RegKind.VGPR, 1, F32_POS_INF)
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertTrue(swapped)
        self.assertEqual(self.interp.mem.read_u32(addr), F32_NEG_INF)

    def test_subnormals_preserved_no_denorm_flush_to_zero(self) -> None:
        self.assertTrue(is_f32_subnormal(F32_MIN_SUBNORMAL))
        self.assertTrue(is_f32_subnormal(F32_MAX_SUBNORMAL))

        addr = 300
        self.interp.mem.write_u32(addr, F32_MIN_SUBNORMAL)
        self.interp.write_reg_u32(RegKind.VGPR, 0, F32_MAX_SUBNORMAL)
        self.interp.write_reg_u32(RegKind.VGPR, 1, F32_MIN_SUBNORMAL)
        self.interp.write_reg_u32(RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, swapped, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertTrue(swapped)
        self.assertEqual(ret, F32_MIN_SUBNORMAL)
        self.assertEqual(self.interp.mem.read_u32(addr), F32_MAX_SUBNORMAL)
        self.assertNotEqual(self.interp.mem.read_u32(addr), 0)

    def test_integer_extremes_boundary_values(self) -> None:
        extremes = [0x00000000, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 0x00000001, 0xFFFFFFFE]
        addr = 400
        self.interp.write_reg_u32(RegKind.VGPR, 2, addr)

        for i in range(len(extremes) - 1):
            curr_val = extremes[i]
            next_val = extremes[i + 1]
            self.interp.mem.write_u32(addr, curr_val)
            self.interp.write_reg_u32(RegKind.VGPR, 0, next_val)  # src
            self.interp.write_reg_u32(RegKind.VGPR, 1, curr_val)  # cmp

            inst = Instruction(
                opcode=Opcode.BUFFER_ATOMIC_CMPSWAP,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
                mods=BufferAtomicModifiers(glc=True),
            )
            ret, swapped, in_bounds = self.interp.step(inst)
            self.assertTrue(swapped)
            self.assertEqual(ret, curr_val)
            self.assertEqual(self.interp.mem.read_u32(addr), next_val)


if __name__ == "__main__":
    unittest.main()
