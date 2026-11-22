"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_and.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory atomic bitwise AND:
  buffer_atomic_and vdata, vaddr, srsrc, soffset [offset:N] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR containing the 32-bit dword value to bitwise AND atomically. If 'glc' is enabled,
       the pre-operation dword value at memory is returned to vdata.
     - vaddr: VGPR containing the byte or index offset (when offen or idxen is enabled).
     - srsrc: 128-bit Buffer Resource Descriptor (4 SGPRs s[r:r+3]):
         Dword 0: Base address low [31:0]
         Dword 1: Base address high [15:0], STRIDE [31:16]
         Dword 2: NUM_RECORDS (buffer limit / capacity in bytes)
         Dword 3: Format, element size, resource type
     - soffset: Scalar SGPR or immediate byte offset.
     - inst_offset: 12-bit unsigned immediate byte offset [0, 4095].
  2. Addressing and Bounds Checking:
     effective_offset = soffset + inst_offset + (vaddr if offen else 0)
     If effective_offset + 4 > NUM_RECORDS:
       Access is Out-Of-Bounds (OOB).
       The atomic operation is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to vdata.
  3. Atomic Operation:
     For in-bounds accesses:
       old_val = memory[effective_address]
       new_val = (old_val & (and_val & 0xFFFFFFFF)) & 0xFFFFFFFF
       memory[effective_address] = new_val
       if glc: vdata = old_val
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


def ref_buffer_atomic_and(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    and_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_and.

    Returns:
      (return_val, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS
    if effective_offset + 4 > desc.num_records:
        # Out-of-bounds: memory untouched, returns 0 if glc
        return (0 if glc else (and_val & MASK32)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u32(phys_addr)
    new_val = (old_val & (and_val & MASK32)) & MASK32
    mem.write_u32(phys_addr, new_val)

    ret_val = old_val if glc else (and_val & MASK32)
    return ret_val, True


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
    BUFFER_ATOMIC_AND = "buffer_atomic_and"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_and instruction."""
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


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarBufferInterpreter:
    """Deterministic scalar interpreter for buffer_atomic_and."""

    def __init__(self, mem_size: int = 65536) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.mem: BufferMemory = BufferMemory(mem_size)
        self.vcc: int = 0
        self.trace: list[str] = []

    def read_reg_u32(self, kind: RegKind, idx: int) -> int:
        if idx < 0 or (kind == RegKind.VGPR and idx >= 256) or (kind == RegKind.SGPR and idx >= 106):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        return (self.vgprs[idx] if kind == RegKind.VGPR else self.sgprs[idx]) & MASK32

    def write_reg_u32(self, kind: RegKind, idx: int, val: int) -> None:
        if idx < 0 or (kind == RegKind.VGPR and idx >= 256) or (kind == RegKind.SGPR and idx >= 106):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
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

    def step(self, inst: Instruction) -> tuple[int, bool]:
        """Executes a single buffer_atomic_and instruction."""
        vdata = self.read_reg_u32(RegKind.VGPR, inst.vdata.index)
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        desc = self.get_buffer_descriptor(inst.srsrc.index)

        soffset = (
            self.read_reg_u32(inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        ret_val, in_bounds = ref_buffer_atomic_and(
            self.mem,
            desc,
            soffset=soffset,
            vaddr=vaddr,
            and_val=vdata,
            inst_offset=inst.mods.offset,
            glc=inst.mods.glc,
            offen=inst.mods.offen,
        )

        if inst.mods.glc:
            self.write_reg_u32(RegKind.VGPR, inst.vdata.index, ret_val)

        self.trace.append(f"{inst.to_assembly()} => ret={ret_val:#010x}, in_bounds={in_bounds}")
        return ret_val, in_bounds

    def run_program(self, prog: Program) -> list[tuple[int, bool]]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


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

    def step(self, inst: Instruction) -> list[tuple[int | None, bool]]:
        """Executes a buffer_atomic_and instruction across all lanes in the wavefront."""
        desc = self.get_buffer_descriptor(inst.srsrc.index)
        soffset = (
            self.read_lane_reg_u32(0, inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        # 1. Latch inputs for each lane
        lane_inputs: list[tuple[int, int]] = []
        for lane in range(self.wave_size):
            vd = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vdata.index)
            va = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index)
            lane_inputs.append((vd, va))

        # 2. Serialize atomic operations across active lanes
        lane_results: list[tuple[int | None, bool]] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append((None, False))
                continue

            vdata, vaddr = lane_inputs[lane]
            ret_val, in_bounds = ref_buffer_atomic_and(
                self.mem,
                desc,
                soffset=soffset,
                vaddr=vaddr,
                and_val=vdata,
                inst_offset=inst.mods.offset,
                glc=inst.mods.glc,
                offen=inst.mods.offen,
            )

            if inst.mods.glc:
                self.write_lane_reg_u32(lane, RegKind.VGPR, inst.vdata.index, ret_val)

            lane_results.append((ret_val, in_bounds))

        return lane_results


class TestBufferAtomicAndSIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=4096,
        )

    def test_wave32_divergent_addresses(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        for lane in range(32):
            simd.mem.write_u32(lane * 4, 0xFFFFFFFF)
            mask = ~(1 << lane) & MASK32
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, mask)      # and value
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)  # address

        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertTrue(results[lane][1])
            expected = ~(1 << lane) & MASK32
            self.assertEqual(simd.mem.read_u32(lane * 4), expected)

    def test_wave64_convergent_atomic_and(self) -> None:
        # 32 lanes each clear one distinct bit in the same dword at address 0
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.mem.write_u32(0, 0xFFFFFFFF)

        # Enable only first 32 lanes
        simd.set_exec_mask(0xFFFFFFFF)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        for lane in range(32):
            mask = ~(1 << lane) & MASK32
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, mask)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0)  # all target address 0

        results = simd.step(inst)
        # All 32 bits cleared -> result must be 0
        self.assertEqual(simd.mem.read_u32(0), 0)
        # Verify lane 0 received initial 0xFFFFFFFF
        self.assertEqual(results[0][0], 0xFFFFFFFF)

    def test_wave32_exec_masking_skips_inactive(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)

        # Active mask: lanes 0..3 active (0x0F)
        simd.set_exec_mask(0x0000000F)

        for lane in range(32):
            simd.mem.write_u32(lane * 4, 0xAAAAAAAA)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 0x00000000)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        # Active lanes: memory zeroed, glc returns 0xAAAAAAAA
        for lane in range(4):
            self.assertEqual(simd.mem.read_u32(lane * 4), 0)
            self.assertEqual(results[lane], (0xAAAAAAAA, True))
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 0xAAAAAAAA)

        # Inactive lanes: memory untouched, result (None, False), register untouched (0)
        for lane in range(4, 32):
            self.assertEqual(simd.mem.read_u32(lane * 4), 0xAAAAAAAA)
            self.assertEqual(results[lane], (None, False))
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 0)

    def test_wave64_alternating_exec_mask(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        # Even lanes active (0x5555555555555555)
        simd.set_exec_mask(0x5555555555555555)

        for lane in range(64):
            simd.mem.write_u32(lane * 4, 0x12345678)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 0x0000FFFF)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        results = simd.step(inst)
        for lane in range(64):
            if lane % 2 == 0:
                self.assertTrue(results[lane][1])
                self.assertEqual(simd.mem.read_u32(lane * 4), 0x00005678)
            else:
                self.assertFalse(results[lane][1])
                self.assertEqual(simd.mem.read_u32(lane * 4), 0x12345678)


class TestBufferAtomicAndRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_vdata_vaddr_aliasing_same_vgpr(self) -> None:
        # v0 acts as both vdata (and_val) and vaddr (address offset)
        addr_and_val = 32
        self.interp.mem.write_u32(addr_and_val, 0xFFFFFFFF)
        self.interp.write_reg_u32(RegKind.VGPR, 0, addr_and_val)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        # Pre-op memory value was 0xFFFFFFFF
        self.assertEqual(ret, 0xFFFFFFFF)
        # Memory at 32 becomes 0xFFFFFFFF & 32 = 32
        self.assertEqual(self.interp.mem.read_u32(addr_and_val), 32)
        # v0 overwritten with pre-op value (0xFFFFFFFF)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 0xFFFFFFFF)

    def test_canary_neighbor_vgprs_unmodified(self) -> None:
        # Preload canary values into VGPRs 0..31
        canaries = [0xDEAD0000 + i for i in range(32)]
        for i, val in enumerate(canaries):
            self.interp.write_reg_u32(RegKind.VGPR, i, val)

        # Set address to 64 and value to 0x00FF00FF
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x00FF00FF)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 64)
        self.interp.mem.write_u32(64, 0x12345678)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        self.interp.step(inst)

        # Registers 2..31 must strictly retain their canaries
        for idx in range(2, 32):
            self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, idx), canaries[idx])

    def test_sgpr_descriptor_and_offset_preserved(self) -> None:
        d0, d1, d2, d3 = self.desc.to_dwords()
        self.interp.write_reg_u32(RegKind.SGPR, 8, 16)  # soffset

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)

        # SGPR descriptor and soffset registers must be completely untouched
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
        simd.mem.write_u32(16, 0xFFFFFFFF)

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 16)  # address and value = 16

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        simd.step(inst)

        # Active lanes: v0 updated to pre-op memory values in sequential order
        # Initial memory was 0xFFFFFFFF. Lane 0 reads 0xFFFFFFFF, sets mem to 0xFFFFFFFF & 16 = 16.
        # Lanes 1, 2, 3 read 16, set mem to 16 & 16 = 16.
        self.assertEqual(simd.read_lane_reg_u32(0, RegKind.VGPR, 0), 0xFFFFFFFF)
        for lane in range(1, 4):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 16)
        # Inactive lanes: v0 untouched (16)
        for lane in range(4, 32):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 16)


class TestBufferAtomicAndScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_scalar_atomic_and_execution(self) -> None:
        self.interp.mem.write_u32(64, 0xFFFFFFFF)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x00FF00FF)  # vdata
        self.interp.write_reg_u32(RegKind.VGPR, 1, 64)          # vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(self.interp.mem.read_u32(64), 0x00FF00FF)
        # Without glc, vdata in register remains untouched
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 0x00FF00FF)

    def test_scalar_atomic_and_glc_writeback(self) -> None:
        self.interp.mem.write_u32(128, 0x12345678)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x0000FFFF)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 128)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x12345678)
        # vdata updated to previous value (0x12345678)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 0x12345678)
        # Memory holds 0x12345678 & 0x0000FFFF = 0x00005678
        self.assertEqual(self.interp.mem.read_u32(128), 0x00005678)

    def test_scalar_oob_suppression(self) -> None:
        self.interp.mem.write_u32(1024, 0x77777777)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x00000000)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 1024)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 0)
        self.assertEqual(self.interp.mem.read_u32(1024), 0x77777777)

    def test_scalar_program_chain(self) -> None:
        self.interp.mem.write_u32(0, 0xFFFFFFFF)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0xFFFF0000)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0x00FFFF00)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        ))
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 2, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        ))

        self.interp.run_program(prog)
        # 0xFFFFFFFF & 0xFFFF0000 & 0x00FFFF00 = 0x00FF0000
        self.assertEqual(self.interp.mem.read_u32(0), 0x00FF0000)
        self.assertEqual(len(self.interp.trace), 2)

    def test_soffset_from_sgpr(self) -> None:
        self.interp.mem.write_u32(100, 0xFF00FF00)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x0F0F0F0F)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 20)          # vaddr
        self.interp.write_reg_u32(RegKind.SGPR, 8, 80)          # soffset in s8

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xFF00FF00)
        self.assertEqual(self.interp.mem.read_u32(100), 0x0F000F00)


class TestBufferAtomicAndAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_construction(self) -> None:
        v0 = Register(RegKind.VGPR, 0, count=1)
        v1 = Register(RegKind.VGPR, 1, count=1)
        s_desc = Register(RegKind.SGPR, 4, count=4)
        s_off = Register(RegKind.SGPR, 8, count=1)

        self.assertEqual(v0.format_asm(), "v0")
        self.assertEqual(v1.format_asm(), "v1")
        self.assertEqual(s_desc.format_asm(), "s[4:7]")
        self.assertEqual(s_off.format_asm(), "s8")

    def test_instruction_formatting(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(offset=16, glc=True, offen=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_and v0, v1, s[4:7], s8 offen offset:16 glc",
        )

        inst_imm = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 2, count=1),
            vaddr=Register(RegKind.VGPR, 3, count=1),
            srsrc=Register(RegKind.SGPR, 12, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True),
        )
        self.assertEqual(
            inst_imm.to_assembly(),
            "buffer_atomic_and v2, v3, s[12:15], 0 offen",
        )

    def test_instruction_validation(self) -> None:
        # Invalid vdata kind (must be VGPR)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_AND,
                vdata=Register(RegKind.SGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

        # Invalid srsrc count (must be 4)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_AND,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=2),
                soffset=0,
            )

        # Invalid offset (> 4095)
        with self.assertRaises(ValueError):
            BufferAtomicModifiers(offset=5000)

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        ))
        expected = "buffer_atomic_and v0, v1, s[4:7], 0 offen glc"
        self.assertEqual(prog.to_assembly(), expected)


class TestBufferAtomicAndOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_atomic_and_in_bounds(self) -> None:
        # Pre-seed address 100 with 0xFF00FF00
        self.mem.write_u32(100, 0xFF00FF00)
        ret, in_bounds = ref_buffer_atomic_and(
            self.mem, self.desc, soffset=0, vaddr=100, and_val=0xF0F0F0F0, glc=False
        )
        self.assertTrue(in_bounds)
        # Memory should now hold 0xFF00FF00 & 0xF0F0F0F0 = 0xF000F000
        self.assertEqual(self.mem.read_u32(100), 0xF000F000)

    def test_atomic_and_glc_returns_pre_op_value(self) -> None:
        self.mem.write_u32(200, 0x12345678)
        ret, in_bounds = ref_buffer_atomic_and(
            self.mem, self.desc, soffset=0, vaddr=200, and_val=0x00FF00FF, glc=True
        )
        self.assertTrue(in_bounds)
        # glc returns old value (0x12345678)
        self.assertEqual(ret, 0x12345678)
        # memory updated to 0x00340078
        self.assertEqual(self.mem.read_u32(200), 0x00340078)

    def test_out_of_bounds_rejection(self) -> None:
        # num_records is 1024; offset 1024 is out of bounds
        self.mem.write_u32(1024, 0xAAAAAAAA)
        ret, in_bounds = ref_buffer_atomic_and(
            self.mem, self.desc, soffset=0, vaddr=1024, and_val=0x55555555, glc=True
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u32(1024), 0xAAAAAAAA)

    def test_bitwise_and_identities(self) -> None:
        # Identity: X & 0xFFFFFFFF == X
        self.mem.write_u32(0, 0xDEADBEEF)
        ret, in_bounds = ref_buffer_atomic_and(
            self.mem, self.desc, soffset=0, vaddr=0, and_val=0xFFFFFFFF, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xDEADBEEF)
        self.assertEqual(self.mem.read_u32(0), 0xDEADBEEF)

        # Zero annihilator: X & 0 == 0
        ret, in_bounds = ref_buffer_atomic_and(
            self.mem, self.desc, soffset=0, vaddr=0, and_val=0, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xDEADBEEF)
        self.assertEqual(self.mem.read_u32(0), 0)

        # Idempotence: 0 & 0 == 0
        ret, in_bounds = ref_buffer_atomic_and(
            self.mem, self.desc, soffset=0, vaddr=0, and_val=0x1234, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u32(0), 0)

    def test_inst_offset_and_soffset_combination(self) -> None:
        # Total offset = soffset(64) + inst_offset(16) + vaddr(20) = 100
        self.mem.write_u32(100, 0x0F0F0F0F)
        ret, in_bounds = ref_buffer_atomic_and(
            self.mem, self.desc, soffset=64, vaddr=20, and_val=0x03030303, inst_offset=16, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x0F0F0F0F)
        self.assertEqual(self.mem.read_u32(100), 0x03030303)

    def test_descriptor_to_from_dwords_roundtrip(self) -> None:
        desc = BufferResourceDescriptor(
            base_address=0x00007FFF12340000,
            stride=64,
            num_records=65536,
            format_flags=0x12345678,
        )
        d0, d1, d2, d3 = desc.to_dwords()
        restored = BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)
        self.assertEqual(desc.base_address, restored.base_address)
        self.assertEqual(desc.stride, restored.stride)
        self.assertEqual(desc.num_records, restored.num_records)
        self.assertEqual(desc.format_flags, restored.format_flags)


if __name__ == "__main__":
    unittest.main()
