"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_umax.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory unsigned atomic maximum:
  buffer_atomic_umax vdata, vaddr, srsrc, soffset [offset:N] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR containing the 32-bit unsigned dword value to compare atomically. If 'glc' is enabled,
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
       The atomic maximum is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to vdata.
  3. Atomic Operation:
     For in-bounds accesses:
       old_val = memory[effective_address]
       new_val = max(old_val, vdata)  # Unsigned integer comparison in [0, 2^32 - 1]
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


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def s32(val: int) -> int:
    """Normalize integer to signed 32-bit range [-2^31, 2^31 - 1]."""
    v = val & MASK32
    return v - 0x100000000 if (v & 0x80000000) else v


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


def ref_buffer_atomic_umax(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    comp_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_umax.

    Returns:
      (return_val, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS
    if effective_offset + 4 > desc.num_records:
        return (0 if glc else (comp_val & MASK32)), False

    phys_addr = (desc.base_address + effective_offset) & 0xFFFFFFFFFFFFFFFF
    old_val = mem.read_u32(phys_addr)
    # Unsigned maximum comparison
    new_val = max(old_val, comp_val & MASK32)
    mem.write_u32(phys_addr, new_val)

    ret_val = old_val if glc else (comp_val & MASK32)
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
    BUFFER_ATOMIC_UMAX = "buffer_atomic_umax"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_umax instruction."""
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
# Condition Codes, Status Flags, and Carry/Borrow Isolation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BufferAtomicConditionCodes:
    """Architectural integer condition codes for buffer atomic operations."""
    zero: bool
    negative: bool
    positive: bool
    value_updated: bool


@dataclasses.dataclass(frozen=True)
class BufferStatusFlags:
    """Buffer memory status flags."""
    in_bounds: bool
    oob_fault: bool


def compute_buffer_condition_codes(old_val: int, comp_val: int, ret_val: int) -> BufferAtomicConditionCodes:
    """Computes condition codes evaluating the result and unsigned max boundary."""
    signed_ret = s32(ret_val)
    updated = (comp_val & MASK32) > (old_val & MASK32)
    return BufferAtomicConditionCodes(
        zero=(signed_ret == 0),
        negative=(signed_ret < 0),
        positive=(signed_ret > 0),
        value_updated=updated,
    )


def compute_buffer_status_flags(in_bounds: bool) -> BufferStatusFlags:
    """Computes buffer memory bounds flags."""
    return BufferStatusFlags(
        in_bounds=in_bounds,
        oob_fault=not in_bounds,
    )


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarBufferInterpreter:
    """Deterministic scalar interpreter for buffer_atomic_umax."""

    def __init__(self, mem_size: int = 65536) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.mem: BufferMemory = BufferMemory(mem_size)
        self.vcc: int = 0
        self.last_condition_codes: BufferAtomicConditionCodes | None = None
        self.last_status_flags: BufferStatusFlags | None = None
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
        """Executes a single buffer_atomic_umax instruction."""
        vdata = self.read_reg_u32(RegKind.VGPR, inst.vdata.index)
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        desc = self.get_buffer_descriptor(inst.srsrc.index)

        soffset = (
            self.read_reg_u32(inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        eff_off = (soffset & MASK32) + (inst.mods.offset & 0xFFF) + (vaddr if inst.mods.offen else 0)
        old_val = self.mem.read_u32(desc.base_address + eff_off) if eff_off + 4 <= desc.num_records else 0

        ret_val, in_bounds = ref_buffer_atomic_umax(
            self.mem,
            desc,
            soffset=soffset,
            vaddr=vaddr,
            comp_val=vdata,
            inst_offset=inst.mods.offset,
            glc=inst.mods.glc,
            offen=inst.mods.offen,
        )

        if inst.mods.glc:
            self.write_reg_u32(RegKind.VGPR, inst.vdata.index, ret_val)

        self.last_condition_codes = compute_buffer_condition_codes(
            old_val, vdata, ret_val
        )
        self.last_status_flags = compute_buffer_status_flags(in_bounds)
        # Note: buffer atomic instructions never modify VCC; carry/borrow register remains untouched.

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
        """Executes a buffer_atomic_umax instruction across all lanes in the wavefront."""
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
            ret_val, in_bounds = ref_buffer_atomic_umax(
                self.mem,
                desc,
                soffset=soffset,
                vaddr=vaddr,
                comp_val=vdata,
                inst_offset=inst.mods.offset,
                glc=inst.mods.glc,
                offen=inst.mods.offen,
            )

            if inst.mods.glc:
                self.write_lane_reg_u32(lane, RegKind.VGPR, inst.vdata.index, ret_val)

            lane_results.append((ret_val, in_bounds))

        return lane_results


class TestBufferAtomicUmaxOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_atomic_umax_in_bounds(self) -> None:
        self.mem.write_u32(100, 50)
        ret, in_bounds = ref_buffer_atomic_umax(
            self.mem, self.desc, soffset=0, vaddr=100, comp_val=150, glc=False
        )
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(100), 150)

    def test_atomic_umax_smaller_value_no_change(self) -> None:
        self.mem.write_u32(100, 50)
        ret, in_bounds = ref_buffer_atomic_umax(
            self.mem, self.desc, soffset=0, vaddr=100, comp_val=25, glc=False
        )
        self.assertTrue(in_bounds)
        # 50 > 25, memory stays 50
        self.assertEqual(self.mem.read_u32(100), 50)

    def test_atomic_umax_glc_returns_pre_op_value(self) -> None:
        self.mem.write_u32(200, 1000)
        ret, in_bounds = ref_buffer_atomic_umax(
            self.mem, self.desc, soffset=0, vaddr=200, comp_val=2000, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 1000)
        self.assertEqual(self.mem.read_u32(200), 2000)

    def test_atomic_umax_unsigned_order_extremes(self) -> None:
        # Unsigned order: 0xFFFFFFFF (4294967295) > 0x7FFFFFFF (2147483647)
        self.mem.write_u32(0, 0x7FFFFFFF)
        ret, in_bounds = ref_buffer_atomic_umax(
            self.mem, self.desc, soffset=0, vaddr=0, comp_val=0xFFFFFFFF, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x7FFFFFFF)
        self.assertEqual(self.mem.read_u32(0), 0xFFFFFFFF)

        # 0x80000000 (2147483648) > 0x00000000
        self.mem.write_u32(4, 0)
        ret, in_bounds = ref_buffer_atomic_umax(
            self.mem, self.desc, soffset=0, vaddr=4, comp_val=0x80000000, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u32(4), 0x80000000)

    def test_out_of_bounds_rejection(self) -> None:
        self.mem.write_u32(1024, 500)
        ret, in_bounds = ref_buffer_atomic_umax(
            self.mem, self.desc, soffset=0, vaddr=1024, comp_val=1000, glc=True
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u32(1024), 500)

    def test_inst_offset_and_soffset_combination(self) -> None:
        # Total offset = soffset(64) + inst_offset(16) + vaddr(20) = 100
        self.mem.write_u32(100, 10)
        ret, in_bounds = ref_buffer_atomic_umax(
            self.mem, self.desc, soffset=64, vaddr=20, comp_val=50, inst_offset=16, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 10)
        self.assertEqual(self.mem.read_u32(100), 50)


class TestBufferAtomicUmaxAST(unittest.TestCase):
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
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(offset=16, glc=True, offen=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_umax v0, v1, s[4:7], s8 offen offset:16 glc",
        )

        inst_imm = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 2, count=1),
            vaddr=Register(RegKind.VGPR, 3, count=1),
            srsrc=Register(RegKind.SGPR, 12, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True),
        )
        self.assertEqual(
            inst_imm.to_assembly(),
            "buffer_atomic_umax v2, v3, s[12:15], 0 offen",
        )

    def test_instruction_validation(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMAX,
                vdata=Register(RegKind.SGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMAX,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=2),
                soffset=0,
            )

        with self.assertRaises(ValueError):
            BufferAtomicModifiers(offset=5000)

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        ))
        expected = "buffer_atomic_umax v0, v1, s[4:7], 0 offen glc"
        self.assertEqual(prog.to_assembly(), expected)


class TestBufferAtomicUmaxScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_scalar_atomic_umax_execution(self) -> None:
        self.interp.mem.write_u32(64, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 150)  # vdata
        self.interp.write_reg_u32(RegKind.VGPR, 1, 64)   # vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        # 150 > 100 -> memory updated to 150
        self.assertEqual(self.interp.mem.read_u32(64), 150)
        # Without glc, vdata in register remains untouched
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 150)

        # Smaller value: does not decrease memory
        self.interp.write_reg_u32(RegKind.VGPR, 0, 80)
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(self.interp.mem.read_u32(64), 150)

    def test_scalar_atomic_umax_glc_writeback(self) -> None:
        self.interp.mem.write_u32(128, 500)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 800)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 128)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 500)
        # vdata updated to pre-op value (500)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 500)
        # Memory updated to max(500, 800) = 800
        self.assertEqual(self.interp.mem.read_u32(128), 800)

    def test_scalar_oob_suppression(self) -> None:
        self.interp.mem.write_u32(1024, 777)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 1000)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 1024)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
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
        self.assertEqual(self.interp.mem.read_u32(1024), 777)

    def test_scalar_program_chain(self) -> None:
        self.interp.mem.write_u32(0, 10)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        prog = Program()
        # inst 1: max(10, 50) -> 50
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        ))
        # inst 2: max(50, 30) -> 50
        self.interp.write_reg_u32(RegKind.VGPR, 2, 30)
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 2, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        ))
        # inst 3: max(50, 200) -> 200
        self.interp.write_reg_u32(RegKind.VGPR, 3, 200)
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 3, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        ))

        self.interp.run_program(prog)
        self.assertEqual(self.interp.mem.read_u32(0), 200)
        self.assertEqual(len(self.interp.trace), 3)


class TestBufferAtomicUmaxSIMD(unittest.TestCase):
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
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, (lane + 1) * 10)  # comp value
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)         # address

        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertTrue(results[lane][1])
            self.assertEqual(simd.mem.read_u32(lane * 4), (lane + 1) * 10)

    def test_wave64_convergent_atomic_umax(self) -> None:
        # All 64 lanes atomically umax to address 0 with strictly increasing values
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.mem.write_u32(0, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        for lane in range(64):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, (lane + 1) * 100)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0)

        results = simd.step(inst)
        # Maximum value in memory must be 64 * 100 = 6400
        self.assertEqual(simd.mem.read_u32(0), 6400)
        # With glc, each lane receives the previous value: lane 0 saw 0, lane 1 saw 100, ..., lane 63 saw 6300
        returned_pre_vals = [results[lane][0] for lane in range(64)]
        self.assertEqual(returned_pre_vals, [lane * 100 for lane in range(64)])

    def test_wave32_exec_masking_partial_lanes(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        exec_mask = 0x000000FF  # Only lanes 0..7 active
        simd.set_exec_mask(exec_mask)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        canary = 0x1000
        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 0x2000)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)
            simd.mem.write_u32(lane * 4, canary)

        results = simd.step(inst)
        for lane in range(32):
            if lane < 8:
                self.assertEqual(results[lane][0], canary)
                self.assertEqual(simd.mem.read_u32(lane * 4), 0x2000)
            else:
                self.assertIsNone(results[lane][0])
                # Memory untouched for inactive lanes
                self.assertEqual(simd.mem.read_u32(lane * 4), canary)
                # Register vdata untouched for inactive lanes
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 0x2000)

    def test_wave64_exec_masking_all_inactive(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        simd.mem.write_u32(0, 1234)
        for lane in range(64):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 5000)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0)

        results = simd.step(inst)
        self.assertEqual(results, [(None, False)] * 64)
        self.assertEqual(simd.mem.read_u32(0), 1234)
        for lane in range(64):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 5000)


class TestBufferAtomicUmaxRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_aliasing_vdata_equals_vaddr_with_glc(self) -> None:
        # v0 serves as both vdata (comp value) and vaddr (byte offset)
        # Pre-seed address 64 with 50
        self.interp.mem.write_u32(64, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 64)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        # Address 64 should now hold max(50, 64) = 64
        self.assertEqual(self.interp.mem.read_u32(64), 64)
        # v0 receives pre-op value (50)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 50)

    def test_aliasing_vdata_equals_vaddr_without_glc(self) -> None:
        self.interp.mem.write_u32(32, 200)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 32)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        # max(200, 32) = 200
        self.assertEqual(self.interp.mem.read_u32(32), 200)
        # Without glc, v0 retains its original value (32)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 32)

    def test_unrelated_vgprs_preserved(self) -> None:
        for idx in range(10, 20):
            self.interp.write_reg_u32(RegKind.VGPR, idx, 0x77000000 + idx)

        self.interp.write_reg_u32(RegKind.VGPR, 0, 10)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
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
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
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
        simd.mem.write_u32(16, 10)

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 16)  # address and value = 16

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        simd.step(inst)

        # Lane 0 saw 10, Lanes 1..3 saw 16
        expected_active = [10, 16, 16, 16]
        for lane in range(4):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), expected_active[lane])
        # Inactive lanes: v0 untouched (16)
        for lane in range(4, 32):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 16)


class TestBufferAtomicUmaxConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow isolation."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_condition_codes_zero_pos_neg(self) -> None:
        # Pre-seed memory at 0 with 0
        self.interp.mem.write_u32(0, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        # Pre-op return value is 0
        self.interp.step(inst)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertTrue(cc.zero)
        self.assertFalse(cc.negative)
        self.assertFalse(cc.positive)

        # Update to positive value 10: pre-op is 0, now mem is 10
        self.interp.write_reg_u32(RegKind.VGPR, 0, 10)
        self.interp.step(inst)
        # Next step: pre-op returned is 10 (positive)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 20)
        self.interp.step(inst)
        cc_pos = self.interp.last_condition_codes
        self.assertIsNotNone(cc_pos)
        assert cc_pos is not None
        self.assertFalse(cc_pos.zero)
        self.assertFalse(cc_pos.negative)
        self.assertTrue(cc_pos.positive)

        # Negative in signed 32-bit (bit 31 set)
        self.interp.mem.write_u32(4, 0x80000000)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x90000000)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 4)
        self.interp.step(inst)
        cc_neg = self.interp.last_condition_codes
        self.assertIsNotNone(cc_neg)
        assert cc_neg is not None
        self.assertTrue(cc_neg.negative)
        self.assertFalse(cc_neg.positive)

    def test_value_updated_flag(self) -> None:
        self.interp.mem.write_u32(0, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        # 50 <= 100, not updated
        self.assertFalse(cc.value_updated)

        # 200 > 100, updated
        self.interp.write_reg_u32(RegKind.VGPR, 0, 200)
        self.interp.step(inst)
        cc_up = self.interp.last_condition_codes
        self.assertIsNotNone(cc_up)
        assert cc_up is not None
        self.assertTrue(cc_up.value_updated)

    def test_status_flags_in_bounds_and_oob(self) -> None:
        self.interp.write_reg_u32(RegKind.VGPR, 0, 1)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 100)  # in bounds (< 1024)

        inst_in = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst_in)
        flags_in = self.interp.last_status_flags
        self.assertIsNotNone(flags_in)
        assert flags_in is not None
        self.assertTrue(flags_in.in_bounds)
        self.assertFalse(flags_in.oob_fault)

        # OOB access
        self.interp.write_reg_u32(RegKind.VGPR, 1, 2000)
        self.interp.step(inst_in)
        flags_oob = self.interp.last_status_flags
        self.assertIsNotNone(flags_oob)
        assert flags_oob is not None
        self.assertFalse(flags_oob.in_bounds)
        self.assertTrue(flags_oob.oob_fault)

    def test_carry_borrow_vcc_isolation(self) -> None:
        canary = 0x1234567887654321
        self.interp.vcc = canary

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)
        self.assertEqual(self.interp.vcc, canary)


class TestBufferAtomicUmaxBoundaryExtremes(unittest.TestCase):
    """Unit tests for boundary extremes, address wrapping, and OOB conditions."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)

    def test_last_valid_dword_boundary(self) -> None:
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=100)
        # Last valid 4-byte dword starts at offset 96 (96 + 4 = 100)
        self.mem.write_u32(96, 42)
        ret, in_bounds = ref_buffer_atomic_umax(self.mem, desc, soffset=0, vaddr=96, comp_val=50, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 42)
        self.assertEqual(self.mem.read_u32(96), 50)

        # Offsets 97, 98, 99 all exceed num_records (97 + 4 = 101 > 100)
        for oob_off in (97, 98, 99, 100):
            ret, in_bounds = ref_buffer_atomic_umax(self.mem, desc, soffset=0, vaddr=oob_off, comp_val=100, glc=True)
            self.assertFalse(in_bounds)
            self.assertEqual(ret, 0)

    def test_zero_capacity_buffer(self) -> None:
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=0)
        ret, in_bounds = ref_buffer_atomic_umax(self.mem, desc, soffset=0, vaddr=0, comp_val=1, glc=True)
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)

    def test_unsigned_integer_extremes(self) -> None:
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        # 0x7FFFFFFF vs 0x80000000:
        # In signed comparison, 0x80000000 (-2147483648) < 0x7FFFFFFF (+2147483647).
        # In unsigned comparison, 0x80000000 (2147483648) > 0x7FFFFFFF (2147483647).
        self.mem.write_u32(0, 0x7FFFFFFF)
        ret, in_bounds = ref_buffer_atomic_umax(self.mem, desc, soffset=0, vaddr=0, comp_val=0x80000000, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x7FFFFFFF)
        self.assertEqual(self.mem.read_u32(0), 0x80000000)

        # 0xFFFFFFFF (max unsigned 32-bit) vs 0xFFFFFFFE
        self.mem.write_u32(4, 0xFFFFFFFE)
        ret, in_bounds = ref_buffer_atomic_umax(self.mem, desc, soffset=0, vaddr=4, comp_val=0xFFFFFFFF, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xFFFFFFFE)
        self.assertEqual(self.mem.read_u32(4), 0xFFFFFFFF)

    def test_float_and_nan_bit_patterns_preserve_raw_unsigned_comparison(self) -> None:
        # Buffer atomic umax treats all bits as raw unsigned integers, ignoring IEEE 754 semantics
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        # Float -inf (0xFF800000) has unsigned int value 4286578688, > float qNaN (0x7FC00000: 2143289344)
        qnan_bits = 0x7FC00000
        neg_inf_bits = 0xFF800000
        self.mem.write_u32(8, qnan_bits)
        ret, in_bounds = ref_buffer_atomic_umax(self.mem, desc, soffset=0, vaddr=8, comp_val=neg_inf_bits, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, qnan_bits)
        self.assertEqual(self.mem.read_u32(8), neg_inf_bits)

        # Float -0.0 (0x80000000) > +0.0 (0x00000000) as unsigned int (2147483648 > 0)
        self.mem.write_u32(12, 0)
        ret, in_bounds = ref_buffer_atomic_umax(self.mem, desc, soffset=0, vaddr=12, comp_val=0x80000000, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u32(12), 0x80000000)


class TestBufferAtomicUmaxNegativeControls(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and invalid AST inputs."""

    def test_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1, count=1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256, count=1)
        with self.assertRaises(ValueError):
            # SGPR 254 with count 4 exceeds 256
            Register(RegKind.SGPR, 254, count=4)

    def test_invalid_destination_kind(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMAX,
                vdata=Register(RegKind.SGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

    def test_invalid_descriptor_size(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMAX,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=2),
                soffset=0,
            )

    def test_invalid_soffset_kind(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMAX,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=Register(RegKind.VGPR, 8, count=1),
            )

    def test_invalid_wave_size(self) -> None:
        with self.assertRaises(ValueError):
            SIMDBufferWavefrontInterpreter(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDBufferWavefrontInterpreter(wave_size=128)


class TestBufferAtomicUmaxMutationDetection(unittest.TestCase):
    """Mutation tests: verifies that intentional sabotages/mutations of the oracle are caught."""

    def test_mutation_oob_check_omission(self) -> None:
        # Sabotage: ignoring NUM_RECORDS bounds check
        mem = BufferMemory(256)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=32)

        def mutated_atomic_umax_no_bounds(m: BufferMemory, d: BufferResourceDescriptor, off: int, val: int) -> int:
            old = m.read_u32(off)
            m.write_u32(off, max(old, val))
            return old

        # An access at offset 64 should be rejected by the oracle
        _, in_bounds = ref_buffer_atomic_umax(mem, desc, soffset=0, vaddr=64, comp_val=10, glc=True)
        self.assertFalse(in_bounds)

        # Mutated oracle incorrectly allows it
        with self.assertRaises(AssertionError):
            self.assertEqual(in_bounds, True)

    def test_mutation_glc_returns_post_op_value(self) -> None:
        # Sabotage: returning new value instead of old value
        mem = BufferMemory(256)
        mem.write_u32(0, 100)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)

        def mutated_atomic_umax_post_op(m: BufferMemory, d: BufferResourceDescriptor, off: int, val: int) -> int:
            old = m.read_u32(off)
            new_val = max(old, val) & MASK32
            m.write_u32(off, new_val)
            return new_val  # BUG: returns post-op instead of pre-op

        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_atomic_umax_post_op(mem, desc, 0, 500), 100)

    def test_mutation_signed_vs_unsigned_comparison_flip(self) -> None:
        # Sabotage: signed integer max instead of unsigned integer max
        mem = BufferMemory(256)
        mem.write_u32(0, 0x7FFFFFFF)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)

        def mutated_atomic_smax(m: BufferMemory, d: BufferResourceDescriptor, off: int, val: int) -> int:
            old = m.read_u32(off)
            # BUG: s32 treats 0x80000000 as -2147483648 < 0x7FFFFFFF (2147483647)
            new_val = max(s32(old), s32(val)) & MASK32
            m.write_u32(off, new_val)
            return old

        mutated_atomic_smax(mem, desc, 0, 0x80000000)
        with self.assertRaises(AssertionError):
            # Correct unsigned max should be 0x80000000, but signed bug leaves 0x7FFFFFFF
            self.assertEqual(mem.read_u32(0), 0x80000000)

    def test_mutation_min_instead_of_max(self) -> None:
        # Sabotage: using min instead of max
        mem = BufferMemory(256)
        mem.write_u32(0, 100)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)

        def mutated_atomic_umin(m: BufferMemory, d: BufferResourceDescriptor, off: int, val: int) -> int:
            old = m.read_u32(off)
            new_val = min(old, val) & MASK32
            m.write_u32(off, new_val)
            return old

        mutated_atomic_umin(mem, desc, 0, 200)
        with self.assertRaises(AssertionError):
            # Memory should be 200, but bug produced 100
            self.assertEqual(mem.read_u32(0), 200)

    def test_mutation_simd_exec_leak(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=256)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        simd.set_buffer_descriptor(4, desc)
        simd.set_exec_mask(0)  # all lanes disabled

        canary = 0x99999999
        simd.mem.write_u32(0, canary)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        simd.step(inst)
        # Memory must remain canary
        self.assertEqual(simd.mem.read_u32(0), canary)

        # Mutated unconditional write would corrupt memory
        simd.mem.write_u32(0, 500)
        with self.assertRaises(AssertionError):
            self.assertEqual(simd.mem.read_u32(0), canary)


if __name__ == "__main__":
    unittest.main()
