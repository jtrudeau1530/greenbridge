from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_umin.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory unsigned atomic minimum:
  buffer_atomic_umin vdata, vaddr, srsrc, soffset [offset:N] [offen] [idxen] [glc] [slc]

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
  3. Atomic Unsigned Minimum Operation:
     For in-bounds accesses:
       old_val = memory[effective_address]
       new_val = min(unsigned32(old_val), unsigned32(vdata))
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
UINT32_MAX: int = 0xFFFFFFFF
UINT32_MIN: int = 0x00000000


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


def ref_buffer_atomic_umin(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    cmp_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference mathematical oracle for RDNA2 buffer_atomic_umin.

    Computes:
      effective_offset = soffset + inst_offset + (vaddr if offen else 0)
      if effective_offset + 4 <= desc.num_records:
        old_val = mem[effective_address]
        new_val = min(u32(old_val), u32(cmp_val))
        mem[effective_address] = new_val
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

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u32(phys_addr)
    min_val = min(u32(old_val), u32(cmp_val))
    mem.write_u32(phys_addr, min_val)

    ret_val = old_val if glc else (cmp_val & MASK32)
    return ret_val, True


class TestBufferAtomicUminOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_unsigned_min_basic(self) -> None:
        # Initial memory has 20, cmp with 10 -> min is 10
        self.mem.write_u32(0, 20)
        ret, ok = ref_buffer_atomic_umin(self.mem, self.desc, soffset=0, vaddr=0, cmp_val=10, glc=True)
        self.assertTrue(ok)
        self.assertEqual(ret, 20)
        self.assertEqual(self.mem.read_u32(0), 10)

        # Compare with 30 -> min is still 10
        ret2, ok2 = ref_buffer_atomic_umin(self.mem, self.desc, soffset=0, vaddr=0, cmp_val=30, glc=True)
        self.assertTrue(ok2)
        self.assertEqual(ret2, 10)
        self.assertEqual(self.mem.read_u32(0), 10)

    def test_unsigned_vs_signed_difference(self) -> None:
        # Key distinction between smin and umin:
        # 0xFFFFFFFF is -1 in signed, but 4,294,967,295 in unsigned!
        # Comparing 10 vs 0xFFFFFFFF:
        # smin(10, 0xFFFFFFFF) = 0xFFFFFFFF (-1)
        # umin(10, 0xFFFFFFFF) = 10
        self.mem.write_u32(4, 10)
        ret, ok = ref_buffer_atomic_umin(self.mem, self.desc, soffset=0, vaddr=4, cmp_val=0xFFFFFFFF, glc=True)
        self.assertTrue(ok)
        self.assertEqual(ret, 10)
        self.assertEqual(self.mem.read_u32(4), 10)

        # 0x80000000 (2^31) vs 5:
        # In unsigned: 5 < 0x80000000 -> min is 5
        self.mem.write_u32(8, 0x80000000)
        ret3, ok3 = ref_buffer_atomic_umin(self.mem, self.desc, soffset=0, vaddr=8, cmp_val=5, glc=True)
        self.assertTrue(ok3)
        self.assertEqual(ret3, 0x80000000)
        self.assertEqual(self.mem.read_u32(8), 5)

    def test_unsigned_min_extremes(self) -> None:
        # 0 vs UINT32_MAX -> min is 0
        self.mem.write_u32(12, UINT32_MAX)
        ret, ok = ref_buffer_atomic_umin(self.mem, self.desc, soffset=0, vaddr=12, cmp_val=UINT32_MIN, glc=True)
        self.assertTrue(ok)
        self.assertEqual(ret, UINT32_MAX)
        self.assertEqual(self.mem.read_u32(12), UINT32_MIN)

    def test_oob_suppression_and_glc_zero(self) -> None:
        # Buffer capacity is 1024, access at 1024 -> OOB
        self.mem.write_u32(100, 42)
        ret, ok = ref_buffer_atomic_umin(self.mem, self.desc, soffset=0, vaddr=1024, cmp_val=5, glc=True)
        self.assertFalse(ok)
        self.assertEqual(ret, 0)  # glc on OOB returns 0


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions for buffer_atomic_umin
# ==============================================================================

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
    BUFFER_ATOMIC_UMIN = "buffer_atomic_umin"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_umin instruction."""
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


class TestBufferAtomicUminAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_formatting(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=Register(RegKind.SGPR, 4),
            mods=BufferAtomicModifiers(offset=16, glc=True, offen=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_umin v0, v1, s[0:3], s4 offen offset:16 glc",
        )

    def test_instruction_validation(self) -> None:
        # Invalid vdata (must be VGPR)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN,
                vdata=Register(RegKind.SGPR, 0),
                vaddr=Register(RegKind.VGPR, 1),
                srsrc=Register(RegKind.SGPR, 0, count=4),
                soffset=0,
            )

        # Invalid srsrc (must be count=4)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN,
                vdata=Register(RegKind.VGPR, 0),
                vaddr=Register(RegKind.VGPR, 1),
                srsrc=Register(RegKind.SGPR, 0, count=2),
                soffset=0,
            )

    def test_program_representation(self) -> None:
        prog = Program()
        prog.append(
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN,
                vdata=Register(RegKind.VGPR, 2),
                vaddr=Register(RegKind.VGPR, 3),
                srsrc=Register(RegKind.SGPR, 8, count=4),
                soffset=0,
                mods=BufferAtomicModifiers(glc=False),
            )
        )
        self.assertIn("buffer_atomic_umin v2, v3, s[8:11], 0 offen", prog.to_assembly())


# ==============================================================================
# Deterministic Scalar Instruction Interpreter for buffer_atomic_umin
# ==============================================================================

class ScalarBufferInterpreter:
    """Deterministic scalar interpreter for buffer_atomic_umin."""

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
        """Executes a single buffer_atomic_umin instruction."""
        vdata = self.read_reg_u32(RegKind.VGPR, inst.vdata.index)
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        desc = self.get_buffer_descriptor(inst.srsrc.index)

        soffset = (
            self.read_reg_u32(inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        ret_val, in_bounds = ref_buffer_atomic_umin(
            self.mem,
            desc,
            soffset=soffset,
            vaddr=vaddr,
            cmp_val=vdata,
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


class TestBufferAtomicUminScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=1024)
        self.interp.set_buffer_descriptor(0, self.desc)

    def test_scalar_umin_glc_execution(self) -> None:
        # Memory at address 0 has 50
        self.interp.mem.write_u32(0, 50)
        # v0 (vdata) = 30
        self.interp.write_reg_u32(RegKind.VGPR, 0, 30)
        # v1 (vaddr) = 0
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True, offen=True),
        )
        ret_val, ok = self.interp.step(inst)
        self.assertTrue(ok)
        self.assertEqual(ret_val, 50)  # glc returns old value
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 50)  # vdata overwritten with old val
        self.assertEqual(self.interp.mem.read_u32(0), 30)  # memory holds min(50, 30) = 30

    def test_scalar_umin_no_glc(self) -> None:
        self.interp.mem.write_u32(8, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 75)
        self.interp.write_reg_u32(RegKind.VGPR, 3, 8)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN,
            vdata=Register(RegKind.VGPR, 2),
            vaddr=Register(RegKind.VGPR, 3),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False, offen=True),
        )
        ret_val, ok = self.interp.step(inst)
        self.assertTrue(ok)
        # Without glc, vdata in register is NOT overwritten with old value
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 2), 75)
        self.assertEqual(self.interp.mem.read_u32(8), 75)


# ==============================================================================
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ==============================================================================

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
        """Executes a buffer_atomic_umin instruction across all lanes in the wavefront."""
        desc = self.get_buffer_descriptor(inst.srsrc.index)
        soffset = (
            self.read_lane_reg_u32(0, inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        lane_inputs: list[tuple[int, int]] = []
        for lane in range(self.wave_size):
            vd = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vdata.index)
            va = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index)
            lane_inputs.append((vd, va))

        lane_results: list[tuple[int | None, bool]] = []
        for lane in range(self.wave_size):
            is_active = (self.exec_mask & (1 << lane)) != 0
            if not is_active:
                lane_results.append((None, False))
                continue

            vdata_val, vaddr_val = lane_inputs[lane]
            ret_val, in_bounds = ref_buffer_atomic_umin(
                self.mem,
                desc,
                soffset=soffset,
                vaddr=vaddr_val,
                cmp_val=vdata_val,
                inst_offset=inst.mods.offset,
                glc=inst.mods.glc,
                offen=inst.mods.offen,
            )
            lane_results.append((ret_val, in_bounds))

            if inst.mods.glc:
                self.write_lane_reg_u32(lane, RegKind.VGPR, inst.vdata.index, ret_val)

        return lane_results


class TestBufferAtomicUminSIMD(unittest.TestCase):
    """Test suite for SIMD wavefront multi-lane divergence and EXEC masking."""

    def test_wave32_divergent_addresses(self) -> None:
        sim = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=2048)
        sim.set_buffer_descriptor(0, desc)

        # Initialize distinct buffer locations
        for i in range(32):
            sim.mem.write_u32(i * 4, 100 + i)
            sim.write_lane_reg_u32(i, RegKind.VGPR, 0, 50)     # vdata = 50
            sim.write_lane_reg_u32(i, RegKind.VGPR, 1, i * 4)  # vaddr = i * 4

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True, offen=True),
        )
        results = sim.step(inst)

        for i in range(32):
            ret, in_bounds = results[i]
            self.assertTrue(in_bounds)
            self.assertEqual(ret, 100 + i)  # glc returns old value
            self.assertEqual(sim.mem.read_u32(i * 4), 50)  # min(100+i, 50) = 50

    def test_wave64_divergent_exec_mask(self) -> None:
        sim = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=2048)
        sim.set_buffer_descriptor(0, desc)

        # Alternating EXEC mask
        sim.set_exec_mask(0x55555555_55555555)

        for i in range(64):
            sim.mem.write_u32(i * 4, 200)
            sim.write_lane_reg_u32(i, RegKind.VGPR, 0, 10)
            sim.write_lane_reg_u32(i, RegKind.VGPR, 1, i * 4)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True, offen=True),
        )
        results = sim.step(inst)

        for i in range(64):
            if i % 2 == 0:
                # Active lane
                self.assertEqual(sim.mem.read_u32(i * 4), 10)
                self.assertEqual(sim.read_lane_reg_u32(i, RegKind.VGPR, 0), 200)
            else:
                # Inactive lane: untouched memory and untouched register
                self.assertEqual(sim.mem.read_u32(i * 4), 200)
                self.assertEqual(sim.read_lane_reg_u32(i, RegKind.VGPR, 0), 10)


if __name__ == "__main__":
    unittest.main()
