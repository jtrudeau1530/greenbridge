from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_smax.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory signed atomic maximum:
  buffer_atomic_smax vdata, vaddr, srsrc, soffset [offset:N] [offen] [idxen] [glc] [slc]

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
  3. Atomic Signed Maximum Operation:
     For in-bounds accesses:
       old_val = memory[effective_address]
       new_val = max(signed32(old_val), signed32(vdata))
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


def ref_buffer_atomic_smax(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    cmp_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference mathematical oracle for RDNA2 buffer_atomic_smax.

    Computes:
      effective_offset = soffset + inst_offset + (vaddr if offen else 0)
      if effective_offset + 4 <= desc.num_records:
        old_val = mem[effective_address]
        new_val = max(s32(old_val), s32(cmp_val))
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
    max_val = max(s32(old_val), s32(cmp_val))
    new_val = u32(max_val)
    mem.write_u32(phys_addr, new_val)

    ret_val = old_val if glc else (cmp_val & MASK32)
    return ret_val, True


class TestBufferAtomicSmaxOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_signed_max_positive_numbers(self) -> None:
        # Initial memory has 10, compare with 20 -> max is 20
        self.mem.write_u32(0, 10)
        ret, ok = ref_buffer_atomic_smax(self.mem, self.desc, soffset=0, vaddr=0, cmp_val=20, glc=True)
        self.assertTrue(ok)
        self.assertEqual(ret, 10)
        self.assertEqual(self.mem.read_u32(0), 20)

        # Compare with 15 -> max is still 20
        ret2, ok2 = ref_buffer_atomic_smax(self.mem, self.desc, soffset=0, vaddr=0, cmp_val=15, glc=True)
        self.assertTrue(ok2)
        self.assertEqual(ret2, 20)
        self.assertEqual(self.mem.read_u32(0), 20)

    def test_signed_max_negative_numbers(self) -> None:
        # Initial memory has -20, compare with -10 (0xFFFFFFF6) -> max is -10
        self.mem.write_u32(4, u32(-20))
        cmp_neg = u32(-10)
        ret, ok = ref_buffer_atomic_smax(self.mem, self.desc, soffset=0, vaddr=4, cmp_val=cmp_neg, glc=True)
        self.assertTrue(ok)
        self.assertEqual(s32(ret), -20)
        self.assertEqual(s32(self.mem.read_u32(4)), -10)

        # Compare -10 with -30 -> max is still -10
        cmp_neg30 = u32(-30)
        ret2, ok2 = ref_buffer_atomic_smax(self.mem, self.desc, soffset=0, vaddr=4, cmp_val=cmp_neg30, glc=True)
        self.assertTrue(ok2)
        self.assertEqual(s32(ret2), -10)
        self.assertEqual(s32(self.mem.read_u32(4)), -10)

    def test_signed_max_extremes(self) -> None:
        # INT32_MIN (0x80000000) vs INT32_MAX (0x7FFFFFFF) -> max is INT32_MAX
        self.mem.write_u32(8, U32_INT32_MIN)
        ret, ok = ref_buffer_atomic_smax(self.mem, self.desc, soffset=0, vaddr=8, cmp_val=U32_INT32_MAX, glc=True)
        self.assertTrue(ok)
        self.assertEqual(ret, U32_INT32_MIN)
        self.assertEqual(self.mem.read_u32(8), U32_INT32_MAX)

    def test_signed_max_with_zero(self) -> None:
        # -5 vs 0 -> max is 0
        self.mem.write_u32(12, u32(-5))
        ret, ok = ref_buffer_atomic_smax(self.mem, self.desc, soffset=0, vaddr=12, cmp_val=U32_ZERO, glc=True)
        self.assertTrue(ok)
        self.assertEqual(s32(ret), -5)
        self.assertEqual(self.mem.read_u32(12), U32_ZERO)

        # 0 vs -10 -> max is 0
        ret2, ok2 = ref_buffer_atomic_smax(self.mem, self.desc, soffset=0, vaddr=12, cmp_val=u32(-10), glc=True)
        self.assertTrue(ok2)
        self.assertEqual(ret2, U32_ZERO)
        self.assertEqual(self.mem.read_u32(12), U32_ZERO)

    def test_oob_suppression_and_glc_zero(self) -> None:
        # Buffer capacity is 1024, access at 1024 (1024 + 4 > 1024) -> OOB
        self.mem.write_u32(100, 42)
        ret, ok = ref_buffer_atomic_smax(self.mem, self.desc, soffset=0, vaddr=1024, cmp_val=50, glc=True)
        self.assertFalse(ok)
        self.assertEqual(ret, 0)  # glc on OOB returns 0


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
    BUFFER_ATOMIC_SMAX = "buffer_atomic_smax"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_smax instruction."""
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


class BufferAtomicSmaxExpr:
    """Typed symbolic expression for buffer_atomic_smax."""

    def __init__(
        self,
        soffset: int,
        vaddr: int,
        cmp_val: int,
        inst_offset: int = 0,
        glc: bool = False,
        offen: bool = True,
    ) -> None:
        self.soffset = soffset
        self.vaddr = vaddr
        self.cmp_val = cmp_val
        self.inst_offset = inst_offset
        self.glc = glc
        self.offen = offen

    def evaluate(self, mem: BufferMemory, desc: BufferResourceDescriptor) -> tuple[int, bool]:
        return ref_buffer_atomic_smax(
            mem,
            desc,
            self.soffset,
            self.vaddr,
            self.cmp_val,
            self.inst_offset,
            self.glc,
            self.offen,
        )


class TestBufferAtomicSmaxAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_definition(self) -> None:
        self.assertEqual(Opcode.BUFFER_ATOMIC_SMAX.value, "buffer_atomic_smax")

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
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8),
            mods=BufferAtomicModifiers(offen=True, offset=64, glc=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_smax v0, v1, s[4:7], s8 offen offset:64 glc",
        )

    def test_instruction_with_immediate_soffset(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 2),
            vaddr=Register(RegKind.VGPR, 3),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=False, slc=True),
        )
        self.assertEqual(inst.to_assembly(), "buffer_atomic_smax v2, v3, s[0:3], 0 slc")

    def test_program_assembly(self) -> None:
        prog = Program()
        inst1 = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
        )
        prog.append(inst1)
        self.assertEqual(prog.to_assembly(), inst1.to_assembly())

    def test_typed_expression_evaluation(self) -> None:
        mem = BufferMemory(256)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        mem.write_u32(0, u32(-50))

        expr = BufferAtomicSmaxExpr(soffset=0, vaddr=0, cmp_val=u32(-10), glc=True)
        ret, ok = expr.evaluate(mem, desc)
        self.assertTrue(ok)
        self.assertEqual(s32(ret), -50)
        self.assertEqual(s32(mem.read_u32(0)), -10)


class ScalarRegisterFile:
    """Scalar processor register file maintaining 256 VGPRs and 106 SGPRs."""

    def __init__(self) -> None:
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106

    def read_vgpr(self, idx: int) -> int:
        return self.vgprs[idx]

    def write_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32

    def read_sgpr(self, idx: int) -> int:
        return self.sgprs[idx]

    def write_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32


class ScalarBufferInterpreter:
    """Deterministic scalar interpreter executing buffer_atomic_smax."""

    def __init__(self, mem: BufferMemory, rf: ScalarRegisterFile | None = None) -> None:
        self.mem = mem
        self.rf = rf if rf is not None else ScalarRegisterFile()

    def set_buffer_descriptor(self, sgpr_base: int, desc: BufferResourceDescriptor) -> None:
        d0, d1, d2, d3 = desc.to_dwords()
        self.rf.write_sgpr(sgpr_base, d0)
        self.rf.write_sgpr(sgpr_base + 1, d1)
        self.rf.write_sgpr(sgpr_base + 2, d2)
        self.rf.write_sgpr(sgpr_base + 3, d3)

    def step(self, inst: Instruction) -> tuple[int, bool]:
        d0 = self.rf.read_sgpr(inst.srsrc.index)
        d1 = self.rf.read_sgpr(inst.srsrc.index + 1)
        d2 = self.rf.read_sgpr(inst.srsrc.index + 2)
        d3 = self.rf.read_sgpr(inst.srsrc.index + 3)
        desc = BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)

        soffset = (
            self.rf.read_sgpr(inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )
        vaddr = self.rf.read_vgpr(inst.vaddr.index)
        cmp_val = self.rf.read_vgpr(inst.vdata.index)

        ret, ok = ref_buffer_atomic_smax(
            self.mem,
            desc,
            soffset=soffset,
            vaddr=vaddr,
            cmp_val=cmp_val,
            inst_offset=inst.mods.offset,
            glc=inst.mods.glc,
            offen=inst.mods.offen,
        )

        if inst.mods.glc:
            self.rf.write_vgpr(inst.vdata.index, ret)

        return ret, ok


class TestBufferAtomicSmaxScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.rf = ScalarRegisterFile()
        self.interp = ScalarBufferInterpreter(self.mem, self.rf)
        self.desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=1024)
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_scalar_smax_basic(self) -> None:
        self.mem.write_u32(0, 100)
        self.rf.write_vgpr(0, 200)  # vdata
        self.rf.write_vgpr(1, 0)    # vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True, glc=False),
        )
        ret, ok = self.interp.step(inst)
        self.assertTrue(ok)
        self.assertEqual(self.mem.read_u32(0), 200)
        # Without glc, vdata in register file is not overwritten
        self.assertEqual(self.rf.read_vgpr(0), 200)

    def test_scalar_smax_with_glc_return(self) -> None:
        self.mem.write_u32(8, u32(-50))
        self.rf.write_vgpr(2, u32(-10))  # vdata
        self.rf.write_vgpr(3, 8)         # vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 2),
            vaddr=Register(RegKind.VGPR, 3),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True, glc=True),
        )
        ret, ok = self.interp.step(inst)
        self.assertTrue(ok)
        # Memory is updated to max(-50, -10) = -10
        self.assertEqual(s32(self.mem.read_u32(8)), -10)
        # Pre-operation value (-50) returned to vdata
        self.assertEqual(s32(ret), -50)
        self.assertEqual(s32(self.rf.read_vgpr(2)), -50)

    def test_scalar_smax_oob_suppression(self) -> None:
        self.rf.write_vgpr(0, 999)
        self.rf.write_vgpr(1, 1024)  # OOB

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True, glc=True),
        )
        ret, ok = self.interp.step(inst)
        self.assertFalse(ok)
        self.assertEqual(ret, 0)
        self.assertEqual(self.rf.read_vgpr(0), 0)

    def test_scalar_smax_repeatability(self) -> None:
        self.mem.write_u32(16, u32(-100))
        self.rf.write_vgpr(0, u32(-80))
        self.rf.write_vgpr(1, 16)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True, glc=True),
        )
        ret1, _ = self.interp.step(inst)
        self.assertEqual(s32(ret1), -100)
        self.assertEqual(s32(self.mem.read_u32(16)), -80)

        # Step again with smaller value (-90): memory remains -80, returns -80
        self.rf.write_vgpr(0, u32(-90))
        ret2, _ = self.interp.step(inst)
        self.assertEqual(s32(ret2), -80)
        self.assertEqual(s32(self.mem.read_u32(16)), -80)


class SIMDBufferWavefrontInterpreter:
    """Multi-lane SIMD wavefront interpreter for buffer_atomic_smax with EXEC masking."""

    def __init__(self, wave_size: int = 64, mem_size: int = 65536) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self._exec_mask = (1 << wave_size) - 1
        self.mem = BufferMemory(mem_size)
        self.vgprs = [[0] * 256 for _ in range(wave_size)]
        self.sgprs = [0] * 106

    @property
    def exec_mask(self) -> int:
        return self._exec_mask

    def set_exec_mask(self, mask: int) -> None:
        limit = (1 << self.wave_size) - 1
        self._exec_mask = mask & limit

    @property
    def exec_lo(self) -> int:
        return self._exec_mask & MASK32

    def set_exec_lo(self, val: int) -> None:
        self._exec_mask = (self._exec_mask & ~MASK32) | (val & MASK32)

    @property
    def exec_hi(self) -> int:
        return (self._exec_mask >> 32) & MASK32

    def set_exec_hi(self, val: int) -> None:
        if self.wave_size == 32:
            raise ValueError("Cannot set exec_hi on Wave32")
        self._exec_mask = (self._exec_mask & MASK32) | ((val & MASK32) << 32)

    def set_buffer_descriptor(self, sgpr_base: int, desc: BufferResourceDescriptor) -> None:
        d0, d1, d2, d3 = desc.to_dwords()
        self.sgprs[sgpr_base] = d0
        self.sgprs[sgpr_base + 1] = d1
        self.sgprs[sgpr_base + 2] = d2
        self.sgprs[sgpr_base + 3] = d3

    def step(self, inst: Instruction) -> list[tuple[int, bool]]:
        d0 = self.sgprs[inst.srsrc.index]
        d1 = self.sgprs[inst.srsrc.index + 1]
        d2 = self.sgprs[inst.srsrc.index + 2]
        d3 = self.sgprs[inst.srsrc.index + 3]
        desc = BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)

        soffset = (
            self.sgprs[inst.soffset.index]
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        results: list[tuple[int, bool]] = []
        for lane in range(self.wave_size):
            if (self._exec_mask >> lane) & 1:
                vaddr = self.vgprs[lane][inst.vaddr.index]
                cmp_val = self.vgprs[lane][inst.vdata.index]
                ret, ok = ref_buffer_atomic_smax(
                    self.mem,
                    desc,
                    soffset=soffset,
                    vaddr=vaddr,
                    cmp_val=cmp_val,
                    inst_offset=inst.mods.offset,
                    glc=inst.mods.glc,
                    offen=inst.mods.offen,
                )
                if inst.mods.glc:
                    self.vgprs[lane][inst.vdata.index] = ret
                results.append((ret, ok))
            else:
                results.append((0, False))

        return results


class TestBufferAtomicSmaxSIMDMultiLane(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    CANARY_OLD: int = 0xDEADBEEF

    def test_wave32_full_exec(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=4096)
        simd.set_buffer_descriptor(4, desc)

        # Initialize distinct memory slots to -100
        for i in range(32):
            simd.mem.write_u32(i * 4, u32(-100))
            simd.vgprs[i][0] = u32(i * 2 - 20)  # vdata: values from -20 to 42
            simd.vgprs[i][1] = i * 4            # vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True, glc=True),
        )
        simd.step(inst)

        for i in range(32):
            # Memory should now have max(-100, i*2 - 20) = i*2 - 20
            self.assertEqual(s32(simd.mem.read_u32(i * 4)), i * 2 - 20)
            # vdata received pre-op -100
            self.assertEqual(s32(simd.vgprs[i][0]), -100)

    def test_wave64_divergence_exec_masking(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=4096)
        simd.set_buffer_descriptor(4, desc)

        # Alternating even/odd lanes in lower and upper 32
        simd.set_exec_lo(0x55555555)
        simd.set_exec_hi(0xAAAAAAAA)

        for lane in range(64):
            simd.mem.write_u32(lane * 4, u32(-50))
            simd.vgprs[lane][0] = self.CANARY_OLD
            simd.vgprs[lane][1] = lane * 4

        # For active lanes, set vdata to a higher value: 10
        for lane in range(64):
            if (simd.exec_mask >> lane) & 1:
                simd.vgprs[lane][0] = 10

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True, glc=True),
        )
        simd.step(inst)

        for lane in range(64):
            is_active = bool((simd.exec_mask >> lane) & 1)
            if is_active:
                self.assertEqual(simd.mem.read_u32(lane * 4), 10)
                self.assertEqual(s32(simd.vgprs[lane][0]), -50)
            else:
                # Inactive lanes: memory untouched, vdata untouched
                self.assertEqual(s32(simd.mem.read_u32(lane * 4)), -50)
                self.assertEqual(simd.vgprs[lane][0], self.CANARY_OLD)

    def test_same_address_atomic_reduction(self) -> None:
        # All active lanes target address 0: computes the global max across all lanes
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=256)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        simd.set_buffer_descriptor(0, desc)

        simd.mem.write_u32(0, u32(-1000))
        for lane in range(64):
            simd.vgprs[lane][0] = u32(lane - 30)  # values from -30 to +33
            simd.vgprs[lane][1] = 0               # all address 0

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True, glc=False),
        )
        simd.step(inst)
        # Global maximum across all lanes is 33
        self.assertEqual(s32(simd.mem.read_u32(0)), 33)

    def test_zero_exec_mask_preserves_everything(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=256)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        simd.set_buffer_descriptor(0, desc)
        simd.set_exec_mask(0)

        simd.mem.write_u32(0, 42)
        for lane in range(64):
            simd.vgprs[lane][0] = self.CANARY_OLD

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True, glc=True),
        )
        simd.step(inst)

        self.assertEqual(simd.mem.read_u32(0), 42)
        for lane in range(64):
            self.assertEqual(simd.vgprs[lane][0], self.CANARY_OLD)


class TestBufferAtomicSmaxRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_vdata_aliases_vaddr_with_glc(self) -> None:
        # v0 acts as both vdata and vaddr
        # v0 = 16 (so address is 16, and compare value is 16)
        mem = BufferMemory(256)
        mem.write_u32(16, u32(-5))
        rf = ScalarRegisterFile()
        rf.write_vgpr(0, 16)

        interp = ScalarBufferInterpreter(mem, rf)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        interp.set_buffer_descriptor(0, desc)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 0),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True, glc=True),
        )
        ret, ok = interp.step(inst)
        self.assertTrue(ok)
        # Memory at 16 updated to max(-5, 16) = 16
        self.assertEqual(mem.read_u32(16), 16)
        # Pre-op value (-5) written back to v0
        self.assertEqual(s32(ret), -5)
        self.assertEqual(s32(rf.read_vgpr(0)), -5)

    def test_adjacent_registers_untouched(self) -> None:
        mem = BufferMemory(256)
        mem.write_u32(0, 10)
        rf = ScalarRegisterFile()
        rf.write_vgpr(0, 50)          # vdata
        rf.write_vgpr(1, 0)           # vaddr
        rf.write_vgpr(2, 0x12345678)  # adjacent canary
        rf.write_vgpr(3, 0x87654321)  # adjacent canary

        interp = ScalarBufferInterpreter(mem, rf)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        interp.set_buffer_descriptor(0, desc)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True, glc=True),
        )
        interp.step(inst)
        # Check adjacent registers are preserved exactly
        self.assertEqual(rf.read_vgpr(2), 0x12345678)
        self.assertEqual(rf.read_vgpr(3), 0x87654321)

    def test_chained_inplace_atomic_smax(self) -> None:
        mem = BufferMemory(256)
        mem.write_u32(0, u32(-100))
        rf = ScalarRegisterFile()
        rf.write_vgpr(0, u32(-50))
        rf.write_vgpr(1, 0)

        interp = ScalarBufferInterpreter(mem, rf)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        interp.set_buffer_descriptor(0, desc)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True, glc=True),
        )
        # Step 1: memory becomes -50, v0 receives -100
        interp.step(inst)
        self.assertEqual(s32(mem.read_u32(0)), -50)
        self.assertEqual(s32(rf.read_vgpr(0)), -100)

        # Step 2: v0 is -100, compare with memory (-50) -> memory remains -50, v0 receives -50
        interp.step(inst)
        self.assertEqual(s32(mem.read_u32(0)), -50)
        self.assertEqual(s32(rf.read_vgpr(0)), -50)

    def test_simd_multi_lane_aliased_operands(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=1024)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=1024)
        simd.set_buffer_descriptor(0, desc)

        # Each lane uses v0 as both data and addr
        for lane in range(32):
            simd.mem.write_u32(lane * 4, u32(-10))
            simd.vgprs[lane][0] = lane * 4  # v0 is both address and value

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 0),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True, glc=True),
        )
        simd.step(inst)

        for lane in range(32):
            # Memory should be max(-10, lane * 4) = lane * 4
            self.assertEqual(simd.mem.read_u32(lane * 4), lane * 4)
            # v0 receives pre-op value -10
            self.assertEqual(s32(simd.vgprs[lane][0]), -10)


class BufferAtomicStatusFlags:
    """Status flags and architectural condition codes for a buffer atomic operation."""

    def __init__(self, old_val: int, new_val: int, in_bounds: bool, addr: int) -> None:
        self.old_val = old_val & MASK32
        self.new_val = new_val & MASK32
        self.in_bounds = in_bounds
        self.is_oob = not in_bounds
        self.unaligned = (addr % 4 != 0)
        self.is_zero = (self.new_val == 0)
        self.is_negative = bool(self.new_val & SIGN_BIT32)
        self.was_updated = (self.old_val != self.new_val) and in_bounds

    def __repr__(self) -> str:
        flags = [k for k, v in self.__dict__.items() if v is True]
        return f"BufferAtomicStatusFlags({', '.join(flags)})"


def evaluate_buffer_atomic_smax_with_flags(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    cmp_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, BufferAtomicStatusFlags]:
    """Execute buffer_atomic_smax and return result along with status flags."""
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    phys_addr = (desc.base_address + effective_offset) & 0xFFFFFFFFFFFFFFFF
    old_mem = mem.read_u32(phys_addr)

    ret, ok = ref_buffer_atomic_smax(
        mem, desc, soffset, vaddr, cmp_val, inst_offset, glc, offen
    )
    new_mem = mem.read_u32(phys_addr)

    flags = BufferAtomicStatusFlags(
        old_val=old_mem,
        new_val=new_mem,
        in_bounds=ok,
        addr=phys_addr,
    )
    return ret, flags


def generate_buffer_vcc_mask(
    simd: SIMDBufferWavefrontInterpreter,
    inst: Instruction,
    predicate: str,
) -> int:
    """Generate 64-bit VCC mask evaluating condition across all active lanes."""
    d0 = simd.sgprs[inst.srsrc.index]
    d1 = simd.sgprs[inst.srsrc.index + 1]
    d2 = simd.sgprs[inst.srsrc.index + 2]
    d3 = simd.sgprs[inst.srsrc.index + 3]
    desc = BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)

    soffset = (
        simd.sgprs[inst.soffset.index]
        if isinstance(inst.soffset, Register)
        else inst.soffset
    )

    vcc = 0
    for lane in range(simd.wave_size):
        if (simd.exec_mask >> lane) & 1:
            vaddr = simd.vgprs[lane][inst.vaddr.index]
            cmp_val = simd.vgprs[lane][inst.vdata.index]

            effective_offset = (soffset & MASK32) + (inst.mods.offset & 0xFFF)
            if inst.mods.offen:
                effective_offset += (vaddr & MASK32)

            in_bounds = (effective_offset + 4 <= desc.num_records)
            phys_addr = (desc.base_address + effective_offset) & 0xFFFFFFFFFFFFFFFF
            cur_mem = simd.mem.read_u32(phys_addr)

            cond = False
            if predicate == "in_bounds":
                cond = in_bounds
            elif predicate == "was_updated":
                cond = in_bounds and (s32(cmp_val) > s32(cur_mem))
            elif predicate == "result_ge_zero":
                res = max(s32(cur_mem), s32(cmp_val)) if in_bounds else s32(cur_mem)
                cond = in_bounds and (res >= 0)
            elif predicate == "result_lt_zero":
                res = max(s32(cur_mem), s32(cmp_val)) if in_bounds else s32(cur_mem)
                cond = in_bounds and (res < 0)
            else:
                raise ValueError(f"Unknown predicate: {predicate}")

            if cond:
                vcc |= (1 << lane)

    return vcc


class TestBufferAtomicSmaxStatusFlagsAndConditionCodes(unittest.TestCase):
    """Unit tests for status flags, condition codes, and VCC predicate generation."""

    def setUp(self) -> None:
        self.mem = BufferMemory(1024)
        self.desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=512)

    def test_status_flags_updated_vs_not_updated(self) -> None:
        # Initial memory has 10
        self.mem.write_u32(0, 10)

        # 1. Compare with 20 -> updated = True
        _, flags1 = evaluate_buffer_atomic_smax_with_flags(
            self.mem, self.desc, soffset=0, vaddr=0, cmp_val=20
        )
        self.assertTrue(flags1.in_bounds)
        self.assertTrue(flags1.was_updated)
        self.assertFalse(flags1.is_negative)
        self.assertFalse(flags1.is_zero)

        # 2. Compare with 15 (memory is already 20) -> updated = False
        _, flags2 = evaluate_buffer_atomic_smax_with_flags(
            self.mem, self.desc, soffset=0, vaddr=0, cmp_val=15
        )
        self.assertTrue(flags2.in_bounds)
        self.assertFalse(flags2.was_updated)

    def test_status_flags_zero_and_negative(self) -> None:
        self.mem.write_u32(4, u32(-50))

        # Memory is -50, cmp with 0 -> new value is 0 (is_zero=True, is_negative=False)
        _, flags = evaluate_buffer_atomic_smax_with_flags(
            self.mem, self.desc, soffset=0, vaddr=4, cmp_val=0
        )
        self.assertTrue(flags.is_zero)
        self.assertFalse(flags.is_negative)
        self.assertTrue(flags.was_updated)

        # Memory at 8 is -100, cmp with -60 -> new value is -60 (is_negative=True)
        self.mem.write_u32(8, u32(-100))
        _, flags_neg = evaluate_buffer_atomic_smax_with_flags(
            self.mem, self.desc, soffset=0, vaddr=8, cmp_val=u32(-60)
        )
        self.assertTrue(flags_neg.is_negative)
        self.assertFalse(flags_neg.is_zero)

    def test_status_flags_oob_and_alignment(self) -> None:
        # OOB access at offset 512
        _, flags_oob = evaluate_buffer_atomic_smax_with_flags(
            self.mem, self.desc, soffset=0, vaddr=512, cmp_val=10
        )
        self.assertTrue(flags_oob.is_oob)
        self.assertFalse(flags_oob.was_updated)

        # Unaligned address
        _, flags_unaligned = evaluate_buffer_atomic_smax_with_flags(
            self.mem, self.desc, soffset=0, vaddr=3, cmp_val=10
        )
        self.assertTrue(flags_unaligned.unaligned)

    def test_vcc_mask_generation_across_wave(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=1024)
        simd.set_buffer_descriptor(0, self.desc)
        # Upper 32 lanes inactive
        simd.set_exec_lo(0xFFFFFFFF)
        simd.set_exec_hi(0x00000000)

        # Even lanes in 0..31 have positive numbers, odd lanes have negative numbers
        for lane in range(64):
            simd.mem.write_u32(lane * 4, u32(-50))
            simd.vgprs[lane][0] = 10 if (lane % 2 == 0) else u32(-80)
            simd.vgprs[lane][1] = lane * 4

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
        )

        # Even lanes in 0..31 will have positive result (0x55555555)
        vcc_ge = generate_buffer_vcc_mask(simd, inst, "result_ge_zero")
        self.assertEqual(vcc_ge, 0x55555555)

        # Odd lanes in 0..31 will have negative result (0xAAAAAAAA)
        vcc_lt = generate_buffer_vcc_mask(simd, inst, "result_lt_zero")
        self.assertEqual(vcc_lt, 0xAAAAAAAA)

        # Inactive lanes (32..63) are 0 in VCC
        self.assertEqual(vcc_ge >> 32, 0)
        self.assertEqual(vcc_lt >> 32, 0)


class TestBufferAtomicSmaxIEEE754BitwiseBoundaries(unittest.TestCase):
    """Unit tests for IEEE 754 bit-pattern behavior under signed integer maximum."""

    def setUp(self) -> None:
        self.mem = BufferMemory(1024)
        self.desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=512)

    def test_signed_zeros_float_bitpatterns(self) -> None:
        # +0.0 float32 is 0x00000000 (signed int 0)
        # -0.0 float32 is 0x80000000 (signed int INT32_MIN = -2147483648)
        # Therefore smax(+0.0, -0.0) yields +0.0 (0x00000000)
        self.mem.write_u32(0, 0x80000000)  # -0.0
        ret, ok = ref_buffer_atomic_smax(
            self.mem, self.desc, soffset=0, vaddr=0, cmp_val=0x00000000, glc=True
        )
        self.assertTrue(ok)
        self.assertEqual(ret, 0x80000000)
        self.assertEqual(self.mem.read_u32(0), 0x00000000)

    def test_positive_vs_negative_float_bitpatterns(self) -> None:
        # +1.0f is 0x3F800000 (positive int 1065353216)
        # -1.0f is 0xBF800000 (negative int -1082130432)
        # smax(+1.0f, -1.0f) = +1.0f
        self.mem.write_u32(4, 0xBF800000)
        ret, ok = ref_buffer_atomic_smax(
            self.mem, self.desc, soffset=0, vaddr=4, cmp_val=0x3F800000, glc=True
        )
        self.assertTrue(ok)
        self.assertEqual(ret, 0xBF800000)
        self.assertEqual(self.mem.read_u32(4), 0x3F800000)

    def test_infinities_float_bitpatterns(self) -> None:
        # +inf is 0x7F800000 (positive int 2139095040)
        # -inf is 0xFF800000 (negative int -8388608)
        # smax(+inf, -inf) yields +inf
        self.mem.write_u32(8, 0xFF800000)
        ret, ok = ref_buffer_atomic_smax(
            self.mem, self.desc, soffset=0, vaddr=8, cmp_val=0x7F800000, glc=True
        )
        self.assertTrue(ok)
        self.assertEqual(ret, 0xFF800000)
        self.assertEqual(self.mem.read_u32(8), 0x7F800000)

    def test_subnormal_float_bitpatterns(self) -> None:
        # Smallest positive subnormal: 0x00000001 (int +1)
        # Smallest-magnitude negative subnormal: 0x80000001 (int INT32_MIN + 1 = -2147483647)
        # smax(pos_sub, neg_sub) yields pos_sub (0x00000001)
        self.mem.write_u32(12, 0x80000001)
        ret, ok = ref_buffer_atomic_smax(
            self.mem, self.desc, soffset=0, vaddr=12, cmp_val=0x00000001, glc=True
        )
        self.assertTrue(ok)
        self.assertEqual(ret, 0x80000001)
        self.assertEqual(self.mem.read_u32(12), 0x00000001)

    def test_extreme_two_complement_limits(self) -> None:
        # INT32_MAX (0x7FFFFFFF = 2147483647) is the supreme upper bound for smax
        self.mem.write_u32(16, 0)
        ref_buffer_atomic_smax(
            self.mem, self.desc, soffset=0, vaddr=16, cmp_val=U32_INT32_MAX, glc=False
        )
        self.assertEqual(self.mem.read_u32(16), U32_INT32_MAX)

        # Comparing INT32_MAX with any value leaves INT32_MAX intact
        ref_buffer_atomic_smax(
            self.mem, self.desc, soffset=0, vaddr=16, cmp_val=1000, glc=False
        )
        self.assertEqual(self.mem.read_u32(16), U32_INT32_MAX)


def parse_buffer_atomic_smax_asm(line: str) -> Instruction:
    """Parse textual assembly instruction into symbolic Instruction AST."""
    line = line.strip()
    if not line:
        raise ValueError("Empty assembly line")

    tokens = line.split(None, 1)
    if not tokens or tokens[0] != Opcode.BUFFER_ATOMIC_SMAX.value:
        raise ValueError(f"Expected opcode '{Opcode.BUFFER_ATOMIC_SMAX.value}', got '{tokens[0] if tokens else ''}'")

    if len(tokens) < 2:
        raise ValueError("Missing operands in assembly line")

    rest = tokens[1].strip()
    parts = [p.strip() for p in rest.split(",")]
    if len(parts) < 4:
        raise ValueError(f"Expected 4 operands (vdata, vaddr, srsrc, soffset), got {len(parts)}")

    # vdata: e.g. v0
    vdata_str = parts[0]
    if not vdata_str.startswith("v"):
        raise ValueError(f"vdata must be a VGPR: {vdata_str}")
    vdata = Register(RegKind.VGPR, int(vdata_str[1:]))

    # vaddr: e.g. v1
    vaddr_str = parts[1]
    if not vaddr_str.startswith("v"):
        raise ValueError(f"vaddr must be a VGPR: {vaddr_str}")
    vaddr = Register(RegKind.VGPR, int(vaddr_str[1:]))

    # srsrc: e.g. s[4:7]
    srsrc_str = parts[2]
    if not (srsrc_str.startswith("s[") and srsrc_str.endswith("]")):
        raise ValueError(f"srsrc must be an SGPR range s[N:N+3]: {srsrc_str}")
    s_inner = srsrc_str[2:-1].split(":")
    if len(s_inner) != 2:
        raise ValueError(f"Malformed SGPR range: {srsrc_str}")
    s_start = int(s_inner[0])
    s_end = int(s_inner[1])
    s_count = s_end - s_start + 1
    if s_count != 4:
        raise ValueError(f"srsrc must be 4 SGPRs, got {s_count}")
    srsrc = Register(RegKind.SGPR, s_start, count=4)

    # soffset and trailing modifiers
    so_tokens = parts[3].split()
    if not so_tokens:
        raise ValueError("Missing soffset")
    so_str = so_tokens[0]
    mod_tokens = so_tokens[1:]

    if so_str.startswith("s"):
        soffset = Register(RegKind.SGPR, int(so_str[1:]))
    else:
        soffset = int(so_str)

    offset = 0
    glc = False
    slc = False
    offen = False
    idxen = False

    for mod in mod_tokens:
        if mod == "glc":
            glc = True
        elif mod == "slc":
            slc = True
        elif mod == "offen":
            offen = True
        elif mod == "idxen":
            idxen = True
        elif mod.startswith("offset:"):
            offset = int(mod.split(":", 1)[1])
        else:
            raise ValueError(f"Unknown modifier: {mod}")

    mods = BufferAtomicModifiers(
        offset=offset,
        glc=glc,
        slc=slc,
        offen=offen,
        idxen=idxen,
    )
    return Instruction(
        opcode=Opcode.BUFFER_ATOMIC_SMAX,
        vdata=vdata,
        vaddr=vaddr,
        srsrc=srsrc,
        soffset=soffset,
        mods=mods,
    )


class TestBufferAtomicSmaxNegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for syntax rejection, negative controls, and mutation detection."""

    def test_syntax_rejection_malformed_instructions(self) -> None:
        with self.assertRaises(ValueError):
            parse_buffer_atomic_smax_asm("")

        with self.assertRaises(ValueError):
            parse_buffer_atomic_smax_asm("buffer_atomic_smin v0, v1, s[0:3], 0")

        with self.assertRaises(ValueError):
            parse_buffer_atomic_smax_asm("buffer_atomic_smax s0, v1, s[0:3], 0")

        with self.assertRaises(ValueError):
            parse_buffer_atomic_smax_asm("buffer_atomic_smax v0, v1, s[0:1], 0")

        with self.assertRaises(ValueError):
            parse_buffer_atomic_smax_asm("buffer_atomic_smax v0, v1, s[0:3], 0 bad_mod")

    def test_mutation_oob_check_omission(self) -> None:
        # Mutant: skips NUM_RECORDS check
        mem = BufferMemory(256)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=32)

        def mutated_smax_no_oob(m: BufferMemory, d: BufferResourceDescriptor, off: int, val: int) -> int:
            old = m.read_u32(off)
            m.write_u32(off, u32(max(s32(old), s32(val))))
            return old

        # Access at offset 64 should be suppressed by the oracle
        _, in_bounds = ref_buffer_atomic_smax(mem, desc, soffset=0, vaddr=64, cmp_val=10, glc=True)
        self.assertFalse(in_bounds)

        # Mutated oracle would incorrectly treat it as valid
        with self.assertRaises(AssertionError):
            self.assertEqual(in_bounds, True)

    def test_mutation_glc_returns_post_op_value(self) -> None:
        # Mutant: returns post-op new_val instead of pre-op old_val
        mem = BufferMemory(256)
        mem.write_u32(0, 10)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)

        def mutated_glc_post_op(m: BufferMemory, d: BufferResourceDescriptor, off: int, val: int) -> int:
            old = m.read_u32(off)
            new_val = u32(max(s32(old), s32(val)))
            m.write_u32(off, new_val)
            return new_val  # BUG: returns new value instead of old

        # For old=10, cmp=20, correct glc return is 10. Mutated returns 20.
        mutated_ret = mutated_glc_post_op(mem, desc, 0, 20)
        correct_ret, _ = ref_buffer_atomic_smax(BufferMemory(256), desc, 0, 0, 20, glc=True)
        self.assertNotEqual(mutated_ret, 10)

    def test_mutation_signed_min_instead_of_max(self) -> None:
        # Mutant: using min() instead of max()
        mem = BufferMemory(256)
        mem.write_u32(0, u32(-50))
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)

        def mutated_smin_instead_of_smax(m: BufferMemory, off: int, val: int) -> None:
            old = m.read_u32(off)
            m.write_u32(off, u32(min(s32(old), s32(val))))  # BUG: min instead of max

        # -50 vs -10: correct max is -10. Mutated produces -50.
        mutated_smin_instead_of_smax(mem, 0, u32(-10))
        self.assertEqual(s32(mem.read_u32(0)), -50)
        self.assertNotEqual(s32(mem.read_u32(0)), -10)

    def test_mutation_unsigned_instead_of_signed_max(self) -> None:
        # Mutant: using unsigned max instead of signed max
        # For -1 (0xFFFFFFFF = 4294967295) and 0:
        # Signed max: max(-1, 0) = 0
        # Unsigned max: max(0xFFFFFFFF, 0) = 0xFFFFFFFF (-1)
        mem = BufferMemory(256)
        mem.write_u32(0, 0)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)

        def mutated_umax(m: BufferMemory, off: int, val: int) -> None:
            old = m.read_u32(off)
            m.write_u32(off, max(old, val))  # BUG: unsigned max

        mutated_umax(mem, 0, U32_MINUS_ONE)
        # Unsigned max produced 0xFFFFFFFF (-1), whereas correct signed max is 0!
        self.assertEqual(mem.read_u32(0), U32_MINUS_ONE)
        self.assertNotEqual(s32(mem.read_u32(0)), 0)

    def test_mutation_simd_exec_leak(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=256)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        simd.set_buffer_descriptor(0, desc)
        simd.set_exec_mask(0)  # all lanes disabled

        canary = 0x12345678
        simd.mem.write_u32(0, canary)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True, glc=True),
        )
        simd.step(inst)
        self.assertEqual(simd.mem.read_u32(0), canary)

        # Mutated leaky write would corrupt memory
        simd.mem.write_u32(0, 999999)
        with self.assertRaises(AssertionError):
            self.assertEqual(simd.mem.read_u32(0), canary)


if __name__ == "__main__":
    unittest.main()
