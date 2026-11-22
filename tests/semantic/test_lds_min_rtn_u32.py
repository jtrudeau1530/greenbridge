from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_min_rtn_u32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 32-bit Unsigned Minimum with Return:
Performs a 32-bit atomic unsigned minimum operation on LDS storage and returns
the previous value in memory to the destination VGPR.

Syntax:
    ds_min_rtn_u32 vdst, vaddr, vdata0 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    LDS[dword_index] = min(old_val, vdata0)   # Unsigned 32-bit comparison
    vdst = old_val

Standard library Python only; zero third-party dependencies; zero game data.
"""

import dataclasses
import enum
import math
import struct
import unittest
from typing import Dict, List, Optional, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
MASK32: int = 0xFFFFFFFF
UINT32_MIN: int = 0x00000000
UINT32_MAX: int = 0xFFFFFFFF
UINT32_MID: int = 0x80000000
UINT32_HIGH_BIT: int = 0x80000000

DEFAULT_LDS_SIZE_BYTES: int = 65536   # 64 KiB default capacity
DEFAULT_LDS_NUM_DWORDS: int = 16384  # 16,384 32-bit dwords
NUM_LDS_BANKS: int = 32              # 32 memory banks
OFFSET16_MASK: int = 0xFFFF          # 16-bit immediate byte offset mask


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = DEFAULT_LDS_NUM_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0 to 31) for an effective address."""
    return (eff_addr >> 2) % NUM_LDS_BANKS


def oracle_min_u32(a: int, b: int) -> int:
    """Pure reference unsigned 32-bit minimum."""
    ua = a & MASK32
    ub = b & MASK32
    return ua if ua < ub else ub


def ref_ds_min_rtn_u32(current_mem: int, candidate: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_min_rtn_u32.

    Computes:
        old_val = current_mem & 0xFFFFFFFF
        new_mem = min(old_val, candidate) in unsigned 32-bit domain

    Returns:
        (old_val, new_mem)
    """
    old_val = u32(current_mem)
    cand = u32(candidate)
    new_mem = oracle_min_u32(old_val, cand)
    return old_val, new_mem


class LDSMemory:
    """Architectural simulation model of AMD RDNA2 Local Data Share (LDS)."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of 2, got {size_bytes}")
        self.size_bytes: int = size_bytes
        self.num_dwords: int = size_bytes // 4
        self.dwords: List[int] = [0] * self.num_dwords

    def get_dword_index(self, byte_addr: int, offset: int = 0) -> int:
        """Compute effective wrapped dword index, dropping bottom 2 unaligned bits."""
        eff_addr = calc_lds_effective_address(byte_addr, offset)
        return calc_lds_dword_index(eff_addr, self.num_dwords)

    def get_bank(self, dword_idx: int) -> int:
        """Compute memory bank index for the given dword (0 to 31)."""
        return dword_idx % NUM_LDS_BANKS

    def read_dword(self, dword_idx: int) -> int:
        """Read a 32-bit dword from LDS."""
        return self.dwords[dword_idx & (self.num_dwords - 1)]

    def write_dword(self, dword_idx: int, val: int) -> None:
        """Write a 32-bit dword into LDS."""
        self.dwords[dword_idx & (self.num_dwords - 1)] = val & MASK32

    def read_by_byte_addr(self, byte_addr: int, offset: int = 0) -> int:
        """Read a 32-bit dword using effective byte address."""
        idx = self.get_dword_index(byte_addr, offset)
        return self.read_dword(idx)

    def write_by_byte_addr(self, byte_addr: int, val: int, offset: int = 0) -> None:
        """Write a 32-bit dword using effective byte address."""
        idx = self.get_dword_index(byte_addr, offset)
        self.write_dword(idx, val)

    def atomic_min_rtn_u32(self, byte_addr: int, val: int, offset: int = 0) -> int:
        """Execute atomic unsigned minimum at (byte_addr + offset), returning previous value."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        new_val = oracle_min_u32(old_val, val)
        self.dwords[idx] = new_val
        return old_val


class TestDSMinRtnU32OracleAndModel(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural LDS model."""

    def test_oracle_basic_min(self) -> None:
        self.assertEqual(oracle_min_u32(10, 20), 10)
        self.assertEqual(oracle_min_u32(20, 10), 10)
        self.assertEqual(oracle_min_u32(50, 50), 50)
        self.assertEqual(oracle_min_u32(0, 0), 0)

    def test_oracle_unsigned_ordering(self) -> None:
        # In unsigned comparison, 0 is strictly smaller than 0xFFFFFFFF
        self.assertEqual(oracle_min_u32(UINT32_MIN, UINT32_MAX), UINT32_MIN)
        self.assertEqual(oracle_min_u32(UINT32_MAX, UINT32_MIN), UINT32_MIN)
        # 0x7FFFFFFF vs 0x80000000
        self.assertEqual(oracle_min_u32(0x7FFFFFFF, 0x80000000), 0x7FFFFFFF)
        # 0x80000000 vs 0xFFFFFFFF
        self.assertEqual(oracle_min_u32(0x80000000, 0xFFFFFFFF), 0x80000000)

    def test_ref_ds_min_rtn_u32(self) -> None:
        old_val, new_mem = ref_ds_min_rtn_u32(100, 40)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_mem, 40)

        # When candidate is larger, memory remains unchanged
        old_val, new_mem = ref_ds_min_rtn_u32(40, 999)
        self.assertEqual(old_val, 40)
        self.assertEqual(new_mem, 40)

    def test_lds_memory_init_and_bounds(self) -> None:
        lds = LDSMemory(65536)
        self.assertEqual(lds.size_bytes, 65536)
        self.assertEqual(lds.num_dwords, 16384)
        self.assertEqual(len(lds.dwords), 16384)
        for i in range(10):
            self.assertEqual(lds.read_dword(i), 0)

        # Invalid sizes
        with self.assertRaises(ValueError):
            LDSMemory(0)
        with self.assertRaises(ValueError):
            LDSMemory(-1024)
        with self.assertRaises(ValueError):
            LDSMemory(50000)  # Not a power of 2

    def test_lds_addressing_and_alignment(self) -> None:
        lds = LDSMemory(65536)
        # Bytes 0, 1, 2, 3 all access dword 0
        self.assertEqual(lds.get_dword_index(0), 0)
        self.assertEqual(lds.get_dword_index(1), 0)
        self.assertEqual(lds.get_dword_index(2), 0)
        self.assertEqual(lds.get_dword_index(3), 0)
        # Byte 4 accesses dword 1
        self.assertEqual(lds.get_dword_index(4), 1)
        # Offset 4 on byte 0 accesses dword 1
        self.assertEqual(lds.get_dword_index(0, offset=4), 1)
        # Offset 8 on byte 4 accesses dword 3
        self.assertEqual(lds.get_dword_index(4, offset=8), 3)

        # Capacity wrapping
        self.assertEqual(lds.get_dword_index(65536), 0)
        self.assertEqual(lds.get_dword_index(65536 + 12), 3)

        # 32-bit address overflow wrapping
        self.assertEqual(lds.get_dword_index(0xFFFFFFFF, offset=1), 0)

    def test_lds_bank_distribution(self) -> None:
        lds = LDSMemory(65536)
        for dword in range(64):
            expected_bank = dword % 32
            self.assertEqual(lds.get_bank(dword), expected_bank)
            self.assertEqual(calc_lds_bank(dword * 4), expected_bank)

    def test_lds_atomic_min_rtn_u32_execution(self) -> None:
        lds = LDSMemory(65536)
        # Initial write
        lds.write_by_byte_addr(0x100, 0x1000)
        self.assertEqual(lds.read_by_byte_addr(0x100), 0x1000)

        # Atomic min with smaller candidate: returns 0x1000, memory becomes 0x0500
        old = lds.atomic_min_rtn_u32(0x100, 0x0500)
        self.assertEqual(old, 0x1000)
        self.assertEqual(lds.read_by_byte_addr(0x100), 0x0500)

        # Atomic min with larger candidate: returns 0x0500, memory remains 0x0500
        old2 = lds.atomic_min_rtn_u32(0x100, 0x0800)
        self.assertEqual(old2, 0x0500)
        self.assertEqual(lds.read_by_byte_addr(0x100), 0x0500)

        # Atomic min with offset
        old3 = lds.atomic_min_rtn_u32(0x0FC, 0x0200, offset=4)
        self.assertEqual(old3, 0x0500)
        self.assertEqual(lds.read_by_byte_addr(0x100), 0x0200)


class DSOpcode(str, enum.Enum):
    """Opcode enumerations for LDS atomic minimum operations."""
    DS_MIN_U32 = "ds_min_u32"
    DS_MIN_RTN_U32 = "ds_min_rtn_u32"

    @property
    def is_return(self) -> bool:
        """True if instruction returns prior LDS value to destination VGPR."""
        return self == DSOpcode.DS_MIN_RTN_U32

    @property
    def is_min(self) -> bool:
        """True if operation performs unsigned 32-bit minimum."""
        return True


class RegKind(enum.Enum):
    """Register kinds in RDNA2 architecture."""
    VGPR = "vgpr"
    SGPR = "sgpr"


@dataclasses.dataclass(frozen=True)
class Register:
    """Architectural register representation with kind and hardware index."""
    kind: RegKind
    index: int

    def __post_init__(self) -> None:
        if self.kind == RegKind.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")
        elif self.kind == RegKind.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")

    def __str__(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        return f"{prefix}{self.index}"


def vgpr(index: int) -> Register:
    """Convenience helper to create a VGPR."""
    return Register(RegKind.VGPR, index)


def sgpr(index: int) -> Register:
    """Convenience helper to create an SGPR."""
    return Register(RegKind.SGPR, index)


@dataclasses.dataclass(frozen=True)
class OffsetImmediate:
    """Unsigned 16-bit byte offset immediate [0, 65535]."""
    value: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.value <= 65535):
            raise ValueError(f"Offset immediate out of range [0, 65535]: {self.value}")

    def __str__(self) -> str:
        return f"offset:{self.value}" if self.value != 0 else ""


@dataclasses.dataclass(frozen=True)
class DSMinRtnU32Instruction:
    """Symbolic AST node representing an RDNA2 ds_min_rtn_u32 instruction."""
    opcode: DSOpcode
    vdst: Optional[Register]
    vaddr: Register
    vdata0: Register
    offset: OffsetImmediate = OffsetImmediate(0)
    gds: bool = False

    def __post_init__(self) -> None:
        if self.opcode.is_return and self.vdst is None:
            raise ValueError(f"Returning opcode {self.opcode.value} requires destination vdst")
        if self.vdst is not None and self.vdst.kind != RegKind.VGPR:
            raise ValueError(f"Destination register must be VGPR, got {self.vdst.kind}")
        if self.vaddr.kind != RegKind.VGPR:
            raise ValueError(f"Address register must be VGPR, got {self.vaddr.kind}")
        if self.vdata0.kind != RegKind.VGPR:
            raise ValueError(f"Data operand register must be VGPR, got {self.vdata0.kind}")

    def __str__(self) -> str:
        parts = [self.opcode.value]
        if self.vdst is not None:
            parts.append(str(self.vdst) + ",")
        parts.append(str(self.vaddr) + ",")
        parts.append(str(self.vdata0))
        if self.offset.value != 0:
            parts.append(str(self.offset))
        if self.gds:
            parts.append("gds")
        return " ".join(parts)


@dataclasses.dataclass
class Program:
    """Sequence of symbolic instructions."""
    instructions: List[DSMinRtnU32Instruction] = dataclasses.field(default_factory=list)

    def add(self, inst: DSMinRtnU32Instruction) -> None:
        self.instructions.append(inst)

    def __len__(self) -> int:
        return len(self.instructions)

    def __iter__(self):
        return iter(self.instructions)


class TestDSMinRtnU32AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instruction_creation(self) -> None:
        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
            offset=OffsetImmediate(8),
            gds=False,
        )
        self.assertEqual(inst.opcode, DSOpcode.DS_MIN_RTN_U32)
        self.assertEqual(inst.vdst, vgpr(0))
        self.assertEqual(inst.vaddr, vgpr(1))
        self.assertEqual(inst.vdata0, vgpr(2))
        self.assertEqual(inst.offset.value, 8)
        self.assertFalse(inst.gds)

    def test_ast_string_representation(self) -> None:
        inst_offset = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(5),
            vaddr=vgpr(6),
            vdata0=vgpr(7),
            offset=OffsetImmediate(16),
            gds=False,
        )
        self.assertEqual(str(inst_offset), "ds_min_rtn_u32 v5, v6, v7 offset:16")

        inst_no_offset = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
            offset=OffsetImmediate(0),
            gds=False,
        )
        self.assertEqual(str(inst_no_offset), "ds_min_rtn_u32 v0, v1, v2")

        inst_gds = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
            offset=OffsetImmediate(4),
            gds=True,
        )
        self.assertEqual(str(inst_gds), "ds_min_rtn_u32 v0, v1, v2 offset:4 gds")

    def test_ast_immutability(self) -> None:
        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            inst.opcode = DSOpcode.DS_MIN_U32  # type: ignore

    def test_ast_validation_errors(self) -> None:
        # Invalid VGPR indices
        with self.assertRaises(ValueError):
            vgpr(-1)
        with self.assertRaises(ValueError):
            vgpr(256)

        # Invalid Offset
        with self.assertRaises(ValueError):
            OffsetImmediate(-1)
        with self.assertRaises(ValueError):
            OffsetImmediate(65536)

        # Returning opcode without vdst
        with self.assertRaises(ValueError):
            DSMinRtnU32Instruction(
                opcode=DSOpcode.DS_MIN_RTN_U32,
                vdst=None,
                vaddr=vgpr(0),
                vdata0=vgpr(1),
            )

        # Non-VGPR registers in operands
        with self.assertRaises(ValueError):
            DSMinRtnU32Instruction(
                opcode=DSOpcode.DS_MIN_RTN_U32,
                vdst=sgpr(0),
                vaddr=vgpr(1),
                vdata0=vgpr(2),
            )
        with self.assertRaises(ValueError):
            DSMinRtnU32Instruction(
                opcode=DSOpcode.DS_MIN_RTN_U32,
                vdst=vgpr(0),
                vaddr=sgpr(1),
                vdata0=vgpr(2),
            )
        with self.assertRaises(ValueError):
            DSMinRtnU32Instruction(
                opcode=DSOpcode.DS_MIN_RTN_U32,
                vdst=vgpr(0),
                vaddr=vgpr(1),
                vdata0=sgpr(2),
            )

    def test_ast_program_container(self) -> None:
        prog = Program()
        inst1 = DSMinRtnU32Instruction(opcode=DSOpcode.DS_MIN_RTN_U32, vdst=vgpr(0), vaddr=vgpr(1), vdata0=vgpr(2))
        inst2 = DSMinRtnU32Instruction(opcode=DSOpcode.DS_MIN_RTN_U32, vdst=vgpr(3), vaddr=vgpr(4), vdata0=vgpr(5))
        prog.add(inst1)
        prog.add(inst2)
        self.assertEqual(len(prog), 2)
        instructions = list(prog)
        self.assertEqual(instructions[0], inst1)
        self.assertEqual(instructions[1], inst2)


class ScalarThreadState:
    """Architectural register and execution state for a single scalar GPU thread."""

    def __init__(self, lds: Optional[LDSMemory] = None) -> None:
        self.vgprs: List[int] = [0] * 256
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()
        self.pc: int = 0
        self.instruction_count: int = 0
        self.cycle_count: int = 0

    def get_vgpr(self, idx: int) -> int:
        """Read 32-bit dword from VGPR file."""
        return self.vgprs[idx] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        """Write 32-bit dword to VGPR file."""
        self.vgprs[idx] = val & MASK32


class ScalarDSInterpreter:
    """Deterministic scalar instruction interpreter for LDS instructions."""

    def __init__(self, state: Optional[ScalarThreadState] = None, lds: Optional[LDSMemory] = None) -> None:
        if state is not None:
            self.state = state
        else:
            self.state = ScalarThreadState(lds=lds)

    def execute_instruction(self, inst: DSMinRtnU32Instruction) -> int:
        """Execute a single ds_min_rtn_u32 instruction deterministically."""
        if inst.opcode not in (DSOpcode.DS_MIN_RTN_U32, DSOpcode.DS_MIN_U32):
            raise NotImplementedError(f"Unsupported opcode: {inst.opcode}")

        # In RDNA2, register reads occur before any register write
        vaddr_val = self.state.get_vgpr(inst.vaddr.index)
        vdata0_val = self.state.get_vgpr(inst.vdata0.index)

        eff_addr = calc_lds_effective_address(vaddr_val, inst.offset.value)
        old_val = self.state.lds.atomic_min_rtn_u32(eff_addr, vdata0_val)

        if inst.opcode.is_return and inst.vdst is not None:
            self.state.set_vgpr(inst.vdst.index, old_val)

        self.state.instruction_count += 1
        self.state.cycle_count += 4
        return old_val

    def run_program(self, prog: Program) -> int:
        """Execute all instructions in the given program sequentially."""
        count = 0
        for inst in prog:
            self.execute_instruction(inst)
            count += 1
        return count


class TestDSMinRtnU32ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter(state=self.state)

    def test_scalar_single_execution(self) -> None:
        # v1 (addr) = 0x40, v2 (data) = 0x200, LDS[0x40] = 0x500
        self.state.set_vgpr(1, 0x40)
        self.state.set_vgpr(2, 0x200)
        self.lds.write_by_byte_addr(0x40, 0x500)

        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
        )
        returned = self.interp.execute_instruction(inst)

        # Destination VGPR v0 and returned value should be old LDS value
        self.assertEqual(returned, 0x500)
        self.assertEqual(self.state.get_vgpr(0), 0x500)
        # LDS memory should now be the minimum
        self.assertEqual(self.lds.read_by_byte_addr(0x40), 0x200)
        self.assertEqual(self.state.instruction_count, 1)

    def test_scalar_sequential_reduction(self) -> None:
        # Address at v1 = 0x80; initial memory = 1000
        self.state.set_vgpr(1, 0x80)
        self.lds.write_by_byte_addr(0x80, 1000)

        steps = [
            (800, 1000, 800),   # (candidate, expected_ret, expected_mem)
            (950, 800, 800),    # candidate is larger, mem unchanged
            (300, 800, 300),    # candidate is smaller, mem updated
            (300, 300, 300),    # candidate equal, mem unchanged
            (50, 300, 50),      # candidate is smaller, mem updated
        ]

        for cand, exp_ret, exp_mem in steps:
            self.state.set_vgpr(2, cand)
            inst = DSMinRtnU32Instruction(
                opcode=DSOpcode.DS_MIN_RTN_U32,
                vdst=vgpr(0),
                vaddr=vgpr(1),
                vdata0=vgpr(2),
            )
            ret = self.interp.execute_instruction(inst)
            self.assertEqual(ret, exp_ret)
            self.assertEqual(self.state.get_vgpr(0), exp_ret)
            self.assertEqual(self.lds.read_by_byte_addr(0x80), exp_mem)

    def test_scalar_with_offset(self) -> None:
        # v1 (addr) = 0x20, offset = 12 -> byte 0x2C
        self.state.set_vgpr(1, 0x20)
        self.state.set_vgpr(2, 0x1111)
        self.lds.write_by_byte_addr(0x2C, 0x9999)

        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
            offset=OffsetImmediate(12),
        )
        ret = self.interp.execute_instruction(inst)
        self.assertEqual(ret, 0x9999)
        self.assertEqual(self.state.get_vgpr(0), 0x9999)
        self.assertEqual(self.lds.read_by_byte_addr(0x2C), 0x1111)

    def test_scalar_program_execution(self) -> None:
        prog = Program()
        # Initialize 3 memory locations
        for i in range(3):
            addr = i * 4
            self.lds.write_by_byte_addr(addr, 100 * (i + 1))
            self.state.set_vgpr(10 + i, addr)
            self.state.set_vgpr(20 + i, 50 * (i + 1))
            prog.add(DSMinRtnU32Instruction(
                opcode=DSOpcode.DS_MIN_RTN_U32,
                vdst=vgpr(i),
                vaddr=vgpr(10 + i),
                vdata0=vgpr(20 + i),
            ))

        executed = self.interp.run_program(prog)
        self.assertEqual(executed, 3)
        self.assertEqual(self.state.get_vgpr(0), 100)
        self.assertEqual(self.lds.read_by_byte_addr(0), 50)
        self.assertEqual(self.state.get_vgpr(1), 200)
        self.assertEqual(self.lds.read_by_byte_addr(4), 100)
        self.assertEqual(self.state.get_vgpr(2), 300)
        self.assertEqual(self.lds.read_by_byte_addr(8), 150)


class SIMDWaveState:
    """Architectural register and execution state for a SIMD wavefront (Wave32 or Wave64)."""

    def __init__(self, wave_size: int = 32, lds: Optional[LDSMemory] = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: List[List[int]] = [[0] * 256 for _ in range(wave_size)]
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()
        self.cycle_count: int = 0
        self.instruction_count: int = 0

    def is_lane_active(self, lane: int) -> bool:
        """Check if lane is active under current EXEC mask."""
        return bool(self.exec_mask & (1 << lane))

    def get_vgpr(self, lane: int, reg_idx: int) -> int:
        """Read 32-bit register for a specific SIMD lane."""
        return self.vgprs[lane][reg_idx] & MASK32

    def set_vgpr(self, lane: int, reg_idx: int, val: int) -> None:
        """Write 32-bit register for a specific SIMD lane."""
        self.vgprs[lane][reg_idx] = val & MASK32


class SIMDDSInterpreter:
    """SIMD wavefront instruction interpreter supporting divergence and EXEC masking."""

    def __init__(
        self,
        state: Optional[SIMDWaveState] = None,
        lds: Optional[LDSMemory] = None,
        wave_size: int = 32,
    ) -> None:
        if state is not None:
            self.state = state
        else:
            self.state = SIMDWaveState(wave_size=wave_size, lds=lds)

    def execute_instruction(self, inst: DSMinRtnU32Instruction) -> Dict[int, int]:
        """Execute a SIMD LDS instruction across active lanes in wavefront."""
        if inst.opcode not in (DSOpcode.DS_MIN_RTN_U32, DSOpcode.DS_MIN_U32):
            raise NotImplementedError(f"Unsupported opcode: {inst.opcode}")

        # Phase 1: Read all source operands across active lanes
        lane_ops: List[Tuple[int, int, int]] = []
        for lane in range(self.state.wave_size):
            if self.state.is_lane_active(lane):
                vaddr = self.state.get_vgpr(lane, inst.vaddr.index)
                vdata0 = self.state.get_vgpr(lane, inst.vdata0.index)
                eff_addr = calc_lds_effective_address(vaddr, inst.offset.value)
                lane_ops.append((lane, eff_addr, vdata0))

        # Phase 2: Atomic memory execution with lane arbitration
        results: Dict[int, int] = {}
        for lane, eff_addr, vdata0 in lane_ops:
            old_val = self.state.lds.atomic_min_rtn_u32(eff_addr, vdata0)
            results[lane] = old_val

        # Phase 3: Writeback results to active lane destination VGPRs
        if inst.opcode.is_return and inst.vdst is not None:
            for lane, old_val in results.items():
                self.state.set_vgpr(lane, inst.vdst.index, old_val)

        self.state.instruction_count += 1
        self.state.cycle_count += 4
        return results


class TestDSMinRtnU32SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)

    def test_simd_wave32_independent_lanes(self) -> None:
        state = SIMDWaveState(wave_size=32, lds=self.lds)
        interp = SIMDDSInterpreter(state=state)

        # Setup each lane to access its own dword
        for lane in range(32):
            addr = lane * 4
            self.lds.write_by_byte_addr(addr, 200 + lane)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, 100 + lane)

        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
        )
        res = interp.execute_instruction(inst)

        self.assertEqual(len(res), 32)
        for lane in range(32):
            addr = lane * 4
            # Return value should be initial memory
            self.assertEqual(res[lane], 200 + lane)
            self.assertEqual(state.get_vgpr(lane, 0), 200 + lane)
            # Memory should be minimum (100 + lane)
            self.assertEqual(self.lds.read_by_byte_addr(addr), 100 + lane)

    def test_simd_exec_masking_partial(self) -> None:
        state = SIMDWaveState(wave_size=32, lds=self.lds)
        # Only lanes 0..3 active (0b1111 = 0xF)
        state.exec_mask = 0x0000000F
        interp = SIMDDSInterpreter(state=state)

        for lane in range(32):
            addr = lane * 4
            self.lds.write_by_byte_addr(addr, 500)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, 100)
            state.set_vgpr(lane, 0, 0xCAFE)  # Canary in destination register

        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
        )
        res = interp.execute_instruction(inst)

        self.assertEqual(len(res), 4)
        for lane in range(32):
            addr = lane * 4
            if lane < 4:
                self.assertEqual(res[lane], 500)
                self.assertEqual(state.get_vgpr(lane, 0), 500)
                self.assertEqual(self.lds.read_by_byte_addr(addr), 100)
            else:
                self.assertNotIn(lane, res)
                # Inactive lane VGPR must remain completely untouched
                self.assertEqual(state.get_vgpr(lane, 0), 0xCAFE)
                # Inactive lane LDS memory must remain completely untouched
                self.assertEqual(self.lds.read_by_byte_addr(addr), 500)

    def test_simd_exec_masking_zero(self) -> None:
        state = SIMDWaveState(wave_size=32, lds=self.lds)
        state.exec_mask = 0x00000000
        interp = SIMDDSInterpreter(state=state)

        for lane in range(32):
            state.set_vgpr(lane, 0, 0xBEEF)

        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
        )
        res = interp.execute_instruction(inst)
        self.assertEqual(len(res), 0)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), 0xBEEF)

    def test_simd_wave64_execution(self) -> None:
        state = SIMDWaveState(wave_size=64, lds=self.lds)
        # Upper 32 lanes active: lanes 32..63
        state.exec_mask = 0xFFFFFFFF00000000
        interp = SIMDDSInterpreter(state=state)

        for lane in range(64):
            addr = lane * 4
            self.lds.write_by_byte_addr(addr, 1000)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, 250)
            state.set_vgpr(lane, 0, 0x1234)

        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
        )
        res = interp.execute_instruction(inst)
        self.assertEqual(len(res), 32)
        for lane in range(64):
            addr = lane * 4
            if lane >= 32:
                self.assertEqual(res[lane], 1000)
                self.assertEqual(state.get_vgpr(lane, 0), 1000)
                self.assertEqual(self.lds.read_by_byte_addr(addr), 250)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), 0x1234)
                self.assertEqual(self.lds.read_by_byte_addr(addr), 1000)

    def test_simd_atomic_collision_same_address(self) -> None:
        state = SIMDWaveState(wave_size=32, lds=self.lds)
        state.exec_mask = 0x000000FF  # 8 lanes active: 0..7
        interp = SIMDDSInterpreter(state=state)

        # All 8 lanes access the exact same address 0x100
        self.lds.write_by_byte_addr(0x100, 1000)
        candidates = [900, 950, 700, 800, 600, 650, 500, 550]
        expected_returns = [1000, 900, 900, 700, 700, 600, 600, 500]

        for lane in range(8):
            state.set_vgpr(lane, 1, 0x100)
            state.set_vgpr(lane, 2, candidates[lane])

        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
        )
        res = interp.execute_instruction(inst)

        for lane in range(8):
            self.assertEqual(res[lane], expected_returns[lane])
            self.assertEqual(state.get_vgpr(lane, 0), expected_returns[lane])

        # Final memory must be the global minimum across all operations
        self.assertEqual(self.lds.read_by_byte_addr(0x100), 500)


class TestDSMinRtnU32RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter(state=self.state)

    def test_alias_vdst_equals_vaddr(self) -> None:
        # v0 serves as both vaddr and vdst
        self.state.set_vgpr(0, 0x40)    # Address = 0x40
        self.state.set_vgpr(1, 15)      # Candidate = 15
        self.lds.write_by_byte_addr(0x40, 100)

        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(0),
            vdata0=vgpr(1),
        )
        old_val = self.interp.execute_instruction(inst)

        # Address must be sampled before v0 is overwritten with return value
        self.assertEqual(old_val, 100)
        self.assertEqual(self.state.get_vgpr(0), 100)
        self.assertEqual(self.lds.read_by_byte_addr(0x40), 15)
        # v1 (data) must be preserved
        self.assertEqual(self.state.get_vgpr(1), 15)

    def test_alias_vdst_equals_vdata0(self) -> None:
        # v1 serves as both vdata0 and vdst
        self.state.set_vgpr(0, 0x40)    # Address = 0x40
        self.state.set_vgpr(1, 15)      # Candidate = 15
        self.lds.write_by_byte_addr(0x40, 100)

        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(1),
            vaddr=vgpr(0),
            vdata0=vgpr(1),
        )
        old_val = self.interp.execute_instruction(inst)

        # Candidate must be sampled before v1 is overwritten with return value
        self.assertEqual(old_val, 100)
        self.assertEqual(self.state.get_vgpr(1), 100)
        self.assertEqual(self.lds.read_by_byte_addr(0x40), 15)
        # v0 (addr) must be preserved
        self.assertEqual(self.state.get_vgpr(0), 0x40)

    def test_alias_vaddr_equals_vdata0(self) -> None:
        # v1 serves as both vaddr and vdata0
        addr_and_cand = 0x20
        self.state.set_vgpr(1, addr_and_cand)
        self.lds.write_by_byte_addr(addr_and_cand, 80)

        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(1),
        )
        old_val = self.interp.execute_instruction(inst)

        # min(80, 0x20) = 0x20 (32)
        self.assertEqual(old_val, 80)
        self.assertEqual(self.state.get_vgpr(0), 80)
        self.assertEqual(self.lds.read_by_byte_addr(addr_and_cand), addr_and_cand)
        # v1 must remain untouched
        self.assertEqual(self.state.get_vgpr(1), addr_and_cand)

    def test_alias_all_operands_same(self) -> None:
        # v0 is vdst, vaddr, and vdata0 simultaneously
        val = 0x10
        self.state.set_vgpr(0, val)
        self.lds.write_by_byte_addr(val, 0x50)

        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(0),
            vdata0=vgpr(0),
        )
        old_val = self.interp.execute_instruction(inst)

        # Memory was 0x50, candidate was 0x10 -> new memory 0x10, v0 becomes old_val 0x50
        self.assertEqual(old_val, 0x50)
        self.assertEqual(self.state.get_vgpr(0), 0x50)
        self.assertEqual(self.lds.read_by_byte_addr(val), val)

    def test_operand_preservation_and_canary_isolation(self) -> None:
        # Fill all VGPRs with distinct canary pattern
        for reg in range(256):
            self.state.set_vgpr(reg, 0xAA000000 | reg)

        self.state.set_vgpr(10, 0x80)   # vaddr
        self.state.set_vgpr(20, 0x30)   # vdata0
        self.lds.write_by_byte_addr(0x80, 0x70)

        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(5),
            vaddr=vgpr(10),
            vdata0=vgpr(20),
        )
        self.interp.execute_instruction(inst)

        # Only vdst (v5) should have changed to old LDS value
        self.assertEqual(self.state.get_vgpr(5), 0x70)
        self.assertEqual(self.state.get_vgpr(10), 0x80)
        self.assertEqual(self.state.get_vgpr(20), 0x30)

        # All other 253 registers must have preserved their canary exactly
        for reg in range(256):
            if reg not in (5, 10, 20):
                self.assertEqual(self.state.get_vgpr(reg), 0xAA000000 | reg)

    def test_simd_aliasing(self) -> None:
        state = SIMDWaveState(wave_size=32, lds=self.lds)
        interp = SIMDDSInterpreter(state=state)

        for lane in range(32):
            addr = lane * 4
            self.lds.write_by_byte_addr(addr, 200)
            state.set_vgpr(lane, 0, addr)   # v0 is vaddr and vdst
            state.set_vgpr(lane, 1, 50 + lane)

        inst = DSMinRtnU32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_U32,
            vdst=vgpr(0),
            vaddr=vgpr(0),
            vdata0=vgpr(1),
        )
        res = interp.execute_instruction(inst)

        for lane in range(32):
            addr = lane * 4
            self.assertEqual(res[lane], 200)
            self.assertEqual(state.get_vgpr(lane, 0), 200)
            self.assertEqual(self.lds.read_by_byte_addr(addr), 50 + lane)


if __name__ == "__main__":
    unittest.main()
