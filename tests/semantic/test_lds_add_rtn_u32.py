from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_add_rtn_u32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 32-bit Add with Return:
Performs a 32-bit atomic addition to LDS memory and returns the original value.

Syntax:
    ds_add_rtn_u32 vdst, vaddr, vdata0 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    LDS[dword_index] = (old_val + vdata0) & 0xFFFFFFFF
    vdst = old_val
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
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


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def ref_ds_add_rtn_u32(
    current_mem: int,
    addend: int,
) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_add_rtn_u32.

    Computes:
        old_val = current_mem & 0xFFFFFFFF
        new_mem = (old_val + addend) & 0xFFFFFFFF

    Returns:
        (old_val, new_mem)
    """
    old_val = u32(current_mem)
    new_mem = u32(old_val + u32(addend))
    return old_val, new_mem


class LDSMemory:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, size_dwords: int = LDS_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"size_dwords must be a positive power of two, got {size_dwords}")
        self.size_dwords = size_dwords
        self.memory = [0] * size_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        self.memory[idx] = u32(val)

    def atomic_add_rtn_u32(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 32-bit integer addition with return on LDS.

        Returns:
            (old_val, new_mem)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        curr = self.memory[idx]
        old_val, new_mem = ref_ds_add_rtn_u32(curr, vdata0)
        self.memory[idx] = new_mem
        return old_val, new_mem


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) atomic opcodes."""
    DS_ADD_RTN_U32 = "ds_add_rtn_u32"


class RegType(Enum):
    """Register file classes."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegisterOperand:
    """Typed symbolic register operand representation."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        max_limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < max_limit):
            raise ValueError(
                f"Register index out of bounds: {self.reg_type.value}[{self.index}] "
                f"(expected 0 <= index < {max_limit})"
            )

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}[{self.index}]"

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Immediate:
    """32-bit immediate constant operand."""
    val: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "val", u32(self.val))

    def __str__(self) -> str:
        return f"0x{self.val:X}"


Operand = Union[RegisterOperand, Immediate]


@dataclass(frozen=True)
class DSAddRtnInstruction:
    """Symbolic AST node for ds_add_rtn_u32 instruction."""
    vdst: RegisterOperand
    vaddr: RegisterOperand
    vdata0: Operand
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_ADD_RTN_U32

    def __post_init__(self) -> None:
        if self.vdst.reg_type != RegType.VGPR:
            raise ValueError(f"vdst must be a VGPR, got {self.vdst.reg_type}")
        if self.vaddr.reg_type != RegType.VGPR:
            raise ValueError(f"vaddr must be a VGPR, got {self.vaddr.reg_type}")
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(
                f"offset out of bounds: {self.offset} (expected 0 <= offset <= {OFFSET16_MASK})"
            )

    def disassemble(self) -> str:
        """Emit canonical AMD assembly string."""
        parts = [self.opcode.value, str(self.vdst), str(self.vaddr), str(self.vdata0)]
        if self.offset != 0:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return f"{parts[0]} {', '.join(parts[1:])}"


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_basic_addition(self) -> None:
        old_val, new_mem = ref_ds_add_rtn_u32(100, 25)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_mem, 125)

    def test_oracle_zero_addend(self) -> None:
        old_val, new_mem = ref_ds_add_rtn_u32(0xDEADBEEF, 0)
        self.assertEqual(old_val, 0xDEADBEEF)
        self.assertEqual(new_mem, 0xDEADBEEF)

    def test_oracle_wrap_around_32bit(self) -> None:
        old_val, new_mem = ref_ds_add_rtn_u32(0xFFFFFFFF, 1)
        self.assertEqual(old_val, 0xFFFFFFFF)
        self.assertEqual(new_mem, 0)

        old_val2, new_mem2 = ref_ds_add_rtn_u32(0xFFFFFFFE, 5)
        self.assertEqual(old_val2, 0xFFFFFFFE)
        self.assertEqual(new_mem2, 3)

    def test_effective_address_calculation(self) -> None:
        eff = calc_lds_effective_address(0x1000, 0x20)
        self.assertEqual(eff, 0x1020)

        # 16-bit offset wrap check
        eff_wrap = calc_lds_effective_address(0xFFFFFFFF, 1)
        self.assertEqual(eff_wrap, 0)

    def test_dword_index_and_alignment(self) -> None:
        self.assertEqual(calc_lds_dword_index(0x1000), 0x400)
        # Low 2 bits ignored: 0x1001, 0x1002, 0x1003 all point to index 0x400
        self.assertEqual(calc_lds_dword_index(0x1001), 0x400)
        self.assertEqual(calc_lds_dword_index(0x1002), 0x400)
        self.assertEqual(calc_lds_dword_index(0x1003), 0x400)
        self.assertEqual(calc_lds_dword_index(0x1004), 0x401)

    def test_bank_calculation(self) -> None:
        # Bank is determined by bits [6:2] (dword index mod 32)
        self.assertEqual(calc_lds_bank(0x00), 0)
        self.assertEqual(calc_lds_bank(0x04), 1)
        self.assertEqual(calc_lds_bank(0x7C), 31)
        self.assertEqual(calc_lds_bank(0x80), 0)

    def test_architectural_memory_atomic_add(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        vaddr = 0x80
        lds.write_dword(vaddr, 500)

        old, new_val = lds.atomic_add_rtn_u32(vaddr=vaddr, vdata0=123, offset=0)
        self.assertEqual(old, 500)
        self.assertEqual(new_val, 623)
        self.assertEqual(lds.read_dword(vaddr), 623)

        # Subsequent atomic add at same address with offset
        old2, new_val2 = lds.atomic_add_rtn_u32(vaddr=0x70, vdata0=100, offset=0x10)
        self.assertEqual(old2, 623)
        self.assertEqual(new_val2, 723)
        self.assertEqual(lds.read_dword(vaddr), 723)


class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_valid_ast_construction(self) -> None:
        vdst = RegisterOperand(0, RegType.VGPR)
        vaddr = RegisterOperand(1, RegType.VGPR)
        vdata0 = RegisterOperand(2, RegType.VGPR)
        instr = DSAddRtnInstruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=16)

        self.assertEqual(instr.opcode, DSOpcode.DS_ADD_RTN_U32)
        self.assertEqual(instr.vdst.index, 0)
        self.assertEqual(instr.vaddr.index, 1)
        self.assertEqual(instr.offset, 16)
        self.assertFalse(instr.gds)

    def test_ast_with_immediate_operand(self) -> None:
        vdst = RegisterOperand(5, RegType.VGPR)
        vaddr = RegisterOperand(6, RegType.VGPR)
        imm = Immediate(0x100)
        instr = DSAddRtnInstruction(vdst=vdst, vaddr=vaddr, vdata0=imm, offset=0)

        self.assertEqual(instr.vdata0, Immediate(0x100))

    def test_disassembly_formatting(self) -> None:
        vdst = RegisterOperand(0, RegType.VGPR)
        vaddr = RegisterOperand(1, RegType.VGPR)
        vdata0 = RegisterOperand(2, RegType.VGPR)

        instr_no_offset = DSAddRtnInstruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=0)
        self.assertEqual(
            instr_no_offset.disassemble(),
            "ds_add_rtn_u32 v[0], v[1], v[2]",
        )

        instr_with_offset = DSAddRtnInstruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=32)
        self.assertEqual(
            instr_with_offset.disassemble(),
            "ds_add_rtn_u32 v[0], v[1], v[2], offset:32",
        )

        instr_gds = DSAddRtnInstruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=4, gds=True)
        self.assertEqual(
            instr_gds.disassemble(),
            "ds_add_rtn_u32 v[0], v[1], v[2], offset:4, gds",
        )

    def test_ast_validation_errors(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        s0 = RegisterOperand(0, RegType.SGPR)

        # vdst cannot be SGPR
        with self.assertRaises(ValueError):
            DSAddRtnInstruction(vdst=s0, vaddr=v0, vdata0=v0)

        # vaddr cannot be SGPR
        with self.assertRaises(ValueError):
            DSAddRtnInstruction(vdst=v0, vaddr=s0, vdata0=v0)

        # Negative offset
        with self.assertRaises(ValueError):
            DSAddRtnInstruction(vdst=v0, vaddr=v0, vdata0=v0, offset=-1)

        # Offset exceeding 16 bits
        with self.assertRaises(ValueError):
            DSAddRtnInstruction(vdst=v0, vaddr=v0, vdata0=v0, offset=65536)


if __name__ == "__main__":
    unittest.main()
