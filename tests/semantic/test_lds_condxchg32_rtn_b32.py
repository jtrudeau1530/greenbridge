"""Semantic reference oracle and architectural model for ds_condxchg32_rtn_b32.

RDNA2 / GFX10 Local Data Share (LDS) Conditional Exchange:
Performs a 32-bit atomic compare-and-exchange operation on LDS memory.
Syntax:
    ds_condxchg32_rtn_b32 vdst, vaddr, vdata0, vdata1 [offset:offset16]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    if old_val == vdata0:
        LDS[dword_index] = vdata1
    vdst = old_val
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent banks
MASK_32: int = 0xFFFF_FFFF          # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.
    
    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.
    
    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_condxchg32(
    current_mem: int,
    cmp_val: int,
    new_val: int,
) -> Tuple[int, int, bool]:
    """Pure mathematical reference oracle for ds_condxchg32_rtn_b32.
    
    Returns:
        (returned_old_val, new_mem_val, exchanged)
    """
    old_val = current_mem & MASK_32
    cmp_masked = cmp_val & MASK_32
    new_masked = new_val & MASK_32

    if old_val == cmp_masked:
        return old_val, new_masked, True
    return old_val, old_val, False


class LDSModel:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        if capacity_dwords <= 0 or (capacity_dwords & (capacity_dwords - 1)) != 0:
            raise ValueError(f"capacity_dwords must be a power of two, got {capacity_dwords}")
        self.capacity_dwords = capacity_dwords
        self.memory = [0] * capacity_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        self.memory[idx] = val & MASK_32

    def atomic_condxchg32(
        self,
        vaddr: int,
        cmp_val: int,
        new_val: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        """Execute atomic conditional exchange on LDS dword."""
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        curr = self.memory[idx]
        old_val, new_mem, exchanged = oracle_ds_condxchg32(curr, cmp_val, new_val)
        self.memory[idx] = new_mem
        return old_val, new_mem, exchanged


class TestLdsCondxchg32ReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_successful_exchange_when_equal(self) -> None:
        old, mem, swapped = oracle_ds_condxchg32(0x12345678, 0x12345678, 0x9ABCDEF0)
        self.assertTrue(swapped)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(mem, 0x9ABCDEF0)

    def test_failed_exchange_when_not_equal(self) -> None:
        old, mem, swapped = oracle_ds_condxchg32(0x11112222, 0x33334444, 0x99990000)
        self.assertFalse(swapped)
        self.assertEqual(old, 0x11112222)
        self.assertEqual(mem, 0x11112222)

    def test_effective_address_with_immediate_offset(self) -> None:
        eff = calc_lds_effective_address(0x1000, 0x0024)
        self.assertEqual(eff, 0x1024)
        self.assertEqual(calc_lds_dword_index(eff), 0x1024 >> 2)

    def test_dword_alignment_and_bit_truncation(self) -> None:
        # Addresses differing only in byte offset within a dword map to the same dword
        eff0 = calc_lds_effective_address(0x200)
        eff1 = calc_lds_effective_address(0x201)
        eff2 = calc_lds_effective_address(0x202)
        eff3 = calc_lds_effective_address(0x203)
        self.assertEqual(calc_lds_dword_index(eff0), 0x80)
        self.assertEqual(calc_lds_dword_index(eff1), 0x80)
        self.assertEqual(calc_lds_dword_index(eff2), 0x80)
        self.assertEqual(calc_lds_dword_index(eff3), 0x80)

    def test_modulo_wrapping_at_capacity_boundary(self) -> None:
        capacity_bytes = LDS_SIZE_DWORDS * 4
        eff = calc_lds_effective_address(capacity_bytes + 0x10)
        idx = calc_lds_dword_index(eff)
        self.assertEqual(idx, 0x10 >> 2)

    def test_32bit_address_overflow_wrapping(self) -> None:
        eff = calc_lds_effective_address(0xFFFF_FFFE, 0x0004)
        self.assertEqual(eff, 0x0000_0002)
        self.assertEqual(calc_lds_dword_index(eff), 0)

    def test_bank_index_distribution(self) -> None:
        for bank in range(32):
            addr = bank * 4
            self.assertEqual(calc_lds_bank(addr), bank)

    def test_lds_model_state_update_on_match(self) -> None:
        lds = LDSModel()
        addr = 0x400
        lds.write_dword(addr, 0xCAFEBABE)
        old, mem, swapped = lds.atomic_condxchg32(addr, 0xCAFEBABE, 0xDEADBEEF, offset=0)
        self.assertTrue(swapped)
        self.assertEqual(old, 0xCAFEBABE)
        self.assertEqual(mem, 0xDEADBEEF)
        self.assertEqual(lds.read_dword(addr), 0xDEADBEEF)

    def test_lds_model_state_preservation_on_mismatch(self) -> None:
        lds = LDSModel()
        addr = 0x800
        lds.write_dword(addr, 0x5555AAAA)
        # Attempt swap at addr with mismatched compare value
        old, mem, swapped = lds.atomic_condxchg32(addr, 0x11112222, 0xFFFFFFFF, offset=0)
        self.assertFalse(swapped)
        self.assertEqual(old, 0x5555AAAA)
        self.assertEqual(mem, 0x5555AAAA)
        # Check original addr was preserved
        self.assertEqual(lds.read_dword(addr), 0x5555AAAA)


# ==============================================================================
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class VGPR:
    """Vector General Purpose Register operand (e.g. v[0] .. v[255])."""
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of architectural range [0, 255]")

    @property
    def name(self) -> str:
        return f"v[{self.index}]"


@dataclass(frozen=True)
class SGPR:
    """Scalar General Purpose Register operand (e.g. s[0] .. s[105])."""
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index {self.index} out of architectural range [0, 105]")

    @property
    def name(self) -> str:
        return f"s[{self.index}]"


@dataclass(frozen=True)
class Immediate:
    """32-bit literal or inline integer constant."""
    value: int

    @property
    def name(self) -> str:
        return f"0x{self.value & MASK_32:X}"


Operand = Union[VGPR, SGPR, Immediate]


class Condition(Enum):
    """Comparison predicate for conditional exchange."""
    EQ = auto()
    NE = auto()
    GT_U32 = auto()
    LT_U32 = auto()


@dataclass(frozen=True)
class DSCondxchg32Instruction:
    """Symbolic AST node for ds_condxchg32_rtn_b32 instruction."""
    dst: VGPR
    addr: VGPR
    cmp_val: Operand
    new_val: Operand
    offset: int = 0
    cond: Condition = Condition.EQ

    def __post_init__(self) -> None:
        if not isinstance(self.dst, VGPR):
            raise TypeError(f"Destination must be a VGPR, got {type(self.dst).__name__}")
        if not isinstance(self.addr, VGPR):
            raise TypeError(f"Address operand must be a VGPR, got {type(self.addr).__name__}")
        if not (0 <= self.offset <= 65535):
            raise ValueError(f"Offset {self.offset} out of 16-bit range [0, 65535]")

    def disassemble(self) -> str:
        offset_part = f" offset:{self.offset}" if self.offset != 0 else ""
        return (
            f"ds_condxchg32_rtn_b32 {self.dst.name}, {self.addr.name}, "
            f"{self.cmp_val.name}, {self.new_val.name}{offset_part}"
        )


class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_ast_construction(self) -> None:
        instr = DSCondxchg32Instruction(
            dst=VGPR(0),
            addr=VGPR(1),
            cmp_val=VGPR(2),
            new_val=VGPR(3),
            offset=16,
        )
        self.assertEqual(instr.dst.index, 0)
        self.assertEqual(instr.addr.index, 1)
        self.assertEqual(instr.offset, 16)
        self.assertEqual(instr.cond, Condition.EQ)

    def test_disassembly_with_and_without_offset(self) -> None:
        instr_no_off = DSCondxchg32Instruction(
            dst=VGPR(5),
            addr=VGPR(6),
            cmp_val=Immediate(0x1234),
            new_val=SGPR(10),
        )
        self.assertEqual(
            instr_no_off.disassemble(),
            "ds_condxchg32_rtn_b32 v[5], v[6], 0x1234, s[10]",
        )

        instr_off = DSCondxchg32Instruction(
            dst=VGPR(12),
            addr=VGPR(13),
            cmp_val=VGPR(14),
            new_val=Immediate(0xABCD),
            offset=64,
        )
        self.assertEqual(
            instr_off.disassemble(),
            "ds_condxchg32_rtn_b32 v[12], v[13], v[14], 0xABCD offset:64",
        )

    def test_dst_type_enforcement_rejects_sgpr(self) -> None:
        with self.assertRaises(TypeError):
            # Destination cannot be an SGPR in DS operations
            DSCondxchg32Instruction(
                dst=SGPR(0),  # type: ignore[arg-type]
                addr=VGPR(1),
                cmp_val=Immediate(0),
                new_val=Immediate(0),
            )

    def test_addr_type_enforcement_rejects_sgpr(self) -> None:
        with self.assertRaises(TypeError):
            # Address operand cannot be an SGPR in DS operations
            DSCondxchg32Instruction(
                dst=VGPR(0),
                addr=SGPR(1),  # type: ignore[arg-type]
                cmp_val=Immediate(0),
                new_val=Immediate(0),
            )

    def test_register_index_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            VGPR(256)
        with self.assertRaises(ValueError):
            VGPR(-1)
        with self.assertRaises(ValueError):
            SGPR(106)
        with self.assertRaises(ValueError):
            SGPR(-1)

    def test_offset_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            DSCondxchg32Instruction(
                dst=VGPR(0),
                addr=VGPR(1),
                cmp_val=Immediate(0),
                new_val=Immediate(0),
                offset=65536,
            )
        with self.assertRaises(ValueError):
            DSCondxchg32Instruction(
                dst=VGPR(0),
                addr=VGPR(1),
                cmp_val=Immediate(0),
                new_val=Immediate(0),
                offset=-1,
            )

    def test_operand_imm_and_reg_equality(self) -> None:
        v0_a = VGPR(0)
        v0_b = VGPR(0)
        v1 = VGPR(1)
        self.assertEqual(v0_a, v0_b)
        self.assertNotEqual(v0_a, v1)
        imm1 = Immediate(42)
        imm2 = Immediate(42)
        self.assertEqual(imm1, imm2)


if __name__ == "__main__":
    unittest.main()
