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


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarState:
    """Scalar machine state comprising register files and LDS memory."""

    def __init__(self, lds_size_dwords: int = LDS_SIZE_DWORDS) -> None:
        self.vgprs: list[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = LDSMemory(size_dwords=lds_size_dwords)

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index]

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = u32(val)

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index]

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = u32(val)

    def get_reg(self, reg: RegisterOperand) -> int:
        if reg.reg_type == RegType.VGPR:
            return self.get_vgpr(reg.index)
        return self.get_sgpr(reg.index)

    def set_reg(self, reg: RegisterOperand, val: int) -> None:
        if reg.reg_type == RegType.VGPR:
            self.set_vgpr(reg.index, val)
        else:
            self.set_sgpr(reg.index, val)

    def resolve_operand(self, op: Operand) -> int:
        if isinstance(op, Immediate):
            return op.val
        return self.get_reg(op)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for ds_add_rtn_u32."""

    def execute(self, instr: DSAddRtnInstruction, state: ScalarState) -> Tuple[int, int]:
        """Execute a single ds_add_rtn_u32 instruction on scalar state.

        Returns:
            Tuple[int, int]: (old_val, new_mem)
        """
        # 1. Latch inputs prior to any register writeback
        vaddr = state.get_reg(instr.vaddr)
        addend = state.resolve_operand(instr.vdata0)

        # 2. Perform atomic LDS update
        old_val, new_mem = state.lds.atomic_add_rtn_u32(
            vaddr=vaddr,
            vdata0=addend,
            offset=instr.offset,
        )

        # 3. Write pre-addition value back to destination register
        state.set_reg(instr.vdst, old_val)

        return old_val, new_mem


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class SIMDState:
    """SIMD wavefront execution state supporting Wave32 and Wave64 execution modes."""

    def __init__(
        self,
        num_lanes: int = 64,
        exec_mask: Optional[int] = None,
        lds_size_dwords: int = LDS_SIZE_DWORDS,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {num_lanes}")
        self.num_lanes = num_lanes
        self.exec_mask = ((1 << num_lanes) - 1) if exec_mask is None else (exec_mask & ((1 << num_lanes) - 1))
        self.vgprs: list[list[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(num_lanes)]
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = LDSMemory(size_dwords=lds_size_dwords)

    def is_lane_active(self, lane: int) -> bool:
        return bool(self.exec_mask & (1 << lane))

    def get_lane_vgpr(self, lane: int, index: int) -> int:
        return self.vgprs[lane][index]

    def set_lane_vgpr(self, lane: int, index: int, val: int) -> None:
        self.vgprs[lane][index] = u32(val)

    def resolve_lane_operand(self, op: Operand, lane: int) -> int:
        if isinstance(op, Immediate):
            return op.val
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index]
        return self.get_lane_vgpr(lane, op.index)


class SIMDInterpreter:
    """SIMD multi-lane instruction interpreter for ds_add_rtn_u32."""

    def execute(
        self,
        instr: DSAddRtnInstruction,
        state: SIMDState,
    ) -> List[Optional[int]]:
        """Execute ds_add_rtn_u32 across SIMD lanes with EXEC masking and collision serialization.

        Returns:
            List[Optional[int]]: Per-lane returned old_val for active lanes, or None for inactive lanes.
        """
        results: List[Optional[int]] = [None] * state.num_lanes

        # 1. Latch per-lane inputs prior to any updates
        lane_inputs: list[tuple[int, int, int, int] | None] = [None] * state.num_lanes
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                continue
            vaddr = state.get_lane_vgpr(lane, instr.vaddr.index)
            addend = state.resolve_lane_operand(instr.vdata0, lane)
            eff_addr = calc_lds_effective_address(vaddr, instr.offset)
            dword_idx = calc_lds_dword_index(eff_addr, state.lds.size_dwords)
            lane_inputs[lane] = (lane, dword_idx, eff_addr, addend)

        # 2. Execute atomic operations in ascending lane order (hardware serialization order)
        for lane in range(state.num_lanes):
            entry = lane_inputs[lane]
            if entry is None:
                continue
            _, dword_idx, eff_addr, addend = entry
            curr_val = state.lds.memory[dword_idx]
            old_val, new_val = ref_ds_add_rtn_u32(curr_val, addend)
            state.lds.memory[dword_idx] = new_val
            results[lane] = old_val

        # 3. Writeback to destination VGPRs for active lanes
        for lane in range(state.num_lanes):
            if results[lane] is not None:
                state.set_lane_vgpr(lane, instr.vdst.index, results[lane])  # type: ignore

        return results



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


class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarState(lds_size_dwords=1024)
        self.interp = ScalarInterpreter()

    def test_scalar_atomic_add_execution(self) -> None:
        # Initial memory at 0x40 is 1000
        addr = 0x40
        self.state.lds.write_dword(addr, 1000)

        # v1 has address 0x40, v2 has addend 250, v0 is vdst
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, 250)
        self.state.set_vgpr(0, 0xCAFEBABE)  # Initial sentinel in vdst

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=0,
        )

        old_val, new_mem = self.interp.execute(instr, self.state)

        self.assertEqual(old_val, 1000)
        self.assertEqual(new_mem, 1250)
        self.assertEqual(self.state.get_vgpr(0), 1000)  # vdst received old_val
        self.assertEqual(self.state.lds.read_dword(addr), 1250)  # LDS received new sum

    def test_scalar_execution_with_offset(self) -> None:
        # Base addr in v1 is 0x100, offset is 0x14 -> effective addr is 0x114
        base_addr = 0x100
        offset = 0x14
        eff_addr = base_addr + offset

        self.state.lds.write_dword(eff_addr, 42)
        self.state.set_vgpr(1, base_addr)
        self.state.set_vgpr(2, 58)

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=offset,
        )

        old_val, new_mem = self.interp.execute(instr, self.state)

        self.assertEqual(old_val, 42)
        self.assertEqual(new_mem, 100)
        self.assertEqual(self.state.get_vgpr(0), 42)
        self.assertEqual(self.state.lds.read_dword(eff_addr), 100)

    def test_scalar_execution_with_immediate_operand(self) -> None:
        addr = 0x200
        self.state.lds.write_dword(addr, 0x1000)
        self.state.set_vgpr(1, addr)

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=Immediate(0x555),
            offset=0,
        )

        old_val, new_mem = self.interp.execute(instr, self.state)

        self.assertEqual(old_val, 0x1000)
        self.assertEqual(new_mem, 0x1555)
        self.assertEqual(self.state.get_vgpr(0), 0x1000)
        self.assertEqual(self.state.lds.read_dword(addr), 0x1555)

    def test_scalar_unaligned_byte_address(self) -> None:
        # 0x301, 0x302, 0x303 all map to dword 0x300 (low 2 bits dropped)
        self.state.lds.write_dword(0x300, 77)
        self.state.set_vgpr(1, 0x303)  # Unaligned byte address

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=Immediate(23),
            offset=0,
        )

        old_val, new_mem = self.interp.execute(instr, self.state)
        self.assertEqual(old_val, 77)
        self.assertEqual(new_mem, 100)
        self.assertEqual(self.state.lds.read_dword(0x300), 100)

    def test_scalar_sequential_accumulation(self) -> None:
        addr = 0x50
        self.state.lds.write_dword(addr, 0)
        self.state.set_vgpr(1, addr)

        for step in range(1, 6):
            instr = DSAddRtnInstruction(
                vdst=RegisterOperand(0, RegType.VGPR),
                vaddr=RegisterOperand(1, RegType.VGPR),
                vdata0=Immediate(step),
            )
            old_val, new_mem = self.interp.execute(instr, self.state)
            self.assertEqual(old_val, sum(range(step)))
            self.assertEqual(new_mem, sum(range(step + 1)))
            self.assertEqual(self.state.get_vgpr(0), old_val)

        self.assertEqual(self.state.lds.read_dword(addr), 15)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_independent_addresses(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        # Initialize LDS: address lane * 4 has value 1000 + lane
        for lane in range(32):
            addr = lane * 4
            state.lds.write_dword(addr, 1000 + lane)
            state.set_lane_vgpr(lane, 1, addr)       # v1: vaddr
            state.set_lane_vgpr(lane, 2, 10)         # v2: addend
            state.set_lane_vgpr(lane, 0, 0xDEADBEEF) # v0: initial vdst sentinel

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=0,
        )

        results = interp.execute(instr, state)

        for lane in range(32):
            self.assertEqual(results[lane], 1000 + lane)
            self.assertEqual(state.get_lane_vgpr(lane, 0), 1000 + lane)
            self.assertEqual(state.lds.read_dword(lane * 4), 1000 + lane + 10)

    def test_wave64_divergent_exec_mask(self) -> None:
        # Alternating lanes active: even lanes active (0x5555555555555555)
        even_mask = 0x5555555555555555
        state = SIMDState(num_lanes=64, exec_mask=even_mask, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        sentinel = 0xCAFEBABE
        for lane in range(64):
            addr = lane * 4
            state.lds.write_dword(addr, 500)
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr(lane, 2, 50)
            state.set_lane_vgpr(lane, 0, sentinel)

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=0,
        )

        results = interp.execute(instr, state)

        for lane in range(64):
            if lane % 2 == 0:
                # Active lane
                self.assertEqual(results[lane], 500)
                self.assertEqual(state.get_lane_vgpr(lane, 0), 500)
                self.assertEqual(state.lds.read_dword(lane * 4), 550)
            else:
                # Inactive lane: untouched register and memory
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_vgpr(lane, 0), sentinel)
                self.assertEqual(state.lds.read_dword(lane * 4), 500)

    def test_multi_lane_lds_address_collision_serialization(self) -> None:
        # 4 lanes all target the exact same LDS dword at address 0x80
        common_addr = 0x80
        state = SIMDState(num_lanes=32, exec_mask=0xF, lds_size_dwords=1024) # Lanes 0..3 active
        interp = SIMDInterpreter()

        initial_val = 100
        state.lds.write_dword(common_addr, initial_val)

        # Lane addends: lane 0 adds 1, lane 1 adds 2, lane 2 adds 3, lane 3 adds 4
        addends = [1, 2, 3, 4]
        for lane in range(4):
            state.set_lane_vgpr(lane, 1, common_addr)
            state.set_lane_vgpr(lane, 2, addends[lane])
            state.set_lane_vgpr(lane, 0, 0)

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=0,
        )

        results = interp.execute(instr, state)

        # Lane 0: sees 100, writes 101
        self.assertEqual(results[0], 100)
        self.assertEqual(state.get_lane_vgpr(0, 0), 100)

        # Lane 1: sees 101, writes 103
        self.assertEqual(results[1], 101)
        self.assertEqual(state.get_lane_vgpr(1, 0), 101)

        # Lane 2: sees 103, writes 106
        self.assertEqual(results[2], 103)
        self.assertEqual(state.get_lane_vgpr(2, 0), 103)

        # Lane 3: sees 106, writes 110
        self.assertEqual(results[3], 106)
        self.assertEqual(state.get_lane_vgpr(3, 0), 106)

        # Final memory must equal initial_val + sum(addends) = 110
        self.assertEqual(state.lds.read_dword(common_addr), 110)

    def test_zero_exec_mask(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        addr = 0x100
        state.lds.write_dword(addr, 999)
        for lane in range(32):
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr(lane, 2, 100)
            state.set_lane_vgpr(lane, 0, 777)

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        results = interp.execute(instr, state)

        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(state.get_lane_vgpr(lane, 0), 777)
        self.assertEqual(state.lds.read_dword(addr), 999)


if __name__ == "__main__":
    unittest.main()


