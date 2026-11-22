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

# IEEE 754 Single-Precision Boundary Constants
POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001
MIN_NORMAL_F32: int = 0x00800000
MIN_SUBNORMAL_F32: int = 0x00000001
MAX_SUBNORMAL_F32: int = 0x007FFFFF
MAX_NORMAL_F32: int = 0x7F7FFFFF
ONE_F32: int = 0x3F800000


def bits_to_f32(bits: int) -> float:
    """Interpret 32-bit unsigned integer as IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Pack IEEE 754 single-precision float into 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def is_f32_nan(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 NaN."""
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def is_f32_subnormal(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 subnormal float."""
    return (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0



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


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Evaluation
# ==============================================================================

@dataclass(frozen=True)
class ArithmeticStatusFlags:
    """Condition codes and arithmetic status flags for 32-bit addition."""
    zero: bool
    sign: bool
    carry: bool
    overflow: bool


def evaluate_add_status_flags(
    old_val: int,
    addend: int,
    result: int,
) -> ArithmeticStatusFlags:
    """Evaluate condition codes, arithmetic flags, and carry/overflow generation for 32-bit add.

    Args:
        old_val: Original 32-bit memory value before addition.
        addend: 32-bit integer operand added to memory.
        result: 32-bit addition result written to memory.

    Returns:
        ArithmeticStatusFlags: Computed zero, sign, carry, and overflow flags.
    """
    a = u32(old_val)
    b = u32(addend)
    res = u32(result)

    zero = (res == 0)
    sign = bool(res & 0x80000000)
    carry = (a + b) > MASK32
    sa, sb, sres = s32(a), s32(b), s32(res)
    overflow = ((sa > 0 and sb > 0 and sres < 0) or (sa < 0 and sb < 0 and sres >= 0))

    return ArithmeticStatusFlags(
        zero=zero,
        sign=sign,
        carry=carry,
        overflow=overflow,
    )




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


class TestRegisterAliasing(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()

    def test_aliasing_vdst_equals_vaddr(self) -> None:
        # ds_add_rtn_u32 v0, v0, v1
        state = ScalarState(lds_size_dwords=1024)
        addr = 0x60
        initial_mem = 300
        addend = 50

        state.lds.write_dword(addr, initial_mem)
        state.set_vgpr(0, addr)    # v0 = addr (also vdst)
        state.set_vgpr(1, addend)  # v1 = addend

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(1, RegType.VGPR),
            offset=0,
        )

        old_val, new_mem = self.interp.execute(instr, state)

        self.assertEqual(old_val, initial_mem)
        self.assertEqual(new_mem, initial_mem + addend)
        self.assertEqual(state.get_vgpr(0), initial_mem) # v0 received old_val
        self.assertEqual(state.get_vgpr(1), addend)      # v1 preserved
        self.assertEqual(state.lds.read_dword(addr), initial_mem + addend)

    def test_aliasing_vdst_equals_vdata0(self) -> None:
        # ds_add_rtn_u32 v1, v0, v1
        state = ScalarState(lds_size_dwords=1024)
        addr = 0x80
        initial_mem = 1000
        addend = 250

        state.lds.write_dword(addr, initial_mem)
        state.set_vgpr(0, addr)    # v0 = addr
        state.set_vgpr(1, addend)  # v1 = addend (also vdst)

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(1, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(1, RegType.VGPR),
            offset=0,
        )

        old_val, new_mem = self.interp.execute(instr, state)

        self.assertEqual(old_val, initial_mem)
        self.assertEqual(new_mem, initial_mem + addend)
        self.assertEqual(state.get_vgpr(0), addr)        # v0 preserved
        self.assertEqual(state.get_vgpr(1), initial_mem) # v1 received old_val
        self.assertEqual(state.lds.read_dword(addr), initial_mem + addend)

    def test_aliasing_vaddr_equals_vdata0(self) -> None:
        # ds_add_rtn_u32 v2, v0, v0
        state = ScalarState(lds_size_dwords=1024)
        addr_and_addend = 0x40  # 64 bytes -> dword index 16
        initial_mem = 100

        state.lds.write_dword(addr_and_addend, initial_mem)
        state.set_vgpr(0, addr_and_addend) # v0 is both addr and addend
        state.set_vgpr(2, 0xCAFEBABE)      # v2 is vdst

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(2, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(0, RegType.VGPR),
            offset=0,
        )

        old_val, new_mem = self.interp.execute(instr, state)

        self.assertEqual(old_val, initial_mem)
        self.assertEqual(new_mem, initial_mem + addr_and_addend)
        self.assertEqual(state.get_vgpr(2), initial_mem)
        self.assertEqual(state.get_vgpr(0), addr_and_addend) # v0 untouched
        self.assertEqual(state.lds.read_dword(addr_and_addend), initial_mem + addr_and_addend)

    def test_triple_aliasing_vdst_vaddr_vdata0(self) -> None:
        # ds_add_rtn_u32 v0, v0, v0
        state = ScalarState(lds_size_dwords=1024)
        val = 0x20
        initial_mem = 500

        state.lds.write_dword(val, initial_mem)
        state.set_vgpr(0, val)

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(0, RegType.VGPR),
            offset=0,
        )

        old_val, new_mem = self.interp.execute(instr, state)

        self.assertEqual(old_val, initial_mem)
        self.assertEqual(new_mem, initial_mem + val)
        self.assertEqual(state.get_vgpr(0), initial_mem) # Overwritten with old_val
        self.assertEqual(state.lds.read_dword(val), initial_mem + val)

    def test_unreferenced_registers_preservation(self) -> None:
        state = ScalarState(lds_size_dwords=1024)
        state.set_vgpr(1, 0x100) # vaddr
        state.set_vgpr(2, 10)    # vdata0

        # Fill background registers
        for r in range(3, 20):
            state.set_vgpr(r, 0x1000 + r)
        for s in range(10):
            state.set_sgpr(s, 0x2000 + s)

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )
        self.interp.execute(instr, state)

        for r in range(3, 20):
            self.assertEqual(state.get_vgpr(r), 0x1000 + r)
        for s in range(10):
            self.assertEqual(state.get_sgpr(s), 0x2000 + s)

    def test_simd_multi_lane_aliasing(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF, lds_size_dwords=1024)
        simd_interp = SIMDInterpreter()

        # Each lane has vdst == vaddr: v0 is both dst and addr
        for lane in range(32):
            addr = lane * 4
            state.lds.write_dword(addr, 50 + lane)
            state.set_lane_vgpr(lane, 0, addr) # v0 is addr and dst
            state.set_lane_vgpr(lane, 1, 100)  # v1 is addend

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(1, RegType.VGPR),
        )

        results = simd_interp.execute(instr, state)

        for lane in range(32):
            self.assertEqual(results[lane], 50 + lane)
            self.assertEqual(state.get_lane_vgpr(lane, 0), 50 + lane) # Received old_val
            self.assertEqual(state.get_lane_vgpr(lane, 1), 100)       # Addend preserved
            self.assertEqual(state.lds.read_dword(lane * 4), 150 + lane)


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_normal_positive_flags(self) -> None:
        old_val, new_mem = ref_ds_add_rtn_u32(10, 20)
        flags = evaluate_add_status_flags(10, 20, new_mem)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.sign)
        self.assertFalse(flags.carry)
        self.assertFalse(flags.overflow)

    def test_zero_flag(self) -> None:
        old_val, new_mem = ref_ds_add_rtn_u32(0, 0)
        flags = evaluate_add_status_flags(0, 0, new_mem)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.sign)
        self.assertFalse(flags.carry)
        self.assertFalse(flags.overflow)

        # Zero produced by 32-bit overflow wrap
        old_val2, new_mem2 = ref_ds_add_rtn_u32(MASK32, 1)
        flags2 = evaluate_add_status_flags(MASK32, 1, new_mem2)
        self.assertEqual(new_mem2, 0)
        self.assertTrue(flags2.zero)
        self.assertFalse(flags2.sign)
        self.assertTrue(flags2.carry)
        self.assertFalse(flags2.overflow)

    def test_sign_flag(self) -> None:
        old_val, new_mem = ref_ds_add_rtn_u32(0x40000000, 0x40000000)
        self.assertEqual(new_mem, 0x80000000)
        flags = evaluate_add_status_flags(0x40000000, 0x40000000, new_mem)
        self.assertTrue(flags.sign)
        self.assertFalse(flags.carry)
        self.assertTrue(flags.overflow)

    def test_unsigned_carry_flag(self) -> None:
        old_val, new_mem = ref_ds_add_rtn_u32(MASK32, 10)
        self.assertEqual(new_mem, 9)
        flags = evaluate_add_status_flags(MASK32, 10, new_mem)
        self.assertTrue(flags.carry)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.overflow)

    def test_signed_overflow_positive_to_negative(self) -> None:
        # 0x7FFFFFFF (+2147483647) + 1 -> 0x80000000 (-2147483648)
        old_val, new_mem = ref_ds_add_rtn_u32(0x7FFFFFFF, 1)
        self.assertEqual(new_mem, 0x80000000)
        flags = evaluate_add_status_flags(0x7FFFFFFF, 1, new_mem)
        self.assertTrue(flags.overflow)
        self.assertTrue(flags.sign)
        self.assertFalse(flags.carry)

    def test_signed_overflow_negative_to_positive(self) -> None:
        # 0x80000000 (-2147483648) + 0x80000000 (-2147483648) -> 0
        old_val, new_mem = ref_ds_add_rtn_u32(0x80000000, 0x80000000)
        self.assertEqual(new_mem, 0)
        flags = evaluate_add_status_flags(0x80000000, 0x80000000, new_mem)
        self.assertTrue(flags.overflow)
        self.assertTrue(flags.carry)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.sign)

    def test_positive_and_negative_no_overflow(self) -> None:
        # 0x7FFFFFFF (+2147483647) + 0xFFFFFFFF (-1) -> 0x7FFFFFFE (+2147483646)
        old_val, new_mem = ref_ds_add_rtn_u32(0x7FFFFFFF, MASK32)
        self.assertEqual(new_mem, 0x7FFFFFFE)
        flags = evaluate_add_status_flags(0x7FFFFFFF, MASK32, new_mem)
        self.assertFalse(flags.overflow)
        self.assertTrue(flags.carry)
        self.assertFalse(flags.sign)
        self.assertFalse(flags.zero)


class TestIEEE754Boundaries(unittest.TestCase):
    """Phase 7: Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_increment_to_normal(self) -> None:
        # ds_add_rtn_u32 operates as raw integer arithmetic; float bits are unmodified by float semantics
        self.assertTrue(is_f32_subnormal(MAX_SUBNORMAL_F32))
        old_val, new_mem = ref_ds_add_rtn_u32(MAX_SUBNORMAL_F32, 1)
        self.assertEqual(old_val, MAX_SUBNORMAL_F32)
        self.assertEqual(new_mem, MIN_NORMAL_F32)
        self.assertFalse(is_f32_subnormal(new_mem))
        self.assertFalse(is_f32_nan(new_mem))

    def test_max_normal_increment_to_infinity(self) -> None:
        old_val, new_mem = ref_ds_add_rtn_u32(MAX_NORMAL_F32, 1)
        self.assertEqual(old_val, MAX_NORMAL_F32)
        self.assertEqual(new_mem, POS_INF_F32)
        self.assertEqual(bits_to_f32(new_mem), float("inf"))

    def test_signaling_nan_preserved_no_quietening(self) -> None:
        # Ensure signaling NaN does not get quietened into a QNaN (no bit 22 forcing)
        self.assertTrue(is_f32_nan(SNAN_F32))
        old_val, new_mem = ref_ds_add_rtn_u32(SNAN_F32, 0)
        self.assertEqual(old_val, SNAN_F32)
        self.assertEqual(new_mem, SNAN_F32)
        self.assertNotEqual(new_mem, QNAN_F32)

    def test_quiet_nan_payload_manipulation(self) -> None:
        self.assertTrue(is_f32_nan(QNAN_F32))
        old_val, new_mem = ref_ds_add_rtn_u32(QNAN_F32, 5)
        self.assertEqual(old_val, QNAN_F32)
        self.assertEqual(new_mem, QNAN_F32 + 5)
        self.assertTrue(is_f32_nan(new_mem))

    def test_negative_zero_preservation(self) -> None:
        # Negative zero (0x80000000) added with positive zero (0) remains 0x80000000
        old_val, new_mem = ref_ds_add_rtn_u32(NEG_ZERO_F32, POS_ZERO_F32)
        self.assertEqual(old_val, NEG_ZERO_F32)
        self.assertEqual(new_mem, NEG_ZERO_F32)
        self.assertEqual(bits_to_f32(new_mem), -0.0)

    def test_subnormal_no_ftz_in_interpreter(self) -> None:
        # Verify scalar interpreter preserves subnormal values in LDS without flush-to-zero
        state = ScalarState(lds_size_dwords=1024)
        interp = ScalarInterpreter()

        addr = 0x20
        state.lds.write_dword(addr, MIN_SUBNORMAL_F32)
        state.set_vgpr(1, addr)
        state.set_vgpr(2, 2)

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        old_val, new_mem = interp.execute(instr, state)
        self.assertEqual(old_val, MIN_SUBNORMAL_F32)
        self.assertEqual(new_mem, MIN_SUBNORMAL_F32 + 2)
        self.assertTrue(is_f32_subnormal(new_mem))
        self.assertEqual(state.lds.read_dword(addr), MIN_SUBNORMAL_F32 + 2)


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Phase 8: Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_registers(self) -> None:
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(106, RegType.SGPR)

    def test_syntax_rejection_instruction_operands(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        s0 = RegisterOperand(0, RegType.SGPR)

        with self.assertRaises(ValueError):
            DSAddRtnInstruction(vdst=s0, vaddr=v0, vdata0=v0)
        with self.assertRaises(ValueError):
            DSAddRtnInstruction(vdst=v0, vaddr=s0, vdata0=v0)
        with self.assertRaises(ValueError):
            DSAddRtnInstruction(vdst=v0, vaddr=v0, vdata0=v0, offset=-1)
        with self.assertRaises(ValueError):
            DSAddRtnInstruction(vdst=v0, vaddr=v0, vdata0=v0, offset=65536)

    def test_syntax_rejection_memory_and_simd_state(self) -> None:
        with self.assertRaises(ValueError):
            LDSMemory(size_dwords=0)
        with self.assertRaises(ValueError):
            LDSMemory(size_dwords=-16)
        with self.assertRaises(ValueError):
            LDSMemory(size_dwords=1000)  # Non-power-of-2

        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_negative_control_bit_perturbation(self) -> None:
        old_val, new_mem = ref_ds_add_rtn_u32(100, 200)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_mem, 300)

        # Perturbed bits must be rejected
        perturbed_old = old_val ^ 1
        perturbed_new = new_mem ^ 1
        self.assertNotEqual(old_val, perturbed_old)
        self.assertNotEqual(new_mem, perturbed_new)

    def test_mutation_detection_add_to_sub(self) -> None:
        def mutant_add(mem: int, val: int) -> Tuple[int, int]:
            # Mutant: subtraction instead of addition
            return u32(mem), u32(u32(mem) - u32(val))

        correct_old, correct_new = ref_ds_add_rtn_u32(100, 40)
        mutant_old, mutant_new = mutant_add(100, 40)

        self.assertEqual(correct_new, 140)
        self.assertEqual(mutant_new, 60)
        self.assertNotEqual(correct_new, mutant_new)

    def test_mutation_detection_missing_32bit_wrap(self) -> None:
        def mutant_no_wrap(mem: int, val: int) -> Tuple[int, int]:
            # Mutant: does not wrap modulo 2^32
            return mem, mem + val

        correct_old, correct_new = ref_ds_add_rtn_u32(MASK32, 1)
        mutant_old, mutant_new = mutant_no_wrap(MASK32, 1)

        self.assertEqual(correct_new, 0)
        self.assertEqual(mutant_new, 0x100000000)
        self.assertNotEqual(correct_new, mutant_new)

    def test_mutation_detection_return_new_instead_of_old(self) -> None:
        def mutant_return_post_val(mem: int, val: int) -> Tuple[int, int]:
            # Mutant: returns new_mem instead of old_val
            new_val = u32(u32(mem) + u32(val))
            return new_val, new_val

        correct_old, correct_new = ref_ds_add_rtn_u32(50, 25)
        mutant_old, mutant_new = mutant_return_post_val(50, 25)

        self.assertEqual(correct_old, 50)
        self.assertEqual(mutant_old, 75)
        self.assertNotEqual(correct_old, mutant_old)

    def test_mutation_detection_dropped_immediate_offset(self) -> None:
        def mutant_eff_addr(vaddr: int, offset: int = 0) -> int:
            # Mutant: drops immediate offset
            return vaddr & MASK32

        base = 0x200
        offset = 0x18
        correct = calc_lds_effective_address(base, offset)
        mutant = mutant_eff_addr(base, offset)

        self.assertEqual(correct, 0x218)
        self.assertEqual(mutant, 0x200)
        self.assertNotEqual(correct, mutant)

    def test_mutation_detection_simd_exec_leak(self) -> None:
        # Mutant: SIMD interpreter executes inactive lanes
        class MutantSIMDInterpreter(SIMDInterpreter):
            def execute(self, instr: DSAddRtnInstruction, state: SIMDState) -> List[Optional[int]]:
                results: List[Optional[int]] = [None] * state.num_lanes
                for lane in range(state.num_lanes):
                    # Bug: ignores is_lane_active
                    vaddr = state.get_lane_vgpr(lane, instr.vaddr.index)
                    addend = state.resolve_lane_operand(instr.vdata0, lane)
                    eff_addr = calc_lds_effective_address(vaddr, instr.offset)
                    dword_idx = calc_lds_dword_index(eff_addr, state.lds.size_dwords)
                    curr_val = state.lds.memory[dword_idx]
                    old_val, new_val = ref_ds_add_rtn_u32(curr_val, addend)
                    state.lds.memory[dword_idx] = new_val
                    results[lane] = old_val
                    state.set_lane_vgpr(lane, instr.vdst.index, old_val)
                return results

        state = SIMDState(num_lanes=32, exec_mask=0, lds_size_dwords=1024) # All inactive
        sentinel = 0x5555
        for lane in range(32):
            state.set_lane_vgpr(lane, 0, sentinel)
            state.set_lane_vgpr(lane, 1, 0x40)
            state.set_lane_vgpr(lane, 2, 1)

        state.lds.write_dword(0x40, 100)
        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        mutant = MutantSIMDInterpreter()
        mutant.execute(instr, state)

        # Mutation caught: inactive lanes were modified
        corrupted = [l for l in range(32) if state.get_lane_vgpr(l, 0) != sentinel]
        self.assertEqual(len(corrupted), 32)
        self.assertNotEqual(state.lds.read_dword(0x40), 100)

    def test_mutation_detection_premature_writeback_aliasing(self) -> None:
        # Mutant: overwrites destination register before reading address/operands
        class PrematureWritebackInterpreter(ScalarInterpreter):
            def execute(self, instr: DSAddRtnInstruction, state: ScalarState) -> Tuple[int, int]:
                # Bug: sets vdst before reading vaddr
                state.set_reg(instr.vdst, 0xDEADBEEF)
                vaddr = state.get_reg(instr.vaddr)  # Reads corrupted address 0xDEADBEEF!
                addend = state.resolve_operand(instr.vdata0)
                old_val, new_mem = state.lds.atomic_add_rtn_u32(vaddr=vaddr, vdata0=addend, offset=instr.offset)
                return old_val, new_mem

        state = ScalarState(lds_size_dwords=1024)
        addr = 0x50
        state.set_vgpr(0, addr) # v0 is both vdst and vaddr
        state.set_vgpr(1, 10)
        state.lds.write_dword(addr, 500)

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(1, RegType.VGPR),
        )

        buggy = PrematureWritebackInterpreter()
        old_val, _ = buggy.execute(instr, state)
        # Buggy interpreter read corrupted address, returning 0 instead of 500
        self.assertNotEqual(old_val, 500)

    def test_mutation_detection_reversed_collision_order(self) -> None:
        # Mutant: serializes colliding lanes in reverse order
        class ReversedCollisionSIMDInterpreter(SIMDInterpreter):
            def execute(self, instr: DSAddRtnInstruction, state: SIMDState) -> List[Optional[int]]:
                results: List[Optional[int]] = [None] * state.num_lanes
                # Bug: iterates in reversed lane order
                for lane in reversed(range(state.num_lanes)):
                    if not state.is_lane_active(lane):
                        continue
                    vaddr = state.get_lane_vgpr(lane, instr.vaddr.index)
                    addend = state.resolve_lane_operand(instr.vdata0, lane)
                    eff_addr = calc_lds_effective_address(vaddr, instr.offset)
                    dword_idx = calc_lds_dword_index(eff_addr, state.lds.size_dwords)
                    curr_val = state.lds.memory[dword_idx]
                    old_val, new_val = ref_ds_add_rtn_u32(curr_val, addend)
                    state.lds.memory[dword_idx] = new_val
                    results[lane] = old_val
                    state.set_lane_vgpr(lane, instr.vdst.index, old_val)
                return results

        state = SIMDState(num_lanes=32, exec_mask=0x3, lds_size_dwords=1024) # Lanes 0 and 1 active
        common_addr = 0x100
        state.lds.write_dword(common_addr, 0)

        # Lane 0 adds 10, Lane 1 adds 20
        state.set_lane_vgpr(0, 1, common_addr)
        state.set_lane_vgpr(0, 2, 10)
        state.set_lane_vgpr(1, 1, common_addr)
        state.set_lane_vgpr(1, 2, 20)

        instr = DSAddRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        buggy = ReversedCollisionSIMDInterpreter()
        results = buggy.execute(instr, state)

        # In reverse order, Lane 1 executes first and sees 0, Lane 0 executes second and sees 20
        # In correct ascending order: Lane 0 sees 0, Lane 1 sees 10!
        self.assertEqual(results[1], 0)   # Mutation: Lane 1 got 0
        self.assertEqual(results[0], 20)  # Mutation: Lane 0 got 20
        self.assertNotEqual(results[0], 0)


if __name__ == "__main__":
    unittest.main()






