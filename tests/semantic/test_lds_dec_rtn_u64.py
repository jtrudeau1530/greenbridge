from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_dec_rtn_u64.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 64-bit Decrement with Return:
Performs a 64-bit atomic wrapping decrement on LDS memory and returns the original 64-bit value.

Syntax:
    ds_dec_rtn_u64 vdst, vaddr, vdata0 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + (offset16 & 0xFFFF)) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_qwords - 1)
    old_val = LDS[qword_index]
    if old_val == 0 or old_val > vdata0:
        new_mem = vdata0
    else:
        new_mem = (old_val - 1) & 0xFFFFFFFFFFFFFFFF
    LDS[qword_index] = new_mem
    vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Capacity: 64 KiB default capacity (8,192 64-bit qwords / 16,384 32-bit dwords).
  - Addressing: Base byte address in VGPR vaddr + 16-bit unsigned immediate offset (offset:offset16).
  - Alignment: 64-bit qword aligned; low 3 bits of byte address are truncated (eff_addr >> 3).
  - Wrap Condition: Unsigned 64-bit comparison:
      If old_val == 0 or old_val > vdata0: new_mem = vdata0
      Else: new_mem = (old_val - 1) mod 2^64.
  - Return: The pre-operation 64-bit value is returned to VGPR pair vdst (v[dst:dst+1]).
"""

from dataclasses import dataclass
from enum import Enum, auto
import math
import struct
from typing import Dict, List, Optional, Tuple, Union
import unittest

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536          # 64 KiB capacity
LDS_SIZE_DWORDS: int = 16384         # 16,384 32-bit dwords
LDS_SIZE_QWORDS: int = 8192          # 8,192 64-bit qwords
LDS_BANK_COUNT: int = 32             # 32 memory banks
MASK_32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
MASK_64: int = 0xFFFF_FFFF_FFFF_FFFF # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF          # 16-bit offset mask
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
DEFAULT_NUM_LANES: int = 64

# IEEE 754 Double-Precision Boundary Constants
POS_ZERO_F64: int = 0x0000000000000000
NEG_ZERO_F64: int = 0x8000000000000000
POS_INF_F64: int = 0x7FF0000000000000
NEG_INF_F64: int = 0xFFF0000000000000
QNAN_F64: int = 0x7FF8000000000000
SNAN_F64: int = 0x7FF0000000000001
MIN_NORMAL_F64: int = 0x0010000000000000
MIN_SUBNORMAL_F64: int = 0x0000000000000001
MAX_SUBNORMAL_F64: int = 0x000FFFFFFFFFFFFF
MAX_NORMAL_F64: int = 0x7FEFFFFFFFFFFFFF
ONE_F64: int = 0x3FF0000000000000


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK_64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as signed two's-complement integer."""
    val &= MASK_64
    return val - (1 << 64) if (val & (1 << 63)) else val


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def split_u64_to_u32s(val: int) -> Tuple[int, int]:
    """Split 64-bit unsigned integer into (lo, hi) 32-bit words."""
    val &= MASK_64
    return val & MASK_32, (val >> 32) & MASK_32


def combine_u32s_to_u64(lo: int, hi: int) -> int:
    """Combine (lo, hi) 32-bit words into a 64-bit unsigned integer."""
    return ((hi & MASK_32) << 32) | (lo & MASK_32)


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate byte offset.
    Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = LDS_SIZE_QWORDS) -> int:
    """Convert an effective byte address to an aligned LDS 64-bit qword index.

    LDS 64-bit addresses are qword-aligned; the low 3 bits are ignored by hardware.
    The index wraps modulo the power-of-two qword capacity.
    """
    return (eff_addr >> 3) & (capacity_qwords - 1)


def ref_ds_dec_u64(old_val: int, vdata0: int) -> int:
    """Pure reference mathematical formula for ds_dec_u64.

    If old_val == 0 or old_val > vdata0:
        new_val = vdata0
    else:
        new_val = (old_val - 1) & 0xFFFFFFFFFFFFFFFF
    """
    a = u64(old_val)
    b = u64(vdata0)
    if a == 0 or a > b:
        return b
    return u64(a - 1)


def oracle_ds_dec_rtn_u64(
    current_mem: int,
    wrap_limit: int,
) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_dec_rtn_u64.

    Computes:
        old_val = current_mem & 0xFFFFFFFFFFFFFFFF
        new_mem = ref_ds_dec_u64(old_val, wrap_limit)

    Returns:
        (old_val, new_mem)
    """
    old_val = u64(current_mem)
    new_mem = ref_ds_dec_u64(old_val, wrap_limit)
    return old_val, new_mem


class LDSModel64:
    """Architectural memory model representing the Local Data Share scratchpad with 64-bit access."""

    def __init__(self, capacity_qwords: int = LDS_SIZE_QWORDS) -> None:
        if capacity_qwords <= 0 or (capacity_qwords & (capacity_qwords - 1)) != 0:
            raise ValueError(f"capacity_qwords must be a positive power of two, got {capacity_qwords}")
        self.capacity_qwords = capacity_qwords
        self.memory: List[int] = [0] * capacity_qwords

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword from LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        return self.memory[idx]

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit qword to LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        self.memory[idx] = u64(val)

    def atomic_dec_rtn_u64(
        self,
        vaddr: int,
        wrap_limit: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 64-bit wrapping decrement with return on LDS.

        Returns:
            Tuple[int, int]: (old_val, new_mem)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        curr = self.memory[idx]
        old_val, new_mem = oracle_ds_dec_rtn_u64(curr, wrap_limit)
        self.memory[idx] = new_mem
        return old_val, new_mem


def bytearray_oracle_ds_dec_rtn_u64(
    lds_bytes: bytearray,
    vaddr: int,
    wrap_limit: int,
    offset: int = 0,
    lds_size_bytes: int = LDS_SIZE_BYTES,
) -> Tuple[int, int]:
    """Reference bytearray mathematical oracle for RDNA2 ds_dec_rtn_u64.

    Computes effective address, reads previous 64-bit qword, applies wrapping
    decrement logic, writes updated 64-bit qword back, and returns (old_val, new_mem).
    """
    eff_addr = calc_lds_effective_address(vaddr, offset) & (lds_size_bytes - 1)
    aligned_addr = eff_addr & ~7  # qword aligned (drop low 3 bits)
    old_val = struct.unpack_from("<Q", lds_bytes, aligned_addr)[0]
    new_mem = ref_ds_dec_u64(old_val, wrap_limit)
    struct.pack_into("<Q", lds_bytes, aligned_addr, new_mem)
    return old_val, new_mem


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Supported Data Share (DS) 64-bit atomic opcodes."""
    DS_DEC_RTN_U64 = "ds_dec_rtn_u64"


class RegType(Enum):
    """Register file classes."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegSingle:
    """Represents a single 32-bit register."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        max_limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < max_limit):
            raise ValueError(
                f"Register index out of bounds: {self.reg_type.value}{self.index} "
                f"(expected 0 <= index < {max_limit})"
            )

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}{self.index}"


@dataclass(frozen=True)
class RegPair:
    """Represents an aligned 64-bit pair of registers (e.g. v[0:1])."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        max_limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < max_limit - 1):
            raise ValueError(
                f"Register pair base index out of bounds: {self.reg_type.value}[{self.index}:{self.index + 1}] "
                f"(expected 0 <= index < {max_limit - 1})"
            )

    @property
    def indices(self) -> Tuple[int, int]:
        return self.index, self.index + 1

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}[{self.index}:{self.index + 1}]"


@dataclass(frozen=True)
class DSDecRtnU64Instruction:
    """Symbolic AST node for ds_dec_rtn_u64 atomic instruction."""
    vdst: RegPair
    vaddr: RegSingle
    vdata0: RegPair
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_DEC_RTN_U64

    def __post_init__(self) -> None:
        if not isinstance(self.vdst, RegPair) or self.vdst.reg_type != RegType.VGPR:
            raise TypeError("vdst operand must be a VGPR RegPair")
        if not isinstance(self.vaddr, RegSingle) or self.vaddr.reg_type != RegType.VGPR:
            raise TypeError("vaddr operand must be a VGPR RegSingle")
        if not isinstance(self.vdata0, RegPair) or self.vdata0.reg_type != RegType.VGPR:
            raise TypeError("vdata0 operand must be a VGPR RegPair")
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"offset out of 16-bit range [0, 65535]: {self.offset}")

    def disasm(self) -> str:
        """Format canonical assembly mnemonic and operand string."""
        text = f"{self.opcode.value} {self.vdst.name}, {self.vaddr.name}, {self.vdata0.name}"
        if self.offset != 0:
            text += f" offset:{self.offset}"
        if self.gds:
            text += " gds"
        return text


# ==============================================================================
# Deterministic Scalar Interpreter & Machine State
# ==============================================================================

class ScalarRegisterFile:
    """Models a 256-register 32-bit scalar VGPR file with 64-bit pair support."""

    def __init__(self) -> None:
        self.regs: List[int] = [0] * DEFAULT_NUM_VGPRS

    def get_u32(self, reg_idx: int) -> int:
        return self.regs[reg_idx] & MASK_32

    def set_u32(self, reg_idx: int, val: int) -> None:
        self.regs[reg_idx] = val & MASK_32

    def get_u64(self, pair: RegPair) -> int:
        lo = self.get_u32(pair.index)
        hi = self.get_u32(pair.index + 1)
        return combine_u32s_to_u64(lo, hi)

    def set_u64(self, pair: RegPair, val: int) -> None:
        lo, hi = split_u64_to_u32s(val)
        self.set_u32(pair.index, lo)
        self.set_u32(pair.index + 1, hi)


class ScalarMachineState:
    """Architectural state for scalar single-lane 64-bit emulation."""

    def __init__(self, capacity_qwords: int = LDS_SIZE_QWORDS) -> None:
        self.rf = ScalarRegisterFile()
        self.sgprs = [0] * DEFAULT_NUM_SGPRS
        self.lds = LDSModel64(capacity_qwords)

    def get_vgpr(self, index: int) -> int:
        return self.rf.get_u32(index)

    def set_vgpr(self, index: int, val: int) -> None:
        self.rf.set_u32(index, val)

    def get_vgpr64(self, pair: RegPair) -> int:
        return self.rf.get_u64(pair)

    def set_vgpr64(self, pair: RegPair, val: int) -> None:
        self.rf.set_u64(pair, val)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for ds_dec_rtn_u64."""

    def step(
        self, instr: DSDecRtnU64Instruction, state: ScalarMachineState
    ) -> Tuple[int, int]:
        """Execute a single ds_dec_rtn_u64 instruction deterministically on scalar state.

        Snapshot operands before updating destination to ensure deterministic register handling.
        """
        vaddr = state.get_vgpr(instr.vaddr.index)
        vdata0 = state.get_vgpr64(instr.vdata0)
        old_val, new_mem = state.lds.atomic_dec_rtn_u64(
            vaddr=vaddr,
            wrap_limit=vdata0,
            offset=instr.offset,
        )
        state.set_vgpr64(instr.vdst, old_val)
        return old_val, new_mem


# ==============================================================================
# SIMD Multi-Lane State and Divergence Interpreter
# ==============================================================================

class SIMDState:
    """SIMD vector machine state with multi-lane VGPR files and shared LDS."""

    def __init__(
        self,
        num_lanes: int = 32,
        exec_mask: int = MASK_32,
        capacity_qwords: int = LDS_SIZE_QWORDS,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {num_lanes}")
        self.num_lanes = num_lanes
        self.exec_mask = (exec_mask & MASK_32) if num_lanes == 32 else (exec_mask & MASK_64)
        self.vgprs: List[List[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(num_lanes)]
        self.lds = LDSModel64(capacity_qwords)

    def is_lane_active(self, lane: int) -> bool:
        if 0 <= lane < self.num_lanes:
            return bool((self.exec_mask >> lane) & 1)
        return False

    def get_lane_vgpr(self, lane: int, reg: int) -> int:
        return self.vgprs[lane][reg] & MASK_32

    def set_lane_vgpr(self, lane: int, reg: int, val: int) -> None:
        self.vgprs[lane][reg] = u32(val)

    def get_lane_vgpr64(self, lane: int, pair: RegPair) -> int:
        lo = self.get_lane_vgpr(lane, pair.index)
        hi = self.get_lane_vgpr(lane, pair.index + 1)
        return combine_u32s_to_u64(lo, hi)

    def set_lane_vgpr64(self, lane: int, pair: RegPair, val: int) -> None:
        lo, hi = split_u64_to_u32s(val)
        self.set_lane_vgpr(lane, pair.index, lo)
        self.set_lane_vgpr(lane, pair.index + 1, hi)


class SIMDInterpreter:
    """SIMD multi-lane execution interpreter handling divergence and EXEC masking for ds_dec_rtn_u64."""

    def step(
        self,
        instr: DSDecRtnU64Instruction,
        state: SIMDState,
    ) -> List[Optional[Tuple[int, int]]]:
        """Execute ds_dec_rtn_u64 across all lanes in the wave.

        Lanes execute according to exec_mask. Inactive lanes do not touch LDS memory
        and their destination VGPRs are not modified.
        Contending active lanes targeting the same address are serialized deterministically
        in lane order (lane 0, lane 1, etc.).
        """
        results: List[Optional[Tuple[int, int]]] = []
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                results.append(None)
                continue

            vaddr = state.get_lane_vgpr(lane, instr.vaddr.index)
            vdata0 = state.get_lane_vgpr64(lane, instr.vdata0)
            old_val, new_mem = state.lds.atomic_dec_rtn_u64(
                vaddr=vaddr,
                wrap_limit=vdata0,
                offset=instr.offset,
            )
            state.set_lane_vgpr64(lane, instr.vdst, old_val)
            results.append((old_val, new_mem))
        return results


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSModel64(capacity_qwords=1024)
        self.lds_bytes = bytearray(LDS_SIZE_BYTES)

    def test_normal_decrement_u64(self) -> None:
        self.mem.write_qword(10 * 8, 5)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=80, wrap_limit=10, offset=0)
        self.assertEqual(old_val, 5)
        self.assertEqual(new_val, 4)
        self.assertEqual(self.mem.read_qword(10 * 8), 4)

        # Cross 32-bit boundary decrement: 0x1_0000_0000 -> 0x0_FFFF_FFFF
        self.mem.write_qword(10 * 8, 0x1_0000_0000)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=80, wrap_limit=0x5_0000_0000, offset=0)
        self.assertEqual(old_val, 0x1_0000_0000)
        self.assertEqual(new_val, 0x0_FFFF_FFFF)
        self.assertEqual(self.mem.read_qword(10 * 8), 0x0_FFFF_FFFF)

    def test_wrap_at_zero_u64(self) -> None:
        bound = 0x1234_5678_9ABC_DEF0
        self.mem.write_qword(20 * 8, 0)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=160, wrap_limit=bound, offset=0)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, bound)
        self.assertEqual(self.mem.read_qword(20 * 8), bound)

    def test_wrap_strictly_greater_than_bound_u64(self) -> None:
        bound = 0x1000_0000_0000_0000
        val_greater = 0xF000_0000_0000_0000
        self.mem.write_qword(30 * 8, val_greater)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=240, wrap_limit=bound, offset=0)
        self.assertEqual(old_val, val_greater)
        self.assertEqual(new_val, bound)
        self.assertEqual(self.mem.read_qword(30 * 8), bound)

        # MASK_64 wraps to bound
        self.mem.write_qword(30 * 8, MASK_64)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=240, wrap_limit=bound, offset=0)
        self.assertEqual(old_val, MASK_64)
        self.assertEqual(new_val, bound)
        self.assertEqual(self.mem.read_qword(30 * 8), bound)

    def test_decrement_when_equal_to_bound_u64(self) -> None:
        bound = 0xCAFE_BABE_DEAD_BEEF
        self.mem.write_qword(40 * 8, bound)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=320, wrap_limit=bound, offset=0)
        self.assertEqual(old_val, bound)
        self.assertEqual(new_val, bound - 1)
        self.assertEqual(self.mem.read_qword(40 * 8), bound - 1)

    def test_boundary_bound_zero_u64(self) -> None:
        self.mem.write_qword(5 * 8, 0)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=40, wrap_limit=0, offset=0)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, 0)

        self.mem.write_qword(5 * 8, 1)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=40, wrap_limit=0, offset=0)
        self.assertEqual(old_val, 1)
        self.assertEqual(new_val, 0)

        self.mem.write_qword(5 * 8, MASK_64)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=40, wrap_limit=0, offset=0)
        self.assertEqual(old_val, MASK_64)
        self.assertEqual(new_val, 0)

    def test_boundary_bound_max_u64(self) -> None:
        self.mem.write_qword(6 * 8, 0)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=48, wrap_limit=MASK_64, offset=0)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, MASK_64)

        self.mem.write_qword(6 * 8, MASK_64)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=48, wrap_limit=MASK_64, offset=0)
        self.assertEqual(old_val, MASK_64)
        self.assertEqual(new_val, MASK_64 - 1)

    def test_countdown_cycle_u64(self) -> None:
        bound = 3
        idx = 15
        self.mem.write_qword(idx * 8, bound)

        expected_sequence = [
            (3, 2),
            (2, 1),
            (1, 0),
            (0, 3),
            (3, 2),
        ]
        for exp_old, exp_new in expected_sequence:
            old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=idx * 8, wrap_limit=bound)
            self.assertEqual(old_val, exp_old)
            self.assertEqual(new_val, exp_new)
            self.assertEqual(self.mem.read_qword(idx * 8), exp_new)

    def test_effective_address_and_qword_alignment(self) -> None:
        # vaddr = 0x1000, offset = 16 -> eff = 0x1010 -> qword index = (0x1010 >> 3) & 1023
        self.mem.write_qword(0x1010, 50)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=0x1000, wrap_limit=100, offset=16)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_val, 49)
        self.assertEqual(self.mem.read_qword(0x1010), 49)

        # Low 3 bits of byte address ignored (offset 7 -> eff 0x1007 -> aligned to 0x1000)
        self.mem.write_qword(0x1000, 25)
        old_val, new_val = self.mem.atomic_dec_rtn_u64(vaddr=0x1000, wrap_limit=100, offset=7)
        self.assertEqual(old_val, 25)
        self.assertEqual(new_val, 24)
        self.assertEqual(self.mem.read_qword(0x1000), 24)

    def test_bytearray_oracle_equivalence_u64(self) -> None:
        addr = 0x200
        val64 = 0x1122_3344_5566_7788
        struct.pack_into("<Q", self.lds_bytes, addr, val64)
        old_val, new_val = bytearray_oracle_ds_dec_rtn_u64(self.lds_bytes, vaddr=addr, wrap_limit=0x1000)
        # val64 > 0x1000 -> wraps to 0x1000
        self.assertEqual(old_val, val64)
        self.assertEqual(new_val, 0x1000)
        self.assertEqual(struct.unpack_from("<Q", self.lds_bytes, addr)[0], 0x1000)


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_enumeration(self) -> None:
        self.assertEqual(DSOpcode.DS_DEC_RTN_U64.value, "ds_dec_rtn_u64")

    def test_reg_single_and_pair_creation(self) -> None:
        v1 = RegSingle(1, RegType.VGPR)
        s0 = RegSingle(0, RegType.SGPR)
        self.assertEqual(v1.name, "v1")
        self.assertEqual(s0.name, "s0")

        v_pair = RegPair(4, RegType.VGPR)
        self.assertEqual(v_pair.name, "v[4:5]")
        self.assertEqual(v_pair.indices, (4, 5))

    def test_ast_construction_and_disassembly(self) -> None:
        instr_basic = DSDecRtnU64Instruction(
            vdst=RegPair(4),
            vaddr=RegSingle(1),
            vdata0=RegPair(6),
            offset=0,
        )
        self.assertEqual(instr_basic.disasm(), "ds_dec_rtn_u64 v[4:5], v1, v[6:7]")

        instr_offset = DSDecRtnU64Instruction(
            vdst=RegPair(4),
            vaddr=RegSingle(1),
            vdata0=RegPair(6),
            offset=32,
        )
        self.assertEqual(instr_offset.disasm(), "ds_dec_rtn_u64 v[4:5], v1, v[6:7] offset:32")

        instr_gds = DSDecRtnU64Instruction(
            vdst=RegPair(4),
            vaddr=RegSingle(1),
            vdata0=RegPair(6),
            offset=16,
            gds=True,
        )
        self.assertEqual(instr_gds.disasm(), "ds_dec_rtn_u64 v[4:5], v1, v[6:7] offset:16 gds")

    def test_ast_immutability_and_equality(self) -> None:
        instr1 = DSDecRtnU64Instruction(
            vdst=RegPair(2),
            vaddr=RegSingle(1),
            vdata0=RegPair(4),
            offset=8,
        )
        instr2 = DSDecRtnU64Instruction(
            vdst=RegPair(2),
            vaddr=RegSingle(1),
            vdata0=RegPair(4),
            offset=8,
        )
        self.assertEqual(instr1, instr2)
        self.assertEqual(hash(instr1), hash(instr2))

    def test_ast_validation_rejects_invalid_types_and_bounds(self) -> None:
        with self.assertRaises(ValueError):
            RegSingle(-1)
        with self.assertRaises(ValueError):
            RegSingle(256)
        with self.assertRaises(ValueError):
            RegPair(255)  # Can't have pair starting at 255 (255:256)

        # vdst must be RegPair
        with self.assertRaises(TypeError):
            DSDecRtnU64Instruction(
                vdst=RegSingle(4),  # type: ignore
                vaddr=RegSingle(1),
                vdata0=RegPair(6),
            )

        # vdata0 must be RegPair
        with self.assertRaises(TypeError):
            DSDecRtnU64Instruction(
                vdst=RegPair(4),
                vaddr=RegSingle(1),
                vdata0=RegSingle(6),  # type: ignore
            )

        # vaddr must be RegSingle
        with self.assertRaises(TypeError):
            DSDecRtnU64Instruction(
                vdst=RegPair(4),
                vaddr=RegPair(0),  # type: ignore
                vdata0=RegPair(6),
            )

        # SGPR operands rejected
        with self.assertRaises(TypeError):
            DSDecRtnU64Instruction(
                vdst=RegPair(4, RegType.SGPR),
                vaddr=RegSingle(1),
                vdata0=RegPair(6),
            )

        # offset bounds
        with self.assertRaises(ValueError):
            DSDecRtnU64Instruction(
                vdst=RegPair(4),
                vaddr=RegSingle(1),
                vdata0=RegPair(6),
                offset=-1,
            )
        with self.assertRaises(ValueError):
            DSDecRtnU64Instruction(
                vdst=RegPair(4),
                vaddr=RegSingle(1),
                vdata0=RegPair(6),
                offset=65536,
            )


class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter for ds_dec_rtn_u64."""

    def setUp(self) -> None:
        self.state = ScalarMachineState(capacity_qwords=1024)
        self.interp = ScalarInterpreter()

    def test_scalar_execute_basic_u64(self) -> None:
        self.state.lds.write_qword(10 * 8, 50)
        self.state.set_vgpr(1, 80)
        self.state.set_vgpr64(RegPair(2), 100)

        instr = DSDecRtnU64Instruction(
            vdst=RegPair(4),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            offset=0,
        )

        old_val, new_val = self.interp.step(instr, self.state)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_val, 49)
        self.assertEqual(self.state.get_vgpr64(RegPair(4)), 50)
        self.assertEqual(self.state.lds.read_qword(10 * 8), 49)

    def test_scalar_execute_wrap_on_zero_u64(self) -> None:
        bound = 0xAAAA_BBBB_CCCC_DDDD
        self.state.lds.write_qword(10 * 8, 0)
        self.state.set_vgpr(1, 80)
        self.state.set_vgpr64(RegPair(2), bound)

        instr = DSDecRtnU64Instruction(
            vdst=RegPair(4),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            offset=0,
        )

        old_val, new_val = self.interp.step(instr, self.state)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, bound)
        self.assertEqual(self.state.get_vgpr64(RegPair(4)), 0)
        self.assertEqual(self.state.lds.read_qword(10 * 8), bound)

    def test_scalar_determinism_across_identical_runs_u64(self) -> None:
        state1 = ScalarMachineState(capacity_qwords=1024)
        state2 = ScalarMachineState(capacity_qwords=1024)
        interp1 = ScalarInterpreter()
        interp2 = ScalarInterpreter()

        val = 0x0123_4567_89AB_CDEF
        for s in (state1, state2):
            s.lds.write_qword(5 * 8, val)
            s.set_vgpr(1, 40)
            s.set_vgpr64(RegPair(2), val)

        instr = DSDecRtnU64Instruction(
            vdst=RegPair(4),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            offset=0,
        )

        res1 = interp1.step(instr, state1)
        res2 = interp2.step(instr, state2)

        self.assertEqual(res1, res2)
        self.assertEqual(state1.rf.regs, state2.rf.regs)
        self.assertEqual(state1.lds.memory, state2.lds.memory)

    def test_scalar_loop_countdown_u64(self) -> None:
        bound = 3
        idx = 8
        self.state.lds.write_qword(idx * 8, bound)
        self.state.set_vgpr(1, idx * 8)
        self.state.set_vgpr64(RegPair(2), bound)

        instr = DSDecRtnU64Instruction(
            vdst=RegPair(4),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            offset=0,
        )

        expected_steps = [(3, 2), (2, 1), (1, 0), (0, 3), (3, 2)]
        for exp_old, exp_new in expected_steps:
            old_val, new_val = self.interp.step(instr, self.state)
            self.assertEqual(old_val, exp_old)
            self.assertEqual(new_val, exp_new)
            self.assertEqual(self.state.get_vgpr64(RegPair(4)), exp_old)
            self.assertEqual(self.state.lds.read_qword(idx * 8), exp_new)

    def test_scalar_untouched_state_preserved_u64(self) -> None:
        self.state.set_vgpr(0, 0x11111111)
        self.state.set_vgpr(1, 80)
        self.state.set_vgpr64(RegPair(2), 50)
        self.state.set_vgpr(6, 0x66666666)
        self.state.lds.write_qword(9 * 8, 0xAAAA_AAAA_AAAA_AAAA)
        self.state.lds.write_qword(10 * 8, 10)
        self.state.lds.write_qword(11 * 8, 0xBBBB_BBBB_BBBB_BBBB)

        instr = DSDecRtnU64Instruction(
            vdst=RegPair(4),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            offset=0,
        )

        self.interp.step(instr, self.state)

        # Target register pair v[4:5] modified
        self.assertEqual(self.state.get_vgpr64(RegPair(4)), 10)
        # Adjacent registers preserved
        self.assertEqual(self.state.get_vgpr(0), 0x11111111)
        self.assertEqual(self.state.get_vgpr(1), 80)
        self.assertEqual(self.state.get_vgpr64(RegPair(2)), 50)
        self.assertEqual(self.state.get_vgpr(6), 0x66666666)
        # Adjacent memory preserved
        self.assertEqual(self.state.lds.read_qword(9 * 8), 0xAAAA_AAAA_AAAA_AAAA)
        self.assertEqual(self.state.lds.read_qword(10 * 8), 9)
        self.assertEqual(self.state.lds.read_qword(11 * 8), 0xBBBB_BBBB_BBBB_BBBB)


class TestSIMDMultiLaneDivergenceAndExecMasking(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking for ds_dec_rtn_u64."""

    def test_simd_wave32_full_exec_independent_addresses_u64(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=MASK_32, capacity_qwords=1024)
        simd = SIMDInterpreter()

        for lane in range(32):
            addr = lane * 8
            state.lds.write_qword(addr, 10 + lane)
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr64(lane, RegPair(2), 100)
            state.set_lane_vgpr64(lane, RegPair(4), 0)

        instr = DSDecRtnU64Instruction(
            vdst=RegPair(4),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            offset=0,
        )

        results = simd.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertEqual(results[lane], (10 + lane, 9 + lane))
            self.assertEqual(state.get_lane_vgpr64(lane, RegPair(4)), 10 + lane)
            self.assertEqual(state.lds.read_qword(lane * 8), 9 + lane)

    def test_simd_wave64_full_exec_independent_addresses_u64(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=MASK_64, capacity_qwords=1024)
        simd = SIMDInterpreter()

        for lane in range(64):
            addr = lane * 8
            state.lds.write_qword(addr, 100 + lane)
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr64(lane, RegPair(2), 200)

        instr = DSDecRtnU64Instruction(
            vdst=RegPair(4),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            offset=0,
        )

        results = simd.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            self.assertEqual(results[lane], (100 + lane, 99 + lane))
            self.assertEqual(state.get_lane_vgpr64(lane, RegPair(4)), 100 + lane)
            self.assertEqual(state.lds.read_qword(lane * 8), 99 + lane)

    def test_simd_wave32_divergent_checkerboard_mask_u64(self) -> None:
        mask = 0x55555555
        state = SIMDState(num_lanes=32, exec_mask=mask, capacity_qwords=1024)
        simd = SIMDInterpreter()
        sentinel = 0xDEAD_BEEF_CAFE_BABE

        for lane in range(32):
            addr = lane * 8
            state.lds.write_qword(addr, 20 + lane)
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr64(lane, RegPair(2), 50)
            state.set_lane_vgpr64(lane, RegPair(4), sentinel)

        instr = DSDecRtnU64Instruction(
            vdst=RegPair(4),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            offset=0,
        )

        results = simd.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], (20 + lane, 19 + lane))
                self.assertEqual(state.get_lane_vgpr64(lane, RegPair(4)), 20 + lane)
                self.assertEqual(state.lds.read_qword(lane * 8), 19 + lane)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_vgpr64(lane, RegPair(4)), sentinel)
                self.assertEqual(state.lds.read_qword(lane * 8), 20 + lane)

    def test_simd_all_lanes_inactive_u64(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0, capacity_qwords=1024)
        simd = SIMDInterpreter()

        for lane in range(32):
            state.lds.write_qword(lane * 8, 42)
            state.set_lane_vgpr(lane, 1, lane * 8)
            state.set_lane_vgpr64(lane, RegPair(2), 50)
            state.set_lane_vgpr64(lane, RegPair(4), 0x1234_5678_9ABC_DEF0)

        instr = DSDecRtnU64Instruction(
            vdst=RegPair(4),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            offset=0,
        )

        results = simd.step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(state.get_lane_vgpr64(lane, RegPair(4)), 0x1234_5678_9ABC_DEF0)
            self.assertEqual(state.lds.read_qword(lane * 8), 42)

    def test_simd_single_active_lane_u64(self) -> None:
        target_lane = 5
        state = SIMDState(num_lanes=32, exec_mask=(1 << target_lane), capacity_qwords=1024)
        simd = SIMDInterpreter()

        for lane in range(32):
            state.lds.write_qword(lane * 8, 10)
            state.set_lane_vgpr(lane, 1, lane * 8)
            state.set_lane_vgpr64(lane, RegPair(2), 50)
            state.set_lane_vgpr64(lane, RegPair(6), 0)

        instr = DSDecRtnU64Instruction(
            vdst=RegPair(6),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            offset=0,
        )

        results = simd.step(instr, state)
        for lane in range(32):
            if lane == target_lane:
                self.assertEqual(results[lane], (10, 9))
                self.assertEqual(state.get_lane_vgpr64(lane, RegPair(6)), 10)
                self.assertEqual(state.lds.read_qword(lane * 8), 9)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_vgpr64(lane, RegPair(6)), 0)
                self.assertEqual(state.lds.read_qword(lane * 8), 10)

    def test_simd_multi_lane_contention_same_address_u64(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0b1111, capacity_qwords=1024)
        simd = SIMDInterpreter()

        target_qword = 16
        target_addr = target_qword * 8
        bound = 3
        state.lds.write_qword(target_addr, bound)

        for lane in range(4):
            state.set_lane_vgpr(lane, 1, target_addr)
            state.set_lane_vgpr64(lane, RegPair(2), bound)
            state.set_lane_vgpr64(lane, RegPair(4), 0)

        instr = DSDecRtnU64Instruction(
            vdst=RegPair(4),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            offset=0,
        )

        results = simd.step(instr, state)
        self.assertEqual(results[0], (3, 2))
        self.assertEqual(state.get_lane_vgpr64(0, RegPair(4)), 3)

        self.assertEqual(results[1], (2, 1))
        self.assertEqual(state.get_lane_vgpr64(1, RegPair(4)), 2)

        self.assertEqual(results[2], (1, 0))
        self.assertEqual(state.get_lane_vgpr64(2, RegPair(4)), 1)

        self.assertEqual(results[3], (0, 3))
        self.assertEqual(state.get_lane_vgpr64(3, RegPair(4)), 0)

        self.assertEqual(state.lds.read_qword(target_addr), 3)


if __name__ == "__main__":
    unittest.main()
