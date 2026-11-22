from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_max_rtn_u64.

RDNA2 / GFX10 Local Data Share (LDS) Atomic Unsigned 64-bit Integer Maximum with Return:
Atomically compares an unsigned 64-bit integer operand with the 64-bit value in LDS memory,
computes the unsigned maximum, updates LDS with the maximum, and returns the original
64-bit value from LDS memory to the destination VGPR pair.

Syntax:
    ds_max_rtn_u64 v[dst_lo:dst_hi], vaddr, v[data_lo:data_hi] [offset:offset16]

Architectural Semantics:
    effective_byte_address = (vaddr + (offset16 & 0xFFFF)) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_qwords - 1)
    old_val = LDS_QWORD[qword_index]
    new_val = max(old_val, vdata_u64)
    LDS_QWORD[qword_index] = new_val
    vdst_lo = old_val & 0xFFFFFFFF
    vdst_hi = (old_val >> 32) & 0xFFFFFFFF

Unsigned 64-bit Integer Properties:
    - Range: [0, 2^64 - 1] ([0, 18446744073709551615])
    - UINT64_MIN = 0
    - UINT64_MAX = 0xFFFFFFFFFFFFFFFF
    - Unsigned comparison: 0xFFFFFFFFFFFFFFFF > 0, 0x8000000000000000 > 0x7FFFFFFFFFFFFFFF
"""

import math
import re
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_SIZE_QWORDS: int = 8192         # 8,192 64-bit qwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask

UINT64_MIN: int = 0
UINT64_MAX: int = 0xFFFF_FFFF_FFFF_FFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword [0, 2^64 - 1]."""
    return val & MASK64


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = LDS_SIZE_QWORDS) -> int:
    """Convert an effective byte address to an aligned LDS 64-bit qword index.

    LDS 64-bit qwords are 8-byte aligned; the low 3 bits are ignored by hardware.
    The index wraps modulo the power-of-two qword capacity.
    """
    return (eff_addr >> 3) & (capacity_qwords - 1)


def calc_lds_qword_banks(eff_addr: int) -> Tuple[int, int]:
    """Determine the two LDS memory bank indices (0..31) accessed by a 64-bit qword."""
    dword_idx = (eff_addr >> 2) & (LDS_SIZE_DWORDS - 1)
    bank_lo = dword_idx % LDS_BANK_COUNT
    bank_hi = (dword_idx + 1) % LDS_BANK_COUNT
    return bank_lo, bank_hi


def ref_ds_max_rtn_u64(
    current_mem: int,
    data_val: int,
) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_max_rtn_u64.

    Computes:
        old_val = current_mem & MASK64
        new_val = max(old_val, data_val & MASK64)

    Returns:
        (old_val, new_val)
    """
    old_val = u64(current_mem)
    data_u = u64(data_val)
    new_val = max(old_val, data_u)
    return old_val, new_val


class LDSMemory64:
    """Architectural memory model representing LDS with 64-bit atomic access support."""

    def __init__(self, size_bytes: int = LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"size_bytes must be a positive power of two, got {size_bytes}")
        self.size_bytes = size_bytes
        self.num_dwords = size_bytes // 4
        self.num_qwords = size_bytes // 8
        self.dwords: List[int] = [0] * self.num_dwords

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit unsigned integer from LDS at 8-byte aligned address."""
        dword_idx = ((eff_addr >> 3) << 1) & (self.num_dwords - 1)
        lo = self.dwords[dword_idx]
        hi = self.dwords[dword_idx + 1]
        return lo | (hi << 32)

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit unsigned integer to LDS at 8-byte aligned address."""
        dword_idx = ((eff_addr >> 3) << 1) & (self.num_dwords - 1)
        v = u64(val)
        self.dwords[dword_idx] = v & MASK32
        self.dwords[dword_idx + 1] = (v >> 32) & MASK32

    def atomic_max_rtn_u64(
        self,
        vaddr: int,
        vdata: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic unsigned 64-bit maximum with return on LDS.

        Returns:
            (old_val, new_val)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        curr = self.read_qword(eff_addr)
        old_val, new_val = ref_ds_max_rtn_u64(curr, vdata)
        self.write_qword(eff_addr, new_val)
        return old_val, new_val


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) 64-bit atomic opcodes."""
    DS_MAX_RTN_U64 = "ds_max_rtn_u64"


class RegType(Enum):
    """Register file classes."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegisterOperand:
    """Typed symbolic register operand representation (scalar or contiguous multi-register)."""
    index: int
    reg_type: RegType = RegType.VGPR
    count: int = 1

    def __post_init__(self) -> None:
        max_limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < max_limit):
            raise ValueError(f"Register index out of range for {self.reg_type.name}: {self.index}")
        if self.count < 1 or (self.index + self.count > max_limit):
            raise ValueError(
                f"Contiguous register sequence exceeds {self.reg_type.name} limits: "
                f"{self.index} + {self.count} > {max_limit}"
            )

    def __str__(self) -> str:
        if self.count == 1:
            return f"{self.reg_type.value}{self.index}"
        return f"{self.reg_type.value}[{self.index}:{self.index + self.count - 1}]"


@dataclass(frozen=True)
class DSMaxRtnU64Instruction:
    """Symbolic AST node representing a ds_max_rtn_u64 instruction.

    Format:
        ds_max_rtn_u64 v[dst_lo:dst_hi], vaddr, v[data_lo:data_hi] [offset:offset16]
    """
    vdst: RegisterOperand
    vaddr: RegisterOperand
    vdata0: RegisterOperand
    offset: int = 0
    opcode: DSOpcode = DSOpcode.DS_MAX_RTN_U64

    def __post_init__(self) -> None:
        if self.vdst.reg_type != RegType.VGPR or self.vdst.count != 2:
            raise ValueError(f"vdst must be a 2-VGPR sequence (e.g. v[0:1]), got {self.vdst}")
        if self.vaddr.reg_type != RegType.VGPR or self.vaddr.count != 1:
            raise ValueError(f"vaddr must be a single VGPR (e.g. v2), got {self.vaddr}")
        if self.vdata0.reg_type != RegType.VGPR or self.vdata0.count != 2:
            raise ValueError(f"vdata0 must be a 2-VGPR sequence (e.g. v[3:4]), got {self.vdata0}")
        if not (0 <= self.offset <= 65535):
            raise ValueError(f"offset out of range [0, 65535]: {self.offset}")

    def to_assembly(self) -> str:
        """Serialize AST node to standard assembly string."""
        asm = f"{self.opcode.value} {self.vdst}, {self.vaddr}, {self.vdata0}"
        if self.offset != 0:
            asm += f" offset:{self.offset}"
        return asm

    @classmethod
    def parse(cls, asm: str) -> DSMaxRtnU64Instruction:
        """Parse an assembly string into a DSMaxRtnU64Instruction AST node."""
        cleaned = re.sub(r"[;#].*$", "", asm).strip()
        if not cleaned:
            raise ValueError("Empty assembly string")

        tokens = cleaned.split()
        if not tokens:
            raise ValueError("Empty instruction")

        op_str = tokens[0].lower()
        if op_str != DSOpcode.DS_MAX_RTN_U64.value:
            raise ValueError(f"Unsupported opcode '{tokens[0]}', expected '{DSOpcode.DS_MAX_RTN_U64.value}'")

        rest = " ".join(tokens[1:])
        parts = [p.strip() for p in rest.split(",") if p.strip()]

        if len(parts) < 3:
            raise ValueError(f"Insufficient operands for {op_str}: expected at least 3, got {len(parts)}")

        vdst_str = parts[0]
        vaddr_str = parts[1]

        vdata_and_mods = parts[2].split()
        vdata_str = vdata_and_mods[0]

        offset_val = 0
        mod_tokens = vdata_and_mods[1:] + parts[3:]
        for m in mod_tokens:
            m_lower = m.lower()
            if m_lower.startswith("offset:"):
                raw_offset = m[len("offset:"):]
                try:
                    offset_val = int(raw_offset, 0)
                except ValueError:
                    raise ValueError(f"Invalid integer offset value: '{raw_offset}'")
            else:
                raise ValueError(f"Unknown modifier or unexpected extra token: '{m}'")

        def parse_vgpr(reg_s: str, expected_count: int) -> RegisterOperand:
            reg_s = reg_s.strip()
            # Match bracketed pair: v[0:1] or V[0:1]
            match_pair = re.match(r"^[vV]\[(\d+):(\d+)\]$", reg_s)
            if match_pair:
                start_idx = int(match_pair.group(1))
                end_idx = int(match_pair.group(2))
                count = end_idx - start_idx + 1
                if count != expected_count:
                    raise ValueError(f"Expected {expected_count} registers, got {count} in '{reg_s}'")
                return RegisterOperand(start_idx, RegType.VGPR, count=count)

            # Match scalar vgpr: v0 or V0
            match_single = re.match(r"^[vV](\d+)$", reg_s)
            if match_single:
                idx = int(match_single.group(1))
                return RegisterOperand(idx, RegType.VGPR, count=expected_count)

            raise ValueError(f"Invalid VGPR operand syntax: '{reg_s}'")

        vdst = parse_vgpr(vdst_str, expected_count=2)
        vaddr = parse_vgpr(vaddr_str, expected_count=1)
        vdata0 = parse_vgpr(vdata_str, expected_count=2)

        return cls(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=offset_val)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarThreadState64:
    """Scalar thread architectural execution state with 64-bit VGPR pair access."""

    def __init__(self, lds: Optional[LDSMemory64] = None) -> None:
        self.vgprs: List[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: List[int] = [0] * DEFAULT_NUM_SGPRS
        self.vcc: int = 0
        self.scc: int = 0
        self.lds: LDSMemory64 = lds if lds is not None else LDSMemory64()

    def get_vgpr_u32(self, idx: int) -> int:
        return self.vgprs[idx]

    def set_vgpr_u32(self, idx: int, val: int) -> None:
        self.vgprs[idx] = u32(val)

    def get_vgpr_u64(self, idx: int) -> int:
        lo = self.vgprs[idx]
        hi = self.vgprs[idx + 1]
        return lo | (hi << 32)

    def set_vgpr_u64(self, idx: int, val: int) -> None:
        v = u64(val)
        self.vgprs[idx] = v & MASK32
        self.vgprs[idx + 1] = (v >> 32) & MASK32


class ScalarDSInterpreter64:
    """Deterministic scalar execution interpreter for 64-bit Data Share (DS) instructions."""

    def execute(self, state: ScalarThreadState64, inst: DSMaxRtnU64Instruction) -> int:
        """Execute ds_max_rtn_u64 on scalar thread state.

        1. Read address from inst.vaddr (single 32-bit VGPR).
        2. Read data from inst.vdata0 (64-bit VGPR pair).
        3. Perform atomic unsigned maximum on LDS at eff_addr = vaddr + offset.
        4. Write original 64-bit value into inst.vdst (64-bit VGPR pair).
        5. Returns the returned old 64-bit memory value.
        """
        vaddr = state.get_vgpr_u32(inst.vaddr.index)
        vdata = state.get_vgpr_u64(inst.vdata0.index)
        old_val, _ = state.lds.atomic_max_rtn_u64(vaddr, vdata, inst.offset)
        state.set_vgpr_u64(inst.vdst.index, old_val)
        return old_val


class TestLDSMaxRtnU64OracleAndModel(unittest.TestCase):
    """Phase 1: Unit tests for unsigned 64-bit integer oracle and LDS architectural model."""

    def test_oracle_unsigned_basic_ordering(self) -> None:
        old_val, new_val = ref_ds_max_rtn_u64(10, 20)
        self.assertEqual(old_val, 10)
        self.assertEqual(new_val, 20)

        old_val2, new_val2 = ref_ds_max_rtn_u64(500, 250)
        self.assertEqual(old_val2, 500)
        self.assertEqual(new_val2, 500)

    def test_oracle_high_bit_unsigned_semantics(self) -> None:
        # 0x8000000000000000 > 0x7FFFFFFFFFFFFFFF in unsigned 64-bit
        val_high = 0x8000000000000000
        val_mid = 0x7FFFFFFFFFFFFFFF
        old_val, new_val = ref_ds_max_rtn_u64(val_mid, val_high)
        self.assertEqual(old_val, val_mid)
        self.assertEqual(new_val, val_high)

    def test_oracle_uint64_extremes(self) -> None:
        # UINT64_MAX vs 0
        old_val, new_val = ref_ds_max_rtn_u64(0, UINT64_MAX)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, UINT64_MAX)

        # UINT64_MAX vs UINT64_MAX - 1
        old_val2, new_val2 = ref_ds_max_rtn_u64(UINT64_MAX - 1, UINT64_MAX)
        self.assertEqual(old_val2, UINT64_MAX - 1)
        self.assertEqual(new_val2, UINT64_MAX)

    def test_oracle_equal_operands(self) -> None:
        val = 0x1122334455667788
        old_val, new_val = ref_ds_max_rtn_u64(val, val)
        self.assertEqual(old_val, val)
        self.assertEqual(new_val, val)

    def test_lds_memory_qword_read_write(self) -> None:
        lds = LDSMemory64(size_bytes=4096)
        val = 0xAABBCCDDEEFF0011
        lds.write_qword(0x20, val)
        self.assertEqual(lds.read_qword(0x20), val)

    def test_lds_qword_alignment_drop_low_bits(self) -> None:
        lds = LDSMemory64(size_bytes=4096)
        val = 0x123456789ABCDEF0
        lds.write_qword(0x40, val)
        # Bytes 0x40 through 0x47 all align to qword starting at 0x40
        for off in range(8):
            self.assertEqual(lds.read_qword(0x40 + off), val)

    def test_lds_memory_capacity_wrapping(self) -> None:
        lds = LDSMemory64(size_bytes=4096)
        val = 0xFEEDFACECAFEBABE
        lds.write_qword(0, val)
        self.assertEqual(lds.read_qword(4096), val)
        self.assertEqual(lds.read_qword(8192), val)

    def test_lds_atomic_max_rtn_u64(self) -> None:
        lds = LDSMemory64(size_bytes=4096)
        addr = 0x80
        lds.write_qword(addr, 1000)

        # max(1000, 5000) -> returns 1000, new is 5000
        old_val, new_val = lds.atomic_max_rtn_u64(addr, 5000)
        self.assertEqual(old_val, 1000)
        self.assertEqual(new_val, 5000)
        self.assertEqual(lds.read_qword(addr), 5000)

        # max(5000, 2000) -> returns 5000, new remains 5000
        old_val2, new_val2 = lds.atomic_max_rtn_u64(addr, 2000)
        self.assertEqual(old_val2, 5000)
        self.assertEqual(new_val2, 5000)
        self.assertEqual(lds.read_qword(addr), 5000)

    def test_lds_qword_bank_distribution(self) -> None:
        # A 64-bit access spans two consecutive dword banks
        b_lo, b_hi = calc_lds_qword_banks(0)
        self.assertEqual((b_lo, b_hi), (0, 1))

        b_lo2, b_hi2 = calc_lds_qword_banks(124)  # dword 31
        self.assertEqual((b_lo2, b_hi2), (31, 0))  # wraps around 32 banks


class TestDSMaxRtnU64AST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_operand_scalar_and_pair(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR, count=1)
        self.assertEqual(str(v0), "v0")
        self.assertEqual(v0.count, 1)

        vpair = RegisterOperand(0, RegType.VGPR, count=2)
        self.assertEqual(str(vpair), "v[0:1]")
        self.assertEqual(vpair.count, 2)

    def test_register_operand_bounds(self) -> None:
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(255, RegType.VGPR, count=2)  # 255 + 2 > 256

    def test_instruction_formatting(self) -> None:
        vdst = RegisterOperand(0, RegType.VGPR, count=2)
        vaddr = RegisterOperand(2, RegType.VGPR, count=1)
        vdata = RegisterOperand(4, RegType.VGPR, count=2)

        inst1 = DSMaxRtnU64Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata)
        self.assertEqual(inst1.to_assembly(), "ds_max_rtn_u64 v[0:1], v2, v[4:5]")

        inst2 = DSMaxRtnU64Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata, offset=32)
        self.assertEqual(inst2.to_assembly(), "ds_max_rtn_u64 v[0:1], v2, v[4:5] offset:32")

    def test_instruction_parsing_valid(self) -> None:
        inst1 = DSMaxRtnU64Instruction.parse("ds_max_rtn_u64 v[0:1], v2, v[4:5]")
        self.assertEqual(inst1.vdst.index, 0)
        self.assertEqual(inst1.vdst.count, 2)
        self.assertEqual(inst1.vaddr.index, 2)
        self.assertEqual(inst1.vdata0.index, 4)
        self.assertEqual(inst1.vdata0.count, 2)
        self.assertEqual(inst1.offset, 0)

        inst2 = DSMaxRtnU64Instruction.parse("ds_max_rtn_u64 v[10:11], v5, v[12:13] offset:64")
        self.assertEqual(inst2.vdst.index, 10)
        self.assertEqual(inst2.offset, 64)

        # Scalar syntax promotion to pair: v0 -> v[0:1]
        inst3 = DSMaxRtnU64Instruction.parse("ds_max_rtn_u64 v0, v2, v4 offset:0x20")
        self.assertEqual(inst3.vdst.index, 0)
        self.assertEqual(inst3.vdst.count, 2)
        self.assertEqual(inst3.vdata0.index, 4)
        self.assertEqual(inst3.vdata0.count, 2)
        self.assertEqual(inst3.offset, 32)

    def test_instruction_roundtrip(self) -> None:
        vdst = RegisterOperand(6, RegType.VGPR, count=2)
        vaddr = RegisterOperand(8, RegType.VGPR, count=1)
        vdata = RegisterOperand(10, RegType.VGPR, count=2)
        inst = DSMaxRtnU64Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata, offset=128)
        parsed = DSMaxRtnU64Instruction.parse(inst.to_assembly())
        self.assertEqual(inst, parsed)


class TestDSMaxRtnU64ScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSMemory64(size_bytes=4096)
        self.state = ScalarThreadState64(lds=self.lds)
        self.interp = ScalarDSInterpreter64()

    def test_scalar_execution_basic_update(self) -> None:
        addr = 0x20
        self.lds.write_qword(addr, 0x100000000)
        self.state.set_vgpr_u32(2, addr)                    # v2 is vaddr
        self.state.set_vgpr_u64(4, 0x200000000)             # v[4:5] is vdata0
        self.state.set_vgpr_u64(0, 0xDEADBEEFCAFEBABE)      # v[0:1] is vdst initial

        inst = DSMaxRtnU64Instruction(
            vdst=RegisterOperand(0, RegType.VGPR, count=2),
            vaddr=RegisterOperand(2, RegType.VGPR, count=1),
            vdata0=RegisterOperand(4, RegType.VGPR, count=2),
        )

        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, 0x100000000)
        self.assertEqual(self.state.get_vgpr_u64(0), 0x100000000)
        self.assertEqual(self.lds.read_qword(addr), 0x200000000)

    def test_scalar_execution_no_update(self) -> None:
        addr = 0x40
        self.lds.write_qword(addr, 0x500000000)
        self.state.set_vgpr_u32(2, addr)
        self.state.set_vgpr_u64(4, 0x300000000)

        inst = DSMaxRtnU64Instruction(
            vdst=RegisterOperand(0, RegType.VGPR, count=2),
            vaddr=RegisterOperand(2, RegType.VGPR, count=1),
            vdata0=RegisterOperand(4, RegType.VGPR, count=2),
        )

        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, 0x500000000)
        self.assertEqual(self.state.get_vgpr_u64(0), 0x500000000)
        self.assertEqual(self.lds.read_qword(addr), 0x500000000)

    def test_scalar_execution_with_offset(self) -> None:
        base_addr = 0x20
        offset = 0x18  # 24 bytes
        eff_addr = base_addr + offset  # 0x38 (aligned to 8 bytes)
        self.lds.write_qword(eff_addr, 0x1111222233334444)
        self.state.set_vgpr_u32(2, base_addr)
        self.state.set_vgpr_u64(4, 0x9999888877776666)

        inst = DSMaxRtnU64Instruction(
            vdst=RegisterOperand(0, RegType.VGPR, count=2),
            vaddr=RegisterOperand(2, RegType.VGPR, count=1),
            vdata0=RegisterOperand(4, RegType.VGPR, count=2),
            offset=offset,
        )

        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, 0x1111222233334444)
        self.assertEqual(self.state.get_vgpr_u64(0), 0x1111222233334444)
        self.assertEqual(self.lds.read_qword(eff_addr), 0x9999888877776666)
        self.assertEqual(self.lds.read_qword(base_addr), 0)

    def test_scalar_execution_sequential_running_maximum(self) -> None:
        addr = 0x80
        self.lds.write_qword(addr, 100)
        self.state.set_vgpr_u32(2, addr)

        inst = DSMaxRtnU64Instruction(
            vdst=RegisterOperand(0, RegType.VGPR, count=2),
            vaddr=RegisterOperand(2, RegType.VGPR, count=1),
            vdata0=RegisterOperand(4, RegType.VGPR, count=2),
        )

        sequence = [
            (200, 100, 200),
            (150, 200, 200),
            (0x100000000, 200, 0x100000000),
            (0x8000000000000000, 0x100000000, 0x8000000000000000),
            (UINT64_MAX, 0x8000000000000000, UINT64_MAX),
            (0x500, UINT64_MAX, UINT64_MAX),
        ]

        for op_val, expected_old, expected_new in sequence:
            self.state.set_vgpr_u64(4, op_val)
            ret = self.interp.execute(self.state, inst)
            self.assertEqual(ret, expected_old)
            self.assertEqual(self.state.get_vgpr_u64(0), expected_old)
            self.assertEqual(self.lds.read_qword(addr), expected_new)

    def test_scalar_unaligned_address_drop_low_3_bits(self) -> None:
        self.lds.write_qword(0x40, 0xCAFE00000000BABE)
        # Low 3 bits of 0x47 are 0b111; should map to aligned 0x40
        self.state.set_vgpr_u32(2, 0x47)
        self.state.set_vgpr_u64(4, 0xCAFE00000000FFFF)

        inst = DSMaxRtnU64Instruction(
            vdst=RegisterOperand(0, RegType.VGPR, count=2),
            vaddr=RegisterOperand(2, RegType.VGPR, count=1),
            vdata0=RegisterOperand(4, RegType.VGPR, count=2),
        )

        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, 0xCAFE00000000BABE)
        self.assertEqual(self.lds.read_qword(0x40), 0xCAFE00000000FFFF)


if __name__ == "__main__":
    unittest.main()
