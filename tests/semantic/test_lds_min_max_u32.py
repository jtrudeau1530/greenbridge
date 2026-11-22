"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_min_max_u32.

Models unsigned 32-bit integer min/max and AMD RDNA2 / GFX10 Local Data Share (LDS)
architectural semantics:
  - ds_min_u32: Atomic unsigned 32-bit minimum on LDS storage.
  - ds_max_u32: Atomic unsigned 32-bit maximum on LDS storage.
  - ds_min_rtn_u32: Atomic unsigned 32-bit minimum returning previous value to destination VGPR.
  - ds_max_rtn_u32: Atomic unsigned 32-bit maximum returning previous value to destination VGPR.

RDNA2 LDS / Unsigned 32-bit Integer ISA Rules:
  1. Local Data Share Architecture:
     LDS is an on-chip scratchpad memory shared across all threads in a workgroup.
     Addressing is byte-addressed but dword-aligned in hardware (low 2 bits dropped).
     Out-of-bounds addresses wrap within the allocated power-of-two LDS slice.
     LDS is organized into 32 memory banks (bank = dword_index % 32).
  2. Unsigned Integer Ordering:
     Operands are treated as unsigned 32-bit integers in [0, 2^32 - 1].
     0 is strictly less than 0xFFFFFFFF (in contrast to signed i32 where -1 < 0).
  3. Atomic Operations & Return Semantics:
     Non-returning variants (ds_min_u32, ds_max_u32) discard the previous memory content.
     Returning variants (ds_min_rtn_u32, ds_max_rtn_u32) atomically fetch the previous
     32-bit value and write it into the destination VGPR.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import os
import sys
import unittest
from typing import Tuple, List, Optional, Dict, Any

MASK32: int = 0xFFFFFFFF
UINT32_MIN: int = 0x00000000
UINT32_MAX: int = 0xFFFFFFFF
UINT32_MID: int = 0x80000000
UINT32_HIGH_BIT: int = 0x80000000

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB
DEFAULT_LDS_NUM_DWORDS: int = 16384
NUM_LDS_BANKS: int = 32


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def oracle_min_u32(a: int, b: int) -> int:
    """Reference unsigned 32-bit minimum."""
    ua = a & MASK32
    ub = b & MASK32
    return ua if ua < ub else ub


def oracle_max_u32(a: int, b: int) -> int:
    """Reference unsigned 32-bit maximum."""
    ua = a & MASK32
    ub = b & MASK32
    return ua if ua > ub else ub


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
        effective_byte = (byte_addr + offset) & MASK32
        return (effective_byte >> 2) & (self.num_dwords - 1)

    def get_bank(self, dword_idx: int) -> int:
        """Compute memory bank index for the given dword (0 to 31)."""
        return dword_idx % NUM_LDS_BANKS

    def read_dword(self, dword_idx: int) -> int:
        """Read a 32-bit dword from LDS."""
        return self.dwords[dword_idx & (self.num_dwords - 1)]

    def write_dword(self, dword_idx: int, val: int) -> None:
        """Write a 32-bit dword into LDS."""
        self.dwords[dword_idx & (self.num_dwords - 1)] = val & MASK32

    def atomic_min_u32(self, byte_addr: int, val: int, offset: int = 0) -> int:
        """Execute atomic unsigned minimum at (byte_addr + offset), returning previous value."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        new_val = oracle_min_u32(old_val, val)
        self.dwords[idx] = new_val
        return old_val

    def atomic_max_u32(self, byte_addr: int, val: int, offset: int = 0) -> int:
        """Execute atomic unsigned maximum at (byte_addr + offset), returning previous value."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        new_val = oracle_max_u32(old_val, val)
        self.dwords[idx] = new_val
        return old_val


class TestLDSMinMaxU32ModelAndOracle(unittest.TestCase):
    """Unit tests for unsigned 32-bit integer model and reference LDS oracle."""

    def test_oracle_min_u32_standard(self) -> None:
        self.assertEqual(oracle_min_u32(10, 20), 10)
        self.assertEqual(oracle_min_u32(20, 10), 10)
        self.assertEqual(oracle_min_u32(500, 500), 500)

    def test_oracle_max_u32_standard(self) -> None:
        self.assertEqual(oracle_max_u32(10, 20), 20)
        self.assertEqual(oracle_max_u32(20, 10), 20)
        self.assertEqual(oracle_max_u32(500, 500), 500)

    def test_oracle_unsigned_extremes(self) -> None:
        # In unsigned arithmetic, 0 < 0xFFFFFFFF
        self.assertEqual(oracle_min_u32(UINT32_MIN, UINT32_MAX), UINT32_MIN)
        self.assertEqual(oracle_max_u32(UINT32_MIN, UINT32_MAX), UINT32_MAX)
        self.assertEqual(oracle_min_u32(UINT32_MID, UINT32_MAX), UINT32_MID)
        self.assertEqual(oracle_max_u32(UINT32_MID, UINT32_MAX), UINT32_MAX)

    def test_lds_memory_model(self) -> None:
        lds = LDSMemory(65536)
        self.assertEqual(lds.num_dwords, 16384)

        # Byte address to dword index drops low 2 bits
        self.assertEqual(lds.get_dword_index(0x0004), 1)
        self.assertEqual(lds.get_dword_index(0x0007), 1)
        self.assertEqual(lds.get_dword_index(0x0008, offset=4), 3)

        # Address wrapping
        self.assertEqual(lds.get_dword_index(65536 + 24), 6)

        # Bank calculation
        self.assertEqual(lds.get_bank(0), 0)
        self.assertEqual(lds.get_bank(31), 31)
        self.assertEqual(lds.get_bank(32), 0)

        # Atomic min u32
        lds.write_dword(10, 100)
        old_val = lds.atomic_min_u32(10 * 4, 25)
        self.assertEqual(old_val, 100)
        self.assertEqual(lds.read_dword(10), 25)

        # Atomic max u32
        old_val2 = lds.atomic_max_u32(10 * 4, 75)
        self.assertEqual(old_val2, 25)
        self.assertEqual(lds.read_dword(10), 75)


class DSOpcode(str, enum.Enum):
    """Opcode enumerations for unsigned 32-bit LDS min/max operations."""
    DS_MIN_U32 = "ds_min_u32"
    DS_MAX_U32 = "ds_max_u32"
    DS_MIN_RTN_U32 = "ds_min_rtn_u32"
    DS_MAX_RTN_U32 = "ds_max_rtn_u32"

    @property
    def is_return(self) -> bool:
        """True if instruction returns prior LDS value to destination VGPR."""
        return self in (DSOpcode.DS_MIN_RTN_U32, DSOpcode.DS_MAX_RTN_U32)

    @property
    def is_min(self) -> bool:
        """True if operation is an unsigned minimum."""
        return self in (DSOpcode.DS_MIN_U32, DSOpcode.DS_MIN_RTN_U32)

    @property
    def is_max(self) -> bool:
        """True if operation is an unsigned maximum."""
        return self in (DSOpcode.DS_MAX_U32, DSOpcode.DS_MAX_RTN_U32)


@dataclasses.dataclass(frozen=True)
class VGPR:
    """Vector General Purpose Register operand v[0] .. v[255]."""
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")

    def __str__(self) -> str:
        return f"v{self.index}"


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
class DSMinMaxU32Instruction:
    """Symbolic AST node representing an RDNA2 DS min/max u32 instruction."""
    opcode: DSOpcode
    vaddr: VGPR
    vdata: VGPR
    vdst: Optional[VGPR] = None
    offset: OffsetImmediate = OffsetImmediate(0)

    def __post_init__(self) -> None:
        if self.opcode.is_return:
            if self.vdst is None:
                raise ValueError(f"Returning opcode {self.opcode.value} requires destination vdst")
        else:
            if self.vdst is not None:
                raise ValueError(f"Non-returning opcode {self.opcode.value} must not have destination vdst")

    def disasm(self) -> str:
        """Format symbolic instruction into canonical RDNA2 assembly text."""
        parts = [self.opcode.value]
        if self.opcode.is_return:
            operands = f"{self.vdst}, {self.vaddr}, {self.vdata}"
        else:
            operands = f"{self.vaddr}, {self.vdata}"
        parts.append(operands)
        if self.offset.value != 0:
            parts.append(str(self.offset))
        return " ".join(parts)

    @classmethod
    def parse(cls, text: str) -> DSMinMaxU32Instruction:
        """Parse canonical RDNA2 assembly string into a DSMinMaxU32Instruction AST."""
        tokens = text.strip().split(None, 1)
        if not tokens:
            raise ValueError("Empty assembly string")
        op_str = tokens[0]
        try:
            opcode = DSOpcode(op_str)
        except ValueError:
            raise ValueError(f"Unknown or unsupported DS opcode: {op_str}")

        if len(tokens) == 1:
            raise ValueError(f"Missing operands for {op_str}")

        rest = tokens[1].strip()
        offset_val = 0
        if "offset:" in rest:
            m = rest.split("offset:")
            rest = m[0].strip()
            offset_tok = m[1].split()[0].rstrip(",")
            offset_val = int(offset_tok, 0)

        reg_tokens = [tok.strip().rstrip(",") for tok in rest.split(",") if tok.strip()]

        def parse_vgpr(s: str) -> VGPR:
            s = s.strip()
            if not s.startswith("v") or not s[1:].isdigit():
                raise ValueError(f"Invalid VGPR syntax: {s}")
            return VGPR(int(s[1:]))

        if opcode.is_return:
            if len(reg_tokens) != 3:
                raise ValueError(f"{opcode.value} expects 3 registers (vdst, vaddr, vdata), got {reg_tokens}")
            vdst = parse_vgpr(reg_tokens[0])
            vaddr = parse_vgpr(reg_tokens[1])
            vdata = parse_vgpr(reg_tokens[2])
            return cls(opcode=opcode, vaddr=vaddr, vdata=vdata, vdst=vdst, offset=OffsetImmediate(offset_val))
        else:
            if len(reg_tokens) != 2:
                raise ValueError(f"{opcode.value} expects 2 registers (vaddr, vdata), got {reg_tokens}")
            vaddr = parse_vgpr(reg_tokens[0])
            vdata = parse_vgpr(reg_tokens[1])
            return cls(opcode=opcode, vaddr=vaddr, vdata=vdata, vdst=None, offset=OffsetImmediate(offset_val))


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions, registers, and assembly parsing."""

    def test_opcode_properties(self) -> None:
        self.assertFalse(DSOpcode.DS_MIN_U32.is_return)
        self.assertTrue(DSOpcode.DS_MIN_U32.is_min)
        self.assertFalse(DSOpcode.DS_MIN_U32.is_max)

        self.assertFalse(DSOpcode.DS_MAX_U32.is_return)
        self.assertFalse(DSOpcode.DS_MAX_U32.is_min)
        self.assertTrue(DSOpcode.DS_MAX_U32.is_max)

        self.assertTrue(DSOpcode.DS_MIN_RTN_U32.is_return)
        self.assertTrue(DSOpcode.DS_MIN_RTN_U32.is_min)
        self.assertFalse(DSOpcode.DS_MIN_RTN_U32.is_max)

        self.assertTrue(DSOpcode.DS_MAX_RTN_U32.is_return)
        self.assertFalse(DSOpcode.DS_MAX_RTN_U32.is_min)
        self.assertTrue(DSOpcode.DS_MAX_RTN_U32.is_max)

    def test_vgpr_validation(self) -> None:
        v0 = VGPR(0)
        v255 = VGPR(255)
        self.assertEqual(str(v0), "v0")
        self.assertEqual(str(v255), "v255")

        with self.assertRaises(ValueError):
            VGPR(-1)
        with self.assertRaises(ValueError):
            VGPR(256)

    def test_offset_immediate_validation(self) -> None:
        off0 = OffsetImmediate(0)
        off64 = OffsetImmediate(64)
        off_max = OffsetImmediate(65535)
        self.assertEqual(str(off0), "")
        self.assertEqual(str(off64), "offset:64")
        self.assertEqual(str(off_max), "offset:65535")

        with self.assertRaises(ValueError):
            OffsetImmediate(-1)
        with self.assertRaises(ValueError):
            OffsetImmediate(65536)

    def test_instruction_post_init_validation(self) -> None:
        DSMinMaxU32Instruction(opcode=DSOpcode.DS_MIN_U32, vaddr=VGPR(0), vdata=VGPR(1))
        with self.assertRaises(ValueError):
            DSMinMaxU32Instruction(opcode=DSOpcode.DS_MIN_U32, vaddr=VGPR(0), vdata=VGPR(1), vdst=VGPR(2))

        DSMinMaxU32Instruction(opcode=DSOpcode.DS_MIN_RTN_U32, vaddr=VGPR(0), vdata=VGPR(1), vdst=VGPR(2))
        with self.assertRaises(ValueError):
            DSMinMaxU32Instruction(opcode=DSOpcode.DS_MIN_RTN_U32, vaddr=VGPR(0), vdata=VGPR(1), vdst=None)

    def test_disasm_and_parse_roundtrip(self) -> None:
        test_cases = [
            DSMinMaxU32Instruction(DSOpcode.DS_MIN_U32, VGPR(2), VGPR(3)),
            DSMinMaxU32Instruction(DSOpcode.DS_MIN_U32, VGPR(2), VGPR(3), offset=OffsetImmediate(16)),
            DSMinMaxU32Instruction(DSOpcode.DS_MAX_U32, VGPR(10), VGPR(11), offset=OffsetImmediate(128)),
            DSMinMaxU32Instruction(DSOpcode.DS_MIN_RTN_U32, VGPR(4), VGPR(5), vdst=VGPR(0)),
            DSMinMaxU32Instruction(DSOpcode.DS_MAX_RTN_U32, VGPR(8), VGPR(9), vdst=VGPR(7), offset=OffsetImmediate(256)),
        ]
        for inst in test_cases:
            with self.subTest(inst=inst):
                asm = inst.disasm()
                parsed = DSMinMaxU32Instruction.parse(asm)
                self.assertEqual(inst, parsed)


if __name__ == "__main__":
    unittest.main()
