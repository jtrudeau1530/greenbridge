"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_and_rtn_b32.

Models 32-bit bitwise AND and AMD RDNA2 / GFX10 Local Data Share (LDS)
architectural semantics:
  - ds_and_b32: Atomic 32-bit bitwise AND on LDS storage without return.
  - ds_and_rtn_b32: Atomic 32-bit bitwise AND on LDS storage returning previous value to destination VGPR.

RDNA2 LDS / 32-bit Bitwise Atomic ISA Rules:
  1. Local Data Share Architecture:
     LDS is an on-chip scratchpad memory shared across all threads in a workgroup.
     Addressing is byte-addressed but dword-aligned in hardware (low 2 bits dropped).
     Out-of-bounds addresses wrap within the allocated power-of-two LDS slice.
     LDS is organized into 32 memory banks (bank = dword_index % 32).
  2. Bitwise AND Properties:
     Operands are treated as raw 32-bit bitfields in [0, 2^32 - 1].
     Identity element: 0xFFFFFFFF (a & 0xFFFFFFFF == a).
     Annihilator / Zero element: 0x00000000 (a & 0x00000000 == 0).
     Idempotence: a & a == a.
     Commutativity: a & b == b & a.
     Associativity: (a & b) & c == a & (b & c).
  3. Atomic Operations & Return Semantics:
     Non-returning variant (ds_and_b32) discards the previous memory content.
     Returning variant (ds_and_rtn_b32) atomically fetches the previous 32-bit
     value and writes it into the destination VGPR.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import struct
import unittest
from typing import Dict, List, Optional, Tuple, Set

MASK32: int = 0xFFFFFFFF
ALL_ONES_B32: int = 0xFFFFFFFF
ALL_ZEROS_B32: int = 0x00000000

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB
DEFAULT_LDS_NUM_DWORDS: int = 16384
NUM_LDS_BANKS: int = 32


def b32(val: int) -> int:
    """Normalize integer to unsigned 32-bit bitfield range [0, 2^32 - 1]."""
    return val & MASK32


def oracle_and_b32(a: int, b: int) -> int:
    """Reference 32-bit bitwise AND oracle."""
    return (a & b) & MASK32


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

    def atomic_and_b32(self, byte_addr: int, val: int, offset: int = 0) -> int:
        """Execute atomic bitwise AND at (byte_addr + offset), returning previous value."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        new_val = oracle_and_b32(old_val, val)
        self.dwords[idx] = new_val
        return old_val


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for 32-bit bitwise AND reference oracle and LDS architectural model."""

    def test_oracle_identity_element(self) -> None:
        values = [0x0, 0x1, 0x55555555, 0xAAAAAAAA, 0x12345678, 0xFFFFFFFF]
        for v in values:
            with self.subTest(v=hex(v)):
                self.assertEqual(oracle_and_b32(v, ALL_ONES_B32), v)
                self.assertEqual(oracle_and_b32(ALL_ONES_B32, v), v)

    def test_oracle_zero_element(self) -> None:
        values = [0x0, 0x1, 0x55555555, 0xAAAAAAAA, 0x12345678, 0xFFFFFFFF]
        for v in values:
            with self.subTest(v=hex(v)):
                self.assertEqual(oracle_and_b32(v, ALL_ZEROS_B32), 0)
                self.assertEqual(oracle_and_b32(ALL_ZEROS_B32, v), 0)

    def test_oracle_idempotence(self) -> None:
        values = [0x0, 0x1, 0xDEADBEEF, 0xCAFEBABE, 0xFFFFFFFF]
        for v in values:
            with self.subTest(v=hex(v)):
                self.assertEqual(oracle_and_b32(v, v), v)

    def test_oracle_commutativity(self) -> None:
        pairs = [
            (0x0F0F0F0F, 0xF0F0F0F0),
            (0x12345678, 0x87654321),
            (0xAAAAAAAA, 0x55555555),
            (0x00FF00FF, 0x0F0F0F0F),
        ]
        for a, b in pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(oracle_and_b32(a, b), oracle_and_b32(b, a))

    def test_oracle_associativity(self) -> None:
        triplets = [
            (0xFF00FF00, 0x0F0F0F0F, 0x33333333),
            (0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0),
            (0x12345678, 0x9ABCDEF0, 0x55555555),
        ]
        for a, b, c in triplets:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                res1 = oracle_and_b32(oracle_and_b32(a, b), c)
                res2 = oracle_and_b32(a, oracle_and_b32(b, c))
                self.assertEqual(res1, res2)

    def test_lds_memory_initialization(self) -> None:
        lds = LDSMemory(65536)
        self.assertEqual(lds.size_bytes, 65536)
        self.assertEqual(lds.num_dwords, 16384)
        for i in range(16):
            self.assertEqual(lds.read_dword(i), 0)

    def test_lds_dword_indexing_and_alignment(self) -> None:
        lds = LDSMemory(65536)
        # Byte addresses 0, 1, 2, 3 all map to dword index 0
        self.assertEqual(lds.get_dword_index(0), 0)
        self.assertEqual(lds.get_dword_index(1), 0)
        self.assertEqual(lds.get_dword_index(2), 0)
        self.assertEqual(lds.get_dword_index(3), 0)
        self.assertEqual(lds.get_dword_index(4), 1)
        self.assertEqual(lds.get_dword_index(7), 1)

    def test_lds_offset_addition(self) -> None:
        lds = LDSMemory(65536)
        self.assertEqual(lds.get_dword_index(0, offset=4), 1)
        self.assertEqual(lds.get_dword_index(8, offset=8), 4)

    def test_lds_out_of_bounds_wrapping(self) -> None:
        lds = LDSMemory(65536)
        # 65536 wraps to 0
        self.assertEqual(lds.get_dword_index(65536), 0)
        self.assertEqual(lds.get_dword_index(65536 + 4), 1)
        self.assertEqual(lds.get_dword_index(0xFFFFFFFC), 16383)

    def test_lds_bank_calculation(self) -> None:
        lds = LDSMemory(65536)
        for i in range(32):
            self.assertEqual(lds.get_bank(i), i)
        self.assertEqual(lds.get_bank(32), 0)
        self.assertEqual(lds.get_bank(33), 1)
        self.assertEqual(lds.get_bank(63), 31)

    def test_lds_atomic_and_b32_execution(self) -> None:
        lds = LDSMemory(65536)
        lds.write_dword(5, 0xFF00FF00)

        # Atomic AND with 0x0F0F0F0F
        old_val = lds.atomic_and_b32(byte_addr=20, val=0x0F0F0F0F)
        self.assertEqual(old_val, 0xFF00FF00)
        self.assertEqual(lds.read_dword(5), 0x0F000F00)

        # Atomic AND with offset
        lds.write_dword(10, 0xAAAAAAAA)
        old_val2 = lds.atomic_and_b32(byte_addr=36, val=0x55555555, offset=4)
        self.assertEqual(old_val2, 0xAAAAAAAA)
        self.assertEqual(lds.read_dword(10), 0x00000000)


class DSOpcode(str, enum.Enum):
    """Opcode enumerations for 32-bit LDS bitwise AND operations."""
    DS_AND_B32 = "ds_and_b32"
    DS_AND_RTN_B32 = "ds_and_rtn_b32"

    @property
    def is_return(self) -> bool:
        """True if instruction returns prior LDS value to destination VGPR."""
        return self == DSOpcode.DS_AND_RTN_B32

    @property
    def bit_width(self) -> int:
        """Operand bit width."""
        return 32


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
class DSAndB32Instruction:
    """Symbolic AST node representing an RDNA2 DS bitwise AND instruction."""
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
    def parse(cls, text: str) -> DSAndB32Instruction:
        """Parse canonical RDNA2 assembly string into a DSAndB32Instruction AST."""
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
            offset_part = m[1].split()
            if not offset_part:
                raise ValueError("Malformed offset immediate")
            offset_tok = offset_part[0].rstrip(",")
            try:
                offset_val = int(offset_tok, 0)
            except ValueError:
                raise ValueError(f"Invalid offset integer: {offset_tok}")

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


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions, registers, and assembly parsing."""

    def test_opcode_properties(self) -> None:
        self.assertFalse(DSOpcode.DS_AND_B32.is_return)
        self.assertEqual(DSOpcode.DS_AND_B32.bit_width, 32)

        self.assertTrue(DSOpcode.DS_AND_RTN_B32.is_return)
        self.assertEqual(DSOpcode.DS_AND_RTN_B32.bit_width, 32)

    def test_vgpr_validation(self) -> None:
        v0 = VGPR(0)
        v100 = VGPR(100)
        v255 = VGPR(255)
        self.assertEqual(str(v0), "v0")
        self.assertEqual(str(v100), "v100")
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
        # Non-returning must not have vdst
        DSAndB32Instruction(opcode=DSOpcode.DS_AND_B32, vaddr=VGPR(0), vdata=VGPR(1))
        with self.assertRaises(ValueError):
            DSAndB32Instruction(opcode=DSOpcode.DS_AND_B32, vaddr=VGPR(0), vdata=VGPR(1), vdst=VGPR(2))

        # Returning must have vdst
        DSAndB32Instruction(opcode=DSOpcode.DS_AND_RTN_B32, vaddr=VGPR(0), vdata=VGPR(1), vdst=VGPR(2))
        with self.assertRaises(ValueError):
            DSAndB32Instruction(opcode=DSOpcode.DS_AND_RTN_B32, vaddr=VGPR(0), vdata=VGPR(1), vdst=None)

    def test_disasm_and_parse_roundtrip(self) -> None:
        test_cases = [
            DSAndB32Instruction(DSOpcode.DS_AND_B32, VGPR(2), VGPR(3)),
            DSAndB32Instruction(DSOpcode.DS_AND_B32, VGPR(2), VGPR(3), offset=OffsetImmediate(16)),
            DSAndB32Instruction(DSOpcode.DS_AND_RTN_B32, VGPR(4), VGPR(5), vdst=VGPR(0)),
            DSAndB32Instruction(DSOpcode.DS_AND_RTN_B32, VGPR(8), VGPR(9), vdst=VGPR(7), offset=OffsetImmediate(256)),
        ]
        for inst in test_cases:
            with self.subTest(inst=inst):
                asm = inst.disasm()
                parsed = DSAndB32Instruction.parse(asm)
                self.assertEqual(inst, parsed)


class ScalarRegisterFile:
    """Architectural register file for a single wavefront lane (256 32-bit VGPRs)."""

    def __init__(self) -> None:
        self.regs: List[int] = [0] * 256

    def get(self, reg: VGPR) -> int:
        """Read 32-bit VGPR value."""
        return self.regs[reg.index] & MASK32

    def set(self, reg: VGPR, val: int) -> None:
        """Write 32-bit VGPR value."""
        self.regs[reg.index] = val & MASK32


class ScalarThreadState:
    """Execution state of a single scalar thread executing LDS instructions."""

    def __init__(self, lds: Optional[LDSMemory] = None) -> None:
        self.rf: ScalarRegisterFile = ScalarRegisterFile()
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()

    def get_vgpr(self, reg: VGPR) -> int:
        return self.rf.get(reg)

    def set_vgpr(self, reg: VGPR, val: int) -> None:
        self.rf.set(reg, val)


class ScalarDSInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 DS bitwise AND opcodes."""

    def execute(self, state: ScalarThreadState, inst: DSAndB32Instruction) -> Optional[int]:
        """Execute a single DS bitwise AND instruction on the scalar thread state."""
        addr = state.get_vgpr(inst.vaddr)
        data = state.get_vgpr(inst.vdata)

        old_val = state.lds.atomic_and_b32(addr, data, offset=inst.offset.value)

        if inst.opcode.is_return:
            assert inst.vdst is not None
            state.set_vgpr(inst.vdst, old_val)
            return old_val
        return None


class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction execution of DS and / and_rtn b32."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_ds_and_b32_non_returning(self) -> None:
        self.lds.write_dword(5, 0b11110000)
        self.state.set_vgpr(VGPR(1), 20)  # dword 5
        self.state.set_vgpr(VGPR(2), 0b11001100)

        inst = DSAndB32Instruction(DSOpcode.DS_AND_B32, vaddr=VGPR(1), vdata=VGPR(2))
        res = self.interp.execute(self.state, inst)

        self.assertIsNone(res)
        self.assertEqual(self.lds.read_dword(5), 0b11000000)
        self.assertEqual(self.state.get_vgpr(VGPR(1)), 20)
        self.assertEqual(self.state.get_vgpr(VGPR(2)), 0b11001100)

    def test_scalar_ds_and_rtn_b32_returning(self) -> None:
        self.lds.write_dword(8, 0xFF00FF00)
        self.state.set_vgpr(VGPR(1), 32)  # dword 8
        self.state.set_vgpr(VGPR(2), 0x0F0F0F0F)
        self.state.set_vgpr(VGPR(0), 0)

        inst = DSAndB32Instruction(DSOpcode.DS_AND_RTN_B32, vaddr=VGPR(1), vdata=VGPR(2), vdst=VGPR(0))
        res = self.interp.execute(self.state, inst)

        self.assertEqual(res, 0xFF00FF00)
        self.assertEqual(self.state.get_vgpr(VGPR(0)), 0xFF00FF00)
        self.assertEqual(self.lds.read_dword(8), 0x0F000F00)

    def test_scalar_ds_and_rtn_b32_with_offset(self) -> None:
        self.lds.write_dword(10, 0x12345678)
        self.state.set_vgpr(VGPR(1), 36)  # dword 9
        self.state.set_vgpr(VGPR(2), 0xFFFF0000)

        inst = DSAndB32Instruction(
            DSOpcode.DS_AND_RTN_B32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
            vdst=VGPR(3),
            offset=OffsetImmediate(4),  # offset 4 -> dword 10
        )
        res = self.interp.execute(self.state, inst)

        self.assertEqual(res, 0x12345678)
        self.assertEqual(self.state.get_vgpr(VGPR(3)), 0x12345678)
        self.assertEqual(self.lds.read_dword(10), 0x12340000)

    def test_scalar_ds_and_rtn_identity_mask(self) -> None:
        self.lds.write_dword(2, 0xDEADBEEF)
        self.state.set_vgpr(VGPR(1), 8)
        self.state.set_vgpr(VGPR(2), 0xFFFFFFFF)

        inst = DSAndB32Instruction(DSOpcode.DS_AND_RTN_B32, vaddr=VGPR(1), vdata=VGPR(2), vdst=VGPR(4))
        res = self.interp.execute(self.state, inst)

        self.assertEqual(res, 0xDEADBEEF)
        self.assertEqual(self.state.get_vgpr(VGPR(4)), 0xDEADBEEF)
        self.assertEqual(self.lds.read_dword(2), 0xDEADBEEF)

    def test_scalar_ds_and_rtn_zero_mask(self) -> None:
        self.lds.write_dword(3, 0xCAFEBABE)
        self.state.set_vgpr(VGPR(1), 12)
        self.state.set_vgpr(VGPR(2), 0x00000000)

        inst = DSAndB32Instruction(DSOpcode.DS_AND_RTN_B32, vaddr=VGPR(1), vdata=VGPR(2), vdst=VGPR(5))
        res = self.interp.execute(self.state, inst)

        self.assertEqual(res, 0xCAFEBABE)
        self.assertEqual(self.state.get_vgpr(VGPR(5)), 0xCAFEBABE)
        self.assertEqual(self.lds.read_dword(3), 0x00000000)

    def test_scalar_sequential_bit_clearing(self) -> None:
        self.lds.write_dword(0, 0xFFFFFFFF)
        self.state.set_vgpr(VGPR(1), 0)

        # Clear upper 16 bits
        self.state.set_vgpr(VGPR(2), 0x0000FFFF)
        r1 = self.interp.execute(self.state, DSAndB32Instruction(DSOpcode.DS_AND_RTN_B32, VGPR(1), VGPR(2), vdst=VGPR(10)))
        self.assertEqual(r1, 0xFFFFFFFF)
        self.assertEqual(self.lds.read_dword(0), 0x0000FFFF)

        # Clear lower 8 bits
        self.state.set_vgpr(VGPR(2), 0x0000FF00)
        r2 = self.interp.execute(self.state, DSAndB32Instruction(DSOpcode.DS_AND_RTN_B32, VGPR(1), VGPR(2), vdst=VGPR(11)))
        self.assertEqual(r2, 0x0000FFFF)
        self.assertEqual(self.lds.read_dword(0), 0x0000FF00)


if __name__ == "__main__":
    unittest.main()
