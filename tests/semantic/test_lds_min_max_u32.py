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


class ScalarThreadState:
    """Architectural state of a single scalar thread executing LDS instructions."""

    def __init__(self, lds: Optional[LDSMemory] = None) -> None:
        self.vgprs: List[int] = [0] * 256
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()

    def get_vgpr(self, reg: VGPR) -> int:
        """Read a 32-bit register value."""
        return self.vgprs[reg.index] & MASK32

    def set_vgpr(self, reg: VGPR, val: int) -> None:
        """Write a 32-bit register value."""
        self.vgprs[reg.index] = val & MASK32


class ScalarDSInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 DS min/max u32 opcodes."""

    def execute(self, state: ScalarThreadState, inst: DSMinMaxU32Instruction) -> Optional[int]:
        """Execute a single DS min/max instruction on the scalar thread state."""
        addr = state.get_vgpr(inst.vaddr)
        data = state.get_vgpr(inst.vdata)

        if inst.opcode.is_min:
            old_val = state.lds.atomic_min_u32(addr, data, offset=inst.offset.value)
        elif inst.opcode.is_max:
            old_val = state.lds.atomic_max_u32(addr, data, offset=inst.offset.value)
        else:
            raise ValueError(f"Unsupported opcode: {inst.opcode}")

        if inst.opcode.is_return:
            assert inst.vdst is not None
            state.set_vgpr(inst.vdst, old_val)
            return old_val
        return None


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction execution of DS min/max u32."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_ds_min_u32_non_returning(self) -> None:
        self.lds.write_dword(5, 100)
        self.state.set_vgpr(VGPR(1), 20)  # dword 5
        self.state.set_vgpr(VGPR(2), 25)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MIN_U32, vaddr=VGPR(1), vdata=VGPR(2))
        res = self.interp.execute(self.state, inst)

        self.assertIsNone(res)
        self.assertEqual(self.lds.read_dword(5), 25)
        self.assertEqual(self.state.get_vgpr(VGPR(1)), 20)
        self.assertEqual(self.state.get_vgpr(VGPR(2)), 25)

    def test_scalar_ds_max_u32_non_returning(self) -> None:
        self.lds.write_dword(8, 10)
        self.state.set_vgpr(VGPR(1), 32)  # dword 8
        self.state.set_vgpr(VGPR(2), 80)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MAX_U32, vaddr=VGPR(1), vdata=VGPR(2))
        res = self.interp.execute(self.state, inst)

        self.assertIsNone(res)
        self.assertEqual(self.lds.read_dword(8), 80)

    def test_scalar_ds_min_rtn_u32_returning(self) -> None:
        self.lds.write_dword(12, 500)
        self.state.set_vgpr(VGPR(1), 48)  # dword 12
        self.state.set_vgpr(VGPR(2), 200)
        self.state.set_vgpr(VGPR(0), 0xDEADBEEF)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MIN_RTN_U32, vaddr=VGPR(1), vdata=VGPR(2), vdst=VGPR(0))
        res = self.interp.execute(self.state, inst)

        self.assertEqual(res, 500)
        self.assertEqual(self.state.get_vgpr(VGPR(0)), 500)
        self.assertEqual(self.lds.read_dword(12), 200)

    def test_scalar_ds_max_rtn_u32_returning(self) -> None:
        self.lds.write_dword(15, 20)
        self.state.set_vgpr(VGPR(1), 60)  # dword 15
        self.state.set_vgpr(VGPR(2), 100)
        self.state.set_vgpr(VGPR(5), 0)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MAX_RTN_U32, vaddr=VGPR(1), vdata=VGPR(2), vdst=VGPR(5))
        res = self.interp.execute(self.state, inst)

        self.assertEqual(res, 20)
        self.assertEqual(self.state.get_vgpr(VGPR(5)), 20)
        self.assertEqual(self.lds.read_dword(15), 100)

    def test_scalar_offset_and_unaligned_address(self) -> None:
        self.lds.write_dword(20, 1000)
        self.state.set_vgpr(VGPR(1), 65)  # byte 65 (unaligned) + offset 15 = 80 -> dword 20
        self.state.set_vgpr(VGPR(2), 250)

        inst = DSMinMaxU32Instruction(
            DSOpcode.DS_MIN_RTN_U32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
            vdst=VGPR(3),
            offset=OffsetImmediate(15),
        )
        self.interp.execute(self.state, inst)

        self.assertEqual(self.state.get_vgpr(VGPR(3)), 1000)
        self.assertEqual(self.lds.read_dword(20), 250)

    def test_sequential_atomic_chain(self) -> None:
        self.lds.write_dword(0, 100)
        self.state.set_vgpr(VGPR(1), 0)

        # min with 150 -> remains 100
        self.state.set_vgpr(VGPR(2), 150)
        self.interp.execute(self.state, DSMinMaxU32Instruction(DSOpcode.DS_MIN_U32, VGPR(1), VGPR(2)))
        self.assertEqual(self.lds.read_dword(0), 100)

        # min with 30 -> becomes 30
        self.state.set_vgpr(VGPR(2), 30)
        self.interp.execute(self.state, DSMinMaxU32Instruction(DSOpcode.DS_MIN_U32, VGPR(1), VGPR(2)))
        self.assertEqual(self.lds.read_dword(0), 30)

        # max with 70 -> becomes 70
        self.state.set_vgpr(VGPR(2), 70)
        self.interp.execute(self.state, DSMinMaxU32Instruction(DSOpcode.DS_MAX_U32, VGPR(1), VGPR(2)))
        self.assertEqual(self.lds.read_dword(0), 70)

        # max rtn with 20 -> remains 70, returns 70
        self.state.set_vgpr(VGPR(2), 20)
        ret = self.interp.execute(self.state, DSMinMaxU32Instruction(DSOpcode.DS_MAX_RTN_U32, VGPR(1), VGPR(2), vdst=VGPR(4)))
        self.assertEqual(ret, 70)
        self.assertEqual(self.state.get_vgpr(VGPR(4)), 70)
        self.assertEqual(self.lds.read_dword(0), 70)


class SIMDWaveState:
    """Architectural wavefront execution state for 32 or 64 lanes."""

    def __init__(self, wave_size: int = 32, exec_mask: int = 0xFFFFFFFF, lds: Optional[LDSMemory] = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wavefront size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.exec_mask: int = exec_mask & ((1 << wave_size) - 1)
        self.vgprs: List[List[int]] = [[0] * 256 for _ in range(wave_size)]
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()

    def is_lane_active(self, lane: int) -> bool:
        """Return True if the given lane is enabled in the current EXEC mask."""
        return bool((self.exec_mask >> lane) & 1)

    def get_vgpr(self, lane: int, reg: VGPR) -> int:
        """Read 32-bit VGPR for a specific lane."""
        return self.vgprs[lane][reg.index] & MASK32

    def set_vgpr(self, lane: int, reg: VGPR, val: int) -> None:
        """Write 32-bit VGPR for a specific lane."""
        self.vgprs[lane][reg.index] = val & MASK32


class SIMDDSInterpreter:
    """Multi-lane SIMD instruction interpreter modeling RDNA2 lockstep execution and LDS atomics."""

    def execute(self, state: SIMDWaveState, inst: DSMinMaxU32Instruction) -> Dict[int, int]:
        """Execute DS min/max u32 instruction across all active lanes of the wave."""
        active_lanes = [lane for lane in range(state.wave_size) if state.is_lane_active(lane)]
        latched_inputs = [
            (lane, state.get_vgpr(lane, inst.vaddr), state.get_vgpr(lane, inst.vdata))
            for lane in active_lanes
        ]

        returns: Dict[int, int] = {}

        for lane, addr, data in latched_inputs:
            if inst.opcode.is_min:
                old_val = state.lds.atomic_min_u32(addr, data, offset=inst.offset.value)
            elif inst.opcode.is_max:
                old_val = state.lds.atomic_max_u32(addr, data, offset=inst.offset.value)
            else:
                raise ValueError(f"Unsupported opcode: {inst.opcode}")

            if inst.opcode.is_return:
                assert inst.vdst is not None
                state.set_vgpr(lane, inst.vdst, old_val)
                returns[lane] = old_val

        return returns


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD lockstep execution, 32/64-bit EXEC masking, and address convergence."""

    def test_wave32_full_mask_divergent_addresses(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        # Initialize LDS dwords 0..31 with 1000
        for i in range(32):
            lds.write_dword(i, 1000)

        # Lane i targets dword i with value (i * 10)
        for lane in range(32):
            wave.set_vgpr(lane, VGPR(1), lane * 4)
            wave.set_vgpr(lane, VGPR(2), lane * 10)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MIN_U32, vaddr=VGPR(1), vdata=VGPR(2))
        interp.execute(wave, inst)

        for i in range(32):
            self.assertEqual(lds.read_dword(i), i * 10)

    def test_wave64_full_mask_divergent_addresses(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=64, exec_mask=0xFFFFFFFFFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        for i in range(64):
            lds.write_dword(i, 5)

        for lane in range(64):
            wave.set_vgpr(lane, VGPR(1), lane * 4)
            wave.set_vgpr(lane, VGPR(2), lane * 20)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MAX_U32, vaddr=VGPR(1), vdata=VGPR(2))
        interp.execute(wave, inst)

        for i in range(64):
            expected = max(5, i * 20)
            self.assertEqual(lds.read_dword(i), expected)

    def test_wave32_partial_exec_masking(self) -> None:
        lds = LDSMemory(65536)
        # Even lanes only: 0x55555555
        wave = SIMDWaveState(wave_size=32, exec_mask=0x55555555, lds=lds)
        interp = SIMDDSInterpreter()

        for i in range(32):
            lds.write_dword(i, 500)

        for lane in range(32):
            wave.set_vgpr(lane, VGPR(1), lane * 4)
            wave.set_vgpr(lane, VGPR(2), 100)
            wave.set_vgpr(lane, VGPR(0), 0xCAFEBABE)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MIN_RTN_U32, vaddr=VGPR(1), vdata=VGPR(2), vdst=VGPR(0))
        returns = interp.execute(wave, inst)

        self.assertEqual(len(returns), 16)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(lds.read_dword(lane), 100)
                self.assertEqual(wave.get_vgpr(lane, VGPR(0)), 500)
                self.assertIn(lane, returns)
            else:
                self.assertEqual(lds.read_dword(lane), 500)
                self.assertEqual(wave.get_vgpr(lane, VGPR(0)), 0xCAFEBABE)
                self.assertNotIn(lane, returns)

    def test_zero_exec_mask_no_mutation(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x0, lds=lds)
        interp = SIMDDSInterpreter()

        lds.write_dword(0, 999)
        for lane in range(32):
            wave.set_vgpr(lane, VGPR(1), 0)
            wave.set_vgpr(lane, VGPR(2), 10)
            wave.set_vgpr(lane, VGPR(3), 0x12345678)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MIN_RTN_U32, vaddr=VGPR(1), vdata=VGPR(2), vdst=VGPR(3))
        returns = interp.execute(wave, inst)

        self.assertEqual(len(returns), 0)
        self.assertEqual(lds.read_dword(0), 999)
        for lane in range(32):
            self.assertEqual(wave.get_vgpr(lane, VGPR(3)), 0x12345678)

    def test_convergent_collision_all_lanes_single_dword(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        # Initial LDS value 1000
        lds.write_dword(4, 1000)

        # Lane values: 320, 310, ..., 10 (Lane 31 writes 10)
        for lane in range(32):
            wave.set_vgpr(lane, VGPR(1), 16)  # dword 4
            wave.set_vgpr(lane, VGPR(2), (32 - lane) * 10)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MIN_U32, vaddr=VGPR(1), vdata=VGPR(2))
        interp.execute(wave, inst)

        # LDS at dword 4 converges to global minimum: 10
        self.assertEqual(lds.read_dword(4), 10)

    def test_convergent_collision_max_u32_all_lanes(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        lds.write_dword(7, 50)

        for lane in range(32):
            wave.set_vgpr(lane, VGPR(1), 28)  # dword 7
            wave.set_vgpr(lane, VGPR(2), lane * 50)  # max is lane 31 (1550)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MAX_U32, vaddr=VGPR(1), vdata=VGPR(2))
        interp.execute(wave, inst)

        self.assertEqual(lds.read_dword(7), 1550)


class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_dst_aliases_addr(self) -> None:
        self.lds.write_dword(3, 500)
        self.state.set_vgpr(VGPR(1), 12)  # byte 12 -> dword 3
        self.state.set_vgpr(VGPR(2), 100)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MIN_RTN_U32, vaddr=VGPR(1), vdata=VGPR(2), vdst=VGPR(1))
        ret = self.interp.execute(self.state, inst)

        self.assertEqual(ret, 500)
        self.assertEqual(self.state.get_vgpr(VGPR(1)), 500)
        self.assertEqual(self.state.get_vgpr(VGPR(2)), 100)
        self.assertEqual(self.lds.read_dword(3), 100)

    def test_scalar_dst_aliases_data(self) -> None:
        self.lds.write_dword(6, 50)
        self.state.set_vgpr(VGPR(1), 24)  # byte 24 -> dword 6
        self.state.set_vgpr(VGPR(2), 250)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MAX_RTN_U32, vaddr=VGPR(1), vdata=VGPR(2), vdst=VGPR(2))
        ret = self.interp.execute(self.state, inst)

        self.assertEqual(ret, 50)
        self.assertEqual(self.state.get_vgpr(VGPR(2)), 50)
        self.assertEqual(self.state.get_vgpr(VGPR(1)), 24)
        self.assertEqual(self.lds.read_dword(6), 250)

    def test_scalar_addr_aliases_data_non_returning(self) -> None:
        self.lds.write_dword(4, 100)
        self.state.set_vgpr(VGPR(5), 16)  # byte 16 -> dword 4, and value 16

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MIN_U32, vaddr=VGPR(5), vdata=VGPR(5))
        self.interp.execute(self.state, inst)

        self.assertEqual(self.state.get_vgpr(VGPR(5)), 16)
        self.assertEqual(self.lds.read_dword(4), 16)

    def test_scalar_triple_alias(self) -> None:
        self.lds.write_dword(8, 200)
        self.state.set_vgpr(VGPR(0), 32)  # byte 32 -> dword 8, and value 32

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MIN_RTN_U32, vaddr=VGPR(0), vdata=VGPR(0), vdst=VGPR(0))
        ret = self.interp.execute(self.state, inst)

        self.assertEqual(ret, 200)
        self.assertEqual(self.state.get_vgpr(VGPR(0)), 200)
        self.assertEqual(self.lds.read_dword(8), 32)

    def test_operand_preservation_non_aliased(self) -> None:
        self.lds.write_dword(10, 10)
        self.state.set_vgpr(VGPR(1), 40)
        self.state.set_vgpr(VGPR(2), 50)
        self.state.set_vgpr(VGPR(3), 0xCAFEBABE)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MAX_RTN_U32, vaddr=VGPR(1), vdata=VGPR(2), vdst=VGPR(3))
        self.interp.execute(self.state, inst)

        self.assertEqual(self.state.get_vgpr(VGPR(1)), 40)
        self.assertEqual(self.state.get_vgpr(VGPR(2)), 50)
        self.assertEqual(self.state.get_vgpr(VGPR(3)), 10)

    def test_simd_aliasing_under_divergence(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xAAAAAAAA, lds=lds)
        interp = SIMDDSInterpreter()

        for i in range(32):
            lds.write_dword(i, 100)

        for lane in range(32):
            wave.set_vgpr(lane, VGPR(1), lane * 4)
            wave.set_vgpr(lane, VGPR(2), lane * 10)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MIN_RTN_U32, vaddr=VGPR(1), vdata=VGPR(2), vdst=VGPR(1))
        returns = interp.execute(wave, inst)

        for lane in range(32):
            if lane % 2 != 0:
                self.assertEqual(wave.get_vgpr(lane, VGPR(1)), 100)
                expected_min = min(100, lane * 10)
                self.assertEqual(lds.read_dword(lane), expected_min)
            else:
                self.assertEqual(wave.get_vgpr(lane, VGPR(1)), lane * 4)
                self.assertEqual(lds.read_dword(lane), 100)


@dataclasses.dataclass
class LDSStatusFlags:
    """Architectural status flags, wrapping conditions, and bank metrics for LDS operations."""
    carry_out: bool = False
    unaligned_address: bool = False
    oob_wrapped: bool = False
    bank_conflicts: int = 0
    address_collisions: int = 0


def evaluate_status_flags(
    wave: SIMDWaveState,
    inst: DSMinMaxU32Instruction,
) -> LDSStatusFlags:
    """Evaluate architectural flags, address wrap conditions, and bank conflicts."""
    flags = LDSStatusFlags()
    active_lanes = [l for l in range(wave.wave_size) if wave.is_lane_active(l)]

    bank_to_dwords: Dict[int, set[int]] = {}
    dword_to_lanes: Dict[int, list[int]] = {}

    for lane in active_lanes:
        addr = wave.get_vgpr(lane, inst.vaddr)
        off = inst.offset.value

        if (addr + off) > MASK32:
            flags.carry_out = True

        effective_byte = (addr + off) & MASK32
        if (effective_byte & 3) != 0:
            flags.unaligned_address = True

        if effective_byte >= wave.lds.size_bytes:
            flags.oob_wrapped = True

        dword_idx = wave.lds.get_dword_index(addr, off)
        bank_idx = wave.lds.get_bank(dword_idx)

        bank_to_dwords.setdefault(bank_idx, set()).add(dword_idx)
        dword_to_lanes.setdefault(dword_idx, []).append(lane)

    for bank, dword_set in bank_to_dwords.items():
        if len(dword_set) > 1:
            flags.bank_conflicts += (len(dword_set) - 1)

    for dword, lanes in dword_to_lanes.items():
        if len(lanes) > 1:
            flags.address_collisions += (len(lanes) - 1)

    return flags


class TestStatusFlagsAndBankConflicts(unittest.TestCase):
    """Unit tests for status flags, bank conflicts, and address wrap conditions."""

    def test_address_carry_out_generation(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x1)
        wave.set_vgpr(0, VGPR(1), 0xFFFFFFFF)
        wave.set_vgpr(0, VGPR(2), 50)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MIN_U32, vaddr=VGPR(1), vdata=VGPR(2), offset=OffsetImmediate(16))
        flags = evaluate_status_flags(wave, inst)

        self.assertTrue(flags.carry_out)

    def test_unaligned_byte_address_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x1)
        wave.set_vgpr(0, VGPR(1), 7)
        wave.set_vgpr(0, VGPR(2), 50)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MIN_U32, vaddr=VGPR(1), vdata=VGPR(2))
        flags = evaluate_status_flags(wave, inst)

        self.assertTrue(flags.unaligned_address)

    def test_oob_address_wrapping_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x1)
        wave.set_vgpr(0, VGPR(1), 70000)
        wave.set_vgpr(0, VGPR(2), 50)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MAX_U32, vaddr=VGPR(1), vdata=VGPR(2))
        flags = evaluate_status_flags(wave, inst)

        self.assertTrue(flags.oob_wrapped)

    def test_bank_conflicts_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x7)
        wave.set_vgpr(0, VGPR(1), 0 * 4)   # bank 0
        wave.set_vgpr(1, VGPR(1), 32 * 4)  # bank 0
        wave.set_vgpr(2, VGPR(1), 64 * 4)  # bank 0

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MIN_U32, vaddr=VGPR(1), vdata=VGPR(2))
        flags = evaluate_status_flags(wave, inst)

        self.assertEqual(flags.bank_conflicts, 2)
        self.assertEqual(flags.address_collisions, 0)

    def test_address_collisions_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0xF)
        for l in range(4):
            wave.set_vgpr(l, VGPR(1), 40)

        inst = DSMinMaxU32Instruction(DSOpcode.DS_MAX_U32, vaddr=VGPR(1), vdata=VGPR(2))
        flags = evaluate_status_flags(wave, inst)

        self.assertEqual(flags.address_collisions, 3)
        self.assertEqual(flags.bank_conflicts, 0)


class TestUnsignedBoundaryExtremes(unittest.TestCase):
    """Exhaustive tests for unsigned 32-bit boundary extremes and bit patterns."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_unsigned_extremes_permutations(self) -> None:
        cases = [
            (0, 0xFFFFFFFF, 0, 0xFFFFFFFF),
            (0xFFFFFFFF, 0, 0, 0xFFFFFFFF),
            (0x7FFFFFFF, 0x80000000, 0x7FFFFFFF, 0x80000000),  # In unsigned, 0x7FFFFFFF < 0x80000000
            (0x80000000, 0xFFFFFFFF, 0x80000000, 0xFFFFFFFF),
            (0, 0, 0, 0),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
        ]
        for a, b, exp_min, exp_max in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(oracle_min_u32(a, b), exp_min)
                self.assertEqual(oracle_max_u32(a, b), exp_max)

                # In-LDS scalar execution test
                self.lds.write_dword(0, a)
                self.state.set_vgpr(VGPR(1), 0)
                self.state.set_vgpr(VGPR(2), b)
                self.interp.execute(self.state, DSMinMaxU32Instruction(DSOpcode.DS_MIN_U32, VGPR(1), VGPR(2)))
                self.assertEqual(self.lds.read_dword(0), exp_min)

                self.lds.write_dword(0, a)
                self.interp.execute(self.state, DSMinMaxU32Instruction(DSOpcode.DS_MAX_U32, VGPR(1), VGPR(2)))
                self.assertEqual(self.lds.read_dword(0), exp_max)

    def test_single_bit_walks(self) -> None:
        # Powers of 2: 1 << 0 through 1 << 31
        bits = [1 << k for k in range(32)]
        for i in range(len(bits) - 1):
            smaller = bits[i]
            larger = bits[i + 1]
            self.assertEqual(oracle_min_u32(smaller, larger), smaller)
            self.assertEqual(oracle_max_u32(smaller, larger), larger)

    def test_alternating_bit_patterns(self) -> None:
        pat_a = 0x55555555  # 01010101...
        pat_b = 0xAAAAAAAA  # 10101010... (larger in unsigned)
        self.assertEqual(oracle_min_u32(pat_a, pat_b), pat_a)
        self.assertEqual(oracle_max_u32(pat_a, pat_b), pat_b)


if __name__ == "__main__":
    unittest.main()
