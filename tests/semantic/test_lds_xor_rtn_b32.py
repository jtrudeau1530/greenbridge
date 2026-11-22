from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_xor_rtn_b32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 32-bit Bitwise XOR with Return:
Performs an atomic bitwise XOR on 32-bit LDS memory and returns the original value.

Syntax:
    ds_xor_rtn_b32 vdst, vaddr, vdata0 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    LDS[dword_index] = (old_val ^ vdata0) & 0xFFFFFFFF
    vdst = old_val
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK_32: int = 0xFFFF_FFFF          # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK_32
    return val - (1 << 32) if (val & 0x8000_0000) else val


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
    """Determine the LDS memory bank index (0..31) for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_xor_rtn_b32(current_mem: int, data: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_xor_rtn_b32.

    Returns:
        (old_val, new_val)
    """
    old_val = u32(current_mem)
    data_masked = u32(data)
    new_val = old_val ^ data_masked
    return old_val, new_val


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
        self.memory[idx] = u32(val)

    def atomic_xor_b32(self, vaddr: int, data: int, offset: int = 0) -> Tuple[int, int]:
        """Execute atomic XOR with return on LDS dword.

        Returns (old_val, new_val).
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        curr = self.memory[idx]
        old_val, new_val = oracle_ds_xor_rtn_b32(curr, data)
        self.memory[idx] = new_val
        return old_val, new_val


class TestLdsXorRtnB32ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and LDS architectural model."""

    def setUp(self) -> None:
        self.lds = LDSModel()

    def test_xor_identity_zero(self) -> None:
        """XOR with zero preserves the original value and returns it."""
        addr = 0x40
        self.lds.write_dword(addr, 0x1234_5678)
        old_val, new_val = self.lds.atomic_xor_b32(addr, 0x0000_0000)
        self.assertEqual(old_val, 0x1234_5678)
        self.assertEqual(new_val, 0x1234_5678)
        self.assertEqual(self.lds.read_dword(addr), 0x1234_5678)

    def test_xor_all_ones_inversion(self) -> None:
        """XOR with all-ones inverts all 32 bits."""
        addr = 0x80
        self.lds.write_dword(addr, 0xAAAA_5555)
        old_val, new_val = self.lds.atomic_xor_b32(addr, 0xFFFF_FFFF)
        self.assertEqual(old_val, 0xAAAA_5555)
        self.assertEqual(new_val, 0x5555_AAAA)
        self.assertEqual(self.lds.read_dword(addr), 0x5555_AAAA)

    def test_xor_self_annihilation(self) -> None:
        """XORing a value with itself yields zero."""
        addr = 0x100
        val = 0xCAFE_BABE
        self.lds.write_dword(addr, val)
        old_val, new_val = self.lds.atomic_xor_b32(addr, val)
        self.assertEqual(old_val, val)
        self.assertEqual(new_val, 0)
        self.assertEqual(self.lds.read_dword(addr), 0)

    def test_xor_involution_roundtrip(self) -> None:
        """Applying the same XOR operand twice restores the original value: (A ^ B) ^ B == A."""
        addr = 0x200
        initial = 0xDEAD_BEEF
        key = 0x1337_C0DE
        self.lds.write_dword(addr, initial)

        # First XOR
        old1, new1 = self.lds.atomic_xor_b32(addr, key)
        self.assertEqual(old1, initial)
        self.assertEqual(new1, initial ^ key)

        # Second XOR with same key restores initial
        old2, new2 = self.lds.atomic_xor_b32(addr, key)
        self.assertEqual(old2, initial ^ key)
        self.assertEqual(new2, initial)
        self.assertEqual(self.lds.read_dword(addr), initial)

    def test_offset_calculation_and_wrap(self) -> None:
        """Offset is 16-bit unsigned and added to vaddr modulo 2^32."""
        base = 0x1000
        offset = 0x24
        eff = calc_lds_effective_address(base, offset)
        self.assertEqual(eff, 0x1024)

        # 16-bit offset masking
        eff_masked = calc_lds_effective_address(base, 0x1_0004)
        self.assertEqual(eff_masked, 0x1004)

        # 32-bit address wraparound
        eff_wrap = calc_lds_effective_address(0xFFFF_FFF0, 0x0020)
        self.assertEqual(eff_wrap, 0x0000_0010)

    def test_dword_alignment_truncation(self) -> None:
        """Low 2 bits of effective address are ignored for dword indexing."""
        idx0 = calc_lds_dword_index(0x400)
        idx1 = calc_lds_dword_index(0x401)
        idx2 = calc_lds_dword_index(0x402)
        idx3 = calc_lds_dword_index(0x403)
        self.assertEqual(idx0, 0x100)
        self.assertEqual(idx1, 0x100)
        self.assertEqual(idx2, 0x100)
        self.assertEqual(idx3, 0x100)

    def test_bank_distribution(self) -> None:
        """32 consecutive dwords map to 32 distinct LDS banks [0..31]."""
        banks = [calc_lds_bank(i * 4) for i in range(32)]
        self.assertEqual(banks, list(range(32)))
        # Bank 32 wraps to 0
        self.assertEqual(calc_lds_bank(32 * 4), 0)

    def test_lds_capacity_wrap(self) -> None:
        """LDS access wraps modulo LDS capacity (default 64 KiB = 16384 dwords)."""
        addr_base = 0x10
        addr_wrapped = addr_base + LDS_SIZE_BYTES
        idx_base = calc_lds_dword_index(addr_base)
        idx_wrapped = calc_lds_dword_index(addr_wrapped)
        self.assertEqual(idx_base, idx_wrapped)

    def test_bit_patterns_walking_ones_and_zeros(self) -> None:
        """Test bitwise XOR across all 32 bit positions with walking ones and walking zeros."""
        addr = 0x300
        for bit in range(32):
            one = 1 << bit
            zero_inv = u32(~one)
            self.lds.write_dword(addr, 0)
            old, new = self.lds.atomic_xor_b32(addr, one)
            self.assertEqual(old, 0)
            self.assertEqual(new, one)

            old2, new2 = self.lds.atomic_xor_b32(addr, zero_inv)
            self.assertEqual(old2, one)
            self.assertEqual(new2, MASK_32)


# ============================================================================
# Section 2: Symbolic AST Opcode Definitions and Typed Expressions
# ============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()
    IMM = auto()


@dataclass(frozen=True)
class VGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index < DEFAULT_NUM_VGPRS):
            raise ValueError(f"VGPR index out of range [0, {DEFAULT_NUM_VGPRS - 1}]: {self.index}")

    def __str__(self) -> str:
        return f"v{self.index}"


@dataclass(frozen=True)
class SGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index < DEFAULT_NUM_SGPRS):
            raise ValueError(f"SGPR index out of range [0, {DEFAULT_NUM_SGPRS - 1}]: {self.index}")

    def __str__(self) -> str:
        return f"s{self.index}"


@dataclass(frozen=True)
class Immediate:
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", u32(self.value))

    def __str__(self) -> str:
        return f"0x{self.value:X}"


@dataclass(frozen=True)
class DSXorRtnB32Instruction:
    dst: VGPR
    addr: VGPR
    data0: Union[VGPR, Immediate]
    offset: int = 0
    gds: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"LDS offset out of 16-bit range [0, 65535]: {self.offset}")
        if not isinstance(self.dst, VGPR):
            raise TypeError(f"dst must be a VGPR, got {type(self.dst)}")
        if not isinstance(self.addr, VGPR):
            raise TypeError(f"addr must be a VGPR, got {type(self.addr)}")
        if not isinstance(self.data0, (VGPR, Immediate)):
            raise TypeError(f"data0 must be VGPR or Immediate, got {type(self.data0)}")

    def disasm(self) -> str:
        tokens = ["ds_xor_rtn_b32", str(self.dst), str(self.addr), str(self.data0)]
        extra = []
        if self.offset != 0:
            extra.append(f"offset:{self.offset}")
        if self.gds:
            extra.append("gds")
        core = f"{tokens[0]} {tokens[1]}, {tokens[2]}, {tokens[3]}"
        return f"{core} {' '.join(extra)}".rstrip()


class ASTExpr:
    """Base class for symbolic typed expressions in RDNA2 bitwise algebra."""

    def simplify(self) -> ASTExpr:
        raise NotImplementedError

    def evaluate(self, env: Dict[str, int]) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Symbol(ASTExpr):
    name: str

    def simplify(self) -> ASTExpr:
        return self

    def evaluate(self, env: Dict[str, int]) -> int:
        if self.name not in env:
            raise KeyError(f"Symbol '{self.name}' not found in evaluation environment")
        return u32(env[self.name])

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Constant(ASTExpr):
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", u32(self.value))

    def simplify(self) -> ASTExpr:
        return self

    def evaluate(self, env: Dict[str, int]) -> int:
        return self.value

    def __str__(self) -> str:
        return f"0x{self.value:08X}"


@dataclass(frozen=True)
class BitXorExpr(ASTExpr):
    left: ASTExpr
    right: ASTExpr

    def _flatten(self) -> List[ASTExpr]:
        terms: List[ASTExpr] = []
        for child in (self.left, self.right):
            if isinstance(child, BitXorExpr):
                terms.extend(child._flatten())
            else:
                terms.append(child)
        return terms

    def simplify(self) -> ASTExpr:
        simplified_left = self.left.simplify()
        simplified_right = self.right.simplify()

        if isinstance(simplified_left, Constant) and isinstance(simplified_right, Constant):
            return Constant(simplified_left.value ^ simplified_right.value)

        terms: List[ASTExpr] = []
        for child in (simplified_left, simplified_right):
            if isinstance(child, BitXorExpr):
                terms.extend(child._flatten())
            else:
                terms.append(child)

        const_accum = 0
        symbol_counts: Dict[str, int] = {}
        for t in terms:
            if isinstance(t, Constant):
                const_accum ^= t.value
            elif isinstance(t, Symbol):
                symbol_counts[t.name] = symbol_counts.get(t.name, 0) + 1

        remaining_symbols: List[Symbol] = [
            Symbol(name) for name, count in sorted(symbol_counts.items()) if (count % 2) != 0
        ]

        if not remaining_symbols:
            return Constant(const_accum)

        result: Optional[ASTExpr] = None
        for sym in remaining_symbols:
            result = sym if result is None else BitXorExpr(result, sym)

        if const_accum != 0:
            assert result is not None
            return BitXorExpr(result, Constant(const_accum))

        assert result is not None
        return result

    def evaluate(self, env: Dict[str, int]) -> int:
        return u32(self.left.evaluate(env) ^ self.right.evaluate(env))

    def __str__(self) -> str:
        return f"({self.left} ^ {self.right})"


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_valid_creation_and_disassembly(self) -> None:
        instr = DSXorRtnB32Instruction(
            dst=VGPR(0),
            addr=VGPR(1),
            data0=VGPR(2),
            offset=16,
        )
        self.assertEqual(instr.disasm(), "ds_xor_rtn_b32 v0, v1, v2 offset:16")

        instr_imm = DSXorRtnB32Instruction(
            dst=VGPR(10),
            addr=VGPR(20),
            data0=Immediate(0x1234),
            offset=0,
            gds=True,
        )
        self.assertEqual(instr_imm.disasm(), "ds_xor_rtn_b32 v10, v20, 0x1234 gds")

    def test_instruction_validation_vgpr_bounds(self) -> None:
        with self.assertRaises(ValueError):
            VGPR(256)
        with self.assertRaises(ValueError):
            VGPR(-1)

    def test_instruction_validation_offset_bounds(self) -> None:
        with self.assertRaises(ValueError):
            DSXorRtnB32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2), offset=65536)
        with self.assertRaises(ValueError):
            DSXorRtnB32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2), offset=-1)

    def test_symbolic_constant_folding(self) -> None:
        expr = BitXorExpr(Constant(0x0F0F), Constant(0x00FF))
        simplified = expr.simplify()
        self.assertIsInstance(simplified, Constant)
        self.assertEqual(simplified.evaluate({}), 0x0FF0)

    def test_symbolic_identity_zero(self) -> None:
        x = Symbol("X")
        expr = BitXorExpr(x, Constant(0))
        simplified = expr.simplify()
        self.assertEqual(simplified, x)

    def test_symbolic_self_inverse(self) -> None:
        x = Symbol("X")
        expr = BitXorExpr(x, x)
        simplified = expr.simplify()
        self.assertEqual(simplified, Constant(0))

    def test_symbolic_involution(self) -> None:
        x = Symbol("X")
        y = Symbol("Y")
        expr = BitXorExpr(BitXorExpr(x, y), y)
        simplified = expr.simplify()
        self.assertEqual(simplified, x)

    def test_symbolic_evaluation_env(self) -> None:
        x = Symbol("A")
        y = Symbol("B")
        expr = BitXorExpr(x, y)
        env = {"A": 0b1010, "B": 0b1100}
        self.assertEqual(expr.evaluate(env), 0b0110)


# ============================================================================
# Section 3: Deterministic Scalar Instruction Interpreter
# ============================================================================

class ScalarMachineState:
    """Register and memory state for scalar single-thread RDNA2 execution."""

    def __init__(self, lds_capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        self.vgprs: List[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: List[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSModel = LDSModel(capacity_dwords=lds_capacity_dwords)

    def get_vgpr(self, idx: int) -> int:
        if not (0 <= idx < DEFAULT_NUM_VGPRS):
            raise IndexError(f"VGPR index out of range: {idx}")
        return self.vgprs[idx]

    def set_vgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < DEFAULT_NUM_VGPRS):
            raise IndexError(f"VGPR index out of range: {idx}")
        self.vgprs[idx] = u32(val)

    def get_sgpr(self, idx: int) -> int:
        if not (0 <= idx < DEFAULT_NUM_SGPRS):
            raise IndexError(f"SGPR index out of range: {idx}")
        return self.sgprs[idx]

    def set_sgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < DEFAULT_NUM_SGPRS):
            raise IndexError(f"SGPR index out of range: {idx}")
        self.sgprs[idx] = u32(val)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for ds_xor_rtn_b32."""

    def resolve_operand(
        self,
        op: Union[VGPR, Immediate],
        state: ScalarMachineState,
    ) -> int:
        if isinstance(op, VGPR):
            return state.get_vgpr(op.index)
        elif isinstance(op, Immediate):
            return op.value
        raise TypeError(f"Unsupported operand type: {type(op)}")

    def step(
        self,
        instr: DSXorRtnB32Instruction,
        state: ScalarMachineState,
    ) -> Tuple[int, int]:
        """Execute a single ds_xor_rtn_b32 instruction deterministically.

        Latching behavior:
        Both vaddr and vdata0 operands are latched before memory is updated or
        the return value is written to vdst.
        """
        vaddr = state.get_vgpr(instr.addr.index)
        data = self.resolve_operand(instr.data0, state)

        # Atomic memory update
        old_val, new_val = state.lds.atomic_xor_b32(vaddr, data, offset=instr.offset)

        # Writeback to destination register
        state.set_vgpr(instr.dst.index, old_val)

        return old_val, new_val

    def execute_sequence(
        self,
        instructions: List[DSXorRtnB32Instruction],
        state: ScalarMachineState,
    ) -> List[Tuple[int, int]]:
        """Execute an ordered sequence of instructions."""
        return [self.step(instr, state) for instr in instructions]


class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()
        self.interp = ScalarInterpreter()

    def test_scalar_basic_execution(self) -> None:
        addr = 0x80
        init_mem = 0x1234_5678
        xor_val = 0x00FF_00FF
        self.state.lds.write_dword(addr, init_mem)
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, xor_val)

        instr = DSXorRtnB32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2))
        old_val, new_val = self.interp.step(instr, self.state)

        self.assertEqual(old_val, init_mem)
        self.assertEqual(new_val, init_mem ^ xor_val)
        self.assertEqual(self.state.get_vgpr(0), init_mem)
        self.assertEqual(self.state.lds.read_dword(addr), init_mem ^ xor_val)

    def test_scalar_with_immediate_operand(self) -> None:
        addr = 0x120
        init_mem = 0xF0F0_AAAA
        imm = 0x0F0F_5555
        self.state.lds.write_dword(addr, init_mem)
        self.state.set_vgpr(5, addr)

        instr = DSXorRtnB32Instruction(dst=VGPR(6), addr=VGPR(5), data0=Immediate(imm))
        old_val, new_val = self.interp.step(instr, self.state)

        self.assertEqual(old_val, init_mem)
        self.assertEqual(new_val, 0xFFFF_FFFF)
        self.assertEqual(self.state.get_vgpr(6), init_mem)
        self.assertEqual(self.state.lds.read_dword(addr), 0xFFFF_FFFF)

    def test_scalar_offset_applied(self) -> None:
        base_addr = 0x200
        offset = 0x1C
        eff_addr = base_addr + offset
        self.state.lds.write_dword(eff_addr, 0xCAFE_0000)
        self.state.set_vgpr(1, base_addr)
        self.state.set_vgpr(2, 0x0000_BABE)

        instr = DSXorRtnB32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2), offset=offset)
        old_val, new_val = self.interp.step(instr, self.state)

        self.assertEqual(old_val, 0xCAFE_0000)
        self.assertEqual(new_val, 0xCAFE_BABE)
        self.assertEqual(self.state.get_vgpr(0), 0xCAFE_0000)
        self.assertEqual(self.state.lds.read_dword(eff_addr), 0xCAFE_BABE)

    def test_scalar_deterministic_instruction_sequence(self) -> None:
        addr = 0x300
        self.state.lds.write_dword(addr, 0)
        self.state.set_vgpr(1, addr)

        # Sequence of 4 XORs toggling distinct byte lanes
        seq = [
            DSXorRtnB32Instruction(dst=VGPR(10), addr=VGPR(1), data0=Immediate(0x0000_00FF)),
            DSXorRtnB32Instruction(dst=VGPR(11), addr=VGPR(1), data0=Immediate(0x0000_FF00)),
            DSXorRtnB32Instruction(dst=VGPR(12), addr=VGPR(1), data0=Immediate(0x00FF_0000)),
            DSXorRtnB32Instruction(dst=VGPR(13), addr=VGPR(1), data0=Immediate(0xFF00_0000)),
        ]
        results = self.interp.execute_sequence(seq, self.state)

        self.assertEqual(results[0], (0x0000_0000, 0x0000_00FF))
        self.assertEqual(results[1], (0x0000_00FF, 0x0000_FFFF))
        self.assertEqual(results[2], (0x0000_FFFF, 0x00FF_FFFF))
        self.assertEqual(results[3], (0x00FF_FFFF, 0xFFFF_FFFF))

        self.assertEqual(self.state.get_vgpr(10), 0x0000_0000)
        self.assertEqual(self.state.get_vgpr(11), 0x0000_00FF)
        self.assertEqual(self.state.get_vgpr(12), 0x0000_FFFF)
        self.assertEqual(self.state.get_vgpr(13), 0x00FF_FFFF)
        self.assertEqual(self.state.lds.read_dword(addr), 0xFFFF_FFFF)

    def test_scalar_matches_oracle_across_bitmasks(self) -> None:
        addr = 0x400
        patterns = [
            (0x0000_0000, 0x0000_0000),
            (0xFFFF_FFFF, 0xFFFF_FFFF),
            (0x5555_5555, 0xAAAA_AAAA),
            (0x1234_5678, 0x9ABC_DEF0),
            (0xDEAD_BEEF, 0x0123_4567),
            (0x8000_0000, 0x7FFF_FFFF),
        ]
        for initial, mask in patterns:
            self.state.lds.write_dword(addr, initial)
            self.state.set_vgpr(1, addr)
            self.state.set_vgpr(2, mask)
            instr = DSXorRtnB32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2))
            old_interp, new_interp = self.interp.step(instr, self.state)
            old_oracle, new_oracle = oracle_ds_xor_rtn_b32(initial, mask)
            self.assertEqual(old_interp, old_oracle)
            self.assertEqual(new_interp, new_oracle)




if __name__ == "__main__":
    unittest.main()
