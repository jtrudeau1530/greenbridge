from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

# ==============================================================================
# Architectural Constants and Masks
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF

# IEEE 754 double-precision bit manipulation constants
SIGN_BIT_64: int = 0x8000_0000_0000_0000
EXP_MASK_64: int = 0x7FF0_0000_0000_0000
FRAC_MASK_64: int = 0x000F_FFFF_FFFF_FFFF
QNAN_BIT_64: int = 0x0008_0000_0000_0000
CANONICAL_QNAN_64: int = 0x7FF8_0000_0000_0000
SIGNALING_NAN_64: int = 0x7FF0_0000_0000_0001
POS_INFINITY_64: int = 0x7FF0_0000_0000_0000
NEG_INFINITY_64: int = 0xFFF0_0000_0000_0000
POS_ZERO_64: int = 0x0000_0000_0000_0000
NEG_ZERO_64: int = 0x8000_0000_0000_0000

# Wavefront lane mask constants
ALL_LANES_W32: int = 0x0000_0000_FFFF_FFFF
ALL_LANES_W64: int = 0xFFFF_FFFF_FFFF_FFFF
EVEN_LANES_W64: int = 0x5555_5555_5555_5555
ODD_LANES_W64: int = 0xAAAA_AAAA_AAAA_AAAA
LOWER_HALF_W64: int = 0x0000_0000_FFFF_FFFF
UPPER_HALF_W64: int = 0xFFFF_FFFF_0000_0000


# ==============================================================================
# Bit-Level Conversion and Classification Helpers
# ==============================================================================

def u64_to_words(val: int) -> Tuple[int, int]:
    """Split a 64-bit unsigned integer into (low_dword, high_dword)."""
    return (val & MASK32), ((val >> 32) & MASK32)


def words_to_u64(lo: int, hi: int) -> int:
    """Combine (low_dword, high_dword) into a 64-bit unsigned integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def u64_to_f64(bits: int) -> float:
    """Reinterpret 64-bit integer bits as IEEE 754 binary64 float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_u64(val: float) -> int:
    """Reinterpret IEEE 754 binary64 float as 64-bit integer bits."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def is_nan_bits(bits: int) -> bool:
    """Return True if 64-bit bits represent IEEE 754 NaN."""
    return ((bits & EXP_MASK_64) == EXP_MASK_64) and ((bits & FRAC_MASK_64) != 0)


def is_snan_bits(bits: int) -> bool:
    """Return True if 64-bit bits represent IEEE 754 Signaling NaN."""
    return is_nan_bits(bits) and not bool(bits & QNAN_BIT_64)


def is_qnan_bits(bits: int) -> bool:
    """Return True if 64-bit bits represent IEEE 754 Quiet NaN."""
    return is_nan_bits(bits) and bool(bits & QNAN_BIT_64)


def is_inf_bits(bits: int) -> bool:
    """Return True if 64-bit bits represent IEEE 754 positive or negative infinity."""
    return ((bits & EXP_MASK_64) == EXP_MASK_64) and ((bits & FRAC_MASK_64) == 0)


def is_subnormal_bits(bits: int) -> bool:
    """Return True if 64-bit bits represent IEEE 754 subnormal float."""
    return ((bits & EXP_MASK_64) == 0) and ((bits & FRAC_MASK_64) != 0)


# ==============================================================================
# Reference Mathematical Oracle
# ==============================================================================

def oracle_s_xor_saveexec_b64(exec_mask: int, src0: int) -> Tuple[int, int, int]:
    """Reference mathematical oracle for RDNA2 s_xor_saveexec_b64 instruction.

    Architectural Operation:
        saved_exec = EXEC;
        EXEC = EXEC ^ src0;
        SCC = (EXEC != 0);

    Parameters:
        exec_mask: The 64-bit architectural EXEC mask prior to instruction execution.
        src0: The 64-bit source operand (SGPR pair, literal constant, or special register).

    Returns:
        A tuple of (saved_exec, new_exec, scc):
            - saved_exec: 64-bit unsigned integer copied to sdst.
            - new_exec: 64-bit unsigned integer written to EXEC.
            - scc: 1 if new_exec != 0, else 0.
    """
    saved_exec = exec_mask & MASK64
    new_exec = (saved_exec ^ (src0 & MASK64)) & MASK64
    scc = 1 if new_exec != 0 else 0
    return saved_exec, new_exec, scc


# ==============================================================================
# Architectural State Model
# ==============================================================================

@dataclass
class SALUArchitecturalState:
    """Architectural state for AMD RDNA2 Scalar ALU (SALU) execution.

    Holds the scalar register file (106 SGPRs standard in RDNA2),
    special registers (EXEC, VCC, SCC, M0), and status flags.
    """
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)
    exec_mask: int = field(default=MASK64)
    vcc: int = field(default=0)
    scc: int = field(default=0)
    m0: int = field(default=0)

    def read_sgpr_u32(self, index: int) -> int:
        if not (0 <= index < len(self.sgprs)):
            raise IndexError(f"SGPR index {index} out of range [0, {len(self.sgprs) - 1}]")
        return self.sgprs[index] & MASK32

    def write_sgpr_u32(self, index: int, val: int) -> None:
        if not (0 <= index < len(self.sgprs)):
            raise IndexError(f"SGPR index {index} out of range [0, {len(self.sgprs) - 1}]")
        self.sgprs[index] = val & MASK32

    def read_sgpr_u64(self, index: int) -> int:
        if not (0 <= index < len(self.sgprs) - 1):
            raise IndexError(f"SGPR pair base index {index} out of range [0, {len(self.sgprs) - 2}]")
        lo = self.read_sgpr_u32(index)
        hi = self.read_sgpr_u32(index + 1)
        return words_to_u64(lo, hi)

    def write_sgpr_u64(self, index: int, val: int) -> None:
        if not (0 <= index < len(self.sgprs) - 1):
            raise IndexError(f"SGPR pair base index {index} out of range [0, {len(self.sgprs) - 2}]")
        lo, hi = u64_to_words(val)
        self.write_sgpr_u32(index, lo)
        self.write_sgpr_u32(index + 1, hi)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegKind(Enum):
    SGPR = auto()
    VGPR = auto()
    SPECIAL = auto()


@dataclass(frozen=True)
class Register:
    """Architectural register representation with kind, base index, and count."""
    kind: RegKind
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        if self.kind == RegKind.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index {self.index} out of range [0, 105]")
            if self.index + self.count > 106:
                raise ValueError(f"SGPR register span {self.index} + {self.count} exceeds SGPR limit 106")
        elif self.kind == RegKind.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index {self.index} out of range [0, 255]")
            if self.index + self.count > 256:
                raise ValueError(f"VGPR register span {self.index} + {self.count} exceeds VGPR limit 256")
        if self.count not in (1, 2):
            raise ValueError(f"Register count must be 1 (32-bit) or 2 (64-bit), got {self.count}")

    @property
    def name(self) -> str:
        prefix = "s" if self.kind == RegKind.SGPR else "v"
        if self.count == 1:
            return f"{prefix}{self.index}"
        return f"{prefix}[{self.index}:{self.index + self.count - 1}]"


@dataclass(frozen=True)
class LiteralConstant64:
    """64-bit scalar immediate or inline literal constant."""
    val: int

    @property
    def name(self) -> str:
        return f"0x{self.val & MASK64:016x}"


class SALUOpcode(Enum):
    S_XOR_SAVEEXEC_B64 = auto()


@dataclass(frozen=True)
class SXorSaveExecB64AST:
    """Symbolic AST node representing RDNA2 s_xor_saveexec_b64 instruction."""
    dst: Register
    src0: Union[Register, LiteralConstant64]
    opcode: SALUOpcode = SALUOpcode.S_XOR_SAVEEXEC_B64

    def __post_init__(self) -> None:
        if self.dst.kind != RegKind.SGPR:
            raise TypeError(f"Destination of s_xor_saveexec_b64 must be SGPR, got {self.dst.kind}")
        if self.dst.count != 2:
            raise ValueError(f"Destination of s_xor_saveexec_b64 must be 64-bit register pair (count=2), got count={self.dst.count}")
        if isinstance(self.src0, Register):
            if self.src0.kind != RegKind.SGPR:
                raise TypeError(f"Register source of s_xor_saveexec_b64 must be SGPR, got {self.src0.kind}")
            if self.src0.count != 2:
                raise ValueError(f"Register source of s_xor_saveexec_b64 must be 64-bit register pair (count=2), got count={self.src0.count}")

    def disassemble(self) -> str:
        return f"s_xor_saveexec_b64 {self.dst.name}, {self.src0.name}"


# ==============================================================================
# Unit Tests: Phase 1 Oracle and Architectural Model
# ==============================================================================

class TestSXorSaveExecB64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_identity_zero_xor(self) -> None:
        """XOR with 0 preserves EXEC, returns old EXEC, and SCC reflects (EXEC != 0)."""
        exec_val = 0x1234_5678_9ABC_DEF0
        saved, new_exec, scc = oracle_s_xor_saveexec_b64(exec_val, 0)
        self.assertEqual(saved, exec_val)
        self.assertEqual(new_exec, exec_val)
        self.assertEqual(scc, 1)

    def test_identity_zero_exec_zero_src(self) -> None:
        """XOR 0 with 0 yields new_exec=0 and SCC=0."""
        saved, new_exec, scc = oracle_s_xor_saveexec_b64(0, 0)
        self.assertEqual(saved, 0)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)

    def test_invert_all_lanes_with_all_ones(self) -> None:
        """XOR with ~0 inverts all 64 bits of EXEC."""
        saved, new_exec, scc = oracle_s_xor_saveexec_b64(ALL_LANES_W64, ALL_LANES_W64)
        self.assertEqual(saved, ALL_LANES_W64)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)

        saved2, new_exec2, scc2 = oracle_s_xor_saveexec_b64(0, ALL_LANES_W64)
        self.assertEqual(saved2, 0)
        self.assertEqual(new_exec2, ALL_LANES_W64)
        self.assertEqual(scc2, 1)

    def test_toggle_alternating_lanes(self) -> None:
        """XOR with odd lane mask toggles between even and odd active lanes."""
        saved, new_exec, scc = oracle_s_xor_saveexec_b64(ALL_LANES_W64, ODD_LANES_W64)
        self.assertEqual(saved, ALL_LANES_W64)
        self.assertEqual(new_exec, EVEN_LANES_W64)
        self.assertEqual(scc, 1)

        saved2, new_exec2, scc2 = oracle_s_xor_saveexec_b64(new_exec, ALL_LANES_W64)
        self.assertEqual(saved2, EVEN_LANES_W64)
        self.assertEqual(new_exec2, ODD_LANES_W64)
        self.assertEqual(scc2, 1)

    def test_toggle_wave64_halves(self) -> None:
        """XOR lower half with upper half transitions wave halves."""
        saved, new_exec, scc = oracle_s_xor_saveexec_b64(LOWER_HALF_W64, ALL_LANES_W64)
        self.assertEqual(saved, LOWER_HALF_W64)
        self.assertEqual(new_exec, UPPER_HALF_W64)
        self.assertEqual(scc, 1)

    def test_scc_boundary_single_bit_set(self) -> None:
        """Single active bit (LSB or MSB) produces SCC=1."""
        saved_lsb, new_lsb, scc_lsb = oracle_s_xor_saveexec_b64(0, 1)
        self.assertEqual(new_lsb, 1)
        self.assertEqual(scc_lsb, 1)

        msb_mask = 1 << 63
        saved_msb, new_msb, scc_msb = oracle_s_xor_saveexec_b64(0, msb_mask)
        self.assertEqual(new_msb, msb_mask)
        self.assertEqual(scc_msb, 1)

    def test_word_packing_and_architectural_state(self) -> None:
        """Test SGPR 64-bit pair read/write and word decomposition."""
        state = SALUArchitecturalState()
        test_val = 0xCAFE_BABE_DEAD_BEEF
        state.write_sgpr_u64(4, test_val)
        self.assertEqual(state.read_sgpr_u64(4), test_val)
        self.assertEqual(state.read_sgpr_u32(4), 0xDEAD_BEEF)
        self.assertEqual(state.read_sgpr_u32(5), 0xCAFE_BABE)

        lo, hi = u64_to_words(test_val)
        self.assertEqual(words_to_u64(lo, hi), test_val)

    def test_self_cancellation_identity(self) -> None:
        """XOR of any value with itself produces 0 (EXEC ^ EXEC == 0)."""
        arbitrary_patterns = [
            0x1,
            0x8000_0000_0000_0000,
            0x5555_5555_AAAA_AAAA,
            0x0123_4567_89AB_CDEF,
            ALL_LANES_W64,
        ]
        for pat in arbitrary_patterns:
            saved, new_exec, scc = oracle_s_xor_saveexec_b64(pat, pat)
            self.assertEqual(saved, pat)
            self.assertEqual(new_exec, 0)
            self.assertEqual(scc, 0)


class TestSXorSaveExecB64AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_valid_sgpr_operands(self) -> None:
        dst = Register(RegKind.SGPR, 4, count=2)
        src = Register(RegKind.SGPR, 6, count=2)
        ast = SXorSaveExecB64AST(dst=dst, src0=src)
        self.assertEqual(ast.opcode, SALUOpcode.S_XOR_SAVEEXEC_B64)
        self.assertEqual(ast.dst.name, "s[4:5]")
        self.assertEqual(ast.src0.name, "s[6:7]")
        self.assertEqual(ast.disassemble(), "s_xor_saveexec_b64 s[4:5], s[6:7]")

    def test_ast_valid_literal_operand(self) -> None:
        dst = Register(RegKind.SGPR, 0, count=2)
        lit = LiteralConstant64(0x1234_5678_9ABC_DEF0)
        ast = SXorSaveExecB64AST(dst=dst, src0=lit)
        self.assertEqual(ast.disassemble(), "s_xor_saveexec_b64 s[0:1], 0x123456789abcdef0")

    def test_ast_dataclass_equality(self) -> None:
        ast1 = SXorSaveExecB64AST(Register(RegKind.SGPR, 8, count=2), LiteralConstant64(0x5555))
        ast2 = SXorSaveExecB64AST(Register(RegKind.SGPR, 8, count=2), LiteralConstant64(0x5555))
        self.assertEqual(ast1, ast2)
        self.assertEqual(hash(ast1), hash(ast2))

    def test_rejection_of_vgpr_dst(self) -> None:
        with self.assertRaises(TypeError):
            SXorSaveExecB64AST(
                dst=Register(RegKind.VGPR, 0, count=2),
                src0=Register(RegKind.SGPR, 2, count=2),
            )

    def test_rejection_of_vgpr_src(self) -> None:
        with self.assertRaises(TypeError):
            SXorSaveExecB64AST(
                dst=Register(RegKind.SGPR, 0, count=2),
                src0=Register(RegKind.VGPR, 2, count=2),
            )

    def test_rejection_of_single_32bit_dst_or_src(self) -> None:
        with self.assertRaises(ValueError):
            SXorSaveExecB64AST(
                dst=Register(RegKind.SGPR, 0, count=1),
                src0=Register(RegKind.SGPR, 2, count=2),
            )
        with self.assertRaises(ValueError):
            SXorSaveExecB64AST(
                dst=Register(RegKind.SGPR, 0, count=2),
                src0=Register(RegKind.SGPR, 2, count=1),
            )

    def test_rejection_of_out_of_range_sgpr(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, -1, count=2)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 106, count=2)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 105, count=2)  # 105+2 = 107 > 106


if __name__ == "__main__":
    unittest.main()
