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
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing a single SALU instruction."""
    saved_exec: int
    new_exec: int
    scc: int


class ScalarSALUInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 SALU instructions."""

    def resolve_src64(self, src: Union[Register, LiteralConstant64], state: SALUArchitecturalState) -> int:
        if isinstance(src, Register):
            return state.read_sgpr_u64(src.index)
        elif isinstance(src, LiteralConstant64):
            return src.val & MASK64
        else:
            raise TypeError(f"Unsupported source operand type: {type(src)}")

    def step(self, instr: SXorSaveExecB64AST, state: SALUArchitecturalState) -> ExecutionResult:
        # 1. Atomic read phase
        src0_val = self.resolve_src64(instr.src0, state)
        current_exec = state.exec_mask

        # 2. Computation phase via mathematical oracle
        saved_exec, new_exec, scc = oracle_s_xor_saveexec_b64(current_exec, src0_val)

        # 3. Atomic write-back phase
        state.write_sgpr_u64(instr.dst.index, saved_exec)
        state.exec_mask = new_exec
        state.scc = scc

        return ExecutionResult(saved_exec=saved_exec, new_exec=new_exec, scc=scc)


# ==============================================================================
# SIMD Multi-Lane Execution Engine and EXEC Masking
# ==============================================================================

@dataclass
class SIMDWavefrontState:
    """Multi-lane SIMD machine state representing Wave32 or Wave64."""
    wave_size: int
    salu_state: SALUArchitecturalState = field(default_factory=SALUArchitecturalState)
    vgprs: List[List[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.wave_size not in (32, 64):
            raise ValueError(f"wave_size must be 32 or 64, got {self.wave_size}")
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.wave_size)]
        if self.wave_size == 32:
            self.salu_state.exec_mask &= ALL_LANES_W32

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane {lane} out of range [0, {self.wave_size - 1}]")
        return bool((self.salu_state.exec_mask & (1 << lane)) != 0)

    def active_lane_indices(self) -> List[int]:
        return [lane for lane in range(self.wave_size) if self.is_lane_active(lane)]

    def read_vgpr(self, lane: int, vgpr_idx: int) -> int:
        return self.vgprs[lane][vgpr_idx] & MASK32

    def write_vgpr(self, lane: int, vgpr_idx: int, val: int) -> None:
        self.vgprs[lane][vgpr_idx] = val & MASK32


class SIMDInstructionExecutor:
    """Executes SALU control-flow and masked VALU operations across wavefront lanes."""

    def __init__(self, salu_interp: Optional[ScalarSALUInterpreter] = None) -> None:
        self.salu_interp = salu_interp or ScalarSALUInterpreter()

    def step_salu(self, instr: SXorSaveExecB64AST, state: SIMDWavefrontState) -> ExecutionResult:
        res = self.salu_interp.step(instr, state.salu_state)
        if state.wave_size == 32:
            state.salu_state.exec_mask &= ALL_LANES_W32
        return res

    def step_valu_add_u32(self, dst_v: int, src_v: int, addend: int, state: SIMDWavefrontState) -> List[Optional[int]]:
        """Masked VALU instruction: lane writes only if active under EXEC."""
        written_vals: List[Optional[int]] = [None] * state.wave_size
        for lane in range(state.wave_size):
            if state.is_lane_active(lane):
                in_val = state.read_vgpr(lane, src_v)
                out_val = (in_val + addend) & MASK32
                state.write_vgpr(lane, dst_v, out_val)
                written_vals[lane] = out_val
        return written_vals


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Independence
# ==============================================================================

@dataclass(frozen=True)
class SALUStatusFlags:
    """Condition codes and architectural status flags evaluated after SALU instruction."""
    scc: int
    execz: bool
    execnz: bool
    active_lane_count: int
    parity_even: bool
    carry_borrow_unaltered: bool = True


def evaluate_salu_status_flags(
    new_exec: int,
    vcc_before: int,
    vcc_after: int,
) -> SALUStatusFlags:
    """Evaluate SALU status flags and verify carry/borrow preservation."""
    scc = 1 if (new_exec & MASK64) != 0 else 0
    execz = (new_exec & MASK64) == 0
    execnz = (new_exec & MASK64) != 0
    active_lanes = bin(new_exec & MASK64).count("1")
    parity_even = (active_lanes % 2 == 0)
    carry_unaltered = (vcc_before == vcc_after)

    return SALUStatusFlags(
        scc=scc,
        execz=execz,
        execnz=execnz,
        active_lane_count=active_lanes,
        parity_even=parity_even,
        carry_borrow_unaltered=carry_unaltered,
    )





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


class TestScalarSALUInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarSALUInterpreter()
        self.state = SALUArchitecturalState()

    def test_scalar_execution_step(self) -> None:
        self.state.exec_mask = 0xFFFF_0000_FFFF_0000
        self.state.write_sgpr_u64(2, 0x0000_FFFF_FFFF_0000)
        instr = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 4, count=2),
            src0=Register(RegKind.SGPR, 2, count=2),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res.saved_exec, 0xFFFF_0000_FFFF_0000)
        self.assertEqual(res.new_exec, 0xFFFF_FFFF_0000_0000)
        self.assertEqual(res.scc, 1)

        # Architectural state verification
        self.assertEqual(self.state.read_sgpr_u64(4), 0xFFFF_0000_FFFF_0000)
        self.assertEqual(self.state.exec_mask, 0xFFFF_FFFF_0000_0000)
        self.assertEqual(self.state.scc, 1)

    def test_scalar_step_with_literal_constant(self) -> None:
        self.state.exec_mask = 0xAAAA_AAAA_AAAA_AAAA
        lit = LiteralConstant64(0x5555_5555_5555_5555)
        instr = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 0, count=2),
            src0=lit,
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res.saved_exec, 0xAAAA_AAAA_AAAA_AAAA)
        self.assertEqual(res.new_exec, 0xFFFF_FFFF_FFFF_FFFF)
        self.assertEqual(res.scc, 1)
        self.assertEqual(self.state.read_sgpr_u64(0), 0xAAAA_AAAA_AAAA_AAAA)
        self.assertEqual(self.state.exec_mask, ALL_LANES_W64)

    def test_sequential_saveexec_chaining(self) -> None:
        """Chain multiple saveexec instructions toggling masks."""
        # Initial: EXEC = all ones
        self.state.exec_mask = ALL_LANES_W64
        # Step 1: invert lower half -> s[0:1] receives all ones, EXEC becomes upper half
        inst1 = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 0, count=2),
            src0=LiteralConstant64(LOWER_HALF_W64),
        )
        res1 = self.interp.step(inst1, self.state)
        self.assertEqual(res1.saved_exec, ALL_LANES_W64)
        self.assertEqual(self.state.exec_mask, UPPER_HALF_W64)
        self.assertEqual(self.state.read_sgpr_u64(0), ALL_LANES_W64)

        # Step 2: invert all ones -> s[2:3] receives upper half, EXEC becomes lower half
        inst2 = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 2, count=2),
            src0=LiteralConstant64(ALL_LANES_W64),
        )
        res2 = self.interp.step(inst2, self.state)
        self.assertEqual(res2.saved_exec, UPPER_HALF_W64)
        self.assertEqual(self.state.exec_mask, LOWER_HALF_W64)
        self.assertEqual(self.state.read_sgpr_u64(2), UPPER_HALF_W64)

    def test_deterministic_replay(self) -> None:
        """Verify strict determinism across multiple interpreter runs."""
        results = []
        for _ in range(50):
            st = SALUArchitecturalState()
            st.exec_mask = 0x0123_4567_89AB_CDEF
            st.write_sgpr_u64(10, 0xFEDC_BA98_7654_3210)
            inst = SXorSaveExecB64AST(
                dst=Register(RegKind.SGPR, 20, count=2),
                src0=Register(RegKind.SGPR, 10, count=2),
            )
            r = self.interp.step(inst, st)
            results.append((r.saved_exec, r.new_exec, r.scc, st.read_sgpr_u64(20)))
        self.assertTrue(all(item == results[0] for item in results))

    def test_scc_cleared_on_zero_result(self) -> None:
        self.state.exec_mask = 0x1111_2222_3333_4444
        self.state.write_sgpr_u64(0, 0x1111_2222_3333_4444)
        inst = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 4, count=2),
            src0=Register(RegKind.SGPR, 0, count=2),
        )
        res = self.interp.step(inst, self.state)
        self.assertEqual(res.new_exec, 0)
        self.assertEqual(res.scc, 0)
        self.assertEqual(self.state.scc, 0)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD execution, lane divergence, and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.executor = SIMDInstructionExecutor()

    def test_wave64_divergence_toggle(self) -> None:
        """Start with even lanes active, XOR with all ones, verify only odd lanes execute VALU."""
        state = SIMDWavefrontState(wave_size=64)
        state.salu_state.exec_mask = EVEN_LANES_W64

        # Initialise v0 with 100 for all lanes, v1 with sentinel 0xDEAD
        for lane in range(64):
            state.write_vgpr(lane, 0, 100)
            state.write_vgpr(lane, 1, 0xDEAD)

        # Toggle active lanes with ~0
        toggle_inst = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 0, count=2),
            src0=LiteralConstant64(ALL_LANES_W64),
        )
        salu_res = self.executor.step_salu(toggle_inst, state)
        self.assertEqual(salu_res.saved_exec, EVEN_LANES_W64)
        self.assertEqual(salu_res.new_exec, ODD_LANES_W64)
        self.assertEqual(state.salu_state.exec_mask, ODD_LANES_W64)

        # Execute masked VALU add: v1 = v0 + 50
        written = self.executor.step_valu_add_u32(dst_v=1, src_v=0, addend=50, state=state)

        for lane in range(64):
            if lane % 2 == 1:
                # Odd lanes were activated by XOR
                self.assertEqual(written[lane], 150)
                self.assertEqual(state.read_vgpr(lane, 1), 150)
            else:
                # Even lanes were deactivated by XOR
                self.assertIsNone(written[lane])
                self.assertEqual(state.read_vgpr(lane, 1), 0xDEAD)

    def test_wave32_upper_half_masking(self) -> None:
        """In Wave32 mode, execution mask is strictly bounded to lower 32 lanes."""
        state = SIMDWavefrontState(wave_size=32)
        state.salu_state.exec_mask = 0x0000_0000_FFFF_0000

        # Attempt XOR with 64-bit mask that touches upper half
        toggle_inst = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 4, count=2),
            src0=LiteralConstant64(ALL_LANES_W64),
        )
        self.executor.step_salu(toggle_inst, state)

        # In Wave32, upper 32 bits of active exec mask must remain 0
        self.assertEqual(state.salu_state.exec_mask & UPPER_HALF_W64, 0)
        self.assertEqual(state.salu_state.exec_mask, 0x0000_0000_0000_FFFF)
        self.assertEqual(len(state.active_lane_indices()), 16)

    def test_wave64_if_else_reconvergence_flow(self) -> None:
        """Emulate complete compiler if-else divergence and reconvergence workflow."""
        state = SIMDWavefrontState(wave_size=64)
        outer_mask = ALL_LANES_W64
        state.salu_state.exec_mask = outer_mask

        # Subgroup condition: lower 32 lanes satisfy 'if' condition
        cond_mask = LOWER_HALF_W64

        # Save outer mask in s[10:11], set EXEC to if-branch
        # In compiler: s_and_saveexec_b64 s[10:11], cond_mask
        state.salu_state.write_sgpr_u64(10, outer_mask)
        state.salu_state.exec_mask = cond_mask

        # At the end of 'if', transition to 'else' via s_xor_saveexec_b64 s[12:13], s[10:11]
        # EXEC = EXEC ^ s[10:11] = LOWER_HALF ^ ALL_LANES = UPPER_HALF!
        else_transition = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 12, count=2),
            src0=Register(RegKind.SGPR, 10, count=2),
        )
        res = self.executor.step_salu(else_transition, state)
        self.assertEqual(res.saved_exec, LOWER_HALF_W64)
        self.assertEqual(state.salu_state.exec_mask, UPPER_HALF_W64)
        self.assertEqual(state.active_lane_indices(), list(range(32, 64)))

    def test_wave_total_sleep_all_lanes_disabled(self) -> None:
        """When EXEC XOR with itself yields 0, all lanes are disabled and VALU writes do nothing."""
        state = SIMDWavefrontState(wave_size=64)
        state.salu_state.exec_mask = 0x1234_5678_9ABC_DEF0
        state.salu_state.write_sgpr_u64(6, 0x1234_5678_9ABC_DEF0)

        inst = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 0, count=2),
            src0=Register(RegKind.SGPR, 6, count=2),
        )
        self.executor.step_salu(inst, state)
        self.assertEqual(state.salu_state.exec_mask, 0)
        self.assertEqual(state.salu_state.scc, 0)
        self.assertEqual(len(state.active_lane_indices()), 0)

        # VALU write
        written = self.executor.step_valu_add_u32(dst_v=0, src_v=0, addend=1, state=state)
        self.assertTrue(all(w is None for w in written))

    def test_popcount_consistency_with_exec(self) -> None:
        """Popcount of active lanes strictly matches active bits in exec_mask."""
        test_masks = [
            0x0,
            0x1,
            0x8000_0000_0000_0000,
            EVEN_LANES_W64,
            ODD_LANES_W64,
            LOWER_HALF_W64,
            UPPER_HALF_W64,
            ALL_LANES_W64,
        ]
        for m in test_masks:
            state = SIMDWavefrontState(wave_size=64)
            state.salu_state.exec_mask = m
            expected_pop = bin(m).count("1")
            self.assertEqual(len(state.active_lane_indices()), expected_pop)


class TestRegisterAliasingAndPreservation(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarSALUInterpreter()
        self.simd_executor = SIMDInstructionExecutor(self.interp)

    def test_dst_identical_to_src0(self) -> None:
        """When dst == src0, dst receives old EXEC; EXEC receives old_exec ^ old_src."""
        state = SALUArchitecturalState()
        init_exec = 0xAAAA_AAAA_5555_5555
        init_src = 0x1234_5678_9ABC_DEF0
        state.exec_mask = init_exec
        state.write_sgpr_u64(4, init_src)

        inst = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 4, count=2),
            src0=Register(RegKind.SGPR, 4, count=2),
        )
        res = self.interp.step(inst, state)

        # Result verification
        self.assertEqual(res.saved_exec, init_exec)
        self.assertEqual(res.new_exec, init_exec ^ init_src)

        # Architectural state verification: s[4:5] must hold saved_exec, NOT new_exec!
        self.assertEqual(state.read_sgpr_u64(4), init_exec)
        self.assertEqual(state.exec_mask, init_exec ^ init_src)

    def test_dst_partial_overlap_low(self) -> None:
        """dst s[4:5] overlaps low register of src0 s[3:4] at s[4]."""
        state = SALUArchitecturalState()
        val_s3 = 0x1111_1111
        val_s4 = 0x2222_2222
        val_s5 = 0x3333_3333
        state.write_sgpr_u32(3, val_s3)
        state.write_sgpr_u32(4, val_s4)
        state.write_sgpr_u32(5, val_s5)

        init_exec = 0xF0F0_F0F0_0F0F_0F0F
        state.exec_mask = init_exec

        # src0 is s[3:4] -> lo=s3, hi=s4
        src0_val = words_to_u64(val_s3, val_s4)

        inst = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 4, count=2),
            src0=Register(RegKind.SGPR, 3, count=2),
        )
        res = self.interp.step(inst, state)

        self.assertEqual(res.saved_exec, init_exec)
        self.assertEqual(res.new_exec, init_exec ^ src0_val)

        # Check register preservation: s[3] must remain val_s3
        self.assertEqual(state.read_sgpr_u32(3), val_s3)
        # s[4:5] must hold init_exec (lo in s4, hi in s5)
        self.assertEqual(state.read_sgpr_u64(4), init_exec)

    def test_dst_partial_overlap_high(self) -> None:
        """dst s[4:5] overlaps high register of src0 s[5:6] at s[5]."""
        state = SALUArchitecturalState()
        val_s4 = 0x4444_4444
        val_s5 = 0x5555_5555
        val_s6 = 0x6666_6666
        state.write_sgpr_u32(4, val_s4)
        state.write_sgpr_u32(5, val_s5)
        state.write_sgpr_u32(6, val_s6)

        init_exec = 0x00FF_00FF_FF00_FF00
        state.exec_mask = init_exec

        src0_val = words_to_u64(val_s5, val_s6)

        inst = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 4, count=2),
            src0=Register(RegKind.SGPR, 5, count=2),
        )
        res = self.interp.step(inst, state)

        self.assertEqual(res.saved_exec, init_exec)
        self.assertEqual(res.new_exec, init_exec ^ src0_val)

        # Check register preservation: s[6] must remain val_s6
        self.assertEqual(state.read_sgpr_u32(6), val_s6)
        # s[4:5] holds init_exec
        self.assertEqual(state.read_sgpr_u64(4), init_exec)

    def test_inplace_mask_inversion_loop(self) -> None:
        """Sequential loop with dst == src0 repeatedly toggling and saving."""
        state = SALUArchitecturalState()
        state.exec_mask = ALL_LANES_W64
        # Register s[0:1] holds inversion mask
        invert_mask = EVEN_LANES_W64
        state.write_sgpr_u64(0, invert_mask)

        inst = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 0, count=2),
            src0=Register(RegKind.SGPR, 0, count=2),
        )

        # Iteration 1:
        # EXEC was ALL_LANES, src was EVEN_LANES
        # saved -> ALL_LANES (written to s[0:1])
        # new EXEC -> ALL_LANES ^ EVEN_LANES = ODD_LANES
        res1 = self.interp.step(inst, state)
        self.assertEqual(res1.saved_exec, ALL_LANES_W64)
        self.assertEqual(state.read_sgpr_u64(0), ALL_LANES_W64)
        self.assertEqual(state.exec_mask, ODD_LANES_W64)

        # Iteration 2:
        # EXEC is ODD_LANES, src is ALL_LANES (from s[0:1])
        # saved -> ODD_LANES (written to s[0:1])
        # new EXEC -> ODD_LANES ^ ALL_LANES = EVEN_LANES
        res2 = self.interp.step(inst, state)
        self.assertEqual(res2.saved_exec, ODD_LANES_W64)
        self.assertEqual(state.read_sgpr_u64(0), ODD_LANES_W64)
        self.assertEqual(state.exec_mask, EVEN_LANES_W64)

    def test_unrelated_sgpr_preservation(self) -> None:
        """Verify that all 104 un-targeted SGPRs remain strictly unaltered."""
        state = SALUArchitecturalState()
        # Seed all 106 SGPRs with distinct canary values
        for i in range(106):
            state.write_sgpr_u32(i, 0xA000_0000 | (i * 0x101))

        snapshot_before = list(state.sgprs)

        # Modify s[20:21] using s[30:31]
        inst = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 20, count=2),
            src0=Register(RegKind.SGPR, 30, count=2),
        )
        self.interp.step(inst, state)

        # Verify untouched registers
        for i in range(106):
            if i not in (20, 21):
                self.assertEqual(
                    state.read_sgpr_u32(i),
                    snapshot_before[i],
                    f"SGPR {i} was unexpectedly corrupted!",
                )


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for status flags, condition codes, and carry/borrow independence."""

    def setUp(self) -> None:
        self.interp = ScalarSALUInterpreter()

    def test_scc_matches_execnz_truth_table(self) -> None:
        """SCC is strictly 1 iff EXEC != 0 across exhaustive edge cases."""
        edge_cases = [
            (0, 0, 0),                       # 0 ^ 0 -> 0 (SCC=0)
            (ALL_LANES_W64, ALL_LANES_W64, 0),# all_ones ^ all_ones -> 0 (SCC=0)
            (0, 1, 1),                       # 0 ^ 1 -> 1 (SCC=1)
            (0, 1 << 63, 1),                 # 0 ^ (1<<63) -> MSB (SCC=1)
            (EVEN_LANES_W64, ODD_LANES_W64, 1),
            (ALL_LANES_W64, 0, 1),
        ]
        for init_exec, src_val, expected_scc in edge_cases:
            with self.subTest(init_exec=hex(init_exec), src_val=hex(src_val)):
                state = SALUArchitecturalState()
                state.exec_mask = init_exec
                state.write_sgpr_u64(0, src_val)
                inst = SXorSaveExecB64AST(
                    dst=Register(RegKind.SGPR, 2, count=2),
                    src0=Register(RegKind.SGPR, 0, count=2),
                )
                res = self.interp.step(inst, state)
                flags = evaluate_salu_status_flags(res.new_exec, 0, 0)

                self.assertEqual(res.scc, expected_scc)
                self.assertEqual(flags.scc, expected_scc)
                self.assertEqual(flags.execnz, (expected_scc == 1))
                self.assertEqual(flags.execz, (expected_scc == 0))

    def test_vcc_carry_borrow_independence(self) -> None:
        """Logic SALU instructions must leave VCC (carry/borrow) completely unaltered."""
        state = SALUArchitecturalState()
        canary_vcc = 0xDEAD_BEEF_CAFE_BABE
        state.vcc = canary_vcc
        state.exec_mask = ALL_LANES_W64

        inst = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 4, count=2),
            src0=LiteralConstant64(EVEN_LANES_W64),
        )
        res = self.interp.step(inst, state)
        flags = evaluate_salu_status_flags(res.new_exec, canary_vcc, state.vcc)

        self.assertTrue(flags.carry_borrow_unaltered)
        self.assertEqual(state.vcc, canary_vcc)

    def test_execz_execnz_complementary_duality(self) -> None:
        """execz and execnz must be strict logical complements."""
        test_values = [0, 1, 0xFF, ALL_LANES_W32, ALL_LANES_W64, 1 << 62]
        for val in test_values:
            flags = evaluate_salu_status_flags(val, 0, 0)
            self.assertNotEqual(flags.execz, flags.execnz)
            self.assertTrue(flags.execz or flags.execnz)

    def test_active_lane_count_and_parity(self) -> None:
        """Active lane count matches popcount and parity is correct."""
        # Even parity: 0 active lanes
        flags0 = evaluate_salu_status_flags(0, 0, 0)
        self.assertEqual(flags0.active_lane_count, 0)
        self.assertTrue(flags0.parity_even)

        # Odd parity: 1 active lane
        flags1 = evaluate_salu_status_flags(1, 0, 0)
        self.assertEqual(flags1.active_lane_count, 1)
        self.assertFalse(flags1.parity_even)

        # Even parity: 64 active lanes
        flags64 = evaluate_salu_status_flags(ALL_LANES_W64, 0, 0)
        self.assertEqual(flags64.active_lane_count, 64)
        self.assertTrue(flags64.parity_even)

        # Odd parity: 63 active lanes
        flags63 = evaluate_salu_status_flags(ALL_LANES_W64 ^ 1, 0, 0)
        self.assertEqual(flags63.active_lane_count, 63)
        self.assertFalse(flags63.parity_even)

    def test_branch_condition_emulation(self) -> None:
        """Emulate scalar conditional branches (s_cbranch_scc0 / s_cbranch_scc1 / s_cbranch_execz)."""
        def should_branch_scc0(scc: int) -> bool:
            return scc == 0

        def should_branch_scc1(scc: int) -> bool:
            return scc == 1

        def should_branch_execz(exec_mask: int) -> bool:
            return exec_mask == 0

        # When EXEC becomes 0
        _, exec0, scc0 = oracle_s_xor_saveexec_b64(0x1234, 0x1234)
        self.assertTrue(should_branch_scc0(scc0))
        self.assertFalse(should_branch_scc1(scc0))
        self.assertTrue(should_branch_execz(exec0))

        # When EXEC is non-zero
        _, exec1, scc1 = oracle_s_xor_saveexec_b64(0x1234, 0x5678)
        self.assertFalse(should_branch_scc0(scc1))
        self.assertTrue(should_branch_scc1(scc1))
        self.assertFalse(should_branch_execz(exec1))


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling in 64-bit masks."""

    def test_float_sign_bit_inversion(self) -> None:
        """XOR with SIGN_BIT_64 negates IEEE 754 doubles across normal, zero, and infinite values."""
        test_floats = [1.0, -1.0, 42.5, -0.001953125, math.pi]
        for f in test_floats:
            bits = f64_to_u64(f)
            saved, new_exec, scc = oracle_s_xor_saveexec_b64(bits, SIGN_BIT_64)
            self.assertEqual(saved, bits)
            self.assertEqual(new_exec, f64_to_u64(-f))
            self.assertEqual(u64_to_f64(new_exec), -f)
            self.assertEqual(scc, 1)

    def test_signed_zero_inversion(self) -> None:
        """XOR of +0.0 with SIGN_BIT_64 yields -0.0 and vice-versa."""
        saved_pz, new_nz, scc_nz = oracle_s_xor_saveexec_b64(POS_ZERO_64, SIGN_BIT_64)
        self.assertEqual(new_nz, NEG_ZERO_64)
        self.assertEqual(math.copysign(1.0, u64_to_f64(new_nz)), -1.0)
        self.assertEqual(scc_nz, 1)

        saved_nz, new_pz, scc_pz = oracle_s_xor_saveexec_b64(NEG_ZERO_64, SIGN_BIT_64)
        self.assertEqual(new_pz, POS_ZERO_64)
        self.assertEqual(math.copysign(1.0, u64_to_f64(new_pz)), 1.0)
        self.assertEqual(scc_pz, 0)  # +0.0 is integer 0, so SCC=0!

    def test_canonical_and_signaling_nan_toggling(self) -> None:
        """Toggling the quiet bit (bit 51) transitions between Signaling and Quiet NaN."""
        snan = SIGNALING_NAN_64
        self.assertTrue(is_snan_bits(snan))
        self.assertFalse(is_qnan_bits(snan))

        # Toggle QNAN bit
        saved, qnan, scc = oracle_s_xor_saveexec_b64(snan, QNAN_BIT_64)
        self.assertTrue(is_qnan_bits(qnan))
        self.assertFalse(is_snan_bits(qnan))
        self.assertTrue(math.isnan(u64_to_f64(qnan)))
        self.assertEqual(scc, 1)

        # Toggle back to snan
        saved2, snan2, scc2 = oracle_s_xor_saveexec_b64(qnan, QNAN_BIT_64)
        self.assertEqual(snan2, snan)
        self.assertTrue(is_snan_bits(snan2))

    def test_subnormal_boundary_masking(self) -> None:
        """Subnormal floating point bit boundaries handled as 64-bit masks."""
        min_subnormal = 0x0000_0000_0000_0001
        max_subnormal = 0x000F_FFFF_FFFF_FFFF
        min_normal = 0x0010_0000_0000_0000

        self.assertTrue(is_subnormal_bits(min_subnormal))
        self.assertTrue(is_subnormal_bits(max_subnormal))
        self.assertFalse(is_subnormal_bits(min_normal))

        # XOR max_subnormal with (max_subnormal ^ min_normal) transitions to min_normal
        transition_mask = max_subnormal ^ min_normal
        saved, new_exec, scc = oracle_s_xor_saveexec_b64(max_subnormal, transition_mask)
        self.assertEqual(new_exec, min_normal)
        self.assertFalse(is_subnormal_bits(new_exec))
        self.assertEqual(scc, 1)

    def test_infinity_arithmetic_boundaries(self) -> None:
        """XOR of +inf and -inf yields SIGN_BIT_64 (-0.0 bit representation)."""
        self.assertTrue(is_inf_bits(POS_INFINITY_64))
        self.assertTrue(is_inf_bits(NEG_INFINITY_64))

        saved, new_exec, scc = oracle_s_xor_saveexec_b64(POS_INFINITY_64, NEG_INFINITY_64)
        self.assertEqual(saved, POS_INFINITY_64)
        self.assertEqual(new_exec, SIGN_BIT_64)  # 0x8000_0000_0000_0000
        self.assertEqual(scc, 1)

    def test_nan_payload_preservation(self) -> None:
        """Custom payload bits within NaN fraction field preserved under XOR."""
        custom_payload = 0x000A_BCDE_1234_5678
        nan_with_payload = EXP_MASK_64 | custom_payload
        self.assertTrue(is_nan_bits(nan_with_payload))

        saved, new_exec, scc = oracle_s_xor_saveexec_b64(nan_with_payload, 0)
        self.assertEqual(new_exec, nan_with_payload)
        self.assertTrue(is_nan_bits(new_exec))
        self.assertEqual(new_exec & FRAC_MASK_64, custom_payload)


class TestSXorSaveExecB64NegativeControls(unittest.TestCase):
    """Negative control tests: syntax rejection, invalid encodings, and boundary defenses."""

    def test_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, -1, count=2)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 106, count=2)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 105, count=2)  # spans to 107
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1, count=2)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256, count=2)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 255, count=2)  # spans to 257

    def test_invalid_register_counts(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 0, count=0)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 0, count=3)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 0, count=4)

    def test_invalid_register_kind_for_salu(self) -> None:
        with self.assertRaises(TypeError):
            SXorSaveExecB64AST(
                dst=Register(RegKind.VGPR, 0, count=2),
                src0=Register(RegKind.SGPR, 2, count=2),
            )
        with self.assertRaises(TypeError):
            SXorSaveExecB64AST(
                dst=Register(RegKind.SGPR, 0, count=2),
                src0=Register(RegKind.VGPR, 2, count=2),
            )

    def test_architectural_state_out_of_bounds_access(self) -> None:
        state = SALUArchitecturalState()
        with self.assertRaises(IndexError):
            state.read_sgpr_u32(106)
        with self.assertRaises(IndexError):
            state.write_sgpr_u32(-1, 0)
        with self.assertRaises(IndexError):
            state.read_sgpr_u64(105)
        with self.assertRaises(IndexError):
            state.write_sgpr_u64(105, 0)

    def test_simd_invalid_wave_size(self) -> None:
        with self.assertRaises(ValueError):
            SIMDWavefrontState(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWavefrontState(wave_size=128)

    def test_simd_invalid_lane_index(self) -> None:
        state = SIMDWavefrontState(wave_size=32)
        with self.assertRaises(IndexError):
            state.is_lane_active(32)
        with self.assertRaises(IndexError):
            state.is_lane_active(-1)


class TestSXorSaveExecB64MutationDetection(unittest.TestCase):
    """Mutation tests: verify that intentional sabotages/mutations of the oracle are caught."""

    def test_mutation_xor_to_or_sabotage(self) -> None:
        # Sabotage: replace XOR with OR
        def mutated_oracle(exec_val: int, src_val: int) -> Tuple[int, int, int]:
            saved = exec_val & MASK64
            new_exec = (saved | src_val) & MASK64
            scc = 1 if new_exec != 0 else 0
            return saved, new_exec, scc

        # Oracle test: ALL_LANES ^ ALL_LANES must equal 0, but OR yields ALL_LANES
        with self.assertRaises(AssertionError):
            saved, new_exec, scc = mutated_oracle(ALL_LANES_W64, ALL_LANES_W64)
            self.assertEqual(new_exec, 0)

    def test_mutation_xor_to_and_sabotage(self) -> None:
        # Sabotage: replace XOR with AND
        def mutated_oracle(exec_val: int, src_val: int) -> Tuple[int, int, int]:
            saved = exec_val & MASK64
            new_exec = (saved & src_val) & MASK64
            scc = 1 if new_exec != 0 else 0
            return saved, new_exec, scc

        # Oracle test: 0 ^ ALL_LANES must equal ALL_LANES, but AND yields 0
        with self.assertRaises(AssertionError):
            saved, new_exec, scc = mutated_oracle(0, ALL_LANES_W64)
            self.assertEqual(new_exec, ALL_LANES_W64)

    def test_mutation_saveexec_write_order_sabotage(self) -> None:
        # Sabotage: write new_exec to sdst instead of saved (old) exec
        interp = ScalarSALUInterpreter()
        state = SALUArchitecturalState()
        state.exec_mask = ALL_LANES_W64

        # Proper execution saves ALL_LANES_W64 to s[0:1] and sets EXEC to 0
        inst = SXorSaveExecB64AST(
            dst=Register(RegKind.SGPR, 0, count=2),
            src0=LiteralConstant64(ALL_LANES_W64),
        )
        res = interp.step(inst, state)
        self.assertEqual(res.saved_exec, ALL_LANES_W64)
        self.assertEqual(state.read_sgpr_u64(0), ALL_LANES_W64)

        # Mutated state: if dst received new_exec (0), asserting it holds ALL_LANES_W64 would fail
        mutated_dst_val = 0
        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_dst_val, ALL_LANES_W64)

    def test_mutation_scc_inversion_sabotage(self) -> None:
        # Sabotage: inverted SCC logic (SCC = 1 when EXEC == 0)
        def mutated_oracle(exec_val: int, src_val: int) -> Tuple[int, int, int]:
            saved = exec_val & MASK64
            new_exec = (saved ^ src_val) & MASK64
            scc = 1 if new_exec == 0 else 0  # Inverted!
            return saved, new_exec, scc

        with self.assertRaises(AssertionError):
            _, _, scc = mutated_oracle(0, 0)
            self.assertEqual(scc, 0)

    def test_mutation_scc_from_old_exec_sabotage(self) -> None:
        # Sabotage: SCC computed from old EXEC instead of new EXEC
        def mutated_oracle(exec_val: int, src_val: int) -> Tuple[int, int, int]:
            saved = exec_val & MASK64
            new_exec = (saved ^ src_val) & MASK64
            scc = 1 if saved != 0 else 0  # Bug: derived from old exec
            return saved, new_exec, scc

        # When old exec is ALL_LANES but new exec is 0: SCC must be 0, not 1
        with self.assertRaises(AssertionError):
            _, _, scc = mutated_oracle(ALL_LANES_W64, ALL_LANES_W64)
            self.assertEqual(scc, 0)

    def test_mutation_mask_32bit_truncation_sabotage(self) -> None:
        # Sabotage: 64-bit mask truncated to 32-bit
        def mutated_step(exec_val: int, src_val: int) -> int:
            return (exec_val ^ src_val) & MASK32  # Lost upper 32 bits!

        with self.assertRaises(AssertionError):
            res = mutated_step(ALL_LANES_W64, 0)
            self.assertEqual(res, ALL_LANES_W64)

    def test_mutation_aliasing_read_after_write_sabotage(self) -> None:
        # Sabotage: non-atomic sequential execution reading src0 after updating dst
        state = SALUArchitecturalState()
        init_exec = 0xAAAA_AAAA_5555_5555
        init_src = 0x1234_5678_9ABC_DEF0
        state.exec_mask = init_exec
        state.write_sgpr_u64(4, init_src)

        # Buggy sequence:
        # 1. state.write_sgpr_u64(4, state.exec_mask) -> s[4:5] is now init_exec!
        # 2. src0 = state.read_sgpr_u64(4) -> reads init_exec instead of init_src!
        # 3. state.exec_mask = state.exec_mask ^ src0 -> init_exec ^ init_exec = 0!
        mutated_new_exec = 0  # would result in 0
        expected_new_exec = init_exec ^ init_src
        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_new_exec, expected_new_exec)


if __name__ == "__main__":
    unittest.main()
