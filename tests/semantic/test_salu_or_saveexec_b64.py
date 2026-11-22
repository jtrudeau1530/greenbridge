from __future__ import annotations

import math
import random
import struct
import sys
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

# ==============================================================================
# Architectural Constants for AMD RDNA2 / GCN Scalar ALU
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
SIGN_BIT64: int = 0x8000_0000_0000_0000
EXP_MASK64: int = 0x7FF0_0000_0000_0000
FRAC_MASK64: int = 0x000F_FFFF_FFFF_FFFF
CANONICAL_QNAN64: int = 0x7FF8_0000_0000_0000

ALL_LANES_ACTIVE_64: int = 0xFFFF_FFFF_FFFF_FFFF
NO_LANES_ACTIVE_64: int = 0x0000_0000_0000_0000
EVEN_LANES_64: int = 0x5555_5555_5555_5555
ODD_LANES_64: int = 0xAAAA_AAAA_AAAA_AAAA
LOWER_HALF_ACTIVE_64: int = 0x0000_0000_FFFF_FFFF
UPPER_HALF_ACTIVE_64: int = 0xFFFF_FFFF_0000_0000


# ==============================================================================
# Helper Conversions and Bitwise Primitives
# ==============================================================================

def u64(val: int) -> int:
    """Mask integer to architectural unsigned 64-bit bounds."""
    return val & MASK64


def u32(val: int) -> int:
    """Mask integer to architectural unsigned 32-bit bounds."""
    return val & MASK32


def split_u64_to_u32s(val: int) -> Tuple[int, int]:
    """Split 64-bit register/mask into (lo, hi) 32-bit words."""
    v = u64(val)
    lo = v & MASK32
    hi = (v >> 32) & MASK32
    return lo, hi


def combine_u32s_to_u64(lo: int, hi: int) -> int:
    """Combine (lo, hi) 32-bit words into a 64-bit integer."""
    return u64((u32(hi) << 32) | u32(lo))


def popcount64(val: int) -> int:
    """Return count of set bits (active execution lanes) in 64-bit mask."""
    return bin(u64(val)).count("1")


def f64_to_bits(val: float) -> int:
    """Convert a Python float to 64-bit IEEE 754 bitcast."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def bits_to_f64(bits: int) -> float:
    """Convert 64-bit integer bits to Python float."""
    return struct.unpack("<d", struct.pack("<Q", u64(bits)))[0]


# ==============================================================================
# Reference Mathematical Oracle for s_or_saveexec_b64
# ==============================================================================

def oracle_s_or_saveexec_b64(exec_in: int, src_val: int) -> Tuple[int, int, int]:
    """Reference mathematical oracle for AMD RDNA2 s_or_saveexec_b64.
    
    ISA Specification (AMD RDNA2 ISA: Scalar ALU Operations):
      Instruction: S_OR_SAVEEXEC_B64 SDEST, SSRC0
      Operation:
        SDEST = EXEC
        EXEC  = SSRC0 | EXEC
        SCC   = (EXEC != 0)
        
    Args:
      exec_in: The initial 64-bit execution mask (EXEC[63:0]).
      src_val: The 64-bit scalar operand from SSRC0.
      
    Returns:
      Tuple of (saved_sdst, new_exec, scc):
        saved_sdst: The previous EXEC value saved to SDEST.
        new_exec: The updated 64-bit EXEC mask (src_val | exec_in).
        scc: Scalar Condition Code, 1 if new_exec != 0 else 0.
    """
    saved_sdst = u64(exec_in)
    new_exec = u64(src_val | exec_in)
    scc = 1 if new_exec != 0 else 0
    return saved_sdst, new_exec, scc


# ==============================================================================
# Phase 1: Reference Mathematical Oracle & Architectural Model Tests
# ==============================================================================

class TestSOrSaveExecB64Oracle(unittest.TestCase):
    """Verifies mathematical invariants, identity laws, and RDNA2 oracle behavior."""

    def test_truth_table_per_bit(self) -> None:
        """Verify the 1-bit boolean truth table for OR logic across all lanes."""
        truth_table = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 1,
        }
        for (exec_bit, src_bit), expected_exec in truth_table.items():
            for bit_pos in (0, 1, 15, 31, 32, 47, 63):
                exec_in = exec_bit << bit_pos
                src_val = src_bit << bit_pos
                saved, new_exec, scc = oracle_s_or_saveexec_b64(exec_in, src_val)
                self.assertEqual(saved, exec_in)
                expected_val = expected_exec << bit_pos
                self.assertEqual(new_exec, expected_val)
                self.assertEqual(scc, 1 if expected_val != 0 else 0)

    def test_identity_zero(self) -> None:
        """OR with 0 leaves EXEC unchanged and saves original EXEC."""
        test_patterns = [
            NO_LANES_ACTIVE_64,
            ALL_LANES_ACTIVE_64,
            EVEN_LANES_64,
            ODD_LANES_64,
            LOWER_HALF_ACTIVE_64,
            UPPER_HALF_ACTIVE_64,
            0x1234_5678_9ABC_DEF0,
        ]
        for pat in test_patterns:
            saved, new_exec, scc = oracle_s_or_saveexec_b64(pat, 0)
            self.assertEqual(saved, pat)
            self.assertEqual(new_exec, pat)
            self.assertEqual(scc, 1 if pat != 0 else 0)

    def test_dominance_all_ones(self) -> None:
        """OR with ALL_LANES_ACTIVE forces new EXEC to all ones and SCC=1."""
        test_patterns = [
            NO_LANES_ACTIVE_64,
            ALL_LANES_ACTIVE_64,
            EVEN_LANES_64,
            ODD_LANES_64,
            0x0000_0000_0000_0001,
        ]
        for pat in test_patterns:
            saved, new_exec, scc = oracle_s_or_saveexec_b64(pat, ALL_LANES_ACTIVE_64)
            self.assertEqual(saved, pat)
            self.assertEqual(new_exec, ALL_LANES_ACTIVE_64)
            self.assertEqual(scc, 1)

    def test_superset_invariant(self) -> None:
        """New EXEC must always be a superset of original EXEC bits."""
        rng = random.Random(0x5A1E_0864)
        for _ in range(100):
            exec_in = rng.getrandbits(64)
            src_val = rng.getrandbits(64)
            saved, new_exec, scc = oracle_s_or_saveexec_b64(exec_in, src_val)
            self.assertEqual(saved, exec_in)
            # Invariant: all bits initially active in EXEC remain active
            self.assertEqual(new_exec & exec_in, exec_in)
            # Invariant: all bits active in SSRC0 become active
            self.assertEqual(new_exec & src_val, src_val)
            # Invariant: active lane count never decreases
            self.assertGreaterEqual(popcount64(new_exec), popcount64(exec_in))

    def test_idempotence(self) -> None:
        """When src == exec_in, new EXEC equals exec_in."""
        for pattern in [0x0, 0x1, 0x8000_0000_0000_0000, 0xDEAD_BEEF_CAFE_BABE, MASK64]:
            saved, new_exec, scc = oracle_s_or_saveexec_b64(pattern, pattern)
            self.assertEqual(saved, pattern)
            self.assertEqual(new_exec, pattern)
            self.assertEqual(scc, 1 if pattern != 0 else 0)

    def test_word_splitting_lo_hi(self) -> None:
        """Ensure 64-bit operation matches independent 32-bit (lo, hi) word operations."""
        rng = random.Random(0x3232_6464)
        for _ in range(50):
            exec_in = rng.getrandbits(64)
            src_val = rng.getrandbits(64)
            saved, new_exec, scc = oracle_s_or_saveexec_b64(exec_in, src_val)
            exec_lo, exec_hi = split_u64_to_u32s(exec_in)
            src_lo, src_hi = split_u64_to_u32s(src_val)
            expected_new_lo = exec_lo | src_lo
            expected_new_hi = exec_hi | src_hi
            new_lo, new_hi = split_u64_to_u32s(new_exec)
            self.assertEqual(new_lo, expected_new_lo)
            self.assertEqual(new_hi, expected_new_hi)
            self.assertEqual(combine_u32s_to_u64(new_lo, new_hi), new_exec)

    def test_scc_flag_truth_table(self) -> None:
        """SCC flag is 1 if and only if new EXEC has at least one active bit."""
        # 1. Both zero -> SCC == 0
        saved, new_exec, scc = oracle_s_or_saveexec_b64(0, 0)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)

        # 2. Exec non-zero, src zero -> SCC == 1
        saved, new_exec, scc = oracle_s_or_saveexec_b64(1, 0)
        self.assertEqual(scc, 1)

        # 3. Exec zero, src non-zero -> SCC == 1
        saved, new_exec, scc = oracle_s_or_saveexec_b64(0, 1 << 63)
        self.assertEqual(scc, 1)

        # 4. Both non-zero -> SCC == 1
        saved, new_exec, scc = oracle_s_or_saveexec_b64(EVEN_LANES_64, ODD_LANES_64)
        self.assertEqual(new_exec, MASK64)
        self.assertEqual(scc, 1)

    def test_walking_ones_oracle(self) -> None:
        """Test single-lane activations across all 64 lane positions."""
        for lane in range(64):
            single_bit = 1 << lane
            saved, new_exec, scc = oracle_s_or_saveexec_b64(0, single_bit)
            self.assertEqual(saved, 0)
            self.assertEqual(new_exec, single_bit)
            self.assertEqual(scc, 1)
            self.assertEqual(popcount64(new_exec), 1)



# ==============================================================================
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    SGPR = auto()
    EXEC = auto()
    VCC = auto()
    TTMP = auto()
    SPECIAL = auto()


@dataclass(frozen=True)
class RegPair64:
    """Architectural 64-bit scalar register pair (e.g. s[0:1], exec, vcc)."""
    index: int
    reg_type: RegType = RegType.SGPR
    custom_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 104):
                raise ValueError(f"SGPR index {self.index} out of bounds [0, 104]")
            if self.index % 2 != 0:
                raise ValueError(f"64-bit SGPR pair base index must be even-aligned, got {self.index}")
        elif self.reg_type == RegType.TTMP:
            if not (0 <= self.index <= 14):
                raise ValueError(f"TTMP index {self.index} out of bounds [0, 14]")
            if self.index % 2 != 0:
                raise ValueError(f"64-bit TTMP pair base index must be even-aligned, got {self.index}")

    @property
    def lo_reg(self) -> int:
        return self.index

    @property
    def hi_reg(self) -> int:
        return self.index + 1

    @property
    def is_aligned(self) -> bool:
        return (self.index % 2) == 0

    @property
    def is_exec(self) -> bool:
        return self.reg_type == RegType.EXEC

    @property
    def is_vcc(self) -> bool:
        return self.reg_type == RegType.VCC

    @property
    def name(self) -> str:
        if self.custom_name is not None:
            return self.custom_name
        if self.reg_type == RegType.SGPR:
            return f"s[{self.index}:{self.index + 1}]"
        elif self.reg_type == RegType.EXEC:
            return "exec"
        elif self.reg_type == RegType.VCC:
            return "vcc"
        elif self.reg_type == RegType.TTMP:
            return f"ttmp[{self.index}:{self.index + 1}]"
        return f"reg64_{self.index}"


@dataclass(frozen=True)
class Literal64:
    """64-bit scalar literal or inline integer constant."""
    bits: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bits", u64(self.bits))

    @classmethod
    def from_int(cls, val: int) -> Literal64:
        return cls(u64(val))

    @property
    def is_inline_constant(self) -> bool:
        # AMD RDNA2 inline constants: -16 to 64 or float constants like 0.0, 1.0, -1.0
        return self.bits == 0 or self.bits == MASK64 or (1 <= self.bits <= 64)

    @property
    def name(self) -> str:
        if self.bits == 0:
            return "0"
        elif self.bits == MASK64:
            return "-1"
        elif 1 <= self.bits <= 64:
            return str(self.bits)
        return f"0x{self.bits:016X}"


@dataclass(frozen=True)
class SaluOperand:
    """Typed SALU expression operand (Register Pair or 64-bit Literal)."""
    source: Union[RegPair64, Literal64]

    @property
    def is_reg(self) -> bool:
        return isinstance(self.source, RegPair64)

    @property
    def is_literal(self) -> bool:
        return isinstance(self.source, Literal64)

    def disassemble(self) -> str:
        return self.source.name


@dataclass(frozen=True)
class SOrSaveExecB64Instruction:
    """Symbolic AST node for S_OR_SAVEEXEC_B64."""
    dst: RegPair64
    src: SaluOperand

    def disassemble(self) -> str:
        return f"s_or_saveexec_b64 {self.dst.name}, {self.src.disassemble()}"

    def validate(self) -> None:
        if not self.dst.is_aligned and self.dst.reg_type in (RegType.SGPR, RegType.TTMP):
            raise ValueError(f"Unaligned destination register pair: {self.dst.name}")
        if self.src.is_reg:
            src_reg = self.src.source  # type: ignore[union-attr]
            if not src_reg.is_aligned and src_reg.reg_type in (RegType.SGPR, RegType.TTMP):
                raise ValueError(f"Unaligned source register pair: {src_reg.name}")

    @classmethod
    def parse(cls, line: str) -> SOrSaveExecB64Instruction:
        """Parse assembly text into typed AST instruction node."""
        cleaned = line.strip()
        if ";" in cleaned:
            cleaned = cleaned.split(";", 1)[0].strip()
        if not cleaned:
            raise ValueError("Empty instruction line")

        parts = cleaned.split(None, 1)
        mnemonic = parts[0].lower()
        if mnemonic != "s_or_saveexec_b64":
            raise ValueError(f"Unexpected mnemonic {mnemonic}, expected s_or_saveexec_b64")
        if len(parts) < 2:
            raise ValueError(f"Missing operands in: {line}")

        raw_ops = [op.strip() for op in parts[1].split(",")]
        if len(raw_ops) != 2:
            raise ValueError(f"Instruction requires exactly 2 operands, got {len(raw_ops)} in: {line}")

        dst_str, src_str = raw_ops[0], raw_ops[1]

        # Parse destination register
        dst_reg = cls._parse_reg(dst_str)

        # Parse source operand (reg or literal)
        src_op = cls._parse_operand(src_str)

        inst = cls(dst=dst_reg, src=src_op)
        inst.validate()
        return inst

    @classmethod
    def _parse_reg(cls, token: str) -> RegPair64:
        tok = token.lower()
        if tok == "exec":
            return RegPair64(index=126, reg_type=RegType.EXEC, custom_name="exec")
        if tok == "vcc":
            return RegPair64(index=106, reg_type=RegType.VCC, custom_name="vcc")
        if tok.startswith("s[") and tok.endswith("]"):
            inner = tok[2:-1]
            if ":" not in inner:
                raise ValueError(f"64-bit SGPR must specify range s[lo:hi], got {token}")
            lo_str, hi_str = inner.split(":", 1)
            lo, hi = int(lo_str), int(hi_str)
            if hi != lo + 1:
                raise ValueError(f"SGPR range must span 2 dwords s[lo:lo+1], got {token}")
            return RegPair64(index=lo, reg_type=RegType.SGPR)
        if tok.startswith("ttmp[") and tok.endswith("]"):
            inner = tok[5:-1]
            lo_str, hi_str = inner.split(":", 1)
            lo, hi = int(lo_str), int(hi_str)
            if hi != lo + 1:
                raise ValueError(f"TTMP range must span 2 dwords, got {token}")
            return RegPair64(index=lo, reg_type=RegType.TTMP)
        raise ValueError(f"Unrecognized register syntax: {token}")

    @classmethod
    def _parse_operand(cls, token: str) -> SaluOperand:
        tok = token.strip().lower()
        try:
            reg = cls._parse_reg(tok)
            return SaluOperand(source=reg)
        except ValueError:
            pass

        # Parse as integer literal
        if tok.startswith("0x") or tok.startswith("-0x"):
            val = int(tok, 16)
        else:
            val = int(tok, 10)
        return SaluOperand(source=Literal64.from_int(val))


class TestSymbolicAST(unittest.TestCase):
    """Verifies typed AST node creation, parsing, validation, and disassembly."""

    def test_reg_pair_creation_and_properties(self) -> None:
        r_sgpr = RegPair64(index=4, reg_type=RegType.SGPR)
        self.assertEqual(r_sgpr.name, "s[4:5]")
        self.assertEqual(r_sgpr.lo_reg, 4)
        self.assertEqual(r_sgpr.hi_reg, 5)
        self.assertTrue(r_sgpr.is_aligned)
        self.assertFalse(r_sgpr.is_exec)
        self.assertFalse(r_sgpr.is_vcc)

        r_exec = RegPair64(index=126, reg_type=RegType.EXEC, custom_name="exec")
        self.assertEqual(r_exec.name, "exec")
        self.assertTrue(r_exec.is_exec)

        r_vcc = RegPair64(index=106, reg_type=RegType.VCC, custom_name="vcc")
        self.assertEqual(r_vcc.name, "vcc")
        self.assertTrue(r_vcc.is_vcc)

    def test_reg_pair_alignment_validation(self) -> None:
        """Odd base index for 64-bit pair raises ValueError."""
        with self.assertRaises(ValueError):
            RegPair64(index=5, reg_type=RegType.SGPR)
        with self.assertRaises(ValueError):
            RegPair64(index=106, reg_type=RegType.SGPR)  # Out of SGPR bounds [0, 104]

    def test_literal64_formatting(self) -> None:
        lit_zero = Literal64.from_int(0)
        self.assertEqual(lit_zero.name, "0")
        self.assertTrue(lit_zero.is_inline_constant)

        lit_neg1 = Literal64.from_int(-1)
        self.assertEqual(lit_neg1.name, "-1")
        self.assertTrue(lit_neg1.is_inline_constant)

        lit_val = Literal64.from_int(0x1234_5678_9ABC_DEF0)
        self.assertEqual(lit_val.name, "0x123456789ABCDEF0")
        self.assertFalse(lit_val.is_inline_constant)

    def test_ast_disassembly_roundtrip(self) -> None:
        dst = RegPair64(index=10, reg_type=RegType.SGPR)
        src = SaluOperand(source=RegPair64(index=12, reg_type=RegType.SGPR))
        inst = SOrSaveExecB64Instruction(dst=dst, src=src)
        dis = inst.disassemble()
        self.assertEqual(dis, "s_or_saveexec_b64 s[10:11], s[12:13]")

        reparsed = SOrSaveExecB64Instruction.parse(dis)
        self.assertEqual(reparsed, inst)

    def test_ast_parse_various_operands(self) -> None:
        lines = [
            ("s_or_saveexec_b64 s[0:1], s[2:3]", "s[0:1]", "s[2:3]"),
            ("s_or_saveexec_b64 s[4:5], vcc", "s[4:5]", "vcc"),
            ("s_or_saveexec_b64 s[8:9], exec", "s[8:9]", "exec"),
            ("s_or_saveexec_b64 s[12:13], 0", "s[12:13]", "0"),
            ("s_or_saveexec_b64 s[14:15], -1", "s[14:15]", "-1"),
            ("s_or_saveexec_b64 s[16:17], 0x5555555555555555", "s[16:17]", "0x5555555555555555"),
        ]
        for line, exp_dst, exp_src in lines:
            inst = SOrSaveExecB64Instruction.parse(line)
            self.assertEqual(inst.dst.name, exp_dst)
            self.assertEqual(inst.src.disassemble(), exp_src)

    def test_invalid_syntax_rejection_in_ast(self) -> None:
        invalid_cases = [
            "s_and_saveexec_b64 s[0:1], s[2:3]",
            "s_or_saveexec_b64 s[0:1]",
            "s_or_saveexec_b64 s[0:1], s[2:3], s[4:5]",
            "s_or_saveexec_b64 s[1:2], s[4:5]",
            "s_or_saveexec_b64 v[0:1], s[2:3]",
        ]
        for inv in invalid_cases:
            with self.assertRaises(ValueError):
                SOrSaveExecB64Instruction.parse(inv)



# ==============================================================================
# Phase 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural state of the AMD RDNA2 Scalar ALU machine."""
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)
    exec_mask: int = ALL_LANES_ACTIVE_64
    vcc: int = 0
    scc: int = 0
    pc: int = 0

    def read_reg_pair(self, reg: RegPair64) -> int:
        """Read 64-bit value from SGPR pair, EXEC, or VCC."""
        if reg.is_exec:
            return u64(self.exec_mask)
        elif reg.is_vcc:
            return u64(self.vcc)
        elif reg.reg_type == RegType.SGPR:
            lo = self.sgprs[reg.lo_reg]
            hi = self.sgprs[reg.hi_reg]
            return combine_u32s_to_u64(lo, hi)
        raise ValueError(f"Unsupported register read: {reg.name}")

    def write_reg_pair(self, reg: RegPair64, val: int) -> None:
        """Write 64-bit value to SGPR pair, EXEC, or VCC."""
        v = u64(val)
        if reg.is_exec:
            self.exec_mask = v
        elif reg.is_vcc:
            self.vcc = v
        elif reg.reg_type == RegType.SGPR:
            lo, hi = split_u64_to_u32s(v)
            self.sgprs[reg.lo_reg] = lo
            self.sgprs[reg.hi_reg] = hi
        else:
            raise ValueError(f"Unsupported register write: {reg.name}")

    def read_operand(self, op: SaluOperand) -> int:
        """Read 64-bit value from a typed SALU operand."""
        if op.is_literal:
            return u64(op.source.bits)  # type: ignore[union-attr]
        elif op.is_reg:
            return self.read_reg_pair(op.source)  # type: ignore[arg-type]
        raise ValueError("Invalid operand type")


class ScalarInterpreter:
    """Deterministic step-by-step scalar instruction interpreter."""

    def __init__(self, initial_state: Optional[ScalarMachineState] = None) -> None:
        self.state = initial_state if initial_state is not None else ScalarMachineState()
        self.history: List[Tuple[int, int, int]] = []

    def step(self, inst: SOrSaveExecB64Instruction) -> Tuple[int, int, int]:
        """Execute a single S_OR_SAVEEXEC_B64 instruction deterministically.
        
        Hardware execution order:
          1. Latch source operand value
          2. Latch current EXEC
          3. Compute saved_sdst = EXEC
          4. Compute new_exec = src_val | EXEC
          5. Update SDEST with saved_sdst
          6. Update EXEC with new_exec
          7. Update SCC = (new_exec != 0)
        """
        src_val = self.state.read_operand(inst.src)
        current_exec = self.state.exec_mask

        saved_sdst, new_exec, scc = oracle_s_or_saveexec_b64(current_exec, src_val)

        # Latch destination register write
        self.state.write_reg_pair(inst.dst, saved_sdst)

        # Update architectural EXEC and SCC
        self.state.exec_mask = new_exec
        self.state.scc = scc
        self.state.pc += 4

        record = (saved_sdst, new_exec, scc)
        self.history.append(record)
        return record

    def run_program(self, instructions: Sequence[SOrSaveExecB64Instruction]) -> List[Tuple[int, int, int]]:
        """Run a sequence of instructions to completion."""
        results = []
        for inst in instructions:
            results.append(self.step(inst))
        return results


class TestScalarInterpreter(unittest.TestCase):
    """Verifies deterministic execution in the architectural scalar interpreter."""

    def test_single_step_execution(self) -> None:
        state = ScalarMachineState(exec_mask=0x0000_0000_FFFF_FFFF)
        interp = ScalarInterpreter(state)

        # s_or_saveexec_b64 s[4:5], 0xFFFF_0000_0000_0000
        inst = SOrSaveExecB64Instruction(
            dst=RegPair64(4, RegType.SGPR),
            src=SaluOperand(Literal64.from_int(0xFFFF_0000_0000_0000)),
        )
        saved, new_exec, scc = interp.step(inst)

        self.assertEqual(saved, 0x0000_0000_FFFF_FFFF)
        self.assertEqual(new_exec, 0xFFFF_0000_FFFF_FFFF)
        self.assertEqual(scc, 1)

        # Check architectural state
        self.assertEqual(state.exec_mask, 0xFFFF_0000_FFFF_FFFF)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.read_reg_pair(RegPair64(4, RegType.SGPR)), 0x0000_0000_FFFF_FFFF)

    def test_sequential_saveexec_chaining(self) -> None:
        """Test accumulation of active lanes across chained calls."""
        state = ScalarMachineState(exec_mask=0x1)
        interp = ScalarInterpreter(state)

        # Chain of 4 instructions setting bits 1, 2, 3, 4
        prog = [
            SOrSaveExecB64Instruction(RegPair64(0, RegType.SGPR), SaluOperand(Literal64.from_int(0x2))),
            SOrSaveExecB64Instruction(RegPair64(2, RegType.SGPR), SaluOperand(Literal64.from_int(0x4))),
            SOrSaveExecB64Instruction(RegPair64(4, RegType.SGPR), SaluOperand(Literal64.from_int(0x8))),
            SOrSaveExecB64Instruction(RegPair64(6, RegType.SGPR), SaluOperand(Literal64.from_int(0x10))),
        ]
        history = interp.run_program(prog)

        self.assertEqual(history[0], (0x1, 0x3, 1))
        self.assertEqual(history[1], (0x3, 0x7, 1))
        self.assertEqual(history[2], (0x7, 0xF, 1))
        self.assertEqual(history[3], (0xF, 0x1F, 1))

        # Check that saved registers contain exact previous states
        self.assertEqual(state.read_reg_pair(RegPair64(0, RegType.SGPR)), 0x1)
        self.assertEqual(state.read_reg_pair(RegPair64(2, RegType.SGPR)), 0x3)
        self.assertEqual(state.read_reg_pair(RegPair64(4, RegType.SGPR)), 0x7)
        self.assertEqual(state.read_reg_pair(RegPair64(6, RegType.SGPR)), 0xF)
        self.assertEqual(state.exec_mask, 0x1F)

    def test_zero_to_active_transition(self) -> None:
        """Starting with EXEC=0 and SCC=0, transition to active."""
        state = ScalarMachineState(exec_mask=0, scc=0)
        interp = ScalarInterpreter(state)

        inst = SOrSaveExecB64Instruction(
            dst=RegPair64(0, RegType.SGPR),
            src=SaluOperand(Literal64.from_int(0x8000_0000_0000_0000)),
        )
        saved, new_exec, scc = interp.step(inst)

        self.assertEqual(saved, 0)
        self.assertEqual(new_exec, 0x8000_0000_0000_0000)
        self.assertEqual(scc, 1)
        self.assertEqual(state.scc, 1)

    def test_vcc_source_operand(self) -> None:
        """Read mask from VCC into EXEC while saving old EXEC."""
        state = ScalarMachineState(exec_mask=0x00FF_00FF_00FF_00FF, vcc=0xFF00_FF00_FF00_FF00)
        interp = ScalarInterpreter(state)

        inst = SOrSaveExecB64Instruction(
            dst=RegPair64(8, RegType.SGPR),
            src=SaluOperand(RegPair64(106, RegType.VCC, custom_name="vcc")),
        )
        saved, new_exec, scc = interp.step(inst)

        self.assertEqual(saved, 0x00FF_00FF_00FF_00FF)
        self.assertEqual(new_exec, 0xFFFF_FFFF_FFFF_FFFF)
        self.assertEqual(scc, 1)



# ==============================================================================
# Phase 4: SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDWaveState:
    """SIMD wavefront state modeling per-lane execution masking and vector registers."""
    wave_size: int = 64
    scalar_state: ScalarMachineState = field(default_factory=ScalarMachineState)
    vgprs: List[List[int]] = field(default_factory=lambda: [[0] * 256 for _ in range(64)])

    def __post_init__(self) -> None:
        if self.wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size {self.wave_size}, must be 32 or 64")
        if len(self.vgprs) != self.wave_size:
            self.vgprs = [[0] * 256 for _ in range(self.wave_size)]

    @property
    def active_mask(self) -> int:
        limit = (1 << self.wave_size) - 1
        return self.scalar_state.exec_mask & limit

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            return False
        return bool((self.active_mask >> lane) & 1)

    def active_lanes(self) -> List[int]:
        return [i for i in range(self.wave_size) if self.is_lane_active(i)]

    def inactive_lanes(self) -> List[int]:
        return [i for i in range(self.wave_size) if not self.is_lane_active(i)]

    def execute_vector_mov(self, dst_vgpr: int, val: int) -> None:
        """v_mov_b32: writes value only to lanes currently enabled by EXEC."""
        for lane in range(self.wave_size):
            if self.is_lane_active(lane):
                self.vgprs[lane][dst_vgpr] = u32(val)

    def execute_vector_add(self, dst_vgpr: int, src0_vgpr: int, src1_vgpr: int) -> None:
        """v_add_u32: adds src0 and src1 only for active lanes."""
        for lane in range(self.wave_size):
            if self.is_lane_active(lane):
                a = self.vgprs[lane][src0_vgpr]
                b = self.vgprs[lane][src1_vgpr]
                self.vgprs[lane][dst_vgpr] = u32(a + b)


class SIMDInterpreter:
    """Coordinates scalar control flow with SIMD vector lane divergence."""

    def __init__(self, wave_state: Optional[SIMDWaveState] = None) -> None:
        self.wave = wave_state if wave_state is not None else SIMDWaveState()
        self.scalar_interp = ScalarInterpreter(self.wave.scalar_state)

    def step_scalar(self, inst: SOrSaveExecB64Instruction) -> Tuple[int, int, int]:
        return self.scalar_interp.step(inst)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Verifies SIMD multi-lane divergence and EXEC masking under s_or_saveexec_b64."""

    def test_active_inactive_lane_masking(self) -> None:
        """Vector ops write only to active lanes; inactive lanes preserve old values."""
        wave = SIMDWaveState(wave_size=64)
        wave.scalar_state.exec_mask = EVEN_LANES_64  # Even lanes active (32 lanes active)

        # Initialize v0 with 0xAAAA across all lanes
        for lane in range(64):
            wave.vgprs[lane][0] = 0xAAAA

        # Execute vector mov with 0x5555
        wave.execute_vector_mov(dst_vgpr=0, val=0x5555)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[lane][0], 0x5555, f"Even lane {lane} should be updated")
            else:
                self.assertEqual(wave.vgprs[lane][0], 0xAAAA, f"Odd lane {lane} should remain untouched")

    def test_lane_activation_expansion_via_or(self) -> None:
        """s_or_saveexec_b64 expands active lanes and saves old execution mask."""
        wave = SIMDWaveState(wave_size=64)
        simd = SIMDInterpreter(wave)

        # Initially, only lower 32 lanes active
        wave.scalar_state.exec_mask = LOWER_HALF_ACTIVE_64
        self.assertEqual(len(wave.active_lanes()), 32)

        # Vector write in lower half
        wave.execute_vector_mov(dst_vgpr=1, val=0x1111)

        # Expand EXEC to include upper 32 lanes via s_or_saveexec_b64 s[4:5], UPPER_HALF
        inst = SOrSaveExecB64Instruction(
            dst=RegPair64(4, RegType.SGPR),
            src=SaluOperand(Literal64.from_int(UPPER_HALF_ACTIVE_64)),
        )
        saved, new_exec, scc = simd.step_scalar(inst)

        self.assertEqual(saved, LOWER_HALF_ACTIVE_64)
        self.assertEqual(new_exec, ALL_LANES_ACTIVE_64)
        self.assertEqual(scc, 1)
        self.assertEqual(len(wave.active_lanes()), 64)

        # Vector write in all 64 lanes
        wave.execute_vector_mov(dst_vgpr=2, val=0x2222)
        for lane in range(64):
            self.assertEqual(wave.vgprs[lane][2], 0x2222)

    def test_divergent_control_flow_merge(self) -> None:
        """Simulate divergent execution join: Path A (lanes 0-15) merges with Path B (lanes 16-31)."""
        wave = SIMDWaveState(wave_size=64)
        simd = SIMDInterpreter(wave)

        mask_a = 0x0000_0000_0000_FFFF  # Lanes 0..15
        mask_b = 0x0000_0000_FFFF_0000  # Lanes 16..31

        wave.scalar_state.exec_mask = mask_a
        wave.execute_vector_mov(dst_vgpr=5, val=100)

        # Merge Path B lanes into EXEC
        inst = SOrSaveExecB64Instruction(
            dst=RegPair64(0, RegType.SGPR),
            src=SaluOperand(Literal64.from_int(mask_b)),
        )
        saved, new_exec, scc = simd.step_scalar(inst)

        self.assertEqual(saved, mask_a)
        self.assertEqual(new_exec, mask_a | mask_b)
        self.assertEqual(scc, 1)
        self.assertEqual(len(wave.active_lanes()), 32)

    def test_wavefront_quadrant_activation(self) -> None:
        """Step-by-step activation of 16-lane quadrants."""
        wave = SIMDWaveState(wave_size=64)
        simd = SIMDInterpreter(wave)
        wave.scalar_state.exec_mask = 0

        q0 = 0x0000_0000_0000_FFFF
        q1 = 0x0000_0000_FFFF_0000
        q2 = 0x0000_FFFF_0000_0000
        q3 = 0xFFFF_0000_0000_0000

        quadrants = [q0, q1, q2, q3]
        expected_counts = [16, 32, 48, 64]

        for i, (q, count) in enumerate(zip(quadrants, expected_counts)):
            inst = SOrSaveExecB64Instruction(
                dst=RegPair64(i * 2, RegType.SGPR),
                src=SaluOperand(Literal64.from_int(q)),
            )
            simd.step_scalar(inst)
            self.assertEqual(len(wave.active_lanes()), count)

        self.assertEqual(wave.active_mask, ALL_LANES_ACTIVE_64)

    def test_wave32_and_wave64_execution_boundaries(self) -> None:
        """Verify active mask semantics when wave_size=32 versus wave_size=64."""
        wave32 = SIMDWaveState(wave_size=32)
        wave32.scalar_state.exec_mask = ALL_LANES_ACTIVE_64
        # In Wave32 mode, active mask only considers lower 32 bits
        self.assertEqual(wave32.active_mask, 0xFFFF_FFFF)
        self.assertEqual(len(wave32.active_lanes()), 32)
        self.assertFalse(wave32.is_lane_active(32))

        wave64 = SIMDWaveState(wave_size=64)
        wave64.scalar_state.exec_mask = ALL_LANES_ACTIVE_64
        self.assertEqual(wave64.active_mask, ALL_LANES_ACTIVE_64)
        self.assertEqual(len(wave64.active_lanes()), 64)
        self.assertTrue(wave64.is_lane_active(32))
        self.assertTrue(wave64.is_lane_active(63))



# ==============================================================================
# Phase 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Verifies architectural register aliasing, latching, and unwritten register preservation."""

    def test_dst_equals_src_aliasing(self) -> None:
        """When dst == src (e.g. s_or_saveexec_b64 s[4:5], s[4:5]):
        
        Hardware semantics:
          - src operand value is read (val_src)
          - old EXEC is latched (old_exec)
          - new EXEC = val_src | old_exec
          - dst receives old_exec
        """
        state = ScalarMachineState(exec_mask=0x1111_2222_3333_4444)
        interp = ScalarInterpreter(state)

        # Set s[4:5] to 0x5555_6666_7777_8888
        reg_s4 = RegPair64(4, RegType.SGPR)
        state.write_reg_pair(reg_s4, 0x5555_6666_7777_8888)

        # Execute aliased instruction
        inst = SOrSaveExecB64Instruction(
            dst=reg_s4,
            src=SaluOperand(reg_s4),
        )
        saved, new_exec, scc = interp.step(inst)

        # Saved sdst must be old EXEC
        self.assertEqual(saved, 0x1111_2222_3333_4444)
        self.assertEqual(state.read_reg_pair(reg_s4), 0x1111_2222_3333_4444)

        # New EXEC must be val_src | old_exec
        expected_exec = 0x5555_6666_7777_8888 | 0x1111_2222_3333_4444
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(state.exec_mask, expected_exec)
        self.assertEqual(scc, 1)

    def test_dst_is_exec_aliasing(self) -> None:
        """When dst is EXEC (s_or_saveexec_b64 exec, s[2:3]):
        
        Hardware semantics:
          - SDEST is EXEC, but EXEC is subsequently updated with src | EXEC.
          - Final EXEC is src | old_exec.
        """
        state = ScalarMachineState(exec_mask=0x0000_FFFF_0000_FFFF)
        interp = ScalarInterpreter(state)

        reg_s2 = RegPair64(2, RegType.SGPR)
        state.write_reg_pair(reg_s2, 0xFFFF_0000_FFFF_0000)

        reg_exec = RegPair64(126, RegType.EXEC, custom_name="exec")
        inst = SOrSaveExecB64Instruction(
            dst=reg_exec,
            src=SaluOperand(reg_s2),
        )
        saved, new_exec, scc = interp.step(inst)

        self.assertEqual(saved, 0x0000_FFFF_0000_FFFF)
        self.assertEqual(new_exec, ALL_LANES_ACTIVE_64)
        self.assertEqual(state.exec_mask, ALL_LANES_ACTIVE_64)
        self.assertEqual(scc, 1)

    def test_src_is_exec_aliasing(self) -> None:
        """When src is EXEC (s_or_saveexec_b64 s[6:7], exec):
        
        Hardware semantics:
          - SDEST receives old EXEC
          - New EXEC = EXEC | EXEC = EXEC
          - EXEC remains unchanged, SCC = (EXEC != 0)
        """
        initial_exec = 0x1234_5678_9ABC_DEF0
        state = ScalarMachineState(exec_mask=initial_exec)
        interp = ScalarInterpreter(state)

        reg_s6 = RegPair64(6, RegType.SGPR)
        reg_exec = RegPair64(126, RegType.EXEC, custom_name="exec")

        inst = SOrSaveExecB64Instruction(
            dst=reg_s6,
            src=SaluOperand(reg_exec),
        )
        saved, new_exec, scc = interp.step(inst)

        self.assertEqual(saved, initial_exec)
        self.assertEqual(new_exec, initial_exec)
        self.assertEqual(state.read_reg_pair(reg_s6), initial_exec)
        self.assertEqual(state.exec_mask, initial_exec)
        self.assertEqual(scc, 1)

    def test_dst_and_src_both_exec_aliasing(self) -> None:
        """When dst and src are both EXEC (s_or_saveexec_b64 exec, exec):"""
        initial_exec = 0xAAAA_5555_AAAA_5555
        state = ScalarMachineState(exec_mask=initial_exec)
        interp = ScalarInterpreter(state)

        reg_exec = RegPair64(126, RegType.EXEC, custom_name="exec")
        inst = SOrSaveExecB64Instruction(
            dst=reg_exec,
            src=SaluOperand(reg_exec),
        )
        saved, new_exec, scc = interp.step(inst)

        self.assertEqual(saved, initial_exec)
        self.assertEqual(new_exec, initial_exec)
        self.assertEqual(state.exec_mask, initial_exec)
        self.assertEqual(scc, 1)

    def test_unwritten_registers_preserved(self) -> None:
        """Verify that registers not targeted by dst remain completely unchanged."""
        state = ScalarMachineState()
        rng = random.Random(0xDEAD_CAFE)

        # Populate all 106 SGPRs with random values
        for i in range(106):
            state.sgprs[i] = rng.getrandbits(32)

        snapshot_sgprs = list(state.sgprs)
        target_pair = RegPair64(20, RegType.SGPR)

        interp = ScalarInterpreter(state)
        inst = SOrSaveExecB64Instruction(
            dst=target_pair,
            src=SaluOperand(Literal64.from_int(0x1234)),
        )
        interp.step(inst)

        # Check that only s20 and s21 changed; all other 104 SGPRs must be bit-for-bit identical
        for i in range(106):
            if i in (20, 21):
                continue
            self.assertEqual(state.sgprs[i], snapshot_sgprs[i], f"SGPR s{i} was unexpectedly modified")



# ==============================================================================
# Phase 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass
class ScalarStatusFlags:
    """Architectural STATUS register flags for AMD RDNA2 SALU."""
    scc: int = 0
    priv: int = 0
    trap_en: int = 0
    spi_prio: int = 0
    carry_borrow: int = 0

    def update_from_salu_or(self, new_exec: int) -> None:
        """Update SCC from resulting EXEC mask.
        
        Logical bitwise operations in RDNA2 SALU do not generate carry or borrow.
        All other status flags must remain untouched.
        """
        self.scc = 1 if (u64(new_exec) != 0) else 0
        self.carry_borrow = 0

    def evaluate_cbranch_scc0(self) -> bool:
        """s_cbranch_scc0: taken if SCC == 0 (all lanes inactive)."""
        return self.scc == 0

    def evaluate_cbranch_scc1(self) -> bool:
        """s_cbranch_scc1: taken if SCC == 1 (at least one lane active)."""
        return self.scc == 1


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Verifies SCC condition flag generation, carry/borrow neutrality, and cbranch semantics."""

    def test_scc_zero_flag_behavior(self) -> None:
        """EXEC == 0 results in SCC == 0."""
        flags = ScalarStatusFlags()
        flags.update_from_salu_or(0)
        self.assertEqual(flags.scc, 0)
        self.assertTrue(flags.evaluate_cbranch_scc0())
        self.assertFalse(flags.evaluate_cbranch_scc1())

    def test_scc_nonzero_flag_behavior(self) -> None:
        """Any bit set in EXEC[63:0] results in SCC == 1."""
        flags = ScalarStatusFlags()
        for lane in range(64):
            flags.update_from_salu_or(1 << lane)
            self.assertEqual(flags.scc, 1, f"Lane {lane} set should produce SCC=1")
            self.assertFalse(flags.evaluate_cbranch_scc0())
            self.assertTrue(flags.evaluate_cbranch_scc1())

    def test_no_carry_borrow_generation(self) -> None:
        """s_or_saveexec_b64 is a bitwise logical op and must never generate carry or borrow."""
        flags = ScalarStatusFlags(carry_borrow=1)
        # Even with all bits set (0xFFFF_FFFF_FFFF_FFFF | 0xFFFF_FFFF_FFFF_FFFF), no carry is generated
        flags.update_from_salu_or(ALL_LANES_ACTIVE_64)
        self.assertEqual(flags.carry_borrow, 0)
        self.assertEqual(flags.scc, 1)

    def test_status_flags_preservation(self) -> None:
        """Verify non-SCC status register bits (priv, trap_en, spi_prio) are preserved."""
        flags = ScalarStatusFlags(priv=1, trap_en=1, spi_prio=3)
        flags.update_from_salu_or(0x1234)
        self.assertEqual(flags.scc, 1)
        self.assertEqual(flags.priv, 1)
        self.assertEqual(flags.trap_en, 1)
        self.assertEqual(flags.spi_prio, 3)

    def test_scc_and_execz_duality(self) -> None:
        """scc == 0 is mathematically identical to exec_mask == 0 after s_or_saveexec_b64."""
        test_masks = [
            0x0,
            0x1,
            0x8000_0000_0000_0000,
            0x0000_FFFF_0000_0000,
            ALL_LANES_ACTIVE_64,
        ]
        flags = ScalarStatusFlags()
        for mask in test_masks:
            flags.update_from_salu_or(mask)
            is_exec_zero = (mask == 0)
            self.assertEqual(flags.scc == 0, is_exec_zero)
            self.assertEqual(flags.scc == 1, not is_exec_zero)



# ==============================================================================
# Phase 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Verifies IEEE 754 bitcasts, subnormals, NaNs, and extreme bit pattern handling in SALU."""

    def test_ieee754_zero_and_neg_zero_masks(self) -> None:
        """+0.0 is all zeros (identity); -0.0 has only MSB (sign bit, lane 63) set."""
        pos_zero_bits = f64_to_bits(0.0)
        neg_zero_bits = f64_to_bits(-0.0)

        self.assertEqual(pos_zero_bits, 0x0000_0000_0000_0000)
        self.assertEqual(neg_zero_bits, 0x8000_0000_0000_0000)

        # OR with +0.0 leaves mask unchanged
        saved, new_exec, scc = oracle_s_or_saveexec_b64(0x1234_5678, pos_zero_bits)
        self.assertEqual(new_exec, 0x1234_5678)

        # OR with -0.0 activates lane 63
        saved, new_exec, scc = oracle_s_or_saveexec_b64(0, neg_zero_bits)
        self.assertEqual(new_exec, 0x8000_0000_0000_0000)
        self.assertEqual(scc, 1)

    def test_ieee754_infinities_and_nans(self) -> None:
        """Infinities and NaNs are treated strictly as bit patterns; no quietening occurs in SALU."""
        pos_inf_bits = f64_to_bits(float("inf"))
        neg_inf_bits = f64_to_bits(float("-inf"))
        qnan_bits = CANONICAL_QNAN64
        snan_bits = 0x7FF0_0000_0000_0001  # SNaN (fraction != 0, quiet bit 51 is 0)

        self.assertEqual(pos_inf_bits, 0x7FF0_0000_0000_0000)
        self.assertEqual(neg_inf_bits, 0xFFF0_0000_0000_0000)

        # Verify SALU preserves SNaN bits without converting to QNaN
        saved, new_exec, scc = oracle_s_or_saveexec_b64(0, snan_bits)
        self.assertEqual(new_exec, snan_bits)
        self.assertEqual(new_exec & 0x0008_0000_0000_0000, 0, "Quiet bit should NOT be spontaneously set by SALU")
        self.assertEqual(scc, 1)

        # Combining +inf with neg_zero sets sign bit and exponent bits
        saved, new_exec, scc = oracle_s_or_saveexec_b64(pos_inf_bits, 0x8000_0000_0000_0000)
        self.assertEqual(new_exec, neg_inf_bits)
        self.assertEqual(scc, 1)

    def test_ieee754_subnormals_bit_patterns(self) -> None:
        """Subnormal bit patterns (exponent=0, fraction!=0) are preserved bit-for-bit."""
        min_subnormal = 0x0000_0000_0000_0001
        max_subnormal = 0x000F_FFFF_FFFF_FFFF

        saved, new_exec, scc = oracle_s_or_saveexec_b64(min_subnormal, max_subnormal)
        self.assertEqual(saved, min_subnormal)
        self.assertEqual(new_exec, max_subnormal)
        self.assertEqual(scc, 1)

    def test_walking_zeros_dense_masks(self) -> None:
        """Test walking zero (single inactive lane) across all 64 lane positions."""
        for lane in range(64):
            dense_mask = u64(~(1 << lane))
            # When combined with all zeros, new_exec is dense_mask
            saved, new_exec, scc = oracle_s_or_saveexec_b64(0, dense_mask)
            self.assertEqual(new_exec, dense_mask)
            self.assertEqual(popcount64(new_exec), 63)
            self.assertEqual(scc, 1)

            # When combined with the missing lane, completes to all lanes active
            saved2, new_exec2, scc2 = oracle_s_or_saveexec_b64(new_exec, 1 << lane)
            self.assertEqual(new_exec2, ALL_LANES_ACTIVE_64)
            self.assertEqual(popcount64(new_exec2), 64)
            self.assertEqual(scc2, 1)

    def test_alternating_and_checkerboard_merges(self) -> None:
        """Alternating lane masks (0x5555... and 0xAAAA...) merge to all ones."""
        saved, new_exec, scc = oracle_s_or_saveexec_b64(EVEN_LANES_64, ODD_LANES_64)
        self.assertEqual(saved, EVEN_LANES_64)
        self.assertEqual(new_exec, ALL_LANES_ACTIVE_64)
        self.assertEqual(scc, 1)
        self.assertEqual(popcount64(new_exec), 64)

    def test_sparse_random_bit_masks(self) -> None:
        """Fuzz with sparse masks having only 1, 2, or 3 random bits set."""
        rng = random.Random(0x7547_7547)
        for _ in range(50):
            lanes = rng.sample(range(64), k=3)
            mask_a = (1 << lanes[0]) | (1 << lanes[1])
            mask_b = 1 << lanes[2]
            saved, new_exec, scc = oracle_s_or_saveexec_b64(mask_a, mask_b)
            self.assertEqual(saved, mask_a)
            self.assertEqual(new_exec, mask_a | mask_b)
            self.assertEqual(popcount64(new_exec), 3)
            self.assertEqual(scc, 1)



# ==============================================================================
# Phase 8: Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

def mutant_and_instead_of_or(exec_in: int, src_val: int) -> Tuple[int, int, int]:
    """Mutant 1: Emits bitwise AND instead of OR."""
    saved_sdst = u64(exec_in)
    new_exec = u64(src_val & exec_in)
    scc = 1 if new_exec != 0 else 0
    return saved_sdst, new_exec, scc


def mutant_xor_instead_of_or(exec_in: int, src_val: int) -> Tuple[int, int, int]:
    """Mutant 2: Emits bitwise XOR instead of OR."""
    saved_sdst = u64(exec_in)
    new_exec = u64(src_val ^ exec_in)
    scc = 1 if new_exec != 0 else 0
    return saved_sdst, new_exec, scc


def mutant_save_new_exec(exec_in: int, src_val: int) -> Tuple[int, int, int]:
    """Mutant 3: Saves updated EXEC into SDEST instead of previous EXEC."""
    new_exec = u64(src_val | exec_in)
    saved_sdst = new_exec
    scc = 1 if new_exec != 0 else 0
    return saved_sdst, new_exec, scc


def mutant_inverted_scc(exec_in: int, src_val: int) -> Tuple[int, int, int]:
    """Mutant 4: Inverts SCC condition flag (sets 1 on zero, 0 on non-zero)."""
    saved_sdst = u64(exec_in)
    new_exec = u64(src_val | exec_in)
    scc = 1 if new_exec == 0 else 0
    return saved_sdst, new_exec, scc


def mutant_scc_from_old_exec(exec_in: int, src_val: int) -> Tuple[int, int, int]:
    """Mutant 5: Computes SCC from old EXEC instead of resulting EXEC."""
    saved_sdst = u64(exec_in)
    new_exec = u64(src_val | exec_in)
    scc = 1 if saved_sdst != 0 else 0
    return saved_sdst, new_exec, scc


def mutant_truncate_32bit(exec_in: int, src_val: int) -> Tuple[int, int, int]:
    """Mutant 6: Incorrectly truncates EXEC mask to 32 bits on Wave64."""
    saved_sdst = u64(exec_in)
    new_exec = u32(src_val | exec_in)
    scc = 1 if new_exec != 0 else 0
    return saved_sdst, new_exec, scc


def mutant_src_negation(exec_in: int, src_val: int) -> Tuple[int, int, int]:
    """Mutant 7: Confuses s_or with s_orn2 (ORs with inverted source)."""
    saved_sdst = u64(exec_in)
    new_exec = u64((~src_val & MASK64) | exec_in)
    scc = 1 if new_exec != 0 else 0
    return saved_sdst, new_exec, scc


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Verifies syntax rejection of invalid assembly and catches subtle behavioral mutations."""

    def test_rejection_of_invalid_mnemonics(self) -> None:
        """Reject invalid mnemonics and opcodes."""
        invalid_mnemonics = [
            "s_or_saveexec_b32 s[0:1], s[2:3]",
            "v_or_saveexec_b64 s[0:1], s[2:3]",
            "s_and_saveexec_b64 s[0:1], s[2:3]",
            "s_or_b64 s[0:1], s[2:3]",
            "s_saveexec_b64 s[0:1], s[2:3]",
        ]
        for line in invalid_mnemonics:
            with self.assertRaises(ValueError):
                SOrSaveExecB64Instruction.parse(line)

    def test_rejection_of_vgpr_and_odd_sgpr_operands(self) -> None:
        """Reject VGPR destinations and unaligned odd SGPR base indices."""
        invalid_registers = [
            "s_or_saveexec_b64 v[0:1], s[2:3]",
            "s_or_saveexec_b64 v0, s[2:3]",
            "s_or_saveexec_b64 s[1:2], s[4:5]",
            "s_or_saveexec_b64 s[3:4], s[4:5]",
            "s_or_saveexec_b64 s[105:106], s[4:5]",
        ]
        for line in invalid_registers:
            with self.assertRaises(ValueError):
                SOrSaveExecB64Instruction.parse(line)

    def test_rejection_of_truncated_and_excess_operands(self) -> None:
        """Reject instruction lines with wrong operand counts or malformed delimiters."""
        malformed = [
            "s_or_saveexec_b64",
            "s_or_saveexec_b64 s[0:1]",
            "s_or_saveexec_b64 s[0:1], s[2:3], s[4:5]",
            "s_or_saveexec_b64 s[0:1] s[2:3]",
            "s_or_saveexec_b64 s[0:1],",
        ]
        for line in malformed:
            with self.assertRaises(ValueError):
                SOrSaveExecB64Instruction.parse(line)

    def test_mutation_detection_and_instead_of_or(self) -> None:
        """Mutant 1 (AND instead of OR) must be detected."""
        exec_in = 0x0000_0000_FFFF_0000
        src_val = 0x0000_0000_0000_FFFF
        ref = oracle_s_or_saveexec_b64(exec_in, src_val)
        mut = mutant_and_instead_of_or(exec_in, src_val)
        self.assertNotEqual(ref, mut)
        self.assertEqual(ref[1], 0x0000_0000_FFFF_FFFF)
        self.assertEqual(mut[1], 0x0)

    def test_mutation_detection_xor_instead_of_or(self) -> None:
        """Mutant 2 (XOR instead of OR) must be detected on overlapping bits."""
        exec_in = 0xFFFF_FFFF_FFFF_FFFF
        src_val = 0xFFFF_FFFF_FFFF_FFFF
        ref = oracle_s_or_saveexec_b64(exec_in, src_val)
        mut = mutant_xor_instead_of_or(exec_in, src_val)
        self.assertNotEqual(ref, mut)
        self.assertEqual(ref[1], ALL_LANES_ACTIVE_64)
        self.assertEqual(mut[1], 0x0)

    def test_mutation_detection_save_new_exec(self) -> None:
        """Mutant 3 (saving new EXEC instead of previous EXEC) must be detected."""
        exec_in = 0x1
        src_val = 0x2
        ref = oracle_s_or_saveexec_b64(exec_in, src_val)
        mut = mutant_save_new_exec(exec_in, src_val)
        self.assertNotEqual(ref, mut)
        self.assertEqual(ref[0], 0x1)  # Saved is old EXEC
        self.assertEqual(mut[0], 0x3)  # Mutant saved new EXEC

    def test_mutation_detection_inverted_scc(self) -> None:
        """Mutant 4 (inverted SCC) must be detected."""
        exec_in = 0x0
        src_val = 0x1
        ref = oracle_s_or_saveexec_b64(exec_in, src_val)
        mut = mutant_inverted_scc(exec_in, src_val)
        self.assertNotEqual(ref, mut)
        self.assertEqual(ref[2], 1)
        self.assertEqual(mut[2], 0)

    def test_mutation_detection_scc_from_old_exec(self) -> None:
        """Mutant 5 (computing SCC before the OR operation) must be detected."""
        exec_in = 0x0  # Old exec is 0
        src_val = 0x100  # New exec will be 0x100 != 0
        ref = oracle_s_or_saveexec_b64(exec_in, src_val)
        mut = mutant_scc_from_old_exec(exec_in, src_val)
        self.assertNotEqual(ref, mut)
        self.assertEqual(ref[2], 1)
        self.assertEqual(mut[2], 0)

    def test_mutation_detection_truncate_32bit(self) -> None:
        """Mutant 6 (truncating mask to 32 bits) must be detected on upper 32 lanes."""
        exec_in = 0x0
        src_val = 0x8000_0000_0000_0000
        ref = oracle_s_or_saveexec_b64(exec_in, src_val)
        mut = mutant_truncate_32bit(exec_in, src_val)
        self.assertNotEqual(ref, mut)
        self.assertEqual(ref[1], 0x8000_0000_0000_0000)
        self.assertEqual(mut[1], 0x0)

    def test_mutation_detection_src_negation(self) -> None:
        """Mutant 7 (confusing with s_orn2) must be detected."""
        exec_in = 0x0
        src_val = 0x0
        ref = oracle_s_or_saveexec_b64(exec_in, src_val)
        mut = mutant_src_negation(exec_in, src_val)
        self.assertNotEqual(ref, mut)
        self.assertEqual(ref[1], 0x0)
        self.assertEqual(mut[1], ALL_LANES_ACTIVE_64)


if __name__ == "__main__":
    unittest.main()







