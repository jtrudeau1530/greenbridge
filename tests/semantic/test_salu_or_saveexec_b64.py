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


if __name__ == "__main__":
    unittest.main()


