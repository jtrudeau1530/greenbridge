"""Semantic interpreter tests and reference mathematical oracle for RDNA2 s_and_saveexec_b64.

Models the AMD RDNA2 / GFX10 / GCN Scalar Arithmetic Logic Unit (SALU) architectural
specification for:
    s_and_saveexec_b64 sdst, ssrc0

Architectural Semantics (SOP1 Encoding):
  1. Latched State Execution:
     sdst[63:0] = EXEC[63:0]
     EXEC[63:0] = ssrc0[63:0] & EXEC[63:0]
     SCC        = (EXEC[63:0] != 0) ? 1 : 0
  2. Operand Rules:
     - sdst: 64-bit destination SGPR pair (or EXEC/VCC special registers).
       Receives the previous (saved) value of the 64-bit EXEC mask.
     - ssrc0: 64-bit source operand (SGPR pair, literal constant, inline constant, or special register).
     - EXEC: 64-bit execution mask governing vector wavefront lane activation in Wave64 mode.
     - SCC: Scalar Condition Code bit in the STATUS register, updated to 1 if any lane in the
       new EXEC mask is non-zero, and 0 if all lanes are inactive.
  3. Register File Constraints:
     - 64-bit SGPR pairs must be 64-bit aligned (even SGPR index: s[0:1], s[2:3], etc.).
     - Destination latches previous EXEC before any potential register overwrite.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import enum
import os
import re
import sys
import unittest
from dataclasses import dataclass, field
from typing import Optional, Union

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT64: int = 0x8000000000000000
ALL_LANES_64: int = 0xFFFFFFFFFFFFFFFF
NO_LANES_64: int = 0x0000000000000000
EVEN_LANES_64: int = 0x5555555555555555
ODD_LANES_64: int = 0xAAAAAAAAAAAAAAAA
LOW32_LANES_64: int = 0x00000000FFFFFFFF
HIGH32_LANES_64: int = 0xFFFFFFFF00000000


class RegKind(enum.Enum):
    """Register kinds in the RDNA2 architectural state."""
    SGPR = "sgpr"
    VGPR = "vgpr"
    EXEC = "exec"
    VCC = "vcc"
    SCC = "scc"
    M0 = "m0"
    LITERAL = "literal"
    INLINE_CONST = "inline_const"


@dataclass(frozen=True)
class Register:
    """Symbolic register representation with architectural width and alignment validation."""
    kind: RegKind
    index: int = 0
    size: int = 32  # 32 or 64 bits

    def validate(self) -> None:
        """Validate architectural constraints for registers."""
        if self.kind == RegKind.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index {self.index} out of architectural range [0..105]")
            if self.size == 64 and (self.index % 2 != 0):
                raise ValueError(f"64-bit SGPR must be even-aligned: s[{self.index}:{self.index + 1}]")
            if self.size == 64 and (self.index > 104):
                raise ValueError(f"64-bit SGPR pair s[{self.index}:{self.index + 1}] exceeds max SGPR 105")
        elif self.kind == RegKind.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index {self.index} out of architectural range [0..255]")
            if self.size == 64 and (self.index > 254):
                raise ValueError(f"64-bit VGPR pair v[{self.index}:{self.index + 1}] exceeds max VGPR 255")

    def __str__(self) -> str:
        if self.kind == RegKind.SGPR:
            return f"s[{self.index}:{self.index + 1}]" if self.size == 64 else f"s{self.index}"
        if self.kind == RegKind.VGPR:
            return f"v[{self.index}:{self.index + 1}]" if self.size == 64 else f"v{self.index}"
        if self.kind == RegKind.EXEC:
            return "exec" if self.size == 64 else "exec_lo"
        if self.kind == RegKind.VCC:
            return "vcc" if self.size == 64 else "vcc_lo"
        if self.kind == RegKind.SCC:
            return "scc"
        if self.kind == RegKind.M0:
            return "m0"
        return f"{self.kind.value}:{self.index}"


@dataclass(frozen=True)
class Operand:
    """Symbolic instruction operand representing either a register or immediate."""
    reg: Optional[Register] = None
    imm: Optional[int] = None
    size: int = 32

    @property
    def is_reg(self) -> bool:
        return self.reg is not None

    @property
    def is_imm(self) -> bool:
        return self.imm is not None

    def __str__(self) -> str:
        if self.reg is not None:
            return str(self.reg)
        if self.imm is not None:
            return f"{self.imm:#x}" if self.imm > 9 or self.imm < -9 else str(self.imm)
        return "<invalid>"


class Opcode(enum.Enum):
    """Opcode enumeration covering SALU and branch instructions."""
    S_AND_SAVEEXEC_B64 = "s_and_saveexec_b64"
    S_OR_SAVEEXEC_B64 = "s_or_saveexec_b64"
    S_ANDN2_SAVEEXEC_B64 = "s_andn2_saveexec_b64"
    S_MOV_B64 = "s_mov_b64"
    S_MOV_B32 = "s_mov_b32"
    S_CBRANCH_EXECZ = "s_cbranch_execz"
    S_CBRANCH_EXECNZ = "s_cbranch_execnz"
    S_CBRANCH_SCC0 = "s_cbranch_scc0"
    S_CBRANCH_SCC1 = "s_cbranch_scc1"
    S_ENDPGM = "s_endpgm"


@dataclass
class Instruction:
    """Symbolic typed instruction expression in the AST."""
    opcode: Opcode
    dst: Optional[Register] = None
    src0: Optional[Operand] = None
    src1: Optional[Operand] = None
    label_target: Optional[str] = None
    comment: str = ""

    def validate(self) -> None:
        """Validate instruction operand typing and sizing."""
        if self.opcode == Opcode.S_AND_SAVEEXEC_B64:
            if self.dst is None or self.src0 is None:
                raise ValueError(f"{self.opcode.value} requires sdst and ssrc0 operands")
            if self.dst.size != 64:
                raise ValueError(f"{self.opcode.value} destination must be 64-bit")
            if self.dst.kind not in (RegKind.SGPR, RegKind.EXEC, RegKind.VCC):
                raise ValueError(f"{self.opcode.value} destination must be SGPR, EXEC, or VCC")
            self.dst.validate()
            if self.src0.is_reg:
                assert self.src0.reg is not None
                if self.src0.reg.size != 64:
                    raise ValueError(f"{self.opcode.value} source register must be 64-bit")
                self.src0.reg.validate()
        elif self.opcode == Opcode.S_MOV_B64:
            if self.dst is None or self.src0 is None:
                raise ValueError(f"{self.opcode.value} requires dst and src0")
            if self.dst.size != 64:
                raise ValueError(f"{self.opcode.value} dst must be 64-bit")
            self.dst.validate()

    def emit_asm(self) -> str:
        """Serialize AST instruction to standard assembly text."""
        parts = [self.opcode.value]
        operands = []
        if self.dst is not None:
            operands.append(str(self.dst))
        if self.src0 is not None:
            operands.append(str(self.src0))
        if self.src1 is not None:
            operands.append(str(self.src1))
        if self.label_target is not None:
            operands.append(self.label_target)
        line = f"{self.opcode.value} {', '.join(operands)}" if operands else self.opcode.value
        if self.comment:
            line = f"{line:<40} ; {self.comment}"
        return line


@dataclass
class Program:
    """Symbolic program representation containing AST instructions and labels."""
    instructions: list[Instruction] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)

    def add_instruction(self, inst: Instruction) -> None:
        inst.validate()
        self.instructions.append(inst)

    def add_label(self, name: str) -> None:
        self.labels[name] = len(self.instructions)

    def emit_asm(self) -> str:
        lines = []
        label_positions = {pos: name for name, pos in self.labels.items()}
        for i, inst in enumerate(self.instructions):
            if i in label_positions:
                lines.append(f"{label_positions[i]}:")
            lines.append(f"    {inst.emit_asm()}")
        if len(self.instructions) in label_positions:
            lines.append(f"{label_positions[len(self.instructions)]}:")
        return "\n".join(lines)


def parse_operand(text: str, default_size: int = 64) -> Operand:
    """Parse string operand into typed Operand AST node."""
    text = text.strip()
    if text == "exec":
        return Operand(reg=Register(RegKind.EXEC, 0, size=64), size=64)
    if text == "exec_lo":
        return Operand(reg=Register(RegKind.EXEC, 0, size=32), size=32)
    if text == "vcc":
        return Operand(reg=Register(RegKind.VCC, 0, size=64), size=64)
    if text == "vcc_lo":
        return Operand(reg=Register(RegKind.VCC, 0, size=32), size=32)
    if text == "scc":
        return Operand(reg=Register(RegKind.SCC, 0, size=32), size=32)
    if text == "m0":
        return Operand(reg=Register(RegKind.M0, 0, size=32), size=32)

    # 64-bit pair s[x:y] or v[x:y]
    m_pair = re.match(r"^([sv])\[(\d+):(\d+)\]$", text)
    if m_pair:
        prefix, start, end = m_pair.group(1), int(m_pair.group(2)), int(m_pair.group(3))
        if end != start + 1:
            raise ValueError(f"Non-contiguous or >2-register pair not supported: {text}")
        kind = RegKind.SGPR if prefix == "s" else RegKind.VGPR
        reg = Register(kind=kind, index=start, size=64)
        reg.validate()
        return Operand(reg=reg, size=64)

    # Single register sX or vX
    m_single = re.match(r"^([sv])(\d+)$", text)
    if m_single:
        prefix, idx = m_single.group(1), int(m_single.group(2))
        kind = RegKind.SGPR if prefix == "s" else RegKind.VGPR
        reg = Register(kind=kind, index=idx, size=default_size)
        reg.validate()
        return Operand(reg=reg, size=default_size)

    # Immediate (decimal or hex)
    try:
        val = int(text, 0)
        return Operand(imm=val, size=default_size)
    except ValueError:
        raise ValueError(f"Unrecognized operand format: {text}")


def parse_instruction(line: str) -> Instruction:
    """Parse a single assembly text line into an Instruction AST node."""
    clean = line.split(";", 1)[0].strip()
    if not clean:
        raise ValueError("Empty instruction line")
    parts = clean.split(None, 1)
    op_str = parts[0].lower()
    try:
        opcode = Opcode(op_str)
    except ValueError:
        raise ValueError(f"Unknown opcode: {op_str}")

    if opcode == Opcode.S_ENDPGM:
        return Instruction(opcode=opcode)

    args_str = parts[1] if len(parts) > 1 else ""
    raw_args = [a.strip() for a in args_str.split(",") if a.strip()]

    if opcode == Opcode.S_AND_SAVEEXEC_B64:
        if len(raw_args) != 2:
            raise ValueError(f"{opcode.value} expects exactly 2 operands, got {len(raw_args)}")
        dst_op = parse_operand(raw_args[0], default_size=64)
        if not dst_op.is_reg:
            raise ValueError(f"{opcode.value} destination must be a register")
        src0_op = parse_operand(raw_args[1], default_size=64)
        inst = Instruction(opcode=opcode, dst=dst_op.reg, src0=src0_op)
        inst.validate()
        return inst

    if opcode in (Opcode.S_CBRANCH_EXECZ, Opcode.S_CBRANCH_EXECNZ, Opcode.S_CBRANCH_SCC0, Opcode.S_CBRANCH_SCC1):
        if len(raw_args) != 1:
            raise ValueError(f"{opcode.value} expects a single label target")
        return Instruction(opcode=opcode, label_target=raw_args[0])

    if opcode == Opcode.S_MOV_B64:
        if len(raw_args) != 2:
            raise ValueError(f"{opcode.value} expects dst, src0")
        dst_op = parse_operand(raw_args[0], default_size=64)
        src0_op = parse_operand(raw_args[1], default_size=64)
        inst = Instruction(opcode=opcode, dst=dst_op.reg, src0=src0_op)
        inst.validate()
        return inst

    raise NotImplementedError(f"Parser not implemented for opcode {opcode.value}")


@dataclass
class ScalarRegisterState:
    """Architectural register state for the RDNA2 scalar execution unit."""
    sgprs: list[int] = field(default_factory=lambda: [0] * 106)
    exec: int = ALL_LANES_64
    vcc: int = 0
    m0: int = 0
    scc: int = 0
    pc: int = 0

    def read_sgpr32(self, idx: int) -> int:
        return self.sgprs[idx] & MASK32

    def write_sgpr32(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32

    def read_sgpr64(self, idx: int) -> int:
        if idx % 2 != 0:
            raise ValueError(f"64-bit SGPR read requires even alignment: s[{idx}:{idx+1}]")
        lo = self.sgprs[idx] & MASK32
        hi = self.sgprs[idx + 1] & MASK32
        return join_dwords(lo, hi)

    def write_sgpr64(self, idx: int, val: int) -> None:
        if idx % 2 != 0:
            raise ValueError(f"64-bit SGPR write requires even alignment: s[{idx}:{idx+1}]")
        lo, hi = split_dwords(val)
        self.sgprs[idx] = lo
        self.sgprs[idx + 1] = hi

    def read_operand(self, op: Operand) -> int:
        if op.imm is not None:
            return op.imm & (MASK64 if op.size == 64 else MASK32)
        assert op.reg is not None
        reg = op.reg
        if reg.kind == RegKind.SGPR:
            return self.read_sgpr64(reg.index) if reg.size == 64 else self.read_sgpr32(reg.index)
        if reg.kind == RegKind.EXEC:
            return self.exec & MASK64 if reg.size == 64 else self.exec & MASK32
        if reg.kind == RegKind.VCC:
            return self.vcc & MASK64 if reg.size == 64 else self.vcc & MASK32
        if reg.kind == RegKind.M0:
            return self.m0 & MASK32
        if reg.kind == RegKind.SCC:
            return self.scc & 1
        raise ValueError(f"Unsupported operand register kind: {reg.kind}")

    def write_reg(self, reg: Register, val: int) -> None:
        if reg.kind == RegKind.SGPR:
            if reg.size == 64:
                self.write_sgpr64(reg.index, val)
            else:
                self.write_sgpr32(reg.index, val)
        elif reg.kind == RegKind.EXEC:
            self.exec = val & (MASK64 if reg.size == 64 else MASK32)
        elif reg.kind == RegKind.VCC:
            self.vcc = val & (MASK64 if reg.size == 64 else MASK32)
        elif reg.kind == RegKind.M0:
            self.m0 = val & MASK32
        elif reg.kind == RegKind.SCC:
            self.scc = val & 1
        else:
            raise ValueError(f"Cannot write to register kind {reg.kind}")


class ScalarSALUInterpreter:
    """Deterministic interpreter simulating the RDNA2 SALU pipeline."""

    def __init__(self, state: Optional[ScalarRegisterState] = None):
        self.state = state if state is not None else ScalarRegisterState()

    def step(self, inst: Instruction, labels: Optional[dict[str, int]] = None) -> bool:
        """Execute a single instruction. Returns False if execution terminated."""
        if inst.opcode == Opcode.S_ENDPGM:
            return False

        if inst.opcode == Opcode.S_AND_SAVEEXEC_B64:
            assert inst.dst is not None and inst.src0 is not None
            # Latched execution: read source value
            src_val = self.state.read_operand(inst.src0)
            old_exec = self.state.exec & MASK64
            new_exec = (old_exec & src_val) & MASK64

            # sdst receives prior EXEC
            self.state.write_reg(inst.dst, old_exec)
            # EXEC receives bitwise AND result
            self.state.exec = new_exec
            # SCC updated to (new_exec != 0)
            self.state.scc = 1 if new_exec != 0 else 0
            self.state.pc += 1
            return True

        if inst.opcode == Opcode.S_MOV_B64:
            assert inst.dst is not None and inst.src0 is not None
            val = self.state.read_operand(inst.src0)
            self.state.write_reg(inst.dst, val)
            self.state.pc += 1
            return True

        if inst.opcode == Opcode.S_MOV_B32:
            assert inst.dst is not None and inst.src0 is not None
            val = self.state.read_operand(inst.src0)
            self.state.write_reg(inst.dst, val)
            self.state.pc += 1
            return True

        if inst.opcode == Opcode.S_CBRANCH_EXECZ:
            assert inst.label_target is not None and labels is not None
            if self.state.exec == 0:
                self.state.pc = labels[inst.label_target]
            else:
                self.state.pc += 1
            return True

        if inst.opcode == Opcode.S_CBRANCH_EXECNZ:
            assert inst.label_target is not None and labels is not None
            if self.state.exec != 0:
                self.state.pc = labels[inst.label_target]
            else:
                self.state.pc += 1
            return True

        if inst.opcode == Opcode.S_CBRANCH_SCC0:
            assert inst.label_target is not None and labels is not None
            if self.state.scc == 0:
                self.state.pc = labels[inst.label_target]
            else:
                self.state.pc += 1
            return True

        if inst.opcode == Opcode.S_CBRANCH_SCC1:
            assert inst.label_target is not None and labels is not None
            if self.state.scc == 1:
                self.state.pc = labels[inst.label_target]
            else:
                self.state.pc += 1
            return True

        raise NotImplementedError(f"Interpreter support for {inst.opcode.value} not implemented")

    def run(self, program: Program, max_steps: int = 1000) -> ScalarRegisterState:
        """Run symbolic program until completion or max_steps exceeded."""
        steps = 0
        self.state.pc = 0
        while self.state.pc < len(program.instructions) and steps < max_steps:
            inst = program.instructions[self.state.pc]
            running = self.step(inst, program.labels)
            steps += 1
            if not running:
                break
        return self.state

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT64: int = 0x8000000000000000
ALL_LANES_64: int = 0xFFFFFFFFFFFFFFFF
NO_LANES_64: int = 0x0000000000000000
EVEN_LANES_64: int = 0x5555555555555555
ODD_LANES_64: int = 0xAAAAAAAAAAAAAAAA
LOW32_LANES_64: int = 0x00000000FFFFFFFF
HIGH32_LANES_64: int = 0xFFFFFFFF00000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def popcount64(val: int) -> int:
    """Return the number of active bits in a 64-bit integer."""
    return bin(val & MASK64).count("1")


def active_lanes_from_exec(exec_val: int) -> list[int]:
    """Extract a list of active lane indices [0..63] from a 64-bit EXEC mask."""
    val = exec_val & MASK64
    return [lane for lane in range(64) if (val & (1 << lane)) != 0]


def exec_from_active_lanes(lanes: list[int]) -> int:
    """Construct a 64-bit EXEC mask from a list of active lane indices [0..63]."""
    mask = 0
    for lane in lanes:
        if 0 <= lane < 64:
            mask |= (1 << lane)
        else:
            raise ValueError(f"Lane index {lane} out of 64-bit bounds [0..63]")
    return mask & MASK64


def ref_s_and_saveexec_b64(exec_mask: int, ssrc0: int) -> tuple[int, int, int]:
    """Pure mathematical reference oracle for RDNA2 s_and_saveexec_b64.

    Args:
        exec_mask: Current 64-bit execution mask.
        ssrc0: 64-bit source operand mask.

    Returns:
        tuple (sdst, new_exec, scc):
          - sdst: The saved prior EXEC mask (64-bit).
          - new_exec: The updated EXEC mask after bitwise AND with ssrc0 (64-bit).
          - scc: 1 if new_exec != 0, else 0.
    """
    exec_clean = exec_mask & MASK64
    src_clean = ssrc0 & MASK64
    sdst = exec_clean
    new_exec = exec_clean & src_clean
    scc = 1 if new_exec != 0 else 0
    return sdst, new_exec, scc


class TestSaluAndSaveexecB64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_oracle_identity_and_zero(self) -> None:
        """Test identity with all ones and clearing with zero mask."""
        # EXEC all lanes active, src all lanes active -> saved all, new all, SCC=1
        sdst, new_exec, scc = ref_s_and_saveexec_b64(ALL_LANES_64, ALL_LANES_64)
        self.assertEqual(sdst, ALL_LANES_64)
        self.assertEqual(new_exec, ALL_LANES_64)
        self.assertEqual(scc, 1)

        # EXEC all lanes, src zero -> saved all, new zero, SCC=0
        sdst, new_exec, scc = ref_s_and_saveexec_b64(ALL_LANES_64, NO_LANES_64)
        self.assertEqual(sdst, ALL_LANES_64)
        self.assertEqual(new_exec, NO_LANES_64)
        self.assertEqual(scc, 0)

        # EXEC zero, src all lanes -> saved zero, new zero, SCC=0
        sdst, new_exec, scc = ref_s_and_saveexec_b64(NO_LANES_64, ALL_LANES_64)
        self.assertEqual(sdst, NO_LANES_64)
        self.assertEqual(new_exec, NO_LANES_64)
        self.assertEqual(scc, 0)

    def test_oracle_scc_flag_transitions(self) -> None:
        """Verify SCC accurately reflects non-zero state of new_exec."""
        # Single bit mask yields SCC=1
        for lane in (0, 1, 31, 32, 62, 63):
            bit = 1 << lane
            sdst, new_exec, scc = ref_s_and_saveexec_b64(ALL_LANES_64, bit)
            self.assertEqual(sdst, ALL_LANES_64)
            self.assertEqual(new_exec, bit)
            self.assertEqual(scc, 1)

        # Disjoint masks yield new_exec=0 and SCC=0
        sdst, new_exec, scc = ref_s_and_saveexec_b64(EVEN_LANES_64, ODD_LANES_64)
        self.assertEqual(sdst, EVEN_LANES_64)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)

    def test_oracle_single_lane_masks(self) -> None:
        """Verify single active lane filtering across all 64 lane positions."""
        for lane in range(64):
            bit = 1 << lane
            sdst, new_exec, scc = ref_s_and_saveexec_b64(bit, bit)
            self.assertEqual(sdst, bit)
            self.assertEqual(new_exec, bit)
            self.assertEqual(scc, 1)

            # Different lane -> zero result
            other_bit = 1 << ((lane + 1) % 64)
            sdst2, new_exec2, scc2 = ref_s_and_saveexec_b64(bit, other_bit)
            self.assertEqual(sdst2, bit)
            self.assertEqual(new_exec2, 0)
            self.assertEqual(scc2, 0)

    def test_oracle_alternating_bitmasks(self) -> None:
        """Test alternating even/odd lane bitmasks."""
        sdst, new_exec, scc = ref_s_and_saveexec_b64(ALL_LANES_64, EVEN_LANES_64)
        self.assertEqual(sdst, ALL_LANES_64)
        self.assertEqual(new_exec, EVEN_LANES_64)
        self.assertEqual(scc, 1)
        self.assertEqual(popcount64(new_exec), 32)

        sdst, new_exec, scc = ref_s_and_saveexec_b64(EVEN_LANES_64, EVEN_LANES_64)
        self.assertEqual(sdst, EVEN_LANES_64)
        self.assertEqual(new_exec, EVEN_LANES_64)
        self.assertEqual(scc, 1)

    def test_oracle_split_join_dwords(self) -> None:
        """Verify 64-bit SGPR pair split/join helpers."""
        lo, hi = split_dwords(0x123456789ABCDEF0)
        self.assertEqual(lo, 0x9ABCDEF0)
        self.assertEqual(hi, 0x12345678)
        self.assertEqual(join_dwords(lo, hi), 0x123456789ABCDEF0)

    def test_oracle_active_lanes_helpers(self) -> None:
        """Test conversion between bitmask and active lane list."""
        lanes = [0, 7, 15, 31, 32, 47, 63]
        mask = exec_from_active_lanes(lanes)
        self.assertEqual(popcount64(mask), len(lanes))
        self.assertEqual(active_lanes_from_exec(mask), lanes)

        with self.assertRaises(ValueError):
            exec_from_active_lanes([64])

    def test_oracle_wave32_upper_lower_masks(self) -> None:
        """Verify behavior on lower 32-bit and upper 32-bit halves."""
        sdst, new_exec, scc = ref_s_and_saveexec_b64(LOW32_LANES_64, HIGH32_LANES_64)
        self.assertEqual(sdst, LOW32_LANES_64)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)

        sdst, new_exec, scc = ref_s_and_saveexec_b64(ALL_LANES_64, LOW32_LANES_64)
        self.assertEqual(sdst, ALL_LANES_64)
        self.assertEqual(new_exec, LOW32_LANES_64)
        self.assertEqual(scc, 1)

    def test_oracle_commutative_and_associative_properties(self) -> None:
        """Verify bitwise boolean algebraic properties."""
        a = 0xF0F0F0F00F0F0F0F
        b = 0xCCCCCCCC33333333
        _, res_ab, _ = ref_s_and_saveexec_b64(a, b)
        _, res_ba, _ = ref_s_and_saveexec_b64(b, a)
        self.assertEqual(res_ab, res_ba)


class TestSaluAndSaveexecB64AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_register_creation_and_bounds(self) -> None:
        """Verify register AST nodes and alignment rules."""
        # Valid 64-bit aligned SGPR pair
        reg = Register(RegKind.SGPR, 4, size=64)
        reg.validate()
        self.assertEqual(str(reg), "s[4:5]")

        # Valid 64-bit special registers
        exec_reg = Register(RegKind.EXEC, 0, size=64)
        exec_reg.validate()
        self.assertEqual(str(exec_reg), "exec")

        vcc_reg = Register(RegKind.VCC, 0, size=64)
        vcc_reg.validate()
        self.assertEqual(str(vcc_reg), "vcc")

        # Invalid unaligned 64-bit SGPR (odd index)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 5, size=64).validate()

        # Out-of-bounds SGPR index (> 104 for 64-bit pair)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 106, size=64).validate()

    def test_ast_instruction_encoding(self) -> None:
        """Verify Instruction AST construction and assembly emission."""
        dst = Register(RegKind.SGPR, 4, size=64)
        src0 = Operand(reg=Register(RegKind.VCC, 0, size=64), size=64)
        inst = Instruction(Opcode.S_AND_SAVEEXEC_B64, dst=dst, src0=src0, comment="save old exec")
        inst.validate()
        asm = inst.emit_asm()
        self.assertTrue(asm.startswith("s_and_saveexec_b64 s[4:5], vcc"))
        self.assertIn("; save old exec", asm)

    def test_ast_program_assembly_emission(self) -> None:
        """Verify Program container emission with labels and instructions."""
        prog = Program()
        prog.add_instruction(Instruction(Opcode.S_MOV_B64, dst=Register(RegKind.SGPR, 0, size=64),
                                         src0=Operand(imm=0xFFFFFFFFFFFFFFFF, size=64)))
        prog.add_instruction(Instruction(Opcode.S_AND_SAVEEXEC_B64, dst=Register(RegKind.SGPR, 4, size=64),
                                         src0=Operand(reg=Register(RegKind.SGPR, 0, size=64), size=64)))
        prog.add_instruction(Instruction(Opcode.S_CBRANCH_EXECZ, label_target="skip_divergent"))
        prog.add_label("skip_divergent")
        prog.add_instruction(Instruction(Opcode.S_ENDPGM))

        asm = prog.emit_asm()
        self.assertIn("s_and_saveexec_b64 s[4:5], s[0:1]", asm)
        self.assertIn("skip_divergent:", asm)
        self.assertIn("s_cbranch_execz skip_divergent", asm)
        self.assertIn("s_endpgm", asm)

    def test_ast_operand_parsing(self) -> None:
        """Verify parsing operands from assembly strings."""
        op_sgpr = parse_operand("s[12:13]")
        self.assertTrue(op_sgpr.is_reg)
        assert op_sgpr.reg is not None
        self.assertEqual(op_sgpr.reg.kind, RegKind.SGPR)
        self.assertEqual(op_sgpr.reg.index, 12)
        self.assertEqual(op_sgpr.size, 64)

        op_exec = parse_operand("exec")
        self.assertTrue(op_exec.is_reg)
        assert op_exec.reg is not None
        self.assertEqual(op_exec.reg.kind, RegKind.EXEC)

        op_imm = parse_operand("0x5555555555555555")
        self.assertTrue(op_imm.is_imm)
        self.assertEqual(op_imm.imm, 0x5555555555555555)

    def test_ast_instruction_line_parsing(self) -> None:
        """Verify parsing instruction lines."""
        inst = parse_instruction("s_and_saveexec_b64 s[8:9], exec ; save exec")
        self.assertEqual(inst.opcode, Opcode.S_AND_SAVEEXEC_B64)
        assert inst.dst is not None
        self.assertEqual(inst.dst.kind, RegKind.SGPR)
        self.assertEqual(inst.dst.index, 8)
        assert inst.src0 is not None and inst.src0.reg is not None
        self.assertEqual(inst.src0.reg.kind, RegKind.EXEC)

    def test_ast_program_semantic_validation(self) -> None:
        """Verify AST semantic validation flags illegal constructs."""
        # Destination not 64-bit
        bad_dst = Register(RegKind.SGPR, 4, size=32)
        bad_src = Operand(reg=Register(RegKind.SGPR, 6, size=64))
        inst = Instruction(Opcode.S_AND_SAVEEXEC_B64, dst=bad_dst, src0=bad_src)
        with self.assertRaises(ValueError):
            inst.validate()


class TestSaluAndSaveexecB64ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def test_interp_basic_execution(self) -> None:
        """Verify s_and_saveexec_b64 saves prior EXEC and updates EXEC with mask."""
        state = ScalarRegisterState(exec=0x123456789ABCDEF0)
        interp = ScalarSALUInterpreter(state)

        # s_and_saveexec_b64 s[4:5], 0x00000000FFFFFFFF
        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 4, size=64),
            src0=Operand(imm=0x00000000FFFFFFFF, size=64),
        )
        interp.step(inst)

        # s[4:5] must hold saved prior EXEC
        self.assertEqual(state.read_sgpr64(4), 0x123456789ABCDEF0)
        # EXEC must be masked
        self.assertEqual(state.exec, 0x000000009ABCDEF0)
        # SCC must be 1 since result is non-zero
        self.assertEqual(state.scc, 1)

    def test_interp_exec_zero_clearing(self) -> None:
        """Verify s_and_saveexec_b64 with zero mask clears EXEC and clears SCC."""
        state = ScalarRegisterState(exec=ALL_LANES_64, scc=1)
        interp = ScalarSALUInterpreter(state)

        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 0, size=64),
            src0=Operand(imm=0, size=64),
        )
        interp.step(inst)

        self.assertEqual(state.read_sgpr64(0), ALL_LANES_64)
        self.assertEqual(state.exec, 0)
        self.assertEqual(state.scc, 0)

    def test_interp_scc_update(self) -> None:
        """Verify SCC tracks (new_exec != 0) deterministically."""
        state = ScalarRegisterState(exec=0x8000000000000000)
        interp = ScalarSALUInterpreter(state)

        # Mask preserving bit 63 -> SCC=1
        inst1 = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 2, size=64),
            src0=Operand(imm=0x8000000000000000, size=64),
        )
        interp.step(inst1)
        self.assertEqual(state.scc, 1)

        # Mask clearing bit 63 -> SCC=0
        inst2 = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 4, size=64),
            src0=Operand(imm=0x7FFFFFFFFFFFFFFF, size=64),
        )
        interp.step(inst2)
        self.assertEqual(state.scc, 0)

    def test_interp_saved_exec_to_sgpr_pair(self) -> None:
        """Verify 64-bit saved EXEC split into s[d] low and s[d+1] high dwords."""
        state = ScalarRegisterState(exec=0xAABBCCDDEEFF0011)
        interp = ScalarSALUInterpreter(state)

        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 10, size=64),
            src0=Operand(imm=ALL_LANES_64, size=64),
        )
        interp.step(inst)

        self.assertEqual(state.read_sgpr32(10), 0xEEFF0011)
        self.assertEqual(state.read_sgpr32(11), 0xAABBCCDD)

    def test_interp_chained_saveexec_flow(self) -> None:
        """Verify nested/chained saveexec save and restore."""
        prog = Program()
        # s_mov_b64 s[0:1], 0x5555555555555555 (even lanes)
        prog.add_instruction(Instruction(Opcode.S_MOV_B64, dst=Register(RegKind.SGPR, 0, size=64),
                                         src0=Operand(imm=EVEN_LANES_64, size=64)))
        # s_and_saveexec_b64 s[4:5], s[0:1] -> saved=ALL, exec=EVEN
        prog.add_instruction(Instruction(Opcode.S_AND_SAVEEXEC_B64, dst=Register(RegKind.SGPR, 4, size=64),
                                         src0=Operand(reg=Register(RegKind.SGPR, 0, size=64), size=64)))
        # s_mov_b64 exec, s[4:5] -> restores ALL
        prog.add_instruction(Instruction(Opcode.S_MOV_B64, dst=Register(RegKind.EXEC, 0, size=64),
                                         src0=Operand(reg=Register(RegKind.SGPR, 4, size=64), size=64)))
        prog.add_instruction(Instruction(Opcode.S_ENDPGM))

        interp = ScalarSALUInterpreter()
        state = interp.run(prog)

        self.assertEqual(state.read_sgpr64(4), ALL_LANES_64)
        self.assertEqual(state.exec, ALL_LANES_64)

    def test_interp_conditional_branching_execz(self) -> None:
        """Verify s_cbranch_execz jumps when exec is zero."""
        prog = Program()
        # s_and_saveexec_b64 s[0:1], 0 -> exec becomes 0
        prog.add_instruction(Instruction(Opcode.S_AND_SAVEEXEC_B64, dst=Register(RegKind.SGPR, 0, size=64),
                                         src0=Operand(imm=0, size=64)))
        prog.add_instruction(Instruction(Opcode.S_CBRANCH_EXECZ, label_target="target"))
        prog.add_instruction(Instruction(Opcode.S_MOV_B32, dst=Register(RegKind.SGPR, 10, size=32),
                                         src0=Operand(imm=999, size=32)))
        prog.add_label("target")
        prog.add_instruction(Instruction(Opcode.S_MOV_B32, dst=Register(RegKind.SGPR, 10, size=32),
                                         src0=Operand(imm=42, size=32)))
        prog.add_instruction(Instruction(Opcode.S_ENDPGM))

        interp = ScalarSALUInterpreter()
        state = interp.run(prog)
        self.assertEqual(state.read_sgpr32(10), 42)


if __name__ == "__main__":
    unittest.main()
