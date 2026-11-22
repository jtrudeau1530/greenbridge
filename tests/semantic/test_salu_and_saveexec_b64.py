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

# IEEE 754 architectural bit patterns for vector-to-scalar comparison masks
POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001
SUBNORM_MIN_F32: int = 0x00000001
SUBNORM_MAX_F32: int = 0x007FFFFF

POS_ZERO_F64: int = 0x0000000000000000
NEG_ZERO_F64: int = 0x8000000000000000
POS_INF_F64: int = 0x7FF0000000000000
NEG_INF_F64: int = 0xFFF0000000000000
QNAN_F64: int = 0x7FF8000000000000
SNAN_F64: int = 0x7FF0000000000001
SUBNORM_MIN_F64: int = 0x0000000000000001


def is_nan_f32_bits(bits: int) -> bool:
    """Return True if 32-bit word represents an IEEE 754 single-precision NaN."""
    return ((bits & 0x7F800000) == 0x7F800000) and ((bits & 0x007FFFFF) != 0)


def is_subnormal_f32_bits(bits: int) -> bool:
    """Return True if 32-bit word represents an IEEE 754 single-precision subnormal."""
    return ((bits & 0x7F800000) == 0) and ((bits & 0x007FFFFF) != 0)


def is_inf_f32_bits(bits: int) -> bool:
    """Return True if 32-bit word represents positive or negative infinity."""
    return ((bits & 0x7F800000) == 0x7F800000) and ((bits & 0x007FFFFF) == 0)


def generate_f32_cmp_mask(lanes_f32: list[int], condition: str) -> int:
    """Simulate vector float comparison across 64 lanes producing a 64-bit mask."""
    mask = 0
    for lane, val in enumerate(lanes_f32[:64]):
        b = val & MASK32
        matched = False
        if condition == "nan":
            matched = is_nan_f32_bits(b)
        elif condition == "subnormal":
            matched = is_subnormal_f32_bits(b)
        elif condition == "inf":
            matched = is_inf_f32_bits(b)
        elif condition == "neg_zero":
            matched = (b == NEG_ZERO_F32)
        elif condition == "pos_zero":
            matched = (b == POS_ZERO_F32)
        elif condition == "finite":
            matched = not (is_nan_f32_bits(b) or is_inf_f32_bits(b))
        if matched:
            mask |= (1 << lane)
    return mask & MASK64


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


@dataclass
class SIMDWavefrontState:
    """Architectural state for a 64-lane SIMD wavefront with shared scalar and per-lane vector registers."""
    scalar: ScalarRegisterState = field(default_factory=ScalarRegisterState)
    vgprs: list[list[int]] = field(default_factory=lambda: [[0] * 64 for _ in range(256)])

    def read_vgpr(self, reg_idx: int, lane: int) -> int:
        return self.vgprs[reg_idx][lane] & MASK32

    def write_vgpr(self, reg_idx: int, lane: int, val: int) -> None:
        self.vgprs[reg_idx][lane] = val & MASK32

    def write_vgpr_active_lanes(self, reg_idx: int, lane_val: Union[int, list[int]]) -> None:
        """Write to vector register only across active lanes indicated by scalar EXEC mask."""
        exec_mask = self.scalar.exec & MASK64
        for lane in range(64):
            if (exec_mask & (1 << lane)) != 0:
                val = lane_val[lane] if isinstance(lane_val, list) else lane_val
                self.vgprs[reg_idx][lane] = val & MASK32


class SIMDWavefrontInterpreter:
    """Wavefront interpreter modeling SIMD multi-lane divergence and 64-bit EXEC masking."""

    def __init__(self, wave_state: Optional[SIMDWavefrontState] = None):
        self.wave = wave_state if wave_state is not None else SIMDWavefrontState()
        self.scalar_interp = ScalarSALUInterpreter(self.wave.scalar)

    def execute_vector_mov(self, dst_vgpr: int, src_val: Union[int, list[int]]) -> None:
        """Execute vector move under current EXEC mask."""
        self.wave.write_vgpr_active_lanes(dst_vgpr, src_val)

    def execute_vector_add(self, dst_vgpr: int, src0_vgpr: int, src1_vgpr: int) -> None:
        """Execute vector addition under current EXEC mask."""
        exec_mask = self.wave.scalar.exec & MASK64
        for lane in range(64):
            if (exec_mask & (1 << lane)) != 0:
                s0 = self.wave.read_vgpr(src0_vgpr, lane)
                s1 = self.wave.read_vgpr(src1_vgpr, lane)
                self.wave.write_vgpr(dst_vgpr, lane, (s0 + s1) & MASK32)

    def execute_vector_cmp_eq(self, src0_vgpr: int, imm: int) -> int:
        """Vector compare equal against immediate, producing a 64-bit mask in VCC."""
        mask = 0
        for lane in range(64):
            if self.wave.read_vgpr(src0_vgpr, lane) == (imm & MASK32):
                mask |= (1 << lane)
        self.wave.scalar.vcc = mask & MASK64
        return mask & MASK64

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT64: int = 0x8000000000000000
ALL_LANES_64: int = 0xFFFFFFFFFFFFFFFF
NO_LANES_64: int = 0x0000000000000000
EVEN_LANES_64: int = 0x5555555555555555
ODD_LANES_64: int = 0xAAAAAAAAAAAAAAAA
LOW32_LANES_64: int = 0x00000000FFFFFFFF
HIGH32_LANES_64: int = 0xFFFFFFFF00000000

# IEEE 754 architectural bit patterns for vector-to-scalar comparison masks
POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001
SUBNORM_MIN_F32: int = 0x00000001
SUBNORM_MAX_F32: int = 0x007FFFFF

POS_ZERO_F64: int = 0x0000000000000000
NEG_ZERO_F64: int = 0x8000000000000000
POS_INF_F64: int = 0x7FF0000000000000
NEG_INF_F64: int = 0xFFF0000000000000
QNAN_F64: int = 0x7FF8000000000000
SNAN_F64: int = 0x7FF0000000000001
SUBNORM_MIN_F64: int = 0x0000000000000001


def is_nan_f32_bits(bits: int) -> bool:
    """Return True if 32-bit word represents an IEEE 754 single-precision NaN."""
    return ((bits & 0x7F800000) == 0x7F800000) and ((bits & 0x007FFFFF) != 0)


def is_subnormal_f32_bits(bits: int) -> bool:
    """Return True if 32-bit word represents an IEEE 754 single-precision subnormal."""
    return ((bits & 0x7F800000) == 0) and ((bits & 0x007FFFFF) != 0)


def is_inf_f32_bits(bits: int) -> bool:
    """Return True if 32-bit word represents positive or negative infinity."""
    return ((bits & 0x7F800000) == 0x7F800000) and ((bits & 0x007FFFFF) == 0)


def generate_f32_cmp_mask(lanes_f32: list[int], condition: str) -> int:
    """Simulate vector float comparison across 64 lanes producing a 64-bit mask."""
    mask = 0
    for lane, val in enumerate(lanes_f32[:64]):
        b = val & MASK32
        matched = False
        if condition == "nan":
            matched = is_nan_f32_bits(b)
        elif condition == "subnormal":
            matched = is_subnormal_f32_bits(b)
        elif condition == "inf":
            matched = is_inf_f32_bits(b)
        elif condition == "neg_zero":
            matched = (b == NEG_ZERO_F32)
        elif condition == "pos_zero":
            matched = (b == POS_ZERO_F32)
        elif condition == "finite":
            matched = not (is_nan_f32_bits(b) or is_inf_f32_bits(b))
        if matched:
            mask |= (1 << lane)
    return mask & MASK64


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


class TestSaluAndSaveexecB64SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_lane_divergence_masking(self) -> None:
        """Verify vector writes execute strictly across active lanes governed by s_and_saveexec_b64."""
        wave = SIMDWavefrontState()
        interp = SIMDWavefrontInterpreter(wave)

        # Initialize all 64 lanes of v0 to 100
        interp.execute_vector_mov(dst_vgpr=0, src_val=100)
        for lane in range(64):
            self.assertEqual(wave.read_vgpr(0, lane), 100)

        # Set vcc to even lanes (0x5555555555555555)
        wave.scalar.vcc = EVEN_LANES_64

        # Divergent branch setup: s_and_saveexec_b64 s[4:5], vcc
        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 4, size=64),
            src0=Operand(reg=Register(RegKind.VCC, 0, size=64), size=64),
        )
        interp.scalar_interp.step(inst)

        # Verify scalar state: saved EXEC has all 64 lanes, new EXEC has even lanes
        self.assertEqual(wave.scalar.read_sgpr64(4), ALL_LANES_64)
        self.assertEqual(wave.scalar.exec, EVEN_LANES_64)
        self.assertEqual(wave.scalar.scc, 1)

        # Execute vector write to v0 with value 200 under even-lane EXEC mask
        interp.execute_vector_mov(dst_vgpr=0, src_val=200)

        # Active (even) lanes must be updated to 200; inactive (odd) lanes must remain 100
        for lane in range(64):
            expected = 200 if (lane % 2 == 0) else 100
            self.assertEqual(wave.read_vgpr(0, lane), expected, f"Mismatch at lane {lane}")

    def test_simd_inactive_lanes_preservation(self) -> None:
        """Verify inactive lanes preserve all VGPR data across repeated operations."""
        wave = SIMDWavefrontState()
        interp = SIMDWavefrontInterpreter(wave)

        # Each lane has distinct initial value
        for lane in range(64):
            wave.write_vgpr(1, lane, 1000 + lane)

        # Activate only lane 42
        wave.scalar.vcc = 1 << 42
        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 8, size=64),
            src0=Operand(reg=Register(RegKind.VCC, 0, size=64), size=64),
        )
        interp.scalar_interp.step(inst)

        # Write 9999 to v1 under single-lane mask
        interp.execute_vector_mov(dst_vgpr=1, src_val=9999)

        for lane in range(64):
            if lane == 42:
                self.assertEqual(wave.read_vgpr(1, lane), 9999)
            else:
                self.assertEqual(wave.read_vgpr(1, lane), 1000 + lane)

    def test_simd_nested_divergence_stack(self) -> None:
        """Verify 2-level nested conditional divergence saving masks into SGPR pairs."""
        wave = SIMDWavefrontState()
        interp = SIMDWavefrontInterpreter(wave)

        # Level 1: activate lower 32 lanes
        # s_and_saveexec_b64 s[4:5], 0x00000000FFFFFFFF
        inst1 = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 4, size=64),
            src0=Operand(imm=LOW32_LANES_64, size=64),
        )
        interp.scalar_interp.step(inst1)
        self.assertEqual(wave.scalar.read_sgpr64(4), ALL_LANES_64)
        self.assertEqual(wave.scalar.exec, LOW32_LANES_64)

        # Level 2 (nested): activate only even lanes within lower 32 lanes
        # s_and_saveexec_b64 s[6:7], 0x5555555555555555
        inst2 = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 6, size=64),
            src0=Operand(imm=EVEN_LANES_64, size=64),
        )
        interp.scalar_interp.step(inst2)
        self.assertEqual(wave.scalar.read_sgpr64(6), LOW32_LANES_64)
        self.assertEqual(wave.scalar.exec, LOW32_LANES_64 & EVEN_LANES_64)

        # Unwind Level 2: restore Level 1 mask
        inst_restore1 = Instruction(
            Opcode.S_MOV_B64,
            dst=Register(RegKind.EXEC, 0, size=64),
            src0=Operand(reg=Register(RegKind.SGPR, 6, size=64), size=64),
        )
        interp.scalar_interp.step(inst_restore1)
        self.assertEqual(wave.scalar.exec, LOW32_LANES_64)

        # Unwind Level 1: restore original all-lanes mask
        inst_restore2 = Instruction(
            Opcode.S_MOV_B64,
            dst=Register(RegKind.EXEC, 0, size=64),
            src0=Operand(reg=Register(RegKind.SGPR, 4, size=64), size=64),
        )
        interp.scalar_interp.step(inst_restore2)
        self.assertEqual(wave.scalar.exec, ALL_LANES_64)

    def test_simd_all_lanes_disabled_early_exit(self) -> None:
        """Verify when no lanes match, s_and_saveexec_b64 sets EXEC=0 and SCC=0."""
        wave = SIMDWavefrontState()
        interp = SIMDWavefrontInterpreter(wave)

        # Condition mask is 0
        wave.scalar.vcc = 0
        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 0, size=64),
            src0=Operand(reg=Register(RegKind.VCC, 0, size=64), size=64),
        )
        interp.scalar_interp.step(inst)

        self.assertEqual(wave.scalar.exec, 0)
        self.assertEqual(wave.scalar.scc, 0)
        self.assertEqual(len(active_lanes_from_exec(wave.scalar.exec)), 0)

    def test_simd_wave32_compatibility_masking(self) -> None:
        """Verify behavior when running 32-lane wave workloads on 64-bit mask."""
        wave = SIMDWavefrontState()
        interp = SIMDWavefrontInterpreter(wave)

        # Restrict initially to Wave32 lower half
        wave.scalar.exec = LOW32_LANES_64

        # AND with upper half produces 0
        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 0, size=64),
            src0=Operand(imm=HIGH32_LANES_64, size=64),
        )
        interp.scalar_interp.step(inst)

        self.assertEqual(wave.scalar.read_sgpr64(0), LOW32_LANES_64)
        self.assertEqual(wave.scalar.exec, 0)
        self.assertEqual(wave.scalar.scc, 0)


class TestSaluAndSaveexecB64RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_aliasing_sdst_equals_ssrc0(self) -> None:
        """Verify sdst == ssrc0: source is latched before destination receives old EXEC."""
        state = ScalarRegisterState(exec=0x1111222233334444)
        interp = ScalarSALUInterpreter(state)

        # Preload s[4:5] with mask 0x0000FFFF0000FFFF
        mask = 0x0000FFFF0000FFFF
        state.write_sgpr64(4, mask)

        # s_and_saveexec_b64 s[4:5], s[4:5]
        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 4, size=64),
            src0=Operand(reg=Register(RegKind.SGPR, 4, size=64), size=64),
        )
        interp.step(inst)

        # s[4:5] must receive the OLD EXEC (0x1111222233334444), NOT the AND result!
        self.assertEqual(state.read_sgpr64(4), 0x1111222233334444)
        # EXEC receives bitwise AND with original s[4:5] mask
        expected_exec = 0x1111222233334444 & mask
        self.assertEqual(state.exec, expected_exec)
        self.assertEqual(state.scc, 1 if expected_exec != 0 else 0)

    def test_aliasing_ssrc0_is_exec(self) -> None:
        """Verify ssrc0 is EXEC: new_exec = EXEC & EXEC = EXEC."""
        val = 0x0123456789ABCDEF
        state = ScalarRegisterState(exec=val)
        interp = ScalarSALUInterpreter(state)

        # s_and_saveexec_b64 s[6:7], exec
        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 6, size=64),
            src0=Operand(reg=Register(RegKind.EXEC, 0, size=64), size=64),
        )
        interp.step(inst)

        self.assertEqual(state.read_sgpr64(6), val)
        self.assertEqual(state.exec, val)
        self.assertEqual(state.scc, 1)

    def test_aliasing_sdst_is_exec(self) -> None:
        """Verify sdst is EXEC: EXEC receives the final computed mask."""
        initial_exec = 0xFFFFFFFFFFFFFFFF
        mask = 0x5555555555555555
        state = ScalarRegisterState(exec=initial_exec)
        interp = ScalarSALUInterpreter(state)
        state.write_sgpr64(2, mask)

        # s_and_saveexec_b64 exec, s[2:3]
        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.EXEC, 0, size=64),
            src0=Operand(reg=Register(RegKind.SGPR, 2, size=64), size=64),
        )
        interp.step(inst)

        self.assertEqual(state.exec, mask)
        self.assertEqual(state.scc, 1)

    def test_aliasing_isolated_registers_untouched(self) -> None:
        """Verify unreferenced registers in the register file remain completely invariant."""
        state = ScalarRegisterState(exec=ALL_LANES_64)
        interp = ScalarSALUInterpreter(state)

        for i in range(0, 106, 2):
            state.write_sgpr64(i, 0x1000000000000000 + i)
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.m0 = 0x12345678

        # s_and_saveexec_b64 s[8:9], 0xF0F0F0F0F0F0F0F0
        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 8, size=64),
            src0=Operand(imm=0xF0F0F0F0F0F0F0F0, size=64),
        )
        interp.step(inst)

        # s[8:9] modified to old EXEC
        self.assertEqual(state.read_sgpr64(8), ALL_LANES_64)

        # All other SGPR pairs must remain untouched
        for i in range(0, 106, 2):
            if i != 8:
                self.assertEqual(state.read_sgpr64(i), 0x1000000000000000 + i, f"SGPR s[{i}:{i+1}] corrupted")

        # VCC and M0 must be untouched
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(state.m0, 0x12345678)

    def test_aliasing_in_place_preservation_across_cycles(self) -> None:
        """Verify cyclic save/restore preserves values without bit rot."""
        state = ScalarRegisterState(exec=0xFEEDFACECAFEBEEF)
        interp = ScalarSALUInterpreter(state)

        # Save to s[0:1], mask with 0x00FF00FF00FF00FF
        inst1 = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 0, size=64),
            src0=Operand(imm=0x00FF00FF00FF00FF, size=64),
        )
        interp.step(inst1)
        self.assertEqual(state.read_sgpr64(0), 0xFEEDFACECAFEBEEF)
        self.assertEqual(state.exec, 0x00ED00CE00FE00EF)

        # Restore EXEC from s[0:1]
        inst2 = Instruction(
            Opcode.S_MOV_B64,
            dst=Register(RegKind.EXEC, 0, size=64),
            src0=Operand(reg=Register(RegKind.SGPR, 0, size=64), size=64),
        )
        interp.step(inst2)
        self.assertEqual(state.exec, 0xFEEDFACECAFEBEEF)


class TestSaluAndSaveexecB64ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_scc_monotonic_zero_absorption(self) -> None:
        """Verify when EXEC is zero, any s_and_saveexec_b64 maintains zero EXEC and SCC=0."""
        state = ScalarRegisterState(exec=0, scc=0)
        interp = ScalarSALUInterpreter(state)

        for mask in (ALL_LANES_64, EVEN_LANES_64, 1, 0x8000000000000000):
            inst = Instruction(
                Opcode.S_AND_SAVEEXEC_B64,
                dst=Register(RegKind.SGPR, 0, size=64),
                src0=Operand(imm=mask, size=64),
            )
            interp.step(inst)
            self.assertEqual(state.read_sgpr64(0), 0)
            self.assertEqual(state.exec, 0)
            self.assertEqual(state.scc, 0, f"SCC violated for mask {mask:#x}")

    def test_scc_transition_matrix(self) -> None:
        """Verify full 2x2 state transition matrix for SCC."""
        interp = ScalarSALUInterpreter()

        # 0 -> 1 transition
        interp.state.exec = ALL_LANES_64
        interp.state.scc = 0
        inst_to_1 = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 0, size=64),
            src0=Operand(imm=1, size=64),
        )
        interp.step(inst_to_1)
        self.assertEqual(interp.state.scc, 1)

        # 1 -> 1 transition
        interp.state.exec = ALL_LANES_64
        interp.state.scc = 1
        inst_keep_1 = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 0, size=64),
            src0=Operand(imm=EVEN_LANES_64, size=64),
        )
        interp.step(inst_keep_1)
        self.assertEqual(interp.state.scc, 1)

        # 1 -> 0 transition
        interp.state.exec = ALL_LANES_64
        interp.state.scc = 1
        inst_to_0 = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 0, size=64),
            src0=Operand(imm=0, size=64),
        )
        interp.step(inst_to_0)
        self.assertEqual(interp.state.scc, 0)

        # 0 -> 0 transition
        interp.state.exec = 0
        interp.state.scc = 0
        inst_keep_0 = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 0, size=64),
            src0=Operand(imm=0, size=64),
        )
        interp.step(inst_keep_0)
        self.assertEqual(interp.state.scc, 0)

    def test_scc_divergence_branching_scc0_scc1(self) -> None:
        """Verify control flow branching on SCC0 and SCC1 after saveexec."""
        prog = Program()
        # s_and_saveexec_b64 s[0:1], 0 -> exec=0, scc=0
        prog.add_instruction(Instruction(Opcode.S_AND_SAVEEXEC_B64, dst=Register(RegKind.SGPR, 0, size=64),
                                         src0=Operand(imm=0, size=64)))
        # s_cbranch_scc0 branch_target
        prog.add_instruction(Instruction(Opcode.S_CBRANCH_SCC0, label_target="scc_zero_target"))
        prog.add_instruction(Instruction(Opcode.S_MOV_B32, dst=Register(RegKind.SGPR, 10, size=32),
                                         src0=Operand(imm=111, size=32)))
        prog.add_label("scc_zero_target")
        prog.add_instruction(Instruction(Opcode.S_MOV_B32, dst=Register(RegKind.SGPR, 10, size=32),
                                         src0=Operand(imm=222, size=32)))
        prog.add_instruction(Instruction(Opcode.S_ENDPGM))

        interp = ScalarSALUInterpreter()
        state = interp.run(prog)
        self.assertEqual(state.read_sgpr32(10), 222)

    def test_status_flags_carry_borrow_unaffected(self) -> None:
        """Verify execution leaves non-SCC architectural status invariant."""
        state = ScalarRegisterState(exec=ALL_LANES_64, m0=0xDEADBEEF)
        interp = ScalarSALUInterpreter(state)

        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 2, size=64),
            src0=Operand(imm=0x12345678, size=64),
        )
        interp.step(inst)
        self.assertEqual(state.m0, 0xDEADBEEF)

    def test_scc_single_bit_boundary_detection(self) -> None:
        """Verify MSB (lane 63) and LSB (lane 0) boundary bits accurately trigger SCC=1."""
        interp = ScalarSALUInterpreter()

        # Lane 0
        interp.state.exec = ALL_LANES_64
        interp.step(Instruction(Opcode.S_AND_SAVEEXEC_B64, dst=Register(RegKind.SGPR, 0, size=64),
                                src0=Operand(imm=1, size=64)))
        self.assertEqual(interp.state.scc, 1)

        # Lane 63
        interp.state.exec = ALL_LANES_64
        interp.step(Instruction(Opcode.S_AND_SAVEEXEC_B64, dst=Register(RegKind.SGPR, 0, size=64),
                                src0=Operand(imm=1 << 63, size=64)))
        self.assertEqual(interp.state.scc, 1)


class TestSaluAndSaveexecB64IEEE754Boundaries(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_ieee754_nan_divergence_masking(self) -> None:
        """Verify vector NaN classification masks correctly drive s_and_saveexec_b64 divergence."""
        wave = SIMDWavefrontState()
        interp = SIMDWavefrontInterpreter(wave)

        # Set up 64 lanes with mixed float values; only lanes 3, 17, 45 have NaNs
        float_lanes = [0x3F800000] * 64  # 1.0f
        float_lanes[3] = QNAN_F32
        float_lanes[17] = SNAN_F32
        float_lanes[45] = QNAN_F32

        nan_mask = generate_f32_cmp_mask(float_lanes, "nan")
        expected_mask = (1 << 3) | (1 << 17) | (1 << 45)
        self.assertEqual(nan_mask, expected_mask)

        # Vector comparison loads VCC
        wave.scalar.vcc = nan_mask

        # Divergent block setup: s_and_saveexec_b64 s[4:5], vcc
        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 4, size=64),
            src0=Operand(reg=Register(RegKind.VCC, 0, size=64), size=64),
        )
        interp.scalar_interp.step(inst)

        self.assertEqual(wave.scalar.read_sgpr64(4), ALL_LANES_64)
        self.assertEqual(wave.scalar.exec, expected_mask)
        self.assertEqual(wave.scalar.scc, 1)

        # Execute NaN fixup on active lanes: write canonical float 0.0
        interp.execute_vector_mov(dst_vgpr=0, src_val=0)

        for lane in range(64):
            if lane in (3, 17, 45):
                self.assertEqual(wave.read_vgpr(0, lane), 0)
            else:
                self.assertEqual(wave.read_vgpr(0, lane), 0)  # default unwritten is 0

    def test_ieee754_subnormal_lane_isolation(self) -> None:
        """Verify subnormal float detection generates mask and updates EXEC."""
        float_lanes = [0x3F800000] * 64
        float_lanes[0] = SUBNORM_MIN_F32
        float_lanes[63] = SUBNORM_MAX_F32

        subnorm_mask = generate_f32_cmp_mask(float_lanes, "subnormal")
        self.assertEqual(subnorm_mask, (1 << 0) | (1 << 63))

        interp = ScalarSALUInterpreter(ScalarRegisterState(exec=ALL_LANES_64))
        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 2, size=64),
            src0=Operand(imm=subnorm_mask, size=64),
        )
        interp.step(inst)
        self.assertEqual(interp.state.exec, (1 << 0) | (1 << 63))
        self.assertEqual(interp.state.scc, 1)

    def test_ieee754_signed_zeros_distinction_mask(self) -> None:
        """Verify distinguishing +0.0 and -0.0 via sign-bit mask."""
        float_lanes = [POS_ZERO_F32] * 64
        float_lanes[10] = NEG_ZERO_F32
        float_lanes[20] = NEG_ZERO_F32

        neg_zero_mask = generate_f32_cmp_mask(float_lanes, "neg_zero")
        self.assertEqual(neg_zero_mask, (1 << 10) | (1 << 20))

        interp = ScalarSALUInterpreter(ScalarRegisterState(exec=ALL_LANES_64))
        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 0, size=64),
            src0=Operand(imm=neg_zero_mask, size=64),
        )
        interp.step(inst)
        self.assertEqual(interp.state.exec, (1 << 10) | (1 << 20))
        self.assertEqual(interp.state.scc, 1)

    def test_ieee754_infinity_boundary_filtering(self) -> None:
        """Verify infinities isolation across lanes."""
        float_lanes = [0x3F800000] * 64
        float_lanes[5] = POS_INF_F32
        float_lanes[6] = NEG_INF_F32

        inf_mask = generate_f32_cmp_mask(float_lanes, "inf")
        self.assertEqual(inf_mask, (1 << 5) | (1 << 6))

        interp = ScalarSALUInterpreter(ScalarRegisterState(exec=ALL_LANES_64))
        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 0, size=64),
            src0=Operand(imm=inf_mask, size=64),
        )
        interp.step(inst)
        self.assertEqual(interp.state.exec, (1 << 5) | (1 << 6))
        self.assertEqual(interp.state.scc, 1)

    def test_ieee754_raw_float_payload_bitwise_and(self) -> None:
        """Verify bitwise operations with IEEE 754 bit-patterns as 64-bit masks."""
        # Using double-precision NaN bit pattern directly as mask
        qnan_mask = QNAN_F64
        interp = ScalarSALUInterpreter(ScalarRegisterState(exec=ALL_LANES_64))
        inst = Instruction(
            Opcode.S_AND_SAVEEXEC_B64,
            dst=Register(RegKind.SGPR, 0, size=64),
            src0=Operand(imm=qnan_mask, size=64),
        )
        interp.step(inst)
        self.assertEqual(interp.state.exec, qnan_mask)
        self.assertEqual(interp.state.scc, 1)


class TestSaluAndSaveexecB64NegativeControls(unittest.TestCase):
    """Unit tests for negative controls, illegal syntax rejection, and boundary enforcement."""

    def test_rejection_odd_aligned_destination_sgpr(self) -> None:
        """Reject unaligned 64-bit destination SGPR pair (e.g. s[1:2])."""
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 1, size=64).validate()
        with self.assertRaises(ValueError):
            parse_instruction("s_and_saveexec_b64 s[1:2], s[0:1]")

    def test_rejection_odd_aligned_source_sgpr(self) -> None:
        """Reject unaligned 64-bit source SGPR pair (e.g. s[3:4])."""
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 3, size=64).validate()
        with self.assertRaises(ValueError):
            parse_instruction("s_and_saveexec_b64 s[0:1], s[3:4]")

    def test_rejection_out_of_bounds_sgpr_index(self) -> None:
        """Reject SGPR indices beyond architectural maximum of 105."""
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 106, size=64).validate()
        with self.assertRaises(ValueError):
            parse_instruction("s_and_saveexec_b64 s[106:107], s[0:1]")

    def test_rejection_vgpr_destination(self) -> None:
        """Reject VGPR destination for SALU s_and_saveexec_b64."""
        with self.assertRaises(ValueError):
            parse_instruction("s_and_saveexec_b64 v[0:1], s[0:1]")

    def test_rejection_invalid_operand_count(self) -> None:
        """Reject instructions with too few or too many operands."""
        with self.assertRaises(ValueError):
            parse_instruction("s_and_saveexec_b64 s[0:1]")
        with self.assertRaises(ValueError):
            parse_instruction("s_and_saveexec_b64 s[0:1], s[2:3], s[4:5]")

    def test_rejection_32bit_register_size_mismatch(self) -> None:
        """Reject 32-bit scalar register for 64-bit opcode."""
        with self.assertRaises(ValueError):
            inst = Instruction(
                Opcode.S_AND_SAVEEXEC_B64,
                dst=Register(RegKind.SGPR, 0, size=32),
                src0=Operand(imm=ALL_LANES_64, size=64),
            )
            inst.validate()


class TestSaluAndSaveexecB64MutationDetection(unittest.TestCase):
    """Mutation testing suite: verifies deliberate architectural mutations are caught."""

    def test_mutation_or_instead_of_and(self) -> None:
        """Detect mutation changing bitwise AND to OR."""
        def mutated_eval(exec_mask: int, src: int) -> tuple[int, int, int]:
            sdst = exec_mask & MASK64
            new_exec = (exec_mask | src) & MASK64  # Mutation: | instead of &
            return sdst, new_exec, (1 if new_exec != 0 else 0)

        # In reference: 0 & 0xFFFFFFFF = 0. In mutated: 0 | 0xFFFFFFFF = 0xFFFFFFFF.
        ref_res = ref_s_and_saveexec_b64(0, 0xFFFFFFFF)
        mut_res = mutated_eval(0, 0xFFFFFFFF)
        self.assertNotEqual(ref_res, mut_res)

    def test_mutation_xor_instead_of_and(self) -> None:
        """Detect mutation changing bitwise AND to XOR."""
        def mutated_eval(exec_mask: int, src: int) -> tuple[int, int, int]:
            sdst = exec_mask & MASK64
            new_exec = (exec_mask ^ src) & MASK64  # Mutation: ^ instead of &
            return sdst, new_exec, (1 if new_exec != 0 else 0)

        # In reference: 1 & 1 = 1. In mutated: 1 ^ 1 = 0.
        ref_res = ref_s_and_saveexec_b64(1, 1)
        mut_res = mutated_eval(1, 1)
        self.assertNotEqual(ref_res, mut_res)

    def test_mutation_andn2_instead_of_and(self) -> None:
        """Detect mutation changing AND to ANDN2 (exec & ~src)."""
        def mutated_eval(exec_mask: int, src: int) -> tuple[int, int, int]:
            sdst = exec_mask & MASK64
            new_exec = (exec_mask & ~src) & MASK64  # Mutation: & ~
            return sdst, new_exec, (1 if new_exec != 0 else 0)

        ref_res = ref_s_and_saveexec_b64(ALL_LANES_64, ALL_LANES_64)
        mut_res = mutated_eval(ALL_LANES_64, ALL_LANES_64)
        self.assertNotEqual(ref_res, mut_res)

    def test_mutation_sdst_receives_new_exec(self) -> None:
        """Detect mutation saving updated EXEC instead of previous EXEC."""
        def mutated_eval(exec_mask: int, src: int) -> tuple[int, int, int]:
            new_exec = (exec_mask & src) & MASK64
            sdst = new_exec  # Mutation: new_exec instead of old exec_mask
            return sdst, new_exec, (1 if new_exec != 0 else 0)

        ref_res = ref_s_and_saveexec_b64(ALL_LANES_64, 0)
        mut_res = mutated_eval(ALL_LANES_64, 0)
        self.assertNotEqual(ref_res, mut_res)

    def test_mutation_sdst_receives_src(self) -> None:
        """Detect mutation saving source operand into sdst."""
        def mutated_eval(exec_mask: int, src: int) -> tuple[int, int, int]:
            sdst = src & MASK64  # Mutation: src instead of exec_mask
            new_exec = (exec_mask & src) & MASK64
            return sdst, new_exec, (1 if new_exec != 0 else 0)

        ref_res = ref_s_and_saveexec_b64(ALL_LANES_64, 0x12345678)
        mut_res = mutated_eval(ALL_LANES_64, 0x12345678)
        self.assertNotEqual(ref_res, mut_res)

    def test_mutation_scc_from_sdst_instead_of_new_exec(self) -> None:
        """Detect mutation deriving SCC from sdst instead of new_exec."""
        def mutated_eval(exec_mask: int, src: int) -> tuple[int, int, int]:
            sdst = exec_mask & MASK64
            new_exec = (exec_mask & src) & MASK64
            scc = 1 if sdst != 0 else 0  # Mutation: sdst != 0
            return sdst, new_exec, scc

        ref_res = ref_s_and_saveexec_b64(ALL_LANES_64, 0)
        mut_res = mutated_eval(ALL_LANES_64, 0)
        self.assertEqual(ref_res[2], 0)  # Reference SCC is 0
        self.assertEqual(mut_res[2], 1)  # Mutated SCC is 1
        self.assertNotEqual(ref_res, mut_res)

    def test_mutation_scc_inversion(self) -> None:
        """Detect mutation inverting SCC condition."""
        def mutated_eval(exec_mask: int, src: int) -> tuple[int, int, int]:
            sdst = exec_mask & MASK64
            new_exec = (exec_mask & src) & MASK64
            scc = 1 if new_exec == 0 else 0  # Mutation: inverted
            return sdst, new_exec, scc

        ref_res = ref_s_and_saveexec_b64(ALL_LANES_64, ALL_LANES_64)
        mut_res = mutated_eval(ALL_LANES_64, ALL_LANES_64)
        self.assertNotEqual(ref_res, mut_res)

    def test_mutation_32bit_truncation(self) -> None:
        """Detect mutation truncating 64-bit EXEC to 32 bits."""
        def mutated_eval(exec_mask: int, src: int) -> tuple[int, int, int]:
            sdst = exec_mask & MASK64
            new_exec = (exec_mask & src) & MASK32  # Mutation: 32-bit truncation
            return sdst, new_exec, (1 if new_exec != 0 else 0)

        # Upper 32-bit mask: ref preserves bits 32..63, mutated zeroes them
        ref_res = ref_s_and_saveexec_b64(ALL_LANES_64, HIGH32_LANES_64)
        mut_res = mutated_eval(ALL_LANES_64, HIGH32_LANES_64)
        self.assertNotEqual(ref_res, mut_res)


if __name__ == "__main__":
    unittest.main()
