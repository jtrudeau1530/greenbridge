from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_sad_u32,
v_sad_hi_u32, and v_mqsad_u32.

Architecture & Instruction Specifications (AMD RDNA2 / GFX10 Vector ALU):
- v_sad_u32 (VOP3 3-operand vector ALU instruction):
    Calculates the 32-bit unsigned sum of absolute differences with accumulation:
        diff = |src0.u32 - src1.u32|
        dst.u32 = (diff + src2.u32) mod 2^32
    Unsigned Semantics:
        The absolute difference is strictly unsigned:
        diff = (src0 >= src1) ? (src0 - src1) : (src1 - src0)
        Operands operate on unsigned 32-bit integers in [0, 2^32 - 1].
    Clamp modifier:
        When clamp is enabled on v_sad_u32, the result saturates to 0xFFFFFFFF on overflow:
        dst.u32 = min(0xFFFFFFFF, diff + src2.u32)

- v_sad_hi_u32:
    Computes the high 32-bit carry / overflow word of the full 64-bit SAD:
        diff = |src0.u32 - src1.u32|
        full_sum = diff + src2.u32
        dst.u32 = (full_sum >> 32) & 0xFFFFFFFF
    Since diff <= 2^32 - 1 and src2 <= 2^32 - 1, full_sum <= 2^33 - 2, so the high
    carry word is strictly in {0, 1}.
    Pairing (v_sad_u32, v_sad_hi_u32) synthesizes a full 64-bit precision sum of
    absolute differences with accumulation.

- v_mqsad_u32 / v_mqsad_u32_u8:
    Quad-byte masked sum of absolute differences with four full 32-bit accumulators:
    S0 is 64 bits (2 VGPRs), S1 is 32 bits (1 VGPR), S2 is 128 bits (4 VGPRs), D is 128 bits (4 VGPRs).
    For each dword k in 0..3:
        W_k = S0[8*k + 31 : 8*k] (32-bit window sliding across S0 byte-by-byte)
        D[k] = (S2[k] + sum_{i=0..3} |W_k.byte[i] - S1.byte[i]| (skipping if S1.byte[i] == 0 when masked)) mod 2^32.

Standard library Python only: unittest, dataclasses, enum, math, struct.
Single module scope, zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import unittest
from typing import List, Sequence, Tuple, Union

# Architectural 32-bit and 64-bit masks
MASK8 = 0xFF
MASK16 = 0xFFFF
MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF
MASK128 = (1 << 128) - 1

CLAMP_MAX_U32 = MASK32


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x8000_0000) else val


def extract_byte(word32: int, byte_idx: int) -> int:
    """Extract byte at byte_idx (0..3) from a 32-bit word."""
    return (word32 >> (8 * (byte_idx & 3))) & MASK8


def byte_window_u32(val64: int, byte_offset: int) -> int:
    """Extract 32-bit window W_k at byte offset k from 64-bit val: bits [8*k+31 : 8*k]."""
    return (val64 >> (8 * byte_offset)) & MASK32


def ref_abs_diff_u32(s0: int, s1: int) -> int:
    """Calculate the strictly unsigned 32-bit absolute difference |s0 - s1|."""
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def ref_sad_u32(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    """Bit-accurate reference mathematical oracle for v_sad_u32.

    diff = |s0.u32 - s1.u32|
    result = (diff + s2.u32) mod 2^32 (or clamped to 0xFFFFFFFF on overflow).
    """
    diff = ref_abs_diff_u32(s0, s1)
    acc = s2 & MASK32
    full_sum = diff + acc
    if clamp:
        return min(CLAMP_MAX_U32, full_sum)
    return full_sum & MASK32


def ref_sad_hi_u32(s0: int, s1: int, s2: int) -> int:
    """Bit-accurate reference mathematical oracle for v_sad_hi_u32 (high 32-bit carry word).

    diff = |s0.u32 - s1.u32|
    full_sum = diff + s2.u32
    carry = (full_sum >> 32) & 0xFFFFFFFF
    """
    diff = ref_abs_diff_u32(s0, s1)
    acc = s2 & MASK32
    full_sum = diff + acc
    return (full_sum >> 32) & MASK32


def ref_quad_byte_sad(win: int, s1: int, *, masked: bool = True) -> int:
    """Compute sum of absolute differences across 4 bytes between win and s1.

    When masked is True, bytes where s1.byte[i] == 0 are skipped.
    """
    total = 0
    for i in range(4):
        b_win = extract_byte(win, i)
        b_s1 = extract_byte(s1, i)
        if masked and b_s1 == 0:
            continue
        total += abs(b_win - b_s1)
    return total


def ref_mqsad_u32(
    s0_64: int,
    s1_32: int,
    s2_128: Tuple[int, int, int, int],
    *,
    masked: bool = True,
) -> Tuple[int, int, int, int]:
    """Bit-accurate reference mathematical oracle for v_mqsad_u32.

    s0_64: 64-bit input source across two VGPRs
    s1_32: 32-bit template operand in one VGPR
    s2_128: 4x 32-bit accumulator dwords across four VGPRs
    Returns 4x 32-bit dwords destination tuple.
    """
    s0 = s0_64 & MASK64
    s1 = s1_32 & MASK32
    out = []
    for k in range(4):
        win = byte_window_u32(s0, k)
        sad = ref_quad_byte_sad(win, s1, masked=masked)
        acc = s2_128[k] & MASK32
        out.append((acc + sad) & MASK32)
    return (out[0], out[1], out[2], out[3])


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class Operand:
    """Represents a register operand or inline literal constant."""
    reg_type: RegType
    index: int = 0
    literal_value: int | None = None
    count: int = 1

    def __post_init__(self) -> None:
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index {self.index} out of range [0, 255]")
            if self.index + self.count > 256:
                raise ValueError(f"VGPR range [{self.index}:{self.index + self.count - 1}] exceeds 255")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index {self.index} out of range [0, 105]")
            if self.index + self.count > 106:
                raise ValueError(f"SGPR range [{self.index}:{self.index + self.count - 1}] exceeds 105")
        elif self.reg_type == RegType.LITERAL:
            if self.literal_value is None:
                raise ValueError("Literal operand requires literal_value")

    @classmethod
    def vgpr(cls, index: int, count: int = 1) -> Operand:
        return cls(RegType.VGPR, index=index, count=count)

    @classmethod
    def sgpr(cls, index: int, count: int = 1) -> Operand:
        return cls(RegType.SGPR, index=index, count=count)

    @classmethod
    def literal(cls, value: int) -> Operand:
        return cls(RegType.LITERAL, literal_value=u32(value), count=1)

    def disassemble(self) -> str:
        if self.reg_type == RegType.VGPR:
            if self.count == 1:
                return f"v{self.index}"
            return f"v[{self.index}:{self.index + self.count - 1}]"
        if self.reg_type == RegType.SGPR:
            if self.count == 1:
                return f"s{self.index}"
            return f"s[{self.index}:{self.index + self.count - 1}]"
        return f"0x{self.literal_value:08X}"


@dataclass(frozen=True)
class VSadU32Instruction:
    """Symbolic AST node for v_sad_u32."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")
        if self.dst.count != 1:
            raise ValueError(f"Destination must be single dword (count=1), got {self.dst.count}")
        for idx, src in enumerate([self.src0, self.src1, self.src2]):
            if src.count != 1:
                raise ValueError(f"src{idx} must be single dword (count=1), got {src.count}")

    def disassemble(self) -> str:
        parts = [
            "v_sad_u32",
            self.dst.disassemble() + ",",
            self.src0.disassemble() + ",",
            self.src1.disassemble() + ",",
            self.src2.disassemble(),
        ]
        if self.clamp:
            parts.append("clamp")
        return " ".join(parts)


@dataclass(frozen=True)
class VSadHiU32Instruction:
    """Symbolic AST node for v_sad_hi_u32."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")
        if self.dst.count != 1:
            raise ValueError(f"Destination must be single dword (count=1), got {self.dst.count}")
        for idx, src in enumerate([self.src0, self.src1, self.src2]):
            if src.count != 1:
                raise ValueError(f"src{idx} must be single dword (count=1), got {src.count}")

    def disassemble(self) -> str:
        return f"v_sad_hi_u32 {self.dst.disassemble()}, {self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"


@dataclass(frozen=True)
class VMqsadU32Instruction:
    """Symbolic AST node for v_mqsad_u32."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    masked: bool = True

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")
        if self.dst.count != 4:
            raise ValueError(f"Destination must be 4 dwords (count=4), got {self.dst.count}")
        if self.src0.count != 2:
            raise ValueError(f"src0 must be 2 dwords (count=2), got {self.src0.count}")
        if self.src1.count != 1:
            raise ValueError(f"src1 must be 1 dword (count=1), got {self.src1.count}")
        if self.src2.count != 4:
            raise ValueError(f"src2 must be 4 dwords (count=4), got {self.src2.count}")

    def disassemble(self) -> str:
        suffix = "" if self.masked else " unmasked"
        return f"v_mqsad_u32 {self.dst.disassemble()}, {self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}{suffix}"


SadInstruction = Union[VSadU32Instruction, VSadHiU32Instruction, VMqsadU32Instruction]


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarRegisterFile:
    """Architectural register file for a single scalar/lane execution context."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read(self, op: Operand) -> int | Tuple[int, ...]:
        """Read operand value as unsigned 32-bit dword or tuple of dwords."""
        if op.reg_type == RegType.LITERAL:
            assert op.literal_value is not None
            return op.literal_value
        if op.reg_type == RegType.VGPR:
            if op.count == 1:
                return self.vgprs[op.index] & MASK32
            return tuple(self.vgprs[op.index + i] & MASK32 for i in range(op.count))
        if op.reg_type == RegType.SGPR:
            if op.count == 1:
                return self.sgprs[op.index] & MASK32
            return tuple(self.sgprs[op.index + i] & MASK32 for i in range(op.count))
        raise ValueError(f"Unknown operand type: {op.reg_type}")

    def read_u64(self, op: Operand) -> int:
        """Read 64-bit qword from 2-register operand."""
        if op.count != 2:
            raise ValueError(f"Operand count must be 2 for 64-bit read, got {op.count}")
        words = self.read(op)
        assert isinstance(words, tuple) and len(words) == 2
        return (words[1] << 32) | words[0]

    def write(self, op: Operand, value: int | Sequence[int]) -> None:
        """Write value to register operand (VGPR or SGPR)."""
        if op.reg_type == RegType.VGPR:
            if op.count == 1:
                assert isinstance(value, int)
                self.vgprs[op.index] = u32(value)
            else:
                assert isinstance(value, (list, tuple)) and len(value) == op.count
                for i in range(op.count):
                    self.vgprs[op.index + i] = u32(value[i])
        elif op.reg_type == RegType.SGPR:
            if op.count == 1:
                assert isinstance(value, int)
                self.sgprs[op.index] = u32(value)
            else:
                assert isinstance(value, (list, tuple)) and len(value) == op.count
                for i in range(op.count):
                    self.sgprs[op.index + i] = u32(value[i])
        else:
            raise TypeError("Cannot write to literal operand")


class ScalarVALUInterpreter:
    """Deterministic instruction interpreter for VALU SAD instructions on a scalar register file."""

    def __init__(self, regs: ScalarRegisterFile | None = None) -> None:
        self.regs: ScalarRegisterFile = regs if regs is not None else ScalarRegisterFile()

    def execute_sad_u32(self, instr: VSadU32Instruction) -> int:
        s0 = self.regs.read(instr.src0)
        s1 = self.regs.read(instr.src1)
        s2 = self.regs.read(instr.src2)
        assert isinstance(s0, int) and isinstance(s1, int) and isinstance(s2, int)
        result = ref_sad_u32(s0, s1, s2, clamp=instr.clamp)
        self.regs.write(instr.dst, result)
        return result

    def execute_sad_hi_u32(self, instr: VSadHiU32Instruction) -> int:
        s0 = self.regs.read(instr.src0)
        s1 = self.regs.read(instr.src1)
        s2 = self.regs.read(instr.src2)
        assert isinstance(s0, int) and isinstance(s1, int) and isinstance(s2, int)
        result = ref_sad_hi_u32(s0, s1, s2)
        self.regs.write(instr.dst, result)
        return result

    def execute_mqsad_u32(self, instr: VMqsadU32Instruction) -> Tuple[int, int, int, int]:
        s0_64 = self.regs.read_u64(instr.src0)
        s1_32 = self.regs.read(instr.src1)
        s2_128 = self.regs.read(instr.src2)
        assert isinstance(s1_32, int)
        assert isinstance(s2_128, tuple) and len(s2_128) == 4
        result = ref_mqsad_u32(s0_64, s1_32, s2_128, masked=instr.masked)
        self.regs.write(instr.dst, result)
        return result

    def step(self, instr: SadInstruction) -> int | Tuple[int, int, int, int]:
        if isinstance(instr, VSadU32Instruction):
            return self.execute_sad_u32(instr)
        if isinstance(instr, VSadHiU32Instruction):
            return self.execute_sad_hi_u32(instr)
        if isinstance(instr, VMqsadU32Instruction):
            return self.execute_mqsad_u32(instr)
        raise TypeError(f"Unsupported instruction AST: {type(instr)}")

    def run_program(self, instructions: Sequence[SadInstruction]) -> None:
        for instr in instructions:
            self.step(instr)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

class SIMDWaveState:
    """Represents the execution state of an AMD RDNA2 wavefront (Wave32 or Wave64)."""

    def __init__(self, wave_size: int = 32, exec_mask: int | None = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"wave_size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.exec_mask_mask: int = MASK32 if wave_size == 32 else MASK64
        self.exec_mask: int = self.exec_mask_mask if exec_mask is None else (exec_mask & self.exec_mask_mask)
        # 256 VGPRs per lane
        self.vgprs: List[List[int]] = [[0] * 256 for _ in range(wave_size)]
        # 106 shared SGPRs
        self.sgprs: List[int] = [0] * 106

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane {lane} out of range [0, {self.wave_size - 1}]")
        return bool((self.exec_mask >> lane) & 1)

    def read_lane(self, lane: int, op: Operand) -> int | Tuple[int, ...]:
        if op.reg_type == RegType.LITERAL:
            assert op.literal_value is not None
            return op.literal_value
        if op.reg_type == RegType.VGPR:
            if op.count == 1:
                return self.vgprs[lane][op.index] & MASK32
            return tuple(self.vgprs[lane][op.index + i] & MASK32 for i in range(op.count))
        if op.reg_type == RegType.SGPR:
            if op.count == 1:
                return self.sgprs[op.index] & MASK32
            return tuple(self.sgprs[op.index + i] & MASK32 for i in range(op.count))
        raise ValueError(f"Unknown operand type: {op.reg_type}")

    def read_lane_u64(self, lane: int, op: Operand) -> int:
        words = self.read_lane(lane, op)
        assert isinstance(words, tuple) and len(words) == 2
        return (words[1] << 32) | words[0]

    def write_lane(self, lane: int, op: Operand, value: int | Sequence[int]) -> None:
        if op.reg_type != RegType.VGPR:
            raise TypeError("VALU SIMD write must target VGPRs")
        if op.count == 1:
            assert isinstance(value, int)
            self.vgprs[lane][op.index] = u32(value)
        else:
            assert isinstance(value, (list, tuple)) and len(value) == op.count
            for i in range(op.count):
                self.vgprs[lane][op.index + i] = u32(value[i])


class SIMDVALUInterpreter:
    """SIMD wavefront instruction interpreter honoring 32/64-bit EXEC masking."""

    def __init__(self, wave: SIMDWaveState) -> None:
        self.wave: SIMDWaveState = wave

    def step(self, instr: SadInstruction) -> List[int | Tuple[int, ...] | None]:
        results: List[int | Tuple[int, ...] | None] = []
        for lane in range(self.wave.wave_size):
            if not self.wave.is_lane_active(lane):
                results.append(None)
                continue

            if isinstance(instr, VSadU32Instruction):
                s0 = self.wave.read_lane(lane, instr.src0)
                s1 = self.wave.read_lane(lane, instr.src1)
                s2 = self.wave.read_lane(lane, instr.src2)
                assert isinstance(s0, int) and isinstance(s1, int) and isinstance(s2, int)
                res = ref_sad_u32(s0, s1, s2, clamp=instr.clamp)
                self.wave.write_lane(lane, instr.dst, res)
                results.append(res)
            elif isinstance(instr, VSadHiU32Instruction):
                s0 = self.wave.read_lane(lane, instr.src0)
                s1 = self.wave.read_lane(lane, instr.src1)
                s2 = self.wave.read_lane(lane, instr.src2)
                assert isinstance(s0, int) and isinstance(s1, int) and isinstance(s2, int)
                res = ref_sad_hi_u32(s0, s1, s2)
                self.wave.write_lane(lane, instr.dst, res)
                results.append(res)
            elif isinstance(instr, VMqsadU32Instruction):
                s0_64 = self.wave.read_lane_u64(lane, instr.src0)
                s1_32 = self.wave.read_lane(lane, instr.src1)
                s2_128 = self.wave.read_lane(lane, instr.src2)
                assert isinstance(s1_32, int)
                assert isinstance(s2_128, tuple) and len(s2_128) == 4
                res_quad = ref_mqsad_u32(s0_64, s1_32, s2_128, masked=instr.masked)
                self.wave.write_lane(lane, instr.dst, res_quad)
                results.append(res_quad)
            else:
                raise TypeError(f"Unsupported instruction AST: {type(instr)}")
        return results

    def run_program(self, instructions: Sequence[SadInstruction]) -> None:
        for instr in instructions:
            self.step(instr)


# ==============================================================================
# Condition Codes, Status Flags, and Carry Generation
# ==============================================================================

@dataclass(frozen=True)
class VALUStatusFlags:
    """Architectural condition codes and status flags for 32-bit ALU operations."""
    zero: bool
    sign: bool
    carry: bool
    overflow: bool

    @classmethod
    def from_sad_u32(cls, s0: int, s1: int, s2: int) -> VALUStatusFlags:
        diff = ref_abs_diff_u32(s0, s1)
        acc = s2 & MASK32
        full_sum = diff + acc
        result = full_sum & MASK32

        zero = (result == 0)
        sign = bool(result & 0x8000_0000)
        carry = (full_sum >= 0x1_0000_0000)
        # Signed overflow between unsigned diff and unsigned acc
        s_diff = s32(diff)
        s_acc = s32(acc)
        s_res = s32(result)
        overflow = ((s_diff > 0 and s_acc > 0 and s_res < 0) or
                    (s_diff < 0 and s_acc < 0 and s_res >= 0))

        return cls(zero=zero, sign=sign, carry=carry, overflow=overflow)


# ==============================================================================
# Unit Tests
# ==============================================================================

class TestValuSadU32Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracles of v_sad_u32, v_sad_hi_u32, and v_mqsad_u32."""

    def test_abs_diff_u32_basic(self) -> None:
        self.assertEqual(ref_abs_diff_u32(10, 4), 6)
        self.assertEqual(ref_abs_diff_u32(4, 10), 6)
        self.assertEqual(ref_abs_diff_u32(0, 0), 0)
        self.assertEqual(ref_abs_diff_u32(MASK32, MASK32), 0)

    def test_abs_diff_u32_commutativity(self) -> None:
        pairs = [
            (0, MASK32),
            (1, 0x8000_0000),
            (0x1234_5678, 0x8765_4321),
            (0x7FFF_FFFF, 0x8000_0000),
            (0xDEAD_BEEF, 0xCAFE_BABE),
        ]
        for a, b in pairs:
            self.assertEqual(
                ref_abs_diff_u32(a, b),
                ref_abs_diff_u32(b, a),
                f"Commutativity failed for ({hex(a)}, {hex(b)})",
            )

    def test_abs_diff_u32_unsigned_vs_signed_distinction(self) -> None:
        self.assertEqual(ref_abs_diff_u32(0, MASK32), MASK32)
        self.assertEqual(ref_abs_diff_u32(MASK32, 0), MASK32)

    def test_triangle_inequality(self) -> None:
        triplets = [
            (100, 250, 400),
            (0, 0x5000_0000, 0xA000_0000),
            (0x1000_0000, 0x3000_0000, 0x7000_0000),
        ]
        for a, b, c in triplets:
            d_ac = ref_abs_diff_u32(a, c)
            d_ab = ref_abs_diff_u32(a, b)
            d_bc = ref_abs_diff_u32(b, c)
            self.assertLessEqual(d_ac, d_ab + d_bc)

    def test_sad_u32_basic_and_identity(self) -> None:
        for val in [0, 1, 0x1234_5678, MASK32]:
            self.assertEqual(ref_sad_u32(val, val, 0x9999), 0x9999)
            self.assertEqual(ref_sad_u32(val, val, 0), 0)

    def test_sad_u32_wraparound_and_clamp(self) -> None:
        self.assertEqual(ref_sad_u32(0, MASK32, 1, clamp=False), 0)
        self.assertEqual(ref_sad_u32(0, MASK32, 1, clamp=True), MASK32)
        self.assertEqual(ref_sad_u32(0, MASK32, MASK32, clamp=False), 0xFFFF_FFFE)
        self.assertEqual(ref_sad_u32(0, MASK32, MASK32, clamp=True), MASK32)
        self.assertEqual(ref_sad_u32(10, 5, 20, clamp=True), 25)

    def test_sad_hi_u32_carry_bounds(self) -> None:
        self.assertEqual(ref_sad_hi_u32(10, 5, 20), 0)
        self.assertEqual(ref_sad_hi_u32(0, MASK32, 0), 0)
        self.assertEqual(ref_sad_hi_u32(0, MASK32, 1), 1)
        self.assertEqual(ref_sad_hi_u32(0, MASK32, MASK32), 1)

    def test_multi_precision_64bit_synthesis(self) -> None:
        test_cases = [
            (0, MASK32, 1, (1 << 32)),
            (0, MASK32, MASK32, (2 * MASK32)),
            (0x8000_0000, 0, 0x8000_0001, (1 << 32) + 1),
            (100, 50, 25, 75),
        ]
        for s0, s1, s2, expected_64 in test_cases:
            lo = ref_sad_u32(s0, s1, s2, clamp=False)
            hi = ref_sad_hi_u32(s0, s1, s2)
            combined_64 = (hi << 32) | lo
            self.assertEqual(combined_64, expected_64)

    def test_quad_byte_sad_masked_vs_unmasked(self) -> None:
        win = 0x0403_0201
        s1 = 0x0400_0200
        self.assertEqual(ref_quad_byte_sad(win, s1, masked=True), 0)
        self.assertEqual(ref_quad_byte_sad(win, s1, masked=False), 4)

    def test_mqsad_u32_sliding_window(self) -> None:
        s0_64 = 0x0807_0605_0403_0201
        self.assertEqual(byte_window_u32(s0_64, 0), 0x0403_0201)
        self.assertEqual(byte_window_u32(s0_64, 1), 0x0504_0302)
        self.assertEqual(byte_window_u32(s0_64, 2), 0x0605_0403)
        self.assertEqual(byte_window_u32(s0_64, 3), 0x0706_0504)

        s1_32 = 0x0403_0201
        s2_128 = (10, 20, 30, 40)
        res = ref_mqsad_u32(s0_64, s1_32, s2_128, masked=False)
        self.assertEqual(res[0], 10)
        self.assertEqual(res[1], 24)
        self.assertEqual(res[2], 38)
        self.assertEqual(res[3], 52)


class TestValuSadU32SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_operand_creation_and_disassembly(self) -> None:
        v0 = Operand.vgpr(0)
        self.assertEqual(v0.disassemble(), "v0")
        v_pair = Operand.vgpr(4, count=2)
        self.assertEqual(v_pair.disassemble(), "v[4:5]")
        v_quad = Operand.vgpr(8, count=4)
        self.assertEqual(v_quad.disassemble(), "v[8:11]")

        s2 = Operand.sgpr(2)
        self.assertEqual(s2.disassemble(), "s2")
        s_pair = Operand.sgpr(10, count=2)
        self.assertEqual(s_pair.disassemble(), "s[10:11]")

        lit = Operand.literal(0x1234_5678)
        self.assertEqual(lit.disassemble(), "0x12345678")

    def test_operand_validation_boundaries(self) -> None:
        with self.assertRaises(ValueError):
            Operand.vgpr(256)
        with self.assertRaises(ValueError):
            Operand.vgpr(-1)
        with self.assertRaises(ValueError):
            Operand.vgpr(255, count=2)
        with self.assertRaises(ValueError):
            Operand.sgpr(106)
        with self.assertRaises(ValueError):
            Operand.sgpr(105, count=2)

    def test_v_sad_u32_ast_construction_and_disasm(self) -> None:
        instr = VSadU32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(2),
            src2=Operand.literal(100),
            clamp=False,
        )
        self.assertEqual(instr.disassemble(), "v_sad_u32 v0, v1, s2, 0x00000064")

        instr_clamped = VSadU32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
            clamp=True,
        )
        self.assertEqual(instr_clamped.disassemble(), "v_sad_u32 v0, v1, v2, v3 clamp")

    def test_v_sad_hi_u32_ast_construction_and_disasm(self) -> None:
        instr = VSadHiU32Instruction(
            dst=Operand.vgpr(5),
            src0=Operand.vgpr(6),
            src1=Operand.vgpr(7),
            src2=Operand.vgpr(8),
        )
        self.assertEqual(instr.disassemble(), "v_sad_hi_u32 v5, v6, v7, v8")

    def test_v_mqsad_u32_ast_construction_and_disasm(self) -> None:
        instr = VMqsadU32Instruction(
            dst=Operand.vgpr(0, count=4),
            src0=Operand.vgpr(4, count=2),
            src1=Operand.vgpr(6, count=1),
            src2=Operand.vgpr(8, count=4),
            masked=True,
        )
        self.assertEqual(instr.disassemble(), "v_mqsad_u32 v[0:3], v[4:5], v6, v[8:11]")

    def test_ast_type_and_count_validation(self) -> None:
        with self.assertRaises(TypeError):
            VSadU32Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
        with self.assertRaises(ValueError):
            VSadU32Instruction(
                dst=Operand.vgpr(0, count=2),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
        with self.assertRaises(ValueError):
            VMqsadU32Instruction(
                dst=Operand.vgpr(0, count=1),
                src0=Operand.vgpr(4, count=2),
                src1=Operand.vgpr(6, count=1),
                src2=Operand.vgpr(8, count=4),
            )


class TestValuSadU32ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = ScalarVALUInterpreter(self.rf)

    def test_scalar_sad_u32_basic(self) -> None:
        self.rf.vgprs[1] = 50
        self.rf.vgprs[2] = 20
        self.rf.vgprs[3] = 100
        instr = VSadU32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        res = self.interp.step(instr)
        self.assertEqual(res, 130)
        self.assertEqual(self.rf.vgprs[0], 130)

    def test_scalar_sad_u32_with_sgpr_and_literal(self) -> None:
        self.rf.vgprs[1] = 100
        self.rf.sgprs[5] = 40
        instr = VSadU32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(5),
            src2=Operand.literal(20),
        )
        res = self.interp.step(instr)
        self.assertEqual(res, 80)
        self.assertEqual(self.rf.vgprs[0], 80)

    def test_scalar_sad_hi_u32_overflow_carry(self) -> None:
        self.rf.vgprs[1] = 0
        self.rf.vgprs[2] = MASK32
        self.rf.vgprs[3] = 10
        instr = VSadHiU32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        res = self.interp.step(instr)
        self.assertEqual(res, 1)
        self.assertEqual(self.rf.vgprs[0], 1)

    def test_64bit_sad_synthesis_program(self) -> None:
        self.rf.vgprs[2] = 0x8000_0000
        self.rf.vgprs[3] = 0
        self.rf.vgprs[4] = 0x8000_0005
        prog = [
            VSadU32Instruction(Operand.vgpr(0), Operand.vgpr(2), Operand.vgpr(3), Operand.vgpr(4)),
            VSadHiU32Instruction(Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3), Operand.vgpr(4)),
        ]
        self.interp.run_program(prog)
        self.assertEqual(self.rf.vgprs[0], 5)
        self.assertEqual(self.rf.vgprs[1], 1)
        res_64 = (self.rf.vgprs[1] << 32) | self.rf.vgprs[0]
        self.assertEqual(res_64, 0x1_0000_0005)

    def test_scalar_mqsad_u32_execution(self) -> None:
        self.rf.vgprs[4] = 0x0403_0201
        self.rf.vgprs[5] = 0x0807_0605
        self.rf.vgprs[6] = 0x0403_0201
        self.rf.vgprs[8] = 100
        self.rf.vgprs[9] = 200
        self.rf.vgprs[10] = 300
        self.rf.vgprs[11] = 400

        instr = VMqsadU32Instruction(
            dst=Operand.vgpr(0, count=4),
            src0=Operand.vgpr(4, count=2),
            src1=Operand.vgpr(6, count=1),
            src2=Operand.vgpr(8, count=4),
            masked=False,
        )
        res = self.interp.step(instr)
        self.assertEqual(res, (100, 204, 308, 412))
        self.assertEqual(self.rf.vgprs[0], 100)
        self.assertEqual(self.rf.vgprs[1], 204)
        self.assertEqual(self.rf.vgprs[2], 308)
        self.assertEqual(self.rf.vgprs[3], 412)


class TestValuSadU32SIMDMultiLane(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFF_FFFF)
        interp = SIMDVALUInterpreter(wave)
        for lane in range(32):
            wave.vgprs[lane][1] = 10 * (lane + 1)
            wave.vgprs[lane][2] = 5 * (lane + 1)
            wave.vgprs[lane][3] = 100

        instr = VSadU32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        results = interp.step(instr)
        for lane in range(32):
            expected = abs(10 * (lane + 1) - 5 * (lane + 1)) + 100
            self.assertEqual(results[lane], expected)
            self.assertEqual(wave.vgprs[lane][0], expected)

    def test_wave32_divergent_mask_preserves_inactive_lanes(self) -> None:
        # Alternating mask: lanes 0, 2, 4... active (0x55555555)
        mask = 0x5555_5555
        wave = SIMDWaveState(wave_size=32, exec_mask=mask)
        interp = SIMDVALUInterpreter(wave)

        for lane in range(32):
            wave.vgprs[lane][0] = 0xCAFE_0000 + lane  # initial dst value
            wave.vgprs[lane][1] = 20
            wave.vgprs[lane][2] = 10
            wave.vgprs[lane][3] = 50

        instr = VSadU32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        results = interp.step(instr)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 60)
                self.assertEqual(wave.vgprs[lane][0], 60)
            else:
                self.assertIsNone(results[lane])
                # Inactive lane must retain original value!
                self.assertEqual(wave.vgprs[lane][0], 0xCAFE_0000 + lane)

    def test_wave32_all_inactive_lanes(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0)
        interp = SIMDVALUInterpreter(wave)
        for lane in range(32):
            wave.vgprs[lane][0] = 0xDEAD_BEEF
        instr = VSadU32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.literal(1),
            src1=Operand.literal(2),
            src2=Operand.literal(3),
        )
        results = interp.step(instr)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][0], 0xDEAD_BEEF)

    def test_wave64_multi_lane_divergence(self) -> None:
        mask64 = 0x0123_4567_89AB_CDEF
        wave = SIMDWaveState(wave_size=64, exec_mask=mask64)
        interp = SIMDVALUInterpreter(wave)

        for lane in range(64):
            wave.vgprs[lane][0] = 0x1111 * (lane + 1)
            wave.vgprs[lane][1] = lane * 100
            wave.vgprs[lane][2] = 500
            wave.vgprs[lane][3] = 1000

        instr = VSadU32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        results = interp.step(instr)
        for lane in range(64):
            is_active = bool((mask64 >> lane) & 1)
            if is_active:
                expected = ref_sad_u32(lane * 100, 500, 1000)
                self.assertEqual(results[lane], expected)
                self.assertEqual(wave.vgprs[lane][0], expected)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(wave.vgprs[lane][0], 0x1111 * (lane + 1))


class TestValuSadU32RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing, in-place operand preservation, and multi-dword hazards."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = ScalarVALUInterpreter(self.rf)

    def test_alias_dst_src0(self) -> None:
        self.rf.vgprs[0] = 50
        self.rf.vgprs[1] = 20
        self.rf.vgprs[2] = 10
        instr = VSadU32Instruction(Operand.vgpr(0), Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2))
        res = self.interp.step(instr)
        self.assertEqual(res, 40)
        self.assertEqual(self.rf.vgprs[0], 40)
        self.assertEqual(self.rf.vgprs[1], 20)
        self.assertEqual(self.rf.vgprs[2], 10)

    def test_alias_dst_src1(self) -> None:
        self.rf.vgprs[0] = 30
        self.rf.vgprs[1] = 80
        self.rf.vgprs[2] = 5
        instr = VSadU32Instruction(Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(0), Operand.vgpr(2))
        res = self.interp.step(instr)
        self.assertEqual(res, 55)
        self.assertEqual(self.rf.vgprs[0], 55)

    def test_alias_dst_src2(self) -> None:
        self.rf.vgprs[0] = 100
        self.rf.vgprs[1] = 40
        self.rf.vgprs[2] = 90
        instr = VSadU32Instruction(Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(0))
        res = self.interp.step(instr)
        self.assertEqual(res, 150)
        self.assertEqual(self.rf.vgprs[0], 150)

    def test_alias_all_same_register(self) -> None:
        self.rf.vgprs[0] = 0x1234_5678
        instr = VSadU32Instruction(Operand.vgpr(0), Operand.vgpr(0), Operand.vgpr(0), Operand.vgpr(0))
        res = self.interp.step(instr)
        self.assertEqual(res, 0x1234_5678)
        self.assertEqual(self.rf.vgprs[0], 0x1234_5678)

    def test_alias_sad_hi_dst_src2(self) -> None:
        self.rf.vgprs[0] = MASK32
        self.rf.vgprs[1] = 0
        self.rf.vgprs[2] = 2
        instr = VSadHiU32Instruction(Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(0))
        res = self.interp.step(instr)
        self.assertEqual(res, 1)
        self.assertEqual(self.rf.vgprs[0], 1)

    def test_mqsad_in_place_accumulator_aliasing(self) -> None:
        self.rf.vgprs[0] = 10
        self.rf.vgprs[1] = 20
        self.rf.vgprs[2] = 30
        self.rf.vgprs[3] = 40

        self.rf.vgprs[4] = 0x0403_0201
        self.rf.vgprs[5] = 0x0807_0605
        self.rf.vgprs[6] = 0x0403_0201

        instr = VMqsadU32Instruction(
            dst=Operand.vgpr(0, count=4),
            src0=Operand.vgpr(4, count=2),
            src1=Operand.vgpr(6, count=1),
            src2=Operand.vgpr(0, count=4),
            masked=False,
        )
        res = self.interp.step(instr)
        self.assertEqual(res, (10, 24, 38, 52))
        self.assertEqual(self.rf.vgprs[0], 10)
        self.assertEqual(self.rf.vgprs[1], 24)
        self.assertEqual(self.rf.vgprs[2], 38)
        self.assertEqual(self.rf.vgprs[3], 52)

    def test_mqsad_overlapping_source_and_destination(self) -> None:
        self.rf.vgprs[1] = 0x0403_0201
        self.rf.vgprs[2] = 0x0807_0605
        self.rf.vgprs[3] = 0x1111_1111
        self.rf.vgprs[4] = 0x2222_2222
        self.rf.vgprs[5] = 0x3333_3333

        self.rf.vgprs[6] = 0x0403_0201

        self.rf.vgprs[8] = 100
        self.rf.vgprs[9] = 200
        self.rf.vgprs[10] = 300
        self.rf.vgprs[11] = 400

        instr = VMqsadU32Instruction(
            dst=Operand.vgpr(2, count=4),
            src0=Operand.vgpr(1, count=2),
            src1=Operand.vgpr(6, count=1),
            src2=Operand.vgpr(8, count=4),
            masked=False,
        )
        res = self.interp.step(instr)
        self.assertEqual(res, (100, 204, 308, 412))
        self.assertEqual(self.rf.vgprs[2], 100)
        self.assertEqual(self.rf.vgprs[3], 204)
        self.assertEqual(self.rf.vgprs[4], 308)
        self.assertEqual(self.rf.vgprs[5], 412)
        self.assertEqual(self.rf.vgprs[1], 0x0403_0201)


class TestValuSadU32StatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry generation."""

    def test_carry_flag_matches_v_sad_hi_u32(self) -> None:
        test_inputs = [
            (0, 0, 0),
            (10, 5, 20),
            (0, MASK32, 0),
            (0, MASK32, 1),
            (0, MASK32, MASK32),
            (0x8000_0000, 0, 0x8000_0000),
            (0x7FFF_FFFF, 0, 0x8000_0000),
            (0x7FFF_FFFF, 0, 0x8000_0001),
            (0xFFFF_0000, 0, 0x0001_0000),
        ]
        for s0, s1, s2 in test_inputs:
            flags = VALUStatusFlags.from_sad_u32(s0, s1, s2)
            sad_hi = ref_sad_hi_u32(s0, s1, s2)
            self.assertEqual(
                1 if flags.carry else 0,
                sad_hi,
                f"Carry flag mismatch for ({hex(s0)}, {hex(s1)}, {hex(s2)})",
            )

    def test_carry_generation_exact_boundary(self) -> None:
        # Sum = 0xFFFFFFFF -> no carry
        flags_no_carry = VALUStatusFlags.from_sad_u32(0, 0xFFFF_FFFE, 1)
        self.assertFalse(flags_no_carry.carry)
        self.assertFalse(flags_no_carry.zero)
        self.assertTrue(flags_no_carry.sign)  # bit 31 is set

        # Sum = 0x100000000 -> carry = True, zero = True, sign = False
        flags_carry = VALUStatusFlags.from_sad_u32(0, MASK32, 1)
        self.assertTrue(flags_carry.carry)
        self.assertTrue(flags_carry.zero)
        self.assertFalse(flags_carry.sign)

    def test_zero_flag_behaviors(self) -> None:
        # Diff = 0, acc = 0 -> zero = True, carry = False, sign = False
        flags_zero = VALUStatusFlags.from_sad_u32(100, 100, 0)
        self.assertTrue(flags_zero.zero)
        self.assertFalse(flags_zero.carry)
        self.assertFalse(flags_zero.sign)

        # Diff = 5, acc = 0 -> zero = False
        flags_nonzero = VALUStatusFlags.from_sad_u32(10, 5, 0)
        self.assertFalse(flags_nonzero.zero)

    def test_sign_flag_behaviors(self) -> None:
        # Result < 0x80000000 -> sign = False
        flags_pos = VALUStatusFlags.from_sad_u32(0, 0x7FFF_FFFF, 0)
        self.assertFalse(flags_pos.sign)

        # Result == 0x80000000 -> sign = True
        flags_neg = VALUStatusFlags.from_sad_u32(0, 0x8000_0000, 0)
        self.assertTrue(flags_neg.sign)


class TestValuSadU32IEEE754Boundaries(unittest.TestCase):
    """Unit tests for IEEE 754 bit-pattern extremes, subnormals, and NaN payload distances."""

    POS_ZERO_F32: int = 0x0000_0000
    NEG_ZERO_F32: int = 0x8000_0000
    POS_INF_F32: int = 0x7F80_0000
    NEG_INF_F32: int = 0xFF80_0000
    QNAN_F32: int = 0x7FC0_0000
    SNAN_F32: int = 0x7F80_0001
    MIN_NORMAL_F32: int = 0x0080_0000
    MAX_SUBNORMAL_F32: int = 0x007F_FFFF
    MIN_SUBNORMAL_F32: int = 0x0000_0001

    @staticmethod
    def f32_to_bits(val: float) -> int:
        return struct.unpack("<I", struct.pack("<f", val))[0]

    def test_ieee754_ulp_distance(self) -> None:
        # 1.0f in binary32 is 0x3F800000
        # The next adjacent float is 1.0000001192092896f (0x3F800001)
        b_one = self.f32_to_bits(1.0)
        b_next = b_one + 1
        dist = ref_sad_u32(b_one, b_next, 0)
        self.assertEqual(dist, 1)

    def test_signed_zero_bit_distance(self) -> None:
        # Distance between +0.0 (0x00000000) and -0.0 (0x80000000)
        dist = ref_sad_u32(self.POS_ZERO_F32, self.NEG_ZERO_F32, 0)
        self.assertEqual(dist, 0x8000_0000)

    def test_subnormal_to_normal_boundary_step(self) -> None:
        # Between max subnormal (0x007FFFFF) and min normal (0x00800000)
        step = ref_sad_u32(self.MIN_NORMAL_F32, self.MAX_SUBNORMAL_F32, 0)
        self.assertEqual(step, 1)

    def test_nan_payload_distance(self) -> None:
        # Distance between infinity and canonical quiet NaN is exactly the quiet bit (0x00400000)
        dist = ref_sad_u32(self.POS_INF_F32, self.QNAN_F32, 0)
        self.assertEqual(dist, 0x0040_0000)

    def test_signaling_nan_payload(self) -> None:
        # Distance between +Inf (0x7F800000) and signaling NaN with payload 1 (0x7F800001)
        dist = ref_sad_u32(self.POS_INF_F32, self.SNAN_F32, 0)
        self.assertEqual(dist, 1)

    def test_nan_clamp_saturation(self) -> None:
        # Adding accumulator to NaN distance that overflows 32-bit space saturates to 0xFFFFFFFF
        dist_clamped = ref_sad_u32(self.NEG_INF_F32, self.POS_ZERO_F32, 0x1000, clamp=True)
        # |0xFF800000 - 0| + 0x1000 = 0xFF801000 (< 0xFFFFFFFF, no overflow)
        self.assertEqual(dist_clamped, 0xFF80_1000)

        # Extreme overflow with clamp
        dist_overflow_clamped = ref_sad_u32(self.NEG_INF_F32, self.POS_ZERO_F32, 0x0080_0000, clamp=True)
        # 0xFF800000 + 0x00800000 = 0x100000000 -> clamps to 0xFFFFFFFF
        self.assertEqual(dist_overflow_clamped, MASK32)


class TestValuSadU32NegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_negative_control_sgpr_as_destination(self) -> None:
        with self.assertRaises(TypeError):
            VSadU32Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
        with self.assertRaises(TypeError):
            VSadHiU32Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
        with self.assertRaises(TypeError):
            VMqsadU32Instruction(
                dst=Operand.sgpr(0, count=4),
                src0=Operand.vgpr(4, count=2),
                src1=Operand.vgpr(6, count=1),
                src2=Operand.vgpr(8, count=4),
            )

    def test_negative_control_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Operand.vgpr(256)
        with self.assertRaises(ValueError):
            Operand.vgpr(253, count=4)  # 253 + 4 - 1 = 256 > 255
        with self.assertRaises(ValueError):
            Operand.sgpr(106)
        with self.assertRaises(ValueError):
            Operand.sgpr(104, count=4)  # 104 + 4 - 1 = 107 > 105

    def test_negative_control_literal_destination(self) -> None:
        with self.assertRaises(TypeError):
            VSadU32Instruction(
                dst=Operand.literal(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )

    def test_negative_control_wave_state_boundaries(self) -> None:
        with self.assertRaises(ValueError):
            SIMDWaveState(wave_size=16)
        wave = SIMDWaveState(wave_size=32)
        with self.assertRaises(IndexError):
            wave.is_lane_active(32)
        with self.assertRaises(IndexError):
            wave.is_lane_active(-1)

    def test_mutation_detection_signed_vs_unsigned_diff(self) -> None:
        def mutant_signed_diff(s0: int, s1: int) -> int:
            return abs(s32(s0) - s32(s1)) & MASK32

        # In unsigned arithmetic |0 - 0xFFFFFFFF| == 0xFFFFFFFF.
        # In signed arithmetic |0 - (-1)| == 1.
        correct = ref_abs_diff_u32(0, MASK32)
        mutant = mutant_signed_diff(0, MASK32)
        self.assertNotEqual(correct, mutant)
        self.assertEqual(correct, MASK32)
        self.assertEqual(mutant, 1)

    def test_mutation_detection_sad_hi_carry(self) -> None:
        def mutant_sad_hi(s0: int, s1: int, s2: int) -> int:
            # Faulty implementation: forgets difference and just does (s0 + s2) >> 32
            return ((s0 & MASK32) + (s2 & MASK32)) >> 32

        # For s0 = 0, s1 = 0xFFFFFFFF, s2 = 1:
        # Correct: diff = 0xFFFFFFFF, diff + 1 = 0x100000000 -> hi = 1
        # Mutant: 0 + 1 = 1 -> hi = 0
        correct = ref_sad_hi_u32(0, MASK32, 1)
        mutant = mutant_sad_hi(0, MASK32, 1)
        self.assertNotEqual(correct, mutant)
        self.assertEqual(correct, 1)
        self.assertEqual(mutant, 0)

    def test_mutation_detection_mqsad_mask_skipping(self) -> None:
        def mutant_quad_byte_sad(win: int, s1: int) -> int:
            # Faulty implementation: ignores mask check and always adds byte differences
            total = 0
            for i in range(4):
                total += abs(extract_byte(win, i) - extract_byte(s1, i))
            return total

        win = 0x0101_0101
        s1 = 0x0001_0001  # bytes 0 and 2 are 0
        correct = ref_quad_byte_sad(win, s1, masked=True)
        mutant = mutant_quad_byte_sad(win, s1)
        # In correct masked behavior: bytes 0 and 2 are skipped; bytes 1 and 3 match (diff=0) -> total = 0
        # In mutant unmasked behavior: bytes 0 and 2 have diff 1 each -> total = 2
        self.assertNotEqual(correct, mutant)
        self.assertEqual(correct, 0)
        self.assertEqual(mutant, 2)

    def test_mutation_detection_clamp_overflow(self) -> None:
        def mutant_clamped_sad(s0: int, s1: int, s2: int) -> int:
            # Faulty clamp: wraps modulo 2^32 instead of saturating
            return (ref_abs_diff_u32(s0, s1) + (s2 & MASK32)) & MASK32

        correct = ref_sad_u32(0, MASK32, 1, clamp=True)
        mutant = mutant_clamped_sad(0, MASK32, 1)
        self.assertNotEqual(correct, mutant)
        self.assertEqual(correct, MASK32)
        self.assertEqual(mutant, 0)


if __name__ == "__main__":
    unittest.main()
