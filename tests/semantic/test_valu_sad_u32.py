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
        # |100 - 40| + 20 = 80
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
        # Program:
        #   v_sad_u32 v0, v2, v3, v4
        #   v_sad_hi_u32 v1, v2, v3, v4
        # Input: |v2 - v3| + v4
        self.rf.vgprs[2] = 0x8000_0000
        self.rf.vgprs[3] = 0
        self.rf.vgprs[4] = 0x8000_0005
        # |0x80000000 - 0| + 0x80000005 = 0x100000005 -> hi=1, lo=5
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
        # v[4:5] = 64-bit source
        self.rf.vgprs[4] = 0x0403_0201
        self.rf.vgprs[5] = 0x0807_0605
        # v6 = template 32-bit
        self.rf.vgprs[6] = 0x0403_0201
        # v[8:11] = accumulators
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


if __name__ == "__main__":
    unittest.main()
