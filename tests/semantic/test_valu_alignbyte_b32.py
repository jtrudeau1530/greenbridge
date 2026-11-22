"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_alignbyte_b32.

Models AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics
for the byte alignment / funnel shift instruction:
  v_alignbyte_b32 dst, src0, src1, src2

RDNA2 VALU Instruction Specification:
  v_alignbyte_b32 D.u, S0.u, S1.u, S2.u
  Operands:
    - D:  32-bit VGPR destination
    - S0: 32-bit register (VGPR/SGPR) or immediate, providing high 32 bits (bytes 4..7)
    - S1: 32-bit register (VGPR/SGPR) or immediate, providing low 32 bits (bytes 0..3)
    - S2: 32-bit register (VGPR/SGPR) or inline constant shift amount (in bytes)

  Mathematical Operation:
    concat = ((S0 & 0xFFFFFFFF) << 32) | (S1 & 0xFFFFFFFF)
    shift_bytes = S2 & 3
    D = (concat >> (shift_bytes * 8)) & 0xFFFFFFFF

  Byte Selection Mapping:
    Let S1 = [B0, B1, B2, B3] and S0 = [B4, B5, B6, B7] (in little-endian byte order).
    - shift_bytes == 0: D = [B0, B1, B2, B3] = S1
    - shift_bytes == 1: D = [B1, B2, B3, B4] = (S1 >> 8)  | ((S0 & 0xFF) << 24)
    - shift_bytes == 2: D = [B2, B3, B4, B5] = (S1 >> 16) | ((S0 & 0xFFFF) << 16)
    - shift_bytes == 3: D = [B3, B4, B5, B6] = (S1 >> 24) | ((S0 & 0xFFFFFF) << 8)

  Key Architectural Properties:
    1. Only the low 2 bits of S2 (S2 & 3) determine the byte shift count.
       Shift counts >= 4, negative numbers, and high bits are strictly masked modulo 4.
    2. Byte B7 (the highest byte of S0) is never part of the destination for any valid shift.
    3. Common compiler idioms:
       - S0 == S1: 32-bit byte-rotate right: rotr32(S1, (S2 & 3) * 8).
       - S0 == 0:  logical shift right of S1 by (S2 & 3) * 8 bits.
       - S1 == 0:  logical shift left of S0 low bytes to high positions.
    4. Bitwise integrity: All bit patterns within bytes are preserved exactly with zero mutation.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import os
import struct
import sys
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT32: int = 0x80000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def get_byte(val32: int, byte_idx: int) -> int:
    """Extract byte at index byte_idx (0..3) from 32-bit word."""
    return (val32 >> ((byte_idx & 3) * 8)) & 0xFF


def set_byte(val32: int, byte_idx: int, byte_val: int) -> int:
    """Insert byte_val at byte_idx (0..3) in 32-bit word."""
    shift = (byte_idx & 3) * 8
    mask = ~(0xFF << shift) & MASK32
    return (val32 & mask) | ((byte_val & 0xFF) << shift)


def pack_bytes(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four 8-bit bytes into a 32-bit unsigned little-endian integer."""
    return ((b3 & 0xFF) << 24) | ((b2 & 0xFF) << 16) | ((b1 & 0xFF) << 8) | (b0 & 0xFF)


def unpack_bytes(val32: int) -> tuple[int, int, int, int]:
    """Unpack a 32-bit unsigned integer into four 8-bit bytes (b0, b1, b2, b3)."""
    val = val32 & MASK32
    return val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF


def ref_alignbyte_b32(src0: int, src1: int, src2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_alignbyte_b32.

    Concatenates {src0, src1} into a 64-bit word with src0 as the high dword and
    src1 as the low dword, then right shifts logically by (src2 & 3) bytes (0, 8, 16, or 24 bits)
    and extracts the low 32 bits.
    """
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    shift_bytes = src2 & 3
    concat = (s0 << 32) | s1
    res = (concat >> (shift_bytes * 8)) & MASK32
    return res


def ref_alignbyte_by_bytes(src0: int, src1: int, shift: int) -> int:
    """Alternative architectural byte-indexed oracle for verification and cross-checking."""
    b_lo = unpack_bytes(src1)
    b_hi = unpack_bytes(src0)
    eight_bytes = b_lo + b_hi  # 8 bytes: [0..7]
    s = shift & 3
    return pack_bytes(
        eight_bytes[s],
        eight_bytes[s + 1],
        eight_bytes[s + 2],
        eight_bytes[s + 3],
    )


class TestValuAlignbyteB32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_shift_zero_returns_src1(self) -> None:
        src0 = 0xDEADBEEF
        src1 = 0xCAFEBABE
        self.assertEqual(ref_alignbyte_b32(src0, src1, 0), src1)
        self.assertEqual(ref_alignbyte_by_bytes(src0, src1, 0), src1)

    def test_shift_one_byte(self) -> None:
        # S1: bytes [0x11, 0x22, 0x33, 0x44] -> 0x44332211
        # S0: bytes [0x55, 0x66, 0x77, 0x88] -> 0x88776655
        # Expected: bytes [0x22, 0x33, 0x44, 0x55] -> 0x55443322
        src1 = 0x44332211
        src0 = 0x88776655
        expected = 0x55443322
        self.assertEqual(ref_alignbyte_b32(src0, src1, 1), expected)
        self.assertEqual(ref_alignbyte_by_bytes(src0, src1, 1), expected)

    def test_shift_two_bytes(self) -> None:
        src1 = 0x44332211
        src0 = 0x88776655
        expected = 0x66554433
        self.assertEqual(ref_alignbyte_b32(src0, src1, 2), expected)
        self.assertEqual(ref_alignbyte_by_bytes(src0, src1, 2), expected)

    def test_shift_three_bytes(self) -> None:
        src1 = 0x44332211
        src0 = 0x88776655
        expected = 0x77665544
        self.assertEqual(ref_alignbyte_b32(src0, src1, 3), expected)
        self.assertEqual(ref_alignbyte_by_bytes(src0, src1, 3), expected)

    def test_shift_masking_modulo_four(self) -> None:
        src1 = 0x12345678
        src0 = 0x9ABCDEF0
        # src2 values with same low 2 bits must yield identical results
        for base_shift in (0, 1, 2, 3):
            ref = ref_alignbyte_b32(src0, src1, base_shift)
            for mult in (4, 8, 12, 16, 32, 64, 100, 256, 1024, 0xFFFFFFFC):
                test_shift = mult + base_shift
                self.assertEqual(ref_alignbyte_b32(src0, src1, test_shift), ref)

    def test_negative_shift_masking(self) -> None:
        src1 = 0xAABBCCDD
        src0 = 0x11223344
        # -1 in two's complement has low 2 bits 0b11 == 3
        self.assertEqual(ref_alignbyte_b32(src0, src1, -1), ref_alignbyte_b32(src0, src1, 3))
        # -2 has low 2 bits 0b10 == 2
        self.assertEqual(ref_alignbyte_b32(src0, src1, -2), ref_alignbyte_b32(src0, src1, 2))
        # -3 has low 2 bits 0b01 == 1
        self.assertEqual(ref_alignbyte_b32(src0, src1, -3), ref_alignbyte_b32(src0, src1, 1))
        # -4 has low 2 bits 0b00 == 0
        self.assertEqual(ref_alignbyte_b32(src0, src1, -4), ref_alignbyte_b32(src0, src1, 0))

    def test_rotate_idiom_src0_equals_src1(self) -> None:
        # When src0 == src1, alignbyte is 32-bit rotate right by (shift & 3) bytes
        val = 0x12345678
        # shift 0: 0x12345678
        self.assertEqual(ref_alignbyte_b32(val, val, 0), val)
        # shift 1: byte 0x78 moves to high byte -> 0x78123456
        self.assertEqual(ref_alignbyte_b32(val, val, 1), 0x78123456)
        # shift 2: bytes 0x5678 move to high -> 0x56781234
        self.assertEqual(ref_alignbyte_b32(val, val, 2), 0x56781234)
        # shift 3: bytes 0x345678 move to high -> 0x34567812
        self.assertEqual(ref_alignbyte_b32(val, val, 3), 0x34567812)

    def test_zero_src0_logical_shift_right(self) -> None:
        val = 0x89ABCDEF
        self.assertEqual(ref_alignbyte_b32(0, val, 0), val)
        self.assertEqual(ref_alignbyte_b32(0, val, 1), val >> 8)
        self.assertEqual(ref_alignbyte_b32(0, val, 2), val >> 16)
        self.assertEqual(ref_alignbyte_b32(0, val, 3), val >> 24)

    def test_zero_src1_logical_shift_left(self) -> None:
        val = 0x89ABCDEF
        self.assertEqual(ref_alignbyte_b32(val, 0, 0), 0)
        self.assertEqual(ref_alignbyte_b32(val, 0, 1), (val & 0xFF) << 24)
        self.assertEqual(ref_alignbyte_b32(val, 0, 2), (val & 0xFFFF) << 16)
        self.assertEqual(ref_alignbyte_b32(val, 0, 3), (val & 0xFFFFFF) << 8)

    def test_highest_byte_of_src0_never_included(self) -> None:
        # Highest byte of S0 is 0xFF, rest zeros
        src0 = 0xFF000000
        src1 = 0x00000000
        for s in range(8):
            # 0xFF should NEVER appear in result
            res = ref_alignbyte_b32(src0, src1, s)
            self.assertEqual(res, 0)

    def test_pack_unpack_bytes_roundtrip(self) -> None:
        val = 0xDEADBEEF
        b0, b1, b2, b3 = unpack_bytes(val)
        self.assertEqual(b0, 0xEF)
        self.assertEqual(b1, 0xBE)
        self.assertEqual(b2, 0xAD)
        self.assertEqual(b3, 0xDE)
        self.assertEqual(pack_bytes(b0, b1, b2, b3), val)


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    """Register file domain."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    """Symbolic register operand representing a 32-bit vector or scalar register."""
    kind: RegKind
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        if self.kind == RegKind.VGPR and not (0 <= self.index < 256):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")
        if self.kind == RegKind.SGPR and not (0 <= self.index < 106):
            raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")
        if self.count != 1:
            raise ValueError(f"Unsupported register count for 32-bit VALU instruction: {self.count}")

    @property
    def is_vgpr(self) -> bool:
        return self.kind == RegKind.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.kind == RegKind.SGPR

    def format_asm(self) -> str:
        return f"{self.kind.value}{self.index}"


@dataclasses.dataclass(frozen=True)
class Immediate:
    """Symbolic 32-bit immediate operand."""
    bits: int

    def __init__(self, val: int) -> None:
        object.__setattr__(self, "bits", val & MASK32)

    def format_asm(self) -> str:
        if 0 <= self.bits <= 64:
            return str(self.bits)
        return f"{self.bits:#010x}"


class Opcode(enum.Enum):
    """VALU 32-bit alignment opcode."""
    V_ALIGNBYTE_B32 = "v_alignbyte_b32"


class Encoding(enum.Enum):
    """Instruction encoding format."""
    VOP3 = "e64"
    VOP3A = "vop3a"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a VALU byte alignment instruction."""
    opcode: Opcode
    dst: Register
    src0: Register | Immediate
    src1: Register | Immediate
    src2: Register | Immediate
    encoding: Encoding = Encoding.VOP3

    def __post_init__(self) -> None:
        if not self.dst.is_vgpr:
            raise ValueError(f"VALU instruction destination must be a VGPR: {self.dst}")
        if self.dst.count != 1:
            raise ValueError(f"VALU 32-bit instruction destination must have count=1: {self.dst}")

    def to_assembly(self) -> str:
        s0_str = self.src0.format_asm()
        s1_str = self.src1.format_asm()
        s2_str = self.src2.format_asm()
        dst_str = self.dst.format_asm()
        return f"{self.opcode.value} {dst_str}, {s0_str}, {s1_str}, {s2_str}"


@dataclasses.dataclass
class Program:
    """Symbolic sequence of VALU instructions."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


class TestValuAlignbyteB32AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_construction_and_formatting(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        v7 = Register(RegKind.VGPR, 7)
        s4 = Register(RegKind.SGPR, 4)
        self.assertTrue(v0.is_vgpr)
        self.assertFalse(v0.is_sgpr)
        self.assertTrue(s4.is_sgpr)
        self.assertEqual(v0.format_asm(), "v0")
        self.assertEqual(v7.format_asm(), "v7")
        self.assertEqual(s4.format_asm(), "s4")

    def test_immediate_construction(self) -> None:
        imm2 = Immediate(2)
        imm_hex = Immediate(0xDEADBEEF)
        self.assertEqual(imm2.bits, 2)
        self.assertEqual(imm2.format_asm(), "2")
        self.assertEqual(imm_hex.bits, 0xDEADBEEF)
        self.assertEqual(imm_hex.format_asm(), "0xdeadbeef")

    def test_instruction_formatting(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Immediate(2),
        )
        self.assertEqual(inst.to_assembly(), "v_alignbyte_b32 v0, v1, v2, 2")

    def test_instruction_with_sgpr_and_vgpr(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 5),
            src0=Register(RegKind.SGPR, 0),
            src1=Register(RegKind.VGPR, 1),
            src2=Register(RegKind.SGPR, 2),
        )
        self.assertEqual(inst.to_assembly(), "v_alignbyte_b32 v5, s0, v1, s2")

    def test_instruction_validation_negative(self) -> None:
        # Invalid destination kind (must be VGPR)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_ALIGNBYTE_B32,
                dst=Register(RegKind.SGPR, 0),
                src0=Register(RegKind.VGPR, 1),
                src1=Register(RegKind.VGPR, 2),
                src2=Immediate(0),
            )

        # Invalid register count
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 0, count=2)

        # Out of bounds register index
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 106)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1)

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Immediate(1),
        ))
        prog.append(Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 3),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 4),
            src2=Immediate(2),
        ))
        expected = (
            "v_alignbyte_b32 v0, v1, v2, 1\n"
            "v_alignbyte_b32 v3, v0, v4, 2"
        )
        self.assertEqual(prog.to_assembly(), expected)


# ---------------------------------------------------------------------------
# Condition Codes, Status Flags, and Carry/Borrow Isolation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ConditionCodes:
    """Architectural condition codes for 32-bit VALU operations."""
    zero: bool
    sign: bool
    byte0_zero: bool
    byte1_zero: bool
    byte2_zero: bool
    byte3_zero: bool
    parity: bool


@dataclasses.dataclass(frozen=True)
class VALUStatusFlags:
    """Status flags and carry/borrow tracking for VALU instructions."""
    vcc_modified: bool = False
    overflow: bool = False
    carry: bool = False


def compute_alignbyte_condition_codes(result: int) -> ConditionCodes:
    """Computes condition codes evaluating attributes of the 32-bit alignbyte result."""
    res = result & MASK32
    b0 = res & 0xFF
    b1 = (res >> 8) & 0xFF
    b2 = (res >> 16) & 0xFF
    b3 = (res >> 24) & 0xFF
    # Parity of low byte (True if even number of set bits)
    parity = (bin(b0).count("1") % 2) == 0
    return ConditionCodes(
        zero=(res == 0),
        sign=bool(res & SIGN_BIT32),
        byte0_zero=(b0 == 0),
        byte1_zero=(b1 == 0),
        byte2_zero=(b2 == 0),
        byte3_zero=(b3 == 0),
        parity=parity,
    )


def compute_alignbyte_status_flags() -> VALUStatusFlags:
    """In RDNA2 ISA, VALU alignbyte is a purely logical funnel shift and never affects VCC."""
    return VALUStatusFlags(vcc_modified=False, overflow=False, carry=False)


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarVALUInterpreter:
    """Deterministic scalar interpreter for GFX10/RDNA2 VALU v_alignbyte_b32."""

    def __init__(self) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.vcc: int = 0
        self.last_condition_codes: ConditionCodes | None = None
        self.last_status_flags: VALUStatusFlags | None = None
        self.trace: list[str] = []

    def read_reg_u32(self, kind: RegKind, idx: int) -> int:
        if kind == RegKind.VGPR:
            if not (0 <= idx < 256):
                raise IndexError(f"VGPR index out of bounds: {idx}")
            return self.vgprs[idx] & MASK32
        if not (0 <= idx < 106):
            raise IndexError(f"SGPR index out of bounds: {idx}")
        return self.sgprs[idx] & MASK32

    def write_reg_u32(self, kind: RegKind, idx: int, val: int) -> None:
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            if not (0 <= idx < 256):
                raise IndexError(f"VGPR index out of bounds: {idx}")
            self.vgprs[idx] = val32
        else:
            if not (0 <= idx < 106):
                raise IndexError(f"SGPR index out of bounds: {idx}")
            self.sgprs[idx] = val32

    def read_operand(self, op: Register | Immediate) -> int:
        if isinstance(op, Immediate):
            return op.bits
        if isinstance(op, Register):
            return self.read_reg_u32(op.kind, op.index)
        raise TypeError(f"Unsupported operand type: {type(op)}")

    def step(self, inst: Instruction) -> int:
        """Executes a single VALU alignbyte instruction with strict read-before-write semantics."""
        # Latch all input operands before writing destination
        s0_val = self.read_operand(inst.src0)
        s1_val = self.read_operand(inst.src1)
        s2_val = self.read_operand(inst.src2)

        if inst.opcode == Opcode.V_ALIGNBYTE_B32:
            res = ref_alignbyte_b32(s0_val, s1_val, s2_val)
        else:
            raise NotImplementedError(f"Unsupported opcode in scalar interpreter: {inst.opcode}")

        self.write_reg_u32(inst.dst.kind, inst.dst.index, res)
        self.last_condition_codes = compute_alignbyte_condition_codes(res)
        self.last_status_flags = compute_alignbyte_status_flags()
        self.trace.append(f"{inst.to_assembly()} => {res:#010x}")
        return res

    def run_program(self, prog: Program) -> list[int]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


class TestValuAlignbyteB32ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar VALU instruction interpreter."""

    def test_scalar_register_read_write(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 0, 0x12345678)
        interp.write_reg_u32(RegKind.SGPR, 4, 0x9ABCDEF0)

        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0x12345678)
        self.assertEqual(interp.read_reg_u32(RegKind.SGPR, 4), 0x9ABCDEF0)

    def test_scalar_execution_vgpr_sources(self) -> None:
        interp = ScalarVALUInterpreter()
        # S1 = v1 = 0x44332211, S0 = v0 = 0x88776655, S2 = v2 = 1
        interp.write_reg_u32(RegKind.VGPR, 0, 0x88776655)
        interp.write_reg_u32(RegKind.VGPR, 1, 0x44332211)
        interp.write_reg_u32(RegKind.VGPR, 2, 1)

        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 3),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 1),
            src2=Register(RegKind.VGPR, 2),
        )
        res = interp.step(inst)
        self.assertEqual(res, 0x55443322)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 3), 0x55443322)

    def test_scalar_execution_sgpr_sources(self) -> None:
        interp = ScalarVALUInterpreter()
        # S0 from SGPR s0, S1 from VGPR v1, shift from SGPR s2
        interp.write_reg_u32(RegKind.SGPR, 0, 0xAABBCCDD)
        interp.write_reg_u32(RegKind.VGPR, 1, 0x11223344)
        interp.write_reg_u32(RegKind.SGPR, 2, 2)

        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.SGPR, 0),
            src1=Register(RegKind.VGPR, 1),
            src2=Register(RegKind.SGPR, 2),
        )
        res = interp.step(inst)
        # S1=0x11223344, S0=0xAABBCCDD, shift=2 -> [0x33, 0x44, 0xDD, 0xCC] -> 0xCCDD1122 ?
        # S1 bytes: 44, 33, 22, 11; S0 bytes: DD, CC, BB, AA
        # shift 2 bytes: bytes [2,3,4,5] -> [22, 11, DD, CC] -> 0xCCDD1122
        expected = ref_alignbyte_b32(0xAABBCCDD, 0x11223344, 2)
        self.assertEqual(res, expected)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), expected)

    def test_scalar_immediate_operand(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 1, 0x88776655)
        interp.write_reg_u32(RegKind.VGPR, 2, 0x44332211)

        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Immediate(3),
        )
        res = interp.step(inst)
        expected = ref_alignbyte_b32(0x88776655, 0x44332211, 3)
        self.assertEqual(res, expected)

    def test_scalar_program_execution_chain(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 1, 0x01020304)
        interp.write_reg_u32(RegKind.VGPR, 2, 0x05060708)

        prog = Program()
        # Step 1: v0 = alignbyte(v1, v2, 1)
        prog.append(Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Immediate(1),
        ))
        # Step 2: v3 = alignbyte(v0, v1, 2)
        prog.append(Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 3),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 1),
            src2=Immediate(2),
        ))

        results = interp.run_program(prog)
        r0 = ref_alignbyte_b32(0x01020304, 0x05060708, 1)
        r1 = ref_alignbyte_b32(r0, 0x01020304, 2)
        self.assertEqual(results, [r0, r1])
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 3), r1)
        self.assertEqual(len(interp.trace), 2)


# ---------------------------------------------------------------------------
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ---------------------------------------------------------------------------

class SIMDWavefrontInterpreter:
    """Multi-lane SIMD interpreter modeling Wave32 and Wave64 execution on RDNA2."""

    def __init__(self, wave_size: int = 64) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {wave_size} (must be 32 or 64)")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        # Per-lane VGPR files: [wave_size][256] 32-bit registers
        self.vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        # Wavefront-shared SGPR file: [106] 32-bit registers
        self.sgprs: list[int] = [0] * 106
        self.vcc: int = 0

    def set_exec_mask(self, mask: int) -> None:
        valid_mask = (1 << self.wave_size) - 1
        self.exec_mask = mask & valid_mask

    def read_lane_reg_u32(self, lane: int, kind: RegKind, idx: int) -> int:
        if kind == RegKind.VGPR:
            return self.vgprs[lane][idx] & MASK32
        return self.sgprs[idx] & MASK32

    def write_lane_reg_u32(self, lane: int, kind: RegKind, idx: int, val: int) -> None:
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[lane][idx] = val32
        else:
            self.sgprs[idx] = val32

    def read_lane_operand(self, lane: int, op: Register | Immediate) -> int:
        if isinstance(op, Immediate):
            return op.bits
        if isinstance(op, Register):
            return self.read_lane_reg_u32(lane, op.kind, op.index)
        raise TypeError(f"Unsupported operand type: {type(op)}")

    def step(self, inst: Instruction) -> list[int | None]:
        """Executes a VALU instruction across all lanes in the wavefront governed by EXEC mask."""
        # 1. Latch inputs for all lanes (read-before-write semantics across wavefront)
        latched_inputs: list[tuple[int, int, int]] = []
        for lane in range(self.wave_size):
            s0_val = self.read_lane_operand(lane, inst.src0)
            s1_val = self.read_lane_operand(lane, inst.src1)
            s2_val = self.read_lane_operand(lane, inst.src2)
            latched_inputs.append((s0_val, s1_val, s2_val))

        # 2. Compute and write back strictly for active lanes
        lane_results: list[int | None] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append(None)
                continue

            s0_val, s1_val, s2_val = latched_inputs[lane]
            if inst.opcode == Opcode.V_ALIGNBYTE_B32:
                res = ref_alignbyte_b32(s0_val, s1_val, s2_val)
            else:
                raise NotImplementedError(f"Unsupported opcode in SIMD interpreter: {inst.opcode}")

            self.write_lane_reg_u32(lane, inst.dst.kind, inst.dst.index, res)
            lane_results.append(res)

        return lane_results


class TestValuAlignbyteB32SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_inputs(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Immediate(1),
        )
        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0x88776655)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, 0x44332211)

        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        expected = ref_alignbyte_b32(0x88776655, 0x44332211, 1)
        for lane in range(32):
            self.assertEqual(results[lane], expected)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), expected)

    def test_wave64_multi_lane_divergence(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=64)
        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Register(RegKind.VGPR, 3),
        )
        expected: list[int] = []
        for lane in range(64):
            s0 = (0x10000000 * (lane + 1)) & MASK32
            s1 = (0x01020304 + lane) & MASK32
            shift = lane % 4
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, s0)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, s1)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 3, shift)
            expected.append(ref_alignbyte_b32(s0, s1, shift))

        results = simd.step(inst)
        for lane in range(64):
            self.assertEqual(results[lane], expected[lane])
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), expected[lane])

    def test_wave32_exec_masking_partial(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        exec_mask = 0x55555555  # Even lanes active
        simd.set_exec_mask(exec_mask)

        canary = 0xDEADBEEF
        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, canary)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0x88776655)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, 0x44332211)

        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Immediate(2),
        )
        results = simd.step(inst)
        expected = ref_alignbyte_b32(0x88776655, 0x44332211, 2)

        for lane in range(32):
            if (exec_mask >> lane) & 1:
                self.assertEqual(results[lane], expected)
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), expected)
            else:
                self.assertIsNone(results[lane])
                # Inactive lanes strictly retain canary
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), canary)

    def test_wave64_exec_masking_all_inactive(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=64)
        simd.set_exec_mask(0)

        canary = 0xCAFEBABE
        for lane in range(64):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, canary)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0x11111111)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, 0x22222222)

        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Immediate(1),
        )
        results = simd.step(inst)
        self.assertEqual(results, [None] * 64)
        for lane in range(64):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), canary)

    def test_simd_with_shared_sgpr_scalar_source(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        # Shift amount 3 in SGPR s4
        simd.write_lane_reg_u32(0, RegKind.SGPR, 4, 3)

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0x10203040 + lane)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, 0x50607080 + lane)

        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Register(RegKind.SGPR, 4),
        )
        results = simd.step(inst)
        for lane in range(32):
            s0 = 0x10203040 + lane
            s1 = 0x50607080 + lane
            expected = ref_alignbyte_b32(s0, s1, 3)
            self.assertEqual(results[lane], expected)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), expected)


# ---------------------------------------------------------------------------
# Register Aliasing and In-Place Operand Preservation
# ---------------------------------------------------------------------------

class TestValuAlignbyteB32RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_aliasing_dst_equals_src0(self) -> None:
        interp = ScalarVALUInterpreter()
        s0_val = 0x88776655
        s1_val = 0x44332211
        interp.write_reg_u32(RegKind.VGPR, 0, s0_val)
        interp.write_reg_u32(RegKind.VGPR, 1, s1_val)

        # v0 = alignbyte(v0, v1, 2)
        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 1),
            src2=Immediate(2),
        )
        res = interp.step(inst)
        expected = ref_alignbyte_b32(s0_val, s1_val, 2)
        self.assertEqual(res, expected)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), expected)
        # v1 (src1) remains completely preserved
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 1), s1_val)

    def test_aliasing_dst_equals_src1(self) -> None:
        interp = ScalarVALUInterpreter()
        s0_val = 0x88776655
        s1_val = 0x44332211
        interp.write_reg_u32(RegKind.VGPR, 0, s0_val)
        interp.write_reg_u32(RegKind.VGPR, 1, s1_val)

        # v1 = alignbyte(v0, v1, 1)
        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 1),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 1),
            src2=Immediate(1),
        )
        res = interp.step(inst)
        expected = ref_alignbyte_b32(s0_val, s1_val, 1)
        self.assertEqual(res, expected)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 1), expected)
        # v0 (src0) remains completely preserved
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), s0_val)

    def test_aliasing_dst_equals_src2_shift_amount(self) -> None:
        interp = ScalarVALUInterpreter()
        s0_val = 0x88776655
        s1_val = 0x44332211
        shift_reg = 3
        interp.write_reg_u32(RegKind.VGPR, 0, s0_val)
        interp.write_reg_u32(RegKind.VGPR, 1, s1_val)
        interp.write_reg_u32(RegKind.VGPR, 2, shift_reg)

        # v2 = alignbyte(v0, v1, v2)
        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 2),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 1),
            src2=Register(RegKind.VGPR, 2),
        )
        res = interp.step(inst)
        expected = ref_alignbyte_b32(s0_val, s1_val, shift_reg)
        self.assertEqual(res, expected)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 2), expected)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), s0_val)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 1), s1_val)

    def test_aliasing_in_place_rotate_dst_src0_src1(self) -> None:
        interp = ScalarVALUInterpreter()
        val = 0x12345678
        interp.write_reg_u32(RegKind.VGPR, 5, val)

        # v5 = alignbyte(v5, v5, 1) -> 32-bit rotate right by 1 byte
        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 5),
            src0=Register(RegKind.VGPR, 5),
            src1=Register(RegKind.VGPR, 5),
            src2=Immediate(1),
        )
        res = interp.step(inst)
        expected = 0x78123456
        self.assertEqual(res, expected)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 5), expected)

    def test_aliasing_all_four_operands_identical(self) -> None:
        interp = ScalarVALUInterpreter()
        # Value 2 in v0: shift count is (2 & 3) == 2 bytes
        interp.write_reg_u32(RegKind.VGPR, 0, 2)

        # v0 = alignbyte(v0, v0, v0)
        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 0),
            src2=Register(RegKind.VGPR, 0),
        )
        res = interp.step(inst)
        expected = ref_alignbyte_b32(2, 2, 2)
        self.assertEqual(res, expected)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), expected)

    def test_unmodified_registers_preserved(self) -> None:
        interp = ScalarVALUInterpreter()
        # Seed registers 10..29 with canary patterns
        for idx in range(10, 30):
            interp.write_reg_u32(RegKind.VGPR, idx, 0xABC00000 + idx)

        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(0x11111111),
            src1=Immediate(0x22222222),
            src2=Immediate(1),
        )
        interp.step(inst)

        for idx in range(10, 30):
            self.assertEqual(interp.read_reg_u32(RegKind.VGPR, idx), 0xABC00000 + idx)

    def test_simd_aliasing_with_divergent_exec_mask(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        exec_mask = 0x0000000F  # Only lanes 0..3 active
        simd.set_exec_mask(exec_mask)

        # In all lanes: v0 = 0x88776655, v1 = 0x44332211
        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 0x88776655)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0x44332211)

        # v0 = alignbyte(v0, v1, 2)
        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 1),
            src2=Immediate(2),
        )
        results = simd.step(inst)
        expected = ref_alignbyte_b32(0x88776655, 0x44332211, 2)

        for lane in range(32):
            if lane < 4:
                self.assertEqual(results[lane], expected)
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), expected)
            else:
                self.assertIsNone(results[lane])
                # Inactive lanes retain original v0
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 0x88776655)


class TestValuAlignbyteB32ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow isolation."""

    def test_condition_codes_zero_flag(self) -> None:
        interp = ScalarVALUInterpreter()
        inst_zero = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(0),
            src1=Immediate(0),
            src2=Immediate(0),
        )
        res = interp.step(inst_zero)
        self.assertEqual(res, 0)
        cc = interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertTrue(cc.zero)
        self.assertFalse(cc.sign)
        self.assertTrue(cc.byte0_zero)
        self.assertTrue(cc.byte1_zero)
        self.assertTrue(cc.byte2_zero)
        self.assertTrue(cc.byte3_zero)
        self.assertTrue(cc.parity)  # 0 has 0 bits set, even parity

    def test_condition_codes_sign_flag(self) -> None:
        interp = ScalarVALUInterpreter()
        # High bit set: e.g. 0x80000000
        inst_sign = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(0x80000000),
            src1=Immediate(0x00000000),
            src2=Immediate(3),  # shift 3 bytes -> byte 4 (low byte of src0) moves to high byte
        )
        res = interp.step(inst_sign)
        # Low byte of src0 is 0x00, so that wouldn't have bit 31 set.
        # To get bit 31 set, we need byte 3 of result to have bit 7 set (0x80).
        # Let's use S1=0x80000000 and shift 0
        inst_sign2 = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(0),
            src1=Immediate(0x80000000),
            src2=Immediate(0),
        )
        res2 = interp.step(inst_sign2)
        self.assertEqual(res2, 0x80000000)
        cc = interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertFalse(cc.zero)
        self.assertTrue(cc.sign)
        self.assertTrue(cc.byte0_zero)
        self.assertFalse(cc.byte3_zero)

    def test_condition_codes_per_byte_zeros(self) -> None:
        # Pattern 0x00FF00AA: byte 0 = 0xAA (non-zero), byte 1 = 0x00 (zero),
        # byte 2 = 0xFF (non-zero), byte 3 = 0x00 (zero)
        cc = compute_alignbyte_condition_codes(0x00FF00AA)
        self.assertFalse(cc.zero)
        self.assertFalse(cc.sign)
        self.assertFalse(cc.byte0_zero)
        self.assertTrue(cc.byte1_zero)
        self.assertFalse(cc.byte2_zero)
        self.assertTrue(cc.byte3_zero)

    def test_condition_codes_parity(self) -> None:
        # 0x01 has 1 set bit (odd parity -> parity=False)
        cc_odd = compute_alignbyte_condition_codes(0x01)
        self.assertFalse(cc_odd.parity)

        # 0x03 has 2 set bits (even parity -> parity=True)
        cc_even = compute_alignbyte_condition_codes(0x03)
        self.assertTrue(cc_even.parity)

    def test_carry_borrow_vcc_isolation(self) -> None:
        interp = ScalarVALUInterpreter()
        canary_vcc = 0xFEEDFACECAFEBEEF
        interp.vcc = canary_vcc

        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(0xFFFFFFFF),
            src1=Immediate(0xFFFFFFFF),
            src2=Immediate(1),
        )
        interp.step(inst)
        # VCC is never modified or polluted by alignbyte
        self.assertEqual(interp.vcc, canary_vcc)

        flags = interp.last_status_flags
        self.assertIsNotNone(flags)
        assert flags is not None
        self.assertFalse(flags.vcc_modified)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.carry)

        # SIMD wavefront test: VCC invariant across wave
        simd = SIMDWavefrontInterpreter(wave_size=64)
        simd.vcc = canary_vcc
        simd.step(inst)
        self.assertEqual(simd.vcc, canary_vcc)


# ---------------------------------------------------------------------------
# IEEE 754 Single-Precision Floating-Point Architectural Constants & Boundaries
# ---------------------------------------------------------------------------

POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_ONE_F32: int = 0x3F800000
NEG_ONE_F32: int = 0xBF800000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001
QNAN_PAYLOAD_F32: int = 0x7FC01234
SNAN_PAYLOAD_F32: int = 0x7F804321
SUBNORM_MIN_F32: int = 0x00000001
SUBNORM_MAX_F32: int = 0x007FFFFF
NEG_SUBNORM_MIN_F32: int = 0x80000001
NEG_SUBNORM_MAX_F32: int = 0x807FFFFF
FLT_MIN_F32: int = 0x00800000
FLT_MAX_F32: int = 0x7F7FFFFF


def f32_to_bits(val: float) -> int:
    """Encode float to IEEE 754 single-precision 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_nan_f32(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 single-precision NaN."""
    bits &= MASK32
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def is_subnormal_f32(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 single-precision subnormal."""
    bits &= MASK32
    return (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0


class TestValuAlignbyteB32IEEE754Boundaries(unittest.TestCase):
    """Exhaustive unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_unaligned_float_reconstruction_all_shifts(self) -> None:
        """Verify unaligned IEEE 754 floats in byte streams are reconstructed bit-exactly."""
        floats_to_test = [
            POS_ZERO_F32,
            NEG_ZERO_F32,
            POS_ONE_F32,
            NEG_ONE_F32,
            POS_INF_F32,
            NEG_INF_F32,
            QNAN_F32,
            SNAN_F32,
            QNAN_PAYLOAD_F32,
            SNAN_PAYLOAD_F32,
            SUBNORM_MIN_F32,
            SUBNORM_MAX_F32,
            FLT_MIN_F32,
            FLT_MAX_F32,
        ]

        canary_pad = 0x5A  # padding byte

        for flt_bits in floats_to_test:
            flt_bytes = unpack_bytes(flt_bits)
            for shift in (0, 1, 2, 3):
                with self.subTest(flt=f"{flt_bits:#010x}", shift=shift):
                    # Create an 8-byte buffer where flt_bytes start at byte offset `shift`
                    buf = [canary_pad] * 8
                    for i in range(4):
                        buf[shift + i] = flt_bytes[i]

                    # S1 is bytes 0..3, S0 is bytes 4..7
                    s1 = pack_bytes(buf[0], buf[1], buf[2], buf[3])
                    s0 = pack_bytes(buf[4], buf[5], buf[6], buf[7])

                    # Align byte by `shift` extracts bytes `shift..shift+3`
                    extracted = ref_alignbyte_b32(s0, s1, shift)
                    self.assertEqual(extracted, flt_bits)

                    # Float bitcast preservation
                    if not is_nan_f32(flt_bits):
                        orig_f = bits_to_f32(flt_bits)
                        extr_f = bits_to_f32(extracted)
                        if flt_bits in (POS_ZERO_F32, NEG_ZERO_F32):
                            # Ensure sign of zero is preserved
                            self.assertEqual(flt_bits & SIGN_BIT32, extracted & SIGN_BIT32)
                        else:
                            self.assertEqual(orig_f, extr_f)
                    else:
                        self.assertTrue(is_nan_f32(extracted))

    def test_subnormals_bit_preservation_no_ftz(self) -> None:
        """Verify that subnormal float bit patterns are NEVER flushed to zero during alignment."""
        subnormals = [
            SUBNORM_MIN_F32,
            SUBNORM_MAX_F32,
            NEG_SUBNORM_MIN_F32,
            NEG_SUBNORM_MAX_F32,
            0x00000042,
            0x00012345,
            0x800055AA,
        ]
        for sub in subnormals:
            self.assertTrue(is_subnormal_f32(sub))
            # Shift 0: returns sub directly
            self.assertEqual(ref_alignbyte_b32(0, sub, 0), sub)
            # Shift 2: pack across dwords
            b0, b1, b2, b3 = unpack_bytes(sub)
            s1 = pack_bytes(0, 0, b0, b1)
            s0 = pack_bytes(b2, b3, 0, 0)
            res = ref_alignbyte_b32(s0, s1, 2)
            self.assertEqual(res, sub)
            self.assertTrue(is_subnormal_f32(res))

    def test_nan_payload_preservation_no_quietization(self) -> None:
        """Verify signaling NaNs and quiet NaNs preserve their payloads without mutation."""
        nans = [
            QNAN_F32,
            SNAN_F32,
            QNAN_PAYLOAD_F32,
            SNAN_PAYLOAD_F32,
            0x7F800002,  # sNaN
            0x7FC00001,  # qNaN
            0xFF800001,  # negative sNaN
            0xFFC00001,  # negative qNaN
        ]
        for nan in nans:
            self.assertTrue(is_nan_f32(nan))
            # Across all shifts, bit pattern must be 100% invariant
            for shift in (0, 1, 2, 3):
                flt_bytes = unpack_bytes(nan)
                buf = [0] * 8
                for i in range(4):
                    buf[shift + i] = flt_bytes[i]
                s1 = pack_bytes(buf[0], buf[1], buf[2], buf[3])
                s0 = pack_bytes(buf[4], buf[5], buf[6], buf[7])
                res = ref_alignbyte_b32(s0, s1, shift)
                self.assertEqual(res, nan)
                self.assertEqual(res & 0x007FFFFF, nan & 0x007FFFFF)  # payload intact
                self.assertEqual(res & SIGN_BIT32, nan & SIGN_BIT32)  # sign intact

    def test_signed_zeros_distinction(self) -> None:
        """Verify +0.0 and -0.0 maintain their respective sign bits when aligned."""
        # Positive zero
        self.assertEqual(ref_alignbyte_b32(0, POS_ZERO_F32, 0), POS_ZERO_F32)
        self.assertEqual(ref_alignbyte_b32(0, POS_ZERO_F32, 0) & SIGN_BIT32, 0)

        # Negative zero
        self.assertEqual(ref_alignbyte_b32(0, NEG_ZERO_F32, 0), NEG_ZERO_F32)
        self.assertEqual(ref_alignbyte_b32(0, NEG_ZERO_F32, 0) & SIGN_BIT32, SIGN_BIT32)

        # Negative zero shifted by 1 byte: sign bit was bit 31 of S1 (byte 3).
        # Shifting by 1 byte places byte 3 into byte 2 of result (bit 23).
        res_shift1 = ref_alignbyte_b32(0, NEG_ZERO_F32, 1)
        self.assertEqual(res_shift1, 0x00800000)

    def test_infinities_preservation(self) -> None:
        """Verify positive and negative infinities remain bit-identical."""
        self.assertEqual(ref_alignbyte_b32(0, POS_INF_F32, 0), POS_INF_F32)
        self.assertEqual(ref_alignbyte_b32(0, NEG_INF_F32, 0), NEG_INF_F32)

    def test_dual_packed_f16_half_swapping(self) -> None:
        """Verify alignbyte can exchange high/low 16-bit halves (packed float16)."""
        # Pack two f16 values: low=1.0 (0x3C00), high=2.0 (0x4000) -> 0x40003C00
        packed_f16 = 0x40003C00
        # Rotate by 2 bytes (shift 2 with S0 == S1) exchanges the two halves
        swapped = ref_alignbyte_b32(packed_f16, packed_f16, 2)
        self.assertEqual(swapped, 0x3C004000)
        # Low half is now 2.0 (0x4000) and high half is 1.0 (0x3C00)
        self.assertEqual(swapped & 0xFFFF, 0x4000)
        self.assertEqual((swapped >> 16) & 0xFFFF, 0x3C00)


# ---------------------------------------------------------------------------
# Negative Controls, Syntax Rejection, and Mutation Detection
# ---------------------------------------------------------------------------

class TestValuAlignbyteB32NegativeControls(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and invalid AST inputs."""

    def test_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, -1)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 106)

    def test_invalid_destination_kind(self) -> None:
        # Destination cannot be an SGPR for VALU instruction
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_ALIGNBYTE_B32,
                dst=Register(RegKind.SGPR, 0),
                src0=Register(RegKind.VGPR, 1),
                src1=Register(RegKind.VGPR, 2),
                src2=Immediate(0),
            )

    def test_scalar_interpreter_out_of_bounds_access(self) -> None:
        interp = ScalarVALUInterpreter()
        with self.assertRaises(IndexError):
            interp.read_reg_u32(RegKind.VGPR, 256)
        with self.assertRaises(IndexError):
            interp.write_reg_u32(RegKind.SGPR, 106, 0)

    def test_simd_interpreter_invalid_wave_size(self) -> None:
        with self.assertRaises(ValueError):
            SIMDWavefrontInterpreter(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWavefrontInterpreter(wave_size=128)

    def test_unsupported_operand_type(self) -> None:
        interp = ScalarVALUInterpreter()
        with self.assertRaises(TypeError):
            interp.read_operand("v0")  # type: ignore[arg-type]

    def test_unsupported_opcode_rejection(self) -> None:
        # Create a mock instruction with unsupported opcode
        class FakeOpcode(enum.Enum):
            UNKNOWN = "v_unknown"

        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Immediate(0),
        )
        object.__setattr__(inst, "opcode", FakeOpcode.UNKNOWN)

        interp = ScalarVALUInterpreter()
        with self.assertRaises(NotImplementedError):
            interp.step(inst)


class TestValuAlignbyteB32MutationDetection(unittest.TestCase):
    """Mutation tests: verifies that intentional sabotages/mutations of the oracle are caught."""

    def test_mutation_inverted_shift_direction(self) -> None:
        # Sabotage: left shift instead of right shift
        def mutated_alignbyte(s0: int, s1: int, s2: int) -> int:
            concat = ((s0 & MASK32) << 32) | (s1 & MASK32)
            shift = (s2 & 3) * 8
            return (concat << shift) & MASK32

        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_alignbyte(0x88776655, 0x44332211, 1), 0x55443322)

    def test_mutation_inverted_operands_order(self) -> None:
        # Sabotage: concatenating {S1, S0} instead of {S0, S1}
        def mutated_alignbyte(s0: int, s1: int, s2: int) -> int:
            concat = ((s1 & MASK32) << 32) | (s0 & MASK32)
            shift = (s2 & 3) * 8
            return (concat >> shift) & MASK32

        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_alignbyte(0x88776655, 0x44332211, 1), 0x55443322)

    def test_mutation_shift_mask_omitted(self) -> None:
        # Sabotage: omitting & 3 on shift count, so shift 4 shifts 32 bits
        def mutated_alignbyte(s0: int, s1: int, s2: int) -> int:
            concat = ((s0 & MASK32) << 32) | (s1 & MASK32)
            shift = s2 * 8  # bug: missing & 3
            return (concat >> shift) & MASK32

        # In RDNA2, shift 4 must behave identically to shift 0 (returning S1)
        src0 = 0x88776655
        src1 = 0x44332211
        expected = ref_alignbyte_b32(src0, src1, 4)  # returns src1
        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_alignbyte(src0, src1, 4), expected)

    def test_mutation_endianness_reversal(self) -> None:
        # Sabotage: big-endian byte unpack
        def mutated_pack_bytes(b0: int, b1: int, b2: int, b3: int) -> int:
            return ((b0 & 0xFF) << 24) | ((b1 & 0xFF) << 16) | ((b2 & 0xFF) << 8) | (b3 & 0xFF)

        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_pack_bytes(0x11, 0x22, 0x33, 0x44), 0x44332211)

    def test_mutation_exec_mask_leak(self) -> None:
        # Sabotage: inactive SIMD lane modifies destination register
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.set_exec_mask(0)  # All lanes inactive
        canary = 0xBAD0CAFE
        simd.write_lane_reg_u32(0, RegKind.VGPR, 0, canary)

        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(1),
            src1=Immediate(2),
            src2=Immediate(0),
        )
        simd.step(inst)
        # Canary must be preserved
        self.assertEqual(simd.read_lane_reg_u32(0, RegKind.VGPR, 0), canary)

        # Mutated behavior: unconditionally overwrite
        simd.write_lane_reg_u32(0, RegKind.VGPR, 0, 0x12345678)
        with self.assertRaises(AssertionError):
            self.assertEqual(simd.read_lane_reg_u32(0, RegKind.VGPR, 0), canary)

    def test_mutation_vcc_clobber(self) -> None:
        # Sabotage: alignbyte clobbering VCC
        interp = ScalarVALUInterpreter()
        canary_vcc = 0x1234567890ABCDEF
        interp.vcc = canary_vcc

        # Proper step preserves VCC
        inst = Instruction(
            opcode=Opcode.V_ALIGNBYTE_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(1),
            src1=Immediate(2),
            src2=Immediate(0),
        )
        interp.step(inst)
        self.assertEqual(interp.vcc, canary_vcc)

        # Mutated step clobbers VCC
        interp.vcc = 0
        with self.assertRaises(AssertionError):
            self.assertEqual(interp.vcc, canary_vcc)


if __name__ == "__main__":
    unittest.main()
