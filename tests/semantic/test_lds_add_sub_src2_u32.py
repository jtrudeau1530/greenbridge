from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_add_sub_src2_u32.

Opcodes:
  - ds_add_src2_u32: Atomic unsigned 32-bit addition with secondary operand in LDS:
      tmp_a = LDS[A]
      tmp_b = LDS[B]
      LDS[A] = (tmp_a + tmp_b) & 0xFFFFFFFF
  - ds_sub_src2_u32: Atomic unsigned 32-bit subtraction with secondary operand in LDS:
      tmp_a = LDS[A]
      tmp_b = LDS[B]
      LDS[A] = (tmp_a - tmp_b) & 0xFFFFFFFF

Architecture & Specification:
  - AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture.
  - Scratchpad memory shared across all lanes of a workgroup.
  - Addressing:
      Byte address computed from VGPR vaddr.
      offset0 (8-bit immediate) and offset1 (8-bit immediate) specify offsets.
      Address A = (vaddr + offset0 * 4) & 0xFFFFFFFF.
      Address B = (vaddr + offset1 * 4) & 0xFFFFFFFF.
      Dword indices are obtained via (addr >> 2) & (lds_dwords - 1).
  - Non-returning: Results are written in-place to LDS; VGPRs remain unchanged.
"""

from dataclasses import dataclass
from enum import Enum
import sys
import unittest

MASK32 = 0xFFFFFFFF
DEFAULT_LDS_DWORDS = 16384  # 64 KiB default capacity in dwords (16384 * 4 bytes)
DEFAULT_NUM_LANES = 64     # Wave64 default width
DEFAULT_NUM_VGPRS = 256    # Standard GFX10 vector register file size


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def ref_ds_add_src2_u32(val_a: int, val_b: int) -> int:
    """Reference ISA calculation for ds_add_src2_u32.

    LDS[A] = (LDS[A] + LDS[B]) mod 2^32.
    """
    return u32(u32(val_a) + u32(val_b))


def ref_ds_sub_src2_u32(val_a: int, val_b: int) -> int:
    """Reference ISA calculation for ds_sub_src2_u32.

    LDS[A] = (LDS[A] - LDS[B]) mod 2^32.
    """
    return u32(u32(val_a) - u32(val_b))


def calc_lds_src2_addresses(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    lds_dwords: int = DEFAULT_LDS_DWORDS,
) -> tuple[int, int, int, int]:
    """Calculate byte addresses and dword indices for address A and address B.

    Address A = (vaddr + offset0 * 4) & MASK32
    Address B = (vaddr + offset1 * 4) & MASK32
    Dword index = (addr >> 2) & (lds_dwords - 1)

    Returns:
        tuple[int, int, int, int]: (byte_addr_a, byte_addr_b, dword_idx_a, dword_idx_b)
    """
    byte_addr_a = u32(vaddr + (offset0 * 4))
    byte_addr_b = u32(vaddr + (offset1 * 4))
    dword_idx_a = (byte_addr_a >> 2) & (lds_dwords - 1)
    dword_idx_b = (byte_addr_b >> 2) & (lds_dwords - 1)
    return byte_addr_a, byte_addr_b, dword_idx_a, dword_idx_b


# ==============================================================================
# Symbolic AST Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Supported DS src2 add/sub opcodes."""
    DS_ADD_SRC2_U32 = "ds_add_src2_u32"
    DS_SUB_SRC2_U32 = "ds_sub_src2_u32"


class RegType(Enum):
    """Register file class."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegisterOperand:
    """Symbolic register operand representation."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        if not (0 <= self.index < DEFAULT_NUM_VGPRS):
            raise ValueError(
                f"Register index out of bounds: {self.reg_type.value}{self.index} "
                f"(expected 0 <= index < {DEFAULT_NUM_VGPRS})"
            )

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}{self.index}"


@dataclass(frozen=True)
class LDSAddressExpression:
    """Symbolic representation of LDS address with dual offset descriptors."""
    base_reg: RegisterOperand
    offset0: int = 0
    offset1: int = 0
    gds: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.offset0 <= 0xFF):
            raise ValueError(f"offset0 out of 8-bit range: {self.offset0}")
        if not (0 <= self.offset1 <= 0xFF):
            raise ValueError(f"offset1 out of 8-bit range: {self.offset1}")
        if self.base_reg.reg_type != RegType.VGPR:
            raise TypeError("LDS base address register must be a VGPR")


@dataclass(frozen=True)
class DSSrc2Instruction:
    """Symbolic AST node for DS src2 atomic instruction."""
    opcode: DSOpcode
    addr: LDSAddressExpression

    def disasm(self) -> str:
        """Format canonical assembly mnemonic and operand string."""
        parts = [self.opcode.value, self.addr.base_reg.name]
        if self.addr.offset0 != 0:
            parts.append(f"offset0:{self.addr.offset0}")
        if self.addr.offset1 != 0:
            parts.append(f"offset1:{self.addr.offset1}")
        if self.addr.gds:
            parts.append("gds")
        return " ".join(parts)


class TestReferenceModelAndOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model."""

    def test_add_src2_basic_addition(self) -> None:
        self.assertEqual(ref_ds_add_src2_u32(0, 0), 0)
        self.assertEqual(ref_ds_add_src2_u32(100, 250), 350)
        self.assertEqual(ref_ds_add_src2_u32(1, 0xFFFE), 0xFFFF)
        self.assertEqual(ref_ds_add_src2_u32(0x12345678, 0x10000001), 0x22345679)

    def test_add_src2_overflow_wrapping(self) -> None:
        self.assertEqual(ref_ds_add_src2_u32(MASK32, 1), 0)
        self.assertEqual(ref_ds_add_src2_u32(MASK32, 2), 1)
        self.assertEqual(ref_ds_add_src2_u32(0x80000000, 0x80000000), 0)
        self.assertEqual(ref_ds_add_src2_u32(0xFFFFFFF0, 0x20), 0x10)

    def test_sub_src2_basic_subtraction(self) -> None:
        self.assertEqual(ref_ds_sub_src2_u32(0, 0), 0)
        self.assertEqual(ref_ds_sub_src2_u32(500, 200), 300)
        self.assertEqual(ref_ds_sub_src2_u32(100, 100), 0)
        self.assertEqual(ref_ds_sub_src2_u32(0x50000000, 0x10000000), 0x40000000)

    def test_sub_src2_underflow_wrapping(self) -> None:
        self.assertEqual(ref_ds_sub_src2_u32(0, 1), MASK32)
        self.assertEqual(ref_ds_sub_src2_u32(0, 5), MASK32 - 4)
        self.assertEqual(ref_ds_sub_src2_u32(10, 20), MASK32 - 9)
        self.assertEqual(ref_ds_sub_src2_u32(0x80000000, 0x80000001), MASK32)

    def test_address_calculation_and_alignment(self) -> None:
        byte_a, byte_b, idx_a, idx_b = calc_lds_src2_addresses(
            vaddr=0x2000, offset0=3, offset1=7, lds_dwords=2048
        )
        self.assertEqual(byte_a, 0x200C)
        self.assertEqual(byte_b, 0x201C)
        self.assertEqual(idx_a, (0x200C >> 2) & 2047)
        self.assertEqual(idx_b, (0x201C >> 2) & 2047)

    def test_self_referencing_aliasing(self) -> None:
        self.assertEqual(ref_ds_add_src2_u32(50, 50), 100)
        self.assertEqual(ref_ds_add_src2_u32(0x80000000, 0x80000000), 0)
        self.assertEqual(ref_ds_sub_src2_u32(50, 50), 0)
        self.assertEqual(ref_ds_sub_src2_u32(MASK32, MASK32), 0)


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_enumeration(self) -> None:
        self.assertEqual(DSOpcode.DS_ADD_SRC2_U32.value, "ds_add_src2_u32")
        self.assertEqual(DSOpcode.DS_SUB_SRC2_U32.value, "ds_sub_src2_u32")

    def test_register_operand_validation(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        self.assertEqual(v0.name, "v0")
        v127 = RegisterOperand(127, RegType.VGPR)
        self.assertEqual(v127.name, "v127")

        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)

    def test_address_expression_validation(self) -> None:
        v4 = RegisterOperand(4, RegType.VGPR)
        expr = LDSAddressExpression(base_reg=v4, offset0=2, offset1=6)
        self.assertEqual(expr.base_reg.name, "v4")
        self.assertEqual(expr.offset0, 2)
        self.assertEqual(expr.offset1, 6)

        with self.assertRaises(ValueError):
            LDSAddressExpression(base_reg=v4, offset0=-1)
        with self.assertRaises(ValueError):
            LDSAddressExpression(base_reg=v4, offset0=256)
        with self.assertRaises(ValueError):
            LDSAddressExpression(base_reg=v4, offset1=300)

        s0 = RegisterOperand(0, RegType.SGPR)
        with self.assertRaises(TypeError):
            LDSAddressExpression(base_reg=s0)

    def test_disasm_formatting(self) -> None:
        v1 = RegisterOperand(1, RegType.VGPR)
        addr = LDSAddressExpression(base_reg=v1, offset0=4, offset1=8)
        instr = DSSrc2Instruction(DSOpcode.DS_ADD_SRC2_U32, addr)
        self.assertEqual(instr.disasm(), "ds_add_src2_u32 v1 offset0:4 offset1:8")

        addr_zero = LDSAddressExpression(base_reg=v1, offset0=0, offset1=0)
        instr_sub = DSSrc2Instruction(DSOpcode.DS_SUB_SRC2_U32, addr_zero)
        self.assertEqual(instr_sub.disasm(), "ds_sub_src2_u32 v1")

    def test_ast_immutability_and_equality(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        addr1 = LDSAddressExpression(v0, offset0=1, offset1=2)
        addr2 = LDSAddressExpression(v0, offset0=1, offset1=2)
        instr1 = DSSrc2Instruction(DSOpcode.DS_ADD_SRC2_U32, addr1)
        instr2 = DSSrc2Instruction(DSOpcode.DS_ADD_SRC2_U32, addr2)
        self.assertEqual(instr1, instr2)


if __name__ == "__main__":
    unittest.main()
