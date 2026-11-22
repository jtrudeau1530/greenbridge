from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_perm_b32.

Tests the vector byte permutation instruction:
    v_perm_b32 D, S0, S1, S2
Lowered semantics:
    Destination byte i (for i in 0..3) is chosen based on selector byte i (S2[8*i+7 : 8*i]):
    - 0..3: selects byte 0..3 of S1 (bits [7:0], [15:8], [23:16], [31:24])
    - 4..7: selects byte 0..3 of S0 (bits [7:0], [15:8], [23:16], [31:24])
    - 8: sign-extension of S1 bit 31 (0xFF if negative, else 0x00)
    - 9: sign-extension of S0 bit 15 (0xFF if negative, else 0x00)
    - 10: sign-extension of S0 bit 31 (0xFF if negative, else 0x00)
    - 11: sign-extension replicate (0xFF if S1 bit 7 else 0x00)
    - 12: constant 0x00
    - >= 13: constant 0xFF

Architectural model:
    32-bit vector and scalar registers (VGPRs/SGPRs) with per-byte lane slicing,
    selector decoding, and bit-exact execution. Standard library only; zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

# ==============================================================================
# Architectural Constants
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
BYTE_MASK: int = 0xFF
BYTE_WIDTH: int = 8

# Selector codes
SEL_SRC1_BYTE0: int = 0
SEL_SRC1_BYTE1: int = 1
SEL_SRC1_BYTE2: int = 2
SEL_SRC1_BYTE3: int = 3

SEL_SRC0_BYTE0: int = 4
SEL_SRC0_BYTE1: int = 5
SEL_SRC0_BYTE2: int = 6
SEL_SRC0_BYTE3: int = 7

SEL_SIGN_EXT_SRC1_B3: int = 8   # S1[31] replicated
SEL_SIGN_EXT_SRC0_B1: int = 9   # S0[15] replicated
SEL_SIGN_EXT_SRC0_B3: int = 10  # S0[31] replicated
SEL_SIGN_EXT_SRC1_B0: int = 11  # S1[7] replicated
SEL_ZERO: int = 12              # Constant 0x00
SEL_ONES_MIN: int = 13          # Constant 0xFF (>= 13)


# ==============================================================================
# Bit Manipulation and Byte Extraction Utilities
# ==============================================================================

def extract_byte(val: int, idx: int) -> int:
    """Extract byte idx (0..3) from an unsigned 32-bit integer."""
    if not 0 <= idx <= 3:
        raise ValueError(f"Byte index out of range [0, 3]: {idx}")
    return (val >> (idx * BYTE_WIDTH)) & BYTE_MASK


def pack_bytes(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack 4 unsigned bytes (b0..b3) into an unsigned 32-bit integer."""
    return ((b0 & BYTE_MASK) |
            ((b1 & BYTE_MASK) << 8) |
            ((b2 & BYTE_MASK) << 16) |
            ((b3 & BYTE_MASK) << 24)) & MASK32


def make_selector(b0: int, b1: int, b2: int, b3: int) -> int:
    """Construct a 32-bit selector word from 4 individual byte selector values."""
    return pack_bytes(b0, b1, b2, b3)


def permute_single_byte(src0: int, src1: int, sel_code: int) -> int:
    """Evaluate a single byte according to RDNA2 v_perm_b32 hardware semantics."""
    sel = sel_code & BYTE_MASK
    if 0 <= sel <= 3:
        return extract_byte(src1, sel)
    elif 4 <= sel <= 7:
        return extract_byte(src0, sel - 4)
    elif sel == SEL_SIGN_EXT_SRC1_B3:
        return 0xFF if bool(src1 & 0x8000_0000) else 0x00
    elif sel == SEL_SIGN_EXT_SRC0_B1:
        return 0xFF if bool(src0 & 0x0000_8000) else 0x00
    elif sel == SEL_SIGN_EXT_SRC0_B3:
        return 0xFF if bool(src0 & 0x8000_0000) else 0x00
    elif sel == SEL_SIGN_EXT_SRC1_B0:
        return 0xFF if bool(src1 & 0x0000_0080) else 0x00
    elif sel == SEL_ZERO:
        return 0x00
    else:  # sel >= SEL_ONES_MIN (13..255)
        return 0xFF


def oracle_v_perm_b32(src0: int, src1: int, selector: int) -> int:
    """Reference mathematical oracle for RDNA2 v_perm_b32.
    
    src0: upper 4 bytes pool (indices 4..7)
    src1: lower 4 bytes pool (indices 0..3)
    selector: 32-bit control word with 4 selector bytes
    """
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    sel = selector & MASK32

    out_b0 = permute_single_byte(s0, s1, extract_byte(sel, 0))
    out_b1 = permute_single_byte(s0, s1, extract_byte(sel, 1))
    out_b2 = permute_single_byte(s0, s1, extract_byte(sel, 2))
    out_b3 = permute_single_byte(s0, s1, extract_byte(sel, 3))

    return pack_bytes(out_b0, out_b1, out_b2, out_b3)


# ==============================================================================
# Architectural Register Models
# ==============================================================================

class VGPRWord:
    """Architectural model of a 32-bit vector register (VGPR)."""

    def __init__(self, value: int = 0) -> None:
        self.raw: int = value & MASK32

    @property
    def u32(self) -> int:
        return self.raw

    @u32.setter
    def u32(self, val: int) -> None:
        self.raw = val & MASK32

    @property
    def i32(self) -> int:
        return struct.unpack("<i", struct.pack("<I", self.raw))[0]

    @property
    def is_negative(self) -> bool:
        return bool(self.raw & 0x8000_0000)

    def get_byte(self, idx: int) -> int:
        return extract_byte(self.raw, idx)

    def set_byte(self, idx: int, b: int) -> None:
        if not 0 <= idx <= 3:
            raise ValueError(f"Byte index out of range: {idx}")
        shift = idx * 8
        mask = ~(0xFF << shift) & MASK32
        self.raw = (self.raw & mask) | ((b & 0xFF) << shift)

    def __repr__(self) -> str:
        return f"VGPRWord(0x{self.raw:08X})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VGPRWord):
            return self.raw == other.raw
        if isinstance(other, int):
            return self.raw == (other & MASK32)
        return NotImplemented


class PermSelectorWord:
    """Architectural model of a 32-bit v_perm_b32 selector word."""

    def __init__(self, raw: int = 0) -> None:
        self.raw: int = raw & MASK32

    @classmethod
    def from_bytes(cls, b0: int, b1: int, b2: int, b3: int) -> PermSelectorWord:
        return cls(pack_bytes(b0, b1, b2, b3))

    @property
    def sel0(self) -> int:
        return extract_byte(self.raw, 0)

    @property
    def sel1(self) -> int:
        return extract_byte(self.raw, 1)

    @property
    def sel2(self) -> int:
        return extract_byte(self.raw, 2)

    @property
    def sel3(self) -> int:
        return extract_byte(self.raw, 3)

    @classmethod
    def identity_src1(cls) -> PermSelectorWord:
        """Picks bytes 0, 1, 2, 3 directly from src1."""
        return cls.from_bytes(0, 1, 2, 3)

    @classmethod
    def identity_src0(cls) -> PermSelectorWord:
        """Picks bytes 0, 1, 2, 3 directly from src0 (selectors 4, 5, 6, 7)."""
        return cls.from_bytes(4, 5, 6, 7)

    @classmethod
    def bswap_src1(cls) -> PermSelectorWord:
        """Reverses byte order of src1."""
        return cls.from_bytes(3, 2, 1, 0)

    @classmethod
    def bswap_src0(cls) -> PermSelectorWord:
        """Reverses byte order of src0."""
        return cls.from_bytes(7, 6, 5, 4)

    @classmethod
    def broadcast_byte(cls, src_byte_idx: int) -> PermSelectorWord:
        """Replicates a single source byte into all 4 destination byte positions."""
        return cls.from_bytes(src_byte_idx, src_byte_idx, src_byte_idx, src_byte_idx)


# ==============================================================================
# Unit Tests: Reference Oracle and Architectural Model
# ==============================================================================

class TestPermB32ReferenceOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural models."""

    def test_identity_src1_selection(self):
        src0 = 0xAABBCCDD
        src1 = 0x11223344
        sel = PermSelectorWord.identity_src1().raw
        self.assertEqual(oracle_v_perm_b32(src0, src1, sel), 0x11223344)

    def test_identity_src0_selection(self):
        src0 = 0xAABBCCDD
        src1 = 0x11223344
        sel = PermSelectorWord.identity_src0().raw
        self.assertEqual(oracle_v_perm_b32(src0, src1, sel), 0xAABBCCDD)

    def test_cross_source_interleaving(self):
        # Pick byte 0 of src1, byte 0 of src0, byte 1 of src1, byte 1 of src0
        # src1: [44, 33, 22, 11] -> byte 0=0x44, byte 1=0x33
        # src0: [DD, CC, BB, AA] -> byte 0=0xDD, byte 1=0xCC
        # Expected result bytes: [44, DD, 33, CC] -> 0xCC33DD44
        src0 = 0xAABBCCDD
        src1 = 0x11223344
        sel = make_selector(0, 4, 1, 5)
        self.assertEqual(oracle_v_perm_b32(src0, src1, sel), 0xCC33DD44)

    def test_endianness_bswap(self):
        src1 = 0x12345678
        src0 = 0x9ABCDEF0
        # Reverse src1: [78, 56, 34, 12] -> [12, 34, 56, 78]
        sel_rev1 = PermSelectorWord.bswap_src1().raw
        self.assertEqual(oracle_v_perm_b32(src0, src1, sel_rev1), 0x78563412)

        # Reverse src0
        sel_rev0 = PermSelectorWord.bswap_src0().raw
        self.assertEqual(oracle_v_perm_b32(src0, src1, sel_rev0), 0xF0DEBC9A)

    def test_byte_broadcast(self):
        src1 = 0x01020304  # byte 0 = 0x04
        sel = PermSelectorWord.broadcast_byte(0).raw
        self.assertEqual(oracle_v_perm_b32(0, src1, sel), 0x04040404)

        src0 = 0x88776655  # byte 7 (idx 3 of src0) = 0x88
        sel7 = PermSelectorWord.broadcast_byte(7).raw
        self.assertEqual(oracle_v_perm_b32(src0, 0, sel7), 0x88888888)

    def test_constant_zero_fill(self):
        src0 = 0xFFFFFFFF
        src1 = 0xFFFFFFFF
        sel = make_selector(SEL_ZERO, SEL_ZERO, SEL_ZERO, SEL_ZERO)
        self.assertEqual(oracle_v_perm_b32(src0, src1, sel), 0x00000000)

        # Mix zero fill with selected bytes
        sel_mix = make_selector(0, SEL_ZERO, 4, SEL_ZERO)
        res = oracle_v_perm_b32(0xAABBCCDD, 0x11223344, sel_mix)
        # byte 0: src1[0] = 0x44, byte 1: 0, byte 2: src0[0] = 0xDD, byte 3: 0
        self.assertEqual(res, 0x00DD0044)

    def test_constant_ones_fill(self):
        src0 = 0x00000000
        src1 = 0x00000000
        # Code 13
        sel13 = make_selector(13, 14, 15, 255)
        self.assertEqual(oracle_v_perm_b32(src0, src1, sel13), 0xFFFFFFFF)

        # Mix ones fill with selected bytes
        sel_mix = make_selector(0, 13, 4, 255)
        res = oracle_v_perm_b32(0x12345678, 0x9ABCDEF0, sel_mix)
        # byte 0: src1[0]=0xF0, byte 1: 0xFF, byte 2: src0[0]=0x78, byte 3: 0xFF
        self.assertEqual(res, 0xFF78FFF0)

    def test_sign_extension_modes(self):
        # SEL_SIGN_EXT_SRC1_B3 (8): sign of S1 bit 31
        s1_pos = 0x7FFF_FFFF  # bit 31 is 0
        s1_neg = 0x8000_0000  # bit 31 is 1
        sel8 = make_selector(8, 8, 8, 8)
        self.assertEqual(oracle_v_perm_b32(0, s1_pos, sel8), 0x00000000)
        self.assertEqual(oracle_v_perm_b32(0, s1_neg, sel8), 0xFFFFFFFF)

        # SEL_SIGN_EXT_SRC0_B1 (9): sign of S0 bit 15
        s0_pos15 = 0x0000_7FFF  # bit 15 is 0
        s0_neg15 = 0x0000_8000  # bit 15 is 1
        sel9 = make_selector(9, 9, 9, 9)
        self.assertEqual(oracle_v_perm_b32(s0_pos15, 0, sel9), 0x00000000)
        self.assertEqual(oracle_v_perm_b32(s0_neg15, 0, sel9), 0xFFFFFFFF)

        # SEL_SIGN_EXT_SRC0_B3 (10): sign of S0 bit 31
        s0_pos31 = 0x7FFF_FFFF
        s0_neg31 = 0x8000_0000
        sel10 = make_selector(10, 10, 10, 10)
        self.assertEqual(oracle_v_perm_b32(s0_pos31, 0, sel10), 0x00000000)
        self.assertEqual(oracle_v_perm_b32(s0_neg31, 0, sel10), 0xFFFFFFFF)

    def test_vgpr_word_model(self):
        vw = VGPRWord(0x12345678)
        self.assertEqual(vw.u32, 0x12345678)
        self.assertEqual(vw.get_byte(0), 0x78)
        self.assertEqual(vw.get_byte(1), 0x56)
        self.assertEqual(vw.get_byte(2), 0x34)
        self.assertEqual(vw.get_byte(3), 0x12)

        vw.set_byte(0, 0xAA)
        self.assertEqual(vw.u32, 0x123456AA)
        vw.set_byte(3, 0xFF)
        self.assertEqual(vw.u32, 0xFF3456AA)
        self.assertTrue(vw.is_negative)

    def test_selector_word_model(self):
        sw = PermSelectorWord.from_bytes(1, 2, 3, 4)
        self.assertEqual(sw.sel0, 1)
        self.assertEqual(sw.sel1, 2)
        self.assertEqual(sw.sel2, 3)
        self.assertEqual(sw.sel3, 4)
        self.assertEqual(sw.raw, 0x04030201)


# ==============================================================================
# Symbolic AST and Typed Expressions
# ==============================================================================

class Opcode(Enum):
    V_PERM_B32 = "v_perm_b32"


class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    INLINE_CONST = auto()
    LITERAL_CONST = auto()


class TypeKind(Enum):
    RAW32 = "b32"
    UINT32 = "u32"
    INT32 = "i32"
    BYTE_LANE = "u8"


@dataclass
class ASTOperand:
    """Represents an operand in an RDNA2 instruction AST."""
    name: str
    kind: OperandKind
    type_kind: TypeKind = TypeKind.RAW32
    reg_index: Optional[int] = None
    constant_value: Optional[int] = None

    @classmethod
    def vgpr(cls, idx: int, type_kind: TypeKind = TypeKind.RAW32) -> ASTOperand:
        return cls(name=f"v{idx}", kind=OperandKind.VGPR, type_kind=type_kind, reg_index=idx)

    @classmethod
    def sgpr(cls, idx: int, type_kind: TypeKind = TypeKind.RAW32) -> ASTOperand:
        return cls(name=f"s{idx}", kind=OperandKind.SGPR, type_kind=type_kind, reg_index=idx)

    @classmethod
    def inline_const(cls, val: int, type_kind: TypeKind = TypeKind.RAW32) -> ASTOperand:
        return cls(name=str(val), kind=OperandKind.INLINE_CONST, type_kind=type_kind, constant_value=val)

    @classmethod
    def literal(cls, val: int, type_kind: TypeKind = TypeKind.RAW32) -> ASTOperand:
        return cls(name=f"0x{val & MASK32:08X}", kind=OperandKind.LITERAL_CONST, type_kind=type_kind, constant_value=val & MASK32)

    def to_asm(self) -> str:
        return self.name

    @property
    def is_vgpr(self) -> bool:
        return self.kind == OperandKind.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.kind == OperandKind.SGPR

    @property
    def is_constant(self) -> bool:
        return self.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST)


@dataclass
class ASTInstruction:
    """Represents a VOP3 instruction in the symbolic AST."""
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand
    src2: ASTOperand

    def to_asm(self) -> str:
        return f"{self.opcode.value} {self.dst.to_asm()}, {self.src0.to_asm()}, {self.src1.to_asm()}, {self.src2.to_asm()}"

    def validate(self) -> None:
        if not self.dst.is_vgpr:
            raise ValueError(f"Destination must be a VGPR, got {self.dst}")
        # At most one literal constant per VOP3 instruction in RDNA2
        literal_count = sum(1 for src in (self.src0, self.src1, self.src2) if src.kind == OperandKind.LITERAL_CONST)
        if literal_count > 1:
            raise ValueError(f"RDNA2 VOP3 allows at most one literal constant, got {literal_count}")


class TypedExpr:
    """Base class for symbolic typed expressions."""
    def __init__(self, type_kind: TypeKind = TypeKind.RAW32) -> None:
        self.type_kind = type_kind

    def eval(self, ctx: dict[str, int]) -> int:
        raise NotImplementedError


class VariableExpr(TypedExpr):
    """Represents a variable or register reference in an expression tree."""
    def __init__(self, name: str, type_kind: TypeKind = TypeKind.RAW32) -> None:
        super().__init__(type_kind)
        self.name = name

    def eval(self, ctx: dict[str, int]) -> int:
        if self.name not in ctx:
            raise KeyError(f"Variable '{self.name}' not found in evaluation context")
        return ctx[self.name] & MASK32

    def __repr__(self) -> str:
        return f"Var({self.name}:{self.type_kind.value})"


class ConstantExpr(TypedExpr):
    """Represents a constant value in an expression tree."""
    def __init__(self, value: int, type_kind: TypeKind = TypeKind.RAW32) -> None:
        super().__init__(type_kind)
        self.value = value & MASK32

    def eval(self, ctx: dict[str, int]) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"Const(0x{self.value:08X}:{self.type_kind.value})"


class ExtractByteExpr(TypedExpr):
    """Extracts a specific byte (0..3) from a 32-bit expression."""
    def __init__(self, source: TypedExpr, byte_idx: int) -> None:
        super().__init__(TypeKind.BYTE_LANE)
        self.source = source
        self.byte_idx = byte_idx

    def eval(self, ctx: dict[str, int]) -> int:
        val = self.source.eval(ctx)
        return extract_byte(val, self.byte_idx)

    def __repr__(self) -> str:
        return f"ExtractByte({self.source}, {self.byte_idx})"


class PermuteByteExpr(TypedExpr):
    """Permutes a single byte from src0 and src1 using a selector byte expression."""
    def __init__(self, src0: TypedExpr, src1: TypedExpr, sel_byte: TypedExpr) -> None:
        super().__init__(TypeKind.BYTE_LANE)
        self.src0 = src0
        self.src1 = src1
        self.sel_byte = sel_byte

    def eval(self, ctx: dict[str, int]) -> int:
        s0 = self.src0.eval(ctx)
        s1 = self.src1.eval(ctx)
        sel = self.sel_byte.eval(ctx) & BYTE_MASK
        return permute_single_byte(s0, s1, sel)

    def __repr__(self) -> str:
        return f"PermuteByte(src0={self.src0}, src1={self.src1}, sel={self.sel_byte})"


class PermB32Expr(TypedExpr):
    """Symbolic expression tree representing a full v_perm_b32 operation."""
    def __init__(self, src0: TypedExpr, src1: TypedExpr, sel: TypedExpr, type_kind: TypeKind = TypeKind.RAW32) -> None:
        super().__init__(type_kind)
        self.src0 = src0
        self.src1 = src1
        self.sel = sel
        self.byte_exprs = [
            PermuteByteExpr(src0, src1, ExtractByteExpr(sel, i))
            for i in range(4)
        ]

    def eval(self, ctx: dict[str, int]) -> int:
        b0 = self.byte_exprs[0].eval(ctx)
        b1 = self.byte_exprs[1].eval(ctx)
        b2 = self.byte_exprs[2].eval(ctx)
        b3 = self.byte_exprs[3].eval(ctx)
        return pack_bytes(b0, b1, b2, b3)

    def __repr__(self) -> str:
        return f"v_perm_b32({self.src0}, {self.src1}, {self.sel})"


# ==============================================================================
# Unit Tests: Symbolic AST and Typed Expressions
# ==============================================================================

class TestPermB32SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instruction_assembly_formatting(self):
        instr1 = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        self.assertEqual(instr1.to_asm(), "v_perm_b32 v0, v1, v2, v3")

        instr2 = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(5),
            src0=ASTOperand.sgpr(0),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.literal(0x03020100),
        )
        self.assertEqual(instr2.to_asm(), "v_perm_b32 v5, s0, v1, 0x03020100")

    def test_ast_instruction_validation(self):
        # Valid instruction with 1 literal
        instr_valid = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.literal(0x12345678),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.sgpr(2),
        )
        instr_valid.validate()  # should not raise

        # Invalid destination (SGPR not allowed as VALU destination)
        instr_invalid_dst = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            instr_invalid_dst.validate()

        # Invalid: more than 1 literal constant
        instr_multi_literal = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.literal(0x11111111),
            src1=ASTOperand.literal(0x22222222),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            instr_multi_literal.validate()

    def test_typed_expression_evaluation(self):
        v0 = VariableExpr("v0")
        v1 = VariableExpr("v1")
        sel_const = ConstantExpr(0x07060504)  # Identity for src0
        perm_expr = PermB32Expr(v0, v1, sel_const)

        ctx = {"v0": 0xDEADBEEF, "v1": 0xCAFEBABE}
        result = perm_expr.eval(ctx)
        self.assertEqual(result, 0xDEADBEEF)

    def test_symbolic_byte_slicing_and_reassembly(self):
        v0 = VariableExpr("v0")
        b1_expr = ExtractByteExpr(v0, 1)
        ctx = {"v0": 0x12345678}
        self.assertEqual(b1_expr.eval(ctx), 0x56)

        v1 = VariableExpr("v1")
        # Selector picks byte 0 from v1 (which is index 0)
        sel_expr = ConstantExpr(0)
        p_byte = PermuteByteExpr(v0, v1, sel_expr)
        ctx2 = {"v0": 0x12345678, "v1": 0x9ABCDEF0}
        self.assertEqual(p_byte.eval(ctx2), 0xF0)

    def test_expression_type_checking(self):
        c = ConstantExpr(0x12345678, TypeKind.UINT32)
        self.assertEqual(c.type_kind, TypeKind.UINT32)
        v = VariableExpr("x", TypeKind.INT32)
        self.assertEqual(v.type_kind, TypeKind.INT32)


if __name__ == "__main__":
    unittest.main()
