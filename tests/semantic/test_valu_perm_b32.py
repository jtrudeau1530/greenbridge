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


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarRegisterFile:
    """Architectural register file for deterministic scalar simulation."""

    def __init__(self) -> None:
        self.vgprs: dict[int, int] = {i: 0 for i in range(256)}
        self.sgprs: dict[int, int] = {i: 0 for i in range(106)}

    def get_vgpr(self, idx: int) -> int:
        if not 0 <= idx < 256:
            raise IndexError(f"VGPR index out of range [0, 255]: {idx}")
        return self.vgprs.get(idx, 0) & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        if not 0 <= idx < 256:
            raise IndexError(f"VGPR index out of range [0, 255]: {idx}")
        self.vgprs[idx] = val & MASK32

    def get_sgpr(self, idx: int) -> int:
        if not 0 <= idx < 106:
            raise IndexError(f"SGPR index out of range [0, 105]: {idx}")
        return self.sgprs.get(idx, 0) & MASK32

    def set_sgpr(self, idx: int, val: int) -> None:
        if not 0 <= idx < 106:
            raise IndexError(f"SGPR index out of range [0, 105]: {idx}")
        self.sgprs[idx] = val & MASK32

    def read_operand(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.VGPR:
            assert op.reg_index is not None
            return self.get_vgpr(op.reg_index)
        elif op.kind == OperandKind.SGPR:
            assert op.reg_index is not None
            return self.get_sgpr(op.reg_index)
        elif op.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            assert op.constant_value is not None
            return op.constant_value & MASK32
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")

    def write_operand(self, op: ASTOperand, val: int) -> None:
        if op.kind == OperandKind.VGPR:
            assert op.reg_index is not None
            self.set_vgpr(op.reg_index, val)
        else:
            raise ValueError(f"Cannot write VALU result to non-VGPR operand: {op}")


class PermB32ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_perm_b32."""

    def __init__(self, rf: Optional[ScalarRegisterFile] = None) -> None:
        self.rf: ScalarRegisterFile = rf if rf is not None else ScalarRegisterFile()

    def step(self, instr: ASTInstruction) -> None:
        instr.validate()
        if instr.opcode != Opcode.V_PERM_B32:
            raise NotImplementedError(f"Unsupported opcode: {instr.opcode}")

        s0 = self.rf.read_operand(instr.src0)
        s1 = self.rf.read_operand(instr.src1)
        s2 = self.rf.read_operand(instr.src2)

        res = oracle_v_perm_b32(s0, s1, s2)
        self.rf.write_operand(instr.dst, res)

    def run_program(self, instructions: list[ASTInstruction]) -> None:
        for instr in instructions:
            self.step(instr)


# ==============================================================================
# Unit Tests: Deterministic Scalar Interpreter
# ==============================================================================

class TestPermB32ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def test_scalar_execution_all_vgpr(self):
        rf = ScalarRegisterFile()
        rf.set_vgpr(1, 0x11223344)  # src0
        rf.set_vgpr(2, 0x55667788)  # src1
        rf.set_vgpr(3, PermSelectorWord.bswap_src1().raw)  # src2 (selector)

        interp = PermB32ScalarInterpreter(rf)
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        interp.step(instr)

        # src1 bytes: [88, 77, 66, 55], reversed: [55, 66, 77, 88] -> 0x88776655
        self.assertEqual(rf.get_vgpr(0), 0x88776655)

    def test_scalar_execution_mixed_sgpr_vgpr_literal(self):
        rf = ScalarRegisterFile()
        rf.set_sgpr(0, 0xAABBCCDD)  # src0 from SGPR
        rf.set_vgpr(1, 0x11223344)  # src1 from VGPR

        interp = PermB32ScalarInterpreter(rf)
        # Select identity of src0 using literal constant 0x07060504
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(2),
            src0=ASTOperand.sgpr(0),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.literal(0x07060504),
        )
        interp.step(instr)
        self.assertEqual(rf.get_vgpr(2), 0xAABBCCDD)

    def test_scalar_execution_inline_constants(self):
        rf = ScalarRegisterFile()
        rf.set_vgpr(0, 0x12345678)
        interp = PermB32ScalarInterpreter(rf)

        # src0=0 (inline), src1=v0, src2=literal 0x0C0C0100 (zero extend low 16 bits of src1)
        # byte 0 = src1[0] = 0x78, byte 1 = src1[1] = 0x56, byte 2 = 0x00, byte 3 = 0x00
        # Result: 0x00005678
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(1),
            src0=ASTOperand.inline_const(0),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.literal(make_selector(0, 1, SEL_ZERO, SEL_ZERO)),
        )
        interp.step(instr)
        self.assertEqual(rf.get_vgpr(1), 0x00005678)

    def test_sequential_pipeline_execution(self):
        rf = ScalarRegisterFile()
        rf.set_vgpr(0, 0x12345678)
        rf.set_vgpr(1, 0x9ABCDEF0)

        interp = PermB32ScalarInterpreter(rf)
        # Program:
        # 1. v2 = perm(v0, v1, 0x04040404) -> broadcasts byte 4 (src0[0] = 0x78)
        # 2. v3 = perm(v2, v1, 0x03020104) -> replace byte 0 with broadcasted 0x78
        prog = [
            ASTInstruction(
                opcode=Opcode.V_PERM_B32,
                dst=ASTOperand.vgpr(2),
                src0=ASTOperand.vgpr(0),
                src1=ASTOperand.vgpr(1),
                src2=ASTOperand.literal(0x04040404),
            ),
            ASTInstruction(
                opcode=Opcode.V_PERM_B32,
                dst=ASTOperand.vgpr(3),
                src0=ASTOperand.vgpr(2),
                src1=ASTOperand.vgpr(1),
                src2=ASTOperand.literal(make_selector(4, 1, 2, 3)),
            ),
        ]
        interp.run_program(prog)

        self.assertEqual(rf.get_vgpr(2), 0x78787878)
        # v3 byte 0 is v2[0] = 0x78; bytes 1..3 are v1[1..3] = 0x9ABCDE
        self.assertEqual(rf.get_vgpr(3), 0x9ABCDE78)

    def test_state_preservation_unmodified_registers(self):
        rf = ScalarRegisterFile()
        rf.set_vgpr(10, 0xCAFEBABE)
        rf.set_sgpr(5, 0xFEEDFACE)

        interp = PermB32ScalarInterpreter(rf)
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.inline_const(0),
            src1=ASTOperand.inline_const(0),
            src2=ASTOperand.inline_const(0),
        )
        interp.step(instr)

        self.assertEqual(rf.get_vgpr(10), 0xCAFEBABE)
        self.assertEqual(rf.get_sgpr(5), 0xFEEDFACE)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

class SIMDWaveState:
    """Architectural model of a SIMD Wavefront (Wave32 or Wave64)."""

    def __init__(self, num_lanes: int = 32, exec_mask: Optional[int] = None) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {num_lanes}")
        self.num_lanes = num_lanes
        default_mask = (1 << num_lanes) - 1
        self.exec_mask: int = (exec_mask if exec_mask is not None else default_mask) & default_mask
        self.vgprs: dict[int, list[int]] = {i: [0] * num_lanes for i in range(256)}
        self.sgprs: dict[int, int] = {i: 0 for i in range(106)}

    def get_lane_vgpr(self, reg_idx: int, lane: int) -> int:
        if not 0 <= lane < self.num_lanes:
            raise IndexError(f"Lane {lane} out of range [0, {self.num_lanes})")
        return self.vgprs[reg_idx][lane] & MASK32

    def set_lane_vgpr(self, reg_idx: int, lane: int, val: int) -> None:
        if not 0 <= lane < self.num_lanes:
            raise IndexError(f"Lane {lane} out of range [0, {self.num_lanes})")
        self.vgprs[reg_idx][lane] = val & MASK32

    def set_wave_vgpr(self, reg_idx: int, vals: list[int]) -> None:
        if len(vals) != self.num_lanes:
            raise ValueError(f"Expected {self.num_lanes} values, got {len(vals)}")
        self.vgprs[reg_idx] = [v & MASK32 for v in vals]

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs.get(idx, 0) & MASK32

    def read_lane_operand(self, op: ASTOperand, lane: int) -> int:
        if op.kind == OperandKind.VGPR:
            assert op.reg_index is not None
            return self.get_lane_vgpr(op.reg_index, lane)
        elif op.kind == OperandKind.SGPR:
            assert op.reg_index is not None
            return self.get_sgpr(op.reg_index)
        elif op.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            assert op.constant_value is not None
            return op.constant_value & MASK32
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def execute_instruction(self, instr: ASTInstruction) -> None:
        instr.validate()
        if instr.opcode != Opcode.V_PERM_B32:
            raise NotImplementedError(f"Unsupported opcode: {instr.opcode}")

        dst_idx = instr.dst.reg_index
        assert dst_idx is not None

        for lane in range(self.num_lanes):
            if self.is_lane_active(lane):
                s0 = self.read_lane_operand(instr.src0, lane)
                s1 = self.read_lane_operand(instr.src1, lane)
                s2 = self.read_lane_operand(instr.src2, lane)
                res = oracle_v_perm_b32(s0, s1, s2)
                self.set_lane_vgpr(dst_idx, lane, res)
            # Inactive lanes strictly preserve prior contents


# ==============================================================================
# Unit Tests: SIMD Multi-Lane Divergence and EXEC Masking
# ==============================================================================

class TestPermB32SIMDMultiLane(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_selector_divergent_data(self):
        wave = SIMDWaveState(num_lanes=32)
        # Each lane has distinct src0 and src1 data
        src0_vals = [0x10000000 + i for i in range(32)]
        src1_vals = [0x20000000 + i for i in range(32)]
        wave.set_wave_vgpr(1, src0_vals)
        wave.set_wave_vgpr(2, src1_vals)

        # Uniform selector: bswap src1
        sel = PermSelectorWord.bswap_src1().raw
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            expected = oracle_v_perm_b32(src0_vals[lane], src1_vals[lane], sel)
            self.assertEqual(wave.get_lane_vgpr(0, lane), expected)

    def test_wave32_divergent_selectors(self):
        wave = SIMDWaveState(num_lanes=32)
        src0_data = [0xAABBCCDD] * 32
        src1_data = [0x11223344] * 32
        # Divergent selectors: lane k broadcasts byte (k % 8)
        sel_vals = [PermSelectorWord.broadcast_byte(k % 8).raw for k in range(32)]
        wave.set_wave_vgpr(1, src0_data)
        wave.set_wave_vgpr(2, src1_data)
        wave.set_wave_vgpr(3, sel_vals)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            expected = oracle_v_perm_b32(src0_data[lane], src1_data[lane], sel_vals[lane])
            self.assertEqual(wave.get_lane_vgpr(0, lane), expected)

    def test_wave64_execution_all_lanes(self):
        wave = SIMDWaveState(num_lanes=64)
        src0_vals = [0x55000000 + i for i in range(64)]
        src1_vals = [0xAA000000 + i for i in range(64)]
        wave.set_wave_vgpr(0, [0xCAFE0000] * 64)  # initial dst
        wave.set_wave_vgpr(1, src0_vals)
        wave.set_wave_vgpr(2, src1_vals)

        sel = PermSelectorWord.identity_src0().raw
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(64):
            self.assertEqual(wave.get_lane_vgpr(0, lane), src0_vals[lane])

    def test_exec_masking_partial_lanes(self):
        # Alternating even lanes enabled in Wave32: 0x55555555
        exec_mask = 0x55555555
        wave = SIMDWaveState(num_lanes=32, exec_mask=exec_mask)
        prior_dst = [0xEEEE0000 + i for i in range(32)]
        wave.set_wave_vgpr(0, prior_dst)
        wave.set_wave_vgpr(1, [0x11111111] * 32)
        wave.set_wave_vgpr(2, [0x22222222] * 32)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(PermSelectorWord.identity_src1().raw),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.get_lane_vgpr(0, lane), 0x22222222)
            else:
                self.assertEqual(wave.get_lane_vgpr(0, lane), prior_dst[lane])

    def test_exec_masking_contiguous_blocks(self):
        # Lower 16 lanes active, upper 16 lanes inactive: 0x0000FFFF
        exec_mask = 0x0000FFFF
        wave = SIMDWaveState(num_lanes=32, exec_mask=exec_mask)
        prior_dst = [0xDEADBEEF] * 32
        wave.set_wave_vgpr(0, prior_dst)
        wave.set_wave_vgpr(1, [0x44332211] * 32)
        wave.set_wave_vgpr(2, [0x88776655] * 32)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(PermSelectorWord.identity_src0().raw),
        )
        wave.execute_instruction(instr)

        for lane in range(16):
            self.assertEqual(wave.get_lane_vgpr(0, lane), 0x44332211)
        for lane in range(16, 32):
            self.assertEqual(wave.get_lane_vgpr(0, lane), 0xDEADBEEF)

    def test_all_lanes_disabled_preserves_dst(self):
        wave = SIMDWaveState(num_lanes=32, exec_mask=0)
        prior_dst = [0xFEEDC0DE + i for i in range(32)]
        wave.set_wave_vgpr(0, prior_dst)
        wave.set_wave_vgpr(1, [0x00000000] * 32)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.literal(0),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            self.assertEqual(wave.get_lane_vgpr(0, lane), prior_dst[lane])


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestPermB32RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_dst_aliases_src0(self):
        # v0 = perm(v0, v1, v2)
        # v0 holds src0 (indices 4..7). In-place update must read old v0 bytes accurately.
        rf = ScalarRegisterFile()
        rf.set_vgpr(0, 0x11223344)  # src0 and dst
        rf.set_vgpr(1, 0xAABBCCDD)  # src1
        # Selector picks byte 0 of src1, byte 0 of src0, byte 1 of src1, byte 1 of src0
        # src1[0]=0xDD, src0[0]=0x44, src1[1]=0xCC, src0[1]=0x33 -> 0x33CC44DD
        sel = make_selector(0, 4, 1, 5)
        rf.set_vgpr(2, sel)

        interp = PermB32ScalarInterpreter(rf)
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        interp.step(instr)
        self.assertEqual(rf.get_vgpr(0), 0x33CC44DD)

    def test_dst_aliases_src1(self):
        # v1 = perm(v0, v1, v2)
        # v1 holds src1 (indices 0..3)
        rf = ScalarRegisterFile()
        rf.set_vgpr(0, 0x11223344)  # src0
        rf.set_vgpr(1, 0xAABBCCDD)  # src1 and dst
        # Reverse src1: [DD, CC, BB, AA] -> [AA, BB, CC, DD] -> 0xDDCCBBAA
        sel = PermSelectorWord.bswap_src1().raw
        rf.set_vgpr(2, sel)

        interp = PermB32ScalarInterpreter(rf)
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(1),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        interp.step(instr)
        self.assertEqual(rf.get_vgpr(1), 0xDDCCBBAA)

    def test_dst_aliases_src2_selector(self):
        # v2 = perm(v0, v1, v2)
        # Selector register is overwritten by the result
        rf = ScalarRegisterFile()
        rf.set_vgpr(0, 0x12345678)
        rf.set_vgpr(1, 0x9ABCDEF0)
        # Identity of src0: 0x07060504
        sel = PermSelectorWord.identity_src0().raw
        rf.set_vgpr(2, sel)

        interp = PermB32ScalarInterpreter(rf)
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(2),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        interp.step(instr)
        self.assertEqual(rf.get_vgpr(2), 0x12345678)

    def test_full_aliasing_all_operands(self):
        # v0 = perm(v0, v0, v0)
        # All operands are the same register v0
        rf = ScalarRegisterFile()
        # Choose a value that serves as both data and selector:
        # byte 0 = 0x01, byte 1 = 0x02, byte 2 = 0x03, byte 3 = 0x00 -> 0x00030201
        val = 0x00030201
        rf.set_vgpr(0, val)

        # Expected:
        # data pool: src0=val (bytes 4..7), src1=val (bytes 0..3)
        # selector bytes: sel0=1, sel1=2, sel2=3, sel3=0
        # byte 0: val[1] = 0x02
        # byte 1: val[2] = 0x03
        # byte 2: val[3] = 0x00
        # byte 3: val[0] = 0x01
        # Pack: [0x02, 0x03, 0x00, 0x01] -> 0x01000302
        interp = PermB32ScalarInterpreter(rf)
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.vgpr(0),
        )
        interp.step(instr)
        self.assertEqual(rf.get_vgpr(0), 0x01000302)

    def test_src0_aliases_src1_self_swizzle(self):
        # v2 = perm(v0, v0, literal)
        rf = ScalarRegisterFile()
        rf.set_vgpr(0, 0x12345678)

        # Select bytes from same register using indices 0..3 or 4..7
        # Index 0 and 4 are both byte 0 (0x78)
        sel = make_selector(0, 4, 1, 5)  # [0x78, 0x78, 0x56, 0x56] -> 0x56567878
        interp = PermB32ScalarInterpreter(rf)
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(2),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.literal(sel),
        )
        interp.step(instr)
        self.assertEqual(rf.get_vgpr(2), 0x56567878)

    def test_simd_wave_aliasing_with_exec_mask(self):
        # Wave32: v1 = perm(v0, v1, v2) with half EXEC mask
        wave = SIMDWaveState(num_lanes=32, exec_mask=0x0000FFFF)
        src0_init = [0x11111111] * 32
        src1_init = [0x22222222 + i for i in range(32)]
        sel_bswap = [PermSelectorWord.bswap_src1().raw] * 32

        wave.set_wave_vgpr(0, src0_init)
        wave.set_wave_vgpr(1, src1_init)
        wave.set_wave_vgpr(2, sel_bswap)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(1),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        wave.execute_instruction(instr)

        for lane in range(16):
            expected = oracle_v_perm_b32(src0_init[lane], src1_init[lane], sel_bswap[lane])
            self.assertEqual(wave.get_lane_vgpr(1, lane), expected)
        for lane in range(16, 32):
            self.assertEqual(wave.get_lane_vgpr(1, lane), src1_init[lane])

    def test_unwritten_registers_unaffected(self):
        rf = ScalarRegisterFile()
        rf.set_vgpr(3, 0x11111111)
        rf.set_vgpr(4, 0x22222222)
        rf.set_vgpr(5, 0x33333333)
        rf.set_vgpr(6, 0x44444444)

        interp = PermB32ScalarInterpreter(rf)
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(4),
            src0=ASTOperand.vgpr(3),
            src1=ASTOperand.vgpr(3),
            src2=ASTOperand.literal(0),
        )
        interp.step(instr)

        self.assertEqual(rf.get_vgpr(3), 0x11111111)
        self.assertEqual(rf.get_vgpr(5), 0x33333333)
        self.assertEqual(rf.get_vgpr(6), 0x44444444)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass
class StatusFlags:
    """Architectural status flags and condition codes for 32-bit execution."""
    zero: bool = False
    negative: bool = False
    parity_even: bool = False
    byte_zeros: Tuple[bool, bool, bool, bool] = (False, False, False, False)
    byte_negatives: Tuple[bool, bool, bool, bool] = (False, False, False, False)
    carry: bool = False
    overflow: bool = False

    @classmethod
    def from_result(cls, res: int, addend: int = 0, is_sub: bool = False) -> StatusFlags:
        val = res & MASK32
        zero = (val == 0)
        negative = bool(val & 0x8000_0000)

        # Bit parity of low byte
        popcount = bin(val & 0xFF).count("1")
        parity_even = (popcount % 2 == 0)

        b_zeros = (
            extract_byte(val, 0) == 0,
            extract_byte(val, 1) == 0,
            extract_byte(val, 2) == 0,
            extract_byte(val, 3) == 0,
        )
        b_negs = (
            bool(extract_byte(val, 0) & 0x80),
            bool(extract_byte(val, 1) & 0x80),
            bool(extract_byte(val, 2) & 0x80),
            bool(extract_byte(val, 3) & 0x80),
        )

        carry = False
        overflow = False
        if is_sub:
            # Emulate subtraction borrow: addend - val
            diff = addend - val
            carry = (diff < 0)  # Borrow
            sa = (addend >> 31) & 1
            sb = (val >> 31) & 1
            sr = ((diff & MASK32) >> 31) & 1
            overflow = bool((sa ^ sb) & (sa ^ sr))
        elif addend != 0:
            # Emulate addition carry: addend + val
            total = addend + val
            carry = bool(total > MASK32)
            sa = (addend >> 31) & 1
            sb = (val >> 31) & 1
            sr = ((total & MASK32) >> 31) & 1
            overflow = bool(not (sa ^ sb) & (sa ^ sr))

        return cls(
            zero=zero,
            negative=negative,
            parity_even=parity_even,
            byte_zeros=b_zeros,
            byte_negatives=b_negs,
            carry=carry,
            overflow=overflow,
        )


def compute_wave_vcc_mask(wave: SIMDWaveState, reg_idx: int, predicate: str = "ne_zero") -> int:
    """Evaluate lane condition codes across active lanes to produce a 32/64-bit VCC mask."""
    vcc_mask = 0
    for lane in range(wave.num_lanes):
        if wave.is_lane_active(lane):
            val = wave.get_lane_vgpr(reg_idx, lane)
            cond = False
            if predicate == "ne_zero":
                cond = (val != 0)
            elif predicate == "eq_zero":
                cond = (val == 0)
            elif predicate == "negative":
                cond = bool(val & 0x8000_0000)
            elif predicate == "byte0_zero":
                cond = (extract_byte(val, 0) == 0)
            elif predicate == "byte3_neg":
                cond = bool(extract_byte(val, 3) & 0x80)
            else:
                raise ValueError(f"Unknown predicate: {predicate}")

            if cond:
                vcc_mask |= (1 << lane)
    return vcc_mask


# ==============================================================================
# Unit Tests: Condition Codes and Status Flags
# ==============================================================================

class TestPermB32StatusFlagsAndConditionCodes(unittest.TestCase):
    """Test suite for condition codes, status flags, and carry/borrow generation."""

    def test_zero_flag_detection(self):
        # Permutation producing all zero bytes
        res = oracle_v_perm_b32(0xFFFFFFFF, 0xFFFFFFFF, make_selector(SEL_ZERO, SEL_ZERO, SEL_ZERO, SEL_ZERO))
        flags = StatusFlags.from_result(res)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.negative)
        self.assertEqual(flags.byte_zeros, (True, True, True, True))

    def test_negative_sign_flag_detection(self):
        # Permutation selecting sign-extended 1s into MSB (byte 3)
        res = oracle_v_perm_b32(0x80000000, 0, make_selector(0, 0, 0, SEL_SIGN_EXT_SRC0_B3))
        flags = StatusFlags.from_result(res)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.negative)
        self.assertTrue(flags.byte_negatives[3])
        self.assertFalse(flags.byte_negatives[0])

    def test_byte_level_flags(self):
        # Result = 0x8000007F: byte 0=0x7F, byte 1=0x00, byte 2=0x00, byte 3=0x80
        res = 0x8000007F
        flags = StatusFlags.from_result(res)
        self.assertEqual(flags.byte_zeros, (False, True, True, False))
        self.assertEqual(flags.byte_negatives, (False, False, False, True))

    def test_parity_flag_calculation(self):
        # Byte 0 = 0x03 (binary 00000011 -> 2 bits set -> even parity)
        flags_even = StatusFlags.from_result(0x00000003)
        self.assertTrue(flags_even.parity_even)

        # Byte 0 = 0x07 (binary 00000111 -> 3 bits set -> odd parity)
        flags_odd = StatusFlags.from_result(0x00000007)
        self.assertFalse(flags_odd.parity_even)

    def test_wave_vcc_mask_generation(self):
        # Wave32 where even lanes have non-zero results and odd lanes have zero results
        wave = SIMDWaveState(num_lanes=32, exec_mask=0xFFFFFFFF)
        vals = [0x12345678 if i % 2 == 0 else 0 for i in range(32)]
        wave.set_wave_vgpr(0, vals)

        vcc_ne = compute_wave_vcc_mask(wave, 0, predicate="ne_zero")
        # Even bits set: 0x55555555
        self.assertEqual(vcc_ne, 0x55555555)

        vcc_eq = compute_wave_vcc_mask(wave, 0, predicate="eq_zero")
        # Odd bits set: 0xAAAAAAAA
        self.assertEqual(vcc_eq, 0xAAAAAAAA)

    def test_wave_vcc_mask_respects_inactive_lanes(self):
        # EXEC mask has only lower 16 lanes active
        wave = SIMDWaveState(num_lanes=32, exec_mask=0x0000FFFF)
        wave.set_wave_vgpr(0, [0xFFFFFFFF] * 32)

        vcc = compute_wave_vcc_mask(wave, 0, predicate="ne_zero")
        # Upper 16 lanes inactive, so VCC bits must not be set
        self.assertEqual(vcc, 0x0000FFFF)

    def test_carry_borrow_generation_from_permuted_operands(self):
        # Unpack byte 0 using v_perm_b32 into a register, then add to trigger carry
        val = oracle_v_perm_b32(0, 0x000000FF, make_selector(0, SEL_ZERO, SEL_ZERO, SEL_ZERO))
        self.assertEqual(val, 0x000000FF)

        # Add 0xFFFFFFFF + 0x000000FF -> generates unsigned carry
        flags_add = StatusFlags.from_result(val, addend=0xFFFFFFFF)
        self.assertTrue(flags_add.carry)

        # Subtract 0x00000010 - 0x000000FF -> generates borrow
        flags_sub = StatusFlags.from_result(val, addend=0x00000010, is_sub=True)
        self.assertTrue(flags_sub.carry)


# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

# IEEE 754 Binary32 (single precision) architectural constants
F32_POS_ZERO: int = 0x0000_0000
F32_NEG_ZERO: int = 0x8000_0000
F32_POS_INF: int = 0x7F80_0000
F32_NEG_INF: int = 0xFF80_0000
F32_QNAN: int = 0x7FC0_0000
F32_SNAN: int = 0x7F80_0001
F32_NEG_QNAN: int = 0xFFC0_0000

F32_POS_MIN_NORMAL: int = 0x0080_0000
F32_POS_MAX_NORMAL: int = 0x7F7F_FFFF
F32_POS_MIN_SUBNORM: int = 0x0000_0001
F32_POS_MAX_SUBNORM: int = 0x007F_FFFF

# IEEE 754 Binary16 (half precision) constants
F16_POS_ZERO: int = 0x0000
F16_NEG_ZERO: int = 0x8000
F16_POS_INF: int = 0x7C00
F16_NEG_INF: int = 0xFC00
F16_QNAN: int = 0x7E00
F16_SNAN: int = 0x7D00


def bswap_f32_bits(f_bits: int) -> int:
    """Byte-swap an IEEE 754 float32 using v_perm_b32."""
    sel = PermSelectorWord.bswap_src1().raw
    return oracle_v_perm_b32(0, f_bits, sel)


def unpack_pk_f16_lo(packed: int) -> int:
    """Unpack lower 16-bit float into 32-bit register with zero-extension using v_perm_b32."""
    sel = make_selector(0, 1, SEL_ZERO, SEL_ZERO)
    return oracle_v_perm_b32(0, packed, sel)


def unpack_pk_f16_hi(packed: int) -> int:
    """Unpack upper 16-bit float into 32-bit register with zero-extension using v_perm_b32."""
    sel = make_selector(2, 3, SEL_ZERO, SEL_ZERO)
    return oracle_v_perm_b32(0, packed, sel)


class TestPermB32IEEE754Boundaries(unittest.TestCase):
    """Test suite for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_f32_endian_swapping_preserves_nan_payloads(self):
        # Signaling NaN with custom payload
        snan = 0x7F80_1234
        swapped = bswap_f32_bits(snan)
        restored = bswap_f32_bits(swapped)
        self.assertEqual(restored, snan)

        # Quiet NaN
        qnan = F32_QNAN
        self.assertEqual(bswap_f32_bits(bswap_f32_bits(qnan)), qnan)

        # Negative Quiet NaN
        neg_qnan = F32_NEG_QNAN
        self.assertEqual(bswap_f32_bits(bswap_f32_bits(neg_qnan)), neg_qnan)

    def test_f32_subnormal_byte_manipulation(self):
        # Smallest positive subnormal: 0x00000001 (byte 0 has bit 0 set)
        min_sub = F32_POS_MIN_SUBNORM
        swapped = bswap_f32_bits(min_sub)
        # In big endian, bit 0 is in byte 3 (MSB position of word): 0x01000000
        self.assertEqual(swapped, 0x0100_0000)
        self.assertEqual(bswap_f32_bits(swapped), min_sub)

        # Largest subnormal: 0x007FFFFF
        max_sub = F32_POS_MAX_SUBNORM
        self.assertEqual(bswap_f32_bits(bswap_f32_bits(max_sub)), max_sub)

    def test_f32_infinity_and_zero_byte_isolation(self):
        # Positive and negative zero
        self.assertEqual(bswap_f32_bits(bswap_f32_bits(F32_POS_ZERO)), F32_POS_ZERO)
        self.assertEqual(bswap_f32_bits(bswap_f32_bits(F32_NEG_ZERO)), F32_NEG_ZERO)

        # Positive and negative infinity
        self.assertEqual(bswap_f32_bits(bswap_f32_bits(F32_POS_INF)), F32_POS_INF)
        self.assertEqual(bswap_f32_bits(bswap_f32_bits(F32_NEG_INF)), F32_NEG_INF)

        # Extract exponent byte (bits 30..23 span byte 2 and byte 3)
        # Using v_perm_b32 to extract byte 2 and byte 3 into lower 16 bits
        sel_exp = make_selector(2, 3, SEL_ZERO, SEL_ZERO)
        exp_words = oracle_v_perm_b32(0, F32_POS_INF, sel_exp)
        # F32_POS_INF = 0x7F800000 -> byte 2=0x80, byte 3=0x7F -> 0x00007F80
        self.assertEqual(exp_words, 0x00007F80)

    def test_f16_unpacking_and_zero_extension(self):
        # Packed pair of f16 values: lo = +1.0 (0x3C00), hi = -1.0 (0xBC00)
        # packed dword: 0xBC00_3C00
        packed = 0xBC00_3C00
        unpacked_lo = unpack_pk_f16_lo(packed)
        self.assertEqual(unpacked_lo, 0x0000_3C00)

        unpacked_hi = unpack_pk_f16_hi(packed)
        self.assertEqual(unpacked_hi, 0x0000_BC00)

    def test_f32_sign_bit_sign_extension(self):
        # Sign-extend sign bit of float: bit 31 is in byte 3
        # Selector 8 performs sign extension of S1 bit 31
        sel_sign = make_selector(8, 8, 8, 8)

        # Positive float: sign bit 0 -> 0x00000000
        pos_float = 0x3F80_0000  # +1.0f
        mask_pos = oracle_v_perm_b32(0, pos_float, sel_sign)
        self.assertEqual(mask_pos, 0x0000_0000)

        # Negative float: sign bit 1 -> 0xFFFFFFFF
        neg_float = 0xBF80_0000  # -1.0f
        mask_neg = oracle_v_perm_b32(0, neg_float, sel_sign)
        self.assertEqual(mask_neg, 0xFFFF_FFFF)

        # Negative Quiet NaN
        mask_nan = oracle_v_perm_b32(0, F32_NEG_QNAN, sel_sign)
        self.assertEqual(mask_nan, 0xFFFF_FFFF)


# ==============================================================================
# Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

def mutated_perm_off_by_one(src0: int, src1: int, selector: int) -> int:
    """Mutant 1: shifts selector index by +1."""
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    sel = selector & MASK32
    b0 = permute_single_byte(s0, s1, (extract_byte(sel, 0) + 1) & 0xFF)
    b1 = permute_single_byte(s0, s1, (extract_byte(sel, 1) + 1) & 0xFF)
    b2 = permute_single_byte(s0, s1, (extract_byte(sel, 2) + 1) & 0xFF)
    b3 = permute_single_byte(s0, s1, (extract_byte(sel, 3) + 1) & 0xFF)
    return pack_bytes(b0, b1, b2, b3)


def mutated_perm_inverted_sources(src0: int, src1: int, selector: int) -> int:
    """Mutant 2: inverts src0 and src1 mapping."""
    return oracle_v_perm_b32(src1, src0, selector)


def mutated_perm_ignore_zero_fill(src0: int, src1: int, selector: int) -> int:
    """Mutant 3: treats selector 12 as picking from src1 byte 0 instead of 0x00."""
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    sel = selector & MASK32
    bytes_out = []
    for i in range(4):
        b = extract_byte(sel, i)
        if b == SEL_ZERO:
            bytes_out.append(extract_byte(s1, 0))  # Buggy replacement
        else:
            bytes_out.append(permute_single_byte(s0, s1, b))
    return pack_bytes(bytes_out[0], bytes_out[1], bytes_out[2], bytes_out[3])


def mutated_perm_ignore_ones_fill(src0: int, src1: int, selector: int) -> int:
    """Mutant 4: returns 0x00 instead of 0xFF for selector >= 13."""
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    sel = selector & MASK32
    bytes_out = []
    for i in range(4):
        b = extract_byte(sel, i)
        if b >= SEL_ONES_MIN:
            bytes_out.append(0x00)  # Buggy replacement
        else:
            bytes_out.append(permute_single_byte(s0, s1, b))
    return pack_bytes(bytes_out[0], bytes_out[1], bytes_out[2], bytes_out[3])


def mutated_perm_wrong_endianness(src0: int, src1: int, selector: int) -> int:
    """Mutant 5: packs output bytes in big-endian order instead of little-endian."""
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    sel = selector & MASK32
    b0 = permute_single_byte(s0, s1, extract_byte(sel, 0))
    b1 = permute_single_byte(s0, s1, extract_byte(sel, 1))
    b2 = permute_single_byte(s0, s1, extract_byte(sel, 2))
    b3 = permute_single_byte(s0, s1, extract_byte(sel, 3))
    # Big-endian packing
    return ((b0 << 24) | (b1 << 16) | (b2 << 8) | b3) & MASK32


class TestPermB32NegativeControlsAndMutations(unittest.TestCase):
    """Test suite for negative controls, syntax rejection, and mutation detection."""

    def test_negative_sgpr_destination_rejected(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            instr.validate()

    def test_negative_multiple_literals_rejected(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.literal(0x11111111),
            src1=ASTOperand.literal(0x22222222),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            instr.validate()

    def test_negative_register_index_bounds(self):
        rf = ScalarRegisterFile()
        with self.assertRaises(IndexError):
            rf.get_vgpr(256)
        with self.assertRaises(IndexError):
            rf.set_vgpr(-1, 0)
        with self.assertRaises(IndexError):
            rf.get_sgpr(106)
        with self.assertRaises(IndexError):
            rf.set_sgpr(-1, 0)

    def test_negative_byte_index_bounds(self):
        with self.assertRaises(ValueError):
            extract_byte(0x12345678, 4)
        with self.assertRaises(ValueError):
            extract_byte(0x12345678, -1)

    def test_negative_wave_size_rejected(self):
        with self.assertRaises(ValueError):
            SIMDWaveState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDWaveState(num_lanes=128)

    def test_mutation_off_by_one_detected(self):
        src0 = 0xAABBCCDD
        src1 = 0x11223344
        sel = PermSelectorWord.identity_src1().raw
        expected = oracle_v_perm_b32(src0, src1, sel)
        mutated = mutated_perm_off_by_one(src0, src1, sel)
        self.assertNotEqual(expected, mutated)

    def test_mutation_inverted_sources_detected(self):
        src0 = 0xAABBCCDD
        src1 = 0x11223344
        sel = PermSelectorWord.identity_src1().raw
        expected = oracle_v_perm_b32(src0, src1, sel)
        mutated = mutated_perm_inverted_sources(src0, src1, sel)
        self.assertNotEqual(expected, mutated)

    def test_mutation_constant_zero_ignored_detected(self):
        src0 = 0xFFFFFFFF
        src1 = 0x55667788
        sel = make_selector(SEL_ZERO, SEL_ZERO, SEL_ZERO, SEL_ZERO)
        expected = oracle_v_perm_b32(src0, src1, sel)
        mutated = mutated_perm_ignore_zero_fill(src0, src1, sel)
        self.assertNotEqual(expected, mutated)

    def test_mutation_constant_ones_ignored_detected(self):
        src0 = 0x00000000
        src1 = 0x00000000
        sel = make_selector(SEL_ONES_MIN, SEL_ONES_MIN, SEL_ONES_MIN, SEL_ONES_MIN)
        expected = oracle_v_perm_b32(src0, src1, sel)
        mutated = mutated_perm_ignore_ones_fill(src0, src1, sel)
        self.assertNotEqual(expected, mutated)

    def test_mutation_exec_mask_ignored_detected(self):
        # A buggy execution engine that ignores EXEC mask writes all lanes
        wave = SIMDWaveState(num_lanes=32, exec_mask=0x00000001)  # Only lane 0 active
        wave.set_wave_vgpr(0, [0xCAFE0000] * 32)
        wave.set_wave_vgpr(1, [0x12345678] * 32)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.literal(0),
        )
        wave.execute_instruction(instr)

        # Inactive lanes (1..31) must remain 0xCAFE0000
        for lane in range(1, 32):
            self.assertEqual(wave.get_lane_vgpr(0, lane), 0xCAFE0000)

    def test_mutation_wrong_endianness_detected(self):
        src0 = 0x12345678
        src1 = 0x9ABCDEF0
        sel = make_selector(0, 1, 2, 3)
        expected = oracle_v_perm_b32(src0, src1, sel)
        mutated = mutated_perm_wrong_endianness(src0, src1, sel)
        self.assertNotEqual(expected, mutated)


if __name__ == "__main__":
    unittest.main()
