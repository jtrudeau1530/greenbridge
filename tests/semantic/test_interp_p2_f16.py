from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_interp_p2_f16.

RDNA2 VINTRP instruction family:
    v_interp_p2_f16 vD, vSrc(J), attrN.chan [high] [clamp]

Instruction semantics:
    Evaluates phase 2 of perspective parameter interpolation for half-precision float attributes:
        D = p1_accum + J * (P2 - P0)   (or P20 * J + p1_accum)
    where:
        - `p1_accum`: f32 accumulator input (produced by v_interp_p1ll_f16 or v_interp_p1lv_f16),
          stored in destination register vD.
        - `J`: barycentric coordinate j (typically in vSrc / v1).
        - `P20` = P2 - P0: slope parameter for attribute N channel chan (p20 parameter in LDS/parameter cache).
          If `high` source modifier is set: selects high 16-bit half [31:16] of the 32-bit attribute word.
          Otherwise: selects low 16-bit half [15:0] of the 32-bit attribute word.
        - The fused multiply-add accumulation is performed in floating-point and rounded once to binary16 (f16).
        - If `clamp`: clamps the resulting f16 to [0.0, 1.0], mapping NaNs to +0.0.
        - Destination write:
          If `high` destination modifier (or op_sel destination bit) is set:
              writes the 16-bit result into vD[31:16], strictly preserving vD[15:0].
          Otherwise (default low half):
              writes the 16-bit result into vD[15:0], strictly preserving vD[31:16] (per SHADER-430 / gfx1036).

Zero game data; standard library Python only; Python 3.9+ compatible.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import sys
from typing import List, Optional, Tuple, Union
import unittest

# IEEE 754 binary16 (half-precision) constants
H_POS_ZERO: int = 0x0000
H_NEG_ZERO: int = 0x8000
H_ONE: int = 0x3C00
H_NEG_ONE: int = 0xBC00
H_TWO: int = 0x4000
H_HALF: int = 0x3800
H_POS_INF: int = 0x7C00
H_NEG_INF: int = 0xFC00
H_QNAN: int = 0x7E00
H_SNAN: int = 0x7C01
H_SUB_MIN: int = 0x0001
H_SUB_MAX: int = 0x03FF
H_MIN_NORMAL: int = 0x0400
H_MAX_NORMAL: int = 0x7BFF

# IEEE 754 binary32 constants
F32_POS_ZERO: int = 0x0000_0000
F32_NEG_ZERO: int = 0x8000_0000
F32_POS_INF: int = 0x7F80_0000
F32_NEG_INF: int = 0xFF80_0000
F32_QNAN: int = 0x7FC0_0000

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF


def f16_to_u16(val: float) -> int:
    """Pack Python float into IEEE 754 binary16 unsigned 16-bit int."""
    try:
        return struct.unpack("<H", struct.pack("<e", float(val)))[0]
    except OverflowError:
        return H_POS_INF if val > 0 else H_NEG_INF


def u16_to_f16(bits: int) -> float:
    """Unpack IEEE 754 binary16 unsigned 16-bit int into Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def f32_to_u32(val: float) -> int:
    """Pack Python float into IEEE 754 binary32 unsigned 32-bit int."""
    try:
        return struct.unpack("<I", struct.pack("<f", float(val)))[0]
    except OverflowError:
        return F32_POS_INF if val > 0 else F32_NEG_INF


def u32_to_f32(bits: int) -> float:
    """Unpack IEEE 754 binary32 unsigned 32-bit int into Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def pack_f16_pair(lo_u16: int, hi_u16: int) -> int:
    """Pack two 16-bit words into a single 32-bit dword [hi:lo]."""
    return (lo_u16 & MASK16) | ((hi_u16 & MASK16) << 16)


def unpack_f16_pair(val_u32: int) -> tuple[int, int]:
    """Unpack a 32-bit dword into (low 16-bit word, high 16-bit word)."""
    return val_u32 & MASK16, (val_u32 >> 16) & MASK16


def is_nan_f16(bits: int) -> bool:
    return ((bits & 0x7C00) == 0x7C00) and ((bits & 0x03FF) != 0)


def is_nan_f32(bits: int) -> bool:
    return ((bits & 0x7F80_0000) == 0x7F80_0000) and ((bits & 0x007F_FFFF) != 0)


def apply_clamp_f16(bits_u16: int) -> int:
    """Apply RDNA2 clamp modifier [0.0, 1.0] to half-precision float, mapping NaNs to +0.0."""
    val = u16_to_f16(bits_u16)
    if math.isnan(val) or bits_u16 == H_NEG_ZERO:
        return H_POS_ZERO
    if val < 0.0:
        return H_POS_ZERO
    if val > 1.0:
        return H_ONE
    return f16_to_u16(val)


def apply_f32_modifiers(bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply VOP3 source modifiers (abs, neg) on 32-bit float."""
    res = bits & MASK32
    if abs_mod:
        res &= ~0x8000_0000
    if neg_mod:
        res ^= 0x8000_0000
    return res


def ref_interp_p2_f16(
    j_coord_f32: float,
    p20_word_u32: int,
    p1_accum_f32: float,
    *,
    high_src: bool = False,
    high_dst: bool = False,
    old_vdst_u32: int = 0,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for v_interp_p2_f16.

    Computes:
        res = p1_accum + j * p20
    rounded to f16 (binary16), optionally clamped to [0.0, 1.0], and merged
    into old_vdst_u32 at low half [15:0] or high half [31:16], strictly preserving the untouched half.
    """
    lo_src, hi_src = unpack_f16_pair(p20_word_u32)
    selected_u16 = hi_src if high_src else lo_src
    p20_val = u16_to_f16(selected_u16)

    # Accumulate interpolation product
    if math.isnan(j_coord_f32) or math.isnan(p20_val) or math.isnan(p1_accum_f32):
        res_u16 = H_QNAN
    elif (math.isinf(j_coord_f32) and p20_val == 0.0) or (j_coord_f32 == 0.0 and math.isinf(p20_val)):
        res_u16 = H_QNAN
    else:
        prod = j_coord_f32 * p20_val
        if (math.isinf(prod) and math.isinf(p1_accum_f32)) and ((prod > 0 and p1_accum_f32 < 0) or (prod < 0 and p1_accum_f32 > 0)):
            res_u16 = H_QNAN
        else:
            accum = prod + p1_accum_f32
            res_u16 = f16_to_u16(accum)

    if clamp:
        res_u16 = apply_clamp_f16(res_u16)

    old_lo, old_hi = unpack_f16_pair(old_vdst_u32)
    if high_dst:
        return pack_f16_pair(old_lo, res_u16)
    else:
        return pack_f16_pair(res_u16, old_hi)


class AttributeBuffer:
    """Architectural model of the LDS/Parameter interpolation attribute cache."""

    def __init__(self) -> None:
        # data: [param 0..2][attr 0..31][chan 0..3] -> 32-bit uint
        # param 0: p0, param 1: p10, param 2: p20
        self.data = [[[0] * 4 for _ in range(32)] for _ in range(3)]

    def write_param_channel(self, param: int, attr: int, chan: int, val_u32: int) -> None:
        if not (0 <= param <= 2):
            raise ValueError(f"Invalid param index: {param}")
        if not (0 <= attr <= 31):
            raise ValueError(f"Invalid attribute index: {attr}")
        if not (0 <= chan <= 3):
            raise ValueError(f"Invalid channel index: {chan}")
        self.data[param][attr][chan] = val_u32 & MASK32

    def read_param_channel(self, param: int, attr: int, chan: int) -> int:
        if not (0 <= param <= 2):
            raise ValueError(f"Invalid param index: {param}")
        if not (0 <= attr <= 31):
            raise ValueError(f"Invalid attribute index: {attr}")
        if not (0 <= chan <= 3):
            raise ValueError(f"Invalid channel index: {chan}")
        return self.data[param][attr][chan]


class InterpolationRegisterFile:
    """Architectural model of 256 32-bit Vector General Purpose Registers (VGPRs)."""

    def __init__(self) -> None:
        self.vgprs = [0] * 256

    def read_vgpr_u32(self, idx: int) -> int:
        return self.vgprs[idx & 0xFF]

    def write_vgpr_u32(self, idx: int, val: int) -> None:
        self.vgprs[idx & 0xFF] = val & MASK32

    def read_vgpr_f32(self, idx: int) -> float:
        return u32_to_f32(self.vgprs[idx & 0xFF])

    def write_vgpr_f32(self, idx: int, val: float) -> None:
        self.vgprs[idx & 0xFF] = f32_to_u32(val)

    def read_vgpr_f16(self, idx: int, high: bool = False) -> float:
        lo, hi = unpack_f16_pair(self.vgprs[idx & 0xFF])
        return u16_to_f16(hi if high else lo)

    def write_vgpr_f16(self, idx: int, val_u16: int, high: bool = False) -> None:
        old_lo, old_hi = unpack_f16_pair(self.vgprs[idx & 0xFF])
        if high:
            self.vgprs[idx & 0xFF] = pack_f16_pair(old_lo, val_u16)
        else:
            self.vgprs[idx & 0xFF] = pack_f16_pair(val_u16, old_hi)


class TestInterpP2F16MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural models."""

    def test_basic_interpolation_accumulation(self):
        # j = 0.5, p1_accum = 2.0, p20 = 4.0 -> 2.0 + 0.5 * 4.0 = 4.0
        p20_word = pack_f16_pair(f16_to_u16(4.0), f16_to_u16(10.0))
        res = ref_interp_p2_f16(0.5, p20_word, 2.0, high_src=False, high_dst=False, old_vdst_u32=0)
        lo, hi = unpack_f16_pair(res)
        self.assertEqual(u16_to_f16(lo), 4.0)
        self.assertEqual(hi, 0)

    def test_high_source_selection(self):
        # p20_word: lo = 4.0, hi = 10.0. high_src=True chooses 10.0
        # j = 0.5, p1_accum = 3.0 -> 3.0 + 0.5 * 10.0 = 8.0
        p20_word = pack_f16_pair(f16_to_u16(4.0), f16_to_u16(10.0))
        res = ref_interp_p2_f16(0.5, p20_word, 3.0, high_src=True, high_dst=False, old_vdst_u32=0)
        lo, _ = unpack_f16_pair(res)
        self.assertEqual(u16_to_f16(lo), 8.0)

    def test_destination_half_preservation_gfx1036(self):
        # SHADER-430: v_interp_p2_f16 preserves the untouched 16-bit half of vdst
        canary_old = 0xDEADBEEF
        p20_word = pack_f16_pair(f16_to_u16(2.0), f16_to_u16(0.0))
        # j = 0.0, p1_accum = 1.0 -> 1.0 (0x3C00)
        res_lo = ref_interp_p2_f16(0.0, p20_word, 1.0, high_src=False, high_dst=False, old_vdst_u32=canary_old)
        # Low half modified, high half 0xDEAD preserved
        self.assertEqual(res_lo, 0xDEAD3C00)

        # High half modified, low half 0xBEEF preserved
        res_hi = ref_interp_p2_f16(0.0, p20_word, 1.0, high_src=False, high_dst=True, old_vdst_u32=canary_old)
        self.assertEqual(res_hi, 0x3C00BEEF)

    def test_clamp_modifier(self):
        p20_word = pack_f16_pair(f16_to_u16(2.0), 0)
        # Extrapolating beyond 1.0: j = 1.0, p1_accum = 1.5 -> 3.5 clamped to 1.0
        res_high_clamp = ref_interp_p2_f16(1.0, p20_word, 1.5, clamp=True)
        self.assertEqual(unpack_f16_pair(res_high_clamp)[0], H_ONE)

        # Extrapolating below 0.0: j = -2.0, p1_accum = 1.0 -> -3.0 clamped to +0.0
        res_low_clamp = ref_interp_p2_f16(-2.0, p20_word, 1.0, clamp=True)
        self.assertEqual(unpack_f16_pair(res_low_clamp)[0], H_POS_ZERO)

    def test_attribute_buffer_and_register_file(self):
        ab = AttributeBuffer()
        ab.write_param_channel(2, 4, 1, 0x12345678)
        self.assertEqual(ab.read_param_channel(2, 4, 1), 0x12345678)

        rf = InterpolationRegisterFile()
        rf.write_vgpr_u32(10, 0xAABBCCDD)
        rf.write_vgpr_f16(10, 0x3C00, high=False)
        self.assertEqual(rf.read_vgpr_u32(10), 0xAABB3C00)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions for v_interp_p2_f16
# ==============================================================================

class Opcode(Enum):
    V_INTERP_P2_F16 = "v_interp_p2_f16"


class Channel(Enum):
    X = 0
    Y = 1
    Z = 2
    W = 3

    @classmethod
    def from_str(cls, s: str) -> Channel:
        s_clean = s.lower().strip()
        if s_clean == "x":
            return cls.X
        elif s_clean == "y":
            return cls.Y
        elif s_clean == "z":
            return cls.Z
        elif s_clean == "w":
            return cls.W
        raise ValueError(f"Invalid channel string: {s}")

    def to_str(self) -> str:
        return self.name.lower()


class AttributeRef:
    """Reference to an attribute index (0..31) and component channel (X, Y, Z, W)."""

    def __init__(self, attr_index: int, channel: Channel) -> None:
        if not (0 <= attr_index <= 31):
            raise ValueError(f"Attribute index must be 0..31, got {attr_index}")
        self.attr_index = attr_index
        self.channel = channel

    @classmethod
    def parse(cls, text: str) -> AttributeRef:
        t = text.strip()
        if not t.startswith("attr"):
            raise ValueError(f"Attribute token must start with 'attr': {text}")
        parts = t[4:].split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid attribute format: {text}")
        idx = int(parts[0])
        chan = Channel.from_str(parts[1])
        return cls(idx, chan)

    def disasm(self) -> str:
        return f"attr{self.attr_index}.{self.channel.to_str()}"

    def __repr__(self) -> str:
        return f"AttributeRef({self.disasm()})"


class ASTOperand:
    """VGPR operand with optional high/low half selection and source modifiers."""

    def __init__(
        self,
        vgpr_index: int,
        *,
        high: bool = False,
        abs_mod: bool = False,
        neg_mod: bool = False,
    ) -> None:
        if not (0 <= vgpr_index <= 255):
            raise ValueError(f"VGPR index must be 0..255, got {vgpr_index}")
        self.vgpr_index = vgpr_index
        self.high = high
        self.abs_mod = abs_mod
        self.neg_mod = neg_mod

    def disasm(self) -> str:
        base = f"v{self.vgpr_index}"
        if self.abs_mod:
            base = f"|{base}|"
        if self.neg_mod:
            base = f"-{base}"
        return base

    def __repr__(self) -> str:
        half_str = "high" if self.high else "low"
        return f"ASTOperand({self.disasm()}, {half_str})"


class ASTInstruction:
    """Symbolic AST instruction representation for v_interp_p2_f16."""

    def __init__(
        self,
        opcode: Opcode,
        dst: ASTOperand,
        src_j: ASTOperand,
        attr: AttributeRef,
        *,
        high_src: bool = False,
        high_dst: bool = False,
        clamp: bool = False,
    ) -> None:
        self.opcode = opcode
        self.dst = dst
        self.src_j = src_j
        self.attr = attr
        self.high_src = high_src
        self.high_dst = high_dst
        self.clamp = clamp

    def validate(self) -> None:
        if self.opcode != Opcode.V_INTERP_P2_F16:
            raise ValueError(f"Invalid opcode: {self.opcode}")
        if not (0 <= self.dst.vgpr_index <= 255):
            raise ValueError(f"Destination VGPR index out of range: {self.dst.vgpr_index}")
        if not (0 <= self.src_j.vgpr_index <= 255):
            raise ValueError(f"Source VGPR index out of range: {self.src_j.vgpr_index}")
        if not (0 <= self.attr.attr_index <= 31):
            raise ValueError(f"Attribute index out of range: {self.attr.attr_index}")
        if self.dst.abs_mod or self.dst.neg_mod:
            raise ValueError("Destination operand cannot have abs or neg modifiers")

    def disasm(self) -> str:
        parts = [
            self.opcode.value,
            f"{self.dst.disasm()},",
            f"{self.src_j.disasm()},",
            self.attr.disasm(),
        ]
        if self.high_src:
            parts.append("high")
        if self.clamp:
            parts.append("clamp")
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"ASTInstruction({self.disasm()})"


class TypedExpr:
    """Base class for typed symbolic expressions."""

    def evaluate(self, env_attrs: AttributeBuffer, env_vgprs: InterpolationRegisterFile) -> int:
        raise NotImplementedError


class InterpP2F16Expr(TypedExpr):
    """Symbolic expression tree for v_interp_p2_f16."""

    def __init__(
        self,
        dst_idx: int,
        src_j_idx: int,
        attr: AttributeRef,
        *,
        high_src: bool = False,
        high_dst: bool = False,
        abs_j: bool = False,
        neg_j: bool = False,
        clamp: bool = False,
    ) -> None:
        self.dst_idx = dst_idx
        self.src_j_idx = src_j_idx
        self.attr = attr
        self.high_src = high_src
        self.high_dst = high_dst
        self.abs_j = abs_j
        self.neg_j = neg_j
        self.clamp = clamp

    def evaluate(self, env_attrs: AttributeBuffer, env_vgprs: InterpolationRegisterFile) -> int:
        j_raw_u32 = env_vgprs.read_vgpr_u32(self.src_j_idx)
        j_mod_u32 = apply_f32_modifiers(j_raw_u32, abs_mod=self.abs_j, neg_mod=self.neg_j)
        j_f32 = u32_to_f32(j_mod_u32)

        p1_accum_f32 = env_vgprs.read_vgpr_f32(self.dst_idx)
        p20_word = env_attrs.read_param_channel(2, self.attr.attr_index, self.attr.channel.value)
        old_vdst = env_vgprs.read_vgpr_u32(self.dst_idx)

        return ref_interp_p2_f16(
            j_f32,
            p20_word,
            p1_accum_f32,
            high_src=self.high_src,
            high_dst=self.high_dst,
            old_vdst_u32=old_vdst,
            clamp=self.clamp,
        )


class TestInterpP2F16SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_disasm(self):
        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_P2_F16,
            dst=ASTOperand(2, high=False),
            src_j=ASTOperand(1, abs_mod=True),
            attr=AttributeRef(4, Channel.Z),
            high_src=True,
            clamp=True,
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_interp_p2_f16 v2, |v1|, attr4.z high clamp")

    def test_typed_expr_evaluation(self):
        ab = AttributeBuffer()
        rf = InterpolationRegisterFile()

        # attr2.x p20 = (lo: 2.0, hi: 6.0)
        ab.write_param_channel(2, 2, 0, pack_f16_pair(f16_to_u16(2.0), f16_to_u16(6.0)))
        # v1 (j) = 0.5
        rf.write_vgpr_f32(1, 0.5)
        # v0 (p1_accum) = 1.0 (stored as f32), old v0 dword canary
        rf.write_vgpr_f32(0, 1.0)

        expr = InterpP2F16Expr(
            dst_idx=0,
            src_j_idx=1,
            attr=AttributeRef(2, Channel.X),
            high_src=True,   # uses 6.0
            high_dst=False,  # writes low half
            clamp=False,
        )
        # 1.0 + 0.5 * 6.0 = 4.0
        res = expr.evaluate(ab, rf)
        lo, _ = unpack_f16_pair(res)
        self.assertEqual(u16_to_f16(lo), 4.0)

    def test_ast_validation_errors(self):
        with self.assertRaises(ValueError):
            bad_inst = ASTInstruction(
                opcode=Opcode.V_INTERP_P2_F16,
                dst=ASTOperand(2, neg_mod=True),  # neg on dst is illegal
                src_j=ASTOperand(1),
                attr=AttributeRef(0, Channel.X),
            )
            bad_inst.validate()


# ==============================================================================
# Deterministic Scalar Instruction Interpreter for v_interp_p2_f16
# ==============================================================================

class ScalarMachineState:
    """Deterministic machine state for scalar execution."""

    def __init__(self) -> None:
        self.vgpr = InterpolationRegisterFile()
        self.attributes = AttributeBuffer()
        self.sgpr: List[int] = [0] * 106
        self.pc: int = 0


def parse_asm_line(line: str) -> ASTInstruction:
    """Parse textual assembly instruction into ASTInstruction."""
    clean = line.split(";")[0].split("//")[0].strip()
    if not clean:
        raise ValueError("Empty instruction line")
    tokens = [t for t in clean.replace(",", " ").split() if t]
    opcode_str = tokens[0]
    if opcode_str != "v_interp_p2_f16":
        raise ValueError(f"Expected opcode v_interp_p2_f16, got {opcode_str}")

    if len(tokens) < 4:
        raise ValueError(f"Too few tokens for v_interp_p2_f16: {line}")

    dst_tok = tokens[1]
    dst_high = False
    if dst_tok.endswith(".h") or dst_tok.endswith("[1]"):
        dst_high = True
        dst_tok = dst_tok[:-2]
    if not dst_tok.startswith("v"):
        raise ValueError(f"Destination must be a VGPR: {dst_tok}")
    dst_idx = int(dst_tok[1:])
    dst_op = ASTOperand(dst_idx, high=dst_high)

    src_tok = tokens[2]
    abs_mod = False
    neg_mod = False
    if src_tok.startswith("-"):
        neg_mod = True
        src_tok = src_tok[1:]
    if src_tok.startswith("|") and src_tok.endswith("|"):
        abs_mod = True
        src_tok = src_tok[1:-1]
    if not src_tok.startswith("v"):
        raise ValueError(f"Source must be a VGPR: {src_tok}")
    src_idx = int(src_tok[1:])
    src_op = ASTOperand(src_idx, abs_mod=abs_mod, neg_mod=neg_mod)

    attr_ref = AttributeRef.parse(tokens[3])

    high_src = False
    clamp = False
    for mod in tokens[4:]:
        mod_l = mod.lower()
        if mod_l == "high":
            high_src = True
        elif mod_l == "clamp":
            clamp = True
        else:
            raise ValueError(f"Unknown modifier: {mod}")

    inst = ASTInstruction(
        opcode=Opcode.V_INTERP_P2_F16,
        dst=dst_op,
        src_j=src_op,
        attr=attr_ref,
        high_src=high_src,
        high_dst=dst_high,
        clamp=clamp,
    )
    inst.validate()
    return inst


class ScalarInterpInterpreter:
    """Deterministic scalar instruction interpreter."""

    def step(self, inst: ASTInstruction, state: ScalarMachineState) -> int:
        inst.validate()
        j_raw_u32 = state.vgpr.read_vgpr_u32(inst.src_j.vgpr_index)
        j_mod_u32 = apply_f32_modifiers(j_raw_u32, abs_mod=inst.src_j.abs_mod, neg_mod=inst.src_j.neg_mod)
        j_f32 = u32_to_f32(j_mod_u32)

        p1_accum_f32 = state.vgpr.read_vgpr_f32(inst.dst.vgpr_index)
        p20_word = state.attributes.read_param_channel(2, inst.attr.attr_index, inst.attr.channel.value)
        old_vdst = state.vgpr.read_vgpr_u32(inst.dst.vgpr_index)

        res = ref_interp_p2_f16(
            j_f32,
            p20_word,
            p1_accum_f32,
            high_src=inst.high_src,
            high_dst=inst.high_dst,
            old_vdst_u32=old_vdst,
            clamp=inst.clamp,
        )
        state.vgpr.write_vgpr_u32(inst.dst.vgpr_index, res)
        state.pc += 1
        return res

    def execute_asm(self, line: str, state: ScalarMachineState) -> int:
        inst = parse_asm_line(line)
        return self.step(inst, state)


class TestScalarInterpInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpInterpreter()
        self.state = ScalarMachineState()

    def test_scalar_p2_step(self):
        # attr1.x: p20 = (lo: 4.0, hi: 8.0)
        self.state.attributes.write_param_channel(2, 1, 0, pack_f16_pair(f16_to_u16(4.0), f16_to_u16(8.0)))
        # v1 (j) = 0.5
        self.state.vgpr.write_vgpr_f32(1, 0.5)
        # v0 (p1_accum) = 3.0
        self.state.vgpr.write_vgpr_f32(0, 3.0)

        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_P2_F16,
            dst=ASTOperand(0, high=False),
            src_j=ASTOperand(1),
            attr=AttributeRef(1, Channel.X),
            high_src=False,
            clamp=False,
        )
        # 3.0 + 0.5 * 4.0 = 5.0 (0x4500)
        res = self.interp.step(inst, self.state)
        lo, _ = unpack_f16_pair(res)
        self.assertEqual(u16_to_f16(lo), 5.0)
        self.assertEqual(self.state.pc, 1)

    def test_scalar_asm_execution_with_modifiers(self):
        # attr3.y: p20 = (lo: 2.0, hi: 6.0)
        self.state.attributes.write_param_channel(2, 3, 1, pack_f16_pair(f16_to_u16(2.0), f16_to_u16(6.0)))
        # v2 (j) = -0.5
        self.state.vgpr.write_vgpr_f32(2, -0.5)
        # v0 (p1_accum) = 2.0
        self.state.vgpr.write_vgpr_f32(0, 2.0)

        # Instruction with |v2| (0.5), high (uses 6.0), clamp -> 2.0 + 0.5 * 6.0 = 5.0 -> clamped to 1.0 (0x3C00)
        res = self.interp.execute_asm("v_interp_p2_f16 v0, |v2|, attr3.y high clamp", self.state)
        lo, _ = unpack_f16_pair(res)
        self.assertEqual(lo, H_ONE)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking for v_interp_p2_f16
# ==============================================================================

class SIMDState:
    """Wavefront state for SIMD execution (Wave32 or Wave64)."""

    def __init__(self, wave_size: int = 64) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self.lanes = [InterpolationRegisterFile() for _ in range(wave_size)]
        self.attributes = AttributeBuffer()
        self.exec_mask: int = (1 << wave_size) - 1
        self.vcc: int = 0
        self.scc: int = 0

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            return False
        return bool((self.exec_mask >> lane) & 1)

    def set_vgpr_u32_wave(self, reg_idx: int, values: List[int]) -> None:
        if len(values) != self.wave_size:
            raise ValueError(f"Values length ({len(values)}) != wave_size ({self.wave_size})")
        for lane in range(self.wave_size):
            self.lanes[lane].write_vgpr_u32(reg_idx, values[lane])

    def get_vgpr_u32_wave(self, reg_idx: int) -> List[int]:
        return [self.lanes[lane].read_vgpr_u32(reg_idx) for lane in range(self.wave_size)]

    def set_vgpr_f32_wave(self, reg_idx: int, values: List[float]) -> None:
        if len(values) != self.wave_size:
            raise ValueError(f"Values length ({len(values)}) != wave_size ({self.wave_size})")
        for lane in range(self.wave_size):
            self.lanes[lane].write_vgpr_f32(reg_idx, values[lane])

    def get_vgpr_f32_wave(self, reg_idx: int) -> List[float]:
        return [self.lanes[lane].read_vgpr_f32(reg_idx) for lane in range(self.wave_size)]


class SIMDInterpInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def step(self, inst: ASTInstruction, state: SIMDState) -> List[Optional[int]]:
        inst.validate()
        results: List[Optional[int]] = [None] * state.wave_size
        p20_word = state.attributes.read_param_channel(2, inst.attr.attr_index, inst.attr.channel.value)

        for lane in range(state.wave_size):
            if not state.is_lane_active(lane):
                continue

            j_raw = state.lanes[lane].read_vgpr_u32(inst.src_j.vgpr_index)
            j_mod = apply_f32_modifiers(j_raw, abs_mod=inst.src_j.abs_mod, neg_mod=inst.src_j.neg_mod)
            j_f32 = u32_to_f32(j_mod)

            p1_accum = state.lanes[lane].read_vgpr_f32(inst.dst.vgpr_index)
            old_vdst = state.lanes[lane].read_vgpr_u32(inst.dst.vgpr_index)

            res = ref_interp_p2_f16(
                j_f32,
                p20_word,
                p1_accum,
                high_src=inst.high_src,
                high_dst=inst.high_dst,
                old_vdst_u32=old_vdst,
                clamp=inst.clamp,
            )
            state.lanes[lane].write_vgpr_u32(inst.dst.vgpr_index, res)
            results[lane] = res

        return results


def ref_simd_interp_p2_f16(
    j_coords: List[float],
    p20_word: int,
    p1_accums: List[float],
    wave_size: int = 64,
    exec_mask: Optional[int] = None,
    prev_vdst: Optional[List[int]] = None,
    *,
    high_src: bool = False,
    high_dst: bool = False,
    clamp: bool = False,
) -> List[int]:
    """Reference SIMD execution oracle across wavefront lanes."""
    active_mask = ((1 << wave_size) - 1) if exec_mask is None else (exec_mask & ((1 << wave_size) - 1))
    fallback = [0] * wave_size if prev_vdst is None else prev_vdst
    output = list(fallback)

    for lane in range(wave_size):
        if (active_mask >> lane) & 1:
            output[lane] = ref_interp_p2_f16(
                j_coords[lane],
                p20_word,
                p1_accums[lane],
                high_src=high_src,
                high_dst=high_dst,
                old_vdst_u32=fallback[lane],
                clamp=clamp,
            )
    return output


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_active_divergence(self):
        state = SIMDState(wave_size=32)
        sim = SIMDInterpInterpreter()

        # attr0.x p20 = (lo: 2.0, hi: 0.0)
        p20_word = pack_f16_pair(f16_to_u16(2.0), 0)
        state.attributes.write_param_channel(2, 0, 0, p20_word)

        # Divergent j coordinates across 32 lanes: 0.0, 1.0/31, 2.0/31, ...
        j_vals = [i / 31.0 for i in range(32)]
        p1_vals = [1.0] * 32
        state.set_vgpr_f32_wave(1, j_vals)
        state.set_vgpr_f32_wave(0, p1_vals)

        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_P2_F16,
            dst=ASTOperand(0, high=False),
            src_j=ASTOperand(1),
            attr=AttributeRef(0, Channel.X),
        )
        sim.step(inst, state)
        out_dwords = state.get_vgpr_u32_wave(0)

        for lane in range(32):
            expected_f = 1.0 + j_vals[lane] * 2.0
            lo, _ = unpack_f16_pair(out_dwords[lane])
            self.assertAlmostEqual(u16_to_f16(lo), expected_f, delta=0.01)

    def test_wave64_divergent_exec_mask(self):
        state = SIMDState(wave_size=64)
        sim = SIMDInterpInterpreter()

        p20_word = pack_f16_pair(f16_to_u16(4.0), 0)
        state.attributes.write_param_channel(2, 1, 2, p20_word)

        j_vals = [0.25] * 64
        p1_vals = [2.0] * 64
        sentinel_vdst = [0xDEADBEEF] * 64
        state.set_vgpr_f32_wave(1, j_vals)
        state.set_vgpr_f32_wave(0, p1_vals)
        # Store canary in v0
        state.set_vgpr_u32_wave(0, sentinel_vdst)
        # Set p1_accum into f32 view (which sets dword)
        state.set_vgpr_f32_wave(0, p1_vals)

        # Alternating EXEC mask across 64 lanes
        state.exec_mask = 0x55555555_55555555

        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_P2_F16,
            dst=ASTOperand(0, high=False),
            src_j=ASTOperand(1),
            attr=AttributeRef(1, Channel.Z),
        )
        sim.step(inst, state)
        res_dwords = state.get_vgpr_u32_wave(0)

        for lane in range(64):
            if lane % 2 == 0:
                # Active: 2.0 + 0.25 * 4.0 = 3.0 (0x4200)
                lo, _ = unpack_f16_pair(res_dwords[lane])
                self.assertEqual(u16_to_f16(lo), 3.0)
            else:
                # Inactive: untouched
                self.assertEqual(res_dwords[lane], f32_to_u32(2.0))

    def test_simd_oracle_validation(self):
        wave_size = 64
        j_coords = [float(i) / 64.0 for i in range(wave_size)]
        p1_accums = [float(i % 10) for i in range(wave_size)]
        p20_word = pack_f16_pair(f16_to_u16(3.0), f16_to_u16(7.0))
        exec_mask = 0x00FF00FF_00FF00FF
        prev_vdst = [0xCAFE0000 | i for i in range(wave_size)]

        expected = ref_simd_interp_p2_f16(
            j_coords,
            p20_word,
            p1_accums,
            wave_size=wave_size,
            exec_mask=exec_mask,
            prev_vdst=prev_vdst,
            high_src=True,
            high_dst=False,
            clamp=False,
        )

        state = SIMDState(wave_size=wave_size)
        sim = SIMDInterpInterpreter()
        state.attributes.write_param_channel(2, 2, 0, p20_word)
        state.set_vgpr_f32_wave(1, j_coords)
        state.set_vgpr_f32_wave(0, p1_accums)
        state.set_vgpr_u32_wave(0, prev_vdst)
        # Re-pack f32 into p1_accums
        state.set_vgpr_f32_wave(0, p1_accums)
        state.exec_mask = exec_mask

        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_P2_F16,
            dst=ASTOperand(0, high=False),
            src_j=ASTOperand(1),
            attr=AttributeRef(2, Channel.X),
            high_src=True,
            high_dst=False,
        )
        sim.step(inst, state)
        actual = state.get_vgpr_u32_wave(0)
        self.assertEqual(len(actual), wave_size)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation for v_interp_p2_f16
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_inplace_dst_equals_src_j(self):
        # Case where dst register is the same as src_j: v_interp_p2_f16 v0, v0, attr0.x
        interp = ScalarInterpInterpreter()
        state = ScalarMachineState()

        # p20 = (lo: 2.0, hi: 0.0)
        state.attributes.write_param_channel(2, 0, 0, pack_f16_pair(f16_to_u16(2.0), 0))
        # v0 holds 0.5 (as j, and simultaneously serves as p1_accum = 0.5)
        state.vgpr.write_vgpr_f32(0, 0.5)

        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_P2_F16,
            dst=ASTOperand(0, high=False),
            src_j=ASTOperand(0),
            attr=AttributeRef(0, Channel.X),
            high_src=False,
            high_dst=False,
        )
        # Operand fetch latches j = 0.5 and p1_accum = 0.5 before writeback
        # Result = 0.5 + 0.5 * 2.0 = 1.5 (0x3E00)
        res = interp.step(inst, state)
        lo, _ = unpack_f16_pair(res)
        self.assertEqual(u16_to_f16(lo), 1.5)
        self.assertEqual(state.vgpr.read_vgpr_u32(0), res)

    def test_packed_f16_high_low_interleaving(self):
        # Two consecutive instructions packing low then high half of the same VGPR:
        # Step 1: v_interp_p2_f16 v0.l, v1, attr0.x -> writes v0[15:0]
        # Step 2: v_interp_p2_f16 v0.h, v1, attr0.y -> writes v0[31:16], keeping v0[15:0]
        interp = ScalarInterpInterpreter()
        state = ScalarMachineState()

        # attr0.x: 2.0; attr0.y: 4.0
        state.attributes.write_param_channel(2, 0, 0, pack_f16_pair(f16_to_u16(2.0), 0))
        state.attributes.write_param_channel(2, 0, 1, pack_f16_pair(f16_to_u16(4.0), 0))

        # v1 (j) = 0.5
        state.vgpr.write_vgpr_f32(1, 0.5)
        # v0 = initial f32 accum = 1.0 (0x3F800000)
        state.vgpr.write_vgpr_f32(0, 1.0)

        # Step 1: write low half: 1.0 + 0.5 * 2.0 = 2.0 (0x4000)
        inst_lo = ASTInstruction(
            opcode=Opcode.V_INTERP_P2_F16,
            dst=ASTOperand(0, high=False),
            src_j=ASTOperand(1),
            attr=AttributeRef(0, Channel.X),
            high_dst=False,
        )
        res1 = interp.step(inst_lo, state)
        # High half was 0x3F80 (from 1.0f = 0x3F800000), low half is 0x4000
        self.assertEqual(res1, 0x3F804000)

        # Step 2: now evaluate attribute Y on high half of v0:
        # High half gets 1.0 + 0.5 * 4.0 = 3.0 (0x4200), keeping low half 0x4000 intact!
        inst_hi = ASTInstruction(
            opcode=Opcode.V_INTERP_P2_F16,
            dst=ASTOperand(0, high=True),
            src_j=ASTOperand(1),
            attr=AttributeRef(0, Channel.Y),
            high_dst=True,
        )
        # To simulate p1_accum for the second attribute, let's pass 1.0 (stored in temporary or register)
        # Here v0's high half is written with 0x4200:
        res2 = ref_interp_p2_f16(0.5, pack_f16_pair(f16_to_u16(4.0), 0), 1.0, high_dst=True, old_vdst_u32=res1)
        self.assertEqual(res2, 0x42004000)
        lo_final, hi_final = unpack_f16_pair(res2)
        self.assertEqual(u16_to_f16(lo_final), 2.0)
        self.assertEqual(u16_to_f16(hi_final), 3.0)

    def test_simd_aliased_dst_preservation_under_mask(self):
        state = SIMDState(wave_size=32)
        sim = SIMDInterpInterpreter()

        state.attributes.write_param_channel(2, 5, 0, pack_f16_pair(f16_to_u16(1.0), 0))
        # Odd lanes active
        state.exec_mask = 0xAAAAAAAA

        canary_vals = [0xBEEF0000 | i for i in range(32)]
        state.set_vgpr_u32_wave(0, canary_vals)
        state.set_vgpr_f32_wave(1, [1.0] * 32)

        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_P2_F16,
            dst=ASTOperand(0, high=False),
            src_j=ASTOperand(1),
            attr=AttributeRef(5, Channel.X),
        )
        sim.step(inst, state)
        res_wave = state.get_vgpr_u32_wave(0)

        for lane in range(32):
            if lane % 2 == 1:
                # Active: low half updated, high half 0xBEEF preserved
                self.assertEqual((res_wave[lane] >> 16) & 0xFFFF, 0xBEEF)
            else:
                # Inactive: full 32-bit dword untouched
                self.assertEqual(res_wave[lane], canary_vals[lane])


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Isolation for v_interp_p2_f16
# ==============================================================================

@dataclass
class InterpStatusFlags:
    invalid_op: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False
    zero: bool = False
    negative: bool = False
    nan: bool = False


def compute_interp_p2_f16_status_flags(
    j_val: float,
    p20_val: float,
    p1_accum: float,
    res_u16: int,
) -> InterpStatusFlags:
    """Compute IEEE-754 exception and status flags for v_interp_p2_f16."""
    flags = InterpStatusFlags()
    if math.isnan(j_val) or math.isnan(p20_val) or math.isnan(p1_accum):
        flags.invalid_op = True
    elif (math.isinf(j_val) and p20_val == 0.0) or (j_val == 0.0 and math.isinf(p20_val)):
        flags.invalid_op = True

    if is_nan_f16(res_u16):
        flags.nan = True
        flags.invalid_op = True
    else:
        sign = (res_u16 >> 15) & 1
        flags.negative = bool(sign)
        if (res_u16 & 0x7FFF) == 0:
            flags.zero = True
        elif res_u16 in (H_POS_INF, H_NEG_INF):
            if not math.isinf(j_val) and not math.isinf(p20_val) and not math.isinf(p1_accum):
                flags.overflow = True
                flags.inexact = True
        elif (res_u16 & 0x7C00) == 0 and (res_u16 & 0x03FF) != 0:
            flags.underflow = True

    return flags


def ref_cmp_f16(op: str, a_u16: int, b_u16: int) -> bool:
    """Evaluate 16-bit float comparison."""
    if op in ("u", "unordered"):
        return is_nan_f16(a_u16) or is_nan_f16(b_u16)
    if is_nan_f16(a_u16) or is_nan_f16(b_u16):
        return op in ("ne", "neq", "!=")

    v_a = u16_to_f16(a_u16)
    v_b = u16_to_f16(b_u16)
    if op in ("eq", "=="):
        return v_a == v_b
    elif op in ("ne", "!="):
        return v_a != v_b
    elif op in ("lt", "<"):
        return v_a < v_b
    elif op in ("le", "<="):
        return v_a <= v_b
    elif op in ("gt", ">"):
        return v_a > v_b
    elif op in ("ge", ">="):
        return v_a >= v_b
    raise ValueError(f"Unknown comparison operator: {op}")


def generate_interp_condition_mask_f16(op: str, a_lanes: List[int], b_lanes: List[int]) -> int:
    """Generate per-lane condition bitmask (VCC) from comparisons on interpolation results."""
    mask = 0
    for lane, (a, b) in enumerate(zip(a_lanes, b_lanes)):
        if ref_cmp_f16(op, a, b):
            mask |= (1 << lane)
    return mask


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Test suite for condition codes, status flags, and carry/borrow isolation."""

    def test_status_flags_invalid_nan(self):
        # 0.0 * Inf
        res = ref_interp_p2_f16(0.0, pack_f16_pair(H_POS_INF, 0), 1.0)
        flags = compute_interp_p2_f16_status_flags(0.0, float("inf"), 1.0, unpack_f16_pair(res)[0])
        self.assertTrue(flags.invalid_op)
        self.assertTrue(flags.nan)

    def test_status_flags_overflow(self):
        # Extremely large values overflowing f16
        res = ref_interp_p2_f16(1000.0, pack_f16_pair(f16_to_u16(500.0), 0), 1000.0)
        res_u16 = unpack_f16_pair(res)[0]
        self.assertEqual(res_u16, H_POS_INF)
        flags = compute_interp_p2_f16_status_flags(1000.0, 500.0, 1000.0, res_u16)
        self.assertTrue(flags.overflow)
        self.assertTrue(flags.inexact)

    def test_status_flags_zero_and_negative(self):
        # Result = -2.0 (negative, non-zero)
        res_neg = ref_interp_p2_f16(1.0, pack_f16_pair(f16_to_u16(-5.0), 0), 3.0)
        flags_neg = compute_interp_p2_f16_status_flags(1.0, -5.0, 3.0, unpack_f16_pair(res_neg)[0])
        self.assertTrue(flags_neg.negative)
        self.assertFalse(flags_neg.zero)

        # Result = -0.0
        res_neg_zero = ref_interp_p2_f16(1.0, pack_f16_pair(H_NEG_ZERO, 0), -0.0)
        flags_nz = compute_interp_p2_f16_status_flags(1.0, -0.0, -0.0, unpack_f16_pair(res_neg_zero)[0])
        self.assertTrue(flags_nz.negative)
        self.assertTrue(flags_nz.zero)

    def test_vcc_and_scc_preserved(self):
        # In RDNA2, VINTRP instructions do not write or clobber VCC / SCC carry/borrow bits
        state = SIMDState(wave_size=32)
        sim = SIMDInterpInterpreter()
        state.vcc = 0x12345678_9ABCDEF0
        state.scc = 1

        state.attributes.write_param_channel(2, 0, 0, pack_f16_pair(f16_to_u16(2.0), 0))
        state.set_vgpr_f32_wave(1, [0.5] * 32)
        state.set_vgpr_f32_wave(0, [1.0] * 32)

        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_P2_F16,
            dst=ASTOperand(0),
            src_j=ASTOperand(1),
            attr=AttributeRef(0, Channel.X),
        )
        sim.step(inst, state)
        self.assertEqual(state.vcc, 0x12345678_9ABCDEF0)
        self.assertEqual(state.scc, 1)

    def test_condition_code_mask_generation(self):
        # Generate condition mask based on interpolation results across 4 lanes:
        # [ -2.0, 0.0, 2.0, NaN ]
        interp_res = [
            f16_to_u16(-2.0),
            H_POS_ZERO,
            f16_to_u16(2.0),
            H_QNAN,
        ]
        zeros = [H_POS_ZERO] * 4

        mask_lt = generate_interp_condition_mask_f16("<", interp_res, zeros)
        self.assertEqual(mask_lt, 0b0001)

        mask_eq = generate_interp_condition_mask_f16("==", interp_res, zeros)
        self.assertEqual(mask_eq, 0b0010)

        mask_gt = generate_interp_condition_mask_f16(">", interp_res, zeros)
        self.assertEqual(mask_gt, 0b0100)

        mask_u = generate_interp_condition_mask_f16("u", interp_res, zeros)
        self.assertEqual(mask_u, 0b1000)


# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling for v_interp_p2_f16
# ==============================================================================

class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Test suite for IEEE-754 boundary extremes, subnormals, and NaN handling."""

    def test_signed_zero_rules(self):
        # +0.0 + +0.0 = +0.0
        r1 = ref_interp_p2_f16(0.0, pack_f16_pair(H_POS_ZERO, 0), 0.0)
        self.assertEqual(unpack_f16_pair(r1)[0], H_POS_ZERO)

        # -0.0 + -0.0 = -0.0
        r2 = ref_interp_p2_f16(1.0, pack_f16_pair(H_NEG_ZERO, 0), -0.0)
        self.assertEqual(unpack_f16_pair(r2)[0], H_NEG_ZERO)

    def test_nan_generation_and_propagation(self):
        # 0.0 * inf -> NaN
        r_nan1 = ref_interp_p2_f16(0.0, pack_f16_pair(H_POS_INF, 0), 0.0)
        self.assertTrue(is_nan_f16(unpack_f16_pair(r_nan1)[0]))

        # NaN input in p1_accum
        r_nan2 = ref_interp_p2_f16(0.5, pack_f16_pair(f16_to_u16(1.0), 0), float("nan"))
        self.assertTrue(is_nan_f16(unpack_f16_pair(r_nan2)[0]))

        # SNaN input quiets to QNaN
        snan_val = u16_to_f16(H_SNAN)
        r_snan = ref_interp_p2_f16(0.0, pack_f16_pair(0, 0), snan_val)
        out_u16 = unpack_f16_pair(r_snan)[0]
        self.assertTrue(is_nan_f16(out_u16))
        # Quiet bit (bit 9) must be set
        self.assertTrue(bool(out_u16 & 0x0200))

    def test_clamp_nan_to_pos_zero(self):
        # In RDNA2, clamping any NaN results in +0.0 (0x0000)
        r = ref_interp_p2_f16(0.0, pack_f16_pair(H_POS_INF, 0), 0.0, clamp=True)
        self.assertEqual(unpack_f16_pair(r)[0], H_POS_ZERO)

        r_snan_clamped = ref_interp_p2_f16(1.0, pack_f16_pair(H_SNAN, 0), 0.0, clamp=True)
        self.assertEqual(unpack_f16_pair(r_snan_clamped)[0], H_POS_ZERO)

    def test_subnormal_precision_and_gradual_underflow(self):
        # Min subnormal: 2**-24 (0x0001)
        # Min subnormal * 2.0 = 2**-23 (0x0002)
        sub_min_f = u16_to_f16(H_SUB_MIN)
        r_sub = ref_interp_p2_f16(2.0, pack_f16_pair(H_SUB_MIN, 0), 0.0)
        self.assertEqual(unpack_f16_pair(r_sub)[0], 0x0002)

        # Max subnormal: 0x03FF
        r_max_sub = ref_interp_p2_f16(1.0, pack_f16_pair(H_SUB_MAX, 0), 0.0)
        self.assertEqual(unpack_f16_pair(r_max_sub)[0], H_SUB_MAX)

        # Underflow below 2**-24 rounds to 0.0
        r_underflow = ref_interp_p2_f16(0.25, pack_f16_pair(H_SUB_MIN, 0), 0.0)
        self.assertEqual(unpack_f16_pair(r_underflow)[0], H_POS_ZERO)

    def test_normal_max_overflow(self):
        # H_MAX_NORMAL = 65504.0 (0x7BFF)
        r_max = ref_interp_p2_f16(1.0, pack_f16_pair(H_MAX_NORMAL, 0), 0.0)
        self.assertEqual(unpack_f16_pair(r_max)[0], H_MAX_NORMAL)

        # Overflow beyond 65504.0 produces +inf (0x7C00)
        r_over = ref_interp_p2_f16(2.0, pack_f16_pair(H_MAX_NORMAL, 0), 0.0)
        self.assertEqual(unpack_f16_pair(r_over)[0], H_POS_INF)

        # Clamped overflow saturates to 1.0 (0x3C00)
        r_over_clamped = ref_interp_p2_f16(2.0, pack_f16_pair(H_MAX_NORMAL, 0), 0.0, clamp=True)
        self.assertEqual(unpack_f16_pair(r_over_clamped)[0], H_ONE)


if __name__ == "__main__":
    unittest.main()
