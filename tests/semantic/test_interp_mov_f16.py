from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_interp_mov_f16.

Tests the half-precision parameter interpolation attribute move instruction:
    v_interp_mov_f16 vD, param, attrN.chan [high] [clamp]

Lowered semantics:
    Moves a 16-bit float attribute from parameter source (p0, p10, p20)
    corresponding to attribute N and channel (x, y, z, w).
    Supports:
      - `high` source modifier: selects high 16-bit half (bits [31:16]) or low 16-bit half (bits [15:0])
        of the packed 32-bit attribute channel word.
      - `high` destination modifier: selects whether to write into bits [31:16] or bits [15:0] of vD,
        strictly preserving the other 16-bit half of the 32-bit destination VGPR.
      - `clamp` modifier: clamps the 16-bit float result to [0.0, 1.0], mapping NaNs to +0.0.

Architectural model:
    Pixel shader VINTRP pipeline where attributes are packed per channel.
    Flat copying of provoking vertex (p0) or slope parameters (p10, p20).
"""

from enum import Enum, auto
import math
import struct
import sys
import unittest

# IEEE 754 half-precision (binary16) constants
H_POS_ZERO = 0x0000
H_NEG_ZERO = 0x8000
H_ONE = 0x3C00
H_NEG_ONE = 0xBC00
H_TWO = 0x4000
H_HALF = 0x3800
H_POS_INF = 0x7C00
H_NEG_INF = 0xFC00
H_QNAN = 0x7E00
H_SNAN = 0x7C01
H_SUB_MIN = 0x0001
H_SUB_MAX = 0x03FF
H_MIN_NORMAL = 0x0400
H_MAX_NORMAL = 0x7BFF

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF


def f16_to_u16(val: float) -> int:
    """Pack Python float into IEEE 754 binary16 (half-precision) unsigned 16-bit int."""
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return H_POS_INF if val > 0 else H_NEG_INF


def u16_to_f16(bits: int) -> float:
    """Unpack IEEE 754 binary16 unsigned 16-bit int into Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def pack_f16_pair(lo_u16: int, hi_u16: int) -> int:
    """Pack two 16-bit words into a single 32-bit dword [hi:lo]."""
    return (lo_u16 & MASK16) | ((hi_u16 & MASK16) << 16)


def unpack_f16_pair(val_u32: int) -> tuple[int, int]:
    """Unpack a 32-bit dword into (low 16-bit word, high 16-bit word)."""
    return val_u32 & MASK16, (val_u32 >> 16) & MASK16


def apply_clamp_f16(bits_u16: int) -> int:
    """Apply RDNA2 clamp modifier [0.0, 1.0] to half-precision float, mapping NaNs to +0.0."""
    val = u16_to_f16(bits_u16)
    if math.isnan(val):
        return H_POS_ZERO
    if val < 0.0:
        return H_POS_ZERO
    if val > 1.0:
        return H_ONE
    return f16_to_u16(val)


def ref_interp_mov_f16(
    attr_word_u32: int,
    *,
    high_src: bool = False,
    high_dst: bool = False,
    old_dst_u32: int = 0,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for v_interp_mov_f16.

    Selects 16-bit source half from attr_word_u32, optionally clamps, and merges
    into old_dst_u32 at either high or low half, strictly preserving the untouched half.
    """
    lo_src, hi_src = unpack_f16_pair(attr_word_u32)
    selected_u16 = hi_src if high_src else lo_src

    if clamp:
        result_u16 = apply_clamp_f16(selected_u16)
    else:
        result_u16 = selected_u16

    old_lo, old_hi = unpack_f16_pair(old_dst_u32)
    if high_dst:
        return pack_f16_pair(old_lo, result_u16)
    else:
        return pack_f16_pair(result_u16, old_hi)


class AttributeBuffer:
    """Architectural model of the LDS/Parameter interpolation attribute cache.

    Maintains 32 attributes x 4 channels (X, Y, Z, W) x 3 parameters (p0, p10, p20).
    """

    def __init__(self) -> None:
        # storage: [param_type 0..2][attr 0..31][chan 0..3] -> 32-bit uint
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
    """Register file for 16-bit and 32-bit VGPR operations."""

    def __init__(self) -> None:
        self.vgprs = [0] * 256

    def read_vgpr_u32(self, idx: int) -> int:
        return self.vgprs[idx]

    def write_vgpr_u32(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32

    def read_vgpr_f16(self, idx: int, high: bool = False) -> float:
        lo, hi = unpack_f16_pair(self.vgprs[idx])
        return u16_to_f16(hi if high else lo)

    def write_vgpr_f16(self, idx: int, val_u16: int, high: bool = False) -> None:
        old_lo, old_hi = unpack_f16_pair(self.vgprs[idx])
        if high:
            self.vgprs[idx] = pack_f16_pair(old_lo, val_u16)
        else:
            self.vgprs[idx] = pack_f16_pair(val_u16, old_hi)


class TestInterpMovF16MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural models."""

    def test_basic_low_and_high_source_selection(self):
        # Attribute packed word with lo = 1.5 (0x3E00), hi = -2.0 (0xC000)
        attr_word = pack_f16_pair(0x3E00, 0xC000)

        # Move low source half to low destination half
        res_lo = ref_interp_mov_f16(attr_word, high_src=False, high_dst=False, old_dst_u32=0)
        self.assertEqual(res_lo, 0x00003E00)

        # Move high source half to low destination half
        res_hi = ref_interp_mov_f16(attr_word, high_src=True, high_dst=False, old_dst_u32=0)
        self.assertEqual(res_hi, 0x0000C000)

    def test_destination_half_preservation(self):
        # Pre-existing destination value
        canary_old = 0xAAAA5555  # hi: 0xAAAA, lo: 0x5555
        new_val_u16 = 0x3C00     # 1.0

        # Writing to low half preserves high half
        res_write_lo = ref_interp_mov_f16(new_val_u16, high_src=False, high_dst=False, old_dst_u32=canary_old)
        self.assertEqual(res_write_lo, 0xAAAA3C00)

        # Writing to high half preserves low half
        res_write_hi = ref_interp_mov_f16(new_val_u16, high_src=False, high_dst=True, old_dst_u32=canary_old)
        self.assertEqual(res_write_hi, 0x3C005555)

    def test_clamp_modifier(self):
        # 2.0 (0x4000) clamped to 1.0 (0x3C00)
        res_clamp_pos = ref_interp_mov_f16(0x4000, clamp=True)
        self.assertEqual(res_clamp_pos, H_ONE)

        # -1.0 (0xBC00) clamped to +0.0 (0x0000)
        res_clamp_neg = ref_interp_mov_f16(0xBC00, clamp=True)
        self.assertEqual(res_clamp_neg, H_POS_ZERO)

        # NaN clamped to +0.0
        res_clamp_nan = ref_interp_mov_f16(H_QNAN, clamp=True)
        self.assertEqual(res_clamp_nan, H_POS_ZERO)

    def test_attribute_buffer_storage(self):
        buf = AttributeBuffer()
        # Write p0, attr 5, chan Y (1)
        buf.write_param_channel(0, 5, 1, 0x12345678)
        self.assertEqual(buf.read_param_channel(0, 5, 1), 0x12345678)

        # Other channels remain 0
        self.assertEqual(buf.read_param_channel(0, 5, 0), 0)
        self.assertEqual(buf.read_param_channel(1, 5, 1), 0)


class Opcode(Enum):
    V_INTERP_MOV_F16 = "v_interp_mov_f16"


class ParamType(Enum):
    P0 = "p0"
    P10 = "p10"
    P20 = "p20"

    @property
    def index(self) -> int:
        if self == ParamType.P0:
            return 0
        elif self == ParamType.P10:
            return 1
        elif self == ParamType.P20:
            return 2
        raise ValueError(f"Unknown param type: {self}")


class Channel(Enum):
    X = 0
    Y = 1
    Z = 2
    W = 3

    @classmethod
    def from_str(cls, s: str) -> Channel:
        s_lower = s.lower()
        if s_lower == "x":
            return cls.X
        elif s_lower == "y":
            return cls.Y
        elif s_lower == "z":
            return cls.Z
        elif s_lower == "w":
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

    def disasm(self) -> str:
        return f"attr{self.attr_index}.{self.channel.to_str()}"

    def __repr__(self) -> str:
        return f"AttributeRef({self.disasm()})"


class ASTOperand:
    """Destination VGPR operand with high/low half selection."""

    def __init__(self, vgpr_index: int, *, high: bool = False) -> None:
        if not (0 <= vgpr_index <= 255):
            raise ValueError(f"VGPR index must be 0..255, got {vgpr_index}")
        self.vgpr_index = vgpr_index
        self.high = high

    def disasm(self) -> str:
        return f"v{self.vgpr_index}"

    def __repr__(self) -> str:
        half_str = "high" if self.high else "low"
        return f"ASTOperand(v{self.vgpr_index}, {half_str})"


class ASTInstruction:
    """Symbolic AST instruction representation for v_interp_mov_f16."""

    def __init__(
        self,
        opcode: Opcode,
        dst: ASTOperand,
        param: ParamType,
        attr: AttributeRef,
        *,
        high_src: bool = False,
        clamp: bool = False,
    ) -> None:
        self.opcode = opcode
        self.dst = dst
        self.param = param
        self.attr = attr
        self.high_src = high_src
        self.clamp = clamp

    def validate(self) -> None:
        if self.opcode != Opcode.V_INTERP_MOV_F16:
            raise ValueError(f"Invalid opcode: {self.opcode}")
        if not (0 <= self.dst.vgpr_index <= 255):
            raise ValueError(f"VGPR index out of range: {self.dst.vgpr_index}")
        if not (0 <= self.attr.attr_index <= 31):
            raise ValueError(f"Attribute index out of range: {self.attr.attr_index}")

    def disasm(self) -> str:
        parts = [
            self.opcode.value,
            f"{self.dst.disasm()},",
            f"{self.param.value},",
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

    def evaluate(self, env: AttributeBuffer, old_dst_u32: int) -> int:
        raise NotImplementedError


class InterpMovExpr(TypedExpr):
    """Symbolic expression tree for interpolation attribute move."""

    def __init__(
        self,
        param: ParamType,
        attr: AttributeRef,
        *,
        high_src: bool = False,
        high_dst: bool = False,
        clamp: bool = False,
    ) -> None:
        self.param = param
        self.attr = attr
        self.high_src = high_src
        self.high_dst = high_dst
        self.clamp = clamp

    def evaluate(self, env: AttributeBuffer, old_dst_u32: int) -> int:
        attr_word = env.read_param_channel(
            self.param.index,
            self.attr.attr_index,
            self.attr.channel.value,
        )
        return ref_interp_mov_f16(
            attr_word,
            high_src=self.high_src,
            high_dst=self.high_dst,
            old_dst_u32=old_dst_u32,
            clamp=self.clamp,
        )

    def __repr__(self) -> str:
        return f"InterpMovExpr({self.param.value}, {self.attr.disasm()}, high_src={self.high_src}, clamp={self.clamp})"


class TestInterpMovF16SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_disasm(self):
        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_MOV_F16,
            dst=ASTOperand(2, high=False),
            param=ParamType.P0,
            attr=AttributeRef(3, Channel.Y),
            high_src=True,
            clamp=True,
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_interp_mov_f16 v2, p0, attr3.y high clamp")

    def test_ast_validation_errors(self):
        # Attribute index out of range (> 31)
        with self.assertRaises(ValueError):
            AttributeRef(32, Channel.X)

        # VGPR index out of range (> 255)
        with self.assertRaises(ValueError):
            ASTOperand(256)

    def test_typed_expression_evaluation(self):
        buf = AttributeBuffer()
        # Set p0 attr 1 chan X to (hi: 2.0, lo: 0.5)
        word = pack_f16_pair(H_HALF, H_TWO)
        buf.write_param_channel(0, 1, Channel.X.value, word)

        expr = InterpMovExpr(
            param=ParamType.P0,
            attr=AttributeRef(1, Channel.X),
            high_src=True,   # selects 2.0
            high_dst=False,
            clamp=True,      # 2.0 clamped to 1.0 (0x3C00)
        )
        old_val = 0xAAAA0000
        res = expr.evaluate(buf, old_val)
        # Should write 0x3C00 to low half, keeping 0xAAAA in high half
        self.assertEqual(res, 0xAAAA3C00)


class InterpMovF16ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_interp_mov_f16."""

    def __init__(
        self,
        reg_file: InterpolationRegisterFile | None = None,
        attr_buf: AttributeBuffer | None = None,
    ) -> None:
        self.reg_file = reg_file if reg_file is not None else InterpolationRegisterFile()
        self.attr_buf = attr_buf if attr_buf is not None else AttributeBuffer()

    def execute(self, inst: ASTInstruction) -> None:
        inst.validate()
        attr_word = self.attr_buf.read_param_channel(
            inst.param.index,
            inst.attr.attr_index,
            inst.attr.channel.value,
        )
        old_val = self.reg_file.read_vgpr_u32(inst.dst.vgpr_index)
        new_val = ref_interp_mov_f16(
            attr_word,
            high_src=inst.high_src,
            high_dst=inst.dst.high,
            old_dst_u32=old_val,
            clamp=inst.clamp,
        )
        self.reg_file.write_vgpr_u32(inst.dst.vgpr_index, new_val)


class TestInterpMovF16ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.rf = InterpolationRegisterFile()
        self.buf = AttributeBuffer()
        self.interp = InterpMovF16ScalarInterpreter(self.rf, self.buf)

    def test_scalar_interpreter_basic_mov_lo_and_hi(self):
        # Pack attr 2 chan Z: lo = 1.0 (0x3C00), hi = 2.0 (0x4000)
        self.buf.write_param_channel(0, 2, Channel.Z.value, pack_f16_pair(H_ONE, H_TWO))

        # 1. Move low half to low half of v0
        inst_lo = ASTInstruction(
            opcode=Opcode.V_INTERP_MOV_F16,
            dst=ASTOperand(0, high=False),
            param=ParamType.P0,
            attr=AttributeRef(2, Channel.Z),
            high_src=False,
        )
        self.interp.execute(inst_lo)
        self.assertEqual(self.rf.read_vgpr_u32(0), 0x00003C00)

        # 2. Move high half to high half of v0 (preserving low half)
        inst_hi = ASTInstruction(
            opcode=Opcode.V_INTERP_MOV_F16,
            dst=ASTOperand(0, high=True),
            param=ParamType.P0,
            attr=AttributeRef(2, Channel.Z),
            high_src=True,
        )
        self.interp.execute(inst_hi)
        self.assertEqual(self.rf.read_vgpr_u32(0), 0x40003C00)

    def test_scalar_interpreter_clamp(self):
        # Pack attr 0 chan X: 3.5 (0x4300)
        u_val = f16_to_u16(3.5)
        self.buf.write_param_channel(0, 0, Channel.X.value, pack_f16_pair(u_val, 0))

        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_MOV_F16,
            dst=ASTOperand(5, high=False),
            param=ParamType.P0,
            attr=AttributeRef(0, Channel.X),
            high_src=False,
            clamp=True,
        )
        self.interp.execute(inst)
        # Clamped to 1.0 (0x3C00)
        self.assertEqual(self.rf.read_vgpr_u32(5), H_ONE)

    def test_scalar_interpreter_deterministic_repeatability(self):
        self.buf.write_param_channel(1, 4, Channel.W.value, 0x12345678)
        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_MOV_F16,
            dst=ASTOperand(10, high=False),
            param=ParamType.P10,
            attr=AttributeRef(4, Channel.W),
            high_src=True,
        )
        self.interp.execute(inst)
        r1 = self.rf.read_vgpr_u32(10)
        self.interp.execute(inst)
        r2 = self.rf.read_vgpr_u32(10)
        self.assertEqual(r1, r2)


class SIMDWavefront:
    """Architectural model of a SIMD wavefront (Wave32 or Wave64) executing v_interp_mov_f16."""

    def __init__(self, wave_size: int = 64, attr_buf: AttributeBuffer | None = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self._exec_mask = (1 << wave_size) - 1
        self.vgprs = [[0] * 256 for _ in range(wave_size)]
        self.attr_buf = attr_buf if attr_buf is not None else AttributeBuffer()

    @property
    def exec_mask(self) -> int:
        return self._exec_mask

    @exec_mask.setter
    def exec_mask(self, val: int) -> None:
        mask_limit = (1 << self.wave_size) - 1
        self._exec_mask = val & mask_limit

    @property
    def exec_lo(self) -> int:
        return self._exec_mask & MASK32

    @exec_lo.setter
    def exec_lo(self, val: int) -> None:
        self._exec_mask = (self._exec_mask & ~MASK32) | (val & MASK32)

    @property
    def exec_hi(self) -> int:
        return (self._exec_mask >> 32) & MASK32

    @exec_hi.setter
    def exec_hi(self, val: int) -> None:
        if self.wave_size == 32:
            raise ValueError("Cannot set exec_hi on Wave32")
        self._exec_mask = (self._exec_mask & MASK32) | ((val & MASK32) << 32)

    def execute(self, inst: ASTInstruction) -> None:
        inst.validate()
        attr_word = self.attr_buf.read_param_channel(
            inst.param.index,
            inst.attr.attr_index,
            inst.attr.channel.value,
        )
        for lane in range(self.wave_size):
            if (self._exec_mask >> lane) & 1:
                old_val = self.vgprs[lane][inst.dst.vgpr_index]
                new_val = ref_interp_mov_f16(
                    attr_word,
                    high_src=inst.high_src,
                    high_dst=inst.dst.high,
                    old_dst_u32=old_val,
                    clamp=inst.clamp,
                )
                self.vgprs[lane][inst.dst.vgpr_index] = new_val


class TestInterpMovF16SIMDMultiLane(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    CANARY_OLD = 0xBEEFCAFE

    def test_wave32_full_exec(self):
        buf = AttributeBuffer()
        buf.write_param_channel(0, 1, Channel.X.value, pack_f16_pair(H_ONE, H_TWO))
        wave = SIMDWavefront(wave_size=32, attr_buf=buf)

        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_MOV_F16,
            dst=ASTOperand(0, high=False),
            param=ParamType.P0,
            attr=AttributeRef(1, Channel.X),
            high_src=False,
        )
        wave.execute(inst)

        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][0], H_ONE)

    def test_wave64_divergence_exec_masking(self):
        buf = AttributeBuffer()
        buf.write_param_channel(0, 3, Channel.Y.value, pack_f16_pair(H_HALF, H_ONE))
        wave = SIMDWavefront(wave_size=64, attr_buf=buf)

        # Alternating even/odd lanes
        wave.exec_lo = 0x55555555
        wave.exec_hi = 0xAAAAAAAA

        for lane in range(64):
            wave.vgprs[lane][4] = self.CANARY_OLD

        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_MOV_F16,
            dst=ASTOperand(4, high=False),
            param=ParamType.P0,
            attr=AttributeRef(3, Channel.Y),
            high_src=True,   # H_ONE (0x3C00)
        )
        wave.execute(inst)

        for lane in range(64):
            is_active = bool((wave.exec_mask >> lane) & 1)
            if is_active:
                # low half becomes H_ONE, high half retains CANARY's high half (0xBEEF)
                expected = (self.CANARY_OLD & 0xFFFF0000) | H_ONE
                self.assertEqual(wave.vgprs[lane][4], expected)
            else:
                self.assertEqual(wave.vgprs[lane][4], self.CANARY_OLD)

    def test_wave64_halves_masking(self):
        buf = AttributeBuffer()
        buf.write_param_channel(0, 0, Channel.X.value, 0x11112222)
        wave = SIMDWavefront(wave_size=64, attr_buf=buf)

        # Lower 32 lanes active, upper 32 inactive
        wave.exec_lo = 0xFFFFFFFF
        wave.exec_hi = 0x00000000

        for lane in range(64):
            wave.vgprs[lane][1] = self.CANARY_OLD

        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_MOV_F16,
            dst=ASTOperand(1, high=True),
            param=ParamType.P0,
            attr=AttributeRef(0, Channel.X),
            high_src=False,
        )
        wave.execute(inst)

        for lane in range(32):
            expected = (0x2222 << 16) | (self.CANARY_OLD & 0xFFFF)
            self.assertEqual(wave.vgprs[lane][1], expected)

        for lane in range(32, 64):
            self.assertEqual(wave.vgprs[lane][1], self.CANARY_OLD)

    def test_single_active_lane(self):
        buf = AttributeBuffer()
        buf.write_param_channel(0, 0, Channel.X.value, pack_f16_pair(H_ONE, H_TWO))
        wave = SIMDWavefront(wave_size=64, attr_buf=buf)
        wave.exec_mask = 1 << 42  # only lane 42 active

        for lane in range(64):
            wave.vgprs[lane][0] = self.CANARY_OLD

        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_MOV_F16,
            dst=ASTOperand(0, high=False),
            param=ParamType.P0,
            attr=AttributeRef(0, Channel.X),
            high_src=False,
        )
        wave.execute(inst)

        for lane in range(64):
            if lane == 42:
                expected = (self.CANARY_OLD & 0xFFFF0000) | H_ONE
                self.assertEqual(wave.vgprs[lane][0], expected)
            else:
                self.assertEqual(wave.vgprs[lane][0], self.CANARY_OLD)

    def test_zero_exec_preserves_all_lanes(self):
        buf = AttributeBuffer()
        buf.write_param_channel(0, 0, Channel.X.value, 0x12345678)
        wave = SIMDWavefront(wave_size=64, attr_buf=buf)
        wave.exec_mask = 0

        for lane in range(64):
            wave.vgprs[lane][0] = self.CANARY_OLD

        inst = ASTInstruction(
            opcode=Opcode.V_INTERP_MOV_F16,
            dst=ASTOperand(0, high=False),
            param=ParamType.P0,
            attr=AttributeRef(0, Channel.X),
            high_src=False,
        )
        wave.execute(inst)

        for lane in range(64):
            self.assertEqual(wave.vgprs[lane][0], self.CANARY_OLD)


if __name__ == "__main__":
    unittest.main()
