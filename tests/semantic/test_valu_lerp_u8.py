from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_lerp_u8.

AMD RDNA VALU Instruction Set Architecture:
  v_lerp_u8 dst, src0, src1, src2

Operation:
  Performs parallel byte-wise linear interpolation (rounded unsigned average)
  across four packed 8-bit unsigned byte lanes within 32-bit dwords:
    For each byte lane i in {0, 1, 2, 3}:
      s0_i = (src0 >> (8 * i)) & 0xFF
      s1_i = (src1 >> (8 * i)) & 0xFF
      s2_i = (src2 >> (8 * i)) & 0x01   # only bit 0 of each byte lane contributes
      sum_i = s0_i + s1_i + s2_i        # range: 0 .. 511 (9-bit intermediate)
      dst_i = (sum_i >> 1) & 0xFF       # range: 0 .. 255 (8-bit bounded result)

Mathematical and Architectural Properties:
  - Range isolation: max sum 255 + 255 + 1 = 511; 511 >> 1 = 255 (0xFF).
    Result is strictly bound within [0, 255] for every byte lane. No cross-byte
    carry or overflow can occur.
  - Rounding control:
    - s2 bit 0 = 0: floor / truncation of the unsigned average.
    - s2 bit 0 = 1: ceiling / round-half-up of the unsigned average.
  - Identity property:
    - When s0_i == s1_i == k, ((k + k + r) >> 1) == (2k + r) >> 1 == k for both r in {0, 1}.
  - Symmetry / Commutativity:
    - src0 and src1 are interchangeable: v_lerp_u8(a, b, c) == v_lerp_u8(b, a, c).
  - Upper bit discard:
    - Bits [7:1] of each byte lane in src2 are strictly ignored by hardware and translator.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union
import unittest

# Architectural bitmasks and constants
MASK8: int = 0xFF
MASK32: int = 0xFFFF_FFFF
BYTE_SHIFTS: Tuple[int, int, int, int] = (0, 8, 16, 24)


def unpack_bytes_u32(val: int) -> Tuple[int, int, int, int]:
    """Unpack a 32-bit unsigned word into 4 byte lanes (b0, b1, b2, b3)."""
    val &= MASK32
    return (
        val & MASK8,
        (val >> 8) & MASK8,
        (val >> 16) & MASK8,
        (val >> 24) & MASK8,
    )


def pack_bytes_u32(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four 8-bit unsigned integers into a 32-bit unsigned word."""
    return (
        (b0 & MASK8)
        | ((b1 & MASK8) << 8)
        | ((b2 & MASK8) << 16)
        | ((b3 & MASK8) << 24)
    ) & MASK32


def ref_lerp_u8_byte(s0: int, s1: int, s2_round: int) -> int:
    """Compute single byte-lane linear interpolation: ((s0 + s1 + (s2 & 1)) >> 1) & 0xFF.

    The intermediate sum ranges from 0 to 511. Right shifting by 1 produces
    a value in [0, 255] with strict lane isolation and no carry propagation.
    """
    b0 = s0 & MASK8
    b1 = s1 & MASK8
    r = s2_round & 0x01
    return ((b0 + b1 + r) >> 1) & MASK8


def oracle_v_lerp_u8(src0: int, src1: int, src2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_lerp_u8 32-bit packed word interpolation."""
    b0_0, b0_1, b0_2, b0_3 = unpack_bytes_u32(src0)
    b1_0, b1_1, b1_2, b1_3 = unpack_bytes_u32(src1)
    b2_0, b2_1, b2_2, b2_3 = unpack_bytes_u32(src2)

    r0 = ref_lerp_u8_byte(b0_0, b1_0, b2_0)
    r1 = ref_lerp_u8_byte(b0_1, b1_1, b2_1)
    r2 = ref_lerp_u8_byte(b0_2, b1_2, b2_2)
    r3 = ref_lerp_u8_byte(b0_3, b1_3, b2_3)

    return pack_bytes_u32(r0, r1, r2, r3)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Reg32:
    """Represents a 32-bit architectural register (VGPR or SGPR)."""
    index: int
    reg_type: RegType

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index {self.index} out of range [0, 255]")

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class Literal32:
    """Represents a 32-bit integer literal or inline constant."""
    value: int

    def __post_init__(self) -> None:
        if not (0 <= self.value <= MASK32):
            raise ValueError(f"Literal32 value {self.value:#x} out of range [0, 0xFFFFFFFF]")

    @property
    def name(self) -> str:
        return f"{self.value:#010x}" if self.value > 9 else str(self.value)


@dataclass(frozen=True)
class VLerpOperand:
    """Typed expression operand representing a register or literal."""
    source: Union[Reg32, Literal32]

    def disassemble(self) -> str:
        return self.source.name


@dataclass(frozen=True)
class VLerpU8Instruction:
    """Symbolic AST node representing an RDNA2 v_lerp_u8 VALU instruction."""
    dst: Reg32
    src0: VLerpOperand
    src1: VLerpOperand
    src2: VLerpOperand

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        return f"v_lerp_u8 {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single execution thread."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def get_vgpr(self, index: int) -> int:
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        return self.vgprs[index] & MASK32

    def set_vgpr(self, index: int, val: int) -> None:
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        self.vgprs[index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        self.sgprs[index] = val & MASK32


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_lerp_u8."""

    def resolve_operand(self, op: VLerpOperand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg32):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.value & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VLerpU8Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0 = self.resolve_operand(instr.src0, state)
        s1 = self.resolve_operand(instr.src1, state)
        s2 = self.resolve_operand(instr.src2, state)

        # Computation phase
        result = oracle_v_lerp_u8(s0, s1, s2)

        # Write-back phase
        state.set_vgpr(instr.dst.index, result)
        return result


# ==============================================================================
# SIMD Multi-Lane Execution Engine and EXEC Masking
# ==============================================================================

@dataclass
class SIMDState:
    """Multi-lane SIMD machine state representing Wave32 or Wave64."""
    num_lanes: int
    exec_mask: Optional[int] = None
    vgprs: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {self.num_lanes}")
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        if self.exec_mask is None:
            self.exec_mask = (1 << self.num_lanes) - 1

    def get_vgpr(self, lane: int, index: int) -> int:
        if not (0 <= lane < self.num_lanes):
            raise IndexError(f"Lane {lane} out of range [0, {self.num_lanes - 1}]")
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        return self.vgprs[lane][index] & MASK32

    def set_vgpr(self, lane: int, index: int, val: int) -> None:
        if not (0 <= lane < self.num_lanes):
            raise IndexError(f"Lane {lane} out of range [0, {self.num_lanes - 1}]")
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        self.vgprs[lane][index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        self.sgprs[index] = val & MASK32


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VLerpOperand, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, Reg32):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(lane, op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.value & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VLerpU8Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * state.num_lanes
        # Atomic read phase across all lanes
        s0_vals = [
            self.resolve_lane_operand(instr.src0, state, lane)
            for lane in range(state.num_lanes)
        ]
        s1_vals = [
            self.resolve_lane_operand(instr.src1, state, lane)
            for lane in range(state.num_lanes)
        ]
        s2_vals = [
            self.resolve_lane_operand(instr.src2, state, lane)
            for lane in range(state.num_lanes)
        ]

        # Execution and selective write-back phase
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = oracle_v_lerp_u8(s0_vals[lane], s1_vals[lane], s2_vals[lane])
                state.set_vgpr(lane, instr.dst.index, res)
                results[lane] = res

        return results





# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestVLerpU8ModelAndOracle(unittest.TestCase):
    """Unit tests for RDNA2 v_lerp_u8 architectural bit model and reference oracle."""

    def test_byte_unpack_pack_roundtrip(self) -> None:
        patterns = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0xA55A3C7E,
            0x01020304,
            0x80402010,
        ]
        for pat in patterns:
            with self.subTest(pattern=hex(pat)):
                b0, b1, b2, b3 = unpack_bytes_u32(pat)
                repacked = pack_bytes_u32(b0, b1, b2, b3)
                self.assertEqual(repacked, pat)

    def test_reference_known_vectors(self) -> None:
        # 0xFF and 0xFF with round=1 -> 0xFF
        self.assertEqual(oracle_v_lerp_u8(0xFF, 0xFF, 1), 0xFF)
        # 0x01 and 0x02 with round=0 -> 1 ((1+2+0)>>1 == 1)
        self.assertEqual(oracle_v_lerp_u8(0x01, 0x02, 0), 0x01)
        # 0x01 and 0x02 with round=1 -> 2 ((1+2+1)>>1 == 2)
        self.assertEqual(oracle_v_lerp_u8(0x01, 0x02, 1), 0x02)
        # Multi-lane distinct bytes
        s0 = 0x10203040
        s1 = 0x30405060
        # lane 0: (0x40+0x60)>>1 = (64+96)>>1 = 80 = 0x50
        # lane 1: (0x30+0x50)>>1 = (48+80)>>1 = 64 = 0x40
        # lane 2: (0x20+0x40)>>1 = (32+64)>>1 = 48 = 0x30
        # lane 3: (0x10+0x30)>>1 = (16+48)>>1 = 32 = 0x20
        self.assertEqual(oracle_v_lerp_u8(s0, s1, 0), 0x20304050)

    def test_rounding_control(self) -> None:
        # Odd sum: 3 + 4 = 7. Round 0 -> 3; Round 1 -> 4
        s0 = 0x03030303
        s1 = 0x04040404
        self.assertEqual(oracle_v_lerp_u8(s0, s1, 0x00000000), 0x03030303)
        self.assertEqual(oracle_v_lerp_u8(s0, s1, 0x01010101), 0x04040404)

        # Mixed round bits per lane
        s2 = pack_bytes_u32(0, 1, 0, 1)
        expected = pack_bytes_u32(3, 4, 3, 4)
        self.assertEqual(oracle_v_lerp_u8(s0, s1, s2), expected)

    def test_s2_upper_bits_discarded(self) -> None:
        s0 = 0x10203040
        s1 = 0x10203040
        # s2 has bits [7:1] set (0xFE in every lane), but bit 0 is 0
        s2_upper = 0xFEFEFEFE
        self.assertEqual(oracle_v_lerp_u8(s0, s1, s2_upper), s0)

        # s2 has bits [7:0] set (0xFF in every lane), bit 0 is 1
        s2_all = 0xFFFFFFFF
        self.assertEqual(oracle_v_lerp_u8(s0, s1, s2_all), oracle_v_lerp_u8(s0, s1, 0x01010101))

    def test_commutativity_src0_src1(self) -> None:
        pairs = [
            (0x00000000, 0xFFFFFFFF, 0x01010101),
            (0x12345678, 0x9ABCDEF0, 0x00000000),
            (0x55555555, 0xAAAAAAAA, 0xFFFFFFFF),
            (0xFEEDBEEF, 0x01020304, 0x01000100),
        ]
        for s0, s1, s2 in pairs:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                self.assertEqual(oracle_v_lerp_u8(s0, s1, s2), oracle_v_lerp_u8(s1, s0, s2))

    def test_identity_property(self) -> None:
        for val in [0x00000000, 0x7F7F7F7F, 0x80808080, 0xFFFFFFFF, 0x12345678]:
            with self.subTest(val=hex(val)):
                self.assertEqual(oracle_v_lerp_u8(val, val, 0x00000000), val)
                self.assertEqual(oracle_v_lerp_u8(val, val, 0x01010101), val)

    def test_no_cross_lane_bleed(self) -> None:
        # Lane 0 reaches max sum 511 (255 + 255 + 1). Does it bleed into Lane 1?
        s0 = 0x000000FF
        s1 = 0x000000FF
        s2 = 0x00000001
        res = oracle_v_lerp_u8(s0, s1, s2)
        self.assertEqual(res, 0x000000FF)
        b0, b1, b2, b3 = unpack_bytes_u32(res)
        self.assertEqual(b0, 0xFF)
        self.assertEqual(b1, 0x00)
        self.assertEqual(b2, 0x00)
        self.assertEqual(b3, 0x00)


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_validation_and_naming(self) -> None:
        v0 = Reg32(0, RegType.VGPR)
        v255 = Reg32(255, RegType.VGPR)
        s12 = Reg32(12, RegType.SGPR)
        self.assertEqual(v0.name, "v0")
        self.assertEqual(v255.name, "v255")
        self.assertEqual(s12.name, "s12")

    def test_register_bounds_rejection(self) -> None:
        with self.assertRaises(ValueError):
            Reg32(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(1000, RegType.SGPR)

    def test_literal_values_and_naming(self) -> None:
        lit_small = Literal32(1)
        lit_hex = Literal32(0x10203040)
        self.assertEqual(lit_small.name, "1")
        self.assertEqual(lit_hex.name, "0x10203040")

        with self.assertRaises(ValueError):
            Literal32(-1)
        with self.assertRaises(ValueError):
            Literal32(0x1_0000_0000)

    def test_instruction_ast_construction_and_disassembly(self) -> None:
        dst = Reg32(0, RegType.VGPR)
        src0 = VLerpOperand(Reg32(1, RegType.VGPR))
        src1 = VLerpOperand(Reg32(4, RegType.SGPR))
        src2 = VLerpOperand(Literal32(1))

        instr = VLerpU8Instruction(dst=dst, src0=src0, src1=src1, src2=src2)
        self.assertEqual(instr.disassemble(), "v_lerp_u8 v0, v1, s4, 1")

    def test_destination_must_be_vgpr(self) -> None:
        dst_sgpr = Reg32(0, RegType.SGPR)
        src = VLerpOperand(Reg32(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VLerpU8Instruction(dst=dst_sgpr, src0=src, src1=src, src2=src)


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()
        self.interp = ScalarInterpreter()

    def test_scalar_state_get_set(self) -> None:
        self.assertEqual(self.state.get_vgpr(0), 0)
        self.assertEqual(self.state.get_sgpr(0), 0)

        self.state.set_vgpr(10, 0x12345678)
        self.assertEqual(self.state.get_vgpr(10), 0x12345678)

        self.state.set_sgpr(5, 0xABCDEF01)
        self.assertEqual(self.state.get_sgpr(5), 0xABCDEF01)

        # 32-bit masking
        self.state.set_vgpr(1, 0x1_0000_0042)
        self.assertEqual(self.state.get_vgpr(1), 0x42)

        with self.assertRaises(IndexError):
            self.state.get_vgpr(-1)
        with self.assertRaises(IndexError):
            self.state.get_vgpr(256)

    def test_scalar_step_vgpr_sources(self) -> None:
        self.state.set_vgpr(1, 0x10203040)
        self.state.set_vgpr(2, 0x30405060)
        self.state.set_vgpr(3, 0x01000100)

        instr = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Reg32(1, RegType.VGPR)),
            src1=VLerpOperand(Reg32(2, RegType.VGPR)),
            src2=VLerpOperand(Reg32(3, RegType.VGPR)),
        )

        res = self.interp.step(instr, self.state)
        expected = oracle_v_lerp_u8(0x10203040, 0x30405060, 0x01000100)
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_vgpr(0), expected)

    def test_scalar_step_mixed_sources_sgpr_and_literal(self) -> None:
        self.state.set_vgpr(1, 0x02040608)
        self.state.set_sgpr(4, 0x0406080A)

        instr = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Reg32(1, RegType.VGPR)),
            src1=VLerpOperand(Reg32(4, RegType.SGPR)),
            src2=VLerpOperand(Literal32(1)),
        )

        res = self.interp.step(instr, self.state)
        expected = oracle_v_lerp_u8(0x02040608, 0x0406080A, 1)
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_vgpr(0), expected)

    def test_sequential_instruction_flow(self) -> None:
        # v0 = lerp(0x10101010, 0x20202020, 0) -> 0x18181818
        # v1 = lerp(v0, 0x30303030, 1) -> lerp(0x18181818, 0x30303030, 1)
        self.state.set_vgpr(2, 0x10101010)
        self.state.set_vgpr(3, 0x20202020)
        self.state.set_vgpr(4, 0x30303030)

        instr1 = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Reg32(2, RegType.VGPR)),
            src1=VLerpOperand(Reg32(3, RegType.VGPR)),
            src2=VLerpOperand(Literal32(0)),
        )
        res1 = self.interp.step(instr1, self.state)
        self.assertEqual(res1, 0x18181818)

        instr2 = VLerpU8Instruction(
            dst=Reg32(1, RegType.VGPR),
            src0=VLerpOperand(Reg32(0, RegType.VGPR)),
            src1=VLerpOperand(Reg32(4, RegType.VGPR)),
            src2=VLerpOperand(Literal32(1)),
        )
        res2 = self.interp.step(instr2, self.state)
        # (0x18 + 0x30 + 1) >> 1 = (24 + 48 + 1) >> 1 = 73 >> 1 = 36 = 0x24
        self.assertEqual(res2, 0x24242424)
        self.assertEqual(self.state.get_vgpr(1), 0x24242424)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        for lane in range(32):
            s0_val = pack_bytes_u32(lane * 2, lane * 3, lane * 4, lane * 5)
            s1_val = pack_bytes_u32(10, 20, 30, 40)
            s2_val = pack_bytes_u32(lane & 1, (lane >> 1) & 1, 0, 1)
            state.set_vgpr(lane, 1, s0_val)
            state.set_vgpr(lane, 2, s1_val)
            state.set_vgpr(lane, 3, s2_val)

        instr = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Reg32(1, RegType.VGPR)),
            src1=VLerpOperand(Reg32(2, RegType.VGPR)),
            src2=VLerpOperand(Reg32(3, RegType.VGPR)),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            expected = oracle_v_lerp_u8(
                state.get_vgpr(lane, 1),
                state.get_vgpr(lane, 2),
                state.get_vgpr(lane, 3),
            )
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_all_lanes_active(self) -> None:
        state = SIMDState(num_lanes=64)
        interp = SIMDInterpreter()

        for lane in range(64):
            state.set_vgpr(lane, 1, pack_bytes_u32(lane, lane ^ 0x55, 100, 200))
            state.set_vgpr(lane, 2, pack_bytes_u32(50, 60, 70, 80))

        instr = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Reg32(1, RegType.VGPR)),
            src1=VLerpOperand(Reg32(2, RegType.VGPR)),
            src2=VLerpOperand(Literal32(1)),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            expected = oracle_v_lerp_u8(
                state.get_vgpr(lane, 1),
                state.get_vgpr(lane, 2),
                1,
            )
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_partial_exec_mask_even_lanes(self) -> None:
        sentinel = 0xDEADBEEF
        state = SIMDState(num_lanes=32, exec_mask=0x55555555)  # only even lanes active
        interp = SIMDInterpreter()

        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, 0x10203040)
            state.set_vgpr(lane, 2, 0x50607080)

        instr = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Reg32(1, RegType.VGPR)),
            src1=VLerpOperand(Reg32(2, RegType.VGPR)),
            src2=VLerpOperand(Literal32(0)),
        )

        results = interp.step(instr, state)
        expected = oracle_v_lerp_u8(0x10203040, 0x50607080, 0)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], expected)
                self.assertEqual(state.get_vgpr(lane, 0), expected)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_zero_exec_mask_no_modification(self) -> None:
        sentinel = 0xCAFEBABE
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)

        instr = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Literal32(0xFFFFFFFF)),
            src1=VLerpOperand(Literal32(0xFFFFFFFF)),
            src2=VLerpOperand(Literal32(1)),
        )

        results = interp.step(instr, state)
        self.assertEqual(results, [None] * 32)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_sgpr_scalar_broadcast_across_divergent_lanes(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        state.set_sgpr(5, 0x20202020)  # broadcast to all lanes
        for lane in range(32):
            state.set_vgpr(lane, 1, pack_bytes_u32(lane, lane, lane, lane))

        instr = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Reg32(1, RegType.VGPR)),
            src1=VLerpOperand(Reg32(5, RegType.SGPR)),
            src2=VLerpOperand(Literal32(1)),
        )

        results = interp.step(instr, state)
        for lane in range(32):
            expected = oracle_v_lerp_u8(
                pack_bytes_u32(lane, lane, lane, lane),
                0x20202020,
                1,
            )
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)


if __name__ == "__main__":
    unittest.main()
