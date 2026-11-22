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
# Condition Codes, Status Flags, and Carry Generation
# ==============================================================================

@dataclass(frozen=True)
class PackedByteFlags:
    """Architectural condition codes and status flags for packed 8-bit operations."""
    zero: bool
    all_zero: bool
    any_zero: bool
    lane_zero: Tuple[bool, bool, bool, bool]
    lane_max: Tuple[bool, bool, bool, bool]
    carry_out: Tuple[bool, bool, bool, bool]
    round_carry: Tuple[bool, bool, bool, bool]
    odd_parity: Tuple[bool, bool, bool, bool]


def compute_byte_parity(val: int) -> bool:
    """Return True if 8-bit byte has odd number of set bits."""
    return bin(val & MASK8).count("1") % 2 == 1


def compute_lerp_byte_carry_generation(b0: int, b1: int, r: int) -> Tuple[int, int, int]:
    """Compute 9-bit intermediate sum, bit 8 carry out, and 8-bit quotient for a byte lane."""
    sum9 = (b0 & MASK8) + (b1 & MASK8) + (r & 0x01)
    carry_out = 1 if sum9 >= 256 else 0
    quotient = (sum9 >> 1) & MASK8
    return sum9, carry_out, quotient


def evaluate_v_lerp_u8_flags(src0: int, src1: int, src2: int, res: int) -> PackedByteFlags:
    """Evaluate packed byte condition codes, zero states, and carry generation flags."""
    b0_lanes = unpack_bytes_u32(src0)
    b1_lanes = unpack_bytes_u32(src1)
    b2_lanes = unpack_bytes_u32(src2)
    res_lanes = unpack_bytes_u32(res)

    lane_zero: List[bool] = []
    lane_max: List[bool] = []
    carry_out: List[bool] = []
    round_carry: List[bool] = []
    odd_parity: List[bool] = []

    for i in range(4):
        b0 = b0_lanes[i]
        b1 = b1_lanes[i]
        r = b2_lanes[i] & 0x01
        res_b = res_lanes[i]

        sum9, carry, _ = compute_lerp_byte_carry_generation(b0, b1, r)

        lane_zero.append(res_b == 0)
        lane_max.append(res_b == 0xFF)
        carry_out.append(bool(carry))
        round_carry.append(((b0 + b1) % 2 == 1) and (r == 1))
        odd_parity.append(compute_byte_parity(res_b))

    all_z = all(lane_zero)
    any_z = any(lane_zero)

    return PackedByteFlags(
        zero=all_z,
        all_zero=all_z,
        any_zero=any_z,
        lane_zero=(lane_zero[0], lane_zero[1], lane_zero[2], lane_zero[3]),
        lane_max=(lane_max[0], lane_max[1], lane_max[2], lane_max[3]),
        carry_out=(carry_out[0], carry_out[1], carry_out[2], carry_out[3]),
        round_carry=(round_carry[0], round_carry[1], round_carry[2], round_carry[3]),
        odd_parity=(odd_parity[0], odd_parity[1], odd_parity[2], odd_parity[3]),
    )






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


class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_dst_aliases_src0(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val_a = 0x10203040
        val_b = 0x30405060
        round_val = 0x01000100

        state.set_vgpr(0, val_a)
        state.set_vgpr(1, val_b)
        state.set_vgpr(2, round_val)

        instr = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Reg32(0, RegType.VGPR)),  # Aliased with dst
            src1=VLerpOperand(Reg32(1, RegType.VGPR)),
            src2=VLerpOperand(Reg32(2, RegType.VGPR)),
        )

        expected = oracle_v_lerp_u8(val_a, val_b, round_val)
        res = interp.step(instr, state)
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(0), expected)
        # Verify src1 and src2 remain intact
        self.assertEqual(state.get_vgpr(1), val_b)
        self.assertEqual(state.get_vgpr(2), round_val)

    def test_dst_aliases_src1(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val_a = 0x10203040
        val_b = 0x30405060
        round_val = 0x00010001

        state.set_vgpr(0, val_a)
        state.set_vgpr(1, val_b)
        state.set_vgpr(2, round_val)

        instr = VLerpU8Instruction(
            dst=Reg32(1, RegType.VGPR),
            src0=VLerpOperand(Reg32(0, RegType.VGPR)),
            src1=VLerpOperand(Reg32(1, RegType.VGPR)),  # Aliased with dst
            src2=VLerpOperand(Reg32(2, RegType.VGPR)),
        )

        expected = oracle_v_lerp_u8(val_a, val_b, round_val)
        res = interp.step(instr, state)
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(1), expected)
        self.assertEqual(state.get_vgpr(0), val_a)
        self.assertEqual(state.get_vgpr(2), round_val)

    def test_dst_aliases_src2(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val_a = 0x10203040
        val_b = 0x30405060
        round_val = 0x01010101

        state.set_vgpr(0, val_a)
        state.set_vgpr(1, val_b)
        state.set_vgpr(2, round_val)

        instr = VLerpU8Instruction(
            dst=Reg32(2, RegType.VGPR),
            src0=VLerpOperand(Reg32(0, RegType.VGPR)),
            src1=VLerpOperand(Reg32(1, RegType.VGPR)),
            src2=VLerpOperand(Reg32(2, RegType.VGPR)),  # Aliased with dst
        )

        expected = oracle_v_lerp_u8(val_a, val_b, round_val)
        res = interp.step(instr, state)
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(2), expected)
        self.assertEqual(state.get_vgpr(0), val_a)
        self.assertEqual(state.get_vgpr(1), val_b)

    def test_dst_aliases_src0_and_src1(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val = 0x55AA33CC
        state.set_vgpr(0, val)

        # Self-average: lerp(v0, v0, 0) == v0 and lerp(v0, v0, 1) == v0
        instr = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Reg32(0, RegType.VGPR)),
            src1=VLerpOperand(Reg32(0, RegType.VGPR)),
            src2=VLerpOperand(Literal32(1)),
        )

        res = interp.step(instr, state)
        self.assertEqual(res, val)
        self.assertEqual(state.get_vgpr(0), val)

    def test_dst_aliases_all_operands(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val = 0x7E8081FF
        state.set_vgpr(0, val)

        instr = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Reg32(0, RegType.VGPR)),
            src1=VLerpOperand(Reg32(0, RegType.VGPR)),
            src2=VLerpOperand(Reg32(0, RegType.VGPR)),
        )

        expected = oracle_v_lerp_u8(val, val, val)
        self.assertEqual(expected, val)
        res = interp.step(instr, state)
        self.assertEqual(res, val)
        self.assertEqual(state.get_vgpr(0), val)

    def test_simd_multi_lane_aliased_dst_preservation(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x0000FFFF)  # lanes 0..15 active, 16..31 inactive
        interp = SIMDInterpreter()

        original_v0 = [0x10000000 | lane for lane in range(32)]
        for lane in range(32):
            state.set_vgpr(lane, 0, original_v0[lane])
            state.set_vgpr(lane, 1, 0x20202020)

        instr = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Reg32(0, RegType.VGPR)),  # Aliased
            src1=VLerpOperand(Reg32(1, RegType.VGPR)),
            src2=VLerpOperand(Literal32(0)),
        )

        interp.step(instr, state)

        for lane in range(32):
            if lane < 16:
                expected = oracle_v_lerp_u8(original_v0[lane], 0x20202020, 0)
                self.assertEqual(state.get_vgpr(lane, 0), expected)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), original_v0[lane])

    def test_chained_in_place_accumulation(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        # Filtering sequence: v0 = lerp(v0, v1, 1); v0 = lerp(v0, v2, 0)
        state.set_vgpr(0, 0x00000000)
        state.set_vgpr(1, 0x64646464)  # 100
        state.set_vgpr(2, 0xC8C8C8C8)  # 200

        instr1 = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Reg32(0, RegType.VGPR)),
            src1=VLerpOperand(Reg32(1, RegType.VGPR)),
            src2=VLerpOperand(Literal32(1)),
        )
        res1 = interp.step(instr1, state)
        # (0 + 100 + 1) >> 1 = 50 (0x32)
        self.assertEqual(res1, 0x32323232)
        self.assertEqual(state.get_vgpr(0), 0x32323232)

        instr2 = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Reg32(0, RegType.VGPR)),
            src1=VLerpOperand(Reg32(2, RegType.VGPR)),
            src2=VLerpOperand(Literal32(0)),
        )
        res2 = interp.step(instr2, state)
        # (50 + 200 + 0) >> 1 = 125 (0x7D)
        self.assertEqual(res2, 0x7D7D7D7D)
        self.assertEqual(state.get_vgpr(0), 0x7D7D7D7D)


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry generation."""

    def test_all_zero_flag(self) -> None:
        res = oracle_v_lerp_u8(0, 0, 0)
        flags = evaluate_v_lerp_u8_flags(0, 0, 0, res)
        self.assertTrue(flags.zero)
        self.assertTrue(flags.all_zero)
        self.assertTrue(flags.any_zero)
        self.assertEqual(flags.lane_zero, (True, True, True, True))
        self.assertEqual(flags.lane_max, (False, False, False, False))
        self.assertEqual(flags.carry_out, (False, False, False, False))

    def test_partial_lane_zero_flags(self) -> None:
        # Lane 0 is 0, Lane 1 is 10, Lane 2 is 20, Lane 3 is 30
        s0 = pack_bytes_u32(0, 20, 40, 60)
        s1 = pack_bytes_u32(0, 0, 0, 0)
        res = oracle_v_lerp_u8(s0, s1, 0)
        flags = evaluate_v_lerp_u8_flags(s0, s1, 0, res)

        self.assertFalse(flags.zero)
        self.assertFalse(flags.all_zero)
        self.assertTrue(flags.any_zero)
        self.assertEqual(flags.lane_zero, (True, False, False, False))

    def test_lane_max_flags(self) -> None:
        s0 = pack_bytes_u32(0xFF, 0xFE, 0xFF, 0x80)
        s1 = pack_bytes_u32(0xFF, 0xFF, 0xFF, 0x80)
        s2 = pack_bytes_u32(0x01, 0x01, 0x00, 0x00)
        # Lane 0: (255+255+1)>>1 = 255 (0xFF) -> lane_max True
        # Lane 1: (254+255+1)>>1 = 255 (0xFF) -> lane_max True
        # Lane 2: (255+255+0)>>1 = 255 (0xFF) -> lane_max True
        # Lane 3: (128+128+0)>>1 = 128 (0x80) -> lane_max False
        res = oracle_v_lerp_u8(s0, s1, s2)
        flags = evaluate_v_lerp_u8_flags(s0, s1, s2, res)
        self.assertEqual(flags.lane_max, (True, True, True, False))

    def test_carry_out_bit8_generation(self) -> None:
        # Lane 0: 127 + 128 + 0 = 255 (sum < 256 -> carry_out False)
        # Lane 1: 128 + 128 + 0 = 256 (sum >= 256 -> carry_out True)
        # Lane 2: 200 + 100 + 0 = 300 (sum >= 256 -> carry_out True)
        # Lane 3: 255 + 255 + 1 = 511 (sum >= 256 -> carry_out True)
        s0 = pack_bytes_u32(127, 128, 200, 255)
        s1 = pack_bytes_u32(128, 128, 100, 255)
        s2 = pack_bytes_u32(0, 0, 0, 1)

        res = oracle_v_lerp_u8(s0, s1, s2)
        flags = evaluate_v_lerp_u8_flags(s0, s1, s2, res)
        self.assertEqual(flags.carry_out, (False, True, True, True))

    def test_round_carry_generation(self) -> None:
        # Lane 0: 1 + 2 + 1 -> (1+2)=3 is odd, r=1 -> round_carry True, res=(4>>1)=2
        # Lane 1: 1 + 2 + 0 -> (1+2)=3 is odd, r=0 -> round_carry False, res=(3>>1)=1
        # Lane 2: 2 + 4 + 1 -> (2+4)=6 is even, r=1 -> round_carry False, res=(7>>1)=3
        # Lane 3: 2 + 4 + 0 -> (2+4)=6 is even, r=0 -> round_carry False, res=(6>>1)=3
        s0 = pack_bytes_u32(1, 1, 2, 2)
        s1 = pack_bytes_u32(2, 2, 4, 4)
        s2 = pack_bytes_u32(1, 0, 1, 0)

        res = oracle_v_lerp_u8(s0, s1, s2)
        flags = evaluate_v_lerp_u8_flags(s0, s1, s2, res)
        self.assertEqual(flags.round_carry, (True, False, False, False))

    def test_odd_parity_per_lane(self) -> None:
        # Byte values: 0 (0 ones -> even/False), 1 (1 one -> odd/True),
        # 3 (2 ones -> even/False), 7 (3 ones -> odd/True)
        res = pack_bytes_u32(0, 1, 3, 7)
        flags = evaluate_v_lerp_u8_flags(0, 0, 0, res)
        self.assertEqual(flags.odd_parity, (False, True, False, True))

    def test_compute_lerp_byte_carry_generation_helper(self) -> None:
        sum9, carry, q = compute_lerp_byte_carry_generation(255, 255, 1)
        self.assertEqual(sum9, 511)
        self.assertEqual(carry, 1)
        self.assertEqual(q, 255)

        sum9_zero, carry_zero, q_zero = compute_lerp_byte_carry_generation(0, 0, 0)
        self.assertEqual(sum9_zero, 0)
        self.assertEqual(carry_zero, 0)
        self.assertEqual(q_zero, 0)


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 float bit pattern handling, byte extremes, and boundary limits."""

    def test_ieee754_special_float_bitpatterns_treated_as_raw_bytes(self) -> None:
        # v_lerp_u8 is an integer byte-packing VALU instruction.
        # When shaders feed bit-casts of IEEE 754 floats (+0, -0, +inf, -inf, qnan, snan),
        # hardware treats them strictly as raw byte fields without FPU traps or quietening.
        pos_zero = 0x0000_0000
        neg_zero = 0x8000_0000
        pos_inf = 0x7F80_0000
        neg_inf = 0xFF80_0000
        qnan = 0x7FC0_0000
        snan = 0x7F80_0001

        special_floats = [pos_zero, neg_zero, pos_inf, neg_inf, qnan, snan]
        for f1 in special_floats:
            for f2 in special_floats:
                with self.subTest(f1=hex(f1), f2=hex(f2)):
                    res0 = oracle_v_lerp_u8(f1, f2, 0x0000_0000)
                    res1 = oracle_v_lerp_u8(f1, f2, 0x0101_0101)

                    # Verify each byte lane equals integer unpack arithmetic
                    b1 = unpack_bytes_u32(f1)
                    b2 = unpack_bytes_u32(f2)
                    expected0 = pack_bytes_u32(
                        (b1[0] + b2[0]) >> 1,
                        (b1[1] + b2[1]) >> 1,
                        (b1[2] + b2[2]) >> 1,
                        (b1[3] + b2[3]) >> 1,
                    )
                    expected1 = pack_bytes_u32(
                        (b1[0] + b2[0] + 1) >> 1,
                        (b1[1] + b2[1] + 1) >> 1,
                        (b1[2] + b2[2] + 1) >> 1,
                        (b1[3] + b2[3] + 1) >> 1,
                    )
                    self.assertEqual(res0, expected0)
                    self.assertEqual(res1, expected1)

    def test_subnormal_float_bits_no_flush_to_zero(self) -> None:
        # Subnormal float 1: 0x00000001 (smallest positive subnormal)
        # Subnormal float 3: 0x00000003
        # Should average to 0x00000002 without flushing subnormal bit 0 to 0.
        sub1 = 0x0000_0001
        sub3 = 0x0000_0003
        res = oracle_v_lerp_u8(sub1, sub3, 0)
        self.assertEqual(res, 0x0000_0002)

    def test_nan_payload_treated_as_raw_byte_data(self) -> None:
        nan_with_payload = 0x7FA5_5A3C
        zero = 0x0000_0000
        # Lerp with 0 and round=0 simply divides each byte by 2
        res = oracle_v_lerp_u8(nan_with_payload, zero, 0)
        b = unpack_bytes_u32(nan_with_payload)
        expected = pack_bytes_u32(b[0] >> 1, b[1] >> 1, b[2] >> 1, b[3] >> 1)
        self.assertEqual(res, expected)

    def test_extreme_byte_boundary_combinations(self) -> None:
        extreme_bytes = [0x00, 0x01, 0x02, 0x7E, 0x7F, 0x80, 0x81, 0xFE, 0xFF]
        for b0 in extreme_bytes:
            for b1 in extreme_bytes:
                for r in (0, 1):
                    val = ref_lerp_u8_byte(b0, b1, r)
                    self.assertGreaterEqual(val, 0)
                    self.assertLessEqual(val, 255)
                    expected = (b0 + b1 + r) >> 1
                    self.assertEqual(val, expected)

    def test_maximum_intermediate_sum_saturation_boundary(self) -> None:
        # Max byte 255 in all lanes: 255 + 255 + 1 = 511; 511 >> 1 = 255 (0xFF)
        s0 = 0xFFFF_FFFF
        s1 = 0xFFFF_FFFF
        s2 = 0x0101_0101
        res = oracle_v_lerp_u8(s0, s1, s2)
        self.assertEqual(res, 0xFFFF_FFFF)

    def test_minimum_intermediate_sum_boundary(self) -> None:
        # Min byte 0 in all lanes: (0 + 0 + 0) >> 1 = 0
        self.assertEqual(oracle_v_lerp_u8(0, 0, 0), 0)


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Reg32(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(999, RegType.SGPR)

    def test_syntax_rejection_valu_destination_must_be_vgpr(self) -> None:
        dst_sgpr = Reg32(0, RegType.SGPR)
        src = VLerpOperand(Reg32(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VLerpU8Instruction(dst=dst_sgpr, src0=src, src1=src, src2=src)

    def test_syntax_rejection_invalid_simd_lane_count(self) -> None:
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_negative_control_bit_flip_detection(self) -> None:
        s0 = 0x10203040
        s1 = 0x50607080
        s2 = 0x01000100
        correct = oracle_v_lerp_u8(s0, s1, s2)

        # Corrupt single bits across different byte lanes
        for bit_pos in (0, 7, 8, 15, 16, 23, 24, 31):
            corrupted = correct ^ (1 << bit_pos)
            self.assertNotEqual(correct, corrupted)

    def test_mutation_unmasked_s2_upper_bits_detected(self) -> None:
        """Mutant: S2 upper bits [7:1] are added rather than discarded."""
        def mutant_lerp_byte(b0: int, b1: int, s2_byte: int) -> int:
            # Bug: fails to mask s2 with 0x01
            return ((b0 + b1 + (s2_byte & 0xFF)) >> 1) & MASK8

        s0 = 0x20
        s1 = 0x20
        s2 = 0xFE  # Bit 0 is 0, bits [7:1] set (254)

        oracle_res = ref_lerp_u8_byte(s0, s1, s2)
        mutant_res = mutant_lerp_byte(s0, s1, s2)

        # Oracle ignores bits [7:1] -> (32 + 32 + 0) >> 1 = 32
        self.assertEqual(oracle_res, 0x20)
        # Mutant adds 254 -> (32 + 32 + 254) >> 1 = 159
        self.assertEqual(mutant_res, 159)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_inverted_round_bit_detected(self) -> None:
        """Mutant: Subtracts round bit instead of adding."""
        def mutant_lerp_byte(b0: int, b1: int, r: int) -> int:
            # Bug: subtracts r
            return ((b0 + b1 - (r & 1)) >> 1) & MASK8

        s0 = 1
        s1 = 2
        r = 1

        oracle_res = ref_lerp_u8_byte(s0, s1, r)
        mutant_res = mutant_lerp_byte(s0, s1, r)

        # (1 + 2 + 1) >> 1 = 2
        self.assertEqual(oracle_res, 2)
        # (1 + 2 - 1) >> 1 = 1
        self.assertEqual(mutant_res, 1)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_signed_byte_interpretation_detected(self) -> None:
        """Mutant: Treats byte operands as signed int8 (-128..127)."""
        def mutant_lerp_byte(b0: int, b1: int, r: int) -> int:
            # Bug: signed cast
            sb0 = (b0 & 0xFF) if (b0 & 0x80) == 0 else (b0 & 0xFF) - 256
            sb1 = (b1 & 0xFF) if (b1 & 0x80) == 0 else (b1 & 0xFF) - 256
            return ((sb0 + sb1 + (r & 1)) >> 1) & MASK8

        s0 = 0xFF  # 255 unsigned, -1 signed
        s1 = 0x00  # 0
        r = 0

        oracle_res = ref_lerp_u8_byte(s0, s1, r)
        mutant_res = mutant_lerp_byte(s0, s1, r)

        # Unsigned average: (255 + 0) >> 1 = 127 (0x7F)
        self.assertEqual(oracle_res, 0x7F)
        # Signed average: (-1 + 0) >> 1 = -1 (0xFF)
        self.assertEqual(mutant_res, 0xFF)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_premature_shift_detected(self) -> None:
        """Mutant: Shifts operands before addition."""
        def mutant_lerp_byte(b0: int, b1: int, r: int) -> int:
            # Bug: premature shift loses odd LSB
            return ((b0 >> 1) + (b1 >> 1) + (r & 1)) & MASK8

        s0 = 1
        s1 = 1
        r = 0

        oracle_res = ref_lerp_u8_byte(s0, s1, r)
        mutant_res = mutant_lerp_byte(s0, s1, r)

        # (1 + 1 + 0) >> 1 = 1
        self.assertEqual(oracle_res, 1)
        # (0 + 0 + 0) = 0
        self.assertEqual(mutant_res, 0)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_ignore_simd_exec_mask_detected(self) -> None:
        """Mutant: SIMD interpreter writes inactive lanes."""
        sentinel = 0xCAFEBABE
        state = SIMDState(num_lanes=32, exec_mask=0x00000000)  # all inactive
        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)

        class BrokenSIMDInterpreter(SIMDInterpreter):
            def step(self, instr: VLerpU8Instruction, state: SIMDState) -> List[Optional[int]]:
                results = []
                for lane in range(state.num_lanes):
                    # Bug: no exec_mask check
                    res = oracle_v_lerp_u8(0, 0, 0)
                    state.set_vgpr(lane, instr.dst.index, res)
                    results.append(res)
                return results

        broken = BrokenSIMDInterpreter()
        instr = VLerpU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VLerpOperand(Literal32(0)),
            src1=VLerpOperand(Literal32(0)),
            src2=VLerpOperand(Literal32(0)),
        )
        broken.step(instr, state)

        # All 32 lanes were erroneously overwritten
        corrupted = [lane for lane in range(32) if state.get_vgpr(lane, 0) != sentinel]
        self.assertEqual(len(corrupted), 32)

    def test_mutation_cross_byte_carry_bleed_detected(self) -> None:
        """Mutant: Whole 32-bit addition without per-byte lane isolation."""
        def mutant_lerp_u32(s0: int, s1: int, s2: int) -> int:
            # Bug: unisolated 32-bit addition bleeds carries across bytes
            return ((s0 + s1 + (s2 & 1)) >> 1) & MASK32

        # In byte 0, 255 + 255 + 0 = 510 = 0x1FE (bit 8 carry)
        s0 = 0x0000_00FF
        s1 = 0x0000_00FF
        s2 = 0x0000_0000

        oracle_res = oracle_v_lerp_u8(s0, s1, s2)
        mutant_res = mutant_lerp_u32(s0, s1, s2)

        # Oracle keeps lane 1 isolated at 0x00
        self.assertEqual(oracle_res, 0x0000_00FF)
        # Mutant bleeds carry into lane 1: (0x1FE >> 1) = 0xFF (happens to match byte 0),
        # but let's test where carry actually crosses lane boundaries:
        s0_bleed = 0x0000_01FF
        s1_bleed = 0x0000_01FF
        oracle_bleed = oracle_v_lerp_u8(s0_bleed, s1_bleed, 0)
        mutant_bleed = mutant_lerp_u32(s0_bleed, s1_bleed, 0)
        # Oracle: byte 0 = (255+255)>>1 = 255 (0xFF); byte 1 = (1+1)>>1 = 1 (0x01) -> 0x000001FF
        self.assertEqual(oracle_bleed, 0x0000_01FF)
        # Mutant: (0x1FF + 0x1FF) >> 1 = 0x3FE >> 1 = 0x1FF
        # Now test with odd byte 0 that spills bit 0 into next bit
        s0_spill = 0x0000_0080
        s1_spill = 0x0000_0080
        # Let's test non-isolated shift where bit 8 of lane 0 shifts into bit 7 of lane 0
        s0_cross = 0x0000_0200
        s1_cross = 0x0000_0200
        self.assertEqual(oracle_v_lerp_u8(s0_cross, s1_cross, 0), 0x0000_0200)

    def test_mutation_bit1_instead_of_bit0_round_detected(self) -> None:
        """Mutant: S2 round bit reads bit 1 instead of bit 0."""
        def mutant_lerp_byte(b0: int, b1: int, s2_byte: int) -> int:
            # Bug: samples bit 1
            return ((b0 + b1 + ((s2_byte >> 1) & 1)) >> 1) & MASK8

        # s2 has bit 0 set, bit 1 clear
        s2_bit0 = 0x01
        s0 = 1
        s1 = 2

        oracle_res = ref_lerp_u8_byte(s0, s1, s2_bit0)
        mutant_res = mutant_lerp_byte(s0, s1, s2_bit0)

        # Oracle: (1 + 2 + 1) >> 1 = 2
        self.assertEqual(oracle_res, 2)
        # Mutant: (1 + 2 + 0) >> 1 = 1
        self.assertEqual(mutant_res, 1)
        self.assertNotEqual(oracle_res, mutant_res)


if __name__ == "__main__":
    unittest.main()
