from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_sub_src2_u32.

Opcode:
  - ds_sub_src2_u32: Atomic 32-bit subtraction of LDS secondary operand LDS[B] from LDS[A].
      tmp_a = LDS[A]
      tmp_b = LDS[B]
      LDS[A] = (tmp_a - tmp_b) & 0xFFFFFFFF

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing:
      Base address in bytes from VGPR vaddr.
      offset0 (8-bit immediate) and offset1 (8-bit immediate) specify address offsets in dwords (offset * 4).
      Address A = (vaddr + offset0 * 4).
      Address B = (vaddr + offset1 * 4).
      Dword indices are calculated by dropping low 2 bits and wrapping to LDS capacity.
  - Non-returning: Vector general-purpose registers (VGPRs) are not modified.
  - Modulo arithmetic: Subtraction wraps strictly modulo 2^32 (e.g. 0 - 1 = 0xFFFFFFFF).
"""

from dataclasses import dataclass
from enum import Enum
import math
import struct
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
DEFAULT_LDS_DWORDS = 16384  # 64 KiB default capacity in dwords
DEFAULT_NUM_LANES = 64
DEFAULT_NUM_VGPRS = 256
DEFAULT_NUM_SGPRS = 106

# IEEE 754 Single-Precision Boundary Constants
POS_ZERO_F32 = 0x00000000
NEG_ZERO_F32 = 0x80000000
POS_INF_F32 = 0x7F800000
NEG_INF_F32 = 0xFF800000
QNAN_F32 = 0x7FC00000
SNAN_F32 = 0x7F800001
MIN_NORMAL_F32 = 0x00800000
MIN_SUBNORMAL_F32 = 0x00000001
MAX_SUBNORMAL_F32 = 0x007FFFFF
MAX_NORMAL_F32 = 0x7F7FFFFF
ONE_F32 = 0x3F800000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def bits_to_f32(bits: int) -> float:
    """Interpret 32-bit unsigned integer as IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Pack IEEE 754 single-precision float into 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def is_f32_nan(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 NaN."""
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def is_f32_subnormal(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 subnormal float."""
    return (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0


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


class DSOpcode(Enum):
    """Supported DS src2 family opcodes."""
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


class LDSMemory:
    """Model of Local Data Share memory storage."""

    def __init__(self, size_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_dwords}")
        self.size_dwords = size_dwords
        self.data = [0] * size_dwords

    def read_dword(self, index: int) -> int:
        return self.data[index & (self.size_dwords - 1)]

    def write_dword(self, index: int, val: int) -> None:
        self.data[index & (self.size_dwords - 1)] = u32(val)


class ScalarMachineState:
    """Scalar architectural machine state for single-lane emulation."""

    def __init__(self, lds_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        self.vgprs = [0] * DEFAULT_NUM_VGPRS
        self.sgprs = [0] * DEFAULT_NUM_SGPRS
        self.lds = LDSMemory(lds_dwords)

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index]

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = u32(val)

    def get_lds(self, dword_idx: int) -> int:
        return self.lds.read_dword(dword_idx)

    def set_lds(self, dword_idx: int, val: int) -> None:
        self.lds.write_dword(dword_idx, val)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for DS src2 opcodes."""

    def execute(self, instr: DSSrc2Instruction, state: ScalarMachineState) -> None:
        base_addr = state.get_vgpr(instr.addr.base_reg.index)
        _, _, idx_a, idx_b = calc_lds_src2_addresses(
            vaddr=base_addr,
            offset0=instr.addr.offset0,
            offset1=instr.addr.offset1,
            lds_dwords=state.lds.size_dwords,
        )

        val_a = state.lds.read_dword(idx_a)
        val_b = state.lds.read_dword(idx_b)

        if instr.opcode == DSOpcode.DS_SUB_SRC2_U32:
            new_a = ref_ds_sub_src2_u32(val_a, val_b)
        else:
            raise NotImplementedError(f"Unsupported opcode: {instr.opcode}")

        # In-place atomic update to LDS location A
        state.lds.write_dword(idx_a, new_a)


class SIMDState:
    """Multi-lane SIMD machine state supporting Wave32 and Wave64."""

    def __init__(
        self,
        num_lanes: int = DEFAULT_NUM_LANES,
        exec_mask: int = MASK64,
        lds_dwords: int = DEFAULT_LDS_DWORDS,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"Unsupported lane count: {num_lanes} (must be 32 or 64)")
        self.num_lanes = num_lanes
        self.exec_mask = exec_mask & ((1 << num_lanes) - 1)
        self.vgprs = [[0] * DEFAULT_NUM_VGPRS for _ in range(num_lanes)]
        self.sgprs = [0] * DEFAULT_NUM_SGPRS
        self.vcc = 0
        self.lds = LDSMemory(lds_dwords)

    def get_lane_vgpr(self, lane: int, reg_index: int) -> int:
        return self.vgprs[lane][reg_index]

    def set_lane_vgpr(self, lane: int, reg_index: int, val: int) -> None:
        self.vgprs[lane][reg_index] = u32(val)


class SIMDInterpreter:
    """SIMD instruction interpreter handling lane divergence and EXEC masking."""

    def execute(self, instr: DSSrc2Instruction, state: SIMDState) -> None:
        if instr.opcode != DSOpcode.DS_SUB_SRC2_U32:
            raise NotImplementedError(f"Unsupported opcode: {instr.opcode}")

        for lane in range(state.num_lanes):
            if not ((state.exec_mask >> lane) & 1):
                continue

            base_addr = state.get_lane_vgpr(lane, instr.addr.base_reg.index)
            _, _, idx_a, idx_b = calc_lds_src2_addresses(
                vaddr=base_addr,
                offset0=instr.addr.offset0,
                offset1=instr.addr.offset1,
                lds_dwords=state.lds.size_dwords,
            )

            val_a = state.lds.read_dword(idx_a)
            val_b = state.lds.read_dword(idx_b)
            new_a = ref_ds_sub_src2_u32(val_a, val_b)
            state.lds.write_dword(idx_a, new_a)


def calc_sub_borrow(val_a: int, val_b: int) -> int:
    """Calculate unsigned 32-bit borrow generation (1 if A < B, else 0)."""
    return 1 if u32(val_a) < u32(val_b) else 0


class TestIEEE754BitPatternHandling(unittest.TestCase):
    """Unit tests for IEEE 754 bit-pattern preservation and untyped raw dword arithmetic."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState(lds_dwords=256)

    def test_signed_zero_bit_patterns(self) -> None:
        # +0.0 (0x00000000) - -0.0 (0x80000000) = 0x80000000 (-0.0)
        self.state.set_vgpr(0, 0)
        self.state.set_lds(0, POS_ZERO_F32)
        self.state.set_lds(1, NEG_ZERO_F32)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=0, offset1=1),
        )
        self.interp.execute(instr, self.state)
        self.assertEqual(self.state.get_lds(0), NEG_ZERO_F32)

        # -0.0 - -0.0 = 0x00000000 (+0.0)
        self.state.set_lds(0, NEG_ZERO_F32)
        self.interp.execute(instr, self.state)
        self.assertEqual(self.state.get_lds(0), POS_ZERO_F32)

    def test_infinity_bit_patterns(self) -> None:
        # +inf (0x7F800000) - +inf = 0
        self.state.set_vgpr(0, 0)
        self.state.set_lds(0, POS_INF_F32)
        self.state.set_lds(1, POS_INF_F32)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=0, offset1=1),
        )
        self.interp.execute(instr, self.state)
        self.assertEqual(self.state.get_lds(0), 0)

        # +inf (0x7F800000) - -inf (0xFF800000) = 0x80000000 mod 2^32
        self.state.set_lds(0, POS_INF_F32)
        self.state.set_lds(1, NEG_INF_F32)
        self.interp.execute(instr, self.state)
        self.assertEqual(self.state.get_lds(0), 0x80000000)

    def test_nan_bit_patterns_unaltered_raw_arithmetic(self) -> None:
        # qNaN (0x7FC00000) - sNaN (0x7F800001) = 0x003FFFFF
        # Pure bitwise integer subtraction without quietening or exception trapping
        self.state.set_vgpr(0, 0)
        self.state.set_lds(0, QNAN_F32)
        self.state.set_lds(1, SNAN_F32)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=0, offset1=1),
        )
        self.interp.execute(instr, self.state)
        expected = u32(QNAN_F32 - SNAN_F32)
        self.assertEqual(self.state.get_lds(0), expected)

    def test_subnormal_no_ftz_integer_subtraction(self) -> None:
        # Subnormal float bit patterns do not flush to zero in integer memory ops
        self.state.set_vgpr(0, 0)
        self.state.set_lds(0, MIN_SUBNORMAL_F32)
        self.state.set_lds(1, 1)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=0, offset1=1),
        )
        self.interp.execute(instr, self.state)
        self.assertEqual(self.state.get_lds(0), 0)

        # 0 - MIN_SUBNORMAL_F32 = 0xFFFFFFFF
        self.state.set_lds(0, 0)
        self.state.set_lds(1, MIN_SUBNORMAL_F32)
        self.interp.execute(instr, self.state)
        self.assertEqual(self.state.get_lds(0), MASK32)


class TestConditionCodesAndBorrow(unittest.TestCase):
    """Unit tests for condition codes, VCC preservation, and carry/borrow logic."""

    def test_borrow_generation_reference_oracle(self) -> None:
        self.assertEqual(calc_sub_borrow(10, 5), 0)
        self.assertEqual(calc_sub_borrow(5, 10), 1)
        self.assertEqual(calc_sub_borrow(0, 1), 1)
        self.assertEqual(calc_sub_borrow(MASK32, MASK32), 0)
        self.assertEqual(calc_sub_borrow(0, 0), 0)

    def test_vcc_strictly_preserved_scalar(self) -> None:
        # In GFX10, LDS atomic instructions do NOT modify VCC
        interp = ScalarInterpreter()
        state = ScalarMachineState(lds_dwords=256)
        state.set_vgpr(0, 0x10)
        state.set_lds(4, 5)
        state.set_lds(5, 10)  # Generates borrow (5 < 10)

        # Pre-set scalar state dummy VCC representation
        vcc_initial = 0x123456789ABCDEF0
        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=0, offset1=1),
        )
        interp.execute(instr, state)
        # LDS value wrapped
        self.assertEqual(state.get_lds(4), MASK32 - 4)

    def test_vcc_strictly_preserved_simd(self) -> None:
        interp = SIMDInterpreter()
        initial_vcc = 0xCAFEBABE11223344
        state = SIMDState(num_lanes=64, exec_mask=MASK64, lds_dwords=1024)
        state.vcc = initial_vcc

        for lane in range(64):
            state.set_lane_vgpr(lane, 0, lane * 8)
            state.lds.write_dword(lane * 2, 0)
            state.lds.write_dword(lane * 2 + 1, 1)  # All lanes generate borrow

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=0, offset1=1),
        )
        interp.execute(instr, state)
        self.assertEqual(state.vcc, initial_vcc, "VCC must remain unmodified by LDS operations")


class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Unit tests for register preservation and memory location self-aliasing."""

    def test_vaddr_register_unmodified_scalar(self) -> None:
        interp = ScalarInterpreter()
        state = ScalarMachineState(lds_dwords=256)
        initial_vaddr = 0x1234
        state.set_vgpr(5, initial_vaddr)
        state.set_lds(10, 100)
        state.set_lds(11, 40)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(5), offset0=0, offset1=1),
        )
        interp.execute(instr, state)
        # vaddr in v5 must remain strictly untouched
        self.assertEqual(state.get_vgpr(5), initial_vaddr)

    def test_vaddr_register_unmodified_simd(self) -> None:
        interp = SIMDInterpreter()
        state = SIMDState(num_lanes=64, exec_mask=MASK64, lds_dwords=512)
        for lane in range(64):
            state.set_lane_vgpr(lane, 3, 0x100 + lane * 4)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(3), offset0=0, offset1=1),
        )
        interp.execute(instr, state)
        for lane in range(64):
            self.assertEqual(state.get_lane_vgpr(lane, 3), 0x100 + lane * 4)

    def test_same_address_self_subtraction_zeroes_memory(self) -> None:
        # If offset0 == offset1, location A and location B are identical.
        # LDS[A] = LDS[A] - LDS[A] = 0
        interp = ScalarInterpreter()
        state = ScalarMachineState(lds_dwords=256)
        state.set_vgpr(0, 0x40)  # dword 16
        state.set_lds(16, 0x98765432)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=2, offset1=2),
        )
        # byte address = 0x40 + 8 = 0x48 -> dword 18
        state.set_lds(18, 0xDEADBEEF)
        interp.execute(instr, state)
        self.assertEqual(state.get_lds(18), 0)

    def test_cross_lane_address_aliasing(self) -> None:
        # Multiple lanes targeting the same LDS dword
        interp = SIMDInterpreter()
        state = SIMDState(num_lanes=32, exec_mask=0xF, lds_dwords=256)  # 4 lanes active
        # All 4 lanes target A=dword 10, B=dword 20
        for lane in range(4):
            state.set_lane_vgpr(lane, 0, 40)  # 40 bytes = dword 10
        state.lds.write_dword(10, 100)
        state.lds.write_dword(20, 10)

        # offset0=0 -> dword 10, offset1=10 -> byte 40+40=80 -> dword 20
        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=0, offset1=10),
        )
        interp.execute(instr, state)
        # 4 consecutive subtractions of 10 from 100 -> 100 - 40 = 60
        self.assertEqual(state.lds.read_dword(10), 60)
        self.assertEqual(state.lds.read_dword(20), 10)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD execution, lane divergence, and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_wave32_full_exec(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=MASK32, lds_dwords=1024)
        for lane in range(32):
            state.set_lane_vgpr(lane, 0, lane * 8)
            state.lds.write_dword(lane * 2, 100)
            state.lds.write_dword(lane * 2 + 1, lane)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=0, offset1=1),
        )
        self.interp.execute(instr, state)
        for lane in range(32):
            self.assertEqual(state.lds.read_dword(lane * 2), 100 - lane)
            self.assertEqual(state.lds.read_dword(lane * 2 + 1), lane)

    def test_wave32_divergent_masking(self) -> None:
        # Alternating mask (0x55555555): even lanes active, odd lanes inactive
        state = SIMDState(num_lanes=32, exec_mask=0x55555555, lds_dwords=1024)
        for lane in range(32):
            state.set_lane_vgpr(lane, 0, lane * 8)
            state.lds.write_dword(lane * 2, 100)
            state.lds.write_dword(lane * 2 + 1, 5)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=0, offset1=1),
        )
        self.interp.execute(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(state.lds.read_dword(lane * 2), 95)
            else:
                self.assertEqual(state.lds.read_dword(lane * 2), 100)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=MASK64, lds_dwords=2048)
        for lane in range(64):
            state.set_lane_vgpr(lane, 1, lane * 8)
            state.lds.write_dword(lane * 2, 200)
            state.lds.write_dword(lane * 2 + 1, lane)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(1), offset0=0, offset1=1),
        )
        self.interp.execute(instr, state)
        for lane in range(64):
            self.assertEqual(state.lds.read_dword(lane * 2), 200 - lane)

    def test_wave64_divergent_masking(self) -> None:
        # Low 32 lanes active, high 32 lanes inactive
        mask = 0x00000000FFFFFFFF
        state = SIMDState(num_lanes=64, exec_mask=mask, lds_dwords=2048)
        for lane in range(64):
            state.set_lane_vgpr(lane, 0, lane * 8)
            state.lds.write_dword(lane * 2, 50)
            state.lds.write_dword(lane * 2 + 1, 10)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=0, offset1=1),
        )
        self.interp.execute(instr, state)
        for lane in range(64):
            if lane < 32:
                self.assertEqual(state.lds.read_dword(lane * 2), 40)
            else:
                self.assertEqual(state.lds.read_dword(lane * 2), 50)

    def test_exec_mask_zero_no_modification(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0, lds_dwords=512)
        for lane in range(64):
            state.set_lane_vgpr(lane, 0, lane * 8)
            state.lds.write_dword(lane * 2, 42)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=0, offset1=1),
        )
        self.interp.execute(instr, state)
        for lane in range(64):
            self.assertEqual(state.lds.read_dword(lane * 2), 42)


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState(lds_dwords=1024)

    def test_scalar_sub_src2_basic(self) -> None:
        # v0 = base address 0x100
        # offset0 = 0 -> A is at dword 0x40 (64)
        # offset1 = 1 -> B is at dword 0x41 (65)
        self.state.set_vgpr(0, 0x100)
        self.state.set_lds(64, 50)
        self.state.set_lds(65, 30)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=0, offset1=1),
        )
        self.interp.execute(instr, self.state)
        # LDS[A] updated to 50 - 30 = 20
        self.assertEqual(self.state.get_lds(64), 20)
        # LDS[B] remains 30
        self.assertEqual(self.state.get_lds(65), 30)

    def test_scalar_sub_src2_with_offsets(self) -> None:
        self.state.set_vgpr(1, 0x200)
        # offset0 = 4 -> A = 0x210 -> dword 132
        # offset1 = 8 -> B = 0x220 -> dword 136
        self.state.set_lds(132, 1000)
        self.state.set_lds(136, 250)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(1), offset0=4, offset1=8),
        )
        self.interp.execute(instr, self.state)
        self.assertEqual(self.state.get_lds(132), 750)
        self.assertEqual(self.state.get_lds(136), 250)

    def test_scalar_sub_src2_underflow_wrap(self) -> None:
        self.state.set_vgpr(0, 0)
        self.state.set_lds(0, 5)
        self.state.set_lds(1, 10)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=0, offset1=1),
        )
        self.interp.execute(instr, self.state)
        # 5 - 10 wraps to MASK32 - 4
        self.assertEqual(self.state.get_lds(0), MASK32 - 4)

    def test_scalar_determinism(self) -> None:
        runs = []
        for _ in range(50):
            st = ScalarMachineState(lds_dwords=256)
            st.set_vgpr(0, 0x40)
            st.set_lds(16, 0x87654321)
            st.set_lds(17, 0x12345678)
            instr = DSSrc2Instruction(
                opcode=DSOpcode.DS_SUB_SRC2_U32,
                addr=LDSAddressExpression(base_reg=RegisterOperand(0), offset0=0, offset1=1),
            )
            self.interp.execute(instr, st)
            runs.append(st.get_lds(16))
        self.assertTrue(all(r == runs[0] for r in runs))


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_enumeration(self) -> None:
        self.assertEqual(DSOpcode.DS_SUB_SRC2_U32.value, "ds_sub_src2_u32")

    def test_register_operand_validation(self) -> None:
        reg = RegisterOperand(index=0, reg_type=RegType.VGPR)
        self.assertEqual(reg.name, "v0")

        with self.assertRaises(ValueError):
            RegisterOperand(index=-1, reg_type=RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(index=256, reg_type=RegType.VGPR)

    def test_address_expression_validation(self) -> None:
        v0 = RegisterOperand(index=0, reg_type=RegType.VGPR)
        addr = LDSAddressExpression(base_reg=v0, offset0=4, offset1=8)
        self.assertEqual(addr.offset0, 4)
        self.assertEqual(addr.offset1, 8)

        # Offsets must fit in 8-bit unsigned integer (0..255)
        with self.assertRaises(ValueError):
            LDSAddressExpression(base_reg=v0, offset0=256)
        with self.assertRaises(ValueError):
            LDSAddressExpression(base_reg=v0, offset1=-1)

        # Base register must be VGPR
        s0 = RegisterOperand(index=0, reg_type=RegType.SGPR)
        with self.assertRaises(TypeError):
            LDSAddressExpression(base_reg=s0)

    def test_disasm_formatting(self) -> None:
        v2 = RegisterOperand(index=2, reg_type=RegType.VGPR)
        instr_basic = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=v2),
        )
        self.assertEqual(instr_basic.disasm(), "ds_sub_src2_u32 v2")

        instr_offsets = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=v2, offset0=4, offset1=16, gds=True),
        )
        self.assertEqual(instr_offsets.disasm(), "ds_sub_src2_u32 v2 offset0:4 offset1:16 gds")

    def test_structural_equality(self) -> None:
        v0 = RegisterOperand(index=0, reg_type=RegType.VGPR)
        i1 = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=v0, offset0=10),
        )
        i2 = DSSrc2Instruction(
            opcode=DSOpcode.DS_SUB_SRC2_U32,
            addr=LDSAddressExpression(base_reg=v0, offset0=10),
        )
        self.assertEqual(i1, i2)


class TestReferenceModelAndOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model."""

    def test_basic_subtraction(self) -> None:
        self.assertEqual(ref_ds_sub_src2_u32(50, 20), 30)
        self.assertEqual(ref_ds_sub_src2_u32(350, 100), 250)
        self.assertEqual(ref_ds_sub_src2_u32(0x23456789, 0x11111111), 0x12345678)

    def test_subtraction_with_zero(self) -> None:
        self.assertEqual(ref_ds_sub_src2_u32(0, 0), 0)
        self.assertEqual(ref_ds_sub_src2_u32(42, 0), 42)
        # 0 - 42 wraps modulo 2^32
        self.assertEqual(ref_ds_sub_src2_u32(0, 42), (1 << 32) - 42)

    def test_subtraction_underflow_modulo_wrapping(self) -> None:
        self.assertEqual(ref_ds_sub_src2_u32(0, 1), MASK32)
        self.assertEqual(ref_ds_sub_src2_u32(5, 10), MASK32 - 4)
        self.assertEqual(ref_ds_sub_src2_u32(0x80000000, 0x80000001), MASK32)
        self.assertEqual(ref_ds_sub_src2_u32(0, MASK32), 1)

    def test_address_calculation_and_alignment(self) -> None:
        byte_a, byte_b, idx_a, idx_b = calc_lds_src2_addresses(
            vaddr=0x1000, offset0=2, offset1=4, lds_dwords=1024
        )
        self.assertEqual(byte_a, 0x1008)
        self.assertEqual(byte_b, 0x1010)
        self.assertEqual(idx_a, (0x1008 >> 2) & 1023)
        self.assertEqual(idx_b, (0x1010 >> 2) & 1023)

    def test_self_referencing_aliasing_oracle(self) -> None:
        # When A and B point to the same location, subtracting self yields zero:
        self.assertEqual(ref_ds_sub_src2_u32(4, 4), 0)
        self.assertEqual(ref_ds_sub_src2_u32(0x80000000, 0x80000000), 0)
        self.assertEqual(ref_ds_sub_src2_u32(MASK32, MASK32), 0)


if __name__ == "__main__":
    unittest.main()
