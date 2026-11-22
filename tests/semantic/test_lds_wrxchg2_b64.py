from __future__ import annotations

import struct
import unittest
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

# LDS Architectural Constants for RDNA2 (GFX10/GFX10.3)
LDS_DEFAULT_SIZE_BYTES: int = 65536  # 64 KiB default Workgroup LDS allocation
LDS_DEFAULT_SIZE_DWORDS: int = 16384
DWORD_BYTES: int = 4
QWORD_BYTES: int = 8
ST64_MULTIPLIER: int = 64
UNIT_B64_BYTES: int = 8
UNIT_B64_ST64_BYTES: int = 8 * ST64_MULTIPLIER  # 512 bytes

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF


class RegType(Enum):
    """Register file domain."""
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class RegSingle:
    """Represents a single 32-bit register."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index out of bounds [0, 255]: {self.index}")

    def __str__(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class RegPair:
    """Represents an aligned 64-bit pair of registers (e.g. v[0:1])."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 254):
            raise ValueError(f"Register pair base index out of bounds [0, 254]: {self.index}")

    @property
    def indices(self) -> Tuple[int, int]:
        return self.index, self.index + 1

    def __str__(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}[{self.index}:{self.index + 1}]"


@dataclass(frozen=True)
class RegQuad:
    """Represents an aligned 128-bit quad of registers (e.g. v[0:3])."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 252):
            raise ValueError(f"Register quad base index out of bounds [0, 252]: {self.index}")

    @property
    def indices(self) -> Tuple[int, int, int, int]:
        return self.index, self.index + 1, self.index + 2, self.index + 3

    def __str__(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}[{self.index}:{self.index + 3}]"


@dataclass(frozen=True)
class LDSAddressExpression:
    """Symbolic representation of LDS address computation."""
    base: RegSingle
    offset: int
    is_st64: bool = False
    lds_size_bytes: int = LDS_DEFAULT_SIZE_BYTES

    def __post_init__(self) -> None:
        if not (0 <= self.offset <= 255):
            raise ValueError(f"LDS offset out of 8-bit immediate range [0, 255]: {self.offset}")

    @property
    def stride(self) -> int:
        return UNIT_B64_ST64_BYTES if self.is_st64 else UNIT_B64_BYTES

    def evaluate(self, base_val: int) -> int:
        return (base_val + self.offset * self.stride) & (self.lds_size_bytes - 1)


@dataclass(frozen=True)
class DSWrxchg2B64Instruction:
    """AST representation of RDNA2 ds_wrxchg2_b64 instruction family."""
    vaddr: RegSingle
    data0: RegPair
    data1: RegPair
    vdst: Optional[RegQuad] = None
    offset0: int = 0
    offset1: int = 0
    is_st64: bool = False
    is_rtn: bool = True

    def __post_init__(self) -> None:
        if not (0 <= self.offset0 <= 255):
            raise ValueError(f"offset0 out of 8-bit range [0, 255]: {self.offset0}")
        if not (0 <= self.offset1 <= 255):
            raise ValueError(f"offset1 out of 8-bit range [0, 255]: {self.offset1}")
        if self.is_rtn and self.vdst is None:
            raise ValueError("Returning form (_rtn_) requires vdst destination quad")
        if self.vdst is not None and self.vdst.reg_type != RegType.VGPR:
            raise TypeError("LDS DS instruction vdst must be VGPRs")
        if self.vaddr.reg_type != RegType.VGPR:
            raise TypeError("LDS DS instruction vaddr must be VGPR")
        if self.data0.reg_type != RegType.VGPR or self.data1.reg_type != RegType.VGPR:
            raise TypeError("LDS DS instruction data operands must be VGPR pairs")

    @property
    def mnemonic(self) -> str:
        st64_str = "st64" if self.is_st64 else ""
        rtn_str = "_rtn" if self.is_rtn else ""
        return f"ds_wrxchg2{st64_str}{rtn_str}_b64"

    def disasm(self) -> str:
        parts = [self.mnemonic]
        if self.is_rtn and self.vdst is not None:
            parts.append(f"{self.vdst},")
        parts.append(f"{self.vaddr},")
        parts.append(f"{self.data0},")
        parts.append(f"{self.data1}")
        line = f"{parts[0]} {' '.join(parts[1:])}"
        if self.offset0 != 0:
            line += f" offset0:{self.offset0}"
        if self.offset1 != 0:
            line += f" offset1:{self.offset1}"
        return line


class ScalarRegisterFile:
    """Models a 256-register 32-bit scalar VGPR file."""

    def __init__(self) -> None:
        self.regs: List[int] = [0] * 256

    def get_u32(self, reg_idx: int) -> int:
        return self.regs[reg_idx] & MASK32

    def set_u32(self, reg_idx: int, val: int) -> None:
        self.regs[reg_idx] = val & MASK32

    def get_u64(self, pair: RegPair) -> int:
        lo = self.get_u32(pair.index)
        hi = self.get_u32(pair.index + 1)
        return combine_u32s_to_u64(lo, hi)

    def set_u64(self, pair: RegPair, val: int) -> None:
        lo, hi = split_u64_to_u32s(val)
        self.set_u32(pair.index, lo)
        self.set_u32(pair.index + 1, hi)

    def set_quad(self, quad: RegQuad, w0: int, w1: int, w2: int, w3: int) -> None:
        self.set_u32(quad.index, w0)
        self.set_u32(quad.index + 1, w1)
        self.set_u32(quad.index + 2, w2)
        self.set_u32(quad.index + 3, w3)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for ds_wrxchg2_b64."""

    def __init__(self, lds_size_bytes: int = LDS_DEFAULT_SIZE_BYTES) -> None:
        self.lds_mem: bytearray = bytearray(lds_size_bytes)
        self.rf: ScalarRegisterFile = ScalarRegisterFile()

    def step(self, instr: DSWrxchg2B64Instruction) -> Tuple[int, int]:
        """Execute a single ds_wrxchg2_b64 instruction on the scalar state."""
        vaddr = self.rf.get_u32(instr.vaddr.index)
        data0 = self.rf.get_u64(instr.data0)
        data1 = self.rf.get_u64(instr.data1)

        old0, old1 = oracle_ds_wrxchg2_b64(
            self.lds_mem,
            vaddr=vaddr,
            data0=data0,
            data1=data1,
            offset0=instr.offset0,
            offset1=instr.offset1,
            is_st64=instr.is_st64,
        )

        if instr.is_rtn and instr.vdst is not None:
            w0, w1 = split_u64_to_u32s(old0)
            w2, w3 = split_u64_to_u32s(old1)
            self.rf.set_quad(instr.vdst, w0, w1, w2, w3)

        return old0, old1


def split_u64_to_u32s(val: int) -> Tuple[int, int]:
    """Split 64-bit unsigned integer into (lo, hi) 32-bit words."""
    val &= MASK64
    return val & MASK32, (val >> 32) & MASK32


def combine_u32s_to_u64(lo: int, hi: int) -> int:
    """Combine (lo, hi) 32-bit words into a 64-bit unsigned integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def compute_lds_address(
    vaddr: int,
    offset: int,
    is_st64: bool = False,
    lds_size_bytes: int = LDS_DEFAULT_SIZE_BYTES,
) -> int:
    """Compute wrapped byte address in LDS for ds_wrxchg2_b64.

    Address formula: (vaddr + offset * unit) & (lds_size_bytes - 1)
    where unit is 8 bytes for standard ds_wrxchg2_b64, or 512 bytes for ds_wrxchg2st64_b64.
    """
    unit = UNIT_B64_ST64_BYTES if is_st64 else UNIT_B64_BYTES
    effective = (vaddr + offset * unit) & (lds_size_bytes - 1)
    return effective


def oracle_ds_wrxchg2_b64(
    lds_mem: bytearray,
    vaddr: int,
    data0: int,
    data1: int,
    offset0: int = 0,
    offset1: int = 0,
    is_st64: bool = False,
) -> Tuple[int, int]:
    """Reference mathematical oracle for RDNA2 ds_wrxchg2_b64.

    Simulates atomic exchange of two 64-bit elements in LDS memory.
    Returns (old_data0, old_data1).
    When addr0 == addr1, slot 0 exchange occurs first, followed by slot 1.
    """
    lds_bytes = len(lds_mem)
    addr0 = compute_lds_address(vaddr, offset0, is_st64, lds_bytes)
    addr1 = compute_lds_address(vaddr, offset1, is_st64, lds_bytes)

    # 4-byte dword alignment mask standard in DS addressing
    addr0 &= ~3
    addr1 &= ~3

    # Slot 0 atomic exchange
    addr0_next = (addr0 + 4) & (lds_bytes - 1)
    old0_lo = struct.unpack_from("<I", lds_mem, addr0)[0]
    old0_hi = struct.unpack_from("<I", lds_mem, addr0_next)[0]
    old0 = combine_u32s_to_u64(old0_lo, old0_hi)

    d0_lo, d0_hi = split_u64_to_u32s(data0)
    struct.pack_into("<I", lds_mem, addr0, d0_lo)
    struct.pack_into("<I", lds_mem, addr0_next, d0_hi)

    # Slot 1 atomic exchange
    addr1_next = (addr1 + 4) & (lds_bytes - 1)
    old1_lo = struct.unpack_from("<I", lds_mem, addr1)[0]
    old1_hi = struct.unpack_from("<I", lds_mem, addr1_next)[0]
    old1 = combine_u32s_to_u64(old1_lo, old1_hi)

    d1_lo, d1_hi = split_u64_to_u32s(data1)
    struct.pack_into("<I", lds_mem, addr1, d1_lo)
    struct.pack_into("<I", lds_mem, addr1_next, d1_hi)

    return old0, old1


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.lds = bytearray(LDS_DEFAULT_SIZE_BYTES)

    def test_basic_exchange_identity(self) -> None:
        # Seed memory at byte 0x40 and 0x48
        struct.pack_into("<Q", self.lds, 0x40, 0x1111_2222_3333_4444)
        struct.pack_into("<Q", self.lds, 0x48, 0x5555_6666_7777_8888)

        old0, old1 = oracle_ds_wrxchg2_b64(
            self.lds,
            vaddr=0x40,
            data0=0xAAAA_BBBB_CCCC_DDDD,
            data1=0xEEEE_FFFF_0000_1111,
            offset0=0,
            offset1=1,  # 0x40 + 1 * 8 = 0x48
            is_st64=False,
        )

        self.assertEqual(old0, 0x1111_2222_3333_4444)
        self.assertEqual(old1, 0x5555_6666_7777_8888)

        # Verify new memory values
        mem0 = struct.unpack_from("<Q", self.lds, 0x40)[0]
        mem1 = struct.unpack_from("<Q", self.lds, 0x48)[0]
        self.assertEqual(mem0, 0xAAAA_BBBB_CCCC_DDDD)
        self.assertEqual(mem1, 0xEEEE_FFFF_0000_1111)

    def test_offset_scaling_unit_8(self) -> None:
        # offset0=3 -> +24 bytes, offset1=7 -> +56 bytes
        vaddr = 0x100
        addr0 = vaddr + 3 * 8
        addr1 = vaddr + 7 * 8
        struct.pack_into("<Q", self.lds, addr0, 0x1234_5678_9ABC_DEF0)
        struct.pack_into("<Q", self.lds, addr1, 0xFEDC_BA98_7654_3210)

        old0, old1 = oracle_ds_wrxchg2_b64(
            self.lds,
            vaddr=vaddr,
            data0=0x0102_0304_0506_0708,
            data1=0x0807_0605_0403_0201,
            offset0=3,
            offset1=7,
            is_st64=False,
        )

        self.assertEqual(old0, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(old1, 0xFEDC_BA98_7654_3210)
        self.assertEqual(struct.unpack_from("<Q", self.lds, addr0)[0], 0x0102_0304_0506_0708)
        self.assertEqual(struct.unpack_from("<Q", self.lds, addr1)[0], 0x0807_0605_0403_0201)

    def test_st64_stride_scaling_unit_512(self) -> None:
        # st64 form scales offsets by 512 bytes (8 * 64)
        vaddr = 0x80
        addr0 = vaddr + 0 * 512  # 0x80
        addr1 = vaddr + 1 * 512  # 0x280
        struct.pack_into("<Q", self.lds, addr0, 0xCAFE_BABE_0000_0001)
        struct.pack_into("<Q", self.lds, addr1, 0xDEAD_BEEF_0000_0002)

        old0, old1 = oracle_ds_wrxchg2_b64(
            self.lds,
            vaddr=vaddr,
            data0=0x1111_1111_1111_1111,
            data1=0x2222_2222_2222_2222,
            offset0=0,
            offset1=1,
            is_st64=True,
        )

        self.assertEqual(old0, 0xCAFE_BABE_0000_0001)
        self.assertEqual(old1, 0xDEAD_BEEF_0000_0002)
        self.assertEqual(struct.unpack_from("<Q", self.lds, addr0)[0], 0x1111_1111_1111_1111)
        self.assertEqual(struct.unpack_from("<Q", self.lds, addr1)[0], 0x2222_2222_2222_2222)

    def test_lds_memory_wraparound_boundary(self) -> None:
        # Wrap around 64 KiB LDS boundary
        vaddr = LDS_DEFAULT_SIZE_BYTES - 8  # 0xFFF8
        offset0 = 0  # 0xFFF8
        offset1 = 1  # 0xFFF8 + 8 = 0x10000 -> wraps to 0x0000
        struct.pack_into("<Q", self.lds, 0xFFF8, 0x9999_8888_7777_6666)
        struct.pack_into("<Q", self.lds, 0x0000, 0x3333_2222_1111_0000)

        old0, old1 = oracle_ds_wrxchg2_b64(
            self.lds,
            vaddr=vaddr,
            data0=0xAAAA_AAAA_AAAA_AAAA,
            data1=0xBBBB_BBBB_BBBB_BBBB,
            offset0=offset0,
            offset1=offset1,
            is_st64=False,
        )

        self.assertEqual(old0, 0x9999_8888_7777_6666)
        self.assertEqual(old1, 0x3333_2222_1111_0000)
        self.assertEqual(struct.unpack_from("<Q", self.lds, 0xFFF8)[0], 0xAAAA_AAAA_AAAA_AAAA)
        self.assertEqual(struct.unpack_from("<Q", self.lds, 0x0000)[0], 0xBBBB_BBBB_BBBB_BBBB)

    def test_same_address_collision_sequential_semantics(self) -> None:
        # Both offset0 and offset1 target the exact same address
        vaddr = 0x200
        addr = vaddr + 2 * 8  # 0x210
        struct.pack_into("<Q", self.lds, addr, 0x0000_0000_FEED_FACE)

        d0 = 0x1111_2222_3333_4444
        d1 = 0x5555_6666_7777_8888
        old0, old1 = oracle_ds_wrxchg2_b64(
            self.lds,
            vaddr=vaddr,
            data0=d0,
            data1=d1,
            offset0=2,
            offset1=2,
            is_st64=False,
        )

        # Slot 0 reads initial memory (0xFEEDFACE) and writes d0
        self.assertEqual(old0, 0x0000_0000_FEED_FACE)
        # Slot 1 reads what slot 0 just wrote (d0) and writes d1
        self.assertEqual(old1, d0)
        # Final memory holds d1
        self.assertEqual(struct.unpack_from("<Q", self.lds, addr)[0], d1)

    def test_dword_little_endian_layout(self) -> None:
        # Check that dwords are stored in little-endian order: low dword at addr, high dword at addr+4
        val = 0x1234_5678_9ABC_DEF0
        d_lo, d_hi = split_u64_to_u32s(val)
        self.assertEqual(d_lo, 0x9ABC_DEF0)
        self.assertEqual(d_hi, 0x1234_5678)

        oracle_ds_wrxchg2_b64(self.lds, vaddr=0x20, data0=val, data1=0, offset0=0, offset1=1)
        stored_lo = struct.unpack_from("<I", self.lds, 0x20)[0]
        stored_hi = struct.unpack_from("<I", self.lds, 0x24)[0]
        self.assertEqual(stored_lo, 0x9ABC_DEF0)
        self.assertEqual(stored_hi, 0x1234_5678)

    def test_boundary_bit_patterns(self) -> None:
        for val0, val1 in [
            (0, MASK64),
            (MASK64, 0),
            (0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA),
            (0x8000_0000_0000_0000, 0x7FFF_FFFF_FFFF_FFFF),
        ]:
            with self.subTest(val0=hex(val0), val1=hex(val1)):
                old0, old1 = oracle_ds_wrxchg2_b64(
                    self.lds, vaddr=0x10, data0=val0, data1=val1, offset0=0, offset1=1
                )
                self.assertEqual(struct.unpack_from("<Q", self.lds, 0x10)[0], val0)
                self.assertEqual(struct.unpack_from("<Q", self.lds, 0x18)[0], val1)


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_reg_single_pair_quad_construction(self) -> None:
        v1 = RegSingle(1)
        self.assertEqual(str(v1), "v1")
        self.assertEqual(v1.index, 1)
        self.assertEqual(v1.reg_type, RegType.VGPR)

        v_pair = RegPair(12)
        self.assertEqual(str(v_pair), "v[12:13]")
        self.assertEqual(v_pair.indices, (12, 13))

        v_quad = RegQuad(16)
        self.assertEqual(str(v_quad), "v[16:19]")
        self.assertEqual(v_quad.indices, (16, 17, 18, 19))

    def test_address_expression_evaluation(self) -> None:
        vaddr = RegSingle(2)
        addr_expr = LDSAddressExpression(base=vaddr, offset=5, is_st64=False)
        self.assertEqual(addr_expr.stride, 8)
        self.assertEqual(addr_expr.evaluate(0x100), 0x100 + 40)

        st64_expr = LDSAddressExpression(base=vaddr, offset=2, is_st64=True)
        self.assertEqual(st64_expr.stride, 512)
        self.assertEqual(st64_expr.evaluate(0x100), 0x100 + 1024)

    def test_instruction_disasm_canonical_formatting(self) -> None:
        instr = DSWrxchg2B64Instruction(
            vdst=RegQuad(16),
            vaddr=RegSingle(1),
            data0=RegPair(12),
            data1=RegPair(14),
            offset0=2,
            offset1=4,
            is_st64=False,
            is_rtn=True,
        )
        self.assertEqual(instr.mnemonic, "ds_wrxchg2_rtn_b64")
        self.assertEqual(
            instr.disasm(),
            "ds_wrxchg2_rtn_b64 v[16:19], v1, v[12:13], v[14:15] offset0:2 offset1:4",
        )

    def test_instruction_st64_mnemonic_and_disasm(self) -> None:
        instr = DSWrxchg2B64Instruction(
            vdst=RegQuad(0),
            vaddr=RegSingle(2),
            data0=RegPair(4),
            data1=RegPair(6),
            offset0=0,
            offset1=1,
            is_st64=True,
            is_rtn=True,
        )
        self.assertEqual(instr.mnemonic, "ds_wrxchg2st64_rtn_b64")
        self.assertEqual(
            instr.disasm(),
            "ds_wrxchg2st64_rtn_b64 v[0:3], v2, v[4:5], v[6:7] offset1:1",
        )

    def test_instruction_non_return_disasm(self) -> None:
        instr = DSWrxchg2B64Instruction(
            vdst=None,
            vaddr=RegSingle(5),
            data0=RegPair(8),
            data1=RegPair(10),
            offset0=3,
            offset1=6,
            is_st64=False,
            is_rtn=False,
        )
        self.assertEqual(instr.mnemonic, "ds_wrxchg2_b64")
        self.assertEqual(
            instr.disasm(),
            "ds_wrxchg2_b64 v5, v[8:9], v[10:11] offset0:3 offset1:6",
        )

    def test_ast_immutability_and_equality(self) -> None:
        instr1 = DSWrxchg2B64Instruction(
            vdst=RegQuad(0),
            vaddr=RegSingle(1),
            data0=RegPair(2),
            data1=RegPair(4),
            offset0=1,
            offset1=2,
        )
        instr2 = DSWrxchg2B64Instruction(
            vdst=RegQuad(0),
            vaddr=RegSingle(1),
            data0=RegPair(2),
            data1=RegPair(4),
            offset0=1,
            offset1=2,
        )
        self.assertEqual(instr1, instr2)
        self.assertEqual(hash(instr1), hash(instr2))


class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()

    def test_scalar_execute_basic_rtn(self) -> None:
        # Seed LDS at 0x20 and 0x28
        struct.pack_into("<Q", self.interp.lds_mem, 0x20, 0x1111_0000_2222_0000)
        struct.pack_into("<Q", self.interp.lds_mem, 0x28, 0x3333_0000_4444_0000)

        # Set registers
        self.interp.rf.set_u32(1, 0x20)  # v1 = vaddr
        self.interp.rf.set_u64(RegPair(4), 0xAAAA_1111_BBBB_2222)  # v[4:5] = data0
        self.interp.rf.set_u64(RegPair(6), 0xCCCC_3333_DDDD_4444)  # v[6:7] = data1

        instr = DSWrxchg2B64Instruction(
            vdst=RegQuad(8),  # v[8:11]
            vaddr=RegSingle(1),
            data0=RegPair(4),
            data1=RegPair(6),
            offset0=0,
            offset1=1,
            is_st64=False,
            is_rtn=True,
        )

        old0, old1 = self.interp.step(instr)
        self.assertEqual(old0, 0x1111_0000_2222_0000)
        self.assertEqual(old1, 0x3333_0000_4444_0000)

        # Check vdst receives old values
        self.assertEqual(self.interp.rf.get_u64(RegPair(8)), old0)
        self.assertEqual(self.interp.rf.get_u64(RegPair(10)), old1)

        # Check memory updated
        self.assertEqual(
            struct.unpack_from("<Q", self.interp.lds_mem, 0x20)[0],
            0xAAAA_1111_BBBB_2222,
        )
        self.assertEqual(
            struct.unpack_from("<Q", self.interp.lds_mem, 0x28)[0],
            0xCCCC_3333_DDDD_4444,
        )

    def test_scalar_execute_st64_rtn(self) -> None:
        vaddr = 0x40
        addr0 = vaddr + 0 * 512  # 0x40
        addr1 = vaddr + 2 * 512  # 0x440
        struct.pack_into("<Q", self.interp.lds_mem, addr0, 0x0123_4567_89AB_CDEF)
        struct.pack_into("<Q", self.interp.lds_mem, addr1, 0xFEDC_BA98_7654_3210)

        self.interp.rf.set_u32(2, vaddr)
        self.interp.rf.set_u64(RegPair(12), 0x9999_8888_7777_6666)
        self.interp.rf.set_u64(RegPair(14), 0x5555_4444_3333_2222)

        instr = DSWrxchg2B64Instruction(
            vdst=RegQuad(16),
            vaddr=RegSingle(2),
            data0=RegPair(12),
            data1=RegPair(14),
            offset0=0,
            offset1=2,
            is_st64=True,
            is_rtn=True,
        )

        old0, old1 = self.interp.step(instr)
        self.assertEqual(old0, 0x0123_4567_89AB_CDEF)
        self.assertEqual(old1, 0xFEDC_BA98_7654_3210)
        self.assertEqual(self.interp.rf.get_u64(RegPair(16)), old0)
        self.assertEqual(self.interp.rf.get_u64(RegPair(18)), old1)

    def test_scalar_execute_non_return(self) -> None:
        struct.pack_into("<Q", self.interp.lds_mem, 0x50, 0x1111_1111_1111_1111)
        struct.pack_into("<Q", self.interp.lds_mem, 0x58, 0x2222_2222_2222_2222)

        self.interp.rf.set_u32(3, 0x50)
        self.interp.rf.set_u64(RegPair(20), 0x3333_3333_3333_3333)
        self.interp.rf.set_u64(RegPair(22), 0x4444_4444_4444_4444)

        # Sentinel in registers 24..27
        for r in range(24, 28):
            self.interp.rf.set_u32(r, 0xCAFE_BABE)

        instr = DSWrxchg2B64Instruction(
            vdst=None,
            vaddr=RegSingle(3),
            data0=RegPair(20),
            data1=RegPair(22),
            offset0=0,
            offset1=1,
            is_st64=False,
            is_rtn=False,
        )

        old0, old1 = self.interp.step(instr)
        self.assertEqual(old0, 0x1111_1111_1111_1111)
        self.assertEqual(old1, 0x2222_2222_2222_2222)

        # Verify registers 24..27 are untouched
        for r in range(24, 28):
            self.assertEqual(self.interp.rf.get_u32(r), 0xCAFE_BABE)

    def test_scalar_step_determinism(self) -> None:
        # Run identical sequence in two independent interpreters and compare
        interp1 = ScalarInterpreter()
        interp2 = ScalarInterpreter()

        for interp in (interp1, interp2):
            struct.pack_into("<Q", interp.lds_mem, 0x100, 0xA1B2_C3D4_E5F6_0718)
            struct.pack_into("<Q", interp.lds_mem, 0x108, 0x1807_F6E5_D4C3_B2A1)
            interp.rf.set_u32(1, 0x100)
            interp.rf.set_u64(RegPair(2), 0x1234_0000_5678_0000)
            interp.rf.set_u64(RegPair(4), 0xABCD_0000_EF01_0000)

        instr = DSWrxchg2B64Instruction(
            vdst=RegQuad(6),
            vaddr=RegSingle(1),
            data0=RegPair(2),
            data1=RegPair(4),
            offset0=0,
            offset1=1,
        )

        res1 = interp1.step(instr)
        res2 = interp2.step(instr)
        self.assertEqual(res1, res2)
        self.assertEqual(bytes(interp1.lds_mem), bytes(interp2.lds_mem))
        self.assertEqual(interp1.rf.regs, interp2.rf.regs)


if __name__ == "__main__":
    unittest.main()
