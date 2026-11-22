"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_min_max_f32.

Models IEEE 754-2008 / IEEE 754-2019 binary32 single-precision minNum/maxNum and AMD RDNA2 / GFX10
Local Data Share (LDS) architectural semantics:
  - ds_min_f32: Atomic float minimum on LDS storage.
  - ds_max_f32: Atomic float maximum on LDS storage.
  - ds_min_rtn_f32: Atomic float minimum on LDS storage returning previous value to VGPR.
  - ds_max_rtn_f32: Atomic float maximum on LDS storage returning previous value to VGPR.

RDNA2 LDS / Floating-Point ISA Rules:
  1. Local Data Share Architecture:
     LDS is an on-chip, software-managed scratchpad memory shared across all threads
     in a workgroup. Addressing is byte-addressed but dword-aligned in hardware (low 2 bits
     dropped). Out-of-bounds addresses wrap within the allocated power-of-two LDS allocation.
     LDS is divided into 32 memory banks (bank = dword_index % 32).
  2. NaN Suppression (IEEE 754-2008 / 2019 minNum / maxNum):
     If one operand is NaN and the other is a valid floating-point number, the valid number is
     returned unchanged. If both operands are NaN, a quiet NaN is returned.
  3. Signed Zero Ordering (-0.0 < +0.0):
     RDNA2 orders -0.0 strictly less than +0.0:
       min(+0.0, -0.0) = -0.0, min(-0.0, +0.0) = -0.0
       max(+0.0, -0.0) = +0.0, max(-0.0, +0.0) = +0.0
  4. Infinities & Subnormals:
     -INF < all finite numbers < +INF.
     Subnormal float values are preserved without flushing to zero in IEEE mode.
  5. Atomic Operations & Return Semantics:
     Non-returning variants (ds_min_f32, ds_max_f32) discard the previous memory content.
     Returning variants (ds_min_rtn_f32, ds_max_rtn_f32) atomically fetch the previous
     32-bit floating-point value and write it into the destination VGPR.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest
from typing import Tuple, List, Optional, Dict, Any

# IEEE 754 binary32 architectural bitmasks
MASK32: int = 0xFFFFFFFF
SIGN_MASK: int = 0x80000000
EXP_MASK: int = 0x7F800000
FRAC_MASK: int = 0x007FFFFF
QNAN_BIT: int = 0x00400000
CANONICAL_QNAN: int = 0x7FC00000

# Canonical IEEE 754 binary32 bit patterns
POS_ZERO: int = 0x00000000
NEG_ZERO: int = 0x80000000
ONE_F32: int = 0x3F800000           # 1.0
NEG_ONE_F32: int = 0xBF800000       # -1.0
HALF_F32: int = 0x3F000000          # 0.5
NEG_HALF_F32: int = 0xBF000000      # -0.5
TWO_F32: int = 0x40000000           # 2.0
NEG_TWO_F32: int = 0xC0000000       # -2.0
THREE_F32: int = 0x40400000         # 3.0
TEN_F32: int = 0x41200000           # 10.0
NEG_TEN_F32: int = 0xC1200000       # -10.0

FLT_MIN: int = 0x00800000           # Smallest positive normal: 2^-126
NEG_FLT_MIN: int = 0x80800000       # -2^-126
FLT_MAX: int = 0x7F7FFFFF           # (2 - 2^-23) * 2^127
NEG_FLT_MAX: int = 0xFF7FFFFF
POS_INF: int = 0x7F800000           # +infinity
NEG_INF: int = 0xFF800000           # -infinity
QNAN_PAYLOAD_A: int = 0x7FC10001    # Quiet NaN with payload A
QNAN_PAYLOAD_B: int = 0xFFC20002    # Quiet NaN with payload B
SNAN_PAYLOAD_A: int = 0x7F810001    # Signaling NaN (MSB of mantissa is 0, payload != 0)
SNAN_PAYLOAD_B: int = 0xFF820002    # Signaling NaN
SUBNORM_MIN: int = 0x00000001       # Smallest subnormal: 2^-149
SUBNORM_MAX: int = 0x007FFFFF       # Largest positive subnormal: (1 - 2^-23) * 2^-126
NEG_SUBNORM_MIN: int = 0x80000001
NEG_SUBNORM_MAX: int = 0x807FFFFF

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB
DEFAULT_LDS_NUM_DWORDS: int = 16384
NUM_LDS_BANKS: int = 32


def f32_to_bits(val: float) -> int:
    """Encode float to IEEE 754 single-precision 32-bit unsigned integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def split_f32_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack 32-bit integer into (sign, exponent, fraction) components."""
    bits &= MASK32
    sign = (bits & SIGN_MASK) >> 31
    exponent = (bits & EXP_MASK) >> 23
    fraction = bits & FRAC_MASK
    return sign, exponent, fraction


def is_nan(bits: int) -> bool:
    """Return True if 32-bit integer represents an IEEE 754 float NaN."""
    bits &= MASK32
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) != 0)


def is_qnan(bits: int) -> bool:
    """Return True if 32-bit integer represents an IEEE 754 float Quiet NaN."""
    return is_nan(bits) and bool(bits & QNAN_BIT)


def is_snan(bits: int) -> bool:
    """Return True if 32-bit integer represents an IEEE 754 float Signaling NaN."""
    return is_nan(bits) and not bool(bits & QNAN_BIT)


def is_inf(bits: int) -> bool:
    """Return True if 32-bit integer represents IEEE 754 positive or negative infinity."""
    bits &= MASK32
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) == 0)


def is_subnormal(bits: int) -> bool:
    """Return True if 32-bit integer represents a subnormal single-precision float."""
    bits &= MASK32
    return ((bits & EXP_MASK) == 0) and ((bits & FRAC_MASK) != 0)


def is_zero(bits: int) -> bool:
    """Return True if 32-bit integer represents positive or negative zero."""
    return (bits & ~SIGN_MASK) == 0


def oracle_min_f32(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 ds_min_f32 atomic operation.

    Follows IEEE 754-2008/2019 minNum:
      - If one operand is NaN and the other is a valid float, returns the valid float.
      - If both are NaN, returns a quiet NaN (preserving payload or quietened).
      - Signed zero ordering: -0.0 < +0.0 (min(+0.0, -0.0) -> -0.0).
      - Subnormals and infinities preserved.
    """
    a = a_bits & MASK32
    b = b_bits & MASK32

    a_is_nan = is_nan(a)
    b_is_nan = is_nan(b)

    if a_is_nan and b_is_nan:
        # Both NaN: return quieted NaN
        return a | QNAN_BIT
    if a_is_nan:
        return b
    if b_is_nan:
        return a

    # Signed zeros
    if is_zero(a) and is_zero(b):
        # If either is negative zero, negative zero is smaller
        return NEG_ZERO if ((a == NEG_ZERO) or (b == NEG_ZERO)) else POS_ZERO

    fa = bits_to_f32(a)
    fb = bits_to_f32(b)

    if fa < fb:
        return a
    elif fb < fa:
        return b
    else:
        # Numerically equal: negative sign bit is strictly smaller
        if (a & SIGN_MASK) != 0:
            return a
        if (b & SIGN_MASK) != 0:
            return b
        return a


def oracle_max_f32(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 ds_max_f32 atomic operation.

    Follows IEEE 754-2008/2019 maxNum:
      - If one operand is NaN and the other is a valid float, returns the valid float.
      - If both are NaN, returns a quiet NaN (preserving payload or quietened).
      - Signed zero ordering: -0.0 < +0.0 (max(+0.0, -0.0) -> +0.0).
      - Subnormals and infinities preserved.
    """
    a = a_bits & MASK32
    b = b_bits & MASK32

    a_is_nan = is_nan(a)
    b_is_nan = is_nan(b)

    if a_is_nan and b_is_nan:
        # Both NaN: return quieted NaN
        return a | QNAN_BIT
    if a_is_nan:
        return b
    if b_is_nan:
        return a

    # Signed zeros
    if is_zero(a) and is_zero(b):
        # If either is positive zero, positive zero is larger
        return POS_ZERO if ((a == POS_ZERO) or (b == POS_ZERO)) else NEG_ZERO

    fa = bits_to_f32(a)
    fb = bits_to_f32(b)

    if fa > fb:
        return a
    elif fb > fa:
        return b
    else:
        # Numerically equal: positive sign bit (0) is strictly larger
        if (a & SIGN_MASK) == 0:
            return a
        if (b & SIGN_MASK) == 0:
            return b
        return a


class LDSMemory:
    """Architectural simulation model of AMD RDNA2 Local Data Share (LDS)."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of 2, got {size_bytes}")
        self.size_bytes: int = size_bytes
        self.num_dwords: int = size_bytes // 4
        self.dwords: List[int] = [0] * self.num_dwords

    def get_dword_index(self, byte_addr: int, offset: int = 0) -> int:
        """Compute effective wrapped dword index, dropping bottom 2 unaligned bits."""
        effective_byte = (byte_addr + offset) & MASK32
        return (effective_byte >> 2) & (self.num_dwords - 1)

    def get_bank(self, dword_idx: int) -> int:
        """Compute memory bank index for the given dword (0 to 31)."""
        return dword_idx % NUM_LDS_BANKS

    def read_dword(self, dword_idx: int) -> int:
        """Read a 32-bit dword from LDS."""
        return self.dwords[dword_idx & (self.num_dwords - 1)]

    def write_dword(self, dword_idx: int, val: int) -> None:
        """Write a 32-bit dword into LDS."""
        self.dwords[dword_idx & (self.num_dwords - 1)] = val & MASK32

    def atomic_min_f32(self, byte_addr: int, val: int, offset: int = 0) -> int:
        """Execute atomic float minimum at (byte_addr + offset), returning previous value."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        new_val = oracle_min_f32(old_val, val)
        self.dwords[idx] = new_val
        return old_val

    def atomic_max_f32(self, byte_addr: int, val: int, offset: int = 0) -> int:
        """Execute atomic float maximum at (byte_addr + offset), returning previous value."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        new_val = oracle_max_f32(old_val, val)
        self.dwords[idx] = new_val
        return old_val


class TestLDSMinMaxF32ModelAndOracle(unittest.TestCase):
    """Unit tests for binary32 floating point model, predicates, and reference LDS oracle."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 100.0, -100.0, 3.1415927]
        for val in values:
            with self.subTest(val=val):
                bits = f32_to_bits(val)
                decoded = bits_to_f32(bits)
                self.assertAlmostEqual(val, decoded, places=6)
                self.assertEqual(bits, f32_to_bits(decoded))

    def test_predicates(self) -> None:
        self.assertTrue(is_zero(POS_ZERO))
        self.assertTrue(is_zero(NEG_ZERO))
        self.assertFalse(is_zero(ONE_F32))

        self.assertTrue(is_nan(CANONICAL_QNAN))
        self.assertTrue(is_nan(QNAN_PAYLOAD_A))
        self.assertTrue(is_nan(SNAN_PAYLOAD_A))
        self.assertFalse(is_nan(POS_INF))
        self.assertFalse(is_nan(ONE_F32))

        self.assertTrue(is_qnan(CANONICAL_QNAN))
        self.assertTrue(is_qnan(QNAN_PAYLOAD_A))
        self.assertFalse(is_qnan(SNAN_PAYLOAD_A))

        self.assertTrue(is_snan(SNAN_PAYLOAD_A))
        self.assertFalse(is_snan(CANONICAL_QNAN))

        self.assertTrue(is_inf(POS_INF))
        self.assertTrue(is_inf(NEG_INF))
        self.assertFalse(is_inf(FLT_MAX))

        self.assertTrue(is_subnormal(SUBNORM_MIN))
        self.assertTrue(is_subnormal(SUBNORM_MAX))
        self.assertFalse(is_subnormal(FLT_MIN))
        self.assertFalse(is_subnormal(POS_ZERO))

    def test_oracle_min_f32_standard_values(self) -> None:
        self.assertEqual(oracle_min_f32(ONE_F32, TWO_F32), ONE_F32)
        self.assertEqual(oracle_min_f32(TWO_F32, ONE_F32), ONE_F32)
        self.assertEqual(oracle_min_f32(NEG_ONE_F32, NEG_TWO_F32), NEG_TWO_F32)
        self.assertEqual(oracle_min_f32(NEG_TEN_F32, TEN_F32), NEG_TEN_F32)

    def test_oracle_max_f32_standard_values(self) -> None:
        self.assertEqual(oracle_max_f32(ONE_F32, TWO_F32), TWO_F32)
        self.assertEqual(oracle_max_f32(TWO_F32, ONE_F32), TWO_F32)
        self.assertEqual(oracle_max_f32(NEG_ONE_F32, NEG_TWO_F32), NEG_ONE_F32)
        self.assertEqual(oracle_max_f32(NEG_TEN_F32, TEN_F32), TEN_F32)

    def test_oracle_signed_zeros(self) -> None:
        # In RDNA2 / IEEE 754-2019: -0.0 < +0.0
        self.assertEqual(oracle_min_f32(POS_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(oracle_min_f32(NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(oracle_max_f32(POS_ZERO, NEG_ZERO), POS_ZERO)
        self.assertEqual(oracle_max_f32(NEG_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_min_f32(NEG_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(oracle_max_f32(POS_ZERO, POS_ZERO), POS_ZERO)

    def test_oracle_nan_suppression(self) -> None:
        # Single NaN suppressed
        self.assertEqual(oracle_min_f32(CANONICAL_QNAN, ONE_F32), ONE_F32)
        self.assertEqual(oracle_min_f32(ONE_F32, CANONICAL_QNAN), ONE_F32)
        self.assertEqual(oracle_min_f32(SNAN_PAYLOAD_A, NEG_TWO_F32), NEG_TWO_F32)
        self.assertEqual(oracle_max_f32(CANONICAL_QNAN, ONE_F32), ONE_F32)
        self.assertEqual(oracle_max_f32(ONE_F32, CANONICAL_QNAN), ONE_F32)
        self.assertEqual(oracle_max_f32(SNAN_PAYLOAD_A, NEG_TWO_F32), NEG_TWO_F32)

        # Dual NaN produces quiet NaN
        res_min = oracle_min_f32(CANONICAL_QNAN, QNAN_PAYLOAD_B)
        self.assertTrue(is_qnan(res_min))
        res_max = oracle_max_f32(SNAN_PAYLOAD_A, SNAN_PAYLOAD_B)
        self.assertTrue(is_qnan(res_max))

    def test_oracle_infinities_and_subnormals(self) -> None:
        self.assertEqual(oracle_min_f32(POS_INF, FLT_MAX), FLT_MAX)
        self.assertEqual(oracle_min_f32(NEG_INF, NEG_FLT_MAX), NEG_INF)
        self.assertEqual(oracle_max_f32(POS_INF, FLT_MAX), POS_INF)
        self.assertEqual(oracle_max_f32(NEG_INF, NEG_FLT_MAX), NEG_FLT_MAX)

        self.assertEqual(oracle_min_f32(SUBNORM_MIN, SUBNORM_MAX), SUBNORM_MIN)
        self.assertEqual(oracle_max_f32(SUBNORM_MIN, SUBNORM_MAX), SUBNORM_MAX)
        self.assertEqual(oracle_min_f32(NEG_SUBNORM_MIN, NEG_ZERO), NEG_SUBNORM_MIN)
        self.assertEqual(oracle_max_f32(SUBNORM_MIN, POS_ZERO), SUBNORM_MIN)

    def test_lds_memory_model(self) -> None:
        lds = LDSMemory(65536)
        self.assertEqual(lds.num_dwords, 16384)

        # Byte address to dword index drops low 2 bits
        self.assertEqual(lds.get_dword_index(0x0004), 1)
        self.assertEqual(lds.get_dword_index(0x0007), 1)
        self.assertEqual(lds.get_dword_index(0x0008, offset=4), 3)

        # Address wrapping
        self.assertEqual(lds.get_dword_index(65536 + 16), 4)

        # Bank calculation
        self.assertEqual(lds.get_bank(0), 0)
        self.assertEqual(lds.get_bank(31), 31)
        self.assertEqual(lds.get_bank(32), 0)

        # Atomic min f32
        lds.write_dword(10, TWO_F32)
        old_val = lds.atomic_min_f32(10 * 4, ONE_F32)
        self.assertEqual(old_val, TWO_F32)
        self.assertEqual(lds.read_dword(10), ONE_F32)

        # Atomic max f32
        old_val2 = lds.atomic_max_f32(10 * 4, THREE_F32)
        self.assertEqual(old_val2, ONE_F32)
        self.assertEqual(lds.read_dword(10), THREE_F32)


class DSOpcode(str, enum.Enum):
    """Opcode enumerations for single-precision LDS min/max operations."""
    DS_MIN_F32 = "ds_min_f32"
    DS_MAX_F32 = "ds_max_f32"
    DS_MIN_RTN_F32 = "ds_min_rtn_f32"
    DS_MAX_RTN_F32 = "ds_max_rtn_f32"

    @property
    def is_return(self) -> bool:
        """True if the instruction returns the previous LDS value to a destination VGPR."""
        return self in (DSOpcode.DS_MIN_RTN_F32, DSOpcode.DS_MAX_RTN_F32)

    @property
    def is_min(self) -> bool:
        """True if operation is a floating-point minimum."""
        return self in (DSOpcode.DS_MIN_F32, DSOpcode.DS_MIN_RTN_F32)

    @property
    def is_max(self) -> bool:
        """True if operation is a floating-point maximum."""
        return self in (DSOpcode.DS_MAX_F32, DSOpcode.DS_MAX_RTN_F32)


@dataclasses.dataclass(frozen=True)
class VGPR:
    """Vector General Purpose Register operand v[0] .. v[255]."""
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")

    def __str__(self) -> str:
        return f"v{self.index}"


@dataclasses.dataclass(frozen=True)
class OffsetImmediate:
    """Unsigned 16-bit byte offset immediate [0, 65535]."""
    value: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.value <= 65535):
            raise ValueError(f"Offset immediate out of range [0, 65535]: {self.value}")

    def __str__(self) -> str:
        return f"offset:{self.value}" if self.value != 0 else ""


@dataclasses.dataclass(frozen=True)
class DSMinMaxInstruction:
    """Symbolic AST node representing an RDNA2 DS min/max f32 instruction."""
    opcode: DSOpcode
    vaddr: VGPR
    vdata: VGPR
    vdst: Optional[VGPR] = None
    offset: OffsetImmediate = OffsetImmediate(0)

    def __post_init__(self) -> None:
        if self.opcode.is_return:
            if self.vdst is None:
                raise ValueError(f"Returning opcode {self.opcode.value} requires destination vdst")
        else:
            if self.vdst is not None:
                raise ValueError(f"Non-returning opcode {self.opcode.value} must not have destination vdst")

    def disasm(self) -> str:
        """Format symbolic instruction into canonical RDNA2 assembly text."""
        parts = [self.opcode.value]
        if self.opcode.is_return:
            operands = f"{self.vdst}, {self.vaddr}, {self.vdata}"
        else:
            operands = f"{self.vaddr}, {self.vdata}"
        parts.append(operands)
        if self.offset.value != 0:
            parts.append(str(self.offset))
        return " ".join(parts)

    @classmethod
    def parse(cls, text: str) -> DSMinMaxInstruction:
        """Parse canonical RDNA2 assembly string into a DSMinMaxInstruction AST."""
        tokens = text.strip().split(None, 1)
        if not tokens:
            raise ValueError("Empty assembly string")
        op_str = tokens[0]
        try:
            opcode = DSOpcode(op_str)
        except ValueError:
            raise ValueError(f"Unknown or unsupported DS opcode: {op_str}")

        if len(tokens) == 1:
            raise ValueError(f"Missing operands for {op_str}")

        rest = tokens[1].strip()
        offset_val = 0
        if "offset:" in rest:
            m = rest.split("offset:")
            rest = m[0].strip()
            offset_tok = m[1].split()[0].rstrip(",")
            offset_val = int(offset_tok, 0)

        # Parse comma-separated registers
        reg_tokens = [tok.strip().rstrip(",") for tok in rest.split(",") if tok.strip()]
        
        def parse_vgpr(s: str) -> VGPR:
            s = s.strip()
            if not s.startswith("v") or not s[1:].isdigit():
                raise ValueError(f"Invalid VGPR syntax: {s}")
            return VGPR(int(s[1:]))

        if opcode.is_return:
            if len(reg_tokens) != 3:
                raise ValueError(f"{opcode.value} expects 3 registers (vdst, vaddr, vdata), got {reg_tokens}")
            vdst = parse_vgpr(reg_tokens[0])
            vaddr = parse_vgpr(reg_tokens[1])
            vdata = parse_vgpr(reg_tokens[2])
            return cls(opcode=opcode, vaddr=vaddr, vdata=vdata, vdst=vdst, offset=OffsetImmediate(offset_val))
        else:
            if len(reg_tokens) != 2:
                raise ValueError(f"{opcode.value} expects 2 registers (vaddr, vdata), got {reg_tokens}")
            vaddr = parse_vgpr(reg_tokens[0])
            vdata = parse_vgpr(reg_tokens[1])
            return cls(opcode=opcode, vaddr=vaddr, vdata=vdata, vdst=None, offset=OffsetImmediate(offset_val))


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions, registers, and assembly parsing."""

    def test_opcode_properties(self) -> None:
        self.assertFalse(DSOpcode.DS_MIN_F32.is_return)
        self.assertTrue(DSOpcode.DS_MIN_F32.is_min)
        self.assertFalse(DSOpcode.DS_MIN_F32.is_max)

        self.assertFalse(DSOpcode.DS_MAX_F32.is_return)
        self.assertFalse(DSOpcode.DS_MAX_F32.is_min)
        self.assertTrue(DSOpcode.DS_MAX_F32.is_max)

        self.assertTrue(DSOpcode.DS_MIN_RTN_F32.is_return)
        self.assertTrue(DSOpcode.DS_MIN_RTN_F32.is_min)
        self.assertFalse(DSOpcode.DS_MIN_RTN_F32.is_max)

        self.assertTrue(DSOpcode.DS_MAX_RTN_F32.is_return)
        self.assertFalse(DSOpcode.DS_MAX_RTN_F32.is_min)
        self.assertTrue(DSOpcode.DS_MAX_RTN_F32.is_max)

    def test_vgpr_validation(self) -> None:
        v0 = VGPR(0)
        v255 = VGPR(255)
        self.assertEqual(str(v0), "v0")
        self.assertEqual(str(v255), "v255")

        with self.assertRaises(ValueError):
            VGPR(-1)
        with self.assertRaises(ValueError):
            VGPR(256)

    def test_offset_immediate_validation(self) -> None:
        off0 = OffsetImmediate(0)
        off64 = OffsetImmediate(64)
        off_max = OffsetImmediate(65535)
        self.assertEqual(str(off0), "")
        self.assertEqual(str(off64), "offset:64")
        self.assertEqual(str(off_max), "offset:65535")

        with self.assertRaises(ValueError):
            OffsetImmediate(-1)
        with self.assertRaises(ValueError):
            OffsetImmediate(65536)

    def test_instruction_post_init_validation(self) -> None:
        # Non-returning must have vdst is None
        DSMinMaxInstruction(opcode=DSOpcode.DS_MIN_F32, vaddr=VGPR(0), vdata=VGPR(1))
        with self.assertRaises(ValueError):
            DSMinMaxInstruction(opcode=DSOpcode.DS_MIN_F32, vaddr=VGPR(0), vdata=VGPR(1), vdst=VGPR(2))

        # Returning requires vdst
        DSMinMaxInstruction(opcode=DSOpcode.DS_MIN_RTN_F32, vaddr=VGPR(0), vdata=VGPR(1), vdst=VGPR(2))
        with self.assertRaises(ValueError):
            DSMinMaxInstruction(opcode=DSOpcode.DS_MIN_RTN_F32, vaddr=VGPR(0), vdata=VGPR(1), vdst=None)

    def test_disasm_and_parse_roundtrip(self) -> None:
        test_cases = [
            DSMinMaxInstruction(DSOpcode.DS_MIN_F32, VGPR(2), VGPR(3)),
            DSMinMaxInstruction(DSOpcode.DS_MIN_F32, VGPR(2), VGPR(3), offset=OffsetImmediate(16)),
            DSMinMaxInstruction(DSOpcode.DS_MAX_F32, VGPR(10), VGPR(11), offset=OffsetImmediate(128)),
            DSMinMaxInstruction(DSOpcode.DS_MIN_RTN_F32, VGPR(4), VGPR(5), vdst=VGPR(0)),
            DSMinMaxInstruction(DSOpcode.DS_MAX_RTN_F32, VGPR(8), VGPR(9), vdst=VGPR(7), offset=OffsetImmediate(256)),
        ]
        for inst in test_cases:
            with self.subTest(inst=inst):
                asm = inst.disasm()
                parsed = DSMinMaxInstruction.parse(asm)
                self.assertEqual(inst, parsed)


if __name__ == "__main__":
    unittest.main()
