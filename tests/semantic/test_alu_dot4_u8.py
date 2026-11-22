from __future__ import annotations

"""Semantic interpreter tests for RDNA2 unsigned 8-bit integer dot products.

Covers:
  - v_dot4_u32_u8 (VOP3/VOP3P): D = sum(S0[i] * S1[i] for i in 0..3) + S2
  - v_dot4acc_u32_u8 / v_dot4c_u32_u8: accumulating forms where an accumulator
    is added to the sum of four 8-bit unsigned products, kept modulo 2^32.

ISA rule (RDNA2):
  Each 32-bit dword S0 and S1 contains four packed unsigned 8-bit integers:
    Byte 0: bits [7:0]
    Byte 1: bits [15:8]
    Byte 2: bits [23:16]
    Byte 3: bits [31:24]
  Each byte product S0[i] * S1[i] produces an unsigned integer in [0, 65025].
  The four products sum to [0, 260100], which is added to a 32-bit accumulator S2.
  Arithmetic wraps modulo 2^32 without saturation unless clamping is requested.
  With clamp, the hardware saturates to 0xFFFFFFFF (unsupported by translator).

Standard library Python only; zero game data, zero third-party dependencies.
"""

import os
import random
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MASK32 = 0xFFFFFFFF
MASK8 = 0xFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

# Measured and mathematically verified reference test vectors:
# (S0, S1, S2_acc, expected_result)
KNOWN_VECTORS: list[tuple[int, int, int, int]] = [
    # 4*8 + 3*7 + 2*6 + 1*5 = 32 + 21 + 12 + 5 = 70
    (0x01020304, 0x05060708, 0x00000000, 70),
    # 70 + 1000 = 1070
    (0x01020304, 0x05060708, 0x000003E8, 1070),
    # Zero multiplier on S0
    (0x00000000, 0x12345678, 0xDEADBEEF, 0xDEADBEEF),
    # Zero multiplier on S1
    (0x12345678, 0x00000000, 0xCAFEBABE, 0xCAFEBABE),
    # Single byte at index 0: 255 * 1 = 255 (diverges strongly from signed i8 which gives -1)
    (0x000000FF, 0x00000001, 0x00000000, 255),
    # Single byte at index 0 squared: 255 * 255 = 65025
    (0x000000FF, 0x000000FF, 0x00000000, 65025),
    # Single byte at index 1: 255 * 2 = 510
    (0x0000FF00, 0x00000200, 0x00000000, 510),
    # Single byte at index 2: 255 * 3 = 765
    (0x00FF0000, 0x00030000, 0x00000000, 765),
    # Single byte at index 3: 255 * 4 = 1020
    (0xFF000000, 0x04000000, 0x00000000, 1020),
    # Identity byte broadcast: 4 * (255 * 1) = 1020
    (0xFFFFFFFF, 0x01010101, 0x00000000, 1020),
    # Maximum 8-bit unsigned product: 4 * (255 * 255) = 260100 (0x3F804)
    (0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 260100),
    # Mid-range power-of-two bytes: 4 * (128 * 128) = 65536 (0x10000)
    (0x80808080, 0x80808080, 0x00000000, 65536),
    # 4 * (128 * 127) + 5 = 65024 + 5 = 65029
    (0x80808080, 0x7F7F7F7F, 0x00000005, 65029),
    # Mixed values across all 4 bytes:
    # 10*20 + 30*40 + 50*60 + 70*80 = 200 + 1200 + 3000 + 5600 = 10000
    (0x46321E0A, 0x503C2814, 0x00000000, 10000),
    # High accumulator without overflow:
    (0x46321E0A, 0x503C2814, 0x10000000, 0x10002710),
    # Wrap-around boundary: 260100 + (0xFFFFFFFF - 260100 + 1) = 0
    (0xFFFFFFFF, 0xFFFFFFFF, (MASK32 - 260100 + 1) & MASK32, 0),
    # Wrap-around with non-zero remainder
    (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 260099),
]


def unpack_u8(dword: int) -> tuple[int, int, int, int]:
    """Unpack a 32-bit dword into four unsigned 8-bit integers (byte0, byte1, byte2, byte3)."""
    dword &= MASK32
    return (
        dword & MASK8,
        (dword >> 8) & MASK8,
        (dword >> 16) & MASK8,
        (dword >> 24) & MASK8,
    )


def pack_u8(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four 8-bit unsigned integers into a 32-bit dword."""
    return ((b0 & MASK8) | ((b1 & MASK8) << 8) | ((b2 & MASK8) << 16) | ((b3 & MASK8) << 24)) & MASK32


def ref_dot4_u32_u8(s0: int, s1: int, s2: int = 0) -> int:
    """Reference oracle for RDNA2 unsigned 8-bit dot product (v_dot4_u32_u8).

    Calculates:
      D = (s2 + sum(s0[i] * s1[i] for i in 0..3)) mod 2^32
    where each s0[i] and s1[i] is an unsigned 8-bit integer in [0, 255].
    """
    b0 = unpack_u8(s0)
    b1 = unpack_u8(s1)
    prod_sum = sum(x * y for x, y in zip(b0, b1))
    return ((s2 & MASK32) + prod_sum) & MASK32


def ref_dot4acc_u32_u8(s0: int, s1: int, s2: int) -> int:
    """Reference oracle for accumulating form v_dot4acc_u32_u8 with explicit accumulator."""
    return ref_dot4_u32_u8(s0, s1, s2)


def ref_dot4_clamp_u32_u8(s0: int, s1: int, s2: int = 0) -> int:
    """Reference oracle with unsigned clamping/saturation at 0xFFFFFFFF."""
    b0 = unpack_u8(s0)
    b1 = unpack_u8(s1)
    total = (s2 & MASK32) + sum(x * y for x, y in zip(b0, b1))
    return min(total, MASK32)


def ref_dot4_i32_i8(s0: int, s1: int, s2: int = 0) -> int:
    """Signed reference oracle for signed 8-bit dot product (v_dot4_i32_i8).

    Used for divergence comparisons against unsigned dot products.
    """
    def s8(x: int) -> int:
        return x - 256 if (x & 0x80) else x

    b0 = unpack_u8(s0)
    b1 = unpack_u8(s1)
    prod_sum = sum(s8(x) * s8(y) for x, y in zip(b0, b1))
    return ((s2 & MASK32) + prod_sum) & MASK32


class ReferenceOracleTests(unittest.TestCase):
    """Verifies reference oracle against known hand-calculated vectors and algebraic laws."""

    def test_unpack_pack_u8_roundtrip(self) -> None:
        """Verifies byte unpacking and packing round-trip across test patterns."""
        patterns = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0xA55AA55A,
            0x01020304,
            0x80808080,
            0x7F8001FE,
        ]
        for val in patterns:
            with self.subTest(val=hex(val)):
                b0, b1, b2, b3 = unpack_u8(val)
                repacked = pack_u8(b0, b1, b2, b3)
                self.assertEqual(repacked, val)

    def test_reference_known_vectors(self) -> None:
        """Verifies reference oracle reproduces all known test vectors."""
        for s0, s1, s2, expected in KNOWN_VECTORS:
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                got = ref_dot4_u32_u8(s0, s1, s2)
                self.assertEqual(got, expected, f"{s0:#x} . {s1:#x} + {s2:#x}")

    def test_commutativity_of_multipliers(self) -> None:
        """Dot product of S0 and S1 is commutative: S0 . S1 == S1 . S0."""
        test_pairs = [
            (0x01020304, 0x05060708),
            (0xFFFFFFFF, 0x12345678),
            (0x80808080, 0x7F7F7F7F),
            (0xA1B2C3D4, 0x1E2D3C4B),
            (0x00FF00FF, 0xFF00FF00),
        ]
        for s0, s1 in test_pairs:
            for s2 in (0, 100, 0x12345678, 0xFFFFFFFF):
                with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                    self.assertEqual(
                        ref_dot4_u32_u8(s0, s1, s2),
                        ref_dot4_u32_u8(s1, s0, s2),
                    )

    def test_zero_identities(self) -> None:
        """Dot product with zero operand returns the accumulator exactly."""
        vectors = [0x01020304, 0xFFFFFFFF, 0x80808080, 0x12345678]
        for v in vectors:
            for acc in (0, 1, 0x7FFFFFFF, 0xFFFFFFFF):
                with self.subTest(v=hex(v), acc=hex(acc)):
                    self.assertEqual(ref_dot4_u32_u8(0, v, acc), acc & MASK32)
                    self.assertEqual(ref_dot4_u32_u8(v, 0, acc), acc & MASK32)

    def test_identity_broadcast_sums_bytes(self) -> None:
        """Dot product with 0x01010101 computes the sum of the four unsigned bytes."""
        test_words = [
            (0x01020304, 1 + 2 + 3 + 4),
            (0xFFFFFFFF, 255 * 4),
            (0x10203040, 0x10 + 0x20 + 0x30 + 0x40),
            (0x000000FF, 255),
            (0x0000FF00, 255),
            (0x00FF0000, 255),
            (0xFF000000, 255),
        ]
        for word, expected_sum in test_words:
            with self.subTest(word=hex(word)):
                self.assertEqual(ref_dot4_u32_u8(word, 0x01010101, 0), expected_sum)
                self.assertEqual(ref_dot4_u32_u8(0x01010101, word, 0), expected_sum)

    def test_byte_lane_isolation(self) -> None:
        """Verifies each byte lane operates independently without bleed across byte boundaries."""
        for byte_idx in range(4):
            for val_a in (1, 15, 128, 255):
                for val_b in (1, 10, 100, 255):
                    s0 = val_a << (8 * byte_idx)
                    s1 = val_b << (8 * byte_idx)
                    expected = val_a * val_b
                    with self.subTest(lane=byte_idx, a=val_a, b=val_b):
                        self.assertEqual(ref_dot4_u32_u8(s0, s1, 0), expected)

    def test_addend_linearity(self) -> None:
        """v_dot4_u32_u8(S0, S1, S2) == (v_dot4_u32_u8(S0, S1, 0) + S2) mod 2^32."""
        s0, s1 = 0x11223344, 0x55667788
        base_dot = ref_dot4_u32_u8(s0, s1, 0)
        for addend in (0, 1, 0x1000, 0x7FFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE):
            with self.subTest(addend=hex(addend)):
                want = (base_dot + addend) & MASK32
                got = ref_dot4_u32_u8(s0, s1, addend)
                self.assertEqual(got, want)

    def test_unsigned_vs_signed_divergence(self) -> None:
        """Unsigned dot product behaves fundamentally differently from signed dot product."""
        # When high bit (bit 7) of bytes is set, signed treats as negative, unsigned as >= 128
        divergent_pairs = [
            # 0xFF * 0x01: unsigned is 255, signed is -1 (0xFFFFFFFF)
            (0x000000FF, 0x00000001, 0),
            # 0xFF * 0xFF: unsigned is 65025, signed is 1
            (0x000000FF, 0x000000FF, 0),
            # 0xFFFFFFFF . 0x01010101: unsigned is 1020, signed is -4
            (0xFFFFFFFF, 0x01010101, 0),
            # 0x80808080 . 0x7F7F7F7F: unsigned is 65024, signed is -65024 (0xFFFF0200)
            (0x80808080, 0x7F7F7F7F, 0),
        ]
        for s0, s1, s2 in divergent_pairs:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                u_res = ref_dot4_u32_u8(s0, s1, s2)
                s_res = ref_dot4_i32_i8(s0, s1, s2)
                self.assertNotEqual(u_res, s_res)

    def test_pseudorandom_fuzz_properties(self) -> None:
        """Deterministic property fuzzing over 100 pseudo-random 32-bit words."""
        rng = random.Random(0x444F5434)  # ASCII 'DOT4'
        for _ in range(100):
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            s2 = rng.getrandbits(32)

            # Commutativity
            self.assertEqual(ref_dot4_u32_u8(s0, s1, s2), ref_dot4_u32_u8(s1, s0, s2))
            # Distributivity of dot product over individual byte terms
            b0 = unpack_u8(s0)
            b1 = unpack_u8(s1)
            manual_sum = ((s2 & MASK32) + b0[0]*b1[0] + b0[1]*b1[1] + b0[2]*b1[2] + b0[3]*b1[3]) & MASK32
            self.assertEqual(ref_dot4_u32_u8(s0, s1, s2), manual_sum)
            # Accumulator oracle equivalence
            self.assertEqual(ref_dot4acc_u32_u8(s0, s1, s2), ref_dot4_u32_u8(s0, s1, s2))


class WavefrontDot4Interpreter:
    """Simulates an RDNA2 SIMD wavefront executing VALU/SALU dot-product instructions.

    Models:
      - Wave32 (32 lanes) and Wave64 (64 lanes).
      - Per-lane VGPR files: vgprs[reg][lane].
      - Uniform SGPR file: sgprs[reg].
      - EXEC mask: 32-bit or 64-bit integer mask.
      - Instructions: v_mov_b32, s_mov_b32, s_mov_b64, v_dot4_u32_u8, v_dot4acc_u32_u8, v_dot4c_u32_u8.
    """

    def __init__(self, wave_size: int = 32):
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave_size {wave_size}; must be 32 or 64")
        self.wave_size = wave_size
        self.vgprs = [[0] * wave_size for _ in range(256)]
        self.sgprs = [0] * 106
        self.exec_mask = (1 << wave_size) - 1

    def set_vgpr(self, reg: int, lane: int, val: int) -> None:
        self.vgprs[reg][lane] = val & MASK32

    def get_vgpr(self, reg: int, lane: int) -> int:
        return self.vgprs[reg][lane] & MASK32

    def set_sgpr(self, reg: int, val: int) -> None:
        self.sgprs[reg] = val & MASK32

    def get_sgpr(self, reg: int) -> int:
        return self.sgprs[reg] & MASK32

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def parse_operand(self, token: str, lane: int) -> int:
        token = token.strip().rstrip(",")
        lower = token.lower()
        if lower.startswith("v") and lower[1:].isdigit():
            idx = int(lower[1:])
            if 0 <= idx < 256:
                return self.vgprs[idx][lane] & MASK32
            raise IndexError(f"VGPR index out of range: {token}")
        if lower.startswith("s") and lower[1:].isdigit():
            idx = int(lower[1:])
            if 0 <= idx < 106:
                return self.sgprs[idx] & MASK32
            raise IndexError(f"SGPR index out of range: {token}")
        if lower.startswith("0x") or lower.startswith("-0x"):
            return int(token, 16) & MASK32
        return int(token) & MASK32

    def execute_line(self, line: str) -> None:
        line = line.split(";", 1)[0].split("//", 1)[0].strip()
        if not line:
            return
        parts = line.split(None, 1)
        opcode = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""

        tokens = [t.strip() for t in args_str.split(",") if t.strip()]

        if opcode == "s_mov_b32":
            dst = tokens[0].lower()
            if not (dst.startswith("s") and dst[1:].isdigit()):
                raise ValueError(f"s_mov_b32 dst must be SGPR, got {tokens[0]}")
            s_idx = int(dst[1:])
            val = self.parse_operand(tokens[1], 0)
            self.sgprs[s_idx] = val

        elif opcode in ("v_mov_b32", "v_mov_b32_e32"):
            dst = tokens[0].lower()
            if not (dst.startswith("v") and dst[1:].isdigit()):
                raise ValueError(f"v_mov_b32 dst must be VGPR, got {tokens[0]}")
            v_idx = int(dst[1:])
            for lane in range(self.wave_size):
                if (self.exec_mask >> lane) & 1:
                    val = self.parse_operand(tokens[1], lane)
                    self.vgprs[v_idx][lane] = val

        elif opcode in ("v_dot4_u32_u8", "v_dot4acc_u32_u8"):
            # VOP3/VOP3P: dst, src0, src1, src2
            dst = tokens[0].lower()
            if not (dst.startswith("v") and dst[1:].isdigit()):
                raise ValueError(f"{opcode} dst must be VGPR, got {tokens[0]}")
            v_idx = int(dst[1:])
            for lane in range(self.wave_size):
                if (self.exec_mask >> lane) & 1:
                    s0 = self.parse_operand(tokens[1], lane)
                    s1 = self.parse_operand(tokens[2], lane)
                    s2 = self.parse_operand(tokens[3], lane) if len(tokens) > 3 else 0
                    self.vgprs[v_idx][lane] = ref_dot4_u32_u8(s0, s1, s2)

        elif opcode == "v_dot4c_u32_u8":
            # VOP2: dst, src0, src1 (dst is accumulator)
            dst = tokens[0].lower()
            if not (dst.startswith("v") and dst[1:].isdigit()):
                raise ValueError(f"v_dot4c_u32_u8 dst must be VGPR, got {tokens[0]}")
            v_idx = int(dst[1:])
            for lane in range(self.wave_size):
                if (self.exec_mask >> lane) & 1:
                    s0 = self.parse_operand(tokens[1], lane)
                    s1 = self.parse_operand(tokens[2], lane)
                    old_acc = self.vgprs[v_idx][lane]
                    self.vgprs[v_idx][lane] = ref_dot4_u32_u8(s0, s1, old_acc)

        elif opcode == "s_endpgm":
            return
        else:
            raise NotImplementedError(f"Unsupported opcode: {opcode}")

    def execute_program(self, program: str | list[str]) -> None:
        lines = program if isinstance(program, list) else program.strip().splitlines()
        for line in lines:
            self.execute_line(line)


class WavefrontInterpreterTests(unittest.TestCase):
    """Verifies SIMD multi-lane wavefront execution and accumulation semantics."""

    def test_wave32_uniform_execution(self) -> None:
        """32-lane SIMD execution under full EXEC mask with divergent per-lane data."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        for lane in range(32):
            s0 = ((lane + 1) * 0x01020304) & MASK32
            s1 = 0x05060708
            acc = lane * 10
            interp.set_vgpr(1, lane, s0)
            interp.set_vgpr(2, lane, s1)
            interp.set_vgpr(3, lane, acc)

        interp.execute_line("v_dot4_u32_u8 v0, v1, v2, v3")

        for lane in range(32):
            s0 = ((lane + 1) * 0x01020304) & MASK32
            s1 = 0x05060708
            acc = lane * 10
            expected = ref_dot4_u32_u8(s0, s1, acc)
            self.assertEqual(interp.get_vgpr(0, lane), expected)

    def test_wave64_uniform_execution(self) -> None:
        """64-lane SIMD execution covering the entire Wave64 execution space."""
        interp = WavefrontDot4Interpreter(wave_size=64)
        for lane in range(64):
            s0 = (0x01010101 * lane) & MASK32
            s1 = 0x02020202
            interp.set_vgpr(1, lane, s0)
            interp.set_vgpr(2, lane, s1)
            interp.set_vgpr(3, lane, 0x1000)

        interp.execute_line("v_dot4acc_u32_u8 v8, v1, v2, v3")

        for lane in range(64):
            s0 = (0x01010101 * lane) & MASK32
            s1 = 0x02020202
            expected = ref_dot4_u32_u8(s0, s1, 0x1000)
            self.assertEqual(interp.get_vgpr(8, lane), expected)

    def test_v_dot4c_u32_u8_accumulator_reads_destination(self) -> None:
        """v_dot4c_u32_u8 (VOP2) uses destination register's prior value as accumulator."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        # Initialize v0 destination accumulator to 500 across all lanes
        for lane in range(32):
            interp.set_vgpr(0, lane, 500)
            interp.set_vgpr(1, lane, 0x01020304)
            interp.set_vgpr(2, lane, 0x05060708)

        # First dot product adds 70 to 500 = 570
        interp.execute_line("v_dot4c_u32_u8 v0, v1, v2")
        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), 570)

        # Second successive dot product adds another 70 = 640
        interp.execute_line("v_dot4c_u32_u8 v0, v1, v2")
        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), 640)

    def test_accumulator_chain_loop(self) -> None:
        """Simulate an inner-loop dot-product reduction accumulating across iterations."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        running_totals = [0] * 32
        steps = [
            (0x01010101, 0x02020202),  # 4 * 2 = 8
            (0x03030303, 0x04040404),  # 4 * 12 = 48
            (0x05050505, 0x06060606),  # 4 * 30 = 120
            (0x07070707, 0x08080808),  # 4 * 56 = 224
        ]
        for s0, s1 in steps:
            for lane in range(32):
                interp.set_vgpr(1, lane, s0)
                interp.set_vgpr(2, lane, s1)
                running_totals[lane] = (running_totals[lane] + ref_dot4_u32_u8(s0, s1, 0)) & MASK32
            interp.execute_line("v_dot4c_u32_u8 v0, v1, v2")

        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), running_totals[lane])
            self.assertEqual(interp.get_vgpr(0, lane), 8 + 48 + 120 + 224)

    def test_wave_lane_divergence_exec_mask(self) -> None:
        """Inactive lanes under partial EXEC mask must strictly preserve destination registers."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        sentinel = 0xDEADBEEF

        # Prepopulate destination register v5 with sentinel
        for lane in range(32):
            interp.set_vgpr(5, lane, sentinel)
            interp.set_vgpr(1, lane, 0x01020304)
            interp.set_vgpr(2, lane, 0x05060708)
            interp.set_vgpr(3, lane, 100)

        # Set EXEC to lower 16 lanes only (0x0000FFFF)
        interp.set_exec(0x0000FFFF)
        interp.execute_line("v_dot4_u32_u8 v5, v1, v2, v3")

        # Active lanes (0..15) must be updated to 170 (70 + 100)
        # Inactive lanes (16..31) must strictly preserve sentinel 0xDEADBEEF
        for lane in range(32):
            if lane < 16:
                self.assertEqual(interp.get_vgpr(5, lane), 170)
            else:
                self.assertEqual(interp.get_vgpr(5, lane), sentinel)

    def test_odd_even_lane_divergence(self) -> None:
        """Divergent execution between alternating even and odd lanes."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        for lane in range(32):
            interp.set_vgpr(10, lane, 0)
            interp.set_vgpr(1, lane, 0x01010101)  # dot = 4 * 10 = 40
            interp.set_vgpr(2, lane, 0x0A0A0A0A)
            interp.set_vgpr(3, lane, 0x02020202)  # dot = 4 * 40 = 160
            interp.set_vgpr(4, lane, 0x14141414)

        # Step 1: Execute even lanes with (v1 . v2)
        interp.set_exec(0x55555555)  # 01010101... (even lanes 0, 2, 4...)
        interp.execute_line("v_dot4_u32_u8 v10, v1, v2, 0")

        # Step 2: Execute odd lanes with (v3 . v4)
        interp.set_exec(0xAAAAAAAA)  # 10101010... (odd lanes 1, 3, 5...)
        interp.execute_line("v_dot4_u32_u8 v10, v3, v4, 0")

        for lane in range(32):
            expected = 40 if (lane % 2 == 0) else 160
            self.assertEqual(interp.get_vgpr(10, lane), expected)

    def test_single_lane_active(self) -> None:
        """Only a single lane active (lane 13), ensuring no collateral writes."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        for lane in range(32):
            interp.set_vgpr(7, lane, 0xBAADF00D)
            interp.set_vgpr(1, lane, 0x01020304)
            interp.set_vgpr(2, lane, 0x05060708)

        target_lane = 13
        interp.set_exec(1 << target_lane)
        interp.execute_line("v_dot4_u32_u8 v7, v1, v2, 0")

        for lane in range(32):
            if lane == target_lane:
                self.assertEqual(interp.get_vgpr(7, lane), 70)
            else:
                self.assertEqual(interp.get_vgpr(7, lane), 0xBAADF00D)

    def test_scalar_source_broadcast(self) -> None:
        """SGPR sources broadcast uniform values across all lanes."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        interp.set_sgpr(4, 0x05060708)
        interp.set_sgpr(5, 5000)

        for lane in range(32):
            interp.set_vgpr(1, lane, 0x01020304)

        interp.execute_line("v_dot4_u32_u8 v0, v1, s4, s5")

        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), 5070)


if __name__ == "__main__":
    unittest.main()
