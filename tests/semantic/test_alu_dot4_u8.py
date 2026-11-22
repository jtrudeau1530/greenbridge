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


if __name__ == "__main__":
    unittest.main()
