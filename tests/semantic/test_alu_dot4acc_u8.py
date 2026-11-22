from __future__ import annotations

"""Semantic interpreter tests for RDNA2 8-bit unsigned integer dot product with accumulation.

Instruction:
  v_dot4acc_u32_u8 (VALU VOP3 / VOP3P)

Semantics:
  D = sum(S0[i] * S1[i] for i in 0..3) + S2 (modulo 2^32)

Each 32-bit dword S0 and S1 contains four packed unsigned 8-bit integers:
  Byte 0: bits [7:0]
  Byte 1: bits [15:8]
  Byte 2: bits [23:16]
  Byte 3: bits [31:24]

Each byte pair product S0[i] * S1[i] produces an unsigned integer in [0, 65025].
The sum of the four products yields an unsigned integer in [0, 260100] (0x3F804).
Adding accumulator S2 yields the final 32-bit value. Arithmetic wraps strictly modulo 2^32
unless clamped to 0xFFFFFFFF.

Zero game data, standard library Python only, Python 3.9+ compatible.
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
MAX_BYTE_VAL = 255
MAX_BYTE_PRODUCT = 255 * 255  # 65025
MAX_DOT4_SUM = 4 * MAX_BYTE_PRODUCT  # 260100 (0x3F804)

# Mathematically verified test vectors: (S0, S1, S2_acc, expected_result)
KNOWN_VECTORS: list[tuple[int, int, int, int]] = [
    # Zero dot product, zero accumulator
    (0x00000000, 0x00000000, 0x00000000, 0),
    # Zero multipliers preserve accumulator
    (0x00000000, 0x12345678, 0xDEADBEEF, 0xDEADBEEF),
    (0x87654321, 0x00000000, 0xCAFEBABE, 0xCAFEBABE),
    # Sequential bytes: 4*8 + 3*7 + 2*6 + 1*5 = 32 + 21 + 12 + 5 = 70
    (0x01020304, 0x05060708, 0x00000000, 70),
    # Sequential bytes with accumulator: 70 + 1000 = 1070
    (0x01020304, 0x05060708, 0x000003E8, 1070),
    # Single-byte activation at lane 0: 255 * 1 = 255
    (0x000000FF, 0x00000001, 0x00000000, 255),
    # Single-byte square at lane 0: 255 * 255 = 65025
    (0x000000FF, 0x000000FF, 0x00000000, 65025),
    # Single-byte activation at lane 1: 255 * 2 = 510
    (0x0000FF00, 0x00000200, 0x00000000, 510),
    # Single-byte activation at lane 2: 255 * 3 = 765
    (0x00FF0000, 0x00030000, 0x00000000, 765),
    # Single-byte activation at lane 3: 255 * 4 = 1020
    (0xFF000000, 0x04000000, 0x00000000, 1020),
    # Broadcast 0xFF with broadcast 0x01: 4 * 255 = 1020
    (0xFFFFFFFF, 0x01010101, 0x00000000, 1020),
    # Maximum 8-bit unsigned dot product: 4 * (255 * 255) = 260100 (0x3F804)
    (0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 260100),
    # Maximum dot product with non-zero accumulator: 260100 + 1000000 = 1260100
    (0xFFFFFFFF, 0xFFFFFFFF, 1000000, 1260100),
    # Power-of-two mid bytes: 4 * (128 * 128) = 4 * 16384 = 65536
    (0x80808080, 0x80808080, 0x00000000, 65536),
    # 4 * (128 * 127) + 5 = 4 * 16256 + 5 = 65024 + 5 = 65029
    (0x80808080, 0x7F7F7F7F, 0x00000005, 65029),
    # Arbitrary structured mixed bytes:
    # 10*20 + 30*40 + 50*60 + 70*80 = 200 + 1200 + 3000 + 5600 = 10000
    (0x46321E0A, 0x503C2814, 0x00000000, 10000),
    (0x46321E0A, 0x503C2814, 0x10000000, 0x10002710),
    # Wrapping at exact boundary: 260100 + (2^32 - 260100) = 0 mod 2^32
    (0xFFFFFFFF, 0xFFFFFFFF, (MASK32 - MAX_DOT4_SUM + 1) & MASK32, 0),
    # Wrapping with remainder: 260100 + 0xFFFFFFFF = 260099
    (0xFFFFFFFF, 0xFFFFFFFF, MASK32, 260099),
]


def unpack_u8(dword: int) -> tuple[int, int, int, int]:
    """Unpack a 32-bit dword into four unsigned 8-bit integers (byte0..byte3)."""
    dword &= MASK32
    return (
        dword & MASK8,
        (dword >> 8) & MASK8,
        (dword >> 16) & MASK8,
        (dword >> 24) & MASK8,
    )


def pack_u8(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four 8-bit unsigned integers into a 32-bit dword."""
    return (
        (b0 & MASK8)
        | ((b1 & MASK8) << 8)
        | ((b2 & MASK8) << 16)
        | ((b3 & MASK8) << 24)
    ) & MASK32


def ref_dot4_u32_u8_sum(s0: int, s1: int) -> int:
    """Calculate the unaccumulated 4-lane 8-bit unsigned integer dot product sum."""
    b0 = unpack_u8(s0)
    b1 = unpack_u8(s1)
    return sum(x * y for x, y in zip(b0, b1))


def ref_dot4acc_u32_u8(s0: int, s1: int, s2: int = 0) -> int:
    """Reference oracle for RDNA2 v_dot4acc_u32_u8:

    D = (sum_{i=0}^3 (s0[i] * s1[i]) + s2) mod 2^32
    """
    prod_sum = ref_dot4_u32_u8_sum(s0, s1)
    return ((s2 & MASK32) + prod_sum) & MASK32


def ref_dot4acc_clamped_u32_u8(s0: int, s1: int, s2: int = 0) -> int:
    """Reference oracle with unsigned clamping/saturation at 0xFFFFFFFF."""
    prod_sum = ref_dot4_u32_u8_sum(s0, s1)
    total = (s2 & MASK32) + prod_sum
    return min(total, MASK32)


def ref_dot4_i32_i8(s0: int, s1: int, s2: int = 0) -> int:
    """Signed 8-bit dot product reference oracle (v_dot4_i32_i8).

    Used for divergence tests to confirm unsigned behavior differs from signed.
    """
    def s8(x: int) -> int:
        return x - 256 if (x & 0x80) else x

    b0 = unpack_u8(s0)
    b1 = unpack_u8(s1)
    prod_sum = sum(s8(x) * s8(y) for x, y in zip(b0, b1))
    return ((s2 & MASK32) + prod_sum) & MASK32


class ReferenceOracleTests(unittest.TestCase):
    """Verifies reference oracle against hand-calculated test vectors and algebraic invariants."""

    def test_unpack_pack_u8_roundtrip(self) -> None:
        """Verifies byte unpacking and packing round-trip across distinct word patterns."""
        patterns = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0xA55AA55A,
            0x01020304,
            0x80808080,
            0x7F8001FE,
            0xDEADBEEF,
        ]
        for val in patterns:
            with self.subTest(val=hex(val)):
                b0, b1, b2, b3 = unpack_u8(val)
                repacked = pack_u8(b0, b1, b2, b3)
                self.assertEqual(repacked, val)

    def test_known_test_vectors(self) -> None:
        """Verifies reference oracle reproduces all hand-calculated vectors."""
        for s0, s1, s2, expected in KNOWN_VECTORS:
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                got = ref_dot4acc_u32_u8(s0, s1, s2)
                self.assertEqual(got, expected, f"{s0:#x} . {s1:#x} + {s2:#x}")

    def test_commutativity_of_multipliers(self) -> None:
        """Multipliers are commutative: S0 . S1 + S2 == S1 . S0 + S2."""
        test_pairs = [
            (0x01020304, 0x05060708),
            (0xFFFFFFFF, 0x12345678),
            (0x80808080, 0x7F7F7F7F),
            (0xA1B2C3D4, 0x1E2D3C4B),
            (0x00FF00FF, 0xFF00FF00),
            (0xDEADBEEF, 0xCAFEBABE),
        ]
        for s0, s1 in test_pairs:
            for s2 in (0, 100, 0x12345678, 0xFFFFFFFF):
                with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                    self.assertEqual(
                        ref_dot4acc_u32_u8(s0, s1, s2),
                        ref_dot4acc_u32_u8(s1, s0, s2),
                    )

    def test_zero_multipliers_preserve_accumulator(self) -> None:
        """Zero multiplier on either operand preserves accumulator exactly."""
        vectors = [0x01020304, 0xFFFFFFFF, 0x80808080, 0x12345678, 0xCAFEBABE]
        for v in vectors:
            for acc in (0, 1, 0x7FFFFFFF, 0xFFFFFFFF, 0x10000000):
                with self.subTest(v=hex(v), acc=hex(acc)):
                    self.assertEqual(ref_dot4acc_u32_u8(0, v, acc), acc & MASK32)
                    self.assertEqual(ref_dot4acc_u32_u8(v, 0, acc), acc & MASK32)

    def test_identity_multiplier_sums_bytes(self) -> None:
        """Dot product with broadcast 0x01010101 computes the sum of the four unsigned bytes."""
        test_words = [
            (0x01020304, 1 + 2 + 3 + 4),
            (0xFFFFFFFF, 255 * 4),
            (0x10203040, 0x10 + 0x20 + 0x30 + 0x40),
            (0x000000FF, 255),
            (0x0000FF00, 255),
            (0x00FF0000, 255),
            (0xFF000000, 255),
            (0x80808080, 128 * 4),
        ]
        for word, expected_sum in test_words:
            with self.subTest(word=hex(word)):
                self.assertEqual(ref_dot4acc_u32_u8(word, 0x01010101, 0), expected_sum)
                self.assertEqual(ref_dot4acc_u32_u8(0x01010101, word, 0), expected_sum)

    def test_max_single_step_sum(self) -> None:
        """Verifies maximum product sum per step is exactly 4 * (255 * 255) = 260100."""
        self.assertEqual(MAX_DOT4_SUM, 260100)
        self.assertEqual(MAX_DOT4_SUM, 0x3F804)
        got = ref_dot4_u32_u8_sum(0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(got, 260100)

    def test_byte_lane_isolation(self) -> None:
        """Each of the 4 byte lanes computes independently without cross-lane bit bleed."""
        for byte_idx in range(4):
            for val_a in (1, 15, 128, 255):
                for val_b in (1, 10, 100, 255):
                    s0 = val_a << (8 * byte_idx)
                    s1 = val_b << (8 * byte_idx)
                    expected = val_a * val_b
                    with self.subTest(lane=byte_idx, a=val_a, b=val_b):
                        self.assertEqual(ref_dot4acc_u32_u8(s0, s1, 0), expected)

    def test_accumulator_linearity(self) -> None:
        """ref_dot4acc_u32_u8(s0, s1, s2) == (ref_dot4acc_u32_u8(s0, s1, 0) + s2) mod 2^32."""
        s0, s1 = 0x11223344, 0x55667788
        base_dot = ref_dot4acc_u32_u8(s0, s1, 0)
        for addend in (0, 1, 0x1000, 0x7FFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE):
            with self.subTest(addend=hex(addend)):
                want = (base_dot + addend) & MASK32
                got = ref_dot4acc_u32_u8(s0, s1, addend)
                self.assertEqual(got, want)

    def test_unsigned_vs_signed_fundamental_divergence(self) -> None:
        """Unsigned dot product behaves fundamentally differently from signed dot product."""
        divergent_pairs = [
            (0x000000FF, 0x00000001, 0),  # 255 * 1 = 255 vs -1 * 1 = -1
            (0x000000FF, 0x000000FF, 0),  # 255 * 255 = 65025 vs (-1)*(-1) = 1
            (0xFFFFFFFF, 0x01010101, 0),  # 4 * 255 = 1020 vs 4 * (-1) = -4
            (0x80808080, 0x7F7F7F7F, 0),  # 4 * (128*127) = 65024 vs 4 * (-128*127) = -65024
            (0xFFFFFFFF, 0xFFFFFFFF, 0),  # 4 * 65025 = 260100 vs 4 * 1 = 4
        ]
        for s0, s1, s2 in divergent_pairs:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                u_res = ref_dot4acc_u32_u8(s0, s1, s2)
                s_res = ref_dot4_i32_i8(s0, s1, s2)
                self.assertNotEqual(u_res, s_res)

    def test_pseudorandom_fuzz_invariants(self) -> None:
        """Deterministic fuzzing over 128 pseudo-random word triples checking invariants."""
        rng = random.Random(0x444F5441)  # ASCII 'DOTA'
        for _ in range(128):
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            s2 = rng.getrandbits(32)

            # Commutativity
            self.assertEqual(ref_dot4acc_u32_u8(s0, s1, s2), ref_dot4acc_u32_u8(s1, s0, s2))

            # Manual byte arithmetic verification
            b0 = unpack_u8(s0)
            b1 = unpack_u8(s1)
            manual_sum = ((s2 & MASK32) + b0[0]*b1[0] + b0[1]*b1[1] + b0[2]*b1[2] + b0[3]*b1[3]) & MASK32
            self.assertEqual(ref_dot4acc_u32_u8(s0, s1, s2), manual_sum)

            # Single-step dot4 sum bounded by MAX_DOT4_SUM
            prod_sum = ref_dot4_u32_u8_sum(s0, s1)
            self.assertGreaterEqual(prod_sum, 0)
            self.assertLessEqual(prod_sum, MAX_DOT4_SUM)


if __name__ == "__main__":
    unittest.main()
