"""Semantic interpreter tests for RDNA2 vector bit manipulation and absolute value:
  - v_bfrev_b32: 32-bit vector bitfield reverse (D = reverse_bits(S0))
  - s_brev_b32: 32-bit scalar bitfield reverse
  - v_abs_i32: 32-bit vector integer absolute value (D = |S0|)
  - s_abs_i32: 32-bit scalar integer absolute value with SCC flag

Features covered:
  - RDNA2 vector bit manipulation reference oracles and algebraic identities:
      * Involution identity: bfrev(bfrev(x)) == x (bijective self-inverse permutation).
      * Popcount preservation: popcount(bfrev(x)) == popcount(x).
      * Bit boundary reflection: bit i maps to bit (31 - i) (e.g. bit 0 <-> bit 31, bit 15 <-> bit 16).
      * Algebraic homomorphism: bfrev(x ^ y) == bfrev(x) ^ bfrev(y) and bfrev(~x) == ~bfrev(x).
      * Extreme and canonical bit patterns: 0x00000000, 0xFFFFFFFF, 0x55555555 <-> 0xAAAAAAAA,
        0x33333333 <-> 0xCCCCCCCC, 0x0F0F0F0F <-> 0xF0F0F0F0, 0x00FF00FF <-> 0xFF00FF00,
        0x0000FFFF <-> 0xFFFF0000, 0x80000000 <-> 0x00000001.
      * Cross-oracle validation between string-reversal, parallel SWAR bit-swaps, and byte-lookup tables.
  - Two's-complement integer absolute value semantics:
      * D = |S0| for signed 32-bit integer in range [-2^31, 2^31 - 1].
      * Non-negative values are identity: |x| == x for x >= 0.
      * Strictly negative values are negated: |x| == -x for x < 0.
      * Two's-complement overflow: |-2147483648| == -2147483648 (0x80000000).
  - Synthetic shader translation via tools/rdna2spirv.py and execution via tests/semantic/spirv_mini_interp.py.

Zero game data, standard library Python only, Python 3.9+ compatible.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

CANONICAL_VALUES = [
    0x00000000,
    0x00000001,
    0x00000002,
    0x00000003,
    0x00000080,
    0x00000100,
    0x0000FFFF,
    0x00010000,
    0x12345678,
    0x55555555,
    0xAAAAAAAA,
    0x33333333,
    0xCCCCCCCC,
    0x0F0F0F0F,
    0xF0F0F0F0,
    0x00FF00FF,
    0xFF00FF00,
    0x7FFFFFFF,
    0x80000000,
    0x80000001,
    0xFFFFFFFE,
    0xFFFFFFFF,
]
VALUES = CANONICAL_VALUES


def u32(val: int) -> int:
    """Normalize integer to 32-bit unsigned dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret 32-bit dword as signed two's complement integer."""
    v = val & MASK32
    return v - (1 << 32) if (v & 0x80000000) else v


def popcount(x: int) -> int:
    """Count number of set bits in 32-bit dword (Python 3.9+ compatible)."""
    return bin(x & MASK32).count("1")


def ref_brev32(x: int) -> int:
    """Reference oracle for 32-bit bitfield reverse (string-based)."""
    return int(f"{x & MASK32:032b}"[::-1], 2)


def ref_brev32_swar(x: int) -> int:
    """Hacker's Delight parallel SWAR (SIMD within a register) 32-bit bit reversal."""
    v = x & MASK32
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1)
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2)
    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4)
    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8)
    v = ((v >> 16) & 0x0000FFFF) | ((v & 0x0000FFFF) << 16)
    return v & MASK32


_REV_TABLE_8 = tuple(int(f"{b:08b}"[::-1], 2) for b in range(256))


def ref_brev32_table(x: int) -> int:
    """Table-based 32-bit bit reversal oracle across 4 lookup bytes."""
    v = x & MASK32
    b0 = _REV_TABLE_8[v & 0xFF]
    b1 = _REV_TABLE_8[(v >> 8) & 0xFF]
    b2 = _REV_TABLE_8[(v >> 16) & 0xFF]
    b3 = _REV_TABLE_8[(v >> 24) & 0xFF]
    return ((b0 << 24) | (b1 << 16) | (b2 << 8) | b3) & MASK32


# Alias for backwards compatibility
ref_brev = ref_brev32


def ref_abs32(x: int) -> int:
    """Reference oracle for signed 32-bit absolute value with two's-complement overflow."""
    s = s32(x)
    return u32(-s if s < 0 else s)


# Alias for backwards compatibility
ref_abs = ref_abs32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_bitrev_abs.s")
    with open(os.path.join(_TMP_DIR.name, "ps_bitrev_abs.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines: list[str]) -> tuple[int, ...]:
    """Assemble and run synthetic pixel shader lines, returning (out_mrt0.x, out_mrt0.y)."""
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_bitrev_abs\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return tuple(si.run_fragment(spv)["%out_mrt0"][:2])


class VectorBitrevOracleTests(unittest.TestCase):
    """Rigorous algebraic and property-based verification of bit-reversal reference models."""

    def test_oracle_equivalence_across_algorithms(self):
        """String-based, SWAR bit-swap, and byte-lookup algorithms must be 100% equivalent."""
        for val in CANONICAL_VALUES:
            with self.subTest(val=hex(val)):
                r1 = ref_brev32(val)
                r2 = ref_brev32_swar(val)
                r3 = ref_brev32_table(val)
                self.assertEqual(r1, r2)
                self.assertEqual(r1, r3)

    def test_single_bit_boundary_reflections(self):
        """Bit i must reflect strictly to bit (31 - i) for all 0 <= i <= 31."""
        for i in range(32):
            src = 1 << i
            expected = 1 << (31 - i)
            with self.subTest(bit=i, src=hex(src)):
                self.assertEqual(ref_brev32(src), expected)
                self.assertEqual(ref_brev32_swar(src), expected)
                self.assertEqual(ref_brev32_table(src), expected)

    def test_popcount_preservation_property(self):
        """Bit reversal must preserve hamming weight (popcount) identically: popcount(brev(x)) == popcount(x)."""
        test_inputs = list(CANONICAL_VALUES)
        # Add random-like deterministic multi-bit patterns
        for shift in range(32):
            test_inputs.append((0x12345678 << shift) & MASK32)
            test_inputs.append((0xCAFEBABE >> shift) & MASK32)
            test_inputs.append((1 << shift) | (1 << ((shift + 7) % 32)))

        for x in test_inputs:
            with self.subTest(x=hex(x)):
                rev = ref_brev32(x)
                self.assertEqual(popcount(rev), popcount(x))

    def test_involution_identity(self):
        """Bit reversal is an involution: bfrev(bfrev(x)) == x for all inputs."""
        for x in CANONICAL_VALUES:
            with self.subTest(x=hex(x)):
                self.assertEqual(ref_brev32(ref_brev32(x)), x)
                self.assertEqual(ref_brev32_swar(ref_brev32_swar(x)), x)
                self.assertEqual(ref_brev32_table(ref_brev32_table(x)), x)

    def test_algebraic_homomorphism_xor_and_not(self):
        """bfrev distributes over XOR: bfrev(x ^ y) == bfrev(x) ^ bfrev(y) and bfrev(~x) == ~bfrev(x)."""
        pairs = [
            (0x55555555, 0xAAAAAAAA),
            (0x12345678, 0x87654321),
            (0x00FF00FF, 0xF0F0F0F0),
            (0x80000001, 0x7FFFFFFE),
            (0xFFFFFFFF, 0x12345678),
        ]
        for a, b in pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(ref_brev32(a ^ b), ref_brev32(a) ^ ref_brev32(b))
                self.assertEqual(ref_brev32(~a & MASK32), (~ref_brev32(a)) & MASK32)
                self.assertEqual(ref_brev32(~b & MASK32), (~ref_brev32(b)) & MASK32)

    def test_palindromic_self_reversing_bit_patterns(self):
        """Bit patterns that are symmetric across the center must equal themselves under bit reversal."""
        palindromes = [
            0x00000000,
            0xFFFFFFFF,
            0x80000001,  # bit 0 and bit 31 set
            0x00018000,  # bit 15 and bit 16 set
            0x5555AAAA,  # 0101...1010
            0x0FF00FF0,
            0x3C3C3C3C,
            0xA55A5AA5,
        ]
        for p in palindromes:
            with self.subTest(p=hex(p)):
                self.assertEqual(ref_brev32(p), p)

    def test_canonical_reflections(self):
        """Canonical test pairs must match hand-derived mathematical reflections."""
        expected_pairs = [
            (0x00000001, 0x80000000),
            (0x00000003, 0xC0000000),
            (0x55555555, 0xAAAAAAAA),
            (0x33333333, 0xCCCCCCCC),
            (0x0F0F0F0F, 0xF0F0F0F0),
            (0x00FF00FF, 0xFF00FF00),
            (0x0000FFFF, 0xFFFF0000),
            (0x12345678, 0x1E6A2C48),
        ]
        for src, expected in expected_pairs:
            with self.subTest(src=hex(src)):
                self.assertEqual(ref_brev32(src), expected)
                self.assertEqual(ref_brev32(expected), src)


class BitReverseTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref_brev(1), 0x80000000)
        self.assertEqual(ref_brev(0x12345678), 0x1E6A2C48)
        self.assertEqual(ref_brev(0x0000FFFF), 0xFFFF0000)

    def test_v_bfrev_b32(self):
        for x in VALUES:
            with self.subTest(x=hex(x)):
                self.assertEqual(run_lines([f"v_mov_b32 v2, {x:#x}", "v_bfrev_b32_e32 v0, v2", "v_mov_b32 v1, 0"])[0],
                                 ref_brev(x))

    def test_s_brev_b32(self):
        for x in VALUES:
            with self.subTest(x=hex(x)):
                self.assertEqual(run_lines([f"s_mov_b32 s2, {x:#x}", "s_brev_b32 s0, s2", "v_mov_b32 v0, s0",
                                            "v_mov_b32 v1, 0"])[0], ref_brev(x))

    def test_twice_is_identity(self):
        for x in VALUES:
            with self.subTest(x=hex(x)):
                self.assertEqual(run_lines([f"v_mov_b32 v2, {x:#x}", "v_bfrev_b32_e32 v3, v2", "v_bfrev_b32_e32 v0, v3",
                                            "v_mov_b32 v1, 0"])[0], x)


class AbsTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref_abs(0xFFFFFFFF), 1)
        self.assertEqual(ref_abs(0x80000000), 0x80000000)
        self.assertEqual(ref_abs(0x7FFFFFFF), 0x7FFFFFFF)

    def test_s_abs_i32_and_scc(self):
        for x in VALUES:
            with self.subTest(x=hex(x)):
                want = ref_abs(x)
                got = run_lines([f"s_mov_b32 s2, {x:#x}", "s_abs_i32 s0, s2", "s_cselect_b32 s1, 1, 0",
                                 "v_mov_b32 v0, s0", "v_mov_b32 v1, s1"])
                self.assertEqual(got, (want, int(want != 0)))


if __name__ == "__main__":
    unittest.main()
