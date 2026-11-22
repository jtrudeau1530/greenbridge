"""Semantic interpreter tests for RDNA2 s_sext_i32_i8, s_sext_i32_i16, and s_flbit_i32.

RDNA2 scalar ALU operations covered:
  - s_sext_i32_i8 D, S0:
      Sign-extends 8-bit signed integer in S0[7:0] to 32-bit signed integer in D.
      Upper source bits S0[31:8] are completely ignored.
      Destination D is in range [-128, 127] (0xFFFFFF80 .. 0x0000007F).
      Scalar Condition Code (SCC) is preserved (SCC is not written).
  - s_sext_i32_i16 D, S0:
      Sign-extends 16-bit signed integer in S0[15:0] to 32-bit signed integer in D.
      Upper source bits S0[31:16] are completely ignored.
      Destination D is in range [-32768, 32767] (0xFFFF8000 .. 0x00007FFF).
      Scalar Condition Code (SCC) is preserved (SCC is not written).
  - s_flbit_i32 D, S0:
      Find first bit from MSB down to LSB before the first bit that differs
      from the sign bit (scalar equivalent of v_ffbh_i32 / FindSMsb).
      Returns bit count from MSB (31 - bit_index) or -1 (0xFFFFFFFF) if all bits
      match the sign bit (0 or -1).
      Scalar Condition Code (SCC) is preserved (SCC is not written).

Zero game data; standard library Python only; Python 3.9+ compatible.
"""
from __future__ import annotations

import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def u32(x: int) -> int:
    """Normalize integer to unsigned 32-bit word [0, 0xFFFFFFFF]."""
    return x & M32


def s32(x: int) -> int:
    """Interpret 32-bit integer as signed [-2^31, 2^31 - 1]."""
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def ref_s_sext_i32_i8(s0: int) -> int:
    """RDNA2 s_sext_i32_i8 reference: sign-extend S0[7:0] to 32-bit signed integer.

    Bits [31:8] of S0 are ignored. Returns unsigned 32-bit word representation.
    """
    b = s0 & 0xFF
    signed_val = b - 0x100 if (b & 0x80) else b
    return signed_val & M32


def ref_s_sext_i32_i16(s0: int) -> int:
    """RDNA2 s_sext_i32_i16 reference: sign-extend S0[15:0] to 32-bit signed integer.

    Bits [31:16] of S0 are ignored. Returns unsigned 32-bit word representation.
    """
    h = s0 & 0xFFFF
    signed_val = h - 0x10000 if (h & 0x8000) else h
    return signed_val & M32


def flbit_i32(x: int) -> int:
    """RDNA2 s_flbit_i32 reference: count bits from MSB before first differing bit."""
    x &= M32
    sign = x >> 31
    for n in range(1, 32):
        if (x >> (31 - n)) & 1 != sign:
            return n
    return M32


OPS = {
    "s_sext_i32_i8": ref_s_sext_i32_i8,
    "s_sext_i32_i16": ref_s_sext_i32_i16,
    "s_flbit_i32": flbit_i32,
}

VALUES = [
    0, 1, 0x7F, 0x80, 0xFF, 0x7FFF, 0x8000, 0xFFFF,
    0x12345678, 0x87654321, 0x7FFFFFFF, 0x80000000,
    0xFFFFFFFF, 0xFFFFFFFE, 0x40000000, 0xC0000000, 0x00010000,
]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_sext.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_sext.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_op(op: str, x: int, scc_in: int) -> tuple[int, int]:
    """Execute shader: SCC = scc_in, then s0 = op(s2); return (s0, SCC after)."""
    if _TMP_DIR is None:
        setUpModule()
    lines = [
        f"s_mov_b32 s2, {x:#x}",
        f"s_mov_b32 s4, {scc_in}",
        "s_cmp_eq_u32 s4, 1",
        f"{op} s0, s2",
        "s_cselect_b32 s1, 0x40000001, 0x40000000",
        "s_and_b32 s5, s0, 0xffff",
        "s_lshr_b32 s6, s0, 16",
        "s_or_b32 s5, s5, 0x40000000",
        "s_or_b32 s6, s6, 0x40000000",
        "v_mov_b32 v0, s5",
        "v_mov_b32 v1, s6",
        "v_mov_b32 v2, s1",
    ]
    asm = (
        "; test_salu_sext\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v2, v2 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi, scc = si.run_fragment(spv)["%out_mrt0"][:3]
    assert lo >> 16 == hi >> 16 == 0x4000 and scc >> 1 == 0x20000000, (hex(lo), hex(hi), hex(scc))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16, scc & 1


class ReferenceSignExtensionTests(unittest.TestCase):
    """Exhaustive and property-based verification of scalar sign extension oracles."""

    def test_ref_s_sext_i32_i8_boundaries(self) -> None:
        # Extreme 8-bit boundaries: -128 (0x80) -> 0xFFFFFF80 (-128 in 32-bit)
        self.assertEqual(ref_s_sext_i32_i8(0x80), 0xFFFFFF80)
        self.assertEqual(s32(ref_s_sext_i32_i8(0x80)), -128)

        # Extreme positive boundary: +127 (0x7F) -> 0x0000007F (+127 in 32-bit)
        self.assertEqual(ref_s_sext_i32_i8(0x7F), 0x0000007F)
        self.assertEqual(s32(ref_s_sext_i32_i8(0x7F)), 127)

        # Zero and positive ones
        self.assertEqual(ref_s_sext_i32_i8(0x00), 0x00000000)
        self.assertEqual(ref_s_sext_i32_i8(0x01), 0x00000001)

        # Negative ones: 0xFF (-1) -> 0xFFFFFFFF (-1)
        self.assertEqual(ref_s_sext_i32_i8(0xFF), 0xFFFFFFFF)
        self.assertEqual(s32(ref_s_sext_i32_i8(0xFF)), -1)

        # Negative two: 0xFE (-2) -> 0xFFFFFFFE (-2)
        self.assertEqual(ref_s_sext_i32_i8(0xFE), 0xFFFFFFFE)
        self.assertEqual(s32(ref_s_sext_i32_i8(0xFE)), -2)

    def test_ref_s_sext_i32_i16_boundaries(self) -> None:
        # Extreme 16-bit boundaries: -32768 (0x8000) -> 0xFFFF8000 (-32768 in 32-bit)
        self.assertEqual(ref_s_sext_i32_i16(0x8000), 0xFFFF8000)
        self.assertEqual(s32(ref_s_sext_i32_i16(0x8000)), -32768)

        # Extreme positive boundary: +32767 (0x7FFF) -> 0x00007FFF (+32767 in 32-bit)
        self.assertEqual(ref_s_sext_i32_i16(0x7FFF), 0x00007FFF)
        self.assertEqual(s32(ref_s_sext_i32_i16(0x7FFF)), 32767)

        # Zero and positive ones
        self.assertEqual(ref_s_sext_i32_i16(0x0000), 0x00000000)
        self.assertEqual(ref_s_sext_i32_i16(0x0001), 0x00000001)

        # Negative ones: 0xFFFF (-1) -> 0xFFFFFFFF (-1)
        self.assertEqual(ref_s_sext_i32_i16(0xFFFF), 0xFFFFFFFF)
        self.assertEqual(s32(ref_s_sext_i32_i16(0xFFFF)), -1)

        # Negative two: 0xFFFE (-2) -> 0xFFFFFFFE (-2)
        self.assertEqual(ref_s_sext_i32_i16(0xFFFE), 0xFFFFFFFE)
        self.assertEqual(s32(ref_s_sext_i32_i16(0xFFFE)), -2)

    def test_high_bits_ignored(self) -> None:
        # High 24 bits must be completely ignored by s_sext_i32_i8
        patterns_8 = [0xDEADBE00, 0x12345600, 0x7FFFFF00, 0x80000000, 0xFFFFFF00]
        for p in patterns_8:
            self.assertEqual(ref_s_sext_i32_i8(p | 0x80), 0xFFFFFF80)
            self.assertEqual(ref_s_sext_i32_i8(p | 0x7F), 0x0000007F)
            self.assertEqual(ref_s_sext_i32_i8(p | 0x00), 0x00000000)
            self.assertEqual(ref_s_sext_i32_i8(p | 0xFF), 0xFFFFFFFF)

        # High 16 bits must be completely ignored by s_sext_i32_i16
        patterns_16 = [0xDEAD0000, 0x12340000, 0x7FFF0000, 0x80000000, 0xFFFF0000]
        for p in patterns_16:
            self.assertEqual(ref_s_sext_i32_i16(p | 0x8000), 0xFFFF8000)
            self.assertEqual(ref_s_sext_i32_i16(p | 0x7FFF), 0x00007FFF)
            self.assertEqual(ref_s_sext_i32_i16(p | 0x0000), 0x00000000)
            self.assertEqual(ref_s_sext_i32_i16(p | 0xFFFF), 0xFFFFFFFF)

    def test_exhaustive_i8_against_struct_pack(self) -> None:
        # Exhaustive verification of all 256 possible 8-bit inputs against Python struct signed byte
        for byte_val in range(256):
            expected_signed = struct.unpack("<b", bytes([byte_val]))[0]
            expected_u32 = expected_signed & M32
            # Test direct byte value
            self.assertEqual(ref_s_sext_i32_i8(byte_val), expected_u32)
            # Test with noisy upper bits
            noisy_input = (byte_val | 0xCAFE00) & M32
            self.assertEqual(ref_s_sext_i32_i8(noisy_input), expected_u32)

    def test_exhaustive_i16_samples_against_struct_pack(self) -> None:
        # Verification of 16-bit inputs across bit flips, boundaries, and strides
        for word_val in list(range(0, 512)) + list(range(0x7F00, 0x8100)) + list(range(0xFF00, 0x10000, 3)):
            expected_signed = struct.unpack("<h", struct.pack("<H", word_val))[0]
            expected_u32 = expected_signed & M32
            self.assertEqual(ref_s_sext_i32_i16(word_val), expected_u32)
            noisy_input = (word_val | 0x55AA0000) & M32
            self.assertEqual(ref_s_sext_i32_i16(noisy_input), expected_u32)

    def test_algebraic_invariants(self) -> None:
        # Idempotence: sext(sext(x)) == sext(x)
        for v in [0, 1, 0x7F, 0x80, 0xFF, 0x12345678, 0x80000000, 0xFFFFFFFF]:
            res_i8 = ref_s_sext_i32_i8(v)
            self.assertEqual(ref_s_sext_i32_i8(res_i8), res_i8)
            res_i16 = ref_s_sext_i32_i16(v)
            self.assertEqual(ref_s_sext_i32_i16(res_i16), res_i16)

        # Composition: 16-bit sign extension of an 8-bit sign-extended value is identity
        for v in range(256):
            ext8 = ref_s_sext_i32_i8(v)
            self.assertEqual(ref_s_sext_i32_i16(ext8), ext8)


class SextFlbitLegacyTests(unittest.TestCase):
    """Integration tests executing synthetic shaders through rdna2spirv and spirv_mini_interp."""

    def test_legacy_reference(self) -> None:
        self.assertEqual(ref_s_sext_i32_i8(0x80), 0xFFFFFF80)
        self.assertEqual(ref_s_sext_i32_i8(0x1234807F), 0x7F)
        self.assertEqual(ref_s_sext_i32_i16(0xABCD8000), 0xFFFF8000)
        self.assertEqual(flbit_i32(1), 31)
        self.assertEqual(flbit_i32(0x40000000), 1)
        self.assertEqual(flbit_i32(0xFFFFFFFE), 31)
        self.assertEqual(flbit_i32(0x80000000), 1)
        self.assertEqual(flbit_i32(0), M32)
        self.assertEqual(flbit_i32(M32), M32)

    def test_values_and_scc_kept(self) -> None:
        for op, fn in OPS.items():
            for x in VALUES:
                for scc_in in (0, 1):
                    with self.subTest(op=op, x=hex(x), scc=scc_in):
                        self.assertEqual(run_op(op, x, scc_in), (fn(x), scc_in))

    def test_seeded(self) -> None:
        rng = random.Random(0x5E)
        for _ in range(30):
            op, x = rng.choice(list(OPS)), rng.getrandbits(32) >> rng.randrange(32)
            if rng.getrandbits(1):
                x = ~x & M32
            with self.subTest(op=op, x=hex(x)):
                self.assertEqual(run_op(op, x, 0)[0], OPS[op](x))


if __name__ == "__main__":
    unittest.main()
