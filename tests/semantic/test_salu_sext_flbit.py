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
  - s_flbit_i32_b32 D, S0:
      Count leading zeros from MSB down to LSB (unsigned bit scan).
      Returns bit count from MSB (31 - highest_set_bit) or -1 (0xFFFFFFFF) on zero.
      Scalar Condition Code (SCC) is preserved.

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


def ref_s_flbit_i32(s0: int) -> int:
    """RDNA2 s_flbit_i32 reference: count bits from MSB before first differing bit.

    Signed MSB scan (FindSMsb):
      - Returns M32 (0xFFFFFFFF, representing -1) if all bits equal the sign bit (0 or -1).
      - For positive values (sign=0): counts bits from bit 31 before the first 1-bit.
      - For negative values (sign=1): counts bits from bit 31 before the first 0-bit.
    """
    s0 &= M32
    if s0 == 0 or s0 == M32:
        return M32
    sign = (s0 >> 31) & 1
    for n in range(1, 32):
        if ((s0 >> (31 - n)) & 1) != sign:
            return n
    return M32


def ref_s_flbit_i32_b32(s0: int) -> int:
    """RDNA2 s_flbit_i32_b32 reference: count leading zeros from MSB down to LSB.

    Unsigned MSB scan (FindUMsb):
      - Returns M32 (0xFFFFFFFF, representing -1) if s0 == 0.
      - Otherwise returns 31 - highest_set_bit_index (32 - s0.bit_length()).
    """
    s0 &= M32
    if s0 == 0:
        return M32
    return 32 - s0.bit_length()


flbit_i32 = ref_s_flbit_i32

OPS = {
    "s_sext_i32_i8": ref_s_sext_i32_i8,
    "s_sext_i32_i16": ref_s_sext_i32_i16,
    "s_flbit_i32": ref_s_flbit_i32,
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


class ScalarAluInterpreter:
    """Deterministic semantic interpreter for RDNA2 scalar ALU instructions.

    Models an array of 106 32-bit SGPRs (s0..s105) and a 1-bit Scalar Condition Code (SCC).
    Enforces register aliasing rules, unaligned/independent register preservation,
    and exact condition code preservation/updates.
    """

    def __init__(self, num_sgprs: int = 106, initial_scc: int = 0) -> None:
        self.num_sgprs = num_sgprs
        self.sgprs = [0] * num_sgprs
        self.scc = initial_scc & 1

    def reset(self, initial_scc: int = 0) -> None:
        self.sgprs = [0] * self.num_sgprs
        self.scc = initial_scc & 1

    def read_sgpr(self, idx: int) -> int:
        assert 0 <= idx < self.num_sgprs, f"SGPR index {idx} out of range"
        return self.sgprs[idx] & M32

    def write_sgpr(self, idx: int, val: int) -> None:
        assert 0 <= idx < self.num_sgprs, f"SGPR index {idx} out of range"
        self.sgprs[idx] = val & M32

    def resolve_src(self, src: int | str) -> int:
        if isinstance(src, int):
            return src & M32
        if isinstance(src, str) and src.startswith("s") and src[1:].isdigit():
            return self.read_sgpr(int(src[1:]))
        raise ValueError(f"Unrecognized scalar source: {src}")

    def exec_s_mov_b32(self, dst: int, src: int | str) -> None:
        val = self.resolve_src(src)
        self.write_sgpr(dst, val)

    def exec_s_sext_i32_i8(self, dst: int, src: int | str) -> None:
        val = self.resolve_src(src)
        res = ref_s_sext_i32_i8(val)
        self.write_sgpr(dst, res)

    def exec_s_sext_i32_i16(self, dst: int, src: int | str) -> None:
        val = self.resolve_src(src)
        res = ref_s_sext_i32_i16(val)
        self.write_sgpr(dst, res)

    def exec_s_flbit_i32(self, dst: int, src: int | str) -> None:
        val = self.resolve_src(src)
        res = ref_s_flbit_i32(val)
        self.write_sgpr(dst, res)

    def exec_s_flbit_i32_b32(self, dst: int, src: int | str) -> None:
        val = self.resolve_src(src)
        res = ref_s_flbit_i32_b32(val)
        self.write_sgpr(dst, res)


class ReferenceSignExtensionTests(unittest.TestCase):
    """Exhaustive and property-based verification of scalar sign extension oracles."""

    def test_ref_s_sext_i32_i8_boundaries(self) -> None:
        self.assertEqual(ref_s_sext_i32_i8(0x80), 0xFFFFFF80)
        self.assertEqual(s32(ref_s_sext_i32_i8(0x80)), -128)

        self.assertEqual(ref_s_sext_i32_i8(0x7F), 0x0000007F)
        self.assertEqual(s32(ref_s_sext_i32_i8(0x7F)), 127)

        self.assertEqual(ref_s_sext_i32_i8(0x00), 0x00000000)
        self.assertEqual(ref_s_sext_i32_i8(0x01), 0x00000001)

        self.assertEqual(ref_s_sext_i32_i8(0xFF), 0xFFFFFFFF)
        self.assertEqual(s32(ref_s_sext_i32_i8(0xFF)), -1)

        self.assertEqual(ref_s_sext_i32_i8(0xFE), 0xFFFFFFFE)
        self.assertEqual(s32(ref_s_sext_i32_i8(0xFE)), -2)

    def test_ref_s_sext_i32_i16_boundaries(self) -> None:
        self.assertEqual(ref_s_sext_i32_i16(0x8000), 0xFFFF8000)
        self.assertEqual(s32(ref_s_sext_i32_i16(0x8000)), -32768)

        self.assertEqual(ref_s_sext_i32_i16(0x7FFF), 0x00007FFF)
        self.assertEqual(s32(ref_s_sext_i32_i16(0x7FFF)), 32767)

        self.assertEqual(ref_s_sext_i32_i16(0x0000), 0x00000000)
        self.assertEqual(ref_s_sext_i32_i16(0x0001), 0x00000001)

        self.assertEqual(ref_s_sext_i32_i16(0xFFFF), 0xFFFFFFFF)
        self.assertEqual(s32(ref_s_sext_i32_i16(0xFFFF)), -1)

        self.assertEqual(ref_s_sext_i32_i16(0xFFFE), 0xFFFFFFFE)
        self.assertEqual(s32(ref_s_sext_i32_i16(0xFFFE)), -2)

    def test_high_bits_ignored(self) -> None:
        patterns_8 = [0xDEADBE00, 0x12345600, 0x7FFFFF00, 0x80000000, 0xFFFFFF00]
        for p in patterns_8:
            self.assertEqual(ref_s_sext_i32_i8(p | 0x80), 0xFFFFFF80)
            self.assertEqual(ref_s_sext_i32_i8(p | 0x7F), 0x0000007F)
            self.assertEqual(ref_s_sext_i32_i8(p | 0x00), 0x00000000)
            self.assertEqual(ref_s_sext_i32_i8(p | 0xFF), 0xFFFFFFFF)

        patterns_16 = [0xDEAD0000, 0x12340000, 0x7FFF0000, 0x80000000, 0xFFFF0000]
        for p in patterns_16:
            self.assertEqual(ref_s_sext_i32_i16(p | 0x8000), 0xFFFF8000)
            self.assertEqual(ref_s_sext_i32_i16(p | 0x7FFF), 0x00007FFF)
            self.assertEqual(ref_s_sext_i32_i16(p | 0x0000), 0x00000000)
            self.assertEqual(ref_s_sext_i32_i16(p | 0xFFFF), 0xFFFFFFFF)

    def test_exhaustive_i8_against_struct_pack(self) -> None:
        for byte_val in range(256):
            expected_signed = struct.unpack("<b", bytes([byte_val]))[0]
            expected_u32 = expected_signed & M32
            self.assertEqual(ref_s_sext_i32_i8(byte_val), expected_u32)
            noisy_input = (byte_val | 0xCAFE00) & M32
            self.assertEqual(ref_s_sext_i32_i8(noisy_input), expected_u32)

    def test_exhaustive_i16_samples_against_struct_pack(self) -> None:
        for word_val in list(range(0, 512)) + list(range(0x7F00, 0x8100)) + list(range(0xFF00, 0x10000, 3)):
            expected_signed = struct.unpack("<h", struct.pack("<H", word_val))[0]
            expected_u32 = expected_signed & M32
            self.assertEqual(ref_s_sext_i32_i16(word_val), expected_u32)
            noisy_input = (word_val | 0x55AA0000) & M32
            self.assertEqual(ref_s_sext_i32_i16(noisy_input), expected_u32)

    def test_algebraic_invariants(self) -> None:
        for v in [0, 1, 0x7F, 0x80, 0xFF, 0x12345678, 0x80000000, 0xFFFFFFFF]:
            res_i8 = ref_s_sext_i32_i8(v)
            self.assertEqual(ref_s_sext_i32_i8(res_i8), res_i8)
            res_i16 = ref_s_sext_i32_i16(v)
            self.assertEqual(ref_s_sext_i32_i16(res_i16), res_i16)

        for v in range(256):
            ext8 = ref_s_sext_i32_i8(v)
            self.assertEqual(ref_s_sext_i32_i16(ext8), ext8)


class ReferenceMsbScanTests(unittest.TestCase):
    """Reference oracle tests for scalar MSB scanning instructions s_flbit_i32 and s_flbit_i32_b32."""

    def test_s_flbit_i32_powers_of_two(self) -> None:
        # Positive values (sign bit 0): count bits from MSB before first 1-bit.
        # 0x40000000 is bit 30 -> 1 bit below sign bit 31 -> returns 1.
        self.assertEqual(ref_s_flbit_i32(0x40000000), 1)
        self.assertEqual(ref_s_flbit_i32(0x20000000), 2)
        self.assertEqual(ref_s_flbit_i32(0x10000000), 3)
        self.assertEqual(ref_s_flbit_i32(0x00000002), 30)
        self.assertEqual(ref_s_flbit_i32(0x00000001), 31)

        # For every power of two from bit 0 to bit 30:
        for bit in range(31):
            val = 1 << bit
            expected = 31 - bit
            self.assertEqual(ref_s_flbit_i32(val), expected)

    def test_s_flbit_i32_negative_patterns(self) -> None:
        # Negative values (sign bit 1): count bits from MSB before first 0-bit.
        # 0x80000000: sign is 1, bit 30 is 0 -> 1 bit below sign bit -> returns 1.
        self.assertEqual(ref_s_flbit_i32(0x80000000), 1)
        # 0xC0000000: bits 31:30 are 1, bit 29 is 0 -> returns 2.
        self.assertEqual(ref_s_flbit_i32(0xC0000000), 2)
        # 0xE0000000: bits 31:29 are 1, bit 28 is 0 -> returns 3.
        self.assertEqual(ref_s_flbit_i32(0xE0000000), 3)
        # 0xFFFFFFFE: bits 31:1 are 1, bit 0 is 0 -> returns 31.
        self.assertEqual(ref_s_flbit_i32(0xFFFFFFFE), 31)

    def test_s_flbit_i32_uniform_cases(self) -> None:
        # All 0s: no bit differs from sign bit 0 -> returns M32 (-1)
        self.assertEqual(ref_s_flbit_i32(0x00000000), M32)
        # All 1s: no bit differs from sign bit 1 -> returns M32 (-1)
        self.assertEqual(ref_s_flbit_i32(0xFFFFFFFF), M32)

    def test_s_flbit_i32_bitwise_not_symmetry(self) -> None:
        # Fundamental property: s_flbit_i32(x) == s_flbit_i32(~x)
        # Inverting all bits swaps sign 0 <-> 1 and target 1 <-> 0, preserving bit position.
        test_values = [
            1, 2, 7, 0x40000000, 0x12345678, 0x7FFFFFFF, 0x55555555,
            0x80000000, 0xC0000000, 0xFFFFFFFE, 0x87654321, 0xAAAAAAAA,
        ]
        for v in test_values:
            not_v = (~v) & M32
            self.assertEqual(ref_s_flbit_i32(v), ref_s_flbit_i32(not_v))

    def test_s_flbit_i32_b32_unsigned_scan(self) -> None:
        # Unsigned scan counts leading zeros (31 - highest_set_bit)
        self.assertEqual(ref_s_flbit_i32_b32(0x80000000), 0)
        self.assertEqual(ref_s_flbit_i32_b32(0x40000000), 1)
        self.assertEqual(ref_s_flbit_i32_b32(0x00000001), 31)
        self.assertEqual(ref_s_flbit_i32_b32(0x00000000), M32)

        for bit in range(32):
            val = 1 << bit
            self.assertEqual(ref_s_flbit_i32_b32(val), 31 - bit)


class ScalarAluInterpreterTests(unittest.TestCase):
    """Interpreter test suite for register aliasing, unaffected registers, and state isolation."""

    def setUp(self) -> None:
        self.interp = ScalarAluInterpreter(num_sgprs=106, initial_scc=0)

    def test_exec_s_flbit_i32_immediate_and_reg(self) -> None:
        # Test immediate source
        self.interp.exec_s_flbit_i32(10, 0x40000000)
        self.assertEqual(self.interp.read_sgpr(10), 1)

        # Test register source
        self.interp.exec_s_mov_b32(11, 0xFFFFFFFE)
        self.interp.exec_s_flbit_i32(12, "s11")
        self.assertEqual(self.interp.read_sgpr(12), 31)

    def test_sgpr_aliasing_in_place(self) -> None:
        # s_flbit_i32 s4, s4
        self.interp.write_sgpr(4, 0x20000000)
        self.interp.exec_s_flbit_i32(4, "s4")
        self.assertEqual(self.interp.read_sgpr(4), 2)

        # s_flbit_i32 s5, s5 on negative pattern
        self.interp.write_sgpr(5, 0x80000000)
        self.interp.exec_s_flbit_i32(5, "s5")
        self.assertEqual(self.interp.read_sgpr(5), 1)

        # s_sext_i32_i8 s6, s6
        self.interp.write_sgpr(6, 0x123480)
        self.interp.exec_s_sext_i32_i8(6, "s6")
        self.assertEqual(self.interp.read_sgpr(6), 0xFFFFFF80)

        # s_sext_i32_i16 s7, s7
        self.interp.write_sgpr(7, 0xABCD7FFF)
        self.interp.exec_s_sext_i32_i16(7, "s7")
        self.assertEqual(self.interp.read_sgpr(7), 0x00007FFF)

    def test_sgpr_isolation_and_unaffected_registers(self) -> None:
        # Write unique marker values across registers
        for i in range(20):
            self.interp.write_sgpr(i, 0x1000 + i)

        # Perform operation on s10
        self.interp.exec_s_flbit_i32(10, 0x40000000)
        self.assertEqual(self.interp.read_sgpr(10), 1)

        # Verify all other registers 0..9 and 11..19 remain untouched
        for i in range(20):
            if i != 10:
                self.assertEqual(self.interp.read_sgpr(i), 0x1000 + i)

    def test_scc_preserved_during_interpreter_execution(self) -> None:
        for init_scc in (0, 1):
            self.interp.reset(initial_scc=init_scc)
            self.interp.exec_s_sext_i32_i8(1, 0x80)
            self.assertEqual(self.interp.scc, init_scc)

            self.interp.exec_s_sext_i32_i16(2, 0x7FFF)
            self.assertEqual(self.interp.scc, init_scc)

            self.interp.exec_s_flbit_i32(3, 0x40000000)
            self.assertEqual(self.interp.scc, init_scc)

            self.interp.exec_s_flbit_i32(4, 0)
            self.assertEqual(self.interp.scc, init_scc)

    def test_compound_sign_extend_then_flbit(self) -> None:
        # Sign extend 0x80 -> 0xFFFFFF80, then flbit
        self.interp.write_sgpr(0, 0x80)
        self.interp.exec_s_sext_i32_i8(1, "s0")
        self.assertEqual(self.interp.read_sgpr(1), 0xFFFFFF80)
        self.interp.exec_s_flbit_i32(2, "s1")
        # In 0xFFFFFF80, sign is 1. Bit 6 is 0 -> 31 - 6 = 25
        self.assertEqual(self.interp.read_sgpr(2), 25)

        # Sign extend 0x7F -> 0x0000007F, then flbit
        self.interp.write_sgpr(3, 0x7F)
        self.interp.exec_s_sext_i32_i8(4, "s3")
        self.assertEqual(self.interp.read_sgpr(4), 0x0000007F)
        self.interp.exec_s_flbit_i32(5, "s4")
        # In 0x0000007F, sign is 0. Bit 6 is 1 -> 31 - 6 = 25
        self.assertEqual(self.interp.read_sgpr(5), 25)


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
