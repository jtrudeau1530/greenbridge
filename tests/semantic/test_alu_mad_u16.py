from __future__ import annotations

"""Semantic interpreter tests for RDNA2 16-bit integer multiply-add (v_mad_u16 and v_mad_i16).

AMD RDNA VALU Instruction Set Architecture (GFX10/RDNA2):
  v_mad_u16 dst, src0, src1, src2 [op_sel:[s0, s1, s2, dst_hi]] [clamp]
  v_mad_i16 dst, src0, src1, src2 [op_sel:[s0, s1, s2, dst_hi]] [clamp]

Mathematical specification:
  - Unsigned (v_mad_u16): operands S0, S1, S2 in [0, 65535] (0x0000 to 0xFFFF).
  - Signed (v_mad_i16): operands S0, S1, S2 sign-extended from 16-bit to signed integers
    in [-32768, 32767] (0x8000 to 0x7FFF).
  - op_sel bits 0..2 select whether each source operand is extracted from the low half-word
    (bits 15:0 if 0) or high half-word (bits 31:16 if 1) of the respective 32-bit register.
  - Multiplication and addition are computed in exact 32-bit integer arithmetic without
    intermediate truncation:
      P = S0 * S1
      T = P + S2
  - Without clamp:
      The 32-bit result wraps modulo 65536 (retaining the lower 16 bits):
      D = T & 0xFFFF
  - With clamp:
      - v_mad_u16: saturates to unsigned range [0, 65535]:
          D = min(65535, max(0, T))
      - v_mad_i16: saturates to signed range [-32768, 32767]:
          D = min(32767, max(-32768, T)) & 0xFFFF
  - Destination packing (VOP3 only):
      dst_hi (op_sel bit 3) determines destination placement:
      - dst_hi = 0: writes D to dst[15:0] and preserves dst[31:16] from the prior VGPR content.
      - dst_hi = 1: writes D to dst[31:16] and preserves dst[15:0] from the prior VGPR content.

Standard library only, zero game data, Python 3.9+ compatible.
"""

import gc
import itertools
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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
OLD = 0xAAAA5555
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_mad_u16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_mad_u16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


def u16(x: int) -> int:
    """Mask integer to 16-bit unsigned range [0, 65535]."""
    return x & MASK16


def s16(x: int) -> int:
    """Sign-extend 16-bit integer to signed two's complement in [-32768, 32767]."""
    v = x & MASK16
    return v - 0x10000 if (v & 0x8000) else v


def half16_u(val: int, hi: int | bool) -> int:
    """Extract low or high 16-bit half-word as unsigned integer."""
    return ((val >> 16) if hi else val) & MASK16


def half16_s(val: int, hi: int | bool) -> int:
    """Extract low or high 16-bit half-word as signed two's complement integer."""
    h = ((val >> 16) if hi else val) & MASK16
    return s16(h)


def merge_dst16(old_dst: int, val16: int, dst_hi: int | bool) -> int:
    """Pack 16-bit result into 32-bit register, preserving the untouched half."""
    val16 &= MASK16
    old_dst &= MASK32
    if dst_hi:
        return ((val16 << 16) | (old_dst & MASK16)) & MASK32
    return ((old_dst & 0xFFFF0000) | val16) & MASK32


def ref_mad_u16(
    s0: int,
    s1: int,
    s2: int,
    sel: tuple[int, int, int] = (0, 0, 0),
    dst_hi: int | bool = 0,
    clamp: bool = False,
    old_dst: int = OLD,
) -> int:
    """Reference oracle for RDNA2 v_mad_u16 (unsigned)."""
    x = half16_u(s0, sel[0])
    y = half16_u(s1, sel[1])
    z = half16_u(s2, sel[2])
    tot = x * y + z
    if clamp:
        tot = min(MASK16, tot)
    r16 = tot & MASK16
    return merge_dst16(old_dst, r16, dst_hi)


def ref_mad_i16(
    s0: int,
    s1: int,
    s2: int,
    sel: tuple[int, int, int] = (0, 0, 0),
    dst_hi: int | bool = 0,
    clamp: bool = False,
    old_dst: int = OLD,
) -> int:
    """Reference oracle for RDNA2 v_mad_i16 (signed)."""
    x = half16_s(s0, sel[0])
    y = half16_s(s1, sel[1])
    z = half16_s(s2, sel[2])
    tot = x * y + z
    if clamp:
        tot = max(-0x8000, min(0x7FFF, tot))
    r16 = tot & MASK16
    return merge_dst16(old_dst, r16, dst_hi)


def run_shader_mad(line: str, v1: int, v2: int, v3: int, old: int = OLD) -> int:
    """Assemble and run synthetic pixel shader under spirv_mini_interp."""
    if _TMP_DIR is None or _SRC_PATH is None:
        setUpModule()
    lines = [
        f"v_mov_b32 v1, {v1 & MASK32:#x}",
        f"v_mov_b32 v2, {v2 & MASK32:#x}",
        f"v_mov_b32 v3, {v3 & MASK32:#x}",
        f"v_mov_b32 v0, {old & MASK32:#x}",
        line,
    ]
    asm = "; test_mad16\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class TestMadU16ReferenceOracle(unittest.TestCase):
    """Mathematical verification and invariant proofs for unsigned 16-bit multiply-add."""

    def test_zero_multiplication_property(self):
        """0 * S1 + S2 must equal S2 for any inputs."""
        for s1 in (0, 1, 100, 32767, 65535):
            for s2 in (0, 5, 255, 60000, 65535):
                with self.subTest(s1=s1, s2=s2):
                    expected = (OLD & 0xFFFF0000) | s2
                    self.assertEqual(ref_mad_u16(0, s1, s2), expected)
                    self.assertEqual(ref_mad_u16(s1, 0, s2), expected)

    def test_identity_multiplication_property(self):
        """1 * S1 + S2 must equal (S1 + S2) mod 65536."""
        for s1 in (0, 1, 50, 500, 32768, 65535):
            for s2 in (0, 1, 50, 500, 32768, 65535):
                with self.subTest(s1=s1, s2=s2):
                    expected_val = (s1 + s2) & MASK16
                    expected = (OLD & 0xFFFF0000) | expected_val
                    self.assertEqual(ref_mad_u16(1, s1, s2), expected)
                    self.assertEqual(ref_mad_u16(s1, 1, s2), expected)

    def test_commutativity_of_product_operands(self):
        """S0 * S1 + S2 must equal S1 * S0 + S2 for all selections."""
        pairs = [
            (7, 11, 13),
            (250, 300, 400),
            (65535, 2, 5),
            (1234, 5678, 9101),
        ]
        for s0, s1, s2 in pairs:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                res1 = ref_mad_u16(s0, s1, s2)
                res2 = ref_mad_u16(s1, s0, s2)
                self.assertEqual(res1, res2)

    def test_modulo_overflow_wrapping(self):
        """Without clamp, intermediate products and sums wrap cleanly modulo 65536."""
        cases = [
            (65535, 2, 10, (65535 * 2 + 10) & MASK16),          # 131080 -> 8
            (65535, 65535, 65535, (65535 * 65535 + 65535) & MASK16),  # 4294901760 -> 0
            (256, 256, 0, (256 * 256 + 0) & MASK16),            # 65536 -> 0
            (256, 256, 1, (256 * 256 + 1) & MASK16),            # 65537 -> 1
            (300, 300, 1000, (300 * 300 + 1000) & MASK16),      # 91000 -> 25464
            (50000, 2, 1, (50000 * 2 + 1) & MASK16),            # 100001 -> 34465
        ]
        for s0, s1, s2, expected_val in cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                expected = (OLD & 0xFFFF0000) | expected_val
                self.assertEqual(ref_mad_u16(s0, s1, s2), expected)

    def test_halfword_extraction_u16(self):
        """Extraction of low or high 16-bit halfwords across all 3 source operands."""
        packed_s0 = 0x0007_0003  # hi: 7, lo: 3
        packed_s1 = 0x000B_0005  # hi: 11, lo: 5
        packed_s2 = 0x000D_0002  # hi: 13, lo: 2

        # (lo, lo, lo): 3 * 5 + 2 = 17
        self.assertEqual(ref_mad_u16(packed_s0, packed_s1, packed_s2, sel=(0, 0, 0)), (OLD & 0xFFFF0000) | 17)
        # (hi, lo, lo): 7 * 5 + 2 = 37
        self.assertEqual(ref_mad_u16(packed_s0, packed_s1, packed_s2, sel=(1, 0, 0)), (OLD & 0xFFFF0000) | 37)
        # (lo, hi, lo): 3 * 11 + 2 = 35
        self.assertEqual(ref_mad_u16(packed_s0, packed_s1, packed_s2, sel=(0, 1, 0)), (OLD & 0xFFFF0000) | 35)
        # (lo, lo, hi): 3 * 5 + 13 = 28
        self.assertEqual(ref_mad_u16(packed_s0, packed_s1, packed_s2, sel=(0, 0, 1)), (OLD & 0xFFFF0000) | 28)
        # (hi, hi, hi): 7 * 11 + 13 = 90
        self.assertEqual(ref_mad_u16(packed_s0, packed_s1, packed_s2, sel=(1, 1, 1)), (OLD & 0xFFFF0000) | 90)

    def test_dst_packing_low_and_high(self):
        """Destination packing verifies preservation of untouched halfword."""
        old = 0x12345678
        res_lo = ref_mad_u16(2, 3, 4, dst_hi=0, old_dst=old)  # 2 * 3 + 4 = 10 (0x000A)
        self.assertEqual(res_lo, 0x1234000A)
        self.assertEqual(res_lo >> 16, 0x1234)
        self.assertEqual(res_lo & MASK16, 10)

        res_hi = ref_mad_u16(2, 3, 4, dst_hi=1, old_dst=old)
        self.assertEqual(res_hi, 0x000A5678)
        self.assertEqual(res_hi >> 16, 10)
        self.assertEqual(res_hi & MASK16, 0x5678)

    def test_comprehensive_boundary_grid(self):
        """Boundary values across {0, 1, 2, 255, 256, 32767, 32768, 65534, 65535}."""
        boundaries = [0, 1, 2, 255, 256, 32767, 32768, 65534, 65535]
        for s0 in boundaries:
            for s1 in (0, 1, 256, 65535):
                for s2 in (0, 1, 65535):
                    expected_val = (s0 * s1 + s2) & MASK16
                    self.assertEqual(ref_mad_u16(s0, s1, s2) & MASK16, expected_val)


class TestMadI16ReferenceOracle(unittest.TestCase):
    """Mathematical verification and invariant proofs for signed 16-bit multiply-add."""

    def test_signed_zero_and_identity(self):
        """0 * S1 + S2 == S2, 1 * S1 + S2 == S1 + S2, and -1 * S1 + 0 == -S1."""
        for s1 in (-32768, -100, -1, 0, 1, 100, 32767):
            for s2 in (-32768, -500, 0, 500, 32767):
                with self.subTest(s1=s1, s2=s2):
                    expected_zero = (OLD & 0xFFFF0000) | (s2 & MASK16)
                    self.assertEqual(ref_mad_i16(0, s1, s2), expected_zero)
                    self.assertEqual(ref_mad_i16(s1, 0, s2), expected_zero)

                    expected_one = (OLD & 0xFFFF0000) | ((s1 + s2) & MASK16)
                    self.assertEqual(ref_mad_i16(1, s1, s2), expected_one)
                    self.assertEqual(ref_mad_i16(s1, 1, s2), expected_one)

            # (-1) * S1 + 0 = -S1 (in 16-bit two's complement)
            expected_neg = (OLD & 0xFFFF0000) | ((-s1) & MASK16)
            self.assertEqual(ref_mad_i16(0xFFFF, s1, 0), expected_neg)

    def test_signed_sign_combinations(self):
        """Sign products: (+) * (+), (+) * (-), (-) * (+), (-) * (-)."""
        cases = [
            (10, 20, 5, 205),                   # (+) * (+) + (+) = 205
            (10, -20, 5, -195),                 # (+) * (-) + (+) = -195
            (-10, 20, -5, -205),                # (-) * (+) + (-) = -205
            (-10, -20, 5, 205),                 # (-) * (-) + (+) = 205
            (-10, -20, -5, 195),                # (-) * (-) + (-) = 195
        ]
        for s0, s1, s2, want in cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                expected = (OLD & 0xFFFF0000) | (want & MASK16)
                self.assertEqual(ref_mad_i16(s0 & MASK16, s1 & MASK16, s2 & MASK16), expected)

    def test_signed_wrapping_without_clamp(self):
        """Intermediate signed results wrap cleanly modulo 65536 when clamp is absent."""
        cases = [
            (-2, 3, 5, -1),                     # -2 * 3 + 5 = -1 (0xFFFF)
            (32767, 2, 0, 65534),               # 32767 * 2 = 65534 -> -2 (0xFFFE)
            (-32768, -1, 0, 32768),             # -32768 * -1 = 32768 -> -32768 (0x8000)
            (-32768, 2, 0, -65536),             # -32768 * 2 = -65536 -> 0 (0x0000)
            (-32768, -32768, 0, 1073741824),    # (-32768)^2 = 0x40000000 -> 0 (0x0000)
        ]
        for s0, s1, s2, want_full in cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                expected = (OLD & 0xFFFF0000) | (want_full & MASK16)
                self.assertEqual(ref_mad_i16(s0 & MASK16, s1 & MASK16, s2 & MASK16), expected)

    def test_signed_boundary_grid(self):
        """Signed boundary values {-32768, -32767, -1, 0, 1, 32766, 32767}."""
        boundaries = [-32768, -32767, -1, 0, 1, 32766, 32767]
        for s0 in boundaries:
            for s1 in (-1, 0, 1, 2, -2):
                for s2 in (-1, 0, 1):
                    full = s0 * s1 + s2
                    expected = (OLD & 0xFFFF0000) | (full & MASK16)
                    self.assertEqual(ref_mad_i16(s0 & MASK16, s1 & MASK16, s2 & MASK16), expected)


class TestClampSaturation(unittest.TestCase):
    """Saturated clamping modifiers: unsigned [0, 65535] vs signed [-32768, 32767]."""

    def test_v_mad_u16_clamp_saturates_above_65535(self):
        """v_mad_u16 clamp saturates full-width values > 65535 to 65535 (0xFFFF)."""
        overflow_cases = [
            (256, 256, 1),          # 256 * 256 + 1 = 65537 -> 65535 (unclamped: 1)
            (65535, 2, 0),          # 65535 * 2 = 131070 -> 65535 (unclamped: 65534)
            (65535, 65535, 65535),  # 4294901760 -> 65535 (unclamped: 0)
            (300, 300, 1000),       # 91000 -> 65535 (unclamped: 25464)
            (256, 256, 0),          # 65536 -> 65535 (unclamped: 0)
        ]
        for s0, s1, s2 in overflow_cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                expected = (OLD & 0xFFFF0000) | 0xFFFF
                self.assertEqual(ref_mad_u16(s0, s1, s2, clamp=True), expected)
                got = run_shader_mad("v_mad_u16 v0, v1, v2, v3 clamp", s0, s1, s2)
                self.assertEqual(got, expected)

    def test_v_mad_u16_clamp_preserves_in_bounds(self):
        """v_mad_u16 clamp preserves values <= 65535 unchanged."""
        in_bounds_cases = [
            (0, 0, 0, 0),
            (3, 4, 5, 17),
            (200, 200, 1000, 41000),
            (1, 65535, 0, 65535),
            (255, 255, 255, 65280),
        ]
        for s0, s1, s2, want in in_bounds_cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                expected = (OLD & 0xFFFF0000) | want
                self.assertEqual(ref_mad_u16(s0, s1, s2, clamp=True), expected)
                got = run_shader_mad("v_mad_u16 v0, v1, v2, v3 clamp", s0, s1, s2)
                self.assertEqual(got, expected)

    def test_v_mad_i16_clamp_positive_saturation_to_32767(self):
        """v_mad_i16 clamp saturates full-width signed values > 32767 to 32767 (0x7FFF)."""
        pos_overflow = [
            (32767, 2, 0),          # 65534 -> 32767 (0x7FFF)
            (200, 200, 0),          # 40000 -> 32767 (0x7FFF)
            (-32768, -2, 0),        # 65536 -> 32767 (0x7FFF)
            (-32768, -32768, 0),    # 1073741824 -> 32767 (0x7FFF)
            (32767, 32767, 32767),  # 1073709056 -> 32767 (0x7FFF)
        ]
        for s0, s1, s2 in pos_overflow:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                expected = (OLD & 0xFFFF0000) | 0x7FFF
                self.assertEqual(ref_mad_i16(s0 & MASK16, s1 & MASK16, s2 & MASK16, clamp=True), expected)
                got = run_shader_mad("v_mad_i16 v0, v1, v2, v3 clamp", s0 & MASK16, s1 & MASK16, s2 & MASK16)
                self.assertEqual(got, expected)

    def test_v_mad_i16_clamp_negative_saturation_to_minus_32768(self):
        """v_mad_i16 clamp saturates full-width signed values < -32768 to -32768 (0x8000)."""
        neg_overflow = [
            (32767, -2, 0),         # -65534 -> -32768 (0x8000)
            (-200, 200, 0),         # -40000 -> -32768 (0x8000)
            (-200, 150, -5000),     # -30000 - 5000 = -35000 -> -32768 (0x8000)
            (-32768, 2, 0),         # -65536 -> -32768 (0x8000)
            (-32768, 1, -1),        # -32768 - 1 = -32769 -> -32768 (0x8000)
        ]
        for s0, s1, s2 in neg_overflow:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                expected = (OLD & 0xFFFF0000) | 0x8000
                self.assertEqual(ref_mad_i16(s0 & MASK16, s1 & MASK16, s2 & MASK16, clamp=True), expected)
                got = run_shader_mad("v_mad_i16 v0, v1, v2, v3 clamp", s0 & MASK16, s1 & MASK16, s2 & MASK16)
                self.assertEqual(got, expected)

    def test_v_mad_i16_clamp_preserves_in_bounds(self):
        """v_mad_i16 clamp preserves values within [-32768, 32767]."""
        in_bounds_cases = [
            (-1, 4, 0, -4),
            (-100, 50, 10, -4990),
            (32767, 1, 0, 32767),
            (-32768, 1, 0, -32768),
            (0, -50, 0, 0),
            (10, 10, 10, 110),
        ]
        for s0, s1, s2, want in in_bounds_cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                expected = (OLD & 0xFFFF0000) | (want & MASK16)
                self.assertEqual(ref_mad_i16(s0 & MASK16, s1 & MASK16, s2 & MASK16, clamp=True), expected)
                got = run_shader_mad("v_mad_i16 v0, v1, v2, v3 clamp", s0 & MASK16, s1 & MASK16, s2 & MASK16)
                self.assertEqual(got, expected)

    def test_clamp_combined_with_opsel_and_dst_hi(self):
        """Clamp modifier works correctly with arbitrary op_sel and dst_hi=1."""
        # Packed words: v1 hi=32767, lo=2; v2 hi=-2, lo=3; v3 hi=0, lo=10
        v1 = (0x7FFF << 16) | 2
        v2 = (0xFFFE << 16) | 3
        v3 = (0x0000 << 16) | 10
        # Select v1 hi (32767), v2 hi (-2), v3 hi (0) -> -65534 clamped to -32768 (0x8000), dst_hi=1
        line = "v_mad_i16 v0, v1, v2, v3 op_sel:[1,1,1,1] clamp"
        old = 0x55551234
        expected = ref_mad_i16(v1, v2, v3, sel=(1, 1, 1), dst_hi=1, clamp=True, old_dst=old)
        got = run_shader_mad(line, v1, v2, v3, old=old)
        self.assertEqual(got, expected)
        self.assertEqual(got >> 16, 0x8000)
        self.assertEqual(got & MASK16, 0x1234)


class TestMadU16TranslationAndInterp(unittest.TestCase):
    """End-to-end translation and execution of v_mad_u16 via rdna2spirv and spirv_mini_interp."""

    def test_basic_known_values(self):
        """Known integer values match expected results in spirv_mini_interp."""
        self.assertEqual(run_shader_mad("v_mad_u16 v0, v1, v2, v3", 7, 11, 13), (OLD & 0xFFFF0000) | 90)
        self.assertEqual(run_shader_mad("v_mad_u16 v0, v1, v2, v3", 100, 200, 300), (OLD & 0xFFFF0000) | 20300)
        self.assertEqual(
            run_shader_mad("v_mad_u16 v0, v1, v2, v3", 65535, 2, 5),
            (OLD & 0xFFFF0000) | ((65535 * 2 + 5) & MASK16),
        )
        self.assertEqual(
            run_shader_mad("v_mad_u16 v0, v1, v2, v3", 0xFFFF, 0xFFFF, 0xFFFF),
            (OLD & 0xFFFF0000) | ((0xFFFF * 0xFFFF + 0xFFFF) & MASK16),
        )

    def test_keeps_high_half_of_destination(self):
        """Untouched high half of destination register is preserved."""
        for old in (0x12345678, 0xCAFE0000, 0x0000BEEF, 0xFEEDFACE):
            with self.subTest(old=hex(old)):
                got = run_shader_mad("v_mad_u16 v0, v1, v2, v3", 3, 5, 7, old=old)
                self.assertEqual(got >> 16, old >> 16)
                self.assertEqual(got & MASK16, 22)

    def test_source_op_sel_patterns(self):
        """Selective extraction of low and high halfwords via op_sel:[s0, s1, s2, 0]."""
        v1 = 0x0007_0003
        v2 = 0x000B_0005
        v3 = 0x000D_0002
        cases = [
            ((0, 0, 0), ref_mad_u16(v1, v2, v3, sel=(0, 0, 0))),
            ((1, 0, 0), ref_mad_u16(v1, v2, v3, sel=(1, 0, 0))),
            ((0, 1, 0), ref_mad_u16(v1, v2, v3, sel=(0, 1, 0))),
            ((0, 0, 1), ref_mad_u16(v1, v2, v3, sel=(0, 0, 1))),
            ((1, 1, 0), ref_mad_u16(v1, v2, v3, sel=(1, 1, 0))),
            ((1, 0, 1), ref_mad_u16(v1, v2, v3, sel=(1, 0, 1))),
            ((0, 1, 1), ref_mad_u16(v1, v2, v3, sel=(0, 1, 1))),
            ((1, 1, 1), ref_mad_u16(v1, v2, v3, sel=(1, 1, 1))),
        ]
        for sel, expected in cases:
            line = f"v_mad_u16 v0, v1, v2, v3 op_sel:[{sel[0]},{sel[1]},{sel[2]},0]"
            with self.subTest(sel=sel):
                self.assertEqual(run_shader_mad(line, v1, v2, v3), expected)

    def test_dst_hi_op_sel(self):
        """Destination high-half placement preserves low half of destination."""
        v1 = 0x0007_0003
        v2 = 0x000B_0005
        v3 = 0x000D_0002
        line = "v_mad_u16 v0, v1, v2, v3 op_sel:[0,0,0,1]"
        expected = ref_mad_u16(v1, v2, v3, sel=(0, 0, 0), dst_hi=1, old_dst=0x12345678)
        got = run_shader_mad(line, v1, v2, v3, old=0x12345678)
        self.assertEqual(got, expected)
        self.assertEqual(got & MASK16, 0x5678)
        self.assertEqual(got >> 16, 17)

    def test_random_against_reference_oracle(self):
        """Deterministic pseudo-random differential test against the reference oracle."""
        rng = random.Random(0x1616)
        candidates = [0, 1, 2, 255, 256, 32767, 32768, 65534, 65535]
        for _ in range(25):
            v1 = rng.choice(candidates) if rng.random() < 0.3 else rng.randrange(0x100000000)
            v2 = rng.choice(candidates) if rng.random() < 0.3 else rng.randrange(0x100000000)
            v3 = rng.choice(candidates) if rng.random() < 0.3 else rng.randrange(0x100000000)
            sel = (rng.randrange(2), rng.randrange(2), rng.randrange(2))
            dst_hi = rng.randrange(2)
            clamp = rng.random() < 0.5
            old = rng.randrange(0x100000000)
            clamp_str = " clamp" if clamp else ""
            line = f"v_mad_u16 v0, v1, v2, v3 op_sel:[{sel[0]},{sel[1]},{sel[2]},{dst_hi}]{clamp_str}"
            expected = ref_mad_u16(v1, v2, v3, sel=sel, dst_hi=dst_hi, clamp=clamp, old_dst=old)
            with self.subTest(sel=sel, dst_hi=dst_hi, clamp=clamp):
                self.assertEqual(run_shader_mad(line, v1, v2, v3, old=old), expected)


class TestMadI16TranslationAndInterp(unittest.TestCase):
    """End-to-end translation and execution of v_mad_i16 via rdna2spirv and spirv_mini_interp."""

    def test_known_signed_values(self):
        """Known signed integer values match expected results."""
        # -2 * 3 + 5 = -1 (0xFFFF)
        self.assertEqual(run_shader_mad("v_mad_i16 v0, v1, v2, v3", 0xFFFE, 3, 5), (OLD & 0xFFFF0000) | 0xFFFF)
        # 32767 * 2 + 0 = 65534 -> -2 (0xFFFE)
        self.assertEqual(run_shader_mad("v_mad_i16 v0, v1, v2, v3", 0x7FFF, 2, 0), (OLD & 0xFFFF0000) | 0xFFFE)
        # 32767 * 2 + 0 clamp -> 32767 (0x7FFF)
        self.assertEqual(run_shader_mad("v_mad_i16 v0, v1, v2, v3 clamp", 0x7FFF, 2, 0), (OLD & 0xFFFF0000) | 0x7FFF)
        # 32767 * -2 + 0 clamp -> -32768 (0x8000)
        self.assertEqual(run_shader_mad("v_mad_i16 v0, v1, v2, v3 clamp", 0x7FFF, 0xFFFE, 0), (OLD & 0xFFFF0000) | 0x8000)
        # -1 * 4 + 0 clamp -> -4 (0xFFFC)
        self.assertEqual(run_shader_mad("v_mad_i16 v0, v1, v2, v3 clamp", 0xFFFF, 4, 0), (OLD & 0xFFFF0000) | 0xFFFC)

    def test_random_signed_against_reference_oracle(self):
        """Deterministic pseudo-random differential test of v_mad_i16 against ref_mad_i16."""
        rng = random.Random(0x1617)
        candidates = [0, 1, 0x7FFF, 0x8000, 0x8001, 0xFFFE, 0xFFFF]
        for _ in range(25):
            v1 = rng.choice(candidates) if rng.random() < 0.3 else rng.randrange(0x100000000)
            v2 = rng.choice(candidates) if rng.random() < 0.3 else rng.randrange(0x100000000)
            v3 = rng.choice(candidates) if rng.random() < 0.3 else rng.randrange(0x100000000)
            sel = (rng.randrange(2), rng.randrange(2), rng.randrange(2))
            dst_hi = rng.randrange(2)
            clamp = rng.random() < 0.5
            old = rng.randrange(0x100000000)
            clamp_str = " clamp" if clamp else ""
            line = f"v_mad_i16 v0, v1, v2, v3 op_sel:[{sel[0]},{sel[1]},{sel[2]},{dst_hi}]{clamp_str}"
            expected = ref_mad_i16(v1, v2, v3, sel=sel, dst_hi=dst_hi, clamp=clamp, old_dst=old)
            with self.subTest(sel=sel, dst_hi=dst_hi, clamp=clamp):
                self.assertEqual(run_shader_mad(line, v1, v2, v3, old=old), expected)


if __name__ == "__main__":
    unittest.main()
