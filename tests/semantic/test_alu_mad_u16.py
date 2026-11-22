from __future__ import annotations

"""Semantic interpreter tests for RDNA2 16-bit unsigned integer multiply-add (v_mad_u16).

AMD RDNA VALU Instruction Set Architecture (GFX10/RDNA2):
  v_mad_u16 dst, src0, src1, src2 [op_sel:[s0, s1, s2, dst_hi]] [clamp]

Mathematical specification:
  - Unsigned 16-bit operands S0, S1, S2 in [0, 65535] (0x0000 to 0xFFFF).
  - op_sel bits 0..2 select whether each source operand is extracted from the low half-word
    (bits 15:0 if 0) or high half-word (bits 31:16 if 1) of the respective 32-bit register.
  - Multiplication and addition are computed in exact 32-bit integer arithmetic without
    intermediate truncation:
      P = S0 * S1
      T = P + S2
  - Without clamp, the 32-bit result wraps modulo 65536 (retaining the lower 16 bits):
      D = T & 0xFFFF
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


def half16_u(val: int, hi: int | bool) -> int:
    """Extract low or high 16-bit half-word as unsigned integer."""
    return ((val >> 16) if hi else val) & MASK16


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
    """Reference oracle for RDNA2 v_mad_u16."""
    x = half16_u(s0, sel[0])
    y = half16_u(s1, sel[1])
    z = half16_u(s2, sel[2])
    tot = x * y + z
    if clamp:
        tot = min(MASK16, tot)
    r16 = tot & MASK16
    return merge_dst16(old_dst, r16, dst_hi)


def run_shader_mad_u16(line: str, v1: int, v2: int, v3: int, old: int = OLD) -> int:
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
    asm = "; test_mad_u16\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
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


class TestMadU16TranslationAndInterp(unittest.TestCase):
    """End-to-end translation and execution of v_mad_u16 via rdna2spirv and spirv_mini_interp."""

    def test_basic_known_values(self):
        """Known integer values match expected results in spirv_mini_interp."""
        self.assertEqual(run_shader_mad_u16("v_mad_u16 v0, v1, v2, v3", 7, 11, 13), (OLD & 0xFFFF0000) | 90)
        self.assertEqual(run_shader_mad_u16("v_mad_u16 v0, v1, v2, v3", 100, 200, 300), (OLD & 0xFFFF0000) | 20300)
        self.assertEqual(
            run_shader_mad_u16("v_mad_u16 v0, v1, v2, v3", 65535, 2, 5),
            (OLD & 0xFFFF0000) | ((65535 * 2 + 5) & MASK16),
        )
        self.assertEqual(
            run_shader_mad_u16("v_mad_u16 v0, v1, v2, v3", 0xFFFF, 0xFFFF, 0xFFFF),
            (OLD & 0xFFFF0000) | ((0xFFFF * 0xFFFF + 0xFFFF) & MASK16),
        )

    def test_keeps_high_half_of_destination(self):
        """Untouched high half of destination register is preserved."""
        for old in (0x12345678, 0xCAFE0000, 0x0000BEEF, 0xFEEDFACE):
            with self.subTest(old=hex(old)):
                got = run_shader_mad_u16("v_mad_u16 v0, v1, v2, v3", 3, 5, 7, old=old)
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
                self.assertEqual(run_shader_mad_u16(line, v1, v2, v3), expected)

    def test_dst_hi_op_sel(self):
        """Destination high-half placement preserves low half of destination."""
        v1 = 0x0007_0003
        v2 = 0x000B_0005
        v3 = 0x000D_0002
        line = "v_mad_u16 v0, v1, v2, v3 op_sel:[0,0,0,1]"
        expected = ref_mad_u16(v1, v2, v3, sel=(0, 0, 0), dst_hi=1, old_dst=0x12345678)
        got = run_shader_mad_u16(line, v1, v2, v3, old=0x12345678)
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
            old = rng.randrange(0x100000000)
            line = f"v_mad_u16 v0, v1, v2, v3 op_sel:[{sel[0]},{sel[1]},{sel[2]},{dst_hi}]"
            expected = ref_mad_u16(v1, v2, v3, sel=sel, dst_hi=dst_hi, old_dst=old)
            with self.subTest(sel=sel, dst_hi=dst_hi):
                self.assertEqual(run_shader_mad_u16(line, v1, v2, v3, old=old), expected)


if __name__ == "__main__":
    unittest.main()
