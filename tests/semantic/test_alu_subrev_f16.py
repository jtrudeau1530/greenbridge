"""Semantic interpreter tests for RDNA2 v_subrev_f16 and v_subrev_nc_u32.

v_subrev_f16: D.f16 = S1.f16 - S0.f16 (VOP2 0x34 on GFX10), the reversed form of v_sub_f16. Like the other f16
VOP2/VOP3 ops it writes the low half of the destination and keeps the high half; in the VOP3 form op_sel bit 0 / 1
reads the high half of src0 / src1 and bit 2 writes the high half of the destination instead (docs/design.md,
"16-bit results"). Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py and compares the exported word with a plain-Python reference: the difference
of the two f16 values computed in double precision and rounded once to f16 (round to nearest even).

Also covers v_subrev_nc_u32: D.u32 = (S1.u32 - S0.u32) & 0xFFFFFFFF, the non-carry 32-bit reverse subtract.

Standard library only: zero game data, no GPU hardware required.
"""
from __future__ import annotations

import itertools
import math
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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xAAAA5555
MASK32 = 0xFFFFFFFF

# Half-precision constants
H_POS_ZERO = 0x0000
H_NEG_ZERO = 0x8000
H_HALF = 0x3800      # 0.5
H_ONE = 0x3C00       # 1.0
H_ONE_P_FIVE = 0x3E00  # 1.5
H_TWO = 0x4000       # 2.0
H_THREE = 0x4200     # 3.0
H_FOUR = 0x4400      # 4.0
H_NEG_HALF = 0xB800  # -0.5
H_NEG_ONE = 0xBC00   # -1.0
H_NEG_TWO = 0xC000   # -2.0
H_POS_INF = 0x7C00   # +inf
H_NEG_INF = 0xFC00   # -inf
H_QNAN = 0x7E00      # quiet NaN
H_SNAN = 0x7C01      # signaling NaN
H_MAX_FINITE = 0x7BFF  # 65504.0
H_NEG_MAX = 0xFBFF   # -65504.0

A = 0x44003E00       # v1: lo 1.5, hi 4.0
B = 0x40003800       # v2: lo 0.5, hi 2.0


def h2f(h: int) -> float:
    """Unpack 16-bit integer as IEEE 754 half-precision float."""
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    """Pack float to 16-bit IEEE 754 half-precision integer with round-to-nearest-even."""
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return H_POS_INF if x > 0 else H_NEG_INF


def is_nan_f16(h: int) -> bool:
    """Check if 16-bit integer represents an IEEE 754 half-precision NaN."""
    return (h & 0x7C00) == 0x7C00 and (h & 0x03FF) != 0


def is_qnan_f16(h: int) -> bool:
    """Check if 16-bit integer represents a quiet NaN."""
    return is_nan_f16(h) and bool(h & 0x0200)


def is_snan_f16(h: int) -> bool:
    """Check if 16-bit integer represents a signaling NaN."""
    return is_nan_f16(h) and not bool(h & 0x0200)


def is_inf_f16(h: int) -> bool:
    """Check if 16-bit integer represents an IEEE 754 infinity."""
    return (h & 0x7FFF) == 0x7C00


def half(word: int, hi: int) -> float:
    """Extract float from low (hi=0) or high (hi=1) 16 bits of a 32-bit word."""
    return h2f(word >> 16 if hi else word)


def merge_half(old: int, h: int, dst_hi: int) -> int:
    """Merge 16-bit result into low (dst_hi=0) or high (dst_hi=1) half of old word."""
    h &= 0xFFFF
    return ((h << 16) | (old & 0xFFFF)) if dst_hi else ((old & 0xFFFF0000) | h)


def pack16(lo: int, hi: int) -> int:
    """Pack two 16-bit values into a 32-bit register word."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def ref_subrev_f16(s0: int, s1: int) -> int:
    """Reference implementation of v_subrev_f16: D.f16 = S1.f16 - S0.f16."""
    s0 &= 0xFFFF
    s1 &= 0xFFFF
    if is_nan_f16(s0) or is_nan_f16(s1):
        return H_QNAN
    inf0, inf1 = is_inf_f16(s0), is_inf_f16(s1)
    if inf0 and inf1:
        if (s0 >> 15) == (s1 >> 15):
            return H_QNAN
        return s1
    if inf1:
        return s1
    if inf0:
        return H_NEG_INF if (s0 >> 15) == 0 else H_POS_INF
    f0, f1 = h2f(s0), h2f(s1)
    diff = f1 - f0
    if diff == 0.0:
        if s1 == H_NEG_ZERO and s0 == H_POS_ZERO:
            return H_NEG_ZERO
        return H_POS_ZERO
    return f2h(diff)


def ref_sub_f16(s0: int, s1: int) -> int:
    """Reference implementation of v_sub_f16: D.f16 = S0.f16 - S1.f16."""
    return ref_subrev_f16(s1, s0)


def ref_subrev_nc_u32(s0: int, s1: int) -> int:
    """Reference implementation of v_subrev_nc_u32: D.u32 = (S1.u32 - S0.u32) & 0xFFFFFFFF."""
    return ((s1 & MASK32) - (s0 & MASK32)) & MASK32


def ref_sub_nc_u32(s0: int, s1: int) -> int:
    """Reference implementation of v_sub_nc_u32: D.u32 = (S0.u32 - S1.u32) & 0xFFFFFFFF."""
    return ((s0 & MASK32) - (s1 & MASK32)) & MASK32


def ref_subrev_f16_opsel(
    w0: int,
    w1: int,
    old: int,
    sel0: int,
    sel1: int,
    dst_hi: int,
    neg0: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Compute VOP3 v_subrev_f16 result with op_sel and input/output modifiers."""
    h0 = (w0 >> 16 if sel0 else w0) & 0xFFFF
    h1 = (w1 >> 16 if sel1 else w1) & 0xFFFF
    if neg0:
        h0 ^= 0x8000
    if neg1:
        h1 ^= 0x8000
    res = ref_subrev_f16(h0, h1)
    if clamp and not is_nan_f16(res):
        f = h2f(res)
        f_clamped = max(0.0, min(1.0, f))
        res = f2h(f_clamped)
    return merge_half(old, res, dst_hi)


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_subrev_f16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_subrev_f16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run(line: str, v1: int, v2: int, old: int = OLD) -> int:
    """Seed v1, v2 and destination v0, run assembly line, return exported v0."""
    return run_shader([line], v1=v1, v2=v2, old=old)


def run_shader(lines: list[str], v1: int = 0, v2: int = 0, old: int = OLD) -> int:
    """Seed v1, v2 and destination v0, run lines, return exported v0."""
    if _TMP_DIR is None:
        setUpModule()
    preamble = [
        f"v_mov_b32 v1, {v1 & MASK32:#x}",
        f"v_mov_b32 v2, {v2 & MASK32:#x}",
        f"v_mov_b32 v0, {old & MASK32:#x}",
    ]
    asm = (
        "; test_subrev_f16\n"
        + "".join(f"\t{ln}\n" for ln in preamble + list(lines))
        + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class SubrevF16ReferenceOracleTests(unittest.TestCase):
    """Mathematical properties and correctness of the half-precision reverse subtraction oracle."""

    def test_formula_s1_minus_s0(self):
        # D = S1 - S0 (reverse subtraction)
        self.assertEqual(ref_subrev_f16(H_ONE_P_FIVE, H_HALF), H_NEG_ONE)
        self.assertEqual(ref_subrev_f16(H_HALF, H_ONE_P_FIVE), H_ONE)
        self.assertEqual(ref_subrev_f16(H_ONE, H_THREE), H_TWO)
        self.assertEqual(ref_subrev_f16(H_TWO, H_ONE), H_NEG_ONE)

    def test_non_commutativity(self):
        # Subtraction is non-commutative: (S1 - S0) != (S0 - S1) when S0 != S1.
        pairs = [
            (H_ONE_P_FIVE, H_HALF),
            (H_ONE, H_THREE),
            (H_TWO, H_FOUR),
            (H_NEG_ONE, H_ONE),
        ]
        for s0, s1 in pairs:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d_rev = ref_subrev_f16(s0, s1)
                d_rev_swapped = ref_subrev_f16(s1, s0)
                self.assertNotEqual(d_rev, d_rev_swapped)

    def test_anti_symmetry(self):
        # Subtraction is anti-symmetric: (S1 - S0) == -(S0 - S1) for finite non-zero values.
        pairs = [
            (H_ONE_P_FIVE, H_HALF),
            (H_ONE, H_THREE),
            (H_TWO, H_FOUR),
            (H_HALF, H_TWO),
            (H_NEG_ONE, H_TWO),
        ]
        for s0, s1 in pairs:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                r1 = ref_subrev_f16(s0, s1)
                r2 = ref_subrev_f16(s1, s0)
                # Opposite signs: sign bit toggled, same magnitude
                self.assertEqual(r1 ^ 0x8000, r2)
                self.assertAlmostEqual(h2f(r1), -h2f(r2), places=3)

    def test_identity_properties(self):
        values = [H_HALF, H_ONE, H_ONE_P_FIVE, H_TWO, H_THREE, H_FOUR, H_NEG_ONE, H_NEG_TWO]
        for v in values:
            with self.subTest(v=hex(v)):
                # S1 - 0 == S1
                self.assertEqual(ref_subrev_f16(H_POS_ZERO, v), v)
                # S1 - S1 == 0
                self.assertEqual(ref_subrev_f16(v, v), H_POS_ZERO)
                # 0 - S0 == -S0
                self.assertEqual(ref_subrev_f16(v, H_POS_ZERO), v ^ 0x8000)

    def test_reversal_duality_with_sub(self):
        # v_subrev_f16(S0, S1) == v_sub_f16(S1, S0)
        rng = random.Random(0x3416)
        samples = [H_POS_ZERO, H_HALF, H_ONE, H_ONE_P_FIVE, H_TWO, H_THREE, H_FOUR, H_NEG_ONE, H_NEG_TWO]
        for _ in range(50):
            s0 = rng.choice(samples)
            s1 = rng.choice(samples)
            self.assertEqual(ref_subrev_f16(s0, s1), ref_sub_f16(s1, s0))

    def test_u32_reverse_subtraction(self):
        # Reference tests for v_subrev_nc_u32: D = S1 - S0 (mod 2^32)
        self.assertEqual(ref_subrev_nc_u32(30, 100), 70)
        self.assertEqual(ref_subrev_nc_u32(100, 30), (30 - 100) & MASK32)
        self.assertEqual(ref_subrev_nc_u32(0, 42), 42)
        self.assertEqual(ref_subrev_nc_u32(42, 0), (-42) & MASK32)
        self.assertEqual(ref_subrev_nc_u32(0x80000000, 0x7FFFFFFF), 0xFFFFFFFF)
        self.assertEqual(ref_subrev_nc_u32(0x7FFFFFFF, 0x80000000), 1)

        # Anti-symmetry for u32
        for a, b in [(10, 50), (100, 25), (0x12345678, 0x87654321)]:
            fwd = ref_sub_nc_u32(a, b)
            rev = ref_subrev_nc_u32(a, b)
            self.assertEqual((fwd + rev) & MASK32, 0)
            self.assertEqual(ref_subrev_nc_u32(a, b), ref_sub_nc_u32(b, a))


class SubrevF16ShaderTranslationTests(unittest.TestCase):
    """End-to-end pixel shader execution comparing rdna2spirv output against reference oracle."""

    def test_vop2_is_src1_minus_src0_and_keeps_the_high_half(self):
        got = run("v_subrev_f16_e32 v0, v1, v2", A, B)
        want = merge_half(OLD, ref_subrev_f16(0x3E00, 0x3800), 0)
        self.assertEqual(got, want)

    def test_vop3_op_sel(self):
        for bits in itertools.product((0, 1), repeat=3):
            line = f"v_subrev_f16_e64 v0, v1, v2 op_sel:[{bits[0]},{bits[1]},{bits[2]}]"
            with self.subTest(line=line):
                want = ref_subrev_f16_opsel(A, B, OLD, bits[0], bits[1], bits[2])
                self.assertEqual(run(line, A, B), want)

    def test_neg_on_src0_and_clamp(self):
        # -(1.5) reversed: 0.5 - (-1.5) = 2.0; clamp then limits it to 1.0.
        self.assertEqual(run("v_subrev_f16_e64 v0, -v1, v2", A, B), merge_half(OLD, f2h(2.0), 0))
        self.assertEqual(run("v_subrev_f16_e64 v0, -v1, v2 clamp", A, B), merge_half(OLD, f2h(1.0), 0))

    def test_matches_v_sub_f16_with_swapped_operands(self):
        rng = random.Random(0x34)
        special = [0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x3555]
        for _ in range(40):
            x, y = (rng.choice(special) if rng.random() < 0.3 else rng.randrange(0x10000) for _ in range(2))
            if (x & 0x7C00) == 0x7C00 or (y & 0x7C00) == 0x7C00:
                continue
            v1, v2 = 0x12340000 | x, 0x56780000 | y
            with self.subTest(x=hex(x), y=hex(y)):
                got = run("v_subrev_f16_e32 v0, v1, v2", v1, v2)
                self.assertEqual(got, run("v_sub_f16_e32 v0, v2, v1", v1, v2))
                want = ref_subrev_f16(x, y)
                if want in (0x0000, 0x8000):
                    self.assertEqual(got & 0xFFFF7FFF, OLD & 0xFFFF0000)
                else:
                    self.assertEqual(got, merge_half(OLD, want, 0))

    def test_v_subrev_nc_u32_shader_execution(self):
        lines = ["v_subrev_nc_u32 v0, v1, v2"]
        self.assertEqual(run_shader(lines, v1=30, v2=100), 70)
        self.assertEqual(run_shader(lines, v1=100, v2=30), (30 - 100) & MASK32)
        self.assertEqual(run_shader(lines, v1=0x80000000, v2=0x7FFFFFFF), 0xFFFFFFFF)

    def test_inline_constants_and_sgpr_sources(self):
        # Test v_subrev_f16 with inline f16 constants
        lines = ["v_subrev_f16_e32 v0, 0x3800, v2"]  # S1 - 0.5 (where S1 is v2.lo = 2.0 -> 1.5)
        self.assertEqual(run_shader(lines, v2=H_TWO), merge_half(OLD, H_ONE_P_FIVE, 0))

        # Test v_subrev_nc_u32 with inline constant and SGPR
        lines_sgpr = [
            "s_mov_b32 s0, 50",
            "v_subrev_nc_u32 v0, s0, v2",  # v2 - 50 = 120 - 50 = 70
        ]
        self.assertEqual(run_shader(lines_sgpr, v2=120), 70)


if __name__ == "__main__":
    unittest.main()
