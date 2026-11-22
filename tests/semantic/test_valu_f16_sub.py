"""Semantic interpreter tests for AMD RDNA2 half-precision float subtraction (v_sub_f16).

Covers:
  - v_sub_f16: D.f16 = S0.f16 - S1.f16 (VOP2 0x33 on GFX10).
  - VOP2 _e32 form: writes low 16-bit half of destination VGPR and keeps the high 16-bit half
    (per docs/design.md, "16-bit results").
  - VOP3 _e64 form: op_sel:[src0, src1, dst] selects low (0) or high (1) 16-bit half for operands
    and destination.
  - VOP3 modifiers: source negation (-s), absolute value (|s|), negative absolute value (-|s|),
    and output modifiers (clamp to [0.0, 1.0], scale mul:2, mul:4, div:2).
  - Inline constants, SGPR scalar broadcast sources, destination aliasing, and duality with v_subrev_f16.

All operations are executed end-to-end through tools/rdna2spirv.py and
tests/semantic/spirv_mini_interp.py and verified against a deterministic reference oracle.

Standard library only: zero game data, no GPU hardware required.
"""
from __future__ import annotations

import gc
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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xAAAA5555  # Prior destination contents: unwritten half must survive

# Representative IEEE 754 half-precision (binary16) encodings
H_POS_ZERO = 0x0000
H_NEG_ZERO = 0x8000
H_0P25 = 0x3400        # 0.25
H_NEG_0P25 = 0xB400    # -0.25
H_0P5 = 0x3800         # 0.5
H_NEG_0P5 = 0xB800     # -0.5
H_0P75 = 0x3A00        # 0.75
H_NEG_0P75 = 0xBA00    # -0.75
H_1P0 = 0x3C00         # 1.0
H_NEG_1P0 = 0xBC00     # -1.0
H_1P25 = 0x3D00        # 1.25
H_1P5 = 0x3E00         # 1.5
H_NEG_1P5 = 0xBE00     # -1.5
H_2P0 = 0x4000         # 2.0
H_NEG_2P0 = 0xC000     # -2.0
H_2P5 = 0x4100         # 2.5
H_3P0 = 0x4200         # 3.0
H_3P5 = 0x4300         # 3.5
H_4P0 = 0x4400         # 4.0
H_5P0 = 0x4500         # 5.0
H_8P0 = 0x4800         # 8.0
H_MAX = 0x7BFF         # 65504.0 (largest finite f16)
H_NEG_MAX = 0xFBFF     # -65504.0
H_POS_INF = 0x7C00     # +infinity
H_NEG_INF = 0xFC00     # -infinity
H_QNAN = 0x7E01        # quiet NaN


def h2f(h: int) -> float:
    """Unpack 16-bit half-precision float to Python float."""
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    """Pack Python float to 16-bit half-precision bit pattern."""
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return H_POS_INF if x > 0 else H_NEG_INF


def half(word: int, hi: int) -> float:
    """Extract half-precision float from low (hi=0) or high (hi=1) word half."""
    return h2f((word >> 16) if hi else word)


def merge_half(old: int, h: int, dst_hi: int) -> int:
    """Merge 16-bit result into 32-bit register, preserving unwritten half."""
    h &= 0xFFFF
    return ((h << 16) | (old & 0xFFFF)) if dst_hi else ((old & 0xFFFF0000) | h)


def clamp01(x: float) -> float:
    """Clamp float to [0.0, 1.0]."""
    return min(max(x, 0.0), 1.0)


def ref_v_sub_f16(s0: int, s1: int) -> int:
    """Reference oracle for v_sub_f16: S0.f16 - S1.f16."""
    return f2h(h2f(s0) - h2f(s1))


def ref_v_subrev_f16(s0: int, s1: int) -> int:
    """Reference oracle for v_subrev_f16: S1.f16 - S0.f16."""
    return f2h(h2f(s1) - h2f(s0))


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_sub_f16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_sub_f16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_sub_f16(line: str, v1: int, v2: int, old: int = OLD) -> int:
    """Seed v1, v2 and destination v0, run instruction line, return v0."""
    if _TMP_DIR is None:
        setUpModule()
    lines = [
        f"v_mov_b32 v1, {v1:#x}",
        f"v_mov_b32 v2, {v2:#x}",
        f"v_mov_b32 v0, {old:#x}",
        line,
    ]
    asm = (
        "; test_valu_f16_sub\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class VectorSubF16ReferenceTests(unittest.TestCase):
    """Tests for the reference mathematical oracle of half-precision subtraction."""

    def test_basic_subtraction(self):
        self.assertEqual(ref_v_sub_f16(H_1P5, H_0P5), H_1P0)
        self.assertEqual(ref_v_sub_f16(H_5P0, H_2P5), H_2P5)
        self.assertEqual(ref_v_sub_f16(H_0P5, H_1P5), H_NEG_1P0)
        self.assertEqual(ref_v_sub_f16(H_2P0, H_4P0), H_NEG_2P0)

    def test_identity_and_cancellation(self):
        for val in (H_0P5, H_1P0, H_1P5, H_2P0, H_3P0, H_MAX):
            self.assertEqual(ref_v_sub_f16(val, H_POS_ZERO), val)
            self.assertEqual(ref_v_sub_f16(H_POS_ZERO, val), f2h(-h2f(val)))
            # Cancellation x - x = +0.0 in round-to-nearest
            self.assertEqual(ref_v_sub_f16(val, val) & 0x7FFF, 0x0000)

    def test_duality_with_subrev(self):
        pairs = [
            (H_1P5, H_0P5),
            (H_0P5, H_1P5),
            (H_3P0, H_1P0),
            (H_2P5, H_4P0),
            (H_0P25, H_0P75),
        ]
        for a, b in pairs:
            self.assertEqual(ref_v_sub_f16(a, b), ref_v_subrev_f16(b, a))

    def test_merge_half_preserves_unwritten_half(self):
        self.assertEqual(merge_half(OLD, H_1P0, dst_hi=0), 0xAAAA3C00)
        self.assertEqual(merge_half(OLD, H_1P0, dst_hi=1), 0x3C005555)


class VectorSubF16SpirvTests(unittest.TestCase):
    """Tests for v_sub_f16 lowering to SPIR-V OpFSub and execution in spirv_mini_interp."""

    def test_vop2_e32_computes_s0_minus_s1_and_preserves_dst_hi(self):
        # 1.5 (0x3E00) - 0.5 (0x3800) = 1.0 (0x3C00)
        got = run_sub_f16("v_sub_f16_e32 v0, v1, v2", H_1P5, H_0P5)
        self.assertEqual(got, merge_half(OLD, H_1P0, 0))

    def test_vop2_e32_negative_difference(self):
        # 0.5 (0x3800) - 1.5 (0x3E00) = -1.0 (0xBC00)
        got = run_sub_f16("v_sub_f16_e32 v0, v1, v2", H_0P5, H_1P5)
        self.assertEqual(got, merge_half(OLD, H_NEG_1P0, 0))

    def test_vop3_op_sel_all_combinations(self):
        # v1: lo = 1.5 (0x3E00), hi = 4.0 (0x4400)
        # v2: lo = 0.5 (0x3800), hi = 2.0 (0x4000)
        v1 = (H_4P0 << 16) | H_1P5
        v2 = (H_2P0 << 16) | H_0P5
        for s0_hi, s1_hi, d_hi in itertools.product((0, 1), repeat=3):
            line = f"v_sub_f16_e64 v0, v1, v2 op_sel:[{s0_hi},{s1_hi},{d_hi}]"
            with self.subTest(line=line):
                val0 = half(v1, s0_hi)
                val1 = half(v2, s1_hi)
                expected_diff = f2h(val0 - val1)
                expected = merge_half(OLD, expected_diff, d_hi)
                self.assertEqual(run_sub_f16(line, v1, v2), expected)

    def test_vop3_source_modifiers(self):
        v1 = H_1P5
        v2 = H_0P5
        # Negation: -v1 - v2 = -1.5 - 0.5 = -2.0
        self.assertEqual(
            run_sub_f16("v_sub_f16_e64 v0, -v1, v2", v1, v2),
            merge_half(OLD, H_NEG_2P0, 0),
        )
        # Negation on src1: v1 - (-v2) = 1.5 - (-0.5) = 2.0
        self.assertEqual(
            run_sub_f16("v_sub_f16_e64 v0, v1, -v2", v1, v2),
            merge_half(OLD, H_2P0, 0),
        )
        # Absolute value: |v1| - |v2| with negative input
        v1_neg = H_NEG_1P5
        v2_neg = H_NEG_0P5
        self.assertEqual(
            run_sub_f16("v_sub_f16_e64 v0, |v1|, |v2|", v1_neg, v2_neg),
            merge_half(OLD, H_1P0, 0),
        )
        # Negative absolute value: -|v1| - v2 = -1.5 - 0.5 = -2.0
        self.assertEqual(
            run_sub_f16("v_sub_f16_e64 v0, -|v1|, v2", v1, v2),
            merge_half(OLD, H_NEG_2P0, 0),
        )

    def test_vop3_clamp_modifier(self):
        v1 = H_1P5
        v2 = H_0P5
        # 1.5 - (-0.5) = 2.0; clamped to 1.0
        self.assertEqual(
            run_sub_f16("v_sub_f16_e64 v0, v1, -v2 clamp", v1, v2),
            merge_half(OLD, H_1P0, 0),
        )
        # 0.5 - 1.5 = -1.0; clamped to 0.0
        self.assertEqual(
            run_sub_f16("v_sub_f16_e64 v0, v2, v1 clamp", v1, v2),
            merge_half(OLD, H_POS_ZERO, 0),
        )

    def test_vop3_scale_output_modifiers(self):
        v1 = H_1P5
        v2 = H_0P5
        # 1.5 - 0.5 = 1.0 -> mul:2 = 2.0
        self.assertEqual(
            run_sub_f16("v_sub_f16_e64 v0, v1, v2 mul:2", v1, v2),
            merge_half(OLD, H_2P0, 0),
        )
        # 1.5 - 0.5 = 1.0 -> mul:4 = 4.0
        self.assertEqual(
            run_sub_f16("v_sub_f16_e64 v0, v1, v2 mul:4", v1, v2),
            merge_half(OLD, H_4P0, 0),
        )
        # 1.5 - 0.5 = 1.0 -> div:2 = 0.5
        self.assertEqual(
            run_sub_f16("v_sub_f16_e64 v0, v1, v2 div:2", v1, v2),
            merge_half(OLD, H_0P5, 0),
        )

    def test_inline_constants_and_sgpr(self):
        if _TMP_DIR is None:
            setUpModule()
        # Inline float constant 1.0 (0x3C00) in s0: 1.0 - 0.5 = 0.5
        lines = [
            f"v_mov_b32 v0, {OLD:#x}",
            f"v_mov_b32 v1, {H_0P5:#x}",
            "v_sub_f16_e32 v0, 1.0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        asm = "; test_inline_const\n" + "".join(f"\t{ln}\n" for ln in lines)
        with open(_SRC_PATH, "w", encoding="utf-8") as f:
            f.write(asm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(_SRC_PATH)
        got = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(got, merge_half(OLD, H_0P5, 0))

        # SGPR scalar source: s0 loaded with 2.5
        lines_sgpr = [
            f"v_mov_b32 v0, {OLD:#x}",
            f"s_mov_b32 s0, {H_2P5:#x}",
            f"v_mov_b32 v1, {H_1P5:#x}",
            "v_sub_f16_e32 v0, s0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        asm_sgpr = "; test_sgpr\n" + "".join(f"\t{ln}\n" for ln in lines_sgpr)
        with open(_SRC_PATH, "w", encoding="utf-8") as f:
            f.write(asm_sgpr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv_sgpr = rdna2spirv.translate(_SRC_PATH)
        got_sgpr = si.run_fragment(spv_sgpr)["%out_mrt0"][0]
        self.assertEqual(got_sgpr, merge_half(OLD, H_1P0, 0))

    def test_destination_aliasing(self):
        if _TMP_DIR is None:
            setUpModule()
        # Aliasing dst with src0: v1 = v1 - v2
        lines = [
            f"v_mov_b32 v1, {OLD:#x}",
            f"v_mov_b32 v2, {H_0P5:#x}",
            "v_mov_b32 v1, 0xAAAA3E00",
            "v_sub_f16_e32 v1, v1, v2",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        asm = "; test_dst_alias\n" + "".join(f"\t{ln}\n" for ln in lines)
        with open(_SRC_PATH, "w", encoding="utf-8") as f:
            f.write(asm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(_SRC_PATH)
        got = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(got, merge_half(0xAAAA3E00, H_1P0, 0))

    def test_differential_fuzz_finite_pairs(self):
        rng = random.Random(0x516)
        candidates = [
            H_0P25, H_NEG_0P25, H_0P5, H_NEG_0P5, H_0P75, H_NEG_0P75,
            H_1P0, H_NEG_1P0, H_1P5, H_NEG_1P5, H_2P0, H_NEG_2P0,
            H_2P5, H_3P0, H_4P0,
        ]
        for _ in range(30):
            x = rng.choice(candidates)
            y = rng.choice(candidates)
            expected = ref_v_sub_f16(x, y)
            with self.subTest(x=hex(x), y=hex(y)):
                got = run_sub_f16("v_sub_f16_e32 v0, v1, v2", x, y)
                self.assertEqual(got, merge_half(OLD, expected, 0))


if __name__ == "__main__":
    unittest.main()
