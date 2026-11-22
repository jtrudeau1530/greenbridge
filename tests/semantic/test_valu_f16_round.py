"""Semantic interpreter tests for RDNA2 16-bit float rounding/fraction VALU ops.

Covers:
  - v_floor_f16: floor(S0.f16)
  - v_ceil_f16:  ceil(S0.f16)
  - v_trunc_f16: trunc(S0.f16)
  - v_rndne_f16: roundToEven(S0.f16)
  - v_fract_f16: S0.f16 - floor(S0.f16)

Per docs/design.md ("16-bit results"), all five write one 16-bit half of the destination VGPR while preserving the
other 16-bit half (VOP2 _e32 writes the low half; VOP3 _e64 op_sel:[src,dst] selects source/destination halves).

v_floor_f16, v_ceil_f16, v_trunc_f16 and v_rndne_f16 are executed end-to-end through tools/rdna2spirv.py and
tests/semantic/spirv_mini_interp.py across finite values, subnormals, +-0.0, .5 ties, +-inf, NaNs, op_sel:[src,dst],
VOP3 source/output modifiers, inline f16 constants, SGPR sources, destination aliasing, and EXEC masking.

Finding (unchanged here): tools/rdna2spirv.py lowers v_fract_f16 to GLSL.std.450 Fract, which
tests/semantic/spirv_mini_interp.py does not yet implement (raising UnsupportedConstruct). v_fract_f16 is therefore
checked via an independent IEEE-754 binary16 oracle plus SPIR-V structural translation verification rather than
end-to-end interpreter execution.
"""
import itertools
import math
import os
import re
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
OLD = 0xBEEF1234

# Representative binary16 encodings.
H_POS_ZERO = 0x0000
H_NEG_ZERO = 0x8000
H_SUB_MIN = 0x0001          # smallest positive subnormal (~5.96e-8)
H_NEG_SUB_MIN = 0x8001      # smallest-magnitude negative subnormal
H_SUB_MAX = 0x03FF          # largest positive subnormal (~6.097e-5)
H_NEG_SUB_MAX = 0x83FF      # largest-magnitude negative subnormal
H_0P25 = 0x3400             # 0.25
H_NEG_0P25 = 0xB400         # -0.25
H_0P5 = 0x3800              # 0.5
H_NEG_0P5 = 0xB800          # -0.5
H_0P75 = 0x3A00             # 0.75
H_NEG_0P75 = 0xBA00         # -0.75
H_1P0 = 0x3C00              # 1.0
H_NEG_1P0 = 0xBC00          # -1.0
H_1P25 = 0x3D00             # 1.25
H_NEG_1P25 = 0xBD00         # -1.25
H_1P5 = 0x3E00              # 1.5
H_NEG_1P5 = 0xBE00          # -1.5
H_1P75 = 0x3F00             # 1.75
H_NEG_1P75 = 0xBF00         # -1.75
H_2P0 = 0x4000              # 2.0
H_NEG_2P0 = 0xC000          # -2.0
H_2P5 = 0x4100              # 2.5
H_NEG_2P5 = 0xC100          # -2.5
H_3P0 = 0x4200              # 3.0
H_NEG_3P0 = 0xC200          # -3.0
H_3P5 = 0x4300              # 3.5
H_NEG_3P5 = 0xC300          # -3.5
H_4P0 = 0x4400              # 4.0
H_NEG_4P0 = 0xC400          # -4.0
H_1022P5 = 0x63FD           # 1022.5 (even integer tie)
H_1023P5 = 0x63FF           # 1023.5 (odd integer tie -> 1024.0)
H_NEG_1022P5 = 0xE3FD       # -1022.5
H_NEG_1023P5 = 0xE3FF       # -1023.5
H_1024 = 0x6400             # 1024.0
H_MAX = 0x7BFF              # 65504.0
H_NEG_MAX = 0xFBFF          # -65504.0
H_POS_INF = 0x7C00          # +inf
H_NEG_INF = 0xFC00          # -inf
H_QNAN = 0x7E01             # +qNaN
H_NEG_QNAN = 0xFE01         # -qNaN
H_SNAN = 0x7C01             # +sNaN


def h2f(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return H_POS_INF if x > 0 else H_NEG_INF


def ref_integral(fn, h: int) -> int:
    x = h2f(h)
    if not math.isfinite(x):
        return h & 0xFFFF
    return f2h(math.copysign(float(fn(x)), x))


def ref_floor(h: int) -> int:
    return ref_integral(math.floor, h)


def ref_ceil(h: int) -> int:
    return ref_integral(math.ceil, h)


def ref_trunc(h: int) -> int:
    return ref_integral(math.trunc, h)


def ref_rndne(h: int) -> int:
    return ref_integral(round, h)


def ref_fract(h: int) -> int:
    x = h2f(h)
    if not math.isfinite(x):
        return H_QNAN
    return f2h(x - math.copysign(float(math.floor(x)), x))


def merge_half(old: int, half: int, dst_hi: int) -> int:
    h = half & 0xFFFF
    return ((h << 16) | (old & 0xFFFF)) if dst_hi else ((old & 0xFFFF0000) | h)


def is_f16_nan(half: int) -> bool:
    return (half & 0x7FFF) > 0x7C00


OPS = {
    "v_floor_f16": ("Floor", ref_floor),
    "v_ceil_f16": ("Ceil", ref_ceil),
    "v_trunc_f16": ("Trunc", ref_trunc),
    "v_rndne_f16": ("RoundEven", ref_rndne),
}

_TMP_DIR = None
_PS_PATH = None


def setUpModule():
    global _TMP_DIR, _PS_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _PS_PATH = os.path.join(disasm_dir, "ps_f16_round.s")
    with open(os.path.join(_TMP_DIR.name, "ps_f16_round.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_valu_f16_round\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_PS_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_PS_PATH)


def run_ps(lines: list[str]) -> int:
    return si.run_fragment(translate_ps(lines))["%out_mrt0"][0]


def run_op(line: str, v1: int, old: int = OLD, s4: int = 0) -> int:
    lines = [f"s_mov_b32 s4, {s4:#x}", f"v_mov_b32 v1, {v1:#x}", f"v_mov_b32 v0, {old:#x}", line]
    return run_ps(lines)


class ReferenceOracleTests(unittest.TestCase):
    def test_integral_and_fract_edge_values(self):
        self.assertEqual(ref_floor(H_NEG_0P25), H_NEG_1P0)
        self.assertEqual(ref_floor(H_NEG_ZERO), H_NEG_ZERO)
        self.assertEqual(ref_ceil(H_NEG_0P25), H_NEG_ZERO)
        self.assertEqual(ref_ceil(H_0P25), H_1P0)
        self.assertEqual(ref_trunc(H_NEG_0P75), H_NEG_ZERO)
        self.assertEqual(ref_trunc(H_1P75), H_1P0)
        self.assertEqual(ref_rndne(H_0P5), H_POS_ZERO)
        self.assertEqual(ref_rndne(H_NEG_0P5), H_NEG_ZERO)
        self.assertEqual(ref_rndne(H_1P5), H_2P0)
        self.assertEqual(ref_rndne(H_2P5), H_2P0)
        self.assertEqual(ref_rndne(H_1022P5), f2h(1022.0))
        self.assertEqual(ref_rndne(H_1023P5), H_1024)
        self.assertEqual(ref_fract(H_1P25), H_0P25)
        self.assertEqual(ref_fract(H_NEG_0P25), H_0P75)
        self.assertEqual(ref_fract(H_NEG_1P75), H_0P25)
        self.assertEqual(ref_fract(H_NEG_ZERO), H_POS_ZERO)
        self.assertTrue(is_f16_nan(ref_fract(H_POS_INF)))
        self.assertEqual(merge_half(OLD, H_1P0, 0), 0xBEEF3C00)
        self.assertEqual(merge_half(OLD, H_1P0, 1), 0x3C001234)


class F16RoundingExecutionTests(unittest.TestCase):
    def test_edge_values_across_e32_e64_and_bare(self):
        samples = [
            H_POS_ZERO, H_NEG_ZERO,
            H_SUB_MIN, H_NEG_SUB_MIN, H_SUB_MAX, H_NEG_SUB_MAX,
            H_0P25, H_NEG_0P25, H_0P5, H_NEG_0P5, H_0P75, H_NEG_0P75,
            H_1P0, H_NEG_1P0, H_1P25, H_NEG_1P25, H_1P5, H_NEG_1P5, H_1P75, H_NEG_1P75,
            H_2P0, H_NEG_2P0, H_2P5, H_NEG_2P5, H_3P0, H_NEG_3P0, H_3P5, H_NEG_3P5,
            H_1022P5, H_1023P5, H_NEG_1022P5, H_NEG_1023P5,
            H_MAX, H_NEG_MAX, H_POS_INF, H_NEG_INF,
        ]
        for op, (_, ref) in OPS.items():
            for suffix in ("_e32", "_e64", ""):
                for h in samples:
                    v1 = 0x55550000 | h
                    with self.subTest(op=op + suffix, h=hex(h)):
                        got = run_op(f"{op}{suffix} v0, v1", v1)
                        self.assertEqual(got, merge_half(OLD, ref(h), 0))

    def test_signed_zero_and_subnormal_signs_preserved(self):
        # IEEE-754 roundToIntegral keeps -0.0 when rounding values in (-1.0, -0.0] to zero.
        self.assertEqual(run_op("v_floor_f16_e32 v0, v1", 0xABCD0000 | H_NEG_ZERO), merge_half(OLD, H_NEG_ZERO, 0))
        self.assertEqual(run_op("v_floor_f16_e32 v0, v1", 0xABCD0000 | H_POS_ZERO), merge_half(OLD, H_POS_ZERO, 0))
        self.assertEqual(run_op("v_floor_f16_e32 v0, v1", 0xABCD0000 | H_SUB_MIN), merge_half(OLD, H_POS_ZERO, 0))
        self.assertEqual(run_op("v_floor_f16_e32 v0, v1", 0xABCD0000 | H_NEG_SUB_MIN), merge_half(OLD, H_NEG_1P0, 0))
        for op in ("v_ceil_f16", "v_trunc_f16", "v_rndne_f16"):
            for h in (H_NEG_ZERO, H_NEG_SUB_MIN, H_NEG_SUB_MAX, H_NEG_0P25):
                with self.subTest(op=op, h=hex(h)):
                    self.assertEqual(run_op(f"{op}_e32 v0, v1", 0xABCD0000 | h), merge_half(OLD, H_NEG_ZERO, 0))

    def test_nan_preserves_nan_and_other_half(self):
        for op in OPS:
            for h in (H_QNAN, H_NEG_QNAN, H_SNAN):
                with self.subTest(op=op, h=hex(h)):
                    got = run_op(f"{op}_e32 v0, v1", 0xCAFE0000 | h)
                    self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)
                    self.assertTrue(is_f16_nan(got & 0xFFFF), f"expected NaN in low half, got {got:#x}")

    def test_op_sel_all_combinations(self):
        # v1 = (hi: -2.5, lo: 3.5) -> clear, distinct results across every op and half.
        v1 = (H_NEG_2P5 << 16) | H_3P5
        for op, (_, ref) in OPS.items():
            for src_hi, dst_hi in itertools.product((0, 1), repeat=2):
                h_in = H_NEG_2P5 if src_hi else H_3P5
                line = f"{op}_e64 v0, v1 op_sel:[{src_hi},{dst_hi}]"
                with self.subTest(line=line):
                    self.assertEqual(run_op(line, v1), merge_half(OLD, ref(h_in), dst_hi))

    def test_vop3_source_and_output_modifiers(self):
        v1 = (H_NEG_1P5 << 16) | H_2P5
        # Source modifiers apply to the selected half before the rounding op.
        self.assertEqual(run_op("v_floor_f16_e64 v0, -v1 op_sel:[0,0]", v1), merge_half(OLD, f2h(-3.0), 0))
        self.assertEqual(run_op("v_ceil_f16_e64 v0, |v1| op_sel:[1,1]", v1), merge_half(OLD, H_2P0, 1))
        self.assertEqual(run_op("v_rndne_f16_e64 v0, -|v1| op_sel:[1,0]", v1), merge_half(OLD, H_NEG_2P0, 0))

        # Output modifiers scale first, then clamp to [0.0, 1.0], then pack to f16.
        # trunc(2.5) = 2.0 -> mul:2 = 4.0; trunc(2.5) * 0.5 = 1.0; clamp -> 1.0.
        self.assertEqual(run_op("v_trunc_f16_e64 v0, v1 mul:2", v1), merge_half(OLD, H_4P0, 0))
        self.assertEqual(run_op("v_trunc_f16_e64 v0, v1 mul:4", v1), merge_half(OLD, f2h(8.0), 0))
        self.assertEqual(run_op("v_trunc_f16_e64 v0, v1 div:2", v1), merge_half(OLD, H_1P0, 0))
        self.assertEqual(run_op("v_trunc_f16_e64 v0, v1 mul:2 clamp", v1), merge_half(OLD, H_1P0, 0))
        # floor(-1.5) = -2.0 -> clamp -> 0.0 (preserving low half when dst_hi=1).
        self.assertEqual(run_op("v_floor_f16_e64 v0, v1 op_sel:[1,1] clamp", v1), merge_half(OLD, H_POS_ZERO, 1))
        # rndne(0.5) = 0.0 -> div:2 = 0.0; trunc(1.25) = 1.0 -> div:2 = 0.5.
        self.assertEqual(run_op("v_trunc_f16_e64 v0, v1 div:2", H_1P25), merge_half(OLD, H_0P5, 0))

    def test_inline_constants_sgpr_and_aliasing(self):
        # Inline float constants populate the low half as f16 (docs/design.md).
        self.assertEqual(run_op("v_ceil_f16_e32 v0, 0.5", 0), merge_half(OLD, H_1P0, 0))
        self.assertEqual(run_op("v_floor_f16_e32 v0, -0.5", 0), merge_half(OLD, H_NEG_1P0, 0))
        self.assertEqual(run_op("v_rndne_f16_e64 v0, -1.0", 0), merge_half(OLD, H_NEG_1P0, 0))

        # SGPR source (reading high half via op_sel).
        s4 = (H_NEG_1P25 << 16) | H_0P75
        self.assertEqual(run_op("v_floor_f16_e32 v0, s4", 0, s4=s4), merge_half(OLD, H_POS_ZERO, 0))
        self.assertEqual(run_op("v_floor_f16_e64 v0, s4 op_sel:[1,1]", 0, s4=s4), merge_half(OLD, H_NEG_2P0, 1))

        # In-place aliasing: dst == src reading high half and writing low half keeps the original high half.
        v0_init = (H_3P5 << 16) | H_0P25
        self.assertEqual(run_ps([f"v_mov_b32 v0, {v0_init:#x}", "v_rndne_f16_e64 v0, v0 op_sel:[1,0]"]),
                         (H_3P5 << 16) | H_4P0)
        self.assertEqual(run_ps([f"v_mov_b32 v0, {v0_init:#x}", "v_trunc_f16_e32 v0, v0"]),
                         (H_3P5 << 16) | H_POS_ZERO)

    def test_exec_masking(self):
        # Inactive single lane preserves pre-existing v0.
        got = run_ps([
            f"v_mov_b32 v1, {H_1P5:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "s_mov_b32 exec_lo, 0",
            "v_ceil_f16_e32 v0, v1",
            "s_mov_b32 exec_lo, 1",
        ])
        self.assertEqual(got, OLD)

        # Disabled ops leave both halves untouched; re-enabling EXEC runs the subsequent op.
        got_toggled = run_ps([
            f"v_mov_b32 v1, {(H_2P5 << 16) | H_NEG_1P25:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "s_mov_b32 exec_lo, 0",
            "v_floor_f16_e64 v0, v1 op_sel:[0,0]",
            "v_ceil_f16_e64 v0, v1 op_sel:[1,1]",
            "s_mov_b32 exec_lo, 1",
            "v_trunc_f16_e64 v0, v1 op_sel:[1,0]",
        ])
        self.assertEqual(got_toggled, merge_half(OLD, H_2P0, 0))

    def test_negative_control_sabotaged_spirv(self):
        spv = translate_ps([f"v_mov_b32 v1, {H_2P5:#x}", f"v_mov_b32 v0, {OLD:#x}", "v_floor_f16_e32 v0, v1"])
        sabotaged, n = re.subn(r"\bFloor\b", "Ceil", spv, count=1)
        self.assertEqual(n, 1)
        got = si.run_fragment(sabotaged)["%out_mrt0"][0]
        self.assertNotEqual(got, merge_half(OLD, H_2P0, 0))
        self.assertEqual(got, merge_half(OLD, H_3P0, 0))


class FractF16TranslationTests(unittest.TestCase):
    """v_fract_f16 lowers to GLSL.std.450 Fract (spirv_mini_interp.py computes it exactly, rounded once)."""

    def test_fract_f16_emits_glsl_fract_op_sel_mods_and_half_keep(self):
        spv = translate_ps([
            f"v_mov_b32 v1, {(H_NEG_1P25 << 16) | H_2P5:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "v_fract_f16_e64 v0, v1 op_sel:[1,1] clamp",
        ])
        self.assertIn("UnpackHalf2x16", spv)
        self.assertRegex(spv, r"OpExtInst %float %glsl Fract %\w+")
        self.assertRegex(spv, r"OpExtInst %float %glsl FClamp %\w+ %f0 %f1")
        # op_sel:[1,1]: fract of the high half (-1.25 -> 0.75 = 0x3a00) into the high half; the low half is kept.
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], (0x3A00 << 16) | (OLD & 0xFFFF))


if __name__ == "__main__":
    unittest.main()
