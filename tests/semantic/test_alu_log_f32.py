from __future__ import annotations

"""Semantic interpreter tests for RDNA2 single-precision base-2 logarithm (v_log_f32).

RDNA2 VALU calculates D.f32 = log2(S0.f32). In tools/rdna2spirv.py, v_log_f32 is translated to GLSL.std.450
extended instruction Log2 (OpExtInst %float %glsl Log2 %src), followed by optional VOP3 output scaling
(mul:2, mul:4, div:2) and FClamp [0.0, 1.0].

This suite tests:
1. Exact reference oracle against IEEE-754 binary32 specifications, power-of-2 exact values, and standard library.
2. VOP1 (_e32) and VOP3 (_e64) execution forms, including literal constants, SGPR sources, and inline constants.
3. VOP3 output modifiers (clamp, mul:2, mul:4, div:2) and source absolute-value modifier (|S0|).
"""

import gc
import math
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
PINF = 0x7F800000
NINF = 0xFF800000
QNAN = 0x7FC00000
EXP_MASK = 0x7F800000
MANT_MASK = 0x007FFFFF

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD_VAL = 0xAAAA5555

POW2_TABLE: list[tuple[float, float]] = [
    (0.0625, -4.0),
    (0.125, -3.0),
    (0.25, -2.0),
    (0.5, -1.0),
    (1.0, 0.0),
    (2.0, 1.0),
    (4.0, 2.0),
    (8.0, 3.0),
    (16.0, 4.0),
    (32.0, 5.0),
    (64.0, 6.0),
    (128.0, 7.0),
    (256.0, 8.0),
    (512.0, 9.0),
    (1024.0, 10.0),
    (2048.0, 11.0),
    (4096.0, 12.0),
    (8192.0, 13.0),
    (16384.0, 14.0),
    (32768.0, 15.0),
    (65536.0, 16.0),
]


def f32_to_bits(val: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return PINF if val > 0 else NINF


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_f32_nan(bits: int) -> bool:
    return (bits & EXP_MASK) == EXP_MASK and (bits & MANT_MASK) != 0


def is_f32_pos_inf(bits: int) -> bool:
    return (bits & MASK32) == PINF


def is_f32_neg_inf(bits: int) -> bool:
    return (bits & MASK32) == NINF


def ref_log2_f32(
    bits: int,
    *,
    scale: float = 1.0,
    clamp: bool = False,
    src_abs: bool = False,
    src_neg: bool = False,
    hardware_model: bool = False,
) -> int:
    """Reference oracle for single-precision base-2 logarithm v_log_f32.

    If hardware_model is True, follows RDNA2 ISA measured behavior:
      v_log_f32(+0.0 / -0.0) = -inf
      v_log_f32(negative) = NaN (0xFFC00000)
    If hardware_model is False (default, SPIR-V/GLSL.std.450 model):
      x <= 0 raises ValueError (undefined in GLSL.std.450 Log2).
    """
    u = bits & MASK32
    if src_abs:
        u &= 0x7FFFFFFF
    if src_neg:
        u ^= 0x80000000

    if is_f32_nan(u):
        if clamp:
            raise ValueError("clamp of NaN is undefined in GLSL.std.450")
        return u

    sign = (u >> 31) & 1
    exp = (u >> 23) & 0xFF
    mant = u & MANT_MASK

    if (u & 0x7FFFFFFF) == 0:
        if hardware_model:
            return NINF
        raise ValueError("Log2 of zero is undefined in GLSL.std.450")

    if sign == 1:
        if hardware_model:
            return 0xFFC00000
        raise ValueError("Log2 of negative is undefined in GLSL.std.450")

    if exp == 0xFF and mant == 0:
        if clamp:
            return f32_to_bits(1.0)
        return PINF

    val = bits_to_f32(u)
    if mant == 0 and exp > 0:
        exact_exp = exp - 127
        log_val = float(exact_exp)
    else:
        log_val = math.log2(val)

    if scale != 1.0:
        log_val *= scale

    if clamp:
        log_val = min(max(log_val, 0.0), 1.0)

    return f32_to_bits(log_val)


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_PS_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _PS_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _PS_PATH = os.path.join(disasm_dir, "ps_log_f32.s")
    with open(os.path.join(_TMP_DIR.name, "ps_log_f32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _PS_PATH
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _PS_PATH = None


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    assert _PS_PATH is not None
    asm = (
        "; test_log_f32\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v0, v0, v0 done vm\n"
        + "\ts_endpgm\n"
    )
    with open(_PS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_PS_PATH)
        gc.collect()
    return spv


def run_ps_v0(lines: list[str]) -> int:
    return si.run_fragment(translate_ps(lines))["%out_mrt0"][0]


def run_op(line: str, v1: int, old: int = OLD_VAL, s4: int = 0) -> int:
    return run_ps_v0([
        f"s_mov_b32 s4, {s4:#x}",
        f"v_mov_b32 v1, {v1:#x}",
        f"v_mov_b32 v0, {old:#x}",
        line,
    ])


class OracleSelfTests(unittest.TestCase):
    """Verifies that the reference oracle accurately computes base-2 logarithm."""

    def test_exact_powers_of_two(self) -> None:
        for x, want in POW2_TABLE:
            with self.subTest(x=x, want=want):
                got_bits = ref_log2_f32(f32_to_bits(x))
                self.assertEqual(got_bits, f32_to_bits(want))

    def test_extreme_powers_of_two(self) -> None:
        self.assertEqual(ref_log2_f32(0x7F000000), f32_to_bits(127.0))
        self.assertEqual(ref_log2_f32(0x00800000), f32_to_bits(-126.0))

    def test_positive_infinity(self) -> None:
        self.assertEqual(ref_log2_f32(PINF), PINF)

    def test_output_scale_modifiers(self) -> None:
        self.assertEqual(ref_log2_f32(f32_to_bits(4.0), scale=2.0), f32_to_bits(4.0))
        self.assertEqual(ref_log2_f32(f32_to_bits(2.0), scale=4.0), f32_to_bits(4.0))
        self.assertEqual(ref_log2_f32(f32_to_bits(16.0), scale=0.5), f32_to_bits(2.0))

    def test_clamp_modifier(self) -> None:
        self.assertEqual(ref_log2_f32(f32_to_bits(0.5), clamp=True), f32_to_bits(0.0))
        self.assertEqual(ref_log2_f32(f32_to_bits(4.0), clamp=True), f32_to_bits(1.0))
        self.assertEqual(ref_log2_f32(f32_to_bits(2.0), clamp=True), f32_to_bits(1.0))

    def test_source_absolute_modifier(self) -> None:
        self.assertEqual(ref_log2_f32(f32_to_bits(-4.0), src_abs=True), f32_to_bits(2.0))
        self.assertEqual(ref_log2_f32(f32_to_bits(-0.5), src_abs=True), f32_to_bits(-1.0))


class LogF32ExecutionTests(unittest.TestCase):
    """End-to-end execution of v_log_f32 across various operand forms and modifiers."""

    def test_powers_of_two_e32_and_e64(self) -> None:
        for x, want in POW2_TABLE:
            x_bits = f32_to_bits(x)
            want_bits = f32_to_bits(want)
            for form in ("v_log_f32_e32 v0, v1", "v_log_f32_e64 v0, v1", "v_log_f32 v0, v1"):
                with self.subTest(x=x, form=form):
                    got = run_op(form, x_bits)
                    self.assertEqual(got, want_bits)

    def test_sgpr_scalar_source(self) -> None:
        for x, want in ((1.0, 0.0), (2.0, 1.0), (4.0, 2.0), (8.0, 3.0), (0.5, -1.0)):
            x_bits = f32_to_bits(x)
            want_bits = f32_to_bits(want)
            with self.subTest(x=x, src="s4"):
                got = run_op("v_log_f32_e32 v0, s4", 0, s4=x_bits)
                self.assertEqual(got, want_bits)

    def test_inline_constants(self) -> None:
        cases = [
            ("v_log_f32_e32 v0, 1.0", f32_to_bits(0.0)),
            ("v_log_f32_e32 v0, 2.0", f32_to_bits(1.0)),
            ("v_log_f32_e32 v0, 4.0", f32_to_bits(2.0)),
            ("v_log_f32_e32 v0, 0.5", f32_to_bits(-1.0)),
        ]
        for line, want_bits in cases:
            with self.subTest(line=line):
                got = run_ps_v0([f"v_mov_b32 v0, {OLD_VAL:#x}", line])
                self.assertEqual(got, want_bits)

    def test_positive_infinity(self) -> None:
        got = run_op("v_log_f32_e32 v0, v1", PINF)
        self.assertEqual(got, PINF)

    def test_vop3_output_modifiers(self) -> None:
        cases = [
            ("v_log_f32_e64 v0, v1 mul:2", 4.0, 4.0),
            ("v_log_f32_e64 v0, v1 mul:4", 2.0, 4.0),
            ("v_log_f32_e64 v0, v1 div:2", 16.0, 2.0),
            ("v_log_f32_e64 v0, v1 clamp", 0.5, 0.0),
            ("v_log_f32_e64 v0, v1 clamp", 4.0, 1.0),
            ("v_log_f32_e64 v0, v1 mul:2 clamp", 2.0, 1.0),
            ("v_log_f32_e64 v0, v1 div:2 clamp", 4.0, 1.0),
        ]
        for line, in_val, want_val in cases:
            with self.subTest(line=line, in_val=in_val):
                got = run_op(line, f32_to_bits(in_val))
                self.assertEqual(got, f32_to_bits(want_val))

    def test_vop3_source_absolute_modifier(self) -> None:
        for x in (1.0, 2.0, 4.0, 8.0, 0.5):
            want = ref_log2_f32(f32_to_bits(x))
            with self.subTest(x=x, sign="positive"):
                got = run_op("v_log_f32_e64 v0, |v1|", f32_to_bits(x))
                self.assertEqual(got, want)
            with self.subTest(x=x, sign="negative"):
                got = run_op("v_log_f32_e64 v0, |v1|", f32_to_bits(-x))
                self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
