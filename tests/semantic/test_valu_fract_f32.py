from __future__ import annotations

"""End-to-end semantic value tests for RDNA2 v_fract_f32 (S0.f32 - floor(S0.f32)).

Covers v_fract_f32 executed end-to-end through tools/rdna2spirv.py and
tests/semantic/spirv_mini_interp.py against an independent Fraction-based
IEEE-754 binary32 `x - floor(x)` reference oracle:
  - Formula: D = S0 - floor(S0), returning values in [0.0, 1.0).
  - Non-negative property: for any finite input x, fract(x) >= 0.0 and fract(x) < 1.0.
  - Negative number fractional part: fract(-0.25) == 0.75, fract(-1.75) == 0.25,
    fract(-0.5) == 0.5, fract(-0.75) == 0.25.
  - Exact integers: fract(N.0) == +0.0 (0.0, 1.0, -1.0, 2.0, -2.0, 42.0, etc.).
  - Large numbers with no fractional bits in mantissa (|x| >= 2^23 = 8388608.0 for f32)
    yield +0.0, while 2^23 - 0.5 (8388607.5) preserves fractional part 0.5.
  - Special IEEE-754 values: +0.0 -> +0.0, -0.0 -> +0.0, +-inf -> NaN, NaN -> NaN.
  - Hardware/translator rounding boundary: when x - floor(x) rounds to 1.0 in
    binary32 (e.g. tiny negatives), the result is clamped to 0x3F7FFFFF (the largest
    float strictly below 1.0, 1 - 2^-24).
  - Forms: _e32, _e64, and bare mnemonics.
  - VOP3 source modifiers (-v1, |v1|, -|v1|) and output modifiers
    (clamp, mul:2, mul:4, div:2, mul:2 clamp, mul:4 clamp).
  - Inline constants (0.5, -0.5, 1.0, -1.0), SGPR sources (s4), destination
    aliasing (v0, v0), and runtime memory inputs via s_buffer_load_dwordx4.

Zero game data; hand-written RDNA2 assembly only; standard library Python only.
"""

from fractions import Fraction
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xBEEF1234
PUSH_ADDR = [0x1000, 0, 32, 0] + [0] * 28
VSHARP_CS = [0, 0, 0xFFFFFFFF, 0x31016FAC]
HEAP_FILL = 0xEEEEEEEE

# Representative IEEE-754 binary32 bit encodings.
F_POS_ZERO = 0x00000000      # +0.0
F_NEG_ZERO = 0x80000000      # -0.0
F_0P125 = 0x3E000000         # 0.125
F_0P25 = 0x3E800000          # 0.25
F_NEG_0P25 = 0xBE800000      # -0.25
F_0P375 = 0x3EC00000         # 0.375
F_0P5 = 0x3F000000           # 0.5
F_NEG_0P5 = 0xBF000000       # -0.5
F_0P75 = 0x3F400000          # 0.75
F_NEG_0P75 = 0xBF400000      # -0.75
F_NEAR_ONE = 0x3F7FFFFF      # 1.0 - 2^-24 (largest f32 < 1.0)
F_1P0 = 0x3F800000           # 1.0
F_NEG_1P0 = 0xBF800000       # -1.0
F_1P25 = 0x3FA00000          # 1.25
F_NEG_1P25 = 0xBFA00000      # -1.25
F_1P5 = 0x3FC00000           # 1.5
F_NEG_1P5 = 0xBFC00000       # -1.5
F_1P75 = 0x3FE00000          # 1.75
F_NEG_1P75 = 0xBFE00000      # -1.75
F_2P0 = 0x40000000           # 2.0
F_NEG_2P0 = 0xC0000000       # -2.0
F_2P5 = 0x40200000           # 2.5
F_NEG_2P5 = 0xC0200000       # -2.5
F_3P0 = 0x40400000           # 3.0
F_NEG_3P0 = 0xC0400000       # -3.0
F_3P75 = 0x40700000          # 3.75
F_NEG_3P75 = 0xC0700000      # -3.75
F_42P0 = 0x42280000          # 42.0
F_NEG_42P0 = 0xC2280000      # -42.0
F_2P23_M0P5 = 0x4AFFFFFF     # 2^23 - 0.5 = 8388607.5
F_2P23 = 0x4B000000          # 2^23 = 8388608.0 (mantissa bits exhausted)
F_NEG_2P23 = 0xCB000000      # -2^23 = -8388608.0
F_2P24 = 0x4B800000          # 2^24 = 16777216.0
F_NEG_2P24 = 0xCB800000      # -2^24 = -16777216.0
F_POS_INF = 0x7F800000       # +inf
F_NEG_INF = 0xFF800000       # -inf
F_QNAN = 0x7FC00000          # +qNaN
F_NEG_QNAN = 0xFFC00000      # -qNaN
F_SNAN = 0x7F800001          # +sNaN
F_NEG_SNAN = 0xFF800001      # -sNaN


# -----------------------------------------------------------------------------
# Independent Fraction-Based IEEE-754 Binary32 Oracle
# -----------------------------------------------------------------------------

def f32_is_finite(w: int) -> bool:
    return ((w >> 23) & 0xFF) != 0xFF


def f32_is_nan(w: int) -> bool:
    return ((w >> 23) & 0xFF) == 0xFF and (w & 0x7FFFFF) != 0


def f32_is_inf(w: int) -> bool:
    return (w & 0x7FFFFFFF) == 0x7F800000


def f32_to_fraction(w: int) -> Fraction:
    """Decode finite IEEE-754 binary32 bit pattern ``w`` into an exact Fraction."""
    w &= 0xFFFFFFFF
    exp = (w >> 23) & 0xFF
    if exp == 0xFF:
        raise ValueError(f"non-finite binary32 {w:#010x}")
    sign = -1 if (w >> 31) else 1
    mant = w & 0x7FFFFF
    if exp == 0:
        return Fraction(0) if mant == 0 else Fraction(sign * mant, 1 << 149)
    return Fraction(sign * ((1 << 23) | mant)) * (Fraction(2) ** (exp - 150))


def fraction_to_f32(q: Fraction) -> int:
    """Round an exact Fraction to IEEE-754 binary32 (round-to-nearest, ties-to-even)."""
    if q == 0:
        return F_POS_ZERO
    sgn = 0x80000000 if q < 0 else 0
    a = abs(q)
    e = max(a.numerator.bit_length() - a.denominator.bit_length(), -126)
    if e > -126 and Fraction(2) ** e > a:
        e -= 1
    m = a / (Fraction(2) ** (e - 23))
    n = m.numerator // m.denominator
    rem = m - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
        n += 1
    if e == -126 and n < (1 << 23):
        return sgn | n
    if n >= (1 << 24):
        n >>= 1
        e += 1
    if e > 127:
        return sgn | F_POS_INF
    return sgn | ((e + 127) << 23) | (n - (1 << 23))


def exact_fract_f32_fraction(w: int) -> Fraction | None:
    """Exact ``x - floor(x)`` as a Fraction in [0, 1), or None for +-inf / NaN."""
    if not f32_is_finite(w):
        return None
    q = f32_to_fraction(w)
    return q - math.floor(q)


def oracle_fract_f32(
    w: int,
    *,
    abs_: bool = False,
    neg: bool = False,
    mul: int = 1,
    div: int = 1,
    clamp: bool = False,
) -> int | None:
    """Independent binary32 reference for v_fract_f32, returning 32-bit int or None for NaN.

    Models RDNA2 source modifiers (abs before neg), exact ``x - floor(x)``,
    the hardware/translator 0x3f800000 -> 0x3f7fffff clamp when no output multiplier is
    active, and VOP3 output scaling (mul/div) + [0, 1] clamp.
    """
    w &= 0xFFFFFFFF
    if abs_:
        w &= 0x7FFFFFFF
    if neg:
        w ^= 0x80000000
    r = exact_fract_f32_fraction(w)
    if r is None:
        return None
    if mul == 1 and div == 1:
        res = fraction_to_f32(r)
        return F_NEAR_ONE if res == F_1P0 else res
    scaled = r * Fraction(mul, div)
    if clamp:
        scaled = min(max(scaled, Fraction(0)), Fraction(1))
    return fraction_to_f32(scaled)


# -----------------------------------------------------------------------------
# Translation & Execution Helpers
# -----------------------------------------------------------------------------

_TMP_DIR = None
_PS_PATH = None
_CS_PATH = None
_CS_FSSM = None


def _make_fssm(rsrc2: int, group: tuple[int, int, int]) -> bytes:
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


def setUpModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH, _CS_FSSM
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    fssm_dir = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(fssm_dir)
    _PS_PATH = os.path.join(disasm_dir, "ps_fract_f32.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_fract_f32.s")
    _CS_FSSM = os.path.join(fssm_dir, "cs_fract_f32.fssm")
    with open(os.path.join(_TMP_DIR.name, "ps_fract_f32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH, _CS_FSSM
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _PS_PATH = None
        _CS_PATH = None
        _CS_FSSM = None


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_valu_fract_f32\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    )
    with open(_PS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_PS_PATH)
        gc.collect()
    return spv


def run_ps(lines: list[str], *, push_constants=None, memory=None) -> int:
    out = si.run_fragment(translate_ps(lines), push_constants=push_constants, memory=memory)
    return out["%out_mrt0"][0]


def run_op(line: str, v1: int = 0, *, old: int = OLD, s4: int = 0) -> int:
    return run_ps([
        f"s_mov_b32 s4, {s4:#x}",
        f"v_mov_b32 v1, {v1:#x}",
        f"v_mov_b32 v0, {old:#x}",
        line,
    ])


def run_cs(lines: list[str], group_size: int, memory: dict[int, int]) -> None:
    if _TMP_DIR is None:
        setUpModule()
    # rsrc2 = 0x0088: USER_SGPR = 4 (s[0:3] V#), TGID_X_EN = 1 (s4), TIDIG_COMP_CNT = 0 (v0 = local_id.x).
    with open(_CS_FSSM, "wb") as f:
        f.write(_make_fssm(0x0088, (group_size, 1, 1)))
    asm = "; test_cs_valu_fract_f32\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_CS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_CS_PATH)
        gc.collect()
    si.run_workgroup(
        spv,
        group_size,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
        push_constants=VSHARP_CS + [0] * 28,
        memory=memory,
    )


class _Heap(dict):
    def get(self, key, default=None):
        return super().get(key, HEAP_FILL)


# -----------------------------------------------------------------------------
# Oracle Unit Tests
# -----------------------------------------------------------------------------

class OracleF32Tests(unittest.TestCase):
    """Verifies the independent Fraction-based binary32 oracle against IEEE-754 semantics."""

    def test_fraction_decode_and_roundtrip(self):
        samples = [
            F_POS_ZERO, F_NEG_ZERO,
            F_0P125, F_0P25, F_NEG_0P25, F_0P375,
            F_0P5, F_NEG_0P5, F_0P75, F_NEG_0P75,
            F_NEAR_ONE, F_1P0, F_NEG_1P0,
            F_1P25, F_NEG_1P25, F_1P5, F_NEG_1P5, F_1P75, F_NEG_1P75,
            F_2P0, F_NEG_2P0, F_2P5, F_NEG_2P5, F_3P0, F_NEG_3P0,
            F_3P75, F_NEG_3P75, F_42P0, F_NEG_42P0,
            F_2P23_M0P5, F_2P23, F_NEG_2P23, F_2P24, F_NEG_2P24,
        ]
        for w in samples:
            with self.subTest(w=hex(w)):
                self.assertTrue(f32_is_finite(w))
                q = f32_to_fraction(w)
                expected_float = struct.unpack("<f", struct.pack("<I", w))[0]
                self.assertEqual(q, Fraction(expected_float))
                want = F_POS_ZERO if w == F_NEG_ZERO else w
                self.assertEqual(fraction_to_f32(q), want)

    def test_hand_worked_fract_f32_rows(self):
        rows = [
            (F_POS_ZERO, F_POS_ZERO),
            (F_NEG_ZERO, F_POS_ZERO),
            (F_0P25, F_0P25),
            (F_NEG_0P25, F_0P75),
            (F_0P5, F_0P5),
            (F_NEG_0P5, F_0P5),
            (F_0P75, F_0P75),
            (F_NEG_0P75, F_0P25),
            (F_1P0, F_POS_ZERO),
            (F_NEG_1P0, F_POS_ZERO),
            (F_1P25, F_0P25),
            (F_NEG_1P25, F_0P75),
            (F_1P75, F_0P75),
            (F_NEG_1P75, F_0P25),
            (F_2P0, F_POS_ZERO),
            (F_NEG_2P0, F_POS_ZERO),
            (F_2P5, F_0P5),
            (F_NEG_2P5, F_0P5),
            (F_3P75, F_0P75),
            (F_NEG_3P75, F_0P25),
            (F_42P0, F_POS_ZERO),
            (F_NEG_42P0, F_POS_ZERO),
            (F_2P23_M0P5, F_0P5),
            (F_2P23, F_POS_ZERO),
            (F_NEG_2P23, F_POS_ZERO),
            (F_2P24, F_POS_ZERO),
            (F_NEG_2P24, F_POS_ZERO),
            (F_NEAR_ONE, F_NEAR_ONE),
        ]
        for w, want in rows:
            with self.subTest(w=hex(w)):
                self.assertEqual(oracle_fract_f32(w), want)

    def test_non_negative_property(self):
        # fract(x) >= 0.0 and fract(x) < 1.0 for all finite numbers
        test_inputs = [
            -100.5, -42.25, -10.75, -2.5, -1.75, -1.25, -0.75, -0.5, -0.25, -0.0,
            0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 42.125, 100.875,
        ]
        for val in test_inputs:
            w = struct.unpack("<I", struct.pack("<f", val))[0]
            res_w = oracle_fract_f32(w)
            res_val = struct.unpack("<f", struct.pack("<I", res_w))[0]
            with self.subTest(val=val):
                self.assertGreaterEqual(res_val, 0.0)
                self.assertLess(res_val, 1.0)

    def test_large_numbers_mantissa_exhaustion(self):
        # |x| >= 2^23 has no fractional bits in IEEE binary32 mantissa -> fract(x) == 0.0
        for exp in range(23, 31):
            val_pos = float(1 << exp)
            val_neg = -val_pos
            w_pos = struct.unpack("<I", struct.pack("<f", val_pos))[0]
            w_neg = struct.unpack("<I", struct.pack("<f", val_neg))[0]
            with self.subTest(exp=exp):
                self.assertEqual(oracle_fract_f32(w_pos), F_POS_ZERO)
                self.assertEqual(oracle_fract_f32(w_neg), F_POS_ZERO)

    def test_special_ieee_oracle(self):
        self.assertEqual(oracle_fract_f32(F_POS_ZERO), F_POS_ZERO)
        self.assertEqual(oracle_fract_f32(F_NEG_ZERO), F_POS_ZERO)
        self.assertIsNone(oracle_fract_f32(F_POS_INF))
        self.assertIsNone(oracle_fract_f32(F_NEG_INF))
        self.assertIsNone(oracle_fract_f32(F_QNAN))
        self.assertIsNone(oracle_fract_f32(F_NEG_QNAN))
        self.assertIsNone(oracle_fract_f32(F_SNAN))
        self.assertIsNone(oracle_fract_f32(F_NEG_SNAN))


# -----------------------------------------------------------------------------
# End-to-End Execution Tests
# -----------------------------------------------------------------------------

class FractF32ExecutionTests(unittest.TestCase):
    """End-to-end execution of RDNA2 v_fract_f32 through rdna2spirv and spirv_mini_interp."""

    def test_finite_values_across_e32_e64_and_bare(self):
        samples = [
            F_POS_ZERO, F_NEG_ZERO,
            F_0P125, F_0P25, F_NEG_0P25, F_0P375,
            F_0P5, F_NEG_0P5, F_0P75, F_NEG_0P75,
            F_NEAR_ONE, F_1P0, F_NEG_1P0,
            F_1P25, F_NEG_1P25, F_1P5, F_NEG_1P5, F_1P75, F_NEG_1P75,
            F_2P0, F_NEG_2P0, F_2P5, F_NEG_2P5, F_3P0, F_NEG_3P0,
            F_3P75, F_NEG_3P75, F_42P0, F_NEG_42P0,
            F_2P23_M0P5, F_2P23, F_NEG_2P23, F_2P24, F_NEG_2P24,
        ]
        for suffix in ("_e32", "_e64", ""):
            for w in samples:
                want = oracle_fract_f32(w)
                with self.subTest(form="v_fract_f32" + suffix, w=hex(w)):
                    self.assertEqual(run_op(f"v_fract_f32{suffix} v0, v1", w), want)

    def test_negative_number_fractional_part(self):
        # Specific assertion: fract(-0.25) == 0.75, fract(-1.75) == 0.25
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_0P25), F_0P75)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_1P75), F_0P25)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_0P5), F_0P5)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_0P75), F_0P25)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_1P25), F_0P75)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_2P5), F_0P5)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_3P75), F_0P25)

    def test_exact_integers_and_signed_zero(self):
        # Integers and +-0.0 must yield +0.0 (sign bit must be 0)
        cases = [
            F_POS_ZERO, F_NEG_ZERO,
            F_1P0, F_NEG_1P0,
            F_2P0, F_NEG_2P0,
            F_3P0, F_NEG_3P0,
            F_42P0, F_NEG_42P0,
            F_2P23, F_NEG_2P23,
            F_2P24, F_NEG_2P24,
        ]
        for w in cases:
            with self.subTest(w=hex(w)):
                got = run_op("v_fract_f32_e32 v0, v1", w)
                self.assertEqual(got, F_POS_ZERO)
                self.assertNotEqual(got, F_NEG_ZERO)

    def test_large_numbers_no_fractional_bits(self):
        # 2^23 has no fractional bits -> 0.0
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_2P23), F_POS_ZERO)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_2P23), F_POS_ZERO)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_2P24), F_POS_ZERO)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_2P24), F_POS_ZERO)
        # 2^23 - 0.5 has fractional bit 0.5
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_2P23_M0P5), F_0P5)

    def test_special_ieee_values_execution(self):
        for w in (F_POS_INF, F_NEG_INF, F_QNAN, F_NEG_QNAN, F_SNAN, F_NEG_SNAN):
            for suffix in ("_e32", "_e64", ""):
                with self.subTest(w=hex(w), suffix=suffix):
                    got = run_op(f"v_fract_f32{suffix} v0, v1", w)
                    self.assertTrue(f32_is_nan(got), f"expected NaN, got {got:#010x}")

    def test_vop3_source_modifiers(self):
        cases = [
            ("v_fract_f32_e64 v0, -v1", F_0P25, False, True),       # -(0.25) -> 0.75
            ("v_fract_f32_e64 v0, -v1", F_NEG_0P25, False, True),   # -(-0.25) -> 0.25
            ("v_fract_f32_e64 v0, -v1", F_1P75, False, True),       # -(1.75) -> 0.25
            ("v_fract_f32_e64 v0, -v1", F_POS_ZERO, False, True),   # -(0.0) -> +0.0
            ("v_fract_f32_e64 v0, -v1", F_1P0, False, True),        # -(1.0) -> +0.0
            ("v_fract_f32_e64 v0, |v1|", F_NEG_0P25, True, False),  # |-0.25| = 0.25 -> 0.25
            ("v_fract_f32_e64 v0, |v1|", F_NEG_1P75, True, False),  # |-1.75| = 1.75 -> 0.75
            ("v_fract_f32_e64 v0, -|v1|", F_0P25, True, True),      # -|0.25| = -0.25 -> 0.75
            ("v_fract_f32_e64 v0, -|v1|", F_NEG_0P25, True, True),  # -|-0.25| = -0.25 -> 0.75
            ("v_fract_f32_e64 v0, -|v1|", F_1P75, True, True),      # -|1.75| = -1.75 -> 0.25
        ]
        for line, w, abs_, neg in cases:
            with self.subTest(line=line, w=hex(w)):
                want = oracle_fract_f32(w, abs_=abs_, neg=neg)
                self.assertEqual(run_op(line, w), want)

    def test_vop3_output_modifiers(self):
        cases = [
            ("clamp", F_0P75, dict(clamp=True)),
            ("clamp", F_NEG_0P25, dict(clamp=True)),
            ("clamp", F_NEG_ZERO, dict(clamp=True)),
            ("mul:2", F_0P5, dict(mul=2)),
            ("mul:2", F_0P75, dict(mul=2)),
            ("mul:2", F_NEG_0P25, dict(mul=2)),
            ("mul:4", F_0P25, dict(mul=4)),
            ("mul:4", F_0P5, dict(mul=4)),
            ("mul:4", F_NEG_1P25, dict(mul=4)),
            ("div:2", F_0P5, dict(div=2)),
            ("div:2", F_NEG_1P25, dict(div=2)),
            ("div:2", F_0P25, dict(div=2)),
            ("mul:2 clamp", F_0P25, dict(mul=2, clamp=True)),
            ("mul:2 clamp", F_0P5, dict(mul=2, clamp=True)),
            ("mul:2 clamp", F_0P75, dict(mul=2, clamp=True)),
            ("mul:2 clamp", F_NEG_0P25, dict(mul=2, clamp=True)),
            ("mul:4 clamp", F_0P5, dict(mul=4, clamp=True)),
        ]
        for suffix, w, kw in cases:
            line = f"v_fract_f32_e64 v0, v1 {suffix}"
            want = oracle_fract_f32(w, **kw)
            with self.subTest(line=line, w=hex(w)):
                self.assertEqual(run_op(line, w), want)

        # Explicit verification of key omod scaling values
        self.assertEqual(run_op("v_fract_f32_e64 v0, v1 mul:2", F_0P5), F_1P0)
        self.assertEqual(run_op("v_fract_f32_e64 v0, v1 mul:2 clamp", F_0P75), F_1P0)
        self.assertEqual(run_op("v_fract_f32_e64 v0, v1 mul:4", F_0P5), F_2P0)
        self.assertEqual(run_op("v_fract_f32_e64 v0, v1 div:2", F_0P5), F_0P25)

    def test_inline_float_constants(self):
        for lit, w in (("0.5", F_0P5), ("-0.5", F_NEG_0P5), ("1.0", F_1P0), ("-1.0", F_NEG_1P0)):
            want = oracle_fract_f32(w)
            for form in ("_e32", "_e64", ""):
                with self.subTest(lit=lit, form=form or "bare"):
                    self.assertEqual(run_op(f"v_fract_f32{form} v0, {lit}"), want)

        self.assertEqual(run_op("v_fract_f32_e64 v0, -0.5 mul:2"), F_1P0)
        self.assertEqual(run_op("v_fract_f32_e64 v0, -0.5 div:2"), F_0P25)

    def test_sgpr_sources_and_destination_aliasing(self):
        s4 = F_NEG_0P25
        self.assertEqual(run_op("v_fract_f32_e32 v0, s4", s4=s4), F_0P75)
        self.assertEqual(run_op("v_fract_f32_e64 v0, s4", s4=s4), F_0P75)
        self.assertEqual(run_op("v_fract_f32_e64 v0, -s4", s4=s4), F_0P25)
        self.assertEqual(run_op("v_fract_f32_e64 v0, |s4|", s4=s4), F_0P25)

        # Destination aliasing: dst == src (v0, v0)
        self.assertEqual(
            run_ps([f"v_mov_b32 v0, {F_NEG_0P25:#x}", "v_fract_f32_e32 v0, v0"]),
            F_0P75,
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v0, {F_3P75:#x}", "v_fract_f32 v0, v0"]),
            F_0P75,
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v0, {F_NEG_1P75:#x}", "v_fract_f32_e64 v0, v0"]),
            F_0P25,
        )

    def test_runtime_memory_inputs_via_s_buffer_load(self):
        lanes = [F_NEG_0P25, F_3P75, F_NEG_1P75, F_POS_ZERO]
        mem = {0x1000 + 4 * i: w for i, w in enumerate(lanes)}
        for idx, w in enumerate(lanes):
            reg = f"v{idx + 1}"
            lines = [
                "s_buffer_load_dwordx4 s[4:7], s[0:3], 0",
                "s_waitcnt lgkmcnt(0)",
                "v_mov_b32_e32 v1, s4",
                "v_mov_b32_e32 v2, s5",
                "v_mov_b32_e32 v3, s6",
                "v_mov_b32_e32 v4, s7",
                f"v_mov_b32_e32 v0, {OLD:#x}",
                f"v_fract_f32_e32 v0, {reg}",
            ]
            got = run_ps(lines, push_constants=PUSH_ADDR, memory=mem)
            self.assertEqual(got, oracle_fract_f32(w))


if __name__ == "__main__":
    unittest.main()
