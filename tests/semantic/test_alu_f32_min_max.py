"""Semantic interpreter tests for single-precision floating point min/max (v_min_f32, v_max_f32).

Models IEEE 754-2008 / 2019 binary32 minNum/maxNum and AMD RDNA2 / GFX10 VALU semantics:
  - v_min_f32: D.f = min(S0.f, S1.f)
  - v_max_f32: D.f = max(S0.f, S1.f)

RDNA2 Floating-Point Rules:
  1. NaN suppression (IEEE 754-2008 minNum/maxNum):
     If one operand is NaN and the other is a valid floating-point number, the non-NaN operand
     is returned with its bit pattern intact. If both operands are NaN, a quiet NaN is returned.
  2. Signed zeros (-0.0 < +0.0):
     RDNA2 orders -0.0 strictly below +0.0 (measured on hardware; docs/design.md).
     min(+0.0, -0.0) = -0.0, min(-0.0, +0.0) = -0.0
     max(+0.0, -0.0) = +0.0, max(-0.0, +0.0) = +0.0
  3. Infinities and Extreme Floats:
     -INF < -FLT_MAX < ... < -FLT_MIN < -0.0 < +0.0 < +FLT_MIN < ... < +FLT_MAX < +INF.
  4. VOP2 (e32) and VOP3 (e64) Encoding Forms & Modifiers:
     Input modifiers: neg (-src), abs (|src|), and neg-abs (-|src|).
     Output modifier: clamp to [0.0, 1.0].
  5. Translation lowering:
     tools/rdna2spirv.py maps v_min_f32 and v_max_f32 to GLSL.std.450 NMin and NMax
     (OpExtInst %float %glsl NMin/NMax), which enforce identical NaN-skipping and signed-zero rules.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

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
SIGN_MASK = 0x80000000
EXP_MASK = 0x7F800000
MANT_MASK = 0x007FFFFF

# Canonical IEEE 754 binary32 bit patterns
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
HALF = 0x3F000000          # 0.5
NEG_HALF = 0xBF000000      # -0.5
ONE = 0x3F800000           # 1.0
NEG_ONE = 0xBF800000       # -1.0
ONE_HALF = 0x3FC00000      # 1.5
TWO = 0x40000000           # 2.0
NEG_TWO = 0xC0000000       # -2.0
THREE = 0x40400000         # 3.0
FOUR = 0x40800000          # 4.0
FLT_MIN = 0x00800000       # Smallest positive normal: 2^-126
NEG_FLT_MIN = 0x80800000   # -2^-126
FLT_MAX = 0x7F7FFFFF       # (2 - 2^-23) * 2^127
NEG_FLT_MAX = 0xFF7FFFFF
POS_INF = 0x7F800000       # +infinity
NEG_INF = 0xFF800000       # -infinity
QNAN = 0x7FC00000          # Canonical quiet NaN
NEG_QNAN = 0xFFC00000      # Negative quiet NaN
QNAN_PAYLOAD = 0x7FC10001  # Quiet NaN with payload
SNAN = 0x7F800001          # Signaling NaN
SUBNORM_MIN = 0x00000001   # Smallest subnormal: 2^-149
SUBNORM_MAX = 0x007FFFFF   # Largest positive subnormal
NEG_SUBNORM_MIN = 0x80000001
NEG_SUBNORM_MAX = 0x807FFFFF

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Encode float to IEEE 754 single-precision 32-bit unsigned integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def is_nan_bits(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 NaN."""
    bits &= MASK32
    return (bits & EXP_MASK) == EXP_MASK and (bits & MANT_MASK) != 0


def is_qnan_bits(bits: int) -> bool:
    """True if bits represent a quiet NaN (MSB of mantissa is 1)."""
    bits &= MASK32
    return is_nan_bits(bits) and bool(bits & 0x00400000)


def is_snan_bits(bits: int) -> bool:
    """True if bits represent a signaling NaN (MSB of mantissa is 0)."""
    bits &= MASK32
    return is_nan_bits(bits) and not bool(bits & 0x00400000)


def is_subnormal_bits(bits: int) -> bool:
    """True if bits represent an IEEE 754 subnormal (denormal) number."""
    bits &= MASK32
    return (bits & EXP_MASK) == 0 and (bits & MANT_MASK) != 0


def zero_ordered_key(v: float) -> tuple[float, float]:
    """Sort key ordering -0.0 strictly below +0.0."""
    return (v, math.copysign(1.0, v))


def ref_min_f32(s0: int, s1: int) -> int:
    """Pure bitwise reference oracle for RDNA2 v_min_f32.

    Rules:
      - If one operand is NaN, return the other operand unchanged.
      - If both operands are NaN, return canonical quiet NaN (0x7FC00000).
      - If values are non-NaN, order -0.0 < +0.0.
    """
    s0 &= MASK32
    s1 &= MASK32
    nan0 = is_nan_bits(s0)
    nan1 = is_nan_bits(s1)
    if nan0 and nan1:
        return QNAN
    if nan0:
        return s1
    if nan1:
        return s0

    f0 = bits_to_f32(s0)
    f1 = bits_to_f32(s1)
    k0 = zero_ordered_key(f0)
    k1 = zero_ordered_key(f1)
    return s0 if k0 <= k1 else s1


def ref_max_f32(s0: int, s1: int) -> int:
    """Pure bitwise reference oracle for RDNA2 v_max_f32.

    Rules:
      - If one operand is NaN, return the other operand unchanged.
      - If both operands are NaN, return canonical quiet NaN (0x7FC00000).
      - If values are non-NaN, order -0.0 < +0.0.
    """
    s0 &= MASK32
    s1 &= MASK32
    nan0 = is_nan_bits(s0)
    nan1 = is_nan_bits(s1)
    if nan0 and nan1:
        return QNAN
    if nan0:
        return s1
    if nan1:
        return s0

    f0 = bits_to_f32(s0)
    f1 = bits_to_f32(s1)
    k0 = zero_ordered_key(f0)
    k1 = zero_ordered_key(f1)
    return s0 if k0 >= k1 else s1


def apply_f32_modifiers(bits: int, neg: bool = False, abs_val: bool = False) -> int:
    """Applies VOP3 input modifiers: abs (|x|) first, then neg (-x)."""
    bits &= MASK32
    if abs_val:
        bits &= ~SIGN_MASK
    if neg:
        bits ^= SIGN_MASK
    return bits & MASK32


def apply_f32_clamp(bits: int) -> int:
    """Applies VOP3 clamp modifier: clamps result to [0.0, 1.0]."""
    bits &= MASK32
    if is_nan_bits(bits):
        return bits
    val = bits_to_f32(bits)
    if val < 0.0 or (val == 0.0 and (bits & SIGN_MASK)):
        return POS_ZERO
    if val > 1.0:
        return ONE
    return bits


_TMP_DIR: tempfile.TemporaryDirectory[str] | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_f32_min_max.s")
    with open(os.path.join(_TMP_DIR.name, "ps_f32_min_max.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_f32_2op(
    op: str,
    s0: int,
    s1: int,
    e64: bool = False,
    neg0: bool = False,
    abs0: bool = False,
    neg1: bool = False,
    abs1: bool = False,
    clamp: bool = False,
) -> int:
    """Translates and runs a 2-operand float min/max instruction through rdna2spirv and spirv_mini_interp."""
    if _TMP_DIR is None or _SRC_PATH is None:
        setUpModule()
        assert _SRC_PATH is not None

    def fmt_src(v_name: str, neg: bool, abs_val: bool) -> str:
        s = v_name
        if abs_val:
            s = f"|{s}|"
        if neg:
            s = f"-{s}"
        return s

    instr_name = f"{op}_e64" if e64 else (f"{op}_e32" if not (neg0 or abs0 or neg1 or abs1 or clamp) else f"{op}_e64")
    src0_str = fmt_src("v1", neg0, abs0)
    src1_str = fmt_src("v2", neg1, abs1)
    clamp_str = " clamp" if clamp else ""

    lines = [
        f"\tv_mov_b32 v1, {s0 & MASK32:#x}",
        f"\tv_mov_b32 v2, {s1 & MASK32:#x}",
        f"\t{instr_name} v0, {src0_str}, {src1_str}{clamp_str}",
        "\texp mrt0 v0, v0, v0, v0 done vm",
        "\ts_endpgm",
    ]
    asm = "; ps_f32_min_max\n" + "\n".join(lines) + "\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class TestF32ReferenceOracle2Op(unittest.TestCase):
    """Verifies the pure-Python reference oracle ref_min_f32 and ref_max_f32."""

    def test_ordinary_comparisons(self) -> None:
        self.assertEqual(ref_min_f32(ONE, TWO), ONE)
        self.assertEqual(ref_min_f32(TWO, ONE), ONE)
        self.assertEqual(ref_max_f32(ONE, TWO), TWO)
        self.assertEqual(ref_max_f32(TWO, ONE), TWO)
        self.assertEqual(ref_min_f32(NEG_ONE, NEG_TWO), NEG_TWO)
        self.assertEqual(ref_max_f32(NEG_ONE, NEG_TWO), NEG_ONE)
        self.assertEqual(ref_min_f32(NEG_ONE, ONE), NEG_ONE)
        self.assertEqual(ref_max_f32(NEG_ONE, ONE), ONE)

    def test_signed_zeros(self) -> None:
        self.assertEqual(ref_min_f32(POS_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(ref_min_f32(NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(ref_min_f32(POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(ref_min_f32(NEG_ZERO, NEG_ZERO), NEG_ZERO)

        self.assertEqual(ref_max_f32(POS_ZERO, NEG_ZERO), POS_ZERO)
        self.assertEqual(ref_max_f32(NEG_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(ref_max_f32(POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(ref_max_f32(NEG_ZERO, NEG_ZERO), NEG_ZERO)

    def test_infinities(self) -> None:
        self.assertEqual(ref_min_f32(NEG_INF, POS_INF), NEG_INF)
        self.assertEqual(ref_max_f32(NEG_INF, POS_INF), POS_INF)
        self.assertEqual(ref_min_f32(POS_INF, FLT_MAX), FLT_MAX)
        self.assertEqual(ref_max_f32(POS_INF, FLT_MAX), POS_INF)
        self.assertEqual(ref_min_f32(NEG_INF, NEG_FLT_MAX), NEG_INF)
        self.assertEqual(ref_max_f32(NEG_INF, NEG_FLT_MAX), NEG_FLT_MAX)

    def test_normals_and_subnormals(self) -> None:
        self.assertEqual(ref_min_f32(FLT_MIN, SUBNORM_MAX), SUBNORM_MAX)
        self.assertEqual(ref_max_f32(FLT_MIN, SUBNORM_MAX), FLT_MIN)
        self.assertEqual(ref_min_f32(SUBNORM_MIN, SUBNORM_MAX), SUBNORM_MIN)
        self.assertEqual(ref_max_f32(SUBNORM_MIN, SUBNORM_MAX), SUBNORM_MAX)
        self.assertEqual(ref_min_f32(NEG_FLT_MIN, NEG_SUBNORM_MAX), NEG_FLT_MIN)
        self.assertEqual(ref_max_f32(NEG_FLT_MIN, NEG_SUBNORM_MAX), NEG_SUBNORM_MAX)

    def test_nan_suppression(self) -> None:
        # If one operand is NaN, return the other operand exactly.
        self.assertEqual(ref_min_f32(QNAN, ONE), ONE)
        self.assertEqual(ref_min_f32(ONE, QNAN), ONE)
        self.assertEqual(ref_max_f32(QNAN, ONE), ONE)
        self.assertEqual(ref_max_f32(ONE, QNAN), ONE)

        # Signaling NaN is also suppressed.
        self.assertEqual(ref_min_f32(SNAN, TWO), TWO)
        self.assertEqual(ref_max_f32(TWO, SNAN), TWO)

        # Both NaN returns quiet NaN.
        self.assertTrue(is_nan_bits(ref_min_f32(QNAN, SNAN)))
        self.assertTrue(is_nan_bits(ref_max_f32(QNAN, SNAN)))

    def test_nan_payload_preservation(self) -> None:
        # The non-NaN operand must have its exact bit pattern preserved.
        custom_float = 0x3EAAAAAB  # 0.33333334
        self.assertEqual(ref_min_f32(QNAN_PAYLOAD, custom_float), custom_float)
        self.assertEqual(ref_max_f32(custom_float, QNAN_PAYLOAD), custom_float)


class TestVMinVMaxF32Execution(unittest.TestCase):
    """Executes synthetic shaders testing v_min_f32 and v_max_f32 in spirv_mini_interp."""

    TEST_PAIRS = [
        (ONE, TWO),
        (TWO, ONE),
        (NEG_ONE, NEG_TWO),
        (NEG_TWO, NEG_ONE),
        (HALF, ONE_HALF),
        (POS_INF, NEG_INF),
        (FLT_MAX, ONE),
        (NEG_FLT_MAX, NEG_ONE),
        (FLT_MIN, SUBNORM_MAX),
        (SUBNORM_MIN, FLT_MIN),
        (QNAN, THREE),
        (FOUR, QNAN),
    ]

    def test_v_min_f32_e32_and_e64(self) -> None:
        for s0, s1 in self.TEST_PAIRS:
            expected = ref_min_f32(s0, s1)
            with self.subTest(s0=hex(s0), s1=hex(s1), form="e32"):
                got_e32 = run_f32_2op("v_min_f32", s0, s1, e64=False)
                self.assertEqual(got_e32, expected)
            with self.subTest(s0=hex(s0), s1=hex(s1), form="e64"):
                got_e64 = run_f32_2op("v_min_f32", s0, s1, e64=True)
                self.assertEqual(got_e64, expected)

    def test_v_max_f32_e32_and_e64(self) -> None:
        for s0, s1 in self.TEST_PAIRS:
            expected = ref_max_f32(s0, s1)
            with self.subTest(s0=hex(s0), s1=hex(s1), form="e32"):
                got_e32 = run_f32_2op("v_max_f32", s0, s1, e64=False)
                self.assertEqual(got_e32, expected)
            with self.subTest(s0=hex(s0), s1=hex(s1), form="e64"):
                got_e64 = run_f32_2op("v_max_f32", s0, s1, e64=True)
                self.assertEqual(got_e64, expected)


class TestSignedZerosTieBreak(unittest.TestCase):
    """Verifies that RDNA2 -0.0 < +0.0 ordering is strictly preserved by translator and interpreter."""

    def test_min_zero_combinations(self) -> None:
        self.assertEqual(run_f32_2op("v_min_f32", POS_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(run_f32_2op("v_min_f32", NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(run_f32_2op("v_min_f32", POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(run_f32_2op("v_min_f32", NEG_ZERO, NEG_ZERO), NEG_ZERO)

    def test_max_zero_combinations(self) -> None:
        self.assertEqual(run_f32_2op("v_max_f32", POS_ZERO, NEG_ZERO), POS_ZERO)
        self.assertEqual(run_f32_2op("v_max_f32", NEG_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(run_f32_2op("v_max_f32", POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(run_f32_2op("v_max_f32", NEG_ZERO, NEG_ZERO), NEG_ZERO)


class TestVALUModifiers2Op(unittest.TestCase):
    """Verifies VOP3 input modifiers (neg, abs) and output clamp."""

    def test_neg_modifier(self) -> None:
        # v_min_f32_e64 v0, -v1, v2 with v1=1.0, v2=0.5 -> min(-1.0, 0.5) = -1.0
        got = run_f32_2op("v_min_f32", ONE, HALF, e64=True, neg0=True)
        self.assertEqual(got, NEG_ONE)

        # v_max_f32_e64 v0, -v1, v2 with v1=1.0, v2=0.5 -> max(-1.0, 0.5) = 0.5
        got = run_f32_2op("v_max_f32", ONE, HALF, e64=True, neg0=True)
        self.assertEqual(got, HALF)

    def test_abs_modifier(self) -> None:
        # v_min_f32_e64 v0, |v1|, v2 with v1=-2.0, v2=1.5 -> min(2.0, 1.5) = 1.5
        got = run_f32_2op("v_min_f32", NEG_TWO, ONE_HALF, e64=True, abs0=True)
        self.assertEqual(got, ONE_HALF)

        # v_max_f32_e64 v0, |v1|, |v2| with v1=-2.0, v2=-3.0 -> max(2.0, 3.0) = 3.0
        got = run_f32_2op("v_max_f32", NEG_TWO, f32_to_bits(-3.0), e64=True, abs0=True, abs1=True)
        self.assertEqual(got, THREE)

    def test_neg_abs_combination(self) -> None:
        # -|src|: abs clears sign bit first, neg flips sign bit second -> -|1.0| = -1.0, -|-2.0| = -2.0
        got_min = run_f32_2op("v_min_f32", TWO, ONE, e64=True, neg0=True, abs0=True)
        self.assertEqual(got_min, NEG_TWO)
        got_max = run_f32_2op("v_max_f32", NEG_TWO, ONE, e64=True, neg0=True, abs0=True)
        self.assertEqual(got_max, ONE)

    def test_clamp_modifier(self) -> None:
        # v_max_f32_e64 v0, v1, v2 clamp with v1=2.0, v2=3.0 -> max(2.0, 3.0) = 3.0, clamped to 1.0
        got = run_f32_2op("v_max_f32", TWO, THREE, e64=True, clamp=True)
        self.assertEqual(got, ONE)

        # v_min_f32_e64 v0, v1, v2 clamp with v1=-1.0, v2=-2.0 -> min(-1.0, -2.0) = -2.0, clamped to +0.0
        got = run_f32_2op("v_min_f32", NEG_ONE, NEG_TWO, e64=True, clamp=True)
        self.assertEqual(got, POS_ZERO)


if __name__ == "__main__":
    unittest.main()
