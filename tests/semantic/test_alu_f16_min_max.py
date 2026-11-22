"""Semantic interpreter tests for RDNA2 half-precision floating-point min/max:
v_min_f16 and v_max_f16 (VOP2/VOP3 scalar forms),
v_pk_min_f16 and v_pk_max_f16 (VOP3P packed SIMD forms).

Features and specifications:
- IEEE 754-2008 / RDNA rules: NaN suppression (return non-NaN operand; return quiet NaN if both are NaN).
- Signed zero ordering: -0.0 < +0.0 tie-breaking.
- Packed 16-bit lane execution (v_pk_min_f16 / v_pk_max_f16 across lo and hi halves).
- Subnormal representation and gradual underflow (denormals preserved).
- Output clamp to [0.0, 1.0].
- Destination register half preservation on scalar f16 ops.
- VOP3 op_sel and input modifiers (negation, absolute value).
- Multi-lane SIMD divergence under EXEC mask.
- Standard library Python only; zero game data.
"""
from __future__ import annotations

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
from support import STUB_DXBC, mrt0_words  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
OLD = 0xA55A5AA5

# IEEE 754-2008 binary16 bit constants
POS_ZERO = 0x0000          # +0.0
NEG_ZERO = 0x8000          # -0.0
POS_ONE = 0x3C00           # +1.0
NEG_ONE = 0xBC00           # -1.0
POS_HALF = 0x3800          # +0.5
NEG_HALF = 0xB800          # -0.5
POS_TWO = 0x4000           # +2.0
NEG_TWO = 0xC000           # -2.0
POS_THREE = 0x4200         # +3.0
NEG_THREE = 0xC200         # -3.0
POS_INF = 0x7C00           # +inf
NEG_INF = 0xFC00           # -inf
QNAN_CANONICAL = 0x7E00    # canonical quiet NaN
QNAN_ALT = 0x7E11          # quiet NaN with payload
SNAN = 0x7D00              # signaling NaN
NEG_QNAN = 0xFE22          # negative quiet NaN

# Subnormal (denormal) binary16 constants
POS_MIN_SUBNORM = 0x0001   # +2^-24 (~5.96e-8), smallest positive subnormal
POS_MAX_SUBNORM = 0x03FF   # largest positive subnormal (~6.097e-5)
NEG_MIN_SUBNORM = 0x8001   # -2^-24, smallest negative subnormal
NEG_MAX_SUBNORM = 0x83FF   # largest negative subnormal

# Normal binary16 boundaries
POS_MIN_NORMAL = 0x0400    # +2^-14 (~6.104e-5), smallest positive normal
POS_MAX_NORMAL = 0x7BFF    # +65504.0, largest finite binary16
NEG_MAX_NORMAL = 0xFBFF    # -65504.0


def f16_to_float(bits: int) -> float:
    """Decode 16-bit integer as IEEE 754 half-precision float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def float_to_f16_bits(f: float) -> int:
    """Encode float into IEEE 754 half-precision bits."""
    if math.isnan(f):
        return QNAN_CANONICAL
    try:
        return struct.unpack("<H", struct.pack("<e", f))[0]
    except OverflowError:
        return POS_INF if f > 0 else NEG_INF


def is_nan16(bits: int) -> bool:
    """Check if 16-bit integer represents a binary16 NaN."""
    return (bits & 0x7C00) == 0x7C00 and (bits & 0x03FF) != 0


def is_subnormal16(bits: int) -> bool:
    """Check if 16-bit integer represents a binary16 subnormal."""
    return (bits & 0x7C00) == 0 and (bits & 0x03FF) != 0


def half_bits(word: int, hi: int) -> int:
    """Extract low (hi=0) or high (hi=1) 16 bits of a 32-bit dword."""
    return ((word >> 16) if hi else word) & MASK16


def merge_half(res16: int, dst_hi: int, old: int = OLD) -> int:
    """Merge 16-bit result into low or high half of 32-bit word, preserving the other half."""
    res16 &= MASK16
    return ((res16 << 16) | (old & MASK16)) if dst_hi else ((old & 0xFFFF0000) | res16)


def ref_min_f16(a_bits: int, b_bits: int, *, clamp: bool = False) -> int:
    """Reference oracle for RDNA2 v_min_f16 conforming to IEEE 754-2008 / GLSL NMin.

    Rules:
    - NaN suppression: if one operand is NaN, return the other operand.
    - If both are NaN, return a quiet NaN.
    - Signed zero ordering: -0.0 < +0.0.
    - Clamping: if clamp=True, output clamped to [0.0, 1.0] (negative / -0.0 maps to +0.0).
    """
    a_nan = is_nan16(a_bits)
    b_nan = is_nan16(b_bits)
    if a_nan and b_nan:
        chosen_bits = QNAN_CANONICAL
    elif a_nan:
        chosen_bits = b_bits & MASK16
    elif b_nan:
        chosen_bits = a_bits & MASK16
    else:
        # Check signed zero tie-break: -0.0 is strictly smaller than +0.0
        is_a_neg_zero = (a_bits & MASK16) == NEG_ZERO
        is_a_pos_zero = (a_bits & MASK16) == POS_ZERO
        is_b_neg_zero = (b_bits & MASK16) == NEG_ZERO
        is_b_pos_zero = (b_bits & MASK16) == POS_ZERO

        if (is_a_neg_zero and is_b_pos_zero) or (is_a_pos_zero and is_b_neg_zero):
            chosen_bits = NEG_ZERO
        elif is_a_neg_zero and is_b_neg_zero:
            chosen_bits = NEG_ZERO
        elif is_a_pos_zero and is_b_pos_zero:
            chosen_bits = POS_ZERO
        else:
            fa, fb = f16_to_float(a_bits), f16_to_float(b_bits)
            key_a = (fa, math.copysign(1.0, fa))
            key_b = (fb, math.copysign(1.0, fb))
            chosen_bits = (a_bits & MASK16) if key_a <= key_b else (b_bits & MASK16)

    if clamp:
        if is_nan16(chosen_bits):
            return POS_ZERO
        cf = f16_to_float(chosen_bits)
        if cf <= 0.0 or (chosen_bits & MASK16) == NEG_ZERO:
            return POS_ZERO
        if cf >= 1.0:
            return POS_ONE
        return float_to_f16_bits(cf)

    return chosen_bits & MASK16


def ref_max_f16(a_bits: int, b_bits: int, *, clamp: bool = False) -> int:
    """Reference oracle for RDNA2 v_max_f16 conforming to IEEE 754-2008 / GLSL NMax.

    Rules:
    - NaN suppression: if one operand is NaN, return the other operand.
    - If both are NaN, return a quiet NaN.
    - Signed zero ordering: -0.0 < +0.0.
    - Clamping: if clamp=True, output clamped to [0.0, 1.0].
    """
    a_nan = is_nan16(a_bits)
    b_nan = is_nan16(b_bits)
    if a_nan and b_nan:
        chosen_bits = QNAN_CANONICAL
    elif a_nan:
        chosen_bits = b_bits & MASK16
    elif b_nan:
        chosen_bits = a_bits & MASK16
    else:
        is_a_neg_zero = (a_bits & MASK16) == NEG_ZERO
        is_a_pos_zero = (a_bits & MASK16) == POS_ZERO
        is_b_neg_zero = (b_bits & MASK16) == NEG_ZERO
        is_b_pos_zero = (b_bits & MASK16) == POS_ZERO

        if (is_a_neg_zero and is_b_pos_zero) or (is_a_pos_zero and is_b_neg_zero):
            chosen_bits = POS_ZERO
        elif is_a_neg_zero and is_b_neg_zero:
            chosen_bits = NEG_ZERO
        elif is_a_pos_zero and is_b_pos_zero:
            chosen_bits = POS_ZERO
        else:
            fa, fb = f16_to_float(a_bits), f16_to_float(b_bits)
            key_a = (fa, math.copysign(1.0, fa))
            key_b = (fb, math.copysign(1.0, fb))
            chosen_bits = (a_bits & MASK16) if key_a >= key_b else (b_bits & MASK16)

    if clamp:
        if is_nan16(chosen_bits):
            return POS_ZERO
        cf = f16_to_float(chosen_bits)
        if cf <= 0.0 or (chosen_bits & MASK16) == NEG_ZERO:
            return POS_ZERO
        if cf >= 1.0:
            return POS_ONE
        return float_to_f16_bits(cf)

    return chosen_bits & MASK16


# -----------------------------------------------------------------------------
# Translation and Execution Harness
# -----------------------------------------------------------------------------

_TMP_DIR = None
_PS_PATH = None


def setUpModule():
    global _TMP_DIR, _PS_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _PS_PATH = os.path.join(disasm_dir, "ps_alu_f16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_alu_f16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR, _PS_PATH
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _PS_PATH = None


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_alu_f16_min_max\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_PS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_PS_PATH)
        gc.collect()
        return spv


def run_ps(lines: list[str]) -> int:
    spv = translate_ps(lines)
    return si.run_fragment(spv)["%out_mrt0"][0]


def run_op(op_line: str, v1: int = 0, v2: int = 0, *, old: int = OLD, s4: int = 0, s5: int = 0) -> int:
    lines = [
        f"v_mov_b32 v0, {old:#x}",
        f"v_mov_b32 v1, {v1:#x}",
        f"v_mov_b32 v2, {v2:#x}",
        f"s_mov_b32 s4, {s4:#x}",
        f"s_mov_b32 s5, {s5:#x}",
        op_line,
    ]
    return run_ps(lines)


# -----------------------------------------------------------------------------
# Commit 1 Tests: Reference Oracle and Scalar v_min_f16 / v_max_f16
# -----------------------------------------------------------------------------

class ReferenceOracleTests(unittest.TestCase):
    """Test IEEE 754 half-precision reference oracle rules."""

    def test_signed_zero_ordering(self):
        self.assertEqual(ref_min_f16(NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(ref_min_f16(POS_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(ref_max_f16(NEG_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(ref_max_f16(POS_ZERO, NEG_ZERO), POS_ZERO)
        self.assertEqual(ref_min_f16(NEG_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(ref_max_f16(POS_ZERO, POS_ZERO), POS_ZERO)

    def test_subnormal_ordering_and_gradual_underflow(self):
        # POS_MIN_SUBNORM is strictly greater than POS_ZERO and strictly less than POS_MIN_NORMAL
        self.assertEqual(ref_min_f16(POS_MIN_SUBNORM, POS_ZERO), POS_ZERO)
        self.assertEqual(ref_max_f16(POS_MIN_SUBNORM, POS_ZERO), POS_MIN_SUBNORM)
        self.assertEqual(ref_min_f16(POS_MAX_SUBNORM, POS_MIN_NORMAL), POS_MAX_SUBNORM)
        self.assertEqual(ref_max_f16(POS_MAX_SUBNORM, POS_MIN_NORMAL), POS_MIN_NORMAL)
        # Negative subnormals vs negative zero
        self.assertEqual(ref_min_f16(NEG_MIN_SUBNORM, NEG_ZERO), NEG_MIN_SUBNORM)
        self.assertEqual(ref_max_f16(NEG_MIN_SUBNORM, NEG_ZERO), NEG_ZERO)

    def test_infinities(self):
        self.assertEqual(ref_min_f16(POS_INF, POS_MAX_NORMAL), POS_MAX_NORMAL)
        self.assertEqual(ref_max_f16(POS_INF, POS_MAX_NORMAL), POS_INF)
        self.assertEqual(ref_min_f16(NEG_INF, NEG_MAX_NORMAL), NEG_INF)
        self.assertEqual(ref_max_f16(NEG_INF, NEG_MAX_NORMAL), NEG_MAX_NORMAL)
        self.assertEqual(ref_min_f16(NEG_INF, POS_INF), NEG_INF)
        self.assertEqual(ref_max_f16(NEG_INF, POS_INF), POS_INF)

    def test_clamp_oracle(self):
        self.assertEqual(ref_min_f16(POS_TWO, POS_THREE, clamp=True), POS_ONE)
        self.assertEqual(ref_min_f16(NEG_TWO, NEG_ONE, clamp=True), POS_ZERO)
        self.assertEqual(ref_max_f16(NEG_HALF, NEG_ONE, clamp=True), POS_ZERO)
        self.assertEqual(ref_max_f16(POS_HALF, POS_HALF, clamp=True), POS_HALF)
        self.assertEqual(ref_min_f16(POS_HALF, POS_ONE, clamp=True), POS_HALF)

    def test_roundtrip_binary16(self):
        test_patterns = [
            POS_ZERO, NEG_ZERO, POS_ONE, NEG_ONE, POS_HALF, NEG_HALF,
            POS_TWO, NEG_TWO, POS_INF, NEG_INF, POS_MIN_SUBNORM,
            POS_MAX_SUBNORM, NEG_MIN_SUBNORM, POS_MIN_NORMAL, POS_MAX_NORMAL,
        ]
        for bits in test_patterns:
            flt = f16_to_float(bits)
            enc = float_to_f16_bits(flt)
            self.assertEqual(enc, bits, f"Pattern {bits:#06x} failed roundtrip, got {enc:#06x}")


class ScalarMinMaxF16Tests(unittest.TestCase):
    """End-to-end interpreter tests for scalar v_min_f16 and v_max_f16."""

    def test_scalar_min_f16_basic(self):
        # 1.0 vs 2.0 -> min is 1.0 (0x3C00)
        got = run_op("v_min_f16_e32 v0, v1, v2", v1=POS_ONE, v2=POS_TWO)
        self.assertEqual(got, merge_half(POS_ONE, dst_hi=0))

        # 2.0 vs -3.0 -> min is -3.0 (0xC200)
        got = run_op("v_min_f16_e32 v0, v1, v2", v1=POS_TWO, v2=NEG_THREE)
        self.assertEqual(got, merge_half(NEG_THREE, dst_hi=0))

    def test_scalar_max_f16_basic(self):
        # 1.0 vs 2.0 -> max is 2.0 (0x4000)
        got = run_op("v_max_f16_e32 v0, v1, v2", v1=POS_ONE, v2=POS_TWO)
        self.assertEqual(got, merge_half(POS_TWO, dst_hi=0))

        # -1.0 vs -2.0 -> max is -1.0 (0xBC00)
        got = run_op("v_max_f16_e32 v0, v1, v2", v1=NEG_ONE, v2=NEG_TWO)
        self.assertEqual(got, merge_half(NEG_ONE, dst_hi=0))

    def test_destination_half_preservation(self):
        # VOP3 op_sel dst bit: dst=0 writes low half, dst=1 writes high half; other half is kept!
        # dst=0
        got_lo = run_op("v_min_f16_e64 v0, v1, v2 op_sel:[0,0,0]", v1=POS_HALF, v2=POS_ONE, old=OLD)
        self.assertEqual(got_lo, merge_half(POS_HALF, dst_hi=0, old=OLD))
        self.assertEqual(got_lo & 0xFFFF0000, OLD & 0xFFFF0000)

        # dst=1
        got_hi = run_op("v_max_f16_e64 v0, v1, v2 op_sel:[0,0,1]", v1=POS_HALF, v2=POS_ONE, old=OLD)
        self.assertEqual(got_hi, merge_half(POS_ONE, dst_hi=1, old=OLD))
        self.assertEqual(got_hi & 0x0000FFFF, OLD & 0x0000FFFF)

    def test_subnormal_handling_scalar(self):
        # Smallest positive subnormal vs positive zero: min is zero, max is subnormal
        got_min = run_op("v_min_f16_e32 v0, v1, v2", v1=POS_MIN_SUBNORM, v2=POS_ZERO)
        self.assertEqual(got_min, merge_half(POS_ZERO, dst_hi=0))

        got_max = run_op("v_max_f16_e32 v0, v1, v2", v1=POS_MIN_SUBNORM, v2=POS_ZERO)
        self.assertEqual(got_max, merge_half(POS_MIN_SUBNORM, dst_hi=0))

        # Subnormal vs normal: max subnormal vs min normal
        got_min2 = run_op("v_min_f16_e32 v0, v1, v2", v1=POS_MAX_SUBNORM, v2=POS_MIN_NORMAL)
        self.assertEqual(got_min2, merge_half(POS_MAX_SUBNORM, dst_hi=0))

        got_max2 = run_op("v_max_f16_e32 v0, v1, v2", v1=POS_MAX_SUBNORM, v2=POS_MIN_NORMAL)
        self.assertEqual(got_max2, merge_half(POS_MIN_NORMAL, dst_hi=0))

    def test_output_clamp_scalar(self):
        # Clamp to [0.0, 1.0]
        # min(2.0, 3.0) = 2.0 -> clamped to 1.0 (0x3C00)
        got = run_op("v_min_f16_e64 v0, v1, v2 clamp", v1=POS_TWO, v2=POS_THREE)
        self.assertEqual(got, merge_half(POS_ONE, dst_hi=0))

        # max(-2.0, -1.0) = -1.0 -> clamped to +0.0 (0x0000)
        got2 = run_op("v_max_f16_e64 v0, v1, v2 clamp", v1=NEG_TWO, v2=NEG_ONE)
        self.assertEqual(got2, merge_half(POS_ZERO, dst_hi=0))

        # min(0.5, 0.75) within range -> unchanged
        got3 = run_op("v_min_f16_e64 v0, v1, v2 clamp", v1=POS_HALF, v2=POS_ONE)
        self.assertEqual(got3, merge_half(POS_HALF, dst_hi=0))

    def test_source_modifiers_neg_abs(self):
        # -v1 negates source operand
        # min(-(-2.0), 1.0) = min(2.0, 1.0) = 1.0
        got = run_op("v_min_f16_e64 v0, -v1, v2", v1=NEG_TWO, v2=POS_ONE)
        self.assertEqual(got, merge_half(POS_ONE, dst_hi=0))

        # |v1| absolute value clears sign
        # min(|-2.0|, 0.5) = min(2.0, 0.5) = 0.5
        got2 = run_op("v_min_f16_e64 v0, |v1|, v2", v1=NEG_TWO, v2=POS_HALF)
        self.assertEqual(got2, merge_half(POS_HALF, dst_hi=0))

        # max(-|v1|, v2) with v1 = 1.0, v2 = -2.0 -> max(-1.0, -2.0) = -1.0
        got3 = run_op("v_max_f16_e64 v0, -|v1|, v2", v1=POS_ONE, v2=NEG_TWO)
        self.assertEqual(got3, merge_half(NEG_ONE, dst_hi=0))

    def test_inline_constants_and_sgprs(self):
        # Inline float constant 1.0 vs v2 = 2.0 -> min is 1.0
        got = run_op("v_min_f16_e64 v0, 1.0, v2", v2=POS_TWO)
        self.assertEqual(got, merge_half(POS_ONE, dst_hi=0))

        # Inline float constant 0.0 vs v2 = -1.0 -> max is 0.0
        got2 = run_op("v_max_f16_e64 v0, 0.0, v2", v2=NEG_ONE)
        self.assertEqual(got2, merge_half(POS_ZERO, dst_hi=0))

        # SGPR operand s4 = 2.0 vs v2 = 0.5 -> min is 0.5
        got3 = run_op("v_min_f16_e64 v0, s4, v2", s4=POS_TWO, v2=POS_HALF)
        self.assertEqual(got3, merge_half(POS_HALF, dst_hi=0))

        # SGPR operand s5 = -0.5 vs v1 = 1.0 -> max is 1.0
        got4 = run_op("v_max_f16_e64 v0, v1, s5", v1=POS_ONE, s5=NEG_HALF)
        self.assertEqual(got4, merge_half(POS_ONE, dst_hi=0))


if __name__ == "__main__":
    unittest.main()
