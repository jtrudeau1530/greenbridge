"""End-to-end semantic value tests for RDNA2 v_fract_f16 (S0.f16 - floor(S0.f16)).

Covers v_fract_f16 executed end-to-end through tools/rdna2spirv.py and
tests/semantic/spirv_mini_interp.py against an independent Fraction-based
IEEE-754 binary16 `x - floor(x)` oracle:
  - Positive & negative finite non-integers, including tiny negatives and
    negative subnormals (-0.0001, -1e-5, -2**-12, -2**-24) whose exact
    x - floor(x) rounds to 1.0 in binary16 and maps to 0x3bff (the largest
    binary16 strictly below 1.0), plus the adjacent 0x3bfe / 0x3bff edge.
  - Exact integers (0.0, 1.0, -1.0, 65504.0, -65504.0 -> +0.0) and -0.0 -> +0.0.
  - +-inf and qNaN/sNaN behavior (GLSL.std.450 Fract returns NaN; fract_below_one
    uses OpFOrdEqual + OpSelect so NaN survives into f16_bits -> qNaN).
  - _e32, _e64, and bare mnemonics, keeping the untouched 16-bit half of dst.
  - Runtime memory inputs via s_buffer_load_dwordx4 alongside immediate inputs.
  - All 4 op_sel:[src,dst] combinations ([0,0], [1,0], [0,1], [1,1]).
  - VOP3 source modifiers (-v1, |v1|, -|v1|) and output modifiers
    (clamp, mul:2, mul:4, div:2, mul:2 clamp, mul:4 clamp).
  - Inline f16 constants (0.5, -0.5, 1.0, -1.0), SGPR sources, destination
    aliasing (v0, v0), single-lane and multi-lane EXEC masking, and SPIR-V
    sabotage negative controls (Fract -> Trunc / Floor).

Note: Under VOP3 `clamp`, GLSL.std.450 FClamp on NaN is undefined in SPIR-V
and spirv_mini_interp.py raises UnsupportedConstruct (documented in
docs/design.md), so `clamp` value tests use finite inputs. Standard library
only; hand-written RDNA2 disassembly listings only; zero game data.
"""
from fractions import Fraction
import gc
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

# Representative binary16 encodings (cross-checked in OracleTests).
H_POS_ZERO = 0x0000         # +0.0
H_NEG_ZERO = 0x8000         # -0.0
H_SUB_MIN = 0x0001          # +2^-24 (~5.96e-8)
H_NEG_SUB_MIN = 0x8001      # -2^-24
H_SUB_1E5 = 0x00A8          # +1e-5 rounded to f16 subnormal (~1.001e-5)
H_NEG_SUB_1E5 = 0x80A8      # -1e-5 rounded to f16 subnormal (~-1.001e-5)
H_SUB_MAX = 0x03FF          # +1023 * 2^-24 (largest +subnormal)
H_NEG_SUB_MAX = 0x83FF      # -1023 * 2^-24 (largest-magnitude -subnormal)
H_NORM_MIN = 0x0400         # +2^-14 (smallest +normal)
H_NEG_NORM_MIN = 0x8400     # -2^-14
H_1E4 = 0x068E              # +0.0001 rounded to f16 (~1.000166e-4)
H_NEG_1E4 = 0x868E          # -0.0001 rounded to f16 (~-1.000166e-4)
H_2PM13 = 0x0800            # +2^-13
H_NEG_2PM13 = 0x8800        # -2^-13
H_NEG_2PM12_M = 0x8BFF      # -(2^-12 - 2^-23): just above -2^-12
H_NEG_2PM12 = 0x8C00        # -2^-12: exact tie between 0x3bff and 0x3c00
H_NEG_2PM12_P = 0x8C01      # -(2^-12 + 2^-22): strictly closer to 0x3bff
H_NEG_3P2M13 = 0x8E00       # -3 * 2^-13: strictly closer to 0x3bff
H_NEG_2PM11 = 0x9000        # -2^-11: 1 - 2^-11 = 0x3bff exact
H_NEG_3P2M12 = 0x9200       # -3 * 2^-12: exact tie between 0x3bfe (even) and 0x3bff (odd) -> 0x3bfe
H_NEG_2PM10 = 0x9400        # -2^-10: 1 - 2^-10 = 0x3bfe exact
H_0P125 = 0x3000            # 0.125
H_0P25 = 0x3400             # 0.25
H_NEG_0P25 = 0xB400         # -0.25
H_0P375 = 0x3600            # 0.375
H_0P5 = 0x3800              # 0.5
H_NEG_0P5 = 0xB800          # -0.5
H_0P75 = 0x3A00             # 0.75
H_NEG_0P75 = 0xBA00         # -0.75
H_NEAR_ONE = 0x3BFF         # 1 - 2^-11 (largest f16 < 1.0)
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
H_3P75 = 0x4380             # 3.75
H_NEG_3P75 = 0xC380         # -3.75
H_1023P5 = 0x63FF           # 1023.5
H_NEG_1023P5 = 0xE3FF       # -1023.5
H_1024 = 0x6400             # 1024.0
H_NEG_1024 = 0xE400         # -1024.0
H_MAX = 0x7BFF              # 65504.0
H_NEG_MAX = 0xFBFF          # -65504.0
H_POS_INF = 0x7C00          # +inf
H_NEG_INF = 0xFC00          # -inf
H_QNAN = 0x7E01             # +qNaN
H_NEG_QNAN = 0xFE01         # -qNaN
H_SNAN = 0x7C01             # +sNaN
H_NEG_SNAN = 0xFC01         # -sNaN


# -----------------------------------------------------------------------------
# Independent Fraction-Based IEEE-754 Binary16 Oracle
# -----------------------------------------------------------------------------

def h16_is_finite(h: int) -> bool:
    return ((h >> 10) & 0x1F) != 0x1F


def h16_is_nan(h: int) -> bool:
    return (h & 0x7FFF) > 0x7C00


def h16_is_inf(h: int) -> bool:
    return (h & 0x7FFF) == 0x7C00


def h16_to_fraction(h: int) -> Fraction:
    """Decode finite IEEE-754 binary16 bit pattern ``h`` into an exact Fraction."""
    h &= 0xFFFF
    exp = (h >> 10) & 0x1F
    if exp == 0x1F:
        raise ValueError(f"non-finite binary16 {h:#06x}")
    sign = -1 if (h >> 15) else 1
    mant = h & 0x3FF
    if exp == 0:
        return Fraction(0) if mant == 0 else Fraction(sign * mant, 1 << 24)
    return Fraction(sign * ((1 << 10) | mant)) * (Fraction(2) ** (exp - 25))


def fraction_to_h16(q: Fraction) -> int:
    """Round an exact Fraction to IEEE-754 binary16 (nearest, ties to even)."""
    if q == 0:
        return H_POS_ZERO
    sgn = 0x8000 if q < 0 else 0
    a = abs(q)
    e = max(a.numerator.bit_length() - a.denominator.bit_length(), -14)
    if e > -14 and Fraction(2) ** e > a:
        e -= 1
    m = a / (Fraction(2) ** (e - 10))
    n = m.numerator // m.denominator
    rem = m - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
        n += 1
    if e == -14 and n < 1024:
        return sgn | n
    if n >= 2048:
        n >>= 1
        e += 1
    if e > 15:
        return sgn | H_POS_INF
    return sgn | ((e + 15) << 10) | (n - 1024)


def exact_fract_fraction(h: int) -> Fraction | None:
    """Exact ``x - floor(x)`` as a Fraction in [0, 1), or None for +-inf / NaN."""
    if not h16_is_finite(h):
        return None
    q = h16_to_fraction(h)
    return q - math.floor(q)


def oracle_fract_f16(
    h: int,
    *,
    abs_: bool = False,
    neg: bool = False,
    mul: int = 1,
    div: int = 1,
    clamp: bool = False,
) -> int | None:
    """Independent binary16 reference for v_fract_f16, returning 16-bit int or None for NaN.

    Models RDNA2 source modifiers (abs before neg), exact ``x - floor(x)``,
    the hardware/translator 0x3c00 -> 0x3bff clamp when no output multiplier is
    present, and VOP3 output scaling (mul/div) + [0, 1] clamp before narrowing.
    """
    h &= 0xFFFF
    if abs_:
        h &= 0x7FFF
    if neg:
        h ^= 0x8000
    r = exact_fract_fraction(h)
    if r is None:
        return None
    if mul == 1 and div == 1:
        res = fraction_to_h16(r)
        return H_NEAR_ONE if res == H_1P0 else res
    scaled = r * Fraction(mul, div)
    if clamp:
        scaled = min(max(scaled, Fraction(0)), Fraction(1))
    return fraction_to_h16(scaled)


def merge_half(old: int, half: int, dst_hi: int) -> int:
    h = half & 0xFFFF
    return ((h << 16) | (old & 0xFFFF)) if dst_hi else ((old & 0xFFFF0000) | h)


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
    _PS_PATH = os.path.join(disasm_dir, "ps_fract_f16.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_fract_f16.s")
    _CS_FSSM = os.path.join(fssm_dir, "cs_fract_f16.fssm")
    with open(os.path.join(_TMP_DIR.name, "ps_fract_f16.dxbc"), "wb") as f:
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
        "; test_valu_fract_f16\n"
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
    asm = "; test_cs_valu_fract_f16\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
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
# Oracle Self-Tests
# -----------------------------------------------------------------------------

class OracleTests(unittest.TestCase):
    """Cross-checks the independent Fraction binary16 oracle against IEEE-754 encodings."""

    def test_exhaustive_binary16_decode_and_roundtrip(self):
        for h in range(0x10000):
            if not h16_is_finite(h):
                self.assertIsNone(exact_fract_fraction(h))
                self.assertIsNone(oracle_fract_f16(h))
                continue
            q = h16_to_fraction(h)
            expected_float = struct.unpack("<e", struct.pack("<H", h))[0]
            self.assertEqual(q, Fraction(expected_float))
            want = H_POS_ZERO if h == H_NEG_ZERO else h
            self.assertEqual(fraction_to_h16(q), want)

    def test_hand_worked_fract_f16_rows(self):
        rows = [
            (H_POS_ZERO, H_POS_ZERO),
            (H_NEG_ZERO, H_POS_ZERO),
            (H_0P25, H_0P25),
            (H_1P25, H_0P25),
            (H_3P75, H_0P75),
            (H_NEG_0P25, H_0P75),
            (H_NEG_1P25, H_0P75),
            (H_NEG_0P5, H_0P5),
            (H_NEG_0P75, H_0P25),
            (H_1P0, H_POS_ZERO),
            (H_NEG_1P0, H_POS_ZERO),
            (H_MAX, H_POS_ZERO),
            (H_NEG_MAX, H_POS_ZERO),
            (H_NEG_2PM10, 0x3BFE),
            (H_NEG_3P2M12, 0x3BFE),
            (H_NEG_2PM11, H_NEAR_ONE),
            (H_NEG_3P2M13, H_NEAR_ONE),
            (H_NEG_2PM12_P, H_NEAR_ONE),
            (H_NEG_2PM12, H_NEAR_ONE),
            (H_NEG_2PM12_M, H_NEAR_ONE),
            (H_NEG_2PM13, H_NEAR_ONE),
            (H_NEG_1E4, H_NEAR_ONE),
            (H_NEG_SUB_1E5, H_NEAR_ONE),
            (H_NEG_SUB_MAX, H_NEAR_ONE),
            (H_NEG_SUB_MIN, H_NEAR_ONE),
            (H_SUB_MIN, H_SUB_MIN),
            (H_SUB_1E5, H_SUB_1E5),
            (H_1E4, H_1E4),
        ]
        for h, want in rows:
            with self.subTest(h=hex(h)):
                self.assertEqual(oracle_fract_f16(h), want)
        # Verify decimal conversions match our binary16 constants.
        self.assertEqual(struct.unpack("<H", struct.pack("<e", -0.0001))[0], H_NEG_1E4)
        self.assertEqual(struct.unpack("<H", struct.pack("<e", -1e-5))[0], H_NEG_SUB_1E5)
        self.assertEqual(struct.unpack("<H", struct.pack("<e", 3.75))[0], H_3P75)


# -----------------------------------------------------------------------------
# End-to-End Execution Tests
# -----------------------------------------------------------------------------

class FractF16ExecutionTests(unittest.TestCase):
    def test_finite_values_across_e32_e64_and_bare(self):
        samples = [
            H_POS_ZERO, H_NEG_ZERO,
            H_SUB_MIN, H_NEG_SUB_MIN, H_SUB_1E5, H_NEG_SUB_1E5,
            H_SUB_MAX, H_NEG_SUB_MAX, H_NORM_MIN, H_NEG_NORM_MIN,
            H_1E4, H_NEG_1E4, H_2PM13, H_NEG_2PM13,
            H_NEG_2PM12_M, H_NEG_2PM12, H_NEG_2PM12_P, H_NEG_3P2M13,
            H_NEG_2PM11, H_NEG_3P2M12, H_NEG_2PM10,
            H_0P125, H_0P25, H_NEG_0P25, H_0P5, H_NEG_0P5, H_0P75, H_NEG_0P75,
            H_NEAR_ONE,
            H_1P0, H_NEG_1P0, H_1P25, H_NEG_1P25, H_1P5, H_NEG_1P5, H_1P75, H_NEG_1P75,
            H_2P0, H_NEG_2P0, H_2P5, H_NEG_2P5, H_3P0, H_NEG_3P0, H_3P75, H_NEG_3P75,
            H_1023P5, H_NEG_1023P5, H_1024, H_NEG_1024, H_MAX, H_NEG_MAX,
        ]
        for suffix in ("_e32", "_e64", ""):
            for h in samples:
                v1 = 0x55550000 | h
                want = merge_half(OLD, oracle_fract_f16(h), 0)
                with self.subTest(form="v_fract_f16" + suffix, h=hex(h)):
                    self.assertEqual(run_op(f"v_fract_f16{suffix} v0, v1", v1), want)

    def test_tiny_negatives_and_subnormals_near_one_boundary(self):
        # Crosses the exact binary16 rounding transition: -2^-10 and -3*2^-12 give 0x3bfe;
        # -2^-11 .. -2^-24 (including -0.0001 and -1e-5) give 0x3bff (never 0x3c00 = 1.0).
        cases = [
            (H_NEG_2PM10, 0x3BFE),
            (H_NEG_3P2M12, 0x3BFE),
            (H_NEG_2PM11, H_NEAR_ONE),
            (H_NEG_3P2M13, H_NEAR_ONE),
            (H_NEG_2PM12_P, H_NEAR_ONE),
            (H_NEG_2PM12, H_NEAR_ONE),
            (H_NEG_2PM12_M, H_NEAR_ONE),
            (H_NEG_2PM13, H_NEAR_ONE),
            (H_NEG_1E4, H_NEAR_ONE),
            (H_NEG_SUB_1E5, H_NEAR_ONE),
            (H_NEG_SUB_MAX, H_NEAR_ONE),
            (H_NEG_SUB_MIN, H_NEAR_ONE),
        ]
        for h, want in cases:
            with self.subTest(h=hex(h)):
                self.assertEqual(oracle_fract_f16(h), want)
                self.assertEqual(run_op("v_fract_f16_e32 v0, v1", 0xCAFE0000 | h), merge_half(OLD, want, 0))
                self.assertEqual(run_op("v_fract_f16_e64 v0, v1 op_sel:[0,1]", 0xCAFE0000 | h),
                                 merge_half(OLD, want, 1))

    def test_exact_integers_and_signed_zero_give_positive_zero(self):
        # GLSL.std.450 Fract(-0.0) in spirv_mini_interp returns +0.0 (copysign == +1), matching hardware.
        self.assertEqual(si._fract(-0.0), 0.0)
        self.assertEqual(math.copysign(1.0, si._fract(-0.0)), 1.0)
        for h in (H_POS_ZERO, H_NEG_ZERO, H_1P0, H_NEG_1P0, H_2P0, H_NEG_2P0,
                  H_3P0, H_NEG_3P0, H_1024, H_NEG_1024, H_MAX, H_NEG_MAX):
            with self.subTest(h=hex(h)):
                got = run_op("v_fract_f16_e32 v0, v1", 0xABCD0000 | h)
                self.assertEqual(got, merge_half(OLD, H_POS_ZERO, 0))
                self.assertNotEqual(got & 0xFFFF, H_NEG_ZERO)
                got_hi = run_op("v_fract_f16_e64 v0, v1 op_sel:[1,1]", (h << 16) | 0x1111)
                self.assertEqual(got_hi, merge_half(OLD, H_POS_ZERO, 1))

    def test_runtime_memory_inputs_via_s_buffer_load(self):
        # Loaded at runtime through s_buffer_load_dwordx4 so constant folding cannot bypass Fract.
        lanes_lo = [H_NEG_1E4, H_NEG_SUB_1E5, H_NEG_ZERO, H_3P75]
        lanes_hi = [H_NEG_MAX, H_NEG_0P25, H_1P25, H_POS_INF]
        packed = [(hi << 16) | lo for lo, hi in zip(lanes_lo, lanes_hi)]
        mem = {0x1000 + 4 * i: w for i, w in enumerate(packed)}
        for idx, (lo, hi) in enumerate(zip(lanes_lo, lanes_hi)):
            reg = f"v{idx + 1}"
            lines_head = [
                "s_buffer_load_dwordx4 s[4:7], s[0:3], 0",
                "s_waitcnt lgkmcnt(0)",
                "v_mov_b32_e32 v1, s4",
                "v_mov_b32_e32 v2, s5",
                "v_mov_b32_e32 v3, s6",
                "v_mov_b32_e32 v4, s7",
                f"v_mov_b32_e32 v0, {OLD:#x}",
            ]
            got_lo = run_ps(lines_head + [f"v_fract_f16_e32 v0, {reg}"], push_constants=PUSH_ADDR, memory=mem)
            self.assertEqual(got_lo, merge_half(OLD, oracle_fract_f16(lo), 0))
            got_hi = run_ps(lines_head + [f"v_fract_f16_e64 v0, {reg} op_sel:[1,1]"],
                            push_constants=PUSH_ADDR, memory=mem)
            want_hi = oracle_fract_f16(hi)
            if want_hi is None:
                self.assertEqual(got_hi & 0xFFFF, OLD & 0xFFFF)
                self.assertTrue(h16_is_nan(got_hi >> 16), hex(got_hi))
            else:
                self.assertEqual(got_hi, merge_half(OLD, want_hi, 1))

    def test_inf_and_nan_produce_nan_and_preserve_untouched_half(self):
        # Verify spirv_mini_interp Fract on +-inf/NaN and translator's OpFOrdEqual guard.
        self.assertTrue(math.isnan(si._fract(float("inf"))))
        self.assertTrue(math.isnan(si._fract(float("-inf"))))
        self.assertTrue(math.isnan(si._fract(float("nan"))))

        for h in (H_POS_INF, H_NEG_INF, H_QNAN, H_NEG_QNAN, H_SNAN, H_NEG_SNAN):
            for suffix in ("_e32", "_e64", ""):
                with self.subTest(h=hex(h), form=suffix or "bare"):
                    got = run_op(f"v_fract_f16{suffix} v0, v1", 0xCAFE0000 | h)
                    self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)
                    self.assertTrue(h16_is_nan(got & 0xFFFF), f"expected NaN in low half, got {got:#010x}")
            for src_hi, dst_hi in itertools.product((0, 1), repeat=2):
                v1 = ((h << 16) | H_0P25) if src_hi else ((H_0P25 << 16) | h)
                with self.subTest(h=hex(h), src_hi=src_hi, dst_hi=dst_hi):
                    got = run_op(f"v_fract_f16_e64 v0, v1 op_sel:[{src_hi},{dst_hi}]", v1)
                    if dst_hi:
                        self.assertEqual(got & 0xFFFF, OLD & 0xFFFF)
                        self.assertTrue(h16_is_nan(got >> 16), hex(got))
                    else:
                        self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)
                        self.assertTrue(h16_is_nan(got & 0xFFFF), hex(got))

    def test_op_sel_all_four_combinations(self):
        pairs = [
            (H_NEG_1P25, H_0P25),      # hi: 0.75, lo: 0.25
            (H_NEG_SUB_1E5, H_3P75),   # hi: 0x3bff, lo: 0.75
        ]
        for hi, lo in pairs:
            v1 = (hi << 16) | lo
            for src_hi, dst_hi in itertools.product((0, 1), repeat=2):
                h_in = hi if src_hi else lo
                line = f"v_fract_f16_e64 v0, v1 op_sel:[{src_hi},{dst_hi}]"
                with self.subTest(line=line, v1=hex(v1)):
                    self.assertEqual(run_op(line, v1), merge_half(OLD, oracle_fract_f16(h_in), dst_hi))

    def test_vop3_source_modifiers(self):
        cases = [
            ("v_fract_f16_e64 v0, -v1 op_sel:[0,0]", (H_NEG_1P25 << 16) | H_0P25, H_0P25, False, True, 0),
            ("v_fract_f16_e64 v0, -v1 op_sel:[1,1]", (H_NEG_0P25 << 16) | H_1P25, H_NEG_0P25, False, True, 1),
            ("v_fract_f16_e64 v0, -v1 op_sel:[0,0]", (H_0P25 << 16) | H_2PM13, H_2PM13, False, True, 0),
            ("v_fract_f16_e64 v0, -v1 op_sel:[0,0]", H_POS_ZERO, H_POS_ZERO, False, True, 0),
            ("v_fract_f16_e64 v0, -v1 op_sel:[0,0]", H_1P0, H_1P0, False, True, 0),
            ("v_fract_f16_e64 v0, |v1| op_sel:[1,1]", (H_NEG_1P25 << 16) | H_0P25, H_NEG_1P25, True, False, 1),
            ("v_fract_f16_e64 v0, |v1| op_sel:[1,0]", (H_NEG_2PM13 << 16) | H_0P5, H_NEG_2PM13, True, False, 0),
            ("v_fract_f16_e64 v0, -|v1| op_sel:[1,0]", (H_NEG_1P25 << 16) | H_0P25, H_NEG_1P25, True, True, 0),
            ("v_fract_f16_e64 v0, -|v1| op_sel:[0,1]", (H_0P5 << 16) | H_3P75, H_3P75, True, True, 1),
            ("v_fract_f16_e64 v0, -|v1| op_sel:[0,0]", H_2PM13, H_2PM13, True, True, 0),
        ]
        for line, v1, h_in, abs_, neg, dst_hi in cases:
            with self.subTest(line=line, v1=hex(v1)):
                want = oracle_fract_f16(h_in, abs_=abs_, neg=neg)
                self.assertEqual(run_op(line, v1), merge_half(OLD, want, dst_hi))

        # Source modifiers on inf/NaN preserve NaN output and keep untouched half.
        for mod in ("-v1", "|v1|", "-|v1|"):
            for h in (H_POS_INF, H_NEG_INF, H_QNAN):
                with self.subTest(mod=mod, h=hex(h)):
                    got = run_op(f"v_fract_f16_e64 v0, {mod}", (H_0P25 << 16) | h)
                    self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)
                    self.assertTrue(h16_is_nan(got & 0xFFFF), hex(got))

    def test_vop3_output_modifiers(self):
        # (asm_suffix, input_h, dst_hi, kwargs_for_oracle)
        cases = [
            # clamp without omod keeps [0, 1) and preserves 0x3c00 -> 0x3bff
            ("clamp", H_0P75, 0, dict(clamp=True)),
            ("clamp", H_NEG_0P25, 0, dict(clamp=True)),
            ("clamp", H_NEG_2PM13, 0, dict(clamp=True)),
            ("clamp", H_NEG_ZERO, 0, dict(clamp=True)),
            # mul:2 / mul:4 / div:2 (true 1.0 stays 0x3c00 when omod is active)
            ("mul:2", H_0P5, 0, dict(mul=2)),
            ("mul:2", H_0P75, 0, dict(mul=2)),
            ("mul:2", H_NEG_0P25, 0, dict(mul=2)),
            ("mul:2", H_NEG_2PM13, 0, dict(mul=2)),
            ("mul:4", H_0P25, 0, dict(mul=4)),
            ("mul:4", H_0P5, 0, dict(mul=4)),
            ("mul:4", H_NEG_1P25, 0, dict(mul=4)),
            ("div:2", H_0P5, 0, dict(div=2)),
            ("div:2", H_NEG_1P25, 0, dict(div=2)),
            ("div:2", H_0P25, 0, dict(div=2)),
            ("div:2", H_NEG_2PM13, 0, dict(div=2)),
            # combined omod + clamp
            ("mul:2 clamp", H_0P25, 0, dict(mul=2, clamp=True)),
            ("mul:2 clamp", H_0P5, 0, dict(mul=2, clamp=True)),
            ("mul:2 clamp", H_0P75, 0, dict(mul=2, clamp=True)),
            ("mul:2 clamp", H_NEG_0P25, 0, dict(mul=2, clamp=True)),
            ("mul:4 clamp", H_0P5, 0, dict(mul=4, clamp=True)),
            ("op_sel:[1,1] mul:2 clamp", H_NEG_1P25, 1, dict(mul=2, clamp=True)),
            ("op_sel:[1,1] mul:4 clamp", H_0P125, 1, dict(mul=4, clamp=True)),
        ]
        for suffix, h_in, dst_hi, kw in cases:
            v1 = ((h_in << 16) | H_0P25) if "op_sel:[1,1]" in suffix else ((H_3P75 << 16) | h_in)
            line = f"v_fract_f16_e64 v0, v1 {suffix}"
            want = oracle_fract_f16(h_in, **kw)
            with self.subTest(line=line, h_in=hex(h_in)):
                self.assertEqual(run_op(line, v1), merge_half(OLD, want, dst_hi))

        # Verify explicit exact values for the critical omod transitions.
        self.assertEqual(run_op("v_fract_f16_e64 v0, v1 mul:2", H_0P5), merge_half(OLD, H_1P0, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, v1 mul:2 clamp", H_0P75), merge_half(OLD, H_1P0, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, v1 mul:4", H_0P5), merge_half(OLD, H_2P0, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, v1 div:2", H_0P5), merge_half(OLD, H_0P25, 0))

    def test_inline_f16_constants(self):
        for lit, h in (("0.5", H_0P5), ("-0.5", H_NEG_0P5), ("1.0", H_1P0), ("-1.0", H_NEG_1P0)):
            want = oracle_fract_f16(h)
            for form in ("_e32", "_e64", ""):
                with self.subTest(lit=lit, form=form or "bare"):
                    self.assertEqual(run_op(f"v_fract_f16{form} v0, {lit}"), merge_half(OLD, want, 0))
            # op_sel:[0,1] writes the inline constant's fract into high half; low half preserved.
            self.assertEqual(run_op(f"v_fract_f16_e64 v0, {lit} op_sel:[0,1]"), merge_half(OLD, want, 1))
            # Inline float constants place +0.0 in the high half (docs/design.md / src_h).
            self.assertEqual(run_op(f"v_fract_f16_e64 v0, {lit} op_sel:[1,0]"), merge_half(OLD, H_POS_ZERO, 0))

        self.assertEqual(run_op("v_fract_f16_e64 v0, -0.5 mul:2"), merge_half(OLD, H_1P0, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, -0.5 div:2"), merge_half(OLD, H_0P25, 0))

    def test_sgpr_sources_and_destination_aliasing(self):
        s4 = (H_NEG_1P25 << 16) | H_0P25
        self.assertEqual(run_op("v_fract_f16_e32 v0, s4", s4=s4), merge_half(OLD, H_0P25, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, s4 op_sel:[0,0]", s4=s4), merge_half(OLD, H_0P25, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, s4 op_sel:[1,0]", s4=s4), merge_half(OLD, H_0P75, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, s4 op_sel:[0,1]", s4=s4), merge_half(OLD, H_0P25, 1))
        self.assertEqual(run_op("v_fract_f16_e64 v0, s4 op_sel:[1,1]", s4=s4), merge_half(OLD, H_0P75, 1))
        self.assertEqual(run_op("v_fract_f16_e64 v0, -s4 op_sel:[1,0]", s4=s4), merge_half(OLD, H_0P25, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, |s4| op_sel:[1,1]", s4=s4), merge_half(OLD, H_0P25, 1))

        # Destination aliasing: dst == src (v0, v0) keeps the untouched half of v0.
        v0_init = (H_3P75 << 16) | H_NEG_0P25
        self.assertEqual(
            run_ps([f"v_mov_b32 v0, {v0_init:#x}", "v_fract_f16_e32 v0, v0"]),
            (H_3P75 << 16) | H_0P75,
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v0, {v0_init:#x}", "v_fract_f16 v0, v0"]),
            (H_3P75 << 16) | H_0P75,
        )
        for src_hi, dst_hi in itertools.product((0, 1), repeat=2):
            h_in = H_3P75 if src_hi else H_NEG_0P25
            want_h = oracle_fract_f16(h_in)
            with self.subTest(alias_op_sel=(src_hi, dst_hi)):
                self.assertEqual(
                    run_ps([
                        f"v_mov_b32 v0, {v0_init:#x}",
                        f"v_fract_f16_e64 v0, v0 op_sel:[{src_hi},{dst_hi}]",
                    ]),
                    merge_half(v0_init, want_h, dst_hi),
                )

    def test_exec_masking_single_lane_and_workgroup(self):
        # Single-lane pixel shader: inactive lane preserves pre-existing v0; re-enabling executes.
        self.assertEqual(run_ps([
            f"v_mov_b32 v1, {H_1P5:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "s_mov_b32 exec_lo, 0",
            "v_fract_f16_e32 v0, v1",
            "v_fract_f16_e64 v0, v1 op_sel:[0,1]",
            "s_mov_b32 exec_lo, 1",
        ]), OLD)

        self.assertEqual(run_ps([
            f"v_mov_b32 v1, {(H_NEG_1P25 << 16) | H_2P5:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "s_mov_b32 exec_lo, 0",
            "v_fract_f16_e64 v0, v1 op_sel:[0,0]",
            "v_fract_f16_e64 v0, v1 op_sel:[1,1]",
            "s_mov_b32 exec_lo, 1",
            "v_fract_f16_e64 v0, v1 op_sel:[1,0]",
        ]), merge_half(OLD, H_0P75, 0))

        # Multi-lane 8-thread compute shader: even lanes active, odd lanes masked off keep OLD.
        lane_inputs = [
            (H_NEG_1P25 << 16) | H_0P25,
            (H_3P75 << 16) | H_NEG_0P25,
            (H_NEG_SUB_1E5 << 16) | H_1P25,
            (H_2P5 << 16) | H_NEG_1E4,
            (H_NEG_ZERO << 16) | H_NEG_0P5,
            (H_1P5 << 16) | H_NEG_MAX,
            (H_NEG_MAX << 16) | H_NEG_2PM12,
            (H_0P75 << 16) | H_3P75,
        ]
        lane_olds = [0xAA00BB00 + 0x00110001 * i for i in range(8)]
        for dst_hi, op_line in (
            (0, "v_fract_f16_e32 v2, v1"),
            (1, "v_fract_f16_e64 v2, v1 op_sel:[1,1]"),
        ):
            mem = _Heap(
                {0x100 + 4 * i: w for i, w in enumerate(lane_inputs)}
                | {0x200 + 4 * i: w for i, w in enumerate(lane_olds)}
            )
            run_cs([
                "v_lshlrev_b32_e32 v5, 2, v0",
                "v_add_nc_u32_e32 v6, 0x100, v5",
                "v_add_nc_u32_e32 v7, 0x200, v5",
                "buffer_load_dword v1, v6, s[0:3], 0 offen",
                "buffer_load_dword v2, v7, s[0:3], 0 offen",
                "v_and_b32_e32 v3, 1, v0",
                "v_cmp_eq_u32_e32 vcc_lo, 0, v3",
                "s_and_saveexec_b32 s6, vcc_lo",
                op_line,
                "s_mov_b32 exec_lo, s6",
                "buffer_store_dword v2, v5, s[0:3], 0 offen",
            ], group_size=8, memory=mem)

            got = [mem.get(4 * i) for i in range(8)]
            want = []
            for lane, (inp, old_w) in enumerate(zip(lane_inputs, lane_olds)):
                if lane & 1:
                    want.append(old_w)
                else:
                    h_in = (inp >> 16) if dst_hi else (inp & 0xFFFF)
                    want.append(merge_half(old_w, oracle_fract_f16(h_in), dst_hi))
            with self.subTest(op_line=op_line):
                self.assertEqual(got, want)

    def test_negative_control_sabotaged_spirv(self):
        # Input -1.25: true fract = 0.75 (0x3a00); Trunc gives -1.0 (0xbc00); Floor gives -2.0 (0xc000).
        spv = translate_ps([
            f"v_mov_b32 v1, {H_NEG_1P25:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "v_fract_f16_e32 v0, v1",
        ])
        expected = merge_half(OLD, H_0P75, 0)
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], expected)

        for replacement, sabotaged_half in (("Trunc", H_NEG_1P0), ("Floor", H_NEG_2P0)):
            sabotaged, n = re.subn(r"\bFract\b", replacement, spv, count=1)
            self.assertEqual(n, 1)
            got = si.run_fragment(sabotaged)["%out_mrt0"][0]
            with self.subTest(replacement=replacement):
                self.assertNotEqual(got, expected)
                self.assertEqual(got, merge_half(OLD, sabotaged_half, 0))


if __name__ == "__main__":
    unittest.main()
