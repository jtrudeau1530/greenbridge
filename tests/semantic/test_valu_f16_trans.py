"""Value tests for the f16 transcendental VALU ops no other test reaches:

  v_exp_f16   D.f16 = 2^S0.f16            (GLSL.std.450 Exp2 on the widened operand)
  v_log_f16   D.f16 = log2(S0.f16)        (Log2; input must be > 0)
  v_sqrt_f16  D.f16 = sqrt(S0.f16)        (Sqrt)
  v_rsq_f16   D.f16 = 1/sqrt(S0.f16)      (InverseSqrt; input must be > 0)

v_rcp_f16 is covered elsewhere; docs/design.md's f16-transcendentals row records these four (with
v_sin/v_cos_f16, covered by test_value_semantics.py) correctly rounded on every non-NaN input,
measured over all 65,536 f16 values on an RX 6700 XT. Like the other f16 VOP1 ops, the default
op_sel reads and writes the low half of each register and keeps the other half; op_sel:[1,1]
reads and writes the high half. MODE rounding is ignored (nearest even always, SHADER-35).

Each case is a synthetic hand-written pixel-shader listing (source word and destination seed loaded
with v_mov_b32, one instruction, exported to mrt0), translated by tools/rdna2spirv.py and executed
by spirv_mini_interp.py. Only inputs with an exactly representable f16 result are used, and inputs
GLSL.std.450 leaves undefined (log2/rsq of x <= 0, sqrt of a negative) are avoided: the interpreter
raises on them and the translator leaves them to the driver. Expected words are hand-computed f16
constants; a plain-Python reference is checked against them. Standard library only.
"""
import math
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import STUB_DXBC, mrt0_words, rdna2spirv  # noqa: E402  (support puts tools/ on sys.path)

MASK32 = 0xFFFFFFFF


def h2f(h):
    """f16 bit pattern -> float."""
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(f):
    """float -> f16 bit pattern (round to nearest even, as the ops do)."""
    return struct.unpack("<H", struct.pack("<e", f))[0]


FN = {
    "v_exp_f16": lambda x: 2.0 ** x,
    "v_log_f16": math.log2,
    "v_sqrt_f16": math.sqrt,
    "v_rsq_f16": lambda x: 1.0 / math.sqrt(x),
}


def ref_lo(op, src, seed):
    """Low half of `src` through the op; the destination's high half is kept."""
    return (seed & 0xFFFF0000) | f2h(FN[op](h2f(src)))


def ref_hi(op, src, seed):
    """op_sel:[1,1]: high half of `src`, result to the high half; the low half is kept."""
    return (seed & 0xFFFF) | (f2h(FN[op](h2f(src >> 16))) << 16)


# (source word, destination seed, expected word). 4.0 = 0x4400, 2.0 = 0x4000, 1.0 = 0x3C00,
# 0.5 = 0x3800, 0.25 = 0x3400, 8.0 = 0x4800, 3.0 = 0x4200, -2.0 = 0xC000, 16.0 = 0x4C00 in f16.
CASES = {
    "v_exp_f16": [
        (0x00004000, 0xAAAA0000, 0xAAAA4400),  # 2^2 = 4.0, high half kept
        (0x0000C000, 0x12340000, 0x12343400),  # 2^-2 = 0.25
        (0x00004200, 0xCAFE0000, 0xCAFE4800),  # 2^3 = 8.0
        (0x0000BC00, 0x00000000, 0x00003800),  # 2^-1 = 0.5
        (0x00000000, 0xABCD0000, 0xABCD3C00),  # 2^0 = 1.0
    ],
    "v_log_f16": [
        (0x00004800, 0xAAAA0000, 0xAAAA4200),  # log2(8) = 3.0
        (0x00003400, 0x11110000, 0x1111C000),  # log2(0.25) = -2.0
        (0x00004000, 0x0000BEEF, 0x00003C00),  # log2(2) = 1.0
        (0x00003C00, 0x56780000, 0x56780000),  # log2(1) = +0
        (0x00004C00, 0x00000000, 0x00004400),  # log2(16) = 4.0
    ],
    "v_sqrt_f16": [
        (0x00004400, 0xAAAA0000, 0xAAAA4000),  # sqrt(4) = 2.0
        (0x00003400, 0xBBBB1234, 0xBBBB3800),  # sqrt(0.25) = 0.5
        (0x00003C00, 0x0000FFFF, 0x00003C00),  # sqrt(1) = 1.0
        (0x00004C00, 0xDEAD0000, 0xDEAD4400),  # sqrt(16) = 4.0
    ],
    "v_rsq_f16": [
        (0x00004400, 0xAAAA0000, 0xAAAA3800),  # rsq(4) = 0.5
        (0x00003400, 0x77770000, 0x77774000),  # rsq(0.25) = 2.0
        (0x00003C00, 0x0000FFFF, 0x00003C00),  # rsq(1) = 1.0
        (0x00004C00, 0x13570000, 0x13573400),  # rsq(16) = 0.25
    ],
}

# op_sel:[1,1]: the source's high half feeds the op, the result lands in the destination's high
# half and the low half is kept.
HI_CASES = {
    "v_exp_f16": [
        (0xC0000000, 0x00001234, 0x34001234),  # 2^-2 = 0.25 in the high half
        (0x40000000, 0x0000BEEF, 0x4400BEEF),  # 2^2 = 4.0
    ],
    "v_log_f16": [
        (0x48000000, 0x00005678, 0x42005678),  # log2(8) = 3.0
    ],
    "v_sqrt_f16": [
        (0x44000000, 0x0000BEEF, 0x4000BEEF),  # sqrt(4) = 2.0
    ],
    "v_rsq_f16": [
        (0x34000000, 0x0000ABCD, 0x4000ABCD),  # rsq(0.25) = 2.0
    ],
}

_TMP = None


def setUpModule():
    global _TMP
    _TMP = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(_TMP.name, "disasm"))
    with open(os.path.join(_TMP.name, "ps_f16_trans.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP
    if _TMP is not None:
        _TMP.cleanup()
        _TMP = None


def run_case(op, src, seed, opsel=False):
    """Translate and run `op v0, v1` (v1 = src word, v0 preloaded with seed); return the mrt0 word."""
    lines = ["; test_f16_trans"]
    lines.append(f"\tv_mov_b32 v1, {src & MASK32:#x}")
    lines.append(f"\tv_mov_b32 v0, {seed & MASK32:#x}")
    lines.append(f"\t{op} v0, v1" + (" op_sel:[1,1]" if opsel else ""))
    lines += ["\texp mrt0 v0, v0, v0, v0 done vm", "\ts_endpgm", ""]
    path = os.path.join(_TMP.name, "disasm", "ps_f16_trans.s")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(path)
    words = mrt0_words(spv)
    if len(set(words)) != 1:
        raise AssertionError(f"{op}: the four exported words differ: {[hex(w) for w in words]}")
    return words[0]


class ReferenceMatchesHandConstants(unittest.TestCase):
    def test_reference(self):
        for op, cases in CASES.items():
            for src, seed, expected in cases:
                with self.subTest(op=op, src=hex(src)):
                    self.assertEqual(ref_lo(op, src, seed), expected)
        for op, cases in HI_CASES.items():
            for src, seed, expected in cases:
                with self.subTest(op=op, src=hex(src), hi=1):
                    self.assertEqual(ref_hi(op, src, seed), expected)


class TranslatedValues(unittest.TestCase):
    def check(self, op):
        for src, seed, expected in CASES[op]:
            with self.subTest(src=hex(src), seed=hex(seed)):
                got = run_case(op, src, seed)
                self.assertEqual(got, expected, f"{op} {src:#010x}: got {got:#010x}, "
                                                f"expected {expected:#010x}")

    def check_hi(self, op):
        for src, seed, expected in HI_CASES[op]:
            with self.subTest(src=hex(src), seed=hex(seed)):
                got = run_case(op, src, seed, opsel=True)
                self.assertEqual(got, expected, f"{op} op_sel {src:#010x}: got {got:#010x}, "
                                                f"expected {expected:#010x}")

    def test_v_exp_f16(self):
        self.check("v_exp_f16")
        self.check_hi("v_exp_f16")

    def test_v_log_f16(self):
        self.check("v_log_f16")
        self.check_hi("v_log_f16")

    def test_v_sqrt_f16(self):
        self.check("v_sqrt_f16")
        self.check_hi("v_sqrt_f16")

    def test_v_rsq_f16(self):
        self.check("v_rsq_f16")
        self.check_hi("v_rsq_f16")


if __name__ == "__main__":
    unittest.main()
