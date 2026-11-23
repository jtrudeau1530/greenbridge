"""Value tests for f32 unary VALU ops (the F1 table) and byte-to-float converts no other test reaches:

  v_exp_f32        D.f = 2^S0.f                    (GLSL.std.450 Exp2)
  v_log_f32        D.f = log2(S0.f)                (Log2; input must be > 0)
  v_sqrt_f32       D.f = sqrt(S0.f)                (Sqrt; sqrt(-0.0) is -0.0)
  v_rsq_f32        D.f = 1/sqrt(S0.f)              (InverseSqrt; input must be > 0)
  v_floor_f32      D.f = floor(S0.f)               (Floor)
  v_ceil_f32       D.f = ceil(S0.f)                (Ceil)
  v_trunc_f32      D.f = trunc(S0.f)               (Trunc, toward zero)

v_rcp_f32 and v_fract_f32, the other two F1 entries, are already covered elsewhere
(test_valu_rcp_fdiv.py, test_float_nan_fract.py), as are the v_cvt_f32_ubyte0..3 converts
(test_int_to_float.py).

Each case is a synthetic hand-written pixel-shader listing (the source word loaded into a VGPR with
v_mov_b32, one instruction, exported to mrt0), translated by tools/rdna2spirv.py and executed by
spirv_mini_interp.py. Only inputs whose result is exactly representable are used: the interpreter
computes Exp2/Log2/Sqrt/InverseSqrt correctly rounded and Floor/Ceil/Trunc exactly (an integral
result keeps the operand's sign, so e.g. trunc(-0.5) is -0.0), and cases that GLSL.std.450 leaves
undefined (log2/rsq of x <= 0, sqrt of a negative) are avoided because it raises on them -- the
translator intentionally leaves those to the driver (docs/design.md, denormal-flush row). The
expected words are hand-computed constants; a plain-Python reference is checked against them so a
typo in either is caught. Standard library only.
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


def u2f(u):
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f2u(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def integral(x, fn):
    """As the interpreter models Floor/Ceil/Trunc: integral result keeps the operand's sign."""
    return f32(math.copysign(float(fn(x)), x))


REFERENCE = {
    "v_exp_f32": lambda u: f2u(f32(2.0 ** u2f(u))),
    "v_log_f32": lambda u: f2u(f32(math.log2(u2f(u)))),
    "v_sqrt_f32": lambda u: f2u(f32(math.sqrt(u2f(u)))),
    "v_rsq_f32": lambda u: f2u(f32(1.0 / math.sqrt(u2f(u)))),
    "v_floor_f32": lambda u: f2u(integral(u2f(u), math.floor)),
    "v_ceil_f32": lambda u: f2u(integral(u2f(u), math.ceil)),
    "v_trunc_f32": lambda u: f2u(integral(u2f(u), math.trunc)),
}

# (source word, expected word) -- f32 bit patterns; expected words computed by hand.
CASES = {
    "v_exp_f32": [
        (0x40800000, 0x41800000),  # 2^4 = 16.0
        (0xBF800000, 0x3F000000),  # 2^-1 = 0.5
        (0x00000000, 0x3F800000),  # 2^0 = 1.0
        (0xC0000000, 0x3E800000),  # 2^-2 = 0.25
        (0x40400000, 0x41000000),  # 2^3 = 8.0
        (0x42FE0000, 0x7F000000),  # 2^127: largest finite power
        (0xC2C80000, 0x0D800000),  # 2^-100, still a normal f32
        (0xC1400000, 0x39800000),  # 2^-12
    ],
    "v_log_f32": [
        (0x41000000, 0x40400000),  # log2(8) = 3
        (0x3E800000, 0xC0000000),  # log2(0.25) = -2
        (0x3F800000, 0x00000000),  # log2(1) = 0
        (0x40000000, 0x3F800000),  # log2(2) = 1
        (0x7F000000, 0x42FE0000),  # log2(2^127) = 127
        (0x00800000, 0xC2FC0000),  # log2(2^-126) = -126 (smallest normal input)
    ],
    "v_sqrt_f32": [
        (0x40800000, 0x40000000),  # sqrt(4) = 2
        (0x40100000, 0x3FC00000),  # sqrt(2.25) = 1.5
        (0x00000000, 0x00000000),  # sqrt(+0) = +0
        (0x80000000, 0x80000000),  # sqrt(-0) = -0
        (0x42C80000, 0x41200000),  # sqrt(100) = 10
        (0x3E800000, 0x3F000000),  # sqrt(0.25) = 0.5
        (0x41800000, 0x40800000),  # sqrt(16) = 4
    ],
    "v_rsq_f32": [
        (0x40800000, 0x3F000000),  # rsq(4) = 0.5
        (0x3E800000, 0x40000000),  # rsq(0.25) = 2
        (0x3F800000, 0x3F800000),  # rsq(1) = 1
        (0x41800000, 0x3E800000),  # rsq(16) = 0.25
        (0x42C80000, 0x3DCCCCCD),  # rsq(100) = 0.1
    ],
    "v_floor_f32": [
        (0x40600000, 0x40400000),  # floor(3.5) = 3
        (0xBFC00000, 0xC0000000),  # floor(-1.5) = -2
        (0xBF000000, 0xBF800000),  # floor(-0.5) = -1
        (0x3F000000, 0x00000000),  # floor(0.5) = +0
        (0x80000000, 0x80000000),  # floor(-0) = -0
        (0x40490FDB, 0x40400000),  # floor(pi) = 3
        (0xC0490FDB, 0xC0800000),  # floor(-pi) = -4
        (0x41200000, 0x41200000),  # floor(10) = 10
    ],
    "v_ceil_f32": [
        (0x40600000, 0x40800000),  # ceil(3.5) = 4
        (0xBFC00000, 0xBF800000),  # ceil(-1.5) = -1
        (0xBF000000, 0x80000000),  # ceil(-0.5) = -0.0 (integral result keeps the sign)
        (0x3F000000, 0x3F800000),  # ceil(0.5) = 1
        (0x40490FDB, 0x40800000),  # ceil(pi) = 4
        (0xC0490FDB, 0xC0400000),  # ceil(-pi) = -3
        (0x00000000, 0x00000000),  # ceil(+0) = +0
        (0x41200000, 0x41200000),  # ceil(10) = 10
    ],
    "v_trunc_f32": [
        (0x40600000, 0x40400000),  # trunc(3.5) = 3
        (0xBFC00000, 0xBF800000),  # trunc(-1.5) = -1
        (0xBF000000, 0x80000000),  # trunc(-0.5) = -0.0
        (0x3F000000, 0x00000000),  # trunc(0.5) = +0
        (0x40490FDB, 0x40400000),  # trunc(pi) = 3
        (0xC0490FDB, 0xC0400000),  # trunc(-pi) = -3
        (0x80000000, 0x80000000),  # trunc(-0) = -0
        (0xC0200000, 0xC0000000),  # trunc(-2.5) = -2
    ],
}

_TMP = None


def setUpModule():
    global _TMP
    _TMP = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(_TMP.name, "disasm"))
    with open(os.path.join(_TMP.name, "ps_f32_unary.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP
    if _TMP is not None:
        _TMP.cleanup()
        _TMP = None


def run_op(op, src):
    """Translate and run `op v0, v1` with v1 loaded by v_mov_b32; return the mrt0 word."""
    lines = ["; test_f32_unary"]
    lines.append(f"\tv_mov_b32 v1, {src & MASK32:#x}")
    lines.append(f"\t{op} v0, v1")
    lines += ["\texp mrt0 v0, v0, v0, v0 done vm", "\ts_endpgm", ""]
    path = os.path.join(_TMP.name, "disasm", "ps_f32_unary.s")
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
            for src, expected in cases:
                with self.subTest(op=op, src=hex(src)):
                    self.assertEqual(REFERENCE[op](src), expected)


class TranslatedValues(unittest.TestCase):
    def check(self, op):
        for src, expected in CASES[op]:
            with self.subTest(src=hex(src)):
                got = run_op(op, src)
                self.assertEqual(got, expected, f"{op} {src:#010x}: got {got:#010x}, "
                                                f"expected {expected:#010x}")

    def test_v_exp_f32(self):
        self.check("v_exp_f32")

    def test_v_log_f32(self):
        self.check("v_log_f32")

    def test_v_sqrt_f32(self):
        self.check("v_sqrt_f32")

    def test_v_rsq_f32(self):
        self.check("v_rsq_f32")

    def test_v_floor_f32(self):
        self.check("v_floor_f32")

    def test_v_ceil_f32(self):
        self.check("v_ceil_f32")

    def test_v_trunc_f32(self):
        self.check("v_trunc_f32")


if __name__ == "__main__":
    unittest.main()
