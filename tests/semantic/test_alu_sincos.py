"""Semantic interpreter tests for RDNA2 v_sin_f32 / v_cos_f32 / v_sin_f16 / v_cos_f16 (SHADER-88).

The operand is an angle in TURNS (valid over +-256 turns). Multiplying by 2*pi before reducing rounds the fractional
turn away in binary32 (up to ~1e-4 rad near +-255 turns), so the translator reduces in turns first. Each case
translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and
compares the exported value with math.sin / math.cos(2*pi*x) computed in double (a reference, not a model of the
translator). Tolerances are absolute and explicit: 1e-6 for f32 (a few ulp of a value near 1), 1e-3 for f16 (one
f16 ulp near 1 is ~4.9e-4 and the result is rounded to f16).
"""
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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xAAAA5555
TOL32 = 1e-6
TOL16 = 1e-3
INPUTS = [0.25, 0.5, 1.0, 100.3, 254.25, 254.51, 255.9, 0.1, 0.0, -0.0, -0.25, -100.3, -254.51, -255.5]

_TMP_DIR = None
_SRC_PATH = None


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def h2f(h):
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def ref(op, x):
    return math.sin(2 * math.pi * x) if "sin" in op else math.cos(2 * math.pi * x)


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_sincos.s")
    with open(os.path.join(_TMP_DIR.name, "ps_sincos.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run(line, v1, old=0):
    if _TMP_DIR is None:
        setUpModule()
    lines = [f"v_mov_b32 v1, {v1:#x}", f"v_mov_b32 v0, {old:#x}", line]
    asm = "; test_sincos\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class SinCosF32Tests(unittest.TestCase):
    def test_values_against_double_reference(self):
        for op in ("v_sin_f32", "v_cos_f32"):
            for x in INPUTS:
                x32 = u2f(f2u(x))                                  # the binary32 operand the shader sees
                with self.subTest(op=op, x=x):
                    got = u2f(run(f"{op}_e32 v0, v1", f2u(x)))
                    self.assertLessEqual(abs(got - ref(op, x32)), TOL32)

    def test_exact_quarter_turns(self):
        self.assertLessEqual(abs(u2f(run("v_sin_f32_e32 v0, v1", f2u(0.25))) - 1.0), TOL32)
        self.assertLessEqual(abs(u2f(run("v_cos_f32_e32 v0, v1", f2u(0.5))) + 1.0), TOL32)
        self.assertLessEqual(abs(u2f(run("v_sin_f32_e32 v0, v1", f2u(254.25))) - 1.0), TOL32)
        self.assertLessEqual(abs(u2f(run("v_cos_f32_e32 v0, v1", f2u(-254.5))) + 1.0), TOL32)

    def test_worst_case_of_the_issue(self):
        # x = 254.51: x * 2*pi first was off by 1.05e-4; reducing in turns is within 1e-6.
        x32 = u2f(f2u(254.51))
        got = u2f(run("v_sin_f32_e32 v0, v1", f2u(254.51)))
        self.assertLessEqual(abs(got - math.sin(2 * math.pi * x32)), TOL32)

    def test_signed_zero_and_special_values(self):
        self.assertEqual(run("v_sin_f32_e32 v0, v1", 0x00000000), 0x00000000)
        self.assertEqual(run("v_sin_f32_e32 v0, v1", 0x80000000), 0x80000000)      # sin(-0) = -0
        self.assertEqual(u2f(run("v_cos_f32_e32 v0, v1", 0x80000000)), 1.0)
        for bits in (0x7F800000, 0xFF800000, 0x7FC00000):
            for op in ("v_sin_f32", "v_cos_f32"):
                with self.subTest(op=op, x=hex(bits)):
                    self.assertTrue(math.isnan(u2f(run(f"{op}_e32 v0, v1", bits))))

    def test_modifiers(self):
        got = u2f(run("v_sin_f32_e64 v0, -v1", f2u(100.3)))               # neg: sin(-x)
        self.assertLessEqual(abs(got - math.sin(-2 * math.pi * u2f(f2u(100.3)))), TOL32)
        got = u2f(run("v_cos_f32_e64 v0, |v1|", f2u(-254.51)))            # abs
        self.assertLessEqual(abs(got - math.cos(2 * math.pi * u2f(f2u(254.51)))), TOL32)
        got = u2f(run("v_sin_f32_e64 v0, v1 clamp", f2u(-0.25)))          # clamp: sin = -1 -> 0
        self.assertEqual(got, 0.0)
        got = u2f(run("v_sin_f32_e64 v0, v1 mul:2", f2u(100.3)))          # omod
        self.assertLessEqual(abs(got - 2 * math.sin(2 * math.pi * u2f(f2u(100.3)))), 2 * TOL32)

    def test_whole_turns_give_plus_zero_for_either_sign(self):
        """Measured on an RX 6800 (gfx1030, #616): v_sin_f32 of a whole number of turns, negative or positive, is
        +0 (only sin(-0) is -0) and v_cos_f32 of it is 1."""
        for x in (-1.0, -2.0, -100.0, 1.0, 2.0, 100.0):
            with self.subTest(x=x):
                self.assertEqual(run("v_sin_f32_e32 v0, v1", f2u(x)), 0x00000000)
                self.assertEqual(run("v_cos_f32_e32 v0, v1", f2u(x)), f2u(1.0))

    def test_tiny_arguments_keep_their_sign_and_size(self):
        """A reducer built on floor() would turn x = -2**-26 into 1.0 - 2**-26 = 1.0 in binary32 and return +0;
        x - RoundEven(x) keeps the small centred argument. The result is below TOL32 either way, so the check is on
        the sign and on a much tighter absolute tolerance (suggested by bobcatchris on #616)."""
        for x in (-2.0 ** -26, 2.0 ** -26, -2.0 ** -20, 2.0 ** -20):
            with self.subTest(x=x):
                got = u2f(run("v_sin_f32_e32 v0, v1", f2u(x)))
                self.assertEqual(math.copysign(1.0, got), math.copysign(1.0, x))
                self.assertNotEqual(got, 0.0)
                self.assertLessEqual(abs(got - math.sin(2 * math.pi * x)), 2e-9)
                self.assertLessEqual(abs(u2f(run("v_cos_f32_e32 v0, v1", f2u(x))) - 1.0), 1e-6)


class SinCosF16Tests(unittest.TestCase):
    def test_values_keep_the_high_half(self):
        for op in ("v_sin_f16", "v_cos_f16"):
            for x in (0.25, 0.5, 100.25, 254.5, -254.25, 0.3):
                h = f2h(x)
                with self.subTest(op=op, x=x):
                    got = run(f"{op}_e32 v0, v1", 0x12340000 | h, OLD)
                    self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)
                    self.assertLessEqual(abs(h2f(got) - ref(op, h2f(h))), TOL16)

    def test_negative_zero_keeps_its_sign(self):
        """f16 -0.0 is 0x8000: sin(-0) stays -0 (the sign bit survives the turns reduction) and cos(-0) is 1;
        the high half of the destination is kept in both."""
        got = run("v_sin_f16_e32 v0, v1", 0x12340000 | 0x8000, OLD)
        self.assertEqual(got, (OLD & 0xFFFF0000) | 0x8000)
        got = run("v_cos_f16_e32 v0, v1", 0x12340000 | 0x8000, OLD)
        self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)
        self.assertEqual(h2f(got), 1.0)

    def test_large_angle_near_the_256_turn_edge(self):
        """Where x * 2*pi first would lose the fractional turn: 254.25 and 255.0 are exact in f16, and the
        reference is computed in double."""
        for op, x in (("v_sin_f16", 254.25), ("v_sin_f16", -254.25), ("v_cos_f16", 254.5), ("v_cos_f16", 255.0),
                      ("v_sin_f16", 255.5), ("v_cos_f16", -255.5)):
            h = f2h(x)
            self.assertEqual(h2f(h), x)                      # the input is exactly representable
            with self.subTest(op=op, x=x):
                got = run(f"{op}_e32 v0, v1", 0x12340000 | h, OLD)
                self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)
                self.assertLessEqual(abs(h2f(got) - ref(op, x)), TOL16)


if __name__ == "__main__":
    unittest.main()
