"""Semantic interpreter tests for RDNA2 v_cvt_flr_i32_f32 and v_cvt_rpi_i32_f32.

RDNA ISA pseudocode: v_cvt_flr_i32_f32 is f32_to_i32(floor(S0)), v_cvt_rpi_i32_f32 is f32_to_i32(floor(S0 + 0.5)),
with the + 0.5 rounded in binary32 like any f32 add. f32_to_i32 saturates as v_cvt_i32_f32 does (#300): at or past
+-2^31, and +-inf, it gives INT_MAX / INT_MIN; NaN gives 0. Each case translates a synthetic pixel shader with
tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the result with a plain-Python
reference. Denormal inputs are left out (the float mode that would flush them is not modelled). The result leaves
through exp as two 16-bit halves ORed into 0x40000000, so no export channel is a NaN or denormal pattern that a
float round trip could change.
"""
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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def u2f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & M32))[0]


def f32(x: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def f32_to_i32(x: float) -> int:
    if math.isnan(x):
        return 0
    if x >= 2.0 ** 31:
        return 0x7FFFFFFF
    if x < -(2.0 ** 31):
        return 0x80000000
    return int(x) & M32


def ref(op: str, u: int) -> int:
    x = u2f(u)
    if op == "v_cvt_rpi_i32_f32":
        x = f32(x + 0.5)                    # exact in double, then one rounding to binary32
    return f32_to_i32(math.floor(x) if math.isfinite(x) else x)


OPS = ("v_cvt_flr_i32_f32", "v_cvt_rpi_i32_f32")
VALUES = [0x00000000, 0x80000000, 0x3F000000, 0xBF000000, 0x3EFFFFFF, 0xBEFFFFFF, 0x3FC00000, 0xBFC00000,
          0x40200000, 0xC0200000, 0x3F7FFFFF, 0x4B000001, 0xCB000001, 0x4EFFFFFF, 0xCEFFFFFF, 0x4F000000,
          0xCF000000, 0xCF000001, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cvt_flr_rpi.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cvt_flr_rpi.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> int:
    """Run the lines, return v0 (exported as two 16-bit halves, see the module docstring)."""
    if _TMP_DIR is None:
        setUpModule()
    lines = list(lines) + ["v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0",
                           "v_or_b32 v5, 0x40000000, v5", "v_or_b32 v6, 0x40000000, v6"]
    asm = "; test_cvt_flr_rpi\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16


def run_op(op: str, u: int) -> int:
    return run_lines([f"v_mov_b32 v1, {u:#x}", f"{op} v0, v1"])


class CvtFlrRpiTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref("v_cvt_flr_i32_f32", 0xC0200000), (-3) & M32)       # floor(-2.5)
        self.assertEqual(ref("v_cvt_rpi_i32_f32", 0xC0200000), (-2) & M32)       # floor(-2.5 + 0.5)
        self.assertEqual(ref("v_cvt_rpi_i32_f32", 0x40200000), 3)                # 2.5 rounds up
        self.assertEqual(ref("v_cvt_rpi_i32_f32", 0x3EFFFFFF), 1)                # 0.49999997 + 0.5 rounds to 1.0
        self.assertEqual(ref("v_cvt_flr_i32_f32", 0xCF000001), 0x80000000)       # below -2^31: saturated
        self.assertEqual(ref("v_cvt_flr_i32_f32", 0x7FC00000), 0)                # NaN

    def test_edge_values(self):
        for op in OPS:
            for u in VALUES:
                with self.subTest(op=op, x=hex(u)):
                    self.assertEqual(run_op(op, u), ref(op, u))

    def test_seeded(self):
        rng = random.Random(0xF1A)
        for _ in range(40):
            op = rng.choice(OPS)
            u = rng.getrandbits(1) << 31 | rng.randrange(120, 160) << 23 | rng.getrandbits(23)   # |x| ~ 2^-7 .. 2^32
            with self.subTest(op=op, x=hex(u)):
                self.assertEqual(run_op(op, u), ref(op, u))

    def test_source_modifiers(self):
        self.assertEqual(run_lines(["v_mov_b32 v1, 0x40200000", "v_cvt_flr_i32_f32_e64 v0, -v1"]), (-3) & M32)
        self.assertEqual(run_lines(["v_mov_b32 v1, 0xc0200000", "v_cvt_rpi_i32_f32_e64 v0, |v1|"]), 3)


if __name__ == "__main__":
    unittest.main()
