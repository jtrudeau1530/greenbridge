"""Semantic interpreter tests for RDNA2 integer-to-float conversions:
v_cvt_f32_i32, v_cvt_f32_u32, and v_cvt_f32_ubyte0..3.

Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs
it in tests/semantic/spirv_mini_interp.py and compares the exported binary32
word with a plain-Python IEEE-754 float32 reference. Negative controls
sabotage the translated SPIR-V text in memory to verify that signedness swaps
and wrong byte shifts are caught. Standard library only.
"""
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

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR = None
_SRC_PATH = None


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def ref_cvt_f32_i32(x: int) -> int:
    x &= MASK32
    signed = x - (1 << 32) if (x >> 31) else x
    return f32_bits(float(signed))


def ref_cvt_f32_u32(x: int) -> int:
    return f32_bits(float(x & MASK32))


def ref_cvt_f32_ubyte(x: int, byte_idx: int) -> int:
    return f32_bits(float(((x & MASK32) >> (8 * byte_idx)) & 0xFF))


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_int_to_float.s")
    with open(os.path.join(_TMP_DIR.name, "ps_int_to_float.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_lines(lines) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_int_to_float\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_lines(lines) -> int:
    spv = translate_lines(lines)
    return si.run_fragment(spv)["%out_mrt0"][0]


I32_CASES = [
    (0, 0x00000000),           # +0.0
    (1, 0x3F800000),           # +1.0
    (0xFFFFFFFF, 0xBF800000),  # -1 -> -1.0
    (0x7FFFFFFF, 0x4F000000),  # +2147483647 -> +2147483648.0 (rounded)
    (0x80000000, 0xCF000000),  # -2147483648 -> -2147483648.0 (exact)
    (0x80000001, 0xCF000000),  # -2147483647 -> -2147483648.0 (rounded)
    (0x01000001, 0x4B800000),  # 16777217 -> 16777216.0 (ties-to-even)
    (0x01000003, 0x4B800002),  # 16777219 -> 16777220.0 (ties-to-even)
    (0xFEFFFFFF, 0xCB800000),  # -16777217 -> -16777216.0 (ties-to-even)
    (0xFEFFFFFD, 0xCB800002),  # -16777219 -> -16777220.0 (ties-to-even)
]

U32_CASES = [
    (0, 0x00000000),           # 0.0
    (1, 0x3F800000),           # 1.0
    (0x7FFFFFFF, 0x4F000000),  # +2147483648.0
    (0x80000000, 0x4F000000),  # +2147483648.0 (must stay positive!)
    (0xFFFFFFFF, 0x4F800000),  # +4294967296.0
    (0x01000001, 0x4B800000),  # 16777217 -> 16777216.0
    (0x01000003, 0x4B800002),  # 16777219 -> 16777220.0
]

UBYTE_VALUES = [
    0x80402010,
    0xFF7F0100,
    0x00000000,
    0xFFFFFFFF,
    0xDEADBEEF,
    0xFFFFFF05,
]


class IntToFloatTests(unittest.TestCase):
    def test_reference_constants(self):
        for x, want in I32_CASES:
            with self.subTest(op="i32", x=hex(x)):
                self.assertEqual(ref_cvt_f32_i32(x), want)
        for x, want in U32_CASES:
            with self.subTest(op="u32", x=hex(x)):
                self.assertEqual(ref_cvt_f32_u32(x), want)
        self.assertEqual(ref_cvt_f32_ubyte(0x80402010, 0), f32_bits(16.0))
        self.assertEqual(ref_cvt_f32_ubyte(0x80402010, 1), f32_bits(32.0))
        self.assertEqual(ref_cvt_f32_ubyte(0x80402010, 2), f32_bits(64.0))
        self.assertEqual(ref_cvt_f32_ubyte(0x80402010, 3), f32_bits(128.0))

    def test_v_cvt_f32_i32(self):
        for x, want in I32_CASES:
            with self.subTest(x=hex(x)):
                got = run_lines([f"v_mov_b32 v2, {x:#x}", "v_cvt_f32_i32_e32 v0, v2", "v_mov_b32 v1, 0"])
                self.assertEqual(got, want)
                self.assertEqual(got, ref_cvt_f32_i32(x))

    def test_v_cvt_f32_u32(self):
        for x, want in U32_CASES:
            with self.subTest(x=hex(x)):
                got = run_lines([f"v_mov_b32 v2, {x:#x}", "v_cvt_f32_u32_e32 v0, v2", "v_mov_b32 v1, 0"])
                self.assertEqual(got, want)
                self.assertEqual(got, ref_cvt_f32_u32(x))

    def test_v_cvt_f32_ubyte0_to_3(self):
        for idx in range(4):
            op = f"v_cvt_f32_ubyte{idx}_e32"
            for x in UBYTE_VALUES:
                with self.subTest(op=op, x=hex(x)):
                    want = ref_cvt_f32_ubyte(x, idx)
                    got = run_lines([f"v_mov_b32 v2, {x:#x}", f"{op} v0, v2", "v_mov_b32 v1, 0"])
                    self.assertEqual(got, want)

    def test_negative_controls(self):
        # 1) Signedness swap on v_cvt_f32_i32 (-1 must not become +4294967295.0)
        spv_i32 = translate_lines(["v_mov_b32 v2, 0xffffffff", "v_cvt_f32_i32_e32 v0, v2", "v_mov_b32 v1, 0"])
        self.assertEqual(si.run_fragment(spv_i32)["%out_mrt0"][0], ref_cvt_f32_i32(0xFFFFFFFF))
        self.assertIn("OpConvertSToF", spv_i32)
        bad_i32 = spv_i32.replace("OpConvertSToF", "OpConvertUToF")
        self.assertNotEqual(si.run_fragment(bad_i32)["%out_mrt0"][0], ref_cvt_f32_i32(0xFFFFFFFF))

        # 2) Signedness swap on v_cvt_f32_u32 (0x80000000 must not become -2147483648.0)
        spv_u32 = translate_lines(["v_mov_b32 v2, 0x80000000", "v_cvt_f32_u32_e32 v0, v2", "v_mov_b32 v1, 0"])
        self.assertEqual(si.run_fragment(spv_u32)["%out_mrt0"][0], ref_cvt_f32_u32(0x80000000))
        self.assertIn("OpConvertUToF", spv_u32)
        bad_u32 = spv_u32.replace("OpConvertUToF", "OpConvertSToF")
        self.assertNotEqual(si.run_fragment(bad_u32)["%out_mrt0"][0], ref_cvt_f32_u32(0x80000000))

        # 3) Wrong byte shift on v_cvt_f32_ubyte1 (shifting by 16 instead of 8 picks 0x40 instead of 0x20)
        spv_ub1 = translate_lines(["v_mov_b32 v2, 0x80402010", "v_cvt_f32_ubyte1_e32 v0, v2", "v_mov_b32 v1, 0"])
        self.assertEqual(si.run_fragment(spv_ub1)["%out_mrt0"][0], ref_cvt_f32_ubyte(0x80402010, 1))
        self.assertIn("= OpConstant %uint 8\n", spv_ub1)
        bad_ub1 = spv_ub1.replace("= OpConstant %uint 8\n", "= OpConstant %uint 16\n")
        self.assertNotEqual(si.run_fragment(bad_ub1)["%out_mrt0"][0], ref_cvt_f32_ubyte(0x80402010, 1))


if __name__ == "__main__":
    unittest.main()
