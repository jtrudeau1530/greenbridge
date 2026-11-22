"""Semantic interpreter and spirv-val tests for RDNA2 v_div_fmas_f32 and v_div_fmas_f64.

v_div_fmas_f32 computes D = VCC[lane] ? (2^32 * fma(S0, S1, S2)) : fma(S0, S1, S2).
v_div_fmas_f64 computes D = VCC[lane] ? (2^64 * fma(S0, S1, S2)) : fma(S0, S1, S2).
Both accept VOP3 input (abs/neg) and output (mul:2, mul:4, div:2, clamp) modifiers.
"""
import math
import os
import re
import shutil
import struct
import subprocess
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


def f2u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def dwords(x: float):
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    return bits & 0xFFFFFFFF, bits >> 32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_div_fmas.s")
    with open(os.path.join(_TMP_DIR.name, "ps_div_fmas.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_lines(lines):
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_div_fmas\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def validate_spirv(spv_text: str):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
    with tempfile.TemporaryDirectory() as d:
        asm_p = os.path.join(d, "s.spvasm")
        bin_p = os.path.join(d, "s.spv")
        with open(asm_p, "w", encoding="utf-8") as f:
            f.write(spv_text)
        r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_p, "-o", bin_p], capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed: {r1.stderr}")
        r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", bin_p], capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed: {r2.stderr}")


def run_f32(x: float, y: float, z: float, vcc: bool, insn: str = "v_div_fmas_f32 v0, v1, v2, v3") -> int:
    vcc_set = "v_cmp_eq_u32_e32 vcc_lo, 1, 1" if vcc else "v_cmp_eq_u32_e32 vcc_lo, 0, 1"
    spv = translate_lines([
        f"v_mov_b32 v1, {f2u(x):#x}",
        f"v_mov_b32 v2, {f2u(y):#x}",
        f"v_mov_b32 v3, {f2u(z):#x}",
        "v_mov_b32 v0, 0",
        vcc_set,
        insn,
    ])
    return si.run_fragment(spv)["%out_mrt0"][0]


class DivFmasF32Tests(unittest.TestCase):
    def test_vcc_clear_and_set(self):
        cases = (
            (1.5, 2.0, 0.5, 3.5),
            (-2.0, 3.0, 1.0, -5.0),
            (2.0 ** -32, 1.0, 0.0, 2.0 ** -32),
            (0.0, 5.0, 0.25, 0.25),
        )
        for x, y, z, base in cases:
            with self.subTest(x=x, y=y, z=z):
                self.assertEqual(run_f32(x, y, z, False), f2u(base))
                self.assertEqual(run_f32(x, y, z, True), f2u(base * float(1 << 32)))

    def test_vcc_scale_overflows_to_inf(self):
        self.assertEqual(run_f32(2.0 ** 100, 1.0, 0.0, False), f2u(2.0 ** 100))
        self.assertEqual(run_f32(2.0 ** 100, 1.0, 0.0, True), f2u(float("inf")))
        self.assertEqual(run_f32(-(2.0 ** 100), 1.0, 0.0, True), f2u(float("-inf")))

    def test_tiny_intermediate_does_not_underflow_when_vcc_set(self):
        # 2^-75 * 2^-75 = 2^-150 would flush to 0 if rounded before multiplying by 2^32,
        # but fused scaling yields 2^-118 (0x04800000).
        self.assertEqual(run_f32(2.0 ** -75, 2.0 ** -75, 0.0, False), 0)
        self.assertEqual(run_f32(2.0 ** -75, 2.0 ** -75, 0.0, True), f2u(2.0 ** -118))

    def test_input_and_output_modifiers(self):
        # |-2.0| * (-1.5) + 4.0 = 1.0; mul:2 -> 2.0; clamp -> 1.0
        self.assertEqual(run_f32(-2.0, 1.5, 4.0, False, "v_div_fmas_f32 v0, |v1|, -v2, v3"), f2u(1.0))
        self.assertEqual(run_f32(-2.0, 1.5, 4.0, False, "v_div_fmas_f32 v0, |v1|, -v2, v3 mul:2"), f2u(2.0))
        self.assertEqual(run_f32(-2.0, 1.5, 4.0, False, "v_div_fmas_f32 v0, |v1|, -v2, v3 mul:2 clamp"), f2u(1.0))
        self.assertEqual(run_f32(2.0 ** -32, 1.0, 0.0, True, "v_div_fmas_f32 v0, v1, v2, v3 div:2"), f2u(0.5))

    def test_spirv_val(self):
        spv = translate_lines([
            "v_mov_b32 v1, 0x3fc00000",
            "v_mov_b32 v2, 0x40000000",
            "v_mov_b32 v3, 0x3f000000",
            "v_cmp_eq_u32_e32 vcc_lo, 1, 1",
            "v_div_fmas_f32 v0, |v1|, -v2, v3 clamp div:2",
        ])
        self.assertNotIn("OpCapability Float64", spv)
        validate_spirv(spv)


class DivFmasF64Tests(unittest.TestCase):
    def test_f64_lowering_modifiers_and_validation(self):
        scale_lo, scale_hi = dwords(float(1 << 64))
        for mods, want_mul, want_clamp in (
            ("", False, False),
            ("mul:2", True, False),
            ("div:2", True, False),
            ("clamp", False, True),
            ("mul:4 clamp", True, True),
        ):
            with self.subTest(mods=mods):
                spv = translate_lines([
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0x3ff80000",
                    "s_mov_b32 s4, 0",
                    "s_mov_b32 s5, 0x40000000",
                    "v_cmp_eq_u32_e32 vcc_lo, 1, 1",
                    f"v_div_fmas_f64 v[0:1], |v[2:3]|, -s[4:5], 0.5 {mods}".rstrip(),
                ])
                self.assertIn("OpCapability Float64", spv)
                self.assertRegex(spv, r"OpExtInst %double %glsl Fma %\w+ %\w+ %\w+")
                self.assertRegex(spv, r"OpExtInst %double %glsl FAbs %\w+")
                self.assertRegex(spv, r"OpFNegate %double %\w+")
                self.assertIn(f"OpConstant %uint {scale_hi}", spv)
                self.assertIn(f"OpConstant %uint {scale_lo}", spv)
                self.assertRegex(spv, r"OpSelect %double %\w+ %\w+ %\w+")
                fmul_count = len(re.findall(r"OpFMul %double", spv))
                # 3 pre-scale multiplies (2^32, 2^32, 2^64) + 1 post-scale multiply (2^64) + optional omod
                self.assertEqual(fmul_count, 5 if want_mul else 4)
                if want_clamp:
                    self.assertIn("FClamp", spv)
                else:
                    self.assertNotIn("FClamp", spv)
                validate_spirv(spv)

    def test_f64_rejects_non_pair_destination(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines(["v_div_fmas_f64 v0, v[2:3], v[4:5], v[6:7]"])


if __name__ == "__main__":
    unittest.main()
