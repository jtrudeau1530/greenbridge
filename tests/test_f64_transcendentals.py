"""Check SPIR-V lowering and output modifiers for v_rcp_f64, v_rsq_f64 and v_sqrt_f64.

LLVM defines all three as VOP1Inst<..., VOP_F64_F64>, whose VOP3 (_e64) form carries input modifiers
(abs, neg) and output modifiers (mul:2, mul:4, div:2, then clamp to [0, 1]).
"""

import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(TEST_DIR.parent / "tools"), str(TEST_DIR)]

from listing import translate_listing


def words(x):
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    return bits & 0xFFFFFFFF, bits >> 32


def load_double(lo_reg, hi_reg, value):
    lo, hi = words(value)
    return f"v_mov_b32 v{lo_reg}, {lo:#x}\nv_mov_b32 v{hi_reg}, {hi:#x}"


class F64TranscendentalsTest(unittest.TestCase):
    def _check_store_pair(self, spv, value_id, dst_lo, dst_hi):
        def result(instruction):
            match = re.search(rf"^ *(%\w+) = {instruction}$", spv, re.M)
            self.assertIsNotNone(match, f"Missing SPIR-V instruction: {instruction}")
            return match.group(1)

        pair = result(rf"OpBitcast %v2uint_d {value_id}")
        for index, reg in ((0, f"%c{dst_lo}"), (1, f"%c{dst_hi}")):
            word = result(rf"OpCompositeExtract %uint {pair} {index}")
            dest = result(rf"OpAccessChain %_ptr_Function_uint %vgpr {reg}")
            old_word = result(rf"OpLoad %uint {dest}")
            masked = result(rf"OpSelect %uint %\w+ {word} {old_word}")
            self.assertRegex(spv, rf"(?m)^ *OpStore {dest} {masked}$")

    def test_v_rcp_rsq_sqrt_f64_lowering_and_modifiers(self):
        ops = (
            ("v_rcp_f64_e32", "", r"OpFDiv %double %\w+ %\w+", None, False),
            ("v_rsq_f64_e32", "", r"OpExtInst %double %glsl InverseSqrt %\w+", None, False),
            ("v_sqrt_f64_e32", "", r"OpExtInst %double %glsl Sqrt %\w+", None, False),
            ("v_rcp_f64_e64", "mul:2", r"OpFDiv %double %\w+ %\w+", 2.0, False),
            ("v_rsq_f64_e64", "div:2 clamp", r"OpExtInst %double %glsl InverseSqrt %\w+", 0.5, True),
            ("v_sqrt_f64_e64", "mul:4 clamp", r"OpExtInst %double %glsl Sqrt %\w+", 4.0, True),
        )
        prefix = load_double(0, 1, 4.0)
        with tempfile.TemporaryDirectory() as directory:
            for op, mods, pattern, scale, clamp in ops:
                with self.subTest(op=op, mods=mods):
                    line = f"{op} v[4:5], v[0:1] {mods}".rstrip()
                    spv = translate_listing(Path(directory), f"{prefix}\n{line}")
                    self.assertIn("OpCapability Float64", spv)
                    m = re.search(rf"^ *(%\w+) = {pattern}$", spv, re.M)
                    self.assertIsNotNone(m, f"Missing {pattern} in:\n{spv}")
                    val = m.group(1)
                    if scale is not None:
                        m_mul = re.search(rf"^ *(%\w+) = OpFMul %double {val} %\w+$", spv, re.M)
                        self.assertIsNotNone(m_mul)
                        val = m_mul.group(1)
                    if clamp:
                        m_cl = re.search(rf"^ *(%\w+) = OpExtInst %double %glsl FClamp {val} %\w+ %\w+$", spv, re.M)
                        self.assertIsNotNone(m_cl)
                        val = m_cl.group(1)
                    self._check_store_pair(spv, val, 4, 5)
                    if shutil.which("spirv-as") and shutil.which("spirv-val"):
                        spv_bin = Path(directory) / f"{op}.spv"
                        p_as = subprocess.run(
                            ["spirv-as", "--target-env", "vulkan1.2", "-o", str(spv_bin), "-"],
                            input=spv, text=True, capture_output=True,
                        )
                        self.assertEqual(p_as.returncode, 0, p_as.stderr)
                        p_val = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(spv_bin)],
                                               text=True, capture_output=True)
                        self.assertEqual(p_val.returncode, 0, p_val.stderr)

    def test_input_modifiers_abs_neg_and_inline_constant(self):
        prefix = load_double(0, 1, -4.0)
        body = "\n".join((
            prefix,
            "v_sqrt_f64_e64 v[2:3], |v[0:1]|",
            "v_rcp_f64_e64 v[4:5], -v[2:3]",
            "v_rsq_f64_e64 v[6:7], 4.0",
        ))
        with tempfile.TemporaryDirectory() as directory:
            spv = translate_listing(Path(directory), body)
            self.assertIn("OpExtInst %double %glsl FAbs", spv)
            self.assertIn("OpFNegate %double", spv)
            self.assertIn("OpExtInst %double %glsl Sqrt", spv)
            self.assertIn("OpFDiv %double", spv)
            self.assertIn("OpExtInst %double %glsl InverseSqrt", spv)
            if shutil.which("spirv-as") and shutil.which("spirv-val"):
                spv_bin = Path(directory) / "mods.spv"
                p_as = subprocess.run(
                    ["spirv-as", "--target-env", "vulkan1.2", "-o", str(spv_bin), "-"],
                    input=spv, text=True, capture_output=True,
                )
                self.assertEqual(p_as.returncode, 0, p_as.stderr)
                p_val = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(spv_bin)],
                                       text=True, capture_output=True)
                self.assertEqual(p_val.returncode, 0, p_val.stderr)


if __name__ == "__main__":
    unittest.main()
