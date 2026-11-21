"""Mutation tests: sabotage a private copy of tools/ in the ways a real translator regression would, and require
that tests/semantic/test_ps_semantics.py then FAILS for the shader that exercises the sabotaged rule.

Each mutation is an exact-text replacement that must match exactly once (so a refactor that moves the code makes
this test fail loudly instead of silently mutating nothing). The repository's own tools/ is never modified.
Each baseline run on the unmutated copy must pass first.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRANSLATOR = "rdna2spirv.py"

# (label, file, old text, new text, test that must fail)
MUTATIONS = [
    ("SDWA sext() ignored", TRANSLATOR,
     'f = "OpBitFieldSExtract" if a.get("sext") else "OpBitFieldUExtract"',
     'f = "OpBitFieldUExtract"',
     "test_ps_sdwa"),
    ("SDWA source field offset dropped", TRANSLATOR,
     'self.emit(f"{r} = {f} %uint {v} {self.cu(off)} {self.cu(width)}")',
     'self.emit(f"{r} = {f} %uint {v} {self.cu(0)} {self.cu(width)}")',
     "test_ps_sdwa"),
    ("v_min3/max3 min and max swapped", TRANSLATOR,
     'f = ("S" if op.endswith("i32") else "U") + ("Min" if "min" in op else "Max")',
     'f = ("S" if op.endswith("i32") else "U") + ("Max" if "min" in op else "Min")',
     "test_ps_ops"),
    ("v_bfe_i32 extracted as unsigned", TRANSLATOR,
     'f = "OpBitFieldSExtract" if op.endswith("i32") else "OpBitFieldUExtract"\n            self.emit(f"{r} = {f} %uint {x} {om} {wm}")',
     'f = "OpBitFieldUExtract"\n            self.emit(f"{r} = {f} %uint {x} {om} {wm}")',
     "test_ps_ops"),
    ("v_pk_fma_f16 multiplies src0 by itself", TRANSLATOR,
     'OpExtInst %v2float %glsl Fma {vx} {vy} {srcs[2]}',
     'OpExtInst %v2float %glsl Fma {vx} {vx} {srcs[2]}',
     "test_ps_ops"),
    ("v_fma_f16 ignores source op_sel half", TRANSLATOR,
     'self.emit(f"{f} = OpCompositeExtract %float {v2} {1 if hi else 0}")',
     'self.emit(f"{f} = OpCompositeExtract %float {v2} 0")',
     "test_ps_ops2"),
    ("v_subrev_co_ci_u32 drops operand swap", TRANSLATOR,
     '            if op.startswith("v_subrev_co"):\n                x, y = y, x',
     '            if False:\n                x, y = y, x',
     "test_ps_ops2"),
    ("v_cmpx_eq_u64 deactivates active lane", TRANSLATOR,
     '                self.emit(f"{both} = OpLogicalAnd %bool {eq} {self.active_b()}")\n                self.emit(f"OpStore %active {both}")',
     '                self.emit(f"{both} = OpLogicalAnd %bool {eq} {self.active_b()}")\n                self.emit("OpStore %active %false")',
     "test_ps_ops2"),
    ("v_mul_i32_i24 zero-extends instead of sign-extending", TRANSLATOR,
     'self.emit(f"{xs} = OpBitFieldSExtract %uint {x} {self.cu(0)} {self.cu(24)}")',
     'self.emit(f"{xs} = OpBitFieldUExtract %uint {x} {self.cu(0)} {self.cu(24)}")',
     "test_ps_ops3"),
    ("v_ffbh_i32 off by one from MSB", TRANSLATOR,
     'self.emit(f"{sub} = OpISub %int {self.as_int(self.cu(31))} {f}")',
     'self.emit(f"{sub} = OpISub %int {self.as_int(self.cu(30))} {f}")',
     "test_ps_ops3"),
    ("v_pk_lshlrev_b16 reads low shift amount for both halves", TRANSLATOR,
     'self.emit(f"{amt} = OpBitFieldUExtract %uint {sh_src} {self.cu(16 * sl[0])} {self.cu(4)}")',
     'self.emit(f"{amt} = OpBitFieldUExtract %uint {sh_src} {self.cu(0)} {self.cu(4)}")',
     "test_ps_ops4"),
    ("v_sin_f16 omits turns-to-radians 2*pi factor", TRANSLATOR,
     '        elif op in ("v_sin_f16", "v_cos_f16"):\n'
     '            u = self.src_u({**a[1], "neg": False, "abs": False})\n'
     '            v2, f, t, r, w2, pk, lo = (self.tmp() for _ in range(7))\n'
     '            self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")\n'
     '            self.emit(f"{f} = OpCompositeExtract %float {v2} 0")\n'
     '            f = self.modify_f(f, a[1])\n'
     '            self.emit(f"{t} = OpFMul %float {f} {self.fconst(6.283185307179586)}")',
     '        elif op in ("v_sin_f16", "v_cos_f16"):\n'
     '            u = self.src_u({**a[1], "neg": False, "abs": False})\n'
     '            v2, f, t, r, w2, pk, lo = (self.tmp() for _ in range(7))\n'
     '            self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")\n'
     '            self.emit(f"{f} = OpCompositeExtract %float {v2} 0")\n'
     '            f = self.modify_f(f, a[1])\n'
     '            self.emit(f"{t} = OpFMul %float {f} {self.fconst(1.0)}")',
     "test_ps_ops4"),
    ("v_mul_legacy_f32 drops zero-times-anything guard", TRANSLATOR,
     '        elif op == "v_mul_legacy_f32":\n'
     '            # Legacy multiply treats 0 * anything (inf/NaN included) as 0.\n'
     '            x, y = self.src_f(a[1]), self.src_f(a[2])\n'
     '            m, zx, zy, either, r = (self.tmp() for _ in range(5))\n'
     '            self.emit(f"{m} = OpFMul %float {x} {y}")\n'
     '            self.emit(f"{zx} = OpFOrdEqual %bool {x} %f0")\n'
     '            self.emit(f"{zy} = OpFOrdEqual %bool {y} %f0")\n'
     '            self.emit(f"{either} = OpLogicalOr %bool {zx} {zy}")\n'
     '            self.emit(f"{r} = OpSelect %float {either} %f0 {m}")',
     '        elif op == "v_mul_legacy_f32":\n'
     '            # Legacy multiply treats 0 * anything (inf/NaN included) as 0.\n'
     '            x, y = self.src_f(a[1]), self.src_f(a[2])\n'
     '            m, zx, zy, either, r = (self.tmp() for _ in range(5))\n'
     '            self.emit(f"{m} = OpFMul %float {x} {y}")\n'
     '            self.emit(f"{zx} = OpFOrdEqual %bool {x} %f0")\n'
     '            self.emit(f"{zy} = OpFOrdEqual %bool {y} %f0")\n'
     '            self.emit(f"{either} = OpLogicalOr %bool {zx} {zy}")\n'
     '            self.emit(f"{r} = OpCopyObject %float {m}")',
     "test_ps_legacy_mul"),
]


def run_semantics(tools_dir):
    # The child finds test_ps_semantics through PYTHONPATH, not cwd=HERE: Windows refuses a working directory longer
    # than 258 characters (WinError 267), which a checkout in a deep folder reaches.
    path = os.pathsep.join(filter(None, [HERE, os.environ.get("PYTHONPATH")]))
    env = dict(os.environ, LEONIDA_TOOLS_DIR=tools_dir, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=path)
    p = subprocess.run([sys.executable, "-m", "unittest", "test_ps_semantics", "-v"], env=env,
                       capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout + p.stderr


def failed_tests(output):
    return set(re.findall(r"^(?:FAIL|ERROR): (test_\w+)", output, re.M))


class MutationTests(unittest.TestCase):
    def make_copy(self, tmp):
        dst = os.path.join(tmp, "tools")
        shutil.copytree(os.path.join(ROOT, "tools"), dst, ignore=shutil.ignore_patterns("__pycache__", "rpf9dump"))
        return dst

    def test_baseline_copy_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out = run_semantics(self.make_copy(tmp))
            self.assertEqual(rc, 0, out)

    def test_each_mutation_is_caught(self):
        for label, fname, old, new, expect in MUTATIONS:
            with self.subTest(mutation=label), tempfile.TemporaryDirectory() as tmp:
                tools = self.make_copy(tmp)
                path = os.path.join(tools, fname)
                with open(path, encoding="utf-8", newline="") as f:
                    text = f.read()
                self.assertEqual(text.count(old), 1, f"mutation anchor must occur exactly once: {old!r}")
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(text.replace(old, new))
                rc, out = run_semantics(tools)
                self.assertNotEqual(rc, 0, f"mutation not detected: {label}\n{out}")
                self.assertIn(expect, failed_tests(out), f"{label}: expected {expect} to fail\n{out}")


if __name__ == "__main__":
    unittest.main()
