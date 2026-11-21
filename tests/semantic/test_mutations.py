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
]


def run_semantics(tools_dir):
    env = dict(os.environ, LEONIDA_TOOLS_DIR=tools_dir, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-m", "unittest", "test_ps_semantics", "-v"], cwd=HERE, env=env,
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
