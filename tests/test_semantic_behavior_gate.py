"""Tests for scripts/check_semantic_behavior.py (standard library only, no game data).

    python -I tests/test_semantic_behavior_gate.py
"""
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import check_semantic_behavior as gate  # noqa: E402
import test_semantic_translator_use as ast_guard  # noqa: E402


class SemanticBehaviorGateTests(unittest.TestCase):
    def test_real_semantic_test_passes_behavior_gate(self):
        rows = gate.check_files(ROOT, [os.path.join("tests", "semantic", "test_alu_int_misc.py")], timeout=30)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verdict"], "REAL_PASS", rows)

    def test_catches_adversarial_bypasses_that_fool_ast_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            sem_dir = os.path.join(tmp, "sem")
            os.mkdir(sem_dir)
            src_sem = os.path.join(ROOT, "tests", "semantic")
            for name in ("support.py", "spirv_mini_interp.py"):
                shutil.copy2(os.path.join(src_sem, name), os.path.join(sem_dir, name))

            adv_discard = os.path.join(sem_dir, "test_adv_discard_output.py")
            with open(adv_discard, "w", encoding="utf-8", newline="\n") as f:
                f.write(
                    "import unittest\nimport support\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_fake(self):\n"
                    "        _ = support.translate_lines('ps_fake', ['v_add_nc_u32 v0, 1, 2', "
                    "'exp mrt0 v0, v0, v0, v0 done vm', 's_endpgm'])\n"
                    "        self.assertEqual(1 + 2, 3)\n"
                )
            adv_header = os.path.join(sem_dir, "test_adv_nonempty_only.py")
            with open(adv_header, "w", encoding="utf-8", newline="\n") as f:
                f.write(
                    "import unittest\nimport support\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_fake_header(self):\n"
                    "        spv = support.translate_lines('ps_fake2', ['v_mul_lo_u32 v0, 3, 4', "
                    "'exp mrt0 v0, v0, v0, v0 done vm', 's_endpgm'])\n"
                    "        self.assertIn('OpCapability Shader', spv)\n"
                    "        self.assertEqual((3 * 4) & 0xFFFFFFFF, 12)\n"
                )

            # Confirm the static AST reachability guard accepts both bypasses.
            self.assertEqual(ast_guard.offenders(sem_dir, allowlist={}, debt={}), [])

            rows = gate.check_files(ROOT, [adv_discard, adv_header], timeout=30)
            verdicts = {r["file"]: r["verdict"] for r in rows}
            self.assertEqual(verdicts, {
                "test_adv_discard_output.py": "FAKE_CAUGHT",
                "test_adv_nonempty_only.py": "FAKE_CAUGHT",
            })


if __name__ == "__main__":
    unittest.main()
