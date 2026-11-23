"""Tests for .agents/skills/verify-translator/scripts/verify_translator.py (stdlib only, no game data).

    python -I tests/test_verify_translator.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(ROOT, ".agents", "skills", "verify-translator", "scripts", "verify_translator.py")


class VerifyTranslatorSkillTests(unittest.TestCase):
    def test_doctor_and_valu_ops_drive_write_surviving_evidence(self):
        with tempfile.TemporaryDirectory() as ev_dir:
            p_doc = subprocess.run(
                [sys.executable, "-I", HARNESS, "doctor", "--root", ROOT, "--evidence", ev_dir],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(p_doc.returncode, 0, p_doc.stdout + p_doc.stderr)
            doc = json.loads(p_doc.stdout)
            self.assertTrue(doc["rdna2spirv_exists"])
            self.assertTrue(doc["semantic_support_exists"])

            p_drv = subprocess.run(
                [sys.executable, "-I", HARNESS, "drive", "--root", ROOT, "--feature", "valu-ops", "--evidence", ev_dir],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(p_drv.returncode, 0, p_drv.stdout + p_drv.stderr)
            ev_path = os.path.join(ev_dir, "valu-ops_pass.json")
            self.assertTrue(os.path.isfile(ev_path))
            with open(ev_path, encoding="utf-8") as f:
                ev = json.load(f)
            self.assertTrue(ev["ok"])
            self.assertEqual(ev["unittest_rc"], 0)


if __name__ == "__main__":
    unittest.main()
