"""Tests for scripts/preflight_pr.py (standard library only, no game data).

    python -I tests/test_preflight_pr.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "preflight_pr.py")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import preflight_pr  # noqa: E402


class PreflightPrTests(unittest.TestCase):
    def test_every_configured_repo_rule_test_exists(self):
        missing = [rel for rel in preflight_pr.REPO_RULE_TESTS if not os.path.isfile(os.path.join(ROOT, *rel.split("/")))]
        self.assertEqual(missing, [], f"REPO_RULE_TESTS names missing files: {missing}")

    def test_list_mode_reports_checks(self):
        p = subprocess.run([sys.executable, "-I", SCRIPT, "--root", ROOT, "--list"],
                           capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = set(p.stdout.splitlines())
        self.assertIn("no-game-data:extensions", lines)
        self.assertIn("tests/test_ci_registration.py", lines)

    def test_catches_forbidden_game_extension_and_passes_clean_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean_rep = os.path.join(tmp, "clean.json")
            p_ok = subprocess.run([sys.executable, "-I", SCRIPT, "--root", tmp, "--jobs", "1", "--json", clean_rep],
                                  capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(p_ok.returncode, 0, p_ok.stdout + p_ok.stderr)
            with open(clean_rep, encoding="utf-8") as f:
                data = json.load(f)
            self.assertTrue(data["ok"])

            bad_file = os.path.join(tmp, "leak.spv")
            with open(bad_file, "wb") as f:
                f.write(b"\x03\x02\x23\x07")
            p_bad = subprocess.run([sys.executable, "-I", SCRIPT, "--root", tmp, "--jobs", "1"],
                                   capture_output=True, text=True, encoding="utf-8")
            self.assertNotEqual(p_bad.returncode, 0, "preflight_pr must reject tracked/present .spv files")
            self.assertIn("no-game-data:extensions", p_bad.stdout)


if __name__ == "__main__":
    unittest.main()
