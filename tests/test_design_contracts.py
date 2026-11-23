"""Verify docs/contracts/*.json schema, status consistency, and sync with docs/design.md.

    python -I tests/test_design_contracts.py

Standard library only; no game data.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_design_doc  # noqa: E402


class DesignContractsTests(unittest.TestCase):
    def test_repo_contracts_validate_and_match_design_md(self):
        current, expected, entries = build_design_doc.validate_and_render(ROOT)
        self.assertGreaterEqual(len(entries), 150)
        self.assertEqual(
            current,
            expected,
            "docs/design.md is out of sync with docs/contracts/*.json; run `python -I tools/build_design_doc.py`",
        )

    def test_validator_rejects_contradictory_status_duplicate_piece_and_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = os.path.join(tmp, "docs")
            contracts = os.path.join(docs, "contracts")
            os.makedirs(contracts)
            design = os.path.join(docs, "design.md")
            with open(design, "w", encoding="utf-8", newline="\n") as f:
                f.write("# Design\n\n| Piece | Contract |\n|---|---|\n| Alpha | stale |\n")

            # 1. Contradictory status must fail
            a_path = os.path.join(contracts, "alpha.json")
            with open(a_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"piece": "Alpha", "status": "gpu_verified", "sentences": ["GPU-checked. **Not GPU-verified.**"]}, f)
            with self.assertRaisesRegex(ValueError, "gpu_verified"):
                build_design_doc.load_contracts(contracts)

            # 2. Valid contract + stale design.md must fail --check
            with open(a_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"piece": "Alpha", "status": "cpu_only", "sentences": ["Fresh text."]}, f)
            self.assertNotEqual(build_design_doc.main(["--root", tmp, "--check"]), 0)

            # 3. Regenerating makes --check pass
            self.assertEqual(build_design_doc.main(["--root", tmp]), 0)
            self.assertEqual(build_design_doc.main(["--root", tmp, "--check"]), 0)

            # 4. Duplicate piece across two files must fail
            b_path = os.path.join(contracts, "alpha_dup.json")
            with open(b_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"piece": "Alpha", "status": "cpu_only", "sentences": ["Other text."]}, f)
            with self.assertRaisesRegex(ValueError, "duplicate piece"):
                build_design_doc.load_contracts(contracts)


if __name__ == "__main__":
    unittest.main()
