import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "carve_all.ps1"
INDEX = {"summary": {"files": 1, "bink2_files": 0, "span": 10, "compressed_bytes": 5, "unread_bytes": 5,
                     "inflated_bytes": 5}, "files": []}


@unittest.skipUnless(os.name == "nt", "PowerShell research script")
class CarveAllGamePathTest(unittest.TestCase):
    def test_relative_and_trailing_separator_game_paths_find_existing_carves(self):
        with tempfile.TemporaryDirectory() as tmp:
            game, out = Path(tmp, "game"), Path(tmp, "out")
            (game / "pack" / "p1").mkdir(parents=True)
            (game / "common.rpf").write_bytes(b"x")
            (game / "pack" / "p1" / "p1.rpf").write_bytes(b"xy")
            # Both archives are already carved, so no carve should start.
            for name in ("common", "pack_p1_p1"):
                (out / name).mkdir(parents=True)
                (out / name / "index.json").write_text(json.dumps(INDEX), encoding="utf-8")
            for arg in ("game", ".\\game", str(game) + "\\"):
                with self.subTest(game=arg):
                    result = subprocess.run(
                        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                         "-Game", arg, "-Out", str(out)], cwd=tmp, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn("carving", result.stdout)
                    rows = json.loads((out / "coverage.json").read_text(encoding="utf-8-sig"))
                    self.assertEqual([r["archive"] for r in rows], ["common.rpf", "pack\\p1\\p1.rpf"])
                    self.assertEqual(sorted(p.name for p in out.iterdir()),
                                     ["common", "coverage.json", "pack_p1_p1"])


if __name__ == "__main__":
    unittest.main()
