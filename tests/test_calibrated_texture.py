"""Build a synthetic TTPF and exercise the same calibrated CLI used on local pools."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tools")]
import test_ttpf_mips as fixtures
import bcpack_tables


def synthetic_profile():
    # Expand each quadrant into its four positions in the BC1 selector word.
    shifts = ((0, 8), (4, 12), (16, 24), (20, 28))
    selectors = [
        [((q & 15) << low) | ((q >> 4) << high) for q in bcpack_tables.BC1_QUADRANT]
        for low, high in shifts
    ]
    return {"version": 1, "bc1": {"endpoints": [[0] * 6 for _ in range(256)], "selectors": selectors}}


@unittest.skipUnless(importlib.util.find_spec("numpy"), "install NumPy for calibrated export")
class CalibratedExport(unittest.TestCase):
    def test_bc1_export_and_existing_output(self):
        import calibrated_texture
        try:
            calibrated_texture.native()
        except OSError as exc:
            self.skipTest(f"build tools/bcpack_calibration first: {exc}")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = root / "pool.ttpf"
            output = root / "result.dds"
            profile = root / "profile.json"
            _, linear, _ = fixtures.build_bc1_bcpack_pool(pool)
            profile.write_text(json.dumps(synthetic_profile()), encoding="utf-8")
            command = [sys.executable, "-IB", str(ROOT / "tools/ttpf_mips.py"),
                       "export", str(pool), "0", str(output), "--calibration", str(profile)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            data = output.read_bytes()
            self.assertEqual(data[:4], b"DDS ")
            self.assertEqual(data[148:], b"".join(linear[i] for i in sorted(linear)))
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            self.assertEqual(output.read_bytes(), data)
            profile.write_text('{"version": 99}', encoding="utf-8")
            self.assertEqual(subprocess.run(command + ["--force"], capture_output=True).returncode, 2)
            self.assertEqual(output.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
