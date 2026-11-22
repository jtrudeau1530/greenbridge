"""scripts/run_gpu_tests.py: the pixel checkers fail a short frame or non-finite values, as check.py does (no device needed)."""
from pathlib import Path
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gpu_tests as rgt

NAN = float("nan")


class PixelChecks(unittest.TestCase):
    def raw(self, data: bytes) -> str:
        fd, path = tempfile.mkstemp(suffix=".raw")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        self.addCleanup(os.remove, path)
        return path

    def test_good_frames_pass(self):
        grad = b"".join(struct.pack("<4f", (i % 64 + 0.5) / 64, (i // 64 + 0.5) / 64, 0.5, 1.0) for i in range(4096))
        self.assertTrue(rgt.check_grad(self.raw(grad), 0.5, 1.0)[0])
        bary = b"".join(struct.pack("<4f", 0.5, 0.25, 0.25, 0.125) for _ in range(4096))
        self.assertTrue(rgt.check_bary(self.raw(bary))[0])

    def test_nan_frame_fails(self):
        nan = self.raw(struct.pack("<4f", NAN, NAN, NAN, NAN) * 4096)
        self.assertFalse(rgt.check_bary(nan)[0])
        self.assertFalse(rgt.check_grad(nan, 0.5, 1.0)[0])

    def test_empty_frame_fails(self):
        empty = self.raw(b"")
        for name, check in (("bary", rgt.check_bary), ("grad", lambda p: rgt.check_grad(p, 0.0, 0.0)),
                            ("wqm", rgt.check_wqm)):
            with self.subTest(check=name):
                self.assertFalse(check(empty)[0])


if __name__ == "__main__":
    unittest.main()
