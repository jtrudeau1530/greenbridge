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

    def test_d16_gradient_rejects_swapped_halves(self):
        # Binary16 encodings of UNORM8 texels 0, 32, ... 224.
        words = (0x0000, 0x3004, 0x3404, 0x3606, 0x3804, 0x3905, 0x3a06, 0x3b07)
        levels = [struct.unpack("<e", struct.pack("<H", word))[0] for word in words]
        frame = [(levels[x // 8], levels[y // 8], levels[4], 1.0) for y in range(64) for x in range(64)]
        good = self.raw(b"".join(struct.pack("<4f", *pixel) for pixel in frame))
        self.assertTrue(rgt.check_d16_texture(good)[0])
        swapped = self.raw(b"".join(struct.pack("<4f", g, r, b, a) for r, g, b, a in frame))
        ok, detail = rgt.check_d16_texture(swapped)
        self.assertFalse(ok)
        self.assertIn("3584", detail)
        self.assertIn("(8,0)", detail)

    def test_packed_interpolation_checks_each_decoded_channel(self):
        def half_word(value: float) -> int:
            # Only the low 16 bits of the first three exports carry an interpolation result.
            return 0xffff0000 | struct.unpack("<H", struct.pack("<e", value))[0]

        frame = []
        for y in range(64):
            v = (y + 0.5) / 64
            for x in range(64):
                u = (x + 0.5) / 64
                frame.append([half_word(u), half_word(v), half_word(u), struct.unpack("<I", struct.pack("<f", v))[0]])
        self.assertEqual(rgt.interp_f16_high_problem(frame), "")
        for pixel in (0, 2071):
            for channel in range(4):
                for bits in ((0x7fc00000, 0xff800000) if channel == 3 else (0x7e00, 0xfc00)):
                    with self.subTest(pixel=pixel, channel=channel, bits=bits):
                        saved = frame[pixel][channel]
                        frame[pixel][channel] = bits
                        problem = rgt.interp_f16_high_problem(frame)
                        frame[pixel][channel] = saved
                        self.assertIn("non-finite", problem)

        frame[2071][1] = half_word(0.0)
        self.assertIn("halves off", rgt.interp_f16_high_problem(frame))

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

    def test_bitmask_perm_narrow_fragment_subgroup_skip(self):
        import check
        self.assertIs(rgt.bitmask_perm_narrow_skip, check.bitmask_perm_narrow_skip)
        simd16 = self.raw(b"".join(struct.pack("<4I", i % 16, (i % 16) ^ 9, 0, 0) for i in range(64 * 64)))
        ok, detail = rgt.check_bitmask_perm(simd16, "device: Intel ANV (subgroup 8..32)\nok 64x64\n")
        self.assertIsNone(ok)
        self.assertIn("16 lanes", detail)

        # Same narrow output on a fixed 32..32 device is a real failure, not a skip.
        ok, detail = rgt.check_bitmask_perm(simd16, "device: NVIDIA (subgroup 32..32)\nok 64x64\n")
        self.assertIs(ok, False)
        self.assertIn("want (9,17,0)", detail)

        # Full 32-lane output on an 8..32 device is still judged.
        simd32 = self.raw(b"".join(
            struct.pack("<4I", i % 32, (i % 32) ^ 9, (((i % 32) & 5) | 16) ^ 1, 0) for i in range(64 * 64)))
        ok, detail = rgt.check_bitmask_perm(simd32, "device: Intel ANV (subgroup 8..32)\nok 64x64\n")
        self.assertIs(ok, True)
        self.assertIn("4096/4096", detail)


if __name__ == "__main__":
    unittest.main()
