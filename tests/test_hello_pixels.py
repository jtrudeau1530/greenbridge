"""Oracle tests for tools/hello_pixels.py: the image checks, file formats and SPIR-V sources.

No GPU and no vk_harness needed: check_image is exercised on buffers built from the same
closed form the fragment shader implements, plus deliberately wrong inputs.
"""

import os
import struct
import sys
import unittest
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import hello_pixels as hp  # noqa: E402

W = H = 64


def make_image(flip=None):
    """RGBA32F buffer of the expected pattern; flip(x,y) mutates one pixel's channels."""
    px = bytearray()
    for y in range(H):
        for x in range(W):
            p = list(hp.expected_pixel(x, y, W, H))
            if flip and (x, y) == flip[0]:
                p = list(flip[1])
            px += struct.pack("<4f", *p)
    return bytes(px)


class TestCheckImage(unittest.TestCase):
    def test_accepts_expected_image(self):
        result = hp.check_image(make_image(), W, H)
        self.assertEqual(result["covered"], W * H)

    def test_rejects_short_buffer(self):
        with self.assertRaises(AssertionError):
            hp.check_image(make_image()[:100], W, H)

    def test_rejects_wrong_pixel(self):
        # A single wrong pixel at a probe point must fail.
        x, y = W // 2, H // 2
        with self.assertRaises(AssertionError):
            hp.check_image(make_image(flip=((x, y), (9.0, 9.0, 9.0, 9.0))), W, H)

    def test_rejects_wrong_pixel_off_probe(self):
        # Pixels away from every probe still must not silently pass: drop coverage
        # (alpha 0) at a non-probe pixel instead.
        px = bytearray(make_image())
        i = (1 * W + 1) * 4
        px[i * 4 + 12 : i * 4 + 16] = struct.pack("<f", 0.0)
        with self.assertRaises(AssertionError):
            hp.check_image(bytes(px), W, H)

    def test_expected_pixel_corners(self):
        # u rises left->right, v rises bottom->top (row 0 is the top of the image).
        bottom_left = hp.expected_pixel(0, H - 1, W, H)
        bottom_right = hp.expected_pixel(W - 1, H - 1, W, H)
        top_right = hp.expected_pixel(W - 1, 0, W, H)
        self.assertGreater(bottom_right[0], bottom_left[0])  # red grows to the right
        self.assertGreater(top_right[1], bottom_right[1])  # green grows to the top
        # b = (u+v)/2 scaled by the grid cell; top-right corner lands on an undimmed cell.
        self.assertAlmostEqual(top_right[2], (top_right[0] + top_right[1]) * 0.5)


class TestEmitFormats(unittest.TestCase):
    def test_mesh_bin_layout(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "mesh.bin")
            hp.write_mesh(p)
            data = open(p, "rb").read()
        nv, ni = struct.unpack("<II", data[:8])
        self.assertEqual((nv, ni), (4, 6))
        self.assertEqual(len(data), 8 + nv * 20 + ni * 4)

    def test_ppm_header(self):
        import tempfile

        px = [0.0, 0.0, 0.0, 1.0] * 4
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "a.ppm")
            hp.write_ppm(p, px, 2, 2)
            head = open(p, "rb").read(20)
        self.assertTrue(head.startswith(b"P6\n2 2\n255\n"))

    def test_png_structure(self):
        import tempfile

        px = [1.0, 0.0, 0.0, 1.0] * 4
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "a.png")
            hp.write_png(p, px, 2, 2)
            data = open(p, "rb").read()
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"IHDR", data)
        self.assertIn(b"IDAT", data)
        self.assertTrue(data.endswith(b"IEND\xaeB\x60\x82"))

    def test_spirv_asm_sources_have_entry_points(self):
        # spirv-as is required for a real draw; at minimum the sources must name their stage entry.
        self.assertIn('OpEntryPoint Vertex %main "main"', hp.VS_ASM)
        self.assertIn('OpEntryPoint Fragment %main "main"', hp.PS_ASM)
        self.assertIn("OpExecutionMode %main OriginUpperLeft", hp.PS_ASM)


if __name__ == "__main__":
    unittest.main()
