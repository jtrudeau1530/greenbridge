"""Tests for DX10 alpha modes using synthetic DDS buffers (no game data).

    python3 -I tests/test_dds_png_dx10.py
"""
import importlib.util
import os
import struct
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "dds_png.py")
    spec = importlib.util.spec_from_file_location("dds_png_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dds = load_module()


class Dx10AlphaTests(unittest.TestCase):
    def test_rgba8_alpha_mode_in_png(self):
        header = bytearray(148)
        header[:4] = b"DDS "
        struct.pack_into("<IIIII", header, 4, 124, 0x100F, 1, 1, 4)
        struct.pack_into("<II4s", header, 76, 32, 4, b"DX10")
        struct.pack_into("<I", header, 108, 0x1000)  # DDSCAPS_TEXTURE
        struct.pack_into("<IIII", header, 128, 28, 3, 0, 1)  # RGBA8, 2D, one slice
        pixel = b"\xff\x00\x00\x00"

        for mode, alpha in ((3, 255), (1, 0)):  # Opaque and straight alpha
            with self.subTest(mode=mode):
                struct.pack_into("<I", header, 144, mode)
                png = dds.dds_to_png_bytes(bytes(header) + pixel)
                # The first chunk after IHDR contains the compressed row.
                pos = 8 + 12 + 13
                size, kind = struct.unpack_from(">I4s", png, pos)
                self.assertEqual(kind, b"IDAT")
                row = zlib.decompress(png[pos + 8:pos + 8 + size])
                self.assertEqual(row, bytes((0, 255, 0, 0, alpha)))


if __name__ == "__main__":
    unittest.main()
