"""Unit tests for tools/tex_headers.py: DDS header synthesis, mipmap dimension calculations,
GFX10 format mapping, and texture descriptor parsing from synthetic blobs.

    python -I tests/test_tex_headers.py
"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import tex_headers

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


class TestGFX10FormatMapping(unittest.TestCase):
    def test_known_bc_format_mappings(self):
        cases = [
            (169, 71, "BC1_UNORM"),
            (170, 72, "BC1_UNORM_SRGB"),
            (171, 74, "BC2_UNORM"),
            (172, 75, "BC2_UNORM_SRGB"),
            (173, 77, "BC3_UNORM"),
            (174, 78, "BC3_UNORM_SRGB"),
            (175, 80, "BC4_UNORM"),
            (176, 81, "BC4_SNORM"),
            (177, 83, "BC5_UNORM"),
            (178, 84, "BC5_SNORM"),
            (179, 95, "BC6H_UF16"),
            (180, 96, "BC6H_SF16"),
            (181, 98, "BC7_UNORM"),
            (182, 99, "BC7_UNORM_SRGB"),
        ]
        for gfx10_code, expected_dxgi, expected_name in cases:
            with self.subTest(gfx10=gfx10_code, name=expected_name):
                dxgi_id, name = tex_headers.map_gfx10_format(gfx10_code)
                self.assertEqual(dxgi_id, expected_dxgi)
                self.assertEqual(name, expected_name)

    def test_known_uncompressed_mappings(self):
        cases = [
            (1, 61, "R8_UNORM"),
            (7, 56, "R16_UNORM"),
            (13, 54, "R16_FLOAT"),
            (14, 49, "R8G8_UNORM"),
            (22, 41, "R32_FLOAT"),
            (29, 34, "R16G16_FLOAT"),
            (43, 26, "R11G11B10_FLOAT"),
            (44, 24, "R10G10B10A2_UNORM"),
            (56, 28, "R8G8B8A8_UNORM"),
            (130, 29, "R8G8B8A8_UNORM_SRGB"),
            (64, 16, "R32G32_FLOAT"),
            (71, 10, "R16G16B16A16_FLOAT"),
            (77, 2, "R32G32B32A32_FLOAT"),
        ]
        for gfx10_code, expected_dxgi, expected_name in cases:
            with self.subTest(gfx10=gfx10_code, name=expected_name):
                dxgi_id, name = tex_headers.map_gfx10_format(gfx10_code)
                self.assertEqual(dxgi_id, expected_dxgi)
                self.assertEqual(name, expected_name)

    def test_unsupported_gfx10_raises(self):
        for invalid in [0, 9999, -1, 400]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    tex_headers.map_gfx10_format(invalid)

    def test_dxgi_to_gfx10_roundtrip(self):
        major_dxgi = [71, 72, 74, 77, 78, 80, 83, 95, 96, 98, 99, 28, 29, 56, 61]
        for dxgi_id in major_dxgi:
            with self.subTest(dxgi=dxgi_id):
                gfx10 = tex_headers.dxgi_to_gfx10(dxgi_id)
                mapped_dxgi, _ = tex_headers.map_gfx10_format(gfx10)
                self.assertEqual(mapped_dxgi, dxgi_id)

    def test_unsupported_dxgi_raises(self):
        with self.assertRaises(ValueError):
            tex_headers.dxgi_to_gfx10(99999)

    def test_dxgi_dict_integrity(self):
        for dxgi_id, name in tex_headers.DXGI.items():
            self.assertIsInstance(dxgi_id, int)
            self.assertIsInstance(name, str)
            self.assertTrue(len(name) > 0)


class TestMipmapCalculations(unittest.TestCase):
    def test_calc_mipmap_count(self):
        cases = [
            (1, 1, 1, 1),
            (2, 2, 1, 2),
            (4, 4, 1, 3),
            (8, 8, 1, 4),
            (256, 256, 1, 9),
            (1024, 1024, 1, 11),
            (1920, 1080, 1, 11),
            (4096, 4096, 1, 13),
            (512, 64, 1, 10),
            (64, 64, 16, 7),
        ]
        for w, h, d, expected in cases:
            with self.subTest(w=w, h=h, d=d):
                self.assertEqual(tex_headers.calc_mipmap_count(w, h, d), expected)

    def test_calc_mipmap_count_zero_or_negative(self):
        self.assertEqual(tex_headers.calc_mipmap_count(0, 100), 0)
        self.assertEqual(tex_headers.calc_mipmap_count(100, 0), 0)
        self.assertEqual(tex_headers.calc_mipmap_count(-10, 100), 0)
        self.assertEqual(tex_headers.calc_mipmap_count(100, 100, 0), 0)

    def test_calc_mipmap_dimensions_full_chain(self):
        levels = tex_headers.calc_mipmap_dimensions(16, 16, 1, mips=0)
        expected = [
            (16, 16, 1),
            (8, 8, 1),
            (4, 4, 1),
            (2, 2, 1),
            (1, 1, 1),
        ]
        self.assertEqual(levels, expected)

    def test_calc_mipmap_dimensions_non_square(self):
        levels = tex_headers.calc_mipmap_dimensions(32, 8, 1, mips=0)
        expected = [
            (32, 8, 1),
            (16, 4, 1),
            (8, 2, 1),
            (4, 1, 1),
            (2, 1, 1),
            (1, 1, 1),
        ]
        self.assertEqual(levels, expected)

    def test_calc_mipmap_dimensions_limited_count(self):
        levels = tex_headers.calc_mipmap_dimensions(1024, 1024, 1, mips=3)
        expected = [
            (1024, 1024, 1),
            (512, 512, 1),
            (256, 256, 1),
        ]
        self.assertEqual(levels, expected)

    def test_calc_mipmap_dimensions_3d_volume(self):
        levels = tex_headers.calc_mipmap_dimensions(16, 16, 8, mips=0)
        expected = [
            (16, 16, 8),
            (8, 8, 4),
            (4, 4, 2),
            (2, 2, 1),
            (1, 1, 1),
        ]
        self.assertEqual(levels, expected)

    def test_calc_mipmap_dimensions_invalid(self):
        self.assertEqual(tex_headers.calc_mipmap_dimensions(0, 10), [])
        self.assertEqual(tex_headers.calc_mipmap_dimensions(10, -5), [])

    def test_calc_mip_level_bytes_bc1(self):
        # BC1 is 8 bytes per 4x4 block
        # 4x4 -> 1 block = 8 bytes
        self.assertEqual(tex_headers.calc_mip_level_bytes(4, 4, 1, fmt=71), 8)
        # 1024x1024 -> 256x256 blocks = 65536 blocks * 8 bytes = 524288 bytes
        self.assertEqual(tex_headers.calc_mip_level_bytes(1024, 1024, 1, fmt=71), 524288)
        # Sub-block mip levels (2x2, 1x1) must still pad to a 4x4 block
        self.assertEqual(tex_headers.calc_mip_level_bytes(2, 2, 1, fmt=71), 8)
        self.assertEqual(tex_headers.calc_mip_level_bytes(1, 1, 1, fmt=71), 8)

    def test_calc_mip_level_bytes_bc7(self):
        # BC7 is 16 bytes per 4x4 block
        self.assertEqual(tex_headers.calc_mip_level_bytes(4, 4, 1, fmt=98), 16)
        self.assertEqual(tex_headers.calc_mip_level_bytes(1024, 1024, 1, fmt=98), 1048576)
        self.assertEqual(tex_headers.calc_mip_level_bytes(1, 1, 1, fmt=98), 16)

    def test_calc_mip_level_bytes_uncompressed(self):
        # R8_UNORM = 1 byte per pixel
        self.assertEqual(tex_headers.calc_mip_level_bytes(64, 64, 1, fmt=61), 4096)
        # R8G8B8A8_UNORM = 4 bytes per pixel
        self.assertEqual(tex_headers.calc_mip_level_bytes(64, 64, 1, fmt=28), 16384)
        # R16G16B16A16_FLOAT = 8 bytes per pixel
        self.assertEqual(tex_headers.calc_mip_level_bytes(64, 64, 1, fmt=10), 32768)
        # R32G32B32A32_FLOAT = 16 bytes per pixel
        self.assertEqual(tex_headers.calc_mip_level_bytes(64, 64, 1, fmt=2), 65536)

    def test_calc_mip_level_bytes_aligned(self):
        # 8 bytes aligned to 256 -> 256
        self.assertEqual(tex_headers.calc_mip_level_bytes(4, 4, 1, fmt=71, align=256), 256)
        # 8 bytes aligned to 4096 -> 4096
        self.assertEqual(tex_headers.calc_mip_level_bytes(4, 4, 1, fmt=71, align=4096), 4096)

    def test_calc_mip_level_bytes_unsupported_format(self):
        with self.assertRaises(ValueError):
            tex_headers.calc_mip_level_bytes(64, 64, 1, fmt=9999)


class TestDDSHeaderSynthesis(unittest.TestCase):
    def test_dds_header_2d_basic(self):
        hdr = tex_headers.dds_header(512, 256, depth=1, mips=1, fmt=28, is_cube=False, linear_size=512 * 256 * 4)
        self.assertEqual(len(hdr), 148)
        self.assertEqual(hdr[:4], b"DDS ")

        info = tex_headers.parse_dds_header(hdr)
        self.assertEqual(info["magic"], b"DDS ")
        self.assertEqual(info["width"], 512)
        self.assertEqual(info["height"], 256)
        self.assertEqual(info["depth"], 0)
        self.assertEqual(info["mips"], 1)
        self.assertEqual(info["fourcc"], b"DX10")
        self.assertEqual(info["dxgi_format"], 28)
        self.assertEqual(info["resource_dimension"], tex_headers.D3D10_RESOURCE_DIMENSION_TEXTURE2D)
        self.assertEqual(info["misc_flags"], 0)
        self.assertEqual(info["array_size"], 1)

    def test_dds_header_with_mips(self):
        hdr = tex_headers.dds_header(1024, 1024, depth=1, mips=11, fmt=71)
        info = tex_headers.parse_dds_header(hdr)
        self.assertEqual(info["width"], 1024)
        self.assertEqual(info["height"], 1024)
        self.assertEqual(info["mips"], 11)
        self.assertTrue(info["flags"] & tex_headers.DDSD_MIPMAPCOUNT)
        self.assertTrue(info["caps"] & tex_headers.DDSCAPS_MIPMAP)
        self.assertTrue(info["caps"] & tex_headers.DDSCAPS_COMPLEX)
        self.assertEqual(info["dxgi_format"], 71)

    def test_dds_header_3d_volume(self):
        hdr = tex_headers.dds_header(64, 64, depth=16, mips=1, fmt=28)
        info = tex_headers.parse_dds_header(hdr)
        self.assertEqual(info["depth"], 16)
        self.assertTrue(info["flags"] & tex_headers.DDSD_DEPTH)
        self.assertTrue(info["caps2"] & tex_headers.DDSCAPS2_VOLUME)
        self.assertEqual(info["resource_dimension"], tex_headers.D3D10_RESOURCE_DIMENSION_TEXTURE3D)

    def test_dds_header_cubemap(self):
        hdr = tex_headers.dds_header(256, 256, depth=1, mips=1, fmt=28, is_cube=True)
        info = tex_headers.parse_dds_header(hdr)
        self.assertTrue(info["caps2"] & tex_headers.DDSCAPS2_CUBEMAP_ALLFACES)
        self.assertEqual(info["misc_flags"], tex_headers.D3D10_RESOURCE_MISC_TEXTURECUBE)
        self.assertEqual(info["array_size"], 6)

    def test_dds_bytes_payload(self):
        payload = b"\xAA\xBB\xCC\xDD" * 64
        data = tex_headers.dds_bytes(16, 16, fmt=71, payload=payload)
        self.assertEqual(len(data), 148 + len(payload))
        self.assertEqual(data[148:], payload)
        info = tex_headers.parse_dds_header(data)
        self.assertEqual(info["width"], 16)
        self.assertEqual(info["height"], 16)
        self.assertEqual(info["pitch_linear"], len(payload))

    def test_parse_corrupt_headers(self):
        # Truncated header (< 128 bytes)
        with self.assertRaises(ValueError):
            tex_headers.parse_dds_header(b"DDS " + b"\x00" * 50)

        # Bad magic
        with self.assertRaises(ValueError):
            tex_headers.parse_dds_header(b"NOTD" + b"\x00" * 140)

        # Bad header size (not 124)
        bad_size = bytearray(b"DDS " + b"\x00" * 140)
        struct.pack_into("<I", bad_size, 4, 100)
        with self.assertRaises(ValueError):
            tex_headers.parse_dds_header(bytes(bad_size))

        # Bad pixel format size (not 32)
        bad_pf = bytearray(b"DDS " + b"\x00" * 140)
        struct.pack_into("<I", bad_pf, 4, 124)
        struct.pack_into("<I", bad_pf, 76, 20)
        with self.assertRaises(ValueError):
            tex_headers.parse_dds_header(bytes(bad_pf))

        # Truncated DX10 extension (< 148 bytes when fourcc == b"DX10")
        bad_dx10 = bytearray(b"DDS " + b"\x00" * 130)
        struct.pack_into("<I", bad_dx10, 4, 124)
        struct.pack_into("<II4s", bad_dx10, 76, 32, 4, b"DX10")
        with self.assertRaises(ValueError):
            tex_headers.parse_dds_header(bytes(bad_dx10))


@unittest.skipUnless(HAVE_NUMPY, "numpy required for scan_archive")
class TestTextureDescriptorParsing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.carve = self.tmp.name
        self.arch = "arch_001"
        self.arch_dir = os.path.join(self.carve, self.arch)
        os.makedirs(self.arch_dir, exist_ok=True)

    def _make_synthetic_texture_blobs(self, w=512, h=256, depth=1, fmt=71, data_off=0x1000, mips=9):
        cpu_off = 0x1000
        gpu_off = 0x2000

        # Construct CPU blob:
        # Texture descriptor layout:
        #   q[k-2] = packed u64
        #   q[k-1] = 0x980 (descriptor type)
        #   q[k]   = 0x60000000 + data_off
        #   q[k+4] = u16 width, u16 height, u16 depth, u8 b6, u8 fmt
        #   q[k+5] = flags / mips ((mips << 16) | flags)
        qwords = [0] * 32
        # Starts with zero qword, but head not 32 zeros
        qwords[0] = 0
        qwords[1] = 0x123456789ABCDEF0

        k = 10
        qwords[k - 2] = 0xCAFEBABE00000001
        qwords[k - 1] = 0x980
        qwords[k] = 0x60000000 + data_off
        d_bytes = struct.pack("<HHHBB", w, h, depth, 0x42, fmt)
        qwords[k + 4] = struct.unpack("<Q", d_bytes)[0]
        qwords[k + 5] = (mips << 16) | 0x0001

        cpu_bytes = struct.pack(f"<{len(qwords)}Q", *qwords)
        gpu_bytes = b"\x00" * (data_off + 4096)

        cpu_head = cpu_bytes[:16].hex()
        gpu_head = gpu_bytes[:16].hex()

        with open(os.path.join(self.arch_dir, f"{cpu_off:012x}.bin"), "wb") as f:
            f.write(cpu_bytes)
        with open(os.path.join(self.arch_dir, f"{gpu_off:012x}.bin"), "wb") as f:
            f.write(gpu_bytes)

        index = {
            "files": [
                {"off": cpu_off, "head": cpu_head, "size": len(cpu_bytes)},
                {"off": gpu_off, "head": gpu_head, "size": len(gpu_bytes)},
            ]
        }
        with open(os.path.join(self.arch_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f)

        return cpu_off, gpu_off, k

    def test_scan_archive_success(self):
        cpu_off, gpu_off, k = self._make_synthetic_texture_blobs(w=1024, h=512, depth=1, fmt=71, data_off=0x2000, mips=11)
        results = tex_headers.scan_archive(self.carve, self.arch)

        self.assertEqual(len(results), 1)
        t = results[0]
        self.assertEqual(t["archive"], self.arch)
        self.assertEqual(t["cpu"], cpu_off)
        self.assertEqual(t["gpu"], gpu_off)
        self.assertEqual(t["w"], 1024)
        self.assertEqual(t["h"], 512)
        self.assertEqual(t["depth"], 1)
        self.assertEqual(t["fmt"], 71)
        self.assertEqual(t["fmt_name"], "BC1_UNORM")
        self.assertEqual(t["data_off"], 0x2000)
        self.assertEqual(t["mips"], 11)
        self.assertEqual(t["b6"], 0x42)
        self.assertEqual(t["dtype"], 0x980)
        self.assertEqual(t["packed"], 0xCAFEBABE00000001)
        self.assertEqual(t["q_index"], k)

    def test_scan_archive_unaligned_pointer_rejected(self):
        # Texture data pointer must be 256-byte aligned (int(q[k]) & 0xFF == 0)
        self._make_synthetic_texture_blobs(data_off=0x1001)
        results = tex_headers.scan_archive(self.carve, self.arch)
        self.assertEqual(len(results), 0)

    def test_scan_archive_invalid_dimensions_rejected(self):
        # Width < 4
        self._make_synthetic_texture_blobs(w=2, h=256)
        self.assertEqual(len(tex_headers.scan_archive(self.carve, self.arch)), 0)

        # Height > 16384
        self._make_synthetic_texture_blobs(w=256, h=20000)
        self.assertEqual(len(tex_headers.scan_archive(self.carve, self.arch)), 0)

        # Depth > 2048
        self._make_synthetic_texture_blobs(w=256, h=256, depth=3000)
        self.assertEqual(len(tex_headers.scan_archive(self.carve, self.arch)), 0)

    def test_scan_archive_unknown_format_rejected(self):
        # Format not in DXGI
        self._make_synthetic_texture_blobs(fmt=254)
        self.assertEqual(len(tex_headers.scan_archive(self.carve, self.arch)), 0)

    def test_scan_archive_gpu_chain_and_skips(self):
        cpu_off = 0x1000
        gpu1_off = 0x2000
        gpu2_off = 0x3000
        bink_off = 0x4000

        qwords = [0] * 32
        qwords[0] = 0
        qwords[1] = 0x123456789ABCDEF0
        k = 10
        qwords[k - 2] = 1
        qwords[k - 1] = 0x980
        qwords[k] = 0x60000000 + 0x1000
        qwords[k + 4] = struct.unpack("<Q", struct.pack("<HHHBB", 256, 256, 1, 0, 71))[0]
        qwords[k + 5] = (9 << 16) | 1

        cpu_bytes = struct.pack(f"<{len(qwords)}Q", *qwords)
        gpu1_bytes = b"\x11" * 4096
        gpu2_bytes = b"\x22" * 4096
        bink_bytes = b"\x33" * 4096

        with open(os.path.join(self.arch_dir, f"{cpu_off:012x}.bin"), "wb") as f:
            f.write(cpu_bytes)
        with open(os.path.join(self.arch_dir, f"{gpu1_off:012x}.bin"), "wb") as f:
            f.write(gpu1_bytes)
        with open(os.path.join(self.arch_dir, f"{gpu2_off:012x}.bin"), "wb") as f:
            f.write(gpu2_bytes)
        with open(os.path.join(self.arch_dir, f"{bink_off:012x}.bin"), "wb") as f:
            f.write(bink_bytes)

        index = {
            "files": [
                {"off": cpu_off, "head": cpu_bytes[:16].hex(), "size": len(cpu_bytes)},
                {"off": gpu1_off, "head": gpu1_bytes[:16].hex(), "size": len(gpu1_bytes)},
                {"off": gpu2_off, "head": gpu2_bytes[:16].hex(), "size": len(gpu2_bytes)},
                {"off": bink_off, "head": "4240000000000000", "size": len(bink_bytes), "type": "bink2"},
            ]
        }
        with open(os.path.join(self.arch_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f)

        results = tex_headers.scan_archive(self.carve, self.arch)
        self.assertEqual(len(results), 1)
        # GPU chain should contain gpu1 and gpu2, stopping before bink
        chain = results[0]["gpu_chain"]
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0], [gpu1_off, len(gpu1_bytes)])
        self.assertEqual(chain[1], [gpu2_off, len(gpu2_bytes)])

    def test_main_cli(self):
        self._make_synthetic_texture_blobs(w=512, h=512, fmt=71)
        out_json = os.path.join(self.tmp.name, "out_textures.json")
        argv = ["tex_headers.py", self.carve, "--out", out_json]
        out_buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out_buf):
            tex_headers.main()

        self.assertTrue(os.path.isfile(out_json))
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["w"], 512)
        self.assertEqual(data[0]["h"], 512)
        self.assertEqual(data[0]["fmt_name"], "BC1_UNORM")


if __name__ == "__main__":
    unittest.main(verbosity=2)
