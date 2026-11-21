"""Tests for tools/dds_png.py, built from synthetic DDS buffers (no game data).

    python -I tests/test_dds_png.py
"""
import contextlib
import importlib.util
import io
import os
import struct
import tempfile
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

DDPF_ALPHAPIXELS = 0x1
DDPF_FOURCC = 0x4
DDPF_RGB = 0x40


def dds_buffer(width, height, payload=b"", *, flags=0, fourcc=b"\x00\x00\x00\x00",
               bit_count=0, masks=(0, 0, 0, 0), mips=0, dxgi=None):
    """A 128-byte DDS header, a DX10 header when dxgi is set, then the payload."""
    header = bytearray(128)
    header[0:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<I", header, 12, height)
    struct.pack_into("<I", header, 16, width)
    struct.pack_into("<I", header, 28, mips)
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<I", header, 80, flags)
    header[84:88] = fourcc
    struct.pack_into("<I", header, 88, bit_count)
    struct.pack_into("<IIII", header, 92, *masks)
    raw = bytes(header)
    if dxgi is not None:
        raw += struct.pack("<IIIII", dxgi, 3, 0, 1, 0)
    return raw + payload


def bc1_block(c0, c1, rows):
    """One BC1 block; rows holds the four per-row index bytes."""
    return struct.pack("<HHI", c0, c1, int.from_bytes(bytes(rows), "little"))


def bc3_block(alpha0, alpha1, alpha_indices, c0=0xFFFF, c1=0x0000, color_bits=0):
    """One BC3 block: explicit alpha endpoints plus 48 index bits, then the BC1 part."""
    alpha = struct.pack("<BB", alpha0, alpha1) + alpha_indices.to_bytes(6, "little")
    return alpha + struct.pack("<HHI", c0, c1, color_bits)


def bc4_block(v0, v1, indices, base=0):
    """One BC4 block starting at struct offset base (BC5 stacks two of them)."""
    return struct.pack("<BB", v0, v1) + indices.to_bytes(6, "little")


def pixel(rgba, width, x, y):
    start = (y * width + x) * 4
    return tuple(rgba[start:start + 4])


def pixels(rgba):
    return [tuple(rgba[i:i + 4]) for i in range(0, len(rgba), 4)]


def png_chunks(png):
    """Split a PNG into (type, payload, stored_crc) tuples without trusting it."""
    assert png[:8] == dds.PNG_SIGNATURE, "missing PNG signature"
    pos = 8
    chunks = []
    while pos < len(png):
        length = struct.unpack_from(">I", png, pos)[0]
        ctype = png[pos + 4:pos + 8]
        payload = png[pos + 8:pos + 8 + length]
        crc = struct.unpack_from(">I", png, pos + 8 + length)[0]
        chunks.append((ctype, payload, crc))
        pos += 12 + length
    return chunks


class HeaderTests(unittest.TestCase):
    def test_legacy_dxt1_header(self):
        info = dds.parse_dds_header(dds_buffer(8, 4, flags=DDPF_FOURCC, fourcc=b"DXT1"))
        self.assertEqual(info["width"], 8)
        self.assertEqual(info["height"], 4)
        self.assertEqual(info["format"], "BC1")
        self.assertIsNone(info["dxgi_format"])
        self.assertEqual(info["mip_count"], 1)
        self.assertEqual(info["data_offset"], 128)

    def test_mip_count_defaults_to_one_when_zero(self):
        self.assertEqual(
            dds.parse_dds_header(dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=b"DXT5", mips=0))["mip_count"],
            1,
        )

    def test_mip_count_is_reported(self):
        self.assertEqual(
            dds.parse_dds_header(dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=b"DXT5", mips=3))["mip_count"],
            3,
        )

    def test_dx10_header_moves_the_payload(self):
        info = dds.parse_dds_header(dds_buffer(16, 16, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=71))
        self.assertEqual(info["format"], "BC1")
        self.assertEqual(info["dxgi_format"], 71)
        self.assertEqual(info["data_offset"], 148)

    def test_fourcc_aliases(self):
        cases = {
            b"DXT1": "BC1", b"DXT3": "BC2", b"DXT5": "BC3",
            b"ATI1": "BC4", b"BC4U": "BC4", b"BC4S": "BC4_SNORM",
            b"ATI2": "BC5", b"BC5U": "BC5", b"BC5S": "BC5_SNORM",
        }
        for fourcc, expected in cases.items():
            with self.subTest(fourcc=fourcc):
                info = dds.parse_dds_header(dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=fourcc))
                self.assertEqual(info["format"], expected)

    def test_canonical_uncompressed_masks(self):
        cases = {
            (0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000): "BGRA8",
            (0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000): "BGRX8",
            (0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000): "RGBA8",
            (0x000000FF, 0x0000FF00, 0x00FF0000, 0x00000000): "RGBX8",
        }
        for masks, expected in cases.items():
            with self.subTest(masks=masks):
                info = dds.parse_dds_header(
                    dds_buffer(
                        2, 2, flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32, masks=masks,
                    )
                )
                self.assertEqual(info["format"], expected)

    def test_alpha_flag_removes_the_alpha_mask(self):
        info = dds.parse_dds_header(
            dds_buffer(
                2, 2, flags=DDPF_RGB, bit_count=32,
                masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000),
            )
        )
        self.assertEqual(info["format"], "BGRX8")

    def test_custom_bitmask_falls_back_to_rgba8(self):
        info = dds.parse_dds_header(
            dds_buffer(
                1, 1, flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32,
                masks=(0x3FF00000, 0x000FFC00, 0x000003FF, 0xC0000000),
            )
        )
        self.assertEqual(info["format"], "RGBA8")

    def test_rejects_short_buffer(self):
        with self.assertRaises(dds.DdsError):
            dds.parse_dds_header(b"DDS " + b"\x00" * 60)

    def test_rejects_bad_magic(self):
        with self.assertRaises(dds.DdsError) as ctx:
            dds.parse_dds_header(b"NOPE" + b"\x00" * 124)
        self.assertIn("magic", str(ctx.exception))

    def test_rejects_zero_dimensions(self):
        with self.assertRaises(dds.DdsError):
            dds.parse_dds_header(dds_buffer(0, 4, flags=DDPF_FOURCC, fourcc=b"DXT1"))

    def test_rejects_unknown_fourcc(self):
        with self.assertRaises(dds.DdsError) as ctx:
            dds.parse_dds_header(dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=b"XXXX"))
        self.assertIn("FourCC", str(ctx.exception))

    def test_rejects_non_32_bit_rgb(self):
        with self.assertRaises(dds.DdsError) as ctx:
            dds.parse_dds_header(
                dds_buffer(4, 4, flags=DDPF_RGB, bit_count=24,
                           masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0))
            )
        self.assertIn("24", str(ctx.exception))

    def test_rejects_zero_rgb_masks(self):
        with self.assertRaises(dds.DdsError):
            dds.parse_dds_header(
                dds_buffer(4, 4, flags=DDPF_RGB, bit_count=32, masks=(0, 0, 0, 0))
            )

    def test_rejects_truncated_dx10(self):
        with self.assertRaises(dds.DdsError) as ctx:
            dds.parse_dds_header(
                dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=b"DX10")
            )
        self.assertIn("DX10", str(ctx.exception))

    def test_rejects_unknown_dxgi_format(self):
        # 98 is BC7_UNORM, which this decoder deliberately does not claim.
        with self.assertRaises(dds.DdsError) as ctx:
            dds.parse_dds_header(dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=98))
        self.assertIn("98", str(ctx.exception))

    def test_rejects_non_contiguous_mask(self):
        with self.assertRaises(dds.DdsError) as ctx:
            dds.parse_dds_header(
                dds_buffer(2, 2, flags=DDPF_RGB, bit_count=32,
                           masks=(0x00FF00FF, 0x0000FF00, 0x000000FF, 0))
            )
        self.assertIn("Non-contiguous", str(ctx.exception))


class UncompressedTests(unittest.TestCase):
    def test_rgba8_is_copied(self):
        payload = bytes((1, 2, 3, 4, 5, 6, 7, 8))
        _, _, rgba = dds.decode_dds(dds_buffer(
            2, 1, payload, flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32,
            masks=(0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000),
        ))
        self.assertEqual(pixels(rgba), [(1, 2, 3, 4), (5, 6, 7, 8)])

    def test_bgra8_swaps_red_and_blue(self):
        _, _, rgba = dds.decode_dds(dds_buffer(
            1, 1, bytes((0x10, 0x20, 0x30, 0x40)), flags=DDPF_RGB | DDPF_ALPHAPIXELS,
            bit_count=32, masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000),
        ))
        self.assertEqual(pixels(rgba), [(0x30, 0x20, 0x10, 0x40)])

    def test_bgrx8_forces_opaque_alpha(self):
        _, _, rgba = dds.decode_dds(dds_buffer(
            1, 1, bytes((0x10, 0x20, 0x30, 0x00)), flags=DDPF_RGB, bit_count=32,
            masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000),
        ))
        self.assertEqual(pixels(rgba), [(0x30, 0x20, 0x10, 0xFF)])

    def test_rgbx8_masks(self):
        _, _, rgba = dds.decode_dds(dds_buffer(
            1, 1, bytes((0x01, 0x02, 0x03, 0x00)), flags=DDPF_RGB, bit_count=32,
            masks=(0x000000FF, 0x0000FF00, 0x00FF0000, 0x00000000),
        ))
        self.assertEqual(pixels(rgba), [(0x01, 0x02, 0x03, 0xFF)])

    def test_r10g10b10a2_uses_the_general_bitmask_path(self):
        px = (1023 << 20) | (512 << 10) | 255 | (3 << 30)
        _, _, rgba = dds.decode_dds(dds_buffer(
            1, 1, struct.pack("<I", px), flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32,
            masks=(0x3FF00000, 0x000FFC00, 0x000003FF, 0xC0000000),
        ))
        # Each channel is scaled by value * 255 / max with round-half-up.
        self.assertEqual(pixels(rgba), [(255, 128, 64, 255)])

    def test_truncated_payload(self):
        with self.assertRaises(dds.DdsError) as ctx:
            dds.decode_dds(dds_buffer(
                2, 1, b"\x00" * 7, flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32,
                masks=(0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000),
            ))
        self.assertIn("Truncated", str(ctx.exception))

    def test_bytearray_input_is_accepted(self):
        buf = dds_buffer(
            1, 1, bytes((9, 8, 7, 6)), flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32,
            masks=(0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000),
        )
        _, _, rgba = dds.decode_dds(bytearray(buf))
        self.assertEqual(pixels(rgba), [(9, 8, 7, 6)])


class Bc1Tests(unittest.TestCase):
    def test_four_color_mode_row(self):
        # White and black endpoints in c0 > c1 mode, indices 0,1,2,3 across the row.
        payload = bc1_block(0xFFFF, 0x0000, [0b11100100, 0, 0, 0])
        _, _, rgba = dds.decode_dds(dds_buffer(
            4, 1, payload, flags=DDPF_FOURCC, fourcc=b"DXT1",
        ))
        self.assertEqual(pixels(rgba), [
            (255, 255, 255, 255),
            (0, 0, 0, 255),
            (170, 170, 170, 255),
            (85, 85, 85, 255),
        ])

    def test_three_color_mode_row(self):
        # c0 <= c1 switches to the three-color palette with a transparent slot.
        payload = bc1_block(0x0000, 0xFFFF, [0b11100100, 0, 0, 0])
        _, _, rgba = dds.decode_dds(dds_buffer(
            4, 1, payload, flags=DDPF_FOURCC, fourcc=b"DXT1",
        ))
        self.assertEqual(pixels(rgba), [
            (0, 0, 0, 255),
            (255, 255, 255, 255),
            (127, 127, 127, 255),
            (0, 0, 0, 0),
        ])

    def test_second_block_lands_at_column_four(self):
        payload = (
            bc1_block(0xFFFF, 0x0000, [0, 0, 0, 0])
            + bc1_block(0xF800, 0x001F, [0, 0, 0, 0])
        )
        width, height, rgba = dds.decode_dds(dds_buffer(
            8, 4, payload, flags=DDPF_FOURCC, fourcc=b"DXT1",
        ))
        self.assertEqual((width, height), (8, 4))
        self.assertEqual(pixel(rgba, 8, 0, 0), (255, 255, 255, 255))
        self.assertEqual(pixel(rgba, 8, 3, 3), (255, 255, 255, 255))
        self.assertEqual(pixel(rgba, 8, 4, 0), (255, 0, 0, 255))
        self.assertEqual(pixel(rgba, 8, 7, 3), (255, 0, 0, 255))


class Bc2Tests(unittest.TestCase):
    def test_alpha_nibbles_scale_to_bytes(self):
        alpha_q = 0x3210  # nibbles 0, 1, 2, 3 from left pixel to right
        payload = struct.pack("<QHHI", alpha_q, 0xFFFF, 0x0000, 0)
        _, _, rgba = dds.decode_dds(dds_buffer(
            4, 1, payload, flags=DDPF_FOURCC, fourcc=b"DXT3",
        ))
        self.assertEqual([p[3] for p in pixels(rgba)], [0, 17, 34, 51])
        self.assertEqual([p[:3] for p in pixels(rgba)], [(255, 255, 255)] * 4)


class Bc3Tests(unittest.TestCase):
    def test_explicit_alpha_interpolation(self):
        # 255 > 0 selects the eight-value alpha ramp.
        indices = 0 | (1 << 3) | (2 << 6) | (7 << 9)
        payload = bc3_block(255, 0, indices)
        _, _, rgba = dds.decode_dds(dds_buffer(
            4, 1, payload, flags=DDPF_FOURCC, fourcc=b"DXT5",
        ))
        self.assertEqual([p[3] for p in pixels(rgba)], [255, 0, 219, 36])
        self.assertEqual([p[:3] for p in pixels(rgba)], [(255, 255, 255)] * 4)


class Bc4Tests(unittest.TestCase):
    def test_unsigned_gradient(self):
        indices = 0 | (1 << 3) | (2 << 6) | (7 << 9)
        payload = bc4_block(200, 100, indices)
        _, _, rgba = dds.decode_dds(dds_buffer(
            4, 1, payload, flags=DDPF_FOURCC, fourcc=b"ATI1",
        ))
        self.assertEqual(pixels(rgba), [
            (200, 200, 200, 255),
            (100, 100, 100, 255),
            (186, 186, 186, 255),
            (114, 114, 114, 255),
        ])

    def test_snorm_negative_endpoint_maps_to_zero(self):
        payload = bc4_block(129, 64, 0 | (1 << 3))
        _, _, rgba = dds.decode_dds(dds_buffer(
            2, 1, payload, flags=DDPF_FOURCC, fourcc=b"BC4S",
        ))
        # 129 as int8 is -127, which maps to 0; 64 maps to round(191.75) = 192.
        self.assertEqual(pixels(rgba), [(0, 0, 0, 255), (192, 192, 192, 255)])


class Bc5Tests(unittest.TestCase):
    def test_two_channel_z_reconstruction(self):
        red = bc4_block(128, 0, 0)
        green = bc4_block(128, 0, 0)
        payload = red + green
        _, _, rgba = dds.decode_dds(dds_buffer(
            1, 1, payload, flags=DDPF_FOURCC, fourcc=b"ATI2",
        ))
        # r = g = 128 is almost exactly normal (0, 0), so z rounds to 255.
        self.assertEqual(pixels(rgba), [(128, 128, 255, 255)])

    def test_z_lut_matches_the_direct_formula(self):
        lut = dds._get_bc5_z_lut()
        self.assertEqual(len(lut), 65536)
        for r in (0, 1, 64, 127, 128, 200, 254, 255):
            for g in (0, 1, 64, 127, 128, 200, 254, 255):
                with self.subTest(r=r, g=g):
                    self.assertEqual(lut[(r << 8) | g], dds._bc5_z_direct(r, g))

    def test_large_image_uses_the_cached_lut(self):
        # 64x64 is 4096 pixels, where the decoder switches to the precomputed table.
        red = bc4_block(200, 100, 0 | (1 << 3) | (2 << 6) | (3 << 9))
        green = bc4_block(100, 200, 0 | (1 << 3) | (2 << 6) | (3 << 9))
        payload = (red + green) * (16 * 16)
        width, height, rgba = dds.decode_dds(dds_buffer(
            64, 64, payload, flags=DDPF_FOURCC, fourcc=b"BC5U",
        ))
        self.assertEqual((width, height), (64, 64))
        self.assertIsNotNone(dds._BC5_Z_LUT)
        lut = dds._get_bc5_z_lut()
        # Red uses the 200 > 100 eight-value ramp, green the 100 <= 200 six-value one.
        for x, expected in enumerate([(200, 100), (100, 200), (186, 120), (171, 140)]):
            r, g = expected
            self.assertEqual(pixel(rgba, 64, x, 0), (r, g, lut[(r << 8) | g], 255))

    def test_snorm_channels(self):
        payload = bc4_block(127, 0, 0 | (1 << 3)) * 2
        _, _, rgba = dds.decode_dds(dds_buffer(
            2, 1, payload, flags=DDPF_FOURCC, fourcc=b"BC5S",
        ))
        self.assertEqual([p[:2] for p in pixels(rgba)], [(255, 255), (128, 128)])
        self.assertEqual([p[3] for p in pixels(rgba)], [255, 255])


class TruncationTests(unittest.TestCase):
    def test_truncated_block_payloads(self):
        cases = {
            b"DXT1": 8, b"DXT3": 16, b"DXT5": 16, b"ATI1": 8, b"ATI2": 16,
        }
        for fourcc, block in cases.items():
            with self.subTest(fourcc=fourcc):
                with self.assertRaises(dds.DdsError) as ctx:
                    dds.decode_dds(dds_buffer(
                        4, 4, b"\x00" * (block - 1), flags=DDPF_FOURCC, fourcc=fourcc,
                    ))
                self.assertIn("Truncated", str(ctx.exception))


class DownscaleTests(unittest.TestCase):
    def rgba_grid(self, width, height):
        buf = bytearray(width * height * 4)
        for i in range(width * height):
            buf[i * 4] = i
            buf[i * 4 + 3] = 255
        return bytes(buf)

    def test_box_filter_4x4(self):
        w, h, out = dds.downscale_rgba(4, 4, self.rgba_grid(4, 4), 2)
        self.assertEqual((w, h), (2, 2))
        # Each output pixel averages a 2x2 block with round-half-up.
        self.assertEqual([p[0] for p in pixels(out)], [3, 5, 11, 13])
        self.assertEqual([p[3] for p in pixels(out)], [255, 255, 255, 255])

    def test_single_column_averages_pairs(self):
        buf = bytearray()
        for value in (0, 10, 20, 30):
            buf += bytes((value, value, value, 255))
        w, h, out = dds.downscale_rgba(1, 4, bytes(buf), 2)
        self.assertEqual((w, h), (1, 2))
        self.assertEqual([p[0] for p in pixels(out)], [5, 25])

    def test_repeated_halving(self):
        w, h, out = dds.downscale_rgba(8, 8, self.rgba_grid(8, 8), 2)
        self.assertEqual((w, h), (2, 2))
        self.assertEqual(len(out), 2 * 2 * 4)

    def test_no_change_needed(self):
        buf = self.rgba_grid(4, 4)
        self.assertEqual(dds.downscale_rgba(4, 4, buf, 4), (4, 4, buf))
        self.assertEqual(dds.downscale_rgba(4, 4, buf, 0), (4, 4, buf))

    def test_invalid_buffer_length(self):
        with self.assertRaises(dds.DdsError):
            dds.downscale_rgba(4, 4, b"\x00" * 8, 2)

    def test_invalid_dimensions(self):
        with self.assertRaises(dds.DdsError):
            dds.downscale_rgba(0, 4, b"", 2)


class PngTests(unittest.TestCase):
    def test_png_structure_and_crcs(self):
        rgba = bytes((1, 2, 3, 4, 5, 6, 7, 8))
        png = dds.encode_png(2, 1, rgba)
        chunks = png_chunks(png)
        self.assertEqual([c[0] for c in chunks], [b"IHDR", b"IDAT", b"IEND"])
        for ctype, payload, crc in chunks:
            expected = zlib.crc32(payload, zlib.crc32(ctype)) & 0xFFFFFFFF
            self.assertEqual(crc, expected, ctype)
        self.assertEqual(chunks[0][1], struct.pack(">IIBBBBB", 2, 1, 8, 6, 0, 0, 0))
        self.assertEqual(chunks[2][1], b"")
        # One filter byte (0 = None) per row, then the RGBA row itself.
        self.assertEqual(zlib.decompress(chunks[1][1]), b"\x00" + rgba)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            dds.encode_png(0, 1, b"")
        with self.assertRaises(ValueError):
            dds.encode_png(2, 1, b"\x00" * 7)

    def test_dds_to_png_roundtrip(self):
        payload = bc1_block(0xFFFF, 0x0000, [0, 0, 0, 0]) * 4
        png = dds.dds_to_png_bytes(dds_buffer(
            8, 8, payload, flags=DDPF_FOURCC, fourcc=b"DXT1",
        ))
        chunks = png_chunks(png)
        self.assertEqual(struct.unpack(">II", chunks[0][1][:8]), (8, 8))
        raw = zlib.decompress(chunks[1][1])
        stride = 8 * 4 + 1
        self.assertEqual(len(raw), stride * 8)
        for row in range(8):
            self.assertEqual(raw[row * stride], 0)
            row_bytes = raw[row * stride + 1:(row + 1) * stride]
            self.assertEqual(row_bytes, b"\xff" * (8 * 4))

    def test_downscaled_dds_to_png(self):
        payload = bc1_block(0xFFFF, 0x0000, [0, 0, 0, 0]) * 16
        png = dds.dds_to_png_bytes(dds_buffer(
            16, 16, payload, flags=DDPF_FOURCC, fourcc=b"DXT1",
        ), max_size=4)
        self.assertEqual(struct.unpack(">II", png_chunks(png)[0][1][:8]), (4, 4))


class CliTests(unittest.TestCase):
    def test_cli_writes_png(self):
        payload = bc1_block(0xFFFF, 0x0000, [0, 0, 0, 0]) * 4
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.dds")
            dst = os.path.join(tmp, "nested", "out.png")
            with open(src, "wb") as f:
                f.write(dds_buffer(8, 8, payload, flags=DDPF_FOURCC, fourcc=b"DXT1"))
            self.assertEqual(dds.main([src, dst]), 0)
            with open(dst, "rb") as f:
                self.assertEqual(f.read(8), dds.PNG_SIGNATURE)

    def test_cli_reports_unsupported_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.dds")
            dst = os.path.join(tmp, "out.png")
            with open(src, "wb") as f:
                f.write(dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=b"XXXX"))
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(dds.main([src, dst]), 1)
            self.assertIn("FourCC", err.getvalue())
            self.assertFalse(os.path.exists(dst))


if __name__ == "__main__":
    unittest.main()
