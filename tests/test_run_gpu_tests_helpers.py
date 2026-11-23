"""scripts/run_gpu_tests.py: the synthetic container builders, the pixel/frame helpers and the environment flags.

The builders are pure byte and string work and the checks read a fixture file, so no assembler, GPU or game data is
needed: the frames here are made by hand.

    python -I tests/test_run_gpu_tests_helpers.py
"""
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gpu_tests as rgt  # noqa: E402

FLAGS = ("LEONIDA_BARY_FALLBACK", "LEONIDA_F16_INTERP_FALLBACK", "LEONIDA_W64_EMULATE")


class FlagEnvironment:
    """Run a flag parser with only the flag under test set (the others would leak in from the caller)."""

    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self._patch = patch.dict(os.environ, self.values, clear=False)
        self._patch.start()
        for name in FLAGS:
            if name not in self.values:
                os.environ.pop(name, None)
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


class DxbcBuilder(unittest.TestCase):
    def parse(self, data):
        magic = data[:4]
        version, total, count = struct.unpack_from("<III", data, 20)
        offsets = list(struct.unpack_from(f"<{count}I", data, 32))
        return magic, version, total, count, offsets

    def chunks(self, data):
        _, _, total, count, offsets = self.parse(data)
        self.assertEqual(total, len(data))
        return [(data[o:o + 4], struct.unpack_from("<I", data, o + 4)[0],
                 data[o + 8:o + 8 + struct.unpack_from("<I", data, o + 4)[0]]) for o in offsets]

    def test_header_names_the_chunks_and_their_offsets(self):
        dxbc = rgt.make_dxbc([(b"ISG1", b"abcd"), (b"OSG1", b"")])
        magic, version, total, count, offsets = self.parse(dxbc)
        self.assertEqual((magic, version, count), (b"DXBC", 1, 2))
        self.assertEqual(total, 32 + 4 * 2 + (4 + 4 + 4) + (4 + 4 + 0))
        self.assertEqual(offsets, [40, 52])

    def test_payloads_round_trip(self):
        dxbc = rgt.make_dxbc([(b"PSV0", b"\x01\x02"), (b"ISG1", b"\x03")])
        self.assertEqual(self.chunks(dxbc), [(b"PSV0", 2, b"\x01\x02"), (b"ISG1", 1, b"\x03")])

    def test_no_chunks_is_a_bare_header(self):
        dxbc = rgt.make_dxbc([])
        magic, version, total, count, offsets = self.parse(dxbc)
        self.assertEqual((magic, version, total, count, offsets), (b"DXBC", 1, 32, 0, []))


class SignatureChunk(unittest.TestCase):
    def test_row_count_and_row_size(self):
        chunk = rgt.make_sig_chunk([(1, 3, 0, 0xF, 0xF), (0, 3, 1, 0xF, 0xF)])
        count, stride = struct.unpack_from("<II", chunk)
        self.assertEqual((count, stride), (2, 8))
        self.assertEqual(len(chunk), 8 + 2 * 32)

    def test_element_fields_are_packed_where_the_signature_reader_expects_them(self):
        chunk = rgt.make_sig_chunk([(2, 3, 1, 0x1, 0xE)])
        self.assertEqual(struct.unpack_from("<6I", chunk, 8), (0, 0, 0, 2, 3, 1))
        self.assertEqual((chunk[8 + 24], chunk[8 + 25]), (0x1, 0xE))
        self.assertEqual(chunk[8 + 26:8 + 32], b"\0\0\0\0\0\0")

    def test_an_empty_signature_is_just_its_header(self):
        self.assertEqual(rgt.make_sig_chunk([]), struct.pack("<II", 0, 8))


class MeshPsv0(unittest.TestCase):
    def payload(self, **kw):
        chunk = rgt.make_ms_psv0(**kw)
        size, = struct.unpack_from("<I", chunk)
        self.assertEqual(size, len(chunk) - 4)
        return chunk[4:]

    def test_the_fields_the_translator_reads(self):
        r = self.payload(threads_x=128, maxv=64, maxp=64)
        self.assertEqual(len(r), 48)
        self.assertEqual(struct.unpack_from("<HH", r, 12), (64, 64))
        self.assertEqual((r[24], r[27]), (13, 2))   # mesh stage, triangles
        self.assertEqual(struct.unpack_from("<III", r, 36), (128, 1, 1))

    def test_group_shared_and_topology_are_written(self):
        r = self.payload(threads_x=32, maxv=3, maxp=1, topology=1, group_shared=128)
        self.assertEqual(struct.unpack_from("<I", r, 0)[0], 128)
        self.assertEqual(r[27], 1)


class FragmentPassthrough(unittest.TestCase):
    def test_it_is_a_fragment_shader_reading_the_requested_location(self):
        text = rgt.make_fs_passthru(1)
        self.assertIn("OpEntryPoint Fragment", text)
        self.assertIn("OpDecorate %in_v Location 1", text)
        self.assertNotIn("OpDecorate %in_v Location 0", text)
        self.assertIn("OpDecorate %out_v Location 0", text)   # the output stays at location 0

    def test_location_zero_and_one_differ_only_in_the_input_location(self):
        zero, one = rgt.make_fs_passthru(0), rgt.make_fs_passthru(1)
        self.assertEqual(zero.replace("%in_v Location 0", "%in_v Location 1"), one)


class PixelConstCheck(unittest.TestCase):
    def raw(self, pixels):
        fd, path = tempfile.mkstemp(suffix=".raw")
        with os.fdopen(fd, "wb") as f:
            for pixel in pixels:
                f.write(struct.pack("<4I", *pixel))
        self.addCleanup(os.remove, path)
        return path

    def test_an_exact_frame_passes(self):
        want = (1, 2, 3, 4)
        ok, detail = rgt.check_pixel_const(self.raw([want] * (64 * 64)), want)
        self.assertEqual((ok, detail), (True, "4096/4096 pixels exact"))

    def test_the_first_mismatch_is_reported_with_its_pixel(self):
        want = (1, 2, 3, 4)
        pixels = [want] * (64 * 64)
        pixels[64 + 3] = (9, 8, 7, 6)
        ok, detail = rgt.check_pixel_const(self.raw(pixels), want)
        self.assertFalse(ok)
        self.assertIn("1/4096 mismatches; first at (3,1)", detail)
        self.assertIn("0x9", detail)
        self.assertIn("0x1", detail)

    def test_a_short_frame_is_rejected_before_judging_pixels(self):
        ok, detail = rgt.check_pixel_const(self.raw([(1, 2, 3, 4)] * 4095), (1, 2, 3, 4))
        self.assertFalse(ok)
        self.assertEqual(detail, "expected 4096 pixels, got 4095")


class FrameProblem(unittest.TestCase):
    def frame(self, pixel=(0.0, 0.0, 0.0, 1.0)):
        return [pixel] * (64 * 64)

    def test_a_full_finite_frame_has_no_problem(self):
        self.assertEqual(rgt.frame_problem(self.frame()), "")

    def test_a_short_frame_is_named_by_its_pixel_count(self):
        self.assertEqual(rgt.frame_problem(self.frame()[:100]), "expected 4096 pixels, got 100")

    def test_nan_and_infinity_are_both_non_finite(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assertEqual(rgt.frame_problem(self.frame((value, 0.0, 0.0, 1.0))),
                                 "non-finite values in the frame")

    def test_every_channel_is_checked(self):
        for channel in range(4):
            with self.subTest(channel=channel):
                pixel = [0.0, 0.0, 0.0, 1.0]
                pixel[channel] = math.nan
                self.assertIn("non-finite", rgt.frame_problem(self.frame(tuple(pixel))))


class EnvironmentFlags(unittest.TestCase):
    def test_bary_fallback_accepts_the_documented_spellings(self):
        for value in ("1", "true", "TRUE", " yes ", "on"):
            with self.subTest(value=value), FlagEnvironment(LEONIDA_BARY_FALLBACK=value):
                self.assertTrue(rgt.bary_fallback())
        for value in ("", "0", "false", "off", "no", "2"):
            with self.subTest(value=value), FlagEnvironment(LEONIDA_BARY_FALLBACK=value):
                self.assertFalse(rgt.bary_fallback())

    def test_bary_fallback_is_off_when_unset(self):
        with FlagEnvironment():
            self.assertFalse(rgt.bary_fallback())

    def test_w64_emulation_accepts_the_same_spellings(self):
        for value in ("1", "true", "yes", "on"):
            with self.subTest(value=value), FlagEnvironment(LEONIDA_W64_EMULATE=value):
                self.assertTrue(rgt.w64_emulated())
        for value in ("", "0", "false", "no", "off"):
            with self.subTest(value=value), FlagEnvironment(LEONIDA_W64_EMULATE=value):
                self.assertFalse(rgt.w64_emulated())

    def test_f16_interp_fallback_can_be_forced_either_way(self):
        for value in ("1", "true", "yes", "on"):
            with self.subTest(value=value), FlagEnvironment(LEONIDA_F16_INTERP_FALLBACK=value):
                self.assertTrue(rgt.f16_interp_fallback())
        for value in ("0", "false", "no", "off"):
            with self.subTest(value=value), FlagEnvironment(LEONIDA_F16_INTERP_FALLBACK=value):
                self.assertFalse(rgt.f16_interp_fallback())

    def test_f16_interp_fallback_follows_the_device_when_unset(self):
        # Unset means "on where the device can't compile PerVertexKHR" (per_vertex_inputs false).
        for answer, expected in ((False, True), (True, False), (None, False)):
            with self.subTest(device=answer), FlagEnvironment(), \
                    patch.object(rgt, "_device_per_vertex", return_value=answer) as probe:
                self.assertIs(rgt.f16_interp_fallback(), expected)
                probe.assert_called_once_with()

    def test_an_explicit_value_does_not_probe_the_device(self):
        with FlagEnvironment(LEONIDA_F16_INTERP_FALLBACK="0"), \
                patch.object(rgt, "_device_per_vertex") as probe:
            self.assertFalse(rgt.f16_interp_fallback())
        probe.assert_not_called()


class GraphicsOnlyFixtures(unittest.TestCase):
    """The wave64 graphics fixtures are excluded from this runner by name (they have their own runner)."""

    def test_a_w64_graphics_module_contributes_its_fixture_names(self):
        import types
        fake = types.SimpleNamespace(FIXTURES={"ps_w64x_a": "a.asm", "ps_w64x_b": "b.asm"})
        with patch.dict(sys.modules, {"w64_graphics": fake}):
            self.assertEqual(rgt._w64_graphics_only(), frozenset({"ps_w64x_a", "ps_w64x_b"}))

    def test_no_module_means_no_exclusions(self):
        with patch.dict(sys.modules, {"w64_graphics": None}):
            self.assertEqual(rgt._w64_graphics_only(), frozenset())


if __name__ == "__main__":
    unittest.main()
