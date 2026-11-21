"""Exercise the synthetic Vulkan test runner without a Vulkan device."""
import contextlib
import io
import struct
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime" / "test" / "shaders"))
import check


# The other wave64-only dump tests (#73's cs_wm_*, the cs_w64x_* cross-half tests): name -> expected out[] words.
WAVE64_DUMPS = {n: w for n, w in check.CS_DUMPS + check.W64X_DUMPS + check.W64X_PARTIAL
                if n not in (*check.WAVE32_DUMPS, "cs_wave64_ballot")}
# A group smaller than a wave can't ask for full subgroups: it asks for subgroup size 64 and reads the device's range.
PARTIAL = {n for n, _ in check.W64X_PARTIAL}


# every test check.py runs: compute, cs_wavemask, the dumps (#73, #152 cross-half, partial), the two store
# tests, pixel tests, and two mesh fixtures (skipped here: build.py does not build mesh shaders)
TOTAL = (len(check.COMPUTE) + 1 + len(check.CS_DUMPS) + len(check.W64X_DUMPS) + len(check.W64X_PARTIAL) + 2
         + len(check.PIXEL_TESTS) + 4 + len(check.BOUNDS))   # + ms_v5_workgroup, ms_w64x_ballot, ps_tex (#230), ps_ssharp


class Wave64Checks(unittest.TestCase):
    def run_checks(self, wavemask, ballot, *, missing_ballot=False, emulated=False):
        def harness(_tool, shader, *args):
            name = Path(shader).stem
            if name in PARTIAL:
                self.assertNotIn("--require-wave64", args)
                words = (WAVE64_DUMPS[name] + [0] * 8)[:8]
                narrow = wavemask[0] == 77
                return 0, f"device: gpu (subgroup 32..{32 if narrow else 64})\nout[0..7] = " + " ".join(map(str, words)) + "\n"
            if emulated:
                self.assertNotIn("--require-wave64", args)
                if name in WAVE64_DUMPS:
                    words = (WAVE64_DUMPS[name] + [0] * 8)[:8]
                    return 0, "out[0..7] = " + " ".join(map(str, words)) + "\n"
                if name == "cs_wavemask":
                    return 0, "0 mismatches\n"
                if name == "cs_wave64_ballot":
                    return 0, "out[0..7] = 32 16 24 40 8 48 4008636142 4008636142\n"
            if name in WAVE64_DUMPS:
                self.assertIn("--require-wave64", args)
                if wavemask[0] == 77:          # same device as cs_wavemask: no 64-wide subgroups
                    return wavemask
                words = (WAVE64_DUMPS[name] + [0] * 8)[:8]
                return 0, "compute subgroup size: 64\nout[0..7] = " + " ".join(map(str, words)) + "\n"
            if name in ("cs_wavemask", "cs_wave64_ballot"):
                self.assertIn("--require-wave64", args)
                if name == "cs_wavemask":
                    return wavemask
                self.assertFalse(missing_ballot, "missing shader was sent to the harness")
                return ballot
            self.assertNotIn("--require-wave64", args)
            if name == "cs_bounds":
                # 0x12345678, the in-bounds reload, the OOB load's 0, then the untouched 0xEEEEEEEE fill.
                return 0, "out[0..7] = 305419896 305419896 0 4008636142 4008636142 4008636142 4008636142 4008636142\n"
            if name == "cs_store":
                return 0, "out[0..7] = 1073774847 3758097407 1006649344 4008636142 1073757184 3154130944 0 0\n"
            if name == "cs_smem_soffset_imm":
                return 0, "out[0..7] = " + " ".join(map(str, check.CS_SMEM)) + "\n"
            if name == "cs_workgroup_yz":     # #186: vk_compute_test's 2x3x4 Y/Z histogram line
                return 0, "2x3x4 workgroup Y/Z: 0 mismatches\n"
            if name in ("cs_imgstore", "cs_imgload"):
                return 0, "0 bad\n0 bad\n0 bad\n"
            return 0, "0 mismatches\n"

        with tempfile.TemporaryDirectory() as tmp:
            spirv = Path(tmp) / "spirv"
            spirv.mkdir()
            (spirv / "cs_wavemask.spv").touch()
            for name in WAVE64_DUMPS:
                (spirv / f"{name}.spv").touch()
            if not missing_ballot:
                (spirv / "cs_wave64_ballot.spv").touch()
            output = io.StringIO()
            with (patch.object(sys, "argv", ["check.py", tmp]), patch.object(check, "run", side_effect=harness),
                  patch.object(check, "pixel_test", return_value=""), patch.object(check, "bindless_test", return_value=""), patch.object(check, "sampler_test", return_value=""),
                  patch.object(check, "W64_EMULATED", emulated),
                  contextlib.redirect_stdout(output)):
                status = check.main()
        return status, output.getvalue()

    def test_wave32_only_device_skips_full_wave_checks(self):
        unsupported = 77, "full 64-wide compute subgroups unavailable\n"
        status, out = self.run_checks(unsupported, unsupported)
        self.assertEqual(status, 0, out)
        self.assertRegex(out, r"cs_wavemask +SKIP")
        self.assertRegex(out, r"cs_wave64_ballot +SKIP")
        self.assertRegex(out, r"cs_exec64 +ok")
        self.assertIn(f"{TOTAL - 4 - len(WAVE64_DUMPS)} of {TOTAL} pass, {4 + len(WAVE64_DUMPS)} skipped", out)

    def test_full_wave_checks_pass_with_expected_results(self):
        status, out = self.run_checks(
            (0, "compute subgroup size: 64\n0 mismatches\n"),
            (0, "compute subgroup size: 64\nout[0..7] = 32 16 24 40 8 48 4008636142 4008636142\n"),
        )
        self.assertEqual(status, 0, out)
        self.assertRegex(out, r"cs_wave64_ballot +ok")
        self.assertIn(f"{TOTAL - 2} of {TOTAL} pass, 2 skipped", out)

    def test_emulated_wave64_runs_without_a_64_wide_subgroup(self):
        # LEONIDA_W64_EMULATE=1: the shaders were translated with --wave64-emulate, so no wave64 test asks the
        # harness for a 64-wide subgroup, and a 32-wide device runs them all.
        unsupported = 77, "full 64-wide compute subgroups unavailable\n"
        status, out = self.run_checks(unsupported, unsupported, emulated=True)
        self.assertEqual(status, 0, out)
        self.assertRegex(out, r"cs_w64x_wg128_branches +ok")
        self.assertIn(f"{TOTAL - 2} of {TOTAL} pass, 2 skipped", out)

    def test_ballot_mismatches_and_harness_errors_fail(self):
        wavemask = 0, "compute subgroup size: 64\n0 mismatches\n"
        for ballot, detail in (
            ((0, "compute subgroup size: 64\nout[0..7] = 16 16 0 64 0 64 0 0\n"), "exit 0"),
            ((0, "compute subgroup size: 64\nout[0..7] = corrupt\n"), "exit 0"),
            ((0, "out[0..7] = 32 16 24 40 8 48 0 0\n"), "64-wide subgroup was not confirmed"),
            ((1, "Vulkan pipeline failed\n"), "exit 1"),
        ):
            with self.subTest(ballot=ballot):
                status, out = self.run_checks(wavemask, ballot)
                self.assertEqual(status, 1, out)
                self.assertRegex(out, rf"cs_wave64_ballot +{detail}")
                self.assertIn(f"{TOTAL - 3} of {TOTAL} pass, 2 skipped", out)

    def test_missing_wave64_fixture_fails_even_on_wave32(self):
        unsupported = 77, "full 64-wide compute subgroups unavailable\n"
        status, out = self.run_checks(unsupported, unsupported, missing_ballot=True)
        self.assertEqual(status, 1, out)
        self.assertIn("missing cs_wave64_ballot.spv", out)


class MeshWorkgroupChecks(unittest.TestCase):
    def test_every_group_has_its_own_pixel(self):
        pixels = [struct.pack("<4f", 32 * g, 32 * g + 1, 32 * g + 31, 1) for g in range(24)]   # #209's v5
        self.assertEqual(check.check_mesh_v5(b"".join(pixels)), "")
        pixels[6], pixels[2] = pixels[2], pixels[6]
        self.assertIn("2/24 mismatches; first (0,1,0)", check.check_mesh_v5(b"".join(pixels)))
        self.assertIn("384 bytes", check.check_mesh_v5(b"".join(pixels[:-1])))
class PixelDeviceLimits(unittest.TestCase):
    """pixel_test with a fake harness: the device's limits decide SKIP instead of a failure."""

    def run_pixel(self, name, code, out, data=b""):
        def harness(_tool, _vs, _fs, raw, env=None):
            if data:
                Path(raw).write_bytes(data)
            return code, out

        with tempfile.TemporaryDirectory() as tmp:
            spirv = Path(tmp) / "spirv"
            spirv.mkdir()
            (spirv / f"{name}.spv").touch()
            with patch.object(check, "run", side_effect=harness):
                return check.pixel_test(tmp, str(spirv), name)

    @staticmethod
    def bitmask_pixels(subgroup):
        lanes = min(subgroup, 32)
        return b"".join(struct.pack("<4I", i % lanes, (i % lanes) ^ 9, (((i % lanes) & 5) | 16) ^ 1, 0)
                        for i in range(64 * 64))

    def test_shader_needing_missing_barycentrics_skips(self):
        problem = self.run_pixel("ps_bary", 77,
                                 "unsupported: ps_bary.spv reads BaryCoordKHR and the device has no "
                                 "VK_KHR_fragment_shader_barycentric\n")
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("BaryCoordKHR", problem)

    def test_other_harness_errors_still_fail(self):
        problem = self.run_pixel("ps_ops", 1, "FAILED vkCreateDevice -> -9\n")
        self.assertNotIsInstance(problem, check.Skip)
        self.assertIn("vk_harness failed", problem)

    def test_bitmask_perm_skips_below_32_lanes(self):
        out = "device: fake (subgroup 8..8)\nok 64x64\n"
        problem = self.run_pixel("ps_bitmask_perm", 0, out, data=self.bitmask_pixels(8))
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("narrower than 32 lanes", problem)

    def test_bitmask_perm_is_judged_at_32_lanes(self):
        out = "device: fake (subgroup 32..32)\nok 64x64\n"
        self.assertEqual(self.run_pixel("ps_bitmask_perm", 0, out, data=self.bitmask_pixels(32)), "")


class SamplerMipChecks(unittest.TestCase):
    """The S#/T# oracle check.py judges vk_sampler_test with, against hand-worked levels."""

    def test_levels_follow_the_s_and_t_fields(self):
        want = {"point": (0, 1), "point max_lod 0": (0, 0), "point min_lod 1": (1, 1), "point bias +1": (1, 1),
                "point bias -1": (0, 0), "none": (0, 0), "none min_lod 1": (0, 0), "point base 1": (1, 1)}
        self.assertEqual({name for name, _, _ in check.SAMPLER_CASES}, set(want))
        for name, words, view in check.SAMPLER_CASES:
            with self.subTest(name=name):
                self.assertEqual((check.sampler_level(words, view, 0), check.sampler_level(words, view, 1)), want[name])

    def test_s_words_put_the_fields_where_mesa_does(self):
        self.assertEqual(check.ssharp_words(1, min_lod=1, max_lod=15, bias=-1), (0x92, 0xf00100, 0x04003f00, 0))

    @staticmethod
    def image(level_lz, level_implicit):
        return b"".join(struct.pack("<4f", *check.sampler_texel(level_lz, x, y), *check.sampler_texel(level_implicit, x, y))
                        for y in range(64) for x in range(64))

    def test_every_channel_and_the_last_pixel_are_judged(self):
        words, view = check.ssharp_words(1), (0, 1)
        good = self.image(0, 1)
        self.assertEqual(check.sampler_mismatch(good, words, view), "")
        for channel in range(4):
            bad = bytearray(good)
            struct.pack_into("<f", bad, len(good) - 16 + 4 * channel, 0.25)
            self.assertTrue(check.sampler_mismatch(bytes(bad), words, view).startswith("(63,63):"))
        self.assertIn("bytes", check.sampler_mismatch(good[:-16], words, view))

    def test_a_wrong_level_or_swapped_uv_fails_where_predicted(self):
        words, view = check.ssharp_words(1), (0, 1)
        self.assertTrue(check.sampler_mismatch(self.image(1, 1), words, view).startswith("(0,0):"))
        self.assertTrue(check.sampler_mismatch(self.image(0, 0), words, view).startswith("(0,0):"))
        swapped = b"".join(struct.pack("<4f", *check.sampler_texel(0, y, x), *check.sampler_texel(1, y, x))
                           for y in range(64) for x in range(64))
        self.assertTrue(check.sampler_mismatch(swapped, words, view).startswith("(32,0):"))

if __name__ == "__main__":
    unittest.main()
