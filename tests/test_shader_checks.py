"""Exercise the synthetic Vulkan test runner without a Vulkan device."""
import ast
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


GOOD_FMT = struct.pack("<64I", *check.FMT_1010102_WORDS, *[0xEEEEEEEE] * 48)
FULL_WAVE = ((0, "compute subgroup size: 64\n0 mismatches\n"),
             (0, "compute subgroup size: 64\nout[0..7] = 32 16 24 40 8 48 4008636142 4008636142\n"))

# The other wave64-only dump tests (#73's cs_wm_*, the cs_w64x_* cross-half tests): name -> expected out[] words.
WAVE64_DUMPS = {n: w for n, w in check.CS_DUMPS + check.W64X_DUMPS + check.W64X_PARTIAL
                if n not in (*check.WAVE32_DUMPS, "cs_wave64_ballot")}
# A group smaller than a wave can't ask for full subgroups: it asks for subgroup size 64 and reads the device's range.
PARTIAL = {n for n, _ in check.W64X_PARTIAL}
WAVE32_WORDS = {n: w for n, w in check.CS_DUMPS if n in check.WAVE32_DUMPS and n not in ("cs_store", "cs_smem_soffset_imm")}


# every test check.py runs: compute, cs_wavemask, cs_fmt_1010102, the dumps (#73, #152 cross-half, partial), the three
# vk_store_test checks (cs_imgstore, cs_imgload, cs_imgload_d16), pixel tests, the two-mip draws, the gradient samples and three mesh fixtures (skipped here: build.py
# does not build mesh shaders)
TOTAL = (len(check.COMPUTE) + 2 + len(check.CS_DUMPS) + len(check.W64X_DUMPS) + len(check.W64X_PARTIAL) + 3
         + len(check.PIXEL_TESTS) + len(check.CLIP_TESTS) + len(check.BOUNDS) + len(check.GLOBAL_OFFSET)
         + len(check.MIP_TEST_PASSES) + len(check.GRADIENT_TESTS)
         + 8)  # three mesh fixtures, ps_tex, cs_cmpx_class_f32, cs_user_sgprs, cs_cvt_f32_sat and ps_gather_wrap
CVT = {bits: [u, s] for bits, u, s in check.CVT_F32_SAT_CASES}


def cvt_answer(env, wrong_for=None):
    """A device running cs_cvt_f32_sat on the word in VKC_SGPRS (s4 reads 0 without it)."""
    bits = int((env or {}).get("VKC_SGPRS", "0"), 0)
    words = [0xBAD, 0xBAD] if bits == wrong_for else CVT.get(bits, [0, 0])
    return 0, "out[0..7] = " + " ".join(map(str, words + [0xEEEEEEEE] * 6)) + "\n"


class Wave64Checks(unittest.TestCase):
    def run_checks(self, wavemask, ballot, *, missing_ballot=False, emulated=False, fmt=(0, GOOD_FMT)):
        def harness(_tool, shader, *args, env=None):
            name = Path(shader).stem
            if name == "cs_fmt_1010102":
                self.assertEqual(args, ("1", "dump"))
                self.fmt_env, dump = env, Path(env["VKC_DUMP"])
                self.assertFalse(dump.exists())
                if fmt[1] is not None:
                    dump.write_bytes(fmt[1])
                return fmt[0], "out[0..7] = 0 0 0 0 0 0 0 0\n"   # stdout is not the oracle
            if name == "cs_user_sgprs":
                # The seeded words come back only if check.py passed them in VKC_SGPRS.
                seeded = (env or {}).get("VKC_SGPRS") == check.USER_SGPRS_ENV["VKC_SGPRS"]
                words = check.USER_SGPRS_WORDS if seeded else [0] * 4 + [0xEEEEEEEE] * 4
                return 0, "out[0..7] = " + " ".join(map(str, words)) + "\n"
            if name == "cs_cvt_f32_sat":
                return cvt_answer(env)
            self.assertNotIn("VKC_SGPRS", env or {})
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
            if name == "cs_cmpx_class_f32":
                return 0, "out[0..7] = " + " ".join(map(str, check.CMPX_CLASS_WORDS)) + "\n"
            if name == "cs_global_neg_offset":
                self.assertEqual(args[0], "40")
                return 0, "out[0..7] = " + " ".join(map(str, check.GLOBAL_OFFSET[0][2])) + "\n"
            if name in dict(check.BOUNDS):    # the SMEM / raw MUBUF rule tests: their words, then the 0xee fill
                words = (dict(check.BOUNDS)[name] + [0xEEEEEEEE] * 8)[:8]
                return 0, "out[0..7] = " + " ".join(map(str, words)) + "\n"
            if name == "cs_store":
                return 0, "out[0..7] = 1073774847 3758097407 1006649344 4008636142 1073757184 3154130944 0 0\n"
            if name == "cs_smem_soffset_imm":
                return 0, "out[0..7] = " + " ".join(map(str, check.CS_SMEM)) + "\n"
            if name in WAVE32_WORDS:           # wave32 dumps run on any subgroup size
                return 0, "out[0..7] = " + " ".join(map(str, (WAVE32_WORDS[name] + [0] * 8)[:8])) + "\n"
            if name == "cs_workgroup_yz":     # #186: vk_compute_test's 2x3x4 Y/Z histogram line
                return 0, "2x3x4 workgroup Y/Z: 0 mismatches\n"
            if name in ("cs_imgstore", "cs_imgload", "cs_imgload_d16"):   # vk_store_test checks
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
                  patch.object(check, "clip_test", return_value=""),
                  patch.object(check, "pixel_test", return_value=""), patch.object(check, "texture_mip_test", return_value=""),
                  patch.object(check, "bindless_test", return_value=""), patch.object(check, "W64_EMULATED", emulated),
                  patch.object(check, "texture_gradient_a16_test", return_value=""),
                  patch.object(check, "gather_wrap_test", return_value=""),
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
        self.assertIn(f"{TOTAL - 5 - len(WAVE64_DUMPS)} of {TOTAL} pass, {5 + len(WAVE64_DUMPS)} skipped", out)

    def test_full_wave_checks_pass_with_expected_results(self):
        status, out = self.run_checks(*FULL_WAVE)
        self.assertEqual(status, 0, out)
        self.assertRegex(out, r"cs_wave64_ballot +ok")
        self.assertIn(f"{TOTAL - 3} of {TOTAL} pass, 3 skipped", out)

    def test_emulated_wave64_runs_without_a_64_wide_subgroup(self):
        # LEONIDA_W64_EMULATE=1: the shaders were translated with --wave64-emulate, so no wave64 test asks the
        # harness for a 64-wide subgroup, and a 32-wide device runs them all.
        unsupported = 77, "full 64-wide compute subgroups unavailable\n"
        status, out = self.run_checks(unsupported, unsupported, emulated=True)
        self.assertEqual(status, 0, out)
        self.assertRegex(out, r"cs_w64x_wg128_branches +ok")
        self.assertIn(f"{TOTAL - 3} of {TOTAL} pass, 3 skipped", out)

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
                self.assertIn(f"{TOTAL - 4} of {TOTAL} pass, 3 skipped", out)

    def test_fmt_1010102_compares_the_whole_dump(self):
        words = list(struct.unpack("<64I", GOOD_FMT))
        for fmt, detail in (
            ((0, struct.pack("<64I", *words[:15], 0xE00003FF, *words[16:])), "word 15: expected 0xffe00003, got 0xe00003ff"),
            ((0, struct.pack("<64I", *words[:63], 0)), "word 63: expected 0xeeeeeeee, got 0x00000000"),
            ((0, GOOD_FMT[:-4]), "expected 256 dump bytes, got 252"),
            ((0, GOOD_FMT + bytes(4)), "expected 256 dump bytes, got 260"),
            ((0, None), "no VKC_DUMP file written"),
            ((1, GOOD_FMT), "exit 1"),
        ):
            with self.subTest(detail=detail):
                status, out = self.run_checks(*FULL_WAVE, fmt=fmt)
                self.assertEqual(status, 1, out)
                self.assertRegex(out, rf"cs_fmt_1010102 +{detail}")
                self.assertIn(f"{TOTAL - 4} of {TOTAL} pass, 3 skipped", out)

    def test_fmt_1010102_child_env_is_private(self):
        parent = {"VK_DRIVER_FILES": "driver.json", "VKC_OUT_WORDS": "128", "VKC_DUMP": "caller.bin"}
        with patch.dict(check.os.environ, parent):
            status, out = self.run_checks(*FULL_WAVE)
            for key, value in parent.items():
                self.assertEqual(check.os.environ[key], value)
        self.assertEqual(status, 0, out)
        self.assertRegex(out, r"cs_fmt_1010102 +ok")
        self.assertEqual((self.fmt_env["VK_DRIVER_FILES"], self.fmt_env["VKC_OUT_WORDS"]), ("driver.json", "64"))
        self.assertNotEqual(self.fmt_env["VKC_DUMP"], parent["VKC_DUMP"])
        self.assertFalse(Path(self.fmt_env["VKC_DUMP"]).exists())

    def test_missing_wave64_fixture_fails_even_on_wave32(self):
        unsupported = 77, "full 64-wide compute subgroups unavailable\n"
        status, out = self.run_checks(unsupported, unsupported, missing_ballot=True)
        self.assertEqual(status, 1, out)
        self.assertIn("missing cs_wave64_ballot.spv", out)


class FloatToIntSweep(unittest.TestCase):
    """check.cvt_f32_sat_test sends each input in VKC_SGPRS and judges both words of every dispatch."""

    def sweep(self, wrong_for=None):
        sent = []

        def harness(_tool, shader, *_args, env=None):
            self.assertEqual(Path(shader).stem, "cs_cvt_f32_sat")
            sent.append(int(env["VKC_SGPRS"], 0))
            return cvt_answer(env, wrong_for)

        with patch.object(check, "run", side_effect=harness):
            return check.cvt_f32_sat_test("tools", "spirv"), sent

    def test_every_input_is_sent_and_exact(self):
        problem, sent = self.sweep()
        self.assertEqual(problem, "")
        self.assertEqual(sent, [bits for bits, _, _ in check.CVT_F32_SAT_CASES])

    def test_one_wrong_input_fails_and_is_named(self):
        problem, _ = self.sweep(wrong_for=0x4F000000)
        self.assertIn("1 of 16 inputs wrong", problem)
        self.assertIn("0x4f000000", problem)

    def test_inputs_match_the_interpreter_test(self):
        # The GPU sweep and tests/semantic/test_float_to_int.py must judge the same inputs the same way.
        src = (Path(__file__).resolve().parent / "semantic" / "test_float_to_int.py").read_text(encoding="utf-8")
        cases = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(src))
                     if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CASES")
        self.assertEqual(tuple(cases), check.CVT_F32_SAT_CASES)


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

    def test_bitmask_perm_skips_when_the_frame_is_short_below_32_lanes(self):
        out = "device: fake (subgroup 8..8)\nok 64x64\n"
        problem = self.run_pixel("ps_bitmask_perm", 0, out, data=b"short")
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("narrower than 32 lanes", problem)

    def test_bitmask_perm_skips_without_a_subgroup_width(self):
        problem = self.run_pixel("ps_bitmask_perm", 0, "ok 64x64\n", data=self.bitmask_pixels(32))
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("no subgroup width", problem)


class SeededPixelTests(unittest.TestCase):
    """ps_vcc_x2 reads a V# from s[0:3]; check.py hands vk_harness that V# and the heap words behind it (TEST-19)."""

    def run_pixel_test(self, words, code=0, out=""):
        seen = {}

        def harness(_tool, _vs, _fs, raw, env=None):
            seen["env"] = env
            if not code:
                Path(raw).write_bytes(struct.pack("<4I", *words) * (64 * 64))
            return code, out

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ps_vcc_x2.spv").touch()
            with patch.object(check, "run", side_effect=harness):
                problem = check.pixel_test(tmp, tmp, "ps_vcc_x2")
        return problem, seen["env"] or {}

    def test_v_sharp_and_heap_reach_the_harness(self):
        problem, env = self.run_pixel_test((0x11223344, 0x55667788, 0, 0))
        self.assertEqual(problem, "")
        self.assertEqual(env.get("VKH_SGPRS"), "0x1000,0,16,0")
        self.assertEqual(env.get("VKH_HEAP"), "0x1000=0x11223344,0x1004=0x55667788")

    def test_unseeded_result_fails(self):
        problem, _ = self.run_pixel_test((0, 0, 0, 0))
        self.assertIn("4096 of 4096 pixels differ", problem)

    def test_device_without_buffer_addresses_skips(self):
        problem, _ = self.run_pixel_test((), code=77, out="unsupported: VKH_HEAP needs bufferDeviceAddress\n")
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("needs bufferDeviceAddress", problem)


class TextureMipChecks(unittest.TestCase):
    def test_mip_draw_reports_unsupported_device_and_preserves_other_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / ("vk_texture_test" + check.EXE)).touch()
            for shader in (check.TEXTURE_MIP_TEST, *check.TEXTURE_MIP_DEPENDENCIES):
                (directory / (shader + ".spv")).touch()
            with patch.object(check, "run", return_value=(77, "unsupported device")):
                self.assertIsInstance(check.texture_mip_test(tmp, tmp), check.Skip)
            with patch.object(check, "run", return_value=(1, "device lost")):
                failure = check.texture_mip_test(tmp, tmp)
                self.assertNotIsInstance(failure, check.Skip)
                self.assertIn("device lost", failure)

    def test_implicit_mip_checks_every_channel_through_last_pixel(self):
        pixel = struct.pack("<4f", 0, 1, 1, 1)
        image = pixel * (64 * 64)
        self.assertEqual(check.texture_mip_mismatch(image, 1), "")
        for channel in range(4):
            damaged = bytearray(image)
            struct.pack_into("<f", damaged, len(image) - 16 + 4 * channel, 0.5)
            self.assertTrue(check.texture_mip_mismatch(damaged, 1).startswith("(63,63):"))
        self.assertTrue(check.texture_mip_mismatch(image[:-1], 1))


class GradientSampleChecks(unittest.TestCase):
    def test_device_without_min_lod_is_a_skip(self):
        answer = 77, "unsupported: shaderResourceMinLod (the shader clamps its LOD with MinLod)\n"
        with patch.object(check, "run", return_value=answer):
            result = check.texture_gradient_a16_test("tools", "spirv", "ps_texture_grad_cl_a16")
        self.assertIsInstance(result, check.Skip)
        self.assertEqual(result, "SKIP (unsupported: shaderResourceMinLod (the shader clamps its LOD with MinLod))")
        with patch.object(check, "run", return_value=(1, "device lost")):
            failure = check.texture_gradient_a16_test("tools", "spirv", "ps_texture_grad_cl_a16")
        self.assertNotIsInstance(failure, check.Skip)
        self.assertIn("device lost", failure)

class TexelOffsetChecks(unittest.TestCase):
    """ps_texoff (image_sample*_o, image_gather4_lz_o): the judgement of a dump, and the skip without maintenance8."""
    LIMITS = "device: gpu\ntexel offset limits: -64..63, gather -32..31\n"

    @staticmethod
    def dump(words=check.texoff_words):
        return b"".join(struct.pack("<4I", *words(i % 64, i // 64)) for i in range(64 * 64))

    def test_hand_derived_words_of_the_asm_header(self):
        self.assertEqual(check.texoff_words(0, 0), (0x00, 0x0761, 0x0000, 0x0000))
        self.assertEqual(check.texoff_words(20, 50), (0x62, 0x0761, 0x1221, 0x5566))
        self.assertEqual(check.texoff_words(63, 63), (0x77, 0x0761, 0x6776, 0x6677))

    def test_expected_dump_passes_and_wrong_offsets_fail(self):
        self.assertEqual(check.texoff_problem(self.dump(), self.LIMITS), ("", "offsets +16/-32 judged"))
        for what, words in (
            ("offset dropped", lambda x, y: (0x44, *check.texoff_words(x, y)[1:])),
            ("x and y swapped", lambda x, y: (y >> 3 | x >> 3 << 4, *check.texoff_words(x, y)[1:])),
            ("constant offset wrong", lambda x, y: (x >> 3 | y >> 3 << 4, 0x0744, *check.texoff_words(x, y)[2:])),
            ("gather offset dropped", lambda x, y: (*check.texoff_words(x, y)[:2], 0x3443, 0x3344)),
            ("descriptor miss", lambda x, y: (0x08, 0x0808, 0x8888, 0x0000)),
        ):
            with self.subTest(what):
                self.assertIn("pixels differ", check.texoff_problem(self.dump(words), self.LIMITS)[0])
        self.assertIn("pixel count", check.texoff_problem(self.dump()[:-16], self.LIMITS)[0])

    def test_wide_offsets_are_judged_only_inside_the_device_limits(self):
        wide_wrong = self.dump(lambda x, y: (x >> 3 | y >> 3 << 4, 0x4461, *check.texoff_words(x, y)[2:]))
        self.assertIn("pixels differ", check.texoff_problem(wide_wrong, self.LIMITS)[0])
        for out in ("texel offset limits: -8..7, gather -8..7\n", "no limits printed\n"):
            problem, covered = check.texoff_problem(wide_wrong, out)
            self.assertEqual(problem, "")
            self.assertIn("not judged", covered)
            self.assertIn("pixels differ", check.texoff_problem(self.dump(lambda x, y: (0x44, 0x0761, 0, 0)), out)[0])

    def test_device_without_maintenance8_is_a_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ps_texoff.spv").touch()
            answer = 77, "unsupported: VK_KHR_maintenance8 (non-constant texel Offset)\n"
            with patch.object(check, "run", return_value=answer) as run:
                result = check.pixel_test("tools", tmp, "ps_texoff")
            self.assertEqual(run.call_args.kwargs["env"]["VKT_NEED_MAINTENANCE8"], "1")
            self.assertIn("vk_texture_test", run.call_args.args[0])
        self.assertIsInstance(result, check.Skip)
        self.assertEqual(result, "SKIP (unsupported: VK_KHR_maintenance8 (non-constant texel Offset))")


class Wave32LaneDumps(unittest.TestCase):
    """cs_wqm_w32 and cs_w32_lane_wrap need 32 lanes: skipped where the widest subgroup is narrower (llvmpipe: 8)."""
    def run_dump(self, name, code, out):
        want = dict(check.CS_DUMPS)[name]
        with patch.object(check, "run", return_value=(code, out)):
            return check.compute_test("tools", "spirv", name, "1", "dump", expected_words=want,
                                      needs_32_lanes=name in check.WAVE32_LANE_DUMPS)

    @staticmethod
    def words(name, flip=False):
        w = list(dict(check.CS_DUMPS)[name])
        if flip:
            w[0] ^= 1
        return "out[0..7] = " + " ".join(map(str, (w + [0] * 8)[:8])) + "\n"

    def test_lane_dumps_are_wave32_dumps(self):
        self.assertTrue(set(check.WAVE32_LANE_DUMPS) <= set(check.WAVE32_DUMPS))
        self.assertTrue(set(check.WAVE32_LANE_DUMPS) <= set(dict(check.CS_DUMPS)))

    def test_skipped_below_32_lanes_whatever_the_words(self):
        for name in check.WAVE32_LANE_DUMPS:
            for flip in (False, True):
                with self.subTest(name=name, flip=flip):
                    problem = self.run_dump(name, 0, "device: llvmpipe (subgroup 8..8)\n" + self.words(name, flip))
                    self.assertIsInstance(problem, check.Skip)
                    self.assertEqual(problem, "SKIP (subgroup 8 is narrower than 32 lanes)")

    def test_judged_on_32_lanes_and_wider(self):
        for name in check.WAVE32_LANE_DUMPS:
            for dev in ("device: gpu (subgroup 32..32)\n", "device: gpu (subgroup 32..64)\n",
                        "device: gpu (subgroup 8..32)\n", ""):
                with self.subTest(name=name, dev=dev):
                    self.assertEqual(self.run_dump(name, 0, dev + self.words(name)), "")
                    problem = self.run_dump(name, 0, dev + self.words(name, flip=True))
                    self.assertTrue(problem)
                    self.assertNotIsInstance(problem, check.Skip)

    def test_other_dumps_are_not_skipped_on_narrow_devices(self):
        out = "device: llvmpipe (subgroup 8..8)\nout[0..7] = 1 2 3 4 5 6 7 8\n"
        with patch.object(check, "run", return_value=(0, out)):
            problem = check.compute_test("tools", "spirv", "cs_store", "1", "dump", expected_words=check.CS_STORE)
        self.assertTrue(problem)
        self.assertNotIsInstance(problem, check.Skip)

    def test_lane_dumps_request_32_lanes(self):
        # #774: like run_gpu_tests.py, so an 8..32 device (Intel ANV) doesn't judge the dump at a narrower size.
        for name in check.WAVE32_LANE_DUMPS:
            with self.subTest(name=name), patch.object(check, "run", return_value=(0, self.words(name))) as run:
                check.compute_test("tools", "spirv", name, "1", "dump", expected_words=dict(check.CS_DUMPS)[name],
                                   needs_32_lanes=True)
                self.assertEqual(run.call_args.kwargs["env"]["VKC_SUBGROUP_SIZE"], "32")

    def test_other_dumps_keep_the_default_size(self):
        with patch.object(check, "run", return_value=(0, self.words("cs_store"))) as run:
            check.compute_test("tools", "spirv", "cs_store", "1", "dump", expected_words=check.CS_STORE)
        self.assertNotIn("env", run.call_args.kwargs)

    def test_skipped_when_32_lanes_cannot_be_pinned(self):
        for name in check.WAVE32_LANE_DUMPS:
            for flip in (False, True):
                with self.subTest(name=name, flip=flip):
                    out = ("device: gpu (subgroup 8..32)\ncompute subgroup size: 32 not available\n"
                           + self.words(name, flip))
                    problem = self.run_dump(name, 0, out)
                    self.assertIsInstance(problem, check.Skip)
                    self.assertIn("at 32 lanes on request", problem)

    def test_judged_when_32_lanes_are_pinned(self):
        for name in check.WAVE32_LANE_DUMPS:
            with self.subTest(name=name):
                out = "device: gpu (subgroup 8..32)\ncompute subgroup size: 32\n"
                self.assertEqual(self.run_dump(name, 0, out + self.words(name)), "")
                self.assertNotIsInstance(self.run_dump(name, 0, out + self.words(name, flip=True)), check.Skip)


if __name__ == "__main__":
    unittest.main()


class PixelTestRegistry(unittest.TestCase):
    def test_pixel_tests_include_every_pixel_fixture(self):
        # A repeated PIXEL_TESTS assignment silently dropped ps_texoff once; keep the registry explicit.
        for name in ("ps_texoff", "ps_wqm", "ps_scc_mask32"):
            with self.subTest(name=name):
                self.assertIn(name, check.PIXEL_TESTS)
        self.assertEqual(len(check.PIXEL_TESTS), len(set(check.PIXEL_TESTS)))

    def test_pixel_test_runs_the_harness_once(self):
        # A duplicated pixels() line ran the whole shader twice and could hide a first-call failure behind a
        # second-call success.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ps_legacy_mul.spv").touch()
            want = check.PIXEL_BITS["ps_legacy_mul"]
            with patch.object(check, "pixels", return_value=([want], "")) as pixels:
                self.assertEqual(check.pixel_test("tools", tmp, "ps_legacy_mul"), "")
            self.assertEqual(pixels.call_count, 1)
