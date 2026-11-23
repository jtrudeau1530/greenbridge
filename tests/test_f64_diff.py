"""CLI and reference checks for tools/f64_diff.py; standard library only."""
import contextlib
import io
import os
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "f64_diff.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import f64_diff as fd


class F64DiffTests(unittest.TestCase):
    def command(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], cwd=ROOT,
                              capture_output=True, text=True, timeout=60)

    def test_cli_self_test(self):
        self.assertEqual(self.command("--self-test").returncode, 0)

    def test_lists_47_supported_forms_and_exclusions(self):
        result = self.command("--list-ops")
        listed = [line for line in result.stdout.splitlines() if line.startswith("v_")]
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(listed), 47)
        self.assertIn("v_div_scale_f64", result.stdout)

    def test_exit_2_for_invalid_cli(self):
        self.assertEqual(self.command("--ops", "v_not_real_f64").returncode, 2)
        self.assertEqual(self.command("--programs", "-1").returncode, 2)

    def test_reference_rounds_all_four_binary64_modes(self):
        half_ulp = Fraction(1, 1 << 53)
        self.assertEqual(fd.round_fraction(Fraction(1) + half_ulp, 0), 0x3FF0000000000000)
        self.assertEqual(fd.round_fraction(Fraction(1) + half_ulp, 1), 0x3FF0000000000001)
        self.assertEqual(fd.round_fraction(Fraction(1) + half_ulp, 2), 0x3FF0000000000000)
        self.assertEqual(fd.round_fraction(Fraction(1) + half_ulp, 3), 0x3FF0000000000000)
        self.assertEqual(fd.round_fraction(-(Fraction(1) + half_ulp), 1), 0xBFF0000000000000)
        self.assertEqual(fd.round_fraction(-(Fraction(1) + half_ulp), 2), 0xBFF0000000000001)

    def test_nan_tolerance_preserves_integer_low_word_after_partial_write(self):
        expected = [0, 0x7FF80000, 0, 0]
        actual = [0, 0xFFFFFFFF, 0, 0]
        self.assertTrue(fd.equivalent(expected, actual, ("int", "f64")))
        self.assertFalse(fd.equivalent([1, *expected[1:]], actual, ("int", "f64")))
        self.assertFalse(fd.equivalent(expected, actual, ("int", "int")))

    def test_exit_77_when_requested_oracle_is_unavailable(self):
        self.assertEqual(self.command("--oracle", "hw").returncode, 77)
        self.assertEqual(self.command("--oracle", "gpu", "--exe", "missing.exe").returncode, 77)
        self.assertEqual(self.command("--oracle", "gpu", "--exe", "missing.exe", "--ops", "not_f64").returncode, 2)

    def test_skips_are_counted_with_reason(self):
        result = self.command("--ops", "v_cvt_f32_f64", "--programs", "0")
        self.assertEqual(result.returncode, 0)
        self.assertIn("1 total", result.stdout)
        self.assertIn("OpFConvert", result.stdout)

    def test_sqrt_preserves_negative_zero(self):
        self.assertEqual(fd.ref_program(["v_sqrt_f64 v[8:9], -0.0"]), [0, 0x80000000, 0, 0])

    def test_rsq_subnormal_is_an_explicit_skip(self):
        lines = ["v_mov_b32 v12, 0x00000001", "v_mov_b32 v13, 0x00000000",
                 "v_rsq_f64 v[8:9], v[12:13]"]
        with tempfile.TemporaryDirectory() as tmp:
            status, reason, _, _ = fd.run_one(lines, "ps_f64_subnormal", tmp)
        self.assertEqual(status, "skip")
        self.assertIn("subnormal", reason)
        self.assertIn("FTZ", reason)

    def test_gpu_rsq_precision_is_counted_as_unjudged(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, reason, _, actual = fd.run_one(["v_rsq_f64 v[8:9], 3.25"],
                                                   "ps_f64_gpu_rsq_precision", tmp, "gpu", {"exe": "unused"})
        self.assertEqual(status, "skip")
        self.assertIn("implementation-defined", reason)
        self.assertIsNone(actual)

    def test_div_fixup_nan_s0_is_skipped_by_default_and_runs_with_known(self):
        setup, body, _, _ = fd.make_case(random.Random(1), "v_div_fixup_f64", known_case="div-fixup-nan-s0")
        lines = setup + body
        with tempfile.TemporaryDirectory() as tmp:
            skipped = fd.run_one(lines, "ps_f64_fixup_default", tmp)
            self.assertEqual(skipped[0], "skip")
            self.assertIn("div-fixup-nan-s0", skipped[1])
            known = fd.run_one(lines, "ps_f64_fixup_known", tmp, include_known=True)
        self.assertEqual(known[0], "mismatch")
        self.assertEqual(fd.words_to_bits(*known[2][:2]), fd.EXP)

    def test_compare_reference_does_not_write_v8_without_cndmask(self):
        for op in ("v_cmp_eq_f64_e64 s20, v[12:13], 0.0",
                   "v_cmp_class_f64_e64 s20, v[12:13], 0x3ff"):
            result = fd.ref_program(["v_mov_b32 v8, 0x12345678", "v_mov_b32 v12, 0x00000000",
                                     "v_mov_b32 v13, 0x3ff00000", op])
            self.assertEqual(result[0], 0x12345678)

    def test_shrink_reuses_prepared_gpu_context(self):
        context = {"exe": "fake"}
        observed = []
        def fake_run_one(lines, name, tmp, oracle, gpu_ctx, include_known=False):
            observed.append(gpu_ctx)
            return ("mismatch" if len(lines) > 1 else "match", "", [], [])
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(fd, "prepare_gpu", side_effect=AssertionError("GPU setup repeated")), \
                mock.patch.object(fd, "run_one", side_effect=fake_run_one):
            reduced = fd.shrink(["first", "second"], [], "gpu", "fake.exe", context, tmp)
        self.assertEqual(reduced, ["first", "second"])
        self.assertTrue(observed)
        self.assertTrue(all(item is context for item in observed))

    def test_known_divergences_only_reinserted_with_known(self):
        normal = self.command("--ops", "v_add_f64", "--programs", "0")
        known = self.command("--ops", "v_add_f64", "--programs", "0", "--no-sweep", "--known", "--all-failures")
        self.assertNotIn("KNOWN_DIVERGENCE", normal.stdout)
        self.assertEqual(known.returncode, 1)
        self.assertIn("MISMATCH:", known.stdout)
        self.assertIn("v_div_fixup_f64", known.stdout)  # the known NaN(S0) case is actually executed

    def test_sabotaged_translator_mismatch_and_shrinking(self):
        lines = ["v_mov_b32 v12, 0x00000000", "v_mov_b32 v13, 0x3ff80000",
                 "v_trunc_f64 v[8:9], v[12:13]"]
        original = fd.translate
        def planted_bug(listing, name, temp):
            out = original(listing, name, temp)
            if any(line.startswith("v_trunc_f64 ") for line in listing): out[0] ^= 1
            return out
        try:
            fd.translate = planted_bug
            with tempfile.TemporaryDirectory() as tmp:
                status, _, expected, actual = fd.run_one(lines, "ps_f64_negative_control", tmp)
                self.assertEqual(status, "mismatch")
                self.assertNotEqual(expected, actual)
                reduced = fd.shrink(lines, expected)
                self.assertEqual(len(reduced), 1)
                self.assertTrue(reduced[0].startswith("v_trunc_f64 "))
        finally:
            fd.translate = original

    def test_cli_exit_1_on_plantable_mismatch(self):
        original = fd.translate
        stdout = io.StringIO()
        try:
            fd.translate = lambda listing, name, temp: [0xDEADBEEF, 0, 0, 0]
            with contextlib.redirect_stdout(stdout):
                code = fd.main(["--ops", "v_trunc_f64", "--programs", "0"])
        finally:
            fd.translate = original
        self.assertEqual(code, 1)
        self.assertIn("minimal program:", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
