"""Tests for tools/f32_fuzz.py, the FUZZ-3 float VALU differential fuzzer (no game data, no GPU, no llvm-mc).

    python -I tests/test_f32_fuzz.py
"""
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.dont_write_bytecode = True

import f32_fuzz as ff  # noqa: E402


class ModelTests(unittest.TestCase):
    def test_round_f32_ties_to_even_and_overflow(self):
        from fractions import Fraction
        self.assertEqual(ff.round_f32(Fraction(1) + Fraction(1, 1 << 24)), ff.bits(1.0))          # tie -> even
        self.assertEqual(ff.round_f32(Fraction(1) + Fraction(3, 1 << 24)), ff.bits(1.0) + 2)
        self.assertEqual(ff.round_f32(Fraction(2) ** 128), ff.PINF)
        self.assertEqual(ff.round_f32(Fraction(-1, 3)), 0xBEAAAAAB)

    def test_mad_and_fma_differ_where_rounding_matters(self):
        x, y, z = 0xBC500C59, 0x435A2099, 0x3E3DB015
        self.assertEqual(ff.add(ff.mul(x, y), z), 0xC0256A01)
        self.assertEqual(ff.fma(x, y, z), 0xC0256A00)

    def test_signed_zeros(self):
        self.assertEqual(ff.add(0x80000000, 0x80000000), 0x80000000)
        self.assertEqual(ff.add(ff.bits(1.0), ff.bits(-1.0)), 0)
        self.assertEqual(ff.legacy_mul(0x80000000, ff.NINF), 0)
        self.assertEqual(ff.rounding(ff.bits(-0.25), "ceil"), 0x80000000)

    def test_rtz_f16(self):
        self.assertEqual(ff.round_rtz_f16(ff.bits(65519.0)), 0x7BFF)
        self.assertEqual(ff.round_rtz_f16(ff.NINF), 0xFC00)
        self.assertEqual(ff.round_rtz_f16(ff.bits(2.0 ** -24 * 1.5)), 0x0001)

    def test_cube_major_axis(self):
        # |z| largest and negative: face 5, sc = -x, tc = -y, ma = 2z.
        x, y, z = ff.bits(0.5), ff.bits(0.25), ff.bits(-2.0)
        self.assertEqual(ff.cube("v_cubeid_f32", x, y, z), ff.bits(5.0))
        self.assertEqual(ff.cube("v_cubesc_f32", x, y, z), ff.bits(-0.5))
        self.assertEqual(ff.cube("v_cubetc_f32", x, y, z), ff.bits(-0.25))
        self.assertEqual(ff.cube("v_cubema_f32", x, y, z), ff.bits(-4.0))


class HarnessTests(unittest.TestCase):
    def test_self_test(self):
        self.assertTrue(ff.self_test())

    def test_generation_is_deterministic(self):
        a = ff.gen(random.Random(5), "v_fma_f32")
        b = ff.gen(random.Random(5), "v_fma_f32")
        self.assertEqual(a, b)

    def test_seeded_run_has_no_mismatch(self):
        # A short seeded sweep over every op: the translator rules found by FUZZ-3 stay fixed. Undefined-in-SPIR-V
        # inputs and denormals are counted, never compared.
        counts, by_op, _ = ff.run_suite(400, 3, tuple(ff.OPS), quiet=True)
        self.assertEqual(by_op, {})
        self.assertEqual(counts["mismatch"], 0)
        self.assertGreater(counts["match"], 300)

    def test_v_mac_f32_is_not_fused(self):
        vals = [0xBC500C59, 0x435A2099] + [0] * 6 + [0x3E3DB015]
        status, detail = ff.check("v_mac_f32", vals, [(0, {}), (1, {})], False, [])
        self.assertEqual(status, "match", detail)

    def test_output_modifiers_on_conversions(self):
        vals = [0x00007800] + [0] * 8
        for op, mods in (("v_cvt_f32_ubyte1", ["clamp"]), ("v_cvt_f32_u32", ["mul:4"]),
                         ("v_cvt_f32_i32", ["div:2"])):
            with self.subTest(op=op, mods=mods):
                status, detail = ff.check(op, vals, [(0, {})], True, mods)
                self.assertEqual(status, "match", detail)

    def test_undefined_spirv_is_its_own_class(self):
        vals = [0] + [0] * 8                                 # v_rcp_f32 of +0: hardware +inf, OpFDiv by zero
        status, _ = ff.check("v_rcp_f32", vals, [(0, {})], False, [])
        self.assertEqual(status, "undefined")

    def test_denormal_operand_is_skipped(self):
        vals = [0x00000001, ff.bits(1.0)] + [0] * 7
        status, detail = ff.check("v_add_f32", vals, [(0, {}), (1, {})], False, [])
        self.assertEqual(status, "skip")
        self.assertIn("denormal", detail)


if __name__ == "__main__":
    unittest.main()
