"""Synthetic checks for tools/interp_hw_probe.py (SHADER-430): the CPU side of the v_interp probe.

The probe itself needs an AMD GPU (tools/hw_oracle). This file tests the exact-model side that the
GPU run is compared against: the Fraction-to-f32/f16 rounding helpers (nearest even, subnormals
kept, infinity on overflow), the bit round trips, the zero comparator, the f32 denormal flush rule,
the LDS attribute-word packing of the two kernel builders, and the CLI's exit-code contract when
no hardware oracle exists. Standard library only; no game files.

    python -I tests/test_interp_hw_probe.py
"""
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction as F
from random import Random
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "interp_hw_probe.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import interp_hw_probe as ihp  # noqa: E402


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f16(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


class RoundingTest(unittest.TestCase):
    def test_exact_values_round_trip(self):
        self.assertEqual(ihp.r32(F(1)), 1.0)
        self.assertEqual(ihp.r32(F(1, 2)), 0.5)
        self.assertEqual(ihp.r16(F(1)), 1.0)

    def test_one_third_rounds_up_to_nearest(self):
        # 1/3 has no exact f32: the larger neighbour is nearer, sign kept.
        self.assertEqual(ihp.r32(F(1, 3)), f32(0x3EAAAAAB))
        self.assertEqual(ihp.r32(-F(1, 3)), -f32(0x3EAAAAAB))
        self.assertEqual(ihp.r16(F(1, 3)), f16(0x3555))

    def test_midpoints_round_to_even(self):
        # a = 1 + 1.5 ulp is the exact midpoint of the 1-ulp-up and 2-ulp-up neighbours, and the
        # 2-ulp neighbour has the even mantissa: that is where the tie lands.
        self.assertEqual(ihp.r32(F(1) + F(1, 2 ** 24)), 1.0)          # midpoint (1.0, 1+2**-23): even keeps 1.0
        self.assertEqual(ihp.r32(F(1) + F(3, 2 ** 24)), f32(0x3F800002))
        self.assertEqual(ihp.r16(F(1) + F(1, 2 ** 11)), 1.0)          # midpoint (1.0, 1+2**-10)

    def test_overflow_gives_signed_infinity(self):
        self.assertEqual(ihp.r32(F(2) ** 128), math.inf)
        self.assertEqual(ihp.r32(-(F(2) ** 128)), -math.inf)
        self.assertEqual(ihp.r16(F(2) ** 16), math.inf)

    def test_subnormals_are_kept(self):
        self.assertEqual(ihp.r32(F(2) ** -149), float.fromhex("0x1p-149"))
        # halfway from 2**-149 to 2**-148 rounds to even: the smaller value.
        self.assertEqual(ihp.r32(F(2) ** -150), 0.0)
        self.assertEqual(ihp.r16(F(1, 3) * F(2) ** -14), f16(0x0155))

    def test_zero(self):
        self.assertEqual(ihp.r32(F(0)), 0.0)


class ComparatorTest(unittest.TestCase):
    def test_same_takes_bit_patterns_with_sign_aware_zero(self):
        self.assertTrue(ihp.same(ihp.f32w(1.5), 1.5))
        self.assertFalse(ihp.same(ihp.f32w(1.5001), 1.5))
        self.assertTrue(ihp.same(ihp.f32w(0.0), 0.0))                 # +0.0 bits vs +0.0
        self.assertFalse(ihp.same(ihp.f32w(0.0), -0.0))               # +0.0 bits vs -0.0
        self.assertTrue(ihp.same(ihp.f16w(0), 0.0, bits=16))
        self.assertTrue(ihp.same(ihp.f16w(-0.0), -0.0, bits=16))   # -0.0 encodes as 0x8000

    def test_finite_helper(self):
        self.assertTrue(ihp.finite(1.0, -0.0, 2.0 ** -130))
        self.assertFalse(ihp.finite(1.0, math.inf))
        self.assertFalse(ihp.finite(0.0, float("nan")))


class FlushTest(unittest.TestCase):
    def test_flush32_only_tiny_nonzero_values_become_signed_zero(self):
        self.assertEqual(ihp.flush32(float.fromhex("0x1p-130"), True), 0.0)
        self.assertEqual(math.copysign(1, ihp.flush32(-float.fromhex("0x1p-130"), True)), -1.0)
        self.assertEqual(ihp.flush32(1.0, True), 1.0)                 # a normal value stays
        self.assertEqual(ihp.flush32(float.fromhex("0x1p-130"), False),
                         float.fromhex("0x1p-130"))                   # flush off keeps the value
        self.assertEqual(ihp.flush32(0.0, True), 0.0)


class KernelBuildersTest(unittest.TestCase):
    def test_gen_word_never_makes_f16_nan_or_inf_halves(self):
        rng = Random(11)
        for _ in range(500):
            word = ihp.gen_word(rng, packed=True)
            lo, hi = word & 0xFFFF, (word >> 16) & 0xFFFF
            self.assertNotEqual(lo & 0x7C00, 0x7C00, hex(word))
            self.assertNotEqual(hi & 0x7C00, 0x7C00, hex(word))

    def test_random_kernel_is_deterministic_per_seed(self):
        a = ihp.random_kernel(Random(7), {"denorm32": 3})[0].lines
        b = ihp.random_kernel(Random(7), {"denorm32": 3})[0].lines
        self.assertEqual(a, b)                                        # the instruction listing

    def test_random_kernel_layout(self):
        kern, info = ihp.random_kernel(Random(3), {"denorm32": 3})
        stores = [l for l in kern.lines if l.startswith("global_store_dword")]
        writes = [l for l in kern.lines if l.startswith("ds_write_b32")]
        self.assertEqual(len(stores), 224)                            # 8 attrs x 4 channels x 7 outputs
        self.assertEqual(len(writes), 160)                            # 3 attribute words each + I/J per lane
        self.assertIn("s_mov_b32 m0, 0", kern.lines)
        self.assertIn("v_interp_p1_f32 v2, v20, attr0.x", kern.lines)
        self.assertEqual(len(info), 4)                                # params, kinds, ij, per-channel tags

    def test_tie_kernel_layout_is_two_stores_per_form(self):
        kern, (groups, lanes, nout) = ihp.tie_kernel({"denorm32": 3})
        stores = [l for l in kern.lines if "global_store_dword" in l]
        p1ll = [l for l in kern.lines if "v_interp_p1ll_f16" in l]
        p2 = [l for l in kern.lines if "v_interp_p2_f16" in l]
        self.assertEqual(nout, 4 * len(groups))
        self.assertEqual(len(stores), nout)                           # p1ll and p2 stored per form per group
        self.assertEqual(len(p1ll), 2 * len(groups))                  # normal and ` high`
        self.assertEqual(len(p2), 2 * len(groups))


class TieCheckTest(unittest.TestCase):
    def test_a_value_that_rounds_differently_once_versus_twice_is_tallied(self):
        # q = 1 + 2**-11 + 2**-24: f16 directly rounds it UP off the tie, while the f32 step ties
        # to even and the f16 re-round then ties DOWN to 1.0.
        h20 = F(2) ** -14
        j = F(8) + F(1, 8192)                      # h20 * j = 2**-11 + 2**-24
        groups = [((1.0, 1.0, 2.0 ** -14), [(1.0, float(j))])]
        lanes = [(0, 1.0, float(j))]
        nout = 4
        q = F(1) + h20 * j
        got = 0x3C01                                # the direct-once f16 result
        res = [ihp.f32w(1.0), got, 0, 0]
        tally, bad = {}, []
        ihp.check_ties(res, (groups, lanes, nout), tally, bad)
        key = ("tie", "p2_f16 rounds once (not f32, then f16)")
        self.assertIn(key, tally)
        self.assertEqual(tally[key][0], 1)
        self.assertEqual(tally[key][1], 0)
        self.assertEqual(bad, [])
        # and the sanity of the case itself: direct f16 differs from the via-f32 double round
        self.assertNotEqual(ihp.f16w(ihp.r16(q)), ihp.f16w(ihp.r16(F(ihp.r32(q)))))

    def test_an_equal_result_is_not_tallied(self):
        h20 = F(1)
        groups = [((1.0, 1.0, 1.0), [(1.0, 0.5)])]
        lanes = [(0, 1.0, 0.5)]
        res = [ihp.f32w(1.0), 0x3C00, 0, 0]         # got == both roundings == 1.0's f16
        tally, bad = {}, []
        ihp.check_ties(res, (groups, lanes, 4), tally, bad)
        self.assertNotIn(("tie", "p2_f16 rounds once (not f32, then f16)"), tally)
        self.assertEqual(bad, [])


class CliTest(unittest.TestCase):
    def run_tool(self, *argv):
        with mock.patch("builtins.print"):
            try:
                return ihp.main(list(argv))
            except SystemExit as e:
                return e.code

    def test_bad_seeds_are_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx:
            ihp.main(["--seeds", "1,x,3"])
        self.assertEqual(ctx.exception.code, 2)

    def test_denorm32_choice_is_enforced(self):
        with self.assertRaises(SystemExit) as ctx:
            ihp.main(["--denorm32", "1"])
        self.assertEqual(ctx.exception.code, 2)

    def test_no_oracle_is_77(self):
        self.assertEqual(self.run_tool("--arch", "gfx1036"), 77)


if __name__ == "__main__":
    unittest.main()
