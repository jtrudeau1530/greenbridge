"""SHADER-177: v_interp reads the barycentric pair through its VGPR source (I = v0, J = v1), and
v_interp_mov_f32 names the vertex with p0 (p10/p20 would be the other two vertices). The single-lane
model returns the attribute's own value, which is only that when the operand really holds the pair,
so anything else is refused instead of silently translating wrong (docs/design.md, PS system inputs):

* v_interp_p1/p2 (f32 and f16 forms) with an I/J source other than v0/v1 -> Unsupported naming the
  register (the corpus has none: 117,001 v0 and 116,997 v1 over 26,503 shaders);
* v_interp_mov_f32 with p10/p20 -> Unsupported naming the select (all 6,926 in the corpus are p0);
* a plain v_mov_b32 copy of v0/v1 is still accepted, including a chain of copies: its single write
  in the whole shader is the mov, so every read that is not undefined garbage reads the barycentric.
  One corpus shader reads its J through a copy of v1 and must keep translating (bit-identical .spv);
* a register written twice (the mov, then anything else) no longer qualifies;
* the standard forms (v0/v1 pairs, p0 mov) translate and compute exactly as before.

On main (before the guards) the refused cases translated silently and passed spirv-val: the operand
was ignored outright, so a second-pair I/J produced the very module of the standard pair.
"""
import struct
import unittest

import support
import rdna2spirv

ATTR0 = (0.5, 0.25, 0.0, 1.0)
FLAT_ATTR1_X = 0x101          # attr1.x read through v_interp_mov only: the provoking vertex's word


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


class InterpPairGuards(unittest.TestCase):
    def test_second_pair_refused(self):
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            support.translate_lines("ps_i", ["v_interp_p1_f32_e32 v2, v4, attr0.x",
                                             "v_interp_p2_f32_e32 v2, v5, attr0.x",
                                             "exp mrt0 v2, v2, v2, v2 done vm",
                                             "s_endpgm"])
        self.assertIn("v_interp_p1_f32 I/J source v4", str(cm.exception))

    def test_second_pair_f16_refused(self):
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            support.translate_lines("ps_i16", ["v_interp_p1ll_f16 v4, v0, attr1.x",
                                               "v_interp_p2_f16 v4, v9, attr1.x, v4",
                                               "exp mrt0 v4, v4, v4, v4 done vm",
                                               "s_endpgm"])
        self.assertIn("v_interp_p2_f16 I/J source v9", str(cm.exception))

    def test_mov_vertex_select_refused(self):
        for sel in ("p10", "p20"):
            with self.subTest(sel=sel):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    support.translate_lines("ps_m", [f"v_interp_mov_f32_e32 v2, {sel}, attr1.x",
                                                     "exp mrt0 v2, v2, v2, v2 done vm",
                                                     "s_endpgm"])
                self.assertIn(f"v_interp_mov_f32 vertex select {sel}: only p0", str(cm.exception))

    def test_copy_of_pair_translates_and_computes(self):
        for chain in (["v_mov_b32_e32 v6, v1"],                    # the corpus form: J read through a copy
                      ["v_mov_b32_e32 v6, v1", "v_mov_b32_e32 v7, v6"]):   # a chain of plain copies
            with self.subTest(chain=len(chain)):
                src = chain[-1].split()[1].rstrip(",")
                lines = chain + ["v_interp_p1_f32_e32 v2, v0, attr0.x",
                                 f"v_interp_p2_f32_e32 v2, {src}, attr0.x",
                                 "exp mrt0 v2, v2, v2, v2 done vm",
                                 "s_endpgm"]
                spv = support.translate_lines("ps_copy", lines)
                words = support.mrt0_words(spv, inputs={"%in_attr0": ATTR0})
                self.assertEqual(words, [f32_bits(ATTR0[0])] * 4)

    def test_copy_written_again_refused(self):
        # Two writes (the copy, then an add) mean the register no longer provably holds the pair.
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            support.translate_lines("ps_copy2", ["v_mov_b32_e32 v6, v1",
                                                 "v_add_f32_e32 v6, v0, v0",
                                                 "v_interp_p1_f32_e32 v2, v0, attr0.x",
                                                 "v_interp_p2_f32_e32 v2, v6, attr0.x",
                                                 "exp mrt0 v2, v2, v2, v2 done vm",
                                                 "s_endpgm"])
        self.assertIn("v_interp_p2_f32 I/J source v6", str(cm.exception))

    def test_copy_with_modifiers_refused(self):
        # A dpp mov is not a plain copy, so the single write does not establish the barycentric.
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            support.translate_lines("ps_copy3", ["v_mov_b32_dpp v6, v1 row_shl:1 row_mask:0xf bank_mask:0xf",
                                                 "v_interp_p1_f32_e32 v2, v0, attr0.x",
                                                 "v_interp_p2_f32_e32 v2, v6, attr0.x",
                                                 "exp mrt0 v2, v2, v2, v2 done vm",
                                                 "s_endpgm"])
        self.assertIn("v_interp_p2_f32 I/J source v6", str(cm.exception))

    def test_standard_pairs_and_p0_mov_unchanged(self):
        spv = support.translate_lines("ps_std", ["v_interp_p1_f32_e32 v2, v0, attr0.x",
                                                 "v_interp_p2_f32_e32 v2, v1, attr0.x",
                                                 "v_interp_mov_f32_e32 v3, p0, attr1.x",
                                                 "exp mrt0 v2, v3, v3, v3 done vm",
                                                 "s_endpgm"])
        words = support.mrt0_words(spv, inputs={"%in_attr0": ATTR0, "%in_attr1": (FLAT_ATTR1_X,) * 4})
        self.assertEqual(words, [f32_bits(ATTR0[0]), FLAT_ATTR1_X, FLAT_ATTR1_X, FLAT_ATTR1_X])
        self.assertIn("OpDecorate %in_attr1 Flat", spv)      # mov-only input stays Flat
        self.assertNotIn("OpDecorate %in_attr0 Flat", spv)

    def test_f16_standard_forms_still_translate(self):
        spv = support.translate_lines("ps_f16", ["v_interp_p1ll_f16 v4, v0, attr1.x",
                                                 "v_interp_p2_f16 v4, v1, attr1.x, v4",
                                                 "exp mrt0 v4, v4, v4, v4 done vm",
                                                 "s_endpgm"])
        self.assertIn("OpEntryPoint Fragment", spv)

    def test_guard_uses_the_copy_analysis(self):
        # Mutation check: with interp_copy_sources returning nothing, the copy case must refuse, so
        # the acceptance above really flows through the tracking and not through a vacuous guard.
        lines = ["v_mov_b32_e32 v6, v1",
                 "v_interp_p1_f32_e32 v2, v0, attr0.x",
                 "v_interp_p2_f32_e32 v2, v6, attr0.x",
                 "exp mrt0 v2, v2, v2, v2 done vm",
                 "s_endpgm"]
        self.assertIn("OpEntryPoint Fragment", support.translate_lines("ps_vc", lines))
        orig = rdna2spirv.interp_copy_sources
        rdna2spirv.interp_copy_sources = lambda insns: {}
        try:
            with self.assertRaises(rdna2spirv.Unsupported):
                support.translate_lines("ps_vc", lines)
        finally:
            rdna2spirv.interp_copy_sources = orig


if __name__ == "__main__":
    unittest.main()
