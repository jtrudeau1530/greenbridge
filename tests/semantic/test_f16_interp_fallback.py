"""rdna2spirv.py --f16-interp-fallback: v_interp_*_f16 without PerVertexKHR inputs (SHADER-196).

By default a location any *_f16 interp reads is a PerVertexKHR vec4[3] of the producer's raw words, decoded and
weighted in the PS (test_interp_f16_high.py). MoltenVK can't compile that decoration to MSL ("PerVertexKHR decoration
is not supported in MSL"), so with the flag the producer (VS `exp paramN`, mesh params) also exports each channel's two
f16 halves decoded to f32, low halves at location 29 - 2N and high halves at 30 - 2N, and the PS reads them as ordinary
smooth inputs. Interpolating the decoded halves is the same P0 + i(P1 - P0) + j(P2 - P0). Here:
* the PS has no PerVertexKHR, no barycentric builtin, and reads 1.75 / 9.0 from the decoded-half inputs (the
  per-vertex values of test_interp_f16_high.py, interpolated as Vulkan would), the f32 control and the flat raw read
  unchanged;
* the VS exports the decoded halves of every param at those locations (one-lane interpreter), and drops the ones
  that would land on its own params; the metadata lists both sides;
* a PS whose decoded-half location is also an attribute it reads is rejected rather than mislinked;
* without the flag nothing changes.
The GPU check is runtime/test/shaders/ps_interp_f16_high drawn with vs_f16_pack, which run_gpu_tests.py translates
with the flag where tools/vk_subgroup_info.py reports per_vertex_inputs false (MoltenVK).
"""
import struct
import unittest

import support
import rdna2spirv
from spirv_mini_interp import run_fragment
from test_interp_f16_high import ATTR0, ATTR1, BARY, CTRL, HI, LO, SHADER


def f32(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def support_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def translate(name, lines, fallback=True):
    old = rdna2spirv.F16_HALVES
    rdna2spirv.F16_HALVES = fallback
    try:
        spv = support.translate_lines(name, lines)
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
    finally:
        rdna2spirv.F16_HALVES = old
    return spv, meta


def lerp(values):
    return sum(b * v for b, v in zip(BARY, values))


# vs_f16_pack's export, cut down: param0 = (f16(u) | f16(v) << 16, f16(v) | f16(u) << 16, 0, 1.0), param1 = f32 (u, v)
# with u, v from the vertex index as in vs_ids.
VS = """v_lshlrev_b32_e32 v1, 1, v0
v_and_b32_e32 v1, 2, v1
v_and_b32_e32 v2, 2, v0
v_cvt_f32_u32_e32 v1, v1
v_cvt_f32_u32_e32 v2, v2
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 1.0
exp pos0 v1, v2, v5, v6 done
v_cvt_pkrtz_f16_f32_e32 v7, v1, v2
v_cvt_pkrtz_f16_f32_e32 v8, v2, v1
exp param0 v7, v8, v5, v6
exp param1 v1, v2, v5, v6
s_endpgm
"""


class FallbackPixelShader(unittest.TestCase):
    def setUp(self):
        self.spv, self.meta = translate("ps_f16_fallback", SHADER.splitlines())

    def inputs(self):
        """What Vulkan hands the PS: the decoded halves and the f32 control interpolated, attr0 flat (vertex 0)."""
        return {"%in_f16h0_0": (lerp(LO), 0.0, 0.0, 0.0), "%in_f16h0_1": (lerp(HI), 0.0, 0.0, 0.0),
                "%in_attr0": tuple(f32(w) for w in ATTR0[0]),
                "%in_attr1": tuple(lerp([f32(v[c]) for v in ATTR1]) for c in range(4))}

    def test_no_per_vertex_inputs_or_barycentrics(self):
        self.assertNotIn("PerVertexKHR", self.spv)
        self.assertNotIn("BaryCoordKHR", self.spv)
        self.assertNotIn("FragmentBarycentricKHR", self.spv)
        self.assertNotIn("fragmentShaderBarycentric", self.meta["features"])

    def test_decoded_half_inputs(self):
        self.assertIn("OpDecorate %in_f16h0_0 Location 29", self.spv)
        self.assertIn("OpDecorate %in_f16h0_1 Location 30", self.spv)
        self.assertNotIn("%in_f16h0_0 Flat", self.spv)
        self.assertEqual(self.meta["f16_halves"], {"0": [29, 30]})
        self.assertEqual(self.meta["inputs"], [0, 1, 29, 30])

    def test_values(self):
        words = support.mrt0_words(self.spv, inputs=self.inputs())
        self.assertEqual(f"{words[0] & 0xFFFF:#06x}", "0x3f00")      # 1.75 from the low halves
        self.assertEqual(f"{words[1] & 0xFFFF:#06x}", "0x4880")      # 9.0 from the high halves
        self.assertEqual(f32(words[2]), lerp(CTRL))                   # f32 control: 0.6875
        self.assertEqual(words[3], 0x101)                             # v_interp_mov_f32: vertex 0's raw word

    def test_without_high_both_reads_are_the_low_half(self):
        spv, _ = translate("ps_f16_fallback", SHADER.replace(" high", "").splitlines())
        self.assertNotIn("%in_f16h0_1", spv)
        words = support.mrt0_words(spv, inputs={k: v for k, v in self.inputs().items() if k != "%in_f16h0_1"})
        self.assertEqual([f"{w & 0xFFFF:#06x}" for w in words[:2]], ["0x3f00", "0x3f00"])

    def test_location_clash_is_rejected(self):
        """attr29 read as f32 sits where attr0's low halves go: refuse instead of reading the wrong data."""
        clash = ["v_interp_p1ll_f16 v2, v0, attr0.x", "v_interp_p2_f16 v2, v1, attr0.x, v2",
                 "v_interp_p1_f32_e32 v3, v0, attr29.x", "v_interp_p2_f32_e32 v3, v1, attr29.x",
                 "exp mrt0 v2, v3, v3, v3 done vm", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            translate("ps_f16_clash", clash)

    def test_off_by_default(self):
        spv, meta = translate("ps_f16_fallback", SHADER.splitlines(), fallback=False)
        self.assertIn("OpDecorate %in_attr0 PerVertexKHR", spv)
        self.assertNotIn("%in_f16h", spv)
        self.assertNotIn("f16_halves", meta)


class FallbackVertexShader(unittest.TestCase):
    def test_exports_decoded_halves(self):
        spv, meta = translate("vs_f16_fallback", VS.splitlines())
        for n, lo, hi in ((0, 29, 30), (1, 27, 28)):
            self.assertIn(f"OpDecorate %out_f16h{n}_0 Location {lo}", spv)
            self.assertIn(f"OpDecorate %out_f16h{n}_1 Location {hi}", spv)
        self.assertEqual(meta["f16_halves"], {"0": [29, 30], "1": [27, 28]})
        for vid in range(3):
            out = run_fragment(spv, inputs={"%vertex_index": vid, "%instance_index": 0})
            u, v = float((vid << 1) & 2), float(vid & 2)
            raw = list(out["%out_param0"])      # outputs come back as 32-bit words
            lo = [struct.unpack("<ee", struct.pack("<I", w))[0] for w in raw]
            hi = [struct.unpack("<ee", struct.pack("<I", w))[1] for w in raw]
            self.assertEqual(lo[:2], [u, v])
            self.assertEqual(hi[:2], [v, u])
            self.assertEqual([f32(w) for w in out["%out_f16h0_0"]], lo)
            self.assertEqual([f32(w) for w in out["%out_f16h0_1"]], hi)
            self.assertEqual([f32(w) for w in out["%out_f16h1_0"]], [0.0] * 4)     # f32 (u, v, 0, 1): low halves
            self.assertEqual(list(out["%out_param1"])[:2], [support_bits(u), support_bits(v)])

    def test_halves_never_overlap_the_shaders_params(self):
        """param29 occupies location 29, so param0's halves (29, 30) are left out; nothing else moves."""
        spv, meta = translate("vs_f16_far", ["v_mov_b32_e32 v1, 1.0", "exp pos0 v1, v1, v1, v1 done",
                                             "exp param0 v1, v1, v1, v1", "exp param29 v1, v1, v1, v1",
                                             "s_endpgm"])
        self.assertNotIn("%out_f16h0", spv)
        self.assertNotIn("f16_halves", meta)
        self.assertIn("OpDecorate %out_param29 Location 29", spv)

    def test_off_by_default(self):
        spv, meta = translate("vs_f16_fallback", VS.splitlines(), fallback=False)
        self.assertNotIn("%out_f16h", spv)
        self.assertNotIn("f16_halves", meta)


if __name__ == "__main__":
    unittest.main()
