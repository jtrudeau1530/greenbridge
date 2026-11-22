"""Semantic and translation tests for v_interp_p1lv_f16, f16 --bary-fallback, and stage/attr guards.

Covers the remaining 16-bit interpolation behaviour in tools/rdna2spirv.py not exercised by
test_interp_f16.py (clamp), test_interp_f16_high.py (p1ll low/high), or test_f16_interp_fallback.py:
- v_interp_p1lv_f16: PerVertexKHR low/high half decoding, standalone f32 intermediate, paired v_interp_p2_f16
  f16 narrowing and VOP3 clamp, nearest-even tie rounding, SV_Position ISG1 layout coexistence,
  --f16-interp-fallback (F16_HALVES) execution and location-clash rejection, and SPIR-V negative controls.
- --bary-fallback (BARY_FALLBACK) with v_interp_p1ll_f16, v_interp_p1lv_f16, and v_interp_p2_f16: omits
  PerVertexKHR / BaryCoordKHR, reads ordinary smooth vec4 inputs (where `high` cannot be distinguished),
  places %bary at the lowest free location when raw v0/v1 reads are also present, and yields to
  --f16-interp-fallback when both flags are enabled.
- I-source guards (interp_pair_src) for v_interp_p1lv_f16 and v_interp_p1ll_f16, plus Unsupported("<op> form")
  rejection on non-PS stages (vs_, cs_) and missing `attr` operands for all three *_f16 opcodes.
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support     # noqa: E402

BARY = (0.25, 0.25, 0.5)   # vertex 0, 1, 2 weights (AMD i = 0.25, j = 0.5)
LO = (0.5, 1.5, 2.5)       # exact interpolated value: 1.75 (0x3f00)
HI = (-4.0, 8.0, 16.0)     # exact interpolated value: 9.0  (0x4880)


def pack_f16(lo, hi):
    """Pack two floats as (low_f16 | high_f16 << 16) in a 32-bit word."""
    return struct.unpack("<I", struct.pack("<ee", lo, hi))[0]


def f16_bits(val):
    return struct.unpack("<H", struct.pack("<e", float(val)))[0]


def f32_bits(val):
    return struct.unpack("<I", struct.pack("<f", float(val)))[0]


def lerp(values, bary=BARY):
    return sum(b * v for b, v in zip(bary, values))


def isg1_dxbc(elements):
    """Synthetic DXBC container with one ISG1 chunk: elements = (register, mask, rw_mask, sysv, ctype)."""
    payload = bytearray(struct.pack("<II", len(elements), 8))
    for reg, mask, rw_mask, sysv, ctype in elements:
        elem = bytearray(32)
        struct.pack_into("<6I", elem, 0, 0, 0, 0, sysv, ctype, reg)
        elem[24], elem[25] = mask, rw_mask
        payload += elem
    part = b"ISG1" + struct.pack("<I", len(payload)) + bytes(payload)
    part += bytes((-len(part)) % 4)
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, 0, 1)
    return header + struct.pack("<I", len(header) + 8) + bytes(4) + part


def cs_fssm(rsrc2, group):
    """Minimal compute FSSM header carrying COMPUTE_PGM_RSRC2 and local thread-group dimensions."""
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


# Default compute FSSM (4 user SGPRs, 64x1x1 threads) and PS ISG1 (attr0.xyzw + attr1.x, float32).
CS_FSSM = cs_fssm(0x8088, (64, 1, 1))
PS_ATTR01_DXBC = isg1_dxbc([(0, 0xF, 0xF, 0, 3), (1, 0x1, 0x1, 0, 3)])
PS_POS_ATTR0_DXBC = isg1_dxbc([(0, 0xF, 0x3, 1, 3), (1, 0xF, 0xF, 0, 3)])


def translate_with_flags(name, lines, *, dxbc=support.STUB_DXBC, fssm=None,
                         bary_fallback=False, f16_halves=False):
    """Translate `lines` with explicit BARY_FALLBACK / F16_HALVES flags and return (spvasm, meta)."""
    old_bf, old_fh = rdna2spirv.BARY_FALLBACK, rdna2spirv.F16_HALVES
    rdna2spirv.BARY_FALLBACK = bary_fallback
    rdna2spirv.F16_HALVES = f16_halves
    try:
        spv = support.translate_lines(name, lines, dxbc=dxbc, fssm=fssm)
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
    finally:
        rdna2spirv.BARY_FALLBACK = old_bf
        rdna2spirv.F16_HALVES = old_fh
    return spv, meta


# Primary v_interp_p1lv_f16 PS exercising low half, high half, clamp, standalone f32 intermediate,
# and f32/flat controls on attr0 (PerVertexKHR) and attr1 (smooth f32).
P1LV_SHADER = [
    "v_mov_b32_e32 v10, 0",
    "v_interp_p1lv_f16 v2, v0, attr0.x, v10",
    "v_interp_p2_f16 v2, v1, attr0.x, v2",
    "v_interp_p1lv_f16 v3, v0, attr0.x, v10 high",
    "v_interp_p2_f16 v3, v1, attr0.x, v3 high",
    "v_interp_p1lv_f16 v4, v0, attr0.y, v10",
    "v_interp_p2_f16 v4, v1, attr0.y, v4 clamp",
    "v_interp_p1lv_f16 v5, v0, attr0.y, v10 high",
    "v_interp_p2_f16 v5, v1, attr0.y, v5 high clamp",
    "v_interp_p1lv_f16 v6, v0, attr0.z, v10",       # standalone f32 intermediate
    "v_interp_p1_f32_e32 v7, v0, attr1.x",
    "v_interp_p2_f32_e32 v7, v1, attr1.x",
    "v_interp_mov_f32_e32 v8, p0, attr0.w",
    "exp mrt0 v2, v3, v4, v5",
    "s_nop 0",
    "exp mrt1 v6, v7, v8, v8 done vm",
    "s_endpgm",
]

P1LV_CLAMP_PAIR = [
    "v_mov_b32_e32 v10, 0",
    "v_interp_p1lv_f16 v2, v0, attr0.x, v10",
    "v_interp_p2_f16 v2, v1, attr0.x, v2 clamp",
    "v_interp_p1lv_f16 v3, v0, attr0.x, v10 high",
    "v_interp_p2_f16 v3, v1, attr0.x, v3 high clamp",
    "v_interp_p1lv_f16 v4, v0, attr0.y, v10",
    "v_interp_p2_f16 v4, v1, attr0.y, v4",
    "v_interp_p1lv_f16 v5, v0, attr0.z, v10",
    "exp mrt0 v2, v3, v4, v5 done vm",
    "s_endpgm",
]


class InterpP1lvF16DefaultTests(unittest.TestCase):
    """Default PerVertexKHR translation and execution for v_interp_p1lv_f16."""

    def _default_attrs(self):
        # attr0.x: (lo=1.75, hi=9.0); attr0.y: (lo=-0.25 -> clamp 0.0, hi=1.75 -> clamp 1.0);
        # attr0.z: lo=(1.0, 3.0, 5.0) -> 3.5 as f32 intermediate; attr0.w: raw flat words.
        y_lo = (-1.0, -1.0, 0.5)    # 0.25*(-1.0) + 0.25*(-1.0) + 0.5*0.5 = -0.25 -> clamped to 0.0
        y_hi = (0.5, 1.5, 2.5)      # 1.75 -> clamped to 1.0
        z_lo = (1.0, 3.0, 5.0)      # 3.5
        w_raw = (0x1234, 0x5678, 0x9ABC)
        attr0 = [(pack_f16(lo, hi), pack_f16(yl, yh), pack_f16(zl, 0.0), wr)
                 for lo, hi, yl, yh, zl, wr in zip(LO, HI, y_lo, y_hi, z_lo, w_raw)]
        attr1 = [(f32_bits(x), 0, 0, 0) for x in (-2.0, 2.0, 4.0)]   # 0.25*(-2)+0.25*2+0.5*4 = 2.0
        return {0: attr0, 1: attr1}

    def test_p1lv_low_high_clamp_and_intermediate_end_to_end(self):
        spv, meta = translate_with_flags("ps_p1lv", P1LV_SHADER, dxbc=PS_ATTR01_DXBC)

        self.assertIn("OpCapability FragmentBarycentricKHR", spv)
        self.assertIn('OpExtension "SPV_KHR_fragment_shader_barycentric"', spv)
        self.assertIn("OpDecorate %in_attr0 PerVertexKHR", spv)
        self.assertIn("OpDecorate %bary BuiltIn BaryCoordKHR", spv)
        self.assertIn("OpDecorate %in_attr1 Location 1", spv)
        self.assertNotIn("OpDecorate %in_attr1 PerVertexKHR", spv)
        self.assertNotIn("OpDecorate %in_attr1 Flat", spv)
        self.assertIn("UnpackHalf2x16", spv)
        self.assertEqual(meta["features"], ["fragmentShaderBarycentric"])
        self.assertEqual(meta["inputs"], [0, 1])

        attrs = self._default_attrs()
        out = support.run_fragment(spv, inputs=support.varyings(spv, attrs, BARY))
        mrt0, mrt1 = out["%out_mrt0"], out["%out_mrt1"]
        self.assertEqual([w & 0xFFFF for w in mrt0], [0x3F00, 0x4880, 0x0000, 0x3C00])
        # Standalone v_interp_p1lv_f16 exports the f32 intermediate (3.5), not f16 bits.
        self.assertEqual(f"{mrt1[0]:#010x}", f"{f32_bits(3.5):#010x}")
        self.assertEqual(f"{mrt1[1]:#010x}", f"{f32_bits(2.0):#010x}")
        self.assertEqual(mrt1[2], 0x1234)

    def test_p1lv_clamp_vs_unclamped_and_in_range(self):
        # attr0.x: lo=1.5 (clamp->1.0), hi=-0.25 (clamp->+0.0); attr0.y: lo=0.5 (in-range); attr0.z: lo=3.5 (p1lv)
        attrs = {0: [(pack_f16(1.5, -0.25), pack_f16(0.5, 0.0), pack_f16(3.5, 0.0), 0)] * 3}
        spv_clamp, _ = translate_with_flags("ps_p1lv_clamp", P1LV_CLAMP_PAIR, dxbc=PS_ATTR01_DXBC)
        self.assertIn("FClamp", spv_clamp)
        words_clamp = support.mrt0_words(spv_clamp, inputs=support.varyings(spv_clamp, attrs, BARY))
        self.assertEqual([w & 0xFFFF for w in words_clamp[:3]], [0x3C00, 0x0000, 0x3800])
        self.assertEqual(words_clamp[3], f32_bits(3.5))

        no_clamp_lines = [ln.replace(" clamp", "") for ln in P1LV_CLAMP_PAIR]
        spv_raw, _ = translate_with_flags("ps_p1lv_noclamp", no_clamp_lines, dxbc=PS_ATTR01_DXBC)
        self.assertNotIn("FClamp", spv_raw)
        words_raw = support.mrt0_words(spv_raw, inputs=support.varyings(spv_raw, attrs, BARY))
        self.assertEqual([w & 0xFFFF for w in words_raw[:3]], [f16_bits(1.5), f16_bits(-0.25), 0x3800])

    def test_p1lv_nearest_even_rounding_and_subnormal_f16(self):
        # Midpoint ties:
        # 1.0 (0x3c00) and 1.0 + 2^-10 (0x3c01) averaged at (0.5, 0.5, 0.0) -> 1.0 + 2^-11 -> 0x3c00 (even)
        # 1.0 + 2^-10 (0x3c01) and 1.0 + 2^-9 (0x3c02) averaged -> 1.0 + 3*2^-11 -> 0x3c02 (even)
        # Subnormals 0x0002 and 0x0004 averaged -> 0x0003 (subnormal preserved by f16_bits)
        lines = [
            "v_mov_b32_e32 v10, 0",
            "v_interp_p1lv_f16 v2, v0, attr0.x, v10",
            "v_interp_p2_f16 v2, v1, attr0.x, v2",
            "v_interp_p1lv_f16 v3, v0, attr0.x, v10 high",
            "v_interp_p2_f16 v3, v1, attr0.x, v3 high",
            "v_interp_p1lv_f16 v4, v0, attr0.y, v10",
            "v_interp_p2_f16 v4, v1, attr0.y, v4",
            "exp mrt0 v2, v3, v4, v4 done vm",
            "s_endpgm",
        ]
        h0 = 0x3C00
        h1 = 0x3C01
        h2 = 0x3C02
        attr0 = [
            ((h0 & 0xFFFF) | ((h1 & 0xFFFF) << 16), 0x0002, 0, 0),
            ((h1 & 0xFFFF) | ((h2 & 0xFFFF) << 16), 0x0004, 0, 0),
            ((h0 & 0xFFFF) | ((h1 & 0xFFFF) << 16), 0x0002, 0, 0),
        ]
        spv, _ = translate_with_flags("ps_p1lv_ties", lines, dxbc=PS_ATTR01_DXBC)
        words = support.mrt0_words(spv, inputs=support.varyings(spv, {0: attr0}, (0.5, 0.5, 0.0)))
        self.assertEqual([w & 0xFFFF for w in words[:3]], [0x3C00, 0x3C02, 0x0003])

    def test_p1lv_with_sv_position_in_isg1(self):
        # SV_Position at ISG1 reg 0 occupies v2..v5; v_interp_p1lv_f16 reads v0 and vsrc2=v10 without conflict.
        lines = [
            "v_mov_b32_e32 v10, 0",
            "v_interp_p1lv_f16 v6, v0, attr0.x, v10 high",
            "v_interp_p2_f16 v6, v1, attr0.x, v6 high",
            "v_add_f32_e32 v7, v2, v3",
            "exp mrt0 v6, v7, v6, v7 done vm",
            "s_endpgm",
        ]
        spv, meta = translate_with_flags("ps_p1lv_pos", lines, dxbc=PS_POS_ATTR0_DXBC)
        self.assertEqual(meta["ps_system_inputs"], ["0", "1"])
        attr0 = [(pack_f16(0.5, hi), 0, 0, 0) for hi in HI]
        inputs = support.varyings(spv, {0: attr0}, BARY)
        inputs["%frag_coord"] = (1.25, 2.5, 0.0, 1.0)
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual(words[0] & 0xFFFF, 0x4880)
        self.assertEqual(words[1], f32_bits(3.75))

    def test_negative_controls_p1lv(self):
        attrs = {0: [(pack_f16(1.5, -0.25), pack_f16(0.5, 0.0), pack_f16(3.5, 0.0), 0)] * 3}
        spv, _ = translate_with_flags("ps_p1lv_neg", P1LV_CLAMP_PAIR, dxbc=PS_ATTR01_DXBC)
        inputs = support.varyings(spv, attrs, BARY)
        baseline = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual([w & 0xFFFF for w in baseline[:3]], [0x3C00, 0x0000, 0x3800])

        # Sabotage 1: force high-half extract (component 1) to low half (component 0).
        bad_half = re.sub(r"OpCompositeExtract %float (%\S+) 1", r"OpCompositeExtract %float \1 0", spv)
        self.assertNotEqual(bad_half, spv)
        self.assertNotEqual(support.mrt0_words(bad_half, inputs=inputs), baseline)

        # Sabotage 2: drop FClamp by replacing FClamp %v %f0 %f1 with FMax %v %f0.
        bad_clamp = re.sub(r"OpExtInst %float %glsl FClamp (%\S+) %f0 %f1", r"OpExtInst %float %glsl FMax \1 %f0", spv)
        self.assertNotEqual(bad_clamp, spv)
        self.assertNotEqual(support.mrt0_words(bad_clamp, inputs=inputs), baseline)


class InterpP1lvF16HalvesFallbackTests(unittest.TestCase):
    """--f16-interp-fallback (F16_HALVES) behaviour with v_interp_p1lv_f16."""

    def test_p1lv_f16_halves_translation_and_execution(self):
        spv, meta = translate_with_flags("ps_p1lv_fh", P1LV_CLAMP_PAIR,
                                         dxbc=PS_ATTR01_DXBC, f16_halves=True)
        self.assertNotIn("PerVertexKHR", spv)
        self.assertNotIn("BaryCoordKHR", spv)
        self.assertNotIn("FragmentBarycentricKHR", spv)
        self.assertNotIn("fragmentShaderBarycentric", meta["features"])
        self.assertIn("OpDecorate %in_f16h0_0 Location 28", spv)
        self.assertIn("OpDecorate %in_f16h0_1 Location 29", spv)
        self.assertNotIn("OpDecorate %in_f16h0_0 Flat", spv)
        self.assertEqual(meta["f16_halves"], {"0": [28, 29]})
        self.assertEqual(meta["inputs"], [28, 29])

        inputs = {
            "%in_f16h0_0": (lerp(LO), 0.5, 3.5, 0.0),   # attr0 low: 1.75 (clamped->1.0), 0.5, 3.5 (p1lv f32)
            "%in_f16h0_1": (lerp(HI), 0.0, 0.0, 0.0),   # attr0 high: 9.0 (clamped->1.0)
        }
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual([w & 0xFFFF for w in words[:3]], [0x3C00, 0x3C00, 0x3800])
        self.assertEqual(words[3], f32_bits(3.5))

    def test_p1lv_f16_halves_location_clash_rejected(self):
        clash = [
            "v_mov_b32_e32 v10, 0",
            "v_interp_p1lv_f16 v2, v0, attr0.x, v10",
            "v_interp_p2_f16 v2, v1, attr0.x, v2",
            "v_interp_p1_f32_e32 v3, v0, attr29.x",
            "v_interp_p2_f32_e32 v3, v1, attr29.x",
            "exp mrt0 v2, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate_with_flags("ps_p1lv_fh_clash", clash, f16_halves=True)
        self.assertIn("decoded-half location", str(cm.exception))


class InterpF16BaryFallbackTests(unittest.TestCase):
    """--bary-fallback (BARY_FALLBACK) translation and execution for *_f16 ops."""

    def test_bary_fallback_for_p1ll_p1lv_p2_uses_smooth_inputs_without_per_vertex(self):
        lines = [
            "v_mov_b32_e32 v10, 0",
            "v_interp_p1ll_f16 v2, v0, attr0.x",
            "v_interp_p2_f16 v2, v1, attr0.x, v2",
            "v_interp_p1lv_f16 v3, v0, attr0.y, v10 high",
            "v_interp_p2_f16 v3, v1, attr0.y, v3 high clamp",
            "v_interp_p1lv_f16 v4, v0, attr0.z, v10",       # standalone f32 intermediate
            "exp mrt0 v2, v3, v4, v2 done vm",
            "s_endpgm",
        ]
        spv, meta = translate_with_flags("ps_f16_bf", lines, dxbc=PS_ATTR01_DXBC, bary_fallback=True)

        self.assertNotIn("PerVertexKHR", spv)
        self.assertNotIn("BaryCoordKHR", spv)
        self.assertNotIn("FragmentBarycentricKHR", spv)
        self.assertNotIn("SPV_KHR_fragment_shader_barycentric", spv)
        self.assertNotIn("%bary", spv)
        self.assertIn("OpDecorate %in_attr0 Location 0", spv)
        self.assertNotIn("OpDecorate %in_attr0 Flat", spv)
        self.assertEqual(meta["features"], [])
        self.assertEqual(meta["inputs"], [0])
        self.assertNotIn("bary_fallback_input", meta)
        self.assertNotIn("f16_halves", meta)

        # Under --bary-fallback there are no per-vertex inputs, so high reads the same smooth f32 channel.
        attrs = {0: [(f32_bits(x), f32_bits(y), f32_bits(z), 0)
                     for x, y, z in zip((0.5, 1.5, 2.5), (-1.0, -1.0, 0.5), (1.0, 3.0, 5.0))]}
        words = support.mrt0_words(spv, inputs=support.varyings(spv, attrs, BARY))
        self.assertEqual(words[0] & 0xFFFF, 0x3F00)      # 1.75
        self.assertEqual(words[1] & 0xFFFF, 0x0000)      # -0.25 clamped to +0.0
        self.assertEqual(words[2], f32_bits(3.5))        # standalone p1lv f32 intermediate

        # Confirm `high` and non-`high` read identically under --bary-fallback.
        lines_same = [
            "v_mov_b32_e32 v10, 0",
            "v_interp_p1ll_f16 v2, v0, attr0.x",
            "v_interp_p2_f16 v2, v1, attr0.x, v2",
            "v_interp_p1lv_f16 v3, v0, attr0.x, v10 high",
            "v_interp_p2_f16 v3, v1, attr0.x, v3 high",
            "exp mrt0 v2, v3, v2, v3 done vm",
            "s_endpgm",
        ]
        spv_same, _ = translate_with_flags("ps_f16_bf_same", lines_same, dxbc=PS_ATTR01_DXBC, bary_fallback=True)
        words_same = support.mrt0_words(spv_same, inputs={"%in_attr0": (1.75, 0.0, 0.0, 0.0)})
        self.assertEqual([w & 0xFFFF for w in words_same[:2]], [0x3F00, 0x3F00])

    def test_bary_fallback_with_raw_bary_read_and_sparse_attr(self):
        for attr_idx, want_bary_loc in ((0, 1), (30, 0)):
            with self.subTest(attr=attr_idx):
                lines = [
                    "v_mov_b32_e32 v10, 0",
                    f"v_interp_p1lv_f16 v2, v0, attr{attr_idx}.x, v10",
                    f"v_interp_p2_f16 v2, v1, attr{attr_idx}.x, v2",
                    "v_mov_b32_e32 v4, v0",
                    "v_mov_b32_e32 v5, v1",
                    "exp mrt0 v2, v4, v5, v2 done vm",
                    "s_endpgm",
                ]
                spv, meta = translate_with_flags(f"ps_f16_bf_raw_{attr_idx}", lines, bary_fallback=True)
                self.assertNotIn("PerVertexKHR", spv)
                self.assertNotIn("BaryCoordKHR", spv)
                self.assertNotIn("FragmentBarycentricKHR", spv)
                self.assertIn(f"OpDecorate %in_attr{attr_idx} Location {attr_idx}", spv)
                self.assertIn(f"OpDecorate %bary Location {want_bary_loc}", spv)
                self.assertIsNone(re.search(r"OpDecorate %bary (Flat|NoPerspective)", spv))
                self.assertEqual(meta["bary_fallback_input"], {"location": want_bary_loc})
                self.assertEqual(meta["ps_system_inputs"], ["bary_i", "bary_j"])

                words = support.mrt0_words(spv, inputs={
                    f"%in_attr{attr_idx}": (1.75, 0.0, 0.0, 0.0),
                    "%bary": BARY,
                })
                self.assertEqual(words[0] & 0xFFFF, 0x3F00)
                self.assertEqual(words[1], f32_bits(BARY[1]))
                self.assertEqual(words[2], f32_bits(BARY[2]))

    def test_f16_halves_takes_precedence_over_bary_fallback(self):
        spv, meta = translate_with_flags("ps_both_flags", P1LV_CLAMP_PAIR,
                                         dxbc=PS_ATTR01_DXBC, bary_fallback=True, f16_halves=True)
        self.assertNotIn("PerVertexKHR", spv)
        self.assertIn("OpDecorate %in_f16h0_0 Location 28", spv)
        self.assertIn("OpDecorate %in_f16h0_1 Location 29", spv)
        self.assertEqual(meta["f16_halves"], {"0": [28, 29]})


class InterpF16GuardsAndRejectionsTests(unittest.TestCase):
    """interp_pair_src guards and Unsupported('<op> form') stage/attr rejections."""

    def test_p1lv_and_p1ll_i_source_guards(self):
        bad_cases = [
            ("v_interp_p1lv_f16 I source v4", [
                "v_mov_b32_e32 v10, 0",
                "v_interp_p1lv_f16 v2, v4, attr0.x, v10",
                "v_interp_p2_f16 v2, v1, attr0.x, v2",
                "exp mrt0 v2, v2, v2, v2 done vm",
                "s_endpgm",
            ]),
            ("v_interp_p1lv_f16 I source v1", [
                "v_mov_b32_e32 v10, 0",
                "v_interp_p1lv_f16 v2, v1, attr0.x, v10",
                "v_interp_p2_f16 v2, v1, attr0.x, v2",
                "exp mrt0 v2, v2, v2, v2 done vm",
                "s_endpgm",
            ]),
            ("v_interp_p1ll_f16 I source v1", [
                "v_interp_p1ll_f16 v2, v1, attr0.x",
                "v_interp_p2_f16 v2, v1, attr0.x, v2",
                "exp mrt0 v2, v2, v2, v2 done vm",
                "s_endpgm",
            ]),
            # An I source holding a zeroed v0 ("v_mov v0, 0; v_mov v6, v0") is a computed I of 0, which the translator
            # models since #1545 (computed I/J barycentrics) instead of rejecting; it is no longer a bad case here.
            ("v_interp_p2_f16 J source v0", [
                "v_mov_b32_e32 v10, 0",
                "v_interp_p1lv_f16 v2, v0, attr0.x, v10",
                "v_interp_p2_f16 v2, v0, attr0.x, v2",
                "exp mrt0 v2, v2, v2, v2 done vm",
                "s_endpgm",
            ]),
        ]
        for msg, lines in bad_cases:
            with self.subTest(msg=msg):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    translate_with_flags("ps_guard_bad", lines, dxbc=PS_ATTR01_DXBC)
                self.assertIn(msg, str(cm.exception))

    def test_p1lv_accepts_saved_copy_of_v0_even_if_v0_later_zeroed(self):
        lines = [
            "v_mov_b32_e32 v6, v0",
            "v_mov_b32_e32 v7, v1",
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v10, 0",
            "v_interp_p1lv_f16 v2, v6, attr0.x, v10 high",
            "v_interp_p2_f16 v2, v7, attr0.x, v2 high",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv, _ = translate_with_flags("ps_p1lv_copy", lines, dxbc=PS_ATTR01_DXBC)
        attrs = {0: [(pack_f16(lo, hi), 0, 0, 0) for lo, hi in zip(LO, HI)]}
        words = support.mrt0_words(spv, inputs=support.varyings(spv, attrs, BARY))
        self.assertEqual(words[0] & 0xFFFF, 0x4880)

    def test_non_ps_stages_rejected_for_all_f16_interp_ops(self):
        ops = [
            ("v_interp_p1ll_f16", "v_interp_p1ll_f16 v2, v0, attr0.x"),
            ("v_interp_p1lv_f16", "v_interp_p1lv_f16 v2, v0, attr0.x, v3"),
            ("v_interp_p2_f16", "v_interp_p2_f16 v2, v1, attr0.x, v2"),
        ]
        for op, insn in ops:
            with self.subTest(stage="vs", op=op):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    translate_with_flags(f"vs_{op}", [insn, "exp pos0 v2, v2, v2, v2 done", "s_endpgm"])
                self.assertIn(f"{op} form", str(cm.exception))
            with self.subTest(stage="cs", op=op):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    translate_with_flags(f"cs_{op}", [insn, "s_endpgm"], fssm=CS_FSSM)
                self.assertIn(f"{op} form", str(cm.exception))

    def test_missing_attr_operand_rejected_for_all_f16_interp_ops(self):
        ops = [
            ("v_interp_p1ll_f16", "v_interp_p1ll_f16 v2, v0, v3"),
            ("v_interp_p1lv_f16", "v_interp_p1lv_f16 v2, v0, v3, v4"),
            ("v_interp_p2_f16", "v_interp_p2_f16 v2, v1, v3, v2"),
        ]
        for op, insn in ops:
            with self.subTest(op=op):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    translate_with_flags(f"ps_noattr_{op}",
                                         [insn, "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"],
                                         dxbc=PS_ATTR01_DXBC)
                self.assertIn(f"{op} form", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
