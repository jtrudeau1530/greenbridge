"""SHADER-10: wave64 vertex/pixel shaders on 32-wide subgroups. Tier classification and partition invariance.

tools/wave64_halves.py decides from the code whether a wave64 VS/PS/GS gives the same results when its wave is cut
into two 32-lane halves (the criterion of docs/design.md, "Wave64 graphics: partition invariance"). Two kinds of
check, both on hand-written listings (no llvm-mc, GPU or game data):

  * Classification: each program is translated, and the tier in its metadata (wave64_graphics) and the reasons
    are asserted. The programs are the idioms of the research note (skip regions, quad masks, waterfall loops,
    wave-aggregated atomics, readfirstlane/readlane, lane counts, lane ids, ...), including the near misses that
    must be tier C.
  * Execution: vertex shaders that use integer arithmetic only run on the strict wave interpreter
    (spirv_wave_interp.py) as one wave of 64 lanes and as two waves of 32 lanes, with vertex index v for global lane
    v (so a lane's data does not depend on how the wave is cut). Tier A and B programs must export the same words
    both ways and the words the header derives; tier C programs must differ in the words the test names, which
    shows the detector is right to refuse them.

The GPU twin of the execution half is runtime/test/shaders/w64_graphics.py (pixel shaders, one subgroup size forced
to 64 and to 32 on the same GPU).
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401  (sets the tools path)
from support import translate_lines, translate_listing  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402

LITERAL_OPS = ("exp", "buffer_", "global_", "ds_", "image_")
VOP3_ONLY = ("v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32", "v_readlane_b32", "v_writelane_b32", "v_lshl_add_u32",
             "v_cmp_", "v_cndmask_b32_e64", "v_add_nc_u32_e64")


def size(text):
    """Byte size of an instruction for the fake encoding the translator's branch offsets are placed with."""
    op = text.split()[0]
    if op.startswith(LITERAL_OPS) or op.endswith("_e64") or op in VOP3_ONLY[:5] or "0x" in text:
        return 8
    if op.startswith("v_cmp") and not op.endswith("_e32") and "vcc" not in text:
        return 8
    return 4


def listing(name, lines):
    out = f"; {name} synthetic\n"
    for text in lines:
        out += f"\t{text} ; encoding: [{','.join(['0x00'] * size(text))}]\n"
    return out.encode()


def meta_of(name, lines):
    """(tier, reasons, idioms) of a translated wave64 program, from the metadata the translator writes."""
    asm = translate_listing(name, listing(name, lines))
    g = rdna2spirv.translate.last_gen
    w = g.w64g
    return (w["tier"], w["reasons"], w["idioms"]) if w else (None, [], []), asm


PRO = ["s_mov_b64 s[10:11], exec"]           # the program is wave64 (a 64-bit op on EXEC)
LANE_ID = ["v_mbcnt_lo_u32_b32 v10, -1, 0", "v_mbcnt_hi_u32_b32 v10, -1, v10"]   # the lane id, 0..63 in the wave
EPI_VS = ["v_mov_b32_e32 v5, 0", "exp pos0 v5, v5, v5, v5 done", "exp param0 v6, v7, v5, v5", "s_endpgm"]
EPI_PS = ["exp mrt0 v6, v7, v5, v5 done vm", "s_endpgm"]


def vs(body, prologue=PRO):
    return prologue + ["v_mov_b32_e32 v6, 0", "v_mov_b32_e32 v7, 0"] + body + EPI_VS


def ps(body):
    return PRO + ["v_mov_b32_e32 v6, 0", "v_mov_b32_e32 v7, 0", "v_mov_b32_e32 v5, 0"] + body + EPI_PS[:-1] + \
        ["s_endpgm"]


# ---- execution of vertex shaders: 64 lanes against two halves of 32 --------------------------------------------
def run_cut(asm, wave, push=None):
    """Outputs of every lane of a 64-lane program run as 64/wave waves of `wave` lanes (vertex index = global lane)."""
    words = []
    for half in range(64 // wave):
        res = wi.run_wave(asm, wave=wave, push_constants=push,
                          builtins={"VertexIndex": lambda lane, base=half * wave: base + lane,
                                     "InstanceIndex": lambda lane: 0})
        words += [o and tuple(sorted(o.items())) for o in res["outputs"]]
    return words


def out_param(words, k=0):
    """Per lane, the words of param `k`'s output as a tuple (None for a lane that wrote nothing)."""
    result = []
    for w in words:
        d = dict(w) if w else {}
        result.append(next((v for name, v in d.items() if name.endswith(f"param{k}")), None))
    return result


class Classification(unittest.TestCase):
    def tier(self, name, lines, tier, reasons=None, idioms=None, contains=None):
        (t, r, i), _ = meta_of(name, lines)
        self.assertEqual(t, tier, f"{name}: tier {t}, reasons {r}, idioms {i}")
        if reasons is not None:
            self.assertEqual(sorted(r), sorted(reasons), name)
        if contains is not None:
            self.assertTrue(set(contains) <= set(r), f"{name}: reasons {r} lack {contains}")
        if idioms is not None:
            self.assertEqual(sorted(i), sorted(idioms), name)

    # -- tier A
    def test_a_skip_wqm_quad(self):
        self.tier("ps_a_skip", ps([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4",
            "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execz 1", "v_mul_u32_u24_e32 v6, 7, v0",
            "s_or_b64 exec, exec, s[12:13]", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_wqm_b64 s[14:15], vcc",
            "v_cndmask_b32_e64 v7, 0, 1, s[14:15]"]), "A", [], [])

    def test_a_vccz_narrowed(self):
        self.tier("ps_a_vccz", ps([
            "v_cmp_eq_u32_e32 vcc, 2, v0", "s_cbranch_vccz 3", "s_and_saveexec_b64 s[16:17], vcc",
            "v_add_nc_u32_e32 v6, 9, v6", "s_or_b64 exec, exec, s[16:17]"]), "A", [], [])

    def test_a_loop_execnz_no_scalar_state(self):
        self.tier("ps_a_loop", ps([
            "v_and_b32_e32 v4, 3, v0", "v_add_nc_u32_e32 v10, 1, v4", "v_mov_b32_e32 v9, 0",
            "v_add_nc_u32_e32 v9, 1, v9", "v_cmp_lt_u32_e32 vcc, v9, v10", "s_and_b64 exec, exec, vcc",
            "s_cbranch_execnz -4", "s_mov_b64 exec, s[10:11]", "v_mov_b32_e32 v6, v9"]), "A", [], [])

    def test_a_bpermute_swizzle_in_half(self):
        self.tier("ps_a_bperm", ps(["v_lshlrev_b32_e32 v4, 2, v0", "ds_bpermute_b32 v6, v4, v0"]), "A", [], [])

    def test_a_bpermute_lane_id_address(self):
        # the lane id is 0..63 in the wave and 0..31 in a half: equal modulo 32 lanes, which is all a wave64
        # ds_bpermute uses (address bits 2..6, per half), so xor/and/or/add on it and a shift by 2 stay exact
        self.tier("ps_a_bperm_id", ps([*LANE_ID, "v_xor_b32_e32 v11, 1, v10", "v_lshlrev_b32_e32 v11, 2, v11",
                                       "ds_bpermute_b32 v6, v11, v0", "v_and_b32_e32 v12, -4, v10",
                                       "v_or_b32_e32 v12, 2, v12", "v_lshl_add_u32 v12, v12, 2, 0",
                                       "ds_bpermute_b32 v7, v12, v0", "s_waitcnt lgkmcnt(0)"]), "A", [], [])

    def test_a_permute_lane_id_address(self):
        self.tier("ps_a_perm_id", ps([*LANE_ID, "v_add_nc_u32_e32 v11, 5, v10", "v_lshlrev_b32_e32 v11, 2, v11",
                                      "ds_permute_b32 v6, v11, v0", "s_waitcnt lgkmcnt(0)"]), "A", [], [])

    # -- tier B
    def test_b_waterfall(self):
        self.tier("ps_b_wf", ps([
            "v_and_b32_e32 v4, 7, v0", "v_readfirstlane_b32 s20, v4", "v_cmp_eq_u32_e32 vcc, s20, v4",
            "s_and_saveexec_b64 s[12:13], vcc", "s_mul_i32 s21, s20, 17", "v_add_nc_u32_e32 v6, s21, v0",
            "s_xor_b64 exec, exec, s[12:13]", "s_cbranch_execnz -7", "s_mov_b64 exec, s[10:11]"]), "B", [],
            ["waterfall"])

    def test_b_readfirstlane_of_uniform(self):
        self.tier("ps_b_rfl", ps(["v_mov_b32_e32 v5, s4", "v_readfirstlane_b32 s20, v5",
                                  "v_add_nc_u32_e32 v6, s20, v0"]), "B", [], ["readfirstlane-uniform"])

    def test_b_readlane_of_uniform_any_lane(self):
        self.tier("ps_b_rl", ps(["v_mov_b32_e32 v5, s4", "v_readlane_b32 s20, v5, 40",
                                 "v_add_nc_u32_e32 v6, s20, v0"]), "B", [], ["readlane-uniform"])

    def test_b_aggregated_atomic(self):
        self.tier("ps_b_atomic", ps([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_and_saveexec_b64 s[12:13], vcc",
            "s_bcnt1_i32_b64 s6, exec", "v_mbcnt_lo_u32_b32 v10, exec_lo, 0", "v_mbcnt_hi_u32_b32 v10, exec_hi, v10",
            "v_cmp_eq_u32_e32 vcc, 0, v10", "s_and_saveexec_b64 s[14:15], vcc", "v_mov_b32_e32 v11, s6",
            "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc", "s_or_b64 exec, exec, s[14:15]",
            "v_readfirstlane_b32 s7, v11", "v_add_nc_u32_e32 v13, s7, v10", "v_lshlrev_b32_e32 v13, 2, v13",
            "buffer_store_dword v0, v13, s[0:3], 0 offen offset:256", "s_mov_b64 exec, s[10:11]"]), "B", [],
            ["aggregated-atomic", "elected-lane"])

    def test_b_spill_and_reload_in_half(self):
        self.tier("ps_b_spill", ps(["v_writelane_b32 v20, s4, 3", "v_readlane_b32 s21, v20, 3",
                                    "v_add_nc_u32_e32 v6, s21, v0"]), "B", [], ["lane-spill"])

    # -- tier C: each near miss of the idioms above
    def test_c_lane_count_exported(self):
        self.tier("ps_c_cnt", ps(["v_cmp_eq_u32_e32 vcc, 1, v0", "s_bcnt1_i32_b64 s6, vcc", "v_mov_b32_e32 v6, s6"]),
                  "C", ["lane-count"])

    def test_c_rank_exported(self):
        self.tier("ps_c_rank", ps(["v_mbcnt_lo_u32_b32 v6, exec_lo, 0", "v_mbcnt_hi_u32_b32 v6, exec_hi, v6"]), "C",
                  ["lane-rank"])

    def test_c_lane_id_exported(self):
        self.tier("ps_c_id", ps(["v_mbcnt_lo_u32_b32 v6, -1, 0", "v_mbcnt_hi_u32_b32 v6, -1, v6"]), "C", ["lane-id"])

    def test_c_bpermute_lane_id_address_near_misses(self):
        # a shift by 1: address bit 7 is not reached, lane bit 5 lands in bit 6, which the permute uses
        self.tier("ps_c_bperm_shl1", ps([*LANE_ID, "v_lshlrev_b32_e32 v11, 1, v10", "ds_bpermute_b32 v6, v11, v0",
                                         "s_waitcnt lgkmcnt(0)"]), "C", ["lane-id"])
        # the low count alone is 32 for every upper-half lane of the wave
        self.tier("ps_c_bperm_lo", ps(["v_mbcnt_lo_u32_b32 v10, -1, 0", "v_lshlrev_b32_e32 v11, 2, v10",
                                       "ds_bpermute_b32 v6, v11, v0", "s_waitcnt lgkmcnt(0)"]), "C", ["lane-id"])
        # a right shift moves lane bit 5 down into the bits the permute uses
        self.tier("ps_c_bperm_shr", ps([*LANE_ID, "v_lshrrev_b32_e32 v11, 1, v10", "v_lshlrev_b32_e32 v11, 3, v11",
                                        "ds_bpermute_b32 v6, v11, v0", "s_waitcnt lgkmcnt(0)"]), "C", ["lane-id"])
        # combined with a lane mask: the mask is not the same in the wave and in a half
        self.tier("ps_c_bperm_mask", ps([*LANE_ID, "v_and_b32_e32 v11, s10, v10", "v_lshlrev_b32_e32 v11, 2, v11",
                                         "ds_bpermute_b32 v6, v11, v0", "s_waitcnt lgkmcnt(0)"]), "C")
        # SDWA selects and integer clamp change the value: a byte of id << 7, a saturating id - 32
        self.tier("ps_c_bperm_sdwa", ps([*LANE_ID, "v_lshlrev_b32_e32 v11, 7, v10",
                                         "v_mov_b32_sdwa v12, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1",
                                         "v_lshlrev_b32_e32 v12, 2, v12", "ds_bpermute_b32 v6, v12, v0",
                                         "s_waitcnt lgkmcnt(0)"]), "C", ["lane-id"])
        self.tier("ps_c_bperm_clamp", ps([*LANE_ID, "v_add_nc_u32_e64 v11, v10, -32 clamp",
                                          "v_lshlrev_b32_e32 v11, 2, v11", "ds_bpermute_b32 v6, v11, v0",
                                          "s_waitcnt lgkmcnt(0)"]), "C", ["lane-id"])
        # the lane id still taints the data it permutes, and the lane id itself
        self.tier("ps_c_bperm_data", ps([*LANE_ID, "v_lshlrev_b32_e32 v11, 2, v10", "ds_bpermute_b32 v6, v11, v10",
                                         "s_waitcnt lgkmcnt(0)"]), "C", ["lane-id"])

    def test_c_first_lane_index(self):
        self.tier("ps_c_ff1", ps(["v_cmp_eq_u32_e32 vcc, 1, v0", "s_ff1_i32_b64 s6, vcc", "v_mov_b32_e32 v6, s6"]),
                  "C", ["first-lane"])

    def test_c_readlane_of_varying(self):
        self.tier("ps_c_rl", ps(["v_readlane_b32 s6, v0, 40", "v_mov_b32_e32 v6, s6"]), "C", ["cross-lane"])
        self.tier("ps_c_rl7", ps(["v_readlane_b32 s6, v0, 7", "v_mov_b32_e32 v6, s6"]), "C", ["cross-lane"])

    def test_c_writelane_of_varying_lane(self):
        self.tier("ps_c_wl", ps(["v_writelane_b32 v6, s4, 40"]), "C", ["cross-lane"])

    def test_c_readfirstlane_of_varying_exported(self):
        self.tier("ps_c_rfl", ps(["v_readfirstlane_b32 s6, v0", "v_mov_b32_e32 v6, s6"]), "C")

    def test_c_waterfall_scalar_escapes(self):
        self.tier("ps_c_wf_out", ps([
            "v_and_b32_e32 v4, 7, v0", "v_readfirstlane_b32 s20, v4", "v_cmp_eq_u32_e32 vcc, s20, v4",
            "s_and_saveexec_b64 s[12:13], vcc", "v_add_nc_u32_e32 v6, s20, v0",
            "s_xor_b64 exec, exec, s[12:13]", "s_cbranch_execnz -5", "s_mov_b64 exec, s[10:11]",
            "v_mov_b32_e32 v7, s20"]), "C")

    def test_c_waterfall_trip_count(self):
        self.tier("ps_c_wf_count", ps([
            "s_mov_b32 s22, 0", "v_and_b32_e32 v4, 7, v0", "v_readfirstlane_b32 s20, v4",
            "v_cmp_eq_u32_e32 vcc, s20, v4", "s_and_saveexec_b64 s[12:13], vcc", "s_add_u32 s22, s22, 1",
            "s_xor_b64 exec, exec, s[12:13]", "s_cbranch_execnz -6", "s_mov_b64 exec, s[10:11]",
            "v_mov_b32_e32 v7, s22"]), "C", ["stale-scalar"])

    def test_c_vote_defines_scalar_read_later(self):
        self.tier("ps_c_stale", ps([
            "v_cmp_eq_u32_e32 vcc, 0, v0", "s_mov_b32 s5, 7", "s_cbranch_vccz 1", "s_add_u32 s5, s5, 1",
            "v_mov_b32_e32 v6, s5"]), "C")

    def test_c_exec_skip_defines_scalar_read_later(self):
        self.tier("ps_c_stale_exec", ps([
            "v_cmp_eq_u32_e32 vcc, 0, v0", "s_and_saveexec_b64 s[12:13], vcc", "s_mov_b32 s5, 7",
            "s_cbranch_execz 1", "s_add_u32 s5, s5, 1", "s_or_b64 exec, exec, s[12:13]", "v_mov_b32_e32 v6, s5"]),
            "C", ["stale-scalar"])

    def test_c_branch_on_count(self):
        self.tier("ps_c_brcnt", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_bcnt1_i32_b64 s6, vcc", "s_cmp_eq_u32 s6, 5", "s_cbranch_scc1 1",
            "v_mov_b32_e32 v6, 1"]), "C")

    def test_c_vccz_without_narrowing(self):
        self.tier("ps_c_vccz", ps(["v_cmp_eq_u32_e32 vcc, 2, v0", "s_cbranch_vccz 1", "v_add_nc_u32_e32 v6, 9, v6"]),
                  "C", ["unsafe-vote-branch"])

    def test_c_half_of_a_mask(self):
        self.tier("ps_c_half", ps(["s_mov_b32 exec_hi, 0"]), "C", ["mask-half"])
        self.tier("ps_c_half2", ps(["v_cmp_eq_u32_e32 vcc, 1, v0", "s_mov_b32 s6, vcc_hi", "v_mov_b32_e32 v6, s6"]),
                  "C", contains=["mask-half"])

    def test_c_mask_stored_as_data(self):
        self.tier("ps_c_maskdata", ps(["v_cmp_eq_u32_e32 vcc, 1, v0", "s_mov_b64 s[6:7], vcc",
                                       "v_mov_b32_e32 v6, s6"]), "C")

    def test_c_elected_lane_side_effect_not_a_count(self):
        self.tier("ps_c_elect", ps([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_and_saveexec_b64 s[12:13], vcc",
            "v_mbcnt_lo_u32_b32 v10, exec_lo, 0", "v_mbcnt_hi_u32_b32 v10, exec_hi, v10",
            "v_cmp_eq_u32_e32 vcc, 0, v10", "s_and_saveexec_b64 s[14:15], vcc", "v_mov_b32_e32 v11, 1",
            "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc",
            "s_or_b64 exec, exec, s[14:15]", "s_mov_b64 exec, s[10:11]"]), "C", ["elected-side-effect"])

    def test_c_rank_into_export_even_with_base(self):
        self.tier("ps_c_slot_exported", ps([
            "v_mbcnt_lo_u32_b32 v10, exec_lo, 0", "v_mbcnt_hi_u32_b32 v10, exec_hi, v10", "v_add_nc_u32_e32 v6, 1, v10"]),
            "C", ["lane-rank"])

    # -- non-wave64 and other stages are not analysed
    def test_wave32_is_not_analysed(self):
        asm = translate_listing("ps_w32", listing("ps_w32", ["v_mov_b32_e32 v6, 0", "exp mrt0 v6, v6, v6, v6 done vm",
                                                           "s_endpgm"]))
        self.assertIsNone(rdna2spirv.translate.last_gen.w64g)
        self.assertNotEqual(asm, "")


class Execution(unittest.TestCase):
    """Vertex shaders: the words of lane k with all 64 lanes in one wave and with the lanes cut into two waves."""

    def run_both(self, name, lines, push=None):
        (tier, reasons, idioms), asm = meta_of(name, lines)
        whole, halves = run_cut(asm, 64, push), run_cut(asm, 32, push)
        return tier, out_param(whole), out_param(halves)

    def test_a_vertex_invariant(self):
        tier, whole, halves = self.run_both("vs_a_skip", vs([
            "v_and_b32_e32 v4, 7, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "v_mov_b32_e32 v6, 0x3e8",
            "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execz 1", "v_mul_u32_u24_e32 v6, 7, v0",
            "s_or_b64 exec, exec, s[12:13]", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_wqm_b64 s[14:15], vcc",
            "v_cndmask_b32_e64 v7, 0, 1, s[14:15]"]))
        self.assertEqual(tier, "A")
        self.assertEqual(whole, halves)
        # lanes with k % 8 == 1 run the skip region; whole-quad mode lights the 4 consecutive lanes around them
        want = [(7 * k if k % 8 == 1 else 1000, int(k % 8 < 4)) for k in range(64)]
        self.assertEqual([w[:2] for w in whole], want)

    def test_b_waterfall_invariant(self):
        tier, whole, halves = self.run_both("vs_b_wf", vs([
            "v_and_b32_e32 v4, 7, v0", "v_readfirstlane_b32 s20, v4", "v_cmp_eq_u32_e32 vcc, s20, v4",
            "s_and_saveexec_b64 s[12:13], vcc", "s_mul_i32 s21, s20, 17", "v_add_nc_u32_e32 v6, s21, v0",
            "s_xor_b64 exec, exec, s[12:13]", "s_cbranch_execnz -7", "s_mov_b64 exec, s[10:11]"]))
        self.assertEqual(tier, "B")
        self.assertEqual(whole, halves)
        self.assertEqual([w[0] for w in whole], [17 * (k & 7) + k for k in range(64)])

    def test_a_bpermute_lane_id_address_invariant(self):
        tier, whole, halves = self.run_both("vs_a_bperm_id", vs([
            *LANE_ID, "v_xor_b32_e32 v11, 1, v10", "v_lshlrev_b32_e32 v11, 2, v11", "ds_bpermute_b32 v6, v11, v0",
            "s_waitcnt lgkmcnt(0)"]))
        self.assertEqual(tier, "A")
        self.assertEqual(whole, halves)
        self.assertEqual([w[0] for w in whole], [k ^ 1 for k in range(64)])

    def test_a_permlane_swizzle_invariant(self):
        # vertex k is lane k of the wave (and lane k % 32 of its half): every read stays in the lane's half
        tier, whole, halves = self.run_both("vs_a_perm", vs([
            "s_mov_b32 s20, 0x67452301", "s_mov_b32 s21, 0xefcdab89", "v_permlane16_b32 v6, v0, s20, s21",
            "s_mov_b32 s22, 0x76543210", "s_mov_b32 s23, 0xfedcba98", "v_permlanex16_b32 v7, v0, s22, s23"]))
        self.assertEqual(tier, "A")
        self.assertEqual(whole, halves)
        self.assertEqual([w[:2] for w in whole], [(k ^ 1, k ^ 16) for k in range(64)])
        tier, whole, halves = self.run_both("vs_a_swz", vs([
            "ds_swizzle_b32 v6, v0 offset:swizzle(QUAD_PERM,3,2,1,0)", "ds_swizzle_b32 v7, v0 offset:swizzle(SWAP,16)",
            "s_waitcnt lgkmcnt(0)"]))
        self.assertEqual(tier, "A")
        self.assertEqual(whole, halves)
        self.assertEqual([w[:2] for w in whole], [(k ^ 3, k ^ 16) for k in range(64)])

    def test_c_bpermute_shift_by_one_differs(self):
        # the negative control of the rule: lane bit 5 reaches an address bit the permute uses
        tier, whole, halves = self.run_both("vs_c_bperm_shl1", vs([
            *LANE_ID, "v_lshlrev_b32_e32 v11, 1, v10", "ds_bpermute_b32 v6, v11, v0", "s_waitcnt lgkmcnt(0)"]))
        self.assertEqual(tier, "C")
        self.assertNotEqual([w[0] for w in whole], [w[0] for w in halves])

    def test_b_readlane_of_uniform_value_at_lane_40(self):
        # lane 40 does not exist in a 32-lane wave: without the rewrite the shuffle reads a lane that is not running
        tier, whole, halves = self.run_both("vs_b_rl", vs([
            "v_mov_b32_e32 v5, s4", "v_readlane_b32 s20, v5, 40", "v_add_nc_u32_e32 v6, s20, v0"]),
            push=[0, 0, 0, 0, 77])
        self.assertEqual(tier, "B")
        self.assertEqual(whole, halves)
        self.assertEqual([w[0] for w in whole], [77 + k for k in range(64)])

    def test_b_readfirstlane_of_uniform(self):
        tier, whole, halves = self.run_both("vs_b_rfl", vs([
            "v_mov_b32_e32 v5, s4", "v_readfirstlane_b32 s20, v5", "v_add_nc_u32_e32 v6, s20, v0"]),
            push=[0, 0, 0, 0, 9])
        self.assertEqual(tier, "B")
        self.assertEqual(whole, halves)
        self.assertEqual([w[0] for w in whole], [9 + k for k in range(64)])

    def test_c_lane_count_differs(self):
        tier, whole, halves = self.run_both("vs_c_cnt", vs([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_bcnt1_i32_b64 s6, vcc",
            "v_mov_b32_e32 v6, s6", "v_mov_b32_e32 v7, v4"]))
        self.assertEqual(tier, "C")
        self.assertEqual([w[1] for w in whole], [w[1] for w in halves])     # per-lane data is the same
        self.assertEqual({w[0] for w in whole}, {16})
        self.assertEqual({w[0] for w in halves}, {8})

    def test_c_lane_id_differs(self):
        tier, whole, halves = self.run_both("vs_c_id", vs([
            "v_mbcnt_lo_u32_b32 v6, -1, 0", "v_mbcnt_hi_u32_b32 v6, -1, v6"]))
        self.assertEqual(tier, "C")
        self.assertEqual([w[0] for w in whole], list(range(64)))
        self.assertEqual([w[0] for w in halves], list(range(32)) * 2)

    def test_c_stale_scalar_differs(self):
        tier, whole, halves = self.run_both("vs_c_stale", vs([
            "v_cmp_eq_u32_e32 vcc, 0, v0", "s_mov_b32 s5, 7", "s_cbranch_vccz 1", "s_add_u32 s5, s5, 1",
            "v_mov_b32_e32 v6, s5"]))
        self.assertEqual(tier, "C")
        self.assertEqual({w[0] for w in whole}, {8})                      # vertex 0 is on, the whole wave adds
        self.assertEqual({w[0] for w in halves[:32]}, {8})
        self.assertEqual({w[0] for w in halves[32:]}, {7})                # the other half voted no


AGG_HEAD = ["v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_and_saveexec_b64 s[12:13], vcc"]
AGG_RANK = ["v_mbcnt_lo_u32_b32 v10, exec_lo, 0", "v_mbcnt_hi_u32_b32 v10, exec_hi, v10"]
AGG_ELECT = ["v_cmp_eq_u32_e32 vcc, 0, v10", "s_and_saveexec_b64 s[14:15], vcc", "v_mov_b32_e32 v11, s6",
             "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc", "s_or_b64 exec, exec, s[14:15]",
             "v_readfirstlane_b32 s7, v11"]


class ReviewRegressions(unittest.TestCase):
    """Programs an independent review used to get a false tier A or B (silent wrong rendering). Each must be C,
    or, as vertex shaders, the 64-lane and the two-half runs must agree."""
    tier = Classification.tier

    def test_scc_of_a_vote_into_data(self):
        for name, body in (
                ("cselect", ["v_cmp_eq_u32_e32 vcc, 1, v0", "s_cmp_lg_u64 vcc, 0", "s_cselect_b32 s5, 1, 0",
                             "v_mov_b32_e32 v6, s5"]),
                ("saveexec_cselect", ["v_cmp_eq_u32_e32 vcc, 1, v0", "s_and_saveexec_b64 s[12:13], vcc",
                                      "s_cselect_b32 s5, 1, 0", "s_or_b64 exec, exec, s[12:13]",
                                      "v_mov_b32_e32 v6, s5"]),
                ("addc", ["v_cmp_eq_u32_e32 vcc, 1, v0", "s_cmp_lg_u64 vcc, 0", "s_addc_u32 s5, s5, 0",
                          "v_mov_b32_e32 v6, s5"]),
                ("cmov", ["v_cmp_eq_u32_e32 vcc, 1, v0", "s_cmp_lg_u64 vcc, 0", "s_mov_b32 s5, 0",
                          "s_cmov_b32 s5, 9", "v_mov_b32_e32 v6, s5"]),
                ("bcnt", ["s_bcnt1_i32_b64 s6, exec", "s_cselect_b32 s5, 1, 0", "v_mov_b32_e32 v6, s5"])):
            self.tier("ps_scc_" + name, ps(body), "C", contains=["cross-lane"])

    def test_scc_vote_still_branches(self):
        # the one allowed consumer of a vote's SCC: s_cbranch_scc*, skipping lane-masked work
        self.tier("ps_scc_branch", ps([
            "v_cmp_eq_u32_e32 vcc, 2, v0", "s_and_saveexec_b64 s[16:17], vcc", "s_cbranch_scc0 1",
            "v_add_nc_u32_e32 v6, 9, v6", "s_or_b64 exec, exec, s[16:17]"]), "A", [], [])

    def test_rank_alone_is_not_a_slot(self):
        self.tier("ps_rank_addr", ps([
            "v_mbcnt_lo_u32_b32 v10, exec_lo, 0", "v_mbcnt_hi_u32_b32 v10, exec_hi, v10",
            "v_lshlrev_b32_e32 v13, 2, v10", "buffer_store_dword v0, v13, s[0:3], 0 offen offset:256"]),
            "C", ["lane-rank"])

    def test_base_alone_is_not_a_slot(self):
        self.tier("ps_base_addr", ps(AGG_HEAD + ["s_bcnt1_i32_b64 s6, exec"] + AGG_RANK + AGG_ELECT + [
            "v_mov_b32_e32 v13, s7", "buffer_store_dword v0, v13, s[0:3], 0 offen offset:256",
            "s_mov_b64 exec, s[10:11]"]), "C", ["allocation-order"])

    def test_atomic_add_of_a_count_by_every_lane(self):
        for name, mods in (("plain", ""), ("glc", " glc")):
            self.tier("ps_allcount_" + name, ps([
                "s_bcnt1_i32_b64 s6, exec", "v_mov_b32_e32 v11, s6", "v_mov_b32_e32 v12, 0",
                "buffer_atomic_add v11, v12, s[0:3], 0 offen" + mods]), "C", ["lane-count"])

    def test_count_of_another_mask_in_the_elected_atomic(self):
        self.tier("ps_cnt_other", ps([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_bcnt1_i32_b64 s6, vcc",
            "v_cmp_eq_u32_e32 vcc, 2, v4", "s_and_saveexec_b64 s[12:13], vcc"] + AGG_RANK + AGG_ELECT[:2] + [
            "v_mov_b32_e32 v11, s6", "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc",
            "s_or_b64 exec, exec, s[14:15]", "s_mov_b64 exec, s[10:11]"]), "C", ["lane-count"])

    def test_count_of_exec_before_it_narrowed(self):
        self.tier("ps_cnt_wider", ps([
            "s_bcnt1_i32_b64 s6, exec", "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4",
            "s_and_saveexec_b64 s[12:13], vcc"] + AGG_RANK + AGG_ELECT[:2] + [
            "v_mov_b32_e32 v11, s6", "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc",
            "s_or_b64 exec, exec, s[14:15]", "s_mov_b64 exec, s[10:11]"]), "C", ["lane-count"])

    def test_vote_whose_other_side_has_code(self):
        self.tier("ps_vccnz_else", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_cbranch_vccnz 2", "v_mov_b32_e32 v6, 5", "s_branch 3",
            "s_and_saveexec_b64 s[16:17], vcc", "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[16:17]"]),
            "C", contains=["unsafe-vote-branch"])
        self.tier("ps_vccnz_else2", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_cbranch_vccnz 2", "v_mov_b32_e32 v6, 5", "s_branch 3",
            "s_and_saveexec_b64 s[16:17], vcc", "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[16:17]",
            "v_mov_b32_e32 v7, 1"]), "C", contains=["unsafe-vote-branch"])

    def test_vccnz_with_a_jump_only_fall_through_is_a_skip(self):
        self.tier("ps_vccnz_jump", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_cbranch_vccnz 1", "s_branch 3", "s_and_saveexec_b64 s[16:17], vcc",
            "v_add_nc_u32_e32 v6, 13, v6", "s_or_b64 exec, exec, s[16:17]"]), "A", [], [])

    def test_exec_zero_side_with_code_of_its_own(self):
        # execz: the taken side (no lane active in the wave) writes an SGPR that is exported afterwards
        self.tier("ps_execz_sgpr", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_mov_b32 s5, 7", "s_and_saveexec_b64 s[12:13], vcc",
            "s_cbranch_execz 2", "v_mov_b32_e32 v8, 100", "s_branch 1", "s_mov_b32 s5, 9",
            "s_or_b64 exec, exec, s[12:13]", "v_mov_b32_e32 v6, s5"]), "C", contains=["unsafe-execz-branch"])
        # execnz: the fall-through is the exec-zero side
        self.tier("ps_execnz_else", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execnz 2",
            "s_mov_b32 s5, 9", "s_branch 1", "v_mov_b32_e32 v8, 100", "s_or_b64 exec, exec, s[12:13]",
            "v_mov_b32_e32 v6, s5"]), "C", contains=["unsafe-execz-branch"])
        # a VALU-only exec-zero side is refused too (kept simple: the side must be the join)
        self.tier("ps_execz_valu", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execz 2",
            "v_mov_b32_e32 v6, 7", "s_branch 1", "v_mov_b32_e32 v6, 5", "s_or_b64 exec, exec, s[12:13]"]),
            "C", contains=["unsafe-execz-branch"])

    def test_exec_zero_side_that_is_the_join_or_a_loop_exit_is_a_skip(self):
        self.tier("ps_execz_join", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execz 1",
            "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[12:13]"]), "A", [], [])
        self.tier("ps_execz_jump_join", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execnz 1",
            "s_branch 1", "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[12:13]"]), "A", [], [])

    # ---- second review: a rank / count / mask that is not bound to the mask it belongs to
    ATOMIC = ["v_mov_b32_e32 v11, s6", "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc",
              "s_or_b64 exec, exec, s[14:15]"]
    SLOT = ["v_readfirstlane_b32 s7, v11", "v_add_nc_u32_e32 v13, s7, {rank}", "v_lshlrev_b32_e32 v13, 2, v13",
            "buffer_store_dword v0, v13, s[0:3], 0 offen offset:256"]
    ELECT_NARROWED = ["v_cmp_eq_u32_e32 vcc, 0, v10", "s_and_saveexec_b64 s[14:15], vcc"]
    ENTRY_RANK = ["v_mbcnt_lo_u32_b32 v14, exec_lo, 0", "v_mbcnt_hi_u32_b32 v14, exec_hi, v14"]

    def test_slot_rank_of_the_entry_exec_is_not_the_counted_mask(self):
        body = (self.ENTRY_RANK + AGG_HEAD + ["s_bcnt1_i32_b64 s6, exec"] + AGG_RANK + self.ELECT_NARROWED
                + self.ATOMIC + [x.format(rank="v14") for x in self.SLOT] + ["s_mov_b64 exec, s[10:11]"])
        self.tier("ps_rank_entry_exec", ps(body), "C", contains=["lane-rank"])

    def test_slot_rank_from_the_saved_exec(self):
        body = (AGG_HEAD + ["s_bcnt1_i32_b64 s6, exec"] + AGG_RANK + self.ELECT_NARROWED + self.ATOMIC + [
            "v_mbcnt_lo_u32_b32 v14, s12, 0", "v_mbcnt_hi_u32_b32 v14, s13, v14"]
            + [x.format(rank="v14") for x in self.SLOT] + ["s_mov_b64 exec, s[10:11]"])
        self.tier("ps_rank_saved_exec", ps(body), "C", contains=["lane-rank"])

    def test_slot_of_the_right_mask_is_still_b(self):
        body = (AGG_HEAD + ["s_bcnt1_i32_b64 s6, exec"] + AGG_RANK + self.ELECT_NARROWED + self.ATOMIC
                + [x.format(rank="v10") for x in self.SLOT] + ["s_mov_b64 exec, s[10:11]"])
        self.tier("ps_rank_right_mask", ps(body), "B", [], ["aggregated-atomic", "elected-lane"])

    def test_lane_mask_combined_with_a_register_is_data(self):
        for name, body in (
                ("and_user_bcnt", ["s_and_b64 s[14:15], exec, s[4:5]", "s_bcnt1_i32_b64 s6, s[14:15]",
                                   "v_mov_b32_e32 v6, s6"]),
                ("and_user_narrowed", ["v_cmp_lt_u32_e32 vcc, 40, v0", "s_and_b64 exec, exec, vcc",
                                       "s_and_b64 s[14:15], exec, s[4:5]", "s_mov_b64 exec, s[10:11]",
                                       "s_bcnt1_i32_b64 s6, s[14:15]", "v_mov_b32_e32 v6, s6"]),
                ("vcc_and_user_cmp", ["v_cmp_lt_u32_e32 vcc, 40, v0", "s_and_b64 s[14:15], vcc, s[4:5]",
                                      "s_cmp_eq_u64 s[14:15], 0", "s_cselect_b32 s6, 1, 2", "v_mov_b32_e32 v6, s6"]),
                ("vcc_and_user_branch", ["v_cmp_lt_u32_e32 vcc, 40, v0", "s_and_b64 s[14:15], vcc, s[4:5]",
                                         "s_cmp_eq_u64 s[14:15], 0", "s_cbranch_scc1 1", "v_mov_b32_e32 v6, 4"]),
                ("or_user_ff1", ["v_cmp_lt_u32_e32 vcc, 40, v0", "s_or_b64 s[14:15], vcc, s[4:5]",
                                 "s_ff1_i32_b64 s6, s[14:15]", "v_mov_b32_e32 v6, s6"]),
                ("andn2_user_bcnt", ["v_cmp_lt_u32_e32 vcc, 40, v0", "s_andn2_b64 s[14:15], s[4:5], vcc",
                                     "s_bcnt1_i32_b64 s6, s[14:15]", "v_mov_b32_e32 v6, s6"]),
                ("and_user_bcnt0", ["v_cmp_lt_u32_e32 vcc, 40, v0", "s_and_b64 s[14:15], vcc, s[4:5]",
                                    "s_bcnt0_i32_b64 s6, s[14:15]", "v_mov_b32_e32 v6, s6"])):
            self.tier("ps_maskdata_" + name, ps(body), "C")

    def test_mask_and_constants_stay_masks(self):
        self.tier("ps_mask_const", ps([
            "v_cmp_lt_u32_e32 vcc, 40, v0", "s_and_b64 s[14:15], vcc, -1", "s_and_saveexec_b64 s[12:13], s[14:15]",
            "v_add_nc_u32_e32 v6, 9, v6", "s_or_b64 exec, exec, s[12:13]"]), "A", [], [])

    def test_waterfall_vgpr_rewritten_before_the_compare(self):
        self.tier("ps_wf_rewritten", ps([
            "v_and_b32_e32 v4, 7, v0", "v_readfirstlane_b32 s20, v4", "v_add_nc_u32_e32 v4, 1, v4",
            "v_cmp_eq_u32_e32 vcc, s20, v4", "s_and_saveexec_b64 s[12:13], vcc", "v_add_nc_u32_e32 v6, s20, v0",
            "s_xor_b64 exec, exec, s[12:13]", "s_cbranch_execnz -7", "s_mov_b64 exec, s[10:11]"]), "C")

    def test_analysis_failure_is_tier_c_and_not_an_error(self):
        import wave64_halves
        old = wave64_halves.analyse

        def boom(*a, **k):
            raise RuntimeError("boom")
        wave64_halves.analyse = boom
        try:
            self.tier("ps_boom", ps(["v_mov_b32_e32 v6, 1"]), "C", ["analysis-error"])
        finally:
            wave64_halves.analyse = old


class ReviewExecution(unittest.TestCase):
    run_both = Execution.run_both

    def lanes(self, name, body):
        tier, whole, halves = self.run_both(name, vs(body))
        return tier, sorted({w[0] if w else None for w in whole}), sorted({w[0] if w else None for w in halves})

    def test_cselect_of_a_vote_differs_and_is_c(self):
        tier, whole, halves = self.lanes("vs_cselect", [
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_cmp_lg_u64 vcc, 0", "s_cselect_b32 s5, 1, 0", "v_mov_b32_e32 v6, s5"])
        self.assertEqual((tier, whole, halves), ("C", [1], [0, 1]))     # vertex 1 is in the first half only

    def test_vote_else_side_differs_and_is_c(self):
        tier, whole, halves = self.lanes("vs_vccnz_else", [
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_cbranch_vccnz 2", "v_mov_b32_e32 v6, 5", "s_branch 3",
            "s_and_saveexec_b64 s[16:17], vcc", "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[16:17]"])
        self.assertEqual(tier, "C")
        self.assertNotEqual(whole, halves)

    def test_vote_with_empty_other_side_is_invariant(self):
        tier, whole, halves = self.lanes("vs_vccz_join", [
            "v_cmp_eq_u32_e32 vcc, 33, v0", "v_mov_b32_e32 v6, 4", "s_cbranch_vccz 3",
            "s_and_saveexec_b64 s[16:17], vcc", "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[16:17]"])
        self.assertEqual(tier, "A")
        self.assertEqual(whole, halves)
        self.assertEqual(whole, [4, 7])                                  # only vertex 33, in the second half


class Refusal(unittest.TestCase):
    def test_error_mode_refuses_tier_c(self):
        old = rdna2spirv.WAVE64_GRAPHICS
        rdna2spirv.WAVE64_GRAPHICS = "error"
        try:
            with self.assertRaises(rdna2spirv.Unsupported) as cm:
                translate_listing("ps_c_cnt", listing("ps_c_cnt", ps(["v_cmp_eq_u32_e32 vcc, 1, v0",
                                                                      "s_bcnt1_i32_b64 s6, vcc",
                                                                      "v_mov_b32_e32 v6, s6"])))
            self.assertIn("lane-count", str(cm.exception))
        finally:
            rdna2spirv.WAVE64_GRAPHICS = old

    def test_off_mode_has_no_tier(self):
        old = rdna2spirv.WAVE64_GRAPHICS
        rdna2spirv.WAVE64_GRAPHICS = "off"
        try:
            translate_listing("ps_c_cnt", listing("ps_c_cnt", ps(["v_cmp_eq_u32_e32 vcc, 1, v0",
                                                                  "s_bcnt1_i32_b64 s6, vcc", "v_mov_b32_e32 v6, s6"])))
            self.assertIsNone(rdna2spirv.translate.last_gen.w64g)
        finally:
            rdna2spirv.WAVE64_GRAPHICS = old


class GeometryRing(unittest.TestCase):
    """TEST-251: a legacy GS's LDS write is an output slot from its offset alone (rdna2spirv.gs_ds drops the VGPR
    address, the lane's GSVS area), so a lane id that only forms that address is no partition dependence; the data
    written, and a lane id in a memory store's address, still are. GPU twins: gs_w64_a_ring, gs_w64_c_lane_id."""
    PRO = ["s_bfe_i32 s0, s3, 0x80008", "s_bfm_b64 exec, s0, 0", "s_cmp_eq_u32 s0, 64", "s_cselect_b64 exec, -1, exec",
           "v_mbcnt_lo_u32_b32 v13, -1, 0", "v_mbcnt_hi_u32_b32 v13, -1, v13"]
    EPI = ["s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)", "s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)",
           "s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)", "s_endpgm"]

    def tier(self, body):
        sys.path.insert(0, os.path.join(HERE, "..", "..", "runtime", "test", "shaders"))
        import build                                       # gs_container: gs_tri's ISG1/OSG1
        name = "gs_w64_ring_test"
        translate_listing(name, listing(name, self.PRO + body + self.EPI), build.gs_container())
        w = rdna2spirv.translate.last_gen.w64g
        return w["tier"], w["reasons"]

    def test_lane_id_in_the_ring_address_is_tier_a(self):
        self.assertEqual(self.tier(["v_mul_u32_u24_e32 v12, 0x60, v13", "v_mov_b32_e32 v14, 0",
                                    "ds_write2_b32 v12, v14, v14 offset0:0 offset1:1"]), ("A", []))

    def test_lane_id_written_to_the_ring_is_tier_c(self):
        self.assertEqual(self.tier(["v_mov_b32_e32 v12, 0", "ds_write2_b32 v12, v13, v13 offset0:0 offset1:1"]),
                         ("C", ["lane-id"]))

    def test_lane_id_in_a_memory_store_address_is_tier_c(self):
        self.assertEqual(self.tier(["v_mov_b32_e32 v14, 0", "v_lshlrev_b32_e32 v12, 2, v13",
                                    "buffer_store_dword v14, v12, s[4:7], 0 offen",
                                    "ds_write2_b32 v14, v14, v14 offset0:0 offset1:1"]), ("C", ["lane-id"]))


if __name__ == "__main__":
    unittest.main()
