"""Wave64 VOP3 lane-mask operands in wave32-syntax listings (rdna2spirv.wave64_mask_operands_as_pairs).

The listings are disassembled by llvm-mc for gfx1013 without -mattr=+wavefrontsize64, i.e. in wave32 syntax: the
SDST / carry / condition operand of a VOP3 (v_cndmask_b32_e64 src2, v_cmp_*_e64 dst, v_add/sub/subrev_co_ci_u32
carry out and in, v_add/sub_co_u32 carry out, v_div_scale sdst, v_mad_u64_u32 carry out) prints as `sN` where a
wave64 program reads or writes the pair s[N:N+1], and VCC / EXEC print as vcc_lo / exec_lo. RDNA2 ISA: these are
64-bit in wave64. Before the fix a data pair with unequal words gave lanes 32-63 the low word's bits, and a mask
written to `sN` never reached s[N+1], which a later 64-bit SALU read saw stale (checked here on the CPU
interpreter). vcc_lo / exec_lo already stood for the whole register (VCC / EXEC hold this lane's bit) and are
guarded here. Wave32 programs are not touched.

Hand-written listings translated with tools/rdna2spirv.py and run on the CPU in spirv_mini_interp.py. Standard
library only; no GPU, no game files.
"""
import gc
import json
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from shader_ir import parse_line  # noqa: E402

LO, HI = 0x0000FFFF, 0xFFFF0000            # s4, s5: a 64-bit data mask whose two words differ
PAIR = HI << 32 | LO
WANT_PAIR = [(PAIR >> i) & 1 for i in range(64)]
HIGH_LANES = [int(i >= 32) for i in range(64)]
STORE = ["v_lshlrev_b32 v6, 2, v0", "buffer_store_dword v4, v6, s[0:3], 0 offen", "s_endpgm"]
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def translate_cs(name, lines, threads, wave32=False, emulate=False):
    """Translate a hand-written compute listing with synthetic FSPX dispatch state."""
    name = "cs_" + name
    with tempfile.TemporaryDirectory() as tmp:
        for directory in ("disasm", "fspx"):
            os.mkdir(os.path.join(tmp, directory))
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        # Eight user SGPRs: s[0:3] holds the output descriptor, s4/s5 the unequal mask words.
        with open(os.path.join(tmp, "fspx", name + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"rsrc2": 0x8090, "threads": [threads, 1, 1], "wave32": wave32}, f)
        with open(os.path.join(tmp, "fspx", name + ".code"), "wb") as f:
            f.write(bytes(64))
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name} synthetic\n" + "".join(f"\t{line}\n" for line in lines))
        previous = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = emulate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
                gc.collect()
            assert rdna2spirv.translate.last_gen.wave64 == (not wave32)
            return spv
        finally:
            rdna2spirv.WAVE64_EMULATE = previous


def run_cs(spv, threads, s4=0, s5=0):
    return si.run_workgroup(
        spv, threads, threads,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [s4 & 0xFFFFFFFF, s5 & 0xFFFFFFFF] + [0] * 26, memory={},
    )


def lanes(name, body, threads=64, wave32=False, s4=LO, s5=HI):
    spv = translate_cs(name, ["v_mov_b32 v4, 0"] + body + STORE, threads, wave32=wave32)
    mem = run_cs(spv, threads, s4=s4, s5=s5)
    return [mem[4 * i] for i in range(threads)], spv


class DataPairRead(unittest.TestCase):
    """A single `sN` condition / carry-in naming a data pair: lane L takes bit L of s[N+1]:s[N]."""

    def test_cndmask_selector(self):
        got, _ = lanes("w64m_cnd", ["v_cndmask_b32_e64 v4, 0, 1, s4"])
        self.assertEqual(got, WANT_PAIR)

    def test_add_co_ci_carry_in(self):
        got, _ = lanes("w64m_addc", ["v_mov_b32 v1, 0", "v_add_co_ci_u32_e64 v4, s8, v1, 0, s4"])
        self.assertEqual(got, WANT_PAIR)

    def test_wave64_syntax_listing_is_the_same_module(self):
        for emulate in (False, True):
            with self.subTest(emulate=emulate):
                a = translate_cs("w64m_same", ["v_cndmask_b32_e64 v4, 0, 1, s4"] + STORE,
                                 64, emulate=emulate)
                b = translate_cs("w64m_same", ["v_cndmask_b32_e64 v4, 0, 1, s[4:5]"] + STORE,
                                 64, emulate=emulate)
                self.assertEqual(a, b)


class MaskWriteReachesHighWord(unittest.TestCase):
    """A mask written to `sN` by a VOP3 is the pair: a 64-bit SALU read of s[N:N+1] sees lanes 32-63."""

    def test_cmp_then_and_saveexec(self):
        got, _ = lanes("w64m_sx", ["v_cmp_le_u32_e64 s4, 32, v0", "s_and_saveexec_b64 s[8:9], s[4:5]",
                                   "v_mov_b32 v4, 1", "s_mov_b64 exec, -1"])
        self.assertEqual(got, HIGH_LANES)

    def test_cmp_then_bcnt1_b64(self):
        got, _ = lanes("w64m_bcnt", ["v_cmp_le_u32_e64 s4, 40, v0", "s_bcnt1_i32_b64 s12, s[4:5]",
                                     "v_mov_b32 v4, s12"])
        self.assertEqual(got, [24] * 64)

    def test_sub_co_ci_carry_out_into_exec(self):
        got, _ = lanes("w64m_subb", ["v_mov_b32 v1, 0", "v_cmp_le_u32_e64 s4, 32, v0",
                                     "v_sub_co_ci_u32_e64 v2, s8, v1, 0, s4", "s_mov_b64 exec, s[8:9]",
                                     "v_mov_b32 v4, 1", "s_mov_b64 exec, -1"])
        self.assertEqual(got, HIGH_LANES)


class VccExecLoAreWholeRegisters(unittest.TestCase):
    """Guards (already right before the fix, not rewritten by it): wave64 vcc_lo / exec_lo in a VOP mask slot."""

    def test_vcc_lo_selector_on_data_vcc(self):
        for body in (["s_mov_b64 vcc, s[4:5]", "v_cndmask_b32_e64 v4, 0, 1, vcc_lo"],
                     ["s_mov_b64 vcc, s[4:5]", "v_mov_b32 v1, 1", "v_cndmask_b32_e32 v4, 0, v1, vcc_lo"]):
            with self.subTest(body=body[-1]):
                got, _ = lanes("w64m_vcc", body)
                self.assertEqual(got, WANT_PAIR)

    def test_vcc_lo_compare_destination_then_vcc_pair_read(self):
        got, _ = lanes("w64m_vccw", ["v_cmp_le_u32_e32 vcc_lo, 32, v0", "s_and_saveexec_b64 s[8:9], vcc",
                                     "v_mov_b32 v4, 1", "s_mov_b64 exec, -1"])
        self.assertEqual(got, HIGH_LANES)

    def test_exec_lo_selector_and_destination(self):
        got, _ = lanes("w64m_exs", ["s_mov_b64 exec, s[4:5]", "v_cndmask_b32_e64 v4, 0, 1, exec_lo",
                                    "s_mov_b64 exec, -1"])
        self.assertEqual(got, WANT_PAIR)
        # A wave64 compare into exec_lo (= EXEC) stays refused, as before; it is not taken as one half.
        with self.assertRaises(rdna2spirv.Unsupported):
            lanes("w64m_exd", ["v_cmp_le_u32_e64 exec_lo, 32, v0", "v_mov_b32 v4, 1", "s_mov_b64 exec, -1"])


class Wave32Unchanged(unittest.TestCase):
    def test_wave32_selector_is_one_sgpr(self):
        got, _ = lanes("w32m_cnd", ["v_cndmask_b32_e64 v4, 0, 1, s4"], threads=32, wave32=True)
        self.assertEqual(got, [(LO >> i) & 1 for i in range(32)])

    def test_wave32_odd_sgpr_selector(self):
        got, _ = lanes("w32m_odd", ["v_cndmask_b32_e64 v4, 0, 1, s5"], threads=32, wave32=True)
        self.assertEqual(got, [(HI >> i) & 1 for i in range(32)])


class Canonicalisation(unittest.TestCase):
    def widen(self, text):
        ins = [parse_line("\t" + text)]
        rdna2spirv.wave64_mask_operands_as_pairs(ins)
        return [(a.get("r"), a.get("n")) if a["k"] == "sgpr" else a.get("t", a["k"]) for a in ins[0]["args"]]

    def test_slots(self):
        self.assertEqual(self.widen("v_cmp_eq_u32_e64 s6, v0, s2"), [(6, 2), "vgpr", (2, 1)])
        self.assertEqual(self.widen("v_cmp_class_f32_e64 s2, v0, v1"), [(2, 2), "vgpr", "vgpr"])
        self.assertEqual(self.widen("v_cndmask_b32_e64 v0, s1, v2, s4"), ["vgpr", (1, 1), "vgpr", (4, 2)])
        self.assertEqual(self.widen("v_add_co_u32 v0, s8, s3, v2"), ["vgpr", (8, 2), (3, 1), "vgpr"])
        self.assertEqual(self.widen("v_add_co_ci_u32_e64 v3, s10, v4, s5, s8"),
                         ["vgpr", (10, 2), "vgpr", (5, 1), (8, 2)])
        self.assertEqual(self.widen("v_subrev_co_ci_u32_e64 v3, s10, v4, v5, s8"),
                         ["vgpr", (10, 2), "vgpr", "vgpr", (8, 2)])
        self.assertEqual(self.widen("v_div_scale_f32 v0, s12, v1, v2, v3")[1], (12, 2))
        self.assertEqual(self.widen("v_mad_u64_u32 v[0:1], s12, v1, v2, v[4:5]")[1], (12, 2))
        # left alone: VCC / EXEC names, data operands, v_cmpx (no SDST on GFX10), readlane, already pairs
        self.assertEqual(self.widen("v_cndmask_b32_e64 v0, v1, v2, vcc_lo")[3], "vcc_lo")
        self.assertEqual(self.widen("v_cmp_eq_u32_e64 exec_lo, v0, v1")[0], "exec_lo")
        self.assertEqual(self.widen("v_cmpx_eq_u32_e64 s2, v1"), [(2, 1), "vgpr"])
        self.assertEqual(self.widen("v_readlane_b32 s4, v0, 3")[0], (4, 1))
        self.assertEqual(self.widen("v_cndmask_b32_e64 v0, v1, v2, s[4:5]")[3], (4, 2))

    def test_short_forms_have_no_mask_slots(self):
        # 3-operand VOP2 forms (implicit VCC carry / condition): src0 is plain data, never a mask slot, so an odd
        # SGPR is accepted and an even one is not widened.
        self.assertEqual(self.widen("v_add_co_ci_u32 v0, s5, v1"), ["vgpr", (5, 1), "vgpr"])
        self.assertEqual(self.widen("v_add_co_ci_u32 v0, s4, v1"), ["vgpr", (4, 1), "vgpr"])
        self.assertEqual(self.widen("v_sub_co_ci_u32 v0, s7, v1"), ["vgpr", (7, 1), "vgpr"])
        self.assertEqual(self.widen("v_subrev_co_ci_u32 v0, s6, v1"), ["vgpr", (6, 1), "vgpr"])
        self.assertEqual(self.widen("v_cndmask_b32 v0, s3, v1"), ["vgpr", (3, 1), "vgpr"])

    def test_short_form_odd_sgpr_data_translates(self):
        # A wave64 VOP2 carry-in add with an odd SGPR data source translates.
        got, _ = lanes("w64m_vop2c", ["v_mov_b32 v1, 0", "v_cmp_eq_u32_e64 vcc_lo, 0, 0",
                                      "v_add_co_ci_u32 v4, s5, v1"], s5=7)
        self.assertEqual(got, [8] * 64)

    def test_odd_sgpr_refused(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            self.widen("v_cndmask_b32_e64 v0, v1, v2, s5")


if __name__ == "__main__":
    unittest.main()
