"""Semantic interpreter tests for RDNA2 v_mad_u64_u32 and v_mad_i64_i32.

{sdst, D.u64} = S0 * S1 + S2 as a 65-bit result (RDNA ISA pseudocode): the 32 x 32 -> 64-bit product, unsigned for
v_mad_u64_u32 and signed for v_mad_i64_i32, plus the 64-bit S2. D is the low 64 bits; sdst, a lane mask, gets bit 64:
the carry out of the unsigned sum, and for the signed one the sign of the exact sum. Each case translates a synthetic
pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the result with a
plain-Python reference. v[0:1] and the sdst bit leave through exp as 16-bit halves ORed into 0x40000000, so no
export channel is a NaN or denormal pattern that a float round trip could change.
"""
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def s32(x: int) -> int:
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def s64(x: int) -> int:
    x &= M64
    return x - (1 << 64) if x >> 63 else x


def ref(op: str, x: int, y: int, z: int) -> tuple:
    """(D.u64, sdst bit) from the 65-bit result."""
    r = (x & M32) * (y & M32) + (z & M64) if op == "v_mad_u64_u32" else s32(x) * s32(y) + s64(z)
    return r & M64, (r >> 64) & 1


SPECIAL32 = [0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFE, 0x12345678]
SPECIAL64 = [0, 1, M64, 1 << 63, (1 << 63) - 1, 0xFFFFFFFF, 1 << 32, 0xFFFFFFFE00000001]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_mad64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_mad64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


HALVES = ["v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0", "v_and_b32 v7, 0xffff, v1",
          "v_lshrrev_b32 v8, 16, v1", "v_or_b32 v5, 0x40000000, v5", "v_or_b32 v6, 0x40000000, v6",
          "v_or_b32 v7, 0x40000000, v7", "v_or_b32 v8, 0x40000000, v8"]


def run_lines(lines, sdst: str = "s[10:11]") -> tuple:
    """Run the lines, return (v[0:1] as one 64-bit value, 1 if lane 0's bit of sdst is set)."""
    if _TMP_DIR is None:
        setUpModule()
    tail = HALVES + [f"v_cndmask_b32 v9, 0x40000000, 0x40000001, {sdst}"]
    exports = "\texp mrt0 v5, v6, v7, v8 vm\n\texp mrt1 v9, v9, v9, v9 done vm\n"
    asm = "; test_mad64\n" + "".join(f"\t{ln}\n" for ln in list(lines) + tail) + exports + "\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    out = si.run_fragment(spv)
    w, flag = out["%out_mrt0"], out["%out_mrt1"][0]
    assert all(x >> 16 == 0x4000 for x in w) and flag >> 1 == 0x20000000, ([hex(x) for x in w], hex(flag))
    return sum((x & 0xFFFF) << (16 * i) for i, x in enumerate(w)), flag & 1


def run_op(op: str, x: int, y: int, z: int) -> tuple:
    return run_lines([f"v_mov_b32 v2, {x:#x}", f"v_mov_b32 v3, {y:#x}", f"v_mov_b32 v12, {z & M32:#x}",
                      f"v_mov_b32 v13, {z >> 32:#x}", "s_mov_b32 s10, 0", "s_mov_b32 s11, 0",
                      f"{op} v[0:1], s[10:11], v2, v3, v[12:13]"])


class Mad64Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref("v_mad_u64_u32", M32, M32, 0), (0xFFFFFFFE00000001, 0))
        self.assertEqual(ref("v_mad_u64_u32", M32, M32, M64), (0xFFFFFFFE00000000, 1))      # carry out
        self.assertEqual(ref("v_mad_i64_i32", M32, 1, 0), (M64, 1))                         # -1: sign set
        self.assertEqual(ref("v_mad_i64_i32", 0x80000000, 0x80000000, (1 << 63) - 1),     # 2^62 + 2^63 - 1
                         ((1 << 62) + (1 << 63) - 1, 0))

    def test_edge_values(self):
        for op in ("v_mad_u64_u32", "v_mad_i64_i32"):
            for x in SPECIAL32:
                for y in SPECIAL32[::2]:
                    for z in SPECIAL64[::3]:
                        with self.subTest(op=op, x=hex(x), y=hex(y), z=hex(z)):
                            self.assertEqual(run_op(op, x, y, z), ref(op, x, y, z))

    def test_seeded(self):
        rng = random.Random(0x64)
        for _ in range(40):
            op = rng.choice(["v_mad_u64_u32", "v_mad_i64_i32"])
            x, y, z = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(64)
            with self.subTest(op=op, x=hex(x), y=hex(y), z=hex(z)):
                self.assertEqual(run_op(op, x, y, z), ref(op, x, y, z))

    def test_inline_accumulator_and_null_sdst(self):
        # S2 = -1 as a 64-bit inline constant (sign-extended), and sdst = null: only D is written.
        got = run_lines(["v_mov_b32 v2, 3", "v_mov_b32 v3, 5", "s_mov_b32 s10, 0", "s_mov_b32 s11, 0",
                         "v_mad_u64_u32 v[0:1], null, v2, v3, -1"])
        self.assertEqual(got, (14, 0))
        got = run_lines(["v_mov_b32 v2, 3", "v_mov_b32 v3, 5", "v_mad_u64_u32 v[0:1], s[10:11], v2, v3, -1"])
        self.assertEqual(got, (14, 1))                                     # 15 + (2^64 - 1) carries out

    def test_carry_feeds_v_add_co_ci(self):
        # The sdst mask is a real carry: add it into a third word.
        got = run_lines(["v_mov_b32 v2, -1", "v_mov_b32 v3, -1", "v_mov_b32 v12, -1", "v_mov_b32 v13, -1",
                         "v_mad_u64_u32 v[0:1], vcc_lo, v2, v3, v[12:13]", "v_mov_b32 v14, 41",
                         "v_add_co_ci_u32 v1, vcc_lo, v14, 0, vcc_lo", "v_mov_b32 v0, 0", "s_mov_b32 s10, 0",
                         "s_mov_b32 s11, 0"])
        self.assertEqual(got[0] >> 32, 42)


class Mad64WaveMaskTests(unittest.TestCase):
    """lane_masks.py must track v_mad_u64_u32 / v_mad_i64_i32 sdst as a lane mask across all 64 lanes."""

    def run_wave64_cs(self, name: str, body_lines):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            s_path = os.path.join(tmp, "disasm", name + ".s")
            with open(s_path, "w", encoding="utf-8") as f:
                f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in body_lines) + "\ts_endpgm\n")
            with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            # COMPUTE_PGM_RSRC2: 0x8088 -> 4 user SGPRs (s[0:3] = output V#), TGID_X, 64x1x1 threads.
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(s_path)
        pc = [0, 0, 0xFFFFFFFF, 0] + [0] * 28
        mem = si.run_workgroup(
            spv, 64, 64,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=pc, memory={},
        )
        return mem

    def test_mad_u64_carry_mask_ballot_wave64(self):
        # Only lanes 3 and 40 carry out (v2=0xffffffff, v3=1, acc=0xffffffff_ffffffff -> carry=1);
        # all other lanes have v3=0 -> carry=0. s_bcnt1_i32_b64 must return 2 and s_ff1_i32_b64 must return 3
        # uniformly in every lane (without lane_masks tracking, lanes 3/40 saw 64 and the rest saw 0).
        mem = self.run_wave64_cs("cs_mad_u64_mask", [
            "s_mov_b64 exec, -1",
            "v_cmp_eq_u32_e64 s[4:5], v0, 3",
            "v_cmp_eq_u32_e64 s[6:7], v0, 40",
            "s_or_b64 s[4:5], s[4:5], s[6:7]",
            "v_cndmask_b32_e64 v3, 0, 1, s[4:5]",
            "v_mov_b32 v2, 0xffffffff",
            "v_mov_b32 v12, 0xffffffff",
            "v_mov_b32 v13, 0xffffffff",
            "v_mad_u64_u32 v[6:7], s[10:11], v2, v3, v[12:13]",
            "s_bcnt1_i32_b64 s12, s[10:11]",
            "s_ff1_i32_b64 s13, s[10:11]",
            "v_mov_b32 v8, s12",
            "v_mov_b32 v9, s13",
            "v_lshlrev_b32 v10, 3, v0",
            "buffer_store_dwordx2 v[8:9], v10, s[0:3], 0 offen",
        ])
        for lane in range(64):
            self.assertEqual(mem[lane * 8], 2, f"lane {lane} bcnt1")
            self.assertEqual(mem[lane * 8 + 4], 3, f"lane {lane} ff1")

    def test_mad_i64_sign_mask_ballot_wave64(self):
        # Only lanes 7, 35 and 63 get v2 = -1 (with v3 = 1, acc = 0 -> exact sum = -1, bit 64 = 1);
        # all other lanes get v2 = 1 -> sum = +1, bit 64 = 0.
        mem = self.run_wave64_cs("cs_mad_i64_mask", [
            "s_mov_b64 exec, -1",
            "v_cmp_eq_u32_e64 s[4:5], v0, 7",
            "v_cmp_eq_u32_e64 s[6:7], v0, 35",
            "s_or_b64 s[4:5], s[4:5], s[6:7]",
            "v_cmp_eq_u32_e64 s[6:7], v0, 63",
            "s_or_b64 s[4:5], s[4:5], s[6:7]",
            "v_cndmask_b32_e64 v2, 1, -1, s[4:5]",
            "v_mov_b32 v3, 1",
            "v_mov_b32 v12, 0",
            "v_mov_b32 v13, 0",
            "v_mad_i64_i32 v[6:7], s[10:11], v2, v3, v[12:13]",
            "s_bcnt1_i32_b64 s12, s[10:11]",
            "s_ff1_i32_b64 s13, s[10:11]",
            "v_mov_b32 v8, s12",
            "v_mov_b32 v9, s13",
            "v_lshlrev_b32 v10, 3, v0",
            "buffer_store_dwordx2 v[8:9], v10, s[0:3], 0 offen",
        ])
        for lane in range(64):
            self.assertEqual(mem[lane * 8], 3, f"lane {lane} bcnt1")
            self.assertEqual(mem[lane * 8 + 4], 7, f"lane {lane} ff1")


if __name__ == "__main__":
    unittest.main()
