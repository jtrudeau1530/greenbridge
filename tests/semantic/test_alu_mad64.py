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


if __name__ == "__main__":
    unittest.main()
