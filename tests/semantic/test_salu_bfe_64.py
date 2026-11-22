"""Semantic interpreter tests for RDNA2 s_bfe_u64 and s_bfe_i64.

64-bit bitfield extract from the SGPR pair S0: offset = S1[5:0], width = S1[22:16] (0-127). A width of 0 gives 0.
A field that runs past bit 63 stops there (LLVM constantFoldBFE: Src0 >> Offset, arithmetic for the signed op),
so the width is cut to 64 - offset. s_bfe_i64 sign-extends the field from its top bit. SCC = (D != 0). Each case
translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and
compares the pair and SCC with a plain-Python reference. The pair leaves through exp as four 16-bit halves ORed into
0x40000000, so no export channel is a NaN or denormal pattern that a float round trip could change.
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

M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OPS = ("s_bfe_u64", "s_bfe_i64")


def ref(op: str, x: int, ctl: int) -> int:
    off, w = ctl & 63, (ctl >> 16) & 0x7F
    if w == 0:
        return 0
    w = min(w, 64 - off)
    field = (x >> off) & ((1 << w) - 1)
    if op == "s_bfe_i64" and field >> (w - 1):
        field -= 1 << w
    return field & M64


def ctl(off: int, width: int) -> int:
    return (width & 0x7F) << 16 | (off & 0x3F)


VALUES = [0, M64, 1 << 63, 0x0123456789ABCDEF, 0xFEDCBA9876543210, 0x8000000080000000, 0x00000000FFFFFFFF]
CONTROLS = [ctl(0, 0), ctl(0, 1), ctl(0, 64), ctl(0, 127), ctl(1, 63), ctl(31, 2), ctl(32, 32), ctl(33, 31),
            ctl(60, 8), ctl(63, 1), ctl(63, 64), ctl(4, 40), 0xFFC0FFFF, 0x00100020]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_bfe64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_bfe64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> tuple:
    """Run the lines, return (s[0:1] as one 64-bit value, SCC)."""
    if _TMP_DIR is None:
        setUpModule()
    tail = ["s_cselect_b32 s20, 0x40000001, 0x40000000"]
    for i, (r, sh) in enumerate(((0, 0), (0, 16), (1, 0), (1, 16))):
        tail += [f"s_lshr_b32 s{10 + i}, s{r}, {sh}", f"s_and_b32 s{10 + i}, s{10 + i}, 0xffff",
                 f"s_or_b32 s{10 + i}, s{10 + i}, 0x40000000", f"v_mov_b32 v{i}, s{10 + i}"]
    tail.append("v_mov_b32 v4, s20")
    asm = ("; test_salu_bfe64\n" + "".join(f"\t{ln}\n" for ln in list(lines) + tail)
           + "\texp mrt0 v0, v1, v2, v3 vm\n\texp mrt1 v4, v4, v4, v4 done vm\n\ts_endpgm\n")
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    out = si.run_fragment(spv)
    w, scc = out["%out_mrt0"], out["%out_mrt1"][0]
    assert all(x >> 16 == 0x4000 for x in w) and scc >> 1 == 0x20000000, ([hex(x) for x in w], hex(scc))
    return sum((x & 0xFFFF) << (16 * i) for i, x in enumerate(w)), scc & 1


def run_op(op: str, x: int, c: int, scc_in: int = 0) -> tuple:
    return run_lines([f"s_mov_b32 s2, {x & 0xFFFFFFFF:#x}", f"s_mov_b32 s3, {x >> 32:#x}", f"s_mov_b32 s4, {c:#x}",
                      f"s_mov_b32 s5, {scc_in}", "s_cmp_eq_u32 s5, 1", f"{op} s[0:1], s[2:3], s4"])


class Bfe64Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref("s_bfe_u64", 0x0123456789ABCDEF, ctl(32, 32)), 0x01234567)
        self.assertEqual(ref("s_bfe_i64", 1 << 63, ctl(60, 8)), (-8) & M64)           # field past bit 63: >> 60
        self.assertEqual(ref("s_bfe_u64", M64, ctl(0, 127)), M64)                     # width cut to 64
        self.assertEqual(ref("s_bfe_i64", 0x80, ctl(4, 4)), (-8) & M64)
        self.assertEqual(ref("s_bfe_u64", M64, ctl(5, 0)), 0)

    def test_values_and_scc(self):
        for op in OPS:
            for x in VALUES:
                for c in CONTROLS:
                    want = ref(op, x, c)
                    with self.subTest(op=op, x=hex(x), ctl=hex(c)):
                        self.assertEqual(run_op(op, x, c, 1 - int(want != 0)), (want, int(want != 0)))

    def test_seeded(self):
        rng = random.Random(0xBFE64)
        for _ in range(30):
            op, x = rng.choice(OPS), rng.getrandbits(64)
            c = ctl(rng.randrange(64), rng.randrange(128)) if rng.random() < 0.8 else rng.getrandbits(32)
            with self.subTest(op=op, x=hex(x), ctl=hex(c)):
                self.assertEqual(run_op(op, x, c)[0], ref(op, x, c))

    def test_lane_mask_source_is_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["s_mov_b32 s4, 0x100000", "s_bfe_u64 s[0:1], exec, s4"])


if __name__ == "__main__":
    unittest.main()
