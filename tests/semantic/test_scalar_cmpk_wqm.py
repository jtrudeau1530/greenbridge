"""Value tests for two scalar-ALU families the translator handles but no other test reaches:

  s_cmpk_<cond>_<ty> sX, imm16   SCC = sX <cond> K, where K is the 16-bit immediate
                                 zero-extended for _u32 and sign-extended for _i32
                                 (tools/rdna2spirv.py: `v = a[1]["v"] & 0xFFFF`,
                                 `v -= 0x10000` when the sign bit is set).
  s_wqm_b32 sD, sS               D = every 4-bit group of S that has any bit set becomes 0xF;
                                 SCC = (D != 0) -- whole-quad mode on plain data
                                 (the lane-mask path is covered by the s_wqm_b64 tests).

Each case is a synthetic hand-written pixel-shader listing (s_mov_b32 seeds the source SGPR, one
instruction under test, s_cselect_b32 materialises SCC, v_mov_b32 carries results to the export),
translated by tools/rdna2spirv.py and executed by spirv_mini_interp.py. Expected values are
hand-computed constants; a plain-Python ISA reference is checked against them so a typo in either
is caught. Standard library only.
"""
import os
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import STUB_DXBC, mrt0_words, rdna2spirv  # noqa: E402  (support puts tools/ on sys.path)

MASK32 = 0xFFFFFFFF


def s32(v):
    v &= MASK32
    return v - (1 << 32) if v & 0x80000000 else v


def ref_cmpk(op, s0, imm):
    """RDNA2 ISA: imm16 zero-extended for u32, sign-extended for i32; then the scalar compare."""
    cond, ty = op[len("s_cmpk_"):].rsplit("_", 1)
    k = imm & 0xFFFF
    if ty == "i32" and k & 0x8000:
        k -= 0x10000
    a, b = (s32(s0), k) if ty == "i32" else (s0 & MASK32, k)
    return int({"eq": a == b, "ne": a != b, "lg": a != b, "lt": a < b,
                "le": a <= b, "gt": a > b, "ge": a >= b}[cond])


def ref_wqm_b32(x):
    """Each nibble with any bit set becomes 0xF."""
    out = 0
    for n in range(8):
        if (x >> (4 * n)) & 0xF:
            out |= 0xF << (4 * n)
    return out


# (op, s0, imm16, expected SCC). The u32 cases with s0's upper bits set discriminate the
# zero-extended immediate from a low-16-bit compare; the i32 cases discriminate sign- from
# zero-extension and signed from unsigned ordering.
CMPK_CASES = [
    ("s_cmpk_eq_u32", 0x0000BEEF, 0xBEEF, 1),
    ("s_cmpk_eq_u32", 0x0001BEEF, 0xBEEF, 0),   # not equal: K is 0x0000BEEF, not s0 & 0xFFFF
    ("s_cmpk_lg_u32", 0x0000BEEF, 0xBEEF, 0),
    ("s_cmpk_lg_u32", 0x00000005, 0x0006, 1),
    ("s_cmpk_gt_u32", 0x00010000, 0xFFFF, 1),   # 65536 > 65535
    ("s_cmpk_gt_u32", 0x0000FFFF, 0xFFFF, 0),   # tie
    ("s_cmpk_gt_u32", 0x80000000, 0x7FFF, 1),   # unsigned: high bit is magnitude, not sign
    ("s_cmpk_lt_u32", 0x00000000, 0x0001, 1),
    ("s_cmpk_lt_u32", 0x0000FFFF, 0xFFFE, 0),
    ("s_cmpk_lt_u32", 0x0000FFFE, 0xFFFF, 1),
    ("s_cmpk_ge_u32", 0x0000FFFF, 0xFFFF, 1),
    ("s_cmpk_ge_u32", 0x0000FFFE, 0xFFFF, 0),
    ("s_cmpk_le_u32", 0x0000FFFF, 0xFFFF, 1),
    ("s_cmpk_le_u32", 0x0000FFFF, 0xFFFE, 0),
    # i32: the immediate sign-extends; pairs picked so the wrong extension/order flips the answer.
    ("s_cmpk_eq_i32", 0xFFFF8000, 0x8000, 1),   # K = -32768
    ("s_cmpk_eq_i32", 0x00008000, 0x8000, 0),   # +32768 != -32768
    ("s_cmpk_eq_i32", 0xFFFFFFFF, 0xFFFF, 1),   # -1 == -1
    ("s_cmpk_gt_i32", 0x00000001, 0xFFFF, 1),   # 1 > -1 (zero-extended K would make this false)
    ("s_cmpk_gt_i32", 0xFFFFFFFE, 0xFFFD, 1),   # -2 > -3
    ("s_cmpk_lt_i32", 0xFFFFFFFF, 0x0000, 1),   # -1 < 0
    ("s_cmpk_lt_i32", 0x00000001, 0xFFFF, 0),   # 1 < -1 is false (1 < 65535 would be true)
    ("s_cmpk_ge_i32", 0xFFFFFFFE, 0xFFFE, 1),   # -2 >= -2
    ("s_cmpk_le_i32", 0xFFFFFFFE, 0xFFFE, 1),   # -2 <= -2
    ("s_cmpk_le_i32", 0xFFFFFFFE, 0xFFFD, 0),   # -2 <= -3 is false
    ("s_cmpk_lg_i32", 0xFFFF8000, 0x8000, 0),
    ("s_cmpk_lg_i32", 0xFFFF8000, 0x8001, 1),   # -32768 != -32767
    ("s_cmpk_lt_i32", 0x80000000, 0x7FFF, 1),   # INT_MIN < 32767 (unsigned order: false)
]

# (s0, expected D, expected SCC) for `s_wqm_b32 s0, s2`.
WQM_CASES = [
    (0x00000001, 0x0000000F, 1),   # lowest nibble only
    (0x00000010, 0x000000F0, 1),   # second nibble
    (0x00040000, 0x000F0000, 1),   # nibble 4
    (0x80000000, 0xF0000000, 1),   # sign bit: still a bit
    (0x00000000, 0x00000000, 0),   # nothing set: D = 0, SCC = 0
    (0x12345678, 0xFFFFFFFF, 1),   # every nibble nonzero
    (0x0000F000, 0x0000F000, 1),   # one full nibble
    (0x01020304, 0x0F0F0F0F, 1),   # low nibble of each byte
    (0xF0F0F0F0, 0xF0F0F0F0, 1),   # nibbles 1,3,5,7 set
    (0x0F0F0F0F, 0x0F0F0F0F, 1),   # nibbles 0,2,4,6 set
]

_TMP = None
_PS_PATH = None


def setUpModule():
    global _TMP, _PS_PATH
    _TMP = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(_TMP.name, "disasm"))
    _PS_PATH = os.path.join(_TMP.name, "disasm", "ps_cmpk_wqm.s")
    with open(os.path.join(_TMP.name, "ps_cmpk_wqm.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP
    if _TMP is not None:
        _TMP.cleanup()
        _TMP = None


def run_ps(lines):
    """Translate PS instruction lines; return the four exported words (v0, v1, v2, v3)."""
    asm = "; test_scalar_cmpk_wqm\n" + "".join(f"\t{ln}\n" for ln in lines) + \
          "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
    with open(_PS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_PS_PATH)
    words = mrt0_words(spv)
    return words[0], words[1], words[2], words[3]


class ReferenceMatchesHandConstants(unittest.TestCase):
    def test_reference(self):
        for op, s0, imm, want in CMPK_CASES:
            with self.subTest(op=op, s0=hex(s0), imm=hex(imm)):
                self.assertEqual(ref_cmpk(op, s0, imm), want)
        for s0, want_d, want_scc in WQM_CASES:
            with self.subTest(s0=hex(s0)):
                self.assertEqual(ref_wqm_b32(s0), want_d)
                self.assertEqual(int(want_d != 0), want_scc)


class TestCmpk(unittest.TestCase):
    """s_cmpk_<cond>_u32/i32: SCC observed through s_cselect_b32."""

    def test_all_cases(self):
        for op, s0, imm, want_scc in CMPK_CASES:
            with self.subTest(op=op, s0=hex(s0), imm=hex(imm)):
                w0, _, _, _ = run_ps([
                    f"s_mov_b32 s2, {s0 & MASK32:#x}",
                    f"{op} s2, {imm:#x}",
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ])
                self.assertEqual(w0, want_scc,
                                 f"{op} s2={s0:#010x} imm={imm:#06x}: SCC {w0}, want {want_scc}")


class TestWqmB32(unittest.TestCase):
    """s_wqm_b32 on plain data: nibble -> 0xF spread, SCC = (D != 0)."""

    def test_sgpr_source(self):
        for s0, want_d, want_scc in WQM_CASES:
            with self.subTest(s0=hex(s0)):
                w0, w1, _, _ = run_ps([
                    f"s_mov_b32 s2, {s0 & MASK32:#x}",
                    "s_wqm_b32 s0, s2",
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ])
                self.assertEqual((w0, w1), (want_d, want_scc),
                                 f"s_wqm_b32 {s0:#010x}: got ({w0:#010x}, {w1}), "
                                 f"want ({want_d:#010x}, {want_scc})")

    def test_immediate_source(self):
        # A literal takes the same data path: 0x1234 has all four low nibbles nonzero.
        w0, w1, _, _ = run_ps([
            "s_wqm_b32 s0, 0x1234",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual((w0, w1), (0x0000FFFF, 1))
        w0, w1, _, _ = run_ps([
            "s_wqm_b32 s0, 0",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual((w0, w1), (0x00000000, 0))


if __name__ == "__main__":
    unittest.main()
