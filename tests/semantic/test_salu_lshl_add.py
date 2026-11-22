"""Semantic interpreter tests for RDNA2 s_lshl1_add_u32 ... s_lshl4_add_u32.

D = (S0 << N) + S1 (low 32 bits). SCC is the unsigned overflow of the sum taken in 64 bits, as the RDNA ISA
pseudocode writes it (tmp = (64'U(S0) << N) + 64'U(S1); SCC = tmp >= 2^32): a bit shifted out of S0 counts as a
carry just like a carry out of the addition. Each case translates a synthetic pixel shader with tools/rdna2spirv.py,
runs it in tests/semantic/spirv_mini_interp.py and compares the exported words (D, and SCC through s_cselect_b32)
with a plain-Python reference.
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

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VALUES = [0, 1, 0x0FFFFFFF, 0x10000000, 0x1FFFFFFF, 0x3FFFFFFF, 0x40000000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
          0x12345678, 0xF0000000]


def ref(n: int, x: int, y: int) -> tuple:
    t = (x << n) + y
    return t & MASK32, int(t >> 32 != 0)


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_lshl_add.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_lshl_add.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> tuple:
    """Run the lines, export v0 and v1, return those two words."""
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_salu_lshl_add\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return tuple(si.run_fragment(spv)["%out_mrt0"][:2])


def run_op(n: int, x: int, y: int, scc_in: int) -> tuple:
    """SCC = scc_in, then s0 = s_lshlN_add_u32(s2, s3); return (s0, SCC after)."""
    return run_lines([f"s_mov_b32 s2, {x:#x}", f"s_mov_b32 s3, {y:#x}", f"s_mov_b32 s4, {scc_in}", "s_cmp_eq_u32 s4, 1",
                      f"s_lshl{n}_add_u32 s0, s2, s3", "s_cselect_b32 s1, 1, 0", "v_mov_b32 v0, s0", "v_mov_b32 v1, s1"])


class LshlAddTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref(2, 3, 4), (16, 0))
        self.assertEqual(ref(1, 0x80000000, 0), (0, 1))                 # the shifted-out bit alone sets SCC
        self.assertEqual(ref(4, 0x0FFFFFFF, 0x10), (0, 1))              # carry out of the addition alone
        self.assertEqual(ref(3, 0x1FFFFFFF, 7), (0xFFFFFFFF, 0))        # fits exactly

    def test_values_and_scc(self):
        for n in (1, 2, 3, 4):
            for x in VALUES:
                for y in VALUES[::3]:
                    want = ref(n, x, y)
                    with self.subTest(n=n, x=hex(x), y=hex(y)):
                        self.assertEqual(run_op(n, x, y, 1 - want[1]), want)

    def test_seeded(self):
        rng = random.Random(0x5A)
        for _ in range(40):
            n, x, y = rng.randint(1, 4), rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(n=n, x=hex(x), y=hex(y)):
                self.assertEqual(run_op(n, x, y, rng.getrandbits(1)), ref(n, x, y))

    def test_scc_drives_s_addc(self):
        # A 64-bit index scaled by 4 into a base: the carry from s_lshl2_add_u32 feeds the high word.
        got = run_lines(["s_mov_b32 s2, 0x40000001", "s_mov_b32 s3, 0xfffffffc", "s_mov_b32 s5, 7",
                         "s_lshl2_add_u32 s0, s2, s3", "s_addc_u32 s1, s5, 0", "v_mov_b32 v0, s0", "v_mov_b32 v1, s1"])
        self.assertEqual(got, (0, 8))


if __name__ == "__main__":
    unittest.main()
