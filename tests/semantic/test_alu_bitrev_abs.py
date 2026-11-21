"""Semantic interpreter tests for RDNA2 v_bfrev_b32, s_brev_b32 and s_abs_i32.

v_bfrev_b32 / s_brev_b32: D = S0 with its 32 bits in reverse order. s_abs_i32: D = |S0| as a signed 32-bit integer,
SCC = (D != 0); 0x80000000 is its own absolute value (two's complement wrap). Each case translates a synthetic pixel
shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the exported words with
a plain-Python reference.
"""
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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VALUES = [0, 1, 2, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0x12345678, 0x0000FFFF, 0xF0F0F0F0, 0x00000100]


def ref_brev(x: int) -> int:
    return int(f"{x & MASK32:032b}"[::-1], 2)


def ref_abs(x: int) -> int:
    x &= MASK32
    return (-(x - (1 << 32)) if x >> 31 else x) & MASK32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_bitrev_abs.s")
    with open(os.path.join(_TMP_DIR.name, "ps_bitrev_abs.dxbc"), "wb") as f:
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
    asm = "; test_bitrev_abs\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return tuple(si.run_fragment(spv)["%out_mrt0"][:2])


class BitReverseTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref_brev(1), 0x80000000)
        self.assertEqual(ref_brev(0x12345678), 0x1E6A2C48)
        self.assertEqual(ref_brev(0x0000FFFF), 0xFFFF0000)

    def test_v_bfrev_b32(self):
        for x in VALUES:
            with self.subTest(x=hex(x)):
                self.assertEqual(run_lines([f"v_mov_b32 v2, {x:#x}", "v_bfrev_b32_e32 v0, v2", "v_mov_b32 v1, 0"])[0],
                                 ref_brev(x))

    def test_s_brev_b32(self):
        for x in VALUES:
            with self.subTest(x=hex(x)):
                self.assertEqual(run_lines([f"s_mov_b32 s2, {x:#x}", "s_brev_b32 s0, s2", "v_mov_b32 v0, s0",
                                            "v_mov_b32 v1, 0"])[0], ref_brev(x))

    def test_twice_is_identity(self):
        for x in VALUES:
            with self.subTest(x=hex(x)):
                self.assertEqual(run_lines([f"v_mov_b32 v2, {x:#x}", "v_bfrev_b32_e32 v3, v2", "v_bfrev_b32_e32 v0, v3",
                                            "v_mov_b32 v1, 0"])[0], x)


class AbsTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref_abs(0xFFFFFFFF), 1)
        self.assertEqual(ref_abs(0x80000000), 0x80000000)
        self.assertEqual(ref_abs(0x7FFFFFFF), 0x7FFFFFFF)

    def test_s_abs_i32_and_scc(self):
        for x in VALUES:
            with self.subTest(x=hex(x)):
                want = ref_abs(x)
                got = run_lines([f"s_mov_b32 s2, {x:#x}", "s_abs_i32 s0, s2", "s_cselect_b32 s1, 1, 0",
                                 "v_mov_b32 v0, s0", "v_mov_b32 v1, s1"])
                self.assertEqual(got, (want, int(want != 0)))


if __name__ == "__main__":
    unittest.main()
