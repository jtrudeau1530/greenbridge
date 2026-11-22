"""Semantic interpreter tests for RDNA2 s_sext_i32_i8, s_sext_i32_i16 and s_flbit_i32.

s_sext_i32_i8 / s_sext_i32_i16: D = S0's low byte / low half, sign-extended to 32 bits. s_flbit_i32: the number of
bits, counted from the MSB, before the first bit that differs from the sign bit (-1, i.e. 0xFFFFFFFF, for 0 and -1);
the scalar form of v_ffbh_i32. None of them writes SCC. Each case translates a synthetic pixel shader with
tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the result with a plain-Python
reference. D leaves through exp as two 16-bit halves ORed into 0x40000000, so no export channel is a NaN or denormal
pattern that a float round trip could change; SCC comes back through s_cselect_b32.
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
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def sext(x: int, bits: int) -> int:
    x &= (1 << bits) - 1
    return (x - (1 << bits) if x >> (bits - 1) else x) & M32


def flbit_i32(x: int) -> int:
    x &= M32
    sign = x >> 31
    for n in range(1, 32):                       # bit 31 - n is the n-th bit below the sign bit
        if (x >> (31 - n)) & 1 != sign:
            return n
    return M32


OPS = {"s_sext_i32_i8": lambda x: sext(x, 8), "s_sext_i32_i16": lambda x: sext(x, 16), "s_flbit_i32": flbit_i32}
VALUES = [0, 1, 0x7F, 0x80, 0xFF, 0x7FFF, 0x8000, 0xFFFF, 0x12345678, 0x87654321, 0x7FFFFFFF, 0x80000000,
          0xFFFFFFFF, 0xFFFFFFFE, 0x40000000, 0xC0000000, 0x00010000]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_sext.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_sext.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_op(op: str, x: int, scc_in: int) -> tuple:
    """SCC = scc_in, then s0 = op(s2); return (s0, SCC after)."""
    if _TMP_DIR is None:
        setUpModule()
    lines = [f"s_mov_b32 s2, {x:#x}", f"s_mov_b32 s4, {scc_in}", "s_cmp_eq_u32 s4, 1", f"{op} s0, s2",
             "s_cselect_b32 s1, 0x40000001, 0x40000000", "s_and_b32 s5, s0, 0xffff", "s_lshr_b32 s6, s0, 16",
             "s_or_b32 s5, s5, 0x40000000", "s_or_b32 s6, s6, 0x40000000",
             "v_mov_b32 v0, s5", "v_mov_b32 v1, s6", "v_mov_b32 v2, s1"]
    asm = "; test_salu_sext\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v2, v2 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi, scc = si.run_fragment(spv)["%out_mrt0"][:3]
    assert lo >> 16 == hi >> 16 == 0x4000 and scc >> 1 == 0x20000000, (hex(lo), hex(hi), hex(scc))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16, scc & 1


class SextFlbitTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(sext(0x80, 8), 0xFFFFFF80)
        self.assertEqual(sext(0x1234807F, 8), 0x7F)
        self.assertEqual(sext(0xABCD8000, 16), 0xFFFF8000)
        self.assertEqual(flbit_i32(1), 31)
        self.assertEqual(flbit_i32(0x40000000), 1)
        self.assertEqual(flbit_i32(0xFFFFFFFE), 31)
        self.assertEqual(flbit_i32(0x80000000), 1)
        self.assertEqual(flbit_i32(0), M32)
        self.assertEqual(flbit_i32(M32), M32)

    def test_values_and_scc_kept(self):
        for op, fn in OPS.items():
            for x in VALUES:
                for scc_in in (0, 1):
                    with self.subTest(op=op, x=hex(x), scc=scc_in):
                        self.assertEqual(run_op(op, x, scc_in), (fn(x), scc_in))

    def test_seeded(self):
        rng = random.Random(0x5E)
        for _ in range(30):
            op, x = rng.choice(list(OPS)), rng.getrandbits(32) >> rng.randrange(32)
            if rng.getrandbits(1):
                x = ~x & M32                     # long runs of sign bits for s_flbit_i32
            with self.subTest(op=op, x=hex(x)):
                self.assertEqual(run_op(op, x, 0)[0], OPS[op](x))


if __name__ == "__main__":
    unittest.main()
