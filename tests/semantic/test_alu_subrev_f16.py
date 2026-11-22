"""Semantic interpreter tests for RDNA2 v_subrev_f16.

v_subrev_f16: D.f16 = S1.f16 - S0.f16 (VOP2 0x34 on GFX10), the reversed form of v_sub_f16. Like the other f16
VOP2/VOP3 ops it writes the low half of the destination and keeps the high half; in the VOP3 form op_sel bit 0 / 1
reads the high half of src0 / src1 and bit 2 writes the high half of the destination instead (docs/design.md,
"16-bit results"). Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py and compares the exported word with a plain-Python reference: the difference
of the two f16 values computed in double precision and rounded once to f16 (round to nearest even).
"""
import itertools
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xAAAA5555                     # the destination before the op; the half that is not written must survive
A = 0x44003E00                       # v1: lo 1.5, hi 4.0
B = 0x40003800                       # v2: lo 0.5, hi 2.0


def h2f(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:            # past the largest finite f16: round to nearest gives infinity
        return 0x7C00 if x > 0 else 0xFC00


def half(word: int, hi: int) -> float:
    return h2f(word >> 16 if hi else word)


def merge(old: int, h: int, dst_hi: int) -> int:
    return (h << 16 | old & 0xFFFF) if dst_hi else (old & 0xFFFF0000 | h)


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_subrev_f16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_subrev_f16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run(line: str, v1: int, v2: int, old: int = OLD) -> int:
    """Seed v1, v2 and the destination v0, run ``line``, return v0."""
    if _TMP_DIR is None:
        setUpModule()
    lines = [f"v_mov_b32 v1, {v1:#x}", f"v_mov_b32 v2, {v2:#x}", f"v_mov_b32 v0, {old:#x}", line]
    asm = "; test_subrev_f16\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class SubrevF16Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(f2h(0.5 - 1.5), 0xBC00)
        self.assertEqual(f2h(65504.0 + 65504.0), 0x7C00)
        self.assertEqual(merge(OLD, 0x3C00, 0), 0xAAAA3C00)
        self.assertEqual(merge(OLD, 0x3C00, 1), 0x3C005555)

    def test_vop2_is_src1_minus_src0_and_keeps_the_high_half(self):
        self.assertEqual(run("v_subrev_f16_e32 v0, v1, v2", A, B), merge(OLD, f2h(0.5 - 1.5), 0))

    def test_vop3_op_sel(self):
        for bits in itertools.product((0, 1), repeat=3):
            line = f"v_subrev_f16_e64 v0, v1, v2 op_sel:[{bits[0]},{bits[1]},{bits[2]}]"
            with self.subTest(line=line):
                want = merge(OLD, f2h(half(B, bits[1]) - half(A, bits[0])), bits[2])
                self.assertEqual(run(line, A, B), want)

    def test_neg_on_src0_and_clamp(self):
        # -(1.5) reversed: 0.5 - (-1.5) = 2.0; clamp then limits it to 1.0.
        self.assertEqual(run("v_subrev_f16_e64 v0, -v1, v2", A, B), merge(OLD, f2h(2.0), 0))
        self.assertEqual(run("v_subrev_f16_e64 v0, -v1, v2 clamp", A, B), merge(OLD, f2h(1.0), 0))

    def test_matches_v_sub_f16_with_swapped_operands(self):
        rng = random.Random(0x34)
        special = [0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x3555]
        for _ in range(40):
            x, y = (rng.choice(special) if rng.random() < 0.3 else rng.randrange(0x10000) for _ in range(2))
            if (x & 0x7C00) == 0x7C00 or (y & 0x7C00) == 0x7C00:
                continue                     # finite inputs only (NaN payloads are not modelled here)
            v1, v2 = 0x12340000 | x, 0x56780000 | y
            with self.subTest(x=hex(x), y=hex(y)):
                got = run("v_subrev_f16_e32 v0, v1, v2", v1, v2)
                self.assertEqual(got, run("v_sub_f16_e32 v0, v2, v1", v1, v2))
                want = f2h(h2f(y) - h2f(x))
                if want in (0x0000, 0x8000):     # x == y: the sign of the zero is not what this test is about
                    self.assertEqual(got & 0xFFFF7FFF, OLD & 0xFFFF0000)
                else:
                    self.assertEqual(got, merge(OLD, want, 0))


if __name__ == "__main__":
    unittest.main()
