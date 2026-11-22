"""Semantic interpreter tests for RDNA2 v_lerp_u8.

Per byte i: D.byte[i] = (S0.byte[i] + S1.byte[i] + (bit 0 of S2.byte[i])) >> 1, an unsigned average of the two
bytes whose rounding (down or up) comes from S2 (RDNA ISA pseudocode). Each case translates a synthetic pixel shader
with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the result with a plain-Python
reference. D leaves through exp as two 16-bit halves ORed into 0x40000000, so no export channel is a NaN or
denormal pattern that a float round trip could change.
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def ref(x: int, y: int, z: int) -> int:
    out = 0
    for i in range(4):
        b = lambda v: (v >> (8 * i)) & 0xFF            # noqa: E731
        out |= ((b(x) + b(y) + (b(z) & 1)) >> 1) << (8 * i)
    return out


VALUES = [0, 0xFFFFFFFF, 0x01010101, 0xFF00FF00, 0x00FF00FF, 0x80808080, 0x7F7F7F7F, 0x12345678, 0xFEDCBA98]
ROUND = [0, 0xFFFFFFFF, 0x01010101, 0x01000100, 0xFEFEFEFE]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_lerp_u8.s")
    with open(os.path.join(_TMP_DIR.name, "ps_lerp_u8.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> int:
    """Run the lines, return v0 (exported as two 16-bit halves, see the module docstring)."""
    if _TMP_DIR is None:
        setUpModule()
    lines = list(lines) + ["v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0",
                           "v_or_b32 v5, 0x40000000, v5", "v_or_b32 v6, 0x40000000, v6"]
    asm = "; test_lerp_u8\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16


def run_op(x: int, y: int, z: int) -> int:
    return run_lines([f"v_mov_b32 v1, {x:#x}", f"v_mov_b32 v2, {y:#x}", f"v_mov_b32 v3, {z:#x}",
                      "v_lerp_u8 v0, v1, v2, v3"])


class LerpU8Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref(0xFF, 0xFF, 1), 0xFF)                       # 511 >> 1: never carries into byte 1
        self.assertEqual(ref(0x01, 0x02, 0), 0x01)                       # 3 >> 1, rounded down
        self.assertEqual(ref(0x01, 0x02, 1), 0x02)                       # rounded up by S2's bit 0
        self.assertEqual(ref(0x00000300, 0x00000400, 0x00000100), 0x00000400)
        self.assertEqual(ref(0x01, 0x02, 0xFE), 0x01)                    # only bit 0 of S2's byte counts

    def test_edge_values(self):
        for x in VALUES:
            for y in VALUES:
                for z in ROUND:
                    with self.subTest(x=hex(x), y=hex(y), z=hex(z)):
                        self.assertEqual(run_op(x, y, z), ref(x, y, z))

    def test_seeded(self):
        rng = random.Random(0x1E8)
        for _ in range(40):
            x, y, z = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(x=hex(x), y=hex(y), z=hex(z)):
                self.assertEqual(run_op(x, y, z), ref(x, y, z))

    def test_scalar_and_inline_operands(self):
        self.assertEqual(run_lines(["s_mov_b32 s2, 0x10203040", "v_mov_b32 v1, 0x30405060", "v_lerp_u8 v0, s2, v1, 1"]),
                         ref(0x10203040, 0x30405060, 1))

    def test_clamp_is_rejected_not_guessed(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_mov_b32 v1, 1", "v_lerp_u8 v0, v1, v1, v1 clamp"])


if __name__ == "__main__":
    unittest.main()
