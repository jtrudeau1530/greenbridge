"""Semantic interpreter tests for RDNA2 v_lshlrev_b64, v_lshrrev_b64 and v_ashrrev_i64.

All three shift the 64-bit register pair S1 by S0 & 63 ("rev": the shift amount is the first source):
  v_lshlrev_b64: D = S1 << (S0 & 63)
  v_lshrrev_b64: D = S1 >> (S0 & 63), zeros shifted in
  v_ashrrev_i64: D = S1 >> (S0 & 63), copies of bit 63 shifted in
Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py
and compares the exported pair with a plain-Python reference.
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


def s64(x: int) -> int:
    x &= M64
    return x - (1 << 64) if x >> 63 else x


OPS = {
    "v_lshlrev_b64": lambda n, x: (x << (n & 63)) & M64,
    "v_lshrrev_b64": lambda n, x: (x & M64) >> (n & 63),
    "v_ashrrev_i64": lambda n, x: (s64(x) >> (n & 63)) & M64,
}
SHIFTS = [0, 1, 31, 32, 33, 63, 64, 65, 0xFFFFFFFF, 0x80000020]
VALUES = [0, 1, M64, 1 << 63, (1 << 63) - 1, 0xFFFFFFFF, 1 << 32, 0x8000000000000001, 0x0123456789ABCDEF,
          0xFEDCBA9876543210]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_shift_b64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_shift_b64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> int:
    """Run the lines, export v[0:1], return the pair as one 64-bit value."""
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_shift_b64\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    w = si.run_fragment(spv)["%out_mrt0"]
    return w[0] | w[1] << 32


def run_op(op: str, n: int, x: int) -> int:
    return run_lines([f"v_mov_b32 v4, {n:#x}", f"v_mov_b32 v2, {x & 0xFFFFFFFF:#x}", f"v_mov_b32 v3, {x >> 32:#x}",
                      f"{op} v[0:1], v4, v[2:3]"])


class ShiftB64Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(OPS["v_lshrrev_b64"](32, 1 << 63), 0x80000000)
        self.assertEqual(OPS["v_ashrrev_i64"](32, 1 << 63), 0xFFFFFFFF80000000)
        self.assertEqual(OPS["v_ashrrev_i64"](63, 1 << 63), M64)
        self.assertEqual(OPS["v_lshrrev_b64"](64, M64), M64)              # amount 64 & 63 = 0

    def test_values(self):
        for op, fn in OPS.items():
            for n in SHIFTS:
                for x in VALUES:
                    with self.subTest(op=op, n=hex(n), x=hex(x)):
                        self.assertEqual(run_op(op, n, x), fn(n, x))

    def test_seeded(self):
        rng = random.Random(64)
        for _ in range(30):
            op, n, x = rng.choice(list(OPS)), rng.getrandbits(32), rng.getrandbits(64)
            with self.subTest(op=op, n=hex(n), x=hex(x)):
                self.assertEqual(run_op(op, n, x), OPS[op](n, x))

    def test_scalar_pair_and_inline_operands(self):
        # Source pair from SGPRs, amount from an inline constant.
        got = run_lines(["s_mov_b32 s2, 0", "s_mov_b32 s3, 0x80000000", "v_ashrrev_i64 v[0:1], 4, s[2:3]"])
        self.assertEqual(got, 0xF800000000000000)
        got = run_lines(["s_mov_b32 s2, 0", "s_mov_b32 s3, 0x80000000", "v_lshrrev_b64 v[0:1], 4, s[2:3]"])
        self.assertEqual(got, 0x0800000000000000)


if __name__ == "__main__":
    unittest.main()
