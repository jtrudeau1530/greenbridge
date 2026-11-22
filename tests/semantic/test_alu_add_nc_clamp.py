"""Semantic interpreter tests for RDNA2 v_add_nc_i32 / v_sub_nc_i32 and the integer clamp bit.

v_add_nc_i32 / v_sub_nc_i32: D = S0 + S1 / S0 - S1, wrapping at 32 bits. With the VOP3 clamp bit an integer
add/sub saturates instead (GFX9+; LLVM selects uaddsat/usubsat to v_add/sub_nc_u32 clamp and saddsat/ssubsat to
v_add/sub_nc_i32 clamp): the unsigned ops to [0, 0xFFFFFFFF], the signed ones to [0x80000000, 0x7FFFFFFF].
v_subrev_nc_u32 is S1 - S0 and saturates the same way. Each case translates a synthetic pixel shader with
tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the result with a plain-Python
reference. D leaves through exp as two 16-bit halves ORed into 0x40000000, so no export channel is a NaN or denormal
pattern that a float round trip could change.
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


def s32(x: int) -> int:
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def ref(op: str, x: int, y: int, clamp: bool) -> int:
    if op == "v_subrev_nc_u32":
        x, y, op = y, x, "v_sub_nc_u32"
    signed = op.endswith("_i32")
    a, b = (s32(x), s32(y)) if signed else (x & M32, y & M32)
    r = a - b if "_sub" in op else a + b
    if clamp:
        lo, hi = (-(1 << 31), (1 << 31) - 1) if signed else (0, M32)
        r = min(max(r, lo), hi)
    return r & M32


OPS = ["v_add_nc_i32", "v_sub_nc_i32", "v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32"]
VALUES = [0, 1, 2, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF, 0xFFFFFFFE, 0x40000000, 0xC0000000, 0x12345678]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_add_nc_clamp.s")
    with open(os.path.join(_TMP_DIR.name, "ps_add_nc_clamp.dxbc"), "wb") as f:
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
    asm = "; test_add_nc_clamp\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16


def run_op(op: str, x: int, y: int, clamp: bool) -> int:
    return run_lines([f"v_mov_b32 v1, {x:#x}", f"v_mov_b32 v2, {y:#x}",
                      f"{op}_e64 v0, v1, v2" + (" clamp" if clamp else "")])


class AddNcClampTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref("v_add_nc_i32", 0x7FFFFFFF, 1, False), 0x80000000)
        self.assertEqual(ref("v_add_nc_i32", 0x7FFFFFFF, 1, True), 0x7FFFFFFF)
        self.assertEqual(ref("v_sub_nc_i32", 0x80000000, 1, True), 0x80000000)
        self.assertEqual(ref("v_add_nc_u32", M32, 2, True), M32)
        self.assertEqual(ref("v_sub_nc_u32", 1, 2, True), 0)
        self.assertEqual(ref("v_subrev_nc_u32", 2, 1, True), 0)

    def test_values(self):
        for op in OPS:
            for clamp in (False, True):
                for x in VALUES:
                    for y in VALUES[::2]:
                        with self.subTest(op=op, clamp=clamp, x=hex(x), y=hex(y)):
                            self.assertEqual(run_op(op, x, y, clamp), ref(op, x, y, clamp))

    def test_seeded(self):
        rng = random.Random(0xC1A)
        for _ in range(40):
            op, clamp = rng.choice(OPS), bool(rng.getrandbits(1))
            x, y = rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(op=op, clamp=clamp, x=hex(x), y=hex(y)):
                self.assertEqual(run_op(op, x, y, clamp), ref(op, x, y, clamp))

    def test_vop2_forms_wrap(self):
        # The VOP2 encodings have no clamp bit: they wrap.
        self.assertEqual(run_lines(["v_mov_b32 v1, -1", "v_add_nc_u32_e32 v0, 2, v1"]), 1)
        self.assertEqual(run_lines(["v_mov_b32 v1, 1", "v_sub_nc_u32_e32 v0, 0, v1"]), M32)


if __name__ == "__main__":
    unittest.main()
