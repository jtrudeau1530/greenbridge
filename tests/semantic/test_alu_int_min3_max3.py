"""Semantic interpreter tests for the RDNA2 three-operand integer min/max (v_min3_i32, v_min3_u32, v_max3_i32,
v_max3_u32).

Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py
and compares the mrt0 word with a plain-Python reference. v_min3_i32 and v_max3_u32 had no value test; v_min3_u32 and
v_max3_i32 had one value each in ps_ops. (The two-operand v_min/v_max and v_mul_lo_u32 are left to the FUZZ-1
differential run, #325.)

The signed/unsigned split is the point of most cases: 0xFFFFFFFF is the largest u32 but -1 as i32, and 0x80000000
is the smallest i32. The translator folds the operands as op(op(x, y), z), so every case runs with its operands in
every order, and with inline-constant operands.
"""
import itertools
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
OPS = ("v_min3_i32", "v_min3_u32", "v_max3_i32", "v_max3_u32")

# Triples where signed and unsigned order disagree, plus ties and ordinary values.
TRIPLES = (
    (5, 0x80000000, 7),
    (0xFFFFFFFF, 0, 1),
    (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
    (0x80000001, 0x80000000, 0x7FFFFFFE),
    (1, 0xFFFFFFFE, 0xFFFFFFFF),
    (9, 9, 0x80000000),
    (0xFFFFFFFF, 0xFFFFFFFF, 0),
    (3, 3, 3),
)


def s32(x: int) -> int:
    x &= MASK32
    return x - (1 << 32) if x & 0x80000000 else x


def ref(op: str, *xs: int) -> int:
    """Plain-Python reference; operands and result are raw 32-bit patterns."""
    key = s32 if op.endswith("_i32") else (lambda v: v & MASK32)
    pick = min if op.startswith("v_min3") else max
    return pick(xs, key=key) & MASK32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_min3.s")
    with open(os.path.join(_TMP_DIR.name, "ps_min3.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run(op: str, *xs, inline=()) -> int:
    """Translate and run `op v0, a, b, c`. Operands are moved into v1..v3 unless their index is in `inline`, in
    which case the value is written into the instruction as an inline constant. Returns the mrt0 word."""
    if _TMP_DIR is None:
        setUpModule()
    lines, operands = [], []
    for i, x in enumerate(xs):
        if i in inline:
            operands.append(str(x))
        else:
            lines.append(f"\tv_mov_b32 v{i + 1}, {x & MASK32:#x}")
            operands.append(f"v{i + 1}")
    lines.append(f"\t{op} v0, {', '.join(operands)}")
    asm = "; test_int_min3_max3\n" + "\n".join(lines) + "\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class ReferenceTests(unittest.TestCase):
    """The reference against hand-worked values, so a failure points at the translator, not the oracle."""

    def test_reference(self):
        self.assertEqual(ref("v_min3_i32", 5, 0x80000000, 7), 0x80000000)  # INT_MIN is the smallest i32
        self.assertEqual(ref("v_min3_u32", 5, 0x80000000, 7), 5)
        self.assertEqual(ref("v_max3_i32", 5, 0x80000000, 7), 7)
        self.assertEqual(ref("v_max3_u32", 5, 0x80000000, 7), 0x80000000)
        self.assertEqual(ref("v_min3_i32", 0xFFFFFFFF, 0, 1), 0xFFFFFFFF)  # -1
        self.assertEqual(ref("v_max3_u32", 0xFFFFFFFF, 0, 1), 0xFFFFFFFF)
        self.assertEqual(ref("v_max3_i32", 0xFFFFFFFF, 0, 1), 1)
        self.assertEqual(ref("v_min3_u32", 0xFFFFFFFF, 0, 1), 0)


class Min3Max3Tests(unittest.TestCase):
    def test_every_order(self):
        for op in OPS:
            for xs in TRIPLES:
                for perm in sorted(set(itertools.permutations(xs))):
                    with self.subTest(op=op, xs=[hex(v) for v in perm]):
                        self.assertEqual(run(op, *perm), ref(op, *perm))

    def test_sweep(self):
        values = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
        for op in OPS:
            for xs in itertools.product(values, repeat=3):
                with self.subTest(op=op, xs=[hex(v) for v in xs]):
                    self.assertEqual(run(op, *xs), ref(op, *xs))

    def test_inline_constants(self):
        # -16 as an inline constant is 0xFFFFFFF0: below 64 as i32, above it as u32.
        cases = [
            ("v_min3_i32", (0x7FFFFFFF, -16, 64), 0xFFFFFFF0),
            ("v_min3_u32", (0x7FFFFFFF, -16, 64), 64),
            ("v_max3_i32", (0x80000000, -16, 64), 64),
            ("v_max3_u32", (0x80000000, -16, 64), 0xFFFFFFF0),
        ]
        for op, xs, want in cases:
            with self.subTest(op=op):
                self.assertEqual(want, ref(op, *xs))
                self.assertEqual(run(op, *xs, inline=(1, 2)), want)


if __name__ == "__main__":
    unittest.main()
