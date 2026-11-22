from __future__ import annotations

"""Semantic interpreter tests for RDNA2 3-way integer min/max instructions:
- v_min3_i32: signed two's complement 32-bit minimum: min(S0, S1, S2)
- v_min3_u32: unsigned 32-bit integer minimum: min(S0, S1, S2)
- v_max3_i32: signed two's complement 32-bit maximum: max(S0, S1, S2)
- v_max3_u32: unsigned 32-bit integer maximum: max(S0, S1, S2)

Features reference models, pairwise reduction equivalence, signed vs unsigned
divergence on identical bit patterns, and shader execution via rdna2spirv and
spirv_mini_interp.

Zero game data, standard library Python only, Python 3.9+ compatible.
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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
SIGN_BIT32 = 0x80000000
INT32_MIN = -0x80000000
INT32_MAX = 0x7FFFFFFF
UINT32_MAX = 0xFFFFFFFF

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OPS = ("v_min3_i32", "v_min3_u32", "v_max3_i32", "v_max3_u32")


def to_signed_i32(x: int) -> int:
    """Interpret raw 32-bit unsigned pattern as two's complement signed 32-bit integer."""
    x &= MASK32
    return x - (1 << 32) if (x & SIGN_BIT32) else x


def to_unsigned_u32(x: int) -> int:
    """Mask integer to 32-bit unsigned integer range [0, 2^32 - 1]."""
    return x & MASK32


def ref_v_min3_i32(s0: int, s1: int, s2: int) -> int:
    """Reference implementation of v_min3_i32: signed minimum of 3 operands."""
    v0 = to_signed_i32(s0)
    v1 = to_signed_i32(s1)
    v2 = to_signed_i32(s2)
    return min(v0, v1, v2) & MASK32


def ref_v_min3_u32(s0: int, s1: int, s2: int) -> int:
    """Reference implementation of v_min3_u32: unsigned minimum of 3 operands."""
    v0 = to_unsigned_u32(s0)
    v1 = to_unsigned_u32(s1)
    v2 = to_unsigned_u32(s2)
    return min(v0, v1, v2) & MASK32


def ref_v_max3_i32(s0: int, s1: int, s2: int) -> int:
    """Reference implementation of v_max3_i32: signed maximum of 3 operands."""
    v0 = to_signed_i32(s0)
    v1 = to_signed_i32(s1)
    v2 = to_signed_i32(s2)
    return max(v0, v1, v2) & MASK32


def ref_v_max3_u32(s0: int, s1: int, s2: int) -> int:
    """Reference implementation of v_max3_u32: unsigned maximum of 3 operands."""
    v0 = to_unsigned_u32(s0)
    v1 = to_unsigned_u32(s1)
    v2 = to_unsigned_u32(s2)
    return max(v0, v1, v2) & MASK32


def ref_pairwise_min3_i32(s0: int, s1: int, s2: int) -> int:
    """Cascaded pairwise signed min: min(min(s0, s1), s2)."""
    step1 = min(to_signed_i32(s0), to_signed_i32(s1))
    return min(step1, to_signed_i32(s2)) & MASK32


def ref_pairwise_min3_u32(s0: int, s1: int, s2: int) -> int:
    """Cascaded pairwise unsigned min: min(min(s0, s1), s2)."""
    step1 = min(to_unsigned_u32(s0), to_unsigned_u32(s1))
    return min(step1, to_unsigned_u32(s2)) & MASK32


def ref_pairwise_max3_i32(s0: int, s1: int, s2: int) -> int:
    """Cascaded pairwise signed max: max(max(s0, s1), s2)."""
    step1 = max(to_signed_i32(s0), to_signed_i32(s1))
    return max(step1, to_signed_i32(s2)) & MASK32


def ref_pairwise_max3_u32(s0: int, s1: int, s2: int) -> int:
    """Cascaded pairwise unsigned max: max(max(s0, s1), s2)."""
    step1 = max(to_unsigned_u32(s0), to_unsigned_u32(s1))
    return max(step1, to_unsigned_u32(s2)) & MASK32


# Backwards-compatible alias
s32 = to_signed_i32


def ref(op: str, *xs: int) -> int:
    """Plain-Python reference; operands and result are raw 32-bit patterns."""
    if len(xs) == 3:
        if op == "v_min3_i32":
            return ref_v_min3_i32(xs[0], xs[1], xs[2])
        if op == "v_min3_u32":
            return ref_v_min3_u32(xs[0], xs[1], xs[2])
        if op == "v_max3_i32":
            return ref_v_max3_i32(xs[0], xs[1], xs[2])
        if op == "v_max3_u32":
            return ref_v_max3_u32(xs[0], xs[1], xs[2])
    key = to_signed_i32 if op.endswith("_i32") else to_unsigned_u32
    pick = min if op.startswith("v_min3") else max
    return pick(xs, key=key) & MASK32


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

_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_min3.s")
    with open(os.path.join(_TMP_DIR.name, "ps_min3.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


def run(op: str, *xs: int, inline: tuple[int, ...] = ()) -> int:
    """Translate and run `op v0, a, b, c`.

    Operands are moved into v1..v3 unless their index is in `inline`, in which
    case the value is written into the instruction as an inline constant.
    Returns the mrt0 word.
    """
    if _TMP_DIR is None:
        setUpModule()
    assert _SRC_PATH is not None
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


class TestAluIntMin3Max3ReferenceOracle(unittest.TestCase):
    """Rigorous tests for the reference oracle models and mathematical properties."""

    def test_signed_vs_unsigned_divergence_core(self) -> None:
        """Verify fundamental bit pattern divergences between signed and unsigned min/max."""
        # 0x0 vs 0xFFFFFFFF (-1 signed, UINT_MAX unsigned)
        self.assertEqual(ref_v_min3_i32(0, 0xFFFFFFFF, 10), 0xFFFFFFFF)  # -1 is minimum
        self.assertEqual(ref_v_min3_u32(0, 0xFFFFFFFF, 10), 0)           # 0 is minimum
        self.assertEqual(ref_v_max3_i32(0, 0xFFFFFFFF, 10), 10)          # 10 is maximum (-1 < 0 < 10)
        self.assertEqual(ref_v_max3_u32(0, 0xFFFFFFFF, 10), 0xFFFFFFFF)  # UINT_MAX is maximum

        # 0x7FFFFFFF (INT_MAX) vs 0x80000000 (INT_MIN signed, 2^31 unsigned)
        self.assertEqual(ref_v_min3_i32(0x7FFFFFFF, 0x80000000, 0), 0x80000000)
        self.assertEqual(ref_v_min3_u32(0x7FFFFFFF, 0x80000000, 0), 0)
        self.assertEqual(ref_v_max3_i32(0x7FFFFFFF, 0x80000000, 0), 0x7FFFFFFF)
        self.assertEqual(ref_v_max3_u32(0x7FFFFFFF, 0x80000000, 0), 0x80000000)

    def test_pairwise_reduction_equivalence(self) -> None:
        """3-way min/max must identically match chained 2-way pairwise min/max."""
        sample_vectors = [
            (0, 0, 0),
            (1, 2, 3),
            (-5, 10, -20),
            (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
            (0x80000000, 0x80000001, 0x7FFFFFFE),
            (0x12345678, 0x87654321, 0x00000000),
            (0xFFFFFFFE, 0xFFFFFFFF, 0x00000001),
        ]
        for s0, s1, s2 in sample_vectors:
            with self.subTest(s0=hex(s0 & MASK32), s1=hex(s1 & MASK32), s2=hex(s2 & MASK32)):
                self.assertEqual(ref_v_min3_i32(s0, s1, s2), ref_pairwise_min3_i32(s0, s1, s2))
                self.assertEqual(ref_v_min3_u32(s0, s1, s2), ref_pairwise_min3_u32(s0, s1, s2))
                self.assertEqual(ref_v_max3_i32(s0, s1, s2), ref_pairwise_max3_i32(s0, s1, s2))
                self.assertEqual(ref_v_max3_u32(s0, s1, s2), ref_pairwise_max3_u32(s0, s1, s2))

    def test_idempotent_property(self) -> None:
        """min3(x, x, x) == x and max3(x, x, x) == x for all x."""
        test_values = [0, 1, -1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 42, -9999]
        for val in test_values:
            raw = val & MASK32
            self.assertEqual(ref_v_min3_i32(val, val, val), raw)
            self.assertEqual(ref_v_min3_u32(val, val, val), raw)
            self.assertEqual(ref_v_max3_i32(val, val, val), raw)
            self.assertEqual(ref_v_max3_u32(val, val, val), raw)

    def test_dispatch_ref_wrapper(self) -> None:
        """Verify the generic ref dispatch wrapper matches direct reference functions."""
        for op, func in (
            ("v_min3_i32", ref_v_min3_i32),
            ("v_min3_u32", ref_v_min3_u32),
            ("v_max3_i32", ref_v_max3_i32),
            ("v_max3_u32", ref_v_max3_u32),
        ):
            for t in TRIPLES:
                self.assertEqual(ref(op, *t), func(*t))


class ReferenceTests(unittest.TestCase):
    """The reference against hand-worked values, so a failure points at the translator, not the oracle."""

    def test_reference(self) -> None:
        self.assertEqual(ref("v_min3_i32", 5, 0x80000000, 7), 0x80000000)  # INT_MIN is the smallest i32
        self.assertEqual(ref("v_min3_u32", 5, 0x80000000, 7), 5)
        self.assertEqual(ref("v_max3_i32", 5, 0x80000000, 7), 7)
        self.assertEqual(ref("v_max3_u32", 5, 0x80000000, 7), 0x80000000)
        self.assertEqual(ref("v_min3_i32", 0xFFFFFFFF, 0, 1), 0xFFFFFFFF)  # -1
        self.assertEqual(ref("v_max3_u32", 0xFFFFFFFF, 0, 1), 0xFFFFFFFF)
        self.assertEqual(ref("v_max3_i32", 0xFFFFFFFF, 0, 1), 1)
        self.assertEqual(ref("v_min3_u32", 0xFFFFFFFF, 0, 1), 0)


class Min3Max3Tests(unittest.TestCase):
    def test_every_order(self) -> None:
        for op in OPS:
            for xs in TRIPLES:
                for perm in sorted(set(itertools.permutations(xs))):
                    with self.subTest(op=op, xs=[hex(v) for v in perm]):
                        self.assertEqual(run(op, *perm), ref(op, *perm))

    def test_sweep(self) -> None:
        values = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
        for op in OPS:
            for xs in itertools.product(values, repeat=3):
                with self.subTest(op=op, xs=[hex(v) for v in xs]):
                    self.assertEqual(run(op, *xs), ref(op, *xs))

    def test_inline_constants(self) -> None:
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
