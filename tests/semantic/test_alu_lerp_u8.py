from __future__ import annotations

"""Semantic interpreter tests for AMD RDNA2 packed 8-bit linear interpolation (v_lerp_u8).

AMD RDNA VALU Instruction Set Architecture (ISA):
  v_lerp_u8 dst, src0, src1, src2

Operation:
  Packs four 8-bit unsigned linear interpolations into a 32-bit vector register:
  For each byte lane i in {0, 1, 2, 3}:
    s0_i = (src0 >> (8 * i)) & 0xFF
    s1_i = (src1 >> (8 * i)) & 0xFF
    s2_i = (src2 >> (8 * i)) & 0x01   # only bit 0 of each byte lane contributes
    sum_i = s0_i + s1_i + s2_i        # range: 0 .. 511 (9 bits)
    dst_i = sum_i >> 1                # range: 0 .. 255 (8 bits)

Arithmetic Properties:
  - Range isolation: max sum 255 + 255 + 1 = 511. 511 >> 1 = 255 (0xFF).
    Result is strictly bound within [0, 255] for every byte lane. No cross-byte
    carry or overflow can occur.
  - Rounding control:
    - s2 bit 0 = 0: floor / truncation of the unsigned average.
    - s2 bit 0 = 1: ceiling / round-half-up of the unsigned average.
  - Identity property:
    - When s0_i == s1_i == k, ((k + k + r) >> 1) == (2k + r) >> 1 == k for both r in {0, 1}.
  - Symmetry / Commutativity:
    - src0 and src1 are interchangeable: v_lerp_u8(a, b, c) == v_lerp_u8(b, a, c).
  - Upper bit discard:
    - Bits [7:1] of each byte lane in src2 are strictly ignored by hardware and translator.

SPIR-V Lowering in rdna2spirv:
  Lowers to 4 parallel lane extractions via OpBitFieldUExtract (offset 0, 8, 16, 24 with
  widths 8 for src0/src1 and width 1 for src2), two OpIAdd operations, OpShiftRightLogical
  by 1, and OpBitwiseOr reassembly into a 32-bit dword.

Standard library only: Python 3.9+ compatible, zero game data, synthetic shaders only.
"""
import gc
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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK8 = 0xFF
MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_lerp_u8.s")
    with open(os.path.join(_TMP_DIR.name, "ps_lerp_u8.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


# ---------------------------------------------------------------------------
# Reference Oracles & Byte Packing Primitives
# ---------------------------------------------------------------------------

def unpack_bytes_u32(val: int) -> tuple[int, int, int, int]:
    """Unpack a 32-bit unsigned word into 4 byte lanes (b0, b1, b2, b3)."""
    val &= MASK32
    return (
        val & MASK8,
        (val >> 8) & MASK8,
        (val >> 16) & MASK8,
        (val >> 24) & MASK8,
    )


def pack_bytes_u32(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack 4 unsigned 8-bit bytes into a 32-bit unsigned word."""
    return (
        (b0 & MASK8)
        | ((b1 & MASK8) << 8)
        | ((b2 & MASK8) << 16)
        | ((b3 & MASK8) << 24)
    )


def ref_lerp_u8_byte(s0: int, s1: int, s2_round: int) -> int:
    """Calculate single byte lane lerp: ((s0 + s1 + (s2 & 1)) >> 1) & 0xFF.

    The intermediate sum ranges from 0 to 511. Right shifting by 1 produces
    a value in [0, 255] with no carry out of the 8-bit boundary.
    """
    b0 = s0 & MASK8
    b1 = s1 & MASK8
    r = s2_round & 1
    return (b0 + b1 + r) >> 1


def ref_lerp_u8_word(s0: int, s1: int, s2: int) -> int:
    """Full 32-bit packed word reference oracle for v_lerp_u8 across all 4 lanes."""
    b0_0, b0_1, b0_2, b0_3 = unpack_bytes_u32(s0)
    b1_0, b1_1, b1_2, b1_3 = unpack_bytes_u32(s1)
    b2_0, b2_1, b2_2, b2_3 = unpack_bytes_u32(s2)

    out0 = ref_lerp_u8_byte(b0_0, b1_0, b2_0)
    out1 = ref_lerp_u8_byte(b0_1, b1_1, b2_1)
    out2 = ref_lerp_u8_byte(b0_2, b1_2, b2_2)
    out3 = ref_lerp_u8_byte(b0_3, b1_3, b2_3)

    return pack_bytes_u32(out0, out1, out2, out3)


def ref(x: int, y: int, z: int) -> int:
    """Compatibility alias matching the existing test suite signature."""
    return ref_lerp_u8_word(x, y, z)


def ref_lerp_fixed_point_blend(s0: int, s1: int, round_nearest: bool = True) -> int:
    """Alternative formulation modeling fixed-point average with round-half-up bias."""
    r_word = 0x01010101 if round_nearest else 0
    return ref_lerp_u8_word(s0, s1, r_word)


# ---------------------------------------------------------------------------
# Synthetic Shader Execution Harness
# ---------------------------------------------------------------------------

VALUES = [
    0,
    0xFFFFFFFF,
    0x01010101,
    0xFF00FF00,
    0x00FF00FF,
    0x80808080,
    0x7F7F7F7F,
    0x12345678,
    0xFEDCBA98,
]
ROUND = [0, 0xFFFFFFFF, 0x01010101, 0x01000100, 0xFEFEFEFE]


def run_lines(lines: list[str]) -> int:
    """Run assembly lines, exporting v0 through mrt0 as two 16-bit halves."""
    if _TMP_DIR is None or _SRC_PATH is None:
        setUpModule()
    assert _SRC_PATH is not None

    augmented = list(lines) + [
        "v_and_b32 v5, 0xffff, v0",
        "v_lshrrev_b32 v6, 16, v0",
        "v_or_b32 v5, 0x40000000, v5",
        "v_or_b32 v6, 0x40000000, v6",
    ]
    asm = (
        "; test_lerp_u8\n"
        + "".join(f"\t{ln}\n" for ln in augmented)
        + "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)


def run_op(x: int, y: int, z: int) -> int:
    """Execute v_lerp_u8 on three 32-bit register inputs."""
    return run_lines([
        f"v_mov_b32 v1, {x:#x}",
        f"v_mov_b32 v2, {y:#x}",
        f"v_mov_b32 v3, {z:#x}",
        "v_lerp_u8 v0, v1, v2, v3",
    ])


# ---------------------------------------------------------------------------
# Commit 1 Tests: Reference Oracle and Mathematical Bounds
# ---------------------------------------------------------------------------

class TestLerpU8ReferenceOracle(unittest.TestCase):
    """Unit tests for the reference oracle, byte lane packing, and arithmetic bounds."""

    def test_unpack_and_pack_identity(self) -> None:
        """Unpacking and repacking 32-bit words must preserve all bits."""
        samples = [0, 0xFFFFFFFF, 0x12345678, 0x80706050, 0x00FF00FF, 0xAA55AA55]
        for val in samples:
            b0, b1, b2, b3 = unpack_bytes_u32(val)
            repacked = pack_bytes_u32(b0, b1, b2, b3)
            self.assertEqual(val, repacked)

    def test_byte_level_basic_arithmetic(self) -> None:
        """Single-byte lane calculations with floor and ceiling rounding."""
        # 0 + 0 + 0 >> 1 = 0
        self.assertEqual(ref_lerp_u8_byte(0, 0, 0), 0)
        self.assertEqual(ref_lerp_u8_byte(0, 0, 1), 0)

        # 1 + 2 = 3; 3 >> 1 = 1; (3 + 1) >> 1 = 2
        self.assertEqual(ref_lerp_u8_byte(1, 2, 0), 1)
        self.assertEqual(ref_lerp_u8_byte(1, 2, 1), 2)

        # 255 + 255 = 510; 510 >> 1 = 255; (510 + 1) >> 1 = 255
        self.assertEqual(ref_lerp_u8_byte(255, 255, 0), 255)
        self.assertEqual(ref_lerp_u8_byte(255, 255, 1), 255)

        # 254 + 255 = 509; 509 >> 1 = 254; (509 + 1) >> 1 = 255
        self.assertEqual(ref_lerp_u8_byte(254, 255, 0), 254)
        self.assertEqual(ref_lerp_u8_byte(254, 255, 1), 255)

        # 0 + 255 = 255; 255 >> 1 = 127; (255 + 1) >> 1 = 128
        self.assertEqual(ref_lerp_u8_byte(0, 255, 0), 127)
        self.assertEqual(ref_lerp_u8_byte(0, 255, 1), 128)

    def test_byte_identity_property(self) -> None:
        """When s0 == s1, the output must equal s0 regardless of the round bit."""
        for val in range(256):
            self.assertEqual(ref_lerp_u8_byte(val, val, 0), val)
            self.assertEqual(ref_lerp_u8_byte(val, val, 1), val)

    def test_byte_commutativity(self) -> None:
        """s0 and s1 operands must be commutative across all byte combinations."""
        test_samples = [0, 1, 2, 42, 127, 128, 254, 255]
        for a in test_samples:
            for b in test_samples:
                for r in (0, 1):
                    self.assertEqual(
                        ref_lerp_u8_byte(a, b, r),
                        ref_lerp_u8_byte(b, a, r),
                        f"Non-commutative at ({a}, {b}, {r})",
                    )

    def test_round_bit_behavior(self) -> None:
        """Round bit 1 increments output by exactly 1 when sum is odd, 0 when sum is even."""
        for a in range(256):
            for b in (0, 1, 127, 128, 254, 255):
                sum_ab = a + b
                res0 = ref_lerp_u8_byte(a, b, 0)
                res1 = ref_lerp_u8_byte(a, b, 1)
                if sum_ab % 2 == 1:
                    self.assertEqual(res1, res0 + 1)
                else:
                    self.assertEqual(res1, res0)

    def test_s2_upper_bits_ignored(self) -> None:
        """Bits 1..7 of each byte in src2 must have zero impact on the computation."""
        patterns = [0xFE, 0xFC, 0x80, 0xAA, 0x54, 0x02]
        for p in patterns:
            # When bit 0 is 0, any upper bits must still act as round 0
            self.assertEqual(ref_lerp_u8_byte(10, 20, p), ref_lerp_u8_byte(10, 20, 0))
            # When bit 0 is 1, any upper bits must still act as round 1
            self.assertEqual(ref_lerp_u8_byte(10, 20, p | 1), ref_lerp_u8_byte(10, 20, 1))

        # Word-level validation: masking S2 with 0x01010101 gives identical result
        s0 = 0x12345678
        s1 = 0x9ABCDEF0
        s2 = 0xFEDCBA98
        self.assertEqual(
            ref_lerp_u8_word(s0, s1, s2),
            ref_lerp_u8_word(s0, s1, s2 & 0x01010101),
        )

    def test_byte_lane_isolation_no_carry_leakage(self) -> None:
        """Carry out of lane i (up to 511) must not bleed into lane i+1."""
        # Lane 0 has 255 + 255 + 1 = 511 -> 255. Lane 1 has 0.
        s0 = 0x000000FF
        s1 = 0x000000FF
        s2 = 0x00000001
        res = ref_lerp_u8_word(s0, s1, s2)
        self.assertEqual(res & 0xFF, 0xFF)
        self.assertEqual((res >> 8) & 0xFF, 0x00)

        # Lane 1 has max sum, Lane 2 and Lane 0 have 0.
        s0 = 0x0000FF00
        s1 = 0x0000FF00
        s2 = 0x00000100
        res = ref_lerp_u8_word(s0, s1, s2)
        self.assertEqual(res & 0xFF, 0x00)
        self.assertEqual((res >> 8) & 0xFF, 0xFF)
        self.assertEqual((res >> 16) & 0xFF, 0x00)

        # Lane 2 has max sum.
        s0 = 0x00FF0000
        s1 = 0x00FF0000
        s2 = 0x00010000
        res = ref_lerp_u8_word(s0, s1, s2)
        self.assertEqual((res >> 8) & 0xFF, 0x00)
        self.assertEqual((res >> 16) & 0xFF, 0xFF)
        self.assertEqual((res >> 24) & 0xFF, 0x00)

        # Lane 3 has max sum.
        s0 = 0xFF000000
        s1 = 0xFF000000
        s2 = 0x01000000
        res = ref_lerp_u8_word(s0, s1, s2)
        self.assertEqual((res >> 16) & 0xFF, 0x00)
        self.assertEqual((res >> 24) & 0xFF, 0xFF)


class TestLerpU8SyntheticShaderTranslation(unittest.TestCase):
    """End-to-end SPIR-V translation and spirv_mini_interp execution tests."""

    def test_reference_known_vectors(self) -> None:
        """Verify baseline test vectors against ref_lerp_u8_word."""
        self.assertEqual(ref_lerp_u8_word(0xFF, 0xFF, 1), 0xFF)
        self.assertEqual(ref_lerp_u8_word(0x01, 0x02, 0), 0x01)
        self.assertEqual(ref_lerp_u8_word(0x01, 0x02, 1), 0x02)
        self.assertEqual(ref_lerp_u8_word(0x00000300, 0x00000400, 0x00000100), 0x00000400)
        self.assertEqual(ref_lerp_u8_word(0x01, 0x02, 0xFE), 0x01)

    def test_edge_values(self) -> None:
        """Run grid of edge values through rdna2spirv and spirv_mini_interp."""
        for x in VALUES:
            for y in VALUES:
                for z in ROUND:
                    with self.subTest(x=hex(x), y=hex(y), z=hex(z)):
                        self.assertEqual(run_op(x, y, z), ref_lerp_u8_word(x, y, z))

    def test_seeded(self) -> None:
        """Randomized 32-bit word verification against spirv_mini_interp."""
        rng = random.Random(0x1E8)
        for _ in range(40):
            x, y, z = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(x=hex(x), y=hex(y), z=hex(z)):
                self.assertEqual(run_op(x, y, z), ref_lerp_u8_word(x, y, z))

    def test_scalar_and_inline_operands(self) -> None:
        """Exercise SGPR scalar source and inline constant operand forms."""
        lines = [
            "s_mov_b32 s2, 0x10203040",
            "v_mov_b32 v1, 0x30405060",
            "v_lerp_u8 v0, s2, v1, 1",
        ]
        self.assertEqual(run_lines(lines), ref_lerp_u8_word(0x10203040, 0x30405060, 1))

    def test_clamp_is_rejected_not_guessed(self) -> None:
        """VOP3 clamp modifier is unsupported on v_lerp_u8 and must raise Unsupported."""
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_mov_b32 v1, 1", "v_lerp_u8 v0, v1, v1, v1 clamp"])


if __name__ == "__main__":
    unittest.main()
