"""Semantic interpreter tests for buffer_load_format_* (the V# FORMAT decode in tools/rdna2spirv.py).

The element is read from PhysicalStorageBuffer memory at V#.base + offset. An element under 4 bytes starts at a byte
shift inside its first dword, and the translation merges dword 1 above it: (w0 >> bsh) | (w1 << (32 - bsh)), picking
w0 when bsh is 0. The interpreter rejects shifts of 32 or more (undefined in SPIR-V, even on the discarded side of an
OpSelect), so every dword-aligned load here also checks that the merge keeps its shift below 32.

Integer and f16 formats are used: their decode is exact, so the expected words need no rounding model.
"""
from __future__ import annotations

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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
BASE = 0x1000
MEMORY = {BASE: 0x11223344, BASE + 4: 0xDEADBEEF, BASE + 8: 0xCAFEF00D, BASE + 12: 0x0BADC0DE}
# V# FORMAT codes (word3 bits 12-18), from rdna2spirv.FORMATS
FMT_U32, FMT_U32X2, FMT_U32X4, FMT_UINT8X4, FMT_UINT16X2, FMT_UINT16X4, FMT_F16 = 20, 62, 75, 60, 27, 69, 13
FMT_F32, FMT_F32X2, FMT_F32X3, FMT_UNORM8, FMT_UNORM8X4 = 22, 64, 74, 1, 56
DST_SEL_XYZW = 4 | 5 << 3 | 6 << 6 | 7 << 9
# DST_SEL (X, 0, 1, 1): code 4, 0, 1, 1 — SHADER-86 measures float 1.0 vs integer 1
DST_SEL_X_0_1_1 = 4 | 0 << 3 | 1 << 6 | 1 << 9

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_buf_fmt.s")
    with open(os.path.join(_TMP_DIR.name, "ps_buf_fmt.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def load_format(fmt: int, offset: int, ncomp: int = 4, dst_sel: int = DST_SEL_XYZW,
                memory: dict | None = None) -> tuple:
    """buffer_load_format_<x..xyzw> of format `fmt` at byte `offset`; returns the four mrt0 words."""
    if _TMP_DIR is None:
        setUpModule()
    suffix = "xyzw"[:ncomp]
    vdata = "v2" if ncomp == 1 else f"v[2:{1 + ncomp}]"
    regs = [f"v{2 + min(i, ncomp - 1)}" for i in range(4)]
    asm = (
        "; test_buffer_load_format\n"
        f"\ts_mov_b32 s0, {BASE:#x}\n"
        "\ts_mov_b32 s1, 0\n"                  # base high bits 0, stride 0 (raw)
        "\ts_mov_b32 s2, 0x100\n"              # num_records (bytes)
        f"\ts_mov_b32 s3, {fmt << 12 | dst_sel:#x}\n"
        f"\tv_mov_b32 v1, {offset:#x}\n"
        f"\tbuffer_load_format_{suffix} {vdata}, v1, s[0:3], 0 offen\n"
        "\ts_waitcnt vmcnt(0)\n"
        f"\texp mrt0 {', '.join(regs)} done vm\n"
        "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv, memory=dict(memory if memory is not None else MEMORY))["%out_mrt0"]


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f16_word(h: int) -> int:
    return struct.unpack("<I", struct.pack("<f", struct.unpack("<e", struct.pack("<H", h))[0]))[0]


class DwordAlignedElements(unittest.TestCase):
    def test_u32(self):
        for off, want in ((0, 0x11223344), (4, 0xDEADBEEF), (12, 0x0BADC0DE)):
            with self.subTest(off=off):
                self.assertEqual(load_format(FMT_U32, off, 1)[0], want)

    def test_u32x2_and_x4(self):
        self.assertEqual(load_format(FMT_U32X2, 4, 2)[:2], (0xDEADBEEF, 0xCAFEF00D))
        self.assertEqual(load_format(FMT_U32X4, 0), (0x11223344, 0xDEADBEEF, 0xCAFEF00D, 0x0BADC0DE))

    def test_uint8x4(self):
        self.assertEqual(load_format(FMT_UINT8X4, 4), (0xEF, 0xBE, 0xAD, 0xDE))

    def test_uint16x2_and_x4(self):
        self.assertEqual(load_format(FMT_UINT16X2, 4, 2)[:2], (0xBEEF, 0xDEAD))
        self.assertEqual(load_format(FMT_UINT16X4, 4), (0xBEEF, 0xDEAD, 0xF00D, 0xCAFE))


class SubDwordElements(unittest.TestCase):
    def test_f16_in_each_half(self):
        for off, half in ((4, 0xBEEF), (6, 0xDEAD), (8, 0xF00D), (10, 0xCAFE)):
            with self.subTest(off=off):
                self.assertEqual(load_format(FMT_F16, off, 1)[0], f16_word(half))


class CrossingDwordBoundary(unittest.TestCase):
    """An element that starts mid-dword must read the following dwords, not re-read dword 0.

    #234 left this open: `need` used to be ceil(size/4), so a 4-byte format at byte 3
    (or an 8-byte format at byte 6) took its tail from the first dword. MEMORY is
    0x11223344, 0xDEADBEEF, 0xCAFEF00D, 0x0BADC0DE at 0x1000..0x100C.
    """

    def test_uint8x4_at_byte_3(self):
        # Bytes at 0x1003..0x1006: 0x11, 0xEF, 0xBE, 0xAD.
        self.assertEqual(load_format(FMT_UINT8X4, 3), (0x11, 0xEF, 0xBE, 0xAD))

    def test_uint8x4_at_byte_1(self):
        # Bytes at 0x1001..0x1004: 0x33, 0x22, 0x11, 0xEF.
        self.assertEqual(load_format(FMT_UINT8X4, 1), (0x33, 0x22, 0x11, 0xEF))

    def test_uint16x2_at_byte_2(self):
        # 4-byte element at 0x1002: 0x1122, then 0xBEEF (low half of the next dword).
        self.assertEqual(load_format(FMT_UINT16X2, 2, 2)[:2], (0x1122, 0xBEEF))

    def test_uint16x4_at_byte_6(self):
        # 8-byte element at 0x1006: 0xDEAD, 0xF00D, 0xCAFE, 0xC0DE.
        self.assertEqual(load_format(FMT_UINT16X4, 6), (0xDEAD, 0xF00D, 0xCAFE, 0xC0DE))


class MissingComponentsRepeat(unittest.TestCase):
    """SHADER-86 / RX 6700: missing channels are c mod n, not 0 or 1."""

    def test_one_component_formats_xxxx(self):
        self.assertEqual(load_format(FMT_U32, 4), (0xDEADBEEF,) * 4)
        self.assertEqual(load_format(FMT_F32, 4), (0xDEADBEEF,) * 4)
        # 8_UNORM at offset 4: low byte 0xEF → 239/255; all four channels match.
        want = f32_bits(239.0 / 255.0)
        self.assertEqual(load_format(FMT_UNORM8, 4), (want,) * 4)

    def test_two_and_three_component_xyxy_and_xyzx(self):
        self.assertEqual(load_format(FMT_U32X2, 4), (0xDEADBEEF, 0xCAFEF00D, 0xDEADBEEF, 0xCAFEF00D))
        self.assertEqual(load_format(FMT_F32X2, 4), (0xDEADBEEF, 0xCAFEF00D, 0xDEADBEEF, 0xCAFEF00D))
        self.assertEqual(load_format(FMT_F32X3, 0), (0x11223344, 0xDEADBEEF, 0xCAFEF00D, 0x11223344))


class UnormCorrectlyRounded(unittest.TestCase):
    """SHADER-86: UNORM is OpFDiv by (2^n-1), not a reciprocal multiply. Raw 3 → 0x3c40c0c1."""

    def test_unorm8x4_raw_3(self):
        mem = {BASE: 0x03030303}
        want = 0x3C40C0C1
        self.assertEqual(load_format(FMT_UNORM8X4, 0, memory=mem), (want,) * 4)
        # f32 reciprocal multiply disagrees (BetterJdue: 504 mismatches on rgba8).
        recip = f32_bits(struct.unpack("<f", struct.pack("<f", 3.0))[0]
                         * struct.unpack("<f", struct.pack("<f", 1.0 / 255.0))[0])
        self.assertNotEqual(want, recip)


class DstSelOneByFormat(unittest.TestCase):
    """SHADER-86: DST_SEL 1 is 1.0f for float/unorm, integer 1 for UINT."""

    def test_dst_sel_one_float_vs_uint(self):
        # DST_SEL (X, 0, 1, 1): Z and W come from the constant-1 code.
        f32 = load_format(FMT_F32, 4, dst_sel=DST_SEL_X_0_1_1)
        self.assertEqual(f32, (0xDEADBEEF, 0, 0x3F800000, 0x3F800000))
        u32 = load_format(FMT_U32, 4, dst_sel=DST_SEL_X_0_1_1)
        self.assertEqual(u32, (0xDEADBEEF, 0, 1, 1))
        unorm = load_format(FMT_UNORM8, 4, dst_sel=DST_SEL_X_0_1_1)
        self.assertEqual(unorm[2:], (0x3F800000, 0x3F800000))
        u8 = load_format(FMT_UINT8X4, 4, dst_sel=DST_SEL_X_0_1_1)
        self.assertEqual(u8[2:], (1, 1))


class InterpreterConversions(unittest.TestCase):
    """The ops the format decode needs from spirv_mini_interp: integer to float and GLSL FMin/FMax."""

    HEADER = (
        "OpCapability Shader\n%glsl = OpExtInstImport \"GLSL.std.450\"\nOpMemoryModel Logical GLSL450\n"
        "OpEntryPoint Fragment %main \"main\" %out\n%void = OpTypeVoid\n%fn = OpTypeFunction %void\n"
        "%uint = OpTypeInt 32 0\n%float = OpTypeFloat 32\n%_ptr_Output_uint = OpTypePointer Output %uint\n"
        "%out = OpVariable %_ptr_Output_uint Output\n"
    )

    def run_expr(self, consts, body):
        text = (self.HEADER + consts + "%main = OpFunction %void None %fn\n%entry = OpLabel\n" + body
                + "%bits = OpBitcast %uint %r\nOpStore %out %bits\nOpReturn\nOpFunctionEnd\n")
        return si.run_fragment(text)["%out"][0]

    def test_convert_u_and_s_to_f(self):
        consts = "%c = OpConstant %uint 4294967295\n"
        self.assertEqual(self.run_expr(consts, "%r = OpConvertUToF %float %c\n"), 0x4F800000)   # 4294967296.0
        self.assertEqual(self.run_expr(consts, "%r = OpConvertSToF %float %c\n"), 0xBF800000)   # -1.0

    def test_fmin_fmax(self):
        consts = "%a = OpConstant %float 1.5\n%b = OpConstant %float -2\n"
        self.assertEqual(self.run_expr(consts, "%r = OpExtInst %float %glsl FMax %a %b\n"), 0x3FC00000)
        self.assertEqual(self.run_expr(consts, "%r = OpExtInst %float %glsl FMin %a %b\n"), 0xC0000000)

    def test_fdiv_correctly_rounded(self):
        consts = "%a = OpConstant %float 3\n%b = OpConstant %float 255\n"
        self.assertEqual(self.run_expr(consts, "%r = OpFDiv %float %a %b\n"), 0x3C40C0C1)


if __name__ == "__main__":
    unittest.main()
