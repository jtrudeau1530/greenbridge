"""Semantic interpreter tests for buffer_load_format_* (the V# FORMAT decode in tools/rdna2spirv.py).

The element is read from PhysicalStorageBuffer memory at V#.base + offset. An element under 4 bytes starts at a byte
shift inside its first dword, and the translation merges dword 1 above it: (w0 >> bsh) | (w1 << (32 - bsh)), picking
w0 when bsh is 0. The interpreter rejects shifts of 32 or more (undefined in SPIR-V, even on the discarded side of an
OpSelect), so every dword-aligned load here also checks that the merge keeps its shift below 32.

Integer and f16 formats are used: their decode is exact, so the expected words need no rounding model.
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
BASE = 0x1000
MEMORY = {BASE: 0x11223344, BASE + 4: 0xDEADBEEF, BASE + 8: 0xCAFEF00D, BASE + 12: 0x0BADC0DE}
# V# FORMAT codes (word3 bits 12-18), from rdna2spirv.FORMATS
FMT_U32, FMT_U32X2, FMT_U32X4, FMT_UINT8X4, FMT_UINT16X2, FMT_UINT16X4, FMT_F16 = 20, 62, 75, 60, 27, 69, 13
DST_SEL_XYZW = 4 | 5 << 3 | 6 << 6 | 7 << 9

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


def load_format(fmt: int, offset: int, ncomp: int = 4) -> tuple:
    """buffer_load_format_<x..xyzw> of format `fmt` at byte `offset` of MEMORY; returns the four mrt0 words."""
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
        f"\ts_mov_b32 s3, {fmt << 12 | DST_SEL_XYZW:#x}\n"
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
    return si.run_fragment(spv, memory=dict(MEMORY))["%out_mrt0"]


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


if __name__ == "__main__":
    unittest.main()
