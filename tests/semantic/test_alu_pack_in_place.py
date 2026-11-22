"""Semantic interpreter tests for AMD RDNA in-place 16-bit integer packing (v_pack_b32_u16).

AMD RDNA VALU Instruction Set Architecture (ISA):
  v_pack_b32_u16 dst, src0, src1 [op_sel:[s0, s1]] [clamp]
  Operation: Packs two 16-bit integer values into a 32-bit vector register (VGPR).
    - dst[15:0]  = half_word(src0, op_sel[0]) [clamped to [0, 0xFFFF] if clamp]
    - dst[31:16] = half_word(src1, op_sel[1]) [clamped to [0, 0xFFFF] if clamp]

In-place semantics:
  When dst aliases src0 or src1:
    - dst = v_pack_b32_u16 dst, src: dst[15:0] is preserved (op_sel[0]=0) while
      dst[31:16] is updated with src's selected half-word (in-place upper pack).
    - dst = v_pack_b32_u16 src, dst: dst[31:16] is preserved (op_sel[1]=1) while
      dst[15:0] is updated with src's selected half-word (in-place lower pack).
    - dst = v_pack_b32_u16 dst, dst: self-permutation (swapping halves, broadcasting
      half-words, or identity).

Modifiers and lowerings:
  - op_sel:[s0, s1]:
      s0=0 selects bits [15:0], s0=1 selects bits [31:16] of src0.
      s1=0 selects bits [15:0], s1=1 selects bits [31:16] of src1.
  - clamp: Unsigned saturation to [0, 65535].
  - Lowering to SPIR-V:
      1. Shift / bitwise mask / OR sequence with OpExtInst GLSL.std.450 UClamp / SClamp.
      2. Bitfield extract / insert sequence with OpBitFieldUExtract / OpBitFieldInsert.

Standard library only: Python 3.9+ compatible, zero game data, zero proprietary assets.
"""
from __future__ import annotations

import itertools
import os
import random
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF


def s32(x: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    x &= MASK32
    return x - (1 << 32) if x & 0x80000000 else x


def ref_extract_half(val: int, hi: bool = False) -> int:
    """Extract low or high 16-bit half-word with 16-bit masking."""
    val &= MASK32
    return ((val >> 16) if hi else val) & MASK16


def ref_clamp_u16(val: int, signed: bool = False) -> int:
    """Clamp integer value to unsigned 16-bit range [0, 65535]."""
    if signed:
        s = s32(val)
        if s < 0:
            return 0
        if s > MASK16:
            return MASK16
        return s & MASK16
    u = val & MASK32
    return min(u, MASK16)


def ref_pack_b32_u16(
    s0: int,
    s1: int,
    op_sel: tuple[int, int] = (0, 0),
    clamp: bool = False,
    signed: bool = False,
) -> int:
    """Pure-Python reference oracle for RDNA2 v_pack_b32_u16 integer packing."""
    s0 &= MASK32
    s1 &= MASK32
    val0 = (s0 >> 16) if op_sel[0] else s0
    val1 = (s1 >> 16) if op_sel[1] else s1

    if clamp:
        h0 = ref_clamp_u16(val0, signed=signed)
        h1 = ref_clamp_u16(val1, signed=signed)
    else:
        h0 = val0 & MASK16
        h1 = val1 & MASK16

    return (h0 & MASK16) | ((h1 & MASK16) << 16)


def ref_pack_in_place(
    dst: int,
    src: int,
    target: str = "hi",
    op_sel: tuple[int, int] = (0, 0),
    clamp: bool = False,
) -> int:
    """Pure-Python reference for in-place packing into an existing register value."""
    dst &= MASK32
    src &= MASK32
    if target == "hi":
        # dst = v_pack_b32_u16 dst, src (replaces high half of dst)
        return ref_pack_b32_u16(dst, src, op_sel=op_sel, clamp=clamp)
    elif target == "lo":
        # dst = v_pack_b32_u16 src, dst (replaces low half of dst)
        return ref_pack_b32_u16(src, dst, op_sel=op_sel, clamp=clamp)
    elif target == "both":
        # dst = v_pack_b32_u16 dst, dst (self-pack / permute)
        return ref_pack_b32_u16(dst, dst, op_sel=op_sel, clamp=clamp)
    raise ValueError(f"Unknown in-place target: {target}")


def ref_decompose_pack(dword: int) -> tuple[int, int]:
    """Decompose a 32-bit packed word into (lo_u16, hi_u16)."""
    dword &= MASK32
    return dword & MASK16, (dword >> 16) & MASK16


def run_spirv_pack_logical(
    s0: int,
    s1: int,
    op_sel: tuple[int, int] = (0, 0),
    clamp: bool = False,
    signed: bool = False,
) -> int:
    """Execute SPIR-V sequence modeling v_pack_b32_u16 via shift/bitwise/clamp ops."""
    s0_u = s0 & MASK32
    s1_u = s1 & MASK32

    h0_extract = (
        "%s0_val = OpShiftRightLogical %uint %c_s0 %c16\n"
        if op_sel[0]
        else "%s0_val = OpCopyObject %uint %c_s0\n"
    )
    h1_extract = (
        "%s1_val = OpShiftRightLogical %uint %c_s1 %c16\n"
        if op_sel[1]
        else "%s1_val = OpCopyObject %uint %c_s1\n"
    )

    if clamp:
        clamp_op = "SClamp" if signed else "UClamp"
        h0_proc = f"%h0 = OpExtInst %uint %glsl {clamp_op} %s0_val %c0 %c65535\n"
        h1_proc = f"%h1 = OpExtInst %uint %glsl {clamp_op} %s1_val %c0 %c65535\n"
    else:
        h0_proc = "%h0 = OpBitwiseAnd %uint %s0_val %c65535\n"
        h1_proc = "%h1 = OpBitwiseAnd %uint %s1_val %c65535\n"

    spv = f"""OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_s0 = OpConstant %uint {s0_u}
%c_s1 = OpConstant %uint {s1_u}
%c0 = OpConstant %uint 0
%c16 = OpConstant %uint 16
%c65535 = OpConstant %uint 65535
%main = OpFunction %void None %fn
%entry = OpLabel
{h0_extract}{h0_proc}{h1_extract}{h1_proc}%h1_hi = OpShiftLeftLogical %uint %h1 %c16
%res = OpBitwiseOr %uint %h0 %h1_hi
OpStore %out %res
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(spv)["%out"][0]


def run_spirv_pack_bitfield(
    s0: int,
    s1: int,
    op_sel: tuple[int, int] = (0, 0),
) -> int:
    """Execute SPIR-V sequence modeling unclamped v_pack_b32_u16 via bitfield ops."""
    off0 = 16 if op_sel[0] else 0
    off1 = 16 if op_sel[1] else 0
    spv = f"""OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_s0 = OpConstant %uint {s0 & MASK32}
%c_s1 = OpConstant %uint {s1 & MASK32}
%c16 = OpConstant %uint 16
%c_off0 = OpConstant %uint {off0}
%c_off1 = OpConstant %uint {off1}
%main = OpFunction %void None %fn
%entry = OpLabel
%h0 = OpBitFieldUExtract %uint %c_s0 %c_off0 %c16
%h1 = OpBitFieldUExtract %uint %c_s1 %c_off1 %c16
%res = OpBitFieldInsert %uint %h0 %h1 %c16 %c16
OpStore %out %res
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(spv)["%out"][0]


class TestPackB32U16ReferenceOracle(unittest.TestCase):
    """Unit tests for pure-Python reference oracle and SPIR-V lowerings."""

    def test_basic_halfword_packing(self):
        self.assertEqual(ref_pack_b32_u16(0, 0), 0)
        self.assertEqual(ref_pack_b32_u16(0x1234, 0x5678), 0x56781234)
        self.assertEqual(ref_pack_b32_u16(0x00FF, 0xFF00), 0xFF0000FF)
        self.assertEqual(ref_pack_b32_u16(0xAAAA, 0x5555), 0x5555AAAA)

    def test_in_place_hi_half_replacement(self):
        # dst = 0xDEADBEEF, pack src=0xCAFE into high half while preserving low half
        dst = 0xDEADBEEF
        src = 0x0000CAFE
        result = ref_pack_in_place(dst, src, target="hi", op_sel=(0, 0))
        self.assertEqual(result, 0xCAFEBEEF)
        self.assertEqual(result & MASK16, dst & MASK16)
        self.assertEqual(result >> 16, src & MASK16)

    def test_in_place_lo_half_replacement(self):
        # dst = 0xDEADBEEF, pack src=0xCAFE into low half while preserving high half
        dst = 0xDEADBEEF
        src = 0x0000CAFE
        result = ref_pack_in_place(dst, src, target="lo", op_sel=(0, 1))
        self.assertEqual(result, 0xDEADCAFE)
        self.assertEqual(result >> 16, dst >> 16)
        self.assertEqual(result & MASK16, src & MASK16)

    def test_in_place_self_permutation(self):
        val = 0x12345678
        # Broadcast low half
        self.assertEqual(ref_pack_in_place(val, val, target="both", op_sel=(0, 0)), 0x56785678)
        # Broadcast high half
        self.assertEqual(ref_pack_in_place(val, val, target="both", op_sel=(1, 1)), 0x12341234)
        # Swap halves
        self.assertEqual(ref_pack_in_place(val, val, target="both", op_sel=(1, 0)), 0x56781234)
        # Identity
        self.assertEqual(ref_pack_in_place(val, val, target="both", op_sel=(0, 1)), 0x12345678)

    def test_op_sel_truth_table(self):
        s0 = 0xA1B2C3D4
        s1 = 0xE5F60718
        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(0, 0)), 0x0718C3D4)
        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(1, 0)), 0x0718A1B2)
        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(0, 1)), 0xE5F6C3D4)
        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(1, 1)), 0xE5F6A1B2)

    def test_unsigned_clamping(self):
        self.assertEqual(ref_pack_b32_u16(100, 200, clamp=True), (200 << 16) | 100)
        self.assertEqual(ref_pack_b32_u16(65535, 65535, clamp=True), 0xFFFFFFFF)
        self.assertEqual(ref_pack_b32_u16(65536, 10, clamp=True), (10 << 16) | 0xFFFF)
        self.assertEqual(ref_pack_b32_u16(10, 70000, clamp=True), (0xFFFF << 16) | 10)
        self.assertEqual(ref_pack_b32_u16(0xFFFFFFFF, 0xFFFFFFFF, clamp=True), 0xFFFFFFFF)
        self.assertEqual(ref_pack_b32_u16(0x80000000, 0x7FFFFFFF, clamp=True), 0xFFFFFFFF)

    def test_signed_clamping(self):
        self.assertEqual(ref_pack_b32_u16(-1, 500, clamp=True, signed=True), 500 << 16)
        self.assertEqual(ref_pack_b32_u16(500, -100, clamp=True, signed=True), 500)
        self.assertEqual(ref_pack_b32_u16(0x80000000, 0x80000000, clamp=True, signed=True), 0)
        self.assertEqual(ref_pack_b32_u16(70000, -5, clamp=True, signed=True), 0x0000FFFF)
        self.assertEqual(ref_pack_b32_u16(-5, 0x7FFFFFFF, clamp=True, signed=True), 0xFFFF0000)

    def test_clamping_with_op_sel(self):
        s0 = 50
        s1 = 100000
        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(0, 0), clamp=True), (0xFFFF << 16) | 50)
        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(1, 0), clamp=True), 0xFFFF0000)
        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(0, 1), clamp=True), (1 << 16) | 50)
        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(1, 1), clamp=True), 1 << 16)

    def test_decomposition_roundtrip(self):
        test_pairs = [(0, 0), (0x1234, 0x5678), (0xFFFF, 0x0001), (0x8000, 0x7FFF)]
        for lo, hi in test_pairs:
            dword = ref_pack_b32_u16(lo, hi)
            self.assertEqual(ref_decompose_pack(dword), (lo, hi))

    def test_spirv_logical_matches_oracle(self):
        vectors = [
            (0x1234, 0x5678),
            (0xA1B2C3D4, 0xE5F60718),
            (0xFFFFFFFF, 0x00000000),
            (0x80000000, 0x7FFFFFFF),
        ]
        for s0, s1 in vectors:
            for op_sel in itertools.product((0, 1), repeat=2):
                with self.subTest(s0=hex(s0), s1=hex(s1), op_sel=op_sel):
                    expected = ref_pack_b32_u16(s0, s1, op_sel=op_sel, clamp=False)
                    got = run_spirv_pack_logical(s0, s1, op_sel=op_sel, clamp=False)
                    self.assertEqual(got, expected)

    def test_spirv_bitfield_matches_oracle(self):
        vectors = [
            (0x0000, 0xFFFF),
            (0x12345678, 0x9ABCDEF0),
            (0x5555AAAA, 0xAAAA5555),
        ]
        for s0, s1 in vectors:
            for op_sel in itertools.product((0, 1), repeat=2):
                with self.subTest(s0=hex(s0), s1=hex(s1), op_sel=op_sel):
                    expected = ref_pack_b32_u16(s0, s1, op_sel=op_sel, clamp=False)
                    got = run_spirv_pack_bitfield(s0, s1, op_sel=op_sel)
                    self.assertEqual(got, expected)

    def test_spirv_clamping_parity(self):
        clamp_vectors = [(0, 0), (100, 200), (70000, 10), (0xFFFFFFFF, 0xFFFFFFFF)]
        for s0, s1 in clamp_vectors:
            for op_sel in itertools.product((0, 1), repeat=2):
                with self.subTest(s0=hex(s0), s1=hex(s1), op_sel=op_sel):
                    expected = ref_pack_b32_u16(s0, s1, op_sel=op_sel, clamp=True)
                    got = run_spirv_pack_logical(s0, s1, op_sel=op_sel, clamp=True)
                    self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
