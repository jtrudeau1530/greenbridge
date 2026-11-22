from __future__ import annotations

"""Regression and semantic tests for bitwise and shift constant folding in the translator.

Verifies that:
- Constant and immediate shifts fold the modulo 32 mask at translate time (no runtime OpBitwiseAnd %c31).
- Constant v_bfe_u32/i32 offset and width fold directly into OpBitField[US]Extract operands (no OpISub/OpULessThan/OpSelect).
- Constant v_alignbyte_b32 with sel=0 folds directly to the low word; non-zero sel folds into exact shifts and OpBitwiseOr.
- Constant v_alignbit_b32 with shift=0 folds directly to the low word; non-zero shift folds into exact shifts and OpBitwiseOr.
- dedupe_types terminates early upon reaching function definitions and correctly renames aliased types across the shader.
"""
import os
import re
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
import support  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


class BitopConstantFoldingTests(unittest.TestCase):
    def translate_lines(self, lines: list[str]) -> str:
        body = "\n".join("\t" + line for line in lines)
        asm = f"; bitop_test\n{body}\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
        with tempfile.TemporaryDirectory() as tmp:
            d_dir = os.path.join(tmp, "disasm")
            os.mkdir(d_dir)
            s_path = os.path.join(d_dir, "ps_bitop.s")
            with open(s_path, "w", encoding="utf-8") as f:
                f.write(asm)
            with open(os.path.join(tmp, "ps_bitop.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(s_path)

    def test_constant_shift_eliminates_runtime_mask(self):
        lines = [
            "v_mov_b32 v1, 0x1234",
            "v_lshlrev_b32 v0, 4, v1",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpBitwiseAnd %uint", spv)
        self.assertIn("OpShiftLeftLogical %uint", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x12340)

    def test_constant_bfe_eliminates_runtime_bounds(self):
        lines = [
            "v_mov_b32 v1, 0x12345678",
            "v_bfe_u32 v0, v1, 8, 12",
        ]
        spv = self.translate_lines(lines)
        # Should directly emit OpBitFieldUExtract without runtime sub, cmp, or bounds select
        self.assertNotIn("OpBitwiseAnd %uint", spv)
        self.assertNotIn("OpISub %uint", spv)
        self.assertNotIn("OpULessThan %bool", spv)
        self.assertIn("OpBitFieldUExtract %uint", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x456)

    def test_constant_alignbyte_zero_shift_folds_directly(self):
        lines = [
            "v_mov_b32 v1, 0xDEADBEEF",
            "v_mov_b32 v2, 0x12345678",
            "v_alignbyte_b32 v0, v1, v2, 0",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpShiftRightLogical", spv)
        self.assertNotIn("OpShiftLeftLogical", spv)
        self.assertNotIn("OpBitwiseOr", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x12345678)

    def test_constant_alignbyte_nonzero_shift_emits_direct_shifts(self):
        lines = [
            "v_mov_b32 v1, 0x8899AABB",
            "v_mov_b32 v2, 0x11223344",
            "v_alignbyte_b32 v0, v1, v2, 1",
        ]
        spv = self.translate_lines(lines)
        # Shift and or are emitted directly without runtime mask or sub
        self.assertNotIn("OpBitwiseAnd %uint", spv)
        self.assertNotIn("OpISub %uint", spv)
        self.assertIn("OpShiftRightLogical %uint", spv)
        self.assertIn("OpShiftLeftLogical %uint", spv)
        self.assertIn("OpBitwiseOr %uint", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0xBB112233)

    def test_constant_alignbit_zero_shift_folds_directly(self):
        lines = [
            "v_mov_b32 v1, 0xDEADBEEF",
            "v_mov_b32 v2, 0x12345678",
            "v_alignbit_b32 v0, v1, v2, 0",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpShiftRightLogical", spv)
        self.assertNotIn("OpShiftLeftLogical", spv)
        self.assertNotIn("OpBitwiseOr", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x12345678)

    def test_constant_alignbit_nonzero_shift_emits_direct_shifts(self):
        lines = [
            "v_mov_b32 v1, 0x12345678",
            "v_mov_b32 v2, 0x9ABCDEF0",
            "v_alignbit_b32 v0, v1, v2, 16",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpBitwiseAnd %uint", spv)
        self.assertNotIn("OpISub %uint", spv)
        self.assertIn("OpShiftRightLogical %uint", spv)
        self.assertIn("OpShiftLeftLogical %uint", spv)
        self.assertIn("OpBitwiseOr %uint", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x56789ABC)

    def test_dedupe_types_merges_duplicate_types_and_substitutes(self):
        spv_sample = (
            "OpCapability Shader\n"
            "%uint = OpTypeInt 32 0\n"
            "%v4uint = OpTypeVector %uint 4\n"
            "%v4uint_st = OpTypeVector %uint 4\n"
            "%ptr_v4uint = OpTypePointer StorageBuffer %v4uint\n"
            "%ptr_v4uint_st = OpTypePointer StorageBuffer %v4uint_st\n"
            "%fn = OpTypeFunction %void\n"
            "%main = OpFunction %void None %fn\n"
            "%entry = OpLabel\n"
            "%var = OpVariable %ptr_v4uint_st StorageBuffer\n"
            "%val = OpLoad %v4uint_st %var\n"
            "OpReturn\n"
            "OpFunctionEnd"
        )
        deduped = rdna2spirv.dedupe_types(spv_sample)
        self.assertNotIn("%v4uint_st =", deduped)
        self.assertNotIn("%ptr_v4uint_st =", deduped)
        self.assertIn("%var = OpVariable %ptr_v4uint StorageBuffer", deduped)
        self.assertIn("%val = OpLoad %v4uint %var", deduped)


if __name__ == "__main__":
    unittest.main()
