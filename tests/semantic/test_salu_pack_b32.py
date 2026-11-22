"""Semantic interpreter tests for RDNA2 scalar 16-bit pack instructions.

This suite tests the following SOP2 scalar ALU instructions:
  - s_pack_ll_b32_b16 D, S0, S1: D = (S1[15:0] << 16) | S0[15:0]
  - s_pack_lh_b32_b16 D, S0, S1: D = (S1[31:16] << 16) | S0[15:0]
  - s_pack_hh_b32_b16 D, S0, S1: D = (S1[31:16] << 16) | S0[31:16]

Hardware semantics (RDNA2 ISA / SOP2 encoding):
  - D[15:0] receives a 16-bit halfword selected from source S0:
      - 'l' in first position: selects S0[15:0] (bits 15..0)
      - 'h' in first position: selects S0[31:16] (bits 31..16)
  - D[31:16] receives a 16-bit halfword selected from source S1:
      - 'l' in second position: selects S1[15:0] (bits 15..0)
      - 'h' in second position: selects S1[31:16] (bits 31..16)
  - The scalar condition code (SCC) is NOT written and remains unaltered.

Shaders are translated via tools/rdna2spirv.py and executed using
tests/semantic/spirv_mini_interp.py against a plain-Python reference oracle.

Standard library only; zero game data, zero proprietary assets.
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

MASK32 = 0xFFFFFFFF
MASK16 = 0xFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def ref_pack_ll_b32_b16(s0: int, s1: int) -> int:
    """Reference oracle for s_pack_ll_b32_b16: D = (S1[15:0] << 16) | S0[15:0]."""
    return (((s1 & MASK16) << 16) | (s0 & MASK16)) & MASK32


def ref_pack_lh_b32_b16(s0: int, s1: int) -> int:
    """Reference oracle for s_pack_lh_b32_b16: D = (S1[31:16] << 16) | S0[15:0]."""
    return ((s1 & (MASK16 << 16)) | (s0 & MASK16)) & MASK32


def ref_pack_hh_b32_b16(s0: int, s1: int) -> int:
    """Reference oracle for s_pack_hh_b32_b16: D = (S1[31:16] << 16) | S0[31:16]."""
    return ((s1 & (MASK16 << 16)) | ((s0 >> 16) & MASK16)) & MASK32


PACK_OPS = {
    "s_pack_ll_b32_b16": ref_pack_ll_b32_b16,
    "s_pack_lh_b32_b16": ref_pack_lh_b32_b16,
    "s_pack_hh_b32_b16": ref_pack_hh_b32_b16,
}


def ref_pack(op: str, s0: int, s1: int) -> int:
    """Dispatch to the corresponding reference oracle function."""
    if op not in PACK_OPS:
        raise ValueError(f"Unknown pack opcode: {op}")
    return PACK_OPS[op](s0, s1)


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_pack_b32.s")
    dxbc_path = os.path.join(_TMP_DIR.name, "ps_salu_pack_b32.dxbc")
    with open(dxbc_path, "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        try:
            _TMP_DIR.cleanup()
        except Exception:
            pass
        _TMP_DIR = None


def run_assembly_lines(lines: list[str]) -> tuple[int, int]:
    """Assemble synthetic shader lines, run via mini interpreter, return (v0, v1)."""
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is None:
        setUpModule()
    assert _SRC_PATH is not None
    asm = (
        "; test_salu_pack_b32\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v0, v0 done vm\n"
        + "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    out = si.run_fragment(spv)
    mrt0 = out["%out_mrt0"]
    return (mrt0[0], mrt0[1])


def run_pack_op(op: str, s0: int, s1: int, scc_in: int = 0) -> tuple[int, int]:
    """Execute pack operation with SCC pre-condition, return (result, scc_after)."""
    lines = [
        f"s_mov_b32 s2, {s0 & MASK32:#x}",
        f"s_mov_b32 s3, {s1 & MASK32:#x}",
        f"s_mov_b32 s4, {scc_in}",
        "s_cmp_eq_u32 s4, 1",
        f"{op} s0, s2, s3",
        "s_cselect_b32 s1, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
    ]
    return run_assembly_lines(lines)


class TestSaluPackB32ReferenceOracle(unittest.TestCase):
    """Unit tests verifying the Python reference oracle against hand-computed values."""

    def test_oracle_hand_computed_bit_patterns(self):
        s0 = 0x12345678  # S0: hi=0x1234, lo=0x5678
        s1 = 0x9ABCDEF0  # S1: hi=0x9ABC, lo=0xDEF0

        # s_pack_ll_b32_b16: hi=S1[15:0]=0xDEF0, lo=S0[15:0]=0x5678
        self.assertEqual(ref_pack_ll_b32_b16(s0, s1), 0xDEF05678)
        self.assertEqual(ref_pack("s_pack_ll_b32_b16", s0, s1), 0xDEF05678)

        # s_pack_lh_b32_b16: hi=S1[31:16]=0x9ABC, lo=S0[15:0]=0x5678
        self.assertEqual(ref_pack_lh_b32_b16(s0, s1), 0x9ABC5678)
        self.assertEqual(ref_pack("s_pack_lh_b32_b16", s0, s1), 0x9ABC5678)

        # s_pack_hh_b32_b16: hi=S1[31:16]=0x9ABC, lo=S0[31:16]=0x1234
        self.assertEqual(ref_pack_hh_b32_b16(s0, s1), 0x9ABC1234)
        self.assertEqual(ref_pack("s_pack_hh_b32_b16", s0, s1), 0x9ABC1234)

    def test_oracle_halfword_extraction_and_masking(self):
        # High half of s0 must not affect s_pack_ll or s_pack_lh
        val_a = ref_pack_ll_b32_b16(0x0000BEEF, 0x0000CAFE)
        val_b = ref_pack_ll_b32_b16(0xFFFFBEEF, 0x0000CAFE)
        self.assertEqual(val_a, val_b)
        self.assertEqual(val_a, 0xCAFEBEEF)

        # Low half of s0 must not affect s_pack_hh
        val_h1 = ref_pack_hh_b32_b16(0x11110000, 0x22220000)
        val_h2 = ref_pack_hh_b32_b16(0x1111FFFF, 0x22220000)
        self.assertEqual(val_h1, val_h2)
        self.assertEqual(val_h1, 0x22221111)

        # Low half of s1 must not affect s_pack_lh or s_pack_hh
        val_lh1 = ref_pack_lh_b32_b16(0x00003333, 0x44440000)
        val_lh2 = ref_pack_lh_b32_b16(0x00003333, 0x4444FFFF)
        self.assertEqual(val_lh1, val_lh2)
        self.assertEqual(val_lh1, 0x44443333)

    def test_oracle_32bit_clamping(self):
        # Oracle handles operands exceeding 32-bit representations cleanly
        res = ref_pack_ll_b32_b16(0x1_0000_1234, 0x2_0000_5678)
        self.assertEqual(res, 0x56781234)

    def test_oracle_unknown_opcode_rejection(self):
        with self.assertRaises(ValueError):
            ref_pack("s_pack_hl_b32_b16", 0, 0)


class TestSaluPackB32ExecutionBasics(unittest.TestCase):
    """Basic semantic execution tests validating rdna2spirv and spirv_mini_interp."""

    def test_basic_execution_matches_oracle(self):
        s0, s1 = 0xA1B2C3D4, 0xE5F60718
        for op in PACK_OPS:
            expected = ref_pack(op, s0, s1)
            got, _ = run_pack_op(op, s0, s1, scc_in=0)
            self.assertEqual(got, expected, f"Failed on op={op}")


if __name__ == "__main__":
    unittest.main()
