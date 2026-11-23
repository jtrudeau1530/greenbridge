"""v_fma_mix_f32 with a 32-bit literal operand whose high half op_sel selects (RDNA2 VOP3P).

A VOP3P literal is one dword after the instruction. llvm-mc prints it as an f16 (its low half) and re-encodes the
high half as 0, so a listing alone loses the half that op_sel[i] = 1 reads. The translator takes the dword from the
code blob next to the listing (fssm/<name>.fssm) when only that half differs.

Synthetic shader: v_fma_mix_f32 v0, 0x3e00, 1.0, 0x3e00 op_sel:[0,0,1] op_sel_hi:[1,0,1] whose literal dword is
0xc1003e00: src0 = f16 low half 1.5, src1 = f32 1.0, src2 = f16 high half -2.5, so fma = 1.5 * 1.0 - 2.5 = -1.0.
Run with: python -I -m unittest discover -s tests/semantic -p test_valu_fma_mix_literal.py
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
# VOP3P v_fma_mix_f32: vdst v0, op_sel 0b100, op_sel_hi 0b101, src0 literal, src1 inline 1.0 (0xf2), src2 literal.
HEAD = bytes.fromhex("006020cc") + bytes.fromhex("ffe4fd0b")
F32 = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]  # noqa: E731


def run(literal: int, blob_literal: int):
    """Translate the listing (which prints `literal`) against a blob holding `blob_literal`; return the export."""
    printed = HEAD + struct.pack("<I", literal)
    blob = HEAD + struct.pack("<I", blob_literal)
    enc = ",".join(f"0x{b:02x}" for b in printed)
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        with open(os.path.join(tmp, "ps_mix_lit.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "ps_mix_lit.fssm"), "wb") as f:
            f.write(blob)
        src = os.path.join(tmp, "disasm", "ps_mix_lit.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; ps_mix_lit  code 0x0..0xc (header), gfx1013, 0 invalid words\n"
                    f"\tv_fma_mix_f32 v0, 0x{literal & 0xFFFF:x}, 1.0, 0x{literal & 0xFFFF:x} "
                    f"op_sel:[0,0,1] op_sel_hi:[1,0,1] ; encoding: [{enc}]\n"
                    "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
    return si.run_fragment(spv)["%out_mrt0"][0]


class FmaMixLiteralHighHalf(unittest.TestCase):
    def test_high_half_comes_from_the_blob_literal(self):
        self.assertEqual(run(0x00003E00, 0xC1003E00), F32(-1.0))

    def test_high_half_zero_reads_plus_zero(self):
        self.assertEqual(run(0x00003E00, 0x00003E00), F32(1.5))

    def test_other_literal_difference_is_still_refused(self):
        # The low half is what llvm-mc printed; a different low half is a real mismatch, not a dropped field.
        with self.assertRaises(rdna2spirv.Unsupported):
            run(0x00003E00, 0xC1003C00)


if __name__ == "__main__":
    unittest.main()
