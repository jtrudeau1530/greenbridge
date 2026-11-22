"""Texel offsets of image_sample*_o / image_gather4*_o in tools/rdna2spirv.py: the shape of the SPIR-V.

These tests feed the translator hand-written ``.s`` listings, so they need neither llvm-mc nor spirv-tools.

docs/design.md, "Texel offsets": the first address dword packs signed 6-bit offsets, x = [5:0], y = [13:8],
z = [21:16], which become the Offset image operand. Sampled values are checked on a GPU by
runtime/test/shaders/ps_texoff.asm. That test cannot see the sign extension on a driver that itself keeps only the
low six bits of an Offset: on an AMD Radeon RX 6800 (driver 26.11.2) a translator that read the fields unsigned
(60 for -4) sampled the same texels. So the extraction is pinned down here: OpBitFieldSExtract, bit 8k, 6 bits wide.
"""
import re

import pytest

from listing import translate_listing

# v10 = the offset dword, v11.. = the coordinates (and LOD); v3 is what listing.EPILOGUE exports.
PROLOGUE = "\n".join(f"v_mov_b32 v{r}, 0" for r in (3, 10, 11, 12, 13, 14))
DESC = "s[8:15], s[16:19]"
FORMS = [   # (instruction, SPIR-V op that takes the Offset, offset components)
    (f"image_sample_lz_o v[4:5], v[10:12], {DESC} dmask:0x3 dim:SQ_RSRC_IMG_2D", "OpImageSampleExplicitLod", 2),
    (f"image_sample_l_o v[4:5], v[10:13], {DESC} dmask:0x3 dim:SQ_RSRC_IMG_2D", "OpImageSampleExplicitLod", 2),
    (f"image_sample_o v[4:5], v[10:12], {DESC} dmask:0x3 dim:SQ_RSRC_IMG_2D", "OpImageSampleImplicitLod", 2),
    (f"image_sample_lz_o v[4:5], v[10:13], {DESC} dmask:0x3 dim:SQ_RSRC_IMG_3D", "OpImageSampleExplicitLod", 3),
    (f"image_gather4_lz_o v[4:7], v[10:12], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D", "OpImageGather", 2),
]


def offset_fields(spv, op):
    """[(first bit, width, signed)] of the fields that make up the Offset operand of the one ``op`` in ``spv``."""
    consts = {m[1]: int(m[2]) for m in re.finditer(r"(%\w+) = OpConstant %uint (\d+)", spv)}
    line, = [ln for ln in spv.splitlines() if f"= {op} " in ln]
    assert "Offset" in line and "ConstOffset" not in line
    vec = line.split()[-1]                      # Offset is the last image operand: its id ends the line
    parts = re.search(rf"{re.escape(vec)} = OpCompositeConstruct %v\dint (.+)", spv)[1].split()
    fields = []
    for part in parts:
        bits = re.search(rf"{re.escape(part)} = OpBitcast %int (%\w+)", spv)[1]
        ext = re.search(rf"{re.escape(bits)} = OpBitField([SU])Extract %uint %\w+ (%\w+) (%\w+)", spv)
        fields.append((consts[ext[2]], consts[ext[3]], ext[1] == "S"))
    return fields


@pytest.mark.parametrize("asm,op,n", FORMS, ids=[f[0].split()[0] + "_" + f[0].rsplit("_", 1)[1] for f in FORMS])
def test_offset_is_signed_six_bit_fields_at_byte_steps(asm, op, n, tmp_path):
    spv = translate_listing(tmp_path, PROLOGUE + "\n" + asm)
    assert offset_fields(spv, op) == [(8 * k, 6, True) for k in range(n)]


def test_sample_without_o_has_no_offset(tmp_path):
    spv = translate_listing(tmp_path, PROLOGUE + f"\nimage_sample_lz v[4:5], v[11:12], {DESC} dmask:0x3 dim:SQ_RSRC_IMG_2D")
    assert "Offset" not in "".join(ln for ln in spv.splitlines(True) if "OpImageSample" in ln)
