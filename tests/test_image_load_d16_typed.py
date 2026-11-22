"""d16 image_load reads integer-format images through their uint/sint views, like the 32-bit image_load.

The runtime binds UINT/SINT images only into the uint/sint storage arrays (docs/design.md, Typed image loads), and a
float sampled view of an integer image is undefined in Vulkan. A d16 load used to skip the T# FORMAT branch and always
read the float sampled view, then pack f16: a Texture2D<min16uint> Load came back as garbage.
"""
import re

import pytest

from conftest import requires_tools

pytestmark = requires_tools


@pytest.mark.parametrize("mask,dst", [("0x1", "v3"), ("0x3", "v3"), ("0x7", "v[3:4]")])
def test_d16_image_load_reads_integer_views(make_shader, mask, dst):
    sh = make_shader("v_mov_b32 v1, 0\nv_mov_b32 v2, 0\n"
                     f"image_load {dst}, v[1:2], s[8:15] dmask:{mask} dim:SQ_RSRC_IMG_2D unorm d16\n"
                     "s_waitcnt vmcnt(0)\nexp mrt0 v3, v4, v3, v4 done vm\ns_endpgm\n", name="ps_d16_load")
    spv = sh.translate()
    reads = re.findall(r"OpImageRead %(\w+) ", spv)
    assert "v4uint_st" in reads and "v4int_st" in reads, "d16 image_load never reads the uint/sint views"
    assert "PackHalf2x16" not in spv      # the float branch keeps the integer RTE f16 pack (#489)
    sh.validate(spv)
