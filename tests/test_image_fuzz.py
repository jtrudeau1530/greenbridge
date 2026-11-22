"""CPU checks for tools/image_fuzz.py: the oracle, the texel mapping and the generated shader, no GPU.

The GPU side (vk_image_fuzz, rdna2spirv.py) is what the fuzzer compares against; these tests make sure the
oracle it compares with is right, so a mismatch is the translator's and not the fuzzer's.
"""
import json
import os
import random
import struct
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import image_fuzz as imf  # noqa: E402


def test_format_codes_match_the_translator_table():
    with open(os.path.join(imf.TOOLS, "gfx10_formats.json"), encoding="utf-8") as f:
        by_name = {name: int(code) for code, name in json.load(f).items()}
    gfx10_names = {
        "R8G8B8A8_UNORM": "GFX10_FORMAT_8_8_8_8_UNORM",
        "R16G16B16A16_SFLOAT": "GFX10_FORMAT_16_16_16_16_FLOAT",
        "R32G32B32A32_SFLOAT": "GFX10_FORMAT_32_32_32_32_FLOAT",
        "R32_SFLOAT": "GFX10_FORMAT_32_FLOAT",
    }
    assert set(gfx10_names) == set(imf.FORMATS)
    for name, (_, code, bpp, model) in imf.FORMATS.items():
        assert by_name[gfx10_names[name]] == code, name
        assert bpp in (4, 8, 16) and model in ("unorm8", "half4", "float4", "float1"), name


def test_unorm_oracle_divides_by_255():
    texels = bytes([0, 1, 127, 255, 10, 20, 30, 40])
    got = imf.oracle("R8G8B8A8_UNORM", texels, 2, 1)
    assert got[0] == [0.0, imf.f32(1 / 255.0), imf.f32(127 / 255.0), 1.0]
    assert got[1] == [imf.f32(v / 255.0) for v in (10, 20, 30, 40)]


def test_half_oracle_reads_four_binary16():
    wanted = [-2.0, -0.0, 0.5, 65504.0]
    texels = b"".join(struct.pack("<e", v) for v in wanted)
    got = imf.oracle("R16G16B16A16_SFLOAT", texels, 1, 1)[0]
    assert got == wanted


def test_float_oracle_and_missing_components():
    texels = struct.pack("<4f", 1.5, -2.25, 1024.0, float("inf"))
    got = imf.oracle("R32G32B32A32_SFLOAT", texels, 1, 1)[0]
    assert got == [1.5, -2.25, 1024.0, float("inf")]
    one = struct.pack("<f", -7.0)
    assert imf.oracle("R32_SFLOAT", one, 1, 1)[0] == [-7.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("size", imf.SIZES)
def test_texel_mapping_over_every_size(size):
    for p in (0, 1, 17, 63):
        assert imf.texel_of_pixel(p, p, size, size) == (p * size // 64, p * size // 64)
    assert imf.texel_of_pixel(63, 0, size, size)[0] == size - 1


def test_pixel_shader_is_tabbed_and_carries_the_size():
    asm = imf.pixel_shader(16, 8)
    code = [line for line in asm.splitlines() if line and not line.startswith(";")]
    assert all(line.startswith("\t") for line in code), code
    assert "\timage_load v[4:7], v[2:3], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D" in asm
    assert "\tv_mul_f32_e32 v2, 16.0, v2" in asm and "\tv_mul_f32_e32 v3, 8.0, v3" in asm


def test_tsharp_word1_carries_the_format_and_the_size():
    rng = random.Random(3)
    for name, (_, code, bpp, _) in imf.FORMATS.items():
        texels, tsharp = imf.make_texels(name, 8, 16, rng)
        assert len(texels) == 8 * 16 * bpp
        assert tsharp[1] == code << 20
        assert tsharp[2] == 7 | (15 << 16)


def test_self_test_runs():
    r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "image_fuzz.py"), "--self-test"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "self-test: ok" in r.stdout
