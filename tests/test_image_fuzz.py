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
        "R8G8B8A8_UINT": "GFX10_FORMAT_8_8_8_8_UINT",
        "R8G8B8A8_SINT": "GFX10_FORMAT_8_8_8_8_SINT",
    }
    assert set(gfx10_names) == set(imf.FORMATS)
    for name, (_, code, bpp, model) in imf.FORMATS.items():
        assert by_name[gfx10_names[name]] == code, name
        assert bpp in (4, 8, 16) and model in ("unorm8", "half4", "float4", "float1", "uint8", "sint8"), name


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


@pytest.mark.parametrize("mask,want", [
    (0x1, [10.0, 2.0, 3.0, 4.0]),
    (0x2, [20.0, 2.0, 3.0, 4.0]),
    (0x5, [10.0, 30.0, 3.0, 4.0]),
    (0xa, [20.0, 40.0, 3.0, 4.0]),
    (0xf, [10.0, 20.0, 30.0, 40.0]),
])
def test_dmask_packs_selected_components_and_preserves_unwritten_registers(mask, want):
    assert imf.masked_components([10.0, 20.0, 30.0, 40.0], mask) == want


@pytest.mark.parametrize("mask", range(1, 16))
def test_pixel_shader_uses_the_requested_dmask(mask):
    asm = imf.pixel_shader(16, 8, mask)
    assert f"dmask:0x{mask:x}" in asm
    for reg, value in enumerate((1.0, 2.0, 3.0, 4.0), 4):
        assert f"v_mov_b32_e32 v{reg}, {value}" in asm


def test_dmask_schedule_covers_each_format_and_keeps_full_mask_first():
    formats = tuple(imf.FORMATS)
    pairs = {(formats[i % len(formats)], imf.DMASKS[(i // len(formats)) % len(imf.DMASKS)])
             for i in range(len(formats) * len(imf.DMASKS))}
    assert imf.DMASKS[0] == 0xf
    assert pairs == {(fmt, mask) for fmt in formats for mask in range(1, 16)}


@pytest.mark.parametrize("fmt,want,kind", [
    ("R8G8B8A8_UINT", [0, 127, 128, 255], "u32"),
    ("R8G8B8A8_SINT", [0, 127, -128, -1], "i32"),
])
def test_integer_oracle_and_conversion(fmt, want, kind):
    texels = bytes([0, 127, 128, 255]) * 64
    assert imf.oracle(fmt, texels, 8, 8)[0] == want
    asm = imf.pixel_shader(8, 8, 0x5, imf.FORMATS[fmt][3])
    assert f"v_cvt_f32_{kind}_e32 v4, v4" in asm
    assert f"v_cvt_f32_{kind}_e32 v5, v5" in asm
    assert f"v_cvt_f32_{kind}_e32 v6, v6" not in asm
    got = [want[0], want[2], 3.0, 4.0] * (imf.DRAW * imf.DRAW)
    assert not imf.compare(fmt, texels, 8, 8, got, 0x5)
    got[1] = want[1]
    assert imf.compare(fmt, texels, 8, 8, got, 0x5)


def test_compare_detects_wrong_dmask_order_and_unwritten_register():
    width = height = 8
    texels = struct.pack("<4f", 10.0, 20.0, 30.0, 40.0) * (width * height)
    got = [10.0, 30.0, 3.0, 4.0] * (imf.DRAW * imf.DRAW)
    assert imf.compare("R32G32B32A32_SFLOAT", texels, width, height, got, 0x5) == []
    got[1] = 20.0  # wrong component instead of B
    got[6] = 30.0  # second pixel: unwritten register changed
    problems = imf.compare("R32G32B32A32_SFLOAT", texels, width, height, got, 0x5)
    assert [(p[0], p[1]) for p in problems] == [(0, 0), (1, 0)]


def test_tsharp_word1_carries_the_format_and_the_size():
    rng = random.Random(3)
    for name, (_, code, bpp, _) in imf.FORMATS.items():
        texels, tsharp = imf.make_texels(name, 8, 16, rng)
        assert len(texels) == 8 * 16 * bpp
        assert tsharp[1] == code << 20
        assert tsharp[2] == 7 | (15 << 16)


def test_compare_distinguishes_signed_zero():
    """Vonsek's case: +0 where -0 is expected must be a mismatch, with the pixel's coordinates."""
    width = height = 8
    texels = b"".join(struct.pack("<4f", -0.0, 0.25, 0.5, 1.0) for _ in range(width * height))
    expected = imf.oracle("R32G32B32A32_SFLOAT", texels, width, height)
    got = []
    for y in range(imf.DRAW):
        for x in range(imf.DRAW):
            tx, ty = imf.texel_of_pixel(x, y, width, height)
            got.extend(expected[ty * width + tx])
    assert imf.compare("R32G32B32A32_SFLOAT", texels, width, height, got) == []
    got[(7 * imf.DRAW + 5) * 4] = 0.0  # the same texel, red +0 instead of -0
    problems = imf.compare("R32G32B32A32_SFLOAT", texels, width, height, got)
    assert [(p[0], p[1]) for p in problems] == [(5, 7)]


def test_compare_treats_any_nan_as_equal():
    """The bit comparison keeps the old policy: NaN payloads are not modelled."""
    width = height = 8
    texels = b"".join(struct.pack("<4f", float("nan"), 1.0, 2.0, 3.0) for _ in range(width * height))
    expected = imf.oracle("R32G32B32A32_SFLOAT", texels, width, height)
    other_nan = struct.unpack("<f", struct.pack("<I", 0x7FC00001))[0]
    got = []
    for y in range(imf.DRAW):
        for x in range(imf.DRAW):
            tx, ty = imf.texel_of_pixel(x, y, width, height)
            value = list(expected[ty * width + tx])
            value[0] = other_nan
            got.extend(value)
    assert imf.compare("R32G32B32A32_SFLOAT", texels, width, height, got) == []


def run_case_with_harness_result(monkeypatch, tmp_path, returncode, stdout):
    """run_case with the translator stubbed out and vk_image_fuzz answering `returncode`/`stdout`."""
    build = tmp_path / "build"
    build.mkdir()
    (build / (imf.HARNESS + (".exe" if os.name == "nt" else ""))).touch()
    monkeypatch.setattr(imf, "assemble", lambda *_args: None)
    monkeypatch.setattr(imf, "translate", lambda work: str(tmp_path / "spirv"))
    monkeypatch.setattr(imf.subprocess, "run",
                        lambda cmd, **_kw: subprocess.CompletedProcess(cmd, returncode, stdout, ""))
    return imf.run_case("R8G8B8A8_UNORM", 4, 4, random.Random(1), str(tmp_path), str(build), False, False)


def test_unsupported_device_skips_with_the_harness_reason(monkeypatch, tmp_path, capsys):
    # TEST-149: a 128-byte push constant device exits 77 with its own reason, which the skip line quotes.
    result = run_case_with_harness_result(
        monkeypatch, tmp_path, 77, "device: X\nunsupported: need 144 push constant bytes, device has 128\n")
    assert result is None
    assert "SKIP (unsupported: need 144 push constant bytes, device has 128)" in capsys.readouterr().out


def test_unsupported_without_a_reason_line_still_skips(monkeypatch, tmp_path, capsys):
    assert run_case_with_harness_result(monkeypatch, tmp_path, 77, "device: X\n") is None
    assert "SKIP (the device does not support the set)" in capsys.readouterr().out


def test_a_harness_error_is_a_failure_not_a_skip(monkeypatch, tmp_path, capsys):
    assert run_case_with_harness_result(monkeypatch, tmp_path, 1, "device: X\nvkCreateDevice failed\n") is False
    out = capsys.readouterr().out
    assert "FAIL (vkCreateDevice failed)" in out and "SKIP" not in out


def test_self_test_runs():
    r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "image_fuzz.py"), "--self-test"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "self-test: ok" in r.stdout
