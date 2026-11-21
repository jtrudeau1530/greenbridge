"""Regression tests for tools/rdna2spirv.py (RDNA2 -> SPIR-V).

Run from the repository root:  python -m pytest tests
Needs llvm-mc (AMDGPU target) and spirv-tools; the tests skip themselves when those are missing.
"""
import glob
import os

import pytest

import rdna2spirv
from conftest import ROOT, requires_tools

pytestmark = requires_tools

SAMPLES = sorted(glob.glob(os.path.join(ROOT, "runtime", "test", "shaders", "*.asm")))

# One minimal pixel shader per instruction family listed in the "Supported" section of rdna2spirv.py's docstring.
# Each body is wrapped in a program that exports its result so nothing is optimised away.
SUPPORTED = {
    "v_mov": "v_mov_b32 v0, 1.0",
    "v_add_f32": "v_mov_b32 v1, 1.0\nv_add_f32 v0, v1, v1",
    "v_sub_f32": "v_mov_b32 v1, 2.0\nv_sub_f32 v0, v1, 1.0",
    "v_mul_f32": "v_mov_b32 v1, 2.0\nv_mul_f32 v0, v1, v1",
    "v_fma_f32": "v_mov_b32 v1, 2.0\nv_fma_f32 v0, v1, v1, v1",
    "v_max_f32": "v_mov_b32 v1, 2.0\nv_max_f32 v0, v1, 1.0",
    "v_min_f32": "v_mov_b32 v1, 2.0\nv_min_f32 v0, v1, 1.0",
    "v_rcp_f32": "v_mov_b32 v1, 4.0\nv_rcp_f32 v0, v1",
    "v_sqrt_f32": "v_mov_b32 v1, 4.0\nv_sqrt_f32 v0, v1",
    "v_floor_f32": "v_mov_b32 v1, 1.5\nv_floor_f32 v0, v1",
    "v_fract_f32": "v_mov_b32 v1, 1.5\nv_fract_f32 v0, v1",
    "v_cvt_f32_u32": "v_mov_b32 v1, 7\nv_cvt_f32_u32 v0, v1",
    "v_cvt_u32_f32": "v_mov_b32 v1, 7.0\nv_cvt_u32_f32 v0, v1",
    "v_add_nc_u32": "v_mov_b32 v1, 7\nv_add_nc_u32 v0, v1, v1",
    "v_and_b32": "v_mov_b32 v1, 7\nv_and_b32 v0, 3, v1",
    "v_lshlrev_b32": "v_mov_b32 v1, 7\nv_lshlrev_b32 v0, 2, v1",
    "v_lshrrev_b32": "v_mov_b32 v1, 7\nv_lshrrev_b32 v0, 1, v1",
    "v_bfe_u32": "v_mov_b32 v1, 0xf0\nv_bfe_u32 v0, v1, 4, 4",
    "v_med3_f32": "v_mov_b32 v1, 2.0\nv_med3_f32 v0, v1, 1.0, 4.0",
    "s_mov_b32": "s_mov_b32 s40, 5\nv_mov_b32 v0, s40",
    "s_add_u32": "s_mov_b32 s40, 5\ns_add_u32 s41, s40, 3\nv_mov_b32 v0, s41",
}


def program(body):
    return f"{body}\nv_mov_b32 v8, v0\nexp mrt0 v8, v8, v8, v8 done vm\ns_endpgm\n"


@pytest.mark.parametrize("path", SAMPLES, ids=[os.path.basename(p) for p in SAMPLES])
def test_gpu_samples_translate_and_validate(path, make_shader):
    """The hand-written GPU-harness samples must translate to SPIR-V that passes spirv-val."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith(("gs_", "ms_")):
        pytest.skip("needs a DXBC with ISG1/OSG1 (gs) or PSV0 (ms) chunks; the synthetic empty DXBC has none")
    name = stem if stem.startswith(("vs_", "ps_", "cs_")) else "ps_" + stem     # the prefix selects the stage
    with open(path, encoding="utf-8") as f:
        shader = make_shader(f.read(), name)
    shader.validate(shader.translate())


@pytest.mark.parametrize("case", sorted(SUPPORTED))
def test_supported_instruction(case, make_shader):
    shader = make_shader(program(SUPPORTED[case]))
    shader.validate(shader.translate())


def test_translation_is_deterministic(make_shader):
    shader = make_shader(program(SUPPORTED["v_fma_f32"]))
    assert shader.translate() == shader.translate()


def test_unsupported_instruction_is_reported_not_guessed(make_shader):
    """Anything outside the supported set must raise Unsupported rather than emit a wrong program."""
    shader = make_shader(program("v_mov_b32 v1, 1.0\nv_sad_u8 v0, v1, v1, v1"))
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


def test_unknown_stage_is_rejected(make_shader):
    shader = make_shader(program("v_mov_b32 v0, 1.0"), name="xx_test")
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


def test_compute_translates_and_validates(make_shader):
    shader = make_shader("v_mov_b32 v1, 1\ns_endpgm\n", name="cs_test")
    shader.validate(shader.translate())


def test_compute_thread_group_size_sgpr_is_rejected(make_shader):
    """TG_SIZE_EN (RSRC2 bit 10) adds an SGPR the translator doesn't model."""
    shader = make_shader("s_endpgm\n", name="cs_test", cs_header=(0x8088 | 1 << 10, (64, 1, 1)))
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


@pytest.mark.parametrize("group", [(0, 1, 1), (2048, 1, 1)], ids=["empty", "over-1024"])
def test_compute_bad_group_size_is_rejected(make_shader, group):
    shader = make_shader("s_endpgm\n", name="cs_test", cs_header=(0x8088, group))
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()
