"""Regression tests for tools/rdna2spirv.py (RDNA2 -> SPIR-V).

Run from the repository root:  python -m pytest tests
Needs llvm-mc (AMDGPU target) and spirv-tools; the tests skip themselves when those are missing.
"""
import glob
import os
import re

import pytest

import rdna2spirv
from conftest import EMPTY_DXBC, KNOWN_VAL_ISSUES, ROOT, SAMPLE_DXBC, make_dxbc, mesh_psv0_chunk, requires_tools, signature_chunk

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
    name = stem if stem.startswith(("vs_", "ps_", "cs_", "gs_", "ms_")) else "ps_" + stem   # the prefix selects the stage
    if name.startswith(("gs_", "ms_")) and stem not in SAMPLE_DXBC:
        pytest.skip(f"no synthetic DXBC (ISG1/OSG1 for gs, PSV0 for ms) defined for {stem} in conftest.SAMPLE_DXBC")
    with open(path, encoding="utf-8") as f:
        shader = make_shader(f.read(), name, dxbc=SAMPLE_DXBC.get(stem, EMPTY_DXBC))
    shader.validate(shader.translate(), KNOWN_VAL_ISSUES.get(stem))


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


def test_geometry_without_signatures_is_rejected(make_shader):
    """A gs needs ISG1/OSG1 to know which registers LDS holds and which outputs to write."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "gs_tri.asm"), encoding="utf-8") as f:
        shader = make_shader(f.read(), "gs_tri")                   # empty DXBC
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


@pytest.mark.parametrize("kw", [dict(topology=1), dict(threads=(2048, 1, 1)), dict(max_vertices=0)],
                         ids=["not-triangles", "over-1024-threads", "no-vertices"])
def test_mesh_bad_psv0_is_rejected(make_shader, kw):
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "ms_tri.asm"), encoding="utf-8") as f:
        shader = make_shader(f.read(), "ms_tri", dxbc=make_dxbc([mesh_psv0_chunk(**kw)]))
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


def test_mesh_without_psv0_is_rejected(make_shader):
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "ms_tri.asm"), encoding="utf-8") as f:
        shader = make_shader(f.read(), "ms_tri")                   # empty DXBC
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


def test_wave64_bfm64_and_half_cmp_emit_wave_ops(make_shader):
    """Wave64 s_bfm_b64 must use wave_lane() and reconstruct the 64-bit mask; 32-bit half ops must guard their half."""
    bfm = make_shader("s_mov_b64 exec, -1\ns_bfm_b64 s[6:7], 16, 8\ns_bcnt1_i32_b64 s8, s[6:7]\ns_endpgm\n",
                      name="cs_bfm64_ci")
    asm_bfm = bfm.translate()
    bfm.validate(asm_bfm)
    assert "OpGroupNonUniformBitwiseOr" in asm_bfm
    assert "OpLoad %v3uint %local_id" in asm_bfm and "OpLoad %uint %sg_lane" not in asm_bfm

    hcmp = make_shader("s_mov_b32 exec_lo, 0\ns_mov_b32 exec_hi, 0xffff\ns_mov_b64 s[6:7], exec\n"
                       "s_cmp_eq_u32 s6, 0\ns_cmp_lg_u32 s7, 0\ns_endpgm\n", name="cs_halfcmp_ci")
    asm_hcmp = hcmp.translate()
    hcmp.validate(asm_hcmp)
    # Low-half s_cmp_eq_u32: lane < 32 (%c20) guards OpSelect(..., %true) feeding OpGroupNonUniformAll.
    assert re.search(
        r"(%t\d+) = OpULessThan %bool %t\d+ %c20\n\s*"
        r"(%t\d+) = OpSelect %bool \1 %t\d+ %true\n\s*"
        r"%t\d+ = OpGroupNonUniformAll %bool %c3 \2",
        asm_hcmp,
    ), "low-half s_cmp_eq_u32 did not guard OpGroupNonUniformAll with lane < 32"
    # High-half s_cmp_lg_u32: lane >= 32 (%c20) guards OpLogicalAnd feeding OpGroupNonUniformAny.
    assert re.search(
        r"(%t\d+) = OpUGreaterThanEqual %bool %t\d+ %c20\n\s*"
        r"(%t\d+) = OpLogicalAnd %bool \1 %t\d+\n\s*"
        r"%t\d+ = OpGroupNonUniformAny %bool %c3 \2",
        asm_hcmp,
    ), "high-half s_cmp_lg_u32 did not guard OpGroupNonUniformAny with lane >= 32"

    # Copying vcc_lo / vcc_hi to a 32-bit SGPR preserves half identity for wave64 s_bcnt1_i32_b32.
    vcc_half = make_shader(
        "s_mov_b64 exec, -1\nv_cmp_gt_u32_e32 vcc_lo, 16, v0\n"
        "s_mov_b32 s6, vcc_lo\ns_bcnt1_i32_b32 s8, s6\n"
        "s_mov_b32 s7, vcc_hi\ns_bcnt1_i32_b32 s9, s7\ns_endpgm\n",
        name="cs_vcchalf_ci",
    )
    asm_vcc = vcc_half.translate()
    vcc_half.validate(asm_vcc)
    assert asm_vcc.count("OpBitCount %uint %c0") == 2
