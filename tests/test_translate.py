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


# ---- pos1..pos3: ClipDistance / CullDistance / misc exports (VS & MS) ----

def _vgpr_to_index_map(spvasm, array_var, ptr_type):
    """Trace each OpStore to `array_var[idx]` back to the VGPR index loaded from `%vgpr`."""
    import re
    vgpr_of = {}
    val_of = {}
    ptr_idx = {}
    stores = {}
    for line in spvasm.splitlines():
        m = re.match(r"\s*(%\w+) = OpAccessChain %_ptr_Function_uint %vgpr %c([0-9a-f]+)", line)
        if m:
            vgpr_of[m.group(1)] = int(m.group(2), 16)
            continue
        m = re.match(r"\s*(%\w+) = OpLoad %uint (%\w+)", line)
        if m and m.group(2) in vgpr_of:
            val_of[m.group(1)] = vgpr_of[m.group(2)]
            continue
        m = re.match(r"\s*(%\w+) = OpBitcast %float (%\w+)", line)
        if m and m.group(2) in val_of:
            val_of[m.group(1)] = val_of[m.group(2)]
            continue
        m = re.match(rf"\s*(%\w+) = OpAccessChain {re.escape(ptr_type)} {re.escape(array_var)} %c(\d+)", line)
        if m:
            ptr_idx[m.group(1)] = int(m.group(2))
            continue
        m = re.match(r"\s*(%\w+) = OpSelect %float %\w+ (%\w+) %\w+", line)
        if m and m.group(2) in val_of:
            val_of[m.group(1)] = val_of[m.group(2)]
            continue
        m = re.match(r"\s*OpStore (%\w+) (%\w+)", line)
        if m and m.group(1) in ptr_idx and m.group(2) in val_of:
            stores[ptr_idx[m.group(1)]] = val_of[m.group(2)]
    return stores


def _ms_header_asm(pos_exports):
    return (
        "v_mbcnt_lo_u32_b32 v0, -1, 0\n"
        "v_mbcnt_hi_u32_b32 v0, -1, v0\n"
        "s_mov_b32 m0, 0x1003\n"
        "s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)\n"
        "v_mov_b32 v1, 0.0\n"
        "v_mov_b32 v2, 1.0\n"
        "v_mov_b32 v10, 0.25\n"
        "v_mov_b32 v11, 0.5\n"
        "v_mov_b32 v12, 1.0\n"
        "v_mov_b32 v13, 2.0\n"
        "v_mov_b32 v14, 4.0\n"
        "exp pos0 v1, v1, v1, v2\n"
        f"{pos_exports}\n"
        "v_mov_b32 v9, 0x200400\n"
        "exp prim v9, off, off, off done\n"
        "s_endpgm\n"
    )


def _vs_header_asm(pos_exports):
    return (
        "v_mov_b32 v1, 0.0\n"
        "v_mov_b32 v2, 1.0\n"
        "v_mov_b32 v10, 0.25\n"
        "v_mov_b32 v11, 0.5\n"
        "v_mov_b32 v12, 1.0\n"
        "v_mov_b32 v13, 2.0\n"
        "v_mov_b32 v14, 4.0\n"
        "exp pos0 v1, v1, v1, v2\n"
        f"{pos_exports}\n"
        "s_endpgm\n"
    )


@pytest.mark.parametrize("n_comp,exp_line,mask", [
    (1, "exp pos1 v10, off, off, off done", 0x1),
    (4, "exp pos1 v10, v11, v12, v13 done", 0xF),
], ids=["1-comp", "4-comp"])
def test_vs_clip_distance_pos1(make_shader, n_comp, exp_line, mask):
    import re
    dxbc = make_dxbc([signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, mask)])])
    shader = make_shader(_vs_header_asm(exp_line), name="vs_clip_test", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert "OpCapability ClipDistance" in spvasm
    assert "OpCapability CullDistance" not in spvasm
    assert "OpDecorate %out_clip BuiltIn ClipDistance" in spvasm
    assert f"%arr_clip = OpTypeArray %float %c{n_comp}" in spvasm
    chains = re.findall(r"OpAccessChain %_ptr_Output_float %out_clip %c(\d+)", spvasm)
    assert [int(x) for x in chains] == list(range(n_comp))
    assert _vgpr_to_index_map(spvasm, "%out_clip", "%_ptr_Output_float") == {i: 10 + i for i in range(n_comp)}
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    assert "shaderClipDistance" in meta["features"]
    assert "shaderCullDistance" not in meta["features"]
    assert 1 not in meta.get("outputs", [])


@pytest.mark.parametrize("n_comp,exp_line,mask", [
    (1, "exp pos1 v10, off, off, off done", 0x1),
    (4, "exp pos1 v10, v11, v12, v13 done", 0xF),
], ids=["1-comp", "4-comp"])
def test_ms_clip_distance_pos1(make_shader, n_comp, exp_line, mask):
    import re
    dxbc = make_dxbc([mesh_psv0_chunk(), signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, mask)])])
    shader = make_shader(_ms_header_asm(exp_line), name="ms_clip_test", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm, KNOWN_VAL_ISSUES.get("ms_tri"))
    assert "OpCapability ClipDistance" in spvasm
    assert "OpDecorate %ms_vtx_blk Block" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 0 BuiltIn Position" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 1 BuiltIn ClipDistance" in spvasm
    assert f"%ms_arr_clip = OpTypeArray %float %c{n_comp}" in spvasm
    chains = re.findall(r"OpAccessChain %_ptr_Function_float %ms_clip %c(\d+)", spvasm)
    assert [int(x) for x in chains] == list(range(n_comp))
    assert _vgpr_to_index_map(spvasm, "%ms_clip", "%_ptr_Function_float") == {i: 10 + i for i in range(n_comp)}
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    assert "shaderClipDistance" in meta["features"]
    assert 1 not in meta.get("outputs", [])
    assert 1 not in meta.get("params", [])


def test_vs_clip_and_cull_distances_across_pos1_pos2(make_shader):
    """2 clip distances + 3 cull distances pack contiguously: pos1.xy=clip[0..1], pos1.zw=cull[0..1], pos2.x=cull[2]."""
    import re
    dxbc = make_dxbc([signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, 0x3), (2, 3, 0x7)])])
    asm = _vs_header_asm("exp pos1 v10, v11, v12, v13\nexp pos2 v14, off, off, off done")
    shader = make_shader(asm, name="vs_clip_cull", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert "OpCapability ClipDistance" in spvasm
    assert "OpCapability CullDistance" in spvasm
    assert "%arr_clip = OpTypeArray %float %c2" in spvasm
    assert "%arr_cull = OpTypeArray %float %c3" in spvasm
    clip_chains = re.findall(r"OpAccessChain %_ptr_Output_float %out_clip %c(\d+)", spvasm)
    cull_chains = re.findall(r"OpAccessChain %_ptr_Output_float %out_cull %c(\d+)", spvasm)
    assert [int(x) for x in clip_chains] == [0, 1]
    assert [int(x) for x in cull_chains] == [0, 1, 2]
    assert _vgpr_to_index_map(spvasm, "%out_clip", "%_ptr_Output_float") == {0: 10, 1: 11}
    assert _vgpr_to_index_map(spvasm, "%out_cull", "%_ptr_Output_float") == {0: 12, 1: 13, 2: 14}
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    assert "shaderClipDistance" in meta["features"]
    assert "shaderCullDistance" in meta["features"]
    assert not ({1, 2} & set(meta.get("outputs", [])))
    assert not ({1, 2} & set(meta.get("params", [])))


def test_vs_misc_pos1_shifts_clip_to_pos2(make_shader):
    """When OSG1 declares SV_RenderTargetArrayIndex (sysv=4), pos1 is the misc vector and ClipDistance starts at pos2."""
    import re
    dxbc = make_dxbc([signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 4, 0x1), (2, 2, 0x3)])])
    asm = _vs_header_asm("exp pos1 off, off, v14, off\nexp pos2 v10, v11, off, off done")
    shader = make_shader(asm, name="vs_misc_clip", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert "OpCapability ShaderLayer" in spvasm
    assert "OpCapability ClipDistance" in spvasm
    assert "OpDecorate %out_layer BuiltIn Layer" in spvasm
    assert "%arr_clip = OpTypeArray %float %c2" in spvasm
    clip_chains = re.findall(r"OpAccessChain %_ptr_Output_float %out_clip %c(\d+)", spvasm)
    assert [int(x) for x in clip_chains] == [0, 1]


def test_pos1_without_osg1_is_ignored(make_shader):
    """Without OSG1 declaring SV_ClipDistance/SV_CullDistance, pos1 exports are discarded in both VS and MS."""
    vs = make_shader(_vs_header_asm("exp pos1 v10, off, off, off done"), name="vs_no_osg1_pos1")
    vs_spv = vs.translate()
    vs.validate(vs_spv)
    assert "OpCapability ClipDistance" not in vs_spv
    assert "BuiltIn ClipDistance" not in vs_spv
    vs_meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, vs_spv)
    assert "shaderClipDistance" not in vs_meta["features"]
    assert 1 not in vs_meta.get("outputs", [])

    ms = make_shader(_ms_header_asm("exp pos1 v10, off, off, off done"), name="ms_no_osg1_pos1",
                     dxbc=make_dxbc([mesh_psv0_chunk()]))
    ms_spv = ms.translate()
    ms.validate(ms_spv, KNOWN_VAL_ISSUES.get("ms_tri"))
    assert "OpCapability ClipDistance" not in ms_spv
    assert "BuiltIn ClipDistance" not in ms_spv
    assert "%ms_out_pos" in ms_spv and "%ms_vtx_blk" not in ms_spv
    ms_meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, ms_spv)
    assert "shaderClipDistance" not in ms_meta["features"]
    assert 1 not in ms_meta.get("params", [])


def test_ms_dummy_all_off_pos1_does_not_emit_clip(make_shader):
    """An all-off dummy pos1 export in a mesh shader must not emit ClipDistance or change the output block."""
    dxbc = make_dxbc([mesh_psv0_chunk(), signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, 0x1)])])
    shader = make_shader(_ms_header_asm("exp pos1 off, off, off, off done"), name="ms_dummy_pos1", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm, KNOWN_VAL_ISSUES.get("ms_tri"))
    assert "OpCapability ClipDistance" not in spvasm
    assert "%ms_out_pos" in spvasm
    assert "%ms_vtx_blk" not in spvasm


def test_ms_clip_and_cull_distances_across_pos1_pos2(make_shader):
    """Mesh shader with 2 clip + 3 cull distances emits both arrays in %ms_vtx_blk and maps v10..v14."""
    dxbc = make_dxbc([mesh_psv0_chunk(), signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, 0x3), (2, 3, 0x7)])])
    asm = _ms_header_asm("exp pos1 v10, v11, v12, v13\nexp pos2 v14, off, off, off done")
    shader = make_shader(asm, name="ms_clip_cull", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm, KNOWN_VAL_ISSUES.get("ms_tri"))
    assert "OpCapability ClipDistance" in spvasm
    assert "OpCapability CullDistance" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 0 BuiltIn Position" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 1 BuiltIn ClipDistance" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 2 BuiltIn CullDistance" in spvasm
    assert "%ms_arr_clip = OpTypeArray %float %c2" in spvasm
    assert "%ms_arr_cull_d = OpTypeArray %float %c3" in spvasm
    assert _vgpr_to_index_map(spvasm, "%ms_clip", "%_ptr_Function_float") == {0: 10, 1: 11}
    assert _vgpr_to_index_map(spvasm, "%ms_cull_dist", "%_ptr_Function_float") == {0: 12, 1: 13, 2: 14}
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    assert "shaderClipDistance" in meta["features"]
    assert "shaderCullDistance" in meta["features"]
    assert not ({1, 2} & set(meta.get("outputs", [])))
    assert not ({1, 2} & set(meta.get("params", [])))


def test_wave64_detected_from_mask_ops_without_exec_markers(make_shader):
    """#210 part 2 / SHADER-28: s_bcnt1_i32_b64 on ordinary SGPRs marks the shader wave64 (no exec_hi / saveexec)."""
    shader = make_shader(
        "; mattr: +wavefrontsize64,-wavefrontsize32\n"
        "v_and_b32_e32 v5, 5, v0\nv_cmp_ne_u32_e64 s[20:21], 0, v5\n"
        "s_bcnt1_i32_b64 s30, s[20:21]\nv_mov_b32 v1, s30\ns_endpgm\n",
        name="cs_mask_detect_ci",
    )
    asm = shader.translate()
    shader.validate(asm)
    assert rdna2spirv.translate.last_gen.wave64


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


def test_wave64_bpermute_reads_zero_from_lanes_that_are_not_running(make_shader):
    """Native wave64 ds_bpermute: a source lane missing from a partial wave must read 0, not an undefined shuffle."""
    shader = make_shader("s_mov_b64 exec, -1\nv_lshlrev_b32_e32 v1, 2, v0\nds_bpermute_b32 v2, v1, v0\n"
                         "s_waitcnt lgkmcnt(0)\ns_endpgm\n", name="cs_bperm64_ci")
    asm = shader.translate()
    shader.validate(asm)
    assert rdna2spirv.translate.last_gen.wave64
    assert re.search(
        r"(%t\d+) = OpGroupNonUniformBallot %v4uint_bal %c3 %true\n\s*"
        r"(%t\d+) = OpGroupNonUniformBallotBitExtract %bool %c3 \1 (%t\d+)\n\s*"
        r"(%t\d+) = OpGroupNonUniformShuffle %uint %c3 %t\d+ \3\n\s*"
        r"%t\d+ = OpSelect %uint \2 \4 %c0\n",
        asm,
    ), "wave64 ds_bpermute did not keep the shuffle only where the source lane is running"


EMULATE_SAMPLES = [p for p in SAMPLES if os.path.basename(p).startswith(("cs_", "ms_"))]


@pytest.mark.parametrize("path", EMULATE_SAMPLES, ids=[os.path.basename(p) for p in EMULATE_SAMPLES])
def test_wave64_emulate_uses_no_subgroup_ops(path, make_shader, monkeypatch):
    """--wave64-emulate: a wave64 compute or mesh shader validates and uses no subgroup operation (everything
    cross-lane goes through %x_buf); a wave32 one comes out exactly as without the flag."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith("ms_") and stem not in SAMPLE_DXBC:
        pytest.skip(f"no synthetic PSV0 for {stem} in conftest.SAMPLE_DXBC")
    with open(path, encoding="utf-8") as f:
        shader = make_shader(f.read(), stem, dxbc=SAMPLE_DXBC.get(stem, EMPTY_DXBC))
    plain = shader.translate()
    monkeypatch.setattr(rdna2spirv, "WAVE64_EMULATE", True)
    emulated = shader.translate()
    if not rdna2spirv.translate.last_gen.wave64:
        assert emulated == plain
        return
    shader.validate(emulated, KNOWN_VAL_ISSUES.get(stem))
    assert "OpGroupNonUniform" not in emulated and "%sg_lane" not in emulated
    assert "%x_buf" in emulated and emulated.count("OpControlBarrier") >= 2
