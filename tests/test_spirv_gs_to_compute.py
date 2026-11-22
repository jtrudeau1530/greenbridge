"""tools/spirv_gs_to_compute.py: a translated geometry shader as a compute capture pass (SHADER-159)."""
import os
import re
import subprocess
import sys

import pytest

from conftest import HAVE_SPIRV, make_dxbc, requires_tools, signature_chunk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import spirv_gs_to_compute as gsc  # noqa: E402

# gs_tri's signature, as scripts/run_gpu_tests.py builds it: SV_Position in register 0, TEXCOORD0 in register 1.
GS_DXBC = make_dxbc([signature_chunk(b"ISG1", [(0, 1, 0xF), (1, 0, 0xF)]),
                     signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 0, 0xF)])])


@pytest.fixture
def gs_tri(make_shader):
    asm = open(os.path.join(ROOT, "runtime", "test", "shaders", "gs_tri.asm"), encoding="utf-8").read()
    shader = make_shader(asm, name="gs_tri", dxbc=GS_DXBC)
    spvasm = shader.translate()
    assert "OpEntryPoint Geometry" in spvasm
    return shader, spvasm


@requires_tools
def test_layout(gs_tri):
    _, spvasm = gs_tri
    _, lay = gsc.rewrite(spvasm)
    assert lay == {"verts_in": 3, "max_verts": 3, "topology": "triangle_strip",
                   "inputs": ["position", "location 1"], "outputs": ["position", "location 1"], "local_size": 64}


@requires_tools
def test_compute_module(gs_tri):
    shader, spvasm = gs_tri
    out, _ = gsc.rewrite(spvasm)
    assert "OpCapability Geometry" not in out
    assert re.search(r"OpEntryPoint GLCompute %\w+ \"main\"", out)
    assert re.search(r"OpExecutionMode %\w+ LocalSize 64 1 1", out)
    assert "OpEmitVertex" not in out and "OpEndPrimitive" not in out
    # no outputs are left, and the only inputs are GlobalInvocationId and the subgroup lane the GS already read
    assert not re.search(r"OpVariable %\w+ Output\b", out)
    inputs = re.findall(r"(%\w+) = OpVariable %\w+ Input\b", out)
    builtins = {v: re.search(rf"OpDecorate {v} BuiltIn (\w+)", out).group(1) for v in inputs}
    assert sorted(builtins.values()) == ["GlobalInvocationId", "SubgroupLocalInvocationId"]
    # three emits, two outputs each, go to the record buffer
    assert len(re.findall(r"OpAccessChain %gsc_ptr_sb_v4 %gsc_rec ", out)) == 6
    for b in (0, 1, 2):
        assert f"Binding {b}" in out
    shader.validate(out)    # spirv-as + spirv-val, as for the translator's own output


@requires_tools
def test_rejects_non_geometry(make_shader):
    shader = make_shader("v_mov_b32 v1, 1.0\ns_endpgm", name="cs_not_gs")
    with pytest.raises(gsc.Unsupported):
        gsc.rewrite(shader.translate())


@requires_tools
def test_cli_writes_module_and_layout(gs_tri, tmp_path):
    shader, spvasm = gs_tri
    sa, sv = tmp_path / "gs.spvasm", tmp_path / "gs.spv"
    sa.write_text(spvasm)
    subprocess.run(["spirv-as", str(sa), "-o", str(sv), "--target-env", "vulkan1.2"], check=True)
    out = tmp_path / "cs.spv"
    r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "spirv_gs_to_compute.py"), str(sv),
                        str(out), "--layout", str(tmp_path / "lay.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and (tmp_path / "lay.json").exists()
    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(out)], check=True)


# A hand-built GS whose entry block branches, with an OpPhi naming the entry label as a predecessor (the translator
# emits those for guarded loads: its block_label is %entry until the first new label). The primitive-count check
# splits the entry block, so the OpPhi must name the block that now ends the entry code (review of #1331).
PHI_GS = """\
OpCapability Shader
OpCapability Geometry
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %in_pos %out_pos
OpExecutionMode %main Triangles
OpExecutionMode %main Invocations 1
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 3
OpDecorate %in_pos BuiltIn Position
OpDecorate %out_pos BuiltIn Position
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%u0 = OpConstant %uint 0
%u1 = OpConstant %uint 1
%u3 = OpConstant %uint 3
%arr = OpTypeArray %v4float %u3
%ptr_in_arr = OpTypePointer Input %arr
%ptr_in_v4 = OpTypePointer Input %v4float
%ptr_out_v4 = OpTypePointer Output %v4float
%in_pos = OpVariable %ptr_in_arr Input
%out_pos = OpVariable %ptr_out_v4 Output
%main = OpFunction %void None %fn
%entry = OpLabel
%c = OpIEqual %bool %u0 %u1
OpSelectionMerge %m None
OpBranchConditional %c %t %m
%t = OpLabel
OpBranch %m
%m = OpLabel
%x = OpPhi %uint %u1 %t %u0 %entry
%p = OpAccessChain %ptr_in_v4 %in_pos %x
%v = OpLoad %v4float %p
OpStore %out_pos %v
OpEmitVertex
OpEmitVertex
OpEmitVertex
OpEndPrimitive
OpReturn
OpFunctionEnd
"""


@pytest.mark.skipif(not HAVE_SPIRV, reason="needs spirv-as and spirv-val on PATH")
def test_phi_from_entry_block_names_the_split_block(tmp_path):
    out, _ = gsc.rewrite(PHI_GS)
    assert "%x = OpPhi %uint %u1 %t %u0 %gsc_body" in out
    sa, sv = tmp_path / "cs.spvasm", tmp_path / "cs.spv"
    sa.write_text(out)
    subprocess.run(["spirv-as", str(sa), "-o", str(sv), "--target-env", "vulkan1.2"], check=True)
    r = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(sv)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_help_is_a_clean_exit_zero(capsys):
    assert gsc.main(["spirv_gs_to_compute.py", "--help"]) == 0
    assert "usage: spirv_gs_to_compute.py" in capsys.readouterr().out