"""tools/spirv_metal_set0.py: set 0's descriptor arrays get fixed, non-overlapping Metal IDs (SHADER-161)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import spirv_metal_set0 as m  # noqa: E402

# The set-0 part of a translated pixel shader (spirv-dis, names kept): sampled 2D images, samplers, the hash table.
MODULE = """\
               OpDecorate %images DescriptorSet 0
               OpDecorate %images Binding 0
               OpDecorate %samplers DescriptorSet 0
               OpDecorate %samplers Binding 1
               OpDecorate %desc_map DescriptorSet 0
               OpDecorate %desc_map Binding 2
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
      %img2d = OpTypeImage %float 2D 0 0 0 1 Unknown
        %smp = OpTypeSampler
     %rt_img = OpTypeRuntimeArray %img2d
     %rt_smp = OpTypeRuntimeArray %smp
    %ptr_img = OpTypePointer UniformConstant %rt_img
    %ptr_smp = OpTypePointer UniformConstant %rt_smp
     %rt_map = OpTypeRuntimeArray %uint
        %map = OpTypeStruct %rt_map
    %ptr_map = OpTypePointer StorageBuffer %map
     %images = OpVariable %ptr_img UniformConstant
   %samplers = OpVariable %ptr_smp UniformConstant
   %desc_map = OpVariable %ptr_map StorageBuffer
"""


def test_capacities_match_vulkan_bindless():
    counts = m.capacities()
    assert counts[:5] == [4096, 1024, 1, 1024, 1024]
    assert len(counts) == 26


def test_metal_ids_are_running_sums():
    assert m.metal_ids([4096, 1024, 1, 1024]) == [0, 4096, 5120, 5121]


def test_rewrite_sizes_arrays_and_spaces_bindings():
    counts = m.capacities()
    out = m.rewrite(MODULE, counts)
    assert "OpDecorate %images Binding 0" in out
    assert "OpDecorate %samplers Binding 4096" in out
    assert "OpDecorate %desc_map Binding 5120" in out
    assert "%msl_cap_b0 = OpConstant %uint 4096" in out
    assert "%msl_arr_b0 = OpTypeArray %img2d %msl_cap_b0" in out
    assert "%images = OpVariable %msl_ptr_b0 UniformConstant" in out
    assert "%msl_arr_b1 = OpTypeArray %smp %msl_cap_b1" in out
    assert "%samplers = OpVariable %msl_ptr_b1 UniformConstant" in out
    # the hash table is a storage buffer, not a descriptor array: only its binding moves
    assert "%desc_map = OpVariable %ptr_map StorageBuffer" in out
    assert "msl_arr_b2" not in out
    # each new type comes after the pointer type it replaces, so its operands are already declared
    assert out.index("%ptr_img = OpTypePointer") < out.index("%msl_cap_b0")


def test_help_is_a_clean_exit_zero(capsys):
    for flag in ("-h", "--help"):
        assert m.main(["spirv_metal_set0.py", flag]) == 0
        assert "Usage: spirv_metal_set0.py" in capsys.readouterr().out
