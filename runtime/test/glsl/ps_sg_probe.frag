#version 460
// A pixel shader that outputs the subgroup size the driver ran it with, (0x40000000 | gl_SubgroupSize, 0x40000000 |
// gl_SubgroupInvocationID, ...) as float bit patterns (the or keeps them normal floats), for d3d12_subgroup_test.
#extension GL_KHR_shader_subgroup_basic : require

layout(location = 0) out vec4 target;

void main() {
    target = uintBitsToFloat(uvec4(0x40000000u | gl_SubgroupSize, 0x40000000u | gl_SubgroupInvocationID, 0x40000000u,
                                   0x40000000u));
}
