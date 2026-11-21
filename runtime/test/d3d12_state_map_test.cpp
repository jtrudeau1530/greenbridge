// D3D12 resource state -> Vulkan layout/access/stage mapping and subresource decoding. Needs the Vulkan headers,
// not a GPU. On Windows the IR's D3D12 constants are also checked against the SDK's d3d12.h.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_replay.h"

#ifdef _WIN32
#include <windows.h>
#include <d3d12.h>

namespace s = leonida::d3d12_state;
static_assert(s::common == D3D12_RESOURCE_STATE_COMMON);
static_assert(s::vertex_and_constant_buffer == D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
static_assert(s::index_buffer == D3D12_RESOURCE_STATE_INDEX_BUFFER);
static_assert(s::render_target == D3D12_RESOURCE_STATE_RENDER_TARGET);
static_assert(s::unordered_access == D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
static_assert(s::depth_write == D3D12_RESOURCE_STATE_DEPTH_WRITE);
static_assert(s::depth_read == D3D12_RESOURCE_STATE_DEPTH_READ);
static_assert(s::non_pixel_shader_resource == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
static_assert(s::pixel_shader_resource == D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
static_assert(s::stream_out == D3D12_RESOURCE_STATE_STREAM_OUT);
static_assert(s::indirect_argument == D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
static_assert(s::copy_dest == D3D12_RESOURCE_STATE_COPY_DEST);
static_assert(s::copy_source == D3D12_RESOURCE_STATE_COPY_SOURCE);
static_assert(s::resolve_dest == D3D12_RESOURCE_STATE_RESOLVE_DEST);
static_assert(s::resolve_source == D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
static_assert(s::raytracing_acceleration_structure == D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
static_assert(s::shading_rate_source == D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE);
static_assert(s::generic_read == D3D12_RESOURCE_STATE_GENERIC_READ);
static_assert(s::present == D3D12_RESOURCE_STATE_PRESENT);
static_assert(leonida::all_subresources == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
static_assert(leonida::barrier_begin_only == D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY);
static_assert(leonida::barrier_end_only == D3D12_RESOURCE_BARRIER_FLAG_END_ONLY);
static_assert(leonida::clear_depth == D3D12_CLEAR_FLAG_DEPTH);
static_assert(leonida::clear_stencil == D3D12_CLEAR_FLAG_STENCIL);
#endif

#include <cstdio>

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

int main() {
    using namespace leonida;
    namespace s = d3d12_state;
    const auto layout = [](std::uint32_t states, bool presentable = false) {
        return vulkan_access(states, presentable).layout;
    };

    // Single states.
    CHECK(layout(s::common) == VK_IMAGE_LAYOUT_GENERAL);
    CHECK(layout(s::present, true) == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    CHECK(layout(s::render_target) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    CHECK(layout(s::render_target, true) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    CHECK(layout(s::unordered_access) == VK_IMAGE_LAYOUT_GENERAL);
    CHECK(layout(s::depth_write) == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    CHECK(layout(s::depth_read) == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    CHECK(layout(s::pixel_shader_resource) == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    CHECK(layout(s::non_pixel_shader_resource) == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    CHECK(layout(s::copy_dest) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    CHECK(layout(s::copy_source) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    CHECK(layout(s::resolve_dest) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    CHECK(layout(s::resolve_source) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Read-only combinations.
    CHECK(layout(s::pixel_shader_resource | s::non_pixel_shader_resource) == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    CHECK(layout(s::depth_read | s::pixel_shader_resource) == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    CHECK(layout(s::depth_read | s::pixel_shader_resource | s::non_pixel_shader_resource) ==
          VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    CHECK(layout(s::pixel_shader_resource | s::copy_source) == VK_IMAGE_LAYOUT_GENERAL);
    CHECK(layout(s::depth_read | s::pixel_shader_resource | s::copy_source) == VK_IMAGE_LAYOUT_GENERAL);
    CHECK(layout(s::generic_read) == VK_IMAGE_LAYOUT_GENERAL);

    // Accesses and stages.
    const auto rt = vulkan_access(s::render_target, false);
    CHECK(rt.access == (VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT));
    CHECK(rt.stages == VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    const auto srv = vulkan_access(s::pixel_shader_resource | s::non_pixel_shader_resource, false);
    CHECK(srv.access == VK_ACCESS_SHADER_READ_BIT);
    CHECK((srv.stages & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT) && (srv.stages & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT));
    const auto generic = vulkan_access(s::generic_read, false);
    CHECK(generic.access & VK_ACCESS_INDEX_READ_BIT);
    CHECK(generic.access & VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    CHECK(generic.access & VK_ACCESS_TRANSFER_READ_BIT);
    CHECK(!(generic.access & (VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)));
    const auto present = vulkan_access(s::present, true);
    CHECK(present.access == 0 && present.stages == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    const auto common = vulkan_access(s::common, false);
    CHECK(common.access == (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT));
    // An unknown bit falls back to GENERAL with every access.
    const auto unknown = vulkan_access(s::pixel_shader_resource | 0x80000000u, false);
    CHECK(unknown.layout == VK_IMAGE_LAYOUT_GENERAL);
    CHECK(unknown.access & VK_ACCESS_MEMORY_WRITE_BIT);
    CHECK(unknown.stages & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    // Subresource indices: mip + layer * mips + plane * mips * layers.
    VkImageSubresourceRange range{};
    ResourceBinding color{};
    color.image = reinterpret_cast<VkImage>(std::uintptr_t{1});
    color.mip_levels = 3;
    color.array_layers = 4;
    CHECK(image_range(color, all_subresources, range));
    CHECK(range.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT && range.baseMipLevel == 0 && range.levelCount == 3);
    CHECK(range.baseArrayLayer == 0 && range.layerCount == 4);
    CHECK(image_range(color, 7, range));  // mip 1, layer 2
    CHECK(range.baseMipLevel == 1 && range.levelCount == 1 && range.baseArrayLayer == 2 && range.layerCount == 1);
    CHECK(image_range(color, 11, range));  // last subresource
    CHECK(range.baseMipLevel == 2 && range.baseArrayLayer == 3);
    CHECK(!image_range(color, 12, range));  // no plane 1 on a colour image

    ResourceBinding depth_stencil = color;
    depth_stencil.aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    depth_stencil.mip_levels = 2;
    depth_stencil.array_layers = 1;
    CHECK(image_range(depth_stencil, 1, range));
    CHECK(range.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT && range.baseMipLevel == 1);
    CHECK(image_range(depth_stencil, 2, range));
    CHECK(range.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT && range.baseMipLevel == 0);
    CHECK(image_range(depth_stencil, 3, range));
    CHECK(range.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT && range.baseMipLevel == 1);
    CHECK(!image_range(depth_stencil, 4, range));
    CHECK(image_range(depth_stencil, all_subresources, range));
    CHECK(range.aspectMask == depth_stencil.aspects && range.levelCount == 2);

    ResourceBinding depth_only = depth_stencil;
    depth_only.aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    CHECK(image_range(depth_only, 1, range) && range.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT);
    CHECK(!image_range(depth_only, 2, range));

    ResourceBinding empty = color;
    empty.mip_levels = 0;
    CHECK(!image_range(empty, all_subresources, range));

    std::printf("state map and subresource decoding OK\n");
    return 0;
}
