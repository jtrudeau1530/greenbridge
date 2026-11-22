#pragma once

// Replays a closed CommandList (command_list.h) into a Vulkan command buffer.
//
// Layouts. With a LayoutTracker, every image's current Vulkan layout is known per subresource, and barriers and
// clears start from it rather than from what the application claims. That is how D3D12's implicit state promotion
// and decay are honoured: a texture in `common` sits in GENERAL, which every promotable access (shader read, copy)
// can use directly, so a later barrier whose `before` names the promoted state still starts from GENERAL; decay
// back to `common` changes no layout. Buffers have no layouts, and simultaneous-access textures stay in GENERAL
// for their whole life, so their decay is free too. Without a tracker, `before` is trusted as the D3D12 runtime
// trusts it.
//
// Depth/stencil planes. With separate_depth_stencil_layouts (Vulkan 1.2 separateDepthStencilLayouts, enabled on
// the device), each plane has its own layout, so a transition or clear of one plane leaves the other alone. Without
// it, both planes always move together.
//
// Requirements on the device and images:
// - a queue with graphics support (the stage masks use ALL_GRAPHICS);
// - images cleared without rectangles have TRANSFER_DST usage; images cleared with rectangles have attachment usage
//   and image_view must return views for them (dynamicRendering, Vulkan 1.3);
// - enhanced barriers need pipeline_barrier2 (synchronization2, Vulkan 1.3).
// A list that needs a missing function fails with ReplayResult::unsupported before anything is written.

#include "command_list.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leonida {

// What the resolver knows about a resource. image == VK_NULL_HANDLE means a buffer in the shared GPU address
// space: its transitions become global memory barriers.
struct ResourceBinding {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    std::uint32_t mip_levels = 1;
    std::uint32_t array_layers = 1;
    bool presentable = false;          // a swapchain image: common/present maps to PRESENT_SRC_KHR
    bool simultaneous_access = false;  // D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS: always GENERAL
    bool not_zeroed = false;           // D3D12_HEAP_FLAG_CREATE_NOT_ZEROED: first use skips the zero fill
    // A descriptor in the bindless set 0 has one layout, but D3D12 lets a texture be sampled both in a
    // shader-resource state and while promoted from common (GENERAL). With this set, the shader-resource states
    // use GENERAL too, so the descriptor's layout is right in both.
    bool sampled_in_general = false;
    VkBuffer buffer = VK_NULL_HANDLE;  // a buffer resource, for copies
    VkDeviceSize size = 0;             // its size in bytes
    std::uint32_t depth = 1;
    bool volume = false;               // one array layer, including at mips with a single slice
    std::uint32_t samples = 1;         // more than one: a multisampled 2D texture
};

struct VulkanAccess {
    VkImageLayout layout;
    VkAccessFlags access;
    VkPipelineStageFlags stages;
};

inline constexpr VkPipelineStageFlags shader_stages =
    VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
inline constexpr VkImageAspectFlags depth_stencil_aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;

// Maps a legacy D3D12_RESOURCE_STATES combination to the Vulkan layout, accesses and stages that cover it.
// Read-only combinations share one layout when every bit agrees on it (depth_read with shader reads stays in
// DEPTH_STENCIL_READ_ONLY_OPTIMAL, which can be sampled); otherwise they fall back to GENERAL.
// Buffer-only and extension states map to GENERAL with conservative memory access.
inline VulkanAccess vulkan_access(std::uint32_t states, bool presentable) {
    namespace s = d3d12_state;
    constexpr VkAccessFlags any_memory = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    if (states == s::common) {
        if (presentable) return {VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, 0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
        return {VK_IMAGE_LAYOUT_GENERAL, any_memory, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    }

    struct Bit {
        std::uint32_t state;
        VkImageLayout layout;
        VkAccessFlags access;
        VkPipelineStageFlags stages;
    };
    constexpr VkPipelineStageFlags fragment_tests =
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    static constexpr Bit bits[] = {
        {s::vertex_and_constant_buffer, VK_IMAGE_LAYOUT_GENERAL,
         VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT,
         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | shader_stages},
        {s::index_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_INDEX_READ_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT},
        {s::render_target, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
        {s::unordered_access, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
         shader_stages},
        {s::depth_write, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, fragment_tests},
        {s::depth_read, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
         fragment_tests},
        {s::non_pixel_shader_resource, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
         shader_stages},
        {s::pixel_shader_resource, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT},
        {s::stream_out, VK_IMAGE_LAYOUT_GENERAL, any_memory, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT},
        {s::indirect_argument, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
         VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT},
        {s::copy_dest, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT},
        {s::copy_source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT},
        {s::resolve_dest, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT},
        {s::resolve_source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT},
        {s::raytracing_acceleration_structure, VK_IMAGE_LAYOUT_GENERAL, any_memory,
         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT},
        {s::shading_rate_source, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_MEMORY_READ_BIT,
         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT},
    };

    const auto merge = [](VkImageLayout a, VkImageLayout b) {
        if (a == VK_IMAGE_LAYOUT_UNDEFINED || a == b) return b;
        const auto pair = [&](VkImageLayout x, VkImageLayout y) { return (a == x && b == y) || (a == y && b == x); };
        if (pair(VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        return VK_IMAGE_LAYOUT_GENERAL;
    };

    VulkanAccess out{VK_IMAGE_LAYOUT_UNDEFINED, 0, 0};
    std::uint32_t known = 0;
    for (const auto& bit : bits) {
        if (!(states & bit.state)) continue;
        known |= bit.state;
        out.layout = merge(out.layout, bit.layout);
        out.access |= bit.access;
        out.stages |= bit.stages;
    }
    if (states & ~known) {
        out.layout = VK_IMAGE_LAYOUT_GENERAL;
        out.access |= any_memory;
        out.stages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    return out;
}

// Enhanced barriers: D3D12_BARRIER_SYNC -> VkPipelineStageFlags2. split (the end half of a split barrier) waits
// for everything, since the begin half's scope is not kept.
inline VkPipelineStageFlags2 vulkan_sync(std::uint32_t sync) {
    namespace y = d3d12_sync;
    if (sync == y::none) return VK_PIPELINE_STAGE_2_NONE;
    constexpr VkPipelineStageFlags2 pre_raster = VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT;
    struct Bit {
        std::uint32_t sync;
        VkPipelineStageFlags2 stages;
    };
    static constexpr Bit bits[] = {
        {y::all, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
        {y::draw, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT},
        {y::index_input, VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT},
        {y::vertex_shading, pre_raster | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT},
        {y::pixel_shading, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT},
        {y::depth_stencil,
         VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT},
        {y::render_target, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
        {y::compute_shading, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
        {y::copy, VK_PIPELINE_STAGE_2_COPY_BIT},
        {y::resolve, VK_PIPELINE_STAGE_2_RESOLVE_BIT},
        {y::execute_indirect, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT},
        {y::all_shading, pre_raster | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
        {y::non_pixel_shading, pre_raster | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
        {y::clear_unordered_access_view, VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
    };
    VkPipelineStageFlags2 out = 0;
    std::uint32_t known = 0;
    for (const auto& bit : bits) {
        if (!(sync & bit.sync)) continue;
        known |= bit.sync;
        out |= bit.stages;
    }
    // Ray tracing, video and split have no narrower equivalent without extensions.
    if (sync & ~known) out |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    return out;
}

// D3D12_BARRIER_ACCESS -> VkAccessFlags2. common means "any access the layout allows".
inline VkAccessFlags2 vulkan_barrier_access(std::uint32_t access) {
    namespace a = d3d12_access;
    constexpr VkAccessFlags2 any_memory = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    if (access & a::no_access) return VK_ACCESS_2_NONE;
    if (access == a::common) return any_memory;
    struct Bit {
        std::uint32_t access;
        VkAccessFlags2 flags;
    };
    static constexpr Bit bits[] = {
        {a::vertex_buffer, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT},
        {a::constant_buffer, VK_ACCESS_2_UNIFORM_READ_BIT},
        {a::index_buffer, VK_ACCESS_2_INDEX_READ_BIT},
        {a::render_target, VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT},
        {a::unordered_access, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT},
        {a::depth_stencil_write,
         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT},
        {a::depth_stencil_read, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT},
        {a::shader_resource, VK_ACCESS_2_SHADER_READ_BIT},
        {a::stream_output, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {a::indirect_argument, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT},
        {a::copy_dest, VK_ACCESS_2_TRANSFER_WRITE_BIT},
        {a::copy_source, VK_ACCESS_2_TRANSFER_READ_BIT},
        {a::resolve_dest, VK_ACCESS_2_TRANSFER_WRITE_BIT},
        {a::resolve_source, VK_ACCESS_2_TRANSFER_READ_BIT},
        {a::raytracing_acceleration_structure_read, VK_ACCESS_2_MEMORY_READ_BIT},
        {a::raytracing_acceleration_structure_write, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {a::shading_rate_source, VK_ACCESS_2_MEMORY_READ_BIT},
    };
    VkAccessFlags2 out = 0;
    std::uint32_t known = 0;
    for (const auto& bit : bits) {
        if (!(access & bit.access)) continue;
        known |= bit.access;
        out |= bit.flags;
    }
    if (access & ~known) out |= any_memory;  // video and unknown bits
    return out;
}

// For one VkAccessFlagBits2 that vulkan_barrier_access produces: the stages Vulkan lets perform it (the "supported
// access types" table of the synchronization2 chapter, with the meta stages that contain them), and the narrowest
// stage to add when a barrier's stages have none of those. MEMORY_READ/WRITE are valid in every stage.
struct AccessStages {
    VkAccessFlags2 access;
    VkPipelineStageFlags2 supported;
    VkPipelineStageFlags2 narrowest;
};

inline constexpr VkPipelineStageFlags2 shader_stages2 =
    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |
    VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |
    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
    VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

inline constexpr AccessStages access_stages[] = {
    {VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
     VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT},
    {VK_ACCESS_2_INDEX_READ_BIT,
     VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT},
    {VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
     VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
         VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT},
    // Shader accesses: D3D12 does not say which shader stage, so every one that can run (all shading).
    {VK_ACCESS_2_UNIFORM_READ_BIT, shader_stages2,
     VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
    {VK_ACCESS_2_SHADER_READ_BIT, shader_stages2,
     VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
    {VK_ACCESS_2_SHADER_WRITE_BIT, shader_stages2,
     VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
    {VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
    {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
    {VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
         VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT},
    {VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
         VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT},
    // Transfers: ALL_GRAPHICS does not contain the transfer stages, and CLEAR only writes.
    {VK_ACCESS_2_TRANSFER_READ_BIT,
     VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT |
         VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT},
    {VK_ACCESS_2_TRANSFER_WRITE_BIT,
     VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT |
         VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT},
};

// One side of an enhanced barrier: D3D12 sync and access to a Vulkan stage mask and access mask that Vulkan accepts
// together. D3D12 lets an access be named with a sync scope whose Vulkan stages cannot perform it (SYNC_ALL_SHADING
// with ACCESS_VERTEX_BUFFER is valid D3D12, but VERTEX_ATTRIBUTE_READ needs VERTEX_ATTRIBUTE_INPUT). The stages are
// widened by the narrowest stage that can, never the access dropped: the dependency still covers every access the
// application named, only over a slightly larger scope. With no stages (SYNC_NONE) there is no access either.
struct BarrierScope {
    VkPipelineStageFlags2 stages;
    VkAccessFlags2 access;
};

inline BarrierScope vulkan_barrier_scope(std::uint32_t sync, std::uint32_t access) {
    BarrierScope out{vulkan_sync(sync), 0};
    if (!out.stages) return out;
    out.access = vulkan_barrier_access(access);
    for (const AccessStages& entry : access_stages)
        if ((out.access & entry.access) && !(out.stages & entry.supported)) out.stages |= entry.narrowest;
    return out;
}

// D3D12_BARRIER_LAYOUT -> VkImageLayout. Queue-specific layouts map like their generic forms.
inline VkImageLayout vulkan_layout(std::uint32_t layout, bool presentable) {
    namespace l = d3d12_layout;
    switch (layout) {
    case l::undefined: return VK_IMAGE_LAYOUT_UNDEFINED;
    case l::common: return presentable ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_GENERAL;
    case l::render_target: return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case l::depth_stencil_write: return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    case l::depth_stencil_read: return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    case l::shader_resource:
    case l::direct_queue_shader_resource:
    case l::compute_queue_shader_resource: return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case l::copy_source:
    case l::resolve_source:
    case l::direct_queue_copy_source:
    case l::compute_queue_copy_source: return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case l::copy_dest:
    case l::resolve_dest:
    case l::direct_queue_copy_dest:
    case l::compute_queue_copy_dest: return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    default: return VK_IMAGE_LAYOUT_GENERAL;  // generic read, UAV, shading rate, video, *_common
    }
}

// The layout an image actually uses: simultaneous-access images live in GENERAL, and so do images in a
// shader-resource state when set 0 names them (sampled_in_general).
inline VkImageLayout image_layout(const ResourceBinding& resource, VkImageLayout layout) {
    if (resource.simultaneous_access && layout != VK_IMAGE_LAYOUT_UNDEFINED &&
        layout != VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        return VK_IMAGE_LAYOUT_GENERAL;
    if (resource.sampled_in_general && layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        return VK_IMAGE_LAYOUT_GENERAL;
    return layout;
}

// Bytes per texel, or per 4x4 block for block-compressed formats, of one aspect as copies lay it out in a buffer
// (D3D12 footprints and Vulkan buffer-image copies agree on these). bytes == 0: not a format copies support yet.
struct BlockFormat {
    std::uint32_t bytes = 0;
    std::uint32_t block = 1;  // texels per block edge
};

inline BlockFormat block_format(VkFormat format, VkImageAspectFlags aspect) {
    if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) return {1, 1};
    switch (format) {
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SINT: return {16, 1};
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R32G32_SFLOAT: return {8, 1};
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:  // depth aspect
    case VK_FORMAT_D24_UNORM_S8_UINT: return {4, 1};
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_D16_UNORM: return {2, 1};
    case VK_FORMAT_R8_UNORM: return {1, 1};
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC4_SNORM_BLOCK: return {8, 4};
    case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC5_UNORM_BLOCK:
    case VK_FORMAT_BC5_SNORM_BLOCK:
    case VK_FORMAT_BC6H_UFLOAT_BLOCK:
    case VK_FORMAT_BC6H_SFLOAT_BLOCK:
    case VK_FORMAT_BC7_UNORM_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK: return {16, 4};
    default: return {};
    }
}

inline bool two_planes(const ResourceBinding& resource) {
    return (resource.aspects & depth_stencil_aspects) == depth_stencil_aspects;
}

// Decodes a D3D12 subresource index. Combined depth-stencil images have two planes (0 depth, 1 stencil);
// every other image has one.
inline bool image_range(const ResourceBinding& resource, std::uint32_t subresource, VkImageSubresourceRange& out) {
    if (!resource.mip_levels || !resource.array_layers) return false;
    if (subresource == all_subresources) {
        out = {resource.aspects, 0, resource.mip_levels, 0, resource.array_layers};
        return true;
    }
    const std::uint64_t per_plane = std::uint64_t(resource.mip_levels) * resource.array_layers;
    const std::uint64_t plane = subresource / per_plane;
    if (plane >= (two_planes(resource) ? 2u : 1u)) return false;
    const auto within = static_cast<std::uint32_t>(subresource % per_plane);
    const VkImageAspectFlags aspect =
        two_planes(resource) ? (plane ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT) : resource.aspects;
    out = {aspect, within % resource.mip_levels, 1, within / resource.mip_levels, 1};
    return true;
}

// Decodes D3D12_BARRIER_SUBRESOURCE_RANGE.
inline bool barrier_range(const ResourceBinding& resource, const BarrierSubresourceRange& range,
                          VkImageSubresourceRange& out) {
    if (!range.mip_count) return image_range(resource, range.index_or_first_mip, out);
    const std::uint32_t planes = two_planes(resource) ? 2 : 1;
    if (range.index_or_first_mip >= resource.mip_levels ||
        range.mip_count > resource.mip_levels - range.index_or_first_mip)
        return false;
    if (range.first_layer >= resource.array_layers || !range.layer_count ||
        range.layer_count > resource.array_layers - range.first_layer)
        return false;
    if (range.first_plane >= planes || !range.plane_count || range.plane_count > planes - range.first_plane)
        return false;
    VkImageAspectFlags aspects = resource.aspects;
    if (planes == 2) {
        aspects = 0;
        for (std::uint32_t p = range.first_plane; p < range.first_plane + range.plane_count; ++p)
            aspects |= p ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    out = {aspects, range.index_or_first_mip, range.mip_count, range.first_layer, range.layer_count};
    return true;
}

// A part of an image range with one current layout.
struct LayoutPiece {
    VkImageSubresourceRange range;
    VkImageLayout layout;
};

// The current Vulkan layout of every tracked image subresource. Shared by all queues of a device; lists are
// emitted in submission order. Images are tracked from creation (their initial state) until destruction.
class LayoutTracker {
public:
    explicit LayoutTracker(bool separate_depth_stencil_layouts = false)
        : separate_(separate_depth_stencil_layouts) {}

    bool separate_depth_stencil_layouts() const { return separate_; }

    void track(ResourceId id, const ResourceBinding& resource, VkImageLayout layout) {
        Entry entry{resource.mip_levels, resource.array_layers, two_planes(resource) ? 2u : 1u,
                    separate_ && two_planes(resource), {}};
        entry.layouts.assign(std::size_t(entry.mips) * entry.layers * (entry.two_planes ? 2 : 1), layout);
        entries_[id] = std::move(entry);
    }

    void forget(ResourceId id) { entries_.erase(id); }
    bool tracked(ResourceId id) const { return entries_.count(id) != 0; }

    // Current layout of one D3D12 subresource; UNDEFINED when the resource or index is unknown.
    VkImageLayout layout(ResourceId id, std::uint32_t subresource) const {
        const auto it = entries_.find(id);
        if (it == entries_.end()) return VK_IMAGE_LAYOUT_UNDEFINED;
        const Entry& e = it->second;
        const std::uint32_t per_plane = e.mips * e.layers;
        const std::uint32_t plane = subresource / per_plane;
        if (plane >= e.planes) return VK_IMAGE_LAYOUT_UNDEFINED;
        return e.layouts[(e.two_planes ? plane : 0) * per_plane + subresource % per_plane];  // merged planes share
    }

    // The current layouts over a range: one piece when they all agree, else one per subresource.
    bool pieces(ResourceId id, const VkImageSubresourceRange& range, std::vector<LayoutPiece>& out) const {
        const auto it = entries_.find(id);
        if (it == entries_.end()) return false;
        const Entry& e = it->second;
        out.clear();
        bool uniform = true;
        VkImageLayout first = VK_IMAGE_LAYOUT_UNDEFINED;
        bool any = false;
        visit(e, range, [&](std::size_t index, VkImageAspectFlags, std::uint32_t, std::uint32_t) {
            if (!any) first = e.layouts[index];
            uniform = uniform && e.layouts[index] == first;
            any = true;
        });
        if (uniform) {
            out.push_back({range, first});
            return true;
        }
        visit(e, range, [&](std::size_t index, VkImageAspectFlags aspect, std::uint32_t mip, std::uint32_t layer) {
            out.push_back({{aspect, mip, 1, layer, 1}, e.layouts[index]});
        });
        return true;
    }

    void set(ResourceId id, const VkImageSubresourceRange& range, VkImageLayout layout) {
        const auto it = entries_.find(id);
        if (it == entries_.end()) return;
        Entry& e = it->second;
        visit(e, range, [&](std::size_t index, VkImageAspectFlags, std::uint32_t, std::uint32_t) {
            e.layouts[index] = layout;
        });
    }

private:
    struct Entry {
        std::uint32_t mips;
        std::uint32_t layers;
        std::uint32_t planes;  // of the resource
        bool two_planes;       // depth and stencil tracked apart
        std::vector<VkImageLayout> layouts;
    };

    template<typename F>
    static void visit(const Entry& e, const VkImageSubresourceRange& range, F&& f) {
        const std::uint32_t mip_end = range.levelCount == VK_REMAINING_MIP_LEVELS ? e.mips
                                                                                  : range.baseMipLevel + range.levelCount;
        const std::uint32_t layer_end = range.layerCount == VK_REMAINING_ARRAY_LAYERS
                                            ? e.layers
                                            : range.baseArrayLayer + range.layerCount;
        for (std::uint32_t plane = 0; plane < (e.two_planes ? 2u : 1u); ++plane) {
            VkImageAspectFlags aspect = range.aspectMask;
            if (e.two_planes) {
                aspect = plane ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
                if (!(range.aspectMask & aspect)) continue;
            }
            for (std::uint32_t layer = range.baseArrayLayer; layer < std::min(layer_end, e.layers); ++layer)
                for (std::uint32_t mip = range.baseMipLevel; mip < std::min(mip_end, e.mips); ++mip)
                    f((std::size_t(plane) * e.layers + layer) * e.mips + mip, aspect, mip, layer);
        }
    }

    bool separate_;
    std::unordered_map<ResourceId, Entry> entries_;
};

struct ReplayFunctions {
    PFN_vkCmdPipelineBarrier pipeline_barrier = nullptr;
    PFN_vkCmdClearColorImage clear_color_image = nullptr;
    PFN_vkCmdClearDepthStencilImage clear_depth_stencil_image = nullptr;
    // Optional; a list that needs a missing one is rejected with ReplayResult::unsupported.
    PFN_vkCmdPipelineBarrier2 pipeline_barrier2 = nullptr;  // enhanced barriers
    PFN_vkCmdBeginRendering begin_rendering = nullptr;      // rectangle clears
    PFN_vkCmdEndRendering end_rendering = nullptr;
    PFN_vkCmdClearAttachments clear_attachments = nullptr;
    PFN_vkCmdCopyBuffer copy_buffer = nullptr;                  // copies
    PFN_vkCmdCopyBufferToImage copy_buffer_to_image = nullptr;
    PFN_vkCmdCopyImageToBuffer copy_image_to_buffer = nullptr;
    PFN_vkCmdCopyImage copy_image = nullptr;
    PFN_vkCmdResolveImage resolve_image = nullptr;              // resolves
    // Returns a 2D-array view of the range (caller-owned, alive until execution completes); rectangle clears.
    std::function<VkImageView(const ResourceBinding&, const VkImageSubresourceRange&)> image_view;
    bool separate_depth_stencil_layouts = false;  // must match the LayoutTracker, if one is used
};

enum class ReplayResult {
    success,
    list_open,
    unresolved_resource,
    invalid_subresource,
    invalid_clear,
    invalid_copy,
    unsupported
};

namespace detail {

struct Prepared {
    ResourceBinding binding;
    VkImageSubresourceRange range{};  // the barrier range, or the cleared view (cleared aspects only)
    VkImageView view = VK_NULL_HANDLE;
    std::vector<VkClearRect> rects;   // rectangle clears, clipped to the mip
    bool skip = false;                // every rectangle was empty
    ResourceBinding source;                   // copies: the source resource
    VkImageSubresourceRange source_range{};   // copies: the source subresources (range: the destination's)
    std::vector<VkBufferCopy> buffer_copies;
    std::vector<VkBufferImageCopy> buffer_image_copies;  // one direction, given by which side is an image
    std::vector<VkImageCopy> image_copies;
    VkImageResolve resolve{};  // resolves
};

// ClearRenderTargetView takes FLOATs even for integer RTVs. For the UINT formats currently exposed by
// the device, Windows D3D12 (WARP and AMD) clamps to the component range and truncates fractions. Vulkan
// instead interprets the matching union member as an integer already; copying float bits changes the value.
inline VkClearColorValue render_target_clear_color(VkFormat format, const ClearColor& input) {
    // Integer formats convert the float numerically: NaN maps to zero, negatives clamp to the low end, positive
    // overflow to the high end, and fractions truncate toward zero. Unsigned uses uint32, signed uses int32.
    std::uint32_t unsigned_maximum = 0;
    std::int64_t signed_minimum = 0, signed_maximum = 0;
    bool is_signed = false;
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UINT: unsigned_maximum = 255; break;
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32G32B32A32_UINT: unsigned_maximum = std::numeric_limits<std::uint32_t>::max(); break;
    case VK_FORMAT_R8G8B8A8_SINT: signed_minimum = -128; signed_maximum = 127; is_signed = true; break;
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32G32B32A32_SINT:
        signed_minimum = std::numeric_limits<std::int32_t>::min();
        signed_maximum = std::numeric_limits<std::int32_t>::max();
        is_signed = true;
        break;
    default: break;
    }
    VkClearColorValue value{};
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (is_signed) {
            const double v = double(input[i]);
            std::int32_t out;
            if (v != v) {
                out = 0;  // NaN
            } else if (v <= double(signed_minimum)) {
                out = static_cast<std::int32_t>(signed_minimum);
            } else if (v >= double(signed_maximum)) {
                out = static_cast<std::int32_t>(signed_maximum);
            } else {
                out = static_cast<std::int32_t>(v);  // truncates toward zero
            }
            value.int32[i] = out;
        } else if (!unsigned_maximum) {
            value.float32[i] = input[i];
        } else if (!(input[i] > 0.0f)) {
            value.uint32[i] = 0; // NaN, negative values and either zero; no out-of-range float-to-int cast.
        } else if (double(input[i]) >= double(unsigned_maximum)) {
            value.uint32[i] = unsigned_maximum; // Includes +infinity; UINT32_MAX itself is not representable as float.
        } else {
            value.uint32[i] = static_cast<std::uint32_t>(input[i]);
        }
    }
    return value;
}

inline bool view_range(const ResourceBinding& resource, const TargetView& view, VkImageAspectFlags aspects,
                       VkImageSubresourceRange& out) {
    if (view.mip >= resource.mip_levels) return false;
    if (view.first_layer >= resource.array_layers || view.layer_count > resource.array_layers - view.first_layer)
        return false;
    out = {aspects, view.mip, 1, view.first_layer, view.layer_count};
    return true;
}

// Without separate depth/stencil layouts both planes of a combined image move together.
inline VkImageSubresourceRange planes_for_layout(const ResourceBinding& resource, VkImageSubresourceRange range,
                                                 bool separate) {
    if (two_planes(resource) && !separate) range.aspectMask = depth_stencil_aspects;
    return range;
}

inline VkExtent2D mip_extent(const ResourceBinding& resource, std::uint32_t mip) {
    return {std::max(1u, resource.width >> mip), std::max(1u, resource.height >> mip)};
}

inline std::vector<VkClearRect> clip(std::span<const Rect> rects, VkExtent2D extent, std::uint32_t layers) {
    std::vector<VkClearRect> out;
    for (const Rect& r : rects) {
        const std::int64_t left = std::max<std::int64_t>(r.left, 0), top = std::max<std::int64_t>(r.top, 0);
        const std::int64_t right = std::min<std::int64_t>(r.right, extent.width);
        const std::int64_t bottom = std::min<std::int64_t>(r.bottom, extent.height);
        if (right <= left || bottom <= top) continue;
        out.push_back({{{std::int32_t(left), std::int32_t(top)}, {std::uint32_t(right - left), std::uint32_t(bottom - top)}},
                       0, layers});
    }
    return out;
}

inline void image_barrier(const ReplayFunctions& vk, VkCommandBuffer command_buffer, VkImage image,
                          const VkImageSubresourceRange& range, VkImageLayout old_layout, VkImageLayout new_layout,
                          VkAccessFlags src_access, VkPipelineStageFlags src_stages, VkAccessFlags dst_access,
                          VkPipelineStageFlags dst_stages) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = range;
    vk.pipeline_barrier(command_buffer, src_stages, dst_stages, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

inline void memory_barrier(const ReplayFunctions& vk, VkCommandBuffer command_buffer, VkAccessFlags src_access,
                           VkPipelineStageFlags src_stages, VkAccessFlags dst_access, VkPipelineStageFlags dst_stages) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    vk.pipeline_barrier(command_buffer, src_stages, dst_stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// The current layouts of a range: from the tracker, or `assumed` for an untracked image.
inline std::vector<LayoutPiece> current(const LayoutTracker* tracker, ResourceId id, const VkImageSubresourceRange& range,
                                        VkImageLayout assumed) {
    std::vector<LayoutPiece> out;
    if (!tracker || !tracker->pieces(id, range, out)) out = {{range, assumed}};
    return out;
}

// Images registered with the tracker in UNDEFINED have never been used. D3D12 committed resources start zeroed,
// so each such piece is cleared to zero (colour 0, depth 0, stencil 0) and moved to `layout`, the layout the
// operation about to run expects, before that operation; with not_zeroed, or a block-compressed format (which
// Vulkan cannot clear), it is only moved. The tracker records it.
inline void initialize(const ReplayFunctions& vk, VkCommandBuffer command_buffer, const ResourceBinding& resource,
                       ResourceId id, std::vector<LayoutPiece>& pieces, VkImageLayout layout, LayoutTracker* tracker) {
    constexpr VkAccessFlags any_memory = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    for (auto& piece : pieces) {
        if (piece.layout != VK_IMAGE_LAYOUT_UNDEFINED) continue;
        // Vulkan cannot clear block-compressed images; those start undefined (applications upload them first).
        const bool compressed = block_format(resource.format, VK_IMAGE_ASPECT_COLOR_BIT).block > 1;
        if (resource.not_zeroed || compressed) {
            image_barrier(vk, command_buffer, resource.image, piece.range, VK_IMAGE_LAYOUT_UNDEFINED, layout, 0,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, any_memory, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        } else {
            const VkImageLayout dst = image_layout(resource, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            image_barrier(vk, command_buffer, resource.image, piece.range, VK_IMAGE_LAYOUT_UNDEFINED, dst, 0,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT);
            if (piece.range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) {
                const VkClearColorValue zero{};
                vk.clear_color_image(command_buffer, resource.image, dst, &zero, 1, &piece.range);
            } else {
                const VkClearDepthStencilValue zero{};
                vk.clear_depth_stencil_image(command_buffer, resource.image, dst, &zero, 1, &piece.range);
            }
            image_barrier(vk, command_buffer, resource.image, piece.range, dst, layout, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, any_memory, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }
        piece.layout = layout;
        if (tracker) tracker->set(id, piece.range, layout);
    }
}

inline bool within(std::uint64_t offset, std::uint64_t size, std::uint64_t total) {
    return offset <= total && size <= total - offset;
}

// The one plane of one subresource a texture copy location names.
inline bool copy_subresource(const ResourceBinding& resource, std::uint32_t subresource, VkImageSubresourceRange& out) {
    return resource.image && subresource != all_subresources && image_range(resource, subresource, out);
}

// CopyTextureRegion: texture to texture, footprint to texture (upload) or texture to footprint (readback).
// Block-compressed footprints are padded to whole blocks; the Vulkan extent stops at the mip edge.
inline ReplayResult prepare_texture_copy(const CopyTexture& copy, Prepared& p) {
    if (!copy.dst.footprint && !copy.src.footprint) {
        VkImageSubresourceRange dst{}, src{};
        if (!copy_subresource(p.binding, copy.dst.subresource, dst) ||
            !copy_subresource(p.source, copy.src.subresource, src))
            return ReplayResult::invalid_subresource;
        if (dst.aspectMask != src.aspectMask || p.binding.volume != p.source.volume ||
            p.binding.samples != p.source.samples)
            return ReplayResult::invalid_copy;
        const VkExtent2D from = mip_extent(p.source, src.baseMipLevel), to = mip_extent(p.binding, dst.baseMipLevel);
        const auto from_depth = std::max(1u, p.source.depth >> src.baseMipLevel);
        const auto to_depth = std::max(1u, p.binding.depth >> dst.baseMipLevel);
        const CopyBox box = copy.has_box ? copy.box : CopyBox{0, 0, 0, from.width, from.height, from_depth};
        if (box.right > from.width || box.bottom > from.height || box.back > from_depth) return ReplayResult::invalid_copy;
        const std::uint32_t w = box.right - box.left, h = box.bottom - box.top, d = box.back - box.front;
        if (!within(copy.x, w, to.width) || !within(copy.y, h, to.height) || !within(copy.z, d, to_depth))
            return ReplayResult::invalid_copy;
        p.image_copies.push_back({{src.aspectMask, src.baseMipLevel, src.baseArrayLayer, 1},
                                  {std::int32_t(box.left), std::int32_t(box.top), std::int32_t(box.front)},
                                  {dst.aspectMask, dst.baseMipLevel, dst.baseArrayLayer, 1},
                                  {std::int32_t(copy.x), std::int32_t(copy.y), std::int32_t(copy.z)},
                                  {w, h, d}});
        p.range = dst;
        p.source_range = src;
        return ReplayResult::success;
    }

    const bool upload = copy.src.footprint;
    const ResourceBinding& image = upload ? p.binding : p.source;
    const ResourceBinding& buffer = upload ? p.source : p.binding;
    const CopyLocation& texture = upload ? copy.dst : copy.src;
    const CopyLocation& placed = upload ? copy.src : copy.dst;
    if (!buffer.buffer) return ReplayResult::invalid_copy;
    VkImageSubresourceRange range{};
    if (!copy_subresource(image, texture.subresource, range)) return ReplayResult::invalid_subresource;
    if (image.samples != 1) return ReplayResult::invalid_copy;  // multisampled textures have no footprint
    const BlockFormat format = block_format(image.format, range.aspectMask);
    if (!format.bytes || placed.row_pitch % format.bytes) return ReplayResult::unsupported;
    const VkExtent2D mip = mip_extent(image, range.baseMipLevel);
    const auto mip_depth = std::max(1u, image.depth >> range.baseMipLevel);
    // The box is in the source's coordinates: the footprint for an upload, the texture for a readback.
    const CopyBox box = copy.has_box ? copy.box
                        : upload     ? CopyBox{0, 0, 0, placed.width, placed.height, placed.depth}
                                     : CopyBox{0, 0, 0, mip.width, mip.height, mip_depth};
    const std::uint32_t texture_x = upload ? copy.x : box.left, texture_y = upload ? copy.y : box.top;
    const std::uint32_t placed_x = upload ? box.left : copy.x, placed_y = upload ? box.top : copy.y;
    std::uint32_t w = box.right - box.left, h = box.bottom - box.top;
    const auto texture_z = upload ? copy.z : box.front, placed_z = upload ? box.front : copy.z;
    const auto depth = box.back - box.front;
    if (!depth || !within(placed_z, depth, placed.depth) || !within(texture_z, depth, mip_depth) ||
        !within(placed_x, w, placed.width) || !within(placed_y, h, placed.height)) return ReplayResult::invalid_copy;
    if (texture_x >= mip.width || texture_y >= mip.height) return ReplayResult::invalid_copy;
    if (format.block > 1) {
        w = std::min(w, mip.width - texture_x);
        h = std::min(h, mip.height - texture_y);
    }
    if (!within(texture_x, w, mip.width) || !within(texture_y, h, mip.height)) return ReplayResult::invalid_copy;
    if ((placed_x | placed_y | texture_x | texture_y) % format.block) return ReplayResult::invalid_copy;
    const auto slice_rows = (std::uint64_t(placed.height) + format.block - 1) / format.block;
    const std::uint64_t slice_pitch = slice_rows * placed.row_pitch;
    if (!slice_pitch || placed.offset > buffer.size || placed_z > (buffer.size - placed.offset) / slice_pitch)
        return ReplayResult::invalid_copy;
    const auto slice_start = placed.offset + placed_z * slice_pitch;
    const auto in_slice = std::uint64_t(placed_y / format.block) * placed.row_pitch +
                          std::uint64_t(placed_x / format.block) * format.bytes;
    if (!within(slice_start, in_slice, buffer.size)) return ReplayResult::invalid_copy;
    const auto offset = slice_start + in_slice;
    const std::uint32_t alignment = (range.aspectMask & depth_stencil_aspects) ? 4 : format.bytes;
    if (offset % alignment) return ReplayResult::unsupported;
    const std::uint64_t rows = (std::uint64_t(h) + format.block - 1) / format.block;
    const std::uint64_t row_bytes = (std::uint64_t(w) + format.block - 1) / format.block * format.bytes;
    if (!rows || !row_bytes || !within(std::uint64_t(placed_x / format.block) * format.bytes, row_bytes, placed.row_pitch) ||
        depth - 1 > (buffer.size - offset) / slice_pitch ||
        !within(offset + (depth - 1) * slice_pitch, (rows - 1) * placed.row_pitch + row_bytes, buffer.size))
        return ReplayResult::invalid_copy;
    VkBufferImageCopy region{};
    region.bufferOffset = offset;
    region.bufferRowLength = placed.row_pitch / format.bytes * format.block;
    region.bufferImageHeight = depth > 1 ? std::uint32_t(slice_rows * format.block) : 0;
    region.imageSubresource = {range.aspectMask, range.baseMipLevel, range.baseArrayLayer, 1};
    region.imageOffset = {std::int32_t(texture_x), std::int32_t(texture_y), std::int32_t(texture_z)};
    region.imageExtent = {w, h, depth};
    p.buffer_image_copies.push_back(region);
    (upload ? p.range : p.source_range) = range;
    return ReplayResult::success;
}

// CopyResource: the whole buffer, or every subresource of two textures of the same shape.
inline ReplayResult prepare_resource_copy(Prepared& p) {
    if (p.binding.buffer && p.source.buffer) {
        if (p.binding.size != p.source.size) return ReplayResult::invalid_copy;
        p.buffer_copies.push_back({0, 0, p.source.size});
        return ReplayResult::success;
    }
    const ResourceBinding &dst = p.binding, &src = p.source;
    if (!dst.image || !src.image || dst.width != src.width || dst.height != src.height ||
        dst.depth != src.depth || dst.volume != src.volume || dst.mip_levels != src.mip_levels ||
        dst.array_layers != src.array_layers || dst.aspects != src.aspects || dst.samples != src.samples ||
        block_format(dst.format, VK_IMAGE_ASPECT_COLOR_BIT).bytes != block_format(src.format, VK_IMAGE_ASPECT_COLOR_BIT).bytes)
        return ReplayResult::invalid_copy;
    for (const VkImageAspectFlags aspect : {VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_STENCIL_BIT}) {
        if (!(dst.aspects & aspect)) continue;
        for (std::uint32_t mip = 0; mip < dst.mip_levels; ++mip) {
            const VkExtent2D extent = mip_extent(dst, mip);
            p.image_copies.push_back({{aspect, mip, 0, dst.array_layers}, {0, 0, 0}, {aspect, mip, 0, dst.array_layers},
                                      {0, 0, 0}, {extent.width, extent.height, std::max(1u, src.depth >> mip)}});
        }
    }
    p.range = {dst.aspects, 0, dst.mip_levels, 0, dst.array_layers};
    p.source_range = {src.aspects, 0, src.mip_levels, 0, src.array_layers};
    return ReplayResult::success;
}

// Resolve a multisampled colour image into a single-sampled image of the same format.
inline ReplayResult prepare_resolve(const ResolveSubresource& resolve, Prepared& p) {
    VkImageSubresourceRange dst{}, src{};
    if (!copy_subresource(p.binding, resolve.dst_subresource, dst) ||
        !copy_subresource(p.source, resolve.src_subresource, src))
        return ReplayResult::invalid_subresource;
    if (p.source.samples <= 1 || p.binding.samples != 1 || p.binding.volume) return ReplayResult::invalid_copy;
    const VkExtent2D from = mip_extent(p.source, src.baseMipLevel), to = mip_extent(p.binding, dst.baseMipLevel);
    VkOffset3D src_offset{}, dst_offset{};
    VkExtent3D extent{from.width, from.height, 1};
    if (const auto& region = resolve.region) {
        dst_offset = {std::int32_t(region->dst_x), std::int32_t(region->dst_y), 0};
        if (const auto& r = region->src_rect) {
            if (std::uint32_t(r->right) > from.width || std::uint32_t(r->bottom) > from.height)
                return ReplayResult::invalid_copy;
            src_offset = {r->left, r->top, 0};
            extent = {std::uint32_t(r->right - r->left), std::uint32_t(r->bottom - r->top), 1};
        }
        if (!within(region->dst_x, extent.width, to.width) || !within(region->dst_y, extent.height, to.height))
            return ReplayResult::invalid_copy;
    } else if (from.width != to.width || from.height != to.height) {
        return ReplayResult::invalid_copy;
    }
    if (src.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT || dst.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT ||
        p.source.format != p.binding.format)
        return ReplayResult::unsupported;
    p.resolve = {{src.aspectMask, src.baseMipLevel, src.baseArrayLayer, 1}, src_offset,
                 {dst.aspectMask, dst.baseMipLevel, dst.baseArrayLayer, 1}, dst_offset, extent};
    p.range = dst;
    p.source_range = src;
    return ReplayResult::success;
}

// The layout a copy uses for a range: D3D12 requires copy_dest/copy_source, or common (GENERAL), where an implicit
// promotion lets the copy run in place. Anything else is moved to `wanted` and tracked.
inline VkImageLayout copy_layout(const ReplayFunctions& vk, VkCommandBuffer command_buffer,
                                 const ResourceBinding& resource, ResourceId id, const VkImageSubresourceRange& range,
                                 VkImageLayout wanted, VkAccessFlags access, LayoutTracker* tracker) {
    const VkImageLayout target = image_layout(resource, wanted);
    auto pieces = current(tracker, id, range, target);
    initialize(vk, command_buffer, resource, id, pieces, target, tracker);
    if (pieces.size() == 1 && (pieces[0].layout == target || pieces[0].layout == VK_IMAGE_LAYOUT_GENERAL))
        return pieces[0].layout;
    for (const auto& piece : pieces) {
        if (piece.layout == target) continue;
        image_barrier(vk, command_buffer, resource.image, piece.range, piece.layout, target,
                      VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, access,
                      VK_PIPELINE_STAGE_TRANSFER_BIT);
    }
    if (tracker) tracker->set(id, range, target);
    return target;
}

// Moves every piece to `target` (src: the D3D12 state the operation requires), runs `body`, and moves the pieces
// back to where they were. Pieces already in `target` need no transition.
template<typename Body>
void round_trip(const ReplayFunctions& vk, VkCommandBuffer command_buffer, VkImage image,
                const std::vector<LayoutPiece>& pieces, VkImageLayout target, const VulkanAccess& state,
                VkAccessFlags target_access, VkPipelineStageFlags target_stages, Body&& body) {
    for (const auto& piece : pieces) {
        if (piece.layout == target) {
            memory_barrier(vk, command_buffer, state.access, state.stages, target_access, target_stages);
            continue;
        }
        image_barrier(vk, command_buffer, image, piece.range, piece.layout, target, state.access, state.stages,
                      target_access, target_stages);
    }
    body();
    for (const auto& piece : pieces) {
        if (piece.layout == target) {
            memory_barrier(vk, command_buffer, target_access, target_stages, state.access, state.stages);
            continue;
        }
        image_barrier(vk, command_buffer, image, piece.range, target, piece.layout, target_access, target_stages,
                      state.access, state.stages);
    }
}

} // namespace detail

// Destination and source of a copy or resolve command.
inline std::pair<ResourceId, ResourceId> copy_ids(const Command& command) {
    if (const auto* c = std::get_if<CopyBuffer>(&command)) return {c->dst, c->src};
    if (const auto* c = std::get_if<CopyTexture>(&command)) return {c->dst.resource, c->src.resource};
    if (const auto* c = std::get_if<CopyResource>(&command)) return {c->dst, c->src};
    if (const auto* c = std::get_if<ResolveSubresource>(&command)) return {c->dst, c->src};
    return {0, 0};
}

// A list checked against the current resources, ready to emit.
class PreparedList {
public:
    const CommandList* list = nullptr;
    std::vector<detail::Prepared> commands;
};

// Resolves and checks every command. Nothing is written to a command buffer.
// Resolve maps a ResourceId to a const ResourceBinding* (nullptr: unknown).
template<typename Resolve>
ReplayResult prepare(const CommandList& list, Resolve&& resolve, const ReplayFunctions& vk, PreparedList& out) {
    if (!list.closed()) return ReplayResult::list_open;
    const auto commands = list.commands();
    out.list = &list;
    out.commands.assign(commands.size(), {});
    const bool rendering = vk.begin_rendering && vk.end_rendering && vk.clear_attachments && vk.image_view;
    const bool copies = vk.copy_buffer && vk.copy_buffer_to_image && vk.copy_image_to_buffer && vk.copy_image;

    for (std::size_t i = 0; i < commands.size(); ++i) {
        const auto& command = commands[i];
        auto& p = out.commands[i];
        const auto bind = [&](ResourceId id) {
            const ResourceBinding* binding = resolve(id);
            if (binding) p.binding = *binding;
            return binding != nullptr;
        };
        const auto rect_view = [&](std::span<const Rect> rects, const TargetView& target, VkImageAspectFlags view_aspects) {
            if (rects.empty()) return ReplayResult::success;
            if (!rendering) return ReplayResult::unsupported;
            p.rects = detail::clip(rects, detail::mip_extent(p.binding, target.mip), target.layer_count);
            if (p.rects.empty()) {
                p.skip = true;
                return ReplayResult::success;
            }
            VkImageSubresourceRange range = p.range;
            range.aspectMask = view_aspects;
            p.view = vk.image_view(p.binding, range);
            return p.view ? ReplayResult::success : ReplayResult::unsupported;
        };

        if (const auto* transition = std::get_if<TransitionBarrier>(&command)) {
            if (!bind(transition->resource)) return ReplayResult::unresolved_resource;
            if (p.binding.image == VK_NULL_HANDLE) {
                if (transition->subresource != 0 && transition->subresource != all_subresources)
                    return ReplayResult::invalid_subresource;
            } else if (!image_range(p.binding, transition->subresource, p.range)) {
                return ReplayResult::invalid_subresource;
            }
        } else if (std::holds_alternative<GlobalBarrier>(command) || std::holds_alternative<BufferBarrier>(command)) {
            if (!vk.pipeline_barrier2) return ReplayResult::unsupported;
            if (const auto* buffer_barrier = std::get_if<BufferBarrier>(&command))
                if (!bind(buffer_barrier->resource)) return ReplayResult::unresolved_resource;
        } else if (const auto* texture = std::get_if<TextureBarrier>(&command)) {
            if (!vk.pipeline_barrier2) return ReplayResult::unsupported;
            if (!bind(texture->resource)) return ReplayResult::unresolved_resource;
            if (!p.binding.image || !barrier_range(p.binding, texture->range, p.range))
                return ReplayResult::invalid_subresource;
        } else if (const auto* color_clear = std::get_if<ClearRenderTarget>(&command)) {
            if (!bind(color_clear->target.resource)) return ReplayResult::unresolved_resource;
            if (!p.binding.image || p.binding.aspects != VK_IMAGE_ASPECT_COLOR_BIT) return ReplayResult::invalid_clear;
            if (!detail::view_range(p.binding, color_clear->target, VK_IMAGE_ASPECT_COLOR_BIT, p.range))
                return ReplayResult::invalid_subresource;
            if (const auto r = rect_view(color_clear->rects, color_clear->target, VK_IMAGE_ASPECT_COLOR_BIT); r != ReplayResult::success)
                return r;
        } else if (const auto* depth_clear = std::get_if<ClearDepthStencil>(&command)) {
            if (!bind(depth_clear->target.resource)) return ReplayResult::unresolved_resource;
            VkImageAspectFlags aspects = 0;
            if (depth_clear->flags & clear_depth) aspects |= VK_IMAGE_ASPECT_DEPTH_BIT;
            if (depth_clear->flags & clear_stencil) aspects |= VK_IMAGE_ASPECT_STENCIL_BIT;
            const VkImageAspectFlags available = p.binding.aspects & depth_stencil_aspects;
            if (!p.binding.image || !available || (aspects & ~available) || (p.binding.aspects & ~depth_stencil_aspects))
                return ReplayResult::invalid_clear;
            if (!detail::view_range(p.binding, depth_clear->target, aspects, p.range))
                return ReplayResult::invalid_subresource;
            if (const auto r = rect_view(depth_clear->rects, depth_clear->target, p.binding.aspects); r != ReplayResult::success)
                return r;
        } else if (std::holds_alternative<CopyBuffer>(command) || std::holds_alternative<CopyTexture>(command) ||
                   std::holds_alternative<CopyResource>(command)) {
            if (!copies) return ReplayResult::unsupported;
            const auto [dst_id, src_id] = copy_ids(command);
            const ResourceBinding* source = resolve(src_id);
            if (!bind(dst_id) || !source) return ReplayResult::unresolved_resource;
            p.source = *source;
            ReplayResult result = ReplayResult::success;
            if (const auto* buffer_copy = std::get_if<CopyBuffer>(&command)) {
                if (!p.binding.buffer || !p.source.buffer ||
                    !detail::within(buffer_copy->dst_offset, buffer_copy->size, p.binding.size) ||
                    !detail::within(buffer_copy->src_offset, buffer_copy->size, p.source.size))
                    return ReplayResult::invalid_copy;
                p.buffer_copies.push_back({buffer_copy->src_offset, buffer_copy->dst_offset, buffer_copy->size});
            } else if (const auto* texture_copy = std::get_if<CopyTexture>(&command)) {
                result = detail::prepare_texture_copy(*texture_copy, p);
            } else {
                result = detail::prepare_resource_copy(p);
            }
            if (result != ReplayResult::success) return result;
        } else if (const auto* resolve_command = std::get_if<ResolveSubresource>(&command)) {
            if (!vk.resolve_image) return ReplayResult::unsupported;
            const ResourceBinding* source = resolve(resolve_command->src);
            if (!bind(resolve_command->dst) || !source) return ReplayResult::unresolved_resource;
            p.source = *source;
            if (const auto result = detail::prepare_resolve(*resolve_command, p); result != ReplayResult::success)
                return result;
        }
    }
    return ReplayResult::success;
}

// Writes a prepared list. With a tracker, barriers start from the tracked layouts and update them.
inline void emit(const PreparedList& prepared, VkCommandBuffer command_buffer, const ReplayFunctions& vk,
                 LayoutTracker* tracker = nullptr) {
    const auto commands = prepared.list->commands();
    const bool separate = vk.separate_depth_stencil_layouts;
    const VulkanAccess transfer_dst = vulkan_access(d3d12_state::copy_dest, false);

    for (std::size_t i = 0; i < commands.size(); ++i) {
        const auto& command = commands[i];
        const auto& p = prepared.commands[i];
        const ResourceId id = std::visit(
            [](const auto& c) -> ResourceId {
                using T = std::decay_t<decltype(c)>;
                if constexpr (std::is_same_v<T, ClearRenderTarget> || std::is_same_v<T, ClearDepthStencil>)
                    return c.target.resource;
                else if constexpr (requires { c.resource; })
                    return c.resource;
                else
                    return 0;
            },
            command);

        if (const auto* transition = std::get_if<TransitionBarrier>(&command)) {
            const VulkanAccess from = vulkan_access(transition->before, p.binding.presentable);
            const VulkanAccess to = vulkan_access(transition->after, p.binding.presentable);
            if (!p.binding.image) {
                detail::memory_barrier(vk, command_buffer, from.access, from.stages, to.access, to.stages);
                continue;
            }
            const auto range = detail::planes_for_layout(p.binding, p.range, separate);
            const VkImageLayout new_layout = image_layout(p.binding, to.layout);
            auto pieces = detail::current(tracker, id, range, image_layout(p.binding, from.layout));
            detail::initialize(vk, command_buffer, p.binding, id, pieces, image_layout(p.binding, from.layout), tracker);
            for (const auto& piece : pieces)
                detail::image_barrier(vk, command_buffer, p.binding.image, piece.range, piece.layout, new_layout,
                                      from.access, from.stages, to.access, to.stages);
            if (tracker) tracker->set(id, range, new_layout);
        } else if (std::holds_alternative<UavBarrier>(command)) {
            const VulkanAccess uav = vulkan_access(d3d12_state::unordered_access, false);
            detail::memory_barrier(vk, command_buffer, VK_ACCESS_SHADER_WRITE_BIT, uav.stages, uav.access, uav.stages);
        } else if (std::holds_alternative<AliasingBarrier>(command)) {
            detail::memory_barrier(vk, command_buffer, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        } else if (std::holds_alternative<GlobalBarrier>(command) || std::holds_alternative<BufferBarrier>(command)) {
            const auto [sync_before, sync_after, access_before, access_after] = std::visit(
                [](const auto& c) -> std::array<std::uint32_t, 4> {
                    if constexpr (requires { c.sync_before; })
                        return {c.sync_before, c.sync_after, c.access_before, c.access_after};
                    else
                        return {};
                },
                command);
            const BarrierScope src = vulkan_barrier_scope(sync_before, access_before);
            const BarrierScope dst = vulkan_barrier_scope(sync_after, access_after);
            VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            barrier.srcStageMask = src.stages;
            barrier.dstStageMask = dst.stages;
            barrier.srcAccessMask = src.access;
            barrier.dstAccessMask = dst.access;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &barrier;
            vk.pipeline_barrier2(command_buffer, &dependency);
        } else if (const auto* texture = std::get_if<TextureBarrier>(&command)) {
            const auto range = detail::planes_for_layout(p.binding, p.range, separate);
            const bool discard = (texture->flags & texture_barrier_discard) ||
                                 texture->layout_before == d3d12_layout::undefined;
            const VkImageLayout new_layout =
                image_layout(p.binding, vulkan_layout(texture->layout_after, p.binding.presentable));
            std::vector<LayoutPiece> pieces;
            if (discard)
                pieces = {{range, VK_IMAGE_LAYOUT_UNDEFINED}};
            else {
                const VkImageLayout before =
                    image_layout(p.binding, vulkan_layout(texture->layout_before, p.binding.presentable));
                pieces = detail::current(tracker, id, range, before);
                detail::initialize(vk, command_buffer, p.binding, id, pieces, before, tracker);
            }
            const BarrierScope src = vulkan_barrier_scope(texture->sync_before, texture->access_before);
            const BarrierScope dst = vulkan_barrier_scope(texture->sync_after, texture->access_after);
            std::vector<VkImageMemoryBarrier2> barriers;
            for (const auto& piece : pieces) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = src.stages;
                b.dstStageMask = dst.stages;
                b.srcAccessMask = src.access;
                b.dstAccessMask = dst.access;
                b.oldLayout = piece.layout;
                b.newLayout = new_layout;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = p.binding.image;
                b.subresourceRange = piece.range;
                barriers.push_back(b);
            }
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.imageMemoryBarrierCount = std::uint32_t(barriers.size());
            dependency.pImageMemoryBarriers = barriers.data();
            vk.pipeline_barrier2(command_buffer, &dependency);
            if (tracker) tracker->set(id, range, new_layout);
        } else if (const auto* color_clear = std::get_if<ClearRenderTarget>(&command)) {
            if (p.skip) continue;
            const VulkanAccess state = vulkan_access(d3d12_state::render_target, false);
            auto pieces = detail::current(tracker, id, p.range, image_layout(p.binding, state.layout));
            detail::initialize(vk, command_buffer, p.binding, id, pieces, image_layout(p.binding, state.layout), tracker);
            const VkClearColorValue color = detail::render_target_clear_color(p.binding.format, color_clear->color);
            if (p.rects.empty()) {
                detail::round_trip(vk, command_buffer, p.binding.image, pieces, image_layout(p.binding, transfer_dst.layout),
                                   state, transfer_dst.access, transfer_dst.stages, [&] {
                    vk.clear_color_image(command_buffer, p.binding.image, image_layout(p.binding, transfer_dst.layout),
                                         &color, 1, &p.range);
                });
                continue;
            }
            const VkImageLayout attachment = image_layout(p.binding, state.layout);
            detail::round_trip(vk, command_buffer, p.binding.image, pieces, attachment, state, state.access,
                               state.stages, [&] {
                VkRenderingAttachmentInfo info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                info.imageView = p.view;
                info.imageLayout = attachment;
                info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
                rendering.renderArea = {{0, 0}, detail::mip_extent(p.binding, color_clear->target.mip)};
                rendering.layerCount = color_clear->target.layer_count;
                rendering.colorAttachmentCount = 1;
                rendering.pColorAttachments = &info;
                vk.begin_rendering(command_buffer, &rendering);
                VkClearAttachment attachment_clear{VK_IMAGE_ASPECT_COLOR_BIT, 0, {}};
                attachment_clear.clearValue.color = color;
                vk.clear_attachments(command_buffer, 1, &attachment_clear, std::uint32_t(p.rects.size()), p.rects.data());
                vk.end_rendering(command_buffer);
            });
        } else if (const auto* depth_clear = std::get_if<ClearDepthStencil>(&command)) {
            if (p.skip) continue;
            // With separate layouts only the cleared planes move; otherwise both do.
            const auto range = detail::planes_for_layout(p.binding, p.range, separate);
            const VulkanAccess state = vulkan_access(d3d12_state::depth_write, false);
            auto pieces = detail::current(tracker, id, range, image_layout(p.binding, state.layout));
            detail::initialize(vk, command_buffer, p.binding, id, pieces, image_layout(p.binding, state.layout), tracker);
            const VkClearDepthStencilValue value{depth_clear->depth, depth_clear->stencil};
            if (p.rects.empty()) {
                detail::round_trip(vk, command_buffer, p.binding.image, pieces, image_layout(p.binding, transfer_dst.layout),
                                   state, transfer_dst.access, transfer_dst.stages, [&] {
                    vk.clear_depth_stencil_image(command_buffer, p.binding.image,
                                                 image_layout(p.binding, transfer_dst.layout), &value, 1, &p.range);
                });
                continue;
            }
            const VkImageLayout attachment = image_layout(p.binding, state.layout);
            detail::round_trip(vk, command_buffer, p.binding.image, pieces, attachment, state, state.access,
                               state.stages, [&] {
                VkRenderingAttachmentInfo info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                info.imageView = p.view;
                info.imageLayout = attachment;
                info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
                rendering.renderArea = {{0, 0}, detail::mip_extent(p.binding, depth_clear->target.mip)};
                rendering.layerCount = depth_clear->target.layer_count;
                if (p.range.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) rendering.pDepthAttachment = &info;
                if (p.range.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) rendering.pStencilAttachment = &info;
                vk.begin_rendering(command_buffer, &rendering);
                VkClearAttachment attachment_clear{p.range.aspectMask, 0, {}};
                attachment_clear.clearValue.depthStencil = value;
                vk.clear_attachments(command_buffer, 1, &attachment_clear, std::uint32_t(p.rects.size()), p.rects.data());
                vk.end_rendering(command_buffer);
            });
        } else if (std::holds_alternative<CopyBuffer>(command) || std::holds_alternative<CopyTexture>(command) ||
                   std::holds_alternative<CopyResource>(command)) {
            const auto [dst_id, src_id] = copy_ids(command);
            if (!p.buffer_copies.empty()) {
                vk.copy_buffer(command_buffer, p.source.buffer, p.binding.buffer, std::uint32_t(p.buffer_copies.size()),
                               p.buffer_copies.data());
            } else if (!p.image_copies.empty()) {
                const VkImageLayout src_layout = detail::copy_layout(
                    vk, command_buffer, p.source, src_id, detail::planes_for_layout(p.source, p.source_range, separate),
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, tracker);
                const VkImageLayout dst_layout = detail::copy_layout(
                    vk, command_buffer, p.binding, dst_id, detail::planes_for_layout(p.binding, p.range, separate),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, tracker);
                vk.copy_image(command_buffer, p.source.image, src_layout, p.binding.image, dst_layout,
                              std::uint32_t(p.image_copies.size()), p.image_copies.data());
            } else if (p.binding.image) {
                const VkImageLayout dst_layout = detail::copy_layout(
                    vk, command_buffer, p.binding, dst_id, detail::planes_for_layout(p.binding, p.range, separate),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, tracker);
                vk.copy_buffer_to_image(command_buffer, p.source.buffer, p.binding.image, dst_layout,
                                        std::uint32_t(p.buffer_image_copies.size()), p.buffer_image_copies.data());
            } else {
                const VkImageLayout src_layout = detail::copy_layout(
                    vk, command_buffer, p.source, src_id, detail::planes_for_layout(p.source, p.source_range, separate),
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, tracker);
                vk.copy_image_to_buffer(command_buffer, p.source.image, src_layout, p.binding.buffer,
                                        std::uint32_t(p.buffer_image_copies.size()), p.buffer_image_copies.data());
            }
        } else if (std::holds_alternative<ResolveSubresource>(command)) {
            // resolve_source and resolve_dest are the transfer layouts, as copy_source and copy_dest are.
            const auto [dst_id, src_id] = copy_ids(command);
            const VkImageLayout src_layout = detail::copy_layout(
                vk, command_buffer, p.source, src_id, detail::planes_for_layout(p.source, p.source_range, separate),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, tracker);
            const VkImageLayout dst_layout = detail::copy_layout(
                vk, command_buffer, p.binding, dst_id, detail::planes_for_layout(p.binding, p.range, separate),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, tracker);
            vk.resolve_image(command_buffer, p.source.image, src_layout, p.binding.image, dst_layout, 1, &p.resolve);
        }
    }
}

// prepare + emit for one list. A failed replay leaves the command buffer and the tracker untouched.
template<typename Resolve>
ReplayResult replay(const CommandList& list, VkCommandBuffer command_buffer, Resolve&& resolve,
                    const ReplayFunctions& vk, LayoutTracker* tracker = nullptr) {
    if (tracker && tracker->separate_depth_stencil_layouts() != vk.separate_depth_stencil_layouts)
        return ReplayResult::unsupported;
    PreparedList prepared;
    if (const auto result = prepare(list, resolve, vk, prepared); result != ReplayResult::success) return result;
    emit(prepared, command_buffer, vk, tracker);
    return ReplayResult::success;
}

} // namespace leonida
