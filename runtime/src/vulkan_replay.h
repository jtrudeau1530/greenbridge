#pragma once

// Replays a closed CommandList (command_list.h) into a Vulkan command buffer.
//
// Resource states follow D3D12: a transition's `before` is trusted as the state the subresource is in when the
// list executes, as the D3D12 runtime does, and becomes the Vulkan oldLayout. Clears assume the states D3D12
// requires for them (render_target, depth_write) and wrap the transfer clear in a layout round trip, so the
// image is back in its D3D12 state afterwards.
//
// Requirements on the device and images:
// - a queue with graphics support (the stage masks use ALL_GRAPHICS);
// - images that can be cleared are created with TRANSFER_DST usage;
// - transitions of a single plane of a combined depth-stencil image need separateDepthStencilLayouts (Vulkan 1.2).
//   Clears move both planes, so a depth-only clear while the stencil plane is in another state is not supported.
// Implicit state promotion and decay (D3D12 "common" state rules) are not modelled.

#include "command_list.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace leonida {

// What the resolver knows about a resource. image == VK_NULL_HANDLE means a buffer in the shared GPU address
// space: its transitions become global memory barriers.
struct ResourceBinding {
    VkImage image = VK_NULL_HANDLE;
    VkImageAspectFlags aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    std::uint32_t mip_levels = 1;
    std::uint32_t array_layers = 1;
    bool presentable = false;  // a swapchain image: common/present maps to PRESENT_SRC_KHR
};

struct VulkanAccess {
    VkImageLayout layout;
    VkAccessFlags access;
    VkPipelineStageFlags stages;
};

inline constexpr VkPipelineStageFlags shader_stages =
    VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

// Maps a D3D12_RESOURCE_STATES combination to the Vulkan layout, accesses and stages that cover it.
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

// Decodes a D3D12 subresource index. Combined depth-stencil images have two planes (0 depth, 1 stencil);
// every other image has one.
inline bool image_range(const ResourceBinding& resource, std::uint32_t subresource, VkImageSubresourceRange& out) {
    if (!resource.mip_levels || !resource.array_layers) return false;
    if (subresource == all_subresources) {
        out = {resource.aspects, 0, resource.mip_levels, 0, resource.array_layers};
        return true;
    }
    constexpr VkImageAspectFlags depth_stencil = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    const bool two_planes = (resource.aspects & depth_stencil) == depth_stencil;
    const std::uint64_t per_plane = std::uint64_t(resource.mip_levels) * resource.array_layers;
    const std::uint64_t plane = subresource / per_plane;
    if (plane >= (two_planes ? 2u : 1u)) return false;
    const auto within = static_cast<std::uint32_t>(subresource % per_plane);
    const VkImageAspectFlags aspect =
        two_planes ? (plane ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT) : resource.aspects;
    out = {aspect, within % resource.mip_levels, 1, within / resource.mip_levels, 1};
    return true;
}

struct ReplayFunctions {
    PFN_vkCmdPipelineBarrier pipeline_barrier;
    PFN_vkCmdClearColorImage clear_color_image;
    PFN_vkCmdClearDepthStencilImage clear_depth_stencil_image;
};

enum class ReplayResult { success, list_open, unresolved_resource, invalid_subresource, invalid_clear };

namespace detail {

struct Prepared {
    ResourceBinding binding;
    VkImageSubresourceRange range;  // the transition range, or the cleared view
};

inline bool view_range(const ResourceBinding& resource, const TargetView& view, VkImageAspectFlags aspects,
                       VkImageSubresourceRange& out) {
    if (view.mip >= resource.mip_levels) return false;
    if (view.first_layer >= resource.array_layers || view.layer_count > resource.array_layers - view.first_layer)
        return false;
    out = {aspects, view.mip, 1, view.first_layer, view.layer_count};
    return true;
}

inline void image_barrier(const ReplayFunctions& vk, VkCommandBuffer command_buffer, VkImage image,
                          const VkImageSubresourceRange& range, const VulkanAccess& from, const VulkanAccess& to) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = from.access;
    barrier.dstAccessMask = to.access;
    barrier.oldLayout = from.layout;
    barrier.newLayout = to.layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = range;
    vk.pipeline_barrier(command_buffer, from.stages, to.stages, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

inline void memory_barrier(const ReplayFunctions& vk, VkCommandBuffer command_buffer, const VulkanAccess& from,
                           const VulkanAccess& to) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = from.access;
    barrier.dstAccessMask = to.access;
    vk.pipeline_barrier(command_buffer, from.stages, to.stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

} // namespace detail

// Resolve maps a ResourceId to a const ResourceBinding* (nullptr: unknown). Every command is resolved and checked
// before anything is written, so a failed replay leaves the command buffer untouched.
template<typename Resolve>
ReplayResult replay(const CommandList& list, VkCommandBuffer command_buffer, Resolve resolve,
                    const ReplayFunctions& vk) {
    if (!list.closed()) return ReplayResult::list_open;
    const auto commands = list.commands();
    std::vector<detail::Prepared> prepared(commands.size());
    constexpr VkImageAspectFlags depth_stencil = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;

    for (std::size_t i = 0; i < commands.size(); ++i) {
        const auto& command = commands[i];
        auto& out = prepared[i];
        const auto bind = [&](ResourceId id) {
            const ResourceBinding* binding = resolve(id);
            if (binding) out.binding = *binding;
            return binding != nullptr;
        };
        if (const auto* barrier = std::get_if<TransitionBarrier>(&command)) {
            if (!bind(barrier->resource)) return ReplayResult::unresolved_resource;
            if (out.binding.image == VK_NULL_HANDLE) {
                if (barrier->subresource != 0 && barrier->subresource != all_subresources)
                    return ReplayResult::invalid_subresource;
            } else if (!image_range(out.binding, barrier->subresource, out.range)) {
                return ReplayResult::invalid_subresource;
            }
        } else if (const auto* clear = std::get_if<ClearRenderTarget>(&command)) {
            if (!bind(clear->target.resource)) return ReplayResult::unresolved_resource;
            if (!out.binding.image || out.binding.aspects != VK_IMAGE_ASPECT_COLOR_BIT)
                return ReplayResult::invalid_clear;
            if (!detail::view_range(out.binding, clear->target, VK_IMAGE_ASPECT_COLOR_BIT, out.range))
                return ReplayResult::invalid_subresource;
        } else if (const auto* clear = std::get_if<ClearDepthStencil>(&command)) {
            if (!bind(clear->target.resource)) return ReplayResult::unresolved_resource;
            VkImageAspectFlags aspects = 0;
            if (clear->flags & clear_depth) aspects |= VK_IMAGE_ASPECT_DEPTH_BIT;
            if (clear->flags & clear_stencil) aspects |= VK_IMAGE_ASPECT_STENCIL_BIT;
            const VkImageAspectFlags available = out.binding.aspects & depth_stencil;
            if (!out.binding.image || !available || (aspects & ~available) ||
                (out.binding.aspects & ~depth_stencil))
                return ReplayResult::invalid_clear;
            if (!detail::view_range(out.binding, clear->target, aspects, out.range))
                return ReplayResult::invalid_subresource;
        }
    }

    const VulkanAccess transfer_dst = vulkan_access(d3d12_state::copy_dest, false);
    for (std::size_t i = 0; i < commands.size(); ++i) {
        const auto& command = commands[i];
        const auto& p = prepared[i];
        if (const auto* barrier = std::get_if<TransitionBarrier>(&command)) {
            const VulkanAccess from = vulkan_access(barrier->before, p.binding.presentable);
            const VulkanAccess to = vulkan_access(barrier->after, p.binding.presentable);
            if (p.binding.image)
                detail::image_barrier(vk, command_buffer, p.binding.image, p.range, from, to);
            else
                detail::memory_barrier(vk, command_buffer, from, to);
        } else if (std::holds_alternative<UavBarrier>(command)) {
            const VulkanAccess uav = vulkan_access(d3d12_state::unordered_access, false);
            detail::memory_barrier(vk, command_buffer, {uav.layout, VK_ACCESS_SHADER_WRITE_BIT, uav.stages}, uav);
        } else if (std::holds_alternative<AliasingBarrier>(command)) {
            const VulkanAccess any = vulkan_access(~0u, false);
            detail::memory_barrier(vk, command_buffer, {any.layout, VK_ACCESS_MEMORY_WRITE_BIT, any.stages}, any);
        } else if (const auto* clear = std::get_if<ClearRenderTarget>(&command)) {
            const VulkanAccess target = vulkan_access(d3d12_state::render_target, false);
            detail::image_barrier(vk, command_buffer, p.binding.image, p.range, target, transfer_dst);
            VkClearColorValue color{};
            std::copy(clear->color.begin(), clear->color.end(), color.float32);
            vk.clear_color_image(command_buffer, p.binding.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1,
                                 &p.range);
            detail::image_barrier(vk, command_buffer, p.binding.image, p.range, transfer_dst, target);
        } else if (const auto* clear = std::get_if<ClearDepthStencil>(&command)) {
            // The layout is shared by both planes unless separateDepthStencilLayouts is used, so the round trip
            // moves every plane the image has; only the requested aspects are cleared.
            VkImageSubresourceRange planes = p.range;
            planes.aspectMask = p.binding.aspects;
            const VulkanAccess target = vulkan_access(d3d12_state::depth_write, false);
            detail::image_barrier(vk, command_buffer, p.binding.image, planes, target, transfer_dst);
            const VkClearDepthStencilValue value{clear->depth, clear->stencil};
            vk.clear_depth_stencil_image(command_buffer, p.binding.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                         &value, 1, &p.range);
            detail::image_barrier(vk, command_buffer, p.binding.image, planes, transfer_dst, target);
        }
    }
    return ReplayResult::success;
}

} // namespace leonida
