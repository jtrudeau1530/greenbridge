#pragma once

#include "clear_commands.h"

#include <vulkan/vulkan.h>

#include <algorithm>

namespace leonida {

enum class ClearReplayResult { success, list_open, unresolved_target };

// The caller transitions the target subresources to TRANSFER_DST_OPTIMAL and keeps them alive
// until execution completes. The resolver maps stable resource IDs to transfer-destination
// images whose formats match the float or normalized views captured at recording time.
template<typename Resolve>
ClearReplayResult replay_clears(const ClearCommandList& list, VkCommandBuffer command_buffer,
                                Resolve resolve, PFN_vkCmdPipelineBarrier barrier,
                                PFN_vkCmdClearColorImage clear_color_image) {
    if (!list.closed()) return ClearReplayResult::list_open;
    const auto commands = list.commands();
    std::vector<VkImage> images;
    images.reserve(commands.size());
    for (const auto& command : commands) {
        const VkImage image = resolve(command.target.resource);
        if (image == VK_NULL_HANDLE) return ClearReplayResult::unresolved_target;
        images.push_back(image);
    }

    VkMemoryBarrier clear_order{};
    clear_order.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    clear_order.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    clear_order.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    for (std::size_t i = 0; i < commands.size(); ++i) {
        // Preserve write order, including when two descriptors alias the same subresource.
        barrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &clear_order, 0, nullptr, 0, nullptr);
        VkClearColorValue color{};
        std::copy(commands[i].color.begin(), commands[i].color.end(), color.float32);
        const auto& target = commands[i].target;
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, target.mip, 1,
                                            target.first_layer, target.layer_count};
        clear_color_image(command_buffer, images[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
    }
    return ClearReplayResult::success;
}

} // namespace leonida
