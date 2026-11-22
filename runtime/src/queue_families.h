#pragma once

// Picking Vulkan queue families for D3D12's direct, compute and copy queues. Header-only and free of the rest of
// the backend, so test harnesses can use it too.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <span>

namespace leonida {

// Queue families for D3D12's three queue types. Direct needs graphics. Compute prefers a family with compute and no
// graphics (async compute), copy one with transfer and neither (a DMA engine); each falls back to a more capable
// family, ending at the graphics one. A dedicated copy family is only used when its image transfer granularity is
// 1x1x1, because D3D12 copies any box.
struct QueueFamilies {
    std::uint32_t direct = 0;
    std::uint32_t compute = 0;
    std::uint32_t copy = 0;
};

inline QueueFamilies pick_queue_families(std::span<const VkQueueFamilyProperties> families, std::uint32_t graphics) {
    QueueFamilies out{graphics, graphics, graphics};
    constexpr VkQueueFlags g = VK_QUEUE_GRAPHICS_BIT, c = VK_QUEUE_COMPUTE_BIT, t = VK_QUEUE_TRANSFER_BIT;
    const auto fine_grained = [](const VkQueueFamilyProperties& f) {
        return f.minImageTransferGranularity.width == 1 && f.minImageTransferGranularity.height == 1 &&
               f.minImageTransferGranularity.depth == 1;
    };
    bool have_compute = false, have_copy = false;
    for (std::uint32_t i = 0; i < families.size(); ++i) {
        const auto& f = families[i];
        if (!f.queueCount) continue;
        if (!have_compute && (f.queueFlags & c) && !(f.queueFlags & g)) {
            out.compute = i;
            have_compute = true;
        }
        if (!have_copy && (f.queueFlags & t) && !(f.queueFlags & (g | c)) && fine_grained(f)) {
            out.copy = i;
            have_copy = true;
        }
    }
    // No DMA family: copies go to the async-compute family, which can transfer too (compute implies transfer).
    if (!have_copy && have_compute && fine_grained(families[out.compute])) out.copy = out.compute;
    return out;
}

}  // namespace leonida
