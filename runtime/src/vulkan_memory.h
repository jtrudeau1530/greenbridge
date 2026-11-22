#pragma once

// Memory-type policy for the D3D12 layer that depends only on VkPhysicalDeviceMemoryProperties, so it can be checked
// without a GPU (d3d12_state_map_test) on the heap layouts real cards report.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <cstdint>

namespace leonida {

// Without Resizable BAR the CPU sees at most a 256 MiB window of VRAM: the DEVICE_LOCAL | HOST_VISIBLE type sits in a
// heap of that size or smaller (256 MiB on AMD and on RADV, 214 MiB on an RTX 3090 with driver 616.56). Anything
// larger is Resizable BAR (or Smart Access Memory), or a GPU whose memory is all device-local and host-visible.
inline constexpr VkDeviceSize kBarWindowBytes = VkDeviceSize(256) << 20;

// True when the device has DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT memory in a device-local heap larger than
// the 256 MiB BAR window (D3D-96): then CPU-written memory that the GPU reads, such as an upload heap, can live in
// VRAM without exhausting a small window that every such allocation would share.
inline bool resizable_bar(const VkPhysicalDeviceMemoryProperties& memory) {
    constexpr VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; ++i) {
        const VkMemoryType& type = memory.memoryTypes[i];
        if ((type.propertyFlags & wanted) != wanted || type.heapIndex >= memory.memoryHeapCount) continue;
        const VkMemoryHeap& heap = memory.memoryHeaps[type.heapIndex];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && heap.size > kBarWindowBytes) return true;
    }
    return false;
}

}  // namespace leonida
