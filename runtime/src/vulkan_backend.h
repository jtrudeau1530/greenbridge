#pragma once

// The execution side of the D3D12 layer, without COM: the per-device Vulkan context (resources, layout tracker,
// image views), queues that execute command lists, and timeline-semaphore fences. The COM objects in
// shims/d3d12_com.cpp are thin wrappers over these. Needs Vulkan 1.2 timeline semaphores; rectangle clears and
// enhanced barriers need the Vulkan 1.3 functions (dynamicRendering, synchronization2) to be enabled.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <span>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace leonida {

struct VulkanDeviceFunctions {
#define LEONIDA_VK_DEVICE_FUNCTIONS(X)                                                                         \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkResetCommandBuffer)         \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkCreateSemaphore)     \
    X(vkDestroySemaphore) X(vkGetSemaphoreCounterValue) X(vkWaitSemaphores) X(vkSignalSemaphore)               \
    X(vkCreateImageView) X(vkDestroyImageView) X(vkCmdPipelineBarrier) X(vkCmdPipelineBarrier2)                \
    X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage) X(vkCmdBeginRendering) X(vkCmdEndRendering)         \
    X(vkCmdClearAttachments) X(vkDeviceWaitIdle) X(vkCreateImage) X(vkDestroyImage)                         \
    X(vkGetImageMemoryRequirements) X(vkBindImageMemory) X(vkCreateBuffer) X(vkDestroyBuffer)                  \
    X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory)  \
    X(vkUnmapMemory) X(vkCmdCopyBuffer) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdCopyImage)
#define LEONIDA_VK_MEMBER(name) PFN_##name name = nullptr;
    LEONIDA_VK_DEVICE_FUNCTIONS(LEONIDA_VK_MEMBER)
#undef LEONIDA_VK_MEMBER

    // The Vulkan 1.3 entry points may be null on older devices; everything else is required.
    bool load(PFN_vkGetDeviceProcAddr get, VkDevice device) {
#define LEONIDA_VK_LOAD(name) name = reinterpret_cast<PFN_##name>(get(device, #name));
        LEONIDA_VK_DEVICE_FUNCTIONS(LEONIDA_VK_LOAD)
#undef LEONIDA_VK_LOAD
        return vkCreateCommandPool && vkDestroyCommandPool && vkAllocateCommandBuffers && vkResetCommandBuffer &&
               vkBeginCommandBuffer && vkEndCommandBuffer && vkQueueSubmit && vkQueueWaitIdle && vkCreateSemaphore &&
               vkDestroySemaphore && vkGetSemaphoreCounterValue && vkWaitSemaphores && vkSignalSemaphore &&
               vkCreateImageView && vkDestroyImageView && vkCmdPipelineBarrier && vkCmdClearColorImage &&
               vkCmdClearDepthStencilImage && vkCmdClearAttachments && vkDeviceWaitIdle && vkCreateImage &&
               vkDestroyImage && vkGetImageMemoryRequirements && vkBindImageMemory && vkCreateBuffer &&
               vkDestroyBuffer && vkGetBufferMemoryRequirements && vkBindBufferMemory && vkAllocateMemory &&
               vkFreeMemory && vkMapMemory && vkUnmapMemory && vkCmdCopyBuffer && vkCmdCopyBufferToImage &&
               vkCmdCopyImageToBuffer && vkCmdCopyImage;
    }
};

// Per device: what the queues share. Resources are registered by their stable ID (the COM resource pointer).
class VulkanContext {
public:
    VulkanContext(PFN_vkGetDeviceProcAddr get, VkDevice device, bool separate_depth_stencil_layouts)
        : device_(device), tracker_(separate_depth_stencil_layouts) {
        loaded_ = vk.load(get, device);
        replay_.pipeline_barrier = vk.vkCmdPipelineBarrier;
        replay_.clear_color_image = vk.vkCmdClearColorImage;
        replay_.clear_depth_stencil_image = vk.vkCmdClearDepthStencilImage;
        replay_.pipeline_barrier2 = vk.vkCmdPipelineBarrier2;
        replay_.begin_rendering = vk.vkCmdBeginRendering;
        replay_.end_rendering = vk.vkCmdEndRendering;
        replay_.clear_attachments = vk.vkCmdClearAttachments;
        replay_.copy_buffer = vk.vkCmdCopyBuffer;
        replay_.copy_buffer_to_image = vk.vkCmdCopyBufferToImage;
        replay_.copy_image_to_buffer = vk.vkCmdCopyImageToBuffer;
        replay_.copy_image = vk.vkCmdCopyImage;
        replay_.image_view = [this](const ResourceBinding& r, const VkImageSubresourceRange& range) {
            return view(r, range);
        };
        replay_.separate_depth_stencil_layouts = separate_depth_stencil_layouts;
    }

    ~VulkanContext() {
        for (const auto& [key, view] : views_) vk.vkDestroyImageView(device_, view, nullptr);
    }

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    bool loaded() const { return loaded_; }
    VkDevice device() const { return device_; }

    // Registers a resource in its D3D12 initial state (a legacy state; images get the matching layout).
    // The image must already be in that layout, e.g. transitioned from UNDEFINED at creation.
    void add_resource(ResourceId id, const ResourceBinding& binding) {
        std::lock_guard lock(mutex_);
        resources_[id] = binding;
    }
    void add_resource(ResourceId id, const ResourceBinding& binding, VkImageLayout current_layout) {
        std::lock_guard lock(mutex_);
        resources_[id] = binding;
        if (binding.image) tracker_.track(id, binding, current_layout);
    }

    void remove_resource(ResourceId id) {
        std::lock_guard lock(mutex_);
        const auto it = resources_.find(id);
        if (it == resources_.end()) return;
        for (auto v = views_.begin(); v != views_.end();) {
            if (std::get<0>(v->first) == it->second.image) {
                vk.vkDestroyImageView(device_, v->second, nullptr);
                v = views_.erase(v);
            } else {
                ++v;
            }
        }
        tracker_.forget(id);
        resources_.erase(it);
    }

    // For tests and traces.
    VkImageLayout layout(ResourceId id, std::uint32_t subresource) {
        std::lock_guard lock(mutex_);
        return tracker_.layout(id, subresource);
    }

    VulkanDeviceFunctions vk;

    // Vulkan requires external synchronisation of a VkQueue; every D3D12 queue of the device may share one.
    std::mutex& submit_mutex() { return submit_mutex_; }

private:
    friend class VulkanQueue;

    const ResourceBinding* resolve(ResourceId id) const {
        const auto it = resources_.find(id);
        return it == resources_.end() ? nullptr : &it->second;
    }

    // Called with mutex_ held (from prepare). Views live until their image is removed.
    VkImageView view(const ResourceBinding& r, const VkImageSubresourceRange& range) {
        const auto key = std::make_tuple(r.image, range.aspectMask, range.baseMipLevel, range.baseArrayLayer,
                                         range.layerCount);
        if (const auto it = views_.find(key); it != views_.end()) return it->second;
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = r.image;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        info.format = r.format;
        info.subresourceRange = range;
        VkImageView view = VK_NULL_HANDLE;
        if (vk.vkCreateImageView(device_, &info, nullptr, &view) != VK_SUCCESS) return VK_NULL_HANDLE;
        views_[key] = view;
        return view;
    }

    VkDevice device_;
    bool loaded_ = false;
    std::mutex mutex_;  // resources, tracker and views; held while a queue prepares and emits
    std::mutex submit_mutex_;
    std::unordered_map<ResourceId, ResourceBinding> resources_;
    LayoutTracker tracker_;
    std::map<std::tuple<VkImage, VkImageAspectFlags, std::uint32_t, std::uint32_t, std::uint32_t>, VkImageView> views_;
    ReplayFunctions replay_;
};

// A Vulkan timeline semaphore with D3D12 fence semantics for the host side.
class VulkanFence {
public:
    VulkanFence(VulkanContext& context, std::uint64_t initial) : context_(context) {
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        type.initialValue = initial;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        if (context.vk.vkCreateSemaphore(context.device(), &info, nullptr, &semaphore_) != VK_SUCCESS)
            semaphore_ = VK_NULL_HANDLE;
    }
    ~VulkanFence() {
        if (semaphore_) context_.vk.vkDestroySemaphore(context_.device(), semaphore_, nullptr);
    }
    VulkanFence(const VulkanFence&) = delete;
    VulkanFence& operator=(const VulkanFence&) = delete;

    VkSemaphore semaphore() const { return semaphore_; }

    std::uint64_t completed() const {
        std::uint64_t value = 0;
        context_.vk.vkGetSemaphoreCounterValue(context_.device(), semaphore_, &value);
        return value;
    }

    // ID3D12Fence::Signal from the CPU. Vulkan timelines only increase: a value at or below the current one is
    // ignored (D3D12 would set it), and the caller is told.
    bool signal(std::uint64_t value) {
        if (value <= completed()) return false;
        VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        info.semaphore = semaphore_;
        info.value = value;
        return context_.vk.vkSignalSemaphore(context_.device(), &info) == VK_SUCCESS;
    }

    // true when the value was reached within the timeout.
    bool wait(std::uint64_t value, std::uint64_t timeout_ns) const {
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &semaphore_;
        info.pValues = &value;
        return context_.vk.vkWaitSemaphores(context_.device(), &info, timeout_ns) == VK_SUCCESS;
    }

private:
    VulkanContext& context_;
    VkSemaphore semaphore_ = VK_NULL_HANDLE;
};

// One D3D12 command queue on one VkQueue. Not thread-safe by itself: D3D12 lets one thread use a queue at a time
// per call, and the COM wrapper serialises its calls.
class VulkanQueue {
public:
    VulkanQueue(VulkanContext& context, VkQueue queue, std::uint32_t family) : context_(context), queue_(queue) {
        auto& vk = context.vk;
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = family;
        ok_ = vk.vkCreateCommandPool(context.device(), &pool, nullptr, &pool_) == VK_SUCCESS;
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        ok_ = ok_ && vk.vkCreateSemaphore(context.device(), &info, nullptr, &progress_) == VK_SUCCESS;
    }

    ~VulkanQueue() {
        auto& vk = context_.vk;
        if (queue_) {
            std::lock_guard lock(context_.submit_mutex());
            vk.vkQueueWaitIdle(queue_);
        }
        if (pool_) vk.vkDestroyCommandPool(context_.device(), pool_, nullptr);
        if (progress_) vk.vkDestroySemaphore(context_.device(), progress_, nullptr);
    }

    VulkanQueue(const VulkanQueue&) = delete;
    VulkanQueue& operator=(const VulkanQueue&) = delete;

    bool ok() const { return ok_; }

    // ExecuteCommandLists: every list is checked before any is written, so a failure submits nothing and leaves
    // the layout tracker as it was. Pending Wait()s are attached to this submission.
    ReplayResult execute(std::span<const CommandList* const> lists) {
        std::lock_guard lock(context_.mutex_);
        std::vector<PreparedList> prepared(lists.size());
        const auto resolve = [this](ResourceId id) { return context_.resolve(id); };
        for (std::size_t i = 0; i < lists.size(); ++i) {
            const auto result = prepare(*lists[i], resolve, context_.replay_, prepared[i]);
            if (result != ReplayResult::success) return result;
        }
        VkCommandBuffer command_buffer = acquire();
        if (!command_buffer) return ReplayResult::unsupported;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        context_.vk.vkBeginCommandBuffer(command_buffer, &begin);
        for (const auto& list : prepared) emit(list, command_buffer, context_.replay_, &context_.tracker_);
        // A fence wait on the CPU makes GPU writes visible to the host only after a barrier to the host domain:
        // readback buffers are read through Map once the application has waited.
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        context_.vk.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                         0, 1, &host, 0, nullptr, 0, nullptr);
        context_.vk.vkEndCommandBuffer(command_buffer);
        return submit(command_buffer, VK_NULL_HANDLE, 0) ? ReplayResult::success : ReplayResult::unsupported;
    }

    // ID3D12CommandQueue::Signal: the fence reaches `value` after all work submitted so far.
    bool signal(VkSemaphore fence, std::uint64_t value) { return submit(VK_NULL_HANDLE, fence, value); }

    // ID3D12CommandQueue::Wait: later submissions wait for the fence. A wait in a Vulkan batch only holds back that
    // batch, so it is kept and attached to the next submission.
    void wait(VkSemaphore fence, std::uint64_t value) {
        waits_.push_back(fence);
        wait_values_.push_back(value);
    }

    bool idle() {
        std::lock_guard lock(context_.submit_mutex());
        return context_.vk.vkQueueWaitIdle(queue_) == VK_SUCCESS;
    }

private:
    VkCommandBuffer acquire() {
        std::uint64_t done = 0;
        context_.vk.vkGetSemaphoreCounterValue(context_.device(), progress_, &done);
        for (auto& slot : buffers_) {
            if (slot.second > done) continue;
            context_.vk.vkResetCommandBuffer(slot.first, 0);
            slot.second = UINT64_MAX;  // in use until submitted
            return slot.first;
        }
        VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        info.commandPool = pool_;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        if (context_.vk.vkAllocateCommandBuffers(context_.device(), &info, &command_buffer) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        buffers_.push_back({command_buffer, UINT64_MAX});
        return command_buffer;
    }

    bool submit(VkCommandBuffer command_buffer, VkSemaphore fence, std::uint64_t value) {
        const std::uint64_t ticket = ++submitted_;
        std::vector<VkSemaphore> signals{progress_};
        std::vector<std::uint64_t> signal_values{ticket};
        if (fence) {
            signals.push_back(fence);
            signal_values.push_back(value);
        }
        const std::vector<VkPipelineStageFlags> stages(waits_.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        timeline.waitSemaphoreValueCount = std::uint32_t(wait_values_.size());
        timeline.pWaitSemaphoreValues = wait_values_.data();
        timeline.signalSemaphoreValueCount = std::uint32_t(signal_values.size());
        timeline.pSignalSemaphoreValues = signal_values.data();
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, &timeline};
        submit.waitSemaphoreCount = std::uint32_t(waits_.size());
        submit.pWaitSemaphores = waits_.data();
        submit.pWaitDstStageMask = stages.data();
        submit.commandBufferCount = command_buffer ? 1 : 0;
        submit.pCommandBuffers = &command_buffer;
        submit.signalSemaphoreCount = std::uint32_t(signals.size());
        submit.pSignalSemaphores = signals.data();
        bool ok;
        {
            std::lock_guard lock(context_.submit_mutex());
            ok = context_.vk.vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
        }
        waits_.clear();
        wait_values_.clear();
        for (auto& slot : buffers_)
            if (slot.first == command_buffer) slot.second = ticket;
        return ok;
    }

    VulkanContext& context_;
    VkQueue queue_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkSemaphore progress_ = VK_NULL_HANDLE;  // our own timeline: which command buffers are done
    std::uint64_t submitted_ = 0;
    std::vector<std::pair<VkCommandBuffer, std::uint64_t>> buffers_;  // buffer, ticket that frees it
    std::vector<VkSemaphore> waits_;
    std::vector<std::uint64_t> wait_values_;
    bool ok_ = false;
};

} // namespace leonida
