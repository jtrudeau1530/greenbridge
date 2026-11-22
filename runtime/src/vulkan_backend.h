#pragma once

// The execution side of the D3D12 layer, without COM: the per-device Vulkan context (resources, layout tracker,
// image views), queues that execute command lists, and timeline-semaphore fences. The COM objects in
// shims/d3d12_com.cpp are thin wrappers over these. Needs Vulkan 1.2 timeline semaphores; rectangle clears and
// enhanced barriers need the Vulkan 1.3 functions (dynamicRendering, synchronization2) to be enabled.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "binary_occlusion_spirv.h"
#include "vulkan_depth24.h"
#include "vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace leonida {

// VK_EXT_debug_utils labels: an instance extension, so a device has them only when its instance enabled it.
struct DebugLabelFunctions {
    PFN_vkCmdInsertDebugUtilsLabelEXT cmd_insert = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT cmd_begin = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT cmd_end = nullptr;
    PFN_vkQueueInsertDebugUtilsLabelEXT queue_insert = nullptr;
    PFN_vkQueueBeginDebugUtilsLabelEXT queue_begin = nullptr;
    PFN_vkQueueEndDebugUtilsLabelEXT queue_end = nullptr;

    bool complete() const { return cmd_insert && cmd_begin && cmd_end && queue_insert && queue_begin && queue_end; }
};

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
    X(vkUnmapMemory) X(vkCmdCopyBuffer) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdCopyImage)  \
    X(vkCmdResolveImage) X(vkCmdFillBuffer) X(vkCmdUpdateBuffer) X(vkCreateSampler) X(vkDestroySampler)     \
    X(vkCmdBeginConditionalRenderingEXT) X(vkCmdEndConditionalRenderingEXT) X(vkCreateQueryPool)            \
    X(vkDestroyQueryPool) X(vkCmdResetQueryPool) X(vkCmdBeginQuery) X(vkCmdEndQuery) X(vkCmdWriteTimestamp) \
    X(vkCmdCopyQueryPoolResults) X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout)              \
    X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) X(vkCreateShaderModule) X(vkDestroyShaderModule)    \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets)          \
    X(vkCmdPushConstants) X(vkCmdDispatch)
#define LEONIDA_VK_MEMBER(name) PFN_##name name = nullptr;
    LEONIDA_VK_DEVICE_FUNCTIONS(LEONIDA_VK_MEMBER)
#undef LEONIDA_VK_MEMBER

    // The Vulkan 1.3 entry points may be null on older devices, the extension ones (conditional rendering) when the
    // extension is not enabled, and the query functions are optional (a list that uses queries without them is
    // unsupported). The compute and descriptor functions are needed only for binary occlusion resolves.
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
               vkCmdCopyImageToBuffer && vkCmdCopyImage && vkCmdResolveImage && vkCmdFillBuffer &&
               vkCmdUpdateBuffer;
    }
};

// The scratch buffer belongs to one queue. Each chunk copies raw results into it, normalizes them in place,
// then transfers UINT64s to the application buffer, which needs no storage-buffer usage.
class BinaryOcclusionResolve {
public:
    BinaryOcclusionResolve(const VulkanDeviceFunctions& vk, VkDevice device) : vk_(vk), device_(device) {}
    BinaryOcclusionResolve(const BinaryOcclusionResolve&) = delete;
    BinaryOcclusionResolve& operator=(const BinaryOcclusionResolve&) = delete;

    ~BinaryOcclusionResolve() {
        if (pipeline_) vk_.vkDestroyPipeline(device_, pipeline_, nullptr);
        if (layout_) vk_.vkDestroyPipelineLayout(device_, layout_, nullptr);
        if (pool_) vk_.vkDestroyDescriptorPool(device_, pool_, nullptr);
        if (set_layout_) vk_.vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
        if (scratch_) vk_.vkDestroyBuffer(device_, scratch_, nullptr);
        if (memory_) vk_.vkFreeMemory(device_, memory_, nullptr);
    }

    bool initialize(const VkPhysicalDeviceMemoryProperties& memory) {
        if (!vk_.vkCreateDescriptorSetLayout || !vk_.vkDestroyDescriptorSetLayout ||
            !vk_.vkCreateDescriptorPool || !vk_.vkDestroyDescriptorPool || !vk_.vkAllocateDescriptorSets ||
            !vk_.vkUpdateDescriptorSets || !vk_.vkCreatePipelineLayout || !vk_.vkDestroyPipelineLayout ||
            !vk_.vkCreateShaderModule || !vk_.vkDestroyShaderModule || !vk_.vkCreateComputePipelines ||
            !vk_.vkDestroyPipeline || !vk_.vkCmdBindPipeline || !vk_.vkCmdBindDescriptorSets ||
            !vk_.vkCmdPushConstants || !vk_.vkCmdDispatch)
            return false;

        VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer.size = scratch_size;
        buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vk_.vkCreateBuffer(device_, &buffer, nullptr, &scratch_) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vk_.vkGetBufferMemoryRequirements(device_, scratch_, &requirements);
        auto type = memory.memoryTypeCount;
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if (!(requirements.memoryTypeBits & (1u << i))) continue;
            if (type == memory.memoryTypeCount) type = i;
            if (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                type = i;
                break;
            }
        }
        if (type == memory.memoryTypeCount) return false;
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        if (vk_.vkAllocateMemory(device_, &allocation, nullptr, &memory_) != VK_SUCCESS ||
            vk_.vkBindBufferMemory(device_, scratch_, memory_, 0) != VK_SUCCESS)
            return false;

        const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo set_layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_layout.bindingCount = 1;
        set_layout.pBindings = &binding;
        if (vk_.vkCreateDescriptorSetLayout(device_, &set_layout, nullptr, &set_layout_) != VK_SUCCESS) return false;
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = 1;
        pool.poolSizeCount = 1;
        pool.pPoolSizes = &size;
        if (vk_.vkCreateDescriptorPool(device_, &pool, nullptr, &pool_) != VK_SUCCESS) return false;
        VkDescriptorSetAllocateInfo set{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set.descriptorPool = pool_;
        set.descriptorSetCount = 1;
        set.pSetLayouts = &set_layout_;
        if (vk_.vkAllocateDescriptorSets(device_, &set, &set_) != VK_SUCCESS) return false;
        const VkDescriptorBufferInfo storage{scratch_, 0, scratch_size};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set_;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &storage;
        vk_.vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);

        const VkPushConstantRange count{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(std::uint32_t)};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &set_layout_;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &count;
        if (vk_.vkCreatePipelineLayout(device_, &layout, nullptr, &layout_) != VK_SUCCESS) return false;
        VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module.codeSize = sizeof(binary_occlusion_spirv);
        module.pCode = binary_occlusion_spirv;
        VkShaderModule shader = VK_NULL_HANDLE;
        if (vk_.vkCreateShaderModule(device_, &module, nullptr, &shader) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                          VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", nullptr};
        pipeline.layout = layout_;
        const VkResult result = vk_.vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &pipeline_);
        vk_.vkDestroyShaderModule(device_, shader, nullptr);
        return result == VK_SUCCESS;
    }

    void record(VkCommandBuffer cb, VkQueryPool queries, std::uint32_t first, std::uint32_t count,
                VkBuffer dst, VkDeviceSize offset) const {
        constexpr auto transfer = VK_PIPELINE_STAGE_TRANSFER_BIT;
        constexpr auto compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        vk_.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vk_.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &set_, 0, nullptr);
        while (count) {
            const std::uint32_t chunk = std::min(count, scratch_count);
            const VkDeviceSize bytes = VkDeviceSize(chunk) * result_size;
            // Reuse is ordered even across submissions on this queue. Each previous copy has read the scratch
            // words before the query pool can write them again.
            scratch_barrier(cb, transfer, transfer, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            vk_.vkCmdCopyQueryPoolResults(cb, queries, first, chunk, scratch_, 0, result_size,
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            scratch_barrier(cb, transfer, compute, VK_ACCESS_TRANSFER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            vk_.vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(chunk), &chunk);
            vk_.vkCmdDispatch(cb, (chunk + group_size - 1) / group_size, 1, 1);
            scratch_barrier(cb, compute, transfer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            const VkBufferCopy copy{0, offset, bytes};
            vk_.vkCmdCopyBuffer(cb, scratch_, dst, 1, &copy);
            first += chunk;
            count -= chunk;
            offset += bytes;
        }
    }

private:
    void scratch_barrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                         VkAccessFlags src_access, VkAccessFlags dst_access) const {
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = src_access;
        barrier.dstAccessMask = dst_access;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = scratch_;
        barrier.size = scratch_size;
        vk_.vkCmdPipelineBarrier(cb, src, dst, 0, 0, nullptr, 1, &barrier, 0, nullptr);
    }

    static constexpr std::uint32_t scratch_count = 1024;
    static constexpr std::uint32_t group_size = 64;  // local_size_x in binary_occlusion.comp
    static constexpr VkDeviceSize result_size = sizeof(std::uint64_t);
    static constexpr VkDeviceSize scratch_size = scratch_count * result_size;
    const VulkanDeviceFunctions& vk_;
    VkDevice device_;
    VkBuffer scratch_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

// Per device: what the queues share. Resources are registered by their stable ID (the COM resource pointer).
class VulkanContext {
public:
    // conditional_rendering: VK_EXT_conditional_rendering is enabled on `device` (SetPredication).
    VulkanContext(PFN_vkGetDeviceProcAddr get, VkDevice device, bool separate_depth_stencil_layouts,
                  bool conditional_rendering = false)
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
        replay_.resolve_image = vk.vkCmdResolveImage;
        replay_.fill_buffer = vk.vkCmdFillBuffer;
        replay_.update_buffer = vk.vkCmdUpdateBuffer;
        // Called with mutex_ held. A linear search: fine for the first frames, an interval map when it shows.
        replay_.buffer_at = [this](std::uint64_t va, std::uint64_t size, VkBuffer& buffer, VkDeviceSize& offset) {
            for (const auto& [id, b] : resources_) {
                if (b.image || !b.buffer || !b.va || va < b.va || va - b.va >= b.size || size > b.size - (va - b.va))
                    continue;
                buffer = b.buffer;
                offset = b.buffer_offset + (va - b.va);
                return true;
            }
            return false;
        };
        if (conditional_rendering) {  // a driver may hand out these pointers even when the extension is off
            replay_.begin_conditional_rendering = vk.vkCmdBeginConditionalRenderingEXT;
            replay_.end_conditional_rendering = vk.vkCmdEndConditionalRenderingEXT;
        }
        replay_.reset_query_pool = vk.vkCmdResetQueryPool;
        replay_.begin_query = vk.vkCmdBeginQuery;
        replay_.end_query = vk.vkCmdEndQuery;
        replay_.write_timestamp = vk.vkCmdWriteTimestamp;
        replay_.copy_query_pool_results = vk.vkCmdCopyQueryPoolResults;
        replay_.image_view = [this](const ResourceBinding& r, const VkImageSubresourceRange& range) {
            return view(r, range);
        };
        replay_.query_pool = [this](QueryHeapId id) -> QueryPool* {  // called with mutex_ held
            const auto it = query_pools_.find(id);
            return it == query_pools_.end() ? nullptr : &it->second;
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

    // Debug markers become labels from here on; call before any queue or list uses the context. Incomplete sets
    // are ignored, so markers stay no-ops.
    void enable_debug_labels(const DebugLabelFunctions& labels) {
        if (!labels.complete()) return;
        labels_ = labels;
        replay_.insert_label = labels.cmd_insert;
        replay_.begin_label = labels.cmd_begin;
        replay_.end_label = labels.cmd_end;
    }
    bool debug_labels() const { return labels_.complete(); }

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

    // A query heap, by its stable ID (the COM query heap pointer). The pool stays the caller's to destroy, after
    // remove_query_pool.
    void add_query_pool(QueryHeapId id, const QueryPool& pool) {
        std::lock_guard lock(mutex_);
        query_pools_[id] = pool;
    }
    void remove_query_pool(QueryHeapId id) {
        std::lock_guard lock(mutex_);
        query_pools_.erase(id);
    }

    // VkPhysicalDeviceLimits::timestampPeriod (nanoseconds per tick), or 0 when the queue writes no timestamps
    // (timestampValidBits 0). Set by the device before any queue uses it.
    double timestamp_period() const { return timestamp_period_; }
    void set_timestamp_period(double period) { timestamp_period_ = period; }

    // Configure before queue creation, when the selected queue family supports compute.
    void set_binary_occlusion_memory(const VkPhysicalDeviceMemoryProperties& memory) {
        binary_occlusion_memory_ = memory;
    }

    // For tests and traces.
    VkImageLayout layout(ResourceId id, std::uint32_t subresource) {
        std::lock_guard lock(mutex_);
        return tracker_.layout(id, subresource);
    }

    bool has_resource(ResourceId id) {
        std::lock_guard lock(mutex_);
        return resources_.find(id) != resources_.end();
    }

    // `prepare` is called with the context locked, at the start of the command buffer of every
    // ExecuteCommandLists, before the lists are written: the device brings the bindless set 0 of the translated
    // shaders up to date there (vulkan_bindless.h) and records what the set needs first. When it fails, nothing
    // of that call is executed.
    void set_bindless_prepare(std::function<bool(VkCommandBuffer)> prepare) {
        std::lock_guard lock(mutex_);
        bindless_prepare_ = std::move(prepare);
    }

    // D24_UNORM_S8_UINT stored as D32: copies of its depth plane run a compute pass (vulkan_depth24.h). `allocate`
    // gives the scratch buffers device-local memory. Without this such a copy is refused as unsupported.
    void enable_depth24(PFN_vkGetDeviceProcAddr get, Depth24Converter::Allocate allocate,
                       std::uint64_t max_storage_range = 1u << 27) {
        std::lock_guard lock(mutex_);
        depth24_ = std::make_unique<Depth24Converter>(get, device_, std::move(allocate), max_storage_range);
        if (!depth24_->loaded()) {
            depth24_.reset();
            return;
        }
        replay_.depth24_prepare = [this](Depth24Job& job, std::vector<Depth24Job>& parts) { return depth24_->prepare(job, parts); };
        replay_.depth24_run = [this](VkCommandBuffer command_buffer, const Depth24Job& job) {
            depth24_->run(command_buffer, job);
        };
    }

    // Called after that command buffer's submit: true when it was accepted, false when vkQueueSubmit failed. The
    // set keeps its upload pending in the second case, because the copy never reached the GPU.
    void set_bindless_commit(std::function<void(bool)> commit) {
        std::lock_guard lock(mutex_);
        bindless_commit_ = std::move(commit);
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
                                         range.layerCount, r.format);
        if (const auto it = views_.find(key); it != views_.end()) return it->second;
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = r.image;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        info.format = r.format;
        info.subresourceRange = range;
        // Reinterpreted RTVs need only color-attachment usage. Inheriting STORAGE usage from the resource
        // would incorrectly require it from an sRGB view, even though this view is never used as storage.
        VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
        usage.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (r.rgba8_typeless) info.pNext = &usage;
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
    std::map<std::tuple<VkImage, VkImageAspectFlags, std::uint32_t, std::uint32_t, std::uint32_t, VkFormat>, VkImageView> views_;
    std::function<bool(VkCommandBuffer)> bindless_prepare_;
    std::unique_ptr<Depth24Converter> depth24_;
    std::function<void(bool)> bindless_commit_;
    std::unordered_map<QueryHeapId, QueryPool> query_pools_;  // element pointers stay valid across inserts
    double timestamp_period_ = 0;
    VkPhysicalDeviceMemoryProperties binary_occlusion_memory_{};
    ReplayFunctions replay_;
    DebugLabelFunctions labels_;
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
            for (; open_labels_; --open_labels_) context_.labels_.queue_end(queue_);
            vk.vkQueueWaitIdle(queue_);
        }
        if (progress_) {
            std::lock_guard lock(context_.mutex_);
            if (context_.depth24_) context_.depth24_->forget(progress_);  // its scratch is idle now
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
        if (context_.depth24_) context_.depth24_->collect();
        std::vector<PreparedList> prepared(lists.size());
        ReplayFunctions replay = context_.replay_;
        if (context_.binary_occlusion_memory_.memoryTypeCount) {
            replay.resolve_binary_occlusion = [this](VkCommandBuffer cb, VkQueryPool pool, std::uint32_t first,
                                                      std::uint32_t count, VkBuffer dst, VkDeviceSize offset) {
                binary_occlusion_->record(cb, pool, first, count, dst, offset);
            };
        }
        const auto resolve = [this](ResourceId id) { return context_.resolve(id); };
        bool binary_occlusion_needed = false;
        for (std::size_t i = 0; i < lists.size(); ++i) {
            const auto result = prepare(*lists[i], resolve, replay, prepared[i]);
            if (result != ReplayResult::success) return result;
            binary_occlusion_needed |= prepared[i].binary_occlusion;
        }
        if (binary_occlusion_needed && !binary_occlusion_) {
            auto helper = std::make_unique<BinaryOcclusionResolve>(context_.vk, context_.device());
            if (!helper->initialize(context_.binary_occlusion_memory_)) return ReplayResult::unsupported;
            binary_occlusion_ = std::move(helper);
        }
        VkCommandBuffer command_buffer = acquire();
        if (!command_buffer) return ReplayResult::unsupported;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        context_.vk.vkBeginCommandBuffer(command_buffer, &begin);
        if (context_.bindless_prepare_ && !context_.bindless_prepare_(command_buffer)) {
            // The command buffer holds at most what the set recorded for itself; submit it so the buffer is reused.
            context_.vk.vkEndCommandBuffer(command_buffer);
            const bool submitted = submit(command_buffer, VK_NULL_HANDLE, 0);
            if (context_.bindless_commit_) context_.bindless_commit_(submitted);
            return ReplayResult::unsupported;
        }
        std::uint32_t open_labels = 0;  // events may span the lists of one ExecuteCommandLists
        for (const auto& list : prepared) emit(list, command_buffer, replay, &context_.tracker_, &open_labels);
        close_labels(command_buffer, replay, open_labels);
        // A fence wait on the CPU makes GPU writes visible to the host only after a barrier to the host domain:
        // readback buffers are read through Map once the application has waited.
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        context_.vk.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                         0, 1, &host, 0, nullptr, 0, nullptr);
        context_.vk.vkEndCommandBuffer(command_buffer);
        const bool submitted = submit(command_buffer, VK_NULL_HANDLE, 0);
        if (context_.depth24_) context_.depth24_->retire(progress_, submitted_);  // what this list's copies prepared
        if (context_.bindless_commit_) context_.bindless_commit_(submitted);
        return submitted ? ReplayResult::success : ReplayResult::unsupported;
    }

    // ID3D12CommandQueue::Signal: the fence reaches `value` after all work submitted so far.
    bool signal(VkSemaphore fence, std::uint64_t value) { return submit(VK_NULL_HANDLE, fence, value); }

    // ID3D12CommandQueue::Wait: later submissions wait for the fence. A wait in a Vulkan batch only holds back that
    // batch, so it is kept and attached to the next submission.
    void wait(VkSemaphore fence, std::uint64_t value) {
        waits_.push_back(fence);
        wait_values_.push_back(value);
    }

    // ID3D12CommandQueue::SetMarker/BeginEvent/EndEvent: queue labels, between the submissions around them. An end
    // with no begin open on this queue is dropped, as Vulkan requires a matching begin. No-ops without the extension.
    void marker(DebugMarker::Kind kind, const char* label) {
        const DebugLabelFunctions& f = context_.labels_;
        if (!f.complete()) return;
        if (kind == DebugMarker::Kind::end && !open_labels_) return;
        VkDebugUtilsLabelEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
        info.pLabelName = label ? label : "";
        std::lock_guard lock(context_.submit_mutex());
        if (kind == DebugMarker::Kind::set) {
            f.queue_insert(queue_, &info);
        } else if (kind == DebugMarker::Kind::begin) {
            f.queue_begin(queue_, &info);
            ++open_labels_;
        } else {
            f.queue_end(queue_);
            --open_labels_;
        }
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
    std::unique_ptr<BinaryOcclusionResolve> binary_occlusion_;
    std::uint32_t open_labels_ = 0;  // queue labels begun and not ended
    bool ok_ = false;
};

} // namespace leonida
