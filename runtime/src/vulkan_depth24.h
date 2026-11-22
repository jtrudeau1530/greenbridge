// The compute pass behind D24_UNORM_S8_UINT on a GPU without D24 (depth24.h, depth24_convert.comp). A copy between a
// footprint and the depth aspect of such a texture goes through a scratch buffer of floats: an upload converts the
// footprint's words into the scratch and a copy command moves the scratch into the image; a readback copies the image
// into the scratch and the pass converts it into the footprint. Single-threaded: the caller holds the context's lock.
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "depth24.h"
#include "depth24_convert_spv.h"

namespace leonida {

class Depth24Converter {
public:
    // A device-local allocation for the scratch buffers; null on failure.
    using Allocate = std::function<VkDeviceMemory(const VkMemoryRequirements&)>;

    Depth24Converter(PFN_vkGetDeviceProcAddr get, VkDevice device, Allocate allocate)
        : device_(device), allocate_(std::move(allocate)) {
        loaded_ = vk_.load(get, device);
    }

    ~Depth24Converter() {
        if (!loaded_) return;
        vk_.vkDeviceWaitIdle(device_);
        release(current_);
        for (auto& batch : retired_) release(batch);
        for (auto& chunk : spare_) destroy(chunk);
        if (pipeline_) vk_.vkDestroyPipeline(device_, pipeline_, nullptr);
        if (pipeline_layout_) vk_.vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        if (set_layout_) vk_.vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
    }

    Depth24Converter(const Depth24Converter&) = delete;
    Depth24Converter& operator=(const Depth24Converter&) = delete;

    bool loaded() const { return loaded_; }

    // Everything the next submission needs for this job: the pipeline (once), a scratch region (unless the job brings
    // its own scratch buffer, as a test does) and a descriptor set over the application's buffer and the scratch.
    bool prepare(Depth24Job& job) {
        constexpr std::uint64_t window_alignment = 256;  // the largest minStorageBufferOffsetAlignment Vulkan allows
        constexpr std::uint64_t maximum_range = 1u << 27;  // maxStorageBufferRange is at least this
        if (!loaded_ || !make_pipeline() || !job.app || !job.width || !job.height || !job.slices || job.app_offset % 4 ||
            job.app_row_words < job.width || job.app_slice_words < std::uint64_t(job.app_row_words) * (job.height - 1) + job.width)
            return false;
        const std::uint64_t span_words = std::uint64_t(job.slices - 1) * job.app_slice_words +
                                         std::uint64_t(job.height - 1) * job.app_row_words + job.width;
        const std::uint64_t window = job.app_offset / window_alignment * window_alignment;
        const std::uint64_t range = job.app_offset - window + span_words * 4;
        if (window + range > job.app_size || range > maximum_range) return false;
        job.app_base_words = std::uint32_t((job.app_offset - window) / 4);

        const std::uint64_t scratch_bytes = std::uint64_t(job.width) * job.height * job.slices * 4;
        if (!job.scratch) {
            if (!scratch(scratch_bytes, job.scratch, job.scratch_offset)) return false;
        }
        job.set = allocate_set();
        if (!job.set) return false;
        const VkDescriptorBufferInfo buffers[2] = {{job.app, window, range}, {job.scratch, job.scratch_offset, scratch_bytes}};
        VkWriteDescriptorSet writes[2]{};
        for (std::uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = job.set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffers[i];
        }
        vk_.vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
        return true;
    }

    // Records the dispatch. The caller records the barriers around it.
    void run(VkCommandBuffer command_buffer, const Depth24Job& job) const {
        struct Push {
            std::uint32_t width, height, slices, to_float, app_base, app_row, app_slice, unused;
        } push{job.width, job.height, job.slices, job.to_float ? 1u : 0u, job.app_base_words, job.app_row_words,
               job.app_slice_words, 0};
        vk_.vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vk_.vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &job.set, 0,
                                    nullptr);
        vk_.vkCmdPushConstants(command_buffer, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vk_.vkCmdDispatch(command_buffer, (job.width + 7) / 8, (job.height + 7) / 8, job.slices);
    }

    // The jobs prepared since the last call belong to a submission that `timeline` reaches `ticket` for.
    void retire(VkSemaphore timeline, std::uint64_t ticket) {
        if (current_.chunks.empty() && current_.pools.empty()) return;
        current_.timeline = timeline;
        current_.ticket = ticket;
        retired_.push_back(std::move(current_));
        current_ = {};
    }

    // The submissions on `timeline` are over and the semaphore is about to go (its queue is idle).
    void forget(VkSemaphore timeline) {
        for (std::size_t i = 0; i < retired_.size();) {
            if (retired_[i].timeline != timeline) {
                ++i;
                continue;
            }
            release(retired_[i]);
            retired_.erase(retired_.begin() + std::ptrdiff_t(i));
        }
    }

    // Frees (or keeps for reuse) what finished submissions used.
    void collect() {
        for (std::size_t i = 0; i < retired_.size();) {
            std::uint64_t done = 0;
            vk_.vkGetSemaphoreCounterValue(device_, retired_[i].timeline, &done);
            if (done < retired_[i].ticket) {
                ++i;
                continue;
            }
            for (auto& chunk : retired_[i].chunks) {
                chunk.used = 0;
                if (spare_.size() < 4) spare_.push_back(chunk);
                else destroy(chunk);
            }
            retired_[i].chunks.clear();
            release(retired_[i]);
            retired_.erase(retired_.begin() + std::ptrdiff_t(i));
        }
    }

private:
    struct Chunk {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0, used = 0;
    };
    struct Batch {
        std::vector<Chunk> chunks;
        std::vector<VkDescriptorPool> pools;
        VkSemaphore timeline = VK_NULL_HANDLE;
        std::uint64_t ticket = 0;
    };

    struct Functions {
#define LEONIDA_DEPTH24_FUNCTIONS(X)                                                                           \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout)                            \
    X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout)                       \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool)      \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreateBuffer) X(vkDestroyBuffer)                 \
    X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkFreeMemory) X(vkCmdBindPipeline)                \
    X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch) X(vkGetSemaphoreCounterValue)            \
    X(vkDeviceWaitIdle)
#define LEONIDA_DEPTH24_MEMBER(name) PFN_##name name = nullptr;
        LEONIDA_DEPTH24_FUNCTIONS(LEONIDA_DEPTH24_MEMBER)
#undef LEONIDA_DEPTH24_MEMBER

        bool load(PFN_vkGetDeviceProcAddr get, VkDevice device) {
            bool all = true;
#define LEONIDA_DEPTH24_LOAD(name)                              \
    name = reinterpret_cast<PFN_##name>(get(device, #name));    \
    all = all && name != nullptr;
            LEONIDA_DEPTH24_FUNCTIONS(LEONIDA_DEPTH24_LOAD)
#undef LEONIDA_DEPTH24_LOAD
#undef LEONIDA_DEPTH24_FUNCTIONS
            return all;
        }
    };

    bool make_pipeline() {
        if (pipeline_) return true;
        VkDescriptorSetLayoutBinding bindings[2]{};
        for (std::uint32_t i = 0; i < 2; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_info.bindingCount = 2;
        set_info.pBindings = bindings;
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = sizeof(depth24_convert_spirv);
        module_info.pCode = depth24_convert_spirv;
        VkShaderModule module = VK_NULL_HANDLE;
        bool ok = vk_.vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &set_layout_) == VK_SUCCESS;
        layout_info.pSetLayouts = &set_layout_;
        ok = ok && vk_.vkCreatePipelineLayout(device_, &layout_info, nullptr, &pipeline_layout_) == VK_SUCCESS &&
             vk_.vkCreateShaderModule(device_, &module_info, nullptr, &module) == VK_SUCCESS;
        if (ok) {
            VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                          module, "main", nullptr};
            info.layout = pipeline_layout_;
            ok = vk_.vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline_) == VK_SUCCESS;
        }
        if (module) vk_.vkDestroyShaderModule(device_, module, nullptr);
        return ok;
    }

    VkDescriptorSet allocate_set() {
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (!current_.pools.empty()) {
                VkDescriptorSetAllocateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                info.descriptorPool = current_.pools.back();
                info.descriptorSetCount = 1;
                info.pSetLayouts = &set_layout_;
                VkDescriptorSet set = VK_NULL_HANDLE;
                if (vk_.vkAllocateDescriptorSets(device_, &info, &set) == VK_SUCCESS) return set;
            }
            constexpr std::uint32_t sets = 64;
            const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sets * 2};
            VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            info.maxSets = sets;
            info.poolSizeCount = 1;
            info.pPoolSizes = &size;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            if (vk_.vkCreateDescriptorPool(device_, &info, nullptr, &pool) != VK_SUCCESS) return VK_NULL_HANDLE;
            current_.pools.push_back(pool);
        }
        return VK_NULL_HANDLE;
    }

    // `bytes` of scratch, aligned for a descriptor offset, from this submission's chunks.
    bool scratch(std::uint64_t bytes, VkBuffer& buffer, VkDeviceSize& offset) {
        constexpr VkDeviceSize alignment = 256, minimum = 4u << 20;
        for (auto& chunk : current_.chunks) {
            const VkDeviceSize start = (chunk.used + alignment - 1) / alignment * alignment;
            if (start + bytes <= chunk.size) {
                chunk.used = start + bytes;
                buffer = chunk.buffer;
                offset = start;
                return true;
            }
        }
        Chunk chunk;
        for (std::size_t i = 0; i < spare_.size(); ++i)
            if (spare_[i].size >= bytes) {
                chunk = spare_[i];
                spare_.erase(spare_.begin() + std::ptrdiff_t(i));
                break;
            }
        if (!chunk.buffer && !make_chunk(std::max<std::uint64_t>(bytes, minimum), chunk)) return false;
        chunk.used = bytes;
        buffer = chunk.buffer;
        offset = 0;
        current_.chunks.push_back(chunk);
        return true;
    }

    bool make_chunk(std::uint64_t size, Chunk& out) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vk_.vkCreateBuffer(device_, &info, nullptr, &out.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vk_.vkGetBufferMemoryRequirements(device_, out.buffer, &requirements);
        out.memory = allocate_ ? allocate_(requirements) : VK_NULL_HANDLE;
        if (!out.memory || vk_.vkBindBufferMemory(device_, out.buffer, out.memory, 0) != VK_SUCCESS) {
            destroy(out);
            return false;
        }
        out.size = size;
        return true;
    }

    void destroy(Chunk& chunk) {
        if (chunk.buffer) vk_.vkDestroyBuffer(device_, chunk.buffer, nullptr);
        if (chunk.memory) vk_.vkFreeMemory(device_, chunk.memory, nullptr);
        chunk = {};
    }

    void release(Batch& batch) {
        for (auto& chunk : batch.chunks) destroy(chunk);
        for (const auto pool : batch.pools) vk_.vkDestroyDescriptorPool(device_, pool, nullptr);
        batch.chunks.clear();
        batch.pools.clear();
    }

    VkDevice device_;
    Allocate allocate_;
    Functions vk_;
    bool loaded_ = false;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    Batch current_;                // prepared, not yet submitted
    std::vector<Batch> retired_;   // submitted, waiting for the GPU
    std::vector<Chunk> spare_;     // finished scratch buffers kept for reuse
};

}  // namespace leonida
