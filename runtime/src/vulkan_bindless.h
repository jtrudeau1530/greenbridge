#pragma once

// The bindless descriptor set of translated shaders: set 0 of the contract in docs/design.md and
// docs/d3d12_vulkan.md section 5, as tools/rdna2spirv.py declares it.
//
//   binding  0  sampled 2D images          4096        5  sampled 1D images          256
//            1  samplers                   1024        6  sampled 3D images          256
//            2  the descriptor hash table     1        7  3D r32ui storage images    256
//            3  sampled 2D-array images    1024        8 + 3 * dimension + class     256 each:
//            4  2D r32ui storage images    1024           storage images by component class
//   dimension: 0 = 2D, 1 = 2D array, 2 = 3D, 3 = 1D (the translator's STORE_DIMS); class: 0 = float, 1 = uint, 2 = sint.
// The counts are those of runtime/test/vk_pipeline_check.cpp, which creates every translated shader against them.
//
// A shader turns a T# or S# into an index through the hash table (descriptor_table.h) and uses that index in
// whichever binding the instruction calls for: the sampled array of the image's dimension, or one of its storage
// arrays. So an image has one index, the same in every binding of its dimension, and each dimension has its own
// index space. The storage arrays hold 256, so an image with storage views gets an index below 256; one without
// takes a higher index while there is one. Index 0 of every array is a fallback, because a miss gives 0: a 1x1
// image of zeros, a nearest clamping sampler, and 1x1 storage images. The caller owns the views and samplers it
// adds and keeps them alive while they are in the set; this class owns only the fallbacks and the table's buffer.
//
// Device features: runtimeDescriptorArray for the shaders. With BindlessFeatures::update_after_bind
// (descriptorBindingSampledImageUpdateAfterBind, ...StorageImage..., ...StorageBuffer...) flush() may run while
// command buffers that use the set are pending, as long as no descriptor they use changes and flush_needs_idle()
// is false; without it the caller flushes only when none is. With partially_bound (descriptorBindingPartiallyBound) only index 0 is written at
// creation; otherwise every element of every array starts as the fallback.
//
// The table of binding 2 lives in DEVICE_LOCAL memory: every divergent T#/S# lookup reads it, and a HOST_VISIBLE
// table measured 19-46x slower there at 100k entries on an RTX 2060 SUPER (vk_descriptor_bench, D3D-56, #639).
// flush() takes the command buffer that will use the set: it uploads only the slots that changed, from a mapped
// staging buffer, and barriers the copy before the draws. The table at creation is empty until that first flush.
//
// Not thread-safe: the owner serialises calls.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "descriptor_table.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace leonida {

// The translator's image dimensions; the values are its STORE_DIMS indices. A cube is a 2D array of six layers.
enum class ImageDimension : std::uint32_t { dim_2d = 0, dim_2d_array = 1, dim_3d = 2, dim_1d = 3 };

// Storage views of one image by the component type a shader reads or writes it with (the translator picks by the
// T# format at run time). r32ui is for the atomic instructions, whose image type is R32ui (2D and 3D only).
// A null view leaves the fallback in that binding.
struct StorageViews {
    VkImageView as_float = VK_NULL_HANDLE;
    VkImageView as_uint = VK_NULL_HANDLE;
    VkImageView as_sint = VK_NULL_HANDLE;
    VkImageView r32ui = VK_NULL_HANDLE;
};

struct BindlessFeatures {
    bool update_after_bind = false;
    bool partially_bound = false;
};

namespace bindless {

struct Binding {
    VkDescriptorType type;
    std::uint32_t count;
};

inline constexpr std::uint32_t table_binding = 2;
inline constexpr std::uint32_t sampler_binding = 1;
inline constexpr std::uint32_t storage_indices = 256;  // what every storage array of bindings 8-19 holds
inline constexpr std::uint32_t no_binding = 0xffffffffu;

// Indexed by binding number.
inline constexpr std::array<Binding, 20> bindings = {{
    {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1024},
    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},   {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
}};

constexpr std::uint32_t sampled_binding(ImageDimension dimension) {
    constexpr std::uint32_t by_dimension[] = {0, 3, 6, 5};
    return by_dimension[std::uint32_t(dimension)];
}

// component_class: 0 = float, 1 = uint, 2 = sint.
constexpr std::uint32_t storage_binding(ImageDimension dimension, std::uint32_t component_class) {
    return 8 + 3 * std::uint32_t(dimension) + component_class;
}

constexpr std::uint32_t r32ui_binding(ImageDimension dimension) {
    return dimension == ImageDimension::dim_2d ? 4 : dimension == ImageDimension::dim_3d ? 7 : no_binding;
}

// How many images of a dimension the set holds, the fallback included.
constexpr std::uint32_t image_capacity(ImageDimension dimension) { return bindings[sampled_binding(dimension)].count; }

} // namespace bindless

class BindlessSet {
public:
    BindlessSet(PFN_vkGetDeviceProcAddr get, VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
                BindlessFeatures features = {})
        : device_(device), memory_(memory), features_(features) {
        ok_ = vk_.load(get, device) && create_layout() && create_fallbacks() && create_table_buffers() && fill();
    }

    ~BindlessSet() {
        release_retired();
        destroy(table_buffers_);
        if (fallback_sampler_) vk_.vkDestroySampler(device_, fallback_sampler_, nullptr);
        for (const VkImageView view : fallback_views_) vk_.vkDestroyImageView(device_, view, nullptr);
        for (const auto& image : fallback_images_) {
            if (image.image) vk_.vkDestroyImage(device_, image.image, nullptr);
            if (image.memory) vk_.vkFreeMemory(device_, image.memory, nullptr);
        }
        if (pool_) vk_.vkDestroyDescriptorPool(device_, pool_, nullptr);  // frees the set
        if (layout_) vk_.vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
    }

    BindlessSet(const BindlessSet&) = delete;
    BindlessSet& operator=(const BindlessSet&) = delete;

    bool ok() const { return ok_; }
    // Set 0 of every pipeline layout made for translated shaders.
    VkDescriptorSetLayout layout() const { return layout_; }
    // Bound at set 0. The same set for the life of this object: flush() updates it in place.
    VkDescriptorSet set() const { return set_; }

    // The fallback images are created with undefined contents: record this once, in a command buffer that executes
    // before anything uses the set. It moves them to GENERAL, the layout their descriptors name, and zeroes them.
    bool fallbacks_ready() const { return fallbacks_ready_; }
    void record_fallback_init(VkCommandBuffer command_buffer) {
        fallbacks_ready_ = true;
        std::vector<VkImageMemoryBarrier> barriers;
        for (const auto& image : fallback_images_) {
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image.image;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            barriers.push_back(barrier);
        }
        vk_.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                 nullptr, 0, nullptr, std::uint32_t(barriers.size()), barriers.data());
        const VkClearColorValue zero{};
        const VkImageSubresourceRange whole{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (const auto& image : fallback_images_)
            vk_.vkCmdClearColorImage(command_buffer, image.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &whole);
        VkMemoryBarrier done{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        done.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        done.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vk_.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                                 &done, 0, nullptr, 0, nullptr);
    }

    // Gives an image an index in its dimension's bindings: 0 when the dimension is full. `sampled` goes to the
    // sampled array in `sampled_layout` (the layout the image is in when shaders read it); the storage views go to
    // the storage arrays in GENERAL. The caller maps the T# to the index with insert_key(). Takes effect at flush().
    std::uint32_t add_image(ImageDimension dimension, VkImageView sampled, VkImageLayout sampled_layout,
                            const StorageViews& storage = {}) {
        const bool stores = storage.as_float || storage.as_uint || storage.as_sint || storage.r32ui;
        auto& used = image_used_[std::uint32_t(dimension)];
        const std::uint32_t low_end = std::min(std::uint32_t(used.size()), bindless::storage_indices);
        std::uint32_t index = stores ? 0 : allocate(used, low_end, std::uint32_t(used.size()));
        if (!index) index = allocate(used, 1, low_end);
        if (index) write_image(dimension, index, sampled, sampled_layout, storage);
        return index;
    }

    // New views or a new layout for an index add_image returned (a texture's state changed, say).
    void update_image(ImageDimension dimension, std::uint32_t index, VkImageView sampled, VkImageLayout sampled_layout,
                      const StorageViews& storage = {}) {
        const auto& used = image_used_[std::uint32_t(dimension)];
        if (index && index < used.size() && used[index]) write_image(dimension, index, sampled, sampled_layout, storage);
    }

    // Returns the index to the fallback and frees it. The caller removes the image's keys with erase_key() first.
    void remove_image(ImageDimension dimension, std::uint32_t index) {
        auto& used = image_used_[std::uint32_t(dimension)];
        if (!index || index >= used.size() || !used[index]) return;
        write_image(dimension, index, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, {});
        used[index] = 0;
    }

    // 0 when the sampler array is full.
    std::uint32_t add_sampler(VkSampler sampler) {
        const std::uint32_t index = allocate(sampler_used_, 1, std::uint32_t(sampler_used_.size()));
        if (index) write_sampler(index, sampler);
        return index;
    }

    void remove_sampler(std::uint32_t index) {
        if (!index || index >= sampler_used_.size() || !sampler_used_[index]) return;
        write_sampler(index, fallback_sampler_);
        sampler_used_[index] = 0;
    }

    // T# and S# keys to the indices above: the table of binding 2 (descriptor_table.h). Changes reach the shaders
    // at flush().
    bool insert_key(const DescriptorKey& key, std::uint32_t index) {
        const auto result = table_.insert_or_update(key.data(), index);
        table_dirty_ |= result != DescriptorTable::InsertResult::unchanged;
        existing_entries_changed_ |= result == DescriptorTable::InsertResult::replaced ||
                                     result == DescriptorTable::InsertResult::added_with_relocation;
        return true;
    }
    // Erasing compacts the key's probe cluster, so other keys' entries may change slots.
    bool erase_key(const DescriptorKey& key) {
        if (!table_.erase(key)) return false;
        table_dirty_ = existing_entries_changed_ = true;
        return true;
    }
    const DescriptorTable& table() const { return table_; }

    // True when the next flush() will change entries a running shader may be probing, or replace the table's
    // buffer: the caller then flushes only once no command buffer that uses the set is still executing. Without
    // it the flush only adds entries, and the copy of an entry is not atomic: a shader probing a different key
    // could in principle see a torn entry that matches it, which is accepted as vanishingly unlikely because the
    // added key is new (there is no live key equal to it).
    bool flush_needs_idle() const {
        return table_dirty_ && (existing_entries_changed_ || force_full_upload_ ||
                                table_.size_bytes() > table_buffers_.table.size);
    }

    // True when the table's buffer was allocated in DEVICE_LOCAL memory (the fallback for a device without it is
    // HOST_VISIBLE, which still works, just slower).
    bool table_in_device_local_memory() const { return table_device_local_; }

    // True when the command buffer that carried the last table upload reached the queue; false keeps the upload
    // obligation, and the next flush() rebuilds the whole table (the changed-slot list was consumed when the copy
    // was recorded). A failed vkQueueSubmit is the caller's signal, not ours.
    void table_upload_result(bool submitted) {
        if (!upload_in_flight_) return;
        upload_in_flight_ = false;
        if (submitted) return;
        table_dirty_ = true;
        force_full_upload_ = true;
    }

    // Writes the pending descriptor changes and uploads the table, in `command_buffer`, before the commands that use
    // the set are recorded after it. A table that has grown gets new buffers (binding 2 is rewritten); the old ones
    // stay until release_retired(). False when they cannot be made: the set then still has the table as of the last
    // flush.
    bool flush(VkCommandBuffer command_buffer) {
        bool written = true;
        if (table_dirty_ || !table_uploaded_) written = upload_table(command_buffer);
        if (!pending_.empty()) {
            std::vector<VkWriteDescriptorSet> writes(pending_.size());
            for (std::size_t i = 0; i < pending_.size(); ++i) {
                writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                writes[i].dstSet = set_;
                writes[i].dstBinding = pending_[i].binding;
                writes[i].dstArrayElement = pending_[i].index;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = bindless::bindings[pending_[i].binding].type;
                writes[i].pImageInfo = &pending_[i].info;
            }
            vk_.vkUpdateDescriptorSets(device_, std::uint32_t(writes.size()), writes.data(), 0, nullptr);
            pending_.clear();
            pending_slots_.clear();
        }
        return written;
    }

    // Frees the buffers of tables that have since grown: when nothing submitted before the last flush() still runs.
    void release_retired() {
        for (auto& buffers : retired_) destroy(buffers);
        retired_.clear();
    }

private:
    struct Functions {
#define LEONIDA_BINDLESS_FUNCTIONS(X)                                                                          \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreateDescriptorPool)                   \
    X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreateBuffer)         \
    X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkAllocateMemory)              \
    X(vkFreeMemory) X(vkMapMemory) X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements)          \
    X(vkBindImageMemory) X(vkCreateImageView) X(vkDestroyImageView) X(vkCreateSampler) X(vkDestroySampler)     \
    X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkCmdUpdateBuffer) X(vkCmdClearColorImage)
#define LEONIDA_BINDLESS_MEMBER(name) PFN_##name name = nullptr;
        LEONIDA_BINDLESS_FUNCTIONS(LEONIDA_BINDLESS_MEMBER)
#undef LEONIDA_BINDLESS_MEMBER

        bool load(PFN_vkGetDeviceProcAddr get, VkDevice device) {
            bool all = true;
#define LEONIDA_BINDLESS_LOAD(name)                                      \
    name = reinterpret_cast<PFN_##name>(get(device, #name));             \
    all = all && name != nullptr;
            LEONIDA_BINDLESS_FUNCTIONS(LEONIDA_BINDLESS_LOAD)
#undef LEONIDA_BINDLESS_LOAD
#undef LEONIDA_BINDLESS_FUNCTIONS
            return all;
        }
    };

    struct FallbackImage {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };

    struct TableBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        std::size_t size = 0;
    };

    // The table and the mapped staging copy the CPU writes before vkCmdCopyBuffer moves the changes.
    struct TableBuffers {
        TableBuffer table;    // DEVICE_LOCAL when the device has it; what binding 2 names
        TableBuffer staging;  // HOST_VISIBLE | HOST_COHERENT
    };

    struct Pending {
        std::uint32_t binding;
        std::uint32_t index;
        VkDescriptorImageInfo info;
    };

    static constexpr std::uint32_t dimensions = 4, classes = 3;

    // The first free index in [begin, end), marked used; 0 when there is none.
    static std::uint32_t allocate(std::vector<std::uint8_t>& used, std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t i = begin; i < end; ++i) {
            if (used[i]) continue;
            used[i] = 1;
            return i;
        }
        return 0;
    }

    std::uint32_t memory_type(std::uint32_t allowed, VkMemoryPropertyFlags wanted) const {
        for (std::uint32_t i = 0; i < memory_.memoryTypeCount; ++i)
            if ((allowed & (1u << i)) && (memory_.memoryTypes[i].propertyFlags & wanted) == wanted) return i;
        return 0xffffffffu;
    }

    bool create_layout() {
        std::array<VkDescriptorSetLayoutBinding, bindless::bindings.size()> entries{};
        std::array<VkDescriptorBindingFlags, bindless::bindings.size()> flags{};
        std::array<VkDescriptorPoolSize, bindless::bindings.size()> sizes{};
        for (std::uint32_t b = 0; b < entries.size(); ++b) {
            entries[b] = {b, bindless::bindings[b].type, bindless::bindings[b].count, VK_SHADER_STAGE_ALL, nullptr};
            flags[b] = (features_.update_after_bind ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0) |
                       (features_.partially_bound ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0);
            sizes[b] = {bindless::bindings[b].type, bindless::bindings[b].count};
        }
        VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        flags_info.bindingCount = std::uint32_t(flags.size());
        flags_info.pBindingFlags = flags.data();
        VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &flags_info};
        layout_info.flags = features_.update_after_bind ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT : 0;
        layout_info.bindingCount = std::uint32_t(entries.size());
        layout_info.pBindings = entries.data();
        if (vk_.vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &layout_) != VK_SUCCESS) return false;

        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.flags = features_.update_after_bind ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = std::uint32_t(sizes.size());
        pool_info.pPoolSizes = sizes.data();
        if (vk_.vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool_) != VK_SUCCESS) return false;
        VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = pool_;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &layout_;
        if (vk_.vkAllocateDescriptorSets(device_, &set_info, &set_) != VK_SUCCESS) return false;

        for (std::uint32_t d = 0; d < dimensions; ++d) {
            image_used_[d].assign(bindless::image_capacity(ImageDimension(d)), 0);
            image_used_[d][0] = 1;
        }
        sampler_used_.assign(bindless::bindings[bindless::sampler_binding].count, 0);
        sampler_used_[0] = 1;
        return true;
    }

    // One 1x1 image for each image type and component class, and the views of them every binding needs.
    bool create_fallbacks() {
        constexpr VkImageType types[] = {VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_3D, VK_IMAGE_TYPE_1D};
        constexpr VkFormat formats[classes] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT};
        // dimension -> (index into types, view type)
        constexpr std::uint32_t type_of[dimensions] = {0, 0, 1, 2};
        constexpr VkImageViewType view_types[dimensions] = {VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                                                            VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_VIEW_TYPE_1D};
        for (std::uint32_t t = 0; t < 3; ++t) {
            for (std::uint32_t c = 0; c < classes; ++c) {
                VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                info.imageType = types[t];
                info.format = formats[c];
                info.extent = {1, 1, 1};
                info.mipLevels = 1;
                info.arrayLayers = 1;
                info.samples = VK_SAMPLE_COUNT_1_BIT;
                info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                             (c == 0 ? VK_IMAGE_USAGE_SAMPLED_BIT : 0);
                FallbackImage& image = fallback_images_[t * classes + c];
                if (vk_.vkCreateImage(device_, &info, nullptr, &image.image) != VK_SUCCESS) return false;
                VkMemoryRequirements requirements{};
                vk_.vkGetImageMemoryRequirements(device_, image.image, &requirements);
                VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocation.allocationSize = requirements.size;
                allocation.memoryTypeIndex = memory_type(requirements.memoryTypeBits, 0);
                if (allocation.memoryTypeIndex == 0xffffffffu ||
                    vk_.vkAllocateMemory(device_, &allocation, nullptr, &image.memory) != VK_SUCCESS ||
                    vk_.vkBindImageMemory(device_, image.image, image.memory, 0) != VK_SUCCESS)
                    return false;
            }
        }
        for (std::uint32_t d = 0; d < dimensions; ++d) {
            for (std::uint32_t c = 0; c < classes; ++c) {
                VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                info.image = fallback_images_[type_of[d] * classes + c].image;
                info.viewType = view_types[d];
                info.format = formats[c];
                info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkImageView view = VK_NULL_HANDLE;
                if (vk_.vkCreateImageView(device_, &info, nullptr, &view) != VK_SUCCESS) return false;
                fallback_views_.push_back(view);
                fallback_view_[d][c] = view;
            }
        }
        VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler.magFilter = sampler.minFilter = VK_FILTER_NEAREST;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.maxLod = VK_LOD_CLAMP_NONE;
        return vk_.vkCreateSampler(device_, &sampler, nullptr, &fallback_sampler_) == VK_SUCCESS;
    }

    bool create(TableBuffer& out, std::size_t bytes, bool device_local, bool* got_device_local = nullptr) const {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     (device_local ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        TableBuffer made;
        made.size = bytes;
        if (vk_.vkCreateBuffer(device_, &info, nullptr, &made.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vk_.vkGetBufferMemoryRequirements(device_, made.buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex =
            device_local ? memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                         : memory_type(requirements.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (device_local && allocation.memoryTypeIndex == 0xffffffffu)
            allocation.memoryTypeIndex = memory_type(requirements.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (got_device_local)
            *got_device_local = allocation.memoryTypeIndex != 0xffffffffu &&
                                (memory_.memoryTypes[allocation.memoryTypeIndex].propertyFlags &
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        bool ready = allocation.memoryTypeIndex != 0xffffffffu &&
                     vk_.vkAllocateMemory(device_, &allocation, nullptr, &made.memory) == VK_SUCCESS &&
                     vk_.vkBindBufferMemory(device_, made.buffer, made.memory, 0) == VK_SUCCESS;
        if (ready && !device_local)
            ready = vk_.vkMapMemory(device_, made.memory, 0, VK_WHOLE_SIZE, 0, &made.mapped) == VK_SUCCESS;
        if (!ready) {
            destroy(made);
            return false;
        }
        out = made;
        return true;
    }

    void destroy(TableBuffer& buffer) const {
        if (buffer.buffer) vk_.vkDestroyBuffer(device_, buffer.buffer, nullptr);
        if (buffer.memory) vk_.vkFreeMemory(device_, buffer.memory, nullptr);  // unmaps it
        buffer = {};
    }

    void destroy(TableBuffers& buffers) const {
        destroy(buffers.staging);
        destroy(buffers.table);
    }

    bool create_table_buffers() {
        const std::size_t bytes = table_.size_bytes();
        TableBuffers made;
        bool device_local = false;
        if (!create(made.table, bytes, true, &device_local) || !create(made.staging, bytes, false)) {
            destroy(made);
            return false;
        }
        table_buffers_ = made;
        table_device_local_ = device_local;
        table_uploaded_ = false;  // the first flush() uploads it, in the command buffer that will use it
        table_dirty_ = existing_entries_changed_ = false;
        return true;
    }

    // Uploads what changed since the last call: the whole table when it grew (the mask moved) or was never uploaded,
    // the changed slots otherwise. The copy and its barrier go into `command_buffer`, before the draws.
    bool upload_table(VkCommandBuffer command_buffer) {
        if (!command_buffer) return false;
        const std::size_t bytes = table_.size_bytes();
        bool full = force_full_upload_ || !table_uploaded_ || bytes > table_buffers_.table.size;
        bool grew = false;
        if (bytes > table_buffers_.table.size) {
            TableBuffers larger;
            bool device_local = false;
            if (!create(larger.table, bytes, true, &device_local) || !create(larger.staging, bytes, false)) {
                destroy(larger);
                return false;
            }
            table_device_local_ = device_local;
            retired_.push_back(table_buffers_);
            table_buffers_ = larger;
            write_table_binding(table_buffers_.table.buffer);
            full = true;
            grew = true;
        }
        const DescriptorTable::Changes changes = table_.take_changes();
        full = full || changes.all;
        bool recorded = false;
        // The staging buffer is written only for buffers that were just created (first upload or growth): a later
        // flush then either updates slots with vkCmdUpdateBuffer or rebuilds from a fresh pair, so no copy still in
        // flight can be reading bytes the CPU rewrites.
        const bool fresh = grew || !table_uploaded_;
        const auto upload_all = [&]() {
            const std::vector<std::uint32_t> words = table_.build_gpu_buffer();
            if (fresh) {
                std::memcpy(table_buffers_.staging.mapped, words.data(), bytes);
                copy_table_range(command_buffer, 0, bytes);
            } else {
                update_table_range(command_buffer, 0, words.data(), bytes);
            }
            recorded = true;
        };
        if (full) {
            upload_all();
        } else if (!changes.slots.empty()) {
            std::vector<std::uint32_t> slots = changes.slots;
            std::sort(slots.begin(), slots.end());
            slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
            bool staged = true;
            for (std::size_t first = 0; staged && first < slots.size();) {
                std::size_t last = first;
                while (last + 1 < slots.size() && slots[last + 1] == slots[last] + 1) ++last;
                const std::uint32_t count = std::uint32_t(last - first + 1);
                std::vector<std::uint32_t> words(std::size_t(count) * 10);
                staged = table_.read_slots(slots[first], count, words.data());
                if (staged) {
                    update_table_range(command_buffer, entry_offset(slots[first]), words.data(),
                                       VkDeviceSize(words.size() * sizeof(std::uint32_t)));
                    recorded = true;
                }
                first = last + 1;
            }
            if (!staged) upload_all();  // an inconsistent slot list: the whole table is still correct
        }
        if (!recorded) return true;  // nothing was copied: keep the obligation for the next flush
        record_table_barrier(command_buffer);
        upload_in_flight_ = true;
        table_dirty_ = existing_entries_changed_ = false;
        table_uploaded_ = true;
        force_full_upload_ = false;
        return true;
    }

    // Copies the data into the command buffer while recording (64 KiB per call, the Vulkan limit), so no staging
    // buffer is read at execution: a later flush can rewrite its own staging without racing a pending copy.
    void update_table_range(VkCommandBuffer command_buffer, VkDeviceSize offset, const void* data,
                            VkDeviceSize size) const {
        constexpr VkDeviceSize max_chunk = 65536;
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for (VkDeviceSize done = 0; done < size; done += max_chunk) {
            const VkDeviceSize chunk = std::min(max_chunk, size - done);
            vk_.vkCmdUpdateBuffer(command_buffer, table_buffers_.table.buffer, offset + done, chunk, bytes + done);
        }
    }

    void copy_table_range(VkCommandBuffer command_buffer, VkDeviceSize offset, VkDeviceSize size) const {
        const VkBufferCopy region{offset, offset, size};
        vk_.vkCmdCopyBuffer(command_buffer, table_buffers_.staging.buffer, table_buffers_.table.buffer, 1, &region);
    }

    void record_table_barrier(VkCommandBuffer command_buffer) const {
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = table_buffers_.table.buffer;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;
        vk_.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                 nullptr, 1, &barrier, 0, nullptr);
    }

    // The entry's first dword after the mask word; each entry is 10 dwords (40 bytes).
    static constexpr VkDeviceSize entry_bytes = 10 * sizeof(std::uint32_t);
    static VkDeviceSize entry_offset(std::uint32_t slot) {
        return sizeof(std::uint32_t) + VkDeviceSize(slot) * entry_bytes;
    }

    void write_table_binding(VkBuffer buffer) const {
        const VkDescriptorBufferInfo info{buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set_;
        write.dstBinding = bindless::table_binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &info;
        vk_.vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    }

    VkDescriptorImageInfo fallback_info(std::uint32_t binding) const {
        if (binding == bindless::sampler_binding) return {fallback_sampler_, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        for (std::uint32_t d = 0; d < dimensions; ++d) {
            const ImageDimension dimension{d};
            std::uint32_t component_class = classes;  // of the fallback view this binding takes
            if (binding == bindless::sampled_binding(dimension)) component_class = 0;
            if (binding == bindless::r32ui_binding(dimension)) component_class = 1;
            for (std::uint32_t c = 0; c < classes; ++c)
                if (binding == bindless::storage_binding(dimension, c)) component_class = c;
            if (component_class < classes)
                return {VK_NULL_HANDLE, fallback_view_[d][component_class], VK_IMAGE_LAYOUT_GENERAL};
        }
        return {};
    }

    // The table at binding 2 and the fallback at index 0 of every array, or in every element without partially_bound.
    bool fill() {
        write_table_binding(table_buffers_.table.buffer);
        std::vector<std::vector<VkDescriptorImageInfo>> infos;
        std::vector<VkWriteDescriptorSet> writes;
        infos.reserve(bindless::bindings.size());
        for (std::uint32_t b = 0; b < bindless::bindings.size(); ++b) {
            if (b == bindless::table_binding) continue;
            const std::uint32_t count = features_.partially_bound ? 1 : bindless::bindings[b].count;
            infos.emplace_back(count, fallback_info(b));
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = set_;
            write.dstBinding = b;
            write.descriptorCount = count;
            write.descriptorType = bindless::bindings[b].type;
            write.pImageInfo = infos.back().data();
            writes.push_back(write);
        }
        vk_.vkUpdateDescriptorSets(device_, std::uint32_t(writes.size()), writes.data(), 0, nullptr);
        return true;
    }

    // A slot has one pending write: a later one takes its place. So a view or sampler that is removed before the
    // next flush(), and may be destroyed by then, is never written.
    void queue(std::uint32_t binding, std::uint32_t index, const VkDescriptorImageInfo& info) {
        const std::uint64_t slot = (std::uint64_t(binding) << 32) | index;
        const auto [at, added] = pending_slots_.try_emplace(slot, pending_.size());
        if (added)
            pending_.push_back({binding, index, info});
        else
            pending_[at->second].info = info;
    }

    void write_sampler(std::uint32_t index, VkSampler sampler) {
        queue(bindless::sampler_binding, index, {sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
    }

    // Queues every binding of the dimension at `index`: the given views, or the fallback where one is null.
    void write_image(ImageDimension dimension, std::uint32_t index, VkImageView sampled, VkImageLayout sampled_layout,
                     const StorageViews& storage) {
        const std::uint32_t sampled_at = bindless::sampled_binding(dimension);
        queue(sampled_at, index,
              sampled ? VkDescriptorImageInfo{VK_NULL_HANDLE, sampled, sampled_layout} : fallback_info(sampled_at));
        const auto store = [&](std::uint32_t binding, VkImageView view) {
            if (binding == bindless::no_binding || index >= bindless::bindings[binding].count) return;
            queue(binding, index,
                  view ? VkDescriptorImageInfo{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL} : fallback_info(binding));
        };
        store(bindless::storage_binding(dimension, 0), storage.as_float);
        store(bindless::storage_binding(dimension, 1), storage.as_uint);
        store(bindless::storage_binding(dimension, 2), storage.as_sint);
        store(bindless::r32ui_binding(dimension), storage.r32ui);
    }

    VkDevice device_;
    VkPhysicalDeviceMemoryProperties memory_;
    BindlessFeatures features_;
    Functions vk_;
    bool ok_ = false;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    std::array<FallbackImage, 3 * classes> fallback_images_{};  // [image type: 2D, 3D, 1D][class]
    std::vector<VkImageView> fallback_views_;
    VkImageView fallback_view_[dimensions][classes] = {};
    VkSampler fallback_sampler_ = VK_NULL_HANDLE;
    bool fallbacks_ready_ = false;
    std::array<std::vector<std::uint8_t>, dimensions> image_used_;
    std::vector<std::uint8_t> sampler_used_;
    std::vector<Pending> pending_;
    std::unordered_map<std::uint64_t, std::size_t> pending_slots_;  // binding and index -> place in pending_
    DescriptorTable table_{64};
    TableBuffers table_buffers_;
    bool table_uploaded_ = false;  // the table's current contents are in table_buffers_.table
    bool table_device_local_ = false;
    bool table_dirty_ = false;    // the table differs from the buffer
    bool existing_entries_changed_ = false;  // ... and not only by entries added
    bool upload_in_flight_ = false;   // a copy is recorded, not yet known to have reached the queue
    bool force_full_upload_ = false;  // the last submit failed: rebuild the whole table on the next flush
    std::vector<TableBuffers> retired_;
};

} // namespace leonida
