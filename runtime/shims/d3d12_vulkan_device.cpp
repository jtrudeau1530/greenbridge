// A Vulkan-backed ID3D12Device (PC ABI) for the Xbox-ABI device proxy (d3d12_xs_device.cpp) to forward to.
// Implemented: queues, allocators, command lists, fences, descriptor heaps, render-target and depth-stencil views,
// committed buffers and textures (2D, multisampled 2D, 3D), allocation info, CPU texture copies (WriteToSubresource/
// ReadFromSubresource), a few feature queries. Everything else is a generated stub that reports itself once. See
// docs/d3d12_vulkan.md section 4.1/4.2 for what each answer means and what is still open.
#include "d3d12_com.h"

#include "d3d12_stubs.h"  // generated
#include "d3d12_com_object.h"
#include "d3d12_xs_device.h"
#include "../src/gfx10_descriptors.h"
#include "../src/vulkan_bindless.h"
#include "../src/resource_tiling.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace rt::d3d12 {
namespace {

// Every descriptor slot is this many bytes; an RTV or DSV slot holds a leonida::TargetView. The console's own
// increments are open question 4 in docs/d3d12_vulkan.md; the application reads them from the device, as it must.
constexpr UINT kDescriptorSize = 32;
static_assert(sizeof(leonida::TargetView) <= kDescriptorSize);

void log_once(const char* what, int value) {
    static std::mutex mutex;
    static std::set<std::pair<std::string, int>> seen;
    std::lock_guard lock(mutex);
    if (seen.insert({what, value}).second) std::fprintf(stderr, "[rt] d3d12: %s %d\n", what, value);
}

struct Format {
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspects = 0;
    bool typeless = false;  // created with MUTABLE_FORMAT so typed views can reinterpret it
};

// DXGI -> Vulkan for the formats render targets, depth buffers and common textures use. Typeless depth families
// become depth formats when the resource allows depth-stencil.
Format vulkan_format(DXGI_FORMAT format, bool depth) {
    constexpr VkImageAspectFlags color = VK_IMAGE_ASPECT_COLOR_BIT, d = VK_IMAGE_ASPECT_DEPTH_BIT,
                                 ds = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return {VK_FORMAT_R32G32B32A32_SFLOAT, color, true};
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return {VK_FORMAT_R32G32B32A32_SFLOAT, color};
    case DXGI_FORMAT_R32G32B32A32_UINT: return {VK_FORMAT_R32G32B32A32_UINT, color};
    case DXGI_FORMAT_R32G32B32A32_SINT: return {VK_FORMAT_R32G32B32A32_SINT, color};
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return {VK_FORMAT_R16G16B16A16_SFLOAT, color, true};
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return {VK_FORMAT_R16G16B16A16_SFLOAT, color};
    case DXGI_FORMAT_R16G16B16A16_UNORM: return {VK_FORMAT_R16G16B16A16_UNORM, color};
    case DXGI_FORMAT_R32G32_FLOAT: return {VK_FORMAT_R32G32_SFLOAT, color};
    case DXGI_FORMAT_R32G32_UINT: return {VK_FORMAT_R32G32_UINT, color};
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return {VK_FORMAT_A2B10G10R10_UNORM_PACK32, color, true};
    case DXGI_FORMAT_R10G10B10A2_UNORM: return {VK_FORMAT_A2B10G10R10_UNORM_PACK32, color};
    case DXGI_FORMAT_R11G11B10_FLOAT: return {VK_FORMAT_B10G11R11_UFLOAT_PACK32, color};
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return {VK_FORMAT_R8G8B8A8_UNORM, color, true};
    case DXGI_FORMAT_R8G8B8A8_UNORM: return {VK_FORMAT_R8G8B8A8_UNORM, color};
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return {VK_FORMAT_R8G8B8A8_SRGB, color};
    case DXGI_FORMAT_R8G8B8A8_SNORM: return {VK_FORMAT_R8G8B8A8_SNORM, color};
    case DXGI_FORMAT_R8G8B8A8_UINT: return {VK_FORMAT_R8G8B8A8_UINT, color};
    case DXGI_FORMAT_R8G8B8A8_SINT: return {VK_FORMAT_R8G8B8A8_SINT, color};
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return {VK_FORMAT_B8G8R8A8_UNORM, color, true};
    case DXGI_FORMAT_B8G8R8A8_UNORM: return {VK_FORMAT_B8G8R8A8_UNORM, color};
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return {VK_FORMAT_B8G8R8A8_SRGB, color};
    case DXGI_FORMAT_R16G16_FLOAT: return {VK_FORMAT_R16G16_SFLOAT, color};
    case DXGI_FORMAT_R8G8_UNORM: return {VK_FORMAT_R8G8_UNORM, color};
    case DXGI_FORMAT_R8G8_SNORM: return {VK_FORMAT_R8G8_SNORM, color};
    case DXGI_FORMAT_R16_UNORM: return {VK_FORMAT_R16_UNORM, color};
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP: return {VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, color};
    // DXGI names packed channels from the high bits down, Vulkan's PACK formats the same way but spelled the other
    // way round: B5G6R5 (blue in bits 0-4) is R5G6B5_PACK16 and B5G5R5A1 (alpha in bit 15) is A1R5G5B5_PACK16.
    case DXGI_FORMAT_B5G6R5_UNORM: return {VK_FORMAT_R5G6B5_UNORM_PACK16, color};
    case DXGI_FORMAT_B5G5R5A1_UNORM: return {VK_FORMAT_A1R5G5B5_UNORM_PACK16, color};
    case DXGI_FORMAT_R16_FLOAT: return {VK_FORMAT_R16_SFLOAT, color};
    case DXGI_FORMAT_R8_UNORM: return {VK_FORMAT_R8_UNORM, color};
    case DXGI_FORMAT_R32_FLOAT: return {VK_FORMAT_R32_SFLOAT, color};
    case DXGI_FORMAT_R32_UINT: return {VK_FORMAT_R32_UINT, color};
    case DXGI_FORMAT_R32_SINT: return {VK_FORMAT_R32_SINT, color};
    case DXGI_FORMAT_BC1_TYPELESS: return {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, color, true};
    case DXGI_FORMAT_BC1_UNORM: return {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, color};
    case DXGI_FORMAT_BC1_UNORM_SRGB: return {VK_FORMAT_BC1_RGBA_SRGB_BLOCK, color};
    case DXGI_FORMAT_BC2_TYPELESS: return {VK_FORMAT_BC2_UNORM_BLOCK, color, true};
    case DXGI_FORMAT_BC2_UNORM: return {VK_FORMAT_BC2_UNORM_BLOCK, color};
    case DXGI_FORMAT_BC2_UNORM_SRGB: return {VK_FORMAT_BC2_SRGB_BLOCK, color};
    case DXGI_FORMAT_BC3_TYPELESS: return {VK_FORMAT_BC3_UNORM_BLOCK, color, true};
    case DXGI_FORMAT_BC3_UNORM: return {VK_FORMAT_BC3_UNORM_BLOCK, color};
    case DXGI_FORMAT_BC3_UNORM_SRGB: return {VK_FORMAT_BC3_SRGB_BLOCK, color};
    case DXGI_FORMAT_BC4_TYPELESS: return {VK_FORMAT_BC4_UNORM_BLOCK, color, true};
    case DXGI_FORMAT_BC4_UNORM: return {VK_FORMAT_BC4_UNORM_BLOCK, color};
    case DXGI_FORMAT_BC4_SNORM: return {VK_FORMAT_BC4_SNORM_BLOCK, color};
    case DXGI_FORMAT_BC5_TYPELESS: return {VK_FORMAT_BC5_UNORM_BLOCK, color, true};
    case DXGI_FORMAT_BC5_UNORM: return {VK_FORMAT_BC5_UNORM_BLOCK, color};
    case DXGI_FORMAT_BC5_SNORM: return {VK_FORMAT_BC5_SNORM_BLOCK, color};
    case DXGI_FORMAT_BC6H_TYPELESS: return {VK_FORMAT_BC6H_UFLOAT_BLOCK, color, true};
    case DXGI_FORMAT_BC6H_UF16: return {VK_FORMAT_BC6H_UFLOAT_BLOCK, color};
    case DXGI_FORMAT_BC6H_SF16: return {VK_FORMAT_BC6H_SFLOAT_BLOCK, color};
    case DXGI_FORMAT_BC7_TYPELESS: return {VK_FORMAT_BC7_UNORM_BLOCK, color, true};
    case DXGI_FORMAT_BC7_UNORM: return {VK_FORMAT_BC7_UNORM_BLOCK, color};
    case DXGI_FORMAT_BC7_UNORM_SRGB: return {VK_FORMAT_BC7_SRGB_BLOCK, color};
    case DXGI_FORMAT_D32_FLOAT: return {VK_FORMAT_D32_SFLOAT, d};
    case DXGI_FORMAT_D16_UNORM: return {VK_FORMAT_D16_UNORM, d};
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return {VK_FORMAT_D32_SFLOAT_S8_UINT, ds};
    case DXGI_FORMAT_D24_UNORM_S8_UINT: return {VK_FORMAT_D24_UNORM_S8_UINT, ds};
    case DXGI_FORMAT_R32_TYPELESS:
        return depth ? Format{VK_FORMAT_D32_SFLOAT, d, true} : Format{VK_FORMAT_R32_SFLOAT, color, true};
    case DXGI_FORMAT_R16_TYPELESS:
        return depth ? Format{VK_FORMAT_D16_UNORM, d, true} : Format{VK_FORMAT_R16_UNORM, color, true};
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        return depth ? Format{VK_FORMAT_D32_SFLOAT_S8_UINT, ds, true} : Format{};
    case DXGI_FORMAT_R24G8_TYPELESS:
        return depth ? Format{VK_FORMAT_D24_UNORM_S8_UINT, ds, true} : Format{};
    default: return {};
    }
}

// The full mip chain's length for a texture this size.
UINT16 full_mips(UINT64 width, UINT height, UINT depth) {
    UINT16 mips = 0;
    for (UINT64 size = std::max<UINT64>({width, height, depth}); size; size >>= 1) ++mips;
    return mips;
}

// The VkImage a texture description becomes, with its mip count already resolved (MipLevels != 0). Shared by
// texture creation and GetResourceAllocationInfo, so the sizes reported are the sizes later allocated.
VkImageCreateInfo texture_create_info(const D3D12_RESOURCE_DESC& desc, const Format& format) {
    const bool volume = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.flags = format.typeless ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
    info.imageType = volume ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    info.format = format.format;
    info.extent = {std::uint32_t(desc.Width), desc.Height, volume ? UINT(desc.DepthOrArraySize) : 1u};
    info.mipLevels = desc.MipLevels;
    info.arrayLayers = volume ? 1 : desc.DepthOrArraySize;
    info.samples = VkSampleCountFlagBits(std::max(1u, desc.SampleDesc.Count));
    info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) info.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (!(desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)) info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    return info;
}

// What a device's children need from it, kept apart from the COM class so children can hold a plain pointer.
struct DeviceState {
    VulkanDeviceInfo info;
    std::unique_ptr<leonida::VulkanContext> context;
    VkPhysicalDeviceMemoryProperties memory{};
    PFN_vkGetPhysicalDeviceFormatProperties format_properties = nullptr;
    PFN_vkGetPhysicalDeviceImageFormatProperties image_format_properties = nullptr;
    PFN_vkGetDeviceImageMemoryRequirements image_requirements = nullptr;  // Vulkan 1.3 or VK_KHR_maintenance4
    PFN_vkGetPhysicalDeviceFormatProperties2 format_properties2 = nullptr;

    // VK_EXT_host_image_copy; null unless the device enabled it and its copies accept GENERAL.
    PFN_vkCopyMemoryToImageEXT copy_memory_to_image = nullptr;
    PFN_vkCopyImageToMemoryEXT copy_image_to_memory = nullptr;
    PFN_vkTransitionImageLayoutEXT transition_image_layout = nullptr;

    // WriteToSubresource/ReadFromSubresource without host image copy: one staging copy at a time, submitted to
    // the device's VkQueue and waited for. Created on first use.
    std::mutex staging_mutex;
    std::unique_ptr<leonida::VulkanQueue> staging_queue;

    bool depth_usable(VkFormat format) const {
        VkFormatProperties properties{};
        format_properties(info.physical, format, &properties);
        return properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }

    // Whether the GPU has this format and usage in this sample count: not every format has every count.
    bool samples_usable(const VkImageCreateInfo& image) const {
        VkImageFormatProperties properties{};
        return image_format_properties(info.physical, image.format, image.imageType, image.tiling, image.usage,
                                       image.flags, &properties) == VK_SUCCESS &&
               (properties.sampleCounts & image.samples);
    }

    // The format a texture description is created with, D24S8's fallback included.
    Format texture_format(const D3D12_RESOURCE_DESC& desc) const {
        Format format = vulkan_format(desc.Format, desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        if (format.format == VK_FORMAT_D24_UNORM_S8_UINT && !depth_usable(format.format)) {
            log_once("D24_UNORM_S8_UINT unavailable, using D32_SFLOAT_S8_UINT; format", desc.Format);
            format.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
        }
        return format;
    }

    // The memory an image made from `create` needs, or false if the device can't make that image.
    bool memory_requirements(const VkImageCreateInfo& create, VkMemoryRequirements& out) const {
        VkImageFormatProperties limits{};
        if (!image_format_properties ||
            image_format_properties(info.physical, create.format, create.imageType, VK_IMAGE_TILING_OPTIMAL,
                                    create.usage, create.flags, &limits) != VK_SUCCESS ||
            create.extent.width > limits.maxExtent.width || create.extent.height > limits.maxExtent.height ||
            create.extent.depth > limits.maxExtent.depth || create.mipLevels > limits.maxMipLevels ||
            create.arrayLayers > limits.maxArrayLayers || !(limits.sampleCounts & create.samples))
            return false;
        if (image_requirements) {
            VkDeviceImageMemoryRequirements query{VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS};
            query.pCreateInfo = &create;
            VkMemoryRequirements2 result{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
            image_requirements(info.device, &query, &result);
            out = result.memoryRequirements;
            return true;
        }
        auto& vk = context->vk;  // without the query, a throwaway image
        VkImage image = VK_NULL_HANDLE;
        if (vk.vkCreateImage(info.device, &create, nullptr, &image) != VK_SUCCESS) return false;
        vk.vkGetImageMemoryRequirements(info.device, image, &out);
        vk.vkDestroyImage(info.device, image, nullptr);
        return true;
    }

    // Set 0 of the translated shaders (vulkan_bindless.h, docs section 3.4) and what has been put in it. A shader
    // finds a T# or S# by its bytes (the descriptor hash table), so one entry serves every heap slot that holds
    // those bytes: a copied descriptor needs nothing more. An image's entries live until its texture is
    // destroyed, samplers until the device is.
    struct ImageEntry {
        leonida::ImageDimension dimension;
        std::uint32_t index;
        VkImageView sampled;
        VkImageView storage;  // the same range without a component mapping, for the storage bindings; or null
        std::uint64_t address;
    };
    std::unique_ptr<leonida::BindlessSet> bindless;
    std::mutex descriptor_mutex;  // the set and the containers below
    std::map<leonida::DescriptorKey, ImageEntry> image_entries;
    std::multimap<std::uint64_t, leonida::DescriptorKey> image_keys;  // by the texture's address
    std::vector<ImageEntry> replaced_images;  // kept until their texture goes: work in flight may still use them
    std::map<leonida::DescriptorKey, VkSampler> sampler_entries;
    // A T# names a texture by an address of its own: above 4 GiB, which the GPU address arena for buffers (docs
    // section 3.3) stays below, and 256 bytes apart (the T# keeps addresses in 256-byte units). Nothing
    // dereferences it.
    std::atomic<std::uint64_t> next_image_address{std::uint64_t(1) << 32};

    // 0 float, 1 unsigned integer, 2 signed integer: which of the three storage-image arrays a view belongs in.
    static int numeric_class(VkFormat format) {
        switch (format) {
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8G8_UINT:
        case VK_FORMAT_R8G8B8A8_UINT:
        case VK_FORMAT_R16_UINT:
        case VK_FORMAT_R16G16_UINT:
        case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R32G32_UINT:
        case VK_FORMAT_R32G32B32A32_UINT:
        case VK_FORMAT_A2B10G10R10_UINT_PACK32: return 1;
        case VK_FORMAT_R8_SINT:
        case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R8G8B8A8_SINT:
        case VK_FORMAT_R16_SINT:
        case VK_FORMAT_R16G16_SINT:
        case VK_FORMAT_R16G16B16A16_SINT:
        case VK_FORMAT_R32_SINT:
        case VK_FORMAT_R32G32_SINT:
        case VK_FORMAT_R32G32B32A32_SINT: return 2;
        default: return 0;
        }
    }

    void release(const ImageEntry& entry) {  // descriptor_mutex held
        bindless->remove_image(entry.dimension, entry.index);
        context->vk.vkDestroyImageView(info.device, entry.sampled, nullptr);
        if (entry.storage) context->vk.vkDestroyImageView(info.device, entry.storage, nullptr);
    }

    // Gives the T# `words` a view in set 0 and an entry in the hash table, unless it has them. `storage`: shaders
    // write through it too (an unordered-access view), so it also goes into the storage arrays, which are
    // shorter. False when set 0 cannot show the image; the descriptor then finds the fallback at index 0.
    bool bind_image(const leonida::gfx10::ImageWords& words, const leonida::gfx10::Image& image, VkImage vk_image,
                    VkFormat image_format, VkImageAspectFlags aspects, bool storage) {
        namespace g = leonida::gfx10;
        std::lock_guard lock(descriptor_mutex);
        if (!bindless) return false;
        const auto existing = image_entries.find(words);
        if (existing != image_entries.end() && (existing->second.storage || !storage)) return true;

        leonida::ImageDimension dimension;
        switch (image.type) {
        case g::ImageType::image_2d: dimension = leonida::ImageDimension::dim_2d; break;
        case g::ImageType::image_2d_array:
        case g::ImageType::cube: dimension = leonida::ImageDimension::dim_2d_array; break;
        case g::ImageType::image_3d: dimension = leonida::ImageDimension::dim_3d; break;
        case g::ImageType::image_1d: dimension = leonida::ImageDimension::dim_1d; break;
        default:
            log_once("set 0 has no array for this image type (1D array or multisampled):", int(image.type));
            return false;
        }
        const VkFormat view_format = g::vulkan_view_format(image_format, image.format);
        if (view_format == VK_FORMAT_UNDEFINED) {
            log_once("no Vulkan view format for the GFX10 format of a view:", int(image.format));
            return false;
        }
        auto& vk = context->vk;
        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = vk_image;
        view_info.viewType = g::vulkan_view_type(image.type);
        view_info.format = view_format;
        view_info.components = g::vulkan_swizzle(image.swizzle, view_format);
        // The cast: MSVC's enums are ints, and it rejects int to VkImageAspectFlags in braces as narrowing.
        view_info.subresourceRange = {
            VkImageAspectFlags((aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                                     : VK_IMAGE_ASPECT_COLOR_BIT),
            image.first_mip, image.mip_count, image.first_layer, image.layer_count};
        ImageEntry entry{dimension, 0, VK_NULL_HANDLE, VK_NULL_HANDLE, image.address};
        if (vk.vkCreateImageView(info.device, &view_info, nullptr, &entry.sampled) != VK_SUCCESS) return false;
        leonida::StorageViews views{};
        if (storage) {
            view_info.components = {};
            if (vk.vkCreateImageView(info.device, &view_info, nullptr, &entry.storage) != VK_SUCCESS) {
                vk.vkDestroyImageView(info.device, entry.sampled, nullptr);
                return false;
            }
            const int numeric = numeric_class(view_format);
            (numeric == 1 ? views.as_uint : numeric == 2 ? views.as_sint : views.as_float) = entry.storage;
            if (view_format == VK_FORMAT_R32_UINT) views.r32ui = entry.storage;
        }
        // Every state a shader can read or write the texture in is GENERAL (ResourceBinding::sampled_in_general).
        entry.index = bindless->add_image(dimension, entry.sampled, VK_IMAGE_LAYOUT_GENERAL, views);
        if (!entry.index) {
            log_once("a bindless image array is full; dimension", int(dimension));
            vk.vkDestroyImageView(info.device, entry.sampled, nullptr);
            if (entry.storage) vk.vkDestroyImageView(info.device, entry.storage, nullptr);
            return false;
        }
        if (existing != image_entries.end()) {
            replaced_images.push_back(existing->second);  // the read-only entry of the same bytes
            existing->second = entry;
        } else {
            image_entries.emplace(words, entry);
            image_keys.emplace(image.address, words);
        }
        bindless->insert_key(words, entry.index);
        return true;
    }

    // The texture at `address` is gone: its views leave set 0 and its T#s the hash table.
    void forget_image(std::uint64_t address) {
        std::lock_guard lock(descriptor_mutex);
        if (!bindless) return;
        const auto [first, last] = image_keys.equal_range(address);
        for (auto it = first; it != last; ++it) {
            const auto entry = image_entries.find(it->second);
            bindless->erase_key(it->second);
            release(entry->second);
            image_entries.erase(entry);
        }
        image_keys.erase(first, last);
        std::erase_if(replaced_images, [&](const ImageEntry& entry) {
            if (entry.address != address) return false;
            release(entry);
            return true;
        });
    }

    // Gives the S# `words` a sampler in set 0 and an entry in the hash table, unless it has them.
    bool bind_sampler(const leonida::gfx10::SamplerWords& words) {
        namespace g = leonida::gfx10;
        std::lock_guard lock(descriptor_mutex);
        if (!bindless) return false;
        const leonida::DescriptorKey key = leonida::make_ssharp_key(words.data());
        if (sampler_entries.count(key)) return true;
        const g::Sampler sampler = g::decode(words);
        VkSamplerCreateInfo sampler_info = g::vulkan_sampler(sampler);
        // What vulkan_sampler leaves to its caller, and features this device may not have enabled.
        if (sampler_info.anisotropyEnable && !info.sampler_anisotropy) {
            log_once("anisotropic sampler without samplerAnisotropy; anisotropy", int(sampler.max_anisotropy));
            sampler_info.anisotropyEnable = VK_FALSE;
        }
        for (VkSamplerAddressMode* mode :
             {&sampler_info.addressModeU, &sampler_info.addressModeV, &sampler_info.addressModeW}) {
            if (*mode != VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE || info.sampler_mirror_clamp_to_edge) continue;
            log_once("mirror-once sampler without samplerMirrorClampToEdge; clamped instead, mode", int(*mode));
            *mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        }
        if (sampler.filter_mode != g::FilterMode::blend)
            log_once("minimum/maximum filter samplers blend for now; filter mode", int(sampler.filter_mode));
        if (sampler.border_color == g::BorderColor::table)
            log_once("custom sampler border colours are transparent black for now; index",
                     int(sampler.border_color_index));
        VkSampler vk_sampler = VK_NULL_HANDLE;
        if (!context->vk.vkCreateSampler ||
            context->vk.vkCreateSampler(info.device, &sampler_info, nullptr, &vk_sampler) != VK_SUCCESS)
            return false;
        const std::uint32_t index = bindless->add_sampler(vk_sampler);
        if (!index) {
            log_once("the bindless sampler array is full; samplers", int(sampler_entries.size()));
            context->vk.vkDestroySampler(info.device, vk_sampler, nullptr);
            return false;
        }
        sampler_entries.emplace(key, vk_sampler);
        bindless->insert_key(key, index);
        return true;
    }

    // An optimal-tiling image of this format can take VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT.
    bool host_transfer_format(VkFormat format) const {
        if (!copy_memory_to_image || !format_properties2) return false;
        VkFormatProperties3 properties3{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
        VkFormatProperties2 properties{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &properties3};
        format_properties2(info.physical, format, &properties);
        return properties3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT;
    }

    // The first memory type with `required`, preferring one that also has `preferred`.
    VkDeviceMemory allocate(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags required,
                            VkMemoryPropertyFlags preferred) const {
        for (const VkMemoryPropertyFlags flags : {required | preferred, required}) {
            for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
                if (!(requirements.memoryTypeBits & (1u << i))) continue;
                if ((memory.memoryTypes[i].propertyFlags & flags) != flags) continue;
                VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocate.allocationSize = requirements.size;
                allocate.memoryTypeIndex = i;
                VkDeviceMemory out = VK_NULL_HANDLE;
                if (context->vk.vkAllocateMemory(info.device, &allocate, nullptr, &out) == VK_SUCCESS) return out;
            }
        }
        return VK_NULL_HANDLE;
    }
};

class DescriptorHeap final
    : public Child<DescriptorHeap, ID3D12DescriptorHeapStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable,
                   ID3D12DescriptorHeap> {
public:
    // {4a7d2c90-5e1b-4f63-8c0a-91d3b6e2f417}
    static constexpr GUID kTag{0x4a7d2c90, 0x5e1b, 0x4f63, {0x8c, 0x0a, 0x91, 0xd3, 0xb6, 0xe2, 0xf4, 0x17}};

    explicit DescriptorHeap(const D3D12_DESCRIPTOR_HEAP_DESC& desc)
        : desc_(desc), storage_(new std::byte[std::size_t(desc.NumDescriptors) * kDescriptorSize]()) {}

    D3D12_DESCRIPTOR_HEAP_DESC STDMETHODCALLTYPE GetDesc() override { return desc_; }
    D3D12_CPU_DESCRIPTOR_HANDLE STDMETHODCALLTYPE GetCPUDescriptorHandleForHeapStart() override {
        return {reinterpret_cast<SIZE_T>(storage_.get())};
    }
    // Shader-visible heaps get a GPU handle; descriptor tables that read it are not implemented yet.
    D3D12_GPU_DESCRIPTOR_HANDLE STDMETHODCALLTYPE GetGPUDescriptorHandleForHeapStart() override {
        if (!(desc_.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)) return {0};
        return {reinterpret_cast<UINT64>(storage_.get())};
    }

private:
    D3D12_DESCRIPTOR_HEAP_DESC desc_;
    std::unique_ptr<std::byte[]> storage_;
};

class Resource final
    : public Child<Resource, ID3D12ResourceStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable, ID3D12Resource> {
public:
    // {9b3e5f12-7c48-4d0e-a6b1-2f8d4c7e9a35}
    static constexpr GUID kTag{0x9b3e5f12, 0x7c48, 0x4d0e, {0xa6, 0xb1, 0x2f, 0x8d, 0x4c, 0x7e, 0x9a, 0x35}};

    Resource(DeviceState& device, const D3D12_RESOURCE_DESC& desc, const D3D12_HEAP_PROPERTIES& heap,
             D3D12_HEAP_FLAGS heap_flags)
        : device_(device), desc_(desc), heap_(heap), heap_flags_(heap_flags) {}

    ~Resource() override {
        auto& vk = device_.context->vk;
        const VkDevice dev = device_.info.device;
        if (image_address_) device_.forget_image(image_address_);
        device_.context->remove_resource(id());
        if (image_) vk.vkDestroyImage(dev, image_, nullptr);
        if (buffer_) vk.vkDestroyBuffer(dev, buffer_, nullptr);
        if (memory_) vk.vkFreeMemory(dev, memory_, nullptr);
    }

    leonida::ResourceId id() const { return resource_id(this); }
    VkImage image() const { return image_; }
    const D3D12_RESOURCE_DESC& desc() const { return desc_; }
    VkFormat format() const { return format_; }                  // of the image
    VkImageAspectFlags aspects() const { return aspects_; }
    std::uint64_t image_address() const { return image_address_; }  // what a T# names the texture by

    HRESULT create_buffer() {
        auto& vk = device_.context->vk;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = desc_.Width;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        if (device_.info.conditional_rendering) info.usage |= VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
        if (vk.vkCreateBuffer(device_.info.device, &info, nullptr, &buffer_) != VK_SUCCESS) return E_OUTOFMEMORY;
        VkMemoryRequirements requirements{};
        vk.vkGetBufferMemoryRequirements(device_.info.device, buffer_, &requirements);
        const bool host = cpu_visible();
        memory_ = host ? device_.allocate(requirements,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                          heap_.Type == D3D12_HEAP_TYPE_READBACK ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0)
                       : device_.allocate(requirements, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!memory_ || vk.vkBindBufferMemory(device_.info.device, buffer_, memory_, 0) != VK_SUCCESS)
            return E_OUTOFMEMORY;
        if (host) {
            if (vk.vkMapMemory(device_.info.device, memory_, 0, VK_WHOLE_SIZE, 0, &mapped_) != VK_SUCCESS)
                return E_OUTOFMEMORY;
            std::memset(mapped_, 0, std::size_t(desc_.Width));  // committed resources start zeroed
        } else {
            log_once("default-heap buffers are not zero-filled yet; dimension", int(desc_.Dimension));
        }
        leonida::ResourceBinding binding{};  // a buffer: no image
        binding.buffer = buffer_;
        binding.size = desc_.Width;
        device_.context->add_resource(id(), binding);
        return S_OK;
    }

    HRESULT create_texture(const Format& format) {
        auto& vk = device_.context->vk;
        const bool volume = desc_.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        const UINT depth = volume ? desc_.DepthOrArraySize : 1;
        if (desc_.MipLevels == 0) desc_.MipLevels = full_mips(desc_.Width, desc_.Height, depth);
        VkImageCreateInfo info = texture_create_info(desc_, format);
        const std::uint32_t samples = info.samples;
        // Textures the CPU reads and writes (WriteToSubresource/ReadFromSubresource) copy on the host when they can.
        host_transfer_ = cpu_visible() && samples == 1 && device_.host_transfer_format(format.format);
        if (host_transfer_) info.usage |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
        if (samples > 1 && !device_.samples_usable(info)) {
            log_once("sample count not supported by the GPU for this format:", int(samples));
            return E_INVALIDARG;
        }
        if (vk.vkCreateImage(device_.info.device, &info, nullptr, &image_) != VK_SUCCESS) return E_OUTOFMEMORY;
        VkMemoryRequirements requirements{};
        vk.vkGetImageMemoryRequirements(device_.info.device, image_, &requirements);
        memory_ = device_.allocate(requirements, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!memory_ || vk.vkBindImageMemory(device_.info.device, image_, memory_, 0) != VK_SUCCESS)
            return E_OUTOFMEMORY;

        leonida::ResourceBinding binding{};
        binding.image = image_;
        binding.rgba8_typeless = desc_.Format == DXGI_FORMAT_R8G8B8A8_TYPELESS;
        binding.format = format.format;
        binding.aspects = format.aspects;
        binding.width = std::uint32_t(desc_.Width);
        binding.height = desc_.Height;
        binding.mip_levels = desc_.MipLevels;
        binding.array_layers = info.arrayLayers;
        binding.depth = depth;
        binding.volume = volume;
        binding.samples = samples;
        binding.simultaneous_access = (desc_.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS) != 0;
        binding.not_zeroed = (heap_flags_ & D3D12_HEAP_FLAG_CREATE_NOT_ZEROED) != 0;
        // With the bindless set, a view's descriptor has one layout for as long as it lives (bind_image).
        binding.sampled_in_general = device_.bindless != nullptr;
        format_ = format.format;
        aspects_ = format.aspects;
        image_address_ = device_.next_image_address.fetch_add(256);
        binding_ = binding;
        if (host_transfer_) {
            // Moved to GENERAL and zero-filled on the host now, so the CPU can copy into it before any command
            // list has touched it (a first GPU use would otherwise zero-fill over what the CPU wrote).
            if (!host_initialize()) return E_OUTOFMEMORY;
            device_.context->add_resource(id(), binding, VK_IMAGE_LAYOUT_GENERAL);
            return S_OK;
        }
        // Never used yet: the first command that touches a part zero-fills it and moves it to the layout of the
        // state that command expects (vulkan_replay.h, initialize), so no submission is needed here.
        device_.context->add_resource(id(), binding, VK_IMAGE_LAYOUT_UNDEFINED);
        return S_OK;
    }

    D3D12_RESOURCE_DESC STDMETHODCALLTYPE GetDesc() override { return desc_; }

    // A reserved resource is a committed one underneath (docs/d3d12_vulkan.md section 4.2); it reports the layout it
    // was created with and has no heap of its own.
    void make_reserved(D3D12_TEXTURE_LAYOUT layout) {
        reserved_ = true;
        desc_.Layout = layout;
    }
    bool reserved() const { return reserved_; }

    // D3D12 answers only for committed and placed resources (E_INVALIDARG for reserved ones, measured on the PC
    // runtime).
    HRESULT STDMETHODCALLTYPE GetHeapProperties(D3D12_HEAP_PROPERTIES* properties, D3D12_HEAP_FLAGS* flags) override {
        if (reserved_) return E_INVALIDARG;
        if (properties) *properties = heap_;
        if (flags) *flags = heap_flags_;
        return S_OK;
    }

    // Buffers on upload and readback heaps are mapped persistently; the ranges are hints, as on D3D12. A texture
    // in a CPU-visible heap has no pointer to give (its layout is the driver's), so D3D12 only lets it be mapped
    // with a null pointer, before WriteToSubresource/ReadFromSubresource.
    HRESULT STDMETHODCALLTYPE Map(UINT subresource, const D3D12_RANGE*, void** data) override {
        if (image_) {
            VkImageSubresourceRange range{};
            return !data && cpu_visible() && subresource != leonida::all_subresources &&
                           leonida::image_range(binding_, subresource, range)
                       ? S_OK
                       : E_INVALIDARG;
        }
        if (!mapped_ || subresource != 0) return E_INVALIDARG;
        if (data) *data = mapped_;
        return S_OK;
    }
    void STDMETHODCALLTYPE Unmap(UINT, const D3D12_RANGE*) override {}

    // CPU copies into and out of a texture subresource. D3D12 completes them before returning and leaves it to
    // the application to keep the GPU off the subresource meanwhile, so a blocking copy is the whole contract.
    HRESULT STDMETHODCALLTYPE WriteToSubresource(UINT subresource, const D3D12_BOX* box, const void* data,
                                                 UINT row_pitch, UINT depth_pitch) override {
        return cpu_copy(true, subresource, box, const_cast<void*>(data), row_pitch, depth_pitch);
    }
    HRESULT STDMETHODCALLTYPE ReadFromSubresource(void* data, UINT row_pitch, UINT depth_pitch, UINT subresource,
                                                  const D3D12_BOX* box) override {
        return cpu_copy(false, subresource, box, data, row_pitch, depth_pitch);
    }

private:
    // One subresource region in copy units: rows of texels, or of 4x4 blocks for block-compressed formats.
    struct Region {
        VkImageSubresourceLayers layers;
        D3D12_BOX box;
        leonida::BlockFormat block;
        std::uint32_t rows;
        std::uint64_t row_bytes;
        std::uint32_t slices;
    };

    // The region a copy names, or false when D3D12 would reject it: an unknown subresource, a box outside the
    // mip, or one that cuts a compressed block anywhere but at the mip edge.
    bool region(UINT subresource, const D3D12_BOX* box, Region& out) const {
        VkImageSubresourceRange range{};
        if (subresource == leonida::all_subresources || !leonida::image_range(binding_, subresource, range)) return false;
        if (binding_.samples > 1) return false;  // D3D12 has no CPU copies of multisampled textures; Vulkan neither
        const std::uint32_t mip = range.baseMipLevel;
        const std::uint32_t width = std::max(1u, binding_.width >> mip), height = std::max(1u, binding_.height >> mip);
        const std::uint32_t depth = binding_.volume ? std::max(1u, binding_.depth >> mip) : 1;
        out.box = box ? *box : D3D12_BOX{0, 0, 0, width, height, depth};
        const D3D12_BOX& b = out.box;
        if (b.left >= b.right || b.top >= b.bottom || b.front >= b.back || b.right > width || b.bottom > height ||
            b.back > depth)
            return false;
        out.block = leonida::block_format(binding_.format, range.aspectMask);
        if (!out.block.bytes) return false;
        const std::uint32_t k = out.block.block;
        if (b.left % k || b.top % k || (b.right % k && b.right != width) || (b.bottom % k && b.bottom != height))
            return false;
        out.layers = {range.aspectMask, mip, range.baseArrayLayer, 1};
        out.rows = (b.bottom - b.top + k - 1) / k;
        out.row_bytes = std::uint64_t((b.right - b.left + k - 1) / k) * out.block.bytes;
        out.slices = b.back - b.front;
        return true;
    }

    HRESULT cpu_copy(bool write, UINT subresource, const D3D12_BOX* box, void* data, UINT row_pitch,
                     UINT depth_pitch) {
        if (!image_ || !data) return E_INVALIDARG;
        Region r{};
        if (!region(subresource, box, r)) return E_INVALIDARG;
        // The application's rows and slices must not overlap.
        if ((r.rows > 1 && row_pitch < r.row_bytes) ||
            (r.slices > 1 && depth_pitch < std::uint64_t(r.rows - 1) * row_pitch + r.row_bytes))
            return E_INVALIDARG;
        if (!cpu_visible())
            log_once(write ? "WriteToSubresource on a texture outside a CPU-visible heap; heap type"
                           : "ReadFromSubresource on a texture outside a CPU-visible heap; heap type",
                     heap_.Type);
        auto* bytes = static_cast<std::byte*>(data);
        // Between the application's pitches and `pitch`, the one a copy reads or writes.
        const auto pack = [&](std::byte* to, std::uint64_t pitch) {
            for (std::uint32_t z = 0; z < r.slices; ++z)
                for (std::uint32_t y = 0; y < r.rows; ++y)
                    std::memcpy(to + (std::uint64_t(z) * r.rows + y) * pitch,
                                bytes + std::uint64_t(z) * depth_pitch + std::uint64_t(y) * row_pitch, r.row_bytes);
        };
        const auto unpack = [&](const std::byte* from, std::uint64_t pitch) {
            for (std::uint32_t z = 0; z < r.slices; ++z)
                for (std::uint32_t y = 0; y < r.rows; ++y)
                    std::memcpy(bytes + std::uint64_t(z) * depth_pitch + std::uint64_t(y) * row_pitch,
                                from + (std::uint64_t(z) * r.rows + y) * pitch, r.row_bytes);
        };
        // On the host only while the subresource is in GENERAL: it is from creation, and stays there until a
        // barrier moves it (D3D12's COMMON is GENERAL). Any other layout goes through the staging copy, which
        // moves it as a CopyTextureRegion would.
        if (host_transfer_ && device_.context->layout(id(), subresource) == VK_IMAGE_LAYOUT_GENERAL) {
            std::vector<std::byte> packed(r.row_bytes * r.rows * r.slices);
            if (write) pack(packed.data(), r.row_bytes);
            if (!host_copy(write, r, packed.data())) return E_FAIL;
            if (!write) unpack(packed.data(), r.row_bytes);
            return S_OK;
        }
        return staged_copy(write, subresource, r, pack, unpack);
    }

    // Tightly packed `memory` to or from the region, with VK_EXT_host_image_copy.
    bool host_copy(bool write, const Region& r, void* memory) {
        const VkOffset3D offset{std::int32_t(r.box.left), std::int32_t(r.box.top), std::int32_t(r.box.front)};
        const VkExtent3D extent{r.box.right - r.box.left, r.box.bottom - r.box.top, r.slices};
        const VkDevice dev = device_.info.device;
        if (write) {
            VkMemoryToImageCopyEXT copy{VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT};
            copy.pHostPointer = memory;
            copy.imageSubresource = r.layers;
            copy.imageOffset = offset;
            copy.imageExtent = extent;
            VkCopyMemoryToImageInfoEXT info{VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT};
            info.dstImage = image_;
            info.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
            info.regionCount = 1;
            info.pRegions = &copy;
            return device_.copy_memory_to_image(dev, &info) == VK_SUCCESS;
        }
        VkImageToMemoryCopyEXT copy{VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY_EXT};
        copy.pHostPointer = memory;
        copy.imageSubresource = r.layers;
        copy.imageOffset = offset;
        copy.imageExtent = extent;
        VkCopyImageToMemoryInfoEXT info{VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO_EXT};
        info.srcImage = image_;
        info.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        info.regionCount = 1;
        info.pRegions = &copy;
        return device_.copy_image_to_memory(dev, &info) == VK_SUCCESS;
    }

    // Every subresource of a new host-transfer image to GENERAL, zero-filled unless the heap said not to.
    bool host_initialize() {
        const VkDevice dev = device_.info.device;
        VkHostImageLayoutTransitionInfoEXT transition{VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT};
        transition.image = image_;
        transition.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        transition.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        transition.subresourceRange = {binding_.aspects, 0, binding_.mip_levels, 0, binding_.array_layers};
        if (device_.transition_image_layout(dev, 1, &transition) != VK_SUCCESS) return false;
        if (binding_.not_zeroed) return true;
        const std::uint32_t planes = leonida::two_planes(binding_) ? 2 : 1;
        const std::uint32_t per_plane = binding_.mip_levels * binding_.array_layers;
        std::vector<std::byte> zero;
        for (UINT subresource = 0; subresource < planes * per_plane; ++subresource) {
            Region r{};
            if (!region(subresource, nullptr, r)) return false;
            zero.assign(r.row_bytes * r.rows * r.slices, std::byte{0});
            if (!host_copy(true, r, zero.data())) return false;
        }
        return true;
    }

    // A host-visible buffer at D3D12's footprint pitch, copied to or from the subresource by a CopyTextureRegion
    // replayed on the device's queue, then waited for. The replay handles the layout (and a first use's zero fill)
    // exactly as it does for the application's own copies.
    template<class Pack, class Unpack>
    HRESULT staged_copy(bool write, UINT subresource, const Region& r, const Pack& pack, const Unpack& unpack) {
        auto& vk = device_.context->vk;
        const VkDevice dev = device_.info.device;
        const std::uint64_t pitch = (r.row_bytes + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) /
                                    D3D12_TEXTURE_DATA_PITCH_ALIGNMENT * D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
        leonida::ResourceBinding staging{};
        staging.size = pitch * r.rows * r.slices;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = staging.size;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vk.vkCreateBuffer(dev, &info, nullptr, &staging.buffer) != VK_SUCCESS) return E_OUTOFMEMORY;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        const auto release = [&](HRESULT hr) {
            if (memory) vk.vkFreeMemory(dev, memory, nullptr);  // unmaps it
            vk.vkDestroyBuffer(dev, staging.buffer, nullptr);
            return hr;
        };
        VkMemoryRequirements requirements{};
        vk.vkGetBufferMemoryRequirements(dev, staging.buffer, &requirements);
        memory = device_.allocate(requirements, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  write ? 0 : VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (!memory || vk.vkBindBufferMemory(dev, staging.buffer, memory, 0) != VK_SUCCESS ||
            vk.vkMapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
            return release(E_OUTOFMEMORY);
        if (write) pack(static_cast<std::byte*>(mapped), pitch);

        // The staging buffer is a resource only for the length of this call; its ID is its binding's address,
        // which no COM resource can share.
        const auto staging_id = reinterpret_cast<leonida::ResourceId>(&staging);
        const std::uint32_t k = r.block.block;
        const leonida::CopyLocation texture{id(), false, subresource, 0, 0, 0, 0, 0};
        const leonida::CopyLocation placed{staging_id, true, 0, 0, std::uint32_t(r.row_bytes / r.block.bytes) * k,
                                           r.rows * k, r.slices, std::uint32_t(pitch)};
        leonida::CopyTexture copy{};
        if (write) {
            copy = {texture, r.box.left, r.box.top, r.box.front, placed, false, {}};
        } else {
            copy = {placed, 0, 0, 0, texture, true,
                    {r.box.left, r.box.top, r.box.front, r.box.right, r.box.bottom, r.box.back}};
        }
        leonida::CommandList list;
        if (!list.copy_texture(copy) || !list.close()) return release(E_INVALIDARG);
        device_.context->add_resource(staging_id, staging);
        leonida::ReplayResult result = leonida::ReplayResult::unsupported;
        {
            std::lock_guard lock(device_.staging_mutex);
            if (!device_.staging_queue) {
                device_.staging_queue = std::make_unique<leonida::VulkanQueue>(*device_.context, device_.info.queue,
                                                                               device_.info.queue_family);
            }
            if (device_.staging_queue->ok()) {
                const leonida::CommandList* lists[] = {&list};
                result = device_.staging_queue->execute(lists);
                if (result == leonida::ReplayResult::success && !device_.staging_queue->idle())
                    result = leonida::ReplayResult::unsupported;
            }
        }
        device_.context->remove_resource(staging_id);
        if (result != leonida::ReplayResult::success) {
            log_once("WriteToSubresource/ReadFromSubresource staging copy failed; replay result", int(result));
            return release(E_FAIL);
        }
        if (!write) unpack(static_cast<const std::byte*>(mapped), pitch);
        return release(S_OK);
    }

    bool cpu_visible() const {
        if (heap_.Type == D3D12_HEAP_TYPE_UPLOAD || heap_.Type == D3D12_HEAP_TYPE_READBACK) return true;
        return heap_.Type == D3D12_HEAP_TYPE_CUSTOM && heap_.CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE &&
               heap_.CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    }

    DeviceState& device_;
    D3D12_RESOURCE_DESC desc_;
    D3D12_HEAP_PROPERTIES heap_;
    D3D12_HEAP_FLAGS heap_flags_;
    VkImage image_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    void* mapped_ = nullptr;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspects_ = 0;
    std::uint64_t image_address_ = 0;
    bool reserved_ = false;
    leonida::ResourceBinding binding_{};  // a texture's, as registered with the context
    bool host_transfer_ = false;          // created with VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT
};

// A view of a texture as Create*View describes it, before the resource's own shape is applied.
struct TextureView {
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;  // UNKNOWN: the resource's
    leonida::gfx10::ImageType type = leonida::gfx10::ImageType::image_2d;
    UINT first_mip = 0;
    UINT mip_count = ~0u;  // ~0: to the last
    UINT first_layer = 0;
    UINT layer_count = ~0u;
    UINT mapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    UINT plane = 0;
    float min_lod = 0.0f;
};

// The T# of a view: what the shader gets to know about it, in the structure the console's shaders read.
// False (with a log) for a view set 0 cannot show yet.
bool image_descriptor(const Resource& resource, const TextureView& view, leonida::gfx10::Image& out) {
    namespace g = leonida::gfx10;
    const D3D12_RESOURCE_DESC& desc = resource.desc();
    if (view.plane != 0) {
        log_once("views of the stencil plane are not supported; plane", int(view.plane));
        return false;
    }
    // Every view here is a 2D one (plain, array or cube), which a volume or a multisampled image cannot back.
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count > 1) {
        log_once("shader views of 3D and multisampled textures are not supported; dimension", int(desc.Dimension));
        return false;
    }
    VkFormat named = resource.format();
    if (view.format != DXGI_FORMAT_UNKNOWN) named = vulkan_format(view.format, false).format;
    const std::uint32_t code = g::format_from_vulkan(named);
    const VkFormat view_format = code ? g::vulkan_view_format(resource.format(), code) : VK_FORMAT_UNDEFINED;
    if (view_format == VK_FORMAT_UNDEFINED) {
        log_once("view format not supported for this texture:", int(view.format));
        return false;
    }
    if (view.first_mip >= desc.MipLevels || view.first_layer >= desc.DepthOrArraySize) {
        log_once("view starts past the texture's mips or layers; first mip", int(view.first_mip));
        return false;
    }
    out = {};
    out.address = resource.image_address();
    out.format = code;
    out.type = view.type;
    out.width = std::uint32_t(desc.Width);
    out.height = desc.Height;
    out.first_mip = view.first_mip;
    out.mip_count = std::min<UINT>(view.mip_count, desc.MipLevels - view.first_mip);
    out.first_layer = view.first_layer;
    out.layer_count = std::min<UINT>(view.layer_count, desc.DepthOrArraySize - view.first_layer);
    if (view.type == g::ImageType::cube) out.layer_count -= out.layer_count % 6;  // whole cubes
    out.resource_mips = desc.MipLevels;
    out.min_lod = view.min_lod;
    if (!out.mip_count || !out.layer_count) {
        log_once("view of no mips or no layers; type", int(view.type));
        return false;
    }
    // D3D12's component mapping picks, for each of R, G, B, A, a channel of the format or a constant; the T#
    // says the same about the hardware's channels, which the format places.
    const g::Swizzle channels = g::format_swizzle(view_format);
    for (UINT i = 0; i < 4; ++i) {
        const UINT from = D3D12_DECODE_SHADER_4_COMPONENT_MAPPING(i, view.mapping);
        out.swizzle[i] = from < 4 ? channels[from]
                                  : from == D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0 ? g::select_zero : g::select_one;
    }
    return true;
}

// The S# of a D3D12 sampler.
leonida::gfx10::Sampler sampler_descriptor(const D3D12_SAMPLER_DESC& desc) {
    namespace g = leonida::gfx10;
    const auto address = [](D3D12_TEXTURE_ADDRESS_MODE mode) {
        switch (mode) {
        case D3D12_TEXTURE_ADDRESS_MODE_MIRROR: return g::AddressMode::mirror;
        case D3D12_TEXTURE_ADDRESS_MODE_CLAMP: return g::AddressMode::clamp_last_texel;
        case D3D12_TEXTURE_ADDRESS_MODE_BORDER: return g::AddressMode::clamp_border;
        case D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE: return g::AddressMode::mirror_once_last_texel;
        default: return g::AddressMode::wrap;
        }
    };
    const bool anisotropic = D3D12_DECODE_IS_ANISOTROPIC_FILTER(desc.Filter);
    const auto filter = [&](D3D12_FILTER_TYPE type) {
        if (anisotropic) return type == D3D12_FILTER_TYPE_LINEAR ? g::Filter::aniso_bilinear : g::Filter::aniso_point;
        return type == D3D12_FILTER_TYPE_LINEAR ? g::Filter::bilinear : g::Filter::point;
    };
    g::Sampler out{};
    out.address_u = address(desc.AddressU);
    out.address_v = address(desc.AddressV);
    out.address_w = address(desc.AddressW);
    out.mag_filter = filter(D3D12_DECODE_MAG_FILTER(desc.Filter));
    out.min_filter = filter(D3D12_DECODE_MIN_FILTER(desc.Filter));
    out.mip_filter = D3D12_DECODE_MIP_FILTER(desc.Filter) == D3D12_FILTER_TYPE_LINEAR ? g::MipFilter::linear
                                                                                    : g::MipFilter::point;
    out.max_anisotropy = anisotropic ? std::clamp<UINT>(desc.MaxAnisotropy, 1, 16) : 1;
    const D3D12_FILTER_REDUCTION_TYPE reduction = D3D12_DECODE_FILTER_REDUCTION(desc.Filter);
    // D3D12's comparison functions are 1 (NEVER) to 8 (ALWAYS); the S#'s are 0 to 7 in the same order.
    if (reduction == D3D12_FILTER_REDUCTION_TYPE_COMPARISON && desc.ComparisonFunc >= D3D12_COMPARISON_FUNC_NEVER)
        out.compare = g::CompareFunction(desc.ComparisonFunc - D3D12_COMPARISON_FUNC_NEVER);
    if (reduction == D3D12_FILTER_REDUCTION_TYPE_MINIMUM) out.filter_mode = g::FilterMode::minimum;
    if (reduction == D3D12_FILTER_REDUCTION_TYPE_MAXIMUM) out.filter_mode = g::FilterMode::maximum;
    out.min_lod = desc.MinLOD;
    out.max_lod = desc.MaxLOD;
    out.lod_bias = desc.MipLODBias;
    const FLOAT* border = desc.BorderColor;
    const bool black = border[0] == 0 && border[1] == 0 && border[2] == 0;
    if (black && border[3] == 0) {
        out.border_color = g::BorderColor::transparent_black;
    } else if (black && border[3] == 1) {
        out.border_color = g::BorderColor::opaque_black;
    } else if (border[0] == 1 && border[1] == 1 && border[2] == 1 && border[3] == 1) {
        out.border_color = g::BorderColor::opaque_white;
    } else {
        out.border_color = g::BorderColor::table;  // a colour of the application's: no table is kept for it yet
    }
    return out;
}

// The Vulkan objects create_vulkan_device_standalone made, destroyed with the device.
struct OwnedRuntime {
    PFN_vkDestroyDevice destroy_device = nullptr;
    PFN_vkDestroyInstance destroy_instance = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    VkInstance instance = VK_NULL_HANDLE;
};

class Device final
    : public Object<Device, ID3D12Device11Stubs, ID3D12Object, ID3D12Device, ID3D12Device1, ID3D12Device2,
                    ID3D12Device3, ID3D12Device4, ID3D12Device5, ID3D12Device6, ID3D12Device7, ID3D12Device8,
                    ID3D12Device9, ID3D12Device10, ID3D12Device11> {
public:
    // {c58e1d27-3a96-4b4f-9e72-6d0a8b5f3c61}
    static constexpr GUID kTag{0xc58e1d27, 0x3a96, 0x4b4f, {0x9e, 0x72, 0x6d, 0x0a, 0x8b, 0x5f, 0x3c, 0x61}};

    explicit Device(const VulkanDeviceInfo& info) {
        state_.info = info;
        const auto get_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            info.get_instance_proc(info.instance, "vkGetDeviceProcAddr"));
        state_.format_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceFormatProperties"));
        state_.image_format_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceImageFormatProperties>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceImageFormatProperties"));
        const auto memory_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceMemoryProperties"));
        if (!get_device || !state_.format_properties || !state_.image_format_properties || !memory_properties) return;
        memory_properties(info.physical, &state_.memory);
        for (const char* name : {"vkGetDeviceImageMemoryRequirements", "vkGetDeviceImageMemoryRequirementsKHR"})
            if (!state_.image_requirements)
                state_.image_requirements =
                    reinterpret_cast<PFN_vkGetDeviceImageMemoryRequirements>(get_device(info.device, name));

        if (state_.info.max_push_constants_size == 0 && info.physical && info.get_instance_proc) {
            const auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
                info.get_instance_proc(info.instance, "vkGetPhysicalDeviceProperties"));
            if (get_props) {
                VkPhysicalDeviceProperties props{};
                get_props(info.physical, &props);
                state_.info.max_push_constants_size = props.limits.maxPushConstantsSize;
                state_.info.push_constants_need_fallback = (state_.info.max_push_constants_size < 144);
            }
        }

        state_.context = std::make_unique<leonida::VulkanContext>(get_device, info.device,
                                                                  info.separate_depth_stencil_layouts,
                                                                  info.conditional_rendering);
        if (info.host_image_copy) load_host_image_copy(get_device);
        if (info.bindless && state_.context->loaded() && !create_bindless(get_device))
            std::fprintf(stderr, "[rt] d3d12: the bindless set 0 could not be created\n");
    }

    // Set 0 of every translated shader that declares bindings: images, samplers and the descriptor hash table
    // (docs section 3.4). Each ExecuteCommandLists brings it up to date before its commands are written.
    bool create_bindless(PFN_vkGetDeviceProcAddr get_device) {
        leonida::BindlessFeatures features{};
        features.update_after_bind = true;  // VulkanDeviceInfo::bindless promises it
        features.partially_bound = state_.info.bindless_partially_bound;
        state_.bindless = std::make_unique<leonida::BindlessSet>(get_device, state_.info.device, state_.memory, features);
        if (!state_.bindless->ok() || !state_.context->vk.vkCreateSampler) {
            state_.bindless.reset();
            return false;
        }
        state_.context->set_bindless_prepare([this](VkCommandBuffer command_buffer) {
            std::lock_guard lock(state_.descriptor_mutex);
            if (!state_.bindless->fallbacks_ready()) state_.bindless->record_fallback_init(command_buffer);
            return state_.bindless->flush();
        });
        return true;
    }

    ~Device() override {
        if (state_.context) {
            std::lock_guard lock(state_.context->submit_mutex());
            state_.context->vk.vkDeviceWaitIdle(state_.info.device);
        }
        state_.staging_queue.reset();  // it uses the context
        if (state_.context && state_.bindless) {
            // Every texture is gone by now (each held this device), and its views with it; samplers remain.
            std::lock_guard lock(state_.descriptor_mutex);
            for (const auto& [key, sampler] : state_.sampler_entries)
                state_.context->vk.vkDestroySampler(state_.info.device, sampler, nullptr);
            state_.sampler_entries.clear();
            state_.bindless.reset();
        }
        state_.context.reset();
        if (owned_.device) owned_.destroy_device(owned_.device, nullptr);
        if (owned_.instance) owned_.destroy_instance(owned_.instance, nullptr);
    }

    bool ok() const { return state_.context && state_.context->loaded(); }
    const VulkanDeviceInfo& info() const { return state_.info; }
    const leonida::BindlessSet* bindless() const { return state_.bindless.get(); }
    void own(const OwnedRuntime& owned) { owned_ = owned; }
    void set_outer(IUnknown* outer) { outer_ = outer; }

    UINT STDMETHODCALLTYPE GetNodeCount() override { return 1; }
    HRESULT STDMETHODCALLTYPE GetDeviceRemovedReason() override { return S_OK; }

    // Residency and power (D3D-33, docs/d3d12_vulkan.md section 4.1). Vulkan has no residency API: memory from
    // vkAllocateMemory stays usable by the device until vkFreeMemory, and paging under pressure is the kernel
    // driver's job (WDDM on Windows), invisible to the application. So every pageable object of this device is
    // always resident, MakeResident has nothing to do and Evict, which D3D12 documents as a hint the runtime may
    // defer, can be dropped. Only the arguments are checked; the objects themselves are not touched.
    HRESULT STDMETHODCALLTYPE MakeResident(UINT count, ID3D12Pageable* const* objects) override {
        return count && !objects ? E_INVALIDARG : S_OK;
    }

    HRESULT STDMETHODCALLTYPE Evict(UINT count, ID3D12Pageable* const* objects) override {
        return count && !objects ? E_INVALIDARG : S_OK;
    }

    // Device1: priorities only order what the OS evicts first under memory pressure. VK_EXT_pageable_device_local_memory
    // could pass them on, but it is optional and changes no result, so they are accepted and dropped. Any UINT is a
    // valid priority (the D3D12_RESIDENCY_PRIORITY values are named points, the low 24 bits a refinement).
    HRESULT STDMETHODCALLTYPE SetResidencyPriority(UINT count, ID3D12Pageable* const* objects,
                                                   const D3D12_RESIDENCY_PRIORITY* priorities) override {
        return count && (!objects || !priorities) ? E_INVALIDARG : S_OK;
    }

    // Device3: the asynchronous MakeResident. The objects are already resident, so the work is complete at once
    // and the fence is signalled from the CPU now. The caller's queue Wait on it must not hang, which is why the
    // signal is kept although the residency part is a no-op. DENY_OVERBUDGET cannot fail: nothing is paged in.
    HRESULT STDMETHODCALLTYPE EnqueueMakeResident(D3D12_RESIDENCY_FLAGS flags, UINT count,
                                                  ID3D12Pageable* const* objects, ID3D12Fence* fence,
                                                  UINT64 value) override {
        if (!fence || (count && !objects) || (flags & ~D3D12_RESIDENCY_FLAG_DENY_OVERBUDGET)) return E_INVALIDARG;
        return fence->Signal(value);
    }

    // Stable power state pins GPU clocks for profiling, a developer-mode feature on Windows. Vulkan has no clock
    // control (vendor tools do it outside the API), and the game's frame does not depend on the clocks, so the
    // request is accepted. Success is what D3D12 returns on a machine where the call is allowed.
    HRESULT STDMETHODCALLTYPE SetStablePowerState(BOOL) override { return S_OK; }

    // Device6: lets the driver run background shader recompilation and profiling-guided optimisation. Vulkan
    // drivers do that, if at all, without an API to steer it, so there is nothing to start or stop. The completion
    // event is set at once, since no measurement is running, and no further measurements are wanted.
    HRESULT STDMETHODCALLTYPE SetBackgroundProcessingMode(D3D12_BACKGROUND_PROCESSING_MODE mode,
                                                          D3D12_MEASUREMENTS_ACTION action, HANDLE event,
                                                          BOOL* further_measurements) override {
        if (unsigned(mode) > unsigned(D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_PROFILING_BY_SYSTEM) ||
            unsigned(action) > unsigned(D3D12_MEASUREMENTS_ACTION_DISCARD_PREVIOUS))
            return E_INVALIDARG;
        if (further_measurements) *further_measurements = FALSE;
        if (event) SetEvent(event);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE CreateCommandQueue(const D3D12_COMMAND_QUEUE_DESC* desc, REFIID riid,
                                                 void** out) override {
        if (!desc) return E_INVALIDARG;
        // Copy and compute queues share the one graphics VkQueue for now (section 8.2, question 12).
        if (desc->Type != D3D12_COMMAND_LIST_TYPE_DIRECT) log_once("command queue type on the direct VkQueue:", desc->Type);
        return create_command_queue(*state_.context, state_.info.queue, state_.info.queue_family, *desc, riid, out,
                                    identity());
    }

    HRESULT STDMETHODCALLTYPE CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void** out) override {
        return create_command_allocator(type, riid, out, identity());
    }

    HRESULT STDMETHODCALLTYPE CreateCommandList(UINT, D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* allocator,
                                                ID3D12PipelineState* state, REFIID riid, void** out) override {
        if (state) stub("ID3D12Device::CreateCommandList(initial pipeline state)");
        return create_command_list(type, allocator, riid, out, identity());
    }

    // Device4: a list created closed, without an allocator.
    HRESULT STDMETHODCALLTYPE CreateCommandList1(UINT, D3D12_COMMAND_LIST_TYPE type, D3D12_COMMAND_LIST_FLAGS,
                                                 REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        ID3D12CommandAllocator* allocator = nullptr;
        HRESULT hr = create_command_allocator(type, IID_PPV_ARGS(&allocator), identity());
        if (FAILED(hr)) return hr;
        ID3D12GraphicsCommandList* list = nullptr;
        hr = create_command_list(type, allocator, IID_PPV_ARGS(&list), identity());
        allocator->Release();
        if (FAILED(hr)) return hr;
        list->Close();
        hr = list->QueryInterface(riid, out);
        list->Release();
        return hr;
    }

    HRESULT STDMETHODCALLTYPE CreateFence(UINT64 initial, D3D12_FENCE_FLAGS flags, REFIID riid, void** out) override {
        return create_fence(*state_.context, initial, flags, riid, out, identity());
    }

    HRESULT STDMETHODCALLTYPE CreateDescriptorHeap(const D3D12_DESCRIPTOR_HEAP_DESC* desc, REFIID riid,
                                                   void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!desc || !desc->NumDescriptors) return E_INVALIDARG;
        auto* heap = new DescriptorHeap(*desc);
        heap->attach_device(identity());
        return hand_out(heap, riid, out);
    }

    UINT STDMETHODCALLTYPE GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE) override {
        return kDescriptorSize;
    }

    void STDMETHODCALLTYPE CreateRenderTargetView(ID3D12Resource* resource, const D3D12_RENDER_TARGET_VIEW_DESC* desc,
                                                  D3D12_CPU_DESCRIPTOR_HANDLE destination) override {
        if (!destination.ptr) return;
        leonida::TargetView view{};
        if (const Resource* r = Resource::from<Resource>(resource)) {
            const D3D12_RESOURCE_DESC rd = const_cast<Resource*>(r)->GetDesc();
            view = {r->id(), 0, 0, rd.DepthOrArraySize};
            if (desc && desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2D) {
                view = {r->id(), desc->Texture2D.MipSlice, 0, 1};
            } else if (desc && desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2DMS) {
                view = {r->id(), 0, 0, 1};
            } else if (desc && desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2DARRAY) {
                view = {r->id(), desc->Texture2DArray.MipSlice, desc->Texture2DArray.FirstArraySlice,
                        desc->Texture2DArray.ArraySize};
            } else if (desc) {
                log_once("render-target view dimension not supported:", desc->ViewDimension);
                view = {};
            }
            const bool typeless_rgba8 = rd.Format == DXGI_FORMAT_R8G8B8A8_TYPELESS;
            const bool typed_rgba8 = rd.Format == DXGI_FORMAT_R8G8B8A8_UNORM ||
                rd.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || rd.Format == DXGI_FORMAT_R8G8B8A8_UINT;
            if (typeless_rgba8 || typed_rgba8) {
                const DXGI_FORMAT requested = desc && desc->Format != DXGI_FORMAT_UNKNOWN ? desc->Format : rd.Format;
                if (typeless_rgba8) {
                    // This first typeless RTV slice deliberately covers only a single 2D subresource.
                    if (!desc || desc->ViewDimension != D3D12_RTV_DIMENSION_TEXTURE2D ||
                        rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || rd.MipLevels != 1 ||
                        rd.DepthOrArraySize != 1 || rd.SampleDesc.Count != 1 || desc->Texture2D.MipSlice ||
                        desc->Texture2D.PlaneSlice || !(rd.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) {
                        view = {};
                    } else {
                        switch (requested) {
                        case DXGI_FORMAT_R8G8B8A8_UNORM: view.format = leonida::ColorViewFormat::rgba8_unorm; break;
                        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: view.format = leonida::ColorViewFormat::rgba8_srgb; break;
                        case DXGI_FORMAT_R8G8B8A8_UINT: view.format = leonida::ColorViewFormat::rgba8_uint; break;
                        default: view = {}; break;
                        }
                        if (view.resource) {
                            const auto format = vulkan_format(requested, false);
                            VkFormatProperties properties{};
                            state_.format_properties(state_.info.physical, format.format, &properties);
                            if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) view = {};
                        }
                    }
                } else if (requested != rd.Format) {
                    view = {}; // No typed-resource format casting is advertised by this backend.
                }
                if (!view.resource) log_once("unsupported RGBA8 RTV format/subresource:", int(requested));
            }
        } else if (resource) {
            stub("ID3D12Device::CreateRenderTargetView(resource from another device)");
        }
        std::memcpy(reinterpret_cast<void*>(destination.ptr), &view, sizeof(view));
    }

    void STDMETHODCALLTYPE CreateDepthStencilView(ID3D12Resource* resource, const D3D12_DEPTH_STENCIL_VIEW_DESC* desc,
                                                  D3D12_CPU_DESCRIPTOR_HANDLE destination) override {
        if (!destination.ptr) return;
        leonida::TargetView view{};
        if (const Resource* r = Resource::from<Resource>(resource)) {
            const D3D12_RESOURCE_DESC rd = const_cast<Resource*>(r)->GetDesc();
            view = {r->id(), 0, 0, rd.DepthOrArraySize};
            if (desc && desc->ViewDimension == D3D12_DSV_DIMENSION_TEXTURE2D) {
                view = {r->id(), desc->Texture2D.MipSlice, 0, 1};
            } else if (desc && desc->ViewDimension == D3D12_DSV_DIMENSION_TEXTURE2DMS) {
                view = {r->id(), 0, 0, 1};
            } else if (desc && desc->ViewDimension == D3D12_DSV_DIMENSION_TEXTURE2DARRAY) {
                view = {r->id(), desc->Texture2DArray.MipSlice, desc->Texture2DArray.FirstArraySlice,
                        desc->Texture2DArray.ArraySize};
            } else if (desc) {
                log_once("depth-stencil view dimension not supported:", desc->ViewDimension);
                view = {};
            }
        } else if (resource) {
            stub("ID3D12Device::CreateDepthStencilView(resource from another device)");
        }
        std::memcpy(reinterpret_cast<void*>(destination.ptr), &view, sizeof(view));
    }

    // A descriptor is its slot's bytes (a TargetView in RTV and DSV heaps, otherwise what the shader loads), so a
    // copy is a byte copy. D3D12 walks the destination and source ranges together, one descriptor at a time; a
    // null size array means every range holds one descriptor. Whichever side runs out first ends the copy.
    void STDMETHODCALLTYPE CopyDescriptors(UINT destination_ranges, const D3D12_CPU_DESCRIPTOR_HANDLE* destinations,
                                           const UINT* destination_sizes, UINT source_ranges,
                                           const D3D12_CPU_DESCRIPTOR_HANDLE* sources, const UINT* source_sizes,
                                           D3D12_DESCRIPTOR_HEAP_TYPE) override {
        if (!destinations || !sources) return;
        UINT d = 0, s = 0, d_at = 0, s_at = 0;  // the range on each side, and the position in it
        while (d < destination_ranges && s < source_ranges) {
            if (d_at >= (destination_sizes ? destination_sizes[d] : 1) || !destinations[d].ptr) {
                ++d;
                d_at = 0;
            } else if (s_at >= (source_sizes ? source_sizes[s] : 1) || !sources[s].ptr) {
                ++s;
                s_at = 0;
            } else {
                std::memmove(reinterpret_cast<std::uint8_t*>(destinations[d].ptr) + std::size_t(d_at++) * kDescriptorSize,
                             reinterpret_cast<const std::uint8_t*>(sources[s].ptr) + std::size_t(s_at++) * kDescriptorSize,
                             kDescriptorSize);
            }
        }
    }

    void STDMETHODCALLTYPE CopyDescriptorsSimple(UINT count, D3D12_CPU_DESCRIPTOR_HANDLE destination,
                                                 D3D12_CPU_DESCRIPTOR_HANDLE source,
                                                 D3D12_DESCRIPTOR_HEAP_TYPE) override {
        if (!destination.ptr || !source.ptr) return;
        std::memmove(reinterpret_cast<void*>(destination.ptr), reinterpret_cast<const void*>(source.ptr),
                     std::size_t(count) * kDescriptorSize);
    }

    // A shader-resource view of a texture is a T# in the slot (docs section 3.4): the shader loads those eight
    // dwords through its descriptor table and finds the image by them. A slot left as zeros is a null descriptor:
    // nothing is registered for it, so the shader gets the fallback image. Buffer views are not implemented.
    void STDMETHODCALLTYPE CreateShaderResourceView(ID3D12Resource* resource,
                                                    const D3D12_SHADER_RESOURCE_VIEW_DESC* desc,
                                                    D3D12_CPU_DESCRIPTOR_HANDLE destination) override {
        if (!destination.ptr) return;
        std::memset(reinterpret_cast<void*>(destination.ptr), 0, kDescriptorSize);
        const Resource* r = Resource::from<Resource>(resource);
        if (!r) {
            if (resource) stub("ID3D12Device::CreateShaderResourceView(resource from another device)");
            return;
        }
        if (!r->image()) return stub("ID3D12Device::CreateShaderResourceView(buffer)");
        TextureView view{};
        if (!desc) {
            if (r->desc().DepthOrArraySize > 1) view.type = leonida::gfx10::ImageType::image_2d_array;
        } else {
            view.format = desc->Format;
            view.mapping = desc->Shader4ComponentMapping;
            switch (desc->ViewDimension) {
            case D3D12_SRV_DIMENSION_TEXTURE2D:
                view.first_mip = desc->Texture2D.MostDetailedMip;
                view.mip_count = desc->Texture2D.MipLevels;
                view.layer_count = 1;
                view.plane = desc->Texture2D.PlaneSlice;
                view.min_lod = desc->Texture2D.ResourceMinLODClamp;
                break;
            case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
                view.type = leonida::gfx10::ImageType::image_2d_array;
                view.first_mip = desc->Texture2DArray.MostDetailedMip;
                view.mip_count = desc->Texture2DArray.MipLevels;
                view.first_layer = desc->Texture2DArray.FirstArraySlice;
                view.layer_count = desc->Texture2DArray.ArraySize;
                view.plane = desc->Texture2DArray.PlaneSlice;
                view.min_lod = desc->Texture2DArray.ResourceMinLODClamp;
                break;
            case D3D12_SRV_DIMENSION_TEXTURECUBE:
                view.type = leonida::gfx10::ImageType::cube;
                view.first_mip = desc->TextureCube.MostDetailedMip;
                view.mip_count = desc->TextureCube.MipLevels;
                view.layer_count = 6;
                view.min_lod = desc->TextureCube.ResourceMinLODClamp;
                break;
            case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
                view.type = leonida::gfx10::ImageType::cube;
                view.first_mip = desc->TextureCubeArray.MostDetailedMip;
                view.mip_count = desc->TextureCubeArray.MipLevels;
                view.first_layer = desc->TextureCubeArray.First2DArrayFace;
                view.layer_count = desc->TextureCubeArray.NumCubes < ~0u / 6 ? desc->TextureCubeArray.NumCubes * 6 : ~0u;
                view.min_lod = desc->TextureCubeArray.ResourceMinLODClamp;
                break;
            default:
                log_once("shader-resource view dimension not supported:", desc->ViewDimension);
                return;
            }
        }
        write_image_descriptor(*r, view, destination, false);
    }

    // An unordered-access view of a texture is a T# too, of one mip; set 0 also holds it as a storage image.
    void STDMETHODCALLTYPE CreateUnorderedAccessView(ID3D12Resource* resource, ID3D12Resource* counter,
                                                     const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc,
                                                     D3D12_CPU_DESCRIPTOR_HANDLE destination) override {
        if (!destination.ptr) return;
        std::memset(reinterpret_cast<void*>(destination.ptr), 0, kDescriptorSize);
        const Resource* r = Resource::from<Resource>(resource);
        if (!r) {
            if (resource) stub("ID3D12Device::CreateUnorderedAccessView(resource from another device)");
            return;
        }
        if (!r->image() || counter) return stub("ID3D12Device::CreateUnorderedAccessView(buffer)");
        if (!(r->desc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
            log_once("unordered-access view of a texture created without ALLOW_UNORDERED_ACCESS; flags",
                     int(r->desc().Flags));
            return;
        }
        TextureView view{};
        view.mip_count = 1;
        if (!desc) {
            if (r->desc().DepthOrArraySize > 1) view.type = leonida::gfx10::ImageType::image_2d_array;
        } else {
            view.format = desc->Format;
            switch (desc->ViewDimension) {
            case D3D12_UAV_DIMENSION_TEXTURE2D:
                view.first_mip = desc->Texture2D.MipSlice;
                view.layer_count = 1;
                view.plane = desc->Texture2D.PlaneSlice;
                break;
            case D3D12_UAV_DIMENSION_TEXTURE2DARRAY:
                view.type = leonida::gfx10::ImageType::image_2d_array;
                view.first_mip = desc->Texture2DArray.MipSlice;
                view.first_layer = desc->Texture2DArray.FirstArraySlice;
                view.layer_count = desc->Texture2DArray.ArraySize;
                view.plane = desc->Texture2DArray.PlaneSlice;
                break;
            default:
                log_once("unordered-access view dimension not supported:", desc->ViewDimension);
                return;
            }
        }
        write_image_descriptor(*r, view, destination, true);
    }

    // A sampler is an S# in the first 16 bytes of the slot; the shader finds the sampler by those four dwords.
    void STDMETHODCALLTYPE CreateSampler(const D3D12_SAMPLER_DESC* desc,
                                         D3D12_CPU_DESCRIPTOR_HANDLE destination) override {
        if (!destination.ptr) return;
        std::memset(reinterpret_cast<void*>(destination.ptr), 0, kDescriptorSize);
        if (!desc) return;
        const leonida::gfx10::SamplerWords words = leonida::gfx10::encode(sampler_descriptor(*desc));
        std::memcpy(reinterpret_cast<void*>(destination.ptr), words.data(), sizeof(words));
        state_.bind_sampler(words);
    }

    HRESULT STDMETHODCALLTYPE CreateCommittedResource(const D3D12_HEAP_PROPERTIES* heap, D3D12_HEAP_FLAGS heap_flags,
                                                      const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES,
                                                      const D3D12_CLEAR_VALUE*, REFIID riid, void** out) override {
        if (out) *out = nullptr;
        if (!heap || !desc) return E_INVALIDARG;
        auto* resource = new Resource(state_, *desc, *heap, heap_flags);
        resource->attach_device(identity());
        HRESULT hr = E_NOTIMPL;
        if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            hr = resource->create_buffer();
        } else if (desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                   desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
            log_once("committed resource dimension not supported:", desc->Dimension);
        } else if (desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D &&
                   (desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))) {
            log_once("volume attachment views not supported, flags:", int(desc->Flags));
        } else if (heap->Type == D3D12_HEAP_TYPE_UPLOAD || heap->Type == D3D12_HEAP_TYPE_READBACK) {
            hr = E_INVALIDARG;  // D3D12 places textures on default heaps unless they are row-major
        } else if (const UINT samples = desc->SampleDesc.Count;
                   samples > 1 && ((samples & (samples - 1)) || desc->MipLevels != 1 ||
                                   desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
                                   !(desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                                                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)))) {
            hr = E_INVALIDARG;  // D3D12: a multisampled texture is a single-mip 2D render target or depth-stencil
        } else {
            const Format format = state_.texture_format(*desc);
            if (format.format == VK_FORMAT_UNDEFINED) {
                log_once("texture format not supported yet:", desc->Format);
                hr = E_INVALIDARG;
            } else {
                hr = resource->create_texture(format);
            }
        }
        if (FAILED(hr)) {
            resource->Release();
            return hr;
        }
        if (!out) {  // a capability check: D3D12 allows a null output
            resource->Release();
            return S_FALSE;
        }
        return hand_out(resource, riid, out);
    }

    // Reserved (tiled) resources, for the first boot (docs/d3d12_vulkan.md section 4.2): the whole resource is
    // reserved linearly and backed at creation, as a committed resource on a default heap would be. Tile mappings
    // are not emulated (queue UpdateTileMappings and CopyTileMappings log and do nothing), so every tile reads and
    // writes its own memory; unmapped tiles start zeroed, as D3D12 tier 2 reads them. The rules for the description
    // are D3D12's: buffers are ROW_MAJOR, textures 64KB_UNDEFINED_SWIZZLE in a format with a tile shape
    // (64KB_STANDARD_SWIZZLE needs StandardSwizzle64KBSupported, which CheckFeatureSupport reports as FALSE). The
    // Xbox tiled layout word 0x101 (slot 76, section 4.2) is accepted as 64KB_UNDEFINED_SWIZZLE; that the console
    // uses it for reserved resources too is an assumption.
    HRESULT STDMETHODCALLTYPE CreateReservedResource(const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES,
                                                     const D3D12_CLEAR_VALUE*, REFIID riid, void** out) override {
        return create_reserved(desc, riid, out);
    }

    // Device4. Protected sessions are not supported.
    HRESULT STDMETHODCALLTYPE CreateReservedResource1(const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES,
                                                      const D3D12_CLEAR_VALUE*, ID3D12ProtectedResourceSession* session,
                                                      REFIID riid, void** out) override {
        if (session) {
            if (out) *out = nullptr;
            log_once("CreateReservedResource1 with a protected session not supported:", 1);
            return E_NOTIMPL;
        }
        return create_reserved(desc, riid, out);
    }

    // Device10: an initial layout instead of a state (resources start in their first use's layout here anyway).
    // Castable formats are not used: typeless resources are already mutable-format images.
    HRESULT STDMETHODCALLTYPE CreateReservedResource2(const D3D12_RESOURCE_DESC* desc, D3D12_BARRIER_LAYOUT,
                                                      const D3D12_CLEAR_VALUE*, ID3D12ProtectedResourceSession* session,
                                                      UINT32 castable_formats, const DXGI_FORMAT*, REFIID riid,
                                                      void** out) override {
        if (castable_formats) log_once("CreateReservedResource2: castable formats ignored, count", int(castable_formats));
        return CreateReservedResource1(desc, D3D12_RESOURCE_STATE_COMMON, nullptr, session, riid, out);
    }

    // Tiling of a reserved resource (src/resource_tiling.h). Anything else, as on the PC runtime, has no tiles:
    // every output is zero and no subresource tiling is written.
    void STDMETHODCALLTYPE GetResourceTiling(ID3D12Resource* resource, UINT* total, D3D12_PACKED_MIP_INFO* packed,
                                             D3D12_TILE_SHAPE* shape, UINT* count, UINT first,
                                             D3D12_SUBRESOURCE_TILING* tilings) override {
        leonida::ResourceTiling tiling;
        Resource* r = Resource::from<Resource>(resource);
        const bool ok = r && r->reserved() && resource_tiling(r->GetDesc(), tiling);
        if (total) *total = ok ? tiling.total_tiles : 0;
        if (packed)
            *packed = ok ? D3D12_PACKED_MIP_INFO{UINT8(tiling.standard_mips), UINT8(tiling.packed_mips),
                                                 tiling.packed_tiles, tiling.packed_start}
                         : D3D12_PACKED_MIP_INFO{};
        if (shape)
            *shape = ok ? D3D12_TILE_SHAPE{tiling.shape.width, tiling.shape.height, tiling.shape.depth}
                        : D3D12_TILE_SHAPE{};
        if (!count) return;
        const UINT available = ok && first < tiling.subresources.size() ? UINT(tiling.subresources.size() - first) : 0;
        const UINT n = tilings ? std::min(*count, available) : 0;
        for (UINT i = 0; i < n; ++i) {
            const leonida::SubresourceTiling& s = tiling.subresources[first + i];
            tilings[i] = {s.width, UINT16(s.height), UINT16(s.depth), s.start};
        }
        *count = n;
    }

    // Footprints for copies through a buffer: rows padded to D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (512 for planar
    // depth-stencil), subresources placed at D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, block-compressed sizes rounded up to whole blocks, depth-stencil
    // formats as two planes. d3d12_vulkan_device_test compares every output with the PC runtime's.
    void STDMETHODCALLTYPE GetCopyableFootprints(const D3D12_RESOURCE_DESC* desc, UINT first, UINT count, UINT64 base,
                                                 D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layouts, UINT* rows,
                                                 UINT64* row_sizes, UINT64* total) override {
        const auto fail = [&] {
            for (UINT i = 0; i < count; ++i) {
                if (layouts) std::memset(&layouts[i], 0xff, sizeof(layouts[i]));
                if (rows) rows[i] = ~0u;
                if (row_sizes) row_sizes[i] = ~0ull;
            }
            if (total) *total = ~0ull;
        };
        if (!desc) return fail();
        const auto align = [](UINT64 value, UINT64 to) { return (value + to - 1) / to * to; };
        if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            if (first != 0 || count != 1) return fail();
            if (layouts) layouts[0] = {base, {DXGI_FORMAT_UNKNOWN, UINT(desc->Width), 1, 1,
                                              UINT(align(desc->Width, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT))}};
            if (rows) rows[0] = 1;
            if (row_sizes) row_sizes[0] = desc->Width;
            if (total) *total = desc->Width;
            return;
        }
        const bool volume = desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        if (!volume && desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
            log_once("GetCopyableFootprints dimension not supported:", desc->Dimension);
            return fail();
        }
        const Format format = vulkan_format(desc->Format, true);  // typeless depth families are planar either way
        UINT mips = desc->MipLevels;
        const UINT depth = volume ? desc->DepthOrArraySize : 1;
        if (!mips)
            for (UINT64 size = std::max<UINT64>({desc->Width, desc->Height, depth}); size; size >>= 1) ++mips;
        const UINT planes = (format.aspects & VK_IMAGE_ASPECT_STENCIL_BIT) ? 2 : 1;
        const UINT per_plane = mips * (volume ? 1 : desc->DepthOrArraySize);
        if (!format.format || UINT64(first) + count > UINT64(per_plane) * planes) return fail();

        UINT64 offset = base, end = base;
        for (UINT i = 0; i < count; ++i) {
            const UINT index = first + i, plane = index / per_plane, mip = index % mips;
            const VkImageAspectFlags aspect =
                planes == 2 ? (plane ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT) : format.aspects;
            const leonida::BlockFormat block = leonida::block_format(format.format, aspect);
            if (!block.bytes) return fail();
            const UINT width = std::max<UINT>(1, UINT(desc->Width >> mip)), height = std::max<UINT>(1, desc->Height >> mip);
            const UINT blocks_wide = (width + block.block - 1) / block.block;
            const UINT blocks_high = (height + block.block - 1) / block.block;
            const UINT64 row_size = UINT64(blocks_wide) * block.bytes;
            // Measured against the PC runtime (d3d12_vulkan_device_test): planar depth-stencil rows are padded to
            // 512 bytes, every other format's to D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (256).
            const UINT pitch = UINT(align(row_size, planes == 2 ? 2 * D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
                                                                : D3D12_TEXTURE_DATA_PITCH_ALIGNMENT));
            offset = align(offset, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
            DXGI_FORMAT footprint_format = desc->Format;
            if (planes == 2) footprint_format = plane ? DXGI_FORMAT_R8_TYPELESS : DXGI_FORMAT_R32_TYPELESS;
            const UINT slices = std::max(1u, depth >> mip);
            if (layouts)
                layouts[i] = {offset, {footprint_format, blocks_wide * block.block, blocks_high * block.block, slices, pitch}};
            if (rows) rows[i] = blocks_high;
            if (row_sizes) row_sizes[i] = row_size;
            const UINT64 total_rows = UINT64(blocks_high) * slices;
            end = offset + (total_rows - 1) * pitch + row_size;
            offset += total_rows * pitch;
        }
        if (total) *total = end - base;
    }

    // How much heap memory resources need and how to align them, for applications that suballocate heaps (section
    // 4.1). Buffers follow D3D12's fixed rule. A texture's size is what the Vulkan image it becomes needs, rounded up
    // to D3D12's alignment for it; the alignment itself is D3D12's, since placed textures bind their own memory
    // (section 4.2). Several resources are laid out one after another, each at its own alignment. An invalid
    // description gives SizeInBytes = UINT64_MAX, as on D3D12. d3d12_vulkan_device_test compares with the PC runtime.
    D3D12_RESOURCE_ALLOCATION_INFO STDMETHODCALLTYPE GetResourceAllocationInfo(
        UINT, UINT count, const D3D12_RESOURCE_DESC* descs) override {
        return place(count, descs, [&](UINT i) { return descs[i]; }, nullptr);
    }
    D3D12_RESOURCE_ALLOCATION_INFO STDMETHODCALLTYPE GetResourceAllocationInfo1(
        UINT, UINT count, const D3D12_RESOURCE_DESC* descs, D3D12_RESOURCE_ALLOCATION_INFO1* each) override {
        return place(count, descs, [&](UINT i) { return descs[i]; }, each);
    }
    D3D12_RESOURCE_ALLOCATION_INFO STDMETHODCALLTYPE GetResourceAllocationInfo2(
        UINT, UINT count, const D3D12_RESOURCE_DESC1* descs, D3D12_RESOURCE_ALLOCATION_INFO1* each) override {
        return place(count, descs, [&](UINT i) {
            const D3D12_RESOURCE_DESC1& d = descs[i];
            if (d.SamplerFeedbackMipRegion.Width || d.SamplerFeedbackMipRegion.Height ||
                d.SamplerFeedbackMipRegion.Depth) {
                log_once("GetResourceAllocationInfo2 sampler feedback not supported, format", d.Format);
                return D3D12_RESOURCE_DESC{};  // an unknown dimension: invalid
            }
            return D3D12_RESOURCE_DESC{d.Dimension, d.Alignment, d.Width, d.Height, d.DepthOrArraySize,
                                       d.MipLevels, d.Format, d.SampleDesc, d.Layout, d.Flags};
        }, each);
    }

    HRESULT STDMETHODCALLTYPE CheckFeatureSupport(D3D12_FEATURE feature, void* data, UINT size) override {
        log_once("CheckFeatureSupport feature", feature);
        if (!data) return E_INVALIDARG;
        switch (feature) {
        case D3D12_FEATURE_FEATURE_LEVELS: {
            if (size != sizeof(D3D12_FEATURE_DATA_FEATURE_LEVELS)) return E_INVALIDARG;
            auto* levels = static_cast<D3D12_FEATURE_DATA_FEATURE_LEVELS*>(data);
            D3D_FEATURE_LEVEL best = D3D_FEATURE_LEVEL(0);
            for (UINT i = 0; i < levels->NumFeatureLevels; ++i)
                if (levels->pFeatureLevelsRequested[i] <= D3D_FEATURE_LEVEL_12_1)
                    best = std::max(best, levels->pFeatureLevelsRequested[i]);
            if (!best) return DXGI_ERROR_UNSUPPORTED;
            levels->MaxSupportedFeatureLevel = best;
            return S_OK;
        }
        case D3D12_FEATURE_ARCHITECTURE:
        case D3D12_FEATURE_ARCHITECTURE1: {
            const bool v1 = feature == D3D12_FEATURE_ARCHITECTURE1;
            if (size != (v1 ? sizeof(D3D12_FEATURE_DATA_ARCHITECTURE1) : sizeof(D3D12_FEATURE_DATA_ARCHITECTURE)))
                return E_INVALIDARG;
            auto* arch = static_cast<D3D12_FEATURE_DATA_ARCHITECTURE1*>(data);
            const UINT node = arch->NodeIndex;
            if (node != 0) return E_INVALIDARG;
            // A discrete desktop GPU: not tile-based, no UMA. The console is UMA (question 11); the layer keeps
            // upload heaps host-visible, so the game's upload path still works.
            arch->TileBasedRenderer = FALSE;
            arch->UMA = FALSE;
            arch->CacheCoherentUMA = FALSE;
            if (v1) arch->IsolatedMMU = TRUE;
            return S_OK;
        }
        case D3D12_FEATURE_D3D12_OPTIONS: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS)) return E_INVALIDARG;
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS*>(data);
            *opts = {};
            opts->DoublePrecisionFloatShaderOps = TRUE;
            opts->OutputMergerLogicOp = TRUE;
            opts->MinPrecisionSupport = D3D12_SHADER_MIN_PRECISION_SUPPORT_NONE;
            opts->TiledResourcesTier = D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED;
            opts->ResourceBindingTier = D3D12_RESOURCE_BINDING_TIER_3;  // Bindless descriptor indexing supported
            opts->PSSpecifiedStencilRefSupported = TRUE;
            opts->TypedUAVLoadAdditionalFormats = TRUE;
            opts->ROVsSupported = TRUE;
            opts->ConservativeRasterizationTier = D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED;
            opts->MaxGPUVirtualAddressBitsPerResource = 40;
            opts->StandardSwizzle64KBSupported = FALSE;
            opts->CrossNodeSharingTier = D3D12_CROSS_NODE_SHARING_TIER_NOT_SUPPORTED;
            opts->CrossAdapterRowMajorTextureSupported = FALSE;
            opts->VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation = TRUE;
            opts->ResourceHeapTier = D3D12_RESOURCE_HEAP_TIER_1;
            return S_OK;
        }
        case D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT: {
            if (size != sizeof(D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT)) return E_INVALIDARG;
            auto* va = static_cast<D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT*>(data);
            // The console and PC BDA support 48-bit GPU virtual addresses (section 4.1).
            va->MaxGPUVirtualAddressBitsPerResource = 40;
            va->MaxGPUVirtualAddressBitsPerProcess = 48;
            return S_OK;
        }
        case D3D12_FEATURE_FORMAT_SUPPORT: {
            if (size != sizeof(D3D12_FEATURE_DATA_FORMAT_SUPPORT)) return E_INVALIDARG;
            auto* fs = static_cast<D3D12_FEATURE_DATA_FORMAT_SUPPORT*>(data);
            fs->Support1 = D3D12_FORMAT_SUPPORT1_NONE;
            fs->Support2 = D3D12_FORMAT_SUPPORT2_NONE;
            const Format fmt = vulkan_format(fs->Format, false);
            const Format depth_fmt = vulkan_format(fs->Format, true);
            const VkFormat vk_fmt = fmt.format != VK_FORMAT_UNDEFINED ? fmt.format : depth_fmt.format;
            if (vk_fmt == VK_FORMAT_UNDEFINED || !state_.format_properties) {
                return DXGI_ERROR_UNSUPPORTED;
            }
            VkFormatProperties props{};
            state_.format_properties(state_.info.physical, vk_fmt, &props);
            const VkFormatFeatureFlags optimal = props.optimalTilingFeatures;
            const VkFormatFeatureFlags buffer = props.bufferFeatures;

            D3D12_FORMAT_SUPPORT1 s1 = D3D12_FORMAT_SUPPORT1_NONE;
            D3D12_FORMAT_SUPPORT2 s2 = D3D12_FORMAT_SUPPORT2_NONE;

            if (buffer != 0) s1 |= D3D12_FORMAT_SUPPORT1_BUFFER;
            if (buffer & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) s1 |= D3D12_FORMAT_SUPPORT1_IA_VERTEX_BUFFER;
            if (fs->Format == DXGI_FORMAT_R16_UINT || fs->Format == DXGI_FORMAT_R32_UINT)
                s1 |= D3D12_FORMAT_SUPPORT1_IA_INDEX_BUFFER;

            if (optimal & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) {
                s1 |= D3D12_FORMAT_SUPPORT1_TEXTURE1D | D3D12_FORMAT_SUPPORT1_TEXTURE2D |
                      D3D12_FORMAT_SUPPORT1_TEXTURE3D | D3D12_FORMAT_SUPPORT1_TEXTURECUBE |
                      D3D12_FORMAT_SUPPORT1_SHADER_LOAD;
            }
            if (buffer & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT) {
                s1 |= D3D12_FORMAT_SUPPORT1_SHADER_LOAD;
            }
            if (optimal & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) {
                s1 |= D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE;
            }
            if (optimal & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) {
                s1 |= D3D12_FORMAT_SUPPORT1_RENDER_TARGET;
                // A D3D12 multisample resolve needs the same image usages as texture_create_info(): colour
                // attachment plus transfer source/destination (and the ordinary sampled usage).
                VkImageFormatProperties image_properties{};
                const VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                const VkImageCreateFlags flags = fmt.typeless ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
                if (state_.image_format_properties &&
                    state_.image_format_properties(state_.info.physical, vk_fmt, VK_IMAGE_TYPE_2D,
                                                   VK_IMAGE_TILING_OPTIMAL, usage, flags, &image_properties) == VK_SUCCESS &&
                    (image_properties.sampleCounts & ~VK_SAMPLE_COUNT_1_BIT)) {
                    s1 |= D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET |
                          D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE;
                }
            }
            if (optimal & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) {
                s1 |= D3D12_FORMAT_SUPPORT1_BLENDABLE;
            }
            if (optimal & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                s1 |= D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL;
            }
            if ((optimal & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ||
                (buffer & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT)) {
                s1 |= D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW;
                s2 |= D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE;
                if ((optimal & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT) ||
                    (buffer & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT)) {
                    s2 |= D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD;
                }
            }
            fs->Support1 = s1;
            fs->Support2 = s2;
            return S_OK;
        }
        case D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS: {
            if (size != sizeof(D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS)) return E_INVALIDARG;
            auto* levels = static_cast<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS*>(data);
            levels->NumQualityLevels = 0;
            if (levels->Flags != D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE ||
                levels->SampleCount == 0 || levels->SampleCount > 64 ||
                (levels->SampleCount & (levels->SampleCount - 1)) != 0)
                return E_INVALIDARG;
            const Format format = vulkan_format(levels->Format, false);
            if (format.format == VK_FORMAT_UNDEFINED || !state_.image_format_properties)
                return DXGI_ERROR_UNSUPPORTED;

            // Vulkan exposes one standard sample pattern per supported count, so each supported count maps to
            // one D3D12 quality level (quality index 0). Query the exact usages used for a D3D12 RT texture.
            const VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            const VkImageCreateFlags flags = format.typeless ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
            VkImageFormatProperties image_properties{};
            const VkResult result = state_.image_format_properties(state_.info.physical, format.format,
                                                                   VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                                   usage, flags, &image_properties);
            const VkSampleCountFlagBits samples = VkSampleCountFlagBits(levels->SampleCount);
            if (result == VK_SUCCESS && (image_properties.sampleCounts & samples))
                levels->NumQualityLevels = 1;
            return S_OK;
        }
        default:
            // Not answered yet. The log above lists every query; answers are added as the trace asks for them.
            return E_INVALIDARG;
        }
    }

private:
    IUnknown* identity() { return outer_ ? outer_ : static_cast<ID3D12Device*>(this); }

    static constexpr D3D12_RESOURCE_ALLOCATION_INFO kInvalidAllocation{UINT64_MAX,
                                                                       D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT};
    static UINT64 align_up(UINT64 value, UINT64 to) { return (value + to - 1) / to * to; }

    // Lays `count` resources out one after another; `each` (optional) gets every resource's offset, size and alignment.
    template<class Desc, class Get>
    D3D12_RESOURCE_ALLOCATION_INFO place(UINT count, const Desc* descs, Get get,
                                         D3D12_RESOURCE_ALLOCATION_INFO1* each) {
        D3D12_RESOURCE_ALLOCATION_INFO total{0, 0};
        bool failed = count && !descs;
        for (UINT i = 0; i < count; ++i) {
            const D3D12_RESOURCE_ALLOCATION_INFO one = failed ? kInvalidAllocation : allocation(get(i));
            failed = failed || one.SizeInBytes == UINT64_MAX;
            if (failed) {
                if (each) each[i] = {UINT64_MAX, one.Alignment, UINT64_MAX};
                continue;
            }
            const UINT64 offset = align_up(total.SizeInBytes, one.Alignment);
            if (each) each[i] = {offset, one.Alignment, one.SizeInBytes};
            total = {offset + one.SizeInBytes, std::max(total.Alignment, one.Alignment)};
        }
        return failed ? kInvalidAllocation : total;
    }

    // One resource's size and alignment, or kInvalidAllocation for a description D3D12 (or this device) rejects.
    D3D12_RESOURCE_ALLOCATION_INFO allocation(const D3D12_RESOURCE_DESC& desc) const {
        constexpr UINT64 align_4k = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT,
                         align_64k = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
                         align_msaa_64k = D3D12_SMALL_MSAA_RESOURCE_PLACEMENT_ALIGNMENT,
                         align_msaa_4m = D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT;
        constexpr D3D12_RESOURCE_FLAGS attachment =
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            if (!desc.Width || desc.Width > UINT64_MAX - align_64k || desc.Height != 1 || desc.DepthOrArraySize != 1 ||
                desc.MipLevels != 1 || desc.Format != DXGI_FORMAT_UNKNOWN || desc.SampleDesc.Count != 1 ||
                desc.SampleDesc.Quality != 0 || desc.Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR ||
                (desc.Alignment && desc.Alignment != align_64k) || (desc.Flags & attachment))
                return kInvalidAllocation;
            return {align_up(desc.Width, align_64k), align_64k};
        }
        const bool volume = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        if (!volume && desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
            if (desc.Dimension != D3D12_RESOURCE_DIMENSION_UNKNOWN)
                log_once("GetResourceAllocationInfo dimension not supported:", desc.Dimension);
            return kInvalidAllocation;
        }
        const UINT samples = desc.SampleDesc.Count, depth = volume ? desc.DepthOrArraySize : 1;
        const UINT64 max_extent =
            volume ? D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION : D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
        const UINT max_layers =
            volume ? D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION : D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION;
        if (!desc.Width || !desc.Height || !desc.DepthOrArraySize || desc.Width > max_extent ||
            desc.Height > max_extent || desc.DepthOrArraySize > max_layers || !samples || samples > 16 ||
            (samples & (samples - 1)) || desc.Layout == D3D12_TEXTURE_LAYOUT_ROW_MAJOR)
            return kInvalidAllocation;
        const UINT16 full = full_mips(desc.Width, desc.Height, depth);
        D3D12_RESOURCE_DESC resolved = desc;
        if (!resolved.MipLevels) resolved.MipLevels = full;
        if (resolved.MipLevels > full) return kInvalidAllocation;
        if (samples > 1 &&
            (volume || resolved.MipLevels != 1 || (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)))
            return kInvalidAllocation;
        if (desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN)
            log_once("GetResourceAllocationInfo layout as UNKNOWN:", desc.Layout);

        const Format format = state_.texture_format(desc);
        if (!format.format) {
            log_once("GetResourceAllocationInfo format not supported yet:", desc.Format);
            return kInvalidAllocation;
        }
        VkMemoryRequirements need{};
        if (!state_.memory_requirements(texture_create_info(resolved, format), need)) return kInvalidAllocation;

        // D3D12's alignment: 64 KiB, or 4 MiB with MSAA, unless the description asks for a small one (4 KiB, 64 KiB
        // with MSAA). 4 KiB needs the resource to fit in 64 KiB and not be a render target or depth buffer; if it
        // doesn't qualify, the PC runtime calls the description invalid rather than answering with 64 KiB. The MSAA
        // 64 KiB alignment has no size limit there. Both measured in d3d12_vulkan_device_test.
        UINT64 alignment = 0;
        if (samples > 1) {
            if (desc.Alignment && desc.Alignment != align_msaa_64k && desc.Alignment != align_msaa_4m)
                return kInvalidAllocation;
            alignment = desc.Alignment ? desc.Alignment : align_msaa_4m;
        } else {
            if (desc.Alignment && desc.Alignment != align_4k && desc.Alignment != align_64k) return kInvalidAllocation;
            if (desc.Alignment == align_4k && ((desc.Flags & attachment) || need.size > align_64k))
                return kInvalidAllocation;
            alignment = desc.Alignment ? desc.Alignment : align_64k;
        }
        return {align_up(need.size, alignment), alignment};
    }

    // Writes the view's T# into the slot and makes sure set 0 and the hash table know it. A view that cannot be
    // shown leaves the slot a null descriptor.
    void write_image_descriptor(const Resource& resource, const TextureView& view,
                                D3D12_CPU_DESCRIPTOR_HANDLE destination, bool storage) {
        leonida::gfx10::Image image{};
        if (!image_descriptor(resource, view, image)) return;
        const leonida::gfx10::ImageWords words = leonida::gfx10::encode(image);
        std::memcpy(reinterpret_cast<void*>(destination.ptr), words.data(), sizeof(words));
        state_.bind_image(words, image, resource.image(), resource.format(), resource.aspects(), storage);
    }

    // D3D12's layout rules for a reserved resource, then a committed resource on a default heap under it.
    HRESULT create_reserved(const D3D12_RESOURCE_DESC* desc, REFIID riid, void** out) {
        if (out) *out = nullptr;
        if (!desc) return E_INVALIDARG;
        constexpr auto xbox_tiled = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);
        D3D12_RESOURCE_DESC backing = *desc;
        if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            if (desc->Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR) return E_INVALIDARG;
        } else {
            if (desc->Layout == xbox_tiled) {
                log_once("reserved resource with the Xbox tiled layout, as 64KB_UNDEFINED_SWIZZLE; dimension",
                         desc->Dimension);
            } else if (desc->Layout != D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE) {
                return E_INVALIDARG;  // 64KB_STANDARD_SWIZZLE too: CheckFeatureSupport reports it unsupported
            }
            backing.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            leonida::ResourceTiling tiling;
            backing.MipLevels = std::max<UINT16>(desc->MipLevels, 1);  // any count: only the format is checked here
            if (!resource_tiling(backing, tiling)) {
                log_once("reserved texture format has no tile shape:", desc->Format);
                return E_INVALIDARG;
            }
            backing.MipLevels = desc->MipLevels;
        }
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        ID3D12Resource* resource = nullptr;
        HRESULT hr = CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &backing, D3D12_RESOURCE_STATE_COMMON,
                                             nullptr, IID_PPV_ARGS(&resource));
        if (FAILED(hr)) return hr;
        Resource::from<Resource>(resource)->make_reserved(desc->Layout);
        if (!out) {  // a capability check: D3D12 allows a null output
            resource->Release();
            return S_FALSE;
        }
        hr = resource->QueryInterface(riid, out);
        resource->Release();
        return hr;
    }

    // The tiling of a reserved resource's description; false for formats without a tile shape (three- and
    // twelve-byte texels, two-plane depth-stencil).
    static bool resource_tiling(const D3D12_RESOURCE_DESC& desc, leonida::ResourceTiling& out) {
        leonida::TilingDesc tiling{};
        tiling.width = desc.Width;
        tiling.height = desc.Height;
        tiling.depth_or_layers = desc.DepthOrArraySize;
        tiling.mips = desc.MipLevels;
        if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            tiling.kind = leonida::TiledKind::buffer;
            return leonida::resource_tiling(tiling, out);
        }
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D)
            return false;
        tiling.kind = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? leonida::TiledKind::texture3d
                                                                           : leonida::TiledKind::texture2d;
        const Format format = vulkan_format(desc.Format, false);
        if (!format.format || (format.aspects & VK_IMAGE_ASPECT_STENCIL_BIT)) return false;
        const leonida::BlockFormat block = leonida::block_format(format.format, format.aspects);
        tiling.element_bytes = block.bytes;
        tiling.block = block.block;
        return leonida::resource_tiling(tiling, out);
    }

    // Host image copies are used only on images in GENERAL (D3D12's COMMON), so GENERAL must be a layout the
    // device copies from and to on the host. Vulkan requires it to be; check rather than assume.
    void load_host_image_copy(PFN_vkGetDeviceProcAddr get_device) {
        const VkDevice dev = state_.info.device;
        const auto properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            state_.info.get_instance_proc(state_.info.instance, "vkGetPhysicalDeviceProperties2"));
        state_.format_properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties2>(
            state_.info.get_instance_proc(state_.info.instance, "vkGetPhysicalDeviceFormatProperties2"));
        const auto to_image = reinterpret_cast<PFN_vkCopyMemoryToImageEXT>(get_device(dev, "vkCopyMemoryToImageEXT"));
        const auto to_memory = reinterpret_cast<PFN_vkCopyImageToMemoryEXT>(get_device(dev, "vkCopyImageToMemoryEXT"));
        const auto transition =
            reinterpret_cast<PFN_vkTransitionImageLayoutEXT>(get_device(dev, "vkTransitionImageLayoutEXT"));
        if (!properties2 || !state_.format_properties2 || !to_image || !to_memory || !transition) {
            std::fprintf(stderr, "[rt] d3d12: VK_EXT_host_image_copy entry points missing; using staging copies\n");
            return;
        }
        VkPhysicalDeviceHostImageCopyPropertiesEXT host{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &host};
        properties2(state_.info.physical, &properties);
        std::vector<VkImageLayout> src(host.copySrcLayoutCount), dst(host.copyDstLayoutCount);
        host.pCopySrcLayouts = src.data();
        host.pCopyDstLayouts = dst.data();
        properties2(state_.info.physical, &properties);
        const auto general = [](const std::vector<VkImageLayout>& layouts) {
            return std::find(layouts.begin(), layouts.end(), VK_IMAGE_LAYOUT_GENERAL) != layouts.end();
        };
        if (!general(src) || !general(dst)) {
            std::fprintf(stderr, "[rt] d3d12: host image copies do not accept GENERAL; using staging copies\n");
            return;
        }
        state_.copy_memory_to_image = to_image;
        state_.copy_image_to_memory = to_memory;
        state_.transition_image_layout = transition;
    }

    DeviceState state_;
    OwnedRuntime owned_;
    IUnknown* outer_ = nullptr;  // the Xbox-ABI proxy, which owns a reference to this device
};

}  // namespace

HRESULT create_vulkan_device(const VulkanDeviceInfo& info, REFIID riid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    auto* device = new Device(info);
    if (!device->ok()) {
        device->Release();
        return E_FAIL;
    }
    return hand_out(device, riid, out);
}

void set_device_outer(ID3D12Device* device, IUnknown* outer) {
    if (Device* d = Device::from<Device>(device)) d->set_outer(outer);
}

VkImage vulkan_image(ID3D12Resource* resource) {
    const Resource* r = Resource::from<Resource>(resource);
    return r ? r->image() : VK_NULL_HANDLE;
}

bool color_resolve_format_matches(DXGI_FORMAT format, ID3D12Resource* dst, ID3D12Resource* src) {
    if (!dst || !src) return false;
    const Format requested = vulkan_format(format, false);
    if (requested.typeless || requested.aspects != VK_IMAGE_ASPECT_COLOR_BIT) return false;
    if (format == DXGI_FORMAT_R32G32B32A32_UINT || format == DXGI_FORMAT_R8G8B8A8_UINT ||
        format == DXGI_FORMAT_R32_UINT)
        return false;
    const auto matches = [&](ID3D12Resource* resource) {
        const D3D12_RESOURCE_DESC desc = resource->GetDesc();
        return vulkan_format(desc.Format, desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL).format ==
               requested.format;
    };
    return matches(dst) && matches(src);
}

std::uint32_t device_max_push_constants_size(ID3D12Device* device) {
    if (!device) return 0;
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    if (!real) real = device;
    if (auto* d = Device::from<Device>(real)) return d->info().max_push_constants_size;
    return 0;
}

bool device_push_constants_need_fallback(ID3D12Device* device) {
    if (!device) return false;
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    if (!real) real = device;
    if (auto* d = Device::from<Device>(real)) return d->info().push_constants_need_fallback;
    return false;
}

namespace {
const leonida::BindlessSet* bindless_of(ID3D12Device* device) {
    if (!device) return nullptr;
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    const Device* d = Device::from<Device>(real ? real : device);
    return d ? d->bindless() : nullptr;
}
} // namespace

VkDescriptorSet bindless_set(ID3D12Device* device) {
    const leonida::BindlessSet* set = bindless_of(device);
    return set ? set->set() : VK_NULL_HANDLE;
}

VkDescriptorSetLayout bindless_set_layout(ID3D12Device* device) {
    const leonida::BindlessSet* set = bindless_of(device);
    return set ? set->layout() : VK_NULL_HANDLE;
}

// Creates its own Vulkan 1.3 instance and device. LEONIDA_VK_VALIDATION=1 adds VK_LAYER_KHRONOS_validation.
HRESULT create_vulkan_device_standalone(REFIID riid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    static const HMODULE library = LoadLibraryA("vulkan-1.dll");
    const auto get = library ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                                   reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr")))
                             : nullptr;
    if (!get) {
        std::fprintf(stderr, "[rt] d3d12: vulkan-1.dll not found\n");
        return E_FAIL;
    }
    const auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(get(nullptr, "vkCreateInstance"));
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "gta6rt";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    const char* layer = "VK_LAYER_KHRONOS_validation";
    char validation[8] = {};
    size_t length = 0;
    if (getenv_s(&length, validation, "LEONIDA_VK_VALIDATION") == 0 && length > 1 && validation[0] != '0') {
        instance_info.enabledLayerCount = 1;
        instance_info.ppEnabledLayerNames = &layer;
    }
    OwnedRuntime owned{};
    if (create_instance(&instance_info, nullptr, &owned.instance) != VK_SUCCESS) {
        std::fprintf(stderr, "[rt] d3d12: vkCreateInstance (Vulkan 1.3) failed\n");
        return E_FAIL;
    }
    owned.destroy_instance = reinterpret_cast<PFN_vkDestroyInstance>(get(owned.instance, "vkDestroyInstance"));
    const auto fail = [&](const char* why) {
        std::fprintf(stderr, "[rt] d3d12: %s\n", why);
        owned.destroy_instance(owned.instance, nullptr);
        return E_FAIL;
    };

#define LEONIDA_INSTANCE(name) const auto name = reinterpret_cast<PFN_##name>(get(owned.instance, #name))
    LEONIDA_INSTANCE(vkEnumeratePhysicalDevices);
    LEONIDA_INSTANCE(vkGetPhysicalDeviceProperties);
    LEONIDA_INSTANCE(vkGetPhysicalDeviceQueueFamilyProperties);
    LEONIDA_INSTANCE(vkGetPhysicalDeviceFeatures2);
    LEONIDA_INSTANCE(vkEnumerateDeviceExtensionProperties);
    LEONIDA_INSTANCE(vkCreateDevice);
    LEONIDA_INSTANCE(vkGetDeviceProcAddr);
#undef LEONIDA_INSTANCE
    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(owned.instance, &count, nullptr);
    std::vector<VkPhysicalDevice> physicals(count);
    vkEnumeratePhysicalDevices(owned.instance, &count, physicals.data());
    VulkanDeviceInfo info{};
    info.get_instance_proc = get;
    info.instance = owned.instance;
    info.separate_depth_stencil_layouts = true;
    VkPhysicalDeviceProperties chosen{};
    bool chosen_bc = false;
    VkPhysicalDeviceFeatures chosen_features{};
    VkPhysicalDeviceVulkan12Features chosen12{};
    for (VkPhysicalDevice candidate : physicals) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(candidate, &props);
        if (props.apiVersion < VK_API_VERSION_1_3) continue;
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
        vkGetPhysicalDeviceFeatures2(candidate, &features);
        const bool bc = features.features.textureCompressionBC;
        if (!f13.dynamicRendering || !f13.synchronization2 || !f12.timelineSemaphore ||
            !f12.separateDepthStencilLayouts)
            continue;
        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
        for (std::uint32_t i = 0; i < family_count; ++i) {
            if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            if (!info.physical || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                info.physical = candidate;
                info.queue_family = i;
                chosen = props;
                chosen_bc = bc;
                chosen_features = features.features;
                chosen12 = f12;
            }
            break;
        }
    }
    if (!info.physical)
        return fail("no Vulkan 1.3 GPU with dynamicRendering, synchronization2, timelineSemaphore and "
                    "separateDepthStencilLayouts");

    // VK_EXT_conditional_rendering for SetPredication, when the GPU has it (optional: docs/d3d12_vulkan.md 4.7).
    VkPhysicalDeviceConditionalRenderingFeaturesEXT conditional{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
    {
        std::uint32_t extension_count = 0;
        vkEnumerateDeviceExtensionProperties(info.physical, nullptr, &extension_count, nullptr);
        std::vector<VkExtensionProperties> extensions(extension_count);
        vkEnumerateDeviceExtensionProperties(info.physical, nullptr, &extension_count, extensions.data());
        for (const auto& e : extensions)
            if (!std::strcmp(e.extensionName, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME)) {
                VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &conditional};
                vkGetPhysicalDeviceFeatures2(info.physical, &features);
                info.conditional_rendering = conditional.conditionalRendering == VK_TRUE;
            }
        if (!info.conditional_rendering)
            std::fprintf(stderr, "[rt] d3d12: the GPU has no VK_EXT_conditional_rendering: SetPredication is unsupported\n");
    }
    VkPhysicalDeviceConditionalRenderingFeaturesEXT enable_conditional{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
    enable_conditional.conditionalRendering = VK_TRUE;

    VkPhysicalDeviceVulkan13Features enable13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                              info.conditional_rendering ? &enable_conditional : nullptr};
    enable13.dynamicRendering = VK_TRUE;
    enable13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &enable13};
    enable12.timelineSemaphore = VK_TRUE;
    enable12.separateDepthStencilLayouts = VK_TRUE;
    // Where the GPU has them: what the bindless set 0 needs (VulkanDeviceInfo::bindless), and the sampler features
    // D3D12 samplers can ask for.
    info.bindless = chosen12.runtimeDescriptorArray && chosen_features.shaderSampledImageArrayDynamicIndexing &&
                    chosen_features.shaderStorageImageArrayDynamicIndexing &&
                    chosen12.descriptorBindingSampledImageUpdateAfterBind &&
                    chosen12.descriptorBindingStorageImageUpdateAfterBind &&
                    chosen12.descriptorBindingStorageBufferUpdateAfterBind;
    info.bindless_partially_bound = info.bindless && chosen12.descriptorBindingPartiallyBound;
    info.sampler_anisotropy = chosen_features.samplerAnisotropy;
    info.sampler_mirror_clamp_to_edge = chosen12.samplerMirrorClampToEdge;
    enable12.runtimeDescriptorArray = info.bindless;
    enable12.descriptorBindingSampledImageUpdateAfterBind = info.bindless;
    enable12.descriptorBindingStorageImageUpdateAfterBind = info.bindless;
    enable12.descriptorBindingStorageBufferUpdateAfterBind = info.bindless;
    enable12.descriptorBindingPartiallyBound = info.bindless_partially_bound;
    enable12.samplerMirrorClampToEdge = info.sampler_mirror_clamp_to_edge;
    if (!info.bindless)
        std::fprintf(stderr, "[rt] d3d12: the GPU lacks a descriptor-indexing feature the bindless set needs\n");
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = info.queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures enable{};
    enable.textureCompressionBC = chosen_bc ? VK_TRUE : VK_FALSE;  // the game's textures are BC1-7
    enable.shaderSampledImageArrayDynamicIndexing = info.bindless;
    enable.shaderStorageImageArrayDynamicIndexing = info.bindless;
    enable.samplerAnisotropy = info.sampler_anisotropy;
    if (!chosen_bc) std::fprintf(stderr, "[rt] d3d12: the GPU has no BC texture support\n");
    const char* conditional_extension = VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME;

    // Optional: VK_EXT_host_image_copy lets WriteToSubresource/ReadFromSubresource copy on the CPU.
    std::uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(info.physical, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(info.physical, nullptr, &extension_count, extensions.data());
    const bool has_host_image_copy = std::any_of(extensions.begin(), extensions.end(), [](const auto& e) {
        return std::strcmp(e.extensionName, VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME) == 0;
    });
    VkPhysicalDeviceHostImageCopyFeaturesEXT host_copy{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT};
    if (has_host_image_copy) {
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &host_copy};
        vkGetPhysicalDeviceFeatures2(info.physical, &query);
    }
    const char* host_image_copy_name = VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME;
    info.host_image_copy = has_host_image_copy && host_copy.hostImageCopy;
    if (info.host_image_copy) {
        host_copy.pNext = enable13.pNext;  // after the conditional rendering features, if any
        enable13.pNext = &host_copy;
    } else {
        std::fprintf(stderr, "[rt] d3d12: no VK_EXT_host_image_copy; CPU texture copies use a staging submit\n");
    }

    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enable12};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pEnabledFeatures = &enable;
    std::vector<const char*> device_extensions;
    if (info.conditional_rendering) device_extensions.push_back(conditional_extension);
    if (info.host_image_copy) device_extensions.push_back(host_image_copy_name);
    device_info.enabledExtensionCount = std::uint32_t(device_extensions.size());
    device_info.ppEnabledExtensionNames = device_extensions.data();
    if (vkCreateDevice(info.physical, &device_info, nullptr, &owned.device) != VK_SUCCESS)
        return fail("vkCreateDevice failed");
    info.device = owned.device;
    owned.destroy_device = reinterpret_cast<PFN_vkDestroyDevice>(vkGetDeviceProcAddr(owned.device, "vkDestroyDevice"));
    reinterpret_cast<PFN_vkGetDeviceQueue>(vkGetDeviceProcAddr(owned.device, "vkGetDeviceQueue"))(
        owned.device, info.queue_family, 0, &info.queue);
    info.max_push_constants_size = chosen.limits.maxPushConstantsSize;
    info.push_constants_need_fallback = (info.max_push_constants_size < 144);
    std::fprintf(stderr, "[rt] d3d12: Vulkan device: %s (maxPushConstantsSize: %u bytes)\n", chosen.deviceName,
                 chosen.limits.maxPushConstantsSize);
    if (info.push_constants_need_fallback) {
        std::fprintf(stderr,
                     "[rt] d3d12: WARNING: GPU maxPushConstantsSize (%u bytes) is below the 144 bytes required by "
                     "the shader contract; uniform-buffer fallback required for the 16-byte tail\n",
                     chosen.limits.maxPushConstantsSize);
    }

    auto* device = new Device(info);
    device->own(owned);  // destroyed with the device from here on, even on failure
    if (!device->ok()) {
        device->Release();
        return E_FAIL;
    }
    return hand_out(device, riid, out);
}

}  // namespace rt::d3d12
