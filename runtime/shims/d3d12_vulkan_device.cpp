// A Vulkan-backed ID3D12Device (PC ABI) for the Xbox-ABI device proxy (d3d12_xs_device.cpp) to forward to.
// Implemented: queues, allocators, command lists, fences, descriptor heaps, render-target and depth-stencil views,
// committed buffers and textures (1D, 1D arrays, 2D, multisampled 2D, 3D) and their 1/2/3 variants, allocation
// info, CPU texture copies (WriteToSubresource/ReadFromSubresource), query heaps, a few feature queries. Everything else is a
// generated stub that reports itself once. See docs/d3d12_vulkan.md section 4.1/4.2 for what each answer means and
// what is still open.
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include "d3d12_com.h"

#include "d3d12_stubs.h"  // generated
#include "d3d12_com_object.h"
#include "d3d12_trace.h"
#include "d3d12_xs_device.h"
#include "dxgi_com.h"
#include "../src/gfx10_descriptors.h"
#include "../src/vulkan_bindless.h"
#include "../src/vulkan_memory.h"
#include "../src/resource_tiling.h"
#include "../src/shader_library.h"
#include "../src/vulkan_present.h"
#include "../src/vulkan_device_filter.h"

#include <vulkan/vulkan_win32.h>


#include <algorithm>
#include <atomic>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <iterator>
#include <map>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace rt::d3d12 {
namespace {

// Every descriptor slot is this many bytes; an RTV or DSV slot holds a leonida::TargetView. The console's own
// increments are open question 4 in docs/d3d12_vulkan.md; the application reads them from the device, as it must.
constexpr UINT kDescriptorSize = descriptor_size;
// The first 64 KiB of the GPU address arena are never handed out: VA 0 stays invalid, and the page is kept zero
// for unbound vertex and index buffers to read (D3D12 reads zeros there).
constexpr VkDeviceSize kArenaReserved = 64 * 1024;
static_assert(sizeof(leonida::TargetView) <= kDescriptorSize);

// The shader model and root signature version this layer can honour. Nothing in the runtime compiles DXIL: the
// game's shaders arrive as SPIR-V from the offline translator, and the only DXIL path (the BVH builder) is lowered
// to standard DXIL, whose baseline is 6.0. Answering the driver's own maximum would tell the game it can hand this
// layer shaders it has no front end for, which section 4.1 of docs/d3d12_vulkan.md calls worse than a false
// "unsupported" at boot. Raise these when a front end for the higher version lands.
constexpr D3D_SHADER_MODEL kHighestShaderModel = D3D_SHADER_MODEL_6_0;
constexpr D3D_ROOT_SIGNATURE_VERSION kHighestRootSignatureVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

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
    bool d24_as_float = false;  // D24_UNORM_S8_UINT stored as D32_SFLOAT_S8_UINT (no D24 on the GPU)
    bool d24 = false;           // a D24_UNORM_S8_UINT family format, native or stored as D32
    bool r9g9b9e5_as_r32_uint = false;  // R9G9B9E5_SHAREDEXP UAV stored as mutable R32_UINT (no E5B9G9R9 storage)
    // A 96-bit R32G32B32 format, held in the R32G32B32A32 image `format` names (ResourceBinding::rgb96).
    bool rgb96 = false;
};

// 96-bit textures are always RGBA32 images, even where the GPU has an RGB32 one (most have none with optimal
// tiling): D3D12 places subresources at multiples of 512 bytes, Vulkan wants a copy's buffer offset to be a
// multiple of the 12-byte texel (VUID-vkCmdCopyBufferToImage-dstImage-07975), so copies need the repack either
// way (docs section 4.2).
constexpr Format rgb96_format(VkFormat wide, bool typeless = false) {
    Format format{wide, VK_IMAGE_ASPECT_COLOR_BIT, typeless};
    format.rgb96 = true;
    return format;
}

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
    case DXGI_FORMAT_R32G32B32_TYPELESS: return rgb96_format(VK_FORMAT_R32G32B32A32_SFLOAT, true);
    case DXGI_FORMAT_R32G32B32_FLOAT: return rgb96_format(VK_FORMAT_R32G32B32A32_SFLOAT);
    case DXGI_FORMAT_R32G32B32_UINT: return rgb96_format(VK_FORMAT_R32G32B32A32_UINT);
    case DXGI_FORMAT_R32G32B32_SINT: return rgb96_format(VK_FORMAT_R32G32B32A32_SINT);
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return {VK_FORMAT_R16G16B16A16_SFLOAT, color, true};
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return {VK_FORMAT_R16G16B16A16_SFLOAT, color};
    case DXGI_FORMAT_R16G16B16A16_UNORM: return {VK_FORMAT_R16G16B16A16_UNORM, color};
    case DXGI_FORMAT_R16G16B16A16_UINT: return {VK_FORMAT_R16G16B16A16_UINT, color};
    case DXGI_FORMAT_R16G16B16A16_SINT: return {VK_FORMAT_R16G16B16A16_SINT, color};
    case DXGI_FORMAT_R32G32_TYPELESS: return {VK_FORMAT_R32G32_SFLOAT, color, true};
    case DXGI_FORMAT_R32G32_FLOAT: return {VK_FORMAT_R32G32_SFLOAT, color};
    case DXGI_FORMAT_R32G32_UINT: return {VK_FORMAT_R32G32_UINT, color};
    case DXGI_FORMAT_R32G32_SINT: return {VK_FORMAT_R32G32_SINT, color};
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return {VK_FORMAT_A2B10G10R10_UNORM_PACK32, color, true};
    case DXGI_FORMAT_R10G10B10A2_UNORM: return {VK_FORMAT_A2B10G10R10_UNORM_PACK32, color};
    case DXGI_FORMAT_R10G10B10A2_UINT: return {VK_FORMAT_A2B10G10R10_UINT_PACK32, color};
    // XR_BIAS is the same packed 10:10:10:2 bit layout as R10G10B10A2_UNORM. Vulkan has no XR_BIAS format:
    // the application biases and scales the channels in its shader, so the resource itself is an ordinary UNORM.
    case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM: return {VK_FORMAT_A2B10G10R10_UNORM_PACK32, color};
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP: return {VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, color};
    case DXGI_FORMAT_R11G11B10_FLOAT: return {VK_FORMAT_B10G11R11_UFLOAT_PACK32, color};
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return {VK_FORMAT_R8G8B8A8_UNORM, color, true};
    case DXGI_FORMAT_R8G8B8A8_UNORM: return {VK_FORMAT_R8G8B8A8_UNORM, color};
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return {VK_FORMAT_R8G8B8A8_SRGB, color};
    case DXGI_FORMAT_R8G8B8A8_SNORM: return {VK_FORMAT_R8G8B8A8_SNORM, color};
    case DXGI_FORMAT_R8G8B8A8_UINT: return {VK_FORMAT_R8G8B8A8_UINT, color};
    case DXGI_FORMAT_R8G8B8A8_SINT: return {VK_FORMAT_R8G8B8A8_SINT, color};
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return {VK_FORMAT_B8G8R8A8_UNORM, color, true};
    case DXGI_FORMAT_B8G8R8A8_UNORM: return {VK_FORMAT_B8G8R8A8_UNORM, color};
    // DXGI names channels from the low bits up, Vulkan from the high bits down: B in bits 0-3, A in 12-15 in both.
    // Vulkan 1.3 requires sampling it (formerly VK_EXT_4444_formats).
    case DXGI_FORMAT_B4G4R4A4_UNORM: return {VK_FORMAT_A4R4G4B4_UNORM_PACK16, color};
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return {VK_FORMAT_B8G8R8A8_SRGB, color};
    case DXGI_FORMAT_R16G16_TYPELESS: return {VK_FORMAT_R16G16_SFLOAT, color, true};
    case DXGI_FORMAT_R16G16_FLOAT: return {VK_FORMAT_R16G16_SFLOAT, color};
    case DXGI_FORMAT_R16G16_UINT: return {VK_FORMAT_R16G16_UINT, color};
    case DXGI_FORMAT_R16G16_SINT: return {VK_FORMAT_R16G16_SINT, color};
    case DXGI_FORMAT_R8G8_TYPELESS: return {VK_FORMAT_R8G8_UNORM, color, true};
    case DXGI_FORMAT_R8G8_UNORM: return {VK_FORMAT_R8G8_UNORM, color};
    case DXGI_FORMAT_R8G8_SNORM: return {VK_FORMAT_R8G8_SNORM, color};
    case DXGI_FORMAT_R16_UNORM: return {VK_FORMAT_R16_UNORM, color};
    // DXGI names packed channels from the high bits down, Vulkan's PACK formats the same way but spelled the other
    // way round: B5G6R5 (blue in bits 0-4) is R5G6B5_PACK16 and B5G5R5A1 (alpha in bit 15) is A1R5G5B5_PACK16.
    case DXGI_FORMAT_B5G6R5_UNORM: return {VK_FORMAT_R5G6B5_UNORM_PACK16, color};
    case DXGI_FORMAT_B5G5R5A1_UNORM: return {VK_FORMAT_A1R5G5B5_UNORM_PACK16, color};
    case DXGI_FORMAT_R8G8_UINT: return {VK_FORMAT_R8G8_UINT, color};
    case DXGI_FORMAT_R8G8_SINT: return {VK_FORMAT_R8G8_SINT, color};
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

Format texture_storage_format(DXGI_FORMAT format, bool depth) {
    return vulkan_format(format == DXGI_FORMAT_A8_UNORM ? DXGI_FORMAT_R8_UNORM : format, depth);
}

// The three-channel format a view of a 96-bit texture names in its T# (GFX10_FORMAT_32_32_32_*): the format
// whose channels the view's swizzle is taken from, so alpha reads as one whatever the image's fourth channel holds.
VkFormat rgb96_named_format(VkFormat image_view_format) {
    switch (image_view_format) {
    case VK_FORMAT_R32G32B32A32_SFLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
    case VK_FORMAT_R32G32B32A32_UINT: return VK_FORMAT_R32G32B32_UINT;
    case VK_FORMAT_R32G32B32A32_SINT: return VK_FORMAT_R32G32B32_SINT;
    default: return VK_FORMAT_UNDEFINED;
    }
}
// And back: the format of the view of the RGBA32 image that shows such a T#.
VkFormat rgb96_view_format(VkFormat image_format, std::uint32_t gfx10_format) {
    for (const VkFormat wide : {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT})
        if (leonida::gfx10::format_from_vulkan(rgb96_named_format(wide)) == gfx10_format)
            return rgb96_named_format(image_format) != VK_FORMAT_UNDEFINED ? wide : VK_FORMAT_UNDEFINED;
    return VK_FORMAT_UNDEFINED;
}

// Bytes per texel or block as D3D12 lays the format out in a footprint.
leonida::BlockFormat footprint_block(const Format& format, VkImageAspectFlags aspect) {
    leonida::BlockFormat block = leonida::block_format(format.format, aspect);
    if (format.rgb96 && block.bytes) block.bytes = leonida::rgb96_bytes;
    return block;
}

// 96-bit textures are sampled and copied, nothing else yet: the flags creation refuses, and
// GetResourceAllocationInfo with it. The PC runtime (WARP) refuses unordered access and depth too; it does make
// 96-bit render targets (d3d12_rgb96_test prints both devices' answers).
bool rgb96_refused(const D3D12_RESOURCE_DESC& desc) {
    return vulkan_format(desc.Format, false).rgb96 &&
           (desc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                          D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL));
}

// The full mip chain's length for a texture this size.
UINT16 full_mips(UINT64 width, UINT height, UINT depth) {
    UINT16 mips = 0;
    for (UINT64 size = std::max<UINT64>({width, height, depth}); size; size >>= 1) ++mips;
    return mips;
}

// XR_BIAS is a display format: the PC runtime accepts it only as a single-mip, single-sample 2D texture, and
// refuses 3D, extra mips and multisampling in creation, allocation and footprints alike.
bool xr_bias_compatible(const D3D12_RESOURCE_DESC& desc) {
    if (desc.Format != DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM) return true;
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1) return false;
    const UINT mips = desc.MipLevels ? desc.MipLevels : full_mips(desc.Width, desc.Height, 1);
    return mips == 1;
}

// A volume RTV covers W slices of one mip. UINT_MAX extends through the last slice of that mip.
leonida::TargetView volume_rtv(leonida::ResourceId id, const D3D12_RESOURCE_DESC& resource,
                               const D3D12_RENDER_TARGET_VIEW_DESC* view) {
    if (!(resource.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) || vulkan_format(resource.Format, false).typeless)
        return {};
    if (view && (view->ViewDimension != D3D12_RTV_DIMENSION_TEXTURE3D ||
                 (view->Format != DXGI_FORMAT_UNKNOWN && view->Format != resource.Format))) return {};
    const D3D12_TEX3D_RTV slices = view ? view->Texture3D : D3D12_TEX3D_RTV{0, 0, UINT_MAX};
    if (slices.MipSlice >= resource.MipLevels) return {};
    const UINT depth = std::max(1u, UINT(resource.DepthOrArraySize) >> slices.MipSlice);
    if (slices.FirstWSlice >= depth) return {};
    const UINT remaining = depth - slices.FirstWSlice;
    const UINT count = slices.WSize == UINT_MAX ? remaining : slices.WSize;
    if (!count || count > remaining) return {};
    return {id, slices.MipSlice, slices.FirstWSlice, count};
}

// The VkImage a texture description becomes, with its mip count already resolved (MipLevels != 0). Shared by
// texture creation and GetResourceAllocationInfo, so the sizes reported are the sizes later allocated.
// Keeps the optional VkImageFormatListCreateInfo alongside VkImageCreateInfo so info.pNext stays valid.
// attachment_usable: the GPU can render to `format`. A UAV texture then also gets COLOR_ATTACHMENT usage, because
// ClearUnorderedAccessView* with rectangles clears through an attachment (vkCmdClearAttachments).
struct TextureCreateInfo {
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    VkImageFormatListCreateInfo format_list{VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
    std::array<VkFormat, 2> view_formats{};

    TextureCreateInfo(const D3D12_RESOURCE_DESC& desc, const Format& format, bool attachment_usable) {
        const bool volume = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        const bool one_d = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D;
        info.flags = format.typeless ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
        if (volume && (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET))
            info.flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
        info.imageType = volume ? VK_IMAGE_TYPE_3D : one_d ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D;
        info.format = format.format;
        info.extent = {std::uint32_t(desc.Width), one_d ? 1u : desc.Height,
                       volume ? UINT(desc.DepthOrArraySize) : 1u};
        info.mipLevels = desc.MipLevels;
        info.arrayLayers = volume ? 1 : desc.DepthOrArraySize;
        info.samples = VkSampleCountFlagBits(std::max(1u, desc.SampleDesc.Count));
        info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) info.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
            info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
            if (!volume && info.samples == VK_SAMPLE_COUNT_1_BIT && attachment_usable)
                info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        }
        if (!(desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)) info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if (format.r9g9b9e5_as_r32_uint) {
            // R32_UINT owns storage and attachment usages; E5B9G9R9 views restrict usage to SAMPLED_BIT.
            info.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
            view_formats = {VK_FORMAT_R32_UINT, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32};
            format_list.viewFormatCount = 2;
            format_list.pViewFormats = view_formats.data();
            info.pNext = &format_list;
        }
    }

    TextureCreateInfo(const TextureCreateInfo&) = delete;
    TextureCreateInfo& operator=(const TextureCreateInfo&) = delete;
};

// What a device's children need from it, kept apart from the COM class so children can hold a plain pointer.
struct DeviceState {
    VulkanDeviceInfo info;
    std::unique_ptr<leonida::VulkanContext> context;
    VkPhysicalDeviceMemoryProperties memory{};
    // D3D-96: upload-heap buffers prefer DEVICE_LOCAL | HOST_VISIBLE memory, so the GPU reads them from VRAM. Only
    // with Resizable BAR (leonida::resizable_bar); LEONIDA_UPLOAD_SYSMEM=1 keeps them in system memory to compare.
    bool upload_in_vram = false;

    // What D3D12_OPTIONS1 answers with: the physical device's own subgroup facts, read on first use.
    struct SubgroupFacts {
        UINT lane_min = 0;
        UINT lane_max = 0;
        bool wave_ops = false;
    };
    SubgroupFacts subgroups;
    bool subgroups_read = false;
    PFN_vkGetPhysicalDeviceFormatProperties format_properties = nullptr;
    PFN_vkGetPhysicalDeviceImageFormatProperties image_format_properties = nullptr;
    PFN_vkGetPhysicalDeviceImageFormatProperties2 image_format_properties2 = nullptr;
    PFN_vkGetDeviceImageMemoryRequirements image_requirements = nullptr;  // Vulkan 1.3 or VK_KHR_maintenance4
    LUID luid{};       // VkPhysicalDeviceIDProperties::deviceLUID; zero when the driver reports none
    QueueClock clock;  // what every queue's GetClockCalibration uses
    PFN_vkGetPhysicalDeviceFormatProperties2 format_properties2 = nullptr;
    // True when VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 lacks native STORAGE_IMAGE but mutable R32_UINT+{R32_UINT,E5B9G9R9}
    // is supported, so R9G9B9E5_SHAREDEXP UAV textures are backed by R32_UINT with separate storage/sampled views.
    bool r9g9b9e5_uav_emulated = false;

    // Cached physical-device no-attachment sample counts for D3D12_OPTIONS19.
    UINT no_output_sample_counts = 0;
    float max_sampler_lod_bias = 0;
    // VkPhysicalDeviceDepthStencilResolveProperties: the modes a depth or stencil ResolveSubresourceRegion can use.
    VkResolveModeFlags depth_resolve_modes = 0;
    VkResolveModeFlags stencil_resolve_modes = 0;

    // VK_EXT_host_image_copy; null unless the device enabled it and its copies accept GENERAL.
    PFN_vkCopyMemoryToImageEXT copy_memory_to_image = nullptr;
    PFN_vkCopyImageToMemoryEXT copy_image_to_memory = nullptr;
    PFN_vkTransitionImageLayoutEXT transition_image_layout = nullptr;

    // WriteToSubresource/ReadFromSubresource without host image copy: one staging copy at a time, submitted to
    // the device's VkQueue and waited for. Created on first use.
    std::mutex staging_mutex;
    std::unique_ptr<leonida::VulkanQueue> staging_queue;

    // The GPU address arena (docs/d3d12_vulkan.md section 3.3): one host-visible, device-addressable VkBuffer that
    // stands in for the console's GPU address space. A console VA is an offset in it, and heap_base (push-constant
    // dwords 32-33) is its device address. It is at most 4 GiB, so every VA fits the 32-bit descriptor-table
    // pointers some shaders use. The first 64 KiB are never handed out, so no resource has VA 0.
    struct Arena {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        std::uint8_t* mapped = nullptr;
        VkDeviceAddress address = 0;
        VkDeviceSize size = 0;
        std::mutex mutex;
        std::map<VkDeviceSize, VkDeviceSize> free;  // offset -> size
    } arena;

    // A zero-size range is refused: it would share its offset with the next range, and its release would collide
    // with that range's free entry. The fit test cannot wrap: free blocks lie inside the arena (below 4 GiB).
    bool arena_allocate(VkDeviceSize bytes, VkDeviceSize alignment, VkDeviceSize& out) {
        if (!bytes || bytes > arena.size) return false;
        std::lock_guard lock(arena.mutex);
        for (auto it = arena.free.begin(); it != arena.free.end(); ++it) {
            const VkDeviceSize start = (it->first + alignment - 1) / alignment * alignment;
            const VkDeviceSize end = it->first + it->second;
            if (start >= end || bytes > end - start) continue;
            const VkDeviceSize block = it->first;
            arena.free.erase(it);
            if (start > block) arena.free[block] = start - block;
            if (start + bytes < end) arena.free[start + bytes] = end - (start + bytes);
            out = start;
            return true;
        }
        return false;
    }

    void arena_release(VkDeviceSize offset, VkDeviceSize bytes) {
        if (!bytes) return;
        std::lock_guard lock(arena.mutex);
        auto it = arena.free.emplace(offset, bytes).first;
        if (auto next = std::next(it); next != arena.free.end() && it->first + it->second == next->first) {
            it->second += next->second;
            arena.free.erase(next);
        }
        if (it != arena.free.begin()) {
            if (auto previous = std::prev(it); previous->first + previous->second == it->first) {
                previous->second += it->second;
                arena.free.erase(it);
            }
        }
    }

    // The translator's output, loaded on first use.
    std::mutex library_mutex;
    bool library_tried = false;
    leonida::ShaderLibrary shader_library;

    const leonida::ShaderLibrary* library() {
        std::lock_guard lock(library_mutex);
        if (!library_tried) {
            library_tried = true;
            std::string dir = info.shader_dir;
            if (dir.empty()) {
                char buffer[1024] = {};
                size_t length = 0;
                if (getenv_s(&length, buffer, "LEONIDA_SHADERS") == 0 && length > 1) dir = buffer;
            }
            std::string error;
            if (dir.empty())
                std::fprintf(stderr, "[rt] d3d12: no translated shaders: set LEONIDA_SHADERS to <work>/shaders/spirv\n");
            else if (!shader_library.load(dir, &error))
                std::fprintf(stderr, "[rt] d3d12: %s\n", error.c_str());
            else
                std::fprintf(stderr, "[rt] d3d12: %zu translated shaders from %s\n", shader_library.size(), dir.c_str());
        }
        return shader_library.loaded() ? &shader_library : nullptr;
    }

    bool depth_usable(VkFormat format) const {
        VkFormatProperties properties{};
        format_properties(info.physical, format, &properties);
        return properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }

    bool color_attachment_usable(VkFormat format) const {
        VkFormatProperties properties{};
        format_properties(info.physical, format, &properties);
        return (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0;
    }

    bool storage_usable(VkFormat format) const {
        VkFormatProperties properties{};
        format_properties(info.physical, format, &properties);
        return (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
    }

    bool sampled_usable(VkFormat format) const {
        VkFormatProperties properties{};
        format_properties(info.physical, format, &properties);
        return (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
    }

    // Whether the GPU needs and supports backing R9G9B9E5_SHAREDEXP UAV textures with a mutable R32_UINT image.
    bool probe_r9g9b9e5_uav_emulation() const {
        if (storage_usable(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32)) return false;
        if (!image_format_properties2 || !storage_usable(VK_FORMAT_R32_UINT) ||
            !sampled_usable(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32))
            return false;
        constexpr VkFormat view_formats[2] = {VK_FORMAT_R32_UINT, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32};
        VkImageFormatListCreateInfo list{VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
        list.viewFormatCount = 2;
        list.pViewFormats = view_formats;
        VkPhysicalDeviceImageFormatInfo2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, &list};
        query.format = VK_FORMAT_R32_UINT;
        query.type = VK_IMAGE_TYPE_2D;
        query.tiling = VK_IMAGE_TILING_OPTIMAL;
        query.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        query.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        VkImageFormatProperties2 props{VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
        return image_format_properties2(info.physical, &query, &props) == VK_SUCCESS;
    }

    // The limits apply to this exact combination of format, type, tiling, usage, create flags and pNext format list.
    HRESULT check_image_support(const VkImageCreateInfo& create) const {
        VkImageFormatProperties limits{};
        VkResult result = VK_ERROR_FORMAT_NOT_SUPPORTED;
        if (image_format_properties2) {
            VkPhysicalDeviceImageFormatInfo2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
            query.pNext = create.pNext;
            query.format = create.format;
            query.type = create.imageType;
            query.tiling = create.tiling;
            query.usage = create.usage;
            query.flags = create.flags;
            VkImageFormatProperties2 props{VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
            result = image_format_properties2(info.physical, &query, &props);
            limits = props.imageFormatProperties;
        } else {
            result = image_format_properties(info.physical, create.format, create.imageType, create.tiling,
                                             create.usage, create.flags, &limits);
        }
        if (result == VK_ERROR_FORMAT_NOT_SUPPORTED) return E_INVALIDARG;
        if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY) return E_OUTOFMEMORY;
        if (result != VK_SUCCESS) return E_FAIL;
        if (create.extent.width > limits.maxExtent.width || create.extent.height > limits.maxExtent.height ||
            create.extent.depth > limits.maxExtent.depth || create.mipLevels > limits.maxMipLevels ||
            create.arrayLayers > limits.maxArrayLayers || !(limits.sampleCounts & create.samples))
            return E_INVALIDARG;
        return S_OK;
    }

    // D3D12_OPTIONS1's wave facts, from the physical device: which subgroup operations exist and what wave sizes it
    // can run. Zeros and false when the device cannot be asked, which the query then reports as unsupported.
    const SubgroupFacts& subgroup_facts() {
        if (subgroups_read) return subgroups;
        subgroups_read = true;
        const auto properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            info.get_instance_proc
                ? info.get_instance_proc(info.instance, "vkGetPhysicalDeviceProperties2")
                : nullptr);
        if (!properties2 || !info.physical) return subgroups;
        VkPhysicalDeviceSubgroupSizeControlProperties size_control{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
                                                   &size_control};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroup};
        properties2(info.physical, &properties);
        // Every group the wave intrinsics map to, because D3D12 promises them all behind the one bool.
        constexpr VkSubgroupFeatureFlags kD3D12WaveOps =
            VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
            VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        subgroups.wave_ops = (subgroup.supportedOperations & kD3D12WaveOps) == kD3D12WaveOps;
        subgroups.lane_min = size_control.minSubgroupSize ? size_control.minSubgroupSize : subgroup.subgroupSize;
        subgroups.lane_max = size_control.maxSubgroupSize ? size_control.maxSubgroupSize : subgroups.lane_min;
        if (subgroups.lane_max < subgroups.lane_min) subgroups.lane_max = subgroups.lane_min;
        // D3D-121: Intel's D3D12 driver answers a fixed 16..16 on its integrated GPUs, where Vulkan reports the
        // compiler's whole 8..32 choice. Measured on Xe-LP: UHD 770 (32.0.101.6129) and Raptor Lake-S Mobile
        // (101.7092). Discrete Arc is not measured yet (TEST-57), so it keeps the Vulkan range.
        constexpr uint32_t kIntelVendorId = 0x8086;
        constexpr UINT kIntelD3D12Lanes = 16;
        if (properties.properties.vendorID == kIntelVendorId &&
            properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU &&
            subgroups.lane_min <= kIntelD3D12Lanes && kIntelD3D12Lanes <= subgroups.lane_max)
            subgroups.lane_min = subgroups.lane_max = kIntelD3D12Lanes;
        return subgroups;
    }

    // Integer sampled images need not support multisampling. Pick a supported count for the
    // zero image in the corresponding set-0 bindings; one means that class is unavailable.
    VkSampleCountFlagBits ms_fallback_samples(VkFormat format) const {
        VkImageFormatProperties props{};
        constexpr VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (image_format_properties(info.physical, format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                    usage, 0, &props) == VK_SUCCESS) {
            for (UINT count = D3D12_MAX_MULTISAMPLE_SAMPLE_COUNT; count > 1; count >>= 1)
                if (props.sampleCounts & count) return VkSampleCountFlagBits(count);
        }
        return VK_SAMPLE_COUNT_1_BIT;
    }

    // The format a texture description is created with, D24S8 and R9G9B9E5 UAV fallbacks included.
    Format texture_format(const D3D12_RESOURCE_DESC& desc) const {
        if (!xr_bias_compatible(desc)) return {};
        if (desc.Format == DXGI_FORMAT_A8_UNORM &&
            (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
             (desc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))))
            return {};
        if (desc.Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP) {
            if (desc.SampleDesc.Count != 1 ||
                (desc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)))
                return {};
            if ((desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) &&
                (!color_attachment_usable(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) ||
                 ((desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) &&
                  !storage_usable(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32))))
                return {};
            if ((desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) && r9g9b9e5_uav_emulated) {
                log_once("R9G9B9E5_SHAREDEXP UAV unavailable natively, using mutable R32_UINT; format", desc.Format);
                return {VK_FORMAT_R32_UINT, VK_IMAGE_ASPECT_COLOR_BIT, false, false, false, true};
            }
        }
        if (rgb96_refused(desc)) {
            if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
                log_once("96-bit (R32G32B32) render targets not supported; format", desc.Format);
            return {};
        }
        Format format = texture_storage_format(desc.Format, desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        format.d24 = format.format == VK_FORMAT_D24_UNORM_S8_UINT;  // native or not: clears are quantised
        if (format.format == VK_FORMAT_D24_UNORM_S8_UINT && !depth_usable(format.format)) {
            log_once("D24_UNORM_S8_UINT unavailable, using D32_SFLOAT_S8_UINT; format", desc.Format);
            format.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
            format.d24_as_float = true;
        }
        return format;
    }

    // The memory an image made from `create` needs, or false if the device can't make that image.
    bool memory_requirements(const VkImageCreateInfo& create, VkMemoryRequirements& out) const {
        if (FAILED(check_image_support(create))) return false;
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
        const std::uint64_t slot_handle =
            (static_cast<std::uint64_t>(entry.dimension) << 32) | static_cast<std::uint64_t>(entry.index);
        context->defer_destroy(leonida::DeferredDeletionQueue::Kind::bindless_slot, slot_handle);
        context->defer_destroy(leonida::DeferredDeletionQueue::Kind::image_view, entry.sampled);
        if (entry.storage) context->defer_destroy(leonida::DeferredDeletionQueue::Kind::image_view, entry.storage);
    }

    // Gives the T# `words` a view in set 0 and an entry in the hash table, unless it has them. `storage`: shaders
    // write through it too (an unordered-access view), so it also goes into the storage arrays, which are
    // shorter. False when set 0 cannot show the image; the descriptor then finds the fallback at index 0.
    bool bind_image(const leonida::gfx10::ImageWords& words, const leonida::gfx10::Image& image, VkImage vk_image,
                    VkFormat image_format, VkImageAspectFlags aspects, bool storage,
                    bool allow_channel_cast = false, bool r9g9b9e5_as_r32_uint = false) {
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
        case g::ImageType::image_2d_msaa: dimension = leonida::ImageDimension::dim_2d_ms; break;
        case g::ImageType::image_2d_msaa_array: dimension = leonida::ImageDimension::dim_2d_ms_array; break;
        default:
            log_once("set 0 has no array for this image type:", int(image.type));
            return false;
        }
        // A T# of 8_UINT on a depth-stencil image is a view of its stencil plane (image_descriptor, D3D-75). Vulkan
        // views it in the image's own format with the stencil aspect, and returns the value in R, as a T# whose
        // format is 8_UINT reads it from X.
        const bool stencil = (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) && image.format == g::stencil_format;
        const VkFormat logical_format = r9g9b9e5_as_r32_uint ? VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 : image_format;
        VkFormat view_format =
            stencil ? image_format : g::vulkan_view_format(logical_format, image.format, allow_channel_cast);
        VkFormat channels_format = view_format;  // whose channels the T#'s swizzle selects from
        if (view_format == VK_FORMAT_UNDEFINED) {  // a 96-bit texture's T# names RGB32; its image is RGBA32
            view_format = rgb96_view_format(image_format, image.format);
            channels_format = rgb96_named_format(view_format);
        }
        if (view_format == VK_FORMAT_UNDEFINED) {
            log_once("no Vulkan view format for the GFX10 format of a view:", int(image.format));
            return false;
        }
        const unsigned sampled_class = g::multisampled(image.type) ? numeric_class(view_format) : 0;
        if (g::multisampled(image.type) && !bindless->supports_ms_class(sampled_class)) {
            log_once("set 0 has no multisample binding for this numeric class:", int(sampled_class));
            return false;
        }
        auto& vk = context->vk;
        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = vk_image;
        view_info.viewType = g::vulkan_view_type(image.type);
        view_info.format = view_format;
        VkImageViewUsageCreateInfo sampled_usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
        if (r9g9b9e5_as_r32_uint) {
            sampled_usage.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
            view_info.pNext = &sampled_usage;
        }
        auto swizzle = image.swizzle;
        VkImageViewMinLodCreateInfoEXT clamp{VK_STRUCTURE_TYPE_IMAGE_VIEW_MIN_LOD_CREATE_INFO_EXT};
        const bool supports_lod_clamp = image.type == g::ImageType::image_1d || image.type == g::ImageType::image_2d ||
                                        image.type == g::ImageType::image_2d_array || image.type == g::ImageType::cube;
        if (!storage && supports_lod_clamp && (aspects & VK_IMAGE_ASPECT_COLOR_BIT)) {
            // Use the clamp in the T# itself: views with identical keys must sample identically.
            clamp.minLod = g::image_min_lod(words);
            if (clamp.minLod > float(image.first_mip + image.mip_count - 1)) {
                // The accessible mip set is empty. Keep the dimensions and constant/missing channels.
                for (auto& channel : swizzle)
                    if (channel != g::select_one) channel = g::select_zero;
            } else if (clamp.minLod > float(image.first_mip)) {
                if (!info.image_view_min_lod) {
                    log_once("SRV requires VK_EXT_image_view_min_lod; first mip", int(image.first_mip));
                    return false;
                }
                clamp.pNext = view_info.pNext;
                view_info.pNext = &clamp;
            }
        }
        view_info.components = g::vulkan_swizzle(swizzle, channels_format);
        // The cast: MSVC's enums are ints, and it rejects int to VkImageAspectFlags in braces as narrowing.
        view_info.subresourceRange = {
            VkImageAspectFlags(stencil                                   ? VK_IMAGE_ASPECT_STENCIL_BIT
                               : (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                                       : VK_IMAGE_ASPECT_COLOR_BIT),
            image.first_mip, image.mip_count, image.first_layer, image.layer_count};
        ImageEntry entry{dimension, 0, VK_NULL_HANDLE, VK_NULL_HANDLE, image.address};
        if (vk.vkCreateImageView(info.device, &view_info, nullptr, &entry.sampled) != VK_SUCCESS) return false;
        leonida::StorageViews views{};
        if (storage) {
            VkImageViewCreateInfo storage_info = view_info;
            storage_info.components = {};
            VkImageViewUsageCreateInfo storage_usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
            const VkFormat storage_format = r9g9b9e5_as_r32_uint ? VK_FORMAT_R32_UINT : view_format;
            if (r9g9b9e5_as_r32_uint) {
                storage_info.format = storage_format;
                storage_usage.usage = VK_IMAGE_USAGE_STORAGE_BIT;
                storage_info.pNext = &storage_usage;
            }
            if (vk.vkCreateImageView(info.device, &storage_info, nullptr, &entry.storage) != VK_SUCCESS) {
                vk.vkDestroyImageView(info.device, entry.sampled, nullptr);
                return false;
            }
            const int numeric = numeric_class(storage_format);
            (numeric == 1 ? views.as_uint : numeric == 2 ? views.as_sint : views.as_float) = entry.storage;
            if (storage_format == VK_FORMAT_R32_UINT) views.r32ui = entry.storage;
        }
        // Every state a shader can read or write the texture in is GENERAL (ResourceBinding::sampled_in_general).
        entry.index = bindless->add_image(dimension, entry.sampled, VK_IMAGE_LAYOUT_GENERAL, views, sampled_class);
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
        VkSamplerCreateInfo sampler_info = g::vulkan_sampler(sampler, max_sampler_lod_bias);
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

    // A memory type with `required`, preferring one that also has `preferred`, in a heap large enough for the
    // allocation (leonida::memory_type_order). `chosen`, when given, receives the property flags of the type the
    // memory came from.
    VkDeviceMemory allocate(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags required,
                            VkMemoryPropertyFlags preferred, VkMemoryPropertyFlags* chosen = nullptr,
                            bool device_address = false) const {
        VkMemoryAllocateFlagsInfo flags_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        for (const std::uint32_t i :
             leonida::memory_type_order(memory, requirements.memoryTypeBits, required, preferred, requirements.size)) {
            VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            if (device_address) allocate.pNext = &flags_info;
            allocate.allocationSize = requirements.size;
            allocate.memoryTypeIndex = i;
            VkDeviceMemory out = VK_NULL_HANDLE;
            if (context->vk.vkAllocateMemory(info.device, &allocate, nullptr, &out) == VK_SUCCESS) {
                if (chosen) *chosen = memory.memoryTypes[i].propertyFlags;
                return out;
            }
        }
        return VK_NULL_HANDLE;
    }
};

// The Vulkan format for a texture this layer can create, with the D24 fallback; VK_FORMAT_UNDEFINED, after a log
// line, when it cannot. Committed and placed textures share it so both refuse the same formats for the same reason.
Format texture_format(const DeviceState& device, const D3D12_RESOURCE_DESC& desc) {
    Format format = vulkan_format(desc.Format, desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    if (format.format == VK_FORMAT_D24_UNORM_S8_UINT && !device.depth_usable(format.format)) {
        log_once("D24_UNORM_S8_UINT unavailable, using D32_SFLOAT_S8_UINT; format", desc.Format);
        format.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    }
    if (format.format == VK_FORMAT_UNDEFINED) log_once("texture format not supported yet:", desc.Format);
    return format;
}

// The castable-format list a committed or placed texture was created with, mapped to Vulkan (D3D-85's committed
// path, D3D-90's placed one). Each format must map and be the same texel size as the resource's: the host PC
// runtime refuses other sizes, and a size-mismatched VkImageFormatListCreateInfo is a Vulkan validation error
// (VUID-VkImageCreateInfo-pNext-06722). A nonzero count with a null list is no list. E_INVALIDARG on a bad format.
HRESULT castable_view_formats(const Format& format, UINT32 count, const DXGI_FORMAT* castable,
                              std::vector<VkFormat>& out) {
    if (!count || !castable) return S_OK;
    const bool depth = (format.aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0;
    const leonida::BlockFormat own = leonida::block_format(format.format, format.aspects);
    for (UINT32 i = 0; i < count; ++i) {
        const Format cast = vulkan_format(castable[i], depth);
        if (cast.format == VK_FORMAT_UNDEFINED) {
            log_once("castable format not supported yet:", castable[i]);
            return E_INVALIDARG;
        }
        const leonida::BlockFormat block = leonida::block_format(cast.format, cast.aspects);
        if (own.bytes && block.bytes && (block.bytes != own.bytes || block.block != own.block)) {
            log_once("castable format is a different texel size than the resource:", castable[i]);
            return E_INVALIDARG;
        }
        if (std::find(out.begin(), out.end(), cast.format) == out.end()) out.push_back(cast.format);
    }
    return S_OK;
}

class DescriptorHeap final
    : public Child<DescriptorHeap, ID3D12DescriptorHeapStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable,
                   ID3D12DescriptorHeap> {
public:
    // {4a7d2c90-5e1b-4f63-8c0a-91d3b6e2f417}
    static constexpr GUID kTag{0x4a7d2c90, 0x5e1b, 0x4f63, {0x8c, 0x0a, 0x91, 0xd3, 0xb6, 0xe2, 0xf4, 0x17}};

    // A shader-visible heap lives in the GPU address arena: its CPU handles point at the arena's host mapping and
    // its GPU handle is the console VA, so the bytes the game or Create*View writes are what shaders load. Other
    // heaps (RTV, DSV, non-visible staging heaps) stay in host memory.
    // A shader-visible heap that the arena cannot hold is not created (ok() is false, E_OUTOFMEMORY), as a
    // default-heap buffer is not: in host memory its GPU handle would be 0, and every table would read VA 0.
    DescriptorHeap(DeviceState& device, const D3D12_DESCRIPTOR_HEAP_DESC& desc) : device_(device), desc_(desc) {
        const VkDeviceSize bytes = VkDeviceSize(desc.NumDescriptors) * kDescriptorSize;
        if ((desc.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) && device.arena.mapped) {
            if (!device.arena_allocate(bytes, 256, arena_offset_)) {
                log_once("GPU address arena full (raise LEONIDA_ARENA_MB); shader-visible heap size in KiB:",
                         int(std::min<VkDeviceSize>(bytes / 1024, INT_MAX)));
                return;
            }
            in_arena_ = true;
            cpu_ = device.arena.mapped + arena_offset_;
            std::memset(cpu_, 0, std::size_t(bytes));
        } else {
            storage_.reset(new std::byte[std::size_t(bytes)]());
            cpu_ = reinterpret_cast<std::uint8_t*>(storage_.get());
        }
    }

    ~DescriptorHeap() override {
        if (in_arena_) device_.arena_release(arena_offset_, VkDeviceSize(desc_.NumDescriptors) * kDescriptorSize);
    }

    bool ok() const { return cpu_ != nullptr; }

    // A node mask of 0 reads back as the one node of a single adapter, as on the PC runtime (d3d12_node_mask_test).
    D3D12_DESCRIPTOR_HEAP_DESC STDMETHODCALLTYPE GetDesc() override {
        D3D12_DESCRIPTOR_HEAP_DESC desc = desc_;
        if (!desc.NodeMask) desc.NodeMask = 1;
        return desc;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE STDMETHODCALLTYPE GetCPUDescriptorHandleForHeapStart() override {
        return {reinterpret_cast<SIZE_T>(cpu_)};
    }
    D3D12_GPU_DESCRIPTOR_HANDLE STDMETHODCALLTYPE GetGPUDescriptorHandleForHeapStart() override {
        if (!(desc_.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)) return {0};
        if (!in_arena_) log_once("shader-visible heap without a GPU address arena; flags", int(desc_.Flags));
        return {in_arena_ ? UINT64(arena_offset_) : 0};
    }

private:
    DeviceState& device_;
    D3D12_DESCRIPTOR_HEAP_DESC desc_;
    std::unique_ptr<std::byte[]> storage_;
    std::uint8_t* cpu_ = nullptr;
    VkDeviceSize arena_offset_ = 0;
    bool in_arena_ = false;
};

class Resource final
    : public Child<Resource, ID3D12ResourceStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable, ID3D12Resource> {
public:
    // {9b3e5f12-7c48-4d0e-a6b1-2f8d4c7e9a35}
    static constexpr GUID kTag{0x9b3e5f12, 0x7c48, 0x4d0e, {0xa6, 0xb1, 0x2f, 0x8d, 0x4c, 0x7e, 0x9a, 0x35}};

    Resource(DeviceState& device, const D3D12_RESOURCE_DESC& desc, const D3D12_HEAP_PROPERTIES& heap,
             D3D12_HEAP_FLAGS heap_flags)
        : device_(device), desc_(desc), heap_(heap), heap_flags_(heap_flags) {}

    // A placed resource lives inside a heap's arena range: the heap owns the range, so only the console VA is kept
    // here and nothing goes back to the arena when this object dies.
    Resource(DeviceState& device, const D3D12_RESOURCE_DESC& desc, const D3D12_HEAP_PROPERTIES& heap,
             D3D12_HEAP_FLAGS heap_flags, VkDeviceSize placed_va)
        : device_(device), desc_(desc), heap_(heap), heap_flags_(heap_flags), va_(placed_va) {}

    ~Resource() override {
        if (accel_) {
            device_.context->unregister_acceleration_structure(gpu_va_);
            if (buffer_va_ && buffer_va_ != gpu_va_) device_.context->unregister_acceleration_structure(buffer_va_);
            device_.context->defer_destroy(leonida::DeferredDeletionQueue::Kind::acceleration_structure, accel_);
        }
        if (image_address_) device_.forget_image(image_address_);
        device_.context->remove_resource(id());
        if (in_arena_) device_.arena_release(arena_offset_, desc_.Width);
        if (image_) device_.context->defer_destroy(leonida::DeferredDeletionQueue::Kind::image, image_);
        if (buffer_) device_.context->defer_destroy(leonida::DeferredDeletionQueue::Kind::buffer, buffer_);
        if (memory_) device_.context->defer_destroy(leonida::DeferredDeletionQueue::Kind::device_memory, memory_);
    }

    leonida::ResourceId id() const { return resource_id(this); }
    VkImage image() const { return image_; }
    VkBuffer buffer() const { return buffer_; }
    const D3D12_RESOURCE_DESC& desc() const { return desc_; }
    VkFormat format() const { return format_; }                  // of the image
    bool has_view_format_list() const { return !view_formats_.empty(); }
    bool allows_view_format(VkFormat format) const {
        return view_formats_.empty() || format == format_ ||
               std::find(view_formats_.begin(), view_formats_.end(), format) != view_formats_.end();
    }
    bool r9g9b9e5_as_r32_uint() const { return r9g9b9e5_as_r32_uint_; }
    VkImageAspectFlags aspects() const { return aspects_; }
    bool rgb96() const { return rgb96_; }                        // a 96-bit texture in an RGBA32 image
    std::uint64_t image_address() const { return image_address_; }  // what a T# names the texture by
    VkMemoryPropertyFlags memory_flags() const { return memory_flags_; }  // of the memory type behind it
    const DeviceState& device_state() const { return device_; }

    // Default-heap buffers are ranges of the GPU address arena: console VA = offset, zeroed through the arena's host
    // mapping. Upload and readback buffers keep their own host memory; with buffer device addresses their console
    // VA is their device address minus heap_base, so a shader's heap_base + VA still lands on them (section 3.3).
    HRESULT create_buffer(D3D12_RESOURCE_STATES initial_state = D3D12_RESOURCE_STATE_COMMON) {
        auto& vk = device_.context->vk;
        // A buffer has at least one byte, and its size must survive alignment to 64 KiB without wrapping.
        if (!desc_.Width || desc_.Width > ~UINT64(0) - D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT) return E_INVALIDARG;
        const bool host = cpu_visible();
        leonida::ResourceBinding binding{};  // a buffer: no image
        binding.size = desc_.Width;
        if (!host && device_.arena.mapped) {
            if (!device_.arena_allocate(desc_.Width, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT, arena_offset_)) {
                log_once("GPU address arena full (raise LEONIDA_ARENA_MB); buffer size in KiB:",
                         int(desc_.Width / 1024));
                return E_OUTOFMEMORY;
            }
            in_arena_ = true;
            std::memset(device_.arena.mapped + arena_offset_, 0, std::size_t(desc_.Width));
            va_ = arena_offset_;
            binding.va = va_;
            binding.buffer = device_.arena.buffer;
            binding.buffer_offset = arena_offset_;
            // A default-heap AS resource is one of these ranges: a GENERIC acceleration structure covers it where
            // it sits. It is named by its real device address (heap base + offset), as instance descriptors and
            // BuildRaytracingAccelerationStructure hand that address to the driver, not a console VA.
            if (device_.info.ray_tracing &&
                initial_state == D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE &&
                vk.vkCreateAccelerationStructureKHR) {
                VkAccelerationStructureCreateInfoKHR asci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
                asci.buffer = device_.arena.buffer;
                asci.offset = arena_offset_;
                asci.size = desc_.Width;
                asci.type = VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR;
                if (vk.vkCreateAccelerationStructureKHR(device_.info.device, &asci, nullptr, &accel_) == VK_SUCCESS) {
                    gpu_va_ = device_.arena.address + va_;
                    device_.context->register_acceleration_structure(gpu_va_, accel_);
                }
            }
            device_.context->add_resource(id(), binding);
            return S_OK;
        }
        const bool addressable = device_.arena.buffer != VK_NULL_HANDLE;
        const bool need_bda = device_.info.buffer_device_address || device_.info.ray_tracing;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = desc_.Width;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        if (device_.info.conditional_rendering) info.usage |= VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
        if (addressable || need_bda) info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        if (device_.info.ray_tracing) {
            info.usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        }
        if (vk.vkCreateBuffer(device_.info.device, &info, nullptr, &buffer_) != VK_SUCCESS) return E_OUTOFMEMORY;
        VkMemoryRequirements requirements{};
        vk.vkGetBufferMemoryRequirements(device_.info.device, buffer_, &requirements);
        // Readback buffers prefer cached memory, which the CPU reads quickly. Upload buffers prefer VRAM the CPU
        // writes through Resizable BAR, where the GPU reads them quickly (D3D-96); without it, system memory.
        VkMemoryPropertyFlags host_preferred = 0;
        if (heap_.Type == D3D12_HEAP_TYPE_READBACK) host_preferred = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        if (heap_.Type == D3D12_HEAP_TYPE_UPLOAD && device_.upload_in_vram)
            host_preferred = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        memory_ = host ? device_.allocate(requirements,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                          host_preferred, &memory_flags_, addressable || need_bda)
                       : device_.allocate(requirements, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &memory_flags_,
                                          addressable || need_bda);
        if (!memory_ || vk.vkBindBufferMemory(device_.info.device, buffer_, memory_, 0) != VK_SUCCESS)
            return E_OUTOFMEMORY;
        if (addressable) {
            VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            address.buffer = buffer_;
            va_ = vk.vkGetBufferDeviceAddress(device_.info.device, &address) - device_.arena.address;
        }
        if (host) {
            if (vk.vkMapMemory(device_.info.device, memory_, 0, VK_WHOLE_SIZE, 0, &mapped_) != VK_SUCCESS)
                return E_OUTOFMEMORY;
            std::memset(mapped_, 0, std::size_t(desc_.Width));  // committed resources start zeroed
        } else {
            log_once("default-heap buffers are not zero-filled yet; dimension", int(desc_.Dimension));
        }
        if (need_bda && vk.vkGetBufferDeviceAddress) {
            VkBufferDeviceAddressInfo bda{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            bda.buffer = buffer_;
            buffer_va_ = vk.vkGetBufferDeviceAddress(device_.info.device, &bda);
            gpu_va_ = buffer_va_;
        }
        if (device_.info.ray_tracing && initial_state == D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE &&
            vk.vkCreateAccelerationStructureKHR && vk.vkGetAccelerationStructureDeviceAddressKHR) {
            VkAccelerationStructureCreateInfoKHR asci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            asci.buffer = buffer_;
            asci.offset = 0;
            asci.size = desc_.Width;
            asci.type = VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR;
            if (vk.vkCreateAccelerationStructureKHR(device_.info.device, &asci, nullptr, &accel_) == VK_SUCCESS) {
                VkAccelerationStructureDeviceAddressInfoKHR dai{
                    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
                dai.accelerationStructure = accel_;
                const std::uint64_t as_va = vk.vkGetAccelerationStructureDeviceAddressKHR(device_.info.device, &dai);
                if (as_va) gpu_va_ = as_va;
                device_.context->register_acceleration_structure(gpu_va_, accel_);
                if (buffer_va_ && buffer_va_ != gpu_va_)
                    device_.context->register_acceleration_structure(buffer_va_, accel_);
            }
        }
        binding.buffer = buffer_;
        binding.size = desc_.Width;
        binding.host_readable = (heap_.Type == D3D12_HEAP_TYPE_READBACK ||
                                 (heap_.Type == D3D12_HEAP_TYPE_CUSTOM &&
                                  heap_.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK));
        binding.va = va_;
        device_.context->add_resource(id(), binding);
        return S_OK;
    }

    // A placed buffer is a range of the heap's arena memory: it has no VkBuffer of its own and its console VA is
    // where it sits in the arena. The arena is host-visible, so an upload or readback place maps that same range.
    HRESULT create_placed_buffer() {
        if (!device_.arena.mapped) return E_OUTOFMEMORY;
        if (cpu_visible()) mapped_ = device_.arena.mapped + va_;
        std::memset(device_.arena.mapped + va_, 0, std::size_t(desc_.Width));  // a fresh place reads as zeroed
        leonida::ResourceBinding binding{};
        binding.size = desc_.Width;
        binding.buffer = device_.arena.buffer;
        binding.buffer_offset = va_;
        device_.context->add_resource(id(), binding);
        return S_OK;
    }

    HRESULT create_texture(const Format& format, const std::vector<VkFormat>& castable = {}) {
        auto& vk = device_.context->vk;
        const bool volume = desc_.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        const UINT depth = volume ? desc_.DepthOrArraySize : 1;
        const bool one_d = desc_.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D;
        if (desc_.MipLevels == 0) desc_.MipLevels = full_mips(desc_.Width, desc_.Height, depth);
        TextureCreateInfo create(desc_, format, device_.color_attachment_usable(format.format));
        VkImageCreateInfo& info = create.info;
        if (!castable.empty()) {
            // D3D12 always allows a view in the resource's own format, whether or not the castable list names
            // it. Vulkan requires every view format to be in the image's format list, so add it when missing.
            view_formats_ = castable;
            if (std::find(view_formats_.begin(), view_formats_.end(), format.format) == view_formats_.end())
                view_formats_.push_back(format.format);
            if (format.r9g9b9e5_as_r32_uint) {
                // The R9G9B9E5 emulation needs its E5B9G9R9 views whatever the list names (TextureCreateInfo).
                for (VkFormat pair : {VK_FORMAT_R32_UINT, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32})
                    if (std::find(view_formats_.begin(), view_formats_.end(), pair) == view_formats_.end())
                        view_formats_.push_back(pair);
            }
            info.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
            create.format_list.viewFormatCount = std::uint32_t(view_formats_.size());
            create.format_list.pViewFormats = view_formats_.data();
            create.format_list.pNext = info.pNext;
            info.pNext = &create.format_list;
        }
        const std::uint32_t samples = info.samples;
        // Textures the CPU reads and writes (WriteToSubresource/ReadFromSubresource) copy on the host when they can.
        // (A D24 texture stored as D32 goes through the staged copy: the host copy would not convert. Nor would it
        // repack a 96-bit texture's 12-byte texels.)
        host_transfer_ = cpu_visible() && samples == 1 && !format.d24_as_float && !format.rgb96 &&
                         device_.host_transfer_format(format.format);
        rgb96_ = format.rgb96;
        if (host_transfer_) info.usage |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
        HRESULT support = device_.check_image_support(info);
        if (support == E_INVALIDARG && host_transfer_) {
            host_transfer_ = false;  // staging copies do not require host-transfer usage
            info.usage &= ~VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
            support = device_.check_image_support(info);
        }
        if (FAILED(support)) return support;
        const VkResult created = vk.vkCreateImage(device_.info.device, &info, nullptr, &image_);
        if (created != VK_SUCCESS) {
            if (created == VK_ERROR_OUT_OF_HOST_MEMORY || created == VK_ERROR_OUT_OF_DEVICE_MEMORY)
                return E_OUTOFMEMORY;
            if (!view_formats_.empty()) {
                log_once("castable format list refused by Vulkan; count", int(view_formats_.size()));
                return E_INVALIDARG;
            }
            return E_OUTOFMEMORY;
        }
        VkMemoryRequirements requirements{};
        vk.vkGetImageMemoryRequirements(device_.info.device, image_, &requirements);
        memory_ = device_.allocate(requirements, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &memory_flags_);
        if (!memory_ || vk.vkBindImageMemory(device_.info.device, image_, memory_, 0) != VK_SUCCESS)
            return E_OUTOFMEMORY;

        leonida::ResourceBinding binding{};
        binding.image = image_;
        binding.rgba8_typeless = desc_.Format == DXGI_FORMAT_R8G8B8A8_TYPELESS;
        binding.castable_view_formats = !view_formats_.empty();  // a castable list made the image mutable
        binding.rgb96 = format.rgb96;
        binding.depth24_as_float = format.d24_as_float;
        binding.depth24 = format.d24;
        binding.format = format.format;
        binding.aspects = format.aspects;
        binding.width = std::uint32_t(desc_.Width);
        binding.height = one_d ? 1u : desc_.Height;
        binding.mip_levels = desc_.MipLevels;
        binding.array_layers = info.arrayLayers;
        binding.depth = depth;
        binding.volume = volume;
        binding.samples = samples;
        binding.color_attachment = (info.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0;
        binding.simultaneous_access = (desc_.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS) != 0;
        binding.not_zeroed = (heap_flags_ & D3D12_HEAP_FLAG_CREATE_NOT_ZEROED) != 0;
        // With the bindless set, a view's descriptor has one layout for as long as it lives (bind_image).
        binding.sampled_in_general = device_.bindless != nullptr;
        format_ = format.format;
        r9g9b9e5_as_r32_uint_ = format.r9g9b9e5_as_r32_uint;
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

    // D3D12 reports the placement alignment of a description that left it 0: 64 KiB, or 4 MiB for a multisampled
    // texture. One the description asked for is kept. Measured on the PC runtime and WARP (d3d12_resource_desc_test).
    D3D12_RESOURCE_DESC STDMETHODCALLTYPE GetDesc() override {
        D3D12_RESOURCE_DESC desc = desc_;
        if (!desc.Alignment) {
            desc.Alignment = desc.SampleDesc.Count > 1 ? D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT
                                                       : D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        }
        return desc;
    }

    // A reserved resource is a committed one underneath (docs/d3d12_vulkan.md section 4.2); it reports the layout it
    // was created with and has no heap of its own.
    void make_reserved(D3D12_TEXTURE_LAYOUT layout) {
        reserved_ = true;
        desc_.Layout = layout;
    }
    bool reserved() const { return reserved_; }

    // D3D12 answers only for committed and placed resources (E_INVALIDARG for reserved ones, measured on the PC
    // runtime).
    // Buffers only, as on D3D12; textures have no GPU VA (images are reached through descriptors). An acceleration
    // structure reports its real device address (instance descriptors and builds hand it to the driver); other
    // buffers report their console VA into the GPU address arena.
    D3D12_GPU_VIRTUAL_ADDRESS STDMETHODCALLTYPE GetGPUVirtualAddress() override { return accel_ ? gpu_va_ : va_; }

    //
    // The node masks read back as the one node of a single adapter (0 becomes 1), and the heap flags as the runtime
    // resolves them: the caller's, without the two creation-only bits (CREATE_NOT_RESIDENT, CREATE_NOT_ZEROED), plus
    // the DENY bits of the resource classes this resource is not (buffers 0xC0, textures without render-target or
    // depth use 0x44, the others 0x84). Measured on the PC runtime and WARP (d3d12_node_mask_test).
    HRESULT STDMETHODCALLTYPE GetHeapProperties(D3D12_HEAP_PROPERTIES* properties, D3D12_HEAP_FLAGS* flags) override {
        if (reserved_) return E_INVALIDARG;
        if (properties) {
            *properties = heap_;
            if (!properties->CreationNodeMask) properties->CreationNodeMask = 1;
            if (!properties->VisibleNodeMask) properties->VisibleNodeMask = 1;
        }
        if (flags) *flags = resolved_heap_flags();
        return S_OK;
    }

    D3D12_HEAP_FLAGS resolved_heap_flags() const {
        const UINT deny_buffers = D3D12_HEAP_FLAG_DENY_BUFFERS, deny_rt_ds = D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES,
                   deny_non_rt_ds = D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES;
        UINT deny = deny_buffers | deny_rt_ds;  // a texture without render-target or depth use
        if (desc_.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
            deny = deny_rt_ds | deny_non_rt_ds;
        else if (desc_.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))
            deny = deny_buffers | deny_non_rt_ds;
        const UINT creation_only = D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED;
        return D3D12_HEAP_FLAGS((UINT(heap_flags_) & ~creation_only) | deny);
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
        out.block = leonida::copy_block_format(binding_, range.aspectMask);
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
        transition.subresourceRange = {binding_.aspects, 0, binding_.mip_levels, 0,
                                       binding_.volume ? VK_REMAINING_ARRAY_LAYERS : binding_.array_layers};
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
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;  // storage: D24 depth conversion
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
    VkMemoryPropertyFlags memory_flags_ = 0;
    VkAccelerationStructureKHR accel_ = VK_NULL_HANDLE;
    void* mapped_ = nullptr;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    std::vector<VkFormat> view_formats_;  // creation-time castable list; empty means no restriction
    bool r9g9b9e5_as_r32_uint_ = false;
    std::uint64_t image_address_ = 0;  // what a T# names the texture by
    bool rgb96_ = false;
    VkImageAspectFlags aspects_ = 0;
    std::uint64_t buffer_va_ = 0;
    std::uint64_t gpu_va_ = 0;
    bool reserved_ = false;
    leonida::ResourceBinding binding_{};  // a texture's, as registered with the context
    bool host_transfer_ = false;          // created with VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT
    D3D12_GPU_VIRTUAL_ADDRESS va_ = 0;
    VkDeviceSize arena_offset_ = 0;
    bool in_arena_ = false;
};

// DXGI -> Vulkan for vertex attributes, with each element's size for D3D12_APPEND_ALIGNED_ELEMENT.
bool vertex_format(DXGI_FORMAT format, VkFormat& out, UINT& bytes) {
    struct Row {
        DXGI_FORMAT dxgi;
        VkFormat vk;
        UINT bytes;
    };
    static constexpr Row rows[] = {
        {DXGI_FORMAT_R32G32B32A32_FLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, 16},
        {DXGI_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_UINT, 16},
        {DXGI_FORMAT_R32G32B32A32_SINT, VK_FORMAT_R32G32B32A32_SINT, 16},
        {DXGI_FORMAT_R32G32B32_FLOAT, VK_FORMAT_R32G32B32_SFLOAT, 12},
        {DXGI_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_UINT, 12},
        {DXGI_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32_SINT, 12},
        {DXGI_FORMAT_R32G32_FLOAT, VK_FORMAT_R32G32_SFLOAT, 8},
        {DXGI_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_UINT, 8},
        {DXGI_FORMAT_R32G32_SINT, VK_FORMAT_R32G32_SINT, 8},
        {DXGI_FORMAT_R32_FLOAT, VK_FORMAT_R32_SFLOAT, 4},
        {DXGI_FORMAT_R32_UINT, VK_FORMAT_R32_UINT, 4},
        {DXGI_FORMAT_R32_SINT, VK_FORMAT_R32_SINT, 4},
        {DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 8},
        {DXGI_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_UNORM, 8},
        {DXGI_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_SNORM, 8},
        {DXGI_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_UINT, 8},
        {DXGI_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R16G16B16A16_SINT, 8},
        {DXGI_FORMAT_R16G16_FLOAT, VK_FORMAT_R16G16_SFLOAT, 4},
        {DXGI_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_UNORM, 4},
        {DXGI_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_SNORM, 4},
        {DXGI_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_UINT, 4},
        {DXGI_FORMAT_R16G16_SINT, VK_FORMAT_R16G16_SINT, 4},
        {DXGI_FORMAT_R16_FLOAT, VK_FORMAT_R16_SFLOAT, 2},
        {DXGI_FORMAT_R16_UNORM, VK_FORMAT_R16_UNORM, 2},
        {DXGI_FORMAT_R16_UINT, VK_FORMAT_R16_UINT, 2},
        {DXGI_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, 4},
        {DXGI_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_SNORM, 4},
        {DXGI_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_UINT, 4},
        {DXGI_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R8G8B8A8_SINT, 4},
        {DXGI_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM, 4},
        {DXGI_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_UNORM, 2},
        {DXGI_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_UINT, 2},
        {DXGI_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM, 1},
        {DXGI_FORMAT_R8_UINT, VK_FORMAT_R8_UINT, 1},
        {DXGI_FORMAT_R10G10B10A2_UNORM, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4},
        {DXGI_FORMAT_R10G10B10A2_UINT, VK_FORMAT_A2B10G10R10_UINT_PACK32, 4},
        {DXGI_FORMAT_R11G11B10_FLOAT, VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4},
    };
    for (const Row& row : rows)
        if (row.dxgi == format) {
            out = row.vk;
            bytes = row.bytes;
            return true;
        }
    return false;
}

VkBlendFactor blend_factor(D3D12_BLEND blend) {
    switch (blend) {
    case D3D12_BLEND_ZERO: return VK_BLEND_FACTOR_ZERO;
    case D3D12_BLEND_ONE: return VK_BLEND_FACTOR_ONE;
    case D3D12_BLEND_SRC_COLOR: return VK_BLEND_FACTOR_SRC_COLOR;
    case D3D12_BLEND_INV_SRC_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case D3D12_BLEND_SRC_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA;
    case D3D12_BLEND_INV_SRC_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case D3D12_BLEND_DEST_ALPHA: return VK_BLEND_FACTOR_DST_ALPHA;
    case D3D12_BLEND_INV_DEST_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case D3D12_BLEND_DEST_COLOR: return VK_BLEND_FACTOR_DST_COLOR;
    case D3D12_BLEND_INV_DEST_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case D3D12_BLEND_SRC_ALPHA_SAT: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case D3D12_BLEND_BLEND_FACTOR: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case D3D12_BLEND_INV_BLEND_FACTOR: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case D3D12_BLEND_SRC1_COLOR: return VK_BLEND_FACTOR_SRC1_COLOR;
    case D3D12_BLEND_INV_SRC1_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case D3D12_BLEND_SRC1_ALPHA: return VK_BLEND_FACTOR_SRC1_ALPHA;
    case D3D12_BLEND_INV_SRC1_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case D3D12_BLEND_ALPHA_FACTOR: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case D3D12_BLEND_INV_ALPHA_FACTOR: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default: return VK_BLEND_FACTOR_ONE;
    }
}

VkBlendOp blend_op(D3D12_BLEND_OP op) {
    switch (op) {
    case D3D12_BLEND_OP_SUBTRACT: return VK_BLEND_OP_SUBTRACT;
    case D3D12_BLEND_OP_REV_SUBTRACT: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case D3D12_BLEND_OP_MIN: return VK_BLEND_OP_MIN;
    case D3D12_BLEND_OP_MAX: return VK_BLEND_OP_MAX;
    default: return VK_BLEND_OP_ADD;
    }
}

// D3D12 and Vulkan list the logic ops in different orders, so each is named.
VkLogicOp logic_op(D3D12_LOGIC_OP op) {
    switch (op) {
    case D3D12_LOGIC_OP_CLEAR: return VK_LOGIC_OP_CLEAR;
    case D3D12_LOGIC_OP_SET: return VK_LOGIC_OP_SET;
    case D3D12_LOGIC_OP_COPY: return VK_LOGIC_OP_COPY;
    case D3D12_LOGIC_OP_COPY_INVERTED: return VK_LOGIC_OP_COPY_INVERTED;
    case D3D12_LOGIC_OP_NOOP: return VK_LOGIC_OP_NO_OP;
    case D3D12_LOGIC_OP_INVERT: return VK_LOGIC_OP_INVERT;
    case D3D12_LOGIC_OP_AND: return VK_LOGIC_OP_AND;
    case D3D12_LOGIC_OP_NAND: return VK_LOGIC_OP_NAND;
    case D3D12_LOGIC_OP_OR: return VK_LOGIC_OP_OR;
    case D3D12_LOGIC_OP_NOR: return VK_LOGIC_OP_NOR;
    case D3D12_LOGIC_OP_XOR: return VK_LOGIC_OP_XOR;
    case D3D12_LOGIC_OP_EQUIV: return VK_LOGIC_OP_EQUIVALENT;
    case D3D12_LOGIC_OP_AND_REVERSE: return VK_LOGIC_OP_AND_REVERSE;
    case D3D12_LOGIC_OP_AND_INVERTED: return VK_LOGIC_OP_AND_INVERTED;
    case D3D12_LOGIC_OP_OR_REVERSE: return VK_LOGIC_OP_OR_REVERSE;
    case D3D12_LOGIC_OP_OR_INVERTED: return VK_LOGIC_OP_OR_INVERTED;
    default: return VK_LOGIC_OP_COPY;
    }
}

// D3D12_COMPARISON_FUNC (NEVER = 1 .. ALWAYS = 8) and D3D12_STENCIL_OP (KEEP = 1 .. DECR = 8) list their values
// in Vulkan's order, starting at 1 instead of 0.
VkCompareOp compare_op(D3D12_COMPARISON_FUNC func) {
    const int value = int(func);
    return value >= 1 && value <= 8 ? VkCompareOp(value - 1) : VK_COMPARE_OP_ALWAYS;
}
VkStencilOp stencil_op(D3D12_STENCIL_OP op) {
    const int value = int(op);
    return value >= 1 && value <= 8 ? VkStencilOp(value - 1) : VK_STENCIL_OP_KEEP;
}

VkStencilOpState stencil_face(const D3D12_DEPTH_STENCILOP_DESC& face, UINT8 read_mask, UINT8 write_mask) {
    VkStencilOpState state{};
    state.failOp = stencil_op(face.StencilFailOp);
    state.passOp = stencil_op(face.StencilPassOp);
    state.depthFailOp = stencil_op(face.StencilDepthFailOp);
    state.compareOp = compare_op(face.StencilFunc);
    state.compareMask = read_mask;
    state.writeMask = write_mask;
    return state;
}

bool equal_semantic(const std::string& a, const char* b) {
    if (!b || a.size() != std::strlen(b)) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) return false;
    return true;
}
// A heap (section 3.3): a range of the GPU address arena, as large as the game asked for, so a placed buffer can
// name any 64 KiB-aligned offset in it. Textures placed in it still get their own VkDeviceMemory; the offset is only
// their key, which keeps optimal-tiling images out of the arena's memory type.
class Heap final
    : public Child<Heap, ID3D12HeapStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable, ID3D12Heap> {
public:
    // {5d20c7a4-8e13-4b6d-9f27-3a1c94e0b7d8}
    static constexpr GUID kTag{0x5d20c7a4, 0x8e13, 0x4b6d, {0x9f, 0x27, 0x3a, 0x1c, 0x94, 0xe0, 0xb7, 0xd8}};

    Heap(DeviceState& device, const D3D12_HEAP_DESC& desc) : device_(device), desc_(desc) {
        reserved_ = device_.arena.mapped &&
                    device_.arena_allocate(desc_.SizeInBytes, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT, offset_);
    }

    ~Heap() override {
        if (reserved_) device_.arena_release(offset_, desc_.SizeInBytes);
    }

    D3D12_HEAP_DESC STDMETHODCALLTYPE GetDesc() override { return desc_; }

    bool reserved() const { return reserved_; }
    VkDeviceSize offset() const { return offset_; }
    UINT64 bytes() const { return desc_.SizeInBytes; }
    const D3D12_HEAP_PROPERTIES& properties() const { return desc_.Properties; }
    D3D12_HEAP_FLAGS flags() const { return desc_.Flags; }

private:
    DeviceState& device_;
    D3D12_HEAP_DESC desc_;
    VkDeviceSize offset_ = 0;
    bool reserved_ = false;
};

// Among index entries for one program (containers with the same FSSM program; 789 groups in the 26,503-shader
// corpus), the one to build: the first that `fits` and declares no set-0 bindings, which pipelines do not bind yet;
// else the first that fits; else the first without bindings; else the first. Null when there is none.
template<typename Fits>
const leonida::ShaderEntry* pick_entry(const std::vector<const leonida::ShaderEntry*>& candidates, Fits&& fits) {
    const leonida::ShaderEntry* fitting = nullptr;
    const leonida::ShaderEntry* unbound = nullptr;
    for (const leonida::ShaderEntry* e : candidates) {
        const bool fit = fits(*e);
        if (fit && !e->binding_count) return e;
        if (fit && !fitting) fitting = e;
        if (!e->binding_count && !unbound) unbound = e;
    }
    if (fitting) return fitting;
    if (unbound) return unbound;
    return candidates.empty() ? nullptr : candidates.front();
}

// Whether the input layout has an element for every ordinary (non-system-value) input of the vertex shader.
bool layout_feeds(const leonida::ShaderEntry& vs, const D3D12_INPUT_LAYOUT_DESC& layout) {
    for (const auto& input : vs.input_signature) {
        if (input.system_value) continue;
        bool fed = false;
        for (UINT i = 0; i < layout.NumElements && !fed; ++i)
            fed = layout.pInputElementDescs[i].SemanticIndex == input.semantic_index &&
                  equal_semantic(input.semantic, layout.pInputElementDescs[i].SemanticName);
        if (!fed) return false;
    }
    return true;
}

class RootSignature final
    : public Child<RootSignature, ID3D12RootSignatureStubs, ID3D12Object, ID3D12DeviceChild, ID3D12RootSignature> {
public:
    // {3e9b4a61-2d7c-4f18-b5a0-c71e8d92f036}
    static constexpr GUID kTag{0x3e9b4a61, 0x2d7c, 0x4f18, {0xb5, 0xa0, 0xc7, 0x1e, 0x8d, 0x92, 0xf0, 0x36}};
    explicit RootSignature(RootArgumentLayout layout) : layout_(std::move(layout)) {}
    const RootArgumentLayout& layout() const { return layout_; }

private:
    RootArgumentLayout layout_;
};

// A pipeline state. Compute only for now: the shader is found by the SHA-256 of its FSSM program in the
// translator's index, and the FSSM blob is copied into the arena as the shader's code base.
class PipelineState final
    : public Child<PipelineState, ID3D12PipelineStateStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable,
                   ID3D12PipelineState> {
public:
    // {8d2f6c14-b0e7-4a93-9c58-1f4a7e3b6d20}
    static constexpr GUID kTag{0x8d2f6c14, 0xb0e7, 0x4a93, {0x9c, 0x58, 0x1f, 0x4a, 0x7e, 0x3b, 0x6d, 0x20}};
    explicit PipelineState(DeviceState& device) : device_(device) {}

    ~PipelineState() override {
        auto& vk = device_.context->vk;
        const VkDevice dev = device_.info.device;
        if (registered_) device_.context->remove_pipeline(id());
        if (pipeline_.pipeline) vk.vkDestroyPipeline(dev, pipeline_.pipeline, nullptr);
        if (pipeline_.layout) vk.vkDestroyPipelineLayout(dev, pipeline_.layout, nullptr);
        if (module_) vk.vkDestroyShaderModule(dev, module_, nullptr);
        if (graphics_.pipeline) vk.vkDestroyPipeline(dev, graphics_.pipeline, nullptr);
        for (const auto& [unbound, built] : variants_) vk.vkDestroyPipeline(dev, built, nullptr);
        if (graphics_.layout) vk.vkDestroyPipelineLayout(dev, graphics_.layout, nullptr);
        for (VkShaderModule module : modules_) vk.vkDestroyShaderModule(dev, module, nullptr);
        if (code_bytes_) device_.arena_release(code_base_, code_bytes_);
    }

    // Pipeline layouts declare the whole 144-byte block as one push-constant range. A device whose
    // maxPushConstantsSize is smaller (128 bytes is the Vulkan minimum, which some Intel and mobile drivers
    // report) needs the uniform-buffer tail #224 flags with push_constants_need_fallback, which is not built yet,
    // so pipelines are refused there with the limit named instead of breaking VUID-VkPushConstantRange-size-00298.
    bool push_constants_fit() const {
        const std::uint32_t limit = device_.info.max_push_constants_size;
        if (limit == 0 || limit >= sizeof(leonida::ShaderUserData)) return true;  // 0: not queried
        log_once("pipelines need 144 bytes of push constants (uniform-buffer fallback not built yet); "
                 "maxPushConstantsSize",
                 int(limit));
        return false;
    }

    std::uintptr_t id() const { return reinterpret_cast<std::uintptr_t>(static_cast<const ID3D12PipelineState*>(this)); }
    bool compute() const { return registered_ && !is_graphics_; }
    bool graphics() const { return registered_ && is_graphics_; }

    // A VS (+ PS) pipeline for dynamic rendering. Vertex input comes from the input layout matched to the VS's
    // input signature by semantic name and index: location = the element's register, the translator's rule
    // (docs/design.md, "Vertex fetch"). Hull, domain and geometry stages, more than one sample, and two stages
    // that both read their code base are refused with a log. Viewport, scissor, topology, vertex strides, stencil
    // reference and blend factor are dynamic, set from the list's state at each draw.
    HRESULT create_graphics(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc) {
        auto& vk = device_.context->vk;
        const VkDevice dev = device_.info.device;
        if (!desc.VS.pShaderBytecode || !desc.VS.BytecodeLength) return E_INVALIDARG;
        if (!device_.arena.mapped) {
            log_once("pipelines need the GPU address arena (bufferDeviceAddress); stage", 1);
            return E_NOTIMPL;
        }
        if (!push_constants_fit()) return E_NOTIMPL;
        if (!vk.graphics_complete()) {
            log_once("graphics pipelines need Vulkan 1.3 dynamic rendering and extended dynamic state; stage", 1);
            return E_NOTIMPL;
        }
        if (desc.NumRenderTargets > D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT) return E_INVALIDARG;
        if (desc.HS.BytecodeLength || desc.DS.BytecodeLength || desc.GS.BytecodeLength || desc.StreamOutput.NumEntries) {
            log_once("graphics pipeline stage not supported yet (hull/domain/geometry/stream output):", 0);
            return E_NOTIMPL;
        }
        if (desc.SampleDesc.Count > 1) {
            log_once("multisampled graphics pipelines not supported yet; samples", int(desc.SampleDesc.Count));
            return E_NOTIMPL;
        }
        const leonida::ShaderLibrary* library = device_.library();
        if (!library) return E_FAIL;

        struct Stage {
            const char* name;
            VkShaderStageFlagBits bit;
            const D3D12_SHADER_BYTECODE& code;
            const leonida::ShaderEntry* entry = nullptr;
            std::vector<std::uint8_t> fssm;
        };
        Stage stages[2] = {{"vs", VK_SHADER_STAGE_VERTEX_BIT, desc.VS}, {"ps", VK_SHADER_STAGE_FRAGMENT_BIT, desc.PS}};
        std::vector<VkPipelineShaderStageCreateInfo> stage_infos;
        const Stage* code_stage = nullptr;
        for (Stage& stage : stages) {
            if (!stage.code.pShaderBytecode || !stage.code.BytecodeLength) continue;  // no PS: depth only
            stage.fssm = leonida::fssm_program(
                {static_cast<const std::uint8_t*>(stage.code.pShaderBytecode), stage.code.BytecodeLength});
            if (stage.fssm.empty()) {
                log_once("shader bytecode holds no FSSM program; bytes", int(stage.code.BytecodeLength));
                return E_INVALIDARG;
            }
            const std::string hash = leonida::sha256_hex(stage.fssm);
            // Containers with the same program can differ in their input signature or bindings: a VS must have its
            // inputs fed by this input layout (else the layout check below reports it).
            const bool vertex = stage.bit == VK_SHADER_STAGE_VERTEX_BIT;
            stage.entry = pick_entry(library->candidates(stage.name, hash), [&](const leonida::ShaderEntry& e) {
                return !vertex || layout_feeds(e, desc.InputLayout);
            });
            if (!stage.entry) {
                std::fprintf(stderr, "[rt] d3d12: no translated %s for FSSM sha256 %s\n", stage.name, hash.c_str());
                return E_INVALIDARG;
            }
            if (stage.entry->binding_count) {
                std::fprintf(stderr, "[rt] d3d12: %s uses set-0 bindings; pipelines do not bind set 0 yet\n",
                             stage.entry->name.c_str());
                return E_NOTIMPL;
            }
            if (stage.entry->uses_code_base) {
                if (code_stage) {
                    log_once("two stages read their code base; one push-constant slot holds one (question 14)", 0);
                    return E_NOTIMPL;
                }
                code_stage = &stage;
            }
            const std::vector<std::uint32_t> words = library->spirv(*stage.entry);
            if (words.empty()) return E_FAIL;
            VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            module_info.codeSize = words.size() * 4;
            module_info.pCode = words.data();
            VkShaderModule module = VK_NULL_HANDLE;
            if (vk.vkCreateShaderModule(dev, &module_info, nullptr, &module) != VK_SUCCESS) return E_FAIL;
            modules_.push_back(module);
            VkPipelineShaderStageCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            info.stage = stage.bit;
            info.module = module;
            info.pName = "main";
            stage_infos.push_back(info);
        }

        // Input layout -> vertex input, by semantic.
        std::vector<VkVertexInputAttributeDescription> attributes;
        std::vector<VkVertexInputBindingDescription> bindings;
        UINT slot_end[D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT] = {};
        for (UINT i = 0; i < desc.InputLayout.NumElements; ++i) {
            const D3D12_INPUT_ELEMENT_DESC& element = desc.InputLayout.pInputElementDescs[i];
            if (element.InputSlot >= D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT) return E_INVALIDARG;
            VkFormat format;
            UINT bytes;
            if (!vertex_format(element.Format, format, bytes)) {
                log_once("vertex attribute format not supported yet:", element.Format);
                return E_INVALIDARG;
            }
            const UINT offset = element.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT
                                    ? (slot_end[element.InputSlot] + 3) / 4 * 4
                                    : element.AlignedByteOffset;
            slot_end[element.InputSlot] = offset + bytes;
            const leonida::SignatureElement* input = nullptr;
            for (const auto& e : stages[0].entry->input_signature)
                if (e.system_value == 0 && e.semantic_index == element.SemanticIndex &&
                    equal_semantic(e.semantic, element.SemanticName))
                    input = &e;
            if (!input) continue;  // the shader does not read this element
            attributes.push_back({input->register_index, element.InputSlot, format, offset});
            const bool instance = element.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA;
            // Step rate 0: every instance reads element 0, which a stride of 0 gives (set at each draw).
            if (instance && element.InstanceDataStepRate == 0) graphics_.zero_step_slots |= 1u << element.InputSlot;
            if (instance && element.InstanceDataStepRate > 1)
                log_once("instance step rates above 1 are not modelled; rate", int(element.InstanceDataStepRate));
            if (!(graphics_.vertex_slots & (1u << element.InputSlot))) {
                graphics_.vertex_slots |= 1u << element.InputSlot;
                bindings.push_back({element.InputSlot, 0,  // stride: dynamic, from IASetVertexBuffers
                                    instance ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
            }
        }
        for (const auto& e : stages[0].entry->input_signature) {
            if (e.system_value) continue;
            const bool fed = std::any_of(attributes.begin(), attributes.end(),
                                         [&](const auto& a) { return a.location == e.register_index; });
            if (!fed) {  // D3D12 rejects an input layout that leaves a shader input unfed
                std::fprintf(stderr, "[rt] d3d12: the input layout does not feed %s%u (register %u)\n",
                             e.semantic.c_str(), e.semantic_index, e.register_index);
                return E_INVALIDARG;
            }
        }
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        switch (desc.PrimitiveTopologyType) {
        case D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT: assembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
        case D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE: assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE: assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
        default:
            log_once("primitive topology type not supported:", desc.PrimitiveTopologyType);
            return E_INVALIDARG;
        }
        if (desc.IBStripCutValue != D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED)
            log_once("index-buffer strip cuts (primitive restart) are not modelled; value", desc.IBStripCutValue);

        const D3D12_RASTERIZER_DESC& r = desc.RasterizerState;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = r.FillMode == D3D12_FILL_MODE_WIREFRAME ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
        raster.cullMode = r.CullMode == D3D12_CULL_MODE_FRONT  ? VK_CULL_MODE_FRONT_BIT
                          : r.CullMode == D3D12_CULL_MODE_BACK ? VK_CULL_MODE_BACK_BIT
                                                               : VK_CULL_MODE_NONE;
        // With the negative-height viewport, framebuffer winding matches D3D12's screen-space winding.
        raster.frontFace = r.FrontCounterClockwise ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
        raster.depthBiasEnable = r.DepthBias || r.SlopeScaledDepthBias != 0.0f;
        raster.depthBiasConstantFactor = float(r.DepthBias);
        raster.depthBiasClamp = r.DepthBiasClamp;
        raster.depthBiasSlopeFactor = r.SlopeScaledDepthBias;
        raster.lineWidth = 1.0f;
        // No depth clipping: fragments outside the viewport's depth range are kept and their depth clamped to it, as
        // D3D12 clamps output depth to MinDepth..MaxDepth anyway (needs depthClamp, checked below).
        raster.depthClampEnable = r.DepthClipEnable ? VK_FALSE : VK_TRUE;

        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        const VkSampleMask sample_mask = desc.SampleMask;
        multisample.pSampleMask = &sample_mask;
        multisample.alphaToCoverageEnable = desc.BlendState.AlphaToCoverageEnable;

        const D3D12_DEPTH_STENCIL_DESC& ds = desc.DepthStencilState;
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = ds.DepthEnable;
        depth.depthWriteEnable = ds.DepthEnable && ds.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ALL;
        depth.depthCompareOp = compare_op(ds.DepthFunc);
        depth.stencilTestEnable = ds.StencilEnable;
        depth.front = stencil_face(ds.FrontFace, ds.StencilReadMask, ds.StencilWriteMask);
        depth.back = stencil_face(ds.BackFace, ds.StencilReadMask, ds.StencilWriteMask);

        std::vector<VkPipelineColorBlendAttachmentState> blends(desc.NumRenderTargets);
        for (UINT i = 0; i < desc.NumRenderTargets; ++i) {
            const D3D12_RENDER_TARGET_BLEND_DESC& b =
                desc.BlendState.RenderTarget[desc.BlendState.IndependentBlendEnable ? i : 0];
            blends[i].blendEnable = b.BlendEnable;
            blends[i].srcColorBlendFactor = blend_factor(b.SrcBlend);
            blends[i].dstColorBlendFactor = blend_factor(b.DestBlend);
            blends[i].colorBlendOp = blend_op(b.BlendOp);
            blends[i].srcAlphaBlendFactor = blend_factor(b.SrcBlendAlpha);
            blends[i].dstAlphaBlendFactor = blend_factor(b.DestBlendAlpha);
            blends[i].alphaBlendOp = blend_op(b.BlendOpAlpha);
            blends[i].colorWriteMask = b.RenderTargetWriteMask & 0xf;  // R, G, B, A in both APIs
        }
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = std::uint32_t(blends.size());
        blend.pAttachments = blends.data();
        // A logic op replaces blending for every target in Vulkan (one per pipeline), and D3D12 takes it from
        // RenderTarget[0]. Vulkan applies it to integer and normalized formats and skips float ones.
        const D3D12_RENDER_TARGET_BLEND_DESC& first = desc.BlendState.RenderTarget[0];
        if (desc.NumRenderTargets && first.LogicOpEnable) {
            if (unsigned(first.LogicOp) > D3D12_LOGIC_OP_OR_INVERTED) return E_INVALIDARG;
            blend.logicOpEnable = VK_TRUE;
            blend.logicOp = logic_op(first.LogicOp);
        }

        // State that needs a Vulkan feature the device did not enable (VulkanDeviceInfo::features) is refused, rather
        // than handed to the driver against the spec.
        const VkPhysicalDeviceFeatures& features = device_.info.features;
        const auto refuse = [&](const char* what, int value) {
            log_once(what, value);
            return E_NOTIMPL;
        };
        if (raster.polygonMode != VK_POLYGON_MODE_FILL && !features.fillModeNonSolid)
            return refuse("wireframe fill needs fillModeNonSolid, which the device lacks; fill mode", r.FillMode);
        if (raster.depthBiasEnable && raster.depthBiasClamp != 0.0f && !features.depthBiasClamp)
            return refuse("a depth-bias clamp needs depthBiasClamp, which the device lacks; clamp x1000",
                          int(raster.depthBiasClamp * 1000));
        for (std::size_t i = 1; i < blends.size(); ++i)
            if (std::memcmp(&blends[i], &blends[0], sizeof(blends[0])) != 0 && !features.independentBlend)
                return refuse("per-render-target blend state needs independentBlend, which the device lacks; target",
                              int(i));
        const auto src1 = [](VkBlendFactor f) {
            return f == VK_BLEND_FACTOR_SRC1_COLOR || f == VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR ||
                   f == VK_BLEND_FACTOR_SRC1_ALPHA || f == VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        };
        for (const auto& b : blends)
            if (b.blendEnable && !features.dualSrcBlend &&
                (src1(b.srcColorBlendFactor) || src1(b.dstColorBlendFactor) || src1(b.srcAlphaBlendFactor) ||
                 src1(b.dstAlphaBlendFactor)))
                return refuse("SRC1 blend factors need dualSrcBlend, which the device lacks; factor",
                              int(b.srcColorBlendFactor));
        if (blend.logicOpEnable && !features.logicOp)
            return refuse("logic ops need logicOp, which the device lacks; op", int(first.LogicOp));
        if (raster.depthClampEnable && !features.depthClamp)
            return refuse("DepthClipEnable = FALSE needs depthClamp, which the device lacks; flag", 0);

        graphics_.color_formats.assign(desc.NumRenderTargets, VK_FORMAT_UNDEFINED);
        for (UINT i = 0; i < desc.NumRenderTargets; ++i) {
            if (desc.RTVFormats[i] == DXGI_FORMAT_UNKNOWN) continue;
            graphics_.color_formats[i] = vulkan_format(desc.RTVFormats[i], false).format;
            if (graphics_.color_formats[i] == VK_FORMAT_UNDEFINED) {
                log_once("render-target format not supported yet:", desc.RTVFormats[i]);
                return E_INVALIDARG;
            }
        }
        if (desc.DSVFormat != DXGI_FORMAT_UNKNOWN) {
            Format f = vulkan_format(desc.DSVFormat, true);
            if (f.format == VK_FORMAT_D24_UNORM_S8_UINT && !device_.depth_usable(f.format))
                f.format = VK_FORMAT_D32_SFLOAT_S8_UINT;  // as CreateCommittedResource does
            graphics_.depth_format = f.format;
            graphics_.depth_aspects = f.aspects;
        }
        // Every stage this path accepts reads the user SGPRs, so the range covers the whole stage table, not only the
        // stages present. A stage added to the table (geometry, mesh) is covered without touching this.
        for (const Stage& stage : stages) graphics_.push_stages |= stage.bit;
        const VkPushConstantRange push{graphics_.push_stages, 0, sizeof(leonida::ShaderUserData)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        if (vk.vkCreatePipelineLayout(dev, &layout_info, nullptr, &graphics_.layout) != VK_SUCCESS) return E_FAIL;

        // Kept for variants: a draw that leaves a slot unbound needs this pipeline with that format undefined.
        recipe_.stages = std::move(stage_infos);
        recipe_.attributes = std::move(attributes);
        recipe_.bindings = std::move(bindings);
        recipe_.assembly = assembly;
        recipe_.raster = raster;
        recipe_.multisample = multisample;
        recipe_.sample_mask = sample_mask;
        recipe_.depth = depth;
        recipe_.blends = std::move(blends);
        recipe_.blend = blend;
        graphics_.pipeline = build(0);
        if (!graphics_.pipeline) return E_FAIL;
        graphics_.variant = [this](std::uint32_t unbound) { return variant(unbound); };

        if (code_stage) {
            VkDeviceSize offset = 0;
            if (!device_.arena_allocate(code_stage->fssm.size(), 256, offset)) return E_OUTOFMEMORY;
            std::memcpy(device_.arena.mapped + offset, code_stage->fssm.data(), code_stage->fssm.size());
            code_bytes_ = code_stage->fssm.size();
            code_base_ = offset;
            graphics_.code_base = offset;
        }
        device_.context->add_graphics_pipeline(id(), graphics_);
        registered_ = true;
        is_graphics_ = true;
        return S_OK;
    }

    HRESULT create_compute(const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc) {
        auto& vk = device_.context->vk;
        const VkDevice dev = device_.info.device;
        if (!desc.CS.pShaderBytecode || !desc.CS.BytecodeLength) return E_INVALIDARG;
        if (!device_.arena.mapped) {
            log_once("pipelines need the GPU address arena (bufferDeviceAddress); stage", 0);
            return E_NOTIMPL;
        }
        if (!push_constants_fit()) return E_NOTIMPL;
        const std::vector<std::uint8_t> fssm = leonida::fssm_program(
            {static_cast<const std::uint8_t*>(desc.CS.pShaderBytecode), desc.CS.BytecodeLength});
        if (fssm.empty()) {
            log_once("compute shader bytecode holds no FSSM program; bytes", int(desc.CS.BytecodeLength));
            return E_INVALIDARG;
        }
        const leonida::ShaderLibrary* library = device_.library();
        if (!library) return E_FAIL;
        const std::string hash = leonida::sha256_hex(fssm);
        // A binding-free entry wins over one with bindings for the same program (pick_entry).
        const leonida::ShaderEntry* entry =
            pick_entry(library->candidates("cs", hash), [](const leonida::ShaderEntry&) { return true; });
        if (!entry) {
            std::fprintf(stderr, "[rt] d3d12: no translated compute shader for FSSM sha256 %s\n", hash.c_str());
            return E_INVALIDARG;
        }
        if (entry->binding_count) {
            // Set 0 (bindless images, samplers, the descriptor hash table): the device keeps it, pipelines do not
            // declare or bind it yet (M2).
            std::fprintf(stderr, "[rt] d3d12: %s uses set-0 bindings; pipelines do not bind set 0 yet\n",
                         entry->name.c_str());
            return E_NOTIMPL;
        }
        if (entry->wave64) log_once("wave64 compute shader runs 32-wide (subgroup size control not requested yet)", 0);
        const std::vector<std::uint32_t> words = library->spirv(*entry);
        if (words.empty()) {
            std::fprintf(stderr, "[rt] d3d12: %s.spv is missing or not SPIR-V\n", entry->name.c_str());
            return E_FAIL;
        }

        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = words.size() * 4;
        module_info.pCode = words.data();
        if (vk.vkCreateShaderModule(dev, &module_info, nullptr, &module_) != VK_SUCCESS) return E_FAIL;
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(leonida::ShaderUserData)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        if (vk.vkCreatePipelineLayout(dev, &layout_info, nullptr, &pipeline_.layout) != VK_SUCCESS) return E_FAIL;
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module_;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = pipeline_.layout;
        if (vk.vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_.pipeline) !=
            VK_SUCCESS)
            return E_FAIL;

        // The code base: shaders read constant tables embedded in their own program through s_getpc_b64.
        VkDeviceSize offset = 0;
        if (!device_.arena_allocate(fssm.size(), 256, offset)) return E_OUTOFMEMORY;
        std::memcpy(device_.arena.mapped + offset, fssm.data(), fssm.size());
        code_bytes_ = fssm.size();
        code_base_ = offset;
        pipeline_.code_base = offset;
        device_.context->add_pipeline(id(), pipeline_);
        registered_ = true;
        return S_OK;
    }

private:
    // What a graphics pipeline is made from, so variants can be built at draw time.
    struct Recipe {
        std::vector<VkPipelineShaderStageCreateInfo> stages;
        std::vector<VkVertexInputAttributeDescription> attributes;
        std::vector<VkVertexInputBindingDescription> bindings;
        VkPipelineInputAssemblyStateCreateInfo assembly{};
        VkPipelineRasterizationStateCreateInfo raster{};
        VkPipelineMultisampleStateCreateInfo multisample{};
        VkSampleMask sample_mask = ~0u;
        VkPipelineDepthStencilStateCreateInfo depth{};
        std::vector<VkPipelineColorBlendAttachmentState> blends;
        VkPipelineColorBlendStateCreateInfo blend{};
    };

    // The graphics pipeline with the formats of the `unbound` slots (leonida::GraphicsPipeline::variant) undefined.
    VkPipeline build(std::uint32_t unbound) const {
        // The stride state is last: a pipeline without vertex bindings (a full-screen pass that builds its vertices
        // from SV_VertexID) leaves it out, since no vkCmdBindVertexBuffers2 would ever set it
        // (VUID-vkCmdDraw-pStrides-04913).
        static constexpr VkDynamicState dynamic_states[] = {
            VK_DYNAMIC_STATE_VIEWPORT,          VK_DYNAMIC_STATE_SCISSOR,          VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
            VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE};
        const Recipe& r = recipe_;
        std::vector<VkFormat> colors = graphics_.color_formats;
        for (std::size_t slot = 0; slot < colors.size(); ++slot)
            if (unbound & (1u << slot)) colors[slot] = VK_FORMAT_UNDEFINED;
        const bool depth_bound = !(unbound & leonida::GraphicsPipeline::unbound_depth);
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount = std::uint32_t(colors.size());
        rendering.pColorAttachmentFormats = colors.data();
        if (depth_bound && (graphics_.depth_aspects & VK_IMAGE_ASPECT_DEPTH_BIT))
            rendering.depthAttachmentFormat = graphics_.depth_format;
        if (depth_bound && (graphics_.depth_aspects & VK_IMAGE_ASPECT_STENCIL_BIT))
            rendering.stencilAttachmentFormat = graphics_.depth_format;

        VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertex_input.vertexBindingDescriptionCount = std::uint32_t(r.bindings.size());
        vertex_input.pVertexBindingDescriptions = r.bindings.data();
        vertex_input.vertexAttributeDescriptionCount = std::uint32_t(r.attributes.size());
        vertex_input.pVertexAttributeDescriptions = r.attributes.data();
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineMultisampleStateCreateInfo multisample = r.multisample;
        multisample.pSampleMask = &r.sample_mask;
        VkPipelineColorBlendStateCreateInfo blend = r.blend;
        blend.attachmentCount = std::uint32_t(r.blends.size());
        blend.pAttachments = r.blends.data();
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = std::uint32_t(std::size(dynamic_states)) - (r.bindings.empty() ? 1 : 0);
        dynamic.pDynamicStates = dynamic_states;

        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
        info.stageCount = std::uint32_t(r.stages.size());
        info.pStages = r.stages.data();
        info.pVertexInputState = &vertex_input;
        info.pInputAssemblyState = &r.assembly;
        info.pViewportState = &viewport;
        info.pRasterizationState = &r.raster;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &r.depth;
        info.pColorBlendState = &blend;
        info.pDynamicState = &dynamic;
        info.layout = graphics_.layout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        if (device_.context->vk.vkCreateGraphicsPipelines(device_.info.device, VK_NULL_HANDLE, 1, &info, nullptr,
                                                          &pipeline) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        return pipeline;
    }

    // Built on first use and kept until the PSO is released (replay may call this from any queue).
    VkPipeline variant(std::uint32_t unbound) {
        std::lock_guard lock(variants_mutex_);
        if (const auto it = variants_.find(unbound); it != variants_.end()) return it->second;
        const VkPipeline pipeline = build(unbound);
        if (!pipeline) {
            log_once("a pipeline variant for unbound attachment slots could not be built; slots", int(unbound));
            return VK_NULL_HANDLE;
        }
        variants_.emplace(unbound, pipeline);
        return pipeline;
    }

    DeviceState& device_;
    leonida::ComputePipeline pipeline_;
    leonida::GraphicsPipeline graphics_;
    Recipe recipe_;
    std::mutex variants_mutex_;
    std::map<std::uint32_t, VkPipeline> variants_;
    VkShaderModule module_ = VK_NULL_HANDLE;
    std::vector<VkShaderModule> modules_;
    VkDeviceSize code_base_ = 0;
    VkDeviceSize code_bytes_ = 0;
    bool registered_ = false;
    bool is_graphics_ = false;
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
    // Plane 1 of a depth-stencil texture is its stencil, viewed as X24_TYPELESS_G8_UINT (D24S8) or
    // X32_TYPELESS_G8X24_UINT (D32S8): eight unsigned bits in G, as D3D12 returns them. The T# is 8_UINT, the
    // format of the console's separate stencil surface, and bind_image views the stencil aspect for it.
    const bool stencil = view.plane == 1;
    if (view.plane > 1 || (stencil && !(resource.aspects() & VK_IMAGE_ASPECT_STENCIL_BIT))) {
        log_once("no such plane in this texture; plane", int(view.plane));
        return false;
    }
    // Each stencil view format belongs to its own DXGI family only, as in D3D12: the family is the texture's DXGI
    // format, not the Vulkan format behind it (a D24 texture may be D32_SFLOAT_S8_UINT underneath).
    const bool stencil_format_fits =
        (view.format == DXGI_FORMAT_X24_TYPELESS_G8_UINT &&
         (desc.Format == DXGI_FORMAT_R24G8_TYPELESS || desc.Format == DXGI_FORMAT_D24_UNORM_S8_UINT)) ||
        (view.format == DXGI_FORMAT_X32_TYPELESS_G8X24_UINT &&
         (desc.Format == DXGI_FORMAT_R32G8X24_TYPELESS || desc.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT));
    if (stencil && !stencil_format_fits) {
        log_once("a stencil-plane view needs its texture family's stencil view format; format", int(view.format));
        return false;
    }
    const bool volume = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const bool msaa = g::multisampled(view.type);
    const bool line = view.type == g::ImageType::image_1d;
    const auto dimension = line ? D3D12_RESOURCE_DIMENSION_TEXTURE1D :
                           view.type == g::ImageType::image_3d ? D3D12_RESOURCE_DIMENSION_TEXTURE3D :
                                                               D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    if (desc.Dimension != dimension || (line && desc.DepthOrArraySize != 1) || msaa != (desc.SampleDesc.Count > 1)) {
        log_once("view type incompatible with texture dimension or sample count; view type", int(view.type));
        return false;
    }
    if (volume && view.min_lod != 0) {
        log_once("volume SRV requires a zero ResourceMinLODClamp; format", int(desc.Format));
        return false;
    }
    if (msaa) {
        // The MS bindings expose typed colour SRVs. An MS view has one mip and can name an array slice range.
        const Format format = texture_storage_format(desc.Format, false);
        if (resource.aspects() != VK_IMAGE_ASPECT_COLOR_BIT || format.typeless ||
            (view.format != DXGI_FORMAT_UNKNOWN && view.format != desc.Format)) {
            log_once("multisample shader view requires a matching typed colour format:", int(desc.Format));
            return false;
        }
        // Four 3-bit selectors followed by D3D12's always-set bit.
        if ((view.mapping >> 12) != 1) {
            log_once("invalid multisample shader view component mapping:", int(view.mapping));
            return false;
        }
    }
    const VkFormat resource_format =
        resource.r9g9b9e5_as_r32_uint() ? VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 : resource.format();
    VkFormat named = resource_format;
    if (volume) {
        const bool compatible = vulkan_format(desc.Format, false).typeless
            ? desc.Format == DXGI_FORMAT_R8G8B8A8_TYPELESS &&
              (view.format == DXGI_FORMAT_R8G8B8A8_UNORM || view.format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB)
            : view.format == DXGI_FORMAT_UNKNOWN || view.format == desc.Format;
        if (!compatible) {
            log_once("unsupported volume SRV format for this resource:", int(view.format));
            return false;
        }
    }
    // The depth plane of a depth-stencil texture, as R24_UNORM_X8_TYPELESS (D24S8) or R32_FLOAT_X8X24_TYPELESS
    // (D32S8): the texture's own depth format, whose T# code is the console's depth format (8_24_UNORM or
    // X24_8_32_FLOAT) with the depth in R; bind_image views the depth aspect for it. As in D3D12, each view format
    // belongs to its own family only: the other family's depth view leaves the slot null, whatever format backs the
    // texture (a D24 texture may be D32_SFLOAT_S8_UINT underneath).
    const DXGI_FORMAT family = desc.Format;
    const bool d24_family = family == DXGI_FORMAT_R24G8_TYPELESS || family == DXGI_FORMAT_D24_UNORM_S8_UINT;
    const bool d32s8_family = family == DXGI_FORMAT_R32G8X24_TYPELESS || family == DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    const bool depth_plane = ((view.format == DXGI_FORMAT_R24_UNORM_X8_TYPELESS && d24_family) ||
                              (view.format == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS && d32s8_family)) &&
                             (resource.aspects() & VK_IMAGE_ASPECT_DEPTH_BIT);
    if (view.format != DXGI_FORMAT_UNKNOWN && !stencil && !depth_plane) {
        if ((view.format == DXGI_FORMAT_A8_UNORM) != (desc.Format == DXGI_FORMAT_A8_UNORM)) {
            log_once("incompatible typed A8 view format:", int(view.format));
            return false;
        }
        if ((view.format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP) != (desc.Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP)) {
            log_once("incompatible R9G9B9E5 view format:", int(view.format));
            return false;
        }
        named = texture_storage_format(view.format, false).format;
    }
    // A 96-bit texture takes views of its own family only (and a four-channel texture none of it), and only the
    // TYPELESS one is cast between FLOAT, UINT and SINT. Its T# names the three-channel format: the swizzle below
    // then reads alpha as one, and bind_image views the RGBA32 image accordingly.
    const bool rgb96 = resource.rgb96();
    if (!stencil && !depth_plane) {
        const bool view_rgb96 = view.format != DXGI_FORMAT_UNKNOWN ? vulkan_format(view.format, false).rgb96 : rgb96;
        if (view_rgb96 != rgb96 ||
            (rgb96 && desc.Format != DXGI_FORMAT_R32G32B32_TYPELESS && named != resource.format())) {
            log_once("view format outside the texture's 96-bit family:", int(view.format));
            return false;
        }
        if (rgb96) named = rgb96_named_format(named);
    }
    if (named == VK_FORMAT_UNDEFINED || !resource.allows_view_format(named)) {
        log_once("view format not in the resource's castable list:", int(view.format));
        return false;
    }
    const std::uint32_t code = stencil ? g::stencil_format : g::format_from_vulkan(named);
    VkFormat view_format = VK_FORMAT_UNDEFINED;
    if (stencil) {
        view_format = resource.format();
    } else if (code && rgb96) {
        view_format = named;
    } else if (code) {
        view_format = g::vulkan_view_format(resource_format, code, resource.has_view_format_list());
    }
    if (view_format == VK_FORMAT_UNDEFINED) {
        log_once("view format not supported for this texture:", int(view.format));
        return false;
    }
    const UINT layers = volume ? 1 : desc.DepthOrArraySize;
    if (view.first_mip >= desc.MipLevels || view.first_layer >= layers) {
        log_once("view starts past the texture's mips or layers; first mip", int(view.first_mip));
        return false;
    }
    if (msaa && view.layer_count != UINT_MAX && view.layer_count > UINT(desc.DepthOrArraySize) - view.first_layer) {
        log_once("multisample shader view ends past the texture array; layers", int(view.layer_count));
        return false;
    }
    out = {};
    out.address = resource.image_address();
    out.format = code;
    out.type = view.type;
    out.width = std::uint32_t(desc.Width);
    out.height = desc.Height;
    out.depth = volume ? desc.DepthOrArraySize : 1;
    out.samples = desc.SampleDesc.Count;
    out.first_mip = view.first_mip;
    out.mip_count = std::min<UINT>(view.mip_count, desc.MipLevels - view.first_mip);
    out.first_layer = view.first_layer;
    out.layer_count = std::min<UINT>(view.layer_count, layers - view.first_layer);
    if (view.type == g::ImageType::cube) out.layer_count -= out.layer_count % 6;  // whole cubes
    out.resource_mips = desc.MipLevels;
    out.min_lod = view.min_lod;
    if (!out.mip_count || !out.layer_count) {
        log_once("view of no mips or no layers; type", int(view.type));
        return false;
    }
    // D3D12's component mapping picks, for each of R, G, B, A, a channel of the format or a constant; the T#
    // says the same about the hardware's channels, which the format places.
    // A stencil view's format has its eight bits in G and no R, B or A of its own (0, 0, 1 when read).
    g::Swizzle channels = g::format_swizzle(view_format);
    if (stencil) channels = {g::select_zero, g::select_x, g::select_zero, g::select_one};
    else if (desc.Format == DXGI_FORMAT_A8_UNORM) channels = {g::select_zero, g::select_zero, g::select_zero, g::select_x};
    for (UINT i = 0; i < 4; ++i) {
        const UINT from = D3D12_DECODE_SHADER_4_COMPONENT_MAPPING(i, view.mapping);
        if (msaa && from > D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1) {
            log_once("invalid multisample shader view component selector:", from);
            return false;
        }
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
    out.compare_never = reduction == D3D12_FILTER_REDUCTION_TYPE_COMPARISON &&
                        desc.ComparisonFunc == D3D12_COMPARISON_FUNC_NEVER;
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

// A query heap: one VkQueryPool, registered with the context so that lists can resolve it by ID
// (docs/d3d12_vulkan.md section 4.6). Occlusion heaps hold both OCCLUSION and BINARY_OCCLUSION queries; timestamp
// and copy-queue timestamp heaps are timestamp pools; pipeline-statistics heaps count all eleven Vulkan
// statistics, whose result order is the field order of D3D12_QUERY_DATA_PIPELINE_STATISTICS.
class QueryHeap final
    : public Child<QueryHeap, ID3D12QueryHeapStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable,
                   ID3D12QueryHeap> {
public:
    // {6e2b9a41-d83c-4f17-b5e9-0a7c3d1f8b62}
    static constexpr GUID kTag{0x6e2b9a41, 0xd83c, 0x4f17, {0xb5, 0xe9, 0x0a, 0x7c, 0x3d, 0x1f, 0x8b, 0x62}};

    explicit QueryHeap(DeviceState& device) : device_(device) {}

    ~QueryHeap() override {
        if (!pool_) return;
        device_.context->remove_query_pool(id());
        device_.context->vk.vkDestroyQueryPool(device_.info.device, pool_, nullptr);
    }

    leonida::QueryHeapId id() const { return query_heap_id(this); }

    HRESULT create(const D3D12_QUERY_HEAP_DESC& desc) {
        namespace h = leonida::d3d12_query_heap;
        auto& vk = device_.context->vk;
        if (!vk.vkCreateQueryPool || !vk.vkDestroyQueryPool) return E_NOTIMPL;
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryCount = desc.Count;
        switch (desc.Type) {
        case h::occlusion: info.queryType = VK_QUERY_TYPE_OCCLUSION; break;
        case h::timestamp:
        case h::copy_queue_timestamp:
            if (!(device_.context->timestamp_period() > 0)) {
                log_once("timestamp query heap: the queue writes no timestamps; type", desc.Type);
                return E_NOTIMPL;
            }
            info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            break;
        case h::pipeline_statistics:
            if (!device_.info.pipeline_statistics_query) {
                log_once("pipeline-statistics query heap: pipelineStatisticsQuery not enabled; type", desc.Type);
                return E_NOTIMPL;
            }
            info.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
            info.pipelineStatistics = kPipelineStatistics;
            break;
        default:
            // Stream output, video decode and mesh-shader statistics (PIPELINE_STATISTICS1) have no path yet.
            log_once("query heap type not supported:", desc.Type);
            return E_NOTIMPL;
        }
        if (vk.vkCreateQueryPool(device_.info.device, &info, nullptr, &pool_) != VK_SUCCESS) {
            pool_ = VK_NULL_HANDLE;
            return E_OUTOFMEMORY;
        }
        leonida::QueryPool pool{};
        pool.pool = pool_;
        pool.heap_type = desc.Type;
        pool.count = desc.Count;
        pool.precise = device_.info.occlusion_query_precise;
        pool.ended.assign(desc.Count, false);
        device_.context->add_query_pool(id(), pool);
        return S_OK;
    }

private:
    // VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT through COMPUTE_SHADER_INVOCATIONS_BIT (0x1-0x400):
    // IAVertices, IAPrimitives, VSInvocations, GSInvocations, GSPrimitives, CInvocations, CPrimitives,
    // PSInvocations, HSInvocations (patches), DSInvocations, CSInvocations.
    static constexpr VkQueryPipelineStatisticFlags kPipelineStatistics = 0x7ff;
    static_assert(sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS) == 11 * sizeof(UINT64));

    DeviceState& device_;
    VkQueryPool pool_ = VK_NULL_HANDLE;
};

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
        queue_busy_.resize(info.command_queues.size(), false);
        const auto get_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            info.get_instance_proc(info.instance, "vkGetDeviceProcAddr"));
        state_.format_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceFormatProperties"));
        state_.image_format_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceImageFormatProperties>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceImageFormatProperties"));
        state_.image_format_properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceImageFormatProperties2>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceImageFormatProperties2"));
        if (!state_.image_format_properties2) {
            state_.image_format_properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceImageFormatProperties2>(
                info.get_instance_proc(info.instance, "vkGetPhysicalDeviceImageFormatProperties2KHR"));
        }
        const auto memory_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceMemoryProperties"));
        const auto device_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceProperties"));
        const auto queue_families = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
        if (!get_device || !state_.format_properties || !state_.image_format_properties || !memory_properties ||
            !device_properties || !queue_families)
            return;
        state_.r9g9b9e5_uav_emulated = state_.probe_r9g9b9e5_uav_emulation();
        memory_properties(info.physical, &state_.memory);
        {
            char text[4] = {};
            size_t length = 0;
            const bool sysmem = getenv_s(&length, text, "LEONIDA_UPLOAD_SYSMEM") == 0 && length > 1 && text[0] != '0';
            state_.upload_in_vram = leonida::resizable_bar(state_.memory) && !sysmem;
            std::fprintf(stderr, "[rt] d3d12: upload heaps in %s\n",
                         state_.upload_in_vram ? "VRAM (DEVICE_LOCAL | HOST_VISIBLE, Resizable BAR)"
                         : sysmem             ? "system memory (LEONIDA_UPLOAD_SYSMEM)"
                                              : "system memory (no Resizable BAR)");
        }
        for (const char* name : {"vkGetDeviceImageMemoryRequirements", "vkGetDeviceImageMemoryRequirementsKHR"})
            if (!state_.image_requirements)
                state_.image_requirements =
                    reinterpret_cast<PFN_vkGetDeviceImageMemoryRequirements>(get_device(info.device, name));

        if (info.physical && info.get_instance_proc) {
            const auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
                info.get_instance_proc(info.instance, "vkGetPhysicalDeviceProperties"));
            if (get_props) {
                VkPhysicalDeviceProperties props{};
                get_props(info.physical, &props);
                if (state_.info.max_push_constants_size == 0) {
                    state_.info.max_push_constants_size = props.limits.maxPushConstantsSize;
                    state_.info.push_constants_need_fallback = (state_.info.max_push_constants_size < 144);
                }
                // The counts D3D12 offers with no outputs are the target-independent-rasterization ones: 1, 4, 8
                // and 16 (ForcedSampleCount is 0, 1, 4, 8 or 16; DirectX-Specs VulkanOn12.md "Sample-frequency MSAA
                // with no render targets bound": a driver reports at least 0b11101). 2x is not one of them, though
                // framebufferNoAttachmentsSampleCounts has it on every GPU here; reporting it would offer a count the
                // PC runtime refuses (its hosts report 0x1d).
                constexpr VkSampleCountFlags kD3D12SampleCounts =
                    VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT | VK_SAMPLE_COUNT_8_BIT | VK_SAMPLE_COUNT_16_BIT;
                state_.no_output_sample_counts =
                    static_cast<UINT>(props.limits.framebufferNoAttachmentsSampleCounts & kD3D12SampleCounts);
            }
            if (const auto get_props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                    info.get_instance_proc(info.instance, "vkGetPhysicalDeviceProperties2"))) {
                VkPhysicalDeviceDepthStencilResolveProperties resolve{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES};
                VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &resolve};
                get_props2(info.physical, &props);
                state_.depth_resolve_modes = resolve.supportedDepthResolveModes;
                state_.stencil_resolve_modes = resolve.supportedStencilResolveModes;
            }
        }

        state_.context = std::make_unique<leonida::VulkanContext>(get_device, info.device,
                                                                  info.separate_depth_stencil_layouts,
                                                                  info.conditional_rendering, info.ray_tracing,
                                                                  info.xdxr_acceleration_structures &&
                                                                      info.buffer_device_address);
        if (info.buffer_device_address && state_.context->loaded() && !create_arena())
            std::fprintf(stderr, "[rt] d3d12: the GPU address arena could not be created\n");
        if (info.conditional_rendering) {
            const auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
                info.get_instance_proc(info.instance, "vkGetPhysicalDeviceProperties"));
            const auto families = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
                info.get_instance_proc(info.instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
            if (properties && families) {
                VkPhysicalDeviceProperties limits{};
                properties(info.physical, &limits);
                std::uint32_t count = 0;
                families(info.physical, &count, nullptr);
                std::vector<VkQueueFamilyProperties> queues(count);
                families(info.physical, &count, queues.data());
                if (info.queue_family < count && (queues[info.queue_family].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                    state_.context->configure_predication(get_device, state_.memory, limits.limits.maxStorageBufferRange, info.queue_family);
                    if (info.storage_buffer_8bit) state_.context->configure_predicated_copy(get_device, limits.limits);
                }
            }
        }
        if (info.host_image_copy) load_host_image_copy(get_device);
        // Depth <-> colour copies are staged through scratch buffers (vulkan_scratch.h), on every GPU.
        state_.context->enable_copy_scratch(get_device, [this](const VkMemoryRequirements& requirements) {
            return state_.allocate(requirements, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        });
        // The copies of a D24 texture stored as D32 convert on the GPU (vulkan_depth24.h), and so do the copies of
        // a 96-bit texture, which every GPU stores as RGBA32 (D3D-62): the converter is there whether or not the
        // GPU has a native D24_UNORM_S8_UINT. It costs nothing until a copy needs it (its pipeline is made on the
        // first job, and a queue takes the tracker lock for it only when a list has such a copy).
        {
            std::uint64_t max_storage_range = 1u << 27;  // the least Vulkan allows, if the limit cannot be read
            if (info.physical && info.get_instance_proc) {
                if (const auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
                        info.get_instance_proc(info.instance, "vkGetPhysicalDeviceProperties"))) {
                    VkPhysicalDeviceProperties props{};
                    get_props(info.physical, &props);
                    max_storage_range = props.limits.maxStorageBufferRange;
                }
            }
            state_.context->enable_depth24(get_device, [this](const VkMemoryRequirements& requirements) {
                return state_.allocate(requirements, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            }, max_storage_range);
        }
        if (info.bindless && state_.context->loaded() && !create_bindless(get_device))
            std::fprintf(stderr, "[rt] d3d12: the bindless set 0 could not be created\n");
        query_luid_and_clock(get_device);
        // Timestamps: what the one VkQueue every D3D12 queue submits to can write.
        VkPhysicalDeviceProperties properties{};
        device_properties(info.physical, &properties);
        state_.max_sampler_lod_bias = properties.limits.maxSamplerLodBias;
        std::uint32_t family_count = 0;
        queue_families(info.physical, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        queue_families(info.physical, &family_count, families.data());
        if (info.queue_family < family_count) {
            const auto& family = families[info.queue_family];
            if (family.timestampValidBits)
                state_.context->set_timestamp_period(properties.limits.timestampPeriod);
            if (family.queueFlags & VK_QUEUE_COMPUTE_BIT)
                state_.context->set_occlusion_resolve_memory(state_.memory);
        }

        if (info.debug_utils) {
            leonida::DebugLabelFunctions labels;
            const auto load = [&](auto& f, const char* name) {
                f = reinterpret_cast<std::remove_reference_t<decltype(f)>>(info.get_instance_proc(info.instance, name));
            };
            load(labels.cmd_insert, "vkCmdInsertDebugUtilsLabelEXT");
            load(labels.cmd_begin, "vkCmdBeginDebugUtilsLabelEXT");
            load(labels.cmd_end, "vkCmdEndDebugUtilsLabelEXT");
            load(labels.queue_insert, "vkQueueInsertDebugUtilsLabelEXT");
            load(labels.queue_begin, "vkQueueBeginDebugUtilsLabelEXT");
            load(labels.queue_end, "vkQueueEndDebugUtilsLabelEXT");
            state_.context->enable_debug_labels(labels);
            if (!state_.context->debug_labels())
                std::fprintf(stderr, "[rt] d3d12: VK_EXT_debug_utils entry points missing; debug markers are off\n");
        }
    }

    // Set 0 of every translated shader that declares bindings: images, samplers and the descriptor hash table
    // (docs section 3.4). Each ExecuteCommandLists brings it up to date before its commands are written.
    bool create_bindless(PFN_vkGetDeviceProcAddr get_device) {
        leonida::BindlessFeatures features{};
        features.update_after_bind = true;  // VulkanDeviceInfo::bindless promises it
        features.partially_bound = state_.info.bindless_partially_bound;
        const auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            state_.info.get_instance_proc(state_.info.instance, "vkGetPhysicalDeviceProperties2"));
        if (get_properties) {
            VkPhysicalDeviceVulkan12Properties indexed{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
            VkPhysicalDeviceVulkan11Properties v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &indexed};
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &v11};
            get_properties(state_.info.physical, &properties);
            features.ms_samples = {VK_SAMPLE_COUNT_4_BIT, state_.ms_fallback_samples(VK_FORMAT_R32_UINT),
                                   state_.ms_fallback_samples(VK_FORMAT_R32_SINT)};
            if (!leonida::bindless::size_bindings(features, properties.properties.limits, v11, indexed))
                std::fprintf(stderr, "[rt] d3d12: the device's descriptor limits are below the bindless set's default "
                                     "sizes; it holds less\n");
        }
        namespace b = leonida::bindless;
        std::fprintf(stderr, "[rt] d3d12: bindless set: %u 2D images, %u samplers, %u 2D-array, %u 3D, %u 1D, "
                             "%u storage indices in 2D\n",
                     features.counts[b::sampled_binding(leonida::ImageDimension::dim_2d)],
                     features.counts[b::sampler_binding],
                     features.counts[b::sampled_binding(leonida::ImageDimension::dim_2d_array)],
                     features.counts[b::sampled_binding(leonida::ImageDimension::dim_3d)],
                     features.counts[b::sampled_binding(leonida::ImageDimension::dim_1d)],
                     features.counts[b::storage_binding(leonida::ImageDimension::dim_2d, 0)]);
        state_.bindless = std::make_unique<leonida::BindlessSet>(get_device, state_.info.device, state_.memory, features);
        if (!state_.bindless->ok() || !state_.context->vk.vkCreateSampler) {
            state_.bindless.reset();
            return false;
        }
        state_.context->set_bindless_release(
            [](void* user, std::uint32_t dim, std::uint32_t idx) {
                auto* st = static_cast<DeviceState*>(user);
                std::lock_guard lock(st->descriptor_mutex);
                if (st->bindless) st->bindless->remove_image(static_cast<leonida::ImageDimension>(dim), idx);
            },
            &state_);
        state_.context->set_bindless_prepare([this](VkCommandBuffer command_buffer) {
            std::lock_guard lock(state_.descriptor_mutex);
            if (!state_.bindless->fallbacks_ready()) state_.bindless->record_fallback_init(command_buffer);
            if (state_.bindless->flush_needs_idle()) {
                // Entries a running shader probes are about to change, or binding 2 is about to name a new buffer:
                // wait until nothing that uses the set still runs, then free the generations that were retired.
                {
                    std::lock_guard submit(state_.context->submit_mutex());
                    state_.context->vk.vkDeviceWaitIdle(state_.info.device);
                }
                state_.bindless->release_retired();
            }
            return state_.bindless->flush(command_buffer);
        });
        state_.context->set_bindless_commit([this](bool submitted) {
            std::lock_guard lock(state_.descriptor_mutex);
            if (state_.bindless) state_.bindless->table_upload_result(submitted);
        });
        return true;
    }

    // LEONIDA_ARENA_MB sets its size (default 256, at most 4095): it holds every default-heap buffer, shader-
    // visible descriptor heap and shader program. M0-M3 keep it host-visible (section 3.3, memory type policy).
    bool create_arena() {
        auto& vk = state_.context->vk;
        if (!vk.vkGetBufferDeviceAddress) return false;
        VkDeviceSize megabytes = 256;
        char text[16] = {};
        size_t length = 0;
        if (getenv_s(&length, text, "LEONIDA_ARENA_MB") == 0 && length > 1)
            megabytes = std::clamp<VkDeviceSize>(std::strtoull(text, nullptr, 10), 16, 4095);
        auto& arena = state_.arena;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = megabytes << 20;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        // Default-heap buffers live here, and any of them can be a SetPredication buffer.
        if (state_.info.conditional_rendering) info.usage |= VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
        // With ray tracing, AS result, scratch and build-input buffers are default-heap ranges of the arena.
        if (state_.info.ray_tracing)
            info.usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        if (vk.vkCreateBuffer(state_.info.device, &info, nullptr, &arena.buffer) != VK_SUCCESS) {
            arena.buffer = VK_NULL_HANDLE;
            return false;
        }
        VkMemoryRequirements requirements{};
        vk.vkGetBufferMemoryRequirements(state_.info.device, arena.buffer, &requirements);
        arena.memory = state_.allocate(requirements,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, nullptr, true);
        void* mapped = nullptr;
        if (!arena.memory || vk.vkBindBufferMemory(state_.info.device, arena.buffer, arena.memory, 0) != VK_SUCCESS ||
            vk.vkMapMemory(state_.info.device, arena.memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
            // Without an arena every buffer keeps its own memory; a leftover handle would make them addressable
            // with heap base 0, so their VAs would be raw device addresses above 4 GiB.
            vk.vkDestroyBuffer(state_.info.device, arena.buffer, nullptr);
            if (arena.memory) vk.vkFreeMemory(state_.info.device, arena.memory, nullptr);
            arena.buffer = VK_NULL_HANDLE;
            arena.memory = VK_NULL_HANDLE;
            return false;
        }
        arena.mapped = static_cast<std::uint8_t*>(mapped);
        std::memset(arena.mapped, 0, kArenaReserved);  // the zero page unbound vertex and index buffers read
        VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        address.buffer = arena.buffer;
        arena.address = vk.vkGetBufferDeviceAddress(state_.info.device, &address);
        arena.size = info.size;
        arena.free[kArenaReserved] = arena.size - kArenaReserved;
        state_.context->set_heap_base(arena.address);
        state_.context->set_zero_page(arena.buffer, 0, kArenaReserved);
        return true;
    }

    ~Device() override {
        if (state_.context) {
            {
                std::lock_guard lock(state_.context->submit_mutex());
                state_.context->vk.vkDeviceWaitIdle(state_.info.device);
            }
            state_.context->drain_deleted_resources(UINT64_MAX);
            state_.context->set_bindless_release(nullptr, nullptr);
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
        if (state_.context && state_.arena.buffer) {
            auto& vk = state_.context->vk;
            vk.vkDestroyBuffer(state_.info.device, state_.arena.buffer, nullptr);
            if (state_.arena.memory) vk.vkFreeMemory(state_.info.device, state_.arena.memory, nullptr);
        }
        state_.context.reset();
        if (owned_.device) owned_.destroy_device(owned_.device, nullptr);
        if (owned_.instance) owned_.destroy_instance(owned_.instance, nullptr);
    }

    bool ok() const { return state_.context && state_.context->loaded(); }
    const VulkanDeviceInfo& info() const { return state_.info; }
    bool upload_in_vram() const { return state_.upload_in_vram; }
    leonida::VulkanContext& context() { return *state_.context; }
    const leonida::BindlessSet* bindless() const { return state_.bindless.get(); }
    const DeviceState& state() const { return state_; }
    IUnknown* holder() { return identity(); }
    bool bind_sampler(const leonida::gfx10::SamplerWords& words) { return state_.bind_sampler(words); }
    void own(const OwnedRuntime& owned) { owned_ = owned; }
    void set_outer(IUnknown* outer) { outer_ = outer; }

    UINT STDMETHODCALLTYPE GetNodeCount() override { return 1; }

    // The Vulkan physical device's LUID, which is the DXGI adapter's on Windows. Zero (logged) when the driver has
    // none to report (deviceLUIDValid false).
    LUID STDMETHODCALLTYPE GetAdapterLuid() override { return state_.luid; }

    // The CUSTOM equivalent of a standard heap type, from the D3D12 documentation's table for a NUMA adapter
    // (D3D12_FEATURE_ARCHITECTURE answers UMA = FALSE): DEFAULT is GPU-only memory in L1, UPLOAD write-combined
    // and READBACK write-back system memory in L0, GPU_UPLOAD write-combined L1. CUSTOM and unknown types have
    // no equivalent and get an all-zero answer (D3D12 reports them through the debug layer).
    D3D12_HEAP_PROPERTIES STDMETHODCALLTYPE GetCustomHeapProperties(UINT node_mask, D3D12_HEAP_TYPE type) override {
        D3D12_HEAP_PROPERTIES properties{};
        if (node_mask > 1) {
            log_once("GetCustomHeapProperties node mask", int(node_mask));
            return properties;
        }
        switch (type) {
        case D3D12_HEAP_TYPE_DEFAULT:
            properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE;
            properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L1;
            break;
        case D3D12_HEAP_TYPE_UPLOAD:
            properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
            properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
            break;
        case D3D12_HEAP_TYPE_READBACK:
            properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
            properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
            break;
        case D3D12_HEAP_TYPE_GPU_UPLOAD:
            properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
            properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L1;
            break;
        default:
            log_once("GetCustomHeapProperties heap type without a custom equivalent:", int(type));
            return properties;
        }
        properties.Type = D3D12_HEAP_TYPE_CUSTOM;
        properties.CreationNodeMask = 1;
        properties.VisibleNodeMask = 1;
        return properties;
    }

    HRESULT STDMETHODCALLTYPE SetEventOnMultipleFenceCompletion(ID3D12Fence* const* fences, const UINT64* values,
                                                                UINT count, D3D12_MULTIPLE_FENCE_WAIT_FLAGS flags,
                                                                HANDLE event) override {
        return set_event_on_multiple_fence_completion(*state_.context, fences, values, count, flags, event);
    }
    HRESULT STDMETHODCALLTYPE GetDeviceRemovedReason() override {
        if (removed_.load(std::memory_order_acquire)) return DXGI_ERROR_DEVICE_REMOVED;
        return (state_.context && state_.context->is_device_lost()) ? DXGI_ERROR_DEVICE_REMOVED : S_OK;
    }

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

    // STUB
    HRESULT STDMETHODCALLTYPE EnumerateMetaCommands(UINT* count, D3D12_META_COMMAND_DESC*) override {
        if (!count) return E_INVALIDARG;
        *count = 0;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE EnumerateMetaCommandParameters(REFGUID, D3D12_META_COMMAND_PARAMETER_STAGE, UINT*,
                                                             UINT*, D3D12_META_COMMAND_PARAMETER_DESC*) override {
        return E_INVALIDARG;
    }

    HRESULT STDMETHODCALLTYPE CreateMetaCommand(REFGUID, UINT, const void*, SIZE_T, REFIID, void** out) override {
        if (out) *out = nullptr;
        return E_INVALIDARG;
    }

    // STUB
    D3D12_DRIVER_MATCHING_IDENTIFIER_STATUS STDMETHODCALLTYPE CheckDriverMatchingIdentifier(
        D3D12_SERIALIZED_DATA_TYPE type, const D3D12_SERIALIZED_DATA_DRIVER_MATCHING_IDENTIFIER*) override {
        return type == D3D12_SERIALIZED_DATA_RAYTRACING_ACCELERATION_STRUCTURE
                   ? D3D12_DRIVER_MATCHING_IDENTIFIER_UNRECOGNIZED
                   : D3D12_DRIVER_MATCHING_IDENTIFIER_UNSUPPORTED_TYPE;
    }

    // STUB
    void STDMETHODCALLTYPE RemoveDevice() override { removed_.store(true, std::memory_order_release); }

    HRESULT STDMETHODCALLTYPE CreateCommandQueue(const D3D12_COMMAND_QUEUE_DESC* desc, REFIID riid,
                                                 void** out) override {
        constexpr auto method = "ID3D12Device::CreateCommandQueue";
        if (!desc) return d3d12_trace::result(method, E_INVALIDARG);
        if (!out) return d3d12_trace::result(method, E_POINTER);
        *out = nullptr;
        if (desc->Type != D3D12_COMMAND_LIST_TYPE_DIRECT && desc->Type != D3D12_COMMAND_LIST_TYPE_COMPUTE &&
            desc->Type != D3D12_COMMAND_LIST_TYPE_COPY) return d3d12_trace::result(method, E_INVALIDARG);
        // Caller-owned devices that only provide the legacy queue cannot acquire more handles from their VkDevice.
        if (state_.info.command_queues.empty())
            return d3d12_trace::result(method, create_command_queue(*state_.context, state_.info.queue,
                                                                     state_.info.queue_family, *desc, riid, out,
                                                                     identity(), state_.clock));
        // Vulkan only guarantees ordering within a VkQueue. Assign a distinct handle to each live D3D12 queue;
        // otherwise a Wait submitted before another queue's Signal can deadlock the shared handle.
        std::size_t slot = queue_busy_.size();
        {
            std::lock_guard lock(queue_mutex_);
            for (std::size_t i = 0; i < queue_busy_.size(); ++i) {
                if (!queue_busy_[i]) {
                    queue_busy_[i] = true;
                    slot = i;
                    break;
                }
            }
        }
        if (slot == queue_busy_.size()) {
            std::fprintf(stderr, "[rt] d3d12: no independent Vulkan queue available\n");
            return d3d12_trace::result(method, E_OUTOFMEMORY);
        }
        // Child holds a device reference until after the lease has been released.
        std::shared_ptr<void> lease(nullptr, [this, slot](void*) {
            std::lock_guard lock(queue_mutex_);
            queue_busy_[slot] = false;
        });
        const VkQueue queue = state_.info.command_queues[slot];
        return d3d12_trace::result(method, create_command_queue(*state_.context, queue,
                                                                 state_.info.queue_family, *desc, riid, out,
                                                                 identity(), state_.clock, std::move(lease)));
    }

    HRESULT STDMETHODCALLTYPE CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void** out) override {
        constexpr auto method = "ID3D12Device::CreateCommandAllocator";
        return d3d12_trace::result(method, create_command_allocator(type, riid, out, identity()));
    }

    HRESULT STDMETHODCALLTYPE CreateCommandList(UINT, D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* allocator,
                                                ID3D12PipelineState* state, REFIID riid, void** out) override {
        constexpr auto method = "ID3D12Device::CreateCommandList";
        HRESULT hr = create_command_list(type, allocator, riid, out, identity());
        // The initial pipeline state is set as SetPipelineState would set it.
        if (SUCCEEDED(hr) && state) {
            ID3D12GraphicsCommandList* list = nullptr;
            if (SUCCEEDED(static_cast<IUnknown*>(*out)->QueryInterface(IID_PPV_ARGS(&list)))) {
                list->SetPipelineState(state);
                list->Release();
            }
        }
        return d3d12_trace::result(method, hr);
    }

    // Device4: a list created closed, without an allocator.
    HRESULT STDMETHODCALLTYPE CreateCommandList1(UINT, D3D12_COMMAND_LIST_TYPE type, D3D12_COMMAND_LIST_FLAGS,
                                                 REFIID riid, void** out) override {
        constexpr auto method = "ID3D12Device4::CreateCommandList1";
        if (!out) return d3d12_trace::result(method, E_POINTER);
        *out = nullptr;
        ID3D12CommandAllocator* allocator = nullptr;
        HRESULT hr = create_command_allocator(type, IID_PPV_ARGS(&allocator), identity());
        if (FAILED(hr)) return d3d12_trace::result(method, hr);
        ID3D12GraphicsCommandList* list = nullptr;
        hr = create_command_list(type, allocator, IID_PPV_ARGS(&list), identity());
        allocator->Release();
        if (FAILED(hr)) return d3d12_trace::result(method, hr);
        list->Close();
        hr = list->QueryInterface(riid, out);
        list->Release();
        return d3d12_trace::result(method, hr);
    }

    HRESULT STDMETHODCALLTYPE CreateFence(UINT64 initial, D3D12_FENCE_FLAGS flags, REFIID riid, void** out) override {
        constexpr auto method = "ID3D12Device::CreateFence";
        return d3d12_trace::result(method, create_fence(*state_.context, initial, flags, riid, out, identity()));
    }

    HRESULT STDMETHODCALLTYPE CreateCommandSignature(const D3D12_COMMAND_SIGNATURE_DESC* desc, ID3D12RootSignature*,
                                                     REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!desc || desc->NumArgumentDescs == 0 || !desc->pArgumentDescs || desc->ByteStride == 0)
            return E_INVALIDARG;
        leonida::IndirectKind kind = leonida::IndirectKind::draw;
        bool found = false;
        for (UINT i = 0; i < desc->NumArgumentDescs; ++i) {
            switch (desc->pArgumentDescs[i].Type) {
            case D3D12_INDIRECT_ARGUMENT_TYPE_DRAW:
                kind = leonida::IndirectKind::draw;
                found = true;
                break;
            case D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED:
                kind = leonida::IndirectKind::draw_indexed;
                found = true;
                break;
            case D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH:
                kind = leonida::IndirectKind::dispatch;
                found = true;
                break;
            default: break;
            }
        }
        if (!found) return E_NOTIMPL;
        return create_command_signature(kind, desc->ByteStride, riid, out, identity());
    }

    void STDMETHODCALLTYPE GetRaytracingAccelerationStructurePrebuildInfo(
        const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS* desc,
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO* out) override {
        if (!out) return;
        *out = {};
        auto& vk = state_.context->vk;
        if (desc && state_.context->xdxr_acceleration_structures()) {
            // XDXR records: the layout depends only on the triangle or instance count (xdxr_build.h), so the sizes
            // are exact. An update rebuilds the structure, so it needs the same scratch.
            leonida::BuildAccelerationStructure build{};
            build.is_tlas = desc->Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
            if (build.is_tlas) {
                build.instance_count = desc->NumDescs;
            } else {
                for (UINT i = 0; i < desc->NumDescs; ++i) {
                    const D3D12_RAYTRACING_GEOMETRY_DESC* gd =
                        (desc->DescsLayout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS)
                            ? (desc->ppGeometryDescs ? desc->ppGeometryDescs[i] : nullptr)
                            : (desc->pGeometryDescs ? &desc->pGeometryDescs[i] : nullptr);
                    if (!gd || gd->Type != D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES) continue;
                    leonida::RtTriangleGeometry t{};
                    t.vertex_count = gd->Triangles.VertexCount;
                    t.index_count = gd->Triangles.IndexCount;
                    t.index_type = gd->Triangles.IndexFormat == DXGI_FORMAT_UNKNOWN ? 0u : 1u;
                    build.triangles.push_back(t);
                }
            }
            leonida::XdxrLayout layout;
            if (!layout.compute(leonida::VulkanContext::xdxr_elements(build), build.is_tlas)) return;
            out->ResultDataMaxSizeInBytes = layout.size;
            out->ScratchDataSizeInBytes = layout.scratch;
            out->UpdateScratchDataSizeInBytes = layout.scratch;
            return;
        }
        if (!desc || !state_.info.ray_tracing || !vk.vkGetAccelerationStructureBuildSizesKHR) return;

        const bool is_tlas = (desc->Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL);
        std::vector<VkAccelerationStructureGeometryKHR> geoms;
        std::vector<std::uint32_t> max_prims;
        if (is_tlas) {
            VkAccelerationStructureGeometryKHR g{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
            g.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            g.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            g.geometry.instances.arrayOfPointers = VK_FALSE;
            geoms.push_back(g);
            max_prims.push_back(desc->NumDescs);
        } else {
            for (UINT i = 0; i < desc->NumDescs; ++i) {
                const D3D12_RAYTRACING_GEOMETRY_DESC* gd =
                    (desc->DescsLayout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS)
                        ? (desc->ppGeometryDescs ? desc->ppGeometryDescs[i] : nullptr)
                        : (desc->pGeometryDescs ? &desc->pGeometryDescs[i] : nullptr);
                if (!gd || gd->Type != D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES) continue;
                VkAccelerationStructureGeometryKHR g{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
                g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                g.flags = (gd->Flags & D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE) ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0u;
                auto& tr = g.geometry.triangles;
                tr.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                tr.vertexFormat = (gd->Triangles.VertexFormat == DXGI_FORMAT_R32G32_FLOAT) ? VK_FORMAT_R32G32_SFLOAT
                                                                                           : VK_FORMAT_R32G32B32_SFLOAT;
                tr.vertexStride = gd->Triangles.VertexBuffer.StrideInBytes;
                tr.maxVertex = gd->Triangles.VertexCount ? (gd->Triangles.VertexCount - 1u) : 0u;
                tr.indexType = (gd->Triangles.IndexFormat == DXGI_FORMAT_R16_UINT)
                                   ? VK_INDEX_TYPE_UINT16
                                   : ((gd->Triangles.IndexFormat == DXGI_FORMAT_R32_UINT) ? VK_INDEX_TYPE_UINT32
                                                                                          : VK_INDEX_TYPE_NONE_KHR);
                geoms.push_back(g);
                const std::uint32_t prims =
                    (gd->Triangles.IndexFormat != DXGI_FORMAT_UNKNOWN && gd->Triangles.IndexCount >= 3)
                        ? (gd->Triangles.IndexCount / 3u)
                        : (gd->Triangles.VertexCount / 3u);
                max_prims.push_back(prims);
            }
        }
        if (geoms.empty()) return;

        VkAccelerationStructureBuildGeometryInfoKHR bgi{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        bgi.type = is_tlas ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
                           : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        if (desc->Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE)
            bgi.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        if (desc->Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE)
            bgi.flags |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        if (desc->Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD)
            bgi.flags |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        bgi.geometryCount = static_cast<std::uint32_t>(geoms.size());
        bgi.pGeometries = geoms.data();

        VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        vk.vkGetAccelerationStructureBuildSizesKHR(state_.info.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                   &bgi, max_prims.data(), &sizes);
        out->ResultDataMaxSizeInBytes = sizes.accelerationStructureSize;
        out->ScratchDataSizeInBytes = sizes.buildScratchSize;
        out->UpdateScratchDataSizeInBytes = sizes.updateScratchSize;
    }

    HRESULT STDMETHODCALLTYPE CreateDescriptorHeap(const D3D12_DESCRIPTOR_HEAP_DESC* desc, REFIID riid,
                                                   void** out) override {
        constexpr auto method = "ID3D12Device::CreateDescriptorHeap";
        if (!out) return d3d12_trace::result(method, E_POINTER);
        *out = nullptr;
        if (!desc || !desc->NumDescriptors) return d3d12_trace::result(method, E_INVALIDARG);
        auto* heap = new DescriptorHeap(state_, *desc);
        heap->attach_device(identity());
        if (!heap->ok()) {
            heap->Release();
            return d3d12_trace::result(method, E_OUTOFMEMORY);
        }
        return d3d12_trace::result(method, hand_out(heap, riid, out));
    }

    HRESULT STDMETHODCALLTYPE CreateQueryHeap(const D3D12_QUERY_HEAP_DESC* desc, REFIID riid, void** out) override {
        if (out) *out = nullptr;
        if (!desc || !desc->Count || desc->NodeMask > 1) return E_INVALIDARG;  // one node
        auto* heap = new QueryHeap(state_);
        heap->attach_device(identity());
        if (const HRESULT hr = heap->create(*desc); FAILED(hr)) {
            heap->Release();
            return hr;
        }
        if (!out) {  // a capability check: D3D12 allows a null output
            heap->Release();
            return S_FALSE;
        }
        return hand_out(heap, riid, out);
    }

    HRESULT STDMETHODCALLTYPE CreateRootSignature(UINT, const void* blob, SIZE_T size, REFIID riid,
                                                  void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!blob || !size) return E_INVALIDARG;
        // The repository's root-signature model (runtime/src/root_signature.h): parse the RTS0 blob, then pack
        // the parameters into the 32 user-data dwords. pack() refuses what it cannot place (more than 32 dwords:
        // the console spills those to memory, in a layout that is still open), and so does this.
        const rt::rootsig::DecodeResult parsed =
            rt::rootsig::parse_rts0(static_cast<const std::uint8_t*>(blob), size);
        if (!parsed.ok) {
            std::fprintf(stderr, "[rt] d3d12: root signature not understood: %s\n", parsed.error.c_str());
            return E_INVALIDARG;
        }
        const rt::rootsig::PackResult packed = rt::rootsig::pack(parsed.sig);
        if (!packed.ok) {
            std::fprintf(stderr, "[rt] d3d12: root signature refused: %s\n", packed.error.c_str());
            return E_INVALIDARG;
        }
        RootArgumentLayout layout;
        for (std::size_t i = 0; i < parsed.sig.params.size(); ++i) {
            std::uint32_t first = UINT32_MAX;
            for (std::uint32_t d = 0; d < packed.dword_count; ++d)
                if (packed.provenance[d].param_index == i) {
                    first = d;
                    break;
                }
            layout.parameters.push_back({parsed.sig.params[i].type, first, parsed.sig.params[i].num_values});
        }
        auto* signature = new RootSignature(std::move(layout));
        signature->attach_device(identity());
        return hand_out(signature, riid, out);
    }

    // PC slot 11; the Xbox ABI reaches it through device slot 62 (the proxy forwards).
    HRESULT STDMETHODCALLTYPE CreateComputePipelineState(const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc, REFIID riid,
                                                         void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!desc) return E_INVALIDARG;
        auto* pipeline = new PipelineState(state_);
        pipeline->attach_device(identity());
        if (const HRESULT hr = pipeline->create_compute(*desc); FAILED(hr)) {
            pipeline->Release();
            return hr;
        }
        return hand_out(pipeline, riid, out);
    }

    HRESULT STDMETHODCALLTYPE CreateGraphicsPipelineState(const D3D12_GRAPHICS_PIPELINE_STATE_DESC* desc, REFIID riid,
                                                          void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!desc) return E_INVALIDARG;
        auto* pipeline = new PipelineState(state_);
        pipeline->attach_device(identity());
        if (const HRESULT hr = pipeline->create_graphics(*desc); FAILED(hr)) {
            pipeline->Release();
            return hr;
        }
        return hand_out(pipeline, riid, out);
    }

    UINT STDMETHODCALLTYPE GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE) override {
        return kDescriptorSize;
    }

    void STDMETHODCALLTYPE CreateRenderTargetView(ID3D12Resource* resource, const D3D12_RENDER_TARGET_VIEW_DESC* desc,
                                                  D3D12_CPU_DESCRIPTOR_HANDLE destination) override {
        d3d12_trace::record("ID3D12Device::CreateRenderTargetView");
        if (!destination.ptr) return;
        leonida::TargetView view{};
        leonida::ResourceId rid = 0;
        D3D12_RESOURCE_DESC rd{};
        const Resource* r = Resource::from<Resource>(resource);
        if (r) {
            rid = r->id();
            rd = const_cast<Resource*>(r)->GetDesc();
        } else if (resource && state_.context->has_resource(resource_id(resource))) {
            rid = resource_id(resource);
            rd = resource->GetDesc();
        }
        if (rid) {
            view = {rid, 0, 0, rd.DepthOrArraySize};
            if (rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
                view = volume_rtv(rid, rd, desc);
            } else if (desc && desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2D) {
                view = {rid, desc->Texture2D.MipSlice, 0, 1};
            } else if (desc && desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2DMS) {
                view = {rid, 0, 0, 1};
            } else if (desc && desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2DARRAY) {
                view = {rid, desc->Texture2DArray.MipSlice, desc->Texture2DArray.FirstArraySlice,
                        desc->Texture2DArray.ArraySize};
            } else if (desc && desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY) {
                // Multisample textures have one mip, so only the slice range is chosen (as for the DSV).
                view = {rid, 0, desc->Texture2DMSArray.FirstArraySlice, desc->Texture2DMSArray.ArraySize};
            } else if (desc) {
                log_once("render-target view dimension not supported:", desc->ViewDimension);
                view = {};
            }
            const bool typeless_rgba8 = rd.Format == DXGI_FORMAT_R8G8B8A8_TYPELESS;
            const bool typed_rgba8 = rd.Format == DXGI_FORMAT_R8G8B8A8_UNORM ||
                rd.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || rd.Format == DXGI_FORMAT_R8G8B8A8_UINT;
            if (rd.Format == DXGI_FORMAT_A8_UNORM || (desc && desc->Format == DXGI_FORMAT_A8_UNORM)) {
                view = {};
            } else if (rd.Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP ||
                       (desc && desc->Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP)) {
                if (rd.Format != DXGI_FORMAT_R9G9B9E5_SHAREDEXP || !(rd.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) ||
                    (desc && desc->Format != DXGI_FORMAT_UNKNOWN && desc->Format != rd.Format))
                    view = {};
            } else if (typeless_rgba8 || typed_rgba8) {
                const DXGI_FORMAT requested = desc && desc->Format != DXGI_FORMAT_UNKNOWN ? desc->Format : rd.Format;
                // A typed UNORM RGBA8 resource with a castable list may be viewed in a listed family format
                // (D3D-85's committed path, D3D-90's placed one): its image is mutable and lists the view
                // format, so the reinterpretation is the typeless path's. Only a UNORM image is advertised,
                // as that is what the clear path reinterprets. The switch knows the RGBA8 family only.
                const bool listed = requested == rd.Format ||
                                    (r && r->has_view_format_list() &&
                                     r->allows_view_format(vulkan_format(requested, false).format));
                if (typeless_rgba8 || (requested != rd.Format && listed &&
                                       rd.Format == DXGI_FORMAT_R8G8B8A8_UNORM)) {
                    const bool plain = desc && desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2D;
                    const bool array = desc && desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
                    const UINT plane = plain ? desc->Texture2D.PlaneSlice :
                        array ? desc->Texture2DArray.PlaneSlice : 0;
                    // Subtraction keeps malformed layer ranges from overflowing during validation.
                    if ((!plain && !array) || rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
                        rd.SampleDesc.Count != 1 || plane || !view.resource || view.mip >= rd.MipLevels ||
                        !view.layer_count || view.first_layer >= rd.DepthOrArraySize ||
                        view.layer_count > rd.DepthOrArraySize - view.first_layer ||
                        !(rd.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) {
                        view = {};
                    } else {
                        switch (requested) {
                        case DXGI_FORMAT_R8G8B8A8_UNORM: view.format = leonida::ColorViewFormat::rgba8_unorm; break;
                        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: view.format = leonida::ColorViewFormat::rgba8_srgb; break;
                        case DXGI_FORMAT_R8G8B8A8_UINT: view.format = leonida::ColorViewFormat::rgba8_uint; break;
                        case DXGI_FORMAT_R8G8B8A8_SINT: view.format = leonida::ColorViewFormat::rgba8_sint; break;
                        case DXGI_FORMAT_R8G8B8A8_SNORM: view.format = leonida::ColorViewFormat::rgba8_snorm; break;
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
        d3d12_trace::record("ID3D12Device::CreateDepthStencilView");
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
            } else if (desc && desc->ViewDimension == D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY) {
                // Multisample textures have one mip, so only the slice range is chosen.
                view = {r->id(), 0, desc->Texture2DMSArray.FirstArraySlice, desc->Texture2DMSArray.ArraySize};
            } else if (desc) {
                log_once("depth-stencil view dimension not supported:", desc->ViewDimension);
                view = {};
            }
            if (r->desc().Format == DXGI_FORMAT_A8_UNORM || (desc && desc->Format == DXGI_FORMAT_A8_UNORM)) view = {};
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
        d3d12_trace::record("ID3D12Device::CreateShaderResourceView");
        using leonida::gfx10::ImageType;
        if (!destination.ptr) return;
        std::memset(reinterpret_cast<void*>(destination.ptr), 0, kDescriptorSize);
        const Resource* r = Resource::from<Resource>(resource);
        if (!r) {
            if (resource) stub("ID3D12Device::CreateShaderResourceView(resource from another device)");
            return;
        }
        if (!r->image()) return stub("ID3D12Device::CreateShaderResourceView(buffer)");
        if (r->desc().Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) {
            log_once("shader-resource view denied by resource flags; format", int(r->desc().Format));
            return;
        }
        TextureView view{};
        if (!desc) {
            const bool array = r->desc().DepthOrArraySize > 1;
            if (r->desc().Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && !array)
                view.type = ImageType::image_1d;
            else if (r->desc().Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
                view.type = ImageType::image_3d;
            else if (r->desc().SampleDesc.Count > 1)
                view.type = array ? ImageType::image_2d_msaa_array : ImageType::image_2d_msaa;
            else if (array)
                view.type = ImageType::image_2d_array;
        } else {
            view.format = desc->Format;
            view.mapping = desc->Shader4ComponentMapping;
            switch (desc->ViewDimension) {
            case D3D12_SRV_DIMENSION_TEXTURE1D:
                view.type = ImageType::image_1d;
                view.first_mip = desc->Texture1D.MostDetailedMip;
                view.mip_count = desc->Texture1D.MipLevels;
                view.min_lod = desc->Texture1D.ResourceMinLODClamp;
                break;
            case D3D12_SRV_DIMENSION_TEXTURE3D:
                view.type = leonida::gfx10::ImageType::image_3d;
                view.first_mip = desc->Texture3D.MostDetailedMip;
                view.mip_count = desc->Texture3D.MipLevels;
                view.min_lod = desc->Texture3D.ResourceMinLODClamp;
                break;
            case D3D12_SRV_DIMENSION_TEXTURE2DMS:
                view.type = ImageType::image_2d_msaa;
                view.layer_count = 1;
                break;
            case D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY:
                view.type = ImageType::image_2d_msaa_array;
                view.first_layer = desc->Texture2DMSArray.FirstArraySlice;
                view.layer_count = desc->Texture2DMSArray.ArraySize;
                break;
            case D3D12_SRV_DIMENSION_TEXTURE2D:
                view.first_mip = desc->Texture2D.MostDetailedMip;
                view.mip_count = desc->Texture2D.MipLevels;
                view.layer_count = 1;
                view.plane = desc->Texture2D.PlaneSlice;
                view.min_lod = desc->Texture2D.ResourceMinLODClamp;
                break;
            case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
                view.type = ImageType::image_2d_array;
                view.first_mip = desc->Texture2DArray.MostDetailedMip;
                view.mip_count = desc->Texture2DArray.MipLevels;
                view.first_layer = desc->Texture2DArray.FirstArraySlice;
                view.layer_count = desc->Texture2DArray.ArraySize;
                view.plane = desc->Texture2DArray.PlaneSlice;
                view.min_lod = desc->Texture2DArray.ResourceMinLODClamp;
                break;
            case D3D12_SRV_DIMENSION_TEXTURECUBE:
                view.type = ImageType::cube;
                view.first_mip = desc->TextureCube.MostDetailedMip;
                view.mip_count = desc->TextureCube.MipLevels;
                view.layer_count = 6;
                view.min_lod = desc->TextureCube.ResourceMinLODClamp;
                break;
            case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
                view.type = ImageType::cube;
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
        d3d12_trace::record("ID3D12Device::CreateUnorderedAccessView");
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
        d3d12_trace::record("ID3D12Device::CreateSampler");
        if (!destination.ptr) return;
        std::memset(reinterpret_cast<void*>(destination.ptr), 0, kDescriptorSize);
        if (!desc) return;
        const leonida::gfx10::SamplerWords words = leonida::gfx10::encode(sampler_descriptor(*desc));
        std::memcpy(reinterpret_cast<void*>(destination.ptr), words.data(), sizeof(words));
        state_.bind_sampler(words);
    }

    // CreateSampler with a UINT border colour and non-normalized coordinates (D3D-28). A description the PC runtime
    // refuses leaves the slot as it was, as the runtime itself does (without the debug layer) on an RTX 4090 and
    // on WARP: docs/d3d12_vulkan.md section 3.2.
    void STDMETHODCALLTYPE CreateSampler2(const D3D12_SAMPLER_DESC2* desc,
                                          D3D12_CPU_DESCRIPTOR_HANDLE destination) override {
        d3d12_trace::record("ID3D12Device11::CreateSampler2");
        if (!destination.ptr || !desc) return;
        const char* refusal = sampler2_refusal(*desc);
        if (refusal) {
            log_once(refusal, desc->Flags);
            return;
        }
        const bool uint_border = desc->Flags & D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR;
        D3D12_SAMPLER_DESC base{desc->Filter,   desc->AddressU,      desc->AddressV,       desc->AddressW,
                                desc->MipLODBias, desc->MaxAnisotropy, desc->ComparisonFunc, {},
                                desc->MinLOD,   desc->MaxLOD};
        if (!uint_border) std::memcpy(base.BorderColor, desc->FloatBorderColor, sizeof(base.BorderColor));
        leonida::gfx10::Sampler sampler = sampler_descriptor(base);
        if (uint_border) {
            // The S#'s border types are the float ones', plus the layer's integer mark (kIntegerBorderPtr), which
            // gives the integer border its own key and VK_BORDER_COLOR_INT_* in Vulkan.
            namespace g = leonida::gfx10;
            const UINT* c = desc->UintBorderColor;
            const bool black = c[0] == 0 && c[1] == 0 && c[2] == 0;
            sampler.border_color = black && c[3] == 0                                 ? g::BorderColor::transparent_black
                                   : black && c[3] == 1                               ? g::BorderColor::opaque_black
                                   : c[0] == 1 && c[1] == 1 && c[2] == 1 && c[3] == 1 ? g::BorderColor::opaque_white
                                                                                      : g::BorderColor::table;
            sampler.integer_border = true;
        }
        sampler.unnormalized = desc->Flags & D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES;
        const leonida::gfx10::SamplerWords words = leonida::gfx10::encode(sampler);
        std::memset(reinterpret_cast<void*>(destination.ptr), 0, kDescriptorSize);
        std::memcpy(reinterpret_cast<void*>(destination.ptr), words.data(), sizeof(words));
        state_.bind_sampler(words);
    }

    // Why the PC runtime refuses a D3D12_SAMPLER_DESC2, or null. The non-normalized rules are its debug layer's
    // messages, which match Vulkan's for unnormalizedCoordinates; AddressW is free in both.
    static const char* sampler2_refusal(const D3D12_SAMPLER_DESC2& desc) {
        if (desc.Flags & ~UINT(D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR | D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES))
            return "CreateSampler2 refused: unknown sampler flags; flags";
        if (!(desc.Flags & D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES)) return nullptr;
        if (D3D12_DECODE_MIN_FILTER(desc.Filter) != D3D12_DECODE_MAG_FILTER(desc.Filter) ||
            D3D12_DECODE_MIP_FILTER(desc.Filter) != D3D12_FILTER_TYPE_POINT ||
            D3D12_DECODE_IS_ANISOTROPIC_FILTER(desc.Filter) ||
            D3D12_DECODE_FILTER_REDUCTION(desc.Filter) == D3D12_FILTER_REDUCTION_TYPE_COMPARISON)
            return "CreateSampler2 refused: non-normalized sampler filter; flags";
        const auto clamps = [](D3D12_TEXTURE_ADDRESS_MODE m) {
            return m == D3D12_TEXTURE_ADDRESS_MODE_CLAMP || m == D3D12_TEXTURE_ADDRESS_MODE_BORDER;
        };
        if (!clamps(desc.AddressU) || !clamps(desc.AddressV))
            return "CreateSampler2 refused: non-normalized sampler address mode; flags";
        if (desc.MinLOD != 0 || desc.MaxLOD != 0 || desc.MipLODBias != 0)
            return "CreateSampler2 refused: non-normalized sampler LOD; flags";
        return nullptr;
    }

    HRESULT STDMETHODCALLTYPE CreateCommittedResource(const D3D12_HEAP_PROPERTIES* heap, D3D12_HEAP_FLAGS heap_flags,
                                                      const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES initial_state,
                                                      const D3D12_CLEAR_VALUE*, REFIID riid, void** out) override {
        constexpr auto method = "ID3D12Device::CreateCommittedResource";
        return d3d12_trace::result(method, create_committed(heap, heap_flags, desc, initial_state, 0, nullptr, riid, out));
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
        constexpr auto method = "ID3D12Device::CreateReservedResource";
        return d3d12_trace::result(method, create_reserved(desc, riid, out));
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

    // The 1/2/3 variants (D3D-28) are the same calls with newer arguments, answered as the PC runtime answers them
    // (probed on an RTX 4090; docs/d3d12_vulkan.md section 3.2 has the table). The initial state or layout is not
    // used: new textures are tracked as UNDEFINED and moved on first use, and buffers have no layout.
    HRESULT STDMETHODCALLTYPE CreateCommittedResource1(const D3D12_HEAP_PROPERTIES* heap, D3D12_HEAP_FLAGS heap_flags,
                                                       const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES state,
                                                       const D3D12_CLEAR_VALUE* clear,
                                                       ID3D12ProtectedResourceSession* session, REFIID riid,
                                                       void** out) override {
        if (out) *out = nullptr;
        if (session) return refuse_protected_session(1);
        return CreateCommittedResource(heap, heap_flags, desc, state, clear, riid, out);
    }

    HRESULT STDMETHODCALLTYPE CreateCommittedResource2(const D3D12_HEAP_PROPERTIES* heap, D3D12_HEAP_FLAGS heap_flags,
                                                       const D3D12_RESOURCE_DESC1* desc, D3D12_RESOURCE_STATES state,
                                                       const D3D12_CLEAR_VALUE* clear,
                                                       ID3D12ProtectedResourceSession* session, REFIID riid,
                                                       void** out) override {
        if (out) *out = nullptr;
        if (!desc) return E_INVALIDARG;
        if (session) return refuse_protected_session(2);
        if (texture_mip_region(*desc)) return E_INVALIDARG;
        const D3D12_RESOURCE_DESC base = resource_desc(*desc);
        return CreateCommittedResource(heap, heap_flags, &base, state, clear, riid, out);
    }

    HRESULT STDMETHODCALLTYPE CreateCommittedResource3(const D3D12_HEAP_PROPERTIES* heap, D3D12_HEAP_FLAGS heap_flags,
                                                       const D3D12_RESOURCE_DESC1* desc, D3D12_BARRIER_LAYOUT layout,
                                                       const D3D12_CLEAR_VALUE*,
                                                       ID3D12ProtectedResourceSession* session, UINT32 castable_count,
                                                       const DXGI_FORMAT* castable, REFIID riid, void** out) override {
        if (out) *out = nullptr;
        if (!desc) return E_INVALIDARG;
        if (session) return refuse_protected_session(3);
        if (texture_mip_region(*desc)) return E_INVALIDARG;
        // Enhanced barriers: a buffer has no layout, so its initial layout must be UNDEFINED.
        if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER && layout != D3D12_BARRIER_LAYOUT_UNDEFINED)
            return E_INVALIDARG;
        // A count with a null array is no list (as on the PC runtime). The layer reports relaxed format casting, so
        // a real list is passed through; only buffers are refused.
        const D3D12_RESOURCE_DESC base = resource_desc(*desc);
        // D3D12_BARRIER_LAYOUT has no acceleration-structure value; an AS buffer's state comes through the
        // legacy CreateCommittedResource path. Buffers here start COMMON, as before.
        return create_committed(heap, heap_flags, &base, D3D12_RESOURCE_STATE_COMMON, castable_count, castable,
                                riid, out);
    }

    // Device9. The creator ID only groups queues for tools; one creator is as good as another here.
    HRESULT STDMETHODCALLTYPE CreateCommandQueue1(const D3D12_COMMAND_QUEUE_DESC* desc, REFIID, REFIID riid,
                                                  void** out) override {
        return CreateCommandQueue(desc, riid, out);
    }

    // Device8. A texture with a mip region gets the failure answer (every output all ones), as on the PC runtime;
    // a buffer's mip region is ignored.
    void STDMETHODCALLTYPE GetCopyableFootprints1(const D3D12_RESOURCE_DESC1* desc, UINT first, UINT count,
                                                  UINT64 base, D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layouts, UINT* rows,
                                                  UINT64* row_sizes, UINT64* total) override {
        const D3D12_RESOURCE_DESC base_desc = desc ? resource_desc(*desc) : D3D12_RESOURCE_DESC{};
        const bool usable = desc && !texture_mip_region(*desc);
        GetCopyableFootprints(usable ? &base_desc : nullptr, first, count, base, layouts, rows, row_sizes, total);
    }

    // Heaps and placed resources (section 3.3). A heap reserves its size in the arena, so a buffer placed at a
    // 64 KiB-aligned offset is simply that offset into the arena, and its console VA is the heap's place plus it.
    HRESULT STDMETHODCALLTYPE CreateHeap(const D3D12_HEAP_DESC* desc, REFIID riid, void** out) override {
        if (out) *out = nullptr;
        if (!desc || !desc->SizeInBytes) return E_INVALIDARG;
        if (!state_.arena.mapped) {
            log_once("heaps need the GPU address arena (bufferDeviceAddress); heap size in KiB:",
                     int(desc->SizeInBytes / 1024));
            return E_NOTIMPL;
        }
        auto* heap = new Heap(state_, *desc);
        heap->attach_device(identity());
        if (!heap->reserved()) {
            log_once("GPU address arena full (raise LEONIDA_ARENA_MB); heap size in KiB:", int(desc->SizeInBytes / 1024));
            heap->Release();
            return E_OUTOFMEMORY;
        }
        if (!out) {
            heap->Release();
            return S_FALSE;
        }
        return hand_out(heap, riid, out);
    }

    HRESULT STDMETHODCALLTYPE CreatePlacedResource(ID3D12Heap* heap_interface, UINT64 heap_offset,
                                                   const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES,
                                                   const D3D12_CLEAR_VALUE*, REFIID riid, void** out) override {
        return create_placed(heap_interface, heap_offset, desc, 0, nullptr, riid, out);
    }

    // Device8: the DESC1 placed variant without a castable list. A texture with a mip region gets the failure
    // answer, as on the PC runtime; a buffer's mip region is ignored.
    HRESULT STDMETHODCALLTYPE CreatePlacedResource1(ID3D12Heap* heap_interface, UINT64 heap_offset,
                                                    const D3D12_RESOURCE_DESC1* desc, D3D12_RESOURCE_STATES,
                                                    const D3D12_CLEAR_VALUE*, REFIID riid, void** out) override {
        if (out) *out = nullptr;
        if (!desc) return E_INVALIDARG;
        if (texture_mip_region(*desc)) return E_INVALIDARG;
        const D3D12_RESOURCE_DESC base = resource_desc(*desc);
        return create_placed(heap_interface, heap_offset, &base, 0, nullptr, riid, out);
    }

    // Device10: an initial layout instead of a state, and the castable-format list, which the placed texture's
    // Vulkan image takes exactly as the committed path's does. A count with a null array is no list (as on the
    // PC runtime); the layer reports RelaxedFormatCastingSupported, so a real list is passed through.
    HRESULT STDMETHODCALLTYPE CreatePlacedResource2(ID3D12Heap* heap_interface, UINT64 heap_offset,
                                                    const D3D12_RESOURCE_DESC1* desc, D3D12_BARRIER_LAYOUT layout,
                                                    const D3D12_CLEAR_VALUE*, UINT32 castable_count,
                                                    const DXGI_FORMAT* castable, REFIID riid, void** out) override {
        if (out) *out = nullptr;
        if (!desc) return E_INVALIDARG;
        if (texture_mip_region(*desc)) return E_INVALIDARG;
        // Enhanced barriers: a buffer has no layout, so its initial layout must be UNDEFINED.
        if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER && layout != D3D12_BARRIER_LAYOUT_UNDEFINED)
            return E_INVALIDARG;
        const D3D12_RESOURCE_DESC base = resource_desc(*desc);
        return create_placed(heap_interface, heap_offset, &base, castable_count, castable, riid, out);
    }

    // The placed path every CreatePlacedResource variant shares. A castable-format list reaches the placed
    // texture's Vulkan image as the committed path's does (create_committed): a buffer with a list is
    // E_INVALIDARG, and so is a format with no Vulkan mapping.
    HRESULT create_placed(ID3D12Heap* heap_interface, UINT64 heap_offset, const D3D12_RESOURCE_DESC* desc,
                          UINT32 castable_count, const DXGI_FORMAT* castable, REFIID riid, void** out) {
        if (out) *out = nullptr;
        auto* heap = Heap::from<Heap>(heap_interface);
        if (!heap || !desc) return E_INVALIDARG;
        if (castable_count && castable && desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            log_once("castable formats on a placed buffer are invalid; count", int(castable_count));
            return E_INVALIDARG;
        }
        if (heap_offset % D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT) {
            log_once("placed resource offset not 64 KiB aligned:", int(heap_offset));
            return E_INVALIDARG;
        }
        if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            if (heap_offset + desc->Width > heap->bytes()) return E_INVALIDARG;
        } else if (heap_offset >= heap->bytes()) {
            return E_INVALIDARG;
        }
        if (desc->Layout == D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE) {
            log_once("tiled placed resource not supported yet; dimension", int(desc->Dimension));
            return E_NOTIMPL;
        }
        // Only buffers are addressable, as in D3D12: a placed texture keeps a null VA and its own Vulkan image.
        const VkDeviceSize placed_va = desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? heap->offset() + heap_offset : 0;
        auto* resource = new Resource(state_, *desc, heap->properties(), heap->flags(), placed_va);
        resource->attach_device(identity());
        HRESULT hr = E_NOTIMPL;
        if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            hr = resource->create_placed_buffer();
        } else if (desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
            log_once("placed resource dimension not supported:", int(desc->Dimension));
        } else if (desc->SampleDesc.Count > 1) {
            log_once("multisampled placed resource not supported; samples", int(desc->SampleDesc.Count));
        } else {
            hr = S_OK;
            const Format format = state_.texture_format(*desc);
            if (format.format == VK_FORMAT_UNDEFINED) {
                hr = E_INVALIDARG;
            } else {
                std::vector<VkFormat> cast_formats;
                hr = castable_view_formats(format, castable_count, castable, cast_formats);
                if (SUCCEEDED(hr)) hr = resource->create_texture(format, cast_formats);
            }
        }
        if (FAILED(hr)) {
            resource->Release();
            return hr;
        }
        if (!out) {
            resource->Release();
            return S_FALSE;
        }
        return hand_out(resource, riid, out);
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
        if (!desc || !xr_bias_compatible(*desc) || rgb96_refused(*desc)) return fail();
        const auto align = [](UINT64 value, UINT64 to) { return (value + to - 1) / to * to; };
        if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            // The PC runtime refuses any buffer field that isn't its one valid value (all measured on WARP, Width,
            // Height, Format, MipLevels 0, Layout and Count 0 also on an NVIDIA GPU). It doesn't look at Flags, so a
            // buffer with the render-target flag still gets its footprint (D3D-122).
            if (first != 0 || count != 1 || !desc->Width || desc->Height != 1 || desc->DepthOrArraySize != 1 ||
                desc->MipLevels != 1 || desc->Format != DXGI_FORMAT_UNKNOWN ||
                desc->Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR || desc->SampleDesc.Count != 1 ||
                desc->SampleDesc.Quality != 0 ||
                (desc->Alignment && desc->Alignment != D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT))
                return fail();
            if (layouts) layouts[0] = {base, {DXGI_FORMAT_UNKNOWN, UINT(desc->Width), 1, 1,
                                              UINT(align(desc->Width, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT))}};
            if (rows) rows[0] = 1;
            if (row_sizes) row_sizes[0] = desc->Width;
            if (total) *total = desc->Width;
            return;
        }
        const bool volume = desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        const bool one_d = desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D;
        if (!volume && !one_d && desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
            log_once("GetCopyableFootprints dimension not supported:", desc->Dimension);
            return fail();
        }
        if (one_d && desc->Height != 1) return fail();
        // Structural checks only, as measured on the PC runtime (D3D-122). Not allocation()'s: the footprint path
        // accepts extra mips, 2D ROW_MAJOR and Quality > 0 at one sample, and extent limits, alignment and
        // multisample quality differ between adapters, so those stay unchecked.
        const UINT samples = desc->SampleDesc.Count;
        const bool multisample_ok = desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc->MipLevels == 1 &&
                                    desc->Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR && !(samples & (samples - 1)) &&
                                    (desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                                                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL));
        if (!desc->Width || !desc->Height || !desc->DepthOrArraySize || !samples || (samples > 1 && !multisample_ok))
            return fail();
        const Format format = texture_storage_format(desc->Format, true);  // typeless depth families are planar either way
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
            const leonida::BlockFormat block = footprint_block(format, aspect);
            if (!block.bytes) return fail();
            const UINT width = std::max<UINT>(1, UINT(desc->Width >> mip));
            const UINT height = one_d ? 1 : std::max<UINT>(1, desc->Height >> mip);
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
        constexpr auto method = "ID3D12Device::CheckFeatureSupport";
        log_once("CheckFeatureSupport feature", feature);
        if (!data) return d3d12_trace::result(method, E_INVALIDARG);
        constexpr VkSampleCountFlags d3d12_multisample_sample_counts =
            VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT | VK_SAMPLE_COUNT_8_BIT |
            VK_SAMPLE_COUNT_16_BIT | VK_SAMPLE_COUNT_32_BIT;
        switch (feature) {
        case D3D12_FEATURE_FEATURE_LEVELS: {
            if (size != sizeof(D3D12_FEATURE_DATA_FEATURE_LEVELS)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* levels = static_cast<D3D12_FEATURE_DATA_FEATURE_LEVELS*>(data);
            // The host (hardware and WARP) refuses an empty or missing list before reading it.
            if (!levels->NumFeatureLevels || !levels->pFeatureLevelsRequested)
                return d3d12_trace::result(method, E_INVALIDARG);
            D3D_FEATURE_LEVEL best = D3D_FEATURE_LEVEL(0);
            for (UINT i = 0; i < levels->NumFeatureLevels; ++i)
                if (levels->pFeatureLevelsRequested[i] <= D3D_FEATURE_LEVEL_12_1)
                    best = std::max(best, levels->pFeatureLevelsRequested[i]);
            levels->MaxSupportedFeatureLevel = best;  // 0 with DXGI_ERROR_UNSUPPORTED, as the host writes it
            if (!best) return d3d12_trace::result(method, DXGI_ERROR_UNSUPPORTED);
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_ARCHITECTURE:
        case D3D12_FEATURE_ARCHITECTURE1: {
            const bool v1 = feature == D3D12_FEATURE_ARCHITECTURE1;
            if (size != (v1 ? sizeof(D3D12_FEATURE_DATA_ARCHITECTURE1) : sizeof(D3D12_FEATURE_DATA_ARCHITECTURE)))
                return d3d12_trace::result(method, E_INVALIDARG);
            auto* arch = static_cast<D3D12_FEATURE_DATA_ARCHITECTURE1*>(data);
            const UINT node = arch->NodeIndex;
            if (node != 0) return d3d12_trace::result(method, E_INVALIDARG);
            // A discrete desktop GPU: not tile-based, no UMA. The console is UMA (question 11); the layer keeps
            // upload heaps host-visible, so the game's upload path still works.
            arch->TileBasedRenderer = FALSE;
            arch->UMA = FALSE;
            arch->CacheCoherentUMA = FALSE;
            if (v1) arch->IsolatedMMU = TRUE;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS)) return d3d12_trace::result(method, E_INVALIDARG);
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
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS1: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS1)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS1*>(data);
            const DeviceState::SubgroupFacts& facts = state_.subgroup_facts();
            *opts = {};
            opts->WaveOps = facts.wave_ops;
            opts->WaveLaneCountMin = facts.lane_min;
            opts->WaveLaneCountMax = facts.lane_max;
            // D3D12 counts every lane of the device here, and Vulkan has no such number. This is one full wave
            // (section 4.1): a game that sizes work from it plans for less parallelism than the GPU has, not more.
            opts->TotalLaneCount = facts.lane_max;
            opts->ExpandedComputeResourceStates = TRUE;  // Vulkan has no resource states to expand
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS8: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS8)) return d3d12_trace::result(method, E_INVALIDARG);
            // Vulkan BC image extents are in texels; only the transfer footprints round up to blocks.
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS8*>(data);
            opts->UnalignedBlockTexturesSupported = state_.info.texture_compression_bc;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS2: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS2)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS2*>(data);
            *opts = {};
            opts->DepthBoundsTestSupported = FALSE;
            opts->ProgrammableSamplePositionsTier = D3D12_PROGRAMMABLE_SAMPLE_POSITIONS_TIER_NOT_SUPPORTED;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS3: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS3)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS3*>(data);
            *opts = {};
            opts->CopyQueueTimestampQueriesSupported = FALSE;
            // Only typeless resources set VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT.
            opts->CastingFullyTypedFormatSupported = FALSE;
            opts->WriteBufferImmediateSupportFlags = D3D12_COMMAND_LIST_SUPPORT_FLAG_NONE;
            opts->ViewInstancingTier = D3D12_VIEW_INSTANCING_TIER_NOT_SUPPORTED;
            opts->BarycentricsSupported = FALSE;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS4: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS4)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS4*>(data);
            *opts = {};
            // GetResourceAllocationInfo implements D3D12_SMALL_MSAA_RESOURCE_PLACEMENT_ALIGNMENT (64 KiB).
            opts->MSAA64KBAlignedTextureSupported = TRUE;
            opts->SharedResourceCompatibilityTier = D3D12_SHARED_RESOURCE_COMPATIBILITY_TIER_0;
            opts->Native16BitShaderOpsSupported = FALSE;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS5: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS5)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS5*>(data);
            *opts = {};
            opts->SRVOnlyTiledResourceTier3 = FALSE;
            opts->RenderPassesTier = D3D12_RENDER_PASS_TIER_0;
            opts->RaytracingTier = D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS7: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS7)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS7*>(data);
            *opts = {};
            opts->MeshShaderTier = D3D12_MESH_SHADER_TIER_NOT_SUPPORTED;
            opts->SamplerFeedbackTier = D3D12_SAMPLER_FEEDBACK_TIER_NOT_SUPPORTED;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS9: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS9)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS9*>(data);
            *opts = {};
            opts->MeshShaderPipelineStatsSupported = FALSE;
            opts->MeshShaderSupportsFullRangeRenderTargetArrayIndex = FALSE;
            opts->AtomicInt64OnTypedResourceSupported = FALSE;
            opts->AtomicInt64OnGroupSharedSupported = FALSE;
            opts->DerivativesInMeshAndAmplificationShadersSupported = FALSE;
            opts->WaveMMATier = D3D12_WAVE_MMA_TIER_NOT_SUPPORTED;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS11: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS11)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS11*>(data);
            *opts = {};
            opts->AtomicInt64OnDescriptorHeapResourceSupported = FALSE;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS13: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS13)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS13*>(data);
            *opts = {};
            opts->UnrestrictedBufferTextureCopyPitchSupported = FALSE;
            opts->UnrestrictedVertexElementAlignmentSupported = FALSE;
            opts->InvertedViewportHeightFlipsYSupported = FALSE;
            opts->InvertedViewportDepthFlipsZSupported = FALSE;
            opts->TextureCopyBetweenDimensionsSupported = FALSE;
            opts->AlphaBlendFactorSupported = FALSE;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS14: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS14)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS14*>(data);
            *opts = {};
            opts->AdvancedTextureOpsSupported = FALSE;
            opts->WriteableMSAATexturesSupported = FALSE;
            opts->IndependentFrontAndBackStencilRefMaskSupported = FALSE;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS15: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS15)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS15*>(data);
            *opts = {};
            opts->TriangleFanSupported = FALSE;
            opts->DynamicIndexBufferStripCutSupported = FALSE;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS16: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS16)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS16*>(data);
            *opts = {};
            opts->DynamicDepthBiasSupported = FALSE;
            opts->GPUUploadHeapSupported = FALSE;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS12: {
            // Vulkan 1.3's VkImageFormatListCreateInfo gives a mutable image the same relaxed format casting that
            // D3D12's castable-format lists describe. The layer also implements enhanced barriers (Barrier).
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS12)) return E_INVALIDARG;
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS12*>(data);
            *opts = {};
            opts->MSPrimitivesPipelineStatisticIncludesCulledPrimitives = D3D12_TRI_STATE_UNKNOWN;
            opts->EnhancedBarriersSupported = TRUE;
            opts->RelaxedFormatCastingSupported = TRUE;
            return S_OK;
        }
        case D3D12_FEATURE_D3D12_OPTIONS17: {
            // Non-normalized samplers are core Vulkan (unnormalizedCoordinates), with the restrictions
            // CreateSampler2 checks. No manual write tracking, as the RTX 4090 answers too.
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS17)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS17*>(data);
            *opts = {};
            opts->NonNormalizedCoordinateSamplersSupported = TRUE;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS18: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS18)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS18*>(data);
            *opts = {};
            opts->RenderPassesValid = FALSE;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS19: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS19)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS19*>(data);
            *opts = {};
            opts->MismatchingOutputDimensionsSupported = FALSE;
            opts->SupportedSampleCountsWithNoOutputs = state_.no_output_sample_counts;
            opts->PointSamplingAddressesNeverRoundUp = FALSE;
            opts->RasterizerDesc2Supported = FALSE;
            opts->NarrowQuadrilateralLinesSupported = FALSE;
            opts->AnisoFilterWithPointMipSupported = state_.info.sampler_anisotropy ? TRUE : FALSE;
            opts->MaxSamplerDescriptorHeapSize = D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;
            opts->MaxSamplerDescriptorHeapSizeWithStaticSamplers = D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;
            opts->MaxViewDescriptorHeapSize = D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_2;
            opts->ComputeOnlyCustomHeapSupported = FALSE;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS20: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS20)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS20*>(data);
            *opts = {};
            opts->ComputeOnlyWriteWatchSupported = FALSE;
            opts->RecreateAtTier = D3D12_RECREATE_AT_TIER_NOT_SUPPORTED;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_D3D12_OPTIONS21: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS21)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS21*>(data);
            *opts = {};
            opts->WorkGraphsTier = D3D12_WORK_GRAPHS_TIER_NOT_SUPPORTED;
            opts->ExecuteIndirectTier = D3D12_EXECUTE_INDIRECT_TIER_1_0;
            opts->SampleCmpGradientAndBiasSupported = FALSE;
            opts->ExtendedCommandInfoSupported = FALSE;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT: {
            if (size != sizeof(D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* va = static_cast<D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT*>(data);
            // The console and PC BDA support 48-bit GPU virtual addresses (section 4.1).
            va->MaxGPUVirtualAddressBitsPerResource = 40;
            va->MaxGPUVirtualAddressBitsPerProcess = 48;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_FORMAT_INFO: {
            if (size != sizeof(D3D12_FEATURE_DATA_FORMAT_INFO)) return E_INVALIDARG;
            auto* info = static_cast<D3D12_FEATURE_DATA_FORMAT_INFO*>(data);
            // Plane count describes the DXGI layout even when the backend cannot create the texture.
            switch (info->Format) {
            case DXGI_FORMAT_R32G8X24_TYPELESS:
            case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
            case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
            case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
            case DXGI_FORMAT_R24G8_TYPELESS:
            case DXGI_FORMAT_D24_UNORM_S8_UINT:
            case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
            case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
            case DXGI_FORMAT_NV12:
            case DXGI_FORMAT_P010:
            case DXGI_FORMAT_P016:
            case DXGI_FORMAT_NV11:
                info->PlaneCount = 2;
                break;
            case DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE:
            case DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE:
            case DXGI_FORMAT_A4B4G4R4_UNORM:
                info->PlaneCount = 1;
                break;
            default:
                // Native D3D12 accepts UNKNOWN but rejects R1_UNORM and the gaps in the enum.
                if (UINT(info->Format) > DXGI_FORMAT_B4G4R4A4_UNORM || info->Format == DXGI_FORMAT_R1_UNORM)
                    return E_INVALIDARG;
                info->PlaneCount = 1;
            }
            return S_OK;
        }
        case D3D12_FEATURE_FORMAT_SUPPORT: {
            if (size != sizeof(D3D12_FEATURE_DATA_FORMAT_SUPPORT)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* fs = static_cast<D3D12_FEATURE_DATA_FORMAT_SUPPORT*>(data);
            fs->Support1 = D3D12_FORMAT_SUPPORT1_NONE;
            fs->Support2 = D3D12_FORMAT_SUPPORT2_NONE;
            const Format fmt = texture_storage_format(fs->Format, false);
            const Format depth_fmt = vulkan_format(fs->Format, true);
            VkFormat vk_fmt = fmt.format != VK_FORMAT_UNDEFINED ? fmt.format : depth_fmt.format;
            // Report what creation makes: a D24 texture on a GPU without Vulkan D24 (AMD) is a D32_SFLOAT_S8_UINT
            // image (texture_format), so its features are D32S8's. Asking D24 itself gave Support1 = 0 there,
            // although CreateCommittedResource(ALLOW_DEPTH_STENCIL) succeeds.
            if (vk_fmt == VK_FORMAT_D24_UNORM_S8_UINT && !state_.depth_usable(vk_fmt))
                vk_fmt = VK_FORMAT_D32_SFLOAT_S8_UINT;
            if (vk_fmt == VK_FORMAT_UNDEFINED || !state_.format_properties) {
                return d3d12_trace::result(method, DXGI_ERROR_UNSUPPORTED);
            }
            if (fs->Format == DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM) {
                // The host reports this display format as 2D, display, cast-within-bit-layout, back-buffer-cast and
                // video-processor output, and no shader reads. The layer has the 2D resource and the bit-compatible
                // view cast; it has no display scan-out or video processor, so it reports only what it provides.
                fs->Support1 = D3D12_FORMAT_SUPPORT1_TEXTURE2D | D3D12_FORMAT_SUPPORT1_CAST_WITHIN_BIT_LAYOUT;
                return S_OK;
            }
            VkFormatProperties props{};
            state_.format_properties(state_.info.physical, vk_fmt, &props);
            VkFormatFeatureFlags optimal = props.optimalTilingFeatures;
            VkFormatFeatureFlags buffer = props.bufferFeatures;
            if (fmt.rgb96) {
                // Textures are RGBA32 images that are sampled and copied, nothing else yet (no render target,
                // no unordered access: creation refuses them). Buffers hold the real 12-byte format; the PC
                // runtime (WARP) has no typed unordered access of these formats, in buffers either.
                optimal &= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
                VkFormatProperties narrow{};
                state_.format_properties(state_.info.physical, rgb96_named_format(vk_fmt), &narrow);
                buffer = narrow.bufferFeatures &
                         ~VkFormatFeatureFlags(VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT |
                                               VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT);
            }

            D3D12_FORMAT_SUPPORT1 s1 = D3D12_FORMAT_SUPPORT1_NONE;
            D3D12_FORMAT_SUPPORT2 s2 = D3D12_FORMAT_SUPPORT2_NONE;

            if (fs->Format == DXGI_FORMAT_A8_UNORM) {
                if (optimal & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) {
                    fs->Support1 = D3D12_FORMAT_SUPPORT1_TEXTURE2D | D3D12_FORMAT_SUPPORT1_TEXTURECUBE |
                                   D3D12_FORMAT_SUPPORT1_SHADER_LOAD;
                    if (optimal & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)
                        fs->Support1 |= D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE;
                    VkImageFormatProperties limits{};
                    const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                    if (state_.image_format_properties &&
                        state_.image_format_properties(state_.info.physical, vk_fmt, VK_IMAGE_TYPE_2D,
                                                       VK_IMAGE_TILING_OPTIMAL, usage, 0, &limits) == VK_SUCCESS &&
                        limits.maxMipLevels > 1)
                        fs->Support1 |= D3D12_FORMAT_SUPPORT1_MIP;
                }
                return d3d12_trace::result(method, S_OK);
            }

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
                    (image_properties.sampleCounts & d3d12_multisample_sample_counts)) {
                    s1 |= D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET |
                          D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE;
                    if ((optimal & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) && !fmt.typeless &&
                        fmt.aspects == VK_IMAGE_ASPECT_COLOR_BIT && leonida::gfx10::format_from_vulkan(vk_fmt) &&
                        state_.bindless && state_.bindless->supports_ms_class(DeviceState::numeric_class(vk_fmt)))
                        s1 |= D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD;
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
                // A typed UAV read is not a storage-image read: the translator reads the texture through its
                // sampled binding with the T#'s numeric type (rdna2spirv.py image_read_typed), and a typed buffer
                // through its address. So every typed UAV format can also be read.
                s2 |= D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE | D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD;
                // Image atomics go through the r32ui storage view, which only an R32_UINT texture gets
                // (StorageViews::r32ui). The translator has no image_atomic_swap, so no exchange.
                if (vk_fmt == VK_FORMAT_R32_UINT && (optimal & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT)) {
                    s2 |= D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_ADD | D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_BITWISE_OPS |
                          D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_COMPARE_STORE_OR_COMPARE_EXCHANGE |
                          D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_SIGNED_MIN_OR_MAX |
                          D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_UNSIGNED_MIN_OR_MAX;
                }
            }
            {
                // D3D12's per-format rules, as native D3D12 reports them (RX 9070 XT, AMD 26.11.2, and WARP):
                // the Vulkan feature bits above say what the image can do, not what D3D12 lets the format do.
                const Format& chosen = fmt.format != VK_FORMAT_UNDEFINED ? fmt : depth_fmt;
                const bool depth = (chosen.aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0;
                const bool block_compressed = vk_fmt >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && vk_fmt <= VK_FORMAT_BC7_SRGB_BLOCK;
                // Gathers read the sampled view like a sample does (OpImageGather); the host reports one with the other.
                if (s1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE) s1 |= D3D12_FORMAT_SUPPORT1_SHADER_GATHER;
                // A mip chain where the image can have one, with the usage it is created with.
                if (s1 & (D3D12_FORMAT_SUPPORT1_TEXTURE1D | D3D12_FORMAT_SUPPORT1_TEXTURE2D |
                          D3D12_FORMAT_SUPPORT1_TEXTURE3D | D3D12_FORMAT_SUPPORT1_TEXTURECUBE |
                          D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL)) {
                    const VkImageUsageFlags usage = depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                                          : VK_IMAGE_USAGE_SAMPLED_BIT;
                    VkImageFormatProperties limits{};
                    if (state_.image_format_properties &&
                        state_.image_format_properties(state_.info.physical, vk_fmt, VK_IMAGE_TYPE_2D,
                                                       VK_IMAGE_TILING_OPTIMAL, usage,
                                                       chosen.typeless ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0,
                                                       &limits) == VK_SUCCESS &&
                        limits.maxMipLevels > 1)
                        s1 |= D3D12_FORMAT_SUPPORT1_MIP;
                    // Depth formats multisample as depth-stencil targets (MULTISAMPLE_QUALITY_LEVELS already
                    // reports their counts); the colour-attachment check above never sees them.
                    if (depth && !chosen.typeless && (s1 & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL) &&
                        state_.image_format_properties &&
                        state_.image_format_properties(state_.info.physical, vk_fmt, VK_IMAGE_TYPE_2D,
                                                       VK_IMAGE_TILING_OPTIMAL,
                                                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 0,
                                                       &limits) == VK_SUCCESS &&
                        (limits.sampleCounts & d3d12_multisample_sample_counts))
                        s1 |= D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET;
                }
                // No resolve of integer formats (ResolveSubresource takes float, UNORM and SNORM only).
                if (DeviceState::numeric_class(vk_fmt) != 0) s1 &= ~D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE;
                // Block-compressed textures are 2D, 3D or cubes: 1D has no 4x4 blocks.
                if (block_compressed) s1 &= ~D3D12_FORMAT_SUPPORT1_TEXTURE1D;
                // Depth formats are 1D, 2D and cube resources whose shaders read them through a colour view
                // (R32_FLOAT, R24_UNORM_X8_TYPELESS, ...), not 3D and not shader resources themselves.
                if (depth)
                    s1 &= ~(D3D12_FORMAT_SUPPORT1_TEXTURE3D | D3D12_FORMAT_SUPPORT1_SHADER_LOAD |
                            D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE | D3D12_FORMAT_SUPPORT1_SHADER_GATHER |
                            D3D12_FORMAT_SUPPORT1_BUFFER | D3D12_FORMAT_SUPPORT1_IA_VERTEX_BUFFER);
                // A TYPELESS format is only storage: views give it a type, so it is a texture with mips and
                // nothing a view would do (no buffer, shader, render-target or UAV use of its own).
                if (chosen.typeless) {
                    s1 &= D3D12_FORMAT_SUPPORT1_TEXTURE1D | D3D12_FORMAT_SUPPORT1_TEXTURE2D |
                          D3D12_FORMAT_SUPPORT1_TEXTURE3D | D3D12_FORMAT_SUPPORT1_TEXTURECUBE |
                          D3D12_FORMAT_SUPPORT1_MIP;
                    s2 = D3D12_FORMAT_SUPPORT2_NONE;
                }
            }
            if (fs->Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP && state_.r9g9b9e5_uav_emulated) {
                s1 |= D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW;
                s2 |= D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE;
                VkFormatProperties r32{};
                state_.format_properties(state_.info.physical, VK_FORMAT_R32_UINT, &r32);
                if (r32.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT) {
                    s2 |= D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD;
                }
            }
            if (fs->Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP) {
                // The native packed format supports single-sample RTVs on some adapters.
                // Only that usage has been verified through the shim's render-target path.
                s1 &= ~(D3D12_FORMAT_SUPPORT1_BLENDABLE |
                        D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL | D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET |
                        D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE | D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD);
            }
            fs->Support1 = s1;
            fs->Support2 = s2;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS: {
            if (size != sizeof(D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS)) return d3d12_trace::result(method, E_INVALIDARG);
            auto* levels = static_cast<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS*>(data);
            levels->NumQualityLevels = 0;
            if (levels->Flags != D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE ||
                levels->SampleCount == 0 || levels->SampleCount > D3D12_MAX_MULTISAMPLE_SAMPLE_COUNT ||
                (levels->SampleCount & (levels->SampleCount - 1)) != 0)
                return d3d12_trace::result(method, E_INVALIDARG);
            const bool alpha_only = levels->Format == DXGI_FORMAT_A8_UNORM;
            if (alpha_only && levels->SampleCount != 1) return d3d12_trace::result(method, S_OK);
            if (levels->Format == DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM && levels->SampleCount != 1)
                return d3d12_trace::result(method, S_OK);  // single-sample 2D only; no quality levels
            Format format = texture_storage_format(levels->Format, false);
            // A typeless format that is only a texture as depth (R24G8_TYPELESS, R32G8X24_TYPELESS) has no colour
            // format; creation makes it a depth image, so ask the same.
            const bool depth_only_typeless = format.format == VK_FORMAT_UNDEFINED &&
                                             (texture_storage_format(levels->Format, true).aspects & VK_IMAGE_ASPECT_DEPTH_BIT);
            if ((format.aspects & VK_IMAGE_ASPECT_DEPTH_BIT) || depth_only_typeless) {
                D3D12_RESOURCE_DESC depth_desc{};
                depth_desc.Format = levels->Format;
                depth_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
                format = state_.texture_format(depth_desc);  // includes the D24-to-D32 fallback used by creation
            }
            if (format.format == VK_FORMAT_UNDEFINED || !state_.image_format_properties)
                return d3d12_trace::result(method, DXGI_ERROR_UNSUPPORTED);
            if (format.rgb96) {  // no 96-bit render targets yet, so no multisampled ones
                levels->NumQualityLevels = levels->SampleCount == 1 ? 1 : 0;
                return d3d12_trace::result(method, S_OK);
            }

            // Vulkan exposes one standard sample pattern per supported count, so each supported count maps to
            // one D3D12 quality level (quality index 0). Query the matching colour or depth/stencil attachment
            // usage, plus the transfer and sampled usages used by the resource implementation.
            VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                      VK_IMAGE_USAGE_SAMPLED_BIT;
            if (format.aspects & VK_IMAGE_ASPECT_COLOR_BIT) {
                if (!alpha_only) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            } else if (format.aspects & VK_IMAGE_ASPECT_DEPTH_BIT)
                usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            else
                return d3d12_trace::result(method, DXGI_ERROR_UNSUPPORTED);
            const VkImageCreateFlags flags = format.typeless ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
            VkImageFormatProperties image_properties{};
            const VkResult result = state_.image_format_properties(state_.info.physical, format.format,
                                                                   VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                                   usage, flags, &image_properties);
            const VkSampleCountFlagBits samples = VkSampleCountFlagBits(levels->SampleCount);
            if (result == VK_SUCCESS && (image_properties.sampleCounts & samples))
                levels->NumQualityLevels = 1;
            return d3d12_trace::result(method, S_OK);
        }
        case D3D12_FEATURE_SHADER_MODEL: {
            if (size != sizeof(D3D12_FEATURE_DATA_SHADER_MODEL)) return E_INVALIDARG;
            auto* model = static_cast<D3D12_FEATURE_DATA_SHADER_MODEL*>(data);
            // Measured against the host PC runtime: a request above the supported model is clamped down to it, 5.1
            // is echoed because it is D3D12's own baseline (WARP and the RTX 5070 both answer it with S_OK), and a
            // value below it is refused with E_INVALIDARG (0x5f crashed the host, so the test does not probe it).
            // Refuse the same way instead of echoing an unrepresentable value back.
            if (model->HighestShaderModel < D3D_SHADER_MODEL_5_1) return E_INVALIDARG;
            if (model->HighestShaderModel > kHighestShaderModel) model->HighestShaderModel = kHighestShaderModel;
            return S_OK;
        }
        case D3D12_FEATURE_ROOT_SIGNATURE: {
            if (size != sizeof(D3D12_FEATURE_DATA_ROOT_SIGNATURE)) return E_INVALIDARG;
            auto* root = static_cast<D3D12_FEATURE_DATA_ROOT_SIGNATURE*>(data);
            if (root->HighestVersion < D3D_ROOT_SIGNATURE_VERSION_1_0) return E_INVALIDARG;
            if (root->HighestVersion > kHighestRootSignatureVersion)
                root->HighestVersion = kHighestRootSignatureVersion;
            return S_OK;
        }
        default:
            // Not answered yet. The log above lists every query; answers are added as the trace asks for them.
            return d3d12_trace::result(method, E_INVALIDARG);
        }
    }

private:
    IUnknown* identity() { return outer_ ? outer_ : static_cast<ID3D12Device*>(this); }

    static constexpr D3D12_RESOURCE_ALLOCATION_INFO kInvalidAllocation{UINT64_MAX,
                                                                       D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT};
    // Saturates at the last multiple of `to` instead of wrapping, as the PC runtime's totals do.
    static UINT64 align_up(UINT64 value, UINT64 to) {
        return value > UINT64_MAX - (to - 1) ? UINT64_MAX / to * to : (value + to - 1) / to * to;
    }

    // Lays `count` resources out one after another; `each` (optional) gets every resource's offset, size and
    // alignment. As on the PC runtime, the total is rounded up to the largest alignment (1 for no resources) and
    // saturates near 2^64 rather than wrapping.
    template<class Desc, class Get>
    D3D12_RESOURCE_ALLOCATION_INFO place(UINT count, const Desc* descs, Get get,
                                         D3D12_RESOURCE_ALLOCATION_INFO1* each) {
        D3D12_RESOURCE_ALLOCATION_INFO total{0, 1};
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
            total = {one.SizeInBytes > UINT64_MAX - offset ? UINT64_MAX : offset + one.SizeInBytes,
                     std::max(total.Alignment, one.Alignment)};
        }
        return failed ? kInvalidAllocation
                      : D3D12_RESOURCE_ALLOCATION_INFO{align_up(total.SizeInBytes, total.Alignment), total.Alignment};
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
        const bool one_d = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D;
        if (!volume && !one_d && desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
            if (desc.Dimension != D3D12_RESOURCE_DIMENSION_UNKNOWN)
                log_once("GetResourceAllocationInfo dimension not supported:", desc.Dimension);
            return kInvalidAllocation;
        }
        if (one_d && desc.Height != 1) return kInvalidAllocation;
        const UINT samples = desc.SampleDesc.Count, depth = volume ? desc.DepthOrArraySize : 1;
        const UINT64 max_extent =
            volume ? D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION : D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
        const UINT max_layers =
            volume ? D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION : D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION;
        if (!desc.Width || !desc.Height || !desc.DepthOrArraySize || desc.Width > max_extent ||
            desc.Height > max_extent || desc.DepthOrArraySize > max_layers || !samples || samples > D3D12_MAX_MULTISAMPLE_SAMPLE_COUNT ||
            (samples & (samples - 1)) || desc.SampleDesc.Quality != 0 ||
            desc.Layout == D3D12_TEXTURE_LAYOUT_ROW_MAJOR)
            return kInvalidAllocation;
        const UINT16 full = full_mips(desc.Width, desc.Height, depth);
        D3D12_RESOURCE_DESC resolved = desc;
        if (!resolved.MipLevels) resolved.MipLevels = full;
        if (resolved.MipLevels > full) return kInvalidAllocation;
        if (samples > 1 && (volume || one_d || resolved.MipLevels != 1 ||
                            (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)))
            return kInvalidAllocation;
        // The PC runtime: simultaneous access only single-sampled and without depth-stencil, and no volume depth.
        const bool depth_stencil = desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        if (((desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS) && (depth_stencil || samples > 1)) ||
            (volume && depth_stencil))
            return kInvalidAllocation;
        if (desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN)
            log_once("GetResourceAllocationInfo layout as UNKNOWN:", desc.Layout);

        const Format format = state_.texture_format(desc);
        if (!format.format) {
            log_once("GetResourceAllocationInfo format not supported yet:", desc.Format);
            return kInvalidAllocation;
        }
        VkMemoryRequirements need{};
        const TextureCreateInfo create(resolved, format, state_.color_attachment_usable(format.format));
        if (!state_.memory_requirements(create.info, need)) return kInvalidAllocation;

        // D3D12's alignment: 64 KiB, or 4 MiB with MSAA, unless the description asks for a small one (4 KiB, 64 KiB
        // with MSAA). 4 KiB needs the top mip of one array slice to fit in 16 tiles of 4 KiB and the resource not to
        // be a render target or depth buffer; mips and array slices don't count. If it doesn't qualify, the PC
        // runtime calls the description invalid rather than answering with 64 KiB. The MSAA 64 KiB alignment has no
        // size limit there. Both measured in d3d12_vulkan_device_test.
        UINT64 alignment = 0;
        if (samples > 1) {
            if (desc.Alignment && desc.Alignment != align_msaa_64k && desc.Alignment != align_msaa_4m)
                return kInvalidAllocation;
            alignment = desc.Alignment ? desc.Alignment : align_msaa_4m;
        } else {
            if (desc.Alignment && desc.Alignment != align_4k && desc.Alignment != align_64k) return kInvalidAllocation;
            if (desc.Alignment == align_4k && ((desc.Flags & attachment) || !small_tiles_fit(desc, format, need.size)))
                return kInvalidAllocation;
            alignment = desc.Alignment ? desc.Alignment : align_64k;
        }
        return {align_up(need.size, alignment), alignment};
    }

    // Whether 16 tiles of 4 KiB (leonida::small_tile_shape; 4096 bytes of a row for 1D) cover the top mip of one
    // array slice, the PC runtime's test for D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT.
    static bool small_tiles_fit(const D3D12_RESOURCE_DESC& desc, const Format& format, VkDeviceSize vulkan_size) {
        const leonida::BlockFormat block = leonida::block_format(format.format, VK_IMAGE_ASPECT_COLOR_BIT);
        const bool volume = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        leonida::TileShape shape = leonida::small_tile_shape(volume, block.bytes);
        if (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && shape.width) shape = {4096 / block.bytes, 1, 1};
        // shortcut: untileable element sizes (3- and 12-byte formats) keep the whole-resource 64 KiB test; measure
        // them on the PC runtime if a game asks for one.
        if (!shape.width) return vulkan_size <= D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        const auto tiles = [](UINT64 texels, UINT64 per_block, UINT64 per_tile) {
            return ((texels + per_block - 1) / per_block + per_tile - 1) / per_tile;
        };
        return tiles(desc.Width, block.block, shape.width) * tiles(desc.Height, block.block, shape.height) *
                   tiles(volume ? desc.DepthOrArraySize : 1, 1, shape.depth) <= 16;
    }

    // Writes the view's T# into the slot and makes sure set 0 and the hash table know it. A view that cannot be
    // shown leaves the slot a null descriptor.
    void write_image_descriptor(const Resource& resource, const TextureView& view,
                                D3D12_CPU_DESCRIPTOR_HANDLE destination, bool storage) {
        leonida::gfx10::Image image{};
        if (!image_descriptor(resource, view, image)) return;
        const leonida::gfx10::ImageWords words = leonida::gfx10::encode(image);
        const bool registered = state_.bind_image(words, image, resource.image(), resource.format(),
                                                  resource.aspects(), storage, false,
                                                  resource.r9g9b9e5_as_r32_uint());
        if (leonida::gfx10::multisampled(image.type) && !registered) return;
        std::memcpy(reinterpret_cast<void*>(destination.ptr), words.data(), sizeof(words));
        state_.bind_image(words, image, resource.image(), resource.format(), resource.aspects(), storage,
                          resource.has_view_format_list(), resource.r9g9b9e5_as_r32_uint());
    }

    // Shared by CreateCommittedResource and CreateCommittedResource3. A castable-format list is only meaningful on
    // textures; the layer reports RelaxedFormatCastingSupported, so each format is mapped and handed to Vulkan as a
    // VkImageFormatListCreateInfo. A nonzero count with a null list is no list, as on the PC runtime.
    HRESULT create_committed(const D3D12_HEAP_PROPERTIES* heap, D3D12_HEAP_FLAGS heap_flags,
                             const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES initial_state,
                             UINT32 castable_count, const DXGI_FORMAT* castable, REFIID riid, void** out) {
        if (out) *out = nullptr;
        if (!heap || !desc) return E_INVALIDARG;
        if (castable_count && castable && desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            log_once("castable formats on a buffer are invalid; count", int(castable_count));
            return E_INVALIDARG;
        }
        auto* resource = new Resource(state_, *desc, *heap, heap_flags);
        resource->attach_device(identity());
        HRESULT hr = E_NOTIMPL;
        if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            hr = desc->SampleDesc.Count == 1 && desc->SampleDesc.Quality == 0
                     ? resource->create_buffer(initial_state)
                     : E_INVALIDARG;
        } else if (desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D &&
                   desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                   desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
            log_once("committed resource dimension not supported:", desc->Dimension);
        } else if (desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && desc->Height != 1) {
            log_once("1D texture height must be 1:", int(desc->Height));
            hr = E_INVALIDARG;
        } else if (desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D &&
                   (desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))) {
            log_once("1D texture cannot be an attachment, flags:", int(desc->Flags));
            hr = E_INVALIDARG;
        } else if (desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D &&
                   (desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) {
            log_once("volume depth-stencil resources not supported, flags:", int(desc->Flags));
        } else if (heap->Type == D3D12_HEAP_TYPE_UPLOAD || heap->Type == D3D12_HEAP_TYPE_READBACK) {
            hr = E_INVALIDARG;  // D3D12 places textures on default heaps unless they are row-major
        } else if (const UINT samples = desc->SampleDesc.Count;
                   !samples || samples > D3D12_MAX_MULTISAMPLE_SAMPLE_COUNT || desc->SampleDesc.Quality != 0 ||
                   (samples > 1 && ((samples & (samples - 1)) || desc->MipLevels != 1 ||
                                    desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
                                    !(desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                                                     D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))))) {
            hr = E_INVALIDARG;  // D3D12: nonzero power-of-two sample count up to 32; MSAA needs a single-mip 2D attachment
        } else {
            hr = S_OK;
            const Format format = state_.texture_format(*desc);
            if (format.format == VK_FORMAT_UNDEFINED) {
                log_once("texture format not supported yet:", desc->Format);
                hr = E_INVALIDARG;
            } else {
                std::vector<VkFormat> cast_formats;
                hr = castable_view_formats(format, castable_count, castable, cast_formats);
                if (SUCCEEDED(hr)) hr = resource->create_texture(format, cast_formats);
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

    // GetAdapterLuid's answer, and how queues read the clocks: calibrated timestamps when the caller enabled the
    // extension and the GPU calibrates its DEVICE domain against QueryPerformanceCounter, else the queue family's
    // timestamp queries.
    void query_luid_and_clock(PFN_vkGetDeviceProcAddr get_device) {
        const VulkanDeviceInfo& info = state_.info;
        const auto properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceProperties2"));
        if (properties2) {
            VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
            properties2(info.physical, &properties);
            static_assert(sizeof(LUID) == VK_LUID_SIZE);
            if (id.deviceLUIDValid) std::memcpy(&state_.luid, id.deviceLUID, sizeof(LUID));
        }
        if (!state_.luid.LowPart && !state_.luid.HighPart) log_once("the Vulkan driver reports no device LUID;", 0);

        state_.clock.get_device_proc = get_device;
        const auto families_of = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
        if (families_of) {
            std::uint32_t count = 0;
            families_of(info.physical, &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            families_of(info.physical, &count, families.data());
            if (info.queue_family < count)
                state_.clock.timestamp_valid_bits = families[info.queue_family].timestampValidBits;
        }
        if (!info.calibrated_timestamps) return;
        // The KHR names first; the EXT ones are aliases with the same signatures.
        auto domains_of = reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR"));
        auto calibrated = reinterpret_cast<PFN_vkGetCalibratedTimestampsKHR>(
            get_device(info.device, "vkGetCalibratedTimestampsKHR"));
        if (!domains_of || !calibrated) {
            domains_of = reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR>(
                info.get_instance_proc(info.instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"));
            calibrated = reinterpret_cast<PFN_vkGetCalibratedTimestampsKHR>(
                get_device(info.device, "vkGetCalibratedTimestampsEXT"));
        }
        if (!domains_of || !calibrated) return;
        std::uint32_t count = 0;
        domains_of(info.physical, &count, nullptr);
        std::vector<VkTimeDomainKHR> domains(count);
        domains_of(info.physical, &count, domains.data());
        const auto has = [&](VkTimeDomainKHR domain) {
            return std::find(domains.begin(), domains.end(), domain) != domains.end();
        };
        if (has(VK_TIME_DOMAIN_DEVICE_KHR) && has(VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR))
            state_.clock.calibrated = calibrated;
        else
            log_once("calibrated timestamps lack the DEVICE or QueryPerformanceCounter domain; domains:", int(count));
    }

    // D3D12's layout rules for a reserved resource, then a committed resource on a default heap under it.
    HRESULT create_reserved(const D3D12_RESOURCE_DESC* desc, REFIID riid, void** out) {
        if (out) *out = nullptr;
        if (!desc) return E_INVALIDARG;
        // The Xbox tiled layout word. 0x101 is outside D3D12_TEXTURE_LAYOUT's enumerators, so it can't be a
        // constexpr enum value (clang rejects the cast in a constant expression); compare it as an integer.
        constexpr UINT xbox_tiled = 0x101;
        D3D12_RESOURCE_DESC backing = *desc;
        if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            if (desc->Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR) return E_INVALIDARG;
        } else {
            if (static_cast<UINT>(desc->Layout) == xbox_tiled) {
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
        if (format.rgb96) return false;  // the tile shape of a 12-byte texel is not measured
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

    // The layer reports no protected-session support, so a session is refused, as D3D12 refuses one on a device
    // without D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_SUPPORT. `variant` is the method's number.
    static HRESULT refuse_protected_session(int variant) {
        log_once("protected resource session refused (not supported) by CreateCommittedResource", variant);
        return E_INVALIDARG;
    }

    // D3D12_RESOURCE_DESC1 is D3D12_RESOURCE_DESC plus SamplerFeedbackMipRegion.
    static D3D12_RESOURCE_DESC resource_desc(const D3D12_RESOURCE_DESC1& desc) {
        return {desc.Dimension, desc.Alignment, desc.Width,      desc.Height, desc.DepthOrArraySize,
                desc.MipLevels, desc.Format,    desc.SampleDesc, desc.Layout, desc.Flags};
    }

    // A sampler-feedback mip region on a texture, which needs a DXGI_FORMAT_SAMPLER_FEEDBACK_* resource the layer
    // does not have. The PC runtime ignores the region on a buffer.
    static bool texture_mip_region(const D3D12_RESOURCE_DESC1& desc) {
        const D3D12_MIP_REGION& region = desc.SamplerFeedbackMipRegion;
        return desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER && (region.Width || region.Height || region.Depth);
    }

    std::mutex queue_mutex_;
    std::vector<bool> queue_busy_;
    DeviceState state_;
    std::atomic<bool> removed_{false};
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

const RootArgumentLayout* root_signature_layout(ID3D12RootSignature* signature) {
    const RootSignature* s = RootSignature::from<RootSignature>(signature);
    return s ? &s->layout() : nullptr;
}

std::uintptr_t compute_pipeline_id(ID3D12PipelineState* pipeline) {
    const PipelineState* p = PipelineState::from<PipelineState>(pipeline);
    return p && p->compute() ? p->id() : 0;
}

std::uintptr_t graphics_pipeline_id(ID3D12PipelineState* pipeline) {
    const PipelineState* p = PipelineState::from<PipelineState>(pipeline);
    return p && p->graphics() ? p->id() : 0;
}

VkImage vulkan_image(ID3D12Resource* resource) {
    const Resource* r = Resource::from<Resource>(resource);
    return r ? r->image() : VK_NULL_HANDLE;
}

VkMemoryPropertyFlags vulkan_memory_flags(ID3D12Resource* resource) {
    const Resource* r = Resource::from<Resource>(resource);
    return r ? r->memory_flags() : 0;
}

VkBuffer vulkan_buffer(ID3D12Resource* resource) {
    const Resource* r = Resource::from<Resource>(resource);
    return r ? r->buffer() : VK_NULL_HANDLE;
}

bool device_supports_ray_tracing(ID3D12Device* device) {
    if (!device) return false;
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    if (auto* d = Device::from<Device>(real ? real : device)) return d->context().ray_tracing_supported();
    return false;
}

VkAccelerationStructureKHR vulkan_acceleration_structure(ID3D12Device* device, D3D12_GPU_VIRTUAL_ADDRESS va) {
    if (!device || !va) return VK_NULL_HANDLE;
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    if (auto* d = Device::from<Device>(real ? real : device))
        return d->context().resolve_acceleration_structure(va);
    return VK_NULL_HANDLE;
}

bool color_resolve_format_matches(DXGI_FORMAT format, ID3D12Resource* dst, ID3D12Resource* src) {
    if (!dst || !src) return false;
    const Format requested = vulkan_format(format, false);
    if (requested.typeless || requested.rgb96 || requested.aspects != VK_IMAGE_ASPECT_COLOR_BIT) return false;
    if (format == DXGI_FORMAT_R32G32B32A32_UINT || format == DXGI_FORMAT_R8G8B8A8_UINT ||
        format == DXGI_FORMAT_R32_UINT || format == DXGI_FORMAT_R16G16B16A16_UINT ||
        format == DXGI_FORMAT_R16G16B16A16_SINT || format == DXGI_FORMAT_R32G32_UINT ||
        format == DXGI_FORMAT_R32G32_SINT || format == DXGI_FORMAT_R10G10B10A2_UINT ||
        format == DXGI_FORMAT_R16G16_UINT || format == DXGI_FORMAT_R16G16_SINT ||
        format == DXGI_FORMAT_R8G8_UINT || format == DXGI_FORMAT_R8G8_SINT)
        return false;
    const auto matches = [&](ID3D12Resource* resource) {
        const D3D12_RESOURCE_DESC desc = resource->GetDesc();
        return vulkan_format(desc.Format, desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL).format ==
               requested.format;
    };
    return matches(dst) && matches(src);
}

bool depth_stencil_resolve_supported(DXGI_FORMAT format, ID3D12Resource* dst, UINT dst_subresource, UINT x, UINT y,
                                     ID3D12Resource* src, UINT src_subresource, const D3D12_RECT* rect,
                                     D3D12_RESOLVE_MODE mode) {
    const Resource* d = Resource::from<Resource>(dst);
    const Resource* s = Resource::from<Resource>(src);
    if (!d || !s) return false;
    // The format names the depth format, or one plane by its shader-resource format. Measured on the PC runtime
    // (RTX 5070 and WARP): Close fails for AVERAGE with the depth format or a stencil plane; the depth plane's
    // own format (R32_FLOAT, R24_UNORM_X8_TYPELESS) takes AVERAGE as well as MIN and MAX.
    DXGI_FORMAT family = format;
    int named_plane = -1;  // -1: any plane of the depth format
    switch (format) {
    case DXGI_FORMAT_R32_FLOAT: family = DXGI_FORMAT_D32_FLOAT, named_plane = 0; break;
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: family = DXGI_FORMAT_D24_UNORM_S8_UINT, named_plane = 0; break;
    case DXGI_FORMAT_X24_TYPELESS_G8_UINT: family = DXGI_FORMAT_D24_UNORM_S8_UINT, named_plane = 1; break;
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: family = DXGI_FORMAT_D32_FLOAT_S8X24_UINT, named_plane = 0; break;
    case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT: family = DXGI_FORMAT_D32_FLOAT_S8X24_UINT, named_plane = 1; break;
    default: break;
    }
    const Format requested = vulkan_format(family, true);
    if (requested.typeless || !(requested.aspects & VK_IMAGE_ASPECT_DEPTH_BIT)) return false;
    // The plane a subresource index names (0 depth, 1 stencil), or -1.
    const auto plane = [&](const Resource* r, UINT subresource) {
        const D3D12_RESOURCE_DESC& desc = r->desc();
        if (!(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) ||
            vulkan_format(desc.Format, true).format != requested.format)
            return -1;
        const UINT planes = (requested.aspects & VK_IMAGE_ASPECT_STENCIL_BIT) ? 2 : 1;
        const UINT per_plane = UINT(desc.MipLevels) * desc.DepthOrArraySize;
        return per_plane && subresource / per_plane < planes ? int(subresource / per_plane) : -1;
    };
    const int resolved = plane(d, dst_subresource);
    if (resolved < 0 || resolved != plane(s, src_subresource) || (named_plane >= 0 && resolved != named_plane) ||
        s->desc().SampleDesc.Count <= 1 || d->desc().SampleDesc.Count != 1)
        return false;
    VkResolveModeFlagBits wanted = VK_RESOLVE_MODE_MIN_BIT;
    if (mode == D3D12_RESOLVE_MODE_MAX) wanted = VK_RESOLVE_MODE_MAX_BIT;
    else if (mode == D3D12_RESOLVE_MODE_AVERAGE && named_plane == 0) wanted = VK_RESOLVE_MODE_AVERAGE_BIT;
    else if (mode != D3D12_RESOLVE_MODE_MIN) return false;
    // The rendering scope that replays it resolves pixels in place (vulkan_replay.h).
    if (rect ? (LONG(x) != rect->left || LONG(y) != rect->top) : (x || y)) return false;
    const DeviceState& state = d->device_state();
    return (resolved ? state.stencil_resolve_modes : state.depth_resolve_modes) & wanted;
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

bool device_upload_in_vram(ID3D12Device* device) {
    if (!device) return false;
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    if (!real) real = device;
    if (auto* d = Device::from<Device>(real)) return d->upload_in_vram();
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

WindowPresenter::WindowPresenter(IUnknown* device, std::unique_ptr<leonida::VulkanPresenter> presenter)
    : device_(device), presenter_(std::move(presenter)) {
    device_->AddRef();
}

WindowPresenter::~WindowPresenter() {
    presenter_.reset();  // before the device it runs on
    device_->Release();
}

namespace {
HRESULT present_result(VkResult result) {
    if (result == VK_SUCCESS) return S_OK;
    std::fprintf(stderr, "[rt] d3d12: present failed (VkResult %d)\n", int(result));
    if (result == VK_ERROR_DEVICE_LOST) return DXGI_ERROR_DEVICE_REMOVED;
    if (result == VK_ERROR_FORMAT_NOT_SUPPORTED) return DXGI_ERROR_UNSUPPORTED;
    return E_FAIL;
}
}  // namespace

HRESULT WindowPresenter::present(ID3D12Resource* back_buffer, UINT sync_interval, const D3D12_RECT* src_rect) {
    if (!back_buffer) return E_INVALIDARG;
    const leonida::PresentRect rect = src_rect ? leonida::PresentRect{src_rect->left, src_rect->top, src_rect->right,
                                                                       src_rect->bottom}
                                               : leonida::PresentRect{};
    return present_result(presenter_->present(resource_id(back_buffer), sync_interval > 0, src_rect ? &rect : nullptr));
}

HRESULT WindowPresenter::present_blank(UINT sync_interval) {
    return present_result(presenter_->present_blank(sync_interval > 0));
}

bool WindowPresenter::enable_capture() { return presenter_->enable_capture(); }

void WindowPresenter::set_abort(std::function<bool()> abort) { presenter_->set_abort(std::move(abort)); }

bool WindowPresenter::read_back(std::vector<std::uint8_t>& pixels, VkExtent2D* extent, VkFormat* format) {
    return presenter_->read_back(pixels, extent, format);
}

VkExtent2D WindowPresenter::extent() const { return presenter_->extent(); }
std::uint64_t WindowPresenter::presented() const { return presenter_->presented(); }

std::unique_ptr<WindowPresenter> create_window_presenter(ID3D12Device* device, HWND window) {
    Device* d = Device::from<Device>(device);
    if (!d || !window) return nullptr;
    const VulkanDeviceInfo& info = d->state().info;
    if (!info.presentation) {
        std::fprintf(stderr, "[rt] d3d12: the Vulkan device was created without presentation support\n");
        return nullptr;
    }
    const auto create_surface = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
        info.get_instance_proc(info.instance, "vkCreateWin32SurfaceKHR"));
    if (!create_surface) return nullptr;
    VkWin32SurfaceCreateInfoKHR surface_info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    surface_info.hinstance = GetModuleHandleW(nullptr);
    surface_info.hwnd = window;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (create_surface(info.instance, &surface_info, nullptr, &surface) != VK_SUCCESS) return nullptr;
    auto presenter = std::make_unique<leonida::VulkanPresenter>(*d->state().context, info.get_instance_proc,
                                                                info.instance, info.physical, info.queue,
                                                                info.queue_family, surface);  // owns the surface
    if (!presenter->ok()) {
        std::fprintf(stderr, "[rt] d3d12: cannot present to this window from the device's queue family\n");
        return nullptr;
    }
    return std::make_unique<WindowPresenter>(d->holder(), std::move(presenter));
}

HRESULT register_samplers(ID3D12Device* device, std::span<const leonida::gfx10::SamplerWords> samplers) {
    if (!device) return E_INVALIDARG;
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    Device* d = Device::from<Device>(real ? real : device);
    if (!d) return E_INVALIDARG;
    bool all = true;
    for (const leonida::gfx10::SamplerWords& words : samplers) all = d->bind_sampler(words) && all;
    return all ? S_OK : E_FAIL;
}

HRESULT create_dxgi_factory(ID3D12Device* device, REFIID riid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    if (!device) return E_INVALIDARG;
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    Device* d = Device::from<Device>(real ? real : device);
    if (!d) return E_INVALIDARG;
    const auto& i = d->info();
    const auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(i.get_instance_proc(i.instance, "vkGetDeviceProcAddr"));
    if (!gdpa) return E_FAIL;
    return rt::dxgi::create_factory(d->context(), i.get_instance_proc, i.instance, i.physical, gdpa, i.queue,
                                    i.queue_family, riid, out);
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
    const auto enum_inst_exts = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
        get(nullptr, "vkEnumerateInstanceExtensionProperties"));
    std::vector<const char*> inst_exts;
    if (enum_inst_exts) {
        std::uint32_t ext_cnt = 0;
        enum_inst_exts(nullptr, &ext_cnt, nullptr);
        std::vector<VkExtensionProperties> available(ext_cnt);
        enum_inst_exts(nullptr, &ext_cnt, available.data());
        bool has_surface = false, has_win32 = false;
        for (const auto& e : available) {
            if (!std::strcmp(e.extensionName, VK_KHR_SURFACE_EXTENSION_NAME)) has_surface = true;
            if (!std::strcmp(e.extensionName, VK_KHR_WIN32_SURFACE_EXTENSION_NAME)) has_win32 = true;
        }
        if (has_surface && has_win32) {
            inst_exts.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
            inst_exts.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
        }
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "gta6rt";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    const char* layer = "VK_LAYER_KHRONOS_validation";
    const VkValidationFeatureEnableEXT sync_validation = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT validation_features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
    validation_features.enabledValidationFeatureCount = 1;
    validation_features.pEnabledValidationFeatures = &sync_validation;
    char validation[8] = {};
    size_t length = 0;
    if (getenv_s(&length, validation, "LEONIDA_VK_VALIDATION") == 0 && length > 1 && validation[0] != '0') {
        instance_info.pNext = &validation_features;
        instance_info.enabledLayerCount = 1;
        instance_info.ppEnabledLayerNames = &layer;
    }
    // VK_EXT_debug_utils when the loader has it, for the game's debug markers (PIX events) in RenderDoc and the like.
    const char* debug_utils = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    bool has_debug_utils = false;
    if (const auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
            get(nullptr, "vkEnumerateInstanceExtensionProperties"))) {
        std::uint32_t extension_count = 0;
        enumerate(nullptr, &extension_count, nullptr);
        std::vector<VkExtensionProperties> extensions(extension_count);
        enumerate(nullptr, &extension_count, extensions.data());
        for (const auto& e : extensions) has_debug_utils = has_debug_utils || std::strcmp(e.extensionName, debug_utils) == 0;
    }
    // One list: the surface extensions above (for DXGI presentation) and debug utils both go on the instance.
    const bool instance_has_surface = !inst_exts.empty();
    if (has_debug_utils) inst_exts.push_back(debug_utils);
    instance_info.enabledExtensionCount = static_cast<std::uint32_t>(inst_exts.size());
    instance_info.ppEnabledExtensionNames = inst_exts.empty() ? nullptr : inst_exts.data();
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
    info.debug_utils = has_debug_utils;
    VkPhysicalDeviceProperties chosen{};
    bool chosen_bc = false;
    VkPhysicalDeviceFeatures chosen_features{};
    VkPhysicalDeviceVulkan12Features chosen12{};
    for (VkPhysicalDevice candidate : physicals) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(candidate, &props);
        if (props.apiVersion < VK_API_VERSION_1_3 || !leonida::vulkan_device_wanted(props.deviceName)) continue;
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
        vkGetPhysicalDeviceFeatures2(candidate, &features);
        const bool bc = features.features.textureCompressionBC;
        if (!f13.dynamicRendering || !f13.synchronization2 || !f12.timelineSemaphore ||
            !f12.separateDepthStencilLayouts || !f12.bufferDeviceAddress)
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
        return fail(leonida::vulkan_device_filter().empty()
                        ? "no Vulkan 1.3 GPU with dynamicRendering, synchronization2, timelineSemaphore and "
                          "separateDepthStencilLayouts"
                        : "no GPU named by LEONIDA_VK_DEVICE has Vulkan 1.3 with dynamicRendering, synchronization2, "
                          "timelineSemaphore and separateDepthStencilLayouts");

    // VK_EXT_conditional_rendering for SetPredication, VK_KHR_swapchain for DXGI presentation, and optional
    // VK_KHR_acceleration_structure + VK_KHR_ray_query for hardware BVH builds (BLAS + TLAS).
    VkPhysicalDeviceConditionalRenderingFeaturesEXT conditional{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
    bool has_swapchain = false;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rq_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    {
        std::uint32_t extension_count = 0;
        vkEnumerateDeviceExtensionProperties(info.physical, nullptr, &extension_count, nullptr);
        std::vector<VkExtensionProperties> extensions(extension_count);
        vkEnumerateDeviceExtensionProperties(info.physical, nullptr, &extension_count, extensions.data());
        bool has_as = false, has_dho = false, has_rq = false;
        for (const auto& e : extensions) {
            if (!std::strcmp(e.extensionName, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME)) {
                VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &conditional};
                vkGetPhysicalDeviceFeatures2(info.physical, &features);
                info.conditional_rendering = conditional.conditionalRendering == VK_TRUE;
            } else if (!std::strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
                has_swapchain = true;
            } else if (!std::strcmp(e.extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) {
                has_as = true;
            } else if (!std::strcmp(e.extensionName, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) {
                has_dho = true;
            } else if (!std::strcmp(e.extensionName, VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
                has_rq = true;
            }
        }
        if (has_as && has_dho && has_rq && chosen12.bufferDeviceAddress) {
            as_features.pNext = &rq_features;
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &as_features};
            vkGetPhysicalDeviceFeatures2(info.physical, &features);
            info.ray_tracing = as_features.accelerationStructure == VK_TRUE && rq_features.rayQuery == VK_TRUE;
        }
        // The game's translated RT shaders read XDXR BVH4 records, which a VkAccelerationStructureKHR is not
        // (docs/d3d12_vulkan.md 4.8): build those, unless LEONIDA_RT_AS=khr asks for #743's KHR builds instead.
        char rt_as[8] = {};
        std::size_t rt_as_length = 0;
        const bool khr = getenv_s(&rt_as_length, rt_as, "LEONIDA_RT_AS") == 0 && rt_as_length > 1 &&
                         std::strcmp(rt_as, "khr") == 0;
        if (!khr && chosen12.bufferDeviceAddress) {
            info.xdxr_acceleration_structures = true;
            info.ray_tracing = false;  // no KHR extensions: AS resources keep their console VAs
        }
        if (!info.conditional_rendering)
            std::fprintf(stderr, "[rt] d3d12: the GPU has no VK_EXT_conditional_rendering: SetPredication is unsupported\n");
    }
    VkPhysicalDeviceConditionalRenderingFeaturesEXT enable_conditional{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
    enable_conditional.conditionalRendering = VK_TRUE;
    VkPhysicalDeviceRayQueryFeaturesKHR enable_rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    enable_rq.rayQuery = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR enable_as{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    enable_as.accelerationStructure = VK_TRUE;

    void* ext_chain = nullptr;
    if (info.ray_tracing) {
        enable_rq.pNext = ext_chain;
        enable_as.pNext = &enable_rq;
        ext_chain = &enable_as;
    }
    if (info.conditional_rendering) {
        enable_conditional.pNext = ext_chain;
        ext_chain = &enable_conditional;
    }

    VkPhysicalDeviceVulkan13Features enable13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, ext_chain};
    enable13.dynamicRendering = VK_TRUE;
    enable13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &enable13};
    enable12.timelineSemaphore = VK_TRUE;
    enable12.separateDepthStencilLayouts = VK_TRUE;
    info.buffer_device_address = chosen12.bufferDeviceAddress == VK_TRUE;
    enable12.bufferDeviceAddress = info.buffer_device_address ? VK_TRUE : VK_FALSE;
    info.storage_buffer_8bit = chosen12.storageBuffer8BitAccess == VK_TRUE;
    enable12.storageBuffer8BitAccess = info.storage_buffer_8bit;
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
    enable12.bufferDeviceAddress = VK_TRUE;
    info.buffer_device_address = true;
    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(info.physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(info.physical, &family_count, families.data());
    const std::uint32_t command_queue_count = std::min(16u, families[info.queue_family].queueCount);
    std::vector<float> priorities(command_queue_count, 1.0f);
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = info.queue_family;
    queue_info.queueCount = command_queue_count;
    queue_info.pQueuePriorities = priorities.data();
    VkPhysicalDeviceFeatures enable{};
    info.texture_compression_bc = chosen_bc;
    enable.textureCompressionBC = chosen_bc ? VK_TRUE : VK_FALSE;  // the game's textures are BC1-7
    enable.shaderSampledImageArrayDynamicIndexing = info.bindless;
    enable.shaderStorageImageArrayDynamicIndexing = info.bindless;
    enable.samplerAnisotropy = info.sampler_anisotropy;
    enable.occlusionQueryPrecise = chosen_features.occlusionQueryPrecise;  // exact D3D12 occlusion counts
    enable.pipelineStatisticsQuery = chosen_features.pipelineStatisticsQuery;
    info.occlusion_query_precise = chosen_features.occlusionQueryPrecise == VK_TRUE;
    info.pipeline_statistics_query = chosen_features.pipelineStatisticsQuery == VK_TRUE;
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
    VkPhysicalDeviceImageViewMinLodFeaturesEXT min_lod{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT};
    if (std::any_of(extensions.begin(), extensions.end(), [](const auto& e) {
            return std::strcmp(e.extensionName, VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME) == 0;
        })) {
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &min_lod};
        vkGetPhysicalDeviceFeatures2(info.physical, &query);
        info.image_view_min_lod = min_lod.minLod == VK_TRUE;
        if (info.image_view_min_lod) {
            min_lod.pNext = enable13.pNext;
            enable13.pNext = &min_lod;
        }
    }
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

#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
    // VK_KHR_shader_fma: OpFmaKHR is a single-rounding multiply-add in SPIR-V (GLSL.std.450 Fma may round twice).
    // Translated shaders built with `rdna2spirv.py --fma khr` need it (index capability FMAKHR, feature
    // shaderFmaFloat32); when it is missing they must not be created, so use `--fma glsl` there.
    VkPhysicalDeviceShaderFmaFeaturesKHR shader_fma{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR};
    if (std::any_of(extensions.begin(), extensions.end(), [](const auto& e) {
            return std::strcmp(e.extensionName, VK_KHR_SHADER_FMA_EXTENSION_NAME) == 0;
        })) {
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &shader_fma};
        vkGetPhysicalDeviceFeatures2(info.physical, &query);
        info.shader_fma = shader_fma.shaderFmaFloat32 == VK_TRUE;
        if (info.shader_fma) {
            shader_fma.pNext = enable13.pNext;
            enable13.pNext = &shader_fma;
        }
    }
#endif
    // What graphics PSOs may ask for: different blend state per render target, wireframe, a depth-bias clamp and
    // SRC1 blend factors. Enabled where the GPU has them; a PSO that needs a missing one is refused.
    enable.independentBlend = chosen_features.independentBlend;
    enable.fillModeNonSolid = chosen_features.fillModeNonSolid;
    enable.depthBiasClamp = chosen_features.depthBiasClamp;
    enable.dualSrcBlend = chosen_features.dualSrcBlend;
    enable.logicOp = chosen_features.logicOp;
    enable.depthClamp = chosen_features.depthClamp;
    // What translated shaders declare (the index's `features`): 64-bit integers in most of the corpus (25,803 of
    // 26,503 entries), doubles, 16-bit integers, and clip and cull distances in a few vertex shaders. Enabled where
    // the GPU has them; without one, vkCreateShaderModule reports the capability (VUID-...-pCode-08740).
    enable.shaderInt64 = chosen_features.shaderInt64;
    enable.shaderFloat64 = chosen_features.shaderFloat64;
    enable.shaderInt16 = chosen_features.shaderInt16;
    enable.shaderClipDistance = chosen_features.shaderClipDistance;
    enable.shaderCullDistance = chosen_features.shaderCullDistance;
    info.features = enable;

    // GetClockCalibration reads both clocks at once with calibrated timestamps (KHR, or the older EXT).
    const char* calibrated = nullptr;
    for (const char* name :
         {VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME}) {
        for (const auto& extension : extensions)
            if (std::strcmp(extension.extensionName, name) == 0) calibrated = name;
        if (calibrated) break;
    }
    info.calibrated_timestamps = calibrated != nullptr;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enable12};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pEnabledFeatures = &enable;
    std::vector<const char*> device_extensions;
    if (info.image_view_min_lod) device_extensions.push_back(VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME);
    if (info.conditional_rendering) device_extensions.push_back(conditional_extension);
    if (info.host_image_copy) device_extensions.push_back(host_image_copy_name);
#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
    if (info.shader_fma) device_extensions.push_back(VK_KHR_SHADER_FMA_EXTENSION_NAME);
#endif
    if (calibrated) device_extensions.push_back(calibrated);
    // VK_KHR_swapchain requires VK_KHR_surface on the instance (VUID-vkCreateDevice-ppEnabledExtensionNames-01387).
    if (has_swapchain && instance_has_surface) device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    if (info.ray_tracing) {
        device_extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        device_extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        device_extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    }
    info.presentation = has_swapchain && instance_has_surface;
    device_info.enabledExtensionCount = std::uint32_t(device_extensions.size());
    device_info.ppEnabledExtensionNames = device_extensions.empty() ? nullptr : device_extensions.data();
    if (vkCreateDevice(info.physical, &device_info, nullptr, &owned.device) != VK_SUCCESS)
        return fail("vkCreateDevice failed");
    info.device = owned.device;
    owned.destroy_device = reinterpret_cast<PFN_vkDestroyDevice>(vkGetDeviceProcAddr(owned.device, "vkDestroyDevice"));
    info.command_queues.resize(command_queue_count);
    const auto get_queue = reinterpret_cast<PFN_vkGetDeviceQueue>(vkGetDeviceProcAddr(owned.device, "vkGetDeviceQueue"));
    for (std::uint32_t i = 0; i < command_queue_count; ++i)
        get_queue(owned.device, info.queue_family, i, &info.command_queues[i]);
    info.queue = info.command_queues[0];
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
