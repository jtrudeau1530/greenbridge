// A Vulkan-backed ID3D12Device (PC ABI) for the Xbox-ABI device proxy (d3d12_xs_device.cpp) to forward to.
// Implemented: queues, allocators, command lists, fences, descriptor heaps, render-target and depth-stencil views,
// committed buffers and 2D textures, a few feature queries. Everything else is a generated stub that reports
// itself once. See docs/d3d12_vulkan.md section 4.1/4.2 for what each answer means and what is still open.
#include "d3d12_com.h"

#include "d3d12_stubs.h"  // generated
#include "d3d12_com_object.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
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
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return {VK_FORMAT_R16G16B16A16_SFLOAT, color, true};
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return {VK_FORMAT_R16G16B16A16_SFLOAT, color};
    case DXGI_FORMAT_R16G16B16A16_UNORM: return {VK_FORMAT_R16G16B16A16_UNORM, color};
    case DXGI_FORMAT_R32G32_FLOAT: return {VK_FORMAT_R32G32_SFLOAT, color};
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return {VK_FORMAT_A2B10G10R10_UNORM_PACK32, color, true};
    case DXGI_FORMAT_R10G10B10A2_UNORM: return {VK_FORMAT_A2B10G10R10_UNORM_PACK32, color};
    case DXGI_FORMAT_R11G11B10_FLOAT: return {VK_FORMAT_B10G11R11_UFLOAT_PACK32, color};
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return {VK_FORMAT_R8G8B8A8_UNORM, color, true};
    case DXGI_FORMAT_R8G8B8A8_UNORM: return {VK_FORMAT_R8G8B8A8_UNORM, color};
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return {VK_FORMAT_R8G8B8A8_SRGB, color};
    case DXGI_FORMAT_R8G8B8A8_UINT: return {VK_FORMAT_R8G8B8A8_UINT, color};
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return {VK_FORMAT_B8G8R8A8_UNORM, color, true};
    case DXGI_FORMAT_B8G8R8A8_UNORM: return {VK_FORMAT_B8G8R8A8_UNORM, color};
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return {VK_FORMAT_B8G8R8A8_SRGB, color};
    case DXGI_FORMAT_R16G16_FLOAT: return {VK_FORMAT_R16G16_SFLOAT, color};
    case DXGI_FORMAT_R8G8_UNORM: return {VK_FORMAT_R8G8_UNORM, color};
    case DXGI_FORMAT_R16_FLOAT: return {VK_FORMAT_R16_SFLOAT, color};
    case DXGI_FORMAT_R8_UNORM: return {VK_FORMAT_R8_UNORM, color};
    case DXGI_FORMAT_R32_FLOAT: return {VK_FORMAT_R32_SFLOAT, color};
    case DXGI_FORMAT_R32_UINT: return {VK_FORMAT_R32_UINT, color};
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

// What a device's children need from it, kept apart from the COM class so children can hold a plain pointer.
struct DeviceState {
    VulkanDeviceInfo info;
    std::unique_ptr<leonida::VulkanContext> context;
    VkPhysicalDeviceMemoryProperties memory{};
    PFN_vkGetPhysicalDeviceFormatProperties format_properties = nullptr;

    bool depth_usable(VkFormat format) const {
        VkFormatProperties properties{};
        format_properties(info.physical, format, &properties);
        return properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
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
        device_.context->remove_resource(id());
        if (image_) vk.vkDestroyImage(dev, image_, nullptr);
        if (buffer_) vk.vkDestroyBuffer(dev, buffer_, nullptr);
        if (memory_) vk.vkFreeMemory(dev, memory_, nullptr);
    }

    leonida::ResourceId id() const { return resource_id(this); }
    VkImage image() const { return image_; }

    HRESULT create_buffer() {
        auto& vk = device_.context->vk;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = desc_.Width;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
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
        if (desc_.MipLevels == 0) {
            UINT64 size = std::max<UINT64>(desc_.Width, desc_.Height);
            UINT16 mips = 1;
            while (size > 1) {
                size >>= 1;
                ++mips;
            }
            desc_.MipLevels = mips;
        }
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.flags = format.typeless ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format.format;
        info.extent = {std::uint32_t(desc_.Width), desc_.Height, 1};
        info.mipLevels = desc_.MipLevels;
        info.arrayLayers = desc_.DepthOrArraySize;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (desc_.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (desc_.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
            info.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (desc_.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        if (!(desc_.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)) info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if (vk.vkCreateImage(device_.info.device, &info, nullptr, &image_) != VK_SUCCESS) return E_OUTOFMEMORY;
        VkMemoryRequirements requirements{};
        vk.vkGetImageMemoryRequirements(device_.info.device, image_, &requirements);
        memory_ = device_.allocate(requirements, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!memory_ || vk.vkBindImageMemory(device_.info.device, image_, memory_, 0) != VK_SUCCESS)
            return E_OUTOFMEMORY;

        leonida::ResourceBinding binding{};
        binding.image = image_;
        binding.format = format.format;
        binding.aspects = format.aspects;
        binding.width = std::uint32_t(desc_.Width);
        binding.height = desc_.Height;
        binding.mip_levels = desc_.MipLevels;
        binding.array_layers = desc_.DepthOrArraySize;
        binding.simultaneous_access = (desc_.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS) != 0;
        binding.not_zeroed = (heap_flags_ & D3D12_HEAP_FLAG_CREATE_NOT_ZEROED) != 0;
        // Never used yet: the first command that touches a part zero-fills it and moves it to the layout of the
        // state that command expects (vulkan_replay.h, initialize), so no submission is needed here.
        device_.context->add_resource(id(), binding, VK_IMAGE_LAYOUT_UNDEFINED);
        return S_OK;
    }

    D3D12_RESOURCE_DESC STDMETHODCALLTYPE GetDesc() override { return desc_; }

    HRESULT STDMETHODCALLTYPE GetHeapProperties(D3D12_HEAP_PROPERTIES* properties, D3D12_HEAP_FLAGS* flags) override {
        if (properties) *properties = heap_;
        if (flags) *flags = heap_flags_;
        return S_OK;
    }

    // Buffers on upload and readback heaps are mapped persistently; the ranges are hints, as on D3D12.
    HRESULT STDMETHODCALLTYPE Map(UINT subresource, const D3D12_RANGE*, void** data) override {
        if (!mapped_ || subresource != 0) return E_INVALIDARG;
        if (data) *data = mapped_;
        return S_OK;
    }
    void STDMETHODCALLTYPE Unmap(UINT, const D3D12_RANGE*) override {}

private:
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
        const auto get_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            info.get_instance_proc(info.instance, "vkGetDeviceProcAddr"));
        state_.format_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceFormatProperties"));
        const auto memory_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceMemoryProperties"));
        if (!get_device || !state_.format_properties || !memory_properties) return;
        memory_properties(info.physical, &state_.memory);
        state_.context = std::make_unique<leonida::VulkanContext>(get_device, info.device,
                                                                  info.separate_depth_stencil_layouts);
    }

    ~Device() override {
        if (state_.context) {
            std::lock_guard lock(state_.context->submit_mutex());
            state_.context->vk.vkDeviceWaitIdle(state_.info.device);
        }
        state_.context.reset();
        if (owned_.device) owned_.destroy_device(owned_.device, nullptr);
        if (owned_.instance) owned_.destroy_instance(owned_.instance, nullptr);
    }

    bool ok() const { return state_.context && state_.context->loaded(); }
    void own(const OwnedRuntime& owned) { owned_ = owned; }
    void set_outer(IUnknown* outer) { outer_ = outer; }

    UINT STDMETHODCALLTYPE GetNodeCount() override { return 1; }
    HRESULT STDMETHODCALLTYPE GetDeviceRemovedReason() override { return S_OK; }

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
            } else if (desc && desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2DARRAY) {
                view = {r->id(), desc->Texture2DArray.MipSlice, desc->Texture2DArray.FirstArraySlice,
                        desc->Texture2DArray.ArraySize};
            } else if (desc) {
                log_once("render-target view dimension not supported:", desc->ViewDimension);
                view = {};
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
        } else if (desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
            log_once("committed resource dimension not supported:", desc->Dimension);
        } else if (desc->SampleDesc.Count > 1) {
            log_once("multisampled committed resource not supported, samples:", int(desc->SampleDesc.Count));
        } else if (heap->Type == D3D12_HEAP_TYPE_UPLOAD || heap->Type == D3D12_HEAP_TYPE_READBACK) {
            hr = E_INVALIDARG;  // D3D12 places textures on default heaps unless they are row-major
        } else {
            Format format = vulkan_format(desc->Format, desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
            if (format.format == VK_FORMAT_D24_UNORM_S8_UINT && !state_.depth_usable(format.format)) {
                log_once("D24_UNORM_S8_UINT unavailable, using D32_SFLOAT_S8_UINT; format", desc->Format);
                format.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
            }
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
        if (desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
            log_once("GetCopyableFootprints dimension not supported:", desc->Dimension);
            return fail();
        }
        const Format format = vulkan_format(desc->Format, true);  // typeless depth families are planar either way
        UINT mips = desc->MipLevels;
        if (!mips)
            for (UINT64 size = std::max<UINT64>(desc->Width, desc->Height); size; size >>= 1) ++mips;
        const UINT planes = (format.aspects & VK_IMAGE_ASPECT_STENCIL_BIT) ? 2 : 1;
        const UINT per_plane = mips * desc->DepthOrArraySize;
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
            if (layouts)
                layouts[i] = {offset, {footprint_format, blocks_wide * block.block, blocks_high * block.block, 1, pitch}};
            if (rows) rows[i] = blocks_high;
            if (row_sizes) row_sizes[i] = row_size;
            end = offset + UINT64(blocks_high - 1) * pitch + row_size;
            offset += UINT64(blocks_high) * pitch;
        }
        if (total) *total = end - base;
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
        default:
            // Not answered yet. The log above lists every query; answers are added as the trace asks for them.
            return E_INVALIDARG;
        }
    }

private:
    IUnknown* identity() { return outer_ ? outer_ : static_cast<ID3D12Device*>(this); }

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
            }
            break;
        }
    }
    if (!info.physical)
        return fail("no Vulkan 1.3 GPU with dynamicRendering, synchronization2, timelineSemaphore and "
                    "separateDepthStencilLayouts");

    VkPhysicalDeviceVulkan13Features enable13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    enable13.dynamicRendering = VK_TRUE;
    enable13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &enable13};
    enable12.timelineSemaphore = VK_TRUE;
    enable12.separateDepthStencilLayouts = VK_TRUE;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = info.queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures enable{};
    enable.textureCompressionBC = chosen_bc ? VK_TRUE : VK_FALSE;  // the game's textures are BC1-7
    if (!chosen_bc) std::fprintf(stderr, "[rt] d3d12: the GPU has no BC texture support\n");
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enable12};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pEnabledFeatures = &enable;
    if (vkCreateDevice(info.physical, &device_info, nullptr, &owned.device) != VK_SUCCESS)
        return fail("vkCreateDevice failed");
    info.device = owned.device;
    owned.destroy_device = reinterpret_cast<PFN_vkDestroyDevice>(vkGetDeviceProcAddr(owned.device, "vkDestroyDevice"));
    reinterpret_cast<PFN_vkGetDeviceQueue>(vkGetDeviceProcAddr(owned.device, "vkGetDeviceQueue"))(
        owned.device, info.queue_family, 0, &info.queue);
    std::fprintf(stderr, "[rt] d3d12: Vulkan device: %s\n", chosen.deviceName);

    auto* device = new Device(info);
    device->own(owned);  // destroyed with the device from here on, even on failure
    if (!device->ok()) {
        device->Release();
        return E_FAIL;
    }
    return hand_out(device, riid, out);
}

}  // namespace rt::d3d12
