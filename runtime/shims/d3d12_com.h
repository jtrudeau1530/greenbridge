// D3D12 COM objects over the Vulkan backend (src/vulkan_backend.h): command allocator, graphics command list,
// command queue and fence. Their vtables are the Windows SDK's (ID3D12GraphicsCommandList8, ID3D12CommandQueue,
// ID3D12CommandAllocator, ID3D12Fence1); every method not implemented here is a generated stub that reports its
// name once (tools/gen_d3d12_stubs.py). The Xbox-specific slots measured for the device do not apply to these
// interfaces as far as is known; docs/d3d12_vulkan.md section 3.1 lists what is still unverified.
//
// Contracts with the rest of the layer (the device proxy that will hand these objects out):
// - an ID3D12Resource's leonida::ResourceId is its pointer value; the device registers it with the VulkanContext;
// - so is an ID3D12QueryHeap's leonida::QueryHeapId, registered with its VkQueryPool;
// - the CPU handle of an RTV or DSV points at a leonida::TargetView in the device's descriptor heap.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include "../src/vulkan_backend.h"

namespace rt::d3d12 {

// `device` (optional) is what the object's GetDevice returns; the object holds a reference to it.
HRESULT create_command_queue(leonida::VulkanContext& context, VkQueue queue, std::uint32_t queue_family,
                             const D3D12_COMMAND_QUEUE_DESC& desc, REFIID riid, void** out,
                             IUnknown* device = nullptr);
HRESULT create_command_allocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void** out, IUnknown* device = nullptr);
HRESULT create_command_list(D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* allocator, REFIID riid,
                            void** out, IUnknown* device = nullptr);
HRESULT create_fence(leonida::VulkanContext& context, UINT64 initial_value, D3D12_FENCE_FLAGS flags, REFIID riid,
                     void** out, IUnknown* device = nullptr);

// The Vulkan objects a D3D12 device runs on. Vulkan 1.3 with dynamicRendering, synchronization2, timelineSemaphore
// and (when separate_depth_stencil_layouts is set) separateDepthStencilLayouts enabled.
struct VulkanDeviceInfo {
    PFN_vkGetInstanceProcAddr get_instance_proc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;  // a graphics queue; every D3D12 queue of the device submits to it
    std::uint32_t queue_family = 0;
    bool separate_depth_stencil_layouts = true;
    std::uint32_t max_push_constants_size = 0;
    bool push_constants_need_fallback = false;
    // Enabled: runtimeDescriptorArray, the dynamic indexing of sampled- and storage-image arrays, and update after
    // bind for sampled images, storage images and storage buffers. The device then keeps the bindless set 0 of
    // the translated shaders (docs section 3.4), and texture views and samplers are put in it as they are
    // created. bindless_partially_bound: descriptorBindingPartiallyBound is enabled as well.
    bool bindless = false;
    bool bindless_partially_bound = false;
    // samplerAnisotropy and samplerMirrorClampToEdge enabled; without them such samplers are approximated.
    bool sampler_anisotropy = false;
    bool sampler_mirror_clamp_to_edge = false;
    // VK_EXT_conditional_rendering (with its conditionalRendering feature) is enabled: SetPredication works, and
    // buffers are created with VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT. Without it a predicated list is
    // rejected at ExecuteCommandLists.
    bool conditional_rendering = false;
    // VK_EXT_host_image_copy and its hostImageCopy feature are enabled on `device`: WriteToSubresource and
    // ReadFromSubresource on textures in CPU-visible heaps copy on the CPU instead of through a staging submit.
    bool host_image_copy = false;
    // The caller enabled textureCompressionBC on this logical device.
    bool texture_compression_bc = false;
    // Optional device features the caller enabled. Without occlusionQueryPrecise, D3D12_QUERY_TYPE_OCCLUSION counts
    // are only zero or non-zero; without pipelineStatisticsQuery, pipeline-statistics query heaps are refused.
    bool occlusion_query_precise = false;
    bool pipeline_statistics_query = false;
    // VK_EXT_image_view_min_lod and its minLod feature are enabled on the logical device.
    bool image_view_min_lod = false;
};

// A Vulkan-backed ID3D12Device (PC ABI, up to ID3D12Device11) on caller-owned Vulkan objects, which must outlive
// the device and every object it created. Wrap it in rt::d3d12_xs::create_device_proxy for the Xbox ABI.
HRESULT create_vulkan_device(const VulkanDeviceInfo& info, REFIID riid, void** out);

// The same, on a Vulkan instance and device of its own (destroyed with the D3D12 device). Picks the first discrete
// GPU with the features above. LEONIDA_VK_VALIDATION=1 enables VK_LAYER_KHRONOS_validation.
HRESULT create_vulkan_device_standalone(REFIID riid, void** out);

// Tells a device from create_vulkan_device* that `outer` (the Xbox-ABI proxy that owns it) is what its children's
// GetDevice must return. No reference is taken: the outer object owns the device.
void set_device_outer(ID3D12Device* device, IUnknown* outer);

// The VkImage behind a texture created by such a device, or VK_NULL_HANDLE. For tests and the swapchain.
VkImage vulkan_image(ID3D12Resource* resource);

// The property flags of the Vulkan memory type behind a resource created by such a device, or 0. For tests.
VkMemoryPropertyFlags vulkan_memory_flags(ID3D12Resource* resource);

// Whether both resources use the requested typed, non-integer colour format for a regional average resolve.
bool color_resolve_format_matches(DXGI_FORMAT format, ID3D12Resource* dst, ID3D12Resource* src);

// The maxPushConstantsSize in bytes of the Vulkan physical device behind `device`, or 0 if not a Vulkan device.
std::uint32_t device_max_push_constants_size(ID3D12Device* device);

// True when `device`'s Vulkan physical device has maxPushConstantsSize < 144 bytes, requiring
// the uniform-buffer fallback for the 16-byte tail (heap/code base).
bool device_push_constants_need_fallback(ID3D12Device* device);

// True when `device` puts upload-heap buffers in DEVICE_LOCAL | HOST_VISIBLE memory: its GPU has Resizable BAR and
// LEONIDA_UPLOAD_SYSMEM is not set (D3D-96).
bool device_upload_in_vram(ID3D12Device* device);

// The bindless set 0 a device with VulkanDeviceInfo::bindless keeps (vulkan_bindless.h), and its layout: what a
// pipeline whose shader declares set-0 bindings is created with and binds. Null without such a set. The set is
// brought up to date at every ExecuteCommandLists.
VkDescriptorSet bindless_set(ID3D12Device* device);
VkDescriptorSetLayout bindless_set_layout(ID3D12Device* device);

// ExecuteCommandLists returns void; this is what the last call on the queue did.
leonida::ReplayResult last_execute_result(ID3D12CommandQueue* queue);

// The front/back stencil references last recorded by ID3D12GraphicsCommandList8::OMSetFrontAndBackStencilRef on
// `list`. The values are kept for the future pipeline path, which is what would apply them. False when `list` is
// not one of this module's command lists.
bool front_back_stencil_ref(ID3D12GraphicsCommandList8* list, UINT& front, UINT& back);

inline leonida::ResourceId resource_id(const ID3D12Resource* resource) {
    return reinterpret_cast<leonida::ResourceId>(resource);
}

// A query heap's ID in the command-list IR and the VulkanContext: its pointer value, as for resources.
inline leonida::QueryHeapId query_heap_id(const ID3D12QueryHeap* heap) {
    return reinterpret_cast<leonida::QueryHeapId>(heap);
}

}  // namespace rt::d3d12
