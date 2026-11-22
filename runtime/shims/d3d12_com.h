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

#include "../src/gfx10_descriptors.h"
#include "../src/vulkan_backend.h"

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "../src/root_signature.h"

namespace leonida {
class VulkanPresenter;
}

namespace rt::d3d12 {

// How a queue answers GetClockCalibration. With `calibrated` (vkGetCalibratedTimestampsKHR or its EXT alias, on a
// device whose time domains include DEVICE and QUERY_PERFORMANCE_COUNTER) both clocks are read in one call. Without
// it, the queue writes a timestamp query and reads QueryPerformanceCounter around it; that needs `get_device_proc`
// and a queue family with timestamp_valid_bits > 0. With neither, GetClockCalibration fails.
struct QueueClock {
    PFN_vkGetCalibratedTimestampsKHR calibrated = nullptr;
    PFN_vkGetDeviceProcAddr get_device_proc = nullptr;
    std::uint32_t timestamp_valid_bits = 0;
};

// `device` (optional) is what the object's GetDevice returns; the object holds a reference to it.
HRESULT create_command_queue(leonida::VulkanContext& context, VkQueue queue, std::uint32_t queue_family,
                             const D3D12_COMMAND_QUEUE_DESC& desc, REFIID riid, void** out,
                             IUnknown* device = nullptr, const QueueClock& clock = {});
HRESULT create_command_allocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void** out, IUnknown* device = nullptr);
HRESULT create_command_list(D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* allocator, REFIID riid,
                            void** out, IUnknown* device = nullptr);
HRESULT create_fence(leonida::VulkanContext& context, UINT64 initial_value, D3D12_FENCE_FLAGS flags, REFIID riid,
                     void** out, IUnknown* device = nullptr);
HRESULT create_command_signature(leonida::IndirectKind kind, UINT byte_stride, REFIID riid, void** out,
                                 IUnknown* device = nullptr);
bool command_signature_info(ID3D12CommandSignature* signature, leonida::IndirectKind& kind, UINT& byte_stride);

// ID3D12Device1::SetEventOnMultipleFenceCompletion on fences from create_fence: sets `event` once every fence
// (D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL) or any fence (_ANY) has reached its value; a null event blocks until then.
HRESULT set_event_on_multiple_fence_completion(leonida::VulkanContext& context, ID3D12Fence* const* fences,
                                               const UINT64* values, UINT count,
                                               D3D12_MULTIPLE_FENCE_WAIT_FLAGS flags, HANDLE event);

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
    // VK_KHR_calibrated_timestamps or VK_EXT_calibrated_timestamps is enabled on `device` (QueueClock).
    bool calibrated_timestamps = false;
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
    bool presentation = false;  // VK_KHR_surface + VK_KHR_win32_surface on the instance, VK_KHR_swapchain on the device
    // VK_EXT_image_view_min_lod and its minLod feature are enabled on the logical device.
    bool image_view_min_lod = false;
    // The instance enabled VK_EXT_debug_utils: SetMarker/BeginEvent/EndEvent become debug labels. Otherwise they
    // do nothing.
    bool debug_utils = false;
    // Vulkan 1.2 bufferDeviceAddress is enabled: buffers and shader-visible heaps then live in the GPU address
    // arena and have console VAs (docs section 3.3), and GetGPUVirtualAddress returns those addresses; ray tracing
    // needs it too. Without it, GetGPUVirtualAddress returns 0 and pipelines cannot be created.
    bool buffer_device_address = false;
    // Optional VK_KHR_acceleration_structure + VK_KHR_deferred_host_operations + VK_KHR_ray_query are enabled:
    // GetRaytracingAccelerationStructurePrebuildInfo and BuildRaytracingAccelerationStructure (BLAS + TLAS) work.
    bool ray_tracing = false;
    // The core features enabled on `device`. Graphics PSOs that need independentBlend, fillModeNonSolid,
    // depthBiasClamp or dualSrcBlend are refused unless it is set here.
    VkPhysicalDeviceFeatures features{};
    // The translator's output folder (spirv/index.json + <name>.spv) for pipeline creation; empty: the
    // LEONIDA_SHADERS environment variable.
    std::string shader_dir;
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
// The VkBuffer behind a buffer created by such a device, or VK_NULL_HANDLE. For tests.
VkBuffer vulkan_buffer(ID3D12Resource* resource);
// True when `device` has VK_KHR_acceleration_structure + VK_KHR_ray_query enabled.
bool device_supports_ray_tracing(ID3D12Device* device);
// The VkAccelerationStructureKHR registered for `va` on `device`, or VK_NULL_HANDLE. For tests.
VkAccelerationStructureKHR vulkan_acceleration_structure(ID3D12Device* device, D3D12_GPU_VIRTUAL_ADDRESS va);

// The property flags of the Vulkan memory type behind a resource created by such a device, or 0. For tests.
VkMemoryPropertyFlags vulkan_memory_flags(ID3D12Resource* resource);

// Whether both resources use the requested typed, non-integer colour format for a regional average resolve.
bool color_resolve_format_matches(DXGI_FORMAT format, ID3D12Resource* dst, ID3D12Resource* src);

// Whether a resolve of one depth or stencil plane can be replayed: both resources are depth-stencil textures of
// `format` (a depth format, or a plane's shader-resource format such as R24_UNORM_X8_TYPELESS), the source
// multisampled and the destination not, the same plane on both sides, MIN or MAX (or AVERAGE of the depth plane by
// its own format), the destination offset the source rectangle's corner (or 0, 0 without one), and the Vulkan device
// has the mode.
bool depth_stencil_resolve_supported(DXGI_FORMAT format, ID3D12Resource* dst, UINT dst_subresource, UINT x, UINT y,
                                     ID3D12Resource* src, UINT src_subresource, const D3D12_RECT* rect,
                                     D3D12_RESOLVE_MODE mode);

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
// Gives each S# a sampler in that set and a key in its hash table, as CreateSampler does, without a heap slot: for
// the literal S#s a shader builds from immediates (`literal_samplers` in the shader index). An S# that has them
// already is left as it is. E_INVALIDARG when `device` is not such a device; E_FAIL when one of them could not be
// registered (no bindless set, or its sampler array is full), the others are registered all the same.
HRESULT register_samplers(ID3D12Device* device, std::span<const leonida::gfx10::SamplerWords> samplers);
// True when GetClockCalibration on `queue` reads calibrated timestamps, false when it submits a timestamp query
// (or `queue` is not one of this module's queues). For tests.
bool queue_uses_calibrated_timestamps(ID3D12CommandQueue* queue);

// Creates a Vulkan-backed IDXGIFactory4 bound to `device`'s VulkanContext, instance, physical device, and queue.
// Its CreateSwapChainForHwnd registers swapchain backbuffers directly with the device's VulkanContext.
HRESULT create_dxgi_factory(ID3D12Device* device, REFIID riid, void** out);

// Puts back buffers of a device from create_vulkan_device* into a window (leonida::VulkanPresenter), for the
// presentation front end the game uses (DXGI or PresentX, still open). Holds a reference to the device.
class WindowPresenter {
public:
    WindowPresenter(IUnknown* device, std::unique_ptr<leonida::VulkanPresenter> presenter);
    ~WindowPresenter();
    WindowPresenter(const WindowPresenter&) = delete;
    WindowPresenter& operator=(const WindowPresenter&) = delete;

    // `back_buffer` is a colour texture in D3D12's PRESENT state. sync_interval 0 presents without vsync.
    // S_OK also covers a skipped present while the window is minimised.
    HRESULT present(ID3D12Resource* back_buffer, UINT sync_interval);
    VkExtent2D extent() const;
    std::uint64_t presented() const;

private:
    IUnknown* device_;
    std::unique_ptr<leonida::VulkanPresenter> presenter_;
};

// Null when `device` is not a Vulkan device created with presentation, or the window cannot be presented to.
std::unique_ptr<WindowPresenter> create_window_presenter(ID3D12Device* device, HWND window);

// Where each root parameter's arguments go in the 32 user-data dwords the translated shaders read, as
// rt::rootsig::pack() lays them out (docs section 3.5). Built when a Vulkan device creates the root signature.
struct RootArgumentLayout {
    struct Parameter {
        rt::rootsig::ParamType type;
        std::uint32_t user_offset;     // first dword; unused for a parameter with no dwords
        std::uint32_t constant_count;  // constants only
    };
    std::vector<Parameter> parameters;
};

// For the command list: the argument layout of a root signature from a Vulkan device, and the id it records for a
// compute pipeline state (0: not a compute PSO of a Vulkan device).
const RootArgumentLayout* root_signature_layout(ID3D12RootSignature* signature);
std::uintptr_t compute_pipeline_id(ID3D12PipelineState* pipeline);
std::uintptr_t graphics_pipeline_id(ID3D12PipelineState* pipeline);

// The size of every descriptor slot (GetDescriptorHandleIncrementSize), for handle ranges.
inline constexpr UINT descriptor_size = 32;

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
