// Typed RTVs of every TYPELESS color family (D3D-275): ClearRenderTargetView through an RTV in each typed format of
// the family, read back byte for byte and compared with the host's D3D12 on the same adapter (matched by LUID, the
// D3D-188 rule) and with WARP. Before D3D-275 only the RGBA8 family was reinterpreted: an R16_UINT or R32G32_UINT RTV
// of a typeless texture cleared through the resource's own (float or UNORM) format.
// A view format the layer does not map yet (its CheckFeatureSupport gives no RENDER_TARGET) is reported and skipped,
// so the R8 family and the 16-bit integer and normalized formats are covered as soon as their mapping lands.
// Synthetic data only. LEONIDA_VK_VALIDATION=1 runs it under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

struct Family {
    DXGI_FORMAT typeless;
    std::vector<DXGI_FORMAT> views;
};

// Values that tell the interpretations apart: a fraction (UNORM / FLOAT), 7 (UINT keeps it, UNORM clamps it to 1),
// a negative one (SINT / SNORM keep it, unsigned ones clamp it), one past every 8- and 16-bit integer range.
constexpr float kValues[2][4] = {{0.5f, 7.0f, -1.0f, 300.0f}, {0.25f, 1.0f, -128.0f, 70000.0f}};

// Clears a 4x4 TYPELESS texture through an RTV in `view` and reads it back. False when a step failed; `made` is false
// when the device refused the texture or the view (an empty descriptor), which only our layer may do.
bool clear_and_read(ID3D12Device* device, DXGI_FORMAT typeless, DXGI_FORMAT view, const float* value,
                    std::vector<unsigned char>& bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 4;
    desc.Height = 4;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = typeless;
    desc.SampleDesc = {1, 0};
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const D3D12_HEAP_PROPERTIES def_heap{D3D12_HEAP_TYPE_DEFAULT}, rb_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> texture, readback;
    if (FAILED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                               nullptr, IID_PPV_ARGS(&texture.p))))
        return false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &row_bytes, &total);
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc = {1, 0};
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&readback.p))))
        return false;
    Ref<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p)))) return false;
    D3D12_RENDER_TARGET_VIEW_DESC rtv{};
    rtv.Format = view;
    rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(texture.p, &rtv, handle);
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&list.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return false;
    list->ClearRenderTargetView(handle, value, 0, nullptr);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {texture.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET,
                          D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION dst{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dst.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION src{texture.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 1)) || FAILED(fence->SetEventOnCompletion(1, nullptr))) return false;
    void* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, &mapped))) return false;
    bytes.clear();
    for (UINT y = 0; y < rows; ++y) {
        const auto* row = static_cast<const unsigned char*>(mapped) + footprint.Offset + y * footprint.Footprint.RowPitch;
        bytes.insert(bytes.end(), row, row + row_bytes);
    }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}

// The host's D3D12 device on the adapter whose LUID the Vulkan device reports (not the default adapter: D3D-188).
bool host_device(const vktest::Device& vk, Ref<ID3D12Device>& out, std::string& name) {
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    vk.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(vk.physical, &props);
    Ref<IDXGIFactory4> factory;
    if (!id.deviceLUIDValid || FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return false;
    LUID luid{};
    std::memcpy(&luid, id.deviceLUID, sizeof(luid));
    Ref<IDXGIAdapter1> adapter;
    if (FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter.p)))) return false;
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    char utf8[256]{};
    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, utf8, sizeof(utf8), nullptr, nullptr);
    name = utf8;
    return SUCCEEDED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&out.p)));
}

std::string hex(const std::vector<unsigned char>& bytes, std::size_t texel) {
    std::string s;
    char b[4];
    for (std::size_t i = 0; i < texel && i < bytes.size(); ++i) {
        std::snprintf(b, sizeof(b), "%02x", bytes[i]);
        s += b;
    }
    return s;
}

}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (hostwine::host_is_wine()) return 77;  // the host D3D12 runtime is the reference
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.bindless = vk.bindless;
    info.bindless_partially_bound = vk.partially_bound;
    info.host_image_copy = vk.host_image_copy;
    Ref<ID3D12Device> ours, host, warp;
    std::string adapter;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
    if (!host_device(vk, host, adapter)) {
        std::fprintf(stderr, "no host D3D12 device on the Vulkan device's adapter\n");
        return 77;
    }
    {
        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> warp_adapter;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) &&
            SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp_adapter.p))))
            D3D12CreateDevice(warp_adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&warp.p));
    }
    std::printf("host reference: %s%s\n", adapter.c_str(), warp.p ? ", and WARP" : "");
    const std::vector<Family> families = {
        {DXGI_FORMAT_R8G8B8A8_TYPELESS, {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                                         DXGI_FORMAT_R8G8B8A8_UINT, DXGI_FORMAT_R8G8B8A8_SNORM, DXGI_FORMAT_R8G8B8A8_SINT}},
        {DXGI_FORMAT_B8G8R8A8_TYPELESS, {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB}},
        {DXGI_FORMAT_R16G16B16A16_TYPELESS, {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_UNORM,
                                             DXGI_FORMAT_R16G16B16A16_UINT, DXGI_FORMAT_R16G16B16A16_SNORM,
                                             DXGI_FORMAT_R16G16B16A16_SINT}},
        {DXGI_FORMAT_R32G32B32A32_TYPELESS, {DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_UINT,
                                             DXGI_FORMAT_R32G32B32A32_SINT}},
        {DXGI_FORMAT_R32G32_TYPELESS, {DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32_SINT}},
        {DXGI_FORMAT_R16G16_TYPELESS, {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_UNORM, DXGI_FORMAT_R16G16_UINT,
                                       DXGI_FORMAT_R16G16_SNORM, DXGI_FORMAT_R16G16_SINT}},
        {DXGI_FORMAT_R10G10B10A2_TYPELESS, {DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R10G10B10A2_UINT}},
        {DXGI_FORMAT_R8G8_TYPELESS, {DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8_UINT, DXGI_FORMAT_R8G8_SNORM,
                                     DXGI_FORMAT_R8G8_SINT}},
        {DXGI_FORMAT_R32_TYPELESS, {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_SINT}},
        {DXGI_FORMAT_R16_TYPELESS, {DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16_UINT,
                                    DXGI_FORMAT_R16_SNORM, DXGI_FORMAT_R16_SINT}},
        {DXGI_FORMAT_R8_TYPELESS, {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8_SNORM, DXGI_FORMAT_R8_SINT}},
    };
    unsigned compared = 0, skipped = 0, failed = 0, warp_differs = 0;
    for (const auto& family : families) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT typeless_support{family.typeless};
        const bool texture = SUCCEEDED(ours->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &typeless_support,
                                                                 sizeof(typeless_support))) &&
                             (typeless_support.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D);
        for (const DXGI_FORMAT view : family.views) {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT support{view};
            if (!texture || FAILED(ours->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) ||
                !(support.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET)) {
                std::printf("typeless %d view %d: SKIP, the layer has no render target in this format yet\n",
                            int(family.typeless), int(view));
                ++skipped;
                continue;
            }
            for (const auto& value : kValues) {
                std::vector<unsigned char> mine, reference, software;
                if (!clear_and_read(host.p, family.typeless, view, value, reference)) {
                    std::fprintf(stderr, "typeless %d view %d: the host refused\n", int(family.typeless), int(view));
                    return 1;
                }
                if (!clear_and_read(ours.p, family.typeless, view, value, mine)) {
                    std::fprintf(stderr, "typeless %d view %d: the layer failed\n", int(family.typeless), int(view));
                    return 1;
                }
                const std::size_t texel = reference.size() / 16;
                ++compared;
                const bool same = mine == reference;
                failed += !same;
                if (warp.p && clear_and_read(warp.p, family.typeless, view, value, software) && software != reference)
                    ++warp_differs;
                std::printf("typeless %d view %d clear (%g, %g, %g, %g): host %s, layer %s%s%s\n", int(family.typeless),
                            int(view), value[0], value[1], value[2], value[3], hex(reference, texel).c_str(),
                            hex(mine, texel).c_str(), same ? "" : "  MISMATCH",
                            software.empty() || software == reference ? "" : ("  WARP " + hex(software, texel)).c_str());
            }
        }
    }
    std::printf("%u clears compared with %s, %u differ; %u view formats skipped; WARP differs from the host in %u\n",
                compared, adapter.c_str(), failed, skipped, warp_differs);
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", vk.validation_errors());
        return 1;
    }
    return failed ? 1 : 0;
}
