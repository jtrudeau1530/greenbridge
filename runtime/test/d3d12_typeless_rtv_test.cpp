
// Compare typed RTVs on one typeless RGBA8 resource against Windows D3D12/WARP.
// No game data or shaders. Run explicitly (needs Vulkan hardware); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <algorithm>
#include <array>
#include <utility>
#include <vector>
#include <cstring>
#include <limits>
#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};
struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

bool clear_and_read(ID3D12Device* device, DXGI_FORMAT whole_format, DXGI_FORMAT rect_format, bool partial,
                    bool rewrite, bool uav, std::vector<unsigned char>& pixels,
                    UINT mip = 0, UINT first_layer = 0, UINT layer_count = 1, bool array = false, bool mip_chain = false) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    const FLOAT full_float[4] = {0.25f, 0.5f, 0.75f, 0.25f}, part_float[4] = {0.0031308f, 0.18f, 0.5f, 0.75f};
    const FLOAT full_uint[4] = {7, 128, 255, 9}, part_uint[4] = {1.5f, 2.5f, 256, -1};
    const FLOAT* full = whole_format == DXGI_FORMAT_R8G8B8A8_UINT ? full_uint : full_float;
    const FLOAT* part = rect_format == DXGI_FORMAT_R8G8B8A8_UINT ? part_uint : part_float;
    constexpr unsigned bytes = 4;
    constexpr UINT width = 5, height = 3; // Odd dimensions expose incorrect treatment of footprint padding.
    Ref<ID3D12Resource> image, readback;
    D3D12_HEAP_PROPERTIES heap_properties{};
    heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = width;
    texture.Height = height;
    texture.DepthOrArraySize = array ? 4 : 1;
    texture.MipLevels = array || mip_chain ? 3 : 1;
    texture.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
    texture.SampleDesc.Count = 1;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (uav) texture.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &texture,
        D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&image.p))));
    const UINT count = texture.MipLevels * texture.DepthOrArraySize;
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> placed(count);
    UINT64 total = 0;
    device->GetCopyableFootprints(&texture, 0, count, 0, placed.data(), nullptr, nullptr, &total);
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = buffer.SampleDesc.Count = 1;
    buffer.DepthOrArraySize = buffer.MipLevels = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    heap_properties.Type = D3D12_HEAP_TYPE_READBACK;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));
    Ref<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap.p))));
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    const D3D12_CPU_DESCRIPTOR_HANDLE other{handle.ptr + device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)};
    D3D12_RENDER_TARGET_VIEW_DESC view{};
    view.ViewDimension = array ? D3D12_RTV_DIMENSION_TEXTURE2DARRAY : D3D12_RTV_DIMENSION_TEXTURE2D;
    if (array || mip_chain) {
        // Give every subresource a distinct sentinel before touching the selected view.
        view.Format = DXGI_FORMAT_R8G8B8A8_UINT;
        for (UINT layer = 0; layer < texture.DepthOrArraySize; ++layer) {
            for (UINT level = 0; level < texture.MipLevels; ++level) {
                if (array) view.Texture2DArray = {level, layer, 1, 0};
                else view.Texture2D = {level, 0};
                device->CreateRenderTargetView(image.p, &view, handle);
                const FLOAT seed[] = {FLOAT(11 + layer * 3 + level), 31, 73, 127};
                list->ClearRenderTargetView(handle, seed, 0, nullptr);
            }
        }
        if (array) view.Texture2DArray = {mip, first_layer, layer_count, 0};
        else view.Texture2D = {mip, 0};
    }
    view.Format = whole_format;
    device->CreateRenderTargetView(image.p, &view, handle);
    view.Format = rect_format;
    device->CreateRenderTargetView(image.p, &view, other);
    list->ClearRenderTargetView(handle, full, 0, nullptr);
    if (rewrite) {
        // ClearRenderTargetView must have captured its descriptor, including format, by value.
        view.Format = whole_format == DXGI_FORMAT_R8G8B8A8_UINT ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UINT;
        device->CreateRenderTargetView(image.p, &view, handle);
    }
    const D3D12_RECT rect = array || mip_chain ? D3D12_RECT{mip == 2 ? 0 : 1, 0,
        LONG(std::max(1u, width >> mip)), LONG(std::max(1u, height >> mip))} : D3D12_RECT{1, 1, 4, 3};
    if (partial) list->ClearRenderTargetView(other, part, 1, &rect);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = image.p;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = image.p;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = readback.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    for (UINT subresource = 0; subresource < count; ++subresource) {
        src.SubresourceIndex = subresource;
        dst.PlacedFootprint = placed[subresource];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    CHECK(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    Event event;
    CHECK(event.handle);
    CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, event.handle)));
    CHECK(WaitForSingleObject(event.handle, 5000) == WAIT_OBJECT_0);
    void* mapped = nullptr;
    const D3D12_RANGE range{0, SIZE_T(total)};
    CHECK(SUCCEEDED(readback->Map(0, &range, &mapped)) && mapped);
    pixels.clear();
    for (const auto& footprint : placed) {
        const auto& f = footprint.Footprint;
        for (UINT y = 0; y < f.Height; ++y) {
            const auto* row = static_cast<unsigned char*>(mapped) + footprint.Offset + y * f.RowPitch;
            pixels.insert(pixels.end(), row, row + f.Width * bytes);
        }
    }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}
bool rejects_view(ID3D12Device* device, DXGI_FORMAT resource_format, const D3D12_RENDER_TARGET_VIEW_DESC* view,
                  UINT16 mips = 1, UINT16 layers = 1) {
    Ref<ID3D12Resource> resource;
    D3D12_HEAP_PROPERTIES heap_properties{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = texture.Height = 4;
    texture.DepthOrArraySize = layers;
    texture.MipLevels = mips;
    texture.Format = resource_format;
    texture.SampleDesc.Count = 1;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &texture,
        D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&resource.p))));
    Ref<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap.p))));
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))));
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    // For the supported shape, overwrite a valid slot too: rejection must not leave its old RTV active.
    D3D12_RENDER_TARGET_VIEW_DESC seed{};
    seed.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    seed.Format = resource_format == DXGI_FORMAT_R8G8B8A8_TYPELESS ? DXGI_FORMAT_R8G8B8A8_UNORM : resource_format;
    device->CreateRenderTargetView(resource.p, &seed, handle);
    device->CreateRenderTargetView(resource.p, view, handle);
    const FLOAT color[] = {1, 1, 1, 1};
    list->ClearRenderTargetView(handle, color, 0, nullptr);
    CHECK(list->Close() == E_INVALIDARG); // The invalid list is never sent to the GPU.
    return true;
}
} // namespace

int main() {
    vktest::Device vk;
    if (!vk.create()) return 1;
    unsigned comparisons = 0, mismatches = 0, rejected = 0;
    {
        Ref<ID3D12Device> ours, host;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> warp;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
            FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p))) ||
            FAILED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
            std::fprintf(stderr, "WARP reference unavailable\n");
            return 1;
        }
        const DXGI_FORMAT formats[] = {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UINT};
        const auto compare = [&](DXGI_FORMAT full, DXGI_FORMAT rect, bool partial, bool uav) {
            std::vector<unsigned char> actual, expected;
            if (!clear_and_read(ours.p, full, rect, partial, true, uav, actual) ||
                !clear_and_read(host.p, full, rect, partial, true, uav, expected)) return false;
            ++comparisons;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::size_t x = (i / 4) % 5, y = i / 20;
                const DXGI_FORMAT format = partial && x >= 1 && x < 4 && y >= 1 ? rect : full;
                const int tolerance = format == DXGI_FORMAT_R8G8B8A8_UINT ? 0 : 1;
                if (std::abs(int(actual[i]) - int(expected[i])) > tolerance) {
                    std::fprintf(stderr, "full %d rect %d partial %d UAV %d byte %zu: Vulkan %u, WARP %u\n",
                        int(full), int(rect), int(partial), int(uav), i, unsigned(actual[i]), unsigned(expected[i]));
                    ++mismatches;
                    break;
                }
            }
            return true;
        };
        for (const auto format : formats) {
            if (!compare(format, format, false, false)) return 1;
            for (const auto rectangle : formats)
                if (!compare(format, rectangle, true, false)) return 1;
        }
        // An sRGB attachment view must not inherit an unrelated STORAGE usage from the typeless image.
        if (!compare(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UINT, true, true)) return 1;
        // Read every mip and layer, including untouched sentinels and pixels outside the rectangle.
        for (const auto format : formats) {
            for (UINT mip : {0u, 1u, 2u}) {
                for (bool partial : {false, true}) {
                    for (const auto selection : {std::pair{0u, 1u}, std::pair{1u, 2u}, std::pair{3u, 1u}}) {
                        std::vector<unsigned char> actual, expected;
                        if (!clear_and_read(ours.p, format, format, partial, true, false, actual,
                            mip, selection.first, selection.second, true) ||
                        !clear_and_read(host.p, format, format, partial, true, false, expected,
                            mip, selection.first, selection.second, true)) return 1;
                        ++comparisons;
                        if (actual.size() != expected.size()) return 1;
                        std::size_t offset = 0;
                        for (UINT layer = 0; layer < 4; ++layer) {
                            for (UINT level = 0; level < 3; ++level) {
                                const UINT w = std::max(1u, 5u >> level), h = std::max(1u, 3u >> level);
                                for (UINT byte = 0; byte < w * h * 4; ++byte) {
                                    const std::size_t i = offset + byte;
                                    const bool selected = level == mip && layer >= selection.first &&
                                    layer - selection.first < selection.second;
                                    const int tolerance = selected && format != DXGI_FORMAT_R8G8B8A8_UINT ? 1 : 0;
                                    if (std::abs(int(actual[i]) - int(expected[i])) > tolerance) {
                                        std::fprintf(stderr, "array format %d mip %u partial %d byte %zu: Vulkan %u WARP %u\n",
                                        int(format), mip, int(partial), i, unsigned(actual[i]), unsigned(expected[i]));
                                        return 1;
                                    }
                                }
                                offset += w * h * 4;
                            }
                        }
                    }
                }
            }
        }
        // Non-array TEXTURE2D views must select higher mips too; read back the entire chain.
        for (const auto format : formats) {
            for (UINT mip : {1u, 2u}) {
                for (bool partial : {false, true}) {
                    std::vector<unsigned char> actual, expected;
                    if (!clear_and_read(ours.p, format, format, partial, true, false, actual,
                                        mip, 0, 1, false, true) ||
                        !clear_and_read(host.p, format, format, partial, true, false, expected,
                                        mip, 0, 1, false, true)) return 1;
                    ++comparisons;
                    if (actual.size() != expected.size()) return 1;
                    std::size_t offset = 0;
                    for (UINT level = 0; level < 3; ++level) {
                        const UINT size = std::max(1u, 5u >> level) * std::max(1u, 3u >> level) * 4;
                        const int tolerance = level == mip && format != DXGI_FORMAT_R8G8B8A8_UINT ? 1 : 0;
                        for (UINT byte = 0; byte < size; ++byte) {
                            if (std::abs(int(actual[offset + byte]) - int(expected[offset + byte])) > tolerance) {
                                std::fprintf(stderr, "2D mip %u format %d partial %d level %u byte %u mismatch\n",
                                             mip, int(format), int(partial), level, byte);
                                return 1;
                            }
                        }
                        offset += size;
                    }
                }
            }
        }
        if (mismatches) return 1;
        const auto rejects = [&](DXGI_FORMAT resource_format, const D3D12_RENDER_TARGET_VIEW_DESC* view,
                                 UINT16 mips = 1, UINT16 layers = 1) {
            if (!rejects_view(ours.p, resource_format, view, mips, layers)) return false;
            ++rejected;
            return true;
        };
        D3D12_RENDER_TARGET_VIEW_DESC invalid{};
        invalid.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        for (const auto format : {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                 DXGI_FORMAT_R8G8B8A8_SINT, DXGI_FORMAT_R8G8B8A8_SNORM}) {
            invalid.Format = format;
            if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid)) return 1;
        }
        if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, nullptr)) return 1;
        invalid.Format = DXGI_FORMAT_R8G8B8A8_UINT;
        if (!rejects(DXGI_FORMAT_R8G8B8A8_UNORM, &invalid)) return 1;
        invalid.Texture2D.MipSlice = 1;
        if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid)) return 1;
        invalid.Texture2D.MipSlice = 0;
        invalid.Texture2D.PlaneSlice = 1;
        if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid)) return 1;
        invalid.Texture2D.PlaneSlice = 0;
        invalid.Texture2D.MipSlice = 2;
        if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid, 2, 1)) return 1;
        if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid, 1, 2)) return 1;
        invalid.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
        invalid.Texture2DArray = {0, 1, 1, 0};
        if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid)) return 1;
        for (const auto range : {std::pair{1u, UINT_MAX}, std::pair{0u, 0u}, std::pair{4u, 1u}, std::pair{3u, 2u}}) {
            invalid.Texture2DArray = {1, range.first, range.second, 0};
            if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid, 3, 4)) return 1;
        }
        invalid.Texture2DArray = {1, 1, 2, 1};
        if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid, 3, 4)) return 1;
    }
    if (vk.validation_errors()) return 1;
    std::printf("typeless RTV: %u WARP readbacks matched, descriptor snapshots and format-cache order checked; %u unsupported/invalid views rejected\n",
                comparisons, rejected);
    return 0;
}
