
// Compare render-target and depth-stencil clears against native Windows D3D12/WARP, through D3D12 clear/copy/Map on
// both sides. Covers UNORM/SNORM/FLOAT/UINT/SINT render targets and depth/stencil, full and rectangle clears.
// No game data or shaders. Run explicitly (needs Vulkan hardware); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <array>
#include <vector>
#include <cstring>
#include <cstdint>
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

bool clear_and_read(ID3D12Device* device, DXGI_FORMAT format, unsigned bytes, const FLOAT* full, const FLOAT* part,
                    std::vector<unsigned char>& pixels) {
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

    constexpr UINT width = 5, height = 3; // Odd dimensions expose incorrect treatment of footprint padding.
    Ref<ID3D12Resource> image, readback;
    D3D12_HEAP_PROPERTIES heap_properties{};
    heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = width;
    texture.Height = height;
    texture.DepthOrArraySize = texture.MipLevels = 1;
    texture.Format = format;
    texture.SampleDesc.Count = 1;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &texture,
        D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&image.p))));
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&texture, 0, 1, 0, &placed, nullptr, nullptr, &total);
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
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap.p))));
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(image.p, nullptr, handle);
    list->ClearRenderTargetView(handle, full, 0, nullptr);
    const D3D12_RECT rect{1, 1, 4, 3};
    list->ClearRenderTargetView(handle, part, 1, &rect);
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
    dst.PlacedFootprint = placed;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
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
    pixels.resize(width * height * bytes);
    for (UINT y = 0; y < height; ++y)
        std::memcpy(pixels.data() + y * width * bytes,
                    static_cast<unsigned char*>(mapped) + y * placed.Footprint.RowPitch, width * bytes);
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}

// Clears a depth-stencil target fully and a rectangle, then reads the depth plane (and the stencil plane, when the
// format has one) back. depth_bytes is the depth plane's bytes per pixel (2 for D16, 4 for D32); has_stencil selects
// the separate stencil plane (subresource 1) of a combined format.
bool clear_depth_and_read(ID3D12Device* device, DXGI_FORMAT format, unsigned depth_bytes, bool has_stencil,
                          FLOAT depth_full, UINT8 stencil_full, FLOAT depth_part, UINT8 stencil_part,
                          std::vector<unsigned char>& depth_pixels, std::vector<unsigned char>& stencil_pixels) {
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

    constexpr UINT width = 5, height = 3;
    Ref<ID3D12Resource> image, readback;
    D3D12_HEAP_PROPERTIES heap_properties{};
    heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = width;
    texture.Height = height;
    texture.DepthOrArraySize = texture.MipLevels = 1;
    texture.Format = format;
    texture.SampleDesc.Count = 1;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &texture,
        D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr, IID_PPV_ARGS(&image.p))));
    const UINT subresources = has_stencil ? 2u : 1u;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed[2]{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&texture, 0, subresources, 0, placed, nullptr, nullptr, &total);
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
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap.p))));
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(image.p, nullptr, handle);
    const D3D12_CLEAR_FLAGS flags =
        static_cast<D3D12_CLEAR_FLAGS>((has_stencil ? D3D12_CLEAR_FLAG_STENCIL : 0) | D3D12_CLEAR_FLAG_DEPTH);
    list->ClearDepthStencilView(handle, flags, depth_full, stencil_full, 0, nullptr);
    const D3D12_RECT rect{1, 1, 4, 3};
    list->ClearDepthStencilView(handle, flags, depth_part, stencil_part, 1, &rect);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = image.p;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    for (UINT i = 0; i < subresources; ++i) {
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = image.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = i;
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = placed[i];
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
    const auto* base = static_cast<unsigned char*>(mapped);
    depth_pixels.resize(width * height * depth_bytes);
    for (UINT y = 0; y < height; ++y)
        std::memcpy(depth_pixels.data() + y * width * depth_bytes,
                    base + placed[0].Offset + y * placed[0].Footprint.RowPitch, width * depth_bytes);
    if (has_stencil) {
        stencil_pixels.resize(width * height);
        for (UINT y = 0; y < height; ++y)
            std::memcpy(stencil_pixels.data() + y * width,
                        base + placed[1].Offset + y * placed[1].Footprint.RowPitch, width);
    }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}
} // namespace

int main() {
    vktest::Device vk;
    if (!vk.create()) return 1;
    const unsigned errors_before = vk.validation_errors();
    unsigned cases_checked = 0;
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
            std::fprintf(stderr, "WARP reference unavailable: differential test cannot run\n");
            return 1;
        }
        const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
        // Integer-format values: negatives, fractions, NaN and overflow exercise the numeric float->int conversion.
        const std::array<float, 4> int_values[] = {
            {1, 2, 3, 4}, {1.5f, 2.5f, 3.5f, 4.5f}, {-1, -0.0f, 254.5f, 255.5f},
            {256, 65536, 16777216, 4294967296.0f}, {nan, inf, -inf, 0.5f},
            {0.499f, 0.501f, 1.499f, 1.501f}, {4294967040.0f, std::numeric_limits<float>::max(), 255, 0}
        };
        // Float-valued formats (UNORM/SNORM/FLOAT); out-of-range values exercise clamping. NaN/inf and SRGB are
        // omitted: NaN/inf conversion is driver-defined, and the linear->sRGB encoding rounds differently at the .5
        // boundary on WARP vs Vulkan (a 1-ulp driver difference the shim cannot control).
        const std::array<float, 4> float_values[] = {
            {0, 0.25f, 0.5f, 1}, {1, 1, 1, 1}, {0.5f, 0.5f, 0.5f, 0.5f}, {-0.5f, 0, 1.5f, 2}
        };
        struct ColorFormat { DXGI_FORMAT format; unsigned bytes; const std::array<float, 4>* values; unsigned count; };
        const ColorFormat color_formats[] = {
            {DXGI_FORMAT_R8G8B8A8_UNORM, 4, float_values, std::size(float_values)},
            {DXGI_FORMAT_R8G8B8A8_SNORM, 4, float_values, std::size(float_values)},
            {DXGI_FORMAT_R32G32B32A32_FLOAT, 16, float_values, std::size(float_values)},
            {DXGI_FORMAT_R8G8B8A8_UINT, 4, int_values, std::size(int_values)},
            {DXGI_FORMAT_R32_UINT, 4, int_values, std::size(int_values)},
            {DXGI_FORMAT_R32G32B32A32_UINT, 16, int_values, std::size(int_values)},
            {DXGI_FORMAT_R8G8B8A8_SINT, 4, int_values, std::size(int_values)},
            {DXGI_FORMAT_R32_SINT, 4, int_values, std::size(int_values)},
        };
        for (const auto& cf : color_formats) {
            for (unsigned i = 0; i < cf.count; ++i) {
                // Different colours outside and inside the rectangle exercise both Vulkan clear paths.
                const auto& full = cf.values[i];
                const auto& part = cf.values[(i + 1) % cf.count];
                std::vector<unsigned char> actual, expected;
                if (!clear_and_read(ours.p, cf.format, cf.bytes, full.data(), part.data(), actual) ||
                    !clear_and_read(host.p, cf.format, cf.bytes, full.data(), part.data(), expected)) return 1;
                if (actual != expected) {
                    for (std::size_t b = 0; b < actual.size(); ++b) {
                        if (actual[b] != expected[b]) {
                            std::fprintf(stderr, "format %d case %u byte %zu: Vulkan %02x, WARP %02x\n",
                                int(cf.format), i, b, unsigned(actual[b]), unsigned(expected[b]));
                            break;
                        }
                    }
                    return 1;
                }
                ++cases_checked;
            }
        }
        // Depth (and stencil) clears, full and rectangle. Depth is [0,1]; stencil is a byte.
        struct DepthFormat { DXGI_FORMAT format; unsigned depth_bytes; bool has_stencil; };
        const DepthFormat depth_formats[] = {
            {DXGI_FORMAT_D32_FLOAT, 4, false},
            {DXGI_FORMAT_D16_UNORM, 2, false},
            {DXGI_FORMAT_D32_FLOAT_S8X24_UINT, 4, true},
        };
        const float depths[] = {0, 0.25f, 0.5f, 1};
        for (const auto& df : depth_formats) {
            for (unsigned i = 0; i < std::size(depths); ++i) {
                const float full = depths[i], part = depths[(i + 1) % std::size(depths)];
                const std::uint8_t stencil_full = std::uint8_t(i * 0x40 + 0x11);
                const std::uint8_t stencil_part = std::uint8_t(i * 0x40 + 0x22);
                std::vector<unsigned char> actual, expected, actual_stencil, expected_stencil;
                if (!clear_depth_and_read(ours.p, df.format, df.depth_bytes, df.has_stencil, full, stencil_full, part,
                                          stencil_part, actual, actual_stencil) ||
                    !clear_depth_and_read(host.p, df.format, df.depth_bytes, df.has_stencil, full, stencil_full, part,
                                          stencil_part, expected, expected_stencil))
                    return 1;
                if (actual != expected || actual_stencil != expected_stencil) {
                    std::fprintf(stderr, "depth format %d case %u: depth/stencil readback mismatch\n", int(df.format), i);
                    return 1;
                }
                ++cases_checked;
            }
        }
    } // Destroy the D3D12 objects before checking the Vulkan validation counter.
    if (vk.validation_errors() != errors_before) return 1;
    std::printf("RTV and DSV clears: %u full/rectangle readbacks matched WARP\n", cases_checked);
    return 0;
}
