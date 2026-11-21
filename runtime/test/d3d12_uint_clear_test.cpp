
// Compare integer RTV clears against native Windows D3D12/WARP, through D3D12 clear/copy/Map on both sides.
// No game data or shaders. Run explicitly (needs Vulkan hardware); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <array>
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
        const std::array<float, 4> values[] = {
            {1, 2, 3, 4}, {1.5f, 2.5f, 3.5f, 4.5f}, {-1, -0.0f, 254.5f, 255.5f},
            {256, 65536, 16777216, 4294967296.0f}, {nan, inf, -inf, 0.5f},
            {0.499f, 0.501f, 1.499f, 1.501f}, {4294967040.0f, std::numeric_limits<float>::max(), 255, 0}
        };
        const std::pair<DXGI_FORMAT, unsigned> formats[] = {
            {DXGI_FORMAT_R8G8B8A8_UINT, 4}, {DXGI_FORMAT_R32_UINT, 4}, {DXGI_FORMAT_R32G32B32A32_UINT, 16}
        };
        for (const auto& [format, bytes] : formats) {
            for (unsigned i = 0; i < std::size(values); ++i) {
                // Different colours outside and inside the rectangle exercise both Vulkan clear paths.
                const auto& full = values[i];
                const auto& part = values[(i + 1) % std::size(values)];
                std::vector<unsigned char> actual, expected;
                if (!clear_and_read(ours.p, format, bytes, full.data(), part.data(), actual) ||
                    !clear_and_read(host.p, format, bytes, full.data(), part.data(), expected)) return 1;
                if (actual != expected) {
                    for (std::size_t b = 0; b < actual.size(); ++b) {
                        if (actual[b] != expected[b]) {
                            std::fprintf(stderr, "format %d case %u byte %zu: Vulkan %02x, WARP %02x\n",
                                int(format), i, b, unsigned(actual[b]), unsigned(expected[b]));
                            break;
                        }
                    }
                    return 1;
                }
                ++cases_checked;
            }
        }
    } // Destroy the D3D12 objects before checking the Vulkan validation counter.
    if (vk.validation_errors() != errors_before) return 1;
    std::printf("UINT RTV clears: %u full/rectangle readbacks matched WARP on three formats\n", cases_checked);
    return 0;
}
